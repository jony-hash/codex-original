"""H-LOMA-I0 full-epoch Identity reconstruction continuation.

The continuation is deliberately separate from the bounded pilot lifecycle.
It consumes exactly the pilot final state and the remaining MNIST examples;
it never changes the formal v2 runtime or the pilot evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ...config import FormalSystemConfig
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
)
from . import optical_manifold as base
from . import optical_manifold_training as pilot


BRANCH = "research/v2-lightgen-optical-manifold-full-epoch-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_identity_full_epoch_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_identity_full_epoch_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_identity_full_epoch_validation.v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_identity_full_epoch_preflight.v1"
TRAINING_SCHEMA = "lightgen_optical_manifold_identity_full_epoch_training.v1"
RESUME_SCHEMA = "lightgen_optical_manifold_identity_full_epoch_resume.v1"
VISUAL_SCHEMA = "lightgen_optical_manifold_identity_full_epoch_visual_panel.v1"
SOURCE_SCHEMA = "lightgen_optical_manifold_identity_pilot_v1"
SOURCE_RECORD_SCHEMA = "lightgen_optical_manifold_identity_training.v1"
SOURCE_MANIFEST_HASH = "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
SOURCE_TRAINING_HASH = "4127f80194f02750737c2e2a1e3f8f7ee4b90df8798cbdd53c4c20369b4a681f"
SOURCE_CHECKPOINT_HASH = "bb072c7b2193b24eb0d989ba731957b52b382c3ff1c83a2be2ddfd7d6d6a2d20"
PILOT_CONFIG_HASH = "0c7b86e9037d4993a9948b7298b103444cd957841433c2e28141341948c7a495"
PILOT_CONFIG_CANONICAL_HASH = "32f676715c46115ab2904f2b00ad11915ce3fc87630467d2967eb393b4993c41"
PILOT_REFERENCE_PAIRED_MSE = 0.06062695011496544
PILOT_SCHEDULE_HASH = "4a5c8357c5695ffe360b53f8f887c7a2058cf8c43a6a4b4d6be17103f244df67"
CONTINUATION_SCHEDULE_HASH = "d15608c4f7f8a5f7709b021155fa252270844bdfab67fc796743014e9897bacf"
NEXT_PASS_ACTION = "manual_review_optical_latent_ols_contract"
MAX_GLOBAL_STEP = 5500
PILOT_GLOBAL_STEP = 1000
CONTINUATION_STEPS = 4500
LOGICAL_BATCH_SIZE = 10
MICROBATCH_SIZE = 1
VISUAL_PER_LABEL = 10
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_PREFLIGHT_SECONDS = 300.0
MAX_WALL_SECONDS = 4500.0
MAX_STORAGE_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class FullEpochSpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str
    pilot_config_path: Path


@dataclass(frozen=True)
class FullEpochData:
    pilot_data: pilot.PilotData
    continuation_schedule: torch.Tensor
    full_schedule: torch.Tensor
    visual_indices: torch.Tensor
    manifest: dict[str, Any]


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"]:
        raise ValueError("full-epoch continuation requires a clean worktree")
    if state["branch"] != BRANCH:
        raise ValueError(f"full-epoch continuation requires branch {BRANCH}")


def _expected_config_payload() -> dict[str, Any]:
    return {
        "schema": CONFIG_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "pilot_config": "configs/research/lightgen_optical_manifold_identity_pilot_v1.yaml",
        "pilot_config_sha256": PILOT_CONFIG_HASH,
        "source_training_evidence": "outputs/20260827_v2/runs/lightgen_optical_manifold_identity_pilot_v1_attempt_03/training.json",
        "source_training_evidence_sha256": SOURCE_TRAINING_HASH,
        "source_checkpoint": "outputs/20260827_v2/runs/lightgen_optical_manifold_identity_pilot_v1_attempt_03/pilot_final_step_001000.pt",
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "source_dataset_manifest_hash": SOURCE_MANIFEST_HASH,
        "source_pilot_paired_mse": PILOT_REFERENCE_PAIRED_MSE,
        "dataset": {
            "train_count": 55000,
            "validation_count": 5000,
            "pilot_consumed_samples": 10000,
            "continuation_samples": 45000,
            "total_epoch_samples": 55000,
            "pilot_steps": 1000,
            "continuation_steps": 4500,
            "total_global_steps": 5500,
            "logical_batch_size": 10,
            "schedule": "pilot_prefix_then_remaining_shuffle",
            "continuation_schedule_seed": 1234,
            "continuation_schedule_sha256": CONTINUATION_SCHEDULE_HASH,
            "no_replacement": True,
            "heldout_per_label": 100,
            "visual_panel_per_label": 10,
            "visual_panel_selection": "first_occurrence_in_heldout_panel",
            "input_shape": [1, 28, 28],
            "rendered_shape": [1, 200, 200],
        },
        "training": {
            "optimizer": "Adam",
            "optimizer_update_device": "cpu_shadow",
            "learning_rate": 1.0e-3,
            "initial_global_step": 1000,
            "continuation_steps": 4500,
            "final_global_step": 5500,
            "logical_batch_size": 10,
            "microbatch_size": 1,
            "activation_checkpoint": True,
            "deterministic": True,
            "checkpoint_interval": 100,
            "resume_required_source_step": 1000,
        },
        "gates": {
            "pilot_reference_paired_mse": PILOT_REFERENCE_PAIRED_MSE,
            "final_paired_mse_strictly_below_pilot": True,
            "final_paired_mse_below_permuted_target": True,
            "negative_control_max_relative_improvement": 0.05,
            "max_preflight_wall_seconds": 300,
            "max_wall_seconds": 4500,
            "min_free_cuda_bytes": 4294967296,
            "max_peak_cuda_bytes": 5368709120,
            "max_storage_bytes": 2147483648,
        },
        "visual_panel": {
            "count": 100,
            "per_label": 10,
            "artifacts": [
                "input_intensity",
                "target",
                "raw_cmos",
                "reconstruction",
                "target_png",
                "reconstruction_png",
            ],
            "affects_training_or_selection": False,
        },
        "authorization": {
            "reviewer_role": "project_owner",
            "decision": "approve_bounded_full_epoch_research_training",
            "full_epoch_research_training_authorized": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal_promotion": False,
            "post_pass_next_action": NEXT_PASS_ACTION,
            "prohibited": [
                "extra_epoch",
                "latent_ols_experiment",
                "smf",
                "fmf",
                "evaluator",
                "generation",
                "promotion",
            ],
        },
    }


def _load_spec(path: str | Path) -> FullEpochSpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload != _expected_config_payload():
        raise ValueError("full-epoch config does not exactly match the registered contract")
    pilot_path = _resolve(payload["pilot_config"], base_path=target.parent)
    if file_sha256(pilot_path) != PILOT_CONFIG_HASH:
        raise ValueError("pilot config hash changed")
    pilot._load_pilot_spec(pilot_path)
    return FullEpochSpec(target, payload, canonical_sha256(payload), pilot_path)


def _load_review(path: str | Path, spec: FullEpochSpec) -> tuple[dict[str, Any], Path]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_bounded_full_epoch_research_training",
        "source_training_evidence": spec.payload["source_training_evidence"],
        "source_training_evidence_sha256": SOURCE_TRAINING_HASH,
        "source_checkpoint": spec.payload["source_checkpoint"],
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "approved_contract": {
            "initial_global_step": 1000,
            "continuation_steps": 4500,
            "final_global_step": 5500,
            "optimizer_update_device": "cpu_shadow",
            "pilot_reference_paired_mse": PILOT_REFERENCE_PAIRED_MSE,
            "post_pass_next_action": NEXT_PASS_ACTION,
        },
        "full_epoch_research_training_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("full-epoch review contract does not match the registered scope")
    return payload, target


def _load_formal(spec: FullEpochSpec) -> tuple[FormalSystemConfig, Path]:
    pilot_spec = pilot._load_pilot_spec(spec.pilot_config_path)
    base_spec = base._load_spec(pilot_spec.base_path)
    return base._load_formal_config(base_spec)


def _assert_ancestor(commit: str, current: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=base._repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("pilot evidence commit is not an ancestor of this branch")


def _load_source(
    spec: FullEpochSpec,
    review_path: Path,
    pilot_training_path: str | Path,
    pilot_checkpoint_path: str | Path,
    current_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    training_path = _resolve(pilot_training_path)
    checkpoint_path = _resolve(pilot_checkpoint_path)
    expected_training = _resolve(spec.payload["source_training_evidence"])
    expected_checkpoint = _resolve(spec.payload["source_checkpoint"])
    if training_path != expected_training or checkpoint_path != expected_checkpoint:
        raise ValueError("full epoch must consume the registered pilot evidence paths")
    if file_sha256(training_path) != SOURCE_TRAINING_HASH:
        raise ValueError("pilot training evidence hash mismatch")
    if file_sha256(checkpoint_path) != SOURCE_CHECKPOINT_HASH:
        raise ValueError("pilot checkpoint hash mismatch")
    training_payload = json.loads(training_path.read_text(encoding="utf-8"))
    if (
        training_payload.get("schema") != SOURCE_RECORD_SCHEMA
        or training_payload.get("hypothesis") != HYPOTHESIS
        or training_payload.get("bounded_research_training_authorized") is not True
        or training_payload.get("training_authorized") is not False
        or training_payload.get("formal_promotion") is not False
    ):
        raise ValueError("pilot training evidence has an invalid schema or authorization")
    record = EvidenceRecordV2.from_dict(training_payload["evidence"])
    record.validate()
    if record.evidence_kind != SOURCE_RECORD_SCHEMA or record.status != "passed":
        raise ValueError("pilot training evidence must be passed")
    _assert_ancestor(record.git_commit, current_commit)
    if record.config_hash != PILOT_CONFIG_CANONICAL_HASH:
        raise ValueError("pilot training evidence config mismatch")
    heldout = training_payload.get("contract", {}).get("heldout", {})
    if heldout.get("paired_mse") != PILOT_REFERENCE_PAIRED_MSE:
        raise ValueError("pilot paired MSE reference changed")
    if training_payload["contract"].get("final_checkpoint") != str(checkpoint_path.resolve()):
        raise ValueError("pilot training evidence checkpoint binding mismatch")
    checkpoint = safe_torch_load(checkpoint_path, expected_sha256=SOURCE_CHECKPOINT_HASH, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("schema") != pilot.RESUME_SCHEMA:
        raise ValueError("pilot checkpoint schema mismatch")
    if (
        checkpoint.get("step") != PILOT_GLOBAL_STEP
        or len(checkpoint.get("loss_history", [])) != PILOT_GLOBAL_STEP
        or checkpoint.get("config_hash") != PILOT_CONFIG_CANONICAL_HASH
        or checkpoint.get("dataset_manifest_hash") != SOURCE_MANIFEST_HASH
        or checkpoint.get("schedule_sha256") != PILOT_SCHEDULE_HASH
        or checkpoint.get("evidence_hashes") != {
            "training_validation_evidence": training_payload["contract"]["training_validation_hash"],
            "training_preflight_evidence": training_payload["contract"]["training_preflight_hash"],
        }
    ):
        raise ValueError("pilot checkpoint state binding mismatch")
    cursor = checkpoint.get("data_cursor")
    if cursor != {
        "schedule": "balanced_one_per_label_without_replacement",
        "logical_batch_size": LOGICAL_BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "next_step": PILOT_GLOBAL_STEP + 1,
    }:
        raise ValueError("pilot checkpoint cursor mismatch")
    if not isinstance(checkpoint.get("model_state"), dict) or not isinstance(checkpoint.get("optimizer_state"), dict):
        raise ValueError("pilot checkpoint is missing model or Adam state")
    if len(checkpoint["optimizer_state"].get("param_groups", [])) != 1:
        raise ValueError("pilot checkpoint Adam group mismatch")
    rng = checkpoint.get("rng_state")
    if not isinstance(rng, dict) or not torch.is_tensor(rng.get("cpu")) or not isinstance(rng.get("cuda"), list):
        raise ValueError("pilot checkpoint RNG state is incomplete")
    return training_payload, checkpoint, training_path, checkpoint_path


def _validate_checkpoint_state(
    formal: FormalSystemConfig, checkpoint: Mapping[str, Any]
) -> None:
    model = base.IdentityOpticalManifoldAutoencoder(formal)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    groups = checkpoint["optimizer_state"].get("param_groups", [])
    if len(groups) != 1 or groups[0].get("lr") != 1.0e-3:
        raise ValueError("pilot checkpoint Adam hyperparameters mismatch")
    if len(groups[0].get("params", [])) != len(base.EXPECTED_TRAINABLE):
        raise ValueError("pilot checkpoint Adam parameter count mismatch")
    if len(checkpoint["optimizer_state"].get("state", {})) != len(base.EXPECTED_TRAINABLE):
        raise ValueError("pilot checkpoint Adam state is incomplete")


def _continuation_schedule(data: pilot.PilotData) -> torch.Tensor:
    labels = data.labels.index_select(0, data.train_indices)
    remaining: list[torch.Tensor] = []
    for label in range(10):
        class_indices = data.train_indices[labels == label]
        if class_indices.numel() <= PILOT_GLOBAL_STEP:
            raise ValueError("pilot prefix exhausts a training class")
        remaining.append(class_indices[PILOT_GLOBAL_STEP:])
    values = torch.cat(remaining)
    if values.numel() != 45000:
        raise ValueError("continuation sample count is not 45000")
    generator = torch.Generator(device="cpu").manual_seed(1234)
    schedule = values[torch.randperm(values.numel(), generator=generator)].reshape(
        CONTINUATION_STEPS, LOGICAL_BATCH_SIZE
    )
    if base.tensor_sha256(schedule) != CONTINUATION_SCHEDULE_HASH:
        raise ValueError("continuation schedule hash mismatch")
    return schedule


def _full_manifest(data: pilot.PilotData, continuation: torch.Tensor, full_schedule: torch.Tensor) -> dict[str, Any]:
    union = torch.cat((data.schedule, continuation))
    sorted_union = torch.sort(union.reshape(-1)).values
    sorted_train = torch.sort(data.train_indices).values
    if not torch.equal(sorted_union, sorted_train):
        raise ValueError("pilot and continuation schedules do not cover exactly the train split")
    manifest = {
        "schema": "lightgen_optical_manifold_identity_full_epoch_mnist_manifest.v1",
        "source_pilot_manifest_hash": SOURCE_MANIFEST_HASH,
        "train_count": int(data.train_indices.numel()),
        "validation_count": int(data.validation_indices.numel()),
        "pilot_consumed_samples": int(data.schedule.numel()),
        "continuation_samples": int(continuation.numel()),
        "total_epoch_samples": int(full_schedule.numel()),
        "pilot_schedule_sha256": base.tensor_sha256(data.schedule),
        "continuation_schedule_sha256": base.tensor_sha256(continuation),
        "full_schedule_sha256": base.tensor_sha256(full_schedule),
        "train_union_sha256": base.tensor_sha256(sorted_union),
        "no_replacement": True,
        "schedule": "pilot_prefix_then_remaining_shuffle",
        "continuation_schedule_seed": 1234,
        "heldout_indices_sha256": base.tensor_sha256(data.heldout_indices),
        "heldout_count": int(data.heldout_indices.numel()),
        "visual_indices_sha256": base.tensor_sha256(data.heldout_indices.reshape(10, 100)[:, :VISUAL_PER_LABEL].reshape(-1)),
        "visual_count": 100,
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return manifest


def _load_data(dataset_root: str | Path) -> FullEpochData:
    pilot_data = pilot._load_pilot_data(dataset_root)
    if pilot_data.manifest["manifest_hash"] != SOURCE_MANIFEST_HASH:
        raise ValueError("pilot dataset manifest changed")
    continuation = _continuation_schedule(pilot_data)
    full_schedule = torch.cat((pilot_data.schedule, continuation))
    visual_indices = pilot_data.heldout_indices.reshape(10, 100)[:, :VISUAL_PER_LABEL].reshape(-1)
    visual_labels = pilot_data.labels.index_select(0, visual_indices)
    if not torch.equal(visual_labels, torch.arange(10).repeat_interleave(VISUAL_PER_LABEL)):
        raise ValueError("visual panel selection is not ordered by label")
    manifest = _full_manifest(pilot_data, continuation, full_schedule)
    return FullEpochData(pilot_data, continuation, full_schedule, visual_indices, manifest)


def _context(spec: FullEpochSpec, formal: FormalSystemConfig, manifest: Mapping[str, Any]) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=str(manifest["manifest_hash"]),
        calibration_hash=canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        exposure_scale=1.0,
    )


def _record_payload(schema: str, record: EvidenceRecordV2, contract: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": schema,
        "evidence": record.to_dict(),
        "hypothesis": HYPOTHESIS,
        "full_epoch_research_training_authorized": True,
        "training_authorized": False,
        "formal_promotion": False,
        "contract": dict(contract),
    }
    if extra:
        payload.update(extra)
    return payload


def _write_new_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {target}")
    atomic_write_json(target, payload)


def _validation_contract(
    spec: FullEpochSpec,
    review_path: Path,
    training_path: Path,
    checkpoint_path: Path,
    data: FullEpochData,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    model = base.IdentityOpticalManifoldAutoencoder(_load_formal(spec)[0])
    parameter_contract = base._parameter_contract(model)
    if tuple(parameter_contract["trainable_parameter_names"]) != base.EXPECTED_TRAINABLE:
        raise ValueError("full-epoch trainable parameter contract failed")
    if data.continuation_schedule.shape != (CONTINUATION_STEPS, LOGICAL_BATCH_SIZE):
        raise ValueError("continuation schedule shape mismatch")
    if data.full_schedule.shape != (MAX_GLOBAL_STEP, LOGICAL_BATCH_SIZE):
        raise ValueError("full schedule shape mismatch")
    if torch.unique(data.full_schedule.reshape(-1)).numel() != 55000:
        raise ValueError("full schedule contains repeated samples")
    labels = data.pilot_data.labels.index_select(0, data.full_schedule.reshape(-1)).reshape(MAX_GLOBAL_STEP, LOGICAL_BATCH_SIZE)
    if not torch.equal(labels[:PILOT_GLOBAL_STEP], torch.arange(10).expand(PILOT_GLOBAL_STEP, -1)):
        raise ValueError("pilot prefix is not balanced one-per-label")
    if not torch.equal(torch.sort(data.full_schedule.reshape(-1)).values, torch.sort(data.pilot_data.train_indices).values):
        raise ValueError("full schedule does not cover the fixed 55k train split")
    return {
        "branch": state["branch"],
        "commit": state["commit"],
        "formal_system_kind": base.EXPECTED_FORMAL_KIND,
        "system_geometry": dict(base.EXPECTED_SYSTEM),
        "input_encoding": "intensity_sqrt_amplitude",
        "target": "camera_intensity_same_fov",
        "loss": "mse",
        "exposure": {"scale": 1.0, "gamma": 1.0},
        "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "logical_batch_size": 10, "microbatch_size": 1},
        "activation_checkpoint": True,
        "source_pilot": {
            "training_evidence": str(training_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "training_evidence_sha256": SOURCE_TRAINING_HASH,
            "checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
            "paired_mse": PILOT_REFERENCE_PAIRED_MSE,
            "step": PILOT_GLOBAL_STEP,
        },
        "trainable_parameter_names": list(base.EXPECTED_TRAINABLE),
        "data_manifest": data.manifest,
        "schedule": {
            "pilot_steps": PILOT_GLOBAL_STEP,
            "continuation_steps": CONTINUATION_STEPS,
            "total_global_steps": MAX_GLOBAL_STEP,
            "continuation_schedule_sha256": CONTINUATION_SCHEDULE_HASH,
            "without_replacement": True,
        },
        "heldout_panel": {"count": 1000, "per_label": 100},
        "visual_panel": {"count": 100, "per_label": 10, "review_only": True},
        "review_contract_sha256": file_sha256(review_path),
        "authorization": {
            "full_epoch_research_training_authorized": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "formal_promotion": False,
            "post_pass_next_action": NEXT_PASS_ACTION,
        },
    }


def _validation_record(state: Mapping[str, Any], spec: FullEpochSpec, formal: FormalSystemConfig, data: FullEpochData, input_hashes: Mapping[str, str]) -> EvidenceRecordV2:
    gates = (
        GateResult("clean_lineage", True, {"clean": True}),
        GateResult("independent_full_epoch_branch", True, {"branch_matches": True}),
        GateResult("pilot_source_evidence_and_checkpoint", True, {"step": 1000}),
        GateResult("formal_geometry_and_identity_ols", True, {"grid": 2048, "identity": True}),
        GateResult("fixed_55k_epoch_union", True, {"samples": 55000, "steps": 5500}),
        GateResult("continuation_schedule", True, {"samples": 45000, "no_replacement": True}),
        GateResult("visual_panel_provenance", True, {"count": 100, "per_label": 10}),
        GateResult("authorization_scope", True, {"continuation_steps": 4500}),
        GateResult("trainable_parameter_set", True, {"count": len(base.EXPECTED_TRAINABLE)}),
    )
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=VALIDATION_SCHEMA,
        tier="diagnostic",
        status="passed",
        git_commit=str(state["commit"]),
        dirty_patch_hash=None,
        config_hash=spec.config_hash,
        context=_context(spec, formal, data.manifest),
        input_hashes=dict(input_hashes),
        gates=gates,
        permitted_next_actions=("run_full_epoch_identity_reconstruction_preflight",),
    )
    record.validate()
    return record


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    training_payload, checkpoint, training_path, checkpoint_path = _load_source(
        spec, review_path, args.pilot_training_evidence, args.pilot_checkpoint, state["commit"]
    )
    _validate_checkpoint_state(formal, checkpoint)
    data = _load_data(args.dataset_root)
    _validation_contract(spec, review_path, training_path, checkpoint_path, data, state)
    del training_payload, checkpoint
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full-epoch validation: {output}")
    input_hashes = {
        "full_epoch_config": file_sha256(spec.path),
        "pilot_config": file_sha256(spec.pilot_config_path),
        "formal_config": file_sha256(formal_path),
        "review_contract": file_sha256(review_path),
        "pilot_training_evidence": SOURCE_TRAINING_HASH,
        "pilot_checkpoint": SOURCE_CHECKPOINT_HASH,
        "pilot_dataset_manifest": SOURCE_MANIFEST_HASH,
        "full_epoch_manifest": data.manifest["manifest_hash"],
    }
    record = _validation_record(state, spec, formal, data, input_hashes)
    payload = _record_payload(
        VALIDATION_SCHEMA,
        record,
        _validation_contract(spec, review_path, training_path, checkpoint_path, data, state),
    )
    _write_new_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(path: str | Path, spec: FullEpochSpec, data: FullEpochData, review_path: Path, state: Mapping[str, Any]) -> tuple[EvidenceRecordV2, str]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != VALIDATION_SCHEMA:
        raise ValueError("full-epoch preflight requires full-epoch validation evidence")
    if payload.get("hypothesis") != HYPOTHESIS or payload.get("full_epoch_research_training_authorized") is not True or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("full-epoch validation authorization scope mismatch")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != VALIDATION_SCHEMA or record.status != "passed":
        raise ValueError("full-epoch validation evidence must be passed")
    if record.git_commit != state["commit"] or state["branch"] != BRANCH or not state["clean"]:
        raise ValueError("full-epoch validation evidence is stale")
    if record.config_hash != spec.config_hash or record.context.dataset_hash != data.manifest["manifest_hash"]:
        raise ValueError("full-epoch validation context mismatch")
    formal_path = _load_formal(spec)[1]
    expected = {
        "full_epoch_config": file_sha256(spec.path),
        "pilot_config": file_sha256(spec.pilot_config_path),
        "formal_config": file_sha256(formal_path),
        "review_contract": file_sha256(review_path),
        "pilot_training_evidence": SOURCE_TRAINING_HASH,
        "pilot_checkpoint": SOURCE_CHECKPOINT_HASH,
        "pilot_dataset_manifest": SOURCE_MANIFEST_HASH,
        "full_epoch_manifest": data.manifest["manifest_hash"],
    }
    for key, value in expected.items():
        if record.input_hashes.get(key) != value:
            raise ValueError(f"full-epoch validation binding mismatch for {key}")
    return record, file_sha256(target)


def _batch(data: FullEpochData, indices: torch.Tensor, device: torch.device, *, target_roll: int = 0) -> base.RenderedBatch:
    rendered = base.render_intensity_batch(data.pilot_data.images.index_select(0, indices))
    target = rendered.target
    if target_roll:
        target = target.roll(shifts=target_roll, dims=0)
    return base.RenderedBatch(rendered.intensity.to(device), rendered.amplitude.to(device), target.to(device))


def _clone_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    return copy.deepcopy(value)


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return isinstance(left, type(right)) and len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _state_cpu(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _phase_names(model: base.IdentityOpticalManifoldAutoencoder) -> tuple[str, ...]:
    parameters = dict(model.named_parameters())
    return tuple(f"{name}[{index}]" for name in base.EXPECTED_TRAINABLE for index in range(parameters[name].shape[0]))


class _DeterministicCpuAdam:
    """Keep formal CUDA optics while making Adam continuation bitwise resumable."""

    def __init__(self, model: base.IdentityOpticalManifoldAutoencoder) -> None:
        named = dict(model.named_parameters())
        if tuple(name for name in named if named[name].requires_grad) != base.EXPECTED_TRAINABLE:
            raise ValueError("full-epoch CPU Adam trainable parameter contract failed")
        self._gpu_parameters = tuple(named[name] for name in base.EXPECTED_TRAINABLE)
        self._cpu_parameters = tuple(
            torch.nn.Parameter(parameter.detach().cpu().clone())
            for parameter in self._gpu_parameters
        )
        self._optimizer = torch.optim.Adam(self._cpu_parameters, lr=1.0e-3)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        return self._optimizer.param_groups

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter in self._gpu_parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> Any:
        for gpu_parameter, cpu_parameter in zip(self._gpu_parameters, self._cpu_parameters, strict=True):
            cpu_parameter.data.copy_(gpu_parameter.detach().cpu())
            cpu_parameter.grad = None if gpu_parameter.grad is None else gpu_parameter.grad.detach().cpu().clone()
        result = self._optimizer.step(closure)
        for gpu_parameter, cpu_parameter in zip(self._gpu_parameters, self._cpu_parameters, strict=True):
            gpu_parameter.data.copy_(cpu_parameter.data)
        return result

    def state_dict(self) -> dict[str, Any]:
        return self._optimizer.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self._optimizer.load_state_dict(state_dict)


def _make_full_optimizer(model: base.IdentityOpticalManifoldAutoencoder) -> _DeterministicCpuAdam:
    return _DeterministicCpuAdam(model)


def _full_step(model: base.IdentityOpticalManifoldAutoencoder, optimizer: _DeterministicCpuAdam, batch: base.RenderedBatch) -> dict[str, Any]:
    before = {name: dict(model.named_parameters())[name].detach().clone() for name in base.EXPECTED_TRAINABLE}
    result = base._step(model, optimizer, batch, activation_checkpoint=True)
    parameters = dict(model.named_parameters())
    gradients = dict(result["gradient_norms"])
    updates: dict[str, float] = {}
    for name in base.EXPECTED_TRAINABLE:
        parameter = parameters[name]
        for index in range(parameter.shape[0]):
            gradients[f"{name}[{index}]"] = float(parameter.grad[index].detach().norm().cpu()) if parameter.grad is not None else 0.0
            updates[f"{name}[{index}]"] = float((parameter[index].detach() - before[name][index]).abs().sum().cpu())
        updates[name] = float((parameter.detach() - before[name]).abs().sum().cpu())
    result["gradient_norms"] = gradients
    result["update_norms"] = updates
    return result


def _finite(result: Mapping[str, Any]) -> bool:
    values = [result["loss"], result["camera_min"], result["camera_max"], *result["gradient_norms"].values(), *result["update_norms"].values()]
    return result["camera_finite"] and all(bool(torch.isfinite(torch.tensor(value))) for value in values)


def _frozen(model: base.IdentityOpticalManifoldAutoencoder, initial: Mapping[str, torch.Tensor]) -> bool:
    return all(torch.equal(model.state_dict()[name].detach().cpu(), initial[name]) for name in model.state_dict() if name not in base.EXPECTED_TRAINABLE)


def _runtime_from_checkpoint(formal: FormalSystemConfig, checkpoint: Mapping[str, Any], device: torch.device) -> tuple[base.IdentityOpticalManifoldAutoencoder, _DeterministicCpuAdam]:
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    optimizer = _make_full_optimizer(model)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    base._restore_rng(checkpoint["rng_state"])
    return model, optimizer


def _resume_payload(model: base.IdentityOpticalManifoldAutoencoder, optimizer: _DeterministicCpuAdam, *, global_step: int, loss_history: list[float], data: FullEpochData, spec: FullEpochSpec, evidence_hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema": RESUME_SCHEMA,
        "model_state": _state_cpu(model),
        "optimizer_state": _clone_tree(optimizer.state_dict()),
        "rng_state": _clone_tree(base._rng_state()),
        "global_step": global_step,
        "continuation_cursor": global_step - PILOT_GLOBAL_STEP,
        "loss_history": list(loss_history),
        "config_hash": spec.config_hash,
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "continuation_schedule_sha256": CONTINUATION_SCHEDULE_HASH,
        "evidence_hashes": dict(evidence_hashes),
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_HASH,
        "data_cursor": {"initial_global_step": PILOT_GLOBAL_STEP, "next_global_step": global_step + 1, "continuation_cursor": global_step - PILOT_GLOBAL_STEP},
    }


def _load_resume(path: str | Path, model: base.IdentityOpticalManifoldAutoencoder, optimizer: _DeterministicCpuAdam, *, data: FullEpochData, spec: FullEpochSpec, evidence_hashes: Mapping[str, str]) -> dict[str, Any]:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != RESUME_SCHEMA:
        raise ValueError("invalid full-epoch resume checkpoint")
    step = payload.get("global_step")
    if not isinstance(step, int) or not 1000 < step < MAX_GLOBAL_STEP:
        raise ValueError("full-epoch resume checkpoint step must be in [1001, 5499]")
    if payload.get("config_hash") != spec.config_hash or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"] or payload.get("continuation_schedule_sha256") != CONTINUATION_SCHEDULE_HASH or payload.get("evidence_hashes") != dict(evidence_hashes) or payload.get("source_checkpoint_sha256") != SOURCE_CHECKPOINT_HASH:
        raise ValueError("full-epoch resume checkpoint lineage mismatch")
    if payload.get("continuation_cursor") != step - PILOT_GLOBAL_STEP or len(payload.get("loss_history", [])) != step:
        raise ValueError("full-epoch resume cursor or history mismatch")
    if payload.get("data_cursor") != {"initial_global_step": PILOT_GLOBAL_STEP, "next_global_step": step + 1, "continuation_cursor": step - PILOT_GLOBAL_STEP}:
        raise ValueError("full-epoch resume data cursor mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    base._restore_rng(payload["rng_state"])
    return payload


def _preflight_record(state: Mapping[str, Any], spec: FullEpochSpec, formal: FormalSystemConfig, data: FullEpochData, input_hashes: Mapping[str, str], gates: tuple[GateResult, ...], status: str, actions: tuple[str, ...], reason: str | None = None) -> EvidenceRecordV2:
    record = EvidenceRecordV2(schema_version=2, evidence_kind=PREFLIGHT_SCHEMA, tier="diagnostic", status=status, git_commit=str(state["commit"]), dirty_patch_hash=None, config_hash=spec.config_hash, context=_context(spec, formal, data.manifest), input_hashes=dict(input_hashes), gates=gates, permitted_next_actions=actions, invalidation_reason=reason)
    record.validate()
    return record


def _source_hash_inputs(spec: FullEpochSpec, review_path: Path, formal_path: Path, validation_hash: str | None = None, preflight_hash: str | None = None) -> dict[str, str]:
    values = {"full_epoch_config": file_sha256(spec.path), "pilot_config": file_sha256(spec.pilot_config_path), "formal_config": file_sha256(formal_path), "review_contract": file_sha256(review_path), "pilot_training_evidence": SOURCE_TRAINING_HASH, "pilot_checkpoint": SOURCE_CHECKPOINT_HASH, "pilot_dataset_manifest": SOURCE_MANIFEST_HASH}
    if validation_hash is not None:
        values["full_epoch_validation_evidence"] = validation_hash
    if preflight_hash is not None:
        values["full_epoch_preflight_evidence"] = preflight_hash
    return values


def train_preflight(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("full-epoch preflight requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    spec = _load_spec(args.config)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    _, _, _, checkpoint_path = _load_source(spec, review_path, args.pilot_training_evidence, args.pilot_checkpoint, state["commit"])
    data = _load_data(args.dataset_root)
    _, validation_hash = _load_validation(args.epoch_validation_evidence, spec, data, review_path, state)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full-epoch preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    input_hashes = _source_hash_inputs(spec, review_path, formal_path, validation_hash=validation_hash)
    if not torch.cuda.is_available():
        gates = (GateResult("cuda_available", False, {}, "CUDA is unavailable"),)
        record = _preflight_record(state, spec, formal, data, input_hashes, gates, "invalid_harness", (), "CUDA is unavailable")
        payload = _record_payload(PREFLIGHT_SCHEMA, record, {"epoch_validation_hash": validation_hash})
        _write_new_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        reason = f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}"
        record = _preflight_record(state, spec, formal, data, input_hashes, (GateResult("free_cuda_memory", False, {"bytes": int(free_bytes)}, reason),), "invalid_harness", (), reason)
        payload = _record_payload(PREFLIGHT_SCHEMA, record, {"epoch_validation_hash": validation_hash, "free_cuda_bytes": int(free_bytes), "total_cuda_bytes": int(total_bytes)})
        _write_new_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    pilot_checkpoint = safe_torch_load(checkpoint_path, expected_sha256=SOURCE_CHECKPOINT_HASH, map_location="cpu")
    model, optimizer = _runtime_from_checkpoint(formal, pilot_checkpoint, device)
    initial_state = _state_cpu(model)
    paired_steps = []
    for index in range(3):
        paired_steps.append(_full_step(model, optimizer, _batch(data, data.continuation_schedule[index], device)))
    paired_state = _state_cpu(model)
    paired_optimizer = _clone_tree(optimizer.state_dict())
    paired_rng = base._rng_state()
    interrupted, interrupted_optimizer = _runtime_from_checkpoint(formal, pilot_checkpoint, device)
    interrupted_history = list(pilot_checkpoint["loss_history"])
    for index in range(2):
        interrupted_history.append(_full_step(interrupted, interrupted_optimizer, _batch(data, data.continuation_schedule[index], device))["loss"])
    resume_path = output.with_name(output.stem + ".resume.pt")
    resume_hashes = {
        "pilot_training_evidence": SOURCE_TRAINING_HASH,
        "pilot_checkpoint": SOURCE_CHECKPOINT_HASH,
        "full_epoch_validation_evidence": validation_hash,
    }
    atomic_torch_save(resume_path, _resume_payload(interrupted, interrupted_optimizer, global_step=1002, loss_history=interrupted_history, data=data, spec=spec, evidence_hashes=resume_hashes))
    del interrupted, interrupted_optimizer
    torch.cuda.empty_cache()
    resumed, resumed_optimizer = _runtime_from_checkpoint(formal, pilot_checkpoint, device)
    _load_resume(resume_path, resumed, resumed_optimizer, data=data, spec=spec, evidence_hashes=resume_hashes)
    resumed_step = _full_step(resumed, resumed_optimizer, _batch(data, data.continuation_schedule[2], device))
    resume_exact = _tree_equal(paired_state, _state_cpu(resumed)) and _tree_equal(paired_optimizer, resumed_optimizer.state_dict()) and _tree_equal(paired_rng, base._rng_state()) and paired_steps[2] == resumed_step
    control, control_optimizer = _runtime_from_checkpoint(formal, pilot_checkpoint, device)
    control_steps = [_full_step(control, control_optimizer, _batch(data, data.continuation_schedule[index], device, target_roll=1)) for index in range(3)]
    preflight_indices = data.pilot_data.heldout_indices.reshape(10, 100)[:, :VISUAL_PER_LABEL].reshape(-1)
    control_baseline = pilot._evaluate(_runtime_from_checkpoint(formal, pilot_checkpoint, device)[0], data.pilot_data, preflight_indices, device)
    control_mse = pilot._evaluate(control, data.pilot_data, preflight_indices, device)
    control_relative = (control_baseline - control_mse) / control_baseline
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    storage = base._directory_size(output.parent)
    phase_names = _phase_names(model)
    all_steps = paired_steps + control_steps + [resumed_step]
    finite = all(_finite(item) for item in all_steps)
    gradients = all(item["gradient_norms"].get(name, 0.0) > 0.0 for item in paired_steps for name in phase_names)
    updates = all(item["update_norms"].get(name, 0.0) > 0.0 for item in paired_steps for name in phase_names)
    frozen = _frozen(model, initial_state)
    negative_control = control_relative < 0.05
    gates = (
        GateResult("finite_values", finite, {"step_count": len(all_steps)}),
        GateResult("four_phase_layer_gradients_nonzero", gradients, {"count": 4}),
        GateResult("four_phase_layer_updates_nonzero", updates, {"count": 4}),
        GateResult("frozen_state_unchanged", frozen, {}),
        GateResult("fixed_formal_relay_and_identity_ols", True, {"grid": 2048, "identity": True}),
        GateResult("fixed_exposure", True, {"scale": 1.0, "gamma": 1.0}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("resume_exact", resume_exact, {}),
        GateResult("permuted_target_negative_control", negative_control, {"baseline_mse": control_baseline, "control_mse": control_mse, "relative_improvement": control_relative, "threshold": 0.05}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
        GateResult("wall_time", elapsed <= MAX_PREFLIGHT_SECONDS, {"seconds": elapsed}),
        GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
    )
    passed = all(gate.passed for gate in gates)
    record = _preflight_record(state, spec, formal, data, {**input_hashes, "resume_artifact": file_sha256(resume_path)}, gates, "passed" if passed else "invalid_harness", ("run_full_epoch_identity_reconstruction",) if passed else (), None if passed else "one or more full-epoch preflight gates failed")
    payload = _record_payload(PREFLIGHT_SCHEMA, record, {"epoch_validation_hash": validation_hash, "source_checkpoint": str(checkpoint_path.resolve()), "resume_artifact": str(resume_path.resolve()), "phase_layer_names": list(phase_names), "paired_steps": paired_steps, "control_steps": control_steps, "resumed_step": resumed_step, "negative_control": {"count": int(preflight_indices.numel()), "baseline_mse": control_baseline, "control_mse": control_mse, "relative_improvement": control_relative}, "free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "peak_cuda_bytes": peak, "wall_seconds": elapsed, "run_storage_bytes": storage})
    _write_new_json(output, payload)
    torch.use_deterministic_algorithms(previous)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 2


def _load_preflight(path: str | Path, spec: FullEpochSpec, data: FullEpochData, validation_hash: str, review_path: Path, state: Mapping[str, Any]) -> tuple[EvidenceRecordV2, str]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("full-epoch train requires full-epoch preflight evidence")
    if payload.get("hypothesis") != HYPOTHESIS or payload.get("full_epoch_research_training_authorized") is not True or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("full-epoch preflight authorization scope mismatch")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != PREFLIGHT_SCHEMA or record.status != "passed":
        raise ValueError("full-epoch preflight evidence must be passed")
    if record.git_commit != state["commit"] or state["branch"] != BRANCH or not state["clean"]:
        raise ValueError("full-epoch preflight evidence is stale")
    if record.config_hash != spec.config_hash or record.context.dataset_hash != data.manifest["manifest_hash"]:
        raise ValueError("full-epoch preflight context mismatch")
    formal_path = _load_formal(spec)[1]
    expected = {"full_epoch_config": file_sha256(spec.path), "pilot_config": file_sha256(spec.pilot_config_path), "formal_config": file_sha256(formal_path), "review_contract": file_sha256(review_path), "pilot_training_evidence": SOURCE_TRAINING_HASH, "pilot_checkpoint": SOURCE_CHECKPOINT_HASH, "pilot_dataset_manifest": SOURCE_MANIFEST_HASH, "full_epoch_manifest": data.manifest["manifest_hash"], "full_epoch_validation_evidence": validation_hash}
    for key, value in expected.items():
        if value is not None and record.input_hashes.get(key) != value:
            raise ValueError(f"full-epoch preflight binding mismatch for {key}")
    return record, file_sha256(target)


def _save_checkpoint(path: Path, model: base.IdentityOpticalManifoldAutoencoder, optimizer: _DeterministicCpuAdam, *, global_step: int, history: list[float], data: FullEpochData, spec: FullEpochSpec, evidence_hashes: Mapping[str, str]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite full-epoch checkpoint: {path}")
    atomic_torch_save(path, _resume_payload(model, optimizer, global_step=global_step, loss_history=history, data=data, spec=spec, evidence_hashes=evidence_hashes))


def _visual_panel(model: base.IdentityOpticalManifoldAutoencoder, data: FullEpochData, device: torch.device, run_dir: Path) -> dict[str, Any]:
    model.eval()
    inputs: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    raw: list[torch.Tensor] = []
    reconstructions: list[torch.Tensor] = []
    with torch.no_grad():
        for index in data.visual_indices:
            source = data.pilot_data.images[index].reshape(1, 1, 28, 28)
            rendered = base.render_intensity_batch(source)
            output = model(rendered.amplitude.to(device), activation_checkpoint=False)
            inputs.append(source.squeeze(0).cpu())
            targets.append(rendered.target.squeeze(0).cpu())
            raw.append(output.raw_camera.squeeze(0).cpu())
            reconstructions.append(output.reconstruction.squeeze(0).cpu())
    model.train()
    input_tensor = torch.stack(inputs)
    target_tensor = torch.stack(targets)
    raw_tensor = torch.stack(raw)
    reconstruction_tensor = torch.stack(reconstructions)
    if not all(torch.isfinite(value).all() for value in (input_tensor, target_tensor, raw_tensor, reconstruction_tensor)):
        raise ValueError("visual panel contains non-finite values")
    panel_path = run_dir / "full_epoch_visual_panel.pt"
    target_png = run_dir / "full_epoch_target_panel.png"
    reconstruction_png = run_dir / "full_epoch_reconstruction_panel.png"
    if panel_path.exists() or target_png.exists() or reconstruction_png.exists():
        raise FileExistsError("refusing to overwrite visual panel artifacts")
    atomic_torch_save(panel_path, {"schema": VISUAL_SCHEMA, "indices": data.visual_indices.cpu(), "labels": data.pilot_data.labels.index_select(0, data.visual_indices), "input_intensity": input_tensor, "target": target_tensor, "raw_cmos": raw_tensor, "reconstruction": reconstruction_tensor, "selection": "first_occurrence_in_heldout_panel", "affects_training_or_selection": False})
    from torchvision.utils import save_image

    save_image(target_tensor, target_png, nrow=VISUAL_PER_LABEL, value_range=(0.0, 1.0))
    save_image(reconstruction_tensor, reconstruction_png, nrow=VISUAL_PER_LABEL, value_range=(0.0, 1.0))
    return {"schema": VISUAL_SCHEMA, "path": str(panel_path.resolve()), "sha256": file_sha256(panel_path), "target_png": str(target_png.resolve()), "target_png_sha256": file_sha256(target_png), "reconstruction_png": str(reconstruction_png.resolve()), "reconstruction_png_sha256": file_sha256(reconstruction_png), "count": 100, "per_label": 10, "tensor_shapes": {"input_intensity": list(input_tensor.shape), "target": list(target_tensor.shape), "raw_cmos": list(raw_tensor.shape), "reconstruction": list(reconstruction_tensor.shape)}, "affects_training_or_selection": False}


def _training_record(state: Mapping[str, Any], spec: FullEpochSpec, formal: FormalSystemConfig, data: FullEpochData, input_hashes: Mapping[str, str], gates: tuple[GateResult, ...], status: str, actions: tuple[str, ...], reason: str | None = None) -> EvidenceRecordV2:
    record = EvidenceRecordV2(schema_version=2, evidence_kind=TRAINING_SCHEMA, tier="diagnostic", status=status, git_commit=str(state["commit"]), dirty_patch_hash=None, config_hash=spec.config_hash, context=_context(spec, formal, data.manifest), input_hashes=dict(input_hashes), gates=gates, permitted_next_actions=actions, invalidation_reason=reason)
    record.validate()
    return record


def train(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("full-epoch training requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    spec = _load_spec(args.config)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    source_training, source_checkpoint, _, source_checkpoint_path = _load_source(spec, review_path, args.pilot_training_evidence, args.pilot_checkpoint, state["commit"])
    data = _load_data(args.dataset_root)
    _, validation_hash = _load_validation(args.epoch_validation_evidence, spec, data, review_path, state)
    _, preflight_hash = _load_preflight(args.epoch_preflight_evidence, spec, data, validation_hash, review_path, state)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite full-epoch training evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA became unavailable after full-epoch preflight")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    pilot_checkpoint = source_checkpoint
    model, optimizer = _runtime_from_checkpoint(formal, pilot_checkpoint, device)
    initial_state = _state_cpu(model)
    baseline_paired = pilot._evaluate(model, data.pilot_data, data.pilot_data.heldout_indices, device)
    baseline_permuted = pilot._evaluate(model, data.pilot_data, data.pilot_data.heldout_indices, device, target_roll=1)
    source_heldout = source_training["contract"]["heldout"]
    baseline_match = abs(baseline_paired - source_heldout["paired_mse"]) <= 1.0e-9 and abs(baseline_permuted - source_heldout["permuted_target_mse"]) <= 1.0e-9
    history = list(pilot_checkpoint["loss_history"])
    start_global_step = PILOT_GLOBAL_STEP
    evidence_hashes = {
        "pilot_training_evidence": SOURCE_TRAINING_HASH,
        "pilot_checkpoint": SOURCE_CHECKPOINT_HASH,
        "full_epoch_validation_evidence": validation_hash,
        "full_epoch_preflight_evidence": preflight_hash,
    }
    if args.resume_checkpoint:
        resume = _load_resume(args.resume_checkpoint, model, optimizer, data=data, spec=spec, evidence_hashes=evidence_hashes)
        history = list(resume["loss_history"])
        start_global_step = int(resume["global_step"])
    step_results: list[dict[str, Any]] = []
    resource_aborted = not baseline_match or time.monotonic() - started > MAX_WALL_SECONDS
    if not baseline_match:
        raise ValueError("pilot checkpoint held-out metrics do not reproduce source evidence")
    for global_step in range(start_global_step + 1, MAX_GLOBAL_STEP + 1):
        if time.monotonic() - started > MAX_WALL_SECONDS:
            resource_aborted = True
            break
        result = _full_step(model, optimizer, _batch(data, data.continuation_schedule[global_step - PILOT_GLOBAL_STEP - 1], device))
        history.append(result["loss"])
        step_results.append(result)
        if global_step % 100 == 0:
            _save_checkpoint(output.parent / f"full_epoch_step_{global_step:06d}.pt", model, optimizer, global_step=global_step, history=history, data=data, spec=spec, evidence_hashes=evidence_hashes)
    completed = not resource_aborted and len(history) == MAX_GLOBAL_STEP
    final_checkpoint_path: Path | None = None
    if completed:
        final_checkpoint_path = output.parent / "full_epoch_final_step_005500.pt"
        _save_checkpoint(final_checkpoint_path, model, optimizer, global_step=MAX_GLOBAL_STEP, history=history, data=data, spec=spec, evidence_hashes=evidence_hashes)
    final_paired = pilot._evaluate(model, data.pilot_data, data.pilot_data.heldout_indices, device) if completed else None
    final_permuted = pilot._evaluate(model, data.pilot_data, data.pilot_data.heldout_indices, device, target_roll=1) if completed else None
    visual = _visual_panel(model, data, device, output.parent) if completed else None
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    storage = base._directory_size(output.parent)
    phase_names = _phase_names(model)
    finite = all(_finite(item) for item in step_results)
    gradients = all(item["gradient_norms"].get(name, 0.0) > 0.0 for item in step_results for name in phase_names)
    updates = all(item["update_norms"].get(name, 0.0) > 0.0 for item in step_results for name in phase_names)
    frozen = _frozen(model, initial_state)
    if resource_aborted or elapsed > MAX_WALL_SECONDS:
        status = "resource_aborted"
        actions: tuple[str, ...] = ()
        reason = "full-epoch continuation exceeded its resource stop"
    elif not completed or not finite or not gradients or not updates or not frozen:
        status = "invalid_harness"
        actions = ()
        reason = "full-epoch execution or isolation gate failed"
    else:
        science = final_paired < PILOT_REFERENCE_PAIRED_MSE and final_paired < final_permuted
        status = "passed" if science else "valid_failed"
        actions = (NEXT_PASS_ACTION,) if status == "passed" else ()
        reason = None if status == "passed" else "pre-registered full-epoch scientific gate did not pass"
    checkpoint_files = sorted(output.parent.glob("full_epoch_step_*.pt"))
    gates = (
        GateResult("baseline_matches_pilot", baseline_match, {"pilot_paired_mse": source_heldout["paired_mse"], "measured_paired_mse": baseline_paired, "pilot_permuted_mse": source_heldout["permuted_target_mse"], "measured_permuted_mse": baseline_permuted}),
        GateResult("completed_4500_continuation_steps", completed, {"global_step": len(history)}),
        GateResult("finite_values", finite, {"measured_steps": len(step_results)}),
        GateResult("four_phase_layer_gradients_nonzero", gradients, {"count": 4}),
        GateResult("four_phase_layer_updates_nonzero", updates, {"count": 4}),
        GateResult("frozen_state_unchanged", frozen, {}),
        GateResult("fixed_formal_relay_and_identity_ols", True, {"grid": 2048, "identity": True}),
        GateResult("fixed_exposure", True, {"scale": 1.0, "gamma": 1.0}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("final_paired_below_pilot", final_paired is not None and final_paired < PILOT_REFERENCE_PAIRED_MSE, {"pilot_reference": PILOT_REFERENCE_PAIRED_MSE, "final_paired_mse": final_paired if final_paired is not None else float("nan")}),
        GateResult("final_paired_below_permuted_target", final_paired is not None and final_permuted is not None and final_paired < final_permuted, {"final_paired_mse": final_paired if final_paired is not None else float("nan"), "final_permuted_mse": final_permuted if final_permuted is not None else float("nan")}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
        GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
        GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
    )
    if status == "passed" and not all(gate.passed for gate in gates):
        status = "invalid_harness"
        actions = ()
        reason = "a harness or resource gate failed after the scientific measurement"
    input_hashes = _source_hash_inputs(spec, review_path, formal_path, validation_hash=validation_hash, preflight_hash=preflight_hash)
    if final_checkpoint_path is not None:
        input_hashes["final_checkpoint"] = file_sha256(final_checkpoint_path)
    if visual is not None:
        input_hashes["visual_panel"] = visual["sha256"]
        input_hashes["target_panel_png"] = visual["target_png_sha256"]
        input_hashes["reconstruction_panel_png"] = visual["reconstruction_png_sha256"]
    record = _training_record(state, spec, formal, data, input_hashes, gates, status, actions, reason)
    checkpoint_hashes = {path.name: file_sha256(path) for path in checkpoint_files}
    payload = _record_payload(TRAINING_SCHEMA, record, {"source_training_evidence_hash": SOURCE_TRAINING_HASH, "source_checkpoint_hash": SOURCE_CHECKPOINT_HASH, "epoch_validation_hash": validation_hash, "epoch_preflight_hash": preflight_hash, "source_checkpoint": str(source_checkpoint_path.resolve()), "steps_completed": len(history), "global_step_start": start_global_step, "global_step_final": len(history), "loss_history": history, "heldout": {"count": 1000, "pilot_reference_paired_mse": PILOT_REFERENCE_PAIRED_MSE, "baseline_paired_mse": baseline_paired, "baseline_permuted_target_mse": baseline_permuted, "final_paired_mse": final_paired, "final_permuted_target_mse": final_permuted}, "phase_layer_names": list(phase_names), "checkpoint_interval": 100, "checkpoints": checkpoint_hashes, "final_checkpoint": str(final_checkpoint_path.resolve()) if final_checkpoint_path else None, "visual_panel": visual, "free_cuda_bytes_before": int(free_bytes), "total_cuda_bytes": int(total_bytes), "peak_cuda_bytes": peak, "wall_seconds": elapsed, "run_storage_bytes": storage, "evaluator_calls": 0})
    _write_new_json(output, payload)
    torch.use_deterministic_algorithms(previous)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "passed" else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "full-epoch-validate":
        return validate(args)
    if args.research_action == "full-epoch-preflight":
        return train_preflight(args)
    if args.research_action == "full-epoch-train":
        return train(args)
    raise ValueError(f"unsupported full-epoch action: {args.research_action}")
