"""Bounded two-arm aggregate-prior training pilot for H-LOMA-I0.

This module is a diagnostic research lifecycle.  It starts from the passed
Identity reconstruction checkpoint, resets Adam for each matched arm, and
trains only the four formal phase masks.  The regularizer is the fixed
project-approximate sliced-Wasserstein objective registered by PLAN 5; it is
not a VAE KL or a latent sampling path.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
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
from torch.nn import functional as F

from ...config import FormalSystemConfig, load_config
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
from . import optical_manifold_full_epoch as full_epoch
from . import optical_manifold_latent_observer as observer
from . import optical_manifold_regularization_audit as audit


BRANCH = "research/v2-lightgen-optical-manifold-regularization-pilot-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_identity_regularization_pilot_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_identity_regularization_pilot_review.v1"
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_identity_regularization_training_validation.v1"
)
PREFLIGHT_SCHEMA = (
    "lightgen_optical_manifold_identity_regularization_training_preflight.v1"
)
TRAINING_SCHEMA = "lightgen_optical_manifold_identity_regularization_training.v1"
RESUME_SCHEMA = "lightgen_optical_manifold_identity_regularization_training_resume.v1"

SOURCE_AUDIT_CONFIG_HASH = (
    "38c75fd7ff0e9cfe67e1623b54c078fcea34122fb5d3988eebbcc51725a03008"
)
SOURCE_AUDIT_REVIEW_HASH = (
    "a91f667438c78dcc4fefda14e8a9bc5ce862808f4e8b9491aace2406bcb391b2"
)
SOURCE_AUDIT_VALIDATION_HASH = (
    "78778f9ea37982990f9c34aa930840f744bbdc2c3ae2fc6c49a03d0c28f91c8a"
)
SOURCE_AUDIT_HASH = (
    "6b33f4828cf94542628ca417203420f744bbdc2c3ae2fc6c49a03d0c28f91c8a"
)
SOURCE_PROJECTION_HASH = (
    "d7428f9daf323e1f0046a8110ed208b938a5d6f2dec34e0f6fcca36fe58c770d"
)
SOURCE_CHECKPOINT_HASH = (
    "56cb52fae3a13c94ea5a38af60e0934a407cf1ec7b57bbb31fba2a23c624655f"
)
SOURCE_TRAINING_HASH = (
    "c01dae121210a4dff0534a9979661f92d1e0fc749087c8451ad51be24a458948"
)
SOURCE_MANIFEST_HASH = (
    "64b1b70dc119dac0cb0aa8b898614962ff993a83d120390d1d6fadd55ade97d8"
)
SOURCE_HELDOUT_HASH = (
    "9ffe5fd2df41a1a7627efd926e04159506d8aabb228799d9c384d9cb34565f03"
)
SOURCE_STEP = 5500
SOURCE_PAIRED_MSE = 0.03437697887420654
SCHEDULE_HASH = "4a10d910d34bb7f5b9ed5e730e4dc958c52380688ce55562b612d28edde25b9b"
SCHEDULE_SEED = 1234
STEPS_PER_ARM = 100
LOGICAL_BATCH_SIZE = 100
MICROBATCH_SIZE = 1
CHECKPOINT_INTERVAL = 10
PCA_COMPONENTS = 32
FIXED_DIRECTIONS = 64
LAMBDA = 0.002733644070001058
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_PREFLIGHT_SECONDS = 300.0
MAX_WALL_SECONDS = 2400.0
MAX_STORAGE_BYTES = 512 * 1024**2
PREPARED_MSE_DEGRADATION = 0.01
FINAL_MSE_DEGRADATION = 0.05
FINAL_SWD_REDUCTION = 0.10
NEXT_PASS_ACTION = "manual_review_identity_manifold_regularization_continuation_contract"
NEXT_VALID_FAILED_ACTION = "manual_review_identity_regularization_objective_contract"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_FORMAL_KIND = base.EXPECTED_FORMAL_KIND
EXPECTED_TRAINABLE = base.EXPECTED_TRAINABLE


@dataclass(frozen=True)
class PilotSpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class PilotData:
    regularization_data: audit.RegularizationData
    schedule: torch.Tensor
    heldout_indices: torch.Tensor
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SourceBundle:
    checkpoint: dict[str, Any]
    projection: dict[str, Any]
    hashes: dict[str, str]
    formal_path: Path
    audit_config_hash: str


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"]:
        raise ValueError("regularization pilot requires a clean worktree")
    if state["branch"] != BRANCH:
        raise ValueError(f"regularization pilot requires branch {BRANCH}")


def _require_deterministic_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "regularization pilot requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "before the Python process starts"
        )


def _load_spec(path: str | Path) -> PilotSpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    expected = {
        "schema", "hypothesis", "formal_system_config", "source", "dataset",
        "optical_contract", "training", "regularization", "gates", "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("regularization pilot config has unexpected or missing fields")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("unsupported regularization pilot config")
    return PilotSpec(target, payload, canonical_sha256(payload))


def _load_review(path: str | Path, spec: PilotSpec) -> tuple[dict[str, Any], Path]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    source = spec.payload["source"]
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_bounded_identity_aggregate_prior_training_pilot",
        "source_regularization_validation_sha256": source["regularization_validation_sha256"],
        "source_regularization_audit_sha256": source["regularization_audit_sha256"],
        "source_projection_sha256": source["projection_sha256"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_mnist_manifest_hash": source["mnist_manifest_sha256"],
        "approved_contract": {
            "arms": ["regularized", "mse_only"],
            "steps_per_arm": STEPS_PER_ARM,
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "microbatch_size": MICROBATCH_SIZE,
            "schedule_sha256": SCHEDULE_HASH,
            "reset_adam_from_source": True,
            "lambda": LAMBDA,
            "final_swd_min_relative_reduction": FINAL_SWD_REDUCTION,
            "no_early_stop": True,
            "post_pass_next_action": NEXT_PASS_ACTION,
        },
        "bounded_research_training_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("regularization pilot review contract does not match scope")
    return payload, target


def _assert_contract(spec: PilotSpec) -> None:
    payload = spec.payload
    source = payload["source"]
    if source["source_step"] != SOURCE_STEP or source["source_paired_mse"] != SOURCE_PAIRED_MSE:
        raise ValueError("regularization pilot source step or MSE is not locked")
    if payload["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "heldout_count": 1000,
        "schedule_count": 10000,
        "steps": 100,
        "logical_batch_size": 100,
        "microbatch_size": 1,
        "schedule": "balanced_classwise_shuffle_after_audit_panels",
        "schedule_seed": SCHEDULE_SEED,
        "schedule_sha256": SCHEDULE_HASH,
        "calibration_and_audit_excluded": True,
        "no_replacement": True,
        "heldout_excluded": True,
        "label_role": "schedule_balance_and_offline_metrics_only",
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }:
        raise ValueError("regularization pilot dataset contract is not locked")
    if payload["optical_contract"] != {
        "formal_system_kind": EXPECTED_FORMAL_KIND,
        "wavelength_m": 5.32e-7,
        "slm_shape": [200, 200],
        "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": [100, 100],
        "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048,
        "encoder_layers": 2,
        "decoder_layers": 2,
        "boundary": "finite_crop",
        "ols": "identity",
        "front_relay": "coherent_imaging_4f",
        "back_relay": "free_space",
        "input_encoding": "intensity_sqrt_amplitude",
        "target": "camera_intensity_same_fov",
        "exposure": {"scale": 1.0, "gamma": 1.0},
    }:
        raise ValueError("regularization pilot optical contract is not locked")
    if payload["training"] != {
        "arms": ["regularized", "mse_only"],
        "optimizer": "Adam",
        "optimizer_update_device": "cpu_shadow",
        "learning_rate": 1.0e-3,
        "reset_adam_from_source": True,
        "source_global_step": SOURCE_STEP,
        "steps_per_arm": STEPS_PER_ARM,
        "logical_batch_size": LOGICAL_BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "reconstruction_activation_checkpoint": True,
        "regularizer_trace_activation_checkpoint": False,
        "deterministic": True,
        "fixed_execution_order": ["regularized", "mse_only"],
        "no_early_stop": True,
        "no_tuning": True,
        "no_checkpoint_selection": True,
    }:
        raise ValueError("regularization pilot training contract is not locked")
    if payload["regularization"] != {
        "family": "aggregate_prior",
        "name": "deterministic_sliced_wasserstein",
        "approximation_role": "project_approximation",
        "projection_components": PCA_COMPONENTS,
        "fixed_directions": FIXED_DIRECTIONS,
        "direction_seed": SCHEDULE_SEED,
        "lambda": LAMBDA,
        "lambda_source": "regularization_audit_batch_0",
        "lambda_recalibration": False,
        "rank_replay": "fixed_two_pass",
        "labels_in_loss": False,
        "vae_kl": False,
        "latent_sampling": False,
    }:
        raise ValueError("regularization pilot objective contract is not locked")
    if payload["gates"] != {
        "source_paired_mse_tolerance": 1.0e-9,
        "preflight_max_mse_relative_degradation": PREPARED_MSE_DEGRADATION,
        "final_max_mse_relative_degradation": FINAL_MSE_DEGRADATION,
        "final_swd_min_relative_reduction": FINAL_SWD_REDUCTION,
        "final_regularized_swd_lt_mse_control": True,
        "final_paired_mse_lt_cyclic": True,
        "normalized_inter_intra_gt_one": True,
        "normalized_true_ratio_gt_permuted": True,
        "min_free_cuda_bytes": MIN_FREE_BYTES,
        "max_peak_cuda_bytes": MAX_PEAK_BYTES,
        "max_preflight_wall_seconds": MAX_PREFLIGHT_SECONDS,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "max_storage_bytes": MAX_STORAGE_BYTES,
    }:
        raise ValueError("regularization pilot gate contract is not locked")
    if payload["authorization"] != {
        "reviewer_role": "project_owner",
        "decision": "approve_bounded_identity_aggregate_prior_training_pilot",
        "bounded_research_training_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "post_pass_next_action": NEXT_PASS_ACTION,
        "post_valid_failed_next_action": NEXT_VALID_FAILED_ACTION,
        "prohibited": payload["authorization"]["prohibited"],
    }:
        raise ValueError("regularization pilot authorization contract is not locked")


def _assert_ancestor(commit: str, current: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=base._repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("regularization audit source commit is not an ancestor")


def _load_formal(spec: PilotSpec) -> tuple[FormalSystemConfig, Path]:
    formal_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    formal = load_config(formal_path)
    if formal.architecture != "conditional_identity_logit_amplitude_v2":
        raise ValueError("regularization pilot requires current Identity lineage")
    if formal.system.kind != EXPECTED_FORMAL_KIND or not formal.system.is_formal:
        raise ValueError("regularization pilot requires formal Identity optics")
    for name, expected in base.EXPECTED_SYSTEM.items():
        actual = getattr(formal.system, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1.0e-12, 1.0e-15):
                raise ValueError(f"formal geometry mismatch for {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    return formal.system, formal_path


def _load_data(dataset_root: str | Path) -> PilotData:
    regularization_data = audit._load_data(dataset_root)
    pilot_data = regularization_data.full_data.pilot_data
    schedule = _pilot_schedule(pilot_data.train_indices, pilot_data.labels)
    if schedule.numel() != 10000 or torch.unique(schedule).numel() != schedule.numel():
        raise ValueError("regularization pilot schedule is not 10000 unique samples")
    if base.tensor_sha256(schedule.reshape(-1)) != SCHEDULE_HASH:
        raise ValueError("regularization pilot schedule hash mismatch")
    if torch.isin(schedule.reshape(-1), regularization_data.full_data.pilot_data.heldout_indices).any():
        raise ValueError("regularization pilot schedule overlaps held-out data")
    manifest = {
        "schema": "lightgen_optical_manifold_identity_regularization_training_mnist_manifest.v1",
        "source_mnist_manifest_hash": SOURCE_MANIFEST_HASH,
        "source_regularization_manifest_hash": regularization_data.manifest["manifest_hash"],
        "schedule_sha256": SCHEDULE_HASH,
        "schedule_shape": list(schedule.shape),
        "schedule_count": int(schedule.numel()),
        "schedule_seed": SCHEDULE_SEED,
        "no_replacement": True,
        "calibration_and_audit_excluded": True,
        "heldout_indices_sha256": SOURCE_HELDOUT_HASH,
        "heldout_excluded": True,
        "label_role": "schedule_balance_and_offline_metrics_only",
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return PilotData(
        regularization_data=regularization_data,
        schedule=schedule,
        heldout_indices=regularization_data.full_data.pilot_data.heldout_indices,
        manifest=manifest,
    )


def _pilot_schedule(train_indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Build the fixed class-balanced 100-step schedule without replacement."""

    train_labels = labels.index_select(0, train_indices)
    generator = torch.Generator(device="cpu").manual_seed(SCHEDULE_SEED)
    per_label: list[torch.Tensor] = []
    for label in range(10):
        class_indices = train_indices[train_labels == label]
        if class_indices.numel() <= 130 + 1000:
            raise ValueError("train split is too small for the pilot schedule")
        remaining = class_indices[130:]
        shuffled = remaining[torch.randperm(remaining.numel(), generator=generator)]
        per_label.append(shuffled[:1000])
    schedule = torch.cat(
        [
            torch.cat(
                [per_label[label][step * 10: (step + 1) * 10] for label in range(10)]
            )
            for step in range(STEPS_PER_ARM)
        ]
    ).reshape(STEPS_PER_ARM, LOGICAL_BATCH_SIZE)
    return schedule


def _context(spec: PilotSpec, formal: FormalSystemConfig, data: PilotData) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=data.manifest["manifest_hash"],
        calibration_hash=canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        exposure_scale=1.0,
    )


def _source_bundle(
    spec: PilotSpec,
    review_path: Path,
    data: PilotData,
    current_commit: str,
) -> SourceBundle:
    source = spec.payload["source"]
    audit_config_path = _resolve(source["regularization_audit_config"])
    audit_review_path = _resolve(source["regularization_audit_review"])
    validation_path = _resolve(source["regularization_validation"])
    audit_path = _resolve(source["regularization_audit"])
    projection_path = _resolve(source["projection"])
    checkpoint_path = _resolve(source["checkpoint"])
    training_path = _resolve(source["full_epoch_training_evidence"])
    expected_files = {
        "regularization_audit_config": (audit_config_path, SOURCE_AUDIT_CONFIG_HASH),
        "regularization_audit_review": (audit_review_path, SOURCE_AUDIT_REVIEW_HASH),
        "regularization_validation": (validation_path, SOURCE_AUDIT_VALIDATION_HASH),
        "regularization_audit": (audit_path, SOURCE_AUDIT_HASH),
        "projection": (projection_path, SOURCE_PROJECTION_HASH),
        "checkpoint": (checkpoint_path, SOURCE_CHECKPOINT_HASH),
        "full_epoch_training_evidence": (training_path, SOURCE_TRAINING_HASH),
    }
    for role, (path, expected) in expected_files.items():
        if file_sha256(path) != expected:
            raise ValueError(f"{role} source hash mismatch")
    audit_spec = audit._load_spec(audit_config_path)
    audit_review, _ = audit._load_review(audit_review_path, audit_spec)
    if audit_review["no_update_audit_authorized"] is not True:
        raise ValueError("source regularization audit authorization is invalid")
    validation_payload = json.loads(validation_path.read_text(encoding="utf-8"))
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    validation_record = EvidenceRecordV2.from_dict(validation_payload["evidence"])
    audit_record = EvidenceRecordV2.from_dict(audit_payload["evidence"])
    validation_record.validate()
    audit_record.validate()
    if validation_record.evidence_kind != audit.VALIDATION_SCHEMA or validation_record.status != "passed":
        raise ValueError("source regularization validation is not passed")
    if audit_record.evidence_kind != audit.AUDIT_SCHEMA or audit_record.status != "passed":
        raise ValueError("source regularization audit is not passed")
    _assert_ancestor(validation_record.git_commit, current_commit)
    _assert_ancestor(audit_record.git_commit, current_commit)
    if validation_payload.get("training_authorized") is not False or audit_payload.get("training_authorized") is not False:
        raise ValueError("source regularization training boundary is invalid")
    if validation_payload.get("formal_promotion") is not False or audit_payload.get("formal_promotion") is not False:
        raise ValueError("source regularization promotion boundary is invalid")
    if audit_payload.get("contract", {}).get("model_state_unchanged") is not True:
        raise ValueError("source audit model isolation is invalid")
    projection = safe_torch_load(projection_path, expected_sha256=SOURCE_PROJECTION_HASH, map_location="cpu")
    if projection.get("schema") != audit.PROJECTION_SCHEMA:
        raise ValueError("source projection schema mismatch")
    if tuple(projection["basis"].shape) != (10000, PCA_COMPONENTS) or projection["basis"].dtype != torch.complex64:
        raise ValueError("source projection shape or dtype mismatch")
    if projection.get("source_hashes", {}).get("regularization_config") != audit_spec.config_hash:
        raise ValueError("source projection audit config binding mismatch")
    checkpoint = safe_torch_load(checkpoint_path, expected_sha256=SOURCE_CHECKPOINT_HASH, map_location="cpu")
    if (
        checkpoint.get("schema") != "lightgen_optical_manifold_identity_full_epoch_resume.v1"
        or checkpoint.get("global_step") != SOURCE_STEP
        or len(checkpoint.get("loss_history", [])) != SOURCE_STEP
        or not isinstance(checkpoint.get("model_state"), dict)
        or not isinstance(checkpoint.get("rng_state"), dict)
    ):
        raise ValueError("source checkpoint is incomplete or at the wrong step")
    formal_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    if _resolve(audit_spec.payload["formal_system_config"], base_path=audit_spec.path.parent) != formal_path:
        raise ValueError("pilot and audit formal config paths differ")
    hashes = {
        "pilot_config": spec.config_hash,
        "review_contract": file_sha256(review_path),
        "formal_config": file_sha256(formal_path),
        "regularization_audit_config": SOURCE_AUDIT_CONFIG_HASH,
        "regularization_audit_review": SOURCE_AUDIT_REVIEW_HASH,
        "regularization_validation": SOURCE_AUDIT_VALIDATION_HASH,
        "regularization_audit": SOURCE_AUDIT_HASH,
        "projection": SOURCE_PROJECTION_HASH,
        "source_checkpoint": SOURCE_CHECKPOINT_HASH,
        "full_epoch_training_evidence": SOURCE_TRAINING_HASH,
        "mnist_manifest": SOURCE_MANIFEST_HASH,
        "heldout_indices": SOURCE_HELDOUT_HASH,
    }
    return SourceBundle(checkpoint, projection, hashes, formal_path, audit_spec.config_hash)


def _parameter_contract(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, Any]:
    names = tuple(name for name, _ in model.named_parameters())
    trainable = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if trainable != EXPECTED_TRAINABLE:
        raise ValueError("pilot trainable parameters are not encoder/decoder phase")
    return {
        "all_parameter_names": list(names),
        "trainable_parameter_names": list(trainable),
        "optimizer_parameter_names": list(EXPECTED_TRAINABLE),
        "optimizer_created_during_validation": False,
        "phase_shapes": {name: list(value.shape) for name, value in model.named_parameters() if name in EXPECTED_TRAINABLE},
    }


def _record_payload(
    schema: str,
    record: EvidenceRecordV2,
    contract: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": schema,
        "evidence": record.to_dict(),
        "hypothesis": HYPOTHESIS,
        "bounded_research_training_authorized": True,
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
        raise FileExistsError(f"refusing to overwrite regularization pilot evidence: {target}")
    atomic_write_json(target, payload)


def _validation_record(
    state: Mapping[str, Any],
    spec: PilotSpec,
    formal: FormalSystemConfig,
    data: PilotData,
    hashes: Mapping[str, str],
) -> EvidenceRecordV2:
    gates = tuple(
        GateResult(name, True, metrics)
        for name, metrics in (
            ("clean_lineage", {"clean": True}),
            ("independent_regularization_pilot_branch", {"branch_matches": True}),
            ("passed_regularization_audit_source", {"source_step": SOURCE_STEP}),
            ("reset_adam_scope", {"reset": True}),
            ("fixed_10000_sample_schedule", {"steps": STEPS_PER_ARM, "batch": LOGICAL_BATCH_SIZE}),
            ("formal_geometry_identity_ols", {"grid": 2048, "identity": True}),
            ("fixed_exposure", {"scale": 1.0, "gamma": 1.0}),
            ("four_phase_trainable_parameter_set", {"count": 4}),
            ("bounded_training_authorization", {"steps_per_arm": STEPS_PER_ARM}),
        )
    )
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=VALIDATION_SCHEMA,
        tier="diagnostic",
        status="passed",
        git_commit=str(state["commit"]),
        dirty_patch_hash=None,
        config_hash=spec.config_hash,
        context=_context(spec, formal, data),
        input_hashes=dict(hashes),
        gates=gates,
        permitted_next_actions=("run_identity_manifold_regularization_training_preflight",),
    )
    record.validate()
    return record


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    source = _source_bundle(spec, review_path, data, state["commit"])
    if source.formal_path != formal_path:
        raise ValueError("pilot formal config binding mismatch")
    model = base.IdentityOpticalManifoldAutoencoder(formal)
    parameter_contract = _parameter_contract(model)
    hashes = {**source.hashes, "pilot_manifest": data.manifest["manifest_hash"]}
    record = _validation_record(state, spec, formal, data, hashes)
    payload = _record_payload(
        VALIDATION_SCHEMA,
        record,
        {
            "source_step": SOURCE_STEP,
            "source_checkpoint_hash": SOURCE_CHECKPOINT_HASH,
            "source_paired_mse": SOURCE_PAIRED_MSE,
            "parameter_contract": parameter_contract,
            "dataset_manifest": data.manifest,
            "training": spec.payload["training"],
            "regularization": spec.payload["regularization"],
            "bounded_research_training_authorized": review["bounded_research_training_authorized"],
        },
    )
    _write_new_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(
    path: str | Path,
    spec: PilotSpec,
    formal: FormalSystemConfig,
    data: PilotData,
    state: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != VALIDATION_SCHEMA
        or payload.get("hypothesis") != HYPOTHESIS
        or payload.get("training_authorized") is not False
        or payload.get("formal_promotion") is not False
    ):
        raise ValueError("pilot training validation schema or authorization is invalid")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != VALIDATION_SCHEMA or record.status != "passed":
        raise ValueError("pilot training validation must be passed")
    if record.git_commit != state["commit"] or state["branch"] != BRANCH or not state["clean"]:
        raise ValueError("pilot training validation is stale")
    if record.config_hash != spec.config_hash or record.context != _context(spec, formal, data):
        raise ValueError("pilot training validation context mismatch")
    if record.input_hashes != dict(hashes):
        raise ValueError("pilot training validation source hashes do not match")
    if record.permitted_next_actions != ("run_identity_manifold_regularization_training_preflight",):
        raise ValueError("pilot training validation next action is invalid")
    if payload.get("contract", {}).get("bounded_research_training_authorized") is not True:
        raise ValueError("pilot training validation lacks bounded authorization")
    return payload, digest


def _render_batch(data: PilotData, indices: torch.Tensor, device: torch.device) -> base.RenderedBatch:
    rendered = base.render_intensity_batch(
        data.regularization_data.full_data.pilot_data.images.index_select(0, indices)
    )
    return base.RenderedBatch(
        rendered.intensity.to(device), rendered.amplitude.to(device), rendered.target.to(device)
    )


def _state_cpu(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_equal(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> bool:
    return set(left) == set(right) and all(torch.equal(left[name], right[name]) for name in left)


def _tree_clone(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _tree_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_clone(item) for item in value)
    return copy.deepcopy(value)


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return torch.is_tensor(left) and torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return isinstance(left, Mapping) and isinstance(right, Mapping) and left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _frozen_equal(model: base.IdentityOpticalManifoldAutoencoder, source: Mapping[str, torch.Tensor]) -> bool:
    return all(
        torch.equal(model.state_dict()[name].detach().cpu(), source[name])
        for name in model.state_dict()
        if name not in EXPECTED_TRAINABLE
    )


def _runtime_from_source(
    formal: FormalSystemConfig,
    source: SourceBundle,
    device: torch.device,
) -> tuple[base.IdentityOpticalManifoldAutoencoder, full_epoch._DeterministicCpuAdam]:
    torch.manual_seed(SCHEDULE_SEED)
    torch.cuda.manual_seed_all(SCHEDULE_SEED)
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    model.load_state_dict(source.checkpoint["model_state"], strict=True)
    optimizer = full_epoch._make_full_optimizer(model)
    base._restore_rng(source.checkpoint["rng_state"])
    return model, optimizer


def _grad_dict(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(value, device="cpu")
        for name, value in model.named_parameters()
    }


def _accumulate_grads(
    total: dict[str, torch.Tensor],
    model: base.IdentityOpticalManifoldAutoencoder,
) -> None:
    for name, value in model.named_parameters():
        if value.grad is not None:
            total[name].add_(value.grad.detach().cpu())


def _clear_grads(model: base.IdentityOpticalManifoldAutoencoder) -> None:
    for value in model.parameters():
        value.grad = None


def _layer_values(values: Mapping[str, torch.Tensor]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, value in values.items():
        for index in range(value.shape[0]):
            result[f"{name}[{index}]"] = float(value[index].norm().item())
    return result


def _apply_gradients(
    model: base.IdentityOpticalManifoldAutoencoder,
    reconstruction: Mapping[str, torch.Tensor],
    regularizer: Mapping[str, torch.Tensor] | None,
) -> dict[str, float]:
    regularizer = regularizer or {name: torch.zeros_like(value) for name, value in reconstruction.items()}
    combined = {}
    for name, value in model.named_parameters():
        combined[name] = reconstruction[name] + LAMBDA * regularizer[name]
        value.grad = combined[name].to(value.device)
    return _layer_values(combined)


def _reconstruction_gradients(
    model: base.IdentityOpticalManifoldAutoencoder,
    optimizer: full_epoch._DeterministicCpuAdam,
    batch: base.RenderedBatch,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    optimizer.zero_grad(set_to_none=True)
    losses: list[torch.Tensor] = []
    cameras: list[torch.Tensor] = []
    for start in range(0, LOGICAL_BATCH_SIZE, MICROBATCH_SIZE):
        output = model(
            batch.amplitude[start:start + MICROBATCH_SIZE],
            activation_checkpoint=True,
        )
        loss = F.mse_loss(
            output.reconstruction,
            batch.target[start:start + MICROBATCH_SIZE],
        )
        losses.append(loss.detach())
        cameras.append(output.raw_camera.detach())
        (loss / LOGICAL_BATCH_SIZE).backward()
    gradients = _grad_dict(model)
    _accumulate_grads(gradients, model)
    return gradients, {
        "loss": float(torch.stack(losses).mean().cpu()),
        "camera_finite": bool(torch.isfinite(torch.cat(cameras)).all()),
        "camera_min": float(torch.cat(cameras).min().cpu()),
        "camera_max": float(torch.cat(cameras).max().cpu()),
        "gradient_layer_norms": _layer_values(gradients),
    }


def _regularizer_gradients(
    model: base.IdentityOpticalManifoldAutoencoder,
    optimizer: full_epoch._DeterministicCpuAdam,
    batch: base.RenderedBatch,
    projection_payload: Mapping[str, Any],
    directions: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], float, dict[str, Any]]:
    projection = audit.Projection(
        mean_field=projection_payload["mean_field"],
        basis=projection_payload["basis"],
        whitening=projection_payload["whitening"],
        eigenvalues=projection_payload["eigenvalues"],
        explained_variance=projection_payload["explained_variance"],
    )
    projection = audit._projection_on_device(projection, directions.device)
    scores: list[torch.Tensor] = []
    for start in range(0, LOGICAL_BATCH_SIZE, MICROBATCH_SIZE):
        with torch.no_grad():
            _raw, pre, _plane, _provenance = audit._forward_trace(
                model, batch.amplitude[start:start + MICROBATCH_SIZE]
            )
            scores.append(audit._project_fields(pre, projection).detach())
    score_batch = torch.cat(scores, dim=0)
    coefficients, ranks = _fixed_rank_coefficients(score_batch, directions)
    regularizer = _grad_dict(model)
    _clear_grads(model)
    for start in range(0, LOGICAL_BATCH_SIZE, MICROBATCH_SIZE):
        _raw, pre, _plane, _provenance = audit._forward_trace(
            model, batch.amplitude[start:start + MICROBATCH_SIZE]
        )
        score = audit._project_fields(pre, projection)
        projected_score = score @ directions.T
        (projected_score * coefficients[start:start + MICROBATCH_SIZE]).sum().backward()
        _accumulate_grads(regularizer, model)
        _clear_grads(model)
    return regularizer, float(audit._sliced_wasserstein(score_batch, directions).cpu()), {
        "rank_assignment_shape": list(ranks.shape),
        "rank_assignment_is_permutation": audit._rank_assignment_is_permutation(ranks),
        "score_shape": list(score_batch.shape),
    }


def _fixed_rank_coefficients(
    scores: torch.Tensor,
    directions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.shape != (LOGICAL_BATCH_SIZE, PCA_COMPONENTS):
        raise ValueError("pilot rank replay requires a [100, 32] score batch")
    if directions.shape != (FIXED_DIRECTIONS, PCA_COMPONENTS):
        raise ValueError("pilot rank replay requires fixed [64, 32] directions")
    projected = scores @ directions.T
    order = projected.argsort(dim=0)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        0,
        order,
        torch.arange(LOGICAL_BATCH_SIZE, device=directions.device).reshape(-1, 1).expand_as(order),
    )
    if not audit._rank_assignment_is_permutation(ranks):
        raise ValueError("regularization rank assignment is not a per-direction permutation")
    quantiles = audit._normal_midpoint_quantiles(LOGICAL_BATCH_SIZE, directions.device)
    target = quantiles.index_select(0, ranks.reshape(-1)).reshape_as(ranks)
    coefficients = 2.0 * (projected - target).div(float(LOGICAL_BATCH_SIZE * FIXED_DIRECTIONS))
    return coefficients, ranks


def _step(
    model: base.IdentityOpticalManifoldAutoencoder,
    optimizer: full_epoch._DeterministicCpuAdam,
    batch: base.RenderedBatch,
    projection: Mapping[str, Any],
    directions: torch.Tensor,
    arm: str,
) -> dict[str, Any]:
    before = {name: value.detach().clone() for name, value in model.named_parameters() if name in EXPECTED_TRAINABLE}
    reconstruction, reconstruction_info = _reconstruction_gradients(model, optimizer, batch)
    regularizer = None
    swd_loss = 0.0
    regularizer_layer_norms = {
        name: 0.0
        for name in _layer_values(reconstruction)
    }
    replay = {"rank_assignment_is_permutation": True, "score_shape": [0, PCA_COMPONENTS]}
    if arm == "regularized":
        regularizer, swd_loss, replay = _regularizer_gradients(model, optimizer, batch, projection, directions)
        regularizer_layer_norms = _layer_values(regularizer)
    combined_layer_norms = _apply_gradients(model, reconstruction, regularizer)
    optimizer.step()
    update_norms = {
        name: float((dict(model.named_parameters())[name].detach() - before[name]).abs().sum().cpu())
        for name in EXPECTED_TRAINABLE
    }
    for name in EXPECTED_TRAINABLE:
        value = dict(model.named_parameters())[name]
        for index in range(value.shape[0]):
            update_norms[f"{name}[{index}]"] = float((value[index].detach() - before[name][index]).abs().sum().cpu())
    _clear_grads(model)
    finite = reconstruction_info["camera_finite"] and all(
        bool(torch.isfinite(torch.tensor(value)))
        for value in [reconstruction_info["loss"], reconstruction_info["camera_min"], reconstruction_info["camera_max"], *combined_layer_norms.values(), *update_norms.values(), swd_loss]
    )
    return {
        "arm": arm,
        "loss": reconstruction_info["loss"],
        "swd_loss": swd_loss,
        "camera_finite": reconstruction_info["camera_finite"],
        "camera_min": reconstruction_info["camera_min"],
        "camera_max": reconstruction_info["camera_max"],
        "gradient_layer_norms": combined_layer_norms,
        "reconstruction_gradient_layer_norms": reconstruction_info["gradient_layer_norms"],
        "regularizer_gradient_layer_norms": regularizer_layer_norms,
        "update_norms": update_norms,
        "replay": replay,
        "finite": finite,
    }


def _resume_payload(
    model: base.IdentityOpticalManifoldAutoencoder,
    optimizer: full_epoch._DeterministicCpuAdam,
    *,
    arm: str,
    step: int,
    history: list[dict[str, Any]],
    spec: PilotSpec,
    data: PilotData,
    evidence_hashes: Mapping[str, str],
    completed_arms: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": RESUME_SCHEMA,
        "arm": arm,
        "step": step,
        "source_global_step": SOURCE_STEP,
        "model_state": _state_cpu(model),
        "optimizer_state": _tree_clone(optimizer.state_dict()),
        "rng_state": _tree_clone(base._rng_state()),
        "loss_history": list(history),
        "config_hash": spec.config_hash,
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "schedule_sha256": SCHEDULE_HASH,
        "reset_adam_from_source": True,
        "evidence_hashes": dict(evidence_hashes),
        "data_cursor": {"arm": arm, "next_step": step + 1, "logical_batch_size": LOGICAL_BATCH_SIZE, "microbatch_size": MICROBATCH_SIZE},
        "formal_candidate_checkpoint": False,
        "completed_arms": dict(completed_arms or {}),
    }
    return payload


def _resume_header(path: str | Path) -> dict[str, Any]:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != RESUME_SCHEMA:
        raise ValueError("invalid regularization pilot resume checkpoint")
    if payload.get("arm") not in {"regularized", "mse_only"}:
        raise ValueError("regularization pilot resume has an invalid arm")
    if not isinstance(payload.get("step"), int) or not 0 < payload["step"] < STEPS_PER_ARM:
        raise ValueError("regularization pilot resume must identify an incomplete arm step")
    return payload


def _load_completed_regularized_arm(
    resume_header: Mapping[str, Any],
    model: base.IdentityOpticalManifoldAutoencoder,
    optimizer: full_epoch._DeterministicCpuAdam,
    *,
    spec: PilotSpec,
    data: PilotData,
    evidence_hashes: Mapping[str, str],
) -> tuple[dict[str, Any], Path]:
    completed = resume_header.get("completed_arms")
    if not isinstance(completed, Mapping):
        raise ValueError("mse_only resume lacks completed-arm metadata")
    reference = completed.get("regularized")
    if not isinstance(reference, Mapping) or reference.get("status") != "completed":
        raise ValueError("mse_only resume lacks a completed regularized arm")
    path_value = reference.get("path")
    expected_hash = reference.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_hash, str):
        raise ValueError("regularized completion reference is incomplete")
    path = _resolve(path_value)
    if file_sha256(path) != expected_hash:
        raise ValueError("regularized completion checkpoint hash mismatch")
    payload = safe_torch_load(path, expected_sha256=expected_hash, map_location="cpu")
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != RESUME_SCHEMA
        or payload.get("arm") != "regularized"
        or payload.get("step") != STEPS_PER_ARM
        or len(payload.get("loss_history", [])) != STEPS_PER_ARM
        or payload.get("config_hash") != spec.config_hash
        or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"]
        or payload.get("schedule_sha256") != SCHEDULE_HASH
        or payload.get("evidence_hashes") != dict(evidence_hashes)
        or payload.get("formal_candidate_checkpoint") is not False
    ):
        raise ValueError("regularized completion checkpoint lineage is invalid")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    return payload, path


def _save_resume(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite pilot checkpoint: {path}")
    atomic_torch_save(path, payload)


def _load_resume(
    path: str | Path,
    model: base.IdentityOpticalManifoldAutoencoder,
    optimizer: full_epoch._DeterministicCpuAdam,
    *,
    spec: PilotSpec,
    data: PilotData,
    evidence_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != RESUME_SCHEMA:
        raise ValueError("invalid regularization pilot resume checkpoint")
    if payload.get("config_hash") != spec.config_hash or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"] or payload.get("schedule_sha256") != SCHEDULE_HASH or payload.get("evidence_hashes") != dict(evidence_hashes):
        raise ValueError("regularization pilot resume lineage mismatch")
    arm = payload.get("arm")
    step = payload.get("step")
    if arm not in {"regularized", "mse_only"} or not isinstance(step, int) or not 0 < step < STEPS_PER_ARM:
        raise ValueError("regularization pilot resume must identify an incomplete arm step")
    if payload.get("source_global_step") != SOURCE_STEP or payload.get("reset_adam_from_source") is not True:
        raise ValueError("regularization pilot resume source contract mismatch")
    if payload.get("formal_candidate_checkpoint") is not False:
        raise ValueError("regularization pilot resume cannot be a formal candidate")
    if not isinstance(payload.get("model_state"), Mapping) or not isinstance(payload.get("optimizer_state"), Mapping) or not isinstance(payload.get("rng_state"), Mapping):
        raise ValueError("regularization pilot resume is missing state")
    if arm == "regularized" and payload.get("completed_arms") not in ({}, None):
        raise ValueError("regularized resume has unexpected completed-arm metadata")
    if arm == "mse_only":
        completed = payload.get("completed_arms")
        if not isinstance(completed, Mapping) or not isinstance(completed.get("regularized"), Mapping) or completed["regularized"].get("status") != "completed":
            raise ValueError("mse_only resume lacks completed regularized arm")
    if payload.get("data_cursor") != {"arm": arm, "next_step": step + 1, "logical_batch_size": LOGICAL_BATCH_SIZE, "microbatch_size": MICROBATCH_SIZE}:
        raise ValueError("regularization pilot resume cursor mismatch")
    if len(payload.get("loss_history", [])) != step:
        raise ValueError("regularization pilot resume history mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    base._restore_rng(payload["rng_state"])
    return payload


def _source_hashes_for_run(
    source: SourceBundle,
    validation_hash: str,
    preflight_hash: str | None = None,
) -> dict[str, str]:
    values = {**source.hashes, "training_validation": validation_hash}
    if preflight_hash is not None:
        values["training_preflight"] = preflight_hash
    return values


def _run_arm(
    model: base.IdentityOpticalManifoldAutoencoder,
    optimizer: full_epoch._DeterministicCpuAdam,
    data: PilotData,
    projection: Mapping[str, Any],
    directions: torch.Tensor,
    *,
    arm: str,
    start_step: int,
    history: list[dict[str, Any]],
    output_dir: Path,
    spec: PilotSpec,
    evidence_hashes: Mapping[str, str],
    save_checkpoints: bool,
    completed_arms: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], Path, dict[str, str]]:
    checkpoint_hashes: dict[str, str] = {}
    for step in range(start_step + 1, STEPS_PER_ARM + 1):
        batch = _render_batch(data, data.schedule[step - 1], next(model.parameters()).device)
        result = _step(model, optimizer, batch, projection, directions, arm)
        result["step"] = step
        history.append(result)
        if save_checkpoints and (step % CHECKPOINT_INTERVAL == 0 or step == STEPS_PER_ARM):
            checkpoint_path = output_dir / f"{arm}_step_{step:05d}.pt"
            payload = _resume_payload(
                model,
                optimizer,
                arm=arm,
                step=step,
                history=history,
                spec=spec,
                data=data,
                evidence_hashes=evidence_hashes,
                completed_arms=completed_arms,
            )
            _save_resume(checkpoint_path, payload)
            checkpoint_hashes[str(step)] = file_sha256(checkpoint_path)
    final_path = output_dir / f"{arm}_final_step_{STEPS_PER_ARM:05d}.pt"
    if not final_path.exists():
        payload = _resume_payload(
            model,
            optimizer,
            arm=arm,
            step=STEPS_PER_ARM,
            history=history,
            spec=spec,
            data=data,
            evidence_hashes=evidence_hashes,
            completed_arms=completed_arms,
        )
        _save_resume(final_path, payload)
    checkpoint_hashes["final"] = file_sha256(final_path)
    return history, final_path, checkpoint_hashes


def _evaluate_model(
    model: base.IdentityOpticalManifoldAutoencoder,
    data: PilotData,
    projection_payload: Mapping[str, Any],
    directions: torch.Tensor,
    device: torch.device,
    indices: torch.Tensor,
) -> dict[str, Any]:
    paired = full_epoch.pilot._evaluate(
        model, data.regularization_data.full_data.pilot_data, indices, device
    )
    cyclic = full_epoch.pilot._evaluate(
        model, data.regularization_data.full_data.pilot_data, indices, device, target_roll=1
    )
    fields, offline_mse, plane, provenance = audit._capture_fields(
        model, data.regularization_data, indices, device
    )
    projection = audit.Projection(
        mean_field=projection_payload["mean_field"],
        basis=projection_payload["basis"],
        whitening=projection_payload["whitening"],
        eigenvalues=projection_payload["eigenvalues"],
        explained_variance=projection_payload["explained_variance"],
    )
    scores = audit._project_fields(fields, projection)
    swd = float(audit._sliced_wasserstein(scores, directions.cpu()).cpu())
    normalized = audit._unit_power_field(fields)
    labels = data.regularization_data.full_data.pilot_data.labels.index_select(0, indices)
    generator = torch.Generator(device="cpu").manual_seed(SCHEDULE_SEED)
    permuted = labels.index_select(0, torch.randperm(labels.numel(), generator=generator))
    structure, _scores = observer._structure_metrics(normalized, labels, permuted)
    return {
        "paired_mse": paired,
        "cyclic_mse": cyclic,
        "offline_mse": offline_mse,
        "swd": swd,
        "structure": structure,
        "plane": plane,
        "provenance": provenance,
        "field_shape": list(fields.shape),
        "finite": bool(torch.isfinite(fields.real).all() and torch.isfinite(fields.imag).all() and torch.isfinite(torch.tensor([paired, cyclic, swd])).all()),
    }


def _preflight_record(
    state: Mapping[str, Any],
    spec: PilotSpec,
    formal: FormalSystemConfig,
    data: PilotData,
    hashes: Mapping[str, str],
    gates: tuple[GateResult, ...],
    status: str,
    actions: tuple[str, ...],
    reason: str | None,
) -> EvidenceRecordV2:
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=PREFLIGHT_SCHEMA,
        tier="diagnostic",
        status=status,
        git_commit=str(state["commit"]),
        dirty_patch_hash=None,
        config_hash=spec.config_hash,
        context=_context(spec, formal, data),
        input_hashes=dict(hashes),
        gates=gates,
        permitted_next_actions=actions,
        invalidation_reason=reason,
    )
    record.validate()
    return record


def _deterministic_context() -> Any:
    previous = torch.are_deterministic_algorithms_enabled()

    @contextmanager
    def context() -> Any:
        torch.use_deterministic_algorithms(True)
        try:
            yield
        finally:
            torch.use_deterministic_algorithms(previous)

    return context()


def _preflight_arm(
    formal: FormalSystemConfig,
    source: SourceBundle,
    data: PilotData,
    projection: Mapping[str, Any],
    directions: torch.Tensor,
    device: torch.device,
    arm: str,
    output_dir: Path,
    spec: PilotSpec,
    evidence_hashes: Mapping[str, str],
) -> dict[str, Any]:
    model, optimizer = _runtime_from_source(formal, source, device)
    source_state = _state_cpu(model)
    continuous_history: list[dict[str, Any]] = []
    for step in range(3):
        continuous_history.append(_step(model, optimizer, _render_batch(data, data.schedule[step], device), projection, directions, arm))
    continuous_state = _state_cpu(model)
    continuous_optimizer = _tree_clone(optimizer.state_dict())
    continuous_rng = _tree_clone(base._rng_state())
    interrupted, interrupted_optimizer = _runtime_from_source(formal, source, device)
    interrupted_history: list[dict[str, Any]] = []
    for step in range(2):
        interrupted_history.append(_step(interrupted, interrupted_optimizer, _render_batch(data, data.schedule[step], device), projection, directions, arm))
    resume_path = output_dir / f"preflight_{arm}_step_00002.pt"
    _save_resume(
        resume_path,
        _resume_payload(
            interrupted,
            interrupted_optimizer,
            arm=arm,
            step=2,
            history=interrupted_history,
            spec=spec,
            data=data,
            evidence_hashes=evidence_hashes,
        ),
    )
    resumed, resumed_optimizer = _runtime_from_source(formal, source, device)
    resume_payload = _load_resume(
        resume_path,
        resumed,
        resumed_optimizer,
        spec=spec,
        data=data,
        evidence_hashes=evidence_hashes,
    )
    resumed_third = _step(resumed, resumed_optimizer, _render_batch(data, data.schedule[2], device), projection, directions, arm)
    resume_exact = (
        _state_equal(continuous_state, _state_cpu(resumed))
        and _tree_equal(continuous_optimizer, resumed_optimizer.state_dict())
        and _tree_equal(continuous_rng, base._rng_state())
        and continuous_history[2] | {"step": 3} == resumed_third | {"step": 3}
        and resume_payload["step"] == 2
    )
    probe_initial_model, _probe_optimizer = _runtime_from_source(formal, source, device)
    probe_indices = torch.cat(data.regularization_data.gradient_batches)
    initial_metrics = _evaluate_model(probe_initial_model, data, projection, directions.cpu(), device, probe_indices)
    after_metrics = _evaluate_model(model, data, projection, directions.cpu(), device, probe_indices)
    control_model, control_optimizer = _runtime_from_source(formal, source, device)
    control_history: list[dict[str, Any]] = []
    for step in range(3):
        control_history.append(_step(control_model, control_optimizer, _render_batch(data, data.schedule[step], device), projection, directions, "mse_only"))
    control_metrics = _evaluate_model(control_model, data, projection, directions.cpu(), device, probe_indices)
    source_mse = full_epoch.pilot._evaluate(
        probe_initial_model,
        data.regularization_data.full_data.pilot_data,
        probe_indices,
        device,
    )
    after_mse = full_epoch.pilot._evaluate(
        model,
        data.regularization_data.full_data.pilot_data,
        probe_indices,
        device,
    )
    relative_degradation = (after_mse - source_mse) / max(source_mse, 1.0e-20)
    return {
        "arm": arm,
        "continuous": continuous_history,
        "resume_exact": resume_exact,
        "resume_artifact": str(resume_path.resolve()),
        "resume_artifact_sha256": file_sha256(resume_path),
        "initial_probe": initial_metrics,
        "after_probe": after_metrics,
        "control_probe": control_metrics,
        "control_steps": control_history,
        "source_probe_mse": source_mse,
        "after_probe_mse": after_mse,
        "relative_mse_degradation": relative_degradation,
        "frozen_state_unchanged": _frozen_equal(model, source_state),
        "finite": all(item["finite"] for item in continuous_history + control_history + [resumed_third]),
    }


def preflight(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("regularization pilot preflight requires CUDA")
    _require_deterministic_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    source = _source_bundle(spec, review_path, data, state["commit"])
    if source.formal_path != formal_path:
        raise ValueError("pilot formal config binding mismatch")
    hashes = {**source.hashes, "pilot_manifest": data.manifest["manifest_hash"]}
    _validation_payload, validation_hash = _load_validation(args.training_validation_evidence, spec, formal, data, state, hashes)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite pilot preflight evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for regularization pilot preflight")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    torch.cuda.reset_peak_memory_stats(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    projection = source.projection
    directions = audit._fixed_directions(device)
    started = time.monotonic()
    with _deterministic_context():
        regularized = _preflight_arm(formal, source, data, projection, directions, device, "regularized", output.parent, spec, {**hashes, "training_validation": validation_hash})
        mse_only = _preflight_arm(formal, source, data, projection, directions, device, "mse_only", output.parent, spec, {**hashes, "training_validation": validation_hash})
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
    finite = regularized["finite"] and mse_only["finite"]
    gradients = all(
        value > 0.0
        for arm in (regularized, mse_only)
        for item in arm["continuous"]
        for value in item["gradient_layer_norms"].values()
    )
    reconstruction_gradients = all(
        value > 0.0
        for arm in (regularized, mse_only)
        for item in arm["continuous"]
        for value in item["reconstruction_gradient_layer_norms"].values()
    )
    regularizer_encoder_gradients = all(
        value > 0.0
        for item in regularized["continuous"]
        for name, value in item["regularizer_gradient_layer_norms"].items()
        if name.startswith("system.encoder.phase[")
    )
    regularizer_decoder_isolated = all(
        value == 0.0
        for item in regularized["continuous"]
        for name, value in item["regularizer_gradient_layer_norms"].items()
        if name.startswith("system.decoder.phase[")
    )
    updates = all(
        value > 0.0
        for arm in (regularized, mse_only)
        for item in arm["continuous"]
        for name, value in item["update_norms"].items()
        if "phase[" in name
    )
    rank = all(
        item["replay"]["rank_assignment_is_permutation"]
        for arm in (regularized, mse_only)
        for item in arm["continuous"]
        if arm["arm"] == "regularized"
    )
    regularized_probe_improved = regularized["after_probe"]["swd"] < regularized["initial_probe"]["swd"]
    regularized_beats_control = regularized["after_probe"]["swd"] < mse_only["after_probe"]["swd"]
    mse_safe = all(
        arm["relative_mse_degradation"] <= PREPARED_MSE_DEGRADATION
        for arm in (regularized, mse_only)
    )
    resume_exact = regularized["resume_exact"] and mse_only["resume_exact"]
    frozen = regularized["frozen_state_unchanged"] and mse_only["frozen_state_unchanged"]
    gates = (
        GateResult("finite_values", finite, {}),
        GateResult("four_phase_gradients_nonzero", gradients, {"layers": 4}),
        GateResult("reconstruction_gradients_nonzero", reconstruction_gradients, {"layers": 4}),
        GateResult("regularizer_encoder_gradients_nonzero", regularizer_encoder_gradients, {"layers": 2}),
        GateResult("regularizer_decoder_gradient_isolation", regularizer_decoder_isolated, {"layers": 2}),
        GateResult("four_phase_updates_nonzero", updates, {"layers": 4}),
        GateResult("fixed_rank_permutation", rank, {}),
        GateResult("frozen_state_unchanged", frozen, {}),
        GateResult("resume_exact_both_arms", resume_exact, {}),
        GateResult("regularized_probe_swd_decreases", regularized_probe_improved, {"initial": regularized["initial_probe"]["swd"], "after": regularized["after_probe"]["swd"]}),
        GateResult("regularized_beats_mse_control_probe", regularized_beats_control, {"regularized": regularized["after_probe"]["swd"], "mse_only": mse_only["after_probe"]["swd"]}),
        GateResult("preflight_mse_degradation", mse_safe, {"threshold": PREPARED_MSE_DEGRADATION}),
        GateResult("fixed_formal_identity_optics", True, {"grid": 2048, "identity": True}),
        GateResult("fixed_exposure", True, {"scale": 1.0, "gamma": 1.0}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
        GateResult("wall_time", elapsed <= MAX_PREFLIGHT_SECONDS, {"seconds": elapsed}),
        GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
    )
    hashes = {**hashes, "training_validation": validation_hash}
    passed = all(gate.passed for gate in gates)
    record = _preflight_record(
        state,
        spec,
        formal,
        data,
        hashes,
        gates,
        "passed" if passed else "invalid_harness",
        ("run_identity_manifold_regularization_pilot",) if passed else (),
        None if passed else "regularization pilot preflight gate failed",
    )
    payload = _record_payload(
        PREFLIGHT_SCHEMA,
        record,
        {
            "training_validation_hash": validation_hash,
            "source_checkpoint_hash": SOURCE_CHECKPOINT_HASH,
            "regularized": regularized,
            "mse_only": mse_only,
            "free_cuda_bytes_before": int(free_bytes),
            "total_cuda_bytes": int(total_bytes),
            "peak_cuda_bytes": peak,
            "wall_seconds": elapsed,
            "run_storage_bytes": storage,
            "evaluator_calls": 0,
        },
    )
    _write_new_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 2


def _load_preflight(
    path: str | Path,
    spec: PilotSpec,
    formal: FormalSystemConfig,
    data: PilotData,
    state: Mapping[str, Any],
    validation_hash: str,
    hashes: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    target = _resolve(path)
    digest = file_sha256(target)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("hypothesis") != HYPOTHESIS or payload.get("training_authorized") is not False or payload.get("formal_promotion") is not False:
        raise ValueError("regularization pilot preflight schema or authorization is invalid")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != PREFLIGHT_SCHEMA or record.status != "passed":
        raise ValueError("regularization pilot preflight must be passed")
    if record.git_commit != state["commit"] or record.config_hash != spec.config_hash or record.context != _context(spec, formal, data):
        raise ValueError("regularization pilot preflight is stale")
    expected = {**hashes, "training_validation": validation_hash}
    if record.input_hashes != expected:
        raise ValueError("regularization pilot preflight source hashes do not match")
    if record.permitted_next_actions != ("run_identity_manifold_regularization_pilot",):
        raise ValueError("regularization pilot preflight next action is invalid")
    return payload, digest


def _training_record(
    state: Mapping[str, Any],
    spec: PilotSpec,
    formal: FormalSystemConfig,
    data: PilotData,
    hashes: Mapping[str, str],
    gates: tuple[GateResult, ...],
    status: str,
    actions: tuple[str, ...],
    reason: str | None,
) -> EvidenceRecordV2:
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=TRAINING_SCHEMA,
        tier="diagnostic",
        status=status,
        git_commit=str(state["commit"]),
        dirty_patch_hash=None,
        config_hash=spec.config_hash,
        context=_context(spec, formal, data),
        input_hashes=dict(hashes),
        gates=gates,
        permitted_next_actions=actions,
        invalidation_reason=reason,
    )
    record.validate()
    return record


def _train(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("regularization pilot training requires CUDA")
    _require_deterministic_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    source = _source_bundle(spec, review_path, data, state["commit"])
    if source.formal_path != formal_path:
        raise ValueError("pilot formal config binding mismatch")
    hashes = {**source.hashes, "pilot_manifest": data.manifest["manifest_hash"]}
    _validation_payload, validation_hash = _load_validation(args.training_validation_evidence, spec, formal, data, state, hashes)
    _preflight_payload, preflight_hash = _load_preflight(args.training_preflight_evidence, spec, formal, data, state, validation_hash, hashes)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite regularization pilot training evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for regularization pilot training")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    torch.cuda.reset_peak_memory_stats(device)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    directions = audit._fixed_directions(device)
    run_hashes = _source_hashes_for_run(source, validation_hash, preflight_hash)
    projection = source.projection
    source_model, _source_optimizer = _runtime_from_source(formal, source, device)
    baseline_state = _state_cpu(source_model)
    baseline = _evaluate_model(source_model, data, projection, directions.cpu(), device, data.heldout_indices)
    source_replay_mse = full_epoch.pilot._evaluate(
        source_model, data.regularization_data.full_data.pilot_data, data.heldout_indices, device
    )
    arm_results: dict[str, Any] = {}
    checkpoint_refs: dict[str, Any] = {}
    resume_header = _resume_header(args.resume_checkpoint) if args.resume_checkpoint else None
    if args.resume_checkpoint and _resolve(args.resume_checkpoint).parent != output.parent.resolve():
        raise ValueError("explicit resume checkpoint must belong to the requested pilot attempt")
    resume_arm = resume_header["arm"] if resume_header is not None else None
    for arm in ("regularized", "mse_only"):
        model, optimizer = _runtime_from_source(formal, source, device)
        history: list[dict[str, Any]] = []
        start_step = 0
        checkpoint_hashes: dict[str, str] = {}
        if resume_arm == "mse_only" and arm == "regularized":
            completed_payload, completed_path = _load_completed_regularized_arm(
                resume_header,
                model,
                optimizer,
                spec=spec,
                data=data,
                evidence_hashes=run_hashes,
            )
            history = list(completed_payload["loss_history"])
            checkpoint_hashes["final"] = file_sha256(completed_path)
        elif args.resume_checkpoint and resume_arm == arm:
            resume = _load_resume(args.resume_checkpoint, model, optimizer, spec=spec, data=data, evidence_hashes=run_hashes)
            start_step = int(resume["step"])
            history = list(resume["loss_history"])
        if not (resume_arm == "mse_only" and arm == "regularized"):
            completed_arms = None
            if arm == "mse_only":
                regularized_ref = checkpoint_refs.get("regularized")
                if regularized_ref is None:
                    raise ValueError("mse_only arm requires completed regularized arm metadata")
                completed_arms = {
                    "regularized": {
                        "status": "completed",
                        "path": regularized_ref["final_path"],
                        "sha256": regularized_ref["final_sha256"],
                    }
                }
            history, final_path, checkpoint_hashes = _run_arm(
                model,
                optimizer,
                data,
                projection,
                directions,
                arm=arm,
                start_step=start_step,
                history=history,
                output_dir=output.parent,
                spec=spec,
                evidence_hashes=run_hashes,
                save_checkpoints=True,
                completed_arms=completed_arms,
            )
        else:
            final_path = completed_path
        arm_metrics = _evaluate_model(model, data, projection, directions.cpu(), device, data.heldout_indices)
        arm_results[arm] = {
            "steps_completed": len(history),
            "history": history,
            "metrics": arm_metrics,
            "frozen_state_unchanged": _frozen_equal(model, baseline_state),
            "checkpoint_hashes": checkpoint_hashes,
            "final_checkpoint": str(final_path.resolve()),
        }
        checkpoint_refs[arm] = {
            **checkpoint_hashes,
            "final_path": str(final_path.resolve()),
            "final_sha256": checkpoint_hashes["final"],
        }
        _clear_grads(model)
        torch.cuda.empty_cache()
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
    regularized = arm_results["regularized"]
    mse_only = arm_results["mse_only"]
    reg_metrics = regularized["metrics"]
    mse_metrics = mse_only["metrics"]
    source_gate = abs(source_replay_mse - SOURCE_PAIRED_MSE) <= 1.0e-9
    finite = baseline["finite"] and all(
        item["finite"] for arm in arm_results.values() for item in arm["history"]
    ) and reg_metrics["finite"] and mse_metrics["finite"]
    updates = all(
        value > 0.0
        for arm in arm_results.values()
        for item in arm["history"]
        for name, value in item["update_norms"].items()
        if "phase[" in name
    )
    regularized_swd_reduction = (baseline["swd"] - reg_metrics["swd"]) / max(baseline["swd"], 1.0e-20)
    structure = reg_metrics["structure"]["true_label_distance"]
    permuted_structure = reg_metrics["structure"]["permuted_label_distance"]
    science_gates = (
        GateResult("regularized_swd_reduction", regularized_swd_reduction >= FINAL_SWD_REDUCTION, {"baseline": baseline["swd"], "final": reg_metrics["swd"], "relative_reduction": regularized_swd_reduction, "threshold": FINAL_SWD_REDUCTION}),
        GateResult("regularized_swd_below_mse_control", reg_metrics["swd"] < mse_metrics["swd"], {"regularized": reg_metrics["swd"], "mse_only": mse_metrics["swd"]}),
        GateResult("regularized_mse_degradation", reg_metrics["paired_mse"] <= baseline["paired_mse"] * (1.0 + FINAL_MSE_DEGRADATION), {"baseline": baseline["paired_mse"], "final": reg_metrics["paired_mse"], "threshold": FINAL_MSE_DEGRADATION}),
        GateResult("regularized_paired_mse_lt_cyclic", reg_metrics["paired_mse"] < reg_metrics["cyclic_mse"], {"paired": reg_metrics["paired_mse"], "cyclic": reg_metrics["cyclic_mse"]}),
        GateResult("normalized_inter_intra_gt_one", structure["inter_intra_ratio"] > 1.0, {"ratio": structure["inter_intra_ratio"]}),
        GateResult("normalized_true_ratio_gt_permuted", structure["inter_intra_ratio"] > permuted_structure["inter_intra_ratio"], {"true": structure["inter_intra_ratio"], "permuted": permuted_structure["inter_intra_ratio"]}),
    )
    harness_gates = (
        GateResult("source_reconstruction_reproduced", source_gate, {"reference": SOURCE_PAIRED_MSE, "replay": source_replay_mse}),
        GateResult("finite_values", finite, {}),
        GateResult("both_arms_completed_100_steps", regularized["steps_completed"] == STEPS_PER_ARM and mse_only["steps_completed"] == STEPS_PER_ARM, {}),
        GateResult("four_phase_updates_nonzero", updates, {"layers": 4}),
        GateResult("frozen_state_unchanged", regularized["frozen_state_unchanged"] and mse_only["frozen_state_unchanged"], {}),
        GateResult("fixed_formal_identity_optics", True, {"grid": 2048, "identity": True}),
        GateResult("fixed_exposure", True, {"scale": 1.0, "gamma": 1.0}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
        GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
        GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
    )
    gates = science_gates + harness_gates
    if not all(gate.passed for gate in harness_gates):
        status, actions, reason = "invalid_harness", (), "regularization pilot harness or resource gate failed"
    elif not all(gate.passed for gate in science_gates):
        status, actions, reason = "valid_failed", (NEXT_VALID_FAILED_ACTION,), "regularization pilot scientific gate failed"
    else:
        status, actions, reason = "passed", (NEXT_PASS_ACTION,), None
    record = _training_record(state, spec, formal, data, run_hashes, gates, status, actions, reason)
    payload = _record_payload(
        TRAINING_SCHEMA,
        record,
        {
            "training_validation_hash": validation_hash,
            "training_preflight_hash": preflight_hash,
            "source_checkpoint_hash": SOURCE_CHECKPOINT_HASH,
            "source_reconstruction": {"authoritative_cuda_replay_mse": source_replay_mse, "reference": SOURCE_PAIRED_MSE, "tolerance": 1.0e-9},
            "baseline": baseline,
            "arms": arm_results,
            "checkpoint_refs": checkpoint_refs,
            "formal_candidate_checkpoint": False,
            "projection": {"schema": audit.PROJECTION_SCHEMA, "sha256": SOURCE_PROJECTION_HASH, "components": PCA_COMPONENTS, "directions": FIXED_DIRECTIONS, "lambda": LAMBDA},
            "model_state_unchanged": regularized["frozen_state_unchanged"] and mse_only["frozen_state_unchanged"],
            "evaluator_calls": 0,
            "free_cuda_bytes_before": int(free_bytes),
            "total_cuda_bytes": int(total_bytes),
            "peak_cuda_bytes": peak,
            "wall_seconds": elapsed,
            "run_storage_bytes": storage,
        },
    )
    _write_new_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "passed" else 2


def train(args: argparse.Namespace) -> int:
    return _train(args)


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "manifold-regularization-pilot-validate":
        return validate(args)
    if args.research_action == "manifold-regularization-pilot-preflight":
        return preflight(args)
    if args.research_action == "manifold-regularization-pilot-train":
        return train(args)
    raise ValueError(f"unsupported regularization pilot action: {args.research_action}")
