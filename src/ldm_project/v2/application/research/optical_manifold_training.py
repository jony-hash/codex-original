"""Bounded H-LOMA-I0 Identity reconstruction pilot lifecycle.

This module is a separate research boundary from the first-round optical
manifold harness.  It consumes the first-round evidence, adds a fixed
stratified MNIST schedule, and exposes only the explicitly authorized
1000-update diagnostic pilot.
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
from torch import nn
from torch.nn import functional as F

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


BRANCH = "research/v2-lightgen-optical-manifold-pilot-v1"
HYPOTHESIS = "H-LOMA-I0"
PILOT_SCHEMA = "lightgen_optical_manifold_identity_training_pilot_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_identity_training_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_identity_training_validation.v1"
PREFLIGHT_SCHEMA = "lightgen_optical_manifold_identity_training_preflight.v1"
TRAINING_SCHEMA = "lightgen_optical_manifold_identity_training.v1"
RESUME_SCHEMA = "lightgen_optical_manifold_identity_training_resume.v1"
SOURCE_VALIDATION_SCHEMA = "lightgen_optical_manifold_identity_validation.v1"
SOURCE_PREFLIGHT_SCHEMA = "lightgen_optical_manifold_identity_preflight.v1"
SOURCE_VALIDATION_HASH = (
    "9c763024021707314decb945d5b7a692d1b86e20421047a89d81746b70dfe57e"
)
SOURCE_PREFLIGHT_HASH = (
    "67a4ff53b86e329596234e8dd7154ce85ac2f22cc6974f089a2acf728eefe166"
)
BASE_CONFIG_HASH = "cfe23da4695ffcf7a1c7749f799e2e02aad971ec706cf211bd96acfbe3984040"
NEXT_PREFLIGHT_ACTION = "run_bounded_identity_reconstruction_pilot"
NEXT_PASS_ACTION = "manual_review_full_epoch_identity_reconstruction_contract"
EXPECTED_TRAINABLE = base.EXPECTED_TRAINABLE
MAX_STEPS = 1000
LOGICAL_BATCH_SIZE = 10
MICROBATCH_SIZE = 1
HELDOUT_PER_LABEL = 100
PREFLIGHT_HELDOUT_PER_LABEL = 10
SPLIT_SEED = 1234
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_PREFLIGHT_SECONDS = 300.0
MAX_PILOT_SECONDS = 1800.0
MAX_STORAGE_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class PilotSpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str
    base_path: Path


@dataclass(frozen=True)
class PilotData:
    images: torch.Tensor
    labels: torch.Tensor
    train_indices: torch.Tensor
    validation_indices: torch.Tensor
    schedule: torch.Tensor
    heldout_indices: torch.Tensor
    preflight_heldout_indices: torch.Tensor
    manifest: dict[str, Any]


@dataclass(frozen=True)
class StepBatch:
    amplitude: torch.Tensor
    target: torch.Tensor


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"]:
        raise ValueError("Identity reconstruction pilot requires a clean worktree")
    if state["branch"] != BRANCH:
        raise ValueError(f"Identity reconstruction pilot requires branch {BRANCH}")


def _expected_pilot_payload() -> dict[str, Any]:
    return {
        "schema": PILOT_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "base_config": "configs/research/lightgen_optical_manifold_identity_v1.yaml",
        "base_config_sha256": BASE_CONFIG_HASH,
        "source_preflight_evidence": (
            "outputs/20260827_v2/runs/"
            "lightgen_optical_manifold_identity_v1_attempt_02/preflight.json"
        ),
        "source_preflight_sha256": SOURCE_PREFLIGHT_HASH,
        "dataset": {
            "train": True,
            "split_seed": 1234,
            "train_count": 55000,
            "validation_count": 5000,
            "schedule": "balanced_one_per_label_without_replacement",
            "input_shape": [1, 28, 28],
            "rendered_shape": [1, 200, 200],
            "heldout_per_label": 100,
            "heldout_selection": "first_occurrence_in_validation_split",
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": 1.0e-3,
            "max_steps": 1000,
            "logical_batch_size": 10,
            "microbatch_size": 1,
            "activation_checkpoint": True,
            "initialization_phase": "zero",
            "initialization_seed": 1234,
            "checkpoint_interval": 100,
            "deterministic": True,
        },
        "counterfactual": {
            "target_permutation": "cyclic_shift_by_one",
            "expected_decision": "reject_5_percent_heldout_gate",
        },
        "gates": {
            "heldout_relative_mse_reduction_min": 0.05,
            "paired_mse_must_be_below_permuted_mse": True,
            "max_wall_seconds": 1800,
            "max_peak_cuda_bytes": 5368709120,
            "max_storage_bytes": 2147483648,
        },
        "authorization": {
            "reviewer_role": "project_owner",
            "decision": "approve_bounded_research_training",
            "bounded_research_training_authorized": True,
            "formal_training_authorized": False,
            "formal_promotion": False,
            "post_pass_next_action": NEXT_PASS_ACTION,
            "prohibited": [
                "full_epoch",
                "smf",
                "fmf",
                "latent_sampling",
                "evaluator",
                "generation",
                "promotion",
            ],
        },
    }


def _load_pilot_spec(path: str | Path) -> PilotSpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload != _expected_pilot_payload():
        raise ValueError("pilot config does not exactly match the registered contract")
    base_path = _resolve(payload["base_config"], base_path=target.parent)
    base_spec = base._load_spec(base_path)
    if base_spec.config_hash != payload["base_config_sha256"]:
        raise ValueError("pilot base config canonical hash mismatch")
    return PilotSpec(target, payload, canonical_sha256(payload), base_path)


def _load_review(path: str | Path, spec: PilotSpec) -> tuple[dict[str, Any], Path]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_bounded_research_training",
        "source_preflight_evidence": spec.payload["source_preflight_evidence"],
        "source_preflight_sha256": SOURCE_PREFLIGHT_HASH,
        "approved_contract": {
            "max_steps": MAX_STEPS,
            "heldout_relative_mse_reduction_min": 0.05,
            "post_pass_next_action": NEXT_PASS_ACTION,
        },
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("project-owner review contract does not match pilot scope")
    return payload, target


def _load_formal(spec: PilotSpec) -> tuple[FormalSystemConfig, Path]:
    base_spec = base._load_spec(spec.base_path)
    return base._load_formal_config(base_spec)


def _assert_ancestor(commit: str, current: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=base._repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("source evidence commit is not an ancestor of the pilot branch")


def _load_source_evidence(
    spec: PilotSpec,
    validation_path: str | Path,
    preflight_path: str | Path,
    current_commit: str,
) -> tuple[EvidenceRecordV2, EvidenceRecordV2, dict[str, str]]:
    validation_target = _resolve(validation_path)
    preflight_target = _resolve(preflight_path)
    expected_validation = _resolve(
        "outputs/20260827_v2/runs/"
        "lightgen_optical_manifold_identity_v1_attempt_02/validation.json"
    )
    expected_preflight = _resolve(spec.payload["source_preflight_evidence"])
    if validation_target != expected_validation or preflight_target != expected_preflight:
        raise ValueError("pilot must consume the registered first-round evidence paths")
    if file_sha256(validation_target) != SOURCE_VALIDATION_HASH:
        raise ValueError("first-round validation evidence hash mismatch")
    if file_sha256(preflight_target) != SOURCE_PREFLIGHT_HASH:
        raise ValueError("first-round preflight evidence hash mismatch")
    validation_payload = json.loads(validation_target.read_text(encoding="utf-8"))
    preflight_payload = json.loads(preflight_target.read_text(encoding="utf-8"))
    if validation_payload.get("schema") != SOURCE_VALIDATION_SCHEMA:
        raise ValueError("unexpected first-round validation evidence schema")
    if preflight_payload.get("schema") != SOURCE_PREFLIGHT_SCHEMA:
        raise ValueError("unexpected first-round preflight evidence schema")
    if (
        validation_payload.get("hypothesis") != HYPOTHESIS
        or preflight_payload.get("hypothesis") != HYPOTHESIS
        or validation_payload.get("training_authorized") is not False
        or preflight_payload.get("training_authorized") is not False
        or validation_payload.get("formal_promotion") is not False
        or preflight_payload.get("formal_promotion") is not False
    ):
        raise ValueError("first-round evidence has an invalid research boundary")
    validation = EvidenceRecordV2.from_dict(validation_payload["evidence"])
    preflight = EvidenceRecordV2.from_dict(preflight_payload["evidence"])
    validation.validate()
    preflight.validate()
    if (
        validation.evidence_kind != SOURCE_VALIDATION_SCHEMA
        or validation.status != "passed"
        or preflight.evidence_kind != SOURCE_PREFLIGHT_SCHEMA
        or preflight.status != "passed"
    ):
        raise ValueError("first-round evidence must be passed")
    if validation.config_hash != BASE_CONFIG_HASH or preflight.config_hash != BASE_CONFIG_HASH:
        raise ValueError("first-round evidence is bound to the wrong base config")
    _assert_ancestor(validation.git_commit, current_commit)
    _assert_ancestor(preflight.git_commit, current_commit)
    return validation, preflight, {
        "source_validation_evidence": file_sha256(validation_target),
        "source_preflight_evidence": file_sha256(preflight_target),
    }


def _stratified_split(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SPLIT_SEED)
    train_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    for label in range(10):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 500:
            raise ValueError("MNIST split has too few examples for a 500-per-class validation split")
        shuffled = indices[torch.randperm(indices.numel(), generator=generator)]
        train_parts.append(shuffled[:-500])
        validation_parts.append(shuffled[-500:])
    return torch.cat(train_parts), torch.cat(validation_parts)


def _schedule(train_indices: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    per_label = []
    for label in range(10):
        selected = train_indices[labels.index_select(0, train_indices) == label]
        if selected.numel() < MAX_STEPS:
            raise ValueError("stratified train split is incomplete")
        per_label.append(selected[:MAX_STEPS])
    return torch.stack(per_label, dim=1)


def _heldout(validation_indices: torch.Tensor, labels: torch.Tensor, count: int) -> torch.Tensor:
    selected: list[torch.Tensor] = []
    validation_labels = labels.index_select(0, validation_indices)
    for label in range(10):
        matches = validation_indices[validation_labels == label]
        if matches.numel() < count:
            raise ValueError("validation split does not contain the held-out panel")
        selected.append(matches[:count])
    return torch.cat(selected)


def _pilot_manifest(
    images: torch.Tensor,
    labels: torch.Tensor,
    dataset_root: str | Path,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    schedule: torch.Tensor,
    heldout_indices: torch.Tensor,
    preflight_heldout_indices: torch.Tensor,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": "lightgen_optical_manifold_identity_training_mnist_manifest.v1",
        "dataset_root": str(Path(dataset_root).resolve()),
        "train": True,
        "split_seed": SPLIT_SEED,
        "image_shape": list(images.shape),
        "label_shape": list(labels.shape),
        "images_sha256": base.tensor_sha256(images),
        "labels_sha256": base.tensor_sha256(labels),
        "train_indices_sha256": base.tensor_sha256(train_indices),
        "validation_indices_sha256": base.tensor_sha256(validation_indices),
        "schedule_sha256": base.tensor_sha256(schedule),
        "schedule_shape": list(schedule.shape),
        "heldout_indices_sha256": base.tensor_sha256(heldout_indices),
        "heldout_shape": list(heldout_indices.shape),
        "preflight_heldout_indices_sha256": base.tensor_sha256(preflight_heldout_indices),
        "preflight_heldout_shape": list(preflight_heldout_indices.shape),
        "train_count": int(train_indices.numel()),
        "validation_count": int(validation_indices.numel()),
        "heldout_per_label": HELDOUT_PER_LABEL,
        "schedule": "balanced_one_per_label_without_replacement",
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return manifest


def _load_pilot_data(dataset_root: str | Path) -> PilotData:
    images, labels = base.load_local_mnist(dataset_root, train=True)
    if tuple(images.shape) != (60000, 1, 28, 28) or tuple(labels.shape) != (60000,):
        raise ValueError("pilot requires the complete local MNIST train split")
    train_indices, validation_indices = _stratified_split(labels)
    schedule = _schedule(train_indices, labels)
    heldout_indices = _heldout(validation_indices, labels, HELDOUT_PER_LABEL)
    preflight_heldout_indices = _heldout(
        validation_indices, labels, PREFLIGHT_HELDOUT_PER_LABEL
    )
    manifest = _pilot_manifest(
        images,
        labels,
        dataset_root,
        train_indices,
        validation_indices,
        schedule,
        heldout_indices,
        preflight_heldout_indices,
    )
    return PilotData(
        images,
        labels,
        train_indices,
        validation_indices,
        schedule,
        heldout_indices,
        preflight_heldout_indices,
        manifest,
    )


def _context(
    spec: PilotSpec, formal: FormalSystemConfig, manifest: Mapping[str, Any]
) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=str(manifest["manifest_hash"]),
        calibration_hash=canonical_sha256(
            {"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}
        ),
        exposure_scale=1.0,
    )


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
        raise FileExistsError(f"refusing to overwrite existing evidence: {target}")
    atomic_write_json(target, payload)


def _validate_training_lineage(
    spec: PilotSpec,
    review: Mapping[str, Any],
    review_path: Path,
    source_hashes: Mapping[str, str],
    formal: FormalSystemConfig,
    data: PilotData,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    if review["source_preflight_sha256"] != SOURCE_PREFLIGHT_HASH:
        raise ValueError("review contract is not bound to the first-round preflight")
    parameter_model = base.IdentityOpticalManifoldAutoencoder(formal)
    parameter_contract = base._parameter_contract(parameter_model)
    if tuple(parameter_contract["trainable_parameter_names"]) != EXPECTED_TRAINABLE:
        raise ValueError("pilot trainable parameter contract failed")
    if parameter_contract["optimizer_parameter_names"] != list(EXPECTED_TRAINABLE):
        raise ValueError("pilot optimizer parameter contract failed")
    if data.schedule.shape != (MAX_STEPS, LOGICAL_BATCH_SIZE):
        raise ValueError("pilot schedule shape contract failed")
    if torch.unique(data.schedule, dim=0).shape[0] != MAX_STEPS:
        raise ValueError("pilot schedule contains repeated batches")
    schedule_labels = data.labels.index_select(0, data.schedule.reshape(-1)).reshape(
        MAX_STEPS, LOGICAL_BATCH_SIZE
    )
    if not torch.equal(schedule_labels, torch.arange(10).expand(MAX_STEPS, -1)):
        raise ValueError("pilot schedule is not one sample per label")
    if data.heldout_indices.numel() != 1000:
        raise ValueError("pilot held-out panel must contain 1000 images")
    if data.preflight_heldout_indices.numel() != 100:
        raise ValueError("pilot preflight held-out panel must contain 100 images")
    return {
        "branch": state["branch"],
        "commit": state["commit"],
        "formal_system_kind": base.EXPECTED_FORMAL_KIND,
        "system_geometry": dict(base.EXPECTED_SYSTEM),
        "input_encoding": "intensity_sqrt_amplitude",
        "target": "camera_intensity_same_fov",
        "loss": "mse",
        "exposure": {"scale": 1.0, "gamma": 1.0},
        "optimizer": {
            "name": "Adam",
            "learning_rate": 1.0e-3,
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "microbatch_size": MICROBATCH_SIZE,
        },
        "activation_checkpoint": True,
        "initialization": {"phase": "zero", "seed": SPLIT_SEED},
        "trainable_parameter_names": list(EXPECTED_TRAINABLE),
        "data_manifest": data.manifest,
        "schedule": {
            "steps": MAX_STEPS,
            "samples": MAX_STEPS * LOGICAL_BATCH_SIZE,
            "without_replacement_per_label": True,
            "schedule_sha256": data.manifest["schedule_sha256"],
        },
        "heldout_panel": {
            "per_label": HELDOUT_PER_LABEL,
            "count": int(data.heldout_indices.numel()),
            "selection": "first_occurrence_in_validation_split",
        },
        "source_evidence_hashes": dict(source_hashes),
        "review_contract_sha256": file_sha256(review_path),
        "authorization": {
            "bounded_research_training_authorized": True,
            "formal_training_authorized": False,
            "formal_promotion": False,
            "max_steps": MAX_STEPS,
            "prohibited": list(spec.payload["authorization"]["prohibited"]),
        },
    }


def _training_validation_record(
    state: Mapping[str, Any],
    spec: PilotSpec,
    formal: FormalSystemConfig,
    data: PilotData,
    input_hashes: Mapping[str, str],
) -> EvidenceRecordV2:
    gates = (
        GateResult("clean_lineage", True, {"clean": True}),
        GateResult("independent_pilot_branch", True, {"branch_matches": True}),
        GateResult("first_round_validation_and_preflight", True, {"passed": True}),
        GateResult("formal_geometry_and_identity_ols", True, {"grid": 2048, "identity": True}),
        GateResult("stratified_mnist_manifest", True, {"train": 55000, "validation": 5000}),
        GateResult("deterministic_balanced_schedule", True, {"steps": MAX_STEPS}),
        GateResult("heldout_panel", True, {"count": 1000, "per_label": 100}),
        GateResult("authorization_scope", True, {"bounded_steps": MAX_STEPS}),
        GateResult("trainable_parameter_set", True, {"count": len(EXPECTED_TRAINABLE)}),
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
        permitted_next_actions=(NEXT_PREFLIGHT_ACTION,),
    )
    record.validate()
    return record


def train_validate(args: argparse.Namespace) -> int:
    spec = _load_pilot_spec(args.config)
    review, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    _, _, source_hashes = _load_source_evidence(
        spec,
        args.source_validation_evidence,
        args.source_preflight_evidence,
        state["commit"],
    )
    data = _load_pilot_data(args.dataset_root)
    contract = _validate_training_lineage(
        spec, review, review_path, source_hashes, formal, data, state
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing training validation: {output}")
    input_hashes = {
        "pilot_config": file_sha256(spec.path),
        "base_config": file_sha256(spec.base_path),
        "formal_config": file_sha256(formal_path),
        "review_contract": file_sha256(review_path),
        **source_hashes,
        "mnist_manifest": data.manifest["manifest_hash"],
    }
    record = _training_validation_record(state, spec, formal, data, input_hashes)
    payload = _record_payload(
        VALIDATION_SCHEMA,
        record,
        contract,
        {"source_preflight_sha256": SOURCE_PREFLIGHT_HASH},
    )
    _write_new_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_training_validation(
    path: str | Path,
    spec: PilotSpec,
    data: PilotData,
    review_path: Path,
    state: Mapping[str, Any],
) -> tuple[EvidenceRecordV2, str]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != VALIDATION_SCHEMA:
        raise ValueError("training preflight requires training validation evidence")
    if (
        payload.get("hypothesis") != HYPOTHESIS
        or payload.get("bounded_research_training_authorized") is not True
        or payload.get("training_authorized") is not False
        or payload.get("formal_promotion") is not False
    ):
        raise ValueError("training validation evidence has an invalid authorization scope")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != VALIDATION_SCHEMA or record.status != "passed":
        raise ValueError("training validation evidence must be passed")
    if record.git_commit != state["commit"] or not state["clean"] or state["branch"] != BRANCH:
        raise ValueError("training validation evidence is stale or branch is not clean")
    if record.config_hash != spec.config_hash:
        raise ValueError("training validation config hash mismatch")
    if record.context.dataset_hash != data.manifest["manifest_hash"]:
        raise ValueError("training validation MNIST manifest mismatch")
    if record.input_hashes.get("review_contract") != file_sha256(review_path):
        raise ValueError("training validation review binding mismatch")
    if record.input_hashes.get("pilot_config") != file_sha256(spec.path):
        raise ValueError("training validation config binding mismatch")
    return record, file_sha256(target)


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
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and left.keys() == right.keys()
            and all(_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _rng_state() -> dict[str, Any]:
    return {
        "cpu": torch.get_rng_state().clone(),
        "cuda": [state.clone() for state in torch.cuda.get_rng_state_all()],
    }


def _restore_rng(state: Mapping[str, Any]) -> None:
    torch.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def _make_optimizer(model: base.IdentityOpticalManifoldAutoencoder) -> torch.optim.Adam:
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name in EXPECTED_TRAINABLE
    ]
    optimizer = torch.optim.Adam(parameters, lr=1.0e-3)
    if len(optimizer.param_groups) != 1:
        raise ValueError("pilot optimizer must contain one parameter group")
    names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    actual = tuple(
        names[id(parameter)]
        for parameter in optimizer.param_groups[0]["params"]
    )
    if actual != EXPECTED_TRAINABLE or optimizer.param_groups[0]["lr"] != 1.0e-3:
        raise ValueError("pilot optimizer parameter contract failed")
    return optimizer


def _step(
    model: base.IdentityOpticalManifoldAutoencoder,
    optimizer: torch.optim.Optimizer,
    batch: StepBatch,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    losses: list[torch.Tensor] = []
    camera_values: list[torch.Tensor] = []
    for start in range(0, LOGICAL_BATCH_SIZE, MICROBATCH_SIZE):
        output = model(
            batch.amplitude[start : start + MICROBATCH_SIZE],
            activation_checkpoint=True,
        )
        loss = F.mse_loss(
            output.reconstruction,
            batch.target[start : start + MICROBATCH_SIZE],
        )
        losses.append(loss.detach())
        camera_values.append(output.raw_camera.detach())
        (loss / LOGICAL_BATCH_SIZE).backward()
    gradient_norms: dict[str, float] = {}
    for name in EXPECTED_TRAINABLE:
        parameter = dict(model.named_parameters())[name]
        if parameter.grad is None:
            continue
        gradient_norms[name] = float(parameter.grad.detach().norm().cpu())
        for index in range(parameter.shape[0]):
            gradient_norms[f"{name}[{index}]"] = float(
                parameter.grad[index].detach().norm().cpu()
            )
    before = {
        name: dict(model.named_parameters())[name].detach().clone()
        for name in EXPECTED_TRAINABLE
    }
    optimizer.step()
    update_norms: dict[str, float] = {}
    for name in EXPECTED_TRAINABLE:
        parameter = dict(model.named_parameters())[name]
        update_norms[name] = float((parameter.detach() - before[name]).abs().sum().cpu())
        for index in range(parameter.shape[0]):
            update_norms[f"{name}[{index}]"] = float(
                (parameter[index].detach() - before[name][index]).abs().sum().cpu()
            )
    cameras = torch.cat(camera_values, dim=0)
    mean_loss = torch.stack(losses).mean()
    return {
        "loss": float(mean_loss.cpu()),
        "gradient_norms": gradient_norms,
        "update_norms": update_norms,
        "camera_finite": bool(torch.isfinite(cameras).all()),
        "camera_min": float(cameras.min().cpu()),
        "camera_max": float(cameras.max().cpu()),
    }


def _batch_for_indices(
    data: PilotData, indices: torch.Tensor, device: torch.device, *, target_roll: int = 0
) -> StepBatch:
    rendered = base.render_intensity_batch(data.images.index_select(0, indices))
    target = rendered.target
    if target_roll:
        target = target.roll(shifts=target_roll, dims=0)
    return StepBatch(rendered.amplitude.to(device), target.to(device))


def _state_on_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _frozen_unchanged(
    model: base.IdentityOpticalManifoldAutoencoder,
    initial_state: Mapping[str, torch.Tensor],
) -> bool:
    return all(
        torch.equal(model.state_dict()[name].detach().cpu(), initial_state[name])
        for name in model.state_dict()
        if name not in EXPECTED_TRAINABLE
    )


def _finite_step(step: Mapping[str, Any]) -> bool:
    values = [step["loss"], step["camera_min"], step["camera_max"]]
    values.extend(step["gradient_norms"].values())
    values.extend(step["update_norms"].values())
    return all(bool(torch.isfinite(torch.tensor(value))) for value in values) and step["camera_finite"]


def _resume_payload(
    model: base.IdentityOpticalManifoldAutoencoder,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    loss_history: list[float],
    rng: Mapping[str, Any],
    spec: PilotSpec,
    data: PilotData,
    evidence_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": RESUME_SCHEMA,
        "model_state": _state_on_cpu(model),
        "optimizer_state": _clone_tree(optimizer.state_dict()),
        "rng_state": _clone_tree(rng),
        "step": step,
        "loss_history": list(loss_history),
        "config_hash": spec.config_hash,
        "dataset_manifest_hash": data.manifest["manifest_hash"],
        "schedule_sha256": data.manifest["schedule_sha256"],
        "evidence_hashes": dict(evidence_hashes),
        "data_cursor": {
            "schedule": "balanced_one_per_label_without_replacement",
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "microbatch_size": MICROBATCH_SIZE,
            "next_step": step + 1,
        },
    }


def _load_resume(
    path: str | Path,
    model: base.IdentityOpticalManifoldAutoencoder,
    optimizer: torch.optim.Optimizer,
    *,
    spec: PilotSpec,
    data: PilotData,
    evidence_hashes: Mapping[str, str],
) -> dict[str, Any]:
    payload = safe_torch_load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != RESUME_SCHEMA:
        raise ValueError("invalid Identity reconstruction pilot resume checkpoint")
    if (
        payload.get("config_hash") != spec.config_hash
        or payload.get("dataset_manifest_hash") != data.manifest["manifest_hash"]
        or payload.get("schedule_sha256") != data.manifest["schedule_sha256"]
        or payload.get("evidence_hashes") != dict(evidence_hashes)
    ):
        raise ValueError("pilot resume checkpoint lineage mismatch")
    step = payload.get("step")
    if not isinstance(step, int) or step < 1 or step >= MAX_STEPS:
        raise ValueError("pilot resume checkpoint must identify a step in [1, 999]")
    cursor = payload.get("data_cursor")
    if cursor != {
        "schedule": "balanced_one_per_label_without_replacement",
        "logical_batch_size": LOGICAL_BATCH_SIZE,
        "microbatch_size": MICROBATCH_SIZE,
        "next_step": step + 1,
    }:
        raise ValueError("pilot resume checkpoint data cursor mismatch")
    if len(payload.get("loss_history", [])) != step:
        raise ValueError("pilot resume checkpoint loss history mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng(payload["rng_state"])
    return payload


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _set_deterministic() -> bool:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    return previous


def _evaluate(
    model: base.IdentityOpticalManifoldAutoencoder,
    data: PilotData,
    indices: torch.Tensor,
    device: torch.device,
    *,
    target_roll: int = 0,
) -> float:
    model.eval()
    values: list[torch.Tensor] = []
    with torch.no_grad():
        for index in indices:
            batch = _batch_for_indices(
                data, index.reshape(1), device, target_roll=target_roll
            )
            output = model(batch.amplitude, activation_checkpoint=False)
            values.append(F.mse_loss(output.reconstruction, batch.target).detach())
    model.train()
    return float(torch.stack(values).mean().cpu())


def _initial_model_and_optimizer(
    formal: FormalSystemConfig, device: torch.device
) -> tuple[base.IdentityOpticalManifoldAutoencoder, torch.optim.Adam]:
    torch.manual_seed(SPLIT_SEED)
    torch.cuda.manual_seed_all(SPLIT_SEED)
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    optimizer = _make_optimizer(model)
    return model, optimizer


def _preflight_record(
    state: Mapping[str, Any],
    spec: PilotSpec,
    formal: FormalSystemConfig,
    data: PilotData,
    input_hashes: Mapping[str, str],
    gates: tuple[GateResult, ...],
    status: str,
    next_action: tuple[str, ...],
    reason: str | None = None,
) -> EvidenceRecordV2:
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=PREFLIGHT_SCHEMA,
        tier="diagnostic",
        status=status,
        git_commit=str(state["commit"]),
        dirty_patch_hash=None,
        config_hash=spec.config_hash,
        context=_context(spec, formal, data.manifest),
        input_hashes=dict(input_hashes),
        gates=gates,
        permitted_next_actions=next_action,
        invalidation_reason=reason,
    )
    record.validate()
    return record


def train_preflight(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("Identity reconstruction pilot preflight requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    spec = _load_pilot_spec(args.config)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_pilot_data(args.dataset_root)
    validation, validation_hash = _load_training_validation(
        args.training_validation_evidence, spec, data, review_path, state
    )
    del validation
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing training preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    input_hashes = {
        "training_validation_evidence": validation_hash,
        "pilot_config": file_sha256(spec.path),
        "formal_config": file_sha256(formal_path),
        "review_contract": file_sha256(review_path),
        "mnist_manifest": data.manifest["manifest_hash"],
        "source_preflight_evidence": SOURCE_PREFLIGHT_HASH,
    }
    if not torch.cuda.is_available():
        record = _preflight_record(
            state,
            spec,
            formal,
            data,
            input_hashes,
            (GateResult("cuda_available", False, {}, "CUDA is unavailable"),),
            "invalid_harness",
            (),
            "CUDA is unavailable",
        )
        payload = _record_payload(PREFLIGHT_SCHEMA, record, {"training_validation_hash": validation_hash})
        _write_new_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        reason = f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}"
        record = _preflight_record(
            state,
            spec,
            formal,
            data,
            input_hashes,
            (GateResult("free_cuda_memory", False, {"bytes": int(free_bytes)}, reason),),
            "invalid_harness",
            (),
            reason,
        )
        payload = _record_payload(
            PREFLIGHT_SCHEMA,
            record,
            {
                "training_validation_hash": validation_hash,
                "free_cuda_bytes": int(free_bytes),
                "total_cuda_bytes": int(total_bytes),
            },
        )
        _write_new_json(output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    previous_deterministic = _set_deterministic()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model, optimizer = _initial_model_and_optimizer(formal, device)
    initial_state = _state_on_cpu(model)
    initial_optimizer = _clone_tree(optimizer.state_dict())
    initial_rng = _rng_state()
    paired_steps: list[dict[str, Any]] = []
    for step in range(3):
        batch = _batch_for_indices(data, data.schedule[step], device)
        paired_steps.append(_step(model, optimizer, batch))
    paired_state = _state_on_cpu(model)
    paired_optimizer = _clone_tree(optimizer.state_dict())
    paired_rng = _rng_state()

    interrupted, interrupted_optimizer = _initial_model_and_optimizer(formal, device)
    interrupted.load_state_dict(initial_state, strict=True)
    interrupted_optimizer.load_state_dict(initial_optimizer)
    _restore_rng(initial_rng)
    interrupted_history = []
    for step in range(2):
        interrupted_history.append(
            _step(
                interrupted,
                interrupted_optimizer,
                _batch_for_indices(data, data.schedule[step], device),
            )["loss"]
        )
    resume_path = output.with_name(output.stem + ".resume.pt")
    resume = _resume_payload(
        interrupted,
        interrupted_optimizer,
        step=2,
        loss_history=interrupted_history,
        rng=_rng_state(),
        spec=spec,
        data=data,
        evidence_hashes={"training_validation_evidence": validation_hash},
    )
    atomic_torch_save(resume_path, resume)
    del interrupted, interrupted_optimizer
    torch.cuda.empty_cache()
    resumed, resumed_optimizer = _initial_model_and_optimizer(formal, device)
    _load_resume(
        resume_path,
        resumed,
        resumed_optimizer,
        spec=spec,
        data=data,
        evidence_hashes={"training_validation_evidence": validation_hash},
    )
    resumed_step = _step(
        resumed,
        resumed_optimizer,
        _batch_for_indices(data, data.schedule[2], device),
    )
    resume_exact = (
        _tree_equal(paired_state, _state_on_cpu(resumed))
        and _tree_equal(paired_optimizer, resumed_optimizer.state_dict())
        and _tree_equal(paired_rng, _rng_state())
        and paired_steps[2] == resumed_step
    )

    control, control_optimizer = _initial_model_and_optimizer(formal, device)
    control_steps: list[dict[str, Any]] = []
    for step in range(3):
        control_steps.append(
            _step(
                control,
                control_optimizer,
                _batch_for_indices(data, data.schedule[step], device, target_roll=1),
            )
        )
    preflight_indices = data.preflight_heldout_indices
    baseline_model, _ = _initial_model_and_optimizer(formal, device)
    baseline_mse = _evaluate(baseline_model, data, preflight_indices, device)
    control_mse = _evaluate(control, data, preflight_indices, device)
    control_relative = (baseline_mse - control_mse) / baseline_mse
    elapsed = time.monotonic() - started
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    storage_bytes = _directory_size(output.parent)
    all_steps = paired_steps + [resumed_step] + control_steps
    finite = all(_finite_step(step) for step in all_steps)
    phase_names = tuple(
        f"{name}[{index}]"
        for name in EXPECTED_TRAINABLE
        for index in range(dict(model.named_parameters())[name].shape[0])
    )
    phase_gradients = all(
        step["gradient_norms"].get(name, 0.0) > 0.0
        for step in paired_steps
        for name in phase_names
    )
    phase_updates = all(
        step["update_norms"].get(name, 0.0) > 0.0
        for step in paired_steps
        for name in phase_names
    )
    frozen = _frozen_unchanged(model, initial_state)
    control_rejected = control_relative < 0.05
    gates = (
        GateResult("finite_values", finite, {"step_count": len(all_steps)}),
        GateResult("four_phase_layer_gradients_nonzero", phase_gradients, {"count": 4}),
        GateResult("four_phase_layer_updates_nonzero", phase_updates, {"count": 4}),
        GateResult("frozen_state_unchanged", frozen, {}),
        GateResult(
            "paired_loss_decreased",
            paired_steps[-1]["loss"] < paired_steps[0]["loss"],
            {
                "initial": paired_steps[0]["loss"],
                "step_3": paired_steps[-1]["loss"],
            },
        ),
        GateResult("fixed_exposure", True, {"scale": 1.0, "gamma": 1.0}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("resume_exact", resume_exact, {}),
        GateResult(
            "permuted_target_negative_control",
            control_rejected,
            {
                "baseline_mse": baseline_mse,
                "control_mse": control_mse,
                "control_relative_reduction": control_relative,
                "threshold": 0.05,
            },
            None if control_rejected else "negative control reached the scientific gate",
        ),
        GateResult("cuda_peak_memory", peak_bytes <= MAX_PEAK_BYTES, {"bytes": peak_bytes}),
        GateResult("wall_time", elapsed <= MAX_PREFLIGHT_SECONDS, {"seconds": elapsed}),
        GateResult("run_storage", storage_bytes <= MAX_STORAGE_BYTES, {"bytes": storage_bytes}),
    )
    passed = all(gate.passed for gate in gates)
    status = "passed" if passed else "invalid_harness"
    record = _preflight_record(
        state,
        spec,
        formal,
        data,
        {
            **input_hashes,
            "resume_artifact": file_sha256(resume_path),
        },
        gates,
        status,
        (NEXT_PREFLIGHT_ACTION,) if passed else (),
        None if passed else "one or more training preflight gates failed",
    )
    payload = _record_payload(
        PREFLIGHT_SCHEMA,
        record,
        {
            "training_validation_hash": validation_hash,
            "formal_system_kind": base.EXPECTED_FORMAL_KIND,
            "input_encoding": "intensity_sqrt_amplitude",
            "target": "camera_intensity_same_fov",
            "optimizer": {"name": "Adam", "learning_rate": 1.0e-3},
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "microbatch_size": MICROBATCH_SIZE,
            "activation_checkpoint": True,
            "trainable_parameter_names": list(EXPECTED_TRAINABLE),
            "phase_layer_names": list(phase_names),
            "resume_artifact": str(resume_path.resolve()),
            "paired_steps": paired_steps,
            "control_steps": control_steps,
            "resumed_step": resumed_step,
            "preflight_heldout": {
                "count": int(preflight_indices.numel()),
                "baseline_mse": baseline_mse,
                "control_mse": control_mse,
                "control_relative_reduction": control_relative,
            },
            "free_cuda_bytes_before": int(free_bytes),
            "total_cuda_bytes": int(total_bytes),
            "peak_cuda_bytes": peak_bytes,
            "wall_seconds": elapsed,
            "run_storage_bytes": storage_bytes,
        },
    )
    _write_new_json(output, payload)
    torch.use_deterministic_algorithms(previous_deterministic)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 2


def _load_training_preflight(
    path: str | Path,
    spec: PilotSpec,
    data: PilotData,
    validation_hash: str,
    review_path: Path,
    state: Mapping[str, Any],
) -> tuple[EvidenceRecordV2, str]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != PREFLIGHT_SCHEMA:
        raise ValueError("pilot requires training preflight evidence")
    if (
        payload.get("hypothesis") != HYPOTHESIS
        or payload.get("bounded_research_training_authorized") is not True
        or payload.get("training_authorized") is not False
        or payload.get("formal_promotion") is not False
    ):
        raise ValueError("training preflight has an invalid authorization scope")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != PREFLIGHT_SCHEMA or record.status != "passed":
        raise ValueError("training preflight evidence must be passed")
    if record.git_commit != state["commit"] or not state["clean"] or state["branch"] != BRANCH:
        raise ValueError("training preflight evidence is stale or branch is not clean")
    if record.config_hash != spec.config_hash:
        raise ValueError("training preflight config hash mismatch")
    if record.context.dataset_hash != data.manifest["manifest_hash"]:
        raise ValueError("training preflight MNIST manifest mismatch")
    if record.input_hashes.get("training_validation_evidence") != validation_hash:
        raise ValueError("training preflight validation binding mismatch")
    if record.input_hashes.get("review_contract") != file_sha256(review_path):
        raise ValueError("training preflight review binding mismatch")
    return record, file_sha256(target)


def _save_checkpoint(
    path: Path,
    model: base.IdentityOpticalManifoldAutoencoder,
    optimizer: torch.optim.Optimizer,
    step: int,
    history: list[float],
    spec: PilotSpec,
    data: PilotData,
    evidence_hashes: Mapping[str, str],
) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite pilot checkpoint: {path}")
    atomic_torch_save(
        path,
        _resume_payload(
            model,
            optimizer,
            step=step,
            loss_history=history,
            rng=_rng_state(),
            spec=spec,
            data=data,
            evidence_hashes=evidence_hashes,
        ),
    )


def _training_record(
    state: Mapping[str, Any],
    spec: PilotSpec,
    formal: FormalSystemConfig,
    data: PilotData,
    input_hashes: Mapping[str, str],
    gates: tuple[GateResult, ...],
    status: str,
    next_actions: tuple[str, ...],
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
        context=_context(spec, formal, data.manifest),
        input_hashes=dict(input_hashes),
        gates=gates,
        permitted_next_actions=next_actions,
        invalidation_reason=reason,
    )
    record.validate()
    return record


def train(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("Identity reconstruction pilot requires CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    spec = _load_pilot_spec(args.config)
    _, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_pilot_data(args.dataset_root)
    validation, validation_hash = _load_training_validation(
        args.training_validation_evidence, spec, data, review_path, state
    )
    del validation
    preflight, preflight_hash = _load_training_preflight(
        args.training_preflight_evidence,
        spec,
        data,
        validation_hash,
        review_path,
        state,
    )
    del preflight
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing pilot evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence_hashes = {
        "training_validation_evidence": validation_hash,
        "training_preflight_evidence": preflight_hash,
    }
    input_hashes = {
        "training_validation_evidence": validation_hash,
        "training_preflight_evidence": preflight_hash,
        "pilot_config": file_sha256(spec.path),
        "formal_config": file_sha256(formal_path),
        "review_contract": file_sha256(review_path),
        "mnist_manifest": data.manifest["manifest_hash"],
        "source_preflight_evidence": SOURCE_PREFLIGHT_HASH,
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA became unavailable after training preflight")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    previous_deterministic = _set_deterministic()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model, optimizer = _initial_model_and_optimizer(formal, device)
    initial_state = _state_on_cpu(model)
    baseline_mse = _evaluate(model, data, data.heldout_indices, device)
    history: list[float] = []
    start_step = 0
    if args.resume_checkpoint:
        resume = _load_resume(
            args.resume_checkpoint,
            model,
            optimizer,
            spec=spec,
            data=data,
            evidence_hashes=evidence_hashes,
        )
        history = list(resume["loss_history"])
        start_step = int(resume["step"])
    step_results: list[dict[str, Any]] = []
    resource_aborted = False
    for step_index in range(start_step, MAX_STEPS):
        if time.monotonic() - started > MAX_PILOT_SECONDS:
            resource_aborted = True
            break
        result = _step(
            model,
            optimizer,
            _batch_for_indices(data, data.schedule[step_index], device),
        )
        history.append(result["loss"])
        step_results.append(result)
        step_number = step_index + 1
        if step_number % 100 == 0:
            _save_checkpoint(
                output.parent / f"pilot_step_{step_number:06d}.pt",
                model,
                optimizer,
                step_number,
                history,
                spec,
                data,
                evidence_hashes,
            )
    completed = not resource_aborted and len(history) == MAX_STEPS
    final_checkpoint: Path | None = None
    if completed:
        final_checkpoint = output.parent / "pilot_final_step_001000.pt"
        _save_checkpoint(
            final_checkpoint,
            model,
            optimizer,
            MAX_STEPS,
            history,
            spec,
            data,
            evidence_hashes,
        )
    paired_mse = _evaluate(model, data, data.heldout_indices, device) if completed else None
    permuted_mse = (
        _evaluate(model, data, data.heldout_indices, device, target_roll=1)
        if completed
        else None
    )
    relative_reduction = (
        (baseline_mse - paired_mse) / baseline_mse if paired_mse is not None else None
    )
    elapsed = time.monotonic() - started
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    storage_bytes = _directory_size(output.parent)
    phase_names = tuple(
        f"{name}[{index}]"
        for name in EXPECTED_TRAINABLE
        for index in range(dict(model.named_parameters())[name].shape[0])
    )
    finite = all(_finite_step(result) for result in step_results)
    phase_gradients = all(
        result["gradient_norms"].get(name, 0.0) > 0.0
        for result in step_results
        for name in phase_names
    )
    phase_updates = all(
        result["update_norms"].get(name, 0.0) > 0.0
        for result in step_results
        for name in phase_names
    )
    frozen = _frozen_unchanged(model, initial_state)
    if resource_aborted:
        status = "resource_aborted"
        next_actions: tuple[str, ...] = ()
        reason = "pilot exceeded the 1800 second resource stop"
    elif not completed or not finite or not phase_gradients or not phase_updates or not frozen:
        status = "invalid_harness"
        next_actions = ()
        reason = "pilot execution or isolation gate failed"
    else:
        paired_gate = relative_reduction is not None and relative_reduction >= 0.05
        permutation_gate = paired_mse is not None and permuted_mse is not None and paired_mse < permuted_mse
        status = "passed" if paired_gate and permutation_gate else "valid_failed"
        next_actions = (NEXT_PASS_ACTION,) if status == "passed" else ()
        reason = None if status == "passed" else "pre-registered scientific gate did not pass"
    gates = (
        GateResult("completed_1000_steps", completed, {"steps": len(history)}),
        GateResult("finite_values", finite, {"measured_steps": len(step_results)}),
        GateResult("four_phase_layer_gradients_nonzero", phase_gradients, {"count": 4}),
        GateResult("four_phase_layer_updates_nonzero", phase_updates, {"count": 4}),
        GateResult("frozen_state_unchanged", frozen, {}),
        GateResult("fixed_exposure", True, {"scale": 1.0, "gamma": 1.0}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult(
            "heldout_relative_mse_reduction",
            relative_reduction is not None and relative_reduction >= 0.05,
            {
                "baseline_mse": baseline_mse,
                "paired_mse": paired_mse if paired_mse is not None else float("nan"),
                "relative_reduction": relative_reduction if relative_reduction is not None else float("nan"),
                "threshold": 0.05,
            },
        ),
        GateResult(
            "paired_below_permuted_target",
            paired_mse is not None and permuted_mse is not None and paired_mse < permuted_mse,
            {
                "paired_mse": paired_mse if paired_mse is not None else float("nan"),
                "permuted_mse": permuted_mse if permuted_mse is not None else float("nan"),
            },
        ),
        GateResult("cuda_peak_memory", peak_bytes <= MAX_PEAK_BYTES, {"bytes": peak_bytes}),
        GateResult("wall_time", elapsed <= MAX_PILOT_SECONDS, {"seconds": elapsed}),
        GateResult("run_storage", storage_bytes <= MAX_STORAGE_BYTES, {"bytes": storage_bytes}),
    )
    if status == "passed" and not all(gate.passed for gate in gates):
        status = "invalid_harness"
        next_actions = ()
        reason = "a harness or resource gate failed after the scientific measurement"
    record = _training_record(
        state,
        spec,
        formal,
        data,
        input_hashes,
        gates,
        status,
        next_actions,
        reason,
    )
    payload = _record_payload(
        TRAINING_SCHEMA,
        record,
        {
            "training_validation_hash": validation_hash,
            "training_preflight_hash": preflight_hash,
            "formal_system_kind": base.EXPECTED_FORMAL_KIND,
            "input_encoding": "intensity_sqrt_amplitude",
            "target": "camera_intensity_same_fov",
            "optimizer": {"name": "Adam", "learning_rate": 1.0e-3},
            "steps_completed": len(history),
            "loss_history": history,
            "heldout": {
                "count": int(data.heldout_indices.numel()),
                "baseline_mse": baseline_mse,
                "paired_mse": paired_mse,
                "permuted_target_mse": permuted_mse,
                "relative_reduction": relative_reduction,
            },
            "phase_layer_names": list(phase_names),
            "checkpoint_interval": 100,
            "final_checkpoint": str(final_checkpoint.resolve()) if final_checkpoint else None,
            "free_cuda_bytes_before": int(free_bytes),
            "total_cuda_bytes": int(total_bytes),
            "peak_cuda_bytes": peak_bytes,
            "wall_seconds": elapsed,
            "run_storage_bytes": storage_bytes,
            "evaluator_calls": 0,
        },
    )
    _write_new_json(output, payload)
    torch.use_deterministic_algorithms(previous_deterministic)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "passed" else 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "train-validate":
        return train_validate(args)
    if args.research_action == "train-preflight":
        return train_preflight(args)
    if args.research_action == "train":
        return train(args)
    raise ValueError(f"unsupported optical-manifold training action: {args.research_action}")
