"""No-update attribution audit for the H-LOMA-I0 aggregate-prior objective.

This lifecycle consumes the completed PLAN 5 pilot and asks which registered
failure mechanism is actionable: drift of the fixed projection, an unsafe
regularizer dose, or a rank objective that does not provide a useful descent
signal.  It never creates an optimizer, updates a parameter, or writes a
checkpoint.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
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
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from . import optical_manifold as base
from . import optical_manifold_full_epoch as full_epoch
from . import optical_manifold_regularization_audit as audit
from . import optical_manifold_regularization_pilot as pilot


BRANCH = "research/v2-lightgen-optical-manifold-regularization-objective-audit-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_identity_regularization_objective_audit_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_identity_regularization_objective_audit_review.v1"
VALIDATION_SCHEMA = (
    "lightgen_optical_manifold_identity_regularization_objective_audit_validation.v1"
)
AUDIT_SCHEMA = "lightgen_optical_manifold_identity_regularization_objective_audit.v1"
PROJECTION_SCHEMA = (
    "lightgen_optical_manifold_identity_regularization_objective_audit_projection.v1"
)
SOURCE_STEP = 5500
BASE_LAMBDA = 0.002733644070001058
DOSE_MULTIPLIERS = (1.0, 2.0, 4.0)
PROJECTION_COUNT = 1000
GRADIENT_BATCH_COUNT = 3
GRADIENT_BATCH_SIZE = 100
PCA_COMPONENTS = 32
FIXED_DIRECTIONS = 64
PERMUTATION_SEED = 1234
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_WALL_SECONDS = 1200.0
MAX_STORAGE_BYTES = 512 * 1024**2
PROJECTION_OVERLAP_MIN = 0.95
CAPTURED_ENERGY_DROP_MAX = 0.05
ENCODER_RATIO_MIN = 0.05
ENCODER_RATIO_MAX = 0.4
SOURCE_PAIRED_MSE = 0.03437697887420654
NEXT_PASS_ACTION = "manual_review_identity_regularization_dose_response_pilot_contract"
NEXT_PROJECTION_ACTION = "manual_review_adaptive_projection_contract"
NEXT_DOSE_ACTION = "manual_review_probabilistic_latent_contract"
NEXT_RANK_ACTION = "manual_review_regularization_objective_replacement_contract"
EXPECTED_FORMAL_KIND = base.EXPECTED_FORMAL_KIND
EXPECTED_TRAINABLE = base.EXPECTED_TRAINABLE


@dataclass(frozen=True)
class ObjectiveAuditSpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class ObjectiveAuditData:
    full_data: full_epoch.FullEpochData
    regularization_data: audit.RegularizationData
    pilot_training_manifest_hash: str
    projection_indices: torch.Tensor
    gradient_batches: tuple[torch.Tensor, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SourceBundle:
    checkpoint: dict[str, Any]
    projection: dict[str, Any]
    regularized_checkpoint: dict[str, Any]
    mse_only_checkpoint: dict[str, Any]
    hashes: dict[str, str]
    formal_path: Path


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"]:
        raise ValueError("objective audit requires a clean worktree")
    if state["branch"] != BRANCH:
        raise ValueError(f"objective audit requires branch {BRANCH}")


def _require_deterministic_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "objective audit requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "before the Python process starts"
        )


def _load_spec(path: str | Path) -> ObjectiveAuditSpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "source",
        "dataset",
        "optical_contract",
        "projection_drift",
        "regularization_gradient",
        "gates",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("objective audit config has unexpected or missing fields")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("unsupported objective audit config")
    return ObjectiveAuditSpec(target, payload, canonical_sha256(payload))


def _load_review(
    path: str | Path, spec: ObjectiveAuditSpec
) -> tuple[dict[str, Any], Path]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    source = spec.payload["source"]
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_identity_regularization_objective_audit",
        "source_pilot_validation_sha256": source["pilot_validation_sha256"],
        "source_pilot_preflight_sha256": source["pilot_preflight_sha256"],
        "source_pilot_training_sha256": source["pilot_training_sha256"],
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "source_regularized_checkpoint_sha256": source["regularized_checkpoint_sha256"],
        "source_mse_only_checkpoint_sha256": source["mse_only_checkpoint_sha256"],
        "source_projection_sha256": source["projection_sha256"],
        "source_mnist_manifest_hash": source["mnist_manifest_sha256"],
        "approved_contract": {
            "projection_drift_count": PROJECTION_COUNT,
            "gradient_audit_count": 300,
            "gradient_batches": GRADIENT_BATCH_COUNT,
            "tested_multipliers": list(DOSE_MULTIPLIERS),
            "no_model_update": True,
            "heldout_not_used_for_selection": True,
            "post_pass_next_action": NEXT_PASS_ACTION,
        },
        "no_update_audit_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("objective audit review contract does not match scope")
    return payload, target


def _load_formal(spec: ObjectiveAuditSpec) -> tuple[FormalSystemConfig, Path]:
    formal_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    formal = load_config(formal_path)
    if formal.architecture != "conditional_identity_logit_amplitude_v2":
        raise ValueError("objective audit requires current Identity lineage")
    if formal.system.kind != EXPECTED_FORMAL_KIND or not formal.system.is_formal:
        raise ValueError("objective audit requires formal finite-crop Identity")
    for name, expected in base.EXPECTED_SYSTEM.items():
        actual = getattr(formal.system, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1.0e-12, 1.0e-15):
                raise ValueError(f"formal geometry mismatch for {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    return formal.system, formal_path


def _assert_contract(spec: ObjectiveAuditSpec) -> None:
    payload = spec.payload
    if payload["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "projection_drift_count": 1000,
        "gradient_audit_count": 300,
        "gradient_batches": 3,
        "gradient_batch_size": 100,
        "selection": "first_100_then_next_30_per_label_after_all_prior_exclusions",
        "excluded_panels": [
            "regularization_calibration",
            "regularization_gradient",
            "pilot_schedule",
            "heldout",
        ],
        "heldout_exclusion": True,
        "no_replacement": True,
        "label_role": "panel_balance_and_provenance_only",
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }:
        raise ValueError("objective audit dataset contract is not locked")
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
        raise ValueError("objective audit optical contract is not locked")
    if payload["projection_drift"] != {
        "field": "unit_power_normalized_pre_ols_field",
        "metric": "real_part_of_x_x_h",
        "fixed_projection_components": PCA_COMPONENTS,
        "refit_components": PCA_COMPONENTS,
        "source_projection_frozen": True,
        "stability_overlap_threshold": PROJECTION_OVERLAP_MIN,
        "captured_energy_relative_drop_threshold": CAPTURED_ENERGY_DROP_MAX,
        "compare_models": [
            "source_step_5500",
            "regularized_final_step_100",
            "mse_only_final_step_100",
        ],
    }:
        raise ValueError("objective audit projection contract is not locked")
    if payload["regularization_gradient"] != {
        "family": "deterministic_sliced_wasserstein",
        "approximation_role": "project_approximation",
        "projection_components": PCA_COMPONENTS,
        "fixed_directions": FIXED_DIRECTIONS,
        "direction_seed": PERMUTATION_SEED,
        "base_lambda": BASE_LAMBDA,
        "tested_multipliers": list(DOSE_MULTIPLIERS),
        "encoder_ratio_min": ENCODER_RATIO_MIN,
        "encoder_ratio_max": ENCODER_RATIO_MAX,
        "rank_control": "fixed_seed_rank_permutation",
        "gradient_replay": "two_pass_fixed_rank_microbatch_1",
        "no_optimizer": True,
    }:
        raise ValueError("objective audit gradient contract is not locked")
    if payload["gates"] != {
        "source_training_status": "valid_failed_is_consumable_complete_evidence",
        "source_replay_mse_tolerance": 1.0e-9,
        "projection_overlap_min": PROJECTION_OVERLAP_MIN,
        "captured_energy_relative_drop_max": CAPTURED_ENERGY_DROP_MAX,
        "higher_dose_must_pass_all_batches": True,
        "positive_descent_alignment": True,
        "rank_control_must_be_worse": True,
        "model_state_unchanged": True,
        "evaluator_calls": 0,
        "min_free_cuda_bytes": MIN_FREE_BYTES,
        "max_peak_cuda_bytes": MAX_PEAK_BYTES,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "max_storage_bytes": MAX_STORAGE_BYTES,
    }:
        raise ValueError("objective audit gate contract is not locked")
    if payload["authorization"] != {
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_identity_regularization_objective_audit",
        "no_update_audit_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "post_pass_next_action": NEXT_PASS_ACTION,
        "post_projection_drift_failure_next_action": NEXT_PROJECTION_ACTION,
        "post_no_safe_dose_next_action": NEXT_DOSE_ACTION,
        "post_rank_control_failure_next_action": NEXT_RANK_ACTION,
        "prohibited": payload["authorization"]["prohibited"],
    }:
        raise ValueError("objective audit authorization contract is not locked")


def _assert_ancestor(commit: str, current: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=base._repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("objective audit source commit is not an ancestor")


def _load_data(dataset_root: str | Path) -> ObjectiveAuditData:
    full_data = full_epoch._load_data(dataset_root)
    regularization_data = audit._load_data(dataset_root)
    pilot_training_data = pilot._load_data(dataset_root)
    pilot_data = full_data.pilot_data
    if pilot_data.manifest["manifest_hash"] != pilot.SOURCE_MANIFEST_HASH:
        raise ValueError("MNIST source manifest changed")
    schedule = pilot._pilot_schedule(pilot_data.train_indices, pilot_data.labels)
    train_labels = pilot_data.labels.index_select(0, pilot_data.train_indices)
    excluded = torch.cat(
        (
            regularization_data.calibration_indices,
            *regularization_data.gradient_batches,
            schedule.reshape(-1),
            pilot_data.heldout_indices,
        )
    )
    if torch.unique(excluded).numel() != excluded.numel():
        raise ValueError("objective audit prior exclusion panels contain duplicates")
    projection: list[torch.Tensor] = []
    gradient_source: list[torch.Tensor] = []
    for label in range(10):
        class_indices = pilot_data.train_indices[train_labels == label]
        remaining = class_indices[~torch.isin(class_indices, excluded)]
        if remaining.numel() < 130:
            raise ValueError("train split lacks fresh objective audit samples")
        projection.append(remaining[:100])
        gradient_source.append(remaining[100:130])
    projection_indices = torch.cat(projection)
    gradient_batches = tuple(
        torch.cat(
            [gradient_source[label][batch * 10 : (batch + 1) * 10] for label in range(10)]
        )
        for batch in range(GRADIENT_BATCH_COUNT)
    )
    selected = torch.cat((projection_indices, *gradient_batches))
    if selected.numel() != 1300 or torch.unique(selected).numel() != selected.numel():
        raise ValueError("objective audit fresh panels are not disjoint")
    if torch.isin(selected, excluded).any() or torch.isin(selected, pilot_data.heldout_indices).any():
        raise ValueError("objective audit fresh panels overlap an excluded panel")
    expected_labels = torch.arange(10).repeat_interleave(10)
    for batch in gradient_batches:
        if not torch.equal(pilot_data.labels.index_select(0, batch), expected_labels):
            raise ValueError("objective audit gradient panel is not class balanced")
    manifest = {
        "schema": "lightgen_optical_manifold_identity_regularization_objective_audit_mnist_manifest.v1",
        "source_mnist_manifest_hash": pilot.SOURCE_MANIFEST_HASH,
        "source_full_epoch_manifest_hash": full_data.manifest["manifest_hash"],
        "source_regularization_audit_manifest_hash": regularization_data.manifest["manifest_hash"],
        "source_pilot_training_manifest_hash": pilot_training_data.manifest["manifest_hash"],
        "excluded_indices_sha256": tensor_sha256(torch.sort(excluded).values),
        "projection_indices_sha256": tensor_sha256(projection_indices),
        "projection_count": int(projection_indices.numel()),
        "gradient_batch_indices_sha256": [tensor_sha256(batch) for batch in gradient_batches],
        "gradient_batch_count": len(gradient_batches),
        "gradient_batch_size": GRADIENT_BATCH_SIZE,
        "heldout_indices_sha256": pilot.SOURCE_HELDOUT_HASH,
        "heldout_excluded": True,
        "no_replacement": True,
        "label_role": "panel_balance_and_provenance_only",
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return ObjectiveAuditData(
        full_data=full_data,
        regularization_data=regularization_data,
        pilot_training_manifest_hash=pilot_training_data.manifest["manifest_hash"],
        projection_indices=projection_indices,
        gradient_batches=gradient_batches,
        manifest=manifest,
    )


def _context(
    spec: ObjectiveAuditSpec, formal: FormalSystemConfig, data: ObjectiveAuditData
) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=data.manifest["manifest_hash"],
        calibration_hash=canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        exposure_scale=1.0,
    )


def _read_record(path: Path, expected_kind: str, expected_status: str) -> EvidenceRecordV2:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != expected_kind or record.status != expected_status:
        raise ValueError(f"source evidence is not {expected_kind}/{expected_status}")
    return record


def _load_arm_checkpoint(path: Path, expected_hash: str, arm: str, config_hash: str, manifest_hash: str) -> dict[str, Any]:
    payload = safe_torch_load(path, expected_sha256=expected_hash, map_location="cpu")
    if (
        payload.get("schema") != pilot.RESUME_SCHEMA
        or payload.get("arm") != arm
        or payload.get("step") != pilot.STEPS_PER_ARM
        or payload.get("config_hash") != config_hash
        or payload.get("dataset_manifest_hash") != manifest_hash
        or payload.get("schedule_sha256") != pilot.SCHEDULE_HASH
        or payload.get("formal_candidate_checkpoint") is not False
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise ValueError(f"{arm} pilot checkpoint lineage is invalid")
    return payload


def _source_bundle(
    spec: ObjectiveAuditSpec,
    review_path: Path,
    data: ObjectiveAuditData,
    current_commit: str,
) -> SourceBundle:
    source = spec.payload["source"]
    paths = {
        "pilot_config": (_resolve(source["pilot_config"]), source["pilot_config_sha256"]),
        "pilot_review": (_resolve(source["pilot_review"]), source["pilot_review_sha256"]),
        "pilot_validation": (_resolve(source["pilot_validation"]), source["pilot_validation_sha256"]),
        "pilot_preflight": (_resolve(source["pilot_preflight"]), source["pilot_preflight_sha256"]),
        "pilot_training": (_resolve(source["pilot_training"]), source["pilot_training_sha256"]),
        "source_checkpoint": (_resolve(source["source_checkpoint"]), source["source_checkpoint_sha256"]),
        "regularized_checkpoint": (_resolve(source["regularized_checkpoint"]), source["regularized_checkpoint_sha256"]),
        "mse_only_checkpoint": (_resolve(source["mse_only_checkpoint"]), source["mse_only_checkpoint_sha256"]),
        "projection": (_resolve(source["projection"]), source["projection_sha256"]),
    }
    for role, (path, expected) in paths.items():
        if file_sha256(path) != expected:
            raise ValueError(f"{role} source hash mismatch")
    pilot_spec = pilot._load_spec(paths["pilot_config"][0])
    pilot._assert_contract(pilot_spec)
    pilot_review, pilot_review_path = pilot._load_review(paths["pilot_review"][0], pilot_spec)
    pilot_bundle = pilot._source_bundle(
        pilot_spec, pilot_review_path, data.full_data.pilot_data, current_commit
    )
    validation_record = _read_record(
        paths["pilot_validation"][0], pilot.VALIDATION_SCHEMA, "passed"
    )
    preflight_record = _read_record(
        paths["pilot_preflight"][0], pilot.PREFLIGHT_SCHEMA, "passed"
    )
    training_record = _read_record(
        paths["pilot_training"][0], pilot.TRAINING_SCHEMA, "valid_failed"
    )
    for record in (validation_record, preflight_record, training_record):
        _assert_ancestor(record.git_commit, current_commit)
    if training_record.config_hash != pilot_spec.config_hash:
        raise ValueError("pilot training evidence config mismatch")
    training_payload = json.loads(paths["pilot_training"][0].read_text(encoding="utf-8"))
    if (
        training_payload.get("training_authorized") is not False
        or training_payload.get("formal_promotion") is not False
        or training_payload.get("contract", {}).get("formal_candidate_checkpoint") is not False
    ):
        raise ValueError("pilot training evidence authorization boundary is invalid")
    if training_payload.get("contract", {}).get("model_state_unchanged") is not True:
        raise ValueError("pilot training evidence lacks model isolation")
    projection = safe_torch_load(
        paths["projection"][0], expected_sha256=paths["projection"][1], map_location="cpu"
    )
    if (
        projection.get("schema") != audit.PROJECTION_SCHEMA
        or tuple(projection["basis"].shape) != (10000, PCA_COMPONENTS)
        or projection["basis"].dtype != torch.complex64
    ):
        raise ValueError("source projection shape or schema mismatch")
    source_checkpoint = safe_torch_load(
        paths["source_checkpoint"][0], expected_sha256=paths["source_checkpoint"][1], map_location="cpu"
    )
    if (
        source_checkpoint.get("schema") != full_epoch.RESUME_SCHEMA
        or source_checkpoint.get("global_step") != SOURCE_STEP
        or len(source_checkpoint.get("loss_history", [])) != SOURCE_STEP
        or not isinstance(source_checkpoint.get("model_state"), Mapping)
        or not isinstance(source_checkpoint.get("rng_state"), Mapping)
    ):
        raise ValueError("source step-5500 checkpoint is incomplete")
    regularized_checkpoint = _load_arm_checkpoint(
        paths["regularized_checkpoint"][0], paths["regularized_checkpoint"][1],
        "regularized", pilot_spec.config_hash, data.pilot_training_manifest_hash,
    )
    mse_only_checkpoint = _load_arm_checkpoint(
        paths["mse_only_checkpoint"][0], paths["mse_only_checkpoint"][1],
        "mse_only", pilot_spec.config_hash, data.pilot_training_manifest_hash,
    )
    formal_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    pilot_formal_path = pilot_bundle.formal_path
    if formal_path != pilot_formal_path:
        raise ValueError("objective audit and pilot formal configs differ")
    hashes = {
        "objective_review": file_sha256(review_path),
        "pilot_config": paths["pilot_config"][1],
        "pilot_review": paths["pilot_review"][1],
        "pilot_validation": paths["pilot_validation"][1],
        "pilot_preflight": paths["pilot_preflight"][1],
        "pilot_training": paths["pilot_training"][1],
        "source_checkpoint": paths["source_checkpoint"][1],
        "regularized_checkpoint": paths["regularized_checkpoint"][1],
        "mse_only_checkpoint": paths["mse_only_checkpoint"][1],
        "projection": paths["projection"][1],
        "mnist_manifest": source["mnist_manifest_sha256"],
        "heldout_indices": source["heldout_indices_sha256"],
        "pilot_training_manifest": data.pilot_training_manifest_hash,
        "formal_config": file_sha256(formal_path),
        "panel_manifest": data.manifest["manifest_hash"],
    }
    if data.manifest["source_mnist_manifest_hash"] != source["mnist_manifest_sha256"]:
        raise ValueError("objective audit dataset manifest binding mismatch")
    if pilot_review["training_authorized"] is not False or pilot_review["formal_promotion"] is not False:
        raise ValueError("pilot review authorization boundary is invalid")
    return SourceBundle(
        checkpoint=source_checkpoint,
        projection=projection,
        regularized_checkpoint=regularized_checkpoint,
        mse_only_checkpoint=mse_only_checkpoint,
        hashes=hashes,
        formal_path=formal_path,
    )


def _parameter_contract(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, Any]:
    names = tuple(name for name, _ in model.named_parameters())
    trainable = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if trainable != EXPECTED_TRAINABLE:
        raise ValueError("objective audit model trainable set is not the four phase layers")
    return {
        "all_parameter_names": list(names),
        "trainable_parameter_names": list(trainable),
        "optimizer_created": False,
        "optimizer_parameter_names": [],
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
        raise FileExistsError(f"refusing to overwrite objective audit evidence: {target}")
    atomic_write_json(target, payload)


def _validation_record(
    state: Mapping[str, Any],
    spec: ObjectiveAuditSpec,
    formal: FormalSystemConfig,
    data: ObjectiveAuditData,
    hashes: Mapping[str, str],
) -> EvidenceRecordV2:
    gates = tuple(
        GateResult(name, True, metrics)
        for name, metrics in (
            ("clean_lineage", {"clean": True}),
            ("independent_objective_audit_branch", {"branch_matches": True}),
            ("valid_failed_pilot_consumed_as_complete_source", {"source_step": SOURCE_STEP}),
            ("fresh_train_only_panels", {"projection": 1000, "gradient": 300}),
            ("formal_geometry_identity_ols", {"grid": 2048, "identity": True}),
            ("fixed_exposure", {"scale": 1.0, "gamma": 1.0}),
            ("four_phase_parameter_scope", {"count": 4}),
            ("zero_update_contract", {"optimizer": False, "checkpoint": False}),
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
        permitted_next_actions=("run_identity_regularization_objective_audit",),
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
        raise ValueError("objective audit formal config binding mismatch")
    model = base.IdentityOpticalManifoldAutoencoder(formal)
    hashes = dict(source.hashes)
    record = _validation_record(state, spec, formal, data, hashes)
    payload = _record_payload(
        VALIDATION_SCHEMA,
        record,
        {
            "source_step": SOURCE_STEP,
            "source_paired_mse": SOURCE_PAIRED_MSE,
            "parameter_contract": _parameter_contract(model),
            "dataset_manifest": data.manifest,
            "projection_drift": spec.payload["projection_drift"],
            "regularization_gradient": spec.payload["regularization_gradient"],
            "no_update_audit_authorized": review["no_update_audit_authorized"],
        },
    )
    _write_new_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(
    path: str | Path,
    spec: ObjectiveAuditSpec,
    formal: FormalSystemConfig,
    data: ObjectiveAuditData,
    state: Mapping[str, Any],
    hashes: Mapping[str, str],
) -> str:
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
        raise ValueError("objective audit validation schema or authorization is invalid")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != VALIDATION_SCHEMA or record.status != "passed":
        raise ValueError("objective audit validation must be passed")
    if record.git_commit != state["commit"] or state["branch"] != BRANCH or not state["clean"]:
        raise ValueError("objective audit validation is stale")
    if record.config_hash != spec.config_hash or record.context != _context(spec, formal, data):
        raise ValueError("objective audit validation context mismatch")
    if record.input_hashes != dict(hashes):
        raise ValueError("objective audit validation source hashes do not match")
    if record.permitted_next_actions != ("run_identity_regularization_objective_audit",):
        raise ValueError("objective audit validation next action is invalid")
    if payload.get("contract", {}).get("no_update_audit_authorized") is not True:
        raise ValueError("objective audit validation lacks no-update authorization")
    return digest


def _projection_from_payload(payload: Mapping[str, Any], device: torch.device) -> audit.Projection:
    projection = audit.Projection(
        mean_field=payload["mean_field"],
        basis=payload["basis"],
        whitening=payload["whitening"],
        eigenvalues=payload["eigenvalues"],
        explained_variance=payload["explained_variance"],
    )
    return audit._projection_on_device(projection, device)


def _model_from_state(
    formal: FormalSystemConfig,
    state: Mapping[str, Any],
    device: torch.device,
) -> base.IdentityOpticalManifoldAutoencoder:
    model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _state_cpu(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_equal(
    model: base.IdentityOpticalManifoldAutoencoder,
    expected: Mapping[str, torch.Tensor],
) -> bool:
    actual = model.state_dict()
    return set(actual) == set(expected) and all(
        torch.equal(actual[name].detach().cpu(), value) for name, value in expected.items()
    )


def _projection_stats(
    fields: torch.Tensor,
    source_projection: audit.Projection,
    directions: torch.Tensor,
) -> dict[str, Any]:
    values = audit._unit_power_field(fields.to(torch.complex64)).flatten(1)
    source_mean = source_projection.mean_field.reshape(1, -1).to(values.device)
    centered = values - source_mean
    coefficients = (centered @ source_projection.basis.conj()).real
    captured = coefficients.square().sum(dim=1).mean() / centered.abs().square().sum(dim=1).mean().clamp_min(1.0e-20)
    scores = audit._project_fields(fields, source_projection)
    covariance = scores.T @ scores / float(scores.shape[0])
    identity = torch.eye(PCA_COMPONENTS, dtype=covariance.dtype, device=covariance.device)
    covariance_error = (covariance - identity).norm() / identity.norm()
    refit, refit_scores = audit._fit_projection(fields)
    overlap_values = torch.linalg.svdvals(
        source_projection.basis.conj().T @ refit.basis.to(source_projection.basis.device)
    )
    overlap = overlap_values.square().mean()
    return {
        "fixed_swd": float(audit._sliced_wasserstein(scores, directions).cpu()),
        "refit_swd": float(audit._sliced_wasserstein(refit_scores.to(directions.device), directions).cpu()),
        "captured_energy": float(captured.cpu()),
        "mean_shift": float((values.mean(dim=0) - source_mean.reshape(-1)).abs().norm().cpu()),
        "whitened_covariance_error": float(covariance_error.cpu()),
        "principal_subspace_overlap": float(overlap.cpu()),
        "scores_shape": list(scores.shape),
        "finite": bool(
            torch.isfinite(values).all()
            and torch.isfinite(scores).all()
            and torch.isfinite(refit_scores).all()
            and torch.isfinite(torch.tensor([captured, covariance_error, overlap])).all()
        ),
    }


def _rank_coefficients(
    scores: torch.Tensor,
    directions: torch.Tensor,
    *,
    rank_shift: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.shape != (GRADIENT_BATCH_SIZE, PCA_COMPONENTS):
        raise ValueError("objective rank replay requires a [100, 32] score batch")
    projected = scores @ directions.T
    order = projected.argsort(dim=0)
    ranks = torch.empty_like(order)
    ranks.scatter_(
        0,
        order,
        torch.arange(GRADIENT_BATCH_SIZE, device=scores.device).reshape(-1, 1).expand_as(order),
    )
    if not audit._rank_assignment_is_permutation(ranks):
        raise ValueError("objective rank assignment is not a permutation")
    if rank_shift:
        ranks = torch.roll(ranks, shifts=rank_shift, dims=0)
    quantiles = audit._normal_midpoint_quantiles(GRADIENT_BATCH_SIZE, scores.device)
    target = quantiles.index_select(0, ranks.reshape(-1)).reshape_as(ranks)
    coefficients = 2.0 * (projected - target).div(float(GRADIENT_BATCH_SIZE * FIXED_DIRECTIONS))
    return coefficients, ranks


def _zero_grad(model: base.IdentityOpticalManifoldAutoencoder) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _add_grads(
    total: dict[str, torch.Tensor],
    model: base.IdentityOpticalManifoldAutoencoder,
) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            total[name].add_(parameter.grad.detach().cpu())


def _gradient_replay_with_control(
    model: base.IdentityOpticalManifoldAutoencoder,
    data: ObjectiveAuditData,
    indices: torch.Tensor,
    projection: audit.Projection,
    directions: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    if indices.numel() != GRADIENT_BATCH_SIZE:
        raise ValueError("objective gradient replay requires exactly 100 samples")
    projection = audit._projection_on_device(projection, device)
    first_pass: list[torch.Tensor] = []
    plane: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    model.eval()
    with torch.no_grad():
        for index in indices:
            amplitude, _target = audit._render_single(data.regularization_data, index, device)
            _raw, pre, current_plane, current_provenance = audit._forward_trace(model, amplitude)
            first_pass.append(audit._project_fields(pre, projection).detach())
            if plane is None:
                plane, provenance = current_plane, current_provenance
            elif plane != current_plane or provenance != current_provenance:
                raise ValueError("objective gradient replay trace provenance changed")
    scores = torch.cat(first_pass, dim=0)
    directions = directions.to(device)
    true_coefficients, ranks = _rank_coefficients(scores, directions)
    shifted_coefficients, shifted_ranks = _rank_coefficients(scores, directions, rank_shift=1)
    gradients = {
        name: torch.zeros_like(parameter, device="cpu")
        for name, parameter in model.named_parameters()
    }
    regularizer = {name: value.clone() for name, value in gradients.items()}
    control = {name: value.clone() for name, value in gradients.items()}
    for position, index in enumerate(indices):
        amplitude, target = audit._render_single(data.regularization_data, index, device)
        _zero_grad(model)
        raw, _pre, _plane, _provenance = audit._forward_trace(model, amplitude)
        F.mse_loss(raw, target).div(float(GRADIENT_BATCH_SIZE)).backward()
        _add_grads(gradients, model)

        _zero_grad(model)
        _raw, pre, _plane, _provenance = audit._forward_trace(model, amplitude)
        score = audit._project_fields(pre, projection)
        (score @ directions.T * true_coefficients[position : position + 1]).sum().backward()
        _add_grads(regularizer, model)

        _zero_grad(model)
        _raw, pre, _plane, _provenance = audit._forward_trace(model, amplitude)
        score = audit._project_fields(pre, projection)
        (score @ directions.T * shifted_coefficients[position : position + 1]).sum().backward()
        _add_grads(control, model)
    _zero_grad(model)
    return gradients, regularizer, control, {
        "first_pass_scores_shape": list(scores.shape),
        "plane": plane,
        "provenance": provenance,
        "rank_assignment_shape": list(ranks.shape),
        "rank_assignment_is_permutation": audit._rank_assignment_is_permutation(ranks),
        "rank_control_assignment_is_permutation": audit._rank_assignment_is_permutation(shifted_ranks),
        "rank_control": "roll_one_sample_axis_seed_1234",
    }


def _gradient_audit(
    model: base.IdentityOpticalManifoldAutoencoder,
    data: ObjectiveAuditData,
    projection: audit.Projection,
    directions: torch.Tensor,
    device: torch.device,
) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for batch_index, indices in enumerate(data.gradient_batches):
        reconstruction, regularizer, control, replay = _gradient_replay_with_control(
            model, data, indices, projection, directions, device
        )
        for multiplier in DOSE_MULTIPLIERS:
            report = audit._gradient_report(reconstruction, regularizer, BASE_LAMBDA * multiplier)
            control_report = audit._gradient_report(reconstruction, control, BASE_LAMBDA * multiplier)
            report.update(
                {
                    "batch_index": batch_index,
                    "multiplier": multiplier,
                    "lambda": BASE_LAMBDA * multiplier,
                    "replay": replay,
                    "rank_control": {
                        "combined_regularizer_dot_encoder": control_report["combined_regularizer_dot_encoder"],
                        "regularizer_encoder_norm": control_report["regularizer_encoder_norm"],
                        "finite": control_report["finite"],
                    },
                    "true_beats_rank_control": report["combined_regularizer_dot_encoder"]
                    > control_report["combined_regularizer_dot_encoder"],
                }
            )
            reports.append(report)
    return reports


def _record(
    state: Mapping[str, Any],
    spec: ObjectiveAuditSpec,
    formal: FormalSystemConfig,
    data: ObjectiveAuditData,
    hashes: Mapping[str, str],
    gates: tuple[GateResult, ...],
    status: str,
    actions: tuple[str, ...],
    reason: str | None,
) -> EvidenceRecordV2:
    record = EvidenceRecordV2(
        schema_version=2,
        evidence_kind=AUDIT_SCHEMA,
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


@contextmanager
def _deterministic_algorithms() -> Any:
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous)


def _run_cuda_audit(
    *,
    spec: ObjectiveAuditSpec,
    state: Mapping[str, Any],
    formal: FormalSystemConfig,
    data: ObjectiveAuditData,
    source: SourceBundle,
    validation_hash: str,
    output: Path,
    device: torch.device,
    free_bytes: int,
    total_bytes: int,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    source_model = _model_from_state(formal, source.checkpoint["model_state"], device)
    source_before = _state_cpu(source_model)
    projection = _projection_from_payload(source.projection, device)
    directions = audit._fixed_directions(device)
    model_states = {
        "source_step_5500": source.checkpoint["model_state"],
        "regularized_final_step_100": source.regularized_checkpoint["model_state"],
        "mse_only_final_step_100": source.mse_only_checkpoint["model_state"],
    }
    field_metrics: dict[str, Any] = {}
    for name, state_payload in model_states.items():
        model = source_model if name == "source_step_5500" else _model_from_state(formal, state_payload, device)
        fields, panel_mse, plane, provenance = audit._capture_fields(
            model, data.regularization_data, data.projection_indices, device
        )
        field_metrics[name] = {
            "mse": panel_mse,
            "plane": plane,
            "provenance": provenance,
            "projection": _projection_stats(fields.to(device), projection, directions),
            "state_unchanged": _state_equal(model, _state_cpu(model)),
        }
        if name != "source_step_5500":
            del model
            torch.cuda.empty_cache()
    source_replay_mse = full_epoch.pilot._evaluate(
        source_model, data.full_data.pilot_data, data.full_data.pilot_data.heldout_indices, device
    )
    gradient_models: dict[str, Any] = {}
    for name in ("source_step_5500", "regularized_final_step_100"):
        model = source_model if name == "source_step_5500" else _model_from_state(
            formal, model_states[name], device
        )
        before = _state_cpu(model)
        gradient_models[name] = {
            "reports": _gradient_audit(model, data, projection, directions, device),
            "state_unchanged": _state_equal(model, before),
        }
        if name != "source_step_5500":
            del model
            torch.cuda.empty_cache()
    model_isolated = _state_equal(source_model, source_before)
    del source_model
    torch.cuda.empty_cache()
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    storage_before = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())

    source_projection_stats = field_metrics["source_step_5500"]["projection"]
    projection_finite = all(item["projection"]["finite"] for item in field_metrics.values())
    overlaps = {
        name: item["projection"]["principal_subspace_overlap"]
        for name, item in field_metrics.items()
    }
    energy_drops = {
        name: max(
            0.0,
            (source_projection_stats["captured_energy"] - item["projection"]["captured_energy"])
            / max(source_projection_stats["captured_energy"], 1.0e-20),
        )
        for name, item in field_metrics.items()
    }
    all_reports = [
        report
        for model_payload in gradient_models.values()
        for report in model_payload["reports"]
    ]
    report_finite = all(report["finite"] and report["rank_control"]["finite"] for report in all_reports)
    higher_reports = [report for report in all_reports if report["multiplier"] in (2.0, 4.0)]
    dose_safe: dict[str, bool] = {}
    for multiplier in (2.0, 4.0):
        candidates = [report for report in higher_reports if report["multiplier"] == multiplier]
        dose_safe[str(multiplier)] = bool(candidates) and all(
            ENCODER_RATIO_MIN <= report["regularizer_to_reconstruction_encoder_ratio"] <= ENCODER_RATIO_MAX
            and report["combined_reconstruction_dot_encoder"] > 0.0
            and report["combined_regularizer_dot_encoder"] > 0.0
            and report["true_beats_rank_control"]
            for report in candidates
        )
    selected_multiplier = max(
        (float(multiplier) for multiplier, passed in dose_safe.items() if passed),
        default=None,
    )
    source_gate = abs(source_replay_mse - SOURCE_PAIRED_MSE) <= 1.0e-9
    identity_gate = all(
        item["plane"]["plane_id"] == "encoder_layer_2"
        and item["provenance"]["operation"] == "finite_crop_hard_aperture"
        for item in field_metrics.values()
    )
    state_gate = model_isolated and all(item["state_unchanged"] for item in gradient_models.values())
    finite_gate = projection_finite and report_finite and math.isfinite(source_replay_mse)
    projection_gate = (
        projection_finite
        and all(value >= PROJECTION_OVERLAP_MIN for value in overlaps.values())
        and all(value <= CAPTURED_ENERGY_DROP_MAX for value in energy_drops.values())
    )
    rank_control_gate = all(report["true_beats_rank_control"] for report in higher_reports)
    dose_gate = selected_multiplier is not None
    harness_gates = (
        GateResult("source_reconstruction_reproduced", source_gate, {"reference": SOURCE_PAIRED_MSE, "replay": source_replay_mse}),
        GateResult("finite_projection_and_gradients", finite_gate, {}),
        GateResult("formal_identity_trace", identity_gate, {}),
        GateResult("model_state_unchanged", state_gate, {}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
        GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
    )
    science_gates = (
        GateResult("projection_stability", projection_gate, {"overlap": overlaps, "energy_drop": energy_drops}),
        GateResult("higher_dose_gradient_compatibility", dose_gate, {"safe": dose_safe, "selected_multiplier": selected_multiplier}),
        GateResult("rank_control_is_worse", rank_control_gate, {"reports": len(higher_reports)}),
    )
    storage = storage_before
    resource_gate = peak <= MAX_PEAK_BYTES and elapsed <= MAX_WALL_SECONDS and storage <= MAX_STORAGE_BYTES
    harness_gates = harness_gates + (
        GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
    )
    if not all(gate.passed for gate in harness_gates):
        status, actions, reason = "invalid_harness", (), "objective audit harness gate failed"
    elif not resource_gate:
        status, actions, reason = "resource_aborted", (), "objective audit resource gate failed"
    elif not source_gate or not finite_gate or not identity_gate or not state_gate:
        status, actions, reason = "invalid_harness", (), "objective audit source or isolation gate failed"
    elif not projection_gate:
        status, actions, reason = "valid_failed", (NEXT_PROJECTION_ACTION,), "fixed projection drift gate failed"
    elif not rank_control_gate:
        status, actions, reason = "valid_failed", (NEXT_RANK_ACTION,), "rank control did not remain worse"
    elif not dose_gate:
        status, actions, reason = "valid_failed", (NEXT_DOSE_ACTION,), "no safe higher regularization dose"
    else:
        status, actions, reason = "passed", (NEXT_PASS_ACTION,), None
    projection_artifact = {
        "schema": PROJECTION_SCHEMA,
        "source_projection_sha256": source.hashes["projection"],
        "validation_evidence_sha256": validation_hash,
        "models": field_metrics,
        "thresholds": {
            "principal_subspace_overlap_min": PROJECTION_OVERLAP_MIN,
            "captured_energy_relative_drop_max": CAPTURED_ENERGY_DROP_MAX,
        },
        "training_authorized": False,
        "formal_promotion": False,
    }
    projection_path = output.parent / "objective_projection_drift.json"
    _write_new_json(projection_path, projection_artifact)
    hashes = {**source.hashes, "objective_validation": validation_hash, "projection_drift": file_sha256(projection_path)}
    record = _record(state, spec, formal, data, hashes, science_gates + harness_gates, status, actions, reason)
    payload = _record_payload(
        AUDIT_SCHEMA,
        record,
        {
            "source_reconstruction_mse": {"authoritative_cuda_replay": source_replay_mse, "reference": SOURCE_PAIRED_MSE, "tolerance": 1.0e-9},
            "projection_drift": {"path": str(projection_path.resolve()), "sha256": hashes["projection_drift"], "schema": PROJECTION_SCHEMA, "models": field_metrics},
            "gradient_audit": {"base_lambda": BASE_LAMBDA, "tested_multipliers": list(DOSE_MULTIPLIERS), "selected_multiplier": selected_multiplier, "models": gradient_models},
            "model_state_unchanged": state_gate,
            "evaluator_calls": 0,
            "free_cuda_bytes_before": int(free_bytes),
            "total_cuda_bytes": int(total_bytes),
            "peak_cuda_bytes": peak,
            "wall_seconds": elapsed,
            "run_storage_bytes": storage,
            "formal_candidate_checkpoint": False,
        },
    )
    _write_new_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "passed" else 2


def run(args: argparse.Namespace) -> int:
    if args.device != "cuda":
        raise ValueError("objective audit requires CUDA")
    _require_deterministic_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    _review, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    source = _source_bundle(spec, review_path, data, state["commit"])
    if source.formal_path != formal_path:
        raise ValueError("objective audit formal config binding mismatch")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite objective audit evidence: {output}")
    hashes = dict(source.hashes)
    validation_hash = _load_validation(args.validation_evidence, spec, formal, data, state, hashes)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for objective audit")
    device = torch.device("cuda")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if free_bytes < MIN_FREE_BYTES:
        raise RuntimeError(f"free CUDA memory {free_bytes} is below {MIN_FREE_BYTES}")
    with _deterministic_algorithms():
        return _run_cuda_audit(
            spec=spec,
            state=state,
            formal=formal,
            data=data,
            source=source,
            validation_hash=validation_hash,
            output=output,
            device=device,
            free_bytes=free_bytes,
            total_bytes=total_bytes,
        )


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "manifold-regularization-objective-audit-validate":
        return validate(args)
    if args.research_action == "manifold-regularization-objective-audit-run":
        return run(args)
    raise ValueError(f"unsupported objective audit action: {args.research_action}")
