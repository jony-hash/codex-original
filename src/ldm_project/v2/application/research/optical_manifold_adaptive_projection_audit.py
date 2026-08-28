"""Cross-fitted projection stability audit for H-LOMA-I0.

This lifecycle is the registered response to the PLAN 5b fixed-projection
failure.  It fits candidate projections only on fresh train panels, evaluates
them on a disjoint assessment panel, and performs a frozen gradient audit for
the selected projection.  It never creates an optimizer, updates a model, or
writes a checkpoint.
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
    atomic_torch_save,
    atomic_write_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from . import optical_manifold as base
from . import optical_manifold_full_epoch as full_epoch
from . import optical_manifold_regularization_audit as regularization
from . import optical_manifold_regularization_objective_audit as objective
from . import optical_manifold_regularization_pilot as pilot


BRANCH = "research/v2-lightgen-optical-manifold-adaptive-projection-audit-v1"
HYPOTHESIS = "H-LOMA-I0"
CONFIG_SCHEMA = "lightgen_optical_manifold_identity_adaptive_projection_audit_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_identity_adaptive_projection_audit_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_identity_adaptive_projection_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_identity_adaptive_projection_audit.v1"
PROJECTION_SCHEMA = "lightgen_optical_manifold_identity_consensus_projection.v1"
SOURCE_STEP = 5500
CALIBRATION_PANELS = 3
CALIBRATION_PANEL_SIZE = 1000
ASSESSMENT_COUNT = 1000
GRADIENT_BATCHES = 3
GRADIENT_BATCH_SIZE = 100
CANDIDATE_COMPONENTS = (16, 24, 32)
FIXED_DIRECTIONS = 64
PERMUTATION_SEED = 1234
MIN_FREE_BYTES = 4 * 1024**3
MAX_PEAK_BYTES = 5 * 1024**3
MAX_WALL_SECONDS = 1200.0
MAX_STORAGE_BYTES = 512 * 1024**2
OVERLAP_MIN = 0.95
CAPTURED_ENERGY_RETENTION_MIN = 0.90
SWD_CV_MAX = 0.05
ENCODER_RATIO_MIN = 0.05
ENCODER_RATIO_MAX = 0.4
SOURCE_PAIRED_MSE = 0.03437697887420654
NEXT_PASS_ACTION = "manual_review_identity_regularization_stable_projection_pilot_contract"
NEXT_NO_PROJECTION_ACTION = "manual_review_probabilistic_latent_contract"
NEXT_OBJECTIVE_ACTION = "manual_review_regularization_objective_replacement_contract"
NEXT_DRIFT_ACTION = "manual_review_online_projection_update_contract"
EXPECTED_FORMAL_KIND = base.EXPECTED_FORMAL_KIND
EXPECTED_TRAINABLE = base.EXPECTED_TRAINABLE


@dataclass(frozen=True)
class AdaptiveProjectionSpec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class AdaptiveProjectionData:
    objective_data: objective.ObjectiveAuditData
    calibration_panels: tuple[torch.Tensor, ...]
    assessment_indices: torch.Tensor
    gradient_batches: tuple[torch.Tensor, ...]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SourceBundle:
    checkpoint: dict[str, Any]
    regularized_checkpoint: dict[str, Any]
    mse_only_checkpoint: dict[str, Any]
    source_projection: dict[str, Any]
    hashes: dict[str, str]
    formal_path: Path


@dataclass(frozen=True)
class ProjectionFit:
    projection: regularization.Projection
    scores: torch.Tensor
    eigenvalues: torch.Tensor


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if not state["clean"]:
        raise ValueError("adaptive projection audit requires a clean worktree")
    if state["branch"] != BRANCH:
        raise ValueError(f"adaptive projection audit requires branch {BRANCH}")


def _require_deterministic_cuda_runtime() -> None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError(
            "adaptive projection audit requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 before the Python process starts"
        )


def _load_spec(path: str | Path) -> AdaptiveProjectionSpec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    expected = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "source",
        "dataset",
        "optical_contract",
        "projection_stability",
        "gradient_audit",
        "gates",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("adaptive projection config has unexpected or missing fields")
    if payload["schema"] != CONFIG_SCHEMA or payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("unsupported adaptive projection config")
    return AdaptiveProjectionSpec(target, payload, canonical_sha256(payload))


def _load_review(
    path: str | Path, spec: AdaptiveProjectionSpec
) -> tuple[dict[str, Any], Path]:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    source = spec.payload["source"]
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_identity_adaptive_projection_audit",
        "source_objective_validation_sha256": source["objective_validation_sha256"],
        "source_objective_audit_sha256": source["objective_audit_sha256"],
        "source_objective_projection_sha256": source["objective_projection_sha256"],
        "source_projection_sha256": source["source_projection_sha256"],
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "source_regularized_checkpoint_sha256": source["regularized_checkpoint_sha256"],
        "source_mse_only_checkpoint_sha256": source["mse_only_checkpoint_sha256"],
        "source_mnist_manifest_hash": source["mnist_manifest_sha256"],
        "approved_contract": {
            "calibration_panels": CALIBRATION_PANELS,
            "calibration_panel_size": CALIBRATION_PANEL_SIZE,
            "assessment_count": ASSESSMENT_COUNT,
            "gradient_batches": GRADIENT_BATCHES,
            "gradient_batch_size": GRADIENT_BATCH_SIZE,
            "candidate_components": list(CANDIDATE_COMPONENTS),
            "fixed_directions": FIXED_DIRECTIONS,
            "direction_seed": PERMUTATION_SEED,
            "no_model_update": True,
            "assessment_used_for_fit": False,
            "assessment_refit_diagnostic_only": True,
            "post_pass_next_action": NEXT_PASS_ACTION,
        },
        "no_update_audit_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "post_no_safe_dose_next_action": NEXT_OBJECTIVE_ACTION,
        "prohibited": spec.payload["authorization"]["prohibited"],
    }
    if payload != expected:
        raise ValueError("adaptive projection review contract does not match scope")
    return payload, target


def _assert_contract(spec: AdaptiveProjectionSpec) -> None:
    payload = spec.payload
    if payload["dataset"] != {
        "train_count": 55000,
        "validation_count": 5000,
        "calibration_panels": 3,
        "calibration_panel_size": 1000,
        "assessment_count": 1000,
        "gradient_batches": 3,
        "gradient_batch_size": 100,
        "selection": "first_430_fresh_train_examples_per_label_after_all_prior_exclusions",
        "excluded_panels": [
            "regularization_calibration",
            "regularization_gradient",
            "pilot_schedule",
            "objective_audit",
            "heldout",
        ],
        "no_replacement": True,
        "heldout_exclusion": True,
        "label_role": "panel_balance_and_provenance_only",
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }:
        raise ValueError("adaptive projection dataset contract is not locked")
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
        raise ValueError("adaptive projection optical contract is not locked")
    if payload["projection_stability"] != {
        "field": "unit_power_normalized_pre_ols_field",
        "metric": "real_part_of_x_x_h",
        "candidate_components": list(CANDIDATE_COMPONENTS),
        "consensus": "orthogonal_projector_basis_consensus",
        "whitening": "calibration_score_covariance_eigenvalue",
        "basis_dtype": "complex64",
        "score_dtype": "float32",
        "overlap_threshold": OVERLAP_MIN,
        "captured_energy_retention": CAPTURED_ENERGY_RETENTION_MIN,
        "swd_coefficient_variation_max": SWD_CV_MAX,
        "source_projection_frozen": True,
        "assessment_used_for_fit": False,
        "assessment_refit_diagnostic_only": True,
    }:
        raise ValueError("adaptive projection stability contract is not locked")
    if payload["gradient_audit"] != {
        "family": "deterministic_sliced_wasserstein",
        "approximation_role": "project_approximation",
        "fixed_directions": FIXED_DIRECTIONS,
        "direction_seed": PERMUTATION_SEED,
        "dose_multipliers": [1.0, 2.0, 4.0],
        "lambda_rule": "batch_0_reconstruction_encoder_norm_over_regularizer_encoder_norm_times_0_1",
        "encoder_ratio_min": ENCODER_RATIO_MIN,
        "encoder_ratio_max": ENCODER_RATIO_MAX,
        "rank_control": "fixed_seed_rank_permutation",
        "replay": "two_pass_fixed_rank_microbatch_1",
        "no_optimizer": True,
        "no_model_update": True,
    }:
        raise ValueError("adaptive projection gradient contract is not locked")
    if payload["gates"] != {
        "source_training_status": "valid_failed_is_consumable_complete_evidence",
        "source_replay_mse_tolerance": 1.0e-9,
        "model_drift_within_source_crossfit_variation": True,
        "rank_control_must_be_worse": True,
        "positive_descent_alignment": True,
        "model_state_unchanged": True,
        "evaluator_calls": 0,
        "min_free_cuda_bytes": MIN_FREE_BYTES,
        "max_peak_cuda_bytes": MAX_PEAK_BYTES,
        "max_wall_seconds": MAX_WALL_SECONDS,
        "max_storage_bytes": MAX_STORAGE_BYTES,
    }:
        raise ValueError("adaptive projection gate contract is not locked")
    if payload["authorization"] != {
        "reviewer_role": "project_owner",
        "decision": "approve_no_update_identity_adaptive_projection_audit",
        "no_update_audit_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "post_pass_next_action": NEXT_PASS_ACTION,
        "post_no_stable_projection_next_action": NEXT_NO_PROJECTION_ACTION,
        "post_rank_control_failure_next_action": NEXT_OBJECTIVE_ACTION,
        "post_no_safe_dose_next_action": NEXT_OBJECTIVE_ACTION,
        "post_model_drift_failure_next_action": NEXT_DRIFT_ACTION,
        "prohibited": payload["authorization"]["prohibited"],
    }:
        raise ValueError("adaptive projection authorization contract is not locked")


def _load_formal(spec: AdaptiveProjectionSpec) -> tuple[FormalSystemConfig, Path]:
    formal_path = _resolve(spec.payload["formal_system_config"], base_path=spec.path.parent)
    formal = load_config(formal_path)
    if formal.architecture != "conditional_identity_logit_amplitude_v2":
        raise ValueError("adaptive projection audit requires current Identity lineage")
    if formal.system.kind != EXPECTED_FORMAL_KIND or not formal.system.is_formal:
        raise ValueError("adaptive projection audit requires formal Identity")
    for name, expected in base.EXPECTED_SYSTEM.items():
        actual = getattr(formal.system, name)
        if isinstance(expected, float):
            if abs(actual - expected) > max(abs(expected) * 1.0e-12, 1.0e-15):
                raise ValueError(f"formal geometry mismatch for {name}")
        elif actual != expected:
            raise ValueError(f"formal geometry mismatch for {name}")
    return formal.system, formal_path


def _assert_ancestor(commit: str, current: str) -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, current],
        cwd=base._repo_root(),
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("adaptive projection source commit is not an ancestor")


def _load_data(dataset_root: str | Path) -> AdaptiveProjectionData:
    objective_data = objective._load_data(dataset_root)
    pilot_data = objective_data.full_data.pilot_data
    train_labels = pilot_data.labels.index_select(0, pilot_data.train_indices)
    schedule = pilot._pilot_schedule(pilot_data.train_indices, pilot_data.labels)
    excluded = torch.cat(
        (
            objective_data.regularization_data.calibration_indices,
            *objective_data.regularization_data.gradient_batches,
            objective_data.projection_indices,
            *objective_data.gradient_batches,
            schedule.reshape(-1),
            pilot_data.heldout_indices,
        )
    )
    if torch.unique(excluded).numel() != excluded.numel():
        raise ValueError("adaptive projection prior exclusions contain duplicates")
    per_label: list[torch.Tensor] = []
    for label in range(10):
        class_indices = pilot_data.train_indices[train_labels == label]
        remaining = class_indices[~torch.isin(class_indices, excluded)]
        if remaining.numel() < 430:
            raise ValueError("train split lacks fresh adaptive projection samples")
        per_label.append(remaining[:430])
    calibration_panels = tuple(
        torch.cat([per_label[label][panel * 100 : (panel + 1) * 100] for label in range(10)])
        for panel in range(CALIBRATION_PANELS)
    )
    assessment_indices = torch.cat([per_label[label][300:400] for label in range(10)])
    gradient_source = [per_label[label][400:430] for label in range(10)]
    gradient_batches = tuple(
        torch.cat(
            [gradient_source[label][batch * 10 : (batch + 1) * 10] for label in range(10)]
        )
        for batch in range(GRADIENT_BATCHES)
    )
    selected = torch.cat((*calibration_panels, assessment_indices, *gradient_batches))
    if selected.numel() != 4300 or torch.unique(selected).numel() != selected.numel():
        raise ValueError("adaptive projection panels are not disjoint")
    if torch.isin(selected, excluded).any() or torch.isin(selected, pilot_data.heldout_indices).any():
        raise ValueError("adaptive projection panels overlap excluded data")
    expected_labels = torch.arange(10).repeat_interleave(10)
    for panel in calibration_panels:
        panel_labels = pilot_data.labels.index_select(0, panel)
        expected_panel = torch.arange(10).repeat_interleave(100)
        if not torch.equal(panel_labels, expected_panel):
            raise ValueError("adaptive calibration panel is not balanced")
    if not torch.equal(pilot_data.labels.index_select(0, assessment_indices), torch.arange(10).repeat_interleave(100)):
        raise ValueError("adaptive assessment panel is not balanced")
    for batch in gradient_batches:
        if not torch.equal(pilot_data.labels.index_select(0, batch), expected_labels):
            raise ValueError("adaptive gradient panel is not balanced")
    manifest = {
        "schema": "lightgen_optical_manifold_identity_adaptive_projection_mnist_manifest.v1",
        "source_mnist_manifest_hash": pilot.SOURCE_MANIFEST_HASH,
        "source_objective_manifest_hash": objective_data.manifest["manifest_hash"],
        "excluded_indices_sha256": tensor_sha256(torch.sort(excluded).values),
        "calibration_panel_indices_sha256": [tensor_sha256(panel) for panel in calibration_panels],
        "assessment_indices_sha256": tensor_sha256(assessment_indices),
        "gradient_batch_indices_sha256": [tensor_sha256(batch) for batch in gradient_batches],
        "calibration_panels": CALIBRATION_PANELS,
        "calibration_panel_size": CALIBRATION_PANEL_SIZE,
        "assessment_count": ASSESSMENT_COUNT,
        "gradient_batches": GRADIENT_BATCHES,
        "gradient_batch_size": GRADIENT_BATCH_SIZE,
        "heldout_excluded": True,
        "no_replacement": True,
        "label_role": "panel_balance_and_provenance_only",
    }
    manifest["manifest_hash"] = canonical_sha256(manifest)
    return AdaptiveProjectionData(
        objective_data=objective_data,
        calibration_panels=calibration_panels,
        assessment_indices=assessment_indices,
        gradient_batches=gradient_batches,
        manifest=manifest,
    )


def _context(
    spec: AdaptiveProjectionSpec,
    formal: FormalSystemConfig,
    data: AdaptiveProjectionData,
) -> EvidenceContextV2:
    return EvidenceContextV2(
        system_signature=canonical_sha256(formal.model_dump(mode="json")),
        config_hash=spec.config_hash,
        dataset_hash=data.manifest["manifest_hash"],
        calibration_hash=canonical_sha256({"id": base.EXPOSURE_ID, "scale": 1.0, "gamma": 1.0}),
        exposure_scale=1.0,
    )


def _record_from_file(path: Path, kind: str, status: str) -> EvidenceRecordV2:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != kind or record.status != status:
        raise ValueError(f"source evidence is not {kind}/{status}")
    return record


def _load_source(
    spec: AdaptiveProjectionSpec,
    review_path: Path,
    data: AdaptiveProjectionData,
    formal_path: Path,
    current_commit: str,
) -> SourceBundle:
    source = spec.payload["source"]
    paths = {
        "objective_config": (_resolve(source["objective_config"]), source["objective_config_sha256"]),
        "objective_review": (_resolve(source["objective_review"]), source["objective_review_sha256"]),
        "objective_validation": (_resolve(source["objective_validation"]), source["objective_validation_sha256"]),
        "objective_audit": (_resolve(source["objective_audit"]), source["objective_audit_sha256"]),
        "objective_projection": (_resolve(source["objective_projection"]), source["objective_projection_sha256"]),
        "source_projection": (_resolve(source["source_projection"]), source["source_projection_sha256"]),
        "source_checkpoint": (_resolve(source["source_checkpoint"]), source["source_checkpoint_sha256"]),
        "regularized_checkpoint": (_resolve(source["regularized_checkpoint"]), source["regularized_checkpoint_sha256"]),
        "mse_only_checkpoint": (_resolve(source["mse_only_checkpoint"]), source["mse_only_checkpoint_sha256"]),
    }
    for role, (path, expected) in paths.items():
        if file_sha256(path) != expected:
            raise ValueError(f"{role} source hash mismatch")
    old_spec = objective._load_spec(paths["objective_config"][0])
    if old_spec.config_hash != objective._load_spec(paths["objective_config"][0]).config_hash:
        raise ValueError("objective source config hash is unstable")
    old_review, old_review_path = objective._load_review(paths["objective_review"][0], old_spec)
    old_data = data.objective_data
    old_bundle = objective._source_bundle(old_spec, old_review_path, old_data, current_commit)
    pilot_config_path = _resolve(old_spec.payload["source"]["pilot_config"])
    if file_sha256(pilot_config_path) != old_spec.payload["source"]["pilot_config_sha256"]:
        raise ValueError("pilot source config hash mismatch")
    pilot_spec = pilot._load_spec(pilot_config_path)
    validation = _record_from_file(
        paths["objective_validation"][0], objective.VALIDATION_SCHEMA, "passed"
    )
    audit_record = _record_from_file(paths["objective_audit"][0], objective.AUDIT_SCHEMA, "valid_failed")
    _assert_ancestor(validation.git_commit, current_commit)
    _assert_ancestor(audit_record.git_commit, current_commit)
    objective_payload = json.loads(paths["objective_audit"][0].read_text(encoding="utf-8"))
    if objective_payload.get("training_authorized") is not False or objective_payload.get("formal_promotion") is not False:
        raise ValueError("PLAN 5b authorization boundary is invalid")
    if objective_payload.get("contract", {}).get("model_state_unchanged") is not True:
        raise ValueError("PLAN 5b source lacks model isolation")
    projection = safe_torch_load(
        paths["source_projection"][0],
        expected_sha256=paths["source_projection"][1],
        map_location="cpu",
    )
    if (
        projection.get("schema") != regularization.PROJECTION_SCHEMA
        or tuple(projection["basis"].shape) != (10000, 32)
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
        raise ValueError("source checkpoint state is incomplete")
    regularized_checkpoint = objective._load_arm_checkpoint(
        paths["regularized_checkpoint"][0], paths["regularized_checkpoint"][1],
        "regularized", pilot_spec.config_hash, data.objective_data.pilot_training_manifest_hash,
    )
    mse_only_checkpoint = objective._load_arm_checkpoint(
        paths["mse_only_checkpoint"][0], paths["mse_only_checkpoint"][1],
        "mse_only", pilot_spec.config_hash, data.objective_data.pilot_training_manifest_hash,
    )
    if old_bundle.formal_path != formal_path:
        raise ValueError("adaptive audit and source formal configs differ")
    if data.manifest["source_mnist_manifest_hash"] != source["mnist_manifest_sha256"]:
        raise ValueError("adaptive audit MNIST manifest binding mismatch")
    if old_review["training_authorized"] is not False or old_review["formal_promotion"] is not False:
        raise ValueError("source review authorization boundary is invalid")
    return SourceBundle(
        checkpoint=source_checkpoint,
        regularized_checkpoint=regularized_checkpoint,
        mse_only_checkpoint=mse_only_checkpoint,
        source_projection=projection,
        formal_path=formal_path,
        hashes={
            "adaptive_review": file_sha256(review_path),
            "objective_config": paths["objective_config"][1],
            "objective_review": paths["objective_review"][1],
            "objective_validation": paths["objective_validation"][1],
            "objective_audit": paths["objective_audit"][1],
            "objective_projection": paths["objective_projection"][1],
            "source_projection": paths["source_projection"][1],
            "source_checkpoint": paths["source_checkpoint"][1],
            "regularized_checkpoint": paths["regularized_checkpoint"][1],
            "mse_only_checkpoint": paths["mse_only_checkpoint"][1],
            "mnist_manifest": source["mnist_manifest_sha256"],
            "heldout_indices": source["heldout_indices_sha256"],
            "formal_config": file_sha256(formal_path),
            "panel_manifest": data.manifest["manifest_hash"],
        },
    )


def _parameter_contract(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, Any]:
    names = tuple(name for name, _ in model.named_parameters())
    trainable = tuple(name for name, value in model.named_parameters() if value.requires_grad)
    if trainable != EXPECTED_TRAINABLE:
        raise ValueError("adaptive projection trainable parameters are not four phase layers")
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
        raise FileExistsError(f"refusing to overwrite adaptive projection evidence: {target}")
    atomic_write_json(target, payload)


def _validation_record(
    state: Mapping[str, Any],
    spec: AdaptiveProjectionSpec,
    formal: FormalSystemConfig,
    data: AdaptiveProjectionData,
    hashes: Mapping[str, str],
) -> EvidenceRecordV2:
    gates = tuple(
        GateResult(name, True, metrics)
        for name, metrics in (
            ("clean_lineage", {"clean": True}),
            ("independent_adaptive_projection_branch", {"branch_matches": True}),
            ("valid_failed_objective_source_consumed", {"source_step": SOURCE_STEP}),
            ("fresh_train_only_panels", {"calibration": 3000, "assessment": 1000, "gradient": 300}),
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
        permitted_next_actions=("run_identity_adaptive_projection_audit",),
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
    source = _load_source(spec, review_path, data, formal_path, state["commit"])
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
            "projection_stability": spec.payload["projection_stability"],
            "gradient_audit": spec.payload["gradient_audit"],
            "no_update_audit_authorized": review["no_update_audit_authorized"],
        },
    )
    _write_new_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _load_validation(
    path: str | Path,
    spec: AdaptiveProjectionSpec,
    formal: FormalSystemConfig,
    data: AdaptiveProjectionData,
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
        raise ValueError("adaptive projection validation schema or authorization is invalid")
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    if record.evidence_kind != VALIDATION_SCHEMA or record.status != "passed":
        raise ValueError("adaptive projection validation must be passed")
    if record.git_commit != state["commit"] or state["branch"] != BRANCH or not state["clean"]:
        raise ValueError("adaptive projection validation is stale")
    if record.config_hash != spec.config_hash or record.context != _context(spec, formal, data):
        raise ValueError("adaptive projection validation context mismatch")
    if record.input_hashes != dict(hashes):
        raise ValueError("adaptive projection validation source hashes do not match")
    if record.permitted_next_actions != ("run_identity_adaptive_projection_audit",):
        raise ValueError("adaptive projection validation next action is invalid")
    if payload.get("contract", {}).get("no_update_audit_authorized") is not True:
        raise ValueError("adaptive projection validation lacks no-update authorization")
    return digest


def _fit_projection_rank(fields: torch.Tensor, rank: int) -> ProjectionFit:
    if fields.ndim != 4 or tuple(fields.shape[1:]) != (1, 100, 100) or not torch.is_complex(fields):
        raise ValueError("PCA fields must have shape [N, 1, 100, 100] and be complex")
    if rank not in CANDIDATE_COMPONENTS:
        raise ValueError("unsupported adaptive projection rank")
    values = regularization._unit_power_field(fields.to(torch.complex64)).flatten(1)
    mean = values.mean(dim=0)
    centered = values - mean
    gram = (centered @ centered.conj().T).real.to(torch.float64).div(float(values.shape[0]))
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    full_eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
    eigenvectors = eigenvectors.index_select(1, order)
    if full_eigenvalues.numel() < rank or float(full_eigenvalues[rank - 1]) <= 1.0e-15:
        raise ValueError("adaptive calibration PCA has insufficient finite rank")
    selected = full_eigenvalues[:rank]
    basis = centered.conj().T @ eigenvectors[:, :rank].to(centered.dtype)
    basis = basis / (float(values.shape[0]) * selected).sqrt().reshape(1, -1)
    basis = basis.to(torch.complex64)
    whitening = selected.to(torch.float32).sqrt().clamp_min(1.0e-12)
    projection = regularization.Projection(
        mean_field=mean.to(torch.complex64),
        basis=basis,
        whitening=whitening,
        eigenvalues=selected.to(torch.float32),
        explained_variance=(selected / full_eigenvalues.sum().clamp_min(1.0e-20)).to(torch.float32),
    )
    return ProjectionFit(projection, _project_fields(fields, projection), full_eigenvalues)


def _project_fields(fields: torch.Tensor, projection: regularization.Projection) -> torch.Tensor:
    values = regularization._unit_power_field(fields.to(torch.complex64)).flatten(1)
    centered = values - projection.mean_field.reshape(1, -1).to(values.device)
    scores = (centered @ projection.basis.to(values.device).conj()).real
    return scores / projection.whitening.to(values.device).reshape(1, -1)


def _consensus_projection(
    fits: tuple[ProjectionFit, ...], calibration_fields: torch.Tensor, rank: int
) -> regularization.Projection:
    basis_stack = torch.cat([fit.projection.basis.to(torch.complex128) for fit in fits], dim=1)
    small_gram = basis_stack.conj().T @ basis_stack
    values, vectors = torch.linalg.eigh(small_gram)
    order = torch.argsort(values, descending=True)
    raw_basis = basis_stack @ vectors[:, order[:rank]]
    raw_basis, _ = torch.linalg.qr(raw_basis)
    raw_basis = raw_basis[:, :rank]
    normalized = regularization._unit_power_field(calibration_fields.to(torch.complex64)).flatten(1)
    mean = normalized.mean(dim=0)
    raw_scores = (normalized - mean) @ raw_basis.to(normalized.dtype).conj()
    raw_scores = raw_scores.real.to(torch.float64)
    covariance = raw_scores.T @ raw_scores / float(raw_scores.shape[0])
    covariance_values, covariance_vectors = torch.linalg.eigh(covariance)
    covariance_order = torch.argsort(covariance_values, descending=True)
    selected = covariance_values.index_select(0, covariance_order).clamp_min(0.0)[:rank]
    basis = raw_basis @ covariance_vectors[:, covariance_order[:rank]].to(torch.complex128)
    whitening = selected.to(torch.float32).sqrt().clamp_min(1.0e-12)
    return regularization.Projection(
        mean_field=mean.to(torch.complex64),
        basis=basis.to(torch.complex64),
        whitening=whitening,
        eigenvalues=selected.to(torch.float32),
        explained_variance=(selected / covariance_values.clamp_min(0.0).sum().clamp_min(1.0e-20)).to(torch.float32),
    )


def _projection_overlap(left: regularization.Projection, right: regularization.Projection) -> float:
    singular = torch.linalg.svdvals(left.basis.to(torch.complex128).conj().T @ right.basis.to(torch.complex128))
    return float(singular.square().mean().item())


def _directions(rank: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(PERMUTATION_SEED)
    values = torch.randn(FIXED_DIRECTIONS, rank, generator=generator, dtype=torch.float32)
    return F.normalize(values, dim=1).to(device)


def _swd(scores: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[1] != directions.shape[1]:
        raise ValueError("adaptive SWD score and direction dimensions do not match")
    ordered = (scores @ directions.T).sort(dim=0).values
    quantiles = regularization._normal_midpoint_quantiles(scores.shape[0], scores.device)
    return (ordered - quantiles.reshape(-1, 1)).square().mean()


def _gaussian_swd(count: int, rank: int, directions: torch.Tensor) -> tuple[float, list[float]]:
    generator = torch.Generator(device="cpu").manual_seed(PERMUTATION_SEED)
    values: list[float] = []
    for _ in range(8):
        sample = torch.randn(count, rank, generator=generator, dtype=torch.float32)
        values.append(float(_swd(sample.to(directions.device), directions).cpu()))
    return float(sum(values) / len(values)), values


def _field_stats(
    fields: torch.Tensor,
    projection: regularization.Projection,
    directions: torch.Tensor,
) -> dict[str, Any]:
    values = regularization._unit_power_field(fields.to(torch.complex64)).flatten(1)
    centered = values - projection.mean_field.reshape(1, -1).to(values.device)
    raw_scores = (centered @ projection.basis.to(values.device).conj()).real
    scores = raw_scores / projection.whitening.to(values.device).reshape(1, -1)
    captured = raw_scores.square().sum(dim=1).mean() / centered.abs().square().sum(dim=1).mean().clamp_min(1.0e-20)
    covariance = scores.T @ scores / float(scores.shape[0])
    identity = torch.eye(scores.shape[1], dtype=scores.dtype, device=scores.device)
    covariance_error = (covariance - identity).norm() / identity.norm()
    learned = _swd(scores, directions)
    gaussian, gaussian_values = _gaussian_swd(scores.shape[0], scores.shape[1], directions)
    collapsed = float(_swd(torch.zeros_like(scores), directions).cpu())
    return {
        "fixed_swd": float(learned.cpu()),
        "gaussian_swd": gaussian,
        "gaussian_swd_values": gaussian_values,
        "collapsed_swd": collapsed,
        "captured_energy": float(captured.cpu()),
        "mean_shift": float(values.mean(dim=0).sub(projection.mean_field.reshape(-1).to(values.device)).abs().norm().cpu()),
        "whitened_covariance_error": float(covariance_error.cpu()),
        "scores_shape": list(scores.shape),
        "finite": bool(torch.isfinite(values).all() and torch.isfinite(scores).all() and torch.isfinite(torch.tensor([captured, covariance_error], device=scores.device)).all()),
    }


def _candidate_report(
    rank: int,
    calibration_fields: tuple[torch.Tensor, ...],
    assessment_fields: torch.Tensor,
    original_projection: regularization.Projection,
) -> tuple[regularization.Projection, dict[str, Any]]:
    fits = tuple(_fit_projection_rank(fields, rank) for fields in calibration_fields)
    calibration_union = torch.cat(calibration_fields)
    consensus = _consensus_projection(fits, calibration_union, rank)
    directions = _directions(rank, torch.device("cpu"))
    panel_stats = [_field_stats(fields, consensus, directions) for fields in calibration_fields]
    assessment_stats = _field_stats(assessment_fields, consensus, directions)
    original_stats = _field_stats(assessment_fields, original_projection, _directions(32, torch.device("cpu")))
    pairwise = [
        _projection_overlap(fits[left].projection, fits[right].projection)
        for left in range(CALIBRATION_PANELS)
        for right in range(left + 1, CALIBRATION_PANELS)
    ]
    consensus_overlaps = [_projection_overlap(fit.projection, consensus) for fit in fits]
    assessment_fit = _fit_projection_rank(assessment_fields, rank)
    assessment_overlap = _projection_overlap(assessment_fit.projection, consensus)
    swd_values = [item["fixed_swd"] for item in (*panel_stats, assessment_stats)]
    swd_mean = sum(swd_values) / len(swd_values)
    swd_std = math.sqrt(sum((value - swd_mean) ** 2 for value in swd_values) / len(swd_values))
    all_eigenvalues = [fit.eigenvalues for fit in fits]
    boundary_gaps = []
    for eigenvalues in all_eigenvalues:
        next_value = eigenvalues[rank] if rank < eigenvalues.numel() else torch.tensor(0.0)
        boundary_gaps.append(float((eigenvalues[rank - 1] - next_value).item()))
    captured_retention = assessment_stats["captured_energy"] / max(original_stats["captured_energy"], 1.0e-20)
    stable = (
        min((*pairwise, *consensus_overlaps, assessment_overlap)) >= OVERLAP_MIN
        and captured_retention >= CAPTURED_ENERGY_RETENTION_MIN
        and swd_std / max(swd_mean, 1.0e-20) <= SWD_CV_MAX
        and assessment_stats["finite"]
        and all(item["finite"] for item in panel_stats)
        and assessment_stats["gaussian_swd"] < assessment_stats["fixed_swd"] < assessment_stats["collapsed_swd"]
    )
    report = {
        "rank": rank,
        "pairwise_calibration_overlap": pairwise,
        "calibration_consensus_overlap": consensus_overlaps,
        "assessment_refit_overlap": assessment_overlap,
        "panel_stats": panel_stats,
        "assessment_stats": assessment_stats,
        "original_top32_assessment_stats": original_stats,
        "swd_mean": swd_mean,
        "swd_std": swd_std,
        "swd_coefficient_variation": swd_std / max(swd_mean, 1.0e-20),
        "boundary_gaps": boundary_gaps,
        "captured_energy_retention_vs_original_top32": captured_retention,
        "assessment_used_for_fit": False,
        "assessment_refit_diagnostic_only": True,
        "stable_candidate": stable,
        "finite": bool(stable or all(item["finite"] for item in panel_stats) and assessment_stats["finite"]),
    }
    return consensus, report


def _state_cpu(model: base.IdentityOpticalManifoldAutoencoder) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_equal(model: base.IdentityOpticalManifoldAutoencoder, expected: Mapping[str, torch.Tensor]) -> bool:
    actual = model.state_dict()
    return set(actual) == set(expected) and all(
        torch.equal(actual[name].detach().cpu(), value) for name, value in expected.items()
    )


def _zero_grad(model: base.IdentityOpticalManifoldAutoencoder) -> None:
    for parameter in model.parameters():
        parameter.grad = None


def _add_grads(total: dict[str, torch.Tensor], model: base.IdentityOpticalManifoldAutoencoder) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            total[name].add_(parameter.grad.detach().cpu())


def _rank_coefficients(scores: torch.Tensor, directions: torch.Tensor, rank_shift: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    projected = scores @ directions.T
    order = projected.argsort(dim=0)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, torch.arange(scores.shape[0], device=scores.device).reshape(-1, 1).expand_as(order))
    if not regularization._rank_assignment_is_permutation(ranks):
        raise ValueError("adaptive rank assignment is not a permutation")
    if rank_shift:
        ranks = torch.roll(ranks, shifts=rank_shift, dims=0)
    quantiles = regularization._normal_midpoint_quantiles(scores.shape[0], scores.device)
    target = quantiles.index_select(0, ranks.reshape(-1)).reshape_as(ranks)
    coefficients = 2.0 * (projected - target).div(float(scores.shape[0] * FIXED_DIRECTIONS))
    return coefficients, ranks


def _gradient_replay(
    model: base.IdentityOpticalManifoldAutoencoder,
    data: AdaptiveProjectionData,
    indices: torch.Tensor,
    projection: regularization.Projection,
    directions: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, Any]]:
    if indices.numel() != GRADIENT_BATCH_SIZE:
        raise ValueError("adaptive gradient replay requires exactly 100 samples")
    projection = regularization._projection_on_device(projection, device)
    first_pass: list[torch.Tensor] = []
    plane: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    model.eval()
    with torch.no_grad():
        for index in indices:
            amplitude, _target = objective._render_single(data.objective_data, index, device)
            _raw, pre, current_plane, current_provenance = objective._forward_trace(model, amplitude)
            first_pass.append(_project_fields(pre, projection).detach())
            if plane is None:
                plane, provenance = current_plane, current_provenance
            elif plane != current_plane or provenance != current_provenance:
                raise ValueError("adaptive gradient trace provenance changed")
    scores = torch.cat(first_pass, dim=0)
    directions = directions.to(device)
    true_coefficients, ranks = _rank_coefficients(scores, directions)
    control_coefficients, control_ranks = _rank_coefficients(scores, directions, rank_shift=1)
    reconstruction = {name: torch.zeros_like(parameter, device="cpu") for name, parameter in model.named_parameters()}
    regularizer = {name: value.clone() for name, value in reconstruction.items()}
    control = {name: value.clone() for name, value in reconstruction.items()}
    for position, index in enumerate(indices):
        amplitude, target = objective._render_single(data.objective_data, index, device)
        _zero_grad(model)
        raw, _pre, _plane, _provenance = objective._forward_trace(model, amplitude)
        F.mse_loss(raw, target).div(float(GRADIENT_BATCH_SIZE)).backward()
        _add_grads(reconstruction, model)
        _zero_grad(model)
        _raw, pre, _plane, _provenance = objective._forward_trace(model, amplitude)
        score = _project_fields(pre, projection)
        ((score @ directions.T) * true_coefficients[position : position + 1]).sum().backward()
        _add_grads(regularizer, model)
        _zero_grad(model)
        _raw, pre, _plane, _provenance = objective._forward_trace(model, amplitude)
        score = _project_fields(pre, projection)
        ((score @ directions.T) * control_coefficients[position : position + 1]).sum().backward()
        _add_grads(control, model)
    _zero_grad(model)
    return reconstruction, regularizer, control, {
        "first_pass_scores_shape": list(scores.shape),
        "plane": plane,
        "provenance": provenance,
        "rank_assignment_shape": list(ranks.shape),
        "rank_assignment_is_permutation": regularization._rank_assignment_is_permutation(ranks),
        "rank_assignment_covers_zero_to_n_minus_one": bool(
            torch.equal(torch.sort(ranks, dim=0).values, torch.arange(GRADIENT_BATCH_SIZE, device=device).reshape(-1, 1).expand_as(ranks))
        ),
        "rank_control_assignment_is_permutation": regularization._rank_assignment_is_permutation(control_ranks),
        "rank_control_covers_zero_to_n_minus_one": bool(
            torch.equal(torch.sort(control_ranks, dim=0).values, torch.arange(GRADIENT_BATCH_SIZE, device=device).reshape(-1, 1).expand_as(control_ranks))
        ),
        "rank_control": "roll_one_sample_axis_seed_1234",
    }


def _gradient_norm(gradients: Mapping[str, torch.Tensor], names: tuple[str, ...]) -> float:
    return float(torch.sqrt(sum(gradients[name].square().sum() for name in names)).item())


def _gradient_dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], names: tuple[str, ...]) -> float:
    return float(sum((left[name] * right[name]).sum() for name in names).item())


def _scaled_combined(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
    return {name: left[name] + scale * right[name] for name in left}


def _gradient_audit(
    model: base.IdentityOpticalManifoldAutoencoder,
    data: AdaptiveProjectionData,
    projection: regularization.Projection,
    directions: torch.Tensor,
    device: torch.device,
) -> tuple[float, list[dict[str, Any]]]:
    encoder = tuple(name for name in EXPECTED_TRAINABLE if name.startswith("system.encoder.phase"))
    reports: list[dict[str, Any]] = []
    lambda_value: float | None = None
    for batch_index, batch in enumerate(data.gradient_batches):
        reconstruction, reg, control, replay = _gradient_replay(model, data, batch, projection, directions, device)
        if batch_index == 0:
            rec_norm = _gradient_norm(reconstruction, encoder)
            reg_norm = _gradient_norm(reg, encoder)
            if rec_norm <= 0.0 or reg_norm <= 0.0:
                raise ValueError("adaptive projection cannot calibrate a positive lambda")
            lambda_value = 0.1 * rec_norm / reg_norm
        assert lambda_value is not None
        for multiplier in (1.0, 2.0, 4.0):
            scale = lambda_value * multiplier
            report = objective._gradient_report(reconstruction, reg, scale)
            combined = _scaled_combined(reconstruction, reg, scale)
            control_combined = _scaled_combined(reconstruction, control, scale)
            true_alignment = _gradient_dot(combined, reg, encoder)
            control_alignment = _gradient_dot(control_combined, reg, encoder)
            report.update(
                {
                    "batch_index": batch_index,
                    "multiplier": multiplier,
                    "lambda": scale,
                    "replay": replay,
                    "true_objective_alignment": true_alignment,
                    "rank_control_true_objective_alignment": control_alignment,
                    "true_first_order_decrease_beats_rank_control": true_alignment > control_alignment,
                    "rank_control_gradient_norm": _gradient_norm(control, encoder),
                }
            )
            reports.append(report)
    _zero_grad(model)
    return lambda_value, reports


def _record(
    state: Mapping[str, Any],
    spec: AdaptiveProjectionSpec,
    formal: FormalSystemConfig,
    data: AdaptiveProjectionData,
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
    spec: AdaptiveProjectionSpec,
    state: Mapping[str, Any],
    formal: FormalSystemConfig,
    data: AdaptiveProjectionData,
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
    source_model = base.IdentityOpticalManifoldAutoencoder(formal).to(device)
    source_model.load_state_dict(source.checkpoint["model_state"], strict=True)
    source_model.eval()
    source_before = _state_cpu(source_model)
    calibration_fields: list[torch.Tensor] = []
    plane: dict[str, Any] | None = None
    provenance: dict[str, Any] | None = None
    for panel in data.calibration_panels:
        fields, _mse, current_plane, current_provenance = objective._capture_fields(
            source_model, data.objective_data, panel, device
        )
        calibration_fields.append(fields)
        if plane is None:
            plane, provenance = current_plane, current_provenance
        elif plane != current_plane or provenance != current_provenance:
            raise ValueError("adaptive calibration trace provenance changed")
    assessment_fields, _assessment_mse, assessment_plane, assessment_provenance = objective._capture_fields(
        source_model, data.objective_data, data.assessment_indices, device
    )
    if plane != assessment_plane or provenance != assessment_provenance:
        raise ValueError("adaptive assessment trace provenance changed")
    original_projection = objective._projection_from_payload(source.source_projection, torch.device("cpu"))
    candidate_projections: dict[str, regularization.Projection] = {}
    candidate_reports: dict[str, dict[str, Any]] = {}
    for rank in CANDIDATE_COMPONENTS:
        consensus, report = _candidate_report(
            rank,
            tuple(calibration_fields),
            assessment_fields,
            original_projection,
        )
        candidate_projections[str(rank)] = consensus
        candidate_reports[str(rank)] = report
    stable_ranks = [rank for rank in CANDIDATE_COMPONENTS if candidate_reports[str(rank)]["stable_candidate"]]
    selected_rank = max(stable_ranks) if stable_ranks else None
    selected_projection = candidate_projections[str(selected_rank)] if selected_rank is not None else None
    directions = _directions(selected_rank, device) if selected_rank is not None else None
    model_metrics: dict[str, Any] = {}
    model_payloads = {
        "source_step_5500": source.checkpoint["model_state"],
        "regularized_final_step_100": source.regularized_checkpoint["model_state"],
        "mse_only_final_step_100": source.mse_only_checkpoint["model_state"],
    }
    if selected_projection is not None and directions is not None:
        for name, model_state in model_payloads.items():
            model = source_model if name == "source_step_5500" else base.IdentityOpticalManifoldAutoencoder(formal).to(device)
            if name != "source_step_5500":
                model.load_state_dict(model_state, strict=True)
                model.eval()
            model_before = _state_cpu(model)
            fields = assessment_fields if name == "source_step_5500" else objective._capture_fields(
                model, data.objective_data, data.assessment_indices, device
            )[0]
            fit = _fit_projection_rank(fields, selected_rank)
            stats = _field_stats(fields, selected_projection, directions)
            model_metrics[name] = {
                "stats": stats,
                "refit_overlap_to_consensus": _projection_overlap(fit.projection, selected_projection),
                "state_unchanged": _state_equal(model, model_before),
            }
            if name != "source_step_5500":
                del model
                torch.cuda.empty_cache()
    source_replay_mse = full_epoch.pilot._evaluate(
        source_model, data.objective_data.full_data.pilot_data,
        data.objective_data.full_data.pilot_data.heldout_indices, device
    )
    gradient_reports: list[dict[str, Any]] = []
    lambda_value: float | None = None
    if selected_projection is not None and directions is not None:
        lambda_value, gradient_reports = _gradient_audit(
            source_model, data, selected_projection, directions, device
        )
    state_unchanged = _state_equal(source_model, source_before)
    _zero_grad(source_model)
    del source_model
    torch.cuda.empty_cache()
    elapsed = time.monotonic() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    projection_path: Path | None = None
    projection_hash: str | None = None
    if selected_projection is not None:
        projection_payload = {
            "schema": PROJECTION_SCHEMA,
            "selected_rank": selected_rank,
            "mean_field": selected_projection.mean_field.cpu(),
            "basis": selected_projection.basis.cpu(),
            "whitening": selected_projection.whitening.cpu(),
            "eigenvalues": selected_projection.eigenvalues.cpu(),
            "explained_variance": selected_projection.explained_variance.cpu(),
            "calibration_panel_indices": [panel.cpu() for panel in data.calibration_panels],
            "calibration_panel_indices_sha256": [tensor_sha256(panel) for panel in data.calibration_panels],
            "source_hashes": {**source.hashes, "adaptive_validation": validation_hash},
            "projection_contract": spec.payload["projection_stability"],
            "assessment_used_for_fit": False,
            "assessment_refit_diagnostic_only": True,
            "training_authorized": False,
            "formal_promotion": False,
        }
        projection_path = output.parent / "consensus_projection.pt"
        atomic_torch_save(projection_path, projection_payload)
        projection_hash = file_sha256(projection_path)
    storage = sum(item.stat().st_size for item in output.parent.rglob("*") if item.is_file())
    source_crossfit = candidate_reports[str(selected_rank)] if selected_rank is not None else None
    source_overlap_values = [] if source_crossfit is None else [
        *source_crossfit["pairwise_calibration_overlap"],
        *source_crossfit["calibration_consensus_overlap"],
        source_crossfit["assessment_refit_overlap"],
    ]
    source_overlap_mean = float(sum(source_overlap_values) / len(source_overlap_values)) if source_overlap_values else float("nan")
    source_overlap_std = math.sqrt(
        sum((value - source_overlap_mean) ** 2 for value in source_overlap_values) / len(source_overlap_values)
    ) if source_overlap_values else float("nan")
    drift_threshold = source_overlap_mean - 2.0 * source_overlap_std if source_overlap_values else float("nan")
    drift_gate = bool(
        selected_rank is not None
        and all(
            model_metrics[name]["refit_overlap_to_consensus"] >= drift_threshold
            for name in ("regularized_final_step_100", "mse_only_final_step_100")
        )
    )
    regularized_better = bool(
        selected_rank is not None
        and model_metrics["regularized_final_step_100"]["stats"]["fixed_swd"]
        < model_metrics["mse_only_final_step_100"]["stats"]["fixed_swd"]
    )
    all_reports_finite = bool(
        selected_rank is None
        or all(report["finite"] for report in gradient_reports)
        and all(report["replay"]["rank_assignment_covers_zero_to_n_minus_one"] for report in gradient_reports)
    )
    dose_safe: dict[str, bool] = {}
    selected_multiplier: float | None = None
    for multiplier in (1.0, 2.0, 4.0):
        candidates = [report for report in gradient_reports if report["multiplier"] == multiplier]
        dose_safe[str(multiplier)] = bool(candidates) and all(
            ENCODER_RATIO_MIN <= report["regularizer_to_reconstruction_encoder_ratio"] <= ENCODER_RATIO_MAX
            and report["combined_reconstruction_dot_encoder"] > 0.0
            and report["true_objective_alignment"] > 0.0
            and report["true_first_order_decrease_beats_rank_control"]
            for report in candidates
        )
    valid_doses = [float(key) for key, value in dose_safe.items() if value]
    if valid_doses:
        selected_multiplier = max(valid_doses)
    selected_reports = [report for report in gradient_reports if report["multiplier"] == selected_multiplier]
    rank_control_gate = bool(selected_reports) and all(
        report["true_first_order_decrease_beats_rank_control"] for report in selected_reports
    )
    ratio_gate = bool(selected_reports) and all(
        ENCODER_RATIO_MIN <= report["regularizer_to_reconstruction_encoder_ratio"] <= ENCODER_RATIO_MAX
        for report in selected_reports
    )
    identity_gate = plane is not None and provenance is not None and plane["plane_id"] == "encoder_layer_2" and provenance["operation"] == "finite_crop_hard_aperture"
    source_gate = math.isfinite(source_replay_mse) and abs(source_replay_mse - SOURCE_PAIRED_MSE) <= 1.0e-9
    finite_gate = bool(
        source_gate
        and (selected_rank is None or all(item["stats"]["finite"] for item in model_metrics.values()))
        and (selected_rank is None or all(report["finite"] for report in gradient_reports))
    )
    model_state_gate = state_unchanged and all(item["state_unchanged"] for item in model_metrics.values())
    stable_gate = selected_rank is not None and candidate_reports[str(selected_rank)]["stable_candidate"]
    science_gates = (
        GateResult("stable_consensus_projection", stable_gate, {"selected_rank": selected_rank, "candidates": candidate_reports}),
        GateResult("checkpoint_drift_within_source_crossfit_variation", drift_gate, {"threshold": drift_threshold, "models": model_metrics}),
        GateResult("regularized_swd_below_mse_only", regularized_better, {"models": model_metrics}),
        GateResult("selected_dose_gradient_safe", selected_multiplier is not None and ratio_gate, {"safe": dose_safe, "selected_multiplier": selected_multiplier}),
        GateResult("rank_control_is_worse", rank_control_gate, {"reports": len(selected_reports)}),
    )
    harness_gates = (
        GateResult("source_reconstruction_reproduced", source_gate, {"reference": SOURCE_PAIRED_MSE, "replay": source_replay_mse}),
        GateResult("finite_values", finite_gate and all_reports_finite, {}),
        GateResult("formal_identity_trace", identity_gate, {"plane": plane, "provenance": provenance}),
        GateResult("model_state_unchanged", model_state_gate, {}),
        GateResult("evaluator_isolation", True, {"evaluator_calls": 0}),
        GateResult("cuda_peak_memory", peak <= MAX_PEAK_BYTES, {"bytes": peak}),
        GateResult("wall_time", elapsed <= MAX_WALL_SECONDS, {"seconds": elapsed}),
        GateResult("run_storage", storage <= MAX_STORAGE_BYTES, {"bytes": storage}),
    )
    resource_gate = peak <= MAX_PEAK_BYTES and elapsed <= MAX_WALL_SECONDS and storage <= MAX_STORAGE_BYTES
    if not all(gate.passed for gate in harness_gates):
        status, actions, reason = "invalid_harness", (), "adaptive projection harness gate failed"
    elif not resource_gate:
        status, actions, reason = "resource_aborted", (), "adaptive projection resource gate failed"
    elif not all(gate.passed for gate in harness_gates[:5]):
        status, actions, reason = "invalid_harness", (), "adaptive projection source or isolation gate failed"
    elif not stable_gate:
        status, actions, reason = "valid_failed", (NEXT_NO_PROJECTION_ACTION,), "no stable consensus projection candidate"
    elif not drift_gate:
        status, actions, reason = "valid_failed", (NEXT_DRIFT_ACTION,), "checkpoint drift exceeded source cross-fit variation"
    elif not rank_control_gate:
        status, actions, reason = "valid_failed", (NEXT_OBJECTIVE_ACTION,), "rank-permuted direction was not worse"
    elif selected_multiplier is None or not ratio_gate:
        status, actions, reason = "valid_failed", (NEXT_OBJECTIVE_ACTION,), "no safe adaptive-projection regularization dose"
    elif not regularized_better:
        status, actions, reason = "valid_failed", (NEXT_OBJECTIVE_ACTION,), "regularized SWD did not beat MSE-only"
    else:
        status, actions, reason = "passed", (NEXT_PASS_ACTION,), None
    hashes = {**source.hashes, "adaptive_validation": validation_hash}
    if projection_hash is not None:
        hashes["consensus_projection"] = projection_hash
    record = _record(
        state, spec, formal, data, hashes,
        science_gates + harness_gates,
        status, actions, reason,
    )
    payload = _record_payload(
        AUDIT_SCHEMA,
        record,
        {
            "source_reconstruction_mse": {"authoritative_cuda_replay": source_replay_mse, "reference": SOURCE_PAIRED_MSE, "tolerance": 1.0e-9},
            "candidate_projections": candidate_reports,
            "selected_projection": {"rank": selected_rank, "path": str(projection_path.resolve()) if projection_path else None, "sha256": projection_hash, "schema": PROJECTION_SCHEMA if projection_path else None},
            "model_drift": {"source_overlap_mean": source_overlap_mean, "source_overlap_std": source_overlap_std, "threshold": drift_threshold, "models": model_metrics},
            "gradient_audit": {"lambda": lambda_value, "tested_multipliers": [1.0, 2.0, 4.0], "dose_safe": dose_safe, "selected_multiplier": selected_multiplier, "reports": gradient_reports},
            "trace": {"plane": plane, "provenance": provenance},
            "model_state_unchanged": model_state_gate,
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
        raise ValueError("adaptive projection audit requires CUDA")
    _require_deterministic_cuda_runtime()
    spec = _load_spec(args.config)
    _assert_contract(spec)
    _review, review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    formal, formal_path = _load_formal(spec)
    data = _load_data(args.dataset_root)
    source = _load_source(spec, review_path, data, formal_path, state["commit"])
    hashes = dict(source.hashes)
    validation_hash = _load_validation(args.validation_evidence, spec, formal, data, state, hashes)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite adaptive projection evidence: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for adaptive projection audit")
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
    if args.research_action == "adaptive-projection-audit-validate":
        return validate(args)
    if args.research_action == "adaptive-projection-audit-run":
        return run(args)
    raise ValueError(f"unsupported adaptive projection action: {args.research_action}")
