"""Frozen downstream-to-upstream recovery audit for Scaled-Chen.

This module is a diagnostic-only evidence consumer.  It reuses the registered
RMFA target, panels, physical forward, and closed-form readout gate while
searching registered representations in a fixed downstream-to-upstream order.
It never changes the optical model, checkpoint, evaluator, or training path.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from ...config import FormalSystemConfig
from ...contracts import ComplexField, OperationProvenance
from ...evidence import (
    EvidenceContextV2,
    EvidenceRecordV2,
    GateResult,
    atomic_torch_save,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from . import optical_manifold as base
from . import optical_manifold_smf_bogt as bogt
from . import optical_manifold_smf_bogt_high_fidelity_relay_reference as hfrr
from . import optical_manifold_smf_bogt_horizon as horizon
from . import optical_manifold_smf_bogt_manifold_readiness as mra
from . import optical_manifold_smf_bogt_reconstruction_manifold_failure as rmfa


CONFIG_SCHEMA = "lightgen_optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization_v1"
REVIEW_SCHEMA = "lightgen_optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization_review.v1"
VALIDATION_SCHEMA = "lightgen_optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization_validation.v1"
AUDIT_SCHEMA = "lightgen_optical_manifold_smf_bogt_reconstruction_mapping_alignment_localization.v1"
PROBE_SCHEMA = "scaled_chen_reconstruction_mapping_alignment_localization_probes.v1"
COMPACT_SCHEMA = "scaled_chen_reconstruction_mapping_alignment_localization_compact_report.v1"
HYPOTHESIS = "H-LOMA-CBOGT-S4-RMAL"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"

PRIMARY_CHECKPOINT = "scaled_chen_step_5000"
SELECTION_COUNT = 1000
SELECTION_PER_LABEL = 100
CONFIRMATION_COUNT = 5000
FIT_COUNT = 1000
FIT_PER_LABEL = 100
FIT_TRAIN_LIMIT = 55000
MICROBATCH_SIZE = 10
PINV_RCOND = 1.0e-12
MATERIAL_RELATIVE_IMPROVEMENT = 0.05
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
CONFIDENCE_LEVEL = 0.95
PHYSICAL_BASELINE_TOLERANCE = 1.0e-6
MIN_FREE_CUDA_BYTES = 4 * 1024**3
MAX_PEAK_CUDA_BYTES = 5 * 1024**3
MIN_DISK_FREE_BYTES = 10 * 1024**3
MAX_STORAGE_BYTES = 512 * 1024**2
SQUARE_LAW_TOLERANCE = 1.0e-5

STAGE_ORDER = (
    "cmos_intensity",
    "cmos_incident_complex_field",
    "decoder_layer_2",
    "decoder_input",
    "modal_complex_100",
    "pre_ols",
)

RMFA_CONSUMER = {
    "config": "configs/research/lightgen_optical_manifold_smf_bogt_reconstruction_manifold_failure_v1.yaml",
    "config_sha256": "5db396683751b2804f10b86b5e22e7261a27ea603587dba425768d9433c1836e",
    "review_contract": "configs/research/lightgen_optical_manifold_smf_bogt_reconstruction_manifold_failure_review_v1.json",
    "review_sha256": "a05c0772210f139e88a78e26e347eeb6ea5855182dfd7e07e75c153380574c59",
    "validation": "outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_manifold_failure_v1_attempt_01/validation.json",
    "validation_sha256": "08e6afd841574a5ac6ba126d11af24b560b6f85bf2d5365d42fd229b373625c0",
    "audit": "outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_manifold_failure_v1_attempt_01/audit.json",
    "audit_sha256": "e0a53625171144b7aba9dbdc3258ff94ca2686219a80d0725f535d2625c7d468",
    "compact_report": "outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_manifold_failure_v1_attempt_01/compact_report.json",
    "compact_report_sha256": "174bdf09c8b5a33896b3a22396f8dfaea6dd22200836f9bbbe46d53b57c403f5",
    "probe": "outputs/20260901_v2/runs/lightgen_scaled_chen_reconstruction_manifold_failure_v1_attempt_01/probe.pt",
    "probe_sha256": "25d9eba0d6203027b3a6be00a9d0a5538225510e45c0927f63cf8f78caaf4a09",
    "implementation_commit": "2445d946fd26867d1c9bd98b04f262b50cd0c90e",
    "audit_status": "passed",
    "audit_classification": "reconstruction_mapping_misalignment_supported",
}

REFERENCES = dict(rmfa.REFERENCES)
FIT_PANEL = dict(rmfa.FIT_PANEL)
EVALUATION_PANELS = json.loads(json.dumps(rmfa.EVALUATION_PANELS))
TARGET = dict(rmfa.TARGET)

CANONICALIZATION = {
    "implementation": "HFRR._canonicalize_field",
    "module": "src/ldm_project/v2/application/research/optical_manifold_smf_bogt_high_fidelity_relay_reference.py",
    "module_sha256": "f2f1e0c33666b8538078dc449772c600cfada02ada9c74501043fea4f6d10e7a",
    "review_contract": "configs/research/lightgen_optical_manifold_smf_bogt_relay_metric_reinterpretation_review_v1.json",
    "review_sha256": "9c4eac44b9e022d0434bb04830486132c3eed3c1513a96a02958eade8fb17037",
    "algorithm_id": "separable_direct_fourier_series_b0_finite_pupil_v1",
    "formula": "exp(-i*pi*C*r2/(wavelength*A)); evaluate at x_target=A*x_source,y_target=A*y_source; multiply abs(A)",
    "valid_backend": "scaled_chen_backend",
    "valid_segment": "decoder_layer_2_to_cmos_incident_field",
    "fit": False,
    "sample_independent": True,
    "target_independent": True,
    "global_phase_fit": False,
    "per_pixel_phase_fit": False,
    "amplitude_normalization": False,
    "physical_forward_modified": False,
    "detector_or_mse_input": False,
    "training_input": False,
    "generation_input": False,
    "primary_stage": "cmos_incident_complex_field",
}

STAGE_CONTRACT = {
    "cmos_intensity": {
        "source_semantics": "output.camera",
        "raw_kind": "real_raw_cmos_detector_intensity",
        "physical_plane": "chen_camera",
        "raw_shape": [1, 200, 200],
        "raw_dtype": "torch.float32",
        "pixel_pitch_m": [2.5e-6, 2.5e-6],
        "physical_fov_m": [500.0e-6, 500.0e-6],
        "feature_projection": "mra._feature_projection(real, complex_value=False)",
        "feature_dim": 256,
        "input_dim": 257,
        "canonicalization": None,
    },
    "cmos_incident_complex_field": {
        "source_semantics": "chen_relay(decoder_layer_2)_before_detector_square_law",
        "raw_kind": "incident_complex_field",
        "physical_plane": "chen_camera",
        "raw_shape": [1, 200, 200],
        "raw_dtype": "torch.complex64",
        "pixel_pitch_m": [2.5e-6, 2.5e-6],
        "physical_fov_m": [500.0e-6, 500.0e-6],
        "feature_projection": "mra._feature_projection(complex_value=True)",
        "feature_dim": 512,
        "input_dim": 513,
        "canonicalization": "HFRR._canonicalize_field(raw_incident, scaled_chen_backend)",
        "feature_view_shape": [1, 100, 100],
        "feature_view_dtype": "torch.complex128",
    },
    "decoder_layer_2": {
        "source_semantics": "decoder_final_attenuated_field_before_chen_relay",
        "raw_kind": "complex_field",
        "physical_plane": "decoder_layer_2",
        "raw_shape": [1, 100, 100],
        "raw_dtype": "torch.complex64",
        "pixel_pitch_m": [2.0e-6, 2.0e-6],
        "physical_fov_m": [200.0e-6, 200.0e-6],
        "feature_projection": "mra._feature_projection(complex_value=True)",
        "feature_dim": 512,
        "input_dim": 513,
        "canonicalization": None,
    },
    "decoder_input": {
        "source_semantics": "output.ols_field_post_OLS_decoder_input",
        "raw_kind": "complex_field",
        "physical_plane": "ols_output",
        "raw_shape": [1, 100, 100],
        "raw_dtype": "torch.complex64",
        "pixel_pitch_m": [2.0e-6, 2.0e-6],
        "physical_fov_m": [200.0e-6, 200.0e-6],
        "feature_projection": "mra._feature_projection(complex_value=True)",
        "feature_dim": 512,
        "input_dim": 513,
        "canonicalization": None,
    },
    "modal_complex_100": {
        "source_semantics": "fixed_FiberArrayOLS_100_complex_modal_coefficients",
        "raw_kind": "complex_modal_vector",
        "physical_plane": "encoder_input_mode_basis",
        "raw_shape": [100],
        "raw_dtype": "torch.complex64",
        "pixel_pitch_m": None,
        "physical_fov_m": [200.0e-6, 200.0e-6],
        "feature_projection": "mra._feature_projection(complex_2d_vector)",
        "feature_dim": 200,
        "input_dim": 201,
        "canonicalization": None,
        "port_order": "FiberArrayOLS._mode_bank row-major y_then_x",
        "area_normalization": "encoder_plane_pixel_area",
    },
    "pre_ols": {
        "source_semantics": "encoder_final_attenuated_field_before_FiberArrayOLS",
        "raw_kind": "complex_field",
        "physical_plane": "encoder_layer_2",
        "raw_shape": [1, 100, 100],
        "raw_dtype": "torch.complex64",
        "pixel_pitch_m": [2.0e-6, 2.0e-6],
        "physical_fov_m": [200.0e-6, 200.0e-6],
        "feature_projection": "mra._feature_projection(complex_value=True)",
        "feature_dim": 512,
        "input_dim": 513,
        "canonicalization": None,
        "positive_anchor": "RMFA_existing_probe",
    },
}

PROBE = {
    "family": "moore_penrose_minimum_norm_linear_readout",
    "solver": "torch.linalg.pinv",
    "rcond": PINV_RCOND,
    "fit_dtype": "float64",
    "storage_dtype": "float32",
    "regularization": "none_parameter_free",
    "adaptive_rank": False,
    "nonlinear": False,
    "lambda": None,
    "fit_once_per_stage_before_selection": True,
    "refit_selection": False,
    "refit_confirmation": False,
}

MEASUREMENT = {
    "objective": "furthest_downstream_bounded_frozen_recoverability",
    "primary_checkpoint": PRIMARY_CHECKPOINT,
    "stage_search_order": list(STAGE_ORDER),
    "primary_metric": "probe_paired_MSE",
    "comparators": ["physical_paired_MSE", "class_template_MSE", "cyclic_MSE"],
    "context_metrics": ["instance_retrieval", "within_label_distance_correlation", "rank"],
    "selection_before_confirmation": True,
    "confirmation_conditional_on_selection": True,
    "no_new_panel": True,
    "physical_forward_unchanged": True,
    "mechanism_evidence_only": True,
    "no_update": True,
    "first_double_pass_stops_upstream_search": True,
}

GATES = {
    "material_relative_improvement": MATERIAL_RELATIVE_IMPROVEMENT,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": CONFIDENCE_LEVEL,
    "physical_baseline_tolerance": PHYSICAL_BASELINE_TOLERANCE,
    "square_law_tolerance": SQUARE_LAW_TOLERANCE,
    "min_free_cuda_bytes": MIN_FREE_CUDA_BYTES,
    "max_peak_cuda_bytes": MAX_PEAK_CUDA_BYTES,
    "min_disk_free_bytes": MIN_DISK_FREE_BYTES,
    "max_storage_bytes": MAX_STORAGE_BYTES,
    "optimizer_created": False,
    "backward_calls": 0,
    "evaluator_calls": 0,
    "checkpoint_written": False,
    "prior_sampled": False,
    "generation_started": False,
    "physical_forward_modified": False,
    "formal_promotion": False,
}

AUTHORIZATION = {
    "reviewer_role": "project_owner",
    "decision": "approve_no_update_scaled_chen_reconstruction_mapping_alignment_localization",
    "no_update_diagnostic_authorized": True,
    "training_authorized": False,
    "formal_training_authorized": False,
    "formal_promotion": False,
    "mechanism_evidence_only": True,
    "prohibited": [
        "optimizer",
        "gradient",
        "backward",
        "regularization",
        "hyperparameter_sweep",
        "refit_after_evaluation",
        "checkpoint_write",
        "prior_sampling",
        "generation",
        "evaluator",
        "fmf",
        "training",
        "promotion",
        "physical_forward_modification",
        "new_samples",
        "stage_reordering",
        "result_driven_raw_canonical_selection",
    ],
}

NEXT_ACTION_RUN = "run_scaled_chen_reconstruction_mapping_alignment_localization"
NEXT_ACTION_RESULT_REVIEW = "manual_review_RMAL_recoverability_result"


@dataclass(frozen=True)
class Spec:
    path: Path
    payload: dict[str, Any]
    config_hash: str


@dataclass(frozen=True)
class RMFAConsumer:
    spec: rmfa.Spec
    review_path: Path
    bundle: rmfa.ParentBundle
    source: mra.SourceBundle
    validation_path: Path
    audit_path: Path
    compact_path: Path
    probe_path: Path
    validation: dict[str, Any]
    audit: dict[str, Any]
    compact: dict[str, Any]
    probe_payload: Mapping[str, Any]


@dataclass(frozen=True)
class Probe:
    stage: str
    weights: torch.Tensor
    fit_indices: torch.Tensor
    feature_dim: int
    input_dim: int
    output_dim: int
    rank: int
    effective_rank: int
    condition_number: float
    cutoff: float
    feature_contract_hash: str
    fit_panel_hash: str
    probe_hash: str
    frozen: bool
    state_unchanged: bool
    ols_state_unchanged: bool


@dataclass(frozen=True)
class CapturedStages:
    values: Mapping[str, torch.Tensor]
    planes: Mapping[str, dict[str, Any]]
    raw_metadata: Mapping[str, dict[str, Any]]
    square_law_max_abs_error: float
    square_law_consistent: bool


@dataclass(frozen=True)
class FeatureView:
    values: torch.Tensor
    metadata: dict[str, Any]


class RMALHarnessFailure(RuntimeError):
    """A registered measurement boundary failed before scientific classification."""


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _require_clean_branch(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise ValueError(f"RMAL requires clean branch {BRANCH}")


def _load_spec(path: str | Path) -> Spec:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    required = {
        "schema",
        "hypothesis",
        "formal_system_config",
        "rmfa_consumer",
        "references",
        "fit_panel",
        "evaluation_panels",
        "stage_order",
        "stages",
        "canonicalization",
        "target",
        "probe",
        "measurement",
        "gates",
        "authorization",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("RMAL config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("RMAL config schema or hypothesis is invalid")
    return Spec(target, payload, canonical_sha256(payload))


def _assert_contract(spec: Spec) -> None:
    payload = spec.payload
    expected = {
        "formal_system_config": "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml",
        "rmfa_consumer": RMFA_CONSUMER,
        "references": REFERENCES,
        "fit_panel": FIT_PANEL,
        "evaluation_panels": EVALUATION_PANELS,
        "stage_order": list(STAGE_ORDER),
        "stages": STAGE_CONTRACT,
        "canonicalization": CANONICALIZATION,
        "target": TARGET,
        "probe": PROBE,
        "measurement": MEASUREMENT,
        "gates": GATES,
        "authorization": AUTHORIZATION,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"RMAL {name} contract is not locked")


def _load_review(path: str | Path, spec: Spec) -> Path:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    expected = {
        "schema": REVIEW_SCHEMA,
        "hypothesis": HYPOTHESIS,
        "reviewer_role": AUTHORIZATION["reviewer_role"],
        "decision": AUTHORIZATION["decision"],
        "approved_contract": {
            "rmfa_consumer": RMFA_CONSUMER,
            "references": {
                "primary_checkpoint": PRIMARY_CHECKPOINT,
                "checkpoint_sha256": REFERENCES["scaled_chen_checkpoint"]["sha256"],
            },
            "fit_panel": {
                key: FIT_PANEL[key]
                for key in ("source", "count", "per_label", "construction", "indices_sha256")
            },
            "evaluation_panels": {
                "selection": EVALUATION_PANELS["selection"]["source"],
                "confirmation": EVALUATION_PANELS["confirmation"]["source"],
                "candidate_prefix_consumed": EVALUATION_PANELS["candidate_prefix"]["consumed"],
            },
            "stage_order": list(STAGE_ORDER),
            "stages": STAGE_CONTRACT,
            "canonicalization": CANONICALIZATION,
            "target": TARGET,
            "probe": PROBE,
            "gates": GATES,
            "no_update": True,
            "first_double_pass_stops_upstream_search": True,
        },
        "no_update_diagnostic_authorized": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal_promotion": False,
        "mechanism_evidence_only": True,
        "prohibited": AUTHORIZATION["prohibited"],
    }
    if payload != expected:
        raise ValueError("RMAL review contract does not match scope")
    return target


def _read_json_hash(path: Path, expected_hash: str, label: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_hash:
        raise ValueError(f"{label} is missing, stale, or tampered: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a mapping: {path}")
    return payload


def _load_rmfa_consumer(spec: Spec, dataset_root: str | Path, *, load_checkpoints: bool) -> RMFAConsumer:
    rmfa_config = _resolve(RMFA_CONSUMER["config"], base_path=spec.path.parent)
    if file_sha256(rmfa_config) != RMFA_CONSUMER["config_sha256"]:
        raise ValueError("RMFA consumer config is missing, stale, or tampered")
    rmfa_spec = rmfa._load_spec(rmfa_config)
    rmfa._assert_contract(rmfa_spec)
    review_path = _resolve(RMFA_CONSUMER["review_contract"], base_path=spec.path.parent)
    if file_sha256(review_path) != RMFA_CONSUMER["review_sha256"]:
        raise ValueError("RMFA consumer review is missing, stale, or tampered")
    rmfa._load_review(review_path, rmfa_spec)
    bundle = rmfa._load_parent(rmfa_spec, dataset_root, load_checkpoints=load_checkpoints)
    source = rmfa._load_source(bundle, load_checkpoints=load_checkpoints)
    checkpoint_path = _resolve(REFERENCES["scaled_chen_checkpoint"]["path"])
    if file_sha256(checkpoint_path) != REFERENCES["scaled_chen_checkpoint"]["sha256"]:
        raise ValueError("Scaled-Chen step-5000 checkpoint identity changed")
    if source.hashes.get(PRIMARY_CHECKPOINT) != REFERENCES["scaled_chen_checkpoint"]["sha256"]:
        raise ValueError("Scaled-Chen source checkpoint hash is not the registered RMFA checkpoint")

    validation_path = _resolve(RMFA_CONSUMER["validation"])
    validation = _read_json_hash(validation_path, RMFA_CONSUMER["validation_sha256"], "RMFA validation")
    validation_record = EvidenceRecordV2.from_dict(validation["evidence"])
    validation_record.validate()
    if (
        validation.get("status") != "passed"
        or validation_record.status != "passed"
        or validation_record.git_commit != RMFA_CONSUMER["implementation_commit"]
        or validation_record.config_hash != rmfa_spec.config_hash
    ):
        raise ValueError("RMFA validation terminal state is stale")

    audit_path = _resolve(RMFA_CONSUMER["audit"])
    audit = _read_json_hash(audit_path, RMFA_CONSUMER["audit_sha256"], "RMFA audit")
    audit_record = EvidenceRecordV2.from_dict(audit["evidence"])
    audit_record.validate()
    if (
        audit.get("status") != RMFA_CONSUMER["audit_status"]
        or audit.get("classification") != RMFA_CONSUMER["audit_classification"]
        or audit_record.git_commit != RMFA_CONSUMER["implementation_commit"]
    ):
        raise ValueError("RMFA audit terminal state is stale or mismatched")

    compact_path = _resolve(RMFA_CONSUMER["compact_report"])
    compact = _read_json_hash(compact_path, RMFA_CONSUMER["compact_report_sha256"], "RMFA compact report")
    if (
        compact.get("classification") != RMFA_CONSUMER["audit_classification"]
        or compact.get("generation_readiness") != "not_ready"
    ):
        raise ValueError("RMFA compact terminal interpretation changed")

    probe_path = _resolve(RMFA_CONSUMER["probe"])
    probe_payload = safe_torch_load(probe_path, expected_sha256=RMFA_CONSUMER["probe_sha256"], map_location="cpu")
    if not isinstance(probe_payload, Mapping):
        raise ValueError("RMFA probe artifact is not a mapping")
    return RMFAConsumer(
        rmfa_spec,
        review_path,
        bundle,
        source,
        validation_path,
        audit_path,
        compact_path,
        probe_path,
        validation,
        audit,
        compact,
        probe_payload,
    )


def _fit_indices(consumer: RMFAConsumer, dataset_root: str | Path) -> torch.Tensor:
    horizon_data = horizon._load_data(dataset_root)
    return rmfa._fit_indices(
        consumer.bundle.parent.parent.data,
        horizon_data,
        rmfa._load_plan_indices(rmfa.PLAN3),
        rmfa._load_plan_indices(rmfa.PLAN4),
    )


def _input_hashes(
    spec: Spec,
    review_path: Path,
    consumer: RMFAConsumer,
    fit_indices: torch.Tensor,
) -> dict[str, str]:
    hashes = rmfa._input_hashes(
        consumer.spec,
        consumer.review_path,
        consumer.bundle,
        consumer.source,
        fit_indices,
    )
    hashes.update(
        {
            "research_config": file_sha256(spec.path),
            "review_contract": file_sha256(review_path),
            "implementation_module": file_sha256(Path(__file__).resolve()),
            "rmfa_config": file_sha256(consumer.spec.path),
            "rmfa_review_contract": file_sha256(consumer.review_path),
            "rmfa_validation": file_sha256(consumer.validation_path),
            "rmfa_audit": file_sha256(consumer.audit_path),
            "rmfa_compact_report": file_sha256(consumer.compact_path),
            "rmfa_probe": file_sha256(consumer.probe_path),
            "canonicalizer_module": CANONICALIZATION["module_sha256"],
            "canonicalization_review": CANONICALIZATION["review_sha256"],
            "stage_contract": canonical_sha256(STAGE_CONTRACT),
            "feature_contract": canonical_sha256({"order": STAGE_ORDER, "stages": STAGE_CONTRACT, "canonicalization": CANONICALIZATION}),
            "target_contract": canonical_sha256(TARGET),
            "probe_contract": canonical_sha256(PROBE),
            "rmfa_consumer_contract": canonical_sha256(RMFA_CONSUMER),
        }
    )
    hashes["fit_indices"] = tensor_sha256(fit_indices)
    hashes["scaled_chen_checkpoint"] = REFERENCES["scaled_chen_checkpoint"]["sha256"]
    return hashes


def _context(spec: Spec, consumer: RMFAConsumer) -> EvidenceContextV2:
    signature = canonical_sha256(
        {
            "measurement": MEASUREMENT,
            "stage_contract": STAGE_CONTRACT,
            "canonicalization": CANONICALIZATION,
            "probe": PROBE,
            "hypothesis": HYPOTHESIS,
        }
    )
    manifest = consumer.bundle.parent.parent.data.manifest["manifest_hash"]
    return EvidenceContextV2(
        signature,
        spec.config_hash,
        dataset_hash=manifest,
        checkpoint_hash=REFERENCES["scaled_chen_checkpoint"]["sha256"],
        dataset_split_manifest_hash=manifest,
    )


def _state_cpu(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _state_digest(state: Mapping[str, torch.Tensor]) -> str:
    return canonical_sha256({name: tensor_sha256(value) for name, value in sorted(state.items())})


def _state_equal(before: Mapping[str, torch.Tensor], after: Mapping[str, torch.Tensor]) -> bool:
    return set(before) == set(after) and all(torch.equal(before[name], after[name]) for name in before)


def _plane_contract(plane: Any) -> dict[str, Any]:
    return {
        "plane_id": plane.plane_id,
        "shape": list(plane.shape),
        "pixel_pitch_m": list(plane.pixel_pitch_m),
        "physical_fov_m": list(plane.fov_m),
        "wavelength_m": plane.wavelength_m,
        "z_m": plane.z_m,
        "coordinate_frame": plane.coordinate_frame,
    }


def _load_rmfa_probe(payload: Mapping[str, Any], fit_indices: torch.Tensor) -> Probe:
    weights = payload.get("weights")
    stored_indices = payload.get("fit_indices")
    metadata = payload.get("metadata")
    if not torch.is_tensor(weights) or not torch.is_tensor(stored_indices) or not isinstance(metadata, Mapping):
        raise ValueError("RMFA probe artifact fields are malformed")
    stored_indices = stored_indices.detach().cpu().to(torch.long)
    if not torch.equal(stored_indices, fit_indices):
        raise ValueError("RMFA probe fit panel does not match RMAL fit panel")
    if tuple(weights.shape) != (513, 40000) or weights.dtype != torch.float32:
        raise ValueError("RMFA probe tensor shape or dtype changed")
    if metadata.get("probe_hash") != tensor_sha256(weights) or metadata.get("fit_panel_hash") != tensor_sha256(fit_indices):
        raise ValueError("RMFA probe metadata hash changed")
    return Probe(
        "pre_ols",
        weights.detach().cpu().contiguous(),
        fit_indices.clone(),
        512,
        int(metadata["input_dim"]),
        int(metadata["output_dim"]),
        int(metadata["rank"]),
        int(metadata["effective_rank"]),
        float(metadata["condition_number"]),
        float(metadata["cutoff"]),
        str(metadata["feature_contract_hash"]),
        str(metadata["fit_panel_hash"]),
        str(metadata["probe_hash"]),
        bool(metadata["frozen"]),
        True,
        True,
    )


def _capture(model: bogt.ScaledChenBOGTModel, images: torch.Tensor, device: torch.device) -> CapturedStages:
    rendered = base.render_intensity_batch(images)
    with torch.no_grad():
        output = model(rendered.amplitude.to(device), return_trace=True)
        if len(output.encoder_records) != 2 or len(output.decoder_records) != 2:
            raise RMALHarnessFailure("Scaled-Chen trace does not contain the registered 2+2 records")
        pre = output.encoder_records[-1]
        decoder = output.decoder_records[-1]
        chip = ComplexField(
            decoder.tensor,
            model.system.chip_plane,
            OperationProvenance(
                "fixed_decoder_group_transmission",
                "decoder_layer_2",
                None,
                {"field_factor": bogt.FIELD_TRANSMISSION**2, "already_applied": True},
            ),
        )
        incident = model.chen_relay(chip)
        expected_intensity = incident.tensor.abs().square()
        camera = output.camera
        if expected_intensity.shape != camera.shape:
            raise RMALHarnessFailure("CMOS incident and detector shapes are inconsistent")
        values = {
            "cmos_intensity": camera,
            "cmos_incident_complex_field": incident.tensor,
            "decoder_layer_2": decoder.tensor,
            "decoder_input": output.ols_field.tensor,
            "modal_complex_100": output.coefficients,
            "pre_ols": pre.tensor,
        }
        if not all(bool(torch.isfinite(value).all()) for value in values.values()):
            raise RMALHarnessFailure("RMAL capture produced non-finite values")
        square_law_error = float((expected_intensity - camera).abs().max().detach().cpu())
        planes = {
            "cmos_intensity": _plane_contract(incident.plane),
            "cmos_incident_complex_field": _plane_contract(incident.plane),
            "decoder_layer_2": _plane_contract(decoder.plane),
            "decoder_input": _plane_contract(output.ols_field.plane),
            "modal_complex_100": _plane_contract(model.system.encoder_plane),
            "pre_ols": _plane_contract(pre.plane),
        }
        raw_metadata = {
            name: {
                "shape": list(value.shape[1:]),
                "dtype": str(value.dtype),
                "plane": planes[name],
            }
            for name, value in values.items()
        }
        cpu_values = {name: value.detach().cpu() for name, value in values.items()}
    return CapturedStages(
        cpu_values,
        planes,
        raw_metadata,
        square_law_error,
        square_law_error <= SQUARE_LAW_TOLERANCE,
    )


def _canonicalize_incident(value: torch.Tensor, formal: FormalSystemConfig) -> torch.Tensor:
    hfrr._FORMAL_CACHE = formal
    canonical = hfrr._canonicalize_field(value, "scaled_chen_backend")
    if canonical.ndim != 4 or tuple(canonical.shape[1:]) != (1, 100, 100) or canonical.dtype != torch.complex128:
        raise RMALHarnessFailure("HFRR incident canonicalization identity changed")
    if not torch.isfinite(canonical).all():
        raise RMALHarnessFailure("HFRR incident canonicalization produced non-finite values")
    return canonical


def _feature_for_stage(capture: CapturedStages, stage: str, formal: FormalSystemConfig) -> FeatureView:
    if stage not in STAGE_ORDER:
        raise ValueError(f"unsupported RMAL stage: {stage}")
    raw = capture.values[stage]
    canonicalization = None
    if stage == "cmos_incident_complex_field":
        view = _canonicalize_incident(raw, formal)
        feature = mra._feature_projection(view, complex_value=True).float()
        canonicalization = CANONICALIZATION
    elif stage == "cmos_intensity":
        view = raw
        feature = mra._feature_projection(view, complex_value=False).float()
    elif stage == "modal_complex_100":
        view = raw
        feature = mra._feature_projection(view, complex_value=True).float()
    else:
        view = raw
        feature = mra._feature_projection(view, complex_value=True).float()
    expected_dim = STAGE_CONTRACT[stage]["feature_dim"]
    if feature.ndim != 2 or feature.shape[1] != expected_dim:
        raise RMALHarnessFailure(f"RMAL feature dimension changed for {stage}")
    if not torch.isfinite(feature).all():
        raise RMALHarnessFailure(f"RMAL feature projection is non-finite for {stage}")
    metadata = {
        "stage": stage,
        "raw": capture.raw_metadata[stage],
        "feature_view_shape": list(view.shape[1:]),
        "feature_view_dtype": str(view.dtype),
        "feature_dim": int(feature.shape[1]),
        "input_dim": int(feature.shape[1] + 1),
        "projection": STAGE_CONTRACT[stage]["feature_projection"],
        "canonicalization": canonicalization,
    }
    return FeatureView(feature, metadata)


def _solve_minimum_norm(features: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    if features.ndim != 2 or targets.ndim != 2 or features.shape[0] != targets.shape[0]:
        raise ValueError("RMAL design and target matrices must have equal row counts")
    if targets.shape[1] != 40000 or not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("RMAL design or target dimensions/values are invalid")
    augmented = torch.cat(
        (features.float(), torch.ones((features.shape[0], 1), dtype=torch.float32)),
        dim=1,
    ).double()
    target_double = targets.float().double()
    singular = torch.linalg.svdvals(augmented)
    if singular.numel() == 0 or not torch.isfinite(singular).all() or float(singular.max()) <= 0.0:
        raise ValueError("RMAL fit design is not finite")
    cutoff = float(singular.max()) * PINV_RCOND
    retained = singular >= cutoff
    rank = int(retained.sum())
    condition = float(singular.max() / singular[retained][-1]) if rank else float("inf")
    weights = torch.linalg.pinv(augmented, rcond=PINV_RCOND).mm(target_double).float().contiguous()
    if not torch.isfinite(weights).all():
        raise ValueError("RMAL probe coefficients are not finite")
    return weights, {
        "rank": rank,
        "effective_rank": rank,
        "condition_number": condition,
        "cutoff": cutoff,
    }


def _fit_probe(
    model: bogt.ScaledChenBOGTModel,
    data: mra.AuditData,
    fit_indices: torch.Tensor,
    device: torch.device,
    formal: FormalSystemConfig,
    stage: str,
) -> Probe:
    model_before = _state_cpu(model)
    ols_before = _state_cpu(model.ols)
    fit_images = data.images.index_select(0, fit_indices)
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    feature_metadata: dict[str, Any] | None = None
    for indices in torch.arange(fit_indices.numel(), dtype=torch.long).split(MICROBATCH_SIZE):
        batch = fit_images.index_select(0, indices)
        capture = _capture(model, batch, device)
        view = _feature_for_stage(capture, stage, formal)
        if feature_metadata is None:
            feature_metadata = view.metadata
        features.append(view.values)
        targets.append(base.render_intensity_batch(batch).target.float().flatten(1))
    x = torch.cat(features, dim=0)
    y = torch.cat(targets, dim=0)
    expected_dim = STAGE_CONTRACT[stage]["feature_dim"]
    if x.shape != (FIT_COUNT, expected_dim) or y.shape != (FIT_COUNT, 40000):
        raise RMALHarnessFailure(f"RMAL probe dimensions changed for {stage}")
    weights, solver = _solve_minimum_norm(x, y)
    if not _state_equal(model_before, _state_cpu(model)) or not _state_equal(ols_before, _state_cpu(model.ols)):
        raise RMALHarnessFailure(f"RMAL model state changed while fitting {stage}")
    feature_contract_hash = canonical_sha256(
        {"stage": stage, "contract": STAGE_CONTRACT[stage], "view": feature_metadata}
    )
    return Probe(
        stage,
        weights,
        fit_indices.clone(),
        expected_dim,
        expected_dim + 1,
        int(y.shape[1]),
        int(solver["rank"]),
        int(solver["effective_rank"]),
        float(solver["condition_number"]),
        float(solver["cutoff"]),
        feature_contract_hash,
        tensor_sha256(fit_indices),
        tensor_sha256(weights),
        True,
        True,
        True,
    )


def _apply_probe(probe: Probe, features: torch.Tensor) -> torch.Tensor:
    if (
        not probe.frozen
        or features.ndim != 2
        or features.shape[1] != probe.feature_dim
        or tuple(probe.weights.shape) != (probe.input_dim, probe.output_dim)
    ):
        raise ValueError("RMAL probe is not frozen or has an invalid feature shape")
    augmented = torch.cat(
        (features.float(), torch.ones((features.shape[0], 1), dtype=torch.float32)),
        dim=1,
    )
    result = augmented.mm(probe.weights).reshape(-1, 1, 200, 200)
    if not torch.isfinite(result).all():
        raise RMALHarnessFailure("RMAL probe output is not finite")
    return result


def _context_metrics_from_compact(prediction: torch.Tensor, target: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    return rmfa._context_metrics_from_compact(prediction, target, labels)


def _stratified_bootstrap_ci(
    delta: torch.Tensor,
    labels: torch.Tensor,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    return rmfa._stratified_bootstrap_ci(delta, labels, replicates=replicates, seed=seed)


def _panel_gate(report: Mapping[str, Any], *, require_bootstrap: bool) -> dict[str, Any]:
    probe = report["probe"]
    bootstrap = report["bootstrap"]
    checks = {
        "finite": report.get("finite") is True,
        "physical_anchor_replay": all(report.get("native_replay", {}).values()),
        "relative_improvement": float(report.get("relative_improvement", float("nan"))) >= MATERIAL_RELATIVE_IMPROVEMENT,
        "strict_template_crossing": float(probe.get("paired_mse", float("inf"))) < float(report.get("class_template_mse", float("-inf"))),
        "paired_below_cyclic": float(probe.get("paired_mse", float("inf"))) < float(report.get("cyclic_mse", float("-inf"))),
        "instance_sensitive": report.get("instance_sensitive") is True,
        "frozen": report.get("state_unchanged") is True and report.get("ols_state_unchanged") is True,
        "square_law_consistency": report.get("square_law_consistency") is True,
        "feature_contract": report.get("feature_contract_valid") is True,
        "probe_contract": report.get("probe_contract_valid") is True,
    }
    if require_bootstrap:
        checks["bootstrap_lower_bound"] = float(bootstrap["ci"][0]) > 0.0
    return {"passed": bool(all(checks.values())), "checks": checks}


def _evaluate_panel(
    model: bogt.ScaledChenBOGTModel,
    probe: Probe,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    formal: FormalSystemConfig,
    stage: str,
    name: str,
) -> dict[str, Any]:
    model_before = _state_cpu(model)
    ols_before = _state_cpu(model.ols)
    target = base.render_intensity_batch(images).target.float()
    templates = torch.stack([target[labels == label].mean(dim=0) for label in range(10)])
    cyclic_target = target.roll(1, 0)
    physical_errors: list[torch.Tensor] = []
    probe_errors: list[torch.Tensor] = []
    cyclic_errors: list[torch.Tensor] = []
    template_errors: list[torch.Tensor] = []
    physical_features: list[torch.Tensor] = []
    probe_features: list[torch.Tensor] = []
    target_features: list[torch.Tensor] = []
    stage_metadata: dict[str, Any] | None = None
    square_law_errors: list[float] = []
    square_law_passed: list[bool] = []
    for indices in torch.arange(images.shape[0], dtype=torch.long).split(MICROBATCH_SIZE):
        batch = images.index_select(0, indices)
        capture = _capture(model, batch, device)
        view = _feature_for_stage(capture, stage, formal)
        if stage_metadata is None:
            stage_metadata = view.metadata
        elif view.metadata["feature_dim"] != stage_metadata["feature_dim"]:
            raise RMALHarnessFailure(f"RMAL feature metadata changed within {stage}")
        predicted = _apply_probe(probe, view.values)
        target_batch = target.index_select(0, indices)
        cyclic_batch = cyclic_target.index_select(0, indices)
        template_batch = templates.index_select(0, labels.index_select(0, indices))
        physical = capture.values["cmos_intensity"]
        physical_errors.append((physical - target_batch).square().mean(dim=(1, 2, 3)))
        probe_errors.append((predicted - target_batch).square().mean(dim=(1, 2, 3)))
        cyclic_errors.append((physical - cyclic_batch).square().mean(dim=(1, 2, 3)))
        template_errors.append((physical - template_batch).square().mean(dim=(1, 2, 3)))
        physical_features.append(mra._feature_projection(physical))
        probe_features.append(mra._feature_projection(predicted))
        target_features.append(mra._feature_projection(target_batch))
        square_law_errors.append(capture.square_law_max_abs_error)
        square_law_passed.append(capture.square_law_consistent)
    physical_mse = torch.cat(physical_errors)
    probe_mse = torch.cat(probe_errors)
    cyclic_mse = torch.cat(cyclic_errors)
    template_mse = torch.cat(template_errors)
    physical_compact = torch.cat(physical_features)
    probe_compact = torch.cat(probe_features)
    target_compact = torch.cat(target_features)
    probe_context = _context_metrics_from_compact(probe_compact, target_compact, labels)
    physical_context = _context_metrics_from_compact(physical_compact, target_compact, labels)
    delta = physical_mse - probe_mse
    ci = _stratified_bootstrap_ci(delta, labels)
    anchor = rmfa.PHYSICAL_ANCHORS[name]
    physical_values = {
        "paired_mse": float(physical_mse.mean()),
        "class_template_mse": float(template_mse.mean()),
        "cyclic_mse": float(cyclic_mse.mean()),
    }
    replay = {
        key: abs(physical_values[key] - value) <= PHYSICAL_BASELINE_TOLERANCE
        for key, value in anchor.items()
    }
    feature_contract_valid = (
        probe.feature_dim == STAGE_CONTRACT[stage]["feature_dim"]
        and probe.input_dim == STAGE_CONTRACT[stage]["input_dim"]
        and probe.output_dim == 40000
        and probe.fit_panel_hash == tensor_sha256(probe.fit_indices)
    )
    report = {
        "stage": stage,
        "panel": name,
        "evaluation_count": int(images.shape[0]),
        "representation": stage_metadata,
        "physical": {
            **physical_values,
            "mse_by_label": [float(physical_mse[labels == label].mean()) for label in range(10)],
            **physical_context,
        },
        "probe": {
            "paired_mse": float(probe_mse.mean()),
            "mse_by_label": [float(probe_mse[labels == label].mean()) for label in range(10)],
            **_context_metrics_from_compact(probe_compact, target_compact, labels),
        },
        "cyclic_mse": float(cyclic_mse.mean()),
        "class_template_mse": float(template_mse.mean()),
        "relative_improvement": float(delta.mean() / max(float(physical_mse.mean()), 1.0e-12)),
        "bootstrap": {
            "delta_definition": "physical_paired_mse - probe_paired_mse",
            "mean_delta": float(delta.mean()),
            "ci": list(ci),
            "replicates": BOOTSTRAP_REPLICATES,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "native_physical_anchor": anchor,
        "native_replay": replay,
        "finite": bool(
            torch.isfinite(physical_mse).all()
            and torch.isfinite(probe_mse).all()
            and torch.isfinite(cyclic_mse).all()
            and torch.isfinite(template_mse).all()
        ),
        "instance_sensitive": bool(
            float(probe_mse.mean()) < float(cyclic_mse.mean())
            and float(probe_context["same_label_top1"]) >= float(probe_context["class_template_same_label_top1"])
        ),
        "state_unchanged": _state_equal(model_before, _state_cpu(model)),
        "ols_state_unchanged": _state_equal(ols_before, _state_cpu(model.ols)),
        "feature_contract_valid": feature_contract_valid,
        "probe_contract_valid": probe.input_dim == STAGE_CONTRACT[stage]["input_dim"] and probe.output_dim == 40000,
        "square_law_max_abs_error": max(square_law_errors),
        "square_law_consistency": all(square_law_passed),
        "sample_mse": {
            "physical": physical_mse,
            "probe": probe_mse,
            "cyclic": cyclic_mse,
            "class_template": template_mse,
        },
    }
    report["selection_gate"] = _panel_gate(report, require_bootstrap=False)
    return report


def _json_report(report: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "sample_mse"}


def _probe_metadata(probe: Probe) -> dict[str, Any]:
    return {
        "stage": probe.stage,
        "feature_dim": probe.feature_dim,
        "input_dim": probe.input_dim,
        "output_dim": probe.output_dim,
        "rank": probe.rank,
        "effective_rank": probe.effective_rank,
        "condition_number": probe.condition_number,
        "cutoff": probe.cutoff,
        "feature_contract_hash": probe.feature_contract_hash,
        "fit_panel_hash": probe.fit_panel_hash,
        "probe_hash": probe.probe_hash,
        "frozen": probe.frozen,
    }


def _probe_artifact(probes: Mapping[str, Probe], fit_indices: torch.Tensor) -> dict[str, Any]:
    return {
        "schema": PROBE_SCHEMA,
        "fit_indices": fit_indices,
        "probes": {
            stage: {"weights": probe.weights, "metadata": _probe_metadata(probe)}
            for stage, probe in probes.items()
        },
        "fit_count": FIT_COUNT,
        "mechanism_evidence_only": True,
        "physical_forward_modified": False,
    }


def _record(
    schema: str,
    status: str,
    classification: str | None,
    spec: Spec,
    consumer: RMFAConsumer,
    state: Mapping[str, Any],
    hashes: Mapping[str, str],
    gates: Sequence[GateResult],
    actions: Sequence[str],
    contract: Mapping[str, Any],
    reason: str | None = None,
) -> dict[str, Any]:
    record = EvidenceRecordV2(
        2,
        schema,
        "diagnostic",
        status,
        str(state["commit"]),
        None,
        spec.config_hash,
        _context(spec, consumer),
        dict(hashes),
        tuple(gates),
        tuple(actions),
        reason,
    )
    record.validate()
    return {
        "schema": schema,
        "hypothesis": HYPOTHESIS,
        "status": status,
        "classification": classification,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "backward_calls": 0,
        "evaluator_calls": 0,
        "checkpoint_written": False,
        "prior_sampled": False,
        "generation_started": False,
        "physical_forward_modified": False,
        "historical_evidence_modified": False,
        "mechanism_evidence_only": True,
        "evidence": record.to_dict(),
        "contract": dict(contract),
    }


def _write_new(path: str | Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    atomic_write_new_json(path, payload, artifact_kind=artifact_kind)


def _prepare_validation_slot(path: Path) -> None:
    if path.exists() or path.parent.exists():
        raise FileExistsError(f"refusing to reuse RMAL validation attempt: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=False)


def _check_run_slot(output_path: Path, validation_path: Path) -> None:
    if output_path.parent.resolve() != validation_path.parent.resolve():
        raise ValueError("RMAL validation and audit must share one attempt directory")
    if not output_path.parent.is_dir():
        raise FileNotFoundError(f"RMAL validation attempt directory is missing: {output_path.parent}")
    existing = {item.name for item in output_path.parent.iterdir()}
    if existing != {validation_path.name}:
        raise FileExistsError("RMAL attempt directory contains unexpected artifacts")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite RMAL audit artifact: {output_path}")


def _resource_values() -> tuple[bool, int, int]:
    if not torch.cuda.is_available():
        return False, 0, 0
    free, total = torch.cuda.mem_get_info()
    return True, int(free), int(total)


def _validation_gates(
    spec: Spec,
    review_path: Path,
    consumer: RMFAConsumer,
    fit_indices: torch.Tensor,
    output_path: Path,
) -> tuple[GateResult, ...]:
    data = consumer.bundle.parent.parent.data
    formal = consumer.bundle.parent.parent.formal
    selection_labels = data.test_labels.index_select(0, data.selection_indices)
    counts = torch.bincount(selection_labels, minlength=10)
    operator = mra._build_research_ols(formal)
    cuda_available, free_cuda, _ = _resource_values()
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    checkpoint_path = _resolve(REFERENCES["scaled_chen_checkpoint"]["path"])
    return (
        GateResult("clean_registered_branch", _git_state().get("branch") == BRANCH and _git_state().get("clean") is True, {}),
        GateResult("rmfa_consumer_identity", file_sha256(consumer.spec.path) == RMFA_CONSUMER["config_sha256"] and file_sha256(consumer.probe_path) == RMFA_CONSUMER["probe_sha256"], {}),
        GateResult("rmfa_terminal_classification", consumer.audit.get("classification") == RMFA_CONSUMER["audit_classification"], {}),
        GateResult("checkpoint_identity", file_sha256(checkpoint_path) == REFERENCES["scaled_chen_checkpoint"]["sha256"], {}),
        GateResult("selection_balanced", bool(torch.equal(counts, torch.full((10,), SELECTION_PER_LABEL, dtype=torch.long))), {"counts": counts.tolist()}),
        GateResult("confirmation_count_locked", data.confirmation_indices.numel() == CONFIRMATION_COUNT and torch.equal(data.confirmation_indices, torch.arange(55000, 60000, dtype=torch.long)), {}),
        GateResult("selection_confirmation_disjoint", not bool(torch.isin(data.selection_indices, data.confirmation_indices).any()), {}),
        GateResult("fit_confirmation_disjoint", not bool(torch.isin(fit_indices, data.confirmation_indices).any()), {}),
        GateResult("fit_panel_locked", fit_indices.numel() == FIT_COUNT and tensor_sha256(fit_indices) == FIT_PANEL["indices_sha256"], {}),
        GateResult("mode_bank_rank_100", mra._mode_rank(operator) == 100, {"rank": mra._mode_rank(operator)}),
        GateResult("finite_mode_bank", bool(torch.isfinite(operator.input_modes).all() and torch.isfinite(operator.output_modes).all()), {}),
        GateResult("formal_identity_boundary", formal.kind == bogt.EXPECTED_FORMAL_KIND and formal.is_formal, {}),
        GateResult("stage_order_locked", list(STAGE_ORDER) == spec.payload["stage_order"], {"order": list(STAGE_ORDER)}),
        GateResult("stage_contract_locked", spec.payload["stages"] == STAGE_CONTRACT and all(value["input_dim"] == value["feature_dim"] + 1 for value in STAGE_CONTRACT.values()), {}),
        GateResult("canonicalization_identity", spec.payload["canonicalization"] == CANONICALIZATION and file_sha256(_resolve(CANONICALIZATION["module"])) == CANONICALIZATION["module_sha256"], {}),
        GateResult("candidate_prefix_not_consumed", EVALUATION_PANELS["candidate_prefix"]["consumed"] is False, {}),
        GateResult("no_update_scope", os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", {"optimizer_created": False, "backward_calls": 0, "evaluator_calls": 0}),
        GateResult("windows_native_checkout", os.name == "nt" and Path.cwd().drive.upper() == "D:", {}),
        GateResult("cuda_available", cuda_available, {"free_cuda_bytes": free_cuda}),
        GateResult("free_cuda_precondition", free_cuda >= MIN_FREE_CUDA_BYTES, {"free_cuda_bytes": free_cuda, "minimum": MIN_FREE_CUDA_BYTES}),
        GateResult("disk_free_precondition", disk.free >= MIN_DISK_FREE_BYTES, {"free_bytes": int(disk.free), "minimum": MIN_DISK_FREE_BYTES}),
        GateResult("output_slot_absent", not output_path.parent.exists(), {"path": str(output_path.parent)}),
    )


def _load_validation(
    path: str | Path,
    spec: Spec,
    review_path: Path,
    consumer: RMFAConsumer,
    fit_indices: torch.Tensor,
    state: Mapping[str, Any],
) -> None:
    target = _resolve(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    record = EvidenceRecordV2.from_dict(payload["evidence"])
    record.validate()
    expected_hashes = _input_hashes(spec, review_path, consumer, fit_indices)
    if (
        payload.get("schema") != VALIDATION_SCHEMA
        or payload.get("hypothesis") != HYPOTHESIS
        or payload.get("status") != "passed"
        or record.status != "passed"
        or record.git_commit != state["commit"]
        or record.config_hash != spec.config_hash
        or record.input_hashes != expected_hashes
        or record.permitted_next_actions != (NEXT_ACTION_RUN,)
    ):
        raise ValueError("RMAL validation evidence is stale or mismatched")


def validate(args: argparse.Namespace) -> int:
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    consumer = _load_rmfa_consumer(spec, args.dataset_root, load_checkpoints=False)
    fit_indices = _fit_indices(consumer, args.dataset_root)
    output_path = _resolve(args.output)
    gates = _validation_gates(spec, review_path, consumer, fit_indices, output_path)
    hashes = _input_hashes(spec, review_path, consumer, fit_indices)
    valid = all(gate.passed for gate in gates)
    contract = {
        "rmfa_consumer": RMFA_CONSUMER,
        "stage_order": list(STAGE_ORDER),
        "stages": STAGE_CONTRACT,
        "canonicalization": CANONICALIZATION,
        "fit_panel": FIT_PANEL,
        "evaluation_panels": EVALUATION_PANELS,
        "target": TARGET,
        "probe": PROBE,
        "measurement": MEASUREMENT,
        "gates": GATES,
        "authorization": AUTHORIZATION,
    }
    payload = _record(
        VALIDATION_SCHEMA,
        "passed" if valid else "invalid_harness",
        None if valid else "invalid_harness",
        spec,
        consumer,
        state,
        hashes,
        gates,
        (NEXT_ACTION_RUN,) if valid else (),
        contract,
        None if valid else "RMAL static validation failed",
    )
    _prepare_validation_slot(output_path)
    _write_new(output_path, payload, artifact_kind="RMAL validation evidence")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


def _resource_preflight(output_path: Path, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RMALHarnessFailure("RMAL requires CUDA; CPU is not a formal fallback")
    free, total = torch.cuda.mem_get_info(device)
    if free < MIN_FREE_CUDA_BYTES:
        raise RMALHarnessFailure("RMAL requires at least 4 GiB free CUDA memory")
    disk = shutil.disk_usage(output_path.anchor or str(output_path.parent))
    if disk.free < MIN_DISK_FREE_BYTES:
        raise RMALHarnessFailure("RMAL requires at least 10 GiB free storage")
    return int(free), int(total)


def _invalid_harness_payload(
    output_path: Path,
    spec: Spec,
    review_path: Path,
    consumer: RMFAConsumer,
    fit_indices: torch.Tensor,
    state: Mapping[str, Any],
    reason: str,
    details: Mapping[str, Any] | None = None,
) -> None:
    gates = (
        GateResult("rmfa_positive_anchor_replay", False, {"reason": reason}),
        GateResult("no_update_scope", True, {"optimizer_created": False, "backward_calls": 0, "evaluator_calls": 0}),
    )
    contract = {
        "stage_order": list(STAGE_ORDER),
        "stages_evaluated": [],
        "stages_skipped_by_stop_rule": [],
        "furthest_downstream_recoverable_stage": None,
        "reason": reason,
        "details": dict(details or {}),
    }
    payload = _record(
        AUDIT_SCHEMA,
        "invalid_harness",
        "invalid_harness",
        spec,
        consumer,
        state,
        _input_hashes(spec, review_path, consumer, fit_indices),
        gates,
        (),
        contract,
        reason,
    )
    payload.update(
        {
            "reconstruction_manifold_supported": False,
            "generative_manifold_supported": False,
            "generation_readiness": "not_ready",
            "bottleneck_attribution_family": "closed_task_relevant_bottleneck_not_supported",
            "bounded_recoverability_established": False,
            "furthest_downstream_recoverable_stage": None,
            "stage_search_order": list(STAGE_ORDER),
            "stages_evaluated": [],
            "stages_skipped_by_stop_rule": [],
        }
    )
    _write_new(output_path, payload, artifact_kind="RMAL invalid-harness audit evidence")


def _stage_classification(stage: str) -> str:
    if stage == "cmos_intensity":
        return "bounded_recoverability_established_at_detector"
    if stage == "cmos_incident_complex_field":
        return "bounded_recoverability_established_at_cmos_incident_complex_field"
    return f"bounded_recoverability_established_at_{stage}"


def _anchor_replay(
    model: bogt.ScaledChenBOGTModel,
    rmfa_probe: Probe,
    data: mra.AuditData,
    device: torch.device,
    formal: FormalSystemConfig,
) -> tuple[dict[str, Any], dict[str, Any] | None, bool]:
    selection_images = data.test_images.index_select(0, data.selection_indices)
    selection_labels = data.test_labels.index_select(0, data.selection_indices)
    selection = _evaluate_panel(model, rmfa_probe, selection_images, selection_labels, device, formal, "pre_ols", "selection")
    selection["probe_contract_valid"] = True
    if not selection["selection_gate"]["passed"]:
        return selection, None, False
    confirmation_images = data.images.index_select(0, data.confirmation_indices)
    confirmation_labels = data.labels.index_select(0, data.confirmation_indices)
    confirmation = _evaluate_panel(model, rmfa_probe, confirmation_images, confirmation_labels, device, formal, "pre_ols", "confirmation")
    confirmation["probe_contract_valid"] = True
    confirmation["selection_gate"] = _panel_gate(confirmation, require_bootstrap=False)
    confirmation["confirmation_gate"] = _panel_gate(confirmation, require_bootstrap=True)
    return selection, confirmation, bool(confirmation["selection_gate"]["passed"] and confirmation["confirmation_gate"]["passed"])


def run(args: argparse.Namespace) -> int:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("RMAL requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    spec = _load_spec(args.config)
    _assert_contract(spec)
    review_path = _load_review(args.review_contract, spec)
    state = _git_state()
    _require_clean_branch(state)
    consumer = _load_rmfa_consumer(spec, args.dataset_root, load_checkpoints=False)
    fit_indices = _fit_indices(consumer, args.dataset_root)
    _load_validation(args.validation_evidence, spec, review_path, consumer, fit_indices, state)
    output_path = _resolve(args.output)
    validation_path = _resolve(args.validation_evidence)
    _check_run_slot(output_path, validation_path)
    compact_path = output_path.with_name("compact_report.json")
    probe_path = output_path.with_name("probes.pt")
    if compact_path.exists() or probe_path.exists():
        raise FileExistsError("refusing to overwrite RMAL companion artifact")
    device = torch.device(args.device)
    try:
        free_bytes, total_bytes = _resource_preflight(output_path, device)
        source = rmfa._load_source(consumer.bundle, load_checkpoints=True)
        formal = consumer.bundle.parent.parent.formal
        checkpoint = source.checkpoints[PRIMARY_CHECKPOINT]
        model = mra._load_scaled_model(formal, checkpoint, device)
        hfrr._FORMAL_CACHE = formal
        state_before = _state_cpu(model)
        ols_before = _state_cpu(model.ols)
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats(device)
        rmfa_probe = _load_rmfa_probe(consumer.probe_payload, fit_indices)
        data = consumer.bundle.parent.parent.data
        anchor_selection, anchor_confirmation, anchor_passed = _anchor_replay(model, rmfa_probe, data, device, formal)
        if not anchor_passed:
            _invalid_harness_payload(
                output_path,
                spec,
                review_path,
                consumer,
                fit_indices,
                state,
                "RMFA positive-anchor replay failed",
                {
                    "selection": _json_report(anchor_selection),
                    "confirmation": _json_report(anchor_confirmation) if anchor_confirmation is not None else {"executed": False},
                },
            )
            return 2

        selection_images = data.test_images.index_select(0, data.selection_indices)
        selection_labels = data.test_labels.index_select(0, data.selection_indices)
        confirmation_images = data.images.index_select(0, data.confirmation_indices)
        confirmation_labels = data.labels.index_select(0, data.confirmation_indices)
        probes: dict[str, Probe] = {}
        stage_results: dict[str, dict[str, Any]] = {}
        evaluated: list[str] = []
        winner: str | None = None
        for stage in STAGE_ORDER:
            probe = _fit_probe(model, data, fit_indices, device, formal, stage)
            probes[stage] = probe
            evaluated.append(stage)
            selection = _evaluate_panel(model, probe, selection_images, selection_labels, device, formal, stage, "selection")
            if not selection["selection_gate"]["passed"]:
                stage_results[stage] = {
                    "selection": _json_report(selection),
                    "confirmation": {"executed": False},
                    "outcome": "bounded_recoverability_not_established",
                }
                continue
            confirmation = _evaluate_panel(model, probe, confirmation_images, confirmation_labels, device, formal, stage, "confirmation")
            confirmation["selection_gate"] = _panel_gate(confirmation, require_bootstrap=False)
            confirmation["confirmation_gate"] = _panel_gate(confirmation, require_bootstrap=True)
            if not confirmation["selection_gate"]["passed"] or not confirmation["confirmation_gate"]["passed"]:
                stage_results[stage] = {
                    "selection": _json_report(selection),
                    "confirmation": _json_report(confirmation),
                    "outcome": "bounded_recoverability_not_established_on_confirmation",
                }
                continue
            stage_results[stage] = {
                "selection": _json_report(selection),
                "confirmation": _json_report(confirmation),
                "outcome": "bounded_recoverability_established",
            }
            winner = stage
            break

        state_unchanged = _state_equal(state_before, _state_cpu(model))
        ols_unchanged = _state_equal(ols_before, _state_cpu(model.ols))
        peak = int(torch.cuda.max_memory_allocated(device))
        elapsed = time.monotonic() - started
        stages_skipped = list(STAGE_ORDER[len(evaluated) :]) if winner is not None else []
        classification = _stage_classification(winner) if winner is not None else "bounded_recoverability_not_established"
        status = "passed" if winner is not None else "valid_failed"
        probe_payload = _probe_artifact(probes, fit_indices)
        estimated_probe_bytes = (
            sum(probe.weights.numel() * probe.weights.element_size() for probe in probes.values())
            + fit_indices.numel() * fit_indices.element_size()
            + 64 * 1024
        )
        existing_output_bytes = sum(
            item.stat().st_size for item in output_path.parent.iterdir() if item.is_file()
        )
        if estimated_probe_bytes + existing_output_bytes > MAX_STORAGE_BYTES:
            raise RMALHarnessFailure("RMAL probe artifacts exceed storage gate")
        all_stage_reports = [
            report
            for item in stage_results.values()
            for report in (item["selection"], item.get("confirmation", {}))
            if report.get("executed", True)
        ]
        harness_ok = (
            anchor_passed
            and state_unchanged
            and ols_unchanged
            and all(report.get("finite") is True for report in all_stage_reports)
            and all(report.get("square_law_consistency") is True for report in all_stage_reports)
            and all(report.get("feature_contract_valid") is True for report in all_stage_reports)
            and all(report.get("probe_contract_valid") is True for report in all_stage_reports)
            and all(all(report.get("native_replay", {}).values()) for report in all_stage_reports)
            and peak <= MAX_PEAK_CUDA_BYTES
        )
        if not harness_ok:
            raise RMALHarnessFailure("RMAL harness or resource gate failed")
        atomic_torch_save(probe_path, probe_payload)
        probe_hash = file_sha256(probe_path)
        compact = {
            "schema": COMPACT_SCHEMA,
            "status": status,
            "classification": classification,
            "next_action": NEXT_ACTION_RESULT_REVIEW,
            "stage_search_order": list(STAGE_ORDER),
            "stages_evaluated": evaluated,
            "stages_skipped_by_stop_rule": stages_skipped,
            "furthest_downstream_recoverable_stage": winner,
            "bounded_recoverability_established": winner is not None,
            "positive_anchor_replay": {
                "passed": anchor_passed,
                "selection": _json_report(anchor_selection),
                "confirmation": _json_report(anchor_confirmation) if anchor_confirmation is not None else {"executed": False},
            },
            "stage_results": stage_results,
            "registered_stage_contract": STAGE_CONTRACT,
            "canonicalization": CANONICALIZATION,
            "solver": PROBE,
            "fit_count": FIT_COUNT,
            "selection_count": SELECTION_COUNT,
            "confirmation_count": CONFIRMATION_COUNT,
            "probe_artifact_sha256": probe_hash,
            "reconstruction_manifold_supported": False,
            "generative_manifold_supported": False,
            "generation_readiness": "not_ready",
            "bottleneck_attribution_family": "closed_task_relevant_bottleneck_not_supported",
            "optimizer_created": False,
            "backward_calls": 0,
            "evaluator_calls": 0,
            "checkpoint_written": False,
            "prior_sampled": False,
            "generation_started": False,
            "physical_forward_modified": False,
            "training_authorized": False,
            "formal_promotion": False,
            "mechanism_evidence_only": True,
        }
        _write_new(compact_path, compact, artifact_kind="RMAL compact report")
        compact_hash = file_sha256(compact_path)
        output_bytes_before_audit = int(sum(item.stat().st_size for item in output_path.parent.iterdir() if item.is_file()))
        audit_contract = {
            "rmfa_positive_anchor_replay": compact["positive_anchor_replay"],
            "stage_search_order": list(STAGE_ORDER),
            "stages_evaluated": evaluated,
            "stages_skipped_by_stop_rule": stages_skipped,
            "furthest_downstream_recoverable_stage": winner,
            "bounded_recoverability_established": winner is not None,
            "registered_stage_contract": STAGE_CONTRACT,
            "stage_results": stage_results,
            "probes": {stage: _probe_metadata(probe) for stage, probe in probes.items()},
            "probe_path": str(probe_path.resolve()),
            "probe_sha256": probe_hash,
            "compact_report_path": str(compact_path.resolve()),
            "compact_report_sha256": compact_hash,
            "model_state_before": _state_digest(state_before),
            "model_state_after": _state_digest(_state_cpu(model)),
            "ols_state_before": _state_digest(ols_before),
            "ols_state_after": _state_digest(_state_cpu(model.ols)),
            "state_unchanged": state_unchanged,
            "ols_state_unchanged": ols_unchanged,
            "resource": {
                "free_cuda_bytes_before": free_bytes,
                "total_cuda_bytes": total_bytes,
                "peak_memory_bytes": peak,
                "wall_clock_seconds": elapsed,
                "wall_clock": "observational_only",
                "output_bytes_before_audit": output_bytes_before_audit,
            },
            "solver": PROBE,
            "canonicalization": CANONICALIZATION,
            "target": TARGET,
            "training_authorized": False,
            "formal_promotion": False,
            "optimizer_created": False,
            "backward_calls": 0,
            "evaluator_calls": 0,
            "checkpoint_written": False,
            "prior_sampled": False,
            "generation_started": False,
            "physical_forward_modified": False,
            "historical_evidence_modified": False,
            "mechanism_evidence_only": True,
        }
        gates = (
            GateResult("rmfa_positive_anchor_replay", anchor_passed, {}),
            GateResult("finite_measurements", all(report.get("finite") is True for report in all_stage_reports), {}),
            GateResult("native_physical_replay", all(all(report.get("native_replay", {}).values()) for report in all_stage_reports), {}),
            GateResult("frozen_model_and_OLS", state_unchanged and ols_unchanged, {}),
            GateResult("probe_frozen_and_disjoint", all(probe.frozen and probe.fit_panel_hash == tensor_sha256(fit_indices) for probe in probes.values()), {}),
            GateResult("stage_order_and_stop_rule", evaluated == list(STAGE_ORDER[: len(evaluated)]) and stages_skipped == list(STAGE_ORDER[len(evaluated) :]), {"evaluated": evaluated, "skipped": stages_skipped}),
            GateResult("no_update_isolation", True, {"optimizer_created": False, "backward_calls": 0, "evaluator_calls": 0, "checkpoint_written": False, "prior_sampled": False, "generation_started": False, "physical_forward_modified": False}),
            GateResult("resource_limits", peak <= MAX_PEAK_CUDA_BYTES and output_bytes_before_audit <= MAX_STORAGE_BYTES, {"peak_bytes": peak, "output_bytes_before_audit": output_bytes_before_audit}),
        )
        hashes = _input_hashes(spec, review_path, consumer, fit_indices)
        hashes.update({"rmal_probes": probe_hash, "rmal_compact_report": compact_hash})
        payload = _record(
            AUDIT_SCHEMA,
            status,
            classification,
            spec,
            consumer,
            state,
            hashes,
            gates,
            (NEXT_ACTION_RESULT_REVIEW,),
            audit_contract,
            "RMAL is diagnostic-only; no downstream training or promotion",
        )
        payload.update(
            {
                "reconstruction_manifold_supported": False,
                "generative_manifold_supported": False,
                "generation_readiness": "not_ready",
                "bottleneck_attribution_family": "closed_task_relevant_bottleneck_not_supported",
                "stage_search_order": list(STAGE_ORDER),
                "stages_evaluated": evaluated,
                "stages_skipped_by_stop_rule": stages_skipped,
                "furthest_downstream_recoverable_stage": winner,
                "bounded_recoverability_established": winner is not None,
                "probe_sha256": probe_hash,
                "compact_report_sha256": compact_hash,
            }
        )
        _write_new(output_path, payload, artifact_kind="RMAL audit evidence")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except RMALHarnessFailure as exc:
        if not output_path.exists():
            _invalid_harness_payload(output_path, spec, review_path, consumer, fit_indices, state, str(exc))
        return 2


def dispatch(args: argparse.Namespace) -> int:
    if args.research_action == "scaled-chen-reconstruction-mapping-alignment-localization-validate":
        return validate(args)
    if args.research_action == "scaled-chen-reconstruction-mapping-alignment-localization-run":
        return run(args)
    raise ValueError(f"unsupported RMAL action: {args.research_action}")


__all__ = [
    "AUDIT_SCHEMA",
    "AUTHORIZATION",
    "CANONICALIZATION",
    "CONFIG_SCHEMA",
    "EVALUATION_PANELS",
    "FIT_PANEL",
    "GATES",
    "HYPOTHESIS",
    "MEASUREMENT",
    "NEXT_ACTION_RESULT_REVIEW",
    "NEXT_ACTION_RUN",
    "PROBE",
    "REFERENCES",
    "RMFA_CONSUMER",
    "REVIEW_SCHEMA",
    "STAGE_CONTRACT",
    "STAGE_ORDER",
    "Spec",
    "Probe",
    "VALIDATION_SCHEMA",
    "_apply_probe",
    "_assert_contract",
    "_capture",
    "_evaluate_panel",
    "_fit_indices",
    "_fit_probe",
    "_load_review",
    "_load_spec",
    "_panel_gate",
    "_solve_minimum_norm",
    "_stratified_bootstrap_ci",
    "dispatch",
    "run",
    "validate",
]
