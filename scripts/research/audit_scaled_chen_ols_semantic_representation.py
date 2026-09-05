"""Audit semantic information in the frozen Scaled-Chen OLS representation.

This is a diagnostic, no-update consumer of the registered step-5000 model.
It stops at the complex coefficients emitted by the fixed OLS modal projection;
decoder, relay, CMOS, MPCDAR, optimizer, backward, training, and prior
generation are deliberately outside the measurement boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt as bogt,
)
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt_horizon as horizon,
)
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt_manifold_readiness as mra,
)
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt_reconstruction_manifold_failure as rmfa,
)
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    tensor_sha256,
)
from scripts.research import replay_scaled_chen_repaired_coordinate_mra as rcr


TASK_ID = "H-LOMA-GEN-S1-OLS-SRA"
HYPOTHESIS = TASK_ID
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_ols_semantic_representation_audit_v1"
AUDIT_SCHEMA = "lightgen_scaled_chen_ols_semantic_representation_audit.v1"
CONTRACT_SCHEMA = "lightgen_scaled_chen_ols_representation_contract.v1"
PHASE_SCHEMA = "lightgen_scaled_chen_ols_phase_gauge_audit.v1"
GEOMETRY_SCHEMA = "lightgen_scaled_chen_ols_geometry_metrics.v1"
KNN_SCHEMA = "lightgen_scaled_chen_ols_knn_metrics.v1"
PROBE_SCHEMA = "lightgen_scaled_chen_ols_linear_probe_metrics.v1"
NULL_SCHEMA = "lightgen_scaled_chen_ols_null_control_metrics.v1"
PHASE_ATTRIBUTION_SCHEMA = "lightgen_scaled_chen_ols_phase_attribution.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_ols_compact_report.v1"
CHECKPOINT_SHA256 = "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de"
CALIBRATION_PAYLOAD_SHA256 = "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07"
CALIBRATION_MANIFEST_SHA256 = "0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6"
MODE_BANK_SHA256 = "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9"
R1_POWER = "R1_POWER"
R2_AMPLITUDE = "R2_AMPLITUDE"
R2_RAW_AMPLITUDE = "R2_RAW_AMPLITUDE"
R3_COMPLEX = "R3_COMPLEX"
S1_POWER_SEMANTICS_SUPPORTED = "S1_POWER_SEMANTICS_SUPPORTED"
S1_AMPLITUDE_SEMANTICS_SUPPORTED = "S1_AMPLITUDE_SEMANTICS_SUPPORTED"
S1_COMPLEX_SEMANTICS_SUPPORTED = "S1_COMPLEX_SEMANTICS_SUPPORTED"
S1_OLS_SEMANTICS_NOT_SUPPORTED = "S1_OLS_SEMANTICS_NOT_SUPPORTED"
S1_SEMANTICS_UNDERDETERMINED = "S1_SEMANTICS_UNDERDETERMINED"
PHASE_SEMANTIC_INCREMENT_SUPPORTED = "PHASE_SEMANTIC_INCREMENT_SUPPORTED"
PHASE_SEMANTIC_INCREMENT_NOT_SUPPORTED = "PHASE_SEMANTIC_INCREMENT_NOT_SUPPORTED"
PHASE_SEMANTIC_INCREMENT_UNRESOLVED = "PHASE_SEMANTIC_INCREMENT_UNRESOLVED"
NEXT_ACTION_SUPPORTED = "PLAN_S2_DIRECT_OLS_SEMANTIC_DECODE_CANARY"
NEXT_ACTION_NOT_SUPPORTED = "REVIEW_OLS_SEMANTIC_REPRESENTATION_FAILURE"
NEXT_ACTION_UNDERDETERMINED = "REVIEW_OLS_SEMANTIC_REPRESENTATION_GAUGE_OR_PHYSICAL_BLOCKER"
PORT_COUNT = 100
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
SELECTION_PER_LABEL = 100
FIT_COUNT = 1000
FIT_PER_LABEL = 100
NULL_SEED = 2053
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
RIDGE_LAMBDA = 1.0
DEFAULT_CONFIG_PATH = Path(
    "configs/research/lightgen_scaled_chen_ols_semantic_representation_audit_v1.yaml"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01"
)
CONFIG_PATH = DEFAULT_CONFIG_PATH
ALLOWED_DIRTY_PATHS = {
    "scripts/research/audit_scaled_chen_ols_semantic_representation.py",
    "tests/v2/test_scaled_chen_ols_semantic_representation_audit.py",
    "configs/research/lightgen_scaled_chen_ols_semantic_representation_audit_v1.yaml",
    "docs/CURRENT_STATUS.md",
    "docs/migration_windows_native_20260830.md",
}


def _resolve(value: str | Path) -> Path:
    return base._resolve_path(value)


def _relative_path(path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return target.relative_to(base._repo_root()).as_posix()
    except ValueError:
        return str(target)


def next_write_once_output(base_path: str | Path) -> Path:
    """Return the first unused numbered attempt without touching the disk."""
    requested = Path(base_path)
    match = re.fullmatch(r"(.+)_attempt_(\d+)", requested.name)
    if match is None:
        raise ValueError("write-once output must end with _attempt_<number>")
    prefix, number_text = match.groups()
    number = int(number_text)
    while True:
        candidate = requested.parent / f"{prefix}_attempt_{number:02d}"
        if not candidate.exists():
            return candidate
        number += 1


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value.lower()


def _assert_config(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "task_id",
        "hypothesis",
        "parent_mra",
        "r2_calibration",
        "checkpoint",
        "dataset",
        "representation",
        "phase_gauge",
        "geometry",
        "linear_probe",
        "null_control",
        "measurement",
        "execution",
        "authorization",
        "governance",
    }
    if set(payload) != required:
        raise ValueError("S1 config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("task_id") != TASK_ID or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("S1 config identity is invalid")

    parent = _require_mapping(payload["parent_mra"], "parent_mra")
    expected_parent = {
        "config": "configs/research/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1.yaml",
        "config_sha256": "95e4587f87fc49e79fe3bef5a930f34502ebc9c95f36a312995d77f993a72cad",
        "config_canonical_sha256": "1a69f77b3e9163b8a756dcaa8c6af039828e90f2b607182b45975f81387f61d9",
        "review_contract": "configs/research/lightgen_optical_manifold_smf_bogt_manifold_readiness_review_v1.json",
        "review_contract_sha256": "3d0c807b0fadf0bb7f6ef0596c5a3279913175b82b792fa6b1015a086b207ac8",
        "output_directory": "outputs/20260831_v2/runs/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05",
        "audit": "outputs/20260831_v2/runs/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05/audit.json",
        "audit_sha256": "6291a2dc03f5ec8d6d3cd9ba88c0bf7f3a4b3f44f83dba7fc16c80375e9dc556",
        "compact_report": "outputs/20260831_v2/runs/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05/compact_report.json",
        "compact_report_sha256": "401fe2159f4342a4740b2c40955cb65fc7bb3600f2416fdc4171e1553eebeee0",
        "panel": "outputs/20260831_v2/runs/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1_attempt_05/panel.pt",
        "panel_sha256": "842f5cfd41909a666f19f3c88e7e7199a825b255bd111a44953e53351fd28724",
        "status": "valid_failed",
        "prior_status": "generation_prior_not_accessible",
    }
    if dict(parent) != expected_parent:
        raise ValueError("registered parent MRA identity is stale")

    repaired = _require_mapping(payload["r2_calibration"], "r2_calibration")
    expected_repaired = {
        "config": "configs/research/lightgen_scaled_chen_bogt_calibration_contract_repair_v2.json",
        "config_sha256": "b8c1efab6fd49c0e68696bc7a466cf6eaeeb499a842c6efa66fe080254b867a9",
        "config_canonical_sha256": "c9de9505a6cf64432e4f2c350bfc5dfa188bc9563cdf8f89c274193123045deb",
        "manifest": "outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/calibration_manifest.json",
        "manifest_sha256": CALIBRATION_MANIFEST_SHA256,
        "payload": "outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/repaired_calibration.pt",
        "payload_sha256": CALIBRATION_PAYLOAD_SHA256,
        "schema": "scaled_chen_bogt_calibration.v2",
        "contract_version": "unit_aware_raw_mean_v2",
        "sample_count": 1000,
        "port_count": PORT_COUNT,
        "calibration_indices_sha256": "369c5bded5673f901ae5236eaa51a0c8da17934dc2fab139a205c9fce41d5543",
        "dataset_manifest_sha256": "741e6402da754a740a677dbb343d48c35983f62b59911bbc129dfcc44f9e9ae9",
        "mode_bank_sha256": MODE_BANK_SHA256,
        "zero_phase_reference": "independent_train_only_zero_phase",
        "plane_identity": "pre_ols_plane_power_reference_v1",
    }
    if dict(repaired) != expected_repaired:
        raise ValueError("registered repaired calibration identity is stale")

    checkpoint = _require_mapping(payload["checkpoint"], "checkpoint")
    expected_checkpoint = {
        "path": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "sha256": CHECKPOINT_SHA256,
        "schema": "scaled_chen_bogt_horizon_checkpoint.v1",
        "global_step": 5000,
    }
    if dict(checkpoint) != expected_checkpoint:
        raise ValueError("registered step-5000 checkpoint identity is stale")

    expected_dataset = {
        "dataset_root": "data",
        "mra_manifest_sha256": "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb",
        "selection_count": SELECTION_COUNT,
        "selection_per_label": SELECTION_PER_LABEL,
        "selection_source": "mnist_test_indices_5000_9999",
        "selection_seed": 1235,
        "confirmation_count": CONFIRMATION_COUNT,
        "confirmation_source": "horizon_heldout_train_indices_55000_59999",
        "confirmation_indices_sha256": "ff7ab50b95d96e727c240edc2baa916d1e6c11402b8f4f9c2ce6e2e7e5e2218a",
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }
    if dict(_require_mapping(payload["dataset"], "dataset")) != expected_dataset:
        raise ValueError("registered dataset identity is stale")

    representation = _require_mapping(payload["representation"], "representation")
    if dict(representation) != {
        "source": "encoder_final_field_to_fixed_ols_modal_projection_output_coefficients",
        "source_tensor_path": "ScaledChenBOGTModel._coefficients(encoded)",
        "source_dtype": "torch.complex64",
        "port_count": PORT_COUNT,
        "representations": [R1_POWER, R2_AMPLITUDE, R2_RAW_AMPLITUDE, R3_COMPLEX],
        "r1_definition": "c_tilde=c/sqrt(s); power=abs(c_tilde)^2=abs(c)^2/s",
        "r2_definition": "amplitude=abs(c_tilde)=abs(c)/sqrt(s)",
        "r2_raw_definition": "descriptive_only_abs(c)",
        "r3_definition": "normalized_complex_c_tilde=c/sqrt(s)",
        "raw_cartesian_primary_geometry": False,
    }:
        raise ValueError("OLS representation contract changed")

    phase = _require_mapping(payload["phase_gauge"], "phase_gauge")
    if dict(phase) != {
        "classification": "GLOBAL_PHASE_GAUGE_CONFIRMED",
        "downstream": "decoder_relay_cmos_intensity",
        "intensity_invariant_under_common_coefficient_phase": True,
        "runtime_decoder_probe": False,
        "primary_complex_metric": "global_phase_gauge_distance",
        "formal_phase_reference": False,
        "raw_cartesian_primary": False,
    }:
        raise ValueError("phase/gauge contract changed")

    geometry = _require_mapping(payload["geometry"], "geometry")
    if dict(geometry) != {
        "primary_knn": 1,
        "descriptive_knn": [3, 5],
        "distance_power": "euclidean",
        "distance_amplitude": "euclidean",
        "distance_complex": "global_phase_gauge",
        "fit_reference_count": FIT_COUNT,
        "fit_reference_per_label": FIT_PER_LABEL,
        "selection_queries": SELECTION_COUNT,
        "confirmation_queries": CONFIRMATION_COUNT,
        "no_self_leakage": True,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }:
        raise ValueError("geometry gate contract changed")

    probe = _require_mapping(payload["linear_probe"], "linear_probe")
    if dict(probe) != {
        "kind": "frozen_multiclass_ridge",
        "lambda": RIDGE_LAMBDA,
        "standardization": "fit_reference_only",
        "selection_queries": SELECTION_COUNT,
        "confirmation_queries": CONFIRMATION_COUNT,
        "complex_probe": "NON_AUTHORITATIVE_WITHOUT_FORMAL_GAUGE_FIXED_VECTOR",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }:
        raise ValueError("linear probe contract changed")

    null_control = _require_mapping(payload["null_control"], "null_control")
    if dict(null_control) != {
        "kind": "one_fixed_label_permutation",
        "seed": NULL_SEED,
        "seed_sweep": False,
        "same_states": True,
        "same_geometry": True,
        "same_class_counts": True,
    }:
        raise ValueError("null control contract changed")

    measurement = _require_mapping(payload["measurement"], "measurement")
    if dict(measurement) != {
        "objective": "S1_OLS_SEMANTIC_REPRESENTATION_AUDIT",
        "purpose": "audit_semantic_information_in_frozen_ols_representation",
        "prefix": "slm_to_front_to_encoder_to_pre_ols_to_complex_coefficients",
        "decoder_entered": False,
        "relay_after_pre_ols": False,
        "cmos_called": False,
        "mpcdar_called": False,
        "optimizer_created": False,
        "backward_called": False,
        "training": False,
        "prior_generation": False,
        "reselection": False,
    }:
        raise ValueError("S1 measurement boundary changed")

    execution = _require_mapping(payload["execution"], "execution")
    if dict(execution) != {
        "device": "cuda",
        "batch_size": 10,
        "min_free_cuda_bytes": 4294967296,
        "max_peak_cuda_bytes": 5368709120,
        "min_disk_free_bytes": 10737418240,
        "max_storage_bytes": 536870912,
        "require_cublas_workspace_config": ":4096:8",
    }:
        raise ValueError("S1 execution contract changed")

    authorization = _require_mapping(payload["authorization"], "authorization")
    if dict(authorization) != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal_promotion": False,
        "semantic_generation_authorized": False,
        "optimizer_created": False,
        "backward_called": False,
        "checkpoint_created": False,
        "prohibited": [
            "training",
            "optimizer",
            "backward",
            "phase_update",
            "prior_alignment",
            "prior_generation",
            "decoder_as_semantic_input",
            "cmos_as_semantic_input",
            "mpcdar_as_semantic_input",
            "formal_promotion",
        ],
    }:
        raise ValueError("S1 authorization boundary changed")

    governance = _require_mapping(payload["governance"], "governance")
    if dict(governance) != {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }:
        raise ValueError("S1 governance boundary changed")


def load_config(path: str | Path = CONFIG_PATH) -> tuple[Path, dict[str, Any], str, str]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("S1 config must be a YAML mapping")
    _assert_config(payload)
    return target, payload, file_sha256(target), canonical_sha256(payload)

def validate_amplitude_scale(scale: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(scale) or scale.ndim != 1 or scale.shape[0] != PORT_COUNT:
        raise ValueError("amplitude scale must be a [100] tensor")
    if not torch.is_floating_point(scale) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("amplitude scale must be finite, floating point, and positive")
    return torch.sqrt(scale)


def _validate_coefficients(coefficients: torch.Tensor) -> None:
    if coefficients.ndim != 2 or coefficients.shape[1] != PORT_COUNT or not torch.is_complex(coefficients):
        raise ValueError("OLS coefficients must be complex [N,100]")
    if not torch.isfinite(coefficients).all():
        raise ValueError("OLS coefficients must be finite")


def normalized_representations(
    coefficients: torch.Tensor, scale: torch.Tensor
) -> dict[str, torch.Tensor]:
    _validate_coefficients(coefficients)
    sqrt_scale = validate_amplitude_scale(scale).to(
        device=coefficients.device, dtype=coefficients.real.dtype
    )
    normalized = coefficients / sqrt_scale[None, :]
    result = {
        R1_POWER: normalized.abs().square(),
        R2_AMPLITUDE: normalized.abs(),
        R2_RAW_AMPLITUDE: coefficients.abs(),
        R3_COMPLEX: normalized,
    }
    if not torch.allclose(result[R1_POWER], result[R2_AMPLITUDE].square(), rtol=1.0e-6, atol=1.0e-8):
        raise ValueError("R1 power is not the square of R2 normalized amplitude")
    if not torch.allclose(result[R1_POWER], result[R3_COMPLEX].abs().square(), rtol=1.0e-6, atol=1.0e-8):
        raise ValueError("R1 power is not the squared modulus of R3")
    return {name: value.detach().cpu().contiguous() for name, value in result.items()}


def gauge_distance_squared(query: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    _validate_coefficients(query)
    _validate_coefficients(support)
    query128 = query.to(torch.complex128)
    support128 = support.to(torch.complex128)
    query_norm = query128.abs().square().sum(dim=1, keepdim=True)
    support_norm = support128.abs().square().sum(dim=1).unsqueeze(0)
    overlap = query128.conj() @ support128.T
    distance = query_norm + support_norm - 2.0 * overlap.abs()
    scale = torch.maximum(query_norm.max(), support_norm.max()).clamp_min(1.0)
    tolerance = 1.0e-10 * scale
    if distance.min() < -tolerance:
        raise ValueError("global phase gauge distance became materially negative")
    return distance.clamp_min(0.0).to(torch.float64)


def phase_gauge_audit(coefficients: torch.Tensor) -> dict[str, Any]:
    _validate_coefficients(coefficients)
    phase = torch.exp(
        1j * torch.linspace(
            -0.73, 0.91, coefficients.shape[0], dtype=torch.float64
        )
    ).to(coefficients)
    rotated = coefficients * phase[:, None]
    power_invariant = torch.allclose(
        coefficients.abs().square(), rotated.abs().square(), rtol=1.0e-6, atol=1.0e-8
    )
    distance_invariant = torch.allclose(
        gauge_distance_squared(coefficients, coefficients),
        gauge_distance_squared(rotated, rotated),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    if not power_invariant or not distance_invariant:
        raise ValueError("global phase gauge invariance failed")
    original_distance = gauge_distance_squared(coefficients, coefficients)
    return {
        "schema": PHASE_SCHEMA,
        "classification": "GLOBAL_PHASE_GAUGE_CONFIRMED",
        "power_invariant": True,
        "distance_invariant": True,
        "self_distance_zero": bool(torch.allclose(torch.diag(original_distance), torch.zeros(coefficients.shape[0], dtype=torch.float64), atol=1.0e-7)),
        "symmetric": bool(torch.allclose(original_distance, original_distance.T, atol=1.0e-7)),
        "downstream": "decoder_relay_cmos_intensity",
        "intensity_invariant_under_common_coefficient_phase": True,
        "runtime_decoder_probe": False,
        "audit_method": "common_phase_cancellation_in_complex_field_intensity",
        "primary_complex_metric": "global_phase_gauge_distance",
        "formal_phase_reference": False,
        "raw_cartesian_primary": False,
    }


def _distance_matrix(query: torch.Tensor, support: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "global_phase_gauge":
        return gauge_distance_squared(query, support)
    if metric != "euclidean":
        raise ValueError(f"unsupported distance metric: {metric}")
    if query.ndim != 2 or support.ndim != 2 or query.shape[1] != support.shape[1]:
        raise ValueError("Euclidean features must be two-dimensional and aligned")
    if torch.is_complex(query) or torch.is_complex(support):
        raise ValueError("complex Euclidean distance is not a primary S1 metric")
    return torch.cdist(query.to(torch.float64), support.to(torch.float64), p=2).square()


def one_nn_metrics(
    query: torch.Tensor,
    support: torch.Tensor,
    query_labels: torch.Tensor,
    support_labels: torch.Tensor,
    *,
    metric: str,
    exclude_self: bool = False,
) -> dict[str, Any]:
    if query_labels.ndim != 1 or support_labels.ndim != 1 or query.shape[0] != query_labels.shape[0] or support.shape[0] != support_labels.shape[0]:
        raise ValueError("1-NN labels do not align with feature banks")
    if exclude_self and (query.shape[0] != support.shape[0] or not torch.equal(query_labels, support_labels)):
        raise ValueError("self exclusion requires identical query/support panels")
    distances = _distance_matrix(query, support, metric)
    if exclude_self:
        distances.fill_diagonal_(float("inf"))
    nearest_distances, nearest_indices = distances.min(dim=1)
    predictions = support_labels.index_select(0, nearest_indices)
    correct = predictions == query_labels
    classes = torch.unique(torch.cat((query_labels, support_labels))).sort().values.tolist()
    confusion = []
    per_class: dict[str, float] = {}
    for true_label in classes:
        row = []
        true_mask = query_labels == true_label
        for predicted_label in classes:
            row.append(int(((predictions == predicted_label) & true_mask).sum()))
        confusion.append(row)
        per_class[str(true_label)] = float(correct[true_mask].to(torch.float64).mean()) if true_mask.any() else float("nan")
    return {
        "metric": metric,
        "sample_count": int(query.shape[0]),
        "self_excluded": exclude_self,
        "overall_accuracy": float(correct.to(torch.float64).mean()),
        "per_class_accuracy": per_class,
        "class_order": [int(value) for value in classes],
        "confusion": confusion,
        "nearest_indices": nearest_indices.cpu().tolist(),
        "nearest_distances_squared": nearest_distances.detach().cpu().tolist(),
        "correct": correct.cpu().tolist(),
    }


def fixed_label_permutation(labels: torch.Tensor, *, seed: int = NULL_SEED) -> torch.Tensor:
    if labels.ndim != 1 or labels.numel() == 0 or labels.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise ValueError("null labels must be a non-empty integer vector")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return labels.index_select(0, torch.randperm(labels.numel(), generator=generator))


def stratified_bootstrap_ci(
    real: torch.Tensor,
    shuffled: torch.Tensor,
    labels: torch.Tensor,
    *,
    reps: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    if real.ndim != 1 or shuffled.ndim != 1 or labels.ndim != 1 or not (real.shape == shuffled.shape == labels.shape) or real.numel() == 0:
        raise ValueError("stratified bootstrap inputs must be aligned vectors")
    if reps < 100 or labels.unique().numel() < 2:
        raise ValueError("stratified bootstrap requires at least two classes and 100 replicates")
    real64 = real.to(torch.float64)
    shuffled64 = shuffled.to(torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total = torch.zeros(reps, dtype=torch.float64)
    total_count = float(labels.numel())
    for label in labels.unique(sorted=True):
        positions = torch.nonzero(labels == label, as_tuple=False).flatten()
        draws = torch.randint(positions.numel(), (reps, positions.numel()), generator=generator)
        sampled = positions.index_select(0, draws.reshape(-1)).reshape(reps, -1)
        delta = (real64 - shuffled64).index_select(0, sampled.reshape(-1)).reshape(reps, -1)
        total += delta.mean(dim=1) * (positions.numel() / total_count)
    point = float((real64 - shuffled64).mean())
    quantiles = torch.quantile(total, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "point_delta": point,
        "lower95": float(quantiles[0]),
        "upper95": float(quantiles[1]),
        "replicates": reps,
        "seed": seed,
        "stratified": True,
    }


def fit_ridge_probe(
    features: torch.Tensor, labels: torch.Tensor, *, ridge_lambda: float = RIDGE_LAMBDA
) -> dict[str, Any]:
    if features.ndim != 2 or features.shape[0] != labels.shape[0] or features.shape[0] < 2 or torch.is_complex(features) or not torch.isfinite(features).all():
        raise ValueError("ridge fit requires finite real [N,D] features")
    if labels.ndim != 1 or labels.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8) or labels.min() < 0:
        raise ValueError("ridge labels must be non-negative integers")
    if ridge_lambda <= 0.0 or not math.isfinite(ridge_lambda):
        raise ValueError("ridge lambda must be positive and finite")
    x = features.to(torch.float64)
    y = labels.to(torch.long)
    mean = x.mean(dim=0)
    std = x.std(dim=0, unbiased=False)
    std = torch.where(std > 1.0e-12, std, torch.ones_like(std))
    standardized = (x - mean) / std
    design = torch.cat((standardized, torch.ones((x.shape[0], 1), dtype=torch.float64)), dim=1)
    class_count = int(y.max().item()) + 1
    target = torch.nn.functional.one_hot(y, num_classes=class_count).to(torch.float64)
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * ridge_lambda
    regularizer[-1, -1] = 0.0
    weights = torch.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return {
        "weights": weights,
        "mean": mean,
        "std": std,
        "class_count": class_count,
        "feature_dim": int(features.shape[1]),
        "ridge_lambda": ridge_lambda,
        "standardization_source": "fit_reference_only",
    }


def predict_ridge_probe(probe: Mapping[str, Any], features: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or features.shape[1] != int(probe["feature_dim"]) or torch.is_complex(features) or not torch.isfinite(features).all():
        raise ValueError("ridge prediction features do not match the fit contract")
    x = features.to(torch.float64)
    standardized = (x - probe["mean"]) / probe["std"]
    design = torch.cat((standardized, torch.ones((x.shape[0], 1), dtype=torch.float64)), dim=1)
    return (design @ probe["weights"]).argmax(dim=1).to(torch.long)


def representation_gate(
    *,
    geometry_confirmation_lower95: float,
    geometry_selection_delta: float,
    probe_confirmation_lower95: float | None,
    probe_selection_delta: float | None,
    physical_contract_valid: bool,
    linear_probe_authoritative: bool,
) -> dict[str, Any]:
    geometry_confirmation = geometry_confirmation_lower95 > 0.0
    geometry_selection = geometry_selection_delta > 0.0
    if linear_probe_authoritative:
        probe_confirmation = probe_confirmation_lower95 is not None and probe_confirmation_lower95 > 0.0
        probe_selection = probe_selection_delta is not None and probe_selection_delta > 0.0
    else:
        probe_confirmation = True
        probe_selection = True
    supported = bool(
        physical_contract_valid
        and geometry_confirmation
        and geometry_selection
        and probe_confirmation
        and probe_selection
    )
    return {
        "physical_contract_valid": physical_contract_valid,
        "geometry_confirmation_lower95_positive": geometry_confirmation,
        "geometry_selection_direction_positive": geometry_selection,
        "probe_confirmation_lower95_positive": probe_confirmation,
        "probe_selection_direction_positive": probe_selection,
        "linear_probe_authoritative": linear_probe_authoritative,
        "linear_accessibility_unresolved": not linear_probe_authoritative,
        "supported": supported,
    }


def minimal_representation(results: Mapping[str, Mapping[str, Any]]) -> str:
    mapping = (
        (R1_POWER, S1_POWER_SEMANTICS_SUPPORTED),
        (R2_AMPLITUDE, S1_AMPLITUDE_SEMANTICS_SUPPORTED),
        (R3_COMPLEX, S1_COMPLEX_SEMANTICS_SUPPORTED),
    )
    for name, classification in mapping:
        if bool(results.get(name, {}).get("supported", False)):
            return classification
    return S1_OLS_SEMANTICS_NOT_SUPPORTED


def classify_s1(
    results: Mapping[str, Mapping[str, Any]],
    *,
    measurement_valid: bool,
    physical_contract_valid: bool,
) -> str:
    if not measurement_valid or not physical_contract_valid:
        return S1_SEMANTICS_UNDERDETERMINED
    return minimal_representation(results)


def phase_increment_classification(
    amplitude_selection_correct: torch.Tensor,
    complex_selection_correct: torch.Tensor,
    amplitude_confirmation_correct: torch.Tensor,
    complex_confirmation_correct: torch.Tensor,
    labels: torch.Tensor,
    *,
    confirmation_labels: torch.Tensor | None = None,
    reps: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if confirmation_labels is None:
        confirmation_labels = labels
    selection = stratified_bootstrap_ci(
        complex_selection_correct,
        amplitude_selection_correct,
        labels,
        reps=reps,
        seed=seed,
    )
    confirmation = stratified_bootstrap_ci(
        complex_confirmation_correct,
        amplitude_confirmation_correct,
        confirmation_labels,
        reps=reps,
        seed=seed,
    )
    if selection["point_delta"] > 0.0 and confirmation["lower95"] > 0.0:
        classification = PHASE_SEMANTIC_INCREMENT_SUPPORTED
    else:
        classification = PHASE_SEMANTIC_INCREMENT_NOT_SUPPORTED
    return {
        "schema": PHASE_ATTRIBUTION_SCHEMA,
        "classification": classification,
        "paired_same_query_reference_identities": True,
        "selection": selection,
        "confirmation": confirmation,
        "changes_minimal_representation": False,
    }


def _validate_registered_indices(
    data: mra.AuditData, config: Mapping[str, Any]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    horizon_data = horizon._load_data(_resolve(config["dataset"]["dataset_root"]))
    fit_indices = rmfa._fit_indices(
        data,
        horizon_data,
        rmfa._load_plan_indices(rmfa.PLAN3),
        rmfa._load_plan_indices(rmfa.PLAN4),
    )
    selection_indices = data.selection_indices
    confirmation_indices = data.confirmation_indices
    dataset = config["dataset"]
    if data.manifest.get("manifest_hash") != dataset["mra_manifest_sha256"]:
        raise ValueError("S1 dataset manifest hash changed")
    if data.manifest.get("confirmation_indices_sha256") != dataset["confirmation_indices_sha256"]:
        raise ValueError("S1 confirmation indices hash changed")
    if fit_indices.numel() != FIT_COUNT or tensor_sha256(fit_indices) != rmfa.FIT_PANEL["indices_sha256"]:
        raise ValueError("S1 fit panel identity changed")
    if selection_indices.numel() != SELECTION_COUNT or confirmation_indices.numel() != CONFIRMATION_COUNT:
        raise ValueError("S1 evaluation panel count changed")
    fit_labels = data.labels.index_select(0, fit_indices)
    selection_labels = data.test_labels.index_select(0, selection_indices)
    confirmation_labels = data.labels.index_select(0, confirmation_indices)
    expected_counts = torch.full((10,), FIT_PER_LABEL, dtype=torch.long)
    if not torch.equal(torch.bincount(fit_labels, minlength=10), expected_counts):
        raise ValueError("S1 fit panel is not balanced")
    if not torch.equal(torch.bincount(selection_labels, minlength=10), expected_counts):
        raise ValueError("S1 selection panel is not balanced")
    if torch.unique(fit_indices).numel() != fit_indices.numel() or torch.unique(selection_indices).numel() != selection_indices.numel() or torch.unique(confirmation_indices).numel() != confirmation_indices.numel():
        raise ValueError("S1 panel indices are not unique")
    if torch.isin(fit_indices, confirmation_indices).any():
        raise ValueError("S1 fit and confirmation panels overlap")
    return fit_indices, selection_indices, confirmation_indices, {
        "fit_labels": fit_labels,
        "selection_labels": selection_labels,
        "confirmation_labels": confirmation_labels,
    }


def _geometry_descriptive(
    features: torch.Tensor, labels: torch.Tensor, *, metric: str
) -> dict[str, Any]:
    distances = _distance_matrix(features, features, metric)
    upper = torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)
    same = labels[:, None] == labels[None, :]
    within = distances[upper & same]
    between = distances[upper & ~same]
    return {
        "metric": metric,
        "within_class_mean_distance_squared": float(within.mean()),
        "between_class_mean_distance_squared": float(between.mean()),
        "between_to_within_ratio": float(between.mean() / within.mean().clamp_min(1.0e-20)),
        "finite": bool(torch.isfinite(distances).all()),
    }


def _correct_tensor(metrics: Mapping[str, Any]) -> torch.Tensor:
    value = metrics.get("correct")
    if not isinstance(value, list):
        raise ValueError("NN metrics do not expose correctness")
    return torch.tensor(value, dtype=torch.float64)


def _probe_split(
    true_probe: Mapping[str, Any],
    null_probe: Mapping[str, Any],
    features: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, Any]:
    true_correct = predict_ridge_probe(true_probe, features) == labels
    null_correct = predict_ridge_probe(null_probe, features) == labels
    ci = stratified_bootstrap_ci(
        true_correct.to(torch.float64),
        null_correct.to(torch.float64),
        labels,
    )
    return {
        "real_accuracy": float(true_correct.to(torch.float64).mean()),
        "null_accuracy": float(null_correct.to(torch.float64).mean()),
        "delta": ci,
        "real_correct": true_correct.cpu().tolist(),
        "null_correct": null_correct.cpu().tolist(),
    }


def _run_probe(
    name: str,
    representations: Mapping[str, Mapping[str, torch.Tensor]],
    labels: Mapping[str, torch.Tensor],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = representations["fit"][name]
    selection = representations["selection"][name]
    confirmation = representations["confirmation"][name]
    if torch.is_complex(fit):
        return {
            "status": "NON_AUTHORITATIVE",
            "reason": "no formal gauge-fixed complex vector is registered",
            "linear_accessibility": "UNRESOLVED",
        }, {}
    true_probe = fit_ridge_probe(fit, labels["fit"])
    null_labels = fixed_label_permutation(labels["fit"], seed=NULL_SEED)
    null_probe = fit_ridge_probe(fit, null_labels)
    result = {
        "status": "AUTHORITATIVE",
        "representation": name,
        "ridge_lambda": RIDGE_LAMBDA,
        "standardization_source": "fit_reference_only",
        "fit_sample_count": int(fit.shape[0]),
        "feature_dim": int(fit.shape[1]),
        "selection": _probe_split(true_probe, null_probe, selection, labels["selection"]),
        "confirmation": _probe_split(true_probe, null_probe, confirmation, labels["confirmation"]),
        "majority_accuracy": float(torch.bincount(labels["fit"], minlength=10).max() / labels["fit"].numel()),
        "empirical_chance_accuracy": 0.1,
    }
    return result, {
        "selection_real_correct": torch.tensor(result["selection"]["real_correct"], dtype=torch.float64),
        "selection_null_correct": torch.tensor(result["selection"]["null_correct"], dtype=torch.float64),
        "confirmation_real_correct": torch.tensor(result["confirmation"]["real_correct"], dtype=torch.float64),
        "confirmation_null_correct": torch.tensor(result["confirmation"]["null_correct"], dtype=torch.float64),
    }


def _artifact_refs(output_path: Path) -> dict[str, dict[str, Any]]:
    return {
        target.relative_to(output_path).as_posix(): {
            "path": target.relative_to(output_path).as_posix(),
            "sha256": file_sha256(target),
            "size_bytes": target.stat().st_size,
        }
        for target in sorted(output_path.iterdir(), key=lambda item: item.name)
        if target.is_file() and target.name != "semantic_manifest.json"
    }


def _assert_output_tree(output_path: Path, *, include_manifest: bool = True) -> None:
    expected = {
        "semantic_manifest.json",
        "ols_representation_contract.json",
        "phase_gauge_audit.json",
        "geometry_metrics.json",
        "knn_metrics.json",
        "linear_probe_metrics.json",
        "null_control_metrics.json",
        "phase_attribution.json",
        "compact_report.json",
        "identity_manifest.json",
        "ols_embeddings.pt",
    }
    if not include_manifest:
        expected.remove("semantic_manifest.json")
    actual = {
        target.relative_to(output_path).as_posix()
        for target in output_path.rglob("*")
        if target.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"S1 output tree is not exact: {sorted(actual)}")
    if any(
        Path(name).name.startswith(".")
        or ".tmp" in name
        or name.lower().endswith((".tif", ".tiff"))
        for name in actual
    ):
        raise RuntimeError("S1 output contains hidden, temporary, or TIFF files")


def _resource_preflight(config: Mapping[str, Any], output_path: Path) -> tuple[int, int, int]:
    execution = config["execution"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != execution["require_cublas_workspace_config"]:
        raise RuntimeError("BLOCKED_AMPLITUDE_CALIBRATION_CONTRACT: CUBLAS_WORKSPACE_CONFIG is not registered")
    device = torch.device(execution["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_UNAVAILABLE")
    free, total = torch.cuda.mem_get_info(device)
    if free < execution["min_free_cuda_bytes"]:
        raise RuntimeError("BLOCKED_S1_RESOURCE: less than 4 GiB free CUDA memory")
    disk = shutil.disk_usage(base._repo_root().anchor or str(base._repo_root()))
    if disk.free < execution["min_disk_free_bytes"]:
        raise RuntimeError("BLOCKED_S1_RESOURCE: less than 10 GiB free disk")
    if output_path.exists():
        raise FileExistsError(f"refusing to reuse write-once output: {_relative_path(output_path)}")
    return int(free), int(total), int(disk.free)


def _git_preflight() -> dict[str, Any]:
    state = base._git_state()
    if state.get("branch") != BRANCH:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_BRANCH: expected {BRANCH}")
    unexpected = sorted(
        str(line[3:]).replace("\\", "/")
        for line in state.get("status", [])
        if str(line[3:]).replace("\\", "/") not in ALLOWED_DIRTY_PATHS
    )
    if unexpected:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_DIRTY_PATH: {unexpected}")
    return state


def _summary_geometry(
    name: str,
    representations: Mapping[str, Mapping[str, torch.Tensor]],
    labels: Mapping[str, torch.Tensor],
    null_labels: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    metric = "global_phase_gauge" if name == R3_COMPLEX else "euclidean"
    descriptive = {
        split: _geometry_descriptive(representations[split][name], labels[split], metric=metric)
        for split in ("fit",)
    }
    support = representations["fit"][name]
    result: dict[str, Any] = {
        "representation": name,
        "metric": metric,
        "primary_k": 1,
        "splits": {},
    }
    for split in ("selection", "confirmation"):
        real = one_nn_metrics(
            representations[split][name],
            support,
            labels[split],
            labels["fit"],
            metric=metric,
        )
        shuffled = one_nn_metrics(
            representations[split][name],
            support,
            labels[split],
            null_labels,
            metric=metric,
        )
        real_correct = _correct_tensor(real)
        null_correct = _correct_tensor(shuffled)
        result["splits"][split] = {
            "real": real,
            "shuffled": shuffled,
            "delta": stratified_bootstrap_ci(real_correct, null_correct, labels[split]),
        }
    gate = representation_gate(
        geometry_confirmation_lower95=result["splits"]["confirmation"]["delta"]["lower95"],
        geometry_selection_delta=result["splits"]["selection"]["delta"]["point_delta"],
        probe_confirmation_lower95=None,
        probe_selection_delta=None,
        physical_contract_valid=True,
        linear_probe_authoritative=False,
    )
    result["geometry_gate"] = gate
    return descriptive, result, {
        "selection_real_correct": _correct_tensor(result["splits"]["selection"]["real"]),
        "selection_shuffled_correct": _correct_tensor(result["splits"]["selection"]["shuffled"]),
        "confirmation_real_correct": _correct_tensor(result["splits"]["confirmation"]["real"]),
        "confirmation_shuffled_correct": _correct_tensor(result["splits"]["confirmation"]["shuffled"]),
    }


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    state_before = _git_preflight()
    config_target, config, config_file_sha, config_canonical_sha = load_config(config_path)
    parent = rcr._validate_parent_mra(config)
    data = mra._load_data(_resolve(config["dataset"]["dataset_root"]))
    rcr._validate_parent_panel(parent["panel"], data, parent["source"])
    repaired = rcr._validate_r2(config)
    checkpoint, checkpoint_path = rcr._load_checkpoint(config, parent["source"])
    if file_sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("step-5000 checkpoint hash changed")
    if repaired["payload_sha256"] != CALIBRATION_PAYLOAD_SHA256:
        raise ValueError("repaired calibration payload hash changed")
    fit_indices, selection_indices, confirmation_indices, label_banks = _validate_registered_indices(data, config)
    calibration = repaired["calibration"]
    scale = calibration.port_scale.detach().cpu().contiguous()
    validate_amplitude_scale(scale)
    if not torch.equal(scale, calibration.raw_port_mean.detach().cpu()):
        raise ValueError("BLOCKED_AMPLITUDE_CALIBRATION_CONTRACT: registered scale is not raw mean")

    requested_output = _resolve(output_path)
    actual_output = next_write_once_output(requested_output)
    free_before, total_cuda, disk_free_before = _resource_preflight(config, actual_output)
    device = torch.device(config["execution"]["device"])
    model = mra._load_scaled_model(parent["formal"], checkpoint, device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("S1 model is not frozen")
    if bogt.mode_bank_sha256(model) != MODE_BANK_SHA256:
        raise ValueError("S1 fixed OLS mode-bank identity changed")
    state_digest_before = rcr.state_digest(model)
    torch.cuda.reset_peak_memory_stats(device)
    fit_coefficients = rcr._collect_coefficients(
        model,
        data.images,
        fit_indices,
        device=device,
        batch_size=config["execution"]["batch_size"],
    )
    selection_coefficients = rcr._collect_coefficients(
        model,
        data.test_images,
        selection_indices,
        device=device,
        batch_size=config["execution"]["batch_size"],
    )
    confirmation_coefficients = rcr._collect_coefficients(
        model,
        data.images,
        confirmation_indices,
        device=device,
        batch_size=config["execution"]["batch_size"],
    )
    state_digest_after = rcr.state_digest(model)
    peak_cuda = int(torch.cuda.max_memory_allocated(device))
    if peak_cuda > config["execution"]["max_peak_cuda_bytes"]:
        raise RuntimeError("BLOCKED_S1_RESOURCE: peak CUDA allocation exceeded 5 GiB")
    if state_digest_before != state_digest_after:
        raise RuntimeError("S1 frozen forward changed learned state")

    coefficient_banks = {
        "fit": fit_coefficients,
        "selection": selection_coefficients,
        "confirmation": confirmation_coefficients,
    }
    representations = {
        split: normalized_representations(coefficients, scale)
        for split, coefficients in coefficient_banks.items()
    }
    labels = {
        "fit": label_banks["fit_labels"].detach().cpu(),
        "selection": label_banks["selection_labels"].detach().cpu(),
        "confirmation": label_banks["confirmation_labels"].detach().cpu(),
    }
    null_labels = fixed_label_permutation(labels["fit"], seed=NULL_SEED)
    phase_audit = phase_gauge_audit(fit_coefficients[: min(128, fit_coefficients.shape[0])])
    geometry_payload: dict[str, Any] = {
        "schema": GEOMETRY_SCHEMA,
        "descriptive": {},
        "representations": {},
        "contract": {
            "within_between": "fit_reference_only",
            "primary_gate": "confirmation_lower95_delta_gt_0_and_selection_direction_gt_0",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }
    knn_payload: dict[str, Any] = {"schema": KNN_SCHEMA, "representations": {}}
    geometry_correct: dict[str, dict[str, torch.Tensor]] = {}
    for name in (R1_POWER, R2_AMPLITUDE, R3_COMPLEX):
        descriptive, knn, correctness = _summary_geometry(name, representations, labels, null_labels)
        geometry_payload["descriptive"][name] = descriptive
        geometry_payload["representations"][name] = knn["geometry_gate"]
        knn_payload["representations"][name] = knn
        geometry_correct[name] = correctness

    probe_payload: dict[str, Any] = {"schema": PROBE_SCHEMA, "representations": {}}
    probe_correct: dict[str, dict[str, torch.Tensor]] = {}
    for name in (R1_POWER, R2_AMPLITUDE, R3_COMPLEX):
        probe, correctness = _run_probe(name, representations, labels)
        probe_payload["representations"][name] = probe
        if correctness:
            probe_correct[name] = correctness
    for name in (R1_POWER, R2_AMPLITUDE):
        probe = probe_payload["representations"][name]
        probe_gate = representation_gate(
            geometry_confirmation_lower95=knn_payload["representations"][name]["splits"]["confirmation"]["delta"]["lower95"],
            geometry_selection_delta=knn_payload["representations"][name]["splits"]["selection"]["delta"]["point_delta"],
            probe_confirmation_lower95=probe["confirmation"]["delta"]["lower95"],
            probe_selection_delta=probe["selection"]["delta"]["point_delta"],
            physical_contract_valid=True,
            linear_probe_authoritative=True,
        )
        geometry_payload["representations"][name] = probe_gate
    for name in (R3_COMPLEX,):
        geometry_payload["representations"][name] = representation_gate(
            geometry_confirmation_lower95=knn_payload["representations"][name]["splits"]["confirmation"]["delta"]["lower95"],
            geometry_selection_delta=knn_payload["representations"][name]["splits"]["selection"]["delta"]["point_delta"],
            probe_confirmation_lower95=None,
            probe_selection_delta=None,
            physical_contract_valid=True,
            linear_probe_authoritative=False,
        )

    representation_results = {
        name: geometry_payload["representations"][name]
        for name in (R1_POWER, R2_AMPLITUDE, R3_COMPLEX)
    }
    final_classification = minimal_representation(representation_results)
    phase_increment = phase_increment_classification(
        geometry_correct[R2_AMPLITUDE]["selection_real_correct"],
        geometry_correct[R3_COMPLEX]["selection_real_correct"],
        geometry_correct[R2_AMPLITUDE]["confirmation_real_correct"],
        geometry_correct[R3_COMPLEX]["confirmation_real_correct"],
        labels["selection"],
        confirmation_labels=labels["confirmation"],
    )
    null_payload = {
        "schema": NULL_SCHEMA,
        "kind": "one_fixed_label_permutation",
        "seed": NULL_SEED,
        "seed_sweep": False,
        "fit_reference_label_sha256": tensor_sha256(labels["fit"]),
        "permuted_fit_reference_label_sha256": tensor_sha256(null_labels),
        "class_counts_equal": bool(torch.equal(torch.bincount(labels["fit"], minlength=10), torch.bincount(null_labels, minlength=10))),
        "same_states_and_geometry": True,
        "representations": {
            name: {
                split: {
                    "real_accuracy": knn_payload["representations"][name]["splits"][split]["real"]["overall_accuracy"],
                    "shuffled_accuracy": knn_payload["representations"][name]["splits"][split]["shuffled"]["overall_accuracy"],
                    "delta": knn_payload["representations"][name]["splits"][split]["delta"],
                }
                for split in ("selection", "confirmation")
            }
            for name in (R1_POWER, R2_AMPLITUDE, R3_COMPLEX)
        },
    }
    identity_payload = {
        "schema": "lightgen_scaled_chen_ols_semantic_identity_manifest.v1",
        "fit": {
            "source": rmfa.FIT_PANEL["source"],
            "count": FIT_COUNT,
            "per_label": FIT_PER_LABEL,
            "indices_sha256": tensor_sha256(fit_indices),
            "labels_sha256": tensor_sha256(labels["fit"]),
        },
        "selection": {
            "source": "mnist_test_indices_5000_9999",
            "count": SELECTION_COUNT,
            "per_label": SELECTION_PER_LABEL,
            "indices_sha256": tensor_sha256(selection_indices),
            "labels_sha256": tensor_sha256(labels["selection"]),
        },
        "confirmation": {
            "source": "horizon_heldout_train_indices_55000_59999",
            "count": CONFIRMATION_COUNT,
            "indices_sha256": tensor_sha256(confirmation_indices),
            "labels_sha256": tensor_sha256(labels["confirmation"]),
        },
        "dataset_manifest_sha256": data.manifest["manifest_hash"],
        "reselection": False,
        "self_leakage": False,
    }
    contract_payload = {
        "schema": CONTRACT_SCHEMA,
        "source_tensor": {
            "path": "ScaledChenBOGTModel._coefficients(encoded)",
            "semantic_field": "output.coefficients",
            "stage": "encoder_final_field_to_fixed_ols_modal_projection",
            "shape": ["N", PORT_COUNT],
            "dtype": "torch.complex64",
            "mode_bank_sha256": MODE_BANK_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        },
        "calibration": {
            "schema": "scaled_chen_bogt_calibration.v2",
            "contract_version": "unit_aware_raw_mean_v2",
            "payload_sha256": CALIBRATION_PAYLOAD_SHA256,
            "port_scale_field": "s_j",
            "sqrt_scale_validation": True,
            "no_refit": True,
            "no_label_scaling": True,
        },
        "representations": {
            R1_POWER: "abs(c/sqrt(s))^2=abs(c)^2/s",
            R2_AMPLITUDE: "abs(c/sqrt(s))",
            R2_RAW_AMPLITUDE: "abs(c), descriptive only",
            R3_COMPLEX: "c/sqrt(s), gauge-aware complex geometry",
        },
        "complex_primary_geometry": "global_phase_gauge_distance",
        "raw_cartesian_primary_geometry": False,
        "decoder_relay_cmos_as_semantic_input": False,
    }
    embeddings_payload = {
        "schema": "lightgen_scaled_chen_ols_semantic_embeddings.v1",
        "representations": representations,
        "labels": labels,
        "indices": {
            "fit": fit_indices,
            "selection": selection_indices,
            "confirmation": confirmation_indices,
        },
    }
    compact_classification = final_classification
    if final_classification == S1_OLS_SEMANTICS_NOT_SUPPORTED:
        next_action = NEXT_ACTION_NOT_SUPPORTED
        semantic_gate = "NOT_SUPPORTED"
    elif final_classification == S1_SEMANTICS_UNDERDETERMINED:
        next_action = NEXT_ACTION_UNDERDETERMINED
        semantic_gate = "UNDETERMINED"
    else:
        next_action = NEXT_ACTION_SUPPORTED
        semantic_gate = "SUPPORTED"
    state_after = base._git_state()
    manifest_payload = {
        "schema": AUDIT_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "valid_completed",
        "gate": compact_classification,
        "ols_contains_label_semantic_information": semantic_gate,
        "minimal_representation": compact_classification,
        "phase_attribution": phase_increment["classification"],
        "next_action": next_action,
        "configuration": {
            "path": _relative_path(config_target),
            "sha256": config_file_sha,
            "canonical_sha256": config_canonical_sha,
        },
        "checkpoint": {
            "path": _relative_path(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
            "global_step": 5000,
        },
        "calibration": {
            "payload_path": _relative_path(repaired["payload_path"]),
            "payload_sha256": repaired["payload_sha256"],
            "manifest_sha256": repaired["manifest_sha256"],
            "contract_version": "unit_aware_raw_mean_v2",
        },
        "mode_bank_sha256": MODE_BANK_SHA256,
        "data_identity": identity_payload,
        "source": contract_payload["source_tensor"],
        "phase_gauge": phase_audit,
        "state_isolation": {
            "model_state_digest_before": state_digest_before,
            "model_state_digest_after": state_digest_after,
            "state_unchanged": state_digest_before == state_digest_after,
            "model_training": False,
            "optimizer_created": False,
            "backward_called": False,
            "phase_updated": False,
            "training_steps": 0,
            "learned_checkpoint_created": False,
            "decoder_entered": False,
            "relay_after_pre_ols": False,
            "cmos_called": False,
            "mpcdar_called": False,
        },
        "resource": {
            "free_cuda_bytes_before": free_before,
            "total_cuda_bytes": total_cuda,
            "peak_cuda_bytes": peak_cuda,
            "free_disk_bytes_before": disk_free_before,
            "wall_clock_seconds": time.perf_counter() - started,
        },
        "authorization": {
            "diagnostic_only": True,
            "training_authorized": False,
            "formal_promotion": False,
            "semantic_generation_authorized": False,
        },
        "governance": config["governance"],
        "git": {
            "branch": state_before["branch"],
            "starting_head": state_before["commit"],
            "ending_head": state_after["commit"],
            "unexpected_dirty_paths": [],
        },
        "layout": {
            "write_once": True,
            "manifest_self_hash_excluded": True,
            "artifacts": "relative_paths_with_sha256_and_size_bytes",
        },
    }
    report_payload = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "classification": compact_classification,
        "semantic_gate": semantic_gate,
        "phase_attribution": phase_increment["classification"],
        "power_observer_only": True,
        "complex_semantic_ols_status": "UNESTABLISHED",
        "label_to_ols_conclusion": "UNRESOLVED",
        "code8_to_ols_conclusion": "PARTIAL_UNRESOLVED",
        "global_prior_family_status": "CLOSED_AFTER_T2_CANARY",
        "next_action": next_action,
        "interpretation_boundary": {
            "allowed": "OLS representation contains or does not contain label information under the registered frozen coordinate contract",
            "forbidden": [
                "LABEL_TO_OLS_SEMANTICS_SUPPORTED",
                "CODE8_OLS_CONTROL_SUPPORTED",
                "GENERATION_READY",
                "GENERATIVE_MANIFOLD_SUPPORTED",
                "OLS_GENERATION_IMPOSSIBLE",
            ],
        },
    }

    if actual_output.exists():
        raise FileExistsError(f"refusing to overwrite S1 output: {_relative_path(actual_output)}")
    actual_output.mkdir(parents=True, exist_ok=False)
    atomic_torch_save_new(actual_output / "ols_embeddings.pt", embeddings_payload, artifact_kind="S1 OLS embeddings")
    atomic_write_new_json(actual_output / "ols_representation_contract.json", contract_payload, artifact_kind="S1 contract")
    atomic_write_new_json(actual_output / "phase_gauge_audit.json", phase_audit, artifact_kind="S1 phase audit")
    atomic_write_new_json(actual_output / "geometry_metrics.json", geometry_payload, artifact_kind="S1 geometry")
    atomic_write_new_json(actual_output / "knn_metrics.json", knn_payload, artifact_kind="S1 kNN")
    atomic_write_new_json(actual_output / "linear_probe_metrics.json", probe_payload, artifact_kind="S1 linear probe")
    atomic_write_new_json(actual_output / "null_control_metrics.json", null_payload, artifact_kind="S1 null control")
    atomic_write_new_json(actual_output / "phase_attribution.json", phase_increment, artifact_kind="S1 phase attribution")
    atomic_write_new_json(actual_output / "identity_manifest.json", identity_payload, artifact_kind="S1 identity")
    atomic_write_new_json(actual_output / "compact_report.json", report_payload, artifact_kind="S1 report")
    _assert_output_tree(actual_output, include_manifest=False)
    refs = _artifact_refs(actual_output)
    storage = sum(item["size_bytes"] for item in refs.values())
    if storage > config["execution"]["max_storage_bytes"]:
        raise RuntimeError("S1 output exceeds the diagnostic storage contract")
    manifest_payload["artifacts"] = refs
    manifest_payload["storage_bytes_excluding_manifest"] = storage
    atomic_write_new_json(actual_output / "semantic_manifest.json", manifest_payload, artifact_kind="S1 manifest")
    _assert_output_tree(actual_output)
    print(json.dumps(manifest_payload, indent=2, sort_keys=True, default=str))
    return manifest_payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    run(args.config, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "CONFIG_SCHEMA",
    "PHASE_SEMANTIC_INCREMENT_NOT_SUPPORTED",
    "PHASE_SEMANTIC_INCREMENT_SUPPORTED",
    "R1_POWER",
    "R2_AMPLITUDE",
    "R2_RAW_AMPLITUDE",
    "R3_COMPLEX",
    "S1_AMPLITUDE_SEMANTICS_SUPPORTED",
    "S1_COMPLEX_SEMANTICS_SUPPORTED",
    "S1_OLS_SEMANTICS_NOT_SUPPORTED",
    "S1_POWER_SEMANTICS_SUPPORTED",
    "S1_SEMANTICS_UNDERDETERMINED",
    "fixed_label_permutation",
    "classify_s1",
    "fit_ridge_probe",
    "gauge_distance_squared",
    "load_config",
    "minimal_representation",
    "normalized_representations",
    "one_nn_metrics",
    "phase_gauge_audit",
    "phase_increment_classification",
    "predict_ridge_probe",
    "representation_gate",
    "run",
    "stratified_bootstrap_ci",
    "validate_amplitude_scale",
]
