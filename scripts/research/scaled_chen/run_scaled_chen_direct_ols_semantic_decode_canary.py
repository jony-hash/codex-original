"""Run the frozen direct complex-OLS semantic decode canary.

The canary compares the registered full optical path with direct re-emission
of the exact observed complex OLS coefficients.  It is a diagnostic
measurement only: no optimizer, backward pass, checkpoint write, prior
sampling, phase ablation, or training path is available here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_manifold_readiness as mra
from ldm_project.v2.config import load_config as load_v2_config
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.conditional import fixed_exposure_view
from ldm_project.v2.student.evaluator import FrozenMNISTEvaluator


TASK_ID = "H-LOMA-GEN-S2-DOSD"
HYPOTHESIS = TASK_ID
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_direct_ols_semantic_decode_canary_v1"
MANIFEST_SCHEMA = "lightgen_scaled_chen_direct_ols_semantic_decode_canary_manifest.v1"
SOURCE_SCHEMA = "lightgen_scaled_chen_direct_ols_source_coefficients.v1"
PREDICTION_SCHEMA = "lightgen_scaled_chen_direct_ols_predictions.v1"
PORT_COUNT = 100
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
BATCH_SIZE = 10
BOOTSTRAP_SEED = 1236
BOOTSTRAP_REPLICATES = 2000
NULL_SEED = 2053
CONFIDENCE_LEVEL = 0.95
FIDELITY_RTOL = 1.0e-5
FIDELITY_ATOL = 1.0e-8
EXPOSURE_SCALE = 0.4913053810596466
MIN_FREE_CUDA_BYTES = 4 * 1024**3
MIN_FREE_DISK_BYTES = 10 * 1024**3
HEX64 = re.compile(r"^[0-9a-f]{64}$")

S1_GATE = "S1_POWER_SEMANTICS_SUPPORTED"
S1_PHASE_ATTRIBUTION = "PHASE_SEMANTIC_INCREMENT_SUPPORTED"
S1_PHASE_GAUGE = "GLOBAL_PHASE_GAUGE_CONFIRMED"
OSERA_GATE = "OUTPUT_SEMANTIC_EVALUATOR_REQUALIFICATION_PASS"
OSERA_CLASSIFICATION = "FROZEN_OUTPUT_SEMANTIC_EVALUATOR_REQUALIFIED_FOR_PAIRED_S2"
FOSEA_CLASSIFICATION = "FROZEN_OUTPUT_SEMANTIC_EVALUATOR_NOT_QUALIFIED"
EVALUATOR_SCHEMA = "frozen_mnist_evaluator_v1"
DIRECT_FIDELITY_GATE = "DIRECT_REEMIT_FIDELITY_PASS"
DIRECT_GATE = "DIRECT_OLS_SEMANTIC_DECODE_SUPPORTED"
DIRECT_REPLAY_SUPPORTED = "DIRECT_OLS_REPLAY_SEMANTIC_ACCESSIBILITY_SUPPORTED"
DIRECT_NOT_SUPPORTED = "S2_DIRECT_OLS_SEMANTIC_DECODE_NOT_SUPPORTED"
INTERFACE_INVALID = "S2_DIRECT_REEMIT_INTERFACE_INVALID"
EVALUATOR_BINDING_INVALID = "S2_SEMANTIC_EVALUATOR_AUTHORITY_INVALID"
STATE_INVALID = "S2_STATE_ISOLATION_INVALID"
NEXT_ACTION_PASS = "PLAN_S3_CONTROL_TO_OLS_WRITABILITY_AUDIT"
NEXT_ACTION_FAIL = "REVIEW_OLS_SEMANTIC_DECODE_DISCONTINUITY"
NEXT_ACTION_INTERFACE = "REVIEW_DIRECT_OLS_INTERFACE_CONTRACT"
NEXT_ACTION_EVALUATOR = "REVIEW_SEMANTIC_EVALUATOR_AUTHORITY"

S1_CONFIG_SHA256 = "60d57cd134a567c17602775ec16f11fff05f56ffe900ac730774abde3432c521"
S1_MANIFEST_SHA256 = "b0199040926618b17373edc4f167cc8bc4788f39f64de174e1a3fb225781833e"
S1_IDENTITY_SHA256 = "522ff5b00770ee34fe7b61b1f414775abe3310f398cc0fdf55049b30be2925e2"
S1_EMBEDDINGS_SHA256 = "9375b08597f03fee3798e05798779556c3e7534db6a9d39033232f7278d32279"
CHECKPOINT_SHA256 = "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de"
CALIBRATION_MANIFEST_SHA256 = "0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6"
CALIBRATION_PAYLOAD_SHA256 = "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07"
MODE_BANK_SHA256 = "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9"
FORMAL_CONFIG_SHA256 = "0c42035c1a525e27ff787d013ca27b0f5ae91338329eb0607c7042ebc5ae8905"
OSERA_MANIFEST_SHA256 = "c6a37a637917f0b32fb96d5f3c51827798696d3bd7924d31c3c441421402850d"
OSERA_ERROR_MODEL_SHA256 = "0adb941c0dea102b93cd0ce2814a76b43bcbe51a0af4d206a4a3c83a6775d028"
FOSEA_MANIFEST_SHA256 = "c8e144da2dd02e9f4ee78f7e8829782800651ec835adb1ae6fd781b1921c1d0f"
EVALUATOR_CHECKPOINT_SHA256 = "a6ac0b4644a459bad21bc7cf3071465d5306af31ad2081fe7acb6f81e87c47e6"
FOSEA_QUALIFICATION_SHA256 = "2fed6db88c0b5003458077198375f746eb995a88be4a9350c2d70cd38acb8568"
FOSEA_OUTPUT_CONTRACT_SHA256 = "515137bd66c71ba6b1b1f4e21fa99d4f2527aca286e11ff079b4b7dfd2fbce56"
FOSEA_EXPOSURE_SHA256 = "e3a9ac15b30c5dd1074914a9f3c05b5f8f21bc3b8349418d2345efc3fb9687f4"
VPIM_MANIFEST_SHA256 = "eff78d5417961d908c148c08084f55485d0c68cd4abc53e2101c811c3eb84336"
VPIM_NULL_PERMUTATION_SHA256 = "34d6437efe47261183ce870d6b91123094c41c6fc72cc4b48b16b0bb39ce135a"
VPIM_NULL_PERMUTATION_TENSOR_SHA256 = "1b3fa37d45fdd646b5b2478a904e5029a402c7b7061d2ecc42a8eeab21a80794"
DATA_MANIFEST_SHA256 = "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb"
SELECTION_INDICES_SHA256 = "11c076affa8511e883f595a0d1ee91e40f6d94747a351e21269520344d3f594f"
SELECTION_LABELS_SHA256 = "31fce460255bda46afc2bd6b6777793873e2e6ee33e23a681def3ab8af6693cb"
CONFIRMATION_INDICES_SHA256 = "ff7ab50b95d96e727c240edc2baa916d1e6c11402b8f4f9c2ce6e2e7e5e2218a"
CONFIRMATION_LABELS_SHA256 = "b05112caa3014d6c2081423b371361eefe1ec7a9f531e54db016b89bfbb77b18"

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs/research/scaled_chen/lightgen_scaled_chen_direct_ols_semantic_decode_canary_v1.yaml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / (
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_direct_ols_semantic_decode_canary_v1_attempt_01"
)


@dataclass(frozen=True)
class AuthorityBundle:
    config_path: Path
    config: dict[str, Any]
    s1_manifest: dict[str, Any]
    s1_identity: dict[str, Any]
    s1_embeddings: dict[str, Any]
    osera_manifest: dict[str, Any]
    fosea_manifest: dict[str, Any]
    evaluator_payload: dict[str, Any]
    vpim_manifest: dict[str, Any]
    vpim_null_payload: dict[str, Any]
    calibration_payload: dict[str, Any]
    checkpoint_payload: dict[str, Any]
    formal_path: Path
    source_paths: dict[str, Path]


@dataclass(frozen=True)
class DirectTrace:
    decoder_input: torch.Tensor
    decoder_output: torch.Tensor
    camera: torch.Tensor


@dataclass(frozen=True)
class SplitReplay:
    coefficients: torch.Tensor
    labels: torch.Tensor
    indices: torch.Tensor
    full_predictions: torch.Tensor
    direct_predictions: torch.Tensor
    fidelity: dict[str, Any]
    api_equivalent: dict[str, Any]


def _resolve(path: str | Path) -> Path:
    target = Path(path)
    return target if target.is_absolute() else REPO_ROOT / target


def _relative(path: str | Path) -> str:
    target = Path(path).resolve()
    return target.relative_to(REPO_ROOT.resolve()).as_posix()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"DOSD authority binding failed: {name} is not a mapping")
    return value


def _require_hex(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value.lower()) is None:
        raise ValueError(f"DOSD authority binding failed: invalid {name}")
    return value.lower()


def _expect_section(payload: Mapping[str, Any], section: str, expected: Mapping[str, Any]) -> None:
    actual = _require_mapping(payload.get(section), section)
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"DOSD authority binding failed: {section}.{key}")


def validate_authority_bindings(payload: Mapping[str, Any]) -> None:
    """Validate immutable config claims before any model or output is created."""
    required = {
        "schema", "task_id", "hypothesis", "s1", "optical", "source", "osera",
        "evaluator", "fidelity", "statistics", "global_prior", "authorization",
        "output", "governance",
    }
    if set(payload) != required:
        raise ValueError("DOSD config fields are not exact")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("task_id") != TASK_ID or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("DOSD config identity is invalid")
    _expect_section(payload, "s1", {
        "config": "configs/research/scaled_chen/lightgen_scaled_chen_ols_semantic_representation_audit_v1.yaml",
        "config_sha256": S1_CONFIG_SHA256,
        "semantic_manifest": "outputs/20260904_v2/runs/lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01/semantic_manifest.json",
        "manifest_sha256": S1_MANIFEST_SHA256,
        "identity_manifest": "outputs/20260904_v2/runs/lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01/identity_manifest.json",
        "identity_manifest_sha256": S1_IDENTITY_SHA256,
        "embeddings": "outputs/20260904_v2/runs/lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01/ols_embeddings.pt",
        "embeddings_sha256": S1_EMBEDDINGS_SHA256,
        "gate": S1_GATE,
        "minimal_representation": "R1_POWER",
        "phase_attribution": S1_PHASE_ATTRIBUTION,
        "phase_gauge": S1_PHASE_GAUGE,
    })
    _expect_section(payload, "optical", {
        "formal_config": "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml",
        "formal_config_sha256": FORMAL_CONFIG_SHA256,
        "checkpoint": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_schema": "scaled_chen_bogt_horizon_checkpoint.v1",
        "global_step": 5000,
        "calibration_manifest": "outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/calibration_manifest.json",
        "calibration_manifest_sha256": CALIBRATION_MANIFEST_SHA256,
        "calibration_payload": "outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/repaired_calibration.pt",
        "calibration_payload_sha256": CALIBRATION_PAYLOAD_SHA256,
        "mode_bank_sha256": MODE_BANK_SHA256,
    })
    _expect_section(payload, "source", {
        "dataset_root": "data",
        "data_manifest_sha256": DATA_MANIFEST_SHA256,
        "selection_count": SELECTION_COUNT,
        "selection_indices_sha256": SELECTION_INDICES_SHA256,
        "selection_labels_sha256": SELECTION_LABELS_SHA256,
        "selection_source": "mnist_test_indices_5000_9999",
        "confirmation_count": CONFIRMATION_COUNT,
        "confirmation_indices_sha256": CONFIRMATION_INDICES_SHA256,
        "confirmation_labels_sha256": CONFIRMATION_LABELS_SHA256,
        "confirmation_source": "horizon_heldout_train_indices_55000_59999",
        "source_tensor": "output.coefficients",
        "source_dtype": "torch.complex64",
        "port_count": PORT_COUNT,
    })
    _expect_section(payload, "osera", {
        "manifest": "outputs/20260906_v2/runs/lightgen_scaled_chen_output_semantic_evaluator_requalification_v1_attempt_01/osera_manifest.json",
        "manifest_sha256": OSERA_MANIFEST_SHA256,
        "error_model": "outputs/20260906_v2/runs/lightgen_scaled_chen_output_semantic_evaluator_requalification_v1_attempt_01/semantic_instrument_error_model.json",
        "error_model_sha256": OSERA_ERROR_MODEL_SHA256,
        "gate": OSERA_GATE,
        "classification": OSERA_CLASSIFICATION,
        "evaluator_scope": "PAIRED_S2_SEMANTIC_MEASUREMENT",
        "historical_fosea_manifest": "outputs/20260904_v2/runs/lightgen_scaled_chen_frozen_output_semantic_evaluator_v1_attempt_01/evaluator_manifest.json",
        "historical_fosea_manifest_sha256": FOSEA_MANIFEST_SHA256,
        "evaluator_checkpoint": "outputs/20260904_v2/runs/lightgen_scaled_chen_frozen_output_semantic_evaluator_v1_attempt_01/frozen_output_semantic_evaluator.pt",
        "evaluator_checkpoint_sha256": EVALUATOR_CHECKPOINT_SHA256,
        "vpim_manifest": "outputs/20260906_v2/runs/lightgen_scaled_chen_output_semantic_evaluator_prediction_identity_v1_attempt_01/prediction_identity_manifest.json",
        "vpim_manifest_sha256": VPIM_MANIFEST_SHA256,
        "vpim_null_permutation": "outputs/20260906_v2/runs/lightgen_scaled_chen_output_semantic_evaluator_prediction_identity_v1_attempt_01/null_permutation_indices.pt",
        "vpim_null_permutation_sha256": VPIM_NULL_PERMUTATION_SHA256,
        "vpim_null_permutation_tensor_sha256": VPIM_NULL_PERMUTATION_TENSOR_SHA256,
        "vpim_null_seed": NULL_SEED,
    })
    _expect_section(payload, "evaluator", {
        "schema": EVALUATOR_SCHEMA,
        "architecture": "FrozenMNISTEvaluator",
        "exposure_scale": EXPOSURE_SCALE,
        "preprocessing": "global_exposure_scale_clamp_0_1_adaptive_avg_pool_28x28",
        "per_sample_normalization": False,
        "prediction_correction": False,
        "threshold_adjustment": False,
        "training": False,
    })
    _expect_section(payload, "fidelity", {
        "rtol": FIDELITY_RTOL,
        "atol": FIDELITY_ATOL,
        "comparisons": ["decoder_input", "decoder_output", "camera_output", "evaluator_view"],
        "gate": DIRECT_FIDELITY_GATE,
        "no_posthoc_tolerance_relaxation": True,
    })
    _expect_section(payload, "statistics", {
        "class_count": 10,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": CONFIDENCE_LEVEL,
        "interval": "percentile",
        "resampling": "within_ground_truth_class_with_replacement",
        "null_seed": NULL_SEED,
        "paired_preservation_rule": "lower95_paired_correctness_delta_greater_equal_zero",
        "selection_positive_point_delta": True,
        "confirmation_positive_lower95_delta": True,
    })
    _expect_section(payload, "global_prior", {
        "status": "CLOSED_AFTER_THIS_CANARY",
        "new_objectives": False,
        "prohibited": ["MMD", "SWD", "SCED", "Chamfer", "Sinkhorn", "new_prior", "beta_tuning"],
    })
    authorization = _require_mapping(payload.get("authorization"), "authorization")
    expected_authorization = {
        "s2_execution": True, "model_execution": True, "optical_execution": True,
        "evaluator_replay": True, "training": False, "optimizer": False,
        "backward": False, "checkpoint_creation": False, "prior_generation": False,
        "phase_update": False, "generation": False, "formal_promotion": False,
    }
    if dict(authorization) != expected_authorization:
        raise ValueError("DOSD authorization boundary is not locked")
    _expect_section(payload, "output", {
        "path": "outputs/20260906_v2/runs/lightgen_scaled_chen_direct_ols_semantic_decode_canary_v1_attempt_01",
        "write_once": True,
    })
    if dict(_require_mapping(payload.get("governance"), "governance")) != {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }:
        raise ValueError("DOSD governance boundary is not locked")


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DOSD config must be a YAML mapping")
    validate_authority_bindings(payload)
    return payload


def next_write_once_output(base_path: str | Path) -> Path:
    """Return the first unused numbered attempt without touching disk."""
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


def validate_direct_coefficients(
    coefficients: torch.Tensor, *, phase: torch.Tensor | None = None
) -> torch.Tensor:
    """Accept only the observed complex coefficient vector used by S2."""
    if coefficients.ndim != 2 or coefficients.shape[1] != PORT_COUNT:
        raise ValueError("direct coefficients have invalid shape; expected [batch,100]")
    if not torch.is_complex(coefficients):
        raise ValueError("direct coefficients must be complex; power-only input is forbidden")
    if phase is not None:
        raise ValueError("direct replay forbids a phase override")
    if not torch.isfinite(coefficients).all():
        raise ValueError("direct coefficients must be finite")
    return coefficients


def compare_fidelity_tensor(
    full: torch.Tensor,
    direct: torch.Tensor,
    *,
    name: str,
    rtol: float = FIDELITY_RTOL,
    atol: float = FIDELITY_ATOL,
) -> dict[str, Any]:
    if full.shape != direct.shape:
        return {
            "name": name, "shape_equal": False, "allclose": False,
            "full_shape": list(full.shape), "direct_shape": list(direct.shape),
            "max_abs": float("inf"), "mse": float("inf"),
            "rtol": rtol, "atol": atol,
        }
    if not torch.isfinite(full).all() or not torch.isfinite(direct).all():
        return {
            "name": name, "shape_equal": True, "allclose": False,
            "full_shape": list(full.shape), "direct_shape": list(direct.shape),
            "max_abs": float("inf"), "mse": float("inf"),
            "rtol": rtol, "atol": atol,
        }
    difference = (full - direct).abs()
    return {
        "name": name,
        "shape_equal": True,
        "allclose": bool(torch.allclose(full, direct, rtol=rtol, atol=atol)),
        "full_shape": list(full.shape),
        "direct_shape": list(direct.shape),
        "max_abs": float(difference.max().detach().cpu()),
        "mse": float(difference.square().mean().detach().cpu()),
        "rtol": rtol,
        "atol": atol,
    }


def fixed_label_permutation(labels: torch.Tensor, *, seed: int = NULL_SEED) -> torch.Tensor:
    if labels.ndim != 1 or labels.numel() == 0 or labels.dtype != torch.long:
        raise ValueError("null labels must be a non-empty CPU int64 vector")
    if labels.device.type != "cpu":
        raise ValueError("null permutation must be generated on CPU")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return labels.index_select(0, torch.randperm(labels.numel(), generator=generator))


def _fixed_permutation_indices(count: int, *, seed: int) -> torch.Tensor:
    if count <= 0:
        raise ValueError("permutation count must be positive")
    return torch.randperm(count, generator=torch.Generator(device="cpu").manual_seed(seed))


def stratified_bootstrap_delta_ci(
    delta: torch.Tensor,
    labels: torch.Tensor,
    *,
    reps: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int | bool]:
    if delta.ndim != 1 or labels.ndim != 1 or delta.shape != labels.shape or delta.numel() == 0:
        raise ValueError("paired bootstrap inputs must be aligned vectors")
    if reps < 100 or labels.unique().numel() < 2:
        raise ValueError("stratified bootstrap requires at least two classes and 100 replicates")
    values = delta.to(torch.float64).cpu()
    classes = labels.cpu()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    distribution = torch.zeros(reps, dtype=torch.float64)
    total_count = float(labels.numel())
    for label in classes.unique(sorted=True):
        positions = torch.nonzero(classes == label, as_tuple=False).flatten()
        draws = torch.randint(positions.numel(), (reps, positions.numel()), generator=generator)
        sampled = positions.index_select(0, draws.reshape(-1)).reshape(reps, -1)
        distribution += values.index_select(0, sampled.reshape(-1)).reshape(reps, -1).mean(dim=1) * (positions.numel() / total_count)
    quantiles = torch.quantile(distribution, torch.tensor([0.025, 0.975], dtype=torch.float64))
    return {
        "point_delta": float(values.mean()),
        "lower95": float(quantiles[0]),
        "upper95": float(quantiles[1]),
        "replicates": reps,
        "seed": seed,
        "stratified": True,
    }


def stratified_bootstrap_ci(
    real: torch.Tensor,
    shuffled: torch.Tensor,
    labels: torch.Tensor,
    *,
    reps: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int | bool]:
    if real.ndim != 1 or shuffled.ndim != 1 or labels.ndim != 1 or not (real.shape == shuffled.shape == labels.shape):
        raise ValueError("null bootstrap inputs must be aligned vectors")
    return stratified_bootstrap_delta_ci(real.to(torch.float64) - shuffled.to(torch.float64), labels, reps=reps, seed=seed)


def classify_s2(
    *,
    fidelity_pass: bool,
    evaluator_binding_pass: bool,
    source_identity_pass: bool,
    selection_delta: float,
    confirmation_lower95: float,
    paired_lower95: float,
    state_isolation_pass: bool,
    finite: bool,
    predictions_identical: bool,
    tensors_identical: bool,
) -> dict[str, Any]:
    if not fidelity_pass:
        return {
            "gate": INTERFACE_INVALID,
            "classification": INTERFACE_INVALID,
            "next_action": NEXT_ACTION_INTERFACE,
            "reason": "DIRECT_REEMIT_FIDELITY_FAIL",
            "semantic_generation_supported": False,
            "replay_semantic_accessibility_supported": False,
        }
    if not evaluator_binding_pass:
        return {
            "gate": EVALUATOR_BINDING_INVALID,
            "classification": EVALUATOR_BINDING_INVALID,
            "next_action": NEXT_ACTION_EVALUATOR,
            "reason": "OSERA_EVALUATOR_AUTHORITY_BINDING_FAIL",
            "semantic_generation_supported": False,
            "replay_semantic_accessibility_supported": False,
        }
    if not source_identity_pass or not state_isolation_pass or not finite:
        reason = "SOURCE_IDENTITY_FAIL" if not source_identity_pass else "STATE_OR_FINITE_FAIL"
        return {
            "gate": STATE_INVALID,
            "classification": STATE_INVALID,
            "next_action": NEXT_ACTION_FAIL,
            "reason": reason,
            "semantic_generation_supported": False,
            "replay_semantic_accessibility_supported": False,
        }
    semantic_pass = selection_delta > 0.0 and confirmation_lower95 > 0.0 and paired_lower95 >= 0.0
    if not semantic_pass:
        return {
            "gate": DIRECT_NOT_SUPPORTED,
            "classification": DIRECT_NOT_SUPPORTED,
            "next_action": NEXT_ACTION_FAIL,
            "reason": "SOURCE_LABEL_SEMANTIC_PRESERVATION_GATE_FAIL",
            "semantic_generation_supported": False,
            "replay_semantic_accessibility_supported": False,
        }
    replay = bool(predictions_identical and tensors_identical)
    return {
        "gate": DIRECT_GATE,
        "classification": DIRECT_GATE,
        "next_action": NEXT_ACTION_PASS,
        "reason": None,
        "semantic_generation_supported": False,
        "replay_semantic_accessibility_supported": replay,
        "replay_classification": DIRECT_REPLAY_SUPPORTED if replay else None,
    }


def _load_json(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise ValueError(f"DOSD authority binding failed: {role}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"DOSD authority binding failed: {role} schema")
    return payload


def _source_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    s1 = _require_mapping(config["s1"], "s1")
    optical = _require_mapping(config["optical"], "optical")
    osera = _require_mapping(config["osera"], "osera")
    return {
        "s1_config": _resolve(s1["config"]),
        "s1_manifest": _resolve(s1["semantic_manifest"]),
        "s1_identity": _resolve(s1["identity_manifest"]),
        "s1_embeddings": _resolve(s1["embeddings"]),
        "formal_config": _resolve(optical["formal_config"]),
        "checkpoint": _resolve(optical["checkpoint"]),
        "calibration_manifest": _resolve(optical["calibration_manifest"]),
        "calibration_payload": _resolve(optical["calibration_payload"]),
        "osera_manifest": _resolve(osera["manifest"]),
        "osera_error_model": _resolve(osera["error_model"]),
        "fosea_manifest": _resolve(osera["historical_fosea_manifest"]),
        "evaluator_checkpoint": _resolve(osera["evaluator_checkpoint"]),
        "vpim_manifest": _resolve(osera["vpim_manifest"]),
        "vpim_null_permutation": _resolve(osera["vpim_null_permutation"]),
    }


def _validate_s1_identity(identity: Mapping[str, Any]) -> None:
    if identity.get("schema") != "lightgen_scaled_chen_ols_semantic_identity_manifest.v1" or identity.get("reselection") is not False or identity.get("self_leakage") is not False:
        raise ValueError("DOSD S1 identity manifest is invalid")
    expected = {
        "selection": (SELECTION_COUNT, SELECTION_INDICES_SHA256, SELECTION_LABELS_SHA256, "mnist_test_indices_5000_9999"),
        "confirmation": (CONFIRMATION_COUNT, CONFIRMATION_INDICES_SHA256, CONFIRMATION_LABELS_SHA256, "horizon_heldout_train_indices_55000_59999"),
    }
    for name, (count, indices_hash, labels_hash, source) in expected.items():
        section = _require_mapping(identity.get(name), f"s1.identity.{name}")
        if dict(section) != ({
            "count": count,
            "indices_sha256": indices_hash,
            "labels_sha256": labels_hash,
            "source": source,
            **({"per_label": 100} if name == "selection" else {}),
        }):
            raise ValueError(f"DOSD S1 {name} identity changed")


def _validate_source_artifacts(config: Mapping[str, Any]) -> AuthorityBundle:
    paths = _source_paths(config)
    expected_hashes = {
        "s1_config": S1_CONFIG_SHA256, "s1_manifest": S1_MANIFEST_SHA256,
        "s1_identity": S1_IDENTITY_SHA256, "s1_embeddings": S1_EMBEDDINGS_SHA256,
        "formal_config": FORMAL_CONFIG_SHA256, "checkpoint": CHECKPOINT_SHA256,
        "calibration_manifest": CALIBRATION_MANIFEST_SHA256, "calibration_payload": CALIBRATION_PAYLOAD_SHA256,
        "osera_manifest": OSERA_MANIFEST_SHA256, "osera_error_model": OSERA_ERROR_MODEL_SHA256,
        "fosea_manifest": FOSEA_MANIFEST_SHA256, "evaluator_checkpoint": EVALUATOR_CHECKPOINT_SHA256,
        "vpim_manifest": VPIM_MANIFEST_SHA256, "vpim_null_permutation": VPIM_NULL_PERMUTATION_SHA256,
    }
    for name, expected in expected_hashes.items():
        if not paths[name].is_file() or file_sha256(paths[name]) != expected:
            raise ValueError(f"DOSD authority binding failed: {name}")

    s1_config = yaml.safe_load(paths["s1_config"].read_text(encoding="utf-8"))
    if not isinstance(s1_config, Mapping) or s1_config.get("schema") != "lightgen_scaled_chen_ols_semantic_representation_audit_v1" or s1_config.get("task_id") != "H-LOMA-GEN-S1-OLS-SRA":
        raise ValueError("DOSD S1 config identity changed")
    s1_manifest = _load_json(paths["s1_manifest"], S1_MANIFEST_SHA256, "S1 semantic manifest")
    if s1_manifest.get("schema") != "lightgen_scaled_chen_ols_semantic_representation_audit.v1" or s1_manifest.get("task_id") != "H-LOMA-GEN-S1-OLS-SRA" or s1_manifest.get("gate") != S1_GATE or s1_manifest.get("minimal_representation") != "S1_POWER_SEMANTICS_SUPPORTED" or s1_manifest.get("phase_attribution") != S1_PHASE_ATTRIBUTION:
        raise ValueError("DOSD S1 semantic authority changed")
    if _require_mapping(s1_manifest.get("phase_gauge"), "S1 phase gauge").get("classification") != S1_PHASE_GAUGE:
        raise ValueError("DOSD S1 phase-gauge authority changed")
    s1_identity = _load_json(paths["s1_identity"], S1_IDENTITY_SHA256, "S1 identity manifest")
    _validate_s1_identity(s1_identity)
    embeddings = safe_torch_load(paths["s1_embeddings"], expected_sha256=S1_EMBEDDINGS_SHA256, map_location="cpu")
    if not isinstance(embeddings, Mapping) or embeddings.get("schema") != "lightgen_scaled_chen_ols_semantic_embeddings.v1":
        raise ValueError("DOSD S1 embeddings schema changed")
    for split, count in (("selection", SELECTION_COUNT), ("confirmation", CONFIRMATION_COUNT)):
        reps = _require_mapping(_require_mapping(embeddings.get("representations"), "S1 representations").get(split), f"S1 representations.{split}")
        labels = _require_mapping(embeddings.get("labels"), "S1 labels")[split]
        indices = _require_mapping(embeddings.get("indices"), "S1 indices")[split]
        r3 = reps.get("R3_COMPLEX")
        if not torch.is_tensor(r3) or r3.dtype != torch.complex64 or tuple(r3.shape) != (count, PORT_COUNT) or not torch.isfinite(r3).all():
            raise ValueError(f"DOSD S1 R3 source shape changed: {split}")
        if not torch.is_tensor(labels) or labels.dtype != torch.long or tuple(labels.shape) != (count,):
            raise ValueError(f"DOSD S1 labels shape changed: {split}")
        if not torch.is_tensor(indices) or indices.dtype != torch.long or tuple(indices.shape) != (count,):
            raise ValueError(f"DOSD S1 indices shape changed: {split}")

    calibration = safe_torch_load(paths["calibration_payload"], expected_sha256=CALIBRATION_PAYLOAD_SHA256, map_location="cpu")
    if not isinstance(calibration, Mapping) or calibration.get("schema") != "scaled_chen_bogt_calibration.v2" or calibration.get("mode_bank_sha256") != MODE_BANK_SHA256:
        raise ValueError("DOSD repaired calibration authority changed")
    checkpoint = safe_torch_load(paths["checkpoint"], expected_sha256=CHECKPOINT_SHA256, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or checkpoint.get("schema") != "scaled_chen_bogt_horizon_checkpoint.v1" or int(checkpoint.get("global_step", -1)) != 5000 or not isinstance(checkpoint.get("model_state"), Mapping):
        raise ValueError("DOSD optical checkpoint authority changed")

    osera_manifest = _load_json(paths["osera_manifest"], OSERA_MANIFEST_SHA256, "OSERA manifest")
    osera_gates = _require_mapping(osera_manifest.get("gates"), "OSERA gates")
    if osera_manifest.get("schema") != "lightgen_scaled_chen_output_semantic_evaluator_requalification_manifest.v1" or osera_manifest.get("task_id") != "H-LOMA-GEN-S2-OSERA" or osera_manifest.get("primary_gate") != OSERA_GATE or osera_manifest.get("classification") != OSERA_CLASSIFICATION or osera_gates.get("gate") != OSERA_GATE or osera_gates.get("classification") != OSERA_CLASSIFICATION:
        raise ValueError("DOSD OSERA authority changed")
    error_model = _load_json(paths["osera_error_model"], OSERA_ERROR_MODEL_SHA256, "OSERA error model")
    if error_model.get("schema") != "lightgen_scaled_chen_output_semantic_evaluator_error_model.v1" or error_model.get("name") != "SEMANTIC_INSTRUMENT_ERROR_MODEL_V1" or error_model.get("frozen") is not True or error_model.get("raw_predictions_authoritative") is not True or error_model.get("prediction_correction_applied") is not False:
        raise ValueError("DOSD semantic error model authority changed")
    fosea_manifest = _load_json(paths["fosea_manifest"], FOSEA_MANIFEST_SHA256, "historical FOSEA manifest")
    fosea_eval = _require_mapping(fosea_manifest.get("evaluator"), "FOSEA evaluator")
    fosea_qualification = _require_mapping(fosea_manifest.get("qualification"), "FOSEA qualification")
    if fosea_manifest.get("classification") != FOSEA_CLASSIFICATION or fosea_eval.get("schema") != EVALUATOR_SCHEMA or fosea_eval.get("checkpoint_sha256") != EVALUATOR_CHECKPOINT_SHA256 or fosea_qualification.get("classification") != FOSEA_CLASSIFICATION or fosea_qualification.get("threshold") != 0.985:
        raise ValueError("DOSD historical FOSEA authority changed")
    fosea_exposure = _require_mapping(fosea_manifest.get("exposure"), "FOSEA exposure")
    if fosea_exposure.get("scale") != EXPOSURE_SCALE or fosea_exposure.get("tuned_on_s2_performance") is not False:
        raise ValueError("DOSD evaluator exposure authority changed")
    evaluator_payload = safe_torch_load(paths["evaluator_checkpoint"], expected_sha256=EVALUATOR_CHECKPOINT_SHA256, map_location="cpu")
    if not isinstance(evaluator_payload, Mapping) or evaluator_payload.get("architecture_kind") != EVALUATOR_SCHEMA or not isinstance(evaluator_payload.get("state_dict"), Mapping):
        raise ValueError("DOSD evaluator checkpoint schema changed")
    vpim_manifest = _load_json(paths["vpim_manifest"], VPIM_MANIFEST_SHA256, "VPIM manifest")
    if vpim_manifest.get("schema") != "lightgen_scaled_chen_output_semantic_evaluator_prediction_identity_manifest.v1" or vpim_manifest.get("task_id") != "H-LOMA-GEN-S2-OSERA-VPIM" or vpim_manifest.get("classification") != "OSERA_VALIDATION_PREDICTION_IDENTITY_READY":
        raise ValueError("DOSD VPIM authority changed")
    null_payload = safe_torch_load(paths["vpim_null_permutation"], expected_sha256=VPIM_NULL_PERMUTATION_SHA256, map_location="cpu")
    if not isinstance(null_payload, Mapping) or null_payload.get("schema") != "lightgen_scaled_chen_output_semantic_evaluator_null_permutation.v1" or null_payload.get("seed") != NULL_SEED or null_payload.get("sample_order_hash") != "aedcb0f58651c77518ae01ca6f908ad2126a81395a7a431a233bd0497b60601e":
        raise ValueError("DOSD VPIM null authority changed")
    permutation = null_payload.get("permutation_indices")
    if not torch.is_tensor(permutation) or permutation.dtype != torch.long or tuple(permutation.shape) != (CONFIRMATION_COUNT,) or tensor_sha256(permutation) != VPIM_NULL_PERMUTATION_TENSOR_SHA256:
        raise ValueError("DOSD VPIM null permutation changed")

    formal_path = paths["formal_config"]
    return AuthorityBundle(
        _resolve(CONFIG_PATH), dict(config), dict(s1_manifest), dict(s1_identity), dict(embeddings),
        dict(osera_manifest), dict(fosea_manifest), dict(evaluator_payload), dict(vpim_manifest),
        dict(null_payload), dict(calibration), dict(checkpoint), formal_path, paths,
    )


def _git_state() -> dict[str, Any]:
    return base._git_state()


def _state_digest(module: torch.nn.Module) -> str:
    return canonical_sha256({name: tensor_sha256(value) for name, value in module.state_dict().items()})


def _validate_data(data: mra.AuditData, identity: Mapping[str, Any]) -> None:
    if data.manifest.get("manifest_hash") != DATA_MANIFEST_SHA256:
        raise ValueError("DOSD dataset manifest authority changed")
    selection = data.selection_indices
    confirmation = data.confirmation_indices
    if tensor_sha256(selection) != SELECTION_INDICES_SHA256 or tensor_sha256(confirmation) != CONFIRMATION_INDICES_SHA256:
        raise ValueError("DOSD source index identity changed")
    for name, actual, expected in (
        ("selection", data.test_labels.index_select(0, selection), SELECTION_LABELS_SHA256),
        ("confirmation", data.labels.index_select(0, confirmation), CONFIRMATION_LABELS_SHA256),
    ):
        if tensor_sha256(actual) != expected:
            raise ValueError(f"DOSD {name} label identity changed")
        registered = _require_mapping(identity[name], f"S1 identity {name}")
        if int(registered["count"]) != actual.numel() or registered["labels_sha256"] != expected:
            raise ValueError(f"DOSD {name} identity binding changed")


def _load_formal(path: Path) -> Any:
    config = load_v2_config(path)
    if config.architecture != "conditional_identity_logit_amplitude_v2" or config.system.kind != bogt.EXPECTED_FORMAL_KIND or not config.system.is_formal:
        raise ValueError("DOSD requires the formal finite-crop Identity geometry")
    formal = config.system
    expected = {
        "wavelength_m": 532.0e-9, "slm_shape": (200, 200), "slm_pixel_pitch_m": 2.5e-6,
        "metasurface_shape": (100, 100), "metasurface_pixel_pitch_m": 2.0e-6,
        "propagation_grid_size": 2048, "encoder_layers": 2, "decoder_layers": 2,
        "layer_spacing_m": 1.50e-3, "front_magnification": -0.4, "back_magnification": -2.5,
        "ols_kind": "identity", "boundary": "finite_crop",
    }
    for name, value in expected.items():
        actual = getattr(formal, name)
        if isinstance(value, float):
            if abs(actual - value) > max(abs(value) * 1.0e-12, 1.0e-15):
                raise ValueError(f"DOSD formal geometry changed: {name}")
        elif actual != value:
            raise ValueError(f"DOSD formal geometry changed: {name}")
    return formal


def _check_resources(output: Path) -> dict[str, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("DOSD requires the registered CUDA execution environment")
    free_cuda, total_cuda = torch.cuda.mem_get_info()
    free_disk = shutil.disk_usage(REPO_ROOT).free
    if free_cuda < MIN_FREE_CUDA_BYTES:
        raise RuntimeError("DOSD CUDA free-memory preflight is below the registered floor")
    if free_disk < MIN_FREE_DISK_BYTES:
        raise RuntimeError("DOSD disk free-space preflight is below the registered floor")
    if output.exists():
        raise FileExistsError(f"DOSD write-once output already exists: {_relative(output)}")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("DOSD requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    return {"free_cuda_bytes": int(free_cuda), "total_cuda_bytes": int(total_cuda), "free_disk_bytes": int(free_disk)}


def _preflight(config_path: str | Path, output_path: str | Path) -> tuple[AuthorityBundle, mra.AuditData, Any, dict[str, Any]]:
    config_target = _resolve(config_path)
    config = load_config(config_target)
    state = _git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_DIRTY_PATH: DOSD requires clean branch {BRANCH}; status={state.get('status')}")
    output = _resolve(output_path)
    resources = _check_resources(output)
    authorities = _validate_source_artifacts(config)
    data = mra._load_data(config["source"]["dataset_root"])
    _validate_data(data, authorities.s1_identity)
    formal = _load_formal(authorities.formal_path)
    return authorities, data, formal, {"git": state, "resources": resources, "config_sha256": file_sha256(config_target)}


def _direct_reemit_trace(model: bogt.ScaledChenBOGTModel, coefficients: torch.Tensor) -> DirectTrace:
    validate_direct_coefficients(coefficients)
    modes = model.ols.output_modes.to(coefficients)
    values = torch.einsum("bm,mhw->bhw", coefficients, modes).unsqueeze(1)
    field = ComplexField(
        values,
        model.system.encoder_plane.with_plane("ols_output"),
        OperationProvenance(
            "fixed_ols_observed_complex_reemission", "ols_output", 0.0,
            {"phase": "preserved_observed_relative_phase", "power_only_injection": False},
        ),
    )
    decoded, _, _ = model.system.decoder(field, "decoder")
    decoded = bogt._attenuate(decoded, bogt.FIELD_TRANSMISSION ** 2, "fixed_decoder_group_transmission")
    camera_field = model.chen_relay(ComplexField(decoded.tensor, model.system.chip_plane, decoded.provenance))
    return DirectTrace(field.tensor, decoded.tensor, camera_field.tensor.abs().square())


def _split_inputs(data: mra.AuditData, split: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if split == "selection":
        indices = data.selection_indices
        return data.test_images.index_select(0, indices), data.test_labels.index_select(0, indices), indices
    if split == "confirmation":
        indices = data.confirmation_indices
        return data.images.index_select(0, indices), data.labels.index_select(0, indices), indices
    raise ValueError(f"unsupported DOSD split: {split}")


def _aggregate_fidelity(records: list[dict[str, Any]], count: int) -> dict[str, Any]:
    names = ("decoder_input", "decoder_output", "camera_output", "evaluator_view")
    result: dict[str, Any] = {}
    for name in names:
        entries = [record[name] for record in records]
        result[name] = {
            "allclose": all(bool(entry["allclose"]) for entry in entries),
            "shape_equal": all(bool(entry["shape_equal"]) for entry in entries),
            "max_abs": max(float(entry["max_abs"]) for entry in entries),
            "mean_mse": sum(float(entry["mse"]) for entry in entries) / len(entries),
            "batch_count": len(entries),
            "sample_count": count,
            "rtol": FIDELITY_RTOL,
            "atol": FIDELITY_ATOL,
        }
    result["allclose"] = all(bool(result[name]["allclose"]) for name in names)
    result["gate"] = DIRECT_FIDELITY_GATE if result["allclose"] else "DIRECT_REEMIT_FIDELITY_FAIL"
    return result


def _collect_split(
    model: bogt.ScaledChenBOGTModel,
    evaluator: FrozenMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
) -> SplitReplay:
    coefficient_chunks: list[torch.Tensor] = []
    full_prediction_chunks: list[torch.Tensor] = []
    direct_prediction_chunks: list[torch.Tensor] = []
    fidelity_records: list[dict[str, Any]] = []
    api_record: dict[str, Any] | None = None
    with torch.inference_mode():
        for batch_images, batch_labels in zip(images.split(BATCH_SIZE), labels.split(BATCH_SIZE), strict=True):
            rendered = base.render_intensity_batch(batch_images)
            amplitude = rendered.amplitude.to(device)
            full = model(amplitude, return_trace=True)
            coefficients = validate_direct_coefficients(full.coefficients)
            if len(full.decoder_records) != 2:
                raise ValueError("DOSD full path did not return the registered two decoder records")
            direct = _direct_reemit_trace(model, coefficients)
            if api_record is None:
                api_camera = model.reemit_decoder(coefficients, phase=None)
                api_record = compare_fidelity_tensor(
                    direct.camera, api_camera, name="reemit_decoder_api", rtol=FIDELITY_RTOL, atol=FIDELITY_ATOL,
                )
            full_view = fixed_exposure_view(full.camera, EXPOSURE_SCALE)
            direct_view = fixed_exposure_view(direct.camera, EXPOSURE_SCALE)
            batch_fidelity = {
                "decoder_input": compare_fidelity_tensor(full.ols_field.tensor, direct.decoder_input, name="decoder_input"),
                "decoder_output": compare_fidelity_tensor(full.decoder_records[-1].tensor, direct.decoder_output, name="decoder_output"),
                "camera_output": compare_fidelity_tensor(full.camera, direct.camera, name="camera_output"),
                "evaluator_view": compare_fidelity_tensor(full_view, direct_view, name="evaluator_view"),
            }
            fidelity_records.append(batch_fidelity)
            _, full_logits = evaluator(full_view)
            _, direct_logits = evaluator(direct_view)
            full_prediction_chunks.append(full_logits.argmax(dim=1).detach().cpu())
            direct_prediction_chunks.append(direct_logits.argmax(dim=1).detach().cpu())
            coefficient_chunks.append(coefficients.detach().cpu().contiguous())
            if not torch.isfinite(full.camera).all() or not torch.isfinite(direct.camera).all():
                raise ValueError("DOSD full/direct camera output is non-finite")
            del rendered, amplitude, full, direct, full_view, direct_view, coefficients
    if api_record is None or not api_record["allclose"]:
        raise ValueError("DOSD existing reemit_decoder interface is not equivalent to the direct primitive")
    coefficients = torch.cat(coefficient_chunks, dim=0)
    full_predictions = torch.cat(full_prediction_chunks, dim=0)
    direct_predictions = torch.cat(direct_prediction_chunks, dim=0)
    if coefficients.shape != (labels.numel(), PORT_COUNT) or full_predictions.shape != labels.shape or direct_predictions.shape != labels.shape:
        raise ValueError("DOSD split replay output shape is invalid")
    return SplitReplay(
        coefficients, labels.clone(), indices.clone(), full_predictions, direct_predictions,
        _aggregate_fidelity(fidelity_records, labels.numel()), api_record,
    )


def _semantic_metrics(predictions: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    if predictions.shape != labels.shape or predictions.ndim != 1:
        raise ValueError("semantic metric vectors are not aligned")
    predictions = predictions.to(torch.long).cpu()
    labels = labels.to(torch.long).cpu()
    correct = predictions.eq(labels)
    confusion = []
    per_class: dict[str, float] = {}
    counts: dict[str, int] = {}
    for label in range(10):
        mask = labels == label
        row = [int(((predictions == predicted) & mask).sum()) for predicted in range(10)]
        confusion.append(row)
        counts[str(label)] = int(mask.sum())
        per_class[str(label)] = float(correct[mask].to(torch.float64).mean()) if mask.any() else float("nan")
    return {
        "sample_count": int(labels.numel()),
        "prediction_hash": tensor_sha256(predictions),
        "correctness_hash": tensor_sha256(correct),
        "overall_accuracy": float(correct.to(torch.float64).mean()),
        "correct_count": int(correct.sum()),
        "class_order": list(range(10)),
        "class_counts": counts,
        "per_class_accuracy": per_class,
        "confusion": confusion,
        "finite": bool(torch.isfinite(predictions.to(torch.float32)).all()),
    }


def _null_metrics(
    replay: SplitReplay,
    null_labels: torch.Tensor,
    null_permutation: torch.Tensor,
) -> dict[str, Any]:
    labels = replay.labels
    direct_real = replay.direct_predictions.eq(labels)
    direct_shuffled = replay.direct_predictions.eq(null_labels)
    full_real = replay.full_predictions.eq(labels)
    full_shuffled = replay.full_predictions.eq(null_labels)
    direct_ci = stratified_bootstrap_ci(direct_real, direct_shuffled, labels)
    full_ci = stratified_bootstrap_ci(full_real, full_shuffled, labels)
    return {
        "schema": "lightgen_scaled_chen_direct_ols_semantic_null.v1",
        "seed": NULL_SEED,
        "only_labels_permuted": True,
        "predictions_unchanged": True,
        "sample_order_unchanged": True,
        "permutation_hash": tensor_sha256(null_permutation),
        "null_labels_hash": tensor_sha256(null_labels),
        "direct": {
            "real_accuracy": float(direct_real.to(torch.float64).mean()),
            "shuffled_accuracy": float(direct_shuffled.to(torch.float64).mean()),
            "bootstrap": direct_ci,
        },
        "full_descriptive": {
            "real_accuracy": float(full_real.to(torch.float64).mean()),
            "shuffled_accuracy": float(full_shuffled.to(torch.float64).mean()),
            "bootstrap": full_ci,
        },
    }


def _source_identity(
    authority: AuthorityBundle,
    replays: Mapping[str, SplitReplay],
    scale: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_payload: dict[str, Any] = {"schema": SOURCE_SCHEMA}
    coefficient_payload: dict[str, Any] = {"schema": SOURCE_SCHEMA, "dtype": "torch.complex64", "port_count": PORT_COUNT}
    all_exact = True
    for split, expected_r3_hash in (("selection", "a3d6e3ddc4509aadb88ccac794485915afd77e040ea596a96103ff09c7037ad2"), ("confirmation", "8b149e33c0b5c443f9660065a5001cd5a04137a0d37b4cbcebf6b388fa1adf18")):
        replay = replays[split]
        registered = _require_mapping(authority.s1_embeddings["representations"], "S1 representations")[split]["R3_COMPLEX"]
        normalized = replay.coefficients / scale.sqrt().unsqueeze(0)
        exact = bool(torch.equal(normalized, registered))
        all_exact = all_exact and exact
        source_payload[split] = {
            "count": int(replay.coefficients.shape[0]),
            "indices_hash": tensor_sha256(replay.indices),
            "labels_hash": tensor_sha256(replay.labels),
            "coefficients_hash": tensor_sha256(replay.coefficients),
            "normalized_r3_hash": tensor_sha256(normalized),
            "registered_s1_r3_hash": tensor_sha256(registered),
            "exact_equal_to_s1_r3": exact,
            "dtype": str(replay.coefficients.dtype),
            "shape": list(replay.coefficients.shape),
            "source": "frozen_parent_output.coefficients",
        }
        coefficient_payload[split] = {
            "coefficients": replay.coefficients,
            "labels": replay.labels,
            "indices": replay.indices,
        }
    source_payload.update({
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "mode_bank_sha256": MODE_BANK_SHA256,
        "s1_manifest_sha256": S1_MANIFEST_SHA256,
        "s1_identity_manifest_sha256": S1_IDENTITY_SHA256,
        "normalized_r3_exact_identity": all_exact,
        "injection": "exact_observed_complex_c_preserve_amplitude_and_relative_phase",
        "power_only_injection": False,
        "phase_override": None,
    })
    return source_payload, coefficient_payload


def _state_isolation(
    model: bogt.ScaledChenBOGTModel,
    evaluator: FrozenMNISTEvaluator,
    before: Mapping[str, str],
) -> dict[str, Any]:
    after = {
        "model": _state_digest(model),
        "encoder": _state_digest(model.system.encoder),
        "decoder": _state_digest(model.system.decoder),
        "ols": _state_digest(model.ols),
        "evaluator": _state_digest(evaluator),
    }
    unchanged = all(before[name] == after[name] for name in before)
    frozen = (not model.training and not evaluator.training and all(not parameter.requires_grad for parameter in model.parameters()) and all(not parameter.requires_grad for parameter in evaluator.parameters()))
    return {
        "schema": "lightgen_scaled_chen_direct_ols_state_isolation.v1",
        "model_state_digest_before": before["model"],
        "model_state_digest_after": after["model"],
        "encoder_state_digest_before": before["encoder"],
        "encoder_state_digest_after": after["encoder"],
        "decoder_state_digest_before": before["decoder"],
        "decoder_state_digest_after": after["decoder"],
        "ols_state_digest_before": before["ols"],
        "ols_state_digest_after": after["ols"],
        "evaluator_state_digest_before": before["evaluator"],
        "evaluator_state_digest_after": after["evaluator"],
        "optimizer_absent": True,
        "backward_calls": 0,
        "training_steps": 0,
        "checkpoint_created": False,
        "phase_updated": False,
        "prior_generation": False,
        "model_and_evaluator_frozen": frozen,
        "state_unchanged": bool(unchanged and frozen),
        "gate": "S2_STATE_ISOLATION_PASS" if unchanged and frozen else STATE_INVALID,
    }


def _artifact_refs(directory: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "s2_manifest.json":
            continue
        refs[path.name] = {"path": path.name, "sha256": file_sha256(path), "size_bytes": path.stat().st_size}
    return refs


def _assert_output_tree(directory: Path) -> None:
    allowed = {
        "s2_manifest.json", "source_ols_identity.json", "source_ols_coefficients.pt",
        "direct_reemit_contract.json", "direct_full_fidelity.json", "semantic_evaluator_binding.json",
        "full_path_semantic_metrics.json", "direct_replay_semantic_metrics.json",
        "paired_semantic_preservation.json", "semantic_null_metrics.json",
        "state_isolation.json", "semantic_predictions.pt", "compact_report.json",
    }
    names = {path.name for path in directory.iterdir()}
    if names != allowed:
        raise ValueError(f"DOSD output tree is not exact: {sorted(names)}")
    if any(path.name.startswith(".") or path.suffix in {".tmp", ".tiff", ".tif"} for path in directory.iterdir()):
        raise ValueError("DOSD output contains hidden, temporary, or TIFF content")


def _run(
    config_path: str | Path = CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    authorities, data, formal, preflight = _preflight(config_path, output_path)
    config = authorities.config
    output = _resolve(output_path)
    device = torch.device("cuda")
    checkpoint = authorities.checkpoint_payload
    model = mra._load_scaled_model(formal, checkpoint, device)
    bogt.require_mode_bank_sha256(model, MODE_BANK_SHA256)
    evaluator = FrozenMNISTEvaluator().to(device)
    evaluator.load_state_dict(authorities.evaluator_payload["state_dict"], strict=True)
    evaluator.freeze()
    before = {
        "model": _state_digest(model),
        "encoder": _state_digest(model.system.encoder),
        "decoder": _state_digest(model.system.decoder),
        "ols": _state_digest(model.ols),
        "evaluator": _state_digest(evaluator),
    }
    scale = authorities.calibration_payload["port_scale"].to(torch.float32).cpu()
    if tuple(scale.shape) != (PORT_COUNT,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("DOSD repaired calibration scale is invalid")
    replays: dict[str, SplitReplay] = {}
    for split in ("selection", "confirmation"):
        images, labels, indices = _split_inputs(data, split)
        replays[split] = _collect_split(model, evaluator, images, labels, indices, device=device)
    source_identity, coefficient_payload = _source_identity(authorities, replays, scale)
    if not source_identity["normalized_r3_exact_identity"]:
        raise ValueError("DOSD source coefficients do not exactly reproduce S1 R3 identity")

    null_permutations: dict[str, torch.Tensor] = {
        "selection": _fixed_permutation_indices(SELECTION_COUNT, seed=NULL_SEED),
        "confirmation": authorities.vpim_null_payload["permutation_indices"].clone().cpu(),
    }
    null_labels = {
        split: replays[split].labels.index_select(0, null_permutations[split])
        for split in ("selection", "confirmation")
    }
    null_metrics = {
        split: _null_metrics(replays[split], null_labels[split], null_permutations[split])
        for split in ("selection", "confirmation")
    }
    paired: dict[str, Any] = {}
    all_paired = True
    all_tensors = True
    for split, replay in replays.items():
        delta = replay.direct_predictions.eq(replay.labels).to(torch.float64) - replay.full_predictions.eq(replay.labels).to(torch.float64)
        paired_ci = stratified_bootstrap_delta_ci(delta, replay.labels)
        agreement = float(replay.direct_predictions.eq(replay.full_predictions).to(torch.float64).mean())
        tensors_identical = all(bool(replay.fidelity[name]["allclose"]) for name in ("decoder_input", "decoder_output", "camera_output", "evaluator_view"))
        predictions_identical = bool(torch.equal(replay.direct_predictions, replay.full_predictions))
        paired[split] = {
            "sample_count": int(replay.labels.numel()),
            "direct_full_prediction_agreement": agreement,
            "direct_full_prediction_identity": predictions_identical,
            "paired_correctness_delta": paired_ci,
            "registered_rule": "lower95_paired_correctness_delta_greater_equal_zero",
            "tensors_identical_within_registered_tolerance": tensors_identical,
        }
        all_paired = all_paired and float(paired_ci["lower95"]) >= 0.0
        all_tensors = all_tensors and tensors_identical

    selection_direct_delta = float(null_metrics["selection"]["direct"]["bootstrap"]["point_delta"])
    confirmation_direct_lower95 = float(null_metrics["confirmation"]["direct"]["bootstrap"]["lower95"])
    fidelity_pass = all(replay.fidelity["allclose"] and replay.api_equivalent["allclose"] for replay in replays.values())
    finite = all(
        torch.isfinite(replay.coefficients).all() and torch.isfinite(replay.labels.to(torch.float32)).all() and torch.isfinite(replay.full_predictions.to(torch.float32)).all() and torch.isfinite(replay.direct_predictions.to(torch.float32)).all()
        for replay in replays.values()
    )
    classification = classify_s2(
        fidelity_pass=fidelity_pass,
        evaluator_binding_pass=True,
        source_identity_pass=bool(source_identity["normalized_r3_exact_identity"]),
        selection_delta=selection_direct_delta,
        confirmation_lower95=confirmation_direct_lower95,
        paired_lower95=min(float(paired[split]["paired_correctness_delta"]["lower95"]) for split in paired),
        state_isolation_pass=True,
        finite=bool(finite),
        predictions_identical=all(bool(paired[split]["direct_full_prediction_identity"]) for split in paired),
        tensors_identical=all_tensors,
    )
    state = _state_isolation(model, evaluator, before)
    if not state["state_unchanged"]:
        classification = classify_s2(
            fidelity_pass=fidelity_pass, evaluator_binding_pass=True,
            source_identity_pass=bool(source_identity["normalized_r3_exact_identity"]),
            selection_delta=selection_direct_delta, confirmation_lower95=confirmation_direct_lower95,
            paired_lower95=min(float(paired[split]["paired_correctness_delta"]["lower95"]) for split in paired),
            state_isolation_pass=False, finite=bool(finite), predictions_identical=False, tensors_identical=False,
        )

    evaluator_binding = {
        "schema": "lightgen_scaled_chen_direct_ols_semantic_evaluator_binding.v1",
        "gate": OSERA_GATE,
        "classification": OSERA_CLASSIFICATION,
        "scope": "PAIRED_S2_SEMANTIC_MEASUREMENT",
        "evaluator": {
            "schema": EVALUATOR_SCHEMA, "architecture": "FrozenMNISTEvaluator",
            "checkpoint_path": _relative(authorities.source_paths["evaluator_checkpoint"]),
            "checkpoint_sha256": EVALUATOR_CHECKPOINT_SHA256,
            "historical_validation_accuracy": 0.9712,
            "historical_threshold": 0.985,
            "historical_fosea_classification": FOSEA_CLASSIFICATION,
        },
        "preprocessing": {
            "source": "output.camera",
            "exposure_scale": EXPOSURE_SCALE,
            "operation": "global_exposure_scale_clamp_0_1_adaptive_avg_pool_28x28",
            "per_sample_normalization": False,
            "prediction_correction": False,
            "threshold_adjustment": False,
        },
        "osera_manifest_sha256": OSERA_MANIFEST_SHA256,
        "osera_error_model_sha256": OSERA_ERROR_MODEL_SHA256,
        "osera_binding_pass": True,
    }
    direct_contract = {
        "schema": "lightgen_scaled_chen_direct_ols_reemit_contract.v1",
        "input": "exact observed output.coefficients",
        "dtype": "torch.complex64",
        "shape": ["N", PORT_COUNT],
        "path": "complex c -> fixed OLS output modes -> frozen decoder -> Chen relay -> camera intensity -> fixed evaluator view",
        "decoder_interface": "ScaledChenBOGTModel.reemit_decoder",
        "primitive": "local exact equivalent used for trace; API checked on first batch",
        "phase": "relative phase preserved",
        "phase_override": None,
        "power_only_injection": False,
        "prior_or_synthetic_input": False,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "mode_bank_sha256": MODE_BANK_SHA256,
        "decoder_architecture_modified": False,
        "fiber_ols_relay_detector_modified": False,
    }
    fidelity = {
        "schema": "lightgen_scaled_chen_direct_ols_reemit_fidelity.v1",
        "rtol": FIDELITY_RTOL,
        "atol": FIDELITY_ATOL,
        "no_posthoc_tolerance_relaxation": True,
        "gate": DIRECT_FIDELITY_GATE if fidelity_pass else "DIRECT_REEMIT_FIDELITY_FAIL",
        "selection": replays["selection"].fidelity,
        "confirmation": replays["confirmation"].fidelity,
        "api_equivalence": {split: replays[split].api_equivalent for split in replays},
    }
    full_metrics = {split: {
        "source_split": split,
        "source_label_authority": True,
        "metrics": _semantic_metrics(replays[split].full_predictions, replays[split].labels),
    } for split in replays}
    direct_metrics = {split: {
        "source_split": split,
        "source_label_authority": True,
        "metrics": _semantic_metrics(replays[split].direct_predictions, replays[split].labels),
    } for split in replays}
    prediction_payload = {
        "schema": PREDICTION_SCHEMA,
        "selection": {"labels": replays["selection"].labels, "full_predictions": replays["selection"].full_predictions, "direct_predictions": replays["selection"].direct_predictions, "null_labels": null_labels["selection"]},
        "confirmation": {"labels": replays["confirmation"].labels, "full_predictions": replays["confirmation"].full_predictions, "direct_predictions": replays["confirmation"].direct_predictions, "null_labels": null_labels["confirmation"]},
        "null_permutations": null_permutations,
    }
    report = {
        "schema": "lightgen_scaled_chen_direct_ols_semantic_decode_compact_report.v1",
        "task_id": TASK_ID,
        "gate": classification["gate"],
        "classification": classification["classification"],
        "next_action": classification["next_action"],
        "reason": classification["reason"],
        "direct_replay_semantic_accessibility": classification["replay_semantic_accessibility_supported"],
        "s1_authority": {"gate": S1_GATE, "minimal_representation": "R1_POWER", "phase_attribution": S1_PHASE_ATTRIBUTION, "phase_gauge": S1_PHASE_GAUGE, "manifest_sha256": S1_MANIFEST_SHA256},
        "osera_authority": {"gate": OSERA_GATE, "classification": OSERA_CLASSIFICATION, "manifest_sha256": OSERA_MANIFEST_SHA256},
        "full_path_semantics": {split: full_metrics[split]["metrics"]["overall_accuracy"] for split in replays},
        "direct_replay_semantics": {split: direct_metrics[split]["metrics"]["overall_accuracy"] for split in replays},
        "paired_lower95": {split: paired[split]["paired_correctness_delta"]["lower95"] for split in replays},
        "source_label_null_direct": {split: null_metrics[split]["direct"]["bootstrap"] for split in replays},
        "interpretation": {
            "complex_ols_direct_decode": "source replay only; observed c is not generated",
            "label_to_ols_controller_semantics": "UNESTABLISHED",
            "code8_to_ols_control": "UNRESOLVED",
            "generative_manifold_supported": False,
            "generation_ready": False,
        },
        "governance": config["governance"],
    }
    state["gate"] = "S2_STATE_ISOLATION_PASS" if state["state_unchanged"] else STATE_INVALID
    artifacts_to_write: dict[str, tuple[str, Any]] = {
        "source_ols_identity.json": ("json", source_identity),
        "source_ols_coefficients.pt": ("torch", coefficient_payload),
        "direct_reemit_contract.json": ("json", direct_contract),
        "direct_full_fidelity.json": ("json", fidelity),
        "semantic_evaluator_binding.json": ("json", evaluator_binding),
        "full_path_semantic_metrics.json": ("json", {"schema": "lightgen_scaled_chen_direct_ols_full_path_semantic_metrics.v1", "splits": full_metrics}),
        "direct_replay_semantic_metrics.json": ("json", {"schema": "lightgen_scaled_chen_direct_ols_direct_replay_semantic_metrics.v1", "splits": direct_metrics}),
        "paired_semantic_preservation.json": ("json", {"schema": "lightgen_scaled_chen_direct_ols_paired_semantic_preservation.v1", "splits": paired, "all_lower95_nonnegative": all_paired}),
        "semantic_null_metrics.json": ("json", {"schema": "lightgen_scaled_chen_direct_ols_semantic_null_metrics.v1", "splits": null_metrics, "seed": NULL_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES}),
        "state_isolation.json": ("json", state),
        "semantic_predictions.pt": ("torch", prediction_payload),
        "compact_report.json": ("json", report),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, (kind, payload) in artifacts_to_write.items():
            target = temporary / name
            if kind == "json":
                atomic_write_new_json(target, payload)
            else:
                atomic_torch_save_new(target, payload)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "task_id": TASK_ID,
            "hypothesis": HYPOTHESIS,
            "status": "valid_completed",
            "gate": classification["gate"],
            "classification": classification["classification"],
            "next_action": classification["next_action"],
            "source": {
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "checkpoint_global_step": 5000,
                "mode_bank_sha256": MODE_BANK_SHA256,
                "s1_manifest_sha256": S1_MANIFEST_SHA256,
                "osera_manifest_sha256": OSERA_MANIFEST_SHA256,
                "selection_count": SELECTION_COUNT,
                "confirmation_count": CONFIRMATION_COUNT,
                "data_manifest_sha256": DATA_MANIFEST_SHA256,
                "source_coefficients_artifact": "source_ols_coefficients.pt",
            },
            "direct_reemit": direct_contract,
            "fidelity": {"artifact": "direct_full_fidelity.json", "gate": fidelity["gate"], "rtol": FIDELITY_RTOL, "atol": FIDELITY_ATOL},
            "evaluator": evaluator_binding,
            "statistics": {"bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES, "null_seed": NULL_SEED, "paired_rule": "lower95_paired_correctness_delta_greater_equal_zero"},
            "state_isolation": {"artifact": "state_isolation.json", "gate": state["gate"]},
            "layout": {"all_paths_relative": True, "manifest_self_hash_excluded": True, "write_once": True, "artifacts": _artifact_refs(temporary)},
            "execution": {"device": "cuda", "batch_size": BATCH_SIZE, "wall_clock_seconds": time.perf_counter() - started, "preflight": preflight},
            "interpretation": report["interpretation"],
            "governance": config["governance"],
        }
        atomic_write_new_json(temporary / "s2_manifest.json", manifest)
        _assert_output_tree(temporary)
        if output.exists():
            raise FileExistsError(f"DOSD write-once output appeared during run: {_relative(output)}")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    _assert_output_tree(output)
    manifest = json.loads((output / "s2_manifest.json").read_text(encoding="utf-8"))
    return {"output": _relative(output), "manifest": manifest, "report": report}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "run"), required=True)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.action == "preflight":
        _, _, _, payload = _preflight(args.config, args.output)
        print(json.dumps({"task_id": TASK_ID, "status": "preflight_pass", **payload}, indent=2, sort_keys=True))
        return 0
    result = _run(args.config, args.output)
    print(json.dumps(result["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
