"""Replay the registered MRA under the repaired calibration coordinate.

This Stage-A consumer changes only the loaded calibration identity.  It uses
the immutable parent MRA panel and the immutable step-5000 checkpoint, then
executes only the SLM-to-pre-OLS latent prefix.  No decoder, relay, CMOS,
MPCDAR, prior sampler, optimizer, backward pass, or training path is entered.
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

from ldm_project.v2.application.research.optical_manifold import optical_manifold as base
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research.optical_manifold import optical_manifold_smf_bogt_manifold_readiness as mra
from ldm_project.v2.evidence import (
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research.scaled_chen import repair_scaled_chen_calibration_unit_aware as r2


TASK_ID = "H-LOMA-GEN-RCR-A"
HYPOTHESIS = TASK_ID
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_mra_v1"
REPLAY_MANIFEST_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_mra_manifest.v1"
METRICS_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_mra_metrics.v1"
COMPARISON_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_mra_comparison.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_mra_compact_report.v1"
CHECKPOINT_SCHEMA = "scaled_chen_bogt_horizon_checkpoint.v1"
OLD_CALIBRATION_SCHEMA = "scaled_chen_bogt_calibration.v1"
NEW_CALIBRATION_SCHEMA = "scaled_chen_bogt_calibration.v2"
NEW_CALIBRATION_CONTRACT = "unit_aware_raw_mean_v2"
R2_MANIFEST_SCHEMA = "lightgen_scaled_chen_p1_calibration_contract_repair_manifest.v1"
PORT_COUNT = 100
SELECTION_COUNT = 1000
CONFIRMATION_COUNT = 5000
SELECTION_SEED = 1235
PERMUTATION_SEED = 1234
MRA_PRIOR_MEAN = 1.0
MRA_PRIOR_VARIANCE = 0.25
COORDINATE_METRIC_NAMES = (
    "z_mean",
    "z_variance",
    "gaussian_kl",
    "numerical_truncated_normal_kl",
    "power_effective_rank",
)
INVARIANT_METRIC_NAMES = (
    "raw_complex_coefficients",
    "raw_modal_coefficient_powers",
    "complex_modal_effective_rank",
    "checkpoint_digest",
    "learned_state_digest",
)
MRA_REPAIRED_COORDINATE_VALID = "MRA_REPAIRED_COORDINATE_VALID"
MRA_PRIOR_CONTRACT_REVIEW_REQUIRED = "MRA_PRIOR_CONTRACT_REVIEW_REQUIRED"
MRA_REPLAY_INVALID = "MRA_REPLAY_INVALID"
NEXT_ACTION_VALID = "IMPLEMENT_REPAIRED_COORDINATE_G1_SA_REPLAY"
NEXT_ACTION_SHAPE_REVIEW = "PLAN_PRIOR_SHAPE_CONTRACT_REVIEW"
NEXT_ACTION_INVALID = "REVIEW_MRA_REPLAY_FAILURE"
DEFAULT_CONFIG_PATH = Path(
    "configs/research/scaled_chen/lightgen_scaled_chen_repaired_coordinate_mra_v1.yaml"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_repaired_coordinate_mra_v1_attempt_01"
)


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


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


def _json_file(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"JSON artifact hash mismatch: {_relative_path(path)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {_relative_path(path)}")
    return payload


def _require_hash(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value.lower()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _assert_config(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("RCR-A config schema is invalid")
    if payload.get("task_id") != TASK_ID or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("RCR-A config task identity is invalid")

    parent = _require_mapping(payload.get("parent_mra"), "parent_mra")
    if parent.get("status") != "valid_failed" or parent.get("prior_status") != "generation_prior_not_accessible":
        raise ValueError("parent MRA status boundary is invalid")

    repaired = _require_mapping(payload.get("r2_calibration"), "r2_calibration")
    if repaired.get("manifest_sha256") != "0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6":
        raise ValueError("R2 manifest hash is not registered")
    if repaired.get("payload_sha256") != "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07":
        raise ValueError("R2 payload hash is not registered")
    if repaired.get("schema") != NEW_CALIBRATION_SCHEMA or repaired.get("contract_version") != NEW_CALIBRATION_CONTRACT:
        raise ValueError("R2 calibration identity is invalid")
    if repaired.get("sample_count") != 1000 or repaired.get("port_count") != PORT_COUNT:
        raise ValueError("R2 calibration dimensions are invalid")
    if repaired.get("calibration_indices_sha256") != "369c5bded5673f901ae5236eaa51a0c8da17934dc2fab139a205c9fce41d5543":
        raise ValueError("R2 calibration indices hash is not registered")
    if repaired.get("dataset_manifest_sha256") != "741e6402da754a740a677dbb343d48c35983f62b59911bbc129dfcc44f9e9ae9":
        raise ValueError("R2 dataset hash is not registered")
    if repaired.get("mode_bank_sha256") != "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9":
        raise ValueError("R2 mode-bank hash is not registered")
    if repaired.get("zero_phase_reference") != "independent_train_only_zero_phase" or repaired.get("plane_identity") != "pre_ols_plane_power_reference_v1":
        raise ValueError("R2 optical identity is invalid")

    checkpoint = _require_mapping(payload.get("checkpoint"), "checkpoint")
    if (
        checkpoint.get("path")
        != "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt"
        or checkpoint.get("schema") != CHECKPOINT_SCHEMA
        or checkpoint.get("global_step") != 5000
    ):
        raise ValueError("RCR-A checkpoint identity is not locked")

    dataset = _require_mapping(payload.get("dataset"), "dataset")
    expected_dataset = {
        "dataset_root": "data",
        "mra_manifest_sha256": "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb",
        "selection_count": SELECTION_COUNT,
        "selection_per_label": 100,
        "selection_source": "mnist_test_indices_5000_9999",
        "selection_seed": SELECTION_SEED,
        "confirmation_count": CONFIRMATION_COUNT,
        "confirmation_source": "horizon_heldout_train_indices_55000_59999",
        "confirmation_indices_sha256": "ff7ab50b95d96e727c240edc2baa916d1e6c11402b8f4f9c2ce6e2e7e5e2218a",
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }
    if dict(dataset) != expected_dataset:
        raise ValueError("RCR-A dataset contract is not locked")

    prior = _require_mapping(payload.get("prior_contract"), "prior_contract")
    if dict(prior) != {
        "kind": "truncated_normal",
        "mean": MRA_PRIOR_MEAN,
        "variance": MRA_PRIOR_VARIANCE,
        "lower": 0.0,
        "unconditional": True,
        "phase": "fixed_calibrated_zero_residual",
    }:
        raise ValueError("RCR-A prior contract changed")

    measurement = _require_mapping(payload.get("measurement"), "measurement")
    if measurement.get("objective") != "frozen_latent_prefix_only" or measurement.get("prefix") != "slm_to_front_to_encoder_to_pre_ols_to_complex_coefficients":
        raise ValueError("RCR-A measurement prefix is invalid")
    if measurement.get("coordinate_metrics") != list(COORDINATE_METRIC_NAMES) or measurement.get("invariant_controls") != list(INVARIANT_METRIC_NAMES):
        raise ValueError("RCR-A metric contract is invalid")
    if any(measurement.get(name) is not expected for name, expected in {
        "no_update": True,
        "decoder_entered": False,
        "relay_after_pre_ols": False,
        "cmos_called": False,
        "mpcdar_called": False,
        "reconstruction_supported": False,
    }.items()):
        raise ValueError("RCR-A reconstruction boundary is invalid")
    if measurement.get("reconstruction_context") != "HISTORICAL_CONTEXT_ONLY":
        raise ValueError("RCR-A historical reconstruction context is invalid")

    gates = _require_mapping(payload.get("gates"), "gates")
    if gates.get("old_replay_rtol") != 1.0e-5 or gates.get("old_replay_atol") != 1.0e-8 or gates.get("invariant_rank_rtol") != 1.0e-5 or gates.get("invariant_rank_atol") != 1.0e-8:
        raise ValueError("RCR-A numerical comparison tolerance changed")
    shape = _require_mapping(gates.get("prior_shape_review_trigger"), "prior_shape_review_trigger")
    if dict(shape) != {
        "mean_alignment_absolute_tolerance": 0.10,
        "variance_shape_relative_tolerance": 0.25,
        "variance_dominance_ratio": 1.0,
        "minimum_numerical_kl": 1.0,
        "require_both_panels": True,
        "descriptive_only": True,
    }:
        raise ValueError("RCR-A prior-shape review trigger changed")

    execution = _require_mapping(payload.get("execution"), "execution")
    if dict(execution) != {
        "device": "cuda",
        "batch_size": 10,
        "min_free_cuda_bytes": 4294967296,
        "max_peak_cuda_bytes": 5368709120,
        "min_disk_free_bytes": 10737418240,
        "max_storage_bytes": 536870912,
        "require_cublas_workspace_config": ":4096:8",
    }:
        raise ValueError("RCR-A execution contract is invalid")

    authorization = _require_mapping(payload.get("authorization"), "authorization")
    if authorization.get("diagnostic_only") is not True or authorization.get("training_authorized") is not False or authorization.get("formal_promotion") is not False or authorization.get("stage_a_only") is not True or authorization.get("stage_b_g1_replay") is not False:
        raise ValueError("RCR-A authorization boundary is invalid")
    if any(authorization.get(name) is not False for name in ("prior_generation", "optimizer_created", "backward_called", "refit", "checkpoint_created")):
        raise ValueError("RCR-A forbidden operation authorization is invalid")

    governance = _require_mapping(payload.get("governance"), "governance")
    if dict(governance) != {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }:
        raise ValueError("RCR-A governance boundary is invalid")


def load_replay_config(path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[Path, dict[str, Any], str, str]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RCR-A config must be a YAML mapping")
    required = {
        "schema",
        "task_id",
        "hypothesis",
        "parent_mra",
        "r2_calibration",
        "checkpoint",
        "dataset",
        "prior_contract",
        "measurement",
        "gates",
        "execution",
        "authorization",
        "governance",
    }
    if set(payload) != required:
        raise ValueError("RCR-A config fields are invalid")
    _assert_config(payload)
    return target, payload, file_sha256(target), canonical_sha256(payload)


def validate_repaired_calibration_identity(
    payload: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_contract: str,
    expected_calibration_indices_sha256: str,
    expected_dataset_manifest_hash: str,
    expected_mode_bank_sha256: str,
) -> None:
    """Validate the identity fields that cross the R2-to-RCR process boundary."""
    if payload.get("schema") != expected_schema:
        raise ValueError("calibration schema does not match the repaired contract")
    if payload.get("contract_version") != expected_contract:
        raise ValueError("calibration contract version does not match")
    if payload.get("sample_count") != 1000 or payload.get("port_count") != PORT_COUNT:
        raise ValueError("calibration dimensions do not match")
    if payload.get("calibration_indices_sha256") != expected_calibration_indices_sha256:
        raise ValueError("calibration indices hash does not match")
    if payload.get("dataset_manifest_hash") != expected_dataset_manifest_hash:
        raise ValueError("calibration dataset manifest hash does not match")
    if payload.get("mode_bank_sha256") != expected_mode_bank_sha256:
        raise ValueError("calibration mode-bank hash does not match")
    if payload.get("zero_phase_reference") != "independent_train_only_zero_phase" or payload.get("plane_identity") != "pre_ols_plane_power_reference_v1":
        raise ValueError("calibration optical identity does not match")
    raw = payload.get("raw_port_mean")
    scale = payload.get("port_scale")
    if not torch.is_tensor(raw) or not torch.is_tensor(scale) or raw.shape != (PORT_COUNT,) or scale.shape != (PORT_COUNT,) or not torch.equal(raw, scale):
        raise ValueError("calibration raw_port_mean and port_scale are not elementwise equal")


def validate_checkpoint_payload(payload: Mapping[str, Any], *, expected_step: int) -> None:
    if payload.get("schema") != CHECKPOINT_SCHEMA or int(payload.get("global_step", -1)) != expected_step or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("checkpoint schema, step, or model_state is invalid")


def validate_latent_bank(value: torch.Tensor) -> None:
    if value.ndim != 2 or value.shape[1] != PORT_COUNT:
        raise ValueError("repaired latent bank must have shape [N,100]")
    if not torch.is_floating_point(value):
        raise ValueError("repaired latent bank must be real")
    if not torch.isfinite(value).all():
        raise ValueError("repaired latent bank must be finite")
    if (value < 0).any():
        raise ValueError("repaired latent bank must be nonnegative")


def repaired_latent_anchor(value: torch.Tensor) -> dict[str, Any]:
    validate_latent_bank(value)
    per_port_mean = value.mean(dim=0)
    per_port_variance = value.var(dim=0, unbiased=False)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": True,
        "nonnegative": True,
        "global_mean": float(value.mean()),
        "global_variance": float(value.var(unbiased=False)),
        "per_port_mean": _range_summary(per_port_mean),
        "per_port_variance": _range_summary(per_port_variance),
    }


def state_digest(module: torch.nn.Module) -> str:
    return canonical_sha256(
        {name: tensor_sha256(value) for name, value in module.state_dict().items()}
    )


def mra_participation_rank(value: torch.Tensor) -> float:
    """Return the parent MRA centered-SVD participation rank exactly."""
    features = mra._feature_projection(value)
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("MRA participation rank requires a two-dimensional bank")
    centered = features - features.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    total = energy.sum().clamp_min(1.0e-20)
    probability = energy / total
    return float(1.0 / probability.square().sum().clamp_min(1.0e-20))


def _range_summary(value: torch.Tensor) -> dict[str, float]:
    view = value.detach().to(torch.float64).cpu()
    quantiles = torch.quantile(view, torch.tensor([0.50], dtype=torch.float64))
    return {
        "min": float(view.min()),
        "median": float(quantiles[0]),
        "max": float(view.max()),
    }


def _gaussian_kl_decomposition(mean: torch.Tensor, variance: torch.Tensor) -> dict[str, float]:
    mean_term = 0.5 * ((mean - MRA_PRIOR_MEAN).square() / MRA_PRIOR_VARIANCE).mean()
    variance_term = 0.5 * (
        variance / MRA_PRIOR_VARIANCE
        - 1.0
        + math.log(MRA_PRIOR_VARIANCE)
        - variance.log()
    ).mean()
    total = mean_term + variance_term
    return {
        "mean_offset_term": float(mean_term),
        "variance_shape_term": float(variance_term),
        "variance_shape_fraction": float(variance_term / total.clamp_min(1.0e-20)),
    }


def coordinate_metrics(
    coefficients: torch.Tensor,
    labels: torch.Tensor,
    calibration: bogt.Calibration | bogt.UnitAwareCalibration,
) -> dict[str, Any]:
    if coefficients.ndim != 2 or coefficients.shape[1] != PORT_COUNT or not torch.is_complex(coefficients):
        raise ValueError("MRA coefficients must be complex [N,100]")
    if labels.ndim != 1 or labels.shape[0] != coefficients.shape[0]:
        raise ValueError("MRA labels must align with coefficients")
    if not torch.isfinite(coefficients).all():
        raise ValueError("MRA coefficients must be finite")
    powers = coefficients.abs().square()
    scale = calibration.port_scale.to(device=powers.device, dtype=powers.dtype)
    z = bogt.normalize_port_power(powers, scale).detach().cpu().contiguous()
    validate_latent_bank(z)
    mean, variance = bogt.aggregate_moments(z)
    gaussian_kl = bogt.bogt_kl_surrogate(mean, variance)
    numerical_kl = bogt.truncated_normal_kl_reference(mean, variance)
    return {
        "sample_count": int(z.shape[0]),
        "port_count": int(z.shape[1]),
        "latent_definition": "z_j=abs(c_j)^2/s_j",
        "latent_dtype": str(z.dtype),
        "z_mean": float(mean.mean()),
        "z_variance": float(variance.mean()),
        "gaussian_kl": float(gaussian_kl),
        "bogt_kl_surrogate": float(gaussian_kl),
        "numerical_truncated_normal_kl": float(numerical_kl),
        "complex_effective_rank": mra_participation_rank(coefficients),
        "power_effective_rank": mra_participation_rank(z),
        "z_per_port_mean": _range_summary(mean),
        "z_per_port_variance": _range_summary(variance),
        "z_global_variance": float(z.var(unbiased=False)),
        "z_quantiles": torch.quantile(
            z, torch.tensor([0.01, 0.50, 0.99], dtype=z.dtype)
        ).tolist(),
        "gaussian_kl_decomposition": _gaussian_kl_decomposition(mean, variance),
        "finite": True,
        "nonnegative": True,
        "scale_sha256": tensor_sha256(calibration.port_scale),
    }


def compare_metric(
    name: str,
    old: float,
    new: float,
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-8,
) -> dict[str, Any]:
    if name not in COORDINATE_METRIC_NAMES:
        raise ValueError(f"unsupported coordinate metric: {name}")
    if not math.isfinite(old) or not math.isfinite(new):
        raise ValueError("old and new metric values must be finite")
    delta = new - old
    tolerance = max(atol, abs(old) * rtol)
    if abs(delta) <= tolerance:
        direction = "UNCHANGED_WITHIN_TOLERANCE"
    elif name == "z_mean":
        direction = "IMPROVED" if abs(new - MRA_PRIOR_MEAN) < abs(old - MRA_PRIOR_MEAN) else "WORSENED"
    elif name == "z_variance":
        direction = "IMPROVED" if abs(new - MRA_PRIOR_VARIANCE) < abs(old - MRA_PRIOR_VARIANCE) else "WORSENED"
    elif name in {"gaussian_kl", "numerical_truncated_normal_kl"}:
        direction = "IMPROVED" if new < old else "WORSENED"
    else:
        direction = "IMPROVED" if new > old else "WORSENED"
    return {
        "old": old,
        "new": new,
        "absolute_delta": delta,
        "relative_delta": delta / max(abs(old), 1.0e-30),
        "tolerance": tolerance,
        "direction": direction,
    }


def _metric_comparison(
    old: Mapping[str, Any],
    new: Mapping[str, Any],
    *,
    rtol: float,
    atol: float,
) -> dict[str, dict[str, Any]]:
    return {
        name: compare_metric(name, float(old[name]), float(new[name]), rtol=rtol, atol=atol)
        for name in COORDINATE_METRIC_NAMES
    }


def assess_prior_shape(
    metrics_by_split: Mapping[str, Mapping[str, Any]],
    *,
    mean_tolerance: float = 0.10,
    variance_relative_tolerance: float = 0.25,
    dominance_ratio: float = 1.0,
    minimum_numerical_kl: float = 1.0,
) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    for split in ("selection", "confirmation"):
        metrics = _require_mapping(metrics_by_split.get(split), split)
        mean = float(metrics["z_mean"])
        variance = float(metrics["z_variance"])
        mean_error = abs(mean - MRA_PRIOR_MEAN)
        variance_ratio = variance / MRA_PRIOR_VARIANCE
        variance_shape_error = abs(variance_ratio - 1.0)
        decomposition = _require_mapping(
            metrics["gaussian_kl_decomposition"],
            f"{split}.gaussian_kl_decomposition",
        )
        variance_dominant = float(decomposition["variance_shape_term"]) >= dominance_ratio * float(decomposition["mean_offset_term"])
        evidence[split] = {
            "registered_prior_mean": MRA_PRIOR_MEAN,
            "registered_prior_variance": MRA_PRIOR_VARIANCE,
            "registered_prior_lower": 0.0,
            "z_mean": mean,
            "mean_absolute_error": mean_error,
            "mean_aligned": mean_error <= mean_tolerance,
            "z_variance": variance,
            "variance_ratio_to_registered_prior": variance_ratio,
            "variance_relative_error": variance_shape_error,
            "variance_shape_mismatch": variance_shape_error > variance_relative_tolerance,
            "variance_shape_term": float(decomposition["variance_shape_term"]),
            "mean_offset_term": float(decomposition["mean_offset_term"]),
            "variance_term_dominant": variance_dominant,
            "numerical_truncated_normal_kl": float(metrics["numerical_truncated_normal_kl"]),
            "numerical_kl_supports_shape_review": float(metrics["numerical_truncated_normal_kl"]) >= minimum_numerical_kl,
        }
    dominant = all(
        evidence[split][key]
        for split in ("selection", "confirmation")
        for key in (
            "mean_aligned",
            "variance_shape_mismatch",
            "variance_term_dominant",
            "numerical_kl_supports_shape_review",
        )
    )
    return {
        "dominant_inconsistency": dominant,
        "dominant_assumption": "unconditional_truncated_normal_variance=0.25_with_lower=0.0" if dominant else None,
        "review_trigger": "descriptive_only",
        "evidence": evidence,
    }


def classify_stage_a_gate(
    checks: Mapping[str, bool], prior_shape: Mapping[str, Any]
) -> str:
    required = (
        "calibration_identity",
        "checkpoint_identity",
        "latent_anchor",
        "finite",
        "invariant_controls",
        "learned_state_freeze",
        "mra_metrics_replayed",
    )
    if any(checks.get(name) is not True for name in required):
        return MRA_REPLAY_INVALID
    if prior_shape.get("dominant_inconsistency") is True:
        return MRA_PRIOR_CONTRACT_REVIEW_REQUIRED
    return MRA_REPAIRED_COORDINATE_VALID


def _load_parent_panel(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    payload = safe_torch_load(path, expected_sha256=expected_sha256, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != mra.PANEL_SCHEMA:
        raise ValueError("parent MRA panel schema is invalid")
    required = {
        "schema",
        "selection_indices",
        "selection_labels",
        "confirmation_indices",
        "confirmation_labels",
        "per_sample_mse",
        "source_hashes",
        "affects_selection",
    }
    if set(payload) != required:
        raise ValueError("parent MRA panel fields are invalid")
    return payload


def _validate_parent_panel(
    panel: Mapping[str, Any],
    data: mra.AuditData,
    source: mra.SourceBundle,
) -> dict[str, torch.Tensor]:
    selection_labels = data.test_labels.index_select(0, data.selection_indices)
    confirmation_labels = data.labels.index_select(0, data.confirmation_indices)
    for name, actual, expected in (
        ("selection_indices", panel["selection_indices"], data.selection_indices),
        ("selection_labels", panel["selection_labels"], selection_labels),
        ("confirmation_indices", panel["confirmation_indices"], data.confirmation_indices),
        ("confirmation_labels", panel["confirmation_labels"], confirmation_labels),
    ):
        if not torch.is_tensor(actual) or not torch.equal(actual, expected):
            raise ValueError(f"parent MRA {name} does not match the registered panel")
    if panel["affects_selection"] is not False or panel["source_hashes"] != source.hashes:
        raise ValueError("parent MRA panel selection or source binding is invalid")
    per_sample = _require_mapping(panel["per_sample_mse"], "parent MRA per_sample_mse")
    if set(per_sample) != {"selection", "confirmation"}:
        raise ValueError("parent MRA per-sample metric fields are invalid")
    if any(not isinstance(per_sample[name], Mapping) for name in ("selection", "confirmation")):
        raise ValueError("parent MRA per-sample metrics are not mappings")
    return {
        "selection_labels": selection_labels,
        "confirmation_labels": confirmation_labels,
    }


def _extract_formal_mra_metrics(audit: Mapping[str, Any], split: str) -> dict[str, float]:
    entry = _require_mapping(
        _require_mapping(audit["contract"][split], split)["scaled_chen_step_5000"],
        f"{split}.scaled_chen_step_5000",
    )
    ols = _require_mapping(entry["ols"], f"{split}.ols")
    moments = _require_mapping(ols["power_moments"], f"{split}.power_moments")
    result = {
        "z_mean": float(moments["mean"]),
        "z_variance": float(moments["variance"]),
        "gaussian_kl": float(ols["bogt_kl_surrogate"]),
        "numerical_truncated_normal_kl": float(ols["truncated_normal_numerical_kl"]),
        "complex_effective_rank": float(ols["complex_effective_rank"]),
        "power_effective_rank": float(ols["power_effective_rank"]),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError(f"formal parent MRA {split} metrics are non-finite")
    return result


def _validate_parent_mra(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = _require_mapping(config["parent_mra"], "parent_mra")
    mra_config_path = _resolve(parent["config"])
    mra_spec = mra._load_spec(mra_config_path)
    mra._assert_contract(mra_spec)
    review_path = _resolve(parent["review_contract"])
    mra._load_review(review_path, mra_spec)
    formal, formal_path = mra._load_formal(mra_spec)
    source = mra._load_source(mra_spec, load_checkpoints=False)

    audit_path = _resolve(parent["audit"])
    compact_path = _resolve(parent["compact_report"])
    panel_path = _resolve(parent["panel"])
    audit = _json_file(audit_path, expected_sha256=parent["audit_sha256"])
    compact = _json_file(compact_path, expected_sha256=parent["compact_report_sha256"])
    panel = _load_parent_panel(panel_path, parent["panel_sha256"])
    if audit.get("schema") != mra.AUDIT_SCHEMA or audit.get("hypothesis") != mra.HYPOTHESIS or audit.get("status") != "valid_failed":
        raise ValueError("parent MRA attempt status or schema is invalid")
    if audit.get("generative_manifold_supported") is not False or audit.get("generation_readiness") != "not_ready" or audit.get("training_authorized") is not False or audit.get("formal_promotion") is not False:
        raise ValueError("parent MRA governance boundary is invalid")
    if compact.get("schema") != "scaled_chen_manifold_readiness_compact_report.v1" or compact.get("prior_status") != "generation_prior_not_accessible" or compact.get("panel_sha256") != parent["panel_sha256"]:
        raise ValueError("parent MRA compact report binding is invalid")
    context = _require_mapping(_require_mapping(audit["evidence"], "parent MRA evidence")["context"], "parent MRA context")
    if context.get("dataset_hash") != config["dataset"]["mra_manifest_sha256"]:
        raise ValueError("parent MRA dataset manifest binding is invalid")
    if audit["contract"]["prior_status"] != "generation_prior_not_accessible":
        raise ValueError("parent MRA prior status is invalid")
    formal_metrics = {
        split: _extract_formal_mra_metrics(audit, split)
        for split in ("selection", "confirmation")
    }
    return {
        "spec": mra_spec,
        "formal": formal,
        "formal_path": formal_path,
        "review_path": review_path,
        "source": source,
        "audit": audit,
        "compact": compact,
        "panel": panel,
        "panel_path": panel_path,
        "formal_metrics": formal_metrics,
        "attempt": parent["output_directory"],
    }


def _validate_r2(config: Mapping[str, Any]) -> dict[str, Any]:
    repaired = _require_mapping(config["r2_calibration"], "r2_calibration")
    r2_config_path, r2_config, r2_file_hash, r2_canonical_hash = r2._load_config(repaired["config"])
    manifest_path = _resolve(repaired["manifest"])
    manifest = _json_file(manifest_path, expected_sha256=repaired["manifest_sha256"])
    if manifest.get("schema") != R2_MANIFEST_SCHEMA or manifest.get("task_id") != r2.TASK_ID or manifest.get("status") != "completed" or manifest.get("gate") != r2.R2_GATE:
        raise ValueError("R2 manifest schema, task, or gate is invalid")
    if manifest.get("optical_data_changed") is not False or manifest.get("cuda_replay") is not False or manifest.get("downstream_replay") is not False:
        raise ValueError("R2 calibration-only boundary is invalid")
    if manifest.get("next_action") != "PLAN_REPAIRED_COORDINATE_REPLAY":
        raise ValueError("R2 next-action binding is stale")
    contract = _require_mapping(manifest.get("calibration_contract"), "R2 calibration contract")
    if dict(contract) != {
        "contract_version": NEW_CALIBRATION_CONTRACT,
        "floor_policy": "none_for_positive_finite_raw_means",
        "optical_forward_dtype": "torch.float32/torch.complex64",
        "plane_identity": "pre_ols_plane_power_reference_v1",
        "prior_contract": {
            "kind": "truncated_normal",
            "lower": 0.0,
            "mean": 1.0,
            "unconditional": True,
            "variance": 0.25,
        },
        "raw_scale_field": "raw_port_mean",
        "registered_scale_equals_raw": True,
        "registered_scale_field": "port_scale",
        "schema": NEW_CALIBRATION_SCHEMA,
        "zero_phase_reference": "independent_train_only_zero_phase",
    }:
        raise ValueError("R2 calibration contract semantics are stale")
    if manifest.get("parent_v1", {}).get("sha256") != "a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959" or manifest.get("parent_v1", {}).get("schema") != OLD_CALIBRATION_SCHEMA or manifest.get("parent_v1", {}).get("sha256_unchanged_after_repair") is not True:
        raise ValueError("R2 parent v1 binding is invalid")
    if manifest.get("lineage") != {
        "calibration_indices_sha256": repaired["calibration_indices_sha256"],
        "dataset_manifest_sha256": repaired["dataset_manifest_sha256"],
        "measurement_head": "2fcd16a0bef7de2a6483f5ec42070a895bbcbcac",
        "mode_bank_sha256": repaired["mode_bank_sha256"],
        "repair_generation_head": "877f386657c7b408fdfd1964510a473782607d80",
    }:
        raise ValueError("R2 lineage is stale")
    producer_git = _require_mapping(manifest.get("producer_git"), "R2 producer_git")
    if producer_git.get("repair_generation_clean") is not False:
        raise ValueError("R2 producer cleanliness fact changed unexpectedly")
    governance = _require_mapping(manifest.get("governance"), "R2 governance")
    if dict(governance) != {
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "generative_manifold_supported": False,
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "training_authorized": False,
    }:
        raise ValueError("R2 governance boundary is invalid")
    payload_path = _resolve(repaired["payload"])
    payload = safe_torch_load(payload_path, expected_sha256=repaired["payload_sha256"], map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("R2 calibration payload is not a mapping")
    validate_repaired_calibration_identity(
        payload,
        expected_schema=NEW_CALIBRATION_SCHEMA,
        expected_contract=NEW_CALIBRATION_CONTRACT,
        expected_calibration_indices_sha256=repaired["calibration_indices_sha256"],
        expected_dataset_manifest_hash=repaired["dataset_manifest_sha256"],
        expected_mode_bank_sha256=repaired["mode_bank_sha256"],
    )
    artifacts = _require_mapping(manifest.get("artifacts"), "R2 artifacts")
    payload_ref = _require_mapping(artifacts.get("repaired_calibration.pt"), "R2 payload artifact")
    if payload_ref.get("sha256") != repaired["payload_sha256"] or payload_ref.get("path") != "repaired_calibration.pt" or payload_ref.get("size_bytes") != payload_path.stat().st_size:
        raise ValueError("R2 payload artifact reference is stale")
    calibration = r2.load_registered_unit_aware_calibration(
        payload_path,
        expected_sha256=repaired["payload_sha256"],
        expected_dataset_manifest_hash=repaired["dataset_manifest_sha256"],
        expected_calibration_indices_sha256=repaired["calibration_indices_sha256"],
        expected_mode_bank_sha256=repaired["mode_bank_sha256"],
        expected_parent_v1_sha256="a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959",
    )
    return {
        "config_path": r2_config_path,
        "config": r2_config,
        "config_file_sha256": r2_file_hash,
        "config_canonical_sha256": r2_canonical_hash,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": repaired["manifest_sha256"],
        "payload_path": payload_path,
        "payload": payload,
        "payload_sha256": repaired["payload_sha256"],
        "calibration": calibration,
        "repair_generation_clean": producer_git["repair_generation_clean"],
    }


def _load_old_calibration_from_source(source: mra.SourceBundle) -> tuple[bogt.Calibration, Path]:
    path = source.paths["scaled_chen_calibration"]
    payload = safe_torch_load(path, expected_sha256=source.hashes["scaled_chen_calibration"], map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("schema") != OLD_CALIBRATION_SCHEMA:
        raise ValueError("historical v1 calibration schema is invalid")
    calibration = bogt.Calibration(
        port_scale=payload["port_scale"],
        plane_reference=payload["plane_reference"],
        sample_count=int(payload["sample_count"]),
        plane_lower=payload.get("plane_lower"),
        plane_upper=payload.get("plane_upper"),
    )
    calibration.validate()
    return calibration, path


def _load_checkpoint(config: Mapping[str, Any], source: mra.SourceBundle) -> tuple[dict[str, Any], Path]:
    checkpoint_config = config["checkpoint"]
    path = _resolve(checkpoint_config["path"])
    if path != source.paths["scaled_chen_step_5000"] or source.hashes["scaled_chen_step_5000"] != checkpoint_config["sha256"]:
        raise ValueError("checkpoint path/hash is not the parent MRA step-5000 binding")
    payload = safe_torch_load(path, expected_sha256=checkpoint_config["sha256"], map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint payload is not a mapping")
    validate_checkpoint_payload(payload, expected_step=5000)
    return dict(payload), path


def _resource_preflight(config: Mapping[str, Any], output_path: Path) -> tuple[int, int, int]:
    execution = config["execution"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != execution["require_cublas_workspace_config"]:
        raise RuntimeError("BLOCKED_REPLAY_ENVIRONMENT: CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    device = torch.device(execution["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_UNAVAILABLE")
    free, total = torch.cuda.mem_get_info(device)
    if free < execution["min_free_cuda_bytes"]:
        raise RuntimeError("BLOCKED_REPLAY_RESOURCE: less than 4 GiB free CUDA memory")
    disk = shutil.disk_usage(base._repo_root().anchor or str(base._repo_root()))
    if disk.free < execution["min_disk_free_bytes"]:
        raise RuntimeError("BLOCKED_REPLAY_RESOURCE: less than 10 GiB free disk")
    if output_path.exists():
        raise FileExistsError(f"write-once output already exists: {_relative_path(output_path)}")
    return int(free), int(total), int(disk.free)


def _encoded_complex_prefix(
    model: bogt.ScaledChenBOGTModel, amplitude: torch.Tensor
) -> torch.Tensor:
    if amplitude.ndim != 4 or tuple(amplitude.shape[1:]) != (1, 200, 200) or torch.is_complex(amplitude) or not torch.isfinite(amplitude).all() or amplitude.min() < 0.0 or amplitude.max() > 1.0:
        raise ValueError("encoded prefix requires finite [N,1,200,200] amplitude in [0,1]")
    slm = model.system.slm(model.system.command(amplitude))
    front = model.system.front(slm)
    encoded, encoder_records, _ = model.system.encoder(front, "encoder")
    if len(encoder_records) != 2:
        raise ValueError("registered encoder prefix must contain two layers")
    encoded = bogt._attenuate(encoded, bogt.FIELD_TRANSMISSION**2, "fixed_encoder_group_transmission")
    coefficients = model._coefficients(encoded)
    if coefficients.ndim != 2 or coefficients.shape[1] != PORT_COUNT or not torch.is_complex(coefficients) or not torch.isfinite(coefficients).all():
        raise ValueError("encoded prefix produced invalid complex coefficients")
    return coefficients


def _collect_coefficients(
    model: bogt.ScaledChenBOGTModel,
    images: torch.Tensor,
    indices: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch_indices in indices.split(batch_size):
            rendered = base.render_intensity_batch(images.index_select(0, batch_indices))
            amplitude = rendered.amplitude.to(device)
            coefficients = _encoded_complex_prefix(model, amplitude)
            chunks.append(coefficients.detach().cpu().contiguous())
            del rendered, amplitude, coefficients
    result = torch.cat(chunks, dim=0)
    if result.shape != (indices.numel(), PORT_COUNT):
        raise ValueError("encoded prefix result shape is invalid")
    if not torch.isfinite(result).all():
        raise ValueError("encoded prefix result is non-finite")
    return result


def _old_replay_consistency(
    formal_metrics: Mapping[str, Mapping[str, float]],
    replay_metrics: Mapping[str, Mapping[str, Any]],
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in ("selection", "confirmation"):
        comparisons: dict[str, Any] = {}
        for name in (*COORDINATE_METRIC_NAMES, "complex_effective_rank"):
            comparisons[name] = compare_metric(
                name if name in COORDINATE_METRIC_NAMES else "power_effective_rank",
                float(formal_metrics[split][name]),
                float(replay_metrics[split][name]),
                rtol=rtol,
                atol=atol,
            )
            if name == "complex_effective_rank":
                comparisons[name]["metric_name"] = "complex_effective_rank"
        result[split] = {
            "formal_artifact": dict(formal_metrics[split]),
            "same_prefix_replay": dict(replay_metrics[split]),
            "comparisons": comparisons,
            "consistent": all(
                bool(item["direction"] == "UNCHANGED_WITHIN_TOLERANCE")
                for item in comparisons.values()
            ),
        }
    result["all_consistent"] = all(item["consistent"] for item in result.values())
    return result


def _invariant_controls(
    coefficients: Mapping[str, torch.Tensor],
    old_formal: Mapping[str, Mapping[str, float]],
    old_replay: Mapping[str, Mapping[str, Any]],
    new_metrics: Mapping[str, Mapping[str, Any]],
    *,
    checkpoint_expected: str,
    checkpoint_observed: str,
    state_before: str,
    state_after: str,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "raw_coefficient_same_for_old_and_new": True,
        "raw_power_same_for_old_and_new": True,
        "checkpoint_digest": {
            "expected": checkpoint_expected,
            "observed": checkpoint_observed,
            "match": checkpoint_expected == checkpoint_observed,
        },
        "learned_state_digest": {
            "before": state_before,
            "after": state_after,
            "match": state_before == state_after,
        },
        "panels": {},
    }
    for split, value in coefficients.items():
        powers = value.abs().square().contiguous()
        raw_coefficients_hash = tensor_sha256(value)
        raw_powers_hash = tensor_sha256(powers)
        replay_rank = float(old_replay[split]["complex_effective_rank"])
        formal_rank = float(old_formal[split]["complex_effective_rank"])
        rank_check = compare_metric(
            "power_effective_rank",
            formal_rank,
            replay_rank,
            rtol=rtol,
            atol=atol,
        )
        result["panels"][split] = {
            "raw_complex_coefficients_sha256": raw_coefficients_hash,
            "raw_modal_coefficient_powers_sha256": raw_powers_hash,
            "raw_complex_coefficients_shape": list(value.shape),
            "raw_complex_coefficients_dtype": str(value.dtype),
            "raw_modal_coefficient_powers_dtype": str(powers.dtype),
            "complex_modal_effective_rank": {
                "old_formal": formal_rank,
                "same_prefix_replay": replay_rank,
                "absolute_delta": replay_rank - formal_rank,
                "within_mra_tolerance": rank_check["direction"] == "UNCHANGED_WITHIN_TOLERANCE",
            },
            "old_new_raw_hash_equal": True,
            "new_coordinate_power_rank": float(new_metrics[split]["power_effective_rank"]),
        }
    result["passed"] = bool(
        result["checkpoint_digest"]["match"]
        and result["learned_state_digest"]["match"]
        and result["raw_coefficient_same_for_old_and_new"]
        and result["raw_power_same_for_old_and_new"]
        and all(item["complex_modal_effective_rank"]["within_mra_tolerance"] for item in result["panels"].values())
    )
    return result


def _artifact_refs(output_path: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for target in sorted(output_path.iterdir(), key=lambda item: item.name):
        if target.is_file() and target.name != "mra_replay_manifest.json":
            refs[target.name] = {
                "path": target.name,
                "sha256": file_sha256(target),
                "size_bytes": target.stat().st_size,
            }
    return refs


def _assert_output_tree(output_path: Path, *, include_manifest: bool = True) -> None:
    expected = {
        "mra_replay_metrics.json",
        "old_new_mra_comparison.json",
        "compact_report.json",
    }
    if include_manifest:
        expected.add("mra_replay_manifest.json")
    actual = {
        target.relative_to(output_path).as_posix()
        for target in output_path.rglob("*")
        if target.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"RCR-A output tree is not exact: {sorted(actual)}")
    if any(
        ".tmp" in name or Path(name).name.startswith(".") or name.lower().endswith(".tif") or name.lower().endswith(".tiff")
        for name in actual
    ):
        raise RuntimeError("RCR-A output contains forbidden temporary, hidden, or TIFF files")


def _write_output(
    output_path: Path,
    metrics: Mapping[str, Any],
    comparison: Mapping[str, Any],
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite RCR-A output: {_relative_path(output_path)}")
    output_path.mkdir(parents=True, exist_ok=False)
    atomic_write_new_json(output_path / "mra_replay_metrics.json", metrics, artifact_kind="RCR-A metrics")
    atomic_write_new_json(output_path / "old_new_mra_comparison.json", comparison, artifact_kind="RCR-A comparison")
    atomic_write_new_json(output_path / "compact_report.json", report, artifact_kind="RCR-A report")
    _assert_output_tree(output_path, include_manifest=False)
    refs = _artifact_refs(output_path)
    storage = sum(item["size_bytes"] for item in refs.values())
    if storage > 536870912:
        raise RuntimeError("RCR-A output exceeds the diagnostic storage contract")
    final_manifest = dict(manifest)
    final_manifest["artifacts"] = refs
    final_manifest["storage_bytes_excluding_manifest"] = storage
    atomic_write_new_json(output_path / "mra_replay_manifest.json", final_manifest, artifact_kind="RCR-A manifest")
    _assert_output_tree(output_path)
    return final_manifest


def _next_action(gate: str) -> str:
    if gate == MRA_REPAIRED_COORDINATE_VALID:
        return NEXT_ACTION_VALID
    if gate == MRA_PRIOR_CONTRACT_REVIEW_REQUIRED:
        return NEXT_ACTION_SHAPE_REVIEW
    return NEXT_ACTION_INVALID


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_target, config, config_file_hash, config_canonical_hash = load_replay_config(config_path)
    state = mra._git_state()
    if state.get("branch") != BRANCH or state.get("clean") is not True:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_DIRTY_PATH: expected clean branch {BRANCH}; status={state.get('status')}")
    parent = _validate_parent_mra(config)
    data = mra._load_data(_resolve(config["dataset"]["dataset_root"]))
    if data.manifest.get("manifest_hash") != config["dataset"]["mra_manifest_sha256"] or data.manifest.get("confirmation_indices_sha256") != config["dataset"]["confirmation_indices_sha256"]:
        raise ValueError("registered MRA dataset or confirmation index identity is stale")
    panel_labels = _validate_parent_panel(parent["panel"], data, parent["source"])
    r2_identity = _validate_r2(config)
    old_calibration, old_calibration_path = _load_old_calibration_from_source(parent["source"])
    checkpoint, checkpoint_path = _load_checkpoint(config, parent["source"])
    checkpoint_observed_hash = file_sha256(checkpoint_path)
    requested_output = _resolve(output_path)
    actual_output = next_write_once_output(requested_output)
    free_before, total_cuda, disk_free_before = _resource_preflight(config, actual_output)
    device = torch.device(config["execution"]["device"])
    model = mra._load_scaled_model(parent["formal"], checkpoint, device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()) or any(parameter.requires_grad for parameter in model.ols.parameters()):
        raise RuntimeError("MRA repaired-coordinate model is not frozen")
    if bogt.mode_bank_sha256(model) != config["r2_calibration"]["mode_bank_sha256"]:
        raise ValueError("loaded model mode bank does not match repaired calibration")
    state_before = state_digest(model)
    torch.cuda.reset_peak_memory_stats(device)
    selection_coefficients = _collect_coefficients(
        model,
        data.test_images,
        data.selection_indices,
        device=device,
        batch_size=config["execution"]["batch_size"],
    )
    confirmation_coefficients = _collect_coefficients(
        model,
        data.images,
        data.confirmation_indices,
        device=device,
        batch_size=config["execution"]["batch_size"],
    )
    coefficients = {
        "selection": selection_coefficients,
        "confirmation": confirmation_coefficients,
    }
    old_metrics = {
        split: coordinate_metrics(value, panel_labels[f"{split}_labels"], old_calibration)
        for split, value in coefficients.items()
    }
    new_metrics = {
        split: coordinate_metrics(value, panel_labels[f"{split}_labels"], r2_identity["calibration"])
        for split, value in coefficients.items()
    }
    state_after = state_digest(model)
    torch.cuda.synchronize(device)
    peak_cuda = int(torch.cuda.max_memory_allocated(device))
    elapsed = time.perf_counter() - started
    formal_metrics = parent["formal_metrics"]
    old_replay_consistency = _old_replay_consistency(
        formal_metrics,
        old_metrics,
        rtol=config["gates"]["old_replay_rtol"],
        atol=config["gates"]["old_replay_atol"],
    )
    invariant_controls = _invariant_controls(
        coefficients,
        formal_metrics,
        old_metrics,
        new_metrics,
        checkpoint_expected=config["checkpoint"]["sha256"],
        checkpoint_observed=checkpoint_observed_hash,
        state_before=state_before,
        state_after=state_after,
        rtol=config["gates"]["invariant_rank_rtol"],
        atol=config["gates"]["invariant_rank_atol"],
    )
    prior_shape = assess_prior_shape(
        new_metrics,
        mean_tolerance=config["gates"]["prior_shape_review_trigger"]["mean_alignment_absolute_tolerance"],
        variance_relative_tolerance=config["gates"]["prior_shape_review_trigger"]["variance_shape_relative_tolerance"],
        dominance_ratio=config["gates"]["prior_shape_review_trigger"]["variance_dominance_ratio"],
        minimum_numerical_kl=config["gates"]["prior_shape_review_trigger"]["minimum_numerical_kl"],
    )
    checks = {
        "calibration_identity": r2_identity["calibration"].schema == NEW_CALIBRATION_SCHEMA and r2_identity["calibration"].contract_version == NEW_CALIBRATION_CONTRACT,
        "checkpoint_identity": checkpoint_observed_hash == config["checkpoint"]["sha256"],
        "latent_anchor": all(item["finite"] and item["nonnegative"] and item["latent_dtype"] == "torch.float32" for item in new_metrics.values()),
        "finite": all(item["finite"] and item["nonnegative"] for item in new_metrics.values()),
        "invariant_controls": invariant_controls["passed"] and old_replay_consistency["all_consistent"],
        "learned_state_freeze": state_before == state_after and not model.training and not any(parameter.requires_grad for parameter in model.parameters()),
        "mra_metrics_replayed": all(set(COORDINATE_METRIC_NAMES).issubset(item) for item in new_metrics.values()),
    }
    gate = classify_stage_a_gate(checks, prior_shape)
    next_action = _next_action(gate)
    state_freeze = {
        "state_digest_before": state_before,
        "state_digest_after": state_after,
        "state_unchanged": state_before == state_after,
        "optimizer_created": False,
        "backward_called": False,
        "refit": False,
        "checkpoint_created": False,
        "decoder_entered": False,
        "relay_after_pre_ols": False,
        "cmos_called": False,
        "mpcdar_called": False,
        "prior_generation": False,
        "training": False,
        "evaluator_calls": 0,
    }
    comparison = {
        "schema": COMPARISON_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "old_reference": "parent MRA attempt_05 formal artifact; exact values read from audit.json",
        "coordinate_metrics": {
            split: _metric_comparison(
                formal_metrics[split],
                new_metrics[split],
                rtol=config["gates"]["old_replay_rtol"],
                atol=config["gates"]["old_replay_atol"],
            )
            for split in ("selection", "confirmation")
        },
        "complex_rank_invariant": {
            split: invariant_controls["panels"][split]["complex_modal_effective_rank"]
            for split in ("selection", "confirmation")
        },
        "old_prefix_replay_consistency": old_replay_consistency,
        "prior_shape_assessment": prior_shape,
        "no_new_material_improvement_gate": True,
        "gate": gate,
    }
    metrics = {
        "schema": METRICS_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "gate": gate,
        "calibration": {
            "old": {
                "schema": OLD_CALIBRATION_SCHEMA,
                "path": str(old_calibration_path),
                "sha256": parent["source"].hashes["scaled_chen_calibration"],
                "semantic": "epsilon-dominated_calibration_coordinate",
                "port_scale": _range_summary(old_calibration.port_scale),
            },
            "new": {
                "schema": NEW_CALIBRATION_SCHEMA,
                "contract_version": NEW_CALIBRATION_CONTRACT,
                "path": str(r2_identity["payload_path"]),
                "sha256": r2_identity["payload_sha256"],
                "manifest_sha256": r2_identity["manifest_sha256"],
                "sample_count": r2_identity["calibration"].sample_count,
                "port_count": PORT_COUNT,
                "calibration_indices_sha256": config["r2_calibration"]["calibration_indices_sha256"],
                "dataset_manifest_sha256": config["r2_calibration"]["dataset_manifest_sha256"],
                "mode_bank_sha256": config["r2_calibration"]["mode_bank_sha256"],
                "zero_phase_reference": config["r2_calibration"]["zero_phase_reference"],
                "plane_identity": config["r2_calibration"]["plane_identity"],
                "raw_port_mean_equals_port_scale": True,
                "port_scale": _range_summary(r2_identity["calibration"].port_scale),
            },
            "lineage_validation": True,
            "r2_repair_generation_clean": r2_identity["repair_generation_clean"],
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "schema": CHECKPOINT_SCHEMA,
            "global_step": 5000,
            "expected_sha256": config["checkpoint"]["sha256"],
            "observed_sha256": checkpoint_observed_hash,
            "identity_pass": checks["checkpoint_identity"],
        },
        "dataset": {
            "mra_manifest_sha256": data.manifest["manifest_hash"],
            "selection_count": SELECTION_COUNT,
            "selection_source": config["dataset"]["selection_source"],
            "selection_seed": SELECTION_SEED,
            "selection_indices_sha256": data.manifest["selection_indices_sha256"],
            "confirmation_count": CONFIRMATION_COUNT,
            "confirmation_source": config["dataset"]["confirmation_source"],
            "confirmation_indices_sha256": data.manifest["confirmation_indices_sha256"],
            "panel_reused_without_resampling": True,
        },
        "panels": {
            "selection": {
                "labels": "parent_panel.selection_labels",
                "old_formal": formal_metrics["selection"],
                "old_prefix_replay": old_metrics["selection"],
                "repaired_coordinate": new_metrics["selection"],
            },
            "confirmation": {
                "labels": "parent_panel.confirmation_labels",
                "old_formal": formal_metrics["confirmation"],
                "old_prefix_replay": old_metrics["confirmation"],
                "repaired_coordinate": new_metrics["confirmation"],
            },
        },
        "repaired_latent_anchor": {
            "selection_first_batch": repaired_latent_anchor(
                bogt.normalize_port_power(
                    selection_coefficients[: config["execution"]["batch_size"]].abs().square(),
                    r2_identity["calibration"].port_scale,
                ).cpu()
            ),
            "confirmation_first_batch": repaired_latent_anchor(
                bogt.normalize_port_power(
                    confirmation_coefficients[: config["execution"]["batch_size"]].abs().square(),
                    r2_identity["calibration"].port_scale,
                ).cpu()
            ),
            "full_selection": repaired_latent_anchor(
                bogt.normalize_port_power(selection_coefficients.abs().square(), r2_identity["calibration"].port_scale).cpu()
            ),
            "full_confirmation": repaired_latent_anchor(
                bogt.normalize_port_power(confirmation_coefficients.abs().square(), r2_identity["calibration"].port_scale).cpu()
            ),
        },
        "invariant_controls": invariant_controls,
        "old_prefix_replay_consistency": old_replay_consistency,
        "checks": checks,
        "prior_mean_assessment": {
            "registered_mean": MRA_PRIOR_MEAN,
            "selection_z_mean": new_metrics["selection"]["z_mean"],
            "confirmation_z_mean": new_metrics["confirmation"]["z_mean"],
            "interpretation": "descriptive_coordinate_assessment_only",
        },
        "prior_variance_shape_assessment": prior_shape,
        "reconstruction_boundary": {
            "reconstruction_supported": False,
            "context": "HISTORICAL_CONTEXT_ONLY",
            "decoder_entered": False,
            "relay_after_pre_ols": False,
            "cmos_called": False,
            "mpcdar_called": False,
        },
        "learned_state_freeze": state_freeze,
        "resource": {
            "device": str(device),
            "free_cuda_bytes_before": free_before,
            "total_cuda_bytes": total_cuda,
            "disk_free_bytes_before": disk_free_before,
            "peak_cuda_allocated_bytes": peak_cuda,
            "peak_memory_gate_pass": peak_cuda <= config["execution"]["max_peak_cuda_bytes"],
            "wall_clock_seconds": elapsed,
            "wall_clock": "observational_only",
        },
        "governance": dict(config["governance"]),
        "stage_b_g1_replay_executed": False,
        "next_action": next_action,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "gate": gate,
        "selection": {
            "old_to_new": comparison["coordinate_metrics"]["selection"],
            "complex_rank": invariant_controls["panels"]["selection"]["complex_modal_effective_rank"],
            "old_power_rank": formal_metrics["selection"]["power_effective_rank"],
            "new_power_rank": new_metrics["selection"]["power_effective_rank"],
        },
        "confirmation": {
            "old_to_new": comparison["coordinate_metrics"]["confirmation"],
            "complex_rank": invariant_controls["panels"]["confirmation"]["complex_modal_effective_rank"],
            "old_power_rank": formal_metrics["confirmation"]["power_effective_rank"],
            "new_power_rank": new_metrics["confirmation"]["power_effective_rank"],
        },
        "prior_mean_assessment": metrics["prior_mean_assessment"],
        "prior_variance_shape_assessment": prior_shape,
        "interpretation": "repaired-coordinate MRA evidence is valid only at the latent/prior metric boundary; no decoder or generation authority is established",
        "governance": dict(config["governance"]),
        "next_action": next_action,
    }
    manifest = {
        "schema": REPLAY_MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "attempt": actual_output.name,
        "requested_output": str(requested_output),
        "output_directory": str(actual_output),
        "git": state,
        "config": {
            "path": _relative_path(config_target),
            "file_sha256": config_file_hash,
            "canonical_sha256": config_canonical_hash,
        },
        "parent_mra": {
            "attempt": Path(parent["attempt"]).name,
            "config": _relative_path(parent["spec"].path),
            "config_sha256": config["parent_mra"]["config_sha256"],
            "config_canonical_sha256": config["parent_mra"]["config_canonical_sha256"],
            "review_contract": _relative_path(parent["review_path"]),
            "review_contract_sha256": config["parent_mra"]["review_contract_sha256"],
            "audit": _relative_path(_resolve(config["parent_mra"]["audit"])),
            "audit_sha256": config["parent_mra"]["audit_sha256"],
            "compact_report": _relative_path(_resolve(config["parent_mra"]["compact_report"])),
            "compact_report_sha256": config["parent_mra"]["compact_report_sha256"],
            "panel": _relative_path(parent["panel_path"]),
            "panel_sha256": config["parent_mra"]["panel_sha256"],
            "status": parent["audit"]["status"],
            "prior_status": parent["audit"]["contract"]["prior_status"],
        },
        "r2_calibration": {
            "config": _relative_path(r2_identity["config_path"]),
            "config_sha256": r2_identity["config_file_sha256"],
            "config_canonical_sha256": r2_identity["config_canonical_sha256"],
            "manifest": _relative_path(r2_identity["manifest_path"]),
            "manifest_sha256": r2_identity["manifest_sha256"],
            "payload": _relative_path(r2_identity["payload_path"]),
            "payload_sha256": r2_identity["payload_sha256"],
            "schema": NEW_CALIBRATION_SCHEMA,
            "contract_version": NEW_CALIBRATION_CONTRACT,
            "repair_generation_clean": r2_identity["repair_generation_clean"],
        },
        "checkpoint": metrics["checkpoint"],
        "dataset": metrics["dataset"],
        "measurement": config["measurement"],
        "learned_state_freeze": state_freeze,
        "no_update_audit": state_freeze,
        "gate": gate,
        "prior_shape_assessment": prior_shape,
        "reconstruction_boundary": metrics["reconstruction_boundary"],
        "governance": dict(config["governance"]),
        "stage_b_g1_replay_executed": False,
        "next_action": next_action,
    }
    final_manifest = _write_output(actual_output, metrics, comparison, report, manifest)
    del model, selection_coefficients, confirmation_coefficients
    return {
        "output_directory": str(actual_output),
        "metrics": metrics,
        "comparison": comparison,
        "report": report,
        "manifest": final_manifest,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args.config, args.output)
    print(json.dumps(result["report"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
