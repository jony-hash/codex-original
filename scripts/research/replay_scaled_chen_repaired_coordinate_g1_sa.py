"""Replay G1 prior-support accessibility under the repaired coordinate.

This Stage-B observer keeps the registered step-5000 checkpoint, dataset
identities, prior sampler, support-distance metric, and bootstrap procedure
fixed.  Only the calibration coordinate changes from the historical
epsilon-dominated v1 scale to the registered unit-aware v2 scale.  The replay
stops at latent support distances and never enters decoder, detector, or
training code.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.application.research import optical_manifold_smf_bogt as bogt
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import measure_scaled_chen_g1_prior_support_accessibility as old_g1
from scripts.research import replay_scaled_chen_repaired_coordinate_mra as rcr


TASK_ID = "H-LOMA-GEN-RCR-B"
HYPOTHESIS = TASK_ID
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_g1_sa_v1"
OBSERVER_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_g1_sa_manifest.v1"
METRICS_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_g1_sa_metrics.v1"
DISTANCES_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_g1_sa_distances.v1"
COMPARISON_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_g1_sa_old_new_comparison.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_g1_sa_compact_report.v1"
ALIGNMENT_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_g1_sa_figure_alignment.v1"
CHECKPOINT_SCHEMA = "scaled_chen_bogt_horizon_checkpoint.v1"
OLD_CALIBRATION_SCHEMA = "scaled_chen_bogt_calibration.v1"
NEW_CALIBRATION_SCHEMA = "scaled_chen_bogt_calibration.v2"
NEW_CALIBRATION_CONTRACT = "unit_aware_raw_mean_v2"
OLD_G1_TASK_ID = "H-LOMA-GEN-G1-SA"
OLD_G1_OBSERVER_SCHEMA = "lightgen_scaled_chen_g1_prior_support_accessibility_observer_manifest.v1"
OLD_G1_METRICS_SCHEMA = "lightgen_scaled_chen_g1_prior_support_accessibility_metrics.v1"
OLD_G1_DISTANCES_SCHEMA = "lightgen_scaled_chen_g1_prior_support_accessibility_distances.v1"
OLD_G1_REPORT_SCHEMA = "lightgen_scaled_chen_g1_prior_support_accessibility_compact_report.v1"
PCR_PSD_TASK_ID = "H-LOMA-GEN-PCR-PSD"
PCR_PSD_CLASSIFICATION = "PCR1_PRIOR_CONTRACT_INTERNALLY_COHERENT"
RCR_A_TASK_ID = "H-LOMA-GEN-RCR-A"
RCR_A_GATE = "MRA_PRIOR_CONTRACT_REVIEW_REQUIRED"
RCR_A_MANIFEST_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_mra_manifest.v1"
RCR_A_METRICS_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_mra_metrics.v1"
RCR_A_REPORT_SCHEMA = "lightgen_scaled_chen_repaired_coordinate_mra_compact_report.v1"
LATENT_DIM = 100
PORT_COUNT = 100
TRAIN_SUPPORT_COUNT = 55_000
HELDOUT_COUNT = 5_000
PRIOR_SEED = 1237
BOOTSTRAP_SEED = 1236
BOOTSTRAP_REPLICATES = 2000
CONFIDENCE_LEVEL = 0.95
QUERY_CHUNK_SIZE = 128
SUPPORT_CHUNK_SIZE = 4096
BATCH_SIZE = 10
MAX_STORAGE_BYTES = 512 * 1024**2
MEAN_SCALE_TOLERANCE = 0.10
MATERIAL_REDUCTION_FACTOR_MIN = 100.0
SAME_ORDER_REDUCTION_FACTOR_MAX = 10.0
P4_SUPPORTED_REPAIRED = "P4_SUPPORTED_REPAIRED"
P4_NOT_SUPPORTED_REPAIRED = "P4_NOT_SUPPORTED_REPAIRED"
P4_UNDERDETERMINED_REPAIRED = "P4_UNDERDETERMINED_REPAIRED"
RC1_CALIBRATION_WAS_DOMINANT_DRIVER = "RC1_CALIBRATION_WAS_DOMINANT_DRIVER"
RC2_CALIBRATION_WAS_PARTIAL_DRIVER_P4_REMAINS = (
    "RC2_CALIBRATION_WAS_PARTIAL_DRIVER_P4_REMAINS"
)
RC4_CALIBRATION_REPAIR_DOES_NOT_RESOLVE_MISMATCH = (
    "RC4_CALIBRATION_REPAIR_DOES_NOT_RESOLVE_MISMATCH"
)
RC5_UNDERDETERMINED = "RC5_UNDERDETERMINED"
NEXT_ACTION_RC1 = "PLAN_G2_PRIOR_DECODER_ACCESSIBILITY"
NEXT_ACTION_RC2_RC4 = "PLAN_GENERATIVE_SUPPORT_ALIGNMENT_TRAINING"
NEXT_ACTION_RC5 = "REVIEW_REPAIRED_COORDINATE_EVIDENCE"
DEFAULT_CONFIG_PATH = Path(
    "configs/research/lightgen_scaled_chen_repaired_coordinate_g1_sa_v1.yaml"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_g1_prior_support_accessibility_repaired_coordinate_v1_attempt_01"
)
EXPECTED_MRA_CONTEXT = {
    "selection": {
        "z_mean": 1.0060573,
        "z_variance": 1.5498618,
        "gaussian_kl": 3.28331,
        "numerical_truncated_normal_kl": 3.24926,
        "power_effective_rank": 7.21653,
        "complex_effective_rank": 12.23617,
    },
    "confirmation": {
        "z_mean": 0.9250944,
        "z_variance": 1.4546335,
        "gaussian_kl": 2.84983,
        "numerical_truncated_normal_kl": 2.74284,
        "power_effective_rank": 7.36835,
        "complex_effective_rank": 12.87396,
    },
}
GOVERNANCE = {
    "mainline_state": "closed_at_current_evidence_boundary",
    "reconstruction_manifold_supported": False,
    "generative_manifold_supported": False,
    "generation_readiness": "not_ready",
    "training_authorized": False,
    "formal_promotion": False,
}
EXPECTED_DIRTY_PATHS = {
    "configs/research/lightgen_scaled_chen_repaired_coordinate_g1_sa_v1.yaml",
    "scripts/research/replay_scaled_chen_repaired_coordinate_g1_sa.py",
    "tests/v2/test_scaled_chen_repaired_coordinate_g1_sa.py",
    "docs/CURRENT_STATUS.md",
    "docs/PROJECT_MEMORY.md",
    "docs/status/current_handoff.md",
}
EXPECTED_OUTPUT_FILES = {
    "observer_manifest.json",
    "support_metrics.json",
    "support_distances.pt",
    "old_new_support_comparison.json",
    "compact_report.json",
    "support_distance_distribution.png",
    "support_distance_distribution.pdf",
    "support_distance_distribution.svg",
    "support_distance_distribution.alignment.json",
    "support_distance_summary.png",
    "support_distance_summary.pdf",
    "support_distance_summary.svg",
    "support_distance_summary.alignment.json",
}


def _resolve(value: str | Path, *, base_path: Path | None = None) -> Path:
    return base._resolve_path(value, base=base_path)


def _relative_path(path: str | Path) -> str:
    target = Path(path).resolve()
    try:
        return target.relative_to(base._repo_root()).as_posix()
    except ValueError:
        return str(target)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _json_file(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    actual = file_sha256(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(f"JSON artifact hash mismatch: {_relative_path(path)}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not an object: {_relative_path(path)}")
    return payload


def _load_config(path: str | Path) -> tuple[Path, dict[str, Any], str, str]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RCR-B config must be a YAML mapping")
    required = {
        "schema",
        "task_id",
        "hypothesis",
        "rcr_a",
        "pcr_psd",
        "old_g1",
        "r2_calibration",
        "checkpoint",
        "dataset",
        "prior_sampling",
        "distance",
        "bootstrap",
        "gate",
        "causal_classification",
        "measurement",
        "execution",
        "authorization",
        "governance",
    }
    if set(payload) != required:
        raise ValueError("RCR-B config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("task_id") != TASK_ID or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("RCR-B config identity is invalid")
    _assert_config(payload)
    return target, payload, file_sha256(target), canonical_sha256(payload)


def _assert_config(payload: Mapping[str, Any]) -> None:
    rcr_binding = _require_mapping(payload["rcr_a"], "rcr_a")
    if dict(rcr_binding) != {
        "config": "configs/research/lightgen_scaled_chen_repaired_coordinate_mra_v1.yaml",
        "config_sha256": "c1bc2d125c753318bb97b9c95f5ff73cf8845b9efd81962c0eebbc4f33387b0f",
        "config_canonical_sha256": "c79f1ec8645289ece2df9e85fe17792a97ae5a52c164a826271e80997043011a",
        "output_directory": "outputs/20260904_v2/runs/lightgen_scaled_chen_repaired_coordinate_mra_v1_attempt_02",
        "manifest": "mra_replay_manifest.json",
        "manifest_sha256": "10a5469468357c8490446ff4c9d1fc86bc1041bba4cfa6fa51af4fc8f01f70da",
        "metrics": "mra_replay_metrics.json",
        "metrics_sha256": "5d2cc5c0c7b6e80f80d4f07239bb737f87327860381c0bf151cbaa686001b612",
        "compact_report": "compact_report.json",
        "compact_report_sha256": "92fa909b36b4eb87f47c3fb4ad8ae5874ab27e67f545e1a601569df5546ea308",
        "gate": RCR_A_GATE,
    }:
        raise ValueError("RCR-A source binding is not locked")

    psd_binding = _require_mapping(payload["pcr_psd"], "pcr_psd")
    if dict(psd_binding) != {
        "output_directory": "outputs/20260904_v2/runs/lightgen_scaled_chen_prior_shape_decomposition_v1_attempt_01",
        "manifest": "prior_shape_manifest.json",
        "manifest_sha256": "228679fda073b2e4e1e46c931b743b490930dc895313e5c201283d244e22d97a",
        "metrics": "prior_shape_metrics.json",
        "metrics_sha256": "4ea665c5a021a09c9ce2d44adfc31ed0cdb593de30b0ed3dbb3b10986bfbe9d2",
        "compact_report": "compact_report.json",
        "compact_report_sha256": "cbd0ceda6a878bcf295ef0252936edf35a6525937b76472610f2c9e384d85f94",
        "classification": PCR_PSD_CLASSIFICATION,
        "task_id": PCR_PSD_TASK_ID,
    }:
        raise ValueError("PCR-PSD source binding is not locked")

    old_binding = _require_mapping(payload["old_g1"], "old_g1")
    if old_binding.get("config") != "configs/research/lightgen_optical_manifold_smf_bogt_prior_support_accessibility_v1.yaml" or old_binding.get("config_sha256") != "b387d5935d8b2f54fa08d99f8e196c03c29ecf411595dba19113633781f0692f" or old_binding.get("config_canonical_sha256") != "fce04103f089dc0f1a18c3266a6919f69ecfe05ede774dc4223abe072de6d473":
        raise ValueError("old G1 config binding is not locked")
    expected_old = {
        "raw_attempt": {
            "directory": "outputs/20260903_v2/runs/lightgen_scaled_chen_g1_prior_support_accessibility_v1_attempt_01",
            "manifest_sha256": "1e47c538deb3c67965eb646503a5029e963a06ee7d4ebcb7e3f282444d9a7d6e",
            "metrics_sha256": "2cbc64f1ab587caf3dc663084e7c9d23082ca9a2080a652b59d6edf7bc3cf826",
            "distances_sha256": "05002920d1b73ab8dd1243ce93f44659ba95ad5d4eb35e3f4e737c0f9d8b2c1b",
            "compact_report_sha256": "2290c103864aac1b27075924db1b757b601cf9c016a80099b58b38fda58f9422",
        },
        "descriptive_attempt": {
            "directory": "outputs/20260903_v2/runs/lightgen_scaled_chen_g1_prior_support_accessibility_v1_attempt_04",
            "manifest_sha256": "257e6b786714b6f0a76fe93e4bba6ddf2f6bc3b691d0ef3f582ebd7c7521f239",
            "metrics_sha256": "2cbc64f1ab587caf3dc663084e7c9d23082ca9a2080a652b59d6edf7bc3cf826",
            "distances_sha256": "05002920d1b73ab8dd1243ce93f44659ba95ad5d4eb35e3f4e737c0f9d8b2c1b",
            "compact_report_sha256": "2290c103864aac1b27075924db1b757b601cf9c016a80099b58b38fda58f9422",
        },
    }
    for role, expected in expected_old.items():
        actual = _require_mapping(old_binding.get(role), f"old_g1.{role}")
        if dict(actual) != expected:
            raise ValueError(f"old G1 {role} authority binding is not locked")

    if dict(payload["r2_calibration"]) != {
        "config": "configs/research/lightgen_scaled_chen_bogt_calibration_contract_repair_v2.json",
        "config_sha256": "b8c1efab6fd49c0e68696bc7a466cf6eaeeb499a842c6efa66fe080254b867a9",
        "config_canonical_sha256": "c9de9505a6cf64432e4f2c350bfc5dfa188bc9563cdf8f89c274193123045deb",
        "manifest": "outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/calibration_manifest.json",
        "manifest_sha256": "0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6",
        "payload": "outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/repaired_calibration.pt",
        "payload_sha256": "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07",
        "schema": NEW_CALIBRATION_SCHEMA,
        "contract_version": NEW_CALIBRATION_CONTRACT,
        "sample_count": 1000,
        "port_count": PORT_COUNT,
        "calibration_indices_sha256": "369c5bded5673f901ae5236eaa51a0c8da17934dc2fab139a205c9fce41d5543",
        "dataset_manifest_sha256": "741e6402da754a740a677dbb343d48c35983f62b59911bbc129dfcc44f9e9ae9",
        "mode_bank_sha256": "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9",
        "zero_phase_reference": "independent_train_only_zero_phase",
        "plane_identity": "pre_ols_plane_power_reference_v1",
    }:
        raise ValueError("R2 repaired calibration binding is not locked")

    if dict(payload["checkpoint"]) != {
        "path": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "sha256": "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de",
        "schema": CHECKPOINT_SCHEMA,
        "global_step": 5000,
    }:
        raise ValueError("step-5000 checkpoint binding is not locked")

    if dict(payload["dataset"]) != {
        "dataset_root": "data",
        "mra_manifest_sha256": "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb",
        "train_support_source": "registered_train_split_0_55000_full",
        "train_support_start": 0,
        "train_support_count": TRAIN_SUPPORT_COUNT,
        "heldout_source": "horizon_heldout_train_indices_55000_59999",
        "heldout_count": HELDOUT_COUNT,
        "heldout_indices_sha256": "ff7ab50b95d96e727c240edc2baa916d1e6c11402b8f4f9c2ce6e2e7e5e2218a",
        "input_shape": [1, 28, 28],
        "rendered_shape": [1, 200, 200],
    }:
        raise ValueError("G1 support/query dataset binding is not locked")

    if dict(payload["prior_sampling"]) != {
        "seed": PRIOR_SEED,
        "kind": "truncated_normal",
        "mean": 1.0,
        "variance": 0.25,
        "lower": 0.0,
        "unconditional": True,
        "class_conditioned": False,
        "count": HELDOUT_COUNT,
        "sampler": "bogt.truncated_normal_sample_inverse_cdf",
    }:
        raise ValueError("registered prior sampling contract is not locked")
    if dict(payload["distance"]) != {
        "metric": "exact_normalized_euclidean",
        "normalization": "divide_by_sqrt_100",
        "query_chunk_size": QUERY_CHUNK_SIZE,
        "support_chunk_size": SUPPORT_CHUNK_SIZE,
        "approximate_search": False,
    }:
        raise ValueError("exact support-distance contract is not locked")
    if dict(payload["bootstrap"]) != {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "confidence_level": CONFIDENCE_LEVEL,
        "resampling_unit": "one_dimensional_distance_vectors",
    }:
        raise ValueError("bootstrap contract is not locked")
    if dict(payload["gate"]) != {
        "name": "P4",
        "supported": "lower_ci_r50_gt_1_and_lower_ci_r95_gt_1",
        "not_supported": "upper_ci_r50_le_1_and_upper_ci_r95_le_1",
        "otherwise": P4_UNDERDETERMINED_REPAIRED,
        "prior_exceedance_h95": "descriptive_only",
    }:
        raise ValueError("repaired P4 gate contract is not locked")
    if dict(payload["causal_classification"]) != {
        "mean_scale_tolerance": MEAN_SCALE_TOLERANCE,
        "material_reduction_factor_min": MATERIAL_REDUCTION_FACTOR_MIN,
        "same_order_reduction_factor_max": SAME_ORDER_REDUCTION_FACTOR_MAX,
        "rc1": RC1_CALIBRATION_WAS_DOMINANT_DRIVER,
        "rc2": RC2_CALIBRATION_WAS_PARTIAL_DRIVER_P4_REMAINS,
        "rc4": RC4_CALIBRATION_REPAIR_DOES_NOT_RESOLVE_MISMATCH,
        "rc5": RC5_UNDERDETERMINED,
    }:
        raise ValueError("causal classification thresholds are not locked")
    measurement = _require_mapping(payload["measurement"], "measurement")
    if dict(measurement) != {
        "objective": "frozen_latent_prefix_only",
        "prefix": "slm_to_front_to_encoder_to_pre_ols_to_complex_coefficients",
        "coordinate": "repaired_calibration_unit_aware_raw_mean_v2",
        "support_distance_scope": "prior_to_encoded_train_support_vs_heldout_to_encoded_train_support",
        "old_authority": "old_g1_attempt_04_descriptive_authority",
        "no_update": True,
        "decoder_entered": False,
        "relay_after_pre_ols": False,
        "cmos_called": False,
        "mpcdar_called": False,
    }:
        raise ValueError("RCR-B measurement boundary is not locked")
    execution = _require_mapping(payload["execution"], "execution")
    if dict(execution) != {
        "device": "cuda",
        "batch_size": BATCH_SIZE,
        "min_free_cuda_bytes": 4294967296,
        "max_peak_cuda_bytes": 5368709120,
        "min_disk_free_bytes": 10737418240,
        "max_storage_bytes": MAX_STORAGE_BYTES,
        "require_cublas_workspace_config": ":4096:8",
    }:
        raise ValueError("RCR-B execution contract is not locked")
    authorization = _require_mapping(payload["authorization"], "authorization")
    if dict(authorization) != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal_promotion": False,
        "stage_b_g1_replay": True,
        "prior_generation": False,
        "optimizer_created": False,
        "backward_called": False,
        "refit": False,
        "checkpoint_created": False,
        "decoder_entered": False,
        "relay_after_pre_ols": False,
        "cmos_called": False,
        "mpcdar_called": False,
    }:
        raise ValueError("RCR-B authorization boundary is not locked")
    if dict(payload["governance"]) != GOVERNANCE:
        raise ValueError("RCR-B governance boundary is not locked")


def validate_repaired_calibration_identity(
    payload: Mapping[str, Any],
    *,
    expected_schema: str,
    expected_contract: str,
    expected_calibration_indices_sha256: str,
    expected_dataset_manifest_hash: str,
    expected_mode_bank_sha256: str,
) -> None:
    """Expose the fail-closed R2-to-RCR calibration boundary to tests."""
    rcr.validate_repaired_calibration_identity(
        payload,
        expected_schema=expected_schema,
        expected_contract=expected_contract,
        expected_calibration_indices_sha256=expected_calibration_indices_sha256,
        expected_dataset_manifest_hash=expected_dataset_manifest_hash,
        expected_mode_bank_sha256=expected_mode_bank_sha256,
    )


validate_latent_bank = rcr.validate_latent_bank
state_digest = rcr.state_digest


def sample_registered_prior(
    count: int,
    *,
    seed: int = PRIOR_SEED,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if count <= 0:
        raise ValueError("RCR-B prior query count must be positive")
    return old_g1.sample_registered_prior(count, seed=seed, device=device, dtype=dtype)


def exact_normalized_nearest_distances(
    query: torch.Tensor,
    support: torch.Tensor,
    *,
    query_chunk_size: int = QUERY_CHUNK_SIZE,
    support_chunk_size: int = SUPPORT_CHUNK_SIZE,
) -> torch.Tensor:
    return old_g1.exact_normalized_nearest_distances(
        query,
        support,
        query_chunk_size=query_chunk_size,
        support_chunk_size=support_chunk_size,
    )


def bootstrap_ratio_ci(
    holdout_distances: torch.Tensor,
    prior_distances: torch.Tensor,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    confidence_level: float = CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    return old_g1.bootstrap_ratio_ci(
        holdout_distances,
        prior_distances,
        replicates=replicates,
        seed=seed,
        confidence_level=confidence_level,
    )


def classify_repaired_p4_gate(
    r50_ci: Sequence[float], r95_ci: Sequence[float]
) -> str:
    if len(r50_ci) != 2 or len(r95_ci) != 2:
        raise ValueError("repaired P4 confidence intervals must be [lower, upper]")
    r50_lower, r50_upper = float(r50_ci[0]), float(r50_ci[1])
    r95_lower, r95_upper = float(r95_ci[0]), float(r95_ci[1])
    if r50_lower > 1.0 and r95_lower > 1.0:
        return P4_SUPPORTED_REPAIRED
    if r50_upper <= 1.0 and r95_upper <= 1.0:
        return P4_NOT_SUPPORTED_REPAIRED
    return P4_UNDERDETERMINED_REPAIRED


def next_write_once_output(base_path: str | Path) -> Path:
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


def _quantile(value: torch.Tensor, probability: float) -> torch.Tensor:
    return torch.quantile(value, probability, interpolation="linear")


def _summary_stats(value: torch.Tensor) -> dict[str, float]:
    view = value.detach().to(torch.float64).cpu().reshape(-1)
    if view.numel() == 0 or not torch.isfinite(view).all() or (view <= 0).any():
        raise ValueError("support distances must be finite and positive")
    return {
        "count": int(view.numel()),
        "mean": float(view.mean()),
        "median": float(_quantile(view, 0.5)),
        "p95": float(_quantile(view, 0.95)),
        "minimum": float(view.min()),
        "maximum": float(view.max()),
    }


def _support_value(metrics: Mapping[str, Any], group: str, statistic: str) -> float:
    value = _require_mapping(metrics.get(group), group).get(statistic)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{group}.{statistic} must be finite")
    return float(value)


def _ratio_value(metrics: Mapping[str, Any], name: str) -> float:
    ratios = _require_mapping(metrics.get("ratios"), "ratios")
    value = ratios.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"ratios.{name} must be finite and positive")
    return float(value)


def compare_old_and_repaired_support(
    old_metrics: Mapping[str, Any], repaired_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    names = ("H50", "H95", "P50", "P95", "R50", "R95")
    old_values = {
        "H50": _support_value(old_metrics, "holdout", "median"),
        "H95": _support_value(old_metrics, "holdout", "p95"),
        "P50": _support_value(old_metrics, "prior", "median"),
        "P95": _support_value(old_metrics, "prior", "p95"),
        "R50": _ratio_value(old_metrics, "r50"),
        "R95": _ratio_value(old_metrics, "r95"),
    }
    repaired_values = {
        "H50": _support_value(repaired_metrics, "holdout", "median"),
        "H95": _support_value(repaired_metrics, "holdout", "p95"),
        "P50": _support_value(repaired_metrics, "prior", "median"),
        "P95": _support_value(repaired_metrics, "prior", "p95"),
        "R50": _ratio_value(repaired_metrics, "r50"),
        "R95": _ratio_value(repaired_metrics, "r95"),
    }
    mapping = {
        name: {"old": old_values[name], "repaired": repaired_values[name]}
        for name in names
    }
    reduction = {
        name: old_values[name] / repaired_values[name]
        for name in ("R50", "R95")
    }
    return {
        "schema": COMPARISON_SCHEMA,
        "task_id": TASK_ID,
        "old_to_repaired": mapping,
        "reduction_factor": reduction,
        "reduction_factor_interpretation": "causal_description_only_not_a_gate",
    }


def classify_final_causal_result(
    repaired_p4_gate: str,
    *,
    old_ratios: Mapping[str, float],
    repaired_ratios: Mapping[str, float],
    selection_mean: float,
    confirmation_mean: float,
    prior_contract_coherent: bool,
    mean_scale_tolerance: float = MEAN_SCALE_TOLERANCE,
    material_reduction_factor_min: float = MATERIAL_REDUCTION_FACTOR_MIN,
    same_order_reduction_factor_max: float = SAME_ORDER_REDUCTION_FACTOR_MAX,
) -> dict[str, Any]:
    if repaired_p4_gate not in {
        P4_SUPPORTED_REPAIRED,
        P4_NOT_SUPPORTED_REPAIRED,
        P4_UNDERDETERMINED_REPAIRED,
    }:
        raise ValueError("unknown repaired P4 gate")
    means_normal = (
        abs(float(selection_mean) - 1.0) <= mean_scale_tolerance
        and abs(float(confirmation_mean) - 1.0) <= mean_scale_tolerance
    )
    factors = {
        name: float(old_ratios[name]) / float(repaired_ratios[name])
        for name in ("r50", "r95")
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in factors.values()):
        raise ValueError("causal reduction factors must be finite and positive")

    if (
        repaired_p4_gate == P4_NOT_SUPPORTED_REPAIRED
        and means_normal
        and prior_contract_coherent
    ):
        classification = RC1_CALIBRATION_WAS_DOMINANT_DRIVER
        next_action = NEXT_ACTION_RC1
        interpretation = (
            "epsilon-dominated calibration was the dominant cause of the old "
            "empirical support-accessibility mismatch; repaired P4 is not "
            "supported under the registered metric."
        )
    elif repaired_p4_gate == P4_SUPPORTED_REPAIRED and min(factors.values()) >= material_reduction_factor_min:
        classification = RC2_CALIBRATION_WAS_PARTIAL_DRIVER_P4_REMAINS
        next_action = NEXT_ACTION_RC2_RC4
        interpretation = (
            "calibration materially inflated the old mismatch, but a genuine "
            "prior-to-encoded-support gap remains under the repaired coordinate."
        )
    elif repaired_p4_gate == P4_SUPPORTED_REPAIRED and max(factors.values()) <= same_order_reduction_factor_max:
        classification = RC4_CALIBRATION_REPAIR_DOES_NOT_RESOLVE_MISMATCH
        next_action = NEXT_ACTION_RC2_RC4
        interpretation = (
            "the repaired coordinate leaves the prior-to-encoded-support "
            "mismatch at the same severe order as the old result."
        )
    else:
        classification = RC5_UNDERDETERMINED
        next_action = NEXT_ACTION_RC5
        interpretation = (
            "the repaired-coordinate P4 result does not satisfy a registered "
            "RC1, RC2, or RC4 causal boundary."
        )
    return {
        "classification": classification,
        "next_action": next_action,
        "interpretation": interpretation,
        "repaired_p4_gate": repaired_p4_gate,
        "mean_scale_normal": means_normal,
        "prior_contract_coherent": bool(prior_contract_coherent),
        "reduction_factor": factors,
        "thresholds": {
            "mean_scale_tolerance": mean_scale_tolerance,
            "material_reduction_factor_min": material_reduction_factor_min,
            "same_order_reduction_factor_max": same_order_reduction_factor_max,
        },
    }


def validate_support_sample_identities(
    train_indices: torch.Tensor,
    heldout_indices: torch.Tensor,
    *,
    expected_train_start: int,
    expected_train_count: int,
    expected_heldout_indices: torch.Tensor,
    expected_heldout_sha256: str,
) -> dict[str, Any]:
    for value, name in ((train_indices, "train support indices"), (heldout_indices, "heldout indices"), (expected_heldout_indices, "expected heldout indices")):
        if not torch.is_tensor(value) or value.ndim != 1 or value.dtype != torch.long:
            raise ValueError(f"{name} must be a one-dimensional int64 tensor")
    expected_train = torch.arange(
        expected_train_start,
        expected_train_start + expected_train_count,
        dtype=torch.long,
    )
    if not torch.equal(train_indices.cpu(), expected_train):
        raise ValueError("train support indices do not match the registered range")
    if not torch.equal(heldout_indices.cpu(), expected_heldout_indices.cpu()):
        raise ValueError("heldout indices do not match the registered panel")
    observed_hash = tensor_sha256(heldout_indices.cpu())
    if observed_hash != expected_heldout_sha256:
        raise ValueError("heldout indices hash does not match the registered identity")
    return {
        "train_support_count": int(train_indices.numel()),
        "train_support_start": expected_train_start,
        "train_support_indices_sha256": tensor_sha256(train_indices.cpu()),
        "heldout_query_count": int(heldout_indices.numel()),
        "heldout_indices_sha256": observed_hash,
    }


def learned_state_freeze(
    before: str, after: str, *, model_training: bool
) -> dict[str, Any]:
    if before != after or model_training:
        raise ValueError("learned state changed or model remained in training mode")
    return {
        "state_digest_before": before,
        "state_digest_after": after,
        "state_unchanged": True,
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


def _artifact_path(directory: Path, name: str) -> Path:
    return directory / name


def _validate_old_g1_attempt(
    config: Mapping[str, Any], role: str
) -> dict[str, Any]:
    old_binding = _require_mapping(config["old_g1"], "old_g1")
    binding = _require_mapping(old_binding[role], f"old_g1.{role}")
    directory = _resolve(binding["directory"])
    manifest_path = _artifact_path(directory, "observer_manifest.json")
    metrics_path = _artifact_path(directory, "support_metrics.json")
    distances_path = _artifact_path(directory, "support_distances.pt")
    report_path = _artifact_path(directory, "compact_report.json")
    manifest = _json_file(manifest_path, expected_sha256=binding["manifest_sha256"])
    metrics = _json_file(metrics_path, expected_sha256=binding["metrics_sha256"])
    report = _json_file(report_path, expected_sha256=binding["compact_report_sha256"])
    distances = safe_torch_load(
        distances_path,
        expected_sha256=binding["distances_sha256"],
        map_location="cpu",
    )
    if manifest.get("schema") != OLD_G1_OBSERVER_SCHEMA or manifest.get("task_id") != OLD_G1_TASK_ID or manifest.get("status") != "completed":
        raise ValueError(f"old G1 {role} manifest identity is invalid")
    if metrics.get("schema") != OLD_G1_METRICS_SCHEMA or metrics.get("task_id") != OLD_G1_TASK_ID or metrics.get("status") != "completed":
        raise ValueError(f"old G1 {role} metrics identity is invalid")
    if report.get("schema") != OLD_G1_REPORT_SCHEMA or report.get("task_id") != OLD_G1_TASK_ID or report.get("result") != "P4_SUPPORTED":
        raise ValueError(f"old G1 {role} report identity is invalid")
    if role == "raw_attempt":
        if manifest.get("render_only_from_distance_artifact") is not None:
            raise ValueError("old G1 raw attempt is unexpectedly render-only")
    else:
        if manifest.get("render_only_from_distance_artifact") is not True or manifest.get("supersedes") != "lightgen_scaled_chen_g1_prior_support_accessibility_v1_attempt_01":
            raise ValueError("old G1 descriptive authority is not the registered redraw")
        source_manifest = _require_mapping(manifest.get("source_observer_manifest"), "old G1 source observer manifest")
        source_distance = _require_mapping(manifest.get("source_distance_artifact"), "old G1 source distance artifact")
        if source_manifest.get("sha256") != config["old_g1"]["raw_attempt"]["manifest_sha256"] or source_distance.get("sha256") != config["old_g1"]["raw_attempt"]["distances_sha256"]:
            raise ValueError("old G1 descriptive authority source binding is invalid")
    old_config_path = _resolve(old_binding["config"])
    old_config_payload = old_g1._load_config(old_config_path)
    old_file_hash = file_sha256(old_config_path)
    old_canonical_hash = canonical_sha256(old_config_payload)
    if old_file_hash != old_binding["config_sha256"] or old_canonical_hash != old_binding["config_canonical_sha256"]:
        raise ValueError("old G1 config hash is stale")
    if manifest.get("config", {}).get("file_sha256") != old_file_hash or manifest.get("config", {}).get("canonical_sha256") != old_canonical_hash:
        raise ValueError("old G1 artifact config identity is stale")
    if manifest.get("lineage", {}).get("checkpoint_sha256") != config["checkpoint"]["sha256"] or manifest.get("lineage", {}).get("checkpoint_step") != 5000 or manifest.get("lineage", {}).get("calibration_sha256") != "a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959":
        raise ValueError("old G1 artifact lineage is stale")
    dataset = _require_mapping(manifest.get("dataset"), "old G1 dataset")
    if dataset.get("train_support_count") != TRAIN_SUPPORT_COUNT or dataset.get("heldout_query_count") != HELDOUT_COUNT or dataset.get("heldout_indices_sha256") != config["dataset"]["heldout_indices_sha256"] or dataset.get("manifest_sha256") != config["dataset"]["mra_manifest_sha256"]:
        raise ValueError("old G1 dataset identity is stale")
    if manifest.get("prior_sampling") != old_config_payload["prior_sampling"] or manifest.get("distance") != old_config_payload["distance"] or manifest.get("bootstrap") != old_config_payload["bootstrap"]:
        raise ValueError("old G1 metric contract is stale")
    gate = _require_mapping(metrics.get("gate"), "old G1 gate")
    if gate.get("classification") != "P4_SUPPORTED":
        raise ValueError("old G1 P4 authority changed")
    freeze = _require_mapping(metrics.get("learned_state_freeze"), "old G1 learned-state freeze")
    if freeze.get("state_unchanged") is not True or freeze.get("optimizer_created") is not False or freeze.get("backward_called") is not False or freeze.get("refit") is not False or freeze.get("checkpoint_created") is not False:
        raise ValueError("old G1 no-update authority is invalid")
    if not isinstance(distances, Mapping) or distances.get("schema") != OLD_G1_DISTANCES_SCHEMA or distances.get("task_id") != OLD_G1_TASK_ID or distances.get("checkpoint_step") != 5000 or distances.get("checkpoint_sha256") != config["checkpoint"]["sha256"]:
        raise ValueError("old G1 distance artifact identity is invalid")
    train_support = _require_mapping(distances.get("train_support"), "old G1 train support")
    heldout_query = _require_mapping(distances.get("heldout_query"), "old G1 heldout query")
    prior_query = _require_mapping(distances.get("prior_query"), "old G1 prior query")
    if dict(train_support) != {"source": "registered_train_split_0_55000_full", "start": 0, "stop": TRAIN_SUPPORT_COUNT, "count": TRAIN_SUPPORT_COUNT} or dict(heldout_query) != {"source": "horizon_heldout_train_indices_55000_59999", "index_sha256": config["dataset"]["heldout_indices_sha256"], "count": HELDOUT_COUNT} or dict(prior_query) != {"seed": PRIOR_SEED, "kind": "truncated_normal", "count": HELDOUT_COUNT}:
        raise ValueError("old G1 sample identity artifact is invalid")
    holdout_distances = distances.get("d_holdout_to_train")
    prior_distances = distances.get("d_prior_to_train")
    if not torch.is_tensor(holdout_distances) or not torch.is_tensor(prior_distances) or holdout_distances.shape != (HELDOUT_COUNT,) or prior_distances.shape != (HELDOUT_COUNT,):
        raise ValueError("old G1 distance vectors have invalid shapes")
    if not torch.isfinite(holdout_distances).all() or not torch.isfinite(prior_distances).all() or (holdout_distances <= 0).any() or (prior_distances <= 0).any():
        raise ValueError("old G1 distance vectors are invalid")
    for name, expected in {
        "holdout": ("median", 0.0006687190311499265),
        "prior": ("median", 1.124173050103102),
    }.items():
        observed = _support_value(metrics, name, expected[0])
        if not math.isclose(observed, expected[1], rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise ValueError(f"old G1 {name} authority value changed")
    if not math.isclose(_support_value(metrics, "holdout", "p95"), 0.0011482780201326767, rel_tol=1.0e-12, abs_tol=1.0e-15) or not math.isclose(_support_value(metrics, "prior", "p95"), 1.2017228794802848, rel_tol=1.0e-12, abs_tol=1.0e-15):
        raise ValueError("old G1 p95 authority value changed")
    if report.get("next_action") != "PLAN_GENERATIVE_TRAINING_FOR_SUPPORT_ALIGNMENT":
        raise ValueError("old G1 next action authority changed")
    return {
        "role": role,
        "directory": directory,
        "manifest_path": manifest_path,
        "metrics_path": metrics_path,
        "distances_path": distances_path,
        "report_path": report_path,
        "manifest": manifest,
        "metrics": metrics,
        "distances": distances,
        "report": report,
        "config": old_config_payload,
    }


def _validate_replay_sources(config: Mapping[str, Any]) -> dict[str, Any]:
    rcr_binding = config["rcr_a"]
    rcr_config_path, rcr_config, rcr_file_hash, rcr_canonical_hash = rcr.load_replay_config(rcr_binding["config"])
    if rcr_file_hash != rcr_binding["config_sha256"] or rcr_canonical_hash != rcr_binding["config_canonical_sha256"]:
        raise ValueError("RCR-A config identity is stale")
    parent = rcr._validate_parent_mra(rcr_config)
    r2_identity = rcr._validate_r2(rcr_config)
    rcr_directory = _resolve(rcr_binding["output_directory"])
    rcr_manifest = _json_file(_artifact_path(rcr_directory, rcr_binding["manifest"]), expected_sha256=rcr_binding["manifest_sha256"])
    rcr_metrics = _json_file(_artifact_path(rcr_directory, rcr_binding["metrics"]), expected_sha256=rcr_binding["metrics_sha256"])
    rcr_report = _json_file(_artifact_path(rcr_directory, rcr_binding["compact_report"]), expected_sha256=rcr_binding["compact_report_sha256"])
    if rcr_manifest.get("schema") != RCR_A_MANIFEST_SCHEMA or rcr_manifest.get("task_id") != RCR_A_TASK_ID or rcr_manifest.get("status") != "completed" or rcr_manifest.get("gate") != RCR_A_GATE or rcr_manifest.get("stage_b_g1_replay_executed") is not False:
        raise ValueError("RCR-A completed authority is invalid")
    if rcr_metrics.get("schema") != RCR_A_METRICS_SCHEMA or rcr_metrics.get("task_id") != RCR_A_TASK_ID or rcr_metrics.get("status") != "completed" or rcr_metrics.get("gate") != RCR_A_GATE or rcr_metrics.get("stage_b_g1_replay_executed") is not False:
        raise ValueError("RCR-A metrics authority is invalid")
    if rcr_report.get("schema") != RCR_A_REPORT_SCHEMA or rcr_report.get("task_id") != RCR_A_TASK_ID or rcr_report.get("gate") != RCR_A_GATE:
        raise ValueError("RCR-A report authority is invalid")
    if dict(rcr_manifest.get("governance", {})) != GOVERNANCE or dict(rcr_metrics.get("governance", {})) != GOVERNANCE:
        raise ValueError("RCR-A governance boundary is invalid")
    for name, payload in {
        "manifest": rcr_manifest,
        "metrics": rcr_metrics,
    }.items():
        freeze = _require_mapping(payload.get("learned_state_freeze"), f"RCR-A {name} learned-state freeze")
        if freeze.get("state_unchanged") is not True or freeze.get("optimizer_created") is not False or freeze.get("backward_called") is not False or freeze.get("refit") is not False or freeze.get("checkpoint_created") is not False or freeze.get("decoder_entered") is not False:
            raise ValueError(f"RCR-A {name} learned-state boundary is invalid")

    psd_binding = config["pcr_psd"]
    psd_directory = _resolve(psd_binding["output_directory"])
    psd_manifest = _json_file(_artifact_path(psd_directory, psd_binding["manifest"]), expected_sha256=psd_binding["manifest_sha256"])
    psd_metrics = _json_file(_artifact_path(psd_directory, psd_binding["metrics"]), expected_sha256=psd_binding["metrics_sha256"])
    psd_report = _json_file(_artifact_path(psd_directory, psd_binding["compact_report"]), expected_sha256=psd_binding["compact_report_sha256"])
    if psd_manifest.get("schema") != "lightgen_scaled_chen_prior_shape_decomposition_manifest.v1" or psd_manifest.get("task_id") != PCR_PSD_TASK_ID or psd_manifest.get("status") != "completed" or psd_manifest.get("classification") != PCR_PSD_CLASSIFICATION or psd_manifest.get("next_action") != "IMPLEMENT_REPAIRED_COORDINATE_G1_SA_REPLAY":
        raise ValueError("PCR-PSD completed authority is invalid")
    if psd_metrics.get("schema") != "lightgen_scaled_chen_prior_shape_decomposition_metrics.v1" or psd_metrics.get("task_id") != PCR_PSD_TASK_ID or psd_metrics.get("status") != "completed" or psd_metrics.get("classification") != PCR_PSD_CLASSIFICATION:
        raise ValueError("PCR-PSD metrics authority is invalid")
    if psd_report.get("schema") != "lightgen_scaled_chen_prior_shape_decomposition_compact_report.v1" or psd_report.get("task_id") != PCR_PSD_TASK_ID or psd_report.get("classification") != PCR_PSD_CLASSIFICATION:
        raise ValueError("PCR-PSD report authority is invalid")
    if dict(psd_manifest.get("governance", {})) != GOVERNANCE or dict(psd_metrics.get("governance", {})) != GOVERNANCE or dict(psd_report.get("governance", {})) != GOVERNANCE:
        raise ValueError("PCR-PSD governance boundary is invalid")
    for split in ("P-S3", "P-S4", "P-S5"):
        final = _require_mapping(psd_metrics.get("final_decomposition"), "PCR-PSD final decomposition")
        entry = _require_mapping(final.get(split), f"PCR-PSD {split}")
        if entry.get("classification") != f"{split}_SUPPORTED" or entry.get("replication_status") != "REPLICATED_DIRECTION":
            raise ValueError(f"PCR-PSD {split} authority changed")

    old_config_path = _resolve(config["old_g1"]["config"])
    old_config = old_g1._load_config(old_config_path)
    old_file_hash = file_sha256(old_config_path)
    old_canonical_hash = canonical_sha256(old_config)
    if old_file_hash != config["old_g1"]["config_sha256"] or old_canonical_hash != config["old_g1"]["config_canonical_sha256"]:
        raise ValueError("old G1 config identity is stale")
    old_raw = _validate_old_g1_attempt(config, "raw_attempt")
    old_descriptive = _validate_old_g1_attempt(config, "descriptive_attempt")
    if old_raw["metrics"] != old_descriptive["metrics"] or old_raw["report"] != old_descriptive["report"] or not torch.equal(old_raw["distances"]["d_holdout_to_train"], old_descriptive["distances"]["d_holdout_to_train"]) or not torch.equal(old_raw["distances"]["d_prior_to_train"], old_descriptive["distances"]["d_prior_to_train"]):
        raise ValueError("old G1 attempt_01 and attempt_04 authorities diverge")
    return {
        "rcr_config_path": rcr_config_path,
        "rcr_config": rcr_config,
        "rcr_manifest": rcr_manifest,
        "rcr_metrics": rcr_metrics,
        "rcr_report": rcr_report,
        "parent": parent,
        "r2": r2_identity,
        "psd_manifest": psd_manifest,
        "psd_metrics": psd_metrics,
        "psd_report": psd_report,
        "old_config_path": old_config_path,
        "old_config": old_config,
        "old_raw": old_raw,
        "old_descriptive": old_descriptive,
    }


def _validate_worktree_scope(state: Mapping[str, Any]) -> None:
    if state.get("branch") != BRANCH:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_BRANCH: expected {BRANCH}")
    status = state.get("status", [])
    paths: set[str] = set()
    for entry in status:
        text = str(entry)
        paths.add(text[3:] if len(text) >= 3 and text[2] == " " else text)
    unexpected = sorted(paths - EXPECTED_DIRTY_PATHS)
    if unexpected:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_DIRTY_PATH: {unexpected}")


def _normalize_coefficients(
    coefficients: torch.Tensor, calibration: bogt.Calibration | bogt.UnitAwareCalibration
) -> torch.Tensor:
    if coefficients.ndim != 2 or coefficients.shape[1] != PORT_COUNT or not torch.is_complex(coefficients) or not torch.isfinite(coefficients).all():
        raise ValueError("raw coefficients must be finite complex [N,100]")
    powers = coefficients.abs().square()
    result = bogt.normalize_port_power(powers, calibration.port_scale).detach().cpu().contiguous()
    validate_latent_bank(result)
    return result


def _mra_context(metrics: Mapping[str, Any]) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for split in ("selection", "confirmation"):
        observed = _require_mapping(metrics["panels"][split]["repaired_coordinate"], f"RCR-A {split} repaired coordinate")
        expected = EXPECTED_MRA_CONTEXT[split]
        result = {
            name: float(observed[name])
            for name in expected
        }
        for name, target in expected.items():
            if not math.isclose(result[name], target, rel_tol=1.0e-5, abs_tol=1.0e-5):
                raise ValueError(f"RCR-A {split} MRA context anchor changed: {name}")
        context[split] = result
    context["complex_rank_invariant"] = True
    context["source"] = "RCR-A mra_replay_metrics.json; PCR-PSD not rerun"
    return context


def _invariant_controls(
    train_coefficients: torch.Tensor,
    heldout_coefficients: torch.Tensor,
    rcr_metrics: Mapping[str, Any],
    *,
    checkpoint_hash: str,
    state_before: str,
    state_after: str,
) -> dict[str, Any]:
    confirmation = _require_mapping(
        _require_mapping(rcr_metrics["invariant_controls"], "RCR-A invariant_controls")["panels"]["confirmation"],
        "RCR-A confirmation invariant",
    )
    train_power = train_coefficients.abs().square().contiguous()
    heldout_power = heldout_coefficients.abs().square().contiguous()
    raw_hashes = {
        "train_support": tensor_sha256(train_coefficients),
        "heldout_query": tensor_sha256(heldout_coefficients),
    }
    power_hashes = {
        "train_support": tensor_sha256(train_power),
        "heldout_query": tensor_sha256(heldout_power),
    }
    expected_raw = confirmation["raw_complex_coefficients_sha256"]
    expected_power = confirmation["raw_modal_coefficient_powers_sha256"]
    if raw_hashes["heldout_query"] != expected_raw or power_hashes["heldout_query"] != expected_power:
        raise ValueError("repaired replay raw coefficient identity disagrees with RCR-A")
    current_rank = rcr.mra_participation_rank(heldout_coefficients)
    expected_rank = float(confirmation["complex_modal_effective_rank"]["same_prefix_replay"])
    rank_match = math.isclose(current_rank, expected_rank, rel_tol=1.0e-5, abs_tol=1.0e-8)
    if not rank_match:
        raise ValueError("repaired replay complex-rank identity disagrees with RCR-A")
    checkpoint_match = checkpoint_hash == "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de"
    state_match = state_before == state_after
    if not checkpoint_match or not state_match:
        raise ValueError("repaired replay learned-state or checkpoint invariant failed")
    return {
        "raw_complex_coefficients_same_for_old_and_repaired": True,
        "raw_modal_coefficient_powers_same_for_old_and_repaired": True,
        "raw_complex_coefficients_sha256": raw_hashes,
        "raw_modal_coefficient_powers_sha256": power_hashes,
        "heldout_raw_hash_matches_rcr_a": True,
        "heldout_raw_power_hash_matches_rcr_a": True,
        "heldout_complex_effective_rank": {
            "rcr_a": expected_rank,
            "replay": current_rank,
            "match": rank_match,
        },
        "checkpoint_identity": {
            "expected_sha256": "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de",
            "match": checkpoint_match,
        },
        "learned_state_identity": {
            "before": state_before,
            "after": state_after,
            "match": state_match,
        },
        "passed": True,
    }


def _figure_alignment(stem: str) -> dict[str, Any]:
    return {
        "schema": ALIGNMENT_SCHEMA,
        "figure": stem,
        "backend": "matplotlib",
        "verdict": "NOT_APPLICABLE_SINGLE_PANEL",
        "panel_count": 1,
        "axes_count": 1,
        "text_editable": True,
        "reason": "single-panel latent support-accessibility diagnostic figure",
    }


def audit_layout_manifest(output_path: Path) -> None:
    """Fail closed if either exported single-panel figure loses its layout contract."""
    for stem in ("support_distance_distribution", "support_distance_summary"):
        path = output_path / f"{stem}.alignment.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{stem} alignment manifest must be an object")
        if payload != _figure_alignment(stem):
            raise ValueError(f"{stem} alignment manifest is invalid")


def _save_figures(
    output_path: Path,
    holdout_distances: torch.Tensor,
    prior_distances: torch.Tensor,
    metrics: Mapping[str, Any],
) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    sorted_holdout = torch.sort(holdout_distances).values.numpy()
    sorted_prior = torch.sort(prior_distances).values.numpy()
    y_holdout = (
        torch.arange(1, holdout_distances.numel() + 1, dtype=torch.float64)
        / holdout_distances.numel()
    ).numpy()
    y_prior = (
        torch.arange(1, prior_distances.numel() + 1, dtype=torch.float64)
        / prior_distances.numel()
    ).numpy()
    x_max = max(float(prior_distances.max()), float(holdout_distances.max())) * 1.05
    figure = plt.figure(figsize=(6.2, 4.0))
    axis = figure.add_axes((0.12, 0.24, 0.72, 0.61))
    axis.step(sorted_holdout, y_holdout, where="post", color="#1f77b4", linewidth=1.6, label="Encoded holdout")
    axis.step(sorted_prior, y_prior, where="post", color="#d62728", linewidth=1.6, label="Truncated-normal prior")
    axis.set_xlim(0.0, x_max)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Normalized nearest-support distance")
    axis.set_ylabel("Empirical cumulative probability")
    axis.set_title("Repaired-coordinate G1 support accessibility\nexact nearest-support ECDF")
    axis.grid(True, alpha=0.25)
    figure.text(
        0.48,
        0.965,
        "REPAIRED CALIBRATION COORDINATE",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    figure.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.48, 0.04),
        ncol=2,
        handlelength=2.2,
        columnspacing=1.4,
    )
    for suffix, kwargs in {
        "png": {"dpi": 220},
        "pdf": {},
        "svg": {},
    }.items():
        figure.savefig(output_path / f"support_distance_distribution.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(figure)

    labels = ["H50", "P50", "H95", "P95"]
    values = [
        metrics["holdout"]["median"],
        metrics["prior"]["median"],
        metrics["holdout"]["p95"],
        metrics["prior"]["p95"],
    ]
    colors = ["#1f77b4", "#d62728", "#1f77b4", "#d62728"]
    figure = plt.figure(figsize=(6.5, 4.0))
    axis = figure.add_axes((0.11, 0.16, 0.55, 0.69))
    axis.bar(labels, values, color=colors, width=0.62)
    axis.set_yscale("log")
    axis.set_ylim(min(values) * 0.75, max(values) * 1.5)
    axis.set_ylabel("Normalized nearest-support distance (log scale)")
    axis.set_title("Repaired-coordinate G1 support-distance summary")
    axis.grid(True, axis="y", alpha=0.25)
    figure.text(
        0.98,
        0.78,
        "R50 = {:.3f} [{:.3f}, {:.3f}]\nR95 = {:.3f} [{:.3f}, {:.3f}]".format(
            metrics["ratios"]["r50"],
            metrics["bootstrap"]["r50_ci"][0],
            metrics["bootstrap"]["r50_ci"][1],
            metrics["ratios"]["r95"],
            metrics["bootstrap"]["r95_ci"][0],
            metrics["bootstrap"]["r95_ci"][1],
        ),
        ha="right",
        va="top",
        fontsize=7.5,
    )
    figure.text(
        0.48,
        0.965,
        "REPAIRED CALIBRATION COORDINATE",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
    )
    for suffix, kwargs in {
        "png": {"dpi": 220},
        "pdf": {},
        "svg": {},
    }.items():
        figure.savefig(output_path / f"support_distance_summary.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(figure)


def _artifact_refs(output_path: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for target in sorted(output_path.iterdir(), key=lambda item: item.name):
        if target.is_file() and target.name != "observer_manifest.json":
            refs[target.name] = {
                "path": target.name,
                "sha256": file_sha256(target),
                "size_bytes": target.stat().st_size,
            }
    return refs


def _assert_output_tree(output_path: Path) -> None:
    actual = {
        target.relative_to(output_path).as_posix()
        for target in output_path.rglob("*")
        if target.is_file()
    }
    if actual != EXPECTED_OUTPUT_FILES - {"observer_manifest.json"} and actual != EXPECTED_OUTPUT_FILES:
        raise RuntimeError(f"RCR-B output tree is not exact: {sorted(actual)}")
    if any(
        ".tmp" in name
        or Path(name).name.startswith(".")
        or name.lower().endswith((".tif", ".tiff"))
        for name in actual
    ):
        raise RuntimeError("RCR-B output contains temporary, hidden, or TIFF files")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_new_json(path, payload, artifact_kind="RCR-B support-accessibility artifact")


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    config_target, config, config_file_hash, config_canonical_hash = _load_config(config_path)
    state = base._git_state()
    _validate_worktree_scope(state)
    sources = _validate_replay_sources(config)

    requested_output = _resolve(output_path)
    actual_output = next_write_once_output(requested_output)
    free_before, total_cuda, disk_free_before = rcr._resource_preflight(config, actual_output)
    data = rcr.mra._load_data(_resolve(config["dataset"]["dataset_root"]))
    if data.manifest.get("manifest_hash") != config["dataset"]["mra_manifest_sha256"] or data.manifest.get("confirmation_indices_sha256") != config["dataset"]["heldout_indices_sha256"]:
        raise ValueError("registered dataset or heldout identity is stale")
    if tuple(data.images.shape) != (60000, 1, 28, 28) or tuple(data.labels.shape) != (60000,):
        raise ValueError("registered MNIST train split shape is invalid")

    train_indices = torch.arange(0, TRAIN_SUPPORT_COUNT, dtype=torch.long)
    heldout_indices = data.confirmation_indices.cpu().contiguous()
    sample_identity = validate_support_sample_identities(
        train_indices,
        heldout_indices,
        expected_train_start=0,
        expected_train_count=TRAIN_SUPPORT_COUNT,
        expected_heldout_indices=sources["parent"]["panel"]["confirmation_indices"],
        expected_heldout_sha256=config["dataset"]["heldout_indices_sha256"],
    )
    if sample_identity["heldout_query_count"] != HELDOUT_COUNT:
        raise ValueError("registered heldout query count is invalid")

    checkpoint, checkpoint_path = rcr._load_checkpoint(config, sources["parent"]["source"])
    checkpoint_hash = file_sha256(checkpoint_path)
    device = torch.device(config["execution"]["device"])
    model = rcr.mra._load_scaled_model(sources["parent"]["formal"], checkpoint, device)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()) or any(parameter.requires_grad for parameter in model.ols.parameters()):
        raise RuntimeError("RCR-B model is not frozen before replay")
    if bogt.mode_bank_sha256(model) != config["r2_calibration"]["mode_bank_sha256"]:
        raise ValueError("RCR-B model mode bank does not match R2 calibration")
    state_before = state_digest(model)
    rcr_state_before = sources["rcr_metrics"]["learned_state_freeze"]["state_digest_before"]
    if state_before != rcr_state_before:
        raise ValueError("RCR-B checkpoint learned-state digest disagrees with RCR-A")
    torch.cuda.reset_peak_memory_stats(device)

    train_coefficients = rcr._collect_coefficients(
        model,
        data.images,
        train_indices,
        device=device,
        batch_size=config["execution"]["batch_size"],
    )
    heldout_coefficients = rcr._collect_coefficients(
        model,
        data.images,
        heldout_indices,
        device=device,
        batch_size=config["execution"]["batch_size"],
    )
    _old_calibration, old_calibration_path = rcr._load_old_calibration_from_source(sources["parent"]["source"])
    repaired_calibration = sources["r2"]["calibration"]
    repaired_train_z = _normalize_coefficients(train_coefficients, repaired_calibration)
    repaired_heldout_z = _normalize_coefficients(heldout_coefficients, repaired_calibration)
    prior_z = sample_registered_prior(HELDOUT_COUNT, seed=PRIOR_SEED)
    if not torch.equal(prior_z, sample_registered_prior(HELDOUT_COUNT, seed=PRIOR_SEED)):
        raise ValueError("registered prior seed is not deterministic")
    old_replay_anchor = _require_mapping(
        sources["old_descriptive"]["metrics"].get("replay_anchor"),
        "old G1 replay_anchor",
    )
    old_prior_mean = old_replay_anchor.get("prior_empirical_mean")
    if not isinstance(old_prior_mean, (int, float)) or not math.isfinite(float(old_prior_mean)):
        raise ValueError("old G1 prior empirical mean identity is missing")
    if not math.isclose(float(prior_z.mean()), old_prior_mean, rel_tol=1.0e-6, abs_tol=1.0e-7):
        raise ValueError("registered prior sample identity disagrees with old G1 authority")
    validate_latent_bank(repaired_train_z)
    validate_latent_bank(repaired_heldout_z)

    torch.cuda.synchronize(device)
    distance_support = repaired_train_z.to(device)
    distance_heldout = repaired_heldout_z.to(device)
    distance_prior = prior_z.to(device)
    holdout_distances = exact_normalized_nearest_distances(
        distance_heldout,
        distance_support,
        query_chunk_size=config["distance"]["query_chunk_size"],
        support_chunk_size=config["distance"]["support_chunk_size"],
    ).cpu()
    prior_distances = exact_normalized_nearest_distances(
        distance_prior,
        distance_support,
        query_chunk_size=config["distance"]["query_chunk_size"],
        support_chunk_size=config["distance"]["support_chunk_size"],
    ).cpu()
    torch.cuda.synchronize(device)
    del distance_support, distance_heldout, distance_prior

    state_after = state_digest(model)
    freeze = learned_state_freeze(state_before, state_after, model_training=model.training)
    invariant_controls = _invariant_controls(
        train_coefficients,
        heldout_coefficients,
        sources["rcr_metrics"],
        checkpoint_hash=checkpoint_hash,
        state_before=state_before,
        state_after=state_after,
    )
    mra_context = _mra_context(sources["rcr_metrics"])
    bootstrap = bootstrap_ratio_ci(
        holdout_distances,
        prior_distances,
        replicates=config["bootstrap"]["replicates"],
        seed=config["bootstrap"]["seed"],
        confidence_level=config["bootstrap"]["confidence_level"],
    )
    repaired_gate = classify_repaired_p4_gate(bootstrap["r50_ci"], bootstrap["r95_ci"])
    repaired_metrics: dict[str, Any] = {
        "schema": METRICS_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "coordinate": {
            "schema": NEW_CALIBRATION_SCHEMA,
            "contract_version": NEW_CALIBRATION_CONTRACT,
            "definition": "z_j=abs(c_j)^2/s_j_v2",
            "dimension": LATENT_DIM,
            "dtype": str(repaired_heldout_z.dtype),
            "distance_dtype": str(holdout_distances.dtype),
            "normalization": "divide_by_sqrt_100",
            "scale_source": "R2 unit-aware raw mean port_scale",
        },
        "dataset": {
            **sample_identity,
            "mra_manifest_sha256": data.manifest["manifest_hash"],
            "train_support_source": config["dataset"]["train_support_source"],
            "heldout_source": config["dataset"]["heldout_source"],
            "input_shape": list(data.images.shape[1:]),
            "rendered_shape": config["dataset"]["rendered_shape"],
            "panel_reused_without_resampling": True,
        },
        "calibration": {
            "old": {
                "schema": OLD_CALIBRATION_SCHEMA,
                "path": _relative_path(old_calibration_path),
                "sha256": "a9a871405e216597b2c6058e8c52cd86a242659f8f6729fe06b85d0ef0f32959",
                "semantic": "epsilon-dominated_calibration_coordinate",
            },
            "repaired": {
                "schema": NEW_CALIBRATION_SCHEMA,
                "contract_version": NEW_CALIBRATION_CONTRACT,
                "path": _relative_path(sources["r2"]["payload_path"]),
                "sha256": sources["r2"]["payload_sha256"],
                "manifest_sha256": sources["r2"]["manifest_sha256"],
                "sample_count": sources["r2"]["calibration"].sample_count,
                "port_count": PORT_COUNT,
                "calibration_indices_sha256": config["r2_calibration"]["calibration_indices_sha256"],
                "dataset_manifest_sha256": config["r2_calibration"]["dataset_manifest_sha256"],
                "mode_bank_sha256": config["r2_calibration"]["mode_bank_sha256"],
                "zero_phase_reference": config["r2_calibration"]["zero_phase_reference"],
                "plane_identity": config["r2_calibration"]["plane_identity"],
                "raw_port_mean_equals_port_scale": True,
            },
        },
        "holdout": _summary_stats(holdout_distances),
        "prior": {
            **_summary_stats(prior_distances),
            "sample_tensor_sha256": tensor_sha256(prior_z),
            "empirical_mean": float(prior_z.mean()),
            "empirical_variance": float(prior_z.var(unbiased=False)),
        },
        "ratios": {
            "r50": bootstrap["r50"],
            "r95": bootstrap["r95"],
        },
        "bootstrap": bootstrap,
        "gate": {
            "name": "P4",
            "coordinate": "repaired_calibration_unit_aware_raw_mean_v2",
            "classification": repaired_gate,
            "rule": "lower CI(R50)>1 and lower CI(R95)>1; or upper CI(R50)<=1 and upper CI(R95)<=1",
        },
        "descriptive": {
            "prior_exceedance_h95": float((prior_distances > _quantile(holdout_distances, 0.95)).to(torch.float64).mean()),
            "interpretation": "descriptive_only_not_a_primary_gate",
        },
        "repaired_latent_anchor": {
            "train_support": rcr.repaired_latent_anchor(repaired_train_z),
            "heldout_query": rcr.repaired_latent_anchor(repaired_heldout_z),
            "prior_query": rcr.repaired_latent_anchor(prior_z),
        },
        "mra_context": mra_context,
        "old_authority": {
            "raw_measurement_attempt": "lightgen_scaled_chen_g1_prior_support_accessibility_v1_attempt_01",
            "descriptive_attempt": "lightgen_scaled_chen_g1_prior_support_accessibility_v1_attempt_04",
            "gate": "P4_SUPPORTED",
            "r50": sources["old_descriptive"]["metrics"]["bootstrap"]["r50"],
            "r50_ci": sources["old_descriptive"]["metrics"]["bootstrap"]["r50_ci"],
            "r95": sources["old_descriptive"]["metrics"]["bootstrap"]["r95"],
            "r95_ci": sources["old_descriptive"]["metrics"]["bootstrap"]["r95_ci"],
        },
        "invariant_controls": invariant_controls,
        "learned_state_freeze": freeze,
        "resource": {
            "device": str(device),
            "free_cuda_bytes_before": free_before,
            "total_cuda_bytes": total_cuda,
            "disk_free_bytes_before": disk_free_before,
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_memory_gate_pass": int(torch.cuda.max_memory_allocated(device)) <= config["execution"]["max_peak_cuda_bytes"],
            "wall_clock_seconds": time.perf_counter() - started,
            "distance_query_chunk_size": config["distance"]["query_chunk_size"],
            "distance_support_chunk_size": config["distance"]["support_chunk_size"],
            "support_size": TRAIN_SUPPORT_COUNT,
            "heldout_query_size": HELDOUT_COUNT,
            "prior_query_size": HELDOUT_COUNT,
            "distance_dtype": str(holdout_distances.dtype),
        },
        "reconstruction_boundary": {
            "reconstruction_supported": False,
            "decoder_entered": False,
            "relay_after_pre_ols": False,
            "cmos_called": False,
            "mpcdar_called": False,
            "context": "HISTORICAL_CONTEXT_ONLY",
        },
        "governance": dict(GOVERNANCE),
    }
    old_metrics = sources["old_descriptive"]["metrics"]
    comparison = compare_old_and_repaired_support(old_metrics, repaired_metrics)
    final_causal = classify_final_causal_result(
        repaired_gate,
        old_ratios={"r50": old_metrics["bootstrap"]["r50"], "r95": old_metrics["bootstrap"]["r95"]},
        repaired_ratios=repaired_metrics["ratios"],
        selection_mean=mra_context["selection"]["z_mean"],
        confirmation_mean=mra_context["confirmation"]["z_mean"],
        prior_contract_coherent=True,
        mean_scale_tolerance=config["causal_classification"]["mean_scale_tolerance"],
        material_reduction_factor_min=config["causal_classification"]["material_reduction_factor_min"],
        same_order_reduction_factor_max=config["causal_classification"]["same_order_reduction_factor_max"],
    )
    repaired_metrics["final_causal_classification"] = final_causal
    repaired_metrics["next_action"] = final_causal["next_action"]
    comparison["old_authority"] = {
        "attempt_01_manifest_sha256": config["old_g1"]["raw_attempt"]["manifest_sha256"],
        "attempt_04_manifest_sha256": config["old_g1"]["descriptive_attempt"]["manifest_sha256"],
        "support_distances_sha256": config["old_g1"]["raw_attempt"]["distances_sha256"],
        "gate": "P4_SUPPORTED",
    }
    comparison["repaired"] = {
        "gate": repaired_gate,
        "coordinate": NEW_CALIBRATION_CONTRACT,
        "classification": final_causal["classification"],
    }

    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "gate": repaired_gate,
        "final_causal_classification": final_causal["classification"],
        "r50": repaired_metrics["ratios"]["r50"],
        "r50_ci": bootstrap["r50_ci"],
        "r95": repaired_metrics["ratios"]["r95"],
        "r95_ci": bootstrap["r95_ci"],
        "prior_exceedance_h95": repaired_metrics["descriptive"]["prior_exceedance_h95"],
        "old_to_repaired": comparison["old_to_repaired"],
        "reduction_factor": comparison["reduction_factor"],
        "interpretation": (
            final_causal["interpretation"]
            + " This latent-support result does not establish decoder acceptance, valid digits, diversity, novelty, or conditional generation."
        ),
        "governance": dict(GOVERNANCE),
        "next_action": final_causal["next_action"],
    }

    actual_output.mkdir(parents=True, exist_ok=False)
    distance_payload = {
        "schema": DISTANCES_SCHEMA,
        "task_id": TASK_ID,
        "coordinate": NEW_CALIBRATION_CONTRACT,
        "checkpoint_step": 5000,
        "checkpoint_sha256": checkpoint_hash,
        "train_support": {
            "source": config["dataset"]["train_support_source"],
            "start": 0,
            "stop": TRAIN_SUPPORT_COUNT,
            "count": TRAIN_SUPPORT_COUNT,
            "indices_sha256": sample_identity["train_support_indices_sha256"],
        },
        "heldout_query": {
            "source": config["dataset"]["heldout_source"],
            "index_sha256": sample_identity["heldout_indices_sha256"],
            "count": HELDOUT_COUNT,
        },
        "prior_query": {
            "seed": PRIOR_SEED,
            "kind": "truncated_normal",
            "count": HELDOUT_COUNT,
            "sample_tensor_sha256": tensor_sha256(prior_z),
        },
        "metric": config["distance"],
        "d_holdout_to_train": holdout_distances,
        "d_prior_to_train": prior_distances,
    }
    atomic_torch_save_new(actual_output / "support_distances.pt", distance_payload, artifact_kind="RCR-B distance artifact")
    _save_figures(actual_output, holdout_distances, prior_distances, repaired_metrics)
    _write_json(actual_output / "support_distance_distribution.alignment.json", _figure_alignment("support_distance_distribution"))
    _write_json(actual_output / "support_distance_summary.alignment.json", _figure_alignment("support_distance_summary"))
    audit_layout_manifest(actual_output)
    _write_json(actual_output / "support_metrics.json", repaired_metrics)
    _write_json(actual_output / "old_new_support_comparison.json", comparison)
    _write_json(actual_output / "compact_report.json", report)
    _assert_output_tree(actual_output)
    refs = _artifact_refs(actual_output)
    storage_bytes = sum(item["size_bytes"] for item in refs.values())
    if storage_bytes > config["execution"]["max_storage_bytes"]:
        raise RuntimeError("RCR-B output exceeds the diagnostic storage contract")
    manifest = {
        "schema": OBSERVER_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "completed",
        "attempt": actual_output.name,
        "requested_output": _relative_path(requested_output),
        "output_directory": _relative_path(actual_output),
        "git": state,
        "config": {
            "path": _relative_path(config_target),
            "file_sha256": config_file_hash,
            "canonical_sha256": config_canonical_hash,
        },
        "rcr_a": {
            "config": _relative_path(sources["rcr_config_path"]),
            "manifest": _relative_path(_artifact_path(_resolve(config["rcr_a"]["output_directory"]), config["rcr_a"]["manifest"])),
            "manifest_sha256": config["rcr_a"]["manifest_sha256"],
            "metrics": _relative_path(_artifact_path(_resolve(config["rcr_a"]["output_directory"]), config["rcr_a"]["metrics"])),
            "metrics_sha256": config["rcr_a"]["metrics_sha256"],
            "compact_report_sha256": config["rcr_a"]["compact_report_sha256"],
            "gate": RCR_A_GATE,
        },
        "pcr_psd": {
            "manifest": _relative_path(_artifact_path(_resolve(config["pcr_psd"]["output_directory"]), config["pcr_psd"]["manifest"])),
            "manifest_sha256": config["pcr_psd"]["manifest_sha256"],
            "metrics_sha256": config["pcr_psd"]["metrics_sha256"],
            "compact_report_sha256": config["pcr_psd"]["compact_report_sha256"],
            "classification": PCR_PSD_CLASSIFICATION,
        },
        "old_g1": {
            "raw_attempt_01_manifest_sha256": config["old_g1"]["raw_attempt"]["manifest_sha256"],
            "descriptive_attempt_04_manifest_sha256": config["old_g1"]["descriptive_attempt"]["manifest_sha256"],
            "support_distances_sha256": config["old_g1"]["raw_attempt"]["distances_sha256"],
            "old_gate": "P4_SUPPORTED",
        },
        "r2_calibration": {
            "schema": NEW_CALIBRATION_SCHEMA,
            "contract_version": NEW_CALIBRATION_CONTRACT,
            "payload": _relative_path(sources["r2"]["payload_path"]),
            "payload_sha256": sources["r2"]["payload_sha256"],
            "manifest": _relative_path(sources["r2"]["manifest_path"]),
            "manifest_sha256": sources["r2"]["manifest_sha256"],
            "sample_count": sources["r2"]["calibration"].sample_count,
            "port_count": PORT_COUNT,
            "calibration_indices_sha256": config["r2_calibration"]["calibration_indices_sha256"],
            "dataset_manifest_sha256": config["r2_calibration"]["dataset_manifest_sha256"],
            "mode_bank_sha256": config["r2_calibration"]["mode_bank_sha256"],
            "zero_phase_reference": config["r2_calibration"]["zero_phase_reference"],
            "plane_identity": config["r2_calibration"]["plane_identity"],
        },
        "checkpoint": {
            "path": _relative_path(checkpoint_path),
            "schema": CHECKPOINT_SCHEMA,
            "global_step": 5000,
            "sha256": checkpoint_hash,
        },
        "dataset": repaired_metrics["dataset"],
        "prior_sampling": config["prior_sampling"],
        "distance": config["distance"],
        "bootstrap": config["bootstrap"],
        "gate": repaired_metrics["gate"],
        "causal_classification": final_causal,
        "measurement": config["measurement"],
        "repaired_latent_anchor": repaired_metrics["repaired_latent_anchor"],
        "invariant_controls": invariant_controls,
        "learned_state_freeze": freeze,
        "no_update_audit": freeze,
        "reconstruction_boundary": repaired_metrics["reconstruction_boundary"],
        "governance": dict(GOVERNANCE),
        "next_action": final_causal["next_action"],
        "artifacts": refs,
        "storage_bytes_excluding_manifest": storage_bytes,
    }
    _write_json(actual_output / "observer_manifest.json", manifest)
    _assert_output_tree(actual_output)
    del model, train_coefficients, heldout_coefficients
    return {
        "output_directory": str(actual_output),
        "metrics": repaired_metrics,
        "comparison": comparison,
        "report": report,
        "manifest": manifest,
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
