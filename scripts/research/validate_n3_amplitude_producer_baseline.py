"""Validate the frozen r=99 N3 amplitude producer on held-out data.

The consumer loads the immutable producer state and internal-validation bank,
performs no refit, and uses fixed CPU RNG streams for positivity-conditioned
producer samples.  It does not invoke an optical generator or evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)


TASK_ID = "VALIDATE_N3_AMPLITUDE_PRODUCER_BASELINE_V1"
GATE = "VALIDATE_N3_AMPLITUDE_PRODUCER_BASELINE_V1"
REPORT_SCHEMA = "n3_amplitude_producer_validation_report.v1"
SAMPLE_BANK_SCHEMA = "n3_amplitude_producer_validation_sample_banks.v1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
CLASSIFICATION_SUPPORTED = "N3_AMPLITUDE_PRODUCER_BASELINE_SUPPORTED"
CLASSIFICATION_NOT_SUPPORTED = "N3_AMPLITUDE_PRODUCER_BASELINE_NOT_SUPPORTED"
CLASSIFICATION_MIXED = "N3_AMPLITUDE_PRODUCER_BASELINE_MIXED"

DATA_DIR = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_ols_producer_data_authority_v1_attempt_01"
)
SOURCE_MANIFEST = DATA_DIR / "ols_producer_source_manifest.v1.json"
PARTITION_MANIFEST = DATA_DIR / "ols_producer_partition_manifest.v1.json"
FIREWALL_MANIFEST = DATA_DIR / "ols_producer_data_firewall_manifest.v1.json"
VALIDATION_BANK = DATA_DIR / "ols_producer_internal_validation_bank.v1.pt"
MODEL_ORDER_RESULT = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_n3_low_rank_model_order_selection_v1_attempt_01/"
    "model_order_result.v1.json"
)
PRODUCER_STATE = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_n3_amplitude_producer_baseline_v1_attempt_01/"
    "n3_amplitude_producer_state.v1.pt"
)
PRODUCER_MANIFEST = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_n3_amplitude_producer_baseline_v1_attempt_01/"
    "n3_amplitude_producer_state_manifest.v1.json"
)

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "d07fc3fa927796e97254fa09a8c0eceb18d6555a99590db43217a685bf0d1687"
)
EXPECTED_SOURCE_ARTIFACT_SHA256 = (
    "9375b08597f03fee3798e05798779556c3e7534db6a9d39033232f7278d32279"
)
EXPECTED_SOURCE_SEMANTIC_MANIFEST_SHA256 = (
    "b0199040926618b17373edc4f167cc8bc4788f39f64de174e1a3fb225781833e"
)
EXPECTED_SOURCE_INDEX_SHA256 = (
    "532f05aa60d235499fdc5c078ad9c3c95c36a0c61f2c8e9d64a8309c1741255d"
)
EXPECTED_SOURCE_LABEL_SHA256 = (
    "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"
)
EXPECTED_SOURCE_R3_SHA256 = (
    "dcade4dca73f86f7bdc8d9be8fe183a8eda6a5f3fe2b810986d846ddf4c12fd3"
)
EXPECTED_PARTITION_MANIFEST_SHA256 = (
    "409b598aacae921da631adea553b339b56560231f9cb2cf75ce8d7e5b439242b"
)
EXPECTED_FIREWALL_MANIFEST_SHA256 = (
    "b711e9d3443f3a86c803ff1f8cae1ff266634439ebe0c0d31c04e807c3841837"
)
EXPECTED_VALIDATION_BANK_SHA256 = (
    "8397c886db73bb260723debd6d4eab332b6172972a055b28f770e7ca2bf3ee25"
)
EXPECTED_VALIDATION_R3_SHA256 = (
    "0c3887331553c474f5a39ab3653c8b0094ce7db8f416cff279627e001d68102c"
)
EXPECTED_VALIDATION_ROW_POSITION_SHA256 = (
    "964724eddd503d596ec870ceea3b0145f151ade64f5903307352297372763a75"
)
EXPECTED_VALIDATION_SOURCE_INDEX_SHA256 = (
    "96a367399d53a48078d7a3328a6063174c2f58031177da0f14953191620b5d80"
)
EXPECTED_PARTITION_LABEL_SHA256 = (
    "c8bacf260ae894e7da623586b9e04c67d95f628c16f52766cee0994ca2ceac19"
)
EXPECTED_MODEL_ORDER_SHA256 = (
    "aeb66fe308b42eb1c7ffa313790f762837df00f6255cefcec79a6210bce78f93"
)
EXPECTED_PRODUCER_STATE_SHA256 = (
    "cd70756674660470786b92f59d27414cbc379cb61b02ad255ed3d596c6362722"
)
EXPECTED_PRODUCER_MANIFEST_SHA256 = (
    "2f8766943eca79e37b4e72b4e1bff37cff7c8f9ce09469bce2daebbb481df373"
)
EXPECTED_PARTITION_IDENTITY_SHA256 = (
    "d1eaf777c27b98eb607e98ca41d2360115eb0b070156b29d21805c7332981d6d"
)

CLASS_COUNT = 10
DIMENSION = 100
RANK = 99
VALIDATION_COUNT = 500
ROWS_PER_CLASS = 50
LAMBDA_DENOMINATOR = 499
SAMPLE_SEED = 100001
MODEL_SAMPLE_SEED_A = 100002
MODEL_SAMPLE_SEED_B = 100003
SCORE_SAMPLE_SEED_A = 100004
SCORE_SAMPLE_SEED_B = 100005
TARGET_ACCEPTED = 500
MAX_PROPOSALS = 1_000_000
PROPOSAL_BATCH_SIZE = 4096
QUANTILE_PROBABILITIES = (0.05, 0.50, 0.95)
ORTHOGONALITY_RTOL = 1.0e-9
ORTHOGONALITY_ATOL = 1.0e-9
REPORT_NAME = "n3_amplitude_producer_validation_report.v1.json"
SAMPLE_BANK_NAME = "n3_amplitude_producer_validation_sample_banks.v1.pt"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _read_json_exact(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(file_sha256(path) == expected_sha256, f"{label} hash changed: {path}")
    return dict(_mapping(json.loads(path.read_text(encoding="utf-8")), label))


def _git_state(repo_root: Path) -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    dirty_paths = [line[3:] if len(line) >= 4 else line for line in status]
    return {"branch": branch, "head": head, "dirty_paths": dirty_paths}


def _summary(value: torch.Tensor) -> dict[str, float]:
    view = value.detach().to(torch.float64).cpu().reshape(-1)
    _require(view.numel() > 0 and bool(torch.isfinite(view).all()), "summary input is invalid")
    quantiles = torch.quantile(view, torch.tensor(QUANTILE_PROBABILITIES, dtype=torch.float64))
    return {
        "min": float(view.min()),
        "p05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "mean": float(view.mean()),
        "p95": float(quantiles[2]),
        "max": float(view.max()),
    }


def _quantiles(value: torch.Tensor) -> list[float]:
    view = value.detach().to(torch.float64).cpu().reshape(-1)
    _require(view.numel() > 0 and bool(torch.isfinite(view).all()), "quantile input is invalid")
    return [
        float(item)
        for item in torch.quantile(view, torch.tensor(QUANTILE_PROBABILITIES, dtype=torch.float64))
    ]


def _covariance_and_correlation(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    view = value.detach().to(torch.float64).cpu().contiguous()
    _require(view.ndim == 2 and view.shape[0] >= 2, "covariance input shape is invalid")
    centered = view - view.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1).mm(centered).div(float(view.shape[0] - 1))
    diagonal = torch.diagonal(covariance)
    _require(bool(torch.isfinite(covariance).all()) and bool(diagonal.gt(0.0).all()), "covariance is invalid")
    correlation = covariance / torch.sqrt(diagonal[:, None] * diagonal[None, :])
    _require(bool(torch.isfinite(correlation).all()), "correlation is invalid")
    return covariance.contiguous(), correlation.contiguous()


def _off_diagonal_summary(covariance: torch.Tensor, correlation: torch.Tensor) -> dict[str, float]:
    dimension = covariance.shape[0]
    mask = ~torch.eye(dimension, dtype=torch.bool)
    abs_corr = correlation.abs()[mask]
    offdiag_covariance = covariance[mask]
    total_energy = covariance.square().sum()
    offdiag_energy = offdiag_covariance.square().sum()
    _require(bool(total_energy.gt(0.0)), "covariance energy is zero")
    return {
        "mean_abs_cross_correlation": float(abs_corr.mean()),
        "median_abs_cross_correlation": float(torch.quantile(abs_corr, 0.50)),
        "p95_abs_cross_correlation": float(torch.quantile(abs_corr, 0.95)),
        "maximum_abs_cross_correlation": float(abs_corr.max()),
        "off_diagonal_covariance_fraction": float(offdiag_energy / total_energy),
    }


def _energy_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    left = first.detach().to(torch.float64).cpu().contiguous()
    right = second.detach().to(torch.float64).cpu().contiguous()
    _require(left.ndim == 2 and right.ndim == 2 and left.shape[1] == right.shape[1], "energy inputs are invalid")
    cross = torch.cdist(left, right, p=2.0).mean()
    within_left = torch.cdist(left, left, p=2.0).mean()
    within_right = torch.cdist(right, right, p=2.0).mean()
    value = 2.0 * cross - within_left - within_right
    _require(bool(torch.isfinite(value)), "energy distance is non-finite")
    _require(float(value) >= -1.0e-10, "energy distance is materially negative")
    return max(0.0, float(value))


def _score_descriptive(value: torch.Tensor) -> dict[str, Any]:
    view = value.detach().to(torch.float64).cpu().contiguous()
    covariance, correlation = _covariance_and_correlation(view)
    mean = view.mean(dim=0)
    variance = view.var(dim=0, unbiased=True)
    centered = view - mean
    standard_deviation = torch.sqrt(variance)
    skew = centered.pow(3).mean(dim=0) / standard_deviation.pow(3)
    excess_kurtosis = centered.pow(4).mean(dim=0) / variance.square() - 3.0
    radial = view.square().sum(dim=1)
    identity = torch.eye(view.shape[1], dtype=torch.float64)
    return {
        "count": int(view.shape[0]),
        "dimension": int(view.shape[1]),
        "mean_abs_summary": _summary(mean.abs()),
        "variance_summary": _summary(variance),
        "covariance_identity_frobenius": float((covariance - identity).norm()),
        "correlation_identity_frobenius": float((correlation - identity).norm()),
        "cross_correlation": _off_diagonal_summary(covariance, correlation),
        "skew_abs_summary": _summary(skew.abs()),
        "excess_kurtosis_abs_summary": _summary(excess_kurtosis.abs()),
        "radial_r2_summary": _summary(radial),
        "radial_r2_mean": float(radial.mean()),
        "radial_r2_variance": float(radial.var(unbiased=True)),
        "gaussian_radial_reference": {
            "degrees_of_freedom": int(view.shape[1]),
            "mean": float(view.shape[1]),
            "variance": float(2 * view.shape[1]),
        },
        "finite": True,
    }


def _score_pair_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    left = first.detach().to(torch.float64).cpu().contiguous()
    right = second.detach().to(torch.float64).cpu().contiguous()
    _require(left.shape == right.shape and left.ndim == 2, "score comparison shapes differ")
    left_cov, left_corr = _covariance_and_correlation(left)
    right_cov, right_corr = _covariance_and_correlation(right)
    left_mean = left.mean(dim=0)
    right_mean = right.mean(dim=0)
    left_variance = left.var(dim=0, unbiased=True)
    right_variance = right.var(dim=0, unbiased=True)
    left_radial = left.square().sum(dim=1)
    right_radial = right.square().sum(dim=1)
    left_skew = (left - left_mean).pow(3).mean(dim=0) / torch.sqrt(left_variance).pow(3)
    right_skew = (right - right_mean).pow(3).mean(dim=0) / torch.sqrt(right_variance).pow(3)
    return {
        "mean_l2": float((left_mean - right_mean).norm()),
        "variance_l2": float((left_variance - right_variance).norm()),
        "correlation_frobenius": float((left_corr - right_corr).norm()),
        "radial_mean_abs_delta": float((left_radial.mean() - right_radial.mean()).abs()),
        "radial_variance_abs_delta": float((left_radial.var(unbiased=True) - right_radial.var(unbiased=True)).abs()),
        "radial_quantile_l2": float(
            torch.tensor(_quantiles(left_radial), dtype=torch.float64).sub(
                torch.tensor(_quantiles(right_radial), dtype=torch.float64)
            ).norm()
        ),
        "mean_abs_skew_delta": float((left_skew.abs().mean() - right_skew.abs().mean()).abs()),
        "energy_distance": _energy_distance(left, right),
    }


def _amplitude_statistics(value: torch.Tensor) -> dict[str, Any]:
    view = value.detach().to(torch.float64).cpu().contiguous()
    covariance, correlation = _covariance_and_correlation(view)
    mean = view.mean(dim=0)
    standard_deviation = view.std(dim=0, unbiased=True)
    per_mode_quantiles = torch.quantile(
        view,
        torch.tensor(QUANTILE_PROBABILITIES, dtype=torch.float64),
        dim=0,
    ).transpose(0, 1).contiguous()
    norms = view.norm(dim=1)
    energies = view.square().sum(dim=1)
    return {
        "count": int(view.shape[0]),
        "dimension": int(view.shape[1]),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "per_mode_quantiles": per_mode_quantiles,
        "covariance": covariance,
        "correlation": correlation,
        "cross_correlation": _off_diagonal_summary(covariance, correlation),
        "norms": norms,
        "energies": energies,
        "mean_summary": _summary(mean),
        "standard_deviation_summary": _summary(standard_deviation),
        "norm_summary": _summary(norms),
        "energy_summary": _summary(energies),
        "finite": bool(torch.isfinite(view).all()),
        "nonnegative": bool(view.ge(0.0).all()),
    }


def _amplitude_pair_metrics(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, float]:
    mean_delta = first["mean"] - second["mean"]
    standard_deviation_delta = first["standard_deviation"] - second["standard_deviation"]
    quantile_delta = first["per_mode_quantiles"] - second["per_mode_quantiles"]
    covariance_first = first["covariance"]
    covariance_second = second["covariance"]
    correlation_first = first["correlation"]
    correlation_second = second["correlation"]
    trace_first = torch.trace(covariance_first)
    trace_second = torch.trace(covariance_second)
    shape_first = covariance_first / trace_first
    shape_second = covariance_second / trace_second
    return {
        "mean_l2": float(mean_delta.norm()),
        "standard_deviation_l2": float(standard_deviation_delta.norm()),
        "per_mode_quantile_l2": float(quantile_delta.norm()),
        "covariance_relative_frobenius": float((covariance_first - covariance_second).norm().square() / covariance_second.norm().square()),
        "correlation_frobenius": float((correlation_first - correlation_second).norm()),
        "off_diagonal_covariance_fraction_abs_delta": float(
            abs(
                first["cross_correlation"]["off_diagonal_covariance_fraction"]
                - second["cross_correlation"]["off_diagonal_covariance_fraction"]
            )
        ),
        "trace_normalized_covariance_shape_frobenius": float((shape_first - shape_second).norm()),
        "norm_quantile_l2": float(
            torch.tensor(_quantiles(first["norms"]), dtype=torch.float64).sub(
                torch.tensor(_quantiles(second["norms"]), dtype=torch.float64)
            ).norm()
        ),
        "energy_quantile_l2": float(
            torch.tensor(_quantiles(first["energies"]), dtype=torch.float64).sub(
                torch.tensor(_quantiles(second["energies"]), dtype=torch.float64)
            ).norm()
        ),
        "energy_distance": _energy_distance(first["_raw"], second["_raw"]),
    }


def _calibration(data_vs_model: Mapping[str, float], model_vs_model: Mapping[str, float]) -> dict[str, Any]:
    _require(set(data_vs_model) == set(model_vs_model), "calibration metric sets differ")
    checks = {
        name: float(data_vs_model[name]) <= float(model_vs_model[name])
        for name in data_vs_model
    }
    return {
        "rule": "each DATA_VS_MODEL discrepancy <= corresponding MODEL_VS_MODEL discrepancy",
        "data_vs_model": dict(data_vs_model),
        "model_vs_model": dict(model_vs_model),
        "component_checks": checks,
        "all_components_within_model_fluctuation": all(checks.values()),
    }


def _sample_positive(
    mu_fit: torch.Tensor,
    W_99: torch.Tensor,
    lambda_values: torch.Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    scale = torch.sqrt(lambda_values)
    accepted_q: list[torch.Tensor] = []
    accepted_z: list[torch.Tensor] = []
    accepted_count = 0
    proposals = 0
    termination = "proposal_budget_exhausted"
    while proposals < MAX_PROPOSALS and accepted_count < TARGET_ACCEPTED:
        block_count = min(PROPOSAL_BATCH_SIZE, MAX_PROPOSALS - proposals)
        z = torch.randn((block_count, RANK), generator=generator, dtype=torch.float64)
        q = mu_fit + (z * scale).mm(W_99.transpose(0, 1))
        valid = torch.isfinite(q).all(dim=1) & q.ge(0.0).all(dim=1)
        accepted_positions = valid.nonzero(as_tuple=False).reshape(-1)
        needed = TARGET_ACCEPTED - accepted_count
        if accepted_positions.numel() >= needed:
            selected = accepted_positions[:needed]
            accepted_q.append(q.index_select(0, selected).contiguous())
            accepted_z.append(z.index_select(0, selected).contiguous())
            proposals += int(selected[-1].item()) + 1
            accepted_count += needed
            termination = "target_accepted_reached"
            break
        if accepted_positions.numel() > 0:
            accepted_q.append(q.index_select(0, accepted_positions).contiguous())
            accepted_z.append(z.index_select(0, accepted_positions).contiguous())
            accepted_count += int(accepted_positions.numel())
        proposals += block_count
    q_result = torch.cat(accepted_q, dim=0) if accepted_q else torch.empty((0, DIMENSION), dtype=torch.float64)
    z_result = torch.cat(accepted_z, dim=0) if accepted_z else torch.empty((0, RANK), dtype=torch.float64)
    _require(q_result.shape == (accepted_count, DIMENSION), "accepted q shape changed")
    _require(z_result.shape == (accepted_count, RANK), "accepted z shape changed")
    _require(bool(torch.isfinite(q_result).all()) and bool(q_result.ge(0.0).all()), "accepted q violates positivity")
    return {
        "seed": seed,
        "rng": "torch.Generator(device='cpu').manual_seed(seed)",
        "proposal_batch_size": PROPOSAL_BATCH_SIZE,
        "target_accepted": TARGET_ACCEPTED,
        "max_proposals": MAX_PROPOSALS,
        "proposal_count": proposals,
        "accepted_count": accepted_count,
        "rejected_count": proposals - accepted_count,
        "acceptance_fraction": float(accepted_count / proposals) if proposals else 0.0,
        "termination": termination,
        "feasible": accepted_count == TARGET_ACCEPTED,
        "q": q_result,
        "z": z_result,
    }


def _sample_score_bank(*, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = torch.randn((TARGET_ACCEPTED, RANK), generator=generator, dtype=torch.float64)
    _require(bool(torch.isfinite(result).all()), "score sample bank is non-finite")
    return result.contiguous()


def _verify_authorities(repo_root: Path) -> dict[str, Any]:
    producer_manifest = _read_json_exact(
        repo_root / PRODUCER_MANIFEST,
        EXPECTED_PRODUCER_MANIFEST_SHA256,
        "producer state manifest",
    )
    _require(producer_manifest.get("schema") == "n3_amplitude_producer_state_manifest.v1", "producer manifest schema changed")
    _require(producer_manifest.get("classification") == "N3_AMPLITUDE_PRODUCER_BASELINE_FITTED", "producer classification changed")
    artifact = _mapping(producer_manifest.get("artifact"), "producer artifact")
    _require(artifact.get("sha256") == EXPECTED_PRODUCER_STATE_SHA256, "producer state binding changed")
    fit = _mapping(producer_manifest.get("fit"), "producer fit contract")
    _require(
        fit.get("count") == VALIDATION_COUNT
        and fit.get("dimension") == DIMENSION
        and fit.get("dtype") == "torch.float64"
        and fit.get("parameter_source") == "fit bank only"
        and fit.get("validation_parameter_leakage") is False,
        "producer fit contract changed",
    )
    producer_authorities = _mapping(producer_manifest.get("authorities"), "producer authorities")
    _require(producer_authorities.get("producer_lineage") == "OLS_LATENT_PRODUCER_V1", "producer lineage binding changed")
    _require(_mapping(producer_authorities.get("source_manifest"), "producer source authority").get("sha256") == EXPECTED_SOURCE_MANIFEST_SHA256, "producer source binding changed")
    _require(_mapping(producer_authorities.get("partition_manifest"), "producer partition authority").get("sha256") == EXPECTED_PARTITION_MANIFEST_SHA256, "producer partition binding changed")
    _require(_mapping(producer_authorities.get("firewall_manifest"), "producer firewall authority").get("sha256") == EXPECTED_FIREWALL_MANIFEST_SHA256, "producer firewall binding changed")
    producer_state = safe_torch_load(
        repo_root / PRODUCER_STATE,
        expected_sha256=EXPECTED_PRODUCER_STATE_SHA256,
    )
    state = _mapping(producer_state, "producer state")
    _require(state.get("schema") == "n3_amplitude_producer_state.v1", "producer state schema changed")
    _require(state.get("coordinate") == "R2_AMPLITUDE", "producer coordinate changed")
    _require(state.get("r") == RANK and state.get("d") == DIMENSION and state.get("n_fit") == 500, "producer dimensions changed")
    _require(state.get("score_model") == "DIAGONAL_GAUSSIAN_LOW_RANK_SCORE_BASELINE_V1", "score model changed")
    _require(state.get("lambda_convention") == "lambda_k=sigma_k^2/(n_fit-1)=sigma_k^2/499", "lambda convention changed")
    _require(state.get("validation_parameter_leakage") is False, "producer validation leakage flag changed")
    _require(state.get("rng") == "NONE", "producer fitting RNG contract changed")
    mu_fit = state.get("mu_fit")
    W_99 = state.get("W_99")
    lambda_values = state.get("lambda_1_to_99")
    singular_values = state.get("singular_values")
    for value, shape, name in (
        (mu_fit, (DIMENSION,), "mu_fit"),
        (W_99, (DIMENSION, RANK), "W_99"),
        (lambda_values, (RANK,), "lambda"),
        (singular_values, (DIMENSION,), "singular_values"),
    ):
        _require(torch.is_tensor(value) and tuple(value.shape) == shape and value.dtype == torch.float64, f"{name} state changed")
        _require(bool(torch.isfinite(value).all()), f"{name} state is non-finite")
    _require(bool(mu_fit.ge(0.0).all()), "mu_fit is negative")
    _require(bool(lambda_values.gt(0.0).all()), "lambda is not positive")
    _require(bool(torch.allclose(W_99.T.mm(W_99), torch.eye(RANK, dtype=torch.float64), rtol=ORTHOGONALITY_RTOL, atol=ORTHOGONALITY_ATOL)), "W_99 is not orthonormal")
    _require(tensor_sha256(mu_fit) == producer_manifest["tensor_identities"]["mu_fit_sha256"], "mu identity changed")
    _require(tensor_sha256(W_99) == producer_manifest["tensor_identities"]["W_99_sha256"], "W identity changed")
    _require(tensor_sha256(lambda_values) == producer_manifest["tensor_identities"]["lambda_1_to_99_sha256"], "lambda identity changed")

    model_order = _read_json_exact(
        repo_root / MODEL_ORDER_RESULT,
        EXPECTED_MODEL_ORDER_SHA256,
        "model-order result",
    )
    _require(model_order.get("classification") == "N3_LOW_RANK_MODEL_ORDER_SELECTED", "model-order classification changed")
    _require(model_order.get("selection", {}).get("r_selected") == RANK, "model-order rank changed")
    _require(model_order.get("semantic_inputs", {}).get("coordinate") == "R2_AMPLITUDE", "model-order coordinate changed")

    source = _read_json_exact(repo_root / SOURCE_MANIFEST, EXPECTED_SOURCE_MANIFEST_SHA256, "source manifest")
    partition = _read_json_exact(repo_root / PARTITION_MANIFEST, EXPECTED_PARTITION_MANIFEST_SHA256, "partition manifest")
    firewall = _read_json_exact(repo_root / FIREWALL_MANIFEST, EXPECTED_FIREWALL_MANIFEST_SHA256, "firewall manifest")
    _require(source.get("schema") == "ols_producer_source_manifest.v1", "source schema changed")
    _require(source.get("producer_lineage") == "OLS_LATENT_PRODUCER_V1", "source lineage changed")
    _require(source.get("source_coordinate") == "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX", "source coordinate changed")
    source_artifact = _mapping(source.get("source_artifact"), "source artifact")
    _require(
        source_artifact.get("sha256") == EXPECTED_SOURCE_ARTIFACT_SHA256
        and source_artifact.get("semantic_manifest_sha256") == EXPECTED_SOURCE_SEMANTIC_MANIFEST_SHA256
        and source_artifact.get("index_sha256") == EXPECTED_SOURCE_INDEX_SHA256
        and source_artifact.get("label_sha256") == EXPECTED_SOURCE_LABEL_SHA256
        and source_artifact.get("r3_complex_sha256") == EXPECTED_SOURCE_R3_SHA256
        and source_artifact.get("dataset") == "mnist_train"
        and source_artifact.get("source") == "mnist_train_indices_0_54999"
        and source_artifact.get("count") == 1000
        and source_artifact.get("per_class") == 100
        and source_artifact.get("shape") == [1000, DIMENSION]
        and source_artifact.get("dtype") == "torch.complex64",
        "source artifact contract changed",
    )
    _require(partition.get("schema") == "ols_producer_partition_manifest.v1", "partition schema changed")
    _require(partition.get("producer_lineage") == "OLS_LATENT_PRODUCER_V1", "partition lineage changed")
    _require(partition.get("source_artifact_sha256") == EXPECTED_SOURCE_ARTIFACT_SHA256, "partition source binding changed")
    _require(partition.get("source_coordinate") == "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX", "partition coordinate changed")
    _require(partition.get("algorithm") == "label_major; within_class_source_index_ascending; first_50_fit_last_50_validation", "partition algorithm changed")
    _require(partition.get("rng") == "none", "partition RNG changed")
    _require(partition.get("runtime_resplit") is False and partition.get("runtime_reselection") is False, "partition runtime mutation changed")
    _require(partition.get("fit_count_per_class") == ROWS_PER_CLASS and partition.get("validation_count_per_class") == ROWS_PER_CLASS, "partition class balance changed")
    overlap = _mapping(partition.get("overlap"), "partition overlap")
    _require(
        overlap.get("fit_validation_row_position_count") == 0
        and overlap.get("fit_validation_source_index_count") == 0
        and overlap.get("union_source_row_count") == 1000,
        "fit/validation partition overlap changed",
    )
    _require(partition.get("partition_identity", {}).get("semantic_identity_sha256") == EXPECTED_PARTITION_IDENTITY_SHA256, "partition identity changed")
    _require(firewall.get("schema") == "ols_producer_data_firewall_manifest.v1" and firewall.get("firewall_pass") is True, "firewall is not passing")
    _require(firewall.get("unregistered_external_candidates_excluded") is True, "unregistered external candidates are not excluded")
    future_consumer = _mapping(firewall.get("future_consumer"), "firewall future consumer")
    _require(
        future_consumer.get("must_verify_target_bank_identity") is True
        and future_consumer.get("runtime_membership_mutation") is False
        and future_consumer.get("runtime_reselection") is False
        and future_consumer.get("runtime_resplit") is False,
        "firewall consumer contract changed",
    )
    protected = {
        item.get("canonical_name"): item
        for item in firewall.get("protected_identities", [])
        if isinstance(item, Mapping)
    }
    for name in ("S1_SELECTION", "S1_CONFIRMATION", "S2_VALIDATION", "FORMAL_S3_PROTECTED_PANEL"):
        entry = _mapping(protected.get(name), f"protected identity {name}")
        _require(entry.get("status") == "PASS_ZERO_OVERLAP", f"protected identity {name} is not isolated")
        _require(entry.get("fit_overlap_count") == 0, f"fit bank overlaps {name}")
        _require(entry.get("validation_overlap_count") == 0, f"validation bank overlaps {name}")

    return {
        "producer_manifest": {
            "path": _relative(repo_root, repo_root / PRODUCER_MANIFEST),
            "sha256": EXPECTED_PRODUCER_MANIFEST_SHA256,
            "state_sha256": EXPECTED_PRODUCER_STATE_SHA256,
            "schema": producer_manifest["schema"],
        },
        "model_order": {
            "path": _relative(repo_root, repo_root / MODEL_ORDER_RESULT),
            "sha256": EXPECTED_MODEL_ORDER_SHA256,
            "schema": model_order["schema"],
            "r": RANK,
        },
        "source_manifest": {"path": _relative(repo_root, repo_root / SOURCE_MANIFEST), "sha256": EXPECTED_SOURCE_MANIFEST_SHA256},
        "partition_manifest": {"path": _relative(repo_root, repo_root / PARTITION_MANIFEST), "sha256": EXPECTED_PARTITION_MANIFEST_SHA256, "semantic_identity_sha256": EXPECTED_PARTITION_IDENTITY_SHA256},
        "firewall_manifest": {"path": _relative(repo_root, repo_root / FIREWALL_MANIFEST), "sha256": EXPECTED_FIREWALL_MANIFEST_SHA256, "firewall_pass": True},
        "coordinate": "R2_AMPLITUDE",
        "coordinate_definition": "q=abs(c/sqrt(s))=abs(R3_COMPLEX)",
        "mu_fit": mu_fit,
        "W_99": W_99,
        "lambda": lambda_values,
    }


def _load_validation_bank(repo_root: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = safe_torch_load(repo_root / VALIDATION_BANK, expected_sha256=EXPECTED_VALIDATION_BANK_SHA256)
    bank = _mapping(payload, "internal-validation bank")
    _require(bank.get("schema") == "ols_producer_bank.v1", "validation bank schema changed")
    _require(bank.get("role") == "OLS_PRODUCER_INTERNAL_VALIDATION_BANK_V1", "validation bank role changed")
    _require(bank.get("count") == VALIDATION_COUNT and bank.get("N") == DIMENSION, "validation bank shape changed")
    _require(bank.get("dtype") == "torch.complex64" and bank.get("shape") == [VALIDATION_COUNT, DIMENSION], "validation bank dtype changed")
    _require(bank.get("producer_lineage") == "OLS_LATENT_PRODUCER_V1", "validation lineage changed")
    _require(bank.get("source_coordinate") == "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX", "validation coordinate changed")
    _require(_mapping(bank.get("partition"), "validation partition").get("semantic_identity_sha256") == EXPECTED_PARTITION_IDENTITY_SHA256, "validation partition changed")
    r3 = bank.get("r3_complex")
    indices = bank.get("source_indices")
    positions = bank.get("source_row_positions")
    labels = bank.get("labels")
    for value, shape, dtype, name in (
        (r3, (VALIDATION_COUNT, DIMENSION), torch.complex64, "R3"),
        (indices, (VALIDATION_COUNT,), torch.int64, "source indices"),
        (positions, (VALIDATION_COUNT,), torch.int64, "row positions"),
        (labels, (VALIDATION_COUNT,), torch.int64, "labels"),
    ):
        _require(torch.is_tensor(value) and tuple(value.shape) == shape and value.dtype == dtype, f"validation {name} changed")
    _require(bool(torch.isfinite(r3).all()), "validation R3 is non-finite")
    _require(tensor_sha256(r3) == EXPECTED_VALIDATION_R3_SHA256, "validation R3 identity changed")
    _require(tensor_sha256(indices) == EXPECTED_VALIDATION_SOURCE_INDEX_SHA256, "validation source-index identity changed")
    _require(tensor_sha256(positions) == EXPECTED_VALIDATION_ROW_POSITION_SHA256, "validation row-position identity changed")
    _require(tensor_sha256(labels) == EXPECTED_PARTITION_LABEL_SHA256, "validation label identity changed")
    _require(torch.equal(torch.bincount(labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), ROWS_PER_CLASS, dtype=torch.int64)), "validation class balance changed")
    q = r3.abs().to(torch.float64).contiguous()
    _require(bool(torch.isfinite(q).all()) and bool(q.ge(0.0).all()), "validation q is invalid")
    return q, {
        "path": _relative(repo_root, repo_root / VALIDATION_BANK),
        "sha256": EXPECTED_VALIDATION_BANK_SHA256,
        "schema": bank["schema"],
        "count": VALIDATION_COUNT,
        "dimension": DIMENSION,
        "per_class": ROWS_PER_CLASS,
        "r3_sha256": EXPECTED_VALIDATION_R3_SHA256,
        "source_index_sha256": EXPECTED_VALIDATION_SOURCE_INDEX_SHA256,
        "row_position_sha256": EXPECTED_VALIDATION_ROW_POSITION_SHA256,
        "label_sha256": EXPECTED_PARTITION_LABEL_SHA256,
    }


def _strip_amplitude_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "count": stats["count"],
        "dimension": stats["dimension"],
        "mean_summary": stats["mean_summary"],
        "standard_deviation_summary": stats["standard_deviation_summary"],
        "norm_summary": stats["norm_summary"],
        "energy_summary": stats["energy_summary"],
        "finite": stats["finite"],
        "nonnegative": stats["nonnegative"],
    }


def _strip_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in sample.items()
        if key not in {"q", "z"}
    }


def _write_run(
    repo_root: Path,
    output_dir: Path,
    sample_payload: Mapping[str, Any],
    report: Mapping[str, Any],
) -> tuple[Path, Path]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once validation output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        sample_path = temporary / SAMPLE_BANK_NAME
        report_path = temporary / REPORT_NAME
        atomic_torch_save_new(sample_path, sample_payload, artifact_kind="N3 validation sample banks")
        payload = dict(report)
        payload["sample_bank_artifact"] = {
            "path": _relative(repo_root, output_dir / SAMPLE_BANK_NAME),
            "sha256": file_sha256(sample_path),
            "schema": SAMPLE_BANK_SCHEMA,
        }
        atomic_write_new_json(report_path, payload, artifact_kind="N3 validation report")
        _require(sorted(path.name for path in temporary.iterdir()) == sorted((REPORT_NAME, SAMPLE_BANK_NAME)), "validation output layout changed")
        if output_dir.exists():
            raise FileExistsError(f"write-once validation output appeared: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / SAMPLE_BANK_NAME, output_dir / REPORT_NAME


def execute(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 authority branch")
    _require(not git["dirty_paths"], f"worktree must be clean: {git['dirty_paths']}")
    producer_state_sha_before = file_sha256(repo_root / PRODUCER_STATE)
    validation_bank_sha_before = file_sha256(repo_root / VALIDATION_BANK)
    authorities = _verify_authorities(repo_root)
    q_validation, validation_identity = _load_validation_bank(repo_root)
    mu_fit = authorities.pop("mu_fit")
    W_99 = authorities.pop("W_99")
    lambda_values = authorities.pop("lambda")
    h_validation = (q_validation - mu_fit).mm(W_99)
    z_validation = h_validation / torch.sqrt(lambda_values)
    _require(bool(torch.isfinite(h_validation).all()) and bool(torch.isfinite(z_validation).all()), "held-out scores are non-finite")
    score_descriptive = _score_descriptive(z_validation)

    positivity = _sample_positive(mu_fit, W_99, lambda_values, seed=SAMPLE_SEED)
    model_vs_model = None
    amplitude_support = {
        "status": "NOT_EVALUATED_POSITIVITY_INFEASIBLE",
        "reason": "500 accepted states were not obtained within the frozen proposal budget",
    }
    gaussian_status = "DESCRIPTIVE_ONLY_NO_MODEL_CALIBRATION"
    score_status = "DESCRIPTIVE_ONLY_NO_MODEL_CALIBRATION"
    sample_payload: dict[str, Any] = {
        "schema": SAMPLE_BANK_SCHEMA,
        "version": 1,
        "positivity_sample": {"seed": SAMPLE_SEED, "q": positivity["q"], "z": positivity["z"]},
        "model_sample_a": None,
        "model_sample_b": None,
        "score_sample_a": None,
        "score_sample_b": None,
    }
    amplitude_data_stats = None
    if positivity["feasible"]:
        model_a = _sample_positive(mu_fit, W_99, lambda_values, seed=MODEL_SAMPLE_SEED_A)
        model_b = _sample_positive(mu_fit, W_99, lambda_values, seed=MODEL_SAMPLE_SEED_B)
        _require(model_a["feasible"] and model_b["feasible"], "model calibration sample bank is infeasible")
        score_a = _sample_score_bank(seed=SCORE_SAMPLE_SEED_A)
        score_b = _sample_score_bank(seed=SCORE_SAMPLE_SEED_B)
        q_model_a = model_a["q"]
        q_model_b = model_b["q"]
        z_model_a = score_a
        z_model_b = score_b
        score_data_vs_model = _score_pair_metrics(z_validation, z_model_a)
        score_model_vs_model = _score_pair_metrics(z_model_a, z_model_b)
        score_calibration = _calibration(score_data_vs_model, score_model_vs_model)
        score_status = "CALIBRATED_COMPATIBLE" if score_calibration["all_components_within_model_fluctuation"] else "CALIBRATED_MISMATCH"
        gaussian_status = score_status
        amplitude_data_stats = _amplitude_statistics(q_validation)
        amplitude_model_a_stats = _amplitude_statistics(q_model_a)
        amplitude_model_b_stats = _amplitude_statistics(q_model_b)
        amplitude_data_stats["_raw"] = q_validation
        amplitude_model_a_stats["_raw"] = q_model_a
        amplitude_model_b_stats["_raw"] = q_model_b
        amplitude_data_vs_model = _amplitude_pair_metrics(amplitude_data_stats, amplitude_model_a_stats)
        amplitude_model_vs_model = _amplitude_pair_metrics(amplitude_model_a_stats, amplitude_model_b_stats)
        amplitude_calibration = _calibration(amplitude_data_vs_model, amplitude_model_vs_model)
        amplitude_support = {
            "status": "CALIBRATED_COMPATIBLE" if amplitude_calibration["all_components_within_model_fluctuation"] else "CALIBRATED_MISMATCH",
            "data": _strip_amplitude_stats(amplitude_data_stats),
            "model_a": _strip_amplitude_stats(amplitude_model_a_stats),
            "model_b": _strip_amplitude_stats(amplitude_model_b_stats),
            "marginal": {
                "mean_abs_delta_summary": _summary((amplitude_data_stats["mean"] - amplitude_model_a_stats["mean"]).abs()),
                "standard_deviation_abs_delta_summary": _summary((amplitude_data_stats["standard_deviation"] - amplitude_model_a_stats["standard_deviation"]).abs()),
                "per_mode_quantile_abs_delta_summary": _summary((amplitude_data_stats["per_mode_quantiles"] - amplitude_model_a_stats["per_mode_quantiles"]).abs()),
            },
            "joint": {
                "data_vs_model": {key: value for key, value in amplitude_data_vs_model.items() if key in {"covariance_relative_frobenius", "correlation_frobenius", "trace_normalized_covariance_shape_frobenius"}},
                "model_vs_model": {key: value for key, value in amplitude_model_vs_model.items() if key in {"covariance_relative_frobenius", "correlation_frobenius", "trace_normalized_covariance_shape_frobenius"}},
            },
            "norm_energy": {
                "data_norm_quantiles": _quantiles(amplitude_data_stats["norms"]),
                "model_norm_quantiles": _quantiles(amplitude_model_a_stats["norms"]),
                "data_energy_quantiles": _quantiles(amplitude_data_stats["energies"]),
                "model_energy_quantiles": _quantiles(amplitude_model_a_stats["energies"]),
                "data_vs_model_norm_energy_metrics": {key: value for key, value in amplitude_data_vs_model.items() if key in {"norm_quantile_l2", "energy_quantile_l2"}},
                "model_vs_model_norm_energy_metrics": {key: value for key, value in amplitude_model_vs_model.items() if key in {"norm_quantile_l2", "energy_quantile_l2"}},
            },
            "global_q_space": {
                "metric": "biased_empirical_energy_distance_euclidean_q_space",
                "data_vs_model": amplitude_data_vs_model["energy_distance"],
                "model_vs_model": amplitude_model_vs_model["energy_distance"],
            },
            "calibration": amplitude_calibration,
        }
        model_vs_model = {
            "positivity_model_a": _strip_sample(model_a),
            "positivity_model_b": _strip_sample(model_b),
            "score_model_a_seed": SCORE_SAMPLE_SEED_A,
            "score_model_b_seed": SCORE_SAMPLE_SEED_B,
            "score_data_vs_model": score_data_vs_model,
            "score_model_vs_model": score_model_vs_model,
            "score_calibration": score_calibration,
        }
        sample_payload["model_sample_a"] = {"seed": MODEL_SAMPLE_SEED_A, "q": q_model_a, "z": model_a["z"]}
        sample_payload["model_sample_b"] = {"seed": MODEL_SAMPLE_SEED_B, "q": q_model_b, "z": model_b["z"]}
        sample_payload["score_sample_a"] = {"seed": SCORE_SAMPLE_SEED_A, "z": z_model_a}
        sample_payload["score_sample_b"] = {"seed": SCORE_SAMPLE_SEED_B, "z": z_model_b}

    q_reconstructed = mu_fit + h_validation.mm(W_99.transpose(0, 1))
    residual = q_validation - q_reconstructed
    residual_norm = residual.norm(dim=1)
    residual_energy = residual.square().sum(dim=1)
    _require(bool(torch.isfinite(residual).all()), "rank-99 residual is non-finite")
    producer_state_sha_after = file_sha256(repo_root / PRODUCER_STATE)
    validation_bank_sha_after = file_sha256(repo_root / VALIDATION_BANK)
    _require(
        producer_state_sha_before == producer_state_sha_after == EXPECTED_PRODUCER_STATE_SHA256,
        "producer state changed during validation",
    )
    _require(
        validation_bank_sha_before == validation_bank_sha_after == EXPECTED_VALIDATION_BANK_SHA256,
        "validation bank changed during validation",
    )

    positivity_status = "SUPPORTED" if positivity["feasible"] else "NOT_SUPPORTED"
    axis_statuses = {
        "SCORE_STRUCTURE": score_status,
        "GAUSSIAN_BASELINE": gaussian_status,
        "POSITIVITY_FEASIBILITY": positivity_status,
        "AMPLITUDE_SUPPORT": amplitude_support["status"],
    }
    if positivity_status == "NOT_SUPPORTED" or amplitude_support["status"] == "CALIBRATED_MISMATCH" or score_status == "CALIBRATED_MISMATCH":
        classification = CLASSIFICATION_NOT_SUPPORTED
    elif score_status == "DESCRIPTIVE_ONLY_NO_MODEL_CALIBRATION" or amplitude_support["status"] == "NOT_EVALUATED_POSITIVITY_INFEASIBLE":
        classification = CLASSIFICATION_MIXED
    else:
        classification = CLASSIFICATION_SUPPORTED

    state_path = repo_root / PRODUCER_STATE
    manifest = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "classification": classification,
        "axis_classifications": axis_statuses,
        "semantic_inputs": {
            "producer_data": "OLS_PRODUCER_DATA_V1",
            "model_order": "N3_LOW_RANK_MODEL_ORDER_SELECTED",
            "producer_state": "N3_AMPLITUDE_PRODUCER_BASELINE_FITTED",
            "coordinate": "R2_AMPLITUDE",
            "score_model": "DIAGONAL_GAUSSIAN_LOW_RANK_SCORE_BASELINE_V1",
            "positivity_contract": "finite q>=0 elementwise; no clamp/abs/ReLU/softplus/normalization",
            "authorities": authorities,
        },
        "frozen_producer": {
            "state_path": _relative(repo_root, state_path),
            "state_sha256": EXPECTED_PRODUCER_STATE_SHA256,
            "r": RANK,
            "d": DIMENSION,
            "n_fit": 500,
            "lambda_denominator": LAMBDA_DENOMINATOR,
            "dtype": "torch.float64",
            "mu_fit_sha256": tensor_sha256(mu_fit),
            "W_99_sha256": tensor_sha256(W_99),
            "lambda_sha256": tensor_sha256(lambda_values),
        },
        "validation_bank": validation_identity,
        "execution": {
            "device": "cpu",
            "dtype": "torch.float64",
            "rng_algorithm": "torch.Generator CPU",
            "no_parameter_refit": True,
            "no_rank_reselection": True,
            "proposal_order": "sequential generator order; fixed batch evaluation preserves row order",
            "proposal_batch_size": PROPOSAL_BATCH_SIZE,
        },
        "held_out_score_structure": {
            "definition": "h_val=(q_val-mu_fit)W_99; z_val=h_val/sqrt(lambda)",
            "covariance_denominator": "N-1=499",
            "descriptive": score_descriptive,
            "target_compatibility": "standard_normal_moment_reference_only",
        },
        "gaussian_baseline": {
            "status": gaussian_status,
            "interpretation": "finite-sample compatibility diagnostic; not Gaussianity scientific support",
            "score_model": "independent standard-normal score draws with fixed CPU seeds when positivity is feasible",
        },
        "positivity_sampling": {
            key: value
            for key, value in positivity.items()
            if key not in {"q", "z"}
        },
        "positivity_feasibility": {
            "status": positivity_status,
            "rule": "accept iff q is finite and every q_j>=0; target 500 within 1000000 proposals",
            "accepted_q_sha256": tensor_sha256(positivity["q"]),
            "accepted_z_sha256": tensor_sha256(positivity["z"]),
        },
        "amplitude_support": amplitude_support,
        "model_vs_model_calibration": model_vs_model,
        "rank_99_residual": {
            "definition": "q_val - (mu_fit + W_99 W_99^T (q_val-mu_fit))",
            "norm_summary": _summary(residual_norm),
            "energy_summary": _summary(residual_energy),
            "omitted_ambient_directions": DIMENSION - RANK,
            "model_order_reopened": False,
        },
        "integrity_checks": {
            "producer_state_verified": True,
            "validation_bank_verified": True,
            "validation_firewall_verified": True,
            "held_out_scores_finite": True,
            "positivity_rule_exact": True,
            "accepted_q_finite_nonnegative": bool(torch.isfinite(positivity["q"]).all()) and bool(positivity["q"].ge(0.0).all()),
            "rank_99_residual_finite": True,
            "input_state_sha256_unchanged": producer_state_sha_before == producer_state_sha_after,
            "validation_bank_sha256_unchanged": validation_bank_sha_before == validation_bank_sha_after,
        },
        "git": {
            "branch": git["branch"],
            "head": git["head"],
            "dirty_paths": git["dirty_paths"],
            "consumer_script": _relative(repo_root, Path(__file__).resolve()),
            "consumer_script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "not_executed": [
            "rank_reselection",
            "producer_refit",
            "validation_parameter_update",
            "phase_source_selection",
            "novel_complex_OLS",
            "optical_generator",
            "evaluator",
            "CUDA",
            "optimizer",
            "backward",
            "training",
            "image_inspection",
            "legacy_8D_BOGT_S3_Code8_reopening",
        ],
    }
    sample_path, report_path = _write_run(repo_root, output_dir, sample_payload, manifest)
    return {
        "status": manifest["status"],
        "classification": classification,
        "axis_classifications": axis_statuses,
        "proposal_count": positivity["proposal_count"],
        "accepted_count": positivity["accepted_count"],
        "sample_bank": _relative(repo_root, sample_path),
        "report": _relative(repo_root, report_path),
        "sample_bank_sha256": file_sha256(sample_path),
        "report_sha256": file_sha256(report_path),
        "git_head": git["head"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/20260908_v2/runs/"
            "lightgen_n3_amplitude_producer_validation_v1_attempt_01"
        ),
        help="fresh write-once validation output directory",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
