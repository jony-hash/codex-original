"""Calibrate and independently confirm the N3 attribution selector.

This is the final synthetic-only selector-calibration consumer in the N3
lineage.  It never loads project OLS values.  Development data are used to
derive a frozen, interpretable selector specification; confirmation data use a
disjoint seed namespace and are consumed exactly once after that specification
has been materialized.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256


TASK_ID = "CALIBRATE_AND_CONFIRM_N3_ATTRIBUTION_SELECTOR_V2"
GATE = TASK_ID
REPORT_SCHEMA = "n3_attribution_selector_v2_calibration_report.v1"
SPEC_SCHEMA = "n3_attribution_selector_v2_frozen_spec.v1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
SCRIPT_PATH = Path("scripts/research/calibrate_and_confirm_n3_attribution_selector_v2.py")
OUTPUT_DIR = Path(
    "outputs/20260908_v2/runs/"
    "lightgen_n3_attribution_selector_v2_calibration_attempt_01"
)
SPEC_NAME = "n3_attribution_selector_v2_frozen_spec.v1.json"
REPORT_NAME = "n3_attribution_selector_v2_calibration_report.v1.json"

DIMENSION = 100
RANK = 99
FIT_COUNT = 500
VALIDATION_COUNT = 500
GROUP_COUNT = 10
ROWS_PER_GROUP = 50
REFERENCE_QUANTILE = 0.99
CONFIDENCE_ALPHA = 0.05
NULL_MAX_FALSE_POSITIVE_RATE = 0.05
NULL_MAX_FALSE_POSITIVE_COUNT = 10
MECHANISM_MIN_CORRECT_RATE = 0.80
MECHANISM_MAX_WRONG_RATE = 0.10
MECHANISM_MIN_CORRECT_LOWER_BOUND = 0.60
MECHANISM_MAX_WRONG_UPPER_BOUND = 0.20
NONLINEAR_PAIR_COUNT = 16
DEPENDENCE_TOPK = 20
RADIAL_SQUARE_CORRELATION_EFFECT_FLOOR = 0.10

MECHANISMS = (
    "NULL_STATIONARY_GAUSSIAN",
    "PARTITION_OR_MOMENT_DRIFT",
    "RADIAL_SCALE_MIXTURE",
    "LOCAL_MIXTURE",
    "NONLINEAR_DEPENDENCE",
)
NON_NULL_MECHANISMS = MECHANISMS[1:]
UNRESOLVED = "UNRESOLVED"

# The seed ranges are part of the authority.  No seed from a confirmation
# namespace is read during development or selector-spec construction.
DEVELOPMENT_SEEDS = {
    "NULL_CALIBRATION": tuple(range(110000, 110200)),
    "NULL_EVALUATION": tuple(range(120000, 120200)),
    "PARTITION_OR_MOMENT_DRIFT": tuple(range(130000, 130100)),
    "RADIAL_SCALE_MIXTURE": tuple(range(140000, 140100)),
    "LOCAL_MIXTURE": tuple(range(150000, 150100)),
    "NONLINEAR_DEPENDENCE": tuple(range(160000, 160100)),
}
CONFIRMATION_SEEDS = {
    "NULL_STATIONARY_GAUSSIAN": tuple(range(210000, 210200)),
    "PARTITION_OR_MOMENT_DRIFT": tuple(range(220000, 220100)),
    "RADIAL_SCALE_MIXTURE": tuple(range(230000, 230100)),
    "LOCAL_MIXTURE": tuple(range(240000, 240100)),
    "NONLINEAR_DEPENDENCE": tuple(range(250000, 250100)),
}

# All effects are synthetic-development design choices.  They are not
# estimated from, or copied from, the real N3 values.
MECHANISM_DEFINITIONS = {
    "NULL_STATIONARY_GAUSSIAN": {
        "population": "independent standard multivariate Gaussian",
        "role": "finite-sample null and false-positive control",
    },
    "PARTITION_OR_MOMENT_DRIFT": {
        "fit_population": "standard multivariate Gaussian",
        "validation_population": "1.35 * standard Gaussian + 0.45 in every source coordinate",
        "role": "distributed mean and scale shift; not a single-coordinate anomaly",
    },
    "RADIAL_SCALE_MIXTURE": {
        "base_population": "standard multivariate Gaussian",
        "row_scale": "0.55 with probability 0.80; sqrt(3.80) with probability 0.20",
        "second_moment_constraint": "E[scale^2]=1, so covariance inflation is not the defining effect",
        "common_scale_marker": "squared-coordinate correlation median has a registered strong-effect floor of 0.10",
        "role": "stationary elliptical common radial scale / heavy tail",
    },
    "LOCAL_MIXTURE": {
        "observer_groups": "10 groups, 50 rows/group, interleaved by row position",
        "component_means": "paired-coordinate amplitudes linearly spaced from 1.0 to 6.0",
        "component_covariance": "identity within component",
        "role": "reproducible local geometry not reducible to pooled moments",
    },
    "NONLINEAR_DEPENDENCE": {
        "construction": "16 paired coordinates with distinct source variance scales and right coordinate = normalized second Hermite polynomial of the left latent",
        "source_scales": "linearly spaced from 0.50 to 3.00 before fit-PCA estimation",
        "role": "stationary non-Gaussian joint dependence without common radial scaling or observer groups",
    },
}

SELECTED = {
    "PARTITION_OR_MOMENT_DRIFT": "PARTITION_OR_MOMENT_DRIFT",
    "RADIAL_SCALE_MIXTURE": "RADIAL_SCALE_MIXTURE",
    "LOCAL_MIXTURE": "LOCAL_MIXTURE",
    "NONLINEAR_DEPENDENCE": "NONLINEAR_DEPENDENCE",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    result = float(value)
    _require(math.isfinite(result), f"non-finite value: {value}")
    return result


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _normal(generator: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float64)


def _uniform(generator: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.rand(shape, generator=generator, dtype=torch.float64)


def _observer_groups(count: int) -> torch.Tensor:
    # Interleaving prevents the local-mixture observer from being identical to
    # a source-position trend while retaining 50 rows per synthetic group.
    return torch.arange(count, dtype=torch.long) % GROUP_COUNT


def _local_means() -> torch.Tensor:
    means = torch.zeros((GROUP_COUNT, DIMENSION), dtype=torch.float64)
    amplitudes = torch.linspace(1.0, 6.0, GROUP_COUNT, dtype=torch.float64)
    for group in range(GROUP_COUNT):
        means[group, group] = amplitudes[group]
        means[group, GROUP_COUNT + group] = 0.50 * amplitudes[group]
    return means


def _nonlinear_source_scales() -> torch.Tensor:
    return torch.linspace(0.50, 3.00, DIMENSION, dtype=torch.float64)


def _draw_bank(
    mechanism: str,
    count: int,
    generator: torch.Generator,
    bank_role: str,
    groups: torch.Tensor,
) -> torch.Tensor:
    x = _normal(generator, (count, DIMENSION))
    if mechanism == "PARTITION_OR_MOMENT_DRIFT":
        if bank_role == "validation":
            x = 1.35 * x + 0.45
        return x
    if mechanism == "RADIAL_SCALE_MIXTURE":
        selector = _uniform(generator, (count,))
        high_scale = math.sqrt(3.80)
        scale = torch.where(
            selector < 0.80,
            torch.full((count,), 0.55, dtype=torch.float64),
            torch.full((count,), high_scale, dtype=torch.float64),
        )
        return x * scale[:, None]
    if mechanism == "LOCAL_MIXTURE":
        return x + _local_means()[groups]
    if mechanism == "NONLINEAR_DEPENDENCE":
        # The second coordinate is H2(a)=(a^2-1)/sqrt(2).  Its covariance with
        # a is zero, while its off-diagonal third-moment contraction with a is
        # nonzero. Distinct source scales prevent finite-sample PCA from freely
        # rotating a covariance-degenerate subspace; all scales are still
        # estimated and standardized through the fit pipeline below.
        source_scales = _nonlinear_source_scales()
        x = x * source_scales
        for pair in range(NONLINEAR_PAIR_COUNT):
            left = 2 * pair
            right = left + 1
            latent = _normal(generator, (count,))
            x[:, left] = latent * source_scales[left]
            x[:, right] = ((latent.square() - 1.0) / math.sqrt(2.0)) * source_scales[right]
        return x
    _require(mechanism == "NULL_STATIONARY_GAUSSIAN", f"unknown mechanism: {mechanism}")
    return x


def _draw_banks(mechanism: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = _generator(seed)
    fit_groups = _observer_groups(FIT_COUNT)
    validation_groups = _observer_groups(VALIDATION_COUNT)
    if mechanism == "NULL_STATIONARY_GAUSSIAN":
        fit = _normal(generator, (FIT_COUNT, DIMENSION))
        validation = _normal(generator, (VALIDATION_COUNT, DIMENSION))
    else:
        fit = _draw_bank(mechanism, FIT_COUNT, generator, "fit", fit_groups)
        validation = _draw_bank(mechanism, VALIDATION_COUNT, generator, "validation", validation_groups)
    return fit, validation, fit_groups, validation_groups


def _fit_score_transform(
    fit: torch.Tensor,
    validation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    fit_mean = fit.mean(dim=0)
    centered_fit = fit - fit_mean
    _, singular_values, right_vectors = torch.linalg.svd(centered_fit, full_matrices=False)
    basis = right_vectors[:RANK].T.contiguous()
    fit_spectrum = singular_values[:RANK].square() / (FIT_COUNT - 1)
    _require(torch.all(fit_spectrum > 0), "fit spectrum is not strictly positive")
    fit_scale = fit_spectrum.sqrt()
    z_fit = centered_fit @ basis / fit_scale
    z_validation = (validation - fit_mean) @ basis / fit_scale
    _require(z_fit.shape == (FIT_COUNT, RANK), "fit score shape changed")
    _require(z_validation.shape == (VALIDATION_COUNT, RANK), "validation score shape changed")
    fit_covariance = _covariance(z_fit)
    identity = torch.eye(RANK, dtype=torch.float64)
    _require((fit_covariance - identity).norm().item() < 1.0e-8, "fit score covariance is not identity")
    return z_fit, z_validation, {
        "fit_mean": "column mean over fit bank",
        "centered_pca": "torch.linalg.svd(fit - fit_mean, full_matrices=False)",
        "rank": RANK,
        "fit_spectrum": "singular_values[:99]**2 / (n_fit - 1)",
        "score": "(x - fit_mean) @ V[:99].T / sqrt(fit_spectrum)",
        "spectrum_denominator": "n_fit - 1 = 499",
        "dtype": "torch.float64",
    }


def _covariance(z: torch.Tensor) -> torch.Tensor:
    centered = z - z.mean(dim=0)
    return centered.T @ centered / (z.shape[0] - 1)


def _upper_triangle(matrix: torch.Tensor) -> torch.Tensor:
    mask = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
    return matrix[mask]


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if denominator.item() <= 1.0e-14:
        return 0.0
    return _as_float((left @ right) / denominator)


def _pairwise_distances(points: torch.Tensor) -> torch.Tensor:
    differences = points[:, None, :] - points[None, :, :]
    distances = differences.square().sum(dim=2).sqrt()
    return _upper_triangle(distances)


def _group_score_means(z: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    return torch.stack([
        z[groups == group].mean(dim=0)
        for group in range(GROUP_COUNT)
    ])


def _local_metrics(
    z_fit: torch.Tensor,
    z_validation: torch.Tensor,
    fit_groups: torch.Tensor,
    validation_groups: torch.Tensor,
) -> dict[str, float]:
    fit_means = _group_score_means(z_fit, fit_groups)
    validation_means = _group_score_means(z_validation, validation_groups)
    fit_distances = _pairwise_distances(fit_means)
    validation_distances = _pairwise_distances(validation_means)
    return {
        "group_pairwise_distance_correlation": _correlation(fit_distances, validation_distances),
        "group_pairwise_distance_mean_validation": _as_float(validation_distances.mean()),
        "group_pairwise_distance_mean_fit": _as_float(fit_distances.mean()),
    }


def _dependence_metrics(z: torch.Tensor) -> dict[str, float]:
    squared = z.square()
    centered = squared - squared.mean(dim=0)
    covariance = centered.T @ centered / (z.shape[0] - 1)
    standard_deviation = covariance.diag().clamp_min(1.0e-18).sqrt()
    correlation = covariance / (standard_deviation[:, None] * standard_deviation[None, :])
    mask = ~torch.eye(RANK, dtype=torch.bool)
    off_diagonal = correlation[mask]
    global_level = off_diagonal.median()
    structured = correlation - global_level * (torch.ones_like(correlation) - torch.eye(RANK, dtype=torch.float64))
    residual_values = structured[mask].abs()
    top_k = min(DEPENDENCE_TOPK, residual_values.numel())
    top_values = torch.topk(residual_values, k=top_k).values
    total = residual_values.sum().clamp_min(1.0e-18)
    return {
        "square_correlation_residual_frobenius": _as_float(residual_values.norm()),
        "square_correlation_residual_max_abs": _as_float(residual_values.max()),
        "square_correlation_residual_topk_abs_share": _as_float(top_values.sum() / total),
        "square_correlation_global_median": _as_float(global_level),
        "square_correlation_pair_count": float(residual_values.numel()),
    }


def _third_moment_metrics(z: torch.Tensor) -> dict[str, float]:
    centered = z - z.mean(dim=0)
    contraction = centered.square().T @ centered / (z.shape[0] - 1)
    off_diagonal = _upper_triangle(contraction).abs()
    top_k = min(DEPENDENCE_TOPK, off_diagonal.numel())
    top_values = torch.topk(off_diagonal, k=top_k).values
    total = off_diagonal.sum().clamp_min(1.0e-18)
    return {
        "third_moment_offdiagonal_frobenius": _as_float(off_diagonal.norm()),
        "third_moment_offdiagonal_max_abs": _as_float(off_diagonal.max()),
        "third_moment_offdiagonal_topk_abs_share": _as_float(top_values.sum() / total),
    }


def _position_observer(z: torch.Tensor) -> dict[str, float]:
    r2 = z.square().sum(dim=1)
    positions = torch.arange(r2.shape[0], dtype=torch.float64)
    centered_position = positions - positions.mean()
    centered_r2 = r2 - r2.mean()
    denominator = centered_position.norm() * centered_r2.norm()
    correlation = 0.0 if denominator.item() <= 1.0e-14 else _as_float((centered_position @ centered_r2) / denominator)
    return {
        "validation_r2_position_correlation": correlation,
        "validation_r2_first_last_mean_delta": _as_float(r2[-50:].mean() - r2[:50].mean()),
    }


def _replicate(mechanism: str, seed: int) -> dict[str, Any]:
    fit, validation, fit_groups, validation_groups = _draw_banks(mechanism, seed)
    z_fit, z_validation, pipeline = _fit_score_transform(fit, validation)
    validation_covariance = _covariance(z_validation)
    fit_r2 = z_fit.square().sum(dim=1)
    validation_r2 = z_validation.square().sum(dim=1)
    identity = torch.eye(RANK, dtype=torch.float64)
    local = _local_metrics(z_fit, z_validation, fit_groups, validation_groups)
    dependence = _dependence_metrics(z_validation)
    angular_centered = z_validation - z_validation.mean(dim=0)
    angular_validation = angular_centered / angular_centered.norm(dim=1, keepdim=True).clamp_min(1.0e-18)
    angular_third = _third_moment_metrics(angular_validation)
    angular_third = {
        f"angular_{name}": value for name, value in angular_third.items()
    }
    metrics = {
        "fit_to_validation_mean_l2": _as_float(
            (z_validation.mean(dim=0) - z_fit.mean(dim=0)).norm()
        ),
        "validation_covariance_trace_abs_from_rank": _as_float(
            (validation_covariance.trace() - RANK).abs()
        ),
        "validation_covariance_identity_frobenius": _as_float(
            (validation_covariance - identity).norm()
        ),
        "validation_radial_r2_variance": _as_float(validation_r2.var(unbiased=True)),
        "validation_radial_r2_q95": _as_float(torch.quantile(validation_r2, 0.95)),
        "validation_radial_r2_median_delta_from_fit": _as_float(
            torch.quantile(validation_r2, 0.50) - torch.quantile(fit_r2, 0.50)
        ),
        "validation_radial_r2_mean": _as_float(validation_r2.mean()),
        **local,
        **dependence,
        **_third_moment_metrics(z_validation),
        **angular_third,
        "position_observer": _position_observer(z_validation),
    }
    return {
        "truth": mechanism,
        "seed": seed,
        "n_fit": FIT_COUNT,
        "n_validation": VALIDATION_COUNT,
        "d_input": DIMENSION,
        "rank": RANK,
        "pipeline": pipeline,
        "metrics": metrics,
    }


REFERENCE_METRICS = {
    "fit_to_validation_mean_l2": ("fit_to_validation_mean_l2",),
    "validation_covariance_trace_abs_from_rank": ("validation_covariance_trace_abs_from_rank",),
    "validation_radial_r2_variance": ("validation_radial_r2_variance",),
    "validation_radial_r2_q95": ("validation_radial_r2_q95",),
    "validation_radial_r2_median_delta_from_fit": (
        "validation_radial_r2_median_delta_from_fit",
    ),
    "group_pairwise_distance_correlation": ("group_pairwise_distance_correlation",),
    "group_pairwise_distance_mean_validation": ("group_pairwise_distance_mean_validation",),
    "square_correlation_residual_frobenius": ("square_correlation_residual_frobenius",),
    "square_correlation_residual_topk_abs_share": ("square_correlation_residual_topk_abs_share",),
    "square_correlation_global_median": ("square_correlation_global_median",),
    "angular_third_moment_offdiagonal_frobenius": (
        "angular_third_moment_offdiagonal_frobenius",
    ),
    "angular_third_moment_offdiagonal_max_abs": (
        "angular_third_moment_offdiagonal_max_abs",
    ),
}


def _metric_value(record: Mapping[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record["metrics"]
    for key in path:
        value = value[key]
    return float(value)


def _quantile_thresholds(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    values = list(records)
    _require(len(values) >= 100, "development NULL calibration is too small")
    thresholds: dict[str, float] = {}
    for name, path in REFERENCE_METRICS.items():
        series = torch.tensor(
            [_metric_value(record, path) for record in values],
            dtype=torch.float64,
        )
        thresholds[name] = _as_float(torch.quantile(series, REFERENCE_QUANTILE))
    return thresholds


def _triggers(record: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, bool]:
    metrics = record["metrics"]
    partition = (
        float(metrics["fit_to_validation_mean_l2"]) > thresholds["fit_to_validation_mean_l2"]
        and float(metrics["validation_radial_r2_median_delta_from_fit"])
        > thresholds["validation_radial_r2_median_delta_from_fit"]
    )
    return {
        "partition_or_moment_drift": partition,
        "radial_scale_mixture": (
            float(metrics["validation_radial_r2_variance"]) > thresholds["validation_radial_r2_variance"]
            and float(metrics["validation_radial_r2_q95"]) > thresholds["validation_radial_r2_q95"]
            and float(metrics["square_correlation_global_median"])
            > thresholds["square_correlation_global_median"]
            and float(metrics["square_correlation_global_median"])
            > RADIAL_SQUARE_CORRELATION_EFFECT_FLOOR
        ),
        "local_mixture": (
            float(metrics["group_pairwise_distance_correlation"])
            > thresholds["group_pairwise_distance_correlation"]
            and float(metrics["group_pairwise_distance_mean_validation"])
            > thresholds["group_pairwise_distance_mean_validation"]
        ),
        "nonlinear_dependence": (
            float(metrics["angular_third_moment_offdiagonal_frobenius"])
            > thresholds["angular_third_moment_offdiagonal_frobenius"]
            and float(metrics["angular_third_moment_offdiagonal_max_abs"])
            > thresholds["angular_third_moment_offdiagonal_max_abs"]
        ),
    }


def _select(record: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    triggers = _triggers(record, thresholds)
    active = [name for name, enabled in triggers.items() if enabled]
    if triggers["partition_or_moment_drift"]:
        selected = SELECTED["PARTITION_OR_MOMENT_DRIFT"]
        rule = "global mean plus fit-to-validation radial-median drift has precedence over secondary radial elevation"
    elif triggers["local_mixture"] and not triggers["nonlinear_dependence"]:
        selected = SELECTED["LOCAL_MIXTURE"]
        rule = "reproducible local profile has precedence over secondary radial elevation"
    elif triggers["radial_scale_mixture"]:
        selected = SELECTED["RADIAL_SCALE_MIXTURE"]
        rule = "registered common-scale squared-correlation effect has precedence over secondary angular elevation"
    elif triggers["nonlinear_dependence"] and not triggers["local_mixture"]:
        selected = SELECTED["NONLINEAR_DEPENDENCE"]
        rule = "structured angular third-moment dependence is the remaining non-radial mechanism"
    elif len(active) == 1:
        selected = {
            "partition_or_moment_drift": SELECTED["PARTITION_OR_MOMENT_DRIFT"],
            "radial_scale_mixture": SELECTED["RADIAL_SCALE_MIXTURE"],
            "local_mixture": SELECTED["LOCAL_MIXTURE"],
            "nonlinear_dependence": SELECTED["NONLINEAR_DEPENDENCE"],
        }[active[0]]
        rule = "exactly one calibrated mechanism predicate is true"
    else:
        selected = UNRESOLVED
        rule = "zero or multiple calibrated mechanism predicates are true"
    return {
        "selected": selected,
        "triggers": triggers,
        "active_trigger_names": active,
        "rule": rule,
    }


def _apply_selector(
    records: Iterable[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        enriched = dict(record)
        enriched["selection"] = _select(record, thresholds)
        result.append(enriched)
    return result


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0.0
    for successes in range(k + 1):
        log_term = (
            math.lgamma(n + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(n - successes + 1)
            + successes * math.log(probability)
            + (n - successes) * math.log1p(-probability)
        )
        total += math.exp(log_term)
    return min(max(total, 0.0), 1.0)


def _binomial_upper_bound(k: int, n: int, alpha: float = CONFIDENCE_ALPHA) -> float:
    if k >= n:
        return 1.0
    low = 0.0
    high = 1.0
    for _ in range(80):
        mid = (low + high) / 2.0
        if _binomial_cdf(k, n, mid) > alpha:
            low = mid
        else:
            high = mid
    return high


def _binomial_lower_bound(k: int, n: int, alpha: float = CONFIDENCE_ALPHA) -> float:
    if k <= 0:
        return 0.0
    low = 0.0
    high = 1.0
    for _ in range(80):
        mid = (low + high) / 2.0
        tail = 1.0 - _binomial_cdf(k - 1, n, mid)
        if tail < alpha:
            low = mid
        else:
            high = mid
    return low


def _selected_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        selected = str(record["selection"]["selected"])
        counts[selected] = counts.get(selected, 0) + 1
    return dict(sorted(counts.items()))


def _truth_summary(truth: str, records: list[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    counts = _selected_counts(records)
    if truth == "NULL_STATIONARY_GAUSSIAN":
        false_positive = count - counts.get(UNRESOLVED, 0)
        return {
            "truth": truth,
            "count": count,
            "selected_counts": counts,
            "false_positive_count": false_positive,
            "false_positive_rate": false_positive / count,
            "false_positive_upper_95": _binomial_upper_bound(false_positive, count),
            "pass": false_positive <= NULL_MAX_FALSE_POSITIVE_COUNT
            and false_positive / count <= NULL_MAX_FALSE_POSITIVE_RATE
            and _binomial_upper_bound(false_positive, count) <= 0.10,
        }
    expected = SELECTED[truth]
    correct = counts.get(expected, 0)
    wrong = count - correct - counts.get(UNRESOLVED, 0)
    unresolved = counts.get(UNRESOLVED, 0)
    correct_rate = correct / count
    wrong_rate = wrong / count
    return {
        "truth": truth,
        "count": count,
        "selected_counts": counts,
        "expected_selection": expected,
        "correct_count": correct,
        "wrong_count": wrong,
        "unresolved_count": unresolved,
        "correct_rate": correct_rate,
        "wrong_rate": wrong_rate,
        "unresolved_rate": unresolved / count,
        "correct_lower_95": _binomial_lower_bound(correct, count),
        "wrong_upper_95": _binomial_upper_bound(wrong, count),
        "pass": correct_rate >= MECHANISM_MIN_CORRECT_RATE
        and wrong_rate <= MECHANISM_MAX_WRONG_RATE
        and _binomial_lower_bound(correct, count) >= MECHANISM_MIN_CORRECT_LOWER_BOUND
        and _binomial_upper_bound(wrong, count) <= MECHANISM_MAX_WRONG_UPPER_BOUND,
    }


def _development_summary(
    null_calibration: list[Mapping[str, Any]],
    null_evaluation: list[Mapping[str, Any]],
    mechanism_records: Mapping[str, list[Mapping[str, Any]]],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    selected_null_evaluation = _apply_selector(null_evaluation, thresholds)
    null_summary = _truth_summary("NULL_STATIONARY_GAUSSIAN", selected_null_evaluation)
    mechanism_summaries = {
        mechanism: _truth_summary(mechanism, _apply_selector(records, thresholds))
        for mechanism, records in mechanism_records.items()
    }
    return {
        "null_calibration_count": len(null_calibration),
        "null_evaluation_count": len(null_evaluation),
        "threshold_quantile": REFERENCE_QUANTILE,
        "thresholds": thresholds,
        "null_evaluation": null_summary,
        "mechanism_development": mechanism_summaries,
    }


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
    return {
        "branch": branch,
        "head": head,
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in status],
    }


def _audit_previous_selector(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "scripts/research/attribute_n3_producer_mismatch.py"
    _require(path.is_file(), "previous N3 selector is missing")
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "sha256": file_sha256(path),
        "historical_selector_modified": False,
        "known_audit_points": [
            "strict sample correlation > 0 was not a finite-sample rejection rule",
            "exact radial moment inequality was not a finite-sample rejection rule",
            "fit-PCA identity covariance cannot justify unconditional class-centering improvement",
        ],
    }


def _build_development(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    null_calibration = [
        _replicate("NULL_STATIONARY_GAUSSIAN", seed)
        for seed in DEVELOPMENT_SEEDS["NULL_CALIBRATION"]
    ]
    thresholds = _quantile_thresholds(null_calibration)
    null_evaluation = [
        _replicate("NULL_STATIONARY_GAUSSIAN", seed)
        for seed in DEVELOPMENT_SEEDS["NULL_EVALUATION"]
    ]
    mechanism_records = {
        mechanism: [
            _replicate(mechanism, seed)
            for seed in DEVELOPMENT_SEEDS[mechanism]
        ]
        for mechanism in NON_NULL_MECHANISMS
    }
    summary = _development_summary(
        null_calibration,
        null_evaluation,
        mechanism_records,
        thresholds,
    )
    return {
        "null_calibration": null_calibration,
        "null_evaluation": null_evaluation,
        "mechanism_records": mechanism_records,
    }, summary


def _frozen_spec(
    repo_root: Path,
    git: Mapping[str, Any],
    audit: Mapping[str, Any],
    development_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SPEC_SCHEMA,
        "task_id": TASK_ID,
        "freeze_boundary": "materialized before any confirmation result is inspected",
        "local_state_at_freeze": {
            "branch": git["branch"],
            "head": git["head"],
            "script_path": SCRIPT_PATH.as_posix(),
            "script_sha256": file_sha256(repo_root / SCRIPT_PATH),
        },
        "pipeline": {
            "d_input": DIMENSION,
            "n_fit": FIT_COUNT,
            "n_validation": VALIDATION_COUNT,
            "rank": RANK,
            "fit_mean": "column mean over fit bank",
            "centered_pca": "torch.linalg.svd(fit - fit_mean, full_matrices=False)",
            "fit_spectrum": "singular_values[:99]**2 / (n_fit - 1)",
            "validation_transform": "(validation - fit_mean) @ V[:99].T / sqrt(fit_spectrum)",
            "dtype": "torch.float64",
        },
        "metrics": {
            "partition": {
                "mean": "fit_to_validation_mean_l2",
                "scale": "validation_radial_r2_median_delta_from_fit",
                "direction": "both greater than development NULL q99",
            },
            "radial": {
                "variance": "validation_radial_r2_variance",
                "tail": "validation_radial_r2_q95",
                "common_scale_marker": "square_correlation_global_median",
                "common_scale_effect_floor": RADIAL_SQUARE_CORRELATION_EFFECT_FLOOR,
                "direction": "variance and tail greater than development NULL q99; common-scale marker greater than both development NULL q99 and the registered strong-effect floor",
            },
            "local": {
                "reproducibility": "group_pairwise_distance_correlation",
                "between_group_scale": "group_pairwise_distance_mean_validation",
                "direction": "both greater than development NULL q99",
            },
            "dependence": {
                "structured_residual": "angular_third_moment_offdiagonal_frobenius",
                "concentration": "angular_third_moment_offdiagonal_max_abs",
                "direction": "both greater than development NULL q99",
            },
        },
        "threshold_quantile": REFERENCE_QUANTILE,
        "thresholds": development_summary["thresholds"],
        "ambiguity_rule": "global drift has registered precedence; otherwise local/dependence structured markers may explain secondary radial elevation; unresolved for local/dependence overlap or zero evidence",
        "precedence": "PARTITION_OR_MOMENT_DRIFT > LOCAL_MIXTURE > RADIAL_SCALE_MIXTURE > NONLINEAR_DEPENDENCE; precedence only resolves registered secondary overlap",
        "mechanism_definitions": MECHANISM_DEFINITIONS,
        "development_seed_namespace": {
            name: list(seeds) for name, seeds in DEVELOPMENT_SEEDS.items()
        },
        "confirmation_seed_namespace": {
            name: list(seeds) for name, seeds in CONFIRMATION_SEEDS.items()
        },
        "previous_selector_audit": audit,
        "real_data_boundary": {
            "project_ols_values_loaded": False,
            "real_values_used_for_thresholds": False,
            "real_values_used_for_mechanism_strengths": False,
        },
    }


def _write_new_json(path: Path, value: Mapping[str, Any], kind: str) -> None:
    atomic_write_new_json(path, value, artifact_kind=kind)


def _classify_confirmation(
    null_summary: Mapping[str, Any],
    mechanism_summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    all_pass = bool(null_summary["pass"]) and all(
        bool(summary["pass"]) for summary in mechanism_summaries.values()
    )
    all_fail = not bool(null_summary["pass"]) and all(
        not bool(summary["pass"]) for summary in mechanism_summaries.values()
    )
    if all_pass:
        return (
            "N3_ATTRIBUTION_SELECTOR_CALIBRATED_V2",
            "NULL and every registered strong synthetic mechanism met the frozen confirmation targets",
        )
    if all_fail:
        return (
            "N3_ATTRIBUTION_SELECTOR_NOT_DISCRIMINATIVE_V2",
            "NULL and every registered strong synthetic mechanism failed the frozen confirmation targets",
        )
    return (
        "N3_ATTRIBUTION_SELECTOR_CALIBRATION_MIXED_V2",
        "confirmation evidence mixed passing and failing frozen targets",
    )


def _run_confirmation(
    thresholds: Mapping[str, float],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    records: dict[str, list[dict[str, Any]]] = {}
    for mechanism, seeds in CONFIRMATION_SEEDS.items():
        raw = [_replicate(mechanism, seed) for seed in seeds]
        records[mechanism] = _apply_selector(raw, thresholds)
    summaries = {
        truth: _truth_summary(truth, values)
        for truth, values in records.items()
    }
    return summaries, records


def _materialize(
    repo_root: Path,
    git: Mapping[str, Any],
    audit: Mapping[str, Any],
    development_summary: Mapping[str, Any],
    spec: Mapping[str, Any],
    spec_sha256: str,
    confirmation_summaries: Mapping[str, Any],
    confirmation_records: Mapping[str, list[Mapping[str, Any]]],
) -> Path:
    output_dir = repo_root / OUTPUT_DIR
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        _write_new_json(temporary / SPEC_NAME, spec, "N3 frozen attribution selector specification")
        _require(file_sha256(temporary / SPEC_NAME) == spec_sha256, "frozen spec hash changed before publication")
        classification, reason = _classify_confirmation(
            confirmation_summaries["NULL_STATIONARY_GAUSSIAN"],
            {
                mechanism: confirmation_summaries[mechanism]
                for mechanism in NON_NULL_MECHANISMS
            },
        )
        report = {
            "schema": REPORT_SCHEMA,
            "task_id": TASK_ID,
            "gate": GATE,
            "classification": classification,
            "classification_reason": reason,
            "local_state": {
                "branch": git["branch"],
                "head_before_artifact": git["head"],
                "dirty_paths_before_artifact": git["dirty_paths"],
                "script_path": SCRIPT_PATH.as_posix(),
                "script_sha256": file_sha256(repo_root / SCRIPT_PATH),
            },
            "development_confirmation_isolation": {
                "development_seed_namespace": {
                    name: list(seeds) for name, seeds in DEVELOPMENT_SEEDS.items()
                },
                "confirmation_seed_namespace": {
                    name: list(seeds) for name, seeds in CONFIRMATION_SEEDS.items()
                },
                "seed_intersection": [],
                "confirmation_started_after_spec_materialized": True,
                "spec_sha256_before_confirmation": spec_sha256,
                "confirmation_results_used_to_change_spec": False,
                "project_ols_values_loaded": False,
            },
            "selector_audit": audit,
            "synthetic_pipeline": {
                "d_input": DIMENSION,
                "n_fit": FIT_COUNT,
                "n_validation": VALIDATION_COUNT,
                "rank": RANK,
                "spectrum_denominator": "n_fit - 1 = 499",
                "all_replications_used_fit_estimated_parameters": True,
                "known_population_parameters_used_for_standardization": False,
                "ambient_rng_used": False,
                "dtype": "torch.float64",
            },
            "mechanism_definitions": MECHANISM_DEFINITIONS,
            "registered_confirmation_targets": {
                "null_false_positive_count_max": NULL_MAX_FALSE_POSITIVE_COUNT,
                "null_false_positive_rate_max": NULL_MAX_FALSE_POSITIVE_RATE,
                "null_false_positive_upper_95_max": 0.10,
                "mechanism_correct_rate_min": MECHANISM_MIN_CORRECT_RATE,
                "mechanism_wrong_rate_max": MECHANISM_MAX_WRONG_RATE,
                "mechanism_correct_lower_95_min": MECHANISM_MIN_CORRECT_LOWER_BOUND,
                "mechanism_wrong_upper_95_max": MECHANISM_MAX_WRONG_UPPER_BOUND,
            },
            "development_summary": development_summary,
            "frozen_selector_spec": {
                "path": SPEC_NAME,
                "sha256": spec_sha256,
                "thresholds": spec["thresholds"],
                "ambiguity_rule": spec["ambiguity_rule"],
                "precedence": spec["precedence"],
            },
            "confirmation": {
                "summaries": confirmation_summaries,
                "confusion_matrix": [confirmation_summaries[name] for name in CONFIRMATION_SEEDS],
                "records": confirmation_records,
            },
            "real_n3_boundary": {
                "attribution_executed": False,
                "label": None,
                "reason": "optional real-data review was not consumed by this synthetic calibration task",
                "frozen_historical_evidence_modified": False,
                "producer_refit": False,
                "rank_reselected": False,
            },
            "future_confirmation_data_boundary": {
                "current_internal_validation_bank_role": "N3_DEVELOPMENT_BANK_ONLY",
                "fresh_confirmation_source_required_for_future_producer_family_claim": True,
                "historical_500_500_partition_changed": False,
            },
            "claims_boundary": {
                "synthetic_selector_discrimination_only": True,
                "real_n3_mechanism_identified": False,
                "producer_family_supported": False,
                "intrinsic_ols_dimension_supported": False,
                "novel_ols_valid": False,
                "semantic_generation_supported": False,
                "generation_readiness": "not_ready",
            },
            "not_executed": [
                "real N3 attribution",
                "producer refit or rank selection",
                "real-data threshold or mechanism-strength tuning",
                "resplitting the historical N3 500/500 partition",
                "CUDA, optimizer, backward, training, forward, evaluator, generation",
                "producer-family implementation",
            ],
        }
        _write_new_json(temporary / REPORT_NAME, report, "N3 attribution selector V2 calibration report")
        expected_names = sorted((SPEC_NAME, REPORT_NAME))
        _require(sorted(path.name for path in temporary.iterdir()) == expected_names, "unexpected calibration output layout")
        if output_dir.exists():
            raise FileExistsError(f"calibration output appeared: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / REPORT_NAME


def execute(repo_root: Path, phase: str) -> dict[str, Any]:
    torch.set_num_threads(1)
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 authority branch")
    audit = _audit_previous_selector(repo_root)
    development_records, development_summary = _build_development(repo_root)
    if phase == "development":
        return {
            "phase": phase,
            "development_summary": development_summary,
            "script_sha256": file_sha256(repo_root / SCRIPT_PATH),
        }

    spec = _frozen_spec(repo_root, git, audit, development_summary)
    descriptor, temporary_name = tempfile.mkstemp(prefix="n3_selector_v2_spec_", suffix=".json")
    os.close(descriptor)
    temporary_spec = Path(temporary_name)
    temporary_spec.unlink()
    try:
        _write_new_json(temporary_spec, spec, "N3 frozen attribution selector specification preview")
        spec_sha256 = file_sha256(temporary_spec)
    finally:
        temporary_spec.unlink(missing_ok=True)

    confirmation_summaries, confirmation_records = _run_confirmation(spec["thresholds"])
    report_path = _materialize(
        repo_root,
        git,
        audit,
        development_summary,
        spec,
        spec_sha256,
        confirmation_summaries,
        confirmation_records,
    )
    return {
        "phase": phase,
        "report": report_path.as_posix(),
        "spec_sha256": spec_sha256,
        "confirmation_summaries": confirmation_summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--phase",
        choices=("development", "full"),
        default="full",
        help="development prints only synthetic development diagnostics; full freezes and confirms once",
    )
    args = parser.parse_args()
    result = execute(args.repo_root.resolve(), args.phase)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
