"""Resolve frozen N3 score-geometry attribution without fitting or sampling.

This is a read-only R1 consumer.  It reuses the already frozen input verifier,
examines score radial/angular structure and registered source order, and makes
no producer-family fit, rank choice, resampling, optical forward, or evaluator
call.  Labels are diagnostic observers only.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

import torch

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256, tensor_sha256


TASK_ID = "RESOLVE_N3_SCORE_GEOMETRY_ATTRIBUTION_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_score_geometry_attribution_report.v1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
PREVIOUS_CONSUMER = Path("scripts/research/attribute_n3_producer_mismatch.py")
EXPECTED_PREVIOUS_CONSUMER_SHA256 = (
    "5f3fb3527ac0f8004e69f5d923c2c91da92ac67e9b474c9905f1bb1531b7f621"
)
CLASS_COUNT = 10
DIMENSION = 100
RANK = 99
BANK_COUNT = 500
ROWS_PER_CLASS = 50
QUANTILE_PROBABILITIES = (0.05, 0.25, 0.50, 0.75, 0.95)
RADIAL_PROBABILITIES = (0.05, 0.50, 0.95)
REPORT_NAME = "n3_score_geometry_attribution_report.v1.json"

CLASSIFICATION_UNRESOLVED = "N3_SCORE_GEOMETRY_ATTRIBUTION_UNRESOLVED"
FAMILY_PARTITION = "DATA_PARTITION_OR_MOMENT_STABILITY_CONFOUND"
FAMILY_RADIAL = "RADIAL_SCALE_MIXTURE_SCORE_BASELINE_V1"
FAMILY_LOCAL = "UNSUPERVISED_LOCAL_MIXTURE_AMPLITUDE_PRODUCER_V1"
FAMILY_DEPENDENCE = "NON_GAUSSIAN_DEPENDENCE_SCORE_PRODUCER_V1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _load_previous_consumer(repo_root: Path) -> ModuleType:
    path = repo_root / PREVIOUS_CONSUMER
    _require(path.is_file(), f"previous attribution consumer is missing: {path}")
    _require(
        file_sha256(path) == EXPECTED_PREVIOUS_CONSUMER_SHA256,
        "previous attribution consumer changed",
    )
    spec = importlib.util.spec_from_file_location("frozen_n3_attribution", path)
    _require(spec is not None and spec.loader is not None, "cannot load frozen attribution consumer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _summary(value: torch.Tensor, probabilities: tuple[float, ...] = QUANTILE_PROBABILITIES) -> dict[str, float]:
    view = value.detach().to(torch.float64).cpu().reshape(-1)
    _require(view.numel() > 0 and bool(torch.isfinite(view).all()), "summary input is invalid")
    quantiles = torch.quantile(view, torch.tensor(probabilities, dtype=torch.float64))
    result = {f"p{int(probability * 100):02d}": float(item) for probability, item in zip(probabilities, quantiles)}
    result.update(
        {
            "min": float(view.min()),
            "mean": float(view.mean()),
            "max": float(view.max()),
        }
    )
    return result


def _quantile_map(value: torch.Tensor, probabilities: tuple[float, ...]) -> dict[str, float]:
    view = value.detach().to(torch.float64).cpu().reshape(-1)
    _require(view.numel() > 0 and bool(torch.isfinite(view).all()), "quantile input is invalid")
    quantiles = torch.quantile(view, torch.tensor(probabilities, dtype=torch.float64))
    return {f"p{int(probability * 100):02d}": float(item) for probability, item in zip(probabilities, quantiles)}


def _covariance_and_correlation(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    view = value.detach().to(torch.float64).cpu().contiguous()
    _require(view.ndim == 2 and view.shape[0] >= 2, "covariance input is invalid")
    centered = view - view.mean(dim=0, keepdim=True)
    covariance = centered.T.mm(centered).div(float(view.shape[0] - 1))
    diagonal = torch.diagonal(covariance)
    _require(bool(torch.isfinite(covariance).all()) and bool(diagonal.gt(0.0).all()), "covariance is invalid")
    correlation = covariance / torch.sqrt(diagonal[:, None] * diagonal[None, :])
    _require(bool(torch.isfinite(correlation).all()), "correlation is invalid")
    return covariance.contiguous(), correlation.contiguous()


def _off_diagonal_summary(matrix: torch.Tensor, correlation: torch.Tensor | None = None) -> dict[str, float]:
    dimension = matrix.shape[0]
    mask = ~torch.eye(dimension, dtype=torch.bool)
    values = matrix[mask]
    total = matrix.square().sum()
    off_diagonal = values.square().sum()
    _require(bool(total.gt(0.0)), "matrix energy is zero")
    result = {
        "mean_abs_off_diagonal": float(values.abs().mean()),
        "p95_abs_off_diagonal": float(torch.quantile(values.abs(), 0.95)),
        "maximum_abs_off_diagonal": float(values.abs().max()),
        "off_diagonal_energy_fraction": float(off_diagonal / total),
    }
    if correlation is not None:
        corr_values = correlation[mask].abs()
        result.update(
            {
                "mean_abs_cross_correlation": float(corr_values.mean()),
                "p95_abs_cross_correlation": float(torch.quantile(corr_values, 0.95)),
                "maximum_abs_cross_correlation": float(corr_values.max()),
            }
        )
    return result


def _pearson(first: torch.Tensor, second: torch.Tensor) -> float | None:
    left = first.to(torch.float64).reshape(-1)
    right = second.to(torch.float64).reshape(-1)
    _require(left.shape == right.shape and left.numel() >= 2, "association inputs are invalid")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if float(denominator) == 0.0:
        return None
    return float(left_centered.dot(right_centered) / denominator)


def _ranks(value: torch.Tensor) -> torch.Tensor:
    values = value.to(torch.float64).reshape(-1).tolist()
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = torch.empty(len(values), dtype=torch.float64)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (float(start + 1) + float(end)) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = rank
        start = end
    return ranks


def _association(first: torch.Tensor, second: torch.Tensor) -> dict[str, float | None]:
    return {
        "pearson": _pearson(first, second),
        "spearman": _pearson(_ranks(first), _ranks(second)),
    }


def _chi_square_quantiles(probabilities: tuple[float, ...]) -> dict[str, float]:
    """Deterministic analytic chi-square quantiles by fixed bisection."""

    probs = torch.tensor(probabilities, dtype=torch.float64)
    lower = torch.zeros_like(probs)
    upper = torch.full_like(probs, 400.0)
    shape = torch.tensor(float(RANK) / 2.0, dtype=torch.float64)
    for _ in range(80):
        middle = (lower + upper) / 2.0
        cdf = torch.special.gammainc(shape, middle / 2.0)
        lower = torch.where(cdf < probs, middle, lower)
        upper = torch.where(cdf >= probs, middle, upper)
    quantiles = (lower + upper) / 2.0
    _require(bool(torch.isfinite(quantiles).all()), "chi-square reference is non-finite")
    return {f"p{int(probability * 100):02d}": float(item) for probability, item in zip(probabilities, quantiles)}


def _radial_profile(scores: torch.Tensor) -> dict[str, Any]:
    view = scores.to(torch.float64).cpu().contiguous()
    radial = view.square().sum(dim=1)
    covariance, correlation = _covariance_and_correlation(view)
    reference_quantiles = _chi_square_quantiles(RADIAL_PROBABILITIES)
    quantiles = _quantile_map(radial, RADIAL_PROBABILITIES)
    return {
        "count": int(view.shape[0]),
        "dimension": int(view.shape[1]),
        "r2_summary": _summary(radial, RADIAL_PROBABILITIES),
        "r2_mean": float(radial.mean()),
        "r2_variance": float(radial.var(unbiased=True)),
        "r2_quantiles": quantiles,
        "gaussian_reference": {
            "distribution": "chi_square_99_for_standard_normal_scores",
            "degrees_of_freedom": RANK,
            "mean": float(RANK),
            "variance": float(2 * RANK),
            "quantiles": reference_quantiles,
        },
        "mean_abs_error": abs(float(radial.mean()) - float(RANK)),
        "variance_abs_error": abs(float(radial.var(unbiased=True)) - float(2 * RANK)),
        "quantile_abs_error": {
            key: abs(quantiles[key] - reference_quantiles[key]) for key in quantiles
        },
        "tail_fraction_below_reference_p05": float((radial < reference_quantiles["p05"]).to(torch.float64).mean()),
        "tail_fraction_above_reference_p95": float((radial > reference_quantiles["p95"]).to(torch.float64).mean()),
        "covariance_identity_frobenius": float((covariance - torch.eye(RANK, dtype=torch.float64)).norm()),
        "cross_correlation": _off_diagonal_summary(covariance, correlation),
        "finite": True,
    }


def _radial_mean_decomposition(
    scores: torch.Tensor,
    latent_coordinates: torch.Tensor,
    lambda_values: torch.Tensor,
) -> dict[str, Any]:
    view = scores.to(torch.float64).cpu().contiguous()
    h = latent_coordinates.to(torch.float64).cpu().contiguous()
    lambdas = lambda_values.to(torch.float64).cpu().contiguous()
    centered = view - view.mean(dim=0, keepdim=True)
    covariance_population = centered.T.mm(centered).div(float(view.shape[0]))
    covariance_unbiased = centered.T.mm(centered).div(float(view.shape[0] - 1))
    mean = view.mean(dim=0)
    radial = view.square().sum(dim=1)
    population_trace = float(torch.trace(covariance_population))
    unbiased_trace = float(torch.trace(covariance_unbiased))
    mean_shift = float(mean.square().sum())
    variance_contribution = population_trace - float(RANK)
    radial_mean = float(radial.mean())
    excess = radial_mean - float(RANK)
    identity_residual = radial_mean - (mean_shift + population_trace)
    fractions = None
    if abs(excess) > 1.0e-12:
        fractions = {
            "mean_shift_fraction_of_excess": mean_shift / excess,
            "variance_scale_fraction_of_excess": variance_contribution / excess,
        }
    rho_unbiased = h.var(dim=0, unbiased=True) / lambdas
    rho_population = h.var(dim=0, unbiased=False) / lambdas
    positive_excess = (rho_unbiased - 1.0).clamp_min(0.0)
    order = torch.argsort(positive_excess, descending=True)
    total_positive = float(positive_excess.sum())
    top_shares: dict[str, float | None] = {}
    if total_positive > 0.0:
        for count in (1, 5, 10):
            top_shares[f"top_{count}_positive_variance_excess_share"] = float(
                positive_excess.index_select(0, order[:count]).sum() / total_positive
            )
    else:
        top_shares = {f"top_{count}_positive_variance_excess_share": None for count in (1, 5, 10)}
    return {
        "formula": "E[||z||^2] = ||E[z]||^2 + trace(C_z), C_z=(z-E[z])^T(z-E[z])/N",
        "count": int(view.shape[0]),
        "dimension": int(view.shape[1]),
        "radial_r2_mean": radial_mean,
        "radial_r2_excess_over_rank": excess,
        "mean_shift_norm_squared": mean_shift,
        "variance_trace_population": population_trace,
        "variance_trace_unbiased_N_minus_1": unbiased_trace,
        "variance_scale_contribution_trace_minus_rank": variance_contribution,
        "variance_scale_contribution_unbiased_trace_minus_rank": unbiased_trace - float(RANK),
        "identity_residual": identity_residual,
        "identity_residual_abs": abs(identity_residual),
        "contribution_fractions_of_radial_excess": fractions,
        "rho_k_definition": "unbiased Var(h_k)_bank / lambda_k, h=(q-mu_fit)W_99; lambda uses n_fit-1",
        "rho_k_summary": _summary(rho_unbiased),
        "rho_k_population_summary": _summary(rho_population),
        "rho_k_fraction_greater_than_one": float((rho_unbiased > 1.0).to(torch.float64).mean()),
        "rho_k_count_greater_than_one": int((rho_unbiased > 1.0).sum()),
        "rho_k_positive_variance_excess_summary": _summary(positive_excess),
        "rho_k_positive_excess_concentration": top_shares,
        "rho_k_positive_excess_count": int((positive_excess > 0.0).sum()),
        "rho_k_positive_excess_concentration_is_descriptive_only": True,
    }


def _standardize_for_observation(scores: torch.Tensor) -> torch.Tensor:
    view = scores.to(torch.float64).cpu().contiguous()
    mean = view.mean(dim=0, keepdim=True)
    standard_deviation = view.std(dim=0, unbiased=True, keepdim=True)
    _require(bool(standard_deviation.gt(0.0).all()), "diagnostic normalization has zero variance")
    result = (view - mean) / standard_deviation
    _require(bool(torch.isfinite(result).all()), "diagnostic normalized scores are non-finite")
    return result.contiguous()


def _angular_matrices(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    view = scores.to(torch.float64).cpu().contiguous()
    radius = view.norm(dim=1)
    _require(bool(radius.gt(0.0).all()), "angular observer encountered a zero score")
    unit = view / radius[:, None]
    second_moment = unit.T.mm(unit).div(float(unit.shape[0]))
    covariance, _ = _covariance_and_correlation(unit)
    return unit, second_moment, covariance


def _angular_profile(scores: torch.Tensor) -> dict[str, Any]:
    unit, second_moment, covariance = _angular_matrices(scores)
    _, correlation = _covariance_and_correlation(unit)
    identity_over_dimension = torch.eye(RANK, dtype=torch.float64).div(float(RANK))
    second_eigenvalues = torch.linalg.eigvalsh(second_moment)
    covariance_eigenvalues = torch.linalg.eigvalsh(covariance)
    gram = unit.mm(unit.T)
    pair_mask = torch.triu(torch.ones_like(gram, dtype=torch.bool), diagonal=1)
    pairwise = gram[pair_mask]
    return {
        "count": int(unit.shape[0]),
        "second_moment_relative_identity_frobenius": float((second_moment - identity_over_dimension).norm()),
        "second_moment_off_diagonal": _off_diagonal_summary(second_moment),
        "second_moment_eigenvalue_summary": _summary(second_eigenvalues),
        "second_moment_top_eigenvalue_mass": float(second_eigenvalues[-1]),
        "second_moment_top_five_eigenvalue_mass": float(second_eigenvalues[-5:].sum()),
        "angular_covariance_identity_frobenius": float((covariance - identity_over_dimension).norm()),
        "angular_covariance_eigenvalue_summary": _summary(covariance_eigenvalues),
        "angular_covariance_max_to_median_eigenvalue": float(
            covariance_eigenvalues[-1] / torch.median(covariance_eigenvalues)
        ),
        "angular_covariance_cross_correlation": _off_diagonal_summary(covariance, correlation),
        "mean_unit_vector_norm": float(unit.mean(dim=0).norm()),
        "pairwise_cosine_summary": _summary(pairwise),
        "pairwise_cosine_abs_summary": _summary(pairwise.abs()),
        "finite": True,
    }


def _angular_discrepancy(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    first_unit, first_second, first_covariance = _angular_matrices(first)
    second_unit, second_second, second_covariance = _angular_matrices(second)
    first_gram = first_unit.mm(first_unit.T)
    second_gram = second_unit.mm(second_unit.T)
    mask = torch.triu(torch.ones_like(first_gram, dtype=torch.bool), diagonal=1)
    return {
        "second_moment_frobenius": float((first_second - second_second).norm()),
        "covariance_frobenius": float((first_covariance - second_covariance).norm()),
        "pairwise_cosine_mean_abs_delta": float(
            (first_gram[mask].mean() - second_gram[mask].mean()).abs()
        ),
        "pairwise_cosine_std_abs_delta": float(
            (first_gram[mask].std(unbiased=True) - second_gram[mask].std(unbiased=True)).abs()
        ),
    }


def _rank_quartile_groups(values: torch.Tensor) -> list[torch.Tensor]:
    view = values.to(torch.float64).cpu().reshape(-1)
    ordered = sorted(range(view.numel()), key=lambda index: (float(view[index]), index))
    result: list[torch.Tensor] = []
    for quartile in range(4):
        start = quartile * (view.numel() // 4)
        end = (quartile + 1) * (view.numel() // 4)
        result.append(torch.tensor(ordered[start:end], dtype=torch.int64))
    _require(sum(int(item.numel()) for item in result) == view.numel(), "radial quartile groups do not cover rows")
    return result


def _radial_quartile_angular(scores: torch.Tensor) -> list[dict[str, Any]]:
    radial = scores.to(torch.float64).square().sum(dim=1)
    groups = _rank_quartile_groups(radial)
    result: list[dict[str, Any]] = []
    for quartile, indices in enumerate(groups, start=1):
        values = radial.index_select(0, indices)
        result.append(
            {
                "quartile": quartile,
                "count": int(indices.numel()),
                "radial_min": float(values.min()),
                "radial_max": float(values.max()),
                "angular": _angular_profile(scores.index_select(0, indices)),
            }
        )
    return result


def _angular_structure(
    fit_scores: torch.Tensor,
    validation_scores: torch.Tensor,
    fit_split: Mapping[str, torch.Tensor],
    validation_split: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    fit_half_a = fit_scores.index_select(0, fit_split["half_a"])
    fit_half_b = fit_scores.index_select(0, fit_split["half_b"])
    validation_half_a = validation_scores.index_select(0, validation_split["half_a"])
    validation_half_b = validation_scores.index_select(0, validation_split["half_b"])
    cross = _angular_discrepancy(fit_scores, validation_scores)
    fit_within = _angular_discrepancy(fit_half_a, fit_half_b)
    validation_within = _angular_discrepancy(validation_half_a, validation_half_b)
    dominance = {
        name: {
            "fit_vs_validation": cross[name],
            "fit_within_reference": fit_within[name],
            "validation_within_reference": validation_within[name],
            "cross_exceeds_both_within_references": cross[name] > fit_within[name] and cross[name] > validation_within[name],
        }
        for name in cross
    }
    return {
        "fit": _angular_profile(fit_scores),
        "validation": _angular_profile(validation_scores),
        "fit_vs_validation": cross,
        "within_bank_references": {
            "fit_half_a_vs_fit_half_b": fit_within,
            "validation_half_a_vs_validation_half_b": validation_within,
        },
        "cross_bank_dominance": dominance,
        "cross_bank_dominates_all_angular_metrics": all(
            row["cross_exceeds_both_within_references"] for row in dominance.values()
        ),
        "radial_rank_quartiles": {
            "fit": _radial_quartile_angular(fit_scores),
            "validation": _radial_quartile_angular(validation_scores),
        },
    }


def _project_scores(
    q: torch.Tensor,
    mu_fit: torch.Tensor,
    W_99: torch.Tensor,
    lambda_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent = (q.to(torch.float64) - mu_fit).mm(W_99)
    scores = latent / torch.sqrt(lambda_values)
    _require(bool(torch.isfinite(latent).all()) and bool(torch.isfinite(scores).all()), "score projection is non-finite")
    return latent.contiguous(), scores.contiguous()


def _source_order_report(
    q_fit: torch.Tensor,
    q_validation: torch.Tensor,
    z_fit: torch.Tensor,
    z_validation: torch.Tensor,
    fit_labels: torch.Tensor,
    validation_labels: torch.Tensor,
    fit_indices: torch.Tensor,
    validation_indices: torch.Tensor,
) -> dict[str, Any]:
    records: list[tuple[int, int, torch.Tensor, torch.Tensor]] = []
    for q, z, labels, indices in (
        (q_fit, z_fit, fit_labels, fit_indices),
        (q_validation, z_validation, validation_labels, validation_indices),
    ):
        for row in range(q.shape[0]):
            records.append((int(indices[row]), int(labels[row]), q[row], z[row]))
    records.sort(key=lambda item: (item[1], item[0]))
    _require(len(records) == CLASS_COUNT * 100, "source-order record count changed")
    metrics: dict[str, list[float]] = {"r2": [], "q_norm": [], "q_energy": [], "score_norm": []}
    per_class: list[dict[str, Any]] = []
    for label in range(CLASS_COUNT):
        class_records = [record for record in records if record[1] == label]
        _require(len(class_records) == 100, f"source-order class {label} count changed")
        position = torch.arange(100, dtype=torch.float64)
        normalized_position = position / 99.0
        r2 = torch.stack([record[3].square().sum() for record in class_records]).to(torch.float64)
        q_norm = torch.stack([record[2].norm() for record in class_records]).to(torch.float64)
        q_energy = q_norm.square()
        score_norm = torch.stack([record[3].norm() for record in class_records]).to(torch.float64)
        class_metrics = {"r2": r2, "q_norm": q_norm, "q_energy": q_energy, "score_norm": score_norm}
        row: dict[str, Any] = {"label": label, "source_indices_first": int(class_records[0][0]), "source_indices_last": int(class_records[-1][0])}
        for name, values in class_metrics.items():
            row[name] = {
                "summary": _summary(values),
                "first_50_mean": float(values[:50].mean()),
                "last_50_mean": float(values[50:].mean()),
                "last_minus_first_50_mean": float(values[50:].mean() - values[:50].mean()),
                "position_association": _association(normalized_position, values),
            }
            metrics[name].extend(values.tolist())
        per_class.append(row)
    pooled_class_stratified: dict[str, Any] = {}
    pooled_position = torch.arange(100, dtype=torch.float64).repeat(CLASS_COUNT) / 99.0
    for name in metrics:
        pooled_values = torch.tensor(metrics[name], dtype=torch.float64)
        # The per-class values are reconstructed from the records so that the
        # pooled association removes only the registered class mean.
        residuals: list[torch.Tensor] = []
        for label in range(CLASS_COUNT):
            class_records = [record for record in records if record[1] == label]
            if name == "r2":
                values = torch.stack([record[3].square().sum() for record in class_records]).to(torch.float64)
            elif name == "q_norm":
                values = torch.stack([record[2].norm() for record in class_records]).to(torch.float64)
            elif name == "q_energy":
                values = torch.stack([record[2].square().sum() for record in class_records]).to(torch.float64)
            else:
                values = torch.stack([record[3].norm() for record in class_records]).to(torch.float64)
            residuals.append(values - values.mean())
        pooled_residual = torch.cat(residuals)
        pooled_class_stratified[name] = {
            "association_with_normalized_position": _association(pooled_position, pooled_residual),
            "pooled_value_summary": _summary(pooled_values),
        }
    r2_differences = [row["r2"]["last_minus_first_50_mean"] for row in per_class]
    r2_correlations = [row["r2"]["position_association"]["spearman"] for row in per_class]
    nonzero_difference_signs = {1 if value > 0.0 else -1 for value in r2_differences if value != 0.0}
    nonzero_correlation_signs = {1 if value > 0.0 else -1 for value in r2_correlations if value is not None and value != 0.0}
    systematic = (
        len(nonzero_difference_signs) == 1
        and len(nonzero_correlation_signs) == 1
        and next(iter(nonzero_difference_signs)) == next(iter(nonzero_correlation_signs))
    )
    return {
        "ordering": "within each class, source_index ascending; normalized position=0..99 divided by 99",
        "source_index_sha256": tensor_sha256(torch.tensor([record[0] for record in records], dtype=torch.int64)),
        "per_class": per_class,
        "pooled_class_stratified": pooled_class_stratified,
        "r2_last_minus_first_sign_counts": {
            "positive": sum(value > 0.0 for value in r2_differences),
            "negative": sum(value < 0.0 for value in r2_differences),
            "zero": sum(value == 0.0 for value in r2_differences),
        },
        "r2_position_spearman_sign_counts": {
            "positive": sum(value is not None and value > 0.0 for value in r2_correlations),
            "negative": sum(value is not None and value < 0.0 for value in r2_correlations),
            "zero_or_undefined": sum(value is None or value == 0.0 for value in r2_correlations),
        },
        "systematic_source_position_trend_supported": systematic,
        "membership_changed": False,
        "new_split_created": False,
    }


def _class_profiles(scores: torch.Tensor, labels: torch.Tensor) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label in range(CLASS_COUNT):
        values = scores[labels == label]
        radial = values.square().sum(dim=1)
        angular = _angular_profile(values)
        result.append(
            {
                "label": label,
                "count": int(values.shape[0]),
                "r2_mean": float(radial.mean()),
                "r2_variance": float(radial.var(unbiased=True)),
                "r2_p95": float(torch.quantile(radial, 0.95)),
                "angular_second_moment_relative_identity_frobenius": angular["second_moment_relative_identity_frobenius"],
                "angular_top_eigenvalue_mass": angular["second_moment_top_eigenvalue_mass"],
                "mean_unit_vector_norm": angular["mean_unit_vector_norm"],
            }
        )
    return result


def _class_reproducibility(
    fit_scores: torch.Tensor,
    validation_scores: torch.Tensor,
    fit_labels: torch.Tensor,
    validation_labels: torch.Tensor,
    fit_split: Mapping[str, torch.Tensor],
    validation_split: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    def metric_rows(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        rows = []
        for label in range(CLASS_COUNT):
            values = scores[labels == label]
            rows.append(
                torch.tensor(
                    [
                        float(values.square().sum(dim=1).mean()),
                        float(_angular_profile(values)["second_moment_relative_identity_frobenius"]),
                        float(_angular_profile(values)["second_moment_top_eigenvalue_mass"]),
                    ],
                    dtype=torch.float64,
                )
            )
        return torch.stack(rows)

    fit_rows = metric_rows(fit_scores, fit_labels)
    validation_rows = metric_rows(validation_scores, validation_labels)
    fit_a = metric_rows(fit_scores.index_select(0, fit_split["half_a"]), fit_labels.index_select(0, fit_split["half_a"]))
    fit_b = metric_rows(fit_scores.index_select(0, fit_split["half_b"]), fit_labels.index_select(0, fit_split["half_b"]))
    validation_a = metric_rows(validation_scores.index_select(0, validation_split["half_a"]), validation_labels.index_select(0, validation_split["half_a"]))
    validation_b = metric_rows(validation_scores.index_select(0, validation_split["half_b"]), validation_labels.index_select(0, validation_split["half_b"]))
    names = ("r2_mean", "angular_identity_deviation", "angular_top_eigenvalue_mass")
    rows: dict[str, Any] = {}
    for column, name in enumerate(names):
        cross = float((fit_rows[:, column] - validation_rows[:, column]).abs().mean())
        fit_ref = float((fit_a[:, column] - fit_b[:, column]).abs().mean())
        validation_ref = float((validation_a[:, column] - validation_b[:, column]).abs().mean())
        rows[name] = {
            "fit_vs_validation_mean_abs_delta": cross,
            "fit_within_reference_mean_abs_delta": fit_ref,
            "validation_within_reference_mean_abs_delta": validation_ref,
            "cross_exceeds_both_within_references": cross > fit_ref and cross > validation_ref,
            "fit_validation_class_vector_association": _association(fit_rows[:, column], validation_rows[:, column]),
        }
    return {
        "fit_class_profiles": _class_profiles(fit_scores, fit_labels),
        "validation_class_profiles": _class_profiles(validation_scores, validation_labels),
        "metrics": rows,
        "reproducible_local_signature_supported": all(
            not row["cross_exceeds_both_within_references"] for row in rows.values()
        ),
        "labels_used_as_observer_only": True,
    }


def _normalized_radial_result(
    fit_scores: torch.Tensor,
    validation_scores: torch.Tensor,
    fit_split: Mapping[str, torch.Tensor],
    validation_split: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    fit_normalized = _standardize_for_observation(fit_scores)
    validation_normalized = _standardize_for_observation(validation_scores)
    validation_half_a = validation_normalized.index_select(0, validation_split["half_a"])
    validation_half_b = validation_normalized.index_select(0, validation_split["half_b"])
    fit_half_a = fit_normalized.index_select(0, fit_split["half_a"])
    validation_profile = _radial_profile(validation_normalized)
    exact_marginal = bool(
        abs(validation_normalized.mean(dim=0)).max() < 1.0e-10
        and (validation_normalized.std(dim=0, unbiased=True) - 1.0).abs().max() < 1.0e-10
    )
    higher_order_residual = bool(
        validation_profile["variance_abs_error"] > 1.0e-8
        or any(value > 1.0e-8 for value in validation_profile["quantile_abs_error"].values())
    )
    return {
        "normalization": "z_tilde=(z-mean_val(z))/std_val(z), observer-only; no producer state changed",
        "fit": {"standardized": _radial_profile(fit_normalized), "angular": _angular_profile(fit_normalized)},
        "validation": {"standardized": validation_profile, "angular": _angular_profile(validation_normalized)},
        "within_bank_radial_references": {
            "fit_half_a_vs_fit_half_b": _radial_profile(fit_half_a),
            "validation_half_a_vs_validation_half_b": _radial_profile(validation_half_a),
            "validation_half_b": _radial_profile(validation_half_b),
        },
        "fit_validation_normalized_radial_discrepancy": {
            "r2_mean_abs_delta": abs(
                _radial_profile(fit_normalized)["r2_mean"] - validation_profile["r2_mean"]
            ),
            "r2_variance_abs_delta": abs(
                _radial_profile(fit_normalized)["r2_variance"] - validation_profile["r2_variance"]
            ),
        },
        "marginal_first_second_moment_removed_by_observer_normalization": exact_marginal,
        "higher_order_radial_residual_present_after_normalization": higher_order_residual,
        "normalization_does_not_authorize_new_model": True,
    }


def _select_family(
    source_order: Mapping[str, Any],
    normalized_radial: Mapping[str, Any],
    angular_raw: Mapping[str, Any],
    class_reproducibility: Mapping[str, Any],
) -> dict[str, Any]:
    source_supported = bool(source_order["systematic_source_position_trend_supported"])
    radial_residual = bool(normalized_radial["higher_order_radial_residual_present_after_normalization"])
    angular_dominant = bool(angular_raw["cross_bank_dominates_all_angular_metrics"])
    local_reproducible = bool(class_reproducibility["reproducible_local_signature_supported"])
    # A source-position trend is a confound/authority problem, not evidence for
    # a new score law.  It therefore blocks producer-family selection until a
    # separate stability/partition gate resolves it.
    candidates = {
        FAMILY_PARTITION: source_supported,
        FAMILY_RADIAL: (not source_supported) and radial_residual and not angular_dominant and not local_reproducible,
        FAMILY_LOCAL: (not source_supported) and radial_residual and angular_dominant and local_reproducible,
        FAMILY_DEPENDENCE: (not source_supported) and radial_residual and angular_dominant and not local_reproducible,
    }
    supported = [name for name, value in candidates.items() if value]
    selected = supported[0] if len(supported) == 1 and supported[0] != FAMILY_PARTITION else None
    dominant_mechanism = FAMILY_PARTITION if source_supported else (selected or None)
    return {
        "classification": "N3_NEXT_PRODUCER_FAMILY_HYPOTHESIS_SELECTED" if selected else CLASSIFICATION_UNRESOLVED,
        "selected_family": selected,
        "dominant_mechanism": dominant_mechanism,
        "candidate_status": candidates,
        "decision_basis": {
            "source_position_trend_supported": source_supported,
            "higher_order_radial_residual_after_marginal_normalization": radial_residual,
            "fit_validation_angular_discrepancy_dominates_within_references": angular_dominant,
            "class_local_signature_reproducible": local_reproducible,
            "selection_rule": "source-position confound blocks producer-family selection; otherwise select exactly one producer candidate only when its full conjunction is true",
        },
        "alternatives_rejected": {
            FAMILY_PARTITION: "requires a same-sign all-class source-position trend; radial residual is descriptive and does not enter partition selection",
            FAMILY_RADIAL: "requires residual radial structure with non-dominant angular discrepancy and no reproducible local signature",
            FAMILY_LOCAL: "requires residual radial structure, dominant global angular discrepancy, and reproducible class-local signature",
            FAMILY_DEPENDENCE: "requires residual radial structure, dominant global angular discrepancy, and no reproducible class-local signature",
            CLASSIFICATION_UNRESOLVED: "retained for a partition confound or whenever zero or multiple producer candidate conjunctions are true",
        },
        "no_family_fit_or_sampling": True,
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
    return {"branch": branch, "head": head, "dirty_paths": status}


def _write_report(repo_root: Path, output_dir: Path, report: Mapping[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        target = temporary / REPORT_NAME
        atomic_write_new_json(target, report, artifact_kind="N3 score geometry attribution report")
        _require(sorted(path.name for path in temporary.iterdir()) == [REPORT_NAME], "report layout changed")
        if output_dir.exists():
            raise FileExistsError(f"write-once output appeared: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / REPORT_NAME


def execute(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 authority branch")
    _require(not git["dirty_paths"], f"worktree must be clean: {git['dirty_paths']}")
    prior = _load_previous_consumer(repo_root)
    authorities = dict(prior._load_and_verify_inputs(repo_root))
    q_fit = authorities.pop("q_fit")
    q_validation = authorities.pop("q_validation")
    fit_labels = authorities.pop("fit_labels")
    validation_labels = authorities.pop("validation_labels")
    fit_positions = authorities.pop("fit_positions")
    validation_positions = authorities.pop("validation_positions")
    mu_fit = authorities.pop("mu_fit")
    W_99 = authorities.pop("W_99")
    lambda_values = authorities.pop("lambda_values")
    # The frozen verifier validates source-index hashes; load those same tensors
    # again only to observe the registered source-index ordering.
    _, fit_labels_again, fit_indices, fit_positions_again, _ = prior._load_bank(
        repo_root,
        prior.FIT_BANK,
        prior.EXPECTED_FIT_BANK_SHA256,
        "OLS_PRODUCER_FIT_BANK_V1",
        prior.EXPECTED_FIT_R3_SHA256,
        prior.EXPECTED_FIT_ROW_POSITION_SHA256,
        prior.EXPECTED_FIT_SOURCE_INDEX_SHA256,
    )
    _, validation_labels_again, validation_indices, validation_positions_again, _ = prior._load_bank(
        repo_root,
        prior.VALIDATION_BANK,
        prior.EXPECTED_VALIDATION_BANK_SHA256,
        "OLS_PRODUCER_INTERNAL_VALIDATION_BANK_V1",
        prior.EXPECTED_VALIDATION_R3_SHA256,
        prior.EXPECTED_VALIDATION_ROW_POSITION_SHA256,
        prior.EXPECTED_VALIDATION_SOURCE_INDEX_SHA256,
    )
    _require(torch.equal(fit_labels, fit_labels_again) and torch.equal(validation_labels, validation_labels_again), "reloaded labels changed")
    _require(torch.equal(fit_positions, fit_positions_again) and torch.equal(validation_positions, validation_positions_again), "reloaded positions changed")
    fit_latent, z_fit = _project_scores(q_fit, mu_fit, W_99, lambda_values)
    validation_latent, z_validation = _project_scores(q_validation, mu_fit, W_99, lambda_values)
    fit_split = prior._class_balanced_split(fit_labels, fit_positions)
    validation_split = prior._class_balanced_split(validation_labels, validation_positions)
    source_order = _source_order_report(
        q_fit,
        q_validation,
        z_fit,
        z_validation,
        fit_labels,
        validation_labels,
        fit_indices,
        validation_indices,
    )
    raw_decomposition = {
        "fit": _radial_mean_decomposition(z_fit, fit_latent, lambda_values),
        "validation": _radial_mean_decomposition(z_validation, validation_latent, lambda_values),
    }
    normalized_radial = _normalized_radial_result(z_fit, z_validation, fit_split, validation_split)
    angular_raw = _angular_structure(z_fit, z_validation, fit_split, validation_split)
    angular_normalized = _angular_structure(
        _standardize_for_observation(z_fit),
        _standardize_for_observation(z_validation),
        fit_split,
        validation_split,
    )
    class_reproducibility = _class_reproducibility(
        z_fit,
        z_validation,
        fit_labels,
        validation_labels,
        fit_split,
        validation_split,
    )
    selection = _select_family(source_order, normalized_radial, angular_raw, class_reproducibility)
    input_paths = (
        prior.FIT_BANK,
        prior.VALIDATION_BANK,
        prior.PRODUCER_STATE,
        prior.PRODUCER_MANIFEST,
        prior.MODEL_ORDER_RESULT,
        prior.FAILED_VALIDATION_REPORT,
        prior.SOURCE_MANIFEST,
        prior.PARTITION_MANIFEST,
        prior.FIREWALL_MANIFEST,
    )
    hashes_before = {str(path): file_sha256(repo_root / path) for path in input_paths}
    _require(hashes_before[str(prior.FIT_BANK)] == prior.EXPECTED_FIT_BANK_SHA256, "fit bank hash changed")
    _require(hashes_before[str(prior.VALIDATION_BANK)] == prior.EXPECTED_VALIDATION_BANK_SHA256, "validation bank hash changed")
    hashes_after = {str(path): file_sha256(repo_root / path) for path in input_paths}
    _require(hashes_before == hashes_after, "frozen input changed during attribution")
    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "classification": selection["classification"],
        "selected_next_family": selection["selected_family"],
        "semantic_inputs": {
            "producer_data": "OLS_PRODUCER_DATA_V1",
            "model_order": "N3_LOW_RANK_MODEL_ORDER_SELECTED",
            "producer_baseline": "N3_AMPLITUDE_PRODUCER_BASELINE_FITTED",
            "failed_validation": "N3_AMPLITUDE_PRODUCER_BASELINE_NOT_SUPPORTED",
            "prior_attribution": "N3_PRODUCER_MISMATCH_ATTRIBUTION_MIXED",
            "coordinate": "R2_AMPLITUDE",
            "score_definition": "z_j=(q-mu_fit)W_99[:,j]/sqrt(lambda_j)",
            "lambda_denominator": "n_fit-1=499",
            "labels": "diagnostic observer only",
            "authority_summary": authorities,
        },
        "radial_mean_decomposition": raw_decomposition,
        "source_index_partition_position_analysis": source_order,
        "diagnostic_moment_normalized_radial_result": normalized_radial,
        "raw_radial_structure": {
            "fit": _radial_profile(z_fit),
            "validation": _radial_profile(z_validation),
            "interpretation": "descriptive radial moment/tail evidence; no new score law was fitted",
        },
        "angular_structure": {
            "raw_frozen_scores": angular_raw,
            "moment_normalized_observer_scores": angular_normalized,
            "interpretation": "descriptive unit-vector geometry; eigensummaries are not rank selection",
        },
        "radial_angular_coupling": {
            "raw_frozen_scores": angular_raw["radial_rank_quartiles"],
            "moment_normalized_observer_scores": angular_normalized["radial_rank_quartiles"],
            "partition": "fixed within-bank radial rank quartiles, descriptive only; no producer split or membership change",
        },
        "class_local_observer_result": class_reproducibility,
        "fit_validation_reproducibility": {
            "raw_angular": angular_raw["cross_bank_dominance"],
            "moment_normalized_angular": angular_normalized["cross_bank_dominance"],
            "local_class_signature": class_reproducibility["metrics"],
            "source_order_observer": source_order["systematic_source_position_trend_supported"],
        },
        "mechanism_attribution": selection,
        "claims": {
            "score_geometry_attribution_resolved": selection["dominant_mechanism"] is not None,
            "producer_family_hypothesis_selected": selection["selected_family"] is not None,
            "next_family_is_model_assumption_only": selection["selected_family"] is not None,
            "labels_not_used_as_producer_inputs": True,
            "partition_membership_unchanged": True,
            "rank_selected": False,
            "producer_fitted": False,
            "score_distribution_scientifically_supported": False,
            "novel_ols_valid": False,
            "generation_readiness": "not_ready",
            "training_authorized": False,
            "formal_promotion": False,
        },
        "input_integrity": {
            "frozen_input_hashes_before": hashes_before,
            "frozen_input_hashes_after": hashes_after,
            "frozen_inputs_unchanged": True,
            "previous_consumer_sha256": EXPECTED_PREVIOUS_CONSUMER_SHA256,
            "source_partition_firewall_verified_by_frozen_consumer": True,
        },
        "git": {
            "branch": git["branch"],
            "head": git["head"],
            "dirty_paths": git["dirty_paths"],
            "consumer_script": _relative(repo_root, Path(__file__).resolve()),
            "consumer_script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "not_executed": [
            "new_split_or_membership_change",
            "producer_refit",
            "rank_reselection",
            "score_distribution_fit",
            "score_sampling",
            "mixture_fitting",
            "Student_t_fitting",
            "ICA_or_flow_fitting",
            "label_conditioned_producer",
            "novel_complex_OLS",
            "optical_forward",
            "evaluator",
            "CUDA",
            "optimizer",
            "backward",
            "training",
            "N4",
        ],
    }
    artifact = _write_report(repo_root, output_dir, report)
    return {
        "status": report["status"],
        "classification": report["classification"],
        "selected_next_family": report["selected_next_family"],
        "artifact": _relative(repo_root, artifact),
        "artifact_sha256": file_sha256(artifact),
        "git_head": git["head"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/20260908_v2/runs/"
            "lightgen_n3_score_geometry_attribution_v1_attempt_01"
        ),
        help="fresh write-once attribution output directory",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
