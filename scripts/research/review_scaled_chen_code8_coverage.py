"""Review frozen Code8 within-class OLS coverage without scientific forward."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ldm_project.v2.evidence import (
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
)
from ldm_project.v2.application.research import optical_manifold as base


TASK_ID = "H-LOMA-GEN-S3-C8"
HYPOTHESIS = "CODE8_WITHIN_CLASS_MANIFOLD_COVERAGE"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_code8_coverage_review_v1"
REPORT_SCHEMA = "lightgen_scaled_chen_code8_coverage_review_report.v1"
MANIFEST_SCHEMA = "lightgen_scaled_chen_code8_coverage_review.v1"
PANEL_SCHEMA = "lightgen_scaled_chen_controller_panel_identity.v1"
S3_SCHEMA = "lightgen_scaled_chen_controller_to_ols_writability_audit.v1"
S1_SCHEMA = "lightgen_scaled_chen_ols_semantic_embeddings.v1"
GEOMETRY_SCHEMA = "lightgen_scaled_chen_controller_to_ols_semantic_geometry_alignment.v1"
PORT_COUNT = 100
CLASS_COUNT = 10
CODE_DIM = 8
EFFECTIVE_RANK_RELATIVE_THRESHOLD = 1.0e-6

S3_DIR = Path(
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_controller_to_ols_writability_audit_v1_attempt_03"
)
S3_MANIFEST = S3_DIR / "s3_manifest.json"
S3_MANIFEST_SHA256 = "0779d8d0ffb0a9958064bb3c05d451058cdd7a670a732d6ae1d4ba49c11eb5e6"
S3_STATE_SHA256 = "45f9b2b18c0cc67bf784ee206059d97164abaab08f0304892c368f53d218b19a"
S3_PANEL_IDENTITY_SHA256 = "8a9dd1c7e3e2b674a152e6c96ac8b47976ff1f8d27428a18477e89795b940f47"
S3_CSAR_SHA256 = "57942dcd4a425960b637bff99299dd6e1260198a22456fa5ef1ca133600302a8"

S1_DIR = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01"
)
S1_MANIFEST = S1_DIR / "semantic_manifest.json"
S1_MANIFEST_SHA256 = "b0199040926618b17373edc4f167cc8bc4788f39f64de174e1a3fb225781833e"

GEOMETRY_DIR = Path(
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_controller_to_ols_semantic_geometry_alignment_v1_attempt_02"
)
GEOMETRY_MANIFEST = GEOMETRY_DIR / "geometry_alignment_manifest.json"
GEOMETRY_MANIFEST_SHA256 = "1112b8331b3a7a31cf4814ad50c6b3ea8b60217771ef222b7182e553d8284682"

DEFAULT_CONFIG_PATH = Path(
    "configs/research/lightgen_scaled_chen_code8_coverage_review_v1.yaml"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_code8_coverage_review_v1_attempt_01"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _repo_root() / path


def _relative(value: str | Path) -> str:
    path = _resolve(value)
    try:
        return path.relative_to(_repo_root()).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not a mapping: {path}")
    return payload


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _assert_exact_tree(directory: Path, expected: set[str], name: str) -> None:
    actual = {
        item.relative_to(directory).as_posix()
        for item in directory.rglob("*")
        if item.is_file()
    }
    if actual != expected:
        raise ValueError(f"{name} tree is not exact: {sorted(actual)}")
    if any(
        Path(item).name.startswith(".")
        or ".tmp" in item
        or item.lower().endswith((".tif", ".tiff"))
        for item in actual
    ):
        raise ValueError(f"{name} contains hidden, temporary, or TIFF files")


def _verify_manifest_tree(
    directory: Path,
    manifest_name: str,
    manifest: Mapping[str, Any],
    name: str,
) -> dict[str, Path]:
    artifacts = _mapping(manifest.get("artifacts"), f"{name} artifacts")
    _assert_exact_tree(directory, set(artifacts) | {manifest_name}, name)
    paths: dict[str, Path] = {}
    for artifact_name, reference_value in artifacts.items():
        reference = _mapping(reference_value, f"{name} artifact {artifact_name}")
        path = directory / artifact_name
        if (
            reference.get("path") != artifact_name
            or not isinstance(reference.get("sha256"), str)
            or file_sha256(path) != reference["sha256"]
            or path.stat().st_size != int(reference.get("size_bytes", -1))
        ):
            raise ValueError(f"{name} artifact changed: {artifact_name}")
        paths[artifact_name] = path
    return paths


def _validate_config(payload: Mapping[str, Any]) -> None:
    if payload.get("schema") != CONFIG_SCHEMA or payload.get("task_id") != TASK_ID:
        raise ValueError("Code8 review config identity is invalid")
    if payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("Code8 review hypothesis is invalid")
    review = _mapping(payload.get("review"), "review")
    expected_review = {
        "device": "cpu",
        "deterministic": True,
        "controller_forward": False,
        "optical_forward": False,
        "jacobian_recompute": False,
        "decoder": False,
        "training": False,
        "checkpoint_write": False,
        "new_panel": False,
        "new_codes": False,
        "new_transform": False,
        "new_hard_gate": False,
    }
    if dict(review) != expected_review:
        raise ValueError("Code8 review execution boundary is not locked")
    governance = _mapping(payload.get("governance"), "governance")
    expected_governance = {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }
    if dict(governance) != expected_governance:
        raise ValueError("Code8 review governance is not locked")
    output = _mapping(payload.get("output"), "output")
    if dict(output) != {
        "directory": _relative(DEFAULT_OUTPUT_PATH),
        "write_once": True,
        "json_only": True,
    }:
        raise ValueError("Code8 review output contract is not locked")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[Path, dict[str, Any], str, str]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Code8 review config must be a mapping")
    _validate_config(payload)
    return target, payload, file_sha256(target), canonical_sha256(payload)


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.to(torch.float64).reshape(-1)
    if values.numel() == 0 or not bool(torch.isfinite(values).all()):
        raise ValueError("summary values must be finite and nonempty")
    quantiles = torch.quantile(
        values,
        torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], dtype=torch.float64),
    )
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=values.numel() > 1)),
        "finite": True,
    }


def _covariance(values: torch.Tensor) -> dict[str, Any]:
    values = values.to(torch.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not bool(torch.isfinite(values).all()):
        raise ValueError("covariance input is invalid")
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / (values.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0.0)
    trace = float(eigenvalues.sum())
    leading = float(eigenvalues[0]) if eigenvalues.numel() else 0.0
    effective_rank = int(
        (eigenvalues >= leading * EFFECTIVE_RANK_RELATIVE_THRESHOLD).sum()
    ) if leading > 0.0 else 0
    normalized = eigenvalues / max(trace, 1.0e-30)
    return {
        "sample_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "trace_total_variance": trace,
        "effective_rank": effective_rank,
        "effective_rank_relative_threshold": EFFECTIVE_RANK_RELATIVE_THRESHOLD,
        "eigenvalues_descending": [float(item) for item in eigenvalues],
        "normalized_eigenvalues_descending": [float(item) for item in normalized],
        "cumulative_explained_variance": [
            float(item) for item in torch.cumsum(normalized, dim=0)
        ],
        "covariance": covariance.tolist(),
        "finite": True,
    }


def _correlation(values: torch.Tensor) -> list[list[float]]:
    values = values.to(torch.float64)
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    scale = torch.sqrt(torch.diag(covariance).clamp_min(1.0e-30))
    correlation = covariance / (scale[:, None] * scale[None, :])
    return correlation.tolist()


def _pairwise(values: torch.Tensor) -> dict[str, Any]:
    return _summary(torch.pdist(values.to(torch.float64)))


def _rank(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float64).reshape(-1)
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    ranks = torch.empty_like(values)
    start = 0
    while start < sorted_values.numel():
        end = start + 1
        while end < sorted_values.numel() and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.to(torch.float64).reshape(-1)
    right = right.to(torch.float64).reshape(-1)
    if left.numel() < 2:
        return None
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left_centered, right_centered) / denominator)


def _binned_trend(x: torch.Tensor, y: torch.Tensor, bins: int = 5) -> list[dict[str, Any]]:
    x = x.to(torch.float64).reshape(-1)
    y = y.to(torch.float64).reshape(-1)
    edges = torch.quantile(
        x,
        torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64),
    )
    result: list[dict[str, Any]] = []
    for index in range(bins):
        if index == bins - 1:
            mask = (x >= edges[index]) & (x <= edges[index + 1])
        else:
            mask = (x >= edges[index]) & (x < edges[index + 1])
        if not bool(mask.any()):
            result.append({"bin": index, "count": 0})
            continue
        result.append(
            {
                "bin": index,
                "count": int(mask.sum()),
                "x_min": float(x[mask].min()),
                "x_max": float(x[mask].max()),
                "x_mean": float(x[mask].mean()),
                "y_mean": float(y[mask].mean()),
                "y_median": float(y[mask].median()),
            }
        )
    return result


def _panel_report(base_codes: torch.Tensor, panel: Mapping[str, Any]) -> dict[str, Any]:
    base_codes = base_codes.to(torch.float64)
    per_dimension: dict[str, Any] = {}
    for dimension in range(CODE_DIM):
        per_dimension[str(dimension)] = _summary(base_codes[:, dimension])
    covariance = _covariance(base_codes)
    per_dim_ranges = [
        per_dimension[str(dimension)]["max"] - per_dimension[str(dimension)]["min"]
        for dimension in range(CODE_DIM)
    ]
    observed_both_sides = all(
        per_dimension[str(dimension)]["min"] < 0.0
        and per_dimension[str(dimension)]["max"] > 0.0
        for dimension in range(CODE_DIM)
    )
    unique_count = int(torch.unique(base_codes, dim=0).shape[0])
    classification = (
        "CODE_PANEL_DOMAIN_COVERAGE_ADEQUATE"
        if observed_both_sides and unique_count == int(base_codes.shape[0])
        else "CODE_PANEL_DOMAIN_COVERAGE_LIMITED"
    )
    return {
        "schema": "lightgen_scaled_chen_code8_panel_coverage.v1",
        "authority": {
            "distribution": panel["distribution"],
            "seed": int(panel["seed"]),
            "base_code_count": int(panel["base_code_count"]),
            "code_dim": int(panel["code_dim"]),
            "same_base_codes_across_labels": bool(panel["same_base_codes_across_labels"]),
            "strict_covering_proof": False,
        },
        "per_dimension": per_dimension,
        "covariance": covariance,
        "correlation": _correlation(base_codes),
        "covariance_condition_number": (
            float(covariance["eigenvalues_descending"][0])
            / max(float(covariance["eigenvalues_descending"][-1]), 1.0e-30)
        ),
        "pairwise_code_distance": _pairwise(base_codes),
        "code_vector_norm": _summary(base_codes.norm(dim=1)),
        "observed_dimension_ranges": per_dim_ranges,
        "unique_code_count": unique_count,
        "classification": classification,
        "classification_scope": "frozen S3 panel only; not all of [-1,1]^8",
        "finite": True,
    }


def _class_coverage(
    controller: torch.Tensor,
    controller_labels: torch.Tensor,
    real: torch.Tensor,
    real_labels: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    per_class: dict[str, Any] = {}
    shape_per_class: dict[str, Any] = {}
    trace_ratios: list[float] = []
    q50_ratios: list[float] = []
    q95_ratios: list[float] = []
    for label in range(CLASS_COUNT):
        controller_class = controller[controller_labels == label]
        real_class = real[real_labels == label]
        controller_center = controller_class.mean(dim=0)
        real_center = real_class.mean(dim=0)
        controller_radius = (controller_class - controller_center).norm(dim=1)
        real_radius = (real_class - real_center).norm(dim=1)
        controller_covariance = _covariance(controller_class)
        real_covariance = _covariance(real_class)
        trace_ratio = controller_covariance["trace_total_variance"] / max(
            real_covariance["trace_total_variance"], 1.0e-30
        )
        controller_q50 = _summary(controller_radius)["q50"]
        real_q50 = _summary(real_radius)["q50"]
        controller_q95 = _summary(controller_radius)["q95"]
        real_q95 = _summary(real_radius)["q95"]
        q50_ratio = controller_q50 / max(real_q50, 1.0e-30)
        q95_ratio = controller_q95 / max(real_q95, 1.0e-30)
        trace_ratios.append(trace_ratio)
        q50_ratios.append(q50_ratio)
        q95_ratios.append(q95_ratio)
        per_class[str(label)] = {
            "controller_count": int(controller_class.shape[0]),
            "real_count": int(real_class.shape[0]),
            "controller_center": controller_center.tolist(),
            "real_center": real_center.tolist(),
            "controller_radius": _summary(controller_radius),
            "real_radius": _summary(real_radius),
            "controller_covariance": controller_covariance,
            "real_covariance": real_covariance,
            "trace_ratio": trace_ratio,
            "q50_radius_ratio": q50_ratio,
            "q95_radius_ratio": q95_ratio,
            "finite": True,
        }
        controller_covariance_matrix = torch.tensor(
            controller_covariance["covariance"], dtype=torch.float64
        )
        real_covariance_matrix = torch.tensor(
            real_covariance["covariance"], dtype=torch.float64
        )
        controller_shape = controller_covariance_matrix / max(
            controller_covariance["trace_total_variance"], 1.0e-30
        )
        real_shape = real_covariance_matrix / max(
            real_covariance["trace_total_variance"], 1.0e-30
        )
        shape_difference = (controller_shape - real_shape).norm()
        real_shape_norm = real_shape.norm()
        controller_spectrum = torch.tensor(
            controller_covariance["normalized_eigenvalues_descending"],
            dtype=torch.float64,
        )
        real_spectrum = torch.tensor(
            real_covariance["normalized_eigenvalues_descending"],
            dtype=torch.float64,
        )
        shape_per_class[str(label)] = {
            "normalized_covariance_frobenius_difference": float(shape_difference),
            "relative_normalized_covariance_frobenius_difference": float(
                shape_difference / real_shape_norm.clamp_min(1.0e-30)
            ),
            "normalized_spectrum_l1_difference": float(
                (controller_spectrum - real_spectrum).abs().sum()
            ),
            "controller_spectral_concentration_top1": float(controller_spectrum[0]),
            "real_spectral_concentration_top1": float(real_spectrum[0]),
            "controller_cumulative_top8": float(controller_spectrum[:8].sum()),
            "real_cumulative_top8": float(real_spectrum[:8].sum()),
            "finite": True,
        }
    ratios = {
        "trace_ratio": {
            "per_class": trace_ratios,
            "median": float(torch.tensor(trace_ratios, dtype=torch.float64).median()),
            "min": min(trace_ratios),
            "max": max(trace_ratios),
            "class_order": list(range(CLASS_COUNT)),
        },
        "q50_radius_ratio": {
            "per_class": q50_ratios,
            "median": float(torch.tensor(q50_ratios, dtype=torch.float64).median()),
            "min": min(q50_ratios),
            "max": max(q50_ratios),
            "class_order": list(range(CLASS_COUNT)),
        },
        "q95_radius_ratio": {
            "per_class": q95_ratios,
            "median": float(torch.tensor(q95_ratios, dtype=torch.float64).median()),
            "min": min(q95_ratios),
            "max": max(q95_ratios),
            "class_order": list(range(CLASS_COUNT)),
        },
    }
    return (
        {
            "schema": "lightgen_scaled_chen_code8_within_class_coverage.v1",
            "controller_split": "S3 confirmation panel",
            "real_split": "S1 confirmation held-out reference",
            "per_class": per_class,
            "ratios": ratios,
            "finite": True,
        },
        {
            "schema": "lightgen_scaled_chen_code8_covariance_shape.v1",
            "per_class": shape_per_class,
            "finite": True,
        },
    )


def _jacobian_report(jacobian: Mapping[str, Any]) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    median_values: list[float] = []
    for label, value in _mapping(jacobian["per_class"], "Jacobian per_class").items():
        singular = torch.tensor(value["singular_values"], dtype=torch.float64)
        if singular.shape != (CODE_DIM,) or not bool(torch.isfinite(singular).all()):
            raise ValueError("saved Jacobian singular values are invalid")
        if bool((singular <= 0.0).any()):
            geometric_mean = None
        else:
            geometric_mean = float(torch.exp(torch.log(singular).mean()))
        smallest_largest = float(singular[-1] / singular[0].clamp_min(1.0e-30))
        median_value = float(singular.median())
        median_values.append(median_value)
        per_class[str(label)] = {
            "singular_values": singular.tolist(),
            "median_singular_value": median_value,
            "geometric_mean_singular_value": geometric_mean,
            "smallest_over_largest": smallest_largest,
            "frobenius_norm": float(torch.linalg.vector_norm(singular)),
            "nuclear_norm": float(singular.sum()),
            "effective_rank": int(value["effective_rank"]),
            "classification": value["classification"],
            "finite": True,
        }
    aggregate = torch.tensor(jacobian["aggregate_singular_values"], dtype=torch.float64)
    return {
        "schema": "lightgen_scaled_chen_code8_jacobian_magnitude.v1",
        "quantity": jacobian["quantity"],
        "source": "saved S3 jacobian_metrics.json; no Jacobian recomputation",
        "per_class": per_class,
        "aggregate_singular_values": aggregate.tolist(),
        "class_median_singular_value_summary": _summary(
            torch.tensor(median_values, dtype=torch.float64)
        ),
        "rank_authority": {
            "aggregate_classification": jacobian["aggregate_classification"],
            "effective_rank_threshold": jacobian["effective_rank_threshold"],
        },
        "JACOBIAN_MAGNITUDE_ROLE": "SUPPORTED_DESCRIPTIVE",
        "full_jacobian_matrix_saved": False,
        "LOCAL_TANGENT_ALIGNMENT": "UNRESOLVED",
        "finite": True,
    }


def _code_norm_excursion(
    codes: torch.Tensor,
    labels: torch.Tensor,
    states: torch.Tensor,
) -> dict[str, Any]:
    code_norm = codes.norm(dim=1)
    excursion = torch.empty(codes.shape[0], dtype=torch.float64)
    per_class: dict[str, Any] = {}
    for label in range(CLASS_COUNT):
        mask = labels == label
        center = states[mask].mean(dim=0)
        class_excursion = (states[mask] - center).norm(dim=1)
        excursion[mask] = class_excursion
        per_class[str(label)] = {
            "spearman_code_norm_vs_ols_excursion": _spearman(code_norm[mask], class_excursion),
            "binned_descriptive_trend": _binned_trend(code_norm[mask], class_excursion),
            "code_norm": _summary(code_norm[mask]),
            "ols_excursion": _summary(class_excursion),
        }
    return {
        "schema": "lightgen_scaled_chen_code8_norm_ols_excursion.v1",
        "split": "S3 confirmation panel",
        "global_spearman": _spearman(code_norm, excursion),
        "global_binned_descriptive_trend": _binned_trend(code_norm, excursion),
        "per_class": per_class,
        "no_control_law_fit": True,
        "finite": True,
    }


def _pairwise_map_diagnostic(
    codes: torch.Tensor,
    labels: torch.Tensor,
    states: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    code_distances: list[torch.Tensor] = []
    state_distances: list[torch.Tensor] = []
    per_class: dict[str, Any] = {}
    for label in range(CLASS_COUNT):
        mask = labels == label
        code_distance = torch.pdist(codes[mask].to(torch.float64))
        state_distance = torch.pdist(states[mask].to(torch.float64))
        code_distances.append(code_distance)
        state_distances.append(state_distance)
        bins = _binned_trend(code_distance, state_distance)
        ratio = state_distance / code_distance.clamp_min(1.0e-30)
        per_class[str(label)] = {
            "pair_count": int(code_distance.numel()),
            "spearman_code_distance_vs_ols_distance": _spearman(
                code_distance, state_distance
            ),
            "code_distance": _summary(code_distance),
            "ols_distance": _summary(state_distance),
            "ols_over_code_distance": _summary(ratio),
            "distance_bins": bins,
            "last_bin_over_first_bin_ols_mean": (
                bins[-1]["y_mean"] / max(bins[0]["y_mean"], 1.0e-30)
                if bins[0].get("count", 0) and bins[-1].get("count", 0)
                else None
            ),
        }
    all_code = torch.cat(code_distances)
    all_state = torch.cat(state_distances)
    global_bins = _binned_trend(all_code, all_state)
    first_ratio = global_bins[0].get("y_mean") if global_bins[0].get("count", 0) else None
    last_ratio = global_bins[-1].get("y_mean") if global_bins[-1].get("count", 0) else None
    ratio_bins = _binned_trend(all_code, all_state / all_code.clamp_min(1.0e-30))
    declining = bool(
        first_ratio is not None
        and last_ratio is not None
        and ratio_bins[0].get("y_mean") is not None
        and ratio_bins[-1].get("y_mean") is not None
        and ratio_bins[-1]["y_mean"] < ratio_bins[0]["y_mean"]
    )
    global_report = {
        "schema": "lightgen_scaled_chen_code8_global_map_confinement.v1",
        "pair_count": int(all_code.numel()),
        "spearman_code_distance_vs_ols_distance": _spearman(all_code, all_state),
        "code_distance": _summary(all_code),
        "ols_distance": _summary(all_state),
        "ols_over_code_distance": _summary(all_state / all_code.clamp_min(1.0e-30)),
        "distance_bins": global_bins,
        "gain_ratio_bins": ratio_bins,
        "saturation_tendency": (
            "DECLINING_OLS_PER_CODE_DISTANCE_WITH_CODE_DISTANCE"
            if declining
            else "NO_CLEAR_DECLINING_OLS_PER_CODE_DISTANCE"
        ),
        "per_class": per_class,
        "descriptive_only": True,
        "finite": True,
    }
    return global_report, {"code_distances": all_code, "state_distances": all_state}


def _folding_report(code_distances: torch.Tensor, state_distances: torch.Tensor) -> dict[str, Any]:
    code_q95 = float(
        torch.quantile(code_distances, torch.tensor(0.95, dtype=torch.float64))
    )
    state_q05 = float(
        torch.quantile(state_distances, torch.tensor(0.05, dtype=torch.float64))
    )
    candidate = (code_distances >= code_q95) & (state_distances <= state_q05)
    exact_duplicates = int((state_distances == 0.0).sum())
    return {
        "schema": "lightgen_scaled_chen_code8_folding_diagnostic.v1",
        "far_code_distance_q95": code_q95,
        "near_ols_distance_q05": state_q05,
        "candidate_pair_count": int(candidate.sum()),
        "exact_ols_duplicate_pair_count": exact_duplicates,
        "classification": (
            "CODE_TO_OLS_GLOBAL_FOLDING_CANDIDATE_SUPPORTED"
            if bool(candidate.any()) and exact_duplicates == 0
            else "CODE_TO_OLS_GLOBAL_FOLDING_CANDIDATE_NOT_SUPPORTED"
        ),
        "topology_folding_strict_proof": False,
        "descriptive_only": True,
        "finite": True,
    }


def _dimension_report(
    base_codes: torch.Tensor,
    code_effect: Mapping[str, Any],
) -> dict[str, Any]:
    sensitivity_by_dimension = [[] for _ in range(CODE_DIM)]
    for label in range(CLASS_COUNT):
        values = _mapping(code_effect["per_class"][str(label)], f"code effect {label}")
        sensitivity = values["per_dimension_sensitivity"]
        if len(sensitivity) != CODE_DIM:
            raise ValueError("per-dimension sensitivity has the wrong shape")
        for dimension, value in enumerate(sensitivity):
            sensitivity_by_dimension[dimension].append(float(value))
    per_dimension: dict[str, Any] = {}
    for dimension in range(CODE_DIM):
        class_values = torch.tensor(sensitivity_by_dimension[dimension], dtype=torch.float64)
        per_dimension[str(dimension)] = {
            "code_variation": _summary(base_codes[:, dimension]),
            "existing_sensitivity_across_classes": _summary(class_values),
            "minimum_existing_sensitivity": float(class_values.min()),
            "maximum_existing_sensitivity": float(class_values.max()),
            "finite": True,
        }
    classification = (
        "ALL_8_CODE_DIMENSIONS_ACTIVELY_UTILIZED"
        if all(
            item["code_variation"]["std"] > 0.0
            and item["minimum_existing_sensitivity"] > 0.0
            for item in per_dimension.values()
        )
        else "CODE_DIMENSION_UTILIZATION_IMBALANCED"
    )
    return {
        "schema": "lightgen_scaled_chen_code8_dimension_utilization.v1",
        "per_dimension": per_dimension,
        "classification": classification,
        "feature_selection_or_deletion": False,
        "finite": True,
    }


def _mechanism_report(
    panel: Mapping[str, Any],
    coverage: Mapping[str, Any],
    shape: Mapping[str, Any],
    jacobian: Mapping[str, Any],
    confinement: Mapping[str, Any],
    geometry_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    panel_classification = panel["classification"]
    trace_ratios = coverage["ratios"]["trace_ratio"]["per_class"]
    shape_differences = [
        value["relative_normalized_covariance_frobenius_difference"]
        for value in shape["per_class"].values()
    ]
    c1 = (
        "NOT_SUPPORTED"
        if panel_classification == "CODE_PANEL_DOMAIN_COVERAGE_ADEQUATE"
        else panel_classification.replace("CODE_PANEL_", "").replace("_", " ")
    )
    c2 = "SUPPORTED" if all(value < 1.0 for value in trace_ratios) else "UNRESOLVED"
    c3 = "UNRESOLVED"
    c4 = (
        "SUPPORTED"
        if confinement["saturation_tendency"]
        == "DECLINING_OLS_PER_CODE_DISTANCE_WITH_CODE_DISTANCE"
        else "NOT_SUPPORTED"
    )
    c5 = (
        "SUPPORTED"
        if geometry_manifest.get("secondary_mechanism")
        == "S3_GEOMETRY_HIGHER_ORDER_MISMATCH_PRIMARY"
        and max(shape_differences) > 0.0
        else "UNRESOLVED"
    )
    c6 = (
        "SUPPORTED"
        if max(trace_ratios) > min(trace_ratios)
        and max(trace_ratios) / max(min(trace_ratios), 1.0e-30) > 1.0
        else "UNRESOLVED"
    )
    mechanisms = {
        "C1_PANEL_UNDERCOVERAGE": c1,
        "C2_CODE_COVERAGE_GAIN_COMPRESSION": c2,
        "C3_LOCAL_JACOBIAN_GAIN_COMPRESSION": c3,
        "C4_GLOBAL_CONTROLLER_MAP_CONFINEMENT": c4,
        "C5_CODE_TANGENT_MISALIGNMENT": c5,
        "C6_CLASS_CONDITIONAL_COVERAGE_COMPRESSION": c6,
    }
    if c2 == "SUPPORTED" and c5 == "SUPPORTED":
        final = "CODE8_COVERAGE_MIXED_GAIN_AND_GEOMETRY"
        next_action = "PLAN_CONTROLLER_SEMANTIC_GEOMETRY_TRAINING_AUTHORITY"
        strongest = "C5_CODE_TANGENT_MISALIGNMENT"
    elif c4 == "SUPPORTED":
        final = "CODE8_COVERAGE_GLOBAL_MAP_CONFINEMENT_PRIMARY"
        next_action = "REVIEW_CONTROLLER_CODE_TO_OLS_NONLINEAR_MAPPING"
        strongest = "C4_GLOBAL_CONTROLLER_MAP_CONFINEMENT"
    elif c5 == "SUPPORTED":
        final = "CODE8_COVERAGE_TANGENT_MISALIGNMENT_PRIMARY"
        next_action = "PLAN_CONTROLLER_SEMANTIC_GEOMETRY_TRAINING_AUTHORITY"
        strongest = "C5_CODE_TANGENT_MISALIGNMENT"
    elif c6 == "SUPPORTED":
        final = "CODE8_COVERAGE_CLASS_CONDITIONAL_MAPPING_PRIMARY"
        next_action = "PLAN_CLASS_CONDITIONAL_CONTROLLER_GEOMETRY_ALIGNMENT_AUTHORITY"
        strongest = "C6_CLASS_CONDITIONAL_COVERAGE_COMPRESSION"
    elif c2 == "SUPPORTED":
        final = "CODE8_COVERAGE_GAIN_COMPRESSION_PRIMARY"
        next_action = "PLAN_CONTROLLER_CODE_COVERAGE_GAIN_ALIGNMENT_AUTHORITY"
        strongest = "C2_CODE_COVERAGE_GAIN_COMPRESSION"
    else:
        final = "CODE8_COVERAGE_MECHANISM_UNDERDETERMINED"
        next_action = "PLAN_CONTROLLER_CODE_COVERAGE_GAIN_ALIGNMENT_AUTHORITY"
        strongest = None
    return {
        "schema": "lightgen_scaled_chen_code8_mechanism_classification.v1",
        "mechanisms": mechanisms,
        "final_review_classification": final,
        "strongest_supported_component": strongest,
        "next_action": next_action,
        "geometry_attempt_02_secondary_preserved": geometry_manifest.get("secondary_mechanism"),
        "jacobian_rank_not_reinterpreted_as_global_coverage": True,
        "new_hard_gate": False,
        "finite": True,
    }


def _validate_authorities(config: Mapping[str, Any]) -> dict[str, Any]:
    authorities = _mapping(config["authorities"], "authorities")
    csar = _mapping(authorities["csar"], "csar")
    csar_dir = _resolve(csar["directory"])
    csar_path = csar_dir / str(csar["authority"])
    if file_sha256(csar_path) != S3_CSAR_SHA256:
        raise ValueError("CSAR authority hash changed")
    csar_payload = _read_json(csar_path)
    if csar_payload.get("gate") != "CONTROLLER_STATE_AUTHORITY_PASS":
        raise ValueError("CSAR authority gate is not registered")
    s3_dir = _resolve(S3_DIR)
    if file_sha256(_resolve(S3_MANIFEST)) != S3_MANIFEST_SHA256:
        raise ValueError("S3 manifest hash changed")
    s3_manifest = _read_json(_resolve(S3_MANIFEST))
    if (
        s3_manifest.get("schema") != S3_SCHEMA
        or s3_manifest.get("task_id") != "H-LOMA-GEN-S3-COWA"
        or s3_manifest.get("classification") != "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED"
    ):
        raise ValueError("S3 authority classification changed")
    s3_paths = _verify_manifest_tree(s3_dir, "s3_manifest.json", s3_manifest, "S3 authority")
    panel = _read_json(s3_paths["controller_panel_identity.json"])
    if panel.get("schema") != PANEL_SCHEMA or panel.get("panel_identity_sha256") != S3_PANEL_IDENTITY_SHA256:
        raise ValueError("controller panel identity changed")
    s1_dir = _resolve(S1_DIR)
    if file_sha256(_resolve(S1_MANIFEST)) != S1_MANIFEST_SHA256:
        raise ValueError("S1 manifest hash changed")
    s1_manifest = _read_json(_resolve(S1_MANIFEST))
    if s1_manifest.get("schema") != "lightgen_scaled_chen_ols_semantic_representation_audit.v1":
        raise ValueError("S1 authority schema changed")
    s1_paths = _verify_manifest_tree(s1_dir, "semantic_manifest.json", s1_manifest, "S1 authority")
    geometry_dir = _resolve(GEOMETRY_DIR)
    if file_sha256(_resolve(GEOMETRY_MANIFEST)) != GEOMETRY_MANIFEST_SHA256:
        raise ValueError("geometry attempt_02 manifest hash changed")
    geometry_manifest = _read_json(_resolve(GEOMETRY_MANIFEST))
    if (
        geometry_manifest.get("schema") != GEOMETRY_SCHEMA
        or geometry_manifest.get("task_id") != "H-LOMA-GEN-S3-GA"
        or geometry_manifest.get("status") != "valid_completed"
        or geometry_manifest.get("secondary_mechanism")
        != "S3_GEOMETRY_HIGHER_ORDER_MISMATCH_PRIMARY"
    ):
        raise ValueError("geometry attempt_02 authority changed")
    geometry_paths = _verify_manifest_tree(
        geometry_dir,
        "geometry_alignment_manifest.json",
        geometry_manifest,
        "geometry attempt_02",
    )
    if geometry_manifest["s3_authority"]["manifest_sha256"] != S3_MANIFEST_SHA256:
        raise ValueError("geometry/S3 binding changed")
    if geometry_manifest["s1_reference"]["semantic_manifest_sha256"] != S1_MANIFEST_SHA256:
        raise ValueError("geometry/S1 binding changed")
    return {
        "csar": {"path": _relative(csar_path), "sha256": S3_CSAR_SHA256, "gate": csar_payload["gate"]},
        "s3": {
            "directory": _relative(s3_dir),
            "manifest": _relative(S3_MANIFEST),
            "manifest_sha256": S3_MANIFEST_SHA256,
            "state_artifact_sha256": S3_STATE_SHA256,
            "panel_identity_sha256": S3_PANEL_IDENTITY_SHA256,
            "classification": s3_manifest["classification"],
        },
        "s1": {
            "directory": _relative(s1_dir),
            "manifest": _relative(S1_MANIFEST),
            "manifest_sha256": S1_MANIFEST_SHA256,
            "representation": "R1_POWER",
            "classification": "S1_POWER_SEMANTICS_SUPPORTED",
        },
        "geometry": {
            "directory": _relative(geometry_dir),
            "manifest": _relative(GEOMETRY_MANIFEST),
            "manifest_sha256": GEOMETRY_MANIFEST_SHA256,
            "primary": geometry_manifest["primary_mechanism"],
            "secondary": geometry_manifest["secondary_mechanism"],
        },
        "s3_paths": s3_paths,
        "s1_paths": s1_paths,
        "geometry_paths": geometry_paths,
        "s3_manifest": s3_manifest,
        "s1_manifest": s1_manifest,
        "geometry_manifest": geometry_manifest,
    }


def _load_inputs(authority: Mapping[str, Any]) -> dict[str, Any]:
    s3_paths = authority["s3_paths"]
    state_path = s3_paths["controller_ols_states.pt"]
    if file_sha256(state_path) != S3_STATE_SHA256:
        raise ValueError("S3 controller state artifact hash changed")
    state = safe_torch_load(state_path, expected_sha256=S3_STATE_SHA256, map_location="cpu")
    if not isinstance(state, Mapping):
        raise ValueError("S3 controller state artifact is not a mapping")
    required = {"codes", "labels", "split", "normalized_power", "panel_identity_sha256"}
    if not required.issubset(state):
        raise ValueError("S3 controller state artifact fields are incomplete")
    codes = state["codes"].to(torch.float64)
    labels = state["labels"].to(torch.int64)
    split = state["split"].to(torch.int64)
    controller_power = state["normalized_power"].to(torch.float64)
    if (
        codes.shape != (1000, CODE_DIM)
        or labels.shape != (1000,)
        or split.shape != (1000,)
        or controller_power.shape != (1000, PORT_COUNT)
        or state["panel_identity_sha256"] != S3_PANEL_IDENTITY_SHA256
        or not bool(torch.isfinite(codes).all())
        or not bool(torch.isfinite(controller_power).all())
    ):
        raise ValueError("S3 controller state tensors are invalid")
    base_codes = codes[labels == 0]
    if base_codes.shape != (100, CODE_DIM):
        raise ValueError("frozen base code panel has the wrong shape")
    for label in range(CLASS_COUNT):
        class_codes = codes[labels == label]
        if class_codes.shape != base_codes.shape or not torch.equal(class_codes, base_codes):
            raise ValueError("frozen base codes are not shared across labels")
    s1_embeddings = safe_torch_load(
        authority["s1_paths"]["ols_embeddings.pt"], map_location="cpu"
    )
    if not isinstance(s1_embeddings, Mapping):
        raise ValueError("S1 embeddings artifact is not a mapping")
    real_confirmation = s1_embeddings["representations"]["confirmation"]["R1_POWER"].to(
        torch.float64
    )
    real_confirmation_labels = s1_embeddings["labels"]["confirmation"].to(torch.int64)
    if (
        real_confirmation.ndim != 2
        or real_confirmation.shape[1] != PORT_COUNT
        or real_confirmation.shape[0] != real_confirmation_labels.shape[0]
        or not bool(torch.isfinite(real_confirmation).all())
    ):
        raise ValueError("S1 confirmation R1 reference is invalid")
    jacobian = _read_json(authority["s3_paths"]["jacobian_metrics.json"])
    code_effect = _read_json(authority["s3_paths"]["code_effect_metrics.json"])
    return {
        "panel": _read_json(authority["s3_paths"]["controller_panel_identity.json"]),
        "codes": codes,
        "base_codes": base_codes,
        "labels": labels,
        "split": split,
        "controller_power": controller_power,
        "real_confirmation": real_confirmation,
        "real_confirmation_labels": real_confirmation_labels,
        "jacobian": jacobian,
        "code_effect": code_effect,
    }


def _artifact_refs(directory: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "code8_coverage_review_manifest.json":
            relative = path.relative_to(directory).as_posix()
            refs[relative] = {
                "path": relative,
                "sha256": file_sha256(path),
                "size_bytes": path.stat().st_size,
            }
    return refs


def _assert_output_tree(directory: Path) -> None:
    expected = {
        "authority_binding.json",
        "code_panel_coverage.json",
        "controller_coverage.json",
        "covariance_shape.json",
        "jacobian_magnitude.json",
        "code_norm_ols_excursion.json",
        "per_dimension_utilization.json",
        "global_map_confinement.json",
        "folding_diagnostic.json",
        "mechanism_classification.json",
        "compact_report.json",
        "code8_coverage_review_manifest.json",
    }
    _assert_exact_tree(directory, expected, "Code8 coverage review output")


def _write_output(
    output: Path,
    payloads: Mapping[str, Mapping[str, Any]],
    manifest_payload: Mapping[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to reuse write-once review output: {_relative(output)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as name:
        staging = Path(name)
        for filename, payload in payloads.items():
            atomic_write_new_json(staging / filename, payload, artifact_kind=f"Code8 review {filename}")
        refs = _artifact_refs(staging)
        manifest = dict(manifest_payload)
        manifest["artifacts"] = refs
        atomic_write_new_json(
            staging / "code8_coverage_review_manifest.json",
            manifest,
            artifact_kind="Code8 coverage review manifest",
        )
        _assert_output_tree(staging)
        staging.replace(output)
    _assert_output_tree(output)
    final_manifest = _read_json(output / "code8_coverage_review_manifest.json")
    if final_manifest.get("artifacts") != _artifact_refs(output):
        raise ValueError("Code8 review artifact references do not verify")


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    config_file, config, config_sha256, config_canonical_sha256 = load_config(config_path)
    authority = _validate_authorities(config)
    inputs = _load_inputs(authority)
    output = _resolve(output_path)
    panel_report = _panel_report(inputs["base_codes"], inputs["panel"])
    controller_confirmation_mask = inputs["split"] == 1
    controller_confirmation = inputs["controller_power"][controller_confirmation_mask]
    controller_confirmation_labels = inputs["labels"][controller_confirmation_mask]
    coverage_report, shape_report = _class_coverage(
        controller_confirmation,
        controller_confirmation_labels,
        inputs["real_confirmation"],
        inputs["real_confirmation_labels"],
    )
    geometry_coverage = _read_json(authority["geometry_paths"]["coverage_metrics.json"])
    registered_median = float(geometry_coverage["confirmation"]["median_variance_ratio"])
    registered_maximum = float(geometry_coverage["confirmation"]["maximum_variance_ratio"])
    reproduced_median = coverage_report["ratios"]["trace_ratio"]["median"]
    reproduced_maximum = coverage_report["ratios"]["trace_ratio"]["max"]
    reproduction = {
        "schema": "lightgen_scaled_chen_code8_authority_reproduction.v1",
        "registered_median_variance_ratio": registered_median,
        "recomputed_median_trace_ratio": reproduced_median,
        "registered_maximum_variance_ratio": registered_maximum,
        "recomputed_maximum_trace_ratio": reproduced_maximum,
        "median_absolute_difference": abs(registered_median - reproduced_median),
        "maximum_absolute_difference": abs(registered_maximum - reproduced_maximum),
        "numerical_tolerance": 1.0e-12,
        "gate": (
            "CODE8_COVERAGE_AUTHORITY_REPRODUCTION_PASS"
            if abs(registered_median - reproduced_median) <= 1.0e-12
            and abs(registered_maximum - reproduced_maximum) <= 1.0e-12
            else "CODE8_COVERAGE_AUTHORITY_MISMATCH"
        ),
        "finite": True,
    }
    if reproduction["gate"] != "CODE8_COVERAGE_AUTHORITY_REPRODUCTION_PASS":
        raise ValueError("CODE8_COVERAGE_AUTHORITY_MISMATCH")
    jacobian_report = _jacobian_report(inputs["jacobian"])
    excursion_report = _code_norm_excursion(
        inputs["codes"][controller_confirmation_mask],
        controller_confirmation_labels,
        controller_confirmation,
    )
    confinement_report, pairwise = _pairwise_map_diagnostic(
        inputs["codes"][controller_confirmation_mask],
        controller_confirmation_labels,
        controller_confirmation,
    )
    folding_report = _folding_report(pairwise["code_distances"], pairwise["state_distances"])
    dimension_report = _dimension_report(inputs["base_codes"], inputs["code_effect"])
    mechanism_report = _mechanism_report(
        panel_report,
        coverage_report,
        shape_report,
        jacobian_report,
        confinement_report,
        authority["geometry_manifest"],
    )
    state = base._git_state()
    compact_report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "status": "valid_completed",
        "panel_coverage": panel_report["classification"],
        "coverage_reproduction_gate": reproduction["gate"],
        "C1_panel_undercoverage": mechanism_report["mechanisms"]["C1_PANEL_UNDERCOVERAGE"],
        "C2_gain_compression": mechanism_report["mechanisms"]["C2_CODE_COVERAGE_GAIN_COMPRESSION"],
        "C3_jacobian_gain_compression": mechanism_report["mechanisms"]["C3_LOCAL_JACOBIAN_GAIN_COMPRESSION"],
        "C4_global_confinement": mechanism_report["mechanisms"]["C4_GLOBAL_CONTROLLER_MAP_CONFINEMENT"],
        "C5_tangent_misalignment": mechanism_report["mechanisms"]["C5_CODE_TANGENT_MISALIGNMENT"],
        "C6_class_conditioned_compression": mechanism_report["mechanisms"]["C6_CLASS_CONDITIONAL_COVERAGE_COMPRESSION"],
        "final_review_classification": mechanism_report["final_review_classification"],
        "historical_geometry_classification": authority["geometry_manifest"]["secondary_mechanism"],
        "s3_formal_classification": authority["s3_manifest"]["classification"],
        "intrinsic_dimension_conclusion": "CODE8_INTRINSIC_DIMENSION_ADEQUACY_UNRESOLVED",
        "eight_d_conclusion": "KEEP_8D_PROVISIONALLY",
        "next_action": mechanism_report["next_action"],
        "governance": config["governance"],
        "finite": True,
    }
    authority_binding = {
        "schema": "lightgen_scaled_chen_code8_authority_binding.v1",
        "task_id": TASK_ID,
        "csar": authority["csar"],
        "s3": authority["s3"],
        "s1": authority["s1"],
        "geometry_attempt_02": authority["geometry"],
        "code_distribution": {
            "panel_identity_sha256": S3_PANEL_IDENTITY_SHA256,
            "distribution": inputs["panel"]["distribution"],
            "seed": inputs["panel"]["seed"],
            "code_dim": CODE_DIM,
            "base_code_count": int(inputs["panel"]["base_code_count"]),
        },
        "review_boundary": config["review"],
        "source_git_state": state,
        "authority_mutated": False,
        "finite": True,
    }
    payloads: dict[str, Mapping[str, Any]] = {
        "authority_binding.json": authority_binding,
        "code_panel_coverage.json": panel_report,
        "controller_coverage.json": coverage_report,
        "covariance_shape.json": shape_report,
        "jacobian_magnitude.json": jacobian_report,
        "code_norm_ols_excursion.json": excursion_report,
        "per_dimension_utilization.json": dimension_report,
        "global_map_confinement.json": confinement_report,
        "folding_diagnostic.json": folding_report,
        "mechanism_classification.json": mechanism_report,
        "compact_report.json": compact_report,
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "task_id": TASK_ID,
        "hypothesis": HYPOTHESIS,
        "status": "valid_completed",
        "configuration": {
            "path": _relative(config_file),
            "sha256": config_sha256,
            "canonical_sha256": config_canonical_sha256,
        },
        "output_directory": _relative(output),
        "final_review_classification": mechanism_report["final_review_classification"],
        "next_action": mechanism_report["next_action"],
        "authority_binding": authority_binding,
        "reproduction": reproduction,
        "diagnostic": {
            "cpu_only": True,
            "controller_forward": False,
            "optical_forward": False,
            "jacobian_recompute": False,
            "decoder": False,
            "training": False,
            "checkpoint_write": False,
            "new_panel": False,
            "new_codes": False,
            "new_transform": False,
            "new_hard_gate": False,
        },
        "governance": config["governance"],
        "layout": {
            "json_only": True,
            "write_once": True,
            "manifest_self_hash_excluded": True,
        },
    }
    _write_output(output, payloads, manifest)
    return {
        "task_id": TASK_ID,
        "status": "valid_completed",
        "output_dir": _relative(output),
        "final_review_classification": compact_report["final_review_classification"],
        "next_action": compact_report["next_action"],
        "coverage_reproduction_gate": reproduction["gate"],
        "panel_coverage": compact_report["panel_coverage"],
        "C1": compact_report["C1_panel_undercoverage"],
        "C2": compact_report["C2_gain_compression"],
        "C3": compact_report["C3_jacobian_gain_compression"],
        "C4": compact_report["C4_global_confinement"],
        "C5": compact_report["C5_tangent_misalignment"],
        "C6": compact_report["C6_class_conditioned_compression"],
        "cuda": False,
        "controller_forward": False,
        "optical_forward": False,
    }


def preflight(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_file, config, config_sha256, config_canonical_sha256 = load_config(config_path)
    authority = _validate_authorities(config)
    state = base._git_state()
    return {
        "task_id": TASK_ID,
        "branch": state["branch"],
        "head": state["commit"],
        "worktree_clean": state["clean"],
        "config": _relative(config_file),
        "config_sha256": config_sha256,
        "config_canonical_sha256": config_canonical_sha256,
        "authority_binding": "CODE8_COVERAGE_AUTHORITY_BINDING_PASS",
        "s3_manifest_sha256": authority["s3"]["manifest_sha256"],
        "s1_manifest_sha256": authority["s1"]["manifest_sha256"],
        "geometry_manifest_sha256": authority["geometry"]["manifest_sha256"],
        "controller_forward": False,
        "optical_forward": False,
        "jacobian_recompute": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "run"), default="run")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    result = (
        preflight(arguments.config)
        if arguments.action == "preflight"
        else run(arguments.config, arguments.output_dir)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
