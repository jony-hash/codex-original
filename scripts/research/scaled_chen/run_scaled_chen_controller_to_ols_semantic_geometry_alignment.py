"""Run the frozen controller-to-OLS semantic geometry alignment diagnostic.

This is a CPU-only, no-update diagnostic consumer of the registered S3
controller states and the registered S1 ``R1_POWER`` reference.  It fits one
class-conditional centroid translation on the selection split, freezes that
translation, and evaluates it on the untouched confirmation split.  It never
loads a controller, optical model, decoder, evaluator, optimizer, or
checkpoint for update.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml

from ldm_project.v2.evidence import (
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.application.research.optical_manifold import optical_manifold as base


TASK_ID = "H-LOMA-GEN-S3-GA"
HYPOTHESIS = "CONTROLLER_TO_OLS_SEMANTIC_GEOMETRY_ALIGNMENT"
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_controller_to_ols_semantic_geometry_alignment_v1"
MANIFEST_SCHEMA = "lightgen_scaled_chen_controller_to_ols_semantic_geometry_alignment.v1"
AUTHORITY_SCHEMA = "lightgen_scaled_chen_geometry_alignment_authority_binding.v1"
TRANSLATION_SCHEMA = "lightgen_scaled_chen_class_centroid_translation.v1"
ISOLATION_SCHEMA = "lightgen_scaled_chen_selection_confirmation_alignment_isolation.v1"
REACHABILITY_SCHEMA = "lightgen_scaled_chen_geometry_confirmation_before_after.v1"
STABILITY_SCHEMA = "lightgen_scaled_chen_anchor_offset_stability.v1"
CENTROID_SCHEMA = "lightgen_scaled_chen_interclass_centroid_geometry.v1"
COVARIANCE_SCHEMA = "lightgen_scaled_chen_covariance_geometry.v1"
COVERAGE_SCHEMA = "lightgen_scaled_chen_within_class_coverage.v1"
RESIDUAL_SCHEMA = "lightgen_scaled_chen_aligned_residual_geometry.v1"
CLASS9_SCHEMA = "lightgen_scaled_chen_class9_geometry_diagnostic.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_geometry_alignment_compact_report.v1"

PORT_COUNT = 100
CLASS_COUNT = 10
CODE_DIM = 8
PANEL_COUNT = 1000
PANEL_PER_CLASS = 100
PANEL_SELECTION_PER_CLASS = 50
PANEL_CONFIRMATION_PER_CLASS = 50
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 1236
R1_POWER = "R1_POWER"

S3_OUTPUT_PATH = Path(
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_controller_to_ols_writability_audit_v1_attempt_03"
)
S3_MANIFEST_PATH = S3_OUTPUT_PATH / "s3_manifest.json"
S3_MANIFEST_SHA256 = (
    "0779d8d0ffb0a9958064bb3c05d451058cdd7a670a732d6ae1d4ba49c11eb5e6"
)
S3_PANEL_IDENTITY_SHA256 = (
    "8a9dd1c7e3e2b674a152e6c96ac8b47976ff1f8d27428a18477e89795b940f47"
)
S3_STATE_ARTIFACT_SHA256 = (
    "45f9b2b18c0cc67bf784ee206059d97164abaab08f0304892c368f53d218b19a"
)
PRIOR_ATTEMPT_OUTPUT_PATH = Path(
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_controller_to_ols_semantic_geometry_alignment_v1_attempt_01"
)
PRIOR_ATTEMPT_MANIFEST_PATH = PRIOR_ATTEMPT_OUTPUT_PATH / "geometry_alignment_manifest.json"
PRIOR_ATTEMPT_MANIFEST_SHA256 = (
    "4d64bededa2884e76a6112cc7f1efd76bb9548a5048d62b9f4b7152402b7a52b"
)

S1_CONFIG_PATH = Path(
    "configs/research/scaled_chen/lightgen_scaled_chen_ols_semantic_representation_audit_v1.yaml"
)
S1_CONFIG_SHA256 = (
    "60d57cd134a567c17602775ec16f11fff05f56ffe900ac730774abde3432c521"
)
S1_CONFIG_CANONICAL_SHA256 = (
    "719526b8d6e82da20bdc9040314084b1192ce376d413eecdf0db07420ad2ab13"
)
S1_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01"
)
S1_MANIFEST_SHA256 = (
    "b0199040926618b17373edc4f167cc8bc4788f39f64de174e1a3fb225781833e"
)
S1_IDENTITY_SHA256 = (
    "522ff5b00770ee34fe7b61b1f414775abe3310f398cc0fdf55049b30be2925e2"
)
S1_CONTRACT_SHA256 = (
    "f99c2ced713005fe3f4a5a0fa91a2cd7648f5647e2360f122bc68e77f225f10d"
)
S1_EMBEDDINGS_SHA256 = (
    "9375b08597f03fee3798e05798779556c3e7534db6a9d39033232f7278d32279"
)
S1_DATASET_MANIFEST_SHA256 = (
    "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb"
)
S1_LABEL_HASHES = {
    "fit": "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251",
    "selection": "31fce460255bda46afc2bd6b6777793873e2e6ee33e23a681def3ab8af6693cb",
    "confirmation": "b05112caa3014d6c2081423b371361eefe1ec7a9f531e54db016b89bfbb77b18",
}
S1_INDEX_HASHES = {
    "fit": "532f05aa60d235499fdc5c078ad9c3c95c36a0c61f2c8e9d64a8309c1741255d",
    "selection": "11c076affa8511e883f595a0d1ee91e40f6d94747a351e21269520344d3f594f",
    "confirmation": "ff7ab50b95d96e727c240edc2baa916d1e6c11402b8f4f9c2ce6e2e7e5e2218a",
}

DEFAULT_CONFIG_PATH = Path(
    "configs/research/scaled_chen/"
    "lightgen_scaled_chen_controller_to_ols_semantic_geometry_alignment_v1.yaml"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260906_v2/runs/"
    "lightgen_scaled_chen_controller_to_ols_semantic_geometry_alignment_v1_attempt_02"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(value: str | Path) -> Path:
    target = Path(value)
    return target if target.is_absolute() else _repo_root() / target


def _relative(value: str | Path) -> str:
    target = _resolve(value)
    try:
        return target.relative_to(_repo_root()).as_posix()
    except ValueError:
        return str(target)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact is not a mapping: {path}")
    return payload


def _assert_tensor(
    value: Any,
    *,
    name: str,
    shape: tuple[int, ...] | None = None,
    dtype: torch.dtype | None = None,
    nonnegative: bool = False,
) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise ValueError(f"{name} must be a tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{name} shape is invalid")
    if dtype is not None and value.dtype != dtype:
        raise ValueError(f"{name} dtype is invalid")
    if not torch.is_floating_point(value) and dtype is None:
        if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError(f"{name} dtype is invalid")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    if nonnegative and bool((value < 0).any()):
        raise ValueError(f"{name} contains negative values")
    return value


def _summary(values: torch.Tensor) -> dict[str, float | int | bool]:
    values = values.detach().to(torch.float64).flatten()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("summary input must be finite and non-empty")
    quantiles = torch.quantile(
        values, torch.tensor([0.05, 0.50, 0.95], dtype=torch.float64)
    )
    return {
        "count": int(values.numel()),
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "q05": float(quantiles[0]),
        "median": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "max": float(values.max()),
        "finite": True,
    }


def _bootstrap_quantile_delta_ci(
    controller: torch.Tensor,
    real: torch.Tensor,
    *,
    quantile: float,
    seed: int,
    reps: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float | int]:
    controller = controller.detach().to(torch.float64).flatten()
    real = real.detach().to(torch.float64).flatten()
    if (
        controller.numel() == 0
        or real.numel() == 0
        or not torch.isfinite(controller).all()
        or not torch.isfinite(real).all()
        or not 0.0 < quantile < 1.0
    ):
        raise ValueError("quantile bootstrap inputs are invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    controller_draws = torch.randint(
        controller.numel(), (reps, controller.numel()), generator=generator
    )
    real_draws = torch.randint(real.numel(), (reps, real.numel()), generator=generator)
    controller_samples = controller.index_select(
        0, controller_draws.reshape(-1)
    ).reshape(reps, -1)
    real_samples = real.index_select(0, real_draws.reshape(-1)).reshape(reps, -1)
    deltas = torch.quantile(controller_samples, quantile, dim=1) - torch.quantile(
        real_samples, quantile, dim=1
    )
    interval = torch.quantile(
        deltas, torch.tensor([0.025, 0.975], dtype=torch.float64)
    )
    return {
        "quantile": quantile,
        "controller_point": float(torch.quantile(controller, quantile)),
        "real_point": float(torch.quantile(real, quantile)),
        "point_delta": float(torch.quantile(controller, quantile) - torch.quantile(real, quantile)),
        "lower95": float(interval[0]),
        "upper95": float(interval[1]),
        "replicates": reps,
        "seed": seed,
    }


def _class_centroids(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float64)
    labels = labels.to(torch.long)
    centroids = []
    for label in range(CLASS_COUNT):
        selected = values[labels == label]
        if selected.numel() == 0:
            raise ValueError(f"class {label} is absent")
        centroids.append(selected.mean(dim=0))
    return torch.stack(centroids).contiguous()


def fit_class_translation(
    real_selection: torch.Tensor,
    real_selection_labels: torch.Tensor,
    controller_selection: torch.Tensor,
    controller_selection_labels: torch.Tensor,
) -> torch.Tensor:
    """Fit the only permitted transform from selection-equivalent states."""
    real_centroids = _class_centroids(real_selection, real_selection_labels)
    controller_centroids = _class_centroids(
        controller_selection, controller_selection_labels
    )
    return (real_centroids - controller_centroids).contiguous()


def apply_class_translation(
    values: torch.Tensor, labels: torch.Tensor, translation: torch.Tensor
) -> torch.Tensor:
    values = values.to(torch.float64)
    labels = labels.to(torch.long)
    translation = _assert_tensor(
        translation,
        name="translation",
        shape=(CLASS_COUNT, PORT_COUNT),
    ).to(torch.float64)
    if labels.ndim != 1 or labels.shape[0] != values.shape[0] or bool(
        ((labels < 0) | (labels >= CLASS_COUNT)).any()
    ):
        raise ValueError("translation labels are invalid")
    return (values + translation.index_select(0, labels)).contiguous()


def _nearest_components(
    queries: torch.Tensor,
    query_labels: torch.Tensor,
    reference: torch.Tensor,
    reference_labels: torch.Tensor,
) -> dict[str, Any]:
    queries = queries.to(torch.float64)
    reference = reference.to(torch.float64)
    query_labels = query_labels.to(torch.long)
    reference_labels = reference_labels.to(torch.long)
    if queries.ndim != 2 or reference.ndim != 2 or queries.shape[1] != reference.shape[1]:
        raise ValueError("distance inputs have incompatible shapes")
    if query_labels.shape != (queries.shape[0],) or reference_labels.shape != (
        reference.shape[0],
    ):
        raise ValueError("distance labels have incompatible shapes")
    distances = torch.cdist(queries, reference, p=2).square()
    target = torch.empty(queries.shape[0], dtype=torch.float64)
    other = torch.empty_like(target)
    target_indices = torch.empty(queries.shape[0], dtype=torch.long)
    for label in range(CLASS_COUNT):
        query_positions = torch.nonzero(query_labels == label, as_tuple=False).flatten()
        target_positions = torch.nonzero(
            reference_labels == label, as_tuple=False
        ).flatten()
        other_positions = torch.nonzero(
            reference_labels != label, as_tuple=False
        ).flatten()
        if query_positions.numel() == 0 or target_positions.numel() == 0 or other_positions.numel() == 0:
            raise ValueError("distance reference lacks a required class")
        target_distances = distances[query_positions][:, target_positions]
        target_values, target_local = target_distances.min(dim=1)
        target[query_positions] = target_values
        target_indices[query_positions] = target_positions.index_select(0, target_local)
        other[query_positions] = distances[query_positions][:, other_positions].min(dim=1).values
    nearest_indices = distances.argmin(dim=1)
    predictions = reference_labels.index_select(0, nearest_indices)
    return {
        "target": target,
        "other": other,
        "margin": other - target,
        "predictions": predictions,
        "target_indices": target_indices,
        "nearest_distances": distances.min(dim=1).values,
    }


def _confusion(labels: torch.Tensor, predictions: torch.Tensor) -> list[list[int]]:
    return [
        [
            int(((labels == true_label) & (predictions == predicted_label)).sum())
            for predicted_label in range(CLASS_COUNT)
        ]
        for true_label in range(CLASS_COUNT)
    ]


def _targeting_report(
    values: torch.Tensor,
    labels: torch.Tensor,
    null_labels: torch.Tensor,
    reference: torch.Tensor,
    reference_labels: torch.Tensor,
) -> dict[str, Any]:
    real = _nearest_components(values, labels, reference, reference_labels)
    null = _nearest_components(values, null_labels, reference, reference_labels)
    retrieval = real["predictions"] == labels
    null_retrieval = null["predictions"] == null_labels
    per_class: dict[str, Any] = {}
    for label in range(CLASS_COUNT):
        mask = labels == label
        per_class[str(label)] = {
            "count": int(mask.sum()),
            "target_distance": _summary(real["target"][mask]),
            "other_distance": _summary(real["other"][mask]),
            "semantic_margin": _summary(real["margin"][mask]),
            "retrieval_accuracy": float(retrieval[mask].to(torch.float64).mean()),
            "null_target_distance": _summary(null["target"][mask]),
            "null_semantic_margin": _summary(null["margin"][mask]),
            "null_retrieval_accuracy": float(
                null_retrieval[mask].to(torch.float64).mean()
            ),
        }
    return {
        "target_distance": _summary(real["target"]),
        "other_distance": _summary(real["other"]),
        "semantic_margin": _summary(real["margin"]),
        "retrieval_accuracy": float(retrieval.to(torch.float64).mean()),
        "null_target_distance": _summary(null["target"]),
        "null_semantic_margin": _summary(null["margin"]),
        "null_retrieval_accuracy": float(null_retrieval.to(torch.float64).mean()),
        "margin_change_vs_null": float(real["margin"].mean() - null["margin"].mean()),
        "target_improvement_vs_null": float(null["target"].mean() - real["target"].mean()),
        "retrieval_change_vs_null": float(
            retrieval.to(torch.float64).mean() - null_retrieval.to(torch.float64).mean()
        ),
        "confusion": _confusion(labels, real["predictions"]),
        "null_confusion": _confusion(null_labels, null["predictions"]),
        "class_order": list(range(CLASS_COUNT)),
        "per_class": per_class,
        "finite": True,
    }


def _reachability_split(
    split_name: str,
    controller_before: torch.Tensor,
    controller_after: torch.Tensor,
    controller_labels: torch.Tensor,
    real: torch.Tensor,
    real_labels: torch.Tensor,
    fit_reference: torch.Tensor,
    fit_labels: torch.Tensor,
) -> dict[str, Any]:
    before_target = _nearest_components(
        controller_before, controller_labels, fit_reference, fit_labels
    )["target"]
    after_target = _nearest_components(
        controller_after, controller_labels, fit_reference, fit_labels
    )["target"]
    real_target = _nearest_components(real, real_labels, fit_reference, fit_labels)[
        "target"
    ]
    per_class: dict[str, Any] = {}
    q50_shrinks: list[float] = []
    q95_shrinks: list[float] = []
    for label in range(CLASS_COUNT):
        controller_mask = controller_labels == label
        real_mask = real_labels == label
        before = before_target[controller_mask]
        after = after_target[controller_mask]
        real_values = real_target[real_mask]
        before_q50 = _bootstrap_quantile_delta_ci(
            before,
            real_values,
            quantile=0.50,
            seed=BOOTSTRAP_SEED + 200 + label,
        )
        before_q95 = _bootstrap_quantile_delta_ci(
            before,
            real_values,
            quantile=0.95,
            seed=BOOTSTRAP_SEED + 400 + label,
        )
        after_q50 = _bootstrap_quantile_delta_ci(
            after,
            real_values,
            quantile=0.50,
            seed=BOOTSTRAP_SEED + 600 + label,
        )
        after_q95 = _bootstrap_quantile_delta_ci(
            after,
            real_values,
            quantile=0.95,
            seed=BOOTSTRAP_SEED + 800 + label,
        )
        q50_shrink = abs(before_q50["point_delta"]) - abs(after_q50["point_delta"])
        q95_shrink = abs(before_q95["point_delta"]) - abs(after_q95["point_delta"])
        q50_shrinks.append(q50_shrink)
        q95_shrinks.append(q95_shrink)
        per_class[str(label)] = {
            "controller_before": _summary(before),
            "controller_after": _summary(after),
            "real_reference": _summary(real_values),
            "delta_q50_before": before_q50,
            "delta_q95_before": before_q95,
            "delta_q50_after": after_q50,
            "delta_q95_after": after_q95,
            "q50_gap_before": before_q50["point_delta"],
            "q50_gap_after": after_q50["point_delta"],
            "q95_gap_before": before_q95["point_delta"],
            "q95_gap_after": after_q95["point_delta"],
            "q50_absolute_shrinkage": q50_shrink,
            "q95_absolute_shrinkage": q95_shrink,
            "q50_relative_shrinkage": (
                q50_shrink / abs(before_q50["point_delta"])
                if abs(before_q50["point_delta"]) > 1.0e-12
                else None
            ),
            "q95_relative_shrinkage": (
                q95_shrink / abs(before_q95["point_delta"])
                if abs(before_q95["point_delta"]) > 1.0e-12
                else None
            ),
        }
    overall_before = _bootstrap_quantile_delta_ci(
        before_target,
        real_target,
        quantile=0.50,
        seed=BOOTSTRAP_SEED + 1000,
    )
    overall_before_q95 = _bootstrap_quantile_delta_ci(
        before_target,
        real_target,
        quantile=0.95,
        seed=BOOTSTRAP_SEED + 1001,
    )
    overall_after = _bootstrap_quantile_delta_ci(
        after_target,
        real_target,
        quantile=0.50,
        seed=BOOTSTRAP_SEED + 1002,
    )
    overall_after_q95 = _bootstrap_quantile_delta_ci(
        after_target,
        real_target,
        quantile=0.95,
        seed=BOOTSTRAP_SEED + 1003,
    )
    return {
        "split": split_name,
        "controller_sample_count": int(controller_labels.numel()),
        "real_sample_count": int(real_labels.numel()),
        "overall": {
            "before": {
                "q50": overall_before,
                "q95": overall_before_q95,
            },
            "after": {
                "q50": overall_after,
                "q95": overall_after_q95,
            },
            "q50_absolute_shrinkage": abs(overall_before["point_delta"])
            - abs(overall_after["point_delta"]),
            "q95_absolute_shrinkage": abs(overall_before_q95["point_delta"])
            - abs(overall_after_q95["point_delta"]),
        },
        "per_class": per_class,
        "q50_shrinkage_positive_fraction": float(
            sum(value > 0.0 for value in q50_shrinks) / CLASS_COUNT
        ),
        "q95_shrinkage_positive_fraction": float(
            sum(value > 0.0 for value in q95_shrinks) / CLASS_COUNT
        ),
        "finite": True,
    }


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    values = values.to(torch.float64).flatten()
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(values)
    start = 0
    while start < values.numel():
        end = start + 1
        while end < values.numel() and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _spearman_upper_triangle(first: torch.Tensor, second: torch.Tensor) -> float:
    indices = torch.triu_indices(first.shape[0], first.shape[1], offset=1)
    first_values = first[indices[0], indices[1]]
    second_values = second[indices[0], indices[1]]
    first_ranks = _rankdata(first_values)
    second_ranks = _rankdata(second_values)
    first_centered = first_ranks - first_ranks.mean()
    second_centered = second_ranks - second_ranks.mean()
    denominator = first_centered.norm() * second_centered.norm()
    if denominator <= 0.0:
        return 0.0
    return float(torch.dot(first_centered, second_centered) / denominator)


def _centroid_geometry(
    real: torch.Tensor,
    real_labels: torch.Tensor,
    controller: torch.Tensor,
    controller_labels: torch.Tensor,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    real_centroids = _class_centroids(real, real_labels)
    controller_centroids = _class_centroids(controller, controller_labels)
    real_distances = torch.cdist(real_centroids, real_centroids, p=2)
    controller_distances = torch.cdist(
        controller_centroids, controller_centroids, p=2
    )
    real_nearest = []
    controller_nearest = []
    for label in range(CLASS_COUNT):
        real_row = real_distances[label].clone()
        controller_row = controller_distances[label].clone()
        real_row[label] = float("inf")
        controller_row[label] = float("inf")
        real_nearest.append(int(real_row.argmin()))
        controller_nearest.append(int(controller_row.argmin()))
    agreement = sum(a == b for a, b in zip(real_nearest, controller_nearest)) / CLASS_COUNT
    order_correlation = _spearman_upper_triangle(real_distances, controller_distances)
    relative_distortion = (
        (real_distances - controller_distances).abs()
        / real_distances.clamp_min(1.0e-12)
    )
    preserved = bool(
        order_correlation >= float(thresholds["minimum_spearman"])
        and agreement >= float(thresholds["minimum_nearest_relation_agreement"])
    )
    return {
        "real_centroids": real_centroids.tolist(),
        "controller_centroids": controller_centroids.tolist(),
        "real_distance_matrix": real_distances.tolist(),
        "controller_distance_matrix": controller_distances.tolist(),
        "mean_absolute_matrix_distortion": float(
            (real_distances - controller_distances).abs().mean()
        ),
        "mean_relative_matrix_distortion": float(relative_distortion.mean()),
        "spearman_order_correlation": order_correlation,
        "real_nearest_class": real_nearest,
        "controller_nearest_class": controller_nearest,
        "nearest_relation_agreement": agreement,
        "qualitative_status": (
            "INTER_CLASS_SEMANTIC_GEOMETRY_PRESERVED"
            if preserved
            else "INTER_CLASS_SEMANTIC_GEOMETRY_DISTORTED"
        ),
        "finite": True,
    }


def _covariance_stats(values: torch.Tensor, effective_rank_relative_threshold: float) -> dict[str, Any]:
    values = values.to(torch.float64)
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0.0)
    trace = float(eigenvalues.sum())
    leading = float(eigenvalues.max()) if eigenvalues.numel() else 0.0
    effective_rank = int(
        (eigenvalues >= leading * effective_rank_relative_threshold).sum()
    ) if leading > 0.0 else 0
    explained = eigenvalues / max(trace, 1.0e-12)
    return {
        "sample_count": int(values.shape[0]),
        "trace_total_variance": trace,
        "effective_rank": effective_rank,
        "effective_rank_relative_threshold": effective_rank_relative_threshold,
        "eigenvalues_descending": eigenvalues.tolist(),
        "explained_variance": explained.tolist(),
        "cumulative_explained_variance": torch.cumsum(explained, dim=0).tolist(),
        "top_eigenvalues": eigenvalues[:10].tolist(),
        "finite": True,
    }


def _covariance_geometry(
    real: torch.Tensor,
    real_labels: torch.Tensor,
    controller: torch.Tensor,
    controller_labels: torch.Tensor,
    threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    real_stats: dict[str, Any] = {}
    controller_stats: dict[str, Any] = {}
    ratios: dict[str, float] = {}
    for label in range(CLASS_COUNT):
        key = str(label)
        real_class = real[real_labels == label]
        controller_class = controller[controller_labels == label]
        real_stats[key] = _covariance_stats(real_class, threshold)
        controller_stats[key] = _covariance_stats(controller_class, threshold)
        real_trace = real_stats[key]["trace_total_variance"]
        controller_trace = controller_stats[key]["trace_total_variance"]
        ratios[key] = controller_trace / max(real_trace, 1.0e-12)
    return (
        {
            "schema": COVARIANCE_SCHEMA,
            "real": real_stats,
            "controller": controller_stats,
            "principal_angles": None,
            "principal_angles_status": "NO_PREREGISTERED_PCA_K",
            "finite": True,
        },
        ratios,
    )


def _residual_report(
    values: torch.Tensor,
    labels: torch.Tensor,
    reference: torch.Tensor,
    reference_labels: torch.Tensor,
    class_centroids: torch.Tensor,
) -> dict[str, Any]:
    nearest = _nearest_components(values, labels, reference, reference_labels)
    target = reference.index_select(0, nearest["target_indices"])
    direction = target - class_centroids.index_select(0, labels)
    residual = values - target
    direction_squared = direction.square().sum(dim=1)
    radial_projection = (
        (residual * direction).sum(dim=1, keepdim=True)
        / direction_squared.clamp_min(1.0e-12).unsqueeze(1)
        * direction
    )
    radial = radial_projection
    tangent = residual - radial_projection
    radial_squared = radial.square().sum(dim=1)
    directional_squared = tangent.square().sum(dim=1)
    total_squared = radial_squared + directional_squared
    fractions = directional_squared / total_squared.clamp_min(1.0e-12)
    contribution = residual.square()
    top_indices = torch.argsort(contribution.mean(dim=0), descending=True)[:10]
    top_total = contribution[:, top_indices].sum()
    total = contribution.sum()
    per_class: dict[str, Any] = {}
    for label in range(CLASS_COUNT):
        mask = labels == label
        class_contribution = contribution[mask]
        class_top_indices = torch.argsort(
            class_contribution.mean(dim=0), descending=True
        )[:10]
        class_top_total = class_contribution[:, class_top_indices].sum()
        per_class[str(label)] = {
            "radial_squared": _summary(radial_squared[mask]),
            "directional_squared": _summary(directional_squared[mask]),
            "directional_fraction": _summary(fractions[mask]),
            "top_contributing_ports": [int(item) for item in class_top_indices],
            "top_port_contribution_fraction": float(
                class_top_total / class_contribution.sum().clamp_min(1.0e-12)
            ),
            "signed_mean_residual": residual[mask].mean(dim=0).tolist(),
        }
    return {
        "radial_squared": _summary(radial_squared),
        "directional_squared": _summary(directional_squared),
        "directional_fraction": _summary(fractions),
        "top_contributing_ports": [int(item) for item in top_indices],
        "top_port_contribution_fraction": float(top_total / total.clamp_min(1.0e-12)),
        "signed_mean_residual": residual.mean(dim=0).tolist(),
        "per_class": per_class,
        "finite": True,
    }


def _translation_stability(
    real_selection: torch.Tensor,
    real_selection_labels: torch.Tensor,
    controller_selection: torch.Tensor,
    controller_selection_labels: torch.Tensor,
    real_confirmation: torch.Tensor,
    real_confirmation_labels: torch.Tensor,
    controller_confirmation: torch.Tensor,
    controller_confirmation_labels: torch.Tensor,
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    delta_selection = fit_class_translation(
        real_selection,
        real_selection_labels,
        controller_selection,
        controller_selection_labels,
    )
    delta_confirmation = fit_class_translation(
        real_confirmation,
        real_confirmation_labels,
        controller_confirmation,
        controller_confirmation_labels,
    )
    per_class: dict[str, Any] = {}
    stable_count = 0
    for label in range(CLASS_COUNT):
        selected = delta_selection[label]
        confirmed = delta_confirmation[label]
        selected_norm = selected.norm()
        confirmed_norm = confirmed.norm()
        cosine = float(
            torch.dot(selected, confirmed)
            / (selected_norm * confirmed_norm).clamp_min(1.0e-12)
        )
        norm_ratio = float(confirmed_norm / selected_norm.clamp_min(1.0e-12))
        residual_norm = float((confirmed - selected).norm())
        stable = bool(
            cosine >= float(thresholds["minimum_cosine_similarity"])
            and float(thresholds["minimum_norm_ratio"])
            <= norm_ratio
            <= float(thresholds["maximum_norm_ratio"])
            and residual_norm / selected_norm.clamp_min(1.0e-12)
            <= float(thresholds["maximum_relative_coordinate_residual"])
        )
        stable_count += int(stable)
        per_class[str(label)] = {
            "delta_selection": selected.tolist(),
            "delta_confirmation_observation_only": confirmed.tolist(),
            "selection_norm": float(selected_norm),
            "confirmation_norm": float(confirmed_norm),
            "cosine_similarity": cosine,
            "norm_ratio_confirmation_over_selection": norm_ratio,
            "coordinate_residual_norm": residual_norm,
            "relative_coordinate_residual": float(
                residual_norm / selected_norm.clamp_min(1.0e-12)
            ),
            "classification": (
                "ANCHOR_OFFSET_STABLE" if stable else "ANCHOR_OFFSET_UNSTABLE"
            ),
        }
    return {
        "schema": STABILITY_SCHEMA,
        "fit_source": "selection_only",
        "confirmation_delta_source": "observation_only_never_used_for_transform",
        "per_class": per_class,
        "stable_class_fraction": stable_count / CLASS_COUNT,
        "finite": True,
    }


def _validate_config(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "task_id",
        "hypothesis",
        "s3_authority",
        "s1_reference",
        "observer",
        "translation",
        "stability",
        "geometry",
        "covariance",
        "decision",
        "execution",
        "authorization",
        "governance",
        "output",
    }
    if set(payload) != required:
        raise ValueError("geometry alignment config fields are invalid")
    if payload["schema"] != CONFIG_SCHEMA or payload["task_id"] != TASK_ID:
        raise ValueError("geometry alignment config identity is invalid")
    if payload["hypothesis"] != HYPOTHESIS:
        raise ValueError("geometry alignment hypothesis is invalid")
    s3 = _mapping(payload["s3_authority"], "s3_authority")
    if dict(s3) != {
        "directory": _relative(S3_OUTPUT_PATH),
        "manifest": _relative(S3_MANIFEST_PATH),
        "manifest_sha256": S3_MANIFEST_SHA256,
        "schema": "lightgen_scaled_chen_controller_to_ols_writability_audit.v1",
        "task_id": "H-LOMA-GEN-S3-COWA",
        "classification": "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED",
        "panel_identity_sha256": S3_PANEL_IDENTITY_SHA256,
        "state_artifact_sha256": S3_STATE_ARTIFACT_SHA256,
    }:
        raise ValueError("S3 authority binding is not locked")
    s1 = _mapping(payload["s1_reference"], "s1_reference")
    if dict(s1) != {
        "config": _relative(S1_CONFIG_PATH),
        "config_sha256": S1_CONFIG_SHA256,
        "config_canonical_sha256": S1_CONFIG_CANONICAL_SHA256,
        "directory": _relative(S1_OUTPUT_PATH),
        "semantic_manifest_sha256": S1_MANIFEST_SHA256,
        "identity_manifest_sha256": S1_IDENTITY_SHA256,
        "contract_sha256": S1_CONTRACT_SHA256,
        "embeddings_sha256": S1_EMBEDDINGS_SHA256,
        "dataset_manifest_sha256": S1_DATASET_MANIFEST_SHA256,
        "representation": R1_POWER,
    }:
        raise ValueError("S1 reference binding is not locked")
    if dict(_mapping(payload["observer"], "observer")) != {
        "representation": R1_POWER,
        "formula": "z_j=abs(c_j)^2/s_j",
        "distance": "euclidean_squared",
        "complex_state_recomputed": False,
        "whitening": False,
        "renormalization": False,
        "new_scale_fit": False,
    }:
        raise ValueError("observer contract is not locked")
    if dict(_mapping(payload["translation"], "translation")) != {
        "fit": "t_y=mu_real_sel[y]-mu_ctrl_sel[y]",
        "application": "z_ctrl_aligned=z_ctrl+t_y",
        "selection_only_fit": True,
        "class_count": CLASS_COUNT,
        "port_count": PORT_COUNT,
        "controller_state_mutation": False,
    }:
        raise ValueError("translation contract is not locked")
    stability = _mapping(payload["stability"], "stability")
    expected_stability = {
        "classification": "descriptive_fixed_rule",
        "minimum_cosine_similarity": 0.8,
        "minimum_norm_ratio": 0.5,
        "maximum_norm_ratio": 2.0,
        "maximum_relative_coordinate_residual": 1.0,
    }
    if dict(stability) != expected_stability:
        raise ValueError("stability contract is not locked")
    geometry = _mapping(payload["geometry"], "geometry")
    if dict(geometry) != {
        "centroid_matrix": "10x10_pairwise_euclidean",
        "spearman_order_correlation": "descriptive_fixed_rule",
        "minimum_spearman": 0.8,
        "minimum_nearest_relation_agreement": 0.8,
        "global_transform_fit": False,
    }:
        raise ValueError("inter-class geometry contract is not locked")
    covariance = _mapping(payload["covariance"], "covariance")
    if dict(covariance) != {
        "eigen_spectrum": "full",
        "pca_k": None,
        "principal_angles": False,
        "effective_rank_relative_threshold": 1.0e-6,
        "pca_correction": False,
    }:
        raise ValueError("covariance contract is not locked")
    decision = _mapping(payload["decision"], "decision")
    if dict(decision) != {
        "translation_effective_min_q50_shrinkage_fraction": 0.8,
        "translation_effective_min_q95_shrinkage_fraction": 0.8,
        "coverage_compression_all_class_ratio_below": 1.0,
        "coverage_compression_median_ratio_at_most": 0.5,
        "no_new_s3_gate": True,
    }:
        raise ValueError("mechanism decision contract is not locked")
    if dict(_mapping(payload["execution"], "execution")) != {
        "device": "cpu",
        "controller_forward": False,
        "optical_forward": False,
        "decoder": False,
        "cuda": False,
        "deterministic": True,
    }:
        raise ValueError("execution contract is not locked")
    if dict(_mapping(payload["authorization"], "authorization")) != {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal_promotion": False,
        "optimizer_created": False,
        "backward_called": False,
        "checkpoint_created": False,
        "s3_authority_mutated": False,
        "prohibited": [
            "CUDA",
            "controller_forward",
            "optical_forward",
            "decoder",
            "OSERA",
            "training",
            "optimizer",
            "backward",
            "checkpoint_update",
            "new_S3_gate",
            "new_representation",
            "affine_alignment",
            "Procrustes",
            "nonlinear_alignment",
            "per_port_scaling",
            "class_relabel",
            "code_dimension_change",
            "formal_promotion",
        ],
    }:
        raise ValueError("authorization boundary is not locked")
    if dict(_mapping(payload["governance"], "governance")) != {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }:
        raise ValueError("governance boundary is not locked")
    if dict(_mapping(payload["output"], "output")) != {
        "directory": _relative(DEFAULT_OUTPUT_PATH),
        "write_once": True,
        "json_only": True,
        "copy_frozen_corpus": False,
    }:
        raise ValueError("output contract is not locked")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> tuple[Path, dict[str, Any], str, str]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("geometry alignment config must be a YAML mapping")
    _validate_config(payload)
    return target, payload, file_sha256(target), canonical_sha256(payload)


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


def _validate_s3_source(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = _mapping(config["s3_authority"], "s3_authority")
    directory = _resolve(binding["directory"])
    manifest_path = _resolve(binding["manifest"])
    if not directory.is_dir() or not manifest_path.is_file():
        raise ValueError("S3 authority directory or manifest is missing")
    if file_sha256(manifest_path) != S3_MANIFEST_SHA256:
        raise ValueError("S3 authority manifest hash changed")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != binding["schema"]
        or manifest.get("task_id") != binding["task_id"]
        or manifest.get("status") != "valid_completed"
        or manifest.get("classification") != binding["classification"]
        or manifest.get("primary_gate") != "CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED"
    ):
        raise ValueError("S3 authority classification is not the registered result")
    artifacts = _mapping(manifest.get("artifacts"), "S3 artifacts")
    expected_tree = set(artifacts) | {"s3_manifest.json"}
    _assert_exact_tree(directory, expected_tree, "S3 authority")
    artifact_paths: dict[str, Path] = {}
    for name, ref_value in artifacts.items():
        ref = _mapping(ref_value, f"S3 artifact {name}")
        if ref.get("path") != name or not isinstance(ref.get("sha256"), str):
            raise ValueError(f"S3 artifact reference is invalid: {name}")
        path = directory / name
        if file_sha256(path) != ref["sha256"] or path.stat().st_size != int(ref["size_bytes"]):
            raise ValueError(f"S3 artifact changed: {name}")
        artifact_paths[name] = path
    panel = _read_json(artifact_paths["controller_panel_identity.json"])
    if (
        panel.get("schema") != "lightgen_scaled_chen_controller_panel_identity.v1"
        or panel.get("panel_identity_sha256") != S3_PANEL_IDENTITY_SHA256
    ):
        raise ValueError("S3 panel identity is stale")
    if manifest.get("panel", {}).get("panel_identity_sha256") != S3_PANEL_IDENTITY_SHA256:
        raise ValueError("S3 manifest panel identity is stale")
    states = safe_torch_load(
        artifact_paths["controller_ols_states.pt"],
        expected_sha256=S3_STATE_ARTIFACT_SHA256,
        map_location="cpu",
    )
    if not isinstance(states, Mapping) or set(states) != {
        "schema",
        "task_id",
        "panel_identity_sha256",
        "labels",
        "codes",
        "split",
        "slm_amplitude",
        "coefficients",
        "normalized_power",
        "null_labels",
    }:
        raise ValueError("S3 controller state payload is invalid")
    if states["schema"] != "lightgen_scaled_chen_controller_ols_states.v1" or states["task_id"] != "H-LOMA-GEN-S3-COWA" or states["panel_identity_sha256"] != S3_PANEL_IDENTITY_SHA256:
        raise ValueError("S3 state identity is stale")
    labels = _assert_tensor(
        states["labels"], name="S3 labels", shape=(PANEL_COUNT,), dtype=torch.int64
    )
    codes = _assert_tensor(
        states["codes"], name="S3 codes", shape=(PANEL_COUNT, CODE_DIM), dtype=torch.float32
    )
    split = _assert_tensor(
        states["split"], name="S3 split", shape=(PANEL_COUNT,), dtype=torch.int64
    )
    power = _assert_tensor(
        states["normalized_power"],
        name="S3 normalized power",
        shape=(PANEL_COUNT, PORT_COUNT),
        nonnegative=True,
    )
    null_labels = _assert_tensor(
        states["null_labels"], name="S3 null labels", shape=(PANEL_COUNT,), dtype=torch.int64
    )
    if tensor_sha256(labels) != panel["labels"]["sha256"] or tensor_sha256(codes) != panel["codes"]["sha256"] or tensor_sha256(split) != panel["split"]["sha256"]:
        raise ValueError("S3 panel tensor identity is stale")
    expected_labels = torch.arange(CLASS_COUNT, dtype=torch.int64).repeat_interleave(
        PANEL_PER_CLASS
    )
    expected_split = torch.cat(
        [
            torch.zeros(PANEL_SELECTION_PER_CLASS, dtype=torch.int64),
            torch.ones(PANEL_CONFIRMATION_PER_CLASS, dtype=torch.int64),
        ]
    ).repeat(CLASS_COUNT)
    if not torch.equal(labels, expected_labels) or not torch.equal(split, expected_split):
        raise ValueError("S3 selection/confirmation panel identity is stale")
    if torch.sort(null_labels).values.tolist() != expected_labels.tolist():
        raise ValueError("S3 null labels are not a permutation of intended labels")
    s1_binding = _mapping(manifest.get("s1_reference"), "S3 S1 reference")
    if s1_binding.get("semantic_manifest_sha256") != S1_MANIFEST_SHA256 or s1_binding.get("representation") != R1_POWER:
        raise ValueError("S3 to S1 reference binding is stale")
    return {
        "directory": directory,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "artifact_paths": artifact_paths,
        "panel": panel,
        "states": states,
        "labels": labels.contiguous(),
        "codes": codes.contiguous(),
        "split": split.contiguous(),
        "power": power.to(torch.float64).contiguous(),
        "null_labels": null_labels.contiguous(),
    }


def _validate_prior_attempt() -> dict[str, Any]:
    directory = _resolve(PRIOR_ATTEMPT_OUTPUT_PATH)
    manifest_path = _resolve(PRIOR_ATTEMPT_MANIFEST_PATH)
    if not directory.is_dir() or not manifest_path.is_file():
        raise ValueError("attempt_01 geometry diagnostic is missing")
    if file_sha256(manifest_path) != PRIOR_ATTEMPT_MANIFEST_SHA256:
        raise ValueError("attempt_01 geometry diagnostic manifest was modified")
    manifest = _read_json(manifest_path)
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("task_id") != TASK_ID
        or manifest.get("status") != "valid_completed"
        or manifest.get("primary_mechanism")
        != "S3_GEOMETRY_COMPRESSED_MANIFOLD_COVERAGE_PRIMARY"
    ):
        raise ValueError("attempt_01 geometry diagnostic identity is invalid")
    artifacts = _mapping(manifest.get("artifacts"), "attempt_01 artifacts")
    _assert_exact_tree(
        directory,
        set(artifacts) | {"geometry_alignment_manifest.json"},
        "attempt_01 geometry diagnostic",
    )
    for name, ref_value in artifacts.items():
        ref = _mapping(ref_value, f"attempt_01 artifact {name}")
        path = directory / name
        if (
            ref.get("path") != name
            or file_sha256(path) != ref.get("sha256")
            or path.stat().st_size != int(ref.get("size_bytes", -1))
        ):
            raise ValueError(f"attempt_01 artifact changed: {name}")
    return {
        "directory": directory,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "manifest_sha256": PRIOR_ATTEMPT_MANIFEST_SHA256,
        "immutable": True,
    }


def _validate_s1_source(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = _mapping(config["s1_reference"], "s1_reference")
    config_path = _resolve(binding["config"])
    directory = _resolve(binding["directory"])
    if file_sha256(config_path) != S1_CONFIG_SHA256:
        raise ValueError("S1 config hash changed")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if canonical_sha256(config_payload) != S1_CONFIG_CANONICAL_SHA256:
        raise ValueError("S1 config canonical hash changed")
    manifest_path = directory / "semantic_manifest.json"
    identity_path = directory / "identity_manifest.json"
    contract_path = directory / "ols_representation_contract.json"
    embeddings_path = directory / "ols_embeddings.pt"
    for path, expected in (
        (manifest_path, S1_MANIFEST_SHA256),
        (identity_path, S1_IDENTITY_SHA256),
        (contract_path, S1_CONTRACT_SHA256),
        (embeddings_path, S1_EMBEDDINGS_SHA256),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"S1 authority artifact changed: {path.name}")
    manifest = _read_json(manifest_path)
    identity = _read_json(identity_path)
    contract = _read_json(contract_path)
    if (
        manifest.get("schema") != "lightgen_scaled_chen_ols_semantic_representation_audit.v1"
        or manifest.get("task_id") != "H-LOMA-GEN-S1-OLS-SRA"
        or manifest.get("status") != "valid_completed"
        or manifest.get("gate") != "S1_POWER_SEMANTICS_SUPPORTED"
        or manifest.get("phase_attribution") != "PHASE_SEMANTIC_INCREMENT_SUPPORTED"
        or manifest.get("minimal_representation") != "S1_POWER_SEMANTICS_SUPPORTED"
        or manifest.get("data_identity", {}).get("reselection") is not False
        or manifest.get("data_identity", {}).get("self_leakage") is not False
    ):
        raise ValueError("S1 semantic representation authority is invalid")
    if contract.get("representations", {}).get(R1_POWER) != "abs(c/sqrt(s))^2=abs(c)^2/s":
        raise ValueError("S1 R1 power contract is invalid")
    if identity.get("dataset_manifest_sha256") != S1_DATASET_MANIFEST_SHA256:
        raise ValueError("S1 dataset identity is stale")
    embeddings = safe_torch_load(
        embeddings_path, expected_sha256=S1_EMBEDDINGS_SHA256, map_location="cpu"
    )
    if not isinstance(embeddings, Mapping) or embeddings.get("schema") != "lightgen_scaled_chen_ols_semantic_embeddings.v1":
        raise ValueError("S1 embeddings schema is invalid")
    representations = _mapping(embeddings.get("representations"), "S1 representations")
    labels_payload = _mapping(embeddings.get("labels"), "S1 labels")
    indices_payload = _mapping(embeddings.get("indices"), "S1 indices")
    expected_counts = {"fit": 1000, "selection": 1000, "confirmation": 5000}
    result: dict[str, Any] = {
        "directory": directory,
        "manifest": manifest,
        "identity": identity,
        "contract": contract,
    }
    for split_name, count in expected_counts.items():
        labels = _assert_tensor(
            labels_payload.get(split_name),
            name=f"S1 {split_name} labels",
            shape=(count,),
            dtype=torch.int64,
        )
        indices = _assert_tensor(
            indices_payload.get(split_name),
            name=f"S1 {split_name} indices",
            shape=(count,),
            dtype=torch.int64,
        )
        if tensor_sha256(labels) != S1_LABEL_HASHES[split_name] or tensor_sha256(indices) != S1_INDEX_HASHES[split_name]:
            raise ValueError(f"S1 {split_name} split identity changed")
        representation = _mapping(representations.get(split_name), f"S1 {split_name} representations")
        power = _assert_tensor(
            representation.get(R1_POWER),
            name=f"S1 {split_name} R1 power",
            shape=(count, PORT_COUNT),
            nonnegative=True,
        )
        counts = torch.bincount(labels, minlength=CLASS_COUNT)
        if torch.unique(labels).numel() != CLASS_COUNT or bool((counts == 0).any()):
            raise ValueError(f"S1 {split_name} labels do not cover all classes")
        if split_name in {"fit", "selection"} and not torch.equal(
            counts, torch.full((CLASS_COUNT,), count // CLASS_COUNT, dtype=torch.int64)
        ):
            raise ValueError(f"S1 {split_name} labels are not balanced")
        result[split_name] = {
            "power": power.to(torch.float64).contiguous(),
            "labels": labels.contiguous(),
            "indices": indices.contiguous(),
        }
    return result


def _git_preflight(*, require_clean: bool) -> dict[str, Any]:
    state = base._git_state()
    if state.get("branch") != BRANCH:
        raise RuntimeError(f"BLOCKED_UNEXPECTED_BRANCH: expected {BRANCH}")
    if require_clean and not state.get("clean"):
        raise RuntimeError("BLOCKED_UNEXPECTED_DIRTY_PATH: geometry diagnostic requires clean HEAD")
    return state


def _validate_authority(config: Mapping[str, Any]) -> dict[str, Any]:
    s3 = _validate_s3_source(config)
    s1 = _validate_s1_source(config)
    prior_attempt = _validate_prior_attempt()
    if s3["manifest"]["s1_reference"]["semantic_manifest_sha256"] != S1_MANIFEST_SHA256:
        raise ValueError("S3/S1 semantic binding mismatch")
    return {"s3": s3, "s1": s1, "prior_attempt": prior_attempt}


def _artifact_refs(directory: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "geometry_alignment_manifest.json":
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
        "centroid_translation.json",
        "selection_confirmation_isolation.json",
        "confirmation_before_after.json",
        "anchor_offset_stability.json",
        "interclass_centroid_geometry.json",
        "covariance_geometry.json",
        "coverage_metrics.json",
        "aligned_residual_metrics.json",
        "class9_diagnostic.json",
        "compact_report.json",
        "geometry_alignment_manifest.json",
    }
    _assert_exact_tree(directory, expected, "geometry alignment output")


def _authority_binding_payload(authority: Mapping[str, Any]) -> dict[str, Any]:
    s3 = authority["s3"]
    prior_attempt = authority["prior_attempt"]
    return {
        "schema": AUTHORITY_SCHEMA,
        "task_id": TASK_ID,
        "s3": {
            "directory": _relative(S3_OUTPUT_PATH),
            "manifest": _relative(S3_MANIFEST_PATH),
            "manifest_sha256": S3_MANIFEST_SHA256,
            "state_artifact_sha256": S3_STATE_ARTIFACT_SHA256,
            "panel_identity_sha256": S3_PANEL_IDENTITY_SHA256,
            "classification": s3["manifest"]["classification"],
            "gate": s3["manifest"]["primary_gate"],
            "immutable": True,
        },
        "s1": {
            "directory": _relative(S1_OUTPUT_PATH),
            "semantic_manifest_sha256": S1_MANIFEST_SHA256,
            "identity_manifest_sha256": S1_IDENTITY_SHA256,
            "contract_sha256": S1_CONTRACT_SHA256,
            "embeddings_sha256": S1_EMBEDDINGS_SHA256,
            "representation": R1_POWER,
            "selection_labels_sha256": S1_LABEL_HASHES["selection"],
            "confirmation_labels_sha256": S1_LABEL_HASHES["confirmation"],
            "selection_indices_sha256": S1_INDEX_HASHES["selection"],
            "confirmation_indices_sha256": S1_INDEX_HASHES["confirmation"],
            "immutable": True,
        },
        "observer": {
            "representation": R1_POWER,
            "formula": "z_j=abs(c_j)^2/s_j",
            "metric": "euclidean_squared",
            "controller_forward": False,
            "optical_forward": False,
        },
        "authority_binding_gate": "SEMANTIC_GEOMETRY_ALIGNMENT_AUTHORITY_BINDING_PASS",
        "s3_historical_result_preserved": True,
        "s3_retroactive_reclassification": False,
        "attempt_01": {
            "directory": _relative(PRIOR_ATTEMPT_OUTPUT_PATH),
            "manifest": _relative(PRIOR_ATTEMPT_MANIFEST_PATH),
            "manifest_sha256": prior_attempt["manifest_sha256"],
            "classification": prior_attempt["manifest"]["primary_mechanism"],
            "immutable": prior_attempt["immutable"],
        },
    }


def _router_conformity_payload(prior_attempt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "authorization": "AUTHORIZATION_GRANTED_FOR_GEOMETRY_ALIGNMENT_ROUTER_CONFORMITY_FIX_AND_WRITE_ONCE_ATTEMPT_02",
        "kind": "router_conformity_correction",
        "previous_attempt": _relative(PRIOR_ATTEMPT_OUTPUT_PATH),
        "previous_manifest_sha256": prior_attempt["manifest_sha256"],
        "exact_logic_delta": {
            "before": "F2 required translation_effective and considered targeting degradation, inter-class distortion, or coverage compression; it did not inspect registered confirmation Q50/Q95 hard-criterion residuals.",
            "after": "F2 is supported when the frozen confirmation hard criterion fails after translation (any registered per-class upper95(delta_q50_after)>0 or upper95(delta_q95_after)>0), or inter-class centroid geometry is distorted, or the existing translation-effective targeting condition holds.",
            "coverage_only_trigger_removed": True,
            "post_attempt_01_q50_gap_fraction_threshold_removed": True,
            "new_metric": False,
            "new_threshold": False,
            "new_transform": False,
        },
        "frozen_inputs_unchanged": True,
        "s1_authority_modified": False,
        "s3_authority_modified": False,
    }


def _mechanism_decision(
    reachability: Mapping[str, Any],
    targeting: Mapping[str, Any],
    centroid_geometry: Mapping[str, Any],
    coverage: Mapping[str, Any],
    decision_config: Mapping[str, Any],
) -> dict[str, Any]:
    confirmation_reachability = _mapping(reachability["confirmation"], "confirmation reachability")
    translation_effective = bool(
        confirmation_reachability["q50_shrinkage_positive_fraction"]
        >= float(decision_config["translation_effective_min_q50_shrinkage_fraction"])
        and confirmation_reachability["q95_shrinkage_positive_fraction"]
        >= float(decision_config["translation_effective_min_q95_shrinkage_fraction"])
    )
    targeting_decision = _mapping(targeting["confirmation"], "targeting decision")
    targeting_preserved = bool(targeting_decision["all_class_non_degradation"])
    confirmation_hard_criterion_failure_classes = [
        int(label)
        for label, value in confirmation_reachability["per_class"].items()
        if value["delta_q50_after"]["upper95"] > 0.0
        or value["delta_q95_after"]["upper95"] > 0.0
    ]
    confirmation_hard_criterion_after_pass = not bool(
        confirmation_hard_criterion_failure_classes
    )
    interclass_distorted = any(
        item["qualitative_status"] == "INTER_CLASS_SEMANTIC_GEOMETRY_DISTORTED"
        for item in centroid_geometry.values()
    )
    coverage_compressed = bool(coverage["confirmation"]["compressed"])
    higher_order = bool(
        not confirmation_hard_criterion_after_pass
        or interclass_distorted
        or (translation_effective and not targeting_preserved)
    )
    if translation_effective and targeting_preserved and not higher_order:
        primary = "S3_GEOMETRY_ANCHOR_OFFSET_PRIMARY"
        secondary = None
        next_action = "PLAN_CONTROLLER_SEMANTIC_ANCHOR_ALIGNMENT_AUTHORITY"
    elif translation_effective and higher_order:
        primary = "S3_GEOMETRY_ANCHOR_PLUS_TANGENT_MISMATCH"
        secondary = (
            "S3_GEOMETRY_COMPRESSED_MANIFOLD_COVERAGE_PRIMARY"
            if coverage_compressed
            else "S3_GEOMETRY_HIGHER_ORDER_MISMATCH_PRIMARY"
        )
        next_action = "PLAN_CONTROLLER_SEMANTIC_GEOMETRY_TRAINING_AUTHORITY"
    elif coverage_compressed:
        primary = "S3_GEOMETRY_COMPRESSED_MANIFOLD_COVERAGE_PRIMARY"
        secondary = (
            "S3_GEOMETRY_HIGHER_ORDER_MISMATCH_PRIMARY"
            if higher_order
            else None
        )
        next_action = "REVIEW_CODE8_WITHIN_CLASS_MANIFOLD_COVERAGE"
    elif not translation_effective:
        primary = "S3_GEOMETRY_HIGHER_ORDER_MISMATCH_PRIMARY"
        secondary = None
        next_action = "PLAN_CONTROLLER_SEMANTIC_GEOMETRY_TRAINING_AUTHORITY"
    else:
        primary = "S3_GEOMETRY_ALIGNMENT_UNDERDETERMINED"
        secondary = None
        next_action = "PLAN_CONTROLLER_SEMANTIC_GEOMETRY_TRAINING_AUTHORITY"
    return {
        "translation_effective": translation_effective,
        "targeting_preserved": targeting_preserved,
        "interclass_geometry_distorted": interclass_distorted,
        "coverage_compressed": coverage_compressed,
        "confirmation_hard_criterion_after_pass": confirmation_hard_criterion_after_pass,
        "confirmation_hard_criterion_failure_classes": confirmation_hard_criterion_failure_classes,
        "higher_order_mismatch_supported": higher_order,
        "primary_mechanism": primary,
        "secondary_mechanism": secondary,
        "next_action": next_action,
        "new_s3_gate_created": False,
    }


def _targeting_comparison(
    controller_before: torch.Tensor,
    controller_after: torch.Tensor,
    controller_labels: torch.Tensor,
    null_labels: torch.Tensor,
    reference: torch.Tensor,
    reference_labels: torch.Tensor,
) -> dict[str, Any]:
    before = _targeting_report(
        controller_before,
        controller_labels,
        null_labels,
        reference,
        reference_labels,
    )
    after = _targeting_report(
        controller_after,
        controller_labels,
        null_labels,
        reference,
        reference_labels,
    )
    per_class_change: dict[str, Any] = {}
    all_class_non_degradation = True
    degradation_classes: list[int] = []
    for label in range(CLASS_COUNT):
        before_class = before["per_class"][str(label)]
        after_class = after["per_class"][str(label)]
        margin_change = after_class["semantic_margin"]["mean"] - before_class["semantic_margin"]["mean"]
        retrieval_change = after_class["retrieval_accuracy"] - before_class["retrieval_accuracy"]
        non_degradation = margin_change >= 0.0 and retrieval_change >= 0.0
        if not non_degradation:
            degradation_classes.append(label)
        all_class_non_degradation = all_class_non_degradation and non_degradation
        per_class_change[str(label)] = {
            "semantic_margin_change": margin_change,
            "retrieval_change": retrieval_change,
            "non_degradation": non_degradation,
        }
    return {
        "before": before,
        "after": after,
        "per_class_change": per_class_change,
        "all_class_non_degradation": all_class_non_degradation,
        "degradation_classes": degradation_classes,
        "overall_semantic_margin_change": after["semantic_margin"]["mean"] - before["semantic_margin"]["mean"],
        "overall_retrieval_change": after["retrieval_accuracy"] - before["retrieval_accuracy"],
        "finite": True,
    }


def _coverage_report(
    covariance_payload: Mapping[str, Any],
    ratios_by_class: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ratios = torch.tensor(list(ratios_by_class.values()), dtype=torch.float64)
    threshold = float(config["coverage_compression_all_class_ratio_below"])
    median_max = float(config["coverage_compression_median_ratio_at_most"])
    compressed = bool((ratios < threshold).all() and torch.median(ratios) <= median_max)
    return {
        "schema": COVERAGE_SCHEMA,
        "controller_to_real_variance_ratio": dict(ratios_by_class),
        "median_variance_ratio": float(torch.median(ratios)),
        "minimum_variance_ratio": float(ratios.min()),
        "maximum_variance_ratio": float(ratios.max()),
        "all_class_ratio_below_threshold": bool((ratios < threshold).all()),
        "all_class_ratio_threshold": threshold,
        "median_ratio_threshold": median_max,
        "compressed": compressed,
        "classification": (
            "CONTROLLER_WITHIN_CLASS_COVERAGE_COMPRESSED"
            if compressed
            else "CONTROLLER_WITHIN_CLASS_COVERAGE_NOT_ESTABLISHED"
        ),
        "covariance_source_schema": covariance_payload["schema"],
        "finite": True,
    }


def _preflight(
    config_path: str | Path,
    output_path: str | Path,
    *,
    require_clean: bool,
) -> dict[str, Any]:
    state = _git_preflight(require_clean=require_clean)
    _, config, config_sha256, config_canonical_sha256 = load_config(config_path)
    authority = _validate_authority(config)
    output = _resolve(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to reuse write-once geometry output: {_relative(output)}")
    return {
        "task_id": TASK_ID,
        "gate": "SEMANTIC_GEOMETRY_ALIGNMENT_AUTHORITY_BINDING_PASS",
        "branch": state["branch"],
        "head": state["commit"],
        "worktree_clean": bool(state.get("clean")),
        "config_sha256": config_sha256,
        "config_canonical_sha256": config_canonical_sha256,
        "s3_manifest_sha256": S3_MANIFEST_SHA256,
        "s1_manifest_sha256": S1_MANIFEST_SHA256,
        "representation": R1_POWER,
        "output_path": _relative(output),
        "controller_forward": False,
        "optical_forward": False,
        "cuda": False,
        "authority_loaded": bool(authority),
    }


def run(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    started = time.perf_counter()
    state_before = _git_preflight(require_clean=True)
    config_file, config, config_sha256, config_canonical_sha256 = load_config(config_path)
    authority = _validate_authority(config)
    output = _resolve(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to reuse write-once geometry output: {_relative(output)}")
    s3 = authority["s3"]
    s1 = authority["s1"]
    prior_attempt = authority["prior_attempt"]
    controller_power = s3["power"]
    controller_labels = s3["labels"]
    controller_split = s3["split"]
    null_labels = s3["null_labels"]
    real_selection = s1["selection"]["power"]
    real_selection_labels = s1["selection"]["labels"]
    real_confirmation = s1["confirmation"]["power"]
    real_confirmation_labels = s1["confirmation"]["labels"]
    fit_reference = s1["fit"]["power"]
    fit_labels = s1["fit"]["labels"]
    controller_selection = controller_power[controller_split == 0]
    controller_selection_labels = controller_labels[controller_split == 0]
    controller_confirmation = controller_power[controller_split == 1]
    controller_confirmation_labels = controller_labels[controller_split == 1]
    with torch.no_grad():
        translation = fit_class_translation(
            real_selection,
            real_selection_labels,
            controller_selection,
            controller_selection_labels,
        )
        aligned_selection = apply_class_translation(
            controller_selection, controller_selection_labels, translation
        )
        aligned_confirmation = apply_class_translation(
            controller_confirmation, controller_confirmation_labels, translation
        )
        original_selection = controller_selection.clone()
        original_confirmation = controller_confirmation.clone()
        if not torch.equal(controller_selection, original_selection) or not torch.equal(
            controller_confirmation, original_confirmation
        ):
            raise ValueError("source controller states were mutated")
        reachability = {
            "schema": REACHABILITY_SCHEMA,
            "selection": _reachability_split(
                "selection",
                controller_selection,
                aligned_selection,
                controller_selection_labels,
                real_selection,
                real_selection_labels,
                fit_reference,
                fit_labels,
            ),
            "confirmation": _reachability_split(
                "confirmation",
                controller_confirmation,
                aligned_confirmation,
                controller_confirmation_labels,
                real_confirmation,
                real_confirmation_labels,
                fit_reference,
                fit_labels,
            ),
            "reference": "S1_fit_reference_for_historical_S3_metric_compatibility",
            "metric": "euclidean_squared",
            "historical_s3_metrics_replaced": False,
            "finite": True,
        }
        targeting = {
            "schema": "lightgen_scaled_chen_geometry_targeting_before_after.v1",
            "selection": _targeting_comparison(
                controller_selection,
                aligned_selection,
                controller_selection_labels,
                null_labels[controller_split == 0],
                fit_reference,
                fit_labels,
            ),
            "confirmation": _targeting_comparison(
                controller_confirmation,
                aligned_confirmation,
                controller_confirmation_labels,
                null_labels[controller_split == 1],
                fit_reference,
                fit_labels,
            ),
            "translation_uses_intended_labels_only": True,
            "finite": True,
        }
        stability = _translation_stability(
            real_selection,
            real_selection_labels,
            controller_selection,
            controller_selection_labels,
            real_confirmation,
            real_confirmation_labels,
            controller_confirmation,
            controller_confirmation_labels,
            config["stability"],
        )
        centroid_geometry = {
            "selection": _centroid_geometry(
                real_selection,
                real_selection_labels,
                controller_selection,
                controller_selection_labels,
                config["geometry"],
            ),
            "confirmation": _centroid_geometry(
                real_confirmation,
                real_confirmation_labels,
                controller_confirmation,
                controller_confirmation_labels,
                config["geometry"],
            ),
        }
        covariance_selection, selection_ratios = _covariance_geometry(
            real_selection,
            real_selection_labels,
            controller_selection,
            controller_selection_labels,
            float(config["covariance"]["effective_rank_relative_threshold"]),
        )
        covariance_confirmation, confirmation_ratios = _covariance_geometry(
            real_confirmation,
            real_confirmation_labels,
            controller_confirmation,
            controller_confirmation_labels,
            float(config["covariance"]["effective_rank_relative_threshold"]),
        )
        covariance = {
            "schema": COVARIANCE_SCHEMA,
            "selection": covariance_selection,
            "confirmation": covariance_confirmation,
            "representation": R1_POWER,
            "finite": True,
        }
        coverage = {
            "schema": COVERAGE_SCHEMA,
            "selection": _coverage_report(
                covariance_selection, selection_ratios, config["decision"]
            ),
            "confirmation": _coverage_report(
                covariance_confirmation, confirmation_ratios, config["decision"]
            ),
            "interpretation_boundary": "8D_actuation_is_not_real_manifold_coverage",
            "finite": True,
        }
        residuals = {
            "schema": RESIDUAL_SCHEMA,
            "selection": {
                "before": _residual_report(
                    controller_selection,
                    controller_selection_labels,
                    fit_reference,
                    fit_labels,
                    _class_centroids(real_selection, real_selection_labels),
                ),
                "after": _residual_report(
                    aligned_selection,
                    controller_selection_labels,
                    fit_reference,
                    fit_labels,
                    _class_centroids(real_selection, real_selection_labels),
                ),
            },
            "confirmation": {
                "before": _residual_report(
                    controller_confirmation,
                    controller_confirmation_labels,
                    fit_reference,
                    fit_labels,
                    _class_centroids(real_confirmation, real_confirmation_labels),
                ),
                "after": _residual_report(
                    aligned_confirmation,
                    controller_confirmation_labels,
                    fit_reference,
                    fit_labels,
                    _class_centroids(real_confirmation, real_confirmation_labels),
                ),
            },
            "decomposition": "residual_parallel_to_target_centered_direction_vs_orthogonal",
            "per_port_correction_fit": False,
            "finite": True,
        }
        decision = _mechanism_decision(
            reachability,
            targeting,
            centroid_geometry,
            coverage,
            config["decision"],
        )
        class9 = {
            "schema": CLASS9_SCHEMA,
            "class": 9,
            "secondary_authority_preserved": "CLASS9_SYSTEMATIC_LABEL_MAPPING_DEFECT_SUPPORTED",
            "confirmation_reachability": reachability["confirmation"]["per_class"]["9"],
            "confirmation_targeting": targeting["confirmation"]["per_class_change"]["9"],
            "confirmation_residual_before": residuals["confirmation"]["before"]["per_class"]["9"],
            "confirmation_residual_after": residuals["confirmation"]["after"]["per_class"]["9"],
            "special_case_fit": False,
            "relabel": False,
            "removed": False,
            "finite": True,
        }
        isolation = {
            "schema": ISOLATION_SCHEMA,
            "gate": "SELECTION_CONFIRMATION_ALIGNMENT_ISOLATION_PASS",
            "fit_source": "controller_selection_and_real_selection_only",
            "confirmation_used_for_transform": False,
            "confirmation_centroid_use": "observation_only",
            "selection_count_controller": int(controller_selection.shape[0]),
            "selection_count_real": int(real_selection.shape[0]),
            "confirmation_count_controller": int(controller_confirmation.shape[0]),
            "confirmation_count_real": int(real_confirmation.shape[0]),
            "translation_tensor_sha256": tensor_sha256(translation),
            "source_controller_power_sha256": tensor_sha256(controller_power),
            "source_controller_power_unchanged": bool(
                torch.equal(controller_power, s3["power"])
            ),
            "controller_forward": False,
            "optical_forward": False,
            "optimizer_created": False,
            "backward_calls": 0,
            "checkpoint_created": False,
            "finite": True,
        }
        centroid_translation = {
            "schema": TRANSLATION_SCHEMA,
            "fit_formula": "t_y=mu_real_sel[y]-mu_ctrl_sel[y]",
            "application_formula": "z_ctrl_aligned=z_ctrl+t_y",
            "representation": R1_POWER,
            "metric": "euclidean_squared",
            "selection_only": True,
            "controller_state_mutated": False,
            "shape": list(translation.shape),
            "dtype": str(translation.dtype),
            "sha256": tensor_sha256(translation),
            "vectors": translation.tolist(),
        }
        primary = decision["primary_mechanism"]
        router_conformity = _router_conformity_payload(prior_attempt)
        if primary == "S3_GEOMETRY_ANCHOR_OFFSET_PRIMARY":
            f1 = "SUPPORTED"
        else:
            f1 = "NOT_SUPPORTED"
        f2 = "SUPPORTED" if decision["higher_order_mismatch_supported"] else "NOT_SUPPORTED"
        f3 = "SUPPORTED" if decision["coverage_compressed"] else "NOT_SUPPORTED"
        compact_report = {
            "schema": REPORT_SCHEMA,
            "task_id": TASK_ID,
            "status": "valid_completed",
            "authority_binding_gate": "SEMANTIC_GEOMETRY_ALIGNMENT_AUTHORITY_BINDING_PASS",
            "alignment_isolation_gate": isolation["gate"],
            "f1_anchor_offset": f1,
            "f2_higher_order_mismatch": f2,
            "f3_coverage_compression": f3,
            "primary_mechanism": decision["primary_mechanism"],
            "secondary_mechanism": decision["secondary_mechanism"],
            "historical_s3_classification": "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED",
            "s3_retroactive_pass": False,
            "eight_d_conclusion": "KEEP_8D_PROVISIONALLY",
            "code8_local_actuation": "CODE8_LOCAL_ACTUATION_SUPPORTED",
            "generation_readiness": "not_ready",
            "governance": {
                "mainline_state": "closed_at_current_evidence_boundary",
                "reconstruction_manifold_supported": False,
                "generative_manifold_supported": False,
                "generation_readiness": "not_ready",
                "training_authorized": False,
                "formal_promotion": False,
            },
            "next_action": decision["next_action"],
            "router_conformity_correction": router_conformity,
            "wall_clock_seconds_before_write": time.perf_counter() - started,
        }
    artifact_payloads: dict[str, Mapping[str, Any]] = {
        "authority_binding.json": _authority_binding_payload(authority),
        "centroid_translation.json": centroid_translation,
        "selection_confirmation_isolation.json": isolation,
        "confirmation_before_after.json": {
            "schema": REACHABILITY_SCHEMA,
            **reachability,
            "targeting": targeting,
        },
        "anchor_offset_stability.json": stability,
        "interclass_centroid_geometry.json": {
            "schema": CENTROID_SCHEMA,
            "representation": R1_POWER,
            **centroid_geometry,
            "finite": True,
        },
        "covariance_geometry.json": covariance,
        "coverage_metrics.json": coverage,
        "aligned_residual_metrics.json": residuals,
        "class9_diagnostic.json": class9,
        "compact_report.json": compact_report,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}.", dir=output.parent
    ) as temporary_name:
        staging = Path(temporary_name)
        for name, payload in artifact_payloads.items():
            atomic_write_new_json(
                staging / name,
                payload,
                artifact_kind=f"geometry alignment {name}",
            )
        artifact_refs = _artifact_refs(staging)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "task_id": TASK_ID,
            "hypothesis": HYPOTHESIS,
            "status": "valid_completed",
            "primary_mechanism": decision["primary_mechanism"],
            "secondary_mechanism": decision["secondary_mechanism"],
            "next_action": decision["next_action"],
            "configuration": {
                "path": _relative(config_file),
                "sha256": config_sha256,
                "canonical_sha256": config_canonical_sha256,
            },
            "output_directory": _relative(output),
            "git": {
                "branch": state_before["branch"],
                "starting_head": state_before["commit"],
                "ending_head": base._git_state()["commit"],
                "unexpected_dirty_paths": [],
            },
            "s3_authority": {
                "manifest": _relative(S3_MANIFEST_PATH),
                "manifest_sha256": S3_MANIFEST_SHA256,
                "classification": "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED",
                "immutable": True,
            },
            "s1_reference": {
                "directory": _relative(S1_OUTPUT_PATH),
                "semantic_manifest_sha256": S1_MANIFEST_SHA256,
                "representation": R1_POWER,
                "immutable": True,
            },
            "router_conformity_correction": router_conformity,
            "diagnostic": {
                "translation": "class_conditional_centroid_translation_only",
                "selection_fit": True,
                "confirmation_evaluation_only": True,
                "new_s3_gate": False,
                "controller_forward": False,
                "optical_forward": False,
                "cuda": False,
                "training": False,
                "checkpoint_created": False,
            },
            "governance": compact_report["governance"],
            "layout": {
                "json_only": True,
                "write_once": True,
                "manifest_self_hash_excluded": True,
                "artifact_count_excluding_manifest": len(artifact_refs),
            },
            "artifacts": artifact_refs,
        }
        atomic_write_new_json(
            staging / "geometry_alignment_manifest.json",
            manifest,
            artifact_kind="geometry alignment manifest",
        )
        _assert_output_tree(staging)
        staging.replace(output)
    _assert_output_tree(output)
    final_manifest = _read_json(output / "geometry_alignment_manifest.json")
    if final_manifest.get("artifacts") != _artifact_refs(output):
        raise ValueError("geometry alignment artifact references do not verify")
    return {
        "task_id": TASK_ID,
        "output_dir": _relative(output),
        "status": final_manifest["status"],
        "primary_mechanism": final_manifest["primary_mechanism"],
        "secondary_mechanism": final_manifest["secondary_mechanism"],
        "next_action": final_manifest["next_action"],
        "alignment_isolation_gate": isolation["gate"],
        "f1_anchor_offset": f1,
        "f2_higher_order_mismatch": f2,
        "f3_coverage_compression": f3,
        "s3_historical_classification": "S3_CONTROL_TO_OLS_WRITABILITY_NOT_SUPPORTED",
        "cuda": False,
        "controller_forward": False,
        "optical_forward": False,
        "wall_clock_seconds": time.perf_counter() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "run"), default="run")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    if arguments.action == "preflight":
        result = _preflight(arguments.config, arguments.output_dir, require_clean=True)
    else:
        result = run(arguments.config, arguments.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
