"""Localize step-6000 cross-evaluator semantic divergence.

This is a frozen, forward-only image-manifold audit.  It consumes the accepted
step-6000 Generator, exact S1 bank, real DEV OLS outputs, canonical TRAIN/DEV
images and E_S/E_I/E_C/E_D.  It fits one unsupervised PCA reference on
canonical TRAIN images only and does not train, refit, select or create any
evaluator or sampler.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.transforms import ScaledTranslation

from ldm_project.v2.evidence import file_sha256, tensor_sha256
from scripts.research import execute_lightgen_2d_assess_step6000_conditional_generation_readiness_v1 as readiness


# Nature-figure Python backend contract: editable SVG/PDF text.
# Source-audit contract: svg.fonttype='none'; pdf.fonttype=42; exports .svg/.pdf/.png/.tiff; dpi=600.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 6
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False
RASTER_DPI = 600


TASK_ID = "LOCALIZE_STEP6000_IMAGE_MANIFOLD_SEMANTIC_DIVERGENCE_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_LOCALIZE_STEP6000_IMAGE_MANIFOLD_SEMANTIC_DIVERGENCE_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "STEP6000_IMAGE_MANIFOLD_SEMANTIC_DIVERGENCE_LOCALIZED"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260921_step6000_image_manifold_semantic_divergence_localization_v1/runs/attempt_01"
)

READINESS_RUN = Path("outputs/20260921_step6000_conditional_generation_readiness_v1/runs/attempt_06")
READINESS_REPORT = READINESS_RUN / "report.json"
READINESS_REPORT_SHA256 = "0cdbbd1e45ee051500b004536c8216dd44a9ec27fc1eddbaa359f848b4a3c4ae"
READINESS_STATUS = READINESS_RUN / "status.json"
READINESS_STATUS_SHA256 = "5314ac1cae8821458cf34055bae7f9dd2f1f6dccd125556214be0f39c75e7517"
READINESS_SELECTION = READINESS_RUN / "fixed_selection_manifest.json"
READINESS_SELECTION_SHA256 = "2d198c652dadf684b06b4e70ff195d18896c4bc9de689c34ae159072615bba55"

STEP3000_SHA256 = "1a87af8aa4fa12e43dfd4ab7ee99aa41e08299843f5414cfd05d0221dc473a3d"
STEP6000_SHA256 = "3387e104f7d33e19051a23444b0dabf1eb387c5faca3ce7150bbd1a2212c9986"

EVALUATOR_ORDER = ("E_S", "E_I", "E_C", "E_D")
CLASS_COUNT = 10
BATCH_SIZE = 10
IMAGE_PIXELS = 28 * 28
PCA_VARIANCE_TARGET = 0.95
PCA_RANK_CAP = 256
DISTANCE_BLOCK = 128
INTERPOLATION_LAMBDAS = (0.0, 0.25, 0.50, 0.75, 1.0)
STRATUM_PANEL_PER_CLASS = 1
WRONG_BASIN_MAJORITY = 0.50

REPORT_SCHEMA = "step6000_image_manifold_semantic_divergence_localization_report.v1"
STATUS_SCHEMA = "step6000_image_manifold_semantic_divergence_localization_status.v1"
PREFLIGHT_SCHEMA = "step6000_image_manifold_semantic_divergence_localization_preflight.v1"

CLASSIFICATION_OFF_MANIFOLD = "STEP6000_OFF_MANIFOLD_IMAGE_REALIZATION_GAP_SUPPORTED"
CLASSIFICATION_SHARED = "STEP6000_ON_MANIFOLD_CROSS_EVALUATOR_DECISION_DIVERGENCE_SUPPORTED"
CLASSIFICATION_CLASS_SPECIFIC = "STEP6000_CLASS_SPECIFIC_CANONICAL_MORPHOLOGY_GAP_SUPPORTED"
CLASSIFICATION_TRANSFER = "STEP6000_REAL_VS_S1_IMAGE_REALIZATION_TRANSFER_GAP_SUPPORTED"
CLASSIFICATION_WRONG_BASIN = "STEP6000_WRONG_CLASS_CANONICAL_BASIN_REALIZATION_SUPPORTED"
CLASSIFICATION_MIXED = "STEP6000_IMAGE_MANIFOLD_SEMANTIC_DIVERGENCE_MIXED"


class LocalizationError(RuntimeError):
    """Raised when an authority or read-only audit invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LocalizationError(message)


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _tensor_or_list(value: object, name: str, *, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=dtype)
    if isinstance(value, list):
        return torch.tensor(value, dtype=dtype)
    raise LocalizationError(f"{name} must be a tensor or list")


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"missing authority artifact: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    readiness._write_json(path, payload, kind=kind)


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    readiness._write_torch(path, payload, kind=kind)


def _summary(value: torch.Tensor, name: str) -> dict[str, Any]:
    return readiness._summary(value, name)


def _panel_label(index: int) -> str:
    value = index
    label = ""
    while True:
        label = chr(97 + (value % 26)) + label
        value = value // 26 - 1
        if value < 0:
            return label


def _add_panel_label(axis: Any, label: str) -> None:
    offset = ScaledTranslation(-3 / 72, 3 / 72, axis.figure.dpi_scale_trans)
    axis.text(0, 1, label, transform=axis.transAxes + offset, ha="left", va="bottom", fontsize=7, fontweight="bold")


def _save_figure(
    fig: Any,
    target: Path,
    stem: str,
    axes: Sequence[Any],
    panel_ids: Sequence[str],
    *,
    row_groups: Sequence[Sequence[str]] | None = None,
    column_groups: Sequence[Sequence[str]] | None = None,
    layout_mode: str = "tight",
) -> dict[str, Any]:
    if layout_mode == "tight":
        fig.tight_layout(pad=0.7, rect=(0.0, 0.055, 1.0, 0.965))
    elif layout_mode == "grid":
        fig.subplots_adjust(left=0.075, right=0.995, bottom=0.09, top=0.95, wspace=0.08, hspace=0.50)
    else:
        raise ValueError(f"unknown figure layout mode: {layout_mode}")
    from audit_panel_alignment import require_matplotlib_panel_alignment

    options: dict[str, Any] = {
        "axes": list(axes),
        "panel_ids": list(panel_ids),
        "tolerance_pt": 1.5,
        "gutter_tolerance_pt": 1.5,
        "require_panel_labels": True,
        "strict": True,
        "json_out": str(target / f"{stem}.alignment.json"),
        "overlay_svg": str(target / f"{stem}.alignment.svg"),
    }
    if row_groups is not None:
        options["row_groups"] = [list(group) for group in row_groups]
    if column_groups is not None:
        options["column_groups"] = [list(group) for group in column_groups]
    require_matplotlib_panel_alignment(fig, **options)
    outputs: dict[str, Any] = {}
    for extension, kwargs in (
        ("svg", {}),
        ("pdf", {}),
        ("png", {"dpi": RASTER_DPI}),
        ("tiff", {"dpi": RASTER_DPI}),
    ):
        path = target / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs[extension] = {"path": path.name, "sha256": file_sha256(path)}
    outputs["alignment_json"] = {"path": f"{stem}.alignment.json", "sha256": file_sha256(target / f"{stem}.alignment.json")}
    outputs["alignment_svg"] = {"path": f"{stem}.alignment.svg", "sha256": file_sha256(target / f"{stem}.alignment.svg")}
    plt.close(fig)
    return outputs


def _validate_image_tensor(images: torch.Tensor, name: str) -> None:
    _require(images.ndim == 4 and tuple(images.shape[1:]) == (1, 28, 28), f"{name} geometry changed")
    _finite(images, name)
    _require(float(images.min()) >= 0.0 and float(images.max()) <= 1.0, f"{name} range changed")


def _classification_and_consensus(logits: Mapping[str, torch.Tensor], labels: torch.Tensor, *, label_source: str) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    predictions: dict[str, torch.Tensor] = {}
    for name in EVALUATOR_ORDER:
        metrics[name], predictions[name] = readiness._classification_report(logits[name], labels, label_source=label_source)
    consensus = readiness._consensus_report(predictions, labels)
    return metrics, predictions, consensus


def _raw_validity(views: torch.Tensor, *, source: str) -> dict[str, Any]:
    flat = views.detach().cpu().to(torch.float64).flatten(1)
    finite = torch.isfinite(flat).all(dim=1)
    nonconstant = (flat.max(dim=1).values > flat.min(dim=1).values)
    _require(bool(finite.all()), f"{source} view is nonfinite")
    return {
        "source": source,
        "finite": True,
        "finite_fraction": float(finite.to(torch.float64).mean()),
        "nonconstant_fraction": float(nonconstant.to(torch.float64).mean()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "shape": list(views.shape),
    }


def _evaluate_views(views: torch.Tensor, labels: torch.Tensor, evaluators: Mapping[str, torch.nn.Module], device: torch.device, *, label_source: str) -> dict[str, Any]:
    _validate_image_tensor(views, "evaluator views")
    labels = labels.to(torch.long).cpu().contiguous()
    _require(views.shape[0] == labels.numel(), "view/label count changed")
    logits_lists: dict[str, list[torch.Tensor]] = {name: [] for name in EVALUATOR_ORDER}
    with torch.inference_mode():
        for start in range(0, views.shape[0], BATCH_SIZE):
            end = min(views.shape[0], start + BATCH_SIZE)
            batch = views[start:end].to(device)
            for name in ("E_S", "E_I", "E_C"):
                _, logits = evaluators[name](batch)
                logits_lists[name].append(logits.detach().cpu())
            logits_lists["E_D"].append(evaluators["E_D"](batch).detach().cpu())
    logits = {name: torch.cat(values, dim=0).contiguous() for name, values in logits_lists.items()}
    metrics, predictions, consensus = _classification_and_consensus(logits, labels, label_source=label_source)
    return {
        "views": views.cpu().contiguous(),
        "labels": labels,
        "logits": logits,
        "metrics": metrics,
        "predictions": predictions,
        "consensus": consensus,
        "raw_validity": _raw_validity(views, source=label_source),
    }


def _forward_generated(consumer: Any, evaluators: Mapping[str, torch.nn.Module], z: torch.Tensor, labels: torch.Tensor, device: torch.device, *, source: str) -> dict[str, Any]:
    _require(z.ndim == 2 and z.shape[1] == 100 and z.is_complex(), f"{source} OLS geometry changed")
    views: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, z.shape[0], BATCH_SIZE):
            end = min(z.shape[0], start + BATCH_SIZE)
            result = consumer.generated_forward(z[start:end].to(device))
            raw = result.raw_intensity.detach()
            _finite(raw, f"{source} raw CMOS")
            view = readiness.continuation._cmos_view(raw).detach().cpu()
            views.append(view)
            if end % 500 == 0 or end == z.shape[0]:
                print(json.dumps({"phase": "FORWARD", "source": source, "processed": end, "total": int(z.shape[0])}, ensure_ascii=False), flush=True)
    combined = torch.cat(views, dim=0).contiguous()
    return _evaluate_views(combined, labels, evaluators, device, label_source="requested class" if source == "s1" else "true class")


def _state_digest(consumer: Any) -> str:
    return readiness._state_digest({name: value.detach().cpu() for name, value in consumer.runtime.state_dict().items()})


def _no_update_preflight(consumer: Any, evaluators: Mapping[str, torch.nn.Module], checkpoint: Mapping[str, Any], real_images: torch.Tensor, s1_z: torch.Tensor, device: torch.device) -> dict[str, Any]:
    consumer.load_trainable_state_dict({"encoder.phase": checkpoint["encoder_state"], "generator.phase": checkpoint["generator_state"]})
    before_runtime = _state_digest(consumer)
    before_evaluators = {name: readiness._module_digest(model) for name, model in evaluators.items()}
    with torch.inference_mode():
        real_result = consumer.runtime.forward_from_image(real_images[:BATCH_SIZE].to(device))
        generated_result = consumer.generated_forward(s1_z[:BATCH_SIZE].to(device))
        _finite(real_result.raw_intensity, "preflight real raw")
        _finite(generated_result.raw_intensity, "preflight S1 raw")
    _require(_state_digest(consumer) == before_runtime, "no-update preflight changed runtime state")
    _require(before_evaluators == {name: readiness._module_digest(model) for name, model in evaluators.items()}, "no-update preflight changed evaluator state")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "preflight left runtime gradients")
    _require(all(parameter.grad is None for model in evaluators.values() for parameter in model.parameters()), "preflight left evaluator gradients")
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": "READY",
        "no_update": True,
        "optimizer_created": False,
        "backward_path_used": False,
        "sampler_fit": False,
        "representative_real_forward": True,
        "representative_s1_forward": True,
        "runtime_state_unchanged": True,
        "evaluator_state_unchanged": True,
        "all_parameters_grad_none": True,
        "protected_test_read": False,
        "sealed_read": False,
    }


def _fit_pca(train_flat: torch.Tensor, device: torch.device) -> dict[str, Any]:
    _require(train_flat.ndim == 2 and train_flat.shape[1] == IMAGE_PIXELS, "PCA input geometry changed")
    x = train_flat.to(device=device, dtype=torch.float32)
    mean = x.mean(dim=0)
    centered = x - mean
    covariance = centered.transpose(0, 1).matmul(centered) / float(max(1, x.shape[0] - 1))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues.index_select(0, order).clamp_min(0.0)
    eigenvectors = eigenvectors.index_select(1, order)
    total = eigenvalues.sum().clamp_min(1.0e-12)
    cumulative = torch.cumsum(eigenvalues, dim=0) / total
    reached = torch.nonzero(cumulative >= PCA_VARIANCE_TARGET, as_tuple=False)
    natural_rank = int(reached[0, 0].item() + 1) if reached.numel() else int(eigenvalues.numel())
    rank = min(natural_rank, PCA_RANK_CAP)
    components = eigenvectors[:, :rank].contiguous()
    train_scores = centered.matmul(components).contiguous()
    explained = float(eigenvalues[:rank].sum() / total)
    _require(rank > 0 and math.isfinite(explained), "PCA rank is invalid")
    return {
        "mean": mean.cpu().contiguous(),
        "components": components.cpu().contiguous(),
        "eigenvalues": eigenvalues[:rank].cpu().contiguous(),
        "train_scores": train_scores.cpu().contiguous(),
        "natural_rank_for_target": natural_rank,
        "rank": rank,
        "rank_cap": PCA_RANK_CAP,
        "target_explained_variance": PCA_VARIANCE_TARGET,
        "explained_variance_at_rank": explained,
        "coordinate": "canonical 28x28 image flattened to R^784; TRAIN-only mean-centering; no class labels used",
    }


def _pca_metrics(views: torch.Tensor, labels: torch.Tensor, pca: Mapping[str, Any], train_labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    flat = views[:, 0].reshape(views.shape[0], -1).to(torch.float32).cpu()
    mean = _tensor(pca["mean"], "PCA mean")
    components = _tensor(pca["components"], "PCA components")
    scores = (flat - mean).matmul(components)
    reconstruction = scores.matmul(components.transpose(0, 1)) + mean
    residual = (flat - reconstruction).square().mean(dim=1).sqrt()
    latent = _nearest_distances(scores, _tensor(pca["train_scores"], "PCA train scores"), train_labels, labels, device=device, dimension=int(pca["rank"]))
    _finite(residual, "PCA residual")
    return {"pca_residual": residual, "latent_nearest": latent["nearest_any"], "scores": scores}


def _nearest_distances(query: torch.Tensor, reference: torch.Tensor, reference_labels: torch.Tensor, requested: torch.Tensor, *, device: torch.device, dimension: int) -> dict[str, torch.Tensor]:
    _require(query.ndim == 2 and reference.ndim == 2 and query.shape[1] == reference.shape[1], "nearest-distance geometry changed")
    q = query.to(device=device, dtype=torch.float32)
    ref = reference.to(device=device, dtype=torch.float32)
    ref_labels_device = reference_labels.to(device=device, dtype=torch.long)
    any_dist = torch.empty(q.shape[0], device=device)
    req_dist = torch.empty(q.shape[0], device=device)
    other_dist = torch.empty(q.shape[0], device=device)
    any_index = torch.empty(q.shape[0], dtype=torch.long, device=device)
    req_index = torch.empty(q.shape[0], dtype=torch.long, device=device)
    other_index = torch.empty(q.shape[0], dtype=torch.long, device=device)
    ref_norm = ref.square().sum(dim=1)
    for start in range(0, q.shape[0], DISTANCE_BLOCK):
        end = min(q.shape[0], start + DISTANCE_BLOCK)
        block = q[start:end]
        distances = (block.square().sum(dim=1, keepdim=True) + ref_norm.unsqueeze(0) - 2.0 * block.matmul(ref.transpose(0, 1))).clamp_min(0.0)
        any_values, any_positions = distances.min(dim=1)
        requested_mask = ref_labels_device.unsqueeze(0) == requested[start:end].to(device=device).unsqueeze(1)
        requested_values, requested_positions = distances.masked_fill(~requested_mask, float("inf")).min(dim=1)
        other_values, other_positions = distances.masked_fill(requested_mask, float("inf")).min(dim=1)
        any_dist[start:end] = (any_values / float(dimension)).clamp_min(0.0).sqrt()
        req_dist[start:end] = (requested_values / float(dimension)).clamp_min(0.0).sqrt()
        other_dist[start:end] = (other_values / float(dimension)).clamp_min(0.0).sqrt()
        any_index[start:end] = any_positions
        req_index[start:end] = requested_positions
        other_index[start:end] = other_positions
    return {
        "nearest_any": any_dist.cpu(),
        "nearest_requested": req_dist.cpu(),
        "nearest_other": other_dist.cpu(),
        "requested_minus_other": req_dist.cpu() - other_dist.cpu(),
        "nearest_any_index": any_index.cpu(),
        "nearest_requested_index": req_index.cpu(),
        "nearest_other_index": other_index.cpu(),
        "nearest_any_label": reference_labels.index_select(0, any_index.cpu()),
        "nearest_other_label": reference_labels.index_select(0, other_index.cpu()),
    }


def _attach_manifold_metrics(source: Mapping[str, Any], pca: Mapping[str, Any], train_images: torch.Tensor, train_labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    views = _tensor(source["views"], "source views")
    labels = _tensor(source["labels"], "source labels").to(torch.long)
    train_flat = train_images[:, 0].reshape(train_images.shape[0], -1)
    query_flat = views[:, 0].reshape(views.shape[0], -1)
    distances = _nearest_distances(query_flat, train_flat, train_labels, labels, device=device, dimension=IMAGE_PIXELS)
    pca_values = _pca_metrics(views, labels, pca, train_labels, device)
    return {**distances, **pca_values}


def _summary_map(values: Mapping[str, torch.Tensor], prefix: str) -> dict[str, Any]:
    return {name: _summary(value, f"{prefix} {name}") for name, value in values.items() if isinstance(value, torch.Tensor) and value.ndim == 1 and value.dtype != torch.long}


def _cohen_d(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.to(torch.float64).reshape(-1)
    right = right.to(torch.float64).reshape(-1)
    if left.numel() < 2 or right.numel() < 2:
        return float("nan")
    left_var = left.var(unbiased=True)
    right_var = right.var(unbiased=True)
    pooled = (((left.numel() - 1) * left_var) + ((right.numel() - 1) * right_var)) / max(1, left.numel() + right.numel() - 2)
    denominator = float(torch.sqrt(pooled).item())
    return float((left.mean() - right.mean()).item() / denominator) if denominator > 0 else 0.0


def _distance_comparison(values: torch.Tensor, control: torch.Tensor, control_q95: float, name: str) -> dict[str, Any]:
    return {
        "value": _summary(values, name),
        "control_value": _summary(control, f"{name} canonical DEV"),
        "control_q95": control_q95,
        "above_control_q95_fraction": float((values > control_q95).to(torch.float64).mean()),
        "median_delta_vs_control": float(values.median() - control.median()),
        "cohen_d_vs_control": _cohen_d(values, control),
    }


def _strata(predictions: Mapping[str, torch.Tensor], labels: torch.Tensor) -> dict[str, torch.Tensor]:
    matrix = torch.stack([predictions[name].to(torch.long) for name in EVALUATOR_ORDER], dim=1)
    requested_votes = (matrix == labels.unsqueeze(1)).sum(dim=1)
    mode, mode_count = readiness._mode_with_count(matrix)
    masks = {
        "G1": requested_votes == 4,
        "G2": requested_votes == 3,
        "G3": (mode_count >= 3) & (mode != labels),
        "G4": mode_count < 3,
    }
    combined = torch.zeros(labels.numel(), dtype=torch.bool)
    for mask in masks.values():
        _require(not bool((combined & mask).any()), "consensus strata overlap")
        combined |= mask
    _require(bool(combined.all()), "consensus strata do not cover all rows")
    return {**masks, "requested_votes": requested_votes, "mode": mode, "mode_count": mode_count, "matrix": matrix}


def _strata_report(source: Mapping[str, Any], strata: Mapping[str, torch.Tensor], control: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    labels = _tensor(source["labels"], "stratum labels")
    report: dict[str, Any] = {}
    control_q95 = {name: float(torch.quantile(values.to(torch.float64), torch.tensor(0.95, dtype=torch.float64))) for name, values in control.items()}
    for name in ("G1", "G2", "G3", "G4"):
        mask = strata[name]
        indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
        requested_hist = torch.bincount(labels.index_select(0, indices), minlength=CLASS_COUNT).tolist() if indices.numel() else [0] * CLASS_COUNT
        entry: dict[str, Any] = {"rows": int(indices.numel()), "requested_class_histogram": requested_hist, "indices": indices.tolist()}
        for metric_name, values in control.items():
            selected = _tensor(source[metric_name], f"source {metric_name}").index_select(0, indices) if indices.numel() else torch.empty(0)
            if indices.numel():
                entry[metric_name] = _distance_comparison(selected, values, control_q95[metric_name], f"{name} {metric_name}")
            else:
                entry[metric_name] = {"rows": 0, "control_q95": control_q95[metric_name]}
        report[name] = entry
    return report


def _select_stratum_examples(strata: Mapping[str, torch.Tensor], labels: torch.Tensor, *, per_class: int) -> dict[str, list[int]]:
    selected: dict[str, list[int]] = {}
    for name in ("G1", "G2", "G3", "G4"):
        values: list[int] = []
        for digit in range(CLASS_COUNT):
            candidates = torch.nonzero(strata[name] & (labels == digit), as_tuple=False).reshape(-1)
            values.extend(candidates[:per_class].tolist())
        selected[name] = values
    return selected


def _class_conditioned_report(source: Mapping[str, Any], strata: Mapping[str, torch.Tensor], distances: Mapping[str, torch.Tensor], pca: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    labels = _tensor(source["labels"], "class labels")
    report: dict[str, Any] = {}
    predictions = source["predictions"]
    for digit in range(CLASS_COUNT):
        class_mask = labels == digit
        class_indices = torch.nonzero(class_mask, as_tuple=False).reshape(-1)
        requested_votes = strata["requested_votes"].index_select(0, class_indices)
        wrong_histograms: dict[str, list[int]] = {}
        for evaluator in EVALUATOR_ORDER:
            wrong = predictions[evaluator].index_select(0, class_indices)
            wrong = wrong[wrong != digit]
            wrong_histograms[evaluator] = torch.bincount(wrong, minlength=CLASS_COUNT).tolist()
        combined_wrong = torch.cat([predictions[evaluator].index_select(0, class_indices) for evaluator in EVALUATOR_ORDER])
        combined_wrong = combined_wrong[combined_wrong != digit]
        nearest_wrong = distances["nearest_other_label"].index_select(0, class_indices)
        nearest_wrong_hist = torch.bincount(nearest_wrong, minlength=CLASS_COUNT).tolist()
        report[str(digit)] = {
            "rows": int(class_indices.numel()),
            "four_of_four_requested_fraction": float((requested_votes == 4).to(torch.float64).mean()),
            "at_least_three_requested_fraction": float((requested_votes >= 3).to(torch.float64).mean()),
            "stratum_fraction": {name: float(strata[name].index_select(0, class_indices).to(torch.float64).mean()) for name in ("G1", "G2", "G3", "G4")},
            "nearest_requested_distance": _summary(distances["nearest_requested"].index_select(0, class_indices), f"class {digit} requested distance"),
            "nearest_other_distance": _summary(distances["nearest_other"].index_select(0, class_indices), f"class {digit} other distance"),
            "pca_residual": _summary(pca["pca_residual"].index_select(0, class_indices), f"class {digit} PCA residual"),
            "nearest_wrong_class_histogram": nearest_wrong_hist,
            "nearest_wrong_class": int(torch.argmax(torch.tensor(nearest_wrong_hist)).item()) if nearest_wrong.numel() else None,
            "dominant_evaluator_confusion": {name: (int(torch.argmax(torch.tensor(hist)).item()) if sum(hist) else None) for name, hist in wrong_histograms.items()},
            "all_evaluator_wrong_class": int(torch.argmax(torch.bincount(combined_wrong, minlength=CLASS_COUNT)).item()) if combined_wrong.numel() else None,
            "wrong_histograms_by_evaluator": wrong_histograms,
        }
    return report


def _real_vs_s1(real: Mapping[str, Any], s1: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("nearest_any", "nearest_requested", "nearest_other", "pca_residual", "latent_nearest"):
        left = _tensor(real[name], f"real {name}")
        right = _tensor(s1[name], f"S1 {name}")
        result[name] = {
            "real_ols": _summary(left, f"real OLS {name}"),
            "s1": _summary(right, f"S1 {name}"),
            "S1_minus_real_mean": float(right.to(torch.float64).mean() - left.to(torch.float64).mean()),
            "S1_minus_real_median": float(right.median() - left.median()),
            "cohen_d_S1_vs_real": _cohen_d(right, left),
        }
    result["consensus"] = {"real_ols": real["consensus"], "s1": s1["consensus"]}
    return result


def _wrong_basin_report(source: Mapping[str, Any], strata: Mapping[str, torch.Tensor], distances: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    labels = _tensor(source["labels"], "wrong-basin labels")
    mask = strata["G3"]
    indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
    mode = strata["mode"].index_select(0, indices)
    requested = labels.index_select(0, indices)
    nearest = distances["nearest_any_label"].index_select(0, indices)
    matches = mode == nearest
    rows: list[dict[str, Any]] = []
    for position in indices.tolist():
        rows.append({
            "bank_position": int(position),
            "requested_class": int(labels[position]),
            "consensus_wrong_class": int(strata["mode"][position]),
            "nearest_canonical_class": int(distances["nearest_any_label"][position]),
            "requested_distance": float(distances["nearest_requested"][position]),
            "wrong_basin_distance": float(distances["nearest_any"][position]),
            "other_class_distance": float(distances["nearest_other"][position]),
        })
    return {
        "G3_rows": int(indices.numel()),
        "consensus_wrong_matches_nearest_canonical_fraction": float(matches.to(torch.float64).mean()) if matches.numel() else None,
        "majority_rule": WRONG_BASIN_MAJORITY,
        "supported_by_majority": bool(matches.to(torch.float64).mean() >= WRONG_BASIN_MAJORITY) if matches.numel() else False,
        "requested_class_histogram": torch.bincount(requested, minlength=CLASS_COUNT).tolist(),
        "consensus_wrong_class_histogram": torch.bincount(mode, minlength=CLASS_COUNT).tolist(),
        "examples": rows,
    }


def _interpolation_selection(strata: Mapping[str, torch.Tensor], labels: torch.Tensor) -> list[int]:
    values: list[int] = []
    for name in ("G3", "G4"):
        for digit in range(CLASS_COUNT):
            candidates = torch.nonzero(strata[name] & (labels == digit), as_tuple=False).reshape(-1)
            if candidates.numel():
                values.append(int(candidates[0]))
    return values


def _interpolate(evaluators: Mapping[str, torch.nn.Module], generated: torch.Tensor, requested_refs: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    all_images: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    for index in range(generated.shape[0]):
        for value in INTERPOLATION_LAMBDAS:
            all_images.append(((1.0 - value) * generated[index] + value * requested_refs[index]).clamp(0.0, 1.0))
            all_labels.append(labels[index])
    images = torch.stack(all_images).contiguous()
    interp_labels = torch.stack(all_labels).to(torch.long)
    evaluated = _evaluate_views(images, interp_labels, evaluators, device, label_source="requested class interpolation")
    for lambda_index, value in enumerate(INTERPOLATION_LAMBDAS):
        positions = torch.arange(lambda_index, images.shape[0], len(INTERPOLATION_LAMBDAS), dtype=torch.long)
        predictions = {name: evaluated["predictions"][name].index_select(0, positions) for name in EVALUATOR_ORDER}
        consensus = readiness._consensus_report(predictions, labels)
        confidence = {name: float(evaluated["metrics"][name]["top1_confidence"]["mean"]) for name in EVALUATOR_ORDER}
        rows.append({
            "lambda": value,
            "at_least_3_of_4_agreement_fraction": consensus["at_least_3_of_4_agreement_fraction"],
            "at_least_3_of_4_requested_fraction": consensus["at_least_3_of_4_true_or_requested_fraction"],
            "4_of_4_requested_fraction": consensus["4_of_4_true_or_requested_fraction"],
            "pairwise_mean_agreement": float(np.mean([consensus["pairwise_prediction_agreement_matrix"][i][j] for i in range(4) for j in range(4) if i != j])),
            "confidence_mean_by_evaluator": confidence,
            "predictions": {name: evaluated["predictions"][name].index_select(0, positions).tolist() for name in EVALUATOR_ORDER},
        })
    return {"selected_source_indices": list(range(generated.shape[0])), "lambdas": rows, "images": evaluated["views"], "labels": interp_labels}


def _render_distance_summary(target: Path, sources: Mapping[str, Mapping[str, torch.Tensor]], strata: Mapping[str, torch.Tensor], control: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), squeeze=False)
    panel_ids = list("abcd")
    source_names = ("canonical DEV", "real OLS", "S1")
    source_keys = ("canonical_dev", "real_ols", "s1")
    for axis, metric, title, label in zip(axes.flat, ("nearest_any", "pca_residual", "nearest_any", "pca_residual"), ("Nearest canonical TRAIN distance", "PCA reconstruction residual", "S1 distance by consensus stratum", "S1 PCA residual by consensus stratum"), panel_ids, strict=True):
        if axis is axes.flat[0] or axis is axes.flat[1]:
            values = [sources[key][metric].numpy() for key in source_keys]
            axis.boxplot(values, tick_labels=source_names, showfliers=False)
            axis.set_ylabel("RMS distance" if metric == "nearest_any" else "pixel RMS")
            axis.tick_params(axis="x", labelrotation=18, labelsize=5)
        else:
            names = ("G1", "G2", "G3", "G4")
            values = [_tensor(sources["s1"][metric], f"S1 {metric}")[strata[name]].numpy() for name in names]
            axis.boxplot(values, tick_labels=names, showfliers=False)
            axis.set_ylabel("RMS distance" if metric == "nearest_any" else "pixel RMS")
            axis.axhline(float(torch.quantile(control[metric].to(torch.float64), torch.tensor(0.95, dtype=torch.float64))), color="#B64342", linestyle="--", linewidth=0.8, label="Canonical DEV q95")
            axis.legend(fontsize=5, loc="best")
        axis.set_title(title, fontsize=7)
        _add_panel_label(axis, label)
    fig.text(0.5, 0.005, "Canonical TRAIN support is fit-only; dashed lines are canonical DEV q95 controls where shown.", ha="center", fontsize=5.5)
    return _save_figure(fig, target, "manifold_distance_summary_panel", list(axes.flat), panel_ids, row_groups=(("a", "b"), ("c", "d")), column_groups=(("a", "c"), ("b", "d")))


def _render_stratum_panel(target: Path, group: str, source: Mapping[str, Any], train_images: torch.Tensor, distances: Mapping[str, torch.Tensor], selected: Sequence[int], strata: Mapping[str, torch.Tensor], class_values: Sequence[int]) -> dict[str, Any]:
    digits = list(class_values)
    fig, axes = plt.subplots(3, len(digits), figsize=(6.8, 4.4), squeeze=False)
    panel_ids = [_panel_label(index) for index in range(3 * len(digits))]
    selected_by_class = {int(source["labels"][index]): index for index in selected}
    for column, digit in enumerate(digits):
        index = selected_by_class.get(digit)
        for row, axis in enumerate(axes[:, column]):
            axis.set_xticks([])
            axis.set_yticks([])
            if index is None:
                axis.imshow(torch.zeros((28, 28), dtype=torch.float32), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                axis.set_title("none", fontsize=5)
            else:
                if row == 0:
                    image = source["views"][index, 0]
                    preds = "/".join(str(int(source["predictions"][name][index])) for name in EVALUATOR_ORDER)
                    axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                    axis.set_title(f"y={digit} p={preds}", fontsize=5.2)
                    axis.set_xlabel(f"d_y={float(distances['nearest_requested'][index]):.3f}\nd_any={float(distances['nearest_any'][index]):.3f}", fontsize=5)
                elif row == 1:
                    ref = train_images[int(distances["nearest_requested_index"][index]), 0]
                    axis.imshow(ref, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                    axis.set_title("requested NN", fontsize=5.2)
                else:
                    ref = train_images[int(distances["nearest_any_index"][index]), 0]
                    nearest_class = int(distances["nearest_any_label"][index])
                    axis.imshow(ref, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
                    axis.set_title(f"any NN={nearest_class}", fontsize=5.2)
            if column == 0:
                axis.set_ylabel(("S1" if row == 0 else "req NN" if row == 1 else "any NN"), fontsize=5.2)
            _add_panel_label(axis, panel_ids[row * len(digits) + column])
    fig.text(0.5, 0.005, f"{group}, classes {digits[0]}–{digits[-1]}: deterministic first sample per requested class; predictions S/I/C/D; global [0,1] range; no per-image stretch.", ha="center", fontsize=5.5)
    return _save_figure(
        fig,
        target,
        f"consensus_{group}_panel_classes_{digits[0]}_{digits[-1]}",
        list(axes.flat),
        panel_ids,
        row_groups=tuple(tuple(panel_ids[row * len(digits):(row + 1) * len(digits)]) for row in range(3)),
        column_groups=tuple(tuple(panel_ids[row * len(digits) + column] for row in range(3)) for column in range(len(digits))),
        layout_mode="grid",
    )


def _render_class_average(target: Path, averages: Mapping[str, torch.Tensor], counts: Mapping[str, Sequence[int]], class_values: Sequence[int]) -> dict[str, Any]:
    digits = list(class_values)
    fig, axes = plt.subplots(4, len(digits), figsize=(6.8, 5.2), squeeze=False)
    panel_ids = [_panel_label(index) for index in range(4 * len(digits))]
    row_names = ("canonical DEV mean", "S1 all", "S1 >=3 requested", "S1 disagreement")
    for row, row_name in enumerate(row_names):
        values = averages[row_name]
        for column, digit in enumerate(digits):
            axis = axes[row, column]
            axis.imshow(values[digit, 0], cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.set_title(str(digit), fontsize=5.5)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(f"{row_name}\nn={counts[row_name][digit]}", fontsize=5.2)
            _add_panel_label(axis, panel_ids[row * len(digits) + column])
    fig.text(0.5, 0.005, f"Classes {digits[0]}–{digits[-1]}; class means use the same fixed [0,1] normalization; disagreement = G3 + G4.", ha="center", fontsize=5.5)
    return _save_figure(
        fig,
        target,
        f"class_average_consensus_panel_classes_{digits[0]}_{digits[-1]}",
        list(axes.flat),
        panel_ids,
        row_groups=tuple(tuple(panel_ids[row * len(digits):(row + 1) * len(digits)]) for row in range(4)),
        column_groups=tuple(tuple(panel_ids[row * len(digits) + column] for row in range(4)) for column in range(len(digits))),
        layout_mode="grid",
    )


def _render_interpolation(target: Path, interpolation: Mapping[str, Any]) -> dict[str, Any]:
    values = interpolation["lambdas"]
    x = [float(value["lambda"]) for value in values]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), squeeze=False)
    panel_ids = ["a", "b"]
    axis = axes[0, 0]
    for key, label, color in (("at_least_3_of_4_agreement_fraction", ">=3 evaluator agreement", "#0F4D92"), ("at_least_3_of_4_requested_fraction", ">=3 requested class", "#B64342"), ("4_of_4_requested_fraction", "4/4 requested class", "#42949E")):
        axis.plot(x, [float(value[key]) for value in values], marker="o", linewidth=1.4, markersize=3.5, label=label, color=color)
    axis.set_xlabel("lambda toward requested-class TRAIN NN")
    axis.set_ylabel("fraction")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Consensus trajectory", fontsize=7)
    axis.legend(fontsize=5, loc="best")
    _add_panel_label(axis, "a")
    axis = axes[0, 1]
    for evaluator, color in zip(EVALUATOR_ORDER, ("#0F4D92", "#42949E", "#B64342", "#767676"), strict=True):
        axis.plot(x, [float(value["confidence_mean_by_evaluator"][evaluator]) for value in values], marker="o", linewidth=1.4, markersize=3.5, label=evaluator, color=color)
    axis.set_xlabel("lambda toward requested-class TRAIN NN")
    axis.set_ylabel("top-1 confidence")
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Confidence trajectory", fontsize=7)
    axis.legend(fontsize=5, loc="best")
    _add_panel_label(axis, "b")
    fig.text(0.5, 0.005, "IMAGE_SPACE_DIAGNOSTIC_ONLY; interpolation is not an optical path and lambda is not optimized.", ha="center", fontsize=5.5)
    return _save_figure(fig, target, "image_space_interpolation_panel", list(axes.flat), panel_ids, row_groups=(("a", "b"),))


def _class_average_tensors(dev_views: torch.Tensor, dev_labels: torch.Tensor, s1_views: torch.Tensor, s1_labels: torch.Tensor, strata: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], dict[str, list[int]]]:
    masks = {"canonical DEV mean": torch.ones(dev_labels.numel(), dtype=torch.bool), "S1 all": torch.ones(s1_views.shape[0], dtype=torch.bool), "S1 >=3 requested": strata["G1"] | strata["G2"], "S1 disagreement": strata["G3"] | strata["G4"]}
    inputs = {"canonical DEV mean": (dev_views, dev_labels), "S1 all": (s1_views, s1_labels), "S1 >=3 requested": (s1_views, s1_labels), "S1 disagreement": (s1_views, s1_labels)}
    averages: dict[str, torch.Tensor] = {}
    counts: dict[str, list[int]] = {}
    for name, (images, labels) in inputs.items():
        mask = masks[name]
        rows: list[torch.Tensor] = []
        class_counts: list[int] = []
        for digit in range(CLASS_COUNT):
            class_mask = mask & (labels == digit)
            class_rows = images[class_mask]
            class_counts.append(int(class_rows.shape[0]))
            rows.append(class_rows.to(torch.float32).mean(dim=0) if class_rows.numel() else torch.zeros((1, 28, 28), dtype=torch.float32))
        averages[name] = torch.stack(rows)
        counts[name] = class_counts
    return averages, counts


def _control_distance_arrays(metrics: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: _tensor(metrics[name], f"control {name}").to(torch.float32) for name in ("nearest_any", "nearest_requested", "nearest_other", "pca_residual", "latent_nearest")}


def _classification_flags(s1: Mapping[str, Any], real_ols: Mapping[str, Any], canonical: Mapping[str, Any], strata_report: Mapping[str, Any], interpolation: Mapping[str, Any], wrong_basin: Mapping[str, Any]) -> tuple[str, str, str, dict[str, Any]]:
    disagreement = strata_report["G3"]["rows"] + strata_report["G4"]["rows"]
    disagreement_entry = {
        "nearest_any": strata_report["G3"].get("nearest_any", {}),
        "pca_residual": strata_report["G3"].get("pca_residual", {}),
    }
    combined_indices = strata_report["G3"]["indices"] + strata_report["G4"]["indices"]
    if combined_indices:
        any_values = torch.cat([s1["nearest_any"].index_select(0, torch.tensor(strata_report["G3"]["indices"])), s1["nearest_any"].index_select(0, torch.tensor(strata_report["G4"]["indices"]))])
        pca_values = torch.cat([s1["pca_residual"].index_select(0, torch.tensor(strata_report["G3"]["indices"])), s1["pca_residual"].index_select(0, torch.tensor(strata_report["G4"]["indices"]))])
    else:
        any_values = torch.empty(0)
        pca_values = torch.empty(0)
    canonical_any_q95 = float(canonical["distance_control_q95"]["nearest_any"])
    canonical_pca_q95 = float(canonical["distance_control_q95"]["pca_residual"])
    disagreement_off = bool(any_values.numel() and float(any_values.median()) > canonical_any_q95 and float(pca_values.median()) > canonical_pca_q95)
    canonical_requested_consensus = float(canonical["consensus"]["at_least_3_of_4_true_or_requested_fraction"])
    interp_half = next((value for value in interpolation["lambdas"] if float(value["lambda"]) <= 0.5 and float(value["at_least_3_of_4_requested_fraction"]) >= canonical_requested_consensus), None)
    interpolation_resolves = interp_half is not None
    real_milder = bool(real_ols["nearest_any"].median() <= s1["nearest_any"].index_select(0, torch.tensor(combined_indices)).median() if combined_indices else False)
    real_milder = real_milder and bool(real_ols["pca_residual"].median() <= s1["pca_residual"].index_select(0, torch.tensor(combined_indices)).median() if combined_indices else False)
    off_manifold = disagreement_off and interpolation_resolves and real_milder
    on_manifold = bool(disagreement and not disagreement_off and not interpolation_resolves and float(s1["consensus"]["at_least_3_of_4_true_or_requested_fraction"]) < canonical_requested_consensus)
    s1_rates = [float(strata_report[name]["rows"]) for name in ("G3", "G4")]
    class_specific = False
    if disagreement:
        class_hist = torch.tensor(strata_report["G3"]["requested_class_histogram"], dtype=torch.float64) + torch.tensor(strata_report["G4"]["requested_class_histogram"], dtype=torch.float64)
        class_specific = bool(float(class_hist.max() / class_hist.sum()) >= 0.30)
    transfer_gap = bool(float(s1["nearest_any"].median()) > float(real_ols["nearest_any"].median()) and float(s1["pca_residual"].median()) > float(real_ols["pca_residual"].median()) and float(s1["consensus"]["at_least_3_of_4_true_or_requested_fraction"]) < float(real_ols["consensus"]["at_least_3_of_4_true_or_requested_fraction"]))
    wrong_basin = bool(wrong_basin.get("supported_by_majority", False))
    flags = {"A_off_manifold": off_manifold, "B_on_manifold_boundary": on_manifold, "C_class_specific": class_specific, "D_real_vs_s1_transfer_gap": transfer_gap, "E_wrong_canonical_basin": wrong_basin}
    supported = [name for name, value in flags.items() if value]
    if len(supported) == 1:
        classification, mechanism, successor = {
            "A_off_manifold": (CLASSIFICATION_OFF_MANIFOLD, "Disagreement strata are beyond canonical DEV support and requested-class interpolation resolves consensus under the preregistered control rule.", "DEFINE_EVALUATOR_AGNOSTIC_IMAGE_MANIFOLD_ALIGNMENT_V1"),
            "B_on_manifold_boundary": (CLASSIFICATION_SHARED, "Disagreement remains within canonical image-manifold control distances and interpolation does not resolve the evaluator split.", "DEFINE_NON_CLASSIFIER_STRUCTURAL_SEMANTIC_CONFIRMATION_V1"),
            "C_class_specific": (CLASSIFICATION_CLASS_SPECIFIC, "Divergence is concentrated in requested-class strata or morphology-confusion pairs rather than being uniform across classes.", "ATTRIBUTE_STEP6000_CLASS_CONDITIONED_CANONICAL_MORPHOLOGY_GAP_V1"),
            "D_real_vs_s1_transfer_gap": (CLASSIFICATION_TRANSFER, "Real-OLS outputs are closer to canonical image support and more consensual than S1 outputs under identical diagnostics.", "ATTRIBUTE_STEP6000_REAL_VS_S1_GENERATOR_LOCAL_TRANSFER_GAP_V1"),
            "E_wrong_canonical_basin": (CLASSIFICATION_WRONG_BASIN, "A majority of G3 wrong-consensus samples are closest to canonical images of the same wrong class.", "DEFINE_REQUESTED_CLASS_SEMANTIC_BASIN_ALIGNMENT_V1"),
        }[supported[0]]
    else:
        classification = CLASSIFICATION_MIXED
        mechanism = "The frozen audit separates off-manifold distance, evaluator-boundary, class-specific, real-vs-S1 and wrong-basin effects without a unique single-cause rule."
        successor = "NONE"
    details = {
        "flags": flags,
        "supported_components": supported,
        "disagreement_rows": disagreement,
        "disagreement_off_manifold": disagreement_off,
        "interpolation_resolves_by_lambda_0.5": interpolation_resolves,
        "interpolation_resolution_lambda": None if interp_half is None else interp_half["lambda"],
        "real_path_is_milder": real_milder,
        "canonical_requested_consensus_control": canonical_requested_consensus,
        "wrong_basin_majority_rule": WRONG_BASIN_MAJORITY,
        "class_specific_rule": "top requested-class fraction among G3+G4 >= 0.30",
        "unused_intermediate": disagreement_entry,
        "unused_s1_rates": s1_rates,
    }
    return classification, mechanism, successor, details


def _load_authorities(root: Path) -> dict[str, Any]:
    _require(file_sha256(root / READINESS_REPORT) == READINESS_REPORT_SHA256, "accepted readiness report identity changed")
    _require(file_sha256(root / READINESS_STATUS) == READINESS_STATUS_SHA256, "accepted readiness status identity changed")
    _require(file_sha256(root / READINESS_SELECTION) == READINESS_SELECTION_SHA256, "accepted fixed selection identity changed")
    readiness_report = json.loads((root / READINESS_REPORT).read_text(encoding="utf-8"))
    readiness_status = json.loads((root / READINESS_STATUS).read_text(encoding="utf-8"))
    selection = json.loads((root / READINESS_SELECTION).read_text(encoding="utf-8"))
    _require(readiness_report.get("result") == "VALID_COMPLETED" and readiness_report.get("outer_gate") == "STEP6000_CONDITIONAL_GENERATION_READINESS_RESOLVED", "accepted readiness result/gate changed")
    _require(readiness_report.get("classification") == "STEP6000_SEMANTIC_GAIN_WITH_CROSS_EVALUATOR_DECISION_DIVERGENCE", "accepted readiness classification changed")
    _require(readiness_status.get("result") == "VALID_COMPLETED" and readiness_status.get("outer_gate_passed") is True, "accepted readiness status changed")
    _require(selection.get("selection_frozen_before_forward") is True and selection.get("schema") == "step6000_conditional_generation_readiness_fixed_selection.v1", "fixed selection contract changed")
    base = readiness._load_authorities(root)
    continuation_authorities = base["continuation"]
    replay_authorities = base["replay"]
    checkpoints = base["checkpoints"]
    step3000 = checkpoints[3000]
    step6000 = checkpoints[6000]
    _require(readiness_report["checkpoints"]["3000"]["sha256"] == STEP3000_SHA256, "step3000 authority changed")
    _require(readiness_report["checkpoints"]["6000"]["sha256"] == STEP6000_SHA256, "step6000 authority changed")
    requested = _tensor(replay_authorities["requested"], "S1 requested labels").to(torch.long).cpu()
    s1_z = _tensor(replay_authorities["candidate_z"]["S1"], "S1 bank").cpu().contiguous()
    real_ols_z = _tensor(replay_authorities["sampler_inputs"]["dev_z"], "real DEV OLS").cpu().contiguous()
    labels = _tensor(continuation_authorities["labels"], "canonical labels").to(torch.long).cpu()
    images = _tensor(continuation_authorities["images"], "canonical images").cpu().contiguous()
    train_indices = _tensor(replay_authorities["sampler_inputs"]["train_indices"], "TRAIN indices").to(torch.long).cpu()
    dev_indices = _tensor(replay_authorities["sampler_inputs"]["dev_indices"], "DEV indices").to(torch.long).cpu()
    train_images = images.index_select(0, train_indices)
    train_labels = labels.index_select(0, train_indices)
    dev_images = images.index_select(0, dev_indices)
    dev_labels = labels.index_select(0, dev_indices)
    _validate_image_tensor(images, "canonical images")
    _require(s1_z.shape == (2000, 100) and s1_z.dtype == torch.complex64, "S1 bank shape/dtype changed")
    _require(real_ols_z.shape == (5000, 100) and real_ols_z.dtype == torch.complex64, "real DEV OLS shape/dtype changed")
    _require(torch.equal(requested, torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(200)), "S1 requested balance changed")
    dev_counts = torch.bincount(dev_labels, minlength=CLASS_COUNT)
    _require(torch.equal(dev_counts, torch.full((CLASS_COUNT,), 500, dtype=torch.long)), "DEV class balance changed")
    sampled_positions = _tensor_or_list(selection["sampled"]["positions_in_bank"], "accepted sampled positions", dtype=torch.long)
    panel_positions = _tensor_or_list(selection["accepted_R_panel"]["sampled_bank_positions"], "accepted panel positions", dtype=torch.long)
    _require(sampled_positions.numel() == 100 and panel_positions.numel() == 200, "fixed visualization position counts changed")
    _require(bool(torch.all((sampled_positions >= 0) & (sampled_positions < s1_z.shape[0]))) and bool(torch.all((panel_positions >= 0) & (panel_positions < s1_z.shape[0]))), "fixed visualization positions out of bank")
    return {
        "readiness_report": readiness_report,
        "readiness_status": readiness_status,
        "selection": selection,
        "base": base,
        "continuation": continuation_authorities,
        "replay": replay_authorities,
        "step3000": step3000,
        "step6000": step6000,
        "s1_z": s1_z,
        "requested": requested,
        "real_ols_z": real_ols_z,
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "train_images": train_images,
        "train_labels": train_labels,
        "dev_indices": dev_indices,
        "dev_images": dev_images,
        "dev_labels": dev_labels,
        "sampled_positions": sampled_positions,
    }


def _run_experiment(root: Path, output_dir: Path) -> dict[str, Any]:
    target = (root / output_dir if not output_dir.is_absolute() else output_dir).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "protected output namespace is forbidden")
    _require(not target.exists(), f"write-once output already exists: {target}")
    authorities = _load_authorities(root)
    readiness.continuation.robust._prepare_cuda_runtime()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "frozen localization requires CUDA")
    device = torch.device("cuda:0")
    consumer, evaluators = readiness._load_models(root, authorities["base"], device)
    checkpoint = authorities["step6000"]
    preflight = _no_update_preflight(consumer, evaluators, checkpoint, authorities["dev_images"], authorities["s1_z"], device)
    pca = _fit_pca(authorities["train_images"][:, 0].reshape(authorities["train_images"].shape[0], -1), device)
    consumer.load_trainable_state_dict({"encoder.phase": checkpoint["encoder_state"], "generator.phase": checkpoint["generator_state"]})
    consumer.runtime.eval()
    canonical = _evaluate_views(authorities["dev_images"], authorities["dev_labels"], evaluators, device, label_source="true class")
    real_ols = _forward_generated(consumer, evaluators, authorities["real_ols_z"], authorities["dev_labels"], device, source="real_ols_step6000")
    s1 = _forward_generated(consumer, evaluators, authorities["s1_z"], authorities["requested"], device, source="s1_step6000")
    for source in (canonical, real_ols, s1):
        source.update(_attach_manifold_metrics(source, pca, authorities["train_images"], authorities["train_labels"], device))
    s1_strata = _strata(s1["predictions"], authorities["requested"])
    s1_strata["requested_labels"] = authorities["requested"]
    control_arrays = _control_distance_arrays(canonical)
    distance_control_q95 = {name: float(torch.quantile(value.to(torch.float64), torch.tensor(0.95, dtype=torch.float64))) for name, value in control_arrays.items()}
    strata_report = _strata_report(s1, s1_strata, control_arrays)
    selected_strata = _select_stratum_examples(s1_strata, authorities["requested"], per_class=STRATUM_PANEL_PER_CLASS)
    interpolation_indices = _interpolation_selection(s1_strata, authorities["requested"])
    interp_generated = s1["views"].index_select(0, torch.tensor(interpolation_indices, dtype=torch.long))
    interp_requested_refs = authorities["train_images"].index_select(0, s1["nearest_requested_index"].index_select(0, torch.tensor(interpolation_indices, dtype=torch.long)))
    interp_labels = authorities["requested"].index_select(0, torch.tensor(interpolation_indices, dtype=torch.long))
    interpolation = _interpolate(evaluators, interp_generated, interp_requested_refs, interp_labels, device)
    interpolation_json = {key: value for key, value in interpolation.items() if key not in {"images", "labels"}}
    wrong_basin = _wrong_basin_report(s1, s1_strata, s1)
    class_report = _class_conditioned_report(s1, s1_strata, s1, s1)
    real_vs_s1 = _real_vs_s1(real_ols, s1)
    canonical_consensus = canonical["consensus"]
    real_consensus = real_ols["consensus"]
    s1_consensus = s1["consensus"]
    canonical_summary = {"metrics": canonical["metrics"], "consensus": canonical_consensus, "raw_validity": canonical["raw_validity"]}
    real_summary = {"metrics": real_ols["metrics"], "consensus": real_consensus, "raw_validity": real_ols["raw_validity"]}
    s1_summary = {"metrics": s1["metrics"], "consensus": s1_consensus, "raw_validity": s1["raw_validity"]}
    canonical["distance_control_q95"] = distance_control_q95
    classification, mechanism, successor, classification_details = _classification_flags(s1, real_ols, canonical, strata_report, interpolation, wrong_basin)
    target.mkdir(parents=True, exist_ok=False)
    _write_json(target / "preflight.json", preflight, kind="image-manifold localization preflight")
    _write_torch(target / "pca_reference.pt", {"schema": "canonical_train_image_pca_reference.v1", "mean": pca["mean"], "components": pca["components"], "eigenvalues": pca["eigenvalues"], "train_scores": pca["train_scores"], "train_labels": authorities["train_labels"], "rank": pca["rank"], "natural_rank_for_target": pca["natural_rank_for_target"], "explained_variance_at_rank": pca["explained_variance_at_rank"]}, kind="TRAIN-only unsupervised image-manifold PCA reference")
    _write_torch(target / "analysis_payload.pt", {"schema": "step6000_image_manifold_localization_payload.v1", "canonical_dev_views": canonical["views"], "real_ols_views": real_ols["views"], "s1_views": s1["views"], "requested_labels": authorities["requested"], "dev_labels": authorities["dev_labels"], "s1_predictions": s1["predictions"], "s1_strata": {key: value for key, value in s1_strata.items() if isinstance(value, torch.Tensor)}, "s1_distances": {key: value for key, value in s1.items() if isinstance(value, torch.Tensor)}, "real_ols_distances": {key: value for key, value in real_ols.items() if isinstance(value, torch.Tensor)}, "canonical_distances": {key: value for key, value in canonical.items() if isinstance(value, torch.Tensor)}}, kind="frozen image-manifold localization tensors")
    _write_json(target / "source_manifest.json", {"schema": "step6000_image_manifold_localization_source_manifest.v1", "task_id": TASK_ID, "authorization": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "execution_source": _relative(root, Path(__file__)), "execution_source_sha256": file_sha256(Path(__file__)), "protected_test_sealed_zero": True}, kind="image-manifold localization source manifest")
    _write_json(target / "execution_manifest.json", {"schema": "step6000_image_manifold_localization_execution_manifest.v1", "task_id": TASK_ID, "authorization": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True}, "outer_gate": OUTER_GATE, "checkpoint_step": 6000, "no_update": True, "optimizer_created": False, "parameter_updates": 0, "sampler_refit": False, "new_evaluator": False, "E_E_created": False, "protected_test_sealed_zero": True, "pca_train_only": True, "interpolation_image_space_only": True, "cuda_runtime": readiness.continuation.robust._prepare_cuda_runtime(), "fixed_selection_authority": _artifact_ref(root, root / READINESS_SELECTION, role="accepted fixed visualization indices")}, kind="image-manifold localization execution manifest")
    _write_json(target / "canonical_real_consensus.json", canonical_summary, kind="canonical real consensus control")
    _write_json(target / "real_ols_step6000_consensus.json", real_summary, kind="real OLS step6000 consensus control")
    _write_json(target / "s1_consensus_strata.json", {"schema": "s1_consensus_strata.v1", "groups": strata_report, "selected_examples": selected_strata, "requested_class_histogram": torch.bincount(authorities["requested"], minlength=CLASS_COUNT).tolist()}, kind="S1 consensus strata")
    _write_json(target / "image_manifold_metrics.json", {"schema": "image_manifold_metrics.v1", "pca": {key: value for key, value in pca.items() if not isinstance(value, torch.Tensor)}, "canonical_dev": {key: _summary(value, f"canonical {key}") for key, value in canonical.items() if isinstance(value, torch.Tensor) and value.ndim == 1}, "real_ols_step6000": {key: _summary(value, f"real OLS {key}") for key, value in real_ols.items() if isinstance(value, torch.Tensor) and value.ndim == 1}, "s1_step6000": {key: _summary(value, f"S1 {key}") for key, value in s1.items() if isinstance(value, torch.Tensor) and value.ndim == 1}, "distance_control_q95": distance_control_q95, "consensus_vs_manifold": strata_report, "real_vs_s1": real_vs_s1}, kind="canonical image-manifold distances and PCA residuals")
    _write_json(target / "class_conditioned_divergence.json", {"schema": "class_conditioned_image_manifold_divergence.v1", "classes": class_report}, kind="class-conditioned divergence")
    _write_json(target / "wrong_basin.json", {"schema": "wrong_class_canonical_basin_diagnostic.v1", **wrong_basin}, kind="wrong canonical basin diagnostic")
    _write_json(target / "interpolation.json", {"schema": "image_space_interpolation_diagnostic.v1", "diagnostic_only": True, **interpolation_json}, kind="image-space interpolation diagnostic")
    _write_torch(target / "interpolation_views.pt", {"schema": "image_space_interpolation_views.v1", "images": interpolation["images"], "labels": interpolation["labels"]}, kind="image-space interpolation diagnostic views")
    averages, average_counts = _class_average_tensors(authorities["dev_images"], authorities["dev_labels"], s1["views"], authorities["requested"], s1_strata)
    _write_torch(target / "class_average_tensors.pt", {"schema": "class_average_consensus_tensors.v1", **averages}, kind="class-conditioned image average panels")
    figure_artifacts: dict[str, Any] = {}
    figure_artifacts["manifold_distance_summary_panel"] = _render_distance_summary(target, {"canonical_dev": canonical, "real_ols": real_ols, "s1": s1}, s1_strata, control_arrays)
    for group in ("G1", "G2", "G3", "G4"):
        for class_values in (tuple(range(0, 5)), tuple(range(5, 10))):
            figure_key = f"consensus_{group}_panel_classes_{class_values[0]}_{class_values[-1]}"
            figure_artifacts[figure_key] = _render_stratum_panel(target, group, s1, authorities["train_images"], s1, selected_strata[group], s1_strata, class_values)
    for class_values in (tuple(range(0, 5)), tuple(range(5, 10))):
        figure_key = f"class_average_consensus_panel_classes_{class_values[0]}_{class_values[-1]}"
        figure_artifacts[figure_key] = _render_class_average(target, averages, average_counts, class_values)
    figure_artifacts["image_space_interpolation_panel"] = _render_interpolation(target, interpolation)
    predictions_payload = {"schema": "step6000_image_manifold_localization_predictions.v1", "canonical_dev": {name: value.cpu() for name, value in canonical["predictions"].items()}, "real_ols_step6000": {name: value.cpu() for name, value in real_ols["predictions"].items()}, "s1_step6000": {name: value.cpu() for name, value in s1["predictions"].items()}, "requested_labels": authorities["requested"], "dev_labels": authorities["dev_labels"]}
    _write_torch(target / "predictions.pt", predictions_payload, kind="frozen evaluator predictions")
    status = {"schema": STATUS_SCHEMA, "task_id": TASK_ID, "result": "VALID_COMPLETED", "outer_gate": OUTER_GATE, "outer_gate_passed": True, "classification": classification, "recommended_single_successor": successor, "next_action": "NONE", "no_training": True, "new_evaluator": False, "E_E_created": False, "protected_test_read": False, "sealed_read": False}
    _write_json(target / "status.json", status, kind="image-manifold localization status")
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "mode": "IMPLEMENT_PREPARED",
        "risk_level": "R2_ARTIFACT_PRODUCING_RESEARCH",
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "local_state": readiness.git_identity(root),
        "input_authorities": {"accepted_readiness_report": _artifact_ref(root, root / READINESS_REPORT, role="accepted step6000 readiness result"), "accepted_readiness_status": _artifact_ref(root, root / READINESS_STATUS, role="accepted readiness status"), "accepted_fixed_selection": _artifact_ref(root, root / READINESS_SELECTION, role="accepted fixed visualization indices"), "step3000_checkpoint": {"path": "outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01/scientific_step_003000.pt", "sha256": STEP3000_SHA256, "role": "historical step3000 Generator"}, "step6000_checkpoint": {"path": "outputs/20260920_three_evaluator_robust_semantic_continuation_v1/runs/attempt_02/scientific_step_006000.pt", "sha256": STEP6000_SHA256, "role": "frozen step6000 Generator"}, "canonical_train_images": _artifact_ref(root, root / readiness.continuation.robust.TRAINING_SOURCE, role="canonical TRAIN image/label source"), "canonical_evaluators": {name: _artifact_ref(root, root / path, role=f"frozen {name}") for name, path in (("E_S", readiness.continuation.robust.SMOOTH_CHECKPOINT), ("E_I", readiness.continuation.robust.INDEPENDENT_CHECKPOINT), ("E_C", readiness.continuation.robust.EC_CHECKPOINT), ("E_D", readiness.continuation.ED_CHECKPOINT))}},
        "preflight": preflight,
        "canonical_real_consensus_control": canonical_summary,
        "real_ols_step6000_consensus": real_summary,
        "s1_consensus_strata": {"groups": strata_report, "selected_examples": selected_strata},
        "canonical_manifold_distances": {"control": {name: _summary(value, f"canonical DEV {name}") for name, value in control_arrays.items()}, "q95": distance_control_q95, "pca": {key: value for key, value in pca.items() if not isinstance(value, torch.Tensor)}},
        "pca_manifold_residual": {"canonical_dev": _summary(canonical["pca_residual"], "canonical DEV PCA residual"), "real_ols_step6000": _summary(real_ols["pca_residual"], "real OLS PCA residual"), "s1_step6000": _summary(s1["pca_residual"], "S1 PCA residual")},
        "consensus_vs_manifold_analysis": strata_report,
        "class_conditioned_divergence": class_report,
        "real_vs_s1_image_manifold_gap": real_vs_s1,
        "wrong_basin_analysis": wrong_basin,
        "image_space_interpolation_diagnostic": interpolation_json,
        "fixed_visual_panels": figure_artifacts,
        "classification": classification,
        "mechanism": mechanism,
        "recommended_single_successor": successor,
        "classification_details": classification_details,
        "claim_boundary": {"final_human_perceptual_quality": False, "hardware_readiness": False, "formal_promotion": False, "training_executed": False, "new_evaluator_created": False, "E_E_created": False, "sampler_modified": False, "protected_test_sealed": False, "interpolation_physical_optical_path": False},
        "files_changed": {"repository_source": [_relative(root, Path(__file__))], "scientific_artifacts": sorted(path.name for path in target.iterdir())},
        "validation": {"cuda": True, "focused_tests": "tests/v2/test_lightgen_2d_localize_step6000_image_manifold_semantic_divergence_v1.py", "figure_backend": "Python", "strict_panel_alignment": True, "pdf_text_and_collision_audit_required": True},
        "unresolved": ["PCA and nearest-neighbor distances are evaluator-independent diagnostics, not a final perceptual-quality or hardware claim.", "E_S/E_I/E_C/E_D remain frozen development evaluators; no E_E endpoint was created."],
        "next_action": "NONE",
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report, kind="step6000 image-manifold localization report")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--authorization-token", required=False)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "no_update": True, "output_dir": args.output_dir}, ensure_ascii=False, indent=2))
        return 0
    if args.authorization_token != AUTHORIZATION_TOKEN:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": "explicit authorization token is required"}, ensure_ascii=False, indent=2))
        return 2
    try:
        report = _run_experiment(_root(), Path(args.output_dir))
    except (LocalizationError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "recommended_single_successor": report["recommended_single_successor"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
