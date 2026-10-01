"""Localize the fixed sampled OLS semantic-manifold gap.

The consumer reads the accepted real TRAIN/DEV OLS, the fixed Q2 bank, the
accepted OLS probe, and the accepted step-0/300/3000 generated CMOS artifacts.
It performs no fitting, training, backward pass, optimizer construction, or
latent resampling.  The only model forwards are frozen evaluator inference and
the preregistered nearest-real-neighbor visual bridge.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import torch
import torch.nn.functional as F

from ldm_project.v2.evidence import file_sha256, safe_torch_load, tensor_sha256
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_phase_completion_repair_v1 as phase_repair
from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as robust
from scripts.research.lightgen import execute_lightgen_2d_ols_to_cmos_semantic_information_localization_v1 as localization
from scripts.research.lightgen import execute_lightgen_2d_post3000_generation_readiness_scaling_v1 as readiness


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42


TASK_ID = "EXECUTE_REAL_VS_SAMPLED_OLS_SEMANTIC_MANIFOLD_GAP_V1"
AUTHORIZATION_TOKEN = "AUTHORIZED_EXECUTE_REAL_VS_SAMPLED_OLS_SEMANTIC_MANIFOLD_GAP_V1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
OUTER_GATE = "REAL_VS_SAMPLED_OLS_SEMANTIC_MANIFOLD_GAP_LOCALIZED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
EXPECTED_BRANCH = readiness.EXPECTED_BRANCH
EXPECTED_HEAD = readiness.EXPECTED_HEAD

PROBE_RUN = Path("outputs/20260919_ols_to_cmos_semantic_information_localization_v1/runs/attempt_01")
PROBE_WEIGHTS = PROBE_RUN / "probe_weights.pt"
PROBE_WEIGHTS_SHA256 = "b76e1d22420bd48e620a3d293ca15e8d6e04ab192df5cc5bee8c1cdb5310b383"
POST3000_REPORT = Path("outputs/20260920_post3000_generation_readiness_scaling_v1/runs/attempt_05/report.json")
POST3000_RUN = POST3000_REPORT.parent
STEP_0 = readiness.STEP_0
STEP_300 = readiness.STEP_300
STEP_3000 = readiness.ROBUST_STEP_3000
CHECKPOINTS = {0: STEP_0, 300: STEP_300, 3000: STEP_3000}

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260920_real_vs_sampled_ols_semantic_manifold_gap_v1/runs/attempt_01"
)
SAMPLE_COUNT = 200
DISTANCE_QUERY_CHUNK = 128
EVALUATOR_BATCH_SIZE = 50
VISUAL_SAMPLE_INDICES = (0, 1, 2, 3, 4)
CLASS_COUNT = 10
EPS = 1.0e-12

CLASSIFICATION_A = "SAMPLED_OLS_SEMANTIC_STRUCTURE_NOT_PRESERVED"
CLASSIFICATION_B = "SAMPLED_OLS_CLASS_CODE_PRESENT_BUT_LOCAL_MANIFOLD_SUPPORT_MISMATCHED"
CLASSIFICATION_C = "SAMPLED_OLS_SEMANTIC_CODE_PRESENT_BUT_GENERATOR_TRANSFER_FAILS"
CLASSIFICATION_D = "SAMPLED_OLS_CLASS_CONDITIONAL_AMPLITUDE_GAP_SUPPORTED"
CLASSIFICATION_E = "SAMPLED_OLS_CLASS_CONDITIONAL_PHASE_GAP_SUPPORTED"
CLASSIFICATION_F = "REAL_VS_SAMPLED_OLS_SEMANTIC_MANIFOLD_GAP_MIXED"


class ManifoldGapError(RuntimeError):
    """Raised when a frozen input or diagnostic boundary cannot be proven."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifoldGapError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, role: str) -> dict[str, Any]:
    target = path.resolve()
    _require(target.is_file(), f"{role} is missing: {target}")
    return {"path": _relative(root, target), "sha256": file_sha256(target), "role": role}


def _write_json(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    robust._write_json(path, payload, kind=kind)


def _write_torch(path: Path, payload: Mapping[str, Any], *, kind: str) -> None:
    robust._write_torch(path, payload, kind=kind)


def _float_list(values: torch.Tensor) -> list[float]:
    return [float(value) for value in values.detach().cpu().reshape(-1)]


def _summary(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    _finite(tensor, "summary input")
    _require(tensor.numel() > 0, "summary input is empty")
    quantiles = torch.quantile(tensor, torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64))
    return {
        "count": int(tensor.numel()),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std_population": float(tensor.std(unbiased=False)),
        "q50": float(quantiles[0]),
        "q90": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "q99": float(quantiles[3]),
    }


def _entropy(histogram: torch.Tensor) -> tuple[float, float]:
    values = histogram.to(torch.float64)
    probabilities = values / values.sum()
    nonzero = probabilities > 0.0
    entropy = float(-(probabilities[nonzero] * probabilities[nonzero].log()).sum())
    return entropy, entropy / math.log(float(CLASS_COUNT))


def _confusion(labels: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
    labels = labels.to(device="cpu", dtype=torch.long).reshape(-1)
    predictions = predictions.to(device="cpu", dtype=torch.long).reshape(-1)
    _require(labels.shape == predictions.shape and labels.numel() > 0, "confusion inputs are incompatible")
    result = torch.zeros((CLASS_COUNT, CLASS_COUNT), dtype=torch.long)
    for actual, predicted in zip(labels.tolist(), predictions.tolist(), strict=True):
        _require(0 <= actual < CLASS_COUNT and 0 <= predicted < CLASS_COUNT, "class index outside 0..9")
        result[actual, predicted] += 1
    return result


def _class_metrics(labels: torch.Tensor, predictions: torch.Tensor) -> dict[str, Any]:
    matrix = _confusion(labels, predictions)
    row_totals = matrix.sum(dim=1)
    recalls = matrix.diag().to(torch.float64) / row_totals.to(torch.float64)
    return {
        "rows": int(labels.numel()),
        "accuracy": float((labels == predictions).to(torch.float64).mean()),
        "macro_recall": float(recalls.mean()),
        "per_class_recall": _float_list(recalls),
        "confusion_matrix_actual_rows_predicted_columns": matrix.tolist(),
    }


def _ols_features(z: torch.Tensor) -> torch.Tensor:
    _require(z.ndim == 2 and z.shape[1] == 100 and torch.is_complex(z), "OLS must be complex [N,100]")
    value = torch.cat((z.real, z.imag), dim=1).to(torch.float64).contiguous()
    _finite(value, "OLS Cartesian features")
    return value


def _standardize_train_only(train: torch.Tensor, *queries: torch.Tensor) -> tuple[torch.Tensor, ...]:
    _require(train.ndim == 2, "TRAIN representation must be rank 2")
    mean = train.mean(dim=0)
    raw_std = train.std(dim=0, unbiased=False)
    safe_std = torch.where(raw_std > 0.0, raw_std, torch.ones_like(raw_std))
    result = tuple(((value.to(torch.float64) - mean) / safe_std).contiguous() for value in (train, *queries))
    for index, value in enumerate(result):
        _finite(value, f"standardized representation {index}")
    return (*result, mean, safe_std)


def _load_probe(root: Path) -> dict[str, Any]:
    path = root / PROBE_WEIGHTS
    _require(file_sha256(path) == PROBE_WEIGHTS_SHA256, "accepted OLS probe identity changed")
    payload = _mapping(safe_torch_load(path, expected_sha256=PROBE_WEIGHTS_SHA256, map_location="cpu"), "accepted OLS probe")
    _require(payload.get("schema") == "ols_to_cmos_semantic_localization_probes.v1", "OLS probe schema changed")
    _require(payload.get("task_id") == localization.TASK_ID, "OLS probe task identity changed")
    primary = _mapping(payload.get("primary_features"), "OLS probe feature contract")
    _require(primary.get("ols") == "concat(real(z), imag(z)) in R^200", "OLS probe representation changed")
    model = _mapping(payload.get("ols"), "OLS probe model")
    weights = _tensor(model.get("weights"), "OLS probe weights")
    bias = _tensor(model.get("bias"), "OLS probe bias")
    _require(weights.shape == (10, 200) and weights.dtype == torch.float64, "OLS probe weights shape/dtype changed")
    _require(bias.shape == (10,) and bias.dtype == torch.float64, "OLS probe bias shape/dtype changed")
    _finite(weights, "OLS probe weights")
    _finite(bias, "OLS probe bias")
    return {"path": path, "payload": payload, "weights": weights, "bias": bias}


def _probe_logits(features: torch.Tensor, mean: torch.Tensor, std: torch.Tensor, probe: Mapping[str, Any]) -> torch.Tensor:
    standardized = (features.to(torch.float64) - mean) / std
    weights = _tensor(probe["weights"], "probe weights")
    bias = _tensor(probe["bias"], "probe bias")
    logits = standardized @ weights.T + bias
    _finite(logits, "OLS probe logits")
    return logits


def _probe_distribution(logits: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, Any]:
    logits = logits.detach().to(device="cpu", dtype=torch.float64)
    log_probabilities = torch.log_softmax(logits, dim=1)
    probabilities = log_probabilities.exp()
    predictions = probabilities.argmax(dim=1)
    histogram = torch.bincount(predictions, minlength=CLASS_COUNT)
    entropy, normalized_entropy = _entropy(histogram)
    ordered = torch.sort(probabilities, dim=1, descending=True).values
    row_entropy = -(probabilities * log_probabilities).sum(dim=1)
    result: dict[str, Any] = {
        "labels_consumed": labels is not None,
        "rows": int(predictions.numel()),
        "predicted_class_histogram": [int(value) for value in histogram],
        "predicted_class_proportions": _float_list(histogram.to(torch.float64) / float(predictions.numel())),
        "predicted_class_entropy_nats": entropy,
        "predicted_class_entropy_normalized": normalized_entropy,
        "predicted_class_dominant_fraction": float(histogram.max().to(torch.float64) / float(predictions.numel())),
        "represented_class_count": int((histogram > 0).sum()),
        "represented_classes": [int(value) for value in torch.nonzero(histogram > 0, as_tuple=False).flatten()],
        "top1_confidence": _summary(ordered[:, 0]),
        "top1_top2_logit_margin": _summary(logits.topk(2, dim=1).values[:, 0] - logits.topk(2, dim=1).values[:, 1]),
        "prediction_entropy_nats": _summary(row_entropy),
        "prediction_entropy_normalized": float(row_entropy.mean() / math.log(float(CLASS_COUNT))),
        "predictions": predictions,
        "logits": logits,
    }
    if labels is not None:
        result.update(_class_metrics(labels, predictions))
    return result


def _nearest_rows(query: torch.Tensor, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    query = query.to(device="cpu", dtype=torch.float64).contiguous()
    reference = reference.to(device="cpu", dtype=torch.float64).contiguous()
    _require(query.ndim == reference.ndim == 2 and query.shape[1] == reference.shape[1], "nearest-row dimensions differ")
    distances: list[torch.Tensor] = []
    indices: list[torch.Tensor] = []
    for start in range(0, query.shape[0], DISTANCE_QUERY_CHUNK):
        stop = min(start + DISTANCE_QUERY_CHUNK, query.shape[0])
        block = torch.cdist(query[start:stop], reference, p=2)
        values, locations = block.min(dim=1)
        distances.append(values)
        indices.append(locations)
    result_distance = torch.cat(distances)
    result_index = torch.cat(indices).to(torch.long)
    _finite(result_distance, "nearest-row distances")
    return result_distance, result_index


def _nearest_within_class(
    query: torch.Tensor,
    reference: torch.Tensor,
    reference_labels: torch.Tensor,
    query_classes: torch.Tensor,
) -> torch.Tensor:
    result = torch.empty((query.shape[0],), dtype=torch.float64)
    for digit in range(CLASS_COUNT):
        query_indices = torch.nonzero(query_classes == digit, as_tuple=False).flatten()
        reference_indices = torch.nonzero(reference_labels == digit, as_tuple=False).flatten()
        _require(query_indices.numel() > 0 and reference_indices.numel() > 0, f"class {digit} missing from nearest-class comparison")
        distances, _ = _nearest_rows(query.index_select(0, query_indices), reference.index_select(0, reference_indices))
        result.index_copy_(0, query_indices, distances)
    _finite(result, "nearest-within-class distances")
    return result


def _support_report(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    dev_features: torch.Tensor,
    dev_labels: torch.Tensor,
    sample_features: torch.Tensor,
    sample_predictions: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    dev_any, _ = _nearest_rows(dev_features, train_features)
    dev_same = _nearest_within_class(dev_features, train_features, train_labels, dev_labels)
    sample_any, sample_nn_indices = _nearest_rows(sample_features, train_features)
    sample_pred = _nearest_within_class(sample_features, train_features, train_labels, sample_predictions)
    dev_any_q95 = float(torch.quantile(dev_any, torch.tensor(0.95, dtype=torch.float64)))
    dev_same_q95 = float(torch.quantile(dev_same, torch.tensor(0.95, dtype=torch.float64)))
    report = {
        "representation": "x=[Re(z),Im(z)] in R^200; TRAIN-only mean/std standardization",
        "real_dev_control": {
            "d_real_any": _summary(dev_any),
            "d_real_same": _summary(dev_same),
            "q95_thresholds": {"d_real_any": dev_any_q95, "d_real_same": dev_same_q95},
        },
        "sampled": {
            "d_sample_any": _summary(sample_any),
            "d_sample_predclass": _summary(sample_pred),
            "sample_support_fraction_q95": {
                "any": float((sample_any <= dev_any_q95).to(torch.float64).mean()),
                "predclass": float((sample_pred <= dev_same_q95).to(torch.float64).mean()),
                "threshold_source": "corresponding real-DEV q95; descriptive only, not a universal accept/reject rule",
            },
            "predicted_class_is_probe_output_only": True,
        },
        "identity_exclusion": {
            "train_dev_source_indices_disjoint": True,
            "sampled_rows_have_no_real_source_identity": True,
            "exact_identity_exclusion_applied": False,
        },
    }
    return report, sample_any, sample_nn_indices


def _logit_geometry(
    train_logits: torch.Tensor,
    train_labels: torch.Tensor,
    dev_logits: torch.Tensor,
    dev_labels: torch.Tensor,
    sample_logits: torch.Tensor,
    sample_predictions: torch.Tensor,
) -> dict[str, Any]:
    centroids = torch.stack([train_logits[train_labels == digit].mean(dim=0) for digit in range(CLASS_COUNT)])
    dev_distances = torch.cdist(dev_logits, centroids, p=2)
    dev_nearest = dev_distances.argmin(dim=1)
    dev_competing = dev_distances.clone()
    dev_competing[torch.arange(dev_logits.shape[0]), dev_labels] = float("inf")
    sample_distances = torch.cdist(sample_logits, centroids, p=2)
    sample_order = torch.sort(sample_distances, dim=1).values
    sample_nearest = sample_distances.argmin(dim=1)
    sample_pred_distance = sample_distances.gather(1, sample_predictions[:, None]).squeeze(1)
    return {
        "coordinate_system": "frozen OLS probe logits; class centroids fit on real TRAIN only",
        "real_dev": {
            "distance_to_true_class_centroid": _summary(dev_distances.gather(1, dev_labels[:, None]).squeeze(1)),
            "distance_to_nearest_competing_centroid": _summary(dev_competing.min(dim=1).values),
            "nearest_centroid_true_class_fraction": float((dev_nearest == dev_labels).to(torch.float64).mean()),
            "probe_prediction_nearest_centroid_agreement": float((dev_nearest == dev_logits.argmax(dim=1)).to(torch.float64).mean()),
            "class_histogram_nearest_centroid": [int(value) for value in torch.bincount(dev_nearest, minlength=CLASS_COUNT)],
        },
        "sampled": {
            "predicted_class": [int(value) for value in sample_predictions],
            "distance_to_predicted_class_centroid": _summary(sample_pred_distance),
            "distance_to_nearest_real_class_centroid": _summary(sample_distances.min(dim=1).values),
            "nearest_centroid_class": [int(value) for value in sample_nearest],
            "nearest_vs_second_nearest_centroid_margin": _summary(sample_order[:, 1] - sample_order[:, 0]),
            "predicted_class_nearest_centroid_agreement": float((sample_predictions == sample_nearest).to(torch.float64).mean()),
            "class_histogram_nearest_centroid": [int(value) for value in torch.bincount(sample_nearest, minlength=CLASS_COUNT)],
            "labels_consumed": False,
        },
        "class_centroids": centroids,
    }


def _component_features(z: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u = phase_repair.statistical_variable(z, scale)
    gauged, _, anchors = phase_repair.max_q_gauge(z, scale)
    phi = phase_repair.wrap_phase(torch.angle(gauged).to(torch.float64))
    amplitude = torch.log(u.to(torch.float64) + 1.0e-12)
    phase = torch.cat((phi.cos(), phi.sin()), dim=1)
    _finite(amplitude, "log-amplitude representation")
    _finite(phase, "phase representation")
    return amplitude, phase, anchors.to(torch.long).cpu()


def _component_geometry_with_predictions(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    dev_features: torch.Tensor,
    dev_labels: torch.Tensor,
    sample_features: torch.Tensor,
    dev_predictions: torch.Tensor,
    sample_predictions: torch.Tensor,
) -> dict[str, Any]:
    train_std, dev_std, sample_std, mean, std = _standardize_train_only(train_features, dev_features, sample_features)
    centroids = torch.stack([train_std[train_labels == digit].mean(dim=0) for digit in range(CLASS_COUNT)])

    def one(query: torch.Tensor, labels: torch.Tensor | None, predictions: torch.Tensor) -> dict[str, Any]:
        distances = torch.cdist(query, centroids, p=2)
        nearest = distances.argmin(dim=1)
        result: dict[str, Any] = {
            "nearest_class_histogram": [int(value) for value in torch.bincount(nearest, minlength=CLASS_COUNT)],
            "nearest_class": [int(value) for value in nearest],
            "nearest_class_distance": _summary(distances.min(dim=1).values),
            "predicted_class_centroid_distance": _summary(distances.gather(1, predictions[:, None]).squeeze(1)),
            "nearest_centroid_predicted_class_agreement": float((nearest == predictions).to(torch.float64).mean()),
            "labels_consumed": labels is not None,
        }
        if labels is not None:
            result["nearest_centroid_true_class_fraction"] = float((nearest == labels).to(torch.float64).mean())
        return result

    return {
        "standardization": {"fit_role": "real TRAIN only", "feature_count": int(train_features.shape[1]), "mean_tensor_sha256": tensor_sha256(mean), "std_tensor_sha256": tensor_sha256(std), "constant_feature_count": int((std == 1.0).sum().item())},
        "real_dev": one(dev_std, dev_labels, dev_predictions),
        "sampled": one(sample_std, None, sample_predictions),
        "centroids": centroids,
    }


def _cmos_view(raw: torch.Tensor) -> torch.Tensor:
    _require(raw.ndim == 4 and raw.shape[1:] == (1, 200, 200), "raw CMOS geometry changed")
    return F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))


def _evaluator_predictions(raw: torch.Tensor, models: Mapping[str, torch.nn.Module], device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, list[torch.Tensor]] = {name: [] for name in models}
    for model in models.values():
        model.eval()
        _require(all(not parameter.requires_grad for parameter in model.parameters()), "evaluator is not frozen")
    with torch.inference_mode():
        for start in range(0, raw.shape[0], EVALUATOR_BATCH_SIZE):
            view = _cmos_view(raw[start : start + EVALUATOR_BATCH_SIZE].to(device))
            for name, model in models.items():
                _, logits = model(view)
                _finite(logits, f"{name} logits")
                result[name].append(logits.argmax(dim=1).cpu())
    return {name: torch.cat(values) for name, values in result.items()}


def _retention_report(
    latent_prediction: torch.Tensor,
    predictions_by_step: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "latent_prediction_source": "frozen real-OLS linear probe applied to sampled z; descriptive only",
        "labels_consumed": False,
        "checkpoints": {},
        "per_latent_predicted_class": {},
    }
    for step, predictions in predictions_by_step.items():
        consensus = predictions["E_S"].eq(predictions["E_I"]) & predictions["E_I"].eq(predictions["E_C"])
        consensus_class = predictions["E_S"]
        step_report: dict[str, Any] = {
            "rows": int(latent_prediction.numel()),
            "agreement_latent_vs_output": {name: float((latent_prediction == values).to(torch.float64).mean()) for name, values in predictions.items()},
            "all_three_output_evaluator_consensus_fraction": float(consensus.to(torch.float64).mean()),
            "latent_agreement_with_consensus_where_consensus_exists": None if not bool(consensus.any()) else float((latent_prediction[consensus] == consensus_class[consensus]).to(torch.float64).mean()),
            "consensus_class_histogram": [int(value) for value in torch.bincount(consensus_class[consensus], minlength=CLASS_COUNT)] if bool(consensus.any()) else [0] * CLASS_COUNT,
            "output_class_histograms": {name: [int(value) for value in torch.bincount(values, minlength=CLASS_COUNT)] for name, values in predictions.items()},
        }
        report["checkpoints"][str(step)] = step_report
        for digit in range(CLASS_COUNT):
            mask = latent_prediction == digit
            key = str(digit)
            report["per_latent_predicted_class"].setdefault(key, {})[str(step)] = {
                "rows": int(mask.sum()),
                "E_S_retention": None if not bool(mask.any()) else float((predictions["E_S"][mask] == digit).to(torch.float64).mean()),
                "E_I_retention": None if not bool(mask.any()) else float((predictions["E_I"][mask] == digit).to(torch.float64).mean()),
                "E_C_retention": None if not bool(mask.any()) else float((predictions["E_C"][mask] == digit).to(torch.float64).mean()),
                "all_three_consensus_and_retained": None if not bool(mask.any()) else float((consensus[mask] & (consensus_class[mask] == digit)).to(torch.float64).mean()),
            }
    return report


def _neighbor_bridge(
    latent_prediction: torch.Tensor,
    nearest_indices: torch.Tensor,
    train_labels: torch.Tensor,
    distance: torch.Tensor,
    predictions_by_step: Mapping[int, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    neighbor_labels = train_labels.index_select(0, nearest_indices)
    return {
        "nearest_real_train_row_class": [int(value) for value in neighbor_labels],
        "nearest_real_train_row_index_in_train_ols": [int(value) for value in nearest_indices],
        "agreement_latent_prediction_with_nearest_neighbor_class": float((latent_prediction == neighbor_labels).to(torch.float64).mean()),
        "nearest_distance": _summary(distance),
        "output_evaluator_agreement_with_nearest_neighbor_class": {
            str(step): {name: float((values == neighbor_labels).to(torch.float64).mean()) for name, values in predictions.items()}
            for step, predictions in predictions_by_step.items()
        },
        "nearest_neighbor_class_histogram": [int(value) for value in torch.bincount(neighbor_labels, minlength=CLASS_COUNT)],
        "sampled_labels_are_not_assumed": True,
    }


def _load_checkpoint(root: Path, descriptor: Path, step: int) -> Mapping[str, Any]:
    path = root / descriptor
    expected = readiness.REAL_CHECKPOINTS[step]["sha256"]
    _require(file_sha256(path) == expected, f"step {step} checkpoint identity changed")
    payload = _mapping(safe_torch_load(path, expected_sha256=expected, map_location="cpu"), f"step {step} checkpoint")
    _require(payload.get("completed_updates") == step, f"step {step} checkpoint step changed")
    _require(payload.get("scientific_artifact") is True and payload.get("operational_artifact") is False, f"step {step} checkpoint role changed")
    return payload


def _load_inputs(root: Path) -> dict[str, Any]:
    git = robust.git_identity(root)
    _require(git["branch"] == EXPECTED_BRANCH and git["head"] == EXPECTED_HEAD, "branch/HEAD conflicts with accepted authorities")
    localization_inputs = localization._load_authorities(root)
    robust_authorities, run_inputs, preflight = readiness._load_authorities_and_preflight(root)

    post_report_path = root / POST3000_REPORT
    post_report = json.loads(post_report_path.read_text(encoding="utf-8"))
    _require(post_report.get("result") == "VALID_COMPLETED", "post3000 authority is not VALID_COMPLETED")
    _require(post_report.get("outer_gate") == readiness.OUTER_GATE, "post3000 authority gate changed")
    _require(post_report.get("classification") == readiness.CLASSIFICATION_REAL_ONLY, "post3000 accepted classification changed")
    _require(post_report.get("next_action") == "NONE", "post3000 authority is not closed")

    train_z = _tensor(localization_inputs["train_z"], "accepted TRAIN OLS")
    dev_z = _tensor(localization_inputs["dev_z"], "accepted DEV OLS")
    train_indices = _tensor(localization_inputs["train_indices"], "accepted TRAIN source indices")
    dev_indices = _tensor(localization_inputs["dev_indices"], "accepted DEV source indices")
    labels = _tensor(robust_authorities["labels"], "accepted labels")
    train_labels = labels.index_select(0, train_indices)
    dev_labels = labels.index_select(0, dev_indices)
    _require(torch.equal(dev_labels, localization_inputs["dev_labels"]), "accepted DEV labels differ from canonical labels")
    _require(train_z.shape == (55_000, 100) and dev_z.shape == (5_000, 100), "real OLS shape changed")
    _require(train_z.dtype == dev_z.dtype == torch.complex64, "real OLS dtype changed")
    _finite(train_z.real, "TRAIN OLS real part")
    _finite(train_z.imag, "TRAIN OLS imaginary part")
    _finite(dev_z.real, "DEV OLS real part")
    _finite(dev_z.imag, "DEV OLS imaginary part")
    _require(torch.unique(torch.cat((train_indices, dev_indices))).numel() == 60_000, "TRAIN/DEV source identity overlap detected")

    sample = preflight["sample"]
    sampled_z = _tensor(sample["z_sample"], "fixed sampled Q2 z")
    _require(sampled_z.shape == (SAMPLE_COUNT, 100) and sampled_z.dtype == torch.complex64, "fixed sampled Q2 z shape/dtype changed")
    _finite(sampled_z.real, "sampled Q2 real part")
    _finite(sampled_z.imag, "sampled Q2 imaginary part")

    raw_by_step: dict[int, torch.Tensor] = {0: _tensor(sample["baseline_raw"], "accepted clean Q2 raw")}
    for item in sample["accepted_artifacts"]:
        step = int(item["step"])
        raw_by_step[step] = _tensor(item["raw"], f"accepted P2/Q2 step {step} raw")
        _require(torch.equal(_tensor(safe_torch_load(root / item["artifact"]["path"], expected_sha256=item["artifact"]["sha256"], map_location="cpu").get("input_z"), f"step {step} input z"), sampled_z), f"step {step} sampled identity changed")
    for step in (0, 300, 3000):
        raw = raw_by_step[step]
        _require(raw.shape == (SAMPLE_COUNT, 1, 200, 200) and raw.dtype == torch.float32, f"step {step} raw geometry changed")
        _finite(raw, f"step {step} raw")

    checkpoints = {step: _load_checkpoint(root, path, step) for step, path in CHECKPOINTS.items()}
    encoder0 = _tensor(checkpoints[0]["encoder_state"], "step0 encoder")
    _require(all(torch.equal(encoder0, _tensor(checkpoints[step]["encoder_state"], f"step{step} encoder")) for step in (300, 3000)), "encoder changed across requested checkpoints")
    probe = _load_probe(root)
    return {
        "git": git,
        "localization": localization_inputs,
        "robust": robust_authorities,
        "run_inputs": run_inputs,
        "preflight": preflight,
        "post_report": post_report,
        "post_report_path": post_report_path,
        "train_z": train_z,
        "dev_z": dev_z,
        "train_indices": train_indices,
        "dev_indices": dev_indices,
        "labels": labels,
        "train_labels": train_labels,
        "dev_labels": dev_labels,
        "sampled_z": sampled_z,
        "raw_by_step": raw_by_step,
        "checkpoints": checkpoints,
        "probe": probe,
        "scale": _tensor(robust_authorities["cpu_context"]["scale"], "accepted OLS scale"),
    }


def _add_panel_label(axis: Any, label: str) -> None:
    offset = ScaledTranslation(-3 / 72, 3 / 72, axis.figure.dpi_scale_trans)
    axis.text(0, 1, label, transform=axis.transAxes + offset, ha="left", va="bottom", fontsize=8, fontweight="bold")


def _load_alignment_helper() -> Any:
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError as error:  # pragma: no cover - environment contract
        raise ManifoldGapError("nature-figure audit_panel_alignment.py is required for visual QA") from error
    return require_matplotlib_panel_alignment


def _save_figure(fig: Any, target: Path, stem: str) -> dict[str, Any]:
    fig.tight_layout(pad=1.0)
    require_matplotlib_panel_alignment = _load_alignment_helper()
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(target / f"{stem}.alignment.json"),
        overlay_svg=str(target / f"{stem}.alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    fig.savefig(target / f"{stem}.svg", bbox_inches="tight")
    fig.savefig(target / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(target / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(target / f"{stem}.tiff", dpi=600, bbox_inches="tight")
    paths = {suffix: {"path": f"{stem}.{suffix}", "sha256": file_sha256(target / f"{stem}.{suffix}")} for suffix in ("svg", "pdf", "png", "tiff")}
    plt.close(fig)
    return {
        "files": paths,
        "alignment_json": {"path": f"{stem}.alignment.json", "sha256": file_sha256(target / f"{stem}.alignment.json")},
        "alignment_svg": {"path": f"{stem}.alignment.svg", "sha256": file_sha256(target / f"{stem}.alignment.svg")},
    }


def _render_bridge_figure(
    target: Path,
    sampled_raw: torch.Tensor,
    neighbor_raw: torch.Tensor,
    target_images: torch.Tensor,
    sampled_indices: tuple[int, ...],
) -> dict[str, Any]:
    sample_view = _cmos_view(sampled_raw.index_select(0, torch.tensor(sampled_indices)))
    neighbor_view = _cmos_view(neighbor_raw)
    target_view = target_images.index_select(0, torch.tensor(sampled_indices))
    panels = (("sampled output · step 3000", sample_view), ("nearest real-OLS output", neighbor_view), ("nearest real target", target_view))
    fig, axes = plt.subplots(3, len(sampled_indices), figsize=(7.204724409448819, 3.7), squeeze=False)
    label_index = 0
    for row, (title, values) in enumerate(panels):
        for column, index in enumerate(sampled_indices):
            axis = axes[row, column]
            image = values[column, 0].detach().cpu()
            axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
            if row == 0:
                axis.set_title(f"z[{index}]", fontsize=7)
            axis.set_xticks([])
            axis.set_yticks([])
            _add_panel_label(axis, chr(ord("a") + label_index))
            label_index += 1
        fig.text(0.01, 0.82 - row * 0.31, title, rotation=90, rotation_mode="anchor", va="center", ha="center", fontsize=6.5)
    return _save_figure(fig, target, "fixed_sampled_nearest_real_bridge")


def _classify(
    probe_sample: Mapping[str, Any],
    probe_dev: Mapping[str, Any],
    support: Mapping[str, Any],
    geometry: Mapping[str, Any],
    components: Mapping[str, Mapping[str, Any]],
    retention: Mapping[str, Any],
) -> tuple[str, str, str, dict[str, Any]]:
    sample_diverse = int(probe_sample["represented_class_count"]) >= 8
    sample_confident = float(probe_sample["top1_confidence"]["mean"]) >= float(probe_dev["top1_confidence"]["q50"])
    sample_geometry_coherent = float(geometry["sampled"]["predicted_class_nearest_centroid_agreement"]) >= 0.5
    support_payload = _mapping(support["sampled"], "sample support")
    support_any = float(support_payload["sample_support_fraction_q95"]["any"])
    support_pred = float(support_payload["sample_support_fraction_q95"]["predclass"])
    real_any_q95 = float(support["real_dev_control"]["d_real_any"]["q95"])
    sample_any_q95 = float(support_payload["d_sample_any"]["q95"])
    real_same_q95 = float(support["real_dev_control"]["d_real_same"]["q95"])
    sample_pred_q95 = float(support_payload["d_sample_predclass"]["q95"])
    support_mismatch = sample_any_q95 > real_any_q95 and sample_pred_q95 > real_same_q95 and support_any < 1.0 and support_pred < 1.0
    agreements = [
        float(retention["checkpoints"][str(step)]["agreement_latent_vs_output"][name])
        for step in (0, 300, 3000)
        for name in ("E_S", "E_I", "E_C")
    ]
    retention_weak = max(agreements) < 0.5
    amp_q95 = float(components["amplitude"]["sampled"]["nearest_class_distance"]["q95"])
    amp_real_q95 = float(components["amplitude"]["real_dev"]["nearest_class_distance"]["q95"])
    phase_q95 = float(components["phase"]["sampled"]["nearest_class_distance"]["q95"])
    phase_real_q95 = float(components["phase"]["real_dev"]["nearest_class_distance"]["q95"])
    amplitude_dominant = amp_q95 > amp_real_q95 and phase_q95 <= phase_real_q95
    phase_dominant = phase_q95 > phase_real_q95 and amp_q95 <= amp_real_q95
    evidence = {
        "sample_diverse": sample_diverse,
        "sample_confident_relative_to_real_dev_median": sample_confident,
        "sample_logit_geometry_coherent": sample_geometry_coherent,
        "support_mismatch": support_mismatch,
        "retention_weak": retention_weak,
        "amplitude_dominant_relative_q95": amplitude_dominant,
        "phase_dominant_relative_q95": phase_dominant,
        "decision_threshold_note": "Relative diagnostic comparisons are recorded; q95 support fractions are not universal acceptance rules.",
    }
    if not sample_diverse or not sample_confident or not sample_geometry_coherent:
        return CLASSIFICATION_A, "The frozen OLS probe does not decode a sufficiently diverse/coherent semantic code from the sampled bank before Generator transfer.", "DEFINE_SEMANTICALLY_CONDITIONED_OLS_SAMPLER_V1", evidence
    if support_mismatch and retention_weak:
        return CLASSIFICATION_B, "The sampled bank carries a diverse probe-decodable code but lies farther from real TRAIN OLS support; weak output retention is therefore consistent with local-manifold support mismatch.", "REPAIR_SAMPLED_OLS_LOCAL_SEMANTIC_MANIFOLD_SUPPORT_V1", evidence
    if not support_mismatch and retention_weak:
        return CLASSIFICATION_C, "The sampled code is probe-decodable and locally supported, but the frozen Generator does not retain that code through the sampled output.", "ATTRIBUTE_REAL_VS_SAMPLED_GENERATOR_LOCAL_JACOBIAN_GAP_V1", evidence
    if amplitude_dominant:
        return CLASSIFICATION_D, "The joint gap is more visible in the class-conditioned amplitude representation than in the phase representation.", "CLASS_CONDITIONED_AMPLITUDE_PRODUCER_DESIGN", evidence
    if phase_dominant:
        return CLASSIFICATION_E, "The joint gap is more visible in the class-conditioned phase representation than in the amplitude representation.", "CLASS_CONDITIONED_PHASE_PRODUCER_DESIGN", evidence
    return CLASSIFICATION_F, "Probe code, local support, component geometry, and Generator retention do not isolate one mechanism under the frozen comparisons.", "DESIGN_A_SINGLE_GAP_LOCALIZATION_FOLLOWUP", evidence


def run_diagnostic(*, repo_root: Path | None = None, output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    root = robust._root() if repo_root is None else repo_root.resolve()
    target = (root / output_dir if not Path(output_dir).is_absolute() else Path(output_dir)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "output must be a distinct child under outputs/")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "protected output namespace is forbidden")
    _require(not target.exists(), f"write-once output already exists: {target}")

    inputs = _load_inputs(root)
    target.mkdir(parents=True, exist_ok=False)
    train_z = inputs["train_z"]
    dev_z = inputs["dev_z"]
    sampled_z = inputs["sampled_z"]
    train_labels = inputs["train_labels"]
    dev_labels = inputs["dev_labels"]
    probe = inputs["probe"]

    train_features, dev_features, sample_features, mean, std = _standardize_train_only(_ols_features(train_z), _ols_features(dev_z), _ols_features(sampled_z))
    _require(tensor_sha256(mean) == "5c71f3111e653cd6643c3c48246f84b7104cf4d7c8d847011b0ae4197cde22c8", "reconstructed OLS probe mean identity changed")
    _require(tensor_sha256(std) == "981cfaf7d151b3305f6406c75f053199f8a4cfd2a28076aa94fd2c05a9fd7d48", "reconstructed OLS probe std identity changed")
    train_logits = _probe_logits(_ols_features(train_z), mean, std, probe)
    dev_logits = _probe_logits(_ols_features(dev_z), mean, std, probe)
    sample_logits = _probe_logits(_ols_features(sampled_z), mean, std, probe)
    probe_train = _probe_distribution(train_logits, labels=train_labels)
    probe_dev = _probe_distribution(dev_logits, labels=dev_labels)
    probe_sample = _probe_distribution(sample_logits, labels=None)
    _require(abs(float(probe_dev["macro_recall"]) - 0.9226) < 1.0e-12, "accepted OLS probe DEV qualification did not reproduce")
    for payload in (probe_train, probe_dev, probe_sample):
        payload.pop("predictions", None)
        payload.pop("logits", None)
    sample_predictions = _probe_distribution(sample_logits, labels=None)["predictions"]
    dev_predictions = _probe_distribution(dev_logits, labels=dev_labels)["predictions"]

    support, sample_any_distance, sample_nn_indices = _support_report(
        train_features,
        train_labels,
        dev_features,
        dev_labels,
        sample_features,
        sample_predictions,
    )
    geometry_raw = _logit_geometry(train_logits, train_labels, dev_logits, dev_labels, sample_logits, sample_predictions)
    geometry = {key: value for key, value in geometry_raw.items() if key != "class_centroids"}

    scale = inputs["scale"].to(device="cpu", dtype=torch.float64)
    train_amp, train_phase, train_anchors = _component_features(train_z, scale)
    dev_amp, dev_phase, dev_anchors = _component_features(dev_z, scale)
    sample_amp, sample_phase, sample_anchors = _component_features(sampled_z, scale)
    component_raw = {
        "amplitude": _component_geometry_with_predictions(train_amp, train_labels, dev_amp, dev_labels, sample_amp, dev_predictions, sample_predictions),
        "phase": _component_geometry_with_predictions(train_phase, train_labels, dev_phase, dev_labels, sample_phase, dev_predictions, sample_predictions),
    }
    component_reports = {name: {key: value for key, value in payload.items() if key != "centroids"} for name, payload in component_raw.items()}

    cuda_runtime = phase_repair._prepare_cuda_runtime_dll_search_path()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "frozen evaluator bridge requires CUDA")
    consumer, smooth, independent, ec = readiness._load_models(root, inputs["robust"], torch.device("cuda:0"))
    models = {"E_S": smooth, "E_I": independent, "E_C": ec}
    predictions_by_step: dict[int, dict[str, torch.Tensor]] = {}
    for step in (0, 300, 3000):
        predictions_by_step[step] = _evaluator_predictions(inputs["raw_by_step"][step], models, consumer.device)
    retention = _retention_report(sample_predictions, predictions_by_step)

    neighbor_labels = train_labels.index_select(0, sample_nn_indices)
    neighbor_bridge = _neighbor_bridge(sample_predictions, sample_nn_indices, train_labels, sample_any_distance, predictions_by_step)

    checkpoint = inputs["checkpoints"][3000]
    consumer.load_trainable_state_dict({"encoder.phase": checkpoint["encoder_state"], "generator.phase": checkpoint["generator_state"]})
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
    selected = torch.tensor(VISUAL_SAMPLE_INDICES, dtype=torch.long)
    nearest_z = sampled_z.index_select(0, selected).new_empty((len(VISUAL_SAMPLE_INDICES), 100))
    nearest_z.copy_(train_z.index_select(0, sample_nn_indices.index_select(0, selected)))
    with torch.inference_mode():
        nearest_raw = consumer.generated_forward(nearest_z.to(consumer.device)).raw_intensity.detach().cpu()
    _finite(nearest_raw, "nearest real-OLS generated raw")
    nearest_source_indices = inputs["train_indices"].index_select(0, sample_nn_indices.index_select(0, selected))
    target_images = inputs["robust"]["images"].index_select(0, nearest_source_indices)
    visual = _render_bridge_figure(target, inputs["raw_by_step"][3000], nearest_raw, target_images, VISUAL_SAMPLE_INDICES)

    component_for_classification = component_reports
    classification, mechanism, successor, classification_evidence = _classify(
        probe_sample,
        probe_dev,
        support,
        geometry,
        component_for_classification,
        retention,
    )
    _write_json(target / "ols_probe_diagnostics.json", {"schema": "real_vs_sampled_ols_probe_diagnostics.v1", "real_train": probe_train, "real_dev": probe_dev, "sampled": probe_sample, "probe_authority": _artifact_ref(root, probe["path"], "accepted frozen OLS probe weights")}, kind="OLS probe diagnostics")
    _write_json(target / "local_manifold_support.json", {"schema": "real_vs_sampled_ols_local_manifold_support.v1", **support}, kind="local OLS manifold support")
    _write_json(target / "semantic_logit_geometry.json", {"schema": "real_vs_sampled_ols_semantic_logit_geometry.v1", **geometry}, kind="semantic logit geometry")
    _write_json(target / "amplitude_geometry.json", {"schema": "real_vs_sampled_ols_amplitude_geometry.v1", **component_reports["amplitude"], "max_q_anchor_histograms": {"real_train": [int(value) for value in torch.bincount(train_anchors, minlength=100)], "real_dev": [int(value) for value in torch.bincount(dev_anchors, minlength=100)], "sampled": [int(value) for value in torch.bincount(sample_anchors, minlength=100)]}}, kind="amplitude geometry")
    _write_json(target / "phase_geometry.json", {"schema": "real_vs_sampled_ols_phase_geometry.v1", **component_reports["phase"], "max_q_gauge": "accepted max-q gauge using the frozen OLS scale", "max_q_anchor_histograms": {"real_train": [int(value) for value in torch.bincount(train_anchors, minlength=100)], "real_dev": [int(value) for value in torch.bincount(dev_anchors, minlength=100)], "sampled": [int(value) for value in torch.bincount(sample_anchors, minlength=100)]}}, kind="phase geometry")
    _write_json(target / "latent_to_output_retention.json", {"schema": "real_vs_sampled_ols_latent_to_output_retention.v1", **retention}, kind="latent-to-output retention")
    _write_json(target / "nearest_real_neighbor_bridge.json", {"schema": "real_vs_sampled_ols_nearest_real_neighbor_bridge.v1", **neighbor_bridge}, kind="nearest real neighbor bridge")
    _write_torch(target / "diagnostic_tensors.pt", {"schema": "real_vs_sampled_ols_semantic_manifold_gap_tensors.v1", "sampled_z": sampled_z, "sampled_probe_logits": sample_logits, "sampled_probe_prediction": sample_predictions, "nearest_real_train_index": sample_nn_indices, "nearest_real_train_class": neighbor_labels}, kind="semantic manifold diagnostic tensors")
    _write_json(target / "figure_contract.json", {"core_conclusion": "The fixed sampled Q2 OLS population either does or does not preserve the real-OLS class code and local support required for Generator transfer; this figure links sampled output to its nearest real-OLS training neighbor without treating the neighbor as sampled ground truth.", "results_level_question": "Does a fixed sampled OLS row resemble a real OLS row in the same semantic manifold before Generator output?", "figure_archetype": "image plate + quant", "backend": "Python/matplotlib", "panel_map": {"row_1": "fixed sampled step3000 output", "row_2": "step3000 output from nearest real-TRAIN OLS neighbor", "row_3": "corresponding nearest real target image"}, "fixed_indices": list(VISUAL_SAMPLE_INDICES), "display_integrity": "all panels use the same clamp-and-pool [0,1] preprocessing; no per-image contrast stretch", "review_risk": "nearest-neighbor class is diagnostic only and does not provide sampled ground truth"}, kind="figure contract")

    report = {
        "schema": "real_vs_sampled_ols_semantic_manifold_gap_report.v1",
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "task_id": TASK_ID,
        "authorization_consumed": {"token": AUTHORIZATION_TOKEN, "id": AUTHORIZATION_ID, "one_authorization": True},
        "outer_gate": OUTER_GATE,
        "outer_gate_passed": True,
        "input_authority": "POST3000_GENERATION_READINESS_AND_SCALING_RESOLVED",
        "local_state": {"branch": inputs["git"]["branch"], "head": inputs["git"]["head"], "worktree_dirty": bool(inputs["git"].get("worktree_dirty", True)), "working_tree_preserved": True},
        "input_authorities": {"post3000_report": _artifact_ref(root, inputs["post_report_path"], "accepted post3000 generation-readiness authority"), "accepted_train_ols": _artifact_ref(root, root / localization.dev_materializer.FINAL_TRAIN_OLS, "accepted real TRAIN complex OLS"), "accepted_dev_ols": _artifact_ref(root, root / localization.DEV_OLS, "accepted real DEV complex OLS"), "accepted_probe": _artifact_ref(root, probe["path"], "accepted frozen OLS linear probe"), "fixed_q2_bank": {"sha256": inputs["preflight"]["sample"]["latent_bank_sha256"], "rows": SAMPLE_COUNT, "labels_consumed": False}, "scale_and_gauge": {"scale_tensor_sha256": tensor_sha256(scale), "max_q_gauge": "phase_repair.max_q_gauge", "transport": "T=I_100 coefficient space", "OLS_interface": "lightgen_10x10_complex_ols_v1"}, "frozen_checkpoints": {str(step): _artifact_ref(root, root / readiness.REAL_CHECKPOINTS[step]["path"], f"frozen Generator checkpoint step {step}") for step in (0, 300, 3000)}, "frozen_evaluators": inputs["preflight"]["preflight"]["frozen_evaluators"]},
        "real_ols_semantic_reference": {"probe_qualification": {"TRAIN": probe_train, "DEV": probe_dev, "accepted_DEV_macro_recall": 0.9226}, "dev_class_geometry": geometry["real_dev"]},
        "sampled_ols_probe_diagnostics": {**probe_sample, "labels_consumed": False, "comparison_to_real_DEV": {"real_DEV": {key: value for key, value in probe_dev.items() if key not in {"confusion_matrix_actual_rows_predicted_columns", "per_class_recall", "accuracy", "macro_recall", "labels_consumed", "rows"}}, "interpretation": "descriptive probe-output comparison; no sampled accuracy"}},
        "local_manifold_support": support,
        "semantic_logit_geometry": geometry,
        "amplitude_geometry": component_reports["amplitude"],
        "phase_geometry": component_reports["phase"],
        "latent_to_output_retention": retention,
        "nearest_real_neighbor_bridge": neighbor_bridge,
        "fixed_visual_link": {"artifact": visual, "nearest_real_train_source_indices": [int(value) for value in nearest_source_indices], "labels_used_only_for_neighbor_identity": True},
        "classification": classification,
        "mechanism": mechanism,
        "classification_evidence": classification_evidence,
        "recommended_single_successor": successor,
        "claim_boundary": "This diagnostic localizes the fixed Q2 sampled OLS class-code, local-support, component-geometry, and frozen-Generator retention relationships. It does not establish sampled ground-truth labels, an optimal sampler family, final generation acceptance, physical hardware validity, or promotion.",
        "files_changed": {"repository_source": ["scripts/research/lightgen/execute_lightgen_2d_real_vs_sampled_ols_semantic_manifold_gap_v1.py"], "scientific_artifacts": sorted(path.name for path in target.iterdir())},
        "validation": {"cuda": True, "cuda_runtime": cuda_runtime, "forward_only": True, "backward_called": False, "optimizer_constructed": False, "model_fitting": False, "latent_resampling": False, "protected_test_sealed_reads": False, "sampled_labels_consumed": False, "probe_DEV_reproduction": True, "figure_backend": "Python/matplotlib", "figure_alignment_gate": "passed during render"},
        "unresolved": ["Sampled classes, nearest-real class, evaluator agreement, and visual bridge remain descriptive because the fixed bank has no sampled ground-truth labels.", "The diagnostic does not identify a unique latent-model family or authorize automatic repair."],
        "next_action": "NONE",
        "output_dir": _relative(root, target),
    }
    _write_json(target / "report.json", report, kind="real-vs-sampled OLS semantic manifold gap report")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="consume the explicit authorization and run the frozen diagnostic")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": AUTHORIZATION_TOKEN, "outer_gate": OUTER_GATE, "output_dir": args.output_dir, "no_update": True}, ensure_ascii=False, indent=2))
        return 0
    try:
        report = run_diagnostic(output_dir=args.output_dir)
    except ManifoldGapError as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "outer_gate": report["outer_gate"], "classification": report["classification"], "recommended_single_successor": report["recommended_single_successor"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
