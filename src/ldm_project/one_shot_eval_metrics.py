from __future__ import annotations

from typing import Any, cast


MetricsPayload = dict[str, Any]

STRICT_STUDENT_READABILITY_THRESHOLDS = {
    "paired_corr": 0.78,
    "paired_dice": 0.72,
    "student_high_saturation_ratio": 0.08,
}


def _eval_correlation(torch_module, prediction, target):
    prediction_flat = prediction.detach().float().flatten(1)
    target_flat = target.detach().float().flatten(1)
    prediction_centered = prediction_flat - prediction_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    denominator = prediction_centered.pow(2).mean(dim=1).sqrt() * target_centered.pow(2).mean(dim=1).sqrt()
    return ((prediction_centered * target_centered).mean(dim=1) / denominator.clamp_min(1.0e-8)).mean()


def _eval_foreground_dice(prediction, target, threshold: float = 0.2):
    prediction_mask = prediction.detach().float().clamp(0.0, 1.0) > threshold
    target_mask = target.detach().float().clamp(0.0, 1.0) > threshold
    dims = tuple(range(1, prediction_mask.ndim))
    intersection = (prediction_mask & target_mask).float().sum(dim=dims)
    denominator = prediction_mask.float().sum(dim=dims) + target_mask.float().sum(dim=dims)
    return (2.0 * intersection / denominator.clamp_min(1.0)).mean()


def _eval_camera_saturation_ratio(camera_prediction):
    camera = camera_prediction.detach().float()
    saturated = (camera >= 0.99) | (camera <= 0.01)
    return saturated.float().mean()


def _eval_camera_low_saturation_ratio(camera_prediction):
    camera = camera_prediction.detach().float()
    return (camera <= 0.01).float().mean()


def _eval_camera_high_saturation_ratio(camera_prediction):
    camera = camera_prediction.detach().float()
    return (camera >= 0.99).float().mean()


def _eval_foreground_hole_ratio(prediction, target, threshold: float = 0.2, hole_threshold: float = 0.1):
    prediction_mask = prediction.detach().float().clamp(0.0, 1.0) < hole_threshold
    target_mask = target.detach().float().clamp(0.0, 1.0) > threshold
    dims = tuple(range(1, target_mask.ndim))
    holes = (prediction_mask & target_mask).float().sum(dim=dims)
    foreground = target_mask.float().sum(dim=dims)
    return (holes / foreground.clamp_min(1.0)).mean()


def _eval_total_variation(tensor):
    diff_y = (tensor[..., 1:, :] - tensor[..., :-1, :]).abs().mean()
    diff_x = (tensor[..., :, 1:] - tensor[..., :, :-1]).abs().mean()
    return diff_x + diff_y


def _mean_pairwise_mse(torch_module, samples) -> float | None:
    sample_count = int(samples.shape[0])
    if sample_count < 2:
        return None
    flattened = samples.detach().float().flatten(1)
    pairwise = (flattened[:, None, :] - flattened[None, :, :]).pow(2).mean(dim=-1)
    upper = torch_module.triu(torch_module.ones(sample_count, sample_count, dtype=torch_module.bool), diagonal=1)
    return float(pairwise[upper].mean().item())


def _mean_pairwise_corr(torch_module, samples) -> float | None:
    sample_count = int(samples.shape[0])
    if sample_count < 2:
        return None
    flattened = samples.detach().float().flatten(1)
    centered = flattened - flattened.mean(dim=1, keepdim=True)
    rms = centered.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1.0e-8)
    normalized = centered / rms
    pairwise = (normalized[:, None, :] * normalized[None, :, :]).mean(dim=-1)
    upper = torch_module.triu(torch_module.ones(sample_count, sample_count, dtype=torch_module.bool), diagonal=1)
    return float(pairwise[upper].mean().item())


def _eval_foreground_ratio(target, threshold: float = 0.2):
    target_clamped = target.detach().float().clamp(0.0, 1.0)
    foreground = (target_clamped > threshold).float()
    dims = tuple(range(1, foreground.ndim))
    return foreground.mean(dim=dims).mean()


def _eval_background_speckle_ratio(prediction, target, threshold: float = 0.2):
    prediction_clamped = prediction.detach().float().clamp(0.0, 1.0)
    target_clamped = target.detach().float().clamp(0.0, 1.0)
    background = target_clamped <= threshold
    if not background.any():
        return prediction_clamped.new_tensor(0.0)
    dims = tuple(range(1, background.ndim))
    bg_energy = (prediction_clamped * background.float()).sum(dim=dims) / background.float().sum(dim=dims).clamp_min(1.0)
    return bg_energy.mean()


def _eval_readable_contrast(prediction, target, threshold: float = 0.2):
    prediction_clamped = prediction.detach().float().clamp(0.0, 1.0)
    target_clamped = target.detach().float().clamp(0.0, 1.0)
    foreground = target_clamped > threshold
    background = target_clamped <= threshold
    dims = tuple(range(1, foreground.ndim))
    fg_mean = (prediction_clamped * foreground.float()).sum(dim=dims) / foreground.float().sum(dim=dims).clamp_min(1.0)
    bg_mean = (prediction_clamped * background.float()).sum(dim=dims) / background.float().sum(dim=dims).clamp_min(1.0)
    return (fg_mean / bg_mean.clamp_min(1e-6)).mean()


def _label_audit_confusion_by_class(audit_payload: MetricsPayload) -> dict[str, dict[str, int]]:
    confusion: dict[str, dict[str, int]] = {}
    per_sample = cast(list[MetricsPayload], audit_payload.get("per_sample", []))
    for item in per_sample:
        requested = str(int(item["requested_label"]))
        nearest = str(int(item["nearest_label"]))
        confusion.setdefault(requested, {})
        confusion[requested][nearest] = confusion[requested].get(nearest, 0) + 1
    return confusion


def add_student_label_audit_metrics(metrics: MetricsPayload, audit_payload: MetricsPayload) -> None:
    per_class = cast(dict[str, MetricsPayload], audit_payload.get("per_class", {}))
    overall = cast(MetricsPayload, metrics.setdefault("overall", {}))
    label_ratios = [
        float(class_metrics["label_correct_ratio"])
        for class_metrics in per_class.values()
        if "label_correct_ratio" in class_metrics
    ]
    label_margin_means = [
        float(class_metrics["label_margin"]["mean"])
        for class_metrics in per_class.values()
        if "label_margin" in class_metrics
    ]
    overall["student_label_correct_count"] = int(audit_payload.get("label_correct_count", 0))
    overall["student_label_correct_ratio"] = float(audit_payload.get("label_correct_ratio", 0.0))
    audit_overall = cast(MetricsPayload, audit_payload.get("overall", {}))
    label_margin = cast(MetricsPayload, audit_overall.get("label_margin", {}))
    overall["student_label_margin_mean"] = float(label_margin.get("mean", 0.0))
    overall["min_class_student_label_correct_ratio"] = min(label_ratios) if label_ratios else 0.0
    overall["min_class_student_label_margin_mean"] = min(label_margin_means) if label_margin_means else 0.0

    confusion = _label_audit_confusion_by_class(audit_payload)
    metric_per_class = cast(dict[str, MetricsPayload], metrics.setdefault("per_class", {}))
    for label, class_metrics in per_class.items():
        target = metric_per_class.setdefault(str(label), {})
        target["student_label_correct_count"] = int(class_metrics.get("label_correct_count", 0))
        target["student_label_correct_ratio"] = float(class_metrics.get("label_correct_ratio", 0.0))
        target["student_label_margin_mean"] = float(class_metrics.get("label_margin", {}).get("mean", 0.0))
        target["student_nearest_label_confusion"] = confusion.get(str(label), {})


def compute_strict_student_readability_acceptance(metrics: MetricsPayload) -> MetricsPayload:
    thresholds = dict(STRICT_STUDENT_READABILITY_THRESHOLDS)
    overall = cast(MetricsPayload, metrics.get("overall", {}))
    per_class = cast(dict[str, MetricsPayload], metrics.get("per_class", {}))

    per_class_dice = [
        float(class_metrics.get("paired_dice", 0.0))
        for class_metrics in per_class.values()
    ]
    per_class_label_correct = [
        float(class_metrics.get("student_label_correct_ratio", 0.0))
        for class_metrics in per_class.values()
    ]
    per_class_label_margin = [
        float(class_metrics.get("student_label_margin_mean", 0.0))
        for class_metrics in per_class.values()
    ]
    checks = {
        "paired_corr": float(overall.get("paired_corr", 0.0)) >= thresholds["paired_corr"],
        "paired_dice": float(overall.get("paired_dice", 0.0)) >= thresholds["paired_dice"],
        "student_high_saturation_ratio": float(overall.get("student_high_saturation_ratio", 1.0))
        <= thresholds["student_high_saturation_ratio"],
    }
    values = {
        "paired_corr": overall.get("paired_corr"),
        "paired_dice": overall.get("paired_dice"),
        "min_class_paired_dice": min(per_class_dice) if per_class_dice else None,
        "student_label_correct_ratio": overall.get("student_label_correct_ratio"),
        "min_class_student_label_correct_ratio": min(per_class_label_correct) if per_class_label_correct else None,
        "min_class_student_label_margin_mean": min(per_class_label_margin) if per_class_label_margin else None,
        "background_speckle_ratio": overall.get("background_speckle_ratio"),
        "foreground_hole_ratio": overall.get("foreground_hole_ratio"),
        "student_saturation_ratio": overall.get("student_saturation_ratio"),
        "student_high_saturation_ratio": overall.get("student_high_saturation_ratio"),
        "readable_contrast": overall.get("readable_contrast"),
    }
    return {
        "passed": all(checks.values()),
        "thresholds": thresholds,
        "checks": checks,
        "values": values,
    }


def compute_one_shot_distill_eval_metrics(
    torch_module,
    *,
    teacher_camera,
    student_camera,
    slm_amplitude,
    class_labels,
    foreground_threshold: float = 0.2,
) -> MetricsPayload:
    teacher = teacher_camera.detach().float().clamp(0.0, 1.0).cpu()
    student = student_camera.detach().float().clamp(0.0, 1.0).cpu()
    slm = slm_amplitude.detach().float().cpu()
    labels = [int(label) for label in class_labels.detach().cpu().tolist()]
    if teacher.shape != student.shape:
        raise ValueError(f"teacher_camera and student_camera must share shape, got {teacher.shape} != {student.shape}")
    if teacher.shape[0] != len(labels):
        raise ValueError("teacher_camera and class_labels must share batch size.")

    mse = float((student - teacher).pow(2).mean().item())
    corr = float(_eval_correlation(torch_module, student, teacher).item())
    dice = float(_eval_foreground_dice(student, teacher, threshold=foreground_threshold).item())
    saturation = float(_eval_camera_saturation_ratio(student).item())
    low_saturation = float(_eval_camera_low_saturation_ratio(student).item())
    high_saturation = float(_eval_camera_high_saturation_ratio(student).item())
    foreground_holes = float(_eval_foreground_hole_ratio(student, teacher, threshold=foreground_threshold).item())
    foreground_ratio = float(_eval_foreground_ratio(teacher, threshold=foreground_threshold).item())
    background_speckle = float(_eval_background_speckle_ratio(student, teacher, threshold=foreground_threshold).item())
    readable_contrast = float(_eval_readable_contrast(student, teacher, threshold=foreground_threshold).item())
    slm_tv = float(_eval_total_variation(slm).item())

    per_class: dict[str, dict[str, float | int | None]] = {}
    teacher_diversities: list[float] = []
    student_diversities: list[float] = []
    ratios: list[float] = []
    for label in sorted(set(labels)):
        indices = [index for index, item in enumerate(labels) if item == label]
        index_tensor = torch_module.tensor(indices, dtype=torch_module.long)
        teacher_class = teacher.index_select(0, index_tensor)
        student_class = student.index_select(0, index_tensor)
        slm_class = slm.index_select(0, index_tensor)
        teacher_pairwise_mse = _mean_pairwise_mse(torch_module, teacher_class)
        student_pairwise_mse = _mean_pairwise_mse(torch_module, student_class)
        teacher_pairwise_corr = _mean_pairwise_corr(torch_module, teacher_class)
        student_pairwise_corr = _mean_pairwise_corr(torch_module, student_class)
        diversity_ratio = None
        if teacher_pairwise_mse is not None and student_pairwise_mse is not None:
            diversity_ratio = student_pairwise_mse / max(teacher_pairwise_mse, 1.0e-8)
            teacher_diversities.append(teacher_pairwise_mse)
            student_diversities.append(student_pairwise_mse)
            ratios.append(diversity_ratio)
        per_class[str(label)] = {
            "count": len(indices),
            "paired_mse": float((student_class - teacher_class).pow(2).mean().item()),
            "paired_corr": float(_eval_correlation(torch_module, student_class, teacher_class).item()),
            "paired_dice": float(_eval_foreground_dice(student_class, teacher_class, threshold=foreground_threshold).item()),
            "student_saturation_ratio": float(_eval_camera_saturation_ratio(student_class).item()),
            "student_low_saturation_ratio": float(_eval_camera_low_saturation_ratio(student_class).item()),
            "student_high_saturation_ratio": float(_eval_camera_high_saturation_ratio(student_class).item()),
            "foreground_hole_ratio": float(
                _eval_foreground_hole_ratio(student_class, teacher_class, threshold=foreground_threshold).item()
            ),
            "foreground_ratio": float(_eval_foreground_ratio(teacher_class, threshold=foreground_threshold).item()),
            "background_speckle_ratio": float(
                _eval_background_speckle_ratio(student_class, teacher_class, threshold=foreground_threshold).item()
            ),
            "readable_contrast": float(
                _eval_readable_contrast(student_class, teacher_class, threshold=foreground_threshold).item()
            ),
            "slm_tv": float(_eval_total_variation(slm_class).item()),
            "teacher_pairwise_mse": teacher_pairwise_mse,
            "student_pairwise_mse": student_pairwise_mse,
            "diversity_ratio": diversity_ratio,
            "teacher_pairwise_corr": teacher_pairwise_corr,
            "student_pairwise_corr": student_pairwise_corr,
        }

    return {
        "foreground_threshold": foreground_threshold,
        "overall": {
            "count": len(labels),
            "paired_mse": mse,
            "paired_corr": corr,
            "paired_dice": dice,
            "student_saturation_ratio": saturation,
            "student_low_saturation_ratio": low_saturation,
            "student_high_saturation_ratio": high_saturation,
            "foreground_hole_ratio": foreground_holes,
            "foreground_ratio": foreground_ratio,
            "background_speckle_ratio": background_speckle,
            "readable_contrast": readable_contrast,
            "slm_tv": slm_tv,
            "teacher_pairwise_mse": sum(teacher_diversities) / len(teacher_diversities) if teacher_diversities else None,
            "student_pairwise_mse": sum(student_diversities) / len(student_diversities) if student_diversities else None,
            "diversity_ratio": sum(ratios) / len(ratios) if ratios else None,
        },
        "per_class": per_class,
    }
