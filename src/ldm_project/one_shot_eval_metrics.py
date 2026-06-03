from __future__ import annotations


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


def compute_one_shot_distill_eval_metrics(
    torch_module,
    *,
    teacher_camera,
    student_camera,
    slm_amplitude,
    class_labels,
    foreground_threshold: float = 0.2,
) -> dict[str, object]:
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
            "slm_tv": slm_tv,
            "teacher_pairwise_mse": sum(teacher_diversities) / len(teacher_diversities) if teacher_diversities else None,
            "student_pairwise_mse": sum(student_diversities) / len(student_diversities) if student_diversities else None,
            "diversity_ratio": sum(ratios) / len(ratios) if ratios else None,
        },
        "per_class": per_class,
    }
