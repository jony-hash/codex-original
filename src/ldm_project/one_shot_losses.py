from __future__ import annotations

import torch
import torch.nn.functional as F

from .training_loss_common import (
    _camera_artifact_metrics,
    _camera_high_saturation_ratio,
    _camera_low_saturation_ratio,
    _camera_saturation_loss,
    _camera_saturation_ratio,
    _camera_topology_loss,
    _foreground_dice_score,
    _foreground_shape_loss_for_samples,
    _masked_mean,
    _masked_mean_per_sample,
    _sobel_magnitude,
    _total_variation,
    _weighted_reconstruction_metrics,
    _weighted_reconstruction_metrics_for_samples,
    _weighted_sample_mean,
    _zero_like_loss,
)


def _build_clean_readable_target(teacher_camera: torch.Tensor) -> dict[str, torch.Tensor]:
    """Derive a clean readable soft mask from the noisy teacher_camera.

    Returns a dict with:
        clean_target: soft mask in [0, 1], background near 0, foreground near 1.
        clean_target_fg_ratio: fraction of pixels above 0.5 (detached scalar).
    """
    B, C, H, W = teacher_camera.shape
    flat = teacher_camera.reshape(B, C, -1)
    q_low = flat.quantile(0.02, dim=-1, keepdim=True)
    q_high = flat.quantile(0.98, dim=-1, keepdim=True)
    normalized = (teacher_camera - q_low.view(B, C, 1, 1)) / (
        (q_high - q_low).view(B, C, 1, 1).clamp_min(1e-6)
    )
    # Small Gaussian blur (~3px sigma) to soften noise without destroying edges.
    sigma = 3.0
    ksize = max(3, int(round(sigma * 3)) // 2 * 2 + 1)
    g = torch.arange(-ksize // 2 + 1, ksize // 2 + 1, device=teacher_camera.device,
                     dtype=teacher_camera.dtype)
    g = torch.exp(-g.pow(2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    g_row = g.view(1, 1, 1, -1)
    g_col = g.view(1, 1, -1, 1)
    clamped = normalized.clamp(0.0, 1.0)
    padded = F.pad(clamped, (ksize // 2, ksize // 2, ksize // 2, ksize // 2), mode="replicate")
    blurred = F.conv2d(padded, g_row.expand(C, 1, 1, -1), groups=C)
    blurred = F.conv2d(blurred, g_col.expand(C, 1, -1, 1), groups=C)
    # Soft sigmoid mask: threshold at 0.30 with steepness 10.
    soft_threshold = 0.30
    mask = torch.sigmoid((blurred - soft_threshold) * 10.0)
    # Morphology close (fill small holes): dilation → erosion.
    close_k = max(3, int(round(0.04 * min(H, W)) // 2 * 2 + 1))  # ~7px at 200
    # erosion of dilated: -max_pool2d(-max_pool2d(mask))
    closed = -F.max_pool2d(
        -F.max_pool2d(mask, kernel_size=close_k, stride=1, padding=close_k // 2),
        kernel_size=close_k, stride=1, padding=close_k // 2,
    )
    # Morphology open (clean tiny speckles): erosion → dilation.
    open_k = max(3, int(round(0.03 * min(H, W)) // 2 * 2 + 1))  # ~5px at 200
    opened = F.max_pool2d(
        -F.max_pool2d(-closed, kernel_size=open_k, stride=1, padding=open_k // 2),
        kernel_size=open_k, stride=1, padding=open_k // 2,
    )
    fg_ratio = (opened > 0.5).float().mean().detach()
    return {
        "clean_target": opened,
        "clean_target_fg_ratio": fg_ratio,
    }


def _gaussian_blur_2d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return tensor
    channels = tensor.shape[1]
    radius = max(1, int(round(float(sigma) * 3.0)))
    coords = torch.arange(-radius, radius + 1, device=tensor.device, dtype=tensor.dtype)
    kernel_1d = torch.exp(-(coords.pow(2)) / (2.0 * float(sigma) * float(sigma)))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1.0e-12)
    kernel_y = kernel_1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    kernel_x = kernel_1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    padded = F.pad(tensor, (radius, radius, radius, radius), mode="replicate")
    blurred = F.conv2d(padded, kernel_y, groups=channels)
    return F.conv2d(blurred, kernel_x, groups=channels)


def transform_aux_topology_target(
    target: torch.Tensor,
    transform: str = "none",
    foreground_threshold: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Return a diagnostic-only target variant for aux topology supervision."""
    if transform == "none":
        transformed = target.float().clamp(0.0, 1.0)
    elif transform == "lowpass":
        transformed = _gaussian_blur_2d(target.float().clamp(0.0, 1.0), sigma=3.0).clamp(0.0, 1.0)
    elif transform == "thick_binary":
        base = (target.float() > foreground_threshold).float()
        thick = F.max_pool2d(base, kernel_size=9, stride=1, padding=4)
        transformed = _gaussian_blur_2d(thick, sigma=1.2).clamp(0.0, 1.0)
    else:
        raise ValueError(
            "aux_topology_target_transform must be one of ['none', 'lowpass', 'thick_binary'], "
            f"got {transform!r}"
        )
    foreground_mask = (transformed > foreground_threshold).float()
    return {
        "target": transformed,
        "foreground_mask": foreground_mask,
        "foreground_ratio": foreground_mask.mean().detach(),
    }


def _one_shot_pairwise_diversity_metrics(
    camera_prediction: torch.Tensor,
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    min_teacher_mse: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or camera_prediction.shape[0] < 2:
        return {
            "code_diversity_base": zero,
            "teacher_pairwise_diversity_base": zero,
            "teacher_pairwise_mse": zero,
            "student_pairwise_mse": zero,
            "diversity_ratio": zero,
            "diversity_pair_count": zero,
        }

    labels = class_labels.detach().reshape(-1)
    if labels.shape[0] != camera_prediction.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size when diversity weights are enabled.")

    prediction_flat = camera_prediction.flatten(1)
    target_flat = target_batch.detach().flatten(1)
    student_distances = []
    teacher_distances = []
    for label in labels.unique(sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        student_class = prediction_flat.index_select(0, indices)
        teacher_class = target_flat.index_select(0, indices)
        student_pairwise = (student_class[:, None, :] - student_class[None, :, :]).pow(2).mean(dim=-1)
        teacher_pairwise = (teacher_class[:, None, :] - teacher_class[None, :, :]).pow(2).mean(dim=-1)
        pair_mask = torch.triu(
            torch.ones(indices.numel(), indices.numel(), dtype=torch.bool, device=camera_prediction.device),
            diagonal=1,
        )
        student_distances.append(student_pairwise[pair_mask])
        teacher_distances.append(teacher_pairwise[pair_mask])

    if not student_distances:
        return {
            "code_diversity_base": zero,
            "teacher_pairwise_diversity_base": zero,
            "teacher_pairwise_mse": zero,
            "student_pairwise_mse": zero,
            "diversity_ratio": zero,
            "diversity_pair_count": zero,
        }

    student_distance = torch.cat(student_distances)
    teacher_distance = torch.cat(teacher_distances)
    active = teacher_distance >= float(min_teacher_mse)
    if not active.any():
        return {
            "code_diversity_base": zero,
            "teacher_pairwise_diversity_base": zero,
            "teacher_pairwise_mse": teacher_distance.mean().detach(),
            "student_pairwise_mse": student_distance.mean().detach(),
            "diversity_ratio": zero,
            "diversity_pair_count": zero,
        }

    active_student = student_distance[active]
    active_teacher = teacher_distance[active].clamp_min(1.0e-8)
    # 只在 teacher 自身存在差异的同类样本对上惩罚塌缩，避免鼓励无意义噪声。
    diversity_ratio = active_student / active_teacher
    code_diversity_base = F.relu(1.0 - diversity_ratio).pow(2).mean()
    teacher_pairwise_base = F.smooth_l1_loss(
        torch.log(active_student.clamp_min(1.0e-8)),
        torch.log(active_teacher),
    )
    return {
        "code_diversity_base": code_diversity_base,
        "teacher_pairwise_diversity_base": teacher_pairwise_base,
        "teacher_pairwise_mse": active_teacher.mean().detach(),
        "student_pairwise_mse": active_student.mean().detach(),
        "diversity_ratio": diversity_ratio.mean().detach(),
        "diversity_pair_count": active.float().sum().detach(),
    }


def _encoder_input_topology_metrics(
    outputs: dict[str, torch.Tensor],
    target: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    planes = []
    for key in ("encoder_input_amplitude", "encoder_input_grid_amplitude"):
        plane = outputs.get(key)
        if plane is None:
            continue
        resized = plane.float()
        if resized.shape[-2:] != target.shape[-2:]:
            resized = F.interpolate(resized, size=target.shape[-2:], mode="bilinear", align_corners=False)
        planes.append(resized.clamp(0.0, 1.0))
    if not planes:
        raise ValueError(
            "encoder_input_topology_weight requires encoder_input_amplitude "
            "or encoder_input_grid_amplitude."
        )

    dice_scores = [_foreground_dice_score(plane, foreground_mask) for plane in planes]
    mse_scores = [torch.mean((plane - target.float()).pow(2)) for plane in planes]
    dice = torch.stack(dice_scores).mean()
    mse = torch.stack(mse_scores).mean()
    return {
        "dice": dice,
        "dice_penalty": 1.0 - dice,
        "mse": mse,
    }


def _one_shot_cross_class_pairwise_metrics(
    camera_prediction: torch.Tensor,
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    min_teacher_mse: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or camera_prediction.shape[0] < 2:
        return {
            "cross_class_pairwise_base": zero,
            "cross_class_teacher_pairwise_mse": zero,
            "cross_class_student_pairwise_mse": zero,
            "cross_class_diversity_ratio": zero,
            "cross_class_pair_count": zero,
        }

    labels = class_labels.detach().reshape(-1)
    if labels.shape[0] != camera_prediction.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size when cross-class diversity is enabled.")

    prediction_flat = camera_prediction.flatten(1)
    target_flat = target_batch.detach().flatten(1)
    student_pairwise = (prediction_flat[:, None, :] - prediction_flat[None, :, :]).pow(2).mean(dim=-1)
    teacher_pairwise = (target_flat[:, None, :] - target_flat[None, :, :]).pow(2).mean(dim=-1)
    different_class = labels[:, None] != labels[None, :]
    pair_mask = torch.triu(
        torch.ones(labels.numel(), labels.numel(), dtype=torch.bool, device=camera_prediction.device),
        diagonal=1,
    ) & different_class
    if not pair_mask.any():
        return {
            "cross_class_pairwise_base": zero,
            "cross_class_teacher_pairwise_mse": zero,
            "cross_class_student_pairwise_mse": zero,
            "cross_class_diversity_ratio": zero,
            "cross_class_pair_count": zero,
        }

    student_distance = student_pairwise[pair_mask]
    teacher_distance = teacher_pairwise[pair_mask]
    active = teacher_distance >= float(min_teacher_mse)
    if not active.any():
        return {
            "cross_class_pairwise_base": zero,
            "cross_class_teacher_pairwise_mse": teacher_distance.mean().detach(),
            "cross_class_student_pairwise_mse": student_distance.mean().detach(),
            "cross_class_diversity_ratio": zero,
            "cross_class_pair_count": zero,
        }

    active_student = student_distance[active]
    active_teacher = teacher_distance[active].clamp_min(1.0e-8)
    pairwise_base = F.smooth_l1_loss(
        torch.log(active_student.clamp_min(1.0e-8)),
        torch.log(active_teacher),
    )
    return {
        "cross_class_pairwise_base": pairwise_base,
        "cross_class_teacher_pairwise_mse": active_teacher.mean().detach(),
        "cross_class_student_pairwise_mse": active_student.mean().detach(),
        "cross_class_diversity_ratio": (active_student / active_teacher).mean().detach(),
        "cross_class_pair_count": active.float().sum().detach(),
    }


def _pairwise_mse_summary(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] < 2:
        return _zero_like_loss(tensor)
    flattened = tensor.detach().float().flatten(1)
    pairwise = (flattened[:, None, :] - flattened[None, :, :]).pow(2).mean(dim=-1)
    pair_mask = torch.triu(
        torch.ones(tensor.shape[0], tensor.shape[0], dtype=torch.bool, device=tensor.device),
        diagonal=1,
    )
    return pairwise[pair_mask].mean()


def _per_class_std_summary(tensor: torch.Tensor, class_labels: torch.Tensor | None) -> torch.Tensor:
    if class_labels is None:
        return _zero_like_loss(tensor)
    labels = class_labels.detach().reshape(-1)
    if labels.shape[0] != tensor.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size for SLM diagnostics.")
    flattened = tensor.detach().float().flatten(1)
    class_stds = []
    for label in labels.unique(sorted=True):
        indices = torch.nonzero(labels == label, as_tuple=False).flatten()
        if indices.numel() < 2:
            continue
        class_stds.append(flattened.index_select(0, indices).std(dim=0, unbiased=False).mean())
    if not class_stds:
        return _zero_like_loss(tensor)
    return torch.stack(class_stds).mean()


def _camera_readability_metrics(
    camera_prediction: torch.Tensor,
    target_batch: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Self-supervised readability regularizer: penalize speckle relative to smoothed teacher."""
    if camera_prediction.shape[0] == 0:
        zero = _zero_like_loss(camera_prediction)
        return {"readability_base": zero, "readability_penalty": zero}
    kernel_size = max(3, int(0.8 * 6) | 1)
    pad = kernel_size // 2
    g = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, device=camera_prediction.device, dtype=camera_prediction.dtype)
    g = torch.exp(-g.pow(2) / (2.0 * 0.8 * 0.8))
    g = g / g.sum()
    g_row = g.view(1, 1, 1, -1)
    g_col = g.view(1, 1, -1, 1)
    target_padded = F.pad(target_batch, (pad, pad, pad, pad), mode="replicate")
    teacher_smoothed = F.conv2d(target_padded, g_row.expand(target_batch.shape[1], 1, 1, -1), groups=target_batch.shape[1])
    teacher_smoothed = F.conv2d(teacher_smoothed, g_col.expand(target_batch.shape[1], 1, -1, 1), groups=target_batch.shape[1])
    squared_diff = (camera_prediction - teacher_smoothed.detach()).pow(2)
    fg_weight = 3.0
    per_sample_mse = (squared_diff.flatten(1) * (foreground_mask.flatten(1) * (fg_weight - 1.0) + 1.0)).mean(dim=1)
    readability_base = per_sample_mse.mean()
    return {"readability_base": readability_base, "readability_penalty": readability_base.detach()}


def _camera_background_speckle_metrics(
    camera_prediction: torch.Tensor,
    background_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if camera_prediction.shape[0] == 0:
        return {"background_speckle_base": zero, "background_speckle_ratio": zero}
    prediction = camera_prediction.clamp_min(0.0)
    bg_pixels = background_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    bg_mean = (prediction.flatten(1) * background_mask.flatten(1)).sum(dim=1) / bg_pixels
    bg_centered = (prediction - bg_mean.view(-1, 1, 1, 1)) * background_mask
    bg_std = (bg_centered.flatten(1).pow(2).sum(dim=1) / bg_pixels).sqrt()
    base = (bg_mean + 0.5 * bg_std).mean()
    ratio = ((prediction > 0.12).float() * background_mask).flatten(1).sum(dim=1) / bg_pixels
    return {"background_speckle_base": base, "background_speckle_ratio": ratio.mean().detach()}


def _camera_background_hotspot_metrics(
    camera_prediction: torch.Tensor,
    background_mask: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    prediction = camera_prediction.clamp(0.0, 1.0)
    hotspot = F.relu(prediction - float(threshold))
    base = _masked_mean_per_sample(hotspot.pow(2), background_mask).mean()
    ratio = _masked_mean_per_sample((prediction > float(threshold)).float(), background_mask).mean()
    return {
        "background_hotspot_base": base,
        "background_hotspot_ratio": ratio.detach(),
    }


def _camera_background_highfreq_metrics(
    camera_prediction: torch.Tensor,
    background_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    prediction = camera_prediction.clamp(0.0, 1.0)
    smooth = F.avg_pool2d(prediction, kernel_size=5, stride=1, padding=2)
    highfreq = (prediction - smooth).pow(2)
    base = _masked_mean_per_sample(highfreq, background_mask).mean()
    return {
        "background_highfreq_base": base,
        "background_highfreq_energy": base.detach(),
    }


def _camera_readable_contrast_metrics(
    camera_prediction: torch.Tensor,
    foreground_mask: torch.Tensor,
    background_mask: torch.Tensor,
    target: float,
) -> dict[str, torch.Tensor]:
    prediction = camera_prediction.clamp_min(0.0)
    fg_mean = _masked_mean_per_sample(prediction, foreground_mask)
    bg_mean = _masked_mean_per_sample(prediction, background_mask)
    contrast = fg_mean / bg_mean.clamp_min(1.0e-6)
    base = F.relu(float(target) - contrast).mean()
    return {
        "readable_contrast_base": base,
        "readable_contrast": contrast.mean().detach(),
    }


def _normalize_for_label_margin(tensor: torch.Tensor) -> torch.Tensor:
    flattened = tensor.float().flatten(1)
    centered = flattened - flattened.mean(dim=1, keepdim=True)
    normalized = centered / centered.std(dim=1, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    return F.normalize(normalized, dim=1, eps=1.0e-6)


def _student_label_margin_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    label_subset: tuple[int, ...],
    target: float,
    focus_label: int | None,
    focus_weight: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or prototypes is None:
        return {
            "student_label_margin_base": zero,
            "student_label_margin_mean": zero,
            "student_label_correct_ratio": zero,
        }
    labels, correlations = _student_prototype_correlations(camera_prediction, class_labels, prototypes)

    valid_mask = torch.ones_like(labels, dtype=torch.bool)
    if label_subset:
        valid_mask = torch.zeros_like(labels, dtype=torch.bool)
        for label in label_subset:
            valid_mask |= labels == int(label)
    if not valid_mask.any():
        return {
            "student_label_margin_base": zero,
            "student_label_margin_mean": zero,
            "student_label_correct_ratio": zero,
        }

    target_corr = correlations.gather(1, labels.view(-1, 1)).squeeze(1)
    other_corr = correlations.clone()
    other_corr.scatter_(1, labels.view(-1, 1), -2.0)
    best_other = other_corr.max(dim=1).values
    margins = target_corr - best_other
    active_margins = margins[valid_mask]
    active_labels = labels[valid_mask]
    nearest_labels = correlations.argmax(dim=1)
    active_correct = (nearest_labels[valid_mask] == labels[valid_mask]).float()
    penalties = F.relu(float(target) - active_margins)
    if focus_label is not None and float(focus_weight) != 1.0:
        weights = torch.where(
            active_labels == int(focus_label),
            penalties.new_full(penalties.shape, float(focus_weight)),
            torch.ones_like(penalties),
        )
        base = (penalties * weights).sum() / weights.sum().clamp_min(1.0e-6)
    else:
        base = penalties.mean()
    return {
        "student_label_margin_base": base,
        "student_label_margin_mean": active_margins.mean().detach(),
        "student_label_correct_ratio": active_correct.mean().detach(),
    }


def _student_prototype_correlations(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = class_labels.detach().reshape(-1).to(device=camera_prediction.device)
    if labels.shape[0] != camera_prediction.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size for student prototype losses.")
    if prototypes.ndim != 4 or prototypes.shape[1] != camera_prediction.shape[1]:
        raise ValueError(
            "student_label_margin_prototypes must be [num_classes, channels, height, width] "
            f"with matching channels, got {tuple(prototypes.shape)}"
        )
    prototype_batch = prototypes.to(device=camera_prediction.device, dtype=camera_prediction.dtype)
    if prototype_batch.shape[-2:] != camera_prediction.shape[-2:]:
        prototype_batch = F.interpolate(
            prototype_batch,
            size=camera_prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if int(labels.max().item()) >= prototype_batch.shape[0] or int(labels.min().item()) < 0:
        raise ValueError("class_labels exceed student_label_margin_prototypes class dimension.")

    sample_features = _normalize_for_label_margin(camera_prediction)
    prototype_features = _normalize_for_label_margin(prototype_batch)
    correlations = sample_features @ prototype_features.t()
    return labels, correlations


def _student_roi_prototype_correlations(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor,
    prototypes: torch.Tensor,
    foreground_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = class_labels.detach().reshape(-1).to(device=camera_prediction.device)
    if labels.shape[0] != camera_prediction.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size for student ROI prototype losses.")
    if prototypes.ndim != 4 or prototypes.shape[1] != camera_prediction.shape[1]:
        raise ValueError(
            "student_label_margin_prototypes must be [num_classes, channels, height, width] "
            f"with matching channels, got {tuple(prototypes.shape)}"
        )
    prototype_batch = prototypes.to(device=camera_prediction.device, dtype=camera_prediction.dtype)
    if prototype_batch.shape[-2:] != camera_prediction.shape[-2:]:
        prototype_batch = F.interpolate(
            prototype_batch,
            size=camera_prediction.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if int(labels.max().item()) >= prototype_batch.shape[0] or int(labels.min().item()) < 0:
        raise ValueError("class_labels exceed student_label_margin_prototypes class dimension.")

    sample_flat = camera_prediction.float().flatten(1)
    prototype_flat = prototype_batch.float().flatten(1)
    roi_source = prototype_batch.index_select(0, labels).float()
    roi_mask = (roi_source > float(foreground_threshold)).float().flatten(1)
    mask_sum = roi_mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    sample_mean = (sample_flat * roi_mask).sum(dim=1, keepdim=True) / mask_sum
    sample_centered = (sample_flat - sample_mean) * roi_mask
    prototype_mean = (
        prototype_flat.unsqueeze(0) * roi_mask.unsqueeze(1)
    ).sum(dim=2) / mask_sum
    prototype_centered = (prototype_flat.unsqueeze(0) - prototype_mean.unsqueeze(2)) * roi_mask.unsqueeze(1)

    numerator = (sample_centered.unsqueeze(1) * prototype_centered).sum(dim=2)
    sample_norm = sample_centered.pow(2).sum(dim=1, keepdim=True).sqrt()
    prototype_norm = prototype_centered.pow(2).sum(dim=2).sqrt()
    correlations = numerator / (sample_norm * prototype_norm).clamp_min(1.0e-6)
    return labels, correlations


def _student_roi_prototype_ce_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    label_subset: tuple[int, ...],
    temperature: float,
    foreground_threshold: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or prototypes is None:
        return {
            "student_roi_prototype_ce_base": zero,
            "student_roi_prototype_ce_correct_ratio": zero,
            "student_roi_prototype_ce_count": zero,
        }
    labels, correlations = _student_roi_prototype_correlations(
        camera_prediction,
        class_labels,
        prototypes,
        foreground_threshold,
    )
    valid_mask = torch.ones_like(labels, dtype=torch.bool)
    if label_subset:
        valid_mask = torch.zeros_like(labels, dtype=torch.bool)
        for label in label_subset:
            valid_mask |= labels == int(label)
    if not valid_mask.any():
        return {
            "student_roi_prototype_ce_base": zero,
            "student_roi_prototype_ce_correct_ratio": zero,
            "student_roi_prototype_ce_count": zero,
        }
    logits = correlations[valid_mask] / float(temperature)
    active_labels = labels[valid_mask]
    predictions = logits.argmax(dim=1)
    return {
        "student_roi_prototype_ce_base": F.cross_entropy(logits, active_labels),
        "student_roi_prototype_ce_correct_ratio": (predictions == active_labels).float().mean().detach(),
        "student_roi_prototype_ce_count": torch.tensor(float(active_labels.numel()), device=camera_prediction.device),
    }


def _student_roi_negative_label_margin_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    margin_config: tuple[tuple[int, int, float], ...],
    foreground_threshold: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or prototypes is None or not margin_config:
        return {
            "student_roi_negative_label_margin_base": zero,
            "student_roi_negative_label_margin_mean": zero,
            "student_roi_negative_label_margin_count": zero,
        }
    labels, correlations = _student_roi_prototype_correlations(
        camera_prediction,
        class_labels,
        prototypes,
        foreground_threshold,
    )
    penalties = []
    margins = []
    for target_class, negative_class, target_margin in margin_config:
        active = labels == int(target_class)
        if not active.any():
            continue
        target_corr = correlations[active, int(target_class)]
        negative_corr = correlations[active, int(negative_class)]
        pair_margins = target_corr - negative_corr
        margins.append(pair_margins)
        penalties.append(F.relu(float(target_margin) - pair_margins))
    if not penalties:
        return {
            "student_roi_negative_label_margin_base": zero,
            "student_roi_negative_label_margin_mean": zero,
            "student_roi_negative_label_margin_count": zero,
        }
    penalty_tensor = torch.cat(penalties)
    margin_tensor = torch.cat(margins)
    return {
        "student_roi_negative_label_margin_base": penalty_tensor.mean(),
        "student_roi_negative_label_margin_mean": margin_tensor.mean().detach(),
        "student_roi_negative_label_margin_count": torch.tensor(
            float(penalty_tensor.numel()),
            device=camera_prediction.device,
        ),
    }


def _student_prototype_ce_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    label_subset: tuple[int, ...],
    temperature: float,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or prototypes is None:
        return {
            "student_prototype_ce_base": zero,
            "student_prototype_ce_correct_ratio": zero,
            "student_prototype_ce_count": zero,
        }
    labels, correlations = _student_prototype_correlations(camera_prediction, class_labels, prototypes)
    valid_mask = torch.ones_like(labels, dtype=torch.bool)
    if label_subset:
        valid_mask = torch.zeros_like(labels, dtype=torch.bool)
        for label in label_subset:
            valid_mask |= labels == int(label)
    if not valid_mask.any():
        return {
            "student_prototype_ce_base": zero,
            "student_prototype_ce_correct_ratio": zero,
            "student_prototype_ce_count": zero,
        }
    logits = correlations[valid_mask] / float(temperature)
    active_labels = labels[valid_mask]
    predictions = logits.argmax(dim=1)
    return {
        "student_prototype_ce_base": F.cross_entropy(logits, active_labels),
        "student_prototype_ce_correct_ratio": (predictions == active_labels).float().mean().detach(),
        "student_prototype_ce_count": torch.tensor(float(active_labels.numel()), device=camera_prediction.device),
    }


def _student_negative_label_margin_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    margin_config: tuple[tuple[int, int, float], ...],
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or prototypes is None or not margin_config:
        return {
            "student_negative_label_margin_base": zero,
            "student_negative_label_margin_mean": zero,
            "student_negative_label_margin_count": zero,
        }
    labels, correlations = _student_prototype_correlations(camera_prediction, class_labels, prototypes)
    penalties = []
    margins = []
    for target_class, negative_class, target_margin in margin_config:
        active = labels == int(target_class)
        if not active.any():
            continue
        target_corr = correlations[active, int(target_class)]
        negative_corr = correlations[active, int(negative_class)]
        pair_margins = target_corr - negative_corr
        margins.append(pair_margins)
        penalties.append(F.relu(float(target_margin) - pair_margins))
    if not penalties:
        return {
            "student_negative_label_margin_base": zero,
            "student_negative_label_margin_mean": zero,
            "student_negative_label_margin_count": zero,
        }
    penalty_tensor = torch.cat(penalties)
    margin_tensor = torch.cat(margins)
    return {
        "student_negative_label_margin_base": penalty_tensor.mean(),
        "student_negative_label_margin_mean": margin_tensor.mean().detach(),
        "student_negative_label_margin_count": torch.tensor(float(penalty_tensor.numel()), device=camera_prediction.device),
    }


def _resize_aux_semantic_plane(
    outputs: dict[str, torch.Tensor],
    plane_key: str,
    target_batch: torch.Tensor,
) -> torch.Tensor:
    if not plane_key:
        raise ValueError("aux_semantic_plane_key is required when aux semantic losses are enabled.")
    plane = outputs.get(plane_key)
    if not torch.is_tensor(plane):
        raise ValueError(f"aux_semantic_plane_key={plane_key!r} was not found in one-shot outputs.")
    plane = plane.float()
    if plane.ndim != target_batch.ndim:
        raise ValueError(
            "aux semantic plane rank must match target_batch rank, "
            f"got {tuple(plane.shape)} and {tuple(target_batch.shape)}"
        )
    if plane.shape[0] != target_batch.shape[0] or plane.shape[1] != target_batch.shape[1]:
        raise ValueError(
            "aux semantic plane batch/channel must match target_batch, "
            f"got {tuple(plane.shape)} and {tuple(target_batch.shape)}"
        )
    if plane.shape[-2:] != target_batch.shape[-2:]:
        plane = F.interpolate(plane, size=target_batch.shape[-2:], mode="bilinear", align_corners=False)
    return plane.clamp(0.0, 1.0)


def _aux_topology_metrics(
    outputs: dict[str, torch.Tensor],
    plane_key: str,
    target: torch.Tensor,
    foreground_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not plane_key:
        raise ValueError("aux_topology_plane_key is required when aux topology loss is enabled.")
    plane = outputs.get(plane_key)
    if not torch.is_tensor(plane):
        raise ValueError(f"aux_topology_plane_key={plane_key!r} was not found in one-shot outputs.")
    resized = plane.float()
    if resized.ndim != target.ndim:
        raise ValueError(
            "aux topology plane rank must match target_batch rank, "
            f"got {tuple(resized.shape)} and {tuple(target.shape)}"
        )
    if resized.shape[0] != target.shape[0] or resized.shape[1] != target.shape[1]:
        raise ValueError(
            "aux topology plane batch/channel must match target_batch, "
            f"got {tuple(resized.shape)} and {tuple(target.shape)}"
        )
    if resized.shape[-2:] != target.shape[-2:]:
        resized = F.interpolate(resized, size=target.shape[-2:], mode="bilinear", align_corners=False)
    resized = resized.clamp(0.0, 1.0)
    dice = _foreground_dice_score(resized, foreground_mask)
    mse = torch.mean((resized - target.float()).pow(2))
    return {
        "dice": dice,
        "dice_penalty": 1.0 - dice,
        "mse": mse,
    }


def _student_pair_contrast_metrics(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    pair_config: tuple[tuple[int, int, float], ...],
) -> dict[str, torch.Tensor]:
    """Penalize confusion pairs: for each (class_a, class_b, target_margin),
    push the per-class mean student features apart if below target_margin.

    Operates on per-class mean features (one prototype per class per batch),
    measuring cosine-distance between class means.
    """
    zero = _zero_like_loss(camera_prediction)
    if class_labels is None or not pair_config:
        return {"student_pair_contrast_base": zero, "student_pair_contrast_count": zero}

    labels = class_labels.detach().reshape(-1)
    if labels.shape[0] != camera_prediction.shape[0]:
        return {"student_pair_contrast_base": zero, "student_pair_contrast_count": zero}

    features = camera_prediction.float().flatten(1)
    # Compute per-class mean features
    unique_labels = labels.unique(sorted=True)
    class_means: dict[int, torch.Tensor] = {}
    for lab in unique_labels.tolist():
        mask = (labels == int(lab))
        if mask.any():
            class_means[int(lab)] = features[mask].mean(dim=0)
            class_means[int(lab)] = F.normalize(class_means[int(lab)].unsqueeze(0), dim=1).squeeze(0)

    total_penalty = zero
    active_pairs = 0
    for class_a, class_b, target_margin in pair_config:
        if class_a not in class_means or class_b not in class_means:
            continue
        feat_a = class_means[class_a]
        feat_b = class_means[class_b]
        # Cosine similarity: high similarity = low margin.
        similarity = (feat_a * feat_b).sum()
        margin_penalty = F.relu(float(target_margin) - (1.0 - similarity)).mean()
        total_penalty = total_penalty + margin_penalty
        active_pairs += 1

    if active_pairs == 0:
        return {"student_pair_contrast_base": zero, "student_pair_contrast_count": zero}

    return {
        "student_pair_contrast_base": total_penalty / float(active_pairs),
        "student_pair_contrast_count": torch.tensor(float(active_pairs), device=camera_prediction.device),
    }


def _student_display_speckle_metrics(
    camera_prediction: torch.Tensor,
    foreground_mask: torch.Tensor,
    background_mask: torch.Tensor,
    speckle_brightness_threshold: float = 0.12,
) -> dict[str, torch.Tensor]:
    """Per-sample quantile-normalize camera prediction, then penalize background
    speckle (bright pixels + high-frequency energy) in the normalized domain.

    This simulates a display/stretching step and penalizes what a human would
    see as background noise.
    """
    B, C = camera_prediction.shape[:2]
    flat = camera_prediction.detach().float().reshape(B, C, -1)
    q01 = flat.quantile(0.01, dim=-1, keepdim=True)
    q99 = flat.quantile(0.99, dim=-1, keepdim=True)
    normalized = (camera_prediction.float() - q01.view(B, C, 1, 1)) / (
        (q99 - q01).view(B, C, 1, 1).clamp_min(1e-6)
    )
    clamped = normalized.clamp(0.0, 1.0)
    # Speckle ratio: fraction of background pixels above threshold in normalized space.
    bright_bg = (clamped > float(speckle_brightness_threshold)).float() * background_mask
    speckle_ratio = bright_bg.flatten(1).sum(dim=1) / background_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    speckle_ratio_penalty = F.relu(speckle_ratio - 0.12).mean()
    bg_energy = _masked_mean_per_sample(clamped, background_mask).mean()
    # High-frequency energy in background of normalized space.
    smooth = F.avg_pool2d(clamped, kernel_size=5, stride=1, padding=2)
    highfreq = (clamped - smooth).pow(2)
    highfreq_base = _masked_mean_per_sample(highfreq, background_mask).mean()
    return {
        "display_speckle_base": bg_energy + speckle_ratio_penalty + 0.3 * highfreq_base,
        "display_speckle_ratio": speckle_ratio.mean().detach(),
        "display_highfreq_energy": highfreq_base.detach(),
    }


def _camera_hotspot_ratio(camera_prediction: torch.Tensor) -> torch.Tensor:
    return (camera_prediction > 1.0).float().mean()


def _resolve_sample_weights(
    reference: torch.Tensor,
    class_labels: torch.Tensor | None,
    class_weight_labels: tuple[int, ...],
    class_weight: float,
) -> torch.Tensor:
    weights = torch.ones(reference.shape[0], device=reference.device, dtype=reference.dtype)
    if class_labels is None or not class_weight_labels or class_weight == 1.0:
        return weights
    labels = class_labels.detach().reshape(-1).to(device=reference.device)
    if labels.shape[0] != reference.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size when class weighting is enabled.")
    weak_mask = torch.zeros_like(labels, dtype=torch.bool)
    for label in class_weight_labels:
        weak_mask |= labels == int(label)
    return torch.where(weak_mask, weights * float(class_weight), weights)


def _one_shot_edge_shape_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    foreground_mask: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    prediction_soft = prediction.clamp(0.0, 1.0)
    target_soft = target.clamp(0.0, 1.0)
    foreground_context = F.max_pool2d(foreground_mask, kernel_size=5, stride=1, padding=2)
    edge_error = (_sobel_magnitude(prediction_soft) - _sobel_magnitude(target_soft)).abs()
    local_prediction = F.avg_pool2d(prediction_soft, kernel_size=3, stride=1, padding=1)
    local_target = F.avg_pool2d(target_soft, kernel_size=3, stride=1, padding=1)
    local_error = (local_prediction - local_target).abs()
    edge_samples = _masked_mean_per_sample(edge_error, foreground_context)
    local_samples = _masked_mean_per_sample(local_error, foreground_context)
    return _weighted_sample_mean(edge_samples + 0.5 * local_samples, sample_weights)


def _dilate_foreground_mask(foreground_mask: torch.Tensor, dilation: int) -> torch.Tensor:
    dilation = int(dilation)
    if dilation <= 0:
        return foreground_mask.float()
    if dilation % 2 == 0:
        dilation += 1
    return F.max_pool2d(
        foreground_mask.float(),
        kernel_size=dilation,
        stride=1,
        padding=dilation // 2,
    ).clamp(0.0, 1.0)


def _resolve_clean_target_masks(
    target_batch: torch.Tensor,
    foreground_mask: torch.Tensor,
    background_mask: torch.Tensor,
    *,
    enabled: bool,
    zero: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not enabled:
        return target_batch, foreground_mask, background_mask, zero
    clean_result = _build_clean_readable_target(target_batch.detach())
    clean_target = clean_result["clean_target"]
    clean_foreground_mask = clean_target.float()
    return (
        clean_target,
        clean_foreground_mask,
        1.0 - clean_foreground_mask,
        clean_result["clean_target_fg_ratio"],
    )


def _clean_target_regularization_terms(
    camera_prediction: torch.Tensor,
    camera_loss_prediction: torch.Tensor,
    clean_target: torch.Tensor,
    clean_foreground_mask: torch.Tensor,
    clean_background_mask: torch.Tensor,
    *,
    enabled: bool,
    foreground_dice_weight: float,
    foreground_hole_weight: float,
    background_speckle_weight: float,
    background_hotspot_weight: float,
    background_highfreq_weight: float,
    background_hotspot_threshold: float,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not enabled:
        return {
            "dice_base": zero,
            "dice_loss": zero,
            "foreground_hole_base": zero,
            "foreground_hole_loss": zero,
            "foreground_hole_ratio": zero,
            "background_speckle_base": zero,
            "background_speckle_loss": zero,
            "background_speckle_ratio": zero,
            "background_hotspot_base": zero,
            "background_hotspot_loss": zero,
            "background_hotspot_ratio": zero,
            "background_highfreq_base": zero,
            "background_highfreq_loss": zero,
            "background_highfreq_energy": zero,
        }
    dice_base = 1.0 - _foreground_dice_score(camera_prediction, clean_foreground_mask)
    hole_metrics = _camera_artifact_metrics(
        camera_loss_prediction,
        clean_target,
        clean_foreground_mask,
    )
    speckle_metrics = _camera_background_speckle_metrics(
        camera_loss_prediction,
        clean_background_mask,
    )
    hotspot_metrics = _camera_background_hotspot_metrics(
        camera_prediction,
        clean_background_mask,
        background_hotspot_threshold,
    )
    highfreq_metrics = _camera_background_highfreq_metrics(
        camera_prediction,
        clean_background_mask,
    )
    return {
        "dice_base": dice_base,
        "dice_loss": foreground_dice_weight * dice_base,
        "foreground_hole_base": hole_metrics["artifact_base"],
        "foreground_hole_loss": foreground_hole_weight * hole_metrics["artifact_base"],
        "foreground_hole_ratio": hole_metrics["foreground_hole_ratio"],
        "background_speckle_base": speckle_metrics["background_speckle_base"],
        "background_speckle_loss": background_speckle_weight * speckle_metrics["background_speckle_base"],
        "background_speckle_ratio": speckle_metrics["background_speckle_ratio"],
        "background_hotspot_base": hotspot_metrics["background_hotspot_base"],
        "background_hotspot_loss": background_hotspot_weight * hotspot_metrics["background_hotspot_base"],
        "background_hotspot_ratio": hotspot_metrics["background_hotspot_ratio"],
        "background_highfreq_base": highfreq_metrics["background_highfreq_base"],
        "background_highfreq_loss": background_highfreq_weight * highfreq_metrics["background_highfreq_base"],
        "background_highfreq_energy": highfreq_metrics["background_highfreq_energy"],
    }


def _camera_readability_terms(
    camera_prediction: torch.Tensor,
    camera_loss_prediction: torch.Tensor,
    target_batch: torch.Tensor,
    foreground_mask: torch.Tensor,
    background_mask: torch.Tensor,
    *,
    readability_weight: float,
    background_speckle_weight: float,
    background_hotspot_weight: float,
    background_hotspot_threshold: float,
    background_highfreq_weight: float,
    readable_contrast_weight: float,
    readable_contrast_target: float,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if readability_weight != 0.0:
        readability_metrics = _camera_readability_metrics(
            camera_loss_prediction,
            target_batch,
            foreground_mask,
        )
        readability_base = readability_metrics["readability_base"]
        readability_penalty = readability_metrics["readability_penalty"]
    else:
        readability_base = zero
        readability_penalty = zero
    speckle_metrics = _camera_background_speckle_metrics(camera_loss_prediction, background_mask)
    hotspot_metrics = _camera_background_hotspot_metrics(
        camera_prediction,
        background_mask,
        background_hotspot_threshold,
    )
    highfreq_metrics = _camera_background_highfreq_metrics(camera_prediction, background_mask)
    contrast_metrics = _camera_readable_contrast_metrics(
        camera_loss_prediction,
        foreground_mask,
        background_mask,
        readable_contrast_target,
    )
    return {
        "readability_base": readability_base,
        "readability_penalty": readability_penalty,
        "readability_loss": readability_weight * readability_base,
        "background_speckle_base": speckle_metrics["background_speckle_base"],
        "background_speckle_ratio": speckle_metrics["background_speckle_ratio"],
        "background_speckle_loss": background_speckle_weight * speckle_metrics["background_speckle_base"],
        "background_hotspot_base": hotspot_metrics["background_hotspot_base"],
        "background_hotspot_ratio": hotspot_metrics["background_hotspot_ratio"],
        "background_hotspot_loss": background_hotspot_weight * hotspot_metrics["background_hotspot_base"],
        "background_highfreq_base": highfreq_metrics["background_highfreq_base"],
        "background_highfreq_energy": highfreq_metrics["background_highfreq_energy"],
        "background_highfreq_loss": background_highfreq_weight * highfreq_metrics["background_highfreq_base"],
        "readable_contrast_base": contrast_metrics["readable_contrast_base"],
        "readable_contrast": contrast_metrics["readable_contrast"],
        "readable_contrast_loss": readable_contrast_weight * contrast_metrics["readable_contrast_base"],
    }


def _student_label_margin_terms(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    *,
    weight: float,
    labels: tuple[int, ...],
    target: float,
    focus_label: int | None,
    focus_weight: float,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if weight != 0.0 or prototypes is not None:
        if weight != 0.0 and prototypes is None:
            raise ValueError("student_label_margin_weight requires student_label_margin_prototypes.")
        metrics = _student_label_margin_metrics(
            camera_prediction,
            class_labels,
            prototypes,
            labels,
            target,
            focus_label,
            focus_weight,
        )
        base = metrics["student_label_margin_base"]
        margin_mean = metrics["student_label_margin_mean"]
        correct_ratio = metrics["student_label_correct_ratio"]
    else:
        base = zero
        margin_mean = zero
        correct_ratio = zero
    return {
        "base": base,
        "loss": weight * base,
        "margin_mean": margin_mean,
        "correct_ratio": correct_ratio,
    }


def _student_prototype_objective_terms(
    camera_prediction: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    *,
    prototype_ce_weight: float,
    prototype_ce_labels: tuple[int, ...],
    prototype_ce_temperature: float,
    negative_margin_weight: float,
    negative_margin_config: tuple[tuple[int, int, float], ...],
    roi_prototype_ce_weight: float,
    roi_prototype_ce_labels: tuple[int, ...],
    roi_prototype_ce_temperature: float,
    roi_negative_margin_weight: float,
    roi_negative_margin_config: tuple[tuple[int, int, float], ...],
    roi_foreground_threshold: float,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if (
        prototype_ce_weight != 0.0
        or negative_margin_weight != 0.0
        or roi_prototype_ce_weight != 0.0
        or roi_negative_margin_weight != 0.0
    ) and prototypes is None:
        raise ValueError(
            "student prototype label objectives "
            "require student_label_margin_prototypes."
        )
    if prototype_ce_weight != 0.0:
        prototype_metrics = _student_prototype_ce_metrics(
            camera_prediction,
            class_labels,
            prototypes,
            prototype_ce_labels,
            prototype_ce_temperature,
        )
        prototype_base = prototype_metrics["student_prototype_ce_base"]
        prototype_correct_ratio = prototype_metrics["student_prototype_ce_correct_ratio"]
        prototype_count = prototype_metrics["student_prototype_ce_count"]
    else:
        prototype_base = zero
        prototype_correct_ratio = zero
        prototype_count = zero
    if negative_margin_weight != 0.0:
        negative_metrics = _student_negative_label_margin_metrics(
            camera_prediction,
            class_labels,
            prototypes,
            negative_margin_config,
        )
        negative_base = negative_metrics["student_negative_label_margin_base"]
        negative_mean = negative_metrics["student_negative_label_margin_mean"]
        negative_count = negative_metrics["student_negative_label_margin_count"]
    else:
        negative_base = zero
        negative_mean = zero
        negative_count = zero
    if roi_prototype_ce_weight != 0.0:
        roi_prototype_metrics = _student_roi_prototype_ce_metrics(
            camera_prediction,
            class_labels,
            prototypes,
            roi_prototype_ce_labels,
            roi_prototype_ce_temperature,
            roi_foreground_threshold,
        )
        roi_prototype_base = roi_prototype_metrics["student_roi_prototype_ce_base"]
        roi_prototype_correct_ratio = roi_prototype_metrics["student_roi_prototype_ce_correct_ratio"]
        roi_prototype_count = roi_prototype_metrics["student_roi_prototype_ce_count"]
    else:
        roi_prototype_base = zero
        roi_prototype_correct_ratio = zero
        roi_prototype_count = zero
    if roi_negative_margin_weight != 0.0:
        roi_negative_metrics = _student_roi_negative_label_margin_metrics(
            camera_prediction,
            class_labels,
            prototypes,
            roi_negative_margin_config,
            roi_foreground_threshold,
        )
        roi_negative_base = roi_negative_metrics["student_roi_negative_label_margin_base"]
        roi_negative_mean = roi_negative_metrics["student_roi_negative_label_margin_mean"]
        roi_negative_count = roi_negative_metrics["student_roi_negative_label_margin_count"]
    else:
        roi_negative_base = zero
        roi_negative_mean = zero
        roi_negative_count = zero
    return {
        "prototype_base": prototype_base,
        "prototype_loss": prototype_ce_weight * prototype_base,
        "prototype_correct_ratio": prototype_correct_ratio,
        "prototype_count": prototype_count,
        "negative_base": negative_base,
        "negative_loss": negative_margin_weight * negative_base,
        "negative_mean": negative_mean,
        "negative_count": negative_count,
        "roi_prototype_base": roi_prototype_base,
        "roi_prototype_loss": roi_prototype_ce_weight * roi_prototype_base,
        "roi_prototype_correct_ratio": roi_prototype_correct_ratio,
        "roi_prototype_count": roi_prototype_count,
        "roi_negative_base": roi_negative_base,
        "roi_negative_loss": roi_negative_margin_weight * roi_negative_base,
        "roi_negative_mean": roi_negative_mean,
        "roi_negative_count": roi_negative_count,
    }


def _aux_semantic_objective_terms(
    outputs: dict[str, torch.Tensor],
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    prototypes: torch.Tensor | None,
    *,
    plane_key: str,
    weight: float,
    prototype_ce_weight: float,
    negative_margin_weight: float,
    prototype_ce_labels: tuple[int, ...],
    prototype_ce_temperature: float,
    negative_margin_config: tuple[tuple[int, int, float], ...],
    zero: torch.Tensor,
) -> dict[str, torch.Tensor]:
    active = weight != 0.0 and (prototype_ce_weight != 0.0 or negative_margin_weight != 0.0)
    if active:
        if prototypes is None:
            raise ValueError("aux semantic objectives require student_label_margin_prototypes.")
        semantic_plane = _resize_aux_semantic_plane(outputs, plane_key, target_batch)
        if prototype_ce_weight != 0.0:
            prototype_metrics = _student_prototype_ce_metrics(
                semantic_plane,
                class_labels,
                prototypes,
                prototype_ce_labels,
                prototype_ce_temperature,
            )
            prototype_base = prototype_metrics["student_prototype_ce_base"]
            prototype_correct_ratio = prototype_metrics["student_prototype_ce_correct_ratio"]
            prototype_count = prototype_metrics["student_prototype_ce_count"]
        else:
            prototype_base = zero
            prototype_correct_ratio = zero
            prototype_count = zero
        if negative_margin_weight != 0.0:
            negative_metrics = _student_negative_label_margin_metrics(
                semantic_plane,
                class_labels,
                prototypes,
                negative_margin_config,
            )
            negative_base = negative_metrics["student_negative_label_margin_base"]
            negative_mean = negative_metrics["student_negative_label_margin_mean"]
            negative_count = negative_metrics["student_negative_label_margin_count"]
        else:
            negative_base = zero
            negative_mean = zero
            negative_count = zero
    else:
        prototype_base = zero
        prototype_correct_ratio = zero
        prototype_count = zero
        negative_base = zero
        negative_mean = zero
        negative_count = zero
    prototype_loss = weight * prototype_ce_weight * prototype_base
    negative_loss = weight * negative_margin_weight * negative_base
    return {
        "loss": prototype_loss + negative_loss,
        "prototype_base": prototype_base,
        "prototype_loss": prototype_loss,
        "prototype_correct_ratio": prototype_correct_ratio,
        "prototype_count": prototype_count,
        "negative_base": negative_base,
        "negative_loss": negative_loss,
        "negative_mean": negative_mean,
        "negative_count": negative_count,
    }


def _aux_topology_terms(
    outputs: dict[str, torch.Tensor],
    topology_target: torch.Tensor,
    *,
    plane_key: str,
    weight: float,
    dice_weight: float,
    mse_weight: float,
    target_transform: str,
    foreground_threshold: float,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    target_info = transform_aux_topology_target(
        topology_target,
        target_transform,
        foreground_threshold,
    )
    if weight != 0.0:
        metrics = _aux_topology_metrics(
            outputs,
            plane_key,
            target_info["target"],
            target_info["foreground_mask"],
        )
        dice = metrics["dice"]
        mse = metrics["mse"]
        base = float(dice_weight) * metrics["dice_penalty"] + float(mse_weight) * mse
    else:
        dice = zero
        mse = zero
        base = zero
    return {
        "target_info": target_info,
        "dice": dice,
        "mse": mse,
        "loss": weight * base,
    }


def _aux_detail_terms(
    camera_prediction: torch.Tensor,
    target_batch: torch.Tensor,
    aux_detail_target: torch.Tensor | None,
    clean_foreground_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    foreground_threshold: float,
    gate_dilation: int,
    shape_weight: float,
    edge_weight: float,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor | None]:
    gate_dilation = int(gate_dilation)
    if gate_dilation > 0:
        gate_mask = _dilate_foreground_mask(clean_foreground_mask, gate_dilation)
        gate_foreground_ratio = gate_mask.mean().detach()
    else:
        gate_mask = None
        gate_foreground_ratio = zero
    if aux_detail_target is not None and (shape_weight != 0.0 or edge_weight != 0.0):
        if aux_detail_target.shape != target_batch.shape:
            raise ValueError(
                "aux_detail_target must match target_batch shape, "
                f"got {tuple(aux_detail_target.shape)} and {tuple(target_batch.shape)}"
            )
        aux_detail = aux_detail_target.float().clamp(0.0, 1.0)
        aux_detail_foreground = (aux_detail > foreground_threshold).float()
        if gate_mask is not None:
            aux_detail = aux_detail * gate_mask
            aux_detail_foreground = aux_detail_foreground * gate_mask
        active_foreground_ratio = aux_detail_foreground.mean().detach()
        has_foreground = bool(aux_detail_foreground.detach().sum().item() > 0)
        shape_base = (
            _foreground_shape_loss_for_samples(camera_prediction, aux_detail_foreground, sample_weights)
            if shape_weight != 0.0 and has_foreground
            else zero
        )
        edge_base = (
            _one_shot_edge_shape_loss(
                camera_prediction,
                aux_detail,
                aux_detail_foreground,
                sample_weights,
            )
            if edge_weight != 0.0 and has_foreground
            else zero
        )
    else:
        shape_base = zero
        edge_base = zero
        active_foreground_ratio = zero
    return {
        "gate_mask": gate_mask,
        "gate_foreground_ratio": gate_foreground_ratio,
        "active_foreground_ratio": active_foreground_ratio,
        "shape_base": shape_base,
        "shape_loss": shape_weight * shape_base,
        "edge_base": edge_base,
        "edge_loss": edge_weight * edge_base,
    }


def _generator_residual_terms(
    outputs: dict[str, torch.Tensor],
    camera_prediction: torch.Tensor,
    foreground_mask: torch.Tensor,
    background_mask: torch.Tensor,
    clean_background_mask: torch.Tensor,
    aux_detail_gate_mask: torch.Tensor | None,
    *,
    l1_weight: float,
    background_weight: float,
    zero: torch.Tensor,
) -> dict[str, torch.Tensor]:
    residual_map = outputs.get("generator_residual_map")
    if residual_map is not None and (l1_weight != 0.0 or background_weight != 0.0):
        residual_abs = residual_map.float().abs()
        l1_base = residual_abs.mean()
        residual_background_mask = (
            (1.0 - aux_detail_gate_mask).clamp(0.0, 1.0)
            if aux_detail_gate_mask is not None
            else clean_background_mask
        )
        background_base = _masked_mean(residual_abs, residual_background_mask)
        background_mask_ratio = residual_background_mask.mean().detach()
    else:
        l1_base = zero
        background_base = zero
        background_mask_ratio = zero
    residual_abs_mean = (
        outputs["generator_residual_abs_mean"].detach().float().mean()
        if "generator_residual_abs_mean" in outputs
        else zero
    )
    residual_background_abs_mean = (
        outputs["generator_residual_bg_abs_mean"].detach().float().mean()
        if "generator_residual_bg_abs_mean" in outputs
        else zero
    )
    generator_base = outputs.get("generator_base_command")
    if generator_base is not None:
        if generator_base.shape != camera_prediction.shape:
            raise ValueError(
                "generator_base_command must match camera_plane shape, "
                f"got {tuple(generator_base.shape)} and {tuple(camera_prediction.shape)}"
            )
        student_base_error = (camera_prediction.float() - generator_base.float()).pow(2)
        student_base_mse = student_base_error.mean().detach()
        student_base_foreground_mse = _masked_mean(student_base_error, foreground_mask).detach()
        student_base_background_mse = _masked_mean(student_base_error, background_mask).detach()
    else:
        student_base_mse = zero
        student_base_foreground_mse = zero
        student_base_background_mse = zero
    return {
        "l1_base": l1_base,
        "l1_loss": l1_weight * l1_base,
        "background_base": background_base,
        "background_loss": background_weight * background_base,
        "background_mask_ratio": background_mask_ratio,
        "residual_abs_mean": residual_abs_mean,
        "residual_background_abs_mean": residual_background_abs_mean,
        "student_base_mse": student_base_mse,
        "student_base_foreground_mse": student_base_foreground_mse,
        "student_base_background_mse": student_base_background_mse,
    }


def _diversity_terms(
    camera_prediction: torch.Tensor,
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    slm_amplitude: torch.Tensor,
    *,
    code_weight: float,
    sample_code_weight: float,
    teacher_pairwise_weight: float,
    cross_class_weight: float,
    minimum_teacher_mse: float,
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    pairwise = _one_shot_pairwise_diversity_metrics(
        camera_prediction,
        target_batch,
        class_labels,
        minimum_teacher_mse,
    )
    cross_class = _one_shot_cross_class_pairwise_metrics(
        camera_prediction,
        target_batch,
        class_labels,
        minimum_teacher_mse,
    )
    return {
        "pairwise": pairwise,
        "cross_class": cross_class,
        "code_loss": code_weight * pairwise["code_diversity_base"],
        "sample_code_loss": sample_code_weight * pairwise["code_diversity_base"],
        "teacher_pairwise_loss": teacher_pairwise_weight * pairwise["teacher_pairwise_diversity_base"],
        "cross_class_loss": cross_class_weight * cross_class["cross_class_pairwise_base"],
        "slm_pairwise_mse": _pairwise_mse_summary(slm_amplitude),
        "slm_per_class_std_mean": _per_class_std_summary(slm_amplitude, class_labels),
    }


def _reconstruction_terms(
    outputs: dict[str, torch.Tensor],
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    *,
    foreground_threshold: float,
    foreground_weight: float,
    background_weight: float,
    class_weight_labels: tuple[int, ...],
    class_weight: float,
) -> dict[str, torch.Tensor]:
    foreground_mask = (target_batch > foreground_threshold).float()
    background_mask = 1.0 - foreground_mask
    camera_prediction = outputs["camera_plane"]
    camera_loss_prediction = outputs.get("camera_plane_loss", camera_prediction)
    sample_weights = _resolve_sample_weights(
        target_batch,
        class_labels,
        class_weight_labels,
        class_weight,
    )
    has_class_weights = class_labels is not None and bool(class_weight_labels) and class_weight != 1.0
    if has_class_weights:
        metrics = _weighted_reconstruction_metrics_for_samples(
            camera_prediction,
            target_batch,
            foreground_mask,
            background_mask,
            foreground_weight,
            background_weight,
            sample_weights,
        )
    else:
        metrics = _weighted_reconstruction_metrics(
            camera_prediction,
            target_batch,
            foreground_mask,
            background_mask,
            foreground_weight,
            background_weight,
        )
    return {
        "foreground_mask": foreground_mask,
        "background_mask": background_mask,
        "camera_prediction": camera_prediction,
        "camera_loss_prediction": camera_loss_prediction,
        "sample_weights": sample_weights,
        "metrics": metrics,
        "zero": _zero_like_loss(camera_prediction),
    }


def one_shot_loss(
    outputs: dict[str, torch.Tensor],
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None = None,
    aux_detail_target: torch.Tensor | None = None,
    foreground_threshold: float = 0.05,
    foreground_weight: float = 3.0,
    background_weight: float = 1.0,
    class_recon_weight_labels: tuple[int, ...] = (),
    class_recon_weight: float = 1.0,
    shape_loss_weight: float = 0.0,
    edge_loss_weight: float = 0.0,
    slm_smoothness_weight: float = 0.0,
    slm_contrast_weight: float = 0.0,
    camera_saturation_weight: float = 0.0,
    camera_topology_weight: float = 0.0,
    camera_artifact_weight: float = 0.0,
    camera_readability_weight: float = 0.0,
    camera_background_speckle_weight: float = 0.0,
    camera_background_hotspot_weight: float = 0.0,
    camera_background_hotspot_threshold: float = 0.08,
    camera_background_highfreq_weight: float = 0.0,
    camera_readable_contrast_weight: float = 0.0,
    camera_readable_contrast_target: float = 4.5,
    camera_recon_weight: float = 1.0,
    clean_target_enabled: bool = False,
    clean_foreground_dice_weight: float = 0.0,
    clean_foreground_hole_weight: float = 0.0,
    clean_background_speckle_weight: float = 0.0,
    clean_background_hotspot_weight: float = 0.0,
    clean_background_highfreq_weight: float = 0.0,
    encoder_input_topology_weight: float = 0.0,
    encoder_input_topology_dice_weight: float = 1.0,
    encoder_input_topology_mse_weight: float = 1.0,
    student_label_margin_weight: float = 0.0,
    student_label_margin_target: float = 0.03,
    student_label_margin_labels: tuple[int, ...] = (),
    student_label_margin_focus_label: int | None = None,
    student_label_margin_focus_weight: float = 1.0,
    student_label_margin_prototypes: torch.Tensor | None = None,
    student_prototype_ce_weight: float = 0.0,
    student_prototype_ce_labels: tuple[int, ...] = (),
    student_prototype_ce_temperature: float = 0.07,
    student_negative_label_margin_weight: float = 0.0,
    student_negative_label_margin_config: tuple[tuple[int, int, float], ...] = (),
    student_roi_prototype_ce_weight: float = 0.0,
    student_roi_prototype_ce_labels: tuple[int, ...] = (),
    student_roi_prototype_ce_temperature: float = 0.07,
    student_roi_negative_label_margin_weight: float = 0.0,
    student_roi_negative_label_margin_config: tuple[tuple[int, int, float], ...] = (),
    student_roi_foreground_threshold: float = 0.20,
    aux_semantic_plane_key: str = "",
    aux_semantic_weight: float = 0.0,
    aux_semantic_prototype_ce_weight: float = 0.0,
    aux_semantic_negative_label_margin_weight: float = 0.0,
    aux_topology_plane_key: str = "",
    aux_topology_weight: float = 0.0,
    aux_topology_dice_weight: float = 1.0,
    aux_topology_mse_weight: float = 1.0,
    aux_topology_target_transform: str = "none",
    student_pair_contrast_config: tuple[tuple[int, int, float], ...] = (),
    student_display_speckle_weight: float = 0.0,
    aux_detail_edge_weight: float = 0.0,
    aux_detail_shape_weight: float = 0.0,
    aux_detail_foreground_gate_dilation: int = 0,
    residual_l1_weight: float = 0.0,
    residual_background_weight: float = 0.0,
    code_diversity_weight: float = 0.0,
    sample_code_diversity_weight: float = 0.0,
    teacher_pairwise_diversity_weight: float = 0.0,
    cross_class_pairwise_weight: float = 0.0,
    diversity_min_teacher_mse: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    reconstruction = _reconstruction_terms(
        outputs,
        target_batch,
        class_labels,
        foreground_threshold=foreground_threshold,
        foreground_weight=foreground_weight,
        background_weight=background_weight,
        class_weight_labels=class_recon_weight_labels,
        class_weight=class_recon_weight,
    )
    foreground_mask = reconstruction["foreground_mask"]
    background_mask = reconstruction["background_mask"]
    camera_prediction = reconstruction["camera_prediction"]
    camera_loss_prediction = reconstruction["camera_loss_prediction"]
    sample_weights = reconstruction["sample_weights"]
    camera_metrics = reconstruction["metrics"]
    zero = reconstruction["zero"]
    clean_target, clean_foreground_mask, clean_background_mask, clean_target_fg_ratio = (
        _resolve_clean_target_masks(
            target_batch,
            foreground_mask,
            background_mask,
            enabled=clean_target_enabled,
            zero=zero,
        )
    )

    shape_base = (
        _foreground_shape_loss_for_samples(camera_prediction, clean_foreground_mask, sample_weights)
        if shape_loss_weight != 0.0
        else zero
    )
    shape_loss = shape_loss_weight * shape_base
    edge_base = (
        _one_shot_edge_shape_loss(camera_prediction, clean_target, clean_foreground_mask, sample_weights)
        if edge_loss_weight != 0.0
        else zero
    )
    edge_loss = edge_loss_weight * edge_base
    slm_tv = _total_variation(outputs["slm_amplitude"])
    slm_smoothness = slm_smoothness_weight * slm_tv
    slm_amplitude = outputs["slm_amplitude"]
    if slm_contrast_weight != 0.0:
        grid_tensor = outputs.get("encoder_input_grid_amplitude")
        if grid_tensor is not None:
            grid_h, grid_w = grid_tensor.shape[-2:]
            slm_grid = F.interpolate(slm_amplitude, size=(grid_h, grid_w), mode="area")
        else:
            slm_grid = slm_amplitude
        per_sample_slm_std = slm_grid.reshape(slm_amplitude.shape[0], -1).std(dim=1)
        slm_contrast_base = F.relu(0.12 - per_sample_slm_std).mean()
        slm_contrast = slm_contrast_weight * slm_contrast_base
    else:
        per_sample_slm_std = slm_amplitude.reshape(slm_amplitude.shape[0], -1).std(dim=1)
        slm_contrast_base = zero
        slm_contrast = zero
    camera_saturation_base = _camera_saturation_loss(camera_loss_prediction)
    camera_saturation = camera_saturation_weight * camera_saturation_base
    # 拓扑约束：保持笔画连接和弯折的结构，对复杂数字(3/4/5)尤为重要。
    topology_target = clean_target if clean_target_enabled else target_batch
    topology_fg = clean_foreground_mask if clean_target_enabled else foreground_mask
    camera_topology_base = (
        _camera_topology_loss(camera_prediction, topology_target, topology_fg)
        if camera_topology_weight != 0.0
        else zero
    )
    camera_topology = camera_topology_weight * camera_topology_base
    # 抗黑斑：惩罚前景笔画内部的暗斑和粗糙纹理。
    artifact_metrics = _camera_artifact_metrics(
        camera_loss_prediction, target_batch, foreground_mask
    )
    camera_artifact_base = artifact_metrics["artifact_base"]
    foreground_hole_ratio = artifact_metrics["foreground_hole_ratio"]
    camera_artifact = camera_artifact_weight * camera_artifact_base
    clean_terms = _clean_target_regularization_terms(
        camera_prediction,
        camera_loss_prediction,
        clean_target,
        clean_foreground_mask,
        clean_background_mask,
        enabled=clean_target_enabled,
        foreground_dice_weight=clean_foreground_dice_weight,
        foreground_hole_weight=clean_foreground_hole_weight,
        background_speckle_weight=clean_background_speckle_weight,
        background_hotspot_weight=clean_background_hotspot_weight,
        background_highfreq_weight=clean_background_highfreq_weight,
        background_hotspot_threshold=camera_background_hotspot_threshold,
        zero=zero,
    )
    clean_dice_base = clean_terms["dice_base"]
    clean_dice_loss = clean_terms["dice_loss"]
    clean_fg_hole_base = clean_terms["foreground_hole_base"]
    clean_fg_hole_loss = clean_terms["foreground_hole_loss"]
    clean_fg_hole_ratio = clean_terms["foreground_hole_ratio"]
    clean_bg_speckle_base = clean_terms["background_speckle_base"]
    clean_bg_speckle_loss = clean_terms["background_speckle_loss"]
    clean_bg_speckle_ratio = clean_terms["background_speckle_ratio"]
    clean_bg_hotspot_base = clean_terms["background_hotspot_base"]
    clean_bg_hotspot_loss = clean_terms["background_hotspot_loss"]
    clean_bg_hotspot_ratio = clean_terms["background_hotspot_ratio"]
    clean_bg_highfreq_base = clean_terms["background_highfreq_base"]
    clean_bg_highfreq_loss = clean_terms["background_highfreq_loss"]
    clean_bg_highfreq_energy = clean_terms["background_highfreq_energy"]

    if encoder_input_topology_weight != 0.0:
        encoder_topology_metrics = _encoder_input_topology_metrics(outputs, topology_target, topology_fg)
        encoder_input_topology_dice = encoder_topology_metrics["dice"]
        encoder_input_topology_mse = encoder_topology_metrics["mse"]
        encoder_input_topology_base = (
            float(encoder_input_topology_dice_weight) * encoder_topology_metrics["dice_penalty"]
            + float(encoder_input_topology_mse_weight) * encoder_input_topology_mse
        )
    else:
        encoder_input_topology_dice = zero
        encoder_input_topology_mse = zero
        encoder_input_topology_base = zero
    encoder_input_topology = encoder_input_topology_weight * encoder_input_topology_base

    readability_terms = _camera_readability_terms(
        camera_prediction,
        camera_loss_prediction,
        target_batch,
        foreground_mask,
        background_mask,
        readability_weight=camera_readability_weight,
        background_speckle_weight=camera_background_speckle_weight,
        background_hotspot_weight=camera_background_hotspot_weight,
        background_hotspot_threshold=camera_background_hotspot_threshold,
        background_highfreq_weight=camera_background_highfreq_weight,
        readable_contrast_weight=camera_readable_contrast_weight,
        readable_contrast_target=camera_readable_contrast_target,
        zero=zero,
    )
    camera_readability_penalty = readability_terms["readability_penalty"]
    camera_readability = readability_terms["readability_loss"]
    camera_background_speckle_base = readability_terms["background_speckle_base"]
    background_speckle_ratio = readability_terms["background_speckle_ratio"]
    camera_background_speckle = readability_terms["background_speckle_loss"]
    camera_background_hotspot_base = readability_terms["background_hotspot_base"]
    background_hotspot_ratio = readability_terms["background_hotspot_ratio"]
    camera_background_hotspot = readability_terms["background_hotspot_loss"]
    camera_background_highfreq_base = readability_terms["background_highfreq_base"]
    background_highfreq_energy = readability_terms["background_highfreq_energy"]
    camera_background_highfreq = readability_terms["background_highfreq_loss"]
    camera_readable_contrast_base = readability_terms["readable_contrast_base"]
    camera_readable_contrast_value = readability_terms["readable_contrast"]
    camera_readable_contrast = readability_terms["readable_contrast_loss"]
    label_margin_terms = _student_label_margin_terms(
        camera_prediction,
        class_labels,
        student_label_margin_prototypes,
        weight=student_label_margin_weight,
        labels=student_label_margin_labels,
        target=student_label_margin_target,
        focus_label=student_label_margin_focus_label,
        focus_weight=student_label_margin_focus_weight,
        zero=zero,
    )
    student_label_margin_base = label_margin_terms["base"]
    student_label_margin = label_margin_terms["loss"]
    student_label_margin_mean = label_margin_terms["margin_mean"]
    student_label_correct_ratio = label_margin_terms["correct_ratio"]
    prototype_terms = _student_prototype_objective_terms(
        camera_prediction,
        class_labels,
        student_label_margin_prototypes,
        prototype_ce_weight=student_prototype_ce_weight,
        prototype_ce_labels=student_prototype_ce_labels,
        prototype_ce_temperature=student_prototype_ce_temperature,
        negative_margin_weight=student_negative_label_margin_weight,
        negative_margin_config=student_negative_label_margin_config,
        roi_prototype_ce_weight=student_roi_prototype_ce_weight,
        roi_prototype_ce_labels=student_roi_prototype_ce_labels,
        roi_prototype_ce_temperature=student_roi_prototype_ce_temperature,
        roi_negative_margin_weight=student_roi_negative_label_margin_weight,
        roi_negative_margin_config=student_roi_negative_label_margin_config,
        roi_foreground_threshold=student_roi_foreground_threshold,
        zero=zero,
    )
    student_prototype_ce_base = prototype_terms["prototype_base"]
    student_prototype_ce_loss = prototype_terms["prototype_loss"]
    student_prototype_ce_correct_ratio = prototype_terms["prototype_correct_ratio"]
    student_prototype_ce_count = prototype_terms["prototype_count"]
    student_negative_label_margin_base = prototype_terms["negative_base"]
    student_negative_label_margin_loss = prototype_terms["negative_loss"]
    student_negative_label_margin_mean = prototype_terms["negative_mean"]
    student_negative_label_margin_count = prototype_terms["negative_count"]
    student_roi_prototype_ce_base = prototype_terms["roi_prototype_base"]
    student_roi_prototype_ce_loss = prototype_terms["roi_prototype_loss"]
    student_roi_prototype_ce_correct_ratio = prototype_terms["roi_prototype_correct_ratio"]
    student_roi_prototype_ce_count = prototype_terms["roi_prototype_count"]
    student_roi_negative_label_margin_base = prototype_terms["roi_negative_base"]
    student_roi_negative_label_margin_loss = prototype_terms["roi_negative_loss"]
    student_roi_negative_label_margin_mean = prototype_terms["roi_negative_mean"]
    student_roi_negative_label_margin_count = prototype_terms["roi_negative_count"]
    aux_semantic_terms = _aux_semantic_objective_terms(
        outputs,
        target_batch,
        class_labels,
        student_label_margin_prototypes,
        plane_key=aux_semantic_plane_key,
        weight=aux_semantic_weight,
        prototype_ce_weight=aux_semantic_prototype_ce_weight,
        negative_margin_weight=aux_semantic_negative_label_margin_weight,
        prototype_ce_labels=student_prototype_ce_labels,
        prototype_ce_temperature=student_prototype_ce_temperature,
        negative_margin_config=student_negative_label_margin_config,
        zero=zero,
    )
    aux_semantic_loss = aux_semantic_terms["loss"]
    aux_semantic_prototype_ce_base = aux_semantic_terms["prototype_base"]
    aux_semantic_prototype_ce_loss = aux_semantic_terms["prototype_loss"]
    aux_semantic_prototype_ce_correct_ratio = aux_semantic_terms["prototype_correct_ratio"]
    aux_semantic_prototype_ce_count = aux_semantic_terms["prototype_count"]
    aux_semantic_negative_label_margin_base = aux_semantic_terms["negative_base"]
    aux_semantic_negative_label_margin_loss = aux_semantic_terms["negative_loss"]
    aux_semantic_negative_label_margin_mean = aux_semantic_terms["negative_mean"]
    aux_semantic_negative_label_margin_count = aux_semantic_terms["negative_count"]
    aux_topology_terms = _aux_topology_terms(
        outputs,
        topology_target,
        plane_key=aux_topology_plane_key,
        weight=aux_topology_weight,
        dice_weight=aux_topology_dice_weight,
        mse_weight=aux_topology_mse_weight,
        target_transform=aux_topology_target_transform,
        foreground_threshold=foreground_threshold,
        zero=zero,
    )
    aux_topology_target_info = aux_topology_terms["target_info"]
    aux_topology_dice = aux_topology_terms["dice"]
    aux_topology_mse = aux_topology_terms["mse"]
    aux_topology_loss = aux_topology_terms["loss"]
    # Student pair contrast: penalize confusion between problematic class pairs.
    if student_pair_contrast_config:
        pair_contrast_metrics = _student_pair_contrast_metrics(
            camera_prediction, class_labels, student_pair_contrast_config,
        )
        student_pair_contrast_base = pair_contrast_metrics["student_pair_contrast_base"]
        student_pair_contrast_count = pair_contrast_metrics["student_pair_contrast_count"]
    else:
        student_pair_contrast_base = zero
        student_pair_contrast_count = zero
    student_pair_contrast_loss = student_pair_contrast_base  # no extra weight; base is the mean penalty
    # Student display-speckle: penalize background noise in display-normalized domain.
    if student_display_speckle_weight != 0.0:
        display_background_mask = 1.0 - clean_foreground_mask if clean_target_enabled else background_mask
        display_speckle_metrics = _student_display_speckle_metrics(
            camera_prediction, foreground_mask, display_background_mask,
        )
        display_speckle_base = display_speckle_metrics["display_speckle_base"]
        display_speckle_ratio = display_speckle_metrics["display_speckle_ratio"]
        display_highfreq_energy = display_speckle_metrics["display_highfreq_energy"]
    else:
        display_speckle_base = zero
        display_speckle_ratio = zero
        display_highfreq_energy = zero
    display_speckle_loss = student_display_speckle_weight * display_speckle_base
    aux_detail_terms = _aux_detail_terms(
        camera_prediction,
        target_batch,
        aux_detail_target,
        clean_foreground_mask,
        sample_weights,
        foreground_threshold=foreground_threshold,
        gate_dilation=aux_detail_foreground_gate_dilation,
        shape_weight=aux_detail_shape_weight,
        edge_weight=aux_detail_edge_weight,
        zero=zero,
    )
    aux_detail_gate_mask = aux_detail_terms["gate_mask"]
    aux_detail_gate_fg_ratio = aux_detail_terms["gate_foreground_ratio"]
    aux_detail_active_fg_ratio = aux_detail_terms["active_foreground_ratio"]
    aux_detail_shape_base = aux_detail_terms["shape_base"]
    aux_detail_shape_loss = aux_detail_terms["shape_loss"]
    aux_detail_edge_base = aux_detail_terms["edge_base"]
    aux_detail_edge_loss = aux_detail_terms["edge_loss"]
    residual_terms = _generator_residual_terms(
        outputs,
        camera_prediction,
        foreground_mask,
        background_mask,
        clean_background_mask,
        aux_detail_gate_mask,
        l1_weight=residual_l1_weight,
        background_weight=residual_background_weight,
        zero=zero,
    )
    residual_l1_base = residual_terms["l1_base"]
    residual_l1_loss = residual_terms["l1_loss"]
    residual_background_base = residual_terms["background_base"]
    residual_background_loss = residual_terms["background_loss"]
    residual_background_mask_ratio = residual_terms["background_mask_ratio"]
    generator_residual_abs_mean = residual_terms["residual_abs_mean"]
    generator_residual_bg_abs_mean = residual_terms["residual_background_abs_mean"]
    student_base_mse = residual_terms["student_base_mse"]
    student_base_foreground_mse = residual_terms["student_base_foreground_mse"]
    student_base_background_mse = residual_terms["student_base_background_mse"]
    diversity_terms = _diversity_terms(
        camera_prediction,
        target_batch,
        class_labels,
        slm_amplitude,
        code_weight=code_diversity_weight,
        sample_code_weight=sample_code_diversity_weight,
        teacher_pairwise_weight=teacher_pairwise_diversity_weight,
        cross_class_weight=cross_class_pairwise_weight,
        minimum_teacher_mse=diversity_min_teacher_mse,
    )
    diversity_metrics = diversity_terms["pairwise"]
    cross_class_metrics = diversity_terms["cross_class"]
    code_diversity = diversity_terms["code_loss"]
    sample_code_diversity = diversity_terms["sample_code_loss"]
    teacher_pairwise_diversity = diversity_terms["teacher_pairwise_loss"]
    cross_class_pairwise = diversity_terms["cross_class_loss"]
    slm_pairwise_mse = diversity_terms["slm_pairwise_mse"]
    slm_per_class_std_mean = diversity_terms["slm_per_class_std_mean"]
    total = (
        camera_recon_weight * camera_metrics["recon_loss"]
        + shape_loss
        + edge_loss
        + slm_smoothness
        + slm_contrast
        + camera_saturation
        + camera_topology
        + camera_artifact
        + camera_readability
        + camera_background_speckle
        + camera_background_hotspot
        + camera_background_highfreq
        + camera_readable_contrast
        + student_label_margin
        + student_prototype_ce_loss
        + student_negative_label_margin_loss
        + student_roi_prototype_ce_loss
        + student_roi_negative_label_margin_loss
        + aux_semantic_loss
        + aux_topology_loss
        + code_diversity
        + sample_code_diversity
        + teacher_pairwise_diversity
        + cross_class_pairwise
        + clean_dice_loss
        + clean_fg_hole_loss
        + clean_bg_speckle_loss
        + clean_bg_hotspot_loss
        + clean_bg_highfreq_loss
        + encoder_input_topology
        + student_pair_contrast_loss
        + display_speckle_loss
        + aux_detail_shape_loss
        + aux_detail_edge_loss
        + residual_l1_loss
        + residual_background_loss
    )
    return {
        "loss": total,
        "camera_recon_loss": camera_metrics["recon_loss"].detach(),
        "camera_fg_mse": camera_metrics["fg_mse"],
        "camera_bg_mse": camera_metrics["bg_mse"],
        "shape_loss": shape_loss.detach(),
        "shape_penalty": shape_base.detach(),
        "edge_loss": edge_loss.detach(),
        "edge_penalty": edge_base.detach(),
        "slm_smoothness_loss": slm_smoothness.detach(),
        "slm_tv": slm_tv.detach(),
        "slm_contrast_loss": slm_contrast.detach(),
        "slm_contrast_penalty": slm_contrast_base.detach(),
        "slm_per_sample_std_mean": per_sample_slm_std.mean().detach(),
        "slm_per_sample_std_min": per_sample_slm_std.min().detach(),
        "camera_saturation_loss": camera_saturation.detach(),
        "camera_saturation_penalty": camera_saturation_base.detach(),
        "camera_saturation_ratio": _camera_saturation_ratio(camera_prediction.detach()),
        "camera_low_saturation_ratio": _camera_low_saturation_ratio(camera_prediction.detach()),
        "camera_high_saturation_ratio": _camera_high_saturation_ratio(camera_prediction.detach()),
        "camera_topology_loss": camera_topology.detach(),
        "camera_artifact_loss": camera_artifact.detach(),
        "camera_readability_loss": camera_readability.detach(),
        "camera_readability_penalty": camera_readability_penalty.detach(),
        "camera_background_speckle_loss": camera_background_speckle.detach(),
        "camera_background_speckle_penalty": camera_background_speckle_base.detach(),
        "background_speckle_ratio": background_speckle_ratio.detach(),
        "camera_background_hotspot_loss": camera_background_hotspot.detach(),
        "camera_background_hotspot_penalty": camera_background_hotspot_base.detach(),
        "background_hotspot_ratio": background_hotspot_ratio.detach(),
        "camera_background_highfreq_loss": camera_background_highfreq.detach(),
        "camera_background_highfreq_penalty": camera_background_highfreq_base.detach(),
        "background_highfreq_energy": background_highfreq_energy.detach(),
        "camera_readable_contrast_loss": camera_readable_contrast.detach(),
        "camera_readable_contrast_penalty": camera_readable_contrast_base.detach(),
        "readable_contrast": camera_readable_contrast_value.detach(),
        "student_label_margin_loss": student_label_margin.detach(),
        "student_label_margin_penalty": student_label_margin_base.detach(),
        "student_label_margin_mean": student_label_margin_mean.detach(),
        "student_label_correct_ratio": student_label_correct_ratio.detach(),
        "student_prototype_ce_loss": student_prototype_ce_loss.detach(),
        "student_prototype_ce_penalty": student_prototype_ce_base.detach(),
        "student_prototype_ce_correct_ratio": student_prototype_ce_correct_ratio.detach(),
        "student_prototype_ce_count": student_prototype_ce_count.detach(),
        "student_negative_label_margin_loss": student_negative_label_margin_loss.detach(),
        "student_negative_label_margin_penalty": student_negative_label_margin_base.detach(),
        "student_negative_label_margin_mean": student_negative_label_margin_mean.detach(),
        "student_negative_label_margin_count": student_negative_label_margin_count.detach(),
        "student_roi_prototype_ce_loss": student_roi_prototype_ce_loss.detach(),
        "student_roi_prototype_ce_penalty": student_roi_prototype_ce_base.detach(),
        "student_roi_prototype_ce_correct_ratio": student_roi_prototype_ce_correct_ratio.detach(),
        "student_roi_prototype_ce_count": student_roi_prototype_ce_count.detach(),
        "student_roi_negative_label_margin_loss": student_roi_negative_label_margin_loss.detach(),
        "student_roi_negative_label_margin_penalty": student_roi_negative_label_margin_base.detach(),
        "student_roi_negative_label_margin_mean": student_roi_negative_label_margin_mean.detach(),
        "student_roi_negative_label_margin_count": student_roi_negative_label_margin_count.detach(),
        "aux_semantic_loss": aux_semantic_loss.detach(),
        "aux_semantic_prototype_ce_loss": aux_semantic_prototype_ce_loss.detach(),
        "aux_semantic_prototype_ce_penalty": aux_semantic_prototype_ce_base.detach(),
        "aux_semantic_prototype_ce_correct_ratio": aux_semantic_prototype_ce_correct_ratio.detach(),
        "aux_semantic_prototype_ce_count": aux_semantic_prototype_ce_count.detach(),
        "aux_semantic_negative_label_margin_loss": aux_semantic_negative_label_margin_loss.detach(),
        "aux_semantic_negative_label_margin_penalty": aux_semantic_negative_label_margin_base.detach(),
        "aux_semantic_negative_label_margin_mean": aux_semantic_negative_label_margin_mean.detach(),
        "aux_semantic_negative_label_margin_count": aux_semantic_negative_label_margin_count.detach(),
        "aux_topology_loss": aux_topology_loss.detach(),
        "aux_topology_dice": aux_topology_dice.detach(),
        "aux_topology_mse": aux_topology_mse.detach(),
        "aux_topology_target_fg_ratio": aux_topology_target_info["foreground_ratio"].detach(),
        "foreground_hole_ratio": foreground_hole_ratio.detach() if hasattr(foreground_hole_ratio, 'detach') else foreground_hole_ratio,
        "camera_foreground_dice": _foreground_dice_score(camera_prediction.detach(), foreground_mask),
        "clean_foreground_dice": _foreground_dice_score(camera_prediction.detach(), clean_foreground_mask),
        "clean_dice_loss": clean_dice_loss.detach(),
        "clean_dice_penalty": clean_dice_base.detach(),
        "clean_fg_hole_loss": clean_fg_hole_loss.detach(),
        "clean_fg_hole_penalty": clean_fg_hole_base.detach(),
        "clean_fg_hole_ratio": clean_fg_hole_ratio.detach() if hasattr(clean_fg_hole_ratio, 'detach') else clean_fg_hole_ratio,
        "clean_bg_speckle_loss": clean_bg_speckle_loss.detach(),
        "clean_bg_speckle_penalty": clean_bg_speckle_base.detach(),
        "clean_bg_speckle_ratio": clean_bg_speckle_ratio.detach() if hasattr(clean_bg_speckle_ratio, 'detach') else clean_bg_speckle_ratio,
        "clean_bg_hotspot_loss": clean_bg_hotspot_loss.detach(),
        "clean_bg_hotspot_penalty": clean_bg_hotspot_base.detach(),
        "clean_bg_hotspot_ratio": clean_bg_hotspot_ratio.detach() if hasattr(clean_bg_hotspot_ratio, 'detach') else clean_bg_hotspot_ratio,
        "clean_bg_highfreq_loss": clean_bg_highfreq_loss.detach(),
        "clean_bg_highfreq_penalty": clean_bg_highfreq_base.detach(),
        "clean_bg_highfreq_energy": clean_bg_highfreq_energy.detach() if hasattr(clean_bg_highfreq_energy, 'detach') else clean_bg_highfreq_energy,
        "clean_target_fg_ratio": clean_target_fg_ratio.detach() if hasattr(clean_target_fg_ratio, 'detach') else clean_target_fg_ratio,
        "encoder_input_topology_loss": encoder_input_topology.detach(),
        "encoder_input_topology_dice": encoder_input_topology_dice.detach(),
        "encoder_input_topology_mse": encoder_input_topology_mse.detach(),
        "student_pair_contrast_loss": student_pair_contrast_loss.detach(),
        "student_pair_contrast_penalty": student_pair_contrast_base.detach() if hasattr(student_pair_contrast_base, 'detach') else zero,
        "student_pair_contrast_count": student_pair_contrast_count.detach() if hasattr(student_pair_contrast_count, 'detach') else zero,
        "display_speckle_loss": display_speckle_loss.detach(),
        "display_speckle_penalty": display_speckle_base.detach() if hasattr(display_speckle_base, 'detach') else zero,
        "display_speckle_ratio": display_speckle_ratio.detach() if hasattr(display_speckle_ratio, 'detach') else zero,
        "display_highfreq_energy": display_highfreq_energy.detach() if hasattr(display_highfreq_energy, 'detach') else zero,
        "aux_detail_shape_loss": aux_detail_shape_loss.detach(),
        "aux_detail_shape_penalty": aux_detail_shape_base.detach(),
        "aux_detail_edge_loss": aux_detail_edge_loss.detach(),
        "aux_detail_edge_penalty": aux_detail_edge_base.detach(),
        "aux_detail_gate_fg_ratio": aux_detail_gate_fg_ratio.detach(),
        "aux_detail_active_fg_ratio": aux_detail_active_fg_ratio.detach(),
        "residual_l1_loss": residual_l1_loss.detach(),
        "residual_l1_penalty": residual_l1_base.detach(),
        "residual_background_loss": residual_background_loss.detach(),
        "residual_background_penalty": residual_background_base.detach(),
        "residual_background_mask_ratio": residual_background_mask_ratio.detach(),
        "generator_residual_abs_mean": generator_residual_abs_mean.detach(),
        "generator_residual_bg_abs_mean": generator_residual_bg_abs_mean.detach(),
        "student_base_mse": student_base_mse.detach(),
        "student_base_foreground_mse": student_base_foreground_mse.detach(),
        "student_base_background_mse": student_base_background_mse.detach(),
        "code_diversity_loss": code_diversity.detach(),
        "code_diversity_penalty": diversity_metrics["code_diversity_base"].detach(),
        "sample_code_diversity_loss": sample_code_diversity.detach(),
        "sample_code_diversity_penalty": diversity_metrics["code_diversity_base"].detach(),
        "teacher_pairwise_diversity_loss": teacher_pairwise_diversity.detach(),
        "teacher_pairwise_diversity_penalty": diversity_metrics["teacher_pairwise_diversity_base"].detach(),
        "teacher_pairwise_mse": diversity_metrics["teacher_pairwise_mse"].detach(),
        "student_pairwise_mse": diversity_metrics["student_pairwise_mse"].detach(),
        "diversity_ratio": diversity_metrics["diversity_ratio"].detach(),
        "diversity_pair_count": diversity_metrics["diversity_pair_count"].detach(),
        "cross_class_pairwise_loss": cross_class_pairwise.detach(),
        "cross_class_pairwise_penalty": cross_class_metrics["cross_class_pairwise_base"].detach(),
        "cross_class_teacher_pairwise_mse": cross_class_metrics["cross_class_teacher_pairwise_mse"].detach(),
        "cross_class_student_pairwise_mse": cross_class_metrics["cross_class_student_pairwise_mse"].detach(),
        "cross_class_diversity_ratio": cross_class_metrics["cross_class_diversity_ratio"].detach(),
        "cross_class_pair_count": cross_class_metrics["cross_class_pair_count"].detach(),
        "slm_pairwise_mse": slm_pairwise_mse.detach(),
        "slm_per_class_std_mean": slm_per_class_std_mean.detach(),
        "class_weight_mean": sample_weights.mean().detach(),
        "fg_pixel_ratio": foreground_mask.mean(),
    }
