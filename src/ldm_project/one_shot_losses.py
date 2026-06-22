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
    labels = class_labels.detach().reshape(-1).to(device=camera_prediction.device)
    if labels.shape[0] != camera_prediction.shape[0]:
        raise ValueError("class_labels must match one_shot_loss batch size for student label margin.")
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

    sample_features = _normalize_for_label_margin(camera_prediction)
    prototype_features = _normalize_for_label_margin(prototype_batch)
    correlations = sample_features @ prototype_features.t()
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
    student_label_margin_weight: float = 0.0,
    student_label_margin_target: float = 0.03,
    student_label_margin_labels: tuple[int, ...] = (),
    student_label_margin_focus_label: int | None = None,
    student_label_margin_focus_weight: float = 1.0,
    student_label_margin_prototypes: torch.Tensor | None = None,
    student_pair_contrast_config: tuple[tuple[int, int, float], ...] = (),
    student_display_speckle_weight: float = 0.0,
    aux_detail_edge_weight: float = 0.0,
    aux_detail_shape_weight: float = 0.0,
    aux_detail_foreground_gate_dilation: int = 0,
    residual_l1_weight: float = 0.0,
    residual_background_weight: float = 0.0,
    code_diversity_weight: float = 0.0,
    teacher_pairwise_diversity_weight: float = 0.0,
    cross_class_pairwise_weight: float = 0.0,
    diversity_min_teacher_mse: float = 1.0e-4,
) -> dict[str, torch.Tensor]:
    foreground_mask = (target_batch > foreground_threshold).float()
    background_mask = 1.0 - foreground_mask
    camera_prediction = outputs["camera_plane"]
    camera_loss_prediction = outputs.get("camera_plane_loss", camera_prediction)
    sample_weights = _resolve_sample_weights(
        target_batch,
        class_labels,
        class_recon_weight_labels,
        class_recon_weight,
    )
    has_class_weights = class_labels is not None and bool(class_recon_weight_labels) and class_recon_weight != 1.0
    if has_class_weights:
        camera_metrics = _weighted_reconstruction_metrics_for_samples(
            camera_prediction,
            target_batch,
            foreground_mask,
            background_mask,
            foreground_weight,
            background_weight,
            sample_weights,
        )
    else:
        camera_metrics = _weighted_reconstruction_metrics(
            camera_prediction,
            target_batch,
            foreground_mask,
            background_mask,
            foreground_weight,
            background_weight,
        )
    zero = _zero_like_loss(camera_prediction)
    # Clean target: derive clean readable mask from teacher_camera for structure supervision.
    if clean_target_enabled:
        _clean_result = _build_clean_readable_target(target_batch.detach())
        clean_target = _clean_result["clean_target"]
        clean_target_fg_ratio = _clean_result["clean_target_fg_ratio"]
        clean_foreground_mask = clean_target.float()
        clean_background_mask = 1.0 - clean_foreground_mask
    else:
        clean_target = target_batch
        clean_foreground_mask = foreground_mask
        clean_background_mask = background_mask
        clean_target_fg_ratio = zero

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
    # 可读性正则项：用平滑教师监督学生输出，抑制散斑噪声。
    # Clean-target losses: structural supervision from derived clean mask.
    if clean_target_enabled:
        clean_dice_base = 1.0 - _foreground_dice_score(camera_prediction, clean_foreground_mask)
        clean_dice_loss = clean_foreground_dice_weight * clean_dice_base
        clean_hole_metrics = _camera_artifact_metrics(
            camera_loss_prediction, clean_target, clean_foreground_mask
        )
        clean_fg_hole_base = clean_hole_metrics["artifact_base"]
        clean_fg_hole_ratio = clean_hole_metrics["foreground_hole_ratio"]
        clean_fg_hole_loss = clean_foreground_hole_weight * clean_fg_hole_base
        clean_bg_speckle_metrics = _camera_background_speckle_metrics(
            camera_loss_prediction, clean_background_mask
        )
        clean_bg_speckle_base = clean_bg_speckle_metrics["background_speckle_base"]
        clean_bg_speckle_ratio = clean_bg_speckle_metrics["background_speckle_ratio"]
        clean_bg_speckle_loss = clean_background_speckle_weight * clean_bg_speckle_base
        clean_bg_hotspot_metrics = _camera_background_hotspot_metrics(
            camera_prediction, clean_background_mask, camera_background_hotspot_threshold
        )
        clean_bg_hotspot_base = clean_bg_hotspot_metrics["background_hotspot_base"]
        clean_bg_hotspot_ratio = clean_bg_hotspot_metrics["background_hotspot_ratio"]
        clean_bg_hotspot_loss = clean_background_hotspot_weight * clean_bg_hotspot_base
        clean_bg_highfreq_metrics = _camera_background_highfreq_metrics(
            camera_prediction, clean_background_mask
        )
        clean_bg_highfreq_base = clean_bg_highfreq_metrics["background_highfreq_base"]
        clean_bg_highfreq_energy = clean_bg_highfreq_metrics["background_highfreq_energy"]
        clean_bg_highfreq_loss = clean_background_highfreq_weight * clean_bg_highfreq_base
    else:
        clean_dice_base = zero
        clean_dice_loss = zero
        clean_fg_hole_base = zero
        clean_fg_hole_loss = zero
        clean_fg_hole_ratio = zero
        clean_bg_speckle_base = zero
        clean_bg_speckle_loss = zero
        clean_bg_speckle_ratio = zero
        clean_bg_hotspot_base = zero
        clean_bg_hotspot_loss = zero
        clean_bg_hotspot_ratio = zero
        clean_bg_highfreq_base = zero
        clean_bg_highfreq_loss = zero
        clean_bg_highfreq_energy = zero

    if camera_readability_weight != 0.0:
        readability_metrics = _camera_readability_metrics(
            camera_loss_prediction, target_batch, foreground_mask
        )
        camera_readability_base = readability_metrics["readability_base"]
        camera_readability_penalty = readability_metrics["readability_penalty"]
    else:
        camera_readability_base = zero
        camera_readability_penalty = zero
    camera_readability = camera_readability_weight * camera_readability_base
    background_speckle_metrics = _camera_background_speckle_metrics(camera_loss_prediction, background_mask)
    camera_background_speckle_base = background_speckle_metrics["background_speckle_base"]
    background_speckle_ratio = background_speckle_metrics["background_speckle_ratio"]
    camera_background_speckle = camera_background_speckle_weight * camera_background_speckle_base
    background_hotspot_metrics = _camera_background_hotspot_metrics(
        camera_prediction,
        background_mask,
        camera_background_hotspot_threshold,
    )
    camera_background_hotspot_base = background_hotspot_metrics["background_hotspot_base"]
    background_hotspot_ratio = background_hotspot_metrics["background_hotspot_ratio"]
    camera_background_hotspot = camera_background_hotspot_weight * camera_background_hotspot_base
    background_highfreq_metrics = _camera_background_highfreq_metrics(camera_prediction, background_mask)
    camera_background_highfreq_base = background_highfreq_metrics["background_highfreq_base"]
    background_highfreq_energy = background_highfreq_metrics["background_highfreq_energy"]
    camera_background_highfreq = camera_background_highfreq_weight * camera_background_highfreq_base
    readable_contrast_metrics = _camera_readable_contrast_metrics(
        camera_loss_prediction,
        foreground_mask,
        background_mask,
        camera_readable_contrast_target,
    )
    camera_readable_contrast_base = readable_contrast_metrics["readable_contrast_base"]
    camera_readable_contrast_value = readable_contrast_metrics["readable_contrast"]
    camera_readable_contrast = camera_readable_contrast_weight * camera_readable_contrast_base
    if student_label_margin_weight != 0.0 or student_label_margin_prototypes is not None:
        if student_label_margin_weight != 0.0 and student_label_margin_prototypes is None:
            raise ValueError("student_label_margin_weight requires student_label_margin_prototypes.")
        student_label_margin_metrics = _student_label_margin_metrics(
            camera_prediction,
            class_labels,
            student_label_margin_prototypes,
            student_label_margin_labels,
            student_label_margin_target,
            student_label_margin_focus_label,
            student_label_margin_focus_weight,
        )
        student_label_margin_base = student_label_margin_metrics["student_label_margin_base"]
        student_label_margin_mean = student_label_margin_metrics["student_label_margin_mean"]
        student_label_correct_ratio = student_label_margin_metrics["student_label_correct_ratio"]
    else:
        student_label_margin_base = zero
        student_label_margin_mean = zero
        student_label_correct_ratio = zero
    student_label_margin = student_label_margin_weight * student_label_margin_base
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
    gate_dilation = int(aux_detail_foreground_gate_dilation)
    if gate_dilation > 0:
        aux_detail_gate_mask = _dilate_foreground_mask(clean_foreground_mask, gate_dilation)
        aux_detail_gate_fg_ratio = aux_detail_gate_mask.mean().detach()
    else:
        aux_detail_gate_mask = None
        aux_detail_gate_fg_ratio = zero
    if aux_detail_target is not None and (aux_detail_shape_weight != 0.0 or aux_detail_edge_weight != 0.0):
        if aux_detail_target.shape != target_batch.shape:
            raise ValueError(
                "aux_detail_target must match target_batch shape, "
                f"got {tuple(aux_detail_target.shape)} and {tuple(target_batch.shape)}"
            )
        aux_detail = aux_detail_target.float().clamp(0.0, 1.0)
        aux_detail_fg = (aux_detail > foreground_threshold).float()
        if aux_detail_gate_mask is not None:
            aux_detail = aux_detail * aux_detail_gate_mask
            aux_detail_fg = aux_detail_fg * aux_detail_gate_mask
        aux_detail_active_fg_ratio = aux_detail_fg.mean().detach()
        has_aux_detail_foreground = bool(aux_detail_fg.detach().sum().item() > 0)
        aux_detail_shape_base = (
            _foreground_shape_loss_for_samples(camera_prediction, aux_detail_fg, sample_weights)
            if aux_detail_shape_weight != 0.0 and has_aux_detail_foreground
            else zero
        )
        aux_detail_edge_base = (
            _one_shot_edge_shape_loss(camera_prediction, aux_detail, aux_detail_fg, sample_weights)
            if aux_detail_edge_weight != 0.0 and has_aux_detail_foreground
            else zero
        )
    else:
        aux_detail_shape_base = zero
        aux_detail_edge_base = zero
        aux_detail_active_fg_ratio = zero
    aux_detail_shape_loss = aux_detail_shape_weight * aux_detail_shape_base
    aux_detail_edge_loss = aux_detail_edge_weight * aux_detail_edge_base

    residual_map = outputs.get("generator_residual_map")
    if residual_map is not None and (residual_l1_weight != 0.0 or residual_background_weight != 0.0):
        residual_abs = residual_map.float().abs()
        residual_l1_base = residual_abs.mean()
        residual_background_mask = (1.0 - aux_detail_gate_mask).clamp(0.0, 1.0) if aux_detail_gate_mask is not None else clean_background_mask
        residual_background_base = _masked_mean(residual_abs, residual_background_mask)
        residual_background_mask_ratio = residual_background_mask.mean().detach()
    else:
        residual_l1_base = zero
        residual_background_base = zero
        residual_background_mask_ratio = zero
    residual_l1_loss = residual_l1_weight * residual_l1_base
    residual_background_loss = residual_background_weight * residual_background_base
    generator_residual_abs_mean = (
        outputs["generator_residual_abs_mean"].detach().float().mean()
        if "generator_residual_abs_mean" in outputs
        else zero
    )
    generator_residual_bg_abs_mean = (
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
    diversity_metrics = _one_shot_pairwise_diversity_metrics(
        camera_prediction,
        target_batch,
        class_labels,
        diversity_min_teacher_mse,
    )
    code_diversity = code_diversity_weight * diversity_metrics["code_diversity_base"]
    teacher_pairwise_diversity = (
        teacher_pairwise_diversity_weight * diversity_metrics["teacher_pairwise_diversity_base"]
    )
    cross_class_metrics = _one_shot_cross_class_pairwise_metrics(
        camera_prediction,
        target_batch,
        class_labels,
        diversity_min_teacher_mse,
    )
    cross_class_pairwise = cross_class_pairwise_weight * cross_class_metrics["cross_class_pairwise_base"]
    slm_pairwise_mse = _pairwise_mse_summary(slm_amplitude)
    slm_per_class_std_mean = _per_class_std_summary(slm_amplitude, class_labels)
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
        + code_diversity
        + teacher_pairwise_diversity
        + cross_class_pairwise
        + clean_dice_loss
        + clean_fg_hole_loss
        + clean_bg_speckle_loss
        + clean_bg_hotspot_loss
        + clean_bg_highfreq_loss
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
