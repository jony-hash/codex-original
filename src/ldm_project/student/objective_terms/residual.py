"""residual objective terms."""
# ruff: noqa: F403,F401,F405
from __future__ import annotations

from ._shared import *

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
        mask = labels == int(lab)
        if mask.any():
            class_means[int(lab)] = features[mask].mean(dim=0)
            class_means[int(lab)] = F.normalize(
                class_means[int(lab)].unsqueeze(0), dim=1
            ).squeeze(0)

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
        "student_pair_contrast_count": torch.tensor(
            float(active_pairs), device=camera_prediction.device
        ),
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
    bright_bg = (
        clamped > float(speckle_brightness_threshold)
    ).float() * background_mask
    speckle_ratio = bright_bg.flatten(1).sum(dim=1) / background_mask.flatten(1).sum(
        dim=1
    ).clamp_min(1.0)
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
        student_base_foreground_mse = _masked_mean(
            student_base_error, foreground_mask
        ).detach()
        student_base_background_mse = _masked_mean(
            student_base_error, background_mask
        ).detach()
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
