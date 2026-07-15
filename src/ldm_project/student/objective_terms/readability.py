"""Readability objective terms and shared display-free image primitives."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ._shared import (
    _camera_artifact_metrics,
    _foreground_dice_score,
    _masked_mean_per_sample,
    _zero_like_loss,
)


def gaussian_blur_2d(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
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


# 兼容旧的 objective 私有导入；新 term 只使用公开名称。
_gaussian_blur_2d = gaussian_blur_2d

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
    g = torch.arange(
        -kernel_size // 2 + 1,
        kernel_size // 2 + 1,
        device=camera_prediction.device,
        dtype=camera_prediction.dtype,
    )
    g = torch.exp(-g.pow(2) / (2.0 * 0.8 * 0.8))
    g = g / g.sum()
    g_row = g.view(1, 1, 1, -1)
    g_col = g.view(1, 1, -1, 1)
    target_padded = F.pad(target_batch, (pad, pad, pad, pad), mode="replicate")
    teacher_smoothed = F.conv2d(
        target_padded,
        g_row.expand(target_batch.shape[1], 1, 1, -1),
        groups=target_batch.shape[1],
    )
    teacher_smoothed = F.conv2d(
        teacher_smoothed,
        g_col.expand(target_batch.shape[1], 1, -1, 1),
        groups=target_batch.shape[1],
    )
    squared_diff = (camera_prediction - teacher_smoothed.detach()).pow(2)
    fg_weight = 3.0
    per_sample_mse = (
        squared_diff.flatten(1) * (foreground_mask.flatten(1) * (fg_weight - 1.0) + 1.0)
    ).mean(dim=1)
    readability_base = per_sample_mse.mean()
    return {
        "readability_base": readability_base,
        "readability_penalty": readability_base.detach(),
    }

def _camera_background_speckle_metrics(
    camera_prediction: torch.Tensor,
    background_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    zero = _zero_like_loss(camera_prediction)
    if camera_prediction.shape[0] == 0:
        return {"background_speckle_base": zero, "background_speckle_ratio": zero}
    prediction = camera_prediction.clamp_min(0.0)
    bg_pixels = background_mask.flatten(1).sum(dim=1).clamp_min(1.0)
    bg_mean = (prediction.flatten(1) * background_mask.flatten(1)).sum(
        dim=1
    ) / bg_pixels
    bg_centered = (prediction - bg_mean.view(-1, 1, 1, 1)) * background_mask
    bg_std = (bg_centered.flatten(1).pow(2).sum(dim=1) / bg_pixels).sqrt()
    base = (bg_mean + 0.5 * bg_std).mean()
    ratio = ((prediction > 0.12).float() * background_mask).flatten(1).sum(
        dim=1
    ) / bg_pixels
    return {
        "background_speckle_base": base,
        "background_speckle_ratio": ratio.mean().detach(),
    }

def _camera_background_hotspot_metrics(
    camera_prediction: torch.Tensor,
    background_mask: torch.Tensor,
    threshold: float,
) -> dict[str, torch.Tensor]:
    prediction = camera_prediction.clamp(0.0, 1.0)
    hotspot = F.relu(prediction - float(threshold))
    base = _masked_mean_per_sample(hotspot.pow(2), background_mask).mean()
    ratio = _masked_mean_per_sample(
        (prediction > float(threshold)).float(), background_mask
    ).mean()
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

def _camera_hotspot_ratio(camera_prediction: torch.Tensor) -> torch.Tensor:
    return (camera_prediction > 1.0).float().mean()

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
        "background_speckle_loss": background_speckle_weight
        * speckle_metrics["background_speckle_base"],
        "background_speckle_ratio": speckle_metrics["background_speckle_ratio"],
        "background_hotspot_base": hotspot_metrics["background_hotspot_base"],
        "background_hotspot_loss": background_hotspot_weight
        * hotspot_metrics["background_hotspot_base"],
        "background_hotspot_ratio": hotspot_metrics["background_hotspot_ratio"],
        "background_highfreq_base": highfreq_metrics["background_highfreq_base"],
        "background_highfreq_loss": background_highfreq_weight
        * highfreq_metrics["background_highfreq_base"],
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
    speckle_metrics = _camera_background_speckle_metrics(
        camera_loss_prediction, background_mask
    )
    hotspot_metrics = _camera_background_hotspot_metrics(
        camera_prediction,
        background_mask,
        background_hotspot_threshold,
    )
    highfreq_metrics = _camera_background_highfreq_metrics(
        camera_prediction, background_mask
    )
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
        "background_speckle_loss": background_speckle_weight
        * speckle_metrics["background_speckle_base"],
        "background_hotspot_base": hotspot_metrics["background_hotspot_base"],
        "background_hotspot_ratio": hotspot_metrics["background_hotspot_ratio"],
        "background_hotspot_loss": background_hotspot_weight
        * hotspot_metrics["background_hotspot_base"],
        "background_highfreq_base": highfreq_metrics["background_highfreq_base"],
        "background_highfreq_energy": highfreq_metrics["background_highfreq_energy"],
        "background_highfreq_loss": background_highfreq_weight
        * highfreq_metrics["background_highfreq_base"],
        "readable_contrast_base": contrast_metrics["readable_contrast_base"],
        "readable_contrast": contrast_metrics["readable_contrast"],
        "readable_contrast_loss": readable_contrast_weight
        * contrast_metrics["readable_contrast_base"],
    }
