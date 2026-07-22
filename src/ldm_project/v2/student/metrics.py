"""Raw-CMOS metrics; display normalization is deliberately absent."""
from __future__ import annotations

import torch


def correlation(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = prediction.flatten(1)
    truth = target.flatten(1)
    pred = pred - pred.mean(dim=1, keepdim=True)
    truth = truth - truth.mean(dim=1, keepdim=True)
    return (pred * truth).sum(dim=1) / (pred.square().sum(dim=1).sqrt() * truth.square().sum(dim=1).sqrt()).clamp_min(1.0e-12)


def foreground_dice(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
    pred = prediction >= threshold
    truth = target >= threshold
    intersection = (pred & truth).sum(dim=(-2, -1)).to(torch.float32)
    return (2.0 * intersection + 1.0e-6) / (pred.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1)) + 1.0e-6)


def high_saturation_ratio(raw_camera: torch.Tensor, exposure_scale: float) -> torch.Tensor:
    return (raw_camera / exposure_scale > 1.0).to(torch.float32).mean()
