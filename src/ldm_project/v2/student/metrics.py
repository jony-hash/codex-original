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


def prototype_correlation_matrix(prediction: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Full-frame ten-by-ten prototype correlations; never an ROI substitute."""
    if prediction.shape[0] != 10 or targets.shape[0] != 10:
        raise ValueError("prototype correlation matrix requires ten labels")
    rows = prediction[:, None].expand(10, 10, *prediction.shape[1:]).reshape(100, *prediction.shape[1:])
    columns = targets[None].expand(10, 10, *targets.shape[1:]).reshape(100, *targets.shape[1:])
    return correlation(rows, columns).reshape(10, 10)


def prototype_margins(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape != (10, 10):
        raise ValueError("prototype margins require a ten-by-ten matrix")
    return matrix.diag() - (matrix + torch.eye(10, device=matrix.device, dtype=matrix.dtype) * -2.0).max(dim=1).values


def foreground_hole_ratio(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
    foreground = target >= threshold
    return ((prediction < threshold) & foreground).to(torch.float32).sum(dim=(-2, -1)) / foreground.to(torch.float32).sum(dim=(-2, -1)).clamp_min(1.0)


def background_speckle_ratio(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.2) -> torch.Tensor:
    background = target < threshold
    return ((prediction >= threshold) & background).to(torch.float32).sum(dim=(-2, -1)) / background.to(torch.float32).sum(dim=(-2, -1)).clamp_min(1.0)
