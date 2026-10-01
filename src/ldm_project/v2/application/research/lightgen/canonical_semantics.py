"""Numerical helpers shared by LightGen canonical semantic objectives."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} is nonfinite")


def norm(value: torch.Tensor, name: str, *, allow_zero: bool = False) -> float:
    finite(value, name)
    result = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    if not math.isfinite(result) or (not allow_zero and result <= 0.0):
        raise ValueError(f"{name} is zero or invalid")
    return result


def rms(value: torch.Tensor, name: str, *, allow_zero: bool = False) -> float:
    finite(value, name)
    result = float(value.detach().to(torch.float64).square().mean().sqrt().cpu())
    if not math.isfinite(result) or (not allow_zero and result <= 0.0):
        raise ValueError(f"{name} RMS is zero or invalid")
    return result


def rms_normalize(value: torch.Tensor, name: str) -> tuple[torch.Tensor, float]:
    """Scale a finite direction to unit RMS while preserving its dtype/device."""

    scale = rms(value, name)
    return value / scale, scale


def dot(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sum(left.detach().to(torch.float64) * right.detach().to(torch.float64)).cpu())


def cosine(left: torch.Tensor, right: torch.Tensor, name: str) -> float:
    value = dot(left, right) / (norm(left, f"{name} left") * norm(right, f"{name} right"))
    if not math.isfinite(value):
        raise ValueError(f"{name} is nonfinite")
    return value


def gradient_stats(value: torch.Tensor, name: str) -> dict[str, Any]:
    finite(value, name)
    return {
        "norm": norm(value, name),
        "rms": rms(value, name),
        "shape": list(value.shape),
        "finite_fraction": float(torch.isfinite(value).to(torch.float64).mean().cpu()),
        "zero_fraction": float((value == 0.0).to(torch.float64).mean().cpu()),
    }


def cmos_view(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4 or value.shape[1] != 1 or tuple(value.shape[-2:]) not in {(28, 28), (200, 200)}:
        raise ValueError("CMOS view shape changed")
    finite(value, "CMOS view input")
    result = F.adaptive_avg_pool2d(value.clamp(0.0, 1.0), (28, 28))
    finite(result, "CMOS view")
    return result


def normalized_feature_loss(
    generated: torch.Tensor,
    target: torch.Tensor,
    name: str,
    *,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    if generated.shape != target.shape or generated.ndim != 2 or generated.shape[1] != 128:
        raise ValueError(f"{name} feature shape changed")
    result = (
        F.normalize(generated, dim=1, eps=epsilon)
        - F.normalize(target.detach(), dim=1, eps=epsilon)
    ).square().sum(dim=1).mean()
    finite(result, name)
    return result


def project_auxiliary_gradient(
    auxiliary: torch.Tensor,
    semantic: torch.Tensor,
    name: str,
    *,
    epsilon: float = 1.0e-12,
    absolute_tolerance: float = 1.0e-8,
    relative_tolerance: float = 1.0e-5,
    norm_floor: float = 1.0e-12,
) -> tuple[torch.Tensor, dict[str, Any]]:
    auxiliary_norm = norm(auxiliary, f"{name} weighted gradient")
    semantic_norm = norm(semantic, "semantic gradient")
    dot_before = dot(auxiliary, semantic)
    projected = dot_before < 0.0
    removed = torch.zeros_like(auxiliary)
    if projected:
        coefficient = dot_before / (
            float(torch.sum(semantic.detach().to(torch.float64).square()).cpu()) + epsilon
        )
        removed = semantic * coefficient
        safe = auxiliary - removed
    else:
        safe = auxiliary
    finite(safe, f"{name}_safe")
    dot_after = dot(safe, semantic)
    if projected:
        tolerance = max(
            absolute_tolerance,
            relative_tolerance * max(norm(safe, f"{name}_safe"), norm_floor) * semantic_norm,
        )
        if abs(dot_after) > tolerance:
            raise ValueError(f"{name}_safe is not orthogonal to semantic direction")
    removed_norm = float(removed.detach().to(torch.float64).square().sum().sqrt().cpu())
    return safe, {
        "dot_before": dot_before,
        "dot_after": dot_after,
        "projected": projected,
        "removed_norm": removed_norm,
        "projection_fraction_norm": removed_norm / auxiliary_norm,
        "eps_proj": epsilon,
    }
