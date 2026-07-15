"""Neutral target transforms shared by teacher payloads and student objectives."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def build_clean_readable_target(teacher_camera: torch.Tensor) -> dict[str, torch.Tensor]:
    """Derive the deterministic clean-target view used by offline distillation.

    This transformation belongs to the payload contract rather than to a student
    loss: teacher filtering and student supervision must produce identical
    targets without either domain importing the other.
    """
    batch, channels, height, width = teacher_camera.shape
    flat = teacher_camera.reshape(batch, channels, -1)
    q_low = flat.quantile(0.02, dim=-1, keepdim=True)
    q_high = flat.quantile(0.98, dim=-1, keepdim=True)
    normalized = (teacher_camera - q_low.view(batch, channels, 1, 1)) / (
        (q_high - q_low).view(batch, channels, 1, 1).clamp_min(1e-6)
    )
    sigma = 3.0
    kernel_size = max(3, int(round(sigma * 3)) // 2 * 2 + 1)
    coords = torch.arange(
        -kernel_size // 2 + 1,
        kernel_size // 2 + 1,
        device=teacher_camera.device,
        dtype=teacher_camera.dtype,
    )
    kernel = torch.exp(-coords.pow(2) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    kernel_row = kernel.view(1, 1, 1, -1)
    kernel_col = kernel.view(1, 1, -1, 1)
    padded = F.pad(
        normalized.clamp(0.0, 1.0),
        (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2),
        mode="replicate",
    )
    blurred = F.conv2d(padded, kernel_row.expand(channels, 1, 1, -1), groups=channels)
    blurred = F.conv2d(blurred, kernel_col.expand(channels, 1, -1, 1), groups=channels)
    mask = torch.sigmoid((blurred - 0.30) * 10.0)
    close_kernel = max(3, int(round(0.04 * min(height, width)) // 2 * 2 + 1))
    closed = -F.max_pool2d(
        -F.max_pool2d(mask, kernel_size=close_kernel, stride=1, padding=close_kernel // 2),
        kernel_size=close_kernel,
        stride=1,
        padding=close_kernel // 2,
    )
    open_kernel = max(3, int(round(0.03 * min(height, width)) // 2 * 2 + 1))
    opened = F.max_pool2d(
        -F.max_pool2d(-closed, kernel_size=open_kernel, stride=1, padding=open_kernel // 2),
        kernel_size=open_kernel,
        stride=1,
        padding=open_kernel // 2,
    )
    return {
        "clean_target": opened,
        "clean_target_fg_ratio": (opened > 0.5).float().mean().detach(),
    }
