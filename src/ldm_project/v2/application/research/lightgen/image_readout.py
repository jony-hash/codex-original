from __future__ import annotations

import torch
import torch.nn.functional as F


def resize_view(raw: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28))


def ssim_batch(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    coordinate = torch.arange(-3, 4, dtype=torch.float64, device=image.device)
    kernel_1d = torch.exp(-coordinate.square() / (2.0 * 1.5**2))
    kernel_1d /= kernel_1d.sum()
    kernel = (kernel_1d[:, None] * kernel_1d[None, :]).to(image.dtype)[None, None]
    x = F.conv2d(image, kernel, padding=3)
    y = F.conv2d(target, kernel, padding=3)
    x2 = F.conv2d(image.square(), kernel, padding=3) - x.square()
    y2 = F.conv2d(target.square(), kernel, padding=3) - y.square()
    xy = F.conv2d(image * target, kernel, padding=3) - x * y
    return (
        ((2.0 * x * y + 0.01**2) * (2.0 * xy + 0.03**2))
        / ((x.square() + y.square() + 0.01**2) * (x2 + y2 + 0.03**2))
    ).mean((1, 2, 3))


def structural_loss_from_raw(
    raw_w_per_pixel: torch.Tensor, gain: torch.Tensor, target28: torch.Tensor,
) -> torch.Tensor:
    view = resize_view(
        raw_w_per_pixel * gain.to(device=raw_w_per_pixel.device, dtype=raw_w_per_pixel.dtype)
    )
    if not view.shape == target28.shape:
        raise RuntimeError("structural production view and original target shapes differ")
    value = (1.0 - ssim_batch(view, target28)).mean()
    if not torch.isfinite(value).item():
        raise RuntimeError("SSIM structural loss is nonfinite")
    return value
