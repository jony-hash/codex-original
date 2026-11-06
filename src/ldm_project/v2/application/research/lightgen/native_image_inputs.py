from __future__ import annotations

import torch
import torch.nn.functional as F


def native_image_size(arm: str) -> int:
    return 200 if arm == "small" else 400


def image_batch(
    pixels: torch.Tensor, rows: torch.Tensor, arm: str, device: torch.device
) -> torch.Tensor:
    return F.interpolate(
        pixels.index_select(0, rows.detach().cpu()).unsqueeze(1).to(device=device, dtype=torch.float32) / 255.0,
        size=(native_image_size(arm), native_image_size(arm)),
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)
