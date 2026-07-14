"""Explicit optical-device contracts used by the formal propagation chain."""
from __future__ import annotations

import torch
from torch import nn


class IdealIdentityElement(nn.Module):
    """Complex-linear identity element, used for the current identity OLS contract."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self.name = name

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        return field


class ReflectiveAmplitudeSLM(nn.Module):
    """Apply an already calibrated complex SLM transfer to the incident field."""

    def forward(self, slm_transfer: torch.Tensor, incoming_field: torch.Tensor) -> torch.Tensor:
        # 电子 command 的标定发生在 student SLM 层；这里仅执行光学复振幅调制。
        return incoming_field * slm_transfer.to(incoming_field.dtype)
