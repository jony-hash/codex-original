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


class IdealAmplitudeSLMModel(nn.Module):
    """Convert the electronic amplitude command into the formal complex SLM transfer."""

    def __init__(
        self, background_floor: float, residual_phase_scale_rad: float = 0.0
    ) -> None:
        super().__init__()
        if not 0.0 <= background_floor <= 1.0:
            raise ValueError(
                f"slm_background_floor must be in [0, 1], got {background_floor}"
            )
        self.background_floor = float(background_floor)
        self.residual_phase_scale_rad = float(residual_phase_scale_rad)

    def forward(self, slm_command: torch.Tensor) -> dict[str, torch.Tensor]:
        # slm_command 是归一化电子控制量；slm_amplitude 才是进入复光场的物理振幅。
        command = slm_command.clamp(0.0, 1.0)
        amplitude = self.background_floor + (1.0 - self.background_floor) * command
        phase = command.new_full(command.shape, self.residual_phase_scale_rad) * command
        return {
            "slm_command": command,
            "slm_amplitude": amplitude,
            "slm_phase": phase,
            "slm_transfer": torch.polar(amplitude, phase),
        }


def apply_identity_ols(field: torch.Tensor) -> torch.Tensor:
    """The formal OLS boundary is complex-linear identity, never a pointwise activation."""
    return field
