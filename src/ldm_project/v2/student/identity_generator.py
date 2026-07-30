"""Current identity-anchored amplitude-SLM generator.

This module intentionally imports neither prototype/capacity artifacts nor
diagnostic generator variants.  It is the only generator implementation that
belongs to the formal v2 runtime.
"""

from __future__ import annotations

import torch
from torch import nn


class IdentityAnchoredConditionalAmplitudeGenerator(nn.Module):
    """Identity-anchored conditional generator for the formal v2 path."""

    architecture_kind = "conditional_identity_logit_amplitude_v2"
    fixed_command_gain = 0.5

    def __init__(
        self,
        *,
        shape: tuple[int, int] = (200, 200),
        code_dim: int = 8,
        hidden_dim: int = 64,
        command_gain: float = fixed_command_gain,
        class_carrier_logits: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if (
            shape[0] <= 0
            or shape[1] <= 0
            or code_dim != 8
            or hidden_dim != 64
            or command_gain != self.fixed_command_gain
        ):
            raise ValueError(
                "identity-anchored generator requires 8D code, 64 hidden units, and fixed gain 0.5"
            )
        self.shape = shape
        self.code_dim = code_dim
        self.hidden_dim = hidden_dim
        self.command_gain = self.fixed_command_gain
        if class_carrier_logits is None:
            carrier = torch.zeros(10, *shape)
        else:
            if tuple(class_carrier_logits.shape) != (10, *shape):
                raise ValueError("class_carrier_logits must have shape [10, H, W]")
            carrier = class_carrier_logits.detach().clone()
        self.class_carrier_logits = nn.Parameter(carrier)
        self.global_modes = nn.Parameter(self._initial_global_modes(shape))
        self.code_projection = nn.Linear(code_dim, hidden_dim, bias=False)
        self.class_heads = nn.Parameter(torch.zeros(10, code_dim, hidden_dim))
        nn.init.kaiming_uniform_(self.code_projection.weight, a=5**0.5)

    @staticmethod
    def _initial_global_modes(shape: tuple[int, int]) -> torch.Tensor:
        height, width = shape
        yy = torch.linspace(-1.0, 1.0, height).view(height, 1)
        xx = torch.linspace(-1.0, 1.0, width).view(1, width)
        modes = torch.stack(
            (
                torch.cos(torch.pi * xx).expand(height, width),
                torch.sin(torch.pi * xx).expand(height, width),
                torch.cos(torch.pi * yy).expand(height, width),
                torch.sin(torch.pi * yy).expand(height, width),
                torch.cos(torch.pi * (xx + yy)),
                torch.sin(torch.pi * (xx + yy)),
                torch.cos(torch.pi * (xx - yy)),
                torch.sin(torch.pi * (xx - yy)),
            )
        )
        return IdentityAnchoredConditionalAmplitudeGenerator.normalized_modes(modes)

    @staticmethod
    def normalized_modes(modes: torch.Tensor) -> torch.Tensor:
        if modes.ndim != 3:
            raise ValueError("global modes must have shape [code_dim, H, W]")
        centered = modes - modes.mean(dim=(-2, -1), keepdim=True)
        rms = centered.square().mean(dim=(-2, -1), keepdim=True).clamp_min(1.0e-8).sqrt()
        return centered / rms

    @property
    def Wz(self) -> nn.Linear:
        return self.code_projection

    @property
    def Wy(self) -> torch.Tensor:
        return self.class_heads

    @property
    def By(self) -> torch.Tensor:
        return self.class_carrier_logits

    @property
    def P(self) -> torch.Tensor:
        return self.global_modes

    def _validate_inputs(self, code: torch.Tensor, labels: torch.Tensor) -> None:
        if code.ndim != 2 or code.shape[1] != self.code_dim:
            raise ValueError("code must have shape [batch, 8]")
        if labels.ndim != 1 or labels.shape[0] != code.shape[0] or labels.numel() == 0:
            raise ValueError("labels must have shape [batch]")
        if labels.min() < 0 or labels.max() > 9:
            raise ValueError("labels must be integers in [0, 9]")

    def coefficients(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self._validate_inputs(code, labels)
        hidden = torch.nn.functional.silu(self.code_projection(code))
        residual = torch.einsum("bkh,bh->bk", self.class_heads.index_select(0, labels), hidden)
        return torch.tanh(code + residual)

    def forward(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        coefficients = self.coefficients(code, labels)
        residual = torch.einsum("bk,khw->bhw", coefficients, self.normalized_modes(self.global_modes))
        logits = self.class_carrier_logits.index_select(0, labels) + self.command_gain * residual
        return torch.sigmoid(logits).unsqueeze(1)

    def mode_orthogonality_penalty(self) -> torch.Tensor:
        modes = self.normalized_modes(self.global_modes).flatten(1)
        gram = modes @ modes.T / modes.shape[1]
        return (gram - torch.eye(self.code_dim, device=modes.device, dtype=modes.dtype)).square().mean()

    def class_carrier_command(self, labels: torch.Tensor) -> torch.Tensor:
        if (
            labels.ndim != 1
            or labels.numel() == 0
            or labels.min() < 0
            or labels.max() > 9
        ):
            raise ValueError("labels must be integers in [0, 9]")
        return torch.sigmoid(self.class_carrier_logits.index_select(0, labels)).unsqueeze(1)


ConditionalIdentityLogitAmplitudeGenerator = IdentityAnchoredConditionalAmplitudeGenerator
