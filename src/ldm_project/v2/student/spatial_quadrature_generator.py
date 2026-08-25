"""Teacher-free hierarchical amplitude generator for the v3 route.

The generator consumes a label and a 32D Gaussian code in one forward pass.
It has no training-image or prototype state and is intentionally independent
from the historical identity generator.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _FiLMUpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, code_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.film = nn.Linear(code_dim, 2 * out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, value: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        value = F.interpolate(value, scale_factor=2.0, mode="bilinear", align_corners=False)
        residual = self.residual(value)
        value = self.norm(self.conv(value))
        scale, bias = self.film(code).chunk(2, dim=1)
        value = value * (1.0 + scale[:, :, None, None]) + bias[:, :, None, None]
        return F.silu(value + residual)


class HierarchicalSpatialAmplitudeGenerator(nn.Module):
    """Single-pass label + 32D code -> 200x200 amplitude SLM command."""

    architecture_kind = "conditional_spatial_quadrature_v3"
    code_dim = 32
    global_code_dim = 8
    local_code_dim = 24

    def __init__(
        self,
        *,
        shape: tuple[int, int] = (200, 200),
        code_dim: int = 32,
        global_code_dim: int = 8,
        local_code_dim: int = 24,
        label_embedding_dim: int = 32,
        base_channels: int = 128,
    ) -> None:
        super().__init__()
        if shape != (200, 200) or (code_dim, global_code_dim, local_code_dim) != (32, 8, 24):
            raise ValueError("v3 spatial generator locks a 200x200 output and an 8+24=32D code")
        if label_embedding_dim != 32 or base_channels != 128:
            raise ValueError("v3 spatial generator locks 32D labels and 128 base channels")
        self.shape = shape
        self.label_embedding = nn.Embedding(10, label_embedding_dim)
        self.base = nn.Linear(label_embedding_dim + global_code_dim, base_channels * 25 * 25)
        self.blocks = nn.ModuleList(
            (
                _FiLMUpsampleBlock(base_channels, 64, 8),
                _FiLMUpsampleBlock(64, 32, 8),
                _FiLMUpsampleBlock(32, 16, 8),
            )
        )
        self.output = nn.Conv2d(16, 1, kernel_size=1)

    @staticmethod
    def _validate(code: torch.Tensor, labels: torch.Tensor) -> None:
        if code.ndim != 2 or code.shape[1] != 32 or not torch.isfinite(code).all():
            raise ValueError("v3 code must be finite with shape [batch, 32]")
        if labels.ndim != 1 or labels.shape[0] != code.shape[0] or labels.numel() == 0:
            raise ValueError("v3 labels must have shape [batch]")
        if labels.dtype not in (torch.int64, torch.int32) or labels.min() < 0 or labels.max() > 9:
            raise ValueError("v3 labels must be integers in [0, 9]")

    def split_code(self, code: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if code.ndim != 2 or code.shape[1] != self.code_dim:
            raise ValueError("v3 code must have shape [batch, 32]")
        return code[:, :8], (code[:, 8:16], code[:, 16:24], code[:, 24:32])

    def forward(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        self._validate(code, labels)
        global_code, local_codes = self.split_code(code)
        embedding = self.label_embedding(labels)
        value = self.base(torch.cat((embedding, global_code), dim=1))
        value = value.view(code.shape[0], 128, 25, 25)
        for block, local_code in zip(self.blocks, local_codes, strict=True):
            value = block(value, local_code)
        output = self.output(value)
        if output.shape[-2:] != self.shape:
            raise RuntimeError("v3 spatial generator produced an invalid SLM shape")
        return torch.sigmoid(output)

    def smoothness_penalty(self) -> torch.Tensor:
        value = self.output.weight.square().mean()
        for block in self.blocks:
            value = value + (
                (block.conv.weight[..., 1:, :] - block.conv.weight[..., :-1, :]).square().mean()
                + (block.conv.weight[..., :, 1:] - block.conv.weight[..., :, :-1]).square().mean()
            )
        return value


SpatialQuadratureGenerator = HierarchicalSpatialAmplitudeGenerator
