"""Diagnostic electronic seed families for the capacity-map experiment."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ShapeMNISTEvaluator(nn.Module):
    """Third, diagnostic-only evaluator with a pooling-free feature path.

    This architecture is intentionally different from both qualified v2.1 CNNs.
    It is never an acceptance evaluator; it resolves disagreement between the
    frozen primary and smooth bundle without relaxing either hard gate.
    """

    architecture_kind = "shape_mnist_evaluator_v22"

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.embedding = nn.Sequential(nn.Linear(96, 128), nn.GELU())
        self.classifier = nn.Linear(128, 10)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.embedding(self.features(images))
        return features, self.classifier(features)

    def freeze(self) -> "ShapeMNISTEvaluator":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


class SpatialSeedDecoder(nn.Module):
    """The pre-registered E2--E4 nonlinear spatial seed expansion.

    The subtraction is deliberately inside the logits: zero random code is an
    exact class carrier regardless of convolutional biases.
    """

    def __init__(self, code_dim: int, *, shape: tuple[int, int] = (200, 200)) -> None:
        super().__init__()
        if code_dim not in {8, 16, 32} or shape != (200, 200):
            raise ValueError("spatial capacity-map decoder requires 8/16/32D and 200x200 commands")
        self.code_dim, self.shape = code_dim, shape
        self.embedding = nn.Embedding(10, 16)
        self.projection = nn.Linear(code_dim + 16, 32 * 25 * 25)
        self.convolutions = nn.ModuleList((nn.Conv2d(32, 16, 3, padding=1), nn.Conv2d(16, 8, 3, padding=1), nn.Conv2d(8, 1, 3, padding=1)))
        self.class_carrier_logits = nn.Parameter(torch.zeros(10, 1, *shape))
        if sum(parameter.numel() for parameter in self.parameters()) > 1_500_000:
            raise AssertionError("capacity-map spatial seed decoder exceeded 1.5M parameters")

    def _residual(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        value = torch.cat((code, self.embedding(labels)), dim=1)
        value = self.projection(value).view(-1, 32, 25, 25)
        for convolution in self.convolutions:
            value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
            value = F.silu(convolution(value))
        return value

    def forward(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if code.ndim != 2 or code.shape[1] != self.code_dim:
            raise ValueError("code dimension does not match the declared electronic variant")
        if labels.ndim != 1 or labels.shape[0] != code.shape[0]:
            raise ValueError("labels must be one label for each code")
        residual = self._residual(code, labels) - self._residual(torch.zeros_like(code), labels)
        return torch.sigmoid(self.class_carrier_logits.index_select(0, labels) + residual)
