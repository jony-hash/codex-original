"""Teacher-free electronic generators and training-only adversaries for v2.3."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.application.research.shared.models import SpatialSeedDecoder


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, label_dim: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.modulation = nn.Linear(label_dim, channels * 2)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, value: torch.Tensor, label_embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(label_embedding).chunk(2, dim=1)
        value = self.norm(value) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        value = F.silu(self.conv1(value))
        return value + self.conv2(value)


class _ConditionalUpsampleGenerator(nn.Module):
    def __init__(self, *, code_dim: int, output_size: int, anchor: str, parameter_budget: int) -> None:
        super().__init__()
        if output_size not in {28, 200} or anchor not in {"none", "exact", "soft"}:
            raise ValueError("v2.3 generator output and anchor contract is invalid")
        self.code_dim, self.output_size, self.anchor = code_dim, output_size, anchor
        self.label_embedding = nn.Embedding(10, 32)
        if output_size == 28:
            self.projection = nn.Linear(code_dim + 32, 128 * 7 * 7)
            channels = (128, 64, 32)
        else:
            self.projection = nn.Linear(code_dim + 32, 128 * 25 * 25)
            channels = (128, 64, 32, 16)
        self.blocks = nn.ModuleList(FiLMResidualBlock(channel, 32) for channel in channels[:-1])
        self.transitions = nn.ModuleList(nn.Conv2d(channels[index], channels[index + 1], 3, padding=1) for index in range(len(channels) - 1))
        self.output = nn.Conv2d(channels[-1], 1, 3, padding=1)
        carrier_size = 7 if output_size == 28 else 25
        self.carrier_logits = nn.Parameter(torch.zeros(10, 1, carrier_size, carrier_size))
        if sum(parameter.numel() for parameter in self.parameters()) > parameter_budget:
            raise ValueError(f"generator exceeds declared {parameter_budget} parameter budget")

    def _residual(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embedding = self.label_embedding(labels)
        value = self.projection(torch.cat((code, embedding), dim=1))
        base = 7 if self.output_size == 28 else 25
        value = value.view(code.shape[0], 128, base, base)
        for index, block in enumerate(self.blocks):
            value = block(value, embedding)
            value = F.interpolate(value, scale_factor=2, mode="bilinear", align_corners=False)
            value = F.silu(self.transitions[index](value))
        value = F.interpolate(value, size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        return self.output(value)

    def forward(self, code: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if code.ndim != 2 or code.shape[1] != self.code_dim or labels.shape != (code.shape[0],):
            raise ValueError("v2.3 generator input shape is incompatible with its candidate")
        residual = self._residual(code, labels)
        if self.anchor == "exact":
            residual = residual - self._residual(torch.zeros_like(code), labels)
        carrier = F.interpolate(self.carrier_logits.index_select(0, labels), size=(self.output_size, self.output_size), mode="bilinear", align_corners=False)
        logits = carrier + residual
        return torch.sigmoid(logits)


class D0ImageGenerator(_ConditionalUpsampleGenerator):
    def __init__(self) -> None:
        super().__init__(code_dim=32, output_size=28, anchor="none", parameter_budget=5_000_000)


class ElectronicImageOracleGenerator(_ConditionalUpsampleGenerator):
    """High-capacity 28x28 electronic upper bound for the O0 contract.

    This model has no optical operator or command-space bottleneck.  It is
    deliberately separate from the historical D0 GAN candidates and uses the
    same label/code interface as the current conditional student.
    """

    architecture_kind = "electronic_objective_o0_image_v1"

    def __init__(self) -> None:
        super().__init__(code_dim=8, output_size=28, anchor="none", parameter_budget=6_000_000)


class HighCapacityCommandGenerator(_ConditionalUpsampleGenerator):
    def __init__(self, *, code_dim: int, anchor: str) -> None:
        super().__init__(code_dim=code_dim, output_size=200, anchor=anchor, parameter_budget=6_000_000)


def build_o0_candidate(candidate: str) -> nn.Module:
    if candidate == "O0-I":
        return ElectronicImageOracleGenerator()
    if candidate == "O0-G":
        from ldm_project.v2.student.identity_generator import IdentityAnchoredConditionalAmplitudeGenerator

        return IdentityAnchoredConditionalAmplitudeGenerator()
    if candidate == "D0":
        return D0ImageGenerator()
    if candidate == "C0":
        return SpatialSeedDecoder(8)
    if candidate == "C1":
        return HighCapacityCommandGenerator(code_dim=8, anchor="exact")
    if candidate == "C2":
        return HighCapacityCommandGenerator(code_dim=8, anchor="soft")
    if candidate == "C3":
        return HighCapacityCommandGenerator(code_dim=32, anchor="soft")
    raise ValueError(f"unknown v2.3 candidate {candidate}")


class ProjectionDiscriminator(nn.Module):
    """Training-only conditional discriminator; never part of inference."""

    def __init__(self, *, code_dim: int, image_size: int) -> None:
        super().__init__()
        if image_size not in {28, 200}:
            raise ValueError("discriminator image size must be 28 or 200")
        self.code_dim, self.image_size = code_dim, image_size
        self.convolutions = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1), nn.LeakyReLU(.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.LeakyReLU(.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.LeakyReLU(.2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.score = nn.Linear(256, 1)
        self.label_projection = nn.Embedding(10, 256)
        self.classifier = nn.Linear(256, 10)
        self.latent_head = nn.Linear(256, code_dim)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.pool(self.convolutions(images)).flatten(1)
        score = self.score(features).squeeze(1) + (features * self.label_projection(labels)).sum(1)
        return score, self.classifier(features), self.latent_head(features), features


def inference_state_dict(generator: nn.Module) -> dict[str, torch.Tensor]:
    """Return a generator-only state; training adversaries cannot leak to inference."""
    return {key: value.detach().cpu() for key, value in generator.state_dict().items()}
