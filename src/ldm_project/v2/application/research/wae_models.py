"""Small, teacher-free WAE models used by the electronic O0 research gate.

The encoder is deliberately a training-only object.  The decoder is the only
state written to the inference artifact; this makes the electronic hypothesis
portable without quietly making the encoder an inference-time component.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


Candidate = Literal["O0-I", "O0-G"]
FIXED_EXPOSURE_SCALE = 0.4913053810596466


class ConditionalWAEEncoder(nn.Module):
    def __init__(self, code_dim: int = 8, label_dim: int = 32) -> None:
        super().__init__()
        self.code_dim = code_dim
        self.label = nn.Embedding(10, label_dim)
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 + label_dim, 64),
            nn.SiLU(),
            nn.Linear(64, code_dim),
        )

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1:] != (1, 28, 28):
            raise ValueError("WAE encoder expects [batch,1,28,28]")
        features = self.features(images).flatten(1)
        return self.projection(torch.cat((features, self.label(labels)), dim=1))


class FiLMResidualBlock(nn.Module):
    def __init__(self, channels: int, label_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.film = nn.Linear(label_dim, channels * 4)

    def forward(self, values: torch.Tensor, label_code: torch.Tensor) -> torch.Tensor:
        scale1, shift1, scale2, shift2 = self.film(label_code).chunk(4, dim=1)
        scale1, shift1 = scale1[:, :, None, None], shift1[:, :, None, None]
        scale2, shift2 = scale2[:, :, None, None], shift2[:, :, None, None]
        hidden = self.norm1(self.conv1(values)) * (1.0 + 0.1 * scale1) + 0.1 * shift1
        hidden = F.silu(hidden)
        hidden = self.norm2(self.conv2(hidden)) * (1.0 + 0.1 * scale2) + 0.1 * shift2
        return F.silu(values + hidden)


class ConditionalWAEDecoder(nn.Module):
    def __init__(
        self, candidate: Candidate, code_dim: int = 8, label_dim: int = 32
    ) -> None:
        super().__init__()
        if candidate not in {"O0-I", "O0-G"}:
            raise ValueError("candidate must be O0-I or O0-G")
        self.candidate = candidate
        self.code_dim = code_dim
        self.label = nn.Embedding(10, label_dim)
        self.start = 7 if candidate == "O0-I" else 25
        self.channels = 128
        self.input = nn.Linear(
            code_dim + label_dim, self.channels * self.start * self.start
        )
        self.block = FiLMResidualBlock(self.channels, label_dim)
        self.up1 = nn.Conv2d(self.channels, 64, 3, padding=1)
        self.up2 = nn.Conv2d(64, 32, 3, padding=1)
        self.up3 = nn.Conv2d(32, 16, 3, padding=1)
        self.output = nn.Conv2d(16, 1, 3, padding=1)

    @property
    def output_size(self) -> int:
        return 28 if self.candidate == "O0-I" else 200

    def forward(self, codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if (
            codes.ndim != 2
            or codes.shape[1] != self.code_dim
            or labels.ndim != 1
            or labels.shape[0] != codes.shape[0]
        ):
            raise ValueError("WAE decoder code/label shapes are invalid")
        label_code = self.label(labels)
        values = self.input(torch.cat((codes, label_code), dim=1)).view(
            -1, self.channels, self.start, self.start
        )
        values = self.block(values, label_code)
        values = F.silu(self.up1(F.interpolate(values, scale_factor=2, mode="nearest")))
        values = F.silu(self.up2(F.interpolate(values, scale_factor=2, mode="nearest")))
        values = F.silu(self.up3(F.interpolate(values, scale_factor=2, mode="nearest")))
        values = self.output(values)
        if values.shape[-1] != self.output_size:
            values = F.interpolate(
                values,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return torch.sigmoid(values)


def render_view(commands: torch.Tensor, candidate: Candidate) -> torch.Tensor:
    if candidate == "O0-I":
        if commands.shape[-2:] != (28, 28):
            raise ValueError("O0-I command must be 28x28")
        return commands
    if commands.shape[-2:] != (200, 200):
        raise ValueError("O0-G command must be 200x200")
    return F.adaptive_avg_pool2d(
        commands.square() / FIXED_EXPOSURE_SCALE, (28, 28)
    ).clamp(0.0, 1.0)


def sobel_l1(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    horizontal = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=prediction.dtype,
        device=prediction.device,
    ).view(1, 1, 3, 3)
    vertical = horizontal.transpose(-1, -2)
    return (
        F.conv2d(prediction, horizontal, padding=1).abs()
        - F.conv2d(target, horizontal, padding=1).abs()
    ).abs().mean() + (
        F.conv2d(prediction, vertical, padding=1).abs()
        - F.conv2d(target, vertical, padding=1).abs()
    ).abs().mean()


def imq_mmd(
    values: torch.Tensor, *, scales: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
) -> torch.Tensor:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("MMD requires a two-dimensional batch")
    reference = torch.randn_like(values)
    xx = torch.cdist(values, values).square()
    yy = torch.cdist(reference, reference).square()
    xy = torch.cdist(values, reference).square()
    result = values.new_zeros(())
    for scale in scales:
        bandwidth = float(scale) ** 2
        result = (
            result
            + (1.0 / (bandwidth + xx)).mean()
            + (1.0 / (bandwidth + yy)).mean()
            - 2.0 * (1.0 / (bandwidth + xy)).mean()
        )
    return result / len(scales)


def classwise_mmd(codes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return torch.stack([imq_mmd(codes[labels == label]) for label in range(10)]).mean()


def wae_loss(
    decoder: ConditionalWAEDecoder,
    encoder: ConditionalWAEEncoder,
    images: torch.Tensor,
    labels: torch.Tensor,
    codes: torch.Tensor,
    *,
    w_mmd: float,
    w_cycle: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    commands = decoder(codes, labels)
    prediction = render_view(commands, decoder.candidate)
    reconstruction = F.binary_cross_entropy(prediction, images) + 0.25 * sobel_l1(
        prediction, images
    )
    mmd = classwise_mmd(encoder(images, labels), labels)
    prior = torch.randn_like(codes)
    cycle_images = render_view(decoder(prior, labels), decoder.candidate)
    cycle = F.smooth_l1_loss(encoder(cycle_images, labels), prior)
    total = reconstruction + float(w_mmd) * mmd + float(w_cycle) * cycle
    return total, {
        "reconstruction": reconstruction.detach(),
        "mmd": mmd.detach(),
        "cycle": cycle.detach(),
    }


def parameter_counts(encoder: nn.Module, decoder: nn.Module) -> dict[str, int]:
    return {
        "encoder": sum(p.numel() for p in encoder.parameters()),
        "decoder": sum(p.numel() for p in decoder.parameters()),
        "total": sum(p.numel() for p in encoder.parameters())
        + sum(p.numel() for p in decoder.parameters()),
    }
