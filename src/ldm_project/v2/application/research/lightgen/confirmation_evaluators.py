"""Frozen confirmation-evaluator architectures shared by LightGen research."""

from __future__ import annotations

import torch
from torch import nn


ED_CHECKPOINT_SCHEMA = "sic_confirmation_evaluator_d_checkpoint.v1"
EC_CHECKPOINT_SCHEMA = "third_confirmation_evaluator_checkpoint.v1"
CHECKPOINT_SCHEMA = EC_CHECKPOINT_SCHEMA
OUTER_GATE = "THIRD_CONFIRMATION_EVALUATOR_AUTHORITY_ESTABLISHED"
CLASSIFICATION_SUPPORTED = OUTER_GATE
SEMANTIC_AUTHORITY_ID = "THIRD_CONFIRMATION_EVALUATOR_V1"


class PixelMLPConfirmationEvaluatorV1(nn.Module):
    """The fixed non-convolutional E_D confirmation evaluator."""

    architecture_kind = "pixel_mlp_confirmation_evaluator_v1"

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("E_D input must have shape [B,1,28,28]")
        return self.network(images)

    def freeze(self) -> "PixelMLPConfirmationEvaluatorV1":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


class ThirdConfirmationEvaluator(nn.Module):
    """The fixed canonical convolutional E_C evaluator."""

    architecture_kind = "third_confirmation_canonical_cnn_v1"

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
            raise ValueError("E_C input must have shape [B,1,28,28]")
        features = self.embedding(self.features(images))
        return features, self.classifier(features)

    def freeze(self) -> "ThirdConfirmationEvaluator":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self
