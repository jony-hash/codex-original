"""Formal camera-reconstruction objective.

This module intentionally contains no student model or diagnostic imports.  The
resolver callback keeps legacy class-weight semantics in the orchestration facade
while making the formal term independently testable.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from ...training_loss_common import (
    _weighted_reconstruction_metrics,
    _weighted_reconstruction_metrics_for_samples,
    _zero_like_loss,
)


def formal_reconstruction_terms(
    outputs: dict[str, torch.Tensor],
    target_batch: torch.Tensor,
    class_labels: torch.Tensor | None,
    *,
    foreground_threshold: float,
    foreground_weight: float,
    background_weight: float,
    class_weight_labels: tuple[int, ...],
    class_weight: float,
    sample_weight_resolver: Callable[[torch.Tensor, torch.Tensor | None, tuple[int, ...], float], torch.Tensor],
) -> dict[str, Any]:
    """Return the unchanged formal reconstruction intermediates."""
    foreground_mask = (target_batch > foreground_threshold).float()
    background_mask = 1.0 - foreground_mask
    camera_prediction = outputs["camera_plane"]
    camera_loss_prediction = outputs.get("camera_plane_loss", camera_prediction)
    sample_weights = sample_weight_resolver(target_batch, class_labels, class_weight_labels, class_weight)
    has_class_weights = class_labels is not None and bool(class_weight_labels) and class_weight != 1.0
    if has_class_weights:
        metrics = _weighted_reconstruction_metrics_for_samples(
            camera_prediction, target_batch, foreground_mask, background_mask,
            foreground_weight, background_weight, sample_weights,
        )
    else:
        metrics = _weighted_reconstruction_metrics(
            camera_prediction, target_batch, foreground_mask, background_mask,
            foreground_weight, background_weight,
        )
    return {
        "foreground_mask": foreground_mask,
        "background_mask": background_mask,
        "camera_prediction": camera_prediction,
        "camera_loss_prediction": camera_loss_prediction,
        "sample_weights": sample_weights,
        "metrics": metrics,
        "zero": _zero_like_loss(camera_prediction),
    }


__all__ = ["formal_reconstruction_terms"]
