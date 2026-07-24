"""Shared tensor primitives for objective terms."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...distillation.targets import build_clean_readable_target
from ...training_loss_common import (
    _camera_artifact_metrics,
    _camera_high_saturation_ratio,
    _camera_low_saturation_ratio,
    _camera_saturation_loss,
    _camera_saturation_ratio,
    _camera_topology_loss,
    _foreground_dice_score,
    _foreground_shape_loss_for_samples,
    _masked_mean,
    _masked_mean_per_sample,
    _sobel_magnitude,
    _total_variation,
    _weighted_sample_mean,
    _zero_like_loss,
)

_build_clean_readable_target = build_clean_readable_target

__all__ = [
    "torch", "F", "build_clean_readable_target", "_build_clean_readable_target",
    "_camera_artifact_metrics", "_camera_high_saturation_ratio",
    "_camera_low_saturation_ratio", "_camera_saturation_loss",
    "_camera_saturation_ratio", "_camera_topology_loss", "_foreground_dice_score",
    "_foreground_shape_loss_for_samples", "_masked_mean", "_masked_mean_per_sample",
    "_sobel_magnitude", "_total_variation", "_weighted_sample_mean", "_zero_like_loss",
]
