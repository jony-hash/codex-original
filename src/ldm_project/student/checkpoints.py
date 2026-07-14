"""Checkpoint contract facade kept separate from the student forward model."""

from .model import (
    load_one_shot_checkpoint,
    load_one_shot_checkpoint_subset,
    validate_one_shot_checkpoint_payload,
)

__all__ = [
    "load_one_shot_checkpoint",
    "load_one_shot_checkpoint_subset",
    "validate_one_shot_checkpoint_payload",
]
