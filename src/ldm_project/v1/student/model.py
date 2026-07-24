"""Public student model facade.

The formal physical model and compatibility/diagnostic model now have
separate implementations.  This module preserves historical import paths.
"""
from .formal_model import FormalOneShotOpticalGenerator
from .experimental.model import OneShotOpticalGenerator
from .checkpoints import (
    load_one_shot_checkpoint,
    load_one_shot_checkpoint_subset,
    validate_one_shot_checkpoint_payload,
)
from .contracts import ONE_SHOT_MODEL_VERSION

__all__ = [
    "ONE_SHOT_MODEL_VERSION",
    "FormalOneShotOpticalGenerator",
    "OneShotOpticalGenerator",
    "load_one_shot_checkpoint",
    "load_one_shot_checkpoint_subset",
    "validate_one_shot_checkpoint_payload",
]
