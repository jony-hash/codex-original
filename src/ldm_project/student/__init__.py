"""Formal single-exposure optical student domain.

Teacher modules may provide targets, but the formal student forward, objective,
checkpoint, training, evaluation and artifact contracts live here.
"""

from .generator import ElectronicGeneratorE
from .layers import (
    IdealAmplitudeSLMModel,
    PhaseAmplitudeDiffractiveStack,
    PhaseOnlyDiffractiveStack,
)
from .model import (
    FormalOneShotOpticalGenerator,
    ONE_SHOT_MODEL_VERSION,
    OneShotOpticalGenerator,
    load_one_shot_checkpoint,
    load_one_shot_checkpoint_subset,
    validate_one_shot_checkpoint_payload,
)
from .formal_contract import validate_formal_student_config

__all__ = [
    "ElectronicGeneratorE",
    "FormalOneShotOpticalGenerator",
    "IdealAmplitudeSLMModel",
    "ONE_SHOT_MODEL_VERSION",
    "OneShotOpticalGenerator",
    "PhaseAmplitudeDiffractiveStack",
    "PhaseOnlyDiffractiveStack",
    "load_one_shot_checkpoint",
    "load_one_shot_checkpoint_subset",
    "validate_one_shot_checkpoint_payload",
    "validate_formal_student_config",
]
