"""Scoped validation facade preserving the legacy error contract."""
from .legacy import (
    _validate_d2nn_training_config,
    _validate_first_stage_training_config,
    _validate_one_shot_diagnostic_config,
    _validate_one_shot_generator_config,
    _validate_one_shot_label_config,
    _validate_one_shot_objective_config,
    _validate_one_shot_selection_and_weights,
    _validate_one_shot_target_config,
    validate_physical_forward_contract,
)

__all__ = [
    "validate_physical_forward_contract",
    "_validate_d2nn_training_config",
    "_validate_first_stage_training_config",
    "_validate_one_shot_diagnostic_config",
    "_validate_one_shot_generator_config",
    "_validate_one_shot_label_config",
    "_validate_one_shot_objective_config",
    "_validate_one_shot_selection_and_weights",
    "_validate_one_shot_target_config",
]
