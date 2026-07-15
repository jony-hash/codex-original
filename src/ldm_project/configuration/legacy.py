"""Compatibility facade for the canonical configuration modules.

The implementation is now split into schema, loading and validation.  Historical
private names remain exported here so existing tools and checkpoint loaders retain
exact behavior and error messages.
"""
from __future__ import annotations

from .schema import *  # noqa: F401,F403
from .loading import (
    _TrainingConfigSections,
    _build_training_config,
    _load_yaml_file,
    _normalize_diffusion_section,
    _normalize_first_stage_section,
    _normalize_one_shot_section,
    _normalize_training_config_sections,
    _validate_legacy_training_fields,
    load_physics_config,
    load_training_config,
)
from .validation import (
    _ONE_SHOT_COMMAND_TRANSFORMS,
    _ONE_SHOT_DECODER_DIAGNOSTIC_MODES,
    _ONE_SHOT_DECODER_MASK_COMPONENTS,
    _ONE_SHOT_DECODER_MODULATION_OVERRIDES,
    _ONE_SHOT_GENERATOR_ARCHES,
    _ONE_SHOT_OLS_CALIBRATIONS,
    _ONE_SHOT_SLM_OVERRIDE_MODES,
    _ONE_SHOT_TARGET_CODE_BRIDGE_ARCHES,
    _ONE_SHOT_TARGET_CODE_BRIDGE_SOURCES,
    _ONE_SHOT_TARGET_CODE_MODES,
    _ONE_SHOT_TEMPLATE_INITIALIZERS,
    _expected_d2nn_modulation,
    _target_code_bridge_mode_is_compatible,
    _validate_d2nn_training_config,
    _validate_first_stage_training_config,
    _validate_one_shot_aux_objective_config,
    _validate_one_shot_decoder_diagnostic_config,
    _validate_one_shot_diagnostic_config,
    _validate_one_shot_generator_arch_requirements,
    _validate_one_shot_generator_config,
    _validate_one_shot_generator_requirements,
    _validate_one_shot_generator_values,
    _validate_one_shot_label_config,
    _validate_one_shot_margin_pairs,
    _validate_one_shot_objective_config,
    _validate_one_shot_selection_and_weights,
    _validate_one_shot_student_objective_values,
    _validate_one_shot_target_code_diagnostic_config,
    _validate_one_shot_target_config,
    _validate_one_shot_class_labels,
    validate_physical_forward_contract,
)
