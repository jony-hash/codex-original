"""YAML loading and normalization facade for scoped callers."""
from .legacy import (
    _build_training_config,
    _load_yaml_file,
    _normalize_diffusion_section,
    _normalize_first_stage_section,
    _normalize_one_shot_section,
    _normalize_training_config_sections,
    load_physics_config,
    load_training_config,
)

__all__ = [
    "load_physics_config",
    "load_training_config",
    "_build_training_config",
    "_load_yaml_file",
    "_normalize_diffusion_section",
    "_normalize_first_stage_section",
    "_normalize_one_shot_section",
    "_normalize_training_config_sections",
]
