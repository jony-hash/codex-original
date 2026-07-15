"""Canonical YAML loading and normalization implementation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import (
    DataConfig, D2NNConfig, DiffusionConfig, DistillationConfig,
    ElectronicDomainConfig, FirstStageConfig, GeometryConfig, MacroOpticsConfig,
    OneShotConfig, OpticsConfig, PhysicsConfig, ReadoutConfig, TrainingConfig,
)

def _load_yaml_file(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load YAML configs.") from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in config file: {path}")
    return data


def load_physics_config(path: str | Path | None = None) -> PhysicsConfig:
    if path is None:
        return PhysicsConfig()

    data = _load_yaml_file(path)
    optics = OpticsConfig(**data.get("optics", {}))
    geometry = GeometryConfig(**data.get("geometry", {}))
    readout = ReadoutConfig(**data.get("readout", {}))
    macro_optics = MacroOpticsConfig(**data.get("macro_optics", {}))
    return PhysicsConfig(
        optics=optics,
        geometry=geometry,
        readout=readout,
        macro_optics=macro_optics,
    )


@dataclass(frozen=True)
class _TrainingConfigSections:
    training: dict[str, Any]
    data: dict[str, Any]
    physics: dict[str, Any] | None
    first_stage: dict[str, Any]
    diffusion: dict[str, Any]
    one_shot: dict[str, Any]
    d2nn: dict[str, Any] | None
    distillation: dict[str, Any]
    electronic_domain: dict[str, Any]


def _validate_legacy_training_fields(training_data: dict[str, Any]) -> None:
    if "smoke_steps" in training_data:
        raise ValueError(
            "training.smoke_steps is no longer supported; use training.first_stage_epochs "
            "and training.diffusion_epochs instead."
        )
    if "visual_save_interval" in training_data:
        raise ValueError(
            "training.visual_save_interval is no longer supported; "
            "use training.visual_save_epoch_interval instead."
        )


def _normalize_first_stage_section(first_stage_data: dict[str, Any]) -> dict[str, Any]:
    normalized = first_stage_data.copy()
    ols_data = normalized.pop("ols", {})
    if "kind" in ols_data:
        normalized["ols_kind"] = ols_data["kind"]
    if "encoder_channels" in normalized:
        normalized["encoder_channels"] = tuple(normalized["encoder_channels"])
    if "hard_class_weight_labels" in normalized:
        normalized["hard_class_weight_labels"] = tuple(normalized["hard_class_weight_labels"])
    if "camera_label_margin_labels" in normalized:
        normalized["camera_label_margin_labels"] = tuple(normalized["camera_label_margin_labels"])
    return normalized


def _normalize_diffusion_section(diffusion_data: dict[str, Any]) -> dict[str, Any]:
    if "hidden_channels" in diffusion_data and "base_channels" not in diffusion_data:
        diffusion_data["base_channels"] = diffusion_data.pop("hidden_channels")
    for name in ("channel_multipliers", "attention_levels", "sample_class_labels", "class_weight_labels"):
        if name in diffusion_data:
            diffusion_data[name] = tuple(diffusion_data[name])
    return diffusion_data


def _normalize_one_shot_section(one_shot_data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    normalized = one_shot_data.copy()
    d2nn_data = normalized.pop("d2nn", None)
    normalized.pop("diffractive_modulation", None)
    for name in (
        "sample_class_labels",
        "class_recon_weight_labels",
        "student_label_margin_labels",
        "student_prototype_ce_labels",
        "student_roi_prototype_ce_labels",
    ):
        if name in normalized:
            normalized[name] = tuple(normalized[name])
    if "decoder_diagnostic_phase_layer_mask" in normalized:
        normalized["decoder_diagnostic_phase_layer_mask"] = tuple(
            normalized["decoder_diagnostic_phase_layer_mask"]
        )
    for name in (
        "student_negative_label_margin_config",
        "student_roi_negative_label_margin_config",
        "student_pair_contrast_config",
    ):
        if name in normalized:
            normalized[name] = tuple(tuple(item) for item in normalized[name])
    return normalized, d2nn_data


def _normalize_training_config_sections(data: dict[str, Any]) -> _TrainingConfigSections:
    training_data = data.get("training", {})
    _validate_legacy_training_fields(training_data)
    data_cfg = data.get("data", {})
    physics_data = data.get("physics")
    first_stage_data = _normalize_first_stage_section(data.get("first_stage", {}))
    diffusion_data = _normalize_diffusion_section(data.get("diffusion", {}))
    one_shot_data, d2nn_data = _normalize_one_shot_section(data.get("one_shot", {}))
    distillation_data = data.get("distillation", {}).copy()
    electronic_domain_data = first_stage_data.pop("electronic_domain", {})
    if "class_labels" in distillation_data:
        distillation_data["class_labels"] = tuple(distillation_data["class_labels"])
    return _TrainingConfigSections(
        training=training_data,
        data=data_cfg,
        physics=physics_data,
        first_stage=first_stage_data,
        diffusion=diffusion_data,
        one_shot=one_shot_data,
        d2nn=d2nn_data,
        distillation=distillation_data,
        electronic_domain=electronic_domain_data,
    )


def _build_training_config(sections: _TrainingConfigSections) -> TrainingConfig:
    training_seed = sections.training.get("seed", TrainingConfig.seed)
    if training_seed is not None:
        training_seed = int(training_seed)
        if training_seed < 0:
            raise ValueError(f"training.seed must be non-negative, got {training_seed}")
    data_config = DataConfig(**sections.data)
    if data_config.validation_fraction < 0.0 or data_config.validation_fraction >= 1.0:
        raise ValueError(
            "data.validation_fraction must be in [0.0, 1.0), "
            f"got {data_config.validation_fraction}"
        )

    d2nn_config = D2NNConfig(**sections.d2nn) if sections.d2nn else None
    physics_config = None
    if sections.physics is not None:
        physics_config = PhysicsConfig(
            optics=OpticsConfig(**sections.physics.get("optics", {})),
            geometry=GeometryConfig(**sections.physics.get("geometry", {})),
            readout=ReadoutConfig(**sections.physics.get("readout", {})),
            macro_optics=MacroOpticsConfig(**sections.physics.get("macro_optics", {})),
            front_path_model=sections.physics.get("front_path_model", "free_space"),
            back_path_model=sections.physics.get("back_path_model", "free_space"),
            front_relay_psf_sigma_px=float(sections.physics.get("front_relay_psf_sigma_px", 0.0)),
            physical_forward_contract=bool(sections.physics.get("physical_forward_contract", False)),
        )

    return TrainingConfig(
        mode=sections.training.get("mode", TrainingConfig.mode),
        batch_size=sections.training.get("batch_size", TrainingConfig.batch_size),
        learning_rate=sections.training.get("learning_rate", TrainingConfig.learning_rate),
        seed=training_seed,
        step_log_interval=sections.training.get("step_log_interval", TrainingConfig.step_log_interval),
        visual_save_epoch_interval=sections.training.get(
            "visual_save_epoch_interval",
            TrainingConfig.visual_save_epoch_interval,
        ),
        validation_epoch_interval=sections.training.get(
            "validation_epoch_interval",
            TrainingConfig.validation_epoch_interval,
        ),
        one_shot_distill_epoch_checkpoint_interval=sections.training.get(
            "one_shot_distill_epoch_checkpoint_interval",
            TrainingConfig.one_shot_distill_epoch_checkpoint_interval,
        ),
        first_stage_epochs=sections.training.get("first_stage_epochs", TrainingConfig.first_stage_epochs),
        diffusion_epochs=sections.training.get("diffusion_epochs", TrainingConfig.diffusion_epochs),
        one_shot_epochs=sections.training.get("one_shot_epochs", TrainingConfig.one_shot_epochs),
        data=data_config,
        first_stage=FirstStageConfig(
            electronic_domain=ElectronicDomainConfig(**sections.electronic_domain),
            **sections.first_stage,
        ),
        diffusion=DiffusionConfig(**sections.diffusion),
        one_shot=OneShotConfig(d2nn=d2nn_config, **sections.one_shot),
        distillation=DistillationConfig(**sections.distillation),
        physics=physics_config,
    )


def load_training_config(path: str | Path | None = None) -> TrainingConfig:
    """Load and validate a training YAML using the canonical scoped modules."""
    if path is None:
        return TrainingConfig()

    data = _load_yaml_file(path)
    sections = _normalize_training_config_sections(data)
    config = _build_training_config(sections)
    from .validation import (
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

    _validate_d2nn_training_config(config)
    _validate_first_stage_training_config(config)
    _validate_one_shot_selection_and_weights(config)
    _validate_one_shot_generator_config(config)
    _validate_one_shot_objective_config(config)
    _validate_one_shot_diagnostic_config(config)
    _validate_one_shot_target_config(config)
    _validate_one_shot_label_config(config)
    if config.one_shot.physical_forward_contract:
        if config.physics is None:
            raise ValueError("physical forward contract requires a physics section")
        validate_physical_forward_contract(config, config.physics)
    return config
