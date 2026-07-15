from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


@dataclass
class OpticsConfig:
    wavelength: float = 532.0e-9
    pixel_size: float = 2.5e-6


@dataclass
class GeometryConfig:
    grid_size: int = 200
    prop_dist: float = 1.0e-3


@dataclass
class ReadoutConfig:
    primary_plane: str = "camera"
    save_camera_plane: bool = True
    normalize_percentile: float = 99.5
    camera_magnification_blend: float = 0.35
    camera_blur_gain: float = 1.0
    relay_psf_sigma_px: float = 0.0

    def __post_init__(self) -> None:
        allowed = {"camera", "probe", "chip_amplitude", "chip_intensity"}
        if self.primary_plane not in allowed:
            raise ValueError(
                "readout.primary_plane must be one of "
                f"{sorted(allowed)}, got {self.primary_plane!r}."
            )
        if self.relay_psf_sigma_px < 0.0:
            raise ValueError("readout.relay_psf_sigma_px must be non-negative.")


@dataclass
class MacroOpticsConfig:
    dist_laser_to_m1: float = 0.350
    dist_m1_to_bs: float = 0.370
    dist_bs_to_slm: float = 0.025
    dist_slm_to_bs: float = 0.025
    dist_bs_to_m3: float = 0.125
    # Formal v55 relay segments.  dist_m3_to_lg is retained only for legacy
    # free-space diagnostics and must not be used by the formal contract.
    dist_m3_to_front_lens1: float = 0.07321428571428572
    dist_front_lens1_to_lens2: float = 0.3125
    dist_front_lens2_to_encoder: float = 0.0892857142857143
    dist_m3_to_lg: float = 0.475
    dist_lg_to_relay_lens: float = 0.100
    dist_relay_lens_to_cmos: float = 0.250
    dist_lg_to_cmos: float = 0.350
    f_relay: float = 0.0714285714
    front_lens1_focal_length: float = 0.22321428571428573
    front_lens2_focal_length: float = 0.0892857142857143
    front_lens1_clear_aperture: float = 0.02375
    front_lens2_clear_aperture: float = 0.02375
    back_relay_clear_aperture: float = 0.0266
    propagation_grid_size: int = 2048

    def __post_init__(self) -> None:
        if self.propagation_grid_size <= 0:
            raise ValueError("macro_optics.propagation_grid_size must be positive")


@dataclass
class PhysicsConfig:
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    macro_optics: MacroOpticsConfig = field(default_factory=MacroOpticsConfig)
    front_path_model: str = "free_space"
    back_path_model: str = "free_space"
    front_relay_psf_sigma_px: float = 0.0
    physical_forward_contract: bool = False

    def __post_init__(self) -> None:
        front_allowed = {"free_space", "coherent_imaging_4f", "relay_identity", "relay_coherent_psf", "relay_intensity_psf"}
        back_allowed = {"free_space", "relay_identity", "relay_psf"}
        if self.front_path_model not in front_allowed:
            raise ValueError(
                "physics.front_path_model must be one of "
                f"{sorted(front_allowed)}, got {self.front_path_model!r}."
            )
        if self.back_path_model not in back_allowed:
            raise ValueError(
                "physics.back_path_model must be one of "
                f"{sorted(back_allowed)}, got {self.back_path_model!r}."
            )
        if self.front_relay_psf_sigma_px < 0.0:
            raise ValueError(
                "physics.front_relay_psf_sigma_px must be non-negative, "
                f"got {self.front_relay_psf_sigma_px}."
            )


@dataclass
class FirstStageConfig:
    type: str = "optical_path_vae"
    image_size: int = 200
    latent_channels: int = 4
    encoder_channels: tuple[int, int, int] = (64, 128, 256)
    readout_plane: str = "camera"
    latent_position: str = "post_encoder_pre_ols"
    ols_kind: str = "pointwise_tanh"
    encoder_fusion_mode: str = "optical_only"
    image_fusion_start_weight: float = 1.0
    image_fusion_end_weight: float = 0.0
    image_fusion_decay_epochs: int = 20
    image_fusion_schedule: str = "linear"
    optical_encoder_input_mode: str = "pf2_intensity"
    optical_encoder_arch: str = "conv"
    metasurface_decoder_arch: str = "conv"
    linear_metasurface_grid_size: int = 50
    linear_metasurface_layer_spacing: float = 0.003
    optical_field_mode: str = "complex_cartesian"
    primary_output: str = "camera"
    freeze_after_pretrain: bool = True
    kl_beta: float = 1.0e-4
    kl_warmup_ratio: float = 0.1
    recon_foreground_threshold: float = 0.05
    recon_foreground_weight: float = 4.0
    recon_background_weight: float = 1.0
    image_recon_weight: float = 0.25
    camera_recon_weight: float = 1.0
    decoder_tail_channels: int = 32
    structure_recon_weight: float = 0.0
    shape_recon_weight: float = 0.5
    field_smoothness_weight: float = 0.02
    phase_smoothness_weight: float = 0.01
    background_leakage_weight: float = 0.02
    field_energy_weight: float = 0.0
    camera_hotspot_weight: float = 0.0
    camera_topology_weight: float = 0.0
    camera_saturation_weight: float = 0.0
    camera_artifact_weight: float = 0.0
    hard_class_weight_labels: tuple[int, ...] = ()
    hard_class_weight: float = 1.0
    camera_label_margin_weight: float = 0.0
    camera_label_margin_target: float = 0.08
    camera_label_margin_labels: tuple[int, ...] = ()
    camera_label_margin_prototypes_per_class: int = 64
    checkpoint_selection_metric: str = "camera_complex_balanced_mnist_score"
    electronic_domain: "ElectronicDomainConfig" = field(default_factory=lambda: ElectronicDomainConfig())


@dataclass
class ElectronicDomainConfig:
    mode: str = "mnist_complex_slm"
    background_floor: float = 0.05
    gamma: float = 1.0
    polarity: str = "non_inverted"
    phase_scale_rad: float = math.pi
    phase_bias_rad: float = 0.0


@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1.0e-4
    beta_end: float = 2.0e-2
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    attention_levels: tuple[int, ...] = (1, 2)
    num_heads: int = 4
    context_dim: int = 128
    num_classes: int = 10
    cond_dropout_prob: float = 0.1
    latent_source: str = "mu"
    guidance_scale: float = 3.0
    sample_class_labels: tuple[int, ...] = ()
    sample_repeats_per_class: int = 2
    class_weight_labels: tuple[int, ...] = ()
    class_weight: float = 1.0


@dataclass
class D2NNConfig:
    """衍射神经网络全局参数 — 单点配置，覆盖散落的 diffractive_* 字段.

    所有距离/长度单位均为 **米 (m)**，与 physics_cfg.yaml 一致.
    """

    wavelength: float = 532.0e-9
    pixel_size: float = 2.5e-6
    grid_size: int = 200
    propagation_grid_size: int = 1024
    encoder_layers: int = 2
    decoder_layers: int = 2
    layer_spacing: float = 0.005
    modulation: str = "phase_only"
    phase_init: str = "uniform_pi"
    amplitude_logit_init: float = 4.0


@dataclass
class OneShotConfig:
    code_dim: int = 128
    label_embed_dim: int = 32
    hidden_dim: int = 256
    num_classes: int = 10
    generator_e_arch: str = "mlp"
    generator_command_transform: str = "direct_amplitude"
    generator_template_init: str = "none"
    code_residual_scale: float = 0.0
    generator_base_lr_scale: float = 1.0
    generator_residual_lr_scale: float = 1.0
    optical_core_arch: str = "dense_linear"
    slm_background_floor: float = 0.05
    slm_residual_phase_scale_rad: float = 0.0
    diagnostic_slm_override_mode: str = "none"
    diagnostic_slm_override_amplitude: float = 0.0
    ols_kind: str = "identity"
    physical_forward_contract: bool = False
    metasurface_grid_size: int = 40
    layer_spacing: float = 0.003
    d2nn: D2NNConfig | None = None
    diffractive_grid_size: int = 200
    diffractive_propagation_grid_size: int = 1024
    encoder_diffractive_layers: int = 2
    decoder_diffractive_layers: int = 2
    diffractive_layer_spacing: float = 0.005
    diffractive_phase_init: str = "uniform_pi"
    phase_learning_rate: float = 1.0e-2
    amplitude_learning_rate: float | None = None
    recon_foreground_threshold: float = 0.05
    recon_foreground_weight: float = 3.0
    recon_background_weight: float = 1.0
    class_recon_weight_labels: tuple[int, ...] = ()
    class_recon_weight: float = 1.0
    shape_loss_weight: float = 0.0
    edge_loss_weight: float = 0.0
    slm_smoothness_weight: float = 0.0
    slm_contrast_weight: float = 0.0
    camera_saturation_weight: float = 0.0
    camera_topology_weight: float = 0.0
    camera_artifact_weight: float = 0.0
    camera_readability_weight: float = 0.0
    camera_background_speckle_weight: float = 0.0
    camera_background_hotspot_weight: float = 0.0
    camera_background_hotspot_threshold: float = 0.08
    camera_background_highfreq_weight: float = 0.0
    camera_readable_contrast_weight: float = 0.0
    camera_readable_contrast_target: float = 4.5
    camera_recon_weight: float = 1.0
    distill_target_source: str = "teacher_camera"
    aux_detail_target_source: str = "none"
    aux_detail_edge_weight: float = 0.0
    aux_detail_shape_weight: float = 0.0
    aux_detail_foreground_gate_dilation: int = 0
    residual_l1_weight: float = 0.0
    residual_background_weight: float = 0.0
    clean_target_enabled: bool = False
    clean_foreground_dice_weight: float = 0.0
    clean_foreground_hole_weight: float = 0.0
    clean_background_speckle_weight: float = 0.0
    clean_background_hotspot_weight: float = 0.0
    clean_background_highfreq_weight: float = 0.0
    encoder_input_topology_weight: float = 0.0
    encoder_input_topology_dice_weight: float = 1.0
    encoder_input_topology_mse_weight: float = 1.0
    student_label_margin_weight: float = 0.0
    student_label_margin_target: float = 0.03
    student_label_margin_labels: tuple[int, ...] = ()
    student_label_margin_prototype_source: str = "mnist"
    student_label_margin_focus_label: int | None = None
    student_label_margin_focus_weight: float = 1.0
    student_prototype_ce_weight: float = 0.0
    student_prototype_ce_labels: tuple[int, ...] = ()
    student_prototype_ce_temperature: float = 0.07
    student_negative_label_margin_weight: float = 0.0
    student_negative_label_margin_config: tuple[tuple[int, int, float], ...] = ()
    student_roi_prototype_ce_weight: float = 0.0
    student_roi_prototype_ce_labels: tuple[int, ...] = ()
    student_roi_prototype_ce_temperature: float = 0.07
    student_roi_negative_label_margin_weight: float = 0.0
    student_roi_negative_label_margin_config: tuple[tuple[int, int, float], ...] = ()
    student_roi_foreground_threshold: float = 0.20
    aux_semantic_plane_key: str = ""
    aux_semantic_weight: float = 0.0
    aux_semantic_prototype_ce_weight: float = 0.0
    aux_semantic_negative_label_margin_weight: float = 0.0
    aux_topology_plane_key: str = ""
    aux_topology_weight: float = 0.0
    aux_topology_dice_weight: float = 1.0
    aux_topology_mse_weight: float = 1.0
    aux_topology_target_transform: str = "none"
    decoder_only_training: bool = False
    decoder_trainability_audit: bool = False
    decoder_diagnostic_mode: str = "normal"
    decoder_diagnostic_modulation_override: str = "none"
    decoder_diagnostic_mask_component: str = "trained"
    decoder_diagnostic_phase_layer_mask: tuple[bool, ...] = ()
    target_code_diagnostic_mode: str = "none"
    target_code_spatial_scale: float | None = None
    target_code_bridge_source: str = "none"
    target_code_bridge_arch: str = "shallow"
    target_code_bridge_hidden_channels: int = 8
    target_code_bridge_trainability_audit: bool = False
    ols_transfer_calibration: str = "none"
    code_diversity_weight: float = 0.0
    sample_code_diversity_weight: float = 0.0
    teacher_pairwise_diversity_weight: float = 0.0
    cross_class_pairwise_weight: float = 0.0
    diversity_min_teacher_mse: float = 1.0e-4
    checkpoint_selection_metric: str = "loss"
    sample_class_labels: tuple[int, ...] = ()
    sample_repeats_per_class: int = 2
    eval_samples_per_class: int | None = None
    eval_preview_per_class: int = 2
    eval_allow_train_fallback: bool = False
    student_label_audit_enabled: bool = False
    # v7.2: class-pair contrast to push apart confused classes.
    student_pair_contrast_config: tuple[tuple[int, int, float], ...] = ()
    # v7.2: display-domain background speckle penalty.
    student_display_speckle_weight: float = 0.0


@dataclass
class DistillationConfig:
    samples_per_class: int = 200
    total_samples: int | None = None
    train_subset_size: int | None = None
    paired_class_batches: bool = False
    multiclass_batches: bool = False
    class_labels: tuple[int, ...] = ()
    output_path: str | None = None
    code_seed: int = 1234
    code_projection_seed: int = 2026
    sampling_steps: int | None = None
    filter_keep_per_class: int = 200
    filter_foreground_trim_fraction: float = 0.05
    filter_contrast_drop_fraction: float = 0.10
    filter_foreground_threshold: float = 0.2
    label_strict_keep_per_class: int = 100
    label_audit_prototypes_per_class: int = 64
    label_audit_batch_size: int | None = None
    readable_min_label_margin: float = 0.03
    readable_min_contrast: float = 0.70
    readable_max_background_speckle: float = 0.20
    readable_max_foreground_hole: float = 0.08
    readable_train_per_class: int = 80
    readable_val_per_class: int = 10
    readable_test_per_class: int = 10
    readable_split_seed: int = 2027


@dataclass
class DataConfig:
    dataset: str = "mnist"
    root: str = "data"
    download: bool = True
    subset_fraction: float = 1.0
    subset_size: int | None = None
    subset_seed: int = 42
    validation_fraction: float = 0.1
    validation_split_seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class TrainingConfig:
    mode: str = "first_stage"
    batch_size: int = 4
    learning_rate: float = 1.0e-4
    seed: int | None = None
    step_log_interval: int = 100
    visual_save_epoch_interval: int = 1
    validation_epoch_interval: int = 1
    one_shot_distill_epoch_checkpoint_interval: int = 0
    first_stage_epochs: int = 10
    diffusion_epochs: int = 20
    one_shot_epochs: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    first_stage: FirstStageConfig = field(default_factory=FirstStageConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    one_shot: OneShotConfig = field(default_factory=OneShotConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    physics: PhysicsConfig | None = None


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


def _validate_d2nn_training_config(config: TrainingConfig) -> None:
    if config.one_shot.d2nn is None:
        return
    if not math.isfinite(float(config.one_shot.d2nn.amplitude_logit_init)):
        raise ValueError(
            "one_shot.d2nn.amplitude_logit_init must be finite, "
            f"got {config.one_shot.d2nn.amplitude_logit_init}"
        )
    if config.one_shot.d2nn.modulation not in {"phase_only", "phase_amplitude"}:
        raise ValueError(
            "Unsupported one_shot.d2nn.modulation: "
            f"{config.one_shot.d2nn.modulation}; expected phase_only or phase_amplitude."
        )
    if _expected_d2nn_modulation(config.one_shot.optical_core_arch) == "phase_only" and (
        config.one_shot.d2nn.modulation != "phase_only"
    ):
        raise ValueError(
            "one_shot.optical_core_arch=diffractive_phase_only requires "
            "d2nn.modulation=phase_only."
        )
    if _expected_d2nn_modulation(config.one_shot.optical_core_arch) == "phase_amplitude" and (
        config.one_shot.d2nn.modulation != "phase_amplitude"
    ):
        raise ValueError(
            "one_shot.optical_core_arch=phase_amplitude_d2nn requires "
            "d2nn.modulation=phase_amplitude."
        )


def _validate_first_stage_training_config(config: TrainingConfig) -> None:
    if config.first_stage.hard_class_weight <= 0.0:
        raise ValueError(
            "first_stage.hard_class_weight must be positive, "
            f"got {config.first_stage.hard_class_weight}"
        )
    for label in config.first_stage.hard_class_weight_labels:
        if int(label) < 0 or int(label) >= int(config.diffusion.num_classes):
            raise ValueError(
                "first_stage.hard_class_weight_labels must be valid class labels "
                f"in [0, {int(config.diffusion.num_classes) - 1}], got {label}"
            )
    if config.first_stage.camera_label_margin_weight < 0.0:
        raise ValueError(
            "first_stage.camera_label_margin_weight must be non-negative, "
            f"got {config.first_stage.camera_label_margin_weight}"
        )
    if config.first_stage.camera_label_margin_target < 0.0:
        raise ValueError(
            "first_stage.camera_label_margin_target must be non-negative, "
            f"got {config.first_stage.camera_label_margin_target}"
        )
    if int(config.first_stage.camera_label_margin_prototypes_per_class) <= 0:
        raise ValueError(
            "first_stage.camera_label_margin_prototypes_per_class must be positive, "
            f"got {config.first_stage.camera_label_margin_prototypes_per_class}"
        )
    for label in config.first_stage.camera_label_margin_labels:
        if int(label) < 0 or int(label) >= int(config.diffusion.num_classes):
            raise ValueError(
                "first_stage.camera_label_margin_labels must be valid class labels "
                f"in [0, {int(config.diffusion.num_classes) - 1}], got {label}"
            )
    allowed_first_stage_selection_metrics = {
        "camera_complex_balanced_mnist_score",
        "camera_class_balanced_mnist_score",
        "camera_balanced_score",
        "camera_label_margin_score",
    }
    if config.first_stage.checkpoint_selection_metric not in allowed_first_stage_selection_metrics:
        raise ValueError(
            "first_stage.checkpoint_selection_metric must be one of "
            f"{sorted(allowed_first_stage_selection_metrics)}, "
            f"got {config.first_stage.checkpoint_selection_metric!r}"
        )


def _validate_one_shot_selection_and_weights(config: TrainingConfig) -> None:
    allowed_one_shot_selection_metrics = {
        "loss",
        "quality",
        "balanced_quality",
        "strict_balanced_quality",
        "strict_readability",
    }
    if config.one_shot.checkpoint_selection_metric not in allowed_one_shot_selection_metrics:
        raise ValueError(
            "one_shot.checkpoint_selection_metric must be one of "
            f"{sorted(allowed_one_shot_selection_metrics)}, "
            f"got {config.one_shot.checkpoint_selection_metric!r}"
        )
    if int(config.one_shot.sample_repeats_per_class) <= 0:
        raise ValueError(
            "one_shot.sample_repeats_per_class must be positive, "
            f"got {config.one_shot.sample_repeats_per_class}"
        )
    if config.one_shot.eval_samples_per_class is not None and int(config.one_shot.eval_samples_per_class) <= 0:
        raise ValueError(
            "one_shot.eval_samples_per_class must be positive when set, "
            f"got {config.one_shot.eval_samples_per_class}"
        )
    if int(config.one_shot.eval_preview_per_class) <= 0:
        raise ValueError(
            "one_shot.eval_preview_per_class must be positive, "
            f"got {config.one_shot.eval_preview_per_class}"
        )
    for name in (
        "camera_recon_weight",
        "clean_foreground_dice_weight",
        "clean_foreground_hole_weight",
        "clean_background_speckle_weight",
        "clean_background_hotspot_weight",
        "clean_background_highfreq_weight",
        "encoder_input_topology_weight",
        "encoder_input_topology_dice_weight",
        "encoder_input_topology_mse_weight",
        "camera_background_speckle_weight",
        "camera_background_hotspot_weight",
        "camera_background_highfreq_weight",
        "camera_readable_contrast_weight",
        "student_label_margin_weight",
        "student_prototype_ce_weight",
        "student_negative_label_margin_weight",
        "student_roi_prototype_ce_weight",
        "student_roi_negative_label_margin_weight",
        "aux_semantic_weight",
        "aux_semantic_prototype_ce_weight",
        "aux_semantic_negative_label_margin_weight",
        "aux_topology_weight",
        "aux_topology_dice_weight",
        "aux_topology_mse_weight",
        "aux_detail_edge_weight",
        "aux_detail_shape_weight",
        "residual_l1_weight",
        "residual_background_weight",
        "sample_code_diversity_weight",
    ):
        value = getattr(config.one_shot, name)
        if float(value) < 0.0:
            raise ValueError(f"one_shot.{name} must be non-negative, got {value}")
    for name in (
        "code_residual_scale",
        "generator_base_lr_scale",
        "generator_residual_lr_scale",
        "diagnostic_slm_override_amplitude",
    ):
        value = getattr(config.one_shot, name)
        if float(value) < 0.0:
            raise ValueError(f"one_shot.{name} must be non-negative, got {value}")
    if (
        config.one_shot.amplitude_learning_rate is not None
        and float(config.one_shot.amplitude_learning_rate) < 0.0
    ):
        raise ValueError(
            "one_shot.amplitude_learning_rate must be non-negative when set, "
            f"got {config.one_shot.amplitude_learning_rate}"
        )


_ONE_SHOT_GENERATOR_ARCHES = {
    "label_template",
    "label_template_code_residual",
    "mlp",
    "template_residual_mlp",
}
_ONE_SHOT_COMMAND_TRANSFORMS = {"direct_amplitude", "intensity_to_amplitude"}
_ONE_SHOT_TEMPLATE_INITIALIZERS = {"none", "mnist_prototype"}
_ONE_SHOT_SLM_OVERRIDE_MODES = {"none", "lowfreq_additive"}


def _validate_one_shot_generator_values(one_shot: OneShotConfig) -> None:
    if one_shot.generator_e_arch not in _ONE_SHOT_GENERATOR_ARCHES:
        raise ValueError(
            "one_shot.generator_e_arch must be one of "
            f"{sorted(_ONE_SHOT_GENERATOR_ARCHES)}, got {one_shot.generator_e_arch!r}"
        )
    if one_shot.generator_command_transform not in _ONE_SHOT_COMMAND_TRANSFORMS:
        raise ValueError(
            "one_shot.generator_command_transform must be one of "
            f"{sorted(_ONE_SHOT_COMMAND_TRANSFORMS)}, "
            f"got {one_shot.generator_command_transform!r}"
        )
    if one_shot.generator_template_init not in _ONE_SHOT_TEMPLATE_INITIALIZERS:
        raise ValueError(
            "one_shot.generator_template_init must be one of "
            f"{sorted(_ONE_SHOT_TEMPLATE_INITIALIZERS)}, "
            f"got {one_shot.generator_template_init!r}"
        )
    if one_shot.diagnostic_slm_override_mode not in _ONE_SHOT_SLM_OVERRIDE_MODES:
        raise ValueError(
            "one_shot.diagnostic_slm_override_mode must be one of "
            f"{sorted(_ONE_SHOT_SLM_OVERRIDE_MODES)}, "
            f"got {one_shot.diagnostic_slm_override_mode!r}"
        )


def _validate_one_shot_generator_arch_requirements(one_shot: OneShotConfig) -> None:
    if one_shot.generator_e_arch == "label_template":
        if one_shot.generator_template_init != "mnist_prototype":
            raise ValueError(
                "one_shot.generator_e_arch='label_template' requires "
                "one_shot.generator_template_init='mnist_prototype'."
            )
        if one_shot.generator_command_transform != "intensity_to_amplitude":
            raise ValueError(
                "one_shot.generator_e_arch='label_template' requires "
                "one_shot.generator_command_transform='intensity_to_amplitude'."
            )
        if one_shot.distill_target_source != "mnist_prototype":
            raise ValueError(
                "one_shot.generator_e_arch='label_template' requires "
                "one_shot.distill_target_source='mnist_prototype'."
            )
    elif one_shot.generator_e_arch == "label_template_code_residual":
        if one_shot.generator_template_init != "mnist_prototype":
            raise ValueError(
                "one_shot.generator_e_arch='label_template_code_residual' requires "
                "one_shot.generator_template_init='mnist_prototype'."
            )
        if one_shot.generator_command_transform != "intensity_to_amplitude":
            raise ValueError(
                "one_shot.generator_e_arch='label_template_code_residual' requires "
                "one_shot.generator_command_transform='intensity_to_amplitude'."
            )
    elif one_shot.generator_template_init != "none":
        raise ValueError(
            "one_shot.generator_template_init is only supported for "
            "one_shot.generator_e_arch='label_template' or "
            "one_shot.generator_e_arch='label_template_code_residual'."
        )


def _validate_one_shot_generator_requirements(one_shot: OneShotConfig) -> None:
    _validate_one_shot_generator_arch_requirements(one_shot)
    if (
        one_shot.generator_command_transform == "intensity_to_amplitude"
        and float(one_shot.slm_background_floor) >= 1.0
    ):
        raise ValueError(
            "one_shot.generator_command_transform='intensity_to_amplitude' requires "
            "one_shot.slm_background_floor < 1.0."
        )
    if (
        one_shot.diagnostic_slm_override_mode == "none"
        and float(one_shot.diagnostic_slm_override_amplitude) != 0.0
    ):
        raise ValueError(
            "one_shot.diagnostic_slm_override_amplitude must be 0.0 when "
            "one_shot.diagnostic_slm_override_mode='none'."
        )


def _validate_one_shot_generator_config(config: TrainingConfig) -> None:
    _validate_one_shot_generator_values(config.one_shot)
    _validate_one_shot_generator_requirements(config.one_shot)


def _validate_one_shot_student_objective_values(one_shot: OneShotConfig) -> None:
    if float(one_shot.student_label_margin_target) < 0.0:
        raise ValueError(
            "one_shot.student_label_margin_target must be non-negative, "
            f"got {one_shot.student_label_margin_target}"
        )
    if float(one_shot.student_label_margin_focus_weight) <= 0.0:
        raise ValueError(
            "one_shot.student_label_margin_focus_weight must be positive, "
            f"got {one_shot.student_label_margin_focus_weight}"
        )
    if float(one_shot.student_prototype_ce_temperature) <= 0.0:
        raise ValueError(
            "one_shot.student_prototype_ce_temperature must be positive, "
            f"got {one_shot.student_prototype_ce_temperature}"
        )
    if float(one_shot.student_roi_prototype_ce_temperature) <= 0.0:
        raise ValueError(
            "one_shot.student_roi_prototype_ce_temperature must be positive, "
            f"got {one_shot.student_roi_prototype_ce_temperature}"
        )
    if float(one_shot.student_roi_foreground_threshold) < 0.0:
        raise ValueError(
            "one_shot.student_roi_foreground_threshold must be non-negative, "
            f"got {one_shot.student_roi_foreground_threshold}"
        )


def _validate_one_shot_aux_objective_config(one_shot: OneShotConfig) -> None:
    allowed_aux_semantic_planes = {
        "",
        "decoder_output_amplitude",
        "chip_intensity_primary",
        "chip_amplitude",
        "probe_plane",
    }
    if one_shot.aux_semantic_plane_key not in allowed_aux_semantic_planes:
        raise ValueError(
            "one_shot.aux_semantic_plane_key must be one of "
            f"{sorted(allowed_aux_semantic_planes)}, "
            f"got {one_shot.aux_semantic_plane_key!r}"
        )
    aux_semantic_active = (
        float(one_shot.aux_semantic_weight) > 0.0
        and (
            float(one_shot.aux_semantic_prototype_ce_weight) > 0.0
            or float(one_shot.aux_semantic_negative_label_margin_weight) > 0.0
        )
    )
    if aux_semantic_active and not one_shot.aux_semantic_plane_key:
        raise ValueError(
            "one_shot.aux_semantic_plane_key is required when aux semantic losses are enabled."
        )
    allowed_aux_topology_planes = {
        "",
        "target_code_amplitude",
        "decoder_input_amplitude",
        "decoder_output_amplitude",
        "chip_intensity_primary",
        "physical_camera_plane",
        "camera_plane",
    }
    if one_shot.aux_topology_plane_key not in allowed_aux_topology_planes:
        raise ValueError(
            "one_shot.aux_topology_plane_key must be one of "
            f"{sorted(allowed_aux_topology_planes)}, "
            f"got {one_shot.aux_topology_plane_key!r}"
        )
    if float(one_shot.aux_topology_weight) > 0.0 and not one_shot.aux_topology_plane_key:
        raise ValueError(
            "one_shot.aux_topology_plane_key is required when aux topology loss is enabled."
        )
    allowed_aux_topology_target_transforms = {"none", "lowpass", "thick_binary"}
    if one_shot.aux_topology_target_transform not in allowed_aux_topology_target_transforms:
        raise ValueError(
            "one_shot.aux_topology_target_transform must be one of "
            f"{sorted(allowed_aux_topology_target_transforms)}, "
            f"got {one_shot.aux_topology_target_transform!r}"
        )


def _validate_one_shot_objective_config(config: TrainingConfig) -> None:
    _validate_one_shot_student_objective_values(config.one_shot)
    _validate_one_shot_aux_objective_config(config.one_shot)


_ONE_SHOT_DECODER_DIAGNOSTIC_MODES = {"normal", "bypass_decoder", "propagation_only_decoder"}
_ONE_SHOT_DECODER_MODULATION_OVERRIDES = {"none", "phase_amplitude"}
_ONE_SHOT_DECODER_MASK_COMPONENTS = {"trained", "phase_only", "amplitude_only"}
_ONE_SHOT_TARGET_CODE_MODES = {
    "none",
    "spatial_identity_code",
    "lowfreq_digit_code",
    "thick_lowfreq_digit_code",
    "sharp_stroke_digit_code",
    "small_sharp_stroke_digit_code",
    "class_codebook_grid",
}
_ONE_SHOT_TARGET_CODE_BRIDGE_SOURCES = {"none", "encoded_amplitude", "ols_output_amplitude"}
_ONE_SHOT_TARGET_CODE_BRIDGE_ARCHES = {"shallow", "deep_residual", "multiscale"}
_ONE_SHOT_TARGET_CODE_BRIDGE_MODES = {
    "spatial_identity_code",
    "lowfreq_digit_code",
    "thick_lowfreq_digit_code",
    "sharp_stroke_digit_code",
    "small_sharp_stroke_digit_code",
}
_ONE_SHOT_OLS_CALIBRATIONS = {"none", "phase_screen", "amplitude_phase_screen"}


def _expected_d2nn_modulation(optical_core_arch: str) -> str | None:
    if optical_core_arch == "diffractive_phase_only":
        return "phase_only"
    if optical_core_arch == "phase_amplitude_d2nn":
        return "phase_amplitude"
    return None


def _target_code_bridge_mode_is_compatible(one_shot: OneShotConfig) -> bool:
    return (
        one_shot.target_code_bridge_source == "none"
        or one_shot.target_code_diagnostic_mode in _ONE_SHOT_TARGET_CODE_BRIDGE_MODES
    )


def _validate_one_shot_decoder_diagnostic_config(one_shot: OneShotConfig) -> None:
    if one_shot.decoder_diagnostic_mode not in _ONE_SHOT_DECODER_DIAGNOSTIC_MODES:
        raise ValueError(
            "one_shot.decoder_diagnostic_mode must be one of "
            f"{sorted(_ONE_SHOT_DECODER_DIAGNOSTIC_MODES)}, "
            f"got {one_shot.decoder_diagnostic_mode!r}"
        )
    if one_shot.decoder_diagnostic_modulation_override not in _ONE_SHOT_DECODER_MODULATION_OVERRIDES:
        raise ValueError(
            "one_shot.decoder_diagnostic_modulation_override must be one of "
            f"{sorted(_ONE_SHOT_DECODER_MODULATION_OVERRIDES)}, "
            f"got {one_shot.decoder_diagnostic_modulation_override!r}"
        )
    if one_shot.decoder_diagnostic_mask_component not in _ONE_SHOT_DECODER_MASK_COMPONENTS:
        raise ValueError(
            "one_shot.decoder_diagnostic_mask_component must be one of "
            f"{sorted(_ONE_SHOT_DECODER_MASK_COMPONENTS)}, "
            f"got {one_shot.decoder_diagnostic_mask_component!r}"
        )
    if (
        one_shot.decoder_diagnostic_mask_component != "trained"
        and one_shot.decoder_diagnostic_mode != "normal"
    ):
        raise ValueError(
            "one_shot.decoder_diagnostic_mask_component phase_only/amplitude_only "
            "requires decoder_diagnostic_mode=normal"
        )
    if (
        one_shot.decoder_diagnostic_mask_component != "trained"
        and one_shot.decoder_diagnostic_modulation_override != "phase_amplitude"
    ):
        raise ValueError(
            "one_shot.decoder_diagnostic_mask_component phase_only/amplitude_only "
            "requires decoder_diagnostic_modulation_override=phase_amplitude"
        )
    phase_layer_mask = tuple(one_shot.decoder_diagnostic_phase_layer_mask)
    if phase_layer_mask:
        decoder_layers = (
            int(one_shot.d2nn.decoder_layers)
            if one_shot.d2nn is not None
            else int(one_shot.decoder_diffractive_layers)
        )
        if len(phase_layer_mask) != decoder_layers:
            raise ValueError(
                "one_shot.decoder_diagnostic_phase_layer_mask length must equal "
                f"decoder layer count {decoder_layers}, got {len(phase_layer_mask)}"
            )
        if not all(type(enabled) is bool for enabled in phase_layer_mask):
            raise ValueError("one_shot.decoder_diagnostic_phase_layer_mask entries must be booleans")
        if not any(phase_layer_mask):
            raise ValueError("one_shot.decoder_diagnostic_phase_layer_mask must enable at least one layer")
        if one_shot.decoder_diagnostic_mode != "normal":
            raise ValueError(
                "one_shot.decoder_diagnostic_phase_layer_mask requires decoder_diagnostic_mode=normal"
            )
        if one_shot.decoder_diagnostic_mask_component != "phase_only":
            raise ValueError(
                "one_shot.decoder_diagnostic_phase_layer_mask requires "
                "decoder_diagnostic_mask_component=phase_only"
            )


def _validate_one_shot_target_code_diagnostic_config(one_shot: OneShotConfig) -> None:
    if one_shot.target_code_diagnostic_mode not in _ONE_SHOT_TARGET_CODE_MODES:
        raise ValueError(
            "one_shot.target_code_diagnostic_mode must be one of "
            f"{sorted(_ONE_SHOT_TARGET_CODE_MODES)}, "
            f"got {one_shot.target_code_diagnostic_mode!r}"
        )
    if one_shot.target_code_spatial_scale is not None:
        target_code_spatial_scale = float(one_shot.target_code_spatial_scale)
        if target_code_spatial_scale <= 0.0 or target_code_spatial_scale > 1.0:
            raise ValueError(
                "one_shot.target_code_spatial_scale must be in (0, 1] when set, "
                f"got {one_shot.target_code_spatial_scale}"
            )
    if one_shot.target_code_bridge_source not in _ONE_SHOT_TARGET_CODE_BRIDGE_SOURCES:
        raise ValueError(
            "one_shot.target_code_bridge_source must be one of "
            f"{sorted(_ONE_SHOT_TARGET_CODE_BRIDGE_SOURCES)}, "
            f"got {one_shot.target_code_bridge_source!r}"
        )
    if one_shot.target_code_bridge_arch not in _ONE_SHOT_TARGET_CODE_BRIDGE_ARCHES:
        raise ValueError(
            "one_shot.target_code_bridge_arch must be one of "
            f"{sorted(_ONE_SHOT_TARGET_CODE_BRIDGE_ARCHES)}, "
            f"got {one_shot.target_code_bridge_arch!r}"
        )
    if int(one_shot.target_code_bridge_hidden_channels) <= 0:
        raise ValueError(
            "one_shot.target_code_bridge_hidden_channels must be positive, "
            f"got {one_shot.target_code_bridge_hidden_channels}"
        )
    if not _target_code_bridge_mode_is_compatible(one_shot):
        raise ValueError(
            "one_shot.target_code_bridge_source requires target_code_diagnostic_mode "
            "to be spatial_identity_code, lowfreq_digit_code, thick_lowfreq_digit_code, "
            "sharp_stroke_digit_code, or small_sharp_stroke_digit_code."
        )


def _validate_one_shot_diagnostic_config(config: TrainingConfig) -> None:
    one_shot = config.one_shot
    _validate_one_shot_decoder_diagnostic_config(one_shot)
    _validate_one_shot_target_code_diagnostic_config(one_shot)
    if one_shot.ols_transfer_calibration not in _ONE_SHOT_OLS_CALIBRATIONS:
        raise ValueError(
            "one_shot.ols_transfer_calibration must be one of "
            f"{sorted(_ONE_SHOT_OLS_CALIBRATIONS)}, "
            f"got {one_shot.ols_transfer_calibration!r}"
        )


def validate_physical_forward_contract(config: TrainingConfig, physics: PhysicsConfig) -> None:
    """Validate the current v55 coherent-relay physical contract."""
    one_shot = config.one_shot
    if not one_shot.physical_forward_contract:
        return
    if one_shot.ols_kind != "identity":
        raise ValueError("physical forward contract requires identity OLS")
    if one_shot.ols_transfer_calibration != "none":
        raise ValueError("physical forward contract forbids OLS transfer calibration")
    if one_shot.optical_core_arch != "diffractive_phase_only":
        raise ValueError("physical forward contract requires diffractive_phase_only core")
    if physics.front_path_model != "coherent_imaging_4f" or physics.back_path_model != "free_space":
        raise ValueError(
            "current physical forward contract requires coherent_imaging_4f front "
            "and thin-lens free_space backend relay"
        )
    if not physics.physical_forward_contract:
        raise ValueError("physical forward contract requires physics.physical_forward_contract=true")
    d2nn = one_shot.d2nn
    if d2nn is None:
        raise ValueError("physical forward contract requires one_shot.d2nn")
    if (d2nn.grid_size, d2nn.pixel_size) != (100, 2.0e-6):
        raise ValueError("physical forward contract requires 100x100 D2NN at 2 um pitch")
    if abs(d2nn.layer_spacing - 1.5e-3) > 1.0e-12:
        raise ValueError("physical forward contract requires 1.50 mm layer spacing")
    if d2nn.modulation != "phase_only":
        raise ValueError("physical forward contract requires phase_only D2NN modulation")
    if (d2nn.encoder_layers, d2nn.decoder_layers) != (2, 2):
        raise ValueError("physical forward contract requires two encoder and two decoder layers")
    if abs(physics.optics.pixel_size - 2.5e-6) > 1.0e-12 or physics.geometry.grid_size != 200:
        raise ValueError("physical forward contract requires 200x200 SLM/CMOS at 2.5 um pitch")
    if physics.front_path_model == "coherent_imaging_4f":
        macro = physics.macro_optics
        if abs(macro.dist_slm_to_bs + macro.dist_bs_to_m3 + macro.dist_m3_to_front_lens1 - macro.front_lens1_focal_length) > 1.0e-9:
            raise ValueError("formal front relay SLM-to-L1 distance must equal f1")
        if abs(macro.dist_front_lens1_to_lens2 - (macro.front_lens1_focal_length + macro.front_lens2_focal_length)) > 1.0e-9:
            raise ValueError("formal front relay L1-to-L2 distance must equal f1+f2")
        if abs(macro.dist_front_lens2_to_encoder - macro.front_lens2_focal_length) > 1.0e-9:
            raise ValueError("formal front relay L2-to-encoder distance must equal f2")
        if abs(
            macro.dist_slm_to_bs
            + macro.dist_bs_to_m3
            + macro.dist_m3_to_front_lens1
            + macro.dist_front_lens1_to_lens2
            + macro.dist_front_lens2_to_encoder
            - 0.625
        ) > 1.0e-9:
            raise ValueError("formal front relay mechanical distance must equal 625 mm")
        if abs(macro.front_lens2_focal_length / macro.front_lens1_focal_length - 0.4) > 1.0e-9:
            raise ValueError("front 4f relay requires magnification -0.4")
        if macro.front_lens1_clear_aperture < 0.02375 or macro.front_lens2_clear_aperture < 0.02375:
            raise ValueError("front relay clear apertures must be at least 23.75 mm")
        if macro.back_relay_clear_aperture < 0.0266:
            raise ValueError("back relay clear aperture must be at least 26.60 mm")
    if one_shot.decoder_diagnostic_mode != "normal":
        raise ValueError("physical forward contract requires normal decoder mode")
    if one_shot.decoder_diagnostic_modulation_override != "none":
        raise ValueError("physical forward contract forbids decoder modulation overrides")
    if one_shot.target_code_diagnostic_mode != "none" or one_shot.target_code_bridge_source != "none":
        raise ValueError("physical forward contract forbids target-code intervention")


def _validate_one_shot_target_config(config: TrainingConfig) -> None:
    one_shot = config.one_shot
    if float(one_shot.camera_readable_contrast_target) < 0.0:
        raise ValueError(
            "one_shot.camera_readable_contrast_target must be non-negative, "
            f"got {one_shot.camera_readable_contrast_target}"
        )
    if float(one_shot.camera_background_hotspot_threshold) < 0.0:
        raise ValueError(
            "one_shot.camera_background_hotspot_threshold must be non-negative, "
            f"got {one_shot.camera_background_hotspot_threshold}"
        )
    if int(one_shot.aux_detail_foreground_gate_dilation) < 0:
        raise ValueError(
            "one_shot.aux_detail_foreground_gate_dilation must be non-negative, "
            f"got {one_shot.aux_detail_foreground_gate_dilation}"
        )
    allowed_student_prototype_sources = {"mnist", "teacher_image", "teacher_camera"}
    if one_shot.student_label_margin_prototype_source not in allowed_student_prototype_sources:
        raise ValueError(
            "one_shot.student_label_margin_prototype_source must be one of "
            f"{sorted(allowed_student_prototype_sources)}, "
            f"got {one_shot.student_label_margin_prototype_source!r}"
        )
    allowed_distill_target_sources = {
        "teacher_camera",
        "clean_teacher_camera",
        "mnist_prototype",
        "mnist_sample",
        "mnist_nearest_teacher_image",
    }
    if one_shot.distill_target_source not in allowed_distill_target_sources:
        raise ValueError(
            "one_shot.distill_target_source must be one of "
            f"{sorted(allowed_distill_target_sources)}, "
            f"got {one_shot.distill_target_source!r}"
        )
    allowed_aux_detail_target_sources = {"none", *allowed_distill_target_sources}
    if one_shot.aux_detail_target_source not in allowed_aux_detail_target_sources:
        raise ValueError(
            "one_shot.aux_detail_target_source must be one of "
            f"{sorted(allowed_aux_detail_target_sources)}, "
            f"got {one_shot.aux_detail_target_source!r}"
        )


def _validate_one_shot_class_labels(labels, *, field_name: str, num_classes: int) -> None:
    for label in labels:
        if int(label) < 0 or int(label) >= num_classes:
            raise ValueError(
                f"one_shot.{field_name} must be valid class labels "
                f"in [0, {num_classes - 1}], got {label}"
            )


def _validate_one_shot_margin_pairs(
    pairs,
    *,
    field_name: str,
    entry_shape: str,
    num_classes: int,
) -> None:
    for pair in pairs:
        if len(pair) != 3:
            raise ValueError(
                f"one_shot.{field_name} entries must be {entry_shape}, got {pair!r}"
            )
        class_a, class_b, target_margin = pair
        for label in (int(class_a), int(class_b)):
            if label < 0 or label >= num_classes:
                raise ValueError(
                    f"one_shot.{field_name} class labels must be valid "
                    f"in [0, {num_classes - 1}], got {label}"
                )
        if float(target_margin) < 0.0:
            raise ValueError(
                f"one_shot.{field_name} target margins must be non-negative, "
                f"got {target_margin}"
            )


def _validate_one_shot_label_config(config: TrainingConfig) -> None:
    one_shot = config.one_shot
    num_classes = int(one_shot.num_classes)
    for field_name in (
        "student_label_margin_labels",
        "student_prototype_ce_labels",
        "student_roi_prototype_ce_labels",
    ):
        _validate_one_shot_class_labels(
            getattr(one_shot, field_name),
            field_name=field_name,
            num_classes=num_classes,
        )
    if one_shot.student_label_margin_focus_label is not None:
        focus_label = int(one_shot.student_label_margin_focus_label)
        if focus_label < 0 or focus_label >= num_classes:
            raise ValueError(
                "one_shot.student_label_margin_focus_label must be a valid class label "
                f"in [0, {num_classes - 1}], got {focus_label}"
            )
    for field_name, entry_shape in (
        ("student_pair_contrast_config", "[class_a, class_b, target_margin]"),
        ("student_negative_label_margin_config", "[target_class, negative_class, target_margin]"),
        ("student_roi_negative_label_margin_config", "[target_class, negative_class, target_margin]"),
    ):
        _validate_one_shot_margin_pairs(
            getattr(one_shot, field_name),
            field_name=field_name,
            entry_shape=entry_shape,
            num_classes=num_classes,
        )


def load_training_config(path: str | Path | None = None) -> TrainingConfig:
    if path is None:
        return TrainingConfig()

    data = _load_yaml_file(path)
    sections = _normalize_training_config_sections(data)
    config = _build_training_config(sections)
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
