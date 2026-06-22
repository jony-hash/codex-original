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
    dist_m3_to_lg: float = 0.475
    dist_lg_to_relay_lens: float = 0.100
    dist_relay_lens_to_cmos: float = 0.250
    dist_lg_to_cmos: float = 0.350
    f_relay: float = 0.0714285714


@dataclass
class PhysicsConfig:
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    readout: ReadoutConfig = field(default_factory=ReadoutConfig)
    macro_optics: MacroOpticsConfig = field(default_factory=MacroOpticsConfig)
    front_path_model: str = "free_space"
    back_path_model: str = "free_space"
    front_relay_psf_sigma_px: float = 0.0

    def __post_init__(self) -> None:
        front_allowed = {"free_space", "relay_identity", "relay_coherent_psf"}
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
    ols_kind: str = "identity"
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
    student_label_margin_weight: float = 0.0
    student_label_margin_target: float = 0.03
    student_label_margin_labels: tuple[int, ...] = ()
    student_label_margin_prototype_source: str = "mnist"
    student_label_margin_focus_label: int | None = None
    student_label_margin_focus_weight: float = 1.0
    code_diversity_weight: float = 0.0
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


def load_training_config(path: str | Path | None = None) -> TrainingConfig:
    if path is None:
        return TrainingConfig()

    data = _load_yaml_file(path)
    training_data = data.get("training", {})
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
    data_cfg = data.get("data", {})
    physics_data = data.get("physics")
    first_stage_data = data.get("first_stage", {}).copy()
    diffusion_data = data.get("diffusion", {})
    one_shot_data = data.get("one_shot", {}).copy()
    d2nn_data = one_shot_data.pop("d2nn", None)
    one_shot_data.pop("diffractive_modulation", None)
    distillation_data = data.get("distillation", {}).copy()
    electronic_domain_data = first_stage_data.pop("electronic_domain", {})

    ols_data = first_stage_data.pop("ols", {})
    if "kind" in ols_data:
        first_stage_data["ols_kind"] = ols_data["kind"]
    if "encoder_channels" in first_stage_data:
        first_stage_data["encoder_channels"] = tuple(first_stage_data["encoder_channels"])
    if "hard_class_weight_labels" in first_stage_data:
        first_stage_data["hard_class_weight_labels"] = tuple(first_stage_data["hard_class_weight_labels"])
    if "camera_label_margin_labels" in first_stage_data:
        first_stage_data["camera_label_margin_labels"] = tuple(first_stage_data["camera_label_margin_labels"])
    if "hidden_channels" in diffusion_data and "base_channels" not in diffusion_data:
        diffusion_data["base_channels"] = diffusion_data.pop("hidden_channels")
    if "channel_multipliers" in diffusion_data:
        diffusion_data["channel_multipliers"] = tuple(diffusion_data["channel_multipliers"])
    if "attention_levels" in diffusion_data:
        diffusion_data["attention_levels"] = tuple(diffusion_data["attention_levels"])
    if "sample_class_labels" in diffusion_data:
        diffusion_data["sample_class_labels"] = tuple(diffusion_data["sample_class_labels"])
    if "class_weight_labels" in diffusion_data:
        diffusion_data["class_weight_labels"] = tuple(diffusion_data["class_weight_labels"])
    if "sample_class_labels" in one_shot_data:
        one_shot_data["sample_class_labels"] = tuple(one_shot_data["sample_class_labels"])
    if "class_recon_weight_labels" in one_shot_data:
        one_shot_data["class_recon_weight_labels"] = tuple(one_shot_data["class_recon_weight_labels"])
    if "student_label_margin_labels" in one_shot_data:
        one_shot_data["student_label_margin_labels"] = tuple(one_shot_data["student_label_margin_labels"])
    if "student_pair_contrast_config" in one_shot_data:
        one_shot_data["student_pair_contrast_config"] = tuple(
            tuple(item) for item in one_shot_data["student_pair_contrast_config"]
        )
    if "class_labels" in distillation_data:
        distillation_data["class_labels"] = tuple(distillation_data["class_labels"])
    training_seed = training_data.get("seed", TrainingConfig.seed)
    if training_seed is not None:
        training_seed = int(training_seed)
        if training_seed < 0:
            raise ValueError(f"training.seed must be non-negative, got {training_seed}")
    data_config = DataConfig(**data_cfg)
    if data_config.validation_fraction < 0.0 or data_config.validation_fraction >= 1.0:
        raise ValueError(
            "data.validation_fraction must be in [0.0, 1.0), "
            f"got {data_config.validation_fraction}"
        )

    d2nn_config = D2NNConfig(**d2nn_data) if d2nn_data else None
    physics_config = None
    if physics_data is not None:
        physics_config = PhysicsConfig(
            optics=OpticsConfig(**physics_data.get("optics", {})),
            geometry=GeometryConfig(**physics_data.get("geometry", {})),
            readout=ReadoutConfig(**physics_data.get("readout", {})),
            macro_optics=MacroOpticsConfig(**physics_data.get("macro_optics", {})),
            front_path_model=physics_data.get("front_path_model", "free_space"),
            back_path_model=physics_data.get("back_path_model", "free_space"),
            front_relay_psf_sigma_px=float(physics_data.get("front_relay_psf_sigma_px", 0.0)),
        )

    config = TrainingConfig(
        mode=training_data.get("mode", TrainingConfig.mode),
        batch_size=training_data.get("batch_size", TrainingConfig.batch_size),
        learning_rate=training_data.get("learning_rate", TrainingConfig.learning_rate),
        seed=training_seed,
        step_log_interval=training_data.get("step_log_interval", TrainingConfig.step_log_interval),
        visual_save_epoch_interval=training_data.get(
            "visual_save_epoch_interval",
            TrainingConfig.visual_save_epoch_interval,
        ),
        validation_epoch_interval=training_data.get(
            "validation_epoch_interval",
            TrainingConfig.validation_epoch_interval,
        ),
        one_shot_distill_epoch_checkpoint_interval=training_data.get(
            "one_shot_distill_epoch_checkpoint_interval",
            TrainingConfig.one_shot_distill_epoch_checkpoint_interval,
        ),
        first_stage_epochs=training_data.get("first_stage_epochs", TrainingConfig.first_stage_epochs),
        diffusion_epochs=training_data.get("diffusion_epochs", TrainingConfig.diffusion_epochs),
        one_shot_epochs=training_data.get("one_shot_epochs", TrainingConfig.one_shot_epochs),
        data=data_config,
        first_stage=FirstStageConfig(
            electronic_domain=ElectronicDomainConfig(**electronic_domain_data),
            **first_stage_data,
        ),
        diffusion=DiffusionConfig(**diffusion_data),
        one_shot=OneShotConfig(d2nn=d2nn_config, **one_shot_data),
        distillation=DistillationConfig(**distillation_data),
        physics=physics_config,
    )
    if config.one_shot.d2nn is not None:
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
        if config.one_shot.optical_core_arch == "diffractive_phase_only" and (
            config.one_shot.d2nn.modulation != "phase_only"
        ):
            raise ValueError(
                "one_shot.optical_core_arch=diffractive_phase_only requires "
                "d2nn.modulation=phase_only."
            )
        if config.one_shot.optical_core_arch == "phase_amplitude_d2nn" and (
            config.one_shot.d2nn.modulation != "phase_amplitude"
        ):
            raise ValueError(
                "one_shot.optical_core_arch=phase_amplitude_d2nn requires "
                "d2nn.modulation=phase_amplitude."
            )
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
        "camera_background_speckle_weight",
        "camera_background_hotspot_weight",
        "camera_background_highfreq_weight",
        "camera_readable_contrast_weight",
        "student_label_margin_weight",
        "aux_detail_edge_weight",
        "aux_detail_shape_weight",
        "residual_l1_weight",
        "residual_background_weight",
    ):
        value = getattr(config.one_shot, name)
        if float(value) < 0.0:
            raise ValueError(f"one_shot.{name} must be non-negative, got {value}")
    for name in (
        "code_residual_scale",
        "generator_base_lr_scale",
        "generator_residual_lr_scale",
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
    allowed_generator_arches = {"label_template", "mlp", "template_residual_mlp"}
    if config.one_shot.generator_e_arch not in allowed_generator_arches:
        raise ValueError(
            "one_shot.generator_e_arch must be one of "
            f"{sorted(allowed_generator_arches)}, got {config.one_shot.generator_e_arch!r}"
        )
    allowed_command_transforms = {"direct_amplitude", "intensity_to_amplitude"}
    if config.one_shot.generator_command_transform not in allowed_command_transforms:
        raise ValueError(
            "one_shot.generator_command_transform must be one of "
            f"{sorted(allowed_command_transforms)}, "
            f"got {config.one_shot.generator_command_transform!r}"
        )
    allowed_template_initializers = {"none", "mnist_prototype"}
    if config.one_shot.generator_template_init not in allowed_template_initializers:
        raise ValueError(
            "one_shot.generator_template_init must be one of "
            f"{sorted(allowed_template_initializers)}, "
            f"got {config.one_shot.generator_template_init!r}"
        )
    if config.one_shot.generator_e_arch == "label_template":
        if config.one_shot.generator_template_init != "mnist_prototype":
            raise ValueError(
                "one_shot.generator_e_arch='label_template' requires "
                "one_shot.generator_template_init='mnist_prototype'."
            )
        if config.one_shot.generator_command_transform != "intensity_to_amplitude":
            raise ValueError(
                "one_shot.generator_e_arch='label_template' requires "
                "one_shot.generator_command_transform='intensity_to_amplitude'."
            )
        if config.one_shot.distill_target_source != "mnist_prototype":
            raise ValueError(
                "one_shot.generator_e_arch='label_template' requires "
                "one_shot.distill_target_source='mnist_prototype'."
            )
    elif config.one_shot.generator_template_init != "none":
        raise ValueError(
            "one_shot.generator_template_init is only supported for "
            "one_shot.generator_e_arch='label_template'."
        )
    if (
        config.one_shot.generator_command_transform == "intensity_to_amplitude"
        and float(config.one_shot.slm_background_floor) >= 1.0
    ):
        raise ValueError(
            "one_shot.generator_command_transform='intensity_to_amplitude' requires "
            "one_shot.slm_background_floor < 1.0."
        )
    if float(config.one_shot.student_label_margin_target) < 0.0:
        raise ValueError(
            "one_shot.student_label_margin_target must be non-negative, "
            f"got {config.one_shot.student_label_margin_target}"
        )
    if float(config.one_shot.student_label_margin_focus_weight) <= 0.0:
        raise ValueError(
            "one_shot.student_label_margin_focus_weight must be positive, "
            f"got {config.one_shot.student_label_margin_focus_weight}"
        )
    if float(config.one_shot.camera_readable_contrast_target) < 0.0:
        raise ValueError(
            "one_shot.camera_readable_contrast_target must be non-negative, "
            f"got {config.one_shot.camera_readable_contrast_target}"
        )
    if float(config.one_shot.camera_background_hotspot_threshold) < 0.0:
        raise ValueError(
            "one_shot.camera_background_hotspot_threshold must be non-negative, "
            f"got {config.one_shot.camera_background_hotspot_threshold}"
        )
    if int(config.one_shot.aux_detail_foreground_gate_dilation) < 0:
        raise ValueError(
            "one_shot.aux_detail_foreground_gate_dilation must be non-negative, "
            f"got {config.one_shot.aux_detail_foreground_gate_dilation}"
        )
    allowed_student_prototype_sources = {"mnist", "teacher_image", "teacher_camera"}
    if config.one_shot.student_label_margin_prototype_source not in allowed_student_prototype_sources:
        raise ValueError(
            "one_shot.student_label_margin_prototype_source must be one of "
            f"{sorted(allowed_student_prototype_sources)}, "
            f"got {config.one_shot.student_label_margin_prototype_source!r}"
        )
    allowed_distill_target_sources = {
        "teacher_camera",
        "clean_teacher_camera",
        "mnist_prototype",
        "mnist_sample",
        "mnist_nearest_teacher_image",
    }
    if config.one_shot.distill_target_source not in allowed_distill_target_sources:
        raise ValueError(
            "one_shot.distill_target_source must be one of "
            f"{sorted(allowed_distill_target_sources)}, "
            f"got {config.one_shot.distill_target_source!r}"
        )
    allowed_aux_detail_target_sources = {"none", *allowed_distill_target_sources}
    if config.one_shot.aux_detail_target_source not in allowed_aux_detail_target_sources:
        raise ValueError(
            "one_shot.aux_detail_target_source must be one of "
            f"{sorted(allowed_aux_detail_target_sources)}, "
            f"got {config.one_shot.aux_detail_target_source!r}"
        )
    for label in config.one_shot.student_label_margin_labels:
        if int(label) < 0 or int(label) >= int(config.one_shot.num_classes):
            raise ValueError(
                "one_shot.student_label_margin_labels must be valid class labels "
                f"in [0, {int(config.one_shot.num_classes) - 1}], got {label}"
            )
    if config.one_shot.student_label_margin_focus_label is not None:
        focus_label = int(config.one_shot.student_label_margin_focus_label)
        if focus_label < 0 or focus_label >= int(config.one_shot.num_classes):
            raise ValueError(
                "one_shot.student_label_margin_focus_label must be a valid class label "
                f"in [0, {int(config.one_shot.num_classes) - 1}], got {focus_label}"
            )
    for pair in config.one_shot.student_pair_contrast_config:
        if len(pair) != 3:
            raise ValueError(
                "one_shot.student_pair_contrast_config entries must be [class_a, class_b, target_margin], "
                f"got {pair!r}"
            )
        class_a, class_b, target_margin = pair
        for label in (int(class_a), int(class_b)):
            if label < 0 or label >= int(config.one_shot.num_classes):
                raise ValueError(
                    "one_shot.student_pair_contrast_config class labels must be valid "
                    f"in [0, {int(config.one_shot.num_classes) - 1}], got {label}"
                )
        if float(target_margin) < 0.0:
            raise ValueError(
                "one_shot.student_pair_contrast_config target margins must be non-negative, "
                f"got {target_margin}"
            )
    return config
