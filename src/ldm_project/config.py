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


@dataclass
class OneShotConfig:
    code_dim: int = 128
    label_embed_dim: int = 32
    hidden_dim: int = 256
    num_classes: int = 10
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
    code_diversity_weight: float = 0.0
    teacher_pairwise_diversity_weight: float = 0.0
    cross_class_pairwise_weight: float = 0.0
    diversity_min_teacher_mse: float = 1.0e-4
    checkpoint_selection_metric: str = "loss"
    sample_class_labels: tuple[int, ...] = ()
    sample_repeats_per_class: int = 2


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
    step_log_interval: int = 100
    visual_save_epoch_interval: int = 1
    validation_epoch_interval: int = 1
    first_stage_epochs: int = 10
    diffusion_epochs: int = 20
    one_shot_epochs: int = 1
    data: DataConfig = field(default_factory=DataConfig)
    first_stage: FirstStageConfig = field(default_factory=FirstStageConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    one_shot: OneShotConfig = field(default_factory=OneShotConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)


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
    if "class_labels" in distillation_data:
        distillation_data["class_labels"] = tuple(distillation_data["class_labels"])
    data_config = DataConfig(**data_cfg)
    if data_config.validation_fraction < 0.0 or data_config.validation_fraction >= 1.0:
        raise ValueError(
            "data.validation_fraction must be in [0.0, 1.0), "
            f"got {data_config.validation_fraction}"
        )

    d2nn_config = D2NNConfig(**d2nn_data) if d2nn_data else None

    return TrainingConfig(
        mode=training_data.get("mode", TrainingConfig.mode),
        batch_size=training_data.get("batch_size", TrainingConfig.batch_size),
        learning_rate=training_data.get("learning_rate", TrainingConfig.learning_rate),
        step_log_interval=training_data.get("step_log_interval", TrainingConfig.step_log_interval),
        visual_save_epoch_interval=training_data.get(
            "visual_save_epoch_interval",
            TrainingConfig.visual_save_epoch_interval,
        ),
        validation_epoch_interval=training_data.get(
            "validation_epoch_interval",
            TrainingConfig.validation_epoch_interval,
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
    )
