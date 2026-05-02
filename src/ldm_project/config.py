from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any


@dataclass
class OpticsConfig:
    wavelength: float = 532.0e-9
    pixel_size: float = 3.0e-6


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
    dist_laser_to_m1: float = 0.115
    dist_m1_to_m2: float = 0.215
    dist_m2_to_pf1: float = 0.215
    dist_pf1_to_pbs: float = 0.115
    dist_pbs_to_bs: float = 0.088
    dist_bs_to_slm: float = 0.038
    dist_slm_to_m3: float = 0.140
    dist_m3_to_4f: float = 0.063
    f_4f: float = 0.100
    dist_4f_to_pf2: float = 0.088
    dist_chip_to_plane: float = 0.003
    dist_plane_to_lens: float = 0.105
    f_relay: float = 0.075
    dist_lens_to_cam: float = 0.260


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
    data: DataConfig = field(default_factory=DataConfig)
    first_stage: FirstStageConfig = field(default_factory=FirstStageConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)


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
    data_config = DataConfig(**data_cfg)
    if data_config.validation_fraction < 0.0 or data_config.validation_fraction >= 1.0:
        raise ValueError(
            "data.validation_fraction must be in [0.0, 1.0), "
            f"got {data_config.validation_fraction}"
        )

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
        data=data_config,
        first_stage=FirstStageConfig(
            electronic_domain=ElectronicDomainConfig(**electronic_domain_data),
            **first_stage_data,
        ),
        diffusion=DiffusionConfig(**diffusion_data),
    )
