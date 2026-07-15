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
