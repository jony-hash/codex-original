"""Canonical configuration validators preserving the legacy error contract."""
from __future__ import annotations

import math

from .schema import OneShotConfig, PhysicsConfig, TrainingConfig

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
    allowed_camera_normalization = {"per_sample_percentile", "fixed_exposure"}
    camera_normalization = config.one_shot.diagnostic_camera_normalization
    if camera_normalization not in allowed_camera_normalization:
        raise ValueError(
            "one_shot.diagnostic_camera_normalization must be one of "
            f"{sorted(allowed_camera_normalization)}, got {camera_normalization!r}"
        )
    if camera_normalization == "fixed_exposure":
        scale = config.one_shot.diagnostic_camera_exposure_scale
        if scale is None or not math.isfinite(float(scale)) or float(scale) <= 0.0:
            raise ValueError(
                "one_shot.diagnostic_camera_exposure_scale must be positive and finite "
                "when diagnostic_camera_normalization='fixed_exposure'."
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
