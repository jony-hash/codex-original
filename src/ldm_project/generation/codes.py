from __future__ import annotations

import math


ONE_SHOT_RAW_DIAGNOSTIC_TENSOR_KEYS = (
    "generator_base_command",
    "generator_desired_intensity",
    "generator_residual_map",
    "generator_residual_abs_mean",
    "generator_residual_bg_abs_mean",
    "diagnostic_slm_override_pattern",
    "diagnostic_slm_override_base_command",
    "slm_command",
    "slm_amplitude",
    "slm_phase",
    "encoder_input_amplitude",
    "encoder_input_phase",
    "encoder_input_grid_amplitude",
    "encoder_input_grid_phase",
    "encoded_amplitude",
    "encoded_phase",
    "ols_input_amplitude",
    "ols_input_phase",
    "ols_output_amplitude",
    "ols_output_phase",
    "ols_amplitude",
    "ols_phase",
    "target_code_amplitude",
    "decoder_input_amplitude",
    "decoder_input_phase",
    "decoder_output_amplitude",
    "decoder_output_phase",
    "diffractive_encoder_layer_amplitude",
    "diffractive_encoder_layer_phase",
    "diffractive_decoder_layer_amplitude",
    "diffractive_decoder_layer_phase",
    "diffractive_encoder_mask_amplitude",
    "diffractive_encoder_mask_phase",
    "diffractive_decoder_mask_amplitude",
    "diffractive_decoder_mask_phase",
    "diffractive_camera_intensity_raw",
    "diffractive_camera_plane_raw",
    "diffractive_camera_plane_loss_raw",
    "physical_camera_plane",
    "physical_camera_plane_loss",
    "physical_camera_intensity_raw",
    "primary_readout_plane_native",
    "primary_readout_plane_loss_native",
    "chip_intensity_primary",
    "chip_intensity_primary_loss",
    "chip_amplitude_primary",
    "chip_amplitude_primary_loss",
    "chip_amplitude",
    "chip_phase",
    "probe_plane",
    "probe_plane_loss",
    "camera_phase",
)
ONE_SHOT_DISPLAY_DIAGNOSTIC_TENSOR_KEYS = (
    "slm_phase_display",
    "encoder_input_amplitude_display",
    "encoder_input_phase_display",
    "encoder_input_grid_amplitude_display",
    "encoder_input_grid_phase_display",
    "encoded_amplitude_display",
    "encoded_phase_display",
    "ols_input_amplitude_display",
    "ols_input_phase_display",
    "ols_output_amplitude_display",
    "ols_output_phase_display",
    "ols_amplitude_display",
    "ols_phase_display",
    "target_code_amplitude_display",
    "decoder_input_amplitude_display",
    "decoder_input_phase_display",
    "decoder_output_amplitude_display",
    "decoder_output_phase_display",
    "chip_field_amplitude_display",
    "chip_intensity_display",
    "chip_phase_display",
)
ONE_SHOT_DISTILL_EVAL_PREVIEW_PER_CLASS = 2
ONE_SHOT_DISTILL_EVAL_FOCUS_TENSOR_KEYS = (
    "teacher_camera_display",
    "student_camera_display",
    "abs_error",
    "chip_field_amplitude_display",
    "chip_intensity_display",
    "chip_phase_display",
)


def _detach_cpu_tensors(tensor_map, keys: tuple[str, ...]) -> dict[str, object]:
    return {key: tensor_map[key].detach().cpu() for key in keys if key in tensor_map}


def _sample_one_shot_codes(torch_module, batch_size: int, train_cfg, device):
    return torch_module.randn(batch_size, train_cfg.one_shot.code_dim, device=device)


def _sample_one_shot_codes_with_seed(
    torch_module, sample_count: int, code_dim: int, seed: int, device
):
    generator = torch_module.Generator().manual_seed(int(seed))
    return torch_module.randn(sample_count, code_dim, generator=generator).to(device)


def project_codes_to_teacher_noise(
    torch_module,
    codes,
    sample_shape: tuple[int, int, int, int],
    projection_seed: int,
    device,
):
    """Project deterministic code vectors into a teacher noise-shaped tensor."""
    if codes.ndim != 2:
        raise ValueError(
            f"codes must be [batch, code_dim], got shape {tuple(codes.shape)}"
        )
    flat_size = int(math.prod(sample_shape[1:]))
    generator = torch_module.Generator().manual_seed(int(projection_seed))
    projection = torch_module.randn(
        codes.shape[1], flat_size, generator=generator, dtype=codes.dtype
    )
    projected = codes.detach().cpu() @ (projection / math.sqrt(float(codes.shape[1])))
    return projected.to(device=device).reshape(sample_shape)


_project_codes_to_teacher_noise = project_codes_to_teacher_noise

# Public neutral sampling boundary; keep the underscore alias for compatibility.
sample_one_shot_codes_with_seed = _sample_one_shot_codes_with_seed
