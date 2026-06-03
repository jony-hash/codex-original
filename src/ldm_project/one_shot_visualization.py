from __future__ import annotations

import torch
import torch.nn.functional as F

from .d2nn_visualization import (
    _broadcast_d2nn_mask_visualization,
    _d2nn_field_layer_visualization_columns,
    _interleave_d2nn_mask_layer_columns,
)
from .visualization_core import (
    VISUALIZATION_AMPLITUDE_KEY,
    VISUALIZATION_PHASE_KEY,
    _amplitude_for_visualization,
    _blank_tensor_like,
    _code_for_visualization,
    _intensity_for_visualization,
    _normalize_for_visualization,
    _phase_for_visualization,
    _phase_tensor_map,
    _raw_amplitude_for_visualization,
    _robust_amplitude_display,
    _tensor_stats,
    _tensor_diff_summary,
    enhance_camera_for_display,
)

ONE_SHOT_SHARED_SCALE_AMPLITUDE_QUANTILE = 0.995
ONE_SHOT_EARLY_RAW_AMPLITUDE_SPECS = (
    ("slm_amplitude", "slm out amp"),
    ("encoder_input_amplitude", "encoder in amp"),
    ("encoder_input_grid_amplitude", "encoder grid in amp"),
)
ONE_SHOT_SHARED_SCALE_AMPLITUDE_SPECS = (
    ("encoded_amplitude", "encoder out amp"),
    ("ols_input_amplitude", "ols in amp"),
    ("ols_output_amplitude", "ols out amp"),
    ("decoder_input_amplitude", "decoder in amp"),
    ("decoder_output_amplitude", "decoder out amp"),
    ("chip_amplitude", "chip amp"),
)
# 相位数据键到振幅列名的映射，None 表示该列无相位等效数据，填入空白。
_AMPLITUDE_KEY_TO_PHASE_SOURCE: dict[str, str | None] = {
    "digit": "digit",
    "noise code": None,
    "teacher": None,
    "reference": None,
    "camera": "camera_phase",
    "slm command": None,
    "slm amplitude grid": "slm_phase",
    "slm in cmd": None,
    "slm in cmd contrast": None,
    "slm out amp": "slm_phase",
    "encoder in amp": "encoder_input_phase",
    "encoder grid in amp": "encoder_input_grid_phase",
    "encoder field after L1": None,
    "encoder field after L2": None,
    "decoder field after L1": None,
    "decoder field after L2": None,
    "encoder out amp": "encoded_phase",
    "ols in amp": "ols_input_phase",
    "ols out amp": "ols_output_phase",
    "decoder in amp": "decoder_input_phase",
    "decoder out amp": "decoder_output_phase",
    "chip amp": "chip_phase",
    "chip I": None,
    "student": "camera_phase",
    "abs error": None,
}


def _shared_amplitude_scale(tensors, quantile: float = ONE_SHOT_SHARED_SCALE_AMPLITUDE_QUANTILE) -> float:
    if quantile <= 0.0 or quantile > 1.0:
        raise ValueError(f"shared amplitude quantile must be within (0, 1], got {quantile}")
    values = [tensor.detach().float().clamp_min(0.0).flatten().cpu() for tensor in tensors]
    if not values:
        return 1.0
    if len(values) == 1:
        concatenated = values[0]
    else:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to build shared-scale visualizations.") from exc

        concatenated = torch.cat(values)
    if concatenated.numel() == 0:
        return 1.0
    scale = float(concatenated.quantile(quantile).item())
    return max(scale, 1e-6)


def _scale_amplitude_for_visualization(tensor, scale: float):
    return (tensor.detach().float().clamp_min(0.0) / max(float(scale), 1e-6)).clamp(0.0, 1.0)


def _slm_contrast_for_visualization(tensor):
    return _normalize_for_visualization(tensor.detach().float())


def _block_grid_preview(tensor, grid_size: int = 40):
    raw = tensor.detach().float()
    if raw.ndim < 4:
        return _normalize_for_visualization(raw)
    coarse = F.interpolate(raw, size=(grid_size, grid_size), mode="area")
    blocky = F.interpolate(coarse, size=raw.shape[-2:], mode="nearest")
    return _normalize_for_visualization(blocky)


def _build_phase_aligned_to_amplitude(
    amplitude: dict[str, object],
    sample_outputs: dict[str, torch.Tensor],
    original_images: torch.Tensor | None = None,
) -> dict[str, object]:
    phase: dict[str, object] = {}
    blank_ref: torch.Tensor | None = None
    for amp_key in amplitude:
        source = _AMPLITUDE_KEY_TO_PHASE_SOURCE.get(amp_key)
        if source is None:
            if blank_ref is None:
                blank_ref = _blank_tensor_like(next(iter(amplitude.values())))
            phase[amp_key] = blank_ref
        elif source == "digit":
            phase[amp_key] = original_images if original_images is not None else (_blank_tensor_like(next(iter(amplitude.values()))) if blank_ref is None else blank_ref)
        elif source in sample_outputs:
            phase[amp_key] = _phase_for_visualization(sample_outputs[source])
        else:
            if blank_ref is None:
                blank_ref = _blank_tensor_like(next(iter(amplitude.values())))
            phase[amp_key] = blank_ref
    return phase



def _build_phase_with_d2nn_columns(
    amplitude: dict[str, object],
    sample_outputs,
    encoder_phase_columns: dict[str, object],
    decoder_phase_columns: dict[str, object],
    *,
    original_images=None,
) -> dict[str, object]:
    aligned = _build_phase_aligned_to_amplitude(amplitude, sample_outputs, original_images=original_images)
    phase: dict[str, object] = {}
    for key, value in aligned.items():
        if key.startswith("encoder mask amp = 1 "):
            phase_key = key.replace("encoder mask amp = 1", "encoder mask phase", 1)
            phase[phase_key] = encoder_phase_columns[phase_key]
        elif key.startswith("decoder mask amp = 1 "):
            phase_key = key.replace("decoder mask amp = 1", "decoder mask phase", 1)
            phase[phase_key] = decoder_phase_columns[phase_key]
        else:
            phase[key] = value
    for key, value in encoder_phase_columns.items():
        phase.setdefault(key, value)
    for key, value in decoder_phase_columns.items():
        phase.setdefault(key, value)
    return phase



def build_one_shot_visualization_tensors(reference_images, sample_outputs, codes=None, original_images=None):
    # one-shot 阶段直接看单次光路 camera，同时保留 SLM 和 chip 诊断。
    camera_raw = sample_outputs["camera_plane"]
    batch_size = int(camera_raw.shape[0])
    encoder_layer_amplitude, encoder_layer_phase = _d2nn_field_layer_visualization_columns(
        sample_outputs,
        (("diffractive_encoder_layer", "encoder field after"),),
    )
    decoder_layer_amplitude, decoder_layer_phase = _d2nn_field_layer_visualization_columns(
        sample_outputs,
        (("diffractive_decoder_layer", "decoder field after"),),
    )
    encoder_mask_amplitude, encoder_mask_phase = _broadcast_d2nn_mask_visualization(
        sample_outputs,
        batch_size,
        (("diffractive_encoder_mask", "encoder mask amp = 1"),),
    )
    decoder_mask_amplitude, decoder_mask_phase = _broadcast_d2nn_mask_visualization(
        sample_outputs,
        batch_size,
        (("diffractive_decoder_mask", "decoder mask amp = 1"),),
    )
    encoder_amplitude_columns = _interleave_d2nn_mask_layer_columns(
        encoder_mask_amplitude,
        encoder_layer_amplitude,
        mask_prefix="encoder mask amp = 1",
        layer_prefix="encoder field after",
    )
    encoder_phase_columns = _interleave_d2nn_mask_layer_columns(
        encoder_mask_phase,
        encoder_layer_phase,
        mask_prefix="encoder mask phase",
        layer_prefix="encoder field after",
    )
    decoder_amplitude_columns = _interleave_d2nn_mask_layer_columns(
        decoder_mask_amplitude,
        decoder_layer_amplitude,
        mask_prefix="decoder mask amp = 1",
        layer_prefix="decoder field after",
    )
    decoder_phase_columns = _interleave_d2nn_mask_layer_columns(
        decoder_mask_phase,
        decoder_layer_phase,
        mask_prefix="decoder mask phase",
        layer_prefix="decoder field after",
    )
    amplitude = {
        "reference": reference_images,
        "camera": enhance_camera_for_display(camera_raw),
        "slm command": _normalize_for_visualization(sample_outputs["slm_command"]),
        "slm amplitude grid": _block_grid_preview(sample_outputs["slm_amplitude"]),
        "slm out amp": _normalize_for_visualization(sample_outputs["slm_amplitude"]),
        "encoder in amp": _normalize_for_visualization(sample_outputs["encoder_input_amplitude"]),
        "encoder grid in amp": _normalize_for_visualization(sample_outputs["encoder_input_grid_amplitude"]),
        **encoder_amplitude_columns,
        "encoder out amp": _robust_amplitude_display(sample_outputs["encoded_amplitude"]),
        "ols in amp": _robust_amplitude_display(sample_outputs["ols_input_amplitude"]),
        "ols out amp": _robust_amplitude_display(sample_outputs["ols_output_amplitude"]),
        "decoder in amp": _robust_amplitude_display(sample_outputs["decoder_input_amplitude"]),
        **decoder_amplitude_columns,
        "decoder out amp": _robust_amplitude_display(sample_outputs["decoder_output_amplitude"]),
        "chip amp": _robust_amplitude_display(sample_outputs["chip_amplitude"]),
        "chip I": _robust_amplitude_display(sample_outputs["chip_amplitude"].detach().float().clamp_min(0.0).pow(2)),
    }
    if original_images is not None:
        amplitude = {"digit": original_images, **amplitude}
    if codes is not None:
        amplitude = {"noise code": _code_for_visualization(codes), **amplitude}
    result_amplitude = dict(amplitude)
    result_phase = _build_phase_with_d2nn_columns(
        result_amplitude,
        sample_outputs,
        encoder_phase_columns,
        decoder_phase_columns,
        original_images=original_images,
    )
    return {
        VISUALIZATION_AMPLITUDE_KEY: result_amplitude,
        VISUALIZATION_PHASE_KEY: result_phase,
    }



def build_one_shot_eval_visualization_tensors(teacher_images, sample_outputs, abs_error, codes=None, original_images=None):
    base = build_one_shot_visualization_tensors(teacher_images, sample_outputs, codes=codes, original_images=original_images)
    amplitude = dict(base[VISUALIZATION_AMPLITUDE_KEY])
    teacher_display = amplitude.pop("reference")
    student_display = amplitude.pop("camera")
    result = {}
    if original_images is not None:
        result["digit"] = amplitude.pop("digit")
    if codes is not None:
        result["noise code"] = amplitude.pop("noise code")
    result = {
        **result,
        "teacher": teacher_display,
        **amplitude,
        "student": student_display,
        "abs error": abs_error,
    }
    base_phase = base[VISUALIZATION_PHASE_KEY]
    result_phase = {}
    for key in result:
        if key.startswith("encoder mask amp = 1 "):
            result_phase[key.replace("encoder mask amp = 1", "encoder mask phase", 1)] = base_phase[
                key.replace("encoder mask amp = 1", "encoder mask phase", 1)
            ]
        elif key.startswith("decoder mask amp = 1 "):
            result_phase[key.replace("decoder mask amp = 1", "decoder mask phase", 1)] = base_phase[
                key.replace("decoder mask amp = 1", "decoder mask phase", 1)
            ]
        elif key in base_phase:
            result_phase[key] = base_phase[key]
        else:
            result_phase[key] = _blank_tensor_like(next(iter(result.values())))
    return {
        VISUALIZATION_AMPLITUDE_KEY: result,
        VISUALIZATION_PHASE_KEY: result_phase,
    }



def _build_one_shot_shared_scale_body(sample_outputs, scale: float) -> dict[str, object]:
    return {
        label: _scale_amplitude_for_visualization(sample_outputs[key], scale)
        for key, label in ONE_SHOT_SHARED_SCALE_AMPLITUDE_SPECS
        if key in sample_outputs
    }



def _build_one_shot_early_raw_amplitude_body(sample_outputs) -> dict[str, object]:
    return {
        label: _normalize_for_visualization(sample_outputs[key])
        for key, label in ONE_SHOT_EARLY_RAW_AMPLITUDE_SPECS
        if key in sample_outputs
    }



def _build_one_shot_shared_scale_phase(reference_images, sample_outputs) -> dict[str, object]:
    phase = _phase_tensor_map(
        sample_outputs,
        (
            ("slm_phase", "slm out phase"),
            ("encoder_input_phase", "encoder in phase"),
            ("encoder_input_grid_phase", "encoder grid in phase"),
            ("encoded_phase", "encoder out phase"),
            ("ols_input_phase", "ols in phase"),
            ("ols_output_phase", "ols out phase"),
            ("decoder_input_phase", "decoder in phase"),
            ("decoder_output_phase", "decoder out phase"),
            ("chip_phase", "chip phase"),
        ),
    )
    return {"reference": reference_images, **phase}



def _phase_column_stats(sample_outputs) -> dict[str, object]:
    phase_specs = (
        ("slm_phase", "slm out phase"),
        ("encoder_input_phase", "encoder in phase"),
        ("encoder_input_grid_phase", "encoder grid in phase"),
        ("encoded_phase", "encoder out phase"),
        ("ols_input_phase", "ols in phase"),
        ("ols_output_phase", "ols out phase"),
        ("decoder_input_phase", "decoder in phase"),
        ("decoder_output_phase", "decoder out phase"),
        ("chip_phase", "chip phase"),
    )
    return {
        label: _tensor_stats(sample_outputs[key])
        for key, label in phase_specs
        if key in sample_outputs
    }



def build_one_shot_shared_scale_diagnostics(sample_outputs, scale: float) -> dict[str, object]:
    columns = [
        label
        for key, label in ONE_SHOT_SHARED_SCALE_AMPLITUDE_SPECS
        if key in sample_outputs
    ]
    slm_command = sample_outputs["slm_command"].detach().float().cpu().flatten(start_dim=1)
    per_sample_p01 = slm_command.quantile(0.01, dim=1)
    per_sample_p99 = slm_command.quantile(0.99, dim=1)
    return {
        "view": "shared_scale_full_chain",
        "amplitude_scale_quantile": ONE_SHOT_SHARED_SCALE_AMPLITUDE_QUANTILE,
        "amplitude_scale": float(scale),
        "amplitude_scale_columns": columns,
        "early_raw_amplitude_columns": [
            "slm in cmd",
            *[
                label
                for key, label in ONE_SHOT_EARLY_RAW_AMPLITUDE_SPECS
                if key in sample_outputs
            ],
        ],
        "slm_command_stats": {
            **_tensor_stats(sample_outputs["slm_command"]),
            "p01_mean": float(per_sample_p01.mean().item()),
            "p99_mean": float(per_sample_p99.mean().item()),
            "p99_minus_p01_mean": float((per_sample_p99 - per_sample_p01).mean().item()),
        },
        "phase_stats": _phase_column_stats(sample_outputs),
    }



def build_one_shot_shared_scale_visualization_tensors(reference_images, sample_outputs, codes=None, original_images=None):
    scale = _shared_amplitude_scale(
        sample_outputs[key]
        for key, _ in ONE_SHOT_SHARED_SCALE_AMPLITUDE_SPECS
        if key in sample_outputs
    )
    batch_size = int(reference_images.shape[0])
    encoder_layer_amplitude, encoder_layer_phase = _d2nn_field_layer_visualization_columns(
        sample_outputs,
        (("diffractive_encoder_layer", "encoder field after"),),
    )
    decoder_layer_amplitude, decoder_layer_phase = _d2nn_field_layer_visualization_columns(
        sample_outputs,
        (("diffractive_decoder_layer", "decoder field after"),),
    )
    encoder_mask_amplitude, encoder_mask_phase = _broadcast_d2nn_mask_visualization(
        sample_outputs,
        batch_size,
        (("diffractive_encoder_mask", "encoder mask amp = 1"),),
    )
    decoder_mask_amplitude, decoder_mask_phase = _broadcast_d2nn_mask_visualization(
        sample_outputs,
        batch_size,
        (("diffractive_decoder_mask", "decoder mask amp = 1"),),
    )
    encoder_amplitude_columns = _interleave_d2nn_mask_layer_columns(
        encoder_mask_amplitude,
        encoder_layer_amplitude,
        mask_prefix="encoder mask amp = 1",
        layer_prefix="encoder field after",
    )
    encoder_phase_columns = _interleave_d2nn_mask_layer_columns(
        encoder_mask_phase,
        encoder_layer_phase,
        mask_prefix="encoder mask phase",
        layer_prefix="encoder field after",
    )
    decoder_amplitude_columns = _interleave_d2nn_mask_layer_columns(
        decoder_mask_amplitude,
        decoder_layer_amplitude,
        mask_prefix="decoder mask amp = 1",
        layer_prefix="decoder field after",
    )
    decoder_phase_columns = _interleave_d2nn_mask_layer_columns(
        decoder_mask_phase,
        decoder_layer_phase,
        mask_prefix="decoder mask phase",
        layer_prefix="decoder field after",
    )
    amplitude = {
        "reference": reference_images,
        "slm in cmd": _raw_amplitude_for_visualization(sample_outputs["slm_command"]),
        "slm in cmd contrast": _slm_contrast_for_visualization(sample_outputs["slm_command"]),
        "slm amplitude grid": _block_grid_preview(sample_outputs["slm_amplitude"]),
        "slm out amp": _normalize_for_visualization(sample_outputs["slm_amplitude"]),
        "encoder in amp": _normalize_for_visualization(sample_outputs["encoder_input_amplitude"]),
        "encoder grid in amp": _normalize_for_visualization(sample_outputs["encoder_input_grid_amplitude"]),
        **encoder_amplitude_columns,
        "encoder out amp": _scale_amplitude_for_visualization(sample_outputs["encoded_amplitude"], scale),
        "ols in amp": _scale_amplitude_for_visualization(sample_outputs["ols_input_amplitude"], scale),
        "ols out amp": _scale_amplitude_for_visualization(sample_outputs["ols_output_amplitude"], scale),
        "decoder in amp": _scale_amplitude_for_visualization(sample_outputs["decoder_input_amplitude"], scale),
        **decoder_amplitude_columns,
        "decoder out amp": _scale_amplitude_for_visualization(sample_outputs["decoder_output_amplitude"], scale),
        "chip amp": _scale_amplitude_for_visualization(sample_outputs["chip_amplitude"], scale),
        "chip I": _intensity_for_visualization(sample_outputs["chip_amplitude"]),
        "camera": enhance_camera_for_display(sample_outputs["camera_plane"]),
    }
    if original_images is not None:
        amplitude = {"digit": original_images, **amplitude}
    if codes is not None:
        amplitude = {"noise code": _code_for_visualization(codes), **amplitude}
    phase = _build_phase_with_d2nn_columns(
        amplitude,
        sample_outputs,
        encoder_phase_columns,
        decoder_phase_columns,
        original_images=original_images,
    )
    return {
        VISUALIZATION_AMPLITUDE_KEY: amplitude,
        VISUALIZATION_PHASE_KEY: phase,
        "diagnostics": build_one_shot_shared_scale_diagnostics(sample_outputs, scale),
    }



def build_one_shot_eval_shared_scale_visualization_tensors(teacher_images, sample_outputs, abs_error, codes=None, original_images=None):
    base = build_one_shot_shared_scale_visualization_tensors(teacher_images, sample_outputs, codes=codes, original_images=original_images)
    amplitude = dict(base[VISUALIZATION_AMPLITUDE_KEY])
    teacher_display = amplitude.pop("reference")
    student_display = amplitude.pop("camera")
    result = {}
    if original_images is not None:
        result["digit"] = amplitude.pop("digit")
    if codes is not None:
        result["noise code"] = amplitude.pop("noise code")
    result = {
        **result,
        "teacher": teacher_display,
        **amplitude,
        "student": student_display,
        "abs error": abs_error,
    }
    aligned_phase = _build_phase_aligned_to_amplitude(result, sample_outputs, original_images=original_images)
    base_phase = base[VISUALIZATION_PHASE_KEY]
    result_phase: dict[str, object] = {}
    for key in result:
        if key.startswith("encoder mask amp = 1 "):
            phase_key = key.replace("encoder mask amp = 1", "encoder mask phase", 1)
            if phase_key in base_phase:
                result_phase[phase_key] = base_phase[phase_key]
        elif key.startswith("decoder mask amp = 1 "):
            phase_key = key.replace("decoder mask amp = 1", "decoder mask phase", 1)
            if phase_key in base_phase:
                result_phase[phase_key] = base_phase[phase_key]
        elif key in base_phase:
            result_phase[key] = base_phase[key]
        elif key in aligned_phase:
            result_phase[key] = aligned_phase[key]
    return {
        VISUALIZATION_AMPLITUDE_KEY: result,
        VISUALIZATION_PHASE_KEY: result_phase,
        "diagnostics": base["diagnostics"],
    }



def build_one_shot_display_diagnostic_tensors(sample_outputs):
    return {
        "slm_phase_display": _phase_for_visualization(sample_outputs["slm_phase"]),
        "encoder_input_amplitude_display": _amplitude_for_visualization(sample_outputs["encoder_input_amplitude"]),
        "encoder_input_phase_display": _phase_for_visualization(sample_outputs["encoder_input_phase"]),
        "encoder_input_grid_amplitude_display": _amplitude_for_visualization(
            sample_outputs["encoder_input_grid_amplitude"]
        ),
        "encoder_input_grid_phase_display": _phase_for_visualization(sample_outputs["encoder_input_grid_phase"]),
        "encoded_amplitude_display": _robust_amplitude_display(sample_outputs["encoded_amplitude"]),
        "encoded_phase_display": _phase_for_visualization(sample_outputs["encoded_phase"]),
        "ols_input_amplitude_display": _robust_amplitude_display(sample_outputs["ols_input_amplitude"]),
        "ols_input_phase_display": _phase_for_visualization(sample_outputs["ols_input_phase"]),
        "ols_output_amplitude_display": _robust_amplitude_display(sample_outputs["ols_output_amplitude"]),
        "ols_output_phase_display": _phase_for_visualization(sample_outputs["ols_output_phase"]),
        "ols_amplitude_display": _robust_amplitude_display(sample_outputs["ols_amplitude"]),
        "ols_phase_display": _phase_for_visualization(sample_outputs["ols_phase"]),
        "decoder_input_amplitude_display": _robust_amplitude_display(sample_outputs["decoder_input_amplitude"]),
        "decoder_input_phase_display": _phase_for_visualization(sample_outputs["decoder_input_phase"]),
        "decoder_output_amplitude_display": _robust_amplitude_display(sample_outputs["decoder_output_amplitude"]),
        "decoder_output_phase_display": _phase_for_visualization(sample_outputs["decoder_output_phase"]),
        "chip_field_amplitude_display": _robust_amplitude_display(sample_outputs["chip_amplitude"]),
        "chip_intensity_display": _robust_amplitude_display(sample_outputs["chip_amplitude"].detach().float().clamp_min(0.0).pow(2)),
        "chip_phase_display": _phase_for_visualization(sample_outputs["chip_phase"]),
    }



def _spatial_pearson_summary(left, right):
    left_flat = left.detach().float().cpu().flatten(start_dim=1)
    right_flat = right.detach().float().cpu().flatten(start_dim=1)
    left_centered = left_flat - left_flat.mean(dim=1, keepdim=True)
    right_centered = right_flat - right_flat.mean(dim=1, keepdim=True)
    denominator = (
        left_centered.square().sum(dim=1).sqrt()
        * right_centered.square().sum(dim=1).sqrt()
    ).clamp_min(1e-8)
    values = (left_centered * right_centered).sum(dim=1) / denominator
    return {
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }



def _top_mask_iou_summary(left, right, quantile: float = 0.75):
    left_flat = left.detach().float().cpu().flatten(start_dim=1)
    right_flat = right.detach().float().cpu().flatten(start_dim=1)
    left_threshold = left_flat.quantile(quantile, dim=1, keepdim=True)
    right_threshold = right_flat.quantile(quantile, dim=1, keepdim=True)
    left_mask = left_flat >= left_threshold
    right_mask = right_flat >= right_threshold
    intersection = (left_mask & right_mask).sum(dim=1).float()
    union = (left_mask | right_mask).sum(dim=1).float().clamp_min(1.0)
    values = intersection / union
    return {
        "quantile": quantile,
        "mean": float(values.mean().item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }



def build_one_shot_visual_diagnostics(sample_outputs, display_diagnostics=None):
    tensors = dict(sample_outputs)
    if display_diagnostics:
        tensors.update(display_diagnostics)

    alias_specs = (
        ("encoded_to_ols_input_amplitude", "encoded_amplitude", "ols_input_amplitude"),
        ("encoded_to_ols_input_phase", "encoded_phase", "ols_input_phase"),
        ("ols_output_to_decoder_input_amplitude", "ols_output_amplitude", "decoder_input_amplitude"),
        ("ols_output_to_decoder_input_phase", "ols_output_phase", "decoder_input_phase"),
        (
            "encoded_to_ols_input_amplitude_display",
            "encoded_amplitude_display",
            "ols_input_amplitude_display",
        ),
        ("encoded_to_ols_input_phase_display", "encoded_phase_display", "ols_input_phase_display"),
        (
            "ols_output_to_decoder_input_amplitude_display",
            "ols_output_amplitude_display",
            "decoder_input_amplitude_display",
        ),
        (
            "ols_output_to_decoder_input_phase_display",
            "ols_output_phase_display",
            "decoder_input_phase_display",
        ),
    )
    alias_diffs = {
        name: _tensor_diff_summary(tensors[left_key], tensors[right_key])
        for name, left_key, right_key in alias_specs
        if left_key in tensors and right_key in tensors
    }

    similarity_specs = (
        ("encoder_input", "encoder_input_amplitude_display", "encoder_input_phase_display"),
        ("encoder_input_grid", "encoder_input_grid_amplitude_display", "encoder_input_grid_phase_display"),
        ("encoder_out_ols_in", "encoded_amplitude_display", "encoded_phase_display"),
        ("ols_out_decoder_in", "ols_output_amplitude_display", "ols_output_phase_display"),
        ("decoder_out", "decoder_output_amplitude_display", "decoder_output_phase_display"),
        ("chip", "chip_field_amplitude_display", "chip_phase_display"),
    )
    spatial_similarity = {
        name: {
            "pearson": _spatial_pearson_summary(tensors[amplitude_key], tensors[phase_key]),
            "top25_iou": _top_mask_iou_summary(tensors[amplitude_key], tensors[phase_key]),
        }
        for name, amplitude_key, phase_key in similarity_specs
        if amplitude_key in tensors and phase_key in tensors
    }

    stat_keys = (
        "slm_command",
        "slm_amplitude",
        "encoder_input_amplitude",
        "encoder_input_grid_amplitude",
        "encoded_amplitude",
        "ols_output_amplitude",
        "decoder_output_amplitude",
        "chip_amplitude",
        "camera_plane",
        "encoded_phase",
        "ols_output_phase",
        "decoder_output_phase",
        "chip_phase",
    )
    tensor_stats = {
        key: _tensor_stats(tensors[key])
        for key in stat_keys
        if key in tensors
    }
    return {
        "view": "compact_full_chain",
        "alias_diffs": alias_diffs,
        "amplitude_phase_similarity": spatial_similarity,
        "tensor_stats": tensor_stats,
    }
