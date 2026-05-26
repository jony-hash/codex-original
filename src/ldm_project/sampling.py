from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .train_utils import validate_class_label_values

if TYPE_CHECKING:
    pass


DEFAULT_DENOISING_TRACE_TIMESTEPS = (750, 500, 250, 100, 50, 0)
DEFAULT_NEAREST_REFERENCE_LIMIT = 256
DEFAULT_AMPLITUDE_DISPLAY_QUANTILE = 0.995
DEFAULT_VISUAL_PREVIEW_PER_CLASS = 2
VISUALIZATION_AMPLITUDE_KEY = "amplitude"
VISUALIZATION_PHASE_KEY = "phase"
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
    "slm in cmd": None,
    "slm in cmd contrast": None,
    "slm out amp": "slm_phase",
    "encoder in amp": "encoder_input_phase",
    "encoder grid in amp": "encoder_input_grid_phase",
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


def build_visualization_labels(torch_module, num_samples: int, num_classes: int, device):
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be positive, got {num_classes}")
    return torch_module.arange(num_samples, device=device, dtype=torch_module.long) % num_classes


def build_post_train_sample_labels(
    torch_module,
    num_classes: int,
    device,
    sample_class_labels: list[int] | tuple[int, ...] | None = None,
    sample_repeats_per_class: int = 1,
):
    if sample_repeats_per_class <= 0:
        raise ValueError(f"sample_repeats_per_class must be positive, got {sample_repeats_per_class}")

    requested_labels = tuple(sample_class_labels or ())
    if not requested_labels:
        labels = tuple(range(num_classes))
    else:
        labels = validate_class_label_values(requested_labels, num_classes, "sample_class_labels")
    # 多次采样时按“0-9 一组”循环排列，最终网格更容易逐组横向比较。
    repeated_labels = [label for label in labels for _ in range(sample_repeats_per_class)]
    return torch_module.tensor(repeated_labels, device=device, dtype=torch_module.long)


def _normalize_for_visualization(tensor):
    min_value = tensor.amin(dim=(-2, -1), keepdim=True)
    max_value = tensor.amax(dim=(-2, -1), keepdim=True)
    return (tensor - min_value) / (max_value - min_value).clamp_min(1e-6)


def _amplitude_for_visualization(tensor, upper_quantile: float = DEFAULT_AMPLITUDE_DISPLAY_QUANTILE):
    if upper_quantile <= 0.0 or upper_quantile > 1.0:
        raise ValueError(f"amplitude display quantile must be within (0, 1], got {upper_quantile}")
    raw = tensor.detach().float().clamp_min(0.0)
    scale = raw.flatten(start_dim=1).quantile(upper_quantile, dim=1)
    scale = scale.reshape(raw.shape[0], *([1] * (raw.ndim - 1))).clamp_min(1e-6)
    return (raw / scale).clamp(0.0, 1.0)


def _robust_amplitude_display(tensor, lower_q: float = 0.02, upper_q: float = 0.98):
    raw = tensor.detach().float().clamp_min(0.0)
    flat = raw.flatten(start_dim=1)
    lo = flat.quantile(lower_q, dim=1)
    hi = flat.quantile(upper_q, dim=1)
    shape = (raw.shape[0], *([1] * (raw.ndim - 1)))
    lo = lo.reshape(shape)
    hi = hi.reshape(shape)
    return ((raw - lo) / (hi - lo).clamp_min(1e-6)).clamp(0.0, 1.0)


def _intensity_for_visualization(amplitude, upper_quantile: float = DEFAULT_AMPLITUDE_DISPLAY_QUANTILE):
    intensity = amplitude.detach().float().clamp_min(0.0).pow(2)
    return _amplitude_for_visualization(intensity, upper_quantile=upper_quantile)


def _load_original_mnist_digits(labels: torch.Tensor, image_size: int, root: str = "data") -> torch.Tensor:
    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise RuntimeError("torchvision is required to load original MNIST digits.") from exc

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    mnist = datasets.MNIST(root=root, train=True, download=False, transform=transform)
    images = []
    for label in labels.tolist():
        found = None
        for img, lbl in mnist:
            if lbl == int(label):
                found = img
                break
        if found is None:
            found = torch.zeros(1, image_size, image_size)
        images.append(found)
    return torch.stack(images)


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


def _raw_amplitude_for_visualization(tensor):
    return tensor.detach().float().clamp(0.0, 1.0)


def _slm_contrast_for_visualization(tensor):
    return _normalize_for_visualization(tensor.detach().float())


def _phase_for_visualization(tensor):
    return tensor.detach().float().remainder(2.0 * math.pi) / (2.0 * math.pi)


def _phase_tensor_map(source: dict, specs: tuple[tuple[str, str], ...]) -> dict[str, object]:
    return {
        label: _phase_for_visualization(source[key])
        for key, label in specs
        if key in source
    }


def _blank_tensor_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.full_like(reference.detach().float(), float("nan"))


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


def _code_for_visualization(codes):
    code = codes.detach().float().unsqueeze(1).unsqueeze(2)
    return _normalize_for_visualization(code)


def build_code_visualization_tensor(codes):
    return _code_for_visualization(codes)


def enhance_camera_for_display(tensor, lower_quantile: float = 0.01, upper_quantile: float = 0.995, gamma: float = 0.75):
    if lower_quantile < 0.0 or upper_quantile > 1.0 or lower_quantile >= upper_quantile:
        raise ValueError("display quantiles must satisfy 0 <= lower < upper <= 1")
    if gamma <= 0.0:
        raise ValueError(f"display gamma must be positive, got {gamma}")
    raw = tensor.detach().float()
    flat = raw.flatten(start_dim=-2)
    low = flat.quantile(lower_quantile, dim=-1, keepdim=True).unsqueeze(-1)
    high = flat.quantile(upper_quantile, dim=-1, keepdim=True).unsqueeze(-1)
    scaled = (raw - low) / (high - low).clamp_min(1e-6)
    return scaled.clamp(0.0, 1.0).pow(gamma)


def format_output_title(relative_path: str, label: str) -> str:
    return f"{relative_path} | {label}"


def build_first_stage_visualization_tensors(input_batch, outputs):
    # 物理优先先看相机平面是否承载数字，再用 decoder image 诊断 latent 语义。
    return {
        VISUALIZATION_AMPLITUDE_KEY: {
            "input": input_batch,
            "camera": outputs["camera_plane"],
            "image": outputs["image_recon"],
            "pf2": outputs["pf2_intensity"],
            "chip amp": _amplitude_for_visualization(outputs["chip_amplitude"]),
            "probe": outputs["probe_plane"],
        },
        VISUALIZATION_PHASE_KEY: _phase_tensor_map(
            outputs,
            (
                ("slm_phase", "slm phase"),
                ("chip_phase", "chip phase"),
            ),
        ),
    }


def build_diffusion_visualization_tensors(reference_images, sample_outputs):
    # diffusion 阶段先看生成 latent 经物理后级读出的 camera，再看 decoder image 作为辅助。
    return {
        VISUALIZATION_AMPLITUDE_KEY: {
            "reference": reference_images,
            "camera": sample_outputs["camera_plane"],
            "image": sample_outputs["image_recon"],
            "chip amp": _amplitude_for_visualization(sample_outputs["chip_amplitude"]),
            "probe": sample_outputs["probe_plane"],
        },
        VISUALIZATION_PHASE_KEY: _phase_tensor_map(
            sample_outputs,
            (
                ("slm_phase", "slm phase"),
                ("chip_phase", "chip phase"),
            ),
        ),
    }


def build_one_shot_visualization_tensors(reference_images, sample_outputs, codes=None, original_images=None):
    # one-shot 阶段直接看单次光路 camera，同时保留 SLM 和 chip 诊断。
    camera_raw = sample_outputs["camera_plane"]
    amplitude = {
        "reference": reference_images,
        "camera": enhance_camera_for_display(camera_raw),
        "slm command": _normalize_for_visualization(sample_outputs["slm_command"]),
        "slm out amp": _normalize_for_visualization(sample_outputs["slm_amplitude"]),
        "encoder in amp": _normalize_for_visualization(sample_outputs["encoder_input_amplitude"]),
        "encoder grid in amp": _normalize_for_visualization(sample_outputs["encoder_input_grid_amplitude"]),
        "encoder out amp": _robust_amplitude_display(sample_outputs["encoded_amplitude"]),
        "ols in amp": _robust_amplitude_display(sample_outputs["ols_input_amplitude"]),
        "ols out amp": _robust_amplitude_display(sample_outputs["ols_output_amplitude"]),
        "decoder in amp": _robust_amplitude_display(sample_outputs["decoder_input_amplitude"]),
        "decoder out amp": _robust_amplitude_display(sample_outputs["decoder_output_amplitude"]),
        "chip amp": _robust_amplitude_display(sample_outputs["chip_amplitude"]),
        "chip I": _robust_amplitude_display(sample_outputs["chip_amplitude"].detach().float().clamp_min(0.0).pow(2)),
    }
    if original_images is not None:
        amplitude = {"digit": original_images, **amplitude}
    if codes is not None:
        amplitude = {"noise code": _code_for_visualization(codes), **amplitude}
    result_amplitude = dict(amplitude)
    result_phase = _build_phase_aligned_to_amplitude(result_amplitude, sample_outputs, original_images=original_images)
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
    result_phase = _build_phase_aligned_to_amplitude(result, sample_outputs, original_images=original_images)
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
    amplitude = {
        "reference": reference_images,
        "slm in cmd": _raw_amplitude_for_visualization(sample_outputs["slm_command"]),
        "slm in cmd contrast": _slm_contrast_for_visualization(sample_outputs["slm_command"]),
        **_build_one_shot_early_raw_amplitude_body(sample_outputs),
        **_build_one_shot_shared_scale_body(sample_outputs, scale),
        "chip I": _intensity_for_visualization(sample_outputs["chip_amplitude"]),
        "camera": enhance_camera_for_display(sample_outputs["camera_plane"]),
    }
    if original_images is not None:
        amplitude = {"digit": original_images, **amplitude}
    if codes is not None:
        amplitude = {"noise code": _code_for_visualization(codes), **amplitude}
    phase = _build_phase_aligned_to_amplitude(amplitude, sample_outputs, original_images=original_images)
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
    result_phase = _build_phase_aligned_to_amplitude(result, sample_outputs, original_images=original_images)
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


def _tensor_stats(tensor):
    value = tensor.detach().float().cpu()
    return {
        "shape": list(value.shape),
        "min": float(value.min().item()),
        "max": float(value.max().item()),
        "mean": float(value.mean().item()),
        "std": float(value.std().item()),
    }


def _tensor_diff_summary(left, right):
    delta = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
    return {
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
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


def flatten_visualization_splits(visualization_splits: dict[str, dict[str, object]]) -> dict[str, object]:
    flat: dict[str, object] = {}
    for prefix, tensor_map in visualization_splits.items():
        for key, value in tensor_map.items():
            flat[f"{prefix}.{key}"] = value
    return flat


def select_balanced_visual_preview_indices(
    labels,
    max_per_class: int = DEFAULT_VISUAL_PREVIEW_PER_CLASS,
    class_labels: tuple[int, ...] | list[int] | None = None,
) -> tuple[list[int], dict[str, object]]:
    if max_per_class <= 0:
        raise ValueError(f"visual preview max_per_class must be positive, got {max_per_class}")
    label_values = [int(label) for label in labels.detach().cpu().tolist()]
    requested_labels = (
        tuple(int(label) for label in class_labels)
        if class_labels is not None
        else tuple(sorted(set(label_values)))
    )
    preview_indices: list[int] = []
    preview_labels: list[int] = []
    per_class: dict[str, dict[str, object]] = {}
    for label in requested_labels:
        label_indices = [index for index, item in enumerate(label_values) if item == label]
        chosen = label_indices[:max_per_class]
        preview_indices.extend(chosen)
        preview_labels.extend([label] * len(chosen))
        per_class[str(label)] = {
            "available": len(label_indices),
            "selected": len(chosen),
            "preview_indices": chosen,
        }
    return preview_indices, {
        "strategy": "balanced_per_class_preview",
        "class_labels": list(requested_labels),
        "max_per_class": max_per_class,
        "visual_preview_count": len(preview_indices),
        "visual_preview_indices": preview_indices,
        "visual_preview_labels": preview_labels,
        "per_class": per_class,
    }


def labels_for_visual_rows(labels) -> list[str]:
    return [str(int(label)) for label in labels.detach().cpu().tolist()]


def save_visualization_grids(
    run_manager,
    visualization_splits: dict[str, dict[str, object]],
    destination: str | Path,
    *,
    title_label: str | None = None,
    max_items: int = 4,
    row_labels: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Path]:
    base_path = Path(destination)
    saved_paths: dict[str, Path] = {}
    for split_name in (VISUALIZATION_AMPLITUDE_KEY, VISUALIZATION_PHASE_KEY):
        tensor_map = visualization_splits.get(split_name, {})
        if not tensor_map:
            continue
        path = base_path.with_name(f"{base_path.stem}_{split_name}{base_path.suffix}")
        title = ""
        if title_label:
            try:
                relative = str(path.relative_to(run_manager.run_dir))
            except ValueError:
                relative = str(path)
            title = format_output_title(relative, f"{title_label} {split_name}")
        saved_paths[split_name] = run_manager.save_tensor_grid(
            tensor_map,
            path,
            title=title,
            max_items=max_items,
            row_labels=row_labels,
            cmap="twilight" if split_name == VISUALIZATION_PHASE_KEY else "gray",
            phase_colorbar=split_name == VISUALIZATION_PHASE_KEY,
        )
    return saved_paths


def build_denoising_trace_visualization_tensors(reference_images, camera_trace: dict[str, "torch.Tensor"]):
    # 每列展示同一条采样轨迹在指定去噪阶段的相机读出，方便直接看噪声如何变成数字。
    return {
        "reference": reference_images,
        **{label: tensor for label, tensor in camera_trace.items()},
    }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _compute_reference_similarity_metrics(
    prediction,
    references,
    foreground_threshold: float,
) -> dict[str, float | int]:
    prediction = prediction.detach().float().clamp(0.0, 1.0).cpu()
    references = references.detach().float().clamp(0.0, 1.0).cpu()
    if references.ndim == prediction.ndim:
        references = references.unsqueeze(0)
    if references.shape[0] <= 0:
        raise ValueError("nearest reference bank must contain at least one image per requested label.")

    prediction_mask = prediction > foreground_threshold
    reference_masks = references > foreground_threshold
    intersections = (reference_masks & prediction_mask.unsqueeze(0)).float().flatten(1).sum(dim=1)
    denominator = (
        prediction_mask.float().sum()
        + reference_masks.float().flatten(1).sum(dim=1)
    ).clamp_min(1.0)
    dice_scores = 2.0 * intersections / denominator
    best_dice, best_dice_index = dice_scores.max(dim=0)

    prediction_flat = prediction.flatten()
    references_flat = references.flatten(1)
    prediction_centered = prediction_flat - prediction_flat.mean()
    references_centered = references_flat - references_flat.mean(dim=1, keepdim=True)
    corr_denominator = (
        prediction_centered.pow(2).mean().sqrt()
        * references_centered.pow(2).mean(dim=1).sqrt()
    )
    corr_scores = (
        (references_centered * prediction_centered.unsqueeze(0)).mean(dim=1)
        / corr_denominator.clamp_min(1.0e-8)
    )
    best_corr, best_corr_index = corr_scores.max(dim=0)
    return {
        "dice": float(best_dice.item()),
        "corr": float(best_corr.item()),
        "dice_reference_index": int(best_dice_index.item()),
        "corr_reference_index": int(best_corr_index.item()),
    }


def _summarize_nearest_neighbor_metrics(
    generated,
    labels: list[int],
    reference_image_bank: dict[int, "torch.Tensor"],
    reference_bank_source: str | None,
    foreground_threshold: float,
) -> dict[str, object]:
    per_sample: list[dict[str, float | int]] = []
    per_class_values: dict[str, dict[str, list[float] | int]] = {}
    reference_counts = {
        str(label): int(images.shape[0])
        for label, images in sorted(reference_image_bank.items())
    }
    for index, label in enumerate(labels):
        if label not in reference_image_bank:
            raise ValueError(f"Missing nearest-neighbor reference images for class label: {label}")
        nearest = _compute_reference_similarity_metrics(
            generated[index],
            reference_image_bank[label],
            foreground_threshold,
        )
        sample_record = {
            "index": index,
            "label": label,
            **nearest,
        }
        per_sample.append(sample_record)
        class_record = per_class_values.setdefault(str(label), {"count": 0, "dice": [], "corr": []})
        class_record["count"] = int(class_record["count"]) + 1
        class_record["dice"].append(float(sample_record["dice"]))
        class_record["corr"].append(float(sample_record["corr"]))

    per_class = {}
    for label, values in sorted(per_class_values.items(), key=lambda item: int(item[0])):
        per_class[label] = {
            "count": int(values["count"]),
            "reference_count": reference_counts[label],
            "dice": _safe_mean(values["dice"]),
            "corr": _safe_mean(values["corr"]),
        }

    return {
        "reference_source": reference_bank_source,
        "reference_counts": reference_counts,
        "overall": {
            "count": len(per_sample),
            "dice": _safe_mean([float(item["dice"]) for item in per_sample]),
            "corr": _safe_mean([float(item["corr"]) for item in per_sample]),
        },
        "per_class": per_class,
        "per_sample": per_sample,
    }


def compute_sample_quality_metrics(
    generated_camera,
    reference_images,
    class_labels,
    foreground_threshold: float = 0.2,
    reference_image_bank: dict[int, "torch.Tensor"] | None = None,
    reference_bank_source: str | None = None,
) -> dict[str, object]:
    generated = generated_camera.detach().float().clamp(0.0, 1.0).cpu()
    reference = reference_images.detach().float().clamp(0.0, 1.0).cpu()
    labels = [int(label) for label in class_labels.detach().cpu().tolist()]
    if generated.shape[0] != reference.shape[0] or generated.shape[0] != len(labels):
        raise ValueError("generated_camera, reference_images, and class_labels must have matching batch size.")

    per_sample: list[dict[str, float | int]] = []
    per_class_values: dict[str, dict[str, list[float] | int]] = {}
    for index, label in enumerate(labels):
        prediction = generated[index]
        target = reference[index]
        prediction_mask = prediction > foreground_threshold
        target_mask = target > foreground_threshold
        intersection = (prediction_mask & target_mask).float().sum()
        denominator = prediction_mask.float().sum() + target_mask.float().sum()
        dice = (2.0 * intersection / denominator.clamp_min(1.0)).item()

        prediction_flat = prediction.flatten()
        target_flat = target.flatten()
        prediction_centered = prediction_flat - prediction_flat.mean()
        target_centered = target_flat - target_flat.mean()
        corr_denominator = prediction_centered.pow(2).mean().sqrt() * target_centered.pow(2).mean().sqrt()
        corr = ((prediction_centered * target_centered).mean() / corr_denominator.clamp_min(1.0e-8)).item()
        hole_proxy = (
            (target_mask & (prediction <= foreground_threshold)).float().sum()
            / target_mask.float().sum().clamp_min(1.0)
        ).item()

        sample_record = {
            "index": index,
            "label": label,
            "dice": float(dice),
            "corr": float(corr),
            "hole_proxy": float(hole_proxy),
        }
        per_sample.append(sample_record)
        class_record = per_class_values.setdefault(str(label), {"count": 0, "dice": [], "corr": [], "hole_proxy": []})
        class_record["count"] = int(class_record["count"]) + 1
        for metric_name in ("dice", "corr", "hole_proxy"):
            class_record[metric_name].append(float(sample_record[metric_name]))

    per_class = {}
    for label, values in sorted(per_class_values.items(), key=lambda item: int(item[0])):
        per_class[label] = {
            "count": int(values["count"]),
            "dice": _safe_mean(values["dice"]),
            "corr": _safe_mean(values["corr"]),
            "hole_proxy": _safe_mean(values["hole_proxy"]),
        }

    metrics = {
        "foreground_threshold": foreground_threshold,
        "overall": {
            "count": len(per_sample),
            "dice": _safe_mean([float(item["dice"]) for item in per_sample]),
            "corr": _safe_mean([float(item["corr"]) for item in per_sample]),
            "hole_proxy": _safe_mean([float(item["hole_proxy"]) for item in per_sample]),
        },
        "per_class": per_class,
        "per_sample": per_sample,
    }
    if reference_image_bank is not None:
        metrics["nearest_neighbor"] = _summarize_nearest_neighbor_metrics(
            generated,
            labels,
            reference_image_bank,
            reference_bank_source,
            foreground_threshold,
        )
    return metrics


def _resolve_label_value(label) -> int:
    if hasattr(label, "item"):
        return int(label.item())
    return int(label)


def _collect_reference_images_from_dataset(
    torch_module,
    dataset,
    requested_labels: list[int],
) -> dict[int, "torch.Tensor"]:
    requested_set = set(requested_labels)
    found_images: dict[int, "torch.Tensor"] = {}
    for index in range(len(dataset)):
        image, label = dataset[index]
        label_value = _resolve_label_value(label)
        if label_value not in requested_set or label_value in found_images:
            continue
        found_images[label_value] = torch_module.as_tensor(image).detach().float().cpu()
        if len(found_images) == len(requested_set):
            break
    return found_images


def _collect_reference_image_bank_from_dataset(
    torch_module,
    dataset,
    requested_labels: list[int],
    max_items_per_label: int = DEFAULT_NEAREST_REFERENCE_LIMIT,
) -> dict[int, "torch.Tensor"]:
    requested_set = set(requested_labels)
    found_images: dict[int, list["torch.Tensor"]] = {label: [] for label in requested_set}
    for index in range(len(dataset)):
        image, label = dataset[index]
        label_value = _resolve_label_value(label)
        if label_value not in requested_set:
            continue
        label_images = found_images[label_value]
        if len(label_images) >= max_items_per_label:
            continue
        label_images.append(torch_module.as_tensor(image).detach().float().cpu())
        if all(len(images) >= max_items_per_label for images in found_images.values()):
            break
    return {
        label: torch_module.stack(images)
        for label, images in found_images.items()
        if images
    }


def build_reference_image_bank_for_labels(
    torch_module,
    class_labels,
    validation_bundle,
    train_bundle,
    max_items_per_label: int = DEFAULT_NEAREST_REFERENCE_LIMIT,
) -> tuple[dict[int, "torch.Tensor"], str]:
    label_values = sorted({int(label) for label in class_labels.detach().cpu().tolist()})
    validation_bank: dict[int, "torch.Tensor"] = {}
    if validation_bundle is not None:
        validation_bank = _collect_reference_image_bank_from_dataset(
            torch_module,
            validation_bundle.dataset,
            label_values,
            max_items_per_label=max_items_per_label,
        )

    found_bank = dict(validation_bank)
    missing_labels = sorted(label for label in label_values if label not in found_bank)
    if missing_labels:
        found_bank.update(
            _collect_reference_image_bank_from_dataset(
                torch_module,
                train_bundle.dataset,
                missing_labels,
                max_items_per_label=max_items_per_label,
            )
        )

    unresolved = [label for label in label_values if label not in found_bank]
    if unresolved:
        raise ValueError(f"Unable to find nearest-neighbor reference images for class labels: {unresolved}")
    if validation_bundle is None:
        reference_bank_source = "train_only"
    elif all(label in validation_bank for label in label_values):
        reference_bank_source = "validation_only"
    else:
        reference_bank_source = "validation_then_train_fallback"
    return found_bank, reference_bank_source


def _collect_label_values_from_dataset(dataset) -> set[int]:
    label_values: set[int] = set()
    for index in range(len(dataset)):
        _, label = dataset[index]
        label_values.add(_resolve_label_value(label))
    return label_values


def select_reference_images_for_labels(
    torch_module,
    class_labels,
    validation_bundle,
    train_bundle,
    device,
):
    label_values = [_resolve_label_value(label) for label in class_labels.detach().cpu()]
    validation_found: dict[int, "torch.Tensor"] = {}
    if validation_bundle is not None:
        validation_found = _collect_reference_images_from_dataset(
            torch_module,
            validation_bundle.dataset,
            label_values,
        )

    found_images = dict(validation_found)
    missing_labels = sorted({label for label in label_values if label not in found_images})
    if missing_labels:
        found_images.update(
            _collect_reference_images_from_dataset(
                torch_module,
                train_bundle.dataset,
                missing_labels,
            )
        )

    unresolved = [label for label in label_values if label not in found_images]
    if unresolved:
        raise ValueError(f"Unable to find reference images for class labels: {sorted(set(unresolved))}")

    reference_images = torch_module.stack([found_images[label] for label in label_values]).to(device=device)
    reference_labels = torch_module.tensor(label_values, device=device, dtype=torch_module.long)
    if validation_bundle is None:
        reference_source = "train_only"
    elif all(label in validation_found for label in set(label_values)):
        reference_source = "validation_only"
    else:
        reference_source = "validation_then_train_fallback"
    return reference_images, reference_labels, reference_source


def select_first_stage_visual_batch(
    torch_module,
    validation_bundle,
    train_bundle,
    num_classes: int,
    device,
):
    # 固定 0-9 验证样本，避免每个 epoch 因训练 batch 变化而误判结构改进。
    visual_labels = build_post_train_sample_labels(
        torch_module,
        num_classes,
        device,
        sample_class_labels=tuple(range(num_classes)),
        sample_repeats_per_class=DEFAULT_VISUAL_PREVIEW_PER_CLASS,
    )
    try:
        return select_reference_images_for_labels(
            torch_module,
            visual_labels,
            validation_bundle,
            train_bundle,
            device,
        )
    except ValueError:
        available_labels = set()
        if validation_bundle is not None:
            available_labels.update(_collect_label_values_from_dataset(validation_bundle.dataset))
        available_labels.update(_collect_label_values_from_dataset(train_bundle.dataset))
        fallback_labels = sorted(label for label in available_labels if 0 <= label < num_classes)
        if not fallback_labels:
            raise
        fallback_tensor = torch_module.tensor(
            [label for label in fallback_labels for _ in range(DEFAULT_VISUAL_PREVIEW_PER_CLASS)],
            device=device,
            dtype=torch_module.long,
        )
        return select_reference_images_for_labels(
            torch_module,
            fallback_tensor,
            validation_bundle,
            train_bundle,
            device,
        )


def select_post_train_sample_batch(
    torch_module,
    validation_bundle,
    train_bundle,
    num_classes: int,
    device,
    sample_class_labels: list[int] | tuple[int, ...] | None = None,
    sample_repeats_per_class: int = 1,
):
    # 正式 MNIST 采样固定 0-9；小测试集缺类时退回到实际存在的类别，避免烟测被样本构造限制卡住。
    sample_labels = build_post_train_sample_labels(
        torch_module,
        num_classes,
        device,
        sample_class_labels=sample_class_labels,
        sample_repeats_per_class=sample_repeats_per_class,
    )
    try:
        reference_images, reference_labels, reference_source = select_reference_images_for_labels(
            torch_module,
            sample_labels,
            validation_bundle,
            train_bundle,
            device,
        )
        return sample_labels, reference_images, reference_labels, reference_source
    except ValueError:
        if sample_class_labels:
            raise
        available_labels = set()
        if validation_bundle is not None:
            available_labels.update(_collect_label_values_from_dataset(validation_bundle.dataset))
        available_labels.update(_collect_label_values_from_dataset(train_bundle.dataset))
        fallback_labels = sorted(label for label in available_labels if 0 <= label < num_classes)
        if not fallback_labels:
            raise
        fallback_tensor = torch_module.tensor(fallback_labels, device=device, dtype=torch_module.long)
        reference_images, reference_labels, reference_source = select_reference_images_for_labels(
            torch_module,
            fallback_tensor,
            validation_bundle,
            train_bundle,
            device,
        )
        return fallback_tensor, reference_images, reference_labels, reference_source


def save_post_train_samples(
    torch_module,
    run_manager,
    first_stage,
    diffusion_model,
    scheduler,
    train_bundle,
    validation_bundle,
    train_cfg,
    latent_scale,
    device,
    sample_latents_fn,
    sample_latents_with_trace_fn=None,
    resolve_trace_steps_fn=None,
) -> dict[str, str]:
    sample_labels, reference_images, reference_labels, reference_source = select_post_train_sample_batch(
        torch_module,
        validation_bundle,
        train_bundle,
        train_cfg.diffusion.num_classes,
        device,
        sample_class_labels=train_cfg.diffusion.sample_class_labels,
        sample_repeats_per_class=train_cfg.diffusion.sample_repeats_per_class,
    )
    reference_image_bank, reference_bank_source = build_reference_image_bank_for_labels(
        torch_module,
        sample_labels,
        validation_bundle,
        train_bundle,
    )
    sample_count = int(sample_labels.shape[0])
    run_manager.save_config_snapshot(
        "post_train_sample_config",
        {
            "num_samples": sample_count,
            "class_labels": sample_labels.detach().cpu().tolist(),
            "reference_labels": reference_labels.detach().cpu().tolist(),
            "reference_source": reference_source,
            "guidance_scale": train_cfg.diffusion.guidance_scale,
            "sample_class_labels": list(train_cfg.diffusion.sample_class_labels),
            "sample_repeats_per_class": train_cfg.diffusion.sample_repeats_per_class,
            "source": "train.py_post_train_visualization",
        },
    )
    sample_shape = (
        sample_count,
        train_cfg.first_stage.latent_channels,
        train_cfg.first_stage.image_size // 8,
        train_cfg.first_stage.image_size // 8,
    )
    with torch_module.no_grad():
        if sample_latents_with_trace_fn is None:
            final_sample_batch = sample_latents_fn(
                diffusion_model,
                scheduler,
                sample_shape,
                device,
                sample_labels,
                guidance_scale=train_cfg.diffusion.guidance_scale,
            )
            trace_latents = {"t000": final_sample_batch}
        else:
            final_sample_batch, trace_latents = sample_latents_with_trace_fn(
                diffusion_model,
                scheduler,
                sample_shape,
                device,
                sample_labels,
                guidance_scale=train_cfg.diffusion.guidance_scale,
                trace_timesteps=DEFAULT_DENOISING_TRACE_TIMESTEPS,
            )
        final_sample_outputs = first_stage.decode_with_intermediates(final_sample_batch * latent_scale)
        camera_trace = {
            label: first_stage.decode_with_intermediates(latents * latent_scale)["camera_plane"]
            for label, latents in trace_latents.items()
        }
    # 训练完成后的最终 sample 统一保存在 train run 目录，避免再创建独立 sample run。
    save_visualization_grids(
        run_manager,
        build_diffusion_visualization_tensors(reference_images, final_sample_outputs),
        run_manager.samples_dir / "post_train_sample.png",
        title_label="post-train sample outputs",
        max_items=sample_count,
        row_labels=labels_for_visual_rows(sample_labels),
    )
    trace_steps = (
        resolve_trace_steps_fn(scheduler.timesteps, DEFAULT_DENOISING_TRACE_TIMESTEPS)
        if resolve_trace_steps_fn is not None
        else []
    )
    run_manager.save_config_snapshot(
        "post_train_denoising_trace_config",
        {
            "num_samples": sample_count,
            "class_labels": sample_labels.detach().cpu().tolist(),
            "reference_labels": reference_labels.detach().cpu().tolist(),
            "reference_source": reference_source,
            "guidance_scale": train_cfg.diffusion.guidance_scale,
            "sample_class_labels": list(train_cfg.diffusion.sample_class_labels),
            "sample_repeats_per_class": train_cfg.diffusion.sample_repeats_per_class,
            "trace_columns": ["reference", *camera_trace.keys()],
            "trace_steps": trace_steps,
            "source": "train.py_post_train_denoising_trace",
        },
    )
    run_manager.save_tensor_grid(
        build_denoising_trace_visualization_tensors(reference_images, camera_trace),
        run_manager.samples_dir / "post_train_denoising_camera_trace.png",
        title=format_output_title(
            "images/samples/post_train_denoising_camera_trace.png",
            "post-train denoising camera trace",
        ),
        max_items=sample_count,
        row_labels=labels_for_visual_rows(sample_labels),
    )
    run_manager.save_tensor_bundle(
        "post_train_sample_tensors.pt",
        {
            "latents": final_sample_batch.detach().cpu(),
            "class_labels": sample_labels.detach().cpu(),
            "reference_images": reference_images.detach().cpu(),
            "reference_labels": reference_labels.detach().cpu(),
            "reference_source": reference_source,
            "guidance_scale": torch_module.tensor(train_cfg.diffusion.guidance_scale),
            "generated_camera": final_sample_outputs["camera_plane"].detach().cpu(),
            "generated_image": final_sample_outputs["image_recon"].detach().cpu(),
            "chip_amplitude": final_sample_outputs["chip_amplitude"].detach().cpu(),
            "chip_phase": final_sample_outputs["chip_phase"].detach().cpu(),
            "probe_plane": final_sample_outputs["probe_plane"].detach().cpu(),
            "camera_plane": final_sample_outputs["camera_plane"].detach().cpu(),
        },
    )
    run_manager.save_tensor_bundle(
        "post_train_denoising_trace_tensors.pt",
        {
            "class_labels": sample_labels.detach().cpu(),
            "reference_images": reference_images.detach().cpu(),
            "reference_labels": reference_labels.detach().cpu(),
            "reference_source": reference_source,
            "guidance_scale": torch_module.tensor(train_cfg.diffusion.guidance_scale),
            "trace_steps": trace_steps,
            "latent_trace": {label: latents.detach().cpu() for label, latents in trace_latents.items()},
            "camera_trace": {label: camera.detach().cpu() for label, camera in camera_trace.items()},
        },
    )
    sample_metrics = compute_sample_quality_metrics(
        final_sample_outputs["camera_plane"],
        reference_images,
        sample_labels,
        reference_image_bank=reference_image_bank,
        reference_bank_source=reference_bank_source,
    )
    metrics_path = run_manager.save_artifact_json("post_train_sample_metrics", sample_metrics)
    return {
        "sample_metric_source": str(metrics_path.relative_to(run_manager.run_dir)),
    }
