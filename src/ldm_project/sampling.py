from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

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
D2NN_LAYER_VISUALIZATION_SPECS = (
    ("diffractive_encoder_layer", "encoder field after"),
    ("diffractive_decoder_layer", "decoder field after"),
)
D2NN_MASK_VISUALIZATION_SPECS = (
    ("diffractive_encoder_mask", "encoder mask amp = 1"),
    ("diffractive_decoder_mask", "decoder mask amp = 1"),
)


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


def _block_grid_preview(tensor, grid_size: int = 40):
    raw = tensor.detach().float()
    if raw.ndim < 4:
        return _normalize_for_visualization(raw)
    coarse = F.interpolate(raw, size=(grid_size, grid_size), mode="area")
    blocky = F.interpolate(coarse, size=raw.shape[-2:], mode="nearest")
    return _normalize_for_visualization(blocky)


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


def _ensure_visual_batch_channel(tensor):
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    return tensor


def _broadcast_d2nn_mask_visualization(
    sample_outputs,
    batch_size: int,
    specs=D2NN_MASK_VISUALIZATION_SPECS,
):
    amplitude: dict[str, object] = {}
    phase: dict[str, object] = {}
    for key_prefix, label_prefix in specs:
        amplitude_key = f"{key_prefix}_amplitude"
        phase_key = f"{key_prefix}_phase"
        if amplitude_key not in sample_outputs or phase_key not in sample_outputs:
            continue
        amp_tensor = sample_outputs[amplitude_key]
        phs_tensor = sample_outputs[phase_key]
        if amp_tensor.ndim < 3 or phs_tensor.ndim < 3:
            continue
        num_layers = int(amp_tensor.shape[0])
        for layer_index in range(num_layers):
            amplitude_label = f"{label_prefix} L{layer_index + 1}"
            phase_label_prefix = label_prefix.replace("mask amp = 1", "mask phase")
            phase_label = f"{phase_label_prefix} L{layer_index + 1}"
            amp_layer = _ensure_visual_batch_channel(amp_tensor[layer_index]).expand(batch_size, -1, -1, -1)
            phs_layer = _ensure_visual_batch_channel(phs_tensor[layer_index]).expand(batch_size, -1, -1, -1)
            amplitude[amplitude_label] = _raw_amplitude_for_visualization(amp_layer)
            phase[phase_label] = _phase_for_visualization(phs_layer)
    return amplitude, phase


def _interleave_d2nn_mask_layer_columns(
    mask_columns: dict[str, object],
    layer_columns: dict[str, object],
    *,
    mask_prefix: str,
    layer_prefix: str,
) -> dict[str, object]:
    layer_indices = []
    for columns, prefix in ((mask_columns, mask_prefix), (layer_columns, layer_prefix)):
        marker = f"{prefix} L"
        for label in columns:
            if label.startswith(marker):
                try:
                    layer_indices.append(int(label.removeprefix(marker)))
                except ValueError:
                    continue
    interleaved: dict[str, object] = {}
    for layer_index in range(1, max(layer_indices, default=0) + 1):
        mask_label = f"{mask_prefix} L{layer_index}"
        layer_label = f"{layer_prefix} L{layer_index}"
        if mask_label in mask_columns:
            interleaved[mask_label] = mask_columns[mask_label]
        if layer_label in layer_columns:
            interleaved[layer_label] = layer_columns[layer_label]
    return interleaved


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


def _cross_sample_amplitude_display(tensor, lower_q: float = 0.02, upper_q: float = 0.98):
    raw = tensor.detach().float().clamp_min(0.0)
    lo = raw.quantile(lower_q)
    hi = raw.quantile(upper_q)
    return ((raw - lo) / max(float(hi - lo), 1e-6)).clamp(0.0, 1.0)


def _resolve_d2nn_layer_dim(tensor_4d: torch.Tensor, key_prefix: str = "") -> int:
    _ = tensor_4d
    if "mask" in key_prefix:
        return 0
    return 1


def _build_d2nn_layer_split(sample_outputs, specs) -> dict[str, dict[str, object]]:
    amplitude: dict[str, object] = {}
    phase: dict[str, object] = {}
    for key_prefix, label_prefix in specs:
        amplitude_key = f"{key_prefix}_amplitude"
        phase_key = f"{key_prefix}_phase"
        if amplitude_key not in sample_outputs or phase_key not in sample_outputs:
            continue
        amplitude_tensor = sample_outputs[amplitude_key]
        phase_tensor = sample_outputs[phase_key]
        if amplitude_tensor.ndim < 4 or phase_tensor.ndim < 4:
            continue
        if amplitude_tensor.ndim == 5:
            layer_count = int(amplitude_tensor.shape[1])
        elif amplitude_tensor.ndim == 4:
            layer_dim = _resolve_d2nn_layer_dim(amplitude_tensor, key_prefix)
            layer_count = int(amplitude_tensor.shape[layer_dim])
        else:
            continue
        for layer_index in range(layer_count):
            amplitude_label = f"{label_prefix} L{layer_index + 1}"
            phase_label_prefix = label_prefix.replace("mask amp = 1", "mask phase")
            phase_label = f"{phase_label_prefix} L{layer_index + 1}"
            if amplitude_tensor.ndim == 5:
                amplitude_layer = amplitude_tensor[:, layer_index]
                phase_layer = phase_tensor[:, layer_index]
            elif layer_dim == 1:
                amplitude_layer = amplitude_tensor[:, layer_index]
                phase_layer = phase_tensor[:, layer_index]
            else:
                amplitude_layer = amplitude_tensor[layer_index]
                phase_layer = phase_tensor[layer_index]
            amplitude_layer = _ensure_visual_batch_channel(amplitude_layer)
            phase_layer = _ensure_visual_batch_channel(phase_layer)
            if "mask" in key_prefix:
                amplitude[amplitude_label] = _raw_amplitude_for_visualization(amplitude_layer)
            else:
                amplitude[amplitude_label] = _cross_sample_amplitude_display(amplitude_layer)
            phase[phase_label] = _phase_for_visualization(phase_layer)
    return {
        VISUALIZATION_AMPLITUDE_KEY: amplitude,
        VISUALIZATION_PHASE_KEY: phase,
    }


def _d2nn_field_layer_visualization_columns(
    sample_outputs,
    specs=D2NN_LAYER_VISUALIZATION_SPECS,
) -> tuple[dict[str, object], dict[str, object]]:
    splits = _build_d2nn_layer_split(sample_outputs, specs)
    return splits[VISUALIZATION_AMPLITUDE_KEY], splits[VISUALIZATION_PHASE_KEY]


def build_one_shot_d2nn_visualization_tensors(sample_outputs) -> dict[str, dict[str, dict[str, object]]]:
    return {
        "layers": _build_d2nn_layer_split(sample_outputs, D2NN_LAYER_VISUALIZATION_SPECS),
        "masks": _build_d2nn_layer_split(sample_outputs, D2NN_MASK_VISUALIZATION_SPECS),
    }


def _extract_d2nn_field_layer(tensor: torch.Tensor, layer_index: int) -> torch.Tensor:
    if tensor.ndim == 5:
        if layer_index >= tensor.shape[1]:
            raise IndexError(f"layer_index {layer_index} is out of range for tensor shape {tuple(tensor.shape)}")
        return _ensure_visual_batch_channel(tensor[:, layer_index])
    if tensor.ndim == 4:
        if layer_index >= tensor.shape[1]:
            raise IndexError(f"layer_index {layer_index} is out of range for tensor shape {tuple(tensor.shape)}")
        return _ensure_visual_batch_channel(tensor[:, layer_index])
    raise ValueError(f"D2NN field layer tensor must have 4 or 5 dimensions, got {tuple(tensor.shape)}")


def build_one_shot_d2nn_focus_visualization_tensors(
    sample_outputs,
    *,
    sample_index: int = 0,
    stack: str = "encoder",
    layer_index: int = 0,
) -> dict[str, dict[str, object]]:
    if stack not in {"encoder", "decoder"}:
        raise ValueError(f"stack must be 'encoder' or 'decoder', got {stack!r}")
    amplitude_key = f"diffractive_{stack}_layer_amplitude"
    phase_key = f"diffractive_{stack}_layer_phase"
    if amplitude_key not in sample_outputs or phase_key not in sample_outputs:
        return {VISUALIZATION_AMPLITUDE_KEY: {}, VISUALIZATION_PHASE_KEY: {}}
    amplitude_tensor = sample_outputs[amplitude_key]
    phase_tensor = sample_outputs[phase_key]
    if amplitude_tensor.ndim < 4 or phase_tensor.ndim < 4:
        return {VISUALIZATION_AMPLITUDE_KEY: {}, VISUALIZATION_PHASE_KEY: {}}
    amplitude_layer = _extract_d2nn_field_layer(amplitude_tensor, layer_index)
    phase_layer = _extract_d2nn_field_layer(phase_tensor, layer_index)
    if sample_index < 0 or sample_index >= amplitude_layer.shape[0]:
        raise IndexError(f"sample_index {sample_index} is out of range for tensor shape {tuple(amplitude_layer.shape)}")
    label = f"{stack} field after L{layer_index + 1}"
    sample_slice = slice(sample_index, sample_index + 1)
    return {
        VISUALIZATION_AMPLITUDE_KEY: {
            f"{label} amplitude": _robust_amplitude_display(amplitude_layer[sample_slice]),
        },
        VISUALIZATION_PHASE_KEY: {
            f"{label} phase": _phase_for_visualization(phase_layer[sample_slice]),
        },
    }


def _focus_sample_tensor(tensor: torch.Tensor, sample_index: int) -> torch.Tensor:
    value = _ensure_visual_batch_channel(tensor)
    if sample_index < 0 or sample_index >= value.shape[0]:
        raise IndexError(f"sample_index {sample_index} is out of range for tensor shape {tuple(value.shape)}")
    return value[sample_index : sample_index + 1]


def build_one_shot_slm_to_encoder_focus_visualization_tensors(
    sample_outputs,
    *,
    sample_index: int = 0,
) -> dict[str, dict[str, object]]:
    required = (
        "slm_amplitude",
        "slm_phase",
        "encoder_input_amplitude",
        "encoder_input_phase",
        "encoder_input_grid_amplitude",
        "encoder_input_grid_phase",
    )
    if any(key not in sample_outputs for key in required):
        return {VISUALIZATION_AMPLITUDE_KEY: {}, VISUALIZATION_PHASE_KEY: {}}
    return {
        VISUALIZATION_AMPLITUDE_KEY: {
            "SLM plane before propagation amplitude": _raw_amplitude_for_visualization(
                _focus_sample_tensor(sample_outputs["slm_amplitude"], sample_index)
            ),
            "encoder input after propagation amplitude": _robust_amplitude_display(
                _focus_sample_tensor(sample_outputs["encoder_input_amplitude"], sample_index)
            ),
            "encoder grid after propagation amplitude": _robust_amplitude_display(
                _focus_sample_tensor(sample_outputs["encoder_input_grid_amplitude"], sample_index)
            ),
        },
        VISUALIZATION_PHASE_KEY: {
            "SLM plane before propagation phase": _phase_for_visualization(
                _focus_sample_tensor(sample_outputs["slm_phase"], sample_index)
            ),
            "encoder input after propagation phase": _phase_for_visualization(
                _focus_sample_tensor(sample_outputs["encoder_input_phase"], sample_index)
            ),
            "encoder grid after propagation phase": _phase_for_visualization(
                _focus_sample_tensor(sample_outputs["encoder_input_grid_phase"], sample_index)
            ),
        },
    }


def _single_image_array(tensor: torch.Tensor):
    value = tensor.detach().float().cpu()
    if value.ndim == 4:
        value = value[0]
    if value.ndim == 3:
        value = value.mean(dim=0) if value.shape[0] > 1 else value.squeeze(0)
    if value.ndim != 2:
        raise ValueError(f"focus tensor must resolve to one 2D image, got shape {tuple(tensor.shape)}")
    return value.clamp(0.0, 1.0).numpy()


def _resolve_focus_destination(run_manager, destination: str | Path) -> Path:
    if isinstance(destination, Path):
        return destination
    return run_manager.run_dir / destination


def save_one_shot_d2nn_focus_figures(
    run_manager,
    focus_visuals: dict[str, dict[str, object]],
    destination: str | Path,
    *,
    title_label: str,
    row_label: str | None = None,
    figure_pixels: int = 1200,
) -> dict[str, Path]:
    base_path = _resolve_focus_destination(run_manager, destination)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
    except ImportError:
        return {}

    dpi = 100
    panel_size = max(float(figure_pixels) / float(dpi), 8.0)
    for split_name, tensor_map in focus_visuals.items():
        if not tensor_map:
            continue
        items = list(tensor_map.items())
        images = [(label, _single_image_array(tensor)) for label, tensor in items]
        path = base_path.with_name(f"{base_path.stem}_{split_name}{base_path.suffix}")
        cmap = "twilight" if split_name == VISUALIZATION_PHASE_KEY else "gray"
        figure, axes = plt.subplots(
            1,
            len(images),
            squeeze=False,
            figsize=(panel_size * len(images), panel_size),
            dpi=dpi,
            constrained_layout=True,
        )
        for column, (label, image) in enumerate(images):
            axis = axes[0][column]
            rendered = axis.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{title_label} {split_name}\n{label}", fontsize=12)
            if row_label and column == 0:
                axis.text(
                    0.02,
                    0.02,
                    row_label,
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=11,
                    color="white" if split_name == VISUALIZATION_AMPLITUDE_KEY else "black",
                    bbox={
                        "facecolor": "black" if split_name == VISUALIZATION_AMPLITUDE_KEY else "white",
                        "alpha": 0.65,
                        "pad": 3,
                    },
                )
        if split_name == VISUALIZATION_PHASE_KEY:
            scalar = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=plt.get_cmap(cmap))
            scalar.set_array([])
            colorbar = figure.colorbar(
                scalar,
                ax=axes.ravel().tolist(),
                orientation="horizontal",
                shrink=0.8,
                pad=0.03,
                ticks=(0.0, 0.5, 1.0),
            )
            colorbar.ax.set_xticklabels(("0", "π", "2π"))
            colorbar.set_label("phase", fontsize=10)
        else:
            _ = rendered
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        saved_paths[split_name] = path
    return saved_paths


def _finite_distribution_stats(tensor) -> dict[str, object]:
    value = tensor.detach().float().cpu()
    finite_mask = torch.isfinite(value)
    finite = value[finite_mask]
    stats: dict[str, object] = {
        "shape": list(value.shape),
        "finite_ratio": float(finite_mask.float().mean().item()),
    }
    if finite.numel() == 0:
        return {
            **stats,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p01": None,
            "p50": None,
            "p99": None,
        }
    return {
        **stats,
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item()),
        "p01": float(finite.quantile(0.01).item()),
        "p50": float(finite.quantile(0.50).item()),
        "p99": float(finite.quantile(0.99).item()),
    }


def _amplitude_feasibility_stats(tensor) -> dict[str, object]:
    raw = tensor.detach().float().clamp_min(0.0).cpu()
    if raw.ndim <= 2:
        batched = raw.reshape(1, -1)
    elif raw.ndim == 3:
        batched = raw.unsqueeze(0).flatten(start_dim=1)
    else:
        batched = raw.flatten(start_dim=1)
    mean_amplitude = batched.mean(dim=1).clamp_min(1.0e-8)
    p99_amplitude = batched.quantile(0.99, dim=1)
    energy = batched.square().sum(dim=1)
    return {
        "energy_mean": float(energy.mean().item()),
        "energy_min": float(energy.min().item()),
        "energy_max": float(energy.max().item()),
        "p99_to_mean_ratio_mean": float((p99_amplitude / mean_amplitude).mean().item()),
        "near_zero_fraction": float((batched <= 1.0e-6).float().mean().item()),
    }


def _phase_local_contrast_stats(tensor) -> dict[str, object]:
    raw = tensor.detach().float().cpu()
    if raw.ndim < 2:
        return {"local_contrast_mean": 0.0, "local_contrast_p99": 0.0}
    sin_phase = torch.sin(raw)
    cos_phase = torch.cos(raw)
    diff_y = (sin_phase[..., 1:, :] - sin_phase[..., :-1, :]).abs() + (
        cos_phase[..., 1:, :] - cos_phase[..., :-1, :]
    ).abs()
    diff_x = (sin_phase[..., :, 1:] - sin_phase[..., :, :-1]).abs() + (
        cos_phase[..., :, 1:] - cos_phase[..., :, :-1]
    ).abs()
    contrast = torch.cat([diff_y.flatten(), diff_x.flatten()])
    if contrast.numel() == 0:
        return {"local_contrast_mean": 0.0, "local_contrast_p99": 0.0}
    return {
        "local_contrast_mean": float(contrast.mean().item()),
        "local_contrast_p99": float(contrast.quantile(0.99).item()),
    }


def _d2nn_stack_diagnostics(sample_outputs, specs) -> dict[str, dict[str, dict[str, object]]]:
    amplitude: dict[str, dict[str, object]] = {}
    phase: dict[str, dict[str, object]] = {}
    for key_prefix, label_prefix in specs:
        amplitude_key = f"{key_prefix}_amplitude"
        phase_key = f"{key_prefix}_phase"
        if amplitude_key not in sample_outputs or phase_key not in sample_outputs:
            continue
        amplitude_tensor = sample_outputs[amplitude_key]
        phase_tensor = sample_outputs[phase_key]
        if amplitude_tensor.ndim < 4 or phase_tensor.ndim < 4:
            continue
        if amplitude_tensor.ndim == 5:
            layer_count = int(amplitude_tensor.shape[1])
            layer_dim = 1
        elif amplitude_tensor.ndim == 4:
            layer_dim = _resolve_d2nn_layer_dim(amplitude_tensor, key_prefix)
            layer_count = int(amplitude_tensor.shape[layer_dim])
        else:
            continue
        for layer_index in range(layer_count):
            amplitude_label = f"{label_prefix} L{layer_index + 1}"
            phase_label_prefix = label_prefix.replace("mask amp = 1", "mask phase")
            phase_label = f"{phase_label_prefix} L{layer_index + 1}"
            if amplitude_tensor.ndim == 5:
                amplitude_layer = amplitude_tensor[:, layer_index]
                phase_layer = phase_tensor[:, layer_index]
            elif layer_dim == 1:
                amplitude_layer = amplitude_tensor[:, layer_index]
                phase_layer = phase_tensor[:, layer_index]
            else:
                amplitude_layer = amplitude_tensor[layer_index]
                phase_layer = phase_tensor[layer_index]
            amplitude[amplitude_label] = {
                **_finite_distribution_stats(amplitude_layer),
                **_amplitude_feasibility_stats(amplitude_layer),
            }
            phase_stats = _finite_distribution_stats(phase_layer)
            if "mask" in key_prefix:
                phase_stats = {**phase_stats, **_phase_local_contrast_stats(phase_layer)}
            phase[phase_label] = phase_stats
    return {
        "amplitude": amplitude,
        "phase": phase,
    }


def _phase_only_mask_checks(sample_outputs) -> dict[str, dict[str, object]]:
    checks: dict[str, dict[str, object]] = {}
    for key in ("diffractive_encoder_mask_amplitude", "diffractive_decoder_mask_amplitude"):
        if key not in sample_outputs:
            continue
        delta = (sample_outputs[key].detach().float().cpu() - 1.0).abs()
        checks[key] = {
            "max_abs_deviation_from_one": float(delta.max().item()),
            "mean_abs_deviation_from_one": float(delta.mean().item()),
            "allclose_to_one_atol_1e-6": bool(delta.max().item() <= 1.0e-6),
        }
    return checks


def _per_sample_energy(tensor) -> torch.Tensor:
    raw = tensor.detach().float().clamp_min(0.0)
    if raw.ndim <= 2:
        batched = raw.reshape(1, -1)
    elif raw.ndim == 3:
        batched = raw.unsqueeze(0).flatten(start_dim=1)
    else:
        batched = raw.flatten(start_dim=1)
    return batched.square().sum(dim=1).cpu()


def _energy_retention_summary(numerator, denominator) -> dict[str, object]:
    numerator_energy = _per_sample_energy(numerator)
    denominator_energy = _per_sample_energy(denominator).clamp_min(1.0e-12)
    if numerator_energy.shape != denominator_energy.shape:
        return {
            "ratio_mean": None,
            "ratio_min": None,
            "ratio_max": None,
            "numerator_energy_mean": float(numerator_energy.mean().item()),
            "denominator_energy_mean": float(denominator_energy.mean().item()),
        }
    ratio = numerator_energy / denominator_energy
    return {
        "ratio_mean": float(ratio.mean().item()),
        "ratio_min": float(ratio.min().item()),
        "ratio_max": float(ratio.max().item()),
        "numerator_energy_mean": float(numerator_energy.mean().item()),
        "denominator_energy_mean": float(denominator_energy.mean().item()),
    }


def _d2nn_energy_retention(sample_outputs) -> dict[str, dict[str, object]]:
    required = (
        "encoder_input_grid_amplitude",
        "decoder_input_amplitude",
        "diffractive_encoder_layer_amplitude",
        "diffractive_decoder_layer_amplitude",
    )
    if any(key not in sample_outputs for key in required):
        return {}
    try:
        encoder_l1 = _extract_d2nn_field_layer(sample_outputs["diffractive_encoder_layer_amplitude"], 0)
        encoder_l2 = _extract_d2nn_field_layer(sample_outputs["diffractive_encoder_layer_amplitude"], 1)
        decoder_l1 = _extract_d2nn_field_layer(sample_outputs["diffractive_decoder_layer_amplitude"], 0)
        decoder_l2 = _extract_d2nn_field_layer(sample_outputs["diffractive_decoder_layer_amplitude"], 1)
    except (IndexError, ValueError):
        return {}
    return {
        "encoder field after L1": _energy_retention_summary(
            encoder_l1,
            sample_outputs["encoder_input_grid_amplitude"],
        ),
        "encoder field after L2": _energy_retention_summary(encoder_l2, encoder_l1),
        "decoder field after L1": _energy_retention_summary(
            decoder_l1,
            sample_outputs["decoder_input_amplitude"],
        ),
        "decoder field after L2": _energy_retention_summary(decoder_l2, decoder_l1),
    }


def _layer_ratio_summary(ratios: torch.Tensor, layer_index: int) -> dict[str, object]:
    if ratios.ndim != 2 or layer_index >= ratios.shape[1]:
        return {"ratio_mean": None, "ratio_min": None, "ratio_max": None}
    values = ratios[:, layer_index].detach().float().cpu()
    return {
        "ratio_mean": float(values.mean().item()),
        "ratio_min": float(values.min().item()),
        "ratio_max": float(values.max().item()),
    }


def _layer_energy_summary(values: torch.Tensor, layer_index: int, prefix: str) -> dict[str, object]:
    if values.ndim != 2 or layer_index >= values.shape[1]:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_min": None,
            f"{prefix}_max": None,
        }
    layer_values = values[:, layer_index].detach().float().cpu()
    return {
        f"{prefix}_mean": float(layer_values.mean().item()),
        f"{prefix}_min": float(layer_values.min().item()),
        f"{prefix}_max": float(layer_values.max().item()),
    }


def _d2nn_crop_energy_retention(sample_outputs) -> dict[str, dict[str, object]]:
    specs = (
        ("diffractive_encoder_crop", "encoder field after"),
        ("diffractive_decoder_crop", "decoder field after"),
    )
    diagnostics: dict[str, dict[str, object]] = {}
    for key_prefix, label_prefix in specs:
        crop_key = f"{key_prefix}_crop_retention"
        propagation_key = f"{key_prefix}_propagation_energy_ratio"
        cropped_energy_key = f"{key_prefix}_cropped_energy"
        propagated_energy_key = f"{key_prefix}_propagated_full_energy"
        padded_energy_key = f"{key_prefix}_padded_input_energy"
        if crop_key not in sample_outputs or propagation_key not in sample_outputs:
            continue
        crop_retention = sample_outputs[crop_key]
        propagation_ratio = sample_outputs[propagation_key]
        layer_count = int(crop_retention.shape[1]) if crop_retention.ndim == 2 else 0
        for layer_index in range(layer_count):
            label = f"{label_prefix} L{layer_index + 1}"
            diagnostics[label] = {
                **_layer_ratio_summary(crop_retention, layer_index),
                **{
                    f"propagation_energy_{key}": value
                    for key, value in _layer_ratio_summary(propagation_ratio, layer_index).items()
                },
            }
            if cropped_energy_key in sample_outputs:
                diagnostics[label].update(
                    _layer_energy_summary(sample_outputs[cropped_energy_key], layer_index, "cropped_energy")
                )
            if propagated_energy_key in sample_outputs:
                diagnostics[label].update(
                    _layer_energy_summary(sample_outputs[propagated_energy_key], layer_index, "propagated_full_energy")
                )
            if padded_energy_key in sample_outputs:
                diagnostics[label].update(
                    _layer_energy_summary(sample_outputs[padded_energy_key], layer_index, "padded_input_energy")
                )
    return diagnostics


def build_one_shot_d2nn_diagnostics(sample_outputs) -> dict[str, object]:
    layers = _d2nn_stack_diagnostics(sample_outputs, D2NN_LAYER_VISUALIZATION_SPECS)
    masks = _d2nn_stack_diagnostics(sample_outputs, D2NN_MASK_VISUALIZATION_SPECS)
    identity_checks = {}
    if "ols_input_amplitude" in sample_outputs and "ols_output_amplitude" in sample_outputs:
        identity_checks["ols_input_to_output_amplitude"] = _tensor_diff_summary(
            sample_outputs["ols_input_amplitude"],
            sample_outputs["ols_output_amplitude"],
        )
    if "ols_input_phase" in sample_outputs and "ols_output_phase" in sample_outputs:
        identity_checks["ols_input_to_output_phase"] = _tensor_diff_summary(
            sample_outputs["ols_input_phase"],
            sample_outputs["ols_output_phase"],
        )
    available = bool(layers["amplitude"] or layers["phase"] or masks["amplitude"] or masks["phase"])
    return {
        "view": "d2nn_phase_only_feasibility",
        "available": available,
        "layers": layers,
        "masks": masks,
        "energy_retention": _d2nn_energy_retention(sample_outputs),
        "crop_energy_retention": _d2nn_crop_energy_retention(sample_outputs),
        "phase_only_mask_checks": _phase_only_mask_checks(sample_outputs),
        "identity_checks": identity_checks,
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
