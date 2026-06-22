from __future__ import annotations

from pathlib import Path

import torch

from .visualization_core import (
    VISUALIZATION_AMPLITUDE_KEY,
    VISUALIZATION_PHASE_KEY,
    _phase_for_visualization,
    _raw_amplitude_for_visualization,
    _robust_amplitude_display,
)

D2NN_LAYER_VISUALIZATION_SPECS = (
    ("diffractive_encoder_layer", "encoder field after"),
    ("diffractive_decoder_layer", "decoder field after"),
)
D2NN_MASK_VISUALIZATION_SPECS = (
    ("diffractive_encoder_mask", "encoder mask transmittance |t|"),
    ("diffractive_decoder_mask", "decoder mask transmittance |t|"),
)


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
            phase_label_prefix = label_prefix.replace("transmittance |t|", "phase arg(t)")
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
                    layer_indices.append(int(label.removeprefix(marker).split()[0]))
                except ValueError:
                    continue
    interleaved: dict[str, object] = {}
    for layer_index in range(1, max(layer_indices, default=0) + 1):
        mask_label = f"{mask_prefix} L{layer_index}"
        layer_label = f"{layer_prefix} L{layer_index}"
        phase_layer_label = f"{layer_label} phase"
        if mask_label in mask_columns:
            interleaved[mask_label] = mask_columns[mask_label]
        if layer_label in layer_columns:
            interleaved[layer_label] = layer_columns[layer_label]
        elif phase_layer_label in layer_columns:
            interleaved[phase_layer_label] = layer_columns[phase_layer_label]
    return interleaved



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
            if "transmittance |t|" in label_prefix:
                phase_label_prefix = label_prefix.replace("transmittance |t|", "phase arg(t)")
                phase_label = f"{phase_label_prefix} L{layer_index + 1}"
            else:
                phase_label = f"{label_prefix} L{layer_index + 1} phase"
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
