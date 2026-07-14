from __future__ import annotations

import torch.nn.functional as F

from ..diagnostics.d2nn_visualization import (
    _d2nn_field_layer_visualization_columns,
)
from ..visualization_core import (
    VISUALIZATION_AMPLITUDE_KEY,
    VISUALIZATION_INTENSITY_KEY,
    VISUALIZATION_PHASE_KEY,
    _amplitude_for_visualization,
    _code_for_visualization,
    _intensity_for_visualization,
    _normalize_for_visualization,
    _phase_for_visualization,
    _raw_amplitude_for_visualization,
    _robust_amplitude_display,
    _tensor_stats,
    _tensor_diff_summary,
    enhance_camera_for_display,
    readable_camera_for_display,
)

ONE_SHOT_SHARED_SCALE_INTENSITY_QUANTILE = 0.995


ONE_SHOT_SHARED_SCALE_INTENSITY_SPECS = (
    ("slm_amplitude", "slm out I"),
    ("encoder_input_amplitude", "encoder in I"),
    ("encoder_input_grid_amplitude", "encoder grid in I"),
    ("encoded_amplitude", "encoder out I"),
    ("ols_input_amplitude", "ols in I"),
    ("ols_output_amplitude", "ols out I"),
    ("decoder_input_amplitude", "decoder in I"),
    ("decoder_output_amplitude", "decoder out I"),
    ("chip_amplitude", "chip I"),
)


def _shared_scale(
    tensors, quantile: float = ONE_SHOT_SHARED_SCALE_INTENSITY_QUANTILE
) -> float:
    if quantile <= 0.0 or quantile > 1.0:
        raise ValueError(f"shared scale quantile must be within (0, 1], got {quantile}")
    values = [
        tensor.detach().float().clamp_min(0.0).flatten().cpu() for tensor in tensors
    ]
    if not values:
        return 1.0
    if len(values) == 1:
        concatenated = values[0]
    else:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to build shared-scale visualizations."
            ) from exc
        # Subsample each tensor to avoid OOM with many large images (e.g. 200 eval samples).
        for i in range(len(values)):
            if values[i].numel() > 1_000_000:
                idx = torch.randperm(values[i].shape[0], device=values[i].device)[
                    :1_000_000
                ]
                values[i] = values[i][idx]
        concatenated = torch.cat(values)
    if concatenated.numel() == 0:
        return 1.0
    scale = float(concatenated.quantile(quantile).item())
    return max(scale, 1e-6)


def _scale_intensity_for_visualization(amplitude, scale: float):
    intensity = amplitude.detach().float().clamp_min(0.0).pow(2)
    return (intensity / max(float(scale), 1e-6)).clamp(0.0, 1.0)


def _slm_contrast_for_visualization(tensor):
    return _normalize_for_visualization(tensor.detach().float())


def _block_grid_preview(tensor, grid_size: int = 40):
    raw = tensor.detach().float()
    if raw.ndim < 4:
        return _normalize_for_visualization(raw)
    coarse = F.interpolate(raw, size=(grid_size, grid_size), mode="area")
    blocky = F.interpolate(coarse, size=raw.shape[-2:], mode="nearest")
    return _normalize_for_visualization(blocky)


def _infer_neuron_lattice_grid_size(sample_outputs) -> int:
    for key in (
        "encoder_input_grid_amplitude",
        "encoded_amplitude",
        "decoder_input_amplitude",
        "decoder_output_amplitude",
        "chip_amplitude",
    ):
        tensor = sample_outputs.get(key)
        if tensor is not None and getattr(tensor, "ndim", 0) >= 4:
            height, width = tensor.shape[-2:]
            if int(height) != int(width):
                raise ValueError(
                    f"{key} must be square for neuron-lattice visualization, got {height}x{width}"
                )
            return int(height)
    raise KeyError("Unable to infer neuron-lattice grid size from one-shot outputs.")


def _to_neuron_lattice_grid(tensor, grid_size: int):
    raw = tensor.detach().float().clamp_min(0.0)
    if raw.ndim < 4:
        raise ValueError(
            f"neuron-lattice amplitude tensors must be 4D, got shape={tuple(raw.shape)}"
        )
    if raw.shape[-2:] == (grid_size, grid_size):
        return raw
    return F.interpolate(raw, size=(grid_size, grid_size), mode="area")


def _neuron_lattice_display(tensor):
    return _robust_amplitude_display(tensor)


def build_one_shot_neuron_lattice_amplitude_tensors(
    sample_outputs, *, pixel_size: float | None = None
):
    """Return a true grid-cell amplitude view for D2NN/SLM neuron-lattice inspection.

    The output tensors are all shaped as the inferred D2NN grid, e.g. 100x100 for
    the two-photon full-connect path. 200x200 image-plane SLM controls are
    explicitly area-resampled to this lattice; propagated fields already live on
    the lattice and are not upsampled.
    """
    grid_size = _infer_neuron_lattice_grid_size(sample_outputs)
    specs = (
        ("slm_command", "SLM command/control grid", "area_resampled_to_lattice"),
        ("slm_amplitude", "SLM amplitude grid |E|", "area_resampled_to_lattice"),
        ("encoder_input_grid_amplitude", "encoder input grid |E|", "raw_lattice"),
        ("encoded_amplitude", "encoder out |E|", "raw_lattice"),
        ("target_code_amplitude", "target code |E|", "raw_lattice"),
        ("decoder_input_amplitude", "decoder in |E|", "raw_lattice"),
        ("decoder_output_amplitude", "decoder out |E|", "raw_lattice"),
        ("chip_amplitude", "chip |E|", "raw_lattice"),
    )
    columns: dict[str, object] = {}
    column_metadata: dict[str, object] = {}
    for key, label, source_policy in specs:
        if key not in sample_outputs:
            continue
        lattice = _to_neuron_lattice_grid(sample_outputs[key], grid_size)
        columns[label] = _neuron_lattice_display(lattice)
        column_metadata[label] = {
            "source_tensor": key,
            "source_shape": list(sample_outputs[key].shape),
            "lattice_shape": list(lattice.shape),
            "source_policy": (
                "raw_lattice"
                if sample_outputs[key].shape[-2:] == (grid_size, grid_size)
                else source_policy
            ),
        }
    metadata = {
        "view": "neuron_lattice_amplitude",
        "grid_size": grid_size,
        "pixel_size_um": None if pixel_size is None else float(pixel_size) * 1.0e6,
        "display_interpolation": "nearest",
        "value_kind": "raw_or_explicitly_normalized_amplitude",
        "display_normalization": "per-sample robust amplitude normalization, q02-q98",
        "columns": column_metadata,
    }
    return columns, metadata


def build_one_shot_slim_cell_chain_preview_tensors(
    sample_outputs, *, pixel_size: float | None = None
):
    """Return a v4-style slim amplitude panel: SLM, encoder, decoder, CMOS."""
    grid_size = _infer_neuron_lattice_grid_size(sample_outputs)
    specs: list[tuple[str, str, str, str]] = [
        ("slm_command", "SLM command", "area_resampled_to_lattice", "amplitude"),
        ("slm_amplitude", "SLM out amp", "area_resampled_to_lattice", "amplitude"),
        ("encoder_input_grid_amplitude", "Encoder in amp", "raw_lattice", "amplitude"),
        ("encoded_amplitude", "Encoder out amp", "raw_lattice", "amplitude"),
        ("target_code_amplitude", "Target code amp", "raw_lattice", "amplitude"),
        ("decoder_input_amplitude", "Decoder in amp", "raw_lattice", "amplitude"),
        ("decoder_output_amplitude", "Decoder out amp", "raw_lattice", "amplitude"),
    ]
    primary_readout = sample_outputs.get("primary_readout_plane", "camera")
    if not isinstance(primary_readout, str):
        primary_readout = "camera"
    if primary_readout == "camera":
        physical_cmos_keys = (
            "physical_camera_plane",
            "generated_camera_raw",
            "generated_camera",
            "student_camera_raw",
            "camera_plane",
            "diffractive_camera_plane_raw",
        )
    else:
        physical_cmos_keys = (
            "physical_camera_plane",
            "diffractive_camera_plane_raw",
            "generated_camera_raw",
            "generated_camera",
            "student_camera_raw",
        )
    physical_cmos_source_key = next(
        (key for key in physical_cmos_keys if key in sample_outputs), None
    )
    if primary_readout == "camera":
        if physical_cmos_source_key is not None:
            specs.append(
                (
                    physical_cmos_source_key,
                    "CMOS intensity",
                    "area_resampled_camera_to_lattice",
                    "intensity",
                )
            )
    else:
        selected_source_key = next(
            (
                key
                for key in ("selected_readout", "selected_readout_raw", "camera_plane")
                if key in sample_outputs
            ),
            None,
        )
        if selected_source_key is not None:
            specs.append(
                (
                    selected_source_key,
                    f"Selected readout ({primary_readout})",
                    "area_resampled_selected_readout_to_lattice",
                    "intensity",
                )
            )
        if physical_cmos_source_key is not None:
            specs.append(
                (
                    physical_cmos_source_key,
                    "Physical CMOS intensity",
                    "area_resampled_physical_cmos_to_lattice",
                    "intensity",
                )
            )
    columns: dict[str, object] = {}
    column_metadata: dict[str, object] = {}
    for key, label, source_policy, value_kind in specs:
        if key not in sample_outputs:
            continue
        source = sample_outputs[key]
        if value_kind == "intensity":
            camera_display = enhance_camera_for_display(source)
            display = F.interpolate(
                camera_display, size=(grid_size, grid_size), mode="area"
            )
            rendered_shape = list(display.shape)
            source_policy = "area_resampled_camera_to_lattice"
        else:
            lattice = _to_neuron_lattice_grid(source, grid_size)
            display = _neuron_lattice_display(lattice)
            rendered_shape = list(lattice.shape)
            if source.shape[-2:] == (grid_size, grid_size):
                source_policy = "raw_lattice"
        columns[label] = display
        column_metadata[label] = {
            "source_tensor": key,
            "source_shape": list(source.shape),
            "rendered_shape": rendered_shape,
            "source_policy": source_policy,
            "value_kind": value_kind,
        }
    metadata = {
        "render_mode": "v4_style_slim_amplitude_preview",
        "columns": [label for _, label, _, _ in specs if label in columns],
        "grid_size": grid_size,
        "cell_pitch_um": None if pixel_size is None else float(pixel_size) * 1.0e6,
        "display_interpolation": "nearest",
        "panel_style": "v4_epoch_amplitude",
        "value_kind": "amplitude_chain_with_cmos_intensity",
        "primary_readout_plane": primary_readout,
        "physical_cmos_source_tensor": physical_cmos_source_key,
        "column_metadata": column_metadata,
    }
    return columns, metadata


def _field_layer_amplitude_columns(
    sample_outputs, specs
) -> tuple[dict[str, object], dict[str, object]]:
    amplitude_columns, phase_columns = _d2nn_field_layer_visualization_columns(
        sample_outputs, specs
    )
    amplitude = {f"{label} |E|": value for label, value in amplitude_columns.items()}
    return amplitude, phase_columns


def _camera_field_amplitude(sample_outputs):
    real = sample_outputs.get("camera_field_real")
    imag = sample_outputs.get("camera_field_imag")
    if real is None or imag is None:
        return None
    return real.detach().float().square().add(imag.detach().float().square()).sqrt()


def _camera_field_phase(sample_outputs):
    real = sample_outputs.get("camera_field_real")
    imag = sample_outputs.get("camera_field_imag")
    if real is None or imag is None:
        return None
    return imag.detach().float().atan2(real.detach().float())


def _build_field_amplitude_columns(
    sample_outputs, encoder_layer_amplitude, decoder_layer_amplitude
):
    amplitude = {
        "SLM command (amplitude control)": _raw_amplitude_for_visualization(
            sample_outputs["slm_command"]
        ),
        "SLM out |E|": _robust_amplitude_display(sample_outputs["slm_amplitude"]),
        "encoder in |E|": _robust_amplitude_display(
            sample_outputs["encoder_input_amplitude"]
        ),
        "encoder grid in |E|": _robust_amplitude_display(
            sample_outputs["encoder_input_grid_amplitude"]
        ),
        **encoder_layer_amplitude,
        "encoder out |E|": _robust_amplitude_display(
            sample_outputs["encoded_amplitude"]
        ),
        "OLS in |E|": _robust_amplitude_display(sample_outputs["ols_input_amplitude"]),
        "OLS out |E|": _robust_amplitude_display(
            sample_outputs["ols_output_amplitude"]
        ),
        "decoder in |E|": _robust_amplitude_display(
            sample_outputs["decoder_input_amplitude"]
        ),
        **decoder_layer_amplitude,
        "decoder out |E|": _robust_amplitude_display(
            sample_outputs["decoder_output_amplitude"]
        ),
        "chip |E|": _robust_amplitude_display(sample_outputs["chip_amplitude"]),
    }
    if bool(sample_outputs.get("camera_field_available", False)):
        camera_amplitude = _camera_field_amplitude(sample_outputs)
        if camera_amplitude is not None:
            amplitude["camera field |E|"] = _robust_amplitude_display(camera_amplitude)
    return amplitude


def _build_field_phase_columns(
    sample_outputs, encoder_layer_phase, decoder_layer_phase
):
    phase = {}

    def add(key: str, label: str) -> None:
        if key in sample_outputs:
            phase[label] = _phase_for_visualization(sample_outputs[key])

    add("slm_phase", "SLM out phase")
    add("encoder_input_phase", "encoder in phase")
    add("encoder_input_grid_phase", "encoder grid in phase")
    phase.update(encoder_layer_phase)
    add("encoded_phase", "encoder out phase")
    add("ols_input_phase", "OLS in phase")
    add("ols_output_phase", "OLS out phase")
    add("decoder_input_phase", "decoder in phase")
    phase.update(decoder_layer_phase)
    add("decoder_output_phase", "decoder out phase")
    add("chip_phase", "chip phase")
    if bool(sample_outputs.get("camera_phase_available", False)):
        camera_phase = _camera_field_phase(sample_outputs)
        if camera_phase is not None:
            phase["camera field phase"] = _phase_for_visualization(camera_phase)
    return phase


def build_one_shot_visualization_tensors(
    reference_images, sample_outputs, codes=None, original_images=None
):
    """3-way split: amplitude=|E|, intensity=|E|²/camera, phase=arg(E)."""
    camera_raw = sample_outputs["camera_plane"]
    encoder_layer_amplitude, encoder_layer_phase = _field_layer_amplitude_columns(
        sample_outputs,
        (("diffractive_encoder_layer", "encoder field after"),),
    )
    decoder_layer_amplitude, decoder_layer_phase = _field_layer_amplitude_columns(
        sample_outputs,
        (("diffractive_decoder_layer", "decoder field after"),),
    )
    amplitude = _build_field_amplitude_columns(
        sample_outputs,
        encoder_layer_amplitude,
        decoder_layer_amplitude,
    )
    intensity = {
        "reference": reference_images,
        "SLM command (amplitude control)": _raw_amplitude_for_visualization(
            sample_outputs["slm_command"]
        ),
        "SLM amplitude grid (control)": _block_grid_preview(
            sample_outputs["slm_amplitude"]
        ),
        "SLM out I": _intensity_for_visualization(sample_outputs["slm_amplitude"]),
        "encoder in I": _intensity_for_visualization(
            sample_outputs["encoder_input_amplitude"]
        ),
        "encoder grid in I": _intensity_for_visualization(
            sample_outputs["encoder_input_grid_amplitude"]
        ),
        "encoder out I": _intensity_for_visualization(
            sample_outputs["encoded_amplitude"]
        ),
        "ols in I": _intensity_for_visualization(sample_outputs["ols_input_amplitude"]),
        "ols out I": _intensity_for_visualization(
            sample_outputs["ols_output_amplitude"]
        ),
        "decoder in I": _intensity_for_visualization(
            sample_outputs["decoder_input_amplitude"]
        ),
        "decoder out I": _intensity_for_visualization(
            sample_outputs["decoder_output_amplitude"]
        ),
        "chip I": _intensity_for_visualization(sample_outputs["chip_amplitude"]),
        "camera intensity": enhance_camera_for_display(camera_raw),
    }
    if "generator_desired_intensity" in sample_outputs:
        intensity = {
            "reference": intensity.pop("reference"),
            "generator desired intensity": sample_outputs["generator_desired_intensity"]
            .detach()
            .float(),
            **intensity,
        }
    if original_images is not None:
        intensity = {"digit": original_images, **intensity}
    if codes is not None:
        intensity = {"noise code": _code_for_visualization(codes), **intensity}
    phase = _build_field_phase_columns(
        sample_outputs,
        encoder_layer_phase,
        decoder_layer_phase,
    )
    return {
        VISUALIZATION_AMPLITUDE_KEY: amplitude,
        VISUALIZATION_INTENSITY_KEY: intensity,
        VISUALIZATION_PHASE_KEY: phase,
    }


def build_one_shot_eval_visualization_tensors(
    teacher_images, sample_outputs, abs_error, codes=None, original_images=None
):
    base = build_one_shot_visualization_tensors(
        teacher_images, sample_outputs, codes=codes, original_images=original_images
    )
    intensity = dict(base[VISUALIZATION_INTENSITY_KEY])
    teacher_display = intensity.pop("reference")
    student_display = intensity.pop("camera intensity")
    result_intensity = {}
    if original_images is not None:
        result_intensity["digit"] = intensity.pop("digit", None)
    if codes is not None:
        result_intensity["noise code"] = intensity.pop("noise code", None)
    result_intensity = {
        **result_intensity,
        "teacher intensity target": teacher_display,
        **intensity,
        "student camera intensity": student_display,
        "intensity abs error": abs_error,
    }
    return {
        VISUALIZATION_AMPLITUDE_KEY: base[VISUALIZATION_AMPLITUDE_KEY],
        VISUALIZATION_INTENSITY_KEY: result_intensity,
        VISUALIZATION_PHASE_KEY: base[VISUALIZATION_PHASE_KEY],
    }


def _shared_intensity_scale(
    tensors, quantile: float = ONE_SHOT_SHARED_SCALE_INTENSITY_QUANTILE
) -> float:
    return _shared_scale(
        (tensor.detach().float().clamp_min(0.0).pow(2) for tensor in tensors),
        quantile=quantile,
    )


def _physical_intensity_scale_inputs(sample_outputs):
    for key, _ in ONE_SHOT_SHARED_SCALE_INTENSITY_SPECS:
        if key in sample_outputs:
            yield sample_outputs[key]
    for key in (
        "diffractive_encoder_layer_amplitude",
        "diffractive_decoder_layer_amplitude",
    ):
        if key not in sample_outputs:
            continue
        tensor = sample_outputs[key]
        if tensor.ndim == 5:
            for layer_index in range(int(tensor.shape[1])):
                yield tensor[:, layer_index]
        elif tensor.ndim == 4:
            for layer_index in range(int(tensor.shape[1])):
                yield tensor[:, layer_index]


def _d2nn_field_layer_intensity_columns(
    sample_outputs, specs, scale: float
) -> dict[str, object]:
    def _ensure_channel(tensor):
        return tensor.unsqueeze(1) if tensor.ndim == 3 else tensor

    columns: dict[str, object] = {}
    for key_prefix, label_prefix in specs:
        amplitude_key = f"{key_prefix}_amplitude"
        if amplitude_key not in sample_outputs:
            continue
        amplitude_tensor = sample_outputs[amplitude_key]
        if amplitude_tensor.ndim == 5:
            layer_count = int(amplitude_tensor.shape[1])
            for layer_index in range(layer_count):
                columns[f"{label_prefix} L{layer_index + 1} I"] = (
                    _scale_intensity_for_visualization(
                        _ensure_channel(amplitude_tensor[:, layer_index]),
                        scale,
                    )
                )
        elif amplitude_tensor.ndim == 4:
            layer_count = int(amplitude_tensor.shape[1])
            for layer_index in range(layer_count):
                columns[f"{label_prefix} L{layer_index + 1} I"] = (
                    _scale_intensity_for_visualization(
                        _ensure_channel(amplitude_tensor[:, layer_index]),
                        scale,
                    )
                )
    return columns


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


def build_one_shot_shared_scale_diagnostics(
    sample_outputs,
    scale: float,
    *,
    amplitude_columns,
    intensity_columns,
    phase_columns,
) -> dict[str, object]:
    slm_command = (
        sample_outputs["slm_command"].detach().float().cpu().flatten(start_dim=1)
    )
    per_sample_p01 = slm_command.quantile(0.01, dim=1)
    per_sample_p99 = slm_command.quantile(0.99, dim=1)
    transform_code = sample_outputs.get("generator_command_transform")
    if hasattr(transform_code, "item"):
        transform_name = (
            "intensity_to_amplitude"
            if int(transform_code.item()) == 1
            else "direct_amplitude"
        )
    else:
        transform_name = str(transform_code or "unknown")
    code_conditioning = sample_outputs.get("generator_code_conditioning_active", True)
    if hasattr(code_conditioning, "item"):
        code_conditioning = bool(code_conditioning.item())
    return {
        "visualization_schema_version": 2,
        "view": "shared_scale_full_chain",
        "amplitude_display_policy": "per_plane_robust_normalization",
        "intensity_scale_quantile": ONE_SHOT_SHARED_SCALE_INTENSITY_QUANTILE,
        "intensity_scale": float(scale),
        "intensity_scale_columns": [
            label for label in intensity_columns if label.endswith(" I")
        ],
        "physical_quantities": {
            "amplitude": "field amplitude |E|; SLM command is labeled as amplitude control",
            "intensity": "field intensity |E|^2 and normalized camera intensity",
            "phase": "field phase arg(E)",
        },
        "columns": {
            "amplitude": list(amplitude_columns),
            "intensity": list(intensity_columns),
            "phase": list(phase_columns),
        },
        "availability": {
            "camera_field_available": bool(
                sample_outputs.get("camera_field_available", False)
            ),
            "camera_phase_available": bool(
                sample_outputs.get("camera_phase_available", False)
            ),
        },
        "generator": {
            "command_transform": transform_name,
            "code_conditioning_active": bool(code_conditioning),
            "desired_intensity_available": "generator_desired_intensity"
            in sample_outputs,
        },
        "slm_command_stats": {
            **_tensor_stats(sample_outputs["slm_command"]),
            "p01_mean": float(per_sample_p01.mean().item()),
            "p99_mean": float(per_sample_p99.mean().item()),
            "p99_minus_p01_mean": float(
                (per_sample_p99 - per_sample_p01).mean().item()
            ),
        },
        "phase_stats": _phase_column_stats(sample_outputs),
    }


def build_one_shot_shared_scale_visualization_tensors(
    reference_images, sample_outputs, codes=None, original_images=None
):
    """3-way shared-scale split: amplitude=|E|, intensity=|E|²/camera, phase=arg(E)."""
    scale = _shared_intensity_scale(_physical_intensity_scale_inputs(sample_outputs))
    encoder_layer_amplitude, encoder_layer_phase = _field_layer_amplitude_columns(
        sample_outputs,
        (("diffractive_encoder_layer", "encoder field after"),),
    )
    decoder_layer_amplitude, decoder_layer_phase = _field_layer_amplitude_columns(
        sample_outputs,
        (("diffractive_decoder_layer", "decoder field after"),),
    )
    encoder_intensity_columns = _d2nn_field_layer_intensity_columns(
        sample_outputs,
        (("diffractive_encoder_layer", "encoder field after"),),
        scale,
    )
    decoder_intensity_columns = _d2nn_field_layer_intensity_columns(
        sample_outputs,
        (("diffractive_decoder_layer", "decoder field after"),),
        scale,
    )
    amplitude = _build_field_amplitude_columns(
        sample_outputs,
        encoder_layer_amplitude,
        decoder_layer_amplitude,
    )
    # -- Intensity: shared-scale |E|² for full chain, camera for sensor --
    intensity = {
        "reference": reference_images,
        "SLM command (amplitude control)": _raw_amplitude_for_visualization(
            sample_outputs["slm_command"]
        ),
        "SLM command contrast (display)": _slm_contrast_for_visualization(
            sample_outputs["slm_command"]
        ),
        "SLM amplitude grid (control)": _block_grid_preview(
            sample_outputs["slm_amplitude"]
        ),
        "SLM out I": _scale_intensity_for_visualization(
            sample_outputs["slm_amplitude"], scale
        ),
        "encoder in I": _scale_intensity_for_visualization(
            sample_outputs["encoder_input_amplitude"], scale
        ),
        "encoder grid in I": _scale_intensity_for_visualization(
            sample_outputs["encoder_input_grid_amplitude"], scale
        ),
        **encoder_intensity_columns,
        "encoder out I": _scale_intensity_for_visualization(
            sample_outputs["encoded_amplitude"], scale
        ),
        "ols in I": _scale_intensity_for_visualization(
            sample_outputs["ols_input_amplitude"], scale
        ),
        "ols out I": _scale_intensity_for_visualization(
            sample_outputs["ols_output_amplitude"], scale
        ),
        "decoder in I": _scale_intensity_for_visualization(
            sample_outputs["decoder_input_amplitude"], scale
        ),
        **decoder_intensity_columns,
        "decoder out I": _scale_intensity_for_visualization(
            sample_outputs["decoder_output_amplitude"], scale
        ),
        "chip I": _scale_intensity_for_visualization(
            sample_outputs["chip_amplitude"], scale
        ),
        "camera intensity": enhance_camera_for_display(sample_outputs["camera_plane"]),
    }
    if "generator_desired_intensity" in sample_outputs:
        intensity = {
            "reference": intensity.pop("reference"),
            "generator desired intensity": sample_outputs["generator_desired_intensity"]
            .detach()
            .float(),
            **intensity,
        }
    if original_images is not None:
        intensity = {"digit": original_images, **intensity}
    if codes is not None:
        intensity = {"noise code": _code_for_visualization(codes), **intensity}
    phase = _build_field_phase_columns(
        sample_outputs,
        encoder_layer_phase,
        decoder_layer_phase,
    )
    return {
        VISUALIZATION_AMPLITUDE_KEY: amplitude,
        VISUALIZATION_INTENSITY_KEY: intensity,
        VISUALIZATION_PHASE_KEY: phase,
        "diagnostics": build_one_shot_shared_scale_diagnostics(
            sample_outputs,
            scale,
            amplitude_columns=amplitude,
            intensity_columns=intensity,
            phase_columns=phase,
        ),
    }


def build_one_shot_eval_shared_scale_visualization_tensors(
    teacher_images, sample_outputs, abs_error, codes=None, original_images=None
):
    base = build_one_shot_shared_scale_visualization_tensors(
        teacher_images, sample_outputs, codes=codes, original_images=original_images
    )
    intensity = dict(base[VISUALIZATION_INTENSITY_KEY])
    teacher_display = intensity.pop("reference")
    student_display = intensity.pop("camera intensity")
    result_intensity = {}
    if original_images is not None:
        result_intensity["digit"] = intensity.pop("digit", None)
    if codes is not None:
        result_intensity["noise code"] = intensity.pop("noise code", None)
    result_intensity = {
        **result_intensity,
        "teacher intensity target": teacher_display,
        **intensity,
        "student camera intensity": student_display,
        "intensity abs error": abs_error,
    }
    diagnostics = dict(base["diagnostics"])
    diagnostics["columns"] = {
        **diagnostics["columns"],
        "intensity": list(result_intensity),
    }
    return {
        VISUALIZATION_AMPLITUDE_KEY: base[VISUALIZATION_AMPLITUDE_KEY],
        VISUALIZATION_INTENSITY_KEY: result_intensity,
        VISUALIZATION_PHASE_KEY: base[VISUALIZATION_PHASE_KEY],
        "diagnostics": diagnostics,
    }


def build_one_shot_display_diagnostic_tensors(sample_outputs):
    tensors = {
        "slm_phase_display": _phase_for_visualization(sample_outputs["slm_phase"]),
        "encoder_input_amplitude_display": _amplitude_for_visualization(
            sample_outputs["encoder_input_amplitude"]
        ),
        "encoder_input_phase_display": _phase_for_visualization(
            sample_outputs["encoder_input_phase"]
        ),
        "encoder_input_grid_amplitude_display": _amplitude_for_visualization(
            sample_outputs["encoder_input_grid_amplitude"]
        ),
        "encoder_input_grid_phase_display": _phase_for_visualization(
            sample_outputs["encoder_input_grid_phase"]
        ),
        "encoded_amplitude_display": _robust_amplitude_display(
            sample_outputs["encoded_amplitude"]
        ),
        "encoded_phase_display": _phase_for_visualization(
            sample_outputs["encoded_phase"]
        ),
        "ols_input_amplitude_display": _robust_amplitude_display(
            sample_outputs["ols_input_amplitude"]
        ),
        "ols_input_phase_display": _phase_for_visualization(
            sample_outputs["ols_input_phase"]
        ),
        "ols_output_amplitude_display": _robust_amplitude_display(
            sample_outputs["ols_output_amplitude"]
        ),
        "ols_output_phase_display": _phase_for_visualization(
            sample_outputs["ols_output_phase"]
        ),
        "ols_amplitude_display": _robust_amplitude_display(
            sample_outputs["ols_amplitude"]
        ),
        "ols_phase_display": _phase_for_visualization(sample_outputs["ols_phase"]),
        "decoder_input_amplitude_display": _robust_amplitude_display(
            sample_outputs["decoder_input_amplitude"]
        ),
        "decoder_input_phase_display": _phase_for_visualization(
            sample_outputs["decoder_input_phase"]
        ),
        "decoder_output_amplitude_display": _robust_amplitude_display(
            sample_outputs["decoder_output_amplitude"]
        ),
        "decoder_output_phase_display": _phase_for_visualization(
            sample_outputs["decoder_output_phase"]
        ),
        "chip_field_amplitude_display": _robust_amplitude_display(
            sample_outputs["chip_amplitude"]
        ),
        "chip_intensity_display": _robust_amplitude_display(
            sample_outputs["chip_amplitude"].detach().float().clamp_min(0.0).pow(2)
        ),
        "chip_phase_display": _phase_for_visualization(sample_outputs["chip_phase"]),
    }
    if "target_code_amplitude" in sample_outputs:
        tensors["target_code_amplitude_display"] = _robust_amplitude_display(
            sample_outputs["target_code_amplitude"]
        )
    return tensors


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
        (
            "ols_output_to_decoder_input_amplitude",
            "ols_output_amplitude",
            "decoder_input_amplitude",
        ),
        (
            "ols_output_to_decoder_input_phase",
            "ols_output_phase",
            "decoder_input_phase",
        ),
        (
            "encoded_to_ols_input_amplitude_display",
            "encoded_amplitude_display",
            "ols_input_amplitude_display",
        ),
        (
            "encoded_to_ols_input_phase_display",
            "encoded_phase_display",
            "ols_input_phase_display",
        ),
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
        (
            "encoder_input",
            "encoder_input_amplitude_display",
            "encoder_input_phase_display",
        ),
        (
            "encoder_input_grid",
            "encoder_input_grid_amplitude_display",
            "encoder_input_grid_phase_display",
        ),
        ("encoder_out_ols_in", "encoded_amplitude_display", "encoded_phase_display"),
        (
            "ols_out_decoder_in",
            "ols_output_amplitude_display",
            "ols_output_phase_display",
        ),
        (
            "decoder_out",
            "decoder_output_amplitude_display",
            "decoder_output_phase_display",
        ),
        ("chip", "chip_field_amplitude_display", "chip_phase_display"),
    )
    spatial_similarity = {
        name: {
            "pearson": _spatial_pearson_summary(
                tensors[amplitude_key], tensors[phase_key]
            ),
            "top25_iou": _top_mask_iou_summary(
                tensors[amplitude_key], tensors[phase_key]
            ),
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
        "target_code_amplitude",
        "decoder_output_amplitude",
        "chip_amplitude",
        "camera_plane",
        "encoded_phase",
        "ols_output_phase",
        "decoder_output_phase",
        "chip_phase",
    )
    tensor_stats = {
        key: _tensor_stats(tensors[key]) for key in stat_keys if key in tensors
    }
    return {
        "view": "compact_full_chain",
        "alias_diffs": alias_diffs,
        "amplitude_phase_similarity": spatial_similarity,
        "tensor_stats": tensor_stats,
    }


def build_one_shot_distill_readability_visualization_tensors(
    teacher_images, sample_outputs, codes=None
):
    """Build readability-focused visualization: teacher readable, student readable, raw student."""
    camera_raw = sample_outputs["camera_plane"]
    teacher_readable = readable_camera_for_display(teacher_images)
    student_readable = readable_camera_for_display(camera_raw)
    student_raw = enhance_camera_for_display(camera_raw)
    intensity = {
        "teacher (readable)": teacher_readable,
        "student (readable)": student_readable,
        "student (raw)": student_raw,
    }
    if codes is not None:
        intensity = {"noise code": _code_for_visualization(codes), **intensity}
    return {
        VISUALIZATION_INTENSITY_KEY: intensity,
        VISUALIZATION_PHASE_KEY: {},
    }


def _signed_unit_for_visualization(tensor):
    raw = tensor.detach().float()
    scale = raw.abs().flatten(start_dim=1).amax(dim=1).clamp_min(1.0e-6)
    view_shape = (raw.shape[0],) + (1,) * (raw.ndim - 1)
    return (0.5 + 0.5 * raw / scale.view(view_shape)).clamp(0.0, 1.0)


def build_one_shot_distill_residual_visualization_tensors(
    reference_images, sample_outputs
):
    """Build residual-probe visualization when template residual diagnostics exist."""
    required = ("camera_plane", "generator_base_command", "generator_residual_map")
    if any(key not in sample_outputs for key in required):
        return None

    student_raw = sample_outputs["camera_plane"].detach().float()
    base_command = sample_outputs["generator_base_command"].detach().float()
    residual_map = sample_outputs["generator_residual_map"].detach().float()
    student_base_diff = student_raw - base_command
    intensity = {
        "reference/prototype": readable_camera_for_display(reference_images),
        "student raw": enhance_camera_for_display(student_raw),
        "generator_base_command": enhance_camera_for_display(base_command),
        "generator_residual_map": _signed_unit_for_visualization(residual_map),
        "abs residual": _normalize_for_visualization(residual_map.abs()),
        "student-base diff": _signed_unit_for_visualization(student_base_diff),
    }
    return {
        VISUALIZATION_INTENSITY_KEY: intensity,
        VISUALIZATION_PHASE_KEY: {},
    }
