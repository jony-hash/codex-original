from __future__ import annotations

import torch

from .d2nn_visualization import (
    D2NN_LAYER_VISUALIZATION_SPECS,
    D2NN_MASK_VISUALIZATION_SPECS,
    _extract_d2nn_field_layer,
    _resolve_d2nn_layer_dim,
)
from .visualization_core import _tensor_diff_summary


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
