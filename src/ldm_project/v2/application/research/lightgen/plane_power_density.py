"""Plane-integrated optical power and intensity-density measurements for research."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch

from ldm_project.v2.contracts import ComplexField, OperationProvenance


METRIC_FIELDS = (
    "power_w",
    "geometric_area_m2",
    "effective_area_m2",
    "mean_intensity_geometric_w_m2",
    "mean_intensity_effective_w_m2",
    "effective_to_geometric_area_ratio",
    "integral_intensity_squared_w2_m2",
)
INTENSITY_QUANTILES = (0.50, 0.90, 0.99)


def plane_geometry(plane: Any) -> dict[str, Any]:
    """Return the physical sampled area, excluding any propagation padding."""
    height, width = plane.shape
    dy, dx = (float(value) for value in plane.pixel_pitch_m)
    if height <= 0 or width <= 0 or not math.isfinite(dy * dx) or dy <= 0.0 or dx <= 0.0:
        raise ValueError("plane must have a positive physical shape and pixel pitch")
    return {
        "plane_id": str(plane.plane_id),
        "coordinate_frame": str(plane.coordinate_frame),
        "shape": [int(height), int(width)],
        "pixel_pitch_m": [dy, dx],
        "origin_m": [float(plane.origin_m[0]), float(plane.origin_m[1])],
        "z_m": float(plane.z_m),
        "geometric_area_m2": float(height * width * dy * dx),
        "measurement_boundary": "declared physical plane/ROI samples only; propagation padding excluded",
    }


def field_plane_metrics(field: ComplexField) -> tuple[np.ndarray, dict[str, Any]]:
    """Return per-image plane integrals and density metrics in SI units."""
    values = field.tensor
    if not torch.is_complex(values) or values.ndim not in (3, 4):
        raise ValueError("plane measurement requires a complex [B,H,W] field")
    if tuple(values.shape[-2:]) != tuple(field.plane.shape):
        raise ValueError("field tensor and physical plane shape differ")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("plane field contains non-finite values")
    dy, dx = (float(value) for value in field.plane.pixel_pitch_m)
    if not math.isfinite(dy * dx) or dy <= 0.0 or dx <= 0.0:
        raise ValueError("plane pixel area must be finite and positive")
    pixel_area = dy * dx
    intensity = values.real.to(torch.float64).square() + values.imag.to(torch.float64).square()
    flat = intensity.reshape(intensity.shape[0], -1)
    power = flat.sum(dim=1) * pixel_area
    intensity_squared_integral = flat.square().sum(dim=1) * pixel_area
    geometric_area = math.prod(field.plane.shape) * pixel_area
    positive = (power > 0.0) & (intensity_squared_integral > 0.0)
    effective_area = torch.full_like(power, float("nan"))
    effective_area[positive] = power[positive].square() / intensity_squared_integral[positive]
    geometric_density = power / geometric_area
    effective_density = torch.full_like(power, float("nan"))
    effective_density[positive] = power[positive] / effective_area[positive]
    area_ratio = torch.full_like(power, float("nan"))
    area_ratio[positive] = effective_area[positive] / geometric_area
    if not bool(torch.isfinite(power).all()) or bool((power < 0.0).any()):
        raise ValueError("plane-integrated power is invalid")
    if bool((effective_area[positive] > geometric_area * (1.0 + 1.0e-10)).any()):
        raise ValueError("effective area exceeds the physical integration plane")
    metrics = torch.stack(
        (
            power,
            torch.full_like(power, geometric_area),
            effective_area,
            geometric_density,
            effective_density,
            area_ratio,
            intensity_squared_integral,
        ),
        dim=1,
    )
    return metrics.detach().cpu().numpy(), plane_geometry(field.plane)


def field_intensity_distribution(field: ComplexField) -> list[dict[str, float | None]]:
    """Return area- and power-weighted intensity quantiles for each image."""
    values = field.tensor
    if not torch.is_complex(values) or values.ndim not in (3, 4):
        raise ValueError("intensity distribution requires a complex [B,H,W] field")
    if tuple(values.shape[-2:]) != tuple(field.plane.shape):
        raise ValueError("field tensor and physical plane shape differ")
    intensity = (values.real.to(torch.float64).square() + values.imag.to(torch.float64).square())
    intensity = intensity.reshape(values.shape[0], -1)
    if not bool(torch.isfinite(intensity).all()) or bool((intensity < 0.0).any()):
        raise ValueError("plane intensity is invalid")
    output: list[dict[str, float | None]] = []
    for row in intensity:
        sorted_values = torch.sort(row).values
        power = row.sum()
        record: dict[str, float | None] = {"maximum_intensity_w_m2": float(row.max().cpu())}
        for quantile in INTENSITY_QUANTILES:
            key = f"q{int(round(quantile * 100)):02d}"
            area_index = min(int(math.ceil(quantile * row.numel())) - 1, row.numel() - 1)
            record[f"area_weighted_{key}_w_m2"] = float(sorted_values[max(area_index, 0)].cpu())
            if float(power) == 0.0:
                record[f"power_weighted_{key}_w_m2"] = None
            else:
                order = torch.argsort(row)
                cumulative = row.index_select(0, order).cumsum(0)
                threshold = power * quantile
                index = int(torch.searchsorted(cumulative, threshold, right=False).clamp_max(row.numel() - 1))
                record[f"power_weighted_{key}_w_m2"] = float(row[order[index]].cpu())
        output.append(record)
    return output


def small_image_result_fields(model: torch.nn.Module, result: Any) -> tuple[list[tuple[str, ComplexField]], dict[str, torch.Tensor]]:
    """Expose the complete current small LightGen image path at physical planes."""
    runtime = getattr(model, "runtime", None)
    if runtime is None or result.image_input is None or result.ols_input_field is None or result.physical_power_trace is None:
        raise RuntimeError("small image-to-OLS forward omitted a required physical field")
    optics = getattr(runtime, "physical_optics", None)
    if optics is not None and getattr(optics, "material", None) is not None:
        raise RuntimeError("plane observer requires a measured field path for the configured material")
    trace = result.physical_power_trace
    source_plane = runtime.slm_plane
    dy, dx = source_plane.pixel_pitch_m
    source_area = math.prod(source_plane.shape) * float(dy) * float(dx)
    amplitude = torch.sqrt(trace["P_input_W"] / source_area).reshape(-1, 1, 1, 1)
    source = ComplexField(
        torch.ones_like(result.image_input.slm_field.tensor)
        * amplitude.to(device=result.image_input.slm_field.tensor.device,
                       dtype=result.image_input.slm_field.tensor.real.dtype),
        source_plane,
        OperationProvenance("uniform_source_before_slm", source_plane.plane_id, None),
    )
    encoder_entry = runtime.input_relay(result.image_input.slm_field)
    camera_field = runtime.output_relay(result.generator_output_field)
    if len(result.encoder_records) != 2 or len(result.generator_records) != 2:
        raise RuntimeError("small architecture layer record count changed")
    fields = [
        ("source_before_slm", source),
        ("slm_output", result.image_input.slm_field),
        ("encoder_entrance", encoder_entry),
        ("encoder_layer_1_aperture", result.encoder_records[0]),
        ("encoder_layer_2_aperture", result.encoder_records[1]),
        ("ols_incident", result.ols_input_field),
        ("fiber_output_and_decoder_entry", result.generator_input_field),
        ("decoder_layer_1_aperture", result.generator_records[0]),
        ("decoder_layer_2_aperture_and_exit", result.generator_records[1]),
        ("camera_physical_roi", camera_field),
    ]
    return fields, trace
