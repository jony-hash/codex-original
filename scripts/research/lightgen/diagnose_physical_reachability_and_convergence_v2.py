"""Synthetic, zero-update C1-C4 physical reachability diagnostics.

This command never loads a dataset/checkpoint or invokes a task evaluator. It
uses the existing LightGen physical encoder, fiber-overlap operator, and camera
producer with synthetic fields only. Phase optimization is diagnostic-only.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any

import torch
import yaml

from ldm_project.v2.application.research.lightgen import lightgen_2d_semantic_generation_input_ols_alignment as alignment
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
    LightGenPhysicalOpticsConfig,
)
from ldm_project.v2.application.research.lightgen.material_nonlinearity import MaterialParameters
from ldm_project.v2.application.research.lightgen.material_nonlinearity_diagnostic import nonlinear_coverage
from ldm_project.v2.application.research.lightgen.physical_power_diagnostics import calibrate_source
from ldm_project.v2.contracts import ComplexField, OperationProvenance
from ldm_project.v2.optics.propagation import clear_production_transfer_cache


REFERENCE_CONFIG = Path("configs/reference/lightgen_power_proxy.yaml")
CAMERA_CONFIG = Path("configs/research/lightgen/physical_reachability_convergence_v2.yaml")
INPUT_NAMES = ("uniform", "vertical_bar", "diagonal", "center_disk")
INPUT_POWER_SCENARIOS_W = (249.3e-6, 1e-3, 10e-3, 100e-3, 300e-3)
I_SAT_SWEEP_W_CM2 = (10.0, 100.0, 520.0, 1.0e3, 1.0e4, 1.0e5, 1.0e6, 1.0e7, 3.0e7)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return value


def _physical_config(
    reference: dict[str, Any],
    *,
    input_power_w: float | None = None,
    layer_transmission: float | None = None,
    global_geometry: bool = True,
    camera_gain: float | None = None,
    material: MaterialParameters | None = None,
) -> LightGenPhysicalOpticsConfig:
    optics, fiber = reference["optics"], reference["fiber"]
    return LightGenPhysicalOpticsConfig(
        input_power_w=float(optics["input_power_w"] if input_power_w is None else input_power_w),
        layer_power_transmission=float(
            optics["layer_power_transmission"] if layer_transmission is None else layer_transmission
        ),
        global_fiber_geometry=global_geometry,
        fiber_pitch_m=float(fiber["pitch_m"]),
        fiber_cladding_diameter_m=float(fiber["cladding_diameter_m"]),
        fiber_mfd_m=float(fiber["mfd_m"]),
        fiber_length_m=float(fiber["length_m"]),
        fiber_attenuation_db_per_km=float(fiber["attenuation_db_per_km"]),
        material=material,
        camera_gain_w_inverse=camera_gain,
    )


def _controls() -> tuple[torch.Tensor, list[str]]:
    axis = torch.linspace(-1.0, 1.0, 28, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    values = torch.stack(
        (
            torch.ones_like(xx),
            ((xx >= -0.18) & (xx <= 0.18)).float(),
            (torch.abs(xx - yy) <= 0.16).float(),
            ((xx.square() + yy.square()) <= 0.32).float(),
        ),
        dim=0,
    )[:, None]
    return values, list(INPUT_NAMES)


def _power(field: ComplexField) -> torch.Tensor:
    dy, dx = field.plane.pixel_pitch_m
    return field.tensor.abs().square().reshape(field.tensor.shape[0], -1).sum(dim=1) * dy * dx


def _relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a64 = a.to(torch.complex128 if torch.is_complex(a) else torch.float64)
    b64 = b.to(torch.complex128 if torch.is_complex(b) else torch.float64)
    return float(torch.linalg.vector_norm(a64 - b64) / torch.linalg.vector_norm(b64).clamp_min(1e-30))


def _best_scalar(source: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Fit one real scalar to all pixels/samples, then report target-normalized L2."""
    a, b = source.detach().to(torch.float64).reshape(-1), target.detach().to(torch.float64).reshape(-1)
    alpha = float(torch.dot(a, b) / torch.dot(a, a).clamp_min(1e-300))
    residual = float(torch.linalg.vector_norm(b - alpha * a) / torch.linalg.vector_norm(b).clamp_min(1e-300))
    return alpha, residual


def _fit_clipped_camera_gain(source_w_per_pixel: torch.Tensor, target_dimensionless: torch.Tensor) -> dict[str, float]:
    """Fit one fixed scalar for the actual clipped sensor mapping on all calibration pixels."""
    source = source_w_per_pixel.detach().to(torch.float64)
    target = target_dimensionless.detach().to(torch.float64)
    linear_gain, linear_residual = _best_scalar(source, target)
    center = math.log10(max(linear_gain, 1e-300))
    lo, hi = center - 8.0, center + 4.0
    golden = (math.sqrt(5.0) - 1.0) / 2.0

    def objective(log_gain: float) -> float:
        gain = 10.0**log_gain
        prediction = (source * gain).clamp(0.0, 1.0)
        return float((prediction - target).square().mean())

    left, right = lo, hi
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1, f2 = objective(x1), objective(x2)
    for _ in range(96):
        if f1 <= f2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = objective(x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = objective(x2)
    log_gain = (left + right) / 2.0
    gain = 10.0**log_gain
    prediction = (source * gain).clamp(0.0, 1.0)
    residual = float(torch.linalg.vector_norm(prediction - target) / torch.linalg.vector_norm(target).clamp_min(1e-300))
    return {
        "gain_w_inverse": gain,
        "clipped_mapping_relative_l2_residual": residual,
        "clipped_mapping_mean_squared_error": objective(log_gain),
        "unclipped_linear_gain_w_inverse": linear_gain,
        "unclipped_linear_relative_l2_residual": linear_residual,
        "log10_gain_search_interval": [lo, hi],
    }


def _phase_alignment(reference: torch.Tensor, test: torch.Tensor) -> tuple[float, torch.Tensor]:
    phi = torch.angle((reference.conj() * test).sum())
    return float(phi), test * torch.exp(torch.complex(torch.zeros_like(phi), -phi))


def _runtime(
    grid_size: int,
    physical: LightGenPhysicalOpticsConfig | None,
    *,
    device: torch.device,
) -> LightGen2DSemanticGenerationInputOLSAlignment:
    return LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(propagation_grid_size=grid_size, physical_optics=physical)
    ).to(device).eval()


def _run_c1_encoder(
    runtime: LightGen2DSemanticGenerationInputOLSAlignment,
    entrance: ComplexField,
) -> dict[str, torch.Tensor | ComplexField]:
    output, _records, stats = runtime.encoder(entrance, "reachability_encoder")
    global_field = ComplexField(
        output.tensor,
        runtime.ols_plane,
        OperationProvenance("reachability_encoder_to_fiber_reference", output.plane.plane_id, None),
    )
    coefficients = runtime.ols.operator.project_coefficients(global_field)
    power_before = _power(global_field)
    power_each = coefficients.abs().square()
    return {
        "field": global_field,
        "coefficients": coefficients,
        "power_before_w": power_before,
        "power_each_w": power_each,
        "eta_each": power_each / power_before[:, None].clamp_min(1e-30),
        "encoder_layer_power": stats["layer_power"],
    }


def _mask_xy(runtime: LightGen2DSemanticGenerationInputOLSAlignment) -> tuple[torch.Tensor, torch.Tensor]:
    n = runtime.encoder.active_shape[0]
    dy, dx = runtime.encoder_entrance_plane.pixel_pitch_m
    y = (torch.arange(n, dtype=torch.float64) - (n - 1) / 2.0) * dy
    x = (torch.arange(n, dtype=torch.float64) - (n - 1) / 2.0) * dx
    return torch.meshgrid(y, x, indexing="ij")


def _fiber_xy(reference: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    pitch = float(reference["fiber"]["pitch_m"])
    axis = (torch.arange(10, dtype=torch.float64) - 4.5) * pitch
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return xx.reshape(-1), yy.reshape(-1)


def _power_stats(values: torch.Tensor) -> dict[str, float]:
    x = values.detach().to(torch.float64).flatten()
    q = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], dtype=x.dtype, device=x.device))
    return {
        "min": float(x.min()),
        "p10": float(q[0]),
        "p50": float(q[1]),
        "mean": float(x.mean()),
        "p90": float(q[2]),
        "max": float(x.max()),
    }


def _kinematic_reachability(reference: dict[str, Any]) -> dict[str, Any]:
    wavelength = float(reference["optics"]["wavelength_m"])
    dx = float(reference["optics"]["sampling_pitch_m"])
    pitch = float(reference["fiber"]["pitch_m"])
    z = float(alignment.LAYER_SPACING_M)
    axis = (torch.arange(10, dtype=torch.float64) - 4.5) * pitch
    qmax = wavelength / (2.0 * dx)
    theta_max = math.asin(qmax)
    rows = []
    grid: list[list[str]] = []
    for r, y in enumerate(axis.tolist()):
        row = []
        for c, x in enumerate(axis.tolist()):
            tx, ty = math.atan(abs(x) / z), math.atan(abs(y) / z)
            qx, qy = abs(math.sin(tx)), abs(math.sin(ty))
            passed = qx <= qmax and qy <= qmax
            status = "PASS" if passed else "FAIL"
            row.append(status)
            rows.append({
                "row": r,
                "column": c,
                "j": 10 * r + c,
                "center_x_m": x,
                "center_y_m": y,
                "theta_x_required_rad": tx,
                "theta_x_required_deg": math.degrees(tx),
                "theta_y_required_rad": ty,
                "theta_y_required_deg": math.degrees(ty),
                "sin_theta_x_required": qx,
                "sin_theta_y_required": qy,
                "sampled_phase_limit_sin_theta": qmax,
                "x_pass": qx <= qmax,
                "y_pass": qy <= qmax,
                "status": status,
            })
        grid.append(row)
    return {
        "wavelength_m": wavelength,
        "dx_m": dx,
        "encoder_physical_size_m": [100 * dx, 100 * dx],
        "encoder_layer_spacings_m": [float(alignment.LAYER_SPACING_M)],
        "encoder_to_fiber_propagation_segments_m": {
            "encoder_mask_1_to_encoder_mask_2": float(alignment.LAYER_SPACING_M),
            "encoder_mask_2_to_fiber_reference_plane": float(alignment.LAYER_SPACING_M),
        },
        "last_encoder_to_fiber_distance_m": z,
        "fiber_pitch_m": pitch,
        "fiber_center_axis_m": axis.tolist(),
        "outermost_fiber_center_x_y_m": {"x": [-4.5 * pitch, 4.5 * pitch], "y": [-4.5 * pitch, 4.5 * pitch]},
        "fiber_array_center_span_m": 9.0 * pitch,
        "fiber_cladding_diameter_m": float(reference["fiber"]["cladding_diameter_m"]),
        "fiber_array_outer_physical_extent_m": 9.0 * pitch + float(reference["fiber"]["cladding_diameter_m"]),
        "fiber_array_half_width_m": (9.0 * pitch + float(reference["fiber"]["cladding_diameter_m"])) / 2.0,
        "sampled_phase_limit_sin_theta": qmax,
        "theta_max_rad": theta_max,
        "theta_max_deg": math.degrees(theta_max),
        "map_10x10": grid,
        "fiber_rows": rows,
        "pass_count": sum(item["status"] == "PASS" for item in rows),
        "fail_count": sum(item["status"] == "FAIL" for item in rows),
        "claim_limit": "sampled-phase spatial-frequency test only; not a coupling-efficiency guarantee",
        "geometry_status": "PROJECT_GEOMETRY_UNMEASURED_REFERENCE_VALUES",
    }


def _architecture_alternatives(kinematics: dict[str, Any]) -> list[dict[str, Any]]:
    qmax = float(kinematics["sampled_phase_limit_sin_theta"])
    xmax = float(kinematics["outermost_fiber_center_x_y_m"]["x"][1])
    z_required = xmax * math.sqrt(1.0 - qmax**2) / qmax
    z_current = float(kinematics["last_encoder_to_fiber_distance_m"])
    max_center_coordinate = z_current * qmax / math.sqrt(1.0 - qmax**2)
    pitch_required = 2.0 * max_center_coordinate / 9.0
    return [
        {
            "alternative": "increase physical trainable aperture",
            "required_change": "add trainable cells while keeping dx=2um, or redesign pitch and cell count together; current 100x100 at 2um is 200um",
            "predicted_advantage": "more spatial degrees of freedom and improved angular focusing resolution; aperture increase alone does not raise sampled-phase Nyquist angle",
            "predicted_cost": "larger modulator, more driven cells, calibration burden and loss of compact footprint",
            "preserves_100x100": False,
            "preserves_2um": True,
            "preserves_compact_2plus2": False,
        },
        {
            "alternative": "increase encoder-to-fiber distance",
            "required_change": f"last-mask-to-fiber distance >= {z_required:.6g} m for the outermost center under the current per-axis steering criterion, versus current {z_current:.6g} m",
            "predicted_advantage": "reduces required steering angles without changing fiber pitch",
            "predicted_cost": "longer optical path, larger footprint, tighter vibration/alignment control and new propagation/convergence checks",
            "preserves_100x100": True,
            "preserves_2um": True,
            "preserves_compact_2plus2": False,
        },
        {
            "alternative": "insert magnifying relay",
            "required_change": "design/calibrate a relay mapping the 200um mask field to the 1.25mm reference array extent; first-order magnification is about 6.25x and must be derived for selected conjugate planes",
            "predicted_advantage": "decouples mask aperture from fiber-array span and controls field placement",
            "predicted_cost": "extra optics, aberration/loss/alignment budget, relay calibration and a changed forward model",
            "preserves_100x100": True,
            "preserves_2um": True,
            "preserves_compact_2plus2": False,
        },
        {
            "alternative": "reduce fiber pitch or use alternative fiber geometry",
            "required_change": f"current per-axis criterion requires pitch <= {pitch_required:.6g} m at this distance; this conflicts with the 125um reference cladding diameter and needs a different geometry/mode bank",
            "predicted_advantage": "brings all centers into the current sampled steering envelope",
            "predicted_cost": "fiber packaging, cross-talk, mode isolation and fabrication constraints; current cladding reference would overlap",
            "preserves_100x100": True,
            "preserves_2um": True,
            "preserves_compact_2plus2": True,
        },
        {
            "alternative": "reduce OLS channel count",
            "required_change": "use at most a centered 4x4 subset at the present 125um pitch and 1.5mm distance for direct sampled-phase reachability; revise the channel interface",
            "predicted_advantage": "retains ports within the current steering envelope without implying unsupported outer-port coverage",
            "predicted_cost": "lower channel capacity and a changed 100-port OLS contract",
            "preserves_100x100": True,
            "preserves_2um": True,
            "preserves_compact_2plus2": True,
        },
        {
            "alternative": "add dedicated fan-out diffractive optic",
            "required_change": "add and calibrate a fan-out optic that splits the field toward the 10x10 fiber centers",
            "predicted_advantage": "creates multiple distributed spots without requiring one phase ramp to steer to every port",
            "predicted_cost": "insertion loss, phase quantization, alignment, cross-talk and new physical validation",
            "preserves_100x100": True,
            "preserves_2um": True,
            "preserves_compact_2plus2": False,
        },
    ]


def _steering_phase(
    runtime: LightGen2DSemanticGenerationInputOLSAlignment,
    x_target_m: float,
    y_target_m: float,
    z_m: float,
    *,
    clip_to_nyquist: bool,
) -> tuple[torch.Tensor, dict[str, float | bool]]:
    yy, xx = _mask_xy(runtime)
    wavelength = runtime.encoder_entrance_plane.wavelength_m
    q_limit = wavelength / (2.0 * runtime.encoder_entrance_plane.pixel_pitch_m[1])
    radius = math.sqrt(x_target_m**2 + y_target_m**2 + z_m**2)
    qx_actual, qy_actual = x_target_m / radius, y_target_m / radius
    qx, qy = qx_actual, qy_actual
    clipped = False
    if clip_to_nyquist:
        qx, qy = min(q_limit, max(-q_limit, qx)), min(q_limit, max(-q_limit, qy))
        clipped = qx != qx_actual or qy != qy_actual
    phase = (2.0 * math.pi / wavelength) * (qx * xx + qy * yy)
    phase = torch.remainder(phase + math.pi, 2.0 * math.pi) - math.pi
    return phase.to(torch.float32), {
        "direction_cosine_x_actual": qx_actual,
        "direction_cosine_y_actual": qy_actual,
        "direction_cosine_x_used": qx,
        "direction_cosine_y_used": qy,
        "phase_nyquist_limit": q_limit,
        "nyquist_clipped": clipped,
    }


def _target_single_pattern(
    reference: dict[str, Any],
    grid_size: int,
    row: int,
    column: int,
    *,
    phase_iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    x_all, y_all = _fiber_xy(reference)
    target = 10 * row + column
    x, y = float(x_all[target]), float(y_all[target])
    distance = float(alignment.LAYER_SPACING_M)
    physical = _physical_config(reference, global_geometry=True)
    runtime = _runtime(grid_size, physical, device=device)
    p0 = float(reference["optics"]["input_power_w"])
    entrance = calibrate_source(
        torch.ones((1, 1, 100, 100), dtype=torch.complex64, device=device),
        runtime.encoder_entrance_plane,
        p0,
    )
    phase, steering = _steering_phase(runtime, x, y, distance, clip_to_nyquist=True)
    with torch.no_grad():
        runtime.encoder.phase.zero_()
        runtime.encoder.phase[1, 0].copy_(phase.to(device))
    initial = _run_c1_encoder(runtime, entrance)
    initial_eta = initial["eta_each"][0].detach().cpu()
    initial_coeff = initial["coefficients"][0].detach().cpu()
    initial_power_before = float(initial["power_before_w"][0])
    initial_total = float(initial_eta.sum())
    initial_target = float(initial_eta[target])
    initial_ct = float(initial_eta.sum() - initial_eta[target])
    phases_best = runtime.encoder.phase.detach().clone()
    best_eta = initial_target
    if phase_iterations > 0:
        optimizer = torch.optim.Adam([runtime.encoder.phase], lr=0.04)
        for _ in range(phase_iterations):
            optimizer.zero_grad(set_to_none=True)
            measured = _run_c1_encoder(runtime, entrance)
            objective = measured["eta_each"][0, target]
            loss = -objective
            objective_value = float(objective.detach())
            if objective_value > best_eta:
                best_eta = objective_value
                phases_best.copy_(runtime.encoder.phase.detach())
            loss.backward()
            optimizer.step()
            del measured
        with torch.no_grad():
            runtime.encoder.phase.copy_(phases_best)
    with torch.no_grad():
        optimized = _run_c1_encoder(runtime, entrance)
    eta = optimized["eta_each"][0].detach().cpu()
    coeff = optimized["coefficients"][0].detach().cpu()
    p_before = float(optimized["power_before_w"][0])
    result = {
        "row": row,
        "column": column,
        "j": target,
        "target_center_x_m": x,
        "target_center_y_m": y,
        "steering": steering,
        "phase_optimization_steps": phase_iterations,
        "optimization_interpretation": "finite-step local achieved value, not a certified global optimum or mathematical upper bound",
        "eta_target_initial": initial_target,
        "eta_target_optimized": float(eta[target]),
        "eta_total_initial": initial_total,
        "eta_total_optimized": float(eta.sum()),
        "target_to_crosstalk_ratio_initial": initial_target / max(initial_ct, 1e-30),
        "target_to_crosstalk_ratio_optimized": float(eta[target]) / max(float(eta.sum() - eta[target]), 1e-30),
        "total_captured_power_w_initial": initial_power_before * initial_total,
        "total_captured_power_w_optimized": p_before * float(eta.sum()),
        "P_before_fiber_w_initial": initial_power_before,
        "P_before_fiber_w_optimized": p_before,
        "all_100_overlap_complex_initial_real_imag": {
            "real": initial_coeff.real.tolist(),
            "imag": initial_coeff.imag.tolist(),
        },
        "all_100_eta_initial": initial_eta.tolist(),
        "all_100_overlap_complex_optimized_real_imag": {
            "real": coeff.real.tolist(),
            "imag": coeff.imag.tolist(),
        },
        "all_100_eta_optimized": eta.tolist(),
    }
    del runtime, entrance, initial, optimized, phases_best
    clear_production_transfer_cache()
    gc.collect()
    return result


def _full_array_optimize(
    reference: dict[str, Any],
    grid_size: int,
    target_distribution: torch.Tensor,
    *,
    iterations: int,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    physical = _physical_config(reference, global_geometry=True)
    runtime = _runtime(grid_size, physical, device=device)
    p0 = float(reference["optics"]["input_power_w"])
    entrance = calibrate_source(
        torch.ones((1, 1, 100, 100), dtype=torch.complex64, device=device),
        runtime.encoder_entrance_plane,
        p0,
    )
    q_target = target_distribution.to(device=device, dtype=torch.float64).reshape(100)
    with torch.no_grad():
        initial = _run_c1_encoder(runtime, entrance)
        initial_eta = initial["eta_each"][0].detach()
        initial_q = initial_eta / initial_eta.sum().clamp_min(1e-30)
        initial_error = float(torch.linalg.vector_norm(initial_q.to(torch.float64) - q_target))
    best_score = initial_error
    best_phase = runtime.encoder.phase.detach().clone()
    optimizer = torch.optim.Adam([runtime.encoder.phase], lr=0.06)
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        measured = _run_c1_encoder(runtime, entrance)
        eta = measured["eta_each"][0]
        q = eta / eta.sum().clamp_min(1e-30)
        loss = (q.to(torch.float64) - q_target).square().sum()
        score = float(torch.sqrt(loss.detach()))
        if score < best_score:
            best_score = score
            best_phase.copy_(runtime.encoder.phase.detach())
        loss.backward()
        optimizer.step()
        del measured
    with torch.no_grad():
        runtime.encoder.phase.copy_(best_phase)
        final = _run_c1_encoder(runtime, entrance)
    eta_each = final["eta_each"][0].detach().cpu()
    p_each = final["power_each_w"][0].detach().cpu()
    p_before = float(final["power_before_w"][0])
    eta_total = float(eta_each.sum())
    actual_distribution = eta_each / eta_each.sum().clamp_min(1e-30)
    error = float(torch.linalg.vector_norm(actual_distribution.to(torch.float64) - q_target.cpu()))
    x_all, y_all = _fiber_xy(reference)
    perimeter = torch.tensor([r == 0 or r == 9 or c == 0 or c == 9 for r in range(10) for c in range(10)])
    corners = torch.tensor([r in (0, 9) and c in (0, 9) for r in range(10) for c in range(10)])
    central = torch.tensor([r in (4, 5) and c in (4, 5) for r in range(10) for c in range(10)])
    edge_mean, center_mean, corner_mean = float(eta_each[perimeter].mean()), float(eta_each[central].mean()), float(eta_each[corners].mean())
    report = {
        "target_distribution": q_target.cpu().tolist(),
        "actual_distribution_of_captured_power": actual_distribution.tolist(),
        "eta_each": eta_each.tolist(),
        "P_i_w": p_each.tolist(),
        "P_before_fiber_w": p_before,
        "P_fiber_total_w": float(p_each.sum()),
        "total_coupling_efficiency": eta_total,
        "per_channel_power_distribution": _power_stats(p_each),
        "per_channel_eta_distribution": _power_stats(eta_each),
        "distribution_l2_error_initial": initial_error,
        "distribution_l2_error_optimized": error,
        "distribution_error_reduction": initial_error - error,
        "coefficient_of_variation_eta": float(eta_each.std(unbiased=False) / eta_each.mean().clamp_min(1e-30)),
        "edge_vs_center_mean_eta_ratio": edge_mean / max(center_mean, 1e-30),
        "corner_vs_center_mean_eta_ratio": corner_mean / max(center_mean, 1e-30),
        "edge_mean_eta": edge_mean,
        "center_four_mean_eta": center_mean,
        "corner_four_mean_eta": corner_mean,
        "phase_optimization_steps": iterations,
        "optimization_interpretation": "finite-step local achieved distribution only; not a certified global optimum or upper bound",
        "optimizer_scope": "both 100x100 encoder phase masks only; synthetic uniform incident field; no data/task/evaluator",
        "optimization_metric_note": "normalized captured-port distribution is used only inside the shape objective; optical fields, powers and reported eta remain absolute and unrenormalized",
        "fiber_centers_x_m": x_all.tolist(),
        "fiber_centers_y_m": y_all.tolist(),
    }
    saved_phase = runtime.encoder.phase.detach().cpu().clone()
    del runtime, entrance, initial, final, best_phase
    clear_production_transfer_cache()
    gc.collect()
    return report, saved_phase


def _c1(reference: dict[str, Any], config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    kinematics = _kinematic_reachability(reference)
    grid_size = int(config["diagnostics"]["propagation_grid_size"])
    anchor_iterations = int(config["diagnostics"]["single_target_optimization_steps"])
    anchors = {
        "nearest_center": (4, 4),
        "top_edge_center": (0, 4),
        "right_edge_center": (4, 9),
        "bottom_edge_center": (9, 5),
        "left_edge_center": (5, 0),
        "top_left_corner": (0, 0),
        "top_right_corner": (0, 9),
        "bottom_right_corner": (9, 9),
        "bottom_left_corner": (9, 0),
    }
    single_initial = [[None for _ in range(10)] for _ in range(10)]
    single_runtime = _runtime(grid_size, _physical_config(reference), device=device)
    single_entrance = calibrate_source(
        torch.ones((1, 1, 100, 100), dtype=torch.complex64, device=device),
        single_runtime.encoder_entrance_plane,
        float(reference["optics"]["input_power_w"]),
    )
    for r in range(10):
        for c in range(10):
            xall, yall = _fiber_xy(reference)
            phase, steering = _steering_phase(
                single_runtime,
                float(xall[10 * r + c]),
                float(yall[10 * r + c]),
                float(alignment.LAYER_SPACING_M),
                clip_to_nyquist=True,
            )
            with torch.no_grad():
                single_runtime.encoder.phase.zero_()
                single_runtime.encoder.phase[1, 0].copy_(phase.to(device))
                measure = _run_c1_encoder(single_runtime, single_entrance)
            eta = measure["eta_each"][0].detach().cpu()
            captured = float(eta.sum())
            target_eta = float(eta[10 * r + c])
            single_initial[r][c] = {
                "target_j": 10 * r + c,
                "target_eta": target_eta,
                "eta_total": captured,
                "target_to_crosstalk_ratio": target_eta / max(captured - target_eta, 1e-30),
                "P_before_fiber_w": float(measure["power_before_w"][0]),
                "total_captured_power_w": float(measure["power_each_w"][0].sum()),
                "steering": steering,
                "all_100_overlap_complex_initial_real_imag": {
                    "real": measure["coefficients"][0].real.detach().cpu().tolist(),
                    "imag": measure["coefficients"][0].imag.detach().cpu().tolist(),
                },
                "all_100_eta_initial": eta.tolist(),
            }
            del measure
    del single_runtime, single_entrance
    clear_production_transfer_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    anchor_results = {}
    optimized_map: list[list[float | None]] = [[None for _ in range(10)] for _ in range(10)]
    for name, (r, c) in anchors.items():
        item = _target_single_pattern(
            reference, grid_size, r, c,
            phase_iterations=anchor_iterations,
            device=device,
        )
        anchor_results[name] = item
        optimized_map[r][c] = item["eta_target_optimized"]
    uniform = torch.full((100,), 1.0 / 100.0, dtype=torch.float64)
    x, y = _fiber_xy(reference)
    radial = x.square() + y.square()
    center_weighted = torch.exp(-radial / (2.0 * (250e-6) ** 2))
    center_weighted /= center_weighted.sum()
    generator = torch.Generator().manual_seed(int(config["diagnostics"]["random_target_seed"]))
    random_target = torch.rand((100,), generator=generator, dtype=torch.float64) + 0.05
    random_target /= random_target.sum()
    iterations = int(config["diagnostics"]["full_array_optimization_steps"])
    full_array = {}
    uniform_phase = None
    for name, target in (("A_uniform_100_channel", uniform), ("B_center_weighted", center_weighted), ("C_random_nonuniform", random_target)):
        report, phase = _full_array_optimize(reference, grid_size, target, iterations=iterations, device=device)
        full_array[name] = report
        if name == "A_uniform_100_channel":
            uniform_phase = phase
    assert uniform_phase is not None
    analytic_eta_map = [[single_initial[r][c]["target_eta"] for c in range(10)] for r in range(10)]
    status_map = kinematics["map_10x10"]
    return {
        "kinematic_reachability": kinematics,
        "single_target_steering": {
            "target_order": "all 100 fibers row-major; phase ramp applied at the last 100x100 encoder mask; upstream mask zero",
            "nyquist_failure_handling": "clip each phase-ramp direction cosine to sampled Nyquist limit, record clipping, do not claim exact steering for clipped targets",
            "all_100_analytic_results": single_initial,
            "analytic_eta_map_10x10": analytic_eta_map,
            "optimized_anchor_results": anchor_results,
            "optimized_anchor_eta_map_10x10_null_elsewhere": optimized_map,
            "phase_optimization_scope": "two encoder phase masks optimized for the nine specified anchor channels only; all-100 steering map is analytic single-mask phase ramp",
            "optimization_interpretation": "finite-step local achieved values; no global-optimum claim",
        },
        "achievable_eta_map": {
            "analytic_phase_ramp_target_eta_10x10": analytic_eta_map,
            "optimized_anchor_target_eta_10x10_null_elsewhere": optimized_map,
            "kinematic_status_map_10x10": status_map,
            "map_units": "target LP01 power divided by total incident power at fiber reference plane",
        },
        "architecture_alternatives_not_implemented": _architecture_alternatives(kinematics),
        "full_array_power_delivery": full_array,
        "uniform_optimized_encoder_phase": uniform_phase,
    }


def _c2(reference: dict[str, Any], device: torch.device) -> dict[str, Any]:
    images, _ = _controls()
    grids = (2048, 3072, 4096)
    results: dict[int, dict[str, Any]] = {}
    for grid in grids:
        physical = _physical_config(reference, global_geometry=True)
        runtime = _runtime(grid, physical, device=device)
        with torch.no_grad():
            result = runtime.forward_from_image(images[0:1].to(device))
        if result.ols_input_field is None or result.physical_power_trace is None:
            raise RuntimeError("C2 physical forward omitted fiber-plane field or power trace")
        field = result.ols_input_field.tensor.detach()[0, 0]
        roi_size = 2048
        y0, x0 = (grid - roi_size) // 2, (grid - roi_size) // 2
        roi = field[y0:y0 + roi_size, x0:x0 + roi_size].cpu()
        coefficients = result.z_ols.coefficients.detach()[0].cpu()
        trace = result.physical_power_trace
        results[grid] = {
            "common_roi_complex": roi,
            "common_roi_intensity": roi.abs().square(),
            "fiber_coefficients": coefficients,
            "eta_each": trace["fiber_coupling_efficiency_each"].detach()[0].cpu(),
            "power_each_w": trace["fiber_coupled_power_W"].detach()[0].cpu(),
            "power_before_fiber_w": float(trace["P_before_fiber_W"][0]),
            "fiber_total_power_w": float(trace["P_fiber_total_W"][0]),
            "camera_total_power_w": float(trace["P_camera_W"][0]),
            "full_field_edge_40um_power_fraction": float(
                (field.abs().square()[0:20].sum() + field.abs().square()[-20:].sum()
                 + field.abs().square()[20:-20, 0:20].sum() + field.abs().square()[20:-20, -20:].sum())
                / field.abs().square().sum().clamp_min(1e-30)
            ),
            "grid_frequency_spacing_cycles_per_m": 1.0 / (grid * float(reference["optics"]["sampling_pitch_m"])),
            "grid_size": grid,
            "window_m": grid * float(reference["optics"]["sampling_pitch_m"]),
        }
        del runtime, result, field
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    comparisons: dict[str, Any] = {}
    for small, large in ((2048, 3072), (2048, 4096), (3072, 4096)):
        a, b = results[small], results[large]
        roi_a, roi_b = a["common_roi_complex"], b["common_roi_complex"]
        phi, aligned_b = _phase_alignment(roi_a, roi_b)
        coeff_a, coeff_b = a["fiber_coefficients"], b["fiber_coefficients"]
        latent_phi, aligned_coeff_b = _phase_alignment(coeff_a, coeff_b)
        p_roi_a, p_roi_b = float(a["common_roi_intensity"].sum()), float(b["common_roi_intensity"].sum())
        comparisons[f"{small}_vs_{large}"] = {
            "common_roi": {
                "shape": [2048, 2048],
                "pixel_pitch_m": float(reference["optics"]["sampling_pitch_m"]),
                "physical_extent_m": [4.096e-3, 4.096e-3],
                "coordinate_crop_indices": {
                    "smaller": [0, 2048, 0, 2048],
                    "larger": [(large - 2048) // 2, (large + 2048) // 2, (large - 2048) // 2, (large + 2048) // 2],
                },
            },
            "global_phase_rad": phi,
            "raw_complex_relative_error": _relative_l2(roi_a, roi_b),
            "complex_relative_error_after_phase_alignment": _relative_l2(roi_a, aligned_b),
            "intensity_relative_error": _relative_l2(a["common_roi_intensity"], b["common_roi_intensity"]),
            "common_roi_power_relative_error": abs(p_roi_a - p_roi_b) / max(abs(p_roi_a), 1e-30),
            "P_before_fiber_relative_error": abs(a["power_before_fiber_w"] - b["power_before_fiber_w"]) / max(abs(a["power_before_fiber_w"]), 1e-30),
            "fiber_total_coupling_absolute_error": abs(a["fiber_total_power_w"] / a["power_before_fiber_w"] - b["fiber_total_power_w"] / b["power_before_fiber_w"]),
            "fiber_total_power_relative_error": abs(a["fiber_total_power_w"] - b["fiber_total_power_w"]) / max(abs(a["fiber_total_power_w"]), 1e-30),
            "per_fiber_power_relative_l2_error": _relative_l2(a["power_each_w"], b["power_each_w"]),
            "per_fiber_power_relative_max_error": float(((a["power_each_w"] - b["power_each_w"]).abs() / b["power_each_w"].abs().clamp_min(1e-30)).max()),
            "per_fiber_power_absolute_error_w_each": (a["power_each_w"] - b["power_each_w"]).abs().tolist(),
            "per_fiber_power_relative_error_each": ((a["power_each_w"] - b["power_each_w"]).abs() / b["power_each_w"].abs().clamp_min(1e-30)).tolist(),
            "per_fiber_complex_latent_global_phase_rad": latent_phi,
            "per_fiber_complex_latent_raw_relative_error": _relative_l2(coeff_a, coeff_b),
            "per_fiber_complex_latent_global_phase_aligned_relative_error": _relative_l2(coeff_a, aligned_coeff_b),
            "per_fiber_complex_latent_absolute_error_after_common_phase_each": (coeff_a - aligned_coeff_b).abs().tolist(),
            "camera_total_power_relative_error": abs(a["camera_total_power_w"] - b["camera_total_power_w"]) / max(abs(a["camera_total_power_w"]), 1e-30),
        }
    phase_tolerance = 1e-4
    aligned_values = [v["complex_relative_error_after_phase_alignment"] for v in comparisons.values()]
    status = "PASS" if max(aligned_values) <= phase_tolerance else "FAIL"
    repeat_runtime = _runtime(2048, _physical_config(reference, global_geometry=True), device=device)
    with torch.no_grad():
        repeat_result = repeat_runtime.forward_from_image(images[0:1].to(device))
    repeat_field = repeat_result.ols_input_field.tensor.detach()[0, 0].cpu()
    ref_field = results[2048]["common_roi_complex"]
    repeat_phi, repeat_aligned = _phase_alignment(ref_field, repeat_field)
    same_grid_repeat = {
        "method": "second independent execution of the same production forward at 2048x2048 on the same synthetic uniform input",
        "global_phase_rad": repeat_phi,
        "raw_complex_relative_error": _relative_l2(ref_field, repeat_field),
        "aligned_complex_relative_error": _relative_l2(ref_field, repeat_aligned),
    }
    del repeat_runtime, repeat_result, repeat_field
    clear_production_transfer_cache()
    gc.collect()
    wavelength = float(reference["optics"]["wavelength_m"])
    dx = float(reference["optics"]["sampling_pitch_m"])
    radial_max_lambda_f = wavelength * math.sqrt(2.0) / (2.0 * dx)
    return {
        "grid_sizes": list(grids),
        "dx_m": dx,
        "window_widths_m": [grid * dx for grid in grids],
        "frequency_spacing_cycles_per_m": {str(grid): results[grid]["grid_frequency_spacing_cycles_per_m"] for grid in grids},
        "per_grid_boundary_40um_power_fraction": {str(grid): results[grid]["full_field_edge_40um_power_fraction"] for grid in grids},
        "comparisons": comparisons,
        "same_grid_repeat": same_grid_repeat,
        "sampled_spectrum": {
            "axis_nyquist_cycles_per_m": 1.0 / (2.0 * dx),
            "maximum_radial_lambda_f": radial_max_lambda_f,
            "sampled_evanescent_components": "none; maximum sampled radial lambda*f is below 1",
        },
        "potential_error_source_evidence": {
            "global_phase_offset": "quantified and removed only with one common phase per field pair",
            "boundary_wrap_around": "40um edge-band fractions are reported; small fractions make wrap-around less supported, not mathematically excluded",
            "evanescent_angular_spectrum_discretization": "sampled spectrum contains no evanescent bins for dx=2um and 532nm",
            "fft_frequency_sampling": "frequency spacing changes with physical window and is reported for each grid",
            "crop_mismatch": "all fields were cropped to identical centered 4.096mm ROI and same dx",
            "floating_point_accumulation": "same-grid repeat is deterministic; cross-grid FFT accumulation remains a candidate if phase-aligned error persists",
        },
        "predeclared_complex_tolerance": phase_tolerance,
        "maximum_aligned_common_roi_complex_relative_error": max(aligned_values),
        "status": status,
        "claim_limit": "one synthetic uniform input and zero phases; no sampled trained-checkpoint phases",
    }


def _run_sensor(runtime: LightGen2DSemanticGenerationInputOLSAlignment, image: torch.Tensor, *, device: torch.device) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        result = runtime.forward_from_image(image.to(device))
    output = {
        "raw_camera": result.raw_camera.tensor.detach().cpu()[0, 0],
        "coefficients": result.z_ols.coefficients.detach().cpu()[0],
    }
    if result.physical_sensor_output_w_per_pixel is not None:
        output["physical_w_per_pixel"] = result.physical_sensor_output_w_per_pixel.detach().cpu()[0, 0]
    if result.task_sensor_output is not None:
        output["task_output"] = result.task_sensor_output.detach().cpu()[0, 0]
    return output


def _shape_scale_residual(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float]:
    t, p = target.to(torch.float64), prediction.to(torch.float64)
    t_sum, p_sum = float(t.sum()), float(p.sum())
    t_shape = t / max(t_sum, 1e-300)
    p_shape = p / max(p_sum, 1e-300)
    return {
        "shape_residual_sum_normalized_metric_only": _relative_l2(p_shape, t_shape),
        "scale_residual_total_power_relative": abs(p_sum - t_sum) / max(abs(t_sum), 1e-300),
        "raw_output_relative_l2": _relative_l2(p, t),
        "metric_only_normalization": "divide by sum solely to measure shape; sensor outputs are not normalized or changed",
    }


def _c3(reference: dict[str, Any], config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    images, names = _controls()
    gain = float(config["camera_gain_w_inverse"])
    by_name: dict[str, dict[str, torch.Tensor]] = {name: {} for name in names}
    specs = {
        "A0": (None, "A0 original dimensionless baseline; compact/local fiber geometry"),
        "A1": (_physical_config(reference, layer_transmission=1.0, global_geometry=False), "physical absolute scaling only; same compact/local geometry as A0"),
        "A2": (_physical_config(reference, layer_transmission=float(reference["optics"]["layer_power_transmission"]), global_geometry=False), "A1 plus uniform transmission at all two encoder and two decoder layers"),
        "A3": (_physical_config(reference, global_geometry=True, camera_gain=gain), "A2 plus global physical fiber coordinates and frozen camera gain"),
        "A4": (_physical_config(reference, global_geometry=True, camera_gain=gain, material=MaterialParameters("identity")), "A3 plus identity material operation"),
    }
    for label, (physical, _meaning) in specs.items():
        runtime = _runtime(2048, physical, device=device)
        for index, name in enumerate(names):
            by_name[name][label] = _run_sensor(runtime, images[index:index + 1], device=device)
        del runtime
        clear_production_transfer_cache()
        gc.collect()
    a0 = torch.stack([by_name[name]["A0"]["raw_camera"] for name in names])
    a1 = torch.stack([by_name[name]["A1"]["physical_w_per_pixel"] for name in names])
    g9a_alpha, g9a_residual = _best_scalar(a0, a1)
    a2 = torch.stack([by_name[name]["A2"]["physical_w_per_pixel"] for name in names])
    g9b_alpha, g9b_residual = _best_scalar(a1, a2)
    a3 = torch.stack([by_name[name]["A3"]["physical_w_per_pixel"] for name in names])
    a4 = torch.stack([by_name[name]["A4"]["physical_w_per_pixel"] for name in names])
    g9c_residual = _relative_l2(a3, a4)
    a2a3_alpha, a2a3_residual = _best_scalar(a2, a3)
    camera_calibration = []
    camera_validation = []
    for index, name in enumerate(names):
        target = by_name[name]["A0"]["raw_camera"]
        a1_task = (by_name[name]["A1"]["physical_w_per_pixel"] * gain).clamp(0.0, 1.0)
        row = _shape_scale_residual(target, a1_task)
        row["sample"] = name
        row["task_output_clipped_pixel_fraction"] = float((a1_task >= 1.0).double().mean())
        row["comparison_scope"] = "A1 physical W/pixel with same compact geometry as A0; fixed gain only"
        if name in ("uniform", "vertical_bar"):
            camera_calibration.append(row)
        else:
            camera_validation.append(row)
    a3_validation = []
    for name in ("diagonal", "center_disk"):
        target = by_name[name]["A0"]["raw_camera"]
        task = by_name[name]["A3"]["task_output"]
        row = _shape_scale_residual(target, task)
        row["sample"] = name
        row["task_output_clipped_pixel_fraction"] = float((task >= 1.0).double().mean())
        row["comparison_scope"] = "A3 global fiber geometry task output; includes physical-model delta"
        a3_validation.append(row)
    status = {
        "G9a": "PASS" if g9a_residual <= 1e-4 else "FAIL",
        "G9b": "PASS" if g9b_residual <= 1e-4 else "FAIL",
        "G9c": "PASS" if g9c_residual <= 1e-4 else "FAIL",
    }
    return {
        "stepwise_G9": {
            "G9a_A0_vs_A1": {
                "alpha_applied_to_A0": g9a_alpha,
                "normalized_residual_l2": g9a_residual,
                "tolerance": 1e-4,
                "status": status["G9a"],
                "A0_units": "legacy dimensionless relative intensity",
                "A1_units": "W/pixel",
                "geometry": "same local 200um compact OLS geometry",
                "interpretation": "absolute scaling invariance test only",
            },
            "G9b_A1_vs_A2": {
                "alpha_applied_to_A1": g9b_alpha,
                "normalized_residual_l2": g9b_residual,
                "tolerance": 1e-4,
                "status": status["G9b"],
                "layer_transmission_A1": 1.0,
                "layer_transmission_A2": float(reference["optics"]["layer_power_transmission"]),
                "interpretation": "uniform attenuation should preserve one common spatial scale in this linear model",
            },
            "A2_vs_A3_physical_model_delta": {
                "best_global_scalar_alpha_A2_to_A3": a2a3_alpha,
                "scale_adjusted_residual_l2": a2a3_residual,
                "raw_relative_l2": _relative_l2(a2, a3),
                "gate": "reported physical-model change; no numerical equivalence requirement",
                "changed_model_property": "A3 uses global 10x10 fiber centers at 125um pitch on the 4.096mm plane; A2 uses compact local OLS geometry",
            },
            "G9c_A3_vs_A4_identity": {
                "strict_relative_l2": g9c_residual,
                "tolerance": 1e-4,
                "status": status["G9c"],
            },
        },
        "camera_interface": {
            "physical_sensor_output": "W/pixel = raw W/m^2 intensity * physical pixel area",
            "task_sensor_output": "clip(camera_gain * physical_sensor_output_w_per_pixel, 0, 1)",
            "camera_gain_proxy_w_inverse": gain,
            "gain_source": "frozen from A0/A1 synthetic calibration subset with geometry held compact; config controlled; not evaluator controlled; not hardware calibration",
            "calibration_subset": ["uniform", "vertical_bar"],
            "validation_subset": ["diagonal", "center_disk"],
            "calibration_residuals": camera_calibration,
            "independent_validation_residuals_same_geometry_A1": camera_validation,
            "independent_validation_residuals_global_geometry_A3": a3_validation,
            "validation_refit_performed": False,
            "per_sample_max_or_l2_normalization_in_sensor_interface": False,
            "fiber_power_renormalization": False,
            "status": "MEASURED_NO_ACCEPTANCE_TOLERANCE",
            "acceptance_tolerance": "none predeclared by request; residuals are reported without inventing a camera-accuracy threshold",
            "claim_limit": "synthetic proxy only; no measured CMOS response, exposure, quantum efficiency or experimental camera gain",
        },
        "status": status,
    }


def _c4(
    reference: dict[str, Any],
    uniform_phase: torch.Tensor,
    grid_size: int,
    device: torch.device,
) -> dict[str, Any]:
    physical = _physical_config(reference, global_geometry=True)
    runtime = _runtime(grid_size, physical, device=device)
    runtime.encoder.phase.data.copy_(uniform_phase.to(device))
    rows = []
    intensity_rows = []
    fiber = reference["fiber"]
    waist = float(fiber["mfd_m"]) / 2.0
    attenuation_db = float(fiber["attenuation_db_per_km"]) * float(fiber["length_m"]) / 1000.0
    attenuation_t = 10.0 ** (-attenuation_db / 10.0)
    for input_power in INPUT_POWER_SCENARIOS_W:
        # Recreate the constant field at encoder entrance for this absolute power.
        entrance = calibrate_source(
            torch.ones((1, 1, 100, 100), dtype=torch.complex64, device=device),
            runtime.encoder_entrance_plane,
            input_power,
        )
        with torch.no_grad():
            measured = _run_c1_encoder(runtime, entrance)
        p_before = float(measured["power_before_w"][0])
        pi = measured["power_each_w"][0].detach().cpu() * attenuation_t
        eta = pi / p_before
        intensity_cm2 = (2.0 * pi / (math.pi * waist**2)) / 1e4
        row = {
            "P_input_at_encoder_entrance_W": input_power,
            "P_before_fiber_W": p_before,
            "P_fiber_total_W": float(pi.sum()),
            "total_coupling_efficiency": float(pi.sum() / p_before),
            "P_i_W_10x10": pi.reshape(10, 10).tolist(),
            "P_i_distribution_W": _power_stats(pi),
            "eta_i_10x10": eta.reshape(10, 10).tolist(),
            "I_i_W_cm2_10x10": intensity_cm2.reshape(10, 10).tolist(),
            "I_i_distribution_W_cm2": _power_stats(intensity_cm2),
            "fiber_attenuation_db": attenuation_db,
            "fiber_output_attenuation_transmission": attenuation_t,
            "intensity_formula": "I_peak_i = 2 P_i / (pi (MFD/2)^2); W/m^2 then /1e4 to W/cm^2",
        }
        rows.append(row)
        for i_sat in I_SAT_SWEEP_W_CM2:
            coverage = nonlinear_coverage(intensity_cm2.reshape(1, 100), i_sat)
            intensity_rows.append({
                "P_input_W": input_power,
                "I_sat_W_cm2": i_sat,
                "linear_fraction": coverage["fraction_linear"],
                "active_fraction": coverage["fraction_nonlinear"],
                "saturated_fraction": coverage["fraction_saturated"],
                "per_channel_ratio_I_over_I_sat": coverage["r_each"][0],
                "distribution": "actual measured per-channel output from phase-optimized uniform-target diagnostic, not mean/100",
            })
        del measured, entrance
        clear_production_transfer_cache()
    del runtime
    clear_production_transfer_cache()
    return {
        "basis": "C1 optimized uniform-target encoder phase; no fiber renormalization; 0dB/km reference attenuation",
        "fiber_parameter_status": "MFD and attenuation are unmeasured reference values, not current hardware measurements",
        "absolute_power_scenarios": rows,
        "nonlinear_coverage_sweep": intensity_rows,
        "classification_boundaries": {
            "linear": "I/I_sat < 0.1",
            "active": "0.1 <= I/I_sat <= 10",
            "saturated": "I/I_sat > 10",
        },
        "claim_limit": "sensitivity study only; not a Te, TaP or CoSi device claim",
    }


def _camera_gain_preflight(reference: dict[str, Any], device: torch.device) -> dict[str, Any]:
    images, _ = _controls()
    a0, a3 = [], []
    for label, physical in (
        ("A0", None),
        ("A1", _physical_config(reference, layer_transmission=1.0, global_geometry=False)),
    ):
        runtime = _runtime(2048, physical, device=device)
        values = []
        for index in (0, 1):
            result = _run_sensor(runtime, images[index:index + 1], device=device)
            values.append(result["raw_camera"] if label == "A0" else result["physical_w_per_pixel"])
        del runtime
        clear_production_transfer_cache()
        if label == "A0":
            a0 = values
        else:
            a3 = values
    source = torch.stack(a3)
    target = torch.stack(a0)
    fitted = _fit_clipped_camera_gain(source, target)
    gain = fitted["gain_w_inverse"]
    predicted = (source * gain).clamp(0.0, 1.0)
    return {
        "schema": "camera_gain_synthetic_preflight.v1",
        "subset": ["uniform", "vertical_bar"],
        "camera_gain_proxy_w_inverse": gain,
        "calibration_residual_relative_l2_after_actual_clip": fitted["clipped_mapping_relative_l2_residual"],
        "unclipped_linear_fit": fitted,
        "fit": "one fixed scalar minimizes residual of the actual clip(gain*W_per_pixel,0,1) mapping across both synthetic calibration patterns and all pixels; A0 and A1 share compact OLS geometry",
        "normalization_or_renormalization": False,
        "suggested_config_value": gain,
        "task_sensor_example_clip_fraction": float((predicted >= 1.0).double().mean()),
        "claim_limit": "simulation calibration parameter only; do not treat as an experimental camera parameter",
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.write_text(json.dumps(_jsonable(value), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--camera-gain-preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    reference = _load_yaml(REFERENCE_CONFIG)
    if args.camera_gain_preflight_only:
        result = _camera_gain_preflight(reference, device)
        _write_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    config = _load_yaml(CAMERA_CONFIG)
    gain = config.get("camera_gain_w_inverse")
    if not isinstance(gain, (float, int)) or not math.isfinite(float(gain)) or float(gain) <= 0:
        raise ValueError("full C1-C4 execution requires a fixed positive camera_gain_w_inverse in the research YAML")
    result: dict[str, Any] = {
        "schema": "physical_reachability_and_convergence.v2",
        "date": "2026-10-13",
        "status": "RUNNING",
        "device": str(device),
        "training_updates": 0,
        "formal_task_training_updates": 0,
        "diagnostic_phase_optimizer_steps": (
            9 * int(config["diagnostics"]["single_target_optimization_steps"])
            + 3 * int(config["diagnostics"]["full_array_optimization_steps"])
        ),
        "training_update_field_definition": "training_updates counts task/model training only; phase-only optical diagnostic optimizer steps are reported separately",
        "dataset_reads": 0,
        "task_evaluator_calls": 0,
        "checkpoint_loaded": False,
        "camera_gain_proxy_w_inverse": float(gain),
        "authorization_boundary": "synthetic diagnostics and phase-only optical optimization only; no formal training, task evaluator, training dataset, loss, or geometry change",
        "camera_gain_preflight_history": [
            {
                "attempt": "diagnostic-only_attempt_01",
                "output": "outputs/20261013_physical_reachability_and_convergence_v2/runs/diagnostic-only_attempt_01/camera_gain_preflight.json",
                "fit_scope": "A0 vs A3, which changes both units and global fiber geometry",
                "residual": 0.9365986306486072,
                "decision": "not used to set camera gain because it confounds scale calibration with the intentional A2-to-A3 geometry delta; preserved as exploratory evidence",
            },
            {
                "attempt": "diagnostic-only_attempt_02",
                "output": "outputs/20261013_physical_reachability_and_convergence_v2/runs/diagnostic-only_attempt_02/camera_gain_preflight.json",
                "fit_scope": "A0 vs A1 same compact geometry; unconstrained linear scalar before task-interface clipping",
                "gain_w_inverse": 160449300.6116074,
                "unclipped_residual": 5.705100665650644e-7,
                "decision": "preserved as exploratory; not used because the shipped task output includes clip(.,0,1)",
            },
            {
                "attempt": "diagnostic-only_attempt_05",
                "output": "outputs/20261013_physical_reachability_and_convergence_v2/runs/diagnostic-only_attempt_05/camera_gain_preflight.json",
                "fit_scope": "A0 vs A1 same compact geometry; optimize one scalar under actual clip(.,0,1) mapping",
                "gain_w_inverse": 160449288.76173845,
                "clipped_calibration_residual": 0.3098718829701307,
                "decision": "used and frozen in the research YAML; simulation proxy only",
            },
        ],
    }
    print("C1: kinematic reachability, per-fiber analytic steering and full-array phase-only diagnostics", flush=True)
    c1 = _c1(reference, config, device)
    result["C1"] = c1
    result["uniform_optimized_encoder_phase"] = c1.pop("uniform_optimized_encoder_phase")
    print("C2: strict complex-field convergence on common physical ROI", flush=True)
    result["C2"] = _c2(reference, device)
    print("C3: scale regressions and fixed camera interface", flush=True)
    result["C3"] = _c3(reference, config, device)
    print("C4: absolute channel-power and nonlinear coverage sweeps", flush=True)
    result["C4"] = _c4(
        reference,
        result["uniform_optimized_encoder_phase"].detach().cpu().to(torch.float32),
        int(config["diagnostics"]["propagation_grid_size"]),
        device,
    )
    del result["uniform_optimized_encoder_phase"]
    statuses = {
        "R1_100_fiber_kinematic_reachability": "PASS" if c1["kinematic_reachability"]["pass_count"] == 100 else "FAIL",
        "R2_100_fiber_achievable_coupling_map": "MEASURED",
        "R3_full_array_target_distribution_feasibility": "MEASURED",
        "R4_edge_center_coupling_balance": "MEASURED",
        "R5_phase_aligned_G3_convergence": result["C2"]["status"],
        "R6_G9a_scale_equivalence": result["C3"]["stepwise_G9"]["G9a_A0_vs_A1"]["status"],
        "R7_G9b_transmission_scale_equivalence": result["C3"]["stepwise_G9"]["G9b_A1_vs_A2"]["status"],
        "R8_G9c_identity_material_equivalence": result["C3"]["stepwise_G9"]["G9c_A3_vs_A4_identity"]["status"],
        "R9_fixed_camera_interface_validation": result["C3"]["camera_interface"]["status"],
    }
    kinematic_pass = c1["kinematic_reachability"]["pass_count"]
    optimized_uniform = c1["full_array_power_delivery"]["A_uniform_100_channel"]
    optimized_eta = torch.tensor(optimized_uniform["eta_each"], dtype=torch.float64)
    center_four_share = float(optimized_eta[[44, 45, 54, 55]].sum() / optimized_eta.sum().clamp_min(1e-30))
    c1["historical_6_47_percent_type_classification"] = {
        "primary_type": "TYPE-B",
        "kinematic_secondary_finding": "TYPE-C under the specified sampled-phase steering criterion",
        "total_coupling_efficiency_uniform_target": float(optimized_uniform["total_coupling_efficiency"]),
        "central_four_share_of_captured_power": center_four_share,
        "edge_center_mean_eta_ratio": float(optimized_uniform["edge_vs_center_mean_eta_ratio"]),
        "corner_center_mean_eta_ratio": float(optimized_uniform["corner_vs_center_mean_eta_ratio"]),
        "reason": "measured captured power is concentrated in the central four; outer-port overlaps are nonzero and this does not prove zero physical field or global impossibility",
    }
    cv = float(optimized_uniform["coefficient_of_variation_eta"])
    # No coupling percentage gate is introduced. Classification reflects the
    # measured kinematic bottleneck, distribution imbalance and unmeasured HW geometry.
    if kinematic_pass < 100 and cv > 1.0:
        architecture = "ARCHITECTURE_MARGINAL"
        rationale = "the sampled-phase kinematic criterion covers only a subset of the reference array and the optimized uniform-target channel distribution remains imbalanced; this is a simulation-geometry conclusion, not a proof of zero outer-port coupling"
    elif result["C2"]["status"] == "FAIL":
        architecture = "INSUFFICIENT_EVIDENCE"
        rationale = "strict phase-aligned convergence remains unresolved at 1e-4, preventing a stable full-chain architecture judgment"
    else:
        architecture = "INSUFFICIENT_EVIDENCE"
        rationale = "the fiber pitch/MFD and optical spacings are reference values rather than measured device geometry, so hardware reachability remains unverified"
    result["training_readiness_gates"] = statuses
    result["architecture_classification"] = {
        "status": architecture,
        "rationale": rationale,
        "no_arbitrary_coupling_threshold_added": True,
        "measured_uniform_target_eta_statistics": _power_stats(optimized_eta),
    }
    result["permitted_next_actions"] = [
        "review the saved synthetic maps and phase-aligned G3 evidence",
        "obtain measured encoder-to-fiber distances, pitch and MFD before mapping the reference-plane result to hardware",
        "if a physical-layout change is later authorized, compare the six listed architecture alternatives without treating this report as implementation approval",
    ]
    result["blocked_actions"] = [
        "formal training or task evaluation",
        "reading a training dataset",
        "loss changes or architecture/geometry modification",
        "camera-gain use as an experimental calibration parameter",
        "material-device claims for Te, TaP or CoSi",
        "hardware deployment or acceptance",
    ]
    result["status"] = "VALID_FAILED" if result["C2"]["status"] == "FAIL" or "FAIL" in statuses.values() else "VALID_DIAGNOSTIC_COMPLETE"
    _write_json(args.output, result)
    print(json.dumps({"status": result["status"], "architecture": architecture, "gates": statuses, "output": str(args.output)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
