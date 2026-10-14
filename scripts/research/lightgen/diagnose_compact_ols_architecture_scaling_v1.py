"""Synthetic phase-only aperture/distance/fiber-pitch scaling study.

Uses the production angular-spectrum propagator and LP01 fiber overlap. It does
not load task data, a checkpoint, a task evaluator, or the formal trainer.
Finite-step phase optimization reports achieved values, not global bounds.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    LightGen2DAlignmentRuntimeConfig,
    LightGen2DSemanticGenerationInputOLSAlignment,
    LightGenPhysicalOpticsConfig,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.propagation import (
    angular_spectrum_propagate,
    clear_production_transfer_cache,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/architecture_sweep/compact_ols_architecture_scaling_v1.yaml"
WAVELENGTH_M = 532.0e-9
PIXEL_PITCH_M = 2.0e-6
OBJECTIVES = ("maximize_total_coupling", "maximize_uniform_distribution", "maximize_effective_dimension")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict) or value.get("schema") != "compact_ols_architecture_scaling.v1":
        raise ValueError("expected compact_ols_architecture_scaling.v1 mapping")
    if value["scope"] != {
        "formal_training": "forbidden",
        "task_evaluator": "forbidden",
        "dataset_read": "forbidden",
        "task_loss": "forbidden",
        "material_nonlinearity": "forbidden",
        "production_geometry_mutation": "forbidden",
    }:
        raise ValueError("the diagnostic-only scope contract was changed")
    optics, fiber, optimization = value["optics"], value["fiber"], value["optimization"]
    if float(optics["wavelength_m"]) != WAVELENGTH_M or float(optics["pixel_pitch_m"]) != PIXEL_PITCH_M:
        raise ValueError("this study is fixed to 532 nm and 2 um sampling")
    if int(optics["propagation_grid_size"]) != int(value["execution"]["required_propagation_grid"]):
        raise ValueError("propagation grid does not match the required production grid")
    if int(optics["encoder_layers"]) != 2 or optimization["phase_only"] is not True:
        raise ValueError("the diagnostic must remain a two-mask phase-only system")
    if tuple(optimization["objective_names"]) != OBJECTIVES:
        raise ValueError("the phase-only objective set was changed")
    if optimization["optimizer"] != "Adam":
        raise ValueError("the diagnostic optimizer is fixed to the configured Adam implementation")
    if int(fiber["rows"]) != 10 or int(fiber["columns"]) != 10:
        raise ValueError("this study is fixed to a 10x10 fiber array")
    return value


def _architecture_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    optics, fiber = config["optics"], config["fiber"]
    values = [
        {
            "aperture_pixels": int(n),
            "aperture_m": int(n) * float(optics["pixel_pitch_m"]),
            "last_distance_m": float(z),
            "fiber_pitch_m": float(pitch),
            "scan_role": "requested_grid",
        }
        for n, z, pitch in itertools.product(
            optics["aperture_pixels"], optics["last_encoder_to_fiber_distance_m"], fiber["pitch_m"]
        )
    ]
    current_z = float(optics["current_reference_distance_m"])
    reference_pitch = float(fiber["pitch_m"][0])
    for n in (100, 200, 300, 400):
        item = {
            "aperture_pixels": n,
            "aperture_m": n * float(optics["pixel_pitch_m"]),
            "last_distance_m": current_z,
            "fiber_pitch_m": reference_pitch,
            "scan_role": "fixed_z_architecture_comparator",
        }
        if item not in values:
            values.append(item)
    return values


def _physical_runtime(
    config: dict[str, Any], pitch_m: float, device: torch.device
) -> LightGen2DSemanticGenerationInputOLSAlignment:
    optics, fiber = config["optics"], config["fiber"]
    physical = LightGenPhysicalOpticsConfig(
        input_power_w=float(optics["input_power_w"]),
        layer_power_transmission=float(optics["layer_power_transmission"]),
        global_fiber_geometry=True,
        fiber_pitch_m=pitch_m,
        fiber_cladding_diameter_m=float(fiber["cladding_diameter_m"]),
        fiber_mfd_m=float(fiber["mfd_m"]),
        fiber_length_m=float(fiber["length_m"]),
        fiber_attenuation_db_per_km=float(fiber["attenuation_db_per_km"]),
    )
    runtime = LightGen2DSemanticGenerationInputOLSAlignment(
        LightGen2DAlignmentRuntimeConfig(
            propagation_grid_size=int(optics["propagation_grid_size"]), physical_optics=physical
        )
    ).to(device)
    return runtime.eval()


def _source_field(runtime: LightGen2DSemanticGenerationInputOLSAlignment, aperture: int, power_w: float) -> ComplexField:
    plane = runtime.encoder_entrance_plane.with_plane(
        f"compact_ols_aperture_{aperture}", shape=(aperture, aperture), coordinate_frame="encoder"
    )
    area = plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1]
    amplitude = math.sqrt(power_w / (aperture * aperture * area))
    values = torch.full(
        (1, 1, aperture, aperture), amplitude, dtype=torch.complex64, device=next(runtime.parameters()).device
    )
    return ComplexField(values, plane, OperationProvenance("synthetic_uniform_aperture_illumination", plane.plane_id, None))


def _propagate_phase_stack(
    runtime: LightGen2DSemanticGenerationInputOLSAlignment,
    entrance: ComplexField,
    phase: torch.Tensor,
    *,
    last_distance_m: float,
    inter_encoder_spacing_m: float,
    layer_power_transmission: float,
) -> ComplexField:
    grid = runtime.config.propagation_grid_size
    active = entrance
    amplitude_transmission = math.sqrt(layer_power_transmission)
    for index in range(2):
        modulation = torch.polar(torch.full_like(phase[index], amplitude_transmission), phase[index])
        modulated = active.tensor * modulation
        pad_y = (grid - modulated.shape[-2]) // 2
        pad_x = (grid - modulated.shape[-1]) // 2
        padded_tensor = torch.nn.functional.pad(
            modulated, (pad_x, grid - modulated.shape[-1] - pad_x, pad_y, grid - modulated.shape[-2] - pad_y)
        )
        padded_plane = active.plane.with_plane(
            f"scaling_layer_{index + 1}_padded", shape=(grid, grid), coordinate_frame="encoder_global"
        )
        padded = ComplexField(
            padded_tensor,
            padded_plane,
            OperationProvenance("finite_crop_center_pad", active.plane.plane_id, None, {"active_shape": list(active.plane.shape)}),
        )
        distance = inter_encoder_spacing_m if index == 0 else last_distance_m
        propagated = angular_spectrum_propagate(padded, distance, f"scaling_layer_{index + 1}_propagated")
        if index == 0:
            start_y, start_x = pad_y, pad_x
            cropped = propagated.tensor[..., start_y : start_y + entrance.plane.shape[0], start_x : start_x + entrance.plane.shape[1]]
            active_plane = active.plane.with_plane(
                "scaling_encoder_layer_2", z_m=propagated.plane.z_m, coordinate_frame="encoder"
            )
            active = ComplexField(
                cropped,
                active_plane,
                OperationProvenance("finite_crop_hard_aperture", propagated.plane.plane_id, distance,
                                    {"active_shape": list(entrance.plane.shape)}),
            )
        else:
            active = ComplexField(
                propagated.tensor,
                propagated.plane.with_plane("scaling_fiber_reference_plane", coordinate_frame="fiber_global"),
                OperationProvenance("finite_mask_global_propagation_output", propagated.plane.plane_id, distance,
                                    {"active_shape": list(entrance.plane.shape)}),
            )
    return active


def _power(field: ComplexField) -> torch.Tensor:
    area = field.plane.pixel_pitch_m[0] * field.plane.pixel_pitch_m[1]
    return field.tensor.abs().square().reshape(field.tensor.shape[0], -1).sum(dim=1) * area


def _power_stats(values: torch.Tensor) -> dict[str, float]:
    x = values.detach().to(torch.float64).flatten()
    quantiles = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], device=x.device, dtype=x.dtype))
    return {"min": float(x.min()), "p10": float(quantiles[0]), "p50": float(quantiles[1]),
            "mean": float(x.mean()), "p90": float(quantiles[2]), "max": float(x.max())}


def _distribution_metrics(
    power_each_w: torch.Tensor, power_before_w: torch.Tensor, input_power_w: float
) -> dict[str, Any]:
    p = power_each_w.detach().to(torch.float64).reshape(100)
    eta = p / power_before_w.detach().to(torch.float64).reshape(()).clamp_min(1.0e-30)
    total = eta.sum()
    neff = total.square() / eta.square().sum().clamp_min(1.0e-30)
    p10, p90 = torch.quantile(p, torch.tensor([0.1, 0.9], device=p.device, dtype=p.dtype))
    corners = p[torch.tensor([0, 9, 90, 99], device=p.device)]
    centers = p[torch.tensor([44, 45, 54, 55], device=p.device)]
    center_mean = centers.mean()
    max_power = p.max().clamp_min(1.0e-30)
    utilization = {
        str(db): float((p > max_power * (10.0 ** (db / 10.0))).to(torch.float64).mean())
        for db in (-40, -30, -20, -10)
    }
    return {
        "eta_total": float(total),
        "P_before_fiber_w": float(power_before_w.detach().to(torch.float64)),
        "P_fiber_total_w": float(p.sum()),
        "system_input_to_fiber_efficiency": float(p.sum()) / input_power_w,
        "N_eff": float(neff),
        "fiber_power_map_w": p.reshape(10, 10).cpu().tolist(),
        "eta_map": eta.reshape(10, 10).cpu().tolist(),
        "per_channel_P_stats_w": _power_stats(p),
        "per_channel_eta_stats": _power_stats(eta),
        "utilization_fraction_by_relative_dB": utilization,
        "edge_retention_corner_mean_over_center_four_mean": float(corners.mean() / center_mean.clamp_min(1.0e-30)),
        "edge_power_mean_w": float(corners.mean()),
        "center_four_power_mean_w": float(center_mean),
        "uniformity_P10_over_P90": float(p10 / p90.clamp_min(1.0e-30)),
        "coefficient_of_variation": float(p.std(unbiased=False) / p.mean().clamp_min(1.0e-30)),
    }


def _objective(name: str, result: dict[str, Any], *, uniform_target: float) -> torch.Tensor:
    eta = torch.as_tensor(result["eta_each_tensor"], dtype=torch.float64)
    if name == "maximize_total_coupling":
        return eta.sum()
    if name == "maximize_uniform_distribution":
        q = eta / eta.sum().clamp_min(1.0e-30)
        return -((q - uniform_target) ** 2).sum()
    if name == "maximize_effective_dimension":
        return eta.sum().square() / eta.square().sum().clamp_min(1.0e-30)
    raise ValueError(f"unknown objective: {name}")


def _measure_for_optimization(
    runtime: LightGen2DSemanticGenerationInputOLSAlignment,
    entrance: ComplexField,
    phase: torch.Tensor,
    *,
    last_distance_m: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    optics = config["optics"]
    output = _propagate_phase_stack(
        runtime, entrance, phase,
        last_distance_m=last_distance_m,
        inter_encoder_spacing_m=float(optics["inter_encoder_spacing_m"]),
        layer_power_transmission=float(optics["layer_power_transmission"]),
    )
    coefficients = runtime.ols.operator.project_coefficients(output)
    eta = coefficients.abs().square()[0] / _power(output)[0].clamp_min(1.0e-30)
    return {"eta_each_tensor": eta, "power_each_tensor": coefficients.abs().square()[0], "power_before_tensor": _power(output)[0]}


def _metrics_from_measurement(
    runtime: LightGen2DSemanticGenerationInputOLSAlignment,
    entrance: ComplexField,
    phase: torch.Tensor,
    *,
    last_distance_m: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    measured = _measure_for_optimization(runtime, entrance, phase, last_distance_m=last_distance_m, config=config)
    return _distribution_metrics(
        measured["power_each_tensor"], measured["power_before_tensor"], float(config["optics"]["input_power_w"])
    )


def _optimize_objective(
    runtime: LightGen2DSemanticGenerationInputOLSAlignment,
    entrance: ComplexField,
    architecture: dict[str, Any],
    objective_name: str,
    config: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    n = int(architecture["aperture_pixels"])
    phase = torch.nn.Parameter(torch.zeros((2, 1, n, n), dtype=torch.float32, device=device))
    before = _metrics_from_measurement(runtime, entrance, phase, last_distance_m=architecture["last_distance_m"], config=config)
    if objective_name == "maximize_total_coupling":
        best_score = before["eta_total"]
    elif objective_name == "maximize_uniform_distribution":
        best_score = -sum((v - 0.01) ** 2 for row in before["eta_map"] for v in row) / max(before["eta_total"] ** 2, 1.0e-30)
    else:
        best_score = before["N_eff"]
    best_phase = phase.detach().clone()
    steps = int(config["optimization"]["steps_per_objective"])
    optimizer = torch.optim.Adam([phase], lr=float(config["optimization"]["learning_rate"]))
    start = time.perf_counter()
    last_grad_norm = 0.0
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        measured = _measure_for_optimization(
            runtime, entrance, phase, last_distance_m=architecture["last_distance_m"], config=config
        )
        score = _objective(
            objective_name,
            measured,
            uniform_target=float(config["optimization"]["uniform_target"]),
        )
        loss = -score
        score_value = float(score.detach())
        if score_value > best_score:
            best_score = score_value
            best_phase.copy_(phase.detach())
        loss.backward()
        last_grad_norm = float(torch.linalg.vector_norm(phase.grad.detach()))
        optimizer.step()
        del measured, score, loss
    with torch.no_grad():
        phase.copy_(best_phase)
        after = _metrics_from_measurement(
            runtime, entrance, phase, last_distance_m=architecture["last_distance_m"], config=config
        )
    if objective_name == "maximize_total_coupling":
        final_score = after["eta_total"]
        initial_score = before["eta_total"]
    elif objective_name == "maximize_uniform_distribution":
        final_score = -sum((v - 0.01) ** 2 for row in after["eta_map"] for v in row) / max(after["eta_total"] ** 2, 1.0e-30)
        initial_score = -sum((v - 0.01) ** 2 for row in before["eta_map"] for v in row) / max(before["eta_total"] ** 2, 1.0e-30)
    else:
        final_score = after["N_eff"]
        initial_score = before["N_eff"]
    report = {
        "optimizer": str(config["optimization"]["optimizer"]),
        "steps": steps,
        "phase_only": True,
        "objective": objective_name,
        "initial": before,
        "optimized": after,
        "objective_score_initial": initial_score,
        "objective_score_optimized": final_score,
        "objective_improvement": final_score - initial_score,
        "wall_time_s": time.perf_counter() - start,
        "final_gradient_l2_norm": last_grad_norm,
        "interpretation": "finite-step local achieved value; not a certified global maximum or an upper bound",
    }
    saved_phase = phase.detach().cpu().clone()
    del optimizer, phase, best_phase
    return report, saved_phase


def _run_architecture(
    runtime: LightGen2DSemanticGenerationInputOLSAlignment,
    architecture: dict[str, Any],
    config: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any]:
    optics = config["optics"]
    entrance = _source_field(runtime, int(architecture["aperture_pixels"]), float(optics["input_power_w"]))
    n = int(architecture["aperture_pixels"])
    phase = torch.zeros((2, 1, n, n), dtype=torch.float32, device=device)
    with torch.no_grad():
        baseline = _metrics_from_measurement(runtime, entrance, phase,
                                             last_distance_m=float(architecture["last_distance_m"]), config=config)
    optimized = {}
    for objective in OBJECTIVES:
        result, _phase = _optimize_objective(runtime, entrance, architecture, objective, config, device=device)
        optimized[objective] = result
        del _phase
    del entrance, phase
    return {**architecture, "baseline": baseline, "optimized_results": optimized}


def _reachability(architecture: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    optics, fiber = config["optics"], config["fiber"]
    wavelength, dx, z, pitch = (float(optics["wavelength_m"]), float(optics["pixel_pitch_m"]),
                                 float(architecture["last_distance_m"]), float(architecture["fiber_pitch_m"]))
    qmax = wavelength / (2.0 * dx)
    theta_max = math.asin(qmax)
    axis = [(index - 4.5) * pitch for index in range(10)]
    rows, matrix = [], []
    for r, y in enumerate(axis):
        row = []
        for c, x in enumerate(axis):
            tx, ty = math.atan(abs(x) / z), math.atan(abs(y) / z)
            qx, qy = abs(math.sin(tx)), abs(math.sin(ty))
            passed = qx <= qmax and qy <= qmax
            row.append("PASS" if passed else "FAIL")
            rows.append({"row": r, "column": c, "j": 10 * r + c, "center_x_m": x, "center_y_m": y,
                         "theta_x_required_deg": math.degrees(tx), "theta_y_required_deg": math.degrees(ty),
                         "sin_theta_x_required": qx, "sin_theta_y_required": qy,
                         "sampled_phase_limit_sin_theta": qmax, "status": "PASS" if passed else "FAIL"})
        matrix.append(row)
    return {
        "theta_max_deg": math.degrees(theta_max),
        "fiber_center_axis_m": axis,
        "fiber_center_span_m": 9.0 * pitch,
        "fiber_array_outer_extent_m": 9.0 * pitch + float(fiber["cladding_diameter_m"]),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "fail_count": sum(row["status"] == "FAIL" for row in rows),
        "map_10x10": matrix,
        "fiber_rows": rows,
        "claim_limit": "sampled phase spatial-frequency necessary condition only; not a coupling guarantee",
    }


def _phase_mode_preflight(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    reference_pitch = float(config["fiber"]["pitch_m"][0])
    runtime = _physical_runtime(config, reference_pitch, device)
    optics = config["optics"]
    n = 100
    entrance = _source_field(runtime, n, float(optics["input_power_w"]))
    phase = torch.zeros((2, 1, n, n), dtype=torch.float32, device=device)
    physical = runtime.physical_optics
    assert physical is not None
    with torch.no_grad():
        old_encoder, _, _ = runtime.encoder(entrance, "compact_scaling_baseline")
        new_field = _propagate_phase_stack(runtime, entrance, phase,
                                          last_distance_m=float(optics["current_reference_distance_m"]),
                                          inter_encoder_spacing_m=float(optics["inter_encoder_spacing_m"]),
                                          layer_power_transmission=float(optics["layer_power_transmission"]))
        old_norm = torch.linalg.vector_norm(old_encoder.tensor)
        baseline_field_relative_error = float(torch.linalg.vector_norm(new_field.tensor - old_encoder.tensor) / old_norm.clamp_min(1e-30))
        old_coeff = runtime.ols.operator.project_coefficients(old_encoder)
        new_coeff = runtime.ols.operator.project_coefficients(new_field)
        baseline_port_relative_error = float(torch.linalg.vector_norm(new_coeff - old_coeff) / torch.linalg.vector_norm(old_coeff).clamp_min(1e-30))

    # Independent plane-wave eigenfunction check at a well-conditioned distance.
    size = 256
    dx, wavelength, distance = PIXEL_PITCH_M, WAVELENGTH_M, 16.0 * WAVELENGTH_M
    plane = PlaneSpec("asm_plane_wave", (size, size), (dx, dx), wavelength, 0.0, "preflight")
    ky, kx = 5, 7
    fx, fy = kx / (size * dx), ky / (size * dx)
    x = (torch.arange(size, device=device, dtype=torch.float32) - size // 2) * dx
    y = (torch.arange(size, device=device, dtype=torch.float32) - size // 2) * dx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    phase_angle = 2.0 * math.pi * (fx * xx + fy * yy)
    wave = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle))[None, None]
    input_field = ComplexField(wave, plane, OperationProvenance("analytic_plane_wave", plane.plane_id, None))
    actual = angular_spectrum_propagate(input_field, distance, "asm_plane_wave_output").tensor
    kz_phase = 2.0 * math.pi * distance / wavelength * math.sqrt(1.0 - (wavelength * fx) ** 2 - (wavelength * fy) ** 2)
    expected = wave * torch.polar(torch.ones((), dtype=torch.float32, device=device), torch.tensor(kz_phase, device=device))
    analytic_relative_error = float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))
    wrong = angular_spectrum_propagate(input_field, distance + 0.1e-3, "asm_plane_wave_wrong_distance").tensor
    negative_control_error = float(torch.linalg.vector_norm(wrong - expected) / torch.linalg.vector_norm(expected))

    # Separately expose, but do not hide or gate away, accumulated FP32 phase
    # sensitivity at the actual 1.5 mm inter-mask spacing used by this design.
    long_distance = float(optics["inter_encoder_spacing_m"])
    long_actual = angular_spectrum_propagate(input_field, long_distance, "asm_plane_wave_long_distance").tensor
    long_kz_phase = 2.0 * math.pi * long_distance / wavelength * math.sqrt(
        1.0 - (wavelength * fx) ** 2 - (wavelength * fy) ** 2
    )
    long_expected = wave * torch.polar(
        torch.ones((), dtype=torch.float32, device=device), torch.tensor(long_kz_phase, device=device)
    )
    long_distance_relative_error = float(
        torch.linalg.vector_norm(long_actual - long_expected) / torch.linalg.vector_norm(long_expected)
    )

    phase_grad = torch.nn.Parameter(torch.zeros((2, 1, n, n), dtype=torch.float32, device=device))
    measured = _measure_for_optimization(runtime, entrance, phase_grad,
                                          last_distance_m=float(optics["current_reference_distance_m"]), config=config)
    gradient = torch.autograd.grad(measured["eta_each_tensor"].sum(), phase_grad)[0]
    grad_norm = float(torch.linalg.vector_norm(gradient))
    extreme_n = int(optics["aperture_pixels"][-1])
    extreme_distance = float(optics["last_encoder_to_fiber_distance_m"][-1])
    extreme_pitch = float(config["fiber"]["pitch_m"][-1])
    extreme_runtime = _physical_runtime(config, extreme_pitch, device)
    extreme_entrance = _source_field(extreme_runtime, extreme_n, float(optics["input_power_w"]))
    extreme_phase = torch.nn.Parameter(torch.zeros((2, 1, extreme_n, extreme_n), dtype=torch.float32, device=device))
    extreme_measurement = _measure_for_optimization(
        extreme_runtime, extreme_entrance, extreme_phase,
        last_distance_m=extreme_distance, config=config,
    )
    extreme_gradient = torch.autograd.grad(extreme_measurement["eta_each_tensor"].sum(), extreme_phase)[0]
    extreme_grad_norm = float(torch.linalg.vector_norm(extreme_gradient))
    memory = None
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        memory = {"free_bytes_after_preflight": free, "total_bytes": total,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device)}
    checks = {
        "baseline_field_matches_existing_production_encoder": baseline_field_relative_error <= 1.0e-6,
        "baseline_fiber_ports_match": baseline_port_relative_error <= 1.0e-6,
        "production_ASM_matches_analytic_plane_wave": analytic_relative_error <= 1.0e-4,
        "wrong_distance_negative_control_detected": negative_control_error >= 1.0e-3,
        "representative_phase_gradient_finite_nonzero": math.isfinite(grad_norm) and grad_norm > 0.0,
        "largest_aperture_longest_distance_gradient_finite": math.isfinite(extreme_grad_norm),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "INVALID_HARNESS",
        "checks": checks,
        "baseline_field_relative_error": baseline_field_relative_error,
        "baseline_fiber_port_relative_error": baseline_port_relative_error,
        "analytic_plane_wave_relative_error": analytic_relative_error,
        "negative_control_wrong_distance_relative_error": negative_control_error,
        "actual_inter_encoder_distance_m": long_distance,
        "actual_inter_encoder_plane_wave_relative_error": long_distance_relative_error,
        "long_distance_precision_note": "long-distance FP32 complex phase drift is reported as a numerical limitation; the sweep remains a measurement of the current production ASM implementation",
        "representative_phase_gradient_l2_norm": grad_norm,
        "largest_aperture_longest_distance": {
            "aperture_pixels": extreme_n, "aperture_m": extreme_n * float(optics["pixel_pitch_m"]),
            "last_distance_m": extreme_distance, "fiber_pitch_m": extreme_pitch,
            "phase_gradient_l2_norm": extreme_grad_norm,
        },
        "diagnostic_optimizer_steps": 0,
        "model_parameter_updates": 0,
        "device_memory_observation": memory,
        "runtime_fiber_geometry": runtime.ols.mode_identity(),
        "physical_assumptions": {
            "uniform_top_hat_coherent_illumination_expands_with_aperture": True,
            "input_power_w": float(optics["input_power_w"]),
            "layer_power_transmission": physical.layer_power_transmission,
            "inter_encoder_spacing_m": float(optics["inter_encoder_spacing_m"]),
            "current_last_distance_m": float(optics["current_reference_distance_m"]),
        },
        "analytic_reference_scope": "strict analytic check uses a 16-wavelength plane-wave propagation to avoid long-phase FP32 conditioning; the actual 1.5mm production-distance comparison is separately reported above",
    }
    del entrance, phase, phase_grad, measured, gradient, runtime, extreme_entrance, extreme_phase
    del extreme_measurement, extreme_gradient, extreme_runtime
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for row in rows:
        for objective, value in row["optimized_results"].items():
            metrics = value["optimized"]
            points.append({"aperture_pixels": row["aperture_pixels"], "aperture_m": row["aperture_m"],
                           "last_distance_m": row["last_distance_m"], "fiber_pitch_m": row["fiber_pitch_m"],
                           "objective": objective, "eta_total": metrics["eta_total"], "N_eff": metrics["N_eff"],
                           "edge_retention": metrics["edge_retention_corner_mean_over_center_four_mean"]})
    frontier = []
    for point in points:
        dominated = any(
            other is not point
            and other["N_eff"] >= point["N_eff"]
            and other["eta_total"] >= point["eta_total"]
            and other["aperture_m"] <= point["aperture_m"]
            and (other["N_eff"] > point["N_eff"] or other["eta_total"] > point["eta_total"] or other["aperture_m"] < point["aperture_m"])
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda item: (item["aperture_m"], -item["N_eff"], -item["eta_total"]))


def _good_screen_candidates(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    screen = config["screen"]
    candidates = []
    for row in rows:
        for objective, result in row["optimized_results"].items():
            metric = result["optimized"]
            if (metric["N_eff"] >= float(screen["effective_dimension_min"])
                    and metric["eta_total"] >= float(screen["total_coupling_efficiency_min"])
                    and metric["edge_retention_corner_mean_over_center_four_mean"] >= float(screen["edge_retention_min"])):
                candidates.append({**{k: row[k] for k in ("aperture_pixels", "aperture_m", "last_distance_m", "fiber_pitch_m", "scan_role")},
                                   "objective": objective, "metrics": metric})
    return sorted(candidates, key=lambda item: (
        item["aperture_pixels"], item["fiber_pitch_m"] != 125e-6, item["last_distance_m"],
        -item["metrics"]["N_eff"], -item["metrics"]["eta_total"]
    ))


def _ranking_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for letter, aperture in zip("ABCD", (100, 200, 300, 400), strict=True):
        matches = [row for row in rows if row["aperture_pixels"] == aperture and row["last_distance_m"] == 1.5e-3 and row["fiber_pitch_m"] == 125e-6]
        if len(matches) != 1:
            raise ValueError(f"missing unique fixed-z comparator for architecture {letter}")
        row = matches[0]
        metric = row["optimized_results"]["maximize_effective_dimension"]["optimized"]
        ranked.append({"architecture": letter, "aperture_pixels": aperture, "aperture_m": aperture * PIXEL_PITCH_M,
                       "z_m": 1.5e-3, "fiber_pitch_m": 125e-6, "eta_total": metric["eta_total"],
                       "N_eff": metric["N_eff"], "utilization": metric["utilization_fraction_by_relative_dB"],
                       "edge_retention": metric["edge_retention_corner_mean_over_center_four_mean"],
                       "uniformity_P10_over_P90": metric["uniformity_P10_over_P90"],
                       "coefficient_of_variation": metric["coefficient_of_variation"],
                       "optimization_difficulty": row["optimized_results"]["maximize_effective_dimension"]["wall_time_s"],
                       "optimization_objective": "maximize_effective_dimension"})
    return ranked


def _nonlinear_sweep(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate["metrics"]
    base_power = torch.tensor(metrics["fiber_power_map_w"], dtype=torch.float64)
    waist = float(config["fiber"]["mfd_m"]) / 2.0
    active_lo, active_hi = map(float, config["nonlinearity"]["active_ratio_open_interval"])
    input_rows = []
    for input_power in config["nonlinearity"]["input_power_w"]:
        input_power = float(input_power)
        channel_power = base_power * (input_power / float(config["optics"]["input_power_w"]))
        intensity_w_cm2 = 2.0 * channel_power / (math.pi * waist**2) / 1.0e4
        power_distribution = _power_stats(channel_power)
        intensity_distribution = _power_stats(intensity_w_cm2)
        coverage = []
        for i_sat in config["nonlinearity"]["i_sat_w_cm2"]:
            ratio = intensity_w_cm2 / float(i_sat)
            coverage.append({
                "I_sat_w_cm2": float(i_sat),
                "linear_fraction_r_le_0_1": float((ratio <= active_lo).to(torch.float64).mean()),
                "active_fraction_0_1_lt_r_lt_10": float(((ratio > active_lo) & (ratio < active_hi)).to(torch.float64).mean()),
                "saturated_fraction_r_ge_10": float((ratio >= active_hi).to(torch.float64).mean()),
            })
        input_rows.append({"input_power_w": input_power, "P_i_w_10x10": channel_power.tolist(),
                           "I_i_w_cm2_10x10": intensity_w_cm2.tolist(),
                           "P_i_distribution_w": power_distribution, "I_i_distribution_w_cm2": intensity_distribution,
                           "coverage_by_I_sat": coverage})
    return {
        "selected_architecture": {k: candidate[k] for k in ("aperture_pixels", "aperture_m", "last_distance_m", "fiber_pitch_m", "objective")},
        "base_power_source": "phase-only optimized synthetic measurement; all per-channel powers retained without normalization",
        "mfd_m": float(config["fiber"]["mfd_m"]), "peak_intensity_formula": "2*P_i/(pi*(MFD/2)^2)",
        "sensitivity_only_not_a_material_or_device_claim": True,
        "input_power_sweep": input_rows,
    }


def _json_write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite output artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_preflight(config_path: Path, output_path: Path, device: torch.device) -> dict[str, Any]:
    config = _load_config(config_path)
    result = _phase_mode_preflight(config, device)
    _json_write_new(output_path, result)
    return result


def run_study(config_path: Path, output_path: Path, preflight_path: Path, progress_path: Path, device: torch.device) -> dict[str, Any]:
    config = _load_config(config_path)
    preflight = run_preflight(config_path, preflight_path, device)
    if preflight["status"] != "PASS":
        payload = {"schema": "compact_ols_architecture_scaling.v1", "status": "INVALID_HARNESS", "preflight": preflight,
                   "formal_training_updates": 0, "dataset_reads": 0, "task_evaluator_calls": 0,
                   "blocked_actions": ["phase-only architecture sweep", "formal training", "task evaluation"]}
        _json_write_new(output_path, payload)
        return payload
    architectures = _architecture_grid(config)
    runtimes: dict[float, LightGen2DSemanticGenerationInputOLSAlignment] = {}
    results = []
    if progress_path.exists():
        raise FileExistsError(f"refusing to append to an existing progress file: {progress_path}")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("x", encoding="utf-8") as progress:
        for index, architecture in enumerate(architectures, start=1):
            pitch = float(architecture["fiber_pitch_m"])
            if pitch not in runtimes:
                runtimes[pitch] = _physical_runtime(config, pitch, device)
            row = _run_architecture(runtimes[pitch], architecture, config, device=device)
            row["kinematic_reachability"] = _reachability(architecture, config)
            results.append(row)
            progress.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            progress.flush()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if index == 1 or index % 10 == 0 or index == len(architectures):
                print(f"completed {index}/{len(architectures)}: N={architecture['aperture_pixels']}, "
                      f"z={architecture['last_distance_m'] * 1e3:g}mm, pitch={pitch * 1e6:g}um", flush=True)
    candidates = _good_screen_candidates(results, config)
    pareto = _pareto_frontier(results)
    if candidates:
        candidate = candidates[0]
        classification = (
            "SIMULATION_SCREEN_MET_REFERENCE_PITCH"
            if candidate["fiber_pitch_m"] == 125e-6
            else "SIMULATION_SCREEN_MET_SENSITIVITY_PITCH"
        )
        classification_reason = "at least one finite-step synthetic phase pattern passes the declared engineering screen; non-reference pitches remain sensitivity cases"
    else:
        # For nonlinear sensitivity only, choose the measured high-dimensional point with the best throughput tie-break.
        points = []
        for row in results:
            metric = row["optimized_results"]["maximize_effective_dimension"]["optimized"]
            points.append({**{k: row[k] for k in ("aperture_pixels", "aperture_m", "last_distance_m", "fiber_pitch_m", "scan_role")},
                           "objective": "maximize_effective_dimension", "metrics": metric})
        candidate = sorted(points, key=lambda item: (-item["metrics"]["N_eff"], -item["metrics"]["eta_total"], item["aperture_pixels"]))[0]
        classification = "NO_MEASURED_SCREENED_CANDIDATE"
        classification_reason = "no measured finite-step result passed all engineering screens; selected point is only the highest achieved N_eff for sensitivity reporting, not a feasibility claim"
    nonlinear = _nonlinear_sweep(candidate, config)
    ranking = _ranking_table(results)
    result = {
        "schema": "compact_ols_architecture_scaling.v1",
        "date": "2026-10-13",
        "status": "VALID_COMPLETED",
        "device": str(device),
        "preflight": preflight,
        "study_scope": {
            "formal_training_updates": 0, "dataset_reads": 0, "task_evaluator_calls": 0,
            "task_loss_changes": 0, "material_nonlinearity_in_forward": False,
            "production_geometry_changed": False, "diagnostic_phase_optimizer_steps": len(results) * len(OBJECTIVES) * int(config["optimization"]["steps_per_objective"]),
            "optimizer_parameters": "two phase masks only; no amplitude parameter", "illumination_assumption": config["optics"]["illumination"],
        },
        "config": config,
        "sweep_design": {"requested_grid_count": 7 * 8 * 5, "fixed_z_comparator_count": 4,
                         "measured_architecture_count": len(results), "objectives_per_architecture": list(OBJECTIVES),
                         "architecture_grid_definition": "aperture_pixels x last_distance_m x fiber_pitch_m; four fixed-z 1.5mm reference comparators appended"},
        "architectures": results,
        "architecture_ranking_A_to_D_at_1p5mm_125um": ranking,
        "pareto_frontier_Neff_eta_min_aperture": pareto,
        "engineering_screen": {"criteria": config["screen"], "qualifying_solution_count": len(candidates),
                               "qualifying_solutions": candidates, "screen_is_not_physical_acceptance": True},
        "selected_architecture_candidate": candidate,
        "architecture_classification": classification,
        "classification_reason": classification_reason,
        "nonlinearity_sensitivity": nonlinear,
        "questions_answered": {
            "Q1_current_200um_scaling": "see architecture A and its full sweep neighbors; finite-step local optimization does not prove a global limit",
            "Q2_minimum_for_100D": "minimum is supported only if at least one configuration passes all three engineering screens; otherwise no sampled minimum is established",
            "Q3_nonlinearity_architecture": "the selected simulated candidate and power coverage are sensitivity-only, not device claims",
            "Q4_experimental_fabrication_size": "not determined unless independent measured relay/fiber geometry is supplied; outputs remain simulation candidates",
        },
        "permitted_next_actions": ["review Pareto candidates", "measure physical relay, illumination and fiber geometry", "plan a separately authorized high-fidelity validation"],
        "blocked_actions": ["formal training", "task evaluator invocation", "dataset reads", "loss changes", "material device claims", "production geometry mutation", "fabrication acceptance"],
        "final_measurement_source": progress_path.as_posix(),
        "claim_limit": "results use a synthetic uniform coherent top-hat expanded with the selected aperture, reference geometry, finite-step Adam phase optimization, and current production ASM/LP01 overlap; no result certifies global optimality or hardware realizability",
    }
    _json_write_new(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    parser.add_argument("--progress-output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if args.preflight_only:
        print(json.dumps(run_preflight(args.config, args.preflight_output, device), indent=2, ensure_ascii=False))
        return
    print(json.dumps(run_study(args.config, args.output, args.preflight_output, args.progress_output, device),
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
