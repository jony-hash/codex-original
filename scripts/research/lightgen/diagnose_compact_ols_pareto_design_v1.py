"""Synthetic phase-only Pareto study for the physical 10x10 LP01 fiber interface.

This is a bounded optical-interface diagnostic, not task training. It uses the
production angular-spectrum propagation and LP01 overlap and keeps raw channel
powers. Finite-step optimization reports achieved points, not global bounds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from scripts.research.lightgen import diagnose_compact_ols_architecture_scaling_v1 as base


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/architecture_sweep/compact_ols_pareto_design_v1.yaml"
OBJECTIVES = (
    "maximize_total_coupling",
    "maximize_effective_dimension",
    "maximize_uniform_distribution",
)
APERTURES = (100, 150, 200, 250, 300, 400, 500, 600)
DISTANCES_M = (0.5e-3, 1e-3, 2e-3, 3e-3, 5e-3, 8e-3, 10e-3)
PITCHES_M = (125e-6, 100e-6, 75e-6, 62.5e-6, 50e-6)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "compact_ols_pareto_design.v1":
        raise ValueError("expected compact_ols_pareto_design.v1 mapping")
    expected_scope = {
        "formal_training": "forbidden",
        "task_evaluator": "forbidden",
        "dataset_read": "forbidden",
        "task_loss": "forbidden",
        "material_nonlinearity_forward_model": "forbidden",
        "production_geometry_mutation": "forbidden",
    }
    if value.get("scope") != expected_scope:
        raise ValueError("diagnostic-only scope contract was changed")
    optics, fiber, optimization = value["optics"], value["fiber"], value["optimization"]
    if float(optics["wavelength_m"]) != 532e-9 or float(optics["pixel_pitch_m"]) != 2e-6:
        raise ValueError("study is fixed to 532 nm and 2 um sampling")
    if tuple(optics["aperture_pixels"]) != APERTURES:
        raise ValueError("aperture sweep differs from the approved plan")
    if tuple(float(x) for x in optics["last_encoder_to_fiber_distance_m"]) != DISTANCES_M:
        raise ValueError("distance sweep differs from the approved plan")
    if tuple(float(x) for x in fiber["pitch_m"]) != PITCHES_M:
        raise ValueError("fiber-pitch sensitivity differs from the approved plan")
    if int(fiber["rows"]) != 10 or int(fiber["columns"]) != 10:
        raise ValueError("study is fixed to a 10x10 fiber array")
    if int(optics["encoder_layers"]) != 2 or optimization["phase_only"] is not True:
        raise ValueError("only two phase-only encoder masks are permitted")
    if tuple(optimization["objective_names"]) != OBJECTIVES or optimization["optimizer"] != "Adam":
        raise ValueError("optimization objectives or optimizer differ from the approved plan")
    expected_grid = int(value["execution"]["required_propagation_grid"])
    if int(optics["propagation_grid_size"]) != expected_grid or expected_grid != _required_grid_size(value):
        raise ValueError("global propagation grid does not cover the derived sweep support")
    if int(value["execution"]["architecture_count"]) != len(APERTURES) * len(DISTANCES_M) * len(PITCHES_M):
        raise ValueError("architecture count does not match the requested cross-product")
    if int(optimization["steps_per_objective"]) != 24:
        raise ValueError("diagnostic optimizer budget differs from the approved 24 steps")
    return value


def _required_grid_size(config: dict[str, Any]) -> int:
    """Power-of-two grid with 1.5x support margin around aperture plus steering."""
    optics, fiber = config["optics"], config["fiber"]
    dx = float(optics["pixel_pitch_m"])
    wavelength = float(optics["wavelength_m"])
    max_aperture = max(int(n) for n in optics["aperture_pixels"]) * dx
    max_distance = max(float(z) for z in optics["last_encoder_to_fiber_distance_m"])
    max_pitch = max(float(p) for p in fiber["pitch_m"])
    max_sin_theta = wavelength / (2 * dx)
    steering_support = max_distance * max_sin_theta
    physical_support = max(max_aperture + 2 * steering_support, 9 * max_pitch + float(fiber["cladding_diameter_m"]))
    needed_pixels = math.ceil(1.5 * physical_support / dx)
    return 1 << (needed_pixels - 1).bit_length()


def _architecture_grid(config: dict[str, Any]) -> list[dict[str, Any]]:
    optics, fiber = config["optics"], config["fiber"]
    return [
        {
            "aperture_pixels": int(n),
            "aperture_m": int(n) * float(optics["pixel_pitch_m"]),
            "aperture_um": int(n) * float(optics["pixel_pitch_m"]) * 1e6,
            "last_distance_m": float(z),
            "fiber_pitch_m": float(pitch),
            "scan_role": "STANDARD_SM_FIBER_REFERENCE" if float(pitch) == 125e-6 else "NON_STANDARD_GEOMETRY",
        }
        for n, z, pitch in itertools.product(APERTURES, DISTANCES_M, PITCHES_M)
    ]


def _measure(
    runtime: torch.nn.Module,
    entrance: Any,
    phase: torch.Tensor,
    architecture: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    optics = config["optics"]
    field = base._propagate_phase_stack(
        runtime,
        entrance,
        phase,
        last_distance_m=float(architecture["last_distance_m"]),
        inter_encoder_spacing_m=float(optics["inter_encoder_spacing_m"]),
        layer_power_transmission=float(optics["layer_power_transmission"]),
    )
    coeff = runtime.ols.operator.project_coefficients(field)
    power_each = coeff.abs().square()[0].to(torch.float64)
    power_before = base._power(field)[0].to(torch.float64)
    return {"power_each_w": power_each, "power_before_w": power_before}


def _distribution_metrics(
    power_each_w: torch.Tensor,
    power_before_w: torch.Tensor,
    input_power_w: float,
    aperture_m: float,
    pixel_pitch_m: float,
) -> dict[str, Any]:
    p = power_each_w.detach().to(torch.float64).reshape(100)
    total_power = p.sum()
    eta = total_power / float(input_power_w)
    conditional_capture = total_power / power_before_w.detach().to(torch.float64).reshape(()).clamp_min(1e-30)
    neff = total_power.square() / p.square().sum().clamp_min(1e-30)
    p10, p90 = torch.quantile(p, torch.tensor([0.1, 0.9], dtype=p.dtype, device=p.device))
    corners = p[torch.tensor([0, 9, 90, 99], device=p.device)]
    centers = p[torch.tensor([44, 45, 54, 55], device=p.device)]
    center_mean = centers.mean()
    max_power = p.max().clamp_min(1e-30)
    utilization = {
        f"{db}_dB": float((p > max_power * (10.0 ** (db / 10.0))).to(torch.float64).mean())
        for db in (-40, -30, -20, -10)
    }
    area_mm2 = (aperture_m * 1e3) ** 2
    return {
        "eta_total": float(eta),
        "P_input_w": float(input_power_w),
        "P_before_fiber_w": float(power_before_w.detach()),
        "P_fiber_total_w": float(total_power),
        "fiber_capture_fraction_after_encoder": float(conditional_capture),
        "N_eff": float(neff),
        "fiber_power_map_w": p.reshape(10, 10).cpu().tolist(),
        "per_channel_power_w": p.cpu().tolist(),
        "utilization_fraction_by_relative_threshold": utilization,
        "P10_over_P90": float(p10 / p90.clamp_min(1e-30)),
        "coefficient_of_variation": float(p.std(unbiased=False) / p.mean().clamp_min(1e-30)),
        "edge_retention_corner_over_center": float(corners.mean() / center_mean.clamp_min(1e-30)),
        "corner_mean_power_w": float(corners.mean()),
        "center_four_mean_power_w": float(center_mean),
        "aperture_area_mm2": area_mm2,
        "compactness_Neff_per_mm2": float(neff / area_mm2),
        "fiber_power_distribution_w": base._power_stats(p),
    }


def _objective(name: str, metrics: dict[str, torch.Tensor], uniform_target: float, input_power_w: float) -> torch.Tensor:
    p = metrics["power_each_w"]
    total = p.sum().clamp_min(1e-30)
    if name == "maximize_total_coupling":
        return total / float(input_power_w)
    if name == "maximize_effective_dimension":
        return total.square() / p.square().sum().clamp_min(1e-30)
    if name == "maximize_uniform_distribution":
        return -((p / total - uniform_target) ** 2).sum()
    raise ValueError(f"unknown objective: {name}")


def _score_from_metrics(name: str, metrics: dict[str, Any]) -> float:
    if name == "maximize_total_coupling":
        return float(metrics["eta_total"])
    if name == "maximize_effective_dimension":
        return float(metrics["N_eff"])
    if name == "maximize_uniform_distribution":
        values = torch.tensor(metrics["per_channel_power_w"], dtype=torch.float64)
        distribution = values / values.sum().clamp_min(1e-30)
        return -float(((distribution - 0.01) ** 2).sum())
    raise ValueError(f"unknown objective: {name}")


def _optimize_objective(
    runtime: torch.nn.Module,
    entrance: Any,
    architecture: dict[str, Any],
    objective_name: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    n = int(architecture["aperture_pixels"])
    phase = torch.nn.Parameter(torch.zeros((2, 1, n, n), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([phase], lr=float(config["optimization"]["learning_rate"]))
    with torch.no_grad():
        initial_measurement = _measure(runtime, entrance, phase, architecture, config)
        initial = _distribution_metrics(
            initial_measurement["power_each_w"], initial_measurement["power_before_w"],
            float(config["optics"]["input_power_w"]), float(architecture["aperture_m"]),
            float(config["optics"]["pixel_pitch_m"]),
        )
    initial_score = _score_from_metrics(objective_name, initial)
    best_score = initial_score
    best_phase = phase.detach().clone()
    started = time.perf_counter()
    last_grad_norm = 0.0
    last_grad_norm_by_layer = [0.0, 0.0]
    for _ in range(int(config["optimization"]["steps_per_objective"])):
        optimizer.zero_grad(set_to_none=True)
        measurement = _measure(runtime, entrance, phase, architecture, config)
        score = _objective(
            objective_name,
            measurement,
            float(config["optimization"]["uniform_target"]),
            float(config["optics"]["input_power_w"]),
        )
        score_value = float(score.detach())
        if score_value > best_score:
            best_score = score_value
            best_phase.copy_(phase.detach())
        loss = -score
        loss.backward()
        last_grad_norm = float(torch.linalg.vector_norm(phase.grad.detach()))
        last_grad_norm_by_layer = [float(torch.linalg.vector_norm(phase.grad[index].detach())) for index in range(2)]
        optimizer.step()
        del measurement, score, loss
    with torch.no_grad():
        optimizer_phase = phase.detach().clone()
        optimizer_phase_update_l2 = float(torch.linalg.vector_norm(optimizer_phase))
        optimizer_phase_update_by_layer_l2 = [
            float(torch.linalg.vector_norm(optimizer_phase[index])) for index in range(2)
        ]
        final_measurement = _measure(runtime, entrance, phase, architecture, config)
        final_metrics = _distribution_metrics(
            final_measurement["power_each_w"], final_measurement["power_before_w"],
            float(config["optics"]["input_power_w"]), float(architecture["aperture_m"]),
            float(config["optics"]["pixel_pitch_m"]),
        )
        final_score = _score_from_metrics(objective_name, final_metrics)
        if final_score > best_score:
            best_score = final_score
            best_phase.copy_(phase.detach())
        phase.copy_(best_phase)
        best_measurement = _measure(runtime, entrance, phase, architecture, config)
        optimized = _distribution_metrics(
            best_measurement["power_each_w"], best_measurement["power_before_w"],
            float(config["optics"]["input_power_w"]), float(architecture["aperture_m"]),
            float(config["optics"]["pixel_pitch_m"]),
        )
    result = {
        **{key: architecture[key] for key in ("aperture_pixels", "aperture_m", "aperture_um", "last_distance_m", "fiber_pitch_m", "scan_role")},
        "optimization_type": objective_name,
        "initial": initial,
        "optimized": optimized,
        "objective_score_initial": initial_score,
        "objective_score_optimized": _score_from_metrics(objective_name, optimized),
        "objective_improvement": _score_from_metrics(objective_name, optimized) - initial_score,
        "optimizer": config["optimization"]["optimizer"],
        "optimizer_steps": int(config["optimization"]["steps_per_objective"]),
        "phase_only": True,
        "trainable_tensor_shape": [2, 1, n, n],
        "final_gradient_l2_norm": last_grad_norm,
        "final_gradient_l2_norm_by_encoder_layer": last_grad_norm_by_layer,
        "phase_parameter_update_l2_by_layer": optimizer_phase_update_by_layer_l2,
        "phase_parameter_update_l2_total": optimizer_phase_update_l2,
        "wall_time_s": time.perf_counter() - started,
        "interpretation": "finite-step local achieved value; not a certified global maximum or upper bound",
    }
    del optimizer, phase, best_phase
    return result


def _manufacturability(architecture: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    n = int(architecture["aperture_pixels"])
    pitch = float(architecture["fiber_pitch_m"])
    reference_n = int(config["manufacturability"]["reference_aperture_pixels"])
    reference_pitch = float(config["manufacturability"]["standard_fiber_pitch_m"])
    feature_count = 2 * n * n
    area_score = 100.0 * (reference_n / n) ** 2
    feature_score = 100.0 * (2 * reference_n * reference_n) / feature_count
    if pitch == reference_pitch:
        geometry_score, geometry_class = 100.0, "STANDARD_SM_FIBER_REFERENCE"
    elif pitch < float(config["manufacturability"]["nonstandard_pitch_below_m"]):
        geometry_score, geometry_class = 0.0, "CUSTOM_REQUIREMENT_BELOW_75UM"
    else:
        geometry_score, geometry_class = 50.0, "NON_STANDARD_GEOMETRY"
    score = (area_score + feature_score + geometry_score) / 3.0
    return {
        "manufacturability_score": score,
        "score_scale": "relative 0-100 ranking proxy within swept candidates; not a probability or process acceptance",
        "component_scores": {
            "aperture_area_inverse_rank": area_score,
            "two_layer_pixel_feature_inverse_rank": feature_score,
            "fiber_geometry_reference_score": geometry_score,
        },
        "aperture_side_um": n * float(config["optics"]["pixel_pitch_m"]) * 1e6,
        "per_layer_fabrication_area_mm2": (n * float(config["optics"]["pixel_pitch_m"]) * 1e3) ** 2,
        "two_layer_feature_count": feature_count,
        "fiber_geometry_class": geometry_class,
        "fiber_pitch_below_75um_custom_requirement": pitch < 75e-6,
        "printer_aperture_feature_and_area_limits": "UNRATED_NOT_SUPPLIED",
        "claim_limit": "proxy ranks relative size, two-layer pixel count, and standard-fiber geometry; no printer-specific fabrication pass is inferred",
    }


def _level(metrics: dict[str, Any], config: dict[str, Any]) -> str:
    screen = config["screen"]["level_a"]
    n = float(metrics["N_eff"])
    if (
        n >= float(screen["effective_dimension_min"])
        and float(metrics["eta_total"]) >= float(screen["input_to_fiber_efficiency_min"])
        and float(metrics["edge_retention_corner_over_center"]) >= float(screen["edge_retention_min"])
    ):
        return "LEVEL_A"
    lower, upper = config["screen"]["level_b_effective_dimension_open_interval"]
    if float(lower) < n < float(upper):
        return "LEVEL_B"
    if n <= float(config["screen"]["level_c_effective_dimension_max"]):
        return "LEVEL_C"
    if n >= float(screen["effective_dimension_min"]):
        return "LEVEL_A_DIMENSION_ONLY_SCREEN_INCOMPLETE"
    return "LEVEL_BOUNDARY_N50"


def _pareto_frontier(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for point in points:
        dominated = any(
            other is not point
            and other["N_eff"] >= point["N_eff"]
            and other["eta_total"] >= point["eta_total"]
            and other["aperture_m"] <= point["aperture_m"]
            and (
                other["N_eff"] > point["N_eff"]
                or other["eta_total"] > point["eta_total"]
                or other["aperture_m"] < point["aperture_m"]
            )
            for other in points
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda x: (x["aperture_m"], -x["N_eff"], -x["eta_total"]))


def _all_points(results: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    for row in results:
        for result in row["optimization_results"]:
            metric = result["optimized"]
            points.append({
                "aperture_pixels": result["aperture_pixels"],
                "aperture_m": result["aperture_m"],
                "aperture_um": result["aperture_um"],
                "last_distance_m": result["last_distance_m"],
                "fiber_pitch_m": result["fiber_pitch_m"],
                "scan_role": result["scan_role"],
                "optimization_type": result["optimization_type"],
                "eta_total": metric["eta_total"],
                "N_eff": metric["N_eff"],
                "edge_retention": metric["edge_retention_corner_over_center"],
                "compactness": metric["compactness_Neff_per_mm2"],
                "level": _level(metric, config),
            })
    return points


def _curve_tables(points: list[dict[str, Any]]) -> dict[str, Any]:
    standard = [point for point in points if point["fiber_pitch_m"] == 125e-6]
    n_curve, eta_curve = [], []
    for n in APERTURES:
        dim_rows = [p for p in standard if p["aperture_pixels"] == n and p["optimization_type"] == "maximize_effective_dimension"]
        eta_rows = [p for p in standard if p["aperture_pixels"] == n and p["optimization_type"] == "maximize_total_coupling"]
        best_dim = max(dim_rows, key=lambda p: (p["N_eff"], p["eta_total"]))
        best_eta = max(eta_rows, key=lambda p: (p["eta_total"], p["N_eff"]))
        n_curve.append({**best_dim, "envelope_definition": "maximum N_eff over scanned z for maximize_effective_dimension"})
        eta_curve.append({**best_eta, "envelope_definition": "maximum eta over scanned z for maximize_total_coupling"})
    return {"N_eff_by_aperture_standard_pitch": n_curve, "eta_by_aperture_standard_pitch": eta_curve}


def _minimum_aperture_over_dimension_threshold(
    points: list[dict[str, Any]], threshold: float
) -> dict[str, Any] | None:
    eligible = [point for point in points if point["N_eff"] > threshold]
    if not eligible:
        return None
    minimum_pixels = min(int(point["aperture_pixels"]) for point in eligible)
    same_aperture = [point for point in eligible if int(point["aperture_pixels"]) == minimum_pixels]
    return max(same_aperture, key=lambda point: (point["N_eff"], point["eta_total"], -point["last_distance_m"]))


def _nonlinear_sensitivity(candidate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    metrics = candidate["optimized"]
    base_power = torch.tensor(metrics["fiber_power_map_w"], dtype=torch.float64)
    waist = float(config["fiber"]["mfd_m"]) / 2
    lower, upper = map(float, config["nonlinearity_sensitivity"]["active_ratio_open_interval"])
    power_rows = []
    for input_w in map(float, config["nonlinearity_sensitivity"]["input_power_w"]):
        p_i = base_power * (input_w / float(config["optics"]["input_power_w"]))
        intensity = 2 * p_i / (math.pi * waist**2) / 1e4
        coverage = []
        for i_sat in map(float, config["nonlinearity_sensitivity"]["i_sat_w_cm2"]):
            ratio = intensity / i_sat
            coverage.append({
                "I_sat_w_cm2": i_sat,
                "linear_fraction_r_le_0_1": float((ratio <= lower).to(torch.float64).mean()),
                "active_fraction_0_1_lt_r_lt_10": float(((ratio > lower) & (ratio < upper)).to(torch.float64).mean()),
                "saturated_fraction_r_ge_10": float((ratio >= upper).to(torch.float64).mean()),
            })
        power_rows.append({
            "P_input_w": input_w,
            "P_before_fiber_w": metrics["P_before_fiber_w"] * (input_w / float(config["optics"]["input_power_w"])),
            "P_fiber_total_w": float(p_i.sum()),
            "P_i_10x10_w": p_i.tolist(),
            "I_i_10x10_w_cm2": intensity.tolist(),
            "P_i_distribution_w": base._power_stats(p_i),
            "I_i_distribution_w_cm2": base._power_stats(intensity),
            "I_peak_w_cm2": float(intensity.max()),
            "coverage_by_I_sat": coverage,
        })
    return {
        "selected_candidate": {key: candidate[key] for key in ("aperture_pixels", "aperture_m", "last_distance_m", "fiber_pitch_m", "optimization_type")},
        "power_source": "raw optimized per-fiber synthetic powers; scaled linearly by declared input power; no post-fiber renormalization",
        "intensity_formula": "I_i=2*P_i/(pi*(MFD/2)^2)",
        "sensitivity_only_not_a_material_or_device_claim": True,
        "input_power_sweep": power_rows,
    }


def _diagnostic_preflight(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result = base._phase_mode_preflight(config, device)
    representative = {
        "aperture_pixels": 200,
        "aperture_m": 200 * float(config["optics"]["pixel_pitch_m"]),
        "last_distance_m": 5.0e-3,
        "fiber_pitch_m": 125e-6,
        "scan_role": "STANDARD_SM_FIBER_REFERENCE",
    }
    representative_runtime = base._physical_runtime(config, representative["fiber_pitch_m"], device)
    representative_entrance = base._source_field(
        representative_runtime, representative["aperture_pixels"], float(config["optics"]["input_power_w"])
    )
    representative_config = {**config, "optimization": {**config["optimization"], "steps_per_objective": 3}}
    representative_optimizer = _optimize_objective(
        representative_runtime,
        representative_entrance,
        representative,
        "maximize_total_coupling",
        representative_config,
        device,
    )
    extreme = {
        "aperture_pixels": max(APERTURES),
        "aperture_m": max(APERTURES) * float(config["optics"]["pixel_pitch_m"]),
        "last_distance_m": max(DISTANCES_M),
        "fiber_pitch_m": min(PITCHES_M),
        "scan_role": "NON_STANDARD_GEOMETRY",
    }
    runtime = base._physical_runtime(config, extreme["fiber_pitch_m"], device)
    entrance = base._source_field(runtime, extreme["aperture_pixels"], float(config["optics"]["input_power_w"]))
    phase = torch.nn.Parameter(torch.zeros((2, 1, extreme["aperture_pixels"], extreme["aperture_pixels"]), dtype=torch.float32, device=device))
    measured = _measure(runtime, entrance, phase, extreme, config)
    input_score = _objective(
        "maximize_total_coupling", measured, float(config["optimization"]["uniform_target"]),
        float(config["optics"]["input_power_w"]),
    )
    gradient = torch.autograd.grad(input_score, phase)[0]
    gradient_norm = float(torch.linalg.vector_norm(gradient))
    input_power = float(base._power(entrance)[0])
    with torch.no_grad():
        wrong_distance = {**extreme, "last_distance_m": min(DISTANCES_M)}
        negative = _measure(runtime, entrance, phase.detach(), wrong_distance, config)
        negative_control_error = float(
            torch.linalg.vector_norm(negative["power_each_w"] - measured["power_each_w"].detach())
            / torch.linalg.vector_norm(measured["power_each_w"].detach()).clamp_min(1e-30)
        )
        initial_metrics = _distribution_metrics(
            measured["power_each_w"], measured["power_before_w"],
            float(config["optics"]["input_power_w"]), float(extreme["aperture_m"]),
            float(config["optics"]["pixel_pitch_m"]),
        )
    result["checks"].update({
        "new_input_normalized_measurement_matches_declared_input_power": abs(input_power / float(config["optics"]["input_power_w"]) - 1.0) <= 1e-6,
        "new_production_metric_gradient_finite_nonzero": math.isfinite(gradient_norm) and gradient_norm > 0.0,
        "production_distance_negative_control_changes_fiber_power": math.isfinite(negative_control_error) and negative_control_error >= 1e-4,
        "three_step_preflight_updates_both_phase_masks": all(
            math.isfinite(x) and x > 0.0
            for x in representative_optimizer["phase_parameter_update_l2_by_layer"]
        ),
        "three_step_preflight_has_finite_nonzero_gradient_for_both_masks": all(
            math.isfinite(x) and x > 0.0
            for x in representative_optimizer["final_gradient_l2_norm_by_encoder_layer"]
        ),
    })
    result["status"] = "PASS" if all(result["checks"].values()) else "INVALID_HARNESS"
    result["preflight_scope"] = "current production ASM/LP01 path; largest swept aperture, longest distance, and smallest pitch gradient exercised"
    result["required_propagation_grid"] = int(config["optics"]["propagation_grid_size"])
    result["representative_measurement_eta_denominator"] = "input power"
    result["new_measurement_checks"] = {
        "source_integrated_power_w": input_power,
        "declared_input_power_w": float(config["optics"]["input_power_w"]),
        "largest_sweep_initial_eta_total": initial_metrics["eta_total"],
        "largest_sweep_initial_N_eff": initial_metrics["N_eff"],
        "input_normalized_efficiency_gradient_l2_norm": gradient_norm,
        "wrong_distance_fiber_power_relative_error": negative_control_error,
        "wrong_distance_negative_control_m": wrong_distance["last_distance_m"],
        "representative_optimizer_architecture": {key: representative[key] for key in representative},
        "representative_optimizer_updates": representative_optimizer["optimizer_steps"],
        "representative_phase_gradient_l2_by_layer": representative_optimizer["final_gradient_l2_norm_by_encoder_layer"],
        "representative_phase_update_l2_by_layer": representative_optimizer["phase_parameter_update_l2_by_layer"],
        "representative_eta_improvement": representative_optimizer["objective_improvement"],
    }
    result["data_reads"] = 0
    result["task_evaluator_calls"] = 0
    result["formal_training_updates"] = 0
    result["phase_only_preflight_optimizer_updates"] = 3
    return result


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def run_preflight(config_path: Path, output_path: Path, device: torch.device) -> dict[str, Any]:
    config = _load_config(config_path)
    result = _diagnostic_preflight(config, device)
    _write_new(output_path, result)
    return result


def run_study(config_path: Path, output_path: Path, preflight_path: Path, progress_path: Path, device: torch.device) -> dict[str, Any]:
    config = _load_config(config_path)
    preflight = run_preflight(config_path, preflight_path, device)
    if preflight["status"] != "PASS":
        failed = {
            "schema": "compact_ols_pareto_results.v1",
            "status": "INVALID_HARNESS",
            "preflight": preflight,
            "formal_training_updates": 0,
            "dataset_reads": 0,
            "task_evaluator_calls": 0,
            "blocked_actions": ["architecture sweep", "formal training", "task evaluation"],
        }
        _write_new(output_path, failed)
        return failed
    if output_path.exists() or progress_path.exists():
        raise FileExistsError("refusing to overwrite an existing result or progress file")
    grid = _architecture_grid(config)
    runtimes: dict[float, torch.nn.Module] = {}
    results: list[dict[str, Any]] = []
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("x", encoding="utf-8") as progress:
        for index, architecture in enumerate(grid, start=1):
            pitch = float(architecture["fiber_pitch_m"])
            if pitch not in runtimes:
                runtimes[pitch] = base._physical_runtime(config, pitch, device)
            runtime = runtimes[pitch]
            entrance = base._source_field(runtime, int(architecture["aperture_pixels"]), float(config["optics"]["input_power_w"]))
            optimizations = [
                _optimize_objective(runtime, entrance, architecture, objective, config, device)
                for objective in OBJECTIVES
            ]
            row = {
                **architecture,
                "kinematic_reachability": base._reachability(architecture, config),
                "manufacturability": _manufacturability(architecture, config),
                "optimization_results": optimizations,
            }
            results.append(row)
            progress.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            progress.flush()
            del entrance
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if index == 1 or index % 10 == 0 or index == len(grid):
                print(
                    f"completed {index}/{len(grid)}: N={architecture['aperture_pixels']}, "
                    f"z={architecture['last_distance_m'] * 1e3:g}mm, pitch={pitch * 1e6:g}um",
                    flush=True,
                )
    points = _all_points(results, config)
    standard_points = [point for point in points if point["fiber_pitch_m"] == 125e-6]
    pareto_all = _pareto_frontier(points)
    pareto_standard = _pareto_frontier(standard_points)
    curves = _curve_tables(points)
    min_n80 = _minimum_aperture_over_dimension_threshold(standard_points, 80.0)
    level_a = [point for point in pareto_standard if point["level"] == "LEVEL_A"]
    if level_a:
        selected_summary = min(level_a, key=lambda p: (p["aperture_pixels"], -p["compactness"], -p["eta_total"]))
        selected_reason = "smallest swept standard-pitch point meeting the stated Level A simulation screen"
        architecture_status = "ARCHITECTURE_FEASIBLE"
    else:
        selected_summary = max(standard_points, key=lambda p: (p["N_eff"], p["eta_total"], -p["aperture_pixels"]))
        selected_reason = "highest measured standard-pitch N_eff with throughput tie-break; no point met all Level A conditions"
        architecture_status = "ARCHITECTURE_MARGINAL" if any(p["N_eff"] > 50.0 for p in standard_points) else "ARCHITECTURE_INFEASIBLE"
    selected_row = next(
        row for row in results
        if row["aperture_pixels"] == selected_summary["aperture_pixels"]
        and row["last_distance_m"] == selected_summary["last_distance_m"]
        and row["fiber_pitch_m"] == selected_summary["fiber_pitch_m"]
    )
    selected = next(
        item for item in selected_row["optimization_results"]
        if item["optimization_type"] == selected_summary["optimization_type"]
    )
    q1_200 = [p for p in standard_points if p["aperture_pixels"] == 100]
    q1_200_summary = {
        "baseline_zero_phase_at_each_distance": [
            {
                "distance_m": row["last_distance_m"],
                "baseline": row["optimization_results"][0]["initial"],
                "best_achieved_Neff_point": max((x["optimized"] for x in row["optimization_results"]), key=lambda x: x["N_eff"]),
            }
            for row in results if row["aperture_pixels"] == 100 and row["fiber_pitch_m"] == 125e-6
        ],
        "measured_200um_level_a_count": sum(p["level"] == "LEVEL_A" for p in q1_200),
        "max_achieved_Neff": max(q1_200, key=lambda p: p["N_eff"]),
        "failure_attribution": "derived from sampled distance/pitch/aperture comparisons after results; see interpretation rules in report",
    }
    result = {
        "schema": "compact_ols_pareto_results.v1",
        "date": "2026-10-13",
        "status": "VALID_COMPLETED",
        "device": str(device),
        "config": config,
        "preflight": preflight,
        "study_scope": {
            "formal_training_updates": 0,
            "dataset_reads": 0,
            "task_evaluator_calls": 0,
            "fid_calls": 0,
            "task_metric_calls": 0,
            "task_loss_changes": 0,
            "material_nonlinearity_in_forward": False,
            "production_geometry_changed": False,
            "diagnostic_phase_optimizer_steps": len(grid) * len(OBJECTIVES) * int(config["optimization"]["steps_per_objective"]),
            "optimizer_parameters": "two phase masks only; no amplitude parameter",
        },
        "grid_derivation": {
            "rule": "ceil(1.5 * max(aperture_width + 2*zmax*sin(theta_max), array_outer_extent) / dx), rounded up to power of two",
            "theta_sampling_limit_sin": float(config["optics"]["wavelength_m"]) / (2 * float(config["optics"]["pixel_pitch_m"])),
            "required_grid_size": _required_grid_size(config),
            "actual_grid_size": int(config["optics"]["propagation_grid_size"]),
            "physical_fov_m": int(config["optics"]["propagation_grid_size"]) * float(config["optics"]["pixel_pitch_m"]),
            "same_dx_all_cases": True,
            "no_field_resize": True,
        },
        "physical_setup": {
            "wavelength_m": float(config["optics"]["wavelength_m"]),
            "pixel_pitch_m": float(config["optics"]["pixel_pitch_m"]),
            "global_plane_shape": [int(config["optics"]["propagation_grid_size"])] * 2,
            "global_plane_fov_m": int(config["optics"]["propagation_grid_size"]) * float(config["optics"]["pixel_pitch_m"]),
            "trainable_encoder_layers": int(config["optics"]["encoder_layers"]),
            "inter_encoder_spacing_m": float(config["optics"]["inter_encoder_spacing_m"]),
            "fiber_array_reference_pitch_m": float(config["fiber"]["pitch_m"][0]),
            "reference_fiber_center_axis_um": [(-4.5 + index) * float(config["fiber"]["pitch_m"][0]) * 1e6 for index in range(10)],
            "reference_outermost_fiber_center_um": 4.5 * float(config["fiber"]["pitch_m"][0]) * 1e6,
            "reference_fiber_center_span_um": 9.0 * float(config["fiber"]["pitch_m"][0]) * 1e6,
            "reference_fiber_array_outer_extent_um": (9.0 * float(config["fiber"]["pitch_m"][0]) + float(config["fiber"]["cladding_diameter_m"])) * 1e6,
            "fiber_cladding_diameter_m": float(config["fiber"]["cladding_diameter_m"]),
        },
        "sweep_design": {
            "architecture_count": len(grid),
            "architectures_per_pitch": len(APERTURES) * len(DISTANCES_M),
            "objectives_per_architecture": list(OBJECTIVES),
            "optimization_steps_per_objective": int(config["optimization"]["steps_per_objective"]),
            "eta_total_definition": "sum_i(P_i)/P_input",
            "N_eff_definition": "(sum_i(P_i)^2)/sum_i(P_i^2)",
            "utilization_thresholds": [-40, -30, -20, -10],
            "fiber_pitch_roles": {"125um": "STANDARD_SM_FIBER_REFERENCE", "others": "NON_STANDARD_GEOMETRY"},
        },
        "architectures": results,
        "pareto_frontier_all_pitches": pareto_all,
        "pareto_frontier_standard_125um": pareto_standard,
        "aperture_curves_standard_125um": curves,
        "minimum_standard_pitch_aperture_with_Neff_over_80": min_n80,
        "Q1_200um_diagnosis": q1_200_summary,
        "selected_simulation_candidate": {**selected_summary, "selection_reason": selected_reason, "full_result": selected},
        "architecture_classification": architecture_status,
        "architecture_classification_scope": "simulation-only for the stated synthetic illumination, phase-only masks, current ASM and LP01 port model; not hardware or fabrication acceptance",
        "level_definition": config["screen"],
        "manufacturability_summary": {
            "score_method": config["manufacturability"]["score_definition"],
            "printer_limits": "UNRATED_NOT_SUPPLIED",
            "fiber_rule": "125um is commercial reference; below 75um is custom requirement; other swept pitches are nonstandard",
            "score_is_a_relative_proxy": True,
        },
        "nonlinearity_sensitivity": _nonlinear_sensitivity(selected, config),
        "answers": {
            "Q1_200um_retention": "see Q1_200um_diagnosis and standard pitch Pareto points; 200um aperture does not meet the 100-channel Level A screen",
            "Q2_minimum_aperture_Neff_over_80": min_n80,
            "Q3_relay_needed": "determined from 125um pitch distance/aperture envelopes; no relay is added in this study",
            "Q4_nonlinearity_architecture": {key: selected_summary[key] for key in ("aperture_pixels", "aperture_m", "last_distance_m", "fiber_pitch_m", "optimization_type")},
            "Q5_final_fabrication_size": "simulation candidate only; no experimental recommendation until measured relay, illumination and printer limits are available",
        },
        "permitted_next_actions": [
            "inspect standard-pitch Pareto candidates",
            "measure relay magnification, illuminated aperture, and physical fiber geometry",
            "obtain printer minimum feature size and maximum fabrication area for an actual process screen",
            "run separately authorized higher-fidelity optical validation",
        ],
        "blocked_actions": [
            "formal training",
            "dataset access",
            "semantic evaluator or FID",
            "loss modification",
            "material nonlinearities in the optical forward model",
            "production geometry mutation",
            "fabrication acceptance",
            "generation-task execution",
        ],
        "claim_limit": "synthetic coherent uniform top-hat illumination, current production ASM and LP01 port overlap, fixed two-mask phase-only model, and finite-step optimization; no global optimum, hardware fabrication, measured relay, or material/device claim",
        "final_measurement_source": progress_path.as_posix(),
    }
    _write_new(output_path, result)
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
        result = run_preflight(args.config, args.preflight_output, device)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if result["status"] != "PASS":
            raise SystemExit(2)
        return
    result = run_study(args.config, args.output, args.preflight_output, args.progress_output, device)
    print(json.dumps({"status": result["status"], "architectures": result.get("sweep_design", {}).get("architecture_count")}, ensure_ascii=False))
    if result["status"] != "VALID_COMPLETED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
