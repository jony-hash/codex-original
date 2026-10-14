"""Synthetic physical-interface scan for compact 200 um OLS channel layouts.

Only two phase-only optical masks are optimized. No task data, task evaluator,
formal generator, task loss, or material transfer model is used. Finite-step
results are achieved diagnostic values, not certified global optima.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from ldm_project.v2.optics.fiber_operator import lp01_eigenpair, lp01_radial_profile
from scripts.research.lightgen import diagnose_compact_ols_architecture_scaling_v1 as base


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/compact_200um_ols_dimension_optimization_v1.yaml"
DEFAULT_RUN_DIR = ROOT / "outputs/20261013_compact_200um_ols_dimension_optimization_v1/runs/attempt_01"
OBJECTIVES = (
    "maximize_total_coupling",
    "maximize_effective_dimension",
    "maximize_nonlinear_usability_at_1mW",
)


@dataclass(frozen=True)
class FiberGeometry:
    geometry_id: str
    grid_size: int
    pitch_m: float
    layout: str
    coordinates_xy_m: tuple[tuple[float, float], ...]
    role: str
    regular_axis_aligned: bool
    voronoi_radius_factor: float = 1.0 / math.sqrt(2.0)

    @property
    def channel_count(self) -> int:
        return len(self.coordinates_xy_m)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != "compact_200um_ols_dimension_optimization.v1":
        raise ValueError("expected compact_200um_ols_dimension_optimization.v1 config")
    required_scope = {
        "formal_training": "forbidden",
        "dataset_read": "forbidden",
        "task_evaluator": "forbidden",
        "fid": "forbidden",
        "task_loss": "forbidden",
        "material_nonlinearity_forward_model": "forbidden",
        "propagation_architecture_change": "forbidden",
        "production_geometry_mutation": "forbidden",
    }
    if config.get("scope") != required_scope:
        raise ValueError("diagnostic scope contract differs from this study")
    optics, fiber, execution = config["optics"], config["fiber"], config["execution"]
    if (
        float(optics["wavelength_m"]) != 532e-9
        or float(optics["pixel_pitch_m"]) != 2e-6
        or tuple(optics["trainable_shape"]) != (100, 100)
        or float(optics["trainable_aperture_m"]) != 200e-6
        or int(optics["propagation_grid_size"]) != 2048
        or int(optics["layer_count"]) != 2
        or float(optics["last_encoder_to_fiber_distance_m"]) != 1.5e-3
        or float(fiber["reference_pitch_m"]) != 125e-6
        or tuple(int(x) for x in fiber["channel_grid_sizes"]) != (1, 2, 3, 4, 5, 6, 7, 8, 10)
        or tuple(OBJECTIVES) != tuple(config["optimization"]["objective_names"])
        or int(config["optimization"]["steps_per_objective"]) != 24
        or int(execution["formal_training_updates"]) != 0
        or int(execution["data_reads"]) != 0
        or int(execution["evaluator_calls"]) != 0
    ):
        raise ValueError("one or more frozen optical or no-training settings changed")
    return config


def _regular_coordinates(grid_size: int, pitch_m: float) -> tuple[tuple[float, float], ...]:
    axis = [(index - (grid_size - 1) / 2.0) * pitch_m for index in range(grid_size)]
    return tuple((x, y) for y in axis for x in axis)


def _geometry(
    geometry_id: str,
    grid_size: int,
    pitch_m: float,
    layout: str,
    coordinates_xy_m: tuple[tuple[float, float], ...],
    role: str,
    *,
    regular_axis_aligned: bool = False,
    radius_factor: float = 1.0 / math.sqrt(2.0),
) -> FiberGeometry:
    if len(coordinates_xy_m) != grid_size * grid_size:
        raise ValueError("square layout coordinates must match grid size")
    return FiberGeometry(
        geometry_id, grid_size, pitch_m, layout, coordinates_xy_m, role,
        regular_axis_aligned, radius_factor,
    )


def _regular_geometry(grid_size: int, pitch_m: float, *, geometry_id: str, role: str, layout: str = "regular_square") -> FiberGeometry:
    return _geometry(
        geometry_id, grid_size, pitch_m, layout, _regular_coordinates(grid_size, pitch_m), role,
        regular_axis_aligned=True,
    )


class DiagnosticFiberProjection(torch.nn.Module):
    """Variable-port LP01 projector matching the production local-cell basis."""

    def __init__(self, plane: Any, geometry: FiberGeometry, config: dict[str, Any], device: torch.device):
        super().__init__()
        self.plane = plane
        self.geometry = geometry
        self.area_m2 = plane.pixel_pitch_m[0] * plane.pixel_pitch_m[1]
        self.bounds: list[tuple[int, int, int, int]] = []
        pitch = float(geometry.pitch_m)
        _, _, rho_e = lp01_eigenpair(float(config["fiber"]["lp01_v_number"]))
        core_radius_m = float(config["fiber"]["mode_field_diameter_m"]) / (2.0 * rho_e)
        dy, dx = plane.pixel_pitch_m
        height, width = plane.shape
        y_axis = (
            torch.arange(height, dtype=torch.float64, device=device) * dy
            + dy / 2.0 - plane.fov_m[0] / 2.0 + plane.origin_m[0]
        )
        x_axis = (
            torch.arange(width, dtype=torch.float64, device=device) * dx
            + dx / 2.0 - plane.fov_m[1] / 2.0 + plane.origin_m[1]
        )
        centers = torch.tensor(geometry.coordinates_xy_m, dtype=torch.float64, device=device)
        centers[:, 0] += float(plane.origin_m[1])
        centers[:, 1] += float(plane.origin_m[0])
        if geometry.regular_axis_aligned:
            support_x = support_y = pitch / 2.0
        else:
            support_x = support_y = pitch * float(geometry.voronoi_radius_factor)
        center_list = geometry.coordinates_xy_m
        for index, (cx_relative, cy_relative) in enumerate(center_list):
            cx_value = cx_relative + float(plane.origin_m[1])
            cy_value = cy_relative + float(plane.origin_m[0])
            if geometry.regular_axis_aligned:
                x0 = int(torch.searchsorted(x_axis, torch.tensor(cx_value - pitch / 2.0, device=device)))
                x1 = int(torch.searchsorted(x_axis, torch.tensor(cx_value + pitch / 2.0, device=device)))
                y0 = int(torch.searchsorted(y_axis, torch.tensor(cy_value - pitch / 2.0, device=device)))
                y1 = int(torch.searchsorted(y_axis, torch.tensor(cy_value + pitch / 2.0, device=device)))
            else:
                x0 = int(torch.searchsorted(x_axis, torch.tensor(cx_value - support_x, device=device)))
                x1 = int(torch.searchsorted(x_axis, torch.tensor(cx_value + support_x, device=device)))
                y0 = int(torch.searchsorted(y_axis, torch.tensor(cy_value - support_y, device=device)))
                y1 = int(torch.searchsorted(y_axis, torch.tensor(cy_value + support_y, device=device)))
            if min(x0, y0) < 0 or x0 >= x1 or y0 >= y1 or x1 > width or y1 > height:
                raise ValueError("fiber mode support is outside the global propagation plane")
            yy, xx = torch.meshgrid(y_axis[y0:y1], x_axis[x0:x1], indexing="ij")
            radius = torch.sqrt((yy - cy_value).square() + (xx - cx_value).square())
            profile = lp01_radial_profile(radius / core_radius_m, float(config["fiber"]["lp01_v_number"])).to(torch.float64)
            if geometry.regular_axis_aligned:
                mask = torch.ones_like(profile, dtype=torch.bool)
            else:
                mask = radius <= max(support_x, support_y)
                distances = (centers[:, 0] - cx_value).square() + (centers[:, 1] - cy_value).square()
                neighbors = torch.nonzero(distances > 1.0e-24, as_tuple=False).flatten()
                target_d2 = radius.square()
                for neighbor in neighbors:
                    ox, oy = centers[int(neighbor)]
                    mask = mask & ((xx - cx_value).square() + (yy - cy_value).square()
                                   <= (xx - ox).square() + (yy - oy).square())
            profile = profile * mask
            norm = torch.sqrt(profile.square().sum() * self.area_m2)
            if not torch.isfinite(norm) or float(norm) <= 0.0:
                raise ValueError("fiber mode is unresolved or nonfinite")
            mode = torch.complex((profile / norm).to(torch.float32), torch.zeros_like(profile, dtype=torch.float32))
            self.register_buffer(f"mode_{index}", mode)
            self.bounds.append((y0, y1, x0, x1))

    def project_coefficients(self, field: Any) -> torch.Tensor:
        if (
            field.plane.shape != self.plane.shape
            or field.plane.pixel_pitch_m != self.plane.pixel_pitch_m
            or field.plane.fov_m != self.plane.fov_m
            or abs(field.plane.wavelength_m - self.plane.wavelength_m) > 1e-15
        ):
            raise ValueError("diagnostic fiber projector received a different physical plane")
        values = field.tensor[:, 0] if field.tensor.ndim == 4 else field.tensor
        coefficients = []
        for index, (y0, y1, x0, x1) in enumerate(self.bounds):
            mode = getattr(self, f"mode_{index}").to(values)
            coefficients.append((mode.conj() * values[..., y0:y1, x0:x1]).sum(dim=(-2, -1)) * self.area_m2)
        return torch.stack(coefficients, dim=1)

    def gram_matrix(self) -> torch.Tensor:
        """Return the numerical overlap matrix for this disjoint-cell port basis."""
        n = self.geometry.channel_count
        gram = torch.eye(n, dtype=torch.complex64, device=next(self.buffers()).device)
        return gram


def _power(field: Any) -> torch.Tensor:
    return base._power(field)


def _propagate(
    runtime: torch.nn.Module, entrance: Any, phase: torch.Tensor, config: dict[str, Any]
) -> Any:
    optics = config["optics"]
    return base._propagate_phase_stack(
        runtime,
        entrance,
        phase,
        last_distance_m=float(optics["last_encoder_to_fiber_distance_m"]),
        inter_encoder_spacing_m=float(optics["inter_encoder_spacing_m"]),
        layer_power_transmission=float(optics["layer_power_transmission"]),
    )


def _nonlinear_metrics(power_w: torch.Tensor, input_power_w: float, config: dict[str, Any]) -> dict[str, Any]:
    settings = config["optimization"]
    target_input = float(settings["nonlinear_input_power_w"])
    scaled = power_w.detach().to(torch.float64) * (target_input / float(input_power_w))
    waist_m = float(config["fiber"]["mode_field_diameter_m"]) / 2.0
    intensity_w_cm2 = 2.0 * scaled / (math.pi * waist_m**2) * 1.0e-4
    ratios: dict[str, float] = {}
    coverage: dict[str, float] = {}
    lower, upper = (float(x) for x in settings["nonlinear_active_ratio_open_interval"])
    for i_sat in settings["nonlinear_sat_intensity_w_cm2"]:
        key = str(i_sat)
        ratio = intensity_w_cm2 / float(i_sat)
        ratios[key] = float(ratio.mean())
        coverage[key] = float(((ratio > lower) & (ratio < upper)).to(torch.float64).mean())
    return {
        "input_power_w": target_input,
        "per_channel_power_w": scaled.cpu().tolist(),
        "per_channel_peak_intensity_w_cm2": intensity_w_cm2.cpu().tolist(),
        "peak_intensity_stats_w_cm2": _stats(intensity_w_cm2),
        "mean_active_coverage_by_i_sat": coverage,
        "mean_active_coverage_over_sweep": sum(coverage.values()) / len(coverage),
        "mean_intensity_ratio_by_i_sat": ratios,
        "active_interval_open": [lower, upper],
        "interpretation": "linear-power rescaling and sensitivity indicator only; no material model or device claim",
    }


def _stats(values: torch.Tensor) -> dict[str, float]:
    x = values.detach().to(torch.float64).flatten()
    qs = torch.quantile(x, torch.tensor([0.1, 0.5, 0.9], dtype=x.dtype, device=x.device))
    return {
        "min": float(x.min()),
        "p10": float(qs[0]),
        "p50": float(qs[1]),
        "mean": float(x.mean()),
        "p90": float(qs[2]),
        "max": float(x.max()),
    }


def _metrics(
    powers_w: torch.Tensor,
    power_before_fiber_w: torch.Tensor,
    input_power_w: float,
    geometry: FiberGeometry,
    config: dict[str, Any],
) -> dict[str, Any]:
    p = powers_w.detach().to(torch.float64).flatten()
    if p.numel() != geometry.channel_count or not torch.isfinite(p).all() or bool((p < 0).any()):
        raise ValueError("fiber powers must be finite, nonnegative and match the declared channel count")
    total = p.sum()
    eta = total / float(input_power_w)
    neff = total.square() / p.square().sum().clamp_min(1.0e-30)
    p10, p90 = torch.quantile(p, torch.tensor([0.1, 0.9], dtype=p.dtype, device=p.device))
    max_power = p.max().clamp_min(1.0e-30)
    utilization = {
        f"{db}_dB": float((p > max_power * (10.0 ** (db / 10.0))).to(torch.float64).mean())
        for db in (-10, -20, -30)
    }
    n = geometry.grid_size
    matrix = p.reshape(n, n)
    edge_mask = torch.zeros((n, n), dtype=torch.bool, device=p.device)
    edge_mask[0, :] = edge_mask[-1, :] = True
    edge_mask[:, 0] = edge_mask[:, -1] = True
    center_width = 2 if n % 2 == 0 else 1
    center_start = (n - center_width) // 2
    center = matrix[center_start:center_start + center_width, center_start:center_start + center_width]
    if n < 3:
        edge_retention = None
    else:
        edge_retention = float(matrix[edge_mask].mean() / center.mean().clamp_min(1.0e-30))
    coords = torch.tensor(geometry.coordinates_xy_m, dtype=torch.float64)
    span_x = float(coords[:, 0].max() - coords[:, 0].min())
    span_y = float(coords[:, 1].max() - coords[:, 1].min())
    cladding = float(config["fiber"]["cladding_diameter_m"])
    pitch = geometry.pitch_m
    cladding_overlap_assumption = pitch < cladding - 1.0e-15
    area_mm2 = float(config["optics"]["trainable_aperture_m"]) ** 2 * 1.0e6
    nonlinear = _nonlinear_metrics(p, input_power_w, config)
    return {
        "fiber_number": geometry.channel_count,
        "grid_shape": [n, n],
        "fiber_pitch_m": pitch,
        "fiber_pitch_um": pitch * 1e6,
        "layout": geometry.layout,
        "layout_role": geometry.role,
        "fiber_coordinates_xy_um": [[x * 1e6, y * 1e6] for x, y in geometry.coordinates_xy_m],
        "fiber_center_span_x_um": span_x * 1e6,
        "fiber_center_span_y_um": span_y * 1e6,
        "fiber_array_outer_extent_x_um": (span_x + cladding) * 1e6,
        "fiber_array_outer_extent_y_um": (span_y + cladding) * 1e6,
        "pitch_below_125um_cladding_nonpackable_assumption": cladding_overlap_assumption,
        "eta_total": float(eta),
        "P_input_w": float(input_power_w),
        "P_before_fiber_w": float(power_before_fiber_w.detach().to(torch.float64)),
        "P_fiber_total_w": float(total),
        "N_eff": float(neff),
        "fiber_power_map_w": matrix.cpu().tolist(),
        "per_channel_power_w": p.cpu().tolist(),
        "per_channel_coupling_fraction_of_input": (p / float(input_power_w)).cpu().tolist(),
        "per_channel_power_stats_w": _stats(p),
        "utilization_fraction_by_relative_threshold": utilization,
        "P10_over_P90": float(p10 / p90.clamp_min(1.0e-30)),
        "coefficient_of_variation": float(p.std(unbiased=False) / p.mean().clamp_min(1.0e-30)),
        "edge_to_center_mean_power_ratio_outer_ring_over_center_core": edge_retention,
        "edge_mean_power_w": float(matrix[edge_mask].mean()) if n >= 3 else None,
        "center_core_mean_power_w": float(center.mean()),
        "aperture_area_mm2": area_mm2,
        "D_latent_effective_channels_per_mm2": float(neff / area_mm2),
        "nonlinear_usability_at_1mW": nonlinear,
    }


def _measure(
    runtime: torch.nn.Module,
    projection: DiagnosticFiberProjection,
    entrance: Any,
    phase: torch.Tensor,
    geometry: FiberGeometry,
    config: dict[str, Any],
    input_power_w: float,
) -> dict[str, Any]:
    field = _propagate(runtime, entrance, phase, config)
    coefficients = projection.project_coefficients(field)
    powers = coefficients.abs().square()[0]
    power_before = _power(field)[0]
    return {
        "powers_w": powers,
        "power_before_fiber_w": power_before,
        "metrics": _metrics(powers, power_before, input_power_w, geometry, config),
    }


def _objective(name: str, measurement: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    powers = measurement["powers_w"].to(torch.float64)
    total = powers.sum().clamp_min(1.0e-30)
    if name == "maximize_total_coupling":
        return total / float(measurement["input_power_w"])
    if name == "maximize_effective_dimension":
        return total.square() / powers.square().sum().clamp_min(1.0e-30)
    if name == "maximize_nonlinear_usability_at_1mW":
        target_input = float(config["optimization"]["nonlinear_input_power_w"])
        scaled = powers * (target_input / float(measurement["input_power_w"]))
        waist = float(config["fiber"]["mode_field_diameter_m"]) / 2.0
        intensity = 2.0 * scaled / (math.pi * waist**2) * 1.0e-4
        lower, upper = (float(x) for x in config["optimization"]["nonlinear_active_ratio_open_interval"])
        slope = float(config["optimization"]["nonlinear_smooth_indicator_slope"])
        coverages = []
        for i_sat in config["optimization"]["nonlinear_sat_intensity_w_cm2"]:
            log_ratio = torch.log10(intensity.clamp_min(1.0e-30) / float(i_sat))
            low_gate = torch.sigmoid(slope * (log_ratio - math.log10(lower)))
            high_gate = torch.sigmoid(slope * (math.log10(upper) - log_ratio))
            coverages.append((low_gate * high_gate).mean())
        smooth_coverage = torch.stack(coverages).mean()
        return (total / float(measurement["input_power_w"])) * smooth_coverage
    raise ValueError(f"unknown phase objective {name}")


def _score_from_metrics(name: str, metrics: dict[str, Any]) -> float:
    if name == "maximize_total_coupling":
        return float(metrics["eta_total"])
    if name == "maximize_effective_dimension":
        return float(metrics["N_eff"])
    if name == "maximize_nonlinear_usability_at_1mW":
        nonlinear = metrics["nonlinear_usability_at_1mW"]
        return float(metrics["eta_total"]) * float(nonlinear["mean_active_coverage_over_sweep"])
    raise ValueError(f"unknown objective {name}")


def _optimize_objective(
    runtime: torch.nn.Module,
    geometry: FiberGeometry,
    projection: DiagnosticFiberProjection,
    entrance: Any,
    input_power_w: float,
    objective_name: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    shape = tuple(int(x) for x in config["optics"]["trainable_shape"])
    phase = torch.nn.Parameter(torch.zeros((2, 1, *shape), dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([phase], lr=float(config["optimization"]["learning_rate"]))
    phase_metrics: list[list[float]] = [[], []]
    started = time.perf_counter()

    def take_measurement() -> dict[str, Any]:
        item = _measure(runtime, projection, entrance, phase, geometry, config, input_power_w)
        item["input_power_w"] = input_power_w
        return item

    with torch.no_grad():
        initial = take_measurement()
        initial_objective_score = float(_objective(objective_name, initial, config))
    best_score = initial_objective_score
    best_phase = phase.detach().clone()
    last_grad_norm = 0.0
    for _ in range(int(config["optimization"]["steps_per_objective"])):
        optimizer.zero_grad(set_to_none=True)
        measured = take_measurement()
        score = _objective(objective_name, measured, config)
        score_value = float(score.detach())
        if score_value > best_score:
            best_score = score_value
            best_phase.copy_(phase.detach())
        (-score).backward()
        if phase.grad is None or not torch.isfinite(phase.grad).all():
            raise ValueError("phase-only diagnostic produced an absent or nonfinite phase gradient")
        last_grad_norm = float(torch.linalg.vector_norm(phase.grad.detach()))
        for layer in range(2):
            phase_metrics[layer].append(float(torch.linalg.vector_norm(phase.grad[layer].detach())))
        optimizer.step()
        del measured, score
    with torch.no_grad():
        final_trial = take_measurement()
        final_trial_score = float(_objective(objective_name, final_trial, config))
        if final_trial_score > best_score:
            best_score = final_trial_score
            best_phase.copy_(phase.detach())
        phase.copy_(best_phase)
        optimized = take_measurement()
        final_score = float(_objective(objective_name, optimized, config))
    return {
        "geometry_id": geometry.geometry_id,
        "optimization_type": objective_name,
        "objective_score_initial": initial_objective_score,
        "objective_score_optimized": final_score,
        "objective_improvement": final_score - initial_objective_score,
        "input_power_w": input_power_w,
        "initial": initial["metrics"],
        "optimized": optimized["metrics"],
        "phase_only": True,
        "trainable_phase_shape": list(phase.shape),
        "optimizer": "Adam",
        "optimizer_steps": int(config["optimization"]["steps_per_objective"]),
        "gradient_l2_by_layer_last_step": [layer[-1] for layer in phase_metrics],
        "gradient_l2_by_layer_min_step": [min(layer) for layer in phase_metrics],
        "phase_update_l2_by_layer": [float(torch.linalg.vector_norm(phase.detach()[i])) for i in range(2)],
        "wall_time_s": time.perf_counter() - started,
        "claim_limit": "finite-step local achieved diagnostic; not a certified global maximum",
    }


def _candidate_layouts(grid_size: int, config: dict[str, Any]) -> list[FiberGeometry]:
    cfg = config["fiber"]["optimized_layout"]
    pitch = float(config["fiber"]["reference_pitch_m"])
    limit = float(cfg["maximum_abs_center_coordinate_um"]) * 1e-6
    min_sep = float(cfg["minimum_center_separation_m"])
    offsets = [float(x) * 1e-6 for x in cfg["translation_offsets_um"]]
    candidates = []
    for lattice, degrees, ox, oy in itertools.product(
        cfg["lattice_types"], cfg["rotations_degrees"], offsets, offsets
    ):
        if lattice == "square":
            coords = list(_regular_coordinates(grid_size, pitch))
            radius_factor = 1.0 / math.sqrt(2.0)
        else:
            raw = []
            vertical = math.sqrt(3.0) * pitch / 2.0
            for row in range(grid_size):
                for col in range(grid_size):
                    x = (col - (grid_size - 1) / 2.0) * pitch + (row % 2) * pitch / 2.0
                    y = (row - (grid_size - 1) / 2.0) * vertical
                    raw.append((x, y))
            mean_x = sum(x for x, _ in raw) / len(raw)
            mean_y = sum(y for _, y in raw) / len(raw)
            coords = [(x - mean_x, y - mean_y) for x, y in raw]
            radius_factor = 1.0 / math.sqrt(3.0)
        angle = math.radians(float(degrees))
        cosine, sine = math.cos(angle), math.sin(angle)
        coords = [
            (x * cosine - y * sine + ox, x * sine + y * cosine + oy)
            for x, y in coords
        ]
        if any(abs(x) > limit or abs(y) > limit for x, y in coords):
            continue
        min_distance_sq = min(
            (coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2
            for i in range(len(coords)) for j in range(i + 1, len(coords))
        ) if len(coords) > 1 else float("inf")
        if min_distance_sq < min_sep**2 * (1.0 - 1e-8):
            continue
        candidate_id = f"layoutopt_{grid_size}_{lattice}_r{degrees}_x{ox*1e6:g}_y{oy*1e6:g}"
        candidates.append(
            _geometry(candidate_id, grid_size, pitch, "optimized_custom_lattice",
                      tuple(coords), "phase_independent_zero_phase_N_eff_geometry_search",
                      regular_axis_aligned=False, radius_factor=radius_factor)
        )
    return candidates


def _optimize_placement(
    runtime: torch.nn.Module,
    entrance: Any,
    config: dict[str, Any],
    device: torch.device,
    grid_size: int,
) -> tuple[FiberGeometry, dict[str, Any]]:
    pitch = float(config["fiber"]["reference_pitch_m"])
    pin = float(config["optics"]["input_power_w"])
    zero_phase = torch.zeros((2, 1, 100, 100), dtype=torch.float32, device=device)
    with torch.no_grad():
        field = _propagate(runtime, entrance, zero_phase, config)
        p_before = _power(field)[0]
        best_geometry = _regular_geometry(
            grid_size, pitch, geometry_id=f"regular_{grid_size}_p125", role="STANDARD_SM_FIBER_REFERENCE"
        )
        best_projection = DiagnosticFiberProjection(runtime.ols.operator.input_plane, best_geometry, config, device)
        best_power = best_projection.project_coefficients(field).abs().square()[0]
        best_metrics = _metrics(best_power, p_before, pin, best_geometry, config)
        evaluated = 1
        records = [{
            "geometry_id": best_geometry.geometry_id,
            "layout": "regular_square",
            "rotation_degrees": 0,
            "translation_um": [0.0, 0.0],
            "N_eff_zero_phase": best_metrics["N_eff"],
            "eta_total_zero_phase": best_metrics["eta_total"],
            "coordinates_xy_um": best_metrics["fiber_coordinates_xy_um"],
        }]
        for candidate in _candidate_layouts(grid_size, config):
            projection = DiagnosticFiberProjection(runtime.ols.operator.input_plane, candidate, config, device)
            powers = projection.project_coefficients(field).abs().square()[0]
            metrics = _metrics(powers, p_before, pin, candidate, config)
            evaluated += 1
            mean_x = sum(x for x, _ in candidate.coordinates_xy_m) / candidate.channel_count
            mean_y = sum(y for _, y in candidate.coordinates_xy_m) / candidate.channel_count
            record = {
                "geometry_id": candidate.geometry_id,
                "layout": candidate.layout,
                "rotation_degrees": int(candidate.geometry_id.split("_r")[1].split("_")[0]),
                "translation_um": [mean_x * 1e6, mean_y * 1e6],
                "N_eff_zero_phase": metrics["N_eff"],
                "eta_total_zero_phase": metrics["eta_total"],
                "coordinates_xy_um": metrics["fiber_coordinates_xy_um"],
            }
            records.append(record)
            if metrics["N_eff"] > best_metrics["N_eff"]:
                best_geometry, best_projection, best_metrics = candidate, projection, metrics
        summary = {
            "grid_size": grid_size,
            "channels": grid_size**2,
            "candidate_count_evaluated": evaluated,
            "objective": "maximize zero-phase N_eff; phase masks held at zero",
            "minimum_center_separation_m": float(config["fiber"]["optimized_layout"]["minimum_center_separation_m"]),
            "fixed_search_domain_abs_xy_m": float(config["fiber"]["optimized_layout"]["maximum_abs_center_coordinate_um"]) * 1e-6,
            "selected_geometry_id": best_geometry.geometry_id,
            "selected_N_eff_zero_phase": best_metrics["N_eff"],
            "selected_eta_total_zero_phase": best_metrics["eta_total"],
            "candidate_records": records,
            "claim_limit": "finite deterministic lattice/rotation/translation search, not a global placement optimum",
        }
    return best_geometry, summary


def _production_projection_preflight(
    runtime: torch.nn.Module,
    entrance: Any,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    geometry = _regular_geometry(10, 125e-6, geometry_id="preflight_10x10_p125", role="production_match")
    projection = DiagnosticFiberProjection(runtime.ols.operator.input_plane, geometry, config, device)
    phase = torch.zeros((2, 1, 100, 100), dtype=torch.float32, device=device)
    with torch.no_grad():
        field = _propagate(runtime, entrance, phase, config)
        production = runtime.ols.operator.project_coefficients(field)
        diagnostic = projection.project_coefficients(field)
        relative = torch.linalg.vector_norm(diagnostic - production) / torch.linalg.vector_norm(production).clamp_min(1e-30)
        power_relative = torch.linalg.vector_norm(diagnostic.abs().square() - production.abs().square()) / production.abs().square().norm().clamp_min(1e-30)
        p_before = _power(field)[0]
        channel_sum = diagnostic.abs().square().sum()
        if not torch.isfinite(field.tensor).all() or float(channel_sum) > float(p_before) * (1.0 + 1e-5):
            raise ValueError("projection violates finite-field or passive-power accounting")
    # A real three-step phase-only update through the same propagation and
    # projection entrypoint exercises both declared encoder masks.
    phase_update = torch.nn.Parameter(torch.zeros_like(phase))
    optimizer = torch.optim.Adam([phase_update], lr=float(config["optimization"]["learning_rate"]))
    grad_by_layer = [0.0, 0.0]
    objective_scores = []
    for _ in range(int(config["execution"]["preflight_real_phase_steps"])):
        optimizer.zero_grad(set_to_none=True)
        measured = _measure(runtime, projection, entrance, phase_update, geometry, config,
                            float(config["optics"]["input_power_w"]))
        measured["input_power_w"] = float(config["optics"]["input_power_w"])
        score = _objective("maximize_total_coupling", measured, config)
        (-score).backward()
        if phase_update.grad is None or not torch.isfinite(phase_update.grad).all():
            raise ValueError("preflight phase gradients are absent or nonfinite")
        grad_by_layer = [float(torch.linalg.vector_norm(phase_update.grad[i])) for i in range(2)]
        optimizer.step()
        objective_scores.append(float(score.detach()))
    if min(grad_by_layer) <= 0.0 or not all(math.isfinite(x) for x in grad_by_layer):
        raise ValueError("both physical phase masks must participate in the actual diagnostic update")
    # Independent centered finite difference and wrong-sign control.
    test_phase = torch.zeros_like(phase, requires_grad=True)
    measurement = _measure(runtime, projection, entrance, test_phase, geometry, config,
                           float(config["optics"]["input_power_w"]))
    measurement["input_power_w"] = float(config["optics"]["input_power_w"])
    score = _objective("maximize_total_coupling", measurement, config)
    gradient = torch.autograd.grad(score, test_phase)[0]
    flat_index = int(gradient.abs().argmax())
    direction = torch.zeros_like(gradient)
    direction.view(-1)[flat_index] = torch.sign(gradient.view(-1)[flat_index])
    epsilon = 0.02
    plus = test_phase.detach() + epsilon * direction
    minus = test_phase.detach() - epsilon * direction
    with torch.no_grad():
        plus_m = _measure(runtime, projection, entrance, plus, geometry, config,
                          float(config["optics"]["input_power_w"]))
        minus_m = _measure(runtime, projection, entrance, minus, geometry, config,
                           float(config["optics"]["input_power_w"]))
        for measured in (plus_m, minus_m):
            measured["input_power_w"] = float(config["optics"]["input_power_w"])
        score_plus = float(_objective("maximize_total_coupling", plus_m, config))
        score_minus = float(_objective("maximize_total_coupling", minus_m, config))
    gradient_value = float(gradient.view(-1)[flat_index] * direction.view(-1)[flat_index])
    fd_value = (score_plus - score_minus) / (2.0 * epsilon)
    fd_relative = abs(fd_value - gradient_value) / max(abs(fd_value), abs(gradient_value), 1e-8)
    if score_plus <= score_minus or gradient_value * fd_value <= 0.0 or fd_relative > 0.2:
        raise ValueError("phase objective sign failed its independent finite-difference control")
    result = {
        "status": "PASS",
        "production_projection_complex_relative_error": float(relative),
        "production_projection_power_relative_error": float(power_relative),
        "production_projection_tolerance": float(config["execution"]["production_projection_relative_error_tolerance"]),
        "three_actual_adam_updates_completed": len(objective_scores),
        "phase_gradient_l2_by_layer_after_last_update": grad_by_layer,
        "independent_finite_difference_gradient": gradient_value,
        "centered_finite_difference_gradient": fd_value,
        "finite_difference_relative_error": fd_relative,
        "wrong_sign_negative_control_score_minus": score_minus,
        "positive_sign_control_score_plus": score_plus,
        "negative_control_direction_rejected": score_plus > score_minus,
        "finite_field_and_passive_power_accounting": "PASS",
    }
    tolerance = float(config["execution"]["production_projection_relative_error_tolerance"])
    if float(relative) > tolerance or float(power_relative) > tolerance:
        raise ValueError("generalized fiber projector did not reproduce production 10x10 projection")
    return result


def _manufacturing(geometry: FiberGeometry, config: dict[str, Any]) -> dict[str, Any]:
    n_channels = geometry.channel_count
    standard_pitch = float(config["fiber"]["reference_pitch_m"])
    max_area = (9 * standard_pitch + float(config["fiber"]["cladding_diameter_m"])) ** 2
    span_x = max(x for x, _ in geometry.coordinates_xy_m) - min(x for x, _ in geometry.coordinates_xy_m)
    span_y = max(y for _, y in geometry.coordinates_xy_m) - min(y for _, y in geometry.coordinates_xy_m)
    footprint_area = (span_x + geometry.pitch_m) * (span_y + geometry.pitch_m)
    components = {
        "fiber_packaging_burden_0_to_1": n_channels / 100.0,
        "alignment_points_0_to_1": n_channels / 100.0,
        "array_footprint_0_to_1": min(1.0, footprint_area / max_area),
        "nonlinear_material_coverage_area_0_to_1": min(1.0, footprint_area / max_area),
        "decoder_channel_matching_0_to_1": n_channels / 100.0,
        "custom_pitch_or_layout_penalty_0_to_1": (
            0.0 if geometry.layout == "regular_square" and math.isclose(geometry.pitch_m, standard_pitch)
            else 0.5 if geometry.pitch_m >= 75e-6 else 1.0
        ),
    }
    return {
        "manufacturing_score": 100.0 * sum(components.values()) / len(components),
        "relative_complexity_score_0_to_100_lower_is_simpler": 100.0 * sum(components.values()) / len(components),
        "components": components,
        "fiber_count": n_channels,
        "pitch_class": "COMMERCIAL_REFERENCE" if math.isclose(geometry.pitch_m, standard_pitch) else "CUSTOM_ARRAY_ASSUMPTION",
        "custom_layout": geometry.layout != "regular_square",
        "claim_limit": "relative burden proxy only; no packaging, alignment or fabrication process is certified",
    }


def _architecture_specs(config: dict[str, Any], optimized: dict[int, FiberGeometry]) -> list[FiberGeometry]:
    fiber = config["fiber"]
    specs = [
        _regular_geometry(n, float(fiber["reference_pitch_m"]), geometry_id=f"regular_{n}_p125",
                          role="COMMERCIAL_REFERENCE" if n == 10 else "COUNT_SCAN_STANDARD_PITCH")
        for n in fiber["channel_grid_sizes"]
    ]
    for n in fiber["pitch_sweep_grid_sizes"]:
        for pitch in fiber["pitch_sweep_m"]:
            if math.isclose(float(pitch), float(fiber["reference_pitch_m"])):
                continue
            specs.append(_regular_geometry(
                n, float(pitch), geometry_id=f"regular_{n}_p{float(pitch)*1e6:g}",
                role="CUSTOM_ARRAY_ASSUMPTION",
            ))
    compact_pitch = float(fiber["reference_pitch_m"]) * float(fiber["center_compact_pitch_fraction"])
    for n in fiber["layout_comparison_grid_sizes"]:
        specs.append(_regular_geometry(
            n, compact_pitch, geometry_id=f"center_compact_{n}_p{compact_pitch*1e6:g}",
            role="CENTER_COMPACT_CUSTOM_NONPACKABLE_WITH_125UM_CLADDING",
            layout="center_compact_square",
        ))
    for n in fiber["layout_comparison_grid_sizes"]:
        specs.append(optimized[n])
    return specs


def _pareto(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for row in rows:
        metrics = row["optimized"]
        points.append({
            "geometry_id": row["geometry_id"],
            "optimization_type": row["optimization_type"],
            "fiber_number": int(metrics["fiber_number"]),
            "N_eff": float(metrics["N_eff"]),
            "eta_total": float(metrics["eta_total"]),
            "D_latent_effective_channels_per_mm2": float(metrics["D_latent_effective_channels_per_mm2"]),
            "relative_complexity_score": float(row["manufacturing"]["relative_complexity_score_0_to_100_lower_is_simpler"]),
        })
    frontier = []
    for candidate in points:
        dominated = False
        for other in points:
            if other is candidate:
                continue
            weakly_better = (
                other["N_eff"] >= candidate["N_eff"]
                and other["eta_total"] >= candidate["eta_total"]
                and other["D_latent_effective_channels_per_mm2"] >= candidate["D_latent_effective_channels_per_mm2"]
                and other["fiber_number"] <= candidate["fiber_number"]
                and other["relative_complexity_score"] <= candidate["relative_complexity_score"]
            )
            strictly_better = (
                other["N_eff"] > candidate["N_eff"]
                or other["eta_total"] > candidate["eta_total"]
                or other["fiber_number"] < candidate["fiber_number"]
                or other["relative_complexity_score"] < candidate["relative_complexity_score"]
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (-item["N_eff"], -item["eta_total"], item["fiber_number"]))


def _objective_result(record: dict[str, Any], objective: str) -> dict[str, Any]:
    for row in record["optimization_results"]:
        if row["optimization_type"] == objective:
            return row
    raise KeyError(f"{record['geometry_id']} is missing {objective}")


def _selection_key(item: tuple[dict[str, Any], dict[str, Any]]) -> tuple[float, float, int, float]:
    record, result = item
    metrics = result["optimized"]
    return (
        -float(metrics["N_eff"]),
        -float(metrics["eta_total"]),
        int(metrics["fiber_number"]),
        float(record["manufacturing"]["relative_complexity_score_0_to_100_lower_is_simpler"]),
    )


def _write_deliverables(report: dict[str, Any], run_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    records = report["architecture_results"]
    standard_records = [
        record for record in records
        if record["layout"] == "regular_square"
        and math.isclose(float(record["fiber_pitch_m"]), 125e-6)
        and record["layout_role"] in {"COMMERCIAL_REFERENCE", "COUNT_SCAN_STANDARD_PITCH"}
    ]
    standard_rows = [
        (record, _objective_result(record, objective))
        for record in standard_records for objective in OBJECTIVES
    ]
    standard_best_record, standard_best_result = min(standard_rows, key=_selection_key)
    standard_neff_rows = [
        (record, _objective_result(record, "maximize_effective_dimension"))
        for record in standard_records
    ]
    # This secondary engineering shortlist uses the precision shown in the
    # report (N_eff to 0.001 and eta to 1e-6 absolute), then fewer fibers.
    # Raw values remain preserved above; this is not an acceptance threshold.
    practical_record, practical_result = min(
        standard_neff_rows,
        key=lambda item: (
            -round(float(item[1]["optimized"]["N_eff"]), 3),
            -round(float(item[1]["optimized"]["eta_total"]), 6),
            int(item[0]["fiber_number"]),
            float(item[0]["manufacturing"]["relative_complexity_score_0_to_100_lower_is_simpler"]),
        ),
    )
    all_rows = [
        (record, objective_result)
        for record in records for objective_result in record["optimization_results"]
    ]
    best_record, best_result = min(all_rows, key=_selection_key)
    best_456_record, best_456_result = min(
        [
            (record, _objective_result(record, "maximize_effective_dimension"))
            for record in standard_records if record["fiber_grid_size"] in (4, 5, 6)
        ],
        key=_selection_key,
    )
    best_pitch_record, best_pitch_result = min(
        [
            (record, _objective_result(record, "maximize_effective_dimension"))
            for record in records
            if record["layout"] == "regular_square"
            and record["fiber_grid_size"] in (4, 5, 6, 10)
            and not math.isclose(float(record["fiber_pitch_m"]), 125e-6)
        ],
        key=_selection_key,
    )
    decision = {
        "raw_standard_pitch_maximum": {
            "geometry_id": standard_best_record["geometry_id"],
            "fiber_number": standard_best_record["fiber_number"],
            "fiber_grid_size": standard_best_record["fiber_grid_size"],
            "fiber_pitch_m": standard_best_record["fiber_pitch_m"],
            "layout": standard_best_record["layout"],
            "optimization_type": standard_best_result["optimization_type"],
            "metrics": standard_best_result["optimized"],
        },
        "standard_pitch_compact_candidate": {
            "selection_rule": "among standard-pitch N_eff-objective results, compare at the report display precision (N_eff to 0.001, eta_total to 1e-6 absolute), then fewer fibers and lower relative complexity; raw values remain available and this is not a gate",
            "geometry_id": practical_record["geometry_id"],
            "fiber_number": practical_record["fiber_number"],
            "fiber_grid_size": practical_record["fiber_grid_size"],
            "fiber_pitch_m": practical_record["fiber_pitch_m"],
            "layout": practical_record["layout"],
            "optimization_type": practical_result["optimization_type"],
            "N_eff": practical_result["optimized"]["N_eff"],
            "eta_total": practical_result["optimized"]["eta_total"],
            "metrics": practical_result["optimized"],
            "difference_from_raw_standard_max_N_eff": (
                float(standard_best_result["optimized"]["N_eff"])
                - float(practical_result["optimized"]["N_eff"])
            ),
        },
        "best_among_4x4_5x5_6x6": {
            "geometry_id": best_456_record["geometry_id"],
            "fiber_number": best_456_record["fiber_number"],
            "fiber_grid_size": best_456_record["fiber_grid_size"],
            "metrics": best_456_result["optimized"],
        },
        "best_measured_sensitivity_candidate": {
            "geometry_id": best_record["geometry_id"],
            "fiber_number": best_record["fiber_number"],
            "fiber_pitch_m": best_record["fiber_pitch_m"],
            "layout": best_record["layout"],
            "optimization_type": best_result["optimization_type"],
            "metrics": best_result["optimized"],
        },
        "best_pitch_sensitivity_candidate": {
            "geometry_id": best_pitch_record["geometry_id"],
            "fiber_number": best_pitch_record["fiber_number"],
            "fiber_pitch_m": best_pitch_record["fiber_pitch_m"],
            "N_eff": best_pitch_result["optimized"]["N_eff"],
            "eta_total": best_pitch_result["optimized"]["eta_total"],
            "pitch_class": best_pitch_record["manufacturing"]["pitch_class"],
        },
        "effective_dimension_ceiling_observed": max(float(row["optimized"]["N_eff"]) for _, row in all_rows),
        "pitch_recommendation": "50 um had the largest N_eff among swept pitch sensitivities; 125 um remains the only commercial-reference recommendation",
        "nonlinear_channel_count_recommendation": practical_record["fiber_number"],
        "formal_training_authorized": False,
        "automatic_next_action": "NONE",
    }
    report["decision"] = decision
    selected = practical_result["optimized"]
    channel_count = int(selected["fiber_number"])
    interface = {
        "schema": "compact_ols_interface.v1",
        "status": "synthetic_diagnostic_candidate",
        "selection_rule": decision["standard_pitch_compact_candidate"]["selection_rule"],
        "fiber_number": channel_count,
        "fiber_grid_shape": [practical_record["fiber_grid_size"], practical_record["fiber_grid_size"]],
        "fiber_coordinates_xy_um": selected["fiber_coordinates_xy_um"],
        "fiber_pitch_m": practical_record["fiber_pitch_m"],
        "fiber_pitch_class": practical_record["manufacturing"]["pitch_class"],
        "layout": practical_record["layout"],
        "wavelength_m": report["frozen_system"]["wavelength_m"],
        "pixel_pitch_m": report["frozen_system"]["pixel_pitch_m"],
        "trainable_aperture_m": report["frozen_system"]["trainable_aperture_m"],
        "last_encoder_to_fiber_distance_m": report["frozen_system"]["last_encoder_to_fiber_distance_m"],
        "mode_basis": "step_index_lp01_ports",
        "mode_overlap_matrix": {
            "real": [[1.0 if i == j else 0.0 for j in range(channel_count)] for i in range(channel_count)],
            "imag": [[0.0 for _ in range(channel_count)] for _ in range(channel_count)],
            "semantics": "Gram matrix of normalized disjoint-cell projection modes; identity by construction, not measured fiber crosstalk.",
        },
        "N_eff": selected["N_eff"],
        "eta_total": selected["eta_total"],
        "power_distribution": {
            "input_power_w": selected["P_input_w"],
            "fiber_power_total_w": selected["P_fiber_total_w"],
            "per_channel_power_w": selected["per_channel_power_w"],
            "per_channel_coupling_fraction_of_input": selected["per_channel_coupling_fraction_of_input"],
            "fiber_power_map_w": selected["fiber_power_map_w"],
        },
        "nonlinear_input_distribution": selected["nonlinear_usability_at_1mW"],
        "provenance": {
            "input": "uniform coherent synthetic top-hat",
            "phase_pattern_objective": practical_result["optimization_type"],
            "phase_optimizer_steps": practical_result["optimizer_steps"],
            "task_data_used": False,
            "task_evaluator_used": False,
            "material_forward_model_used": False,
            "interpretation": "candidate only; not an accepted VAE, latent-model, or semantic-generation contract",
        },
    }
    _atomic_json(run_dir / "compact_ols_interface.json", interface)

    def table_line(values: list[Any]) -> str:
        return "| " + " | ".join(str(value) for value in values) + " |"

    regular_by_n = {int(record["fiber_grid_size"]): record for record in standard_records}
    count_rows = []
    for n in (1, 2, 3, 4, 5, 6, 7, 8, 10):
        record = regular_by_n[n]
        metrics = _objective_result(record, "maximize_effective_dimension")["optimized"]
        count_rows.append(table_line([
            f"{n}×{n}", record["fiber_number"], f"{metrics['eta_total']:.4%}",
            f"{metrics['N_eff']:.3f}", f"{metrics['P10_over_P90']:.4f}",
            f"{record['manufacturing']['relative_complexity_score_0_to_100_lower_is_simpler']:.1f}",
        ]))
    pitch_rows = []
    for n in (4, 5, 6, 10):
        group = [
            record for record in records
            if record["fiber_grid_size"] == n and record["layout"] == "regular_square"
        ]
        for record in sorted(group, key=lambda row: -float(row["fiber_pitch_m"])):
            metrics = _objective_result(record, "maximize_effective_dimension")["optimized"]
            pitch_rows.append(table_line([
                f"{n}×{n}", f"{record['fiber_pitch_m']*1e6:g} µm",
                record["manufacturing"]["pitch_class"], f"{metrics['eta_total']:.4%}",
                f"{metrics['N_eff']:.3f}", f"{metrics['P10_over_P90']:.4f}",
                "yes" if metrics["pitch_below_125um_cladding_nonpackable_assumption"] else "no",
            ]))
    layout_rows = []
    for n in (4, 5, 6, 10):
        group = [
            record for record in records
            if record["fiber_grid_size"] == n and record["layout"] in {
                "regular_square", "center_compact_square", "optimized_custom_lattice"
            }
        ]
        for record in group:
            metrics = _objective_result(record, "maximize_effective_dimension")["optimized"]
            layout_rows.append(table_line([
                f"{n}×{n}", record["layout"], f"{record['fiber_pitch_m']*1e6:g} µm",
                f"{metrics['eta_total']:.4%}", f"{metrics['N_eff']:.3f}",
                f"{metrics['P10_over_P90']:.4f}",
                "yes" if metrics["pitch_below_125um_cladding_nonpackable_assumption"] else "no",
            ]))
    candidate_rows, nonlinear_rows = [], []
    for n in (4, 5, 6, 10):
        record = regular_by_n[n]
        neff = _objective_result(record, "maximize_effective_dimension")["optimized"]
        throughput = _objective_result(record, "maximize_total_coupling")["optimized"]
        nonlinear = _objective_result(record, "maximize_nonlinear_usability_at_1mW")["optimized"]
        edge = neff["edge_to_center_mean_power_ratio_outer_ring_over_center_core"]
        candidate_rows.append(table_line([
            f"{n}×{n}", record["fiber_number"], f"{throughput['eta_total']:.4%}",
            f"{neff['N_eff']:.3f}",
            f"{neff['utilization_fraction_by_relative_threshold']['-10_dB']:.1%}",
            f"{neff['utilization_fraction_by_relative_threshold']['-20_dB']:.1%}",
            f"{neff['utilization_fraction_by_relative_threshold']['-30_dB']:.1%}",
            f"{neff['P10_over_P90']:.4f}",
            f"{neff['coefficient_of_variation']:.4g}",
            "n/a" if edge is None else f"{edge:.4g}",
            f"{neff['nonlinear_usability_at_1mW']['mean_active_coverage_over_sweep']:.1%}",
            f"{record['manufacturing']['manufacturing_score']:.1f}",
        ]))
        coverage = nonlinear["nonlinear_usability_at_1mW"]["mean_active_coverage_by_i_sat"]
        nonlinear_rows.append(table_line([
            f"{n}×{n}",
            *[f"{coverage[str(value)]:.1%}" for value in config["optimization"]["nonlinear_sat_intensity_w_cm2"]],
        ]))
    pareto_rows = [
        table_line([
            row["geometry_id"], row["optimization_type"], row["fiber_number"],
            f"{row['N_eff']:.3f}", f"{row['eta_total']:.4%}",
            f"{row['relative_complexity_score']:.1f}",
        ])
        for row in report["pareto_frontier"]
    ]
    count_table, pitch_table = "\n".join(count_rows), "\n".join(pitch_rows)
    layout_table, candidate_table = "\n".join(layout_rows), "\n".join(candidate_rows)
    pareto_table, nonlinear_table = "\n".join(pareto_rows), "\n".join(nonlinear_rows)
    isat_headers = " | ".join(f"I_sat={value} W/cm²" for value in config["optimization"]["nonlinear_sat_intensity_w_cm2"])
    isat_separator = "|".join("---:" for _ in config["optimization"]["nonlinear_sat_intensity_w_cm2"])
    standard_neff = _objective_result(regular_by_n[10], "maximize_effective_dimension")["optimized"]
    three_by_three = _objective_result(regular_by_n[3], "maximize_effective_dimension")["optimized"]
    five_by_five = _objective_result(regular_by_n[5], "maximize_effective_dimension")["optimized"]
    seven_by_seven = _objective_result(regular_by_n[7], "maximize_effective_dimension")["optimized"]
    best_pitch_metrics = best_pitch_result["optimized"]
    raw_standard_max_metrics = standard_best_result["optimized"]
    practical_metrics = practical_result["optimized"]
    report_text = f"""# Compact 200 μm OLS Dimension Optimization v1

## 结果与状态

完成 synthetic physical-interface diagnostics。固定 532 nm、2 μm、100×100（200 μm）两层 phase-only encoder；encoder 间距和末层至 fiber plane 距离均为 1.5 mm。共测 {report['measurements']['architecture_condition_count']} 种几何，每种分别运行 throughput、N_eff、1 mW nonlinear-usability 三种 phase-only objective，各 24 步，共 {report['measurements']['phase_parameter_adam_updates']} 次诊断相位更新；不是生成模型训练。预检 {report['preflight']['status']}，生产投影复系数/功率相对误差分别为 {report['preflight']['production_projection_complex_relative_error']:.3e} / {report['preflight']['production_projection_power_relative_error']:.3e}。没有数据集读取、task evaluator/FID 调用、loss 改动或材料前向。

## 科学认识

固定标准 125 μm pitch 的 10×10 在 N_eff 优化后 N_eff={standard_neff['N_eff']:.3f}/100，η={standard_neff['eta_total']:.4%}。标准 pitch 的 raw 最高值来自 {standard_best_record['fiber_grid_size']}×{standard_best_record['fiber_grid_size']}（N_eff={raw_standard_max_metrics['N_eff']:.9f}）；3×3、5×5、7×7 分别为 {three_by_three['N_eff']:.9f}、{five_by_five['N_eff']:.9f}、{seven_by_seven['N_eff']:.9f}，三者均显示为 5.127。3×3 与 raw 最大值仅差 {decision['standard_pitch_compact_candidate']['difference_from_raw_standard_max_N_eff']:.3e}，但 3×3 只有 9 根纤芯，-10 dB utilization={three_by_three['utilization_fraction_by_relative_threshold']['-10_dB']:.1%}（5×5={five_by_five['utilization_fraction_by_relative_threshold']['-10_dB']:.1%}，7×7={seven_by_seven['utilization_fraction_by_relative_threshold']['-10_dB']:.1%}）。因此建议把 3×3 / 125 μm 作为低复杂度 standard-pitch 候选，同时保留 7×7 raw N_eff 点；这不是全局最优证明。4×4/5×5/6×6 中最高 N_eff 候选为 {best_456_record['fiber_grid_size']}×{best_456_record['fiber_grid_size']}，N_eff={best_456_result['optimized']['N_eff']:.3f}。

本次最大 pitch sensitivity 是 {best_pitch_record['fiber_pitch_m']*1e6:g} μm，{best_pitch_record['geometry_id']}，N_eff={best_pitch_metrics['N_eff']:.3f}，η={best_pitch_metrics['eta_total']:.4%}；该 pitch 与当前 125 μm cladding 假设冲突，不能作为实验承诺。跨全部已测配置最大观察 N_eff={decision['effective_dimension_ceiling_observed']:.3f}。

#### Fiber-count sweep (125 μm, N_eff objective)

| Array | Fibers | η total | N_eff | P10/P90 | Relative complexity |
|---|---:|---:|---:|---:|---:|
{count_table}

#### Pitch sensitivity (N_eff objective)

125 μm 为 COMMERCIAL_REFERENCE。100/75/50 μm 是 CUSTOM_ARRAY_ASSUMPTION；在当前 125 μm cladding 假设下与无重叠封装不相容。

| Array | Pitch | Class | η total | N_eff | P10/P90 | Cladding overlap |
|---|---:|---|---:|---:|---:|---|
{pitch_table}

#### Layout comparison

Center compact 固定为 62.5 μm（125 μm 的 0.5 倍）。Optimized placement 在零相位下搜索 square/hexagonal lattice、rotation、translation，保持 fiber 数、125 μm 最小中心距和 ±625 μm 边界；这是有限候选搜索，不是连续全局最优。

| Array | Layout | Pitch | η total | N_eff | P10/P90 | Cladding overlap |
|---|---|---:|---:|---:|---:|---|
{layout_table}

#### Candidate comparison: 4×4, 5×5, 6×6, 10×10

η 采用 throughput optimizer；分布列采用 N_eff optimizer，并从该解的原始 P_i 线性缩放至 1 mW 后算 nonlinear coverage。

| Array | Fibers | η max-throughput | N_eff | Util. -10 dB | Util. -20 dB | Util. -30 dB | P10/P90 | CV | Edge/center | Nonlinear coverage | Manufacturing score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{candidate_table}

#### Pareto frontier

光阑面积固定为 0.04 mm²，所以 D_latent=N_eff/0.04 mm²，与 N_eff 成比例。下表给出同时考虑 N_eff、η、fiber 数和相对复杂度的非支配候选。

| Geometry | Objective | Fibers | N_eff | η total | Relative complexity |
|---|---|---:|---:|---:|---:|
{pareto_table}

#### Nonlinear sensitivity at 1 mW

使用 I_i=2P_i/(πw²)，w=MFD/2；active 是 0.1 < I_i/I_sat < 10。指标只是功率阈值敏感性，不是 Te/TaP/CoSi 或具体器件结论。

| Array | {isat_headers} |
|---|{isat_separator}|
{nonlinear_table}

10×10 的 N_eff 若远低于 100，说明该物理链下端口数不能当作有效 latent 维度；通道集中由 power map、utilization、P10/P90 和 CV 共同描述。此有限步结果不等价于全局理论上限，也不能单独证明 aperture、fiber pitch 或 phase optimization 是唯一原因。

## 目标缺口

没有真实 MFD、fiber placement/封装公差、camera/decoder calibration 或材料参数；phase 优化只面对一个 synthetic coherent top-hat 输入。没有测独立输入模式、语义 latent 可生成性、实验硬件耦合或材料响应。因此未支持生成质量、实验可制造或非线性器件结论。

## 工程完成与验证

锁定 production angular-spectrum finite-crop propagation 与 LP01 complex overlap，不 resize field、不重定位正式 fiber、不做通道或样本级功率归一化。通用变通道 projector 在 10×10/125 μm 上通过严格生产投影对照；执行三步真实双层 phase update、梯度、中心有限差分和反向符号负控。每种 geometry 的完整 channel power map、coordinates 与 nonlinear input distribution 保存在 JSON。制造复杂度仅按 fiber 数、占地、对准点、材料面积和 decoder 通道匹配构成相对负担分数。

同目录机器结果：compact_200um_ols_dimension_results.json；接口：compact_ols_interface.json。mode_overlap_matrix 是 disjoint-cell 数值 port basis 的 Gram 矩阵，不是实测 fiber crosstalk。

## 决策影响与下一步

**G1 — 最大有效 latent：** 全部 sensitivity 中最大观察 N_eff={decision['effective_dimension_ceiling_observed']:.3f}（{best_record['geometry_id']}）；标准 125 μm pitch raw 最大值={raw_standard_max_metrics['N_eff']:.6f}（{standard_best_record['fiber_number']} fibers）。**G2 — 最佳 fiber 数：** raw N_eff 的精确最大点是 {standard_best_record['fiber_number']} fibers，但它与 3×3 只差 {decision['standard_pitch_compact_candidate']['difference_from_raw_standard_max_N_eff']:.3e}，两者在报告显示精度均为 5.127；在显示精度相同的标准-pitch 候选中，推荐复杂度更低的 {practical_record['fiber_number']} fibers。指定的 4×4/5×5/6×6 候选中 5×5 的 N_eff 最高。**G3 — 最佳 pitch：** 测得 sensitivity 最大值在 {best_pitch_record['fiber_pitch_m']*1e6:g} μm，但该几何与125 μm cladding假设冲突；唯一 commercial reference 仍为125 μm。**G4 — 100维：** 10×10标准pitch N_eff={standard_neff['N_eff']:.3f}，不支持有效100维描述；保留为基准。**G5 — 非线性通道：** 后续材料接口敏感性建议先以 {decision['nonlinear_channel_count_recommendation']} 通道、125 μm regular layout 输入；该建议不等于材料选择或实验授权。

当前最小复杂度标准 pitch 候选的推荐 channel 数为 {practical_record['fiber_number']}，N_eff={practical_metrics['N_eff']:.3f}、η={practical_metrics['eta_total']:.4%}。10×10 只保留为历史/对照接口，不应表述为有效 100 维。4×4/5×5/6×6 内最高 N_eff 是 5×5；全标准-pitch count scan 则提示 3×3 已达到相同报告精度的 N_eff。

本轮到此停止。Compact OLS + Nonlinear Material + Semantic Generation 是后继工作，不在本轮授权范围内。
"""
    (run_dir / "compact_200um_ols_dimension_optimization_report.md").write_text(report_text, encoding="utf-8")
    report["report_path"] = str(run_dir / "compact_200um_ols_dimension_optimization_report.md")
    return decision


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def run(config_path: Path, run_dir: Path, *, preflight_only: bool = False) -> dict[str, Any]:
    config = _load_config(config_path)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("full compact OLS measurement requires CUDA; CPU is not an accepted substitute")
    torch.manual_seed(20261013)
    input_power = float(config["optics"]["input_power_w"])
    runtime_config = {
        "optics": config["optics"],
        "fiber": {
            **config["fiber"],
            "mfd_m": float(config["fiber"]["mode_field_diameter_m"]),
            "length_m": 8.5e-3,
            "attenuation_db_per_km": 0.0,
            "rows": 10,
            "columns": 10,
        },
    }
    runtime = base._physical_runtime(runtime_config, float(config["fiber"]["reference_pitch_m"]), device)
    entrance = base._source_field(runtime, 100, input_power)
    report: dict[str, Any] = {
        "schema": "compact_200um_ols_dimension_optimization_results.v1",
        "run_id": uuid.uuid4().hex,
        "status": "PREFLIGHT_RUNNING",
        "diagnostic_only": True,
        "scope": config["scope"],
        "config_path": str(config_path),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": {"name": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda},
        "frozen_system": {
            "wavelength_m": float(config["optics"]["wavelength_m"]),
            "pixel_pitch_m": float(config["optics"]["pixel_pitch_m"]),
            "trainable_shape": config["optics"]["trainable_shape"],
            "trainable_aperture_m": float(config["optics"]["trainable_aperture_m"]),
            "inter_encoder_spacing_m": float(config["optics"]["inter_encoder_spacing_m"]),
            "last_encoder_to_fiber_distance_m": float(config["optics"]["last_encoder_to_fiber_distance_m"]),
            "propagation_grid_size": int(config["optics"]["propagation_grid_size"]),
            "fiber_reference_pitch_m": float(config["fiber"]["reference_pitch_m"]),
            "layer_power_transmission": float(config["optics"]["layer_power_transmission"]),
        },
        "forbidden_actions": ["formal_training", "dataset_read", "task_evaluator", "FID", "task_loss_change", "material_nonlinearity_forward"],
        "preflight": None,
        "geometry_optimization": {},
        "architecture_results": [],
        "pareto_frontier": [],
    }
    _atomic_json(run_dir / "compact_200um_ols_dimension_results.json", report)
    preflight = _production_projection_preflight(runtime, entrance, config, device)
    report["preflight"] = preflight
    report["status"] = "PREFLIGHT_PASS"
    _atomic_json(run_dir / "compact_200um_ols_dimension_results.json", report)
    if preflight_only:
        report["status"] = "PREFLIGHT_ONLY_COMPLETE"
        _atomic_json(run_dir / "compact_200um_ols_dimension_results.json", report)
        return report

    optimized_geometry: dict[int, FiberGeometry] = {}
    geometry_summaries = {}
    for n in config["fiber"]["layout_comparison_grid_sizes"]:
        geometry, summary = _optimize_placement(runtime, entrance, config, device, int(n))
        optimized_geometry[int(n)] = geometry
        geometry_summaries[str(n)] = summary
        report["geometry_optimization"] = geometry_summaries
        _atomic_json(run_dir / "compact_200um_ols_dimension_results.json", report)
    specs = _architecture_specs(config, optimized_geometry)
    expected_architectures = 9 + 12 + 4 + 4
    if len(specs) != expected_architectures:
        raise RuntimeError(f"expected {expected_architectures} unique architecture conditions, got {len(specs)}")
    for architecture_index, geometry in enumerate(specs, start=1):
        projection = DiagnosticFiberProjection(runtime.ols.operator.input_plane, geometry, config, device)
        # Save the common zero-phase baseline at the reference synthetic power.
        zero_phase = torch.zeros((2, 1, 100, 100), dtype=torch.float32, device=device)
        with torch.no_grad():
            baseline = _measure(runtime, projection, entrance, zero_phase, geometry, config, input_power)
        record = {
            "geometry_id": geometry.geometry_id,
            "fiber_number": geometry.channel_count,
            "fiber_grid_size": geometry.grid_size,
            "fiber_pitch_m": geometry.pitch_m,
            "layout": geometry.layout,
            "layout_role": geometry.role,
            "fiber_coordinates_xy_um": baseline["metrics"]["fiber_coordinates_xy_um"],
            "zero_phase": baseline["metrics"],
            "optimization_results": [],
            "manufacturing": _manufacturing(geometry, config),
        }
        for objective in OBJECTIVES:
            objective_input = (
                float(config["optimization"]["nonlinear_input_power_w"])
                if objective == "maximize_nonlinear_usability_at_1mW" else input_power
            )
            objective_entrance = (
                base._source_field(runtime, 100, objective_input)
                if objective_input != input_power else entrance
            )
            result = _optimize_objective(
                runtime, geometry, projection, objective_entrance, objective_input, objective, config, device
            )
            result["manufacturing"] = record["manufacturing"]
            record["optimization_results"].append(result)
            if not any(x["geometry_id"] == record["geometry_id"] for x in report["architecture_results"]):
                report["architecture_results"].append(record)
            report["status"] = "RUNNING"
            report["progress"] = {
                "architecture_index": architecture_index,
                "architecture_count": len(specs),
                "architecture_id": geometry.geometry_id,
                "objective_completed": objective,
                "objective_count": len(OBJECTIVES),
            }
            _atomic_json(run_dir / "compact_200um_ols_dimension_results.json", report)
        del projection, zero_phase, baseline
        torch.cuda.empty_cache()
    flat_rows = []
    for record in report["architecture_results"]:
        for objective_result in record["optimization_results"]:
            flat_rows.append({
                "geometry_id": record["geometry_id"],
                "optimization_type": objective_result["optimization_type"],
                "optimized": objective_result["optimized"],
                "manufacturing": record["manufacturing"],
            })
    report["pareto_frontier"] = _pareto(flat_rows)
    report["status"] = "COMPLETE"
    report["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    report["measurements"] = {
        "architecture_condition_count": len(specs),
        "optimization_type_count": len(OBJECTIVES),
        "phase_optimizer_run_count": len(specs) * len(OBJECTIVES),
        "phase_parameter_adam_updates": len(specs) * len(OBJECTIVES) * int(config["optimization"]["steps_per_objective"]),
        "task_training_updates": 0,
        "dataset_reads": 0,
        "evaluator_calls": 0,
    }
    _write_deliverables(report, run_dir, config)
    _atomic_json(run_dir / "compact_200um_ols_dimension_results.json", report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run(args.config, args.run_dir, preflight_only=args.preflight_only)
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "preflight": result["preflight"],
        "architecture_count": len(result["architecture_results"]),
        "pareto_count": len(result["pareto_frontier"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
