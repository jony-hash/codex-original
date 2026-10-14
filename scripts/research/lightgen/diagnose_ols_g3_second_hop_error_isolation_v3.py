"""Zero-update localization of the fixed-window second-hop OLS G3 residual."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import (
    GaussianFiberArrayOLS,
    GaussianFiberGeometry,
)
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    FiberArrayOLS,
    FiniteCropPhaseStack,
    LightGen2DSemanticGenerationInputOLSAlignment,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics.propagation import (
    angular_spectrum_propagate,
    clear_production_transfer_cache,
)
from scripts.research.lightgen import diagnose_ols_gaussian_dimension_reaudit_v1 as baseline


SCHEMA = "ols_g3_second_hop_error_isolation.v3"
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_g3_second_hop_error_isolation_v3.yaml")
DEFAULT_PHASES = Path(
    "outputs/20261014_ols_gaussian_zero_update_closeout_v2/inputs/fixed_phase_screens.pt"
)
DEFAULT_HISTORICAL_REPORT = Path(
    "outputs/20261014_ols_g3_second_hop_error_isolation_v3/inputs/historical_g3_report.json"
)
DEFAULT_RUN_DIR = Path(
    "outputs/20261014_ols_g3_second_hop_error_isolation_v3/runs/attempt_01"
)
PHASE_NAMES = ("zero", "last_layer_x_1deg", "last_layer_xy_1deg")
GRID_SIZES = (2048, 3072, 4096)
DX_M = 2.0e-6
WAVELENGTH_M = 532.0e-9
LAYER_TRANSMISSION = 0.889
SOURCE_POWER_W = 249.3e-6
ROI_SIZE = 2048
G3_TOLERANCE = 1.0e-4
REFERENCE_TOLERANCE = 1.0e-7
EPS = 1.0e-30


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} configuration")
    expected = {
        "wavelength_m": WAVELENGTH_M,
        "pixel_pitch_m": DX_M,
        "encoder_shape": [100, 100],
        "encoder_aperture_m": 200.0e-6,
        "propagation_windows_mm": [4.096, 6.144, 8.192],
        "propagation_grid_sizes": list(GRID_SIZES),
        "inter_encoder_spacing_m": 1.5e-3,
        "last_encoder_to_fiber_m": 1.5e-3,
        "phase_screen_count": 2,
        "layer_power_transmission": LAYER_TRANSMISSION,
        "input_power_w": SOURCE_POWER_W,
    }
    for key, expected_value in expected.items():
        if value["optics"].get(key) != expected_value:
            raise ValueError(f"frozen optics.{key} changed")
    if value["fiber"] != {
        "reference_pitch_m": 125.0e-6,
        "mode_field_diameter_m": 14.222e-6,
        "gaussian_1e_amplitude_radius_m": 7.111e-6,
        "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
    }:
        raise ValueError("frozen fiber geometry or unmeasured status changed")
    if value["reference"] != {
        "type": "nonperiodic_bandlimited_sampled_field_angular_spectrum_integral",
        "dtype": "complex128",
        "quadrature": "composite_gauss_legendre",
        "quadrature_order_per_frequency_cell": 4,
        "frequency_nodes_per_axis": [512, 1024, 2048, 4096],
        "roi_probe_indices": [128, 512, 1023, 1535, 1919],
        "fiber_center_probe_count": 100,
        "convergence_tolerance": REFERENCE_TOLERANCE,
        "reference_field_scale": "sqrt_input_power_over_200um",
        "include_pixel_sinc_factor": False,
        "apply_additional_spectral_filter": False,
        "expand_spatial_propagation_grid": False,
    }:
        raise ValueError("independent reference definition changed")
    if value["gates"] != {
        "g3_complex_relative_error_after_common_global_phase_max": G3_TOLERANCE,
        "production_dtype": "complex64",
        "high_precision_diagnostic_dtype": "complex128",
    }:
        raise ValueError("G3 gate or precision definitions changed")
    if value["scope"] != {
        "formal_training_updates": 0,
        "phase_optimizer_updates": 0,
        "dataset_reads": 0,
        "task_evaluator_calls": 0,
        "fid_calls": 0,
        "loss_changes": 0,
        "material_forward_calls": 0,
        "propagation_solver_replacement": "forbidden",
        "optical_geometry_change": "forbidden",
        "hidden_grid_expansion": "forbidden",
        "per_sample_or_per_port_renormalization": "forbidden",
    }:
        raise ValueError("scope contract changed")
    return value


def _load_phases(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"explicit fixed-phase source is missing: {path}")
    phases = torch.load(path, map_location="cpu", weights_only=True)
    missing = sorted(set(PHASE_NAMES) - set(phases))
    if missing:
        raise ValueError(f"fixed-phase file is missing required cases: {missing}")
    result = {name: phases[name].detach().cpu() for name in PHASE_NAMES}
    for name, phase in result.items():
        if tuple(phase.shape) != (2, 1, 100, 100) or not torch.isfinite(phase).all():
            raise ValueError(f"invalid saved phase tensor for {name}: {tuple(phase.shape)}")
    for name in PHASE_NAMES[1:]:
        if not torch.equal(result["zero"][0], result[name][0]):
            raise ValueError("the three fixed cases do not share one first-layer phase screen")
    return result


def _frequency_quadrature(node_count: int) -> tuple[np.ndarray, np.ndarray]:
    order = 4
    if node_count <= 0 or node_count % order:
        raise ValueError("frequency node count must be a positive multiple of four")
    nyquist = 1.0 / (2.0 * DX_M)
    interval_count = node_count // order
    interval_width = (2.0 * nyquist) / interval_count
    nodes, weights = np.polynomial.legendre.leggauss(order)
    centers = -nyquist + (np.arange(interval_count, dtype=np.float64) + 0.5) * interval_width
    frequencies = (centers[:, None] + 0.5 * interval_width * nodes[None, :]).reshape(-1)
    quadrature_weights = (0.5 * interval_width * weights[None, :]).repeat(interval_count, axis=0).reshape(-1)
    if frequencies.size != node_count or not math.isclose(
        float(quadrature_weights.sum()), 2.0 * nyquist, rel_tol=1.0e-14
    ):
        raise RuntimeError("frequency quadrature nodes or total integration measure are invalid")
    return frequencies, quadrature_weights


def _bandlimited_integral(
    field: np.ndarray,
    distance_m: float,
    positions_xy_m: np.ndarray,
    node_count: int,
    *,
    y_chunk: int = 256,
    point_batch: int = 16,
) -> np.ndarray:
    """CPU complex128 quadrature of the stated nonperiodic sampled-field integral."""
    source = np.asarray(field, dtype=np.complex128)
    if source.shape != (100, 100) or positions_xy_m.ndim != 2 or positions_xy_m.shape[1] != 2:
        raise ValueError("reference input must be 100x100 with XY sample positions")
    frequencies, weights = _frequency_quadrature(node_count)
    source_axis = (np.arange(100, dtype=np.float64) + 0.5) * DX_M - 100.0 * DX_M / 2.0
    exponent_x = np.exp(-2j * np.pi * frequencies[:, None] * source_axis[None, :])
    exponent_y = exponent_x
    fourier_x = source @ exponent_x.T
    x_positions = positions_xy_m[:, 0]
    y_positions = positions_xy_m[:, 1]
    output = np.zeros(positions_xy_m.shape[0], dtype=np.complex128)
    wavelength_squared = WAVELENGTH_M**2
    phase_scale = 2.0 * math.pi * distance_m / WAVELENGTH_M
    area = DX_M**2

    for y_start in range(0, node_count, y_chunk):
        y_stop = min(y_start + y_chunk, node_count)
        fy = frequencies[y_start:y_stop]
        spectrum = exponent_y[y_start:y_stop] @ fourier_x
        radicand = 1.0 - wavelength_squared * (
            fy[:, None] ** 2 + frequencies[None, :] ** 2
        )
        transfer = np.exp(1j * phase_scale * np.sqrt(np.maximum(radicand, 0.0)))
        transfer *= radicand >= 0.0
        weighted = (
            spectrum
            * transfer
            * weights[y_start:y_stop, None]
            * weights[None, :]
            * area
        )
        for point_start in range(0, positions_xy_m.shape[0], point_batch):
            point_stop = min(point_start + point_batch, positions_xy_m.shape[0])
            x_batch = x_positions[point_start:point_stop]
            y_batch = y_positions[point_start:point_stop]
            inverse_x = np.exp(2j * np.pi * frequencies[:, None] * x_batch[None, :])
            inverse_y = np.exp(2j * np.pi * fy[:, None] * y_batch[None, :])
            transformed = weighted @ inverse_x
            output[point_start:point_stop] += np.sum(transformed * inverse_y, axis=0)
    return output


def _relative_complex_error(reference: np.ndarray, test: np.ndarray) -> float:
    numerator = np.linalg.norm(np.asarray(reference) - np.asarray(test))
    denominator = max(float(np.linalg.norm(reference)), EPS)
    return float(numerator / denominator)


def _preflight_integral_checks() -> dict[str, Any]:
    grid_axis = (np.arange(100, dtype=np.float64) + 0.5) * DX_M - 100.0 * DX_M / 2.0
    center_index = 50
    x_center = float(grid_axis[center_index])
    y_center = float(grid_axis[center_index])
    sample = np.zeros((100, 100), dtype=np.complex128)
    sample[center_index, center_index] = 1.0 + 0.0j
    point = np.asarray([[x_center + 0.37 * DX_M, y_center - 0.21 * DX_M]])
    observed = _bandlimited_integral(sample, 0.0, point, 512)[0]
    expected = np.sinc(0.37) * np.sinc(-0.21)
    sinc_error = abs(observed - expected)
    omitted_area_observed = observed / DX_M**2
    omitted_area_negative_error = abs(omitted_area_observed - expected) / max(abs(expected), EPS)

    coherent = np.zeros((100, 100), dtype=np.complex128)
    coherent[center_index, center_index] = 1.0 + 0.0j
    coherent[center_index, center_index + 1] = 0.0 + 1.0j
    coherent_point = np.asarray([[x_center + 0.5 * DX_M, y_center]])
    coherent_observed = _bandlimited_integral(coherent, 0.0, coherent_point, 512)[0]
    coherent_expected = np.sinc(0.5) + 1j * np.sinc(-0.5)
    coherent_error = abs(coherent_observed - coherent_expected)
    lost_phase_error = abs(coherent_observed.real - coherent_expected) / abs(coherent_expected)

    shifted_point = np.asarray([[x_center + 0.5 * DX_M, y_center]])
    shifted_observed = _bandlimited_integral(sample, 0.0, shifted_point, 512)[0]
    shifted_coordinate_error = abs(shifted_observed - 1.0)
    omitted_area_ratio = 1.0 / DX_M**2
    passed = (
        sinc_error <= 1.0e-10
        and coherent_error <= 1.0e-10
        and lost_phase_error >= 0.5
        and omitted_area_negative_error > 1.0e6
        and shifted_coordinate_error >= 0.3
        and math.isclose(omitted_area_ratio, 2.5e11, rel_tol=1.0e-14)
    )
    if not passed:
        raise RuntimeError("independent integral sinc/phase/coordinate/area preflight failed")
    return {
        "status": "PASS",
        "single_pixel_zero_distance_sinc_absolute_error": float(sinc_error),
        "coherent_two_pixel_known_phase_absolute_error": float(coherent_error),
        "dropped_imaginary_phase_negative_control_relative_error": float(lost_phase_error),
        "dropped_complex_phase_rejected": lost_phase_error >= 0.5,
        "omitted_area_negative_control_relative_error": float(omitted_area_negative_error),
        "omitted_pixel_area_rejected": omitted_area_negative_error > 1.0e6,
        "half_pixel_center_shift_negative_control_relative_error": float(shifted_coordinate_error),
        "wrong_pixel_center_rejected": shifted_coordinate_error >= 0.3,
        "omitted_pixel_area_scale_ratio": omitted_area_ratio,
        "pixel_area_included_in_integral": True,
        "quadrature_weight_sum_per_axis": 1.0 / DX_M,
    }


def _runtime_phase(phase: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return phase.to(device=device, dtype=dtype)


def _first_hop(
    runtime,
    entrance: ComplexField,
    first_phase: torch.Tensor,
    grid: int,
    prefix: str,
) -> tuple[ComplexField, torch.Tensor]:
    active_shape = tuple(entrance.plane.shape)
    phase = first_phase.to(device=entrance.tensor.device, dtype=entrance.tensor.real.dtype)
    transfer = torch.polar(torch.full_like(phase, math.sqrt(LAYER_TRANSMISSION)), phase).to(
        entrance.tensor.dtype
    )
    modulated = entrance.tensor * transfer[None, None]
    padding_y = (grid - active_shape[0]) // 2
    padding_x = (grid - active_shape[1]) // 2
    padded_tensor = F.pad(modulated, (padding_x, padding_x, padding_y, padding_y))
    padded_plane = entrance.plane.with_plane(
        f"{prefix}_padded",
        shape=(grid, grid),
        coordinate_frame=f"{prefix}_grid",
    )
    propagated = angular_spectrum_propagate(
        ComplexField(
            padded_tensor,
            padded_plane,
            OperationProvenance("g3_v3_first_layer_center_pad", entrance.plane.plane_id, None),
        ),
        1.5e-3,
        f"{prefix}_propagated",
    )
    crop_y = (grid - active_shape[0]) // 2
    crop_x = (grid - active_shape[1]) // 2
    middle_tensor = propagated.tensor[
        ..., crop_y : crop_y + active_shape[0], crop_x : crop_x + active_shape[1]
    ]
    middle_plane = entrance.plane.with_plane(
        f"{prefix}_aperture_crop",
        z_m=propagated.plane.z_m,
    )
    middle = ComplexField(
        middle_tensor,
        middle_plane,
        OperationProvenance(
            "g3_v3_first_hop_hard_aperture_crop",
            propagated.plane.plane_id,
            1.5e-3,
            {"active_shape": list(active_shape)},
        ),
    )
    roi_offset = (grid - ROI_SIZE) // 2
    common_roi = propagated.tensor[
        ..., roi_offset : roi_offset + ROI_SIZE, roi_offset : roi_offset + ROI_SIZE
    ].detach().cpu()
    return middle, common_roi


def _second_hop(
    runtime,
    middle: ComplexField,
    last_phase: torch.Tensor,
    grid: int,
    prefix: str,
) -> ComplexField:
    phase = last_phase.to(device=middle.tensor.device, dtype=middle.tensor.real.dtype)
    transfer = torch.polar(torch.full_like(phase, math.sqrt(LAYER_TRANSMISSION)), phase).to(
        middle.tensor.dtype
    )
    modulated = middle.tensor * transfer[None, None]
    padding_y = (grid - middle.plane.shape[0]) // 2
    padding_x = (grid - middle.plane.shape[1]) // 2
    padded_tensor = F.pad(modulated, (padding_x, padding_x, padding_y, padding_y))
    padded_plane = middle.plane.with_plane(
        f"{prefix}_padded",
        shape=(grid, grid),
        coordinate_frame=f"{prefix}_grid",
    )
    return angular_spectrum_propagate(
        ComplexField(
            padded_tensor,
            padded_plane,
            OperationProvenance("g3_v3_second_layer_center_pad", middle.plane.plane_id, None),
        ),
        1.5e-3,
        f"{prefix}_propagated",
    )


def _preflight(config: dict[str, Any], phase_path: Path, device: torch.device) -> dict[str, Any]:
    phases = _load_phases(phase_path)
    baseline_config_path = Path(config["source_evidence"]["baseline_diagnostic_config"])
    baseline_config = baseline._load_config(baseline_config_path)
    plane_wave = baseline._analytic_plane_wave_check(device)
    integral = _preflight_integral_checks()

    grid = 2048
    runtime = baseline._make_runtime(grid, torch.float64, device)
    entrance = baseline._encoder_entrance(runtime, torch.float64, device)
    full_output = baseline._run_encoder(
        runtime, entrance, _runtime_phase(phases["zero"], torch.float64, device), "g3_v3_preflight_full"
    )
    middle, _first_roi = _first_hop(
        runtime,
        entrance,
        phases["zero"][0, 0],
        grid,
        "g3_v3_preflight_first",
    )
    second = _second_hop(
        runtime,
        middle,
        phases["zero"][1, 0],
        grid,
        "g3_v3_preflight_second",
    )
    standalone_error = baseline._rel_l2(full_output.tensor, second.tensor)
    if standalone_error > 1.0e-12:
        raise RuntimeError(
            f"standalone second-hop call does not reproduce production stack: {standalone_error:.6g}"
        )

    phase_screen_common_first_layer = all(
        torch.equal(phases["zero"][0], phases[name][0]) for name in PHASE_NAMES
    )
    if not phase_screen_common_first_layer:
        raise RuntimeError("fixed phase cases do not share the expected first-layer state")
    return {
        "status": "PASS",
        "production_device": str(device),
        "production_gpu": torch.cuda.get_device_name(device),
        "baseline_config_validated": str(baseline_config_path),
        "fixed_phase_source": str(phase_path),
        "phase_shapes": {name: list(phases[name].shape) for name in PHASE_NAMES},
        "phase_first_layer_identical_across_three_cases": phase_screen_common_first_layer,
        "phase_optimizer_updates": 0,
        "analytic_production_plane_wave": plane_wave,
        "independent_integral_negative_controls": integral,
        "full_two_layer_vs_standalone_second_hop_relative_error": standalone_error,
        "second_hop_production_call_verified": True,
        "formal_training_updates": 0,
        "dataset_reads": 0,
        "task_evaluator_calls": 0,
        "fid_calls": 0,
        "loss_changes": 0,
        "material_forward_calls": 0,
        "propagation_solver_replacement": False,
    }


def _fiber_probe_points(device: torch.device) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime = baseline._make_runtime(ROI_SIZE, torch.float64, device)
    operator = GaussianFiberArrayOLS(
        runtime.ols_plane,
        GaussianFiberGeometry(port_pitch_m=125.0e-6, waist_m=7.111e-6),
    ).to(device=device)
    centers_yx = operator.centers_m.detach().cpu().numpy()
    roi_axis = (np.arange(ROI_SIZE, dtype=np.float64) + 0.5) * DX_M - ROI_SIZE * DX_M / 2.0
    points: list[dict[str, Any]] = []
    for row_index in (128, 512, 1023, 1535, 1919):
        for column_index in (128, 512, 1023, 1535, 1919):
            points.append(
                {
                    "group": "roi_grid_25",
                    "fiber_index": None,
                    "row_index": row_index,
                    "column_index": column_index,
                    "x_m": float(roi_axis[column_index]),
                    "y_m": float(roi_axis[row_index]),
                }
            )
    fiber_points: list[dict[str, Any]] = []
    for fiber_index, (center_y, center_x) in enumerate(centers_yx):
        row_index = int(np.argmin(np.abs(roi_axis - center_y)))
        column_index = int(np.argmin(np.abs(roi_axis - center_x)))
        fiber_points.append(
            {
                "group": "fiber_center_nearest_roi_pixel_100",
                "fiber_index": fiber_index,
                "fiber_center_x_m": float(center_x),
                "fiber_center_y_m": float(center_y),
                "row_index": row_index,
                "column_index": column_index,
                "x_m": float(roi_axis[column_index]),
                "y_m": float(roi_axis[row_index]),
                "nearest_pixel_offset_x_m": float(roi_axis[column_index] - center_x),
                "nearest_pixel_offset_y_m": float(roi_axis[row_index] - center_y),
            }
        )
    points.extend(fiber_points)
    positions = np.asarray([[item["x_m"], item["y_m"]] for item in points], dtype=np.float64)
    selection = {
        "roi_point_count": 25,
        "fiber_point_count": len(fiber_points),
        "selection_rule": "nearest physical common-ROI pixel center; exact ties select the lowest row/column index",
        "pixel_pitch_m": DX_M,
        "common_roi_extent_m": ROI_SIZE * DX_M,
        "fiber_centers_xy_row_major": [
            {"x_m": item["fiber_center_x_m"], "y_m": item["fiber_center_y_m"]}
            for item in fiber_points
        ],
        "fiber_center_nearest_pixel_offsets_xy_m": [
            {"x_m": item["nearest_pixel_offset_x_m"], "y_m": item["nearest_pixel_offset_y_m"]}
            for item in fiber_points
        ],
    }
    selection["points"] = points
    selection["positions_xy_m"] = positions
    return points, selection


def _roi_sample(field: torch.Tensor, points: list[dict[str, Any]]) -> np.ndarray:
    values = field.detach().cpu().numpy()
    return np.asarray(
        [values[item["row_index"], item["column_index"]] for item in points], dtype=np.complex128
    )


def _reference_error(reference: np.ndarray, test: np.ndarray, field_scale: float) -> dict[str, float]:
    difference = np.asarray(test, dtype=np.complex128) - np.asarray(reference, dtype=np.complex128)
    intensity_reference = np.abs(reference) ** 2
    intensity_test = np.abs(test) ** 2
    return {
        "raw_complex_relative_l2_error_no_phase_fit": _relative_complex_error(reference, test),
        "max_abs_complex_error_over_E0": float(np.max(np.abs(difference)) / field_scale),
        "intensity_relative_l2_error": float(
            np.linalg.norm(intensity_test - intensity_reference)
            / max(float(np.linalg.norm(intensity_reference)), EPS)
        ),
    }


def _cpu_reference(
    config: dict[str, Any],
    phases: dict[str, torch.Tensor],
    common_middle: torch.Tensor,
    positions: np.ndarray,
    points: list[dict[str, Any]],
    fixed_outputs: dict[str, dict[str, dict[int, torch.Tensor]]],
    run_dir: Path,
) -> dict[str, Any]:
    input_field = common_middle.detach().cpu().numpy().astype(np.complex128, copy=False)
    field_scale = math.sqrt(SOURCE_POWER_W) / 200.0e-6
    node_counts = config["reference"]["frequency_nodes_per_axis"]
    all_results: dict[str, Any] = {}
    production_comparisons: dict[str, Any] = {}
    last_refinement: dict[str, Any] = {}
    for phase_name in PHASE_NAMES:
        last_phase = phases[phase_name][1, 0].numpy().astype(np.float64, copy=False)
        active = input_field * math.sqrt(LAYER_TRANSMISSION) * np.exp(1j * last_phase)
        by_nodes: dict[str, np.ndarray] = {}
        refinement_records: dict[str, Any] = {}
        for node_count in node_counts:
            values = _bandlimited_integral(
                active,
                1.5e-3,
                positions,
                int(node_count),
            )
            if not np.isfinite(values).all():
                raise RuntimeError(f"independent reference returned nonfinite values at {node_count} nodes")
            by_nodes[str(node_count)] = values
        group_slices = {"roi_grid_25": slice(0, 25), "fiber_nearest_pixel_100": slice(25, 125), "combined_125": slice(0, 125)}
        for group, selected in group_slices.items():
            previous = by_nodes[str(node_counts[-2])][selected]
            latest = by_nodes[str(node_counts[-1])][selected]
            difference = latest - previous
            refinement_records[group] = {
                "relative_l2_difference_last_two_levels": _relative_complex_error(previous, latest),
                "maximum_point_difference_over_E0": float(np.max(np.abs(difference)) / field_scale),
                "relative_l2_tolerance": REFERENCE_TOLERANCE,
                "max_point_over_E0_tolerance": REFERENCE_TOLERANCE,
                "converged": bool(
                    _relative_complex_error(previous, latest) <= REFERENCE_TOLERANCE
                    and np.max(np.abs(difference)) / field_scale <= REFERENCE_TOLERANCE
                ),
            }
        refinement_records["all_groups_converged"] = all(
            refinement_records[group]["converged"] for group in group_slices
        )
        all_results[phase_name] = {
            "quadrature_values_at_selected_points": {
                node: [
                    {"real": float(value.real), "imag": float(value.imag)}
                    for value in values
                ]
                for node, values in by_nodes.items()
            },
            "refinement": refinement_records,
        }
        last_refinement[phase_name] = refinement_records
        latest = by_nodes[str(node_counts[-1])]
        production_comparisons[phase_name] = {}
        for precision in ("complex64", "complex128"):
            production_comparisons[phase_name][precision] = {}
            for grid in GRID_SIZES:
                sample = _roi_sample(fixed_outputs[precision][phase_name][grid], points)
                production_comparisons[phase_name][precision][str(grid)] = {
                    "roi_grid_25": _reference_error(latest[:25], sample[:25], field_scale),
                    "fiber_nearest_pixel_100": _reference_error(latest[25:], sample[25:], field_scale),
                    "combined_125": _reference_error(latest, sample, field_scale),
                    "global_phase_alignment_applied": False,
                }
    convergence_valid = all(
        bool(last_refinement[phase]["all_groups_converged"]) for phase in PHASE_NAMES
    )
    result = {
        "status": "VALID" if convergence_valid else "INVALID_NONCONVERGED_REFERENCE",
        "definition": (
            "CPU complex128 composite four-point Gauss-Legendre evaluation of the nonperiodic bandlimited sampled-field "
            "angular-spectrum integral over the fixed Nyquist square; no pixel sinc factor, extra filter, or spatial-grid expansion"
        ),
        "distance_m": 1.5e-3,
        "pixel_pitch_m": DX_M,
        "wavelength_m": WAVELENGTH_M,
        "frequency_nodes_per_axis": node_counts,
        "reference_field_scale_E0": field_scale,
        "selected_positions": [
            {key: value for key, value in item.items() if key not in {"group"}}
            | {"group": item["group"]}
            for item in points
        ],
        "convergence_by_phase": last_refinement,
        "production_selected_point_comparisons": production_comparisons,
        "global_phase_alignment_applied": False,
        "all_three_phase_references_converged": convergence_valid,
    }
    _write_json(run_dir / "independent_reference_convergence.json", result)
    return result


def _power(field: torch.Tensor) -> float:
    return float(field.abs().square().sum().to(torch.float64) * DX_M**2)


def _mode_readout(
    field_roi: torch.Tensor,
    precision: str,
    phase_name: str,
    grid: int,
    device: torch.device,
) -> dict[str, Any]:
    real_dtype = torch.float32 if precision == "complex64" else torch.float64
    runtime = baseline._make_runtime(ROI_SIZE, real_dtype, device)
    mode_field = ComplexField(
        field_roi.to(device=device)[None, None],
        runtime.ols_plane,
        OperationProvenance("g3_v3_fixed_second_hop_mode_readout", runtime.ols_plane.plane_id, None),
    )
    records: dict[str, Any] = {}
    for mode_name in ("gaussian_s12", "lp01"):
        coefficient = baseline._mode_coefficients(mode_field, runtime, mode_name)[0].detach().cpu().to(
            torch.complex128
        )
        power = coefficient.abs().square()
        total = float(power.sum())
        before_fiber = _power(field_roi)
        records[mode_name] = {
            "complex_coefficients_sqrt_w_row_major": [
                {"real": float(item.real), "imag": float(item.imag)} for item in coefficient
            ],
            "per_fiber_power_w_row_major": power.tolist(),
            "total_coupling_w": total,
            "eta_fiber_face": total / max(before_fiber, EPS),
            "eta_system_input_to_fiber": total / SOURCE_POWER_W,
        }
    return {
        "phase": phase_name,
        "precision": precision,
        "grid_size": grid,
        "power_before_fiber_w_common_roi": _power(field_roi),
        "modes": records,
    }


def _error_map_records(
    fields: dict[str, Any],
    run_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    axis = (np.arange(ROI_SIZE, dtype=np.float64) + 0.5) * DX_M - ROI_SIZE * DX_M / 2.0
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    center_mask = (np.abs(xx) <= 100.0e-6) & (np.abs(yy) <= 100.0e-6)
    array_mask = (np.abs(xx) <= 625.0e-6) & (np.abs(yy) <= 625.0e-6)
    masks = {
        "center_200um_square": center_mask,
        "fiber_array_1p25mm_square": array_mask,
        "remainder_of_common_roi": ~array_mask,
    }
    comparisons: dict[str, Any] = {}
    maps: dict[str, np.ndarray] = {}
    for precision in ("complex64", "complex128"):
        comparisons[precision] = {}
        for phase_name in PHASE_NAMES:
            comparisons[precision][phase_name] = {}
            for reference_grid, test_grid in ((2048, 3072), (2048, 4096), (3072, 4096)):
                reference = fields[phase_name][precision][str(reference_grid)].to(torch.complex128)
                test = fields[phase_name][precision][str(test_grid)].to(torch.complex128)
                phase, aligned = baseline._phase_alignment(reference, test)
                record: dict[str, Any] = {
                    "reference_grid": reference_grid,
                    "test_grid": test_grid,
                    "common_global_phase_rad": phase,
                    "raw_complex_relative_l2_error": baseline._rel_l2(reference, test),
                    "aligned_complex_relative_l2_error": baseline._rel_l2(reference, aligned),
                    "error_energy_fraction_by_region": {"raw": {}, "aligned": {}},
                    "error_energy_relative_to_reference_by_region": {"raw": {}, "aligned": {}},
                }
                ref_norm_sq = float(reference.abs().square().sum())
                for kind, candidate in (("raw", test), ("aligned", aligned)):
                    diff = (reference - candidate).abs()
                    maps[
                        f"{precision}_{phase_name}_{reference_grid}_vs_{test_grid}_{kind}_abs_error"
                    ] = diff.detach().cpu().numpy().astype(np.float32)
                    squared = diff.square()
                    total_error = float(squared.sum())
                    for region, region_mask_np in masks.items():
                        region_mask = torch.from_numpy(region_mask_np)
                        region_error = float(squared[region_mask].sum())
                        record["error_energy_fraction_by_region"][kind][region] = (
                            region_error / max(total_error, EPS)
                        )
                        record["error_energy_relative_to_reference_by_region"][kind][region] = (
                            region_error / max(ref_norm_sq, EPS)
                        )
                comparisons[precision][phase_name][f"{reference_grid}_vs_{test_grid}"] = record
    map_path = run_dir / "g3_error_maps.npz"
    np.savez_compressed(map_path, **maps)
    return {
        "status": "COMPLETE",
        "map_file": map_path.name,
        "map_count": len(maps),
        "map_encoding": "float32 absolute complex-field difference in V/m per common-ROI pixel",
        "region_definitions": {
            "center_200um_square": "abs(x)<=100 um and abs(y)<=100 um on common-ROI pixel centers",
            "fiber_array_1p25mm_square": "abs(x)<=625 um and abs(y)<=625 um on common-ROI pixel centers",
            "remainder_of_common_roi": "complement of the 1.25 mm square within the 4.096 mm ROI",
        },
        "comparisons": comparisons,
    }, maps


def _reconstruct_middle_fields(
    phases: dict[str, torch.Tensor],
    device: torch.device,
    run_dir: Path,
) -> tuple[dict[str, Any], torch.Tensor, dict[str, Any]]:
    stages: dict[str, Any] = {"complex64": {}, "complex128": {}}
    for precision, real_dtype in (("complex64", torch.float32), ("complex128", torch.float64)):
        for phase_name in PHASE_NAMES:
            stages[precision][phase_name] = {}
            for grid in GRID_SIZES:
                runtime = baseline._make_runtime(grid, real_dtype, device)
                entrance = baseline._encoder_entrance(runtime, real_dtype, device)
                middle, first_roi = _first_hop(
                    runtime,
                    entrance,
                    phases[phase_name][0, 0],
                    grid,
                    f"g3_v3_first_{precision}_{phase_name}_{grid}",
                )
                stages[precision][phase_name][grid] = {
                    "middle_aperture": middle.tensor.detach().cpu()[0, 0],
                    "middle_plane": middle.plane,
                    "first_common_roi": first_roi[0, 0],
                }
                del runtime, entrance, middle, first_roi
                clear_production_transfer_cache()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    stage_payload: dict[str, Any] = {"schema": "ols_g3_v3_saved_first_hop_fields.v1"}
    stage_summary: dict[str, Any] = {"first_hop_comparisons": {}}
    for precision in ("complex64", "complex128"):
        stage_summary["first_hop_comparisons"][precision] = {}
        for phase_name in PHASE_NAMES:
            stage_summary["first_hop_comparisons"][precision][phase_name] = {}
            for grid in GRID_SIZES:
                data = stages[precision][phase_name][grid]
                stage_payload.setdefault(phase_name, {}).setdefault(precision, {})[str(grid)] = {
                    "intermediate_200um_aperture_crop": data["middle_aperture"],
                    "first_propagation_common_roi": data["first_common_roi"],
                    "middle_plane": data["middle_plane"].to_dict(),
                }
            for reference_grid, test_grid in ((2048, 3072), (2048, 4096), (3072, 4096)):
                records = {}
                for stage_name in ("first_common_roi", "middle_aperture"):
                    reference = stages[precision][phase_name][reference_grid][stage_name]
                    test = stages[precision][phase_name][test_grid][stage_name]
                    phase, aligned = baseline._phase_alignment(reference, test)
                    records[stage_name] = {
                        "raw_complex_relative_l2_error": baseline._rel_l2(reference, test),
                        "aligned_complex_relative_l2_error": baseline._rel_l2(reference, aligned),
                        "common_global_phase_rad": phase,
                        "intensity_relative_l2_error": baseline._rel_l2(
                            reference.abs().square(), test.abs().square()
                        ),
                        "power_relative_error": abs(_power(test) - _power(reference))
                        / max(_power(reference), EPS),
                    }
                stage_summary["first_hop_comparisons"][precision][phase_name][
                    f"{reference_grid}_vs_{test_grid}"
                ] = records
    middle_payload = {
        "schema": "ols_g3_v3_middle_fields.v1",
        "fields": stage_payload,
        "common_second_hop_input_4096_complex128": stages["complex128"]["zero"][4096][
            "middle_aperture"
        ],
    }
    torch.save(middle_payload, run_dir / "middle_aperture_fields.pt")
    common_middle = stages["complex128"]["zero"][4096]["middle_aperture"].clone()
    common_metadata = {
        "source_phase": "zero first-layer screen shared identically across all three cases",
        "source_grid": 4096,
        "source_dtype": "complex128",
        "shape": list(common_middle.shape),
        "input_conversion_for_complex64": "one cast from this common complex128 field, reused across all windows",
        "common_field_norm": float(torch.linalg.vector_norm(common_middle)),
        "source_power_w_at_second_encoder_plane": _power(common_middle),
    }
    return stages, common_middle, {**stage_summary, "common_second_hop_input": common_metadata}


def _fixed_second_hop_runs(
    phases: dict[str, torch.Tensor],
    common_middle: torch.Tensor,
    device: torch.device,
    run_dir: Path,
) -> tuple[dict[str, dict[str, dict[int, torch.Tensor]]], dict[str, Any], dict[str, Any]]:
    outputs: dict[str, dict[str, dict[int, torch.Tensor]]] = {"complex64": {}, "complex128": {}}
    readouts: dict[str, Any] = {"complex64": {}, "complex128": {}}
    field_payload: dict[str, Any] = {"schema": "ols_g3_v3_fixed_second_hop_fields.v1"}
    for precision, real_dtype in (("complex64", torch.float32), ("complex128", torch.float64)):
        complex_dtype = torch.complex64 if precision == "complex64" else torch.complex128
        converted_middle = common_middle.to(device=device, dtype=complex_dtype)
        metadata_runtime = baseline._make_runtime(4096, real_dtype, device)
        middle_plane = PlaneSpec(
            "g3_v3_common_second_hop_input",
            (100, 100),
            (DX_M, DX_M),
            WAVELENGTH_M,
            metadata_runtime.encoder_entrance_plane.z_m + 1.5e-3,
            metadata_runtime.encoder_entrance_plane.coordinate_frame,
        )
        middle = ComplexField(
            converted_middle[None, None],
            middle_plane,
            OperationProvenance("g3_v3_fixed_common_second_hop_input", middle_plane.plane_id, None),
        )
        for phase_name in PHASE_NAMES:
            outputs[precision][phase_name] = {}
            readouts[precision][phase_name] = {}
            field_payload.setdefault(phase_name, {}).setdefault(precision, {})
            for grid in GRID_SIZES:
                runtime = baseline._make_runtime(grid, real_dtype, device)
                full = _second_hop(
                    runtime,
                    middle,
                    phases[phase_name][1, 0],
                    grid,
                    f"g3_v3_fixed_second_{precision}_{phase_name}_{grid}",
                )
                offset = (grid - ROI_SIZE) // 2
                roi = full.tensor[
                    0, 0, offset : offset + ROI_SIZE, offset : offset + ROI_SIZE
                ].detach().cpu()
                outputs[precision][phase_name][grid] = roi
                field_payload[phase_name][precision][str(grid)] = roi
                readouts[precision][phase_name][grid] = _mode_readout(
                    roi, precision, phase_name, grid, device
                )
                del runtime, full, roi
                clear_production_transfer_cache()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
        del converted_middle, middle, metadata_runtime
    torch.save(field_payload, run_dir / "fixed_second_hop_common_roi_fields.pt")

    comparisons: dict[str, Any] = {"fixed_input_window_comparisons": {}, "full_vs_fixed_input": {}}
    full_fields = torch.load(
        run_dir / "g3_common_roi_fields.pt", map_location="cpu", weights_only=True
    )
    for precision in ("complex64", "complex128"):
        comparisons["fixed_input_window_comparisons"][precision] = {}
        comparisons["full_vs_fixed_input"][precision] = {}
        for phase_name in PHASE_NAMES:
            comparisons["fixed_input_window_comparisons"][precision][phase_name] = {}
            comparisons["full_vs_fixed_input"][precision][phase_name] = {}
            for reference_grid, test_grid in ((2048, 3072), (2048, 4096), (3072, 4096)):
                reference = outputs[precision][phase_name][reference_grid].to(torch.complex128)
                test = outputs[precision][phase_name][test_grid].to(torch.complex128)
                phase_shift, aligned = baseline._phase_alignment(reference, test)
                power_reference, power_test = _power(reference), _power(test)
                comparisons["fixed_input_window_comparisons"][precision][phase_name][
                    f"{reference_grid}_vs_{test_grid}"
                ] = {
                    "raw_complex_relative_l2_error": baseline._rel_l2(reference, test),
                    "aligned_complex_relative_l2_error": baseline._rel_l2(reference, aligned),
                    "common_global_phase_rad": phase_shift,
                    "intensity_relative_l2_error": baseline._rel_l2(
                        reference.abs().square(), test.abs().square()
                    ),
                    "common_roi_power_relative_error": abs(power_test - power_reference)
                    / max(power_reference, EPS),
                    "power_reference_w": power_reference,
                    "power_test_w": power_test,
                }
            for grid in GRID_SIZES:
                original = full_fields[phase_name][precision][str(grid)].to(torch.complex128)
                fixed = outputs[precision][phase_name][grid].to(torch.complex128)
                phase_shift, aligned = baseline._phase_alignment(original, fixed)
                comparisons["full_vs_fixed_input"][precision][phase_name][str(grid)] = {
                    "raw_complex_relative_l2_error": baseline._rel_l2(original, fixed),
                    "aligned_complex_relative_l2_error": baseline._rel_l2(original, aligned),
                    "common_global_phase_rad": phase_shift,
                    "intensity_relative_l2_error": baseline._rel_l2(
                        original.abs().square(), fixed.abs().square()
                    ),
                    "common_roi_power_relative_error": abs(_power(fixed) - _power(original))
                    / max(_power(original), EPS),
                }
    del full_fields
    return outputs, readouts, comparisons


def _historical_crosscheck(path: Path, g3: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"explicit historical G3 report is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    historical_g3 = value.get("G3", value.get("g3"))
    if not isinstance(historical_g3, dict) or "comparisons" not in historical_g3:
        raise ValueError("historical report does not contain the expected G3 comparisons")
    old_values = [
        float(item["complex_relative_error_after_phase_alignment"])
        for phase in historical_g3["comparisons"]["complex64"].values()
        for item in phase.values()
    ]
    new_values = [
        float(item["complex_relative_error_after_phase_alignment"])
        for phase in g3["comparisons"]["complex64"].values()
        for item in phase.values()
    ]
    old_max = max(old_values)
    new_max = max(new_values)
    return {
        "historical_report": str(path),
        "historical_g3_status": historical_g3.get("status"),
        "historical_complex64_max_aligned_error": old_max,
        "rerun_complex64_max_aligned_error": new_max,
        "absolute_difference": abs(new_max - old_max),
        "crosscheck_is_provenance_only_not_an_admission_lock": True,
    }


def _summarize_g3(g3: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for precision in ("complex64", "complex128"):
        records = [
            (phase_name, pair_name, record)
            for phase_name, pairs in g3["comparisons"][precision].items()
            for pair_name, record in pairs.items()
        ]
        summary[precision] = {
            "maximum_raw_complex_relative_error": max(
                record["raw_complex_relative_error"] for _phase, _pair, record in records
            ),
            "maximum_aligned_complex_relative_error": max(
                record["complex_relative_error_after_phase_alignment"]
                for _phase, _pair, record in records
            ),
            "maximum_aligned_case": {
                "phase": max(
                    records,
                    key=lambda item: item[2]["complex_relative_error_after_phase_alignment"],
                )[0],
                "window_pair": max(
                    records,
                    key=lambda item: item[2]["complex_relative_error_after_phase_alignment"],
                )[1],
            },
            "maximum_intensity_relative_error": max(
                record["intensity_relative_error"] for _phase, _pair, record in records
            ),
            "maximum_common_roi_power_relative_error": max(
                record["common_roi_power_relative_error"] for _phase, _pair, record in records
            ),
            "comparison_count": len(records),
        }
    g3["summary_by_dtype"] = summary
    return summary


def run(
    config_path: Path,
    phase_path: Path,
    historical_report_path: Path,
    run_dir: Path,
    *,
    preflight_only: bool,
) -> dict[str, Any]:
    config = _load_config(config_path)
    phases = _load_phases(phase_path)
    if not torch.cuda.is_available():
        raise RuntimeError("this measurement requires the configured AutoDL CUDA device")
    device = torch.device("cuda:0")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not preflight_only and (run_dir / "report.json").exists():
        raise FileExistsError(f"refusing to overwrite completed run report: {run_dir / 'report.json'}")
    started = time.perf_counter()
    shutil.copyfile(config_path, run_dir / "configuration.yaml")
    if not (run_dir / "fixed_phase_screens.pt").exists():
        shutil.copyfile(phase_path, run_dir / "fixed_phase_screens.pt")
    if not (run_dir / "historical_g3_report.json").exists():
        shutil.copyfile(historical_report_path, run_dir / "historical_g3_report.json")

    preflight_path = run_dir / "preflight.json"
    if preflight_only:
        preflight = _preflight(config, phase_path, device)
        _write_json(preflight_path, preflight)
        return preflight
    if not preflight_path.is_file():
        raise FileNotFoundError("full run requires the saved PASS preflight from the same attempt")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS":
        raise RuntimeError("full run is blocked because the saved preflight did not PASS")

    baseline_config_path = Path(config["source_evidence"]["baseline_diagnostic_config"])
    baseline_config = baseline._load_config(baseline_config_path)
    g3 = baseline._run_g3(baseline_config, phase_path, device, run_dir)
    g3_summary = _summarize_g3(g3)
    historical = _historical_crosscheck(historical_report_path, g3)

    raw_fields = torch.load(run_dir / "g3_common_roi_fields.pt", map_location="cpu", weights_only=True)
    error_maps, _map_arrays = _error_map_records(raw_fields, run_dir)
    del raw_fields, _map_arrays
    stages, common_middle, stage_report = _reconstruct_middle_fields(phases, device, run_dir)
    fixed_outputs, mode_readouts, second_hop_report = _fixed_second_hop_runs(
        phases, common_middle, device, run_dir
    )
    points, selection = _fiber_probe_points(device)
    positions = selection.pop("positions_xy_m")
    cpu_reference = _cpu_reference(
        config,
        phases,
        common_middle,
        positions,
        points,
        fixed_outputs,
        run_dir,
    )
    selection["points"] = points
    stage_report["probe_selection"] = selection

    first_hop_max = max(
        record["first_common_roi"]["aligned_complex_relative_l2_error"]
        for precision_records in stage_report["first_hop_comparisons"].values()
        for phase_records in precision_records.values()
        for record in phase_records.values()
    )
    aperture_crop_max = max(
        record["middle_aperture"]["aligned_complex_relative_l2_error"]
        for precision_records in stage_report["first_hop_comparisons"].values()
        for phase_records in precision_records.values()
        for record in phase_records.values()
    )
    fixed_second_hop_max = {
        precision: max(
            record["aligned_complex_relative_l2_error"]
            for phase_records in second_hop_report["fixed_input_window_comparisons"][precision].values()
            for record in phase_records.values()
        )
        for precision in ("complex64", "complex128")
    }
    mechanism_assessment = {
        "first_hop_max_aligned_roi_error": first_hop_max,
        "first_hop_max_100x100_aperture_crop_error": aperture_crop_max,
        "fixed_common_input_second_hop_max_aligned_window_error": fixed_second_hop_max,
        "second_hop_window_dependence_survives_fixed_common_input": any(
            value > G3_TOLERANCE for value in fixed_second_hop_max.values()
        ),
        "interpretation": (
            "If the fixed common second-hop input retains window error above 1e-4 while first-hop and aperture-crop "
            "errors are smaller, first-hop window variation alone cannot explain G3. A valid nonperiodic integral "
            "can compare each window with the stated sampled-field reference, but this diagnostic does not force a "
            "separate causal label for frequency-grid sampling and periodic-boundary interaction."
        ),
        "independent_reference_valid_for_attribution": cpu_reference["status"] == "VALID",
        "no_boundary_cause_inferred_from_edge_power_alone": True,
    }

    try:
        head = __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, __import__("subprocess").CalledProcessError):
        head = None
    report = {
        "schema": SCHEMA,
        "status": "VALID_COMPLETED" if g3["production_dtype_pass"] else "VALID_FAILED",
        "scientific_gate": "G3_PASS" if g3["production_dtype_pass"] else "G3_FAIL",
        "decision_question": (
            "whether a supported minimal implementation defect can be localized and corrected so the fixed-window "
            "production complex64 G3 gate passes"
        ),
        "source_config": str(config_path),
        "fixed_phase_source": str(phase_path),
        "historical_G3_crosscheck": historical,
        "frozen_physical_system": config["optics"],
        "fiber_assumptions": config["fiber"],
        "loaded_source_modules": {
            "v3_diagnostic_runner": str(Path(__file__).resolve()),
            "baseline_g3_runner": str(Path(inspect.getfile(baseline._run_g3)).resolve()),
            "lightgen_physical_runtime": str(Path(inspect.getfile(LightGen2DSemanticGenerationInputOLSAlignment)).resolve()),
            "gaussian_s12_ols": str(Path(inspect.getfile(GaussianFiberArrayOLS)).resolve()),
            "lp01_fiber_operator": str(Path(inspect.getfile(FiberArrayOLS)).resolve()),
            "finite_crop_phase_stack": str(Path(inspect.getfile(FiniteCropPhaseStack)).resolve()),
            "angular_spectrum_propagation": str(Path(inspect.getfile(angular_spectrum_propagate)).resolve()),
        },
        "provenance_git_head": head,
        "execution": {
            "python": __import__("sys").executable,
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "elapsed_seconds_observation": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        },
        "preflight": preflight,
        "G3": g3,
        "G3_summary": g3_summary,
        "raw_and_aligned_error_maps": error_maps,
        "first_hop_and_aperture_crop": stage_report,
        "fixed_common_second_hop_input": {
            "source_grid": 4096,
            "source_dtype": "complex128",
            "shape": list(common_middle.shape),
            "relative_to_input_scale": float(
                torch.linalg.vector_norm(common_middle)
                / max(math.sqrt(SOURCE_POWER_W) / 200.0e-6, EPS)
            ),
            "common_field_details": stage_report["common_second_hop_input"],
        },
        "fixed_input_second_hop": second_hop_report,
        "mechanism_assessment": mechanism_assessment,
        "fixed_input_mode_readouts": mode_readouts,
        "independent_cpu_reference": cpu_reference,
        "decision_limits": {
            "only_full_two_hop_production_complex64_all_nine_common_roi_pairs_consume_G3": True,
            "g3_tolerance": G3_TOLERANCE,
            "complex128_pass_does_not_substitute_for_production_complex64": True,
            "reference_points_and_region_error_maps_do_not_substitute_for_G3": True,
            "solver_or_geometry_changed": False,
            "dimension_matrices_or_rankings_generated": False,
            "phase_optimizer_updates": 0,
        },
        "updates_and_forbidden_io": {
            "formal_training_updates": 0,
            "phase_optimizer_updates": 0,
            "dataset_reads": 0,
            "task_evaluator_calls": 0,
            "fid_calls": 0,
            "loss_changes": 0,
            "material_forward_calls": 0,
            "production_geometry_changes": 0,
            "propagation_solver_replacement": 0,
            "hidden_spatial_grid_expansion": 0,
        },
        "claim_limits": [
            "fiber coordinates and MFD remain unmeasured simulation assumptions",
            "the CPU integral represents the explicitly defined nonperiodic bandlimited sampled field, not a measured continuous aperture",
            "a nonconverged independent quadrature cannot support error-source attribution",
            "small edge power alone does not rule out periodic boundary interaction",
            "no OLS channel-count or capacity conclusion is made in this task",
        ],
    }
    _write_json(run_dir / "report.json", report)
    del stages, fixed_outputs
    clear_production_transfer_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--phase-source", type=Path, default=DEFAULT_PHASES)
    parser.add_argument("--historical-g3-report", type=Path, default=DEFAULT_HISTORICAL_REPORT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(
        args.config,
        args.phase_source,
        args.historical_g3_report,
        args.run_dir,
        preflight_only=args.preflight_only,
    )
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "scientific_gate": result.get("scientific_gate"),
                "preflight": result.get("status") if args.preflight_only else result.get("preflight", {}).get("status"),
                "G3_status": result.get("G3", {}).get("status"),
                "independent_reference_status": result.get("independent_cpu_reference", {}).get("status"),
                "run_dir": str(args.run_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
