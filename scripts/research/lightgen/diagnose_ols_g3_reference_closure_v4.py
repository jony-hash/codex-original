"""Close the independent quadrature evidence and recheck the saved G3 fields."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scripts.research.lightgen import diagnose_ols_g3_second_hop_error_isolation_v3 as v3
from scripts.research.lightgen import diagnose_ols_gaussian_dimension_reaudit_v1 as baseline
from ldm_project.v2.optics.propagation import clear_production_transfer_cache


SCHEMA = "ols_g3_reference_closure.v4"
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_g3_reference_closure_v4.yaml")
DEFAULT_V3_RUN_DIR = Path(
    "outputs/20261014_ols_g3_second_hop_error_isolation_v3/runs/attempt_01"
)
DEFAULT_RUN_DIR = Path("outputs/20261014_ols_g3_reference_closure_v4/runs/attempt_01")
NODE_COUNTS = (2048, 4096, 8192, 16384, 32768)
QUADRATURE_ORDER = 4
CROSS_ORDER = 8
TOLERANCE = 1.0e-7
G3_TOLERANCE = 1.0e-4
E0 = math.sqrt(v3.SOURCE_POWER_W) / 200.0e-6
POINT_GROUPS = {
    "roi_grid_25": slice(0, 25),
    "fiber_nearest_pixel_100": slice(25, 125),
    "combined_125": slice(0, 125),
}


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} configuration")
    optics = value.get("optics", {})
    expected_optics = {
        "wavelength_m": v3.WAVELENGTH_M,
        "pixel_pitch_m": v3.DX_M,
        "encoder_shape": [100, 100],
        "encoder_aperture_m": 200.0e-6,
        "propagation_windows_mm": [4.096, 6.144, 8.192],
        "propagation_grid_sizes": list(v3.GRID_SIZES),
        "inter_encoder_spacing_m": 1.5e-3,
        "last_encoder_to_fiber_m": 1.5e-3,
        "phase_screen_count": 2,
        "layer_power_transmission": v3.LAYER_TRANSMISSION,
        "input_power_w": v3.SOURCE_POWER_W,
    }
    if optics != expected_optics:
        raise ValueError("frozen optical geometry or power changed")
    if value.get("fiber") != {
        "reference_pitch_m": 125.0e-6,
        "mode_field_diameter_m": 14.222e-6,
        "gaussian_1e_amplitude_radius_m": 7.111e-6,
        "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
    }:
        raise ValueError("frozen fiber assumptions or unmeasured status changed")
    reference = value.get("reference", {})
    if reference != {
        "type": "nonperiodic_bandlimited_sampled_field_angular_spectrum_integral",
        "dtype": "complex128",
        "quadrature": "composite_gauss_legendre",
        "refinement_order_per_frequency_cell": QUADRATURE_ORDER,
        "refinement_frequency_nodes_per_axis": list(NODE_COUNTS),
        "cross_order_per_frequency_cell": CROSS_ORDER,
        "cross_order_frequency_nodes_per_axis": NODE_COUNTS[-1],
        "roi_probe_indices": [128, 512, 1023, 1535, 1919],
        "fiber_center_probe_count": 100,
        "convergence_tolerance": TOLERANCE,
        "reference_field_scale": "sqrt_input_power_over_200um",
        "frequency_row_chunk": 64,
        "position_batch": 16,
        "include_pixel_sinc_factor": False,
        "apply_additional_spectral_filter": False,
        "expand_spatial_propagation_grid": False,
    }:
        raise ValueError("reference definition or fixed convergence thresholds changed")
    if value.get("gates") != {
        "g3_complex_relative_error_after_common_global_phase_max": G3_TOLERANCE,
        "production_dtype": "complex64",
        "high_precision_diagnostic_dtype": "complex128",
    }:
        raise ValueError("G3 gate or dtype definition changed")
    if value.get("scope") != {
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


def _frequency_quadrature(
    node_count: int, order: int = QUADRATURE_ORDER
) -> tuple[np.ndarray, np.ndarray]:
    if order not in (4, 8) or node_count <= 0 or node_count % order:
        raise ValueError("quadrature requires order 4 or 8 and a positive divisible node count")
    nyquist = 1.0 / (2.0 * v3.DX_M)
    interval_count = node_count // order
    interval_width = (2.0 * nyquist) / interval_count
    nodes, weights = np.polynomial.legendre.leggauss(order)
    centers = -nyquist + (np.arange(interval_count, dtype=np.float64) + 0.5) * interval_width
    frequencies = (centers[:, None] + 0.5 * interval_width * nodes[None, :]).reshape(-1)
    quadrature_weights = (
        (0.5 * interval_width * weights[None, :]).repeat(interval_count, axis=0).reshape(-1)
    )
    if frequencies.size != node_count or not math.isclose(
        float(quadrature_weights.sum()), 2.0 * nyquist, rel_tol=1.0e-14
    ):
        raise RuntimeError("frequency nodes or integration measure are invalid")
    return frequencies, quadrature_weights


def _bandlimited_integral(
    field: np.ndarray,
    distance_m: float,
    positions_xy_m: np.ndarray,
    node_count: int,
    *,
    quadrature_order: int = QUADRATURE_ORDER,
    y_chunk: int = 64,
    point_batch: int = 16,
) -> np.ndarray:
    """Evaluate the fixed Nyquist-square integral in CPU complex128."""
    source = np.asarray(field, dtype=np.complex128)
    positions = np.asarray(positions_xy_m, dtype=np.float64)
    if source.shape != (100, 100) or positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("reference input must be 100x100 with finite XY points")
    if not np.isfinite(source).all() or not np.isfinite(positions).all():
        raise ValueError("reference input contains nonfinite values")
    if not math.isfinite(distance_m) or distance_m < 0.0:
        raise ValueError("propagation distance must be finite and nonnegative")
    frequencies, weights = _frequency_quadrature(node_count, quadrature_order)
    source_axis = (np.arange(100, dtype=np.float64) + 0.5) * v3.DX_M - 100 * v3.DX_M / 2
    area = v3.DX_M**2

    # At z=0 the same integral factors exactly into two 1-D quadratures. This is
    # used for analytic sinc controls and avoids allocating a 32768-by-32768 grid.
    if distance_m == 0.0:
        source_phase = np.exp(-2j * np.pi * frequencies[:, None] * source_axis[None, :])
        x_phase = np.exp(2j * np.pi * frequencies[:, None] * positions[:, 0][None, :])
        y_phase = np.exp(2j * np.pi * frequencies[:, None] * positions[:, 1][None, :])
        kernel_x = (x_phase * weights[:, None]).T @ source_phase
        kernel_y = (y_phase * weights[:, None]).T @ source_phase
        # Array axis 0 is y/row and axis 1 is x/column.
        values = np.einsum("pi,ij,pj->p", kernel_y, source, kernel_x, optimize=True) * area
        return np.asarray(values, dtype=np.complex128)

    exponent_x = np.exp(-2j * np.pi * frequencies[:, None] * source_axis[None, :])
    fourier_x = source @ exponent_x.T
    inverse_x = np.exp(2j * np.pi * frequencies[:, None] * positions[:, 0][None, :])
    inverse_y = np.exp(2j * np.pi * frequencies[:, None] * positions[:, 1][None, :])
    output = np.zeros(positions.shape[0], dtype=np.complex128)
    wavelength_squared = v3.WAVELENGTH_M**2
    phase_scale = 2.0 * math.pi * distance_m / v3.WAVELENGTH_M
    for y_start in range(0, node_count, y_chunk):
        y_stop = min(y_start + y_chunk, node_count)
        fy = frequencies[y_start:y_stop]
        spectrum = exponent_x[y_start:y_stop] @ fourier_x
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
        for point_start in range(0, positions.shape[0], point_batch):
            point_stop = min(point_start + point_batch, positions.shape[0])
            transformed = weighted @ inverse_x[:, point_start:point_stop]
            output[point_start:point_stop] += np.sum(
                transformed * inverse_y[y_start:y_stop, point_start:point_stop], axis=0
            )
    return output


def _load_points(v3_report: dict[str, Any]) -> tuple[list[dict[str, Any]], np.ndarray]:
    points = v3_report["independent_cpu_reference"]["selected_positions"]
    if len(points) != 125:
        raise ValueError("v3 evidence must contain exactly 125 selected physical positions")
    if sum(item["group"] == "roi_grid_25" for item in points) != 25:
        raise ValueError("v3 ROI point selection is incomplete")
    if sum(item["group"] == "fiber_center_nearest_roi_pixel_100" for item in points) != 100:
        raise ValueError("v3 fiber point selection is incomplete")
    positions = np.asarray([[p["x_m"], p["y_m"]] for p in points], dtype=np.float64)
    if positions.shape != (125, 2) or not np.isfinite(positions).all():
        raise ValueError("v3 probe coordinates are invalid")
    return points, positions


def _source_paths(config: dict[str, Any], v3_run_dir: Path) -> dict[str, Path]:
    records = config["source_evidence"]
    if Path(records["v3_run_dir"]) != v3_run_dir:
        raise ValueError("the explicit v3 evidence directory differs from the frozen config")
    return {key: v3_run_dir / records[key] for key in (
        "v3_report", "fixed_phase_screens", "middle_aperture_fields",
        "g3_common_roi_fields", "fixed_second_hop_fields", "v3_reference_report"
    )}


def _validate_saved_raw_level(
    path: Path,
    *,
    phase_name: str,
    order: int,
    node_count: int,
    expected_positions: np.ndarray,
    expected_input: np.ndarray,
    expected_phase: np.ndarray,
) -> dict[str, Any]:
    required = {
        "metadata_json", "frequency_nodes_hz", "frequency_weights_per_hz",
        "positions_xy_m", "common_middle_complex128", "last_phase_rad",
        "reference_values_complex128",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"raw reference evidence is missing required arrays: {missing}")
        metadata = json.loads(str(archive["metadata_json"].item()))
        if metadata != {
            "schema": "ols_g3_reference_level.v1",
            "phase": phase_name,
            "quadrature_order": order,
            "frequency_nodes_per_axis": node_count,
        }:
            raise ValueError(f"raw reference metadata does not match its explicit level: {path.name}")
        frequencies = np.asarray(archive["frequency_nodes_hz"], dtype=np.float64)
        weights = np.asarray(archive["frequency_weights_per_hz"], dtype=np.float64)
        positions = np.asarray(archive["positions_xy_m"], dtype=np.float64)
        field = np.asarray(archive["common_middle_complex128"], dtype=np.complex128)
        phase = np.asarray(archive["last_phase_rad"], dtype=np.float64)
        values = np.asarray(archive["reference_values_complex128"], dtype=np.complex128)
    expected_frequencies, expected_weights = _frequency_quadrature(node_count, order)
    if frequencies.shape != (node_count,) or weights.shape != (node_count,):
        raise ValueError("saved quadrature axes have the wrong shape")
    if not np.array_equal(frequencies, expected_frequencies) or not np.array_equal(weights, expected_weights):
        raise ValueError("saved quadrature nodes or weights differ from the stated rule")
    if not np.array_equal(positions, expected_positions):
        raise ValueError("saved selected positions differ from the consumed v3 coordinates")
    if field.shape != (100, 100) or not np.array_equal(field, expected_input):
        raise ValueError("saved second-hop input differs from the consumed v3 field")
    if phase.shape != (100, 100) or not np.array_equal(phase, expected_phase):
        raise ValueError("saved last-layer phase differs from the consumed v3 phase")
    if values.shape != (125,) or not np.isfinite(values).all():
        raise ValueError("saved reference complex values are missing, malformed, or nonfinite")
    return {"metadata": metadata, "values": values}


def _write_level(
    path: Path,
    *,
    phase_name: str,
    order: int,
    node_count: int,
    frequencies: np.ndarray,
    weights: np.ndarray,
    positions: np.ndarray,
    field: np.ndarray,
    last_phase: np.ndarray,
    values: np.ndarray,
) -> None:
    metadata = json.dumps({
        "schema": "ols_g3_reference_level.v1",
        "phase": phase_name,
        "quadrature_order": order,
        "frequency_nodes_per_axis": node_count,
    }, sort_keys=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            metadata_json=np.asarray(metadata),
            frequency_nodes_hz=frequencies,
            frequency_weights_per_hz=weights,
            positions_xy_m=positions,
            common_middle_complex128=field,
            last_phase_rad=last_phase,
            reference_values_complex128=values,
        )
    temporary.replace(path)


def _level_name(phase_name: str, order: int, nodes: int) -> str:
    safe_phase = phase_name.replace("/", "_")
    return f"{safe_phase}__gl{order}__nodes{nodes}.npz"


def _analytic_controls() -> dict[str, Any]:
    cases = {
        "center": (50, 50),
        "edge": (50, 99),
        "corner": (0, 0),
    }
    axis = (np.arange(100, dtype=np.float64) + 0.5) * v3.DX_M - 100 * v3.DX_M / 2
    offset = np.asarray([0.37 * v3.DX_M, -0.21 * v3.DX_M])
    records: dict[str, Any] = {}
    maximum_abs_over_e0 = 0.0
    maximum_missing_area_error_over_e0 = 0.0
    maximum_half_pixel_error_over_e0 = 0.0
    for label, (row, column) in cases.items():
        field = np.zeros((100, 100), dtype=np.complex128)
        field[row, column] = 1.0
        position = np.asarray([[axis[column] + offset[0], axis[row] + offset[1]]])
        observed = _bandlimited_integral(field, 0.0, position, 2048)[0]
        expected = np.sinc(offset[0] / v3.DX_M) * np.sinc(offset[1] / v3.DX_M)
        absolute_error = float(abs(observed - expected))
        maximum_abs_over_e0 = max(maximum_abs_over_e0, absolute_error / E0)
        missing_area_error = float(abs(observed / (v3.DX_M**2) - expected) / E0)
        maximum_missing_area_error_over_e0 = max(maximum_missing_area_error_over_e0, missing_area_error)
        wrong_position = position + np.asarray([[0.5 * v3.DX_M, 0.0]])
        shifted = _bandlimited_integral(field, 0.0, wrong_position, 2048)[0]
        half_pixel_error = float(abs(shifted - expected) / E0)
        maximum_half_pixel_error_over_e0 = max(maximum_half_pixel_error_over_e0, half_pixel_error)
        records[label] = {
            "source_pixel_row_column": [row, column],
            "target_xy_m": position[0].tolist(),
            "observed_complex": {"real": float(observed.real), "imag": float(observed.imag)},
            "analytic_real": float(expected),
            "absolute_error": absolute_error,
            "absolute_error_over_E0": absolute_error / E0,
            "omitted_area_negative_error_over_E0": missing_area_error,
            "half_pixel_shift_negative_error_over_E0": half_pixel_error,
        }

    far_field = np.zeros((100, 100), dtype=np.complex128)
    far_field[50, 50] = np.exp(0.37j)
    far_position = np.asarray([[1.71e-3 + axis[50], -1.83e-3 + axis[50]]])
    far_observed = _bandlimited_integral(far_field, 0.0, far_position, 32768)[0]
    dx, dy = far_position[0] - np.asarray([axis[50], axis[50]])
    far_expected = np.exp(0.37j) * np.sinc(dx / v3.DX_M) * np.sinc(dy / v3.DX_M)
    far_error = float(abs(far_observed - far_expected))
    maximum_abs_over_e0 = max(maximum_abs_over_e0, far_error / E0)

    coherent = np.zeros((100, 100), dtype=np.complex128)
    coherent[50, 50] = 1.0
    coherent[50, 51] = 1.0j
    coherent_position = np.asarray([[axis[50] + 0.5 * v3.DX_M, axis[50]]])
    coherent_observed = _bandlimited_integral(coherent, 0.0, coherent_position, 2048)[0]
    coherent_expected = np.sinc(0.5) + 1.0j * np.sinc(-0.5)
    coherent_error = float(abs(coherent_observed - coherent_expected))
    maximum_abs_over_e0 = max(maximum_abs_over_e0, coherent_error / E0)
    lost_phase_error_over_e0 = float(abs(coherent_observed.real - coherent_expected) / E0)

    rng = np.random.default_rng(20261014)
    random_field = rng.normal(size=(100, 100)) + 1j * rng.normal(size=(100, 100))
    fx, fy = 0.173 / v3.DX_M, -0.319 / v3.DX_M
    source_axis = axis
    ex = np.exp(-2j * np.pi * fx * source_axis)
    ey = np.exp(-2j * np.pi * fy * source_axis)
    factored = ey @ random_field @ ex
    direct = np.sum(random_field * (ey[:, None] * ex[None, :]))
    direct_error = float(abs(factored - direct) / max(abs(direct), 1.0))
    passed = (
        maximum_abs_over_e0 <= 1.0e-10
        and direct_error <= 1.0e-12
        and maximum_missing_area_error_over_e0 > 1.0e-10
        and maximum_half_pixel_error_over_e0 > 1.0e-10
        and lost_phase_error_over_e0 > 1.0e-10
    )
    if not passed:
        raise RuntimeError("analytic sinc, complex-phase, far-position, or direct Fourier control failed")
    return {
        "status": "PASS",
        "zero_distance_single_pixel_sinc_cases": records,
        "far_position_case": {
            "target_xy_m": far_position[0].tolist(),
            "quadrature_nodes_per_axis": 32768,
            "observed_complex": {"real": float(far_observed.real), "imag": float(far_observed.imag)},
            "analytic_complex": {"real": float(far_expected.real), "imag": float(far_expected.imag)},
            "absolute_error": far_error,
            "absolute_error_over_E0": far_error / E0,
        },
        "coherent_two_pixel_known_phase": {
            "observed_complex": {"real": float(coherent_observed.real), "imag": float(coherent_observed.imag)},
            "analytic_complex": {"real": float(coherent_expected.real), "imag": float(coherent_expected.imag)},
            "absolute_error_over_E0": coherent_error / E0,
            "lost_phase_rejected": lost_phase_error_over_e0 > 1.0e-10,
        },
        "independent_direct_fourier_coefficient_relative_error": direct_error,
        "maximum_analytic_absolute_error_over_E0": maximum_abs_over_e0,
        "analytic_tolerance_over_E0": 1.0e-10,
        "omitted_area_negative_control_max_error_over_E0": maximum_missing_area_error_over_e0,
        "half_pixel_shift_negative_control_max_error_over_E0": maximum_half_pixel_error_over_e0,
        "dropped_imaginary_phase_negative_control_error_over_E0": lost_phase_error_over_e0,
        "pixel_area_included": True,
        "known_complex_phase_compared_without_alignment": True,
    }


def _missing_value_negative_control() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ols_g3_v4_missing_value_") as temporary:
        path = Path(temporary) / "missing.npz"
        np.savez_compressed(
            path,
            metadata_json=np.asarray(json.dumps({
                "schema": "ols_g3_reference_level.v1",
                "phase": "zero",
                "quadrature_order": 4,
                "frequency_nodes_per_axis": 4,
            })),
            frequency_nodes_hz=np.zeros(4),
            frequency_weights_per_hz=np.ones(4),
            positions_xy_m=np.zeros((125, 2)),
            common_middle_complex128=np.zeros((100, 100), dtype=np.complex128),
            last_phase_rad=np.zeros((100, 100)),
        )
        try:
            _validate_saved_raw_level(
                path,
                phase_name="zero",
                order=4,
                node_count=4,
                expected_positions=np.zeros((125, 2)),
                expected_input=np.zeros((100, 100), dtype=np.complex128),
                expected_phase=np.zeros((100, 100)),
            )
        except ValueError as error:
            return {"status": "PASS", "missing_complex_values_rejected": True, "rejection": str(error)}
    raise RuntimeError("raw-evidence consumer accepted a level with missing complex values")


def _coordinate_phase_archive_negative_controls() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="ols_g3_v4_bad_metadata_") as temporary:
        root = Path(temporary)
        frequencies, weights = _frequency_quadrature(4, 4)
        correct_positions = np.zeros((125, 2), dtype=np.float64)
        correct_input = np.zeros((100, 100), dtype=np.complex128)
        correct_phase = np.zeros((100, 100), dtype=np.float64)
        common = {
            "frequency_nodes_hz": frequencies,
            "frequency_weights_per_hz": weights,
            "common_middle_complex128": correct_input,
            "reference_values_complex128": np.zeros(125, dtype=np.complex128),
        }
        metadata = np.asarray(json.dumps({
            "schema": "ols_g3_reference_level.v1",
            "phase": "zero",
            "quadrature_order": 4,
            "frequency_nodes_per_axis": 4,
        }))
        cases = {
            "wrong_half_pixel_coordinates": {
                **common,
                "metadata_json": metadata,
                "positions_xy_m": correct_positions + np.asarray([0.5 * v3.DX_M, 0.0]),
                "last_phase_rad": correct_phase,
            },
            "wrong_saved_phase": {
                **common,
                "metadata_json": metadata,
                "positions_xy_m": correct_positions,
                "last_phase_rad": np.full((100, 100), 0.125),
            },
        }
        rejected: dict[str, bool] = {}
        for name, arrays in cases.items():
            path = root / f"{name}.npz"
            np.savez_compressed(path, **arrays)
            try:
                _validate_saved_raw_level(
                    path,
                    phase_name="zero",
                    order=4,
                    node_count=4,
                    expected_positions=correct_positions,
                    expected_input=correct_input,
                    expected_phase=correct_phase,
                )
            except ValueError:
                rejected[name] = True
            else:
                rejected[name] = False
    if not all(rejected.values()):
        raise RuntimeError("raw evidence consumer accepted a coordinate or phase mismatch")
    return {"status": "PASS", "consumer_rejections": rejected}


def _preflight(config: dict[str, Any], source_paths: dict[str, Path], phase_path: Path) -> dict[str, Any]:
    for key, path in source_paths.items():
        if key != "v3_run_dir" and not path.is_file():
            raise FileNotFoundError(f"explicit v3 source evidence is missing: {path}")
    phases = v3._load_phases(phase_path)
    v3_report = json.loads(source_paths["v3_report"].read_text(encoding="utf-8"))
    if v3_report.get("schema") != "ols_g3_second_hop_error_isolation.v3":
        raise ValueError("source evidence is not the named v3 diagnostic")
    points, positions = _load_points(v3_report)
    v3_reference = json.loads(source_paths["v3_reference_report"].read_text(encoding="utf-8"))
    if v3_reference.get("selected_positions") != v3_report["independent_cpu_reference"].get("selected_positions"):
        raise ValueError("v3 selected physical coordinates disagree between the saved report files")
    middle_payload = torch.load(
        source_paths["middle_aperture_fields"], map_location="cpu", weights_only=True, mmap=True
    )
    common_middle = middle_payload.get("common_second_hop_input_4096_complex128")
    if common_middle is None or tuple(common_middle.shape) != (100, 100) or not common_middle.is_complex():
        raise ValueError("v3 saved common second-hop input is absent or has the wrong shape/dtype")
    if not torch.isfinite(common_middle).all():
        raise ValueError("v3 saved common second-hop input contains nonfinite values")
    del middle_payload, common_middle
    integral_controls = _analytic_controls()
    metadata_controls = _coordinate_phase_archive_negative_controls()
    missing_control = _missing_value_negative_control()
    if not torch.cuda.is_available():
        raise RuntimeError("v4 representative production preflight requires AutoDL CUDA")
    v3_config_path = source_paths["v3_run_dir"] / "configuration.yaml" if "v3_run_dir" in source_paths else None
    if v3_config_path is None:
        raise ValueError("internal error: v3 run directory was not retained in the source paths")
    v3_config = v3._load_config(v3_config_path)
    production = v3._preflight(v3_config, phase_path, torch.device("cuda:0"))
    return {
        "status": "PASS",
        "reference_controls": integral_controls,
        "negative_controls": {
            "quadrature_analytic": {
                "omitted_pixel_area_rejected": integral_controls["omitted_area_negative_control_max_error_over_E0"] > 1e-10,
                "dropped_complex_phase_rejected": integral_controls["dropped_imaginary_phase_negative_control_error_over_E0"] > 1e-10,
                "half_pixel_coordinate_shift_rejected": integral_controls["half_pixel_shift_negative_control_max_error_over_E0"] > 1e-10,
            },
            "saved_evidence_consumer": metadata_controls,
            "missing_raw_level_values": missing_control,
        },
        "v3_input_report": str(source_paths["v3_report"]),
        "selected_position_count": len(points),
        "selected_position_shape": list(positions.shape),
        "v3_representative_production_preflight": production,
        "formal_training_updates": 0,
        "phase_optimizer_updates": 0,
        "dataset_reads": 0,
        "task_evaluator_calls": 0,
        "fid_calls": 0,
        "loss_changes": 0,
        "material_forward_calls": 0,
        "propagation_solver_replacement": False,
    }


def _recompute_g3(v3_report: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {"complex64": {}, "complex128": {}}
    for precision in ("complex64", "complex128"):
        for phase_name in v3.PHASE_NAMES:
            comparisons[precision][phase_name] = {}
            for reference_grid, test_grid in ((2048, 3072), (2048, 4096), (3072, 4096)):
                reference = fields[phase_name][precision][str(reference_grid)].to(torch.complex128)
                test = fields[phase_name][precision][str(test_grid)].to(torch.complex128)
                phase, aligned = baseline._phase_alignment(reference, test)
                record = {
                    "reference_grid": reference_grid,
                    "test_grid": test_grid,
                    "raw_complex_relative_error": baseline._rel_l2(reference, test),
                    "complex_relative_error_after_phase_alignment": baseline._rel_l2(reference, aligned),
                    "common_global_phase_rad": phase,
                    "intensity_relative_error": baseline._rel_l2(reference.abs().square(), test.abs().square()),
                    "common_roi_power_relative_error": abs(v3._power(test) - v3._power(reference)) / max(v3._power(reference), v3.EPS),
                    "common_roi_power_reference_w": v3._power(reference),
                    "common_roi_power_test_w": v3._power(test),
                }
                comparisons[precision][phase_name][f"{reference_grid}_vs_{test_grid}"] = record
    values = [
        item["complex_relative_error_after_phase_alignment"]
        for phase in comparisons["complex64"].values()
        for item in phase.values()
    ]
    max_c64 = max(values)
    report_comparisons = v3_report["G3"]["comparisons"]
    metric_key_pairs = {
        "raw_complex_relative_error": "raw_complex_relative_error",
        "complex_relative_error_after_phase_alignment": "complex_relative_error_after_phase_alignment",
        "intensity_relative_error": "intensity_relative_error",
        "common_roi_power_relative_error": "common_roi_power_relative_error",
        "global_phase_rad": "common_global_phase_rad",
    }
    metric_deltas = {key: 0.0 for key in metric_key_pairs}
    for precision in ("complex64", "complex128"):
        for phase_name, pairs in comparisons[precision].items():
            for pair_name, record in pairs.items():
                old = report_comparisons[precision][phase_name][pair_name]
                for old_key, new_key in metric_key_pairs.items():
                    metric_deltas[old_key] = max(metric_deltas[old_key], abs(record[new_key] - old[old_key]))
    maximum_metric_delta = max(metric_deltas.values())
    original_pass = bool(v3_report["G3"]["production_dtype_pass"])
    decision_matches = (max_c64 <= G3_TOLERANCE) == original_pass
    if not decision_matches:
        raise RuntimeError(
            "saved-field recomputation changes the original 1e-4 complex64 G3 decision; "
            f"maximum_cpu_gpu_metric_delta={maximum_metric_delta:.6g}"
        )
    return {
        "status": "VALID_PASS" if max_c64 <= G3_TOLERANCE else "VALID_FAILED",
        "scientific_gate": "G3_PASS" if max_c64 <= G3_TOLERANCE else "G3_FAIL",
        "tolerance": G3_TOLERANCE,
        "production_dtype": "complex64",
        "production_dtype_pass": max_c64 <= G3_TOLERANCE,
        "maximum_aligned_complex_relative_error_complex64": max_c64,
        "maximum_aligned_complex_relative_error_complex128": max(
            item["complex_relative_error_after_phase_alignment"]
            for phase in comparisons["complex128"].values() for item in phase.values()
        ),
        "comparison_count_per_dtype": 9,
        "recomputed_from_saved_full_two_hop_fields": True,
        "maximum_difference_from_v3_report_by_metric": metric_deltas,
        "maximum_observed_metric_difference_from_v3": maximum_metric_delta,
        "cpu_reread_and_original_report_g3_decisions_match": decision_matches,
        "original_v3_production_gate_decision": "G3_PASS" if original_pass else "G3_FAIL",
        "comparisons": comparisons,
    }


def _verify_mode_readouts(
    v3_report: dict[str, Any], fixed_path: Path, device: torch.device
) -> dict[str, Any]:
    saved = torch.load(fixed_path, map_location="cpu", weights_only=True, mmap=True)
    expected = v3_report["fixed_input_mode_readouts"]
    recomputed: dict[str, Any] = {"complex64": {}, "complex128": {}}
    maximum_coefficient_relative_error = 0.0
    maximum_power_relative_error = 0.0
    count = 0
    with torch.no_grad():
        for precision in ("complex64", "complex128"):
            for phase_name in v3.PHASE_NAMES:
                recomputed[precision][phase_name] = {}
                for grid in v3.GRID_SIZES:
                    field = saved[phase_name][precision][str(grid)]
                    measured = v3._mode_readout(field, precision, phase_name, grid, device)
                    recomputed[precision][phase_name][str(grid)] = measured
                    prior = expected[precision][phase_name][str(grid)]
                    for mode in ("gaussian_s12", "lp01"):
                        coefficient = np.asarray([
                            complex(item["real"], item["imag"])
                            for item in measured["modes"][mode]["complex_coefficients_sqrt_w_row_major"]
                        ])
                        old_coefficient = np.asarray([
                            complex(item["real"], item["imag"])
                            for item in prior["modes"][mode]["complex_coefficients_sqrt_w_row_major"]
                        ])
                        maximum_coefficient_relative_error = max(
                            maximum_coefficient_relative_error,
                            float(np.linalg.norm(coefficient - old_coefficient) / max(np.linalg.norm(old_coefficient), v3.EPS)),
                        )
                        new_power = np.asarray(measured["modes"][mode]["per_fiber_power_w_row_major"])
                        old_power = np.asarray(prior["modes"][mode]["per_fiber_power_w_row_major"])
                        maximum_power_relative_error = max(
                            maximum_power_relative_error,
                            float(np.linalg.norm(new_power - old_power) / max(np.linalg.norm(old_power), v3.EPS)),
                        )
                    count += 1
    if maximum_coefficient_relative_error > 1.0e-8 or maximum_power_relative_error > 1.0e-8:
        raise RuntimeError("recomputed Gaussian/LP01 readouts do not reproduce saved v3 values")
    return {
        "status": "PASS",
        "field_count": count,
        "mode_count": 2,
        "maximum_complex_coefficient_relative_difference_from_v3": maximum_coefficient_relative_error,
        "maximum_per_fiber_power_relative_difference_from_v3": maximum_power_relative_error,
        "readouts": recomputed,
    }


def _save_progress(path: Path, record: dict[str, Any]) -> None:
    v3._write_json(path, record)


def _write_and_reread_level(
    directory: Path,
    phase_name: str,
    order: int,
    node_count: int,
    positions: np.ndarray,
    common_input: np.ndarray,
    phase: np.ndarray,
) -> Path:
    frequencies, weights = _frequency_quadrature(node_count, order)
    active = common_input * math.sqrt(v3.LAYER_TRANSMISSION) * np.exp(1j * phase)
    started = time.perf_counter()
    values = _bandlimited_integral(
        active,
        1.5e-3,
        positions,
        node_count,
        quadrature_order=order,
        y_chunk=64,
        point_batch=16,
    )
    if not np.isfinite(values).all():
        raise RuntimeError(f"nonfinite reference values for {phase_name}, GL{order}, {node_count}")
    path = directory / _level_name(phase_name, order, node_count)
    _write_level(
        path,
        phase_name=phase_name,
        order=order,
        node_count=node_count,
        frequencies=frequencies,
        weights=weights,
        positions=positions,
        field=common_input,
        last_phase=phase,
        values=values,
    )
    _validate_saved_raw_level(
        path,
        phase_name=phase_name,
        order=order,
        node_count=node_count,
        expected_positions=positions,
        expected_input=common_input,
        expected_phase=phase,
    )
    print(json.dumps({
        "event": "quadrature_level_saved_and_reread",
        "phase": phase_name,
        "order": order,
        "nodes": node_count,
        "elapsed_seconds_observation": time.perf_counter() - started,
        "file": path.name,
        "bytes": path.stat().st_size,
    }, sort_keys=True), flush=True)
    return path


def _refinement_summary(
    level_paths: dict[str, dict[int, Path]],
    phase_inputs: dict[str, np.ndarray],
    positions: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[int, np.ndarray]]]:
    values_by_phase: dict[str, dict[int, np.ndarray]] = {}
    for phase_name, by_nodes in level_paths.items():
        values_by_phase[phase_name] = {}
        for nodes, path in by_nodes.items():
            values_by_phase[phase_name][nodes] = _validate_saved_raw_level(
                path,
                phase_name=phase_name,
                order=QUADRATURE_ORDER,
                node_count=nodes,
                expected_positions=positions,
                expected_input=phase_inputs["common_input"],
                expected_phase=phase_inputs[phase_name],
            )["values"]
    summary: dict[str, Any] = {"by_phase": {}, "all_groups_converged_on_top_two_intervals": True}
    intervals_to_require = ((8192, 16384), (16384, 32768))
    for phase_name, by_nodes in values_by_phase.items():
        summary["by_phase"][phase_name] = {"adjacent_refinements": {}}
        for before, after in zip(NODE_COUNTS, NODE_COUNTS[1:]):
            records = {}
            for group_name, selected in POINT_GROUPS.items():
                reference = by_nodes[before][selected]
                refined = by_nodes[after][selected]
                difference = refined - reference
                relative = float(np.linalg.norm(difference) / max(np.linalg.norm(refined), 1e-300))
                maximum = float(np.max(np.abs(difference)) / E0)
                records[group_name] = {
                    "relative_l2_difference": relative,
                    "maximum_point_difference_over_E0": maximum,
                    "relative_l2_tolerance": TOLERANCE,
                    "maximum_point_over_E0_tolerance": TOLERANCE,
                    "converged": relative <= TOLERANCE and maximum <= TOLERANCE,
                }
            summary["by_phase"][phase_name]["adjacent_refinements"][f"{before}_to_{after}"] = records
        for before, after in intervals_to_require:
            records = summary["by_phase"][phase_name]["adjacent_refinements"][f"{before}_to_{after}"]
            if not all(record["converged"] for record in records.values()):
                summary["all_groups_converged_on_top_two_intervals"] = False
    return summary, values_by_phase


def _cross_order_summary(
    level_paths: dict[str, dict[int, Path]],
    cross_paths: dict[str, Path],
    phase_inputs: dict[str, np.ndarray],
    positions: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {"by_phase": {}, "all_groups_converged": True}
    for phase_name, path in cross_paths.items():
        order8 = _validate_saved_raw_level(
            path,
            phase_name=phase_name,
            order=CROSS_ORDER,
            node_count=NODE_COUNTS[-1],
            expected_positions=positions,
            expected_input=phase_inputs["common_input"],
            expected_phase=phase_inputs[phase_name],
        )["values"]
        order4 = _validate_saved_raw_level(
            level_paths[phase_name][NODE_COUNTS[-1]],
            phase_name=phase_name,
            order=QUADRATURE_ORDER,
            node_count=NODE_COUNTS[-1],
            expected_positions=positions,
            expected_input=phase_inputs["common_input"],
            expected_phase=phase_inputs[phase_name],
        )["values"]
        result["by_phase"][phase_name] = {}
        for group_name, selected in POINT_GROUPS.items():
            difference = order8[selected] - order4[selected]
            relative = float(np.linalg.norm(difference) / max(np.linalg.norm(order8[selected]), 1e-300))
            maximum = float(np.max(np.abs(difference)) / E0)
            passed = relative <= TOLERANCE and maximum <= TOLERANCE
            result["by_phase"][phase_name][group_name] = {
                "relative_l2_difference": relative,
                "maximum_point_difference_over_E0": maximum,
                "relative_l2_tolerance": TOLERANCE,
                "maximum_point_over_E0_tolerance": TOLERANCE,
                "converged": passed,
            }
            result["all_groups_converged"] &= passed
    return result


def _production_point_comparisons(
    positions: np.ndarray,
    points: list[dict[str, Any]],
    reference: dict[str, np.ndarray],
    fixed_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    fields = torch.load(fixed_path, map_location="cpu", weights_only=True, mmap=True)
    arrays: dict[str, np.ndarray] = {}
    result: dict[str, Any] = {}
    for precision in ("complex64", "complex128"):
        result[precision] = {}
        for phase_name in v3.PHASE_NAMES:
            result[precision][phase_name] = {}
            target = reference[phase_name]
            for grid in v3.GRID_SIZES:
                sample = v3._roi_sample(fields[phase_name][precision][str(grid)], points)
                key = f"{precision}__{phase_name}__grid{grid}"
                arrays[f"{key}__production"] = sample
                arrays[f"{key}__reference"] = target
                result[precision][phase_name][str(grid)] = {}
                for group_name, selected in POINT_GROUPS.items():
                    reference_group = target[selected]
                    sample_group = sample[selected]
                    difference = sample_group - reference_group
                    intensity_difference = np.abs(sample_group) ** 2 - np.abs(reference_group) ** 2
                    result[precision][phase_name][str(grid)][group_name] = {
                        "raw_complex_relative_l2_no_phase_fit": float(
                            np.linalg.norm(difference) / max(np.linalg.norm(reference_group), 1e-300)
                        ),
                        "maximum_absolute_complex_difference_over_E0": float(np.max(np.abs(difference)) / E0),
                        "intensity_relative_l2": float(
                            np.linalg.norm(intensity_difference)
                            / max(np.linalg.norm(np.abs(reference_group) ** 2), 1e-300)
                        ),
                        "global_phase_fitting_applied": False,
                    }
    with output_path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    return {
        "status": "COMPLETE",
        "raw_values_file": output_path.name,
        "raw_comparison_count": len(arrays) // 2,
        "phase_fitting_applied": False,
        "by_precision_phase_grid": result,
    }


def run(
    config_path: Path,
    v3_run_dir: Path,
    run_dir: Path,
    *,
    preflight_only: bool,
) -> dict[str, Any]:
    config = _load_config(config_path)
    source_paths = _source_paths(config, v3_run_dir)
    source_paths["v3_run_dir"] = v3_run_dir
    phase_path = source_paths["fixed_phase_screens"]
    if not phase_path.is_file():
        raise FileNotFoundError(f"explicit fixed phase source is missing: {phase_path}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if not preflight_only and (run_dir / "report.json").exists():
        raise FileExistsError(f"refusing to overwrite completed v4 report: {run_dir / 'report.json'}")
    shutil.copyfile(config_path, run_dir / "configuration.yaml")
    if not (run_dir / "fixed_phase_screens.pt").exists():
        shutil.copyfile(phase_path, run_dir / "fixed_phase_screens.pt")
    if preflight_only:
        preflight = _preflight(config, source_paths, phase_path)
        v3._write_json(run_dir / "preflight.json", preflight)
        return preflight
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("full run requires a saved PASS preflight from this v4 attempt")
    if not torch.cuda.is_available():
        raise RuntimeError("saved-field readout verification requires AutoDL CUDA")
    started = time.perf_counter()
    progress_path = run_dir / "progress.json"
    v3_report = json.loads(source_paths["v3_report"].read_text(encoding="utf-8"))
    v3_reference_record = json.loads(source_paths["v3_reference_report"].read_text(encoding="utf-8"))
    if v3_reference_record.get("selected_positions") != v3_report["independent_cpu_reference"].get("selected_positions"):
        raise ValueError("v3 selected physical coordinates disagree between the saved report files")
    points, positions = _load_points(v3_report)
    phases = v3._load_phases(phase_path)
    middle_payload = torch.load(source_paths["middle_aperture_fields"], map_location="cpu", weights_only=True, mmap=True)
    common_middle_tensor = middle_payload["common_second_hop_input_4096_complex128"]
    if tuple(common_middle_tensor.shape) != (100, 100) or not common_middle_tensor.is_complex():
        raise ValueError("v3 common second-hop input is not a 100x100 complex field")
    common_middle = common_middle_tensor.numpy().astype(np.complex128, copy=False)
    if not np.isfinite(common_middle).all():
        raise ValueError("v3 common second-hop input contains nonfinite values")
    phase_inputs = {"common_input": common_middle}
    for phase_name in v3.PHASE_NAMES:
        phase_inputs[phase_name] = phases[phase_name][1, 0].numpy().astype(np.float64, copy=False)

    # Recompute the production G3 decision from the complete fields already saved by v3.
    g3_fields = torch.load(source_paths["g3_common_roi_fields"], map_location="cpu", weights_only=True, mmap=True)
    recomputed_g3 = _recompute_g3(v3_report, g3_fields)
    del g3_fields
    fixed_mode_readouts = _verify_mode_readouts(v3_report, source_paths["fixed_second_hop_fields"], torch.device("cuda:0"))

    levels_dir = run_dir / "reference_levels"
    levels_dir.mkdir(parents=True, exist_ok=True)
    level_paths: dict[str, dict[int, Path]] = {phase: {} for phase in v3.PHASE_NAMES}
    for nodes in NODE_COUNTS:
        for phase_name in v3.PHASE_NAMES:
            path = _write_and_reread_level(
                levels_dir, phase_name, QUADRATURE_ORDER, nodes, positions,
                common_middle, phase_inputs[phase_name]
            )
            level_paths[phase_name][nodes] = path
            _save_progress(progress_path, {
                "stage": "four_point_refinement",
                "completed_phase": phase_name,
                "completed_nodes": nodes,
                "completed_level_count": sum(len(items) for items in level_paths.values()),
                "expected_level_count": len(v3.PHASE_NAMES) * len(NODE_COUNTS),
                "formal_training_updates": 0,
            })

    refinement, values_by_phase = _refinement_summary(level_paths, phase_inputs, positions)
    top_two_pass = refinement["all_groups_converged_on_top_two_intervals"]
    cross_paths: dict[str, Path] = {}
    cross_order = None
    if top_two_pass:
        for phase_name in v3.PHASE_NAMES:
            path = _write_and_reread_level(
                levels_dir, phase_name, CROSS_ORDER, NODE_COUNTS[-1], positions,
                common_middle, phase_inputs[phase_name]
            )
            cross_paths[phase_name] = path
            _save_progress(progress_path, {
                "stage": "eight_point_cross_order",
                "completed_phase": phase_name,
                "completed_level_count": len(cross_paths),
                "expected_level_count": len(v3.PHASE_NAMES),
                "formal_training_updates": 0,
            })
        cross_order = _cross_order_summary(level_paths, cross_paths, phase_inputs, positions)
    reference_valid = bool(top_two_pass and cross_order and cross_order["all_groups_converged"])

    production_reference_comparison = {"status": "NOT_RUN_REFERENCE_NOT_VALID"}
    if reference_valid:
        final_values = {}
        for phase_name in v3.PHASE_NAMES:
            final_values[phase_name] = _validate_saved_raw_level(
                level_paths[phase_name][NODE_COUNTS[-1]],
                phase_name=phase_name,
                order=QUADRATURE_ORDER,
                node_count=NODE_COUNTS[-1],
                expected_positions=positions,
                expected_input=common_middle,
                expected_phase=phase_inputs[phase_name],
            )["values"]
        production_reference_comparison = _production_point_comparisons(
            positions,
            points,
            final_values,
            source_paths["fixed_second_hop_fields"],
            run_dir / "production_point_comparisons.npz",
        )

    report = {
        "schema": SCHEMA,
        "status": "VALID_COMPLETED" if recomputed_g3["production_dtype_pass"] else "VALID_FAILED",
        "scientific_gate": recomputed_g3["scientific_gate"],
        "decision_question": "whether a saved, nonperiodic independent reference can explain the second-hop window difference or support a production defect claim",
        "source_config": str(config_path),
        "source_v3_run_dir": str(v3_run_dir),
        "source_files": {key: str(path) for key, path in source_paths.items() if key != "v3_run_dir"},
        "loaded_source_modules": {
            "v4_runner": str(Path(__file__).resolve()),
            "v3_second_hop_runner": str(Path(inspect.getfile(v3.run)).resolve()),
            "v3_baseline_g3_producer": str(Path(inspect.getfile(baseline._run_g3)).resolve()),
            "gaussian_s12_ols": str(Path(inspect.getfile(v3.GaussianFiberArrayOLS)).resolve()),
            "lp01_and_phase_stack": str(Path(inspect.getfile(v3.FiniteCropPhaseStack)).resolve()),
            "angular_spectrum_propagation": str(Path(inspect.getfile(v3.angular_spectrum_propagate)).resolve()),
        },
        "source_restoration_receipt_expected": "source_restoration_receipt.json",
        "frozen_physical_system": config["optics"],
        "fiber_assumptions": config["fiber"],
        "execution": {
            "python": __import__("sys").executable,
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(torch.device("cuda:0")),
            "elapsed_seconds_observation": time.perf_counter() - started,
            "frequency_row_chunk": 64,
            "position_batch": 16,
        },
        "preflight": json.loads(preflight_path.read_text(encoding="utf-8")),
        "G3_recomputed_from_saved_full_fields": recomputed_g3,
        "Gaussian_LP01_saved_field_readout_crosscheck": fixed_mode_readouts,
        "independent_reference": {
            "status": "VALID" if reference_valid else "INVALID_NONCONVERGED_REFERENCE",
            "definition": "CPU complex128 nonperiodic bandlimited sampled-field angular-spectrum integral over the fixed Nyquist square; no pixel sinc factor, extra spectral filter, spatial-grid expansion, normalization, or phase fit",
            "frequency_nodes_per_axis_GL4": list(NODE_COUNTS),
            "frequency_nodes_per_axis_GL8_cross_order": NODE_COUNTS[-1] if top_two_pass else None,
            "quadrature_domain_hz": [-1.0 / (2.0 * v3.DX_M), 1.0 / (2.0 * v3.DX_M)],
            "reference_field_scale_E0": E0,
            "convergence_tolerance": TOLERANCE,
            "required_top_two_adjacent_intervals_converged": top_two_pass,
            "refinement": refinement,
            "cross_order": cross_order,
            "point_count": len(points),
            "point_groups": {key: (25 if key == "roi_grid_25" else 100 if key == "fiber_nearest_pixel_100" else 125) for key in POINT_GROUPS},
            "raw_reference_levels": [str(path) for by_phase in level_paths.values() for path in by_phase.values()] + [str(path) for path in cross_paths.values()],
            "raw_values_are_reread_before_convergence_decision": True,
            "production_point_comparisons": production_reference_comparison,
            "cause_scope": "125 selected positions only; cannot prove the full common-ROI error mechanism",
        },
        "decision_limits": {
            "G3_consumes_only_original_nine_full_two_hop_production_complex64_common_roi_pairs": True,
            "g3_tolerance": G3_TOLERANCE,
            "point_reference_does_not_substitute_for_full_roi_G3": True,
            "no_production_solver_or_geometry_changed": True,
            "no_dimension_matrix_or_ranking_generated": True,
        },
        "updates_and_forbidden_io": {
            "formal_training_updates": 0,
            "engineering_adam_updates": 0,
            "phase_optimizer_updates": 0,
            "dataset_reads": 0,
            "task_evaluator_calls": 0,
            "fid_calls": 0,
            "loss_changes": 0,
            "material_forward_calls": 0,
            "production_geometry_changes": 0,
            "propagation_solver_replacement": 0,
        },
        "claim_limits": [
            "the nonperiodic integral is an explicit sampled-field model, not a measured continuous aperture",
            "fiber geometry and MFD remain unmeasured simulation assumptions",
            "125 selected positions do not establish the full-ROI error source",
            "an invalid independent reference does not indicate an architecture failure",
            "no OLS capacity or final channel-count conclusion is made",
        ],
    }
    v3._write_json(run_dir / "report.json", report)
    _save_progress(progress_path, {
        "stage": "complete",
        "report": "report.json",
        "reference_status": report["independent_reference"]["status"],
        "G3_status": recomputed_g3["scientific_gate"],
        "formal_training_updates": 0,
    })
    clear_production_transfer_cache()
    torch.cuda.empty_cache()
    gc.collect()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v3-run-dir", type=Path, default=DEFAULT_V3_RUN_DIR)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.v3_run_dir, args.run_dir, preflight_only=args.preflight_only)
    print(json.dumps({
        "status": result.get("status"),
        "scientific_gate": result.get("scientific_gate"),
        "reference_status": result.get("independent_reference", {}).get("status"),
        "run_dir": str(args.run_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
