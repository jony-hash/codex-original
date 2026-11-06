"""Zero-update amplitude and local phase controllability audit for frozen OLS phases."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ldm_project.v2.contracts import ComplexField, OperationProvenance
from scripts.research.lightgen import diagnose_ols_gaussian_dimension_reaudit_v1 as baseline
from scripts.research.lightgen import diagnose_ols_window_dimension_power_v5 as v5


SCHEMA = "ols_four_nine_window_closure.v8"
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_four_nine_window_closure_v8.yaml")
DEFAULT_RUN_DIR = Path("outputs/20261015_ols_four_nine_window_closure_v8/runs/attempt_01")
SOURCE_GRIDS = (3072, 4096, 6144)
GRIDS = (4096, 6144, 8192)
BASE_GRID = 4096
ROI = 2048
DX = 2.0e-6
WAVELENGTH = 532.0e-9
SOURCE_POWER = 249.3e-6
PITCH = 125.0e-6
WAIST = 7.111e-6
TRANSMISSION = 0.889
SIZES = (2, 3)
MODES = ("gaussian_s12", "lp01")
TARGET_NAMES = {
    "2:C_absolute_uniform_delivery": "candidate_2x2_C",
    "3:B_max_neff_fraction": "candidate_3x3_B",
}
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
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} config")
    expected_optics = {
        "wavelength_m": WAVELENGTH, "pixel_pitch_m": DX, "encoder_shape": [100, 100],
        "encoder_aperture_m": 200.0e-6, "propagation_grid_sizes": list(GRIDS),
        "primary_response_grid": BASE_GRID,
        "common_fiber_readout_roi_mm": 4.096, "common_fiber_readout_roi_pixels": ROI,
        "inter_encoder_spacing_m": 1.5e-3, "last_encoder_to_fiber_m": 1.5e-3,
        "layer_power_transmission": TRANSMISSION, "input_power_w": SOURCE_POWER,
        "illumination": "uniform_coherent_top_hat",
    }
    if config.get("optics") != expected_optics:
        raise ValueError("v8 optical system or numerical windows changed")
    expected_fiber = {
        "reference_pitch_m": PITCH, "mode_field_diameter_m": 14.222e-6,
        "gaussian_1e_amplitude_radius_m": WAIST, "lp01_v_number": 2.2,
        "transport": "ideal_identity", "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
    }
    if config.get("fiber") != expected_fiber:
        raise ValueError("v8 fiber model assumptions changed")
    candidates = config.get("candidates", {})
    if candidates.get("zero_phase_controls") != []:
        raise ValueError("v8 is restricted to the two named frozen candidates")
    optimized = candidates.get("optimized", [])
    observed = {item.get("endpoint_key") for item in optimized}
    if observed != set(TARGET_NAMES):
        raise ValueError("v8 candidate set must contain only 2x2-C and 3x3-B")
    expected_source = {
        "v7_run_dir": "outputs/20261014_ols_frozen_phase_controllability_v7/runs/attempt_09",
        "v7_report": "outputs/20261014_ols_frozen_phase_controllability_v7/runs/attempt_09/report.json",
    }
    if config.get("source_evidence") != expected_source:
        raise ValueError("v8 must consume the explicit v7 attempt_09 evidence")
    measurement = config.get("measurement", {})
    if measurement.get("amplitude_probe_seed") != 2026101407 or measurement.get("amplitude_probe_count") != 32:
        raise ValueError("v8 amplitude probe population or seed changed")
    expected_measurement = {
        "amplitude_probe_values": [0.25, 0.75], "fixed_amplitude_controls": [0.5, 1.0],
        "target_base_command": 0.5, "command_bounds": [0.25, 0.75],
        "target_magnitudes_sqrt_input_power": [1.0e-4, 1.0e-3, 1.0e-2],
        "main_relative_svd_threshold": 1.0e-3, "primary_grid": BASE_GRID,
        "pilot_grid_sizes": list(GRIDS), "pilot_complex_precisions": ["complex64", "complex128"],
    }
    if any(measurement.get(key) != value for key, value in expected_measurement.items()):
        raise ValueError("v8 fixed amplitude controls, thresholds, or pilot plan changed")
    if measurement.get("relative_svd_thresholds") != [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4]:
        raise ValueError("v8 singular-value thresholds changed")
    if config.get("gates", {}).get("coefficient_relative_error_max") != 1.0e-4:
        raise ValueError("v8 coefficient gate must remain 1e-4")
    if config.get("gates", {}).get("window_aligned_complex_roi_error_max") != 1.0e-4:
        raise ValueError("v8 common-ROI window gate must remain 1e-4")
    expected_scope = {
        "formal_training_updates": 0, "phase_optimization_updates": 0, "dataset_reads": 0,
        "task_evaluator_calls": 0, "fid_calls": 0, "loss_changes": 0,
        "material_forward_calls": 0, "decoder_or_camera_changes": 0,
        "production_default_migration": "forbidden", "fiber_geometry_changes": "forbidden",
    }
    if config.get("scope") != expected_scope:
        raise ValueError("v8 scope contract changed")
    return config


def _coords(size: int, device: torch.device) -> list[tuple[float, float]]:
    axis = baseline._array_coordinates(size, PITCH, device).detach().cpu().tolist()
    return [(float(y), float(x)) for y in axis for x in axis]


def _coordinate_pool(sizes: list[int], device: torch.device) -> list[tuple[float, float]]:
    unique = {coordinate for size in sizes for coordinate in _coords(size, device)}
    return sorted(unique, key=lambda item: (item[0], item[1]))


def _pool_indices(pool: list[tuple[float, float]], size: int) -> list[int]:
    lookup = {coordinate: index for index, coordinate in enumerate(pool)}
    requested = _coords(size, torch.device("cpu"))
    try:
        return [lookup[coordinate] for coordinate in requested]
    except KeyError as exc:
        raise ValueError(f"array {size}x{size} coordinates are absent from its response pool") from exc


def _load_v7_phases(config: dict[str, Any], root: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    report_path = Path(config["source_evidence"]["v7_report"])
    run_path = Path(config["source_evidence"]["v7_run_dir"])
    report_path = report_path if report_path.is_absolute() else root / report_path
    run_path = run_path if run_path.is_absolute() else root / run_path
    if not report_path.is_file() or not run_path.is_dir():
        raise FileNotFoundError("explicit v7 report or source run directory is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "ols_frozen_phase_controllability.v7" or report.get("status") != "VALID_COMPLETED":
        raise ValueError("v8 requires the named VALID_COMPLETED v7 report")
    phases: dict[str, torch.Tensor] = {}
    source_records: dict[str, Any] = {}
    for endpoint_key, phase_name in TARGET_NAMES.items():
        case = report.get("cases", {}).get(phase_name)
        if not isinstance(case, dict) or case.get("endpoint_key") != endpoint_key or int(case.get("size", -1)) != int(endpoint_key.split(":", 1)[0]):
            raise ValueError(f"v7 saved candidate metadata is missing or inconsistent for {endpoint_key}")
        if case.get("optimized") is not True or case.get("phase_name") != phase_name:
            raise ValueError(f"v7 candidate is not the expected frozen optimized phase: {endpoint_key}")
        phase_path = run_path / "phase_screens_v7.pt"
        if not phase_path.is_file():
            raise FileNotFoundError(f"explicit v7 phase archive is missing: {phase_path}")
        phase_payload = torch.load(phase_path, map_location="cpu", weights_only=True)
        phase = phase_payload.get("phases", {}).get(phase_name)
        if not isinstance(phase, torch.Tensor) or tuple(phase.shape) != (2, 1, 100, 100) or not torch.isfinite(phase).all():
            raise ValueError(f"invalid frozen phase tensor for {endpoint_key}")
        if phase.dtype != torch.float32:
            phase = phase.to(torch.float32)
        phases[phase_name] = phase.clone()
        source_records[phase_name] = {
            "endpoint_key": endpoint_key, "phase_path": str(phase_path),
            "phase_dtype_loaded": str(phase.dtype), "source_v7_status": report.get("status"),
            "source_case_phase_state_id": case.get("phase_state_id"),
            "phase_tensor_shape": list(phase.shape),
        }
    source_ids = phase_payload.get("phase_state_ids", {})
    for name, phase in phases.items():
        if source_ids.get(name) != source_records[name]["source_case_phase_state_id"]:
            raise ValueError(f"v7 phase-state identity mismatch for {name}")
    return phases, {
        "report_path": str(report_path), "run_path": str(run_path), "status": report["status"],
        "selected_endpoints": source_records, "phase_state_ids": {name: source_ids[name] for name in phases},
        "report": report,
    }


def _phase_state_ids(phases: dict[str, torch.Tensor], cases: list[dict[str, Any]], source_ids: dict[str, str] | None = None) -> dict[str, str]:
    names = list(dict.fromkeys(case["phase_name"] for case in cases))
    return {name: (source_ids or {}).get(name, f"phase_v8_{index:02d}_{name}") for index, name in enumerate(names)}


def _phase_transfer_adjoint(value: torch.Tensor, plane, distance_m: float) -> torch.Tensor:
    return baseline._adjoint_transfer(value, plane, distance_m)


def _mode_global(mode_name: str, coordinate: tuple[float, float], roi_plane, global_plane) -> torch.Tensor:
    cy, cx = coordinate
    return v5._mode_on_global_grid(mode_name, cy, cx, roi_plane, global_plane)


def _active_bounds(plane) -> tuple[int, int, int, int]:
    h, w = plane.shape
    y0, x0 = (h - 100) // 2, (w - 100) // 2
    return y0, y0 + 100, x0, x0 + 100


def _matrix_and_phase_jacobian_row(
    output_mode: torch.Tensor,
    entrance: torch.Tensor,
    first_layer_output: torch.Tensor,
    phases: torch.Tensor,
    plane,
    *,
    include_phase_jacobian: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    y0, y1, x0, x1 = _active_bounds(plane)
    back2 = _phase_transfer_adjoint(output_mode, plane, 1.5e-3)
    q2 = back2[y0:y1, x0:x1]
    scale = math.sqrt(TRANSMISSION)
    s2 = torch.polar(torch.full_like(phases[1, 0], scale), phases[1, 0])
    padded_adjoint = torch.zeros_like(output_mode)
    padded_adjoint[y0:y1, x0:x1] = q2 * s2.conj()
    back1 = _phase_transfer_adjoint(padded_adjoint, plane, 1.5e-3)
    q1 = back1[y0:y1, x0:x1]
    s1 = torch.polar(torch.full_like(phases[0, 0], scale), phases[0, 0])
    active = q1 * s1.conj()
    area = DX * DX
    root_area = math.sqrt(area)
    k = active.conj().reshape(-1) * root_area
    h = k * entrance.reshape(-1) * root_area
    if not include_phase_jacobian:
        return k, h, None
    e0 = entrance * 0.5
    e1 = first_layer_output
    j1 = (1j * q1.conj() * s1 * e0 * area).reshape(-1)
    j2 = (1j * q2.conj() * s2 * e1 * area).reshape(-1)
    return k, h, torch.cat((j1, j2))


def _build_response_pool(
    *, run_dir: Path, runtime, roi_plane, entrance: ComplexField, phases: torch.Tensor,
    phase_state_id: str, mode_name: str, grid: int, coordinates: list[tuple[float, float]],
    first_layer_output: torch.Tensor | None, include_phase_jacobian: bool,
) -> dict[str, torch.Tensor]:
    cache_dir = run_dir / "response_rows"
    cache_dir.mkdir(parents=True, exist_ok=True)
    phase_cpu = phases.detach().cpu().to(torch.float32)
    coord_array = np.asarray(coordinates, dtype=np.float64)
    block_size = 4
    k_parts: list[np.ndarray] = []
    h_parts: list[np.ndarray] = []
    j_parts: list[np.ndarray] = []
    source_field = entrance.tensor[0, 0].to(torch.complex128)
    phase_device = phases.to(device=source_field.device, dtype=torch.float64)
    for start in range(0, len(coordinates), block_size):
        stop = min(start + block_size, len(coordinates))
        path = cache_dir / f"{phase_state_id}_{mode_name}_{grid}_rows_{start:03d}_{stop:03d}.npz"
        expected_meta = {
            "schema": "ols_frozen_phase_rows.v1", "phase_state_id": phase_state_id,
            "mode": mode_name, "grid": grid, "start": start, "stop": stop,
            "dx_m": DX, "roi_size": ROI, "includes_phase_jacobian": bool(include_phase_jacobian),
        }
        block_coords = coord_array[start:stop]
        if path.is_file():
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata_json"].item()))
                stored_phase = torch.from_numpy(np.asarray(archive["phase_screens"], dtype=np.float32).copy())
                coords = np.asarray(archive["coordinates_yx_m"], dtype=np.float64)
                k_block = np.asarray(archive["K_dimensionless"], dtype=np.complex128)
                h_block = np.asarray(archive["H_sqrt_w"], dtype=np.complex128)
                j_block = np.asarray(archive["J_phi_sqrt_w_per_rad"], dtype=np.complex128) if include_phase_jacobian else None
            if meta != expected_meta or not torch.equal(stored_phase, phase_cpu) or not np.array_equal(coords, block_coords):
                raise ValueError(f"v7 saved response block does not match the explicit phase and coordinates: {path}")
            if k_block.shape != (stop - start, 10000) or h_block.shape != k_block.shape:
                raise ValueError(f"invalid saved v7 response row shape: {path}")
            if include_phase_jacobian and (j_block is None or j_block.shape != (stop - start, 20000)):
                raise ValueError(f"invalid saved v7 phase Jacobian row shape: {path}")
        else:
            k_rows: list[np.ndarray] = []
            h_rows: list[np.ndarray] = []
            j_rows: list[np.ndarray] = []
            with torch.no_grad():
                for cy, cx in coordinates[start:stop]:
                    mode = _mode_global(mode_name, (cy, cx), roi_plane, runtime.ols_plane)
                    k, h, j = _matrix_and_phase_jacobian_row(
                        mode, source_field, first_layer_output, phase_device, runtime.ols_plane,
                        include_phase_jacobian=include_phase_jacobian,
                    )
                    k_rows.append(k.detach().cpu().numpy().astype(np.complex128, copy=False))
                    h_rows.append(h.detach().cpu().numpy().astype(np.complex128, copy=False))
                    if include_phase_jacobian:
                        assert j is not None
                        j_rows.append(j.detach().cpu().numpy().astype(np.complex128, copy=False))
                    del mode, k, h, j
            k_block = np.stack(k_rows)
            h_block = np.stack(h_rows)
            j_block = np.stack(j_rows) if include_phase_jacobian else None
            temp = path.with_suffix(path.suffix + ".tmp")
            with temp.open("wb") as stream:
                archive_values: dict[str, Any] = {
                    "metadata_json": np.asarray(json.dumps(expected_meta, sort_keys=True)),
                    "phase_screens": phase_cpu.numpy(), "coordinates_yx_m": block_coords,
                    "K_dimensionless": k_block, "H_sqrt_w": h_block,
                }
                if j_block is not None:
                    archive_values["J_phi_sqrt_w_per_rad"] = j_block
                np.savez_compressed(stream, **archive_values)
            temp.replace(path)
        k_parts.append(k_block)
        h_parts.append(h_block)
        if j_block is not None:
            j_parts.append(j_block)
        _write_json(run_dir / "progress.json", {
            "stage": "response_matrix_rows", "phase_state_id": phase_state_id,
            "mode": mode_name, "grid": grid, "completed_rows": stop,
            "total_rows": len(coordinates), "phase_optimization_updates": 0,
        })
        del k_block, h_block, j_block
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    result = {
        "K": torch.from_numpy(np.concatenate(k_parts, axis=0)),
        "H": torch.from_numpy(np.concatenate(h_parts, axis=0)),
        "coordinates_yx_m": torch.from_numpy(coord_array.copy()),
    }
    if include_phase_jacobian:
        result["J_phi"] = torch.from_numpy(np.concatenate(j_parts, axis=0))
    return result


def _mode_profiles(size: int, mode_name: str, roi_plane, device: torch.device):
    if mode_name == "gaussian_s12":
        dy, dx = roi_plane.pixel_pitch_m
        h, w = roi_plane.shape
        y = (torch.arange(h, dtype=torch.float64, device=device) + 0.5) * dy - roi_plane.fov_m[0] / 2 + roi_plane.origin_m[0]
        x = (torch.arange(w, dtype=torch.float64, device=device) + 0.5) * dx - roi_plane.fov_m[1] / 2 + roi_plane.origin_m[1]
        axis = baseline._array_coordinates(size, PITCH, device)
        amp = (2.0 / (math.pi * WAIST**2)) ** 0.25
        return (
            amp * torch.exp(-((y[None] - axis[:, None]) / WAIST).square()),
            amp * torch.exp(-((x[None] - axis[:, None]) / WAIST).square()),
        )
    if mode_name == "lp01":
        axis = baseline._array_coordinates(size, PITCH, device)
        return [baseline._mode_patch("lp01", float(cy), float(cx), roi_plane, 2.2, WAIST) for cy in axis for cx in axis]
    raise ValueError(mode_name)


def _project_roi_batch(fields: torch.Tensor, size: int, mode_name: str, roi_plane) -> torch.Tensor:
    # fields: [batch, 2048, 2048], exactly the shared physical ROI.
    dy, dx = roi_plane.pixel_pitch_m
    if mode_name == "gaussian_s12":
        modes_y, modes_x = _mode_profiles(size, mode_name, roi_plane, fields.device)
        return torch.einsum("yh,bhw,xw->byx", modes_y.to(torch.complex128).conj(), fields.to(torch.complex128), modes_x.to(torch.complex128).conj()).reshape(fields.shape[0], -1) * dy * dx
    outputs: list[torch.Tensor] = []
    profiles = _mode_profiles(size, mode_name, roi_plane, fields.device)
    for profile, (y0, y1, x0, x1) in profiles:
        outputs.append((fields[:, y0:y1, x0:x1].to(torch.complex128) * profile.to(torch.complex128).conj()).sum(dim=(-1, -2)) * dy * dx)
    return torch.stack(outputs, dim=1)


def _crop_roi(fields: torch.Tensor, grid: int) -> torch.Tensor:
    offset = (grid - ROI) // 2
    if offset < 0 or 2 * offset + ROI != grid:
        raise ValueError(f"grid {grid} cannot hold the centered common 4.096 mm ROI")
    return fields[..., offset : offset + ROI, offset : offset + ROI]


def _stack_common_roi_batch(fields: list[torch.Tensor], grid: int) -> torch.Tensor:
    return torch.stack([_crop_roi(field, grid) for field in fields])


def _power_trace_arrays(power_by_pattern: dict[int, dict[str, Any]], grid: int) -> dict[str, np.ndarray]:
    pattern_indices = sorted(power_by_pattern)
    keys = sorted(power_by_pattern[pattern_indices[0]])
    return {
        f"power_trace_{grid}_{key}": np.asarray(
            [power_by_pattern[index][key] for index in pattern_indices], dtype=np.float64
        )
        for key in keys
    }


def _field_relative_error(reference: torch.Tensor, test: torch.Tensor, *, align: bool) -> tuple[float, float]:
    ref = reference.to(torch.complex128).reshape(-1)
    got = test.to(torch.complex128).reshape(-1)
    raw = float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref).clamp_min(EPS))
    if align:
        inner = torch.sum(ref.conj() * got)
        phase = torch.angle(inner)
        got = got * torch.polar(torch.ones((), dtype=torch.float64, device=got.device), -phase)
    else:
        phase = torch.zeros((), dtype=torch.float64, device=ref.device)
    aligned = float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref).clamp_min(EPS))
    return raw, aligned


def _window_pair_metrics(reference: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    ref = reference.to(torch.complex128).reshape(-1)
    got = test.to(torch.complex128).reshape(-1)
    inner = torch.sum(ref.conj() * got)
    phase = torch.angle(inner)
    aligned = got * torch.polar(torch.ones((), dtype=torch.float64), -phase)
    ref_intensity = ref.abs().square()
    got_intensity = got.abs().square()
    power_left, power_right = float(ref_intensity.sum() * DX * DX), float(got_intensity.sum() * DX * DX)
    return {
        "raw_complex_relative_error": float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref).clamp_min(EPS)),
        "common_global_phase_aligned_complex_relative_error": float(torch.linalg.vector_norm(aligned - ref) / torch.linalg.vector_norm(ref).clamp_min(EPS)),
        "common_global_phase_rad": float(phase),
        "intensity_relative_l2_error": float(torch.linalg.vector_norm(got_intensity - ref_intensity) / torch.linalg.vector_norm(ref_intensity).clamp_min(EPS)),
        "roi_power_left_w": power_left,
        "roi_power_right_w": power_right,
        "roi_power_relative_error": abs(power_right - power_left) / max(power_left, EPS),
    }

def _relative_vector_error(expected: torch.Tensor, observed: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(expected - observed) / torch.linalg.vector_norm(observed).clamp_min(EPS))


def _analytic_common_phase_errors(
    base: torch.Tensor,
    first_screen_piston: torch.Tensor,
    second_screen_piston: torch.Tensor,
    opposite_screen_pistons: torch.Tensor,
    phase_rad: float,
) -> dict[str, float]:
    angle = torch.as_tensor(phase_rad, dtype=torch.float64, device=base.device)
    expected_common = base.to(torch.complex128) * torch.polar(torch.ones_like(angle), angle)
    reference = base.to(torch.complex128)
    return {
        "first_screen_common_phase_relative_error": _relative_vector_error(expected_common, first_screen_piston),
        "second_screen_common_phase_relative_error": _relative_vector_error(expected_common, second_screen_piston),
        "opposite_screen_pistons_relative_error": _relative_vector_error(reference, opposite_screen_pistons),
    }


def _spectrum(
    matrix: torch.Tensor,
    thresholds: list[float],
    *,
    singular_value_units: str,
    squared_gain_units: str,
) -> dict[str, Any]:
    singular = torch.linalg.svdvals(matrix)
    maximum = float(singular.max()) if singular.numel() else 0.0
    return {
        "singular_values": singular.detach().cpu().tolist(),
        "maximum_singular_value": maximum,
        "singular_value_squared_gain": maximum**2,
        "singular_value_units": singular_value_units,
        "singular_value_squared_gain_units": squared_gain_units,
        "effective_dimension_by_relative_threshold": {
            str(t): int((singular >= maximum * t).sum()) if maximum > 0.0 else 0 for t in thresholds
        },
    }


def _response_spectra(k: torch.Tensor, h: torch.Tensor, thresholds: list[float]) -> dict[str, Any]:
    real_h = torch.cat((h.real, h.imag), dim=0)
    return {
        "K_complex": _spectrum(k, thresholds, singular_value_units="dimensionless", squared_gain_units="dimensionless squared"),
        "H_complex": _spectrum(h, thresholds, singular_value_units="sqrt(W) per unit command", squared_gain_units="W per squared command norm"),
        "H_real_amplitude_control": _spectrum(real_h, thresholds, singular_value_units="sqrt(W) per unit real-command norm", squared_gain_units="W per squared real-command norm"),
        "definitions": {
            "K": "c=K a; a_p=E_source,p sqrt(pixel_area) t_p has sqrt(W); K dimensionless",
            "H": "H=K diag(E_source sqrt(pixel_area)); c=H t; H is sqrt(W) per unit command",
            "real_control": "stack([Re H; Im H]); t is real and bounded; do not divide its rank by two to claim complex channels",
        },
    }


def _power_metrics(coefficients: torch.Tensor, input_power_w: float, before_roi_power_w: float) -> dict[str, Any]:
    power = coefficients.abs().square().to(torch.float64)
    total = float(power.sum())
    maximum = float(power.max()) if power.numel() else 0.0
    sorted_power = torch.sort(power).values
    p10 = float(torch.quantile(sorted_power, 0.1))
    p50 = float(torch.quantile(sorted_power, 0.5))
    p90 = float(torch.quantile(sorted_power, 0.9))
    mean = float(power.mean())
    std = float(torch.sqrt(torch.mean((power - mean) ** 2)))
    row_size = int(round(math.sqrt(power.numel())))
    edge_indices = []
    corner_indices = []
    if row_size > 1:
        for i in range(power.numel()):
            r, c = divmod(i, row_size)
            if r in (0, row_size - 1) or c in (0, row_size - 1):
                edge_indices.append(i)
            if r in (0, row_size - 1) and c in (0, row_size - 1):
                corner_indices.append(i)
    axis = baseline._array_coordinates(row_size, PITCH, torch.device("cpu")).tolist()
    coordinates = [(y, x) for y in axis for x in axis]
    distances = [math.hypot(y, x) for y, x in coordinates]
    nearest = min(distances)
    center_indices = [i for i, d in enumerate(distances) if abs(d - nearest) <= 1.0e-12]
    center_power = float(power[center_indices].mean())
    edge_power = float(power[edge_indices].mean()) if edge_indices else None
    corner_power = float(power[corner_indices].mean()) if corner_indices else None
    neff = total**2 / max(float(power.square().sum()), EPS)
    utilization = {}
    for db in (-10, -20, -30, -40):
        threshold = maximum * 10.0 ** (db / 10.0)
        utilization[str(db)] = int((power >= threshold).sum()) / int(power.numel()) if maximum > 0.0 else 0.0
    return {
        "fiber_power_w_row_major": power.cpu().tolist(),
        "total_coupled_power_w": total,
        "eta_system_fixed_source": total / input_power_w,
        "eta_fiber_face": total / before_roi_power_w if before_roi_power_w > 0.0 else None,
        "power_participation_N_eff": neff,
        "fiber_utilization_fraction_by_max_relative_threshold_db": utilization,
        "P10_over_P90": p10 / p90 if p90 > 0.0 else None,
        "coefficient_of_variation": std / mean if mean > 0.0 else None,
        "edge_to_nearest_center_power_ratio": edge_power / center_power if edge_power is not None and center_power > 0.0 else None,
        "corner_to_nearest_center_power_ratio": corner_power / center_power if corner_power is not None and center_power > 0.0 else None,
        "power_distribution_w": {"min": float(power.min()), "p10": p10, "p50": p50, "mean": mean, "p90": p90, "max": maximum},
        "center_channel_indices": center_indices,
        "edge_channel_count": len(edge_indices),
        "corner_channel_count": len(corner_indices),
        "edge_ratio_status": "not_applicable_single_channel" if row_size == 1 else "measured",
    }


def _bounded_control(
    m: torch.Tensor, u: torch.Tensor, singular: torch.Tensor, vh: torch.Tensor,
    coefficients: torch.Tensor, size: int, thresholds: list[float], main_threshold: float,
    target: torch.Tensor, command_bounds: tuple[float, float], target_base: float,
) -> tuple[dict[str, Any], torch.Tensor]:
    cutoff = float(singular.max()) * main_threshold if singular.numel() else 0.0
    retained = singular >= cutoff
    inv = torch.where(retained, singular.reciprocal(), torch.zeros_like(singular))
    delta = vh.T @ (inv * (u.T @ target))
    lower, upper = command_bounds
    limits = torch.where(delta > 0, (upper - target_base) / delta.clamp_min(EPS), (lower - target_base) / delta.clamp_max(-EPS))
    feasible_scale = float(torch.clamp(limits.min(), min=0.0, max=1.0))
    applied = delta * feasible_scale
    achieved = m @ applied
    residual = float(torch.linalg.vector_norm(achieved - target) / torch.linalg.vector_norm(target).clamp_min(EPS))
    powers_before = coefficients.abs().square()
    predicted = coefficients + torch.complex(achieved[:size*size], achieved[size*size:])
    powers_after = predicted.abs().square()
    target_channel = int(torch.argmax(target.abs())) % (size * size)
    target_component = "real" if int(torch.argmax(target.abs())) < size * size else "imag"
    channel_delta = predicted - coefficients
    cross = torch.linalg.vector_norm(torch.cat((channel_delta[:target_channel], channel_delta[target_channel + 1 :])))
    target_magnitude = channel_delta[target_channel].abs()
    return {
        "target_component": target_component,
        "target_channel_index": target_channel,
        "svd_threshold": main_threshold,
        "retained_singular_dimension": int(retained.sum()),
        "requested_delta_command_l2": float(torch.linalg.vector_norm(delta)),
        "requested_delta_command_max_abs": float(delta.abs().max()),
        "feasible_fraction_of_requested_step_within_bounds": feasible_scale,
        "target_fully_within_command_bounds": feasible_scale >= 1.0 - 1.0e-12,
        "applied_delta_command_l2": float(torch.linalg.vector_norm(applied)),
        "applied_command_min": float((target_base + applied).min()),
        "applied_command_max": float((target_base + applied).max()),
        "residual_relative_to_original_target": residual,
        "target_channel_delta_sqrt_w": float(target_magnitude),
        "crosstalk_l2_sqrt_w": float(cross),
        "target_to_crosstalk_ratio": float(target_magnitude / cross.clamp_min(EPS)),
        "predicted_power_change_by_channel_w": (powers_after - powers_before).detach().cpu().tolist(),
        "target_magnitude_sqrt_w": float(torch.linalg.vector_norm(target)),
        "sensitivity_threshold_counts": {str(t): int((singular >= singular.max() * t).sum()) for t in thresholds},
    }, applied.detach().cpu()


def _amp_probes(config: dict[str, Any]) -> tuple[list[str], torch.Tensor]:
    rng = np.random.default_rng(int(config["measurement"]["amplitude_probe_seed"]))
    values = np.asarray(config["measurement"]["amplitude_probe_values"], dtype=np.float32)
    probes = rng.choice(values, size=(32, 10000)).astype(np.float32)
    names = [f"binary_probe_{i:02d}" for i in range(32)]
    controls = np.stack((np.full(10000, 0.5, np.float32), np.ones(10000, np.float32)))
    return names + ["uniform_t_0.5", "uniform_t_1.0"], torch.from_numpy(np.concatenate((probes, controls), axis=0))


def _response_at_command(response: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
    return response @ command.to(device=response.device, dtype=response.dtype)


def _compose_forward_validation_pattern(command: torch.Tensor, *, base_amplitude: float = 0.5) -> torch.Tensor:
    command_cpu = command.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    return torch.full_like(command_cpu, float(base_amplitude)) + command_cpu


def _phase_direction_probes(count: int, seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(int(seed))
    directions = rng.standard_normal((count, 2, 1, 100, 100))
    directions /= np.sqrt(np.mean(directions**2, axis=(1, 2, 3, 4), keepdims=True))
    return torch.as_tensor(directions, dtype=torch.float64, device=device)


def _runtime_forward_batch(runtime, entrance: ComplexField, phase: torch.Tensor, patterns: torch.Tensor, grid: int, *, batch_size: int) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    outputs: list[torch.Tensor] = []
    traces: list[torch.Tensor] = []
    device = entrance.tensor.device
    phase_value = phase.to(device=device, dtype=runtime.encoder.phase.dtype)
    with torch.no_grad():
        runtime.encoder.phase.copy_(phase_value)
        for start in range(0, len(patterns), batch_size):
            command = patterns[start : start + batch_size].to(device=device, dtype=entrance.tensor.real.dtype).reshape(-1, 1, 100, 100)
            tensor = entrance.tensor.expand(command.shape[0], -1, -1, -1) * command
            batch_field = ComplexField(tensor, entrance.plane, OperationProvenance("v8_frozen_amplitude_command", entrance.plane.plane_id, None))
            output, _records, statistics = runtime.encoder(batch_field, f"ols_v8_{grid}")
            outputs.extend([value.detach() for value in output.tensor[:, 0]])
            traces.extend([value.detach() for value in statistics["layer_power"]])
            del command, tensor, batch_field, output, _records, statistics
    return outputs, traces


def _phase_jacobian_shape_report(j: torch.Tensor, c: torch.Tensor, thresholds: list[float]) -> dict[str, Any]:
    raw = torch.cat((j.real, j.imag), dim=0)
    common_phase = torch.cat(((-1j * c).real, (-1j * c).imag))
    norm2 = torch.dot(common_phase, common_phase).clamp_min(EPS)
    projected = raw - common_phase[:, None] * ((common_phase @ raw) / norm2)[None]
    return {
        "raw_real_control_spectrum": _spectrum(raw, thresholds, singular_value_units="sqrt(W) per radian-norm", squared_gain_units="W per squared radian-norm"),
        "global_phase_projected_real_control_spectrum": _spectrum(projected, thresholds, singular_value_units="sqrt(W) per radian-norm", squared_gain_units="W per squared radian-norm"),
        "global_phase_direction": "one shared output global-phase direction; projection is for shape-spectrum reporting only",
    }


def _phase_jacobian_preflight(
    *, runtime, entrance: ComplexField, phase: torch.Tensor, phase_state_id: str,
    cases: list[dict[str, Any]], state_pool: dict[str, torch.Tensor], roi_plane,
    device: torch.device, config: dict[str, Any],
) -> dict[str, Any]:
    # Scalar pixel finite differences and fixed random RMS directions use the production phase stack.
    phase_name = state_pool[phase_state_id]["phase_name"]
    phase_cases = [case for case in cases if case["phase_name"] == phase_name]
    size = max(int(case["size"]) for case in phase_cases)
    mode_records: dict[str, dict[str, torch.Tensor]] = {}
    coordinates = state_pool[phase_state_id]["modes"][MODES[0]][BASE_GRID]["coordinates_yx_m"].tolist()
    indices = _pool_indices([tuple(v) for v in coordinates], size)
    for mode in MODES:
        mode_grid = state_pool[phase_state_id]["modes"][mode][BASE_GRID]
        mode_records[mode] = {
            "J_phi": mode_grid["J_phi"].index_select(0, torch.tensor(indices, dtype=torch.long)),
            "H": mode_grid["H"].index_select(0, torch.tensor(indices, dtype=torch.long)),
        }
    t0 = torch.full((10000,), 0.5, dtype=torch.float64)
    coefficients0 = {mode: _response_at_command(record["H"], t0) for mode, record in mode_records.items()}
    base_phase = phase.to(device=device, dtype=torch.float64)
    entrance64 = ComplexField(entrance.tensor.to(torch.complex128), entrance.plane, entrance.provenance)
    pixel_epsilon = float(config["measurement"]["phase_pixel_central_difference_epsilon_rad"])
    pixel_coords = config["measurement"]["phase_pixel_probe_coordinates_yx"]
    def observe(perturbation: torch.Tensor) -> dict[str, torch.Tensor]:
        changed = base_phase + perturbation.to(device=device, dtype=torch.float64)
        patterns = torch.full((1, 10000), 0.5, dtype=torch.float64)
        fields, _trace = _runtime_forward_batch(runtime, entrance64, changed, patterns, BASE_GRID, batch_size=1)
        roi_field = _crop_roi(fields[0], BASE_GRID)
        coefficients = {
            mode: _project_roi_batch(roi_field[None], size, mode, roi_plane)[0].cpu()
            for mode in MODES
        }
        with torch.no_grad():
            runtime.encoder.phase.copy_(base_phase)
        del fields, roi_field
        return coefficients

    finite_differences: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for layer in range(2):
        for yx in pixel_coords:
            y, x = (int(yx[0]), int(yx[1]))
            perturb = torch.zeros_like(base_phase)
            perturb[layer, 0, y, x] = pixel_epsilon
            plus, minus = observe(perturb), observe(-perturb)
            col = layer * 10000 + y * 100 + x
            for mode in MODES:
                fd = (plus[mode] - minus[mode]) / (2.0 * pixel_epsilon)
                predicted = mode_records[mode]["J_phi"][:, col]
                absolute = float(torch.linalg.vector_norm(fd - predicted))
                relative = _relative_vector_error(predicted, fd)
                tol = float(config["gates"]["phase_pixel_jacobian_absolute_tolerance_sqrt_w"]) + float(config["gates"]["phase_pixel_jacobian_relative_tolerance"]) * float(torch.linalg.vector_norm(fd))
                finite_differences[mode].append({
                    "layer_index": layer, "pixel_yx": [y, x], "epsilon_rad": pixel_epsilon,
                    "absolute_error_sqrt_w_per_rad": absolute, "relative_error": relative,
                    "tolerance_sqrt_w_per_rad": tol, "passed": absolute <= tol,
                })
            del plus, minus
    directions = _phase_direction_probes(
        int(config["measurement"]["phase_direction_count"]),
        int(config["measurement"]["phase_direction_seed"]),
        device,
    )
    direction_checks: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for direction_index, direction in enumerate(directions):
        predictions = {
            mode: _response_at_command(record["J_phi"], direction.reshape(-1).cpu())
            for mode, record in mode_records.items()
        }
        for rms in config["measurement"]["phase_direction_rms_rad"]:
            plus = observe(direction * float(rms))
            minus = observe(direction * -float(rms))
            for mode in MODES:
                prediction = predictions[mode]
                central = (plus[mode] - minus[mode]) / (2.0 * float(rms))
                direction_checks[mode].append({
                    "direction_index": direction_index, "direction_rms_rad": float(rms),
                    "central_difference_derivative_relative_error": _relative_vector_error(prediction, central),
                    "positive_linear_prediction_relative_error": _relative_vector_error(coefficients0[mode] + prediction * float(rms), plus[mode]),
                    "negative_linear_prediction_relative_error": _relative_vector_error(coefficients0[mode] - prediction * float(rms), minus[mode]),
                    "acceptance_threshold": "reported as local sensitivity; no extra gate specified",
                })
            del plus, minus, central
    per_mode = {
        mode: {
            "complex128_central_difference_by_screen_and_pixel": finite_differences[mode],
            "all_pixel_checks_pass": all(item["passed"] for item in finite_differences[mode]),
            "fixed_random_direction_checks": direction_checks[mode],
            "coefficients_at_t0_sqrt_w": [[float(v.real), float(v.imag)] for v in coefficients0[mode]],
        }
        for mode in MODES
    }
    return {
        "phase_state_id": phase_state_id, "phase_source_name": phase_name,
        "validated_array_size": size,
        "per_mode": per_mode,
        "all_pixel_checks_pass": all(value["all_pixel_checks_pass"] for value in per_mode.values()),
        "random_direction_seed": int(config["measurement"]["phase_direction_seed"]),
        "phase_optimization_updates": 0,
    }


def _validate_matrix_jacobian_grid(matrix_grid: int, jacobian_grid: int, primary_grid: int) -> None:
    if matrix_grid != primary_grid or jacobian_grid != primary_grid:
        raise ValueError("response matrix and phase Jacobian must use the configured primary grid")


def _pattern_matrix_and_window_pass(item: dict[str, Any], coefficient_gate: float) -> bool:
    errors = item.get("matrix_prediction_relative_error_by_grid", {})
    return bool(item.get("window_qualified", False)) and set(errors) == {str(grid) for grid in GRIDS} and all(
        float(error) <= coefficient_gate for error in errors.values()
    )


def _saved_pair_aligned_errors(pattern_errors: dict[str, Any]) -> dict[str, float]:
    pair_names = ("3072_vs_4096", "3072_vs_6144", "4096_vs_6144")
    result: dict[str, float] = {}
    for name in pair_names:
        item = pattern_errors.get(name)
        if not isinstance(item, dict) or "common_global_phase_aligned_complex_relative_error" not in item:
            raise ValueError(f"v7 saved window error is missing a numeric pair record: {name}")
        result[name] = float(item["common_global_phase_aligned_complex_relative_error"])
    return result


def _replay_v7_worst_binary_probes(
    config: dict[str, Any], phases: dict[str, torch.Tensor], source_record: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    run_path = Path(source_record["run_path"])
    report = source_record["report"]
    roi_plane = v5._make_runtime(ROI, torch.float64, device).ols_plane
    replay: dict[str, Any] = {}
    regenerated_names, regenerated = _amp_probes(config)
    for case_name in TARGET_NAMES.values():
        phase = phases[case_name]
        phase_state_id = source_record["phase_state_ids"][case_name]
        saved_phase = report["direct_probe_window_and_coefficient_checks"][phase_state_id]
        errors = saved_phase["window_common_roi_errors_by_pattern"]
        pattern_name = max(
            (name for name in errors if name.startswith("binary_probe_")),
            key=lambda name: errors[name]["3072_vs_6144"]["common_global_phase_aligned_complex_relative_error"],
        )
        archive_path = run_path / "direct_readouts" / f"{phase_state_id}_amplitude_probe_coefficients.npz"
        if not archive_path.is_file():
            raise FileNotFoundError(f"v7 saved amplitude probe coefficients missing: {archive_path}")
        with np.load(archive_path, allow_pickle=False) as saved:
            names = [str(value) for value in saved["pattern_names"].tolist()]
            index = names.index(pattern_name)
            command = np.asarray(saved["amplitude_commands"][index], dtype=np.float32)
            if regenerated_names[index] != pattern_name or not np.array_equal(command, regenerated[index].numpy()):
                raise ValueError(f"v7 binary probe identity does not match its fixed seed: {pattern_name}")
            saved_coeffs = {grid: {mode: np.asarray(saved[f"coeff_{grid}_{case_name}_{mode}"][index], dtype=np.complex128) for mode in MODES} for grid in SOURCE_GRIDS}
        actual_roi: dict[int, torch.Tensor] = {}
        coefficient_errors: dict[str, dict[str, float]] = {}
        for grid in SOURCE_GRIDS:
            runtime = v5._make_runtime(grid, torch.float32, device)
            entrance = v5._encoder_entrance(runtime, torch.float32, device)
            fields, _ = _runtime_forward_batch(runtime, entrance, phase, torch.from_numpy(command[None]), grid, batch_size=1)
            roi = _crop_roi(fields[0], grid)
            actual_roi[grid] = roi.detach().cpu()
            coefficient_errors[str(grid)] = {}
            for mode in MODES:
                expected = torch.from_numpy(saved_coeffs[grid][mode])
                size = int(round(math.sqrt(expected.numel())))
                actual = _project_roi_batch(roi[None], size, mode, roi_plane)[0].detach().cpu()
                coefficient_errors[str(grid)][mode] = _relative_vector_error(actual, expected)
            del runtime, entrance, fields, roi
            torch.cuda.empty_cache()
            gc.collect()
        actual_pairs: dict[str, bool] = {}
        saved_pairs: dict[str, bool] = {}
        for left, right in ((3072, 4096), (3072, 6144), (4096, 6144)):
            _, aligned = _field_relative_error(actual_roi[left], actual_roi[right], align=True)
            key = f"{left}_vs_{right}"
            actual_pairs[key] = aligned <= float(config["gates"]["window_aligned_complex_roi_error_max"])
            saved_pairs[key] = bool(errors[pattern_name][key]["passed"])
        replay[case_name] = {
            "phase_state_id": phase_state_id, "pattern_name": pattern_name, "pattern_index": index,
            "probe_identity_reproduced": True, "v7_saved_aligned_errors": {
                **_saved_pair_aligned_errors(errors[pattern_name])
            },
            "actual_replay_window_classification": actual_pairs, "v7_saved_window_classification": saved_pairs,
            "classification_reproduced": actual_pairs == saved_pairs,
            "coefficient_relative_error_by_grid_and_mode": coefficient_errors,
        }
        del actual_roi
    return replay


def _preflight(config: dict[str, Any], config_path: Path, run_dir: Path, root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("v8 preflight requires AutoDL CUDA")
    device = torch.device("cuda:0")
    phases, source_record = _load_v7_phases(config, root)
    checks: dict[str, Any] = {}
    roi_runtime = v5._make_runtime(ROI, torch.float64, device)
    roi_plane = roi_runtime.ols_plane

    runtime = v5._make_runtime(BASE_GRID, torch.float64, device)
    active = torch.randn((100, 100), dtype=torch.complex128, device=device)
    random_mode = torch.randn((BASE_GRID, BASE_GRID), dtype=torch.complex128, device=device)
    y0, y1, x0, x1 = _active_bounds(runtime.ols_plane)
    padded = torch.zeros((BASE_GRID, BASE_GRID), dtype=torch.complex128, device=device)
    padded[y0:y1, x0:x1] = active
    adjoint = baseline._adjoint_transfer(random_mode, runtime.ols_plane, 1.5e-3)
    transfer = baseline._transfer_function(runtime.ols_plane.shape, pixel_pitch_m=runtime.ols_plane.pixel_pitch_m,
        wavelength_m=runtime.ols_plane.wavelength_m, distance_m=1.5e-3, device=device, real_dtype=torch.float64)
    lhs = torch.vdot(torch.fft.ifft2(torch.fft.fft2(padded, norm="ortho") * transfer, norm="ortho").reshape(-1), random_mode.reshape(-1))
    rhs = torch.vdot(padded.reshape(-1), adjoint.reshape(-1))
    checks["propagation_adjoint_inner_product_relative_error"] = float((lhs - rhs).abs() / lhs.abs().clamp_min(EPS))
    checks["propagation_adjoint_inner_product_pass"] = checks["propagation_adjoint_inner_product_relative_error"] <= 1.0e-10

    roi_field = torch.complex(torch.randn((ROI, ROI), device=device), torch.randn((ROI, ROI), device=device)).to(torch.complex128)
    roi_power = float(roi_field.abs().square().sum() * DX * DX)
    passivity = {}
    for mode in MODES:
        coefficients = _project_roi_batch(roi_field[None], 3, mode, roi_plane)[0]
        captured = float(coefficients.abs().square().sum())
        passivity[mode] = {"input_power": roi_power, "captured_power": captured, "ratio": captured / roi_power}
    checks["mode_projection_passivity"] = passivity
    checks["mode_projection_passivity_pass"] = all(item["ratio"] <= 1.0 + 1.0e-5 for item in passivity.values())
    direct = _project_roi_batch(roi_field[None], 3, "gaussian_s12", roi_plane)[0]
    checks["omitted_pixel_area_rejected"] = _relative_vector_error(direct / DX**2, direct) > 1.0e-4
    phase_test = torch.tensor([[1.0 + 2.0j, -0.5 + 0.75j]], dtype=torch.complex128) * torch.exp(torch.tensor(0.37j, dtype=torch.complex128))
    checks["dropped_complex_phase_rejected"] = _relative_vector_error(phase_test.real.to(torch.complex128), phase_test) > 1.0e-4
    axis = torch.arange(BASE_GRID, dtype=torch.float64, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    centered = torch.exp(-((yy - BASE_GRID // 2).square() + (xx - BASE_GRID // 2).square()) / (2.0 * 30.0**2)).to(torch.complex128)
    correct = _crop_roi(centered, BASE_GRID)
    shifted = centered[BASE_GRID // 2 - ROI // 2 + 1:BASE_GRID // 2 + ROI // 2 + 1, BASE_GRID // 2 - ROI // 2:BASE_GRID // 2 + ROI // 2]
    checks["misaligned_crop_negative_control_relative_error"] = _relative_vector_error(shifted, correct)
    checks["misaligned_crop_rejected"] = checks["misaligned_crop_negative_control_relative_error"] > 1.0e-4
    try:
        _validate_matrix_jacobian_grid(BASE_GRID, 3072, BASE_GRID)
        checks["mixed_primary_matrix_jacobian_grid_rejected"] = False
    except ValueError:
        checks["mixed_primary_matrix_jacobian_grid_rejected"] = True

    source_record["v7_worst_probe_replay"] = _replay_v7_worst_binary_probes(config, phases, source_record, device)
    replay_items = list(source_record["v7_worst_probe_replay"].values())
    checks["v7_worst_probe_replay_pass"] = all(item["classification_reproduced"] and max(
        error for grid_values in item["coefficient_relative_error_by_grid_and_mode"].values() for error in grid_values.values()
    ) <= float(config["gates"]["coefficient_relative_error_max"]) for item in replay_items)

    entrance64 = v5._encoder_entrance(runtime, torch.float64, device)
    phase_epsilon = float(config["measurement"]["phase_pixel_central_difference_epsilon_rad"])
    piston_gate = float(config["gates"]["coefficient_relative_error_max"])
    piston_checks: dict[str, Any] = {}
    for candidate in config["candidates"]["optimized"]:
        phase_name = TARGET_NAMES[candidate["endpoint_key"]]
        phase = phases[phase_name].to(device=device, dtype=torch.float64)
        first_piston = phase.clone()
        first_piston[0] += phase_epsilon
        second_piston = phase.clone()
        second_piston[1] += phase_epsilon
        opposite_pistons = phase.clone()
        opposite_pistons[0] += phase_epsilon
        opposite_pistons[1] -= phase_epsilon
        command = torch.full((1, 10000), 0.5, dtype=torch.float64, device=device)
        readouts: dict[str, dict[str, torch.Tensor]] = {}
        for name, tested_phase in (
            ("base", phase),
            ("first_screen", first_piston),
            ("second_screen", second_piston),
            ("opposite_screens", opposite_pistons),
        ):
            outputs, traces = _runtime_forward_batch(runtime, entrance64, tested_phase, command, BASE_GRID, batch_size=1)
            roi = _crop_roi(outputs[0], BASE_GRID)
            readouts[name] = {
                mode: _project_roi_batch(roi[None], int(candidate["size"]), mode, roi_plane)[0]
                for mode in MODES
            }
            del outputs, traces, roi
        per_mode = {}
        for mode in MODES:
            errors = _analytic_common_phase_errors(
                readouts["base"][mode],
                readouts["first_screen"][mode],
                readouts["second_screen"][mode],
                readouts["opposite_screens"][mode],
                phase_epsilon,
            )
            per_mode[mode] = {**errors, "passed": max(errors.values()) <= piston_gate}
        piston_checks[candidate["name"]] = {
            "phase_state_id": phase_name,
            "grid": BASE_GRID,
            "phase_epsilon_rad": phase_epsilon,
            "coefficient_relative_error_gate": piston_gate,
            "per_mode": per_mode,
            "passed": all(item["passed"] for item in per_mode.values()),
        }
        del phase, first_piston, second_piston, opposite_pistons, command, readouts
    checks["analytic_common_phase_control"] = piston_checks
    checks["analytic_common_phase_control_pass"] = all(item["passed"] for item in piston_checks.values())

    del runtime, entrance64, active, random_mode, padded, adjoint, transfer, roi_field, centered, yy, xx, roi_runtime
    torch.cuda.empty_cache()
    runtime = v5._make_runtime(8192, torch.float32, device)
    entrance = v5._encoder_entrance(runtime, torch.float32, device)
    first_phase = next(iter(phases.values()))
    command = torch.full((1, 10000), 0.5, dtype=torch.float32)
    outputs, traces = _runtime_forward_batch(runtime, entrance, first_phase, command, 8192, batch_size=1)
    roi = _crop_roi(outputs[0], 8192)
    coeff = _project_roi_batch(roi[None], 3, "gaussian_s12", v5._make_runtime(ROI, torch.float64, device).ols_plane)[0]
    checks["8192_representative_forward"] = {
        "field_finite": bool(torch.isfinite(outputs[0]).all()),
        "coefficient_norm_sqrt_w": float(torch.linalg.vector_norm(coeff)),
        "layer_power_trace_finite": bool(torch.isfinite(traces[0]).all()),
    }
    checks["8192_representative_forward_pass"] = checks["8192_representative_forward"]["field_finite"] and checks["8192_representative_forward"]["coefficient_norm_sqrt_w"] > 0.0 and checks["8192_representative_forward"]["layer_power_trace_finite"]
    all_checks = [checks["propagation_adjoint_inner_product_pass"], checks["mode_projection_passivity_pass"],
        checks["omitted_pixel_area_rejected"], checks["dropped_complex_phase_rejected"], checks["misaligned_crop_rejected"],
        checks["mixed_primary_matrix_jacobian_grid_rejected"], checks["v7_worst_probe_replay_pass"],
        checks["analytic_common_phase_control_pass"], checks["8192_representative_forward_pass"]]
    result = {"schema": "ols_four_nine_window_closure.v8.preflight", "status": "PASS" if all(all_checks) else "FAIL_ENGINEERING_PREFLIGHT",
        "passed": all(all_checks), "checks": checks, "input_phase_sources": {key: value for key, value in source_record.items() if key != "report"},
        "runtime_module_sources": {"v8_cli": str(Path(__file__).resolve()), "v7_baseline": str(Path(baseline.__file__).resolve()),
            "v5_runtime": str(Path(v5.__file__).resolve())},
        "cuda_device": torch.cuda.get_device_name(device), "probe_pattern_count": len(_amp_probes(config)[0]),
        "formal_training_updates": 0, "phase_optimization_updates": 0, "dataset_reads": 0, "task_evaluator_calls": 0, "fid_calls": 0}
    _write_json(run_dir / "preflight.json", result)
    del runtime, entrance, outputs, traces, roi, coeff
    torch.cuda.empty_cache()
    return result

def _build_cases(config: dict[str, Any], phases: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for candidate in config["candidates"]["optimized"]:
        endpoint_key = candidate["endpoint_key"]
        cases.append({
            "case_name": candidate["name"], "size": int(candidate["size"]),
            "objective": candidate["objective"], "endpoint_key": endpoint_key,
            "phase_name": TARGET_NAMES[endpoint_key], "optimized": True,
        })
    return cases


def _row_selection(pool_result: dict[str, torch.Tensor], size: int, device: torch.device) -> dict[str, torch.Tensor]:
    indices = torch.tensor(_pool_indices([tuple(v) for v in pool_result["coordinates_yx_m"].cpu().tolist()], size), dtype=torch.long)
    return {
        name: pool_result[name].index_select(0, indices).to(device=device)
        for name in ("K", "H", "J_phi")
        if name in pool_result
    }


def _control_targets_all_channels(h: torch.Tensor, coefficients: torch.Tensor, size: int, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    threshold = float(config["measurement"]["main_relative_svd_threshold"])
    thresholds = [float(value) for value in config["measurement"]["relative_svd_thresholds"]]
    target_scales = [float(value) * math.sqrt(SOURCE_POWER) for value in config["measurement"]["target_magnitudes_sqrt_input_power"]]
    count = size * size
    center = int(_power_metrics(coefficients, SOURCE_POWER, 1.0)["center_channel_indices"][0])
    edge = (size // 2)
    corner = 0
    named_channels = [("center", center), ("edge", edge), ("corner", corner)]
    m = torch.cat((h.real, h.imag), dim=0)
    u, singular, vh = torch.linalg.svd(m, full_matrices=False)
    retained = singular >= singular.max() * threshold
    kept = torch.where(retained)[0]
    strongest = 0
    weakest = int(kept[-1]) if len(kept) else 0
    records: list[dict[str, Any]] = []
    direct_patterns: list[tuple[str, torch.Tensor]] = []
    command_payload: list[tuple[str, torch.Tensor]] = []
    middle_index = 1
    for scale_index, magnitude in enumerate(target_scales):
        for channel in range(count):
            for component, offset in (("real", 0), ("imag", count)):
                target = torch.zeros((2 * count,), dtype=torch.float64, device=h.device)
                target[offset + channel] = magnitude
                rec, delta = _bounded_control(
                    m, u, singular, vh, coefficients, size, thresholds, threshold,
                    target, tuple(float(value) for value in config["measurement"]["command_bounds"]),
                    float(config["measurement"]["target_base_command"]),
                )
                command_id = f"target_channel_{channel:02d}_{component}_scale_{scale_index}"
                command_payload.append((command_id, delta))
                rec.update({"command_id": command_id, "target_family": "single_channel_basis",
                    "target_channel_index": channel, "target_component": component,
                    "target_magnitude_scale_index": scale_index,
                    "target_magnitude_requested_sqrt_w": magnitude})
                records.append(rec)
                if scale_index == middle_index and (channel in {value for _, value in named_channels}):
                    labels = [label for label, selected in named_channels if selected == channel]
                    for label in labels:
                        direct_patterns.append((f"control_{label}_{component}_at_{magnitude:.6e}", delta))
    for label, direction_index in (("strongest_retained_direction", strongest), ("weakest_retained_direction", weakest)):
        delta = vh[direction_index] * (0.25 / vh[direction_index].abs().max().clamp_min(EPS))
        delta = delta.to(torch.float64)
        direct_patterns.append((f"control_{label}_sigma_index_{direction_index}", delta))
        command_payload.append((f"control_{label}_sigma_index_{direction_index}", delta))
        records.append({"target_family": "real_amplitude_singular_direction", "direction_label": label,
            "command_id": f"control_{label}_sigma_index_{direction_index}", "singular_index": direction_index,
            "singular_value_sqrt_w": float(singular[direction_index]), "delta_command_l2": float(torch.linalg.vector_norm(delta)),
            "delta_command_max_abs": float(delta.abs().max()), "command_min": float((0.5 + delta).min()),
            "command_max": float((0.5 + delta).max())})
    return records, direct_patterns, command_payload

def _window_probe_and_direct_measurement(
    *, config: dict[str, Any], cases: list[dict[str, Any]], phases: dict[str, torch.Tensor],
    state_ids: dict[str, str], state_pool: dict[str, Any], run_dir: Path, device: torch.device,
    direct_patterns_by_case_mode: dict[str, dict[str, list[tuple[str, torch.Tensor]]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    probe_names, base_patterns = _amp_probes(config)
    roi_runtime = v5._make_runtime(ROI, torch.float64, device)
    roi_plane = roi_runtime.ols_plane
    phase_report: dict[str, Any] = {}
    phase_pattern_counts: dict[str, int] = {}
    for phase_name in dict.fromkeys(case["phase_name"] for case in cases):
        phase_state_id = state_ids[phase_name]
        relevant_cases = [case for case in cases if case["phase_name"] == phase_name]
        patterns_by_name: dict[str, torch.Tensor] = {name: row.clone() for name, row in zip(probe_names, base_patterns, strict=True)}
        controls_by_case: dict[str, Any] = {}
        for case in relevant_cases:
            size = int(case["size"])
            case_controls: dict[str, Any] = {}
            for mode in MODES:
                direct_patterns = direct_patterns_by_case_mode[case["case_name"]][mode]
                for label, command in direct_patterns:
                    pattern_name = f"{case['case_name']}__{mode}__{label}"
                    patterns_by_name[pattern_name] = _compose_forward_validation_pattern(command)
                case_controls[mode] = {"direct_pattern_names": [f"{case['case_name']}__{mode}__{label}" for label, _ in direct_patterns]}
            controls_by_case[case["case_name"]] = case_controls
        pattern_names = list(patterns_by_name)
        phase_pattern_counts[phase_state_id] = len(pattern_names)
        patterns = torch.stack([patterns_by_name[name] for name in pattern_names]).to(torch.float32)
        # The baseline ROI fields are retained in host memory only until both larger windows are compared.
        baseline_roi: list[torch.Tensor] = []
        middle_roi: list[torch.Tensor] = []
        per_grid_outputs: dict[int, dict[str, Any]] = {}
        window_errors: dict[str, Any] = {name: {} for name in pattern_names}
        direct_coefficients: dict[int, dict[str, dict[int, dict[str, torch.Tensor]]]] = {}
        direct_power: dict[int, dict[int, dict[str, float]]] = {}
        batch_size = int(config["measurement"]["probe_forward_batch_size"])
        for grid in GRIDS:
            runtime = v5._make_runtime(grid, torch.float32, device)
            entrance = v5._encoder_entrance(runtime, torch.float32, device)
            grid_coeffs: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
            grid_powers: dict[int, dict[str, float]] = {}
            for start in range(0, len(pattern_names), batch_size):
                stop = min(start + batch_size, len(pattern_names))
                batch_patterns = patterns[start:stop]
                batch_outputs, batch_traces = _runtime_forward_batch(runtime, entrance, phases[phase_name], batch_patterns, grid, batch_size=1 if grid == 8192 else batch_size)
                rois = _stack_common_roi_batch(batch_outputs, grid)
                if grid == GRIDS[0]:
                    baseline_roi.extend([value.detach().cpu() for value in rois])
                elif grid == GRIDS[1]:
                    for local, field in enumerate(rois):
                        index = start + local
                        metrics = _window_pair_metrics(baseline_roi[index], field.detach().cpu())
                        metrics["passed"] = metrics["common_global_phase_aligned_complex_relative_error"] <= float(config["gates"]["window_aligned_complex_roi_error_max"])
                        window_errors[pattern_names[index]][f"{GRIDS[0]}_vs_{grid}"] = metrics
                    middle_roi.extend([value.detach().cpu() for value in rois])
                else:
                    for local, field in enumerate(rois):
                        index = start + local
                        gate = float(config["gates"]["window_aligned_complex_roi_error_max"])
                        first_metrics = _window_pair_metrics(baseline_roi[index], field.detach().cpu())
                        first_metrics["passed"] = first_metrics["common_global_phase_aligned_complex_relative_error"] <= gate
                        middle_metrics = _window_pair_metrics(middle_roi[index], field.detach().cpu())
                        middle_metrics["passed"] = middle_metrics["common_global_phase_aligned_complex_relative_error"] <= gate
                        window_errors[pattern_names[index]][f"{GRIDS[0]}_vs_{grid}"] = first_metrics
                        window_errors[pattern_names[index]][f"{GRIDS[1]}_vs_{grid}"] = middle_metrics
                for case in relevant_cases:
                    size = int(case["size"])
                    case_key = case["case_name"]
                    grid_coeffs.setdefault(case_key, {})
                    for mode in MODES:
                        values = _project_roi_batch(rois, size, mode, roi_plane).detach().cpu()
                        grid_coeffs[case_key].setdefault(size, {})[mode] = torch.cat((grid_coeffs[case_key].get(size, {}).get(mode, torch.empty((0, size*size), dtype=torch.complex128)), values), dim=0)
                for local, field in enumerate(batch_outputs):
                    roi_power = float(rois[local].abs().square().sum().to(torch.float64) * DX * DX)
                    full_power = float(field.abs().square().sum().to(torch.float64) * DX * DX)
                    grid_powers[start + local] = {
                        "source_input_power_w": SOURCE_POWER,
                        "post_amplitude_command_power_w": float((entrance.tensor[0, 0].abs().square().to(torch.float64) * batch_patterns[local].to(device=device, dtype=torch.float64).reshape(100, 100).square()).sum() * DX * DX),
                        "layer_1_power_before_and_after_transmission_w": [float(batch_traces[local][0, 0]), float(batch_traces[local][0, 1])],
                        "layer_2_power_before_and_after_transmission_w": [float(batch_traces[local][1, 0]), float(batch_traces[local][1, 1])],
                        "full_global_output_power_w": full_power,
                        "common_roi_power_before_fiber_w": roi_power,
                    }
                del rois, batch_outputs, batch_traces, batch_patterns
            direct_coefficients[grid] = grid_coeffs
            direct_power[grid] = grid_powers
            per_grid_outputs[grid] = {"phase_dtype": str(runtime.encoder.phase.dtype), "pattern_count": len(pattern_names)}
            del runtime, entrance
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        for name in pattern_names:
            window_errors[name]["all_three_window_comparisons_pass"] = all(
                window_errors[name].get(f"{left}_vs_{right}", {}).get("passed", False)
                for left, right in ((GRIDS[0], GRIDS[1]), (GRIDS[0], GRIDS[2]), (GRIDS[1], GRIDS[2]))
            )
            pattern_index = pattern_names.index(name)
            for left, right in ((GRIDS[0], GRIDS[1]), (GRIDS[0], GRIDS[2]), (GRIDS[1], GRIDS[2])):
                pair_key = f"{left}_vs_{right}"
                latent_errors: dict[str, Any] = {}
                for case in relevant_cases:
                    case_name, size = case["case_name"], int(case["size"])
                    for mode in MODES:
                        cref = direct_coefficients[left][case_name][size][mode][pattern_index].to(torch.complex128)
                        cgot = direct_coefficients[right][case_name][size][mode][pattern_index].to(torch.complex128)
                        phase = torch.angle(torch.sum(cref.conj() * cgot))
                        aligned = cgot * torch.polar(torch.ones((), dtype=torch.float64), -phase)
                        pref, pgot = cref.abs().square(), cgot.abs().square()
                        latent_errors[f"{case_name}/{mode}"] = {
                            "complex_latent_raw_relative_l2": float(torch.linalg.vector_norm(cgot - cref) / torch.linalg.vector_norm(cref).clamp_min(EPS)),
                            "complex_latent_common_global_phase_aligned_relative_l2": float(torch.linalg.vector_norm(aligned - cref) / torch.linalg.vector_norm(cref).clamp_min(EPS)),
                            "common_global_phase_rad": float(phase),
                            "total_coupled_power_relative_error": abs(float(pgot.sum() - pref.sum())) / max(float(pref.sum()), EPS),
                            "per_fiber_power_relative_l2": float(torch.linalg.vector_norm(pgot - pref) / torch.linalg.vector_norm(pref).clamp_min(EPS)),
                            "per_fiber_power_error_relative_to_floor": (torch.abs(pgot - pref) / torch.clamp(pref, min=float(config["measurement"].get("per_fiber_power_denominator_floor_w", 1.0e-18)))).tolist(),
                        }
                window_errors[name][pair_key]["fiber_modes"] = latent_errors
        predictions: dict[str, Any] = {}
        for case in relevant_cases:
            case_name, size = case["case_name"], int(case["size"])
            predictions[case_name] = {"by_mode": {}}
            for mode in MODES:
                result_rows = []
                by_grid = {}
                for grid in GRIDS:
                    selected_h = _row_selection(state_pool[phase_state_id]["modes"][mode][grid], size, torch.device("cpu"))["H"]
                    observed = direct_coefficients[grid][case_name][size][mode]
                    errors = []
                    for index, command in enumerate(patterns):
                        predicted = selected_h @ command.to(torch.complex128)
                        error = _relative_vector_error(predicted, observed[index])
                        errors.append(error)
                        if grid == GRIDS[0]:
                            result_rows.append({"pattern": pattern_names[index], "matrix_prediction_relative_error_by_grid": {},
                                "window_qualified": window_errors[pattern_names[index]]["all_three_window_comparisons_pass"]})
                        result_rows[index]["matrix_prediction_relative_error_by_grid"][str(grid)] = error
                    by_grid[str(grid)] = {"max_coefficient_relative_error": max(errors, default=0.0), "all_patterns_pass": all(value <= float(config["gates"]["coefficient_relative_error_max"]) for value in errors)}
                for item in result_rows:
                    item["coefficient_relative_error"] = item["matrix_prediction_relative_error_by_grid"][str(BASE_GRID)]
                    item["matrix_prediction_all_grids_pass"] = set(item["matrix_prediction_relative_error_by_grid"]) == {str(grid) for grid in GRIDS} and all(
                        error <= float(config["gates"]["coefficient_relative_error_max"])
                        for error in item["matrix_prediction_relative_error_by_grid"].values()
                    )
                    item["passed"] = item["window_qualified"] and item["matrix_prediction_all_grids_pass"]
                binary_checks = [item for item in result_rows if item["pattern"].startswith("binary_probe_")]
                observed_primary = direct_coefficients[GRIDS[0]][case_name][size][mode]
                baseline_coefficients = observed_primary[pattern_names.index("uniform_t_0.5")]
                pattern_power_metrics = [
                    {
                        "pattern": pattern_names[index],
                        **_power_metrics(
                            observed_primary[index], SOURCE_POWER,
                            direct_power[GRIDS[0]][index]["common_roi_power_before_fiber_w"],
                        ),
                    }
                    for index in range(len(pattern_names))
                ]
                direct_effects = []
                for direct_name in controls_by_case[case_name][mode]["direct_pattern_names"]:
                    direct_index = pattern_names.index(direct_name)
                    coefficient_after = observed_primary[direct_index]
                    delta_coefficients = coefficient_after - baseline_coefficients
                    delta_magnitude = delta_coefficients.abs()
                    tail = direct_name.rsplit("__", 1)[-1]
                    target_label = next((label for label in ("center", "edge", "corner") if f"_{label}_" in tail), None)
                    if target_label is not None:
                        candidates = _power_metrics(baseline_coefficients, SOURCE_POWER, 1.0)["center_channel_indices"] if target_label == "center" else None
                        if target_label == "center":
                            target_index = int(candidates[0])
                        elif target_label == "corner":
                            target_index = 0
                        else:
                            target_index = (size // 2)
                    else:
                        target_index = int(torch.argmax(delta_magnitude))
                    other_indices = [index for index in range(size * size) if index != target_index]
                    crosstalk = torch.linalg.vector_norm(delta_coefficients[other_indices]) if other_indices else torch.zeros((), dtype=torch.float64)
                    before_power = baseline_coefficients.abs().square()
                    after_power = coefficient_after.abs().square()
                    direct_effects.append({
                        "pattern": direct_name,
                        "window_qualified": window_errors[direct_name]["all_three_window_comparisons_pass"],
                        "matrix_prediction_relative_error": result_rows[pattern_names.index(direct_name)]["coefficient_relative_error"],
                        "assumed_target_label": target_label,
                        "target_channel_index": target_index,
                        "target_channel_delta_sqrt_w": float(delta_magnitude[target_index]),
                        "other_channel_crosstalk_l2_sqrt_w": float(crosstalk),
                        "target_to_crosstalk_ratio": float(delta_magnitude[target_index] / crosstalk.clamp_min(EPS)),
                        "complex_coefficient_delta_by_channel_sqrt_w": [[float(value.real), float(value.imag)] for value in delta_coefficients],
                        "power_before_by_channel_w": before_power.tolist(),
                        "power_after_by_channel_w": after_power.tolist(),
                        "power_change_by_channel_w": (after_power - before_power).tolist(),
                        "total_power_before_w": float(before_power.sum()),
                        "total_power_after_w": float(after_power.sum()),
                    })
                predictions[case_name]["by_mode"][mode] = {
                    "matrix_to_production_forward_by_grid": by_grid,
                    "per_pattern_matrix_and_window_checks": result_rows,
                    f"power_metrics_by_pattern_{BASE_GRID}": pattern_power_metrics,
                    f"observed_direct_control_effects_{BASE_GRID}": direct_effects,
                    "qualifying_binary_probe_count": sum(_pattern_matrix_and_window_pass(item, float(config["gates"]["coefficient_relative_error_max"])) for item in binary_checks),
                    "binary_probe_count": len(binary_checks),
                    "all_32_binary_probe_checks_pass": len(binary_checks) == 32 and all(_pattern_matrix_and_window_pass(item, float(config["gates"]["coefficient_relative_error_max"])) for item in binary_checks),
                    "uniform_t_0.5_window_qualified": window_errors["uniform_t_0.5"]["all_three_window_comparisons_pass"],
                    "uniform_t_0.5_matrix_and_window_pass": _pattern_matrix_and_window_pass(next(item for item in result_rows if item["pattern"] == "uniform_t_0.5"), float(config["gates"]["coefficient_relative_error_max"])),
                    "fixed_control_checks": {name: _pattern_matrix_and_window_pass(next(item for item in result_rows if item["pattern"] == name), float(config["gates"]["coefficient_relative_error_max"])) for name in ("uniform_t_0.5", "uniform_t_1.0")},
                    "selected_direct_control_count": len(controls_by_case[case_name][mode]["direct_pattern_names"]),
                    "all_selected_direct_control_checks_pass": bool(controls_by_case[case_name][mode]["direct_pattern_names"]) and all(
                        next((_pattern_matrix_and_window_pass(item, float(config["gates"]["coefficient_relative_error_max"])) for item in result_rows if item["pattern"] == direct_name), False)
                        for direct_name in controls_by_case[case_name][mode]["direct_pattern_names"]
                    ),
                }
        # Save direct complex coefficients and powers as compact typed arrays, never as rounded display strings.
        phase_npz = run_dir / "direct_readouts" / f"{phase_state_id}_amplitude_probe_coefficients.npz"
        phase_npz.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"pattern_names": np.asarray(pattern_names, dtype="U96"), "amplitude_commands": patterns.numpy()}
        for grid in GRIDS:
            for case in relevant_cases:
                for mode in MODES:
                    arrays[f"coeff_{grid}_{case['case_name']}_{mode}"] = direct_coefficients[grid][case["case_name"]][int(case["size"])][mode].numpy().astype(np.complex128, copy=False)
            arrays.update(_power_trace_arrays(direct_power[grid], grid))
        temp = phase_npz.with_suffix(".npz.tmp")
        with temp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temp.replace(phase_npz)
        phase_report[phase_state_id] = {
            "source_phase_names": [name for name, state_id in state_ids.items() if state_id == phase_state_id],
            "cases": [case["case_name"] for case in relevant_cases],
            "measurement_grids": per_grid_outputs,
            "amplitude_probe_seed": int(config["measurement"]["amplitude_probe_seed"]),
            "amplitude_probe_values": [0.25, 0.75],
            "probe_count": 32,
            "fixed_controls": {"t_0.5": "included", "t_1.0": "included"},
            "window_common_roi_errors_by_pattern": window_errors,
            "per_candidate_matrix_to_direct_checks": predictions,
            "power_and_layer_trace_by_pattern_and_grid": {str(grid): direct_power[grid] for grid in GRIDS},
            "target_control_design": controls_by_case,
            "coefficient_archive": str(phase_npz.relative_to(run_dir)).replace("\\", "/"),
            f"{BASE_GRID}_t_0.5_reference_field_archive": None,
        }
        # Keep one audit-ready complex64 field per unique frozen phase state.
        base_index = pattern_names.index("uniform_t_0.5")
        field_path = run_dir / "representative_fields" / f"{phase_state_id}_complex64_{BASE_GRID}_t05.npz"
        field_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(field_path, field=baseline_roi[base_index].numpy(), metadata_json=np.asarray(json.dumps({"phase_state_id": phase_state_id, "pattern": "uniform_t_0.5", "grid": BASE_GRID, "roi_size": ROI, "dx_m": DX, "dtype": "complex64"}, sort_keys=True)))
        phase_report[phase_state_id][f"{BASE_GRID}_t_0.5_reference_field_archive"] = str(field_path.relative_to(run_dir)).replace("\\", "/")
        _write_json(run_dir / "probe_progress.json", {"completed_phase_state_id": phase_state_id, "completed_pattern_count": len(pattern_names), "phase_optimization_updates": 0})
        del baseline_roi, middle_roi, direct_coefficients, direct_power, patterns_by_name, patterns, arrays
        gc.collect()
    return phase_report, {"cases": [case["case_name"] for case in cases], "fixed_probe_pattern_count": 34,
        "total_pattern_count_by_phase_state_id": phase_pattern_counts}


def _load_v7_response_pool(source_record: dict[str, Any], phase_state_id: str, case_name: str, mode: str, grid: int, coordinates: list[tuple[float, float]]) -> dict[str, torch.Tensor]:
    run_path = Path(source_record["run_path"])
    case = source_record["report"]["cases"][case_name]
    relative = case["modes"][mode]["matrix_archive_by_grid"][str(grid)]
    rel_path = Path(relative)
    if rel_path.is_absolute() or ".." in rel_path.parts:
        raise ValueError(f"v7 response archive path is unsafe: {relative}")
    path = run_path / rel_path
    if not path.is_file():
        raise FileNotFoundError(f"v7 response matrix is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        saved_coordinates = np.asarray(archive["coordinates_yx_m"], dtype=np.float64)
        k = np.asarray(archive["K_dimensionless"], dtype=np.complex128)
        h = np.asarray(archive["H_sqrt_w"], dtype=np.complex128)
    expected_coordinates = np.asarray(coordinates, dtype=np.float64)
    if (metadata.get("phase_state_id") != phase_state_id or metadata.get("mode") != mode
            or metadata.get("grid") != grid or metadata.get("row_pool_count") != len(coordinates)
            or metadata.get("pixel_count") != 10000 or metadata.get("K_unit") != "dimensionless"
            or metadata.get("H_unit") != "sqrt(W) per unit command"):
        raise ValueError(f"v7 response archive metadata does not match selected phase/mode/grid: {path}")
    if saved_coordinates.shape != expected_coordinates.shape or not np.array_equal(saved_coordinates, expected_coordinates):
        raise ValueError(f"v7 response archive coordinates do not match the selected physical fiber geometry: {path}")
    expected_shape = (len(coordinates), 10000)
    if k.shape != expected_shape or h.shape != expected_shape or not np.isfinite(k).all() or not np.isfinite(h).all():
        raise ValueError(f"v7 response archive matrix payload is invalid: {path}")
    return {"K": torch.from_numpy(k.copy()), "H": torch.from_numpy(h.copy()),
        "coordinates_yx_m": torch.from_numpy(saved_coordinates.copy())}


def _save_matrix_archive(run_dir: Path, phase_state_id: str, mode: str, grid: int, pool: dict[str, torch.Tensor], source: str) -> str:
    path = run_dir / "response_matrices" / f"{phase_state_id}_{mode}_{grid}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"K_dimensionless": pool["K"].cpu().numpy(), "H_sqrt_w": pool["H"].cpu().numpy(),
        "coordinates_yx_m": pool["coordinates_yx_m"].cpu().numpy()}
    if "J_phi" in pool:
        arrays["J_phi_sqrt_w_per_rad"] = pool["J_phi"].cpu().numpy()
    arrays["metadata_json"] = np.asarray(json.dumps({"phase_state_id": phase_state_id, "mode": mode, "grid": grid,
        "row_pool_count": len(pool["coordinates_yx_m"]), "pixel_count": 10000,
        "K_unit": "dimensionless", "H_unit": "sqrt(W) per unit command",
        "J_phi_unit": "sqrt(W) per rad", "source": source}, sort_keys=True))
    np.savez_compressed(path, **arrays)
    return str(path.relative_to(run_dir)).replace("\\", "/")


def _run_window_pilot(config: dict[str, Any], phases: dict[str, torch.Tensor], source_record: dict[str, Any], run_dir: Path, device: torch.device) -> dict[str, Any]:
    run_path = Path(source_record["run_path"])
    source_report = source_record["report"]
    roi_plane = v5._make_runtime(ROI, torch.float64, device).ols_plane
    result: dict[str, Any] = {}
    worst_saved_map: dict[str, Any] = {}
    gate = float(config["gates"]["window_aligned_complex_roi_error_max"])
    for case_name in TARGET_NAMES.values():
        phase_name = case_name
        phase_id = source_record["phase_state_ids"][phase_name]
        v7_phase = source_report["direct_probe_window_and_coefficient_checks"][phase_id]
        v7_errors = v7_phase["window_common_roi_errors_by_pattern"]
        worst_name = max((name for name in v7_errors if name.startswith("binary_probe_")),
            key=lambda name: v7_errors[name]["3072_vs_6144"]["common_global_phase_aligned_complex_relative_error"])
        archive_path = run_path / "direct_readouts" / f"{phase_id}_amplitude_probe_coefficients.npz"
        with np.load(archive_path, allow_pickle=False) as archive:
            source_names = [str(value) for value in archive["pattern_names"].tolist()]
            source_index = source_names.index(worst_name)
            worst_command = np.asarray(archive["amplitude_commands"][source_index], dtype=np.float32)
        pattern_values = {worst_name: worst_command, "uniform_t_0.5": np.full(10000, 0.5, dtype=np.float32)}
        case_result: dict[str, Any] = {"phase_state_id": phase_id, "patterns": {}, "pilot_complex64_pass": True}
        for pattern_name, pattern in pattern_values.items():
            pattern_result: dict[str, Any] = {"input_identity": {"source_v7_pattern": pattern_name, "v7_probe_index": source_index if pattern_name == worst_name else None,
                "command_min": float(pattern.min()), "command_max": float(pattern.max()), "per_sample_renormalized": False}, "by_precision": {}}
            for precision, real_dtype in (("complex64", torch.float32), ("complex128", torch.float64)):
                fields: dict[int, torch.Tensor] = {}
                coefficients: dict[int, dict[str, torch.Tensor]] = {}
                for grid in GRIDS:
                    runtime = v5._make_runtime(grid, real_dtype, device)
                    entrance = v5._encoder_entrance(runtime, real_dtype, device)
                    outputs, _traces = _runtime_forward_batch(runtime, entrance, phases[phase_name], torch.from_numpy(pattern[None]), grid, batch_size=1)
                    roi = _crop_roi(outputs[0], grid).detach().cpu()
                    fields[grid] = roi
                    coefficients[grid] = {mode: _project_roi_batch(roi[None].to(device), int(case_name.split("_")[1][0]), mode, roi_plane)[0].detach().cpu() for mode in MODES}
                    del runtime, entrance, outputs, _traces, roi
                    torch.cuda.empty_cache()
                    gc.collect()
                pair_results: dict[str, Any] = {}
                for left, right in ((4096, 6144), (4096, 8192), (6144, 8192)):
                    ref = fields[left].to(torch.complex128).reshape(-1)
                    got = fields[right].to(torch.complex128).reshape(-1)
                    inner = torch.sum(ref.conj() * got)
                    phase = torch.angle(inner)
                    aligned = got * torch.polar(torch.ones((), dtype=torch.float64), -phase)
                    raw_error = float(torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref).clamp_min(EPS))
                    aligned_error = float(torch.linalg.vector_norm(aligned - ref) / torch.linalg.vector_norm(ref).clamp_min(EPS))
                    ref_intensity = ref.abs().square()
                    got_intensity = got.abs().square()
                    intensity_error = float(torch.linalg.vector_norm(got_intensity - ref_intensity) / torch.linalg.vector_norm(ref_intensity).clamp_min(EPS))
                    power_left, power_right = float(ref_intensity.sum() * DX * DX), float(got_intensity.sum() * DX * DX)
                    pair = {"raw_complex_relative_error": raw_error, "aligned_complex_relative_error": aligned_error,
                        "common_global_phase_rad": float(phase), "intensity_relative_l2_error": intensity_error,
                        "roi_power_left_w": power_left, "roi_power_right_w": power_right,
                        "roi_power_relative_error": abs(power_right - power_left) / max(power_left, EPS), "fiber_modes": {}}
                    for mode in MODES:
                        cref, cgot = coefficients[left][mode], coefficients[right][mode]
                        raw_c = float(torch.linalg.vector_norm(cgot - cref) / torch.linalg.vector_norm(cref).clamp_min(EPS))
                        cphase = torch.angle(torch.sum(cref.conj() * cgot))
                        caligned = cgot * torch.polar(torch.ones((), dtype=torch.float64), -cphase)
                        aligned_c = float(torch.linalg.vector_norm(caligned - cref) / torch.linalg.vector_norm(cref).clamp_min(EPS))
                        pref, pgot = cref.abs().square(), cgot.abs().square()
                        pair["fiber_modes"][mode] = {"complex_latent_raw_relative_l2": raw_c,
                            "complex_latent_common_global_phase_aligned_relative_l2": aligned_c,
                            "latent_global_phase_rad": float(cphase),
                            "total_coupled_power_relative_error": abs(float(pgot.sum() - pref.sum())) / max(float(pref.sum()), EPS),
                            "per_fiber_power_relative_l2": float(torch.linalg.vector_norm(pgot - pref) / torch.linalg.vector_norm(pref).clamp_min(EPS)),
                            "coefficients_left_sqrt_w": [[float(value.real), float(value.imag)] for value in cref],
                            "coefficients_right_sqrt_w": [[float(value.real), float(value.imag)] for value in cgot]}
                    pair["passed"] = aligned_error <= gate
                    if precision == "complex64":
                        case_result["pilot_complex64_pass"] = case_result["pilot_complex64_pass"] and pair["passed"]
                        if aligned_error > float(worst_saved_map.get("error", -1.0)):
                            test_aligned = fields[right].to(torch.complex128) * torch.polar(torch.ones((), dtype=torch.float64), -phase)
                            diff = test_aligned - fields[left].to(torch.complex128)
                            worst_saved_map = {"error": aligned_error, "case": case_name, "pattern": pattern_name,
                                "left": left, "right": right, "field_difference": diff.cpu().numpy().astype(np.complex64),
                                "intensity_difference": (got_intensity - ref_intensity).reshape(ROI, ROI).cpu().numpy().astype(np.float32),
                                "phase": float(phase)}
                    pair_results[f"{left}_vs_{right}"] = pair
                pattern_result["by_precision"][precision] = {"window_comparisons": pair_results,
                    "all_complex64_comparisons_pass": all(item["passed"] for item in pair_results.values()) if precision == "complex64" else None}
                del fields, coefficients
                gc.collect()
            case_result["patterns"][pattern_name] = pattern_result
        result[case_name] = case_result
    if worst_saved_map:
        path = run_dir / "pilot_error_fields" / "worst_complex64_difference.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, field_difference=worst_saved_map["field_difference"], intensity_difference=worst_saved_map["intensity_difference"],
            metadata_json=np.asarray(json.dumps({key: value for key, value in worst_saved_map.items() if key not in ("field_difference", "intensity_difference")}, sort_keys=True)))
        result["worst_complex64_error_field"] = str(path.relative_to(run_dir)).replace("\\", "/")
    result["all_candidate_complex64_pilots_pass"] = all(value["pilot_complex64_pass"] for value in result.values() if isinstance(value, dict) and "pilot_complex64_pass" in value)
    result["gate"] = gate
    return result


def _run_full(config: dict[str, Any], config_path: Path, run_dir: Path, root: Path) -> dict[str, Any]:
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("v8 full measurement requires this attempt's passed preflight")
    device = torch.device("cuda:0")
    phases, source_record = _load_v7_phases(config, root)
    cases = _build_cases(config, phases)
    state_ids = _phase_state_ids(phases, cases, source_record["phase_state_ids"])
    if state_ids != {name: source_record["phase_state_ids"][name] for name in state_ids}:
        raise ValueError("loaded v7 phase-state identities changed before v8 measurement")
    saved_phases = {name: value.detach().cpu().to(torch.float32) for name, value in phases.items()}
    torch.save({"schema": "ols_frozen_phase_screens.v8", "phases": saved_phases, "phase_state_ids": state_ids}, run_dir / "phase_screens_v8.pt")
    _write_json(run_dir / "configuration.json", {"config_path": str(config_path), "configuration": config,
        "source_evidence": {key: value for key, value in source_record.items() if key != "report"}, "phase_state_ids": state_ids})

    pilot = _run_window_pilot(config, phases, source_record, run_dir, device)
    _write_json(run_dir / "window_pilot.json", pilot)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not pilot["all_candidate_complex64_pilots_pass"]:
        report = {"schema": SCHEMA, "status": "VALID_FAILED", "decision_question": "Do frozen 2x2-C and 3x3-B candidates pass expanded-window numerical qualification?",
            "result": "WINDOW_GATE_NOT_MET_AT_PILOT", "preflight": preflight, "window_pilot": pilot,
            "historical_v4_g3_status": "2048_WINDOW_G3_FAIL_PRESERVED", "source_v7_run": source_record["run_path"],
            "fixed_physical_system": {"wavelength_m": WAVELENGTH, "dx_m": DX, "phase_screens": [100, 100], "aperture_m": 200e-6,
                "spacing_each_m": 1.5e-3, "layer_power_transmission": TRANSMISSION, "source_power_w": SOURCE_POWER,
                "pitch_m": PITCH, "mode_field_diameter_m": 14.222e-6, "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
                "propagation_grids": list(GRIDS), "common_roi_pixels": ROI},
            "updates_and_forbidden_io": {"formal_training_updates": 0, "phase_optimization_updates": 0, "dataset_reads": 0,
                "task_evaluator_calls": 0, "fid_calls": 0, "loss_changes": 0, "material_forward_calls": 0, "decoder_camera_changes": 0},
            "scientific_decision": {"status": "INSUFFICIENT_EVIDENCE", "permitted_next_action": "report the failed expanded-window qualification; do not expand windows or optimize phases"},
            "execution": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.get_device_name(device), "elapsed_seconds": time.time() - _START_TIME}}
        _write_json(run_dir / "report.json", report)
        (run_dir / "exit_status.txt").write_text("0\n", encoding="ascii")
        return report

    roi_runtime = v5._make_runtime(ROI, torch.float64, device)
    roi_plane = roi_runtime.ols_plane
    state_pool: dict[str, Any] = {}
    cases_report: dict[str, Any] = {}
    control_records: dict[str, Any] = {}
    direct_patterns_by_case_mode: dict[str, dict[str, list[tuple[str, torch.Tensor]]]] = {}
    bounded_command_arrays: dict[str, np.ndarray] = {}
    source_archives: dict[str, Any] = {}

    for case in cases:
        case_name, size, phase_name = case["case_name"], int(case["size"]), case["phase_name"]
        state_id = state_ids[phase_name]
        coordinates = _coordinate_pool([size], device)
        state_pool[state_id] = {"pool_coordinates": coordinates, "phase_tensor": phases[phase_name], "phase_name": phase_name, "modes": {}}
        phase = phases[phase_name].to(device=device, dtype=torch.float64)
        primary_runtime = v5._make_runtime(BASE_GRID, torch.float64, device)
        entrance = v5._encoder_entrance(primary_runtime, torch.float64, device)
        with torch.no_grad():
            primary_runtime.encoder.phase.copy_(phase)
            first_fields, first_records, first_stats = primary_runtime.encoder(
                ComplexField(entrance.tensor * 0.5, entrance.plane, OperationProvenance("v8_jacobian_reference", entrance.plane.plane_id, None)),
                f"ols_v8_jacobian_{state_id}")
        first_layer_output = first_records[0].tensor[0, 0].to(torch.complex128)
        del first_fields, first_records, first_stats
        for mode in MODES:
            mode_entry: dict[int, dict[str, torch.Tensor]] = {}
            for grid in GRIDS:
                if grid in (BASE_GRID, 6144):
                    loaded = _load_v7_response_pool(source_record, state_id, case_name, mode, grid, coordinates)
                else:
                    loaded = {}
                rebuild_error: dict[str, float] | None = None
                if grid == 6144:
                    computed = loaded
                    source = "reused_v7_archive"
                else:
                    grid_runtime = primary_runtime if grid == BASE_GRID else v5._make_runtime(grid, torch.float64, device)
                    computed = _build_response_pool(run_dir=run_dir, runtime=grid_runtime, roi_plane=roi_plane,
                        entrance=entrance, phases=phase, phase_state_id=state_id, mode_name=mode, grid=grid,
                        coordinates=coordinates, first_layer_output=first_layer_output if grid == BASE_GRID else None,
                        include_phase_jacobian=grid == BASE_GRID)
                    if grid == BASE_GRID:
                        rebuild_error = {"K": _relative_vector_error(computed["K"], loaded["K"]),
                            "H": _relative_vector_error(computed["H"], loaded["H"])}
                        if max(rebuild_error.values()) > float(config["gates"]["coefficient_relative_error_max"]):
                            raise ValueError(f"v7 primary-grid response differs from its independently rebuilt production adjoint for {case_name}/{mode}")
                        computed["K"], computed["H"] = loaded["K"], loaded["H"]
                        source = "v7_archive_with_v8_rebuilt_J_phi"
                    else:
                        source = "computed_v8"
                    if grid != BASE_GRID:
                        del grid_runtime
                mode_entry[grid] = computed
                source_archives[f"{case_name}/{mode}/{grid}"] = {"source": source,
                    "K_H_rebuild_relative_error": rebuild_error,
                    "archive": _save_matrix_archive(run_dir, state_id, mode, grid, computed, source)}
                if grid == 8192:
                    del computed
                    torch.cuda.empty_cache()
                    gc.collect()
            state_pool[state_id]["modes"][mode] = mode_entry
        del primary_runtime, entrance, first_layer_output
        torch.cuda.empty_cache()
        gc.collect()

    thresholds = [float(value) for value in config["measurement"]["relative_svd_thresholds"]]
    for case in cases:
        case_name, size, phase_name = case["case_name"], int(case["size"]), case["phase_name"]
        state_id = state_ids[phase_name]
        indices = _pool_indices(state_pool[state_id]["pool_coordinates"], size)
        cases_report[case_name] = {**case, "phase_state_id": state_id, "channel_count": size * size,
            "fiber_coordinates_xy_m_row_major": [[x, y] for y, x in _coords(size, torch.device("cpu"))], "modes": {}}
        control_records[case_name] = {}
        direct_patterns_by_case_mode[case_name] = {}
        for mode in MODES:
            mode_entry = state_pool[state_id]["modes"][mode]
            selected = {grid: {key: value.index_select(0, torch.tensor(indices, dtype=torch.long)) for key, value in mode_entry[grid].items()
                if key in ("K", "H", "J_phi")} for grid in GRIDS}
            _validate_matrix_jacobian_grid(BASE_GRID, BASE_GRID, BASE_GRID)
            base_k, base_h = selected[BASE_GRID]["K"].to(device), selected[BASE_GRID]["H"].to(device)
            spectra = _response_spectra(base_k, base_h, thresholds)
            base_real = torch.cat((base_h.real, base_h.imag), dim=0)
            base_real_sv, base_complex_sv = torch.linalg.svdvals(base_real), torch.linalg.svdvals(base_k)
            window_metrics: dict[str, Any] = {}
            for grid in GRIDS:
                if grid == BASE_GRID:
                    continue
                k_delta, h_delta = base_k - selected[grid]["K"].to(device), base_h - selected[grid]["H"].to(device)
                k_delta_norm = float(torch.linalg.svdvals(k_delta).max())
                h_delta_norm = float(torch.linalg.svdvals(torch.cat((h_delta.real, h_delta.imag), dim=0)).max())
                grid_sv = torch.linalg.svdvals(torch.cat((selected[grid]["H"].real, selected[grid]["H"].imag), dim=0))
                weak = {str(threshold): int(((base_real_sv - h_delta_norm) > threshold * (base_real_sv.max() + h_delta_norm)).sum()) for threshold in thresholds}
                window_metrics[str(grid)] = {"K_operator_difference_spectral_norm": k_delta_norm,
                    "K_operator_difference_relative_to_4096_max_sigma": k_delta_norm / max(float(base_complex_sv.max()), EPS),
                    "H_real_control_operator_difference_spectral_norm_sqrt_w": h_delta_norm,
                    "H_real_control_difference_relative_to_4096_max_sigma": h_delta_norm / max(float(base_real_sv.max()), EPS),
                    "grid_real_control_max_singular_value_sqrt_w": float(grid_sv.max()),
                    "weak_singular_directions_above_window_uncertainty_by_relative_threshold": weak,
                    "interpretation": "Weyl bound from measured cross-window operator difference; not detector-noise observability"}
            j = selected[BASE_GRID]["J_phi"].to(device)
            c0 = _response_at_command(base_h, torch.full((10000,), 0.5, dtype=torch.float64))
            gt = 2.0 * torch.real(c0.conj()[:, None] * base_h)
            gphi = 2.0 * torch.real(c0.conj()[:, None] * j)
            mode_report = {"matrix_archive_by_grid": {str(grid): source_archives[f"{case_name}/{mode}/{grid}"]["archive"] for grid in GRIDS},
                "spectra_at_4096": spectra, "matrix_window_uncertainty": window_metrics,
                "complex_K_shape": list(base_k.shape), "complex_H_shape": list(base_h.shape),
                "H_units": "sqrt(W) per unit real amplitude command",
                "power_jacobian_Gt": {"shape": list(gt.shape), "units": "W per unit amplitude command",
                    "spectrum": _spectrum(gt, thresholds, singular_value_units="W per unit command", squared_gain_units="W squared per squared command norm")},
                "complex_coefficient_phase_jacobian_Jphi": {"shape": list(j.shape), "units": "sqrt(W) per radian",
                    "spectrum": _phase_jacobian_shape_report(j, c0, thresholds),
                    "power_jacobian_Gphi": {"shape": list(gphi.shape), "units": "W per radian",
                        "spectrum": _spectrum(gphi, thresholds, singular_value_units="W per radian-norm", squared_gain_units="W squared per squared radian-norm")}}}
            target_records, direct_patterns, all_commands = _control_targets_all_channels(base_h, c0, size, config)
            direct_patterns_by_case_mode[case_name][mode] = direct_patterns
            for command_id, command in all_commands:
                bounded_command_arrays[f"{case_name}__{mode}__{command_id}"] = command.detach().cpu().numpy().astype(np.float64, copy=False)
            control_records[case_name][mode] = {"all_channel_bounded_single_output_controls": target_records,
                "channels_analyzed": list(range(size * size)), "component_targets": ["real", "imag"],
                "direct_validation_pattern_names": [label for label, _ in direct_patterns],
                "bounded_command_count": len(all_commands), "command_vectors_saved_in": "bounded_control_commands_v8.npz"}
            cases_report[case_name]["modes"][mode] = mode_report
            del selected, base_k, base_h, base_real, base_real_sv, base_complex_sv, spectra, j, c0, gt, gphi
        gc.collect()
    bounded_path = run_dir / "bounded_control_commands_v8.npz"
    np.savez_compressed(bounded_path, command_names=np.asarray(list(bounded_command_arrays), dtype="U160"),
        delta_commands=np.stack(list(bounded_command_arrays.values())).astype(np.float64))

    jacobian_report: dict[str, Any] = {}
    for phase_name in dict.fromkeys(case["phase_name"] for case in cases):
        state_id = state_ids[phase_name]
        runtime = v5._make_runtime(BASE_GRID, torch.float64, device)
        entrance = v5._encoder_entrance(runtime, torch.float64, device)
        jacobian_report[state_id] = _phase_jacobian_preflight(runtime=runtime, entrance=entrance, phase=phases[phase_name],
            phase_state_id=state_id, cases=cases, state_pool=state_pool, roi_plane=roi_plane, device=device, config=config)
        del runtime, entrance
        torch.cuda.empty_cache()
        gc.collect()

    phase_probe_report, probe_design = _window_probe_and_direct_measurement(config=config, cases=cases, phases=phases,
        state_ids=state_ids, state_pool=state_pool, run_dir=run_dir, device=device,
        direct_patterns_by_case_mode=direct_patterns_by_case_mode)
    report = {"schema": SCHEMA, "status": "VALID_COMPLETED", "decision_question": "Do expanded numerical windows support reliable amplitude-control and absolute-power evidence for the frozen four- and nine-channel candidates?",
        "historical_v4_g3_status": "2048_WINDOW_G3_FAIL_PRESERVED", "v7_source": {key: value for key, value in source_record.items() if key != "report"},
        "v7_worst_probe_replay": preflight["checks"].get("v7_worst_probe_replay_pass"), "window_pilot": pilot,
        "fixed_physical_system": {"wavelength_m": WAVELENGTH, "dx_m": DX, "phase_screens": [100, 100], "aperture_m": 200e-6,
            "spacing_each_m": 1.5e-3, "layer_power_transmission": TRANSMISSION, "source_power_w": SOURCE_POWER,
            "pitch_m": PITCH, "mode_field_diameter_m": 14.222e-6, "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
            "propagation_grids": list(GRIDS), "primary_response_grid": BASE_GRID, "common_roi_pixels": ROI},
        "preflight": preflight, "frozen_phase_states": {name: {"phase_state_id": state_ids[name], "source": source_record["selected_endpoints"][name]} for name in phases},
        "amplitude_probe_design": probe_design, "cases": cases_report, "bounded_controls": control_records,
        "direct_probe_window_and_coefficient_checks": phase_probe_report, "phase_jacobian_validation": jacobian_report,
        "response_archive_provenance": source_archives,
        "matrix_payload": "response_matrices/*.npz; 4096/6144 K/H reused from v7, 4096 J_phi recomputed, 8192 K/H newly computed",
        "phase_payload": "phase_screens_v8.pt", "bounded_command_payload": "bounded_control_commands_v8.npz",
        "measurement_contract": {"response_definitions": {"K": "c=Ka; K dimensionless", "H": "c=Ht; H=K diag(E_source sqrt(pixel_area)); H in sqrt(W) per unit command",
            "M_H": "[Re H; Im H] real command response", "G_t": "2 Re(diag(c*) H)", "G_phi": "2 Re(diag(c*) J_phi)"},
            "primary_grid": BASE_GRID, "phase_jacobian_grid": BASE_GRID, "phase_svd_global_phase_projection": "raw and common-phase-projected shape spectra are kept separately",
            "matrix_prediction_gate": float(config["gates"]["coefficient_relative_error_max"]), "full_roi_window_gate": float(config["gates"]["window_aligned_complex_roi_error_max"]),
            "probe_commands_not_renormalized": True, "probe_input_power_fixed_before_amplitude_command_w": SOURCE_POWER,
            "all_real_and_imaginary_channel_targets_analyzed": True, "no_detector_noise_or_semantic_interface_model": True},
        "updates_and_forbidden_io": {"formal_training_updates": 0, "phase_optimization_updates": 0, "dataset_reads": 0, "task_evaluator_calls": 0,
            "fid_calls": 0, "loss_changes": 0, "material_forward_calls": 0, "decoder_camera_changes": 0},
        "execution": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.get_device_name(device), "elapsed_seconds": time.time() - _START_TIME},
        "claim_limits": ["pixel-level amplitude controllability is an upper bound for a future input interface, not semantic or measured camera observability",
            "mathematical singular rank is not detector-noise-limited experimental dimension", "Gaussian is the primary numerical reference and LP01 is a same-field sensitivity check",
            "all fiber/mode/optics parameters remain unmeasured simulation assumptions", "finite phase candidates do not establish global optimization or a physical capacity bound"],
        "runtime_module_sources": {"v8_cli": str(Path(__file__).resolve()), "v7_baseline": str(Path(baseline.__file__).resolve()),
            "v5_runtime": str(Path(v5.__file__).resolve())},
        "scientific_decision": _scientific_decision(cases_report, phase_probe_report, jacobian_report)}
    _write_json(run_dir / "report.json", report)
    (run_dir / "exit_status.txt").write_text("0\n", encoding="ascii")
    del state_pool, phases, cases, roi_runtime
    torch.cuda.empty_cache()
    gc.collect()
    return report

def _scientific_decision(cases: dict[str, Any], probes: dict[str, Any], jacobians: dict[str, Any]) -> dict[str, Any]:
    qualified: dict[str, bool] = {}
    for name in TARGET_NAMES.values():
        matching = [candidate for phase in probes.values() for case_name, candidate in phase.get("per_candidate_matrix_to_direct_checks", {}).items() if case_name == name]
        mode_checks = [mode for item in matching for mode in item.get("by_mode", {}).values()]
        qualified[name] = len(matching) == 1 and len(mode_checks) == len(MODES) and all(
            mode.get("all_32_binary_probe_checks_pass", False)
            and mode.get("all_selected_direct_control_checks_pass", False)
            and mode.get("uniform_t_0.5_matrix_and_window_pass", False)
            and all(mode.get("fixed_control_checks", {}).values())
            for mode in mode_checks)
    phase_checks = [value.get("all_pixel_checks_pass", False) for value in jacobians.values()]
    both_candidates = all(qualified.values()) and len(qualified) == 2
    jacobians_pass = bool(phase_checks) and all(phase_checks)
    ready = both_candidates and jacobians_pass
    return {"status": "CONTROL_EVIDENCE_AVAILABLE" if ready else "INSUFFICIENT_EVIDENCE",
        "selected_candidate_probe_gates": qualified,
        "four_channel_candidate_qualified": qualified.get("candidate_2x2_C", False),
        "nine_channel_candidate_qualified": qualified.get("candidate_3x3_B", False),
        "four_and_nine_channel_candidates_qualified": both_candidates,
        "phase_pixel_jacobian_checks_pass": jacobians_pass,
        "interpretation": "Supports planning a frozen four-channel versus nine-channel Decoder/camera interface comparison only when both candidate probe gates and phase Jacobian checks pass. It does not select an OLS dimension or establish camera observability.",
        "permitted_next_action": "plan a frozen four-channel versus nine-channel Decoder/camera coordinate and unit interface comparison" if ready else "retain only candidates whose complete numerical and control evidence passes; do not infer a physical capacity bound"}


_START_TIME = time.time()


def run(config_path: Path, run_dir: Path, *, preflight_only: bool) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = Path.cwd().resolve()
    config = _load_config(config_path)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    source_dir = run_dir / "source_snapshot"
    invocation_dir = source_dir / f"invocation_{time.time_ns()}"
    invocation_dir.mkdir(parents=True, exist_ok=False)
    config_copy = invocation_dir / "configuration.yaml"
    snapshot = invocation_dir / "diagnose_ols_four_nine_window_closure_v8.py"
    config_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    snapshot.write_bytes(Path(__file__).read_bytes())
    invocation_record = {"explicit_config_path": str(config_path),
        "configuration_snapshot": str(config_copy.relative_to(run_dir)).replace("\\", "/"),
        "source_snapshot": str(snapshot.relative_to(run_dir)).replace("\\", "/")}
    if preflight_only:
        result = _preflight(config, config_path, run_dir, root)
        result["run_invocation"] = invocation_record
        _write_json(run_dir / "preflight.json", result)
        return result
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("v8 complete measurement requires this attempt's passed preflight")
    result = _run_full(config, config_path, run_dir, root)
    result["run_invocation"] = invocation_record
    _write_json(run_dir / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.run_dir, preflight_only=args.preflight_only)
    print(json.dumps({"status": result.get("status"), "passed": result.get("passed"), "scientific_decision": result.get("scientific_decision", {}).get("status"), "run_dir": str(args.run_dir)}, ensure_ascii=False))
    if args.preflight_only and result.get("passed") is not True:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
