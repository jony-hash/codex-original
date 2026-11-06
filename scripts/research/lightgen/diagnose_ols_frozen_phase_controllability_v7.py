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


SCHEMA = "ols_frozen_phase_controllability.v7"
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_frozen_phase_controllability_v7.yaml")
DEFAULT_RUN_DIR = Path("outputs/20261014_ols_frozen_phase_controllability_v7/runs/attempt_01")
GRIDS = (3072, 4096, 6144)
ROI = 2048
DX = 2.0e-6
WAVELENGTH = 532.0e-9
SOURCE_POWER = 249.3e-6
PITCH = 125.0e-6
WAIST = 7.111e-6
TRANSMISSION = 0.889
SIZES = (2, 3, 5, 10)
MODES = ("gaussian_s12", "lp01")
TARGET_NAMES = {
    "2:A_max_throughput": "candidate_2x2_A",
    "2:C_absolute_uniform_delivery": "candidate_2x2_C",
    "3:B_max_neff_fraction": "candidate_3x3_B",
    "5:B_max_neff_fraction": "candidate_5x5_B",
    "10:A_max_throughput": "candidate_10x10_A",
    "10:C_absolute_uniform_delivery": "candidate_10x10_C",
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
        "common_fiber_readout_roi_mm": 4.096, "common_fiber_readout_roi_pixels": ROI,
        "inter_encoder_spacing_m": 1.5e-3, "last_encoder_to_fiber_m": 1.5e-3,
        "layer_power_transmission": TRANSMISSION, "input_power_w": SOURCE_POWER,
        "illumination": "uniform_coherent_top_hat",
    }
    if config.get("optics") != expected_optics:
        raise ValueError("v7 optical system or numerical windows changed")
    expected_fiber = {
        "reference_pitch_m": PITCH, "mode_field_diameter_m": 14.222e-6,
        "gaussian_1e_amplitude_radius_m": WAIST, "lp01_v_number": 2.2,
        "transport": "ideal_identity", "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
    }
    if config.get("fiber") != expected_fiber:
        raise ValueError("v7 fiber model assumptions changed")
    candidates = config.get("candidates", {})
    if candidates.get("zero_phase_controls") != list(SIZES):
        raise ValueError("v7 zero-phase controls must remain 2x2, 3x3, 5x5, and 10x10")
    optimized = candidates.get("optimized", [])
    observed = {item.get("endpoint_key") for item in optimized}
    if observed != set(TARGET_NAMES):
        raise ValueError("v7 optimized candidate set changed")
    measurement = config.get("measurement", {})
    if measurement.get("amplitude_probe_seed") != 2026101407 or measurement.get("amplitude_probe_count") != 32:
        raise ValueError("v7 amplitude probe population or seed changed")
    if measurement.get("relative_svd_thresholds") != [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4]:
        raise ValueError("v7 singular-value thresholds changed")
    if config.get("gates", {}).get("coefficient_relative_error_max") != 1.0e-4:
        raise ValueError("v7 coefficient gate must remain 1e-4")
    if config.get("gates", {}).get("window_aligned_complex_roi_error_max") != 1.0e-4:
        raise ValueError("v7 common-ROI window gate must remain 1e-4")
    expected_scope = {
        "formal_training_updates": 0, "phase_optimization_updates": 0, "dataset_reads": 0,
        "task_evaluator_calls": 0, "fid_calls": 0, "loss_changes": 0,
        "material_forward_calls": 0, "decoder_or_camera_changes": 0,
        "production_default_migration": "forbidden", "fiber_geometry_changes": "forbidden",
    }
    if config.get("scope") != expected_scope:
        raise ValueError("v7 scope contract changed")
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


def _v6_endpoint_window_qualified(endpoint: dict[str, Any]) -> bool:
    if endpoint.get("status") != "WINDOW_QUALIFIED_REPRESENTATIVE":
        return False
    metrics = endpoint.get("metrics")
    if not isinstance(metrics, dict) or metrics.get("status") != "MEASURED":
        return False
    window_comparisons = metrics.get("window_comparisons", {})
    complex64 = window_comparisons.get("complex64", {})
    expected_pairs = ("3072_vs_4096", "3072_vs_6144", "4096_vs_6144")
    if window_comparisons.get("production_threshold") != 1.0e-4 or window_comparisons.get("production_complex64_all_three_pairs_pass") is not True:
        return False
    return all(
        isinstance(complex64.get(pair), dict)
        and isinstance(complex64[pair].get("complex_relative_error_after_common_global_phase"), (int, float))
        and float(complex64[pair]["complex_relative_error_after_common_global_phase"]) <= 1.0e-4
        for pair in expected_pairs
    )


def _load_v6_phases(config: dict[str, Any], root: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    report_path = Path(config["source_evidence"]["v6_report"])
    run_path = Path(config["source_evidence"]["v6_run_dir"])
    report_path = report_path if report_path.is_absolute() else root / report_path
    run_path = run_path if run_path.is_absolute() else root / run_path
    if not report_path.is_file() or not run_path.is_dir():
        raise FileNotFoundError("explicit v6 report or source run directory is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "ols_phase_dimension_power.v6" or report.get("status") != "VALID_COMPLETED":
        raise ValueError("v7 requires the named VALID_COMPLETED v6 report")
    phases: dict[str, torch.Tensor] = {}
    source_records: dict[str, Any] = {}
    for endpoint_key, phase_name in TARGET_NAMES.items():
        endpoint = report.get("representative_endpoints", {}).get(endpoint_key)
        if not isinstance(endpoint, dict) or endpoint.get("status") != "WINDOW_QUALIFIED_REPRESENTATIVE":
            raise ValueError(f"v6 representative {endpoint_key} is missing or not window-qualified")
        phase_rel = endpoint.get("phase_path")
        if not isinstance(phase_rel, str) or Path(phase_rel).is_absolute() or ".." in Path(phase_rel).parts:
            raise ValueError(f"v6 representative {endpoint_key} has an unsafe or missing phase path")
        phase_path = run_path / phase_rel
        if not phase_path.is_file():
            raise FileNotFoundError(f"explicit v6 representative phase is missing: {phase_path}")
        phase = torch.load(phase_path, map_location="cpu", weights_only=True)
        if not isinstance(phase, torch.Tensor) or tuple(phase.shape) != (2, 1, 100, 100) or not torch.isfinite(phase).all():
            raise ValueError(f"invalid frozen phase tensor for {endpoint_key}")
        if phase.dtype != torch.float32:
            phase = phase.to(torch.float32)
        if not _v6_endpoint_window_qualified(endpoint):
            raise ValueError(f"v6 representative {endpoint_key} did not pass the production window gate")
        phases[phase_name] = phase.clone()
        source_records[phase_name] = {
            "endpoint_key": endpoint_key, "phase_path": str(phase_path),
            "phase_dtype_loaded": str(phase.dtype), "endpoint_status": endpoint.get("status"),
            "window_qualification_evidence": endpoint.get("metrics", {}).get("window_comparisons", {}).get("complex64"),
            "phase_tensor_shape": list(phase.shape),
        }
    phases["zero_phase"] = torch.zeros((2, 1, 100, 100), dtype=torch.float32)
    source_records["zero_phase"] = {"source": "exact all-zero two-screen diagnostic control", "phase_tensor_shape": [2, 1, 100, 100]}
    return phases, {"report_path": str(report_path), "status": report["status"], "selected_endpoints": source_records}


def _phase_state_ids(phases: dict[str, torch.Tensor], cases: list[dict[str, Any]]) -> dict[str, str]:
    names = list(dict.fromkeys(case["phase_name"] for case in cases))
    return {name: f"phase_{index:02d}_{name}" for index, name in enumerate(names)}


def _phase_pool_sizes(cases: list[dict[str, Any]], phase_name: str) -> list[int]:
    if phase_name == "zero_phase":
        return list(SIZES)
    return sorted({int(case["size"]) for case in cases if case["phase_name"] == phase_name})


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


def _relative_vector_error(expected: torch.Tensor, observed: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(expected - observed) / torch.linalg.vector_norm(observed).clamp_min(EPS))


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
            batch_field = ComplexField(tensor, entrance.plane, OperationProvenance("v7_frozen_amplitude_command", entrance.plane.plane_id, None))
            output, _records, statistics = runtime.encoder(batch_field, f"ols_v7_{grid}")
            outputs.extend([value.detach() for value in output.tensor[:, 0]])
            traces.extend([value.detach() for value in statistics["layer_power"]])
            del command, tensor, batch_field, output, _records, statistics
    return outputs, traces


def _readout_batch(field_list: list[torch.Tensor], sizes: list[int], mode_name: str, roi_plane, device: torch.device) -> dict[int, torch.Tensor]:
    if not field_list:
        return {}
    batch = torch.stack([_crop_roi(field, field.shape[-1]) for field in field_list]).to(device=device)
    result = {size: _project_roi_batch(batch, size, mode_name, roi_plane).detach().cpu() for size in sizes}
    del batch
    return result


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
    coordinates = state_pool[phase_state_id]["modes"][MODES[0]][3072]["coordinates_yx_m"].tolist()
    indices = _pool_indices([tuple(v) for v in coordinates], size)
    for mode in MODES:
        mode_grid = state_pool[phase_state_id]["modes"][mode][3072]
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
        fields, _trace = _runtime_forward_batch(runtime, entrance64, changed, patterns, 3072, batch_size=1)
        roi_field = _crop_roi(fields[0], 3072)
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


def _preflight(config: dict[str, Any], config_path: Path, run_dir: Path, root: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("v7 production preflight requires AutoDL CUDA")
    device = torch.device("cuda:0")
    phases, source_record = _load_v6_phases(config, root)
    runtime = v5._make_runtime(3072, torch.float64, device)
    entrance = v5._encoder_entrance(runtime, torch.float64, device)
    roi_runtime = v5._make_runtime(ROI, torch.float64, device)
    roi_plane = roi_runtime.ols_plane
    phase = phases["zero_phase"].to(device=device, dtype=torch.float64)
    probes, _ = _amp_probes(config)
    checks: dict[str, Any] = {}
    random = torch.randn((100, 100), dtype=torch.float64, device=device) + 1j * torch.randn((100, 100), dtype=torch.float64, device=device)
    random_mode = torch.randn((3072, 3072), dtype=torch.float64, device=device) + 1j * torch.randn((3072, 3072), dtype=torch.float64, device=device)
    plane = runtime.ols_plane
    y0, y1, x0, x1 = _active_bounds(plane)
    padded = torch.zeros((3072, 3072), dtype=torch.complex128, device=device)
    padded[y0:y1, x0:x1] = random
    prop = baseline._adjoint_transfer(random_mode, plane, 1.5e-3)
    transfer = baseline._transfer_function(
        plane.shape,
        pixel_pitch_m=plane.pixel_pitch_m,
        wavelength_m=plane.wavelength_m,
        distance_m=1.5e-3,
        device=device,
        real_dtype=torch.float64,
    )
    lhs = torch.vdot(
        torch.fft.ifft2(torch.fft.fft2(padded, norm="ortho") * transfer, norm="ortho").reshape(-1),
        random_mode.reshape(-1),
    )
    rhs = torch.vdot(padded.reshape(-1), prop.reshape(-1))
    adjoint_error = float((lhs - rhs).abs() / lhs.abs().clamp_min(EPS))
    checks["propagation_adjoint_inner_product_relative_error"] = adjoint_error
    checks["propagation_adjoint_inner_product_pass"] = adjoint_error <= 1.0e-10
    # Production mode normalization and projection passivity on a finite random ROI field.
    roi_field = torch.complex(torch.randn((ROI, ROI), device=device), torch.randn((ROI, ROI), device=device)).to(torch.complex128)
    before = float(roi_field.abs().square().sum() * DX * DX)
    mode_passivity: dict[str, Any] = {}
    for mode in MODES:
        coeff = _project_roi_batch(roi_field[None], 10, mode, roi_plane)[0]
        captured = float(coeff.abs().square().sum())
        mode_passivity[mode] = {"input_power_arbitrary_units": before, "projected_power_arbitrary_units": captured, "ratio": captured / before}
    checks["mode_projection_passivity"] = mode_passivity
    checks["mode_projection_passivity_pass"] = all(value["ratio"] <= 1.0 + 1.0e-5 for value in mode_passivity.values())
    # Pixel-area omission and complex-phase loss are deliberate measurement counterexamples.
    direct = _project_roi_batch(roi_field[None], 3, "gaussian_s12", roi_plane)[0]
    omitted_area = direct / (DX * DX)
    checks["omitted_pixel_area_negative_control_relative_error"] = _relative_vector_error(omitted_area, direct)
    checks["omitted_pixel_area_rejected"] = checks["omitted_pixel_area_negative_control_relative_error"] > 1.0e-4
    complex_input = torch.tensor([[1.0 + 2.0j, -0.5 + 0.75j]], dtype=torch.complex128)
    phase_preserved = complex_input * torch.exp(torch.tensor(1j * 0.37, dtype=torch.complex128))
    phase_dropped = phase_preserved.real.to(torch.complex128)
    checks["dropped_complex_phase_negative_control_relative_error"] = _relative_vector_error(phase_dropped, phase_preserved)
    checks["dropped_complex_phase_rejected"] = checks["dropped_complex_phase_negative_control_relative_error"] > 1.0e-4
    # The direct production forward confirms the nonzero zero-phase response that rejects the old zero-response adjoint.
    patterns = torch.full((1, 10000), 0.5, dtype=torch.float64)
    outputs, _traces = _runtime_forward_batch(runtime, entrance, phase, patterns, 3072, batch_size=1)
    roi_out = _crop_roi(outputs[0], 3072)
    direct_coeff = _project_roi_batch(roi_out[None], 3, "gaussian_s12", roi_plane)[0]
    checks["old_adjoint_zero_response_negative_control"] = {
        "production_zero_phase_coefficient_norm_sqrt_w": float(torch.linalg.vector_norm(direct_coeff)),
        "legacy_zero_response_sqrt_w": 0.0,
        "rejected": bool(torch.linalg.vector_norm(direct_coeff) > 0.0),
    }
    checks["production_forward_rejects_legacy_zero_response"] = checks["old_adjoint_zero_response_negative_control"]["rejected"]
    # Each screen-wide piston has dc=i*c*dphi; opposite screen pistons cancel.
    epsilon = 1.0e-4
    piston_derivative_errors: dict[str, float] = {}
    for layer in range(2):
        plus = torch.zeros_like(phase)
        minus = torch.zeros_like(phase)
        plus[layer].fill_(epsilon)
        minus[layer].fill_(-epsilon)
        fields_plus, _ = _runtime_forward_batch(runtime, entrance, phase + plus, patterns, 3072, batch_size=1)
        fields_minus, _ = _runtime_forward_batch(runtime, entrance, phase + minus, patterns, 3072, batch_size=1)
        c_plus = _project_roi_batch(_crop_roi(fields_plus[0], 3072)[None], 3, "gaussian_s12", roi_plane)[0]
        c_minus = _project_roi_batch(_crop_roi(fields_minus[0], 3072)[None], 3, "gaussian_s12", roi_plane)[0]
        observed = (c_plus - c_minus) / (2.0 * epsilon)
        piston_derivative_errors[str(layer)] = _relative_vector_error(1j * direct_coeff, observed)
        del fields_plus, fields_minus, c_plus, c_minus
    checks["screen_common_piston_dc_equals_i_c_relative_error_by_layer"] = piston_derivative_errors
    checks["screen_common_piston_dc_equals_i_c_pass"] = max(piston_derivative_errors.values()) <= 1.0e-7
    piston = torch.zeros_like(phase)
    eps = 1.0e-3
    piston[0].fill_(eps)
    piston[1].fill_(-eps)
    plus_phase = phase + piston
    outputs_changed, _ = _runtime_forward_batch(runtime, entrance, plus_phase, patterns, 3072, batch_size=1)
    c_changed = _project_roi_batch(_crop_roi(outputs_changed[0], 3072)[None], 3, "gaussian_s12", roi_plane)[0]
    checks["opposite_screen_pistons_cancel_relative_error"] = _relative_vector_error(c_changed, direct_coeff)
    checks["opposite_screen_pistons_cancel_pass"] = checks["opposite_screen_pistons_cancel_relative_error"] <= 1.0e-8
    # Custom batched readout must agree with the existing single-field producer.
    direct_single = v5._readout_coefficients(_crop_roi(outputs[0], 3072), "complex128", device)
    custom_10 = {mode: _project_roi_batch(_crop_roi(outputs[0], 3072)[None], 10, mode, roi_plane)[0].detach().cpu() for mode in MODES}
    checks["custom_readout_vs_existing_producer_max_relative_error"] = max(
        _relative_vector_error(custom_10[mode], direct_single[mode]) for mode in MODES
    )
    checks["custom_readout_consumer_pass"] = checks["custom_readout_vs_existing_producer_max_relative_error"] <= 1.0e-4
    all_checks = [
        checks["propagation_adjoint_inner_product_pass"], checks["mode_projection_passivity_pass"],
        checks["omitted_pixel_area_rejected"], checks["dropped_complex_phase_rejected"],
        checks["production_forward_rejects_legacy_zero_response"], checks["screen_common_piston_dc_equals_i_c_pass"], checks["opposite_screen_pistons_cancel_pass"],
        checks["custom_readout_consumer_pass"],
    ]
    result = {
        "schema": "ols_frozen_phase_controllability.v7.preflight", "status": "PASS" if all(all_checks) else "FAIL_ENGINEERING_PREFLIGHT",
        "passed": all(all_checks), "checks": checks, "input_phase_sources": source_record,
        "cuda_device": torch.cuda.get_device_name(device), "probe_pattern_count": len(probes),
        "formal_training_updates": 0, "phase_optimization_updates": 0,
        "dataset_reads": 0, "task_evaluator_calls": 0, "fid_calls": 0,
    }
    _write_json(run_dir / "preflight.json", result)
    del runtime, entrance, roi_runtime, roi_field, outputs, outputs_changed
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
    for size in config["candidates"]["zero_phase_controls"]:
        cases.append({
            "case_name": f"zero_phase_{size}x{size}", "size": int(size),
            "objective": "zero_phase_control", "endpoint_key": None,
            "phase_name": "zero_phase", "optimized": False,
        })
    return cases


def _row_selection(pool_result: dict[str, torch.Tensor], size: int, device: torch.device) -> dict[str, torch.Tensor]:
    indices = torch.tensor(_pool_indices([tuple(v) for v in pool_result["coordinates_yx_m"].cpu().tolist()], size), dtype=torch.long)
    return {name: value.index_select(0, indices).to(device=device) for name, value in pool_result.items() if name != "coordinates_yx_m" and isinstance(value, torch.Tensor)}


def _control_targets(h: torch.Tensor, coefficients: torch.Tensor, size: int, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[tuple[str, torch.Tensor]], list[tuple[str, torch.Tensor]]]:
    threshold = float(config["measurement"]["main_relative_svd_threshold"])
    thresholds = [float(v) for v in config["measurement"]["relative_svd_thresholds"]]
    target_scales = [float(v) * math.sqrt(SOURCE_POWER) for v in config["measurement"]["target_magnitudes_sqrt_input_power"]]
    center_candidates = _power_metrics(coefficients, SOURCE_POWER, 1.0)["center_channel_indices"]
    center = int(center_candidates[0])
    col = size // 2
    edge_row = 0
    edge_col = min(max(col, 0), size - 1)
    edge = edge_row * size + edge_col
    corner = 0
    channel_targets = [("center", center), ("edge", edge), ("corner", corner)]
    m = torch.cat((h.real, h.imag), dim=0)
    _u, singular, vh = torch.linalg.svd(m, full_matrices=False)
    keep = singular >= singular.max() * threshold
    kept_indices = torch.where(keep)[0]
    top_idx = 0
    weak_idx = int(kept_indices[-1]) if len(kept_indices) else 0
    full_control_patterns: list[tuple[str, torch.Tensor]] = []
    bounded_command_payload: list[tuple[str, torch.Tensor]] = []
    records: list[dict[str, Any]] = []
    for magnitude_index, magnitude in enumerate(target_scales):
        for label, channel in channel_targets:
            for component, offset in (("real", 0), ("imag", size * size)):
                target = torch.zeros((2 * size * size,), dtype=torch.float64, device=h.device)
                target[offset + channel] = magnitude
                rec, command = _bounded_control(
                    m, _u, singular, vh, coefficients, size, thresholds, threshold, target,
                    tuple(float(v) for v in config["measurement"]["command_bounds"]),
                    float(config["measurement"]["target_base_command"]),
                )
                command_id = f"target_{label}_{component}_scale_{magnitude_index}"
                command_id = f"{command_id}_{len(bounded_command_payload):05d}"
                bounded_command_payload.append((command_id, command))
                rec["command_id"] = command_id
                rec.update({"target_family": "single_channel_basis", "target_label": label, "target_magnitude_scale_index": magnitude_index, "target_magnitude_requested_sqrt_w": magnitude})
                records.append(rec)
                if magnitude_index == 1 and component == "real":
                    full_control_patterns.append((f"control_{label}_real_at_{magnitude:.6e}", command))
    for label, direction_index in (("strongest_retained_direction", top_idx), ("weakest_retained_direction", weak_idx)):
        direction = vh[direction_index]
        delta = direction * (0.25 / direction.abs().max().clamp_min(EPS))
        direction_command = delta.to(torch.float64)
        full_control_patterns.append((f"control_{label}_sigma_index_{direction_index}", direction_command))
        command_id = f"control_{label}_sigma_index_{direction_index}"
        bounded_command_payload.append((command_id, direction_command))
        records.append({
            "target_family": "real_amplitude_singular_direction", "direction_label": label,
            "command_id": command_id,
            "singular_index": direction_index, "singular_value_sqrt_w": float(singular[direction_index]),
            "delta_command_l2": float(torch.linalg.vector_norm(delta)),
            "delta_command_max_abs": float(delta.abs().max()), "command_min": float((0.5 + delta).min()),
            "command_max": float((0.5 + delta).max()),
        })
    return records, full_control_patterns, bounded_command_payload


def _window_probe_and_direct_measurement(
    *, config: dict[str, Any], cases: list[dict[str, Any]], phases: dict[str, torch.Tensor],
    state_ids: dict[str, str], state_pool: dict[str, Any], run_dir: Path, device: torch.device,
    direct_patterns_by_case_mode: dict[str, dict[str, list[tuple[str, torch.Tensor]]]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    probe_names, base_patterns = _amp_probes(config)
    roi_runtime = v5._make_runtime(ROI, torch.float64, device)
    roi_plane = roi_runtime.ols_plane
    phase_report: dict[str, Any] = {}
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
                batch_outputs, batch_traces = _runtime_forward_batch(runtime, entrance, phases[phase_name], batch_patterns, grid, batch_size=batch_size)
                rois = _stack_common_roi_batch(batch_outputs, grid)
                if grid == GRIDS[0]:
                    baseline_roi.extend([value.detach().cpu() for value in rois])
                elif grid == GRIDS[1]:
                    for local, field in enumerate(rois):
                        index = start + local
                        raw, aligned = _field_relative_error(baseline_roi[index], field.detach().cpu(), align=True)
                        window_errors[pattern_names[index]][f"{GRIDS[0]}_vs_{grid}"] = {"raw_complex_relative_error": raw, "common_global_phase_aligned_complex_relative_error": aligned, "passed": aligned <= float(config["gates"]["window_aligned_complex_roi_error_max"])}
                    middle_roi.extend([value.detach().cpu() for value in rois])
                else:
                    for local, field in enumerate(rois):
                        index = start + local
                        raw_base, aligned_base = _field_relative_error(baseline_roi[index], field.detach().cpu(), align=True)
                        raw_middle, aligned_middle = _field_relative_error(middle_roi[index], field.detach().cpu(), align=True)
                        gate = float(config["gates"]["window_aligned_complex_roi_error_max"])
                        window_errors[pattern_names[index]][f"{GRIDS[0]}_vs_{grid}"] = {"raw_complex_relative_error": raw_base, "common_global_phase_aligned_complex_relative_error": aligned_base, "passed": aligned_base <= gate}
                        window_errors[pattern_names[index]][f"{GRIDS[1]}_vs_{grid}"] = {"raw_complex_relative_error": raw_middle, "common_global_phase_aligned_complex_relative_error": aligned_middle, "passed": aligned_middle <= gate}
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
                            result_rows.append({"pattern": pattern_names[index], "coefficient_relative_error": error, "window_qualified": window_errors[pattern_names[index]]["all_three_window_comparisons_pass"], "passed": error <= float(config["gates"]["coefficient_relative_error_max"])})
                    by_grid[str(grid)] = {"max_coefficient_relative_error": max(errors, default=0.0), "all_patterns_pass": all(value <= float(config["gates"]["coefficient_relative_error_max"]) for value in errors)}
                observed_3072 = direct_coefficients[GRIDS[0]][case_name][size][mode]
                baseline_coefficients = observed_3072[pattern_names.index("uniform_t_0.5")]
                pattern_power_metrics = [
                    {
                        "pattern": pattern_names[index],
                        **_power_metrics(
                            observed_3072[index], SOURCE_POWER,
                            direct_power[GRIDS[0]][index]["common_roi_power_before_fiber_w"],
                        ),
                    }
                    for index in range(len(pattern_names))
                ]
                direct_effects = []
                for direct_name in controls_by_case[case_name][mode]["direct_pattern_names"]:
                    direct_index = pattern_names.index(direct_name)
                    coefficient_after = observed_3072[direct_index]
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
                    "per_pattern_3072": result_rows,
                    "power_metrics_by_pattern_3072": pattern_power_metrics,
                    "observed_direct_control_effects_3072": direct_effects,
                    "qualifying_binary_probe_count": sum(item["window_qualified"] and item["passed"] for item in result_rows if item["pattern"].startswith("binary_probe_")),
                    "all_32_binary_probe_checks_pass": all(item["window_qualified"] and item["passed"] for item in result_rows if item["pattern"].startswith("binary_probe_")),
                    "uniform_t_0.5_window_qualified": window_errors["uniform_t_0.5"]["all_three_window_comparisons_pass"],
                    "all_selected_direct_control_checks_pass": all(
                        next((item["window_qualified"] and item["passed"] for item in result_rows if item["pattern"] == direct_name), False)
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
            "3072_t_0.5_reference_field_archive": None,
        }
        # Keep one audit-ready complex64 field per unique frozen phase state.
        base_index = pattern_names.index("uniform_t_0.5")
        field_path = run_dir / "representative_fields" / f"{phase_state_id}_complex64_3072_t05.npz"
        field_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(field_path, field=baseline_roi[base_index].numpy(), metadata_json=np.asarray(json.dumps({"phase_state_id": phase_state_id, "pattern": "uniform_t_0.5", "grid": 3072, "roi_size": ROI, "dx_m": DX, "dtype": "complex64"}, sort_keys=True)))
        phase_report[phase_state_id]["3072_t_0.5_reference_field_archive"] = str(field_path.relative_to(run_dir)).replace("\\", "/")
        _write_json(run_dir / "probe_progress.json", {"completed_phase_state_id": phase_state_id, "completed_pattern_count": len(pattern_names), "phase_optimization_updates": 0})
        del baseline_roi, middle_roi, direct_coefficients, direct_power, patterns_by_name, patterns, arrays
        gc.collect()
    return phase_report, {"cases": [case["case_name"] for case in cases], "probe_pattern_count": 34}


def _run_full(config: dict[str, Any], config_path: Path, run_dir: Path, root: Path) -> dict[str, Any]:
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("v7 full measurement requires a saved PASS preflight from this exact attempt")
    device = torch.device("cuda:0")
    phases, source_record = _load_v6_phases(config, root)
    cases = _build_cases(config, phases)
    state_ids = _phase_state_ids(phases, cases)
    saved_phases = {name: value.detach().cpu().to(torch.float32) for name, value in phases.items()}
    torch.save({"schema": "ols_frozen_phase_screens.v7", "phases": saved_phases, "phase_state_ids": state_ids}, run_dir / "phase_screens_v7.pt")
    _write_json(run_dir / "configuration.json", {"config_path": str(config_path), "configuration": config, "source_evidence": source_record, "phase_state_ids": state_ids})
    roi_runtime = v5._make_runtime(ROI, torch.float64, device)
    roi_plane = roi_runtime.ols_plane
    state_pool: dict[str, Any] = {}
    phase_names = list(dict.fromkeys(case["phase_name"] for case in cases))
    phase_real_dtype = torch.float64
    for phase_name in phase_names:
        phase_state_id = state_ids[phase_name]
        sizes = _phase_pool_sizes(cases, phase_name)
        coordinates = _coordinate_pool(sizes, device)
        if phase_state_id not in state_pool:
            state_pool[phase_state_id] = {"pool_coordinates": coordinates, "modes": {}}
        elif state_pool[phase_state_id]["pool_coordinates"] != coordinates:
            # Equal phase tensors can feed nested cases; their union is expanded before any matrix is consumed.
            union = sorted(set(state_pool[phase_state_id]["pool_coordinates"]) | set(coordinates), key=lambda item: (item[0], item[1]))
            if state_pool[phase_state_id]["modes"]:
                raise ValueError("an equal frozen phase was associated with different array coordinate pools after matrix construction")
            state_pool[phase_state_id]["pool_coordinates"] = union
        phase = phases[phase_name].to(device=device, dtype=phase_real_dtype)
        state_pool[phase_state_id]["phase_tensor"] = phases[phase_name]
        for grid in GRIDS:
            runtime = v5._make_runtime(grid, torch.float64, device)
            entrance = v5._encoder_entrance(runtime, torch.float64, device)
            with torch.no_grad():
                runtime.encoder.phase.copy_(phase)
                fields, records, _stats = runtime.encoder(
                    ComplexField(entrance.tensor * 0.5, entrance.plane, OperationProvenance("v7_jacobian_reference", entrance.plane.plane_id, None)),
                    f"ols_v7_jacobian_{phase_state_id}",
                )
            first_out = records[0].tensor[0, 0].to(torch.complex128)
            del fields, records, _stats
            for mode in MODES:
                pool = _build_response_pool(
                    run_dir=run_dir, runtime=runtime, roi_plane=roi_plane, entrance=entrance,
                    phases=phase, phase_state_id=phase_state_id, mode_name=mode, grid=grid,
                    coordinates=state_pool[phase_state_id]["pool_coordinates"],
                    first_layer_output=first_out if grid == GRIDS[0] else None,
                    include_phase_jacobian=grid == GRIDS[0],
                )
                mode_entry = state_pool[phase_state_id]["modes"].setdefault(mode, {})
                mode_entry[grid] = pool
                archive_path = run_dir / "response_matrices" / f"{phase_state_id}_{mode}_{grid}.npz"
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                arrays = {"K_dimensionless": pool["K"].numpy(), "H_sqrt_w": pool["H"].numpy(), "coordinates_yx_m": pool["coordinates_yx_m"].numpy()}
                if "J_phi" in pool:
                    arrays["J_phi_sqrt_w_per_rad"] = pool["J_phi"].numpy()
                arrays["metadata_json"] = np.asarray(json.dumps({"phase_state_id": phase_state_id, "mode": mode, "grid": grid, "row_pool_count": len(coordinates), "pixel_count": 10000, "K_unit": "dimensionless", "H_unit": "sqrt(W) per unit command", "J_phi_unit": "sqrt(W) per rad"}, sort_keys=True))
                np.savez_compressed(archive_path, **arrays)
                mode_entry[grid]["archive_path"] = str(archive_path.relative_to(run_dir)).replace("\\", "/")
            del runtime, entrance, first_out
            torch.cuda.empty_cache()
            gc.collect()
        state_pool[phase_state_id]["phase_name"] = phase_name
    # Spectra, uncertainty envelopes, and bounded controls are computed only after every matrix grid is present.
    thresholds = [float(v) for v in config["measurement"]["relative_svd_thresholds"]]
    cases_report: dict[str, Any] = {}
    control_records: dict[str, Any] = {}
    direct_patterns_by_case_mode: dict[str, dict[str, list[tuple[str, torch.Tensor]]]] = {}
    bounded_command_arrays: dict[str, np.ndarray] = {}
    for case in cases:
        state_id, size, case_name = state_ids[case["phase_name"]], int(case["size"]), case["case_name"]
        indices = _pool_indices(state_pool[state_id]["pool_coordinates"], size)
        cases_report[case_name] = {**case, "phase_state_id": state_id, "channel_count": size * size, "fiber_coordinates_xy_m_row_major": [[x, y] for y, x in _coords(size, torch.device("cpu"))], "modes": {}}
        control_records[case_name] = {}
        direct_patterns_by_case_mode[case_name] = {}
        for mode in MODES:
            mode_entry = state_pool[state_id]["modes"][mode]
            selected_by_grid = {grid: {key: tensor.index_select(0, torch.tensor(indices, dtype=torch.long)) for key, tensor in mode_entry[grid].items() if key in ("K", "H", "J_phi")} for grid in GRIDS}
            base_k = selected_by_grid[3072]["K"].to(device)
            base_h = selected_by_grid[3072]["H"].to(device)
            spectra = _response_spectra(base_k, base_h, thresholds)
            window_metrics: dict[str, Any] = {}
            base_real = torch.cat((base_h.real, base_h.imag), dim=0)
            base_real_singular = torch.linalg.svdvals(base_real)
            base_complex_singular = torch.linalg.svdvals(base_k)
            for grid in GRIDS[1:]:
                k_delta = base_k - selected_by_grid[grid]["K"].to(device)
                h_delta = base_h - selected_by_grid[grid]["H"].to(device)
                k_delta_norm = float(torch.linalg.svdvals(k_delta).max())
                h_delta_norm = float(torch.linalg.svdvals(torch.cat((h_delta.real, h_delta.imag), dim=0)).max())
                h_grid = selected_by_grid[grid]["H"].to(device)
                h_singular = torch.linalg.svdvals(torch.cat((h_grid.real, h_grid.imag), dim=0))
                weak = {}
                for threshold in thresholds:
                    weak[str(threshold)] = int(((base_real_singular - h_delta_norm) > threshold * (base_real_singular.max() + h_delta_norm)).sum())
                window_metrics[str(grid)] = {
                    "K_operator_difference_spectral_norm": k_delta_norm,
                    "K_operator_difference_relative_to_3072_max_sigma": k_delta_norm / max(float(base_complex_singular.max()), EPS),
                    "H_real_control_operator_difference_spectral_norm_sqrt_w": h_delta_norm,
                    "H_real_control_difference_relative_to_3072_max_sigma": h_delta_norm / max(float(base_real_singular.max()), EPS),
                    "grid_real_control_max_singular_value_sqrt_w": float(h_singular.max()),
                    "weak_singular_directions_above_window_uncertainty_by_relative_threshold": weak,
                    "interpretation": "Weyl bound based on measured cross-grid operator difference; not detector-noise observability",
                }
            mode_report = {
                "matrix_archive_by_grid": {str(grid): mode_entry[grid]["archive_path"] for grid in GRIDS},
                "spectra_at_3072": spectra,
                "matrix_window_uncertainty": window_metrics,
                "complex_K_shape": list(base_k.shape), "complex_H_shape": list(base_h.shape),
                "H_units": "sqrt(W) per unit real amplitude command",
            }
            j = selected_by_grid[3072]["J_phi"].to(device)
            t0 = torch.full((10000,), 0.5, dtype=torch.float64)
            c0 = _response_at_command(base_h, t0)
            gt = 2.0 * torch.real(c0.conj()[:, None] * base_h)
            gphi = 2.0 * torch.real(c0.conj()[:, None] * j)
            mode_report["power_jacobian_Gt"] = {
                "shape": list(gt.shape), "units": "W per unit amplitude command",
                "spectrum": _spectrum(gt, thresholds, singular_value_units="W per unit command", squared_gain_units="W squared per squared command norm"),
            }
            mode_report["complex_coefficient_phase_jacobian_Jphi"] = {
                "shape": list(j.shape), "units": "sqrt(W) per radian",
                "spectrum": _phase_jacobian_shape_report(j, c0, thresholds),
                "power_jacobian_Gphi": {"shape": list(gphi.shape), "units": "W per radian", "spectrum": _spectrum(gphi, thresholds, singular_value_units="W per radian-norm", squared_gain_units="W squared per squared radian-norm")},
            }
            target_records, direct_patterns, all_commands = _control_targets(base_h, c0, size, config)
            direct_patterns_by_case_mode[case_name][mode] = direct_patterns
            for command_id, command in all_commands:
                qualified_name = f"{case_name}__{mode}__{command_id}"
                bounded_command_arrays[qualified_name] = command.detach().cpu().numpy().astype(np.float64, copy=False)
            control_records[case_name][mode] = {
                "all_channel_bounded_single_output_controls": target_records,
                "direct_validation_pattern_names": [label for label, _ in direct_patterns],
                "bounded_command_count": len(all_commands),
                "command_vectors_saved_in": "bounded_control_commands_v7.npz",
            }
            cases_report[case_name]["modes"][mode] = mode_report
            del gt, gphi, j
            del base_k, base_h, base_real, base_real_singular, base_complex_singular, spectra
        del selected_by_grid
    bounded_path = run_dir / "bounded_control_commands_v7.npz"
    np.savez_compressed(
        bounded_path,
        command_names=np.asarray(list(bounded_command_arrays), dtype="U128"),
        delta_commands=np.stack(list(bounded_command_arrays.values())).astype(np.float64),
    )
    jacobian_report: dict[str, Any] = {}
    for phase_name in phase_names:
        state_id = state_ids[phase_name]
        runtime = v5._make_runtime(3072, torch.float64, device)
        entrance = v5._encoder_entrance(runtime, torch.float64, device)
        jacobian_report[state_id] = _phase_jacobian_preflight(
            runtime=runtime, entrance=entrance, phase=phases[phase_name], phase_state_id=state_id,
            cases=cases, state_pool=state_pool, roi_plane=roi_plane, device=device, config=config,
        )
        del runtime, entrance
        torch.cuda.empty_cache()
        gc.collect()
    phase_probe_report, probe_design = _window_probe_and_direct_measurement(
        config=config, cases=cases, phases=phases, state_ids=state_ids, state_pool=state_pool, run_dir=run_dir, device=device,
        direct_patterns_by_case_mode=direct_patterns_by_case_mode,
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    report = {
        "schema": SCHEMA, "status": "VALID_COMPLETED", "decision_question": "Do frozen v6 phase candidates have numerical, absolute-power amplitude-control evidence for independently changing OLS channels?",
        "historical_v4_g3_status": "G3_FAIL_PRESERVED", "v5_window_baseline": "3072_4096_6144", "v6_phase_source": source_record,
        "fixed_physical_system": {"wavelength_m": WAVELENGTH, "dx_m": DX, "phase_screens": [100, 100], "aperture_m": 200e-6, "spacing_each_m": 1.5e-3, "layer_power_transmission": TRANSMISSION, "source_power_w": SOURCE_POWER, "pitch_m": PITCH, "mode_field_diameter_m": 14.222e-6, "geometry_status": "PROJECT_GEOMETRY_UNMEASURED"},
        "preflight": preflight, "frozen_phase_states": {name: {"phase_state_id": state_ids[name], "source": source_record["selected_endpoints"].get(name, {"source": "exact all-zero phase pair"})} for name in phases},
        "amplitude_probe_design": probe_design, "cases": cases_report, "bounded_controls": control_records,
        "direct_probe_window_and_coefficient_checks": phase_probe_report, "phase_jacobian_validation": jacobian_report,
        "matrix_payload": "response_matrices/*.npz (one compressed K/H archive per frozen phase, mode, and grid; 3072 archives also contain J_phi)", "phase_payload": "phase_screens_v7.pt",
        "measurement_contract": {
            "response_definitions": {"K": "c=Ka; q_p=1/sqrt(pixel area), a_p=E_source sqrt(pixel area) t_p; K dimensionless", "H": "c=Ht; H=K diag(E_source sqrt(pixel area)), sqrt(W) per unit command", "M_H": "[Re H; Im H] real command response", "G_t": "2 Re(diag(c*) H)", "G_phi": "2 Re(diag(c*) J_phi)"},
            "phase_svd_global_phase_projection": "raw spectrum is retained; only shape spectrum may remove one common output global-phase direction",
            "matrix_prediction_gate": float(config["gates"]["coefficient_relative_error_max"]),
            "full_roi_window_gate": float(config["gates"]["window_aligned_complex_roi_error_max"]),
            "probe_commands_not_renormalized": True, "probe_input_power_fixed_before_amplitude_command_w": SOURCE_POWER,
            "no_detector_noise_or_semantic_interface_model": True,
        },
        "updates_and_forbidden_io": {"formal_training_updates": 0, "phase_optimization_updates": 0, "dataset_reads": 0, "task_evaluator_calls": 0, "fid_calls": 0, "loss_changes": 0, "material_forward_calls": 0, "decoder_camera_changes": 0},
        "execution": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.cuda.get_device_name(device), "elapsed_seconds": time.time() - _START_TIME},
        "claim_limits": [
            "pixel-level amplitude controllability is an upper bound for a future input interface, not semantic or measured camera observability",
            "mathematical singular rank is not detector-noise-limited experimental dimension",
            "Gaussian is the main reference and LP01 is a same-field model sensitivity check, not an independently optimized upper bound",
            "all fiber/mode/optics parameters remain unmeasured simulation assumptions",
            "finite v6 phase candidates do not establish global optimization or a physical channel-capacity bound",
        ],
        "scientific_decision": _scientific_decision(cases_report, phase_probe_report, jacobian_report),
    }
    _write_json(run_dir / "report.json", report)
    (run_dir / "exit_status.txt").write_text("0\n", encoding="ascii")
    del state_pool, phases, cases, roi_runtime
    torch.cuda.empty_cache()
    gc.collect()
    return report


def _scientific_decision(cases: dict[str, Any], probes: dict[str, Any], jacobians: dict[str, Any]) -> dict[str, Any]:
    required = tuple(TARGET_NAMES.values())
    qualified: dict[str, bool] = {}
    for name in required:
        matching = [
            candidate_record
            for phase in probes.values()
            for candidate, candidate_record in phase.get("per_candidate_matrix_to_direct_checks", {}).items()
            if candidate == name
        ]
        mode_checks = [mode for item in matching for mode in item.get("by_mode", {}).values()]
        qualified[name] = len(matching) == 1 and len(mode_checks) == len(MODES) and all(
            mode.get("all_32_binary_probe_checks_pass", False)
            and mode.get("all_selected_direct_control_checks_pass", False)
            and mode.get("uniform_t_0.5_window_qualified", False)
            for mode in mode_checks
        )
    all_jac = all(value.get("all_pixel_checks_pass", False) for value in jacobians.values())
    four_nine = all(qualified[name] for name in ("candidate_2x2_A", "candidate_2x2_C", "candidate_3x3_B", "candidate_5x5_B"))
    hundred = all(qualified[name] for name in ("candidate_10x10_A", "candidate_10x10_C"))
    zero_controls = {
        f"zero_phase_{size}x{size}": any(
            case_name == f"zero_phase_{size}x{size}" and all(
                value.get("all_32_binary_probe_checks_pass", False)
                and value.get("all_selected_direct_control_checks_pass", False)
                and value.get("uniform_t_0.5_window_qualified", False)
                for value in item.get("by_mode", {}).values()
            )
            for phase in probes.values()
            for case_name, item in phase.get("per_candidate_matrix_to_direct_checks", {}).items()
        )
        for size in SIZES
    }
    valid = four_nine and all_jac
    return {
        "status": "CONTROL_EVIDENCE_AVAILABLE" if valid else "INSUFFICIENT_EVIDENCE",
        "selected_candidate_probe_gates": qualified,
        "four_channel_and_nine_channel_candidates_qualified": four_nine,
        "100_channel_candidate_measurements_qualified": hundred,
        "zero_phase_control_probe_gates": zero_controls,
        "phase_pixel_jacobian_checks_pass": all_jac,
        "interpretation": "Supports planning a frozen four-vs-nine decoder/camera interface comparison only when both four- and nine-channel phase candidates and phase-Jacobian checks are numerically qualified; 100-channel qualification is reported separately. It does not select an OLS dimension or establish camera observability.",
        "permitted_next_action": "plan a frozen four-vs-nine decoder/camera coordinate and unit interface comparison" if valid else "repair or repeat only invalid numerical checks; do not infer capacity from failed measurements",
    }


_START_TIME = time.time()


def run(config_path: Path, run_dir: Path, *, preflight_only: bool) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = Path.cwd().resolve()
    config = _load_config(config_path)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_copy = run_dir / "configuration.yaml"
    if config_copy.exists() and config_copy.read_text(encoding="utf-8") != config_path.read_text(encoding="utf-8"):
        raise ValueError("existing v7 run directory contains a different explicit config")
    if not config_copy.exists():
        config_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    source_dir = run_dir / "source_snapshot"
    source_dir.mkdir(parents=True, exist_ok=True)
    snapshot = source_dir / "diagnose_ols_frozen_phase_controllability_v7.py"
    if snapshot.exists() and snapshot.read_bytes() != Path(__file__).read_bytes():
        raise ValueError("saved v7 source snapshot differs from the invoked CLI")
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    if preflight_only:
        if (run_dir / "preflight.json").exists():
            raise FileExistsError("refusing to overwrite existing v7 preflight evidence")
        return _preflight(config, config_path, run_dir, root)
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file() or json.loads(preflight_path.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("v7 complete measurement requires this attempt's passed preflight")
    return _run_full(config, config_path, run_dir, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.run_dir, preflight_only=args.preflight_only)
    print(json.dumps({"status": result.get("status"), "passed": result.get("passed"), "scientific_decision": result.get("scientific_decision", {}).get("status"), "run_dir": str(args.run_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
