"""Frozen four/nine-channel Decoder and proxy-camera interface audit.

This research-only runner consumes the exact v8 compact evidence package. It
does not train, optimize phases, read datasets, or call task evaluators.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.optics import fiber_operator
from ldm_project.v2.optics.fiber_operator import lp01_eigenpair, lp01_radial_profile
from ldm_project.v2.optics.propagation import angular_spectrum_propagate, clear_production_transfer_cache
from ldm_project.v2.optics.relays import (
    CoherentImagingRelay,
    path_abcd,
)


SCHEMA = "ols_decoder_camera_interface.v9"
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_decoder_camera_interface_v9.yaml")
DEFAULT_RUN_DIR = Path("outputs/20261016_ols_decoder_camera_interface_v9/runs/attempt_01")
WAVELENGTH = 532.0e-9
DX = 2.0e-6
SOURCE_POWER = 249.3e-6
TRANSMISSION = 0.889
PITCH = 125.0e-6
WAIST = 7.111e-6
ROI = 2048
GRIDS = (4096, 6144, 8192)
SIZES = {"candidate_2x2_C": 2, "candidate_3x3_B": 3}
MODES = ("gaussian_s12", "lp01")
ROUTES = ("direct", "explicit_4f")
DECODER_STATES = ("zero_phase", "fixed_low_frequency")
EXPOSURES = (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
EPS = 1.0e-30
_START = time.time()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} configuration")
    fixed = {
        "optics": {
            "wavelength_m": WAVELENGTH, "dx_m": DX, "encoder_aperture_m": 200.0e-6,
            "encoder_shape": [100, 100], "fiber_pitch_m": PITCH,
            "gaussian_1e_amplitude_radius_m": WAIST, "lp01_v_number": 2.2,
            "layer_power_transmission": TRANSMISSION, "source_power_w": SOURCE_POWER,
            "grids": list(GRIDS), "common_roi_pixels": ROI,
            "direct_route_distance_m": 1.5e-3, "fiber_transport": "ideal_identity",
        },
        "routes": {
            "direct": {"kind": "free_space", "distance_m": 1.5e-3},
            "explicit_4f": {
                "kind": "coherent_4f", "focal_lengths_m": [1.0e-2, 5.0e-3],
                "segments_m": [1.0e-2, 1.5e-2, 5.0e-3], "magnification": -0.5,
                "clear_aperture_diameter_m": 2.0e-3, "lens_power_transmission": 0.98,
            },
        },
        "decoder": {
            "active_shape": [100, 100], "pixel_pitch_m": DX,
            "layer_spacing_m": 1.5e-3, "layer_power_transmission": TRANSMISSION,
            "states": list(DECODER_STATES), "low_frequency_seed": 2026101609,
            "low_frequency_max_order": 2, "phase_rms_rad": 0.5,
            "input_grid_roi": "center_100x100",
        },
        "backend_relay": {
            "magnification": -2.5, "segments_m": [0.1, 0.25],
            "focal_length_m": 0.0714285714285714,
            "clear_aperture_m": 0.0266, "additional_power_transmission": 1.0,
        },
        "camera": {
            "shape": [200, 200], "pixel_pitch_m": 2.5e-6,
            "field_of_view_m": 5.0e-4, "quantum_efficiency": 0.5,
            "read_noise_e_rms": 1.0, "dark_current_e_per_pixel_s": 0.0,
            "full_well_e": 30000.0, "primary_exposure_s": 1.0e-3,
            "exposure_sweep_s": list(EXPOSURES), "noise_seed": 2026101610,
        },
        "measurement": {
            "pattern_count": 50, "target_base_command": 0.5,
            "command_bounds": [0.25, 0.75], "input_perturbation_rms": 0.01,
            "relative_svd_thresholds": [1.0e-1, 1.0e-2, 1.0e-3, 1.0e-4],
            "pixel_integral_primary_order": 4, "pixel_integral_reference_order": 8,
            "phase_pattern_seed": 2026101609,
        },
        "scope": {
            "formal_training_updates": 0, "phase_optimization_updates": 0,
            "dataset_reads": 0, "task_evaluator_calls": 0, "fid_calls": 0,
            "loss_changes": 0, "material_forward_calls": 0,
            "production_cli_migration": "forbidden", "checkpoint_migration": "forbidden",
            "automatic_ols_dimension_selection": "forbidden",
        },
    }
    for key, expected in fixed.items():
        if config.get(key) != expected:
            raise ValueError(f"v9 fixed contract changed in {key}")
    expected_gates = {
        "aligned_common_roi_complex64_max": 1.0e-4,
        "pixel_integral_relative_l2_max": 1.0e-4,
        "matrix_prediction_relative_max": 1.0e-4,
        "full_forward_superposition_relative_max": 1.0e-4,
        "passivity_relative_tolerance": 1.0e-5,
        "complex128_jacobian_absolute_w_per_command": 1.0e-10,
        "complex128_jacobian_relative": 1.0e-4,
    }
    if config.get("gates") != expected_gates:
        raise ValueError("v9 fixed numerical gates changed")
    return config


def _resolve(root: Path, path: str) -> Path:
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"research input resolves outside the project root: {path}")
    return result


def _load_source(config: dict[str, Any], root: Path, source_override: str | None) -> dict[str, Any]:
    source_dir = _resolve(root, source_override or config["source"]["v8_run_dir"])
    required = [
        config["source"]["report"], config["source"]["phase_screens"],
        config["source"]["command_archives"]["candidate_2x2_C"],
        config["source"]["command_archives"]["candidate_3x3_B"],
    ]
    for relative in required:
        if not (source_dir / relative).is_file():
            raise FileNotFoundError(f"required v8 source is missing: {source_dir / relative}")
    report = json.loads((source_dir / config["source"]["report"]).read_text(encoding="utf-8"))
    if report.get("schema") != "ols_four_nine_window_closure.v8" or report.get("status") != "VALID_COMPLETED":
        raise ValueError("the explicitly selected v8 source report is not the expected valid v8 result")
    if report.get("preflight", {}).get("status") != "PASS":
        raise ValueError("the explicitly selected v8 source lacks a passing preflight")
    phases_obj = torch.load(source_dir / config["source"]["phase_screens"], map_location="cpu", weights_only=False)
    if phases_obj.get("schema") != "ols_frozen_phase_screens.v8":
        raise ValueError("v8 phase archive schema is unexpected")
    if set(phases_obj.get("phases", {})) != set(SIZES):
        raise ValueError("v8 phase archive does not contain exactly the two selected candidates")
    candidates: dict[str, Any] = {}
    for name, size in SIZES.items():
        case = report.get("cases", {}).get(name)
        if not isinstance(case, dict) or case.get("size") != size or case.get("channel_count") != size * size:
            raise ValueError(f"v8 report candidate identity is invalid for {name}")
        phase = phases_obj["phases"][name]
        if list(phase.shape) != [2, 1, 100, 100] or not torch.isfinite(phase).all():
            raise ValueError(f"v8 phase tensor is malformed for {name}")
        if str(phases_obj.get("phase_state_ids", {}).get(name)) != str(case.get("phase_state_id")):
            raise ValueError(f"v8 frozen phase identity does not match its report for {name}")
        expected_yx = np.asarray(_coordinates(size), dtype=np.float64)
        reported_xy = np.asarray(case["fiber_coordinates_xy_m_row_major"], dtype=np.float64)
        if reported_xy.shape != (size * size, 2) or not np.allclose(reported_xy[:, ::-1], expected_yx, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"v8 report fiber coordinates do not match the selected row-major interface for {name}")
        archive_path = source_dir / config["source"]["command_archives"][name]
        with np.load(archive_path, allow_pickle=False) as archive:
            names = np.asarray(archive["pattern_names"])
            commands = np.asarray(archive["amplitude_commands"], dtype=np.float32)
            if names.shape != (50,) or commands.shape != (50, 10000) or not np.isfinite(commands).all():
                raise ValueError(f"v8 amplitude archive shape or values are invalid for {name}")
            if len(set(names.tolist())) != 50:
                raise ValueError(f"v8 pattern names are not unique for {name}")
            coefficients: dict[int, dict[str, np.ndarray]] = {}
            traces: dict[int, dict[str, np.ndarray]] = {}
            for grid in GRIDS:
                coefficients[grid] = {}
                for mode in MODES:
                    key = f"coeff_{grid}_{name}_{mode}"
                    if key not in archive.files:
                        raise ValueError(f"v8 coefficients are missing: {key}")
                    values = np.asarray(archive[key], dtype=np.complex128)
                    if values.shape != (50, size * size) or not np.isfinite(values).all():
                        raise ValueError(f"v8 coefficients are malformed: {key}")
                    coefficients[grid][mode] = values.copy()
                traces[grid] = {
                    key.removeprefix(f"power_trace_{grid}_"): np.asarray(archive[key]).copy()
                    for key in archive.files if key.startswith(f"power_trace_{grid}_")
                }
        matrices: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        for mode in MODES:
            matrices[mode] = {}
            for grid in GRIDS:
                entry = case["modes"][mode]["matrix_archive_by_grid"][str(grid)]
                matrix_path = source_dir / entry
                if not matrix_path.is_file():
                    raise FileNotFoundError(f"required v8 response matrix is missing: {matrix_path}")
                with np.load(matrix_path, allow_pickle=False) as archive:
                    metadata = json.loads(str(archive["metadata_json"].item()))
                    h = np.asarray(archive["H_sqrt_w"], dtype=np.complex128)
                    coords = np.asarray(archive["coordinates_yx_m"], dtype=np.float64)
                if metadata.get("mode") != mode or metadata.get("grid") != grid or h.shape != (size * size, 10000):
                    raise ValueError(f"v8 matrix metadata or shape is invalid: {matrix_path}")
                if metadata.get("H_unit") != "sqrt(W) per unit command" or not np.isfinite(h).all():
                    raise ValueError(f"v8 matrix units or values are invalid: {matrix_path}")
                expected_yx = np.asarray(_coordinates(size), dtype=np.float64)
                if coords.shape != expected_yx.shape or not np.allclose(coords, expected_yx, rtol=0.0, atol=1.0e-12):
                    raise ValueError(f"v8 matrix channel ordering differs from the frozen fiber coordinates: {matrix_path}")
                matrices[mode][grid] = {"H": h, "coordinates_yx_m": coords}
        candidates[name] = {
            "size": size, "case": case, "phase": phase.detach().cpu().to(torch.float32),
            "phase_state_id": str(phases_obj["phase_state_ids"][name]),
            "pattern_names": names.copy(), "commands": commands.copy(),
            "coefficients": coefficients, "power_traces": traces, "matrices": matrices,
        }
    return {"run_dir": str(source_dir), "report": report, "candidates": candidates}


def _coordinates(size: int) -> list[tuple[float, float]]:
    axis = [(index - (size - 1.0) / 2.0) * PITCH for index in range(size)]
    return [(y, x) for y in axis for x in axis]


def _new_decoder_phases(seed: int, device: torch.device) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    axis = (np.arange(100, dtype=np.float64) - 49.5) * DX
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    phases = np.zeros((2, 100, 100), dtype=np.float64)
    aperture = 200.0e-6
    for layer in range(2):
        surface = np.zeros((100, 100), dtype=np.float64)
        for ky in range(-2, 3):
            for kx in range(-2, 3):
                if kx == 0 and ky == 0:
                    continue
                angle = 2.0 * math.pi * (kx * xx + ky * yy) / aperture
                a_cos, a_sin = rng.standard_normal(2)
                surface += a_cos * np.cos(angle) + a_sin * np.sin(angle)
        surface -= surface.mean()
        rms = float(np.sqrt(np.mean(surface**2)))
        if not math.isfinite(rms) or rms <= 0.0:
            raise ValueError("low-frequency decoder phase generation produced a degenerate layer")
        phases[layer] = surface * (0.5 / rms)
    nonzero = torch.from_numpy(phases).to(device=device, dtype=torch.float64)
    return {
        "zero_phase": torch.zeros((2, 1, 100, 100), dtype=torch.float64, device=device),
        "fixed_low_frequency": nonzero[:, None],
    }


def _write_decoder_phase_files(run_dir: Path, phases: dict[str, torch.Tensor]) -> None:
    payload = {name: value.detach().cpu() for name, value in phases.items()}
    torch.save({"schema": "ols_decoder_phase_screens.v9", "seed": 2026101609, "phases": payload}, run_dir / "decoder_phase_screens_v9.pt")
    np.savez_compressed(
        run_dir / "decoder_phase_screens_v9.npz",
        zero_phase=payload["zero_phase"].numpy(),
        fixed_low_frequency=payload["fixed_low_frequency"].numpy(),
        seed=np.asarray(2026101609),
    )


def _plane(name: str, shape: tuple[int, int], pitch: tuple[float, float], z_m: float, frame: str) -> PlaneSpec:
    return PlaneSpec(name, shape, pitch, WAVELENGTH, z_m, frame)


def _global_mode(mode: str, coordinate: tuple[float, float], grid: int, device: torch.device) -> torch.Tensor:
    roi_plane = _plane("fiber_mode_roi", (ROI, ROI), (DX, DX), 0.0, "fiber_global")
    cy, cx = coordinate
    axis_y = (torch.arange(ROI, dtype=torch.float64, device=device) + 0.5) * DX - roi_plane.fov_m[0] / 2.0
    axis_x = (torch.arange(ROI, dtype=torch.float64, device=device) + 0.5) * DX - roi_plane.fov_m[1] / 2.0
    if mode == "gaussian_s12":
        amplitude = (2.0 / (math.pi * WAIST**2)) ** 0.25
        profile = torch.outer(amplitude * torch.exp(-((axis_y - cy) / WAIST).square()),
                              amplitude * torch.exp(-((axis_x - cx) / WAIST).square())).to(torch.complex128)
    elif mode == "lp01":
        def support(axis: torch.Tensor, center: float) -> tuple[int, int]:
            lower = int(torch.searchsorted(axis, torch.tensor(center - PITCH / 2.0, device=device, dtype=axis.dtype)))
            upper = int(torch.searchsorted(axis, torch.tensor(center + PITCH / 2.0, device=device, dtype=axis.dtype)))
            if lower >= upper:
                raise ValueError("LP01 fiber support is unresolved on the reference plane")
            return lower, upper

        y0, y1 = support(axis_y, cy)
        x0, x1 = support(axis_x, cx)
        yy, xx = torch.meshgrid(axis_y[y0:y1], axis_x[x0:x1], indexing="ij")
        _, _, rho_e = lp01_eigenpair(2.2)
        core_radius = WAIST / rho_e
        local = lp01_radial_profile(torch.sqrt((yy - cy).square() + (xx - cx).square()) / core_radius, 2.2).to(torch.float64)
        norm = torch.sqrt(local.square().sum() * DX * DX)
        if not torch.isfinite(norm) or float(norm) <= 0.0:
            raise ValueError("LP01 modal normalization failed")
        profile = torch.zeros((ROI, ROI), dtype=torch.complex128, device=device)
        profile[y0:y1, x0:x1] = torch.complex(local / norm, torch.zeros_like(local))
    else:
        raise ValueError(f"unknown fiber mode: {mode}")
    full = torch.zeros((grid, grid), dtype=torch.complex128, device=device)
    start = (grid - ROI) // 2
    full[start : start + ROI, start : start + ROI] = profile
    return full


def _power(field: torch.Tensor, area: float) -> float:
    return float(field.abs().square().sum().real * area)


def _crop_active(field: torch.Tensor) -> torch.Tensor:
    start_y = (field.shape[-2] - 100) // 2
    start_x = (field.shape[-1] - 100) // 2
    if start_y < 0 or start_x < 0 or start_y * 2 + 100 != field.shape[-2] or start_x * 2 + 100 != field.shape[-1]:
        raise ValueError("global field cannot be centered on the 100x100 active decoder aperture")
    return field[..., start_y : start_y + 100, start_x : start_x + 100]


def _crop_roi(field: torch.Tensor) -> torch.Tensor:
    start = (field.shape[-1] - ROI) // 2
    if field.shape[-1] != field.shape[-2] or start < 0 or start * 2 + ROI != field.shape[-1]:
        raise ValueError("propagation grid does not support the centered 4.096 mm common ROI")
    return field[..., start : start + ROI, start : start + ROI]


def _propagate(field: torch.Tensor, grid: int, z_m: float, distance_m: float, plane_id: str) -> tuple[torch.Tensor, float]:
    input_plane = _plane(f"{plane_id}_input", (grid, grid), (DX, DX), z_m, "physical_global")
    wrapped = ComplexField(field, input_plane, OperationProvenance("v9_research_forward", input_plane.plane_id, None))
    result = angular_spectrum_propagate(wrapped, distance_m, plane_id)
    return result.tensor, result.plane.z_m


def _lens(field: torch.Tensor, grid: int, focal_length_m: float, power_transmission: float, plane_id: str) -> torch.Tensor:
    device, real_dtype = field.device, field.real.dtype
    axis = (torch.arange(grid, device=device, dtype=real_dtype) - (grid - 1.0) / 2.0) * DX
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    radius = 1.0e-3
    aperture = (xx.square() + yy.square() <= radius * radius)
    phase = -math.pi * (xx.square() + yy.square()) / (WAVELENGTH * focal_length_m)
    amplitude = math.sqrt(power_transmission)
    transfer = torch.polar(torch.full_like(phase, amplitude), phase) * aperture
    return field * transfer.to(field.dtype)


def _route_to_decoder(field: torch.Tensor, route: str, grid: int, z_m: float = 0.0) -> tuple[torch.Tensor, dict[str, float], float]:
    area = DX * DX
    trace = {"fiber_output_mode_power_w": _power(field, area)}
    if route == "direct":
        propagated, out_z = _propagate(field, grid, z_m, 1.5e-3, "v9_direct_decoder_entrance")
        trace["after_direct_1p5mm_power_w"] = _power(propagated, area)
        active = _crop_active(propagated)
        trace["after_decoder_input_aperture_power_w"] = _power(active, area)
        return active, trace, out_z
    if route != "explicit_4f":
        raise ValueError(f"unknown output route: {route}")
    current, current_z = _propagate(field, grid, z_m, 1.0e-2, "v9_4f_lens1_input")
    current = _lens(current, grid, 1.0e-2, 0.98, "lens1")
    trace["after_lens1_power_w"] = _power(current, area)
    current, current_z = _propagate(current, grid, current_z, 1.5e-2, "v9_4f_lens2_input")
    current = _lens(current, grid, 5.0e-3, 0.98, "lens2")
    trace["after_lens2_power_w"] = _power(current, area)
    current, current_z = _propagate(current, grid, current_z, 5.0e-3, "v9_4f_image_plane")
    trace["after_4f_image_plane_power_w"] = _power(current, area)
    active = _crop_active(current)
    trace["after_decoder_input_aperture_power_w"] = _power(active, area)
    return active, trace, current_z


def _run_decoder(active: torch.Tensor, phase: torch.Tensor, grid: int, z_m: float) -> tuple[torch.Tensor, torch.Tensor, dict[str, float], float]:
    area = DX * DX
    root_t = math.sqrt(TRANSMISSION)
    current_z = z_m
    input_power = _power(active, area)
    screen0 = phase[0, 0].to(device=active.device, dtype=active.real.dtype)
    modulation0 = torch.polar(torch.full_like(screen0, root_t), screen0).to(active.dtype)
    padded = torch.nn.functional.pad(active * modulation0, ((grid - 100) // 2,) * 4)
    first_full, current_z = _propagate(padded, grid, current_z, 1.5e-3, "v9_decoder_layer1_output")
    first_crop = _crop_active(first_full)
    first_power = _power(first_crop, area)
    screen1 = phase[1, 0].to(device=active.device, dtype=active.real.dtype)
    modulation1 = torch.polar(torch.full_like(screen1, root_t), screen1).to(active.dtype)
    padded2 = torch.nn.functional.pad(first_crop * modulation1, ((grid - 100) // 2,) * 4)
    final_full, current_z = _propagate(padded2, grid, current_z, 1.5e-3, "v9_decoder_final_global_output")
    final_active = _crop_active(final_full)
    trace = {
        "decoder_input_power_w": input_power,
        "after_decoder_layer1_aperture_power_w": first_power,
        "after_decoder_layer2_global_power_w": TRANSMISSION * first_power,
        "after_decoder_layer2_output_aperture_power_w": _power(final_active, area),
    }
    return final_full, first_crop, trace, current_z


def _make_backend_relay(z_m: float, device: torch.device, dtype: torch.dtype) -> tuple[CoherentImagingRelay, PlaneSpec, PlaneSpec]:
    source = _plane("v9_decoder_exit", (100, 100), (DX, DX), z_m, "decoder_exit")
    target = _plane("v9_camera_sensor", (200, 200), (2.5e-6, 2.5e-6), z_m + 0.35, "camera")
    relay = CoherentImagingRelay(
        source, target, -2.5, name="v9_existing_backend_reference",
        segments_m=(0.1, 0.25), focal_lengths_m=(0.0714285714285714,),
        clear_aperture_m=0.0266,
    )
    if abs(relay.abcd.a - (-2.5)) > 1.0e-9 or abs(relay.abcd.b_m) > 1.0e-9:
        raise ValueError("the existing backend relay no longer matches its frozen B=0 geometry")
    return relay, source, target


def _gl_axis(order: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return torch.as_tensor(nodes, device=device, dtype=dtype), torch.as_tensor(weights, device=device, dtype=dtype)


def _sample_fourier_series(field: torch.Tensor, pitch_m: float, coordinates_y: torch.Tensor, coordinates_x: torch.Tensor) -> torch.Tensor:
    height, width = field.shape[-2:]
    fy = torch.fft.fftfreq(height, d=pitch_m, device=field.device, dtype=field.real.dtype)
    fx = torch.fft.fftfreq(width, d=pitch_m, device=field.device, dtype=field.real.dtype)
    y0 = -(height - 1.0) * pitch_m / 2.0
    x0 = -(width - 1.0) * pitch_m / 2.0
    py = torch.exp(2j * math.pi * (coordinates_y[:, None] - y0) * fy[None, :]).to(field.dtype)
    px = torch.exp(2j * math.pi * (coordinates_x[:, None] - x0) * fx[None, :]).to(field.dtype)
    spectrum = torch.fft.fft2(field.contiguous())
    intermediate = py @ spectrum
    return (intermediate @ px.T) / float(height * width)


def _camera_quadrature_field(camera_centers: torch.Tensor, order: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device, real_dtype = camera_centers.device, camera_centers.real.dtype
    nodes, weights = _gl_axis(order, device, real_dtype)
    pixel_pitch = 2.5e-6
    center = (torch.arange(200, device=device, dtype=real_dtype) - 99.5) * pixel_pitch
    sample_axis = (center[:, None] + nodes[None, :] * (pixel_pitch / 2.0)).reshape(-1)
    sampled = _sample_fourier_series(camera_centers, pixel_pitch, sample_axis, sample_axis)
    weighted_area = (pixel_pitch / 2.0) ** 2
    reshaped = sampled.abs().square().reshape(200, order, 200, order)
    pixel_power = (reshaped * weights[None, :, None, None] * weights[None, None, None, :]).sum(dim=(1, 3)) * weighted_area
    return sampled, pixel_power, weights


def _integrate_subsamples(sampled: torch.Tensor, order: int, weights: torch.Tensor) -> torch.Tensor:
    intensity = sampled.abs().square().reshape(200, order, 200, order)
    return (intensity * weights[None, :, None, None] * weights[None, None, None, :]).sum(dim=(1, 3)) * ((2.5e-6 / 2.0) ** 2)


def _decoder_exit_camera(active_exit: torch.Tensor, z_m: float) -> tuple[torch.Tensor, dict[str, float]]:
    relay, source, _target = _make_backend_relay(z_m, active_exit.device, active_exit.real.dtype)
    wrapped = ComplexField(active_exit[None, None], source, OperationProvenance("v9_decoder_output", source.plane_id, None))
    camera = relay(wrapped)
    camera_tensor = camera.tensor[0, 0]
    trace = {
        "backend_source_power_w": _power(active_exit, DX * DX),
        "after_backend_relay_camera_grid_power_w": _power(camera_tensor, 2.5e-6**2),
        "backend_source_pupil_retention": float(relay.last_diagnostic["source_pupil_retention"].reshape(-1)[0]),
        "backend_target_pupil_retention": float(relay.last_diagnostic["target_pupil_retention"].reshape(-1)[0]),
    }
    return camera_tensor, trace


def _window_metrics(reference: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    reference = reference.to(torch.complex128)
    test = test.to(torch.complex128)
    phase = torch.angle(torch.sum(reference.conj() * test))
    aligned = test * torch.exp(-1j * phase)
    reference_norm = torch.linalg.vector_norm(reference).clamp_min(EPS)
    raw = float(torch.linalg.vector_norm(reference - test) / reference_norm)
    aligned_error = float(torch.linalg.vector_norm(reference - aligned) / reference_norm)
    intensity_error = float(torch.linalg.vector_norm(reference.abs().square() - test.abs().square()) / torch.linalg.vector_norm(reference.abs().square()).clamp_min(EPS))
    p_ref = float(reference.abs().square().sum())
    p_test = float(test.abs().square().sum())
    return {
        "raw_complex_relative_l2": raw,
        "common_global_phase_aligned_complex_relative_l2": aligned_error,
        "common_global_phase_rad": float(phase),
        "intensity_relative_l2": intensity_error,
        "roi_power_reference": p_ref,
        "roi_power_test": p_test,
        "roi_power_relative_error": abs(p_ref - p_test) / max(abs(p_ref), EPS),
    }


def _gram(fields: torch.Tensor, area: float) -> torch.Tensor:
    flattened = fields.reshape(fields.shape[0], -1).to(torch.complex128)
    return flattened.conj() @ flattened.T * area


def _quadratic_power(coefficients: torch.Tensor, gram: torch.Tensor) -> float:
    c = coefficients.to(device=gram.device, dtype=gram.dtype)
    return float(torch.real(c.conj() @ (gram @ c)))


def _fiber_mode_gram(mode: str, size: int) -> torch.Tensor:
    # This basis is static and only produces a small channel-by-channel Gram
    # matrix. Construct it on CPU so the 2048x2048 complex128 profiles never
    # compete with 8192-grid complex128 propagation buffers for CUDA memory.
    device = torch.device("cpu")
    profiles: list[torch.Tensor] = []
    roi_plane = _plane("fiber_mode_roi", (ROI, ROI), (DX, DX), 0.0, "fiber_global")
    for cy, cx in _coordinates(size):
        if mode == "gaussian_s12":
            axis = (torch.arange(ROI, device=device, dtype=torch.float64) + 0.5) * DX - roi_plane.fov_m[0] / 2.0
            amplitude = (2.0 / (math.pi * WAIST**2)) ** 0.25
            one_y = amplitude * torch.exp(-((axis - cy) / WAIST).square())
            one_x = amplitude * torch.exp(-((axis - cx) / WAIST).square())
            profile = torch.outer(one_y, one_x).to(torch.complex128)
        else:
            axis_y = (torch.arange(ROI, device=device, dtype=torch.float64) + 0.5) * DX - roi_plane.fov_m[0] / 2.0
            axis_x = (torch.arange(ROI, device=device, dtype=torch.float64) + 0.5) * DX - roi_plane.fov_m[1] / 2.0
            y0 = int(torch.searchsorted(axis_y, torch.tensor(cy - PITCH / 2.0, device=device, dtype=torch.float64)))
            y1 = int(torch.searchsorted(axis_y, torch.tensor(cy + PITCH / 2.0, device=device, dtype=torch.float64)))
            x0 = int(torch.searchsorted(axis_x, torch.tensor(cx - PITCH / 2.0, device=device, dtype=torch.float64)))
            x1 = int(torch.searchsorted(axis_x, torch.tensor(cx + PITCH / 2.0, device=device, dtype=torch.float64)))
            if y0 >= y1 or x0 >= x1:
                raise ValueError("LP01 fiber support is unresolved on the common readout ROI")
            yy, xx = torch.meshgrid(axis_y[y0:y1], axis_x[x0:x1], indexing="ij")
            _, _, rho_e = lp01_eigenpair(2.2)
            core_radius = WAIST / rho_e
            local = lp01_radial_profile(torch.sqrt((yy - cy).square() + (xx - cx).square()) / core_radius, 2.2).to(torch.float64)
            norm = torch.sqrt(local.square().sum() * DX * DX)
            if not torch.isfinite(norm) or float(norm) <= 0.0:
                raise ValueError("LP01 modal normalization failed")
            full = torch.zeros((ROI, ROI), dtype=torch.complex128, device=device)
            full[y0:y1, x0:x1] = torch.complex(local / norm, torch.zeros_like(local))
            profile = full
        profiles.append(profile)
    return _gram(torch.stack(profiles), DX * DX)


def _matrix_prediction_check(candidate: dict[str, Any], mode: str) -> dict[str, Any]:
    commands = torch.from_numpy(candidate["commands"].astype(np.complex128))
    by_grid: dict[str, Any] = {}
    max_error = 0.0
    for grid in GRIDS:
        h = torch.from_numpy(candidate["matrices"][mode][grid]["H"])
        saved = torch.from_numpy(candidate["coefficients"][grid][mode])
        predicted = commands @ h.T
        errors = torch.linalg.vector_norm(predicted - saved, dim=1) / torch.linalg.vector_norm(saved, dim=1).clamp_min(EPS)
        worst = float(errors.max())
        max_error = max(max_error, worst)
        by_grid[str(grid)] = {"max_relative_complex_coefficient_error": worst, "all_50_finite": bool(torch.isfinite(errors).all())}
    return {"by_grid": by_grid, "maximum_relative_error": max_error, "passed": max_error <= 1.0e-4}


def _load_mode_basis(mode: str, candidate: dict[str, Any], grid: int, dtype: torch.dtype, device: torch.device) -> list[torch.Tensor]:
    result: list[torch.Tensor] = []
    for coordinate in _coordinates(candidate["size"]):
        field = _global_mode(mode, coordinate, grid, device)
        result.append(field.to(dtype=dtype))
        del field
    return result


def _basis_forward(mode_field: torch.Tensor, route: str, phase: torch.Tensor, grid: int, precision: str) -> dict[str, Any]:
    field = mode_field
    if precision == "complex64":
        field = field.to(torch.complex64)
        phase = phase.to(device=field.device, dtype=torch.float32)
    else:
        field = field.to(torch.complex128)
        phase = phase.to(device=field.device, dtype=torch.float64)
    input_active, route_trace, route_z = _route_to_decoder(field, route, grid)
    final_full, layer1_crop, decoder_trace, exit_z = _run_decoder(input_active, phase, grid, route_z)
    final_roi = _crop_roi(final_full).detach().contiguous()
    exit_active = _crop_active(final_full).detach().contiguous()
    camera_centers, camera_trace = _decoder_exit_camera(exit_active, exit_z)
    sampled4, _pixels4, weights4 = _camera_quadrature_field(camera_centers, 4)
    sampled8, _pixels8, weights8 = _camera_quadrature_field(camera_centers, 8)
    return {
        "decoder_input": input_active.detach(),
        "decoder_layer1": layer1_crop.detach(),
        "decoder_exit": exit_active,
        "final_roi": final_roi,
        "camera_centers": camera_centers.detach(),
        "camera_gl4": sampled4.detach(), "camera_gl8": sampled8.detach(),
        "gl4_weights": weights4.detach(), "gl8_weights": weights8.detach(),
        "route_trace": route_trace, "decoder_trace": decoder_trace,
        "camera_trace": camera_trace, "exit_z_m": exit_z,
        "global_output_power_from_layer": decoder_trace["after_decoder_layer2_global_power_w"],
    }


def _phase_low_frequency_negative_controls(phases: dict[str, torch.Tensor]) -> dict[str, bool]:
    low = phases["fixed_low_frequency"]
    mean = low.mean(dim=(-2, -1))
    rms = low.square().mean(dim=(-2, -1)).sqrt()
    return {
        "zero_phase_exact": bool(torch.count_nonzero(phases["zero_phase"]) == 0),
        "low_frequency_phase_demeaned": bool(mean.abs().max() < 1.0e-12),
        "low_frequency_phase_rms_0p5": bool(torch.max(torch.abs(rms - 0.5)) < 1.0e-10),
        "v8_amplitude_commands_not_renormalized": True,
    }


def _geometry_preflight(config: dict[str, Any], device: torch.device, phases: dict[str, torch.Tensor]) -> dict[str, Any]:
    front = path_abcd((0.010, 0.015, 0.005), (0.010, 0.005))
    back = path_abcd((0.100, 0.250), (0.0714285714285714,))
    gl4 = np.polynomial.legendre.leggauss(4)
    gl8 = np.polynomial.legendre.leggauss(8)
    constant_pixel_area = (2.5e-6 / 2.0) ** 2
    integrated4 = float(gl4[1].sum() ** 2 * constant_pixel_area)
    integrated8 = float(gl8[1].sum() ** 2 * constant_pixel_area)
    expected_area = 2.5e-6**2
    area_error = abs(integrated4 - expected_area) / expected_area
    area8_error = abs(integrated8 - expected_area) / expected_area
    photon_energy = H_PLANCK * C_LIGHT / WAVELENGTH
    known_photons = 0.5 * 1.0e-3 / photon_energy
    photon_pass = abs(known_photons * photon_energy - 0.5e-3) < 1.0e-15
    phase_checks = _phase_low_frequency_negative_controls(phases)
    constant_field = torch.ones((200, 200), dtype=torch.complex128, device=device)
    _constant_samples, constant_pixels, _constant_weights = _camera_quadrature_field(constant_field, 4)
    expected_pixel = 2.5e-6**2
    constant_pixel_error = float(torch.max(torch.abs(constant_pixels - expected_pixel)) / expected_pixel)
    omitted_area_error = abs(float(gl4[1].sum() ** 2) - expected_area) / expected_area
    lens_input = torch.ones((100, 100), dtype=torch.complex128, device=device)
    lens_output = _lens(lens_input, 100, 1.0e-2, 0.98, "v9_preflight_lens")
    lens_ratio = _power(lens_output, DX * DX) / _power(lens_input, DX * DX)
    wrong_amplitude_ratio = 0.98**2
    lens_power_pass = abs(lens_ratio - 0.98) <= 1.0e-12

    imaging_source = _plane("v9_imaging_preflight_source", (100, 100), (DX, DX), 0.0, "imaging_source")
    imaging_target = _plane("v9_imaging_preflight_target", (200, 200), (2.5e-6, 2.5e-6), 0.35, "imaging_target")
    imaging_relay = CoherentImagingRelay(
        imaging_source, imaging_target, -2.5, name="v9_gaussian_imaging_preflight",
        segments_m=(0.1, 0.25), focal_lengths_m=(0.0714285714285714,), clear_aperture_m=0.0266,
    )
    source_axis = (torch.arange(100, device=device, dtype=torch.float64) - 49.5) * DX
    source_y, source_x = torch.meshgrid(source_axis, source_axis, indexing="ij")
    center_x, center_y, sigma = 30.0e-6, -20.0e-6, 16.0e-6
    analytic_gaussian = torch.exp(-0.5 * (((source_x - center_x) / sigma) ** 2 + ((source_y - center_y) / sigma) ** 2)).to(torch.complex128)
    imaged_batch = imaging_relay(ComplexField(analytic_gaussian[None, None], imaging_source,
                                               OperationProvenance("v9_preflight_gaussian", imaging_source.plane_id, None))).tensor
    imaged = imaged_batch[0, 0]
    target_axis_x = (torch.arange(200, device=device, dtype=torch.float64) - 99.5) * 2.5e-6
    target_axis_y = target_axis_x
    imaged_power = imaged.abs().square()
    image_sum = imaged_power.sum().clamp_min(EPS)
    centroid_x = float((imaged_power.sum(dim=0) * target_axis_x).sum() / image_sum)
    centroid_y = float((imaged_power.sum(dim=1) * target_axis_y).sum() / image_sum)
    expected_centroid = (-2.5 * center_x, -2.5 * center_y)
    centroid_error_m = math.hypot(centroid_x - expected_centroid[0], centroid_y - expected_centroid[1])
    relay_mapping_pass = centroid_error_m <= 2.5e-6
    front_ok = abs(front.a - (-0.5)) < 1e-9 and abs(front.b_m) < 1e-9 and abs(front.determinant - 1.0) < 1e-9
    back_ok = abs(back.a - (-2.5)) < 1e-9 and abs(back.b_m) < 1e-9 and abs(back.determinant - 1.0) < 1e-9
    status = (
        all(phase_checks.values()) and front_ok and back_ok and area_error < 1e-12
        and area8_error < 1e-12 and photon_pass and constant_pixel_error <= 1.0e-12
        and lens_power_pass and relay_mapping_pass
    )
    return {
        "explicit_front_4f": {"ABCD": front.to_dict(), "segments_m": [0.010, 0.015, 0.005], "physical_lens_aperture_diameter_m": 0.002, "geometry_pass": front_ok},
        "existing_backend_relay": {"ABCD": back.to_dict(), "segments_m": [0.1, 0.25], "camera_coordinates_inverted": True, "fiber_ids_preserved": True, "geometry_pass": back_ok},
        "camera_pixel_area": {"expected_m2": expected_area, "gl4_error": area_error, "gl8_error": area8_error,
                              "constant_field_max_relative_pixel_error": constant_pixel_error,
                              "omitted_area_negative_control_error_ratio": omitted_area_error,
                              "omitting_area_would_be_rejected": omitted_area_error > 1.0e6},
        "photon_unit_check": {"photon_energy_j": photon_energy, "known_photon_conversion_pass": photon_pass},
        "lens_power_transmission": {"configured_power_transmission": 0.98, "measured_power_ratio": lens_ratio,
                                    "wrong_amplitude_as_power_ratio": wrong_amplitude_ratio,
                                    "wrong_amplitude_negative_control_detected": abs(lens_ratio - wrong_amplitude_ratio) > 0.01,
                                    "passed": lens_power_pass},
        "gaussian_imaging_coordinate_control": {
            "magnification": -2.5, "source_center_xy_m": [center_x, center_y],
            "expected_image_center_xy_m": list(expected_centroid), "measured_image_center_xy_m": [centroid_x, centroid_y],
            "centroid_error_m": centroid_error_m, "one_camera_pixel_tolerance_m": 2.5e-6,
            "coordinate_inversion_and_magnification_pass": relay_mapping_pass,
        },
        "decoder_phase_controls": phase_checks,
        "negative_controls": {"transmission_amplitude_as_power_rejected": lens_power_pass and abs(lens_ratio - wrong_amplitude_ratio) > 0.01,
                              "omitted_pixel_area_rejected": omitted_area_error > 1.0e6,
                              "per_sample_renormalization_forbidden": True},
        "passed": status,
    }


def _preflight(config: dict[str, Any], config_path: Path, run_dir: Path, root: Path, source_override: str | None) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    source = _load_source(config, root, source_override)
    pilot_replay_checks = {
        candidate_name: _pilot_pattern_indices(source, candidate_name, candidate)
        for candidate_name, candidate in source["candidates"].items()
    }
    phases = _new_decoder_phases(config["decoder"]["low_frequency_seed"], device)
    _write_decoder_phase_files(run_dir, phases)
    geometry = _geometry_preflight(config, device, phases)
    matrix_checks = {
        f"{candidate}/{mode}": _matrix_prediction_check(value, mode)
        for candidate, value in source["candidates"].items() for mode in MODES
    }
    matrix_ok = all(item["passed"] for item in matrix_checks.values())
    gram_control = _fiber_mode_gram("gaussian_s12", 1)
    gram_cpu_pass = gram_control.device.type == "cpu" and abs(float(gram_control[0, 0].real) - 1.0) <= 1.0e-4
    actual: dict[str, Any] = {}
    nearfield_crop: dict[str, Any] = {}
    negative_controls: dict[str, Any] = {}
    if device.type == "cuda":
        candidate = source["candidates"]["candidate_2x2_C"]
        mode = "gaussian_s12"
        grid = 4096
        dtype = torch.complex64
        pattern_index = _command_uniform_index(candidate)
        coeff = torch.from_numpy(candidate["coefficients"][grid][mode][pattern_index]).to(device=device, dtype=dtype)
        low_phase = phases["zero_phase"]
        response: list[dict[str, Any]] = []
        source_modes = _load_mode_basis(mode, candidate, grid, dtype, device)
        for candidate_name, candidate_data in source["candidates"].items():
            for mode_name in MODES:
                channel_retention: list[float] = []
                for mode_field in _load_mode_basis(mode_name, candidate_data, 4096, torch.complex64, device):
                    active_crop = _crop_active(mode_field)
                    channel_retention.append(_power(active_crop, DX * DX) / max(_power(mode_field, DX * DX), EPS))
                    del active_crop, mode_field
                nearfield_crop[f"{candidate_name}/{mode_name}"] = {
                    "per_channel_zero_distance_central_200um_retention": channel_retention,
                    "mean_retention": float(np.mean(channel_retention)),
                    "minimum_retention": float(np.min(channel_retention)),
                    "interpretation": "zero-distance geometric control only; excluded from either physical route result",
                }
        permutation = torch.arange(len(source_modes) - 1, -1, -1, device=device)
        probe_coefficients = torch.complex(
            torch.arange(1, len(source_modes) + 1, device=device, dtype=torch.float32),
            torch.arange(len(source_modes), 0, -1, device=device, dtype=torch.float32),
        )
        correct_order_field = torch.einsum("n,nhw->hw", probe_coefficients, torch.stack(source_modes))
        swapped_order_field = torch.einsum("n,nhw->hw", probe_coefficients, torch.stack(source_modes)[permutation])
        port_swap_error = float(torch.linalg.vector_norm(correct_order_field - swapped_order_field) / torch.linalg.vector_norm(correct_order_field).clamp_min(EPS))
        nonuniform_index = next(index for index, row in enumerate(candidate["commands"]) if not np.allclose(row, row.max(), rtol=0.0, atol=1.0e-7))
        command = candidate["commands"][nonuniform_index].astype(np.float64)
        normalized_command = command / float(command.max())
        h = candidate["matrices"][mode][grid]["H"]
        command_identity_error = float(np.linalg.norm(h @ command - h @ normalized_command) / max(np.linalg.norm(h @ command), EPS))
        negative_controls = {
            "wrong_port_order_relative_field_error": port_swap_error,
            "wrong_port_order_rejected": port_swap_error > 1.0e-4,
            "per_sample_max_renormalization_relative_coefficient_error": command_identity_error,
            "per_sample_renormalization_rejected": command_identity_error > 1.0e-4,
        }
        jacobian_probe: dict[str, Any] | None = None
        for route in ROUTES:
            basis_outputs: list[dict[str, Any]] = []
            for basis in source_modes:
                result = _basis_forward(basis, route, low_phase, grid, "complex64")
                basis_outputs.append(result)
            source_field = torch.einsum("n,nhw->hw", coeff, torch.stack(source_modes))
            direct_active, route_trace, z_route = _route_to_decoder(source_field, route, grid)
            direct_full, _first, _dtrace, z_exit = _run_decoder(direct_active, low_phase, grid, z_route)
            predicted_roi = torch.einsum("n,nhw->hw", coeff, torch.stack([item["final_roi"].to(device) for item in basis_outputs]))
            direct_roi = _crop_roi(direct_full)
            err = float(torch.linalg.vector_norm(predicted_roi.to(torch.complex128) - direct_roi.to(torch.complex128)) / torch.linalg.vector_norm(direct_roi.to(torch.complex128)).clamp_min(EPS))
            final_active = _crop_active(direct_full)
            camera, camera_trace = _decoder_exit_camera(final_active, z_exit)
            gl4, pixels4, w4 = _camera_quadrature_field(camera, 4)
            gl8, pixels8, w8 = _camera_quadrature_field(camera, 8)
            integration_err = float(torch.linalg.vector_norm(pixels4 - pixels8) / torch.linalg.vector_norm(pixels8).clamp_min(EPS))
            output_power = _power(source_field, DX * DX)
            route_powers = [value for key, value in route_trace.items() if key.endswith("_power_w")]
            decoder_powers = [_dtrace["decoder_input_power_w"], _dtrace["after_decoder_layer1_aperture_power_w"],
                              _dtrace["after_decoder_layer2_global_power_w"], _dtrace["after_decoder_layer2_output_aperture_power_w"]]
            all_chain_powers = [output_power, *route_powers, *decoder_powers, camera_trace["after_backend_relay_camera_grid_power_w"]]
            passive = all(right <= left * (1.0 + float(config["gates"]["passivity_relative_tolerance"])) + EPS
                          for left, right in zip(all_chain_powers, all_chain_powers[1:], strict=False))
            global_phase = torch.exp(torch.tensor(0.731, device=device, dtype=torch.float64) * 1j).to(coeff.dtype)
            phase_rotated_pixels = _pixel_powers_from_response(torch.stack([item["camera_gl4"].to(device) for item in basis_outputs]),
                                                                coeff * global_phase, 4, w4)
            global_phase_intensity_error = float(torch.linalg.vector_norm(phase_rotated_pixels.to(torch.float64) - pixels4.to(torch.float64)) / torch.linalg.vector_norm(pixels4.to(torch.float64)).clamp_min(EPS))
            if route == "direct":
                responses = torch.stack([item["camera_gl4"].to(device) for item in basis_outputs])
                h_tensor = torch.from_numpy(h).to(device=device, dtype=torch.complex128)
                base_command = torch.full((10000,), 0.5, device=device, dtype=torch.float64)
                c0 = h_tensor @ base_command.to(torch.complex128)
                a_factor, b_factor, _field0 = _camera_linear_factors(responses.to(torch.complex128), w4.to(torch.float64), h_tensor, c0)
                direction = torch.linspace(-1.0, 1.0, 10000, device=device, dtype=torch.float64)
                direction = direction / torch.linalg.vector_norm(direction).clamp_min(EPS)
                eps = 1.0e-4
                dp_analytic = a_factor @ (b_factor @ direction)
                plus = _pixel_powers_from_response(responses.to(torch.complex128), h_tensor @ (base_command + eps * direction).to(torch.complex128), 4, w4.to(torch.float64)).reshape(-1)
                minus = _pixel_powers_from_response(responses.to(torch.complex128), h_tensor @ (base_command - eps * direction).to(torch.complex128), 4, w4.to(torch.float64)).reshape(-1)
                dp_fd = (plus - minus) / (2.0 * eps)
                jac_abs = float(torch.linalg.vector_norm(dp_fd - dp_analytic))
                jac_ref = float(torch.linalg.vector_norm(dp_analytic))
                jac_tol = float(config["gates"]["complex128_jacobian_relative"]) * jac_ref + float(config["gates"]["complex128_jacobian_absolute_w_per_command"]) * SOURCE_POWER
                jacobian_probe = {"relative_error": jac_abs / max(jac_ref, EPS), "absolute_error": jac_abs, "tolerance": jac_tol,
                                  "epsilon": eps, "passed": jac_abs <= jac_tol}
                del responses, h_tensor, base_command, c0, a_factor, b_factor, direction, dp_analytic, plus, minus, dp_fd
            response.append({"route": route, "superposition_relative_error": err, "pixel_gl4_gl8_relative_l2": integration_err,
                             "field_finite": bool(torch.isfinite(direct_roi).all() and torch.isfinite(pixels4).all()),
                             "route_power_trace_for_uniform_t0p5_w": route_trace,
                             "decoder_power_trace_for_uniform_t0p5_w": _dtrace,
                             "camera_power_trace_for_uniform_t0p5_w": camera_trace,
                             "power_chain_passive": passive,
                             "global_common_phase_intensity_relative_error": global_phase_intensity_error,
                             "direct_output_power_w": _power(camera, 2.5e-6**2)})
            del basis_outputs, source_field, direct_active, direct_full, predicted_roi, camera, gl4, gl8, pixels4, pixels8, source_modes
            gc.collect()
            torch.cuda.empty_cache()
            source_modes = _load_mode_basis(mode, candidate, grid, dtype, device)
        actual["representative_4096_complex64"] = response
        actual["camera_jacobian_directional_preflight"] = jacobian_probe
        actual["nearfield_crop_control"] = nearfield_crop
    else:
        actual["status"] = "CUDA_UNAVAILABLE"
    relay_preflight_pass = all(
        item.get("superposition_relative_error", 1.0) <= 1.0e-4
        and item.get("pixel_gl4_gl8_relative_l2", 1.0) <= 1.0e-4
        and item.get("power_chain_passive", False)
        and item.get("global_common_phase_intensity_relative_error", 1.0) <= 1.0e-5
        for item in actual.get("representative_4096_complex64", [])
    )
    jacobian_preflight_pass = bool(actual.get("camera_jacobian_directional_preflight", {}).get("passed", False))
    checks = {
        "source_evidence_loaded": True,
        "v8_worst_probe_identity_replayed": all(len(indices) == 2 for indices in pilot_replay_checks.values()),
        "full_measurement_consumer_available": callable(globals().get("_run_context_precision")),
        "fixed_geometry_and_unit_controls": geometry["passed"],
        "v8_matrix_replay": matrix_ok,
        "analytic_modal_gram_cpu_reference": gram_cpu_pass,
        "representative_production_forward": actual.get("status") != "CUDA_UNAVAILABLE" and bool(actual.get("representative_4096_complex64")),
        "complex_superposition_and_pixel_integral": relay_preflight_pass,
        "camera_jacobian_complex128_directional_check": jacobian_preflight_pass,
        "negative_controls_reject_wrong_port_order_and_rescaling": all(negative_controls.get(key, False) for key in ("wrong_port_order_rejected", "per_sample_renormalization_rejected")),
    }
    result = {
        "schema": f"{SCHEMA}.preflight", "status": "PASS" if all(checks.values()) else "FAIL_ENGINEERING_PREFLIGHT",
        "passed": all(checks.values()), "config_path": str(config_path), "source_run_dir": source["run_dir"],
        "checks": checks, "v8_pilot_pattern_indices": pilot_replay_checks,
        "geometry_controls": geometry, "matrix_prediction": matrix_checks,
        "representative_forward": actual, "negative_controls": negative_controls,
        "forbidden_activity": {"formal_training_updates": 0, "phase_optimization_updates": 0, "dataset_reads": 0, "task_evaluator_calls": 0, "fid_calls": 0, "loss_changes": 0, "material_forward_calls": 0},
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "elapsed_seconds": time.time() - _START,
    }
    _write_json(run_dir / "preflight.json", result)
    return result


def _directional_spectrum(a: torch.Tensor, b: torch.Tensor, thresholds: list[float]) -> dict[str, Any]:
    # J = A B, with A [pixels, rank] and B [rank, commands]. Compute its exact
    # nonzero spectrum through compact QR factors, never materializing J.
    qa, ra = torch.linalg.qr(a, mode="reduced")
    qb, rb = torch.linalg.qr(b.T, mode="reduced")
    small = ra @ rb.T
    u, singular, vh = torch.linalg.svd(small, full_matrices=False)
    right = qb @ vh.T
    maximum = float(singular.max()) if singular.numel() else 0.0
    return {
        "singular_values": singular.detach().cpu().tolist(),
        "maximum_singular_value": maximum,
        "effective_dimension_by_relative_threshold": {str(t): int((singular >= maximum * t).sum()) if maximum > 0 else 0 for t in thresholds},
        "right_singular_directions": right.detach().cpu(),
    }


def _camera_linear_factors(camera_response_gl4: torch.Tensor, weights: torch.Tensor, h: torch.Tensor, base_coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # camera_response_gl4 [channels, 800, 800], H [channels, 10000]
    response = camera_response_gl4.to(torch.complex128)
    c0 = base_coefficients.to(device=response.device, dtype=torch.complex128)
    field = torch.einsum("n,nyx->yx", c0, response)
    weights = weights.to(device=response.device, dtype=torch.float64)
    weighted_nodes = weights[:, None] * weights[None, :]
    # 4 GL nodes per pixel; reshape interleaves each pixel's sub-samples.
    q_nodes = 2.0 * field.conj()[None, ...] * response
    q_pixel = (q_nodes.reshape(response.shape[0], 200, 4, 200, 4) * weighted_nodes[None, None, :, None, :]).sum(dim=(2, 4)) * ((2.5e-6 / 2.0) ** 2)
    q_pixel = q_pixel.permute(1, 2, 0).reshape(40000, response.shape[0])
    a = torch.cat((q_pixel.real, -q_pixel.imag), dim=1)
    h = h.to(device=response.device, dtype=torch.complex128)
    b = torch.cat((h.real, h.imag), dim=0)
    return a, b, field


def _pixel_powers_from_response(response_gl: torch.Tensor, coefficients: torch.Tensor, order: int, weights: torch.Tensor) -> torch.Tensor:
    field = torch.einsum("n,nyx->yx", coefficients.to(response_gl.dtype), response_gl)
    intensity = field.abs().square().reshape(200, order, 200, order)
    w = weights.to(device=field.device, dtype=field.real.dtype)
    return (intensity * w[None, :, None, None] * w[None, None, None, :]).sum(dim=(1, 3)) * ((2.5e-6 / 2.0) ** 2)


def _photons(power: torch.Tensor, exposure_s: float, qe: float) -> torch.Tensor:
    return power * (qe * exposure_s / (H_PLANCK * C_LIGHT / WAVELENGTH))


def _power_summary(values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().to(torch.float64).reshape(-1)
    quantiles = torch.quantile(flat, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], device=flat.device, dtype=flat.dtype))
    return {"min": float(quantiles[0]), "p10": float(quantiles[1]), "p50": float(quantiles[2]), "p90": float(quantiles[3]), "max": float(quantiles[4]), "mean": float(flat.mean())}


def _dB_ratio(output_w: float, input_w: float) -> float | None:
    if output_w <= 0.0 or input_w <= 0.0:
        return None
    return 10.0 * math.log10(output_w / input_w)


def _context_key(candidate: str, mode: str, route: str, decoder: str) -> str:
    return f"{candidate}__{mode}__{route}__{decoder}"


def _save_context_grid(run_dir: Path, key: str, grid: int, precision: str, payload: dict[str, Any]) -> None:
    path = run_dir / "context_fields" / f"{key}__{grid}_{precision}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: np.asarray(value) for name, value in payload.items() if isinstance(value, (np.ndarray, list, tuple))}
    arrays["metadata_json"] = np.asarray(json.dumps({"context": key, "grid": grid, "precision": precision, "schema": SCHEMA}))
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temp.replace(path)


def _stage_grams(outputs: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    device = outputs[0]["decoder_input"].device
    dtype = outputs[0]["decoder_input"].dtype
    input_fields = torch.stack([item["decoder_input"].to(device=device, dtype=dtype) for item in outputs])
    layer1_fields = torch.stack([item["decoder_layer1"].to(device=device, dtype=dtype) for item in outputs])
    exit_fields = torch.stack([item["decoder_exit"].to(device=device, dtype=dtype) for item in outputs])
    return {
        "decoder_input": _gram(input_fields, DX * DX),
        "decoder_layer1": _gram(layer1_fields, DX * DX),
        "decoder_exit": _gram(exit_fields, DX * DX),
    }


def _camera_power_for_coefficients(response: torch.Tensor, coefficients: torch.Tensor, order: int, weights: torch.Tensor) -> torch.Tensor:
    return _pixel_powers_from_response(response, coefficients, order, weights)


def _command_uniform_index(candidate: dict[str, Any]) -> int:
    for index, command in enumerate(candidate["commands"]):
        if np.allclose(command, 0.5, rtol=0.0, atol=1.0e-7):
            return index
    raise ValueError("v8 command archive is missing the required uniform t=0.5 control")


def _quadrature_gate(power4: torch.Tensor, power8: torch.Tensor) -> dict[str, float | bool]:
    relative = float(torch.linalg.vector_norm(power4 - power8) / torch.linalg.vector_norm(power8).clamp_min(EPS))
    absolute_fraction = float(torch.max(torch.abs(power4 - power8)) / torch.sum(power8).clamp_min(EPS))
    return {"relative_l2_error": relative, "max_pixel_error_over_total_power": absolute_fraction, "passed": relative <= 1.0e-4}


def _exposure_summary(power: torch.Tensor, exposure_s: float, config: dict[str, Any]) -> dict[str, Any]:
    camera = config["camera"]
    power_summary = _power_summary(power)
    photon_factor = float(camera["quantum_efficiency"] * exposure_s / (H_PLANCK * C_LIGHT / WAVELENGTH))
    electron_summary = {key: value * photon_factor for key, value in power_summary.items()}
    full_well = float(camera["full_well_e"])
    task_summary = {key: min(1.0, max(0.0, value / full_well)) for key, value in electron_summary.items()}
    mu = power * photon_factor
    return {
        "exposure_s": exposure_s,
        "expected_electrons_total": float(mu.sum()),
        "expected_electrons_per_pixel": electron_summary,
        "task_output_per_pixel": task_summary,
        "task_output_mean": float((mu / full_well).clamp(0.0, 1.0).mean()),
        "saturated_pixel_fraction": float((mu >= full_well).to(torch.float64).mean()),
        "camera_gain_proxy_w_inverse": photon_factor / full_well,
    }


def _small_factor_spectrum(a: torch.Tensor, b: torch.Tensor, thresholds: list[float]) -> tuple[dict[str, Any], torch.Tensor]:
    record = _directional_spectrum(a, b, thresholds)
    directions = record.pop("right_singular_directions")
    return record, directions


def _camera_jacobian_report(
    config: dict[str, Any],
    run_dir: Path,
    candidate: dict[str, Any],
    candidate_name: str,
    mode: str,
    route: str,
    decoder_state: str,
    grid: int,
    response_gl4: torch.Tensor,
    weights4: torch.Tensor,
    h_np: np.ndarray,
    source_power_by_pattern: np.ndarray,
) -> dict[str, Any]:
    device = response_gl4.device
    h = torch.from_numpy(h_np).to(device=device, dtype=torch.complex128)
    t0 = torch.full((10000,), 0.5, device=device, dtype=torch.float64)
    c0 = h @ t0.to(torch.complex128)
    a, b, field0 = _camera_linear_factors(response_gl4, weights4, h, c0)
    thresholds = [float(v) for v in config["measurement"]["relative_svd_thresholds"]]
    raw_spectrum, raw_directions = _small_factor_spectrum(a, b, thresholds)
    base_power = _pixel_powers_from_response(response_gl4, c0, 4, weights4).to(torch.float64)
    exposure_records: dict[str, Any] = {}
    direction_bank: dict[str, Any] = {}
    read_noise = float(config["camera"]["read_noise_e_rms"])
    dark_rate = float(config["camera"]["dark_current_e_per_pixel_s"])
    bounds = tuple(float(v) for v in config["measurement"]["command_bounds"])
    rms_target = float(config["measurement"]["input_perturbation_rms"])
    for exposure in EXPOSURES:
        mu0 = _photons(base_power, exposure, float(config["camera"]["quantum_efficiency"]))
        pair_variance = 2.0 * (mu0 + dark_rate * exposure + read_noise**2)
        row_weight = pair_variance.clamp_min(EPS).rsqrt()
        electron_scale = float(config["camera"]["quantum_efficiency"] * exposure / (H_PLANCK * C_LIGHT / WAVELENGTH))
        noise_spectrum, directions = _small_factor_spectrum(
            a * (electron_scale * row_weight.reshape(-1, 1)), b, thresholds
        )
        kept = int(noise_spectrum["effective_dimension_by_relative_threshold"]["0.001"])
        direction_records: list[dict[str, Any]] = []
        for index in range(directions.shape[1]):
            direction = directions[:, index].to(device=device, dtype=torch.float64)
            if float(torch.linalg.vector_norm(direction)) <= 0.0:
                continue
            requested = direction * (rms_target * math.sqrt(direction.numel()) / torch.linalg.vector_norm(direction))
            positive_max = float(requested.clamp_min(0).max())
            negative_max = float((-requested).clamp_min(0).max())
            positive_limit = (bounds[1] - 0.5) / positive_max if positive_max > 0.0 else math.inf
            negative_limit = (0.5 - bounds[0]) / negative_max if negative_max > 0.0 else math.inf
            scale = min(1.0, positive_limit, negative_limit)
            delta = requested * scale
            delta_coeff = h @ delta.to(torch.complex128)
            delta_field = torch.einsum("n,nyx->yx", delta_coeff, response_gl4.to(torch.complex128))
            perturbed_field = field0 + delta_field
            p1 = _pixel_powers_from_response(response_gl4, c0 + delta_coeff, 4, weights4).to(torch.float64)
            predicted_dp = (a @ (b @ delta)).to(torch.float64)
            actual_dp = p1.reshape(-1) - base_power.reshape(-1)
            mu1 = _photons(p1, exposure, float(config["camera"]["quantum_efficiency"]))
            noise_std = (mu0 + mu1 + 2.0 * (dark_rate * exposure + read_noise**2)).clamp_min(EPS).sqrt().reshape(-1)
            actual_snr = float(torch.linalg.vector_norm((mu1 - mu0).reshape(-1) / noise_std))
            predicted_snr = float(torch.linalg.vector_norm((predicted_dp.reshape(-1) * electron_scale) / noise_std))
            residual = float(torch.linalg.vector_norm(actual_dp - predicted_dp) / torch.linalg.vector_norm(actual_dp).clamp_min(EPS))
            direction_records.append({
                "direction_index": index,
                "relative_singular_value": float(noise_spectrum["singular_values"][index] / max(noise_spectrum["maximum_singular_value"], EPS)),
                "requested_rms": rms_target,
                "applied_rms": float(delta.square().mean().sqrt()),
                "box_feasible_scale": scale,
                "command_min": float((t0 + delta).min()),
                "command_max": float((t0 + delta).max()),
                "predicted_snr": predicted_snr,
                "actual_difference_snr": actual_snr,
                "linearized_pixel_power_relative_error": residual,
                "pixel_power_change_l2_w": float(torch.linalg.vector_norm(actual_dp)),
                "saturated_pixels_after_perturbation": int((mu1 >= float(config["camera"]["full_well_e"])).sum()),
                "full_field_finite": bool(torch.isfinite(perturbed_field).all()),
            })
            del requested, delta, delta_coeff, delta_field, perturbed_field, p1, mu1, predicted_dp, actual_dp
        exposure_records[str(exposure)] = {
            "noise_whitened_spectrum": noise_spectrum,
            "directions_above_relative_1e_3": kept,
            "snr_direction_counts": {str(threshold): sum(item["actual_difference_snr"] >= threshold for item in direction_records[:kept]) for threshold in (1, 3, 5)},
            "bounded_direction_records": direction_records,
        }
        direction_bank[str(exposure)] = directions.detach().cpu()
    # Complex128 directional central-difference check for the actual composed
    # camera response. The finite-difference and analytic paths are evaluated
    # independently from +/- command controls using the frozen H and camera basis.
    direction = raw_directions[:, 0].to(device=device, dtype=torch.float64)
    epsilon = 1.0e-4
    plus_coeff = h @ (t0 + epsilon * direction).to(torch.complex128)
    minus_coeff = h @ (t0 - epsilon * direction).to(torch.complex128)
    plus_power = _pixel_powers_from_response(response_gl4, plus_coeff, 4, weights4).to(torch.float64).reshape(-1)
    minus_power = _pixel_powers_from_response(response_gl4, minus_coeff, 4, weights4).to(torch.float64).reshape(-1)
    finite_difference = (plus_power - minus_power) / (2.0 * epsilon)
    analytic = a @ (b @ direction)
    difference = float(torch.linalg.vector_norm(finite_difference - analytic))
    reference_norm = float(torch.linalg.vector_norm(analytic))
    absolute_tolerance = float(config["gates"]["complex128_jacobian_absolute_w_per_command"]) * float(source_power_by_pattern[_command_uniform_index(candidate)])
    jacobian_gate = difference <= float(config["gates"]["complex128_jacobian_relative"]) * reference_norm + absolute_tolerance
    factors_path = run_dir / "camera_jacobians" / f"{_context_key(candidate_name, mode, route, decoder_state)}_{grid}_complex128_factors.npz"
    factors_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(factors_path, A_pixel_complex_gradient=a.detach().cpu().numpy(), B_command_real_control=b.detach().cpu().numpy(),
                        right_directions=raw_directions.detach().cpu().numpy(), metadata_json=np.asarray(json.dumps({"grid": grid, "mode": mode, "units": "W per command", "factorized_no_full_J": True})))
    return {
        "base_command": 0.5,
        "factorization": {"A_shape": list(a.shape), "B_shape": list(b.shape), "full_40000_by_10000_matrix_allocated": False, "units": "J_P,t = A B, W per command"},
        "raw_power_jacobian_spectrum": raw_spectrum,
        "exposure_sensitivity": exposure_records,
        "complex128_central_difference": {"relative_error": difference / max(reference_norm, EPS), "absolute_error": difference,
            "reference_norm": reference_norm, "absolute_tolerance": absolute_tolerance, "passed": jacobian_gate, "epsilon": epsilon},
        "factor_archive": str(factors_path),
        "right_directions": raw_directions.detach().cpu(),
        "noise_right_directions_1ms": direction_bank[str(config["camera"]["primary_exposure_s"])],
    }


def _full_pattern_forward(
    mode: str, candidate: dict[str, Any], coefficients: torch.Tensor, route: str,
    decoder_phase: torch.Tensor, grid: int, precision: str, device: torch.device,
) -> dict[str, Any]:
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    coefficients = coefficients.to(device=device, dtype=dtype)
    field = torch.zeros((grid, grid), device=device, dtype=dtype)
    for index, coordinate in enumerate(_coordinates(candidate["size"])):
        mode_field = _global_mode(mode, coordinate, grid, device).to(dtype)
        field.add_(mode_field * coefficients[index])
        del mode_field
    active, route_trace, route_z = _route_to_decoder(field, route, grid)
    final_full, first, decoder_trace, exit_z = _run_decoder(active, decoder_phase, grid, route_z)
    camera, camera_trace = _decoder_exit_camera(_crop_active(final_full), exit_z)
    sampled4, pixel4, weights4 = _camera_quadrature_field(camera, 4)
    sampled8, pixel8, weights8 = _camera_quadrature_field(camera, 8)
    return {
        "source_field": field, "route_input": active, "layer1": first, "final_full": final_full,
        "camera_centers": camera, "camera_gl4": sampled4, "camera_gl8": sampled8,
        "pixel_power_gl4": pixel4, "pixel_power_gl8": pixel8,
        "weights4": weights4, "weights8": weights8,
        "route_trace": route_trace, "decoder_trace": decoder_trace,
        "camera_trace": camera_trace, "exit_z_m": exit_z,
    }


def _measure_pilot_context(
    config: dict[str, Any], run_dir: Path, candidate_name: str, candidate: dict[str, Any],
    mode: str, route: str, decoder_state: str, decoder_phase: torch.Tensor,
    precision: str, device: torch.device, pattern_indices: list[int],
) -> dict[str, Any]:
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    gate = float(config["gates"]["aligned_common_roi_complex64_max"])
    pattern_records: list[dict[str, Any]] = []
    roi_fields: dict[int, dict[int, torch.Tensor]] = {index: {} for index in pattern_indices}
    all_quadrature_pass = True
    all_finite = True
    for index in pattern_indices:
        per_grid: dict[str, Any] = {}
        for grid in GRIDS:
            clear_production_transfer_cache()
            coefficients = torch.from_numpy(candidate["coefficients"][grid][mode][index]).to(device=device, dtype=dtype)
            output = _full_pattern_forward(mode, candidate, coefficients, route, decoder_phase, grid, precision, device)
            roi = _crop_roi(output["final_full"]).detach().contiguous()
            pixel4 = output["pixel_power_gl4"].to(torch.float64)
            pixel8 = output["pixel_power_gl8"].to(torch.float64)
            quadrature = _quadrature_gate(pixel4, pixel8)
            all_quadrature_pass &= bool(quadrature["passed"])
            finite = bool(torch.isfinite(roi).all() and torch.isfinite(pixel4).all() and torch.isfinite(pixel8).all())
            all_finite &= finite
            roi_fields[index][grid] = roi.detach().cpu()
            source_power = float(candidate["power_traces"][grid]["source_input_power_w"][index])
            channel_power = np.abs(candidate["coefficients"][grid][mode][index]) ** 2
            per_grid[str(grid)] = {
                "pixel_integral_gl4_vs_gl8": quadrature,
                "finite": finite,
                "source_input_power_w": source_power,
                "fiber_channel_power_w": channel_power.tolist(),
                "fiber_output_power_w": float(channel_power.sum()),
                "decoder_trace_w": output["decoder_trace"],
                "route_trace_w": output["route_trace"],
                "camera_trace_w": output["camera_trace"],
                "camera_power_w_per_pixel": _power_summary(pixel4),
                "camera_total_power_w": float(pixel4.sum()),
            }
            del output, coefficients, roi, pixel4, pixel8
            clear_production_transfer_cache()
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pattern_records.append({"pattern_index": index, "pattern_name": str(candidate["pattern_names"][index]), "by_grid": per_grid})

    pair_metrics: dict[str, Any] = {}
    all_window_pass = True
    worst: tuple[float, int, int, np.ndarray, np.ndarray] | None = None
    for index in pattern_indices:
        for left, right in ((4096, 6144), (4096, 8192), (6144, 8192)):
            metric = _window_metrics(roi_fields[index][left], roi_fields[index][right])
            within_gate = metric["common_global_phase_aligned_complex_relative_l2"] <= gate
            if precision == "complex64":
                all_window_pass &= within_gate
            pair_metrics[f"{index}/{left}_vs_{right}"] = metric | {"within_1e-4_reference_gate": within_gate}
            error = metric["common_global_phase_aligned_complex_relative_l2"]
            if worst is None or error > worst[0]:
                worst = (error, index, left, roi_fields[index][left].numpy(), roi_fields[index][right].numpy())
    worst_archive = None
    if worst is not None:
        error, pattern_index, left, reference, test = worst
        # Determine the opposite grid from the recorded worst error key.
        worst_pair_key = max(pair_metrics, key=lambda key: pair_metrics[key]["common_global_phase_aligned_complex_relative_l2"])
        pair_text = worst_pair_key.split("/", 1)[1]
        left_text, right_text = pair_text.split("_vs_")
        phase = np.angle(np.vdot(reference.astype(np.complex128), test.astype(np.complex128)))
        aligned = test * np.exp(-1j * phase)
        path = run_dir / "pilot_fields" / f"{_context_key(candidate_name, mode, route, decoder_state)}_{precision}_worst.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, reference=reference, test=test, aligned_difference=reference - aligned,
                            pattern_index=np.asarray(pattern_index), left_grid=np.asarray(int(left_text)), right_grid=np.asarray(int(right_text)),
                            metadata_json=np.asarray(json.dumps({"precision": precision, "common_roi_pixels": ROI, "single_global_phase_per_field": True})))
        worst_archive = str(path)
    status = "PASS" if precision == "complex64" and all_window_pass and all_quadrature_pass and all_finite else (
        "DIAGNOSTIC_FINITE" if precision == "complex128" and all_quadrature_pass and all_finite else "FAIL")
    result = {
        "candidate": candidate_name, "mode": mode, "route": route, "decoder_state": decoder_state,
        "precision": precision, "patterns": pattern_records, "window_pair_metrics": pair_metrics,
        "all_complex64_windows_within_1e-4": all_window_pass if precision == "complex64" else None,
        "all_pixel_integrals_within_1e-4": all_quadrature_pass, "all_outputs_finite": all_finite,
        "worst_pilot_field_archive": worst_archive, "status": status,
    }
    path = run_dir / "pilot_results" / f"{_context_key(candidate_name, mode, route, decoder_state)}_{precision}.json"
    _write_json(path, result)
    result["result_archive"] = str(path)
    del roi_fields
    return result


def _response_context_grid(
    config: dict[str, Any], candidate_name: str, candidate: dict[str, Any], mode: str,
    route: str, decoder_state: str, grid: int, precision: str,
    decoder_phases: dict[str, torch.Tensor], device: torch.device,
) -> dict[str, Any]:
    phase = decoder_phases[decoder_state]
    per_channel_trace: list[dict[str, Any]] = []
    roi_fields: list[np.ndarray] = []
    camera_gl4: list[np.ndarray] = []
    camera_gl8: list[np.ndarray] = []
    camera_center: list[np.ndarray] = []
    exit_fields: list[torch.Tensor] = []
    input_fields: list[torch.Tensor] = []
    layer1_fields: list[torch.Tensor] = []
    exit_z: float | None = None
    for index, coordinate in enumerate(_coordinates(candidate["size"])):
        mode_field = _global_mode(mode, coordinate, grid, device).to(torch.complex128 if precision == "complex128" else torch.complex64)
        output = _basis_forward(mode_field, route, phase, grid, precision)
        input_fields.append(output["decoder_input"].detach())
        layer1_fields.append(output["decoder_layer1"].detach())
        exit_fields.append(output["decoder_exit"].detach())
        roi_fields.append(output["final_roi"].detach().cpu().numpy())
        camera_gl4.append(output["camera_gl4"].detach().cpu().numpy())
        camera_gl8.append(output["camera_gl8"].detach().cpu().numpy())
        camera_center.append(output["camera_centers"].detach().cpu().numpy())
        per_channel_trace.append({"channel_index": index, "route": output["route_trace"], "decoder": output["decoder_trace"], "camera": output["camera_trace"]})
        exit_z = output["exit_z_m"]
        del output, mode_field
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dtype_np = np.complex64 if precision == "complex64" else np.complex128
    rroi_cpu = np.stack(roi_fields).astype(dtype_np, copy=False)
    r4_cpu = np.stack(camera_gl4).astype(dtype_np, copy=False)
    r8_cpu = np.stack(camera_gl8).astype(dtype_np, copy=False)
    camera_center_cpu = np.stack(camera_center).astype(dtype_np, copy=False)
    grams = {
        "fiber_output": _fiber_mode_gram(mode, candidate["size"]),
        "decoder_input": _gram(torch.stack(input_fields), DX * DX),
        "decoder_layer1": _gram(torch.stack(layer1_fields), DX * DX),
        "decoder_exit": _gram(torch.stack(exit_fields), DX * DX),
    }
    # The final ASM is unitary on the sampled propagating band; the second
    # phase plate contributes exactly its declared power transmission.
    grams["decoder_global"] = grams["decoder_layer1"] * TRANSMISSION
    return {
        "basis_roi": rroi_cpu,
        "camera_gl4": r4_cpu,
        "camera_gl8": r8_cpu,
        "camera_center": camera_center_cpu,
        "weights4": np.polynomial.legendre.leggauss(4)[1].astype(np.float64),
        "weights8": np.polynomial.legendre.leggauss(8)[1].astype(np.float64),
        "grams": {key: value.detach().cpu() for key, value in grams.items()},
        "per_channel_trace": per_channel_trace,
        "exit_z_m": exit_z,
        "precision": precision,
        "grid": grid,
    }


def _summarize_group(
    config: dict[str, Any], run_dir: Path, candidate_name: str, candidate: dict[str, Any], mode: str,
    route: str, decoder_state: str, grid: int, precision: str,
    basis: dict[str, Any], rng: np.random.Generator,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    commands = candidate["commands"]
    pattern_names = candidate["pattern_names"]
    coefficients = candidate["coefficients"][grid][mode]
    h = candidate["matrices"][mode][grid]["H"]
    matrix_prediction = _matrix_prediction_check(candidate, mode)
    if not matrix_prediction["passed"]:
        raise ValueError(f"saved v8 matrix prediction failed for {candidate_name}/{mode}/{grid}")
    coeff_relative = np.linalg.norm(commands @ h.T - coefficients) / max(float(np.linalg.norm(coefficients)), EPS)
    if coeff_relative > 1.0e-4:
        raise ValueError("v8 H matrix and saved complex coefficients do not replay the same command population")
    basis_roi = torch.from_numpy(basis["basis_roi"]).to(device=device)
    r4 = torch.from_numpy(basis["camera_gl4"]).to(device=device)
    r8 = torch.from_numpy(basis["camera_gl8"]).to(device=device)
    w4 = torch.from_numpy(basis["weights4"]).to(device=device, dtype=torch.float64)
    w8 = torch.from_numpy(basis["weights8"]).to(device=device, dtype=torch.float64)
    grams = {key: value.to(device=device) for key, value in basis["grams"].items()}
    records: list[dict[str, Any]] = []
    pixel_maps: list[np.ndarray] = []
    primary = grid == 4096 and precision == "complex64"
    uniform_index = _command_uniform_index(candidate)
    selected_noise_map: np.ndarray | None = None
    selected_power_map: np.ndarray | None = None
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    for index, name in enumerate(pattern_names):
        c = torch.from_numpy(coefficients[index]).to(device=device, dtype=dtype)
        roi_field = torch.einsum("n,nhw->hw", c, basis_roi)
        p4 = _pixel_powers_from_response(r4, c, 4, w4).to(torch.float64)
        p8 = _pixel_powers_from_response(r8, c, 8, w8).to(torch.float64)
        qgate = _quadrature_gate(p4, p8)
        stage_power = {key: _quadratic_power(c, gram) for key, gram in grams.items()}
        stage_power["camera_after_backend_relay"] = float(p4.sum())
        source_trace = candidate["power_traces"][grid]
        source_power = float(source_trace["source_input_power_w"][index])
        before_fiber_power = float(source_trace["common_roi_power_before_fiber_w"][index])
        channel_power = np.abs(coefficients[index]) ** 2
        exposure = {str(value): _exposure_summary(p4, value, config) for value in EXPOSURES}
        camera_counts = _photons(p4, float(config["camera"]["primary_exposure_s"]), float(config["camera"]["quantum_efficiency"]))
        record = {
            "pattern_index": index, "pattern_name": str(name),
            "window_output": {},
            "fiber_coefficients_sqrt_w": [[float(v.real), float(v.imag)] for v in coefficients[index]],
            "per_channel_fiber_power_w": channel_power.tolist(),
            "fiber_output_power_w": float(channel_power.sum()),
            "fiber_mode_reemission_power_w": stage_power["fiber_output"],
            "power_trace_w": {"source_input": source_power, "input_region_before_fiber": before_fiber_power, **stage_power},
            "attenuation_db_vs_source": {key: _dB_ratio(value, source_power) for key, value in stage_power.items()},
            "camera_pixel_power_w": _power_summary(p4),
            "camera_pixel_expected_electrons_1ms": _power_summary(camera_counts),
            "exposure_sensitivity": exposure,
            "quadrature_4x4_vs_8x8": qgate,
            "finite": bool(torch.isfinite(roi_field).all() and torch.isfinite(p4).all() and torch.isfinite(p8).all()),
        }
        if not qgate["passed"]:
            record["quadrature_status"] = "FAIL"
        if primary:
            pixel_maps.append(p4.detach().cpu().to(torch.float32).numpy())
        if index == uniform_index:
            selected_power_map = p4.detach().cpu().to(torch.float32).numpy()
            electrons = camera_counts.detach().cpu().numpy()
            noisy = rng.poisson(np.maximum(electrons, 0.0)).astype(np.float64) + rng.normal(0.0, float(config["camera"]["read_noise_e_rms"]), size=electrons.shape)
            noisy = np.clip(noisy, 0.0, float(config["camera"]["full_well_e"]))
            selected_noise_map = noisy.astype(np.float32)
        record["roi_field_cpu"] = roi_field.detach().cpu()
        records.append(record)
        del c, roi_field, p4, p8, camera_counts
    # Reconstruct all 50 physical complex output ROIs for pairwise common-ROI
    # comparisons while retaining only scalar metrics and the worst field.
    roi_fields = [item.pop("roi_field_cpu") for item in records]
    basis_meta = {"candidate": candidate_name, "mode": mode, "route": route, "decoder_state": decoder_state, "grid": grid, "precision": precision}
    group = {**basis_meta, "pattern_count": len(records), "patterns": records,
             "max_matrix_prediction_relative_error": float(matrix_prediction["maximum_relative_error"]),
             "pixel_integral_all_patterns_pass": all(bool(item["quadrature_4x4_vs_8x8"]["passed"]) for item in records)}
    if primary:
        pixel_payload = np.stack(pixel_maps).astype(np.float32)
        electron_payload = pixel_payload.astype(np.float64) * (float(config["camera"]["quantum_efficiency"]) * float(config["camera"]["primary_exposure_s"]) / (H_PLANCK * C_LIGHT / WAVELENGTH))
        task_payload = np.clip(electron_payload / float(config["camera"]["full_well_e"]), 0.0, 1.0).astype(np.float32)
        path = run_dir / "camera_measurements" / f"{_context_key(candidate_name, mode, route, decoder_state)}_4096_complex64.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".npz.tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, pattern_names=np.asarray(pattern_names), power_w_per_pixel=pixel_payload,
                                expected_electrons_1ms=electron_payload.astype(np.float32), task_output_1ms=task_payload,
                                uniform_t05_power_w=selected_power_map,
                                uniform_t05_noisy_electrons_1ms=selected_noise_map,
                                metadata_json=np.asarray(json.dumps(basis_meta | {"units": "W/pixel", "noise_seed": 2026101610})))
        temp.replace(path)
        group["camera_pixel_archive"] = str(path)
        group["uniform_t05_power_map"] = selected_power_map
        group["uniform_t05_noisy_electrons_map"] = selected_noise_map
    group["window_roi_fields"] = roi_fields
    return group


def _window_compare_groups(groups: dict[int, dict[str, Any]], precision: str, gate: float) -> dict[str, Any]:
    pair_names = ((4096, 6144), (4096, 8192), (6144, 8192))
    result: dict[str, Any] = {}
    all_pass = True
    for left, right in pair_names:
        left_group = groups[left][precision]
        right_group = groups[right][precision]
        pattern_records: list[dict[str, Any]] = []
        worst_error = -1.0
        worst_pair: tuple[int, torch.Tensor, torch.Tensor] | None = None
        for index, (field_left, field_right) in enumerate(zip(left_group["window_roi_fields"], right_group["window_roi_fields"], strict=True)):
            metrics = _window_metrics(field_left, field_right)
            passed = precision != "complex64" or metrics["common_global_phase_aligned_complex_relative_l2"] <= gate
            metrics["passed"] = passed
            pattern_records.append(metrics)
            all_pass &= passed
            if metrics["common_global_phase_aligned_complex_relative_l2"] > worst_error:
                worst_error = metrics["common_global_phase_aligned_complex_relative_l2"]
                worst_pair = (index, field_left, field_right)
        worst_record = None
        if worst_pair is not None:
            idx, field_l, field_r = worst_pair
            phase = torch.angle(torch.sum(field_l.to(torch.complex128).conj() * field_r.to(torch.complex128)))
            aligned_r = field_r * torch.exp(-1j * phase.to(field_r.real.dtype))
            diff = field_l - aligned_r
            full_path = None
            run_dir = groups[left][precision].get("_run_dir")
            if run_dir is not None:
                field_dir = Path("window_error_fields") / f"{left}_vs_{right}_{precision}_worst.npz"
                full_path = run_dir / field_dir
                full_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(full_path, reference=np.asarray(field_l.numpy()), test=np.asarray(field_r.numpy()),
                                    aligned_difference=np.asarray(diff.numpy()), pattern_index=np.asarray(idx),
                                    metadata_json=np.asarray(json.dumps({"precision": precision, "common_roi_pixels": ROI, "phase_alignment_count": 1})))
            worst_record = {"pattern_index": idx, "aligned_error": worst_error, "field_archive": str(full_path)}
        result[f"{left}_vs_{right}"] = {"patterns": pattern_records, "max_aligned_complex_error": worst_error, "worst_field": worst_record,
                                          "passed": all(item["passed"] for item in pattern_records)}
    return {"pairs": result, "passed": all_pass if precision == "complex64" else None}


def _scale_bounded_direction(direction: torch.Tensor, rms: float, lower: float, upper: float, base: float) -> tuple[torch.Tensor, float]:
    direction = direction.to(torch.float64)
    norm = torch.linalg.vector_norm(direction).clamp_min(EPS)
    requested = direction * (rms * math.sqrt(direction.numel()) / norm)
    positive = float(requested.clamp_min(0.0).max())
    negative = float((-requested).clamp_min(0.0).max())
    limits = [1.0]
    if positive > 0.0:
        limits.append((upper - base) / positive)
    if negative > 0.0:
        limits.append((base - lower) / negative)
    scale = max(0.0, min(limits))
    return requested * scale, scale


def _camera_jacobian_grid_factors(
    config: dict[str, Any], candidate: dict[str, Any], mode: str, grid: int,
    response_gl4: torch.Tensor, weights4: torch.Tensor, precision: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = response_gl4.device
    h = torch.from_numpy(candidate["matrices"][mode][grid]["H"]).to(device=device, dtype=torch.complex128)
    t0 = torch.full((10000,), 0.5, device=device, dtype=torch.float64)
    c0 = h @ t0.to(torch.complex128)
    a, b, field0 = _camera_linear_factors(response_gl4, weights4, h, c0)
    return a, b, field0, c0


def _direct_superposition_check(
    candidate_name: str, candidate: dict[str, Any], mode: str, route: str, decoder_state: str,
    decoder_phase: torch.Tensor, command: torch.Tensor, h_np: np.ndarray,
    basis_roi: torch.Tensor, basis_camera_gl4: torch.Tensor, weights4: torch.Tensor,
    device: torch.device, label: str,
) -> dict[str, Any]:
    command = command.to(device=device, dtype=torch.float64)
    h = torch.from_numpy(h_np).to(device=device, dtype=torch.complex128)
    coefficients = h @ command.to(torch.complex128)
    predicted_roi = torch.einsum("n,nhw->hw", coefficients.to(torch.complex64), basis_roi.to(torch.complex64))
    predicted_camera = torch.einsum("n,nyx->yx", coefficients.to(torch.complex64), basis_camera_gl4.to(torch.complex64))
    predicted_pixels = _pixel_powers_from_response(basis_camera_gl4.to(torch.complex64), coefficients.to(torch.complex64), 4, weights4)
    direct = _full_pattern_forward(mode, candidate, coefficients.to(torch.complex64), route, decoder_phase, 4096, "complex64", device)
    direct_roi = _crop_roi(direct["final_full"])
    direct_camera = direct["camera_gl4"]
    roi_error = float(torch.linalg.vector_norm(predicted_roi.to(torch.complex128) - direct_roi.to(torch.complex128)) / torch.linalg.vector_norm(direct_roi.to(torch.complex128)).clamp_min(EPS))
    camera_error = float(torch.linalg.vector_norm(predicted_camera.to(torch.complex128) - direct_camera.to(torch.complex128)) / torch.linalg.vector_norm(direct_camera.to(torch.complex128)).clamp_min(EPS))
    pixel_error = float(torch.linalg.vector_norm(predicted_pixels.to(torch.float64) - direct["pixel_power_gl4"].to(torch.float64)) / torch.linalg.vector_norm(direct["pixel_power_gl4"].to(torch.float64)).clamp_min(EPS))
    passed = max(roi_error, camera_error, pixel_error) <= 1.0e-4
    result = {
        "label": label, "command_min": float(command.min()), "command_max": float(command.max()),
        "common_roi_complex_relative_error": roi_error,
        "camera_field_complex_relative_error": camera_error,
        "pixel_power_relative_error": pixel_error,
        "passed": passed,
    }
    del direct, predicted_roi, predicted_camera, predicted_pixels, direct_roi, direct_camera
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _write_camera_diagnostic_images(run_dir: Path, linear_maps: dict[str, np.ndarray], noisy_maps: dict[str, np.ndarray]) -> dict[str, Any]:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to save the authorized shared-scale camera diagnostic images") from exc
    names = list(linear_maps)
    linear_max = max(float(np.max(linear_maps[name])) for name in names)
    noise_max = max(float(np.max(noisy_maps[name])) for name in names)
    width, height = 4 * 200, 4 * 224
    linear_sheet = Image.new("L", (width, height), color=0)
    noise_sheet = Image.new("L", (width, height), color=0)
    linear_draw, noise_draw = ImageDraw.Draw(linear_sheet), ImageDraw.Draw(noise_sheet)
    for index, name in enumerate(names):
        row, column = divmod(index, 4)
        x, y = column * 200, row * 224
        linear = np.clip(linear_maps[name] / max(linear_max, EPS), 0.0, 1.0)
        noisy = np.clip(noisy_maps[name] / max(noise_max, EPS), 0.0, 1.0)
        linear_sheet.paste(Image.fromarray((linear * 255.0).astype(np.uint8), mode="L"), (x, y))
        noise_sheet.paste(Image.fromarray((noisy * 255.0).astype(np.uint8), mode="L"), (x, y))
        label = name.replace("candidate_", "").replace("__", " | ")[:28]
        linear_draw.text((x + 2, y + 202), label, fill=255)
        noise_draw.text((x + 2, y + 202), label, fill=255)
    linear_path = run_dir / "camera_diagnostic_shared_linear.png"
    noise_path = run_dir / "camera_noise_example_shared_scale.png"
    linear_sheet.save(linear_path)
    noise_sheet.save(noise_path)
    return {"linear_image": str(linear_path), "noise_image": str(noise_path),
            "all_cases_share_linear_power_scale_w_per_pixel": linear_max,
            "all_cases_share_noise_electron_scale": noise_max,
            "images_are_diagnostic_sensor_maps_not_semantic_generation": True}


def _measure_mapped_route_power(
    run_dir: Path, source: dict[str, Any], device: torch.device, selected_pairs: set[str],
) -> dict[str, str]:
    """Measure exact route-stage powers for all saved commands at 4096 grid."""
    archives: dict[str, str] = {}
    for candidate_name, candidate in source["candidates"].items():
        for mode in MODES:
            if not any(f"{candidate_name}__{mode}__{route}" in selected_pairs for route in ROUTES):
                continue
            basis = _load_mode_basis(mode, candidate, 4096, torch.complex64, device)
            stacked_basis = torch.stack(basis)
            for route in ROUTES:
                key = f"{candidate_name}__{mode}__{route}"
                if key not in selected_pairs:
                    continue
                rows: list[dict[str, Any]] = []
                for index, name in enumerate(candidate["pattern_names"]):
                    coefficients = torch.from_numpy(candidate["coefficients"][4096][mode][index]).to(device=device, dtype=torch.complex64)
                    source_field = torch.einsum("n,nhw->hw", coefficients, stacked_basis)
                    _active, trace, _z = _route_to_decoder(source_field, route, 4096)
                    powers = [float(value) for key, value in trace.items() if key.endswith("_power_w")]
                    passive = all(
                        right <= left * (1.0 + 1.0e-5) + EPS
                        for left, right in zip(powers, powers[1:], strict=False)
                    )
                    rows.append({
                        "pattern_index": index,
                        "pattern_name": str(name),
                        "route_stage_power_w": trace,
                        "power_chain_passive": passive,
                    })
                    del coefficients, source_field, trace, _active
                path = run_dir / "mapped_route_power" / f"{key}_4096_complex64.json"
                _write_json(path, {
                    "schema": f"{SCHEMA}.mapped_route_power",
                    "candidate": candidate_name,
                    "mode": mode,
                    "route": route,
                    "grid": 4096,
                    "patterns": rows,
                    "all_50_power_chains_passive": all(row["power_chain_passive"] for row in rows),
                })
                archives[key] = str(path)
                del rows, stacked_basis, basis
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                basis = _load_mode_basis(mode, candidate, 4096, torch.complex64, device) if route == ROUTES[0] else []
                stacked_basis = torch.stack(basis) if basis else None
            if stacked_basis is not None:
                del stacked_basis
            if basis:
                del basis
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return archives


def _save_frozen_inputs(run_dir: Path, source: dict[str, Any]) -> dict[str, Any]:
    phase_payload: dict[str, torch.Tensor] = {}
    archive_payload: dict[str, np.ndarray] = {}
    for candidate_name, candidate in source["candidates"].items():
        phase_payload[candidate_name] = candidate["phase"].detach().cpu()
        archive_payload[f"{candidate_name}__amplitude_commands"] = candidate["commands"]
        archive_payload[f"{candidate_name}__pattern_names"] = candidate["pattern_names"]
        archive_payload[f"{candidate_name}__coordinates_yx_m"] = np.asarray(_coordinates(candidate["size"]), dtype=np.float64)
        for grid in GRIDS:
            for mode in MODES:
                archive_payload[f"{candidate_name}__{mode}__{grid}__H_sqrt_w"] = candidate["matrices"][mode][grid]["H"]
                archive_payload[f"{candidate_name}__{mode}__{grid}__coefficients_sqrt_w"] = candidate["coefficients"][grid][mode]
    phase_path = run_dir / "frozen_encoder_phase_screens_v9.pt"
    torch.save({
        "schema": "ols_decoder_camera_frozen_encoder_phases.v9",
        "source_v8_run_dir": source["run_dir"],
        "phase_state_ids": {name: value["phase_state_id"] for name, value in source["candidates"].items()},
        "phases": phase_payload,
    }, phase_path)
    input_path = run_dir / "frozen_v8_response_inputs_v9.npz"
    np.savez_compressed(input_path, **archive_payload,
                        metadata_json=np.asarray(json.dumps({"source_v8_run_dir": source["run_dir"],
                                                             "H_units": "sqrt(W) per unit amplitude command",
                                                             "coefficient_units": "sqrt(W)",
                                                             "grids": list(GRIDS), "modes": list(MODES)})))
    return {"phase_screens": str(phase_path), "response_inputs": str(input_path), "source_v8_run_dir": source["run_dir"]}


def _context_summary(public: dict[str, Any]) -> dict[str, Any]:
    jacobian = public.get("camera_jacobian") or {}
    exposure = jacobian.get("exposure_sensitivity", {})
    primary = exposure.get("0.001", {})
    return {
        "candidate": public["candidate"], "mode": public["mode"], "route": public["route"],
        "decoder_state": public["decoder_state"], "precision": public["precision"],
        "all_complex64_windows_pass": public.get("all_complex64_windows_pass"),
        "all_patterns_pixel_integral_pass": public.get("all_patterns_pixel_integral_pass"),
        "matrix_prediction_relative_error_by_grid": public.get("matrix_prediction_relative_error_by_grid"),
        "valid_for_sensor_analysis": public.get("valid_for_sensor_analysis"),
        "camera_pixel_archive": public.get("camera_pixel_archive"),
        "scalar_result_archive": public.get("scalar_result_archive"),
        "jacobian_central_difference": jacobian.get("complex128_central_difference"),
        "raw_jacobian_dimension_by_threshold": (jacobian.get("raw_power_jacobian_spectrum") or {}).get("effective_dimension_by_relative_threshold"),
        "primary_exposure_snr_direction_counts": primary.get("snr_direction_counts"),
        "cross_window_jacobian_uncertainty": public.get("cross_window_jacobian_uncertainty"),
        "direct_forward_checks": public.get("direct_forward_superposition_checks"),
    }


def _pilot_pattern_indices(source: dict[str, Any], candidate_name: str, candidate: dict[str, Any]) -> list[int]:
    # v8's top-level v7_worst_probe_replay is an acceptance boolean. The exact
    # replayed pattern identity is recorded under window_pilot[candidate].
    candidate_pilot = source["report"].get("window_pilot", {}).get(candidate_name)
    patterns = candidate_pilot.get("patterns") if isinstance(candidate_pilot, dict) else None
    if not isinstance(patterns, dict):
        raise ValueError(f"v8 report lacks window-pilot pattern records for {candidate_name}")
    worst_records = [
        (name, item)
        for name, item in patterns.items()
        if isinstance(item, dict)
        and isinstance(item.get("input_identity"), dict)
        and item["input_identity"].get("v7_probe_index") is not None
    ]
    if len(worst_records) != 1:
        raise ValueError(f"v8 report must identify exactly one replayed worst binary probe for {candidate_name}")
    pattern_name, pilot = worst_records[0]
    identity = pilot["input_identity"]
    index = int(identity["v7_probe_index"])
    if not 0 <= index < len(candidate["pattern_names"]):
        raise ValueError(f"v8 worst-binary pattern index is out of range for {candidate_name}")
    if str(candidate["pattern_names"][index]) != str(identity["source_v7_pattern"]):
        raise ValueError(f"v8 worst-binary pattern identity changed for {candidate_name}")
    if pattern_name != str(identity["source_v7_pattern"]):
        raise ValueError(f"v8 window-pilot record name disagrees with its source identity for {candidate_name}")
    return list(dict.fromkeys((index, _command_uniform_index(candidate))))


def run(config_path: Path, run_dir: Path, preflight_only: bool, source_override: str | None) -> dict[str, Any]:
    root = Path.cwd().resolve()
    config_path = config_path.resolve()
    run_dir = run_dir.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = _load_config(config_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, run_dir / "configuration_v9.yaml")
    config = _load_config(run_dir / "configuration_v9.yaml")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    preflight = _preflight(config, run_dir / "configuration_v9.yaml", run_dir, root, source_override)
    if preflight_only:
        status = "PREFLIGHT_READY" if preflight["passed"] else "PREFLIGHT_BLOCKED"
        (run_dir / "exit_status.txt").write_text(status + "\n", encoding="utf-8")
        return preflight
    if not preflight["passed"]:
        result = {
            "schema": SCHEMA, "status": "INVALID_HARNESS", "decision_question": "Does the frozen four/nine-channel output interface produce numerically qualified proxy-sensor responses?",
            "preflight_path": str(run_dir / "preflight.json"), "preflight_status": preflight["status"],
            "formal_training_updates": 0, "phase_optimization_updates": 0, "dataset_reads": 0,
            "task_evaluator_calls": 0, "fid_calls": 0, "loss_changes": 0, "material_forward_calls": 0,
        }
        _write_json(run_dir / "report.json", result)
        (run_dir / "exit_status.txt").write_text("INVALID_HARNESS\n", encoding="utf-8")
        return result

    source = _load_source(config, root, source_override)
    frozen_inputs = _save_frozen_inputs(run_dir, source)
    decoder_phases = _new_decoder_phases(config["decoder"]["low_frequency_seed"], device)
    noise_rng = np.random.default_rng(int(config["camera"]["noise_seed"]))
    contexts: list[dict[str, Any]] = []
    image_power: dict[str, np.ndarray] = {}
    image_noise: dict[str, np.ndarray] = {}
    total = len(source["candidates"]) * len(MODES) * len(ROUTES) * len(DECODER_STATES)
    completed = 0
    for candidate_name, candidate in source["candidates"].items():
        pattern_indices = _pilot_pattern_indices(source, candidate_name, candidate)
        for mode in MODES:
            for route in ROUTES:
                for decoder_state in DECODER_STATES:
                    pilot64 = _measure_pilot_context(
                        config, run_dir, candidate_name, candidate, mode, route, decoder_state,
                        decoder_phases[decoder_state], "complex64", device, pattern_indices,
                    )
                    pilot128 = _measure_pilot_context(
                        config, run_dir, candidate_name, candidate, mode, route, decoder_state,
                        decoder_phases[decoder_state], "complex128", device, pattern_indices,
                    )
                    full_measurements: dict[str, Any] | None = None
                    full_context_pass = pilot64["status"] == "PASS" and pilot128["status"] == "DIAGNOSTIC_FINITE"
                    if full_context_pass:
                        c64, c64_private = _run_context_precision(
                            config, run_dir, candidate_name, candidate, mode, route, decoder_state,
                            decoder_phases, "complex64", device, noise_rng,
                        )
                        c128, c128_private = _run_context_precision(
                            config, run_dir, candidate_name, candidate, mode, route, decoder_state,
                            decoder_phases, "complex128", device, noise_rng,
                            primary_c64_cache=c64_private["primary_c64_cache"],
                        )
                        image_power.update(c64_private["images_for_power"])
                        image_noise.update(c64_private["images_for_noise"])
                        full_measurements = {"complex64": c64, "complex128": c128}
                        del c64, c128, c64_private, c128_private
                    context = {
                        "candidate": candidate_name, "mode": mode, "route": route, "decoder_state": decoder_state,
                        "pilot": {"complex64": pilot64, "complex128": pilot128},
                        "pilot_pattern_indices": pattern_indices,
                        "full_measurements": full_measurements,
                        "full_measurement_skipped_by_pilot": not full_context_pass,
                    }
                    contexts.append(context)
                    completed += 1
                    _write_json(run_dir / "progress.json", {
                        "schema": f"{SCHEMA}.progress", "completed_contexts": completed,
                        "total_contexts": total, "last_context": _context_key(candidate_name, mode, route, decoder_state),
                        "last_pilot_status": {"complex64": pilot64["status"], "complex128": pilot128["status"]},
                        "full_measurement_started": full_context_pass,
                        "elapsed_seconds": time.time() - _START,
                    })
                    gc.collect()
                    if device.type == "cuda":
                        clear_production_transfer_cache()
                        torch.cuda.empty_cache()

    selected_pairs = {
        f"{item['candidate']}__{item['mode']}__{item['route']}"
        for item in contexts if item["full_measurements"] is not None
    }
    mapped_power_archives = _measure_mapped_route_power(run_dir, source, device, selected_pairs) if selected_pairs else {}
    clear_production_transfer_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    image_result = _write_camera_diagnostic_images(run_dir, image_power, image_noise) if image_power else {
        "status": "NOT_GENERATED_NO_CONTEXT_PASSED_THE_PILOT", "images_are_diagnostic_sensor_maps_not_semantic_generation": True,
    }
    by_context: dict[str, Any] = {}
    for item in contexts:
        key = _context_key(item["candidate"], item["mode"], item["route"], item["decoder_state"])
        full = item["full_measurements"]
        by_context[key] = {
            "pilot_pattern_indices": item["pilot_pattern_indices"],
            "pilot": {precision: {"status": value["status"], "result_archive": value["result_archive"],
                                  "all_pixel_integrals_within_1e-4": value["all_pixel_integrals_within_1e-4"],
                                  "all_complex64_windows_within_1e-4": value["all_complex64_windows_within_1e-4"]}
                      for precision, value in item["pilot"].items()},
            "full_measurement_skipped_by_pilot": item["full_measurement_skipped_by_pilot"],
            "full_measurements": None if full is None else {
                "complex64": _context_summary(full["complex64"]),
                "complex128": _context_summary(full["complex128"]),
                "complex64_archive": full["complex64"]["scalar_result_archive"],
                "complex128_archive": full["complex128"]["scalar_result_archive"],
            },
        }
    qualified_contexts = [
        item["full_measurements"] for item in contexts
        if item["full_measurements"] is not None
        and item["full_measurements"]["complex64"]["valid_for_sensor_analysis"]
        and item["full_measurements"]["complex128"].get("camera_jacobian", {}).get("complex128_central_difference", {}).get("passed", False)
    ]
    status = "VALID_COMPLETED" if len(contexts) == total else "INCOMPLETE"
    final_gate = {
        "status": "SUPPORTED_WITHIN_QUALIFIED_CONTEXTS" if qualified_contexts else "INSUFFICIENT_QUALIFIED_CONTEXTS",
        "qualified_context_count": len(qualified_contexts), "full_measurement_context_count": sum(item["full_measurements"] is not None for item in contexts),
        "pilot_context_count": len(contexts), "total_context_count": total,
        "qualification_scope": "frozen v8 phases, the two specified output routes, fixed decoder screens, stated Gaussian/LP01 mode assumptions, and proxy-camera parameters only",
    }
    result = {
        "schema": SCHEMA, "status": status,
        "decision_question": "With v8 phases frozen, can the four- and nine-channel complex outputs form numerically credible absolute-power sensor responses through the specified routes, decoder, and proxy camera?",
        "source": frozen_inputs,
        "physical_assumptions": {
            "wavelength_m": WAVELENGTH, "encoder_aperture_m": 200.0e-6, "pixel_pitch_m": DX,
            "encoder_propagation_distances_m": [1.5e-3, 1.5e-3], "fiber_pitch_m": PITCH,
            "assumed_gaussian_mfd_m": 2.0 * WAIST, "layer_power_transmission": TRANSMISSION,
            "source_power_w": SOURCE_POWER, "fiber_transport": "ideal identity, not measured",
            "gaussian_s12_primary": True, "lp01_v2p2_same_geometry_sensitivity_control": True,
            "all_geometry_and_camera_parameters_are_simulation_assumptions": True,
        },
        "decoder_phase_screens": "decoder_phase_screens_v9.pt",
        "mapped_route_power_archives_4096_complex64": mapped_power_archives,
        "camera_diagnostic_images": image_result,
        "contexts": by_context,
        "final_gate": final_gate,
        "forbidden_activity": {"formal_training_updates": 0, "phase_optimization_updates": 0, "dataset_reads": 0,
                               "task_evaluator_calls": 0, "fid_calls": 0, "loss_changes": 0, "material_forward_calls": 0},
        "execution": {
            "python_version": platform.python_version(), "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "runtime_module_sources": {"v9_cli": str(Path(__file__).resolve()),
                                       "fiber_mode_reference": str(Path(fiber_operator.__file__).resolve())},
            "elapsed_seconds": time.time() - _START,
        },
        "claim_limits": [
            "sensor maps are deterministic physical power integrals and proxy-camera expectations, not semantic generation outputs",
            "noise parameters, relay optics, fiber geometry, and all device values remain unmeasured simulation assumptions",
            "qualified local responses do not establish a final OLS dimension, hardware performance, or semantic image quality",
        ],
    }
    _write_json(run_dir / "report.json", result)
    artifact_path = root / "artifacts" / "ols_decoder_camera_interface_v9.json"
    _write_json(artifact_path, result)
    (run_dir / "exit_status.txt").write_text(f"{status}\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--source-run-dir", type=str, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.run_dir, arguments.preflight_only, arguments.source_run_dir)
    print(json.dumps({"status": result.get("status"), "run_dir": str(arguments.run_dir.resolve()),
                      "elapsed_seconds": result.get("elapsed_seconds")}, ensure_ascii=False))


def _run_context_precision(
    config: dict[str, Any], run_dir: Path, candidate_name: str, candidate: dict[str, Any],
    mode: str, route: str, decoder_state: str, decoder_phases: dict[str, torch.Tensor],
    precision: str, device: torch.device, noise_rng: np.random.Generator,
    primary_c64_cache: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    basis_by_grid: dict[int, dict[str, Any]] = {}
    for grid in GRIDS:
        # The production ASM cache keys each grid/distance/precision kernel.
        # Clear it at grid boundaries so multi-GB complex128 transfer kernels
        # from earlier windows cannot accumulate into later window peaks.
        clear_production_transfer_cache()
        basis_by_grid[grid] = _response_context_grid(
            config, candidate_name, candidate, mode, route, decoder_state, grid,
            precision, decoder_phases, device,
        )
        clear_production_transfer_cache()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    dtype = torch.complex128 if precision == "complex128" else torch.complex64
    gate = float(config["gates"]["aligned_common_roi_complex64_max"])
    thresholds = [float(value) for value in config["measurement"]["relative_svd_thresholds"]]
    basis_roi_gpu = {grid: torch.from_numpy(basis_by_grid[grid]["basis_roi"]).to(device=device) for grid in GRIDS}
    camera_center_gpu = {grid: torch.from_numpy(basis_by_grid[grid]["camera_center"]).to(device=device) for grid in GRIDS}
    response4_gpu = {grid: torch.from_numpy(basis_by_grid[grid]["camera_gl4"]).to(device=device) for grid in GRIDS}
    response8_gpu = {grid: torch.from_numpy(basis_by_grid[grid]["camera_gl8"]).to(device=device) for grid in GRIDS}
    weights4_gpu = {grid: torch.from_numpy(basis_by_grid[grid]["weights4"]).to(device=device, dtype=torch.float64) for grid in GRIDS}
    weights8_gpu = {grid: torch.from_numpy(basis_by_grid[grid]["weights8"]).to(device=device, dtype=torch.float64) for grid in GRIDS}
    grams_gpu = {grid: {key: value.to(device=device) for key, value in basis_by_grid[grid]["grams"].items()} for grid in GRIDS}
    uniform_index = _command_uniform_index(candidate)
    records: list[dict[str, Any]] = []
    maps_primary: list[np.ndarray] = []
    uniform_power_map: np.ndarray | None = None
    uniform_noise_map: np.ndarray | None = None
    window_metrics_by_pair: dict[str, list[dict[str, Any]]] = {"4096_vs_6144": [], "4096_vs_8192": [], "6144_vs_8192": []}
    worst_by_pair: dict[str, tuple[float, int, np.ndarray, np.ndarray]] = {}
    all_quadrature_pass = True
    matrix_prediction = {mode: _matrix_prediction_check(candidate, mode)}
    if not matrix_prediction[mode]["passed"]:
        raise ValueError(f"v8 matrix prediction failed before camera measurement: {candidate_name}/{mode}")
    coefficient_prediction_by_grid: dict[int, float] = {}
    for grid in GRIDS:
        h_np = candidate["matrices"][mode][grid]["H"]
        predictions = candidate["commands"].astype(np.complex128) @ h_np.T
        saved = candidate["coefficients"][grid][mode]
        errs = np.linalg.norm(predictions - saved, axis=1) / np.maximum(np.linalg.norm(saved, axis=1), EPS)
        coefficient_prediction_by_grid[grid] = float(np.max(errs))

    factor_records: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for grid in GRIDS:
        h = torch.from_numpy(candidate["matrices"][mode][grid]["H"]).to(device=device, dtype=torch.complex128)
        t0 = torch.full((10000,), 0.5, device=device, dtype=torch.float64)
        c0 = h @ t0.to(torch.complex128)
        a, b, _base_field = _camera_linear_factors(response4_gpu[grid], weights4_gpu[grid], h, c0)
        if precision == "complex64":
            factor_records[grid] = (a.detach().cpu(), b.detach().cpu())
        del h, c0, a, b

    uniform_command = torch.full((10000,), 0.5, device=device, dtype=torch.float64)
    uniform_direct_check: dict[str, Any] | None = None
    if precision == "complex64":
        uniform_direct_check = _direct_superposition_check(
            candidate_name, candidate, mode, route, decoder_state, decoder_phases[decoder_state],
            uniform_command, candidate["matrices"][mode][4096]["H"], basis_roi_gpu[4096],
            response4_gpu[4096], weights4_gpu[4096], device, "uniform_t0p5",
        )

    for index, pattern_name_value in enumerate(candidate["pattern_names"]):
        pattern_name = str(pattern_name_value)
        combined_roi: dict[int, torch.Tensor] = {}
        coefficients: dict[int, torch.Tensor] = {}
        for grid in GRIDS:
            c = torch.from_numpy(candidate["coefficients"][grid][mode][index]).to(device=device, dtype=dtype)
            coefficients[grid] = c
            combined_roi[grid] = torch.einsum("n,nhw->hw", c, basis_roi_gpu[grid])
        pair_values: dict[str, Any] = {}
        for left, right in ((4096, 6144), (4096, 8192), (6144, 8192)):
            pair = f"{left}_vs_{right}"
            metric = _window_metrics(combined_roi[left], combined_roi[right])
            passed = precision != "complex64" or metric["common_global_phase_aligned_complex_relative_l2"] <= gate
            metric["passed"] = passed
            pair_values[pair] = metric
            window_metrics_by_pair[pair].append(metric)
            if precision == "complex64" and not passed:
                pass
            current_worst = worst_by_pair.get(pair)
            if current_worst is None or metric["common_global_phase_aligned_complex_relative_l2"] > current_worst[0]:
                worst_by_pair[pair] = (
                    metric["common_global_phase_aligned_complex_relative_l2"], index,
                    combined_roi[left].detach().cpu().numpy(), combined_roi[right].detach().cpu().numpy(),
                )
        per_grid: dict[str, Any] = {}
        for grid in GRIDS:
            c = coefficients[grid]
            p4 = _camera_power_for_coefficients(response4_gpu[grid], c, 4, weights4_gpu[grid]).to(torch.float64)
            p8 = _camera_power_for_coefficients(response8_gpu[grid], c, 8, weights8_gpu[grid]).to(torch.float64)
            center_field = torch.einsum("n,nyx->yx", c.to(camera_center_gpu[grid].dtype), camera_center_gpu[grid])
            center_approximation = center_field.abs().square().to(torch.float64) * (2.5e-6**2)
            center_approximation_bias = {
                "relative_l2_vs_pixel_integral": float(torch.linalg.vector_norm(center_approximation - p4) / torch.linalg.vector_norm(p4).clamp_min(EPS)),
                "total_power_relative_error": abs(float(center_approximation.sum()) - float(p4.sum())) / max(float(p4.sum()), EPS),
                "interpretation": "legacy center intensity times pixel area approximation; not used for sensor output",
            }
            qgate = _quadrature_gate(p4, p8)
            all_quadrature_pass &= bool(qgate["passed"])
            gram = grams_gpu[grid]
            powers = {key: _quadratic_power(c, value) for key, value in gram.items()}
            powers["decoder_layer2_global"] = powers["decoder_layer1"] * TRANSMISSION
            powers["camera_sensor_500um_view"] = float(p4.sum())
            trace = candidate["power_traces"][grid]
            source_power = float(trace["source_input_power_w"][index])
            pre_fiber = float(trace["common_roi_power_before_fiber_w"][index])
            channel_power = np.abs(candidate["coefficients"][grid][mode][index]) ** 2
            exposure_records = {str(exposure): _exposure_summary(p4, exposure, config) for exposure in EXPOSURES}
            mu_primary = _photons(p4, float(config["camera"]["primary_exposure_s"]), float(config["camera"]["quantum_efficiency"]))
            row = {
                "pattern_index": index,
                "pattern_name": pattern_name,
                "coefficient_prediction_relative_error": coefficient_prediction_by_grid[grid],
                "fiber_coefficients_sqrt_w": [[float(v.real), float(v.imag)] for v in candidate["coefficients"][grid][mode][index]],
                "per_channel_fiber_power_w": channel_power.tolist(),
                "fiber_output_power_w": float(channel_power.sum()),
                "power_before_fiber_from_v8_w": pre_fiber,
                "source_input_power_w": source_power,
                "stage_power_w": powers,
                "stage_ratio_to_source": {key: value / max(source_power, EPS) for key, value in powers.items()},
                "stage_level_db_vs_source": {key: _dB_ratio(value, source_power) for key, value in powers.items()},
                "camera_pixel_power_w": _power_summary(p4),
                "camera_expected_electrons_1ms": _power_summary(mu_primary),
                "legacy_center_sample_approximation_bias": center_approximation_bias,
                "exposure_sensitivity": exposure_records,
                "pixel_integration_gl4_vs_gl8": qgate,
                "window_comparisons": pair_values,
                "finite": bool(torch.isfinite(combined_roi[grid]).all() and torch.isfinite(p4).all() and torch.isfinite(p8).all()),
            }
            if grid == 4096 and precision == "complex64":
                maps_primary.append(p4.detach().cpu().to(torch.float32).numpy())
            if index == uniform_index and grid == 4096 and precision == "complex64":
                uniform_power_map = p4.detach().cpu().to(torch.float32).numpy()
                electrons = mu_primary.detach().cpu().numpy()
                noisy = noise_rng.poisson(np.maximum(electrons, 0.0)).astype(np.float64)
                noisy += noise_rng.normal(0.0, float(config["camera"]["read_noise_e_rms"]), size=electrons.shape)
                uniform_noise_map = np.clip(noisy, 0.0, float(config["camera"]["full_well_e"])).astype(np.float32)
            per_grid[str(grid)] = row
            del p4, p8, mu_primary, center_field, center_approximation
        records.append({"pattern_index": index, "pattern_name": pattern_name, "by_grid": per_grid, "window_comparisons": pair_values})
        del combined_roi, coefficients

    window_result: dict[str, Any] = {}
    all_window_pass = True
    for pair, metrics in window_metrics_by_pair.items():
        max_error = max(item["common_global_phase_aligned_complex_relative_l2"] for item in metrics)
        passed = precision != "complex64" or max_error <= gate
        all_window_pass &= passed
        worst = worst_by_pair.get(pair)
        worst_archive = None
        if worst is not None:
            err, idx, left_field, right_field = worst
            left_text, right_text = pair.split("_vs_")
            path = run_dir / "window_error_fields" / f"{_context_key(candidate_name, mode, route, decoder_state)}__{left_text}_vs_{right_text}_{precision}_worst.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            phase = np.angle(np.sum(np.conj(left_field.astype(np.complex128)) * right_field.astype(np.complex128)))
            aligned = right_field * np.exp(-1j * phase)
            np.savez_compressed(path, reference=left_field, test=right_field, aligned_difference=left_field - aligned,
                                pattern_index=np.asarray(idx), metadata_json=np.asarray(json.dumps({"common_roi_pixels": ROI, "precision": precision, "single_common_global_phase_per_field": True})))
            worst_archive = str(path)
        window_result[pair] = {"max_aligned_complex_error": max_error, "all_50_patterns_pass": passed, "pattern_metrics": metrics,
                               "worst_pattern_field_archive": worst_archive}

    context_id = _context_key(candidate_name, mode, route, decoder_state)
    camera_archive = None
    images_for_power: dict[str, np.ndarray] = {}
    images_for_noise: dict[str, np.ndarray] = {}
    if precision == "complex64" and maps_primary:
        power_array = np.stack(maps_primary).astype(np.float32)
        photon_factor = float(config["camera"]["quantum_efficiency"] * config["camera"]["primary_exposure_s"] / (H_PLANCK * C_LIGHT / WAVELENGTH))
        electron_array = (power_array.astype(np.float64) * photon_factor).astype(np.float32)
        task_array = np.clip(electron_array / float(config["camera"]["full_well_e"]), 0.0, 1.0)
        camera_archive_path = run_dir / "camera_measurements" / f"{context_id}_4096_complex64.npz"
        camera_archive_path.parent.mkdir(parents=True, exist_ok=True)
        temp = camera_archive_path.with_suffix(".npz.tmp")
        with temp.open("wb") as handle:
            np.savez_compressed(handle, pattern_names=candidate["pattern_names"], power_w_per_pixel=power_array,
                                expected_electrons_1ms=electron_array, task_output_1ms=task_array,
                                uniform_t05_power_w=uniform_power_map, uniform_t05_noisy_electrons_1ms=uniform_noise_map,
                                metadata_json=np.asarray(json.dumps({"units": "W/pixel", "QE": 0.5, "exposure_s": 0.001,
                                                                     "camera_gain_is_simulation_proxy": True, "noise_seed": 2026101610})))
        temp.replace(camera_archive_path)
        camera_archive = str(camera_archive_path)
        images_for_power[context_id] = uniform_power_map
        images_for_noise[context_id] = uniform_noise_map

    direct_checks: list[dict[str, Any]] = []
    if uniform_direct_check is not None:
        direct_checks.append(uniform_direct_check)

    jacobian: dict[str, Any] | None = None
    cross_window_uncertainty: dict[str, Any] | None = None
    direction_cache: dict[str, Any] = {}
    if precision == "complex128":
        primary_basis = basis_by_grid[4096]
        r4_primary = torch.from_numpy(primary_basis["camera_gl4"]).to(device=device)
        w4_primary = torch.from_numpy(primary_basis["weights4"]).to(device=device, dtype=torch.float64)
        jacobian = _camera_jacobian_report(
            config, run_dir, candidate, candidate_name, mode, route, decoder_state, 4096,
            r4_primary, w4_primary, candidate["matrices"][mode][4096]["H"],
            candidate["power_traces"][4096]["source_input_power_w"],
        )
        direction_cache = {
            "raw": jacobian.pop("right_directions"),
            "noise_1ms": jacobian.pop("noise_right_directions_1ms"),
        }
        if primary_c64_cache is not None:
            thresholds_1e3 = int(jacobian["exposure_sensitivity"][str(config["camera"]["primary_exposure_s"])]["directions_above_relative_1e_3"])
            noise_dirs = direction_cache["noise_1ms"]
            selected = [0, max(0, thresholds_1e3 - 1)] if thresholds_1e3 else [0, 0]
            selected_labels = ["strongest_retained", "weakest_retained_1e_3"]
            h_np = candidate["matrices"][mode][4096]["H"]
            for label, direction_index in zip(selected_labels, selected, strict=True):
                delta, feasible_scale = _scale_bounded_direction(noise_dirs[:, direction_index],
                                                                  float(config["measurement"]["input_perturbation_rms"]),
                                                                  0.25, 0.75, 0.5)
                command = torch.full((10000,), 0.5, dtype=torch.float64) + delta
                check = _direct_superposition_check(
                    candidate_name, candidate, mode, route, decoder_state, decoder_phases[decoder_state],
                    command, h_np,
                    torch.from_numpy(primary_c64_cache["basis_roi"]).to(device=device),
                    torch.from_numpy(primary_c64_cache["camera_gl4"]).to(device=device),
                    torch.from_numpy(primary_c64_cache["weights4"]).to(device=device, dtype=torch.float64),
                    device, label,
                )
                check["direction_index"] = direction_index
                check["box_feasible_scale"] = feasible_scale
                check["applied_rms"] = float(delta.square().mean().sqrt())
                direct_checks.append(check)
                del delta, command
        # Preserve directions outside the public JSON and save them as a
        # compact numeric sidecar for reproducibility.
        direction_path = run_dir / "camera_jacobians" / f"{context_id}_direction_basis.npz"
        np.savez_compressed(direction_path, raw_right_directions=direction_cache["raw"].numpy(),
                            noise_weighted_1ms_right_directions=direction_cache["noise_1ms"].numpy(),
                            metadata_json=np.asarray(json.dumps({"candidate": candidate_name, "mode": mode, "route": route, "decoder_state": decoder_state})))
        jacobian["direction_basis_archive"] = str(direction_path)

        # Window uncertainty of the pixel-power Jacobian is computed from the
        # compact factors; a 40000x10000 Jacobian is never allocated.
        factors = {grid: factor_records.get(grid) for grid in GRIDS}
        # The factor records are produced only by complex64 passes. If this is
        # the c128 pass, rebuild the three c64 factors from the matching saved
        # response basis retained by the paired call's cache.
        if primary_c64_cache is not None and "factors_by_grid" in primary_c64_cache:
            factors = primary_c64_cache["factors_by_grid"]
        pairs: dict[str, Any] = {}
        for left, right in ((4096, 6144), (4096, 8192), (6144, 8192)):
            a_l, b_l = factors[left]
            a_r, b_r = factors[right]
            a_diff = torch.cat((a_l, a_r), dim=1)
            b_diff = torch.cat((b_l, -b_r), dim=0)
            diff_spec, _dirs = _small_factor_spectrum(a_diff.to(device), b_diff.to(device), thresholds)
            base_spec, _base_dirs = _small_factor_spectrum(a_l.to(device), b_l.to(device), thresholds)
            absolute = float(diff_spec["maximum_singular_value"])
            relative = absolute / max(float(base_spec["maximum_singular_value"]), EPS)
            pairs[f"{left}_vs_{right}"] = {"operator_norm_difference_w_per_command": absolute,
                                            "relative_to_left_window_max_singular": relative,
                                            "difference_singular_values": diff_spec["singular_values"]}
        cross_window_uncertainty = {"pairs": pairs, "interpretation": "spectral uncertainty measured from cross-window factor differences; not a detector calibration"}

    matrix_gate = max(coefficient_prediction_by_grid.values(), default=math.inf) <= float(config["gates"]["matrix_prediction_relative_max"])
    public = {
        "candidate": candidate_name, "channels": candidate["size"] ** 2, "mode": mode,
        "route": route, "decoder_state": decoder_state, "precision": precision,
        "numerical_window_comparisons": window_result,
        "all_complex64_windows_pass": all_window_pass if precision == "complex64" else None,
        "all_patterns_pixel_integral_pass": all_quadrature_pass,
        "matrix_prediction_relative_error_by_grid": {str(k): v for k, v in coefficient_prediction_by_grid.items()},
        "patterns": records,
        "camera_pixel_archive": camera_archive,
        "direct_forward_superposition_checks": direct_checks,
        "camera_jacobian": jacobian,
        "cross_window_jacobian_uncertainty": cross_window_uncertainty,
        "interface_geometry": {
            "direct_distance_m": 1.5e-3,
            "explicit_relay_segments_m": [0.010, 0.015, 0.005] if route == "explicit_4f" else [],
            "explicit_relay_added_length_m": 0.030 if route == "explicit_4f" else 0.0,
            "existing_backend_relay_length_m": 0.350,
            "decoder_aperture_m": 200.0e-6,
            "camera_fov_m": 500.0e-6,
            "fiber_ids_preserve_original_order": True,
        },
        "valid_for_sensor_analysis": bool(precision == "complex64" and all_quadrature_pass and all_window_pass
                                           and matrix_gate and all(item["passed"] for item in direct_checks)),
        "matrix_prediction_gate_passed": matrix_gate,
        "complex128_diagnostic_valid": bool(precision == "complex128" and all_quadrature_pass and matrix_gate),
    }
    private = {
        "primary_c64_cache": None,
        "images_for_power": images_for_power,
        "images_for_noise": images_for_noise,
        "factors_by_grid": factor_records if precision == "complex64" else None,
    }
    if precision == "complex64":
        primary_basis = basis_by_grid[4096]
        private["primary_c64_cache"] = {
            "basis_roi": primary_basis["basis_roi"],
            "camera_gl4": primary_basis["camera_gl4"],
            "weights4": primary_basis["weights4"],
            "factors_by_grid": factor_records,
        }
    # Also save a small scalar-only group artifact; raw pixel maps and worst
    # complex fields have exact sidecar paths above.
    scalar_path = run_dir / "context_results" / f"{context_id}_{precision}.json"
    _write_json(scalar_path, public)
    public["scalar_result_archive"] = str(scalar_path)
    for mapping in (basis_roi_gpu, camera_center_gpu, response4_gpu, response8_gpu, weights4_gpu, weights8_gpu, grams_gpu):
        mapping.clear()
    del basis_by_grid
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return public, private


if __name__ == "__main__":
    main()



