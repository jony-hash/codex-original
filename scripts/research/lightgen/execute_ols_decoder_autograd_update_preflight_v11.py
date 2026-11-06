"""Three-step differentiable Decoder engineering acceptance for the frozen v9 OLS interface.

This is a research-only diagnostic. It imports the v9 physical operators but
does not call the detached v9 basis wrapper, read datasets, or enter a formal
training/evaluation path.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ldm_project.v2.optics.propagation import clear_production_transfer_cache
from scripts.research.lightgen import diagnose_ols_decoder_camera_interface_v9 as v9


SCHEMA = "ols_decoder_autograd_update_preflight.v11"
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_decoder_autograd_update_preflight_v11.yaml")
DEFAULT_RUN_DIR = Path("outputs/20261017_ols_decoder_autograd_update_preflight_v11/runs/attempt_01")
DEFAULT_SOURCE = Path("outputs/20261016_ols_decoder_camera_interface_v9/runs/attempt_02")
INPUT_NAMES = ("uniform_t_0.5", "binary_probe_00")
TRAIN_MODES = ("gaussian_s12", "lp01")
ROUTES = ("explicit_4f", "direct")
H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
EPS = 1.0e-30
_START = time.time()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} configuration")
    fixed = (
        (config["optics"]["wavelength_m"], 532e-9),
        (config["optics"]["dx_m"], 2e-6),
        (config["optics"]["source_power_w"], 249.3e-6),
        (config["optics"]["encoder_aperture_m"], 200e-6),
        (config["optics"]["encoder_layer_spacing_m"], 1.5e-3),
        (config["optics"]["encoder_to_fiber_distance_m"], 1.5e-3),
        (config["optics"]["encoder_layer_transmission_power"], 0.889),
        (config["optics"]["decoder_layer_transmission_power"], 0.889),
        (config["optics"]["fiber_pitch_m"], 125e-6),
        (config["camera"]["primary_exposure_s"], 1e-3),
        (config["measurement"]["learning_rate"], 1e-3),
        (config["gates"]["aligned_common_roi_complex64_error"], 1e-4),
        (config["routes"]["explicit_4f"]["magnification"], -0.5),
        (config["routes"]["direct"]["propagation_distance_m"], 1.5e-3),
        (config["backend_relay"]["magnification"], -2.5),
        (config["camera"]["quantum_efficiency"], 0.5),
        (config["camera"]["full_well_e"], 30000.0),
    )
    if any(not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-15) for actual, expected in fixed):
        raise ValueError("v11 fixed physical or acceptance parameters changed")
    if config["optics"]["grids"] != [4096, 6144, 8192] or config["measurement"]["update_steps"] != 3:
        raise ValueError("v11 grid or engineering update count changed")
    if (config["optics"]["decoder_shape"] != [100, 100]
            or config["optics"]["decoder_layers"] != 2
            or config["optics"]["decoder_pixel_pitch_m"] != 2e-6
            or config["optics"]["decoder_spacing_m"] != 1.5e-3
            or config["camera"]["shape"] != [200, 200]
            or config["camera"]["pixel_pitch_m"] != 2.5e-6
            or config["routes"]["explicit_4f"]["focal_lengths_m"] != [1.0e-2, 5.0e-3]
            or config["routes"]["explicit_4f"]["segments_m"] != [1.0e-2, 1.5e-2, 5.0e-3]
            or config["routes"]["explicit_4f"]["clear_aperture_diameter_m"] != 2.0e-3
            or config["routes"]["explicit_4f"]["lens_power_transmission"] != 0.98
            or config["backend_relay"]["segments_m"] != [0.1, 0.25]
            or config["backend_relay"]["focal_length_m"] != 0.0714285714285714
            or config["backend_relay"]["clear_aperture_m"] != 0.0266
            or config["measurement"]["input_patterns"] != list(INPUT_NAMES)):
        raise ValueError("v11 fixed Decoder, camera, or input-batch geometry changed")
    if config["scope"]["scientific_training_updates"] != 0:
        raise ValueError("v11 cannot authorize scientific training")
    return config


def _source_assets(root: Path, source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    root = root.resolve()
    if not source_dir.is_relative_to(root):
        raise ValueError("v9 evidence path must remain under the repository")
    report_path = source_dir / "report.json"
    response_path = source_dir / "frozen_v8_response_inputs_v9.npz"
    phase_path = source_dir / "decoder_phase_screens_v9.pt"
    if not all(path.is_file() for path in (report_path, response_path, phase_path)):
        raise FileNotFoundError("the explicitly selected v9 report, response archive, or phase archive is missing")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "ols_decoder_camera_interface.v9" or report.get("status") != "VALID_COMPLETED":
        raise ValueError("v9 source report is not a completed valid measurement")
    phase_bundle = torch.load(phase_path, map_location="cpu", weights_only=False)
    if phase_bundle.get("schema") != "ols_decoder_phase_screens.v9" or "fixed_low_frequency" not in phase_bundle.get("phases", {}):
        raise ValueError("v9 fixed decoder phase archive is malformed")
    phase = phase_bundle["phases"]["fixed_low_frequency"].to(torch.float64)
    if tuple(phase.shape) != (2, 1, 100, 100) or not torch.isfinite(phase).all():
        raise ValueError("v9 fixed_low_frequency phase has an invalid shape or value")

    with np.load(response_path, allow_pickle=False) as response:
        metadata = json.loads(str(response["metadata_json"].item()))
        candidates: dict[str, Any] = {}
        for name, size in v9.SIZES.items():
            names = response[f"{name}__pattern_names"].astype(str)
            commands = response[f"{name}__amplitude_commands"].astype(np.float64)
            coordinates = np.asarray(response[f"{name}__coordinates_yx_m"], dtype=np.float64)
            expected_coordinates = np.asarray(v9._coordinates(size), dtype=np.float64)
            if names.shape != (50,) or commands.shape != (50, 10000) or len(set(names.tolist())) != 50:
                raise ValueError(f"v9 pattern names or amplitude commands are invalid for {name}")
            if coordinates.shape != expected_coordinates.shape or not np.allclose(coordinates, expected_coordinates, atol=1e-12, rtol=0.0):
                raise ValueError(f"v9 channel coordinates or order changed for {name}")
            if not np.isfinite(commands).all() or not np.isfinite(coordinates).all():
                raise ValueError(f"v9 command or coordinate contains nonfinite values for {name}")
            by_grid: dict[int, dict[str, np.ndarray]] = {}
            for grid in v9.GRIDS:
                by_grid[grid] = {}
                for mode in v9.MODES:
                    h = np.asarray(response[f"{name}__{mode}__{grid}__H_sqrt_w"], dtype=np.complex128)
                    coefficients = np.asarray(response[f"{name}__{mode}__{grid}__coefficients_sqrt_w"], dtype=np.complex128)
                    if h.shape != (size * size, 10000) or coefficients.shape != (50, size * size):
                        raise ValueError(f"v9 response matrix or coefficients have invalid dimensions for {name}/{mode}/{grid}")
                    prediction = commands @ h.T
                    relative = np.linalg.norm(prediction - coefficients, axis=1) / np.maximum(np.linalg.norm(coefficients, axis=1), EPS)
                    if not np.isfinite(relative).all() or float(relative.max()) > 1e-4:
                        raise ValueError(f"v9 matrix does not reproduce saved coefficients for {name}/{mode}/{grid}")
                    by_grid[grid][mode] = coefficients.copy()
            camera: dict[str, dict[str, np.ndarray]] = {}
            camera_dir = source_dir / "camera_measurements"
            for state in ("zero_phase", "fixed_low_frequency"):
                filename = f"{name}__gaussian_s12__explicit_4f__{state}_4096_complex64.npz"
                camera_path = camera_dir / filename
                if not camera_path.is_file():
                    raise FileNotFoundError(f"v9 camera archive is missing: {camera_path}")
                with np.load(camera_path, allow_pickle=False) as saved:
                    camera_names = saved["pattern_names"].astype(str)
                    power = np.asarray(saved["power_w_per_pixel"], dtype=np.float64)
                    task = np.asarray(saved["task_output_1ms"], dtype=np.float64)
                if not np.array_equal(names, camera_names) or power.shape != (50, 200, 200) or task.shape != power.shape:
                    raise ValueError(f"v9 camera array identity/shape mismatch for {name}/{state}")
                camera[state] = {"power": power, "task": task}
            report_case = report.get("contexts", {}).get(f"{name}__gaussian_s12__explicit_4f__fixed_low_frequency", {})
            if not report_case.get("full_measurements", {}).get("complex64", {}).get("valid_for_sensor_analysis", False):
                raise ValueError(f"v9 fixed-low-frequency Gaussian 4f context is not valid for {name}")
            context_path = source_dir / "context_results" / f"{name}__gaussian_s12__explicit_4f__fixed_low_frequency_complex64.json"
            context = json.loads(context_path.read_text(encoding="utf-8"))
            worst_index = max(
                range(32),
                key=lambda index: max(
                    float(pair["pattern_metrics"][index]["common_global_phase_aligned_complex_relative_l2"])
                    for pair in context["numerical_window_comparisons"].values()
                ),
            )
            candidates[name] = {
                "size": size,
                "pattern_names": names.copy(),
                "commands": commands.copy(),
                "coordinates_yx_m": coordinates.copy(),
                "coefficients": by_grid,
                "camera": camera,
                "worst_binary_index": int(worst_index),
                "worst_binary_name": str(names[worst_index]),
            }
    if metadata.get("coefficient_units") != "sqrt(W)" or metadata.get("H_units") != "sqrt(W) per unit amplitude command":
        raise ValueError("v9 response archive lacks its unit metadata")
    indices = {}
    for name, candidate in candidates.items():
        names = candidate["pattern_names"].tolist()
        if names.count("uniform_t_0.5") != 1 or names.count("binary_probe_00") != 1:
            raise ValueError(f"v9 required input patterns are missing for {name}")
        indices[name] = {"uniform_t_0.5": names.index("uniform_t_0.5"), "binary_probe_00": names.index("binary_probe_00"),
                          "worst_saved_v9_binary": candidate["worst_binary_index"]}
    return {"source_dir": source_dir, "report": report, "source_metadata": metadata, "phases": phase,
            "candidates": candidates, "input_indices": indices}


def _front_cache_path(cache_dir: Path, candidate: str, mode: str, route: str, grid: int, pattern_name: str) -> Path:
    safe_name = pattern_name.replace("/", "_").replace("\\", "_")
    return cache_dir / f"{candidate}__{mode}__{route}__{grid}__{safe_name}.pt"


def _compose_fiber_field(candidate: dict[str, Any], mode: str, coefficients: np.ndarray, grid: int,
                         dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    field = torch.zeros((grid, grid), dtype=dtype, device=device)
    coefficients_tensor = torch.as_tensor(coefficients, dtype=dtype, device=device)
    for index, coordinate in enumerate(candidate["coordinates_yx_m"]):
        mode_field = v9._global_mode(mode, (float(coordinate[0]), float(coordinate[1])), grid, device).to(dtype=dtype)
        field = field + mode_field * coefficients_tensor[index]
        del mode_field
    return field


def _make_front_cache(assets: dict[str, Any], run_dir: Path, candidate_name: str, mode: str,
                      route: str, grid: int, pattern_index: int, dtype: torch.dtype,
                      device: torch.device) -> torch.Tensor:
    candidate = assets["candidates"][candidate_name]
    pattern_name = str(candidate["pattern_names"][pattern_index])
    cache_dir = run_dir / "frozen_frontend_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _front_cache_path(cache_dir, candidate_name, mode, route, grid, pattern_name)
    if path.exists():
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        expected = (candidate_name, mode, route, grid, pattern_name)
        actual = (bundle.get("candidate"), bundle.get("mode"), bundle.get("route"), bundle.get("grid"), bundle.get("pattern_name"))
        active = bundle.get("decoder_entrance_field")
        if actual != expected or not isinstance(active, torch.Tensor) or tuple(active.shape) != (100, 100):
            raise ValueError(f"cached frozen front-end tensor identity is invalid: {path}")
        return active.to(device=device, dtype=dtype)
    coefficient = candidate["coefficients"][grid][mode][pattern_index]
    with torch.no_grad():
        source = _compose_fiber_field(candidate, mode, coefficient, grid, dtype, device)
        active, trace, route_z = v9._route_to_decoder(source, route, grid)
        active = active.detach().cpu().contiguous()
        payload = {
            "candidate": candidate_name, "mode": mode, "route": route, "grid": grid,
            "pattern_index": int(pattern_index), "pattern_name": pattern_name,
            "coefficients_sqrt_w": coefficient, "decoder_entrance_field": active,
            "route_z_m": float(route_z), "route_trace_w": trace,
            "units": "sqrt(W)/m at decoder entrance; complex field; dx=2 um",
        }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    clear_production_transfer_cache()
    del source
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return active.to(device=device, dtype=dtype)


def _cached_active(assets: dict[str, Any], run_dir: Path, candidate_name: str, mode: str,
                   route: str, grid: int, pattern_index: int, dtype: torch.dtype,
                   device: torch.device) -> tuple[torch.Tensor, float]:
    active = _make_front_cache(assets, run_dir, candidate_name, mode, route, grid, pattern_index, dtype, device)
    pattern_name = str(assets["candidates"][candidate_name]["pattern_names"][pattern_index])
    path = _front_cache_path(run_dir / "frozen_frontend_cache", candidate_name, mode, route, grid, pattern_name)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    return active, float(bundle["route_z_m"])


def _camera_task(power_w_per_pixel: torch.Tensor, config: dict[str, Any], exposure_s: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    camera = config["camera"]
    exposure = float(camera["primary_exposure_s"] if exposure_s is None else exposure_s)
    photon_energy = H_PLANCK * C_LIGHT / float(config["optics"]["wavelength_m"])
    expected_electrons = power_w_per_pixel * (float(camera["quantum_efficiency"]) * exposure / photon_energy)
    task = torch.clamp(expected_electrons / float(camera["full_well_e"]), 0.0, 1.0)
    return expected_electrons, task


def _camera_forward(active: torch.Tensor, phase: torch.Tensor, grid: int, z_m: float,
                    config: dict[str, Any], *, quadrature_order: int = 4,
                    keep_roi: bool = False) -> dict[str, Any]:
    final_full, layer1, decoder_trace, exit_z = v9._run_decoder(active, phase, grid, z_m)
    roi = v9._crop_roi(final_full) if keep_roi else None
    exit_active = v9._crop_active(final_full)
    camera_field, camera_trace = v9._decoder_exit_camera(exit_active, exit_z)
    _sampled, pixel_power, weights = v9._camera_quadrature_field(camera_field, quadrature_order)
    expected_electrons, task = _camera_task(pixel_power, config)
    result = {"pixel_power_w": pixel_power, "expected_electrons": expected_electrons,
              "task_output": task, "camera_field": camera_field,
              "decoder_exit": exit_active, "layer1": layer1, "decoder_trace": decoder_trace,
              "camera_trace": camera_trace, "weights": weights}
    if roi is not None:
        result["common_roi"] = roi
    return result


def _decoder_roi_forward(active: torch.Tensor, phase: torch.Tensor, grid: int,
                         z_m: float) -> torch.Tensor:
    final_full, _layer1, _trace, _exit_z = v9._run_decoder(active, phase, grid, z_m)
    return v9._crop_roi(final_full)


def _relative_l2(left: torch.Tensor | np.ndarray, right: torch.Tensor | np.ndarray) -> float:
    a = torch.as_tensor(left, dtype=torch.float64).detach().cpu()
    b = torch.as_tensor(right, dtype=torch.float64).detach().cpu()
    return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(EPS))


def _loss_for_candidate(assets: dict[str, Any], run_dir: Path, candidate_name: str,
                        phase: torch.Tensor, config: dict[str, Any], device: torch.device,
                        precision: str = "complex64", *, with_grad: bool = False,
                        save_maps: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
    candidate = assets["candidates"][candidate_name]
    dtype = torch.complex128 if precision == "complex128" else torch.complex64
    phase_dtype = torch.float64 if precision == "complex128" else torch.float32
    phase = phase.to(device=device, dtype=phase_dtype)
    mode = "gaussian_s12"
    index_by_name = assets["input_indices"][candidate_name]
    losses: list[torch.Tensor] = []
    details: dict[str, Any] = {"patterns": {}}
    for pattern_name in INPUT_NAMES:
        pattern_index = index_by_name[pattern_name]
        active, z_m = _cached_active(assets, run_dir, candidate_name, mode, "explicit_4f", 4096,
                                     pattern_index, dtype, device)
        output = _camera_forward(active, phase, 4096, z_m, config, quadrature_order=4)
        target_power = torch.as_tensor(candidate["camera"]["zero_phase"]["power"][pattern_index],
                                       dtype=output["pixel_power_w"].dtype, device=device)
        _target_electrons, target = _camera_task(target_power, config)
        loss = torch.mean((output["task_output"] - target).square())
        losses.append(loss)
        if save_maps:
            details["patterns"][pattern_name] = {
                "power_w_per_pixel": output["pixel_power_w"].detach().cpu(),
                "expected_electrons": output["expected_electrons"].detach().cpu(),
                "task_output": output["task_output"].detach().cpu(),
                "target_task_output": target.detach().cpu(),
                "decoder_trace": output["decoder_trace"], "camera_trace": output["camera_trace"],
            }
        del output, active, target_power
        clear_production_transfer_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    total = torch.stack(losses).mean()
    details["loss"] = float(total.detach().cpu())
    return total, details


def _loss_grad_complex128(assets: dict[str, Any], run_dir: Path, candidate_name: str,
                          phase_values: torch.Tensor, config: dict[str, Any], device: torch.device) -> tuple[float, torch.Tensor]:
    phase = phase_values.detach().to(device=device, dtype=torch.float64).clone().requires_grad_(True)
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    accumulated = torch.zeros_like(phase)
    candidate = assets["candidates"][candidate_name]
    for pattern_name in INPUT_NAMES:
        pattern_index = assets["input_indices"][candidate_name][pattern_name]
        active, z_m = _cached_active(assets, run_dir, candidate_name, "gaussian_s12", "explicit_4f",
                                     4096, pattern_index, torch.complex128, device)
        output = _camera_forward(active, phase, 4096, z_m, config, quadrature_order=4)
        target_power = torch.as_tensor(candidate["camera"]["zero_phase"]["power"][pattern_index],
                                       dtype=torch.float64, device=device)
        _mu, target = _camera_task(target_power, config)
        loss = torch.mean((output["task_output"] - target).square()) / len(INPUT_NAMES)
        grad = torch.autograd.grad(loss, phase, retain_graph=False, create_graph=False)[0]
        accumulated.add_(grad.detach())
        total_loss = total_loss + loss.detach()
        del output, active, target_power, grad, loss
        clear_production_transfer_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return float(total_loss.detach().cpu()), accumulated.detach()


def _loss_value_complex128(assets: dict[str, Any], run_dir: Path, candidate_name: str,
                           phase_values: torch.Tensor, config: dict[str, Any], device: torch.device) -> float:
    phase = phase_values.to(device=device, dtype=torch.float64)
    with torch.no_grad():
        loss, _ = _loss_for_candidate(assets, run_dir, candidate_name, phase, config, device,
                                      precision="complex128", with_grad=False, save_maps=False)
    return float(loss.detach().cpu())


def _wrong_transmission_forward(active: torch.Tensor, phase: torch.Tensor, grid: int, z_m: float,
                                amplitude: float) -> dict[str, Any]:
    area = v9.DX * v9.DX
    current_z = z_m
    screen0 = phase[0, 0].to(device=active.device, dtype=active.real.dtype)
    mod0 = torch.polar(torch.full_like(screen0, amplitude), screen0).to(active.dtype)
    first, current_z = v9._propagate(torch.nn.functional.pad(active * mod0, ((grid - 100) // 2,) * 4),
                                     grid, current_z, 1.5e-3, "v11_negative_first")
    first = v9._crop_active(first)
    screen1 = phase[1, 0].to(device=active.device, dtype=active.real.dtype)
    mod1 = torch.polar(torch.full_like(screen1, amplitude), screen1).to(active.dtype)
    final, current_z = v9._propagate(torch.nn.functional.pad(first * mod1, ((grid - 100) // 2,) * 4),
                                     grid, current_z, 1.5e-3, "v11_negative_final")
    camera_field, _trace = v9._decoder_exit_camera(v9._crop_active(final), current_z)
    _sampled, pixel, _weights = v9._camera_quadrature_field(camera_field, 4)
    return {"final": final, "pixel_power_w": pixel, "area": area}


def _wrong_port_order_map(assets: dict[str, Any], candidate_name: str, pattern_index: int,
                          phase: torch.Tensor, config: dict[str, Any], device: torch.device) -> torch.Tensor:
    candidate = assets["candidates"][candidate_name]
    dtype = torch.complex64
    coefficients = candidate["coefficients"][4096]["gaussian_s12"][pattern_index]
    permuted = coefficients[::-1].copy()
    field = _compose_fiber_field(candidate, "gaussian_s12", permuted, 4096, dtype, device)
    with torch.no_grad():
        active, _trace, z_m = v9._route_to_decoder(field, "explicit_4f", 4096)
        output = _camera_forward(active, phase, 4096, z_m, config, quadrature_order=4)
        result = output["pixel_power_w"].detach()
    del field, active, output
    clear_production_transfer_cache()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _preflight(assets: dict[str, Any], config: dict[str, Any], config_path: Path,
               run_dir: Path, device: torch.device) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    v9_phase = assets["phases"].to(torch.float32)
    torch.save({"schema": f"{SCHEMA}.initial_phase", "source_archive": str(assets["source_dir"] / "decoder_phase_screens_v9.pt"),
                "phase_state": "fixed_low_frequency", "phase_unwrapped": assets["phases"]}, run_dir / "initial_decoder_phase_v9.pt")
    (run_dir / "configuration_v11.yaml").write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    for candidate_name, candidate in assets["candidates"].items():
        target_indices = [assets["input_indices"][candidate_name][pattern] for pattern in INPUT_NAMES]
        np.savez_compressed(run_dir / f"{candidate_name}_fixed_zero_phase_targets.npz",
                            pattern_names=np.asarray(INPUT_NAMES), pattern_indices=np.asarray(target_indices, dtype=np.int64),
                            commands=candidate["commands"][target_indices].astype(np.float32),
                            target_power_w_per_pixel=candidate["camera"]["zero_phase"]["power"][target_indices].astype(np.float32),
                            target_task_output_1ms=candidate["camera"]["zero_phase"]["task"][target_indices].astype(np.float32))
    results: dict[str, Any] = {"source_run_dir": str(assets["source_dir"]), "candidate_checks": {},
                               "negative_controls": {}, "scope": config["scope"]}
    gates = config["gates"]
    all_pass = True
    for candidate_name, candidate in assets["candidates"].items():
        indices = assets["input_indices"][candidate_name]
        value_checks: dict[str, Any] = {}
        initial_loss, _ = _loss_for_candidate(assets, run_dir, candidate_name, v9_phase, config,
                                               device, precision="complex64", save_maps=True)
        for pattern_name in INPUT_NAMES:
            pattern_index = indices[pattern_name]
            active, z_m = _cached_active(assets, run_dir, candidate_name, "gaussian_s12", "explicit_4f",
                                         4096, pattern_index, torch.complex64, device)
            with torch.no_grad():
                computed = _camera_forward(active, v9_phase, 4096, z_m, config, quadrature_order=4)
                saved_power = candidate["camera"]["fixed_low_frequency"]["power"][pattern_index]
                saved_task = candidate["camera"]["fixed_low_frequency"]["task"][pattern_index]
                value_checks[pattern_name] = {
                    "w_per_pixel_relative_l2": _relative_l2(computed["pixel_power_w"], saved_power),
                    "task_output_relative_l2": _relative_l2(computed["task_output"], saved_task),
                    "finite": bool(torch.isfinite(computed["pixel_power_w"]).all() and torch.isfinite(computed["task_output"]).all()),
                }
                sampled8, power8, weights8 = v9._camera_quadrature_field(computed["camera_field"], 8)
                value_checks[pattern_name]["gl4_vs_gl8_power_relative_l2"] = _relative_l2(computed["pixel_power_w"], power8)
                value_checks[pattern_name]["gl4_vs_gl8_max_abs_w"] = float((computed["pixel_power_w"] - power8).abs().max().cpu())
                value_checks[pattern_name]["decoder_trace_w"] = computed["decoder_trace"]
                value_checks[pattern_name]["camera_trace"] = computed["camera_trace"]
                if value_checks[pattern_name]["w_per_pixel_relative_l2"] > gates["forward_relative_error"] or value_checks[pattern_name]["task_output_relative_l2"] > gates["forward_relative_error"] or value_checks[pattern_name]["gl4_vs_gl8_power_relative_l2"] > gates["forward_relative_error"]:
                    all_pass = False
            del computed, active, sampled8, power8, weights8
            clear_production_transfer_cache()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        phase0 = v9_phase.to(device=device, dtype=torch.float64)
        loss0, gradient = _loss_grad_complex128(assets, run_dir, candidate_name, phase0, config, device)
        layer_norms = torch.linalg.vector_norm(gradient.reshape(2, -1), dim=1)
        if not torch.isfinite(gradient).all() or torch.any(layer_norms <= 0):
            all_pass = False
        gradient_checks: list[dict[str, Any]] = []
        generator = torch.Generator(device="cpu").manual_seed(int(config["measurement"]["finite_difference_seed"]) + candidate["size"])
        random_direction = torch.randn((2, 1, 100, 100), generator=generator, dtype=torch.float64).to(device)
        for layer in range(2):
            rms = torch.sqrt(torch.mean(random_direction[layer].square())).clamp_min(EPS)
            random_direction[layer] = random_direction[layer] / rms
        for layer in range(2):
            neg_direction = torch.zeros_like(gradient)
            layer_grad = gradient[layer]
            grad_rms = torch.sqrt(torch.mean(layer_grad.square())).clamp_min(EPS)
            neg_direction[layer] = -layer_grad / grad_rms
            layer_mask = torch.nn.functional.one_hot(torch.tensor(layer), 2).to(device=device, dtype=torch.float64)[:, None, None, None]
            directions = [(f"layer{layer}_fixed_random", random_direction.clone() * layer_mask),
                          (f"layer{layer}_normalized_negative_gradient", neg_direction)]
            for label, direction in directions:
                analytic = float(torch.sum(gradient * direction).detach().cpu())
                negative_descent = label.endswith("normalized_negative_gradient")
                finite_values = []
                for delta in config["measurement"]["finite_difference_steps_rad"]:
                    plus = _loss_value_complex128(assets, run_dir, candidate_name, phase0 + float(delta) * direction, config, device)
                    minus = _loss_value_complex128(assets, run_dir, candidate_name, phase0 - float(delta) * direction, config, device)
                    finite = (plus - minus) / (2.0 * float(delta))
                    tolerance = float(gates["gradient_relative_error"]) * abs(finite) + float(gates["gradient_loss_scale_absolute"]) * loss0
                    passed = math.isfinite(finite) and abs(analytic - finite) <= tolerance
                    finite_values.append({"step_rad": float(delta), "finite_difference": finite, "absolute_error": abs(analytic - finite),
                                          "tolerance": tolerance, "passed": passed})
                    all_pass &= passed
                gradient_checks.append({"direction": label, "analytic_directional_derivative": analytic,
                                        "finite_difference_step_checks": finite_values,
                                        "step_size_estimate_difference": abs(finite_values[0]["finite_difference"] - finite_values[1]["finite_difference"]),
                                        "negative_gradient_descent_direction": negative_descent,
                                        "descent_sign_passed": (not negative_descent) or (analytic < 0 and all(item["finite_difference"] < 0 for item in finite_values))})
                if negative_descent and not gradient_checks[-1]["descent_sign_passed"]:
                    all_pass = False
        candidate_check = {
            "value_and_units": value_checks,
            "initial_engineering_loss": float(initial_loss.detach().cpu()),
            "complex128_initial_loss": loss0,
            "phase_gradient_l2_by_layer": layer_norms.detach().cpu(),
            "phase_gradient_finite_nonzero_by_layer": [bool(torch.isfinite(value) and value > 0) for value in layer_norms],
            "gradient_checks": gradient_checks,
        }

        # Actual full-chain counterfactuals. Each wrong path is compared with
        # the saved fixed-low-frequency camera map from the same physical input.
        pattern_index = indices["binary_probe_00"]
        saved_map = candidate["camera"]["fixed_low_frequency"]["power"][pattern_index]
        active, z_m = _cached_active(assets, run_dir, candidate_name, "gaussian_s12", "explicit_4f", 4096,
                                     pattern_index, torch.complex64, device)
        with torch.no_grad():
            baseline = _camera_forward(active, v9_phase, 4096, z_m, config, quadrature_order=4)
        with torch.enable_grad():
            detached_phase = v9_phase.to(device).clone().requires_grad_(True)
            source = _compose_fiber_field(candidate, "gaussian_s12",
                                          candidate["coefficients"][4096]["gaussian_s12"][pattern_index],
                                          4096, torch.complex64, device)
            detached_basis = v9._basis_forward(source, "explicit_4f", detached_phase, 4096, "complex64")
            detached_pixels = v9._integrate_subsamples(detached_basis["camera_gl4"], 4, detached_basis["gl4_weights"])
            detached_task = _camera_task(detached_pixels, config)[1]
            detached_loss = torch.mean((detached_task - detached_task.new_zeros(detached_task.shape)).square())
            try:
                torch.autograd.grad(detached_loss, detached_phase, allow_unused=True)
                detached_rejected = False
            except RuntimeError:
                detached_rejected = True
        with torch.no_grad():
            sampled4, _pixels4, weights4 = v9._camera_quadrature_field(baseline["camera_field"], 4)
            no_area = v9._integrate_subsamples(sampled4, 4, weights4) / (2.5e-6**2)
            no_area_task = _camera_task(no_area, config)[1]
            no_area_error = _relative_l2(no_area, saved_map)
            normalized_task = _camera_task(baseline["pixel_power_w"] / baseline["pixel_power_w"].max().clamp_min(EPS), config)[1]
            normalize_error = _relative_l2(normalized_task, candidate["camera"]["fixed_low_frequency"]["task"][pattern_index])
            wrong_tx = _wrong_transmission_forward(active, v9_phase.to(device), 4096, z_m, v9.TRANSMISSION)
            wrong_tx_error = _relative_l2(wrong_tx["pixel_power_w"], saved_map)
            wrong_order = _wrong_port_order_map(assets, candidate_name, pattern_index, v9_phase.to(device), config, device)
            wrong_order_error = _relative_l2(wrong_order, saved_map)
            true_power = baseline["pixel_power_w"]
            sat_limit = float(config["camera"]["full_well_e"]) / (float(config["camera"]["quantum_efficiency"]) * float(config["camera"]["primary_exposure_s"]) / (H_PLANCK * C_LIGHT / v9.WAVELENGTH))
            boundary = torch.tensor([sat_limit * (1.0 - 1e-6), sat_limit, sat_limit * (1.0 + 1e-6)], dtype=torch.float64, device=device)
            boundary_task = _camera_task(boundary, config)[1]
            boundary_check = {"power_w_per_pixel": boundary.detach().cpu(), "task_output": boundary_task.detach().cpu(),
                              "below_to_at_left_slope": float(((boundary_task[1] - boundary_task[0]) / (boundary[1] - boundary[0])).cpu()),
                              "above_to_at_right_slope": float(((boundary_task[2] - boundary_task[1]) / (boundary[2] - boundary[1])).cpu())}
            negative_controls = {
                "detached_v9_style_consumer_rejects_gradient": bool(detached_rejected),
                "omitted_pixel_area_detected_by_saved_w_per_pixel": bool(no_area_error > gates["forward_relative_error"]),
                "power_transmission_used_as_amplitude_detected": bool(wrong_tx_error > gates["forward_relative_error"]),
                "wrong_fiber_port_order_detected": bool(wrong_order_error > gates["forward_relative_error"]),
                "per_sample_max_normalization_detected": bool(normalize_error > gates["forward_relative_error"]),
                "omitted_area_relative_error": no_area_error,
                "wrong_amplitude_transmission_relative_error": wrong_tx_error,
                "wrong_fiber_order_relative_error": wrong_order_error,
                "per_sample_normalization_task_relative_error": normalize_error,
                "clamp_boundary_one_sided_values": boundary_check,
            }
        required_negative = ("detached_v9_style_consumer_rejects_gradient", "omitted_pixel_area_detected_by_saved_w_per_pixel",
                            "power_transmission_used_as_amplitude_detected", "wrong_fiber_port_order_detected",
                            "per_sample_max_normalization_detected")
        if not all(negative_controls[key] for key in required_negative):
            all_pass = False
        candidate_check["negative_controls"] = negative_controls
        candidate_check["passed"] = bool(all(negative_controls[key] for key in required_negative)
                                           and all(item["finite"] for item in value_checks.values())
                                           and all(item["w_per_pixel_relative_l2"] <= gates["forward_relative_error"] and item["task_output_relative_l2"] <= gates["forward_relative_error"] and item["gl4_vs_gl8_power_relative_l2"] <= gates["forward_relative_error"] for item in value_checks.values())
                                           and all(check["passed"] for group in gradient_checks for check in group["finite_difference_step_checks"])
                                           and all(check["descent_sign_passed"] for check in gradient_checks)
                                           and all(candidate_check["phase_gradient_finite_nonzero_by_layer"]))

        # Common-piston control: two equal phase offsets change only one global
        # complex phase, leaving every integrated sensor pixel unchanged.
        with torch.no_grad():
            piston = 0.2
            base = _camera_forward(active, v9_phase, 4096, z_m, config, quadrature_order=4, keep_roi=True)
            shifted = _camera_forward(active, v9_phase + piston, 4096, z_m, config, quadrature_order=4, keep_roi=True)
            phase_ratio = torch.sum(base["common_roi"].conj() * shifted["common_roi"])
            global_phase = float(torch.angle(phase_ratio).cpu())
            phase_scalar = torch.as_tensor(-1j * global_phase, dtype=base["common_roi"].dtype, device=device)
            phase_aligned = shifted["common_roi"] * torch.exp(phase_scalar)
            piston_aligned_error = float(torch.linalg.vector_norm(base["common_roi"] - phase_aligned) / torch.linalg.vector_norm(base["common_roi"]).clamp_min(EPS))
            piston_power_error = _relative_l2(base["pixel_power_w"], shifted["pixel_power_w"])
        candidate_check["common_phase_piston"] = {"phase_per_decoder_screen_rad": piston,
            "measured_common_output_phase_rad": global_phase, "expected_common_phase_rad": 2 * piston,
            "phase_aligned_complex_error": piston_aligned_error, "pixel_power_relative_error": piston_power_error,
            "passed": abs(global_phase - 2 * piston) < 1e-4 and piston_power_error < 1e-4 and piston_aligned_error < 1e-4}
        all_pass &= candidate_check["common_phase_piston"]["passed"]
        left_slope_expected = float(config["camera"]["quantum_efficiency"]) * float(config["camera"]["primary_exposure_s"]) / (H_PLANCK * C_LIGHT / v9.WAVELENGTH) / float(config["camera"]["full_well_e"])
        boundary_pass = abs(boundary_check["below_to_at_left_slope"] - left_slope_expected) <= 1e-4 * left_slope_expected and abs(boundary_check["above_to_at_right_slope"]) <= 1e-10
        candidate_check["negative_controls"]["clamp_boundary_one_sided_passed"] = bool(boundary_pass)
        candidate_check["passed"] = bool(candidate_check["passed"] and boundary_pass)
        all_pass &= boundary_pass
        initial_windows = _window_checks(assets, config, run_dir, candidate_name, v9_phase, 0, INPUT_NAMES, device)
        candidate_check["step_0_window_checks"] = initial_windows
        candidate_check["passed"] = bool(candidate_check["passed"] and initial_windows["passed"])
        all_pass &= initial_windows["passed"]
        results["candidate_checks"][candidate_name] = candidate_check
        del active, baseline, wrong_tx, wrong_order, no_area_task, normalized_task, base, shifted, gradient, source, detached_basis, detached_pixels, detached_task
        clear_production_transfer_cache()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    results["passed"] = bool(all_pass and all(value["passed"] for value in results["candidate_checks"].values()))
    results["status"] = "PREFLIGHT_PASS_UPDATES_PENDING" if results["passed"] else "PREFLIGHT_BLOCKED_ENGINEERING"
    results["elapsed_seconds"] = time.time() - _START
    results["device"] = str(device)
    results["config_path"] = str(config_path)
    results["v9_source_run_dir"] = str(assets["source_dir"])
    results["input_pattern_indices"] = assets["input_indices"]
    results["worst_saved_binary_patterns"] = {name: candidate["worst_binary_name"] for name, candidate in assets["candidates"].items()}
    _write_json(run_dir / "preflight.json", results)
    return results


def _window_checks(assets: dict[str, Any], config: dict[str, Any], run_dir: Path,
                   candidate_name: str, phase: torch.Tensor, step: int,
                   pattern_names: tuple[str, ...], device: torch.device) -> dict[str, Any]:
    candidate = assets["candidates"][candidate_name]
    checks: dict[str, Any] = {"candidate": candidate_name, "step": int(step), "patterns": {}}
    all_pass = True
    for pattern_name in pattern_names:
        pattern_index = assets["input_indices"][candidate_name][pattern_name] if pattern_name in assets["input_indices"][candidate_name] else int(np.where(candidate["pattern_names"] == pattern_name)[0][0])
        fields: dict[int, torch.Tensor] = {}
        for grid in config["optics"]["grids"]:
            active, z_m = _cached_active(assets, run_dir, candidate_name, "gaussian_s12", "explicit_4f", int(grid),
                                         pattern_index, torch.complex64, device)
            with torch.no_grad():
                roi = _decoder_roi_forward(active, phase.to(device=device, dtype=torch.float32), int(grid), z_m)
                fields[int(grid)] = roi.detach().cpu()
            del active, roi
            clear_production_transfer_cache()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        pair_metrics: dict[str, Any] = {}
        grids = [int(value) for value in config["optics"]["grids"]]
        for i in range(len(grids)):
            for j in range(i + 1, len(grids)):
                left, right = grids[i], grids[j]
                metric = v9._window_metrics(fields[left], fields[right])
                metric["passed"] = metric["common_global_phase_aligned_complex_relative_l2"] <= float(config["gates"]["aligned_common_roi_complex64_error"])
                pair_metrics[f"{left}_vs_{right}"] = metric
                all_pass &= bool(metric["passed"])
        checks["patterns"][pattern_name] = {"pairwise_common_roi_metrics": pair_metrics,
                                                "all_pairwise_complex64_checks_pass": all(item["passed"] for item in pair_metrics.values())}
        del fields
        gc.collect()
    checks["passed"] = bool(all_pass)
    return checks


def _rng_state() -> dict[str, Any]:
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _save_checkpoint(path: Path, candidate_name: str, phase: torch.Tensor,
                     optimizer: torch.optim.Optimizer, step: int, config: dict[str, Any],
                     history: list[dict[str, Any]], rng_state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": f"{SCHEMA}.checkpoint", "candidate": candidate_name,
                "step": int(step), "phase_unwrapped": phase.detach().cpu(),
                "optimizer_state": optimizer.state_dict(), "config": config,
                "input_order": list(INPUT_NAMES), "rng_state": rng_state,
                "history": history}, path)


def _train_candidate(assets: dict[str, Any], config: dict[str, Any], run_dir: Path,
                     candidate_name: str, device: torch.device, resume_path: Path | None = None) -> dict[str, Any]:
    candidate = assets["candidates"][candidate_name]
    output_dir = run_dir / "optimization" / candidate_name
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_dir = output_dir if resume_path is None else output_dir / "independent_resume"
    checkpoint_dir = trajectory_dir / "checkpoints"
    if resume_path is None:
        phase = assets["phases"].clone().to(device=device, dtype=torch.float32).requires_grad_(True)
        optimizer = torch.optim.Adam([phase], lr=float(config["measurement"]["learning_rate"]),
                                     betas=tuple(config["measurement"]["betas"]),
                                     eps=float(config["measurement"]["epsilon"]),
                                     weight_decay=float(config["measurement"]["weight_decay"]))
        start_step = 0
        history: list[dict[str, Any]] = []
        _save_checkpoint(checkpoint_dir / "checkpoint_step_000.pt", candidate_name, phase, optimizer, 0, config, history, _rng_state())
    else:
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != f"{SCHEMA}.checkpoint" or payload.get("candidate") != candidate_name or int(payload.get("step", -1)) != 1:
            raise ValueError("resume checkpoint identity or step is invalid")
        if payload.get("input_order") != list(INPUT_NAMES) or payload.get("config") != config:
            raise ValueError("resume checkpoint does not match fixed v11 inputs/configuration")
        phase = payload["phase_unwrapped"].to(device=device, dtype=torch.float32).clone().requires_grad_(True)
        optimizer = torch.optim.Adam([phase], lr=float(config["measurement"]["learning_rate"]),
                                     betas=tuple(config["measurement"]["betas"]),
                                     eps=float(config["measurement"]["epsilon"]),
                                     weight_decay=float(config["measurement"]["weight_decay"]))
        optimizer.load_state_dict(payload["optimizer_state"])
        torch.set_rng_state(payload["rng_state"]["cpu"])
        if device.type == "cuda" and payload["rng_state"]["cuda"]:
            torch.cuda.set_rng_state_all(payload["rng_state"]["cuda"])
        start_step = 1
        history = list(payload["history"])
    state_metrics: dict[str, Any] = {}
    if start_step == 0:
        preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
        state_metrics["step_0"] = preflight["candidate_checks"][candidate_name]["step_0_window_checks"]
    for step in range(start_step + 1, int(config["measurement"]["update_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, details = _loss_for_candidate(assets, run_dir, candidate_name, phase, config, device,
                                            precision="complex64", with_grad=True, save_maps=True)
        loss.backward()
        if phase.grad is None or not torch.isfinite(phase.grad).all():
            raise RuntimeError(f"Decoder phase gradient is absent or nonfinite at {candidate_name} step {step}")
        grad_norm_by_layer = torch.linalg.vector_norm(phase.grad.detach().reshape(2, -1), dim=1).cpu()
        previous = phase.detach().clone()
        optimizer.step()
        phase_delta = phase.detach() - previous
        update_norm_by_layer = torch.linalg.vector_norm(phase_delta.reshape(2, -1), dim=1).cpu()
        if torch.any(grad_norm_by_layer <= 0) or torch.any(update_norm_by_layer <= 0) or not torch.isfinite(update_norm_by_layer).all():
            raise RuntimeError(f"both Decoder phase screens must receive finite nonzero gradients and updates at {candidate_name} step {step}")
        record = {"step": step, "loss_before_update": float(loss.detach().cpu()),
                  "loss_by_pattern_before_update": {name: float(torch.mean((details["patterns"][name]["task_output"] - details["patterns"][name]["target_task_output"]).square()) / len(INPUT_NAMES)) for name in INPUT_NAMES},
                  "phase_gradient_l2_by_layer": grad_norm_by_layer,
                  "phase_update_l2_by_layer": update_norm_by_layer,
                  "phase_rms_rad_by_layer_after_update": torch.sqrt(torch.mean(phase.detach().square(), dim=(-2, -1))).cpu(),
                  "adam_step_by_parameter": [int(state["step"].item()) for state in optimizer.state.values()],
                  "parameters_updated": ["decoder_layer_0_phase", "decoder_layer_1_phase"],
                  "frozen_parameters_updated": [],
                  "physical_power_w_by_pattern_before_update": {name: float(details["patterns"][name]["power_w_per_pixel"].sum()) for name in INPUT_NAMES},
                  "task_output_mean_by_pattern_before_update": {name: float(details["patterns"][name]["task_output"].mean()) for name in INPUT_NAMES}}
        history.append(record)
        checkpoint = checkpoint_dir / f"checkpoint_step_{step:03d}.pt"
        _save_checkpoint(checkpoint, candidate_name, phase, optimizer, step, config, history, _rng_state())
        state_metrics[f"step_{step}"] = _window_checks(assets, config, run_dir, candidate_name,
                                                         phase.detach(), step, INPUT_NAMES, device)
        _write_json(trajectory_dir / "progress.json", {"candidate": candidate_name, "current_step": step,
                                                    "planned_steps": 3, "window_metrics": state_metrics,
                                                    "history": history, "checkpoint": str(checkpoint)})
        del loss, details, phase_delta, previous
        gc.collect()
        clear_production_transfer_cache()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    endpoint_loss, endpoint_details = _loss_for_candidate(assets, run_dir, candidate_name, phase.detach(), config,
                                                          device, precision="complex64", save_maps=True)
    for pattern_name, maps in endpoint_details["patterns"].items():
        np.savez_compressed(trajectory_dir / f"step_003_{pattern_name}_camera.npz",
                            power_w_per_pixel=maps["power_w_per_pixel"].numpy().astype(np.float32),
                            expected_electrons=maps["expected_electrons"].numpy().astype(np.float32),
                            task_output=maps["task_output"].numpy().astype(np.float32),
                            target_task_output=maps["target_task_output"].numpy().astype(np.float32))
    result = {"candidate": candidate_name, "status": "THREE_STEP_ENGINEERING_COMPLETE",
              "last_step": int(start_step + (3 - start_step)), "endpoint_loss": float(endpoint_loss.detach().cpu()),
              "history": history, "window_metrics": state_metrics,
              "step1_checkpoint": str(output_dir / "checkpoints" / "checkpoint_step_001.pt"),
              "final_checkpoint": str(checkpoint_dir / "checkpoint_step_003.pt"),
              "loaded_from_checkpoint": str(resume_path) if resume_path else None,
              "adam_call_count": 3 - start_step}
    terminal_name = "resume_terminal.pt" if resume_path else "continuous_terminal.pt"
    torch.save({"candidate": candidate_name, "step": 3,
                "phase_unwrapped": phase.detach().cpu(), "optimizer_state": optimizer.state_dict(),
                "rng_state": _rng_state(), "input_order": list(INPUT_NAMES), "config": config},
               trajectory_dir / terminal_name)
    _write_json(output_dir / ("resume_result.json" if resume_path else "continuous_result.json"), result)
    return result


def _tree_compare(left: Any, right: Any, atol: float, rtol: float, prefix: str = "") -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        a, b = left.detach().cpu(), right.detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype or not torch.allclose(a, b, atol=atol, rtol=rtol):
            max_error = float((a.to(torch.float64) - b.to(torch.float64)).abs().max()) if a.numel() == b.numel() else math.inf
            differences.append({"key": prefix, "left_shape": list(a.shape), "right_shape": list(b.shape), "max_abs_error": max_error})
    elif isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            differences.append({"key": prefix, "reason": "dictionary_keys_differ"})
        else:
            for key in left:
                differences.extend(_tree_compare(left[key], right[key], atol, rtol, f"{prefix}.{key}"))
    elif isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            differences.append({"key": prefix, "reason": "sequence_lengths_differ"})
        else:
            for index, (a, b) in enumerate(zip(left, right, strict=True)):
                differences.extend(_tree_compare(a, b, atol, rtol, f"{prefix}[{index}]"))
    elif isinstance(left, (float, int, np.number)) and isinstance(right, (float, int, np.number)):
        if not math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol):
            differences.append({"key": prefix, "left": float(left), "right": float(right)})
    elif left != right:
        differences.append({"key": prefix, "left": str(left), "right": str(right)})
    return differences


def _camera_endpoint_for_checkpoint(assets: dict[str, Any], config: dict[str, Any], run_dir: Path,
                                    candidate_name: str, checkpoint_path: Path, device: torch.device) -> torch.Tensor:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    phase = payload["phase_unwrapped"].to(device=device, dtype=torch.float32)
    candidate = assets["candidates"][candidate_name]
    index = assets["input_indices"][candidate_name]["uniform_t_0.5"]
    active, z_m = _cached_active(assets, run_dir, candidate_name, "gaussian_s12", "explicit_4f", 4096,
                                 index, torch.complex64, device)
    with torch.no_grad():
        output = _camera_forward(active, phase, 4096, z_m, config, quadrature_order=4)
        return output["task_output"].detach().cpu()


def _run_continuity_and_resume(assets: dict[str, Any], config: dict[str, Any], config_path: Path,
                               run_dir: Path, device: torch.device) -> dict[str, Any]:
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("--run requires a completed preflight.json in the same explicit run directory")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if not preflight.get("passed") or preflight.get("status") != "PREFLIGHT_PASS_UPDATES_PENDING":
        raise ValueError("the saved preflight did not pass; refusing any engineering Adam update")
    records: dict[str, Any] = {"candidates": {}, "expected_engineering_adam_calls": 10}
    for candidate_name in config["measurement"]["candidates"]:
        continuous_path = run_dir / "optimization" / candidate_name / "continuous_result.json"
        if continuous_path.is_file():
            continuous = json.loads(continuous_path.read_text(encoding="utf-8"))
            trajectory_dir = run_dir / "optimization" / candidate_name
            expected_step1 = trajectory_dir / "checkpoints" / "checkpoint_step_001.pt"
            expected_final = trajectory_dir / "checkpoints" / "checkpoint_step_003.pt"
            expected_terminal = trajectory_dir / "continuous_terminal.pt"
            if (continuous.get("candidate") != candidate_name
                    or continuous.get("status") != "THREE_STEP_ENGINEERING_COMPLETE"
                    or continuous.get("last_step") != 3 or continuous.get("adam_call_count") != 3
                    or not expected_step1.is_file() or not expected_final.is_file() or not expected_terminal.is_file()
                    or Path(continuous.get("step1_checkpoint", "")).resolve() != expected_step1.resolve()
                    or Path(continuous.get("final_checkpoint", "")).resolve() != expected_final.resolve()
                    or set(continuous.get("window_metrics", {})) != {"step_0", "step_1", "step_2", "step_3"}
                    or not all(value.get("passed", False) for value in continuous["window_metrics"].values())):
                raise ValueError(f"saved continuous trajectory is incomplete or invalid for {candidate_name}")
            for step, checkpoint_path in ((1, expected_step1), (3, expected_final)):
                checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if (checkpoint_state.get("schema") != f"{SCHEMA}.checkpoint"
                        or checkpoint_state.get("candidate") != candidate_name
                        or int(checkpoint_state.get("step", -1)) != step
                        or checkpoint_state.get("config") != config
                        or checkpoint_state.get("input_order") != list(INPUT_NAMES)
                        or len(checkpoint_state.get("history", [])) != step):
                    raise ValueError(f"saved continuous checkpoint identity/configuration is invalid at step {step} for {candidate_name}")
            continuous_terminal = torch.load(expected_terminal, map_location="cpu", weights_only=False)
            if (continuous_terminal.get("candidate") != candidate_name
                    or continuous_terminal.get("step") != 3
                    or continuous_terminal.get("config") != config
                    or continuous_terminal.get("input_order") != list(INPUT_NAMES)):
                raise ValueError(f"saved continuous terminal identity/configuration is invalid for {candidate_name}")
        else:
            continuous = _train_candidate(assets, config, run_dir, candidate_name, device)
        checkpoint = Path(continuous["step1_checkpoint"])
        command = [sys.executable, "-m", "scripts.research.lightgen.execute_ols_decoder_autograd_update_preflight_v11",
                   "--config", str(config_path), "--source-run-dir", str(assets["source_dir"]),
                   "--run-dir", str(run_dir), "--device", str(device), "--resume", str(checkpoint),
                   "--candidate", candidate_name]
        completed = subprocess.run(command, cwd=Path.cwd(), check=False, capture_output=True, text=True)
        log_dir = run_dir / "resume_process_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"{candidate_name}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (log_dir / f"{candidate_name}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"independent resume process failed for {candidate_name}; exit={completed.returncode}")
        resume_result = json.loads((run_dir / "optimization" / candidate_name / "resume_result.json").read_text(encoding="utf-8"))
        continuous_state = torch.load(continuous["final_checkpoint"], map_location="cpu", weights_only=False)
        restored_state = torch.load(resume_result["final_checkpoint"], map_location="cpu", weights_only=False)
        phase_differences = _tree_compare(continuous_state["phase_unwrapped"], restored_state["phase_unwrapped"],
                                          float(config["gates"]["restore_phase_atol"]), float(config["gates"]["restore_phase_rtol"]))
        optimizer_differences = _tree_compare(continuous_state["optimizer_state"], restored_state["optimizer_state"],
                                              float(config["gates"]["restore_adam_atol"]), float(config["gates"]["restore_adam_rtol"]))
        rng_differences = _tree_compare(continuous_state["rng_state"], restored_state["rng_state"], 0.0, 0.0)
        history_differences = _tree_compare(continuous_state["history"], restored_state["history"], 0.0, 0.0)
        continuous_camera = _camera_endpoint_for_checkpoint(assets, config, run_dir, candidate_name,
            Path(continuous["final_checkpoint"]), device)
        restored_camera = _camera_endpoint_for_checkpoint(assets, config, run_dir, candidate_name,
            Path(resume_result["final_checkpoint"]), device)
        camera_error = _relative_l2(restored_camera, continuous_camera)
        continuous_windows_pass = all(value.get("passed", False) for value in continuous["window_metrics"].values())
        restored_windows_pass = all(value.get("passed", False) for value in resume_result["window_metrics"].values())
        updates_nonzero = len(continuous["history"]) == 3 and all(
            all(float(item) > 0.0 for item in record["phase_gradient_l2_by_layer"])
            and all(float(item) > 0.0 for item in record["phase_update_l2_by_layer"])
            for record in continuous["history"])
        candidate_passed = (not phase_differences and not optimizer_differences and not rng_differences
                            and not history_differences and camera_error <= float(config["gates"]["restore_camera_output_relative_error"])
                            and continuous_windows_pass and restored_windows_pass and updates_nonzero)
        records["candidates"][candidate_name] = {
            "continuous_adam_calls": continuous["adam_call_count"], "restore_adam_calls": resume_result["adam_call_count"],
            "phase_final_differences": phase_differences, "optimizer_final_differences": optimizer_differences,
            "rng_final_differences": rng_differences, "history_differences": history_differences,
            "loaded_step1_state_and_adam_reloaded": True,
            "continuous_and_resume_input_order_equal": continuous_state["input_order"] == restored_state["input_order"] == list(INPUT_NAMES),
            "continuous_all_step_window_checks_pass": continuous_windows_pass,
            "resume_replayed_step_window_checks_pass": restored_windows_pass,
            "both_screens_had_nonzero_gradient_and_update_all_three_steps": updates_nonzero,
            "camera_output_relative_l2": camera_error, "camera_output_gate": float(config["gates"]["restore_camera_output_relative_error"]),
            "independent_resume_process_exit_code": completed.returncode,
            "continuous_final_checkpoint": continuous["final_checkpoint"], "resume_final_checkpoint": resume_result["final_checkpoint"],
            "continuous_window_metrics": continuous["window_metrics"], "resume_window_metrics": resume_result["window_metrics"],
            "passed": candidate_passed,
        }
    records["actual_engineering_adam_calls"] = sum(v["continuous_adam_calls"] + v["restore_adam_calls"] for v in records["candidates"].values())
    records["passed"] = records["actual_engineering_adam_calls"] == 10 and all(v["passed"] for v in records["candidates"].values())
    records["status"] = "ENGINEERING_ACCEPTED" if records["passed"] else "ENGINEERING_ACCEPTANCE_INCOMPLETE"
    _write_json(run_dir / "recovery_and_updates.json", records)
    return records


def _tensor_summary(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().to(torch.float64).reshape(-1)
    quantiles = torch.quantile(flat, torch.tensor([0.0, 0.1, 0.5, 0.9, 1.0], dtype=torch.float64, device=flat.device))
    return {"min": float(quantiles[0].cpu()), "p10": float(quantiles[1].cpu()),
            "p50": float(quantiles[2].cpu()), "p90": float(quantiles[3].cpu()),
            "max": float(quantiles[4].cpu()), "mean": float(flat.mean().cpu())}


def _endpoint_diagnostics(assets: dict[str, Any], config: dict[str, Any], run_dir: Path,
                          device: torch.device) -> dict[str, Any]:
    report: dict[str, Any] = {"candidates": {}, "array_directory": str(run_dir / "endpoint_arrays")}
    array_dir = run_dir / "endpoint_arrays"
    array_dir.mkdir(parents=True, exist_ok=True)
    for candidate_name in config["measurement"]["candidates"]:
        candidate = assets["candidates"][candidate_name]
        continuous_path = run_dir / "optimization" / candidate_name / "checkpoints" / "checkpoint_step_003.pt"
        payload = torch.load(continuous_path, map_location="cpu", weights_only=False)
        phases = {"step_0": assets["phases"].to(device=device, dtype=torch.float32),
                  "step_3": payload["phase_unwrapped"].to(device=device, dtype=torch.float32)}
        candidate_record: dict[str, Any] = {"modes": {}, "direct_zero_update_control": {},
                                            "worst_saved_binary_pattern": candidate["worst_binary_name"]}
        for mode in TRAIN_MODES:
            mode_record: dict[str, Any] = {}
            for label, phase in phases.items():
                pattern_records = {}
                pattern_indices = set(assets["input_indices"][candidate_name][name] for name in INPUT_NAMES)
                if label == "step_3":
                    pattern_indices.add(int(candidate["worst_binary_index"]))
                for pattern_index in sorted(pattern_indices):
                    pattern_name = str(candidate["pattern_names"][pattern_index])
                    active, z_m = _cached_active(assets, run_dir, candidate_name, mode, "explicit_4f", 4096,
                                                 pattern_index, torch.complex64, device)
                    with torch.no_grad():
                        out = _camera_forward(active, phase, 4096, z_m, config, quadrature_order=4, keep_roi=True)
                        sensitivity = {}
                        if label == "step_0":
                            sensitivity = {"10us": _camera_task(out["pixel_power_w"], config, 1e-5)[1],
                                           "100us": _camera_task(out["pixel_power_w"], config, 1e-4)[1],
                                           "10ms": _camera_task(out["pixel_power_w"], config, 1e-2)[1]}
                        pixel8 = v9._camera_quadrature_field(out["camera_field"], 8)[1]
                        archive_path = array_dir / f"{candidate_name}__{mode}__{label}__{pattern_name}.npz"
                        arrays = dict(
                            pixel_power_w_per_pixel=out["pixel_power_w"].detach().cpu().numpy().astype(np.float32),
                            expected_electrons_1ms=out["expected_electrons"].detach().cpu().numpy().astype(np.float32),
                            task_output_1ms=out["task_output"].detach().cpu().numpy().astype(np.float32))
                        arrays.update({f"task_output_{name}": value.detach().cpu().numpy().astype(np.float32) for name, value in sensitivity.items()})
                        np.savez_compressed(archive_path, **arrays)
                        saturation = {"1ms": float((out["task_output"] >= 1).to(torch.float64).mean().cpu())}
                        saturation.update({name: float((value >= 1).to(torch.float64).mean().cpu()) for name, value in sensitivity.items()})
                        pattern_records[pattern_name] = {
                            "array_archive": str(archive_path),
                            "pixel_power_w_per_pixel": _tensor_summary(out["pixel_power_w"]),
                            "expected_electrons_1ms": _tensor_summary(out["expected_electrons"]),
                            "task_output_1ms": _tensor_summary(out["task_output"]),
                            "saturation_fraction_by_exposure": saturation,
                            "exposure_sensitivity_is_zero_update_only": label == "step_0",
                            "gl4_vs_gl8_power_relative_l2": _relative_l2(out["pixel_power_w"], pixel8),
                            "power_w_by_plane": out["decoder_trace"], "camera_trace": out["camera_trace"]}
                        if pattern_name in INPUT_NAMES:
                            active128, z128 = _cached_active(assets, run_dir, candidate_name, mode, "explicit_4f", 4096,
                                                             pattern_index, torch.complex128, device)
                            high = _camera_forward(active128, phase.to(torch.float64), 4096, z128,
                                                   config, quadrature_order=4, keep_roi=True)
                            pattern_records[pattern_name]["complex128_diagnostic"] = {
                                "complex64_vs_complex128_roi": v9._window_metrics(out["common_roi"], high["common_roi"]),
                                "pixel_power_c64_vs_c128_relative_l2": _relative_l2(out["pixel_power_w"], high["pixel_power_w"]),
                                "gl4_vs_gl8_power_relative_l2": _relative_l2(high["pixel_power_w"], v9._camera_quadrature_field(high["camera_field"], 8)[1])}
                            del active128, high
                    del active, out, pixel8, sensitivity
                    clear_production_transfer_cache()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                mode_record[label] = pattern_records
            candidate_record["modes"][mode] = mode_record
        for mode in TRAIN_MODES:
            for pattern_name in INPUT_NAMES:
                pattern_index = assets["input_indices"][candidate_name][pattern_name]
                active, z_m = _cached_active(assets, run_dir, candidate_name, mode, "direct", 4096,
                                             pattern_index, torch.complex64, device)
                with torch.no_grad():
                    out = _camera_forward(active, phases["step_0"], 4096, z_m, config, quadrature_order=4)
                    archive_path = array_dir / f"{candidate_name}__{mode}__direct_step_0__{pattern_name}.npz"
                    np.savez_compressed(archive_path,
                        pixel_power_w_per_pixel=out["pixel_power_w"].detach().cpu().numpy().astype(np.float32),
                        task_output_1ms=out["task_output"].detach().cpu().numpy().astype(np.float32))
                    candidate_record["direct_zero_update_control"][f"{mode}/{pattern_name}"] = {
                        "array_archive": str(archive_path), "pixel_power_w_per_pixel": _tensor_summary(out["pixel_power_w"]),
                        "task_output_1ms": _tensor_summary(out["task_output"]), "power_w_by_plane": out["decoder_trace"],
                        "camera_trace": out["camera_trace"]}
                del active, out
                clear_production_transfer_cache()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        worst_name = candidate["worst_binary_name"]
        candidate_record["worst_binary_step_3_window_check"] = _window_checks(
            assets, config, run_dir, candidate_name, phases["step_3"], 3, (worst_name,), device)
        gl4_values = [
            float(pattern_record["gl4_vs_gl8_power_relative_l2"])
            for mode_record in candidate_record["modes"].values()
            for state_record in mode_record.values()
            if isinstance(state_record, dict)
            for pattern_record in state_record.values()
            if isinstance(pattern_record, dict) and "gl4_vs_gl8_power_relative_l2" in pattern_record
        ]
        candidate_record["pixel_quadrature_gl4_vs_gl8_max_relative_l2"] = max(gl4_values, default=math.inf)
        candidate_record["passed"] = bool(candidate_record["worst_binary_step_3_window_check"]["passed"]
                                           and gl4_values
                                           and candidate_record["pixel_quadrature_gl4_vs_gl8_max_relative_l2"] <= float(config["gates"]["forward_relative_error"]))
        report["candidates"][candidate_name] = candidate_record
    report["passed"] = all(candidate["passed"] for candidate in report["candidates"].values())
    report["status"] = "ENDPOINT_READOUT_COMPLETE" if report["passed"] else "ENDPOINT_READOUT_NUMERICAL_CHECK_INCOMPLETE"
    _write_json(run_dir / "endpoint_diagnostics.json", report)
    return report


def _resume_process(assets: dict[str, Any], config: dict[str, Any], run_dir: Path,
                    candidate_name: str, resume_path: Path, device: torch.device) -> None:
    result = _train_candidate(assets, config, run_dir, candidate_name, device, resume_path=resume_path)
    _write_json(run_dir / "optimization" / candidate_name / "resume_result.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-run-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--run", action="store_true", help="run the fixed three-step and independent recovery engineering acceptance")
    action.add_argument("--resume", type=Path, help="explicit step-1 checkpoint for a fresh process; internal recovery entry")
    parser.add_argument("--candidate", choices=tuple(v9.SIZES), help="required only with --resume")
    args = parser.parse_args()
    if args.resume and not args.candidate:
        parser.error("--resume requires an explicit --candidate")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable; CPU fallback is forbidden for the cloud run")
    config = _load_config(args.config)
    root = Path.cwd().resolve()
    source = args.source_run_dir if args.source_run_dir.is_absolute() else root / args.source_run_dir
    run_dir = args.run_dir if args.run_dir.is_absolute() else root / args.run_dir
    device = torch.device(args.device)
    assets = _source_assets(root, source)
    if args.preflight_only:
        preflight = _preflight(assets, config, args.config, run_dir, device)
        print(json.dumps({"status": preflight["status"], "passed": preflight["passed"], "report": str(run_dir / "preflight.json")}, ensure_ascii=False))
        return
    if args.resume:
        candidate_name = args.candidate
        _resume_process(assets, config, run_dir, candidate_name, args.resume.resolve(), device)
        print(json.dumps({"status": "RESUME_PROCESS_COMPLETE", "candidate": candidate_name}, ensure_ascii=False))
        return
    if args.run:
        preflight_path = run_dir / "preflight.json"
        if not preflight_path.is_file():
            raise FileNotFoundError("--run requires --preflight-only to pass in this run directory")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if not preflight.get("passed"):
            raise ValueError("preflight failed; refusing engineering updates")
        update_result = _run_continuity_and_resume(assets, config, args.config, run_dir, device)
        if not update_result["passed"]:
            raise RuntimeError("phase/Adam/RNG/camera recovery did not meet the fixed acceptance")
        endpoints = _endpoint_diagnostics(assets, config, run_dir, device)
        final_passed = bool(update_result["passed"] and endpoints["passed"])
        final = {"schema": SCHEMA, "status": "ENGINEERING_ACCEPTED" if final_passed else "ENGINEERING_ACCEPTANCE_INCOMPLETE",
                 "decision_question": "Can the frozen four- and nine-channel candidates complete the actual differentiable camera-loss chain, three Decoder-only Adam updates and independent recovery?",
                 "preflight": preflight, "updates_and_recovery": update_result, "endpoint_diagnostics": endpoints,
                 "engineering_adam_calls": update_result["actual_engineering_adam_calls"],
                 "scientific_training_updates": 0, "dataset_reads": 0, "task_evaluator_calls": 0,
                 "fid_calls": 0, "material_forward_calls": 0, "loss_changes": 0,
                 "elapsed_seconds": time.time() - _START, "config_path": str(args.config),
                 "source_run_dir": str(assets["source_dir"]), "run_dir": str(run_dir),
                 "warning": "Three-step engineering loss changes are not reconstruction-quality or scientific-training evidence."}
        _write_json(run_dir / "report.json", final)
        print(json.dumps({"status": final["status"], "engineering_adam_calls": final["engineering_adam_calls"], "report": str(run_dir / "report.json")}, ensure_ascii=False))
        if final["status"] != "ENGINEERING_ACCEPTED":
            raise RuntimeError("v11 engineering acceptance did not pass")
        return


if __name__ == "__main__":
    main()
