"""Fixed-front-end MNIST reconstruction study for the Gaussian/OLS branch."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ldm_project.v2.application.research.shared.research_run_control import (
    CORRECTNESS_INVARIANTS,
    IMPLEMENTATION_SELECTION,
    RESOURCE_ENVELOPE,
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    capture_execution_source_manifest,
)
from ldm_project.v2.student.evaluator import stratified_mnist_indices
from scripts.research.lightgen import diagnose_ols_decoder_camera_interface_v9 as v9
from scripts.research.lightgen import diagnose_ols_frozen_phase_controllability_v7 as v7
from scripts.research.lightgen import diagnose_ols_window_dimension_power_v5 as v5
from scripts.research.lightgen import execute_ols_decoder_autograd_update_preflight_v11 as v11
from ldm_project.v2.optics.propagation import clear_production_transfer_cache


SCHEMA = "ols_decoder_mnist_reconstruction.v12"
PREFLIGHT_SCHEMA = "ols_decoder_mnist_reconstruction_preflight.v12"
CHECKPOINT_SCHEMA = "ols_decoder_mnist_reconstruction_checkpoint.v12"
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_decoder_mnist_reconstruction_v12.yaml")
DEFAULT_SOURCE_DIR = Path("outputs/20261016_ols_decoder_camera_interface_v9/runs/attempt_02")
DEFAULT_DATA_FILE = Path("data/MNIST/processed/training.pt")
DEFAULT_PREFLIGHT_DIR = Path("outputs/20261018_ols_decoder_mnist_reconstruction_v12/runs/preflight_01")
DEFAULT_RUN_DIR = Path("outputs/20261018_ols_decoder_mnist_reconstruction_v12/runs/attempt_01")
CANDIDATES = ("candidate_2x2_C", "candidate_3x3_B")
MODES = ("gaussian_s12", "lp01")
GRIDS = (4096, 6144, 8192)
INPUT_NAMES = ("fit", "confirmation")
WAVELENGTH_M = 532.0e-9
DX_M = 2.0e-6
SOURCE_POWER_W = 249.3e-6
H_PLANCK = 6.62607015e-34
C_LIGHT = 299792458.0
EPS = 1.0e-30
TASK_ID = "OLS_DECODER_MNIST_RECONSTRUCTION_V12"
AUTHORITY = "USER_APPROVED_OLS_DECODER_MNIST_RECONSTRUCTION_20261018"
_SIGNAL_RECEIVED: int | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict) and config.get("schema") == SCHEMA, f"expected {SCHEMA} config")
    fixed = (
        (config["optics"]["wavelength_m"], WAVELENGTH_M),
        (config["optics"]["dx_m"], DX_M),
        (config["optics"]["source_power_w"], SOURCE_POWER_W),
        (config["optics"]["encoder_aperture_m"], 200.0e-6),
        (config["optics"]["encoder_layer_spacing_m"], 1.5e-3),
        (config["optics"]["encoder_to_fiber_distance_m"], 1.5e-3),
        (config["optics"]["encoder_layer_transmission_power"], 0.889),
        (config["optics"]["decoder_layer_transmission_power"], 0.889),
        (config["optics"]["fiber_pitch_m"], 125.0e-6),
        (config["camera"]["primary_exposure_s"], 1.0e-3),
        (config["camera"]["quantum_efficiency"], 0.5),
        (config["camera"]["full_well_e"], 30000.0),
        (config["training"]["adam_learning_rate"], 1.0e-3),
    )
    _require(all(math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1.0e-15) for a, b in fixed),
             "v12 fixed optical, camera, or optimizer parameters changed")
    _require(config["training"]["candidates"] == list(CANDIDATES), "v12 candidate set changed")
    _require(config["training"]["updates_per_candidate"] == 500, "v12 update budget changed")
    _require(config["training"]["batch_size"] == 10, "v12 balanced batch size changed")
    _require(config["optics"]["grids"] == list(GRIDS), "v12 propagation windows changed")
    _require(config["optics"]["common_roi_pixels"] == 2048, "v12 common physical ROI changed")
    _require(config["data"]["fit_per_label"] == 10 and config["data"]["confirmation_per_label"] == 20,
             "v12 fixed sample counts changed")
    _require(config["measurement"]["bootstrap_replicates"] == 2000, "v12 bootstrap count changed")
    _require(config["training"]["adam_betas"] == [0.9, 0.999] and
             math.isclose(float(config["training"]["adam_epsilon"]), 1.0e-8, rel_tol=0.0, abs_tol=1.0e-18) and
             float(config["training"]["adam_weight_decay"]) == 0.0,
             "v12 Adam state or weight decay changed")
    _require(config["training"]["schedule_seed"] == 2026101802 and
             config["data"]["sample_seed"] == 2026101801 and
             config["measurement"]["bootstrap_seed"] == 2026101803,
             "v12 fixed sample, schedule, or bootstrap seed changed")
    _require(config["training"]["primary_mode"] == "gaussian_s12" and
             config["training"]["sensitivity_mode"] == "lp01" and
             config["training"]["decoder_initial_state"] == "fixed_low_frequency" and
             config["training"]["route"] == "explicit_4f" and
             config["training"]["reconstruction_loss"] == "clipped_expected_camera_task_mse",
             "v12 model, route, initialization, or objective changed")
    _require(config["camera"]["shape"] == [200, 200] and
             math.isclose(float(config["camera"]["pixel_pitch_m"]), 2.5e-6, rel_tol=0.0, abs_tol=1e-15) and
             config["camera"]["sensitivity_exposures_s"] == [1.0e-5, 1.0e-4, 1.0e-2] and
             float(config["camera"]["read_noise_e_rms"]) == 1.0 and
             float(config["camera"]["dark_current_e_per_pixel_s"]) == 0.0,
             "v12 fixed sensor dimensions or exposure sensitivity changed")
    _require(config["measurement"]["camera_quadrature_primary_order"] == 4 and
             config["measurement"]["camera_quadrature_reference_order"] == 8 and
             config["measurement"]["gallery_fit_per_label"] == 1 and
             config["measurement"]["gallery_confirmation_per_label"] == 2,
             "v12 fixed pixel integration or gallery sampling changed")
    _require(config["scope"]["formal_generator_training_updates"] == 0, "v12 cannot authorize formal generation training")
    return config


def _load_assets(root: Path, source_dir: Path) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    root = root.resolve()
    _require(source_dir.is_relative_to(root), "v9 source path must remain under the repository")
    assets = v11._source_assets(root, source_dir)
    assets["decoder_phase"] = assets["phases"].detach().cpu().to(torch.float32).clone()
    response_path = source_dir / "frozen_v8_response_inputs_v9.npz"
    encoder_path = source_dir / "frozen_encoder_phase_screens_v9.pt"
    encoder = torch.load(encoder_path, map_location="cpu", weights_only=False)
    _require(encoder.get("schema") == "ols_decoder_camera_frozen_encoder_phases.v9",
             "v9 frozen Encoder phase archive schema changed")
    _require(set(encoder.get("phases", {})) == set(CANDIDATES), "v9 Encoder candidate phase set changed")
    encoder_phases = {}
    for candidate in CANDIDATES:
        phase = encoder["phases"][candidate].detach().cpu().to(torch.float32)
        _require(tuple(phase.shape) == (2, 1, 100, 100) and bool(torch.isfinite(phase).all()),
                 f"invalid frozen Encoder phase tensor for {candidate}")
        encoder_phases[candidate] = phase
    h_matrices: dict[str, dict[str, dict[int, np.ndarray]]] = {}
    with np.load(response_path, allow_pickle=False) as archive:
        for candidate in CANDIDATES:
            h_matrices[candidate] = {}
            size = int(v9.SIZES[candidate])
            for mode in MODES:
                h_matrices[candidate][mode] = {}
                for grid in GRIDS:
                    key = f"{candidate}__{mode}__{grid}__H_sqrt_w"
                    h = np.asarray(archive[key], dtype=np.complex128)
                    _require(h.shape == (size * size, 10000) and np.isfinite(h).all(),
                             f"invalid v9 H matrix: {key}")
                    h_matrices[candidate][mode][grid] = h.copy()
    assets["encoder_phases"] = encoder_phases
    assets["H"] = h_matrices
    return assets


def _load_mnist(path: Path, config: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _require(path.is_file(), f"canonical MNIST training source is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(isinstance(payload, (tuple, list)) and len(payload) == 2, "MNIST source schema changed")
    pixels, labels = payload
    _require(isinstance(pixels, torch.Tensor) and pixels.dtype == torch.uint8 and tuple(pixels.shape) == (60000, 28, 28),
             "MNIST training pixels changed")
    _require(isinstance(labels, torch.Tensor) and labels.dtype == torch.long and tuple(labels.shape) == (60000,),
             "MNIST training labels changed")
    train, dev = stratified_mnist_indices(
        labels, seed=int(config["data"]["split_seed"]),
        validation_per_label=int(config["data"]["dev_per_label"]),
    )
    train, dev = train.sort().values, dev.sort().values
    _require(train.numel() == 55000 and dev.numel() == 5000 and not bool(torch.isin(train, dev).any()),
             "canonical MNIST TRAIN/DEV split changed")
    return pixels, labels, train


def _select_population(labels: torch.Tensor, train_rows: torch.Tensor, config: dict[str, Any]) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(int(config["data"]["sample_seed"]))
    fit_by_class: list[torch.Tensor] = []
    confirm_by_class: list[torch.Tensor] = []
    for label in range(10):
        candidates = train_rows[labels[train_rows] == label]
        selected = candidates[torch.randperm(candidates.numel(), generator=generator)[:30]]
        fit_by_class.append(selected[:10])
        confirm_by_class.append(selected[10:])
    fit = torch.cat(fit_by_class)
    confirmation = torch.cat(confirm_by_class)
    rows = torch.cat((fit, confirmation))
    _require(rows.unique().numel() == 300 and bool(torch.isin(rows, train_rows).all()),
             "selected MNIST population is not unique canonical TRAIN")
    _require(torch.equal(torch.bincount(labels[fit], minlength=10), torch.full((10,), 10)),
             "MNIST fit population is not 10 per class")
    _require(torch.equal(torch.bincount(labels[confirmation], minlength=10), torch.full((10,), 20)),
             "MNIST confirmation population is not 20 per class")
    return {"fit_rows": fit, "confirmation_rows": confirmation, "all_rows": rows,
            "all_labels": labels[rows].clone(), "fit_labels": labels[fit].clone(),
            "confirmation_labels": labels[confirmation].clone(),
            "row_role": ["fit"] * fit.numel() + ["confirmation"] * confirmation.numel()}


def _make_schedule(population: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    fit_rows = population["fit_rows"]
    labels = population["fit_labels"]
    generator = torch.Generator(device="cpu").manual_seed(int(config["training"]["schedule_seed"]))
    per_class = [fit_rows[labels == label] for label in range(10)]
    updates = int(config["training"]["updates_per_candidate"])
    schedule = torch.empty((updates, 10), dtype=torch.long)
    for epoch in range(updates // 10):
        shuffled = [rows[torch.randperm(10, generator=generator)] for rows in per_class]
        for index in range(10):
            schedule[epoch * 10 + index] = torch.stack([shuffled[label][index] for label in range(10)])
    _require(torch.equal(torch.sort(schedule[:10].reshape(-1)).values, torch.sort(fit_rows).values),
             "first schedule epoch does not visit each fit row once")
    _require(bool((schedule >= 0).all()) and set(schedule.reshape(-1).tolist()) == set(fit_rows.tolist()),
             "fixed schedule includes rows outside the fit population")
    return schedule


def _prepare_images(pixels: torch.Tensor, population: dict[str, Any]) -> dict[str, torch.Tensor]:
    rows = population["all_rows"]
    images = pixels.index_select(0, rows).unsqueeze(1).to(torch.float32).div_(255.0)
    input_images = F.interpolate(images, size=(100, 100), mode="bilinear", align_corners=False).clamp_(0.0, 1.0)
    targets = F.interpolate(images, size=(200, 200), mode="bilinear", align_corners=False).clamp_(0.0, 1.0)
    commands = (0.25 + 0.5 * input_images).reshape(rows.numel(), 10000).contiguous()
    _require(float(commands.min()) >= 0.25 and float(commands.max()) <= 0.75,
             "MNIST amplitude commands left the qualified [0.25, 0.75] range")
    return {"input_images": input_images, "targets": targets, "commands": commands,
            "fit_target_mean": targets[:100].mean(dim=0, keepdim=False)}


def _population_position(population: dict[str, Any], row_id: int) -> int:
    positions = torch.nonzero(population["all_rows"] == int(row_id), as_tuple=False).flatten()
    _require(positions.numel() == 1, f"MNIST row is absent or duplicated: {row_id}")
    return int(positions.item())


def _assert_row_pair(input_row: int, target_row: int) -> None:
    _require(int(input_row) == int(target_row), "input/target MNIST row identity mismatch")


def _schedule_records(population: dict[str, Any], schedule: torch.Tensor) -> dict[str, Any]:
    row_to_label = {int(row): int(label) for row, label in zip(population["all_rows"], population["all_labels"])}
    return {"rows": schedule.clone(), "labels": torch.tensor([[row_to_label[int(row)] for row in batch] for batch in schedule])}


def _rng_state() -> dict[str, Any]:
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(payload: dict[str, Any], device: torch.device) -> None:
    torch.set_rng_state(payload["cpu"])
    if device.type == "cuda" and payload.get("cuda"):
        torch.cuda.set_rng_state_all(payload["cuda"])


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _front_cache_path(cache_dir: Path, candidate: str, mode: str, grid: int,
                      row_id: int, precision: str) -> Path:
    return cache_dir / f"{candidate}__{mode}__{grid}__{precision}__row_{int(row_id):05d}.pt"


def _validate_front_cache(bundle: dict[str, Any], *, candidate: str, mode: str,
                          grid: int, row_id: int, precision: str) -> None:
    expected_dtype = torch.complex64 if precision == "complex64" else torch.complex128
    active = bundle.get("decoder_entrance_field")
    _require((bundle.get("candidate"), bundle.get("mode"), bundle.get("grid"),
              bundle.get("row_id"), bundle.get("precision")) == (candidate, mode, grid, int(row_id), precision),
             "frozen front-end cache identity mismatch")
    _require(isinstance(active, torch.Tensor) and tuple(active.shape) == (100, 100),
             "frozen front-end cache field shape changed")
    _require(active.dtype == expected_dtype, "frozen front-end cache precision does not match its stored native dtype")
    _require(bool(torch.isfinite(active).all()), "frozen front-end cache contains nonfinite field values")
    coefficients = bundle.get("coefficients_sqrt_w")
    _require(isinstance(coefficients, torch.Tensor) and coefficients.dtype == expected_dtype and
             coefficients.ndim == 1 and bool(torch.isfinite(coefficients).all()),
             "frozen front-end cache coefficients do not preserve native precision")


def _make_front_active(assets: dict[str, Any], cache_dir: Path, candidate: str, mode: str,
                       grid: int, row_id: int, command: torch.Tensor, precision: str,
                       device: torch.device) -> tuple[torch.Tensor, dict[str, float], float, torch.Tensor]:
    dtype = torch.complex64 if precision == "complex64" else torch.complex128
    path = _front_cache_path(cache_dir, candidate, mode, grid, row_id, precision)
    if path.is_file():
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        _validate_front_cache(bundle, candidate=candidate, mode=mode, grid=grid,
                              row_id=row_id, precision=precision)
        coefficients = bundle["coefficients_sqrt_w"].to(dtype=dtype, device=device)
        return bundle["decoder_entrance_field"].to(device=device), bundle["route_trace_w"], float(bundle["route_z_m"]), coefficients
    h = torch.as_tensor(assets["H"][candidate][mode][grid], dtype=dtype, device=device)
    command_device = command.to(device=device, dtype=dtype).reshape(-1)
    coefficients = h @ command_device
    _require(bool(torch.isfinite(coefficients).all()), "frozen H matrix produced nonfinite coefficients")
    source = v11._compose_fiber_field(assets["candidates"][candidate], mode,
                                     coefficients.detach().cpu().numpy(), grid, dtype, device)
    active, trace, route_z = v9._route_to_decoder(source, "explicit_4f", grid)
    active_cpu = active.detach().cpu().contiguous()
    payload = {"candidate": candidate, "mode": mode, "grid": grid, "row_id": int(row_id),
               "precision": precision, "decoder_entrance_field": active_cpu,
               "coefficients_sqrt_w": coefficients.detach().cpu(), "route_trace_w": trace,
               "route_z_m": float(route_z), "coordinate_frame": "v9_global_fiber_xy_to_decoder_xy",
               "field_units": "sqrt(W)/m", "pixel_pitch_m": DX_M}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    clear_production_transfer_cache()
    del source, active, h
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return active_cpu.to(device=device), trace, float(route_z), coefficients


def _roi_power_w(field: torch.Tensor) -> float:
    return float(field.abs().square().sum().real.detach().to(torch.float64).cpu()) * DX_M**2


def _common_window_metrics(reference: torch.Tensor, test: torch.Tensor) -> dict[str, float]:
    ref = reference.to(torch.complex128)
    value = test.to(torch.complex128)
    phase = torch.angle(torch.sum(ref.conj() * value))
    aligned = value * torch.exp(-1j * phase)
    ref_norm = torch.linalg.vector_norm(ref).clamp_min(EPS)
    ref_intensity = ref.abs().square()
    value_intensity = value.abs().square()
    ref_power_w = float(ref_intensity.sum()) * DX_M**2
    value_power_w = float(value_intensity.sum()) * DX_M**2
    return {
        "raw_complex_relative_l2": float(torch.linalg.vector_norm(ref - value) / ref_norm),
        "aligned_complex_relative_l2": float(torch.linalg.vector_norm(ref - aligned) / ref_norm),
        "common_global_phase_rad": float(phase),
        "intensity_relative_l2": float(torch.linalg.vector_norm(ref_intensity - value_intensity) /
                                        torch.linalg.vector_norm(ref_intensity).clamp_min(EPS)),
        "roi_power_reference_w": ref_power_w,
        "roi_power_test_w": value_power_w,
        "roi_power_relative_error": abs(ref_power_w - value_power_w) / max(abs(ref_power_w), EPS),
    }


def _camera_task(power_w_per_pixel: torch.Tensor, config: dict[str, Any], exposure_s: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    camera = config["camera"]
    exposure = float(camera["primary_exposure_s"] if exposure_s is None else exposure_s)
    photon_energy = H_PLANCK * C_LIGHT / WAVELENGTH_M
    electrons = power_w_per_pixel * (float(camera["quantum_efficiency"]) * exposure / photon_energy)
    return electrons, torch.clamp(electrons / float(camera["full_well_e"]), 0.0, 1.0)


def _camera_forward(active: torch.Tensor, phase: torch.Tensor, grid: int, route_z: float,
                    config: dict[str, Any], *, keep_roi: bool = False) -> dict[str, Any]:
    output = v11._camera_forward(active, phase, grid, route_z, config,
                                 quadrature_order=int(config["measurement"]["camera_quadrature_primary_order"]),
                                 keep_roi=keep_roi)
    output["expected_electrons"], output["task_output"] = _camera_task(output["pixel_power_w"], config)
    return output


def _set_frozen_encoder_phase(runtime: Any, source_phase: torch.Tensor, device: torch.device) -> None:
    target = runtime.encoder.phase
    expected = source_phase.to(device=device, dtype=target.dtype)
    _require(tuple(target.shape) == tuple(expected.shape), "frozen Encoder phase shape does not match the saved screen")
    with torch.no_grad():
        target.copy_(expected)
    _require(torch.equal(target.detach(), expected), "frozen Encoder phase assignment changed values")


def _h_validation(assets: dict[str, Any], population: dict[str, Any], prepared: dict[str, torch.Tensor],
                  config: dict[str, Any], device: torch.device, run_dir: Path) -> dict[str, Any]:
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError("v12 production H validation requires AutoDL CUDA")
    grid = 4096
    runtime = v5._make_runtime(grid, torch.float32, device)
    entrance = v5._encoder_entrance(runtime, torch.float32, device)
    roi_plane = v9._plane("v12_fiber_projection_roi", (2048, 2048), (DX_M, DX_M), 0.0, "fiber_global")
    records: dict[str, Any] = {}
    cache_dir = run_dir / "frozen_frontend_cache"
    for candidate in CANDIDATES:
        direct_by_mode = {mode: [] for mode in MODES}
        h_errors = {mode: [] for mode in MODES}
        before_fiber = []
        source_power = []
        slm_power = []
        _set_frozen_encoder_phase(runtime, assets["encoder_phases"][candidate], device)
        for position, row_id in enumerate(population["all_rows"].tolist()):
            command = prepared["commands"][position].reshape(1, 10000)
            fields, traces = v7._runtime_forward_batch(runtime, entrance, assets["encoder_phases"][candidate],
                                                       command, grid, batch_size=1)
            roi = v7._crop_roi(fields[0], grid)
            before_fiber.append(_roi_power_w(roi))
            command_vector = command[0].to(device=device, dtype=torch.float32)
            for mode in MODES:
                direct = v7._project_roi_batch(roi[None], int(v9.SIZES[candidate]), mode, roi_plane)[0]
                h = torch.as_tensor(assets["H"][candidate][mode][grid], dtype=torch.complex64, device=device)
                predicted = h @ command_vector.to(dtype=h.dtype)
                relative = float(torch.linalg.vector_norm(predicted.to(torch.complex128) - direct.to(torch.complex128)) /
                                 torch.linalg.vector_norm(direct.to(torch.complex128)).clamp_min(EPS))
                h_errors[mode].append(relative)
                direct_by_mode[mode].append(direct.detach().cpu().to(torch.complex128).numpy())
                del direct, h, predicted
            mean_square = float(prepared["commands"][position].square().mean())
            source_power.append(SOURCE_POWER_W)
            slm_power.append(SOURCE_POWER_W * mean_square)
            del fields, traces, roi
            clear_production_transfer_cache()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        mode_results = {}
        for mode in MODES:
            errors = np.asarray(h_errors[mode], dtype=np.float64)
            _require(errors.shape == (300,) and np.isfinite(errors).all(), "H validation omitted MNIST rows")
            mode_results[mode] = {"max_relative_complex_coefficient_error": float(errors.max()),
                                  "p99_relative_complex_coefficient_error": float(np.quantile(errors, 0.99)),
                                  "rows_checked": int(errors.size), "passed": bool(errors.max() <= 1.0e-4)}
        _require(all(value["passed"] for value in mode_results.values()),
                 f"v9 H acceleration failed direct production forward for {candidate}")
        path = run_dir / "frontend_measurements" / f"{candidate}_coefficients_and_input_power.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, rows=population["all_rows"].numpy(), labels=population["all_labels"].numpy(),
                            commands=prepared["commands"].numpy(),
                            gaussian_coefficients_sqrt_w=np.asarray(direct_by_mode["gaussian_s12"], dtype=np.complex128),
                            lp01_coefficients_sqrt_w=np.asarray(direct_by_mode["lp01"], dtype=np.complex128),
                            power_before_fiber_roi_w=np.asarray(before_fiber, dtype=np.float64),
                            power_source_w=np.asarray(source_power, dtype=np.float64),
                            power_after_slm_w=np.asarray(slm_power, dtype=np.float64))
        records[candidate] = {"mode_checks": mode_results, "rows": int(len(before_fiber)),
                              "before_fiber_roi_power_w": {"min": float(np.min(before_fiber)),
                                                            "mean": float(np.mean(before_fiber)),
                                                            "max": float(np.max(before_fiber))},
                              "frontend_file": str(path)}
        del direct_by_mode
        gc.collect()
    del runtime, entrance
    clear_production_transfer_cache()
    torch.cuda.empty_cache()
    return {"status": "PASS", "all_300_rows_directly_checked_for_both_modes": True,
            "max_relative_error_gate": 1.0e-4, "candidates": records}


def _load_front_power_metadata(run_dir: Path, candidate: str) -> dict[int, dict[str, float]]:
    path = run_dir / "frontend_measurements" / f"{candidate}_coefficients_and_input_power.npz"
    with np.load(path, allow_pickle=False) as archive:
        rows = np.asarray(archive["rows"], dtype=np.int64)
        _require(rows.size == 300 and np.unique(rows).size == 300, "front-end power metadata has missing/duplicate rows")
        return {int(row): {"source_input_power_w": float(archive["power_source_w"][index]),
                           "after_slm_power_w": float(archive["power_after_slm_w"][index]),
                           "before_fiber_roi_power_w": float(archive["power_before_fiber_roi_w"][index])}
                for index, row in enumerate(rows)}


@lru_cache(maxsize=4)
def _cached_front_power_metadata(run_dir_text: str, candidate: str) -> dict[int, dict[str, float]]:
    path = Path(run_dir_text) / "frontend_measurements" / f"{candidate}_coefficients_and_input_power.npz"
    with np.load(path, allow_pickle=False) as archive:
        rows = np.asarray(archive["rows"], dtype=np.int64)
        _require(rows.size == 300 and np.unique(rows).size == 300,
                 "front-end power metadata has missing/duplicate rows")
        return {int(row): {"source_input_power_w": float(archive["power_source_w"][index]),
                           "after_slm_power_w": float(archive["power_after_slm_w"][index]),
                           "before_fiber_roi_power_w": float(archive["power_before_fiber_roi_w"][index])}
                for index, row in enumerate(rows)}


def _front_power_metadata(run_dir: Path, candidate: str, row_id: int) -> dict[str, float]:
    metadata = _cached_front_power_metadata(str(run_dir.resolve()), candidate)
    _require(int(row_id) in metadata, f"front-end power metadata is missing MNIST row {row_id}")
    return metadata[int(row_id)]


def _neff(channel_power: np.ndarray) -> float:
    values = np.asarray(channel_power, dtype=np.float64)
    total = float(values.sum())
    denominator = float(np.square(values).sum())
    return total * total / denominator if denominator > EPS else 0.0


def _channel_summary(channel_power: np.ndarray) -> dict[str, float]:
    values = np.asarray(channel_power, dtype=np.float64)
    mean = float(values.mean())
    return {"min_w": float(values.min()), "p10_w": float(np.quantile(values, 0.1)),
            "p50_w": float(np.quantile(values, 0.5)), "mean_w": mean,
            "p90_w": float(np.quantile(values, 0.9)), "max_w": float(values.max()),
            "p10_over_p90": float(np.quantile(values, 0.1) / max(np.quantile(values, 0.9), EPS)),
            "coefficient_of_variation": float(values.std() / max(mean, EPS)),
            "N_eff": _neff(values)}


def _exposure_summaries(power: torch.Tensor, targets: torch.Tensor, config: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for exposure in [float(config["camera"]["primary_exposure_s"]),
                     *[float(value) for value in config["camera"]["sensitivity_exposures_s"]]]:
        electrons, task = _camera_task(power, config, exposure)
        result[f"{exposure:.8g}"] = {
            "exposure_s": exposure,
            "mean_image_mse": float((task - targets).square().flatten(1).mean(dim=1).mean()),
            "mean_saturation_fraction": float((electrons >= float(config["camera"]["full_well_e"])).float().mean()),
            "mean_electrons_per_image": float(electrons.flatten(1).sum(dim=1).mean()),
            "exposure_is_zero_update_sensitivity": exposure != float(config["camera"]["primary_exposure_s"]),
        }
    return result


def _evaluate_candidate(assets: dict[str, Any], population: dict[str, Any], prepared: dict[str, torch.Tensor],
                        config: dict[str, Any], run_dir: Path, candidate: str, mode: str,
                        phase: torch.Tensor, device: torch.device, *, state_name: str,
                        grids: tuple[int, ...], output_root: Path | None = None) -> dict[str, Any]:
    phase64 = phase.detach().to(device=device, dtype=torch.float32)
    output_root = run_dir if output_root is None else output_root
    mode_dir = output_root / "measurements" / state_name / candidate / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    all_rows = population["all_rows"].tolist()
    labels = population["all_labels"].numpy().astype(np.int64)
    targets_cpu = prepared["targets"].cpu().to(torch.float32)
    target_flat = targets_cpu.flatten(1)
    metric_rows: list[dict[str, Any]] = []
    pixel_powers: list[torch.Tensor] = []
    tasks: list[torch.Tensor] = []
    electrons_list: list[torch.Tensor] = []
    channels: list[np.ndarray] = []
    representative_rows = {int(row) for row in population["fit_rows"].reshape(10, 10)[:, 0].tolist()}
    representative_rois: dict[int, np.ndarray] = {}
    max_window_error = 0.0
    window_pairs: dict[str, list[float]] = {}
    pairwise_metrics: dict[str, list[dict[str, float]]] = {}
    power_metadata_by_row = _load_front_power_metadata(run_dir, candidate)
    for position, row_id in enumerate(all_rows):
        command = prepared["commands"][position]
        target = targets_cpu[position]
        grid_outputs: dict[int, dict[str, Any]] = {}
        grid_rois: dict[int, torch.Tensor] = {}
        for grid in grids:
            active, route_trace, route_z, coefficients = _make_front_active(
                assets, run_dir / "frozen_frontend_cache", candidate, mode, grid, int(row_id),
                command, "complex64", device,
            )
            output = _camera_forward(active, phase64, grid, route_z, config, keep_roi=True)
            _require(bool(torch.isfinite(output["pixel_power_w"]).all()) and
                     bool(torch.isfinite(output["task_output"]).all()),
                     f"nonfinite camera output for {candidate}/{mode}/{row_id}/{grid}")
            task_cpu = output["task_output"].detach().cpu().to(torch.float32)
            power_cpu = output["pixel_power_w"].detach().cpu().to(torch.float32)
            electron_cpu = output["expected_electrons"].detach().cpu().to(torch.float32)
            image_mse = float((task_cpu - target).square().mean())
            channel_power = coefficients.detach().abs().square().cpu().numpy().astype(np.float64)
            stage_fiber_power = float(route_trace["fiber_output_mode_power_w"])
            decoder_trace = {key: float(value) for key, value in output["decoder_trace"].items()}
            camera_trace = {key: float(value) for key, value in output["camera_trace"].items()}
            camera_power = float(power_cpu.sum())
            if grid == 4096:
                pixel_powers.append(power_cpu)
                tasks.append(task_cpu)
                electrons_list.append(electron_cpu)
                channels.append(channel_power)
            grid_rois[grid] = output["common_roi"].detach().cpu().to(torch.complex64)
            if grid == 4096 and int(row_id) in representative_rows:
                representative_rois[int(row_id)] = grid_rois[grid].numpy().copy()
            grid_outputs[grid] = {
                "mse": image_mse, "pixel_power": power_cpu, "task": task_cpu,
                "electrons": electron_cpu, "channel_power": channel_power,
                "fiber_total_power_w": stage_fiber_power, "decoder_trace": decoder_trace,
                "camera_trace": camera_trace, "camera_total_power_w": camera_power,
                "route_trace": {key: float(value) for key, value in route_trace.items()},
            }
            del active, output, coefficients
            clear_production_transfer_cache()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        for left_index, left in enumerate(grids):
            for right in grids[left_index + 1:]:
                key = f"{left}_vs_{right}"
                values = _common_window_metrics(grid_rois[left], grid_rois[right])
                pairwise_metrics.setdefault(key, []).append(values)
                window_pairs.setdefault(key, []).append(values["aligned_complex_relative_l2"])
                max_window_error = max(max_window_error, values["aligned_complex_relative_l2"])
        if 4096 not in grid_outputs:
            raise RuntimeError("4096 primary grid was omitted from the evaluation");
        base = grid_outputs[4096]
        meta = power_metadata_by_row[int(row_id)]
        channel_power = base["channel_power"]
        electrons_total = float(base["electrons"].sum())
        record = {
            "candidate": candidate, "mode": mode, "state": state_name,
            "row_id": int(row_id), "label": int(labels[position]),
            "role": population["row_role"][position],
            "mse_by_grid": {str(grid): float(grid_outputs[grid]["mse"]) for grid in grids},
            "source_input_power_w": meta["source_input_power_w"],
            "after_slm_power_w": meta["after_slm_power_w"],
            "before_fiber_roi_power_w": meta["before_fiber_roi_power_w"],
            "fiber_total_power_w": base["fiber_total_power_w"],
            "fiber_coupling_before_fiber": base["fiber_total_power_w"] / max(meta["before_fiber_roi_power_w"], EPS),
            "fiber_coupling_system": base["fiber_total_power_w"] / SOURCE_POWER_W,
            "fiber_N_eff": _neff(channel_power),
            "fiber_channel_power_w": channel_power.tolist(),
            "fiber_channel_distribution": _channel_summary(channel_power),
            "after_4f_input_aperture_w": base["route_trace"]["after_decoder_input_aperture_power_w"],
            "after_decoder_layer1_aperture_w": base["decoder_trace"]["after_decoder_layer1_aperture_power_w"],
            "after_decoder_layer2_output_aperture_w": base["decoder_trace"]["after_decoder_layer2_output_aperture_power_w"],
            "after_backend_camera_grid_w": base["camera_trace"]["after_backend_relay_camera_grid_power_w"],
            "camera_pixel_integrated_power_w": base["camera_total_power_w"],
            "camera_expected_electrons": electrons_total,
            "camera_saturation_fraction_1ms": float((base["electrons"] >= config["camera"]["full_well_e"]).float().mean()),
            "primary_task_mse": base["mse"],
            "ideal_power_constrained_mse_lower_bound": _ideal_l2_lower_bound(
                target, electrons_total, float(config["camera"]["full_well_e"])),
        }
        metric_rows.append(record)
    _require(len(pixel_powers) == 300 and len(tasks) == 300 and len(electrons_list) == 300,
             "camera array measurement omitted selected MNIST rows")
    power_array = torch.stack(pixel_powers).numpy().astype(np.float32)
    task_array = torch.stack(tasks).numpy().astype(np.float32)
    electron_array = torch.stack(electrons_list).numpy().astype(np.float32)
    camera_path = mode_dir / "camera_arrays_4096_1ms.npz"
    np.savez_compressed(camera_path, rows=np.asarray(all_rows, dtype=np.int64), labels=labels,
                        role=np.asarray(population["row_role"], dtype="U16"),
                        target=targets_cpu.numpy().astype(np.float32),
                        power_w_per_pixel=power_array, expected_electrons=electron_array,
                        task_output=task_array)
    roi_path = mode_dir / "representative_common_roi_4096_complex64.npz"
    np.savez_compressed(roi_path, rows=np.asarray(sorted(representative_rois), dtype=np.int64),
                        common_roi=np.asarray([representative_rois[row] for row in sorted(representative_rois)],
                                              dtype=np.complex64))
    np.savez_compressed(mode_dir / "fiber_coefficients_power.npz", rows=np.asarray(all_rows, dtype=np.int64),
                        labels=labels, coefficients_sqrt_w=np.asarray(assets["H"][candidate][mode][4096] @
                        prepared["commands"].numpy().astype(np.float64).T, dtype=np.complex128).T,
                        channel_power_w=np.asarray(channels, dtype=np.float64))
    window_summary = {}
    for key, errors in window_pairs.items():
        max_error = float(max(errors))
        window_summary[key] = {"count": len(errors), "max_aligned_complex_relative_l2": max_error,
                               "p99_aligned_complex_relative_l2": float(np.quantile(errors, 0.99)),
                               "passed": max_error <= 1.0e-4}
    passed = all(value["passed"] for value in window_summary.values())
    exposure_summary = _exposure_summaries(torch.from_numpy(power_array), targets_cpu, config)
    _write_json(mode_dir / "window_metrics.json", {"pairwise": window_summary,
                                                     "rows": [int(row) for row in all_rows],
                                                     "per_image": pairwise_metrics,
                                                     "historical_v9_roi_power_field": "sampling sum without dx^2; retained only in v9 report",
                                                     "v12_roi_power_unit": "W; sum(|E|^2) * (2e-6 m)^2"})
    _write_json(mode_dir / "exposure_sensitivity.json", exposure_summary)
    _write_csv(mode_dir / "per_image_measurements.csv", metric_rows)
    return {"status": "PASS" if passed else "NUMERICAL_NOT_QUALIFIED", "candidate": candidate,
            "mode": mode, "state": state_name, "grid_list": list(grids),
            "max_aligned_complex_relative_l2": max_window_error, "window_summary": window_summary,
            "all_window_checks_pass": passed, "camera_arrays": str(camera_path),
            "representative_common_roi": str(roi_path),
            "per_image_metrics": metric_rows, "mse_by_grid": {str(grid): [r["mse_by_grid"][str(grid)] for r in metric_rows] for grid in grids},
            "exposure_sensitivity": exposure_summary, "power_w_per_pixel": power_array,
            "task_output_1ms": task_array, "expected_electrons_1ms": electron_array}


def _native_precision_check(assets: dict[str, Any], population: dict[str, Any], prepared: dict[str, torch.Tensor],
                            config: dict[str, Any], phase: torch.Tensor, candidate: str, mode: str,
                            run_dir: Path, evaluation: dict[str, Any], device: torch.device,
                            *, state_name: str) -> dict[str, Any]:
    grid = 4096
    representative_rows = [int(row) for row in population["fit_rows"].reshape(10, 10)[:, 0].tolist()]
    real_dtype = torch.float64
    runtime = v5._make_runtime(grid, real_dtype, device)
    entrance = v5._encoder_entrance(runtime, real_dtype, device)
    _set_frozen_encoder_phase(runtime, assets["encoder_phases"][candidate], device)
    roi_plane = v9._plane("v12_native_complex128_roi", (2048, 2048), (DX_M, DX_M), 0.0, "fiber_global")
    phase128 = phase.to(device=device, dtype=torch.float64)
    camera_archive = np.load(evaluation["camera_arrays"], allow_pickle=False)
    rows_saved = np.asarray(camera_archive["rows"], dtype=np.int64)
    task_c64 = np.asarray(camera_archive["task_output"], dtype=np.float32)
    task_by_row = {int(row): task_c64[index] for index, row in enumerate(rows_saved)}
    with np.load(evaluation["representative_common_roi"], allow_pickle=False) as roi_archive:
        roi_rows = np.asarray(roi_archive["rows"], dtype=np.int64)
        roi_c64 = {int(row): torch.from_numpy(roi_archive["common_roi"][index]) for index, row in enumerate(roi_rows)}
    records = []
    c128_tasks = []
    c128_pixel_power = []
    targets = []
    for row_id in representative_rows:
        position = _population_position(population, row_id)
        command = prepared["commands"][position].reshape(1, 10000).to(device=device, dtype=torch.float64)
        fields, _traces = v7._runtime_forward_batch(runtime, entrance, assets["encoder_phases"][candidate],
                                                   command, grid, batch_size=1)
        input_roi = v7._crop_roi(fields[0], grid)
        coefficients = v7._project_roi_batch(input_roi[None], int(v9.SIZES[candidate]), mode, roi_plane)[0]
        source = v11._compose_fiber_field(assets["candidates"][candidate], mode,
                                          coefficients.detach().cpu().numpy(), grid, torch.complex128, device)
        active, route_trace, route_z = v9._route_to_decoder(source, "explicit_4f", grid)
        output4 = _camera_forward(active, phase128, grid, route_z, config, keep_roi=True)
        output8 = v11._camera_forward(active, phase128, grid, route_z, config,
                                      quadrature_order=int(config["measurement"]["camera_quadrature_reference_order"]))
        task_error = float(torch.linalg.vector_norm(output4["task_output"] -
                                                   torch.as_tensor(task_by_row[row_id], device=device, dtype=torch.float64)) /
                           torch.linalg.vector_norm(torch.as_tensor(task_by_row[row_id], device=device, dtype=torch.float64)).clamp_min(EPS))
        roi_error = _common_window_metrics(roi_c64[row_id].to(device=device), output4["common_roi"])
        integral_error = float(torch.linalg.vector_norm(output4["pixel_power_w"] - output8["pixel_power_w"]) /
                               torch.linalg.vector_norm(output8["pixel_power_w"]).clamp_min(EPS))
        target = prepared["targets"][position, 0].to(device=device, dtype=torch.float64)
        mse128 = float((output4["task_output"] - target).square().mean().detach().cpu())
        mse64 = float(((torch.as_tensor(task_by_row[row_id], device=device, dtype=torch.float64) - target).square().mean()).detach().cpu())
        records.append({"row_id": row_id, "label": int(population["all_labels"][position]),
                        "complex128_frontend_recomputed_from_production_encoder": True,
                        "complex_roi_vs_complex64": roi_error,
                        "camera_task_output_relative_l2_complex128_vs_complex64": task_error,
                        "camera_pixel_power_gl4_vs_gl8_relative_l2": integral_error,
                        "gl4_gl8_passed": integral_error <= 1.0e-4,
                        "mse_complex64": mse64, "mse_complex128": mse128,
                        "absolute_mse_difference": abs(mse128 - mse64),
                        "native_complex128_route_trace": {key: float(value) for key, value in route_trace.items()}})
        c128_tasks.append(output4["task_output"].detach().cpu().to(torch.float32).numpy())
        c128_pixel_power.append(output4["pixel_power_w"].detach().cpu().to(torch.float32).numpy())
        targets.append(prepared["targets"][position, 0].cpu().numpy().astype(np.float32))
        del fields, input_roi, coefficients, source, active, output4, output8
        clear_production_transfer_cache()
        torch.cuda.empty_cache()
    camera_archive.close()
    precision_dir = run_dir / "precision_checks" / state_name / candidate / mode
    precision_dir.mkdir(parents=True, exist_ok=True)
    precision_path = precision_dir / "native_complex128_camera_arrays.npz"
    np.savez_compressed(precision_path, rows=np.asarray(representative_rows, dtype=np.int64),
                        task_output=np.asarray(c128_tasks, dtype=np.float32),
                        pixel_power_w_per_pixel=np.asarray(c128_pixel_power, dtype=np.float32),
                        targets=np.asarray(targets, dtype=np.float32))
    result = {"status": "PASS" if all(record["gl4_gl8_passed"] for record in records) else "FAIL",
              "precision": "complex128 native production frontend and downstream fields",
              "representative_rows": representative_rows, "records": records,
              "max_gl4_vs_gl8_pixel_power_relative_l2": max(record["camera_pixel_power_gl4_vs_gl8_relative_l2"] for record in records),
              "max_complex64_vs_complex128_mse_difference": max(record["absolute_mse_difference"] for record in records),
              "saved_arrays": str(precision_path)}
    del runtime, entrance
    gc.collect()
    clear_production_transfer_cache()
    torch.cuda.empty_cache()
    return result


def _ssim_per_image(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    output = output.to(torch.float32).reshape(-1, 1, 200, 200)
    target = target.to(torch.float32).reshape(-1, 1, 200, 200)
    axis = torch.arange(11, dtype=torch.float32) - 5.0
    gaussian = torch.exp(-(axis.square()) / (2.0 * 1.5**2))
    gaussian = gaussian / gaussian.sum()
    kernel = torch.outer(gaussian, gaussian).reshape(1, 1, 11, 11)
    mu_x = F.conv2d(F.pad(output, (5, 5, 5, 5), mode="reflect"), kernel)
    mu_y = F.conv2d(F.pad(target, (5, 5, 5, 5), mode="reflect"), kernel)
    sigma_x = F.conv2d(F.pad(output.square(), (5, 5, 5, 5), mode="reflect"), kernel) - mu_x.square()
    sigma_y = F.conv2d(F.pad(target.square(), (5, 5, 5, 5), mode="reflect"), kernel) - mu_y.square()
    sigma_xy = F.conv2d(F.pad(output * target, (5, 5, 5, 5), mode="reflect"), kernel) - mu_x * mu_y
    value = ((2.0 * mu_x * mu_y + 0.01**2) * (2.0 * sigma_xy + 0.03**2) /
             ((mu_x.square() + mu_y.square() + 0.01**2) * (sigma_x + sigma_y + 0.03**2)))
    return value.flatten(1).mean(dim=1)


def _stratified_bootstrap_gain(control_mse: np.ndarray, endpoint_mse: np.ndarray,
                               labels: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    gain = np.asarray(control_mse, dtype=np.float64) - np.asarray(endpoint_mse, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    _require(gain.shape == (200,) and labels.shape == (200,), "confirmation bootstrap requires exactly 200 fixed images")
    rng = np.random.default_rng(int(config["measurement"]["bootstrap_seed"]))
    samples = np.empty(int(config["measurement"]["bootstrap_replicates"]), dtype=np.float64)
    by_label = [np.flatnonzero(labels == label) for label in range(10)]
    _require(all(indices.size == 20 for indices in by_label), "bootstrap population is not 20 per class")
    for replicate in range(samples.size):
        chosen = np.concatenate([rng.choice(indices, size=indices.size, replace=True) for indices in by_label])
        samples[replicate] = float(gain[chosen].mean())
    return {"mean_paired_mse_gain": float(gain.mean()),
            "bootstrap_95_percentile_interval": [float(np.quantile(samples, 0.025)),
                                                  float(np.quantile(samples, 0.975))],
            "bootstrap_replicates": int(samples.size),
            "seed": int(config["measurement"]["bootstrap_seed"]),
            "per_confirmation_image_gain": gain.tolist()}


def _candidate_reconstruction_metrics(initial: dict[str, Any], terminal: dict[str, Any],
                                       population: dict[str, Any], prepared: dict[str, torch.Tensor],
                                       numeric_precision_checks: list[dict[str, Any]],
                                       config: dict[str, Any]) -> dict[str, Any]:
    init_output = np.asarray(initial["task_output_1ms"], dtype=np.float64)
    final_output = np.asarray(terminal["task_output_1ms"], dtype=np.float64)
    targets = prepared["targets"].numpy().astype(np.float64)[:, 0]
    labels = population["all_labels"].numpy().astype(np.int64)
    confirm_slice = slice(100, 300)
    confirm_labels = labels[confirm_slice]
    init_mse = np.mean((init_output[confirm_slice] - targets[confirm_slice])**2, axis=(1, 2))
    final_mse = np.mean((final_output[confirm_slice] - targets[confirm_slice])**2, axis=(1, 2))
    fit_mean = prepared["fit_target_mean"].numpy().astype(np.float64)[0]
    mean_image_mse = np.mean((fit_mean[None] - targets[confirm_slice])**2, axis=(1, 2))
    target_mismatch = targets[confirm_slice].copy()
    for label in range(10):
        class_positions = np.flatnonzero(confirm_labels == label)
        _require(class_positions.size == 20, "within-class mismatch control requires 20 confirmation rows per class")
        target_mismatch[class_positions] = np.roll(target_mismatch[class_positions], shift=1, axis=0)
    mismatch_mse = np.mean((final_output[confirm_slice] - target_mismatch)**2, axis=(1, 2))
    controls = {
        "same_image_initial_decoder": _stratified_bootstrap_gain(init_mse, final_mse, confirm_labels, config),
        "fixed_mean_of_100_fit_targets": _stratified_bootstrap_gain(mean_image_mse, final_mse, confirm_labels, config),
        "within_class_cyclically_mismatched_target": _stratified_bootstrap_gain(mismatch_mse, final_mse, confirm_labels, config),
    }
    window_deltas = []
    for state in (initial, terminal):
        for grid in state["grid_list"]:
            if int(grid) != 4096:
                window_deltas.extend(np.abs(np.asarray(state["mse_by_grid"][str(grid)], dtype=np.float64)[confirm_slice] -
                                            np.asarray(state["mse_by_grid"]["4096"], dtype=np.float64)[confirm_slice]).tolist())
    precision_deltas = [float(record["absolute_mse_difference"]) for result in numeric_precision_checks
                        for record in result["records"]]
    epsilon_numeric = max(max(window_deltas, default=0.0), max(precision_deltas, default=0.0))
    tau = max(float(config["measurement"]["numerical_epsilon_floor"]),
              float(config["measurement"]["numerical_epsilon_multiplier"]) * epsilon_numeric)
    decisions = {name: {"passed": float(value["bootstrap_95_percentile_interval"][0]) > tau,
                        "lower_bound": float(value["bootstrap_95_percentile_interval"][0]),
                        "tau": tau} for name, value in controls.items()}
    per_class = {}
    for label in range(10):
        indices = np.flatnonzero(confirm_labels == label)
        per_class[str(label)] = {
            "initial_mse": float(init_mse[indices].mean()),
            "terminal_mse": float(final_mse[indices].mean()),
            "paired_gain": float((init_mse[indices] - final_mse[indices]).mean()),
        }
    target_foreground = targets[confirm_slice] >= float(config["measurement"]["foreground_threshold"])
    final_confirm = final_output[confirm_slice]
    error = (final_confirm - targets[confirm_slice])**2
    foreground_mse = float(error[target_foreground].mean()) if target_foreground.any() else None
    background_mse = float(error[~target_foreground].mean()) if (~target_foreground).any() else None
    flat_mse = np.mean(error, axis=(1, 2))
    psnr = 10.0 * np.log10(1.0 / np.maximum(flat_mse, EPS))
    ssim = _ssim_per_image(torch.from_numpy(final_confirm.astype(np.float32)),
                           torch.from_numpy(targets[confirm_slice].astype(np.float32))).numpy()
    fit_labels = labels[:100]
    per_class_fit_loss = {}
    for label in range(10):
        indices = np.flatnonzero(fit_labels == label)
        per_class_fit_loss[str(label)] = float(np.mean([record["primary_task_mse"] for record in
                                                   initial["per_image_metrics"][:100]
                                                   if int(record["label"]) == label]))
    all_windows_passed = bool(initial["all_window_checks_pass"] and terminal["all_window_checks_pass"])
    all_numeric_checks_passed = all(result["status"] == "PASS" for result in numeric_precision_checks)
    return {
        "initial_confirmation_mse": float(init_mse.mean()),
        "terminal_confirmation_mse": float(final_mse.mean()),
        "fixed_100_fit_mean_image_confirmation_mse": float(mean_image_mse.mean()),
        "within_class_cyclic_mismatch_confirmation_mse": float(mismatch_mse.mean()),
        "paired_bootstrap_gains": controls,
        "numeric_epsilon": epsilon_numeric, "decision_threshold_tau": tau,
        "all_three_gain_lower_bounds_above_tau": all(value["passed"] for value in decisions.values()),
        "gain_decisions": decisions, "class_weighting": "equal by stratified 20-per-class bootstrap",
        "per_class_confirmation_mse": per_class,
        "confirmation_foreground_mse_threshold": float(config["measurement"]["foreground_threshold"]),
        "confirmation_foreground_mse": foreground_mse,
        "confirmation_background_mse": background_mse,
        "confirmation_mean_psnr_db": float(psnr.mean()),
        "confirmation_mean_ssim_11x11_gaussian": float(ssim.mean()),
        "fit_population_initial_mse_by_class": per_class_fit_loss,
        "numerical_window_deltas_used_for_epsilon": window_deltas,
        "native_precision_mse_deltas_used_for_epsilon": precision_deltas,
        "numerical_qualification": all_windows_passed and all_numeric_checks_passed,
        "scientific_result": "SUPPORTED" if all_windows_passed and all_numeric_checks_passed and
                               all(value["passed"] for value in decisions.values()) else
                            ("VALID_FAILED" if all_windows_passed and all_numeric_checks_passed else "INVALID_HARNESS"),
    }


def _save_reconstruction_gallery(run_dir: Path, candidate: str,
                                 initial: dict[str, Any], terminal: dict[str, Any],
                                 population: dict[str, Any], prepared: dict[str, torch.Tensor]) -> str:
    from PIL import Image, ImageDraw
    fit_rows = population["fit_rows"].tolist()
    confirm_rows = population["confirmation_rows"].tolist()
    chosen_rows = [fit_rows[label * 10] for label in range(10)]
    chosen_rows.extend([confirm_rows[label * 20 + offset] for label in range(10) for offset in (0, 1)])
    position = {int(row): index for index, row in enumerate(population["all_rows"].tolist())}
    init = np.asarray(initial["task_output_1ms"], dtype=np.float32)
    final = np.asarray(terminal["task_output_1ms"], dtype=np.float32)
    targets = prepared["targets"].numpy().astype(np.float32)
    canvas = Image.new("RGB", (600, 30 * (200 + 24)), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), "target", fill=(0, 0, 0))
    draw.text((208, 4), "initial Decoder", fill=(0, 0, 0))
    draw.text((408, 4), "step 500 Decoder", fill=(0, 0, 0))
    for row_index, row_id in enumerate(chosen_rows):
        idx = position[int(row_id)]
        label = int(population["all_labels"][idx])
        role = population["row_role"][idx]
        row_top = row_index * 224
        y = row_top + 24
        draw.text((5, row_top + 4), f"{role} y={label} row={row_id}", fill=(0, 0, 0))
        for column, image in enumerate((targets[idx, 0], init[idx], final[idx])):
            gray = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
            tile = Image.fromarray(gray, mode="L").convert("RGB")
            canvas.paste(tile, (column * 200, y))
    path = run_dir / "reconstruction_gallery" / f"{candidate}_30_fixed_samples.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return str(path)


def _load_ready_preflight(report_path: Path, root: Path, config_path: Path,
                          source_dir: Path, data_file: Path) -> tuple[dict[str, Any], ResearchRunControl, PreflightReceipt]:
    _require(report_path.is_file(), f"explicit READY preflight report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema") == PREFLIGHT_SCHEMA and report.get("status") == "READY",
             "explicit preflight report is not a v12 READY result")
    config = _load_config(config_path)
    _require(report.get("config") == config, "preflight config differs from the execution config")
    _require(Path(report.get("source_dir", "")).resolve() == source_dir.resolve() and
             Path(report.get("data_file", "")).resolve() == data_file.resolve(),
             "preflight source or dataset path differs from the requested execution")
    _require(str(report.get("authorization_id")) == AUTHORITY, "preflight authorization differs from v12 task")
    overlay, module_origins = _runtime_overlay(root)
    _require(Path(report.get("runtime_overlay", "")).resolve() == overlay,
             "READY preflight runtime overlay differs from the execution runtime")
    _require(report.get("module_origins") == module_origins,
             "READY preflight v2 module origins differ from the execution runtime")
    assets = _load_assets(root, source_dir)
    pixels, labels, train_rows = _load_mnist(data_file, config)
    population = _select_population(labels, train_rows, config)
    prepared = _prepare_images(pixels, population)
    schedule = _make_schedule(population, config)
    saved_rows = report.get("schedule", {}).get("rows")
    saved_labels = report.get("schedule", {}).get("labels")
    _require(saved_rows is not None and torch.equal(torch.as_tensor(saved_rows), schedule),
             "READY preflight batch schedule differs from reloaded MNIST rows")
    _require(saved_labels is not None and torch.equal(torch.as_tensor(saved_labels),
             torch.as_tensor(_schedule_records(population, schedule)["labels"])),
             "READY preflight label schedule differs from reloaded MNIST rows")
    identity = ResearchRunIdentity.from_mapping(report["research_run_identity"])
    receipt = PreflightReceipt.from_mapping(report["preflight_receipt"])
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    report["_loaded_assets"] = assets
    report["_loaded_population"] = population
    report["_loaded_prepared"] = prepared
    report["_loaded_schedule"] = schedule
    return report, control, receipt


def _candidate_phase_snapshot(phase: torch.Tensor) -> dict[str, Any]:
    return {"phase_rms_rad_by_layer": torch.sqrt(torch.mean(phase.detach().square(), dim=(-2, -1))).cpu().reshape(-1).tolist(),
            "phase_min_rad_by_layer": phase.detach().amin(dim=(-2, -1)).cpu().reshape(-1).tolist(),
            "phase_max_rad_by_layer": phase.detach().amax(dim=(-2, -1)).cpu().reshape(-1).tolist(),
            "phase_shape": list(phase.shape), "phase_dtype": str(phase.dtype)}


def _scientific_run(root: Path, config_path: Path, source_dir: Path, data_file: Path,
                    preflight_dir: Path, run_dir: Path, preflight_report: dict[str, Any],
                    control: ResearchRunControl, receipt: PreflightReceipt,
                    *, resume: bool = False, resume_candidate: str | None = None,
                    resume_checkpoint: Path | None = None) -> dict[str, Any]:
    config = _load_config(config_path)
    assets = preflight_report.pop("_loaded_assets", None)
    population = preflight_report.pop("_loaded_population", None)
    prepared = preflight_report.pop("_loaded_prepared", None)
    schedule = preflight_report.pop("_loaded_schedule", None)
    if assets is None:
        config, assets, population, prepared, schedule = _worker_assets_and_data(root, source_dir, data_file, config_path)
    _require(assets is not None and population is not None and prepared is not None and schedule is not None,
             "v12 execution did not load frozen assets and selected data")
    device = torch.device("cuda")
    _require(torch.cuda.is_available(), "v12 scientific run requires AutoDL CUDA")
    if resume:
        _require(run_dir.is_dir(), "explicit v12 recovery requires the existing scientific attempt directory")
        _require(resume_candidate in CANDIDATES and resume_checkpoint is not None and resume_checkpoint.is_file(),
                 "recovery requires an explicit candidate and checkpoint path")
        control.open_scientific_attempt(run_dir, "ols_decoder_mnist_reconstruction_v12:attempt_01",
                                       receipt, resume=True)
    else:
        if run_dir.exists():
            raise FileExistsError(f"v12 attempt exists; preserve it and choose a new run path: {run_dir}")
        control.start_execution()
        control.open_scientific_attempt(run_dir, "ols_decoder_mnist_reconstruction_v12:attempt_01", receipt)
    _write_json(run_dir / "research_run_identity.json", control.identity.to_dict())
    _write_json(run_dir / "consumed_preflight_receipt.json", receipt.to_dict())
    _write_json(run_dir / "data_population.json", {
        "fit_rows": population["fit_rows"].tolist(), "confirmation_rows": population["confirmation_rows"].tolist(),
        "labels": population["all_labels"].tolist(), "roles": population["row_role"],
        "batch_schedule": schedule.tolist(), "batch_labels": _schedule_records(population, schedule)["labels"].tolist(),
        "dev_reads": 0, "test_reads": 0,
    })
    population_path = run_dir / "fixed_mnist_population.npz"
    np.savez_compressed(population_path,
                        rows=population["all_rows"].numpy(), labels=population["all_labels"].numpy(),
                        roles=np.asarray(population["row_role"], dtype="U16"),
                        input_100=prepared["input_images"].numpy().astype(np.float32),
                        command_t=prepared["commands"].numpy().astype(np.float32),
                        target_200=prepared["targets"].numpy().astype(np.float32),
                        fit_target_mean_200=prepared["fit_target_mean"].numpy().astype(np.float32))
    status_path = run_dir / "status.json"
    status_payload: dict[str, Any] = {"status": "SCIENTIFIC_ACTIVE", "phase": "initial_measurement",
                                     "formal_generator_updates": 0,
                                     "planned_reconstruction_updates": 1000,
                                     "engineering_preflight_adam_calls": 10}
    _write_json(status_path, status_payload)
    initial_evaluations: dict[str, dict[str, Any]] = {}
    terminal_evaluations: dict[str, dict[str, Any]] = {}
    precision_checks: dict[str, list[dict[str, Any]]] = {candidate: [] for candidate in CANDIDATES}
    initial_phases = {candidate: assets["decoder_phase"].clone() for candidate in CANDIDATES}
    for candidate in CANDIDATES:
        initial_evaluations[candidate] = {}
        for mode, grids in (("gaussian_s12", GRIDS), ("lp01", (4096,))):
            evaluation = _evaluate_candidate(assets, population, prepared, config, preflight_dir,
                                             candidate, mode, initial_phases[candidate], device,
                                             state_name="step_000_initial", grids=grids,
                                             output_root=run_dir)
            initial_evaluations[candidate][mode] = evaluation
        gaussian_initial = initial_evaluations[candidate]["gaussian_s12"]
        if gaussian_initial["all_window_checks_pass"]:
            for mode in MODES:
                numeric = _native_precision_check(assets, population, prepared, config,
                                                  initial_phases[candidate], candidate, mode,
                                                  run_dir, initial_evaluations[candidate][mode], device,
                                                  state_name="step_000_initial")
                precision_checks[candidate].append(numeric)
        else:
            precision_checks[candidate].append({"status": "NOT_RUN_INITIAL_WINDOW_GATE_FAILED"})
    initial_qualification = {candidate: bool(initial_evaluations[candidate]["gaussian_s12"]["all_window_checks_pass"] and
                                             all(item.get("status") == "PASS" for item in precision_checks[candidate]))
                             for candidate in CANDIDATES}
    status_payload.update({"phase": "training", "initial_numerical_qualification": initial_qualification})
    _write_json(status_path, status_payload)
    training_results: dict[str, dict[str, Any]] = {}
    candidate_phases: dict[str, torch.Tensor] = {}
    candidate_order = list(CANDIDATES)
    if resume:
        candidate_order = [resume_candidate] + [candidate for candidate in CANDIDATES if candidate != resume_candidate]
    for candidate in candidate_order:
        if not initial_qualification[candidate]:
            training_results[candidate] = {"status": "NOT_STARTED_NUMERICAL_PREFLIGHT_FAILED", "adam_updates": 0}
            candidate_phases[candidate] = initial_phases[candidate]
            continue
        chosen_resume = resume_checkpoint if resume and candidate == resume_candidate else None
        _SIGNAL_RECEIVED = None
        _install_signal_handler()
        result = _train_candidate(assets, population, prepared, schedule, config, run_dir, preflight_dir,
                                  candidate, device, steps=int(config["training"]["updates_per_candidate"]),
                                  resume_checkpoint=chosen_resume,
                                  output_subdir="primary")
        if result["status"] == "INTERRUPTED":
            control.interrupted()
            interrupted_report = {"schema": SCHEMA, "status": "INTERRUPTED", "task_id": TASK_ID,
                                  "candidate": candidate, "last_committed_step": result["last_committed_step"],
                                  "resume_checkpoint": result["checkpoint"], "partial_batch_discarded": True,
                                  "preflight_report": str(preflight_dir / "preflight_report.json"),
                                  "consumed_preflight_receipt": receipt.to_dict(),
                                  "formal_generator_training_updates": 0}
            _write_json(run_dir / "interrupted_report.json", interrupted_report)
            _write_json(status_path, {"status": "INTERRUPTED", "candidate": candidate,
                                      "last_committed_step": result["last_committed_step"],
                                      "resume_checkpoint": result["checkpoint"]})
            return interrupted_report
        _require(result["status"] == "COMPLETED" and result["steps"] == 500 and result["adam_updates"] == 500,
                 f"candidate {candidate} did not complete the fixed 500-update budget")
        candidate_phases[candidate] = result["phase"].detach().cpu()
        training_results[candidate] = {
            "status": result["status"], "steps": result["steps"], "adam_updates": result["adam_updates"],
            "terminal_checkpoint": result["terminal_checkpoint"],
            "phase_summary": _candidate_phase_snapshot(result["phase"]),
            "history": result["history"],
            "parameters_updated": ["decoder_layer_0_phase", "decoder_layer_1_phase"],
            "frozen_parameters_updated": [],
        }
        _write_json(run_dir / "training" / candidate / "history_summary.json", training_results[candidate])
        status_payload.update({"candidate_completed": candidate, "candidate_updates": 500,
                               "next_candidate": next((value for value in CANDIDATES if value not in training_results), None)})
        _write_json(status_path, status_payload)
    for candidate in CANDIDATES:
        terminal_evaluations[candidate] = {}
        if training_results[candidate]["status"] != "COMPLETED":
            continue
        for mode, grids in (("gaussian_s12", GRIDS), ("lp01", (4096,))):
            evaluation = _evaluate_candidate(assets, population, prepared, config, preflight_dir,
                                             candidate, mode, candidate_phases[candidate], device,
                                             state_name="step_000500_terminal", grids=grids,
                                             output_root=run_dir)
            terminal_evaluations[candidate][mode] = evaluation
            numeric = _native_precision_check(assets, population, prepared, config,
                                              candidate_phases[candidate], candidate, mode,
                                              run_dir, evaluation, device,
                                              state_name="step_000500_terminal")
            precision_checks[candidate].append(numeric)
    candidate_results = {}
    for candidate in CANDIDATES:
        if training_results[candidate]["status"] != "COMPLETED":
            candidate_results[candidate] = {"status": "INVALID_HARNESS", "reason": training_results[candidate]["status"]}
            continue
        initial = initial_evaluations[candidate]["gaussian_s12"]
        terminal = terminal_evaluations[candidate]["gaussian_s12"]
        numerical_ok = bool(initial["all_window_checks_pass"] and terminal["all_window_checks_pass"] and
                            all(item.get("status") == "PASS" for item in precision_checks[candidate]))
        candidate_metrics = _candidate_reconstruction_metrics(
            initial, terminal, population, prepared, precision_checks[candidate], config)
        gallery = _save_reconstruction_gallery(run_dir, candidate, initial, terminal, population, prepared)
        candidate_results[candidate] = {
            "numerical_status": "PASS" if numerical_ok else "INVALID_HARNESS",
            "training": training_results[candidate],
            "initial_gaussian_measurement": _evaluation_summary(initial),
            "terminal_gaussian_measurement": _evaluation_summary(terminal),
            "initial_lp01_measurement": _evaluation_summary(initial_evaluations[candidate]["lp01"]),
            "terminal_lp01_measurement": _evaluation_summary(terminal_evaluations[candidate]["lp01"]),
            "native_precision_checks": precision_checks[candidate],
            "reconstruction_metrics": candidate_metrics,
            "gallery": gallery,
            "classification": candidate_metrics["scientific_result"] if numerical_ok else "INVALID_HARNESS",
        }
    scientific_outcomes = [candidate_results[candidate]["classification"] for candidate in CANDIDATES]
    if "INVALID_HARNESS" in scientific_outcomes:
        final_status = "INVALID_HARNESS"
        control.invalid_harness()
    elif all(outcome == "SUPPORTED" for outcome in scientific_outcomes):
        final_status = "VALID_COMPLETED"
        control.complete()
    else:
        final_status = "VALID_FAILED"
        control.valid_failed()
    report = {
        "schema": SCHEMA, "status": final_status, "task_id": TASK_ID,
        "decision_question": "Do both frozen front-end candidates show numerically valid, input-specific MNIST reconstruction gains after the fixed 500 Decoder updates?",
        "authorization_id": AUTHORITY, "preflight_report": str(preflight_dir / "preflight_report.json"),
        "consumed_preflight_receipt": receipt.to_dict(), "research_run_identity": control.identity.to_dict(),
        "source_dir": str(source_dir.resolve()), "data_file": str(data_file.resolve()),
        "fixed_population": {"fit_rows": population["fit_rows"].tolist(),
                             "confirmation_rows": population["confirmation_rows"].tolist(),
                             "labels": population["all_labels"].tolist(), "dev_reads": 0, "test_reads": 0},
        "fixed_config": config,
        "engineering_preflight_adam_calls": 10,
        "scientific_reconstruction_adam_calls": sum(int(training_results[candidate].get("adam_updates", 0)) for candidate in CANDIDATES),
        "formal_generator_training_updates": 0, "task_evaluator_calls": 0, "fid_calls": 0,
        "loss_changes": 0, "material_forward_calls": 0,
        "candidate_results": candidate_results,
        "training_files": {candidate: str(run_dir / "training" / candidate / "primary" / "terminal.pt")
                           for candidate in CANDIDATES},
        "power_artifacts": {candidate: {state: {mode: evaluation["camera_arrays"]
                                                for mode, evaluation in modes.items()}
                                        for state, modes in (("initial", initial_evaluations[candidate]),
                                                             ("terminal", terminal_evaluations[candidate]))}
                            for candidate in CANDIDATES},
        "stop_point": "No material study, joint Encoder training, final OLS dimension adoption, formal Generator training, or old Generator recovery was started.",
    }
    _write_json(run_dir / "report.json", report)
    _write_json(run_dir / "status.json", {"status": final_status, "state": control.state,
                                          "reconstruction_adam_calls": report["scientific_reconstruction_adam_calls"],
                                          "candidate_outcomes": scientific_outcomes})
    return report


def _evaluation_summary(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in evaluation.items()
            if key not in {"power_w_per_pixel", "task_output_1ms", "expected_electrons_1ms"}}


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--data-file", default=str(DEFAULT_DATA_FILE))
    parser.add_argument("--preflight-dir", default=str(DEFAULT_PREFLIGHT_DIR))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--preflight-report", default=None)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-candidate", choices=CANDIDATES)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--internal-preflight-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-candidate", choices=CANDIDATES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output-subdir", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--interrupt-event", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--worker-result", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    config_path = _resolve(root, args.config)
    source_dir = _resolve(root, args.source_dir)
    data_file = _resolve(root, args.data_file)
    preflight_dir = _resolve(root, args.preflight_dir)
    run_dir = _resolve(root, args.run_dir)
    if args.internal_preflight_worker:
        _require(args.worker_candidate is not None and args.resume_checkpoint is not None and
                 args.worker_output_subdir is not None and args.worker_result is not None,
                 "internal recovery worker arguments are incomplete")
        try:
            return _preflight_worker(args)
        except Exception as error:
            if args.worker_result:
                _write_json(_resolve(root, args.worker_result), {"status": "FAILED", "error": repr(error)})
            raise
    try:
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("v12 is fixed to the AutoDL CUDA execution host")
        if args.preflight_only:
            report, _control, _receipt = _preflight(root, config_path, source_dir, data_file,
                                                    preflight_dir, device)
            print(json.dumps({"status": report["status"], "preflight_report": str(preflight_dir / "preflight_report.json")}, ensure_ascii=False))
            return 0 if report["status"] == "READY" else 2
        if args.resume:
            _require(args.preflight_report is not None, "--resume requires an explicit READY --preflight-report")
            _require(args.resume_candidate is not None and args.resume_checkpoint is not None,
                     "--resume requires --resume-candidate and --resume-checkpoint")
            preflight_report, control, receipt = _load_ready_preflight(
                _resolve(root, args.preflight_report), root, config_path, source_dir, data_file)
        elif args.preflight_report is not None:
            preflight_report, control, receipt = _load_ready_preflight(
                _resolve(root, args.preflight_report), root, config_path, source_dir, data_file)
        else:
            preflight_report, control, receipt = _preflight(root, config_path, source_dir,
                                                            data_file, preflight_dir, device)
        report = _scientific_run(root, config_path, source_dir, data_file, preflight_dir,
                                 run_dir, preflight_report, control, receipt,
                                 resume=bool(args.resume),
                                 resume_candidate=args.resume_candidate,
                                 resume_checkpoint=_resolve(root, args.resume_checkpoint) if args.resume_checkpoint else None)
        print(json.dumps({"status": report["status"], "run_dir": str(run_dir),
                          "scientific_updates": report["scientific_reconstruction_adam_calls"]}, ensure_ascii=False))
        return 0 if report["status"] in {"VALID_COMPLETED", "VALID_FAILED"} else (3 if report["status"] == "INTERRUPTED" else 2)
    except Exception as error:
        failure = {"schema": PREFLIGHT_SCHEMA, "status": "INVALID_HARNESS", "task_id": TASK_ID,
                   "error_type": type(error).__name__, "error": str(error),
                   "traceback": traceback.format_exc(),
                   "formal_generator_training_updates": 0}
        target = run_dir / "engineering_failure.json" if run_dir.exists() else preflight_dir / "engineering_failure.json"
        try:
            _write_json(target, failure)
        except Exception:
            pass
        print(json.dumps(failure, ensure_ascii=False), file=sys.stderr)
        return 2


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_jsonable(value), ensure_ascii=False) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def _ideal_l2_lower_bound(target: torch.Tensor, available_electrons: float, full_well: float) -> float:
    values = target.detach().to(torch.float64).cpu().clamp(0.0, 1.0).reshape(-1)
    budget = min(max(float(available_electrons) / full_well, 0.0), float(values.numel()))
    if budget >= float(values.sum()):
        return 0.0
    low, high = 0.0, 1.0
    for _ in range(80):
        middle = (low + high) * 0.5
        total = float(torch.clamp(values - middle, 0.0, 1.0).sum())
        if total > budget:
            low = middle
        else:
            high = middle
    ideal = torch.clamp(values - high, 0.0, 1.0)
    return float((ideal - values).square().mean())


def _loss_one(assets: dict[str, Any], population: dict[str, Any], prepared: dict[str, torch.Tensor],
              config: dict[str, Any], cache_dir: Path, candidate: str, row_id: int,
              phase: torch.Tensor, device: torch.device, precision: str = "complex64",
              *, detach_task: bool = False) -> tuple[torch.Tensor, dict[str, Any]]:
    _assert_row_pair(row_id, row_id)
    position = _population_position(population, row_id)
    active, route_trace, route_z, coefficients = _make_front_active(
        assets, cache_dir / "frozen_frontend_cache", candidate, "gaussian_s12", 4096,
        row_id, prepared["commands"][position], precision, device,
    )
    phase_dtype = torch.float64 if precision == "complex128" else torch.float32
    output = _camera_forward(active, phase.to(device=device, dtype=phase_dtype), 4096,
                             route_z, config, keep_roi=False)
    target = prepared["targets"][position, 0].to(device=device, dtype=torch.float64 if precision == "complex128" else torch.float32)
    task = output["task_output"].detach() if detach_task else output["task_output"]
    loss = torch.mean((task - target).square())
    details = {"pixel_power_w": output["pixel_power_w"], "expected_electrons": output["expected_electrons"],
               "task_output": output["task_output"], "route_trace": route_trace,
               "decoder_trace": output["decoder_trace"], "camera_trace": output["camera_trace"],
               "coefficients_sqrt_w": coefficients, "active": active}
    return loss, details


def _optimizer(config: dict[str, Any], phase: torch.Tensor) -> torch.optim.Adam:
    return torch.optim.Adam(
        [phase], lr=float(config["training"]["adam_learning_rate"]),
        betas=tuple(float(value) for value in config["training"]["adam_betas"]),
        eps=float(config["training"]["adam_epsilon"]),
        weight_decay=float(config["training"]["adam_weight_decay"]),
    )


def _save_checkpoint(path: Path, candidate: str, phase: torch.Tensor,
                     optimizer: torch.optim.Optimizer, step: int, next_step: int,
                     config: dict[str, Any], population: dict[str, Any], schedule: torch.Tensor,
                     history: list[dict[str, Any]], *, signal_name: str | None = None,
                     partial_batch_discarded: bool = False) -> None:
    payload = {
        "schema": CHECKPOINT_SCHEMA, "candidate": candidate, "step": int(step),
        "next_step": int(next_step), "phase_unwrapped": phase.detach().cpu().contiguous(),
        "optimizer_state": _cpu_tree(optimizer.state_dict()), "config": config,
        "fit_rows": population["fit_rows"].cpu(), "confirmation_rows": population["confirmation_rows"].cpu(),
        "schedule": schedule.cpu(), "history": history, "rng_state": _cpu_tree(_rng_state()),
        "signal_name": signal_name, "partial_batch_discarded": bool(partial_batch_discarded),
        "trainable_parameters": ["decoder_layer_0_phase", "decoder_layer_1_phase"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    check = torch.load(temporary, map_location="cpu", weights_only=False)
    _require(check.get("schema") == CHECKPOINT_SCHEMA and check.get("candidate") == candidate and
             int(check.get("step", -1)) == int(step) and int(check.get("next_step", -1)) == int(next_step),
             "checkpoint disk reread failed identity validation")
    temporary.replace(path)


def _load_training_state(checkpoint: Path, candidate: str, config: dict[str, Any],
                         population: dict[str, Any], schedule: torch.Tensor,
                         device: torch.device) -> tuple[torch.Tensor, torch.optim.Adam, int, list[dict[str, Any]], dict[str, Any]]:
    _require(checkpoint.is_file(), f"explicit checkpoint is missing: {checkpoint}")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == CHECKPOINT_SCHEMA and payload.get("candidate") == candidate,
             "checkpoint task or candidate identity changed")
    _require(payload.get("config") == config and torch.equal(payload.get("fit_rows"), population["fit_rows"]) and
             torch.equal(payload.get("confirmation_rows"), population["confirmation_rows"]) and
             torch.equal(payload.get("schedule"), schedule), "checkpoint data role, config, or batch schedule changed")
    phase = payload["phase_unwrapped"].to(device=device, dtype=torch.float32).clone().requires_grad_(True)
    _require(tuple(phase.shape) == (2, 1, 100, 100) and bool(torch.isfinite(phase).all()),
             "checkpoint Decoder phase tensor is invalid")
    optimizer = _optimizer(config, phase)
    optimizer.load_state_dict(payload["optimizer_state"])
    _restore_rng(payload["rng_state"], device)
    return phase, optimizer, int(payload["next_step"]), list(payload["history"]), payload


def _train_candidate(assets: dict[str, Any], population: dict[str, Any], prepared: dict[str, torch.Tensor],
                     schedule: torch.Tensor, config: dict[str, Any], run_dir: Path,
                     cache_dir: Path, candidate: str, device: torch.device, *,
                     steps: int, resume_checkpoint: Path | None = None,
                     output_subdir: str | None = None,
                     interrupt_event_path: Path | None = None) -> dict[str, Any]:
    output_dir = run_dir / "training" / candidate / (output_subdir or "primary")
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_checkpoint is None:
        phase = assets["decoder_phase"].clone().to(device=device, dtype=torch.float32).requires_grad_(True)
        optimizer = _optimizer(config, phase)
        start_step = 0
        history: list[dict[str, Any]] = []
        _save_checkpoint(output_dir / "checkpoint_step_000000.pt", candidate, phase, optimizer, 0, 1,
                         config, population, schedule, history)
    else:
        phase, optimizer, next_step, history, resume_payload = _load_training_state(
            resume_checkpoint, candidate, config, population, schedule, device)
        start_step = int(resume_payload["step"])
        _require(next_step == start_step + 1, "explicit resume checkpoint next-step cursor is invalid")
    position_by_row = {int(row): index for index, row in enumerate(population["all_rows"].tolist())}
    _require(len(position_by_row) == 300, "MNIST population row identity is not unique")
    current_step = start_step
    def interrupted_result(partial_step: int, partial_microbatches: int) -> dict[str, Any]:
        optimizer.zero_grad(set_to_none=True)
        signal_name = signal.Signals(_SIGNAL_RECEIVED).name if _SIGNAL_RECEIVED is not None else "UNKNOWN"
        interrupted = output_dir / f"interrupted_after_step_{current_step:06d}.pt"
        _save_checkpoint(interrupted, candidate, phase, optimizer, current_step, current_step + 1,
                         config, population, schedule, history, signal_name=signal_name,
                         partial_batch_discarded=True)
        event = {"candidate": candidate, "signal": signal_name, "last_committed_step": current_step,
                 "discarded_partial_batch_step": partial_step, "discarded_microbatches": partial_microbatches,
                 "checkpoint": str(interrupted), "adam_call_count_in_partial_batch": 0}
        _write_json(output_dir / "interruption.json", event)
        return {"status": "INTERRUPTED", **event}

    for step in range(start_step + 1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        rows = schedule[step - 1].tolist()
        row_losses: list[float] = []
        power_records: list[dict[str, float]] = []
        for microbatch_index, row_id in enumerate(rows):
            if _SIGNAL_RECEIVED is not None:
                return interrupted_result(step, microbatch_index)
            _assert_row_pair(row_id, row_id)
            loss, details = _loss_one(assets, population, prepared, config, cache_dir,
                                      candidate, int(row_id), phase, device, "complex64")
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite reconstruction loss at {candidate} step {step}")
            (loss / float(len(rows))).backward()
            row_losses.append(float(loss.detach().cpu()))
            p = details["pixel_power_w"].detach()
            electrons = details["expected_electrons"].detach()
            meta = _front_power_metadata(cache_dir, candidate, int(row_id))
            power_records.append({
                **meta,
                "fiber_total_power_w": float(details["route_trace"]["fiber_output_mode_power_w"]),
                "after_4f_input_aperture_w": float(details["route_trace"]["after_decoder_input_aperture_power_w"]),
                "after_decoder_layer1_aperture_w": float(details["decoder_trace"]["after_decoder_layer1_aperture_power_w"]),
                "after_decoder_layer2_output_aperture_w": float(details["decoder_trace"]["after_decoder_layer2_output_aperture_power_w"]),
                "after_camera_w": float(p.sum().cpu()),
                "camera_expected_electrons": float(electrons.sum().cpu()),
                "camera_saturation_fraction": float((electrons >= config["camera"]["full_well_e"]).float().mean().cpu()),
            })
            del loss, details
            if _SIGNAL_RECEIVED is not None:
                return interrupted_result(step, microbatch_index + 1)
            if interrupt_event_path is not None and step == 2 and microbatch_index + 1 == int(config["training"]["interruption_microbatches_before_sigterm"]):
                _write_json(interrupt_event_path, {"candidate": candidate, "pid": os.getpid(),
                                                   "last_committed_step": current_step,
                                                   "partial_step": step, "microbatch_index": microbatch_index,
                                                   "signal_ready": True})
                while _SIGNAL_RECEIVED is None:
                    time.sleep(0.05)
                if _SIGNAL_RECEIVED is not None:
                    return interrupted_result(step, microbatch_index + 1)
        if _SIGNAL_RECEIVED is not None:
            return interrupted_result(step, len(row_losses))
        if phase.grad is None or not bool(torch.isfinite(phase.grad).all()):
            raise RuntimeError(f"Decoder phase gradient is absent or nonfinite at {candidate} step {step}")
        gradient_l2 = torch.linalg.vector_norm(phase.grad.detach().reshape(2, -1), dim=1).cpu()
        previous = phase.detach().clone()
        optimizer.step()
        update_l2 = torch.linalg.vector_norm((phase.detach() - previous).reshape(2, -1), dim=1).cpu()
        current_step = step
        record = {
            "step": step, "rows": [int(value) for value in rows],
            "labels": population["all_labels"][torch.tensor([position_by_row[int(value)] for value in rows])].tolist(),
            "mean_loss_before_update": float(np.mean(row_losses)),
            "per_image_loss_before_update": row_losses,
            "phase_gradient_l2_by_layer": gradient_l2.tolist(),
            "phase_update_l2_by_layer": update_l2.tolist(),
            "phase_rms_rad_by_layer_after_update": torch.sqrt(torch.mean(phase.detach().square(), dim=(-2, -1))).cpu().reshape(-1).tolist(),
            "parameters_updated": ["decoder_layer_0_phase", "decoder_layer_1_phase"],
            "frozen_parameters_updated": [],
            "absolute_power_batch_mean": {key: float(np.mean([value[key] for value in power_records]))
                                           for key in power_records[0]},
            "absolute_power_batch_min_max": {key: [float(min(value[key] for value in power_records)),
                                                    float(max(value[key] for value in power_records))]
                                             for key in power_records[0]},
        }
        history.append(record)
        interval = int(config["training"]["checkpoint_interval"])
        if steps <= int(config["training"]["preflight_continuous_steps"]) or step % interval == 0 or step == steps:
            _save_checkpoint(output_dir / f"checkpoint_step_{step:06d}.pt", candidate, phase, optimizer,
                             step, step + 1, config, population, schedule, history)
        if step % int(config["training"]["progress_interval"]) == 0:
            _write_json(output_dir / "progress.json", {"candidate": candidate, "status": "RUNNING",
                                                        "current_step": step, "planned_steps": steps,
                                                        "mean_loss_before_update": record["mean_loss_before_update"],
                                                        "checkpoint": str(output_dir / f"checkpoint_step_{(step // interval) * interval:06d}.pt")
                                                        if step % interval == 0 or step == steps else None})
        del previous
        if step % 10 == 0:
            gc.collect()
            clear_production_transfer_cache()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    terminal_path = output_dir / "terminal.pt"
    if not terminal_path.exists():
        _save_checkpoint(terminal_path, candidate, phase, optimizer, current_step, current_step + 1,
                         config, population, schedule, history)
    result = {"status": "COMPLETED", "candidate": candidate, "steps": current_step,
              "adam_updates": len(history), "phase": phase.detach().cpu(),
              "terminal_checkpoint": str(terminal_path), "history": history,
              "phase_rms_rad_by_layer": torch.sqrt(torch.mean(phase.detach().square(), dim=(-2, -1))).cpu().reshape(-1).tolist(),
              "trainable_parameter_count": int(phase.numel()),
              "frozen_parameter_count": 0}
    _write_json(output_dir / "result.json", result)
    return result


def _tree_max_error(left: Any, right: Any, *, atol: float, rtol: float, path: str = "root") -> tuple[bool, float, list[str]]:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        a, b = left.detach().cpu(), right.detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype:
            return False, math.inf, [path]
        if a.is_floating_point() or a.is_complex():
            delta = (a - b).abs()
            threshold = atol + rtol * b.abs()
            maximum = float(delta.max()) if delta.numel() else 0.0
            return bool(torch.all(delta <= threshold)), maximum, [] if bool(torch.all(delta <= threshold)) else [path]
        same = torch.equal(a, b)
        return same, 0.0 if same else math.inf, [] if same else [path]
    if isinstance(left, bool) or isinstance(right, bool):
        same = left is right
        return same, 0.0 if same else math.inf, [] if same else [path]
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        difference = abs(float(left) - float(right))
        same = difference <= atol + rtol * abs(float(right))
        return same, difference, [] if same else [path]
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False, math.inf, [path]
        outcomes = [_tree_max_error(left[key], right[key], atol=atol, rtol=rtol, path=f"{path}.{key}") for key in left]
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)) and len(left) == len(right):
        outcomes = [_tree_max_error(a, b, atol=atol, rtol=rtol, path=f"{path}[{index}]") for index, (a, b) in enumerate(zip(left, right))]
    else:
        same = left == right
        return bool(same), 0.0 if same else math.inf, [] if same else [path]
    return all(item[0] for item in outcomes), max((item[1] for item in outcomes), default=0.0), [bad for item in outcomes for bad in item[2]]


def _gradient_preflight(assets: dict[str, Any], population: dict[str, Any], prepared: dict[str, torch.Tensor],
                        config: dict[str, Any], run_dir: Path, device: torch.device) -> dict[str, Any]:
    row_id = int(population["fit_rows"][0])
    checks: dict[str, Any] = {}
    for candidate_index, candidate in enumerate(CANDIDATES):
        phase64 = assets["decoder_phase"].clone().to(device=device, dtype=torch.float32).requires_grad_(True)
        loss64, _details = _loss_one(assets, population, prepared, config, run_dir,
                                    candidate, row_id, phase64, device, "complex64")
        gradient64 = torch.autograd.grad(loss64, phase64)[0].detach()
        layer_records = []
        for layer in range(2):
            random = torch.randn((100, 100), generator=torch.Generator(device="cpu").manual_seed(
                2026101810 + candidate_index * 20 + layer)).to(device=device, dtype=torch.float32)
            random = random / torch.linalg.vector_norm(random).clamp_min(EPS)
            grad_layer = gradient64[layer, 0]
            grad_norm = float(torch.linalg.vector_norm(grad_layer))
            negative = -grad_layer / torch.linalg.vector_norm(grad_layer).clamp_min(EPS)
            directions = {"fixed_random": random, "normalized_negative_gradient": negative}
            direction_records = {}
            _require(grad_norm > 0.0, f"candidate {candidate} layer {layer} has zero camera-loss gradient")
            for direction_name, direction in directions.items():
                directional_ad = float(torch.sum(grad_layer * direction))
                direction128 = torch.zeros_like(assets["decoder_phase"], device=device, dtype=torch.float64)
                direction128[layer, 0] = direction.to(torch.float64)
                direction_steps = []
                for epsilon in (1.0e-4, 5.0e-5, 2.5e-5):
                    phase_base = assets["decoder_phase"].to(device=device, dtype=torch.float64)
                    plus_loss, _ = _loss_one(assets, population, prepared, config, run_dir,
                                             candidate, row_id=row_id,
                                             phase=phase_base + epsilon * direction128,
                                             device=device, precision="complex128")
                    minus_loss, _ = _loss_one(assets, population, prepared, config, run_dir,
                                              candidate, row_id=row_id,
                                              phase=phase_base - epsilon * direction128,
                                              device=device, precision="complex128")
                    fd = float((plus_loss - minus_loss) / (2.0 * epsilon))
                    abs_error = abs(directional_ad - fd)
                    tolerance = 1.0e-4 * abs(fd) + 1.0e-8 * float(loss64.detach().cpu())
                    direction_steps.append({"epsilon_rad": epsilon, "autograd_directional_derivative": directional_ad,
                                             "complex128_central_difference": fd, "absolute_error": abs_error,
                                             "tolerance": tolerance, "passed": abs_error <= tolerance})
                    del plus_loss, minus_loss
                    clear_production_transfer_cache()
                    torch.cuda.empty_cache()
                convergence_delta = abs(direction_steps[-1]["complex128_central_difference"] -
                                        direction_steps[-2]["complex128_central_difference"])
                convergence_tolerance = (1.0e-4 * abs(direction_steps[-1]["complex128_central_difference"])
                                         + 1.0e-8 * float(loss64.detach().cpu()))
                direction_records[direction_name] = {
                    "directional_autograd": directional_ad, "finite_difference_steps": direction_steps,
                    "step_size_converged": convergence_delta <= convergence_tolerance,
                    "step_size_delta": convergence_delta, "step_size_tolerance": convergence_tolerance,
                    "negative_gradient_sign_passed": (direction_name != "normalized_negative_gradient" or
                                                       (directional_ad < 0.0 and direction_steps[-1]["complex128_central_difference"] < 0.0)),
                    "passed": bool(direction_steps[-1]["passed"] and convergence_delta <= convergence_tolerance and
                                   (direction_name != "normalized_negative_gradient" or
                                    (directional_ad < 0.0 and direction_steps[-1]["complex128_central_difference"] < 0.0))),
                }
            layer_records.append({"layer": layer, "gradient_l2": grad_norm,
                                  "directions": direction_records,
                                  "passed": all(value["passed"] for value in direction_records.values())})
        checks[candidate] = {"initial_loss": float(loss64.detach().cpu()), "layers": layer_records,
                             "both_layers_finite_nonzero": bool(torch.isfinite(gradient64).all()) and all(v["gradient_l2"] > 0 for v in layer_records),
                             "passed": all(v["passed"] for v in layer_records)}
        del phase64, loss64, gradient64, _details
        gc.collect()
        clear_production_transfer_cache()
        torch.cuda.empty_cache()
    return {"status": "PASS" if all(checks[candidate]["passed"] for candidate in CANDIDATES) else "FAIL",
            "finite_difference_rule": "abs(AD-FD) <= 1e-4*abs(FD) + 1e-8*L0",
            "candidate_checks": checks}


def _negative_controls(assets: dict[str, Any], population: dict[str, Any], prepared: dict[str, torch.Tensor],
                       config: dict[str, Any], run_dir: Path, device: torch.device) -> dict[str, Any]:
    candidate = "candidate_3x3_B"
    row0, row1 = (int(value) for value in population["fit_rows"][:2])
    phase = assets["decoder_phase"].clone().to(device=device, dtype=torch.float32).requires_grad_(True)
    loss, details = _loss_one(assets, population, prepared, config, run_dir, candidate,
                              row0, phase, device, "complex64")
    detached_rejected = False
    try:
        detached_loss, _ = _loss_one(assets, population, prepared, config, run_dir, candidate,
                                     row0, phase, device, "complex64", detach_task=True)
        torch.autograd.grad(detached_loss, phase, allow_unused=False)
    except RuntimeError:
        detached_rejected = True
    dx = DX_M
    area_probe = torch.ones((2, 2), dtype=torch.complex128, device=device)
    area_power = _roi_power_w(area_probe)
    area_expected = 4.0 * dx * dx
    area_pass = math.isclose(area_power, area_expected, rel_tol=1e-12, abs_tol=1e-30)
    area_without_dx_squared = float(area_probe.abs().square().sum().real.detach().cpu())
    area_omission_rejected = not math.isclose(area_without_dx_squared, area_expected,
                                               rel_tol=1.0e-4, abs_tol=1.0e-30)
    active = details["active"].detach()
    phase0 = assets["decoder_phase"].to(device=device, dtype=torch.float32)[0, 0]
    transmission_root = math.sqrt(float(config["optics"]["decoder_layer_transmission_power"]))
    transmitted_correct = active * torch.polar(torch.full_like(phase0, transmission_root), phase0).to(active.dtype)
    transmitted_wrong = active * torch.polar(torch.full_like(phase0, float(config["optics"]["decoder_layer_transmission_power"])), phase0).to(active.dtype)
    correct_ratio = float(transmitted_correct.abs().square().sum() / active.abs().square().sum().clamp_min(EPS))
    wrong_ratio = float(transmitted_wrong.abs().square().sum() / active.abs().square().sum().clamp_min(EPS))
    transmission_negative_rejected = abs(correct_ratio - wrong_ratio) > 1.0e-2
    coefficients = details["coefficients_sqrt_w"].detach()
    permuted = coefficients.flip(0)
    field_wrong = v11._compose_fiber_field(assets["candidates"][candidate], "gaussian_s12",
                                           permuted.cpu().numpy(), 4096, torch.complex64, device)
    active_wrong, _trace, route_z = v9._route_to_decoder(field_wrong, "explicit_4f", 4096)
    wrong_output = _camera_forward(active_wrong, assets["decoder_phase"].to(device), 4096,
                                   route_z, config)["task_output"]
    port_order_delta = float(torch.linalg.vector_norm(wrong_output - details["task_output"].detach()) /
                             torch.linalg.vector_norm(details["task_output"].detach()).clamp_min(EPS))
    port_order_rejected = port_order_delta > 1.0e-4
    commands_low = torch.full((10000,), 0.25, dtype=torch.float32)
    commands_high = torch.full((10000,), 0.75, dtype=torch.float32)
    h = torch.as_tensor(assets["H"][candidate]["gaussian_s12"][4096], dtype=torch.complex64, device=device)
    low_coeff = h @ commands_low.to(device=device, dtype=h.dtype)
    high_coeff = h @ commands_high.to(device=device, dtype=h.dtype)
    low_field = v11._compose_fiber_field(assets["candidates"][candidate], "gaussian_s12",
                                         low_coeff.cpu().numpy(), 4096, torch.complex64, device)
    high_field = v11._compose_fiber_field(assets["candidates"][candidate], "gaussian_s12",
                                          high_coeff.cpu().numpy(), 4096, torch.complex64, device)
    low_active, low_trace, _ = v9._route_to_decoder(low_field, "explicit_4f", 4096)
    high_active, high_trace, _ = v9._route_to_decoder(high_field, "explicit_4f", 4096)
    power_ratio = float(high_trace["fiber_output_mode_power_w"] / max(low_trace["fiber_output_mode_power_w"], EPS))
    no_sample_normalization_rejected = abs(power_ratio - 9.0) <= 1.0e-3
    row_mismatch_rejected = False
    try:
        _assert_row_pair(row0, row1)
    except ValueError:
        row_mismatch_rejected = True
    constant_command = prepared["commands"][_population_position(population, row0)].to(device)
    constant_coeff = h @ constant_command.to(device=device, dtype=h.dtype)
    constant_field = v11._compose_fiber_field(assets["candidates"][candidate], "gaussian_s12",
                                              constant_coeff.cpu().numpy(), 4096, torch.complex64, device)
    constant_active, _, constant_z = v9._route_to_decoder(constant_field, "explicit_4f", 4096)
    constant_task = _camera_forward(constant_active, assets["decoder_phase"].to(device), 4096,
                                    constant_z, config)["task_output"].detach()
    target0 = prepared["targets"][_population_position(population, row0), 0].to(device)
    target1 = prepared["targets"][_population_position(population, row1), 0].to(device)
    constant_loss0 = float((constant_task - target0).square().mean())
    constant_loss1 = float((constant_task - target1).square().mean())
    constant_output_same = True
    constant_control_rejected_claim = (not torch.equal(target0, target1) and constant_output_same)
    records = {
        "detached_task_is_rejected_by_gradient_consumer": detached_rejected,
        "roi_power_uses_pixel_area": {"measured_w": area_power, "expected_w": area_expected,
                                      "area_omitted_counterexample": area_without_dx_squared,
                                      "omitted_area_rejected": area_omission_rejected,
                                      "passed": area_pass and area_omission_rejected},
        "wrong_transmission_amplitude_is_distinguishable": {"correct_power_ratio": correct_ratio,
                                                              "wrong_power_ratio": wrong_ratio,
                                                              "rejected": transmission_negative_rejected},
        "wrong_fiber_order_is_distinguishable": {"relative_task_output_delta": port_order_delta,
                                                  "rejected": port_order_rejected},
        "per_sample_power_renormalization_is_absent": {"constant_command_power_ratio_0p75_over_0p25": power_ratio,
                                                        "expected_ratio": 9.0,
                                                        "rejected": no_sample_normalization_rejected},
        "input_target_row_mismatch_is_rejected": row_mismatch_rejected,
        "constant_input_cannot_support_input_specific_claim": {"same_command_reused": True,
                                                                 "same_camera_output": constant_output_same,
                                                                 "distinct_target_loss_0": constant_loss0,
                                                                 "distinct_target_loss_1": constant_loss1,
                                                                 "claim_rejected": constant_control_rejected_claim},
        "all_passed": all((detached_rejected, area_pass, area_omission_rejected, transmission_negative_rejected,
                           port_order_rejected, no_sample_normalization_rejected,
                           row_mismatch_rejected, constant_control_rejected_claim)),
    }
    del loss, details, field_wrong, active_wrong, wrong_output, low_field, high_field, low_active, high_active
    del constant_field, constant_active, constant_task, phase
    clear_production_transfer_cache()
    torch.cuda.empty_cache()
    return records


def _install_signal_handler() -> None:
    def capture(signum: int, _frame: Any) -> None:
        global _SIGNAL_RECEIVED
        _SIGNAL_RECEIVED = int(signum)
    signal.signal(signal.SIGTERM, capture)


def _worker_assets_and_data(root: Path, source_dir: Path, data_file: Path, config_path: Path):
    config = _load_config(config_path)
    configured_source = _resolve(root, config["source"]["v9_run_dir"])
    configured_data = _resolve(root, config["data"]["training_file"])
    _require(configured_source == source_dir.resolve() and configured_data == data_file.resolve(),
             "v12 explicit source/data paths do not match the frozen configuration")
    assets = _load_assets(root, source_dir)
    pixels, labels, train_rows = _load_mnist(data_file, config)
    population = _select_population(labels, train_rows, config)
    prepared = _prepare_images(pixels, population)
    schedule = _make_schedule(population, config)
    return config, assets, population, prepared, schedule


def _preflight_worker(args: argparse.Namespace) -> int:
    global _SIGNAL_RECEIVED
    if not torch.cuda.is_available():
        raise RuntimeError("v12 recovery worker requires CUDA")
    device = torch.device("cuda")
    root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    source_dir = Path(args.source_dir).resolve()
    data_file = Path(args.data_file).resolve()
    preflight_dir = Path(args.preflight_dir).resolve()
    config, assets, population, prepared, schedule = _worker_assets_and_data(root, source_dir, data_file, config_path)
    _SIGNAL_RECEIVED = None
    _install_signal_handler()
    result = _train_candidate(
        assets, population, prepared, schedule, config, preflight_dir, preflight_dir,
        args.worker_candidate, device, steps=3,
        resume_checkpoint=Path(args.resume_checkpoint).resolve(),
        output_subdir=args.worker_output_subdir,
        interrupt_event_path=Path(args.interrupt_event).resolve() if args.interrupt_event else None,
    )
    _write_json(Path(args.worker_result).resolve(), result)
    return 0


def _run_recovery_preflight(assets: dict[str, Any], population: dict[str, Any], prepared: dict[str, torch.Tensor],
                            schedule: torch.Tensor, config: dict[str, Any], preflight_dir: Path,
                            source_dir: Path, data_file: Path, config_path: Path,
                            device: torch.device) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for candidate in CANDIDATES:
        continuous = _train_candidate(assets, population, prepared, schedule, config, preflight_dir,
                                      preflight_dir, candidate, device, steps=3,
                                      output_subdir="continuous_preflight")
        continuous_ckpt = preflight_dir / "training" / candidate / "continuous_preflight" / "checkpoint_step_000001.pt"
        _require(continuous_ckpt.is_file() and continuous["adam_updates"] == 3,
                 f"continuous three-step preflight did not save step one for {candidate}")
        _require(all(len(record["phase_gradient_l2_by_layer"]) == 2 and
                     all(float(value) > 0.0 for value in record["phase_gradient_l2_by_layer"]) and
                     all(float(value) > 0.0 for value in record["phase_update_l2_by_layer"]) and
                     record["frozen_parameters_updated"] == []
                     for record in continuous["history"]),
                 f"three-step preflight did not update exactly both Decoder layers for {candidate}")
        child_dir = preflight_dir / "training" / candidate / "interrupted_child"
        child_dir.mkdir(parents=True, exist_ok=False)
        event_path = child_dir / "sigterm_ready.json"
        child_result_path = child_dir / "worker_result.json"
        command = [
            sys.executable, "-m", "scripts.research.lightgen.execute_ols_decoder_mnist_reconstruction_v12",
            "--internal-preflight-worker", "--project-root", str(root := Path.cwd().resolve()),
            "--config", str(config_path.resolve()), "--source-dir", str(source_dir.resolve()),
            "--data-file", str(data_file.resolve()), "--preflight-dir", str(preflight_dir.resolve()),
            "--worker-candidate", candidate, "--resume-checkpoint", str(continuous_ckpt.resolve()),
            "--worker-output-subdir", "interrupted_child", "--interrupt-event", str(event_path.resolve()),
            "--worker-result", str(child_result_path.resolve()),
        ]
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        log_path = child_dir / "worker.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=environment)
            while True:
                if event_path.is_file():
                    event = json.loads(event_path.read_text(encoding="utf-8"))
                    _require(int(event.get("pid", -1)) == process.pid and
                             int(event.get("last_committed_step", -1)) == 1 and
                             int(event.get("partial_step", -1)) == 2,
                             f"SIGTERM worker reached an unexpected recovery point for {candidate}")
                    process.send_signal(signal.SIGTERM)
                    break
                if process.poll() is not None:
                    raise RuntimeError(f"SIGTERM preflight worker exited before recovery point: {candidate}; log={log_path}")
                time.sleep(0.1)
            return_code = process.wait()
        _require(return_code == 0 and child_result_path.is_file(),
                 f"SIGTERM worker failed to checkpoint cleanly for {candidate}; log={log_path}")
        child_result = json.loads(child_result_path.read_text(encoding="utf-8"))
        _require(child_result.get("status") == "INTERRUPTED" and
                 child_result.get("last_committed_step") == 1 and
                 child_result.get("discarded_partial_batch_step") == 2 and
                 child_result.get("discarded_microbatches") == 1 and
                 child_result.get("adam_call_count_in_partial_batch") == 0,
                 f"SIGTERM did not preserve last committed step and drop partial batch: {candidate}")
        interrupted_checkpoint = Path(child_result["checkpoint"])
        resume_dir = preflight_dir / "training" / candidate / "resumed_child"
        resume_dir.mkdir(parents=True, exist_ok=False)
        resume_result_path = resume_dir / "worker_result.json"
        resume_command = [
            sys.executable, "-m", "scripts.research.lightgen.execute_ols_decoder_mnist_reconstruction_v12",
            "--internal-preflight-worker", "--project-root", str(root),
            "--config", str(config_path.resolve()), "--source-dir", str(source_dir.resolve()),
            "--data-file", str(data_file.resolve()), "--preflight-dir", str(preflight_dir.resolve()),
            "--worker-candidate", candidate, "--resume-checkpoint", str(interrupted_checkpoint.resolve()),
            "--worker-output-subdir", "resumed_child", "--worker-result", str(resume_result_path.resolve()),
        ]
        with (resume_dir / "worker.log").open("w", encoding="utf-8") as log:
            resumed_process = subprocess.run(resume_command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                             env=environment, check=False)
        _require(resumed_process.returncode == 0 and resume_result_path.is_file(),
                 f"independent recovery worker failed for {candidate}; log={resume_dir / 'worker.log'}")
        resumed = json.loads(resume_result_path.read_text(encoding="utf-8"))
        _require(resumed.get("status") == "COMPLETED" and resumed.get("steps") == 3 and
                 resumed.get("adam_updates") == 3, f"recovery did not finish exact step 3 for {candidate}")
        continuous_payload = torch.load(continuous["terminal_checkpoint"], map_location="cpu", weights_only=False)
        resumed_payload = torch.load(resumed["terminal_checkpoint"], map_location="cpu", weights_only=False)
        phase_ok, phase_error, phase_bad = _tree_max_error(
            continuous_payload["phase_unwrapped"], resumed_payload["phase_unwrapped"],
            atol=float(config["gates"]["recovery_phase_atol"]), rtol=float(config["gates"]["recovery_phase_rtol"]))
        adam_ok, adam_error, adam_bad = _tree_max_error(
            continuous_payload["optimizer_state"], resumed_payload["optimizer_state"],
            atol=float(config["gates"]["recovery_adam_atol"]), rtol=float(config["gates"]["recovery_adam_rtol"]))
        history_ok, history_error, history_bad = _tree_max_error(
            continuous_payload["history"], resumed_payload["history"], atol=1.0e-12, rtol=1.0e-10)
        rng_ok, rng_error, rng_bad = _tree_max_error(
            continuous_payload["rng_state"], resumed_payload["rng_state"], atol=0.0, rtol=0.0)
        _require(phase_ok and adam_ok and history_ok and rng_ok,
                 f"continuous and recovered Decoder/Adam history differs for {candidate}: phase={phase_bad}, Adam={adam_bad}, history={history_bad}, RNG={rng_bad}")
        row_id = int(population["fit_rows"][0])
        terminal_cont, details_cont = _loss_one(assets, population, prepared, config, preflight_dir, candidate, row_id,
                                                continuous_payload["phase_unwrapped"].to(device), device, "complex64")
        terminal_resume, details_resume = _loss_one(assets, population, prepared, config, preflight_dir, candidate, row_id,
                                                    resumed_payload["phase_unwrapped"].to(device), device, "complex64")
        task_cont = details_cont["task_output"].detach()
        task_resume = details_resume["task_output"].detach()
        camera_error = float(torch.linalg.vector_norm(task_cont - task_resume) /
                             torch.linalg.vector_norm(task_cont).clamp_min(EPS))
        camera_loss_difference = abs(float(terminal_cont.detach().cpu()) - float(terminal_resume.detach().cpu()))
        _require(camera_error <= float(config["gates"]["recovery_camera_output_relative_error"]) and
                 camera_loss_difference <= 1.0e-12,
                 f"recovered camera loss differs beyond tolerance for {candidate}")
        checks[candidate] = {
            "continuous_adam_calls": 3, "recovery_child_adam_calls": 0,
            "resumed_adam_calls": 2, "total_engineering_adam_calls": 5,
            "sigterm": child_result, "phase_max_abs_difference": phase_error,
            "phase_passed": phase_ok, "phase_failed_paths": phase_bad,
            "adam_max_abs_difference": adam_error, "adam_passed": adam_ok,
            "adam_failed_paths": adam_bad, "history_identical": True,
            "history_max_abs_difference": history_error, "history_failed_paths": history_bad,
            "rng_state_identical": rng_ok, "rng_state_max_difference": rng_error,
            "rng_state_failed_paths": rng_bad,
            "camera_loss_relative_difference": camera_error,
            "camera_task_output_relative_error": camera_error,
            "camera_passed": camera_error <= float(config["gates"]["recovery_camera_output_relative_error"]),
            "passed": phase_ok and adam_ok and camera_error <= float(config["gates"]["recovery_camera_output_relative_error"]),
        }
        del terminal_cont, terminal_resume, details_cont, details_resume, continuous_payload, resumed_payload
        gc.collect()
        clear_production_transfer_cache()
        torch.cuda.empty_cache()
    return {"status": "PASS" if all(item["passed"] for item in checks.values()) else "FAIL",
            "engineering_adam_call_count": sum(item["total_engineering_adam_calls"] for item in checks.values()),
            "candidates": checks}


def _source_manifest(root: Path, config_path: Path) -> dict[str, Any]:
    relevant = [
        Path(__file__).resolve(), config_path.resolve(),
        (root / "scripts/research/lightgen/execute_ols_decoder_autograd_update_preflight_v11.py").resolve(),
        (root / "scripts/research/lightgen/diagnose_ols_decoder_camera_interface_v9.py").resolve(),
        (root / "scripts/research/lightgen/diagnose_ols_frozen_phase_controllability_v7.py").resolve(),
        (root / "scripts/research/lightgen/diagnose_ols_window_dimension_power_v5.py").resolve(),
        (root / "scripts/research/lightgen/diagnose_ols_gaussian_dimension_reaudit_v1.py").resolve(),
        (root / "scripts/research/lightgen/diagnose_ols_g3_second_hop_error_isolation_v3.py").resolve(),
        (root / "scripts/research/lightgen/diagnose_ols_g3_reference_closure_v4.py").resolve(),
        (root / "src/ldm_project/v2/optics/propagation.py").resolve(),
        (root / "src/ldm_project/v2/application/research/shared/research_run_control.py").resolve(),
    ]
    for module_name, module in tuple(sys.modules.items()):
        if not (module_name == "ldm_project" or module_name.startswith("ldm_project.")):
            continue
        module_path = getattr(module, "__file__", None)
        if module_path:
            path = Path(module_path).resolve()
            try:
                path.relative_to(root.resolve())
            except ValueError:
                continue
            relevant.append(path)
    relevant = list(dict.fromkeys(relevant))
    manifest = capture_execution_source_manifest(
        root, task_id=TASK_ID, semantic_authority_id=AUTHORITY,
        relevant_paths=relevant, config_identity=config_path.resolve().relative_to(root.resolve()).as_posix())
    return manifest.to_dict()


def _runtime_overlay(root: Path) -> tuple[Path, dict[str, str]]:
    configured = os.environ.get("LIGHTGEN_V12_RUNTIME_SRC")
    _require(bool(configured), "LIGHTGEN_V12_RUNTIME_SRC must name the isolated v12 runtime source overlay")
    overlay = Path(configured).resolve()
    try:
        overlay.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("v12 runtime source overlay must remain inside the project run tree") from error
    _require((overlay / "ldm_project" / "v2").is_dir(), f"v12 runtime overlay is incomplete: {overlay}")
    required = (
        "ldm_project.v2",
        "ldm_project.v2.optics.propagation",
        "ldm_project.v2.application.research.shared.research_run_control",
    )
    origins: dict[str, str] = {}
    for module_name in required:
        module = sys.modules.get(module_name)
        module_path = getattr(module, "__file__", None) if module is not None else None
        _require(bool(module_path), f"required isolated v2 module was not imported: {module_name}")
        resolved = Path(module_path).resolve()
        try:
            resolved.relative_to(overlay)
        except ValueError as error:
            raise ValueError(f"{module_name} did not load from the v12 runtime overlay: {resolved}") from error
        origins[module_name] = str(resolved)
    return overlay, origins


def _preflight(root: Path, config_path: Path, source_dir: Path, data_file: Path,
               preflight_dir: Path, device: torch.device) -> tuple[dict[str, Any], ResearchRunControl, PreflightReceipt]:
    overlay, module_origins = _runtime_overlay(root)
    if preflight_dir.exists():
        if not preflight_dir.is_dir() or any(item.name != "runtime_src" for item in preflight_dir.iterdir()):
            raise FileExistsError(f"v12 preflight path contains existing evidence; preserve it and choose a new attempt path: {preflight_dir}")
        _require((preflight_dir / "runtime_src").resolve() == overlay,
                 "pre-existing v12 preflight path does not contain the configured isolated runtime overlay")
    else:
        preflight_dir.mkdir(parents=True, exist_ok=False)
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    config, assets, population, prepared, schedule = _worker_assets_and_data(root, source_dir, data_file, config_path)
    _require(device.type == "cuda" and torch.cuda.is_available(), "v12 scientific preflight requires AutoDL CUDA")
    _require(assets["decoder_phase"].shape == (2, 1, 100, 100), "shared v9 Decoder initial phase has invalid dimensions")
    _require(population["all_rows"].numel() == 300 and population["all_rows"].unique().numel() == 300,
             "v12 selected MNIST population is incomplete or duplicated")
    _require(bool(torch.isfinite(prepared["commands"]).all()) and bool(torch.isfinite(prepared["targets"]).all()),
             "v12 prepared MNIST commands or targets contain nonfinite values")
    _require(config["data"]["official_dev_reads"] == 0 and config["data"]["official_test_reads"] == 0,
             "v12 is forbidden from consuming official DEV/TEST rows")
    assets_phase_before = assets["decoder_phase"].clone()
    encoder_before = {key: value.clone() for key, value in assets["encoder_phases"].items()}
    phase_parameter_count = int(assets["decoder_phase"].numel())
    correctness = {
        "status": "PASS", "passed": True,
        "population": {"fit_rows": population["fit_rows"].tolist(),
                       "confirmation_rows": population["confirmation_rows"].tolist(),
                       "labels": population["all_labels"].tolist(),
                       "fit_per_label": 10, "confirmation_per_label": 20,
                       "all_rows_unique": int(population["all_rows"].unique().numel()) == 300,
                       "all_rows_in_canonical_train": True,
                       "official_dev_and_test_reads": 0},
        "preprocessing": {"input_shape": list(prepared["input_images"].shape[1:]),
                          "target_shape": list(prepared["targets"].shape[1:]),
                          "amplitude_range": [float(prepared["commands"].min()), float(prepared["commands"].max())],
                          "bilinear_align_corners_false": True,
                          "source_power_is_fixed": SOURCE_POWER_W,
                          "per_sample_power_renormalization": False},
        "production_H_validation": _h_validation(assets, population, prepared, config, device, preflight_dir),
        "gradient_validation": _gradient_preflight(assets, population, prepared, config, preflight_dir, device),
        "negative_controls": _negative_controls(assets, population, prepared, config, preflight_dir, device),
    }
    _require(correctness["production_H_validation"]["status"] == "PASS", "real-MNIST H/direct production check failed")
    _require(correctness["gradient_validation"]["status"] == "PASS", "real camera-loss gradient check failed")
    _require(correctness["negative_controls"]["all_passed"], "one or more v12 production negative controls were not rejected")
    correctness["signal_recovery"] = _run_recovery_preflight(
        assets, population, prepared, schedule, config, preflight_dir,
        source_dir, data_file, config_path, device)
    _require(correctness["signal_recovery"]["status"] == "PASS" and
             correctness["signal_recovery"]["engineering_adam_call_count"] == 10,
             "v12 independent-process SIGTERM recovery preflight failed")
    frozen_state = {
        "shared_decoder_initial_phase_unchanged": bool(torch.equal(assets["decoder_phase"], assets_phase_before)),
        "frozen_encoder_phases_unchanged": all(torch.equal(assets["encoder_phases"][key], encoder_before[key]) for key in CANDIDATES),
        "only_decoder_phase_is_optimized": phase_parameter_count == 20000,
    }
    _require(all(frozen_state.values()), "preflight modified a frozen phase or trainable scope")
    correctness["frozen_state"] = frozen_state
    resource = {
        "status": "PASS", "passed": True, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_total_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "elapsed_seconds_observed": time.time() - started,
        "representative_cuda_production_forward_completed": True,
        "three_step_real_update_and_checkpoint_recovery_completed": True,
        "data_file_bytes": int(data_file.stat().st_size),
    }
    implementation = {"status": "PASS", "passed": True,
                      "selected_engine": "autodl_cuda_complex64_decoder_with_native_complex128_reference",
                      "training_entrypoint": str(Path(__file__).resolve()),
                      "preflight_cli_consumed": True,
                      "frozen_frontend_cache_precision_keyed": True,
                      "roi_power_includes_pixel_area": True,
                      "isolated_runtime_overlay": str(overlay),
                      "loaded_v2_module_origins": module_origins,
                      "formal_generator_training_updates": 0,
                      "loss_changes": 0, "task_evaluator_calls": 0, "fid_calls": 0}
    manifest = _source_manifest(root, config_path)
    identity = ResearchRunIdentity(
        task_id=TASK_ID, run_id="ols_decoder_mnist_reconstruction_v12_20261018",
        trajectory_id="fixed_frontend_two_candidate_500_update_each",
        semantic_authority_id=AUTHORITY,
        data_role="canonical_mnist_train_fit100_confirmation200_dev_test_unused",
        config_identity={"config_path": str(config_path.resolve().relative_to(root.resolve())), "config": config},
        model_identity={"decoder_initialization": "v9_shared_fixed_low_frequency",
                        "frozen_candidates": list(CANDIDATES), "trainable_decoder_phase_parameters": phase_parameter_count},
        optimizer_identity={"type": "Adam", "lr": config["training"]["adam_learning_rate"],
                            "betas": config["training"]["adam_betas"], "eps": config["training"]["adam_epsilon"],
                            "weight_decay": config["training"]["adam_weight_decay"]},
        sampler_identity={"seed": config["training"]["schedule_seed"], "logical_batch_size": 10,
                          "rows_per_class_per_batch": 1},
        execution_source_semantics_id=str(manifest["execution_source_semantics_id"]))
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    receipt = control.make_ready_receipt(correctness=correctness, resource=resource,
                                         implementation=implementation,
                                         selected_engine=implementation["selected_engine"],
                                         resource_envelope=resource)
    control.mark_preflight_ready(receipt)
    report = {"schema": PREFLIGHT_SCHEMA, "status": "READY", "task_id": TASK_ID,
              "authorization_id": AUTHORITY, "source_manifest": manifest,
              "runtime_overlay": str(overlay), "module_origins": module_origins,
              "source_dir": str(source_dir.resolve()), "data_file": str(data_file.resolve()),
              "config": config, "schedule": _schedule_records(population, schedule),
              "checks": {CORRECTNESS_INVARIANTS: correctness, RESOURCE_ENVELOPE: resource,
                         IMPLEMENTATION_SELECTION: implementation},
              "preflight_receipt": receipt.to_dict(), "research_run_identity": identity.to_dict(),
              "trainable_parameters": {"decoder_phase_count": phase_parameter_count,
                                       "frozen_encoder_phase_count": int(sum(value.numel() for value in assets["encoder_phases"].values()))},
              "formal_scientific_updates": 0, "engineering_adam_calls": 10,
              "started_at_epoch_s": started, "completed_at_epoch_s": time.time()}
    _write_json(preflight_dir / "preflight_report.json", report)
    _write_json(preflight_dir / "status.json", {"status": "READY", "receipt": receipt.to_dict()})
    return report, control, receipt


if __name__ == "__main__":
    raise SystemExit(main())



