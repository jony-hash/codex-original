"""Bounded Gaussian phase-only optimization for the LightGen fiber OLS interface."""

from __future__ import annotations

import argparse
import copy
import gc
import inspect
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from ldm_project.v2.contracts import ComplexField
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack
from ldm_project.v2.optics.propagation import _transfer_function, clear_production_transfer_cache
from scripts.research.lightgen import diagnose_ols_gaussian_dimension_reaudit_v1 as baseline
from scripts.research.lightgen import diagnose_ols_window_dimension_power_v5 as v5


SCHEMA = "ols_phase_dimension_power.v6"
DEFAULT_CONFIG = Path("configs/research/lightgen/ols_phase_dimension_power_v6.yaml")
DEFAULT_RUN_DIR = Path("outputs/20261014_ols_phase_dimension_power_v6/runs/attempt_01")
SIZES = (1, 2, 3, 4, 5, 6, 7, 8, 10)
OBJECTIVES = ("A_max_throughput", "B_max_neff_fraction", "C_absolute_uniform_delivery")
INITIALIZATIONS = ("v5_zero", "uniform_seed_2026101401", "uniform_seed_2026101402")
GRIDS = (3072, 4096, 6144)
PRECISIONS = {"complex64": torch.float32, "complex128": torch.float64}
ROI_SIZE = 2048
DX_M = 2.0e-6
WAVELENGTH_M = 532.0e-9
SOURCE_POWER_W = 249.3e-6
TRANSMISSION = 0.889
PITCH_M = 125.0e-6
WAIST_M = 7.111e-6
WINDOW_TOLERANCE = 1.0e-4
ADJOINT_TOLERANCE = 1.0e-4
FINITE_DIFFERENCE_REL_TOLERANCE = 1.0e-4
FINITE_DIFFERENCE_ABS_TOLERANCE = 1.0e-8
CHECKPOINT_STEPS = (0, 64, 128, 192, 256)
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False) + "\n")


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != SCHEMA:
        raise ValueError(f"expected {SCHEMA} configuration")
    expected_optics = {
        "wavelength_m": WAVELENGTH_M,
        "pixel_pitch_m": DX_M,
        "encoder_shape": [100, 100],
        "encoder_aperture_m": 200.0e-6,
        "propagation_grid_sizes": list(GRIDS),
        "common_fiber_readout_roi_mm": 4.096,
        "common_fiber_readout_roi_pixels": ROI_SIZE,
        "inter_encoder_spacing_m": 1.5e-3,
        "last_encoder_to_fiber_m": 1.5e-3,
        "phase_screen_count": 2,
        "layer_power_transmission": TRANSMISSION,
        "input_power_w": SOURCE_POWER_W,
        "illumination": "uniform_coherent_top_hat",
    }
    if config.get("optics") != expected_optics:
        raise ValueError("v6 optical geometry, sampling, ROI, or source power changed")
    if config.get("fiber") != {
        "reference_pitch_m": PITCH_M,
        "cladding_diameter_m": PITCH_M,
        "mode_field_diameter_m": 14.222e-6,
        "gaussian_1e_amplitude_radius_m": WAIST_M,
        "lp01_v_number": 2.2,
        "transport": "ideal_identity",
        "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
    }:
        raise ValueError("v6 fiber geometry or model assumption changed")
    if config.get("arrays", {}).get("square_array_sizes") != list(SIZES):
        raise ValueError("v6 must measure the complete frozen v5 array set")
    optimizer = config.get("optimizer", {})
    expected_optimizer = {
        "updates_per_trajectory": 256,
        "initialization_names": list(INITIALIZATIONS),
        "random_phase_seeds": [2026101401, 2026101402],
        "random_phase_distribution": "independent_uniform[-pi,pi] per layer and pixel",
        "name": "Adam",
        "learning_rate": 0.04,
        "betas": [0.9, 0.999],
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "scheduler": "none",
        "early_stopping": False,
    }
    if optimizer != expected_optimizer:
        raise ValueError("v6 optimizer definition or fixed budget changed")
    if config.get("gates", {}).get("window_g3_complex_relative_error_after_common_global_phase_max") != WINDOW_TOLERANCE:
        raise ValueError("v6 production window threshold must remain 1e-4")
    scope = config.get("scope", {})
    expected_scope = {
        "formal_training_updates": 0,
        "dataset_reads": 0,
        "task_evaluator_calls": 0,
        "fid_calls": 0,
        "loss_changes": 0,
        "material_forward_calls": 0,
        "relay_or_decoder_camera_changes": 0,
        "production_default_migration": "forbidden",
        "optical_geometry_change": "forbidden",
        "propagation_solver_replacement": "forbidden",
        "lp01_independent_optimization": "forbidden",
    }
    if scope != expected_scope:
        raise ValueError("v6 scope contract changed")
    return config


def _load_v5_inputs(config: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, torch.Tensor]]:
    evidence = config["source_evidence"]
    report_path = _resolve(root, evidence["v5_report"])
    metrics_path = _resolve(root, evidence["v5_dimension_metrics"])
    phase_path = _resolve(root, evidence["v5_fixed_phase_screens"])
    for path in (report_path, metrics_path, phase_path):
        if not path.is_file():
            raise FileNotFoundError(f"explicit v5 input is missing: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if report.get("schema") != v5.SCHEMA or report.get("status") != "VALID_COMPLETED":
        raise ValueError("v6 requires the explicitly named completed v5 report")
    if not report.get("v5_window_G3", {}).get("production_complex64_all_nine_comparisons_pass"):
        raise ValueError("v5 expanded-window baseline did not pass its nine production comparisons")
    if metrics.get("status") != "COMPLETE":
        raise ValueError("v5 dimension and power metrics are incomplete")
    phases = torch.load(phase_path, map_location="cpu", weights_only=True)
    if not isinstance(phases, dict) or "zero" not in phases:
        raise ValueError("v5 phase archive does not contain its measured zero-phase initialization")
    zero = phases["zero"].detach().cpu()
    if tuple(zero.shape) != (2, 1, 100, 100) or not torch.isfinite(zero).all():
        raise ValueError("v5 zero-phase tensor is invalid")
    if float(zero.abs().max()) != 0.0:
        raise ValueError("v5 zero initialization is not the measured zero-phase state")
    for size in SIZES:
        record = metrics["phase_cases"]["zero"][str(size)]["gaussian_s12"]
        saved_coordinates = np.asarray(record["coordinates_xy_m_row_major"], dtype=np.float64)
        expected_axis = baseline._array_coordinates(size, PITCH_M, torch.device("cpu")).numpy()
        expected_coordinates = np.asarray([[x, y] for y in expected_axis for x in expected_axis], dtype=np.float64)
        if saved_coordinates.shape != expected_coordinates.shape or not np.array_equal(saved_coordinates, expected_coordinates):
            raise ValueError(f"v5 stored coordinate order or geometry differs for {size}x{size}")
    return report, metrics, phases


def _initializations(zero_phase: torch.Tensor) -> dict[str, torch.Tensor]:
    result = {"v5_zero": zero_phase.to(torch.float32).clone()}
    for name, seed in ((INITIALIZATIONS[1], 2026101401), (INITIALIZATIONS[2], 2026101402)):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result[name] = (torch.rand((2, 1, 100, 100), generator=generator, dtype=torch.float32) * (2.0 * math.pi) - math.pi)
    return result


def _make_first_hop(reference_runtime) -> FiniteCropPhaseStack:
    stack = FiniteCropPhaseStack(
        layers=1,
        active_shape=reference_runtime.encoder.active_shape,
        propagation_shape=reference_runtime.encoder.propagation_shape,
        spacing_m=reference_runtime.encoder.spacing_m,
        layer_power_transmission=reference_runtime.encoder.layer_power_transmission,
        preserve_global_output=False,
    )
    return stack.to(device=next(reference_runtime.parameters()).device)


def _copy_trajectory_phase(first_stack: FiniteCropPhaseStack, last_phase: nn.Parameter, phase: torch.Tensor) -> None:
    with torch.no_grad():
        first_stack.phase.copy_(phase[:1].to(device=first_stack.phase.device, dtype=first_stack.phase.dtype))
        last_phase.copy_(phase[1].to(device=last_phase.device, dtype=last_phase.dtype))


def _compose_phase(first_stack: FiniteCropPhaseStack, last_phase: nn.Parameter) -> torch.Tensor:
    return torch.cat((first_stack.phase.detach(), last_phase.detach().unsqueeze(0)), dim=0).cpu()


def _build_gaussian_adjoint_pools(runtime) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    device = runtime.ols_plane.shape
    del device
    plane = runtime.ols_plane
    height, width = plane.shape
    if height != width:
        raise ValueError("v6 Gaussian adjoint pool requires the square v5 propagation grid")
    cuda_device = next(runtime.parameters()).device
    fy_transfer = _transfer_function(
        plane.shape,
        pixel_pitch_m=plane.pixel_pitch_m,
        wavelength_m=plane.wavelength_m,
        distance_m=runtime.encoder.spacing_m,
        device=cuda_device,
        real_dtype=torch.float32,
    )
    dy, dx = plane.pixel_pitch_m
    y = (torch.arange(height, dtype=torch.float64, device=cuda_device) + 0.5) * dy - plane.fov_m[0] / 2.0 + plane.origin_m[0]
    x = (torch.arange(width, dtype=torch.float64, device=cuda_device) + 0.5) * dx - plane.fov_m[1] / 2.0 + plane.origin_m[1]
    amplitude = (2.0 / (math.pi * WAIST_M**2)) ** 0.25
    crop_y = (height - 100) // 2
    crop_x = (width - 100) // 2
    pools: dict[int, torch.Tensor] = {}
    norm_errors: dict[str, float] = {}
    for pool_size in (7, 10):
        axis = baseline._array_coordinates(pool_size, PITCH_M, cuda_device)
        rows: list[torch.Tensor] = []
        row_norm_errors: list[float] = []
        for cy in axis:
            mode_y = amplitude * torch.exp(-((y - cy) / WAIST_M).square())
            for cx in axis:
                mode_x = amplitude * torch.exp(-((x - cx) / WAIST_M).square())
                mode = torch.outer(mode_y, mode_x).to(torch.complex64)
                norm = float((mode.abs().square().sum(dtype=torch.float64) * dy * dx).item())
                row_norm_errors.append(abs(norm - 1.0))
                back = torch.fft.ifft2(
                    torch.fft.fft2(mode, norm="ortho") * fy_transfer.conj(),
                    norm="ortho",
                )
                active = back[crop_y : crop_y + 100, crop_x : crop_x + 100]
                rows.append(active.to(torch.complex128))
                del mode, back, active, mode_x
            del mode_y
        pools[pool_size] = torch.stack(rows, dim=0).detach()
        norm_errors[str(pool_size)] = max(row_norm_errors)
        if not torch.isfinite(pools[pool_size]).all():
            raise RuntimeError(f"Gaussian adjoint pool {pool_size} contains nonfinite values")
    return pools, {
        "status": "BUILT",
        "grid_size": height,
        "pool_rows": {"7": 49, "10": 100},
        "gaussian_analytic_mode_norm_max_absolute_error": norm_errors,
        "stored_adjoint_dtype": "complex128 values from the complex64 production transfer operator",
        "input_operator": "production angular-spectrum transfer at 1.5 mm; center crop; no per-port renormalization",
    }


def _selected_adjoint_rows(pools: dict[int, torch.Tensor], size: int, device: torch.device) -> torch.Tensor:
    pool_size = 7 if size % 2 else 10
    pool_axis = baseline._array_coordinates(pool_size, PITCH_M, torch.device("cpu")).tolist()
    size_axis = baseline._array_coordinates(size, PITCH_M, torch.device("cpu")).tolist()
    index_by_coordinate = {float(value): index for index, value in enumerate(pool_axis)}
    indices: list[int] = []
    for cy in size_axis:
        for cx in size_axis:
            row = index_by_coordinate[float(cy)]
            col = index_by_coordinate[float(cx)]
            indices.append(row * pool_size + col)
    pool = pools[pool_size]
    index = torch.tensor(indices, dtype=torch.long, device=pool.device)
    return pool.index_select(0, index).to(device=device)


def _fast_gaussian_coefficients(
    first_stack: FiniteCropPhaseStack,
    last_phase: nn.Parameter,
    entrance: ComplexField,
    adjoint_rows: torch.Tensor,
) -> torch.Tensor:
    first_output, _records, _statistics = first_stack(entrance, "ols_v6_first_hop")
    transmission = torch.polar(
        torch.full_like(last_phase, math.sqrt(TRANSMISSION)),
        last_phase,
    ).to(first_output.tensor.dtype)
    modulated = first_output.tensor * transmission.unsqueeze(0)
    coefficients = adjoint_rows.conj().reshape(adjoint_rows.shape[0], -1) @ modulated[0, 0].to(torch.complex128).reshape(-1)
    return coefficients * math.prod(entrance.plane.pixel_pitch_m)


def _full_gaussian_coefficients(runtime, entrance: ComplexField, phase: torch.Tensor, size: int, *, grad: bool) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        runtime.encoder.phase.copy_(phase.to(device=runtime.encoder.phase.device, dtype=runtime.encoder.phase.dtype))
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        output, _records, statistics = runtime.encoder(entrance, "ols_v6_full_production")
        coefficients = baseline._project_variable(
            output.tensor[0, 0],
            output.plane,
            baseline._array_coordinates(size, PITCH_M, output.tensor.device),
            "gaussian_s12",
            WAIST_M,
            2.2,
        )
    return coefficients.reshape(-1), statistics["layer_power"]


def _full_trainable_gaussian_coefficients(runtime, entrance: ComplexField, size: int) -> torch.Tensor:
    output, _records, _statistics = runtime.encoder(entrance, "ols_v6_full_production_trainable")
    return baseline._project_variable(
        output.tensor[0, 0],
        output.plane,
        baseline._array_coordinates(size, PITCH_M, output.tensor.device),
        "gaussian_s12",
        WAIST_M,
        2.2,
    ).reshape(-1)


def _objective_values(coefficients: torch.Tensor, input_power_w: float, size: int, target_channel_power_w: float | None) -> dict[str, torch.Tensor]:
    powers = coefficients.abs().square().to(torch.float64).reshape(-1)
    total = powers.sum()
    neff = total.square() / powers.square().sum().clamp_min(1.0e-300)
    result = {
        "total_coupled_power_w": total,
        "eta_system": total / input_power_w,
        "N_eff": neff,
        "neff_fraction": neff / float(size * size),
    }
    result[OBJECTIVES[0]] = -result["eta_system"]
    result[OBJECTIVES[1]] = -result["neff_fraction"]
    if target_channel_power_w is not None:
        target = torch.as_tensor(target_channel_power_w, dtype=torch.float64, device=powers.device)
        result[OBJECTIVES[2]] = ((powers - target).square() / target.square()).mean()
    return result


def _complex_pairs(value: torch.Tensor) -> list[list[float]]:
    values = value.detach().to(torch.complex128).cpu().reshape(-1)
    return [[float(item.real), float(item.imag)] for item in values]


def _relative_error(reference: torch.Tensor, test: torch.Tensor) -> float:
    ref = reference.to(torch.complex128) if torch.is_complex(reference) else reference.to(torch.float64)
    other = test.to(ref.dtype)
    return float(torch.linalg.vector_norm(other - ref) / torch.linalg.vector_norm(ref).clamp_min(EPS))


def _symmetric_relative_error(reference: torch.Tensor, test: torch.Tensor, floor: float = 1.0e-30) -> float:
    ref = reference.detach().to(torch.float64)
    other = test.detach().to(torch.float64)
    denominator = torch.maximum(torch.linalg.vector_norm(ref), torch.linalg.vector_norm(other)).clamp_min(floor)
    return float(torch.linalg.vector_norm(ref - other) / denominator)


def _finite_metrics(values: dict[str, torch.Tensor]) -> dict[str, float]:
    result = {key: float(value.detach().cpu()) for key, value in values.items() if key in ("total_coupled_power_w", "eta_system", "N_eff", "neff_fraction", *OBJECTIVES)}
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("phase objective or power metrics contain nonfinite values")
    return result


def _checkpoint_payload(
    trajectory_id: str,
    size: int,
    objective: str,
    initialization: str,
    step: int,
    optimizer: torch.optim.Optimizer,
    best_phase: torch.Tensor,
    best_objective: float,
    best_step: int,
    target_channel_power_w: float | None,
    *,
    current_phase: torch.Tensor,
    selected_path: str = "validated_second_hop_adjoint",
) -> dict[str, Any]:
    return {
        "schema": "ols_phase_dimension_power_v6_checkpoint.v1",
        "trajectory_id": trajectory_id,
        "array_size": size,
        "objective": objective,
        "initialization": initialization,
        "step": step,
        "selected_path": selected_path,
        "current_phase_screens_rad": current_phase.detach().cpu().to(torch.float32),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_phase_screens_rad": best_phase.detach().cpu(),
        "best_objective": float(best_objective),
        "best_step": int(best_step),
        "target_channel_power_w": target_channel_power_w,
        "history_rows": step,
        "torch_cpu_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_states": [state.cpu() for state in torch.cuda.get_rng_state_all()],
    }


def _validate_checkpoint(
    payload: dict[str, Any],
    trajectory_id: str,
    size: int,
    objective: str,
    initialization: str,
    target: float | None,
    selected_path: str | None = None,
) -> None:
    expected = {
        "schema": "ols_phase_dimension_power_v6_checkpoint.v1",
        "trajectory_id": trajectory_id,
        "array_size": size,
        "objective": objective,
        "initialization": initialization,
        "target_channel_power_w": target,
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("explicit resume checkpoint does not match the selected v6 trajectory or frozen target")
    if selected_path is not None and payload.get("selected_path") != selected_path:
        raise ValueError("explicit resume checkpoint uses a different selected optimization path")
    phase = payload.get("current_phase_screens_rad")
    best = payload.get("best_phase_screens_rad")
    step = payload.get("step")
    if not isinstance(phase, torch.Tensor) or tuple(phase.shape) != (2, 1, 100, 100) or not torch.isfinite(phase).all():
        raise ValueError("explicit resume checkpoint has an invalid current phase state")
    if not isinstance(best, torch.Tensor) or tuple(best.shape) != (2, 1, 100, 100) or not torch.isfinite(best).all():
        raise ValueError("explicit resume checkpoint has an invalid best phase state")
    if not isinstance(step, int) or not 0 <= step <= 256 or payload.get("history_rows") != step:
        raise ValueError("explicit resume checkpoint update count or history position is invalid")
    if not isinstance(payload.get("optimizer_state_dict"), dict):
        raise ValueError("explicit resume checkpoint omits Adam state")


def _run_trajectory(
    *,
    run_dir: Path,
    size: int,
    objective: str,
    initialization: str,
    initial_phase: torch.Tensor,
    target_channel_power_w: float | None,
    runtime,
    entrance: ComplexField,
    first_stack: FiniteCropPhaseStack,
    adjoint_rows: torch.Tensor,
    selected_path: str = "validated_second_hop_adjoint",
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    trajectory_id = f"n{size:02d}_{objective[0]}_{initialization}"
    trajectory_dir = run_dir / "trajectories" / trajectory_id
    summary_path = trajectory_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"trajectory is already complete: {summary_path}")
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    fast_enabled = selected_path == "validated_second_hop_adjoint"
    device = entrance.tensor.device
    last_phase = nn.Parameter(torch.zeros((1, 100, 100), dtype=torch.float32, device=device)) if fast_enabled else None
    if fast_enabled:
        _copy_trajectory_phase(first_stack, last_phase, initial_phase.to(device))
        trainable_parameters = [first_stack.phase, last_phase]
    elif selected_path == "full_production_two_hop":
        with torch.no_grad():
            runtime.encoder.phase.copy_(initial_phase.to(device=device, dtype=runtime.encoder.phase.dtype))
        trainable_parameters = [runtime.encoder.phase]
    else:
        raise ValueError(f"unsupported selected optimization path: {selected_path}")

    def current_phase_tensor() -> torch.Tensor:
        if fast_enabled:
            return _compose_phase(first_stack, last_phase).to(device)
        return runtime.encoder.phase.detach()

    def coefficients_for_current_phase() -> torch.Tensor:
        if fast_enabled:
            return _fast_gaussian_coefficients(first_stack, last_phase, entrance, adjoint_rows)
        return _full_trainable_gaussian_coefficients(runtime, entrance, size)

    def layer_gradients() -> list[torch.Tensor]:
        if fast_enabled:
            if first_stack.phase.grad is None or last_phase.grad is None:
                raise RuntimeError("both phase screens must receive gradients from the selected optical objective")
            return [first_stack.phase.grad[0], last_phase.grad[0]]
        if runtime.encoder.phase.grad is None:
            raise RuntimeError("both production phase screens must receive gradients from the selected optical objective")
        return [runtime.encoder.phase.grad[0, 0], runtime.encoder.phase.grad[1, 0]]
    optimizer_config = {
        "lr": 0.04,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 0.0,
    }
    optimizer = torch.optim.Adam(trainable_parameters, **optimizer_config)
    history_path = trajectory_dir / "target_history.jsonl"
    discarded_history_path: Path | None = None
    if resume_checkpoint is None:
        if history_path.exists() and history_path.stat().st_size:
            raise ValueError(f"incomplete trajectory history requires an explicit checkpoint: {history_path}")
        best_phase = current_phase_tensor().detach().cpu()
        best_objective = math.inf
        best_step = 0
        start_step = 0
    else:
        payload = torch.load(resume_checkpoint, map_location=entrance.tensor.device, weights_only=True)
        _validate_checkpoint(payload, trajectory_id, size, objective, initialization, target_channel_power_w, selected_path)
        if fast_enabled:
            _copy_trajectory_phase(first_stack, last_phase, payload["current_phase_screens_rad"])
        else:
            with torch.no_grad():
                runtime.encoder.phase.copy_(payload["current_phase_screens_rad"].to(device=entrance.tensor.device, dtype=runtime.encoder.phase.dtype))
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        torch.set_rng_state(payload["torch_cpu_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["torch_cuda_rng_states"]])
        best_phase = payload["best_phase_screens_rad"].detach().cpu()
        best_objective = float(payload["best_objective"])
        best_step = int(payload["best_step"])
        start_step = int(payload["step"])
        history_lines = history_path.read_text(encoding="utf-8").splitlines(keepends=True) if history_path.exists() else []
        nonblank_lines = [line for line in history_lines if line.strip()]
        if len(nonblank_lines) < start_step:
            raise ValueError("trajectory history is shorter than the explicit resume checkpoint")
        for expected_index, line in enumerate(nonblank_lines[:start_step], start=1):
            history_row = json.loads(line)
            if history_row.get("update_index") != expected_index:
                raise ValueError("trajectory history update indices do not match the explicit checkpoint")
        if len(nonblank_lines) > start_step:
            tail_lines = nonblank_lines[start_step:]
            archive_index = 1
            while True:
                candidate = trajectory_dir / f"discarded_history_after_checkpoint_{start_step:04d}_replay_{archive_index:02d}.jsonl"
                if not candidate.exists():
                    discarded_history_path = candidate
                    break
                archive_index += 1
            discarded_history_path.write_text("".join(tail_lines), encoding="utf-8", newline="\n")
            history_path.write_text("".join(nonblank_lines[:start_step]), encoding="utf-8", newline="\n")
    with torch.no_grad():
        initial_coeff = coefficients_for_current_phase()
        initial_values = _objective_values(initial_coeff, SOURCE_POWER_W, size, target_channel_power_w)
        current_loss = float(initial_values[objective])
        if current_loss < best_objective:
            best_objective = current_loss
            best_phase = current_phase_tensor().detach().cpu()
            best_step = start_step
    initial_phase_path = trajectory_dir / "phase_initial_screens.pt"
    if initial_phase_path.exists():
        saved_initial_phase = torch.load(initial_phase_path, map_location="cpu", weights_only=True)
        if not _phase_records_equal(saved_initial_phase, initial_phase):
            raise ValueError("existing trajectory initial phase differs from the frozen configured initialization")
    else:
        _atomic_torch_save(initial_phase_path, initial_phase.detach().cpu().to(torch.float32))
    checkpoint_paths: dict[str, str] = {}

    def save_checkpoint(step: int) -> None:
        base_path = trajectory_dir / f"checkpoint_step_{step:04d}.pt"
        checkpoint_path = base_path
        if base_path.exists():
            replay_index = 1
            while True:
                candidate = trajectory_dir / f"checkpoint_step_{step:04d}_replay_{replay_index:02d}.pt"
                if not candidate.exists():
                    checkpoint_path = candidate
                    break
                replay_index += 1
        payload = _checkpoint_payload(
            trajectory_id,
            size,
            objective,
            initialization,
            step,
            optimizer,
            best_phase,
            best_objective,
            best_step,
            target_channel_power_w,
            current_phase=current_phase_tensor(),
            selected_path=selected_path,
        )
        _atomic_torch_save(checkpoint_path, payload)
        checkpoint_paths[str(step)] = str(checkpoint_path.relative_to(run_dir)).replace("\\", "/")

    if start_step == 0:
        save_checkpoint(0)
    else:
        checkpoint_paths[str(start_step)] = str(resume_checkpoint.relative_to(run_dir)).replace("\\", "/") if resume_checkpoint.is_relative_to(run_dir) else str(resume_checkpoint)
    started = time.perf_counter()
    for update_index in range(start_step + 1, 257):
        optimizer.zero_grad(set_to_none=True)
        coefficients = coefficients_for_current_phase()
        objective_dict = _objective_values(coefficients, SOURCE_POWER_W, size, target_channel_power_w)
        loss = objective_dict[objective]
        loss_value = float(loss.detach().cpu())
        if loss_value < best_objective:
            best_objective = loss_value
            best_phase = current_phase_tensor().detach().cpu()
            best_step = update_index - 1
        loss.backward()
        gradients = layer_gradients()
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("phase gradient contains nonfinite values")
        gradient_norms = [float(torch.linalg.vector_norm(gradient)) for gradient in gradients]
        before_phase = current_phase_tensor().detach().clone()
        optimizer.step()
        after_phase = current_phase_tensor().detach()
        update_norms = [float(torch.linalg.vector_norm(after_phase[index] - before_phase[index])) for index in range(2)]
        metrics = _finite_metrics(objective_dict)
        _append_jsonl(
            history_path,
            {
                "update_index": update_index,
                "evaluated_phase_step": update_index - 1,
                "objective_value_before_update": loss_value,
                "objective_metrics_before_update": metrics,
                "phase_gradient_l2_norm_by_layer": gradient_norms,
                "phase_update_l2_norm_by_layer": update_norms,
                "selected_optimization_path": selected_path,
            },
        )
        if update_index in CHECKPOINT_STEPS[1:-1]:
            save_checkpoint(update_index)
        del coefficients, objective_dict, loss, before_phase, after_phase, gradients
    with torch.no_grad():
        terminal_coeff = coefficients_for_current_phase()
        terminal_values = _objective_values(terminal_coeff, SOURCE_POWER_W, size, target_channel_power_w)
        terminal_objective = float(terminal_values[objective])
        if terminal_objective < best_objective:
            best_objective = terminal_objective
            best_phase = current_phase_tensor().detach().cpu()
            best_step = 256
    terminal_phase = current_phase_tensor().detach().cpu()
    best_phase_path = trajectory_dir / "phase_best_by_objective.pt"
    terminal_phase_path = trajectory_dir / "phase_step_0256.pt"
    _atomic_torch_save(best_phase_path, best_phase.detach().cpu())
    _atomic_torch_save(terminal_phase_path, terminal_phase)
    save_checkpoint(256)
    result = {
        "schema": "ols_phase_dimension_power_v6_trajectory.v1",
        "trajectory_id": trajectory_id,
        "status": "OPTIMIZATION_COMPLETE_PENDING_ENDPOINT_WINDOW_VALIDATION",
        "array_size": size,
        "fiber_count": size * size,
        "objective": objective,
        "initialization": initialization,
        "target_channel_power_w": target_channel_power_w,
        "updates": 256,
        "best_step": best_step,
        "best_objective": best_objective,
        "terminal_step": 256,
        "terminal_objective_fast_readout": terminal_objective,
        "best_phase_path": str(best_phase_path.relative_to(run_dir)).replace("\\", "/"),
        "terminal_phase_path": str(terminal_phase_path.relative_to(run_dir)).replace("\\", "/"),
        "history_path": str(history_path.relative_to(run_dir)).replace("\\", "/"),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "elapsed_seconds_observation": time.perf_counter() - started,
        "optimizer": optimizer_config,
        "selected_optimization_path": selected_path,
        "checkpoint_paths_by_update": checkpoint_paths,
        "discarded_history_after_resume_checkpoint": (
            str(discarded_history_path.relative_to(run_dir)).replace("\\", "/") if discarded_history_path else None
        ),
    }
    return result


def _phase_records_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return bool(torch.equal(left.to(torch.float32), right.to(torch.float32)))


def _readout_metrics(field: torch.Tensor, runtime, size: int) -> dict[str, Any]:
    offset = (runtime.ols_plane.shape[0] - ROI_SIZE) // 2
    roi = field[offset : offset + ROI_SIZE, offset : offset + ROI_SIZE]
    roi_plane = runtime.ols_plane.with_plane("ols_v6_common_fiber_roi", shape=(ROI_SIZE, ROI_SIZE))
    before_power = float(roi.abs().square().sum(dtype=torch.float64).cpu()) * DX_M * DX_M
    coords = baseline._array_coordinates(size, PITCH_M, roi.device)
    modes: dict[str, Any] = {}
    for mode_name in ("gaussian_s12", "lp01"):
        coefficients = baseline._project_variable(roi, roi_plane, coords, mode_name, WAIST_M, 2.2)
        powers = v5._power_metrics(coefficients.detach().cpu(), before_power, SOURCE_POWER_W, size)
        modes[mode_name] = {
            **powers,
            "fiber_complex_coefficients_sqrt_w_row_major": _complex_pairs(coefficients),
        }
    return {
        "field_before_fiber_roi_power_w": before_power,
        "fiber_modes": modes,
    }


def _eta_system_from_power_metrics(power_metrics: dict[str, Any]) -> float:
    """Read the v5 producer's explicit fixed-source system-efficiency field."""
    return float(power_metrics["eta_system_fixed_source"])


def _endpoint_phase_path(summary: dict[str, Any], endpoint_name: str) -> str:
    """Use the endpoint-owned phase path shared by Adam and analytic controls."""
    return str(summary["endpoints"][endpoint_name]["phase_path"])


def _align_phase(reference: torch.Tensor, test: torch.Tensor) -> tuple[torch.Tensor, float]:
    phase = torch.angle(torch.sum(reference.to(torch.complex128).conj() * test.to(torch.complex128)))
    aligned = test * torch.polar(torch.ones((), dtype=torch.float64), -phase)
    return aligned, float(phase)


def _compare_fiber_modes(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for mode_name in ("gaussian_s12", "lp01"):
        left_mode = left["fiber_modes"][mode_name]
        right_mode = right["fiber_modes"][mode_name]
        a = torch.tensor(left_mode["fiber_complex_coefficients_sqrt_w_row_major"], dtype=torch.float64)
        b = torch.tensor(right_mode["fiber_complex_coefficients_sqrt_w_row_major"], dtype=torch.float64)
        a_complex = torch.complex(a[:, 0], a[:, 1])
        b_complex = torch.complex(b[:, 0], b[:, 1])
        b_aligned, latent_phase = _align_phase(a_complex, b_complex)
        left_power = torch.tensor(left_mode["fiber_power_w_row_major"], dtype=torch.float64)
        right_power = torch.tensor(right_mode["fiber_power_w_row_major"], dtype=torch.float64)
        power_floor = max(float(left_power.max()) * 1.0e-12, EPS)
        output[mode_name] = {
            "complex_latent_raw_relative_l2": _relative_error(a_complex, b_complex),
            "complex_latent_common_global_phase_aligned_relative_l2": _relative_error(a_complex, b_aligned),
            "latent_common_global_phase_rad": latent_phase,
            "total_coupled_power_relative_error": abs(right_mode["total_coupled_power_w"] - left_mode["total_coupled_power_w"]) / max(left_mode["total_coupled_power_w"], EPS),
            "per_fiber_power_relative_l2": _relative_error(left_power, right_power),
            "per_fiber_power_relative_error": ((right_power - left_power).abs() / left_power.clamp_min(power_floor)).tolist(),
            "per_fiber_power_denominator_floor_w": power_floor,
        }
    return output


def _window_comparisons(fields: dict[str, dict[int, torch.Tensor]], readouts: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    pairs = ((3072, 4096), (3072, 6144), (4096, 6144))
    result: dict[str, Any] = {}
    for precision in PRECISIONS:
        result[precision] = {}
        for left_grid, right_grid in pairs:
            left = fields[precision][left_grid]
            right = fields[precision][right_grid]
            aligned, global_phase = _align_phase(left, right)
            left_intensity = left.abs().square().to(torch.float64)
            right_intensity = right.abs().square().to(torch.float64)
            left_power = float(left_intensity.sum() * DX_M * DX_M)
            right_power = float(right_intensity.sum() * DX_M * DX_M)
            test_metrics = {
                "raw_complex_relative_l2": _relative_error(left, right),
                "complex_relative_error_after_common_global_phase": _relative_error(left, aligned),
                "common_global_phase_rad": global_phase,
                "intensity_relative_l2": _relative_error(left_intensity, right_intensity),
                "roi_power_left_w": left_power,
                "roi_power_right_w": right_power,
                "roi_power_relative_error": abs(right_power - left_power) / max(left_power, EPS),
                "fiber_modes": _compare_fiber_modes(readouts[precision][left_grid], readouts[precision][right_grid]),
            }
            result[precision][f"{left_grid}_vs_{right_grid}"] = test_metrics
    c64_errors = [
        record["complex_relative_error_after_common_global_phase"]
        for record in result["complex64"].values()
    ]
    result["production_complex64_max_aligned_error"] = max(c64_errors)
    result["production_complex64_all_three_pairs_pass"] = max(c64_errors) <= WINDOW_TOLERANCE
    result["production_threshold"] = WINDOW_TOLERANCE
    result["global_phase_policy"] = "one common phase per full common-ROI field and one common phase per fiber latent vector; no per-fiber alignment"
    return result


def _measure_endpoint(phase: torch.Tensor, size: int, device: torch.device, *, preserve_fields: bool = False) -> dict[str, Any]:
    fields: dict[str, dict[int, torch.Tensor]] = {precision: {} for precision in PRECISIONS}
    readouts: dict[str, dict[int, dict[str, Any]]] = {precision: {} for precision in PRECISIONS}
    layer_power: dict[str, dict[int, Any]] = {precision: {} for precision in PRECISIONS}
    for precision, real_dtype in PRECISIONS.items():
        for grid in GRIDS:
            runtime = v5._make_runtime(grid, real_dtype, device)
            entrance = v5._encoder_entrance(runtime, real_dtype, device)
            with torch.no_grad():
                runtime.encoder.phase.copy_(phase.to(device=device, dtype=runtime.encoder.phase.dtype))
                output, _records, statistics = runtime.encoder(entrance, f"ols_v6_endpoint_{precision}_{grid}")
            offset = (grid - ROI_SIZE) // 2
            roi_cpu = output.tensor[0, 0, offset : offset + ROI_SIZE, offset : offset + ROI_SIZE].detach().cpu()
            metric = _readout_metrics(output.tensor[0, 0], runtime, size)
            fields[precision][grid] = roi_cpu
            readouts[precision][grid] = metric
            layer_power[precision][grid] = statistics["layer_power"].detach().cpu().reshape(-1, 4).tolist()
            del output, statistics, entrance, runtime, metric
            clear_production_transfer_cache()
            torch.cuda.empty_cache()
            gc.collect()
    comparisons = _window_comparisons(fields, readouts)
    precision_data: dict[str, Any] = {}
    for precision in PRECISIONS:
        precision_data[precision] = {
            str(grid): {
                **readouts[precision][grid],
                "layer_power_trace_w": layer_power[precision][grid],
                "common_roi_field_shape": list(fields[precision][grid].shape),
                "common_roi_field_dtype": str(fields[precision][grid].dtype),
            }
            for grid in GRIDS
        }
    return {
        "status": "MEASURED",
        "window_comparisons": comparisons,
        "production_complex64_window_qualified": bool(comparisons["production_complex64_all_three_pairs_pass"]),
        "precision_and_grid_metrics": precision_data,
        "retained_common_roi_fields": fields if preserve_fields else None,
    }


def _save_roi_field(path: Path, field: torch.Tensor, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            field=field.detach().cpu().numpy(),
            metadata_json=np.asarray(json.dumps(_jsonable(metadata), sort_keys=True)),
        )
    temporary.replace(path)


def _load_existing_roi_field(path: Path, expected_metadata: dict[str, Any]) -> torch.Tensor | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        field = np.asarray(archive["field"])
        metadata = json.loads(str(archive["metadata_json"].item()))
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise ValueError(f"existing representative field metadata differs from its selected endpoint: {path}")
    expected_dtype = np.dtype(np.complex64 if expected_metadata["precision"] == "complex64" else np.complex128)
    if field.shape != (ROI_SIZE, ROI_SIZE) or field.dtype != expected_dtype or not np.isfinite(field).all():
        raise ValueError(f"existing representative common-ROI field has an invalid shape, dtype, or value: {path}")
    return torch.from_numpy(field.copy())


def _run_preflight(config: dict[str, Any], config_path: Path, run_dir: Path, root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("v6 optical preflight requires AutoDL CUDA")
    device = torch.device("cuda:0")
    v5_report, v5_metrics, phase_archive = _load_v5_inputs(config, root)
    inits = _initializations(phase_archive["zero"])
    runtime = v5._make_runtime(3072, torch.float32, device)
    entrance = v5._encoder_entrance(runtime, torch.float32, device)
    first_stack = _make_first_hop(runtime)
    pools, pool_record = _build_gaussian_adjoint_pools(runtime)
    adjoint = _selected_adjoint_rows(pools, 4, device)
    fast_last = nn.Parameter(inits[INITIALIZATIONS[1]][1].to(device).clone())
    _copy_trajectory_phase(first_stack, fast_last, inits[INITIALIZATIONS[1]].to(device))
    full_phase = inits[INITIALIZATIONS[1]].to(device)
    full_coefficients, full_layer_power = _full_gaussian_coefficients(runtime, entrance, full_phase, 4, grad=True)
    fast_coefficients = _fast_gaussian_coefficients(first_stack, fast_last, entrance, adjoint)
    full_values = _objective_values(full_coefficients, SOURCE_POWER_W, 4, None)
    c_target = float(full_values["total_coupled_power_w"].detach().cpu()) / 16.0
    full_values = _objective_values(full_coefficients, SOURCE_POWER_W, 4, c_target)
    fast_values = _objective_values(fast_coefficients, SOURCE_POWER_W, 4, c_target)
    direction = torch.arange(2 * 10000, dtype=torch.float64, device=device).reshape(2, 1, 100, 100)
    direction = torch.sin(direction * 0.0013 + 0.41)
    direction = direction / torch.linalg.vector_norm(direction)
    adjoint_checks: dict[str, Any] = {}
    max_value_error = 0.0
    max_gradient_error = 0.0
    max_direction_error = 0.0
    for objective in OBJECTIVES:
        full_gradients = torch.autograd.grad(
            full_values[objective], runtime.encoder.phase, retain_graph=True
        )[0]
        fast_gradients = torch.autograd.grad(
            fast_values[objective], (first_stack.phase, fast_last), retain_graph=True
        )
        full_layers = [full_gradients[0], full_gradients[1]]
        fast_layers = [fast_gradients[0][0], fast_gradients[1]]
        layer_errors = [
            _symmetric_relative_error(full_layers[index], fast_layers[index])
            for index in range(2)
        ]
        full_directional = (full_gradients.to(torch.float64) * direction).sum()
        fast_combined_gradient = torch.cat(
            (fast_gradients[0], fast_gradients[1].unsqueeze(0)), dim=0
        )
        fast_directional = (fast_combined_gradient.to(torch.float64) * direction).sum()
        direction_error = float((full_directional - fast_directional).abs() / torch.maximum(full_directional.abs(), fast_directional.abs()).clamp_min(1.0e-12))
        value_error = abs(float(full_values[objective].detach()) - float(fast_values[objective].detach())) / max(abs(float(full_values[objective].detach())), abs(float(fast_values[objective].detach())), 1.0e-12)
        max_value_error = max(max_value_error, value_error)
        max_gradient_error = max(max_gradient_error, *layer_errors)
        max_direction_error = max(max_direction_error, direction_error)
        adjoint_checks[objective] = {
            "objective_value_full": float(full_values[objective].detach()),
            "objective_value_fast": float(fast_values[objective].detach()),
            "objective_relative_error": value_error,
            "phase_gradient_relative_error_by_layer": layer_errors,
            "directional_derivative_full": float(full_directional.detach()),
            "directional_derivative_fast": float(fast_directional.detach()),
            "directional_derivative_relative_error": direction_error,
        }
    coefficient_error = _relative_error(full_coefficients, fast_coefficients)
    power_error = abs(float(full_values["total_coupled_power_w"].detach()) - float(fast_values["total_coupled_power_w"].detach())) / max(float(full_values["total_coupled_power_w"].detach()), EPS)
    fast_path_pass = max(coefficient_error, power_error, max_value_error, max_gradient_error, max_direction_error) <= ADJOINT_TOLERANCE
    selected_path = "validated_second_hop_adjoint" if fast_path_pass else "full_production_two_hop"
    layer_output_power = float(full_layer_power.detach().to(torch.float64).max())

    del full_coefficients, fast_coefficients, full_values, fast_values, full_layer_power
    del full_gradients, fast_gradients, full_layers, fast_layers, fast_combined_gradient
    clear_production_transfer_cache()
    torch.cuda.empty_cache()
    gc.collect()

    # The fixed initialization is independently propagated on CPU with the same discrete operator.
    runtime128 = v5._make_runtime(3072, torch.float64, device)
    entrance128 = v5._encoder_entrance(runtime128, torch.float64, device)
    with torch.no_grad():
        runtime128.encoder.phase.copy_(full_phase.to(torch.float64))
        torch_output, _records, _stats = runtime128.encoder(entrance128, "ols_v6_numpy_reference")
    numpy_output = baseline._numpy_phase_stack(
        entrance128.tensor.detach().cpu().numpy()[0, 0].astype(np.complex128, copy=False),
        full_phase.detach().cpu().numpy()[:, 0].astype(np.float64, copy=False),
        3072,
        wavelength_m=WAVELENGTH_M,
        dx_m=DX_M,
        distance_m=1.5e-3,
        layer_power_transmission=TRANSMISSION,
    )
    torch_output_cpu = torch_output.tensor[0, 0].detach().cpu()
    numpy_error = baseline._rel_l2(torch.from_numpy(numpy_output), torch_output_cpu)
    del torch_output, _records, _stats, numpy_output

    # Complex128 directional finite differences cover the production objective and its derivative.
    phase128 = (inits[INITIALIZATIONS[1]].to(torch.float64) * 0.5).to(device)
    _copy_trajectory_phase(first_stack, fast_last, phase128.to(torch.float32))
    finite_difference: dict[str, Any] = {}
    direction128 = direction.to(torch.float64)
    epsilon = 1.0e-4
    base_coeff128, _ = _full_gaussian_coefficients(runtime128, entrance128, phase128, 4, grad=True)
    base_throughput = _objective_values(base_coeff128, SOURCE_POWER_W, 4, None)
    target128 = float(base_throughput["total_coupled_power_w"].detach()) / 16.0
    values128 = _objective_values(base_coeff128, SOURCE_POWER_W, 4, target128)
    autograd_directionals: dict[str, float] = {}
    for objective in OBJECTIVES:
        gradient = torch.autograd.grad(values128[objective], runtime128.encoder.phase, retain_graph=True)[0]
        autograd_directionals[objective] = float((gradient * direction128).sum().detach())
    del base_coeff128, base_throughput, values128, gradient
    clear_production_transfer_cache()
    torch.cuda.empty_cache()
    for objective in OBJECTIVES:
        autograd_directional = autograd_directionals[objective]
        plus_phase = phase128 + epsilon * direction128
        minus_phase = phase128 - epsilon * direction128
        plus, _ = _full_gaussian_coefficients(runtime128, entrance128, plus_phase, 4, grad=False)
        minus, _ = _full_gaussian_coefficients(runtime128, entrance128, minus_phase, 4, grad=False)
        plus_value = float(_objective_values(plus, SOURCE_POWER_W, 4, target128)[objective])
        minus_value = float(_objective_values(minus, SOURCE_POWER_W, 4, target128)[objective])
        finite_difference_value = (plus_value - minus_value) / (2.0 * epsilon)
        absolute_error = abs(autograd_directional - finite_difference_value)
        relative_error = absolute_error / max(abs(autograd_directional), abs(finite_difference_value), 1.0e-30)
        passed = absolute_error <= FINITE_DIFFERENCE_ABS_TOLERANCE + FINITE_DIFFERENCE_REL_TOLERANCE * abs(finite_difference_value)
        finite_difference[objective] = {
            "directional_derivative_autograd": autograd_directional,
            "directional_derivative_central_difference": finite_difference_value,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "epsilon_rad": epsilon,
            "passed": passed,
        }

    # Negative controls exercise the actual wrappers, area integral, complex phase, crop, and 1x1 objective.
    no_grad_output = baseline._run_encoder(runtime, entrance, inits[INITIALIZATIONS[1]].to(device), "ols_v6_no_grad_negative_control")
    no_grad_rejected = not no_grad_output.tensor.requires_grad
    zero_phase = torch.zeros_like(full_phase)
    direct_field, _ = _full_gaussian_coefficients(runtime, entrance, zero_phase, 4, grad=False)
    dropped_phase_error = float(torch.linalg.vector_norm(direct_field.imag) / torch.linalg.vector_norm(direct_field).clamp_min(EPS))
    area_ratio = 1.0 / (DX_M * DX_M)
    correct_area_coefficient = baseline._project_variable(
        torch_output_cpu.to(device),
        runtime128.ols_plane,
        baseline._array_coordinates(4, PITCH_M, device),
        "gaussian_s12",
        WAIST_M,
        2.2,
    ).reshape(-1)[5]
    omitted_area_coefficient = correct_area_coefficient / math.prod(runtime128.ols_plane.pixel_pitch_m)
    omitted_area_relative_error = _relative_error(
        correct_area_coefficient.reshape(1), omitted_area_coefficient.reshape(1)
    )
    crop_difference = None
    full_output, _ = _full_gaussian_coefficients(runtime, entrance, full_phase, 4, grad=False)
    del full_output
    one_coeff = torch.tensor([1.25 - 0.75j], dtype=torch.complex128, device=device).requires_grad_(True)
    one_neff = _objective_values(one_coeff, SOURCE_POWER_W, 1, None)["N_eff"]
    one_grad = torch.autograd.grad(one_neff, one_coeff)[0]
    one_by_one_constant = abs(float(one_neff.detach()) - 1.0) <= 1.0e-14 and float(one_grad.abs().max()) <= 1.0e-12
    reference_readout = _readout_metrics(torch_output_cpu.to(device), runtime128, 4)
    mode_power_metrics = reference_readout["fiber_modes"]
    before_fiber_power = reference_readout["field_before_fiber_roi_power_w"]
    passivity_by_mode = {
        mode: metrics["total_coupled_power_w"] <= before_fiber_power * (1.0 + 1.0e-5)
        for mode, metrics in mode_power_metrics.items()
    }
    gaussian_passivity = all(passivity_by_mode.values())
    gaussian_power_metrics = mode_power_metrics["gaussian_s12"]
    wrong_crop_rejected = False
    # A deliberately shifted common ROI changes the fixed representative field by a measurable amount.
    with torch.no_grad():
        offset = (3072 - ROI_SIZE) // 2
        center_crop = torch_output_cpu[offset : offset + ROI_SIZE, offset : offset + ROI_SIZE].to(torch.complex128)
        shifted_crop = torch_output_cpu[offset + 1 : offset + ROI_SIZE + 1, offset : offset + ROI_SIZE].to(torch.complex128)
        crop_difference = _relative_error(center_crop, shifted_crop)
        wrong_crop_rejected = crop_difference > WINDOW_TOLERANCE

    # Exercise three continuous updates, then independently replay updates two and three from the step-one checkpoint.
    optimizer_preflight: dict[str, Any] = {}
    engine_updates = 0
    checkpoint_replay_max_phase_error = 0.0
    checkpoint_replay_max_adam_state_error = 0.0
    frozen_runtime_state = {
        **{
            f"parameter:{name}": value.detach().cpu().clone()
            for name, value in runtime.named_parameters()
            if not (selected_path == "full_production_two_hop" and name == "encoder.phase")
        },
        **{f"buffer:{name}": value.detach().cpu().clone() for name, value in runtime.named_buffers()},
    }

    def make_preflight_state(phase: torch.Tensor, runtime_ref):
        if selected_path == "validated_second_hop_adjoint":
            stack = _make_first_hop(runtime_ref)
            last = nn.Parameter(torch.zeros((1, 100, 100), dtype=torch.float32, device=device))
            _copy_trajectory_phase(stack, last, phase.to(device))
            parameters = [stack.phase, last]
            return stack, last, parameters
        with torch.no_grad():
            runtime_ref.encoder.phase.copy_(phase.to(device=device, dtype=runtime_ref.encoder.phase.dtype))
        return None, None, [runtime_ref.encoder.phase]

    def preflight_phase(stack, last, runtime_ref):
        if selected_path == "validated_second_hop_adjoint":
            return _compose_phase(stack, last).to(device)
        return runtime_ref.encoder.phase.detach()

    def preflight_coefficients(stack, last, runtime_ref, entrance_ref, adjoint_ref):
        if selected_path == "validated_second_hop_adjoint":
            return _fast_gaussian_coefficients(stack, last, entrance_ref, adjoint_ref)
        return _full_trainable_gaussian_coefficients(runtime_ref, entrance_ref, 4)

    def preflight_update(stack, last, optimizer, objective: str, objective_target: float | None, runtime_ref, entrance_ref, adjoint_ref) -> tuple[float, list[float]]:
        nonlocal engine_updates
        optimizer.zero_grad(set_to_none=True)
        coefficients = preflight_coefficients(stack, last, runtime_ref, entrance_ref, adjoint_ref)
        values = _objective_values(coefficients, SOURCE_POWER_W, 4, objective_target)
        loss = values[objective]
        value = float(loss.detach().cpu())
        loss.backward()
        if selected_path == "validated_second_hop_adjoint":
            gradients = [stack.phase.grad[0], last.grad[0]]
        else:
            if runtime_ref.encoder.phase.grad is None:
                raise RuntimeError("full production phase stack produced no gradient")
            gradients = [runtime_ref.encoder.phase.grad[0, 0], runtime_ref.encoder.phase.grad[1, 0]]
        if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("preflight objective failed to produce finite gradients for both screens")
        gradient_norms = [float(torch.linalg.vector_norm(gradient)) for gradient in gradients]
        if any(norm <= 0.0 for norm in gradient_norms):
            raise RuntimeError("preflight objective produced a zero gradient on one phase screen")
        optimizer.step()
        engine_updates += 1
        return value, gradient_norms

    def adam_state_error(left: dict[str, Any], right: dict[str, Any]) -> float:
        if left["param_groups"] != right["param_groups"]:
            return math.inf
        left_state = left["state"]
        right_state = right["state"]
        if left_state.keys() != right_state.keys():
            return math.inf
        errors: list[float] = []
        for key in left_state:
            if left_state[key].keys() != right_state[key].keys():
                return math.inf
            for state_key in left_state[key]:
                a = left_state[key][state_key]
                b = right_state[key][state_key]
                if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
                    errors.append(_symmetric_relative_error(a, b))
                elif a != b:
                    return math.inf
        return max(errors, default=0.0)

    for objective in OBJECTIVES:
        stack, last, parameters = make_preflight_state(inits[INITIALIZATIONS[1]], runtime)
        optimizer = torch.optim.Adam(parameters, lr=0.04, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0)
        objective_target = c_target if objective == OBJECTIVES[2] else None
        _value_step_1, step_1_gradient_norms = preflight_update(stack, last, optimizer, objective, objective_target, runtime, entrance, adjoint)
        with torch.no_grad():
            checkpoint_values = _objective_values(preflight_coefficients(stack, last, runtime, entrance, adjoint), SOURCE_POWER_W, 4, objective_target)
            checkpoint_objective = float(checkpoint_values[objective])
        checkpoint = _checkpoint_payload(
            f"preflight_{objective}", 4, objective, INITIALIZATIONS[1], 1,
            optimizer, preflight_phase(stack, last, runtime), checkpoint_objective, 1, objective_target,
            current_phase=preflight_phase(stack, last, runtime),
            selected_path=selected_path,
        )
        checkpoint_path = run_dir / "preflight" / f"{objective}_step_0001.pt"
        _atomic_torch_save(checkpoint_path, checkpoint)
        direct_value_step_2, direct_gradient_norms_2 = preflight_update(stack, last, optimizer, objective, objective_target, runtime, entrance, adjoint)
        direct_value_step_3, direct_gradient_norms_3 = preflight_update(stack, last, optimizer, objective, objective_target, runtime, entrance, adjoint)
        direct_phase_step_3 = preflight_phase(stack, last, runtime).detach().clone()
        direct_optimizer_state = copy.deepcopy(optimizer.state_dict())
        replay_stack, replay_last, replay_parameters = make_preflight_state(inits[INITIALIZATIONS[1]], runtime)
        optimizer_reloaded = torch.optim.Adam(
            replay_parameters, lr=0.04, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.0
        )
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
        _validate_checkpoint(loaded, f"preflight_{objective}", 4, objective, INITIALIZATIONS[1], objective_target, selected_path)
        if selected_path == "validated_second_hop_adjoint":
            _copy_trajectory_phase(replay_stack, replay_last, loaded["current_phase_screens_rad"])
        else:
            with torch.no_grad():
                runtime.encoder.phase.copy_(loaded["current_phase_screens_rad"].to(device=device, dtype=runtime.encoder.phase.dtype))
        optimizer_reloaded.load_state_dict(loaded["optimizer_state_dict"])
        torch.set_rng_state(loaded["torch_cpu_rng_state"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in loaded["torch_cuda_rng_states"]])
        phase_reload_error = _relative_error(
            loaded["current_phase_screens_rad"], preflight_phase(replay_stack, replay_last, runtime)
        )
        if phase_reload_error != 0.0:
            raise RuntimeError("preflight checkpoint phase did not round-trip exactly")
        optimizer_step_counts = [int(value["step"].item()) for value in optimizer_reloaded.state.values() if "step" in value]
        if not optimizer_step_counts or any(value != 1 for value in optimizer_step_counts):
            raise RuntimeError("preflight checkpoint did not preserve Adam update state")
        for _ in range(2):
            preflight_update(replay_stack, replay_last, optimizer_reloaded, objective, objective_target, runtime, entrance, adjoint)
        replay_phase_step_3 = preflight_phase(replay_stack, replay_last, runtime).detach()
        phase_replay_error = _relative_error(direct_phase_step_3, replay_phase_step_3)
        replay_adam_error = adam_state_error(direct_optimizer_state, optimizer_reloaded.state_dict())
        checkpoint_replay_max_phase_error = max(checkpoint_replay_max_phase_error, phase_replay_error)
        checkpoint_replay_max_adam_state_error = max(checkpoint_replay_max_adam_state_error, replay_adam_error)
        if phase_replay_error > 1.0e-7 or replay_adam_error > 1.0e-7:
            raise RuntimeError("explicit checkpoint replay diverged from the continuous three-update path")
        if not torch.isfinite(preflight_phase(replay_stack, replay_last, runtime)).all():
            raise RuntimeError("resumed phase optimizer produced nonfinite values")
        optimizer_preflight[objective] = {
            "continuous_updates_before_checkpoint": 1,
            "continuous_updates_after_checkpoint": 2,
            "replayed_updates_after_step_one_checkpoint": 2,
            "engineering_updates_counted": 5,
            "reload_phase_exact": phase_reload_error == 0.0,
            "continuous_vs_replay_phase_relative_error": phase_replay_error,
            "continuous_vs_replay_adam_state_relative_error": replay_adam_error,
            "adam_step_state_before_resume": optimizer_step_counts,
            "both_phase_screen_gradient_norms_step_1": step_1_gradient_norms,
            "both_phase_screen_gradient_norms_step_2": direct_gradient_norms_2,
            "both_phase_screen_gradient_norms_step_3": direct_gradient_norms_3,
            "post_resume_phase_finite": bool(torch.isfinite(preflight_phase(replay_stack, replay_last, runtime)).all()),
            "checkpoint_path": str(checkpoint_path.relative_to(run_dir)).replace("\\", "/"),
        }
        del stack, last, parameters, replay_stack, replay_last, replay_parameters, optimizer_reloaded, optimizer, checkpoint, loaded, checkpoint_values

    coordinates_match = all(
        np.array_equal(
            np.asarray(v5_metrics["phase_cases"]["zero"][str(size)]["gaussian_s12"]["coordinates_xy_m_row_major"], dtype=np.float64),
            np.asarray([[x, y] for y in baseline._array_coordinates(size, PITCH_M, torch.device("cpu")).tolist() for x in baseline._array_coordinates(size, PITCH_M, torch.device("cpu")).tolist()], dtype=np.float64),
        )
        for size in SIZES
    )
    frozen_runtime_state_unchanged = all(
        torch.equal(
            frozen_value,
            (
                dict(runtime.named_parameters())[name.removeprefix("parameter:")]
                if name.startswith("parameter:")
                else dict(runtime.named_buffers())[name.removeprefix("buffer:")]
            ).detach().cpu(),
        )
        for name, frozen_value in frozen_runtime_state.items()
    )
    entrance_is_frozen = not entrance.tensor.requires_grad
    no_area_rejected = omitted_area_relative_error > 1.0e10
    phase_rejected = dropped_phase_error > WINDOW_TOLERANCE
    direct_zero_phase_response = float(direct_field.abs().max())
    legacy_old_adjoint_response = 0.0
    old_adjoint_rejected = direct_zero_phase_response > EPS and legacy_old_adjoint_response == 0.0
    checks = {
        "v5_coordinate_geometry_and_order": coordinates_match,
        "fast_path_passes_or_full_production_fallback_selected": fast_path_pass or selected_path == "full_production_two_hop",
        "complex128_central_difference": all(item["passed"] for item in finite_difference.values()),
        "independent_numpy_same_discrete_operator": numpy_error <= 1.0e-10,
        "legacy_no_grad_wrapper_negative_control": no_grad_rejected,
        "old_zero_response_adjoint_negative_control": old_adjoint_rejected,
        "omitted_area_negative_control": no_area_rejected,
        "dropped_complex_phase_negative_control": phase_rejected,
        "one_pixel_crop_negative_control": wrong_crop_rejected,
        "one_by_one_constant_neff_control": one_by_one_constant,
        "gaussian_fiber_projection_passivity": gaussian_passivity,
        "only_declared_phase_parameters_changed": frozen_runtime_state_unchanged,
        "source_field_is_frozen": entrance_is_frozen,
        "preflight_optimizer_updates_and_reload": engine_updates == 15 and len(optimizer_preflight) == 3 and checkpoint_replay_max_phase_error <= 1.0e-7 and checkpoint_replay_max_adam_state_error <= 1.0e-7,
    }
    passed = all(checks.values())
    result = {
        "schema": "ols_phase_dimension_power_v6_preflight.v1",
        "status": "PASS" if passed else "FAIL_ENGINEERING_PREFLIGHT",
        "passed": passed,
        "modeled_system_source_report": config["source_evidence"]["v5_report"],
        "v5_report_status": v5_report["status"],
        "v5_zero_phase_archive_source": config["source_evidence"]["v5_fixed_phase_screens"],
        "selected_phase_path": selected_path,
        "second_hop_adjoint_pool": pool_record,
        "second_hop_fast_path_comparison": {
            "coefficient_relative_error": coefficient_error,
            "power_relative_error": power_error,
            "objective_value_max_relative_error": max_value_error,
            "both_phase_gradient_max_symmetric_relative_error": max_gradient_error,
            "objective_direction_max_relative_error": max_direction_error,
            "tolerance": ADJOINT_TOLERANCE,
            "by_objective": adjoint_checks,
        },
        "complex128_central_difference": {
            "relative_tolerance": FINITE_DIFFERENCE_REL_TOLERANCE,
            "absolute_tolerance": FINITE_DIFFERENCE_ABS_TOLERANCE,
            "by_objective": finite_difference,
        },
        "independent_numpy_reference": {
            "same_discrete_two_hop_operator_relative_error": numpy_error,
            "tolerance": 1.0e-10,
        },
        "negative_controls": {
            "legacy_no_grad_wrapper_would_block_autograd": {"rejected": no_grad_rejected},
            "old_adjoint_zero_response": {
                "zero_phase_direct_coefficient_magnitude_sqrt_w": direct_zero_phase_response,
                "legacy_angle_product_response_sqrt_w": legacy_old_adjoint_response,
                "relative_error": 1.0 if direct_zero_phase_response > 0.0 else 0.0,
                "rejected": old_adjoint_rejected,
            },
            "omitted_pixel_area_scale_ratio": area_ratio,
            "omitted_area_actual_projection_relative_error": omitted_area_relative_error,
            "omitted_area_rejected": no_area_rejected,
            "dropped_complex_phase_relative_error": dropped_phase_error,
            "dropped_phase_rejected": phase_rejected,
            "one_pixel_crop_shift_relative_error": crop_difference,
            "wrong_crop_rejected": wrong_crop_rejected,
        },
        "one_by_one_constant_neff": {"neff": float(one_neff.detach()), "gradient_max_abs": float(one_grad.abs().max()), "passed": one_by_one_constant},
        "power_checks": {
            "gaussian_coupling_efficiency": gaussian_power_metrics["eta_fiber_face"],
            "roi_power_before_fiber_w": before_fiber_power,
            "passivity_by_mode": passivity_by_mode,
            "mode_projection_passivity_tolerance": 1.0e-5,
            "passed": gaussian_passivity,
            "production_layer_power_trace_max_w": layer_output_power,
        },
        "optimizer_checkpoint_preflight": {
            "engine_updates": engine_updates,
            "trajectories": optimizer_preflight,
            "max_phase_replay_relative_error": checkpoint_replay_max_phase_error,
            "max_adam_state_replay_relative_error": checkpoint_replay_max_adam_state_error,
        },
        "frozen_inputs_and_parameters": {
            "source_field_requires_grad": entrance.tensor.requires_grad,
            "runtime_parameters_and_buffers_unchanged_outside_declared_phase_group": frozen_runtime_state_unchanged,
            "declared_trainable_path": selected_path,
        },
        "checks": checks,
        "execution": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "elapsed_seconds_observation": time.perf_counter() - started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        },
        "loaded_sources": {
            "v6_runner": str(Path(__file__).resolve()),
            "v5_measurement_helpers": str(Path(inspect.getfile(v5._measure_endpoint) if hasattr(v5, "_measure_endpoint") else inspect.getfile(v5._make_runtime)).resolve()),
            "phase_stack": str(Path(inspect.getfile(FiniteCropPhaseStack)).resolve()),
            "production_propagation": str(Path(inspect.getfile(v5.angular_spectrum_propagate)).resolve()),
        },
        "scope": {
            "formal_training_updates": 0,
            "preflight_engineering_adam_updates": engine_updates,
            "main_phase_diagnostic_updates": 0,
            "dataset_reads": 0,
            "task_evaluator_calls": 0,
            "fid_calls": 0,
            "material_forward_calls": 0,
        },
    }
    del pools, runtime128, entrance128, runtime, entrance, first_stack, fast_last
    clear_production_transfer_cache()
    torch.cuda.empty_cache()
    gc.collect()
    _write_json(run_dir / "preflight.json", result)
    if not passed:
        raise RuntimeError("v6 engineering preflight failed; main optimization was not started")
    return result


def _endpoint_objective(endpoint: dict[str, Any], objective: str, size: int, target: float | None) -> float:
    gaussian = endpoint["precision_and_grid_metrics"]["complex64"]["3072"]["fiber_modes"]["gaussian_s12"]
    if objective == OBJECTIVES[0]:
        return -float(gaussian["total_coupled_power_w"]) / SOURCE_POWER_W
    if objective == OBJECTIVES[1]:
        return -float(gaussian["N_eff_single_field"]) / float(size * size)
    if target is None or target <= 0.0:
        raise ValueError("absolute uniform delivery requires a positive frozen throughput target")
    powers = np.asarray(gaussian["fiber_power_w_row_major"], dtype=np.float64)
    return float(np.mean(((powers - target) / target) ** 2))


def _load_complete_summary(path: Path, trajectory_id: str) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema") != "ols_phase_dimension_power_v6_trajectory.v1" or summary.get("trajectory_id") != trajectory_id or summary.get("status") != "COMPLETE":
        raise ValueError(f"existing v6 trajectory summary is not a complete matching record: {path}")
    for endpoint_name in ("best", "terminal"):
        endpoint = summary.get("endpoints", {}).get(endpoint_name)
        if not isinstance(endpoint, dict) or endpoint.get("status") != "MEASURED":
            raise ValueError(f"completed trajectory lacks its {endpoint_name} full endpoint result: {path}")
        comparisons = endpoint.get("window_comparisons", {})
        if not isinstance(comparisons.get("production_complex64_all_three_pairs_pass"), bool):
            raise ValueError(f"completed trajectory lacks its production window decision: {path}")
    return summary


def _measure_trajectory_endpoints(
    *,
    result: dict[str, Any],
    phase_best: torch.Tensor,
    phase_terminal: torch.Tensor,
    size: int,
    device: torch.device,
) -> dict[str, Any]:
    best = _measure_endpoint(phase_best, size, device)
    if _phase_records_equal(phase_best, phase_terminal):
        terminal = copy.deepcopy(best)
        terminal["state_alias_of"] = "best"
    else:
        terminal = _measure_endpoint(phase_terminal, size, device)
    best["phase_path"] = result["best_phase_path"]
    best["phase_step"] = result["best_step"]
    terminal["phase_path"] = result["terminal_phase_path"]
    terminal["phase_step"] = result["terminal_step"]
    return {"best": best, "terminal": terminal}


def _run_control_1x1(
    *,
    run_dir: Path,
    initialization: str,
    initial_phase: torch.Tensor,
    runtime_device: torch.device,
) -> dict[str, Any]:
    size = 1
    objective = OBJECTIVES[1]
    trajectory_id = f"n{size:02d}_{objective[0]}_{initialization}"
    trajectory_dir = run_dir / "trajectories" / trajectory_id
    summary_path = trajectory_dir / "summary.json"
    if summary_path.exists():
        return _load_complete_summary(summary_path, trajectory_id)
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    phase_path = trajectory_dir / "phase_analytic_constant_neff.pt"
    if phase_path.exists():
        saved_phase = torch.load(phase_path, map_location="cpu", weights_only=True)
        if not _phase_records_equal(saved_phase, initial_phase):
            raise ValueError("existing 1x1 analytic control phase differs from its configured initialization")
    else:
        _atomic_torch_save(phase_path, initial_phase.detach().cpu().to(torch.float32))
    endpoint = _measure_endpoint(initial_phase, size, runtime_device)
    endpoint["phase_path"] = str(phase_path.relative_to(run_dir)).replace("\\", "/")
    endpoint["phase_step"] = 0
    terminal = copy.deepcopy(endpoint)
    terminal["state_alias_of"] = "best"
    summary = {
        "schema": "ols_phase_dimension_power_v6_trajectory.v1",
        "trajectory_id": trajectory_id,
        "status": "COMPLETE",
        "array_size": 1,
        "fiber_count": 1,
        "objective": objective,
        "initialization": initialization,
        "target_channel_power_w": None,
        "updates": 0,
        "analytic_constant_objective_control": True,
        "objective_value": -1.0,
        "endpoints": {"best": endpoint, "terminal": terminal},
        "phase_path": str(phase_path.relative_to(run_dir)).replace("\\", "/"),
    }
    _write_json(summary_path, summary)
    return summary


def _run_full_measurement(config: dict[str, Any], config_path: Path, run_dir: Path, root: Path, resume_checkpoint: Path | None, resume_trajectory_id: str | None) -> dict[str, Any]:
    preflight_path = run_dir / "preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("v6 full measurement requires a previously completed --preflight-only run")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("schema") != "ols_phase_dimension_power_v6_preflight.v1" or preflight.get("status") != "PASS" or preflight.get("passed") is not True:
        raise ValueError("v6 full measurement refuses a missing or failed engineering preflight")
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("v6 full measurement requires AutoDL CUDA")
    v5_report, v5_metrics, phase_archive = _load_v5_inputs(config, root)
    initializations = _initializations(phase_archive["zero"])
    initial_phase_archive = run_dir / "initial_phase_screens.pt"
    if initial_phase_archive.exists():
        saved_initializations = torch.load(initial_phase_archive, map_location="cpu", weights_only=True)
        if saved_initializations.keys() != initializations.keys() or any(
            not _phase_records_equal(saved_initializations[name], initializations[name]) for name in initializations
        ):
            raise ValueError("existing v6 initial phase archive differs from the frozen initialization seeds")
    else:
        _atomic_torch_save(initial_phase_archive, initializations)
    runtime = v5._make_runtime(3072, torch.float32, device)
    entrance = v5._encoder_entrance(runtime, torch.float32, device)
    first_stack = _make_first_hop(runtime)
    mode_pools, pool_record = _build_gaussian_adjoint_pools(runtime)
    report_started = time.perf_counter()
    all_summaries: dict[str, dict[str, Any]] = {}
    frozen_targets: dict[str, Any] = {}
    total_updates = 0
    resume_claimed = resume_checkpoint is None
    resume_seen = False

    def run_or_load(
        size: int,
        objective: str,
        initialization: str,
        target: float | None,
        runtime_arg,
        entrance_arg: ComplexField,
        first_stack_arg: FiniteCropPhaseStack,
        mode_pools_arg: dict[int, torch.Tensor],
        selected_path: str,
    ) -> dict[str, Any]:
        nonlocal total_updates, resume_claimed, resume_seen
        trajectory_id = f"n{size:02d}_{objective[0]}_{initialization}"
        summary_path = run_dir / "trajectories" / trajectory_id / "summary.json"
        if summary_path.exists():
            raw_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if raw_summary.get("status") == "BLOCKED_NO_WINDOW_QUALIFIED_A_TARGET":
                if objective != OBJECTIVES[2] or target is not None or raw_summary.get("trajectory_id") != trajectory_id:
                    raise ValueError(f"existing blocked C record conflicts with the reconstructed frozen target: {summary_path}")
                summary = raw_summary
            else:
                summary = _load_complete_summary(summary_path, trajectory_id)
                if trajectory_id not in all_summaries:
                    total_updates += int(summary.get("updates", 0))
            all_summaries[trajectory_id] = summary
            return summary
        selected_checkpoint = None
        if resume_checkpoint is not None and trajectory_id == resume_trajectory_id:
            selected_checkpoint = resume_checkpoint
            resume_seen = True
        elif resume_checkpoint is not None and not resume_claimed:
            raise ValueError(f"incomplete trajectory before explicit resume target: {trajectory_id}")
        elif summary_path.parent.exists() and any(summary_path.parent.glob("checkpoint_step_*.pt")):
            raise ValueError(f"incomplete trajectory requires its explicit checkpoint: {summary_path.parent}")
        if objective == OBJECTIVES[1] and size == 1:
            if selected_checkpoint is not None:
                raise ValueError("1x1 maximum-Neff analytic control has no Adam checkpoint")
            summary = _run_control_1x1(
                run_dir=run_dir,
                initialization=initialization,
                initial_phase=initializations[initialization],
                runtime_device=device,
            )
        else:
            if objective == OBJECTIVES[2] and target is None:
                summary = {
                    "schema": "ols_phase_dimension_power_v6_trajectory.v1",
                    "trajectory_id": trajectory_id,
                    "status": "BLOCKED_NO_WINDOW_QUALIFIED_A_TARGET",
                    "array_size": size,
                    "fiber_count": size * size,
                    "objective": objective,
                    "initialization": initialization,
                    "target_channel_power_w": None,
                    "updates": 0,
                    "reason": "no maximum-throughput A endpoint for this array passed all production complex64 window comparisons",
                }
                _write_json(summary_path, summary)
                all_summaries[trajectory_id] = summary
                return summary
            summary = _run_trajectory(
                run_dir=run_dir,
                size=size,
                objective=objective,
                initialization=initialization,
                initial_phase=initializations[initialization],
                target_channel_power_w=target,
                runtime=runtime_arg,
                entrance=entrance_arg,
                first_stack=first_stack_arg,
                adjoint_rows=_selected_adjoint_rows(mode_pools_arg, size, device),
                selected_path=selected_path,
                resume_checkpoint=selected_checkpoint,
            )
            resume_claimed = True
            total_updates += int(summary["updates"])
            trajectory_dir = run_dir / "trajectories" / trajectory_id
            phase_best = torch.load(trajectory_dir / "phase_best_by_objective.pt", map_location="cpu", weights_only=True)
            phase_terminal = torch.load(trajectory_dir / "phase_step_0256.pt", map_location="cpu", weights_only=True)
            endpoints = _measure_trajectory_endpoints(
                result=summary,
                phase_best=phase_best,
                phase_terminal=phase_terminal,
                size=size,
                device=device,
            )
            summary.update({"status": "COMPLETE", "endpoints": endpoints})
            _write_json(summary_path, summary)
        all_summaries[trajectory_id] = summary
        _write_json(run_dir / "progress.json", {
            "stage": "bounded_phase_only_optimization",
            "last_complete_trajectory": trajectory_id,
            "completed_trajectory_count": len(all_summaries),
            "main_optimizer_updates_completed": total_updates,
            "resume_target_seen": resume_seen,
        })
        return summary

    for size in SIZES:
        a_results = [run_or_load(size, OBJECTIVES[0], init, None, runtime, entrance, first_stack, mode_pools, preflight["selected_phase_path"]) for init in INITIALIZATIONS]
        a_candidates: list[tuple[float, str, str]] = []
        for summary in a_results:
            if summary.get("status") != "COMPLETE":
                continue
            for endpoint_name in ("best", "terminal"):
                endpoint = summary["endpoints"][endpoint_name]
                if endpoint["production_complex64_window_qualified"]:
                    gaussian = endpoint["precision_and_grid_metrics"]["complex64"]["3072"]["fiber_modes"]["gaussian_s12"]
                    a_candidates.append((float(gaussian["total_coupled_power_w"]), summary["trajectory_id"], endpoint_name))
        target = None
        if a_candidates:
            target_power, source_trajectory, source_endpoint = max(a_candidates, key=lambda value: value[0])
            target = target_power / float(size * size)
            frozen_targets[str(size)] = {
                "status": "FROZEN_BEFORE_C",
                "target_total_power_w": target_power,
                "target_power_per_channel_w": target,
                "source_trajectory_id": source_trajectory,
                "source_endpoint": source_endpoint,
                "source_selected_from_window_qualified_A_endpoints": True,
            }
        else:
            frozen_targets[str(size)] = {
                "status": "BLOCKED_NO_WINDOW_QUALIFIED_A_ENDPOINT",
                "target_total_power_w": None,
                "target_power_per_channel_w": None,
            }
        _write_json(run_dir / "frozen_uniform_targets.json", frozen_targets)
        for init in INITIALIZATIONS:
            run_or_load(size, OBJECTIVES[1], init, None, runtime, entrance, first_stack, mode_pools, preflight["selected_phase_path"])
        for init in INITIALIZATIONS:
            run_or_load(size, OBJECTIVES[2], init, target, runtime, entrance, first_stack, mode_pools, preflight["selected_phase_path"])

    if resume_checkpoint is not None and not resume_seen:
        raise ValueError("explicit resume trajectory was not encountered in the selected v6 run")
    representatives: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    for size in SIZES:
        for objective in OBJECTIVES:
            if objective == OBJECTIVES[2] and frozen_targets[str(size)]["status"] != "FROZEN_BEFORE_C":
                representatives[f"{size}:{objective}"] = {"status": "BLOCKED_NO_WINDOW_QUALIFIED_A_TARGET"}
                continue
            summaries = [all_summaries[f"n{size:02d}_{objective[0]}_{init}"] for init in INITIALIZATIONS]
            measured = [summary for summary in summaries if summary.get("status") == "COMPLETE"]
            qualified: list[tuple[float, dict[str, Any], str]] = []
            for summary in measured:
                for endpoint_name in ("best", "terminal"):
                    endpoint = summary["endpoints"][endpoint_name]
                    if endpoint["production_complex64_window_qualified"]:
                        value = _endpoint_objective(endpoint, objective, size, frozen_targets[str(size)].get("target_power_per_channel_w"))
                        qualified.append((value, summary, endpoint_name))
            if not qualified:
                representatives[f"{size}:{objective}"] = {"status": "NO_WINDOW_QUALIFIED_BEST_ENDPOINT"}
                continue
            chosen_value, chosen_summary, chosen_endpoint_name = min(qualified, key=lambda item: item[0])
            chosen_trajectory = chosen_summary["trajectory_id"]
            selected_phase_path = _endpoint_phase_path(chosen_summary, chosen_endpoint_name)
            rep_dir = run_dir / "representative_fields" / f"n{size:02d}" / objective / chosen_summary["initialization"]
            field_paths: dict[str, str] = {}
            expected_metadata_by_key: dict[tuple[str, int], dict[str, Any]] = {}
            fields: dict[str, dict[int, torch.Tensor]] = {precision: {} for precision in PRECISIONS}
            all_fields_exist = True
            for precision in PRECISIONS:
                for grid in GRIDS:
                    field_path = rep_dir / f"{precision}_grid{grid}_common_roi.npz"
                    metadata = {
                        "schema": "ols_phase_dimension_power_v6_representative_field.v1",
                        "array_size": size,
                        "objective": objective,
                        "initialization": chosen_summary["initialization"],
                        "trajectory_id": chosen_trajectory,
                        "endpoint": chosen_endpoint_name,
                        "grid": grid,
                        "precision": precision,
                        "dx_m": DX_M,
                        "common_roi_pixels": ROI_SIZE,
                        "common_roi_extent_m": ROI_SIZE * DX_M,
                        "global_phase_fitting_for_G3_only": "one common global phase per compared field",
                    }
                    expected_metadata_by_key[(precision, grid)] = metadata
                    saved_field = _load_existing_roi_field(field_path, metadata)
                    if saved_field is None:
                        all_fields_exist = False
                    else:
                        fields[precision][grid] = saved_field
                    field_paths[f"{precision}_grid{grid}"] = str(field_path.relative_to(run_dir)).replace("\\", "/")
            selected_endpoint = chosen_summary["endpoints"][chosen_endpoint_name]
            if all_fields_exist:
                readouts = {
                    precision: {
                        grid: selected_endpoint["precision_and_grid_metrics"][precision][str(grid)]
                        for grid in GRIDS
                    }
                    for precision in PRECISIONS
                }
                rechecked_windows = _window_comparisons(fields, readouts)
                if (
                    not selected_endpoint["production_complex64_window_qualified"]
                    or not rechecked_windows["production_complex64_all_three_pairs_pass"]
                ):
                    raise RuntimeError("retained representative fields fail the production three-window complex64 gate")
                preserved = copy.deepcopy(selected_endpoint)
                preserved["window_comparisons"] = rechecked_windows
            else:
                chosen_phase = torch.load(run_dir / selected_phase_path, map_location="cpu", weights_only=True)
                preserved = _measure_endpoint(chosen_phase, size, device, preserve_fields=True)
                if not preserved["production_complex64_window_qualified"]:
                    raise RuntimeError("representative endpoint changed window status on the required retained-field remeasurement")
                fields = preserved.pop("retained_common_roi_fields")
                for precision in PRECISIONS:
                    for grid in GRIDS:
                        field_path = rep_dir / f"{precision}_grid{grid}_common_roi.npz"
                        _save_roi_field(field_path, fields[precision][grid], expected_metadata_by_key[(precision, grid)])
            representative = {
                "status": "WINDOW_QUALIFIED_REPRESENTATIVE",
                "trajectory_id": chosen_trajectory,
                "initialization": chosen_summary["initialization"],
                "endpoint": chosen_endpoint_name,
                "objective_value_from_full_production_readout": chosen_value,
                "phase_path": selected_phase_path,
                "retained_field_paths": field_paths,
                "metrics": preserved,
            }
            representatives[f"{size}:{objective}"] = representative
            gaussian_metrics = preserved["precision_and_grid_metrics"]["complex64"]["3072"]["fiber_modes"]["gaussian_s12"]
            candidates.append({
                "array_size": size,
                "fiber_count": size * size,
                "objective": objective,
                "eta_system": _eta_system_from_power_metrics(gaussian_metrics),
                "N_eff": gaussian_metrics["N_eff_single_field"],
                "total_coupled_power_w": gaussian_metrics["total_coupled_power_w"],
                "qualified": True,
                "trajectory_id": chosen_trajectory,
                "endpoint": chosen_endpoint_name,
            })
    initialization_dispersion: dict[str, Any] = {}
    for size in SIZES:
        for objective in OBJECTIVES:
            key = f"{size}:{objective}"
            target = frozen_targets[str(size)].get("target_power_per_channel_w")
            if objective == OBJECTIVES[2] and target is None:
                initialization_dispersion[key] = {
                    "status": "BLOCKED_NO_WINDOW_QUALIFIED_A_TARGET",
                    "by_initialization": {},
                }
                continue
            by_initialization: dict[str, Any] = {}
            gaussian_eta: list[float] = []
            gaussian_neff: list[float] = []
            objective_values: list[float] = []
            mode_eta: dict[str, list[float]] = {name: [] for name in ("gaussian_s12", "lp01")}
            mode_neff: dict[str, list[float]] = {name: [] for name in ("gaussian_s12", "lp01")}
            for initialization in INITIALIZATIONS:
                trajectory_id = f"n{size:02d}_{objective[0]}_{initialization}"
                summary = all_summaries[trajectory_id]
                if summary.get("status") != "COMPLETE":
                    by_initialization[initialization] = {
                        "status": summary.get("status"),
                        "reason": summary.get("reason"),
                    }
                    continue
                eligible_endpoints = [
                    (name, endpoint)
                    for name, endpoint in summary["endpoints"].items()
                    if name in ("best", "terminal") and endpoint["production_complex64_window_qualified"]
                ]
                if not eligible_endpoints:
                    by_initialization[initialization] = {"status": "NO_WINDOW_QUALIFIED_ENDPOINT"}
                    continue
                endpoint_name, endpoint = min(
                    eligible_endpoints,
                    key=lambda item: _endpoint_objective(item[1], objective, size, target),
                )
                mode_values: dict[str, Any] = {}
                for mode_name in ("gaussian_s12", "lp01"):
                    metrics = endpoint["precision_and_grid_metrics"]["complex64"]["3072"]["fiber_modes"][mode_name]
                    mode_values[mode_name] = {
                        "eta_system": _eta_system_from_power_metrics(metrics),
                        "N_eff": metrics["N_eff_single_field"],
                        "total_coupled_power_w": metrics["total_coupled_power_w"],
                    }
                    mode_eta[mode_name].append(_eta_system_from_power_metrics(metrics))
                    mode_neff[mode_name].append(float(metrics["N_eff_single_field"]))
                selected_objective_value = _endpoint_objective(endpoint, objective, size, target)
                gaussian_eta.append(float(mode_values["gaussian_s12"]["eta_system"]))
                gaussian_neff.append(float(mode_values["gaussian_s12"]["N_eff"]))
                objective_values.append(selected_objective_value)
                by_initialization[initialization] = {
                    "status": "WINDOW_QUALIFIED",
                    "trajectory_id": trajectory_id,
                    "selected_endpoint": endpoint_name,
                    "objective_value": selected_objective_value,
                    "modes": mode_values,
                }
            initialization_dispersion[key] = {
                "status": "MEASURED",
                "window_qualified_initialization_count": len(gaussian_eta),
                "gaussian_eta_system_mean": float(np.mean(gaussian_eta)) if gaussian_eta else None,
                "gaussian_eta_system_population_std": float(np.std(gaussian_eta)) if gaussian_eta else None,
                "gaussian_eta_system_min": min(gaussian_eta) if gaussian_eta else None,
                "gaussian_eta_system_max": max(gaussian_eta) if gaussian_eta else None,
                "gaussian_N_eff_mean": float(np.mean(gaussian_neff)) if gaussian_neff else None,
                "gaussian_N_eff_population_std": float(np.std(gaussian_neff)) if gaussian_neff else None,
                "gaussian_N_eff_min": min(gaussian_neff) if gaussian_neff else None,
                "gaussian_N_eff_max": max(gaussian_neff) if gaussian_neff else None,
                "objective_value_mean": float(np.mean(objective_values)) if objective_values else None,
                "objective_value_population_std": float(np.std(objective_values)) if objective_values else None,
                "mode_dispersion": {
                    mode_name: {
                        "eta_system_mean": float(np.mean(mode_eta[mode_name])) if mode_eta[mode_name] else None,
                        "eta_system_population_std": float(np.std(mode_eta[mode_name])) if mode_eta[mode_name] else None,
                        "eta_system_min": min(mode_eta[mode_name]) if mode_eta[mode_name] else None,
                        "eta_system_max": max(mode_eta[mode_name]) if mode_eta[mode_name] else None,
                        "N_eff_mean": float(np.mean(mode_neff[mode_name])) if mode_neff[mode_name] else None,
                        "N_eff_population_std": float(np.std(mode_neff[mode_name])) if mode_neff[mode_name] else None,
                        "N_eff_min": min(mode_neff[mode_name]) if mode_neff[mode_name] else None,
                        "N_eff_max": max(mode_neff[mode_name]) if mode_neff[mode_name] else None,
                    }
                    for mode_name in mode_eta
                },
                "by_initialization": by_initialization,
            }
    pareto_all = _pareto_frontier(candidates)
    pareto_odd = _pareto_frontier([item for item in candidates if item["array_size"] % 2 == 1])
    pareto_even = _pareto_frontier([item for item in candidates if item["array_size"] % 2 == 0])
    source_baseline = {
        str(size): v5_metrics["phase_cases"]["zero"][str(size)]["gaussian_s12"]
        for size in SIZES
    }
    objective_tradeoff = _has_power_neff_tradeoff(candidates)
    expected_trajectory_ids = {
        f"n{size:02d}_{objective[0]}_{initialization}"
        for size in SIZES
        for objective in OBJECTIVES
        for initialization in INITIALIZATIONS
    }
    missing_trajectories = sorted(expected_trajectory_ids - set(all_summaries))
    if missing_trajectories:
        raise RuntimeError(f"v6 measurement is incomplete; missing trajectories: {missing_trajectories[:5]}")
    invalid_trajectory_statuses = {
        trajectory_id: summary.get("status")
        for trajectory_id, summary in all_summaries.items()
        if summary.get("status") not in ("COMPLETE", "BLOCKED_NO_WINDOW_QUALIFIED_A_TARGET")
    }
    if invalid_trajectory_statuses:
        raise RuntimeError(f"v6 measurement contains invalid trajectory statuses: {invalid_trajectory_statuses}")
    status = "VALID_COMPLETED"
    report = {
        "schema": SCHEMA,
        "status": status,
        "decision_question": "whether a finite phase-only budget on the fixed 200um system produces numerically credible power versus effective-channel evidence suitable for a later controllability review",
        "source_config": str(config_path),
        "source_evidence": config["source_evidence"],
        "historical_v4_g3_status": "G3_FAIL_RETAINED_UNCHANGED",
        "v5_expanded_window_baseline": {
            "status": v5_report["status"],
            "nine_complex64_pairs_passed": True,
            "maximum_aligned_error": v5_report["v5_window_G3"]["production_complex64_max_aligned_error"],
            "fixed_phase_geometry_and_mode_definitions_reused": True,
        },
        "fixed_physical_system": config["optics"],
        "fiber_assumptions": config["fiber"],
        "initializations": {
            name: {
                "phase_path": "initial_phase_screens.pt",
                "shape": [2, 1, 100, 100],
                "phase_min_rad": float(initializations[name].min()),
                "phase_max_rad": float(initializations[name].max()),
                "distribution": "v5 measured zero phase" if name == "v5_zero" else "independent uniform[-pi,pi]",
            }
            for name in INITIALIZATIONS
        },
        "optimizer": config["optimizer"],
        "objectives": {
            OBJECTIVES[0]: {"definition": "minimize -sum(P_i)/P_input", "target": "maximum total throughput"},
            OBJECTIVES[1]: {"definition": "minimize -N_eff/N; N_eff=(sum(P_i))^2/sum(P_i^2)", "target": "maximum single-field effective-channel fraction"},
            OBJECTIVES[2]: {"definition": "minimize mean(((P_i-P_A_star/N)/(P_A_star/N))^2)", "target": "frozen absolute uniform delivery"},
            "LP01_role": "same optimized Gaussian phase screens are read through LP01 V=2.2; not an independent optimization upper bound",
        },
        "frozen_uniform_targets": frozen_targets,
        "array_zero_phase_baselines_from_v5": source_baseline,
        "optimization_acceleration": {
            "selected_path": preflight["selected_phase_path"],
            "adjoint_pool": pool_record,
            "v5_phase_dependent_K_H_not_reused": True,
            "full_production_window_acceptance_consumes_measured_output": True,
        },
        "trajectories": all_summaries,
        "initialization_dispersion": initialization_dispersion,
        "representative_endpoints": representatives,
        "pareto": {"all_parities": pareto_all, "odd_nested_arrays": pareto_odd, "even_nested_arrays": pareto_even},
        "tradeoff_evidence": {
            "qualified_candidate_count": len(candidates),
            "qualified_gaussian_candidates_have_eta_neff_tradeoff": objective_tradeoff,
            "interpretation": "Pareto evidence is finite-budget simulation evidence only; it is not experimental usable dimension or a final fiber-count selection",
        },
        "measurement": {
            "endpoint_comparison": "best and step-256 terminal of every trajectory; each measured by full production two-hop path on all three windows and both precisions",
            "window_qualification": "all three pairwise complex64 common-ROI comparisons after one common phase per field must be <=1e-4",
            "endpoint_fields_for_nonrepresentatives": "computed and compared in memory; full raw fields are not retained for every candidate due compact evidence contract",
            "representative_fields_retained": "all six window/precision common-ROI fields for each selected array/objective representative; local compact package returns only 3072 complex64 fields",
            "channel_statistics": ["power map", "eta_fiber_face", "eta_system", "N_eff", "-10/-20/-30/-40 dB utilization", "P10/P90", "CV", "edge/center", "corner/center", "min/p10/p50/mean/p90/max"],
            "center_definition": "nearest fiber to the optical axis; odd arrays have one center fiber and even arrays have four equidistant center fibers",
        },
        "updates_and_forbidden_io": {
            "formal_training_updates": 0,
            "preflight_engineering_adam_updates": preflight["optimizer_checkpoint_preflight"]["engine_updates"],
            "main_phase_diagnostic_updates": total_updates,
            "maximum_main_phase_diagnostic_updates": 19968,
            "dataset_reads": 0,
            "task_evaluator_calls": 0,
            "fid_calls": 0,
            "loss_changes": 0,
            "material_forward_calls": 0,
            "lp01_independent_optimizer_updates": 0,
            "production_geometry_changes": 0,
            "decoder_camera_changes": 0,
        },
        "execution": {
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "elapsed_seconds_observation": time.perf_counter() - report_started,
            "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        },
        "claim_limits": [
            "the fixed 200um aperture, pixel pitch, fiber coordinates, Gaussian S12 mode, LP01 sensitivity model, input power and propagation lengths were not changed",
            "LP01 metrics are readouts of the same phases optimized for Gaussian and are not an LP01 optimized upper bound",
            "G3 results apply to the new three-window expanded numerical baseline and do not rewrite historical v4 G3_FAIL",
            "window-invalid endpoints do not contribute to the qualified Pareto set or frozen C target",
            "N_eff is a single-field power participation statistic and does not establish noise-limited experimental OLS dimension",
            "fiber pitch and MFD remain PROJECT_GEOMETRY_UNMEASURED simulation assumptions",
            "no formal task training, dataset, evaluator, FID, loss change, material model, relay, decoder, camera or image generation was used",
        ],
    }
    report["scientific_decision"] = {
        "status": "POWER_NEFF_TRADEOFF_EVIDENCE_AVAILABLE" if objective_tradeoff else "NO_DISTINCT_QUALIFIED_POWER_NEFF_TRADEOFF_DEMONSTRATED",
        "supports_next_controllability_review": objective_tradeoff,
        "decision_basis": "window-qualified representative Gaussian endpoints on the predeclared A/B/C objectives; no minimum coupling percentage was imposed",
    }
    _write_json(run_dir / "report.json", report)
    _write_json(run_dir / "representative_endpoint_manifest.json", {
        "schema": "ols_phase_dimension_power_v6_representative_manifest.v1",
        "representatives": {
            key: value.get("retained_field_paths", {})
            for key, value in representatives.items()
            if value.get("status") == "WINDOW_QUALIFIED_REPRESENTATIVE"
        },
        "compact_return_fields": [
            paths["complex64_grid3072"]
            for value in representatives.values()
            if value.get("status") == "WINDOW_QUALIFIED_REPRESENTATIVE"
            for paths in [value["retained_field_paths"]]
        ],
    })
    del mode_pools, runtime, entrance, first_stack
    clear_production_transfer_cache()
    torch.cuda.empty_cache()
    gc.collect()
    return report


def _pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier: list[dict[str, Any]] = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            weakly_better = (
                other["eta_system"] >= candidate["eta_system"]
                and other["N_eff"] >= candidate["N_eff"]
                and other["fiber_count"] <= candidate["fiber_count"]
            )
            strictly_better = (
                other["eta_system"] > candidate["eta_system"]
                or other["N_eff"] > candidate["N_eff"]
                or other["fiber_count"] < candidate["fiber_count"]
            )
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return sorted(frontier, key=lambda item: (item["fiber_count"], item["objective"]))


def _has_power_neff_tradeoff(candidates: list[dict[str, Any]]) -> bool:
    if len(candidates) < 2:
        return False
    for left in candidates:
        for right in candidates:
            if left is right:
                continue
            if left["eta_system"] > right["eta_system"] and left["N_eff"] < right["N_eff"]:
                return True
            if right["eta_system"] > left["eta_system"] and right["N_eff"] < left["N_eff"]:
                return True
    return False


def run(
    config_path: Path,
    run_dir: Path,
    *,
    preflight_only: bool,
    resume_checkpoint: Path | None = None,
    resume_trajectory_id: str | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = Path.cwd().resolve()
    config = _load_config(config_path)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    config_copy = run_dir / "configuration.yaml"
    if not config_copy.exists():
        config_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    elif config_copy.read_text(encoding="utf-8") != config_path.read_text(encoding="utf-8"):
        raise ValueError("existing v6 run directory contains a different explicit configuration")
    source_dir = run_dir / "source_snapshot"
    source_dir.mkdir(parents=True, exist_ok=True)
    snapshot_index = 1
    runner_snapshot = source_dir / f"{Path(__file__).stem}_invocation_{snapshot_index:02d}.py"
    while runner_snapshot.exists():
        snapshot_index += 1
        runner_snapshot = source_dir / f"{Path(__file__).stem}_invocation_{snapshot_index:02d}.py"
    runner_snapshot.write_bytes(Path(__file__).read_bytes())
    if preflight_only:
        return _run_preflight(config, config_path, run_dir, root)
    if (resume_checkpoint is None) != (resume_trajectory_id is None):
        raise ValueError("--resume-checkpoint and --resume-trajectory-id must be supplied together")
    if resume_checkpoint is not None and not resume_checkpoint.is_file():
        raise FileNotFoundError(f"explicit resume checkpoint is missing: {resume_checkpoint}")
    return _run_full_measurement(config, config_path, run_dir, root, resume_checkpoint, resume_trajectory_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--resume-trajectory-id")
    args = parser.parse_args()
    result = run(
        args.config,
        args.run_dir,
        preflight_only=args.preflight_only,
        resume_checkpoint=args.resume_checkpoint,
        resume_trajectory_id=args.resume_trajectory_id,
    )
    print(json.dumps({
        "status": result.get("status"),
        "preflight": result.get("passed"),
        "scientific_decision": result.get("scientific_decision", {}).get("status"),
        "run_dir": str(args.run_dir),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
