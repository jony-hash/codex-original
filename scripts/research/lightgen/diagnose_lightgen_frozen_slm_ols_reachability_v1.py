"""Bounded amplitude-command reachability diagnostic for frozen small G1/G4."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    GAUSSIAN_OLS_MODEL,
)
from scripts.research.lightgen import execute_lightgen_matched_architecture_v1 as base
from scripts.research.lightgen import execute_lightgen_small_g1_g4_native_readability_v1 as readability


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_FROZEN_SLM_OLS_INPUT_REACHABILITY_V1"
METHODS = ("G1", "G4")
CLASS_COUNT = 10
IMAGE_SHAPE = (200, 200)
COEFFICIENTS = 100
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/frozen_slm_ols_input_reachability_v1.yaml"


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"configuration is not a mapping: {path}", ValueError)
    return value


def _resolve(relative: str) -> Path:
    path = Path(relative)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected.reshape(-1))
    _require(torch.isfinite(denominator).item() and denominator.item() > 0.0,
             "relative-error reference has zero or nonfinite norm")
    return float(torch.linalg.vector_norm((actual - expected).reshape(-1)) / denominator)


def _phase_error(actual: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    """Return the best single global phase and its normalized complex-field error."""
    inner = torch.vdot(target.reshape(-1), actual.reshape(-1))
    phase = float(torch.angle(inner)) if inner.abs().item() else 0.0
    aligned = target * torch.exp(torch.as_tensor(1j * phase, dtype=target.dtype, device=target.device))
    return phase, _rel_l2(actual, aligned)


def _projected_spectral_norm(matrix: torch.Tensor) -> float:
    real = torch.cat((matrix.real, matrix.imag), dim=0)
    gram = real @ real.T
    largest = float(torch.linalg.eigvalsh(gram)[-1])
    _require(math.isfinite(largest) and largest > 0.0, "response matrix has no finite positive spectral norm")
    return largest


def _bounded_phase_solve(
    matrix: torch.Tensor,
    target: torch.Tensor,
    *,
    starts: int = 8,
    max_steps: int = 5000,
    stop_error: float = 1e-6,
    lipschitz: float | None = None,
    checkpoint: Callable[[dict[str, Any]], None] | None = None,
    initial_state: dict[str, Any] | None = None,
    checkpoint_interval: int = 500,
) -> dict[str, Any]:
    """Minimize ||B a - exp(i phi)z|| with a box projection and one phase."""
    _require(matrix.ndim == 2 and matrix.shape[0] == target.numel(), "response dimensions do not match target")
    _require(torch.is_complex(matrix) and torch.is_complex(target), "response and target must be complex")
    _require(matrix.dtype == torch.complex128 and target.dtype == torch.complex128,
             "bounded solver requires complex128 response and target")
    _require(starts == 8 and max_steps == 5000, "the authorized fixed solver budget changed", ValueError)
    device = matrix.device
    pixels = matrix.shape[1]
    lipschitz = _projected_spectral_norm(matrix) if lipschitz is None else float(lipschitz)
    _require(math.isfinite(lipschitz) and lipschitz > 0.0,
             "bounded solver requires a finite positive cached spectral constant")
    target_norm = torch.linalg.vector_norm(target)
    _require(torch.isfinite(target_norm).item() and target_norm.item() > 0.0,
             "target coefficient vector has zero or nonfinite norm")
    if initial_state is None:
        amplitudes = torch.full((starts, pixels), 0.5, dtype=torch.float64, device=device)
        phases = torch.arange(starts, dtype=torch.float64, device=device) * (2.0 * math.pi / starts)
        best_amplitudes = amplitudes.clone()
        best_errors = torch.full((starts,), float("inf"), dtype=torch.float64, device=device)
        first_iteration = 0
    else:
        amplitudes = initial_state["amplitudes"].to(device=device, dtype=torch.float64)
        phases = initial_state["phases"].to(device=device, dtype=torch.float64)
        best_amplitudes = initial_state["best_amplitudes"].to(device=device, dtype=torch.float64)
        best_errors = initial_state["best_errors"].to(device=device, dtype=torch.float64)
        best_phases = initial_state["best_phases"].to(device=device, dtype=torch.float64)
        first_iteration = int(initial_state["iteration"])
        _require(tuple(amplitudes.shape) == (starts, pixels)
                 and tuple(best_amplitudes.shape) == (starts, pixels)
                 and tuple(phases.shape) == (starts,) and tuple(best_phases.shape) == (starts,)
                 and tuple(best_errors.shape) == (starts,)
                 and 0 < first_iteration <= max_steps,
                 "saved solver cursor or command shape is invalid")
        _require(torch.isfinite(amplitudes).all().item() and torch.isfinite(best_amplitudes).all().item()
                 and torch.isfinite(phases).all().item() and torch.isfinite(best_phases).all().item()
                 and torch.isfinite(best_errors).all().item()
                 and float(amplitudes.min()) >= 0.0 and float(amplitudes.max()) <= 1.0
                 and float(best_amplitudes.min()) >= 0.0 and float(best_amplitudes.max()) <= 1.0
                 and float(best_errors.min()) >= 0.0,
                 "saved solver checkpoint contains nonfinite or out-of-range state")
    if initial_state is None:
        best_phases = phases.clone()
    iterations = first_iteration
    target_batch = target.reshape(1, -1)
    while iterations < max_steps:
        predicted = amplitudes.to(torch.complex128) @ matrix.T
        aligned_targets = target_batch * torch.exp(1j * phases).reshape(-1, 1)
        residual = predicted - aligned_targets
        gradients = (residual @ matrix.conj()).real
        amplitudes = torch.clamp(amplitudes - gradients / lipschitz, 0.0, 1.0)
        predicted = amplitudes.to(torch.complex128) @ matrix.T
        correlations = (target_batch.conj() * predicted).sum(dim=1)
        phases = torch.angle(correlations)
        aligned_targets = target_batch * torch.exp(1j * phases).reshape(-1, 1)
        errors = torch.linalg.vector_norm(predicted - aligned_targets, dim=1) / target_norm
        improved = errors < best_errors
        best_errors = torch.where(improved, errors, best_errors)
        best_amplitudes = torch.where(improved[:, None], amplitudes, best_amplitudes)
        best_phases = torch.where(improved, phases, best_phases)
        iterations += 1
        if bool((best_errors <= stop_error).any().item()):
            break
        if checkpoint is not None and iterations % checkpoint_interval == 0:
            checkpoint({
                "amplitudes": amplitudes.detach().cpu(),
                "phases": phases.detach().cpu(),
                "best_amplitudes": best_amplitudes.detach().cpu(),
                "best_errors": best_errors.detach().cpu(),
                "best_phases": best_phases.detach().cpu(),
                "iteration": iterations,
            })
    selected = int(torch.argmin(best_errors).item())
    _require(torch.isfinite(best_errors[selected]).item(), "solver failed to produce a finite candidate")
    return {
        "amplitude": best_amplitudes[selected],
        "phase_rad": float(best_phases[selected]),
        "relative_error": float(best_errors[selected]),
        "iterations": iterations,
        "best_errors": best_errors.detach().cpu(),
    }


def _analytic_solver_controls() -> dict[str, float | bool]:
    matrix = torch.eye(2, dtype=torch.complex128)
    command = torch.tensor([0.23, 0.78], dtype=torch.float64)
    base_target = matrix @ command.to(torch.complex128)
    phase = 0.73
    positive_target = base_target * torch.exp(torch.tensor(1j * phase, dtype=torch.complex128))
    positive = _bounded_phase_solve(matrix, positive_target)
    positive_actual = matrix @ positive["amplitude"].to(torch.complex128)
    positive_error = _phase_error(positive_actual, positive_target)[1]

    impossible_target = torch.tensor([1.0 + 0.0j, 0.0 + 1.0j], dtype=torch.complex128)
    impossible = _bounded_phase_solve(matrix, impossible_target)
    impossible_actual = matrix @ impossible["amplitude"].to(torch.complex128)
    impossible_error = _phase_error(impossible_actual, impossible_target)[1]

    excessive_target = torch.tensor([1.5 + 0.0j, 0.0 + 0.0j], dtype=torch.complex128)
    excessive = _bounded_phase_solve(matrix, excessive_target)
    excessive_actual = matrix @ excessive["amplitude"].to(torch.complex128)
    excessive_error = _phase_error(excessive_actual, excessive_target)[1]
    bounded = bool((positive["amplitude"] >= 0.0).all() and (positive["amplitude"] <= 1.0).all()
                   and float(excessive["amplitude"].max()) <= 1.0)
    _require(positive_error <= 1e-6, "known command with one shared phase was not recovered")
    _require(impossible_error > 1e-4, "analytic two-port impossible phase control was accepted")
    _require(excessive_error > 1e-4 and bounded,
             "out-of-range target was normalized or accepted by the bounded solve")
    return {
        "known_command_global_phase_relative_error": positive_error,
        "impossible_two_port_relative_error": impossible_error,
        "amplitude_overflow_target_relative_error": excessive_error,
        "commands_bounded_0_1": bounded,
    }


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    _require(config.get("schema") == "lightgen_frozen_slm_ols_input_reachability.v1"
             and config.get("task_id") == TASK_ID and config.get("evidence_tier") == "diagnostic_only",
             "task configuration identity changed", ValueError)
    source = config["source"]
    _require(source.get("branch") == "C" and source.get("cumulative_step") == 45_000
             and source.get("model") == "C_seed_2026101501" and source.get("stream") == 0
             and source.get("methods") == list(METHODS) and source.get("samples_per_method") == 100
             and source.get("samples_per_class_per_method") == 10,
             "frozen model or selected population changed", ValueError)
    fixed_source_paths = {
        "readability_config": "configs/research/lightgen/small_g1_g4_native_readability_v1.yaml",
        "checkpoint": "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/runs/autodl_backup_attempt_01/C/checkpoint_step_010000.pt",
        "baseline_evidence_dir": "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/runs/autodl_backup_attempt_01",
        "replication_evidence_dir": "outputs/20261021_lightgen_small_gaussian_replication_prior_confirmation_v1/runs/autodl_backup_attempt_01",
        "selection_manifest": "outputs/20261022_lightgen_small_g1_g4_native_readability_v1/runs/autodl_backup_attempt_01/selection_manifest.json",
        "selected_replay": "outputs/20261022_lightgen_small_g1_g4_native_readability_v1/runs/autodl_backup_attempt_01/selected_replay.pt",
    }
    _require(all(source.get(key) == value for key, value in fixed_source_paths.items()),
             "explicit source or saved selection path changed", ValueError)
    optics = config["optics"]
    _require(optics.get("ols_basis") == GAUSSIAN_OLS_MODEL
             and optics.get("propagation_grids") == [2048, 4096]
             and optics.get("encoder_layers") == 2 and optics.get("decoder_layers") == 2
             and optics.get("phase_shape") == [100, 100] and optics.get("input_shape") == [200, 200]
             and math.isclose(float(optics.get("source_power_w", 0.0)), 2.493e-4, rel_tol=0.0, abs_tol=1e-12)
             and optics.get("global_gain") == "frozen_C_checkpoint"
             and optics.get("runtime_config") == "configs/research/lightgen/small_gaussian_system_baseline_v1.yaml"
             and optics.get("geometry_status") == "PROJECT_GEOMETRY_UNMEASURED",
             "fixed optics, gain or geometry annotation changed", ValueError)
    solver = config["solver"]
    _require(solver.get("dtype") == "float64" and solver.get("phase_starts") == 8
             and solver.get("max_steps_per_start") == 5000
             and solver.get("coefficient_stop_relative_error") == 1e-6
             and solver.get("witness_coefficient_relative_error") == 1e-4
             and solver.get("witness_cmos_intensity_relative_error") == 1e-4
             and solver.get("checkpoint_interval_steps") == 500,
             "approved bounded solve definition changed", ValueError)
    preflight = config["preflight"]
    _require(preflight.get("expected_rows") == 20
             and preflight.get("matrix_prediction_relative_error_max") == 1e-6
             and preflight.get("direction_derivative_relative_error_max") == 1e-5
             and preflight.get("reference_replay_rtol") == 2e-6
             and preflight.get("reference_replay_atol") == 5e-8
             and preflight.get("cross_grid_relative_error_max") == 1e-4
             and all(preflight.get(field) == 0 for field in (
                 "model_updates", "adam_calls", "prior_fit", "new_sampling", "new_human_review"
             )),
             "fixed preflight thresholds or zero-update boundary changed", ValueError)
    expected_output = "outputs/20261025_lightgen_frozen_slm_ols_reachability_v1/runs/attempt_01"
    _require(config["output"].get("experiment_root") == expected_output
             and config["output"].get("final_decision_consumer") ==
             "rerun_saved_instructions_and_recompute_all_errors",
             "fixed output path or final decision consumer changed", ValueError)
    expected_config = (ROOT / "configs/research/lightgen/frozen_slm_ols_input_reachability_v1.yaml").resolve()
    _require(config_path.resolve() == expected_config,
             "diagnostic must use the fixed explicit config path", ValueError)


def _load_inputs(config: dict[str, Any]) -> dict[str, Any]:
    source = config["source"]
    checkpoint_path = _resolve(source["checkpoint"])
    baseline = _resolve(source["baseline_evidence_dir"])
    replication = _resolve(source["replication_evidence_dir"])
    selection_path = _resolve(source["selection_manifest"])
    replay_path = _resolve(source["selected_replay"])
    for path in (checkpoint_path, selection_path, replay_path):
        _require(path.is_file(), f"explicit fixed input is missing: {path}", FileNotFoundError)
    old_config, runtime_config = readability._read_config(_resolve(source["readability_config"]))
    lineage = readability._load_inputs(
        checkpoint_path, baseline, replication, old_config, runtime_config,
    )
    checkpoint = lineage["checkpoint"]
    _require(checkpoint.get("branch") == "C" and checkpoint.get("completed_global_step") == 45_000
             and checkpoint.get("seed") == 2026101501,
             "loaded checkpoint is not the fixed C seed at cumulative step 45,000")
    _require(tuple(checkpoint["model"]["runtime.encoder.phase"].shape) == (2, 1, 100, 100)
             and tuple(checkpoint["model"]["runtime.generator.phase"].shape) == (2, 1, 100, 100),
             "checkpoint does not contain the fixed two-layer Encoder and Decoder")
    optimizer_state = checkpoint.get("optimizer", {}).get("state", {})
    optimizer_steps = [int(torch.as_tensor(state["step"]).item()) for state in optimizer_state.values()
                       if isinstance(state, dict) and "step" in state]
    _require(len(optimizer_steps) == 2 and all(step == 45_000 for step in optimizer_steps),
             "frozen source checkpoint optimizer state does not match step 45,000")
    readability_config = _load_yaml(_resolve(source["readability_config"]))
    _require(readability_config.get("task_id") == "LIGHTGEN_SMALL_G1_G4_NATIVE_READABILITY_V1"
             and readability_config["source"].get("cumulative_step") == 45_000
             and readability_config["source"].get("branch") == "C"
             and readability_config["source"].get("checkpoint") == source["checkpoint"],
             "fixed native-readability configuration points at another source")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    _require(selection.get("schema") == "lightgen_small_g1_g4_native_readability_selection.v1"
             and selection.get("source_step") == 45_000 and selection.get("stream") == 0
             and selection.get("per_method_per_class") == 10
             and selection.get("task_id") == old_config["task_id"]
             and selection.get("source_run_id") == lineage["source_run_id"]
             and selection.get("classifier_brightness_or_appearance_used_for_selection") is False,
             "saved selection manifest identity or population changed")
    entries = selection.get("items")
    _require(isinstance(entries, list) and len(entries) == 200,
             "saved fixed blind-review manifest must contain exactly 200 selected rows")
    _require(all(isinstance(item, dict) for item in entries), "saved selection entries must be mappings")
    row_keys = {(item.get("method"), item.get("cohort_row")) for item in entries
                }
    proposal_keys = {
        (item.get("method"), item.get("requested_label"), item.get("proposal_id"))
        for item in entries
    }
    _require(len(row_keys) == 200 and len(proposal_keys) == 200
             and all(item.get("stream") == 0 and item.get("requested_label") in range(CLASS_COUNT)
                     for item in entries),
             "saved selection repeats an input row/proposal or changes stream/class")
    replay = torch.load(replay_path, map_location="cpu", weights_only=False)
    _require(replay.get("schema") == "lightgen_small_g1_g4_native_selected_replay.v1"
             and replay.get("task_id") == old_config["task_id"]
             and replay.get("run_id") == selection.get("run_id")
             and replay.get("source_step") == 45_000,
             "saved replay identity does not match the fixed selection manifest")
    cohorts: dict[str, dict[str, Any]] = {}
    replay_indices: dict[str, dict[int, int]] = {}
    for method in METHODS:
        cohort_path = replication / "generation" / source["model"] / f"{method}_stream_0.pt"
        _require(cohort_path.is_file(), f"saved explicit {method} cohort is missing: {cohort_path}", FileNotFoundError)
        cohort = lineage["cohorts"][method]
        _require(cohort.get("method") == method and cohort.get("model") == source["model"]
                 and cohort.get("stream") == 0 and cohort.get("global_step") == 45_000
                 and cohort.get("classifier_based_sample_selection") is False
                 and cohort.get("per_image_normalization") is False,
                 f"saved {method} cohort identity or selection rule changed")
        _require(tuple(cohort.get("ols_coefficients", torch.empty(0)).shape) == (10_000, 100)
                 and tuple(cohort.get("requested_labels", torch.empty(0)).shape) == (10_000,)
                 and tuple(cohort.get("proposal_ids", torch.empty(0)).shape) == (10_000,)
                 and tuple(cohort.get("component_ids", torch.empty(0)).shape) == (10_000,),
                 f"saved {method} coefficient cohort is incomplete")
        selected = [item for item in entries if item.get("method") == method]
        _require(len(selected) == 100, f"saved {method} manifest does not contain 100 samples")
        _require(torch.equal(torch.as_tensor(replay[method]["cohort_rows"]).long(),
                             torch.tensor([int(item["cohort_row"]) for item in selected])),
                 f"saved {method} replay rows no longer follow the frozen manifest order")
        _require(tuple(replay[method]["native200"].shape) == (100, 1, 200, 200)
                 and tuple(replay[method]["view28"].shape) == (100, 1, 28, 28),
                 f"saved {method} blind-review reference images are incomplete")
        _require(torch.isfinite(replay[method]["native200"]).all().item()
                 and torch.isfinite(replay[method]["view28"]).all().item()
                 and float(replay[method]["native200"].min()) >= 0.0
                 and float(replay[method]["native200"].max()) <= 1.0,
                 f"saved {method} blind-review reference has invalid image values")
        rows_by_class = {label: [item for item in selected if item.get("requested_label") == label]
                         for label in range(CLASS_COUNT)}
        _require(all(len(items) == 10 for items in rows_by_class.values()),
                 f"saved {method} class allocation changed")
        for item in selected:
            row = int(item["cohort_row"])
            label = int(item["requested_label"])
            _require(0 <= row < 10_000 and int(cohort["requested_labels"][row]) == label
                     and int(cohort["proposal_ids"][row]) == int(item["proposal_id"])
                     and int(cohort["component_ids"][row]) == int(item["component_id"]),
                     f"saved {method} manifest identity disagrees with its cohort row")
            _require(torch.isfinite(cohort["ols_coefficients"][row].real).all().item()
                     and torch.isfinite(cohort["ols_coefficients"][row].imag).all().item(),
                     f"saved {method} target coefficients are nonfinite")
        cohorts[method] = cohort
        replay_indices[method] = {int(item["cohort_row"]): index for index, item in enumerate(selected)}
    _require(all(sum(item["method"] == method for item in entries) == 100 for method in METHODS),
             "selected population method denominators changed")
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "selection": selection,
        "entries": entries,
        "cohorts": cohorts,
        "replay": replay,
        "replay_indices": replay_indices,
        "baseline_dir": baseline,
        "replication_dir": replication,
        "source_run_id": checkpoint.get("run_id"),
    }


def _record_input_authority(
    run_dir: Path, identity: dict[str, Any], config_path: Path,
    config: dict[str, Any], inputs: dict[str, Any],
) -> None:
    snapshot_path = run_dir / "run_config_snapshot.json"
    snapshot = {
        "config_path": config_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "config": config,
    }
    if snapshot_path.is_file():
        _require(json.loads(snapshot_path.read_text(encoding="utf-8")) == snapshot,
                 "saved run configuration differs from this invocation")
    else:
        _atomic_json(snapshot_path, snapshot)
    identity["source_run_id"] = inputs["source_run_id"]
    identity["inputs"] = {
        "config_path": config_path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "checkpoint": str(inputs["checkpoint_path"]),
        "selection_manifest": str(_resolve(config["source"]["selection_manifest"])),
        "selected_replay": str(_resolve(config["source"]["selected_replay"])),
        "baseline_evidence_dir": str(inputs["baseline_dir"]),
        "replication_evidence_dir": str(inputs["replication_dir"]),
        "fixed_methods": list(METHODS),
        "selected_rows": [
            {
                "method": item["method"], "cohort_row": int(item["cohort_row"]),
                "label": int(item["requested_label"]), "proposal_id": int(item["proposal_id"]),
                "component_id": int(item["component_id"]),
            }
            for item in inputs["entries"]
        ],
    }
    _atomic_json(run_dir / "input_authority.json", identity["inputs"])


def _prepare_output(run_dir: Path, *, allow_closed: bool) -> dict[str, Any]:
    expected = (ROOT / "outputs/20261025_lightgen_frozen_slm_ols_reachability_v1/runs/attempt_01").resolve()
    _require(run_dir.resolve() == expected, "output must use the fixed new diagnostic attempt path", ValueError)
    identity_path = run_dir / "run_identity.json"
    if run_dir.exists():
        _require(identity_path.is_file(), "fixed output directory exists without this diagnostic run identity")
        identity = json.loads(identity_path.read_text(encoding="utf-8"))
        _require(identity.get("task_id") == TASK_ID and
                 (identity.get("status") == "ACTIVE" or (allow_closed and identity.get("status") == "CLOSED")),
                 "fixed output directory belongs to another run or is already closed")
        return identity
    run_dir.mkdir(parents=True)
    identity = {
        "schema": "lightgen_frozen_slm_ols_input_reachability_run.v1",
        "task_id": TASK_ID,
        "run_id": uuid.uuid4().hex,
        "status": "ACTIVE",
        "checkpoint_step": 45_000,
        "methods": list(METHODS),
        "sample_count": 200,
        "model_updates": 0,
        "adam_calls": 0,
        "prior_fit": 0,
        "new_sampling": 0,
        "new_human_review": 0,
    }
    _atomic_json(identity_path, identity)
    return identity


def _new_model(checkpoint: dict[str, Any], grid: int, device: torch.device) -> base.SmallMatchedAdapter:
    model = base.SmallMatchedAdapter(grid, GAUSSIAN_OLS_MODEL)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys,
             "fixed C model state was not loaded exactly")
    model.to(device=device)
    for parameter in model.parameters():
        _require(parameter.is_floating_point(), "frozen optical parameters must be real-valued phases")
        if parameter.dtype != torch.float64:
            parameter.data = parameter.detach().to(dtype=torch.float64)
    _require(all(not buffer.is_floating_point() or buffer.dtype == torch.float64
                 for buffer in model.buffers()),
             "real optical buffers must retain float64 precision")
    _require(all(not torch.is_complex(buffer) or buffer.dtype == torch.complex128
                 for buffer in model.buffers()),
             "complex optical buffers must retain complex128 precision")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _require(all(not parameter.requires_grad for parameter in model.parameters()),
             "frozen optical model contains a trainable parameter")
    return model


def _new_reference_model(checkpoint: dict[str, Any], device: torch.device) -> base.SmallMatchedAdapter:
    """Load the fixed decoder at its saved parameter precision for exact legacy replay."""
    model = base.SmallMatchedAdapter(2048, GAUSSIAN_OLS_MODEL)
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys,
             "fixed C reference model state was not loaded exactly")
    model.to(device=device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _require(all(not parameter.requires_grad for parameter in model.parameters()),
             "frozen reference model contains a trainable parameter")
    _require(torch.is_complex(model.runtime.ols.operator.transport)
             and model.runtime.ols.operator.transport.dtype == torch.complex128,
             "reference model did not preserve the complex Gaussian transport")
    return model


def _make_response(
    model: base.SmallMatchedAdapter,
    device: torch.device,
    progress: Callable[[dict[str, Any]], None],
) -> torch.Tensor:
    """Differentiate the production image entry, including sqrt(a²) and the full Encoder path."""
    command = torch.full((1, 1, *IMAGE_SHAPE), 0.5, dtype=torch.float64, device=device,
                         requires_grad=True)
    result = model.forward_from_image(command.square())
    coefficients = result["ols_coefficients"].reshape(-1)
    _require(coefficients.numel() == COEFFICIENTS and coefficients.dtype == torch.complex128,
             "production OLS response did not retain complex128 precision")
    rows: list[torch.Tensor] = []
    for index in range(COEFFICIENTS):
        real_gradient = torch.autograd.grad(coefficients[index].real, command, retain_graph=True)[0]
        imaginary_gradient = torch.autograd.grad(
            coefficients[index].imag, command, retain_graph=index != COEFFICIENTS - 1
        )[0]
        rows.append(torch.complex(real_gradient.reshape(-1), imaginary_gradient.reshape(-1)))
        if (index + 1) % 10 == 0:
            progress({"phase": "response_autograd", "completed_scalar_outputs": 2 * (index + 1),
                      "total_scalar_outputs": 2 * COEFFICIENTS, "model_updates": 0})
    matrix = torch.stack(rows)
    _require(tuple(matrix.shape) == (COEFFICIENTS, IMAGE_SHAPE[0] * IMAGE_SHAPE[1])
             and torch.isfinite(matrix.real).all().item() and torch.isfinite(matrix.imag).all().item(),
             "autograd response matrix has wrong shape or nonfinite entries")
    return matrix.detach()


def _matrix_and_derivative_checks(
    model: base.SmallMatchedAdapter, matrix: torch.Tensor, device: torch.device,
) -> dict[str, float]:
    generator = torch.Generator(device="cpu").manual_seed(20261025)
    known = (torch.rand(IMAGE_SHAPE, generator=generator, dtype=torch.float64) * 0.8 + 0.1).to(device)
    with torch.no_grad():
        direct = model.forward_from_image(known.square().reshape(1, 1, *IMAGE_SHAPE))[
            "ols_coefficients"
        ].reshape(-1)
    predicted = matrix @ known.reshape(-1).to(torch.complex128)
    matrix_error = _rel_l2(predicted, direct)

    center = torch.full(IMAGE_SHAPE, 0.5, dtype=torch.float64, device=device)
    direction = torch.randint(0, 2, IMAGE_SHAPE, generator=generator, dtype=torch.int64).to(
        device=device, dtype=torch.float64
    ) * 2.0 - 1.0
    epsilon = 2e-4
    with torch.no_grad():
        plus = model.forward_from_image((center + epsilon * direction).square().reshape(1, 1, *IMAGE_SHAPE))[
            "ols_coefficients"
        ].reshape(-1)
        minus = model.forward_from_image((center - epsilon * direction).square().reshape(1, 1, *IMAGE_SHAPE))[
            "ols_coefficients"
        ].reshape(-1)
    numeric = (plus - minus) / (2.0 * epsilon)
    analytic = matrix @ direction.reshape(-1).to(torch.complex128)
    derivative_error = _rel_l2(analytic, numeric)
    _require(matrix_error <= 1e-6, f"response matrix prediction error {matrix_error:.3g} exceeds 1e-6")
    _require(derivative_error <= 1e-5,
             f"complex128 directional derivative error {derivative_error:.3g} exceeds 1e-5")
    return {"matrix_prediction_relative_error": matrix_error,
            "direction_derivative_relative_error": derivative_error}


def _item_key(item: dict[str, Any]) -> str:
    return f"{item['method']}_{int(item['cohort_row']):05d}"


def _target_for(inputs: dict[str, Any], item: dict[str, Any]) -> torch.Tensor:
    return inputs["cohorts"][item["method"]]["ols_coefficients"][int(item["cohort_row"])].to(
        dtype=torch.complex128
    )


def _original_target_for(inputs: dict[str, Any], item: dict[str, Any]) -> torch.Tensor:
    """Return the saved coefficient row without changing its historical dtype."""
    return inputs["cohorts"][item["method"]]["ols_coefficients"][int(item["cohort_row"])]


def _camera_pixel_area(model: base.SmallMatchedAdapter) -> float:
    pitch_x, pitch_y = model.runtime.camera_plane.pixel_pitch_m
    return float(pitch_x * pitch_y)


def _sample_call(
    model: base.SmallMatchedAdapter,
    amplitude: torch.Tensor,
    target: torch.Tensor,
    gain: float,
) -> dict[str, torch.Tensor | float]:
    amplitude = amplitude.to(device=next(model.parameters()).device, dtype=torch.float64)
    with torch.no_grad():
        command_result = model.forward_from_image(amplitude.square().reshape(1, 1, *IMAGE_SHAPE))
        direct_result = model.forward_from_ols(target.reshape(1, -1))
    actual_coefficients = command_result["ols_coefficients"].reshape(-1)
    direct_coefficients = target.reshape(-1)
    actual_raw = command_result["raw_intensity_w_per_m2"][0, 0]
    direct_raw = direct_result["raw_intensity_w_per_m2"][0, 0]
    actual_w_per_pixel = command_result["physical_w_per_pixel"][0, 0]
    direct_w_per_pixel = direct_result["physical_w_per_pixel"][0, 0]
    actual_view = torch.nn.functional.adaptive_avg_pool2d(
        (actual_w_per_pixel * gain).clamp(0.0, 1.0).reshape(1, 1, *IMAGE_SHAPE), (28, 28)
    )[0]
    direct_view = torch.nn.functional.adaptive_avg_pool2d(
        (direct_w_per_pixel * gain).clamp(0.0, 1.0).reshape(1, 1, *IMAGE_SHAPE), (28, 28)
    )[0]
    phase, coefficient_error = _phase_error(actual_coefficients, direct_coefficients)
    raw_error = _rel_l2(actual_raw, direct_raw)
    return {
        "actual_coefficients": actual_coefficients.detach().cpu(),
        "direct_coefficients": direct_coefficients.detach().cpu(),
        "actual_raw_intensity_w_per_m2": actual_raw.detach().cpu(),
        "direct_raw_intensity_w_per_m2": direct_raw.detach().cpu(),
        "actual_camera_power_w": float(actual_w_per_pixel.sum()),
        "direct_camera_power_w": float(direct_w_per_pixel.sum()),
        "camera_pixel_area_m2": _camera_pixel_area(model),
        "plane_powers_w": command_result["plane_powers_w"][0].detach().cpu(),
        "phase_rad": phase,
        "coefficient_relative_error": coefficient_error,
        "raw_cmos_intensity_relative_error": raw_error,
        "actual_view28": actual_view.detach().cpu(),
        "direct_view28": direct_view.detach().cpu(),
    }


def _save_solver_state(run_dir: Path, item: dict[str, Any], state: dict[str, Any]) -> None:
    _atomic_torch(run_dir / "solver_state.pt", {"task_id": TASK_ID, "sample_key": _item_key(item), **state})


def _solve_item(
    matrix: torch.Tensor, target: torch.Tensor, item: dict[str, Any], run_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    state_path = run_dir / "solver_state.pt"
    state: dict[str, Any] | None = None
    if state_path.is_file():
        saved = torch.load(state_path, map_location="cpu", weights_only=False)
        _require(saved.get("task_id") == TASK_ID, "saved solver cursor belongs to another diagnostic")
        _require(saved.get("sample_key") == _item_key(item),
                 "saved solver cursor does not match the next frozen sample")
        state = saved
    solver = config["solver"]
    return _bounded_phase_solve(
        matrix,
        target,
        starts=int(solver["phase_starts"]),
        max_steps=int(solver["max_steps_per_start"]),
        stop_error=float(solver["coefficient_stop_relative_error"]),
        checkpoint=lambda value: _save_solver_state(run_dir, item, value),
        initial_state=state,
        checkpoint_interval=int(solver["checkpoint_interval_steps"]),
    )


def _run_item(
    item: dict[str, Any], inputs: dict[str, Any], matrix: torch.Tensor,
    models: dict[int, base.SmallMatchedAdapter], run_dir: Path,
    config: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    sample_path = run_dir / "samples" / f"{_item_key(item)}.pt"
    if sample_path.is_file():
        saved = torch.load(sample_path, map_location="cpu", weights_only=False)
        _require(saved.get("task_id") == TASK_ID and saved.get("sample_key") == _item_key(item),
                 f"existing sample evidence does not match the frozen row: {_item_key(item)}")
        state_path = run_dir / "solver_state.pt"
        if state_path.is_file():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            if state.get("sample_key") == _item_key(item):
                state_path.unlink()
        return saved
    target_cpu = _target_for(inputs, item)
    target = target_cpu.to(device)
    solver = _solve_item(matrix, target, item, run_dir, config)
    amplitude = solver["amplitude"].reshape(IMAGE_SHAPE).to(device=device, dtype=torch.float64)
    gain = float(inputs["checkpoint"]["camera_gain"])
    by_grid: dict[str, Any] = {}
    for grid in (2048, 4096):
        measured = _sample_call(models[grid], amplitude, target, gain)
        if grid == 2048:
            predicted = matrix @ amplitude.reshape(-1).to(torch.complex128)
            measured["response_matrix_prediction_relative_error"] = _rel_l2(
                predicted, measured["actual_coefficients"].to(device)
            )
        else:
            measured["response_matrix_prediction_relative_error"] = None
        by_grid[str(grid)] = measured
    result2048 = by_grid["2048"]
    result4096 = by_grid["4096"]
    cross_grid_coeff = _rel_l2(result2048["actual_coefficients"], result4096["actual_coefficients"])
    cross_grid_raw = _rel_l2(result2048["actual_raw_intensity_w_per_m2"],
                             result4096["actual_raw_intensity_w_per_m2"])
    coefficient_limit = float(config["solver"]["witness_coefficient_relative_error"])
    camera_limit = float(config["solver"]["witness_cmos_intensity_relative_error"])
    command = solver["amplitude"].detach().cpu().double().reshape(IMAGE_SHAPE)
    bounded = bool(torch.isfinite(command).all().item() and command.min() >= 0.0 and command.max() <= 1.0)
    witness = bool(
        bounded
        and result2048["coefficient_relative_error"] <= coefficient_limit
        and result4096["coefficient_relative_error"] <= coefficient_limit
        and result2048["raw_cmos_intensity_relative_error"] <= camera_limit
        and result4096["raw_cmos_intensity_relative_error"] <= camera_limit
        and cross_grid_coeff <= float(config["preflight"]["cross_grid_relative_error_max"])
        and cross_grid_raw <= float(config["preflight"]["cross_grid_relative_error_max"])
    )
    replay_index = inputs["replay_indices"][item["method"]][int(item["cohort_row"])]
    payload = {
        "schema": "lightgen_frozen_slm_ols_reachability_sample.v1",
        "task_id": TASK_ID,
        "sample_key": _item_key(item),
        "method": item["method"],
        "label": int(item["requested_label"]),
        "stream": 0,
        "cohort_row": int(item["cohort_row"]),
        "proposal_id": int(item["proposal_id"]),
        "component_id": int(item["component_id"]),
        "target_coefficients": target_cpu.to(torch.complex64),
        "amplitude_command": command,
        "intensity_command": command.square(),
        "solver": {
            "method": "bounded_projected_gradient_global_phase",
            "starts": 8,
            "max_steps_per_start": 5000,
            "iterations_executed": int(solver["iterations"]),
            "best_matrix_relative_error": float(solver["relative_error"]),
            "phase_rad": float(solver["phase_rad"]),
            "stop_error": 1e-6,
        },
        "legacy_native200": inputs["replay"][item["method"]]["native200"][replay_index, 0].float(),
        "legacy_view28": inputs["replay"][item["method"]]["view28"][replay_index].float(),
        "grids": {
            grid: {
                field: (value.float() if isinstance(value, torch.Tensor) and not torch.is_complex(value)
                        else value.to(torch.complex64) if isinstance(value, torch.Tensor) and torch.is_complex(value)
                        else value)
                for field, value in measured.items()
            }
            for grid, measured in by_grid.items()
        },
        "cross_grid_coefficient_relative_error": cross_grid_coeff,
        "cross_grid_cmos_intensity_relative_error": cross_grid_raw,
        "command_bounded": bounded,
        "numeric_reproduction_witness": witness,
        "interpretation_on_failure": "no witness found within the fixed solver budget; not proof of physical unreachability",
    }
    sample_dir = run_dir / "samples"
    sample_dir.mkdir(exist_ok=True)
    _atomic_torch(sample_path, payload)
    state_path = run_dir / "solver_state.pt"
    if state_path.is_file():
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if state.get("sample_key") == _item_key(item):
            state_path.unlink()
    return payload


def _preflight_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    first_by_label: dict[tuple[str, int], dict[str, Any]] = {}
    for item in entries:
        first_by_label.setdefault((str(item["method"]), int(item["requested_label"])), item)
    rows = [first_by_label[(method, label)] for method in METHODS for label in range(CLASS_COUNT)]
    _require(len(rows) == 20, "frozen preflight sample population changed")
    return rows


def _snapshot_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _assert_state_unchanged(model: torch.nn.Module, before: dict[str, torch.Tensor]) -> None:
    after = model.state_dict()
    _require(before.keys() == after.keys() and all(torch.equal(before[key], after[key].detach().cpu())
                                                   for key in before),
             "frozen model state changed during the diagnostic")
    _require(all(not parameter.requires_grad for parameter in model.parameters()),
             "model parameters were not all frozen")


def _reference_replay_all(
    model: base.SmallMatchedAdapter,
    inputs: dict[str, Any],
    gain: float,
    run_dir: Path,
    *,
    result_name: str = "reference_replay.json",
    run_id: str | None = None,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    replay_outputs: dict[str, dict[str, Any]] = {}
    area = _camera_pixel_area(model)
    before = _snapshot_state(model)
    with torch.no_grad():
        for method in METHODS:
            items = [item for item in inputs["entries"] if item["method"] == method]
            native_rows: list[torch.Tensor] = []
            view_rows: list[torch.Tensor] = []
            for offset in range(0, len(items), 10):
                chunk = items[offset:offset + 10]
                coefficients = torch.stack([_original_target_for(inputs, item) for item in chunk]).to(
                    device=next(model.parameters()).device
                )
                direct = model.forward_from_ols(coefficients)
                w_per_pixel = direct["physical_w_per_pixel"].float()
                gained = w_per_pixel * torch.as_tensor(
                    gain, dtype=torch.float32, device=w_per_pixel.device
                )
                native = gained.clamp(0.0, 1.0)
                view28 = readability.resize_view(gained)
                for local, item in enumerate(chunk):
                    replay_index = inputs["replay_indices"][method][int(item["cohort_row"])]
                    expected = inputs["replay"][method]["native200"][replay_index, 0].to(native.device)
                    actual = native[local]
                    passed = torch.allclose(actual.float(), expected.float(), rtol=2e-6, atol=5e-8)
                    error = _rel_l2(actual.double(), expected.double())
                    expected_view = inputs["replay"][method]["view28"][replay_index, 0].to(view28.device)
                    view_passed = torch.allclose(view28[local, 0].float(), expected_view.float(),
                                                 rtol=2e-6, atol=5e-8)
                    native_rows.append(actual.detach().cpu())
                    view_rows.append(view28[local].detach().cpu())
                    checks.append({
                        "sample_key": _item_key(item), "method": method,
                        "cohort_row": int(item["cohort_row"]),
                        "relative_error": error, "passed": bool(passed),
                        "view28_relative_error": _rel_l2(view28[local, 0].double(), expected_view.double()),
                        "view28_passed": bool(view_passed),
                    })
            replay_outputs[method] = {
                "cohort_rows": torch.tensor([int(item["cohort_row"]) for item in items]),
                "native200": torch.stack(native_rows),
                "view28": torch.stack(view_rows),
            }
    result = {
        "schema": "lightgen_frozen_slm_ols_reference_replay.v1",
        "task_id": TASK_ID,
        "rows": len(checks),
        "all_passed": len(checks) == 200 and all(row["passed"] and row["view28_passed"] for row in checks),
        "rtol": 2e-6,
        "atol": 5e-8,
        "max_relative_error": max(row["relative_error"] for row in checks),
        "max_view28_relative_error": max(row["view28_relative_error"] for row in checks),
        "camera_pixel_area_m2": area,
        "checks": checks,
    }
    _assert_state_unchanged(model, before)
    _atomic_torch(run_dir / Path(result_name).with_suffix(".pt").name, {
        "schema": "lightgen_frozen_slm_ols_reference_replay_outputs.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "methods": replay_outputs,
    })
    _atomic_json(run_dir / result_name, result)
    _require(result["rows"] == 200 and result["all_passed"],
             "original 200-row native or 28x28 replay failed; stop before the optical solve")
    return result


def _write_comparison_atlas(run_dir: Path, inputs: dict[str, Any], samples: list[dict[str, Any]]) -> None:
    tile = 28
    gap = 5
    title_height = 13
    label_width = 35
    cell_width = label_width + 10 * (3 * tile + 3 * gap)
    image = Image.new("RGB", (2 * cell_width + 3 * gap, 10 * (10 * tile + title_height + 2 * gap) + 3 * gap), "white")
    draw = ImageDraw.Draw(image)
    sample_map = {(sample["method"], sample["label"], sample["cohort_row"]): sample for sample in samples}
    for method_index, method in enumerate(METHODS):
        for label in range(CLASS_COUNT):
            x0 = gap + method_index * (cell_width + gap)
            y0 = gap + label * (10 * tile + title_height + 2 * gap)
            draw.text((x0, y0), f"{method} class {label}: reference | direct | SLM", fill="black")
            rows = [item for item in inputs["entries"]
                    if item["method"] == method and int(item["requested_label"]) == label]
            for position, item in enumerate(rows):
                sample = sample_map[(method, label, int(item["cohort_row"]))]
                x = x0 + label_width
                y = y0 + title_height + position * tile
                images = [
                    sample["legacy_view28"][0].numpy(),
                    sample["grids"]["2048"]["direct_view28"][0].numpy(),
                    sample["grids"]["2048"]["actual_view28"][0].numpy(),
                ]
                for column, values in enumerate(images):
                    thumb = Image.fromarray(np.uint8(np.clip(values, 0.0, 1.0) * 255.0), mode="L").convert("RGB")
                    image.paste(thumb, (x + column * (tile + gap), y))
    image.save(run_dir / "comparison_atlas.png")


def _save_measurement_tables(run_dir: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    for item in inputs["entries"]:
        path = run_dir / "samples" / f"{_item_key(item)}.pt"
        _require(path.is_file(), f"sample evidence is missing: {path}", FileNotFoundError)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        payloads.append(payload)
    _require(len(payloads) == 200, "final evidence population is not 200 rows")
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        row: dict[str, Any] = {
            "method": payload["method"], "label": payload["label"], "cohort_row": payload["cohort_row"],
            "proposal_id": payload["proposal_id"], "component_id": payload["component_id"],
            "source_power_w": 2.493e-4,
            "solver_iterations": payload["solver"]["iterations_executed"],
            "solver_matrix_error": payload["solver"]["best_matrix_relative_error"],
            "phase_rad": payload["solver"]["phase_rad"],
            "command_min": float(payload["amplitude_command"].min()),
            "command_max": float(payload["amplitude_command"].max()),
            "command_mean": float(payload["amplitude_command"].mean()),
            "numeric_reproduction_witness": payload["numeric_reproduction_witness"],
            "cross_grid_coefficient_error": payload["cross_grid_coefficient_relative_error"],
            "cross_grid_cmos_intensity_error": payload["cross_grid_cmos_intensity_relative_error"],
        }
        for grid in ("2048", "4096"):
            values = payload["grids"][grid]
            for field in ("coefficient_relative_error", "raw_cmos_intensity_relative_error",
                          "response_matrix_prediction_relative_error", "actual_camera_power_w",
                          "direct_camera_power_w", "camera_pixel_area_m2"):
                row[f"{grid}_{field}"] = values[field]
            for index, name in enumerate(("encoder1_power_w", "encoder2_power_w", "ols_input_power_w",
                                          "decoder1_power_w", "decoder2_power_w")):
                row[f"{grid}_{name}"] = float(values["plane_powers_w"][index])
        rows.append(row)
    with (run_dir / "per_image_power_and_error.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_comparison_atlas(run_dir, inputs, payloads)
    method_results = {
        method: {
            "denominator": sum(row["method"] == method for row in rows),
            "witnesses": sum(row["method"] == method and row["numeric_reproduction_witness"] for row in rows),
            "by_class": {
                str(label): {
                    "denominator": sum(row["method"] == method and row["label"] == label for row in rows),
                    "witnesses": sum(row["method"] == method and row["label"] == label
                                     and row["numeric_reproduction_witness"] for row in rows),
                    "median_solver_error": float(np.median([
                        row["solver_matrix_error"] for row in rows
                        if row["method"] == method and row["label"] == label
                    ])),
                }
                for label in range(CLASS_COUNT)
            },
        }
        for method in METHODS
    }
    return {"samples": payloads, "rows": rows, "method_results": method_results,
            "witnesses": sum(bool(row["numeric_reproduction_witness"]) for row in rows)}


def _final_decision(
    run_dir: Path, inputs: dict[str, Any], config: dict[str, Any],
    matrix: torch.Tensor, models: dict[int, base.SmallMatchedAdapter],
    device: torch.device, identity: dict[str, Any],
) -> dict[str, Any]:
    """Rerun each saved command through both production grids and recompute every gate value."""
    records: list[dict[str, Any]] = []
    amplitude_commands: list[torch.Tensor] = []
    intensity_commands: list[torch.Tensor] = []
    command_identities: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    snapshots = {grid: _snapshot_state(model) for grid, model in models.items()}
    gain = float(inputs["checkpoint"]["camera_gain"])
    for item in inputs["entries"]:
        path = run_dir / "samples" / f"{_item_key(item)}.pt"
        _require(path.is_file(), f"saved instruction is missing: {path}", FileNotFoundError)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        _require(saved.get("task_id") == TASK_ID and saved.get("sample_key") == _item_key(item)
                 and saved.get("cohort_row") == int(item["cohort_row"])
                 and saved.get("method") == item["method"]
                 and saved.get("label") == int(item["requested_label"])
                 and saved.get("stream") == 0
                 and saved.get("proposal_id") == int(item["proposal_id"])
                 and saved.get("component_id") == int(item["component_id"]),
                 f"saved instruction is misattributed: {_item_key(item)}")
        saved_target = saved["target_coefficients"].to(torch.complex64)
        expected_target = _target_for(inputs, item).to(torch.complex64)
        _require(torch.equal(saved_target, expected_target),
                 f"saved target coefficients changed for {_item_key(item)}")
        amplitude_saved = saved["amplitude_command"].double()
        intensity_saved = saved["intensity_command"].double()
        command_shapes = (IMAGE_SHAPE, (IMAGE_SHAPE[0] * IMAGE_SHAPE[1],))
        _require(tuple(amplitude_saved.shape) in command_shapes
                 and tuple(intensity_saved.shape) in command_shapes,
                 f"saved amplitude/intensity command shape is invalid: {_item_key(item)}")
        amplitude_cpu = amplitude_saved.reshape(IMAGE_SHAPE)
        intensity_cpu = intensity_saved.reshape(IMAGE_SHAPE)
        _require(torch.isfinite(amplitude_cpu).all().item()
                 and float(amplitude_cpu.min()) >= 0.0 and float(amplitude_cpu.max()) <= 1.0
                 and torch.allclose(intensity_cpu, amplitude_cpu.square(), rtol=0.0, atol=2e-7),
                 f"saved amplitude/intensity command is invalid: {_item_key(item)}")
        amplitude_commands.append(amplitude_cpu.cpu())
        intensity_commands.append(intensity_cpu.cpu())
        command_identities.append({
            "sample_key": _item_key(item), "method": item["method"],
            "cohort_row": int(item["cohort_row"]),
            "label": int(item["requested_label"]),
            "proposal_id": int(item["proposal_id"]),
            "component_id": int(item["component_id"]),
        })
        target = _target_for(inputs, item).to(device)
        matrix_error = _phase_error(
            matrix @ amplitude_cpu.to(device).reshape(-1).to(torch.complex128), target
        )[1]
        reruns: dict[str, Any] = {}
        for grid in (2048, 4096):
            rerun = _sample_call(models[grid], amplitude_cpu.to(device), target, gain)
            saved_grid = saved["grids"][str(grid)]
            saved_output_error = _rel_l2(
                rerun["actual_raw_intensity_w_per_m2"].double(),
                saved_grid["actual_raw_intensity_w_per_m2"].double(),
            )
            saved_coefficient_error = _rel_l2(
                rerun["actual_coefficients"].to(torch.complex128),
                saved_grid["actual_coefficients"].to(torch.complex128),
            )
            saved_direct_raw_error = _rel_l2(
                rerun["direct_raw_intensity_w_per_m2"].double(),
                saved_grid["direct_raw_intensity_w_per_m2"].double(),
            )
            reruns[str(grid)] = {
                "coefficient_relative_error": _phase_error(rerun["actual_coefficients"].to(device), target)[1],
                "raw_cmos_intensity_relative_error": _rel_l2(
                    rerun["actual_raw_intensity_w_per_m2"].to(device),
                    rerun["direct_raw_intensity_w_per_m2"].to(device),
                ),
                "saved_output_replay_relative_error": saved_output_error,
                "saved_coefficient_replay_relative_error": saved_coefficient_error,
                "saved_direct_output_replay_relative_error": saved_direct_raw_error,
                "raw_direct": rerun["direct_raw_intensity_w_per_m2"].double(),
                "coeff_direct": rerun["direct_coefficients"].to(torch.complex128),
                "actual_camera_power_w": float(rerun["actual_camera_power_w"]),
                "direct_camera_power_w": float(rerun["direct_camera_power_w"]),
                "raw_actual": rerun["actual_raw_intensity_w_per_m2"].double(),
                "coeff_actual": rerun["actual_coefficients"].to(torch.complex128),
            }
        cross_coeff = _rel_l2(reruns["2048"]["coeff_actual"], reruns["4096"]["coeff_actual"])
        cross_raw = _rel_l2(reruns["2048"]["raw_actual"], reruns["4096"]["raw_actual"])
        response_prediction_error = _rel_l2(
            matrix @ amplitude_cpu.to(device).reshape(-1).to(torch.complex128),
            reruns["2048"]["coeff_actual"].to(device)
        )
        _require(response_prediction_error <= 1e-6,
                 f"saved instruction response matrix no longer matches production: {_item_key(item)}")
        _require(all(reruns[grid]["saved_output_replay_relative_error"] <= 1e-6
                     and reruns[grid]["saved_coefficient_replay_relative_error"] <= 1e-6
                     and reruns[grid]["saved_direct_output_replay_relative_error"] <= 1e-6
                     for grid in ("2048", "4096")),
                 f"saved output or frozen reference replay changed: {_item_key(item)}")
        _require(cross_coeff <= 1e-4 and cross_raw <= 1e-4,
                 f"2048/4096 numerical window failed for {_item_key(item)}")
        witness = bool(
            all(reruns[grid]["coefficient_relative_error"] <= 1e-4
                and reruns[grid]["raw_cmos_intensity_relative_error"] <= 1e-4
                and reruns[grid]["saved_output_replay_relative_error"] <= 1e-6
                for grid in ("2048", "4096"))
            and cross_coeff <= 1e-4 and cross_raw <= 1e-4
        )
        _require(witness == bool(saved["numeric_reproduction_witness"]),
                 f"saved witness status differs from final production replay: {_item_key(item)}")
        compact_rows.append({
            "target_coefficients": target.detach().cpu().to(torch.complex128),
            "identity": command_identities[-1],
            "grids": {
                grid: {
                    "actual_coefficients": reruns[grid]["coeff_actual"].detach().cpu().to(torch.complex128),
                    "direct_coefficients": reruns[grid]["coeff_direct"].detach().cpu().to(torch.complex128),
                    "actual_raw_cmos_intensity_w_per_m2": reruns[grid]["raw_actual"].detach().cpu(),
                    "direct_raw_cmos_intensity_w_per_m2": reruns[grid]["raw_direct"].detach().cpu(),
                    "source_saved_actual_coefficients": saved["grids"][grid]["actual_coefficients"].detach().cpu(),
                    "source_saved_direct_coefficients": saved["grids"][grid]["direct_coefficients"].detach().cpu(),
                    "source_saved_actual_raw_cmos_intensity_w_per_m2": saved["grids"][grid][
                        "actual_raw_intensity_w_per_m2"
                    ].detach().cpu().double(),
                    "source_saved_direct_raw_cmos_intensity_w_per_m2": saved["grids"][grid][
                        "direct_raw_intensity_w_per_m2"
                    ].detach().cpu().double(),
                    "saved_output_replay_relative_error": reruns[grid]["saved_output_replay_relative_error"],
                    "saved_coefficient_replay_relative_error": reruns[grid]["saved_coefficient_replay_relative_error"],
                    "saved_direct_output_replay_relative_error": reruns[grid]["saved_direct_output_replay_relative_error"],
                }
                for grid in ("2048", "4096")
            },
        })
        records.append({
            "method": item["method"], "label": int(item["requested_label"]),
            "cohort_row": int(item["cohort_row"]), "proposal_id": int(item["proposal_id"]),
            "response_matrix_target_relative_error": matrix_error,
            "response_matrix_prediction_relative_error": response_prediction_error,
            "cross_grid_coefficient_relative_error": cross_coeff,
            "cross_grid_cmos_intensity_relative_error": cross_raw,
            "grids": {grid: {key: value for key, value in reruns[grid].items()
                              if not isinstance(value, torch.Tensor)} for grid in ("2048", "4096")},
            "witness": witness,
        })
    for grid, model in models.items():
        _assert_state_unchanged(model, snapshots[grid])
    _require(len(amplitude_commands) == 200 and len(intensity_commands) == 200,
             "final consumer changed the saved command denominator")
    _atomic_torch(run_dir / "amplitude_and_intensity_commands.pt", {
        "schema": "lightgen_frozen_slm_ols_amplitude_commands.v1",
        "amplitude_command": torch.stack(amplitude_commands).unsqueeze(1),
        "intensity_command": torch.stack(intensity_commands).unsqueeze(1),
        "sample_identities": command_identities,
    })
    compact_grids: dict[str, dict[str, torch.Tensor]] = {}
    for grid in ("2048", "4096"):
        compact_grids[grid] = {
            "actual_coefficients": torch.stack([
                row["grids"][grid]["actual_coefficients"] for row in compact_rows
            ]),
            "direct_coefficients": torch.stack([
                row["grids"][grid]["direct_coefficients"] for row in compact_rows
            ]),
            "actual_raw_cmos_intensity_w_per_m2": torch.stack([
                row["grids"][grid]["actual_raw_cmos_intensity_w_per_m2"] for row in compact_rows
            ]).unsqueeze(1).double(),
            "direct_raw_cmos_intensity_w_per_m2": torch.stack([
                row["grids"][grid]["direct_raw_cmos_intensity_w_per_m2"] for row in compact_rows
            ]).unsqueeze(1).double(),
            "source_saved_actual_coefficients": torch.stack([
                row["grids"][grid]["source_saved_actual_coefficients"] for row in compact_rows
            ]).to(torch.complex128),
            "source_saved_direct_coefficients": torch.stack([
                row["grids"][grid]["source_saved_direct_coefficients"] for row in compact_rows
            ]).to(torch.complex128),
            "source_saved_actual_raw_cmos_intensity_w_per_m2": torch.stack([
                row["grids"][grid]["source_saved_actual_raw_cmos_intensity_w_per_m2"]
                for row in compact_rows
            ]).unsqueeze(1).double(),
            "source_saved_direct_raw_cmos_intensity_w_per_m2": torch.stack([
                row["grids"][grid]["source_saved_direct_raw_cmos_intensity_w_per_m2"]
                for row in compact_rows
            ]).unsqueeze(1).double(),
            "saved_output_replay_relative_error": torch.tensor([
                row["grids"][grid]["saved_output_replay_relative_error"] for row in compact_rows
            ], dtype=torch.float64),
            "saved_coefficient_replay_relative_error": torch.tensor([
                row["grids"][grid]["saved_coefficient_replay_relative_error"] for row in compact_rows
            ], dtype=torch.float64),
            "saved_direct_output_replay_relative_error": torch.tensor([
                row["grids"][grid]["saved_direct_output_replay_relative_error"] for row in compact_rows
            ], dtype=torch.float64),
        }
    _atomic_torch(run_dir / "compact_replay_evidence.pt", {
        "schema": "lightgen_frozen_slm_ols_compact_replay_evidence.v1",
        "task_id": TASK_ID,
        "run_id": identity["run_id"],
        "sample_identities": [row["identity"] for row in compact_rows],
        "target_coefficients": torch.stack([
            row["target_coefficients"] for row in compact_rows
        ]),
        "grids": compact_grids,
    })
    total = len(records)
    by_method = {
        method: {
            "denominator": sum(row["method"] == method for row in records),
            "passed": sum(row["method"] == method and row["witness"] for row in records),
            "by_class": {
                str(label): {
                    "denominator": sum(row["method"] == method and row["label"] == label for row in records),
                    "passed": sum(row["method"] == method and row["label"] == label and row["witness"]
                                  for row in records),
                    "median_coefficient_error_2048": float(np.median([
                        row["grids"]["2048"]["coefficient_relative_error"] for row in records
                        if row["method"] == method and row["label"] == label
                    ])),
                    "median_cmos_error_2048": float(np.median([
                        row["grids"]["2048"]["raw_cmos_intensity_relative_error"] for row in records
                        if row["method"] == method and row["label"] == label
                    ])),
                }
                for label in range(CLASS_COUNT)
            },
        }
        for method in METHODS
    }
    _require(total == 200 and all(value["denominator"] == 100 for value in by_method.values()),
             "final consumer changed the fixed sample denominator")
    passed = sum(row["witness"] for row in records)
    decision = (
        "ALL_200_NUMERICAL_COMMAND_WITNESSES_VERIFIED" if passed == 200 else
        "PARTIAL_WITNESSES_OTHERS_NOT_FOUND_WITHIN_FIXED_SOLVER_BUDGET" if passed > 0 else
        "NO_WITNESSES_FOUND_WITHIN_FIXED_SOLVER_BUDGET"
    )
    result = {
        "schema": "lightgen_frozen_slm_ols_reachability_final_decision.v1",
        "task_id": TASK_ID,
        "run_id": identity["run_id"],
        "status": decision,
        "decision_scope": "continuous ideal bounded-amplitude model and the fixed 200 saved G1/G4 samples",
        "decision": decision,
        "population": {"G1": 100, "G4": 100, "total": 200, "class_count_per_method": 10},
        "optics": {
            "source_power_w": 2.493e-4,
            "global_gain": gain,
            "gaussian_ols": GAUSSIAN_OLS_MODEL,
            "propagation_grids": [2048, 4096],
            "encoder_layers": 2,
            "decoder_layers": 2,
            "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
        },
        "solver": {
            "dtype": "float64",
            "starts": 8,
            "max_steps_per_start": 5000,
            "coefficient_stop_relative_error": 1e-6,
            "witness_coefficient_relative_error": 1e-4,
            "witness_cmos_intensity_relative_error": 1e-4,
        },
        "passed": passed,
        "by_method": by_method,
        "command_artifact": "amplitude_and_intensity_commands.pt",
        "model_updates": 0,
        "optical_parameter_gradients": 0,
        "optical_parameter_updates": 0,
        "adam_calls": 0,
        "prior_fit": 0,
        "new_sampling": 0,
        "new_human_review": 0,
        "matrix_and_grid_replay_recomputed_from_saved_amplitude_commands": True,
        "historical_valid_failed_preserved": True,
        "geometry_status": "PROJECT_GEOMETRY_UNMEASURED",
        "hardware_acceptance": False,
        "records": records,
    }
    _atomic_json(run_dir / "final_decision.json", result)
    return result


def _write_report(run_dir: Path, result: dict[str, Any], preflight: dict[str, Any]) -> None:
    passed = result["passed"]
    state = "完成；200 条全部通过" if passed == 200 else f"完成；{passed}/200 条通过"
    if passed < 200:
        state += "，其余仅表示固定预算内未找到见证"
    science = (
        "支持这批固定 G1/G4 复系数在连续理想振幅模型下存在经完整生产光路验证的输入指令。"
        if passed == 200 else
        "保留已找到的数值见证；其余样本在指定求解预算内未找到见证，不能解释为物理不可达。"
    )
    report = f"""# 冻结小架构 SLM→OLS 输入可达性诊断

结果与状态：{state}。G1 {result['by_method']['G1']['passed']}/100，G4 {result['by_method']['G4']['passed']}/100；样本分母由最终消费者从保存输入逐项重算。工程执行状态与科学支持范围分开记录。

科学认识：{science}严格限于固定 C 累计 45,000 步、Gaussian S12、两层 Encoder/Decoder、249.3 μW 源功率、原全局增益和这 200 条原盲评样本。系数只允许单一公共相位；未缩放目标、未补功率或改曝光。历史 VALID_FAILED 与 PROJECT_GEOMETRY_UNMEASURED 保持。本结果不代表硬件容差、实时映射或语义质量提升。

目标缺口：本轮没有实物灰阶响应、残余相位、SLM 标定曲线、实测光纤复传输、器件装配偏差或相机实测校准；因此尚不能支持实物部署、一次电子映射、实时生成或全系统可读性结论。

工程完成与验证：Autograd 响应矩阵预测误差 {preflight['matrix_prediction_relative_error']:.3g}，方向导数误差 {preflight['direction_derivative_relative_error']:.3g}；解析正负控误差为正例 {preflight['analytic_controls']['known_command_global_phase_relative_error']:.3g}、两端口不可达 {preflight['analytic_controls']['impossible_two_port_relative_error']:.3g}、超幅目标 {preflight['analytic_controls']['amplitude_overflow_target_relative_error']:.3g}。原 200 行直接注入回放在预检和最终消费者中分别重放，证据见 reference_replay.json、reference_replay_final.json 和 preflight.json。完整消费者从 200 个保存命令重放两网格，并重新计算系数、强度、跨网格误差和通过分母；200×200 振幅/强度指令位于 amplitude_and_intensity_commands.pt，紧凑复场/原生 CMOS 复放位于 compact_replay_evidence.pt，逐图功率/误差表为 per_image_power_and_error.csv，对照图为 comparison_atlas.png。Windows 独立复消费记录在 windows_reconsumption.json；独立复审证据见 independent_review.md。光学参数更新、Adam 调用、先验拟合、重采样与新人工评阅均为 0。模型状态前后相同。

决策影响与下一步：全部通过时仅支持这批样本在指定连续理想模型下存在离线制备指令；部分通过时，未通过者只限制当前求解器与预算的见证结论。建议依据本结果另行设计灰阶/相位标定与物理传输验证；该建议不属于本轮授权。本轮到此停止，不启动训练、电子映射、实时生成或实物部署。
"""
    (run_dir / "report.md").write_text(report, encoding="utf-8")


def _preflight(
    run_dir: Path, identity: dict[str, Any], inputs: dict[str, Any], config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any], dict[int, base.SmallMatchedAdapter]]:
    def progress(value: dict[str, Any]) -> None:
        _atomic_json(run_dir / "progress.json", {"task_id": TASK_ID, "run_id": identity["run_id"], **value})

    models = {
        2048: _new_model(inputs["checkpoint"], 2048, device),
        4096: _new_model(inputs["checkpoint"], 4096, device),
    }
    reference_model = _new_reference_model(inputs["checkpoint"], device)
    before = {grid: _snapshot_state(model) for grid, model in models.items()}
    reference = _reference_replay_all(
        reference_model, inputs, float(inputs["checkpoint"]["camera_gain"]), run_dir,
        run_id=identity["run_id"],
    )
    analytic = _analytic_solver_controls()
    matrix_path = run_dir / "response_matrix.pt"
    if matrix_path.is_file():
        saved = torch.load(matrix_path, map_location="cpu", weights_only=False)
        _require(saved.get("task_id") == TASK_ID and tuple(saved["matrix"].shape) == (100, 40_000),
                 "saved production response matrix is invalid")
        matrix = saved["matrix"].to(device=device, dtype=torch.complex128)
    else:
        matrix = _make_response(models[2048], device, progress)
        _atomic_torch(matrix_path, {"task_id": TASK_ID, "dtype": "complex128", "matrix": matrix.cpu()})
    matrix_checks = _matrix_and_derivative_checks(models[2048], matrix, device)
    rows = _preflight_rows(inputs["entries"])
    sample_payloads: list[dict[str, Any]] = []
    for index, item in enumerate(rows, start=1):
        payload = _run_item(item, inputs, matrix, models, run_dir, config, device)
        sample_payloads.append(payload)
        _require(payload["grids"]["2048"]["response_matrix_prediction_relative_error"] <= 1e-6,
                 f"production response matrix failed its command-path check: {_item_key(item)}")
        _require(payload["cross_grid_coefficient_relative_error"] <= 1e-4
                 and payload["cross_grid_cmos_intensity_relative_error"] <= 1e-4,
                 f"2048/4096 numerical window failed for {_item_key(item)}")
        progress({"phase": "twenty_sample_preflight", "completed_samples": index, "total_samples": 20,
                  "model_updates": 0, "adam_calls": 0})
    for grid, model in models.items():
        _assert_state_unchanged(model, before[grid])
    _require(all(not parameter.requires_grad for parameter in reference_model.parameters()),
             "reference model became trainable during the zero-update preflight")
    preflight = {
        "schema": "lightgen_frozen_slm_ols_input_reachability_preflight.v1",
        "task_id": TASK_ID,
        "run_id": identity["run_id"],
        "status": "PREFLIGHT_PASSED",
        "preflight_population": {"rows": 20, "G1": 10, "G4": 10, "per_class_per_method": 1},
        **matrix_checks,
        "response_matrix_scalar_input_vjps": 2 * COEFFICIENTS,
        "optical_parameter_gradients": 0,
        "optical_parameter_updates": 0,
        "analytic_controls": analytic,
        "reference_replay": {
            "rows": reference["rows"], "all_passed": reference["all_passed"],
            "max_relative_error": reference["max_relative_error"],
            "rtol": reference["rtol"], "atol": reference["atol"],
        },
        "cross_grid": {
            "rows": 20,
            "max_coefficient_relative_error": max(
                float(sample["cross_grid_coefficient_relative_error"]) for sample in sample_payloads
            ),
            "max_cmos_intensity_relative_error": max(
                float(sample["cross_grid_cmos_intensity_relative_error"]) for sample in sample_payloads
            ),
            "limit": 1e-4,
        },
        "frozen_model_state_unchanged": True,
        "model_updates": 0,
        "adam_calls": 0,
        "prior_fit": 0,
        "new_sampling": 0,
        "new_human_review": 0,
    }
    _atomic_json(run_dir / "preflight.json", preflight)
    identity["preflight_status"] = "PREFLIGHT_PASSED"
    _atomic_json(run_dir / "run_identity.json", identity)
    return matrix, preflight, models


def _execute_full(
    run_dir: Path, identity: dict[str, Any], inputs: dict[str, Any], config: dict[str, Any],
    matrix: torch.Tensor, models: dict[int, base.SmallMatchedAdapter], device: torch.device,
) -> None:
    preflight_path = run_dir / "preflight.json"
    _require(preflight_path.is_file(),
             "full population requires a saved 200-row reference and 20-row engineering preflight")
    receipt = json.loads(preflight_path.read_text(encoding="utf-8"))
    _require(receipt.get("task_id") == TASK_ID and receipt.get("run_id") == identity["run_id"]
             and receipt.get("status") == "PREFLIGHT_PASSED"
             and receipt.get("frozen_model_state_unchanged") is True
             and receipt.get("reference_replay", {}).get("rows") == 200
             and receipt.get("reference_replay", {}).get("all_passed") is True,
             "full population is blocked by invalid preflight or reference replay")
    done = 0
    for item in inputs["entries"]:
        sample_path = run_dir / "samples" / f"{_item_key(item)}.pt"
        if sample_path.is_file():
            saved = torch.load(sample_path, map_location="cpu", weights_only=False)
            _require(saved.get("sample_key") == _item_key(item) and saved.get("task_id") == TASK_ID,
                     f"existing saved sample does not match the fixed input: {_item_key(item)}")
            state_path = run_dir / "solver_state.pt"
            if state_path.is_file():
                state = torch.load(state_path, map_location="cpu", weights_only=False)
                if state.get("sample_key") == _item_key(item):
                    state_path.unlink()
            done += 1
            continue
        _run_item(item, inputs, matrix, models, run_dir, config, device)
        done += 1
        if done % 5 == 0:
            _atomic_json(run_dir / "progress.json", {
                "task_id": TASK_ID, "run_id": identity["run_id"], "phase": "fixed_200_population",
                "completed_samples": done, "total_samples": 200, "model_updates": 0,
                "adam_calls": 0, "prior_fit": 0, "new_sampling": 0, "new_human_review": 0,
            })


def _verify_evidence_bundle(
    run_dir: Path,
    identity: dict[str, Any],
    inputs: dict[str, Any],
) -> dict[str, Any]:
    """Independently recompute the final decision from the compact saved evidence on Windows."""
    _require(identity.get("task_id") == TASK_ID and identity.get("status") == "CLOSED",
             "Windows reconsumption requires this completed diagnostic run")
    run_id = identity.get("run_id")
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    final = json.loads((run_dir / "final_decision.json").read_text(encoding="utf-8"))
    reference = json.loads((run_dir / "reference_replay_final.json").read_text(encoding="utf-8"))
    _require(preflight.get("task_id") == TASK_ID and preflight.get("run_id") == run_id
             and preflight.get("status") == "PREFLIGHT_PASSED"
             and preflight.get("reference_replay", {}).get("rows") == 200
             and preflight.get("reference_replay", {}).get("all_passed") is True,
             "saved zero-update preflight is missing, mismatched or failed")
    _require(final.get("task_id") == TASK_ID and final.get("run_id") == run_id
             and reference.get("task_id") == TASK_ID and reference.get("rows") == 200
             and reference.get("all_passed") is True,
             "saved final decision or historical replay belongs to another run")
    for field in ("model_updates", "optical_parameter_gradients", "optical_parameter_updates",
                  "adam_calls", "prior_fit", "new_sampling", "new_human_review"):
        _require(final.get(field) == 0, f"saved final decision reports a forbidden action: {field}")

    matrix_payload = torch.load(run_dir / "response_matrix.pt", map_location="cpu", weights_only=False)
    commands = torch.load(run_dir / "amplitude_and_intensity_commands.pt",
                          map_location="cpu", weights_only=False)
    compact = torch.load(run_dir / "compact_replay_evidence.pt", map_location="cpu", weights_only=False)
    replay_outputs = torch.load(run_dir / "reference_replay_final.pt",
                                map_location="cpu", weights_only=False)
    _require(matrix_payload.get("task_id") == TASK_ID
             and tuple(matrix_payload["matrix"].shape) == (COEFFICIENTS, 40_000),
             "saved response matrix has the wrong producer identity or dimensions")
    _require(commands.get("schema") == "lightgen_frozen_slm_ols_amplitude_commands.v1"
             and compact.get("schema") == "lightgen_frozen_slm_ols_compact_replay_evidence.v1"
             and compact.get("task_id") == TASK_ID and compact.get("run_id") == run_id
             and replay_outputs.get("schema") == "lightgen_frozen_slm_ols_reference_replay_outputs.v1"
             and replay_outputs.get("task_id") == TASK_ID and replay_outputs.get("run_id") == run_id,
             "saved command or compact replay evidence has the wrong identity")
    entries = inputs["entries"]
    _require(len(entries) == 200 and len(commands.get("sample_identities", [])) == 200
             and len(compact.get("sample_identities", [])) == 200
             and tuple(commands["amplitude_command"].shape) == (200, 1, 200, 200)
             and tuple(commands["intensity_command"].shape) == (200, 1, 200, 200)
             and tuple(compact["target_coefficients"].shape) == (200, COEFFICIENTS),
             "saved evidence changed the fixed 200-row population or image shape")

    expected_identities = [
        {
            "sample_key": _item_key(item), "method": item["method"],
            "cohort_row": int(item["cohort_row"]), "label": int(item["requested_label"]),
            "proposal_id": int(item["proposal_id"]), "component_id": int(item["component_id"]),
        }
        for item in entries
    ]
    _require(commands["sample_identities"] == expected_identities
             and compact["sample_identities"] == expected_identities,
             "saved command identities no longer match the fixed selection manifest")
    matrix = matrix_payload["matrix"].to(torch.complex128)
    amplitude = commands["amplitude_command"][:, 0].reshape(200, -1).double()
    intensity = commands["intensity_command"][:, 0].reshape(200, -1).double()
    _require(torch.isfinite(amplitude).all().item() and float(amplitude.min()) >= 0.0
             and float(amplitude.max()) <= 1.0
             and torch.allclose(intensity, amplitude.square(), rtol=0.0, atol=2e-7),
             "saved amplitude command is out of range or its intensity command is inconsistent")
    targets = torch.stack([_target_for(inputs, item).cpu().to(torch.complex128) for item in entries])
    _require(torch.equal(compact["target_coefficients"].to(torch.complex128), targets),
             "compact replay targets differ from the fixed selected coefficients")
    final_records = final.get("records")
    _require(isinstance(final_records, list) and len(final_records) == 200,
             "saved final consumer did not preserve all 200 records")
    record_checks: list[dict[str, Any]] = []
    max_coefficient_error = 0.0
    max_cmos_intensity_error = 0.0
    max_saved_actual = 0.0
    max_saved_direct = 0.0
    max_saved_coeff = 0.0
    max_cross_coefficient_error = 0.0
    max_cross_cmos_error = 0.0
    max_response_matrix_prediction_error = 0.0
    compact_grids = compact["grids"]
    for grid in ("2048", "4096"):
        values = compact_grids.get(grid, {})
        for name, shape in (
            ("actual_coefficients", (200, COEFFICIENTS)),
            ("direct_coefficients", (200, COEFFICIENTS)),
            ("source_saved_actual_coefficients", (200, COEFFICIENTS)),
            ("source_saved_direct_coefficients", (200, COEFFICIENTS)),
            ("actual_raw_cmos_intensity_w_per_m2", (200, 1, 200, 200)),
            ("direct_raw_cmos_intensity_w_per_m2", (200, 1, 200, 200)),
            ("source_saved_actual_raw_cmos_intensity_w_per_m2", (200, 1, 200, 200)),
            ("source_saved_direct_raw_cmos_intensity_w_per_m2", (200, 1, 200, 200)),
        ):
            _require(tuple(values.get(name, torch.empty(0)).shape) == shape,
                     f"compact replay field has wrong dimensions: {grid}.{name}")

    for index, (item, identity_row, record) in enumerate(zip(entries, expected_identities, final_records)):
        _require(record.get("method") == identity_row["method"]
                 and record.get("label") == identity_row["label"]
                 and record.get("cohort_row") == identity_row["cohort_row"]
                 and record.get("proposal_id") == identity_row["proposal_id"],
                 f"final-decision record is misattributed: {identity_row['sample_key']}")
        target = targets[index]
        predicted = matrix @ amplitude[index].to(torch.complex128)
        matrix_target_error = _phase_error(predicted, target)[1]
        grid_metrics: dict[str, dict[str, float]] = {}
        for grid in ("2048", "4096"):
            values = compact_grids[grid]
            actual_coeff = values["actual_coefficients"][index].to(torch.complex128)
            source_actual_coeff = values["source_saved_actual_coefficients"][index].to(torch.complex128)
            actual_raw = values["actual_raw_cmos_intensity_w_per_m2"][index].double()
            direct_raw = values["direct_raw_cmos_intensity_w_per_m2"][index].double()
            source_actual_raw = values["source_saved_actual_raw_cmos_intensity_w_per_m2"][index].double()
            source_direct_raw = values["source_saved_direct_raw_cmos_intensity_w_per_m2"][index].double()
            metrics = {
                "coefficient_relative_error": _phase_error(actual_coeff, target)[1],
                "raw_cmos_intensity_relative_error": _rel_l2(actual_raw, direct_raw),
                "saved_output_replay_relative_error": _rel_l2(actual_raw, source_actual_raw),
                "saved_coefficient_replay_relative_error": _rel_l2(actual_coeff, source_actual_coeff),
                "saved_direct_output_replay_relative_error": _rel_l2(direct_raw, source_direct_raw),
            }
            _require(all(math.isfinite(value) for value in metrics.values()),
                     f"recomputed nonfinite result: {identity_row['sample_key']} {grid}")
            max_coefficient_error = max(max_coefficient_error, metrics["coefficient_relative_error"])
            max_cmos_intensity_error = max(max_cmos_intensity_error,
                                           metrics["raw_cmos_intensity_relative_error"])
            max_saved_actual = max(max_saved_actual, metrics["saved_output_replay_relative_error"])
            max_saved_direct = max(max_saved_direct, metrics["saved_direct_output_replay_relative_error"])
            max_saved_coeff = max(max_saved_coeff, metrics["saved_coefficient_replay_relative_error"])
            recorded_grid = record.get("grids", {}).get(grid, {})
            for name, value in metrics.items():
                recorded = recorded_grid.get(name)
                _require(isinstance(recorded, (float, int))
                         and abs(float(recorded) - value) <= max(1e-10, abs(value) * 1e-8),
                         f"remote final metric differs from Windows recomputation: {identity_row['sample_key']} {grid} {name}")
            grid_metrics[grid] = metrics
        cross_coefficient_error = _rel_l2(
            compact_grids["2048"]["actual_coefficients"][index].to(torch.complex128),
            compact_grids["4096"]["actual_coefficients"][index].to(torch.complex128),
        )
        cross_cmos_error = _rel_l2(
            compact_grids["2048"]["actual_raw_cmos_intensity_w_per_m2"][index].double(),
            compact_grids["4096"]["actual_raw_cmos_intensity_w_per_m2"][index].double(),
        )
        prediction_error = _rel_l2(
            predicted,
            compact_grids["2048"]["actual_coefficients"][index].to(torch.complex128),
        )
        max_cross_coefficient_error = max(max_cross_coefficient_error, cross_coefficient_error)
        max_cross_cmos_error = max(max_cross_cmos_error, cross_cmos_error)
        max_response_matrix_prediction_error = max(max_response_matrix_prediction_error, prediction_error)
        _require(prediction_error <= 1e-6 and cross_coefficient_error <= 1e-4
                 and cross_cmos_error <= 1e-4,
                 f"Windows recomputation failed a matrix or cross-grid check: {identity_row['sample_key']}")
        for key, value in (
            ("response_matrix_target_relative_error", matrix_target_error),
            ("response_matrix_prediction_relative_error", prediction_error),
            ("cross_grid_coefficient_relative_error", cross_coefficient_error),
            ("cross_grid_cmos_intensity_relative_error", cross_cmos_error),
        ):
            recorded = record.get(key)
            _require(isinstance(recorded, (float, int))
                     and abs(float(recorded) - value) <= max(1e-10, abs(value) * 1e-8),
                     f"remote final metric differs from Windows recomputation: {identity_row['sample_key']} {key}")
        _require(all(grid_metrics[grid]["saved_output_replay_relative_error"] <= 1e-6
                     and grid_metrics[grid]["saved_coefficient_replay_relative_error"] <= 1e-6
                     and grid_metrics[grid]["saved_direct_output_replay_relative_error"] <= 1e-6
                     for grid in ("2048", "4096")),
                 f"saved production replay differs from its source sample: {identity_row['sample_key']}")
        witness = bool(
            all(grid_metrics[grid]["coefficient_relative_error"] <= 1e-4
                and grid_metrics[grid]["raw_cmos_intensity_relative_error"] <= 1e-4
                and grid_metrics[grid]["saved_output_replay_relative_error"] <= 1e-6
                for grid in ("2048", "4096"))
            and cross_coefficient_error <= 1e-4 and cross_cmos_error <= 1e-4
        )
        _require(bool(record.get("witness")) == witness,
                 f"recomputed witness differs from final decision: {identity_row['sample_key']}")
        record_checks.append({**identity_row, "witness": witness})

    method_counts = {
        method: {
            "denominator": sum(row["method"] == method for row in record_checks),
            "passed": sum(row["method"] == method and row["witness"] for row in record_checks),
        }
        for method in METHODS
    }
    _require(all(value["denominator"] == 100 for value in method_counts.values()),
             "Windows recomputation changed a fixed method denominator")
    passed = sum(row["witness"] for row in record_checks)
    decision = (
        "ALL_200_NUMERICAL_COMMAND_WITNESSES_VERIFIED" if passed == 200 else
        "PARTIAL_WITNESSES_OTHERS_NOT_FOUND_WITHIN_FIXED_SOLVER_BUDGET" if passed > 0 else
        "NO_WITNESSES_FOUND_WITHIN_FIXED_SOLVER_BUDGET"
    )
    _require(final.get("decision") == decision and final.get("passed") == passed
             and all(final.get("by_method", {}).get(method, {}).get("denominator") == counts["denominator"]
                     and final.get("by_method", {}).get(method, {}).get("passed") == counts["passed"]
                     for method, counts in method_counts.items()),
             "saved final decision summary differs from the independently recomputed population")

    reference_methods = replay_outputs.get("methods", {})
    reference_checks = {row.get("sample_key"): row for row in reference.get("checks", [])}
    _require(len(reference_checks) == 200 and set(reference_methods) == set(METHODS),
             "historical reference replay output is incomplete")
    reference_max_native = 0.0
    reference_max_view = 0.0
    reference_pass_count = 0
    for method in METHODS:
        method_items = [item for item in entries if item["method"] == method]
        method_output = reference_methods[method]
        cohort_rows = method_output["cohort_rows"].long().tolist()
        _require(len(cohort_rows) == 100
                 and tuple(method_output["native200"].shape) == (100, 1, 200, 200)
                 and tuple(method_output["view28"].shape) == (100, 1, 28, 28),
                 f"saved original replay output is incomplete for {method}")
        _require(cohort_rows == [int(item["cohort_row"]) for item in method_items],
                 f"saved original replay rows changed for {method}")
        for row_index, item in enumerate(method_items):
            sample_key = _item_key(item)
            replay_index = inputs["replay_indices"][method][int(item["cohort_row"])]
            expected_native = inputs["replay"][method]["native200"][replay_index].float()
            expected_view = inputs["replay"][method]["view28"][replay_index].float()
            actual_native = method_output["native200"][row_index].float()
            actual_view = method_output["view28"][row_index].float()
            native_passed = torch.allclose(actual_native, expected_native, rtol=2e-6, atol=5e-8)
            view_passed = torch.allclose(actual_view, expected_view, rtol=2e-6, atol=5e-8)
            native_error = _rel_l2(actual_native.double(), expected_native.double())
            view_error = _rel_l2(actual_view.double(), expected_view.double())
            reference_max_native = max(reference_max_native, native_error)
            reference_max_view = max(reference_max_view, view_error)
            check = reference_checks[sample_key]
            _require(bool(check.get("passed")) == native_passed
                     and bool(check.get("view28_passed")) == view_passed,
                     f"saved reference-replay check differs from Windows recomputation: {sample_key}")
            _require(native_passed and view_passed, f"historical saved view replay failed: {sample_key}")
            reference_pass_count += 1
    _require(reference_pass_count == 200 and reference.get("max_relative_error") is not None
             and abs(float(reference["max_relative_error"]) - reference_max_native) <= 1e-10
             and abs(float(reference.get("max_view28_relative_error", math.inf)) - reference_max_view) <= 1e-10,
             "historical reference-replay summary differs from Windows recomputation")

    result = {
        "schema": "lightgen_frozen_slm_ols_windows_reconsumption.v1",
        "task_id": TASK_ID,
        "run_id": run_id,
        "status": "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED",
        "decision": decision,
        "population": {"G1": method_counts["G1"], "G4": method_counts["G4"], "total": 200},
        "recomputed_from": [
            "response_matrix.pt", "amplitude_and_intensity_commands.pt",
            "compact_replay_evidence.pt", "reference_replay_final.pt",
            "fixed_selection_manifest_and_replay",
        ],
        "historical_reference_replay": {
            "rows": reference_pass_count, "all_passed": reference_pass_count == 200,
            "max_native_relative_error": reference_max_native,
            "max_view28_relative_error": reference_max_view,
        },
        "saved_command_replay": {
            "rows": 200, "passed": passed,
            "max_actual_output_replay_relative_error": max_saved_actual,
            "max_direct_output_replay_relative_error": max_saved_direct,
            "max_coefficient_replay_relative_error": max_saved_coeff,
        },
        "recomputed_witness_metrics": {
            "max_phase_aligned_coefficient_relative_error": max_coefficient_error,
            "max_native_cmos_intensity_relative_error": max_cmos_intensity_error,
            "max_cross_grid_coefficient_relative_error": max_cross_coefficient_error,
            "max_cross_grid_cmos_intensity_relative_error": max_cross_cmos_error,
            "max_response_matrix_prediction_relative_error": max_response_matrix_prediction_error,
        },
        "model_updates": 0, "adam_calls": 0, "prior_fit": 0,
        "new_sampling": 0, "new_human_review": 0,
        "hardware_acceptance": False,
    }
    report_path = run_dir / "report.md"
    report = report_path.read_text(encoding="utf-8")
    marker = "模型状态前后相同。"
    review_phrase = "独立复审证据见 independent_review.md。"
    report = report.replace("Windows 独立复消费记录在 windows_reconsumption.json；", "")
    report = report.replace(review_phrase, "")
    windows_sentence = "Windows 端独立重消费再次重算 200 条指令及两网格误差"
    report_end = "；通过数见 windows_reconsumption.json。"
    if windows_sentence in report:
        start = report.index(windows_sentence)
        end = report.find(report_end, start)
        if end >= 0:
            report = report[:start] + report[end + len(report_end):]
        else:
            report = report.replace(
                windows_sentence + "，通过数见 windows_reconsumption.json；", ""
            )
    required_headings = (
        "结果与状态：", "科学认识：", "目标缺口：", "工程完成与验证：", "决策影响与下一步：",
    )
    _require((run_dir / "independent_review.md").is_file()
             and report.count(marker) == 1
             and all(report.count(heading) == 1 for heading in required_headings),
             "fixed five-part report is missing its local review or model-freeze record")
    metrics = result["recomputed_witness_metrics"]
    detail = (
        f"，最大公共相位系数误差 {metrics['max_phase_aligned_coefficient_relative_error']:.3g}，"
        f"最大原生 CMOS 强度误差 {metrics['max_native_cmos_intensity_relative_error']:.3g}，"
        f"最大跨网格系数/强度误差 "
        f"{metrics['max_cross_grid_coefficient_relative_error']:.3g}/"
        f"{metrics['max_cross_grid_cmos_intensity_relative_error']:.3g}"
    )
    addition = windows_sentence + detail + report_end + review_phrase
    report = report.replace(marker, addition + marker)
    report_path.write_text(report, encoding="utf-8")
    _atomic_json(run_dir / "windows_reconsumption.json", result)
    return result


def _verify_saved(
    run_dir: Path, identity: dict[str, Any], inputs: dict[str, Any], config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    matrix_path = run_dir / "response_matrix.pt"
    _require(matrix_path.is_file(), "final consumer is missing the saved production response matrix")
    saved = torch.load(matrix_path, map_location="cpu", weights_only=False)
    _require(saved.get("task_id") == TASK_ID, "response matrix belongs to another run")
    matrix = saved["matrix"].to(device=device, dtype=torch.complex128)
    models = {grid: _new_model(inputs["checkpoint"], grid, device) for grid in (2048, 4096)}
    reference_model = _new_reference_model(inputs["checkpoint"], device)
    final_reference = _reference_replay_all(
        reference_model, inputs, float(inputs["checkpoint"]["camera_gain"]), run_dir,
        result_name="reference_replay_final.json", run_id=identity["run_id"],
    )
    result = _final_decision(run_dir, inputs, config, matrix, models, device, identity)
    result["original_reference_replay_final"] = {
        "rows": final_reference["rows"],
        "all_passed": final_reference["all_passed"],
        "max_relative_error": final_reference["max_relative_error"],
        "max_view28_relative_error": final_reference["max_view28_relative_error"],
    }
    tables = _save_measurement_tables(run_dir, inputs)
    result["sample_summary"] = tables["method_results"]
    _atomic_json(run_dir / "final_decision.json", result)
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8"))
    _write_report(run_dir, result, preflight)
    identity["status"] = "CLOSED"
    identity["final_decision"] = result["decision"]
    identity["passed"] = result["passed"]
    _atomic_json(run_dir / "run_identity.json", identity)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    modes.add_argument("--verify-evidence-only", action="store_true")
    args = parser.parse_args()
    run_dir: Path | None = None
    try:
        config_path = args.config.resolve()
        config = _load_yaml(config_path)
        _validate_config(config, config_path)
        run_dir = args.output_dir.resolve()
        identity = _prepare_output(run_dir, allow_closed=args.verify_only or args.verify_evidence_only)
        inputs = _load_inputs(config)
        _record_input_authority(run_dir, identity, config_path, config, inputs)
        if args.verify_evidence_only:
            result = _verify_evidence_bundle(run_dir, identity, inputs)
            print(json.dumps({"status": result["status"], "decision": result["decision"],
                              "passed": result["population"]["G1"]["passed"]
                              + result["population"]["G4"]["passed"], "total": 200}, ensure_ascii=False))
            return
        _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized AutoDL measurement")
        device = torch.device("cuda")
        if args.preflight_only:
            matrix, preflight, models = _preflight(run_dir, identity, inputs, config, device)
            _atomic_json(run_dir / "preflight_execution.json", {
                "status": preflight["status"], "rows": 20, "reference_rows": 200,
                "response_matrix_shape": list(matrix.shape), "device": str(device),
                "model_updates": 0, "adam_calls": 0,
            })
            print(json.dumps({"status": preflight["status"], "run_id": identity["run_id"],
                              "preflight": str(run_dir / "preflight.json")}, ensure_ascii=False))
            return
        if args.execute:
            receipt_path = run_dir / "preflight.json"
            _require(receipt_path.is_file(), "--execute requires a passing same-run preflight")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            _require(receipt.get("status") == "PREFLIGHT_PASSED" and receipt.get("run_id") == identity["run_id"],
                     "--execute is blocked because the engineering preflight did not pass for this run")
            matrix_payload = torch.load(run_dir / "response_matrix.pt", map_location="cpu", weights_only=False)
            matrix = matrix_payload["matrix"].to(device=device, dtype=torch.complex128)
            models = {grid: _new_model(inputs["checkpoint"], grid, device) for grid in (2048, 4096)}
            _execute_full(run_dir, identity, inputs, config, matrix, models, device)
            print(json.dumps({"status": "POPULATION_COMPLETE", "run_id": identity["run_id"],
                              "samples": 200}, ensure_ascii=False))
            return
        result = _verify_saved(run_dir, identity, inputs, config, device)
        print(json.dumps({"status": result["status"], "run_id": identity["run_id"],
                          "passed": result["passed"], "total": 200}, ensure_ascii=False))
    except Exception as exc:
        if run_dir is not None and run_dir.is_dir():
            _atomic_json(run_dir / "engineering_failure.json", {
                "task_id": TASK_ID, "status": "INVALID_HARNESS",
                "exception_type": type(exc).__name__, "message": str(exc),
            })
        print(f"ENGINEERING_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
