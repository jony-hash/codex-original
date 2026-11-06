"""Run a zero-update paired common-phase feasibility diagnosis for G1."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    RUN_COMPLETED,
    RUN_SCIENTIFIC_ACTIVE,
)
from scripts.research.lightgen import diagnose_lightgen_frozen_slm_ols_reachability_v1 as reachability
from scripts.research.lightgen import execute_lightgen_g1_projected_fit20_box_ab_v1 as fit20
from scripts.research.lightgen import execute_lightgen_g1_projected_box_fullpop_v1 as fullpop
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_G1_FULLPOP_FEASIBILITY_DIAGNOSIS_V1"
AUTHORITY = "USER_AUTHORIZED_LIGHTGEN_G1_FULLPOP_FEASIBILITY_DIAGNOSIS_20261027"
SCHEMA = "lightgen_g1_fullpop_feasibility_diagnosis.v1"
MODULE_PATH = f"scripts.research.lightgen.{Path(__file__).stem}"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_fullpop_feasibility_diagnosis_v1.yaml"
RUNS_DIR = ROOT / "outputs/20261027_lightgen_g1_fullpop_feasibility_diagnosis_v1/runs"
PREFLIGHT_DIR = RUNS_DIR / "preflight_01"
ATTEMPT_DIR = RUNS_DIR / "attempt_01"
REVIEW_DIR = RUNS_DIR / "review_repair_01"
TARGET_COUNT = 20
CLASS_COUNT = 10
PORT_COUNT = 100
PIXEL_COUNT = 40_000
GRIDS = (2048, 4096)
METRIC_KEYS = fit20.METRIC_KEYS
GATE_LIMIT = 1.0e-4
SOLVER_STOP = 1.0e-6
SOLVER_STEPS = 5_000
TOTAL_START_ITERATION_LIMIT = 900_000
RECOVERY_TOLERANCE = 1.0e-12
SOURCE_COMMAND_TOLERANCE = 1.0e-6
LEGACY_PHASE_ERROR_CROSS_DEVICE_TOLERANCE = 1.0e-8
CONDITIONS = ("frozen_v2", "box_b5000", "fixed_phase_solver", "free_phase_solver")


def _require(condition: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not condition:
        raise error(message)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    fullpop._atomic_json(path, value)


def _atomic_torch(path: Path, value: Any) -> None:
    fullpop._atomic_torch(path, value)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    _require(path.resolve() == DEFAULT_CONFIG.resolve(),
             "feasibility diagnosis uses its fixed reviewed configuration", ValueError)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "feasibility configuration must be a mapping", ValueError)
    expected_indices = [200 * label + offset for label in range(CLASS_COUNT) for offset in (2, 3)]
    _require(config.get("schema") == SCHEMA and config.get("task_id") == TASK_ID
             and config.get("semantic_authority_id") == AUTHORITY
             and config.get("evidence_tier") == "diagnostic_only"
             and config.get("population") == {
                 "role": "development_feasibility20",
                 "source_role": "original_validation_excluding_fit20_rows_1980",
                 "source_validation_indices": expected_indices,
                 "positive_controls": "first_saved_known_reachable_target_per_class",
                 "no_resampling": True,
             }
             and config.get("solver") == {
                 "fixed_phase": {"starts": 1, "initial_amplitude": 0.5,
                                 "max_steps": SOLVER_STEPS, "stop_error": SOLVER_STOP,
                                 "phase_source": "box_b_unclipped_mapping"},
                 "free_phase": {"starts": 8, "initial_amplitude": 0.5,
                                 "initial_phases": "equally_spaced_2pi_over_8",
                                 "max_steps_per_start": SOLVER_STEPS, "stop_error": SOLVER_STOP},
                 "precision": "float64_complex128",
                 "max_total_start_iterations": TOTAL_START_ITERATION_LIMIT,
             }
             and config.get("measurement") == {
                 "grids": list(GRIDS), "conditions": list(CONDITIONS),
                 "max_relative_error": GATE_LIMIT,
                 "fixed_phase_target_max_relative_error": GATE_LIMIT,
                 "coefficient_alignment": "one_global_phase_per_sample",
                 "native_cmos_normalization": "none", "power_matching": False,
                 "gain_adjustment": False,
             }, "feasibility population, solver, or measurement contract changed", ValueError)
    expected_paths = {
        "experiment_root": RUNS_DIR, "preflight_root": PREFLIGHT_DIR,
        "attempt_root": ATTEMPT_DIR,
    }
    _require(all(_resolve(config["output"][key]) == value.resolve()
                 for key, value in expected_paths.items()),
             "feasibility output paths changed", ValueError)
    return config


def _set_numerics() -> None:
    projected._set_numerics()


def _run_identity(config: Mapping[str, Any], source_run_id: str, run_id: str) -> ResearchRunIdentity:
    semantics = json.dumps({"task_id": TASK_ID, "authority": AUTHORITY,
                            "source": config["source"], "population": config["population"],
                            "solver": config["solver"], "measurement": config["measurement"],
                            "source_run_id": source_run_id},
                           sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id, trajectory_id=f"{run_id}:common_phase_feasibility20",
        semantic_authority_id=AUTHORITY, data_role="DEVELOPMENT_FEASIBILITY20_NO_RESAMPLING",
        config_identity=dict(config),
        model_identity={"frozen_v2": "projected_mapping_v2_terminal_step_5000",
                        "box_b": "projected_box_fullpop_terminal_step_5000",
                        "optics": "C_seed_2026101501_step_45000_frozen",
                        "solver": "bounded_float64_complex128_amplitude_plus_common_phase"},
        optimizer_identity={"kind": "none", "scientific_updates": 0,
                            "optical_updates": 0, "evaluator_updates": 0},
        sampler_identity={"source_validation_indices": config["population"]["source_validation_indices"],
                          "role": "development_feasibility20", "resampling": False},
        execution_source_semantics_id=semantics,
    )


def _verify_cloud_source_repair(config: Mapping[str, Any], inputs: Mapping[str, Any]
                                ) -> tuple[ResearchRunIdentity, dict[str, Any], torch.Tensor,
                                           dict[str, Any], dict[str, Any]]:
    source = config["source"]
    evidence_dir = _resolve(source["fullpop_cloud_evidence"])
    review_dir = _resolve(source["fullpop_cloud_repair"])
    source_manifest = fullpop._verify_attempt_manifest(evidence_dir)
    identity = fullpop._assert_attempt(evidence_dir)
    population, schedule = fullpop._load_saved_population(
        evidence_dir / "fixed_population_and_schedule.pt", expected_run_id=identity.run_id,
        expected_source_run_id=inputs["source_mapper"]["run_id"])
    review_manifest = fullpop._json(review_dir / "transfer_manifest.json")
    summary = fullpop._json(review_dir / "review_summary.json")
    _require(review_manifest.get("schema") == "lightgen_g1_projected_box_fullpop_review_transfer.v1"
             and review_manifest.get("task_id") == fullpop.TASK_ID
             and review_manifest.get("run_id") == identity.run_id
             and review_manifest.get("file_count") == len(review_manifest.get("files", []))
             and review_manifest.get("bytes_total") == sum(
                 int(item["bytes"]) for item in review_manifest.get("files", [])),
             "cloud source repair transfer manifest differs from the full-population attempt")
    manifest_sizes: dict[str, int] = {}
    for item in review_manifest["files"]:
        relative = Path(str(item.get("path", "")))
        key = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and key not in manifest_sizes,
                 "cloud source repair manifest has an unsafe or duplicate path")
        manifest_sizes[key] = int(item["bytes"])
        path = review_dir / relative
        _require(path.is_file() and path.stat().st_size == manifest_sizes[key],
                 f"cloud source repair differs from its manifest: {key}")
    _require("transfer_manifest.json" not in manifest_sizes,
             "cloud source repair manifest must not list itself")
    manifest_sizes["transfer_manifest.json"] = (review_dir / "transfer_manifest.json").stat().st_size
    source_states = fullpop._read_review_source_states(
        review_dir, evidence_dir, inputs, identity, schedule)
    replay_check = fullpop._check_repair_replays(evidence_dir, review_dir, source_states)
    receipt_path = review_dir / "windows_review_receipt.json"
    _require(receipt_path.is_file(), "Windows independent source review receipt was not returned to AutoDL",
             FileNotFoundError)
    windows_receipt = fullpop._json(receipt_path)
    size_report = windows_receipt.get("repair_file_size_reconciliation", {})
    reported_items = size_report.get("files", [])
    reported_files = {str(item.get("path")): item for item in reported_items}
    _require(size_report.get("status") == "COMPLETE"
             and size_report.get("file_count") == len(manifest_sizes)
             and len(reported_files) == len(reported_items)
             and set(reported_files) == set(manifest_sizes)
             and all(item.get("remote_bytes") == size
                     and item.get("receipt_local_bytes") == size
                     and item.get("current_bytes") == size
                     for path, size in manifest_sizes.items()
                     for item in (reported_files[path],)),
             "Windows source repair receipt does not reconcile the transferred bytes")
    _require(windows_receipt.get("schema") ==
             "lightgen_g1_projected_box_fullpop_windows_review_receipt.v1"
             and windows_receipt.get("task_id") == fullpop.TASK_ID
             and windows_receipt.get("run_id") == identity.run_id
             and windows_receipt.get("status") == "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED"
             and windows_receipt.get("source_bound_command_replay", {}).get("rows") == fullpop.HOLDOUT_ROWS,
             "Windows source-binding review did not accept the repaired full-population evidence")
    recovery = fullpop._recompute_recovery_raw(
        evidence_dir, identity, inputs["source_mapper"]["run_id"], schedule)
    _require(summary.get("status") == "SOURCE_REPAIR_AND_COMMAND_REPLAY_PASSED"
             and summary.get("scientific_updates") == 0
             and replay_check.get("rows") == fullpop.HOLDOUT_ROWS
             and recovery.get("status") == "RAW_RECOVERY_RECOMPUTATION_PASSED",
             "cloud full-population repair or recovery evidence is incomplete")
    return identity, population, schedule, source_states, {
        "cloud_manifest": source_manifest, "repair_summary": summary,
        "windows_review_receipt": windows_receipt, "replay_check": replay_check,
        "recovery": recovery,
    }


def _load_sources(config: Mapping[str, Any], host: str) -> dict[str, Any]:
    _require(host in {"cloud", "windows"}, "source host must be explicit")
    full_config = fullpop._load_config(_resolve(config["source"]["fullpop_config"]))
    inputs = fullpop._load_inputs(full_config)
    if host == "cloud":
        identity, population, schedule, source_states, review = _verify_cloud_source_repair(config, inputs)
        evidence_dir = _resolve(config["source"]["fullpop_cloud_evidence"])
        review_dir = _resolve(config["source"]["fullpop_cloud_repair"])
    else:
        evidence_dir = _resolve(config["source"]["fullpop_windows_evidence"])
        review_dir = _resolve(config["source"]["fullpop_windows_repair"])
        receipt = fullpop._verify_saved(evidence_dir, inputs, review_dir)
        identity = fullpop._assert_attempt(evidence_dir)
        population, schedule = fullpop._load_saved_population(
            evidence_dir / "fixed_population_and_schedule.pt", expected_run_id=identity.run_id,
            expected_source_run_id=inputs["source_mapper"]["run_id"])
        source_states = fullpop._read_review_source_states(
            review_dir, evidence_dir, inputs, identity, schedule)
        review = {"windows_review_receipt": receipt}
    return {"fullpop_config": full_config, "inputs": inputs, "identity": identity,
            "population": population, "schedule": schedule, "source_states": source_states,
            "review": review, "evidence_dir": evidence_dir, "review_dir": review_dir,
            "host": host}


def _target_payload(config: Mapping[str, Any], source: Mapping[str, Any],
                    device: torch.device) -> dict[str, Any]:
    development = source["population"]["development"]
    desired = torch.tensor(config["population"]["source_validation_indices"], dtype=torch.long)
    indices = development["source_validation_indices"].to(torch.long)
    positions = {int(value): index for index, value in enumerate(indices.tolist())}
    _require(set(desired.tolist()).issubset(positions),
             "fixed development targets are missing from the reviewed source population")
    selected = torch.tensor([positions[int(value)] for value in desired.tolist()], dtype=torch.long)
    targets = development["coefficients"][selected].clone()
    labels = development["labels"][selected].clone()
    proposals = development["proposal_ids"][selected].clone()
    _require(tuple(targets.shape) == (TARGET_COUNT, PORT_COUNT)
             and torch.equal(labels, torch.arange(CLASS_COUNT).repeat_interleave(2))
             and torch.is_complex(targets) and torch.isfinite(targets.real).all().item()
             and torch.isfinite(targets.imag).all().item(),
             "selected development feasibility population is malformed")
    mapper = projected._new_mapper(device, source["source_states"]["box_b_candidate"]).eval()
    phases = []
    for target in targets:
        features = projected._normalized_features(target.reshape(1, -1), source["inputs"]["scale"], device)
        target_device = target.reshape(1, -1).to(device=device, dtype=torch.complex128)
        with torch.no_grad():
            _, details = projected._map_amplitude(
                mapper, features, target_device,
                source["inputs"]["matrix"].to(device=device, dtype=torch.float64),
                source["inputs"]["inverse"].to(device=device, dtype=torch.float64),
                source["inputs"]["scale"].to(device=device, dtype=torch.float64),
                include_preclip=True)
        phases.append(float(details["phase"][0].detach().cpu()))
    return {"schema": "lightgen_g1_fullpop_feasibility_targets.v1",
            "task_id": TASK_ID, "source_task_id": fullpop.TASK_ID,
            "source_run_id": source["identity"].run_id,
            "population_role": "development_feasibility20",
            "source_role": "original_validation_excluding_fit20_rows_1980",
            "source_validation_indices": desired, "labels": labels,
            "proposal_ids": proposals, "coefficients": targets,
            "box_b_public_phase_rad": torch.tensor(phases, dtype=torch.float64),
            "used_for_training": False, "resampled": False}


def _spectral_lipschitz(matrix: torch.Tensor) -> float:
    return reachability._projected_spectral_norm(matrix)


def _fixed_phase_relative_error(actual: torch.Tensor, target: torch.Tensor) -> float:
    actual = actual.reshape(-1).to(dtype=torch.complex128)
    target = target.reshape(-1).to(device=actual.device, dtype=torch.complex128)
    denominator = torch.linalg.vector_norm(target)
    _require(torch.isfinite(denominator).item() and denominator.item() > 0.0,
             "fixed-phase target has zero or nonfinite norm")
    return float(torch.linalg.vector_norm(actual - target) / denominator)


def _phase_shifted_target(target: torch.Tensor, phase: float) -> torch.Tensor:
    target_complex = target.reshape(-1).to(dtype=torch.complex128)
    phase_factor = torch.exp(torch.tensor(
        1j * float(phase), dtype=torch.complex128, device=target_complex.device))
    return target_complex * phase_factor


def _fixed_phase_grid_errors(outputs: Mapping[int, Mapping[str, torch.Tensor]],
                             target: torch.Tensor, phase: float) -> dict[str, float]:
    shifted_target = _phase_shifted_target(target, phase)
    return {str(grid): _fixed_phase_relative_error(
        outputs[grid]["coefficients"][0], shifted_target) for grid in GRIDS}


def _fixed_phase_solve(matrix: torch.Tensor, target: torch.Tensor, phase: float, *,
                       max_steps: int = SOLVER_STEPS,
                       initial_state: Mapping[str, Any] | None = None,
                       checkpoint: Any = None,
                       checkpoint_interval: int = 500) -> dict[str, Any]:
    _require(matrix.dtype == torch.complex128 and target.dtype == torch.complex128
             and matrix.ndim == 2 and target.ndim == 1 and matrix.shape[0] == target.numel(),
             "fixed-phase solver requires a complex128 response and target")
    _require(0 < max_steps <= SOLVER_STEPS, "fixed-phase solver budget is outside its fixed limit", ValueError)
    device, pixels = matrix.device, matrix.shape[1]
    lipschitz = _spectral_lipschitz(matrix)
    target_norm = torch.linalg.vector_norm(target)
    aligned = target * torch.exp(torch.tensor(1j * phase, dtype=torch.complex128, device=device))
    if initial_state is None:
        amplitude = torch.full((pixels,), 0.5, dtype=torch.float64, device=device)
        best_amplitude = amplitude.clone()
        best_error = torch.tensor(float("inf"), dtype=torch.float64, device=device)
        best_loss = float("inf")
        iteration = 0
    else:
        amplitude = initial_state["amplitude"].to(device=device, dtype=torch.float64)
        best_amplitude = initial_state["best_amplitude"].to(device=device, dtype=torch.float64)
        best_error = torch.as_tensor(initial_state["best_error"], dtype=torch.float64, device=device)
        best_loss = float(initial_state["loss"])
        iteration = int(initial_state["iteration"])
        _require(tuple(amplitude.shape) == (pixels,) and tuple(best_amplitude.shape) == (pixels,)
                 and 0 <= iteration <= max_steps and torch.isfinite(amplitude).all().item()
                 and torch.isfinite(best_amplitude).all().item()
                 and torch.isfinite(best_error).item() and math.isfinite(best_loss)
                 and float(best_error) >= 0.0
                 and 0.0 <= float(amplitude.min()) <= float(amplitude.max()) <= 1.0
                 and 0.0 <= float(best_amplitude.min()) <= float(best_amplitude.max()) <= 1.0,
                 "fixed-phase solver checkpoint is invalid")
    _require(torch.isfinite(target_norm).item() and target_norm.item() > 0.0,
             "fixed-phase target has zero or nonfinite norm")
    while iteration < max_steps:
        actual = amplitude.to(torch.complex128).reshape(1, -1) @ matrix.T
        residual = actual.reshape(-1) - aligned
        gradient = (residual.reshape(1, -1) @ matrix.conj()).real.reshape(-1)
        amplitude = torch.clamp(amplitude - gradient / lipschitz, 0.0, 1.0)
        actual = amplitude.to(torch.complex128).reshape(1, -1) @ matrix.T
        residual = actual.reshape(-1) - aligned
        error = torch.linalg.vector_norm(residual) / target_norm
        loss = 0.5 * torch.linalg.vector_norm(residual).square()
        if error < best_error:
            best_error = error.detach().clone()
            best_amplitude = amplitude.detach().clone()
            best_loss = float(loss.detach().cpu())
        iteration += 1
        if checkpoint is not None and iteration % checkpoint_interval == 0:
            checkpoint({"amplitude": amplitude.detach().cpu().clone(),
                        "best_amplitude": best_amplitude.detach().cpu().clone(),
                        "best_error": best_error.detach().cpu().clone(),
                        "iteration": iteration, "loss": best_loss})
        if float(best_error) <= SOLVER_STOP:
            break
    state = {"amplitude": amplitude.detach().cpu().clone(),
             "best_amplitude": best_amplitude.detach().cpu().clone(),
             "best_error": best_error.detach().cpu().clone(), "iteration": iteration,
             "loss": best_loss}
    return {"amplitude": best_amplitude.detach().cpu(), "phase_rad": float(phase),
            "relative_error": float(best_error.detach().cpu()), "iterations": iteration,
            "loss": best_loss, "state": state}


def _analytic_controls() -> dict[str, Any]:
    matrix = torch.eye(2, dtype=torch.complex128)
    command = torch.tensor([0.23, 0.78], dtype=torch.float64)
    target = matrix @ command.to(torch.complex128)
    fixed = _fixed_phase_solve(matrix, target, 0.0, max_steps=20)
    phase = 0.73
    rotated = target * torch.exp(torch.tensor(1j * phase, dtype=torch.complex128))
    fixed_rotated = _fixed_phase_solve(matrix, rotated, 0.0, max_steps=100)
    free = reachability._bounded_phase_solve(matrix, rotated, starts=8, max_steps=5000,
                                             stop_error=SOLVER_STOP)
    excessive = _fixed_phase_solve(matrix, torch.tensor([1.5 + 0j, 0j], dtype=torch.complex128),
                                   0.0, max_steps=50)
    direction_matrix = torch.tensor([[1.0 + 0.2j, 0.3 - 0.1j],
                                     [0.1 + 0.4j, 0.8 + 0.2j]], dtype=torch.complex128)
    direction_target = torch.tensor([0.4 + 0.2j, 0.3 - 0.1j], dtype=torch.complex128)
    point = torch.tensor([0.37, 0.64], dtype=torch.float64)
    target_fixed = direction_target * torch.exp(torch.tensor(1j * 0.31, dtype=torch.complex128))
    residual = point.to(torch.complex128) @ direction_matrix.T - target_fixed
    analytic = (residual @ direction_matrix.conj()).real
    direction = torch.tensor([0.6, -0.8], dtype=torch.float64)
    epsilon = 1.0e-6

    def objective(value: torch.Tensor) -> torch.Tensor:
        difference = value.to(torch.complex128) @ direction_matrix.T - target_fixed
        return 0.5 * difference.abs().square().sum()

    finite_difference = (objective(point + epsilon * direction)
                         - objective(point - epsilon * direction)) / (2.0 * epsilon)
    analytic_direction = torch.dot(analytic, direction)
    descent = point - 1.0e-2 * analytic
    derivative_error = abs(float(finite_difference - analytic_direction))
    boundary_point = torch.tensor([0.0, 1.0], dtype=torch.float64)
    boundary_residual = boundary_point.to(torch.complex128) @ direction_matrix.T - target_fixed
    boundary_gradient = (boundary_residual @ direction_matrix.conj()).real
    inward_direction = torch.tensor([1.0, -1.0], dtype=torch.float64)
    one_sided = (objective(boundary_point + epsilon * inward_direction)
                 - objective(boundary_point)) / epsilon
    boundary_derivative_error = abs(float(one_sided - torch.dot(boundary_gradient, inward_direction)))
    _require(fixed["relative_error"] <= SOLVER_STOP
             and fixed_rotated["relative_error"] > 1.0e-4
             and free["relative_error"] <= SOLVER_STOP
             and excessive["relative_error"] > 1.0e-4
             and derivative_error <= 1.0e-5
             and boundary_derivative_error <= 1.0e-5
             and float(objective(descent)) < float(objective(point)),
             "analytic fixed/free phase, box boundary, derivative, or descent control failed")
    return {"known_box_command_error": fixed["relative_error"],
            "fixed_phase_rotated_target_error": fixed_rotated["relative_error"],
            "free_phase_rotated_target_error": free["relative_error"],
            "out_of_box_target_error": excessive["relative_error"],
            "complex128_directional_derivative_abs_error": derivative_error,
            "complex128_boundary_one_sided_derivative_abs_error": boundary_derivative_error,
            "descent_direction_decreases_objective": True,
            "all_commands_bounded": all(0.0 <= float(item["amplitude"].min())
                                         and float(item["amplitude"].max()) <= 1.0
                                         for item in (fixed, fixed_rotated, free, excessive))}


def _measure_commands(targets: torch.Tensor, commands: Mapping[str, torch.Tensor],
                      inputs: Mapping[str, Any], device: torch.device, *,
                      target_index_offset: int = 0
                      ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    count = int(targets.shape[0])
    outputs = {condition: {grid: {"coefficients": [], "raw_cmos": []} for grid in GRIDS}
               for condition in commands}
    direct = {grid: [] for grid in GRIDS}
    for grid in GRIDS:
        model = pilot._new_optical_model(inputs["checkpoint"], grid, device)
        for index in range(count):
            target = targets[index:index + 1].to(device=device, dtype=torch.complex128)
            with torch.no_grad():
                reference = projected._forward_target(model, target)
            direct[grid].append(reference["raw_cmos"].detach().cpu()[0])
            for condition, all_commands in commands.items():
                command = all_commands[index:index + 1].to(device=device, dtype=torch.float32)
                with torch.no_grad():
                    actual = projected._forward_amplitude(model, command)
                for key in ("coefficients", "raw_cmos"):
                    outputs[condition][grid][key].append(actual[key].detach().cpu()[0])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for condition in commands:
        for grid in GRIDS:
            outputs[condition][grid] = {
                key: torch.stack(value) for key, value in outputs[condition][grid].items()}
    direct = {grid: torch.stack(value) for grid, value in direct.items()}
    rows = []
    for index in range(count):
        per_grid = {condition: {grid: {key: value[index:index + 1]
                                       for key, value in outputs[condition][grid].items()}
                                for grid in GRIDS} for condition in commands}
        direct_row = {grid: direct[grid][index:index + 1] for grid in GRIDS}
        for condition in commands:
            row = projected._error_row(per_grid[condition], targets[index:index + 1], direct_row, 0)
            rows.append({"target_index": target_index_offset + index, "condition": condition,
                         **row, "row_passed": all(row[key] <= GATE_LIMIT for key in METRIC_KEYS)})
    return outputs, direct, rows


def _repair_measurement_row_identity(payload: dict[str, Any], *, section: str,
                                     target_index: int, conditions: Sequence[str],
                                     repair_rows: list[dict[str, Any]]) -> bool:
    _require(section in {"baseline", "conditions"}
             and payload.get("target_index") == target_index,
             "measurement row repair is not bound to its outer target", ValueError)
    changed = False
    label = int(payload["label"])
    source_index = int(payload["source_validation_index"])
    saved_rows = payload.get("rows")
    if isinstance(saved_rows, list):
        _require(all(isinstance(row, dict) for row in saved_rows)
                 and len(saved_rows) == len(conditions),
                 f"measurement row list is malformed: {section}/{target_index}")
        row_map = {str(row.get("condition")): row for row in saved_rows}
        _require(len(row_map) == len(saved_rows),
                 f"measurement row list contains duplicate conditions: {section}/{target_index}")
    else:
        row_map = saved_rows
    _require(isinstance(row_map, dict) and set(row_map) == set(conditions),
             f"measurement row condition set is invalid: {section}/{target_index}")
    for condition in conditions:
        row = row_map.get(condition)
        _require(isinstance(row, dict) and row.get("condition") == condition,
                 f"measurement row condition identity is invalid: {section}/{target_index}/{condition}")
        old_index = int(row.get("target_index", -1))
        old_label = row.get("label")
        old_source_index = row.get("source_validation_index")
        _require((old_index == target_index or (target_index > 0 and old_index == 0))
                 and (old_label is None or int(old_label) == label)
                 and (old_source_index is None or int(old_source_index) == source_index),
                 f"measurement row identity cannot be safely repaired: {section}/{target_index}/{condition}")
        row_changed = (old_index != target_index or old_label is None or old_source_index is None)
        if row_changed:
            entry = {"section": section, "target_index": target_index,
                     "condition": condition, "old_target_index": old_index,
                     "new_target_index": target_index, "old_label": old_label,
                     "new_label": label, "old_source_validation_index": old_source_index,
                     "new_source_validation_index": source_index}
            matching = [item for item in repair_rows
                        if item.get("section") == section
                        and item.get("target_index") == target_index
                        and item.get("condition") == condition]
            _require(len(matching) <= 1,
                     f"measurement row repair already contains duplicate identity: {section}/{target_index}/{condition}")
            if matching:
                _require(matching[0] == entry,
                         f"measurement row repair changed across resume: {section}/{target_index}/{condition}")
            else:
                repair_rows.append(entry)
            row["target_index"] = target_index
            row["label"] = label
            row["source_validation_index"] = source_index
            changed = True
    return changed


def _positive_controls(source: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    witness_path_key = ("fit20_cloud_witness_measurements" if source["host"] == "cloud"
                        else "fit20_witness_measurements")
    witness_path = _resolve(source["fullpop_config"]["source"][witness_path_key])
    witness = torch.load(witness_path, map_location="cpu", weights_only=False)
    known = source["population"]["known_reachable"]
    amplitudes = witness["amplitudes"][::2].to(torch.float32).contiguous()
    targets = known["coefficients"][::2].contiguous()
    labels = known["labels"][::2].contiguous()
    source_indices = known["source_validation_indices"][::2].contiguous()
    _require(witness.get("population_role") == "known_reachable_fit20"
             and torch.equal(witness.get("target_coefficients"), known["coefficients"])
             and torch.equal(labels, torch.arange(CLASS_COUNT))
             and tuple(amplitudes.shape) == (CLASS_COUNT, 1, 200, 200),
             "positive-control commands differ from the saved known-reachable witnesses")
    command_map = {"known_reachable_witness": amplitudes}
    outputs, direct, rows = _measure_commands(targets, command_map, source["inputs"], device)
    qualified = []
    for index in range(CLASS_COUNT):
        row = next(row for row in rows if row["target_index"] == index)
        row = {**row, "label": int(labels[index]),
               "source_validation_index": int(source_indices[index])}
        qualified.append(row)
    passed = sum(bool(row["row_passed"]) for row in qualified)
    _require(passed == CLASS_COUNT,
             "one-per-class saved positive-control witness failed the full optical six-metric check")
    return {"status": "TEN_POSITIVE_CONTROLS_PASSED", "count": CLASS_COUNT,
            "passed": passed, "all_six_metrics_recomputed": True,
            "training_labels": False, "inference_fallback": False,
            "rows": qualified, "outputs": outputs, "direct_raw_cmos": direct,
            "amplitudes": amplitudes, "target_coefficients": targets,
            "labels": labels, "source_validation_indices": source_indices}


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    before = fullpop._rng_snapshot()
    _atomic_torch(path, copy.deepcopy(dict(payload)))
    _require(fullpop._rng_equal(before, fullpop._rng_snapshot()),
             "solver checkpoint serialization changed process RNG")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    _require(saved.get("task_id") == TASK_ID and saved.get("run_id") == payload.get("run_id")
             and saved.get("target_cursor") == payload.get("target_cursor")
             and saved.get("branch") == payload.get("branch")
             and saved.get("iteration") == payload.get("iteration")
             and saved.get("committed_start_iterations") == payload.get("committed_start_iterations")
             and saved.get("training_updates") == 0 and saved.get("optimizer_calls") == 0
             and fullpop._state_dicts_equal(saved.get("solver_state"), payload.get("solver_state"))
             and fullpop._state_dicts_equal(saved.get("completed_target_results"),
                                            payload.get("completed_target_results")),
             "saved solver checkpoint differs from its state cursor")


def _estimate_diagnostic_storage_bytes(preflight_bytes: int) -> int:
    one_amplitude = PIXEL_COUNT * torch.tensor([], dtype=torch.float64).element_size()
    fixed_state = 2 * one_amplitude
    completed_result = 2 * one_amplitude + 8 * torch.tensor([], dtype=torch.float64).element_size()
    free_state = 2 * CLASS_COUNT * one_amplitude + 3 * CLASS_COUNT * 8
    checkpoint_steps = SOLVER_STEPS // 500
    solver_bytes = 0
    for target_index in range(TARGET_COUNT):
        solver_bytes += checkpoint_steps * (target_index * completed_result + fixed_state)
        solver_bytes += target_index * completed_result + one_amplitude
        solver_bytes += checkpoint_steps * (target_index * completed_result + one_amplitude + free_state)
        solver_bytes += (target_index + 1) * completed_result
    solution_bytes = TARGET_COUNT * completed_result
    latest_checkpoint_bytes = (TARGET_COUNT * completed_result + one_amplitude + free_state)
    raw_measurement_bytes = TARGET_COUNT * (10 * PIXEL_COUNT * 8 + 4 * PIXEL_COUNT * 4)
    baseline_bytes = TARGET_COUNT * (6 * PIXEL_COUNT * 8 + 2 * PIXEL_COUNT * 4)
    evidence_margin = 512 * 1024 * 1024
    return (preflight_bytes + solver_bytes + solution_bytes + latest_checkpoint_bytes
            + raw_measurement_bytes + baseline_bytes + evidence_margin)


def _recovery_worker(config_path: Path, checkpoint_path: Path, output_path: Path,
                     evidence_dir: Path) -> None:
    config = _load_config(config_path)
    _require(evidence_dir.resolve() == PREFLIGHT_DIR.resolve(),
             "solver recovery worker requires the fixed preflight evidence directory", ValueError)
    source = _load_sources(config, "cloud")
    inputs = source["inputs"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    preflight_identity = _json(PREFLIGHT_DIR / "preflight_identity.json")
    _require(checkpoint.get("task_id") == TASK_ID and checkpoint.get("branch") == "fixed_phase"
             and checkpoint.get("iteration") == 3,
             "solver recovery worker requires the exact third-iteration fixed-phase checkpoint")
    targets = torch.load(evidence_dir / "fixed_targets.pt", map_location="cpu", weights_only=False)
    index = int(checkpoint["target_cursor"])
    _require(checkpoint.get("run_id") == preflight_identity.get("run_id")
             and checkpoint.get("source_run_id") == source["identity"].run_id
             and checkpoint.get("scientific_updates") == 0 and index == 0,
             "solver recovery checkpoint is not bound to this fixed target and source")
    device = torch.device("cuda:0")
    matrix = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    target = targets["coefficients"][index].to(device=device, dtype=torch.complex128)
    result = _fixed_phase_solve(matrix, target, float(targets["box_b_public_phase_rad"][index]),
                                max_steps=4, initial_state=checkpoint["solver_state"])
    payload = {"task_id": TASK_ID, "run_id": checkpoint["run_id"],
               "source_run_id": checkpoint["source_run_id"],
               "target_cursor": index, "branch": "fixed_phase",
               "iteration": result["iterations"], "state": result["state"],
               "relative_error": result["relative_error"], "amplitude": result["amplitude"]}
    _atomic_torch(output_path, payload)


def _solver_recovery(config_path: Path, inputs: Mapping[str, Any], targets: Mapping[str, Any],
                     run_id: str) -> dict[str, Any]:
    recovery_dir = PREFLIGHT_DIR / "solver_recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    matrix = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    index = 0
    target = targets["coefficients"][index].to(device=device, dtype=torch.complex128)
    phase = float(targets["box_b_public_phase_rad"][index])
    first = _fixed_phase_solve(matrix, target, phase, max_steps=3)
    checkpoint = {"schema": "lightgen_g1_fullpop_feasibility_solver_checkpoint.v1",
                  "task_id": TASK_ID, "run_id": run_id, "source_run_id": targets["source_run_id"],
                  "target_cursor": index, "branch": "fixed_phase", "iteration": 3,
                  "solver_state": first["state"], "completed_target_results": [],
                  "committed_start_iterations": 3, "training_updates": 0,
                  "optimizer_calls": 0, "scientific_updates": 0}
    path = recovery_dir / "checkpoint_target_000_fixed_000003.pt"
    _save_checkpoint(path, checkpoint)
    live = _fixed_phase_solve(matrix, target, phase, max_steps=4,
                              initial_state=checkpoint["solver_state"])
    live_path = recovery_dir / "original_process_step4.pt"
    _atomic_torch(live_path, {"task_id": TASK_ID, "run_id": run_id,
                              "source_run_id": targets["source_run_id"],
                              "target_cursor": index, "branch": "fixed_phase", "iteration": 4,
                              "state": live["state"], "relative_error": live["relative_error"],
                              "amplitude": live["amplitude"]})
    restored_path = recovery_dir / "independent_process_step4.pt"
    result = subprocess.run([
        sys.executable, "-m", MODULE_PATH, "--mode", "recovery-worker",
        "--config", str(config_path.resolve()), "--checkpoint", str(path.resolve()),
        "--evidence-dir", str(PREFLIGHT_DIR.resolve()), "--worker-output", str(restored_path.resolve()),
    ], cwd=ROOT, text=True, capture_output=True, check=False)
    _require(result.returncode == 0 and restored_path.is_file(),
             f"independent solver recovery process failed: {result.stderr[-2500:]}")
    live_saved = torch.load(live_path, map_location="cpu", weights_only=False)
    restored = torch.load(restored_path, map_location="cpu", weights_only=False)
    _require(live_saved.get("task_id") == restored.get("task_id") == TASK_ID
             and live_saved.get("run_id") == restored.get("run_id") == run_id
             and live_saved.get("source_run_id") == restored.get("source_run_id") == targets["source_run_id"]
             and live_saved.get("target_cursor") == restored.get("target_cursor") == index
             and live_saved.get("iteration") == restored.get("iteration") == 4
             and abs(float(live_saved.get("relative_error", math.inf))
                     - float(restored.get("relative_error", math.inf))) <= RECOVERY_TOLERANCE
             and fullpop._state_dicts_equal(live_saved.get("state"), live["state"]),
             "saved original-process fourth solver state differs from its live result")
    mapper_delta = float((live_saved["state"]["amplitude"] - restored["state"]["amplitude"]).abs().max())
    state_delta = max(float((live_saved["state"][key] - restored["state"][key]).abs().max())
                      for key in ("amplitude", "best_amplitude", "best_error"))
    loss_delta = abs(float(live_saved["state"]["loss"]) - float(restored["state"]["loss"]))
    _require(mapper_delta <= RECOVERY_TOLERANCE and state_delta <= RECOVERY_TOLERANCE
             and loss_delta <= RECOVERY_TOLERANCE
             and live_saved["state"]["iteration"] == restored["state"]["iteration"]
             and live_saved["iteration"] == restored["iteration"] == 4,
             "original and restored fourth solver iteration do not agree")
    return {"status": "SOLVER_RECOVERY_PASSED", "initial_iterations": 3,
            "original_fourth_iteration": 4, "restored_fourth_iteration": 4,
            "state_max_abs_delta": state_delta, "loss_abs_delta": loss_delta,
            "iteration_exact": True, "rng_preserved_by_save": True,
            "adam_calls": 0, "scientific_updates": 0}


def _preflight(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "feasibility preflight requires AutoDL CUDA")
    _require(not ATTEMPT_DIR.exists(), "feasibility attempt path already exists", FileExistsError)
    source = _load_sources(config, "cloud")
    previous = None
    identity_path = PREFLIGHT_DIR / "preflight_identity.json"
    if PREFLIGHT_DIR.exists():
        _require(identity_path.is_file(),
                 "existing feasibility preflight has no resumable identity", FileExistsError)
        previous = _json(identity_path)
        summary_path = PREFLIGHT_DIR / "preflight_summary.json"
        _require(not summary_path.is_file() or _json(summary_path).get("status") != "READY",
                 "feasibility preflight is already READY and terminal", FileExistsError)
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = str(previous["run_id"]) if previous else f"g1phase20_{uuid.uuid4().hex}"
    identity = _run_identity(config, source["identity"].run_id, run_id)
    if previous is not None:
        previous_identity = ResearchRunIdentity.from_mapping(previous["run_identity"])
        _require(previous.get("task_id") == TASK_ID
                 and previous.get("source_run_id") == source["identity"].run_id
                 and previous_identity.execution_semantics() == identity.execution_semantics(),
                 "existing preflight identity is bound to different sources or task semantics")
    _atomic_json(identity_path, {"task_id": TASK_ID, "run_id": run_id,
                                 "source_run_id": source["identity"].run_id,
                                 "run_identity": identity.to_dict()})
    _atomic_json(PREFLIGHT_DIR / "run_config_snapshot.yaml", config)
    targets = _target_payload(config, source, torch.device("cuda:0"))
    _atomic_torch(PREFLIGHT_DIR / "fixed_targets.pt", targets)
    _atomic_torch(PREFLIGHT_DIR / "mapper_source_states.pt", {
        "schema": "lightgen_g1_fullpop_feasibility_mapper_sources.v1",
        "task_id": TASK_ID, "run_id": run_id,
        "source_run_id": source["identity"].run_id,
        "mapper_states": source["source_states"],
    })
    analytic = _analytic_controls()
    positive = _positive_controls(source, torch.device("cuda:0"))
    _atomic_torch(PREFLIGHT_DIR / "positive_control_measurements.pt", {
        key: value for key, value in positive.items() if key != "status"})
    recovery = _solver_recovery(config_path, source["inputs"], targets, run_id)
    disk = shutil.disk_usage(RUNS_DIR.drive or ROOT)
    memory = torch.cuda.mem_get_info(torch.device("cuda:0"))
    stored_bytes = sum(path.stat().st_size for path in PREFLIGHT_DIR.rglob("*") if path.is_file())
    estimated_peak = _estimate_diagnostic_storage_bytes(stored_bytes)
    _require(disk.free >= estimated_peak,
             "measured AutoDL free space does not cover preflight evidence and diagnostic output estimate",
             OSError)
    correctness = {"passed": True, "status": "PASS", "source_repair_windows_reconsumed": True,
                   "population_identity_and_roles_verified": True,
                   "analytic_box_and_common_phase_controls_passed": True,
                   "complex128_derivative_and_descent_passed": True,
                   "ten_known_witnesses_passed_full_optics": True,
                   "solver_restore_fourth_iteration_passed": True,
                   "scientific_updates": 0, "optimizer_calls": 0}
    resource = {"passed": True, "status": "PASS", "device": torch.cuda.get_device_name(0),
                "free_memory_bytes_observation": int(memory[0]),
                "total_memory_bytes_observation": int(memory[1]),
                "free_memory_is_gate": False,
                "data_disk_free_bytes_observation": int(disk.free),
                "preflight_evidence_bytes": stored_bytes,
                "estimated_diagnostic_peak_bytes_with_observation_margin": estimated_peak}
    implementation = {"passed": True, "status": "PASS",
                      "selected_engine": "CUDA_float64_complex128_bounded_solver_and_full_optics",
                      "fixed_and_free_phase_paths_checked": True,
                      "inference_fallback": False, "training_updates": 0,
                      "matrix_shape": list(source["inputs"]["response_matrix"].shape),
                      "total_start_iteration_limit": TOTAL_START_ITERATION_LIMIT}
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    receipt = control.make_ready_receipt(correctness=correctness, resource=resource,
                                         implementation=implementation,
                                         selected_engine=implementation["selected_engine"],
                                         resource_envelope=resource)
    control.mark_preflight_ready(receipt)
    summary = {"schema": "lightgen_g1_fullpop_feasibility_preflight.v1",
               "task_id": TASK_ID, "run_id": run_id, "status": "READY",
               "source_run_id": source["identity"].run_id,
               "source_review": source["review"],
               "target_count": TARGET_COUNT,
               "source_validation_indices": targets["source_validation_indices"].tolist(),
               "analytic_controls": analytic, "positive_controls": {
                   "status": positive["status"], "count": positive["count"],
                   "passed": positive["passed"], "rows": positive["rows"]},
               "solver_recovery": recovery, "resource_observations": resource,
               "scientific_updates": 0, "optimizer_calls": 0}
    _atomic_json(PREFLIGHT_DIR / "preflight_receipt.json", receipt.to_dict())
    _atomic_json(PREFLIGHT_DIR / "preflight_summary.json", summary)
    return {"status": "READY", "run_id": run_id,
            "scientific_updates": 0, "optimizer_calls": 0,
            "estimated_peak_bytes": estimated_peak}


def _load_ready_control() -> tuple[ResearchRunControl, PreflightReceipt]:
    identity_payload = _json(PREFLIGHT_DIR / "preflight_identity.json")
    identity = ResearchRunIdentity.from_mapping(identity_payload["run_identity"])
    receipt = PreflightReceipt.from_mapping(_json(PREFLIGHT_DIR / "preflight_receipt.json"))
    summary = _json(PREFLIGHT_DIR / "preflight_summary.json")
    _require(identity_payload.get("task_id") == TASK_ID and summary.get("task_id") == TASK_ID
             and summary.get("run_id") == identity.run_id and summary.get("status") == "READY"
             and summary.get("scientific_updates") == 0
             and receipt.status == "READY" and receipt.authorization_id == AUTHORITY
             and receipt.source_semantics_id == identity.execution_source_semantics_id,
             "feasibility execution requires its complete READY preflight receipt")
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    return control, receipt


def _assert_attempt(attempt_dir: Path) -> ResearchRunIdentity:
    identity = ResearchRunIdentity.from_mapping(_json(attempt_dir / "run_identity.json"))
    expected_payload = _json(attempt_dir / "preflight_evidence" / "preflight_identity.json")
    expected = ResearchRunIdentity.from_mapping(expected_payload["run_identity"])
    expected = expected.with_scientific_attempt(f"{expected.run_id}:attempt_01")
    _require(identity.execution_semantics() == expected.execution_semantics()
             and expected_payload.get("task_id") == TASK_ID,
             "feasibility attempt differs from its READY identity")
    return identity


def _copy_preflight(attempt_dir: Path) -> None:
    target = attempt_dir / "preflight_evidence"
    target.mkdir(exist_ok=False)
    for path in PREFLIGHT_DIR.rglob("*"):
        if path.is_file():
            destination = target / path.relative_to(PREFLIGHT_DIR)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def _task_state(state: str, *, cursor: int = 0, solver_iterations: int = 0) -> None:
    identity = _assert_attempt(ATTEMPT_DIR)
    _atomic_json(ATTEMPT_DIR / "task_state.json", {
        "schema": "lightgen_g1_fullpop_feasibility_state.v1", "task_id": TASK_ID,
        "run_id": identity.run_id, "state": state, "target_cursor": cursor,
        "committed_start_iterations": solver_iterations,
        "scientific_updates": 0, "optimizer_calls": 0,
        "optical_updates": 0, "evaluator_updates": 0, "updated_at_unix": time.time(),
    })


def _one_mapper_command(mapper_state: Mapping[str, torch.Tensor], target: torch.Tensor,
                        inputs: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, float,
                                                                                torch.Tensor]:
    mapper = projected._new_mapper(device, mapper_state).eval()
    features = projected._normalized_features(target.reshape(1, -1), inputs["scale"], device)
    target_device = target.reshape(1, -1).to(device=device, dtype=torch.complex128)
    with torch.no_grad():
        command, details = projected._map_amplitude(
            mapper, features, target_device,
            inputs["matrix"].to(device=device, dtype=torch.float64),
            inputs["inverse"].to(device=device, dtype=torch.float64),
            inputs["scale"].to(device=device, dtype=torch.float64), include_preclip=True)
    return command.detach().cpu().to(torch.float32), float(details["phase"][0].cpu()), \
        details["preclip_amplitude"].detach().cpu().to(torch.float64)


def _require_exact_command_replay(replayed: torch.Tensor, saved: torch.Tensor,
                                  target_index: int, condition: str) -> None:
    _require(torch.is_tensor(replayed) and torch.is_tensor(saved)
             and replayed.dtype == torch.float32 and saved.dtype == torch.float32
             and tuple(replayed.shape) == tuple(saved.shape)
             and torch.isfinite(replayed).all().item()
             and torch.equal(replayed, saved),
             f"saved mapper command differs from exact CUDA replay at {target_index}/{condition}")


def _residual_decomposition(response: torch.Tensor, target: torch.Tensor, phase: float,
                            preclip: torch.Tensor, amplitude: torch.Tensor,
                            actual_coefficients: torch.Tensor) -> dict[str, Any]:
    v = preclip.reshape(-1).to(device=response.device, dtype=torch.float64)
    clipped = v.clamp(0.0, 1.0)
    a = amplitude.reshape(-1).to(device=response.device, dtype=torch.float32).double()
    target_phase = target.to(device=response.device, dtype=torch.complex128) * torch.exp(
        torch.tensor(1j * phase, dtype=torch.complex128, device=response.device))

    def apply(value: torch.Tensor) -> torch.Tensor:
        return value.to(torch.complex128).reshape(1, -1) @ response.T

    terms = {
        "preclip": (apply(v) - target_phase.reshape(1, -1)).reshape(-1),
        "box_clip": apply(clipped - v).reshape(-1),
        "float32_rounding": apply(a - clipped).reshape(-1),
        "optical_response": (actual_coefficients.to(response.device, dtype=torch.complex128).reshape(1, -1)
                             - apply(a)).reshape(-1),
    }
    total = sum(terms.values())
    actual_residual = actual_coefficients.to(response.device, dtype=torch.complex128) - target_phase
    cross = {}
    names = list(terms)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            cross[f"{left}__{right}"] = float(2.0 * torch.vdot(terms[left], terms[right]).real.cpu())
    summed_norm_sq = sum(float(value.abs().square().sum().cpu()) for value in terms.values())
    identity_norm_sq = summed_norm_sq + sum(cross.values())
    return {"phase_rad": float(phase),
            "term_vectors": {key: value.detach().cpu() for key, value in terms.items()},
            "term_norm_sq": {key: float(value.abs().square().sum().cpu())
                             for key, value in terms.items()},
            "cross_terms_twice_real_inner_product": cross,
            "actual_fixed_phase_residual_norm_sq": float(actual_residual.abs().square().sum().cpu()),
            "expanded_residual_norm_sq": identity_norm_sq,
            "identity_vector_max_abs_error": float((total - actual_residual).abs().max().cpu()),
            "identity_norm_sq_abs_error": abs(identity_norm_sq -
                                                float(actual_residual.abs().square().sum().cpu())),
            "preclip_out_of_box_fraction": float(((v < 0.0) | (v > 1.0)).double().mean().cpu())}


def _baseline_measurement(source: Mapping[str, Any], targets: Mapping[str, Any],
                          attempt_dir: Path, device: torch.device) -> dict[str, Any]:
    inputs = source["inputs"]
    response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    for index in range(TARGET_COUNT):
        path = attempt_dir / "measurements" / "baseline" / f"target_{index:02d}.pt"
        if path.is_file():
            saved = torch.load(path, map_location="cpu", weights_only=False)
            _require(saved.get("target_index") == index
                     and torch.equal(saved.get("target_coefficients"), targets["coefficients"][index]),
                     f"saved baseline target identity changed at {index}")
            continue
        target = targets["coefficients"][index]
        v2_command, _, _ = _one_mapper_command(source["source_states"]["frozen_v2_baseline"],
                                                target, inputs, device)
        box_command, phase, preclip = _one_mapper_command(source["source_states"]["box_b_candidate"],
                                                           target, inputs, device)
        commands = {"frozen_v2": v2_command, "box_b5000": box_command}
        outputs, direct, rows = _measure_commands(target.reshape(1, -1),
                                                  commands,
                                                  inputs, device, target_index_offset=index)
        for row in rows:
            row["label"] = int(targets["labels"][index])
            row["source_validation_index"] = int(targets["source_validation_indices"][index])
        decomposition = _residual_decomposition(
            response, target, phase, preclip, box_command,
            outputs["box_b5000"][2048]["coefficients"][0])
        _require(decomposition["identity_vector_max_abs_error"] <= 1.0e-10,
                 f"complex residual decomposition does not close at target {index}")
        payload = {"schema": "lightgen_g1_fullpop_feasibility_baseline_target.v1",
                   "task_id": TASK_ID, "source_run_id": source["identity"].run_id,
                   "target_index": index,
                   "source_validation_index": int(targets["source_validation_indices"][index]),
                   "label": int(targets["labels"][index]),
                   "proposal_id": int(targets["proposal_ids"][index]),
                   "target_coefficients": target.clone(),
                   "commands": {key: value.clone() for key, value in commands.items()},
                   "b_public_phase_rad": phase, "b_preclip_amplitude": preclip,
                   "outputs": outputs, "direct_raw_cmos": direct,
                   "rows": rows, "residual_decomposition": decomposition}
        _atomic_torch(path, payload)
    return {"status": "ZERO_UPDATE_BASELINE_COMPLETE", "targets": TARGET_COUNT,
            "conditions": ["frozen_v2", "box_b5000"], "optical_updates": 0,
            "evaluator_updates": 0}


def _solver_checkpoint_payload(identity: ResearchRunIdentity, targets: Mapping[str, Any],
                               results: Sequence[Mapping[str, Any]], cursor: int,
                               branch: str | None, state: Mapping[str, Any] | None,
                               iteration: int) -> dict[str, Any]:
    completed_iterations = sum(
        int(row[route]["iterations"]) * (8 if route == "free_phase" else 1)
        for row in results for route in ("fixed_phase", "free_phase") if route in row)
    if state is not None:
        completed_iterations += int(iteration) * (8 if branch == "free_phase" else 1)
    return {"schema": "lightgen_g1_fullpop_feasibility_solver_checkpoint.v1",
            "task_id": TASK_ID, "run_id": identity.run_id,
            "source_run_id": targets["source_run_id"],
            "source_fullpop_run_id": targets["source_run_id"],
            "target_cursor": cursor, "branch": branch,
            "solver_state": copy.deepcopy(state), "iteration": iteration,
            "completed_target_results": copy.deepcopy(list(results)),
            "committed_start_iterations": completed_iterations,
            "training_updates": 0, "optimizer_calls": 0,
            "schedule_identity": "development_validation_indices_200y_plus_2_plus_3_fixed_order"}


def _execute(config_path: Path, config: Mapping[str, Any], *,
             resume_checkpoint: Path | None = None) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "feasibility solver requires AutoDL CUDA")
    control, receipt = _load_ready_control()
    resume = resume_checkpoint is not None
    if not resume:
        _require(not ATTEMPT_DIR.exists(), "feasibility attempt already exists", FileExistsError)
        control.open_scientific_attempt(ATTEMPT_DIR, f"{control.identity.run_id}:attempt_01", receipt)
        _atomic_json(ATTEMPT_DIR / "run_identity.json", control.identity.to_dict())
        _atomic_json(ATTEMPT_DIR / "run_config_snapshot.yaml", config)
        _copy_preflight(ATTEMPT_DIR)
        _atomic_torch(ATTEMPT_DIR / "fixed_targets.pt",
                      torch.load(PREFLIGHT_DIR / "fixed_targets.pt", map_location="cpu", weights_only=False))
        _atomic_torch(ATTEMPT_DIR / "mapper_source_states.pt",
                      torch.load(PREFLIGHT_DIR / "mapper_source_states.pt", map_location="cpu", weights_only=False))
        _task_state(RUN_SCIENTIFIC_ACTIVE)
    else:
        _require(ATTEMPT_DIR.is_dir() and resume_checkpoint.is_file(),
                 "resume requires an explicit existing solver checkpoint", FileNotFoundError)
        _assert_attempt(ATTEMPT_DIR)
        state = _json(ATTEMPT_DIR / "task_state.json")
        _require(state.get("state") in {RUN_SCIENTIFIC_ACTIVE, "SOLVER_ACTIVE", "INTERRUPTED"},
                 "closed feasibility attempt cannot be resumed")
        control.open_scientific_attempt(ATTEMPT_DIR, f"{control.identity.run_id}:attempt_01",
                                        receipt, resume=True)
    source = _load_sources(config, "cloud")
    targets = torch.load(ATTEMPT_DIR / "fixed_targets.pt", map_location="cpu", weights_only=False)
    identity = _assert_attempt(ATTEMPT_DIR)
    evidence_dir = ATTEMPT_DIR / "solver_checkpoints"
    evidence_dir.mkdir(exist_ok=True)
    if not resume:
        initial_checkpoint = _solver_checkpoint_payload(
            identity, targets, [], 0, "fixed_phase", None, 0)
        _save_checkpoint(evidence_dir / "target_00_fixed_00000.pt", initial_checkpoint)
        _save_checkpoint(evidence_dir / "latest_solver_checkpoint.pt", initial_checkpoint)
    baseline = _baseline_measurement(source, targets, ATTEMPT_DIR, torch.device("cuda:0"))
    if resume:
        checkpoint = torch.load(resume_checkpoint.resolve(), map_location="cpu", weights_only=False)
        _require(checkpoint.get("task_id") == TASK_ID and checkpoint.get("run_id") == identity.run_id
                 and checkpoint.get("source_run_id") == source["identity"].run_id
                 and checkpoint.get("source_fullpop_run_id") == source["identity"].run_id
                 and checkpoint.get("schedule_identity") ==
                     "development_validation_indices_200y_plus_2_plus_3_fixed_order",
                 "explicit solver checkpoint belongs to another run or population")
        results = copy.deepcopy(checkpoint.get("completed_target_results"))
        cursor, branch = int(checkpoint["target_cursor"]), checkpoint.get("branch")
        branch_state = checkpoint.get("solver_state")
        iteration = int(checkpoint.get("iteration", -1))
        _require(isinstance(results, list) and 0 <= cursor < TARGET_COUNT
                 and branch in {"fixed_phase", "free_phase"}
                 and checkpoint.get("training_updates") == 0
                 and checkpoint.get("optimizer_calls") == 0,
                 "explicit solver checkpoint cursor or update scope is invalid")
        expected_count = cursor if branch == "fixed_phase" else cursor + 1
        _require(len(results) == expected_count
                 and [int(item.get("target_index", -1)) for item in results] == list(range(expected_count))
                 and all(int(item.get("source_validation_index", -1)) ==
                         int(targets["source_validation_indices"][index])
                         for index, item in enumerate(results)),
                 "explicit solver checkpoint completed-target prefix is invalid")
        if branch == "fixed_phase":
            _require((branch_state is None and iteration == 0)
                     or (isinstance(branch_state, Mapping) and 0 < iteration <= SOLVER_STEPS
                         and int(branch_state.get("iteration", -1)) == iteration),
                     "explicit fixed-phase checkpoint state is invalid")
        else:
            _require(results[-1].get("target_index") == cursor
                     and "fixed_phase" in results[-1] and "free_phase" not in results[-1]
                     and ((branch_state is None and iteration == 0)
                          or (isinstance(branch_state, Mapping) and 0 < iteration <= SOLVER_STEPS
                              and int(branch_state.get("iteration", -1)) == iteration)),
                     "explicit free-phase checkpoint state is invalid")
        committed = sum(int(item[route]["iterations"]) * (8 if route == "free_phase" else 1)
                        for item in results for route in ("fixed_phase", "free_phase") if route in item)
        if branch_state is not None:
            committed += iteration * (8 if branch == "free_phase" else 1)
        _require(checkpoint.get("committed_start_iterations") == committed
                 and 0 <= committed <= TOTAL_START_ITERATION_LIMIT,
                 "explicit solver checkpoint iteration accounting is inconsistent")
    else:
        results, cursor, branch, branch_state = [], 0, "fixed_phase", None
    matrix = source["inputs"]["response_matrix"].to(device="cuda:0", dtype=torch.complex128)
    while cursor < TARGET_COUNT:
        target = targets["coefficients"][cursor].to(device="cuda:0", dtype=torch.complex128)
        phase = float(targets["box_b_public_phase_rad"][cursor])
        row = dict(results[-1]) if results and results[-1].get("target_index") == cursor else {
            "target_index": cursor,
            "source_validation_index": int(targets["source_validation_indices"][cursor]),
            "label": int(targets["labels"][cursor]), "box_b_public_phase_rad": phase,
        }
        if branch in {None, "fixed_phase"}:
            def fixed_callback(state: dict[str, Any]) -> None:
                payload = _solver_checkpoint_payload(identity, targets, results, cursor,
                                                     "fixed_phase", state, int(state["iteration"]))
                _save_checkpoint(evidence_dir / f"target_{cursor:02d}_fixed_{int(state['iteration']):05d}.pt",
                                 payload)
                _save_checkpoint(evidence_dir / "latest_solver_checkpoint.pt", payload)

            fixed = _fixed_phase_solve(matrix, target, phase, initial_state=branch_state,
                                       checkpoint=fixed_callback)
            row["fixed_phase"] = {"amplitude": fixed["amplitude"], "phase_rad": fixed["phase_rad"],
                                  "relative_error": fixed["relative_error"],
                                  "iterations": fixed["iterations"], "loss": fixed["loss"]}
            results = [item for item in results if int(item.get("target_index", -1)) != cursor]
            results.append(row)
            branch, branch_state = "free_phase", None
            transition = _solver_checkpoint_payload(identity, targets, results, cursor,
                                                    branch, branch_state, 0)
            _save_checkpoint(evidence_dir / f"target_{cursor:02d}_after_fixed.pt", transition)
            _save_checkpoint(evidence_dir / "latest_solver_checkpoint.pt", transition)

        def free_callback(state: dict[str, Any]) -> None:
            payload = _solver_checkpoint_payload(identity, targets, results, cursor,
                                                 "free_phase", state, int(state["iteration"]))
            _save_checkpoint(evidence_dir / f"target_{cursor:02d}_free_{int(state['iteration']):05d}.pt",
                             payload)
            _save_checkpoint(evidence_dir / "latest_solver_checkpoint.pt", payload)

        free = reachability._bounded_phase_solve(
            matrix, target, starts=8, max_steps=SOLVER_STEPS, stop_error=SOLVER_STOP,
            checkpoint=free_callback, initial_state=branch_state)
        row["free_phase"] = {"amplitude": free["amplitude"].detach().cpu().double(),
                             "phase_rad": float(free["phase_rad"]),
                             "relative_error": float(free["relative_error"]),
                             "iterations": int(free["iterations"]),
                             "best_errors": free["best_errors"].clone()}
        results = [item for item in results if int(item.get("target_index", -1)) != cursor]
        results.append(row)
        results.sort(key=lambda item: int(item["target_index"]))
        cursor += 1
        branch, branch_state = "fixed_phase", None
        completed_iterations = sum(
            int(item[key]["iterations"]) * (8 if key == "free_phase" else 1)
            for item in results for key in ("fixed_phase", "free_phase") if key in item)
        _require(completed_iterations <= TOTAL_START_ITERATION_LIMIT,
                 "fixed feasibility solver start-iteration budget was exceeded")
        payload = _solver_checkpoint_payload(identity, targets, results, cursor,
                                             branch if cursor < TARGET_COUNT else None, None, 0)
        payload["committed_start_iterations"] = completed_iterations
        _save_checkpoint(evidence_dir / f"target_{cursor - 1:02d}_complete.pt", payload)
        _save_checkpoint(evidence_dir / "latest_solver_checkpoint.pt", payload)
        _atomic_json(ATTEMPT_DIR / "solver_progress.json", {
            "task_id": TASK_ID, "run_id": identity.run_id, "completed_targets": cursor,
            "target_count": TARGET_COUNT, "committed_start_iterations": completed_iterations,
            "max_total_start_iterations": TOTAL_START_ITERATION_LIMIT,
            "branch": "fixed_phase" if cursor < TARGET_COUNT else "complete",
            "scientific_updates": 0, "optimizer_calls": 0,
        })
    total_iterations = sum(
        int(row[key]["iterations"]) * (8 if key == "free_phase" else 1)
        for row in results for key in ("fixed_phase", "free_phase"))
    _require(len(results) == TARGET_COUNT and total_iterations <= TOTAL_START_ITERATION_LIMIT,
             "fixed feasibility solver did not complete the authorized population and budget")
    solution = {"schema": "lightgen_g1_fullpop_feasibility_solutions.v1",
                "task_id": TASK_ID, "run_id": identity.run_id,
                "source_fullpop_run_id": source["identity"].run_id,
                "source_validation_indices": targets["source_validation_indices"],
                "rows": results, "total_start_iterations": total_iterations,
                "max_total_start_iterations": TOTAL_START_ITERATION_LIMIT,
                "scientific_updates": 0, "optimizer_calls": 0}
    _atomic_torch(ATTEMPT_DIR / "solver_solutions.pt", solution)
    _task_state("SOLVER_COMPLETE_AWAITING_FULL_OPTICS_MEASUREMENT", cursor=TARGET_COUNT,
                solver_iterations=total_iterations)
    return {"status": "SOLVER_COMPLETE_AWAITING_FULL_OPTICS_MEASUREMENT",
            "run_id": identity.run_id, "completed_targets": TARGET_COUNT,
            "total_start_iterations": total_iterations, "scientific_updates": 0,
            "optimizer_calls": 0, "baseline": baseline}


def _load_baseline_rows(attempt_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for index in range(TARGET_COUNT):
        path = attempt_dir / "measurements" / "baseline" / f"target_{index:02d}.pt"
        _require(path.is_file(), f"zero-update baseline target is missing: {index}", FileNotFoundError)
        rows.append(torch.load(path, map_location="cpu", weights_only=False))
    return rows


def _measure_attempt(config: Mapping[str, Any]) -> dict[str, Any]:
    _set_numerics()
    _require(torch.cuda.is_available(), "full-optics feasibility measurement requires AutoDL CUDA")
    identity = _assert_attempt(ATTEMPT_DIR)
    state = _json(ATTEMPT_DIR / "task_state.json")
    _require(state.get("state") in {"SOLVER_COMPLETE_AWAITING_FULL_OPTICS_MEASUREMENT",
                                     "MEASURING_FULL_OPTICS"},
             "full-optics measurement requires both solver routes to be complete")
    source = _load_sources(config, "cloud")
    targets = torch.load(ATTEMPT_DIR / "fixed_targets.pt", map_location="cpu", weights_only=False)
    solution = torch.load(ATTEMPT_DIR / "solver_solutions.pt", map_location="cpu", weights_only=False)
    _require(solution.get("run_id") == identity.run_id and len(solution.get("rows", [])) == TARGET_COUNT,
             "fixed-phase/free-phase solutions are incomplete or belong to another run")
    baseline = _load_baseline_rows(ATTEMPT_DIR)
    measurements_dir = ATTEMPT_DIR / "measurements" / "conditions"
    measurements_dir.mkdir(parents=True, exist_ok=True)
    repair_path = ATTEMPT_DIR / "measurement_row_identity_repair.json"
    repair_rows: list[dict[str, Any]] = []
    phase_metric_repairs: list[dict[str, Any]] = []
    if repair_path.is_file():
        previous_repair = _json(repair_path)
        _require(previous_repair.get("schema") ==
                 "lightgen_g1_fullpop_measurement_row_identity_repair.v1"
                 and previous_repair.get("task_id") == TASK_ID
                 and previous_repair.get("run_id") == identity.run_id
                 and previous_repair.get("status") in {"IN_PROGRESS", "COMPLETE"}
                 and previous_repair.get("raw_measurement_arrays_unchanged") is True
                 and isinstance(previous_repair.get("rows"), list),
                 "existing measurement row identity repair record is invalid")
        repair_rows = list(previous_repair["rows"])
        _require(all(isinstance(row, dict) for row in repair_rows),
                 "existing measurement row identity repair entries are malformed")
        repair_keys = [(row.get("section"), row.get("target_index"), row.get("condition"))
                       for row in repair_rows]
        _require(len(set(repair_keys)) == len(repair_keys),
                 "existing measurement row identity repair contains duplicate entries")
        previous_phase_repair = previous_repair.get("fixed_phase_metric_repair")
        if previous_phase_repair is not None:
            _require(previous_phase_repair.get("schema") ==
                     "lightgen_g1_fullpop_fixed_phase_metric_repair.v1"
                     and previous_phase_repair.get("status") in {"IN_PROGRESS", "COMPLETE"}
                     and previous_phase_repair.get("raw_measurement_arrays_unchanged") is True
                     and isinstance(previous_phase_repair.get("rows"), list)
                     and all(isinstance(row, dict) for row in previous_phase_repair["rows"]),
                     "existing fixed-phase metric repair record is invalid")
            phase_metric_repairs = list(previous_phase_repair["rows"])
            phase_indices = [row.get("target_index") for row in phase_metric_repairs]
            _require(len(set(phase_indices)) == len(phase_indices)
                     and all(isinstance(index, int) and 0 <= index < TARGET_COUNT
                             for index in phase_indices),
                     "existing fixed-phase metric repair contains duplicate or invalid targets")
    _task_state("MEASURING_FULL_OPTICS", cursor=0,
                solver_iterations=int(solution["total_start_iterations"]))

    def save_repair_status(status: str) -> None:
        phase_metric_summary = {
            "targets_recomputed": len(phase_metric_repairs),
            "updated_targets": sum(row["previous_saved_errors"] != row["corrected_errors"]
                                   for row in phase_metric_repairs),
            "decision_changes": sum(row["previous_saved_passed"] != row["corrected_passed"]
                                     for row in phase_metric_repairs),
        }
        _atomic_json(repair_path, {
            "schema": "lightgen_g1_fullpop_measurement_row_identity_repair.v1",
            "task_id": TASK_ID, "run_id": identity.run_id, "status": status,
            "reason": "single-target metric rows carried local index zero instead of the fixed population index",
            "raw_measurement_arrays_unchanged": True,
            "rows": sorted(repair_rows, key=lambda item: (item["section"],
                                                            item["target_index"],
                                                            item["condition"])),
            "fixed_phase_metric_repair": {
                "schema": "lightgen_g1_fullpop_fixed_phase_metric_repair.v1",
                "status": status,
                "reason": "cast the complex64 target to complex128 before multiplying by the shared phase",
                "raw_measurement_arrays_unchanged": True,
                "recomputed_from_saved_actual_coefficients": True,
                **phase_metric_summary,
                "rows": sorted(phase_metric_repairs, key=lambda item: item["target_index"]),
            },
        })

    for index in range(TARGET_COUNT):
        baseline_path = ATTEMPT_DIR / "measurements" / "baseline" / f"target_{index:02d}.pt"
        saved_baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
        raw_fields = ("target_coefficients", "commands", "outputs", "direct_raw_cmos")
        raw_before = {key: copy.deepcopy(saved_baseline[key]) for key in raw_fields}
        changed = _repair_measurement_row_identity(
            saved_baseline, section="baseline", target_index=index,
            conditions=("frozen_v2", "box_b5000"), repair_rows=repair_rows)
        if changed:
            _require(all(fullpop._state_dicts_equal(saved_baseline[key], raw_before[key])
                         for key in raw_fields),
                     "baseline row identity repair altered a saved optical measurement array")
            save_repair_status("IN_PROGRESS")
            _atomic_torch(baseline_path, saved_baseline)

    for index, solution_row in enumerate(solution["rows"]):
        shard_path = measurements_dir / f"target_{index:02d}.pt"
        if shard_path.is_file():
            saved = torch.load(shard_path, map_location="cpu", weights_only=False)
            _require(saved.get("run_id") == identity.run_id and saved.get("target_index") == index
                     and torch.equal(saved.get("target_coefficients"), targets["coefficients"][index]),
                     f"existing full-optics target measurement is invalid: {index}")
            raw_fields = ("target_coefficients", "commands", "outputs", "direct_raw_cmos")
            raw_before = {key: copy.deepcopy(saved[key]) for key in raw_fields}
            changed = _repair_measurement_row_identity(
                saved, section="conditions", target_index=index, conditions=CONDITIONS,
                repair_rows=repair_rows)
            fixed_row = saved["rows"]["fixed_phase_solver"]
            old_errors = {str(grid): float(saved["fixed_phase_target_errors"][str(grid)])
                          for grid in GRIDS}
            old_passed = bool(saved["fixed_phase_target_passed"])
            old_row_errors = {str(grid): float(fixed_row["fixed_phase_target_errors"][str(grid)])
                              for grid in GRIDS}
            old_row_passed = bool(fixed_row["fixed_phase_target_passed"])
            _require(old_errors == old_row_errors and old_passed == old_row_passed,
                     f"saved fixed-phase metric copies disagree at target {index}")
            phase = float(targets["box_b_public_phase_rad"][index])
            recomputed_errors = _fixed_phase_grid_errors(
                saved["outputs"]["fixed_phase_solver"], targets["coefficients"][index], phase)
            recomputed_passed = max(recomputed_errors.values()) <= GATE_LIMIT
            phase_repair = next((row for row in phase_metric_repairs
                                 if row["target_index"] == index), None)
            metric_changed = False
            if phase_repair is None:
                legacy_target = targets["coefficients"][index] * torch.exp(torch.tensor(
                    1j * phase, dtype=torch.complex128))
                legacy_errors = _fixed_phase_grid_errors(
                    saved["outputs"]["fixed_phase_solver"], legacy_target, 0.0)
                phase_repair = {
                    "target_index": index,
                    "previous_saved_errors": old_errors,
                    "previous_saved_passed": old_passed,
                    "previous_cpu_reproduction_errors": legacy_errors,
                    "previous_cpu_reproduction_passed": max(legacy_errors.values()) <= GATE_LIMIT,
                    "corrected_errors": recomputed_errors,
                    "corrected_passed": recomputed_passed,
                }
                phase_metric_repairs.append(phase_repair)
                metric_changed = True
            else:
                _require(fullpop.fit20._numeric_tree_matches(
                             phase_repair.get("corrected_errors"), recomputed_errors, 1.0e-14)
                         and phase_repair.get("corrected_passed") is recomputed_passed,
                         f"fixed-phase metric repair changed across resume at target {index}")
            if old_errors != recomputed_errors or old_passed != recomputed_passed:
                metric_changed = True
            saved["fixed_phase_target_errors"] = recomputed_errors
            saved["fixed_phase_target_passed"] = recomputed_passed
            fixed_row["fixed_phase_target_errors"] = recomputed_errors
            fixed_row["fixed_phase_target_passed"] = recomputed_passed
            changed = changed or metric_changed
            if changed:
                _require(all(fullpop._state_dicts_equal(saved[key], raw_before[key])
                             for key in raw_fields),
                         "measurement metadata repair altered a saved optical measurement array")
                save_repair_status("IN_PROGRESS")
                _atomic_torch(shard_path, saved)
            continue
        base = baseline[index]
        commands = {"frozen_v2": base["commands"]["frozen_v2"],
                    "box_b5000": base["commands"]["box_b5000"],
                    "fixed_phase_solver": solution_row["fixed_phase"]["amplitude"].to(torch.float32).reshape(1, 1, 200, 200),
                    "free_phase_solver": solution_row["free_phase"]["amplitude"].to(torch.float32).reshape(1, 1, 200, 200)}
        outputs, direct, rows = _measure_commands(targets["coefficients"][index:index + 1],
                                                  commands, source["inputs"], torch.device("cuda:0"),
                                                  target_index_offset=index)
        metric_by_condition = {condition: next(row for row in rows if row["condition"] == condition)
                               for condition in commands}
        fixed_targets = {}
        for grid in GRIDS:
            actual = outputs["fixed_phase_solver"][grid]["coefficients"][0]
            target_phase = _phase_shifted_target(
                targets["coefficients"][index], float(targets["box_b_public_phase_rad"][index]))
            fixed_targets[str(grid)] = _fixed_phase_relative_error(actual, target_phase)
        fixed_phase_pass = max(fixed_targets.values()) <= GATE_LIMIT
        for condition, row in metric_by_condition.items():
            row["label"] = int(targets["labels"][index])
            row["source_validation_index"] = int(targets["source_validation_indices"][index])
            row["condition_passed"] = bool(row["row_passed"])
            if condition == "fixed_phase_solver":
                row["fixed_phase_target_errors"] = fixed_targets
                row["fixed_phase_target_passed"] = fixed_phase_pass
        payload = {"schema": "lightgen_g1_fullpop_feasibility_measurement_target.v1",
                   "task_id": TASK_ID, "run_id": identity.run_id,
                   "source_fullpop_run_id": source["identity"].run_id,
                   "target_index": index,
                   "source_validation_index": int(targets["source_validation_indices"][index]),
                   "label": int(targets["labels"][index]),
                   "target_coefficients": targets["coefficients"][index].clone(),
                   "commands": commands, "outputs": outputs, "direct_raw_cmos": direct,
                   "rows": metric_by_condition,
                   "box_b_public_phase_rad": float(targets["box_b_public_phase_rad"][index]),
                   "fixed_phase_target_errors": fixed_targets,
                   "fixed_phase_target_passed": fixed_phase_pass,
                   "solver_metadata": {"fixed_phase": {key: value for key, value in
                       solution_row["fixed_phase"].items() if key != "amplitude"},
                       "free_phase": {key: value for key, value in
                       solution_row["free_phase"].items() if key not in {"amplitude", "best_errors"}}}}
        _atomic_torch(shard_path, payload)
        _atomic_json(ATTEMPT_DIR / "measurement_progress.json", {
            "task_id": TASK_ID, "run_id": identity.run_id, "completed_targets": index + 1,
            "target_count": TARGET_COUNT, "scientific_updates": 0})
    if repair_rows or repair_path.is_file():
        repair_keys = [(row["section"], int(row["target_index"]), row["condition"])
                       for row in repair_rows]
        _require(len(set(repair_keys)) == len(repair_keys),
                 "measurement row identity repair contains duplicate entries")
        save_repair_status("COMPLETE")
    if repair_path.is_file():
        _require(len(phase_metric_repairs) == TARGET_COUNT
                 and sorted(row["target_index"] for row in phase_metric_repairs) == list(range(TARGET_COUNT)),
                 "fixed-phase metric repair did not reconsume all saved target outputs")
        phase_metric_summary = {
            "status": "COMPLETE", "targets_recomputed": len(phase_metric_repairs),
            "updated_targets": sum(row["previous_saved_errors"] != row["corrected_errors"]
                                   for row in phase_metric_repairs),
            "decision_changes": sum(row["previous_saved_passed"] != row["corrected_passed"]
                                     for row in phase_metric_repairs),
            "raw_measurement_arrays_unchanged": True,
        }
    else:
        _require(not phase_metric_repairs,
                 "fixed-phase metric repair rows exist without their repair record")
        phase_metric_summary = {"status": "NOT_REQUIRED", "targets_recomputed": 0,
                                "updated_targets": 0, "decision_changes": 0,
                                "raw_measurement_arrays_unchanged": True}
    repair_summary = {"status": "COMPLETE" if repair_path.is_file() else "NOT_REQUIRED",
                      "repaired_rows": len(repair_rows),
                      "raw_measurement_arrays_unchanged": True,
                      "fixed_phase_metric_repair": phase_metric_summary}
    rows = []
    for index in range(TARGET_COUNT):
        payload = torch.load(measurements_dir / f"target_{index:02d}.pt",
                             map_location="cpu", weights_only=False)
        for condition in CONDITIONS:
            rows.append({**payload["rows"][condition],
                         "target_index": index,
                         "source_validation_index": payload["source_validation_index"],
                         "label": payload["label"], "condition": condition,
                         "fixed_phase_target_errors": payload.get("fixed_phase_target_errors")
                             if condition == "fixed_phase_solver" else None})
    summary = _summarize_conditions(rows)
    witness = _reachability_summary(rows)
    _atomic_torch(ATTEMPT_DIR / "measurements" / "conditions.pt", {
        "schema": "lightgen_g1_fullpop_feasibility_conditions.v1",
        "task_id": TASK_ID, "run_id": identity.run_id,
        "rows": rows, "summaries": summary, "reachability": witness,
    })
    _write_rows_csv(ATTEMPT_DIR / "measurements" / "condition_rows.csv", rows)
    report = _write_report(config, identity.run_id, targets, solution, rows, summary, witness)
    total_iterations = int(solution["total_start_iterations"])
    final = {"schema": "lightgen_g1_fullpop_feasibility_decision.v1",
             "task_id": TASK_ID, "run_id": identity.run_id,
             "source_fullpop_run_id": source["identity"].run_id,
             "lifecycle_state": RUN_COMPLETED,
             "diagnostic_result": witness["classification"],
             "population_role": "development_feasibility20",
             "target_count": TARGET_COUNT,
             "source_validation_indices": targets["source_validation_indices"].tolist(),
             "conditions": list(CONDITIONS), "metric_names": list(METRIC_KEYS),
             "threshold": GATE_LIMIT, "fixed_phase_target_threshold": GATE_LIMIT,
             "condition_summaries": summary, "reachability": witness,
             "measurement_row_identity_repair": repair_summary,
             "fixed_phase_metric_repair": phase_metric_summary,
             "solver": {"fixed_phase_iterations": sum(int(row["fixed_phase"]["iterations"])
                                                        for row in solution["rows"]),
                        "free_phase_start_iterations": sum(int(row["free_phase"]["iterations"]) * 8
                                                             for row in solution["rows"]),
                        "total_start_iterations": total_iterations,
                        "max_total_start_iterations": TOTAL_START_ITERATION_LIMIT,
                        "scientific_updates": 0, "optimizer_calls": 0},
             "report": report, "training_labels": False, "inference_fallback": False,
             "hardware_readiness_claim": False}
    _atomic_json(ATTEMPT_DIR / "final_decision.json", final)
    _task_state(RUN_COMPLETED, cursor=TARGET_COUNT, solver_iterations=total_iterations)
    _atomic_json(ATTEMPT_DIR / "research_lifecycle_state.json", {
        "task_id": TASK_ID, "run_id": identity.run_id, "state": RUN_COMPLETED,
        "scientific_updates": 0, "optimizer_calls": 0, "optical_updates": 0,
        "evaluator_updates": 0, "diagnostic_result": witness["classification"]})
    _build_transfer_manifest(ATTEMPT_DIR, identity.run_id)
    return final


def _review_saved_phase_metrics(evidence_dir: Path, review_dir: Path,
                                config: Mapping[str, Any]) -> dict[str, Any]:
    _set_numerics()
    _require(evidence_dir.resolve() == ATTEMPT_DIR.resolve()
             and review_dir.resolve() == REVIEW_DIR.resolve(),
             "phase-metric review uses the fixed completed attempt and review directory", ValueError)
    identity = _assert_attempt(evidence_dir)
    task_state = _json(evidence_dir / "task_state.json")
    lifecycle = _json(evidence_dir / "research_lifecycle_state.json")
    _require(task_state.get("state") == RUN_COMPLETED
             and lifecycle.get("state") == RUN_COMPLETED
             and all(task_state.get(key) == 0 for key in
                     ("scientific_updates", "optimizer_calls", "optical_updates", "evaluator_updates"))
             and all(lifecycle.get(key) == 0 for key in
                     ("scientific_updates", "optimizer_calls", "optical_updates", "evaluator_updates")),
             "phase-metric review requires the completed zero-update attempt")
    source = _load_sources(config, "cloud")
    targets = torch.load(evidence_dir / "fixed_targets.pt", map_location="cpu", weights_only=False)
    expected_targets = _target_payload(config, source, torch.device("cpu"))
    _require(targets.get("task_id") == TASK_ID
             and torch.equal(targets.get("source_validation_indices"),
                             expected_targets["source_validation_indices"])
             and torch.equal(targets.get("coefficients"), expected_targets["coefficients"])
             and targets.get("source_run_id") == source["identity"].run_id,
             "phase-metric review population differs from its fixed source")
    repair = _json(evidence_dir / "measurement_row_identity_repair.json")
    _require(repair.get("status") == "COMPLETE"
             and repair.get("task_id") == TASK_ID and repair.get("run_id") == identity.run_id
             and repair.get("raw_measurement_arrays_unchanged") is True
             and len(repair.get("rows", [])) == 116,
             "completed measurement identity repair is missing or incomplete")
    _require(not review_dir.exists() or
             ((review_dir / "review_identity.json").is_file()
              and _json(review_dir / "review_identity.json").get("run_id") == identity.run_id),
             "phase-metric review directory belongs to another attempt", FileExistsError)
    original_decision = _json(evidence_dir / "final_decision.json")
    conditions = torch.load(evidence_dir / "measurements" / "conditions.pt",
                            map_location="cpu", weights_only=False)
    _require(conditions.get("task_id") == TASK_ID and conditions.get("run_id") == identity.run_id
             and len(conditions.get("rows", [])) == TARGET_COUNT * len(CONDITIONS),
             "completed measurement aggregation is incomplete")
    source_states = torch.load(evidence_dir / "mapper_source_states.pt",
                               map_location="cpu", weights_only=False)
    _require(source_states.get("task_id") == TASK_ID
             and source_states.get("source_run_id") == source["identity"].run_id
             and fullpop._state_dicts_equal(
                 source_states.get("mapper_states", {}).get("frozen_v2_baseline"),
                 source["source_states"]["frozen_v2_baseline"])
             and fullpop._state_dicts_equal(
                 source_states.get("mapper_states", {}).get("box_b_candidate"),
                 source["source_states"]["box_b_candidate"]),
             "saved mapper states differ from the frozen source checkpoints")
    _require(torch.cuda.is_available(), "exact mapper-command replay requires AutoDL CUDA")
    replay_device = torch.device("cuda:0")
    command_replay_rows = []
    corrected_rows = copy.deepcopy(conditions["rows"])
    audit_rows = []
    for index in range(TARGET_COUNT):
        payload = torch.load(evidence_dir / "measurements" / "conditions" / f"target_{index:02d}.pt",
                             map_location="cpu", weights_only=False)
        _require(payload.get("run_id") == identity.run_id and payload.get("target_index") == index
                 and torch.equal(payload.get("target_coefficients"), targets["coefficients"][index]),
                 f"completed measurement shard identity differs at target {index}")
        baseline_payload = torch.load(
            evidence_dir / "measurements" / "baseline" / f"target_{index:02d}.pt",
            map_location="cpu", weights_only=False)
        v2_command, _, _ = _one_mapper_command(
            source["source_states"]["frozen_v2_baseline"], targets["coefficients"][index],
            source["inputs"], replay_device)
        box_command, replayed_phase, replayed_preclip = _one_mapper_command(
            source["source_states"]["box_b_candidate"], targets["coefficients"][index],
            source["inputs"], replay_device)
        _require_exact_command_replay(v2_command, baseline_payload["commands"]["frozen_v2"],
                                      index, "frozen_v2")
        _require_exact_command_replay(box_command, baseline_payload["commands"]["box_b5000"],
                                      index, "box_b5000")
        _require(replayed_phase == float(baseline_payload["b_public_phase_rad"])
                 and torch.equal(replayed_preclip, baseline_payload["b_preclip_amplitude"]),
                 f"saved mapper phase or preclip amplitude differs from exact CUDA replay at {index}")
        command_replay_rows.append({
            "target_index": index,
            "source_validation_index": int(targets["source_validation_indices"][index]),
            "target_coefficients": targets["coefficients"][index].clone(),
            "commands": {"frozen_v2": v2_command, "box_b5000": box_command},
            "box_b_public_phase_rad": replayed_phase,
            "box_b_preclip_amplitude": replayed_preclip,
        })
        stored = payload["rows"]["fixed_phase_solver"]
        old_errors = {str(grid): float(payload["fixed_phase_target_errors"][str(grid)])
                      for grid in GRIDS}
        old_row_errors = {str(grid): float(stored["fixed_phase_target_errors"][str(grid)])
                          for grid in GRIDS}
        old_passed = bool(payload["fixed_phase_target_passed"])
        _require(old_errors == old_row_errors
                 and old_passed is bool(stored["fixed_phase_target_passed"]),
                 f"historical fixed-phase metric copies disagree at target {index}")
        phase = float(targets["box_b_public_phase_rad"][index])
        legacy_target = targets["coefficients"][index] * torch.exp(torch.tensor(
            1j * phase, dtype=torch.complex128))
        legacy_errors = _fixed_phase_grid_errors(payload["outputs"]["fixed_phase_solver"],
                                                 legacy_target, 0.0)
        _require(all(abs(old_errors[str(grid)] - legacy_errors[str(grid)])
                      <= LEGACY_PHASE_ERROR_CROSS_DEVICE_TOLERANCE
                      for grid in GRIDS)
                 and old_passed is (max(legacy_errors.values()) <= GATE_LIMIT),
                 f"historical fixed-phase metric does not reproduce its complex64 phase path at {index}")
        corrected_errors = _fixed_phase_grid_errors(payload["outputs"]["fixed_phase_solver"],
                                                     targets["coefficients"][index], phase)
        corrected_passed = max(corrected_errors.values()) <= GATE_LIMIT
        audit_rows.append({"target_index": index, "previous_saved_errors": old_errors,
                           "previous_saved_passed": old_passed,
                           "previous_cpu_reproduction_errors": legacy_errors,
                           "previous_cpu_reproduction_passed":
                               max(legacy_errors.values()) <= GATE_LIMIT,
                           "corrected_errors": corrected_errors,
                           "corrected_passed": corrected_passed})
        row = next((item for item in corrected_rows
                    if int(item["target_index"]) == index
                    and item["condition"] == "fixed_phase_solver"), None)
        _require(row is not None, f"fixed-phase condition row is missing at target {index}")
        row["fixed_phase_target_errors"] = corrected_errors
        row["fixed_phase_target_passed"] = corrected_passed
    summary = _summarize_conditions(corrected_rows)
    witness = _reachability_summary(corrected_rows)
    phase_summary = {
        "status": "COMPLETE", "targets_recomputed": len(audit_rows),
        "updated_targets": sum(row["previous_saved_errors"] != row["corrected_errors"]
                               for row in audit_rows),
        "decision_changes": sum(row["previous_saved_passed"] != row["corrected_passed"]
                                 for row in audit_rows),
        "raw_measurement_arrays_unchanged": True,
    }
    _require(original_decision.get("run_id") == identity.run_id
             and original_decision.get("diagnostic_result") ==
                 conditions.get("reachability", {}).get("classification"),
             "historical decision does not match the completed raw measurement")
    review_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(review_dir / "review_identity.json", {
        "schema": "lightgen_g1_fullpop_phase_metric_review.v1",
        "task_id": TASK_ID, "run_id": identity.run_id,
        "source_run_id": source["identity"].run_id,
        "parent_attempt_state": RUN_COMPLETED, "status": "COMPLETE",
        "scientific_updates": 0, "optimizer_calls": 0,
        "optical_updates": 0, "evaluator_updates": 0,
        "exact_cuda_mapper_command_replay_targets": len(command_replay_rows),
    })
    _atomic_json(review_dir / "fixed_phase_metric_repair.json", {
        "schema": "lightgen_g1_fullpop_fixed_phase_metric_repair.v1",
        "task_id": TASK_ID, "run_id": identity.run_id,
        "source_run_id": source["identity"].run_id,
        "status": "COMPLETE", "reason":
            "cast the complex64 target to complex128 before multiplying by the shared phase",
        "recomputed_from_saved_actual_coefficients": True,
        "raw_measurement_arrays_unchanged": True,
        **{key: value for key, value in phase_summary.items() if key != "status"},
        "rows": audit_rows,
    })
    _atomic_torch(review_dir / "conditions.pt", {
        "schema": "lightgen_g1_fullpop_feasibility_conditions.v1",
        "task_id": TASK_ID, "run_id": identity.run_id,
        "rows": corrected_rows, "summaries": summary, "reachability": witness,
    })
    _atomic_torch(review_dir / "mapper_command_replay.pt", {
        "schema": "lightgen_g1_fullpop_mapper_command_replay.v1",
        "task_id": TASK_ID, "run_id": identity.run_id,
        "source_run_id": source["identity"].run_id,
        "device": "cuda:0", "precision": "production_float32_mapper_float64_correction",
        "exact_command_count": len(command_replay_rows), "rows": command_replay_rows,
    })
    _write_rows_csv(review_dir / "condition_rows.csv", corrected_rows)
    report = _write_phase_metric_review_report(
        identity.run_id, original_decision, targets, phase_summary, summary, witness,
        len(command_replay_rows))
    (review_dir / "final_report.md").write_text(report, encoding="utf-8")
    revised_decision = copy.deepcopy(original_decision)
    revised_decision.update({
        "diagnostic_result": witness["classification"],
        "condition_summaries": summary, "reachability": witness,
        "fixed_phase_metric_repair": phase_summary,
        "measurement_row_identity_repair": {
            "status": repair["status"], "repaired_rows": len(repair["rows"]),
            "raw_measurement_arrays_unchanged": True,
            "fixed_phase_metric_repair": phase_summary,
        },
        "scientific_updates": 0, "optimizer_calls": 0,
        "optical_updates": 0, "evaluator_updates": 0,
        "report": report,
        "review_revision": "phase_metric_complex128_repair_01",
    })
    _atomic_json(review_dir / "final_decision.json", revised_decision)
    manifest = _build_review_transfer_manifest(review_dir, identity.run_id)
    return {"status": "PHASE_METRIC_REVIEW_COMPLETE", "run_id": identity.run_id,
            "classification": witness["classification"],
            "fixed_phase_metric_repair": phase_summary,
            "exact_cuda_mapper_command_replay_targets": len(command_replay_rows),
            "transfer_file_count": manifest["file_count"],
            "transfer_bytes": manifest["bytes_total"],
            "scientific_updates": 0, "optimizer_calls": 0}


def _summarize_conditions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summarize(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {key: {
            "mean": float(np.mean([float(row[key]) for row in selected])),
            "rms": float(np.sqrt(np.mean([float(row[key]) ** 2 for row in selected]))),
            "median": float(np.median([float(row[key]) for row in selected])),
            "maximum": float(np.max([float(row[key]) for row in selected])),
        } for key in METRIC_KEYS}

    result = {}
    for condition in CONDITIONS:
        selected = [row for row in rows if row["condition"] == condition]
        per_class = {}
        for label in range(CLASS_COUNT):
            class_rows = [row for row in selected if int(row["label"]) == label]
            per_class[str(label)] = {
                "denominator": len(class_rows),
                "passed_rows": sum(bool(row["row_passed"]) for row in class_rows),
                "all_six_passed": bool(class_rows)
                and all(bool(row["row_passed"]) for row in class_rows),
                "metrics": summarize(class_rows),
            }
        result[condition] = {"denominator": len(selected),
                             "passed_rows": sum(bool(row["row_passed"]) for row in selected),
                             "all_six_passed": bool(selected) and all(bool(row["row_passed"]) for row in selected),
                             "metrics": summarize(selected), "per_class": per_class}
    return result


def _reachability_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_target = {index: {row["condition"]: row for row in rows if int(row["target_index"]) == index}
                 for index in range(TARGET_COUNT)}
    any_witness = []
    fixed_witness = []
    for index in range(TARGET_COUNT):
        conditions = by_target[index]
        eligible = []
        for condition in CONDITIONS:
            row = conditions[condition]
            passed = bool(row["row_passed"])
            if condition == "fixed_phase_solver":
                passed = passed and bool(row.get("fixed_phase_target_passed"))
            if passed:
                eligible.append(condition)
        any_witness.append({"target_index": index, "witness_conditions": eligible,
                            "has_witness": bool(eligible)})
        fixed_row = conditions["fixed_phase_solver"]
        fixed_witness.append({"target_index": index,
                              "has_witness": bool(fixed_row["row_passed"]
                                                  and fixed_row.get("fixed_phase_target_passed"))})
    any_count = sum(item["has_witness"] for item in any_witness)
    fixed_count = sum(item["has_witness"] for item in fixed_witness)
    if any_count == TARGET_COUNT and fixed_count == TARGET_COUNT:
        classification = "CURRENT_PHASE_WITNESS_20_OF_20"
        interpretation = ("四种预定指令中20/20均有完整光路见证，固定B相位求解也20/20通过；"
                          "这些目标在本次数值光路和预算下存在当前相位指令。")
    elif any_count == TARGET_COUNT:
        classification = "FREE_PHASE_ADDED_WITNESSES"
        interpretation = ("20/20均有完整光路见证，但固定B相位未覆盖全部目标；"
                          "重新选择等效公共相位在本预算内增加了见证。")
    else:
        classification = "BOUNDED_BUDGET_EVIDENCE_INSUFFICIENT"
        interpretation = ("固定预算内未为全部目标找到完整光路见证；这不能判定目标物理不可制备。")
    return {"classification": classification, "interpretation": interpretation,
            "denominator": TARGET_COUNT, "any_condition_witness_rows": any_count,
            "fixed_phase_witness_rows": fixed_count,
            "all_four_conditions_measured": True,
            "any_condition_witnesses": any_witness,
            "fixed_phase_witnesses": fixed_witness}


def _write_rows_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = ["target_index", "source_validation_index", "label", "condition",
               *METRIC_KEYS, "row_passed", "fixed_phase_target_passed"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in rows)


def _write_report(config: Mapping[str, Any], run_id: str, targets: Mapping[str, Any],
                  solution: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                  summary: Mapping[str, Any], witness: Mapping[str, Any]) -> str:
    repair_path = ATTEMPT_DIR / "measurement_row_identity_repair.json"
    repair_record = _json(repair_path) if repair_path.is_file() else {}
    repair_count = len(repair_record.get("rows", []))
    phase_repair = repair_record.get("fixed_phase_metric_repair", {})
    phase_note = (
        f"固定相位误差在{phase_repair.get('targets_recomputed', 0)}条保存输出上重新计算，"
        f"其中{phase_repair.get('updated_targets', 0)}条数值修订、"
        f"{phase_repair.get('decision_changes', 0)}条门槛判定变化；"
        "先将目标系数转换为complex128再施加公共相位，未重跑光路。"
        if phase_repair.get("status") == "COMPLETE" else
        "固定相位误差未执行精度修订。")
    repair_note = (f"测量行身份修复记录{repair_count}条；仅修正目标索引、类别和来源索引，"
                   f"并已逐分片确认系数、指令和CMOS原始数组不变。{phase_note}" if repair_count else
                   f"本次无需修复测量行身份。{phase_note}")
    report = f"""# G1全人口结果：公共相位配对可达性诊断

## 结果与状态

诊断任务`{TASK_ID}`对固定20条开发目标完成零映射器更新的有界求解和2048/4096完整光路测量。至少一种预定指令有见证的目标为{witness['any_condition_witness_rows']}/20；固定B相位求解有见证的目标为{witness['fixed_phase_witness_rows']}/20。诊断分类为`{witness['classification']}`。

## 科学认识

{witness['interpretation']}四种条件分别报告六项误差和逐类统计。该结果只覆盖这20条开发目标、当前冻结C45k数值光路及固定求解预算；无映射器训练、光学更新、评价器更新或物理相位控制。

## 目标缺口

即使获得全部数值见证，也不证明G1总体可制备、语义可读或实物可部署。若部分目标没有见证，只能说明本次有界算法和预算未找到证据，不能证明其不可达。历史2,000条留出已经消费，不在本诊断中恢复为独立留出。

## 工程完成与验证

目标索引为`{targets['source_validation_indices'].tolist()}`；固定相位每目标至多{SOLVER_STEPS}步，自由相位八起点各至多{SOLVER_STEPS}步，总起点迭代{solution['total_start_iterations']}/{TOTAL_START_ITERATION_LIMIT}。{repair_note}逐目标和逐类的六项误差均值、RMS、中位数、最大值及通过数保存在`measurements/conditions.pt`。残差分解保留裁剪前、边界裁剪、float32舍入、实际传播差异及全部交叉项。所有CMOS误差使用原生强度，不归一化、不匹配功率或增益。

## 决策影响与下一步

本次固定相位全部有见证时，保留映射优化、训练暴露次数和当前相位表示作为后续可学习性解释；只有自由相位补足见证时，优先保留电子端公共相位选择的研究依据；未全部见证时，下一步仍需独立界定算法覆盖率或实际可达性，不自动追加训练。按本计划完成证据回传和Windows复核后停止，不自动扩网、改损失、推进G4、评分或实物实验。
"""
    path = ATTEMPT_DIR / "final_report.md"
    path.write_text(report, encoding="utf-8")
    return report


def _write_phase_metric_review_report(run_id: str, original_decision: Mapping[str, Any],
                                      targets: Mapping[str, Any], phase_summary: Mapping[str, Any],
                                      summary: Mapping[str, Any], witness: Mapping[str, Any],
                                      command_replay_count: int) -> str:
    class_coverage = "；".join(
        f"{label}:{item['passed_rows']}/{item['denominator']}"
        for label, item in summary["fixed_phase_solver"]["per_class"].items())
    identity_repaired = int(original_decision.get("measurement_row_identity_repair", {}).get(
        "repaired_rows", 0))
    return f"""# G1固定相位误差精度修订

## 结果与状态

对已完成诊断`{run_id}`的保存输出完成零更新指标复核。原attempt、原始测量和历史报告保留。修订后的分类为`{witness['classification']}`；任一条件完整光路见证为{witness['any_condition_witness_rows']}/20，固定B相位见证为{witness['fixed_phase_witness_rows']}/20。

## 科学认识

修订仅把复数目标先转换为complex128，再乘公共相位，消除原指标路径中的complex64舍入。固定相位误差在{phase_summary['targets_recomputed']}条保存输出上重新计算，数值改变{phase_summary['updated_targets']}条，门槛判定改变{phase_summary['decision_changes']}条。原始光路输出、指令和六项CMOS/系数测量均未改动；未执行训练、优化器更新、求解或光路传播。

## 目标缺口

该修订只校正固定相位系数辅助判据，不增加人口，也不证明G1总体可制备、硬件可部署或语义可读。原六项测量仍由Windows从原始输出重新计算。

## 工程完成与验证

固定来源索引为`{targets['source_validation_indices'].tolist()}`。测量行身份修复记录{identity_repaired}条；云端CUDA映射器指令逐位重放{command_replay_count}条目标。本次相位修订只保存在独立目录。固定相位逐类通过数为{class_coverage}。原attempt状态保持`{original_decision['lifecycle_state']}`。逐类指标保存在`conditions.pt`，跨网格与CMOS主指标继承原始光路数据。

## 决策影响与下一步

原判定为`{original_decision['diagnostic_result']}`，修订判定为`{witness['classification']}`。本次修订完成后停止；不自动启动训练、扩网、G4、量化、人工评分或实物实验。
"""


def _build_review_transfer_manifest(review_dir: Path, run_id: str) -> dict[str, Any]:
    excluded = {"transfer_manifest.json", "transfer_receipt.json", "windows_reconsumption.json"}
    files = [{"path": path.relative_to(review_dir).as_posix(), "bytes": path.stat().st_size}
             for path in sorted(review_dir.rglob("*"))
             if path.is_file() and path.name not in excluded and ".tmp." not in path.name]
    remote_dir = REVIEW_DIR.relative_to(ROOT).as_posix()
    payload = {"schema": "lightgen_g1_fullpop_phase_metric_review_transfer.v1",
               "task_id": TASK_ID, "run_id": run_id,
               "remote_run_dir": "/root/autodl-tmp/codex-original/" + remote_dir,
               "files": files, "file_count": len(files),
               "bytes_total": sum(int(item["bytes"]) for item in files)}
    _atomic_json(review_dir / "transfer_manifest.json", payload)
    return payload


def _display_uint8(raw_cmos_w_per_m2: np.ndarray, pixel_area_m2: float,
                   camera_gain_w_inverse: float) -> np.ndarray:
    raw = np.asarray(raw_cmos_w_per_m2, dtype=np.float64)
    _require(np.isfinite(raw).all() and float(raw.min()) >= 0.0
             and math.isfinite(pixel_area_m2) and pixel_area_m2 > 0.0
             and math.isfinite(camera_gain_w_inverse) and camera_gain_w_inverse > 0.0,
             "display conversion has invalid CMOS data or frozen camera units")
    display = np.rint(np.clip(raw * pixel_area_m2 * camera_gain_w_inverse, 0.0, 1.0) * 255.0)
    independent = (torch.from_numpy(raw) * pixel_area_m2 * camera_gain_w_inverse
                   ).clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    result = display.astype(np.uint8)
    _require(np.array_equal(result, independent),
             "NumPy and Torch camera-unit display conversions differ pixelwise")
    return result


def _render_windows_display(evidence_dir: Path, source: Mapping[str, Any],
                            targets: Mapping[str, Any], *,
                            output_dir: Path | None = None) -> dict[str, Any]:
    from PIL import Image, ImageDraw

    display_dir = (output_dir or evidence_dir) / "windows_display_evidence"
    display_dir.mkdir(exist_ok=True)
    camera_model = pilot._new_optical_model(source["inputs"]["checkpoint"], 2048, torch.device("cpu"))
    pixel_area = pilot._camera_pixel_area_m2(camera_model)
    del camera_model
    gain = float(source["inputs"]["camera_gain"])
    _require(math.isclose(pixel_area, 6.25e-12, rel_tol=1.0e-12, abs_tol=0.0),
             "frozen camera pixel area differs from the repaired optical system")
    role_order = ("direct_decoder_reference", "frozen_v2", "box_b5000",
                  "fixed_phase_solver", "free_phase_solver")
    labels = ("Direct Decoder", "Frozen V2", "B5000", "Fixed phase", "Free phase")
    raw_rows = []
    for index in range(TARGET_COUNT):
        payload = torch.load(evidence_dir / "measurements" / "conditions" / f"target_{index:02d}.pt",
                             map_location="cpu", weights_only=False)
        _require(payload.get("target_index") == index
                 and int(payload.get("label", -1)) == int(targets["labels"][index])
                 and int(payload.get("source_validation_index", -1)) ==
                     int(targets["source_validation_indices"][index]),
                 f"display source identity changed at target {index}")
        raw_rows.append(np.stack([
            payload["direct_raw_cmos"][2048][0, 0].numpy(),
            *[payload["outputs"][condition][2048]["raw_cmos"][0, 0].numpy()
              for condition in role_order[1:]],
        ]))
    raw = np.stack(raw_rows).astype(np.float64, copy=False)
    display = _display_uint8(raw, pixel_area, gain)
    arrays = {"display_uint8": display,
              "population_indices": np.arange(TARGET_COUNT, dtype=np.int64),
              "source_validation_indices": targets["source_validation_indices"].numpy().astype(np.int64),
              "labels": targets["labels"].numpy().astype(np.int64),
              "role_order": np.asarray(role_order)}
    array_path = display_dir / "display_uint8.npz"
    if array_path.is_file():
        with np.load(array_path, allow_pickle=False) as saved:
            _require(all(np.array_equal(saved[key], value) for key, value in arrays.items()),
                     "existing Windows display arrays differ from current raw measurements")
    else:
        np.savez_compressed(array_path, **arrays)

    row_height, header_height = 222, 28
    canvas = Image.new("RGB", (5 * 200, TARGET_COUNT * row_height + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((column * 200 + 4, 5), label, fill="black")
    for index in range(TARGET_COUNT):
        y = header_height + index * row_height
        draw.text((4, y + 2), f"target {index} / class {int(targets['labels'][index])}", fill="black")
        for column in range(5):
            image = Image.fromarray(display[index, column], mode="L").convert("RGB")
            canvas.paste(image, (column * 200, y + 20))
    image_path = display_dir / "all_20_targets_2048_contact_sheet.png"
    if image_path.is_file():
        with Image.open(image_path) as saved_image:
            _require(saved_image.size == canvas.size
                     and np.array_equal(np.asarray(saved_image.convert("RGB")), np.asarray(canvas)),
                     "existing Windows display contact sheet differs from current pixel output")
    else:
        canvas.save(image_path)
    identity = _assert_attempt(evidence_dir)
    report = {"schema": "lightgen_g1_fullpop_feasibility_windows_display.v1",
              "task_id": TASK_ID, "run_id": identity.run_id, "role_order": list(role_order),
              "target_count": TARGET_COUNT, "camera_pixel_area_m2": pixel_area,
              "camera_gain_w_inverse": gain,
              "conversion": "raw_W_per_m2*pixel_area_m2*fixed_gain; clip[0,1]; round_to_uint8",
              "native_cmos_normalization": "none", "power_matching": False,
              "gain_fitting": False, "numpy_torch_pixelwise_equal": True,
              "pixel_count_checked": int(display.size),
              "array_file": array_path.name, "contact_sheet": image_path.name}
    _atomic_json(display_dir / "windows_display_reconsumption.json", report)
    return report


def _build_transfer_manifest(evidence_dir: Path, run_id: str) -> dict[str, Any]:
    excluded = {"transfer_manifest.json", "transfer_receipt.json", "windows_reconsumption.json"}
    files = [{"path": path.relative_to(evidence_dir).as_posix(), "bytes": path.stat().st_size}
             for path in sorted(evidence_dir.rglob("*"))
             if path.is_file() and path.name not in excluded and ".tmp." not in path.name]
    payload = {"schema": "lightgen_g1_fullpop_feasibility_transfer.v1",
               "task_id": TASK_ID, "run_id": run_id, "files": files,
               "file_count": len(files), "bytes_total": sum(item["bytes"] for item in files)}
    _atomic_json(evidence_dir / "transfer_manifest.json", payload)
    return payload


def _verify_transfer(evidence_dir: Path) -> dict[str, Any]:
    manifest = _json(evidence_dir / "transfer_manifest.json")
    receipt = _json(evidence_dir / "transfer_receipt.json")
    expected_remote = RUNS_DIR.relative_to(ROOT).as_posix() + "/attempt_01"
    _require(manifest.get("schema") == "lightgen_g1_fullpop_feasibility_transfer.v1"
             and manifest.get("task_id") == TASK_ID
             and manifest.get("run_id") == _json(evidence_dir / "run_identity.json")["run_id"]
             and receipt.get("status") == "COMPLETE"
             and receipt.get("remote_alias") == "autodl-monthly"
             and str(receipt.get("remote_run_dir", "")).rstrip("/").endswith("/" + expected_remote),
             "feasibility transfer is not bound to the AutoDL attempt")
    expected: dict[str, int] = {}
    for entry in manifest.get("files", []):
        relative = Path(str(entry.get("path", "")))
        key = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and key not in expected,
                 "feasibility transfer manifest contains an unsafe or duplicate path")
        expected[key] = int(entry["bytes"])
    _require("transfer_manifest.json" not in expected,
             "feasibility transfer manifest must not list itself")
    expected["transfer_manifest.json"] = (evidence_dir / "transfer_manifest.json").stat().st_size
    received: dict[str, Mapping[str, Any]] = {}
    for entry in receipt.get("files", []):
        relative = Path(str(entry.get("path", "")))
        key = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and key not in received,
                 "feasibility transfer receipt contains an unsafe or duplicate path")
        received[key] = entry
    _require(set(received) == set(expected), "feasibility transfer receipt path set differs from manifest")
    checked = []
    for relative, remote_size in sorted(expected.items()):
        path = evidence_dir / Path(relative)
        item = received[relative]
        current = path.stat().st_size if path.is_file() else -1
        _require(path.is_file() and item.get("status") == "COPIED"
                 and item.get("scp_exit_code") == 0
                 and int(item.get("remote_size_bytes", -1)) == remote_size
                 and int(item.get("local_size_bytes", -1)) == remote_size
                 and current == remote_size,
                 f"feasibility transfer size/status mismatch: {relative}")
        checked.append({"path": relative, "remote_bytes": remote_size,
                        "receipt_local_bytes": int(item["local_size_bytes"]),
                        "current_bytes": current})
    return {"status": "COMPLETE", "file_count": len(checked),
            "bytes_total": sum(item["current_bytes"] for item in checked), "files": checked}


def _verify_review_transfer(review_dir: Path, run_id: str) -> dict[str, Any]:
    manifest = _json(review_dir / "transfer_manifest.json")
    receipt = _json(review_dir / "transfer_receipt.json")
    expected_remote = "/root/autodl-tmp/codex-original/" + REVIEW_DIR.relative_to(ROOT).as_posix()
    _require(manifest.get("schema") == "lightgen_g1_fullpop_phase_metric_review_transfer.v1"
             and manifest.get("task_id") == TASK_ID and manifest.get("run_id") == run_id
             and manifest.get("remote_run_dir") == expected_remote
             and manifest.get("file_count") == len(manifest.get("files", []))
             and manifest.get("bytes_total") == sum(
                 int(item["bytes"]) for item in manifest.get("files", []))
             and receipt.get("status") == "COMPLETE"
             and receipt.get("remote_alias") == "autodl-monthly"
             and str(receipt.get("remote_run_dir", "")).rstrip("/") == expected_remote,
             "phase-metric review transfer is incomplete or belongs to another run")
    expected: dict[str, int] = {}
    for entry in manifest.get("files", []):
        relative = Path(str(entry.get("path", "")))
        key = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and key not in expected,
                 "phase-metric review manifest contains an unsafe or duplicate path")
        expected[key] = int(entry["bytes"])
    _require("transfer_manifest.json" not in expected,
             "phase-metric review manifest must not list itself")
    expected["transfer_manifest.json"] = (review_dir / "transfer_manifest.json").stat().st_size
    received: dict[str, Mapping[str, Any]] = {}
    for entry in receipt.get("files", []):
        relative = Path(str(entry.get("path", "")))
        key = relative.as_posix()
        _require(not relative.is_absolute() and ".." not in relative.parts and key not in received,
                 "phase-metric review receipt contains an unsafe or duplicate path")
        received[key] = entry
    _require(set(received) == set(expected),
             "phase-metric review receipt path set differs from manifest")
    checked = []
    for relative, remote_size in sorted(expected.items()):
        path = review_dir / Path(relative)
        item = received[relative]
        current = path.stat().st_size if path.is_file() else -1
        _require(path.is_file() and item.get("status") == "COPIED"
                 and item.get("scp_exit_code") == 0
                 and int(item.get("remote_size_bytes", -1)) == remote_size
                 and int(item.get("local_size_bytes", -1)) == remote_size
                 and current == remote_size,
                 f"phase-metric review transfer size/status mismatch: {relative}")
        checked.append({"path": relative, "remote_bytes": remote_size,
                        "receipt_local_bytes": int(item["local_size_bytes"]),
                        "current_bytes": current})
    return {"status": "COMPLETE", "file_count": len(checked),
            "bytes_total": sum(item["current_bytes"] for item in checked), "files": checked}


def _verify_preflight_evidence(evidence_dir: Path, source: Mapping[str, Any],
                               targets: Mapping[str, Any],
                               identity: ResearchRunIdentity) -> dict[str, Any]:
    preflight_dir = evidence_dir / "preflight_evidence"
    identity_payload = _json(preflight_dir / "preflight_identity.json")
    ready_identity = ResearchRunIdentity.from_mapping(identity_payload["run_identity"])
    expected_identity = ready_identity.with_scientific_attempt(
        f"{ready_identity.run_id}:attempt_01")
    summary = _json(preflight_dir / "preflight_summary.json")
    receipt = PreflightReceipt.from_mapping(_json(preflight_dir / "preflight_receipt.json"))
    _require(identity_payload.get("task_id") == TASK_ID
             and identity.execution_semantics() == expected_identity.execution_semantics()
             and summary.get("task_id") == TASK_ID
             and summary.get("run_id") == ready_identity.run_id
             and summary.get("status") == "READY"
             and summary.get("source_run_id") == source["identity"].run_id
             and summary.get("scientific_updates") == 0
             and summary.get("optimizer_calls") == 0
             and receipt.status == "READY"
             and receipt.authorization_id == AUTHORITY
             and receipt.source_semantics_id == ready_identity.execution_source_semantics_id,
             "saved feasibility attempt lacks its source-bound READY preflight")
    target_indices = targets["source_validation_indices"].tolist()
    _require(summary.get("target_count") == TARGET_COUNT
             and summary.get("source_validation_indices") == target_indices
             and torch.equal(torch.load(preflight_dir / "fixed_targets.pt", map_location="cpu",
                                       weights_only=False)["source_validation_indices"],
                             targets["source_validation_indices"])
             and fit20._numeric_tree_matches(summary.get("analytic_controls"),
                                              _analytic_controls(), 1.0e-14),
             "saved preflight target identities or analytic controls differ from independent recomputation")

    review = summary.get("source_review", {})
    windows_review = review.get("windows_review_receipt", {})
    positive_summary = summary.get("positive_controls", {})
    recovery_summary = summary.get("solver_recovery", {})
    _require(windows_review.get("status") == "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED"
             and windows_review.get("source_bound_command_replay", {}).get("rows") == fullpop.HOLDOUT_ROWS
             and positive_summary.get("status") == "TEN_POSITIVE_CONTROLS_PASSED"
             and positive_summary.get("count") == CLASS_COUNT
             and positive_summary.get("passed") == CLASS_COUNT
             and recovery_summary.get("status") == "SOLVER_RECOVERY_PASSED"
             and recovery_summary.get("scientific_updates") == 0
             and recovery_summary.get("adam_calls") == 0,
             "saved preflight did not consume its repaired source, positive controls, or recovery check")

    witness_path = _resolve(source["fullpop_config"]["source"]["fit20_witness_measurements"])
    witness = torch.load(witness_path, map_location="cpu", weights_only=False)
    known = source["population"]["known_reachable"]
    expected_commands = witness["amplitudes"][::2].to(torch.float32).contiguous()
    expected_targets = known["coefficients"][::2].contiguous()
    expected_labels = known["labels"][::2].contiguous()
    expected_indices = known["source_validation_indices"][::2].contiguous()
    positive = torch.load(preflight_dir / "positive_control_measurements.pt",
                          map_location="cpu", weights_only=False)
    _require(witness.get("population_role") == "known_reachable_fit20"
             and torch.equal(witness.get("target_coefficients"), known["coefficients"])
             and torch.equal(positive.get("amplitudes"), expected_commands)
             and torch.equal(positive.get("target_coefficients"), expected_targets)
             and torch.equal(positive.get("labels"), expected_labels)
             and torch.equal(positive.get("source_validation_indices"), expected_indices)
             and tuple(expected_commands.shape) == (CLASS_COUNT, 1, 200, 200)
             and expected_labels.tolist() == list(range(CLASS_COUNT)),
             "preflight positive-control commands or targets differ from saved witnesses")
    response = source["inputs"]["response_matrix"].to(dtype=torch.complex128, device="cpu")
    outputs = positive.get("outputs", {}).get("known_reachable_witness", {})
    direct = positive.get("direct_raw_cmos", {})
    recomputed_rows = []
    for grid in GRIDS:
        grid_output = outputs.get(grid, {})
        coefficients = grid_output.get("coefficients")
        raw_cmos = grid_output.get("raw_cmos")
        direct_cmos = direct.get(grid)
        _require(torch.is_tensor(coefficients) and coefficients.dtype == torch.complex128
                 and tuple(coefficients.shape) == (CLASS_COUNT, PORT_COUNT)
                 and torch.isfinite(coefficients.real).all().item()
                 and torch.isfinite(coefficients.imag).all().item()
                 and torch.is_tensor(raw_cmos) and raw_cmos.dtype == torch.float64
                 and tuple(raw_cmos.shape) == (CLASS_COUNT, 1, 200, 200)
                 and torch.isfinite(raw_cmos).all().item()
                 and torch.is_tensor(direct_cmos) and direct_cmos.dtype == torch.float64
                 and tuple(direct_cmos.shape) == (CLASS_COUNT, 1, 200, 200)
                 and torch.isfinite(direct_cmos).all().item(),
                 f"preflight positive-control raw outputs are malformed at grid {grid}")
    predicted = expected_commands.reshape(CLASS_COUNT, PIXEL_COUNT).double().to(torch.complex128) @ response.T
    matrix_error = torch.sqrt(pilot._phase_invariant_loss(
        predicted, outputs[2048]["coefficients"].to(torch.complex128)))
    _require(float(matrix_error.max()) <= projected.MATRIX_LIMIT,
             "preflight positive-control coefficient response differs from the frozen optical path")
    for index in range(CLASS_COUNT):
        row_outputs = {grid: {"coefficients": outputs[grid]["coefficients"][index:index + 1],
                              "raw_cmos": outputs[grid]["raw_cmos"][index:index + 1]}
                       for grid in GRIDS}
        row_direct = {grid: direct[grid][index:index + 1] for grid in GRIDS}
        metrics = projected._error_row(row_outputs, expected_targets[index:index + 1], row_direct, 0)
        recomputed_rows.append({"target_index": index,
                                "condition": "known_reachable_witness", **metrics,
                                "row_passed": all(metrics[key] <= GATE_LIMIT for key in METRIC_KEYS),
                                "label": int(expected_labels[index]),
                                "source_validation_index": int(expected_indices[index])})
    _require(all(row["row_passed"] for row in recomputed_rows)
             and fit20._numeric_tree_matches(positive.get("rows"), recomputed_rows, 1.0e-14)
             and positive_summary.get("passed") == sum(row["row_passed"] for row in recomputed_rows)
             and fit20._numeric_tree_matches(positive_summary.get("rows"), recomputed_rows, 1.0e-14),
             "preflight positive-control result differs from raw full-optics arrays")

    recovery_dir = preflight_dir / "solver_recovery"
    checkpoint, live, restored = (torch.load(recovery_dir / name, map_location="cpu", weights_only=False)
                                  for name in ("checkpoint_target_000_fixed_000003.pt",
                                               "original_process_step4.pt",
                                               "independent_process_step4.pt"))
    expected_source = source["identity"].run_id
    expected_recovery_rows = ((checkpoint, 3), (live, 4), (restored, 4))
    _require(all(item.get("task_id") == TASK_ID and item.get("run_id") == ready_identity.run_id
                 and item.get("source_run_id") == expected_source
                 and item.get("target_cursor") == 0 and item.get("branch") == "fixed_phase"
                 and item.get("iteration") == step
                 for item, step in expected_recovery_rows),
             "preflight solver recovery states have the wrong task, source, or cursor")
    checkpoint_state = checkpoint.get("solver_state", {})
    live_state, restored_state = live.get("state", {}), restored.get("state", {})
    expected_state_keys = {"amplitude", "best_amplitude", "best_error", "iteration", "loss"}
    _require(all(set(state) == expected_state_keys for state in
                 (checkpoint_state, live_state, restored_state))
             and checkpoint_state.get("iteration") == 3
             and live_state.get("iteration") == restored_state.get("iteration") == 4,
             "preflight solver recovery state fields or iterations are incomplete")
    state_deltas = []
    for key in ("amplitude", "best_amplitude", "best_error"):
        left, right = live_state[key], restored_state[key]
        _require(torch.is_tensor(left) and torch.is_tensor(right)
                 and left.shape == right.shape and left.dtype == right.dtype,
                 f"preflight solver recovery tensor is malformed: {key}")
        state_deltas.append(float((left - right).abs().max()))
    state_delta = max(state_deltas)
    loss_delta = abs(float(live_state["loss"]) - float(restored_state["loss"]))
    _require(state_delta <= RECOVERY_TOLERANCE and loss_delta <= RECOVERY_TOLERANCE
             and torch.equal(live["amplitude"], live_state["best_amplitude"])
             and torch.equal(restored["amplitude"], restored_state["best_amplitude"]),
             "preflight original and independent fourth solver states differ")
    target = targets["coefficients"][0].to(torch.complex128)
    phase = float(targets["box_b_public_phase_rad"][0])
    matrix = response
    target_norm = torch.linalg.vector_norm(target)
    for item, state in ((checkpoint, checkpoint_state), (live, live_state), (restored, restored_state)):
        actual = state["best_amplitude"].to(torch.complex128).reshape(1, -1) @ matrix.T
        expected_target = target * torch.exp(torch.tensor(1j * phase, dtype=torch.complex128))
        residual = actual.reshape(-1) - expected_target
        relative_error = float(torch.linalg.vector_norm(residual) / target_norm)
        loss = 0.5 * float(residual.abs().square().sum())
        if item is not checkpoint:
            _require(abs(relative_error - float(item.get("relative_error", math.inf)))
                     <= RECOVERY_TOLERANCE
                     and abs(relative_error - float(state["best_error"])) <= RECOVERY_TOLERANCE,
                     "preflight solver recovery residual does not match its saved result")
        _require(abs(loss - float(state["loss"])) <= RECOVERY_TOLERANCE,
                 "preflight solver recovery loss differs from its saved amplitude")
    _require(recovery_summary.get("state_max_abs_delta", math.inf) <= RECOVERY_TOLERANCE
             and recovery_summary.get("loss_abs_delta", math.inf) <= RECOVERY_TOLERANCE
             and recovery_summary.get("iteration_exact") is True,
             "preflight recovery summary does not match its raw solver states")
    return {"status": "PREFLIGHT_EVIDENCE_RECONSUMED", "ready": True,
            "positive_control_rows": len(recomputed_rows),
            "positive_controls_all_passed": True,
            "solver_recovery_state_max_abs_delta": state_delta,
            "solver_recovery_loss_abs_delta": loss_delta,
            "scientific_updates": 0, "optimizer_calls": 0}


def _verify_saved(evidence_dir: Path, config: Mapping[str, Any], *, review_dir: Path) -> dict[str, Any]:
    _require(review_dir is not None,
             "verify-saved requires the explicit transferred phase-metric review directory", ValueError)
    sizes = _verify_transfer(evidence_dir)
    _set_numerics()
    source = _load_sources(config, "windows")
    identity = _assert_attempt(evidence_dir)
    review_sizes = _verify_review_transfer(review_dir, identity.run_id) if review_dir is not None else None
    if review_dir is not None:
        review_identity = _json(review_dir / "review_identity.json")
        _require(review_identity.get("schema") == "lightgen_g1_fullpop_phase_metric_review.v1"
                 and review_identity.get("task_id") == TASK_ID
                 and review_identity.get("run_id") == identity.run_id
                 and review_identity.get("source_run_id") == source["identity"].run_id
                 and review_identity.get("status") == "COMPLETE"
                 and review_identity.get("parent_attempt_state") == RUN_COMPLETED
                 and all(review_identity.get(key) == 0 for key in
                         ("scientific_updates", "optimizer_calls", "optical_updates", "evaluator_updates")),
                 "phase-metric review identity or zero-update scope is invalid")
    _require(identity.run_id == _json(evidence_dir / "run_identity.json")["run_id"],
             "feasibility attempt identity changed")
    targets = torch.load(evidence_dir / "fixed_targets.pt", map_location="cpu", weights_only=False)
    expected_targets = _target_payload(config, source, torch.device("cpu"))
    _require(torch.equal(targets.get("source_validation_indices"), expected_targets["source_validation_indices"])
             and torch.equal(targets.get("coefficients"), expected_targets["coefficients"])
             and torch.equal(targets.get("labels"), expected_targets["labels"])
             and torch.equal(targets.get("proposal_ids"), expected_targets["proposal_ids"])
             and torch.allclose(targets.get("box_b_public_phase_rad"),
                                expected_targets["box_b_public_phase_rad"],
                                atol=SOURCE_COMMAND_TOLERANCE, rtol=0.0)
             and targets.get("source_run_id") == source["identity"].run_id
             and targets.get("task_id") == TASK_ID
             and targets.get("population_role") == "development_feasibility20",
             "saved feasibility population differs from fixed source rows")
    target_indices = targets["source_validation_indices"].tolist()
    preflight = _verify_preflight_evidence(evidence_dir, source, targets, identity)
    repair_path = evidence_dir / "measurement_row_identity_repair.json"
    repair_summary = {"status": "NOT_REQUIRED", "repaired_rows": 0,
                      "raw_measurement_arrays_unchanged": True,
                      "fixed_phase_metric_repair": {"status": "NOT_REQUIRED",
                                                     "targets_recomputed": 0,
                                                     "updated_targets": 0,
                                                     "decision_changes": 0,
                                                     "raw_measurement_arrays_unchanged": True}}
    phase_metric_repair_by_target: dict[int, dict[str, Any]] = {}
    if repair_path.is_file():
        repair = _json(repair_path)
        repair_rows = repair.get("rows")
        _require(repair.get("schema") ==
                 "lightgen_g1_fullpop_measurement_row_identity_repair.v1"
                 and repair.get("task_id") == TASK_ID
                 and repair.get("run_id") == identity.run_id
                 and repair.get("status") == "COMPLETE"
                 and repair.get("raw_measurement_arrays_unchanged") is True
                 and isinstance(repair_rows, list)
                 and all(isinstance(entry, dict) for entry in repair_rows),
                 "measurement row identity repair receipt is incomplete or mismatched")
        repair_keys = []
        for entry in repair_rows:
            section = entry.get("section")
            target_index = entry.get("target_index")
            condition = entry.get("condition")
            _require(section in {"baseline", "conditions"}
                     and isinstance(target_index, int) and 0 <= target_index < TARGET_COUNT
                     and condition in (("frozen_v2", "box_b5000") if section == "baseline"
                                       else CONDITIONS)
                     and entry.get("new_target_index") == target_index
                     and entry.get("new_label") == int(targets["labels"][target_index])
                     and entry.get("new_source_validation_index") ==
                         int(targets["source_validation_indices"][target_index])
                     and (entry.get("old_target_index") == target_index
                          or (target_index > 0 and entry.get("old_target_index") == 0))
                     and (entry.get("old_label") is None
                          or entry.get("old_label") == entry.get("new_label"))
                     and (entry.get("old_source_validation_index") is None
                          or entry.get("old_source_validation_index") ==
                              entry.get("new_source_validation_index")),
                     "measurement row identity repair entry is invalid")
            repair_keys.append((section, target_index, condition))
        expected_repair_keys = {
            *(('baseline', index, condition) for index in range(TARGET_COUNT)
              for condition in ("frozen_v2", "box_b5000")),
            *(('conditions', index, condition) for index in range(1, TARGET_COUNT)
              for condition in CONDITIONS),
        }
        _require(len(set(repair_keys)) == len(repair_keys)
                 and set(repair_keys) == expected_repair_keys,
                 "measurement row identity repair list does not cover the complete failed aggregation")
        phase_repair = (_json(review_dir / "fixed_phase_metric_repair.json")
                        if review_dir is not None else repair.get("fixed_phase_metric_repair", {}))
        phase_repair_rows = phase_repair.get("rows")
        _require(phase_repair.get("schema") == "lightgen_g1_fullpop_fixed_phase_metric_repair.v1"
                 and phase_repair.get("status") == "COMPLETE"
                 and phase_repair.get("raw_measurement_arrays_unchanged") is True
                 and phase_repair.get("recomputed_from_saved_actual_coefficients") is True
                 and (review_dir is None or
                      (phase_repair.get("task_id") == TASK_ID
                       and phase_repair.get("run_id") == identity.run_id
                       and phase_repair.get("source_run_id") == source["identity"].run_id))
                 and isinstance(phase_repair_rows, list)
                 and len(phase_repair_rows) == TARGET_COUNT
                 and all(isinstance(row, dict) for row in phase_repair_rows),
                 "fixed-phase metric repair record is incomplete or mismatched")
        phase_indices = [row.get("target_index") for row in phase_repair_rows]
        _require(all(isinstance(index, int) for index in phase_indices)
                 and sorted(phase_indices) == list(range(TARGET_COUNT)),
                 "fixed-phase metric repair does not cover every target exactly once")
        for entry in phase_repair_rows:
            index = int(entry["target_index"])
            for field in ("previous_saved_errors", "previous_cpu_reproduction_errors", "corrected_errors"):
                errors = entry.get(field)
                _require(isinstance(errors, dict) and set(errors) == {str(grid) for grid in GRIDS}
                         and all(math.isfinite(float(errors[str(grid)]))
                                 and float(errors[str(grid)]) >= 0.0 for grid in GRIDS),
                         f"fixed-phase metric repair errors are malformed at target {index}/{field}")
            _require(entry.get("previous_saved_passed") is
                         (max(float(x) for x in entry["previous_saved_errors"].values()) <= GATE_LIMIT)
                     and entry.get("previous_cpu_reproduction_passed") is
                         (max(float(x) for x in entry["previous_cpu_reproduction_errors"].values()) <= GATE_LIMIT)
                     and entry.get("corrected_passed") is
                         (max(float(x) for x in entry["corrected_errors"].values()) <= GATE_LIMIT),
                     f"fixed-phase metric repair decision fields are inconsistent at target {index}")
            phase_metric_repair_by_target[index] = entry
        phase_metric_summary = {
            "status": "COMPLETE", "targets_recomputed": len(phase_repair_rows),
            "updated_targets": sum(row["previous_saved_errors"] != row["corrected_errors"]
                                   for row in phase_repair_rows),
            "decision_changes": sum(row["previous_saved_passed"] != row["corrected_passed"]
                                     for row in phase_repair_rows),
            "raw_measurement_arrays_unchanged": True,
        }
        _require(phase_repair.get("targets_recomputed") == phase_metric_summary["targets_recomputed"]
                 and phase_repair.get("updated_targets") == phase_metric_summary["updated_targets"]
                 and phase_repair.get("decision_changes") == phase_metric_summary["decision_changes"],
                 "fixed-phase metric repair summary differs from its per-target records")
        repair_summary = {"status": repair["status"], "repaired_rows": len(repair_rows),
                          "raw_measurement_arrays_unchanged": True,
                          "fixed_phase_metric_repair": phase_metric_summary}
    sources = torch.load(evidence_dir / "mapper_source_states.pt", map_location="cpu", weights_only=False)
    _require(sources.get("task_id") == TASK_ID
             and sources.get("source_run_id") == source["identity"].run_id
             and fullpop._state_dicts_equal(sources.get("mapper_states", {}).get("frozen_v2_baseline"),
                                            source["source_states"]["frozen_v2_baseline"])
             and fullpop._state_dicts_equal(sources.get("mapper_states", {}).get("box_b_candidate"),
                                            source["source_states"]["box_b_candidate"]),
             "saved feasibility mapper sources differ from repaired explicit checkpoints")
    command_replay = torch.load(review_dir / "mapper_command_replay.pt",
                                map_location="cpu", weights_only=False)
    command_replay_rows = command_replay.get("rows", [])
    _require(command_replay.get("schema") == "lightgen_g1_fullpop_mapper_command_replay.v1"
             and command_replay.get("task_id") == TASK_ID
             and command_replay.get("run_id") == identity.run_id
             and command_replay.get("source_run_id") == source["identity"].run_id
             and command_replay.get("device") == "cuda:0"
             and command_replay.get("exact_command_count") == TARGET_COUNT
             and len(command_replay_rows) == TARGET_COUNT
             and [row.get("target_index") for row in command_replay_rows] == list(range(TARGET_COUNT)),
             "saved exact CUDA mapper-command replay is incomplete or misidentified")
    solution = torch.load(evidence_dir / "solver_solutions.pt", map_location="cpu", weights_only=False)
    _require(solution.get("task_id") == TASK_ID
             and solution.get("run_id") == identity.run_id
             and solution.get("source_fullpop_run_id") == source["identity"].run_id
             and torch.equal(solution.get("source_validation_indices"),
                             targets["source_validation_indices"])
             and len(solution.get("rows", [])) == TARGET_COUNT,
             "saved solver terminal is missing targets or has a changed source")
    response = source["inputs"]["response_matrix"].to(device="cpu", dtype=torch.complex128)
    total_iterations = 0
    solver_response_max_error = 0.0
    for index, row in enumerate(solution["rows"]):
        _require(row.get("target_index") == index
                 and row.get("source_validation_index") == int(targets["source_validation_indices"][index]),
                 f"solver target order differs at row {index}")
        for branch, starts in (("fixed_phase", 1), ("free_phase", 8)):
            result = row[branch]
            _require(0 < int(result["iterations"]) <= SOLVER_STEPS
                     and torch.is_tensor(result.get("amplitude"))
                     and result["amplitude"].dtype == torch.float64
                     and tuple(result["amplitude"].shape) == (PIXEL_COUNT,)
                     and torch.isfinite(result["amplitude"]).all().item()
                     and float(result["amplitude"].min()) >= 0.0
                     and float(result["amplitude"].max()) <= 1.0
                     and math.isfinite(float(result["relative_error"])),
                     f"saved {branch} solution is malformed at target {index}")
            if branch == "free_phase":
                best_errors = result.get("best_errors")
                _require(torch.is_tensor(best_errors) and tuple(best_errors.shape) == (8,)
                         and best_errors.dtype == torch.float64
                         and torch.isfinite(best_errors).all().item()
                         and float(best_errors.min()) >= 0.0
                         and abs(float(best_errors.min()) - float(result["relative_error"])) <= 1.0e-12,
                         f"saved free-phase best-start record differs from selected solution at {index}")
            total_iterations += int(result["iterations"]) * starts
            predicted = response @ result["amplitude"].to(torch.complex128)
            if branch == "fixed_phase":
                phase = float(targets["box_b_public_phase_rad"][index])
                expected = _phase_shifted_target(targets["coefficients"][index], phase)
                response_error = _fixed_phase_relative_error(predicted, expected)
                _require(abs(float(result["phase_rad"]) - phase) <= 1.0e-14,
                         f"fixed-phase solver changed its prescribed phase at target {index}")
            else:
                expected_phase, response_error = reachability._phase_error(
                    predicted, targets["coefficients"][index].to(torch.complex128))
                phase_delta = math.atan2(math.sin(float(result["phase_rad"]) - expected_phase),
                                         math.cos(float(result["phase_rad"]) - expected_phase))
                _require(abs(phase_delta) <= 1.0e-12,
                         f"free-phase solver phase differs from its saved amplitude at target {index}")
            _require(abs(float(result["relative_error"]) - response_error) <= 1.0e-12,
                     f"saved {branch} coefficient residual differs from independent CPU response at {index}")
    _require(total_iterations <= TOTAL_START_ITERATION_LIMIT
             and solution.get("total_start_iterations") == total_iterations,
             "saved solver exceeds its fixed start-iteration budget")
    conditions_path = ((review_dir / "conditions.pt") if review_dir is not None else
                       (evidence_dir / "measurements" / "conditions.pt"))
    conditions = torch.load(conditions_path,
                            map_location="cpu", weights_only=False)
    rows = conditions.get("rows", [])
    _require(conditions.get("schema") == "lightgen_g1_fullpop_feasibility_conditions.v1"
             and conditions.get("task_id") == TASK_ID
             and conditions.get("run_id") == identity.run_id
             and len(rows) == TARGET_COUNT * len(CONDITIONS),
             "saved full-optics feasibility measurements are incomplete")
    review_rows_by_target = {(int(row["target_index"]), row["condition"]): row for row in rows}
    _require(len(review_rows_by_target) == len(rows),
             "saved review condition rows contain duplicate target-condition identities")
    raw_rows = []
    for index in range(TARGET_COUNT):
        path = evidence_dir / "measurements" / "conditions" / f"target_{index:02d}.pt"
        _require(path.is_file(), f"raw full-optics measurement is missing: {index}", FileNotFoundError)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(payload.get("run_id") == identity.run_id
                 and payload.get("target_index") == index
                 and payload.get("label") == int(targets["labels"][index])
                 and payload.get("source_validation_index") ==
                     int(targets["source_validation_indices"][index])
                 and torch.equal(payload.get("target_coefficients"), targets["coefficients"][index]),
                 f"raw full-optics target differs from fixed source at {index}")
        baseline_payload = torch.load(
            evidence_dir / "measurements" / "baseline" / f"target_{index:02d}.pt",
            map_location="cpu", weights_only=False)
        baseline_rows = baseline_payload.get("rows")
        _require(isinstance(baseline_rows, list) and len(baseline_rows) == 2
                 and all(isinstance(row, dict) for row in baseline_rows)
                 and {row.get("condition") for row in baseline_rows} ==
                     {"frozen_v2", "box_b5000"}
                 and all(row.get("target_index") == index
                         and row.get("label") == int(targets["labels"][index])
                         and row.get("source_validation_index") ==
                             int(targets["source_validation_indices"][index])
                         for row in baseline_rows),
                 f"baseline metric-row identity differs from its target shard at {index}")
        saved_v2 = baseline_payload.get("commands", {}).get("frozen_v2")
        saved_box = baseline_payload.get("commands", {}).get("box_b5000")
        replay_row = command_replay_rows[index]
        _require(replay_row.get("source_validation_index") ==
                     int(targets["source_validation_indices"][index])
                 and torch.equal(replay_row.get("target_coefficients"), targets["coefficients"][index]),
                 f"exact CUDA mapper-command replay target identity differs at {index}")
        _require_exact_command_replay(replay_row.get("commands", {}).get("frozen_v2"),
                                      saved_v2, index, "frozen_v2")
        _require_exact_command_replay(replay_row.get("commands", {}).get("box_b5000"),
                                      saved_box, index, "box_b5000")
        _require(replay_row.get("box_b_public_phase_rad") ==
                     float(baseline_payload.get("b_public_phase_rad", math.nan))
                 and torch.equal(replay_row.get("box_b_preclip_amplitude"),
                                 baseline_payload.get("b_preclip_amplitude")),
                 f"saved baseline phase or preclip amplitude differs from exact CUDA replay at {index}")
        solution_row = solution["rows"][index]
        expected_commands = {
            "frozen_v2": baseline_payload["commands"]["frozen_v2"],
            "box_b5000": baseline_payload["commands"]["box_b5000"],
            "fixed_phase_solver": solution_row["fixed_phase"]["amplitude"].to(torch.float32)
                .reshape(1, 1, 200, 200),
            "free_phase_solver": solution_row["free_phase"]["amplitude"].to(torch.float32)
                .reshape(1, 1, 200, 200),
        }
        _require(baseline_payload.get("source_run_id") == source["identity"].run_id
                 and baseline_payload.get("target_index") == index
                 and torch.equal(baseline_payload.get("target_coefficients"), targets["coefficients"][index])
                 and float(baseline_payload.get("b_public_phase_rad", math.nan)) ==
                     float(targets["box_b_public_phase_rad"][index])
                 and set(payload.get("commands", {})) == set(CONDITIONS)
                 and set(payload.get("rows", {})) == set(CONDITIONS)
                 and all(torch.equal(payload["commands"][condition], expected_commands[condition])
                         for condition in CONDITIONS)
                 and all(payload["commands"][condition].dtype == torch.float32
                         and torch.isfinite(payload["commands"][condition]).all().item()
                         and float(payload["commands"][condition].min()) >= 0.0
                         and float(payload["commands"][condition].max()) <= 1.0
                         for condition in CONDITIONS)
                 and float(payload.get("box_b_public_phase_rad", math.nan)) ==
                     float(targets["box_b_public_phase_rad"][index]),
                 f"full-optics commands are not bound to their mapper or solver sources at {index}")
        preclip = baseline_payload["b_preclip_amplitude"]
        baseline_box_coefficients = baseline_payload["outputs"]["box_b5000"][2048]["coefficients"][0]
        recomputed_decomposition = _residual_decomposition(
            response, targets["coefficients"][index], float(targets["box_b_public_phase_rad"][index]),
            preclip, baseline_payload["commands"]["box_b5000"], baseline_box_coefficients)
        saved_decomposition = baseline_payload.get("residual_decomposition", {})
        vector_names = ("preclip", "box_clip", "float32_rounding", "optical_response")
        decomposition_summary = {key: value for key, value in recomputed_decomposition.items()
                                 if key != "term_vectors"}
        saved_summary = {key: value for key, value in saved_decomposition.items()
                         if key != "term_vectors"}
        _require(all(torch.allclose(saved_decomposition.get("term_vectors", {}).get(name),
                                    recomputed_decomposition["term_vectors"][name],
                                    atol=1.0e-10, rtol=1.0e-12)
                     for name in vector_names)
                 and fit20._numeric_tree_matches(saved_summary, decomposition_summary, 1.0e-10),
                 f"saved residual decomposition differs from independent CPU reconstruction at {index}")
        for condition in CONDITIONS:
            command = payload["commands"][condition][0, 0].reshape(-1).to(torch.float64)
            predicted_response = response @ command.to(torch.complex128)
            actual_response = payload["outputs"][condition][2048]["coefficients"][0].reshape(-1)
            response_error = float(torch.linalg.vector_norm(predicted_response - actual_response)
                                   / torch.linalg.vector_norm(
                                       targets["coefficients"][index].to(torch.complex128)))
            solver_response_max_error = max(solver_response_max_error, response_error)
            _require(response_error <= 1.0e-6,
                     f"full-optics 2048 coefficient differs from fixed response at target {index}, {condition}")
        _require(all(torch.equal(payload["direct_raw_cmos"][grid], baseline_payload["direct_raw_cmos"][grid])
                     for grid in GRIDS),
                 f"direct Decoder reference differs across paired measurements at {index}")
        predicted_metrics = {}
        for condition in CONDITIONS:
            outputs = {grid: payload["outputs"][condition][grid] for grid in GRIDS}
            metric = projected._error_row(outputs, payload["target_coefficients"].reshape(1, -1),
                                          payload["direct_raw_cmos"], 0)
            original_stored = payload["rows"][condition]
            stored = review_rows_by_target[(index, condition)] if review_dir is not None else original_stored
            _require(original_stored.get("target_index") == index
                     and original_stored.get("label") == int(targets["labels"][index])
                     and original_stored.get("source_validation_index") ==
                         int(targets["source_validation_indices"][index])
                     and original_stored.get("condition") == condition
                     and stored.get("target_index") == index
                     and stored.get("label") == int(targets["labels"][index])
                     and stored.get("source_validation_index") ==
                         int(targets["source_validation_indices"][index])
                     and stored.get("condition") == condition,
                     f"stored condition metric-row identity differs from its target shard at {index}/{condition}")
            _require(all(abs(float(original_stored[key]) - float(metric[key])) <= 1.0e-14
                         and abs(float(stored[key]) - float(metric[key])) <= 1.0e-14
                         for key in METRIC_KEYS),
                     f"saved {condition} raw measurement metrics changed at target {index}")
            metric["row_passed"] = all(metric[key] <= GATE_LIMIT for key in METRIC_KEYS)
            metric["label"] = int(payload["label"])
            metric["source_validation_index"] = int(payload["source_validation_index"])
            metric["target_index"] = index
            metric["condition"] = condition
            if condition == "fixed_phase_solver":
                phase = float(targets["box_b_public_phase_rad"][index])
                errors = _fixed_phase_grid_errors(outputs, targets["coefficients"][index], phase)
                legacy_target = targets["coefficients"][index] * torch.exp(
                    torch.tensor(1j * phase, dtype=torch.complex128))
                legacy_errors = _fixed_phase_grid_errors(outputs, legacy_target, 0.0)
                phase_repair = phase_metric_repair_by_target.get(index)
                if phase_repair is not None:
                    _require(fullpop.fit20._numeric_tree_matches(
                                 phase_repair["previous_cpu_reproduction_errors"],
                                 legacy_errors, LEGACY_PHASE_ERROR_CROSS_DEVICE_TOLERANCE)
                             and fullpop.fit20._numeric_tree_matches(
                                 phase_repair["corrected_errors"], errors, 1.0e-14)
                             and phase_repair["previous_cpu_reproduction_passed"] is
                                 (max(legacy_errors.values()) <= GATE_LIMIT)
                             and phase_repair["corrected_passed"] is
                                 (max(errors.values()) <= GATE_LIMIT),
                             f"fixed-phase correction audit differs from saved raw coefficients at {index}")
                    historical_errors_match = all(
                        abs(float(payload["fixed_phase_target_errors"][str(grid)])
                            - float(phase_repair["previous_saved_errors"][str(grid)]))
                            <= LEGACY_PHASE_ERROR_CROSS_DEVICE_TOLERANCE
                        and abs(float(original_stored["fixed_phase_target_errors"][str(grid)])
                                - float(phase_repair["previous_saved_errors"][str(grid)]))
                                <= LEGACY_PHASE_ERROR_CROSS_DEVICE_TOLERANCE
                        for grid in GRIDS)
                    _require(historical_errors_match
                             and payload.get("fixed_phase_target_passed") is
                                 phase_repair["previous_saved_passed"]
                             and original_stored.get("fixed_phase_target_passed") is
                                 phase_repair["previous_saved_passed"],
                             f"phase-metric repair does not preserve the historical metric at {index}")
                fixed_phase_passed = max(errors.values()) <= GATE_LIMIT
                stored_phase_errors = stored.get("fixed_phase_target_errors")
                _require(isinstance(stored_phase_errors, dict)
                         and all(abs(errors[str(grid)] - float(stored_phase_errors[str(grid)]))
                                 <= 1.0e-14 for grid in GRIDS)
                         and stored.get("fixed_phase_target_passed") is fixed_phase_passed,
                         f"fixed-phase target error changed at target {index}")
                metric["fixed_phase_target_errors"] = errors
                metric["fixed_phase_target_passed"] = fixed_phase_passed
            metric["condition_passed"] = metric["row_passed"]
            if condition != "fixed_phase_solver":
                metric["fixed_phase_target_errors"] = None
            predicted_metrics[condition] = metric
            raw_rows.append(metric)
        _require(set(predicted_metrics) == set(CONDITIONS),
                 f"not all four conditions were recomputed at target {index}")
    summary = _summarize_conditions(raw_rows)
    witness = _reachability_summary(raw_rows)
    _require(fullpop.fit20._numeric_tree_matches(conditions.get("summaries"), summary, 1.0e-14)
             and fullpop.fit20._numeric_tree_matches(conditions.get("rows"), raw_rows, 1.0e-14)
             and fullpop.fit20._numeric_tree_matches(conditions.get("reachability"), witness, 1.0e-14),
             "saved feasibility rows or summary differs from independent raw-array recomputation")
    final_decision = _json((review_dir / "final_decision.json") if review_dir is not None else
                           (evidence_dir / "final_decision.json"))
    solver_counts = {"fixed_phase_iterations": sum(int(row["fixed_phase"]["iterations"])
                                                      for row in solution["rows"]),
                     "free_phase_start_iterations": sum(int(row["free_phase"]["iterations"]) * 8
                                                          for row in solution["rows"]),
                     "total_start_iterations": total_iterations,
                     "max_total_start_iterations": TOTAL_START_ITERATION_LIMIT,
                     "scientific_updates": 0, "optimizer_calls": 0}
    _require(final_decision.get("schema") == "lightgen_g1_fullpop_feasibility_decision.v1"
             and final_decision.get("task_id") == TASK_ID
             and final_decision.get("run_id") == identity.run_id
             and final_decision.get("source_fullpop_run_id") == source["identity"].run_id
             and final_decision.get("lifecycle_state") == RUN_COMPLETED
             and final_decision.get("diagnostic_result") == witness["classification"]
             and final_decision.get("source_validation_indices") == target_indices
             and fullpop.fit20._numeric_tree_matches(final_decision.get("condition_summaries"),
                                                      summary, 1.0e-14)
             and final_decision.get("measurement_row_identity_repair") == repair_summary
             and final_decision.get("fixed_phase_metric_repair") ==
                 repair_summary["fixed_phase_metric_repair"]
             and fullpop.fit20._numeric_tree_matches(final_decision.get("reachability"),
                                                      witness, 1.0e-14)
             and fullpop.fit20._numeric_tree_matches(final_decision.get("solver"),
                                                      solver_counts, 1.0e-14)
             and final_decision.get("scientific_updates") == 0
             and final_decision.get("optimizer_calls") == 0
             and final_decision.get("optical_updates") == 0
             and final_decision.get("evaluator_updates") == 0,
             "saved diagnosis decision differs from independent raw-array recomputation")
    task_state = _json(evidence_dir / "task_state.json")
    lifecycle = _json(evidence_dir / "research_lifecycle_state.json")
    original_decision = _json(evidence_dir / "final_decision.json")
    _require(task_state.get("task_id") == TASK_ID and task_state.get("run_id") == identity.run_id
             and task_state.get("state") == RUN_COMPLETED
             and task_state.get("target_cursor") == TARGET_COUNT
             and task_state.get("committed_start_iterations") == total_iterations
             and task_state.get("scientific_updates") == 0
             and task_state.get("optimizer_calls") == 0
             and task_state.get("optical_updates") == 0
             and task_state.get("evaluator_updates") == 0
             and lifecycle.get("task_id") == TASK_ID and lifecycle.get("run_id") == identity.run_id
             and lifecycle.get("state") == RUN_COMPLETED
              and lifecycle.get("diagnostic_result") == original_decision.get("diagnostic_result")
             and all(lifecycle.get(key) == 0 for key in
                     ("scientific_updates", "optimizer_calls", "optical_updates", "evaluator_updates")),
             "saved terminal lifecycle state or update scope differs from the reconsumed result")
    report = ((review_dir / "final_report.md") if review_dir is not None else
              (evidence_dir / "final_report.md")).read_text(encoding="utf-8")
    _require(all(section in report for section in
                 ("## 结果与状态", "## 科学认识", "## 目标缺口", "## 工程完成与验证",
                  "## 决策影响与下一步"))
             and witness["classification"] in report
             and f"{witness['any_condition_witness_rows']}/20" in report
             and f"{witness['fixed_phase_witness_rows']}/20" in report
              and f"云端CUDA映射器指令逐位重放{TARGET_COUNT}条目标" in report
             and f"固定相位误差在{repair_summary['fixed_phase_metric_repair']['targets_recomputed']}条保存输出上重新计算" in report
             and ("测量行身份修复记录" in report if repair_summary["repaired_rows"] else
                  "本次无需修复测量行身份" in report),
             "saved five-part diagnosis report does not describe the independently recomputed result")
    display = _render_windows_display(evidence_dir, source, targets, output_dir=review_dir)
    receipt = {"schema": "lightgen_g1_fullpop_feasibility_windows_reconsumption.v1",
               "task_id": TASK_ID, "run_id": identity.run_id,
               "status": "WINDOWS_EVIDENCE_RECONSUMPTION_PASSED",
               "scientific_updates": 0, "optimizer_calls": 0,
               "source_fullpop_windows_review": source["review"].get("windows_review_receipt", {}).get("status"),
               "population_rows_recomputed": TARGET_COUNT,
               "condition_rows_recomputed": len(raw_rows),
               "all_six_metrics_recomputed": True,
               "preflight_evidence": preflight,
               "total_start_iterations": total_iterations,
               "fixed_response_2048_max_relative_error": solver_response_max_error,
                "exact_cuda_mapper_command_replay_targets": command_replay.get("exact_command_count"),
                "file_size_reconciliation": {"attempt": sizes,
                                             "review": review_sizes} if review_dir is not None else sizes,
               "measurement_row_identity_repair": repair_summary,
               "fixed_phase_metric_repair": repair_summary["fixed_phase_metric_repair"],
               "reachability": witness, "condition_summaries": summary,
               "windows_display_reconsumption": display}
    _atomic_json((review_dir or evidence_dir) / "windows_reconsumption.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True,
                        choices=("preflight", "execute", "resume", "measure", "review-saved",
                                 "verify-saved", "recovery-worker"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, help="explicit solver checkpoint required for resume")
    parser.add_argument("--evidence-dir", type=Path, help="explicit local evidence for verify-saved")
    parser.add_argument("--review-dir", type=Path, help="explicit phase-metric review output or evidence")
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = _load_config(config_path)
    if args.mode == "recovery-worker":
        _require(args.checkpoint is not None and args.evidence_dir is not None and args.review_dir is None
                 and args.worker_output is not None,
                 "recovery-worker requires an explicit checkpoint, evidence directory, and output", ValueError)
        _recovery_worker(config_path, args.checkpoint.resolve(), args.worker_output.resolve(),
                         args.evidence_dir.resolve())
        result = {"status": "SOLVER_RECOVERY_STEP_COMPLETE"}
    elif args.mode == "preflight":
        _require(args.checkpoint is None and args.evidence_dir is None and args.review_dir is None,
                 "preflight accepts no checkpoint or evidence directory", ValueError)
        result = _preflight(config_path, config)
    elif args.mode == "execute":
        _require(args.checkpoint is None and args.evidence_dir is None and args.review_dir is None,
                 "execute starts from the fixed preflight and accepts no checkpoint", ValueError)
        result = _execute(config_path, config)
    elif args.mode == "resume":
        _require(args.checkpoint is not None and args.evidence_dir is None and args.review_dir is None,
                 "resume requires an explicit --checkpoint", ValueError)
        result = _execute(config_path, config, resume_checkpoint=args.checkpoint.resolve())
    elif args.mode == "measure":
        _require(args.checkpoint is None and args.evidence_dir is None and args.review_dir is None,
                 "measure uses the completed fixed solver outputs", ValueError)
        result = _measure_attempt(config)
    elif args.mode == "review-saved":
        _require(args.checkpoint is None and args.evidence_dir is not None and args.review_dir is not None,
                 "review-saved requires explicit completed evidence and a new review directory", ValueError)
        result = _review_saved_phase_metrics(args.evidence_dir.resolve(), args.review_dir.resolve(), config)
    else:
        _require(args.checkpoint is None and args.evidence_dir is not None and args.review_dir is not None,
                 "verify-saved requires explicit original evidence and transferred review evidence", ValueError)
        result = _verify_saved(args.evidence_dir.resolve(), config, review_dir=args.review_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
