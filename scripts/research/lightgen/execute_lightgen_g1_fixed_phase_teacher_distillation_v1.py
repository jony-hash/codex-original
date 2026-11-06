"""Train one fixed-phase G1 mapper from measured legal-command teachers."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
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
    PreflightReceipt, ResearchRunControl, ResearchRunIdentity,
)
from scripts.research.lightgen import execute_lightgen_g1_fixed_frame_semantic_v1 as fixed
from scripts.research.lightgen import execute_lightgen_g1_semantic_exposure200_v1 as exposure
from scripts.research.lightgen import execute_lightgen_g1_slm_input_mapping_pilot_v1 as pilot
from scripts.research.lightgen import execute_lightgen_g1_slm_projected_mapping_v2 as projected
from scripts.research.lightgen import execute_lightgen_g1_fixed_frame_transfer_diagnosis_v1 as transfer


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_G1_FIXED_PHASE_TEACHER_DISTILLATION_V1"
AUTHORITY = "USER_AUTHORIZED_LIGHTGEN_G1_FIXED_PHASE_TEACHER_DISTILLATION_20261030"
DEFAULT_CONFIG = ROOT / "configs/research/lightgen/g1_fixed_phase_teacher_distillation_v1.yaml"
CLASS_COUNT, PORT_COUNT, PIXEL_COUNT = 10, 100, 40_000
TRAIN_N, DEV_N, HOLDOUT_N = 2_000, 200, 2_000
UPDATES, BATCH, EXPOSURES = 15_000, 10, 75
GRIDS, METRIC_KEYS, LIMIT = (2048, 4096), exposure.METRIC_KEYS, 1.0e-4
RECOVERY_TOL = 5.0e-8


def _require(ok: bool, message: str, error: type[Exception] = RuntimeError) -> None:
    if not ok:
        raise error(message)


def _path(value: str | Path) -> Path:
    item = Path(value)
    return item if item.is_absolute() else ROOT / item


def _json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(isinstance(result, dict), f"expected JSON object at {path}")
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{uuid.uuid4().hex}")
    torch.save(value, temp)
    os.replace(temp, path)


def _config(path: Path) -> dict[str, Any]:
    _require(path.resolve() == DEFAULT_CONFIG.resolve(), "registered teacher-distillation config is required",
             ValueError)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "lightgen_g1_fixed_phase_teacher_distillation.v1",
        "task_id": TASK_ID,
        "semantic_authority_id": AUTHORITY,
        "evidence_tier": "diagnostic_only",
        "source": {
            "fixed_frame_config": "configs/research/lightgen/g1_fixed_frame_semantic_v1.yaml",
            "fixed_frame_terminal_origin": "outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/autodl_backup_attempt_01",
            "fixed_frame_terminal_backup": "outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/source_inputs_01/fixed_frame_terminal_backup",
            "physical_phase_prior_origin": "outputs/20261029_lightgen_g1_fixed_frame_semantic_v1/runs/autodl_backup_attempt_01/prior_evidence/physical_phase_gaussian_prior.pt",
            "physical_phase_prior": "outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/source_inputs_01/fixed_frame_terminal_backup/prior_evidence/physical_phase_gaussian_prior.pt",
            "v2_terminal_checkpoint": "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/terminal_checkpoint.pt",
            "response_projector": "outputs/20261027_lightgen_g1_slm_projected_mapping_v2/runs/autodl_backup_attempt_01/preflight_evidence/fixed_response_and_projector.pt",
        },
        "population": {
            "train_per_class": 200, "development_per_class": 20, "holdout_per_class": 200,
            "train_seed_base": 202610300010, "development_seed_base": 202610300030,
            "holdout_seed_base": 202610300050, "max_proposals_per_class": 1_000_000,
            "proposal_batch_size": 256, "rejection": "original_total_modal_power_only",
        },
        "teacher": {
            "phase_rad": 0.0, "initializer": 0.5, "iterations_per_target": 5000,
            "checkpoint_interval": 500, "relative_residual_stop": 1.0e-6,
            "dual_gap_stop": 1.0e-8, "command_dtype": "float32",
        },
        "training": {
            "new_updates": 15_000, "batch_size": 10, "epochs": 75,
            "exposures_per_target": 75, "optimizer": "Adam", "learning_rate": 1.0e-4,
            "betas": [0.9, 0.999], "epsilon": 1.0e-8, "weight_decay": 0.0,
            "box_lambda": 5.0, "teacher_lambda": 1.0,
            "rng_seed": 202610300070, "schedule_seed": 202610300071,
            "mapper_precision": "float32", "correction_precision": "float64",
            "optical_precision": "float64_complex128", "amp": False, "tf32": False,
            "checkpoint_interval": 500, "progress_interval": 100,
            "observation_steps": [0, 5000, 10000, 15000],
        },
        "measurement": {
            "grids": [2048, 4096], "relative_error_max": LIMIT,
            "prescribed_phase_error_max": LIMIT, "native_cmos_unit": "raw_w_per_m2",
            "camera_pixel_area_m2": 6.25e-12, "cmos_normalization": False,
            "cmos_power_matching": False, "cmos_gain_adjustment": False,
            "semantic_accuracy_drop_max_absolute": 0.02,
            "class_diversity_retention_min": 0.8,
        },
        "output": {
            "runs_root": "outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs",
            "source_dir": "outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/source_inputs_01",
            "population_dir": "outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/population_01",
            "teacher_dir": "outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/teacher_01",
            "preflight_dir": "outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/preflight_01",
            "attempt_dir": "outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/attempt_01",
            "verification_dir": "outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/windows_verify_01",
            "windows_backup_dir": "outputs/20261030_lightgen_g1_fixed_phase_teacher_distillation_v1/runs/autodl_backup_attempt_01",
        },
    }
    for key, expected_value in expected.items():
        _require(value.get(key) == expected_value,
                 f"registered fixed task contract changed: {key}", ValueError)
    return value


def _roots(config: Mapping[str, Any]) -> dict[str, Path]:
    roots = {name: _path(config["output"][name]) for name in (
        "runs_root", "source_dir", "population_dir", "teacher_dir", "preflight_dir", "attempt_dir",
        "verification_dir", "windows_backup_dir")}
    selection = roots["runs_root"] / "preflight_selection.json"
    if selection.is_file():
        payload = _json(selection)
        candidate = (roots["runs_root"] / str(payload.get("selected_path", ""))).resolve()
        _require(candidate.is_relative_to(roots["runs_root"].resolve())
                 and candidate.is_dir() and payload.get("task_id") == TASK_ID
                 and payload.get("status") == "READY",
                 "selected preflight pointer is malformed or escapes the task runs root")
        roots["preflight_dir"] = candidate
    return roots


def _next_preflight_dir(config: Mapping[str, Any]) -> Path:
    base_roots = {name: _path(config["output"][name]) for name in ("runs_root", "preflight_dir")}
    base = base_roots["preflight_dir"]
    if not base.exists():
        return base
    if (base / "preflight_summary.json").is_file():
        summary = _json(base / "preflight_summary.json")
        _require(summary.get("status") != "READY", "a successful READY preflight is write-once")
    for index in range(2, 100):
        candidate = base_roots["runs_root"] / f"preflight_repair_{index:02d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError("no unused preflight repair directory remains")


def _load_inputs(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[int, dict[str, torch.Tensor]], dict[str, Any]]:
    source_config = fixed._load_config(_path(config["source"]["fixed_frame_config"]))
    original_config = exposure._load_config(_path(source_config["source"]["g1_source_config"]))
    # The registered fixed-frame loader also opens the original MNIST image tensor to
    # revalidate the 55k fit split. This task reuses the already fitted prior and does
    # not refit it, so load only the coefficient/optics inputs needed for this run.
    inputs = exposure._load_inputs(original_config)
    root = _path(config["source"]["fixed_frame_terminal_backup"])
    checkpoint_path = root / "checkpoints" / "terminal_checkpoint.pt"
    _require(checkpoint_path.is_file(),
             f"explicitly staged fixed-frame terminal checkpoint is missing: {checkpoint_path}",
             FileNotFoundError)
    checkpoint = torch.load(checkpoint_path,
                            map_location="cpu", weights_only=False)
    prior_path = _path(config["source"]["physical_phase_prior"])
    _require(prior_path.is_file() and prior_path.resolve().is_relative_to(root.resolve()),
             "the explicitly staged physical-phase prior is missing or outside its source snapshot",
             FileNotFoundError)
    prior_payload = torch.load(prior_path, map_location="cpu", weights_only=False)
    _require(checkpoint.get("task_id") == fixed.TASK_ID
             and checkpoint.get("completed_updates") == 15000
             and checkpoint.get("source_mapper_updates") == 10000
             and checkpoint.get("lineage_mapper_updates") == 25000
             and float(checkpoint.get("selected_alpha", -1.0)) == 1.0
             and prior_payload.get("task_id") == fixed.TASK_ID
             and prior_payload.get("ridge_rule") == "1e-6 times mean sample covariance diagonal"
             and prior_payload.get("population_count") == 55_000
             and torch.equal(prior_payload["scale"], inputs["scale"]),
             "fixed-frame terminal or physical-phase prior lineage is invalid")
    prior = prior_payload["fit"]
    _require(set(prior) == set(range(CLASS_COUNT)), "physical-phase G1 prior lacks a class")
    source = _path(config["source"]["v2_terminal_checkpoint"])
    _require(inputs["source_mapper_path"].resolve() == source.resolve()
             and inputs["source_mapper"].get("completed_updates") == 5000
             and fixed._adam_steps(inputs["source_mapper"].get("optimizer_state", {})) == [5000] * 4,
             "V2 starting mapper or optimizer lineage is invalid")
    projector = torch.load(_path(config["source"]["response_projector"]),
                           map_location="cpu", weights_only=False)
    _require(torch.equal(projector["response_matrix"], inputs["response_matrix"])
             and torch.equal(projector["full_right_inverse"], inputs["inverse"])
             and torch.equal(projector["scale"], inputs["scale"])
             and projector["real_imag_order"] == "per_port_interleaved_real_imag",
             "explicit fixed response, right inverse or modal scale changed")
    return inputs, prior, checkpoint


def _sample(config: Mapping[str, Any], prior: Mapping[int, Mapping[str, torch.Tensor]],
            inputs: Mapping[str, Any], *, role: str, per_class: int,
            seed_base: int) -> tuple[dict[str, Any], dict[str, Any]]:
    cohort, accounting = fixed._cohort(
        prior, inputs["scale"], alpha=1.0, role=role, count=per_class,
        seed_base=seed_base, config=config)
    coeff = cohort["coefficients"].to(torch.complex128)
    labels = torch.arange(CLASS_COUNT).repeat_interleave(per_class)
    proposal_identity = torch.stack((cohort["labels"], cohort["proposal_ids"]), dim=1)
    _require(torch.equal(cohort["labels"], labels)
             and bool(torch.all(coeff.abs().square().sum(1) <= pilot.base.MAX_MODAL_POWER_W))
             and torch.unique(proposal_identity, dim=0).shape[0] == CLASS_COUNT * per_class,
             f"{role} sample identities or power rejection are invalid")
    cohort.update({"role": role, "source_indices": torch.arange(CLASS_COUNT * per_class),
                   "coefficients": coeff})
    return cohort, accounting


def _schedule(config: Mapping[str, Any]) -> torch.Tensor:
    rng = torch.Generator(device="cpu").manual_seed(int(config["training"]["schedule_seed"]))
    classes = [torch.arange(y * 200, (y + 1) * 200) for y in range(CLASS_COUNT)]
    rows = []
    for _ in range(EXPOSURES):
        order = [values[torch.randperm(200, generator=rng)] for values in classes]
        rows.extend(torch.stack([order[y][k] for y in range(CLASS_COUNT)]) for k in range(200))
    result = torch.stack(rows).long()
    _require(tuple(result.shape) == (UPDATES, BATCH)
             and torch.equal(torch.bincount(result.flatten(), minlength=TRAIN_N),
                             torch.full((TRAIN_N,), EXPOSURES)),
             "training schedule does not expose all 2,000 targets exactly 75 times")
    return result


def _prepare(config: Mapping[str, Any], *, resume: Path | None) -> dict[str, Any]:
    roots = _roots(config)
    inputs, prior, _ = _load_inputs(config)
    if resume is None:
        for key in ("population_dir", "teacher_dir"):
            _require(not roots[key].exists(), f"output already exists: {roots[key]}", FileExistsError)
        roots["population_dir"].mkdir(parents=True)
        cohorts, accounting = {}, {}
        settings = (("train", 200, "fixed_train2000", "train_seed_base"),
                    ("development", 20, "fixed_development200", "development_seed_base"),
                    ("holdout", 200, "new_final_holdout2000", "holdout_seed_base"))
        for name, n, role, seed_key in settings:
            cohorts[name], accounting[name] = _sample(
                config, prior, inputs, role=role, per_class=n,
                seed_base=int(config["population"][seed_key]))
        overlaps = {}
        for left, right in (("train", "development"), ("train", "holdout"), ("development", "holdout")):
            left_rows = {row.contiguous().numpy().tobytes()
                         for row in cohorts[left]["coefficients"].to(torch.complex128)}
            right_rows = {row.contiguous().numpy().tobytes()
                          for row in cohorts[right]["coefficients"].to(torch.complex128)}
            overlaps[f"{left}_{right}"] = bool(left_rows.intersection(right_rows))
            _require(not overlaps[f"{left}_{right}"], f"{left} and {right} contain an identical target")
        payload = {"schema": f"{TASK_ID}.population.v1", "task_id": TASK_ID,
                   "cohorts": cohorts, "sampling_accounting": accounting,
                   "exact_target_overlap": overlaps,
                   "holdout_not_measured_during_teacher_or_training": True}
        _save(roots["population_dir"] / "population.pt", payload)
        _write_json(roots["population_dir"] / "sampling_summary.json", {
            "task_id": TASK_ID, "status": "POPULATION_FIXED", "counts": {k: len(v["labels"]) for k, v in cohorts.items()},
            "sampling_accounting": accounting, "exact_target_overlap": overlaps})
    else:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        _require(payload.get("task_id") == TASK_ID and payload.get("schema") == f"{TASK_ID}.population.v1",
                 "explicit population resume file has a different identity")
        cohorts = payload["cohorts"]
    _require(tuple(cohorts["train"]["coefficients"].shape) == (TRAIN_N, PORT_COUNT)
             and tuple(cohorts["development"]["coefficients"].shape) == (DEV_N, PORT_COUNT)
             and tuple(cohorts["holdout"]["coefficients"].shape) == (HOLDOUT_N, PORT_COUNT),
             "fixed populations have an invalid shape")
    return {"status": "POPULATION_FIXED", "population_dir": str(roots["population_dir"]),
            "counts": {name: int(len(cohort["labels"])) for name, cohort in cohorts.items()},
            "scientific_updates": 0}


def _load_population(config: Mapping[str, Any]) -> dict[str, Any]:
    path = _roots(config)["population_dir"] / "population.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    _require(data.get("schema") == f"{TASK_ID}.population.v1" and data.get("task_id") == TASK_ID,
             "fixed task populations are missing or invalid", FileNotFoundError)
    return data


def _measure_command(inputs: Mapping[str, Any], target: torch.Tensor, command: torch.Tensor,
                     models: Mapping[int, Any]) -> tuple[dict[str, float], dict[int, dict[str, torch.Tensor]]]:
    outputs, references = {}, {}
    device = next(models[2048].parameters()).device
    target = target.reshape(1, PORT_COUNT).to(device=device, dtype=torch.complex128)
    command = command.reshape(1, 1, 200, 200).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        for grid in GRIDS:
            outputs[grid] = projected._forward_amplitude(models[grid], command)
            references[grid] = projected._forward_target(models[grid], target)["raw_cmos"]
    metrics = projected._error_row(outputs, target, references, 0)
    result = {grid: {**outputs[grid], "direct_raw_cmos": references[grid]} for grid in GRIDS}
    return {key: float(value) for key, value in metrics.items()}, result


def _teacher(config: Mapping[str, Any], *, resume: Path | None) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "teacher qualification requires AutoDL CUDA")
    fixed._set_numerics()
    roots = _roots(config)
    population = _load_population(config)["cohorts"]
    inputs, _, _ = _load_inputs(config)
    _require(not (roots["teacher_dir"] / "teacher_summary.json").is_file(),
             "teacher qualification has a terminal result and cannot be reopened")
    _require(not roots["teacher_dir"].exists() or resume is not None,
             "teacher attempt exists; resume requires its explicit checkpoint", FileExistsError)
    if resume is not None:
        _require(resume.is_file() and resume.resolve().is_relative_to(roots["teacher_dir"].resolve()),
                 "teacher resume must name a saved checkpoint inside this teacher attempt", ValueError)
    roots["teacher_dir"].mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    response = inputs["response_matrix"].to(device=device, dtype=torch.complex128)
    spectral_lipschitz = float(exposure.feasibility._spectral_lipschitz(response))
    _require(math.isfinite(spectral_lipschitz) and spectral_lipschitz > 0.0,
             "fixed teacher solver requires a finite cached spectral constant")
    models = {g: exposure._make_runtime_model(inputs, g, device) for g in GRIDS}
    row_start, active = 0, None
    if resume is not None:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        _require(state.get("task_id") == TASK_ID and state.get("schema") == f"{TASK_ID}.teacher_cursor.v1",
                 "teacher recovery checkpoint identity is invalid")
        row_start, active = int(state["next_row"]), state.get("solver_state")
        _require(0 <= row_start <= TRAIN_N and int(state.get("completed_rows", -1)) == row_start
                 and isinstance(state.get("rng_state"), dict)
                 and float(state.get("spectral_lipschitz", -1.0)) == spectral_lipschitz,
                 "teacher recovery cursor is inconsistent")
        torch.set_rng_state(state["rng_state"]["cpu"])
        if torch.cuda.is_available() and state["rng_state"].get("cuda"):
            torch.cuda.set_rng_state_all(state["rng_state"]["cuda"])
    records = []
    for idx in range(row_start):
        prior_row = torch.load(roots["teacher_dir"] / f"row_{idx:04d}.pt", map_location="cpu", weights_only=False)
        _require(prior_row.get("task_id") == TASK_ID and prior_row.get("index") == idx
                 and bool(prior_row.get("qualified")), "teacher checkpoint prefix is incomplete")
        records.append(prior_row["summary"])
    failed = None
    # A crash can occur after an atomic row write and before the cursor write. Consume
    # that committed row instead of repeating its already completed solver budget.
    while row_start < TRAIN_N:
        committed_path = roots["teacher_dir"] / f"row_{row_start:04d}.pt"
        if not committed_path.is_file():
            break
        committed = torch.load(committed_path, map_location="cpu", weights_only=False)
        expected = population["train"]["coefficients"][row_start]
        _require(committed.get("task_id") == TASK_ID and committed.get("index") == row_start
                 and torch.equal(committed.get("target_coefficients"), expected),
                 f"committed teacher row {row_start} is not bound to the fixed training target")
        records.append(committed["summary"])
        row_start += 1
        active = None
        if not committed.get("qualified"):
            failed = committed["summary"]
            break
        _save(roots["teacher_dir"] / "active_solver.pt", {
            "schema": f"{TASK_ID}.teacher_cursor.v1", "task_id": TASK_ID,
            "next_row": row_start, "solver_state": None,
            "completed_rows": row_start, "spectral_lipschitz": spectral_lipschitz,
            "rng_state": _rng()})
    for index in range(row_start, TRAIN_N) if failed is None else ():
        target_cpu = population["train"]["coefficients"][index].to(torch.complex128)
        target = target_cpu.to(device=device)
        init = torch.full((PIXEL_COUNT,), 0.5, dtype=torch.float64, device=device)
        prior_iteration_upper_bound = 0
        iteration_limit = int(config["teacher"]["iterations_per_target"])
        if index == 0:
            failure_path = (roots["runs_root"] / "engineering_repair_01"
                            / "teacher_invalid_harness_01" / "engineering_failure.json")
            if failure_path.is_file():
                failure = _json(failure_path)
                _require(failure.get("task_id") == TASK_ID
                         and failure.get("classification") == "INVALID_HARNESS"
                         and failure.get("failed_target_index") == 0
                         and failure.get("resume_cursor_saved") is False
                         and failure.get("solver_returned_before_failure") is True
                         and failure.get("checkpoint_interval") == 500,
                         "uncommitted teacher retry evidence is inconsistent")
                prior_iteration_upper_bound = int(
                    failure.get("uncommitted_solver_iteration_upper_bound", -1))
                _require(0 <= prior_iteration_upper_bound < 500,
                         "uncommitted teacher iteration bound is invalid")
                iteration_limit -= prior_iteration_upper_bound
        _require(0 < iteration_limit <= int(config["teacher"]["iterations_per_target"]),
                 "remaining per-target teacher iteration budget is invalid")
        initial = active if index == row_start and active is not None else fixed._initial_fixed_phase_state(
            response, target, init)
        def save_cursor(state: Mapping[str, Any]) -> None:
            _save(roots["teacher_dir"] / "active_solver.pt", {
                "schema": f"{TASK_ID}.teacher_cursor.v1", "task_id": TASK_ID,
                "next_row": index, "solver_state": copy.deepcopy(dict(state)),
                "completed_rows": index, "spectral_lipschitz": spectral_lipschitz,
                "rng_state": _rng()})
        solved = exposure._fista_fixed_phase_solve(
            response, target, 0.0, max_steps=iteration_limit, lipschitz=spectral_lipschitz,
            initial_state=initial,
            checkpoint=save_cursor, checkpoint_interval=500)
        command = solved["amplitude"].to(torch.float32).reshape(1, 1, 200, 200)
        metrics, outputs = _measure_command(inputs, target_cpu, command, models)
        actual = outputs[2048]["coefficients"].reshape(-1).to(torch.complex128)
        prescribed = target_cpu.reshape(-1)
        phase_error = float(torch.linalg.vector_norm(actual.cpu() - prescribed)
                            / torch.linalg.vector_norm(prescribed).clamp_min(1.0e-30))
        qualified = all(metrics[k] <= LIMIT for k in METRIC_KEYS) and phase_error <= LIMIT
        summary = {"index": index, "label": int(population["train"]["labels"][index]),
                   "proposal_id": int(population["train"]["proposal_ids"][index]),
                   "solver_iterations": int(solved["iterations"]),
                   "solver_iteration_limit": iteration_limit,
                   "prior_uncommitted_solver_iteration_upper_bound": prior_iteration_upper_bound,
                   "cumulative_solver_iteration_upper_bound": (
                       prior_iteration_upper_bound + int(solved["iterations"])),
                   "solver_relative_error": float(solved["relative_error"]),
                   "prescribed_phase_coefficient_error": phase_error,
                   "metrics": metrics, "qualified": qualified}
        _save(roots["teacher_dir"] / f"row_{index:04d}.pt", {
            "schema": f"{TASK_ID}.teacher_row.v1", "task_id": TASK_ID, **summary,
            "target_coefficients": target_cpu.clone(), "amplitude_command": command.cpu().clone(),
            "outputs": {g: {k: v.detach().cpu().clone() for k, v in values.items()}
                        for g, values in outputs.items()}})
        records.append(summary)
        _save(roots["teacher_dir"] / "active_solver.pt", {
            "schema": f"{TASK_ID}.teacher_cursor.v1", "task_id": TASK_ID,
            "next_row": index + 1, "solver_state": None,
            "completed_rows": index + 1, "spectral_lipschitz": spectral_lipschitz,
            "rng_state": _rng()})
        _write_json(roots["teacher_dir"] / "progress.json", {
            "task_id": TASK_ID, "completed_rows": index + 1, "total_rows": TRAIN_N,
            "completed_solver_iterations": sum(r["solver_iterations"] for r in records),
            "cumulative_solver_iteration_upper_bound": sum(
                r["cumulative_solver_iteration_upper_bound"] for r in records),
            "last_row_qualified": qualified})
        active = None
        if not qualified:
            failed = summary
            break
    passed = sum(bool(r["qualified"]) for r in records)
    status = "TEACHER_QUALIFIED" if passed == TRAIN_N else "TRAINING_REFERENCE_INSUFFICIENT"
    _write_json(roots["teacher_dir"] / "teacher_summary.json", {
        "task_id": TASK_ID, "status": status, "teacher_rows_completed": len(records),
        "teacher_rows_passed": passed, "failed_row": failed,
        "iteration_total": sum(r["solver_iterations"] for r in records),
        "cumulative_solver_iteration_upper_bound": sum(
            r["cumulative_solver_iteration_upper_bound"] for r in records),
        "train_ready": False, "scientific_updates": 0})
    return {"task_id": TASK_ID, "status": status, "teacher_rows_passed": passed,
            "teacher_rows_completed": len(records), "failed_row": failed}


def _load_teachers(config: Mapping[str, Any]) -> torch.Tensor:
    root = _roots(config)["teacher_dir"]
    summary = _json(root / "teacher_summary.json")
    _require(summary.get("task_id") == TASK_ID and summary.get("status") == "TEACHER_QUALIFIED"
             and summary.get("teacher_rows_passed") == TRAIN_N, "all 2,000 qualified teachers are required")
    commands, cohorts = [], _load_population(config)["cohorts"]["train"]
    for index in range(TRAIN_N):
        row = torch.load(root / f"row_{index:04d}.pt", map_location="cpu", weights_only=False)
        _require(row.get("task_id") == TASK_ID and row.get("index") == index and row.get("qualified") is True
                 and 0 < int(row.get("solver_iterations", 0))
                 <= int(row.get("solver_iteration_limit", 0)) <= 5000
                 and 0 <= int(row.get("prior_uncommitted_solver_iteration_upper_bound", -1)) < 500
                 and int(row.get("cumulative_solver_iteration_upper_bound", -1))
                 == (int(row["prior_uncommitted_solver_iteration_upper_bound"])
                     + int(row["solver_iterations"]))
                 and int(row["cumulative_solver_iteration_upper_bound"]) <= 5000
                 and int(row["label"]) == int(cohorts["labels"][index])
                 and int(row["proposal_id"]) == int(cohorts["proposal_ids"][index])
                 and torch.equal(row["target_coefficients"], cohorts["coefficients"][index])
                 and all(float(row["metrics"][k]) <= LIMIT for k in METRIC_KEYS)
                 and float(row["prescribed_phase_coefficient_error"]) <= LIMIT,
                 f"teacher row {index} is not bound to its measured target")
        command = row["amplitude_command"]
        _require(tuple(command.shape) == (1, 1, 200, 200) and command.dtype == torch.float32
                 and torch.isfinite(command).all().item()
                 and float(command.min()) >= 0.0 and float(command.max()) <= 1.0
                 and row.get("outputs", {}).keys() == set(GRIDS),
                 f"teacher command or saved production outputs are malformed at row {index}")
        commands.append(command.reshape(1, PIXEL_COUNT))
    return torch.cat(commands).to(torch.float32)


def _features(target: torch.Tensor, scale: torch.Tensor, device: torch.device) -> torch.Tensor:
    return projected._normalized_features(target.detach().cpu(), scale, device)


def _mapper(inputs: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    return projected._new_mapper(device, inputs["source_mapper"]["mapper_state"])


def _optimizer(mapper: torch.nn.Module, config: Mapping[str, Any]) -> torch.optim.Adam:
    t = config["training"]
    return torch.optim.Adam(mapper.parameters(), lr=t["learning_rate"], betas=tuple(t["betas"]),
                            eps=t["epsilon"], weight_decay=t["weight_decay"])


def _fixed_map(mapper: torch.nn.Module, target: torch.Tensor, inputs: Mapping[str, Any],
               device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    target = target.reshape(-1, PORT_COUNT).to(device=device, dtype=torch.complex128)
    features = _features(target, inputs["scale"], device)
    return transfer._fixed_phase_projection(
        mapper(features), target, inputs["matrix"].to(device), inputs["inverse"].to(device),
        inputs["scale"].to(device), 0.0)


def _update(mapper: torch.nn.Module, optimizer: torch.optim.Optimizer,
            target_cpu: torch.Tensor, teacher_cpu: torch.Tensor,
            inputs: Mapping[str, Any], model: Any, device: torch.device,
            engine: str, config: Mapping[str, Any], *, audit: bool = False
            ) -> dict[str, Any]:
    target = target_cpu.to(device=device, dtype=torch.complex128)
    teachers = teacher_cpu.to(device=device, dtype=torch.float32).reshape(-1, 1, 200, 200)
    before = [p.detach().clone() for p in mapper.parameters()] if audit else None
    optimizer.zero_grad(set_to_none=True)
    command, detail = _fixed_map(mapper, target, inputs, device)
    if engine == "full_optics":
        actual = projected._forward_amplitude(model, command)["coefficients"]
    elif engine == "response_matrix":
        actual = exposure._forward_matrix(command, inputs["response_matrix"].to(device))
    else:
        raise ValueError(f"unsupported fixed engine: {engine}")
    per_row = pilot._phase_invariant_loss(actual.to(torch.complex128), target)
    lc = per_row.mean()
    preclip = detail["preclip_amplitude"].reshape(len(target), -1).double()
    box = (torch.relu(-preclip).square() + torch.relu(preclip - 1.0).square()).mean()
    teacher_loss = (preclip - teachers.reshape(len(target), -1).double()).square().mean()
    total = (lc + float(config["training"]["box_lambda"]) * box
             + float(config["training"]["teacher_lambda"]) * teacher_loss)
    _require(torch.isfinite(total).item() and torch.isfinite(lc).item()
             and torch.isfinite(box).item() and torch.isfinite(teacher_loss).item(),
             "training produced a nonfinite objective")
    total.backward()
    norms = {}
    for name, parameter in mapper.named_parameters():
        _require(parameter.grad is not None and torch.isfinite(parameter.grad).all().item(),
                 f"mapper gradient missing or nonfinite: {name}")
        norms[name] = float(torch.linalg.vector_norm(parameter.grad.detach().double()))
    optimizer.step()
    if before is not None:
        _require(all(not torch.equal(old, p.detach()) for old, p in zip(before, mapper.parameters(), strict=True)),
                 "a mapper parameter tensor did not update")
    return {"loss": float(total.detach()), "coefficient_loss": float(lc.detach()),
            "box_loss": float(box.detach()), "teacher_loss": float(teacher_loss.detach()),
            "coefficient_relative_error_mean": float(torch.sqrt(per_row.detach()).mean()),
            "coefficient_relative_error_rms": float(torch.sqrt(per_row.detach().mean())),
            "preclip_out_of_bounds_fraction": float(((preclip < 0) | (preclip > 1)).double().mean()),
            "command_boundary_fraction": float(((command <= 0) | (command >= 1)).double().mean()),
            "gradient_norms": norms}


def _rng() -> dict[str, Any]:
    return {"cpu": torch.get_rng_state().clone(),
            "cuda": [x.cpu().clone() for x in torch.cuda.get_rng_state_all()]}


def _identity(config: Mapping[str, Any], run_id: str) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id, trajectory_id=f"{run_id}:fixed_phase_teacher_distillation",
        semantic_authority_id=AUTHORITY, data_role="RAW_PHASE_G1_TRAIN2000_DEV200_HOLDOUT2000",
        config_identity=dict(config),
        model_identity={"source_mapper": config["source"]["v2_terminal_checkpoint"],
                        "prior": "fixed physical-phase alpha=1.0", "optics": "frozen C45k",
                        "objective": "Lc+5Lbox+Lteacher; phase=0"},
        optimizer_identity={"kind": "Adam", "learning_rate": 1.0e-4,
                            "new_updates": UPDATES, "inherited_state": False},
        sampler_identity={"train_rows": TRAIN_N, "development_rows": DEV_N,
                          "holdout_rows": HOLDOUT_N, "exposures_per_target": EXPOSURES,
                          "schedule_seed": int(config["training"]["schedule_seed"]),
                          "rng_seed": int(config["training"]["rng_seed"])})


def _save_training_state(path: Path, mapper: torch.nn.Module, optimizer: torch.optim.Optimizer,
                         history: Sequence[Mapping[str, Any]], counts: torch.Tensor,
                         schedule: torch.Tensor, run_id: str, completed: int) -> dict[str, Any]:
    before = _rng()
    payload = {"schema": f"{TASK_ID}.checkpoint.v1", "task_id": TASK_ID, "run_id": run_id,
               "completed_updates": completed, "schedule_cursor": completed,
               "next_batch": schedule[completed].clone() if completed < UPDATES else None,
               "schedule": schedule.clone(), "history": copy.deepcopy(list(history)),
               "counts": counts.clone(), "history_count": len(history),
               "mapper_state": {k: v.detach().cpu().clone() for k, v in mapper.state_dict().items()},
               "optimizer_state": copy.deepcopy(optimizer.state_dict()), "rng_state": before,
               "optimizer_deepcopied": True, "adam_step": completed,
               "lineage_mapper_updates": 25_000}
    _save(path, payload)
    _require(torch.equal(before["cpu"], torch.get_rng_state()) and all(
        torch.equal(a, b) for a, b in zip(before["cuda"], torch.cuda.get_rng_state_all(), strict=True)),
        "checkpoint serialization changed RNG")
    reopened = torch.load(path, map_location="cpu", weights_only=False)
    _require(reopened["task_id"] == TASK_ID and reopened["completed_updates"] == completed
             and reopened["history_count"] == completed and reopened["optimizer_deepcopied"] is True,
             "checkpoint readback failed")
    return reopened


def _verify_teacher_outputs(config: Mapping[str, Any], inputs: Mapping[str, Any],
                           population: Mapping[str, Any], teachers: torch.Tensor,
                           models: Mapping[int, Any], device: torch.device) -> dict[str, Any]:
    root = _roots(config)["teacher_dir"]
    compared_rows = 0
    max_output_delta = 0.0
    max_metric_delta = 0.0
    max_phase_error = 0.0
    for start in range(0, TRAIN_N, BATCH):
        stop = start + BATCH
        target = population["train"]["coefficients"][start:stop].to(
            device=device, dtype=torch.complex128)
        commands = teachers[start:stop].reshape(BATCH, 1, 200, 200).to(
            device=device, dtype=torch.float32)
        actual, references = {}, {}
        with torch.no_grad():
            for grid in GRIDS:
                actual[grid] = projected._forward_amplitude(models[grid], commands)
                references[grid] = projected._forward_target(models[grid], target)["raw_cmos"]
        for local, index in enumerate(range(start, stop)):
            row = torch.load(root / f"row_{index:04d}.pt", map_location="cpu", weights_only=False)
            expected_metrics = {key: float(value) for key, value in row["metrics"].items()}
            measured_metrics = projected._error_row(
                {grid: {key: value[local:local + 1] for key, value in actual[grid].items()}
                 for grid in GRIDS}, target[local:local + 1],
                {grid: references[grid][local:local + 1] for grid in GRIDS}, 0)
            for grid in GRIDS:
                saved = row["outputs"][grid]
                for key in ("coefficients", "raw_cmos", "direct_raw_cmos"):
                    new_value = (references[grid][local:local + 1] if key == "direct_raw_cmos"
                                 else actual[grid][key][local:local + 1])
                    old_value = saved[key].to(device=device)
                    delta = float((new_value.double() - old_value.double()).abs().max())
                    max_output_delta = max(max_output_delta, delta)
                    _require(delta <= 1.0e-14,
                             f"teacher row {index} saved {grid} {key} differs from production replay")
            phase_error = float(torch.linalg.vector_norm(
                actual[2048]["coefficients"][local] - target[local]) /
                torch.linalg.vector_norm(target[local]).clamp_min(1.0e-30))
            max_phase_error = max(max_phase_error, phase_error)
            _require(phase_error <= LIMIT, f"teacher row {index} failed fixed phase-zero replay")
            for key in METRIC_KEYS:
                delta = abs(float(measured_metrics[key]) - expected_metrics[key])
                max_metric_delta = max(max_metric_delta, delta)
                _require(delta <= 1.0e-14 and measured_metrics[key] <= LIMIT,
                         f"teacher row {index} metric {key} differs from its saved qualification")
            compared_rows += 1
    result = {"task_id": TASK_ID, "status": "ALL_TEACHER_OUTPUTS_REPLAYED",
              "rows": compared_rows, "max_output_absolute_delta": max_output_delta,
              "max_metric_absolute_delta": max_metric_delta,
              "max_prescribed_phase_error": max_phase_error,
              "source": "explicit saved float32 command and target replayed through both frozen grids"}
    _write_json(root / "production_replay_summary.json", result)
    return result


def _preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA preflight requires AutoDL CUDA")
    fixed._set_numerics()
    roots = _roots(config)
    selection_path = roots["runs_root"] / "preflight_selection.json"
    _require(not selection_path.exists(), "a selected READY preflight is write-once", FileExistsError)
    current_preflight = roots["preflight_dir"]
    if (current_preflight / "preflight_summary.json").is_file():
        _require(_json(current_preflight / "preflight_summary.json").get("status") != "READY",
                 "a successful READY preflight is write-once", FileExistsError)
    roots["preflight_dir"] = _next_preflight_dir(config)
    _require(not roots["preflight_dir"].exists(), "preflight output already exists", FileExistsError)
    inputs, _, _ = _load_inputs(config)
    population_data = _load_population(config)
    population = population_data["cohorts"]
    teachers = _load_teachers(config)
    schedule = _schedule(config)
    device = torch.device("cuda:0")
    camera_area = fixed._camera_area(inputs, device)
    _require(abs(camera_area - 6.25e-12) <= 1.0e-22, "frozen camera pixel area changed")
    # Independently replay one teacher per class through the production optical consumer.
    models = {g: exposure._make_runtime_model(inputs, g, device) for g in GRIDS}
    frozen_optics = {g: {k: v.detach().cpu().clone() for k, v in models[g].state_dict().items()}
                     for g in GRIDS}
    _require(all(not parameter.requires_grad for model in models.values()
                 for parameter in model.parameters()),
             "frozen production optics expose trainable parameters during mapper training")
    teacher_replay = _verify_teacher_outputs(config, inputs, population, teachers, models, device)
    positive = []
    for idx in range(0, TRAIN_N, 200):
        metrics, outputs = _measure_command(inputs, population["train"]["coefficients"][idx],
                                             teachers[idx], models)
        _require(all(metrics[k] <= LIMIT for k in METRIC_KEYS),
                 f"teacher positive control failed at train row {idx}")
        positive.append({"index": idx, "metrics": metrics})
    wrong_metrics, _ = _measure_command(inputs, population["train"]["coefficients"][0],
                                         teachers[200], models)
    wrong_intensity, _ = _measure_command(inputs, population["train"]["coefficients"][0],
                                           teachers[0].square(), models)
    _require(any(wrong_metrics[k] > LIMIT for k in METRIC_KEYS)
             and any(wrong_intensity[k] > LIMIT for k in METRIC_KEYS),
             "production consumer accepted a mismatched teacher or intensity encoded as amplitude")

    outcomes: dict[str, Any] = {}
    for engine in ("full_optics", "response_matrix"):
        mapper = _mapper(inputs, device)
        optimizer = _optimizer(mapper, config)
        initial = {k: v.detach().clone() for k, v in mapper.state_dict().items()}
        history = []
        state3 = None
        torch.manual_seed(int(config["training"]["rng_seed"]))
        torch.cuda.manual_seed_all(int(config["training"]["rng_seed"]))
        for step in range(4):
            ix = schedule[step]
            record = _update(mapper, optimizer, population["train"]["coefficients"][ix],
                             teachers[ix], inputs, models[2048], device, engine, config, audit=step == 0)
            history.append({"step": step + 1, "indices": ix.tolist(), **record})
            if step == 2:
                state3 = {"schema": f"{TASK_ID}.recovery.v1", "task_id": TASK_ID,
                          "completed_updates": 3, "schedule_cursor": 3,
                          "mapper_state": {k: v.detach().cpu().clone()
                                           for k, v in mapper.state_dict().items()},
                          "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                          "history": copy.deepcopy(history), "rng_state": _rng(),
                          "schedule": schedule.clone(), "next_batch": schedule[3].clone(),
                          "engine": engine}
        outcomes[engine] = {"mapper_state": {k: v.detach().cpu().clone()
                                                for k, v in mapper.state_dict().items()},
                            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
                            "loss": history[-1]["loss"], "history": history,
                            "batch4": schedule[3].clone(),
                            "initial_state": initial, "state3": state3, "rng_state": _rng()}
    for grid in GRIDS:
        projected._assert_frozen(models[grid], frozen_optics[grid], f"optics_{grid}")
    model_delta = fixed._compare_states(outcomes["full_optics"]["mapper_state"],
                                        outcomes["response_matrix"]["mapper_state"])
    _, optimizer_delta = fixed.fit20._tree_max_delta(outcomes["full_optics"]["optimizer_state"],
                                                      outcomes["response_matrix"]["optimizer_state"])
    loss_delta = abs(outcomes["full_optics"]["loss"] - outcomes["response_matrix"]["loss"])
    matrix_ok = max(model_delta, optimizer_delta, loss_delta) <= RECOVERY_TOL
    engine = "response_matrix" if matrix_ok else "full_optics"
    roots["preflight_dir"].mkdir(parents=True)
    # Reuse the already executed three-step prefix and direct fourth step. The independent
    # process adds the ninth engineering Adam call without duplicating an update.
    selected = outcomes[engine]
    state3 = selected["state3"]
    third = roots["preflight_dir"] / "recovery_step_00003.pt"
    _save(third, state3)
    direct4 = {"task_id": TASK_ID, "completed_updates": 4, "indices": schedule[3].clone(),
               "mapper_state": selected["mapper_state"],
               "optimizer_state": selected["optimizer_state"],
               "loss": selected["loss"], "rng_state": selected["rng_state"],
               "history": selected["history"], "engine": engine}
    direct_path = roots["preflight_dir"] / "recovery_direct_step_00004.pt"
    _save(direct_path, direct4)
    replay_path = roots["preflight_dir"] / "recovery_replayed_step_00004.pt"
    child = subprocess.run([sys.executable, "-m", __name__, "--mode", "recovery-worker",
                           "--config", str(DEFAULT_CONFIG), "--checkpoint", str(third),
                           "--evidence-dir", str(roots["preflight_dir"]),
                           "--output", str(replay_path)], cwd=ROOT, capture_output=True,
                          text=True, check=False)
    _require(child.returncode == 0 and replay_path.is_file(),
             "independent recovery worker failed: " + child.stderr[-3000:])
    replay4 = torch.load(replay_path, map_location="cpu", weights_only=False)
    model_recovery = fixed._compare_states(direct4["mapper_state"], replay4["mapper_state"])
    _, adam_recovery = fixed.fit20._tree_max_delta(direct4["optimizer_state"], replay4["optimizer_state"])
    cuda_rng_equal = len(direct4["rng_state"]["cuda"]) == len(replay4["rng_state"]["cuda"]) and all(
        torch.equal(a, b) for a, b in zip(direct4["rng_state"]["cuda"],
                                          replay4["rng_state"]["cuda"], strict=True))
    _require(model_recovery <= RECOVERY_TOL and adam_recovery <= RECOVERY_TOL
             and abs(direct4["loss"] - replay4["loss"]) <= RECOVERY_TOL
             and torch.equal(direct4["indices"], replay4["indices"])
             and torch.equal(direct4["rng_state"]["cpu"], replay4["rng_state"]["cpu"])
             and cuda_rng_equal and direct4["history"] == replay4["history"]
             and torch.equal(state3["schedule"], schedule),
             "independent fourth-step recovery differs from in-process result")

    run_id = uuid.uuid4().hex
    identity = ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id, trajectory_id=f"{run_id}:teacher_distillation",
        semantic_authority_id=AUTHORITY, data_role="TRAIN2000_DEV200_HOLDOUT2000_G1_ALPHA1",
        config_identity=dict(config), model_identity={"source": config["source"]["v2_terminal_checkpoint"],
            "prior": "physical phase alpha=1", "teacher_phase": 0.0},
        optimizer_identity={"kind": "Adam", "learning_rate": 1e-4, "updates": UPDATES,
                            "teacher_lambda": 1.0, "box_lambda": 5.0},
        sampler_identity={"train": TRAIN_N, "dev": DEV_N, "holdout": HOLDOUT_N,
                          "exposures": EXPOSURES, "schedule_seed": config["training"]["schedule_seed"]})
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "status": "PASS", "teachers": TRAIN_N,
                    "positive_controls": positive, "wrong_teacher_rejected": True,
                    "wrong_intensity_encoding_rejected": True, "camera_area_m2": camera_area,
                    "teacher_replay_rows": teacher_replay["rows"],
                    "teacher_replay_max_output_delta": teacher_replay["max_output_absolute_delta"]},
        resource={"passed": True, "status": "PASS", "device": torch.cuda.get_device_name(device),
                  "free_memory_observation_bytes": int(torch.cuda.mem_get_info(device)[0]),
                  "fixed_quota": False},
        implementation={"passed": True, "status": "PASS", "selected_engine": engine,
                        "matrix_four_step_equivalence": matrix_ok, "model_delta": model_delta,
                        "optimizer_delta": optimizer_delta, "loss_delta": loss_delta,
                        "recovery_model_delta": model_recovery, "recovery_optimizer_delta": adam_recovery,
                        "engineering_adam_calls": 9}, selected_engine=engine)
    control.mark_preflight_ready(receipt)
    _save(roots["preflight_dir"] / "population.pt", population_data)
    _save(roots["preflight_dir"] / "teachers.pt", teachers.cpu())
    _save(roots["preflight_dir"] / "schedule.pt", schedule)
    _write_json(roots["preflight_dir"] / "identity.json", identity.to_dict())
    _write_json(roots["preflight_dir"] / "ready_receipt.json", receipt.to_dict())
    _write_json(roots["preflight_dir"] / "preflight_summary.json", {
        "task_id": TASK_ID, "run_id": run_id, "status": "READY", "engine": engine,
        "matrix_four_step_equivalence": matrix_ok, "model_delta": model_delta,
        "optimizer_delta": optimizer_delta, "loss_delta": loss_delta,
        "recovery_model_delta": model_recovery, "recovery_optimizer_delta": adam_recovery,
        "recovery_loss_delta": abs(direct4["loss"] - replay4["loss"]),
        "engineering_adam_calls": 9, "scientific_updates": 0,
        "wrong_teacher_rejected": True, "wrong_intensity_encoding_rejected": True,
        "teacher_output_replay_rows": teacher_replay["rows"],
        "teacher_output_replay_max_abs_delta": teacher_replay["max_output_absolute_delta"],
        "recovery_cpu_rng_equal": torch.equal(direct4["rng_state"]["cpu"], replay4["rng_state"]["cpu"]),
        "recovery_cuda_rng_equal": cuda_rng_equal,
        "recovery_history_equal": direct4["history"] == replay4["history"]})
    _write_json(roots["runs_root"] / "preflight_selection.json", {
        "task_id": TASK_ID, "selected_path": roots["preflight_dir"].relative_to(
            roots["runs_root"]).as_posix(), "status": "READY"})
    return _json(roots["preflight_dir"] / "preflight_summary.json")


def _ready(config: Mapping[str, Any]) -> tuple[dict[str, Any], PreflightReceipt]:
    root = _roots(config)["preflight_dir"]
    summary, receipt_path, identity_path = (root / "preflight_summary.json",
                                             root / "ready_receipt.json", root / "identity.json")
    _require(summary.is_file() and receipt_path.is_file() and identity_path.is_file(),
             "an executed READY preflight is required", FileNotFoundError)
    result = _json(summary)
    teacher_replay = _json(_roots(config)["teacher_dir"] / "production_replay_summary.json")
    receipt = PreflightReceipt.from_mapping(_json(receipt_path))
    identity = ResearchRunIdentity.from_mapping(_json(identity_path))
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    _require(result.get("task_id") == TASK_ID and result.get("status") == "READY"
             and result.get("engineering_adam_calls") == 9
             and receipt.status == "READY"
             and teacher_replay.get("status") == "ALL_TEACHER_OUTPUTS_REPLAYED"
             and teacher_replay.get("rows") == TRAIN_N
             and result.get("teacher_output_replay_rows") == TRAIN_N,
             "READY receipt, teacher replay, or summary failed its consumer")
    return result, receipt


def _append_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _observe(config: Mapping[str, Any], inputs: Mapping[str, Any], population: Mapping[str, Any],
             mapper: torch.nn.Module, device: torch.device, step: int) -> None:
    root = _roots(config)["attempt_dir"] / "observations"
    _require(not (root / f"step_{step:05d}_train.pt").exists(), "observation point already exists")
    model = exposure._make_runtime_model(inputs, 2048, device)
    records = {}
    for role in ("train", "development"):
        cohort = population[role]
        commands, actual = [], []
        with torch.no_grad():
            for start in range(0, len(cohort["labels"]), BATCH):
                target = cohort["coefficients"][start:start + BATCH].to(device=device, dtype=torch.complex128)
                cmd, _ = _fixed_map(mapper, target, inputs, device)
                out = projected._forward_amplitude(model, cmd)
                commands.append(cmd.cpu())
                actual.append(out["coefficients"].cpu())
        coeff = torch.cat(actual)
        targets = cohort["coefficients"].to(torch.complex128)
        errors = [projected._phase_error(coeff[i], targets[i]) for i in range(len(targets))]
        summary = {"task_id": TASK_ID, "step": step, "role": cohort["role"],
                   "count": len(targets), "coefficient_error_mean": float(np.mean(errors)),
                   "coefficient_error_rms": float(np.sqrt(np.mean(np.square(errors)))),
                   "coefficient_error_median": float(np.median(errors)),
                   "coefficient_error_max": float(np.max(errors))}
        records[role] = summary
        _save(root / f"step_{step:05d}_{role}.pt", {
            **summary, "labels": cohort["labels"], "proposal_ids": cohort["proposal_ids"],
            "target_coefficients": targets, "commands": torch.cat(commands),
            "actual_coefficients_2048": coeff})
    _write_json(root / f"step_{step:05d}.json", records)


def _load_checkpoint(path: Path, mapper: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     schedule: torch.Tensor, run_id: str) -> tuple[int, list[dict[str, Any]], torch.Tensor]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    step = int(saved.get("completed_updates", -1))
    _require(saved.get("task_id") == TASK_ID and saved.get("schema") == f"{TASK_ID}.checkpoint.v1"
             and saved.get("run_id") == run_id and 0 <= step <= UPDATES
             and saved.get("schedule_cursor") == step and saved.get("history_count") == step
             and len(saved.get("history", [])) == step and saved.get("optimizer_deepcopied") is True
             and torch.equal(saved.get("schedule"), schedule)
             and torch.equal(saved.get("next_batch"), schedule[step] if step < UPDATES else None)
             and saved.get("lineage_mapper_updates") == 25_000,
             "explicit checkpoint identity, schedule, or history prefix is invalid")
    counts = saved.get("counts")
    expected = torch.bincount(schedule[:step].reshape(-1), minlength=TRAIN_N)
    _require(torch.is_tensor(counts) and counts.dtype == torch.long and torch.equal(counts, expected)
             and int(saved.get("adam_step", -1)) == step,
             "checkpoint per-target exposure counts or Adam step are invalid")
    for index, row in enumerate(saved["history"]):
        _require(row.get("completed_update") == index + 1
                 and row.get("sample_indices") == schedule[index].tolist()
                 and row.get("labels") == list(range(CLASS_COUNT))
                 and row.get("loss_is_pre_update") is True,
                 f"checkpoint history or discrete sample schedule differs at update {index + 1}")
    mapper.load_state_dict(saved["mapper_state"], strict=True)
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer_state"]))
    adam_steps = fixed._adam_steps(optimizer.state_dict())
    expected_adam_steps = [] if step == 0 else [step] * 4
    _require(adam_steps == expected_adam_steps,
             f"restored Adam parameter steps differ from checkpoint: {adam_steps}")
    _require(all(torch.isfinite(value).all().item() for value in mapper.state_dict().values()),
             "checkpoint contains a nonfinite mapper parameter")
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                _require(torch.isfinite(value).all().item(), "checkpoint contains nonfinite Adam state")
    torch.set_rng_state(saved["rng_state"]["cpu"])
    torch.cuda.set_rng_state_all(saved["rng_state"]["cuda"])
    return step, [dict(x) for x in saved["history"]], counts.clone()


def _train(config: Mapping[str, Any], checkpoint_path: Path | None = None) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "scientific training requires AutoDL CUDA")
    fixed._set_numerics()
    roots = _roots(config)
    preflight, receipt = _ready(config)
    population_data = _load_population(config)
    cohorts = population_data["cohorts"]
    teachers = _load_teachers(config)
    schedule = _schedule(config)
    inputs, _, _ = _load_inputs(config)
    device = torch.device("cuda:0")
    run_id = str(preflight["run_id"])
    if checkpoint_path is None:
        _require(not roots["attempt_dir"].exists(), "scientific attempt already exists", FileExistsError)
        mapper = _mapper(inputs, device)
        optimizer = _optimizer(mapper, config)
        _require(not optimizer.state and fixed._compare_states(
            mapper.state_dict(), inputs["source_mapper"]["mapper_state"]) == 0.0,
            "scientific mapper must begin from the explicit V2 endpoint with empty Adam")
        identity = ResearchRunIdentity.from_mapping(_json(roots["preflight_dir"] / "identity.json"))
        control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
        control.mark_preflight_ready(receipt)
        attempt_identity = control.open_scientific_attempt(
            roots["attempt_dir"], f"{run_id}:attempt_01", receipt)
        (roots["attempt_dir"] / "checkpoints").mkdir(parents=True)
        _save(roots["attempt_dir"] / "population.pt", population_data)
        _save(roots["attempt_dir"] / "teachers.pt", teachers.cpu())
        _save(roots["attempt_dir"] / "schedule.pt", schedule)
        _save(roots["attempt_dir"] / "mapper_initialization.pt", {
            "task_id": TASK_ID, "source_checkpoint": config["source"]["v2_terminal_checkpoint"],
            "mapper_state": {k: v.detach().cpu().clone() for k, v in mapper.state_dict().items()},
            "adam_state": {}, "lineage_mapper_updates": 10_000})
        _write_json(roots["attempt_dir"] / "scientific_attempt_identity.json", {
            "task_id": TASK_ID, "run_id": run_id, "identity": attempt_identity.to_dict()})
        _write_json(roots["attempt_dir"] / "run_config.json", config)
        torch.manual_seed(int(config["training"]["rng_seed"]))
        torch.cuda.manual_seed_all(int(config["training"]["rng_seed"]))
        history, counts, start = [], torch.zeros(TRAIN_N, dtype=torch.long), 0
        _save_training_state(roots["attempt_dir"] / "checkpoints" / "checkpoint_step_00000.pt",
                             mapper, optimizer, history, counts, schedule, run_id, start)
        _write_json(roots["attempt_dir"] / "task_state.json", {
            "task_id": TASK_ID, "run_id": run_id, "state": "SCIENTIFIC_ACTIVE",
            "completed_updates": 0, "authorized_updates": UPDATES})
        for role in ("train", "development"):
            cohorts[role]["role"] = cohorts[role].get("role", role)
        _observe(config, inputs, cohorts, mapper, device, 0)
        # Direct decoder development semantics are the fixed reference for the final proxy gate.
        semantic, _ = fixed._direct_decoder_semantics(inputs, cohorts["development"], device)
        _write_json(roots["attempt_dir"] / "semantic_reference_development.json", semantic)
    else:
        _require(roots["attempt_dir"].is_dir() and checkpoint_path.is_file()
                 and checkpoint_path.resolve().is_relative_to((roots["attempt_dir"] / "checkpoints").resolve()),
                 "resume requires a checkpoint inside the explicit active attempt", ValueError)
        state = _json(roots["attempt_dir"] / "task_state.json")
        _require(state.get("state") == "SCIENTIFIC_ACTIVE"
                 and not (roots["attempt_dir"] / "final_decision.json").exists(),
                 "terminal attempt cannot be resumed")
        attempt = _json(roots["attempt_dir"] / "scientific_attempt_identity.json")
        identity = ResearchRunIdentity.from_mapping(_json(roots["preflight_dir"] / "identity.json"))
        control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
        control.mark_preflight_ready(receipt)
        opened = control.open_scientific_attempt(roots["attempt_dir"], f"{run_id}:attempt_01", receipt,
                                                 resume=True)
        _require(ResearchRunIdentity.from_mapping(attempt["identity"]) == opened,
                 "attempt lineage differs from preflight READY")
        mapper, optimizer = _mapper(inputs, device), None
        optimizer = _optimizer(mapper, config)
        start, history, counts = _load_checkpoint(checkpoint_path.resolve(), mapper, optimizer,
                                                  schedule, run_id)
    model = exposure._make_runtime_model(inputs, 2048, device)
    engine = preflight["engine"]
    history_file = (roots["attempt_dir"] / "training_history.jsonl" if checkpoint_path is None else
                    roots["attempt_dir"] / "training_history_segments" /
                    f"resume_from_{start:05d}_{uuid.uuid4().hex}.jsonl")
    latest = roots["attempt_dir"] / "checkpoints" / "latest.pt"
    flushed = start
    for step in range(start, UPDATES):
        indices = schedule[step]
        rec = _update(mapper, optimizer, cohorts["train"]["coefficients"][indices], teachers[indices],
                      inputs, model, device, engine, config, audit=(step == start))
        counts.index_add_(0, indices, torch.ones_like(indices))
        item = {"completed_update": step + 1, "sample_indices": indices.tolist(),
                "labels": list(range(CLASS_COUNT)), "loss_is_pre_update": True,
                "exposure_count_after_update": counts[indices].tolist(), **rec}
        history.append(item)
        complete = step + 1
        if complete % config["training"]["progress_interval"] == 0:
            _append_history(history_file, history[flushed:])
            flushed = len(history)
            _write_json(roots["attempt_dir"] / "progress.json", {
                "task_id": TASK_ID, "run_id": run_id, "completed_updates": complete,
                "authorized_updates": UPDATES, "engine": engine, "updated_unix": time.time(),
                "latest_losses": {k: item[k] for k in ("coefficient_loss", "box_loss", "teacher_loss")}})
            _write_json(roots["attempt_dir"] / "task_state.json", {
                "task_id": TASK_ID, "run_id": run_id, "state": "SCIENTIFIC_ACTIVE",
                "completed_updates": complete, "authorized_updates": UPDATES,
                "adam_step": complete, "lineage_mapper_updates": 10_000 + complete,
                "engine": engine})
        if complete % config["training"]["checkpoint_interval"] == 0:
            if latest.exists():
                shutil.copy2(latest, roots["attempt_dir"] / "checkpoints" / "previous.pt")
            saved = _save_training_state(latest, mapper, optimizer, history, counts, schedule, run_id, complete)
            if complete in (5000, 10000, 15000):
                fixed_path = roots["attempt_dir"] / "checkpoints" / f"checkpoint_step_{complete:05d}.pt"
                _save(fixed_path, saved)
                _write_json(roots["attempt_dir"] / "task_state.json", {
                    "task_id": TASK_ID, "run_id": run_id, "state": "SCIENTIFIC_ACTIVE",
                    "completed_updates": complete, "authorized_updates": UPDATES,
                    "adam_step": complete, "lineage_mapper_updates": 10_000 + complete,
                    "engine": engine})
                if complete in (5000, 10000):
                    _observe(config, inputs, cohorts, mapper, device, complete)
    _require(torch.equal(counts, torch.full((TRAIN_N,), EXPOSURES)),
             "training did not expose every fixed target exactly 75 times")
    _append_history(history_file, history[flushed:])
    _save_training_state(roots["attempt_dir"] / "checkpoints" / "terminal_checkpoint.pt",
                         mapper, optimizer, history, counts, schedule, run_id, UPDATES)
    _write_json(roots["attempt_dir"] / "task_state.json", {
        "task_id": TASK_ID, "run_id": run_id, "state": "TERMINAL_MEASUREMENT_PENDING",
        "completed_updates": UPDATES, "authorized_updates": UPDATES,
        "adam_step": UPDATES, "lineage_mapper_updates": 25_000, "engine": engine})
    for role in ("train", "development"):
        _observe(config, inputs, cohorts, mapper, device, UPDATES)
    return {"task_id": TASK_ID, "run_id": run_id, "completed_updates": UPDATES,
            "lineage_mapper_updates": 25_000, "state": "TERMINAL_MEASUREMENT_PENDING",
            "terminal_checkpoint": str(roots["attempt_dir"] / "checkpoints" / "terminal_checkpoint.pt")}


def _recovery_worker(config: Mapping[str, Any], checkpoint_path: Path,
                     evidence_dir: Path, output_path: Path) -> None:
    _require(torch.cuda.is_available(), "recovery worker requires CUDA")
    fixed._set_numerics()
    roots = _roots(config)
    evidence_dir = evidence_dir.resolve()
    checkpoint_path = checkpoint_path.resolve()
    output_path = output_path.resolve()
    _require(evidence_dir.is_relative_to(roots["runs_root"].resolve())
             and checkpoint_path.is_file()
             and checkpoint_path.is_relative_to(evidence_dir)
             and output_path.is_relative_to(evidence_dir),
             "recovery worker requires explicit paths inside its selected preflight directory")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    schedule = _schedule(config)
    _require(state.get("task_id") == TASK_ID and state.get("schema") == f"{TASK_ID}.recovery.v1"
             and state.get("completed_updates") == 3 and state.get("schedule_cursor") == 3
             and torch.equal(state.get("schedule"), schedule)
             and torch.equal(state.get("next_batch"), schedule[3])
             and len(state.get("history", [])) == 3,
             "recovery checkpoint does not contain the exact three-update prefix")
    inputs, _, _ = _load_inputs(config)
    population = _load_population(config)["cohorts"]
    teachers = _load_teachers(config)
    device = torch.device("cuda:0")
    mapper = _mapper(inputs, device)
    mapper.load_state_dict(state["mapper_state"], strict=True)
    optimizer = _optimizer(mapper, config)
    optimizer.load_state_dict(copy.deepcopy(state["optimizer_state"]))
    torch.set_rng_state(state["rng_state"]["cpu"])
    torch.cuda.set_rng_state_all(state["rng_state"]["cuda"])
    model = exposure._make_runtime_model(inputs, 2048, device)
    index = schedule[3]
    update = _update(mapper, optimizer, population["train"]["coefficients"][index],
                     teachers[index], inputs, model, device, state["engine"], config)
    payload = {"task_id": TASK_ID, "schema": f"{TASK_ID}.recovery_step4.v1",
               "completed_updates": 4, "indices": index.clone(), "engine": state["engine"],
               "mapper_state": {k: v.detach().cpu().clone() for k, v in mapper.state_dict().items()},
               "optimizer_state": copy.deepcopy(optimizer.state_dict()), "loss": update["loss"],
               "rng_state": _rng(),
               "history": [*state["history"], {"step": 4, "indices": index.tolist(), **update}]}
    _save(output_path, payload)


def _restore_mapper(inputs: Mapping[str, Any], mapper_state: Mapping[str, torch.Tensor],
                    device: torch.device) -> torch.nn.Module:
    mapper = _mapper(inputs, device)
    mapper.load_state_dict(mapper_state, strict=True)
    mapper.eval()
    return mapper


def _legacy_command(mapper: torch.nn.Module, target: torch.Tensor, inputs: Mapping[str, Any],
                    device: torch.device) -> torch.Tensor:
    target = target.reshape(1, PORT_COUNT).to(device=device, dtype=torch.complex128)
    features = _features(target, inputs["scale"], device)
    command, _ = projected._map_amplitude(
        mapper, features, target, inputs["matrix"].to(device), inputs["inverse"].to(device),
        inputs["scale"].to(device))
    return command.detach().to(torch.float32)


def _command_for_condition(condition: str, mapper: torch.nn.Module,
                           target: torch.Tensor, inputs: Mapping[str, Any],
                           device: torch.device) -> torch.Tensor:
    if condition == "student_fixed_phase_zero":
        return _fixed_map(mapper, target, inputs, device)[0].detach().to(torch.float32)
    if condition in {"v2_mapper", "fixed_frame_terminal"}:
        return _legacy_command(mapper, target, inputs, device)
    raise ValueError(f"unregistered terminal command condition: {condition}")


def _terminal_states(config: Mapping[str, Any], inputs: Mapping[str, Any],
                     terminal: Mapping[str, Any], fixed_terminal: Mapping[str, Any],
                     device: torch.device
                     ) -> dict[str, torch.nn.Module]:
    return {
        "student_fixed_phase_zero": _restore_mapper(inputs, terminal["mapper_state"], device),
        "v2_mapper": _restore_mapper(inputs, inputs["source_mapper"]["mapper_state"], device),
        "fixed_frame_terminal": _restore_mapper(inputs, fixed_terminal["mapper_state"], device),
    }


def _command_seed_dir(config: Mapping[str, Any]) -> Path:
    return _roots(config)["attempt_dir"] / "terminal_command_seed"


def _create_command_seeds(config: Mapping[str, Any], checkpoint_path: Path) -> dict[str, Any]:
    roots = _roots(config)
    attempt = roots["attempt_dir"]
    _require(checkpoint_path.is_file() and checkpoint_path.resolve() ==
             (attempt / "checkpoints" / "terminal_checkpoint.pt").resolve(),
             "measurement must use the explicit terminal checkpoint", ValueError)
    terminal = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    _require(terminal.get("task_id") == TASK_ID and terminal.get("completed_updates") == UPDATES
             and terminal.get("lineage_mapper_updates") == 25_000,
             "terminal mapper checkpoint is not the fixed 15,000-update endpoint")
    inputs, _, fixed_terminal = _load_inputs(config)
    population = _load_population(config)["cohorts"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mappers = _terminal_states(config, inputs, terminal, fixed_terminal, device)
    # Use the fixed-frame checkpoint explicitly loaded from its frozen source directory.
    _require(fixed_terminal.get("completed_updates") == 15000
             and fixed_terminal.get("lineage_mapper_updates") == 25000
             and float(fixed_terminal.get("selected_alpha", -1.0)) == 1.0,
             "paired fixed-frame endpoint does not match the registered 15,000-step source")
    seed_dir = _command_seed_dir(config)
    seed_dir.mkdir(parents=True, exist_ok=True)
    command_counts = {"train": 0, "development": 0, "holdout": 0}
    for role in ("train", "development", "holdout"):
        cohort = population[role]
        for start in range(0, len(cohort["labels"]), BATCH):
            stop = min(start + BATCH, len(cohort["labels"]))
            path = seed_dir / f"{role}_{start:04d}_{stop:04d}.pt"
            conditions = ["student_fixed_phase_zero"]
            if role == "holdout":
                conditions.extend(("v2_mapper", "fixed_frame_terminal"))
            commands = {condition: [] for condition in conditions}
            with torch.no_grad():
                for index in range(start, stop):
                    target = cohort["coefficients"][index]
                    for condition in conditions:
                        commands[condition].append(_command_for_condition(
                            condition, mappers[condition], target, inputs, device).cpu())
            payload = {"schema": f"{TASK_ID}.terminal_command_seed.v1", "task_id": TASK_ID,
                       "run_id": _json(attempt / "task_state.json")["run_id"],
                       "role": role, "start": start, "stop": stop,
                       "labels": cohort["labels"][start:stop].clone(),
                       "proposal_ids": cohort["proposal_ids"][start:stop].clone(),
                       "target_coefficients": cohort["coefficients"][start:stop].clone(),
                       "conditions": {key: torch.cat(value, dim=0).to(torch.float32)
                                      for key, value in commands.items()}}
            if path.exists():
                previous = torch.load(path, map_location="cpu", weights_only=False)
                _require(previous.get("task_id") == TASK_ID and previous.get("role") == role
                         and previous.get("start") == start and previous.get("stop") == stop
                         and all(torch.equal(previous["conditions"][key], payload["conditions"][key])
                                 for key in conditions),
                         f"existing command seed differs from explicit terminal checkpoint: {path.name}")
            else:
                _save(path, payload)
            command_counts[role] += stop - start
    state_payload = {key: {name: value.detach().cpu().clone()
                           for name, value in mappers[key].state_dict().items()}
                     for key in mappers}
    state_path = attempt / "terminal_command_seed" / "loaded_model_states.pt"
    if state_path.exists():
        previous = torch.load(state_path, map_location="cpu", weights_only=False)
        _require(previous.get("task_id") == TASK_ID and previous.get("terminal_run_id") == terminal.get("run_id")
                 and all(fixed._compare_states(previous["states"][key], state_payload[key]) == 0.0
                         for key in state_payload),
                 "existing command-seed model binding differs from the explicit checkpoints")
    else:
        _save(state_path, {"task_id": TASK_ID, "terminal_run_id": terminal.get("run_id"),
                           "states": state_payload})
    summary = {"task_id": TASK_ID, "status": "COMMAND_SEEDS_FIXED",
               "checkpoint": str(checkpoint_path), "counts": command_counts,
               "conditions": ["student_fixed_phase_zero", "v2_mapper", "fixed_frame_terminal"],
               "single_sample_inference": True}
    _write_json(seed_dir / "summary.json", summary)
    return summary


def _measurement_worker(config: Mapping[str, Any], checkpoint_path: Path,
                        output_dir: Path) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "full optical terminal measurement requires AutoDL CUDA")
    fixed._set_numerics()
    roots = _roots(config)
    attempt = roots["attempt_dir"]
    _require(checkpoint_path.is_file() and checkpoint_path.resolve() ==
             (attempt / "checkpoints" / "terminal_checkpoint.pt").resolve()
             and output_dir.resolve() == (attempt / "measurements").resolve(),
             "measurement worker must consume this attempt's explicit terminal endpoint", ValueError)
    terminal = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = _json(attempt / "task_state.json")
    _require(state.get("task_id") == TASK_ID and state.get("state") == "TERMINAL_MEASUREMENT_PENDING"
             and state.get("completed_updates") == UPDATES,
             "measurement is allowed only after the fixed terminal update count")
    inputs, _, fixed_terminal = _load_inputs(config)
    population = _load_population(config)["cohorts"]
    schedule = _schedule(config)
    device = torch.device("cuda:0")
    mapper = _mapper(inputs, device)
    optimizer = _optimizer(mapper, config)
    run_id = str(state["run_id"])
    start, history, counts = _load_checkpoint(checkpoint_path, mapper, optimizer, schedule, run_id)
    _require(start == UPDATES and torch.equal(counts, torch.full((TRAIN_N,), EXPOSURES)),
             "terminal checkpoint does not contain the complete fixed exposure schedule")
    _require(len(history) == UPDATES, "terminal checkpoint history is incomplete")
    seed_dir = _command_seed_dir(config)
    seed_summary = _json(seed_dir / "summary.json")
    _require(seed_summary.get("task_id") == TASK_ID and seed_summary.get("status") == "COMMAND_SEEDS_FIXED",
             "single-sample terminal command seed set is missing")
    mappers = _terminal_states(config, inputs, terminal, fixed_terminal, device)
    saved_loaded_states = torch.load(seed_dir / "loaded_model_states.pt", map_location="cpu", weights_only=False)
    for condition, model_mapper in mappers.items():
        _require(fixed._compare_states(saved_loaded_states["states"][condition],
                                       model_mapper.state_dict()) == 0.0,
                 f"independent CUDA mapper differs from saved command seed: {condition}")
    _save(attempt / "independent_loaded_model_states.pt", {
        "task_id": TASK_ID, "run_id": run_id,
        "states": {condition: {key: value.detach().cpu().clone()
                               for key, value in model_mapper.state_dict().items()}
                   for condition, model_mapper in mappers.items()}})
    models = {grid: exposure._make_runtime_model(inputs, grid, device) for grid in GRIDS}
    fixed_states = {grid: {key: value.detach().cpu().clone()
                           for key, value in models[grid].state_dict().items()} for grid in GRIDS}
    output_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, list[dict[str, Any]]] = {role: [] for role in population}
    replayed = {"train": 0, "development": 0, "holdout": {"student_fixed_phase_zero": 0,
                                                              "v2_mapper": 0,
                                                              "fixed_frame_terminal": 0}}
    command_delta = 0
    for role in ("train", "development", "holdout"):
        cohort = population[role]
        for start_index in range(0, len(cohort["labels"]), BATCH):
            stop_index = min(start_index + BATCH, len(cohort["labels"]))
            seed_path = seed_dir / f"{role}_{start_index:04d}_{stop_index:04d}.pt"
            seed = torch.load(seed_path, map_location="cpu", weights_only=False)
            _require(seed.get("task_id") == TASK_ID and seed.get("run_id") == run_id
                     and seed.get("role") == role and seed.get("start") == start_index
                     and seed.get("stop") == stop_index
                     and torch.equal(seed["labels"], cohort["labels"][start_index:stop_index])
                     and torch.equal(seed["proposal_ids"], cohort["proposal_ids"][start_index:stop_index])
                     and torch.equal(seed["target_coefficients"],
                                     cohort["coefficients"][start_index:stop_index]),
                     f"terminal command seed has wrong target identity: {seed_path.name}")
            targets = cohort["coefficients"][start_index:stop_index].to(
                device=device, dtype=torch.complex128)
            condition_names = list(seed["conditions"])
            generated: dict[str, torch.Tensor] = {name: [] for name in condition_names}
            with torch.no_grad():
                for local in range(stop_index - start_index):
                    target = targets[local:local + 1]
                    for condition in condition_names:
                        generated[condition].append(_command_for_condition(
                            condition, mappers[condition], target, inputs, device).cpu())
            commands = {name: torch.cat(values).to(torch.float32)
                        for name, values in generated.items()}
            for condition in condition_names:
                _require(torch.equal(commands[condition], seed["conditions"][condition]),
                         f"independent CUDA terminal command replay is not bitwise identical: {role}/{condition}")
                command_delta = max(command_delta, float((commands[condition]
                    - seed["conditions"][condition]).abs().max()))
                if role == "holdout":
                    replayed[role][condition] += stop_index - start_index
            if role != "holdout":
                replayed[role] += stop_index - start_index
            measurement_path = output_dir / f"{role}_{start_index:04d}_{stop_index:04d}.pt"
            if measurement_path.exists():
                previous = torch.load(measurement_path, map_location="cpu", weights_only=False)
                _require(previous.get("task_id") == TASK_ID and previous.get("run_id") == run_id
                         and previous.get("role") == role and previous.get("start") == start_index
                         and previous.get("stop") == stop_index
                         and all(torch.equal(previous["commands"][name], commands[name])
                                 for name in condition_names),
                         f"existing measurement shard is not bound to its replayed commands: {measurement_path.name}")
                for rows in previous["metric_rows"].values():
                    records[role].extend(rows)
                continue
            output_values: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
            references = {}
            with torch.no_grad():
                for grid in GRIDS:
                    references[grid] = projected._forward_target(models[grid], targets)["raw_cmos"]
                for condition in condition_names:
                    output_values[condition] = {
                        grid: {key: value for key, value in projected._forward_amplitude(
                            models[grid], commands[condition].to(device=device, dtype=torch.float32)).items()
                               if key in ("coefficients", "raw_cmos")}
                        for grid in GRIDS}
            metric_rows: dict[str, list[dict[str, Any]]] = {}
            for condition in condition_names:
                metric_rows[condition] = []
                for local in range(stop_index - start_index):
                    metrics = projected._error_row(
                        {grid: {key: value[local:local + 1]
                                for key, value in output_values[condition][grid].items()}
                         for grid in GRIDS}, targets[local:local + 1],
                        {grid: references[grid][local:local + 1] for grid in GRIDS}, 0)
                    row_index = start_index + local
                    row = {"population_index": row_index,
                           "label": int(cohort["labels"][row_index]),
                           "sample_index_within_class": row_index % (len(cohort["labels"]) // CLASS_COUNT),
                           "proposal_id": int(cohort["proposal_ids"][row_index]),
                           **metrics,
                           "row_passed": all(float(metrics[key]) <= LIMIT for key in METRIC_KEYS)}
                    metric_rows[condition].append(row)
                    records[role].append({**row, "condition": condition})
            shard = {"schema": f"{TASK_ID}.measurement_shard.v1", "task_id": TASK_ID,
                     "run_id": run_id, "role": role, "start": start_index, "stop": stop_index,
                     "labels": cohort["labels"][start_index:stop_index].clone(),
                     "proposal_ids": cohort["proposal_ids"][start_index:stop_index].clone(),
                     "target_coefficients": cohort["coefficients"][start_index:stop_index].clone(),
                     "commands": {key: value.clone() for key, value in commands.items()},
                     "references": {grid: references[grid].detach().cpu().clone() for grid in GRIDS},
                     "outputs": {condition: {grid: {key: value.detach().cpu().clone()
                                                   for key, value in output_values[condition][grid].items()}
                                              for grid in GRIDS} for condition in condition_names},
                     "metric_rows": metric_rows,
                     "command_binding": "independent CUDA process reloaded terminal and source checkpoints; exact replay passed"}
            _save(measurement_path, shard)
    for grid in GRIDS:
        projected._assert_frozen(models[grid], fixed_states[grid], f"terminal_measurement_optics_{grid}")
    _save(attempt / "terminal_loaded_mapper_state.pt", {
        "task_id": TASK_ID, "run_id": run_id,
        "mapper_state": {key: value.detach().cpu().clone() for key, value in mapper.state_dict().items()},
        "adam_steps": fixed._adam_steps(optimizer.state_dict()),
        "completed_updates": start})
    summary = {"task_id": TASK_ID, "status": "TERMINAL_MEASUREMENT_COMPLETE",
               "run_id": run_id, "terminal_checkpoint": str(checkpoint_path),
               "single_sample_command_replay": True, "maximum_command_absolute_delta": command_delta,
               "measured_rows": replayed, "scientific_updates": 0}
    _write_json(output_dir / "worker_summary.json", summary)
    return summary


def _numpy_metric_row(outputs: Mapping[int, Mapping[str, torch.Tensor]],
                      target: torch.Tensor,
                      references: Mapping[int, torch.Tensor]) -> dict[str, float]:
    def phase_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
        a = actual.detach().cpu().numpy().astype(np.complex128).reshape(-1)
        t = expected.detach().cpu().numpy().astype(np.complex128).reshape(-1)
        denominator = float(np.linalg.norm(t))
        _require(denominator > 0.0 and np.isfinite(a.real).all() and np.isfinite(a.imag).all()
                 and np.isfinite(t.real).all() and np.isfinite(t.imag).all(),
                 "independent NumPy complex metric inputs are invalid")
        phase = float(np.angle(np.vdot(t, a)))
        return float(np.linalg.norm(a - np.exp(1j * phase) * t) / denominator)

    def raw_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
        a = actual.detach().cpu().numpy().astype(np.float64).reshape(-1)
        t = expected.detach().cpu().numpy().astype(np.float64).reshape(-1)
        denominator = float(np.linalg.norm(t))
        _require(denominator > 0.0 and np.isfinite(a).all() and np.isfinite(t).all(),
                 "independent NumPy raw CMOS metric inputs are invalid")
        return float(np.linalg.norm(a - t) / denominator)

    a2048, a4096 = outputs[2048], outputs[4096]
    return {"coefficient_2048": phase_error(a2048["coefficients"][0], target[0]),
            "coefficient_4096": phase_error(a4096["coefficients"][0], target[0]),
            "native_cmos_2048": raw_error(a2048["raw_cmos"][0], references[2048][0]),
            "native_cmos_4096": raw_error(a4096["raw_cmos"][0], references[4096][0]),
            "cross_grid_coefficient": phase_error(a2048["coefficients"][0], a4096["coefficients"][0]),
            "cross_grid_native_cmos": raw_error(a2048["raw_cmos"][0], a4096["raw_cmos"][0])}


def _measurement_rows(config: Mapping[str, Any], evidence_dir: Path,
                      population: Mapping[str, Any]
                      ) -> tuple[dict[str, dict[str, list[dict[str, Any]]]],
                                 dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    output_dir = evidence_dir / "measurements"
    worker = _json(output_dir / "worker_summary.json")
    _require(worker.get("task_id") == TASK_ID and worker.get("status") == "TERMINAL_MEASUREMENT_COMPLETE"
             and worker.get("single_sample_command_replay") is True,
             "complete independent terminal measurement worker record is required")
    row_groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        role: {} for role in ("train", "development", "holdout")}
    raw_groups: dict[str, dict[int, list[torch.Tensor]]] = {
        name: {grid: [] for grid in GRIDS}
        for name in ("direct_decoder", "student_fixed_phase_zero", "v2_mapper", "fixed_frame_terminal")}
    command_counts = {"train": 0, "development": 0, "holdout": 0}
    for role in ("train", "development", "holdout"):
        count = len(population[role]["labels"])
        seen = torch.zeros(count, dtype=torch.bool)
        expected_conditions = (["student_fixed_phase_zero"] if role != "holdout" else
                               ["student_fixed_phase_zero", "v2_mapper", "fixed_frame_terminal"])
        chunks = sorted(output_dir.glob(f"{role}_*.pt"))
        _require(chunks, f"measurement population is missing shards: {role}", FileNotFoundError)
        for path in chunks:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            start, stop = int(shard.get("start", -1)), int(shard.get("stop", -1))
            _require(shard.get("schema") == f"{TASK_ID}.measurement_shard.v1"
                     and shard.get("task_id") == TASK_ID and shard.get("role") == role
                     and shard.get("run_id") == worker.get("run_id")
                     and 0 <= start < stop <= count and not seen[start:stop].any().item()
                     and set(shard.get("outputs", {})) == set(expected_conditions)
                     and set(shard.get("commands", {})) == set(expected_conditions)
                     and set(shard.get("metric_rows", {})) == set(expected_conditions)
                     and torch.equal(shard["labels"], population[role]["labels"][start:stop])
                     and torch.equal(shard["proposal_ids"], population[role]["proposal_ids"][start:stop])
                     and torch.equal(shard["target_coefficients"],
                                     population[role]["coefficients"][start:stop])
                     and "independent CUDA process" in shard.get("command_binding", ""),
                     f"measurement shard identity or role mismatch: {path.name}")
            seed_path = evidence_dir / "terminal_command_seed" / f"{role}_{start:04d}_{stop:04d}.pt"
            seed = torch.load(seed_path, map_location="cpu", weights_only=False)
            _require(seed.get("task_id") == TASK_ID and seed.get("role") == role
                     and seed.get("start") == start and seed.get("stop") == stop
                     and all(torch.equal(shard["commands"][condition], seed["conditions"][condition])
                             for condition in expected_conditions),
                     f"measured commands differ from the independently prepared seed: {path.name}")
            seen[start:stop] = True
            command_counts[role] += stop - start
            for condition in expected_conditions:
                command = shard["commands"][condition]
                _require(tuple(command.shape) == (stop - start, 1, 200, 200)
                         and command.dtype == torch.float32 and torch.isfinite(command).all().item()
                         and float(command.min()) >= 0.0 and float(command.max()) <= 1.0,
                         f"saved terminal command is malformed: {path.name}/{condition}")
                rows = shard["metric_rows"][condition]
                _require(len(rows) == stop - start, f"metric row count differs from shard: {path.name}")
                for local, row in enumerate(rows):
                    index = start + local
                    _require(int(row.get("population_index", -1)) == index
                             and int(row.get("label", -1)) == int(population[role]["labels"][index])
                             and int(row.get("proposal_id", -1)) == int(population[role]["proposal_ids"][index]),
                             f"metric row identity differs: {path.name}/{condition}/{local}")
                    measured_outputs = {grid: {key: shard["outputs"][condition][grid][key][local:local + 1]
                                               for key in ("coefficients", "raw_cmos")}
                                        for grid in GRIDS}
                    refs = {grid: shard["references"][grid][local:local + 1] for grid in GRIDS}
                    target = shard["target_coefficients"][local:local + 1]
                    actual_metrics = _numpy_metric_row(measured_outputs, target, refs)
                    for key in METRIC_KEYS:
                        _require(math.isfinite(float(row[key]))
                                 and abs(float(row[key]) - float(actual_metrics[key])) <= 1.0e-14,
                                 f"saved terminal metric does not recompute: {path.name}/{condition}/{key}")
                    pass_value = all(float(actual_metrics[key]) <= LIMIT for key in METRIC_KEYS)
                    _require(bool(row.get("row_passed")) is pass_value,
                             f"saved row decision differs from recomputed six metrics: {path.name}")
                row_groups[role].setdefault(condition, []).extend(rows)
                if role == "holdout":
                    for grid in GRIDS:
                        raw_groups[condition][grid].append(
                            shard["outputs"][condition][grid]["raw_cmos"].clone())
            if role == "holdout":
                for grid in GRIDS:
                    raw_groups["direct_decoder"][grid].append(shard["references"][grid].clone())
        _require(bool(seen.all()), f"terminal measurement has missing {role} rows")
    return row_groups, {condition: {grid: torch.cat(values)
                                     for grid, values in grid_values.items()}
                        for condition, grid_values in raw_groups.items()}, {
        "task_id": TASK_ID, "rows": command_counts,
        "condition_counts": {"train": ["student_fixed_phase_zero"],
                             "development": ["student_fixed_phase_zero"],
                             "holdout": ["student_fixed_phase_zero", "v2_mapper", "fixed_frame_terminal"]}}


def _write_fixed_examples(output_path: Path, inputs: Mapping[str, Any], cohort: Mapping[str, Any],
                          raw_by_condition: Mapping[str, torch.Tensor], area: float, gain: float) -> None:
    from PIL import Image, ImageDraw

    conditions = ("direct_decoder", "student_fixed_phase_zero", "v2_mapper", "fixed_frame_terminal")
    readouts = {condition: pilot._readout_for_evaluator(raw_by_condition[condition], area, gain)
                for condition in conditions}
    cell = 144
    canvas = Image.new("RGB", (len(conditions) * cell, CLASS_COUNT * 2 * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for row in range(CLASS_COUNT * 2):
        index = (row // 2) * (len(cohort["labels"]) // CLASS_COUNT) + (row % 2)
        label = int(cohort["labels"][index])
        for col, condition in enumerate(conditions):
            values = (readouts[condition][index, 0].clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
            image = Image.fromarray(values, mode="L").resize((112, 112), Image.Resampling.NEAREST)
            x, y = col * cell + 16, row * cell + 24
            canvas.paste(image.convert("RGB"), (x, y))
            draw.text((col * cell + 4, row * cell + 4), f"{condition}", fill="black")
            draw.text((x, y + 114), f"class={label}, sample={row % 2}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _finalize_measurement(config: Mapping[str, Any], evidence_dir: Path) -> dict[str, Any]:
    _require(not (evidence_dir / "final_decision.json").exists(),
             "terminal measurement is already closed", FileExistsError)
    inputs, _, _ = _load_inputs(config)
    population = _load_population(config)["cohorts"]
    row_groups, raw_groups, counts = _measurement_rows(config, evidence_dir, population)
    summaries = {role: {condition: projected._summarize_rows(rows)
                        for condition, rows in groups.items()}
                 for role, groups in row_groups.items()}
    area = fixed._camera_area(inputs, torch.device("cpu"))
    gain = float(inputs["camera_gain"])
    holdout = population["holdout"]
    semantic_summaries = {}
    for condition, raw in raw_groups.items():
        readouts = pilot._readout_for_evaluator(raw[2048], area, gain)
        semantic_summaries[condition] = fixed._summary_from_readouts(inputs, holdout, readouts)
    semantic_pass, semantic_gate = fixed._semantic_candidate_passes(
        semantic_summaries["student_fixed_phase_zero"], semantic_summaries["direct_decoder"], config)
    numeric_pass = summaries["holdout"]["student_fixed_phase_zero"]["all_rows_passed"]
    final_pass = bool(numeric_pass and semantic_pass)
    final_state = "VALID_COMPLETED" if final_pass else "VALID_FAILED"
    terminal = torch.load(evidence_dir / "checkpoints" / "terminal_checkpoint.pt",
                          map_location="cpu", weights_only=False)
    decision = {"task_id": TASK_ID, "run_id": terminal["run_id"], "state": final_state,
                "evidence_tier": "diagnostic_only", "completed_updates": UPDATES,
                "lineage_mapper_updates": 25_000, "new_holdout_count": HOLDOUT_N,
                "numeric_gate_passed": bool(numeric_pass), "semantic_proxy_gate_passed": bool(semantic_pass),
                "final_gate_passed": final_pass, "holdout_numeric_summary": summaries["holdout"][
                    "student_fixed_phase_zero"], "semantic_proxy_gate": semantic_gate,
                "scientific_update_count": UPDATES,
                "scope": "one fixed physical-phase prior candidate and its new fixed holdout"}
    report = {"task_id": TASK_ID, "run_id": terminal["run_id"], "state": final_state,
              "population_counts": counts["rows"], "metric_keys": list(METRIC_KEYS),
              "threshold": LIMIT, "numeric_summaries": summaries,
              "semantic_summaries": semantic_summaries,
              "semantic_gate": semantic_gate, "final_decision": decision,
              "training": {"updates": UPDATES, "adam_steps": fixed._adam_steps(terminal["optimizer_state"]),
                           "exposures_each_train_target": EXPOSURES,
                           "history_rows": len(terminal["history"]),
                           "loss": "Lc + 5 Lbox + 1 Lteacher"},
              "human_readability": "not_measured", "physical_hardware_readiness": "not_measured"}
    _write_json(evidence_dir / "terminal_measurement_summary.json", report)
    _write_json(evidence_dir / "final_decision.json", decision)
    _write_fixed_examples(evidence_dir / "fixed_holdout_examples.png", inputs, holdout,
                          {key: raw_groups[key][2048] for key in raw_groups}, area, gain)
    state = _json(evidence_dir / "task_state.json")
    state.update({"state": final_state, "terminal_measurement_complete": True,
                  "numeric_gate_passed": bool(numeric_pass),
                  "semantic_proxy_gate_passed": bool(semantic_pass),
                  "final_gate_passed": final_pass})
    _write_json(evidence_dir / "task_state.json", state)
    markdown = (
        "# G1固定零相位教师蒸馏终态报告\n\n"
        "## 1．结果与状态\n\n"
        f"终态：`{final_state}`。训练更新 {UPDATES} 步；新留出数值通过 {summaries['holdout']['student_fixed_phase_zero']['passed_rows']}/{HOLDOUT_N}。\n\n"
        "## 2．科学认识\n\n"
        "候选固定公共相位为0，使用合法命令教师；V2与旧固定相位终点仅作同人口描述性对照。语义结果是冻结评价器代理。\n\n"
        "## 3．目标缺口\n\n"
        f"六项数值门：`{bool(numeric_pass)}`；语义代理门：`{bool(semantic_pass)}`。\n\n"
        "## 4．工程完成与验证\n\n"
        "结果来自独立CUDA进程的单样本指令重放和2048/4096完整光路测量；Windows应从原始分片复核。\n\n"
        "## 5．决策影响与下一步\n\n"
        "本报告只对固定先验、固定训练/开发/留出人口及单个候选作出结论；不证明人眼可读性或实物部署。终态关闭，不自动续训。\n")
    (evidence_dir / "closeout_report.md").write_text(markdown, encoding="utf-8")
    _write_transfer_manifest(config)
    return decision


def _write_transfer_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    roots = _roots(config)
    runs_root = roots["runs_root"].resolve()
    _require(runs_root.is_dir(), "the task runs root must exist before manifest creation")
    selected_dirs = [roots[name].resolve() for name in (
        "source_dir", "population_dir", "teacher_dir", "preflight_dir", "attempt_dir")]
    files = []
    for directory in selected_dirs:
        _require(directory.is_dir(), f"required evidence directory is missing: {directory}", FileNotFoundError)
        for item in sorted(directory.rglob("*")):
            if item.is_file() and ".tmp." not in item.name:
                files.append(item.relative_to(runs_root).as_posix())
    manifest_path = runs_root / "transfer_manifest.json"
    files = sorted(set(files) | {"preflight_selection.json", "transfer_manifest.json"})
    payload = {"schema": f"{TASK_ID}.transfer_manifest.v1", "task_id": TASK_ID,
               "files": files, "file_count": len(files),
               "includes_raw_optical_measurements": True,
               "includes_teacher_commands_and_measurements": True}
    _write_json(manifest_path, payload)
    return payload


def _numpy_tree_delta(left: Any, right: Any) -> float:
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape:
            return float("inf")
        if left.is_complex() or right.is_complex():
            delta = (left.to(torch.complex128) - right.to(torch.complex128)).abs()
        else:
            delta = (left.to(torch.float64) - right.to(torch.float64)).abs()
        return float(delta.max()) if delta.numel() else 0.0
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return float("inf")
        return max((_numpy_tree_delta(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return float("inf")
        return max((_numpy_tree_delta(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else float("inf")


def _verify_saved(config: Mapping[str, Any], evidence_dir: Path,
                  report_dir: Path | None = None) -> dict[str, Any]:
    root = evidence_dir.resolve()
    attempt = root / "attempt_01"
    population_path = root / "population_01" / "population.pt"
    teacher_dir = root / "teacher_01"
    selection_path = root / "preflight_selection.json"
    _require(selection_path.is_file(), "saved evidence lacks its selected READY preflight", FileNotFoundError)
    selection = _json(selection_path)
    preflight_dir = (root / str(selection.get("selected_path", ""))).resolve()
    _require(selection.get("task_id") == TASK_ID and selection.get("status") == "READY"
             and preflight_dir.is_relative_to(root)
             and preflight_dir.is_dir(),
             "selected READY preflight pointer is invalid")
    _require(root.is_dir() and attempt.is_dir() and population_path.is_file()
             and (root / "source_inputs_01").is_dir()
             and teacher_dir.is_dir() and preflight_dir.is_dir(),
             "verify-saved requires the complete task runs root or its Windows backup", FileNotFoundError)
    manifest_path = root / "transfer_manifest.json"
    _require(manifest_path.is_file(), "saved evidence lacks its explicit transfer manifest", FileNotFoundError)
    manifest = _json(manifest_path)
    manifest_files = manifest.get("files")
    _require(manifest.get("task_id") == TASK_ID and isinstance(manifest_files, list)
             and len(manifest_files) == len(set(manifest_files))
             and "attempt_01/final_decision.json" in manifest_files
             and manifest.get("includes_raw_optical_measurements") is True,
             "transfer manifest is malformed or omits required raw measurement evidence")
    for relative in manifest_files:
        candidate = (root / relative).resolve()
        _require(candidate.is_relative_to(root) and candidate.is_file(),
                 f"manifest-listed file is missing or escapes the evidence root: {relative}", FileNotFoundError)
    transfer_receipt_path = root / "transfer_receipt.json"
    require_transfer = root.name.startswith("autodl_backup_")
    if require_transfer:
        _require(transfer_receipt_path.is_file(), "Windows backup has no real transfer receipt", FileNotFoundError)
        receipt = _json(transfer_receipt_path)
        received = receipt.get("files", [])
        _require(receipt.get("status") == "COMPLETE" and len(received) == len(manifest_files)
                 and {item.get("path") for item in received} == set(manifest_files),
                 "transfer receipt is incomplete or does not match the remote manifest")
        for item in received:
            local_path = root / str(item["path"])
            actual_bytes = local_path.stat().st_size if local_path.is_file() else -1
            _require(item.get("status") == "COPIED" and int(item.get("scp_exit_code", -1)) == 0
                     and int(item.get("remote_size_bytes", -1)) == actual_bytes
                     and int(item.get("local_size_bytes", -1)) == actual_bytes,
                     f"current file size differs from the successful remote transfer receipt: {item.get('path')}")

    population_payload = torch.load(population_path, map_location="cpu", weights_only=False)
    _require(population_payload.get("task_id") == TASK_ID
             and population_payload.get("schema") == f"{TASK_ID}.population.v1"
             and population_payload.get("holdout_not_measured_during_teacher_or_training") is True,
             "saved fixed population identity or holdout role is invalid")
    population = population_payload["cohorts"]
    _require(tuple(population["train"]["coefficients"].shape) == (TRAIN_N, PORT_COUNT)
             and tuple(population["development"]["coefficients"].shape) == (DEV_N, PORT_COUNT)
             and tuple(population["holdout"]["coefficients"].shape) == (HOLDOUT_N, PORT_COUNT),
             "saved train/development/holdout populations have invalid sizes")
    inputs, _, fixed_terminal = _load_inputs(config)
    terminal_path = attempt / "checkpoints" / "terminal_checkpoint.pt"
    terminal = torch.load(terminal_path, map_location="cpu", weights_only=False)
    preflight = _json(preflight_dir / "preflight_summary.json")
    receipt_path = preflight_dir / "ready_receipt.json"
    identity_path = preflight_dir / "identity.json"
    receipt = PreflightReceipt.from_mapping(_json(receipt_path))
    identity = ResearchRunIdentity.from_mapping(_json(identity_path))
    control = ResearchRunControl(identity=identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    _require(receipt.status == "READY" and receipt.selected_engine == preflight.get("engine")
             and preflight.get("status") == "READY" and preflight.get("engineering_adam_calls") == 9
             and preflight.get("scientific_updates") == 0,
             "preflight receipt is not the executed nine-call engineering READY")
    teacher_replay = _json(teacher_dir / "production_replay_summary.json")
    _require(teacher_replay.get("task_id") == TASK_ID
             and teacher_replay.get("status") == "ALL_TEACHER_OUTPUTS_REPLAYED"
             and teacher_replay.get("rows") == TRAIN_N,
             "all 2,000 teacher commands lack an independent production replay")
    teacher_summary = _json(teacher_dir / "teacher_summary.json")
    _require(teacher_summary.get("status") == "TEACHER_QUALIFIED"
             and teacher_summary.get("teacher_rows_passed") == TRAIN_N
             and teacher_summary.get("teacher_rows_completed") == TRAIN_N,
             "fixed-phase teacher population did not qualify in full")
    teachers = torch.load(attempt / "teachers.pt", map_location="cpu", weights_only=False)
    for index in range(TRAIN_N):
        row = torch.load(teacher_dir / f"row_{index:04d}.pt", map_location="cpu", weights_only=False)
        target = population["train"]["coefficients"][index:index + 1]
        command = row.get("amplitude_command")
        _require(row.get("task_id") == TASK_ID and row.get("index") == index
                 and row.get("qualified") is True
                 and int(row["label"]) == int(population["train"]["labels"][index])
                 and int(row["proposal_id"]) == int(population["train"]["proposal_ids"][index])
                 and torch.equal(row["target_coefficients"].reshape(1, -1), target)
                 and tuple(command.shape) == (1, 1, 200, 200) and command.dtype == torch.float32
                 and torch.isfinite(command).all().item() and float(command.min()) >= 0.0
                 and float(command.max()) <= 1.0
                 and torch.equal(teachers[index].reshape(1, 1, 200, 200), command),
                 f"saved teacher target or command binding is invalid at row {index}")
        outputs = {grid: {key: row["outputs"][grid][key]
                          for key in ("coefficients", "raw_cmos")}
                   for grid in GRIDS}
        refs = {grid: row["outputs"][grid]["direct_raw_cmos"] for grid in GRIDS}
        metrics = _numpy_metric_row(outputs, target, refs)
        _require(all(abs(float(metrics[key]) - float(row["metrics"][key])) <= 1.0e-14
                     and float(metrics[key]) <= LIMIT for key in METRIC_KEYS)
                 and float(row["prescribed_phase_coefficient_error"]) <= LIMIT,
                 f"teacher six-metric evidence does not recompute at row {index}")

    run_id = terminal.get("run_id")
    schedule = _schedule(config)
    _require(terminal.get("task_id") == TASK_ID and terminal.get("completed_updates") == UPDATES
             and terminal.get("run_id") == run_id and terminal.get("lineage_mapper_updates") == 25_000
             and terminal.get("adam_step") == UPDATES
             and terminal.get("optimizer_deepcopied") is True
             and torch.equal(terminal.get("schedule"), schedule)
             and terminal.get("schedule_cursor") == UPDATES
             and terminal.get("history_count") == UPDATES
             and len(terminal.get("history", [])) == UPDATES,
             "terminal checkpoint is not the fixed 15,000-step endpoint")
    counts = torch.bincount(schedule.flatten(), minlength=TRAIN_N)
    _require(torch.equal(counts, torch.full((TRAIN_N,), EXPOSURES))
             and torch.equal(terminal.get("counts"), counts)
             and fixed._adam_steps(terminal["optimizer_state"]) == [UPDATES] * 4,
             "terminal exposure counts or Adam steps do not match the fixed training plan")
    init = torch.load(attempt / "mapper_initialization.pt", map_location="cpu", weights_only=False)
    _require(init.get("task_id") == TASK_ID and init.get("lineage_mapper_updates") == 10_000
             and init.get("source_checkpoint") == config["source"]["v2_terminal_checkpoint"]
             and init.get("adam_state") == {}
             and fixed._compare_states(init["mapper_state"], inputs["source_mapper"]["mapper_state"]) == 0.0,
             "scientific attempt did not start from the explicit V2 endpoint with empty Adam")
    loaded_state = torch.load(attempt / "terminal_loaded_mapper_state.pt", map_location="cpu",
                              weights_only=False)
    _require(loaded_state.get("task_id") == TASK_ID and loaded_state.get("run_id") == run_id
             and loaded_state.get("completed_updates") == UPDATES
             and loaded_state.get("adam_steps") == [UPDATES] * 4
             and fixed._compare_states(loaded_state["mapper_state"], terminal["mapper_state"]) == 0.0,
             "independently loaded terminal CUDA model state does not match the saved checkpoint")
    seed_states = torch.load(attempt / "terminal_command_seed" / "loaded_model_states.pt",
                             map_location="cpu", weights_only=False)
    independent_states = torch.load(attempt / "independent_loaded_model_states.pt",
                                    map_location="cpu", weights_only=False)
    expected_states = {"student_fixed_phase_zero": terminal["mapper_state"],
                       "v2_mapper": inputs["source_mapper"]["mapper_state"],
                       "fixed_frame_terminal": fixed_terminal["mapper_state"]}
    _require(seed_states.get("task_id") == TASK_ID and independent_states.get("task_id") == TASK_ID
             and independent_states.get("run_id") == run_id,
             "terminal source mapper binding records are missing")
    for condition, expected in expected_states.items():
        _require(fixed._compare_states(seed_states["states"][condition], expected) == 0.0
                 and fixed._compare_states(independent_states["states"][condition], expected) == 0.0,
                 f"terminal command mapper state differs from its explicit source: {condition}")
    history = terminal["history"]
    exposure_counts = torch.zeros(TRAIN_N, dtype=torch.long)
    for index, row in enumerate(history):
        batch = schedule[index]
        exposure_counts.index_add_(0, batch, torch.ones_like(batch))
        _require(row.get("completed_update") == index + 1
                 and row.get("sample_indices") == batch.tolist()
                 and row.get("labels") == list(range(CLASS_COUNT))
                 and row.get("loss_is_pre_update") is True
                 and row.get("exposure_count_after_update") == exposure_counts[batch].tolist()
                 and all(math.isfinite(float(row[key])) for key in ("loss", "coefficient_loss", "box_loss", "teacher_loss")),
                 f"training history differs from the fixed sample schedule at update {index + 1}")
    logged = {}
    for history_path in attempt.rglob("training_history*.jsonl"):
        for line_no, line in enumerate(history_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            step = int(item.get("completed_update", -1))
            _require(1 <= step <= UPDATES and item.get("sample_indices") == schedule[step - 1].tolist()
                     and item.get("completed_update") == step,
                     f"saved training log has an invalid schedule row: {history_path.name}:{line_no}")
            logged.setdefault(step, set()).add(tuple(item["sample_indices"]))
    _require(all(len(values) == 1 for values in logged.values()),
             "training progress logs contain conflicting sample identities")

    # The three-step checkpoint, direct fourth step, and independent restored fourth step
    # are consumed from their raw states; a summary-only recovery claim is not accepted.
    state3 = torch.load(preflight_dir / "recovery_step_00003.pt", map_location="cpu", weights_only=False)
    direct4 = torch.load(preflight_dir / "recovery_direct_step_00004.pt", map_location="cpu", weights_only=False)
    replay4 = torch.load(preflight_dir / "recovery_replayed_step_00004.pt", map_location="cpu", weights_only=False)
    state_delta = fixed._compare_states(direct4["mapper_state"], replay4["mapper_state"])
    _, adam_delta = fixed.fit20._tree_max_delta(direct4["optimizer_state"], replay4["optimizer_state"])
    _require(state3.get("completed_updates") == 3 and direct4.get("completed_updates") == 4
             and replay4.get("completed_updates") == 4
             and torch.equal(state3["schedule"], schedule)
             and torch.equal(state3["next_batch"], schedule[3])
             and state_delta <= RECOVERY_TOL and adam_delta <= RECOVERY_TOL
             and abs(float(direct4["loss"]) - float(replay4["loss"])) <= RECOVERY_TOL
             and torch.equal(direct4["indices"], schedule[3])
             and torch.equal(replay4["indices"], schedule[3])
             and direct4["history"] == replay4["history"]
             and _numpy_tree_delta(direct4["rng_state"], replay4["rng_state"]) == 0.0,
             "raw restored fourth-step model, Adam, loss, history, RNG, or schedule differs")
    worker_summary = _json(attempt / "measurements" / "worker_summary.json")
    _require(worker_summary.get("run_id") == run_id and worker_summary.get("single_sample_command_replay") is True
             and worker_summary.get("maximum_command_absolute_delta") == 0.0
             and worker_summary.get("measured_rows") == {
                 "train": TRAIN_N, "development": DEV_N,
                 "holdout": {"student_fixed_phase_zero": HOLDOUT_N,
                             "v2_mapper": HOLDOUT_N, "fixed_frame_terminal": HOLDOUT_N}},
             "independent CUDA command replay counts or exact equality failed")

    # CPU replay is a separate numerical reference. It rejects command changes that preserve
    # the measured optical output through a zero-space perturbation.
    cpu_mappers = _terminal_states(config, inputs, terminal, fixed_terminal, torch.device("cpu"))
    max_cpu_command_delta = 0.0
    for role in ("train", "development", "holdout"):
        for seed_path in sorted((attempt / "terminal_command_seed").glob(f"{role}_*.pt")):
            seed = torch.load(seed_path, map_location="cpu", weights_only=False)
            start, stop = int(seed["start"]), int(seed["stop"])
            cohort = population[role]
            for condition, saved_commands in seed["conditions"].items():
                for local, index in enumerate(range(start, stop)):
                    regenerated = _command_for_condition(
                        condition, cpu_mappers[condition], cohort["coefficients"][index],
                        inputs, torch.device("cpu"))[0].cpu()
                    delta = float((regenerated - saved_commands[local]).abs().max())
                    max_cpu_command_delta = max(max_cpu_command_delta, delta)
                    _require(delta <= 1.0e-6,
                             f"CPU command source binding failed; possible zero-space replacement: {condition}/{role}/{index}")

    row_groups, raw_groups, population_counts = _measurement_rows(config, attempt, population)
    summaries = {role: {condition: projected._summarize_rows(rows)
                        for condition, rows in groups.items()}
                 for role, groups in row_groups.items()}
    area = fixed._camera_area(inputs, torch.device("cpu"))
    gain = float(inputs["camera_gain"])
    semantic = {condition: fixed._summary_from_readouts(
        inputs, population["holdout"], pilot._readout_for_evaluator(raw[2048], area, gain))
                for condition, raw in raw_groups.items()}
    semantic_pass, semantic_gate = fixed._semantic_candidate_passes(
        semantic["student_fixed_phase_zero"], semantic["direct_decoder"], config)
    numeric_pass = summaries["holdout"]["student_fixed_phase_zero"]["all_rows_passed"]
    final_pass = bool(numeric_pass and semantic_pass)
    final_state = "VALID_COMPLETED" if final_pass else "VALID_FAILED"
    decision = _json(attempt / "final_decision.json")
    _require(decision.get("task_id") == TASK_ID and decision.get("run_id") == run_id
             and decision.get("state") == final_state
             and decision.get("numeric_gate_passed") is bool(numeric_pass)
             and decision.get("semantic_proxy_gate_passed") is bool(semantic_pass)
             and decision.get("final_gate_passed") is final_pass
             and bool(decision["holdout_numeric_summary"]["all_rows_passed"]) is bool(numeric_pass),
             "saved final decision differs from independently recomputed holdout evidence")
    report = _json(attempt / "terminal_measurement_summary.json")
    _require(report.get("task_id") == TASK_ID and report.get("state") == final_state
             and report.get("training", {}).get("history_rows") == UPDATES
             and report.get("training", {}).get("adam_steps") == [UPDATES] * 4
             and report.get("numeric_summaries") == summaries
             and report.get("semantic_gate") == semantic_gate,
             "saved report differs from independently recomputed numeric and semantic evidence")
    closeout = (attempt / "closeout_report.md").read_text(encoding="utf-8")
    _require(all(header in closeout for header in (
        "## 1．结果与状态", "## 2．科学认识", "## 3．目标缺口",
        "## 4．工程完成与验证", "## 5．决策影响与下一步")),
        "five-part terminal report is incomplete")
    destination = (_roots(config)["verification_dir"] if report_dir is None else report_dir.resolve())
    _require(not destination.exists() and destination.is_relative_to(_roots(config)["runs_root"].resolve()),
             "verification report path must be new and stay within this study runs root", FileExistsError)
    destination.mkdir(parents=True)
    verification = {"task_id": TASK_ID, "status": "SAVED_EVIDENCE_VERIFIED",
                    "evidence_dir": str(root), "transfer_receipt_required": require_transfer,
                    "manifest_file_count": len(manifest_files),
                    "training_updates": UPDATES, "adam_steps": [UPDATES] * 4,
                    "teacher_rows_bound_and_recomputed": TRAIN_N,
                    "holdout_rows": HOLDOUT_N,
                    "holdout_all_six_passed": bool(numeric_pass),
                    "semantic_proxy_gate_passed": bool(semantic_pass),
                    "final_state": final_state,
                    "independent_numpy_metrics": True,
                    "cpu_command_binding_max_abs_delta": max_cpu_command_delta,
                    "recovery_model_delta": state_delta, "recovery_optimizer_delta": adam_delta,
                    "population_counts": population_counts["rows"]}
    _write_json(destination / "verification_summary.json", verification)
    _write_json(destination / "recomputed_final_decision.json", {
        "numeric_summaries": summaries, "semantic_gate": semantic_gate,
        "final_gate_passed": final_pass, "state": final_state})
    return verification


def _measure(config: Mapping[str, Any]) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "terminal measurement requires AutoDL CUDA")
    roots = _roots(config)
    state = _json(roots["attempt_dir"] / "task_state.json")
    _require(state.get("task_id") == TASK_ID and state.get("state") == "TERMINAL_MEASUREMENT_PENDING"
             and state.get("completed_updates") == UPDATES
             and not (roots["attempt_dir"] / "final_decision.json").exists(),
             "measure requires a completed, open terminal checkpoint")
    terminal_path = roots["attempt_dir"] / "checkpoints" / "terminal_checkpoint.pt"
    seed_summary = _create_command_seeds(config, terminal_path)
    measurement_dir = roots["attempt_dir"] / "measurements"
    child = subprocess.run([sys.executable, "-m", __name__, "--mode", "measurement-worker",
                            "--config", str(DEFAULT_CONFIG), "--checkpoint", str(terminal_path),
                            "--output", str(measurement_dir)], cwd=ROOT, capture_output=True,
                           text=True, check=False)
    _require(child.returncode == 0 and (measurement_dir / "worker_summary.json").is_file(),
             "independent terminal measurement worker failed: " + child.stderr[-3000:])
    decision = _finalize_measurement(config, roots["attempt_dir"])
    return {"task_id": TASK_ID, "state": decision["state"], "seed_counts": seed_summary["counts"],
            "final_gate_passed": decision["final_gate_passed"],
            "final_decision": str(roots["attempt_dir"] / "final_decision.json")}


def _cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=(
        "prepare", "qualify-teachers", "preflight", "execute", "resume", "measure",
        "verify-saved", "recovery-worker", "measurement-worker"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _cli().parse_args()
    config_path = args.config.resolve()
    config = _config(config_path)
    if args.mode == "prepare":
        _require(args.checkpoint is None, "prepare does not accept a checkpoint", ValueError)
        result = _prepare(config, resume=None)
    elif args.mode == "qualify-teachers":
        result = _teacher(config, resume=None if args.checkpoint is None else _path(args.checkpoint).resolve())
    elif args.mode == "preflight":
        _require(args.checkpoint is None, "preflight does not accept a checkpoint", ValueError)
        result = _preflight(config)
    elif args.mode == "execute":
        _require(args.checkpoint is None, "execute starts only from the fixed zero-update initialization", ValueError)
        result = _train(config)
    elif args.mode == "resume":
        _require(args.checkpoint is not None, "resume requires an explicit checkpoint", ValueError)
        result = _train(config, checkpoint_path=_path(args.checkpoint).resolve())
    elif args.mode == "measure":
        _require(args.checkpoint is None, "measure consumes the registered terminal checkpoint", ValueError)
        result = _measure(config)
    elif args.mode == "verify-saved":
        _require(args.evidence_dir is not None,
                 "verify-saved requires the explicit task runs root or Windows backup", ValueError)
        result = _verify_saved(config, _path(args.evidence_dir).resolve(),
                               None if args.report_dir is None else _path(args.report_dir).resolve())
    elif args.mode == "recovery-worker":
        _require(args.checkpoint is not None and args.evidence_dir is not None and args.output is not None,
                 "recovery-worker requires an explicit checkpoint, evidence directory, and output", ValueError)
        _recovery_worker(config, _path(args.checkpoint).resolve(), _path(args.evidence_dir).resolve(),
                         _path(args.output).resolve())
        result = {"task_id": TASK_ID, "status": "RECOVERY_STEP4_SAVED",
                  "output": str(_path(args.output).resolve())}
    else:
        _require(args.checkpoint is not None and args.output is not None,
                 "measurement-worker requires an explicit terminal checkpoint and output", ValueError)
        result = _measurement_worker(config, _path(args.checkpoint).resolve(), _path(args.output).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
