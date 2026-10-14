"""Run the fixed three-seed cloud DEV comparison for Generator foreground training."""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import math
import os
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import audit_lightgen_2d_generator_cross_port_joint_structure_v1 as frozen
from scripts.research.lightgen import audit_lightgen_2d_generator_frozen_joint_prior_comparison_v1 as prior_reference
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import diagnose_lightgen_2d_generator_foreground_objective_gradient_v1 as gradient
from scripts.research.lightgen import execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as artifact_source
from scripts.research.lightgen import execute_lightgen_2d_generator_foreground_objective_training_comparison_v1 as historical_training
from scripts.research.lightgen import execute_lightgen_2d_generator_frozen_conditional_ols_generation_v1 as generation
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity
from scripts.research.lightgen import execute_lightgen_2d_generator_semantic_target_comparison_v1 as semantic
from scripts.research.lightgen import execute_lightgen_2d_generator_stroke_fidelity_comparison_v1 as stroke


TASK_ID = "M0_GENERATOR_CLOUD_FOREGROUND_DEV_MULTISEED_V1"
ENGINEERING_HISTORICAL_UPDATES = 3
ENGINEERING_NEW_UPDATE_LIMIT = 42
ENGINEERING_CUMULATIVE_UPDATE_LIMIT = 45
SOURCE_STEP = 4_700
FIT_ROWS = 2_000
FIT_PER_CLASS = 200
DEV_ROWS = 1_000
DEV_PER_CLASS = 100
BATCH_SIZE = 10
UPDATES_PER_ARM = 2_000
EPOCHS = 10
BATCHES_PER_EPOCH = FIT_ROWS // BATCH_SIZE
TOTAL_UPDATES = 3 * 2 * UPDATES_PER_ARM
TRAIN_SEEDS = (2026101201, 2026101202, 2026101203)
SAMPLE_SEEDS = (2026101221, 2026101222, 2026101223, 2026101224, 2026101225)
SAMPLES_PER_CLASS = 30
FOCUS_CLASSES = (2, 4)
FIXED_LAMBDA = 0.11159457624315472
SELECTION_SEED = 2026101211
DEV_SELECTION_SEED = 2026101212
SPLIT_SEED = 1234
VALIDATION_PER_CLASS = 500
ANCESTOR_ROWS = 3_300
OLD_COMPARISON_ROWS = 500
NEW_ROWS_PER_CLASS = 180
REPLAY_RTOL = 2.0e-6
REPLAY_ATOL = 5.0e-8
GRADIENT_RTOL = 2.0e-5
GRADIENT_ATOL = 2.0e-7
DECISION_ATOL = 2.0e-7
DECISION_RTOL = 2.0e-5
CHUNK_ROWS = 50
NUMERIC_MODE = "strict_fp32_optics_float64_qfg_statistics"
CHECKPOINT_SCHEMA = "lightgen_cloud_foreground_dev_multiseed_checkpoint_v1"
CHECKPOINT_INTERVAL = 500
CHECKPOINT_KEEP_RECENT = 2
DEFAULT_OUTPUT = Path("outputs/20261012_m0_generator_cloud_foreground_dev_multiseed_v1/runs/attempt_01")
DEFAULT_ENGINEERING_OUTPUT = Path("outputs/20261012_autodl_training_workflow_improvement_v1/runs/attempt_01")
EVALUATOR_NAMES = ("E_S", "E_I", "E_C", "E_D")
TRAIN_EVALUATOR_NAMES = ("E_S", "E_I", "E_C")
STRICT_FP32_SETTINGS = {
    "dtype": "float32",
    "autocast": False,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "float32_matmul_precision": "highest",
}
DEV_FIELDS = (
    "scenario", "position", "source_row_index", "label", "role", "q_fg", "q_bg", "q_raw200",
    "raw_cmos_integral", "clipped_fraction", "e_s_prediction", "e_i_prediction", "e_c_prediction", "e_d_prediction",
)
SAMPLE_FIELDS = (
    "scenario", "sample_seed", "draw_index", "requested_class", "e_s_prediction", "e_i_prediction",
    "e_c_prediction", "e_d_prediction", "raw_cmos_integral", "clipped_fraction", "generator_input_power",
)


class CloudForegroundError(RuntimeError):
    """An invalid source, preflight, or saved measurement blocks a scientific result."""


class TrainingInterrupted(RuntimeError):
    """An interrupt was recorded at a durable checkpoint boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CloudForegroundError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _portable_resolve(root: Path, value: str | Path) -> Path:
    """Resolve repository paths recorded on Windows when this run is on Linux."""
    raw = str(value)
    windows = PureWindowsPath(raw)
    if windows.is_absolute():
        parts = windows.parts
        _require(len(parts) >= 3 and parts[1].casefold() == "codex-original", f"declared Windows path is outside codex-original: {raw}")
        candidate = root.joinpath(*parts[2:]).resolve()
    else:
        candidate = Path(raw)
        if candidate.is_absolute():
            candidate = candidate.resolve()
        else:
            candidate = (root / candidate).resolve()
    _require(candidate == root.resolve() or root.resolve() in candidate.parents, f"resolved path escapes the repository: {raw}")
    return candidate


@contextlib.contextmanager
def _legacy_path_translation(root: Path) -> Iterator[None]:
    """Temporarily translate only legacy helper path resolvers; never patch torch.load."""
    saved: list[tuple[Any, Any]] = []
    for module in tuple(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("scripts.research.lightgen."):
            continue
        resolver = getattr(module, "_resolve", None)
        if callable(resolver):
            saved.append((module, resolver))
            module._resolve = lambda _base, value, _root=root: _portable_resolve(_root, value)
    try:
        yield
    finally:
        for module, resolver in saved:
            module._resolve = resolver


def _json(path: Path) -> Any:
    _require(path.is_file(), f"required JSON evidence is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _input_file_inventory(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    unique = {path.resolve().as_posix(): path.resolve() for path in paths}
    inventory = [
        {"path": path.as_posix(), "exists": path.is_file(), "size_bytes": path.stat().st_size if path.is_file() else None}
        for path in unique.values()
    ]
    return inventory, [row["path"] for row in inventory if not row["exists"]]


def _write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        _require(_json(path) == value, f"refusing to overwrite different evidence: {path.name}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


@contextlib.contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.name == "nt":
            import msvcrt

            stream.seek(0, os.SEEK_END)
            if stream.tell() == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _engineering_budget_file() -> Path:
    return (_root() / DEFAULT_ENGINEERING_OUTPUT / "engineering_update_budget.json").resolve()


def _read_engineering_budget(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema": "autodl_engineering_adam_budget_v1",
            "limit": ENGINEERING_CUMULATIVE_UPDATE_LIMIT,
            "budget_changes": [],
            "reservations": [],
        }
    value = _json(path)
    _require(
        value.get("schema") == "autodl_engineering_adam_budget_v1"
        and value.get("limit") in {42, ENGINEERING_CUMULATIVE_UPDATE_LIMIT},
        "engineering update budget identity or limit differs",
    )
    reservations = value.get("reservations")
    _require(
        isinstance(reservations, list) and len(reservations) <= value["limit"],
        "engineering update budget record is malformed or exceeds its limit",
    )
    _require(all(row.get("index") == index + 1 for index, row in enumerate(reservations)), "engineering update budget reservation sequence is incomplete")
    return value


def _initialize_engineering_budget(path: Path) -> dict[str, Any]:
    with _exclusive_file_lock(path.with_suffix(path.suffix + ".lock")):
        value = _read_engineering_budget(path)
        if not path.is_file():
            _write_json(path, value)
        return value


def _extend_engineering_budget(path: Path) -> dict[str, Any]:
    """Apply the approved 42-to-45 cumulative engineering budget once."""
    with _exclusive_file_lock(path.with_suffix(path.suffix + ".lock")):
        value = _read_engineering_budget(path)
        if value["limit"] == 42:
            reservations = value["reservations"]
            _require(
                len(reservations) == ENGINEERING_HISTORICAL_UPDATES
                and all(row.get("status") == "optimizer_step_completed" for row in reservations),
                "approved budget extension requires the three recorded historical updates",
            )
            for row in reservations:
                row["origin"] = "historical_before_budget_extension"
            value["budget_changes"] = [
                {
                    "previous_limit": 42,
                    "new_limit": ENGINEERING_CUMULATIVE_UPDATE_LIMIT,
                    "historical_reservations": ENGINEERING_HISTORICAL_UPDATES,
                    "new_diagnostic_allowance": ENGINEERING_NEW_UPDATE_LIMIT,
                    "authorization": "user-approved engineering closeout plan",
                }
            ]
            value["limit"] = ENGINEERING_CUMULATIVE_UPDATE_LIMIT
            _write_json(path, value, replace=True)
        else:
            changes = value.get("budget_changes", [])
            _require(
                isinstance(changes, list)
                and any(
                    row.get("previous_limit") == 42
                    and row.get("new_limit") == ENGINEERING_CUMULATIVE_UPDATE_LIMIT
                    and row.get("new_diagnostic_allowance") == ENGINEERING_NEW_UPDATE_LIMIT
                    for row in changes
                ),
                "cumulative budget has no recorded approved 42-to-45 extension",
            )
        return value


def _reserve_engineering_update(inputs: Mapping[str, Any]) -> int | None:
    raw_path = inputs.get("engineering_budget_file")
    if raw_path is None:
        return None
    path = Path(raw_path)
    context = dict(inputs.get("engineering_update_context", {}))
    attempt_value = inputs.get("engineering_attempt_dir")
    _require(attempt_value is not None, "engineering updates require an explicit attempt directory")
    attempt_path = Path(attempt_value).resolve()
    repository_root = _root().resolve()
    _require(repository_root in attempt_path.parents, "engineering attempt directory must be inside the repository")
    context["engineering_attempt_dir"] = attempt_path.relative_to(repository_root).as_posix()
    with _exclusive_file_lock(path.with_suffix(path.suffix + ".lock")):
        value = _read_engineering_budget(path)
        reservations = value["reservations"]
        historical = sum(row.get("origin") == "historical_before_budget_extension" for row in reservations)
        _require(value.get("limit") == ENGINEERING_CUMULATIVE_UPDATE_LIMIT and historical == ENGINEERING_HISTORICAL_UPDATES, "engineering budget extension and its three historical reservations must be recorded before an update")
        new_reserved = len(reservations) - historical
        _require(
            new_reserved < ENGINEERING_NEW_UPDATE_LIMIT and len(reservations) < ENGINEERING_CUMULATIVE_UPDATE_LIMIT,
            "task-wide engineering Adam update budget is exhausted; no additional optimizer step was started",
        )
        reservation = {
            "index": len(reservations) + 1,
            "origin": "current_engineering_diagnostic",
            "status": "reserved_before_optimizer_step",
            "context": context,
        }
        reservations.append(reservation)
        _write_json(path, value, replace=True)
        reservation_index = int(reservation["index"])
    _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", str(context.get("phase", "unknown")), value)
    return reservation_index


def _complete_engineering_update(inputs: Mapping[str, Any], reservation_index: int | None) -> None:
    raw_path = inputs.get("engineering_budget_file")
    if raw_path is None or reservation_index is None:
        return
    path = Path(raw_path)
    with _exclusive_file_lock(path.with_suffix(path.suffix + ".lock")):
        value = _read_engineering_budget(path)
        reservation = value["reservations"][reservation_index - 1]
        _require(reservation.get("index") == reservation_index and reservation.get("status") == "reserved_before_optimizer_step", "engineering update reservation cannot be completed")
        reservation["status"] = "optimizer_step_completed"
        _write_json(path, value, replace=True)
    context = dict(inputs.get("engineering_update_context", {}))
    _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", str(context.get("phase", "unknown")), value)


def _write_engineering_stage(target: Path, phase: str, value: Mapping[str, Any]) -> Path:
    path = target / "engineering_stages" / f"{phase}.json"
    evidence_identity = {
        "input_population": "input_population.json",
        "batch_orders": "batch_orders.json",
        "run_config": "engineering_run_config.json",
    }
    payload = {"task_id": TASK_ID, "phase": phase, "evidence_identity": evidence_identity, **dict(value)}
    _write_json(path, payload)
    _require(_json(path) == payload, f"completed engineering phase failed readback: {phase}")
    return path


def _engineering_stage_path(target: Path, phase: str) -> Path:
    return target / "engineering_stages" / f"{phase}.json"


def _read_engineering_stage(target: Path, phase: str) -> dict[str, Any] | None:
    path = _engineering_stage_path(target, phase)
    if not path.is_file():
        return None
    payload = _json(path)
    _require(
        payload.get("task_id") == TASK_ID
        and payload.get("phase") == phase
        and payload.get("status") == "COMPLETE",
        f"engineering stage is incomplete or belongs to another task: {phase}",
    )
    _require(
        payload.get("evidence_identity")
        == {
            "input_population": "input_population.json",
            "batch_orders": "batch_orders.json",
            "run_config": "engineering_run_config.json",
        },
        f"engineering stage evidence identity is incomplete: {phase}",
    )
    return payload


def _write_engineering_progress(
    inputs: Mapping[str, Any],
    status: str,
    phase: str,
    budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    budget_path_value = inputs.get("engineering_budget_file")
    if budget is None and budget_path_value is not None:
        budget = _read_engineering_budget(Path(budget_path_value))
    reservations = list((budget or {}).get("reservations", []))
    completed = sum(row.get("status") == "optimizer_step_completed" for row in reservations)
    historical = sum(row.get("origin") == "historical_before_budget_extension" for row in reservations)
    new_reserved = len(reservations) - historical
    new_completed = sum(
        row.get("origin") != "historical_before_budget_extension" and row.get("status") == "optimizer_step_completed"
        for row in reservations
    )
    attempt_value = inputs.get("engineering_attempt_dir")
    attempt = Path(attempt_value) if attempt_value is not None else None
    reusable_stage = None
    if attempt is not None:
        stage = attempt / "engineering_stages" / "production_preflight.json"
        preflight = attempt / "preflight.json"
        if stage.is_file() and preflight.is_file():
            try:
                stage_value = _json(stage)
                preflight_value = _json(preflight)
                checks = preflight_value.get("checks", {})
                if (
                    stage_value.get("task_id") == TASK_ID
                    and stage_value.get("status") == "COMPLETE"
                    and preflight_value.get("task_id") == TASK_ID
                    and preflight_value.get("status") == "READY"
                    and checks
                    and all(row.get("executed") is True and row.get("passed") is True for row in checks.values())
                ):
                    reusable_stage = stage.relative_to(attempt).as_posix()
            except (OSError, ValueError, json.JSONDecodeError):
                reusable_stage = None
    record = {
        "task_id": TASK_ID,
        "status": status,
        "phase": phase,
        **dict(inputs.get("engineering_update_context", {})),
        "engineering_updates_reserved": len(reservations),
        "engineering_updates_completed": completed,
        "historical_engineering_updates": historical,
        "new_engineering_updates_reserved": new_reserved,
        "new_engineering_updates_completed": new_completed,
        "engineering_update_limit": ENGINEERING_CUMULATIVE_UPDATE_LIMIT,
        "engineering_new_update_limit": ENGINEERING_NEW_UPDATE_LIMIT,
        "scientific_updates": 0,
        "effective_recovery_path": reusable_stage,
    }
    progress_path = inputs.get("engineering_progress_file")
    if progress_path is not None:
        path = Path(progress_path)
        _write_json(path, record, replace=path.exists())
    if attempt is not None:
        path = attempt / "progress.json"
        _write_json(path, record, replace=path.exists())
    return record


def _validate_engineering_resume_attempt(
    target: Path,
    output_root: Path,
    evidence_dir: Path,
    inputs: Mapping[str, Any],
    source_args: argparse.Namespace,
    budget_path: Path,
) -> dict[str, Any]:
    _require(target.is_dir() and target.resolve().parent == output_root.resolve(), "engineering resume must name one explicit attempt inside the task output")
    _require(target.name.startswith("engineering_attempt_"), "engineering resume path is not a registered diagnostic attempt")
    _require((target / "plan.md").is_file() and (target / "input_probe.json").is_file(), "engineering resume attempt lacks its input probe")
    probe = _json(target / "input_probe.json")
    _require(Path(probe.get("evidence_dir", "")).resolve() == evidence_dir.resolve(), "engineering resume evidence directory differs from the completed attempt")
    _require(_json(target / "input_population.json") == inputs["population"], "engineering resume population differs from the completed attempt")
    expected_orders = {str(seed): inputs["batch_orders"][seed] for seed in TRAIN_SEEDS}
    _require(_json(target / "batch_orders.json") == expected_orders, "engineering resume batch orders differ from the completed attempt")
    expected_config = {
        "task_id": TASK_ID,
        "source_checkpoint": inputs["population"]["source_checkpoint"],
        "source_checkpoint_step": SOURCE_STEP,
        "source_adam_step": SOURCE_STEP,
        "requested_microbatch_size": int(source_args.microbatch_size),
        "trajectory_microbatch_sizes": [1, 2, 5, 10, 1, 1],
        "logical_batch_size": BATCH_SIZE,
        "propagation_grid": 2_048,
        "optical_dtype": "float32",
        "qfg_dtype": "float64",
        "fixed_lambda": FIXED_LAMBDA,
        "generator_only_adam": True,
    }
    _require(_json(target / "engineering_run_config.json") == expected_config, "engineering resume execution configuration differs")
    stage_files = sorted((target / "engineering_stages").glob("*.json")) if (target / "engineering_stages").is_dir() else []
    stages: dict[str, dict[str, Any]] = {}
    for stage_file in stage_files:
        phase = stage_file.stem
        stage = _read_engineering_stage(target, phase)
        _require(stage is not None, f"engineering stage could not be loaded: {phase}")
        stages[phase] = stage
        checkpoint_relative = stage.get("checkpoint")
        checkpoint_root_relative = stage.get("checkpoint_root")
        if checkpoint_relative is not None:
            _require(checkpoint_root_relative is not None, f"engineering stage is missing its checkpoint root: {phase}")
            checkpoint_path = (target / checkpoint_relative).resolve()
            _require(target.resolve() in checkpoint_path.parents, f"engineering stage checkpoint escapes its attempt: {phase}")
            if checkpoint_path.is_file():
                checkpoint_root = (target / checkpoint_root_relative).resolve()
                _require(target.resolve() in checkpoint_root.parents, f"engineering checkpoint root escapes its attempt: {phase}")
                payload = _read_checkpoint_file(checkpoint_path, checkpoint_root, inputs, inputs["batch_orders"])
                _require(
                    payload.get("task_local_updates") == stage.get("task_local_update", stage.get("updates"))
                    and payload.get("optimizer_step") == SOURCE_STEP + payload.get("task_local_updates", -1),
                    f"engineering stage checkpoint cursor differs: {phase}",
                )
            else:
                _require(
                    stage.get("checkpoint_retention") == "keep_recent_two_after_readback"
                    and int(stage.get("task_local_update", 0)) < 3,
                    f"engineering stage checkpoint is missing: {phase}",
                )
        if phase.startswith("file_recovery_"):
            _require((target / "preflight" / "repeatability_envelope.json").is_file(), "file recovery stage lacks its fixed repeatability envelope")
            trial_index = int(phase.rsplit("_", 1)[1])
            expected_signal = (None, "SIGINT", "SIGTERM")[trial_index - 1] if 1 <= trial_index <= 3 else "unsupported"
            comparison_path = (target / str(stage.get("comparison", ""))).resolve()
            _require(target.resolve() in comparison_path.parents and comparison_path.is_file(), f"file recovery comparison is missing: {phase}")
            comparison = _json(comparison_path)
            _require(comparison.get("status") == "PASS" and comparison.get("signal_kind") == expected_signal, f"file recovery comparison is invalid: {phase}")
            recovery_root = comparison_path.parent
            boundary = _read_checkpoint_file(target / comparison["checkpoint"], recovery_root, inputs, inputs["batch_orders"])
            terminal = _read_checkpoint_file(target / comparison["terminal_checkpoint"], recovery_root, inputs, inputs["batch_orders"])
            _require(boundary.get("task_local_updates") == 1 and terminal.get("task_local_updates") == 3, f"file recovery checkpoint boundary is invalid: {phase}")
            envelope = _json(target / "preflight" / "repeatability_envelope.json")
            continuous_states = [
                torch.load(target / "preflight" / "continuous_repeats" / f"repeat_{index:02d}.pt", map_location="cpu", weights_only=True)
                for index in range(1, 6)
            ]
            phase_values = [row["generator_phase"] for row in continuous_states]
            optimizer_values = [row["optimizer_state"] for row in continuous_states]
            _require(_continuous_tensor_envelope(phase_values) == envelope.get("phase"), f"saved continuous phase envelope differs on disk: {phase}")
            _require(_continuous_adam_envelope(optimizer_values) == envelope.get("adam"), f"saved continuous Adam envelope differs on disk: {phase}")
            phase_comparison = _tensor_repeatability_envelope(phase_values, terminal["trainable_parameter_state"]["generator.phase"], fixed_envelope=envelope["phase"])
            optimizer_comparison = _adam_repeatability_envelope(optimizer_values, terminal["optimizer_state"], fixed_envelope=envelope["adam"])
            _require(phase_comparison.get("passed") is True and optimizer_comparison.get("passed") is True, f"saved recovery endpoint no longer meets its fixed same-device envelope: {phase}")
            _require(capacity._tree_equal(phase_comparison, comparison.get("phase_comparison")) and capacity._tree_equal(optimizer_comparison, comparison.get("adam_comparison")), f"saved recovery comparison values do not recheck: {phase}")
            _require(capacity._tree_equal(boundary["training_history"], terminal["training_history"][:1]), f"saved recovery history prefix differs: {phase}")
            _require(capacity._tree_equal(terminal["rng_state"], continuous_states[0]["rng_state"]), f"saved recovery terminal RNG differs: {phase}")
            _require(terminal["batch_order"] == [list(batch) for batch in inputs["batch_orders"][terminal["seed"]]], f"saved recovery batch order differs: {phase}")
            save_result = _json(recovery_root / "save_worker_result.json")
            resume_result = _json(recovery_root / "recovery_worker_result.json")
            save_exit = _json(recovery_root / "save_process_exit.json")
            resume_exit = _json(recovery_root / "resume_process_exit.json")
            _require(save_result.get("signal_kind") == expected_signal and save_result.get("signal_save_verified") is (expected_signal is not None), f"file recovery signal record is invalid: {phase}")
            _require(save_exit.get("process_exited_before_resume") is True and save_exit.get("process_id") == comparison.get("save_process_id"), f"file recovery save process was not confirmed exited: {phase}")
            _require(resume_exit.get("process_exited") is True and resume_exit.get("process_id") == comparison.get("resume_process_id") and resume_exit.get("return_code") in {0, None}, f"file recovery resume process exit was not confirmed: {phase}")
            _require(resume_result.get("process_id") == comparison.get("resume_process_id") and resume_result.get("process_id") != save_result.get("process_id"), f"file recovery resume process identity is invalid: {phase}")
    for phase in stages:
        if phase.startswith(("trajectory_", "zero_lambda_", "continuous_repeat_", "file_recovery_")):
            _require("microbatch_performance" in stages, f"engineering stage is missing its microbatch dependency: {phase}")
    input_stage = stages.get("input_validation")
    if input_stage is not None:
        _require(input_stage.get("evidence_dir") == evidence_dir.as_posix() and input_stage.get("source_checkpoint") == inputs["population"]["source_checkpoint"], "engineering input-validation stage identity differs")
    performance_stage = stages.get("microbatch_performance")
    if performance_stage is not None:
        _require(performance_stage.get("before_real_updates") is True and performance_stage.get("passed") is True, "engineering pre-update performance stage is incomplete")
    preflight_path = target / "preflight.json"
    if not preflight_path.is_file():
        return {"status": "PARTIAL", "completed_stages": sorted(stages)}
    preflight = _json(preflight_path)
    if preflight.get("status") != "READY":
        return {"status": "PARTIAL", "completed_stages": sorted(stages)}
    checks = preflight.get("checks", {})
    _require(preflight.get("task_id") == TASK_ID, "engineering resume preflight belongs to another task")
    required_checks = {
        "source_model_and_optimizer", "cloud_local_replay", "qfg_independent_value_gradient", "fixed_prior_and_sample_bank",
        "merged_production_gradient_and_speed", "six_seed_arm_three_step_updates", "explicit_checkpoint_recovery", "total_engineering_adam_updates",
    }
    _require(required_checks <= set(checks) and all(row.get("executed") is True and row.get("passed") is True for row in checks.values()), "engineering resume found an incomplete or failed preflight check")
    budget = _read_engineering_budget(budget_path)
    reservations = budget["reservations"]
    completed = sum(row.get("status") == "optimizer_step_completed" for row in reservations)
    historical = sum(row.get("origin") == "historical_before_budget_extension" for row in reservations)
    new_reserved = len(reservations) - historical
    new_completed = sum(row.get("origin") != "historical_before_budget_extension" and row.get("status") == "optimizer_step_completed" for row in reservations)
    _require(
        len(reservations) <= ENGINEERING_CUMULATIVE_UPDATE_LIMIT
        and budget.get("limit") == ENGINEERING_CUMULATIVE_UPDATE_LIMIT
        and historical == ENGINEERING_HISTORICAL_UPDATES
        and new_reserved == ENGINEERING_NEW_UPDATE_LIMIT
        and new_completed == ENGINEERING_NEW_UPDATE_LIMIT
        and completed == ENGINEERING_CUMULATIVE_UPDATE_LIMIT
        and checks["total_engineering_adam_updates"].get("count") == ENGINEERING_NEW_UPDATE_LIMIT,
        "engineering resume budget evidence is incomplete or inconsistent",
    )
    _require(_json(target / "engineering_stages" / "production_preflight.json").get("status") == "COMPLETE", "READY preflight stage is missing")
    _require(int(checks["merged_production_gradient_and_speed"].get("selected_microbatch_size", -1)) == int(source_args.microbatch_size), "engineering resume microbatch differs from its READY preflight")
    _require((target / "microbatch_performance_pre_updates.csv").is_file() and (target / "microbatch_performance.csv").is_file(), "engineering resume performance evidence is incomplete")
    _require((target / "ols_fit_coefficients.pt").is_file() and (target / "conditional_sample_bank.pt").is_file() and (target / "fitted_joint_prior.pt").is_file(), "engineering resume fitted-prior evidence is incomplete")
    _require((target / "preflight" / "repeatability_envelope.json").is_file(), "engineering resume repeatability envelope is missing")
    required_stages = {"input_validation", "microbatch_performance", "production_preflight"}
    required_stages.update(
        _engineering_update_stage_name("trajectory", f"seed_{seed}_arm_{arm.lower()}", local)
        for seed in TRAIN_SEEDS for arm in ("A", "B") for local in (1, 2, 3)
    )
    required_stages.update(
        _engineering_update_stage_name("zero_lambda", f"seed_{seed}_arm_b", 1)
        for seed in TRAIN_SEEDS
    )
    required_stages.update(f"trajectory_seed_{seed}_arm_{arm.lower()}" for seed in TRAIN_SEEDS for arm in ("A", "B"))
    required_stages.update(f"zero_lambda_seed_{seed}" for seed in TRAIN_SEEDS)
    for repeat_number in range(2, 6):
        repeat_group = f"continuous_repeat_{repeat_number:02d}"
        repeat_key = f"seed_{TRAIN_SEEDS[0]}_arm_a"
        required_stages.update(_engineering_update_stage_name(repeat_group, repeat_key, local) for local in (1, 2, 3))
        required_stages.add(repeat_group)
    required_stages.update(f"file_recovery_{trial:02d}" for trial in range(1, 4))
    missing_stages = required_stages - set(stages)
    if missing_stages:
        return {"status": "PARTIAL", "completed_stages": sorted(stages), "missing_stages": sorted(missing_stages)}
    return preflight


def _record_engineering_failure(root: Path, args: argparse.Namespace, error: Exception) -> None:
    output_root = (root / DEFAULT_ENGINEERING_OUTPUT).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    budget_path = output_root / "engineering_update_budget.json"
    try:
        budget = _read_engineering_budget(budget_path)
        reserved = len(budget["reservations"])
        completed = sum(row.get("status") == "optimizer_step_completed" for row in budget["reservations"])
        historical = sum(row.get("origin") == "historical_before_budget_extension" for row in budget["reservations"])
        if budget.get("limit") == 42 and historical == 0 and reserved == ENGINEERING_HISTORICAL_UPDATES and completed == ENGINEERING_HISTORICAL_UPDATES:
            historical = ENGINEERING_HISTORICAL_UPDATES
            new_reserved = new_completed = 0
        else:
            new_reserved = reserved - historical
            new_completed = sum(row.get("origin") != "historical_before_budget_extension" and row.get("status") == "optimizer_step_completed" for row in budget["reservations"])
    except Exception:
        reserved = completed = historical = new_reserved = new_completed = None
    attempt = None
    attempt_value = getattr(args, "engineering_attempt_dir", None) or getattr(args, "engineering_resume_evidence_dir", None)
    if attempt_value is not None:
        candidate = _portable_resolve(root, attempt_value)
        if candidate.is_dir() and output_root in candidate.parents:
            attempt = candidate
    phase = "engineering_preflight"
    effective_recovery_path = None
    if attempt is not None:
        progress_path = attempt / "progress.json"
        if progress_path.is_file():
            try:
                phase = str(_json(progress_path).get("phase", phase))
            except Exception:
                pass
        record = {
            "task_id": TASK_ID,
            "status": "INVALID_HARNESS",
            "phase": phase,
            "failure_type": type(error).__name__,
            "failure": str(error),
            "engineering_updates_reserved": reserved,
            "engineering_updates_completed": completed,
            "historical_engineering_updates": historical,
            "new_engineering_updates_reserved": new_reserved,
            "new_engineering_updates_completed": new_completed,
            "engineering_update_limit": ENGINEERING_CUMULATIVE_UPDATE_LIMIT,
            "engineering_new_update_limit": ENGINEERING_NEW_UPDATE_LIMIT,
            "scientific_updates": 0,
            "effective_recovery_path": None,
        }
        reusable_stage = attempt / "engineering_stages" / "production_preflight.json"
        preflight_path = attempt / "preflight.json"
        if reusable_stage.is_file() and preflight_path.is_file():
            try:
                stage_value = _json(reusable_stage)
                preflight_value = _json(preflight_path)
                checks = preflight_value.get("checks", {})
                if (
                    stage_value.get("task_id") == TASK_ID
                    and stage_value.get("status") == "COMPLETE"
                    and preflight_value.get("task_id") == TASK_ID
                    and preflight_value.get("status") == "READY"
                    and checks
                    and all(row.get("executed") is True and row.get("passed") is True for row in checks.values())
                ):
                    record["effective_recovery_path"] = reusable_stage.relative_to(attempt).as_posix()
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        _write_json(attempt / "progress.json", record, replace=True)
        invalid_path = attempt / "invalid_harness.json"
        if invalid_path.is_file() and _json(invalid_path) != record:
            failure_index = 1
            while (attempt / f"engineering_failure_{failure_index:02d}.json").exists():
                failure_index += 1
            os.replace(invalid_path, attempt / f"engineering_failure_{failure_index:02d}.json")
        _write_json(invalid_path, record, replace=invalid_path.exists())
        effective_recovery_path = record.get("effective_recovery_path")
    root_record = {
        "task_id": TASK_ID,
        "status": "INVALID_HARNESS",
        "phase": phase,
        "failure_type": type(error).__name__,
        "failure": str(error),
        "engineering_updates_reserved": reserved,
        "engineering_updates_completed": completed,
        "historical_engineering_updates": historical,
        "new_engineering_updates_reserved": new_reserved,
        "new_engineering_updates_completed": new_completed,
        "engineering_update_limit": ENGINEERING_CUMULATIVE_UPDATE_LIMIT,
        "engineering_new_update_limit": ENGINEERING_NEW_UPDATE_LIMIT,
        "scientific_updates": 0,
        "effective_recovery_path": effective_recovery_path,
        "attempt_dir": attempt.relative_to(output_root).as_posix() if attempt is not None else None,
    }
    _write_json(output_root / "progress.json", root_record, replace=(output_root / "progress.json").exists())


def _write_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite saved tensor evidence: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_torch_exact(path: Path, value: Any) -> None:
    if path.is_file():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        _require(capacity._tree_equal(existing, value), f"existing tensor evidence differs and will be preserved: {path}")
        return
    _write_torch(path, value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            existing = list(csv.DictReader(stream))
        _require(len(existing) == len(rows), f"refusing to overwrite different CSV evidence: {path.name}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path, required_fields: Sequence[str] = ()) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        _require(set(required_fields) <= set(reader.fieldnames or ()), f"saved CSV is missing required fields: {path}")
        return list(reader)


def _load_sources(root: Path, artifact_dir: Path, checkpoint: Path, gradient_dir: Path, reference_dir: Path) -> dict[str, Any]:
    _require(artifact_dir.is_dir() and gradient_dir.is_dir() and reference_dir.is_dir(), "an explicitly named source directory is missing")
    _require(checkpoint == (artifact_dir / "arm_a_step4700.pt").resolve(), "checkpoint must be the A4700 copy in the named energy-comparison attempt")
    with _legacy_path_translation(root):
        source = frozen._load_inputs(root, artifact_dir, checkpoint)

    grad_report = _json(gradient_dir / "report.json")
    grad_population = _json(gradient_dir / "input_population.json")
    calibration = _json(gradient_dir / "fit_calibration.json")
    grad_preflight = _json(gradient_dir / "preflight.json")
    fit_gradients = gradient_dir / "fit_gradients.pt"
    _require(grad_report.get("task_id") == gradient.TASK_ID and grad_report.get("status") == "VALID_COMPLETED", "fixed-weight gradient audit is not a valid completed measurement")
    _require(grad_preflight.get("task_id") == gradient.TASK_ID and grad_preflight.get("status") == "READY", "fixed-weight gradient audit lacks READY preflight evidence")
    _require(calibration.get("lambda") == FIXED_LAMBDA and grad_report.get("measurement", {}).get("lambda") == FIXED_LAMBDA, "the audited foreground weight differs from the prespecified value")
    _require(calibration.get("fit_rows") == 200 and calibration.get("confirmation_rows_used_for_calibration") == 0, "foreground weight was not fixed from the original 200 fit rows")
    _require(fit_gradients.is_file() and torch.load(fit_gradients, map_location="cpu", weights_only=True).get("lambda") == FIXED_LAMBDA, "fixed-weight gradient tensor evidence is missing or differs")

    old_report = _json(reference_dir / "report.json")
    old_population = _json(reference_dir / "input_population.json")
    _require(old_report.get("task_id") == historical_training.TASK_ID and old_report.get("status") == "VALID_COMPLETED", "historical A5400/B5400 comparison is not a valid completed attempt")
    _require(old_population.get("source_row_indices") == source["population"].get("source_row_indices"), "historical 500-row comparison does not share the named A4700 population")
    old_checkpoints = {}
    for arm in ("A", "B"):
        cp_path = reference_dir / f"arm_{arm.lower()}_step5400.pt"
        _require(cp_path.is_file(), f"historical {arm}5400 checkpoint is missing")
        payload = torch.load(cp_path, map_location="cpu", weights_only=True)
        _require(payload.get("task_id") == historical_training.TASK_ID and payload.get("arm") == arm and payload.get("completed_updates") == 5_400 and payload.get("optimizer_step") == 5_400, f"historical {arm}5400 endpoint identity or Adam step differs")
        old_checkpoints[arm] = payload

    ancestor_checkpoint_path = _portable_resolve(root, source["population"]["source_checkpoint"])
    ancestor_dir = ancestor_checkpoint_path.parent
    ancestor_population = _json(ancestor_dir / "input_population.json")
    ancestor_report = _json(ancestor_dir / "report.json")
    _require(ancestor_population.get("task_id") == "M0_GENERATOR_TRAINING_POPULATION_EXPANSION_COMPARISON_V1", "named 3,300-row ancestor population has a different task identity")
    ancestor_rows = ancestor_population.get("source_row_indices", [])
    _require(len(ancestor_rows) == ANCESTOR_ROWS and len(set(ancestor_rows)) == ANCESTOR_ROWS, "named ancestor population is not exactly 3,300 unique rows")
    _require(ancestor_report.get("task_id") == ancestor_population["task_id"], "ancestor report/population identities differ")

    audit_rows = grad_population.get("source_row_indices", [])
    _require(audit_rows == old_population.get("source_row_indices"), "gradient audit rows differ from the historical 500-row comparison")
    _require(grad_population.get("labels") == old_population.get("labels") and grad_population.get("roles") == old_population.get("roles"), "gradient audit labels or roles differ from the historical comparison")
    source_payload = source["checkpoint"]
    lineage_payload = source["generation_inputs"]["source"]
    _validate_source_checkpoint(source_payload, lineage_payload)
    return {
        "source": source,
        "gradient_audit": {"directory": gradient_dir.as_posix(), "task_id": grad_report["task_id"], "status": grad_report["status"], "lambda": FIXED_LAMBDA, "calibration": calibration},
        "historical": {"directory": reference_dir.as_posix(), "report": old_report, "population": old_population, "checkpoints": old_checkpoints},
        "ancestor": {"directory": ancestor_dir, "population": ancestor_population, "report": ancestor_report},
        "source_paths": source["generation_inputs"]["paths"],
    }


def _validate_source_checkpoint(source_payload: Mapping[str, Any], lineage_payload: Mapping[str, Any]) -> None:
    _require(source_payload.get("task_id") == artifact_source.TASK_ID and source_payload.get("arm") == "A_unscaled_target", "named A4700 copy belongs to a different source arm")
    _require(source_payload.get("completed_updates") == SOURCE_STEP and source_payload.get("optimizer_step") == SOURCE_STEP, "A4700 model/Adam state is not at step 4700")
    _require(source_payload.get("source_history_count") == SOURCE_STEP and source_payload.get("task_local_updates") == 0, "A4700 copy provenance does not bind the full source history")
    _require(lineage_payload.get("completed_updates") == SOURCE_STEP and lineage_payload.get("optimizer_step") == SOURCE_STEP and len(lineage_payload.get("training_history", [])) == SOURCE_STEP, "source lineage history or step is incomplete")
    _require(tuple(source_payload["trainable_parameter_state"]["generator.phase"].shape) == (340_000,), "A4700 Generator phase layout differs")
    _require(tuple(source_payload["trainable_parameter_state"]["encoder.phase"].shape) == (2, 1, 100, 100), "A4700 Encoder phase layout differs")
    _require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in source_payload["trainable_parameter_state"].values()), "A4700 phase state is nonfinite or not FP32")
    adam = source_payload["optimizer_state"]
    _require(len(adam["param_groups"]) == 1 and adam["param_groups"][0]["params"] == [0] and set(adam["state"]) == {0}, "A4700 Adam parameter mapping differs")
    group, state = adam["param_groups"][0], adam["state"][0]
    _require(float(group["lr"]) == 1.0e-3 and tuple(group["betas"]) == (0.9, 0.999) and float(group["eps"]) == 1.0e-8 and float(group["weight_decay"]) == 0.0, "A4700 Adam hyperparameters differ")
    _require(int(float(state["step"].item())) == SOURCE_STEP and tuple(state["exp_avg"].shape) == (340_000,) and tuple(state["exp_avg_sq"].shape) == (340_000,), "A4700 Adam step or moment geometry differs")
    _require(bool(torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()), "A4700 Adam moments are nonfinite")
    _require(capacity._tree_equal(adam, lineage_payload["optimizer_state"]), "A4700 copied Adam differs from its named source lineage")
    for field in ("trainable_parameter_state", "frozen_runtime_state", "references", "weights"):
        _require(capacity._tree_equal(source_payload[field], lineage_payload[field]), f"A4700 copy differs from source lineage field {field}")


def _validate_selected_rows(
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    dev_indices: torch.Tensor,
    old_fit_rows: torch.Tensor,
    added_rows: torch.Tensor,
    fit_rows: torch.Tensor,
    dev_rows: torch.Tensor,
    ancestor_rows: torch.Tensor,
    old_rows: torch.Tensor,
    *,
    fit_per_class: int = FIT_PER_CLASS,
    dev_per_class: int = DEV_PER_CLASS,
    new_per_class: int = NEW_ROWS_PER_CLASS,
) -> None:
    row_count = labels.numel()
    sets = {name: set(value.tolist()) for name, value in {
        "train": train_indices, "dev": dev_indices, "old_fit": old_fit_rows, "added": added_rows,
        "fit": fit_rows, "confirmation": dev_rows, "ancestor": ancestor_rows, "old": old_rows,
    }.items()}
    for name, values in sets.items():
        _require(all(type(index) is int and 0 <= index < row_count for index in values), f"{name} population contains an out-of-range row")
    _require(len(sets["train"]) == train_indices.numel() and len(sets["dev"]) == dev_indices.numel(), "canonical split contains duplicate rows")
    _require(not (sets["train"] & sets["dev"]), "canonical TRAIN and DEV overlap")
    _require(not (sets["fit"] & sets["confirmation"]), "DEV row entered the training population")
    _require(sets["old_fit"].issubset(sets["fit"]) and len(sets["fit"]) == FIT_ROWS, "the original fit rows were not retained exactly once")
    _require(not (sets["added"] & (sets["ancestor"] | sets["old"])), "new fit row overlaps a named exclusion population")
    _require(sets["added"].issubset(sets["train"]) and sets["fit"].issubset(sets["train"]), "fit rows include a non-TRAIN row")
    _require(sets["confirmation"].issubset(sets["dev"]), "confirmation row is outside canonical DEV")
    _require(torch.bincount(labels.index_select(0, fit_rows), minlength=10).tolist() == [fit_per_class] * 10, "fit rows are not class balanced")
    _require(torch.bincount(labels.index_select(0, added_rows), minlength=10).tolist() == [new_per_class] * 10, "new fit rows are not class balanced")
    _require(torch.bincount(labels.index_select(0, dev_rows), minlength=10).tolist() == [dev_per_class] * 10, "DEV rows are not class balanced")


def _select_class_rows(labels: torch.Tensor, candidates: torch.Tensor, excluded: set[int], per_class: int, seed: int) -> torch.Tensor:
    generator_cpu = torch.Generator(device="cpu").manual_seed(seed)
    selected_by_class = []
    for digit in range(10):
        pool = candidates[(labels.index_select(0, candidates) == digit)]
        if excluded:
            keep = torch.tensor([int(row) not in excluded for row in pool.tolist()], dtype=torch.bool)
            pool = pool[keep]
        _require(pool.numel() >= per_class, f"canonical TRAIN class {digit} has fewer than {per_class} eligible rows")
        order = torch.randperm(pool.numel(), generator=generator_cpu)[:per_class]
        selected_by_class.append(pool.index_select(0, order))
    return torch.stack(selected_by_class, dim=1).reshape(-1).to(torch.long)


def _population_inputs(root: Path, sources: Mapping[str, Any]) -> dict[str, Any]:
    legacy = sources["source"]["generation_inputs"]
    data_path = sources["source_paths"]["training_data"]
    image_store = diagnostic._load_torch(data_path, mmap=True)
    image_rows, canonical_labels = diagnostic._read_image_labels(image_store)
    train_indices, dev_indices = diagnostic.stratified_mnist_indices(
        canonical_labels, seed=SPLIT_SEED, validation_per_label=VALIDATION_PER_CLASS
    )
    ancestor_rows = torch.tensor(sources["ancestor"]["population"]["source_row_indices"], dtype=torch.long)
    old_rows = torch.tensor(sources["historical"]["population"]["source_row_indices"], dtype=torch.long)
    old_fit_rows = old_rows[:200].clone()
    train_mask = torch.zeros(60_000, dtype=torch.bool)
    dev_mask = torch.zeros(60_000, dtype=torch.bool)
    train_mask[train_indices.cpu()] = True
    dev_mask[dev_indices.cpu()] = True
    _require(bool(train_mask.index_select(0, ancestor_rows).all() and train_mask.index_select(0, old_rows).all()), "named training ancestors contain a non-TRAIN row")
    _require(bool(dev_mask.index_select(0, dev_indices.cpu()).all()), "canonical DEV split construction failed")
    excluded = set(ancestor_rows.tolist()) | set(old_rows.tolist())
    added_rows = _select_class_rows(canonical_labels, train_indices.cpu(), excluded, NEW_ROWS_PER_CLASS, SELECTION_SEED)
    dev_rows = _select_class_rows(canonical_labels, dev_indices.cpu(), set(), DEV_PER_CLASS, DEV_SELECTION_SEED)
    fit_rows = torch.cat((old_fit_rows, added_rows))
    fit_labels = canonical_labels.index_select(0, fit_rows)
    dev_labels = canonical_labels.index_select(0, dev_rows)
    _require(tuple(fit_rows.shape) == (FIT_ROWS,) and torch.unique(fit_rows).numel() == FIT_ROWS, "2,000-row fit population is incomplete or duplicated")
    _require(tuple(dev_rows.shape) == (DEV_ROWS,) and torch.unique(dev_rows).numel() == DEV_ROWS, "1,000-row DEV population is incomplete or duplicated")
    _validate_selected_rows(canonical_labels, train_indices.cpu(), dev_indices.cpu(), old_fit_rows, added_rows, fit_rows, dev_rows, ancestor_rows, old_rows)
    all_rows = torch.cat((fit_rows, dev_rows))
    all_labels = canonical_labels.index_select(0, all_rows)
    roles = ["fit"] * FIT_ROWS + ["dev"] * DEV_ROWS
    images = image_rows.index_select(0, all_rows).unsqueeze(1).to(torch.float32).div_(255.0).contiguous()
    _require(tuple(images.shape) == (FIT_ROWS + DEV_ROWS, 1, 28, 28) and bool(torch.isfinite(images).all()) and 0.0 <= float(images.min()) <= float(images.max()) <= 1.0, "fixed training/DEV image tensor is invalid")
    population = {
        "task_id": TASK_ID,
        "source_checkpoint": sources["source"]["checkpoint_path"].as_posix(),
        "source_checkpoint_step": SOURCE_STEP,
        "source_adam_step": SOURCE_STEP,
        "source_artifact_attempt": sources["source"]["artifact_dir"].as_posix(),
        "fixed_lambda": FIXED_LAMBDA,
        "canonical_split": {"seed": SPLIT_SEED, "validation_per_class": VALIDATION_PER_CLASS, "fit_role": "canonical TRAIN", "confirmation_role": "canonical DEV with historical evaluator-development use"},
        "fit_selection": {"seed": SELECTION_SEED, "original_fit_rows_reused": old_fit_rows.tolist(), "new_rows_per_class": NEW_ROWS_PER_CLASS, "new_rows": added_rows.tolist(), "excluded_ancestor_rows": ancestor_rows.tolist(), "excluded_old_comparison_rows": old_rows.tolist(), "excluded_union_count": len(excluded), "old_comparison_overlap_with_ancestor_count": len(set(old_rows.tolist()) & set(ancestor_rows.tolist()))},
        "dev_selection": {"seed": DEV_SELECTION_SEED, "rows_per_class": DEV_PER_CLASS, "source_row_indices": dev_rows.tolist()},
        "source_row_indices": all_rows.tolist(),
        "labels": all_labels.tolist(),
        "roles": roles,
        "role_positions": {"fit": [0, FIT_ROWS], "dev": [FIT_ROWS, FIT_ROWS + DEV_ROWS]},
        "fit_rows": FIT_ROWS,
        "dev_rows": DEV_ROWS,
        "fit_rows_per_class": FIT_PER_CLASS,
        "dev_rows_per_class": DEV_PER_CLASS,
        "training_seeds": list(TRAIN_SEEDS),
        "sample_seeds": list(SAMPLE_SEEDS),
        "updates_per_arm": UPDATES_PER_ARM,
        "scientific_updates": TOTAL_UPDATES,
        "batch_size": BATCH_SIZE,
        "batches_per_epoch": BATCHES_PER_EPOCH,
        "epochs": EPOCHS,
        "numeric_mode": NUMERIC_MODE,
        "claim_boundary": "DEV is a Generator held-out confirmation set; historical evaluator development use prevents a whole-project blind-test claim.",
    }
    return {"population": population, "rows": all_rows, "labels": all_labels, "roles": roles, "images": images, "fit_rows": fit_rows, "fit_labels": fit_labels, "fit_images": images[:FIT_ROWS], "dev_rows": dev_rows, "dev_labels": dev_labels, "dev_images": images[FIT_ROWS:], "legacy": legacy, "image_store": image_store}


def _make_schedule(fit_labels: torch.Tensor, seed: int) -> list[list[int]]:
    _require(tuple(fit_labels.shape) == (FIT_ROWS,) and torch.bincount(fit_labels, minlength=10).tolist() == [FIT_PER_CLASS] * 10, "schedule labels do not match the fixed fit population")
    generator_cpu = torch.Generator(device="cpu").manual_seed(seed)
    class_positions = [torch.nonzero(fit_labels == digit, as_tuple=False).flatten() for digit in range(10)]
    schedule: list[list[int]] = []
    for _epoch in range(EPOCHS):
        orders = [positions.index_select(0, torch.randperm(FIT_PER_CLASS, generator=generator_cpu)) for positions in class_positions]
        for batch in range(BATCHES_PER_EPOCH):
            row_positions = [int(orders[digit][batch]) for digit in range(10)]
            _require(fit_labels.index_select(0, torch.tensor(row_positions)).tolist() == list(range(10)), "batch schedule lost its one-row-per-class mapping")
            schedule.append(row_positions)
    _require(len(schedule) == UPDATES_PER_ARM, "fixed schedule does not contain exactly 2,000 updates")
    return schedule


def _new_endpoint(sources: Mapping[str, Any], device: torch.device) -> tuple[Any, torch.optim.Optimizer, dict[str, nn.Parameter]]:
    source_payload = sources["source"]["checkpoint"]
    legacy = sources["source"]["generation_inputs"]
    consumer, _optimizer, _params, references, weights = diagnostic._new_endpoint(
        "step2200", legacy["base2200"], source_payload["references"], source_payload["weights"], device
    )
    consumer.runtime.generator = capacity.ExpandedGenerator(
        torch.zeros((2, 1, 100, 100), dtype=torch.float32, device=device), propagation_grid_size=2_048
    ).to(device)
    parameters = capacity._parameters(consumer)
    with torch.no_grad():
        for name in ("encoder.phase", "generator.phase"):
            parameters[name].copy_(source_payload["trainable_parameter_state"][name].to(device))
    consumer.runtime.eval()
    optimizer = capacity._new_generator_optimizer(parameters["generator.phase"])
    optimizer.load_state_dict(copy.deepcopy(source_payload["optimizer_state"]))
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == SOURCE_STEP, "loaded Generator Adam state differs from A4700")
    _require(consumer.runtime.config.propagation_grid_size == 2_048, "formal propagation grid differs from 2,048")
    _require(tuple(parameters["generator.phase"].shape) == (340_000,), "formal Generator parameter count differs")
    _require(parameters["encoder.phase"].requires_grad and parameters["generator.phase"].requires_grad, "semantic gradient graph lost a phase group")
    capacity._assert_state_equal(source_payload["frozen_runtime_state"], capacity._frozen_runtime_state(consumer), "A4700 frozen OLS/runtime state")
    _require(capacity._tree_equal(references.to_payload(), diagnostic.bogt.FrozenBogtReferences.from_payload(source_payload["references"]).to_payload()), "A4700 reference object did not bind")
    _require(capacity._tree_equal(weights.to_payload(), diagnostic.bogt.BogtLossWeights.from_payload(source_payload["weights"]).to_payload()), "A4700 loss weights did not bind")
    return consumer, optimizer, parameters


def _evaluator_logits(evaluators: Mapping[str, nn.Module], view: torch.Tensor) -> dict[str, torch.Tensor]:
    return {name: capacity._evaluator_logits(evaluators[name], view, name) for name in EVALUATOR_NAMES}


def _cached_fit_target_shapes(cache: Mapping[str, Any]) -> torch.Tensor:
    coefficients = cache.get("coefficients")
    _require(torch.is_tensor(coefficients) and coefficients.ndim == 2 and coefficients.shape[1] == 100, "saved OLS coefficient geometry is incomplete")
    if "target_shape_vectors" in cache:
        shapes = cache["target_shape_vectors"]
    else:
        targets = cache.get("target_slm_intensity")
        _require(torch.is_tensor(targets) and tuple(targets.shape) == (coefficients.shape[0], 1, 200, 200), "saved legacy frozen target association is incomplete")
        shapes = frozen._shape_vectors(targets, 28)
    _require(torch.is_tensor(shapes) and tuple(shapes.shape) == (coefficients.shape[0], 28 * 28), "saved frozen target shape vectors are incomplete")
    _require(bool(torch.isfinite(shapes).all() and (shapes >= 0.0).all()), "saved frozen target shape vectors are invalid")
    return shapes


def _ols_cache(consumer: Any, inputs: Mapping[str, Any], target: Path) -> dict[str, torch.Tensor]:
    cache_path = target / "ols_fit_coefficients.pt"
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        _require(cached.get("task_id") == TASK_ID and torch.equal(cached.get("source_row_indices"), inputs["fit_rows"]), "saved OLS cache source rows differ")
        _require(tuple(cached.get("coefficients", torch.empty(0)).shape) == (FIT_ROWS, 100), "saved OLS cache geometry differs")
        _require(cached["coefficients"].is_complex() and bool(torch.isfinite(cached["coefficients"].real).all() and torch.isfinite(cached["coefficients"].imag).all()), "saved OLS coefficients are invalid")
        return {"coefficients": cached["coefficients"], "target_shape_vectors": _cached_fit_target_shapes(cached), "cache_was_existing": True}
    coefficients: list[torch.Tensor] = []
    target_shapes: list[torch.Tensor] = []
    with torch.inference_mode():
        for index in range(FIT_ROWS):
            result = consumer.runtime.forward_from_image(inputs["fit_images"][index : index + 1].to(device=consumer.device, dtype=torch.float32))
            coefficients.append(result.z_ols.coefficients.detach().cpu())
            target_shapes.append(frozen._shape_vectors(result.image_input.slm_intensity.detach().cpu(), 28))
            del result
    payload = {
        "task_id": TASK_ID,
        "source_row_indices": inputs["fit_rows"].clone(),
        "coefficients": torch.cat(coefficients, dim=0).contiguous(),
        "target_shape_vectors": torch.cat(target_shapes, dim=0).contiguous(),
        "role": "read-only frozen Encoder/OLS coefficient cache for distribution statistics and prior fitting; compact normalized fit-target shapes for sample summaries",
    }
    _require(tuple(payload["coefficients"].shape) == (FIT_ROWS, 100) and payload["coefficients"].is_complex(), "computed OLS cache is invalid")
    _require(tuple(payload["target_shape_vectors"].shape) == (FIT_ROWS, 28 * 28), "computed fit-target shape cache is invalid")
    _write_torch(cache_path, payload)
    return {"coefficients": payload["coefficients"], "target_shape_vectors": payload["target_shape_vectors"], "cache_was_existing": False}


def _verify_cached_ols(consumer: Any, inputs: Mapping[str, Any], cache: Mapping[str, Any]) -> dict[str, Any]:
    rows_to_check = FIT_ROWS if cache.get("cache_was_existing") else BATCH_SIZE
    max_z_error = 0.0
    max_target_shape_error = 0.0
    cached_shapes = _cached_fit_target_shapes(cache)
    with torch.inference_mode():
        for start in range(0, rows_to_check, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, rows_to_check)
            z_rows, target_shapes = [], []
            for index in range(start, stop):
                result = consumer.runtime.forward_from_image(inputs["fit_images"][index : index + 1].to(device=consumer.device, dtype=torch.float32))
                z_rows.append(result.z_ols.coefficients.detach().cpu())
                target_shapes.append(frozen._shape_vectors(result.image_input.slm_intensity.detach().cpu(), 28))
                del result
            recomputed_z = torch.cat(z_rows, dim=0)
            recomputed_shapes = torch.cat(target_shapes, dim=0)
            cached_z = cache["coefficients"][start:stop]
            cached_target_shapes = cached_shapes[start:stop]
            _require(torch.allclose(recomputed_z, cached_z, rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"cached OLS coefficients differ from the named Encoder on rows {start}:{stop}")
            _require(torch.allclose(recomputed_shapes, cached_target_shapes, rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"cached normalized target shapes differ from the named inputs on rows {start}:{stop}")
            max_z_error = max(max_z_error, float((recomputed_z.to(torch.complex128) - cached_z.to(torch.complex128)).abs().max()))
            max_target_shape_error = max(max_target_shape_error, float((recomputed_shapes.to(torch.float64) - cached_target_shapes.to(torch.float64)).abs().max()))
    return {
        "executed": True,
        "passed": True,
        "rows_recomputed": rows_to_check,
        "cache_was_existing": bool(cache.get("cache_was_existing")),
        "max_abs_coefficient_difference": max_z_error,
        "max_abs_target_shape_difference": max_target_shape_error,
    }


def _to_interleaved(z: torch.Tensor) -> torch.Tensor:
    _require(z.is_complex() and tuple(z.shape) == (FIT_ROWS, 100), "fit coefficients must be complex[2000,100]")
    _require(bool(torch.isfinite(z.real).all() and torch.isfinite(z.imag).all()), "OLS fit coefficients are nonfinite")
    return torch.stack((z.real.to(torch.float64), z.imag.to(torch.float64)), dim=-1).reshape(FIT_ROWS, 200).contiguous()


def _fit_prior(z: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    x = _to_interleaved(z)
    _require(torch.bincount(labels.cpu(), minlength=10).tolist() == [FIT_PER_CLASS] * 10, "prior fit population is not 200 rows per class")
    means, scales, cholesky, correlations = [], [], [], []
    reconstruction_errors, minimum_eigenvalues = [], []
    for digit in range(10):
        rows = x[labels.cpu() == digit]
        mean = rows.mean(dim=0)
        centered = rows - mean
        covariance = centered.T @ centered / rows.shape[0]
        diagonal = covariance.diag()
        _require(bool(torch.isfinite(covariance).all() and (diagonal > 0.0).all()), f"class {digit} prior has invalid marginal variance")
        scale = diagonal.sqrt()
        correlation = covariance / scale[:, None] / scale[None, :]
        correlation = 0.5 * (correlation + correlation.T)
        block = torch.zeros_like(correlation)
        for port in range(100):
            begin = 2 * port
            block[begin : begin + 2, begin : begin + 2] = correlation[begin : begin + 2, begin : begin + 2]
        shrunk = 0.5 * correlation + 0.5 * block
        chol = torch.linalg.cholesky(shrunk)
        error = float((chol @ chol.T - shrunk).abs().max())
        eigenvalue = float(torch.linalg.eigvalsh(shrunk).min())
        _require(math.isfinite(eigenvalue) and eigenvalue > 0.0 and error <= 1.0e-12, f"class {digit} prior is not positive definite without regularization")
        means.append(mean)
        scales.append(scale)
        cholesky.append(chol)
        correlations.append(shrunk)
        reconstruction_errors.append(error)
        minimum_eigenvalues.append(eigenvalue)
    return {
        "means": torch.stack(means), "scales": torch.stack(scales), "joint_cholesky": torch.stack(cholesky),
        "joint_correlation": torch.stack(correlations), "reconstruction_error": torch.tensor(reconstruction_errors, dtype=torch.float64),
        "minimum_eigenvalue": torch.tensor(minimum_eigenvalues, dtype=torch.float64),
        "estimator": "per-class covariance divided by n; 0.5 full correlation plus 0.5 within-port 2x2 correlation blocks; CPU float64; no jitter or PSD repair",
    }


def _sample_bank(priors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    banks, identifiers = [], []
    for seed in SAMPLE_SEEDS:
        rng = torch.Generator(device="cpu").manual_seed(seed)
        epsilon = torch.randn((10, SAMPLES_PER_CLASS, 200), generator=rng, dtype=torch.float64)
        per_seed = []
        for digit in range(10):
            correlated = epsilon[digit] @ priors["joint_cholesky"][digit].T
            real_vector = priors["means"][digit] + correlated * priors["scales"][digit]
            per_seed.append(torch.complex(real_vector[:, 0::2], real_vector[:, 1::2]).to(torch.complex64))
            identifiers.extend({"sample_seed": seed, "requested_class": digit, "draw_index": draw} for draw in range(SAMPLES_PER_CLASS))
        banks.append(torch.stack(per_seed))
    bank = torch.stack(banks)
    _require(tuple(bank.shape) == (len(SAMPLE_SEEDS), 10, SAMPLES_PER_CLASS, 100), "fixed conditional sample bank has the wrong shape")
    _require(bool(torch.isfinite(bank.real).all() and torch.isfinite(bank.imag).all()), "fixed conditional sample bank is nonfinite")
    return {"task_id": TASK_ID, "z": bank, "identifiers": identifiers, "prior": "one 50%-shrunk joint Gaussian per class fitted to the 2,000 fit-row OLS coefficients"}


def _autograd_group(loss: torch.Tensor, parameters: Mapping[str, nn.Parameter], *, retain_graph: bool) -> dict[str, torch.Tensor]:
    names = ("encoder.phase", "generator.phase")
    gradients = torch.autograd.grad(loss, tuple(parameters[name] for name in names), retain_graph=retain_graph, allow_unused=True)
    return {name: torch.zeros_like(parameters[name]) if value is None else value.detach() for name, value in zip(names, gradients, strict=True)}


def _canonical_semantic_direction(
    evaluator_gradients: Mapping[str, Mapping[str, torch.Tensor]],
) -> tuple[dict[str, torch.Tensor] | None, str | None]:
    _require(set(evaluator_gradients) == set(TRAIN_EVALUATOR_NAMES), "semantic gradients do not contain the three fixed evaluators")
    semantic_inputs = {
        key: evaluator_gradients[name]
        for name, key in zip(TRAIN_EVALUATOR_NAMES, ("L_S", "L_I", "L_C"), strict=True)
    }
    return diagnostic._safe_canonical_gradient(semantic_inputs)


def _float_tensor_comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype or not reference.dtype.is_floating_point:
        return {"passed": False, "reason": "shape_or_dtype_mismatch"}
    if not bool(torch.isfinite(reference).all() and torch.isfinite(candidate).all()):
        return {"passed": False, "reason": "nonfinite_state"}
    difference = (reference.to(torch.float64) - candidate.to(torch.float64)).abs()
    scale = max(float(reference.abs().max()), float(candidate.abs().max()))
    tolerance_rtol = 2.0 * torch.finfo(reference.dtype).eps
    tolerance = tolerance_rtol * scale
    maximum = float(difference.max())
    return {
        "passed": maximum <= tolerance,
        "max_abs_difference": maximum,
        "scale": scale,
        "tolerance": tolerance,
        "relative_tolerance": tolerance_rtol,
        "rule": "max_abs_difference <= relative_tolerance * max_abs_tensor_value",
    }


def _adam_state_comparison(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    if set(reference) != set(candidate) or not capacity._tree_equal(reference.get("param_groups"), candidate.get("param_groups")):
        return {"passed": False, "reason": "optimizer_groups_or_state_keys_differ"}
    reference_state, candidate_state = reference.get("state"), candidate.get("state")
    if not isinstance(reference_state, Mapping) or not isinstance(candidate_state, Mapping) or set(reference_state) != set(candidate_state):
        return {"passed": False, "reason": "optimizer_parameter_state_differs"}
    tensor_comparisons: dict[str, Any] = {}
    for parameter_id, reference_fields in reference_state.items():
        candidate_fields = candidate_state[parameter_id]
        if not isinstance(reference_fields, Mapping) or not isinstance(candidate_fields, Mapping) or set(reference_fields) != set(candidate_fields):
            return {"passed": False, "reason": f"optimizer_state_fields_differ_for_parameter_{parameter_id}"}
        for name, reference_value in reference_fields.items():
            candidate_value = candidate_fields[name]
            if name in {"exp_avg", "exp_avg_sq"} and torch.is_tensor(reference_value) and torch.is_tensor(candidate_value):
                comparison = _float_tensor_comparison(reference_value, candidate_value)
                tensor_comparisons[f"{parameter_id}.{name}"] = comparison
                if not comparison["passed"]:
                    return {"passed": False, "reason": f"optimizer_{name}_exceeds_relative_tolerance", "tensor_comparisons": tensor_comparisons}
            elif not capacity._tree_equal(reference_value, candidate_value):
                return {"passed": False, "reason": f"optimizer_{name}_differs", "tensor_comparisons": tensor_comparisons}
    return {"passed": True, "param_groups_exact": True, "step_exact": True, "tensor_comparisons": tensor_comparisons}


def _tensor_repeatability_comparison(reference: torch.Tensor, repeated: torch.Tensor, resumed: torch.Tensor) -> dict[str, Any]:
    tensors = (reference, repeated, resumed)
    if any(value.shape != reference.shape or value.dtype != reference.dtype or not value.dtype.is_floating_point for value in tensors):
        return {"passed": False, "reason": "shape_or_dtype_mismatch"}
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        return {"passed": False, "reason": "nonfinite_state"}

    def difference(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
        delta = (left.to(torch.float64) - right.to(torch.float64)).abs()
        return {
            "max_abs_difference": float(delta.max()),
            "rms_abs_difference": float(torch.sqrt(delta.square().mean())),
        }

    repeat_variation = difference(reference, repeated)
    resumed_variation = difference(reference, resumed)
    return {
        "passed": all(resumed_variation[key] <= repeat_variation[key] for key in repeat_variation),
        "continuous_repeat_variation": repeat_variation,
        "resume_variation": resumed_variation,
        "rule": "resume max_abs and RMS differences must not exceed an independent same-device continuous repeat",
    }


def _adam_state_repeatability_comparison(
    reference: Mapping[str, Any],
    repeated: Mapping[str, Any],
    resumed: Mapping[str, Any],
) -> dict[str, Any]:
    if set(reference) != set(repeated) or set(reference) != set(resumed):
        return {"passed": False, "reason": "optimizer_state_keys_differ"}
    for candidate in (repeated, resumed):
        if not capacity._tree_equal(reference.get("param_groups"), candidate.get("param_groups")):
            return {"passed": False, "reason": "optimizer_parameter_groups_differ"}
    reference_state, repeated_state, resumed_state = reference.get("state"), repeated.get("state"), resumed.get("state")
    if any(not isinstance(value, Mapping) for value in (reference_state, repeated_state, resumed_state)):
        return {"passed": False, "reason": "optimizer_parameter_state_is_missing"}
    if set(reference_state) != set(repeated_state) or set(reference_state) != set(resumed_state):
        return {"passed": False, "reason": "optimizer_parameter_ids_differ"}
    tensor_comparisons: dict[str, Any] = {}
    for parameter_id, reference_fields in reference_state.items():
        repeated_fields, resumed_fields = repeated_state[parameter_id], resumed_state[parameter_id]
        if any(not isinstance(value, Mapping) for value in (reference_fields, repeated_fields, resumed_fields)):
            return {"passed": False, "reason": f"optimizer_fields_missing_for_parameter_{parameter_id}"}
        if set(reference_fields) != set(repeated_fields) or set(reference_fields) != set(resumed_fields):
            return {"passed": False, "reason": f"optimizer_state_fields_differ_for_parameter_{parameter_id}"}
        for name, reference_value in reference_fields.items():
            repeated_value, resumed_value = repeated_fields[name], resumed_fields[name]
            if name in {"exp_avg", "exp_avg_sq"} and all(torch.is_tensor(value) for value in (reference_value, repeated_value, resumed_value)):
                comparison = _tensor_repeatability_comparison(reference_value, repeated_value, resumed_value)
                tensor_comparisons[f"{parameter_id}.{name}"] = comparison
                if not comparison["passed"]:
                    return {"passed": False, "reason": f"optimizer_{name}_exceeds_continuous_repeatability", "tensor_comparisons": tensor_comparisons}
            elif not capacity._tree_equal(reference_value, repeated_value) or not capacity._tree_equal(reference_value, resumed_value):
                return {"passed": False, "reason": f"optimizer_{name}_differs_exactly", "tensor_comparisons": tensor_comparisons}
    return {"passed": True, "param_groups_exact": True, "step_exact": True, "tensor_comparisons": tensor_comparisons}


def _continuous_tensor_envelope(values: Sequence[torch.Tensor]) -> dict[str, float]:
    _require(len(values) >= 2 and all(value.shape == values[0].shape and value.dtype == values[0].dtype for value in values), "continuous repeat states differ in shape or dtype")
    _require(all(bool(torch.isfinite(value).all()) for value in values), "continuous repeat state is nonfinite")

    def metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
        delta = (left.to(torch.float64) - right.to(torch.float64)).abs()
        return {"max_abs_difference": float(delta.max()), "rms_abs_difference": float(torch.sqrt(delta.square().mean()))}

    pairs = [metrics(values[left], values[right]) for left in range(len(values)) for right in range(left + 1, len(values))]
    return {name: max(pair[name] for pair in pairs) for name in pairs[0]}


def _tensor_repeatability_envelope(
    values: Sequence[torch.Tensor],
    candidate: torch.Tensor,
    *,
    fixed_envelope: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if len(values) < 2 or any(value.shape != values[0].shape or value.dtype != values[0].dtype for value in [*values, candidate]):
        return {"passed": False, "reason": "repeatability_shape_or_dtype_mismatch"}
    if any(not bool(torch.isfinite(value).all()) for value in [*values, candidate]):
        return {"passed": False, "reason": "repeatability_state_is_nonfinite"}

    def metrics(left: torch.Tensor, right: torch.Tensor) -> dict[str, float]:
        delta = (left.to(torch.float64) - right.to(torch.float64)).abs()
        return {"max_abs_difference": float(delta.max()), "rms_abs_difference": float(torch.sqrt(delta.square().mean()))}

    envelope = dict(fixed_envelope) if fixed_envelope is not None else _continuous_tensor_envelope(values)
    if envelope != _continuous_tensor_envelope(values):
        return {"passed": False, "reason": "persisted_repeatability_envelope_does_not_match_continuous_repeats"}
    _require(set(envelope) == {"max_abs_difference", "rms_abs_difference"}, "fixed repeatability envelope fields differ")
    candidate_difference = metrics(values[0], candidate)
    return {
        "passed": all(candidate_difference[name] <= envelope[name] for name in envelope),
        "continuous_pairwise_envelope": envelope,
        "recovery_difference_from_reference": candidate_difference,
        "continuous_repeat_count": len(values),
        "rule": "recovery maximum absolute and RMS differences do not exceed the precomputed five-run same-device pairwise envelope",
    }


def _continuous_adam_envelope(values: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    _require(len(values) >= 2, "Adam repeatability envelope needs five continuous states")
    base = values[0].get("state")
    _require(isinstance(base, Mapping) and all(set(value) == set(values[0]) and capacity._tree_equal(values[0].get("param_groups"), value.get("param_groups")) for value in values[1:]), "continuous Adam state layouts differ")
    envelope: dict[str, dict[str, float]] = {}
    for parameter_id, fields in base.items():
        _require(all(isinstance(value.get("state"), Mapping) and set(value["state"]) == set(base) for value in values), "continuous Adam parameter mapping differs")
        for name, reference in fields.items():
            repeated = [value["state"][parameter_id][name] for value in values]
            if name in {"exp_avg", "exp_avg_sq"} and torch.is_tensor(reference):
                envelope[f"{parameter_id}.{name}"] = _continuous_tensor_envelope(repeated)
            else:
                _require(all(capacity._tree_equal(reference, value) for value in repeated[1:]), f"continuous Adam {name} states differ exactly")
    return envelope


def _adam_repeatability_envelope(
    values: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
    *,
    fixed_envelope: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, Any]:
    if len(values) < 2 or any(set(value) != set(values[0]) for value in [*values, candidate]):
        return {"passed": False, "reason": "optimizer_state_keys_differ"}
    if any(not capacity._tree_equal(values[0].get("param_groups"), value.get("param_groups")) for value in [*values[1:], candidate]):
        return {"passed": False, "reason": "optimizer_parameter_groups_differ"}
    base = values[0].get("state")
    if not isinstance(base, Mapping) or any(not isinstance(value.get("state"), Mapping) or set(value["state"]) != set(base) for value in [*values, candidate]):
        return {"passed": False, "reason": "optimizer_parameter_state_is_missing"}
    comparisons: dict[str, Any] = {}
    for parameter_id, fields in base.items():
        if not isinstance(fields, Mapping):
            return {"passed": False, "reason": f"optimizer_fields_missing_for_parameter_{parameter_id}"}
        for value in [*values, candidate]:
            candidate_fields = value["state"].get(parameter_id)
            if not isinstance(candidate_fields, Mapping) or set(candidate_fields) != set(fields):
                return {"passed": False, "reason": f"optimizer_state_fields_differ_for_parameter_{parameter_id}"}
        for name, reference_value in fields.items():
            repeated = [value["state"][parameter_id][name] for value in values]
            recovery = candidate["state"][parameter_id][name]
            key = f"{parameter_id}.{name}"
            if name in {"exp_avg", "exp_avg_sq"} and torch.is_tensor(reference_value) and torch.is_tensor(recovery):
                comparisons[key] = _tensor_repeatability_envelope(
                    repeated, recovery, fixed_envelope=(fixed_envelope or {}).get(key)
                )
                if not comparisons[key]["passed"]:
                    return {"passed": False, "reason": f"optimizer_{name}_exceeds_fixed_repeatability_envelope", "tensor_comparisons": comparisons}
            elif any(not capacity._tree_equal(reference_value, value) for value in [*repeated[1:], recovery]):
                return {"passed": False, "reason": f"optimizer_{name}_differs_exactly", "tensor_comparisons": comparisons}
    return {"passed": True, "param_groups_exact": True, "step_exact": True, "tensor_comparisons": comparisons}


def _fast_batch_gradient(
    consumer: Any,
    parameters: Mapping[str, nn.Parameter],
    evaluators: Mapping[str, nn.Module],
    inputs: Mapping[str, Any],
    batch_positions: Sequence[int],
    ols_coefficients: torch.Tensor,
    qfg_lambda: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Accumulate exact logical-batch gradients in bounded optical microbatches."""
    _require(len(batch_positions) == BATCH_SIZE and math.isfinite(qfg_lambda) and qfg_lambda >= 0.0, "fast gradient batch or foreground weight is invalid")
    microbatch_size = int(inputs.get("microbatch_size", 1))
    _require(microbatch_size in {1, 2, 5, 10}, "microbatch size is outside the fixed engineering candidates")
    positions = torch.tensor(batch_positions, dtype=torch.long)
    images = inputs["fit_images"].index_select(0, positions).to(device=consumer.device, dtype=torch.float32)
    labels = inputs["fit_labels"].index_select(0, positions).to(device=consumer.device)
    _require(torch.bincount(labels.detach().cpu(), minlength=10).tolist() == [1] * 10, "fast gradient batch is not one row from each class")
    phase = parameters["generator.phase"]
    group_gradients = {name: {key: torch.zeros_like(value) for key, value in parameters.items()} for name in TRAIN_EVALUATOR_NAMES}
    generator_objective_gradient = torch.zeros_like(phase)
    metric_names = ("raw_mse", "ssim_loss", "q_fg", "l_op", *[f"{name}_ce" for name in TRAIN_EVALUATOR_NAMES])
    totals = {name: torch.zeros((), dtype=torch.float64, device=consumer.device) for name in metric_names}
    semantic_gradient_norms = {name: {key: 0.0 for key in parameters} for name in TRAIN_EVALUATOR_NAMES}
    forward_count = 0
    for start in range(0, BATCH_SIZE, microbatch_size):
        end = min(start + microbatch_size, BATCH_SIZE)
        chunk_weight = (end - start) / BATCH_SIZE
        result = consumer.runtime.forward_from_image(images[start:end])
        forward_count += 1
        raw = result.raw_intensity
        target = result.image_input.slm_intensity
        terms = stroke.reconstruction_terms(raw, target, structural=True)
        qfg_value = gradient._qfg_components(raw, target)["q_fg"].mean()
        planes = diagnostic.bogt.contracted_plane_powers(result)
        l_op = diagnostic.bogt.optical_power_loss_from_powers(planes, consumer.references.power_reference)
        view = diagnostic.canonical.cmos_view(raw)
        ce_losses = {
            name: torch.nn.functional.cross_entropy(capacity._evaluator_logits(evaluators[name], view, name), labels[start:end])
            for name in TRAIN_EVALUATOR_NAMES
        }
        combined_objective = semantic.BETA * (terms["objective"] + qfg_lambda * qfg_value) + semantic.GAMMA * l_op
        generator_objective_gradient += torch.autograd.grad(combined_objective, phase, retain_graph=True)[0].detach() * chunk_weight
        for name_index, name in enumerate(TRAIN_EVALUATOR_NAMES):
            group = _autograd_group(ce_losses[name], parameters, retain_graph=name_index < len(TRAIN_EVALUATOR_NAMES) - 1)
            for key in parameters:
                group_gradients[name][key] += group[key] * chunk_weight
        totals["raw_mse"] += terms["raw_mse"].detach().to(torch.float64) * chunk_weight
        totals["ssim_loss"] += terms["ssim_loss"].detach().to(torch.float64) * chunk_weight
        totals["q_fg"] += qfg_value.detach().to(torch.float64) * chunk_weight
        totals["l_op"] += l_op.detach().to(torch.float64) * chunk_weight
        for name, loss in ce_losses.items():
            totals[f"{name}_ce"] += loss.detach().to(torch.float64) * chunk_weight
        del result, raw, target, terms, qfg_value, planes, l_op, view, ce_losses, combined_objective

    semantic_direction, unavailable = _canonical_semantic_direction(group_gradients)
    _require(semantic_direction is not None, f"three-evaluator semantic direction is unavailable: {unavailable}")
    for name in TRAIN_EVALUATOR_NAMES:
        semantic_gradient_norms[name] = {key: gradient._rms(value) for key, value in group_gradients[name].items()}
    for key in parameters:
        semantic_gradient_norms.setdefault("canonical", {})[key] = gradient._rms(semantic_direction[key])
    cached_z = ols_coefficients.index_select(0, positions).to(consumer.device)
    l_dist, _d_dist = capacity._dist_loss_and_ols_gradient(cached_z, consumer.references.scale)
    _require(all(parameter.grad is None for parameter in parameters.values()), "fast read-only gradient populated a model .grad field")
    combined = generator_objective_gradient + semantic.KAPPA * semantic_direction["generator.phase"]
    _require(bool(torch.isfinite(combined).all()), "fast combined Generator gradient is nonfinite")
    return combined.detach(), {
        "metrics": {**{name: float(value.cpu()) for name, value in totals.items()}, "l_dist": float(l_dist.detach().to(torch.float64).cpu())},
        "semantic_group_gradients": group_gradients,
        "semantic_gradient_norms": semantic_gradient_norms,
        "generator_objective_gradient": generator_objective_gradient.detach(),
        "forward_count": forward_count,
        "microbatch_size": microbatch_size,
    }


def _train_step(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, nn.Parameter],
    evaluators: Mapping[str, nn.Module],
    inputs: Mapping[str, Any],
    ols_coefficients: torch.Tensor,
    batch_order: Sequence[Sequence[int]],
    seed: int,
    arm: str,
    local_update: int,
    *,
    qfg_lambda: float | None = None,
) -> dict[str, Any]:
    _require(seed in TRAIN_SEEDS and arm in {"A", "B"} and 0 <= local_update < UPDATES_PER_ARM, "training cursor, seed, or arm is outside the fixed contract")
    positions = list(batch_order[local_update])
    effective_lambda = 0.0 if arm == "A" else FIXED_LAMBDA if qfg_lambda is None else float(qfg_lambda)
    _require(math.isfinite(effective_lambda) and effective_lambda >= 0.0, "effective Q_fg weight is invalid")
    before = parameters["generator.phase"].detach().clone()
    generator_gradient, detail = _fast_batch_gradient(consumer, parameters, evaluators, inputs, positions, ols_coefficients, effective_lambda)
    optimizer.zero_grad(set_to_none=True)
    parameters["generator.phase"].grad = generator_gradient.clone()
    _require(parameters["encoder.phase"].grad is None, "Encoder gradient entered the Generator-only optimizer")
    reservation_index = _reserve_engineering_update(inputs)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = int(float(optimizer.state[parameters["generator.phase"]]["step"].item()))
    _require(optimizer_step == SOURCE_STEP + local_update + 1, "Generator Adam step differs from A4700 plus the task-local update")
    displacement = parameters["generator.phase"].detach() - before
    _require(bool(torch.isfinite(displacement).all() and (displacement != 0).any()), "Generator phase update is invalid or zero")
    _complete_engineering_update(inputs, reservation_index)
    _require(all(parameter.grad is None for parameter in parameters.values()), "training left a parameter gradient populated")
    labels = inputs["fit_labels"].index_select(0, torch.tensor(positions)).tolist()
    _require(labels == list(range(10)), "saved training sequence is not class-balanced")
    metrics = detail["metrics"]
    return {
        "task_local_update": local_update + 1,
        "seed": seed,
        "arm": arm,
        "optimizer_step": optimizer_step,
        "epoch": local_update // BATCHES_PER_EPOCH,
        "batch_cursor": local_update % BATCHES_PER_EPOCH,
        "fit_positions": positions,
        "source_row_indices": [int(inputs["fit_rows"][position]) for position in positions],
        "labels": labels,
        "effective_qfg_lambda": effective_lambda,
        "raw_mse": metrics["raw_mse"],
        "ssim_loss": metrics["ssim_loss"],
        "q_fg": metrics["q_fg"],
        "l_op": metrics["l_op"],
        "l_dist": metrics["l_dist"],
        "ce_by_evaluator": {name: metrics[f"{name}_ce"] for name in TRAIN_EVALUATOR_NAMES},
        "semantic_gradient_norms": detail["semantic_gradient_norms"],
        "generator_gradient_rms": gradient._rms(generator_gradient),
        "generator_update_rms": gradient._rms(displacement),
        "production_forwards": detail["forward_count"],
        "finite": True,
    }


def _closed_form_adam_step(
    parameter_before: torch.Tensor,
    optimizer_state: Mapping[str, Any],
    gradient_value: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    groups = optimizer_state.get("param_groups", [])
    states = optimizer_state.get("state", {})
    _require(len(groups) == 1 and groups[0].get("params") == [0] and set(states) == {0}, "independent Adam reference received an unexpected optimizer layout")
    group = groups[0]
    _require(float(group.get("weight_decay", 0.0)) == 0.0 and not group.get("maximize", False), "independent Adam reference supports only the fixed no-decay minimization contract")
    beta1, beta2 = (float(value) for value in group["betas"])
    learning_rate, epsilon = float(group["lr"]), float(group["eps"])
    state = states[0]
    next_step = int(float(state["step"].item())) + 1
    old_mean = state["exp_avg"].to(device=gradient_value.device, dtype=torch.float64)
    old_square = state["exp_avg_sq"].to(device=gradient_value.device, dtype=torch.float64)
    grad = gradient_value.detach().to(torch.float64)
    mean = beta1 * old_mean + (1.0 - beta1) * grad
    square = beta2 * old_square + (1.0 - beta2) * grad.square()
    corrected_mean = mean / (1.0 - beta1**next_step)
    corrected_square = square / (1.0 - beta2**next_step)
    parameter = parameter_before.to(torch.float64) - learning_rate * corrected_mean / (corrected_square.sqrt() + epsilon)
    predicted_state = {
        "step": torch.tensor(float(next_step)),
        "exp_avg": mean.to(dtype=gradient_value.dtype),
        "exp_avg_sq": square.to(dtype=gradient_value.dtype),
    }
    return parameter.to(dtype=parameter_before.dtype), predicted_state


def _production_reference_batch_gradient(
    consumer: Any,
    parameters: Mapping[str, nn.Parameter],
    evaluators: Mapping[str, nn.Module],
    inputs: Mapping[str, Any],
    positions: Sequence[int],
    weight: float,
) -> torch.Tensor:
    images = inputs["fit_images"].index_select(0, torch.tensor(positions)).to(device=consumer.device, dtype=torch.float32)
    labels = inputs["fit_labels"].index_select(0, torch.tensor(positions)).to(device=consumer.device)
    base, _reconstruction_gradient, q_gradient, _detail = gradient._single_microbatch_batch_gradients(consumer, parameters, evaluators, images, labels)
    return gradient._candidate_generator_gradient(base, q_gradient, weight).detach()


def _verify_first_update_against_reference(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, nn.Parameter],
    evaluators: Mapping[str, nn.Module],
    inputs: Mapping[str, Any],
    ols_coefficients: torch.Tensor,
    positions: Sequence[int],
    seed: int,
    arm: str,
    local_update: int,
) -> dict[str, Any]:
    weight = 0.0 if arm == "A" else FIXED_LAMBDA
    reference_gradient = _production_reference_batch_gradient(consumer, parameters, evaluators, inputs, positions, weight)
    microbatch_gradient, _detail = _fast_batch_gradient(consumer, parameters, evaluators, inputs, positions, ols_coefficients, weight)
    gradient_match = torch.allclose(microbatch_gradient, reference_gradient, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL)
    _require(gradient_match, f"microbatch {inputs['microbatch_size']} first-step gradient differs from the original production objective beyond the registered tolerance")
    parameter_before = parameters["generator.phase"].detach().clone()
    optimizer_before = copy.deepcopy(capacity._cpu_tree(optimizer.state_dict()))
    expected_parameter, expected_state = _closed_form_adam_step(parameter_before, optimizer_before, microbatch_gradient)
    record = _train_step(consumer, optimizer, parameters, evaluators, inputs, ols_coefficients, inputs["batch_orders"][seed], seed, arm, local_update)
    actual_parameter = parameters["generator.phase"].detach()
    actual_state = optimizer.state[parameters["generator.phase"]]
    parameter_match = _float_tensor_comparison(expected_parameter, actual_parameter)
    mean_match = _float_tensor_comparison(expected_state["exp_avg"], actual_state["exp_avg"])
    square_match = _float_tensor_comparison(expected_state["exp_avg_sq"], actual_state["exp_avg_sq"])
    _require(parameter_match["passed"] and mean_match["passed"] and square_match["passed"], "actual candidate first Adam update differs from the independent closed-form Adam reference")
    reference_parameter, _reference_state = _closed_form_adam_step(parameter_before, optimizer_before, reference_gradient)
    return {
        "executed": True,
        "passed": True,
        "microbatch_size": inputs["microbatch_size"],
        "original_production_gradient_match": True,
        "original_gradient_max_abs_difference": float((microbatch_gradient.to(torch.float64) - reference_gradient.to(torch.float64)).abs().max().cpu()),
        "adam_closed_form_parameter": parameter_match,
        "adam_closed_form_exp_avg": mean_match,
        "adam_closed_form_exp_avg_sq": square_match,
        "actual_vs_original_gradient_update_max_abs_difference": float((actual_parameter.to(torch.float64) - reference_parameter.to(torch.float64)).abs().max().cpu()),
        "optimizer_step": record["optimizer_step"],
        "training_record": record,
    }


def _checkpoint_path(target: Path, seed: int, arm: str, local: int) -> Path:
    return target / "checkpoints" / f"seed_{seed}_arm_{arm.lower()}_step{SOURCE_STEP + local:04d}.pt"


def _checkpoint_payload(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    inputs: Mapping[str, Any],
    seed: int,
    arm: str,
    local: int,
    batch_order: Sequence[Sequence[int]],
    history: Sequence[Mapping[str, Any]],
    *,
    reason: str,
) -> dict[str, Any]:
    parameters = capacity._parameters(consumer)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "seed": seed,
        "arm": arm,
        "source_step": SOURCE_STEP,
        "task_local_updates": local,
        "completed_updates": SOURCE_STEP + local,
        "optimizer_step": int(float(optimizer.state[parameters["generator.phase"]]["step"].item())),
        "epoch": local // BATCHES_PER_EPOCH,
        "next_batch_cursor": local % BATCHES_PER_EPOCH,
        "batch_order": [list(batch) for batch in batch_order],
        "lambda": FIXED_LAMBDA,
        "numeric_mode": NUMERIC_MODE,
        "source_checkpoint": inputs["sources"]["source"]["checkpoint_path"].as_posix(),
        "source_row_indices": inputs["population"]["source_row_indices"],
        "trainable_parameter_state": {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()},
        "optimizer_state": copy.deepcopy(capacity._cpu_tree(optimizer.state_dict())),
        "frozen_runtime_state": capacity._cpu_tree(capacity._frozen_runtime_state(consumer)),
        "rng_state": capacity._rng_state(),
        "training_history": copy.deepcopy(list(history)),
        "checkpoint_reason": reason,
    }


def _save_checkpoint(target: Path, consumer: Any, optimizer: torch.optim.Optimizer, inputs: Mapping[str, Any], seed: int, arm: str, local: int, batch_order: Sequence[Sequence[int]], history: Sequence[Mapping[str, Any]], *, reason: str) -> Path:
    path = _checkpoint_path(target, seed, arm, local)
    payload = _checkpoint_payload(consumer, optimizer, inputs, seed, arm, local, batch_order, history, reason=reason)
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        keys = ("seed", "arm", "task_local_updates", "source_row_indices", "batch_order", "trainable_parameter_state", "optimizer_state", "rng_state", "training_history")
        _require(all(capacity._tree_equal(existing.get(key), payload.get(key)) for key in keys), f"existing checkpoint differs at {path.name}")
    else:
        _write_torch(path, payload)
    # Validate the serialized bytes through the same consumer used for explicit recovery.
    saved = _read_checkpoint_file(path, target, inputs, {seed: batch_order})
    exact_fields = (
        "trainable_parameter_state", "optimizer_state", "frozen_runtime_state", "rng_state",
        "batch_order", "next_batch_cursor", "training_history",
    )
    _require(all(capacity._tree_equal(saved.get(key), payload.get(key)) for key in exact_fields), f"saved checkpoint readback differs at {path.name}")
    _write_json(target / "training_history" / f"seed_{seed}_arm_{arm.lower()}.json", {"task_id": TASK_ID, "seed": seed, "arm": arm, "additional_updates": local, "records": list(history)}, replace=True)
    _rotate_verified_checkpoints(target, seed, arm, _checkpoint_policy(target))
    return path


def _rotate_verified_checkpoints(target: Path, seed: int, arm: str, policy: Mapping[str, Any]) -> None:
    if policy["keep_recent"] is not None:
        checkpoints = sorted(
            (target / "checkpoints").glob(f"seed_{seed}_arm_{arm.lower()}_step*.pt"),
            key=lambda item: int(item.stem.rsplit("step", 1)[1]),
        )
        terminal = [item for item in checkpoints if int(item.stem.rsplit("step", 1)[1]) == SOURCE_STEP + UPDATES_PER_ARM]
        recent = [item for item in checkpoints if item not in terminal][-int(policy["keep_recent"]) :]
        keep = {item.resolve() for item in [*recent, *terminal]}
        for old in checkpoints:
            if old.resolve() not in keep:
                old.unlink()


def _checkpoint_policy(target: Path) -> dict[str, Any]:
    fixed_path = target / "fixed_parameters.json"
    fixed = json.loads(fixed_path.read_text(encoding="utf-8")) if fixed_path.is_file() else {}
    saved = fixed.get("checkpoint_policy")
    if saved is None:
        return {"interval": 100, "keep_recent": None, "source": "legacy_default"}
    _require(
        isinstance(saved, Mapping)
        and saved.get("interval") == CHECKPOINT_INTERVAL
        and saved.get("keep_recent") == CHECKPOINT_KEEP_RECENT,
        "registered checkpoint policy is invalid",
    )
    return {"interval": int(saved["interval"]), "keep_recent": int(saved["keep_recent"]), "source": "registered"}


def _validate_checkpoint(payload: Mapping[str, Any], path: Path, target: Path, inputs: Mapping[str, Any], batch_orders: Mapping[int, Sequence[Sequence[int]]]) -> None:
    _require(path.is_file() and target.resolve() in path.resolve().parents, "resume checkpoint must be an explicit file inside this task output")
    _require(payload.get("schema") == CHECKPOINT_SCHEMA and payload.get("task_id") == TASK_ID, "resume checkpoint belongs to another task")
    seed, arm, local = payload.get("seed"), payload.get("arm"), payload.get("task_local_updates")
    _require(seed in TRAIN_SEEDS and arm in {"A", "B"} and type(local) is int and 0 <= local <= UPDATES_PER_ARM, "resume seed/arm/update cursor is invalid")
    _require(payload.get("source_step") == SOURCE_STEP and payload.get("completed_updates") == SOURCE_STEP + local and payload.get("optimizer_step") == SOURCE_STEP + local, "resume source or Adam step differs")
    _require(payload.get("lambda") == FIXED_LAMBDA and payload.get("numeric_mode") == NUMERIC_MODE, "resume fixed weight or numeric mode differs")
    _require(payload.get("source_checkpoint") == inputs["sources"]["source"]["checkpoint_path"].as_posix() and payload.get("source_row_indices") == inputs["population"]["source_row_indices"], "resume source identity or population differs")
    _require(payload.get("batch_order") == [list(batch) for batch in batch_orders[seed]], "resume batch order differs from the registered seed trajectory")
    _require(payload.get("epoch") == local // BATCHES_PER_EPOCH and payload.get("next_batch_cursor") == local % BATCHES_PER_EPOCH, "resume epoch or next batch cursor differs")
    _require(path.resolve() == _checkpoint_path(target, seed, arm, local).resolve(), "resume checkpoint name does not identify its seed, arm, and update position")
    history = payload.get("training_history", [])
    _require(len(history) == local and all(row.get("task_local_update") == index + 1 and row.get("seed") == seed and row.get("arm") == arm for index, row in enumerate(history)), "resume training history is incomplete or mismatched")
    parameters = payload.get("trainable_parameter_state", {})
    _require(set(parameters) == {"encoder.phase", "generator.phase"} and tuple(parameters["encoder.phase"].shape) == (2, 1, 100, 100) and tuple(parameters["generator.phase"].shape) == (340_000,), "resume phase state geometry differs")
    _require(all(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) for value in parameters.values()), "resume phase state is invalid")
    optimizer = payload.get("optimizer_state", {})
    groups, states = optimizer.get("param_groups", []), optimizer.get("state", {})
    _require(len(groups) == 1 and groups[0].get("params") == [0] and set(states) == {0}, "resume Adam mapping differs")
    slot = states[0]
    _require(int(float(slot["step"].item())) == SOURCE_STEP + local and tuple(slot["exp_avg"].shape) == (340_000,) and tuple(slot["exp_avg_sq"].shape) == (340_000,), "resume Adam state step or geometry differs")
    _require(bool(torch.isfinite(slot["exp_avg"]).all() and torch.isfinite(slot["exp_avg_sq"]).all()), "resume Adam moments are nonfinite")
    _require(isinstance(payload.get("rng_state"), Mapping), "resume checkpoint lacks RNG state")


def _read_checkpoint_file(path: Path, target: Path, inputs: Mapping[str, Any], batch_orders: Mapping[int, Sequence[Sequence[int]]]) -> dict[str, Any]:
    _require(path.is_file(), f"explicit checkpoint file is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    _validate_checkpoint(payload, path, target, inputs, batch_orders)
    return payload


def _validate_explicit_resume_is_latest(target: Path, resume_path: Path, payload: Mapping[str, Any]) -> None:
    seed, arm = int(payload["seed"]), str(payload["arm"])
    candidates = list((target / "checkpoints").glob(f"seed_{seed}_arm_{arm.lower()}_step*.pt"))
    _require(bool(candidates), "explicit resume checkpoint is not in the registered trajectory")
    latest = max(candidates, key=lambda candidate: int(candidate.stem.rsplit("step", 1)[1]))
    _require(resume_path.resolve() == latest.resolve(), "explicit resume checkpoint is older than a saved checkpoint for this trajectory")


def _restore_checkpoint(payload: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    parameters = capacity._parameters(consumer)
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(payload["trainable_parameter_state"][name].to(device=parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == payload["optimizer_step"], "restored Adam step differs")
    capacity._assert_state_equal(payload["frozen_runtime_state"], capacity._frozen_runtime_state(consumer), "restored frozen runtime")
    capacity._restore_rng_state(payload["rng_state"])


def _engineering_checkpoint_root(target: Path, group: str, key: str) -> Path:
    return target / "engineering_checkpoints" / group / key


def _engineering_update_stage_name(group: str, key: str, local_update: int) -> str:
    _require(1 <= local_update <= 3, "engineering update stage cursor must be between one and three")
    return f"{group}_{key}_update_{local_update:02d}"


def _engineering_update_prefix(
    target: Path,
    inputs: Mapping[str, Any],
    group: str,
    key: str,
    seed: int,
    arm: str,
    microbatch_size: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    stages: list[dict[str, Any]] = []
    checkpoint_root = _engineering_checkpoint_root(target, group, key)
    budget_path_value = inputs.get("engineering_budget_file")
    _require(budget_path_value is not None, "partial engineering update recovery requires the task-wide budget")
    budget = _read_engineering_budget(Path(budget_path_value))
    if group == "trajectory":
        expected_phase = "six_seed_arm_three_step_updates"
        repeat = None
    elif group == "zero_lambda":
        expected_phase = "zero_lambda_negative_control"
        repeat = None
    elif group.startswith("continuous_repeat_"):
        expected_phase = "continuous_repeat"
        repeat = int(group.rsplit("_", 1)[1])
    else:
        raise CloudForegroundError(f"unknown resumable engineering update group: {group}")
    attempt_dir = target.resolve().relative_to(_root().resolve()).as_posix()
    for local_update in range(1, 4):
        phase = _engineering_update_stage_name(group, key, local_update)
        stage = _read_engineering_stage(target, phase)
        if stage is None:
            checkpoint_path = _checkpoint_path(checkpoint_root, seed, arm, local_update)
            matching = [
                row for row in budget["reservations"]
                if row.get("origin") == "current_engineering_diagnostic"
                and row.get("context", {}).get("engineering_attempt_dir") == attempt_dir
                and row.get("context", {}).get("phase") == expected_phase
                and row.get("context", {}).get("seed") == seed
                and row.get("context", {}).get("arm") == arm
                and row.get("context", {}).get("task_local_update") == local_update
                and row.get("context", {}).get("microbatch_size") == microbatch_size
                and (repeat is None or row.get("context", {}).get("repeat") == repeat)
            ]
            _require(len(matching) <= 1, f"multiple budget reservations identify one diagnostic update: {group}/{key}/{local_update}")
            if not checkpoint_path.is_file():
                _require(not matching, f"engineering update reservation has no reusable checkpoint or stage: {group}/{key}/{local_update}")
                break
            _require(len(matching) == 1 and matching[0].get("status") == "optimizer_step_completed", f"partial checkpoint lacks one completed budget reservation: {group}/{key}/{local_update}")
            payload = _read_checkpoint_file(checkpoint_path, checkpoint_root, inputs, inputs["batch_orders"])
            _require(payload.get("task_local_updates") == local_update, f"partial checkpoint cursor differs: {group}/{key}/{local_update}")
            training_record = copy.deepcopy(payload["training_history"][-1])
            first_update_verification = training_record.pop("engineering_first_update_verification", None)
            if group == "trajectory" and local_update == 1:
                _require(isinstance(first_update_verification, Mapping) and first_update_verification.get("passed") is True, "first trajectory update lacks its independent gradient/Adam verification")
            checkpoint_retention = (
                "preserve_first_step_control" if group == "trajectory" and arm == "A" and local_update == 1
                else "preserve_audit" if group == "zero_lambda"
                else "keep_recent_two_after_readback"
            )
            value = {
                "status": "COMPLETE",
                "seed": seed,
                "arm": arm,
                "task_local_update": local_update,
                "optimizer_step": payload["optimizer_step"],
                "batch_cursor": payload["next_batch_cursor"],
                "microbatch_size": microbatch_size,
                "checkpoint": checkpoint_path.relative_to(target).as_posix(),
                "checkpoint_root": checkpoint_root.relative_to(target).as_posix(),
                "checkpoint_retention": checkpoint_retention,
                "checkpoint_readback_passed": True,
                "training_record": payload["training_history"][-1],
            }
            if first_update_verification is not None:
                value["first_update_verification"] = first_update_verification
            _write_engineering_stage(target, phase, value)
            stage = _read_engineering_stage(target, phase)
            _require(stage is not None, f"recovered checkpoint stage did not persist: {phase}")
        _require(
            stage.get("seed") == seed
            and stage.get("arm") == arm
            and stage.get("task_local_update") == local_update
            and stage.get("microbatch_size") == microbatch_size,
            f"engineering update stage identity differs: {phase}",
        )
        stages.append(stage)
    _require(
        not any(_engineering_stage_path(target, _engineering_update_stage_name(group, key, index)).is_file() for index in range(len(stages) + 2, 4)),
        f"engineering update stages contain a gap: {group}/{key}",
    )
    if not stages:
        return None, [], None
    last_stage = stages[-1]
    checkpoint_path = (target / str(last_stage["checkpoint"])).resolve()
    checkpoint_root = (target / str(last_stage["checkpoint_root"])).resolve()
    _require(target.resolve() in checkpoint_path.parents and checkpoint_root.resolve() in checkpoint_path.parents, "engineering resume checkpoint path escapes its registered scope")
    payload = _read_checkpoint_file(checkpoint_path, checkpoint_root, inputs, inputs["batch_orders"])
    _require(
        payload.get("task_local_updates") == len(stages)
        and len(payload.get("training_history", [])) == len(stages)
        and payload.get("seed") == seed
        and payload.get("arm") == arm,
        f"engineering resume checkpoint does not contain the complete staged prefix: {group}/{key}",
    )
    for index, stage in enumerate(stages):
        _require(
            capacity._tree_equal(stage.get("training_record"), payload["training_history"][index]),
            f"engineering stage training record differs from the checkpoint prefix: {group}/{key}/{index + 1}",
        )
    return payload, [dict(stage) for stage in stages], last_stage


def _save_engineering_update(
    target: Path,
    checkpoint_root: Path,
    inputs: Mapping[str, Any],
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    seed: int,
    arm: str,
    local_update: int,
    batch_order: Sequence[Sequence[int]],
    history: Sequence[Mapping[str, Any]],
    *,
    group: str,
    key: str,
    microbatch_size: int,
    first_update_verification: Mapping[str, Any] | None = None,
    checkpoint_retention: str = "keep_recent_two_after_readback",
) -> tuple[Path, dict[str, Any]]:
    path = _save_checkpoint(
        checkpoint_root,
        consumer,
        optimizer,
        inputs,
        seed,
        arm,
        local_update,
        batch_order,
        history,
        reason=f"engineering_{group}_{local_update}_update_boundary",
    )
    saved = _read_checkpoint_file(path, checkpoint_root, inputs, inputs["batch_orders"])
    _require(
        saved.get("task_local_updates") == local_update
        and len(saved.get("training_history", [])) == local_update
        and capacity._tree_equal(saved.get("training_history"), list(history)),
        f"engineering update checkpoint did not pass serialized production readback: {group}/{key}/{local_update}",
    )
    value = {
        "status": "COMPLETE",
        "seed": seed,
        "arm": arm,
        "task_local_update": local_update,
        "optimizer_step": saved["optimizer_step"],
        "batch_cursor": saved["next_batch_cursor"],
        "microbatch_size": microbatch_size,
        "checkpoint": path.relative_to(target).as_posix(),
        "checkpoint_root": checkpoint_root.relative_to(target).as_posix(),
        "checkpoint_retention": checkpoint_retention,
        "checkpoint_readback_passed": True,
        "training_record": copy.deepcopy(dict(history[-1])),
    }
    if first_update_verification is not None:
        value["first_update_verification"] = copy.deepcopy(dict(first_update_verification))
    phase = _engineering_update_stage_name(group, key, local_update)
    stage_path = _write_engineering_stage(target, phase, value)
    _require(_json(stage_path) == {"task_id": TASK_ID, "phase": phase, "evidence_identity": {
        "input_population": "input_population.json",
        "batch_orders": "batch_orders.json",
        "run_config": "engineering_run_config.json",
    }, **value}, f"engineering update stage did not read back: {phase}")
    if checkpoint_retention == "keep_recent_two_after_readback":
        checkpoints = sorted(
            checkpoint_root.joinpath("checkpoints").glob(f"seed_{seed}_arm_{arm.lower()}_step*.pt"),
            key=lambda item: int(item.stem.rsplit("step", 1)[1]),
        )
        keep = {item.resolve() for item in checkpoints[-2:]}
        first_stage = _engineering_stage_path(target, _engineering_update_stage_name(group, key, 1))
        if first_stage.is_file():
            first_record = _json(first_stage)
            if first_record.get("checkpoint_retention") == "preserve_first_step_control":
                first_checkpoint = (target / str(first_record["checkpoint"])).resolve()
                keep.add(first_checkpoint)
        for old in checkpoints:
            if old.resolve() not in keep:
                old.unlink()
    return path, saved


def _numpy_qfg(raw: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    image = raw.detach().cpu().numpy().astype(np.float64).reshape(raw.shape[0], -1)
    truth = target.detach().cpu().numpy().astype(np.float64).reshape(target.shape[0], -1)
    image_mass = image.sum(axis=1, keepdims=True)
    target_mass = truth.sum(axis=1, keepdims=True)
    _require(np.isfinite(image).all() and np.isfinite(truth).all() and (image_mass > 0).all() and (target_mass > 0).all(), "NumPy Q_fg reference received invalid masses")
    p_image = image / image_mass
    p_target = truth / target_mass
    mask = truth > 0.1
    denominator = np.square(p_target).sum(axis=1)
    _require((mask.any(axis=1) & (~mask).any(axis=1) & (denominator > 0)).all(), "NumPy Q_fg reference has an empty mask or denominator")
    return (np.square(p_image - p_target) * mask).sum(axis=1) / denominator


def _local_readout(consumer: Any, evaluators: Mapping[str, nn.Module], images: torch.Tensor) -> dict[str, torch.Tensor]:
    pieces: dict[str, list[torch.Tensor]] = {key: [] for key in ("raw_cmos", "target_slm_intensity", "evaluator_view", "z")}
    logits = {name: [] for name in EVALUATOR_NAMES}
    with torch.inference_mode():
        for index in range(images.shape[0]):
            result = consumer.runtime.forward_from_image(images[index : index + 1].to(device=consumer.device, dtype=torch.float32))
            view = diagnostic.canonical.cmos_view(result.raw_intensity)
            pieces["raw_cmos"].append(result.raw_intensity.detach().cpu())
            pieces["target_slm_intensity"].append(result.image_input.slm_intensity.detach().cpu())
            pieces["evaluator_view"].append(view.detach().cpu())
            pieces["z"].append(result.z_ols.coefficients.detach().cpu())
            for name, values in logits.items():
                values.append(capacity._evaluator_logits(evaluators[name], view, name).detach().cpu())
            del result, view
    return {**{key: torch.cat(values) for key, values in pieces.items()}, **{f"{name.lower()}_logits": torch.cat(values) for name, values in logits.items()}}


def _cuda_resources(target: Path, *, estimated_output_bytes: int = 4_500_000_000) -> dict[str, Any]:
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized production preflight")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    disk_root = target
    while not disk_root.exists():
        disk_root = disk_root.parent
    disk_free = shutil.disk_usage(disk_root).free
    estimated_output = int(estimated_output_bytes)
    _require(estimated_output > 0, "CUDA preflight output estimate must be positive")
    _require(disk_free >= estimated_output + 1_000_000_000, f"output estimate plus reserve does not fit: free={disk_free}, required={estimated_output + 1_000_000_000}")
    return {
        "executed": True,
        "passed": True,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "cuda_total_bytes": int(total_bytes),
        "cuda_free_bytes_observation": int(free_bytes),
        "torch_allocated_bytes_observation": int(torch.cuda.memory_allocated(device)),
        "torch_reserved_bytes_observation": int(torch.cuda.memory_reserved(device)),
        "gpu_quota": None,
        "disk_root": disk_root.as_posix(),
        "disk_free_bytes": int(disk_free),
        "estimated_remaining_output_bytes": estimated_output,
        "required_post_output_reserve_bytes": 1_000_000_000,
    }


def _device_preflight(inputs: dict[str, Any], target: Path, source_args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda:0")
    sources = inputs["sources"]
    evaluators = generation.diagnostic.load_evaluators(sources["source"]["generation_paths"], device)
    checks: dict[str, Any] = {}
    engineering_estimate = 256 * 1024 * 1024 if source_args.engineering_check_only else 4_500_000_000
    resource = _cuda_resources(target, estimated_output_bytes=engineering_estimate)
    checks["cuda_and_disk"] = resource

    consumer, optimizer, parameters = _new_endpoint(sources, device)
    _require(len(evaluators) == 4 and set(evaluators) == set(EVALUATOR_NAMES), "all four frozen evaluators were not loaded")
    _require(len(optimizer.param_groups[0]["params"]) == 1 and optimizer.param_groups[0]["params"][0] is parameters["generator.phase"], "optimizer contains a non-Generator parameter")
    _require(all(parameter is not parameters["encoder.phase"] for parameter in optimizer.param_groups[0]["params"]), "Encoder phase entered Generator Adam")
    checks["source_model_and_optimizer"] = {
        "executed": True,
        "passed": True,
        "checkpoint_step": SOURCE_STEP,
        "adam_step": int(float(optimizer.state[parameters["generator.phase"]]["step"].item())),
        "generator_parameters": parameters["generator.phase"].numel(),
        "encoder_requires_grad_for_semantics": parameters["encoder.phase"].requires_grad,
        "optimizer_names": ["generator.phase"],
        "propagation_grid": consumer.runtime.config.propagation_grid_size,
        "strict_fp32": True,
    }

    representative = inputs["fit_images"][:BATCH_SIZE]
    first = _local_readout(consumer, evaluators, representative)
    second_consumer, _second_optimizer, _second_parameters = _new_endpoint(sources, device)
    second = _local_readout(second_consumer, evaluators, representative)
    repeated_fields = ("raw_cmos", "target_slm_intensity", "evaluator_view", "z", "e_s_logits", "e_i_logits", "e_c_logits", "e_d_logits")
    repeat_diffs = {}
    for field in repeated_fields:
        left, right = first[field], second[field]
        _require(left.shape == right.shape and torch.allclose(left, right, rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"cloud-local repeated source replay differs for {field}")
        repeat_diffs[field] = float((left.to(torch.complex128 if left.is_complex() else torch.float64) - right.to(torch.complex128 if right.is_complex() else torch.float64)).abs().max())
    checks["cloud_local_replay"] = {"executed": True, "passed": True, "rows": BATCH_SIZE, "rtol": REPLAY_RTOL, "atol": REPLAY_ATOL, "max_abs_difference": repeat_diffs, "historical_windows_replay_status": "REPLAY_MISMATCH_ENGINEERING_REVIEW_REQUIRED; preserved and not used as a cloud-local gate"}
    del second_consumer, _second_optimizer, _second_parameters

    with torch.enable_grad():
        raw = first["raw_cmos"][:1].to(device=device, dtype=torch.float64).requires_grad_(True)
        target_raw = first["target_slm_intensity"][:1].to(device=device, dtype=torch.float64)
        value = gradient._qfg_components(raw, target_raw)["q_fg"].sum()
        autograd_image = torch.autograd.grad(value, raw)[0]
        analytic_image = gradient._qfg_gradient_reference(raw.detach(), target_raw)
        numpy_values = _numpy_qfg(first["raw_cmos"][:1], first["target_slm_intensity"][:1])
        torch_values = gradient._qfg_components(first["raw_cmos"][:1], first["target_slm_intensity"][:1])["q_fg"].detach().cpu().numpy()
        image_error = float((autograd_image - analytic_image).abs().max().cpu())
        _require(np.allclose(numpy_values, torch_values, rtol=1.0e-12, atol=1.0e-14), "independent NumPy Q_fg value differs from production float64 calculation")
        _require(image_error <= 1.0e-10, f"Q_fg analytic image derivative differs from autograd: {image_error}")
    image = inputs["fit_images"][:1].to(device=device, dtype=torch.float32)
    q_value, phase_gradient = gradient._qfg_phase_gradient(consumer, parameters, image)
    direction = gradient._unit_descent(phase_gradient, "cloud Q_fg phase reference")
    fd_checks = [gradient._check_fd(consumer, parameters, image, direction, epsilon=value) for value in (historical_training.FD_EPSILON, historical_training.FD_EPSILON / 2)]
    _require(all(record["passed"] for record in fd_checks), "Q_fg phase central-difference check failed")
    checks["qfg_independent_value_gradient"] = {
        "executed": True, "passed": True, "torch_numpy_qfg_max_abs_difference": float(np.max(np.abs(numpy_values - torch_values))),
        "analytic_image_gradient_max_abs_difference": image_error, "qfg_at_source": q_value, "phase_center_difference": fd_checks,
        "whole_image_mass_normalization_included": True,
    }
    cache = _ols_cache(consumer, inputs, target)
    inputs["ols_coefficients"] = cache["coefficients"]
    inputs["fit_target_shape_vectors"] = cache["target_shape_vectors"]
    cache_check = _verify_cached_ols(consumer, inputs, cache)
    checks["cached_ols_source_recheck"] = cache_check
    priors = _fit_prior(cache["coefficients"], inputs["fit_labels"])
    sample_bank = _sample_bank(priors)
    inputs["priors"] = priors
    inputs["sample_bank"] = sample_bank
    _write_torch_exact(target / "conditional_sample_bank.pt", sample_bank)
    _write_torch_exact(target / "fitted_joint_prior.pt", priors)
    checks["fixed_prior_and_sample_bank"] = {
        "executed": True, "passed": True, "fit_rows": FIT_ROWS, "rows_per_class": FIT_PER_CLASS,
        "dtype": "CPU float64 fit, real/imag interleaved", "joint_weight": 0.5, "within_port_block_weight": 0.5,
        "minimum_eigenvalue": float(priors["minimum_eigenvalue"].min()), "max_cholesky_reconstruction_error": float(priors["reconstruction_error"].max()),
        "sample_count": len(SAMPLE_SEEDS) * 10 * SAMPLES_PER_CLASS, "sample_seeds": list(SAMPLE_SEEDS),
    }

    schedule = inputs["batch_orders"]
    representative_batch = schedule[TRAIN_SEEDS[0]][0]
    images = inputs["fit_images"].index_select(0, torch.tensor(representative_batch)).to(device)
    labels = inputs["fit_labels"].index_select(0, torch.tensor(representative_batch)).to(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    reference_started = time.perf_counter()
    reference_base, _reconstruction, reference_q, reference_detail = gradient._single_microbatch_batch_gradients(consumer, parameters, evaluators, images, labels)
    torch.cuda.synchronize(device)
    measured_reference_seconds = time.perf_counter() - reference_started
    measured_reference_peak = int(torch.cuda.max_memory_allocated(device))
    reference_candidate = gradient._candidate_generator_gradient(reference_base, reference_q, FIXED_LAMBDA)
    saved_performance_stage = _read_engineering_stage(target, "microbatch_performance") if source_args.engineering_check_only else None
    saved_performance = saved_performance_stage.get("measurement") if saved_performance_stage is not None else None
    if saved_performance is not None:
        _require(
            saved_performance_stage.get("requested_microbatch_size") == int(source_args.microbatch_size)
            and saved_performance.get("selected_microbatch_size") == int(source_args.microbatch_size),
            "saved microbatch timing stage uses a different requested configuration",
        )
        reference_timing = copy.deepcopy(saved_performance["original_production_arm_timing"])
        reference_seconds = float(saved_performance["reference_seconds_single_observation"])
        reference_peak = int(saved_performance["reference_peak_allocated_bytes"])
    else:
        reference_seconds = measured_reference_seconds
        reference_peak = measured_reference_peak
        reference_timing = {}
        for arm, weight in (("A", 0.0), ("B", FIXED_LAMBDA)):
            torch.cuda.reset_peak_memory_stats(device)
            for _ in range(2):
                _production_reference_batch_gradient(consumer, parameters, evaluators, inputs, representative_batch, weight)
            timings = []
            for _ in range(5):
                torch.cuda.synchronize(device)
                timed_started = time.perf_counter()
                _production_reference_batch_gradient(consumer, parameters, evaluators, inputs, representative_batch, weight)
                torch.cuda.synchronize(device)
                timings.append(time.perf_counter() - timed_started)
            reference_timing[arm] = {"warmups": 2, "timed_runs": timings, "median_seconds": float(np.median(timings)), "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device))}
    reference_mean_median = math.fsum(row["median_seconds"] for row in reference_timing.values()) / 2.0
    selected_microbatch = int(inputs.get("microbatch_size", 1))
    _require(selected_microbatch in {1, 2, 5, 10}, "selected microbatch size is invalid")
    candidate_results: list[dict[str, Any]] = []
    reference_details: dict[float, tuple[torch.Tensor, dict[str, Any]]] = {}
    saved_candidates = {
        int(row["microbatch_size"]): copy.deepcopy(row)
        for row in (saved_performance or {}).get("microbatch_candidates", [])
    }
    for size in (1, 2, 5, 10):
        inputs["microbatch_size"] = size
        by_lambda: dict[float, tuple[torch.Tensor, dict[str, Any]]] = {}
        try:
            by_lambda = {
                weight: _fast_batch_gradient(consumer, parameters, evaluators, inputs, representative_batch, cache["coefficients"], weight)
                for weight in (0.0, FIXED_LAMBDA)
            }
            if size == 1:
                reference_details = by_lambda
            for weight, (candidate_gradient, candidate_detail) in by_lambda.items():
                reference_gradient, reference_detail_fast = reference_details[weight]
                _require(torch.allclose(candidate_gradient, reference_gradient, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL), f"microbatch {size} combined gradient differs from microbatch 1")
                _require(torch.allclose(candidate_detail["generator_objective_gradient"], reference_detail_fast["generator_objective_gradient"], rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL), f"microbatch {size} objective gradient differs from microbatch 1")
                for evaluator_name in TRAIN_EVALUATOR_NAMES:
                    for group_name in ("encoder.phase", "generator.phase"):
                        _require(torch.allclose(candidate_detail["semantic_group_gradients"][evaluator_name][group_name], reference_detail_fast["semantic_group_gradients"][evaluator_name][group_name], rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL), f"microbatch {size} {evaluator_name} {group_name} semantic gradient differs from microbatch 1")
                for metric_name in ("raw_mse", "ssim_loss", "q_fg", "l_op", "l_dist"):
                    _require(math.isclose(candidate_detail["metrics"][metric_name], reference_detail_fast["metrics"][metric_name], rel_tol=GRADIENT_RTOL, abs_tol=GRADIENT_ATOL), f"microbatch {size} {metric_name} differs from microbatch 1")
            saved_candidate = saved_candidates.get(size)
            if saved_candidate is not None:
                _require(saved_candidate.get("equivalence_passed") is True, f"saved microbatch {size} candidate did not pass its original equivalence stage")
                candidate_results.append({**saved_candidate, "actual_update_verified": False})
            else:
                torch.cuda.reset_peak_memory_stats(device)
                arm_timing: dict[str, dict[str, Any]] = {}
                for arm, weight in (("A", 0.0), ("B", FIXED_LAMBDA)):
                    for _ in range(2):
                        _fast_batch_gradient(consumer, parameters, evaluators, inputs, representative_batch, cache["coefficients"], weight)
                    timings = []
                    for _ in range(5):
                        torch.cuda.synchronize(device)
                        timed_started = time.perf_counter()
                        _fast_batch_gradient(consumer, parameters, evaluators, inputs, representative_batch, cache["coefficients"], weight)
                        torch.cuda.synchronize(device)
                        timings.append(time.perf_counter() - timed_started)
                    arm_timing[arm] = {"warmups": 2, "timed_runs": timings, "median_seconds": float(np.median(timings))}
                median_seconds = math.fsum(row["median_seconds"] for row in arm_timing.values()) / 2.0
                candidate_results.append({"microbatch_size": size, "equivalence_passed": True, "arm_timing": arm_timing, "median_seconds": median_seconds, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)), "forward_calls": by_lambda[FIXED_LAMBDA][1]["forward_count"], "actual_update_verified": False})
        except (CloudForegroundError, torch.cuda.OutOfMemoryError) as error:
            candidate_results.append({"microbatch_size": size, "equivalence_passed": False, "reason": str(error), "warmups": 0, "timed_runs": [], "actual_update_verified": False})
            torch.cuda.empty_cache()
        del by_lambda
    passed_candidates = [row for row in candidate_results if row["equivalence_passed"]]
    _require(bool(passed_candidates) and any(row["microbatch_size"] == 1 for row in passed_candidates), "microbatch-1 production reference did not pass equivalence")
    size1_time = next(row["median_seconds"] for row in passed_candidates if row["microbatch_size"] == 1)
    inputs["microbatch_size"] = selected_microbatch
    fast_a, fast_a_detail = _fast_batch_gradient(consumer, parameters, evaluators, inputs, representative_batch, cache["coefficients"], 0.0)
    fast_b, fast_b_detail = _fast_batch_gradient(consumer, parameters, evaluators, inputs, representative_batch, cache["coefficients"], FIXED_LAMBDA)
    base_error = float((fast_a.to(torch.float64) - reference_base.to(torch.float64)).abs().max().cpu())
    candidate_error = float((fast_b.to(torch.float64) - reference_candidate.to(torch.float64)).abs().max().cpu())
    _require(torch.allclose(fast_a, reference_base, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL), "merged original objective gradient differs from its production reference")
    _require(torch.allclose(fast_b, reference_candidate, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL), "merged fixed-Q_fg gradient differs from the production reference")
    dot = float(torch.dot(fast_b.flatten().to(torch.float64), reference_candidate.flatten().to(torch.float64)).cpu())
    _require(math.isfinite(dot) and dot > 0.0, "optimized gradient changed the production update direction sign")
    selected_timing = next(row for row in passed_candidates if row["microbatch_size"] == selected_microbatch)
    fast_seconds = selected_timing["median_seconds"]
    fast_peak = selected_timing["peak_allocated_bytes"]
    group_gradients = fast_b_detail["semantic_group_gradients"]
    without_encoder = {name: {key: value.clone() for key, value in fields.items()} for name, fields in group_gradients.items()}
    for fields in without_encoder.values():
        fields["encoder.phase"].zero_()
    encoder_omitted_direction, omitted_reason = _canonical_semantic_direction(without_encoder)
    _require(encoder_omitted_direction is not None, f"encoder-gradient negative control was unavailable: {omitted_reason}")
    full_direction, full_reason = _canonical_semantic_direction(group_gradients)
    _require(full_direction is not None, f"full semantic direction was unavailable: {full_reason}")
    encoder_effect = float((encoder_omitted_direction["generator.phase"] - full_direction["generator.phase"]).abs().max().cpu())
    _require(encoder_effect > 0.0, "dropping Encoder semantic gradients did not alter the normalized Generator semantic direction")
    speed_ratio = reference_seconds / fast_seconds if fast_seconds > 0 else math.inf
    checks["merged_production_gradient_and_speed"] = {
        "executed": True, "passed": True, "same_batch": True, "reference_forwards": 2 * BATCH_SIZE, "optimized_forwards": fast_b_detail["forward_count"],
        "reference_seconds_single_observation": reference_seconds,
        "optimized_seconds_single_observation": fast_seconds,
        "observed_speed_ratio_single_observation": speed_ratio,
        "original_production_arm_timing": reference_timing,
        "original_production_equal_arm_mean_median_seconds": reference_mean_median,
        "reference_peak_allocated_bytes": reference_peak, "optimized_peak_allocated_bytes": fast_peak,
        "A_max_abs_gradient_difference": base_error, "B_max_abs_gradient_difference": candidate_error,
        "rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL, "update_direction_dot": dot,
        "encoder_semantic_omission_changes_generator_direction": True, "omission_max_abs_change": encoder_effect,
        "Encoder_autograd_enabled": parameters["encoder.phase"].requires_grad,
        "microbatch_candidates": candidate_results,
        "selected_microbatch_size": selected_microbatch,
        "recommended_microbatch_size": None,
        "recommendation_rule": "after first-step actual-update verification, choose the fastest candidate equivalent for both arms and within both production consumers only when its equal-weighted A/B median is at least 10 percent faster than microbatch 1; recommendation does not authorize a scientific run",
    }
    if source_args.engineering_check_only:
        early_table = target / "microbatch_performance_pre_updates.csv"
        if saved_performance_stage is None:
            _write_microbatch_performance_table(early_table, checks["merged_production_gradient_and_speed"])
            _write_engineering_stage(target, "microbatch_performance", {
                "status": "COMPLETE",
                "passed": True,
                "before_real_updates": True,
                "requested_microbatch_size": int(source_args.microbatch_size),
                "performance_table": early_table.relative_to(target).as_posix(),
                "candidate_count": len(candidate_results),
                "measurement": checks["merged_production_gradient_and_speed"],
            })
        else:
            _require(early_table.is_file(), "saved microbatch timing stage lost its performance table")
        _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", "microbatch_performance")
    _require(all(value.get("executed") is True and value.get("passed") is True for value in checks.values()), "preflight contains an unexecuted or failed check")

    evaluator_states = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    trials = []
    first_a_state: dict[int, dict[str, Any]] = {}
    continuous_seed_zero_a: dict[str, Any] | None = None
    candidate_by_size = {int(row["microbatch_size"]): row for row in candidate_results}
    requested_sizes = (1, 2, 5, 10, 1, 1)
    a_microbatch_by_seed: dict[int, int] = {}
    trajectory_index = 0
    for seed in TRAIN_SEEDS:
        for arm in ("A", "B"):
            requested_size = requested_sizes[trajectory_index]
            trajectory_index += 1
            candidate_record = candidate_by_size[requested_size]
            actual_size = requested_size if candidate_record["equivalence_passed"] else 1
            inputs["microbatch_size"] = actual_size
            trial_consumer, trial_optimizer, trial_parameters = _new_endpoint(sources, device)
            checkpoint_key = f"seed_{seed}_arm_{arm.lower()}"
            checkpoint_root = _engineering_checkpoint_root(target, "trajectory", checkpoint_key)
            if source_args.engineering_check_only:
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                prefix_payload, prefix_stages, _last_prefix_stage = _engineering_update_prefix(
                    target, inputs, "trajectory", checkpoint_key, seed, arm, actual_size,
                )
            else:
                prefix_payload, prefix_stages = None, []
            if prefix_payload is None:
                capacity._seed_all(seed)
                history: list[dict[str, Any]] = []
                local_start = 0
                update_verification = None
            else:
                _restore_checkpoint(prefix_payload, trial_consumer, trial_optimizer)
                history = copy.deepcopy(list(prefix_payload["training_history"]))
                local_start = int(prefix_payload["task_local_updates"])
                update_verification = copy.deepcopy(prefix_stages[0].get("first_update_verification"))
                _require(update_verification is not None and update_verification.get("passed") is True, f"resumed {seed}/{arm} trajectory lacks its verified first-update evidence")
                candidate_record["actual_update_verified"] = candidate_record["actual_update_verified"] or requested_size == actual_size
            encoder_before = trial_parameters["encoder.phase"].detach().cpu().clone()
            frozen_before = capacity._frozen_runtime_state(trial_consumer)
            for local in range(local_start, 3):
                inputs["engineering_update_context"] = {"phase": "six_seed_arm_three_step_updates", "seed": seed, "arm": arm, "task_local_update": local + 1, "requested_microbatch_size": requested_size, "microbatch_size": actual_size}
                if local == 0:
                    update_verification = _verify_first_update_against_reference(
                        trial_consumer, trial_optimizer, trial_parameters, evaluators, inputs, cache["coefficients"], schedule[seed][local], seed, arm, local,
                    )
                    first_update_verification = {key: value for key, value in update_verification.items() if key != "training_record"}
                    first_record = update_verification.pop("training_record")
                    first_record["engineering_first_update_verification"] = copy.deepcopy(first_update_verification)
                    history.append(first_record)
                    candidate_record["actual_update_verified"] = candidate_record["actual_update_verified"] or requested_size == actual_size
                else:
                    history.append(_train_step(trial_consumer, trial_optimizer, trial_parameters, evaluators, inputs, cache["coefficients"], schedule[seed], seed, arm, local))
                    first_update_verification = None
                if arm == "A" and local == 0:
                    first_a_state[seed] = {
                        "phase": trial_parameters["generator.phase"].detach().cpu().clone(),
                        "optimizer": copy.deepcopy(capacity._cpu_tree(trial_optimizer.state_dict())),
                    }
                if source_args.engineering_check_only:
                    _save_engineering_update(
                        target,
                        checkpoint_root,
                        inputs,
                        trial_consumer,
                        trial_optimizer,
                        seed,
                        arm,
                        local + 1,
                        schedule[seed],
                        history,
                        group="trajectory",
                        key=checkpoint_key,
                        microbatch_size=actual_size,
                        first_update_verification=first_update_verification,
                        checkpoint_retention="preserve_first_step_control" if arm == "A" and local == 0 else "keep_recent_two_after_readback",
                    )
                    _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", f"trajectory_{seed}_{arm}_update_{local + 1}")
            _require(torch.equal(encoder_before, trial_parameters["encoder.phase"].detach().cpu()), f"three-step preflight {seed}/{arm} changed the frozen Encoder")
            capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(trial_consumer), f"three-step preflight {seed}/{arm} frozen runtime")
            _require(int(float(trial_optimizer.state[trial_parameters["generator.phase"]]["step"].item())) == SOURCE_STEP + 3, f"three-step preflight {seed}/{arm} did not make three real Adam updates")
            first_stage = _read_engineering_stage(target, _engineering_update_stage_name("trajectory", checkpoint_key, 1)) if source_args.engineering_check_only else None
            if arm == "A" and source_args.engineering_check_only:
                _require(first_stage is not None, f"A trajectory lacks its preserved first-step control: {seed}")
                first_checkpoint = _read_checkpoint_file(
                    (target / str(first_stage["checkpoint"])).resolve(),
                    (target / str(first_stage["checkpoint_root"])).resolve(),
                    inputs,
                    inputs["batch_orders"],
                )
                first_a_state[seed] = {
                    "phase": first_checkpoint["trainable_parameter_state"]["generator.phase"],
                    "optimizer": first_checkpoint["optimizer_state"],
                }
            if update_verification is None and first_stage is not None:
                update_verification = copy.deepcopy(first_stage.get("first_update_verification")) if first_stage is not None else None
            if source_args.engineering_check_only:
                _require(update_verification is not None and update_verification.get("passed") is True, f"trajectory first-update verification is missing: {seed}/{arm}")
            trials.append({"seed": seed, "arm": arm, "updates": len(history), "optimizer_step": SOURCE_STEP + 3, "requested_microbatch_size": requested_size, "microbatch_size": actual_size, "candidate_equivalence_passed": candidate_record["equivalence_passed"], "candidate_first_update": update_verification, "generator_changed": True, "encoder_unchanged": True, "history": history})
            if arm == "A":
                a_microbatch_by_seed[seed] = actual_size
            if seed == TRAIN_SEEDS[0] and arm == "A":
                continuous_seed_zero_a = {
                    "phase": trial_parameters["generator.phase"].detach().cpu().clone(),
                    "optimizer": copy.deepcopy(capacity._cpu_tree(trial_optimizer.state_dict())),
                    "rng": capacity._rng_state(),
                    "history": copy.deepcopy(history),
                }
            if source_args.engineering_check_only:
                saved_path = _checkpoint_path(checkpoint_root, seed, arm, 3)
                saved_state = _read_checkpoint_file(saved_path, checkpoint_root, inputs, inputs["batch_orders"])
                _require(
                    saved_state.get("task_local_updates") == 3
                    and capacity._tree_equal(saved_state["training_history"], history)
                    and saved_state["optimizer_step"] == SOURCE_STEP + 3,
                    f"completed diagnostic trajectory failed serialized checkpoint readback: {seed}/{arm}",
                )
                trajectory_stage = {
                    "status": "COMPLETE",
                    "seed": seed,
                    "arm": arm,
                    "updates": 3,
                    "optimizer_step": saved_state["optimizer_step"],
                    "checkpoint": saved_path.relative_to(target).as_posix(),
                    "checkpoint_root": checkpoint_root.relative_to(target).as_posix(),
                    "checkpoint_retention": "retain_endpoint",
                    "batch_cursor": saved_state["next_batch_cursor"],
                    "history_rows": len(saved_state["training_history"]),
                }
                final_stage_path = _engineering_stage_path(target, f"trajectory_seed_{seed}_{arm.lower()}")
                if final_stage_path.is_file():
                    _require(_json(final_stage_path) == {"task_id": TASK_ID, "phase": f"trajectory_seed_{seed}_{arm.lower()}", "evidence_identity": {
                        "input_population": "input_population.json",
                        "batch_orders": "batch_orders.json",
                        "run_config": "engineering_run_config.json",
                    }, **trajectory_stage}, f"completed trajectory stage changed on resume: {seed}/{arm}")
                else:
                    _write_engineering_stage(target, f"trajectory_seed_{seed}_{arm.lower()}", trajectory_stage)
                _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", "trajectory_complete")
                if seed == TRAIN_SEEDS[0] and arm == "A":
                    _save_continuous_repeat_evidence(target, 1, continuous_seed_zero_a, seed, arm, schedule[seed])
            del trial_consumer, trial_optimizer, trial_parameters
        zero_consumer, zero_optimizer, zero_parameters = _new_endpoint(sources, device)
        inputs["microbatch_size"] = a_microbatch_by_seed[seed]
        inputs["engineering_update_context"] = {"phase": "zero_lambda_negative_control", "seed": seed, "arm": "B", "task_local_update": 1, "microbatch_size": inputs["microbatch_size"]}
        zero_key = f"seed_{seed}_arm_b"
        zero_root = _engineering_checkpoint_root(target, "zero_lambda", zero_key)
        if source_args.engineering_check_only:
            zero_root.mkdir(parents=True, exist_ok=True)
            zero_payload, _zero_stages, _last_zero_stage = _engineering_update_prefix(
                target, inputs, "zero_lambda", zero_key, seed, "B", inputs["microbatch_size"],
            )
        else:
            zero_payload = None
        if zero_payload is None:
            capacity._seed_all(seed)
            zero_history: list[dict[str, Any]] = []
            zero_record = _train_step(zero_consumer, zero_optimizer, zero_parameters, evaluators, inputs, cache["coefficients"], schedule[seed], seed, "B", 0, qfg_lambda=0.0)
            zero_history.append(zero_record)
            if source_args.engineering_check_only:
                _save_engineering_update(
                    target,
                    zero_root,
                    inputs,
                    zero_consumer,
                    zero_optimizer,
                    seed,
                    "B",
                    1,
                    schedule[seed],
                    zero_history,
                    group="zero_lambda",
                    key=zero_key,
                    microbatch_size=inputs["microbatch_size"],
                    checkpoint_retention="preserve_audit",
                )
        else:
            _restore_checkpoint(zero_payload, zero_consumer, zero_optimizer)
            zero_record = copy.deepcopy(zero_payload["training_history"][-1])
            _require(zero_record.get("effective_qfg_lambda") == 0.0, f"saved zero-lambda update changed objective: {seed}")
        phase_comparison = _float_tensor_comparison(first_a_state[seed]["phase"], zero_parameters["generator.phase"].detach().cpu())
        _require(phase_comparison["passed"], f"zero-lambda B phase differs from A beyond float roundoff for seed {seed}: {phase_comparison}")
        optimizer_comparison = _adam_state_comparison(first_a_state[seed]["optimizer"], capacity._cpu_tree(zero_optimizer.state_dict()))
        _require(optimizer_comparison["passed"], f"zero-lambda B Adam differs from A beyond float roundoff for seed {seed}: {optimizer_comparison}")
        trials[-1]["zero_lambda_control"] = {
            "executed": True,
            "passed": True,
            "seed": seed,
            "optimizer_step": zero_record["optimizer_step"],
            "phase_comparison": phase_comparison,
            "optimizer_comparison": optimizer_comparison,
        }
        if source_args.engineering_check_only:
            zero_stage = {
                "status": "COMPLETE",
                "passed": True,
                "seed": seed,
                "microbatch_size": inputs["microbatch_size"],
                "phase_comparison": phase_comparison,
                "optimizer_comparison": optimizer_comparison,
                "optimizer_step": zero_record["optimizer_step"],
                "update_stage": _engineering_update_stage_name("zero_lambda", zero_key, 1),
            }
            summary_stage_path = _engineering_stage_path(target, f"zero_lambda_seed_{seed}")
            if summary_stage_path.is_file():
                existing_summary = _read_engineering_stage(target, f"zero_lambda_seed_{seed}")
                for field in ("status", "passed", "seed", "microbatch_size", "phase_comparison", "optimizer_comparison", "optimizer_step", "update_stage"):
                    _require(existing_summary.get(field) == zero_stage.get(field), f"saved zero-lambda summary changed on resume: {seed}/{field}")
            else:
                _write_engineering_stage(target, f"zero_lambda_seed_{seed}", zero_stage)
        del zero_consumer, zero_optimizer, zero_parameters
    inputs["microbatch_size"] = selected_microbatch
    update_eligible = [row for row in candidate_results if row["equivalence_passed"] and row["actual_update_verified"]]
    _require(any(row["microbatch_size"] == 1 for row in update_eligible), "microbatch-1 candidate did not receive a verified actual optimizer update")
    fastest = min(update_eligible, key=lambda row: row["median_seconds"])
    recommended_microbatch = fastest["microbatch_size"] if fastest["median_seconds"] <= 0.9 * size1_time else 1
    checks["merged_production_gradient_and_speed"]["recommended_microbatch_size"] = recommended_microbatch
    checks["merged_production_gradient_and_speed"]["candidate_actual_update_eligibility"] = [
        {"microbatch_size": row["microbatch_size"], "equivalence_passed": row["equivalence_passed"], "actual_update_verified": row["actual_update_verified"]}
        for row in candidate_results
    ]
    checks["merged_production_gradient_and_speed"]["recommendation_rule"] = "select the fastest candidate that passed both-arm gradient checks and an actual optimizer update; recommend it only when its equal-weighted A/B median is at least 10 percent faster than microbatch 1"
    for name, model in evaluators.items():
        capacity._assert_state_equal(evaluator_states[name], capacity._state_snapshot(model), f"preflight frozen evaluator {name}")
    checks["six_seed_arm_three_step_updates"] = {"executed": True, "passed": True, "trajectories": trials, "updates": 18, "all_zero_lambda_controls_matched_first_A_within_float_roundoff": True}

    _require(continuous_seed_zero_a is not None, "continuous A4700 reference state was not captured")
    inputs["microbatch_size"] = a_microbatch_by_seed[TRAIN_SEEDS[0]]
    recovery = _recovery_preflight(inputs, cache["coefficients"], evaluators, target, source_args, continuous_seed_zero_a)
    checks["explicit_checkpoint_recovery"] = recovery
    for name, model in evaluators.items():
        capacity._assert_state_equal(evaluator_states[name], capacity._state_snapshot(model), f"post-recovery frozen evaluator {name}")
    budget = _read_engineering_budget(Path(inputs["engineering_budget_file"]))
    reservations = budget["reservations"]
    historical_updates = sum(row.get("origin") == "historical_before_budget_extension" for row in reservations)
    new_updates_reserved = len(reservations) - historical_updates
    new_updates_completed = sum(row.get("origin") != "historical_before_budget_extension" and row.get("status") == "optimizer_step_completed" for row in reservations)
    cumulative_updates = len(reservations)
    checks["total_engineering_adam_updates"] = {
        "executed": True,
        "passed": historical_updates == ENGINEERING_HISTORICAL_UPDATES
        and new_updates_reserved == ENGINEERING_NEW_UPDATE_LIMIT
        and new_updates_completed == ENGINEERING_NEW_UPDATE_LIMIT
        and cumulative_updates <= ENGINEERING_CUMULATIVE_UPDATE_LIMIT,
        "count": new_updates_completed,
        "new_updates_reserved": new_updates_reserved,
        "historical_updates": historical_updates,
        "cumulative_reserved": cumulative_updates,
        "new_update_limit": ENGINEERING_NEW_UPDATE_LIMIT,
        "limit": ENGINEERING_CUMULATIVE_UPDATE_LIMIT,
    }
    _require(all(value.get("executed") is True and value.get("passed") is True for value in checks.values()), "READY requires every production preflight check to pass")
    del consumer, optimizer, parameters, evaluators
    torch.cuda.empty_cache()
    return {"task_id": TASK_ID, "status": "READY", "checks": checks, "device": "cuda:0", "strict_fp32": STRICT_FP32_SETTINGS}


def _save_continuous_repeat_evidence(
    target: Path,
    repeat_index: int,
    state: Mapping[str, Any],
    seed: int,
    arm: str,
    batch_order: Sequence[Sequence[int]],
) -> Path:
    path = target / "preflight" / "continuous_repeats" / f"repeat_{repeat_index:02d}.pt"
    payload = {
        "task_id": TASK_ID,
        "repeat_index": repeat_index,
        "seed": seed,
        "arm": arm,
        "task_local_updates": 3,
        "batch_order": [list(batch) for batch in batch_order],
        "generator_phase": state["phase"],
        "optimizer_state": state["optimizer"],
        "rng_state": state["rng"],
        "training_history": state["history"],
    }
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        _require(capacity._tree_equal(saved, payload), f"continuous repeat evidence differs from its saved state: {repeat_index}")
    else:
        _write_torch_exact(path, payload)
        saved = torch.load(path, map_location="cpu", weights_only=True)
    _require(capacity._tree_equal(saved, payload), f"continuous repeat state failed serialized readback: {repeat_index}")
    return path


def _process_is_alive(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _recovery_preflight(
    inputs: Mapping[str, Any],
    ols_coefficients: torch.Tensor,
    evaluators: Mapping[str, nn.Module],
    target: Path,
    source_args: argparse.Namespace,
    continuous_seed_zero_a: Mapping[str, Any],
) -> dict[str, Any]:
    seed, arm = TRAIN_SEEDS[0], "A"
    batch_order = inputs["batch_orders"][seed]
    continuous_states = [continuous_seed_zero_a]
    continuous_trials = [{"seed": seed, "arm": arm, "updates": 3, "reused_from": "six_seed_arm_three_step_updates"}]
    continuous_microbatch = int(inputs["microbatch_size"])
    _require(continuous_microbatch == 1, "repeatability reference must reuse the verified microbatch-1 A trajectory")
    for repeat_index in range(1, 5):
        repeat_number = repeat_index + 1
        key = f"seed_{seed}_arm_{arm.lower()}"
        group = f"continuous_repeat_{repeat_number:02d}"
        checkpoint_root = _engineering_checkpoint_root(target, group, key)
        if source_args.engineering_check_only:
            checkpoint_root.mkdir(parents=True, exist_ok=True)
        consumer, optimizer, parameters = _new_endpoint(inputs["sources"], torch.device("cuda:0"))
        if source_args.engineering_check_only:
            prefix_payload, _prefix_stages, _last_prefix_stage = _engineering_update_prefix(
                target, inputs, group, key, seed, arm, continuous_microbatch,
            )
        else:
            prefix_payload = None
        if prefix_payload is None:
            capacity._seed_all(seed)
            history: list[dict[str, Any]] = []
            local_start = 0
        else:
            _restore_checkpoint(prefix_payload, consumer, optimizer)
            history = copy.deepcopy(list(prefix_payload["training_history"]))
            local_start = int(prefix_payload["task_local_updates"])
        encoder_before = parameters["encoder.phase"].detach().cpu().clone()
        frozen_before = capacity._frozen_runtime_state(consumer)
        for local in range(local_start, 3):
            inputs["microbatch_size"] = continuous_microbatch
            inputs["engineering_update_context"] = {"phase": "continuous_repeat", "seed": seed, "arm": arm, "repeat": repeat_number, "task_local_update": local + 1, "microbatch_size": continuous_microbatch}
            history.append(_train_step(consumer, optimizer, parameters, evaluators, inputs, ols_coefficients, batch_order, seed, arm, local))
            if source_args.engineering_check_only:
                _save_engineering_update(
                    target,
                    checkpoint_root,
                    inputs,
                    consumer,
                    optimizer,
                    seed,
                    arm,
                    local + 1,
                    batch_order,
                    history,
                    group=group,
                    key=key,
                    microbatch_size=continuous_microbatch,
                )
                _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", f"{group}_update_{local + 1}")
        continuous_states.append({
            "phase": parameters["generator.phase"].detach().cpu().clone(),
            "optimizer": copy.deepcopy(capacity._cpu_tree(optimizer.state_dict())),
            "rng": capacity._rng_state(),
            "history": history,
        })
        if source_args.engineering_check_only:
            _save_continuous_repeat_evidence(target, repeat_number, continuous_states[-1], seed, arm, batch_order)
        _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), "continuous recovery repeat changed the frozen Encoder")
        capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), "continuous recovery repeat frozen runtime")
        continuous_trials.append({"seed": seed, "arm": arm, "updates": len(history), "repeat": repeat_number})
        del consumer, optimizer, parameters
    def schedule_trace(history: Sequence[Mapping[str, Any]]) -> list[tuple[Any, ...]]:
        return [(row.get("task_local_update"), row.get("seed"), row.get("arm"), row.get("fit_positions"), row.get("labels")) for row in history]

    for state in continuous_states[1:]:
        _require(capacity._tree_equal(continuous_states[0]["rng"], state["rng"]), "continuous recovery repeats consumed different terminal RNG state")
        _require(schedule_trace(continuous_states[0]["history"]) == schedule_trace(state["history"]), "continuous recovery repeats produced different batch/history prefixes")
    for repeat_index, state in enumerate(continuous_states, start=1):
        evidence_path = target / "preflight" / "continuous_repeats" / f"repeat_{repeat_index:02d}.pt"
        if repeat_index == 1 and not evidence_path.is_file():
            evidence_path = _save_continuous_repeat_evidence(target, repeat_index, state, seed, arm, batch_order)
        saved = torch.load(evidence_path, map_location="cpu", weights_only=True)
        _require(capacity._tree_equal(saved["generator_phase"], state["phase"]) and capacity._tree_equal(saved["optimizer_state"], state["optimizer"]) and capacity._tree_equal(saved["rng_state"], state["rng"]) and capacity._tree_equal(saved["training_history"], state["history"]), f"continuous repeat evidence differs after readback: {repeat_index}")
        if source_args.engineering_check_only:
            _write_engineering_stage(target, f"continuous_repeat_{repeat_index:02d}", {
                "status": "COMPLETE",
                "seed": seed,
                "arm": arm,
                "updates": 3,
                "serialized_state": evidence_path.relative_to(target).as_posix(),
                "state_readback_passed": True,
                "matched_fixed_batch_and_rng_prefix": True,
            })

    phase_values = [state["phase"] for state in continuous_states]
    optimizer_values = [state["optimizer"] for state in continuous_states]
    envelope_path = target / "preflight" / "repeatability_envelope.json"
    fixed_envelope = {
        "schema": "cloud_foreground_repeatability_envelope_v1",
        "continuous_repeat_count": len(continuous_states),
        "phase": _continuous_tensor_envelope(phase_values),
        "adam": _continuous_adam_envelope(optimizer_values),
        "source": "five independent continuous A4700/A/three-step trajectories; first trajectory reused from the six-trajectory preflight",
        "rule": "freeze these maximum pairwise absolute and RMS differences before any file recovery trial; do not widen after a failed recovery",
    }
    if envelope_path.is_file():
        persisted_envelope = _json(envelope_path)
    else:
        _write_json(envelope_path, fixed_envelope)
        persisted_envelope = _json(envelope_path)
    _require(persisted_envelope == fixed_envelope, "saved continuous-repeat envelope did not read back exactly")

    recovery_trials = []
    recovery_base = target / "preflight" / "file_recovery"
    signal_kinds: tuple[str | None, ...] = (None, "SIGINT", "SIGTERM")
    for trial_index in range(1, 4):
        recovery_seed = seed
        recovery_arm = "A"
        recovery_root = recovery_base / f"attempt_{trial_index:02d}"
        recovery_root.mkdir(parents=True, exist_ok=True)
        expected_signal = signal_kinds[trial_index - 1]
        comparison_path = recovery_root / "recovery_comparison.json"
        if comparison_path.is_file():
            comparison_record = _json(comparison_path)
            _require(comparison_record.get("status") == "PASS" and comparison_record.get("signal_kind") == expected_signal, f"saved file recovery comparison is incomplete or mismatched: {trial_index}")
            stage = _read_engineering_stage(target, f"file_recovery_{trial_index:02d}")
            _require(stage is not None and stage.get("passed") is True and stage.get("signal_kind") == expected_signal, f"saved file recovery diagnostic stage is missing: {trial_index}")
            saved_root = target / "preflight" / "file_recovery" / f"attempt_{trial_index:02d}"
            saved_boundary = _read_checkpoint_file(target / comparison_record["checkpoint"], saved_root, inputs, inputs["batch_orders"])
            saved_terminal = _read_checkpoint_file(target / comparison_record["terminal_checkpoint"], saved_root, inputs, inputs["batch_orders"])
            _require(saved_boundary.get("task_local_updates") == 1 and saved_terminal.get("task_local_updates") == 3, f"saved file recovery checkpoints are incomplete: {trial_index}")
            phase_comparison = _tensor_repeatability_envelope(phase_values, saved_terminal["trainable_parameter_state"]["generator.phase"], fixed_envelope=persisted_envelope["phase"])
            optimizer_comparison = _adam_repeatability_envelope(optimizer_values, saved_terminal["optimizer_state"], fixed_envelope=persisted_envelope["adam"])
            _require(phase_comparison.get("passed") is True and optimizer_comparison.get("passed") is True, f"saved file recovery endpoint no longer meets its fixed envelope: {trial_index}")
            _require(capacity._tree_equal(phase_comparison, comparison_record.get("phase_comparison")) and capacity._tree_equal(optimizer_comparison, comparison_record.get("adam_comparison")), f"saved file recovery comparison changed on recheck: {trial_index}")
            _require(capacity._tree_equal(saved_boundary["training_history"], saved_terminal["training_history"][:1]), f"saved file recovery history prefix differs: {trial_index}")
            save_result = _json(saved_root / "save_worker_result.json")
            resume_result = _json(saved_root / "recovery_worker_result.json")
            exit_record = _json(saved_root / "save_process_exit.json")
            resume_exit = _json(saved_root / "resume_process_exit.json")
            _require(save_result.get("process_id") == comparison_record.get("save_process_id") and exit_record.get("process_exited_before_resume") is True, f"saved file recovery did not confirm writer exit: {trial_index}")
            _require(resume_result.get("process_id") == comparison_record.get("resume_process_id") and resume_result.get("process_id") != save_result.get("process_id"), f"saved file recovery process identities differ: {trial_index}")
            _require(resume_exit.get("process_exited") is True and resume_exit.get("process_id") == resume_result.get("process_id") and resume_exit.get("return_code") in {0, None}, f"saved resume process exit was not confirmed: {trial_index}")
            recovery_trials.append({
                "seed": recovery_seed,
                "arm": recovery_arm,
                "updates": 3,
                "checkpoint": comparison_record["checkpoint"],
                "terminal_checkpoint": comparison_record["terminal_checkpoint"],
                "save_process_exited": True,
                "resume_started_in_distinct_process": True,
                "signal_kind": expected_signal,
                "boundary_model_adam_rng_batch_history_exact": True,
                "phase_comparison": comparison_record["phase_comparison"],
                "optimizer_comparison": comparison_record["adam_comparison"],
            })
            continue
        checkpoint_path = _checkpoint_path(recovery_root, recovery_seed, recovery_arm, 1)
        terminal_path = _checkpoint_path(recovery_root, recovery_seed, recovery_arm, 3)
        common_command = [
            sys.executable, "-m", "scripts.research.lightgen.execute_lightgen_2d_generator_cloud_foreground_dev_multiseed_v1",
            "--artifact-dir", str(source_args.artifact_dir),
            "--checkpoint", str(source_args.checkpoint),
            "--gradient-audit-dir", str(source_args.gradient_audit_dir),
            "--reference-dir", str(source_args.reference_dir),
            "--output-dir", recovery_root.relative_to(_root()).as_posix(),
            "--ols-cache", (target / "ols_fit_coefficients.pt").relative_to(_root()).as_posix(),
            "--microbatch-size", "1",
            "--engineering-recovery-worker",
        ]
        save_result_path = recovery_root / "save_worker_result.json"
        save_exit_path = recovery_root / "save_process_exit.json"
        if not save_result_path.is_file():
            budget = _read_engineering_budget(Path(inputs["engineering_budget_file"]))
            prior_save_reservations = [
                row for row in budget["reservations"]
                if row.get("context", {}).get("phase") == "independent_save_process"
                and row.get("context", {}).get("trial") == recovery_root.name
                and row.get("context", {}).get("task_local_update") == 1
            ]
            _require(not checkpoint_path.exists() and not prior_save_reservations, f"recovery save process left ambiguous partial evidence in trial {trial_index}; its update will not be repeated")
            save_command = [*common_command, "--engineering-recovery-stage", "save1"]
            if expected_signal is not None:
                save_command.extend(("--engineering-signal-kind", expected_signal))
            writer = subprocess.run(save_command, cwd=_root(), text=True, capture_output=True, check=False)
            (recovery_root / "save_worker_stdout.txt").write_text(writer.stdout, encoding="utf-8", newline="\n")
            (recovery_root / "save_worker_stderr.txt").write_text(writer.stderr, encoding="utf-8", newline="\n")
            _require(writer.returncode == 0, f"independent save process failed in trial {trial_index}: {writer.stderr[-2000:]}")
            _require(save_result_path.is_file(), "independent save process exited without a result record")
            launched_result = _json(save_result_path)
            _write_json(save_exit_path, {"process_id": launched_result.get("process_id"), "return_code": writer.returncode, "process_exited_before_resume": True, "observed_by_controller": True})
        save_result = _json(save_result_path)
        _require(save_result.get("status") == "PASS" and save_result.get("stage") == "save1" and save_result.get("signal_kind") == expected_signal, "independent save process did not report the expected verified boundary")
        _require(save_result.get("signal_save_verified") is (expected_signal is not None), "signal-driven checkpoint save did not confirm its requested signal")
        if not save_exit_path.is_file():
            _require(not _process_is_alive(int(save_result["process_id"])), "save process has not exited; recovery will not start until it does")
            _write_json(save_exit_path, {"process_id": save_result["process_id"], "return_code": None, "process_exited_before_resume": True, "observed_by_controller": False})
        exit_record = _json(save_exit_path)
        _require(exit_record.get("process_id") == save_result.get("process_id") and exit_record.get("process_exited_before_resume") is True, "save process exit was not confirmed before resume")
        _require(checkpoint_path.is_file(), "independent save process exited without an explicit recovery checkpoint")
        saved = _read_checkpoint_file(checkpoint_path, recovery_root, inputs, inputs["batch_orders"])
        _require(saved.get("task_local_updates") == 1 and len(saved.get("training_history", [])) == 1, "independent save process wrote an invalid one-update boundary")
        worker_result_path = recovery_root / "recovery_worker_result.json"
        resume_exit_path = recovery_root / "resume_process_exit.json"
        if worker_result_path.is_file():
            worker_result = _json(worker_result_path)
            _require(worker_result.get("status") == "PASS" and terminal_path.is_file(), f"saved recovery process result is incomplete in trial {trial_index}")
            if not resume_exit_path.is_file():
                _require(not _process_is_alive(int(worker_result["process_id"])), "resume process is still alive; recovery evidence will be consumed after it exits")
                _write_json(resume_exit_path, {"process_id": worker_result["process_id"], "return_code": None, "process_exited": True, "observed_by_controller": False})
        else:
            worker = subprocess.run(
                [*common_command, "--resume-checkpoint", checkpoint_path.relative_to(_root()).as_posix(), "--repeatability-envelope", envelope_path.relative_to(_root()).as_posix(), "--engineering-recovery-stage", "resume2"],
                cwd=_root(), text=True, capture_output=True, check=False,
            )
            (recovery_root / "resume_worker_stdout.txt").write_text(worker.stdout, encoding="utf-8", newline="\n")
            (recovery_root / "resume_worker_stderr.txt").write_text(worker.stderr, encoding="utf-8", newline="\n")
            _require(worker.returncode == 0, f"fresh-process checkpoint recovery failed in trial {trial_index}: {worker.stderr[-2000:]}")
            worker_result = _json(worker_result_path)
            _write_json(resume_exit_path, {"process_id": worker_result.get("process_id"), "return_code": worker.returncode, "process_exited": True, "observed_by_controller": True})
        resume_exit = _json(resume_exit_path)
        _require(resume_exit.get("process_id") == worker_result.get("process_id") and resume_exit.get("process_exited") is True, "resume process exit was not confirmed before final comparison")
        _require(
            worker_result.get("status") == "PASS"
            and worker_result.get("stage") == "resume2"
            and worker_result.get("resume_checkpoint") == checkpoint_path.relative_to(recovery_root).as_posix()
            and worker_result.get("process_id") != save_result.get("process_id"),
            "independent recovery process did not consume the explicit saved file after the save process exited",
        )
        terminal_path = _checkpoint_path(recovery_root, recovery_seed, recovery_arm, 3)
        recovered = _read_checkpoint_file(terminal_path, recovery_root, inputs, inputs["batch_orders"])
        _require(capacity._tree_equal(saved["training_history"], recovered["training_history"][:1]), "fresh-process recovery did not preserve the saved history prefix exactly")
        phase_comparison = _tensor_repeatability_envelope(phase_values, recovered["trainable_parameter_state"]["generator.phase"], fixed_envelope=persisted_envelope["phase"])
        optimizer_comparison = _adam_repeatability_envelope(optimizer_values, recovered["optimizer_state"], fixed_envelope=persisted_envelope["adam"])
        _require(phase_comparison["passed"], f"file-resumed Generator phase exceeds the persisted five-run same-device envelope: {phase_comparison}")
        _require(optimizer_comparison["passed"], f"file-resumed Adam exceeds the persisted five-run same-device envelope: {optimizer_comparison}")
        _require(capacity._tree_equal(recovered["rng_state"], continuous_states[0]["rng"]), "file-resumed terminal RNG differs from continuous recovery reference")
        _require(recovered["batch_order"] == [list(batch) for batch in inputs["batch_orders"][recovery_seed]], "file-resumed batch order differs")
        _require(schedule_trace(recovered["training_history"]) == schedule_trace(continuous_states[0]["history"]), "file-resumed batch/history trace differs from the continuous reference")
        comparison_record = {
            "status": "PASS",
            "signal_kind": expected_signal,
            "save_process_id": save_result["process_id"],
            "save_process_exited_before_resume_process_started": True,
            "resume_process_id": worker_result["process_id"],
            "boundary_state_exact_after_disk_reload": True,
            "history_prefix_exact": True,
            "rng_batch_order_and_cursor_exact": True,
            "phase_comparison": phase_comparison,
            "adam_comparison": optimizer_comparison,
            "envelope_path": envelope_path.relative_to(target).as_posix(),
            "checkpoint": checkpoint_path.relative_to(target).as_posix(),
            "terminal_checkpoint": terminal_path.relative_to(target).as_posix(),
        }
        _write_json(comparison_path, comparison_record)
        if source_args.engineering_check_only:
            recovery_stage = {
                "status": "COMPLETE",
                "passed": True,
                "seed": recovery_seed,
                "arm": recovery_arm,
                "updates": 3,
                "save_process_exited": True,
                "resume_started_in_distinct_process": True,
                "comparison": recovery_root.relative_to(target).as_posix() + "/recovery_comparison.json",
                "boundary_state_exact": True,
                "same_device_envelope_passed": True,
                "signal_kind": expected_signal,
            }
            stage_name = f"file_recovery_{trial_index:02d}"
            existing_stage = _read_engineering_stage(target, stage_name)
            if existing_stage is None:
                _write_engineering_stage(target, stage_name, recovery_stage)
            else:
                _require(all(existing_stage.get(key) == value for key, value in recovery_stage.items()), f"saved file recovery stage differs: {trial_index}")
            _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", "file_recovery")
        recovery_trials.append({
            "seed": recovery_seed,
            "arm": recovery_arm,
            "updates": 3,
            "checkpoint": checkpoint_path.relative_to(target).as_posix(),
            "terminal_checkpoint": terminal_path.relative_to(target).as_posix(),
            "save_process_exited": True,
            "resume_started_in_distinct_process": True,
            "signal_kind": expected_signal,
            "boundary_model_adam_rng_batch_history_exact": True,
            "phase_comparison": phase_comparison,
            "optimizer_comparison": optimizer_comparison,
        })

    return {
        "executed": True,
        "passed": True,
        "continuous_repeat_count": len(continuous_states),
        "continuous_repeat_updates": 15,
        "continuous_repeat_additional_updates": 12,
        "recovery_trials": recovery_trials,
        "recovery_updates": 9,
        "recovery_updates_executed_in_new_processes": True,
        "checkpoint_boundary_model_adam_rng_batch_history_exact": True,
        "repeatability_envelope_fixed_before_recovery": True,
        "repeatability_envelope": persisted_envelope,
        "continuous_repeats": continuous_trials,
        "numerical_claim": "recovery differences did not exceed the observed same-device continuous repeatability envelope; this is not strict numerical identity",
    }


def _engineering_recovery_worker(args: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    artifact_dir = _portable_resolve(root, args.artifact_dir)
    checkpoint = _portable_resolve(root, args.checkpoint)
    gradient_dir = _portable_resolve(root, args.gradient_audit_dir)
    reference_dir = _portable_resolve(root, args.reference_dir)
    target = _portable_resolve(root, args.output_dir)
    args.engineering_attempt_dir = target.parents[2] if len(target.parents) >= 3 else None
    resume = _portable_resolve(root, args.resume_checkpoint) if args.resume_checkpoint is not None else None
    ols_cache_path = _portable_resolve(root, args.ols_cache) if args.ols_cache is not None else None
    envelope_path = _portable_resolve(root, args.repeatability_envelope) if args.repeatability_envelope is not None else None
    budget_file = _engineering_budget_file()
    _require(torch.cuda.is_available(), "engineering recovery worker requires CUDA")
    _require(target.is_dir() and args.engineering_recovery_stage in {"save1", "resume2"}, "engineering recovery worker requires an explicit supported stage and diagnostic directory")
    _require(getattr(args, "engineering_signal_kind", None) is None or args.engineering_recovery_stage == "save1", "signal injection is only valid for the independent save worker")
    _require((resume is None) if args.engineering_recovery_stage == "save1" else (resume is not None and resume.is_file() and target.resolve() in resume.resolve().parents), "engineering recovery stage requires the matching explicit resume checkpoint state")
    _require(target.resolve().is_relative_to((_root() / DEFAULT_ENGINEERING_OUTPUT).resolve()) and budget_file.is_file(), "engineering recovery worker must consume the task-wide budget inside the registered diagnostic")
    _require(ols_cache_path is not None and ols_cache_path.is_file(), "engineering recovery worker requires the explicit preflight OLS cache")
    inputs = _load_all_inputs(root, artifact_dir, checkpoint, gradient_dir, reference_dir)
    inputs["microbatch_size"] = args.microbatch_size
    attempt_root = target.parents[2]
    expected_orders = {str(seed): inputs["batch_orders"][seed] for seed in TRAIN_SEEDS}
    expected_config = {
        "task_id": TASK_ID,
        "source_checkpoint": inputs["population"]["source_checkpoint"],
        "source_checkpoint_step": SOURCE_STEP,
        "source_adam_step": SOURCE_STEP,
        "requested_microbatch_size": int(args.microbatch_size),
        "trajectory_microbatch_sizes": [1, 2, 5, 10, 1, 1],
        "logical_batch_size": BATCH_SIZE,
        "propagation_grid": 2_048,
        "optical_dtype": "float32",
        "qfg_dtype": "float64",
        "fixed_lambda": FIXED_LAMBDA,
        "generator_only_adam": True,
    }
    _require(_json(attempt_root / "input_population.json") == inputs["population"], "recovery worker source model or population differs from its parent attempt")
    _require(_json(attempt_root / "batch_orders.json") == expected_orders, "recovery worker batch orders differ from its parent attempt")
    _require(_json(attempt_root / "engineering_run_config.json") == expected_config, "recovery worker configuration differs from its parent attempt")
    ols_cache = torch.load(ols_cache_path, map_location="cpu", weights_only=True)
    _require(ols_cache.get("task_id") == TASK_ID and torch.equal(ols_cache.get("source_row_indices"), inputs["fit_rows"]), "engineering recovery OLS cache task or population differs")
    _require(tuple(ols_cache.get("coefficients", torch.empty(0)).shape) == (FIT_ROWS, 100) and ols_cache["coefficients"].is_complex(), "engineering recovery OLS coefficients are incomplete")
    inputs["ols_coefficients"] = ols_cache["coefficients"]
    inputs["engineering_budget_file"] = budget_file
    inputs["engineering_progress_file"] = _root() / DEFAULT_ENGINEERING_OUTPUT / "progress.json"
    inputs["engineering_attempt_dir"] = attempt_root
    args.engineering_attempt_dir = attempt_root
    if args.engineering_recovery_stage == "save1":
        seed, arm = TRAIN_SEEDS[0], "A"
        evaluators = generation.diagnostic.load_evaluators(inputs["sources"]["source"]["generation_paths"], torch.device("cuda:0"))
        evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
        consumer, optimizer, parameters = _new_endpoint(inputs["sources"], torch.device("cuda:0"))
        capacity._seed_all(seed)
        encoder_before = parameters["encoder.phase"].detach().cpu().clone()
        inputs["engineering_update_context"] = {"phase": "independent_save_process", "trial": target.name, "seed": seed, "arm": arm, "task_local_update": 1, "microbatch_size": args.microbatch_size}
        stop_requested = {"value": False}
        previous_handlers: dict[int, Any] = {}
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda _signal, _frame: stop_requested.__setitem__("value", True))
            history = [_train_step(consumer, optimizer, parameters, evaluators, inputs, inputs["ols_coefficients"], inputs["batch_orders"][seed], seed, arm, 0)]
            signal_kind = getattr(args, "engineering_signal_kind", None)
            if signal_kind is not None:
                signum = signal.SIGINT if signal_kind == "SIGINT" else signal.SIGTERM
                os.kill(os.getpid(), signum)
            _require(signal_kind is None or stop_requested["value"], "requested recovery signal was not observed by the save process")
        finally:
            for signum, previous_handler in previous_handlers.items():
                signal.signal(signum, previous_handler)
        _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), "independent save process changed the frozen Encoder")
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"independent save frozen evaluator {name}")
        reason = "signal_requested" if stop_requested["value"] else "independent_save_process_one_update"
        saved_path = _save_checkpoint(target, consumer, optimizer, inputs, seed, arm, 1, inputs["batch_orders"][seed], history, reason=reason)
        result = {"status": "PASS", "stage": "save1", "process_id": os.getpid(), "checkpoint": saved_path.relative_to(target).as_posix(), "updates": 1, "process_exit_required": True, "signal_kind": signal_kind, "signal_save_verified": stop_requested["value"]}
        _write_json(target / "save_worker_result.json", result, replace=(target / "save_worker_result.json").exists())
        del consumer, optimizer, parameters, evaluators
        torch.cuda.empty_cache()
        return result

    _require(resume is not None, "resume worker lost its explicit checkpoint path")
    _require(envelope_path is not None and envelope_path.is_file(), "resume process requires the persisted continuous-repeat envelope")
    envelope = _json(envelope_path)
    _require(envelope.get("schema") == "cloud_foreground_repeatability_envelope_v1" and envelope.get("continuous_repeat_count") == 5, "resume process found an invalid fixed repeatability envelope")
    payload = _read_checkpoint_file(resume, target, inputs, inputs["batch_orders"])
    _require(payload.get("task_local_updates") == 1, "engineering recovery worker accepts only the saved one-update boundary")
    evaluators = generation.diagnostic.load_evaluators(inputs["sources"]["source"]["generation_paths"], torch.device("cuda:0"))
    evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    consumer, optimizer, parameters = _new_endpoint(inputs["sources"], torch.device("cuda:0"))
    _restore_checkpoint(payload, consumer, optimizer)
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    frozen_before = capacity._frozen_runtime_state(consumer)
    for name, parameter in parameters.items():
        _require(torch.equal(payload["trainable_parameter_state"][name], parameter.detach().cpu()), f"fresh-process loaded checkpoint phase differs for {name}")
    _require(capacity._tree_equal(payload["optimizer_state"], capacity._cpu_tree(optimizer.state_dict())), "fresh-process loaded Adam differs at checkpoint boundary")
    _require(capacity._tree_equal(payload["rng_state"], capacity._rng_state()), "fresh-process loaded RNG differs at checkpoint boundary")
    _require(len(payload["training_history"]) == 1 and payload["training_history"][0]["task_local_update"] == 1, "checkpoint history prefix is not a one-update boundary")
    _require(payload["batch_order"] == [list(batch) for batch in inputs["batch_orders"][payload["seed"]]] and payload["next_batch_cursor"] == 1, "checkpoint batch cursor or order differs")
    history = copy.deepcopy(list(payload["training_history"]))
    current_payload = payload
    current_local = 1
    budget = _read_engineering_budget(budget_file)
    reservations = budget["reservations"]
    for local_update in (2, 3):
        stage_path = target / f"resume_update_{local_update}.json"
        checkpoint_path = _checkpoint_path(target, int(payload["seed"]), str(payload["arm"]), local_update)
        matching_reservations = [
            row for row in reservations
            if row.get("context", {}).get("phase") == "independent_resume_process"
            and row.get("context", {}).get("trial") == target.name
            and row.get("context", {}).get("task_local_update") == local_update
        ]
        if not stage_path.is_file():
            _require(len(matching_reservations) <= 1, f"recovery update {local_update} has duplicate budget reservations")
            if checkpoint_path.is_file():
                _require(len(matching_reservations) == 1 and matching_reservations[0].get("status") == "optimizer_step_completed", f"recovery update {local_update} checkpoint lacks a completed reservation")
                recovered_stage_payload = _read_checkpoint_file(checkpoint_path, target, inputs, inputs["batch_orders"])
                _require(recovered_stage_payload.get("task_local_updates") == local_update and len(recovered_stage_payload.get("training_history", [])) == local_update, f"recovery update {local_update} checkpoint cursor is invalid")
                stage_payload = {
                    "status": "COMPLETE",
                    "task_local_update": local_update,
                    "checkpoint": checkpoint_path.relative_to(target).as_posix(),
                    "training_history": copy.deepcopy(recovered_stage_payload["training_history"]),
                    "process_id": os.getpid(),
                    "recovered_from_checkpoint_after_process_restart": True,
                }
                _write_json(stage_path, stage_payload)
            else:
                _require(not matching_reservations, f"recovery update {local_update} has a reserved or completed update but no checkpoint")
    _require(not (target / "resume_update_3.json").is_file() or (target / "resume_update_2.json").is_file(), "recovery update 3 exists without its update 2 dependency")
    for next_local in (2, 3):
        stage_path = target / f"resume_update_{next_local}.json"
        if not stage_path.is_file():
            break
        stage = _json(stage_path)
        _require(stage.get("status") == "COMPLETE" and stage.get("task_local_update") == next_local, f"fresh-process recovery update stage is invalid: {next_local}")
        checkpoint_path = (target / stage.get("checkpoint", "")).resolve()
        _require(target.resolve() in checkpoint_path.parents and checkpoint_path.is_file(), f"fresh-process recovery checkpoint is missing: {next_local}")
        loaded = _read_checkpoint_file(checkpoint_path, target, inputs, inputs["batch_orders"])
        _require(loaded.get("task_local_updates") == next_local and capacity._tree_equal(loaded["training_history"], stage.get("training_history")), f"fresh-process recovery checkpoint does not match its stage: {next_local}")
        _restore_checkpoint(loaded, consumer, optimizer)
        history = copy.deepcopy(list(loaded["training_history"]))
        current_payload = loaded
        current_local = next_local
    for local in range(current_local, 3):
        inputs["engineering_update_context"] = {"phase": "independent_resume_process", "trial": target.name, "seed": payload["seed"], "arm": payload["arm"], "task_local_update": local + 1, "microbatch_size": args.microbatch_size}
        history.append(_train_step(consumer, optimizer, parameters, evaluators, inputs, inputs["ols_coefficients"], inputs["batch_orders"][payload["seed"]], int(payload["seed"]), str(payload["arm"]), local))
        checkpoint_path = _save_checkpoint(target, consumer, optimizer, inputs, int(payload["seed"]), str(payload["arm"]), local + 1, inputs["batch_orders"][payload["seed"]], history, reason=f"engineering_recovery_update_{local + 1}")
        saved = _read_checkpoint_file(checkpoint_path, target, inputs, inputs["batch_orders"])
        _require(saved.get("task_local_updates") == local + 1 and capacity._tree_equal(saved["training_history"], history), "recovery subprocess update checkpoint failed its production readback")
        _write_json(target / f"resume_update_{local + 1}.json", {
            "status": "COMPLETE",
            "task_local_update": local + 1,
            "checkpoint": checkpoint_path.relative_to(target).as_posix(),
            "training_history": copy.deepcopy(history),
            "process_id": os.getpid(),
        })
        current_payload = saved
        current_local = local + 1
    terminal_path = _checkpoint_path(target, int(payload["seed"]), str(payload["arm"]), 3)
    _require(current_local == 3 and current_payload.get("task_local_updates") == 3, "fresh-process recovery did not reach the three-update endpoint")
    _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), "fresh-process recovery changed the frozen Encoder")
    capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), "fresh-process recovery frozen runtime")
    for name, model in evaluators.items():
        capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"fresh-process recovery frozen evaluator {name}")
    _write_json(target / "recovery_worker_result.json", {"status": "PASS", "stage": "resume2", "process_id": os.getpid(), "resume_checkpoint": resume.relative_to(target).as_posix(), "terminal_checkpoint": terminal_path.relative_to(target).as_posix(), "updates": 2, "fresh_process": True}, replace=(target / "recovery_worker_result.json").exists())
    del consumer, optimizer, parameters, evaluators
    torch.cuda.empty_cache()
    return {"status": "PASS", "terminal_checkpoint": terminal_path.relative_to(target).as_posix()}


def _copy_endpoint_state(consumer: Any, payload: Mapping[str, Any]) -> None:
    parameters = capacity._parameters(consumer)
    _require(payload.get("task_id") in {TASK_ID, historical_training.TASK_ID}, "measurement checkpoint task identity differs")
    _require(payload.get("completed_updates") in {SOURCE_STEP, SOURCE_STEP + UPDATES_PER_ARM, 5_400}, "measurement checkpoint step is outside the declared endpoints")
    phase_state = payload.get("trainable_parameter_state", {})
    _require(set(phase_state) == set(parameters), "measurement checkpoint phase groups are incomplete")
    with torch.no_grad():
        for name, parameter in parameters.items():
            _require(parameter.shape == phase_state[name].shape and parameter.dtype == phase_state[name].dtype, f"measurement checkpoint geometry differs for {name}")
            parameter.copy_(phase_state[name].to(parameter.device))
    capacity._assert_state_equal(payload["frozen_runtime_state"], capacity._frozen_runtime_state(consumer), "measurement endpoint frozen runtime")


def _dev_row(
    raw: torch.Tensor,
    target_raw: torch.Tensor,
    logits: Mapping[str, torch.Tensor],
    *,
    scenario: str,
    position: int,
    source_row: int,
    label: int,
    pixel_area_m2: float,
) -> dict[str, Any]:
    q = gradient._qfg_components(raw, target_raw)
    return {
        "scenario": scenario,
        "position": position,
        "source_row_index": source_row,
        "label": label,
        "role": "dev",
        "q_fg": float(q["q_fg"][0].cpu()),
        "q_bg": float(q["q_bg"][0].cpu()),
        "q_raw200": float(q["q_raw200"][0].cpu()),
        "raw_cmos_integral": float(raw.to(torch.float64).sum().cpu()) * pixel_area_m2,
        "clipped_fraction": float((raw > 1.0).to(torch.float64).mean().cpu()),
        **{f"{name.lower()}_prediction": int(logits[name].argmax(1).item()) for name in EVALUATOR_NAMES},
    }


def _save_dev_readout(
    target: Path,
    inputs: Mapping[str, Any],
    evaluators: Mapping[str, nn.Module],
    consumer: Any,
    scenario: str,
    positions: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    scenario_dir = target / "readouts" / scenario
    saved_csv = scenario_dir / "paired_rows.csv"
    selected_positions = list(range(DEV_ROWS)) if positions is None else [int(position) for position in positions]
    _require(bool(selected_positions) and len(set(selected_positions)) == len(selected_positions), "DEV diagnostic positions must be nonempty and unique")
    _require(all(0 <= position < DEV_ROWS for position in selected_positions), "DEV diagnostic position is outside the declared population")
    class_counts = torch.bincount(inputs["dev_labels"].index_select(0, torch.tensor(selected_positions)), minlength=10).tolist()
    _require(len(selected_positions) % 10 == 0 and len(set(class_counts)) == 1 and class_counts[0] > 0, "DEV readout positions must contain an equal number of rows per class")
    if saved_csv.is_file():
        rows = _read_csv(saved_csv, DEV_FIELDS)
        _require(len(rows) == len(selected_positions), f"saved DEV row count differs for {scenario}")
        return _recheck_dev_scenario(target, inputs, scenario, selected_positions)
    output_rows: list[dict[str, Any]] = []
    for chunk_number, start in enumerate(range(0, len(selected_positions), CHUNK_ROWS)):
        chunk_positions = selected_positions[start : start + CHUNK_ROWS]
        raw_parts, target_parts, view_parts = [], [], []
        logit_parts = {name: [] for name in EVALUATOR_NAMES}
        records = []
        with torch.inference_mode():
            for position in chunk_positions:
                image = inputs["dev_images"][position : position + 1].to(device=consumer.device, dtype=torch.float32)
                result = consumer.runtime.forward_from_image(image)
                raw = result.raw_intensity
                target_raw = result.image_input.slm_intensity
                view = diagnostic.canonical.cmos_view(raw)
                logits = _evaluator_logits(evaluators, view)
                _require(bool(torch.isfinite(raw).all() and torch.isfinite(target_raw).all() and (raw >= 0).all()), f"DEV readout is invalid at {scenario}/{position}")
                raw_parts.append(raw.detach().cpu())
                target_parts.append(target_raw.detach().cpu())
                view_parts.append(view.detach().cpu())
                for name in EVALUATOR_NAMES:
                    logit_parts[name].append(logits[name].detach().cpu())
                record = _dev_row(
                    raw.detach().cpu(), target_raw.detach().cpu(), {name: logits[name].detach().cpu() for name in EVALUATOR_NAMES},
                    scenario=scenario, position=position, source_row=int(inputs["dev_rows"][position]),
                    label=int(inputs["dev_labels"][position]), pixel_area_m2=float(inputs["sources"]["source"]["pixel_area_m2"]),
                )
                records.append(record)
                del image, result, raw, target_raw, view, logits
        payload = {
            "task_id": TASK_ID,
            "scenario": scenario,
            "positions": torch.tensor(chunk_positions, dtype=torch.long),
            "source_row_indices": inputs["dev_rows"].index_select(0, torch.tensor(chunk_positions)).clone(),
            "labels": inputs["dev_labels"].index_select(0, torch.tensor(chunk_positions)).clone(),
            "roles": ["dev"] * len(chunk_positions),
            "raw_cmos": torch.cat(raw_parts).contiguous(),
            "target_slm_intensity": torch.cat(target_parts).contiguous(),
            "evaluator_view": torch.cat(view_parts).contiguous(),
            "logits": {name: torch.cat(values).contiguous() for name, values in logit_parts.items()},
        }
        path = scenario_dir / f"chunk_{chunk_number:04d}.pt"
        if path.is_file():
            old = torch.load(path, map_location="cpu", weights_only=True)
            _require(old.get("task_id") == TASK_ID and old.get("scenario") == scenario and all(capacity._tree_equal(old.get(key), payload.get(key)) for key in ("positions", "source_row_indices", "labels", "raw_cmos", "target_slm_intensity", "evaluator_view", "logits")), f"existing saved DEV chunk differs: {path.name}")
        else:
            _write_torch(path, payload)
        output_rows.extend(records)
    _write_csv(saved_csv, output_rows, DEV_FIELDS)
    return _recheck_dev_scenario(target, inputs, scenario, selected_positions)


def _recheck_dev_scenario(
    target: Path,
    inputs: Mapping[str, Any],
    scenario: str,
    positions: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    scenario_dir = target / "readouts" / scenario
    csv_rows = _read_csv(scenario_dir / "paired_rows.csv", DEV_FIELDS)
    selected_positions = list(range(DEV_ROWS)) if positions is None else [int(position) for position in positions]
    _require(len(csv_rows) == len(selected_positions), f"saved DEV CSV does not contain {len(selected_positions):,} rows for {scenario}")
    by_position = {int(row["position"]): row for row in csv_rows}
    _require(set(by_position) == set(selected_positions), f"saved DEV CSV has missing or duplicate rows for {scenario}")
    expected_per_class = len(selected_positions) // 10
    _require(expected_per_class > 0 and len(selected_positions) == expected_per_class * 10, "saved DEV recheck needs a balanced ten-class population")
    expected_labels = inputs["dev_labels"].index_select(0, torch.tensor(selected_positions))
    _require(torch.bincount(expected_labels, minlength=10).tolist() == [expected_per_class] * 10, "saved DEV recheck positions are not class balanced")
    recomputed = []
    for chunk_number, start in enumerate(range(0, len(selected_positions), CHUNK_ROWS)):
        chunk_positions = selected_positions[start : start + CHUNK_ROWS]
        path = scenario_dir / f"chunk_{chunk_number:04d}.pt"
        _require(path.is_file(), f"saved DEV raw chunk is missing: {scenario}/{path.name}")
        chunk = torch.load(path, map_location="cpu", weights_only=True)
        expected_tensor = torch.tensor(chunk_positions, dtype=torch.long)
        expected_rows = inputs["dev_rows"].index_select(0, expected_tensor)
        expected_chunk_labels = inputs["dev_labels"].index_select(0, expected_tensor)
        _require(chunk.get("task_id") == TASK_ID and chunk.get("scenario") == scenario and torch.equal(chunk.get("positions"), expected_tensor), f"saved DEV chunk identity or ordering differs: {scenario}/{chunk_number}")
        _require(torch.equal(chunk["source_row_indices"], expected_rows) and torch.equal(chunk["labels"], expected_chunk_labels) and chunk["roles"] == ["dev"] * len(chunk_positions), f"saved DEV row identity or role differs: {scenario}/{chunk_number}")
        replay_view = diagnostic.canonical.cmos_view(chunk["raw_cmos"])
        _require(torch.allclose(replay_view, chunk["evaluator_view"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"saved DEV production view does not replay from raw intensity: {scenario}/{chunk_number}")
        for name in EVALUATOR_NAMES:
            _require(tuple(chunk["logits"][name].shape) == (len(chunk_positions), 10) and bool(torch.isfinite(chunk["logits"][name]).all()), f"saved DEV {name} logits are incomplete")
        _require(tuple(chunk["raw_cmos"].shape) == (len(chunk_positions), 1, 200, 200) and bool(torch.isfinite(chunk["raw_cmos"]).all() and (chunk["raw_cmos"] >= 0).all()), f"saved DEV raw CMOS is invalid: {scenario}/{chunk_number}")
        for offset, position in enumerate(chunk_positions):
            row = _dev_row(
                chunk["raw_cmos"][offset : offset + 1], chunk["target_slm_intensity"][offset : offset + 1],
                {name: chunk["logits"][name][offset : offset + 1] for name in EVALUATOR_NAMES}, scenario=scenario,
                position=position, source_row=int(inputs["dev_rows"][position]), label=int(inputs["dev_labels"][position]),
                pixel_area_m2=float(inputs["sources"]["source"]["pixel_area_m2"]),
            )
            observed = by_position[position]
            for field in DEV_FIELDS:
                if field in {"scenario", "role"}:
                    _require(observed[field] == str(row[field]), f"saved DEV {field} differs from its chunk: {scenario}/{position}")
                elif field in {"position", "source_row_index", "label", "e_s_prediction", "e_i_prediction", "e_c_prediction", "e_d_prediction"}:
                    _require(int(observed[field]) == int(row[field]), f"saved DEV {field} differs from its chunk: {scenario}/{position}")
                else:
                    _require(math.isclose(float(observed[field]), float(row[field]), rel_tol=1.0e-12, abs_tol=1.0e-14), f"saved DEV {field} differs from independent recomputation: {scenario}/{position}")
            recomputed.append(row)
    _require(all(sum(int(row["label"]) == digit for row in recomputed) == expected_per_class for digit in range(10)), f"saved DEV class coverage differs for {scenario}")
    return recomputed


def _sample_rows_for_seed(seed: int, bank: Mapping[str, Any], draws_per_class: int = SAMPLES_PER_CLASS) -> list[dict[str, Any]]:
    _require(1 <= draws_per_class <= SAMPLES_PER_CLASS, "conditional draw count is outside the saved sample bank")
    seed_index = SAMPLE_SEEDS.index(seed)
    rows = []
    flat = 0
    for digit in range(10):
        for draw in range(draws_per_class):
            rows.append({"position": flat, "sample_seed": seed, "requested_class": digit, "draw_index": draw, "z": bank["z"][seed_index, digit, draw]})
            flat += 1
    return rows


def _index_sample_rows(rows: Sequence[Mapping[str, Any]], scenario: str, seed: int, draws_per_class: int = SAMPLES_PER_CLASS) -> dict[tuple[int, int], Mapping[str, Any]]:
    _require(1 <= draws_per_class <= SAMPLES_PER_CLASS, "conditional draw count is outside the saved sample bank")
    _require(len(rows) == 10 * draws_per_class, f"conditional sample CSV is incomplete: {scenario}/{seed}")
    _require(all(set(SAMPLE_FIELDS) <= set(row) for row in rows), f"conditional sample CSV fields are incomplete: {scenario}/{seed}")
    by_class_draw = {(int(row["requested_class"]), int(row["draw_index"])): row for row in rows}
    expected = {(digit, draw) for digit in range(10) for draw in range(draws_per_class)}
    _require(len(by_class_draw) == len(rows) and set(by_class_draw) == expected, f"conditional sample CSV has missing or duplicate draws: {scenario}/{seed}")
    return by_class_draw


def _save_sample_readout(
    target: Path,
    inputs: Mapping[str, Any],
    evaluators: Mapping[str, nn.Module],
    consumer: Any,
    scenario: str,
    sample_seeds: Sequence[int] = SAMPLE_SEEDS,
    draws_per_class: int = SAMPLES_PER_CLASS,
) -> list[dict[str, Any]]:
    all_rows = []
    bank = inputs["sample_bank"]
    _require(bool(sample_seeds) and len(set(sample_seeds)) == len(sample_seeds) and set(sample_seeds) <= set(SAMPLE_SEEDS), "conditional sample seeds must be unique members of the fixed sample bank")
    for seed in sample_seeds:
        seed_rows = _sample_rows_for_seed(seed, bank, draws_per_class)
        seed_dir = target / "readouts" / "conditional_samples" / scenario / f"seed_{seed}"
        csv_path = seed_dir / "paired_rows.csv"
        if csv_path.is_file():
            saved = _read_csv(csv_path, SAMPLE_FIELDS)
            _require(len(saved) == 10 * draws_per_class, f"saved conditional readout is incomplete: {scenario}/{seed}")
            all_rows.extend(saved)
            continue
        rows_for_seed = []
        for chunk_number, start in enumerate(range(0, len(seed_rows), CHUNK_ROWS)):
            entries = seed_rows[start : start + CHUNK_ROWS]
            raw_parts, view_parts, z_parts, input_power_parts = [], [], [], []
            logits_parts = {name: [] for name in EVALUATOR_NAMES}
            records = []
            with torch.inference_mode():
                for entry in entries:
                    z = entry["z"].unsqueeze(0).to(device=consumer.device)
                    result = consumer.runtime.forward_from_ols(z)
                    raw = result.raw_intensity
                    view = diagnostic.canonical.cmos_view(raw)
                    logits = _evaluator_logits(evaluators, view)
                    input_power = diagnostic.bogt.physical_plane_power(result.generator_input_field).to(torch.float64)
                    _require(bool(torch.isfinite(raw).all() and torch.isfinite(input_power).all() and (raw >= 0).all()), f"conditional output is invalid: {scenario}/{seed}/{entry['position']}")
                    raw_parts.append(raw.detach().cpu())
                    view_parts.append(view.detach().cpu())
                    z_parts.append(entry["z"].clone())
                    input_power_parts.append(input_power.detach().cpu())
                    for name in EVALUATOR_NAMES:
                        logits_parts[name].append(logits[name].detach().cpu())
                    record = {
                        "scenario": scenario,
                        "sample_seed": seed,
                        "draw_index": entry["draw_index"],
                        "requested_class": entry["requested_class"],
                        **{f"{name.lower()}_prediction": int(logits[name].argmax(1).item()) for name in EVALUATOR_NAMES},
                        "raw_cmos_integral": float(raw.to(torch.float64).sum().cpu()) * float(inputs["sources"]["source"]["pixel_area_m2"]),
                        "clipped_fraction": float((raw > 1.0).to(torch.float64).mean().cpu()),
                        "generator_input_power": float(input_power.mean().cpu()),
                    }
                    records.append(record)
                    del z, result, raw, view, logits, input_power
            payload = {
                "task_id": TASK_ID,
                "scenario": scenario,
                "sample_seed": seed,
                "flat_start": start,
                "rows": [{key: value for key, value in entry.items() if key != "z"} for entry in entries],
                "z": torch.stack(z_parts),
                "raw_cmos": torch.cat(raw_parts),
                "evaluator_view": torch.cat(view_parts),
                "e_d_logits": torch.cat(logits_parts["E_D"]),
                "logits": {name: torch.cat(values) for name, values in logits_parts.items()},
                "generator_input_power": torch.cat(input_power_parts).reshape(-1),
            }
            path = seed_dir / f"chunk_{chunk_number:04d}.pt"
            if path.is_file():
                old = torch.load(path, map_location="cpu", weights_only=True)
                _require(old.get("task_id") == TASK_ID and old.get("scenario") == scenario and old.get("sample_seed") == seed and capacity._tree_equal(old, payload), f"existing conditional chunk differs: {path.name}")
            else:
                _write_torch(path, payload)
            rows_for_seed.extend(records)
        _write_csv(csv_path, rows_for_seed, SAMPLE_FIELDS)
        all_rows.extend(rows_for_seed)
    return all_rows


def _z_features(z: torch.Tensor) -> torch.Tensor:
    return prior_reference._to_interleaved(z)


def _distance_decomposition(generated: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    generated, target = generated.to(torch.float64), target.to(torch.float64)
    cross = float(torch.cdist(generated, target, p=2).mean())
    within_generated = float(torch.cdist(generated, generated, p=2).mean())
    within_target = float(torch.cdist(target, target, p=2).mean())
    return {"cross_mean_distance": cross, "generated_within_mean_distance": within_generated, "fit_target_within_mean_distance": within_target, "energy_distance": 2.0 * cross - within_generated - within_target}


def _summarize_samples(target: Path, inputs: Mapping[str, Any], all_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _write_csv(target / "conditional_sample_rows.csv", all_rows, SAMPLE_FIELDS)
    saved_rows = _read_csv(target / "conditional_sample_rows.csv", SAMPLE_FIELDS)
    summaries = []
    fit_labels = inputs["fit_labels"]
    fit_x = _z_features(inputs["ols_coefficients"])
    for scenario in ("A4700", *(f"seed_{seed}_{arm}_step{SOURCE_STEP + UPDATES_PER_ARM}" for seed in TRAIN_SEEDS for arm in ("A", "B"))):
        for seed in SAMPLE_SEEDS:
            directory = target / "readouts" / "conditional_samples" / scenario / f"seed_{seed}"
            chunks = [torch.load(directory / f"chunk_{index:04d}.pt", map_location="cpu", weights_only=True) for index in range(6)]
            _require(all(item.get("task_id") == TASK_ID and item.get("scenario") == scenario and item.get("sample_seed") == seed for item in chunks), f"conditional sample chunks have a mismatched identity: {scenario}/{seed}")
            raw = torch.cat([item["raw_cmos"] for item in chunks])
            z = torch.cat([item["z"] for item in chunks])
            logits = {name: torch.cat([item["logits"][name] for item in chunks]) for name in EVALUATOR_NAMES}
            input_power = torch.cat([item["generator_input_power"] for item in chunks])
            rows = [row for row in saved_rows if row["scenario"] == scenario and int(row["sample_seed"]) == seed]
            _require(len(rows) == 300 and tuple(raw.shape) == (300, 1, 200, 200) and tuple(z.shape) == (300, 100), f"conditional sample row count differs: {scenario}/{seed}")
            for name in EVALUATOR_NAMES:
                _require(torch.equal(logits[name].argmax(1), torch.tensor([int(row[f"{name.lower()}_prediction"]) for row in rows])), f"conditional CSV predictions differ from saved logits: {scenario}/{seed}/{name}")
            requested = torch.arange(10).repeat_interleave(SAMPLES_PER_CLASS)
            sample_x = _z_features(z)
            for digit in range(10):
                positions = torch.nonzero(requested == digit, as_tuple=False).flatten()
                generated_shape = frozen._shape_vectors(raw.index_select(0, positions), 28)
                target_shape = inputs["fit_target_shape_vectors"][fit_labels == digit]
                distance = _distance_decomposition(generated_shape, target_shape)
                standardized = (sample_x.index_select(0, positions) - inputs["priors"]["means"][digit]) / inputs["priors"]["scales"][digit]
                fit_standardized = (fit_x[fit_labels == digit] - inputs["priors"]["means"][digit]) / inputs["priors"]["scales"][digit]
                nearest = torch.cdist(standardized, fit_standardized, p=2).min(dim=1).values
                selected_rows = [rows[int(index)] for index in positions]
                summaries.append({
                    "scenario": scenario,
                    "sample_seed": seed,
                    "requested_class": digit,
                    "samples": SAMPLES_PER_CLASS,
                    **distance,
                    "e_s_correct": sum(int(row["e_s_prediction"]) == digit for row in selected_rows),
                    "e_i_correct": sum(int(row["e_i_prediction"]) == digit for row in selected_rows),
                    "e_c_correct": sum(int(row["e_c_prediction"]) == digit for row in selected_rows),
                    "e_d_correct": sum(int(row["e_d_prediction"]) == digit for row in selected_rows),
                    "mean_within_class_generated_distance": distance["generated_within_mean_distance"],
                    "nearest_fit_ols_distance_mean": float(nearest.mean()),
                    "nearest_fit_ols_distance_min": float(nearest.min()),
                    "nearest_fit_ols_distance_max": float(nearest.max()),
                    "mean_raw_cmos_integral": math.fsum(float(row["raw_cmos_integral"]) for row in selected_rows) / SAMPLES_PER_CLASS,
                    "mean_generator_input_power": float(input_power.index_select(0, positions).mean()),
                    "mean_raw_clipped_fraction": math.fsum(float(row["clipped_fraction"]) for row in selected_rows) / SAMPLES_PER_CLASS,
                })
    _require(len(summaries) == 7 * len(SAMPLE_SEEDS) * 10, "conditional sample summary is incomplete")
    _write_csv(target / "conditional_sample_class_summary.csv", summaries, tuple(summaries[0].keys()))
    result = {
        "task_id": TASK_ID,
        "status": "DESCRIPTIVE_ONLY",
        "scenarios": 7,
        "sample_seeds": list(SAMPLE_SEEDS),
        "rows_per_scenario": len(SAMPLE_SEEDS) * 10 * SAMPLES_PER_CLASS,
        "class_summary_rows": len(summaries),
        "metrics": "requested-class evaluator reads, 28x28 normalized shape energy-distance decomposition, generated within-class distance, fit OLS nearest distance, output intensity and input power",
        "selection_or_readability_gate": False,
    }
    _write_json(target / "conditional_sample_summary.json", result)
    return result


def _recheck_sample_scenario(
    target: Path,
    inputs: Mapping[str, Any],
    scenario: str,
    sample_seeds: Sequence[int] = SAMPLE_SEEDS,
    draws_per_class: int = SAMPLES_PER_CLASS,
) -> int:
    total = 0
    pixel_area = float(inputs["sources"]["source"]["pixel_area_m2"])
    bank = inputs["sample_bank"]["z"]
    _require(bool(sample_seeds) and len(set(sample_seeds)) == len(sample_seeds) and set(sample_seeds) <= set(SAMPLE_SEEDS), "conditional sample seeds must be unique members of the fixed sample bank")
    _require(1 <= draws_per_class <= SAMPLES_PER_CLASS, "conditional draw count is outside the saved sample bank")
    for seed in sample_seeds:
        seed_index = SAMPLE_SEEDS.index(seed)
        directory = target / "readouts" / "conditional_samples" / scenario / f"seed_{seed}"
        rows = _read_csv(directory / "paired_rows.csv", SAMPLE_FIELDS)
        by_class_draw = _index_sample_rows(rows, scenario, seed, draws_per_class)
        total_seed_rows = 10 * draws_per_class
        expected_chunk_count = math.ceil(total_seed_rows / CHUNK_ROWS)
        for chunk_index in range(expected_chunk_count):
            start = chunk_index * CHUNK_ROWS
            expected_count = min(CHUNK_ROWS, total_seed_rows - start)
            chunk = torch.load(directory / f"chunk_{chunk_index:04d}.pt", map_location="cpu", weights_only=True)
            _require(chunk.get("task_id") == TASK_ID and chunk.get("scenario") == scenario and chunk.get("sample_seed") == seed and chunk.get("flat_start") == start, f"conditional sample chunk identity differs: {scenario}/{seed}/{chunk_index}")
            _require(tuple(chunk["raw_cmos"].shape) == (expected_count, 1, 200, 200) and tuple(chunk["z"].shape) == (expected_count, 100), f"conditional sample chunk geometry differs: {scenario}/{seed}/{chunk_index}")
            _require(torch.allclose(diagnostic.canonical.cmos_view(chunk["raw_cmos"]), chunk["evaluator_view"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"conditional production view does not replay from raw output: {scenario}/{seed}/{chunk_index}")
            for name in EVALUATOR_NAMES:
                _require(tuple(chunk["logits"][name].shape) == (expected_count, 10) and bool(torch.isfinite(chunk["logits"][name]).all()), f"conditional {name} logits are incomplete")
            expected_rows = []
            for offset in range(expected_count):
                flat = start + offset
                digit, draw = divmod(flat, draws_per_class)
                _require(chunk["rows"][offset] == {"position": flat, "sample_seed": seed, "requested_class": digit, "draw_index": draw}, f"conditional chunk-to-bank mapping differs: {scenario}/{seed}/{flat}")
                _require(torch.equal(chunk["z"][offset], bank[seed_index, digit, draw]), f"conditional OLS code differs from the common sample bank: {scenario}/{seed}/{flat}")
                raw = chunk["raw_cmos"][offset : offset + 1]
                logits = {name: chunk["logits"][name][offset : offset + 1] for name in EVALUATOR_NAMES}
                row = {
                    "scenario": scenario,
                    "sample_seed": seed,
                    "draw_index": draw,
                    "requested_class": digit,
                    **{f"{name.lower()}_prediction": int(logits[name].argmax(1).item()) for name in EVALUATOR_NAMES},
                    "raw_cmos_integral": float(raw.to(torch.float64).sum()) * pixel_area,
                    "clipped_fraction": float((raw > 1.0).to(torch.float64).mean()),
                    "generator_input_power": float(chunk["generator_input_power"][offset]),
                }
                expected_rows.append(row)
                observed = by_class_draw[(digit, draw)]
                for field in SAMPLE_FIELDS:
                    if field in {"scenario"}:
                        _require(observed[field] == str(row[field]), f"conditional CSV scenario differs from raw chunk: {scenario}/{seed}/{draw}")
                    elif field in {"sample_seed", "draw_index", "requested_class", "e_s_prediction", "e_i_prediction", "e_c_prediction", "e_d_prediction"}:
                        _require(int(observed[field]) == int(row[field]), f"conditional CSV {field} differs from raw chunk: {scenario}/{seed}/{draw}")
                    else:
                        _require(math.isclose(float(observed[field]), float(row[field]), rel_tol=1.0e-12, abs_tol=1.0e-14), f"conditional CSV {field} differs from raw chunk: {scenario}/{seed}/{draw}")
            total += len(expected_rows)
    _require(total == len(sample_seeds) * 10 * draws_per_class, f"conditional sample row count differs for {scenario}")
    return total


def _recompute_saved_primary_decision(evidence_dir: Path, inputs: Mapping[str, Any], saved_report: Mapping[str, Any]) -> dict[str, Any]:
    scenarios = (
        "A4700",
        *(f"seed_{seed}_{arm}_step{SOURCE_STEP + UPDATES_PER_ARM}" for seed in TRAIN_SEEDS for arm in ("A", "B")),
    )
    rows = []
    for scenario in scenarios:
        rows.extend(_recheck_dev_scenario(evidence_dir, inputs, scenario))
    recomputed = _decision(rows)
    expected = saved_report.get("decision")
    _require(isinstance(expected, Mapping) and recomputed == expected, "raw DEV chunks do not reproduce the prior full-run decision")
    _require(len(rows) == 7 * DEV_ROWS, "historical primary raw DEV readout is incomplete")
    return {
        "executed": True,
        "passed": True,
        "scenarios": list(scenarios),
        "rows_recomputed_from_raw_chunks": len(rows),
        "decision": recomputed,
        "matches_saved_report": True,
        "historical_A5400_B5400_not_used_for_decision": True,
    }


def _saved_evidence_pipeline_diagnostic(target: Path, evidence_dir: Path, inputs: Mapping[str, Any], saved_report: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    evaluators = generation.diagnostic.load_evaluators(inputs["sources"]["source"]["generation_paths"], device)
    evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    consumer, _optimizer, parameters = _new_endpoint(inputs["sources"], device)
    labels = inputs["dev_labels"].tolist()
    positions_by_class = {digit: [] for digit in range(10)}
    for position, label in enumerate(labels):
        if len(positions_by_class[int(label)]) < 2:
            positions_by_class[int(label)].append(position)
    positions = [position for digit in range(10) for position in positions_by_class[digit]]
    _require(len(positions) == 20 and all(len(positions_by_class[digit]) == 2 for digit in range(10)), "cannot select the fixed two-per-class DEV diagnostic subset")
    dev_rows = _save_dev_readout(target, inputs, evaluators, consumer, "A4700", positions=positions)
    _require(len(dev_rows) == 20 and {int(row["label"]) for row in dev_rows} == set(range(10)), "20-row DEV production readout did not cover all classes")
    sample_seeds = list(SAMPLE_SEEDS[:2])
    sample_rows = _save_sample_readout(
        target, inputs, evaluators, consumer, "A4700", sample_seeds=sample_seeds, draws_per_class=2,
    )
    sample_rows_rechecked = _recheck_sample_scenario(
        target, inputs, "A4700", sample_seeds=sample_seeds, draws_per_class=2,
    )
    _require(len(sample_rows) == 40 and sample_rows_rechecked == 40, "fixed conditional sample diagnostic did not save and re-read 40 rows")
    raw_decision = _recompute_saved_primary_decision(evidence_dir, inputs, saved_report)
    for name, model in evaluators.items():
        capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"diagnostic frozen evaluator {name}")
    capacity._assert_state_equal(inputs["sources"]["source"]["checkpoint"]["frozen_runtime_state"], capacity._frozen_runtime_state(consumer), "diagnostic frozen source runtime")
    del consumer, _optimizer, parameters, evaluators
    torch.cuda.empty_cache()
    return {
        "executed": True,
        "passed": True,
        "dev_rows_saved_and_rechecked": len(dev_rows),
        "dev_class_counts": {str(digit): sum(int(row["label"]) == digit for row in dev_rows) for digit in range(10)},
        "conditional_sample_rows_saved_and_rechecked": sample_rows_rechecked,
        "conditional_sample_seeds": sample_seeds,
        "draws_per_class": 2,
        "historical_primary_raw_decision": raw_decision,
        "frozen_evaluators_unchanged": True,
        "source_runtime_unchanged": True,
    }


def _write_microbatch_performance_table(path: Path, check: Mapping[str, Any]) -> None:
    fields = ("method", "microbatch_size", "arm", "equivalence_passed", "actual_update_verified", "warmups", "timed_runs", "median_seconds", "peak_allocated_bytes", "recommended")
    rows = []
    for arm, timing in check["original_production_arm_timing"].items():
        rows.append({"method": "original_production", "microbatch_size": "", "arm": arm, "equivalence_passed": True, "actual_update_verified": "reference", "warmups": timing["warmups"], "timed_runs": json.dumps(timing["timed_runs"]), "median_seconds": timing["median_seconds"], "peak_allocated_bytes": timing["peak_allocated_bytes"], "recommended": False})
    for candidate in check["microbatch_candidates"]:
        if candidate["equivalence_passed"]:
            for arm, timing in candidate["arm_timing"].items():
                rows.append({"method": "merged_microbatch", "microbatch_size": candidate["microbatch_size"], "arm": arm, "equivalence_passed": True, "actual_update_verified": candidate["actual_update_verified"], "warmups": timing["warmups"], "timed_runs": json.dumps(timing["timed_runs"]), "median_seconds": timing["median_seconds"], "peak_allocated_bytes": candidate["peak_allocated_bytes"], "recommended": candidate["microbatch_size"] == check["recommended_microbatch_size"]})
        else:
            rows.append({"method": "merged_microbatch", "microbatch_size": candidate["microbatch_size"], "arm": "", "equivalence_passed": False, "actual_update_verified": False, "warmups": 0, "timed_runs": "[]", "median_seconds": "", "peak_allocated_bytes": "", "recommended": False})
    _write_csv(path, rows, fields, replace=True)


def _write_engineering_delivery_manifests(output_root: Path, target: Path, evidence_dir: Path, required_dependency: str) -> dict[str, Any]:
    attempt_relative = target.relative_to(output_root).as_posix()
    package_root = target / "review_package"
    package_manifest = _json(package_root / "review_manifest.json")
    review_files = [f"{attempt_relative}/review_package/{relative}" for relative in package_manifest["files"]]
    review_files.append(f"{attempt_relative}/review_package/package_report.json")
    required_paths = [
        "plan.md", "progress.json", "engineering_update_budget.json", "diagnostic_report.json",
        "diagnostic_validation.json", "review_delivery_manifest.json", "diagnostic_audit_manifest.json",
        "historical_audit_manifest.json", "resource_telemetry.csv",
        f"{attempt_relative}/plan.md", f"{attempt_relative}/input_probe.json", f"{attempt_relative}/input_population.json",
        f"{attempt_relative}/batch_orders.json", f"{attempt_relative}/preflight.json", f"{attempt_relative}/preflight_trials.json",
        f"{attempt_relative}/microbatch_performance.csv", f"{attempt_relative}/saved_evidence_pipeline.json",
        f"{attempt_relative}/diagnostic_report.json", f"{attempt_relative}/diagnostic_validation.json", f"{attempt_relative}/closeout.md",
        f"{attempt_relative}/preflight/repeatability_envelope.json", f"{attempt_relative}/review_package/package_report.json",
        f"{attempt_relative}/readouts/A4700/paired_rows.csv",
        f"{attempt_relative}/readouts/conditional_samples/A4700/seed_{SAMPLE_SEEDS[0]}/paired_rows.csv",
        f"{attempt_relative}/readouts/conditional_samples/A4700/seed_{SAMPLE_SEEDS[1]}/paired_rows.csv",
    ]
    recovery_comparisons = sorted(target.glob("preflight/file_recovery/attempt_*/recovery_comparison.json"))
    _require(len(recovery_comparisons) == 3, "review manifest requires all three file recovery comparisons")
    review_files.extend(f"{attempt_relative}/{path.relative_to(target).as_posix()}" for path in recovery_comparisons)
    review_files = sorted(set([*review_files, *required_paths]))
    dependency_list = sorted(set([*package_manifest.get("required_local_dependencies", []), required_dependency]))

    import scripts.maintenance.autodl_review as review_tool

    audit_sources, historical_dependencies = review_tool._review_files(evidence_dir, "audit")
    historical_manifest = {
        "schema": "autodl_review_manifest_v1",
        "profile": "audit",
        "remote_run_dir": evidence_dir.relative_to(_root()).as_posix(),
        "files": sorted(path.relative_to(evidence_dir).as_posix() for path in audit_sources),
        "required_local_dependencies": sorted(set(historical_dependencies)),
        "purpose": "explicit full raw-pixel audit inventory for the existing named science run; not included in the review transfer",
    }
    _write_json(output_root / "historical_audit_manifest.json", historical_manifest, replace=True)

    diagnostic_raw = sorted(path for path in target.glob("readouts/**/*.pt") if path.is_file())
    recovery_weights = sorted(path for path in target.glob("preflight/file_recovery/attempt_*/checkpoints/*.pt") if path.is_file())
    diagnostic_audit = {
        "schema": "autodl_review_manifest_v1",
        "profile": "audit",
        "remote_run_dir": output_root.relative_to(_root()).as_posix(),
        "files": sorted([
            "diagnostic_audit_manifest.json",
            *review_files,
            *[f"{attempt_relative}/{path.relative_to(target).as_posix()}" for path in diagnostic_raw],
            *[f"{attempt_relative}/{path.relative_to(target).as_posix()}" for path in recovery_weights],
        ]),
        "required_local_dependencies": dependency_list,
        "purpose": "explicit diagnostic raw-readout and complete recovery-checkpoint audit inventory; excluded from the compact review transfer",
    }
    _write_json(output_root / "diagnostic_audit_manifest.json", diagnostic_audit, replace=True)
    review_manifest = {
        "schema": "autodl_review_manifest_v1",
        "profile": "review",
        "remote_run_dir": output_root.relative_to(_root()).as_posix(),
        "files": review_files,
        "required_local_dependencies": dependency_list,
        "limitations": ["review profile omits raw diagnostic pixel chunks and full recovery checkpoint tensors; see diagnostic_audit_manifest.json", "historical full raw-pixel audit inventory is separately listed in historical_audit_manifest.json"],
    }
    _write_json(output_root / "review_delivery_manifest.json", review_manifest, replace=True)
    return {
        "status": "PREPARED",
        "review_file_count": len(review_files),
        "review_manifest": "review_delivery_manifest.json",
        "diagnostic_audit_manifest": "diagnostic_audit_manifest.json",
        "historical_audit_manifest": "historical_audit_manifest.json",
        "resource_telemetry_expected_from_autodl_run_wrapper": "resource_telemetry.csv",
        "raw_diagnostic_readout_files": len(diagnostic_raw),
        "recovery_checkpoint_files": len(recovery_weights),
    }


def _engineering_closeout_markdown(report: Mapping[str, Any]) -> str:
    checks = report["preflight"]["checks"]
    speed = checks["merged_production_gradient_and_speed"]
    pipeline = report["saved_evidence_pipeline"]
    package = report["review_package"]
    return "\n".join([
        "# AutoDL训练工作流工程诊断收尾",
        "",
        "## 结果与状态",
        "",
        f"工程诊断：`{report['status']}`，范围标记为`diagnostic-only`。完成{report['engineering_adam_updates']}次本次获准工程Adam更新（新增上限42、累计上限45）；科学更新为0，未打开正式训练attempt或消费READY凭据。",
        "",
        "## 科学认识",
        "",
        "科学判断未改变。本轮只核验恢复、计算路径、保存回读、存储和交付流程，不重新解释既有训练结果。",
        "",
        "## 目标缺口",
        "",
        "训练方法是否稳定推广、人眼可读性、模型采用和物理部署均不由工程诊断回答，仍需各自预先定义的研究或验收证据。",
        "",
        "## 工程完成与验证",
        "",
        f"恢复采用两个独立进程：保存进程完成1步、原子写入并回读checkpoint后退出；恢复进程消费显式文件并续做2步。五次同机连续三步重复的相位和Adam最大成对绝对差/RMS包络在第一次恢复前保存；三次恢复均要求边界模型、Adam、RNG、批次与历史前缀精确，恢复终点差异不得超过该固定包络。任务级预算通过原子预留累计，历史3次保留计数，本次新增最多42次、累计不超过45次；本次记录{report['engineering_adam_updates']}次。",
        f"六条三步轨迹覆盖请求微批1/2/5/10/1/1；失败候选回退到1。每个通过候选均比较原生产梯度并验证实际首步Adam更新，性能表对A/B两臂及原生产方法各做2次预热和5次计时；等权A/B中位耗时建议微批{speed['recommended_microbatch_size']}。六条轨迹、零lambda负控、Encoder/评价器隔离、OLS来源重算、独立Q_fg值/导数和恢复证据见`preflight.json`及相关小报告。",
        f"保存回读包括{pipeline['dev_rows_saved_and_rechecked']}行十类DEV诊断、{pipeline['conditional_sample_rows_saved_and_rechecked']}行条件样本，以及从既有完整raw块重算并匹配原报告的主终点判定。Review包为{package['package_bytes_including_manifest']}字节，包含三张PNG，总图像大小不超过2 MiB；分级清单将恢复checkpoint和诊断raw列入单独audit。精确回传与Windows读回状态见`review_delivery_manifest.json`及本地收据。",
        "",
        "## 决策影响与下一步",
        "",
        "本结果仅说明本轮被执行的工程检查是否有效；建议在工程验证通过后采用已记录的500步保存、双恢复点、低频GPT检查和review/audit分级交付。原训练计划中的工程诊断授权不扩展为新的科学更新、模型采用或后继实验授权。",
        "",
    ])


def _focus_mean(rows: Sequence[Mapping[str, Any]], scenario: str) -> tuple[float, dict[str, float]]:
    means = {}
    for digit in FOCUS_CLASSES:
        group = [row for row in rows if row["scenario"] == scenario and row["role"] == "dev" and int(row["label"]) == digit]
        _require(len(group) == DEV_PER_CLASS, f"primary Q_fg group is incomplete for {scenario}/class {digit}")
        means[str(digit)] = math.fsum(float(row["q_fg"]) for row in group) / DEV_PER_CLASS
    return math.fsum(means.values()) / len(FOCUS_CLASSES), means


def _decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    q0, q0_classes = _focus_mean(rows, "A4700")
    by_seed = {}
    all_pass = True
    for seed in TRAIN_SEEDS:
        a_scenario = f"seed_{seed}_A_step{SOURCE_STEP + UPDATES_PER_ARM}"
        b_scenario = f"seed_{seed}_B_step{SOURCE_STEP + UPDATES_PER_ARM}"
        qa, qa_classes = _focus_mean(rows, a_scenario)
        qb, qb_classes = _focus_mean(rows, b_scenario)
        difference = qb - qa
        tolerance = DECISION_ATOL + DECISION_RTOL * max(abs(qa), abs(qb))
        passes = difference < -tolerance and qb < q0
        all_pass = all_pass and passes
        by_seed[str(seed)] = {
            "A": qa,
            "B": qb,
            "Q0": q0,
            "D_B_minus_A": difference,
            "tolerance": tolerance,
            "B_below_shared_start": qb < q0,
            "passes_both_conditions": passes,
            "class_means": {"A": qa_classes, "B": qb_classes, "Q0": q0_classes},
        }
    return {
        "classification": "FOREGROUND_GAIN_REPEATED_ALL_THREE_SEEDS" if all_pass else "FOREGROUND_GAIN_NOT_REPEATED_ALL_THREE_SEEDS",
        "all_three_seeds_pass": all_pass,
        "seed_decisions": by_seed,
        "primary_population": "canonical DEV, classes 2 and 4, exactly 100 rows per class, equal-class weighting",
        "rule": "Each seed requires D=Q_B-Q_A < -(2e-7+2e-5*max(abs(Q_A),abs(Q_B))) and Q_B<Q0.",
        "shared_start_mean_q_fg": q0,
        "historical_A5400_B5400_used_for_decision": False,
        "dev_used_for_updates_or_tuning": False,
    }


def _class_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scenarios = sorted({str(row["scenario"]) for row in rows})
    result = []
    for scenario in scenarios:
        for digit in range(10):
            group = [row for row in rows if row["scenario"] == scenario and int(row["label"]) == digit]
            _require(len(group) == DEV_PER_CLASS, f"full DEV summary lacks class {digit}: {scenario}")
            record: dict[str, Any] = {"scenario": scenario, "label": digit, "rows": len(group)}
            for field in ("q_fg", "q_bg", "q_raw200", "raw_cmos_integral", "clipped_fraction"):
                record[f"mean_{field}"] = math.fsum(float(row[field]) for row in group) / len(group)
            for name in EVALUATOR_NAMES:
                record[f"{name.lower()}_correct"] = sum(int(row[f"{name.lower()}_prediction"]) == digit for row in group)
            result.append(record)
    return result


def _measure_all(target: Path, inputs: dict[str, Any], evaluators: Mapping[str, nn.Module], endpoints: Mapping[tuple[int, str], Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    source_consumer, _source_optimizer, _source_parameters = _new_endpoint(inputs["sources"], torch.device("cuda:0"))
    all_rows.extend(_save_dev_readout(target, inputs, evaluators, source_consumer, "A4700"))
    del source_consumer, _source_optimizer, _source_parameters
    for arm in ("A", "B"):
        historical_payload = inputs["sources"]["historical"]["checkpoints"][arm]
        old_consumer, _old_optimizer, _old_parameters = _load_endpoint(inputs["sources"], historical_payload, torch.device("cuda:0"))
        all_rows.extend(_save_dev_readout(target, inputs, evaluators, old_consumer, f"historical_{arm}5400"))
        del old_consumer, _old_optimizer, _old_parameters
    for seed, arm in _trajectory_order():
        scenario = f"seed_{seed}_{arm}_step{SOURCE_STEP + UPDATES_PER_ARM}"
        consumer, optimizer, parameters = _load_endpoint(inputs["sources"], endpoints[(seed, arm)], torch.device("cuda:0"))
        all_rows.extend(_save_dev_readout(target, inputs, evaluators, consumer, scenario))
        del consumer, optimizer, parameters
        torch.cuda.empty_cache()
    for scenario in ("A4700", *(f"seed_{seed}_{arm}_step{SOURCE_STEP + UPDATES_PER_ARM}" for seed, arm in _trajectory_order())):
        consumer = _measurement_endpoint(target, inputs, scenario)
        all_rows.extend(_save_sample_readout(target, inputs, evaluators, consumer, scenario))
        del consumer
        torch.cuda.empty_cache()
    all_sample_rows = []
    for scenario in ("A4700", *(f"seed_{seed}_{arm}_step{SOURCE_STEP + UPDATES_PER_ARM}" for seed, arm in _trajectory_order())):
        all_sample_rows.extend(_read_saved_sample_rows(target, scenario))
        _recheck_sample_scenario(target, inputs, scenario)
    sample_summary = _summarize_samples(target, inputs, all_sample_rows)
    rechecked_rows = []
    for scenario in ("A4700", "historical_A5400", "historical_B5400", *(f"seed_{seed}_{arm}_step{SOURCE_STEP + UPDATES_PER_ARM}" for seed, arm in _trajectory_order())):
        rechecked_rows.extend(_recheck_dev_scenario(target, inputs, scenario))
    _write_csv(target / "paired_rows.csv", rechecked_rows, DEV_FIELDS, replace=True)
    summaries = _class_summary(rechecked_rows)
    _write_csv(target / "dev_class_summary.csv", summaries, tuple(summaries[0].keys()), replace=True)
    decision = _decision(rechecked_rows)
    return rechecked_rows, {"decision": decision, "sample_summary": sample_summary, "paired_rows": len(rechecked_rows), "sample_rows": len(all_sample_rows), "class_summary_rows": len(summaries)}


def _measurement_endpoint(target: Path, inputs: Mapping[str, Any], scenario: str) -> Any:
    if scenario == "A4700":
        consumer, _optimizer, _parameters = _new_endpoint(inputs["sources"], torch.device("cuda:0"))
        return consumer
    prefix, seed_text, arm, _step = scenario.split("_")
    _require(prefix == "seed" and arm in {"A", "B"}, f"invalid conditional measurement scenario: {scenario}")
    seed = int(seed_text)
    payload = torch.load(_checkpoint_path(target, seed, arm, UPDATES_PER_ARM), map_location="cpu", weights_only=True)
    consumer, _optimizer, _parameters = _load_endpoint(inputs["sources"], payload, torch.device("cuda:0"))
    return consumer


def _read_saved_sample_rows(target: Path, scenario: str) -> list[dict[str, Any]]:
    rows = []
    for seed in SAMPLE_SEEDS:
        rows.extend(_read_csv(target / "readouts" / "conditional_samples" / scenario / f"seed_{seed}" / "paired_rows.csv", SAMPLE_FIELDS))
    return rows


def _load_preflight_caches(target: Path, inputs: dict[str, Any]) -> None:
    cache = torch.load(target / "ols_fit_coefficients.pt", map_location="cpu", weights_only=True)
    prior = torch.load(target / "fitted_joint_prior.pt", map_location="cpu", weights_only=True)
    bank = torch.load(target / "conditional_sample_bank.pt", map_location="cpu", weights_only=True)
    _require(cache.get("task_id") == TASK_ID and torch.equal(cache.get("source_row_indices"), inputs["fit_rows"]), "saved OLS cache task or fit population differs")
    _require(tuple(cache["coefficients"].shape) == (FIT_ROWS, 100) and cache["coefficients"].is_complex(), "saved OLS cache is incomplete")
    target_shapes = _cached_fit_target_shapes(cache)
    _require(tuple(prior["means"].shape) == (10, 200) and tuple(prior["joint_cholesky"].shape) == (10, 200, 200), "saved conditional prior is incomplete")
    _require(bank.get("task_id") == TASK_ID and tuple(bank["z"].shape) == (len(SAMPLE_SEEDS), 10, SAMPLES_PER_CLASS, 100), "saved conditional bank identity or geometry differs")
    inputs["ols_coefficients"] = cache["coefficients"]
    inputs["fit_target_shape_vectors"] = target_shapes
    inputs["priors"] = prior
    inputs["sample_bank"] = bank


def _report_markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    lines = [
        "# 云训练1：固定前景目标三种子 DEV 确认",
        "",
        "## 结果与状态",
        "",
        f"状态：`{report['status']}`；判定：`{decision['classification']}`。六条轨迹各完成2,000次Generator Adam更新，共`{report['scientific_updates']}`次科学更新；最终测量已回读共同A4700及六个终点的完整1,000行DEV原始读出。",
        "",
        "## 科学认识",
        "",
        "唯一主问题是固定lambda前景项在扩大拟合人口后是否重复改善类别2、4 DEV前景误差。每个种子的A/B来自同一A4700模型和Adam状态，并使用相同批次序列；仅当三个种子均满足预设D与Q0条件时，本范围内重复收益才得到支持。",
        "",
    ]
    for seed in TRAIN_SEEDS:
        record = decision["seed_decisions"][str(seed)]
        lines.append(f"种子`{seed}`：Q_A=`{record['A']:.10g}`，Q_B=`{record['B']:.10g}`，Q0=`{record['Q0']:.10g}`，D=`{record['D_B_minus_A']:.10g}`，τ=`{record['tolerance']:.10g}`；两项条件同时满足：`{record['passes_both_conditions']}`。")
    lines.extend([
        "",
        "旧A5400/B5400在同一DEV上的读数仅作描述，不进入主判定。条件采样结果也仅作数值描述，不构成可读性、新颖性或模型采用证据。",
        "",
        "## 目标缺口",
        "",
        "本结果只涉及固定A4700起点、固定拟合/确认人口、固定lambda与每臂2,000步的Generator-only范围。DEV曾用于评价器开发，因此不代表全项目盲测。正式识读、人眼可读性、模型采用、Encoder联合更新和物理部署仍未被本轮验证；canonical TEST保留给后续验收。",
        "",
        "## 工程完成与验证",
        "",
        f"云端独立READY预检记录在`{report['output_dir']}/preflight.json`，涵盖本机重复性、独立NumPy/解析/相位中心差分、原生产与合并梯度等价、Encoder语义梯度反例、六条轨迹各三步真实更新、零lambda负控、恢复及冻结隔离。训练中每100步保存包含批次序列、Generator/Encoder状态、Adam、RNG和历史的恢复点。",
        "",
        f"最终DEV原始块、逐行CSV与判定已独立回读；附属固定OLS采样测量为`{report['sampling']['rows_per_scenario']}`行/端点，未筛样。历史Windows/云端冻结重放状态保持`REPLAY_MISMATCH_ENGINEERING_REVIEW_REQUIRED`，原`rtol=2e-6, atol=5e-8`未改；本轮云端准入使用独立本机重复性，不声称严格复现Windows结果。",
        "",
        "## 决策影响与下一步",
        "",
        "训练建议：`NONE`；后续动作：`NONE`。本轮完成后停止；结果仅提供是否保留固定前景目标进入后续关键识读设计的依据，不自动采用模型或启动新训练。",
        "",
    ])
    return "\n".join(lines)


def _finalize(target: Path, inputs: Mapping[str, Any], control: ResearchRunControl, preflight: Mapping[str, Any], measurements: Mapping[str, Any], started: float) -> dict[str, Any]:
    rows = []
    for scenario in ("A4700", "historical_A5400", "historical_B5400", *(f"seed_{seed}_{arm}_step{SOURCE_STEP + UPDATES_PER_ARM}" for seed, arm in _trajectory_order())):
        rows.extend(_recheck_dev_scenario(target, inputs, scenario))
    independent_decision = _decision(rows)
    _require(independent_decision == measurements["decision"], "final scientific decision changed when recomputed from saved DEV chunks")
    _require(len(rows) == 9 * DEV_ROWS, "full DEV readout set is incomplete")
    sample_scenarios = ("A4700", *(f"seed_{seed}_{arm}_step{SOURCE_STEP + UPDATES_PER_ARM}" for seed, arm in _trajectory_order()))
    sample_rows = sum(_recheck_sample_scenario(target, inputs, scenario) for scenario in sample_scenarios)
    _require(sample_rows == 7 * len(SAMPLE_SEEDS) * 10 * SAMPLES_PER_CLASS, "conditional sample saved evidence is incomplete")
    control.complete()
    status = "VALID_COMPLETED" if independent_decision["all_three_seeds_pass"] else "VALID_FAILED"
    report = {
        "schema": "lightgen_cloud_foreground_dev_multiseed_report_v1",
        "task_id": TASK_ID,
        "status": status,
        "classification": independent_decision["classification"],
        "decision": independent_decision,
        "source_checkpoint_step": SOURCE_STEP,
        "final_generator_adam_step": SOURCE_STEP + UPDATES_PER_ARM,
        "training_seeds": list(TRAIN_SEEDS),
        "updates_per_arm_per_seed": UPDATES_PER_ARM,
        "scientific_updates": TOTAL_UPDATES,
        "lambda": FIXED_LAMBDA,
        "fit_rows": FIT_ROWS,
        "fit_rows_per_class": FIT_PER_CLASS,
        "dev_rows": DEV_ROWS,
        "dev_rows_per_class": DEV_PER_CLASS,
        "training_arm_objectives": {
            "A": "original production objective",
            "B": f"original production objective + {FIXED_LAMBDA:.17g}*Q_fg",
            "generator_scope": "Generator phase only; Encoder autograd retained for normalized E_S/E_I/E_C semantic gradients; Encoder and all four evaluators frozen from optimizer updates",
        },
        "paired_training_batches": True,
        "full_dev_rows_rechecked": len(rows),
        "historical_endpoints_descriptive_only": ["historical_A5400", "historical_B5400"],
        "sampling": measurements["sample_summary"],
        "sampling_readout_rows_rechecked": sample_rows,
        "preflight": preflight,
        "research_run_state": control.state,
        "historical_cross_platform_replay": {"status": "REPLAY_MISMATCH_ENGINEERING_REVIEW_REQUIRED", "rtol": REPLAY_RTOL, "atol": REPLAY_ATOL, "rewritten": False, "used_for_new_cloud_admission": False},
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "stop": True,
        "output_dir": target.as_posix(),
        "elapsed_seconds_observation_only": time.perf_counter() - started,
    }
    _write_csv(target / "paired_rows.csv", rows, DEV_FIELDS, replace=True)
    summaries = _class_summary(rows)
    _write_csv(target / "dev_class_summary.csv", summaries, tuple(summaries[0].keys()), replace=True)
    _write_json(target / "validation.json", {
        "task_id": TASK_ID,
        "status": "VALID",
        "full_dev_rows_rechecked": len(rows),
        "sample_rows_rechecked": sample_rows,
        "decision_recomputed_from_saved_chunks": independent_decision,
        "csv_matches_raw_chunks": True,
        "all_classes_and_roles_present": True,
        "scientific_updates": TOTAL_UPDATES,
    })
    _write_json(target / "report.json", report)
    (target / "closeout.md").write_text(_report_markdown(report), encoding="utf-8", newline="\n")
    _write_json(target / "completed.json", {"task_id": TASK_ID, "status": status, "classification": independent_decision["classification"], "scientific_updates": TOTAL_UPDATES, "training_recommendation": "NONE", "next_action": "NONE"})
    _write_json(target / "progress.json", {"task_id": TASK_ID, "status": "COMPLETED", "scientific_updates_completed": TOTAL_UPDATES, "scientific_updates_authorized": TOTAL_UPDATES}, replace=True)
    return report


def _runtime_record(target: Path) -> dict[str, Any]:
    device = torch.device("cuda:0")
    free, total = torch.cuda.mem_get_info(device)
    return {
        "task_id": TASK_ID,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "device_total_bytes": int(total),
        "device_free_bytes_observation": int(free),
        "numeric_mode": NUMERIC_MODE,
        "working_directory": Path.cwd().as_posix(),
        "output_dir": target.as_posix(),
    }


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    artifact_dir = _portable_resolve(root, args.artifact_dir)
    checkpoint = _portable_resolve(root, args.checkpoint)
    gradient_dir = _portable_resolve(root, args.gradient_audit_dir)
    reference_dir = _portable_resolve(root, args.reference_dir)
    target = _portable_resolve(root, args.output_dir)
    if args.engineering_recovery_worker:
        _require(args.engineering_recovery_stage is not None, "engineering recovery worker requires an explicit save1 or resume2 stage")
        return _engineering_recovery_worker(args)
    _require(args.engineering_recovery_stage is None and args.ols_cache is None and args.repeatability_envelope is None, "hidden engineering recovery inputs require the recovery worker mode")
    if args.engineering_check_only:
        _require(args.resume_checkpoint is None, "engineering-check-only cannot resume a scientific or prior diagnostic checkpoint")
        return _execute_engineering_check(args, root, artifact_dir, checkpoint, gradient_dir, reference_dir, target)
    _require(getattr(args, "engineering_attempt_dir", None) is None, "--engineering-attempt-dir is only valid with --engineering-check-only")
    _require(args.engineering_resume_evidence_dir is None, "--engineering-resume-evidence-dir is only valid with --engineering-check-only")
    resume_path = _portable_resolve(root, args.resume_checkpoint) if args.resume_checkpoint is not None else None
    _require(args.preflight_only != args.execute, "choose exactly one of --preflight-only or --execute")
    _require(not (args.preflight_only and resume_path is not None), "--preflight-only cannot be combined with --resume-checkpoint")
    _prepare_target(root, target, resume_path)
    if (target / "report.json").is_file():
        return _json(target / "report.json")
    _preserve_prior_failure(target)
    inputs = _load_all_inputs(root, artifact_dir, checkpoint, gradient_dir, reference_dir)
    inputs["microbatch_size"] = args.microbatch_size
    _write_json(target / "input_population.json", inputs["population"])
    expected_orders = {str(seed): inputs["batch_orders"][seed] for seed in TRAIN_SEEDS}
    _write_json(target / "batch_orders.json", expected_orders)
    _write_json(target / "fixed_parameters.json", {
        "task_id": TASK_ID,
        "classification": "diagnostic-only",
        "engineering_check_only": True,
        "scientific_updates": 0,
        "engineering_adam_update_limit": ENGINEERING_CUMULATIVE_UPDATE_LIMIT,
        "engineering_new_adam_update_limit": ENGINEERING_NEW_UPDATE_LIMIT,
        "engineering_historical_adam_updates": ENGINEERING_HISTORICAL_UPDATES,
        "microbatch_size_under_test": args.microbatch_size,
        "checkpoint_policy": {"interval": CHECKPOINT_INTERVAL, "keep_recent": CHECKPOINT_KEEP_RECENT, "terminal_per_trajectory": "retain"},
    })
    started = time.perf_counter()
    with capacity.strict_fp32_mode() as numeric_mode:
        _require(numeric_mode == STRICT_FP32_SETTINGS, "strict FP32 cloud runtime settings differ")
        if (target / "preflight.json").is_file():
            control, receipt, preflight = _validate_ready(target, inputs)
            _load_preflight_caches(target, inputs)
            cache_consumer, _cache_optimizer, _cache_parameters = _new_endpoint(inputs["sources"], torch.device("cuda:0"))
            cache_check = _verify_cached_ols(cache_consumer, inputs, {
                "coefficients": inputs["ols_coefficients"],
                "target_shape_vectors": inputs["fit_target_shape_vectors"],
                "cache_was_existing": True,
            })
            _require(cache_check.get("passed") is True and cache_check.get("rows_recomputed") == FIT_ROWS, "saved READY OLS cache failed its full source recheck")
            del cache_consumer, _cache_optimizer, _cache_parameters
            torch.cuda.empty_cache()
        else:
            preflight = _device_preflight(inputs, target, args)
            inputs["preflight_trials"] = preflight["checks"]["six_seed_arm_three_step_updates"]
            _require(preflight.get("status") == "READY", "cloud production preflight did not return READY")
            control, receipt = _make_receipt(inputs, preflight)
            _stage_ready(target, inputs, preflight, receipt, resume=False)
        if args.preflight_only:
            _write_json(target / "preflight_runtime.json", _runtime_record(target), replace=True)
            return {"task_id": TASK_ID, "status": "READY", "scientific_updates": 0, "output_dir": target.as_posix(), "checks": preflight["checks"], "next_action": "NONE"}
        control, receipt, preflight = _validate_ready(target, inputs)
        _write_json(target / "execution_runtime.json", _runtime_record(target), replace=True)
        evaluators = generation.diagnostic.load_evaluators(inputs["sources"]["source"]["generation_paths"], torch.device("cuda:0"))
        evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
        control.start_execution()
        control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt, resume=True)
        endpoints = _run_training(target, inputs, evaluators, resume_path)
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"formal frozen evaluator {name}")
        torch.cuda.synchronize(torch.device("cuda:0"))
        _write_json(target / "progress.json", {"task_id": TASK_ID, "status": "MEASURING", "scientific_updates_completed": TOTAL_UPDATES, "scientific_updates_authorized": TOTAL_UPDATES}, replace=True)
        _rows, measurements = _measure_all(target, inputs, evaluators, endpoints)
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"post-measurement frozen evaluator {name}")
        torch.cuda.synchronize(torch.device("cuda:0"))
        return _finalize(target, inputs, control, preflight, measurements, started)


def _execute_engineering_check(
    args: argparse.Namespace,
    root: Path,
    artifact_dir: Path,
    checkpoint: Path,
    gradient_dir: Path,
    reference_dir: Path,
    target: Path,
) -> dict[str, Any]:
    evidence_dir = _portable_resolve(root, args.evidence_dir) if args.evidence_dir is not None else None
    _require(evidence_dir is not None, "--engineering-check-only requires an explicit --evidence-dir")
    output_root = (root / DEFAULT_ENGINEERING_OUTPUT).resolve()
    _require(target.resolve() == output_root, "engineering diagnostics must use the dedicated task output path")
    if (output_root / "diagnostic_report.json").is_file():
        return _json(output_root / "diagnostic_report.json")
    if output_root.exists():
        _require(output_root.is_dir() and (output_root / "plan.md").is_file(), "existing engineering output is not owned by this diagnostic task")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    plan_text = (
        "# AutoDL training workflow engineering diagnostics\n\n"
        "Status: diagnostic-only. No formal scientific attempt, READY receipt, model decision, or scientific update is created.\n"
        "The diagnostic permits at most 42 new engineering Adam updates. The three historical updates remain counted under the approved cumulative limit of 45.\n"
    )
    if not (output_root / "plan.md").is_file():
        (output_root / "plan.md").write_text(plan_text, encoding="utf-8", newline="\n")
    resume_value = getattr(args, "engineering_resume_evidence_dir", None)
    resuming = resume_value is not None
    if resuming:
        _require(getattr(args, "engineering_attempt_dir", None) is None, "choose either a new engineering attempt directory or an explicit resume directory")
        target = _portable_resolve(root, resume_value)
        _require(target.is_dir() and target.resolve().parent == output_root, "--engineering-resume-evidence-dir must name an existing attempt inside this task output")
        _require((target / "plan.md").is_file() and (target / "plan.md").read_text(encoding="utf-8") == plan_text, "engineering resume attempt plan differs")
        _preserve_prior_failure(target)
    else:
        prior_runs = sorted(path for path in output_root.glob("engineering_attempt_[0-9][0-9]") if path.is_dir())
        run_index = max((int(path.name.rsplit("_", 1)[1]) for path in prior_runs), default=0) + 1
        expected_target = output_root / f"engineering_attempt_{run_index:02d}"
        attempt_value = getattr(args, "engineering_attempt_dir", None)
        _require(attempt_value is not None, "a fresh engineering diagnostic requires an explicit --engineering-attempt-dir")
        target = _portable_resolve(root, attempt_value)
        _require(target.resolve() == expected_target.resolve(), "--engineering-attempt-dir must name the next unused engineering attempt")
        target.mkdir(parents=True, exist_ok=False)
        (target / "plan.md").write_text(plan_text, encoding="utf-8", newline="\n")
    args.engineering_attempt_dir = target
    required = [
        evidence_dir / "report.json", evidence_dir / "validation.json", evidence_dir / "paired_rows.csv",
        artifact_dir / "input_population.json", checkpoint,
        gradient_dir / "report.json", gradient_dir / "input_population.json", gradient_dir / "fit_calibration.json",
        gradient_dir / "preflight.json", gradient_dir / "fit_gradients.pt",
        reference_dir / "report.json", reference_dir / "input_population.json",
        reference_dir / "arm_a_step5400.pt", reference_dir / "arm_b_step5400.pt",
    ]
    inventory, missing = _input_file_inventory(required)
    if not resuming:
        _write_json(target / "input_probe.json", {"evidence_dir": evidence_dir.as_posix(), "files": inventory, "missing_files": missing})
    _require(not missing, "engineering input probe found missing explicit files; review input_probe.json before CUDA work")
    evidence_report = _json(evidence_dir / "report.json")
    evidence_validation = _json(evidence_dir / "validation.json")
    _require(evidence_report.get("status") in {"VALID_COMPLETED", "VALID_FAILED"} and evidence_validation.get("status") == "VALID", "named prior evidence is incomplete or invalid")
    inputs = _load_all_inputs(root, artifact_dir, checkpoint, gradient_dir, reference_dir)
    source = inputs["sources"]["source"]
    generation_inputs = source["generation_inputs"]
    ancestor_dir = Path(inputs["sources"]["ancestor"]["directory"])
    transitive_candidates: list[Path | str] = [
        artifact_dir / "report.json", artifact_dir / "input_population.json", artifact_dir / "readouts_start.pt",
        generation_inputs["source_readout_path"], ancestor_dir / "report.json", ancestor_dir / "input_population.json",
        Path(inputs["sources"]["historical"]["directory"]) / "report.json",
        Path(inputs["sources"]["historical"]["directory"]) / "input_population.json",
    ]
    transitive_candidates.extend(generation_inputs["paths"].values())
    transitive_candidates.extend(source["generation_paths"].values())
    transitive_paths = []
    for candidate in transitive_candidates:
        path = _portable_resolve(root, candidate)
        if path not in transitive_paths:
            transitive_paths.append(path)
    transitive_inventory, transitive_missing = _input_file_inventory(transitive_paths)
    combined_inventory = {row["path"]: row for row in [*inventory, *transitive_inventory]}
    input_probe = {
        "evidence_dir": evidence_dir.as_posix(),
        "direct_files": inventory,
        "transitive_files": transitive_inventory,
        "missing_files": transitive_missing,
        "total_input_bytes": sum(row["size_bytes"] or 0 for row in combined_inventory.values()),
    }
    if resuming:
        saved_probe = _json(target / "input_probe.json")
        _require(saved_probe == input_probe, "engineering resume explicit input inventory differs from the completed attempt")
    else:
        _write_json(target / "input_probe.json", input_probe, replace=True)
    _require(not transitive_missing, "transitive engineering input probe found missing files; CUDA was not started")
    inputs["microbatch_size"] = args.microbatch_size
    expected_orders = {str(seed): inputs["batch_orders"][seed] for seed in TRAIN_SEEDS}
    expected_config = {
        "task_id": TASK_ID,
        "source_checkpoint": inputs["population"]["source_checkpoint"],
        "source_checkpoint_step": SOURCE_STEP,
        "source_adam_step": SOURCE_STEP,
        "requested_microbatch_size": int(args.microbatch_size),
        "trajectory_microbatch_sizes": [1, 2, 5, 10, 1, 1],
        "logical_batch_size": BATCH_SIZE,
        "propagation_grid": 2_048,
        "optical_dtype": "float32",
        "qfg_dtype": "float64",
        "fixed_lambda": FIXED_LAMBDA,
        "generator_only_adam": True,
    }
    if resuming:
        _require(_json(target / "input_population.json") == inputs["population"], "engineering resume population differs before CUDA work")
        _require(_json(target / "batch_orders.json") == expected_orders, "engineering resume batch schedule differs before CUDA work")
        _require(_json(target / "engineering_run_config.json") == expected_config, "engineering resume configuration differs before CUDA work")
    else:
        _write_json(target / "input_population.json", inputs["population"])
        _write_json(target / "batch_orders.json", expected_orders)
        _write_json(target / "engineering_run_config.json", expected_config)
    input_stage = {
        "status": "COMPLETE",
        "evidence_dir": evidence_dir.as_posix(),
        "source_checkpoint": inputs["population"]["source_checkpoint"],
        "fit_rows": len(inputs["fit_rows"]),
        "dev_rows": len(inputs["dev_rows"]),
        "batch_orders": expected_orders,
        "missing_direct_files": 0,
        "missing_transitive_files": 0,
    }
    if _engineering_stage_path(target, "input_validation").is_file():
        _require(_read_engineering_stage(target, "input_validation") == {"task_id": TASK_ID, "phase": "input_validation", "evidence_identity": {
            "input_population": "input_population.json",
            "batch_orders": "batch_orders.json",
            "run_config": "engineering_run_config.json",
        }, **input_stage}, "saved engineering input stage differs")
    else:
        _write_engineering_stage(target, "input_validation", input_stage)
    budget_file = _engineering_budget_file()
    _initialize_engineering_budget(budget_file)
    _extend_engineering_budget(budget_file)
    inputs["engineering_budget_file"] = budget_file
    inputs["engineering_progress_file"] = output_root / "progress.json"
    inputs["engineering_attempt_dir"] = target
    budget = _read_engineering_budget(budget_file)
    _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", "input_validation", budget)
    started = time.perf_counter()
    with capacity.strict_fp32_mode() as numeric_mode:
        _require(numeric_mode == STRICT_FP32_SETTINGS, "strict FP32 cloud runtime settings differ")
        if resuming:
            _validate_engineering_resume_attempt(target, output_root, evidence_dir, inputs, args, budget_file)
        diagnostic_preflight = _device_preflight(inputs, target, args)
    _require(diagnostic_preflight.get("status") == "READY", "production checks did not complete; diagnostic remains an engineering failure")
    production_stage = {
        "status": "COMPLETE",
        "source_checkpoint": inputs["population"]["source_checkpoint"],
        "microbatch_size": int(args.microbatch_size),
        "engineering_updates_consumed": diagnostic_preflight["checks"]["total_engineering_adam_updates"]["count"],
        "cumulative_updates_reserved": diagnostic_preflight["checks"]["total_engineering_adam_updates"]["cumulative_reserved"],
        "checks_passed": True,
    }
    if not _engineering_stage_path(target, "production_preflight").is_file():
        _write_json(target / "preflight.json", diagnostic_preflight, replace=(target / "preflight.json").exists())
        _write_json(target / "preflight_trials.json", diagnostic_preflight["checks"]["six_seed_arm_three_step_updates"], replace=(target / "preflight_trials.json").exists())
        _write_microbatch_performance_table(target / "microbatch_performance.csv", diagnostic_preflight["checks"]["merged_production_gradient_and_speed"])
        _write_engineering_stage(target, "production_preflight", production_stage)
    else:
        _require(_read_engineering_stage(target, "production_preflight") == {"task_id": TASK_ID, "phase": "production_preflight", "evidence_identity": {
            "input_population": "input_population.json",
            "batch_orders": "batch_orders.json",
            "run_config": "engineering_run_config.json",
        }, **production_stage}, "saved production preflight stage differs")
    if _engineering_stage_path(target, "saved_evidence_pipeline").is_file():
        saved_evidence_pipeline = _json(target / "saved_evidence_pipeline.json")
        _require(saved_evidence_pipeline.get("passed") is True and saved_evidence_pipeline.get("dev_rows_saved_and_rechecked") == 20 and saved_evidence_pipeline.get("conditional_sample_rows_saved_and_rechecked") == 40, "saved evidence pipeline stage is incomplete")
        _require(_read_engineering_stage(target, "saved_evidence_pipeline").get("passed") is True, "saved evidence pipeline stage failed readback")
        _require((target / "readouts" / "A4700" / "paired_rows.csv").is_file(), "saved evidence pipeline lost its DEV row table")
        _require((target / "readouts" / "conditional_samples" / "A4700" / f"seed_{SAMPLE_SEEDS[0]}" / "paired_rows.csv").is_file(), "saved evidence pipeline lost the first conditional sample table")
        _require((target / "readouts" / "conditional_samples" / "A4700" / f"seed_{SAMPLE_SEEDS[1]}" / "paired_rows.csv").is_file(), "saved evidence pipeline lost the second conditional sample table")
    else:
        _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", "saved_evidence_pipeline")
        saved_evidence_pipeline = _saved_evidence_pipeline_diagnostic(target, evidence_dir, inputs, evidence_report)
        _write_json(target / "saved_evidence_pipeline.json", saved_evidence_pipeline)
        _write_engineering_stage(target, "saved_evidence_pipeline", {"status": "COMPLETE", **saved_evidence_pipeline})
    diagnostic_preflight["checks"]["saved_evidence_pipeline"] = saved_evidence_pipeline
    _write_json(target / "preflight.json", diagnostic_preflight, replace=True)
    if _engineering_stage_path(target, "review_package").is_file():
        review_package = _json(target / "review_package" / "package_report.json")
        _require(review_package.get("status") == "PACKAGED" and _read_engineering_stage(target, "review_package").get("passed") is True, "saved review package stage is incomplete")
    else:
        _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", "review_package")
        package_run = subprocess.run(
            [
                sys.executable, "-m", "scripts.maintenance.autodl_review", "package",
                "--evidence-dir", str(evidence_dir), "--output-dir", str(target / "review_package"), "--profile", "review",
            ],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
        _require(package_run.returncode == 0, f"read-only review package generation failed: {package_run.stderr[-2000:]}")
        review_package = json.loads(package_run.stdout)
        _write_engineering_stage(target, "review_package", {"status": "COMPLETE", "passed": True, "output_dir": "review_package", "file_count": review_package.get("file_count")})
    _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", "report_write")
    report = {
        "task_id": TASK_ID,
        "status": "DIAGNOSTIC_COMPLETE",
        "classification": "diagnostic-only",
        "formal_attempt_opened": False,
        "ready_receipt_created_or_consumed": False,
        "science_updates": 0,
        "engineering_adam_updates": diagnostic_preflight["checks"]["total_engineering_adam_updates"]["count"],
        "engineering_adam_update_limit": ENGINEERING_CUMULATIVE_UPDATE_LIMIT,
        "engineering_new_adam_update_limit": ENGINEERING_NEW_UPDATE_LIMIT,
        "engineering_historical_adam_updates": ENGINEERING_HISTORICAL_UPDATES,
        "evidence_dir": evidence_dir.as_posix(),
        "output_dir": target.as_posix(),
        "preflight": {**diagnostic_preflight, "status": "DIAGNOSTIC_COMPLETE"},
        "runtime": _runtime_record(target),
        "review_package": review_package,
        "delivery_manifests": {"review": "review_delivery_manifest.json", "diagnostic_audit": "diagnostic_audit_manifest.json", "historical_audit": "historical_audit_manifest.json"},
        "saved_evidence_pipeline": saved_evidence_pipeline,
        "elapsed_seconds_observation_only": time.perf_counter() - started,
        "stop": True,
    }
    _write_json(target / "diagnostic_report.json", report)
    _write_json(target / "diagnostic_validation.json", {
        "status": "VALID_ENGINEERING_DIAGNOSTIC",
        "formal_attempt_opened": False,
        "scientific_updates": 0,
        "engineering_adam_updates": report["engineering_adam_updates"],
        "limit": ENGINEERING_CUMULATIVE_UPDATE_LIMIT,
        "new_update_limit": ENGINEERING_NEW_UPDATE_LIMIT,
        "historical_adam_updates": ENGINEERING_HISTORICAL_UPDATES,
        "all_checks_executed": all(item.get("executed") is True and item.get("passed") is True for item in diagnostic_preflight["checks"].values()),
        "saved_evidence_pipeline": saved_evidence_pipeline,
    })
    _write_json(output_root / "diagnostic_report.json", report)
    _write_json(output_root / "diagnostic_validation.json", _json(target / "diagnostic_validation.json"))
    (output_root / "closeout.md").write_text(_engineering_closeout_markdown(report), encoding="utf-8", newline="\n")
    delivery = _write_engineering_delivery_manifests(output_root, target, evidence_dir, str(inputs["population"]["source_checkpoint"]))
    _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_RUNNING", "delivery_manifests")
    report["delivery_manifest_preparation"] = delivery
    _write_json(target / "diagnostic_report.json", report, replace=True)
    _write_json(output_root / "diagnostic_report.json", report, replace=True)
    (target / "closeout.md").write_text(_engineering_closeout_markdown(report), encoding="utf-8", newline="\n")
    (output_root / "closeout.md").write_text(_engineering_closeout_markdown(report), encoding="utf-8", newline="\n")
    _write_engineering_stage(target, "diagnostic_report", {"status": "COMPLETE", "report": "diagnostic_report.json", "validation": "diagnostic_validation.json", "delivery_manifests": delivery})
    budget_after = _read_engineering_budget(budget_file)
    final_progress = _write_engineering_progress(inputs, "ENGINEERING_DIAGNOSTIC_COMPLETE", "complete", budget_after)
    final_progress["review_manifest"] = "review_delivery_manifest.json"
    _write_json(output_root / "progress.json", final_progress, replace=True)
    _write_json(target / "progress.json", final_progress, replace=True)
    return report


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gradient-audit-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--engineering-check-only", action="store_true")
    mode.add_argument("--engineering-recovery-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--engineering-recovery-stage", choices=("save1", "resume2"), help=argparse.SUPPRESS)
    parser.add_argument("--engineering-signal-kind", choices=("SIGINT", "SIGTERM"), help=argparse.SUPPRESS)
    parser.add_argument("--resume-checkpoint", type=Path, help="explicit in-task recovery checkpoint; never resolved automatically")
    parser.add_argument("--evidence-dir", type=Path, help="explicit prior run evidence directory for diagnostic-only checks")
    parser.add_argument("--engineering-attempt-dir", type=Path, help="explicit next-unused diagnostic attempt directory")
    parser.add_argument("--engineering-resume-evidence-dir", type=Path, help="explicit partial or completed diagnostic attempt whose validated stages may be consumed")
    parser.add_argument("--microbatch-size", choices=(1, 2, 5, 10), type=int, default=1, help="rows per optical microbatch; logical training batch remains ten")
    parser.add_argument("--ols-cache", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--repeatability-envelope", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def execute(args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or _arguments()
    try:
        return _execute(args)
    except TrainingInterrupted as error:
        return {"task_id": TASK_ID, "status": "INTERRUPTED", "failure": str(error), "next_action": "resume only by naming the durable checkpoint explicitly"}
    except Exception as error:
        try:
            if args.engineering_check_only or args.engineering_recovery_worker:
                _record_engineering_failure(_root(), args, error)
            target = _portable_resolve(_root(), args.output_dir)
            if args.engineering_check_only and getattr(args, "engineering_resume_evidence_dir", None) is not None:
                target = _portable_resolve(_root(), args.engineering_resume_evidence_dir)
            elif args.engineering_check_only and target.is_dir():
                attempts = sorted(path for path in target.glob("engineering_attempt_[0-9][0-9]") if path.is_dir())
                if attempts:
                    target = attempts[-1]
            if not args.engineering_check_only and target.is_dir() and not (target / "report.json").is_file() and not (target / "invalid_harness.json").is_file():
                checkpoints = sorted(path.relative_to(target).as_posix() for path in (target / "checkpoints").glob("seed_*_arm_*_step*.pt")) if (target / "checkpoints").is_dir() else []
                _write_json(target / "invalid_harness.json", {
                    "task_id": TASK_ID,
                    "status": "INVALID_HARNESS",
                    "failure_type": type(error).__name__,
                    "failure": str(error),
                    "formal_checkpoint_inventory": checkpoints,
                    "scientific_conclusion": None,
                    "next_action": "repair the affected engineering path and recheck under the same task identity; formal training resume requires an explicit checkpoint",
                })
        except Exception:
            pass
        raise


def main() -> None:
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))


def _make_identity(inputs: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-attempt-01",
        trajectory_id="A4700_generator_only_fixed_foreground_three_seed_paired_2000_updates",
        semantic_authority_id="fixed_A4700_BOGT_references_and_audited_lambda",
        data_role="2,000 canonical TRAIN rows for updates; 1,000 canonical DEV rows for held-out Generator confirmation; TEST reserved",
        config_identity={
            "fit_rows": FIT_ROWS,
            "dev_rows": DEV_ROWS,
            "training_seeds": list(TRAIN_SEEDS),
            "updates_per_arm_per_seed": UPDATES_PER_ARM,
            "scientific_updates": TOTAL_UPDATES,
            "lambda": FIXED_LAMBDA,
            "microbatch_size": int(inputs.get("microbatch_size", 1)),
            "strict_fp32_optics": True,
            "qfg_statistics": "float64 raw 200x200 intensity with whole-image mass normalization and fixed target>0.1 mask",
        },
        model_identity={
            "source_checkpoint": inputs["population"]["source_checkpoint"],
            "source_step": SOURCE_STEP,
            "architecture": "two_original_100x100_then_two_400x400_phase_planes_flattened",
            "generator_parameters": 340_000,
            "propagation_grid": 2_048,
        },
        optimizer_identity={"class": "Adam", "parameter_names": ["generator.phase"], "source_step": SOURCE_STEP, "lr": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "history_reset": False},
        sampler_identity={"fit_source_rows": inputs["fit_rows"].tolist(), "dev_source_rows": inputs["dev_rows"].tolist(), "batch_sequence_seeds": list(TRAIN_SEEDS)},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )


def _make_receipt(inputs: Mapping[str, Any], preflight: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt]:
    checks = preflight.get("checks", {})
    _require(bool(checks) and all(item.get("executed") is True and item.get("passed") is True for item in checks.values()), "READY requires every executed preflight check to pass")
    control = ResearchRunControl(_make_identity(inputs), authorization_id=f"{TASK_ID}:authorization:01")
    receipt = control.make_ready_receipt(
        correctness={
            "passed": True,
            "status": "PASS",
            "source_model_and_adam": checks["source_model_and_optimizer"]["passed"],
            "cloud_local_replay": checks["cloud_local_replay"]["passed"],
            "cached_ols_source_recheck": checks["cached_ols_source_recheck"]["passed"],
            "qfg_independent_value_gradient": checks["qfg_independent_value_gradient"]["passed"],
            "merged_production_gradient": checks["merged_production_gradient_and_speed"]["passed"],
            "six_seed_arm_three_step_updates": checks["six_seed_arm_three_step_updates"]["passed"],
            "zero_lambda_negative_control": checks["six_seed_arm_three_step_updates"]["all_zero_lambda_controls_matched_first_A_within_float_roundoff"],
            "explicit_checkpoint_recovery": checks["explicit_checkpoint_recovery"]["passed"],
        },
        resource={"passed": checks["cuda_and_disk"]["passed"], "status": "PASS", "observation": checks["cuda_and_disk"]},
        implementation={
            "passed": True,
            "status": "PASS",
            "selected_engine": f"cuda_2048_microbatch_{int(inputs.get('microbatch_size', 1))}_fp32_generator_only_with_cached_ols_and_merged_loss_gradient",
            "fixed_lambda": FIXED_LAMBDA,
            "source_adam_inherited": True,
            "encoder_autograd_for_semantics": checks["merged_production_gradient_and_speed"]["Encoder_autograd_enabled"],
            "explicit_checkpoint_recovery": True,
        },
        selected_engine=f"cuda_2048_microbatch_{int(inputs.get('microbatch_size', 1))}_fp32_generator_only_with_cached_ols_and_merged_loss_gradient",
        resource_envelope={"cuda_and_disk": checks["cuda_and_disk"]},
    )
    return control, receipt


def _stage_ready(target: Path, inputs: Mapping[str, Any], preflight: Mapping[str, Any], receipt: PreflightReceipt, *, resume: bool) -> None:
    expected = {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": dict(preflight)}
    if resume:
        _require(_json(target / "input_population.json") == inputs["population"], "saved READY population differs from reconstructed inputs")
        saved = _json(target / "preflight.json")
        _require(saved.get("task_id") == TASK_ID and saved.get("status") == "READY", "saved preflight is not READY")
        return
    _write_json(target / "input_population.json", inputs["population"])
    _write_json(target / "fixed_parameters.json", {
        "task_id": TASK_ID,
        "source_step": SOURCE_STEP,
        "updates_per_arm_per_seed": UPDATES_PER_ARM,
        "training_seeds": list(TRAIN_SEEDS),
        "scientific_updates": TOTAL_UPDATES,
        "microbatch_size": int(inputs.get("microbatch_size", 1)),
        "checkpoint_policy": {"interval": CHECKPOINT_INTERVAL, "keep_recent": CHECKPOINT_KEEP_RECENT, "terminal_per_trajectory": "retain"},
        "lambda": FIXED_LAMBDA,
        "optimizer": {"lr": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "reset": False},
        "measurements": "common A4700, historical A5400/B5400 descriptive readouts, and six new final endpoints only",
        "decision": "for each of three seeds D=Q_B-Q_A < -(2e-7+2e-5*max(abs(Q_A),abs(Q_B))) and Q_B<Q_0",
    })
    _write_json(target / "batch_orders.json", {str(seed): inputs["batch_orders"][seed] for seed in TRAIN_SEEDS})
    _write_json(target / "preflight.json", expected)
    _write_json(target / "preflight_trials.json", preflight["checks"]["six_seed_arm_three_step_updates"])


def _validate_ready(target: Path, inputs: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt, dict[str, Any]]:
    payload = _json(target / "preflight.json")
    _require(payload.get("task_id") == TASK_ID and payload.get("status") == "READY", "saved cloud preflight is not READY")
    _require(_json(target / "input_population.json") == inputs["population"], "population differs from the one consumed by READY")
    saved_orders = _json(target / "batch_orders.json")
    _require(saved_orders == {str(seed): inputs["batch_orders"][seed] for seed in TRAIN_SEEDS}, "saved seed-specific batch orders differ")
    preflight = payload.get("preflight", {})
    checks = preflight.get("checks", {})
    _require(preflight.get("status") == "READY" and checks and all(item.get("executed") is True and item.get("passed") is True for item in checks.values()), "READY consumer found a missing or failed production check")
    registered_microbatch = checks.get("merged_production_gradient_and_speed", {}).get("selected_microbatch_size", 1)
    _require(int(inputs.get("microbatch_size", 1)) == registered_microbatch, "requested microbatch differs from the size consumed by READY")
    control = ResearchRunControl(_make_identity(inputs), authorization_id=f"{TASK_ID}:authorization:01")
    receipt = PreflightReceipt.from_mapping(payload["receipt"])
    _require(receipt.source_semantics_id == TASK_ID and receipt.authorization_id == f"{TASK_ID}:authorization:01" and receipt.status == "READY", "saved receipt identity or status differs")
    _require(all(group.get("passed") is True for group in receipt.checks.values()), "saved receipt includes a failed or missing check group")
    control.mark_preflight_ready(receipt)
    return control, receipt, preflight


def _prepare_target(root: Path, target: Path, resume_checkpoint: Path | None) -> None:
    expected = (root / DEFAULT_OUTPUT).resolve()
    outputs = (root / "outputs").resolve()
    _require(target.resolve() == expected, "output path must be the dedicated attempt_01 path in the plan")
    _require(outputs in target.resolve().parents and not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in target.resolve().relative_to(outputs).parts), "protected or restored output namespace is forbidden")
    if not target.exists():
        _require(resume_checkpoint is None, "explicit resume requires an existing task output directory")
        target.mkdir(parents=True, exist_ok=False)
        (target / "plan.md").write_text(_plan_text(), encoding="utf-8", newline="\n")
        return
    _require(target.is_dir() and (target / "plan.md").is_file(), "existing output directory is not owned by this registered task")
    if (target / "report.json").is_file():
        _require(resume_checkpoint is None, "completed task does not accept a resume checkpoint")
        return
    formal_checkpoints = list((target / "checkpoints").glob("seed_*_arm_*_step*.pt")) if (target / "checkpoints").is_dir() else []
    if resume_checkpoint is not None:
        _require(resume_checkpoint.is_file() and target.resolve() in resume_checkpoint.resolve().parents, "resume checkpoint must be an explicit existing file inside this task output")
    elif formal_checkpoints:
        _require(False, "formal checkpoints already exist; continue only with an explicit --resume-checkpoint")


def _preserve_prior_failure(target: Path) -> None:
    failure = target / "invalid_harness.json"
    if not failure.is_file():
        return
    index = 1
    while (target / f"engineering_failure_{index:02d}.json").exists():
        index += 1
    os.replace(failure, target / f"engineering_failure_{index:02d}.json")


def _plan_text() -> str:
    return """# 云端独立预检、三种子训练与终点测量\n\n科学问题：从同一A4700模型与Adam状态出发，扩大canonical TRAIN拟合人口后，固定lambda=0.11159457624315472前景项能否在三个配对训练批次轨迹上都降低DEV类别2、4的前景误差，并优于共同起点？\n\nA臂保留原生产目标；B臂仅增加固定前景项。每个训练种子的两臂均从共同A4700状态恢复，配对批次序列相同；仅Generator相位进入Adam，Encoder与四个评价器冻结，但Encoder autograd继续参与E_S/E_I/E_C语义梯度联合归一化。每臂每种子2,000步，三种子共12,000次科学更新。\n\n拟合人口为2,000行canonical TRAIN，保留旧200行并新增每类180行；新增行排除3,300行祖先人口与旧500行对照。确认人口为1,000行canonical DEV，每类100行，不参与更新、调权或终点选择。DEV有历史评价器开发用途，不称全项目盲测。\n\n云端READY需执行重复本机前向、独立Q_fg值/导数、优化梯度等价、三种子六臂三步更新、零lambda负控、冻结隔离和显式恢复检查。保留旧Windows/云端冻结重放不匹配及其原容差，不把旧重放用于本轮新准入。\n\n正式测量只消费共同A4700和六个新终点的完整1,000行DEV保存读出；旧A5400/B5400只作描述。每种子必须同时满足D=Q_B-Q_A<-(2e-7+2e-5*max(|Q_A|,|Q_B|))且Q_B<Q_0才支持本范围的重复收益。完成有效测量后停止，不自动采用模型或追加训练。"""


def _load_all_inputs(root: Path, artifact_dir: Path, checkpoint: Path, gradient_dir: Path, reference_dir: Path) -> dict[str, Any]:
    sources = _load_sources(root, artifact_dir, checkpoint, gradient_dir, reference_dir)
    inputs = _population_inputs(root, sources)
    inputs["sources"] = sources
    inputs["batch_orders"] = {seed: _make_schedule(inputs["fit_labels"], seed) for seed in TRAIN_SEEDS}
    return inputs


def _load_endpoint(sources: Mapping[str, Any], checkpoint: Mapping[str, Any], device: torch.device) -> tuple[Any, torch.optim.Optimizer, dict[str, nn.Parameter]]:
    consumer, optimizer, parameters = _new_endpoint(sources, device)
    if checkpoint.get("task_id") == TASK_ID:
        _copy_endpoint_state(consumer, checkpoint)
        optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer_state"]))
        capacity._restore_rng_state(checkpoint["rng_state"])
    else:
        _copy_endpoint_state(consumer, checkpoint)
    return consumer, optimizer, parameters


def _trajectory_order() -> list[tuple[int, str]]:
    return [(seed, arm) for seed in TRAIN_SEEDS for arm in ("A", "B")]


def _checkpoint_progress_state(target: Path, checkpoint_path: Path | None, local_update: int, reason: str | None) -> dict[str, Any]:
    if checkpoint_path is None:
        _require(local_update == 0 and reason is None, "checkpoint progress without a saved path must remain at the task start")
        return {"last_checkpoint": None, "last_checkpoint_update": 0, "last_checkpoint_adam_step": SOURCE_STEP, "last_checkpoint_reason": None}
    _require(checkpoint_path.is_file() and target.resolve() in checkpoint_path.resolve().parents and local_update > 0, "last checkpoint progress must name an existing saved file inside the task output")
    return {
        "last_checkpoint": checkpoint_path.relative_to(target).as_posix(),
        "last_checkpoint_update": local_update,
        "last_checkpoint_adam_step": SOURCE_STEP + local_update,
        "last_checkpoint_reason": reason,
    }


def _run_trajectory(target: Path, inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module], seed: int, arm: str, resume: Mapping[str, Any] | None) -> dict[str, Any]:
    device = torch.device("cuda:0")
    consumer, optimizer, parameters = _new_endpoint(inputs["sources"], device)
    batch_order = inputs["batch_orders"][seed]
    if resume is None:
        local = 0
        history: list[dict[str, Any]] = []
        capacity._seed_all(seed)
    else:
        _restore_checkpoint(resume, consumer, optimizer)
        local = int(resume["task_local_updates"])
        history = copy.deepcopy(list(resume["training_history"]))
    last_checkpoint_path = _checkpoint_path(target, seed, arm, local) if local > 0 else None
    if last_checkpoint_path is not None:
        _require(last_checkpoint_path.is_file(), "resume cursor does not point to an existing verified checkpoint")
    last_checkpoint_update = local if last_checkpoint_path is not None else 0
    last_checkpoint_reason = str(resume.get("checkpoint_reason")) if resume is not None else None
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    frozen_before = capacity._frozen_runtime_state(consumer)
    checkpoint_policy = _checkpoint_policy(target)
    stop_requested = {"value": False}
    previous_handlers: dict[int, Any] = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda _signal, _frame: stop_requested.__setitem__("value", True))
        while local < UPDATES_PER_ARM:
            history.append(_train_step(consumer, optimizer, parameters, evaluators, inputs, inputs["ols_coefficients"], batch_order, seed, arm, local))
            local += 1
            checkpoint_path = None
            should_checkpoint = local % checkpoint_policy["interval"] == 0 or local == UPDATES_PER_ARM or stop_requested["value"]
            if should_checkpoint:
                reason = "signal_requested" if stop_requested["value"] else f"registered_{checkpoint_policy['interval']}_update_boundary" if local < UPDATES_PER_ARM else "terminal"
                checkpoint_path = _save_checkpoint(target, consumer, optimizer, inputs, seed, arm, local, batch_order, history, reason=reason)
                last_checkpoint_path = checkpoint_path
                last_checkpoint_update = local
                last_checkpoint_reason = reason
            if local % 100 == 0 or local == UPDATES_PER_ARM or stop_requested["value"]:
                completed_before_current = 0
                for prior_seed, prior_arm in _trajectory_order():
                    if (prior_seed, prior_arm) == (seed, arm):
                        break
                    prior_final = _checkpoint_path(target, prior_seed, prior_arm, UPDATES_PER_ARM)
                    if prior_final.is_file():
                        completed_before_current += UPDATES_PER_ARM
                _write_json(target / "progress.json", {
                    "task_id": TASK_ID,
                    "status": "RUNNING",
                    "current_seed": seed,
                    "current_arm": arm,
                    "current_task_local_update": local,
                    "current_optimizer_step": SOURCE_STEP + local,
                    "scientific_updates_completed": completed_before_current + local,
                    "scientific_updates_authorized": TOTAL_UPDATES,
                    **_checkpoint_progress_state(target, last_checkpoint_path, last_checkpoint_update, last_checkpoint_reason),
                }, replace=True)
                if stop_requested["value"]:
                    raise TrainingInterrupted(f"{seed}/{arm} checkpointed at update {local}; resume only from that explicit file")
        _require(local == UPDATES_PER_ARM and int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == SOURCE_STEP + UPDATES_PER_ARM, f"{seed}/{arm} did not complete exactly 2,000 updates")
        _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), f"{seed}/{arm} changed the frozen Encoder")
        capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), f"{seed}/{arm} frozen runtime")
        _require(len(history) == UPDATES_PER_ARM, f"{seed}/{arm} training history is incomplete")
        return _read_checkpoint_file(_checkpoint_path(target, seed, arm, UPDATES_PER_ARM), target, inputs, inputs["batch_orders"])
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)
        del consumer, optimizer, parameters
        torch.cuda.empty_cache()


def _run_training(target: Path, inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module], resume_path: Path | None) -> dict[tuple[int, str], dict[str, Any]]:
    batch_orders = inputs["batch_orders"]
    selected_payload = None
    selected_key = None
    if resume_path is not None:
        selected_payload = _read_checkpoint_file(resume_path, target, inputs, batch_orders)
        _validate_explicit_resume_is_latest(target, resume_path, selected_payload)
        selected_key = (int(selected_payload["seed"]), str(selected_payload["arm"]))
    endpoints: dict[tuple[int, str], dict[str, Any]] = {}
    seen_selected = selected_payload is None
    for key in _trajectory_order():
        seed, arm = key
        final_path = _checkpoint_path(target, seed, arm, UPDATES_PER_ARM)
        if final_path.is_file():
            payload = _read_checkpoint_file(final_path, target, inputs, batch_orders)
            endpoints[key] = payload
            if key == selected_key:
                seen_selected = True
            continue
        existing = [path for path in (target / "checkpoints").glob(f"seed_{seed}_arm_{arm.lower()}_step*.pt")]
        if selected_key == key:
            _require(selected_payload is not None, "resume checkpoint payload is missing")
            start_payload = selected_payload
            seen_selected = True
        else:
            _require(not existing, f"{seed}/{arm} has partial checkpoints and requires its explicit --resume-checkpoint")
            start_payload = None
        endpoints[key] = _run_trajectory(target, inputs, evaluators, seed, arm, start_payload)
    _require(seen_selected, "explicit resume checkpoint does not correspond to an unfinished registered trajectory")
    _require(len(endpoints) == 6 and all(payload["task_local_updates"] == UPDATES_PER_ARM for payload in endpoints.values()), "all six scientific trajectories must complete exactly 2,000 updates")
    return endpoints


if __name__ == "__main__":
    main()
