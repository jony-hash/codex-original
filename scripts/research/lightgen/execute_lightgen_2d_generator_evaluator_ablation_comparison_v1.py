"""Compare three-evaluator and two-evaluator training on the fixed large Generator."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity
from scripts.research.lightgen import execute_lightgen_2d_generator_capacity_accumulation_comparison_v1 as accumulation


TASK_ID = "M0_GENERATOR_EVALUATOR_ABLATION_COMPARISON_V1"
REPORT_SCHEMA = "m0_generator_evaluator_ablation_comparison_report.v1"
PREFLIGHT_SCHEMA = "m0_generator_evaluator_ablation_preflight.v1"
CHECKPOINT_SCHEMA = "m0_generator_evaluator_ablation_checkpoint.v1"
SOURCE_TASK_ID = accumulation.TASK_ID
DEFAULT_CAPACITY = Path("outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_03")
DEFAULT_OUTPUT = Path("outputs/20261008_m0_generator_evaluator_ablation_comparison_v1/runs/attempt_01")
PARENT_UPDATES = 1000
TOTAL_UPDATES = 1700
ADDED_UPDATES = TOTAL_UPDATES - PARENT_UPDATES
FIT_ROWS = capacity.FIT_ROWS
HISTORIC_ROWS = capacity.HISTORIC_ROWS
CONFIRMATION_ROWS = capacity.CONFIRMATION_ROWS
TOTAL_ROWS = capacity.TOTAL_ROWS
BATCH_COUNT = capacity.BATCH_COUNT
BATCH_SIZE = capacity.BATCH_SIZE
ARM_NAMES = ("A_three_evaluator", "B_two_evaluator")
EVALUATOR_NAMES = capacity.EVALUATOR_NAMES
TRAINING_EVALUATORS = {"A_three_evaluator": ("E_S", "E_I", "E_C"), "B_two_evaluator": ("E_I", "E_C")}
FOCUS_CLASSES = capacity.FOCUS_CLASSES
DISPLAY_CLASSES = capacity.DISPLAY_CLASSES
RTOL, ATOL = capacity.RTOL, capacity.ATOL
CHECKPOINT_INTERVAL = 100
CHECKPOINT_PATTERN = re.compile(r"arm_(a|b)_step(\d{4})\.pt$")
RESOURCE_NAME = "resource_telemetry_evaluator_ablation.jsonl"
NUMERIC_MODE = {
    "dtype": "float32",
    "autocast": False,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "float32_matmul_precision": "highest",
}


class EvaluatorAblationRunError(RuntimeError):
    """Invalid task input or violated training/recovery invariant."""


class TrainingInterrupted(RuntimeError):
    """An interrupt was recorded at a safe optimizer boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluatorAblationRunError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _cpu_tree(value: Any) -> Any:
    return capacity._cpu_tree(value)


def _tree_equal(left: Any, right: Any) -> bool:
    return capacity._tree_equal(left, right)


def _state_snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return capacity._state_snapshot(module)


def _write_torch_new(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    capacity._write_torch_new(path, payload, kind)


def _write_json_new(path: Path, payload: Mapping[str, Any], kind: str) -> None:
    capacity._write_json_new(path, payload, kind)


def _load_parent_inputs(root: Path, capacity_dir: Path):
    population_path = capacity_dir / "input_population.json"
    report_path = capacity_dir / "report.json"
    _require(population_path.is_file() and report_path.is_file(), "the explicit 1000-step accumulation attempt is incomplete")
    source_population = json.loads(population_path.read_text(encoding="utf-8"))
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(source_population.get("task_id") == SOURCE_TASK_ID, "accumulation input population is from another task")
    _require(source_report.get("task_id") == SOURCE_TASK_ID, "accumulation report is from another task")
    _require(
        source_report.get("status") == "VALID_FAILED"
        and source_report.get("classification") == "MIXED_OR_INSUFFICIENT",
        "the named 1000-step source attempt is not the completed mixed/insufficient result",
    )
    _require(source_population.get("total_updates_per_arm") == PARENT_UPDATES, "source population does not declare the shared 1000-step endpoint")
    parent_value = source_population.get("source_parent_task_dir")
    _require(isinstance(parent_value, str) and parent_value, "the source population has no explicit semantic-target parent")
    semantic_parent = _resolve(root, parent_value)
    inputs, semantic_population, paths, source_state = capacity._load_inputs(root, semantic_parent)
    _require(source_population.get("source_row_indices") == semantic_population["source_row_indices"], "source TRAIN rows differ from the fixed semantic-target population")
    _require(source_population.get("labels") == semantic_population["labels"], "source labels differ from the fixed semantic-target population")
    _require(source_population.get("roles") == semantic_population["roles"], "source row roles differ from the fixed semantic-target population")

    source_checkpoint_path = capacity_dir / "arm_b_step1000.pt"
    _require(source_checkpoint_path.is_file(), "the explicit shared B1000 checkpoint is missing")
    source_checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=True)
    _require(source_checkpoint.get("schema") == accumulation.CHECKPOINT_SCHEMA, "B1000 checkpoint schema differs")
    _require(source_checkpoint.get("task_id") == SOURCE_TASK_ID and source_checkpoint.get("arm") == "B_large", "B1000 checkpoint identity differs")
    _require(source_checkpoint.get("completed_updates") == PARENT_UPDATES and source_checkpoint.get("optimizer_step") == PARENT_UPDATES, "B1000 Generator/Adam step differs")
    _require(source_checkpoint.get("population") == source_population, "B1000 checkpoint population differs from its source run")
    _require(source_checkpoint.get("architecture_layout") == "two_original_100x100_then_two_400x400_phase_planes_flattened", "B1000 Generator layout differs")
    _require(source_checkpoint.get("optimizer_parameter_names") == ["generator.phase"], "B1000 Adam scope differs")
    _require(source_checkpoint.get("next_batch_index") == 0 and source_checkpoint.get("next_batch_cursor") == 0, "B1000 next-batch cursor differs")
    parameters = source_checkpoint.get("trainable_parameter_state")
    _require(isinstance(parameters, Mapping) and tuple(parameters["generator.phase"].shape) == (capacity.PHASE_LARGE_COUNT,), "B1000 Generator phase shape differs")
    _require(tuple(parameters["encoder.phase"].shape) == (2, 1, 100, 100), "B1000 Encoder phase shape differs")
    _require(all(bool(torch.isfinite(value).all().item()) for value in parameters.values()), "B1000 model state is nonfinite")
    adam_state = source_checkpoint["optimizer_state"]["state"][0]
    _require(int(float(adam_state["step"].item())) == PARENT_UPDATES, "B1000 Adam moments are not at step 1000")
    _require(tuple(adam_state["exp_avg"].shape) == (capacity.PHASE_LARGE_COUNT,) and tuple(adam_state["exp_avg_sq"].shape) == (capacity.PHASE_LARGE_COUNT,), "B1000 Adam moment shape differs")
    _require(bool(torch.isfinite(adam_state["exp_avg"]).all().item()) and bool(torch.isfinite(adam_state["exp_avg_sq"]).all().item()), "B1000 Adam moments are nonfinite")
    _require(isinstance(source_checkpoint.get("training_history"), list) and len(source_checkpoint["training_history"]) == PARENT_UPDATES, "B1000 history length differs")
    source_rows = inputs["source_indices"].tolist()
    source_labels = inputs["labels"].tolist()
    for step, record in enumerate(source_checkpoint["training_history"], start=1):
        batch_index = (step - 1) % BATCH_COUNT
        positions = list(range(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE))
        _require(record.get("update") == step, f"B1000 history update count differs at step {step}")
        _require(record.get("batch_index") == batch_index and record.get("fit_positions") == positions, f"B1000 history batch mapping differs at step {step}")
        _require(record.get("source_row_indices") == [source_rows[position] for position in positions], f"B1000 history TRAIN rows differ at step {step}")
        _require(record.get("labels") == [source_labels[position] for position in positions], f"B1000 history labels differ at step {step}")
    _require(_tree_equal(source_checkpoint["references"], source_state["references"]), "B1000 BOGT references differ from the shared source")
    _require(_tree_equal(source_checkpoint["weights"], source_state["weights"]), "B1000 BOGT weights differ from the shared source")

    parent_readouts = {}
    path = capacity_dir / "readouts_B_step1000.pt"
    _require(path.is_file(), "source B1000 readout is missing")
    readout = torch.load(path, map_location="cpu", weights_only=True)
    capacity._validate_readout(readout, inputs, "B_step1000")
    parent_readouts["B_step1000"] = readout
    parent_summary = source_report["endpoint_metrics"]["B_step1000"]["evaluators"]["E_D"]["by_role"]["confirmation"]
    derived_summary, _ = capacity._measure_statistics(parent_readouts["B_step1000"], inputs, "B_step1000")
    _require(parent_summary["per_class_correct"] == derived_summary["evaluators"]["E_D"]["by_role"]["confirmation"]["per_class_correct"], "B1000 saved readout and source report disagree")
    source_checkpoints = {arm: dict(source_checkpoint) for arm in ARM_NAMES}
    parent_summaries = {"B_step1000": derived_summary}

    population = {
        "task_id": TASK_ID,
        "source_capacity_dir": capacity_dir.as_posix(),
        "source_capacity_task_id": SOURCE_TASK_ID,
        "source_capacity_status": source_report["status"],
        "source_capacity_classification": source_report["classification"],
        "source_checkpoint_paths": {"shared_B_step1000": source_checkpoint_path.as_posix()},
        "source_parent_task_dir": semantic_parent.as_posix(),
        "source_row_indices": list(source_population["source_row_indices"]),
        "labels": list(source_population["labels"]),
        "roles": list(source_population["roles"]),
        "fit_rows": FIT_ROWS,
        "historic_comparison_rows": HISTORIC_ROWS,
        "confirmation_rows": CONFIRMATION_ROWS,
        "updates_in_source_checkpoint": PARENT_UPDATES,
        "additional_updates_per_arm": ADDED_UPDATES,
        "total_updates_per_arm": TOTAL_UPDATES,
        "fit_batch_size": BATCH_SIZE,
        "fit_batch_count": BATCH_COUNT,
        "additional_cycles": ADDED_UPDATES // BATCH_COUNT,
        "source_input_paths": dict(semantic_population["input_paths"]),
        "source_evaluator_sets": {arm: list(names) for arm, names in TRAINING_EVALUATORS.items()},
        "confirmation_is_canonical_train_not_claimed_unseen": True,
    }
    return inputs, population, paths, source_state, source_population, source_report, source_checkpoints, parent_readouts, parent_summaries


def _validate_source_checkpoint(checkpoint: Mapping[str, Any], population: Mapping[str, Any], arm: str) -> None:
    _require(checkpoint.get("schema") == capacity.CHECKPOINT_SCHEMA, f"{arm} source checkpoint schema differs")
    _require(checkpoint.get("task_id") == SOURCE_TASK_ID and checkpoint.get("arm") == arm, f"{arm} source task/arm identity differs")
    _require(checkpoint.get("completed_updates") == PARENT_UPDATES and checkpoint.get("optimizer_step") == PARENT_UPDATES, f"{arm} source endpoint is not at Generator Adam step 300")
    _require(checkpoint.get("source_checkpoint_step") == capacity.SOURCE_STEP, f"{arm} source checkpoint step differs")
    _require(checkpoint.get("source_optimizer_steps") == {"encoder.phase": 2200, "generator.phase": 2500}, f"{arm} source Adam provenance differs")
    _require(checkpoint.get("optimizer_parameter_names") == ["generator.phase"], f"{arm} source optimizer scope differs")
    _require(checkpoint.get("population") == dict(population), f"{arm} source checkpoint population differs")
    _require(checkpoint.get("architecture_layout") == ("two_original_100x100_phase_planes" if arm == "A_three_evaluator" else "two_original_100x100_then_two_400x400_phase_planes_flattened"), f"{arm} source phase layout differs")
    _require(checkpoint.get("next_batch_index") == PARENT_UPDATES % BATCH_COUNT and checkpoint.get("next_batch_cursor") == PARENT_UPDATES % BATCH_COUNT, f"{arm} source next-batch cursor differs")
    _require(checkpoint.get("numeric_mode") == NUMERIC_MODE, f"{arm} source numeric mode differs")
    state = checkpoint.get("trainable_parameter_state")
    _require(isinstance(state, Mapping) and set(state) == {"encoder.phase", "generator.phase"}, f"{arm} source model state is incomplete")
    _require(tuple(state["encoder.phase"].shape) == (2, 1, 100, 100), f"{arm} source Encoder shape differs")
    expected_generator_shape = (2, 1, 100, 100) if arm == "A_three_evaluator" else (capacity.PHASE_LARGE_COUNT,)
    _require(tuple(state["generator.phase"].shape) == expected_generator_shape, f"{arm} source Generator shape differs")
    _require(all(bool(torch.isfinite(value).all().item()) for value in state.values()), f"{arm} source model state is nonfinite")
    optimizer = checkpoint.get("optimizer_state")
    _require(isinstance(optimizer, Mapping), f"{arm} source Adam state is missing")
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    _require(isinstance(groups, list) and len(groups) == 1 and groups[0].get("params") == [0], f"{arm} source Adam parameter mapping differs")
    _require(isinstance(states, Mapping) and set(states) == {0}, f"{arm} source Adam has unexpected state entries")
    group = groups[0]
    _require(group.get("lr") == capacity.GENERATOR_LEARNING_RATE, f"{arm} source Adam learning rate differs")
    _require(tuple(group.get("betas", ())) == capacity.GENERATOR_BETAS and group.get("eps") == capacity.GENERATOR_EPS, f"{arm} source Adam coefficients differ")
    _require(group.get("weight_decay") == capacity.GENERATOR_WEIGHT_DECAY, f"{arm} source Adam weight decay differs")
    adam = states[0]
    _require(int(float(adam["step"].item())) == PARENT_UPDATES, f"{arm} saved Generator Adam moments are not at step 300")
    _require(tuple(adam["exp_avg"].shape) == expected_generator_shape and tuple(adam["exp_avg_sq"].shape) == expected_generator_shape, f"{arm} source Adam moment shapes differ")
    _require(bool(torch.isfinite(adam["exp_avg"]).all().item()) and bool(torch.isfinite(adam["exp_avg_sq"]).all().item()), f"{arm} source Adam moments are nonfinite")
    history = checkpoint.get("training_history")
    _require(isinstance(history, list) and len(history) == PARENT_UPDATES, f"{arm} source history does not contain 300 updates")


def _validate_history_rows(history: Sequence[Mapping[str, Any]], inputs: Mapping[str, Any], arm: str) -> None:
    source_indices = inputs["source_indices"].tolist()
    labels = inputs["labels"].tolist()
    for step, record in enumerate(history, start=1):
        batch_index = (step - 1) % BATCH_COUNT
        positions = list(range(batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE))
        _require(record.get("update") == step and record.get("arm") == arm, f"{arm} source history step/arm differs at update {step}")
        _require(record.get("batch_index") == batch_index and record.get("fit_positions") == positions, f"{arm} source history batch mapping differs at update {step}")
        _require(record.get("source_row_indices") == [source_indices[position] for position in positions], f"{arm} source history TRAIN rows differ at update {step}")
        _require(record.get("labels") == [labels[position] for position in positions], f"{arm} source history labels differ at update {step}")


def _load_arm(
    inputs: Mapping[str, Any],
    base_state: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    arm: str,
    device: torch.device,
):
    _require(arm in ARM_NAMES, f"unknown evaluator ablation arm: {arm}")
    consumer, optimizer, parameters = capacity._new_endpoint(inputs, base_state, "B_large", device)
    parameter_state = checkpoint["trainable_parameter_state"]
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = parameter_state[name]
            _require(tuple(value.shape) == tuple(parameter.shape), f"{arm} checkpoint parameter shape changed for {name}")
            _require(value.dtype == parameter.dtype and bool(torch.isfinite(value).all().item()), f"{arm} checkpoint parameter dtype or values changed for {name}")
            parameter.copy_(value.to(parameter.device))
    _load_adam_state(optimizer, parameters["generator.phase"], checkpoint["optimizer_state"], expected_step=int(checkpoint["completed_updates"]))
    capacity._assert_state_equal(checkpoint["frozen_runtime_state"], capacity._frozen_runtime_state(consumer), f"{arm} loaded OLS/runtime state")
    capacity._restore_rng_state(checkpoint["rng_state"])
    return consumer, optimizer, parameters


def _load_adam_state(
    optimizer: torch.optim.Optimizer,
    phase: nn.Parameter,
    saved_state: Mapping[str, Any],
    *,
    expected_step: int,
) -> None:
    _require(
        len(optimizer.param_groups) == 1
        and len(optimizer.param_groups[0]["params"]) == 1
        and optimizer.param_groups[0]["params"][0] is phase,
        "task Adam parameter scope differs",
    )
    _require(len(saved_state.get("param_groups", [])) == 1, "saved Adam has multiple or missing parameter groups")
    group = saved_state["param_groups"][0]
    _require(group.get("params") == [0], "saved Adam is not bound to one Generator parameter")
    _require(group.get("lr") == capacity.GENERATOR_LEARNING_RATE, "saved Adam learning rate changed")
    _require(tuple(group.get("betas", ())) == capacity.GENERATOR_BETAS and group.get("eps") == capacity.GENERATOR_EPS, "saved Adam coefficients changed")
    _require(group.get("weight_decay") == capacity.GENERATOR_WEIGHT_DECAY, "saved Adam weight decay changed")
    state = saved_state.get("state")
    _require(isinstance(state, Mapping) and set(state) == {0}, "saved Adam state is absent or out of scope")
    _require(tuple(state[0]["exp_avg"].shape) == tuple(phase.shape) and tuple(state[0]["exp_avg_sq"].shape) == tuple(phase.shape), "saved Adam moment shape differs")
    _require(bool(torch.isfinite(state[0]["exp_avg"]).all().item()) and bool(torch.isfinite(state[0]["exp_avg_sq"]).all().item()), "saved Adam moments are nonfinite")
    # PyTorch may retain the input CPU scalar step tensor by reference when
    # loading Adam state. A disposable optimizer step must not mutate the
    # source checkpoint used by the next arm or an A/A replay.
    optimizer.load_state_dict(_cpu_tree(saved_state))
    actual = int(float(optimizer.state[phase]["step"].item()))
    _require(actual == expected_step, f"loaded Adam step is {actual}, expected {expected_step}")


def _batch_for_update(cumulative_completed: int) -> int:
    _require(PARENT_UPDATES <= cumulative_completed < TOTAL_UPDATES, "next update is outside the authorized 1001..1700 range")
    return cumulative_completed % BATCH_COUNT


def _checkpoint_name(arm: str, completed: int) -> str:
    _require(arm in ARM_NAMES, f"unknown checkpoint arm: {arm}")
    letter = "a" if arm == ARM_NAMES[0] else "b"
    return f"arm_{letter}_step{completed:04d}.pt"


def _checkpoint_payload(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    arm: str,
    completed: int,
    history: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    _require(PARENT_UPDATES <= completed <= TOTAL_UPDATES, "checkpoint total update count is outside 1000..1700")
    parameters = capacity._parameters(consumer)
    step = int(float(optimizer.state[parameters["generator.phase"]]["step"].item()))
    _require(step == completed, "checkpoint Generator Adam step differs from cumulative updates")
    _require(len(history) == completed, "checkpoint history does not contain cumulative updates")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "arm": arm,
        "architecture_layout": source_checkpoint["architecture_layout"],
        "source_capacity_checkpoint": population["source_checkpoint_paths"]["shared_B_step1000"],
        "parent_updates": PARENT_UPDATES,
        "additional_updates": completed - PARENT_UPDATES,
        "completed_updates": completed,
        "next_batch_index": completed % BATCH_COUNT,
        "next_batch_cursor": completed % BATCH_COUNT,
        "population": dict(population),
        "trainable_parameter_state": capacity._snapshot_parameters(parameters),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "optimizer_parameter_names": ["generator.phase"],
        "optimizer_step": step,
        "frozen_runtime_state": capacity._frozen_runtime_state(consumer),
        "references": consumer.references.to_payload(),
        "weights": consumer.weights.to_payload(),
        "training_history": [dict(row) for row in history],
        "source_rng_state": source_checkpoint["rng_state"],
        "rng_state": capacity._rng_state(),
        "numeric_mode": dict(NUMERIC_MODE),
    }


def _save_checkpoint(
    target: Path,
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    arm: str,
    completed: int,
    history: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
) -> Path:
    path = target / _checkpoint_name(arm, completed)
    capacity.diagnostic._check_output_budget(target)
    _write_torch_new(path, _checkpoint_payload(consumer, optimizer, arm, completed, history, population, source_checkpoint), "1000-to-1700 evaluator ablation Generator Adam recovery checkpoint")
    return path


def _read_checkpoint(path: Path, target: Path, population: Mapping[str, Any], source_checkpoints: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _require(path.resolve().parent == target.resolve() and path.is_file(), "resume requires an explicit checkpoint inside this task output")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA and checkpoint.get("task_id") == TASK_ID, "resume checkpoint is not from the evaluator ablation task")
    arm, completed = checkpoint.get("arm"), checkpoint.get("completed_updates")
    _require(arm in ARM_NAMES and isinstance(completed, int) and PARENT_UPDATES <= completed <= TOTAL_UPDATES, "resume checkpoint arm/update count is invalid")
    _require(checkpoint.get("parent_updates") == PARENT_UPDATES and checkpoint.get("additional_updates") == completed - PARENT_UPDATES, "resume source/added update counts differ")
    _require(checkpoint.get("population") == dict(population), "resume checkpoint population, roles, labels, or inputs changed")
    _require(checkpoint.get("next_batch_index") == completed % BATCH_COUNT and checkpoint.get("next_batch_cursor") == completed % BATCH_COUNT, "resume batch cursor differs")
    _require(path.name == _checkpoint_name(arm, completed), "resume filename does not match checkpoint state")
    _require(checkpoint.get("optimizer_step") == completed and checkpoint.get("optimizer_parameter_names") == ["generator.phase"], "resume checkpoint does not retain cumulative Generator Adam state")
    _require(checkpoint.get("numeric_mode") == NUMERIC_MODE, "resume numeric mode differs")
    _require(checkpoint.get("source_capacity_checkpoint") == population["source_checkpoint_paths"]["shared_B_step1000"], "resume parent checkpoint lineage differs")
    history = checkpoint.get("training_history")
    source_history = source_checkpoints[arm]["training_history"]
    _require(isinstance(history, list) and len(history) == completed and _tree_equal(history[:PARENT_UPDATES], source_history), "resume history is not the source 1000-step history plus continuation")
    for step, record in enumerate(history[PARENT_UPDATES:], start=PARENT_UPDATES + 1):
        _require(record.get("update") == step and record.get("arm") == arm, f"resume continuation history is invalid at step {step}")
    state = checkpoint.get("trainable_parameter_state")
    _require(isinstance(state, Mapping) and set(state) == {"encoder.phase", "generator.phase"}, "resume model state is incomplete")
    source_parameters = source_checkpoints[arm]["trainable_parameter_state"]
    for name, value in state.items():
        _require(tuple(value.shape) == tuple(source_parameters[name].shape) and value.dtype == source_parameters[name].dtype, f"resume parameter shape/dtype changed for {name}")
        _require(bool(torch.isfinite(value).all().item()), "resume model state is nonfinite")
    optimizer_parameter = nn.Parameter(torch.zeros_like(state["generator.phase"]))
    optimizer = torch.optim.Adam(
        [optimizer_parameter],
        lr=capacity.GENERATOR_LEARNING_RATE,
        betas=capacity.GENERATOR_BETAS,
        eps=capacity.GENERATOR_EPS,
        weight_decay=capacity.GENERATOR_WEIGHT_DECAY,
    )
    _load_adam_state(optimizer, optimizer_parameter, checkpoint["optimizer_state"], expected_step=completed)
    _require(len(checkpoint.get("rng_state", {})) >= 3, "resume RNG state is incomplete")
    _require(_tree_equal(checkpoint.get("frozen_runtime_state"), source_checkpoints[arm]["frozen_runtime_state"]), "resume changed frozen runtime buffers")
    later = []
    for candidate in target.glob("arm_*_step*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if match is None:
            continue
        candidate_arm = "A_three_evaluator" if match.group(1) == "a" else "B_two_evaluator"
        candidate_step = int(match.group(2))
        if arm == "A_three_evaluator":
            candidate_is_later = (
                candidate_arm == "A_three_evaluator" and candidate_step > completed
            ) or (candidate_arm == "B_two_evaluator" and candidate_step > PARENT_UPDATES)
        else:
            candidate_is_later = (
                candidate_arm == "B_two_evaluator" and candidate_step > completed
            ) or (candidate_arm == "A_three_evaluator" and candidate_step != TOTAL_UPDATES)
        if candidate_is_later:
            later.append(candidate.name)
    _require(not later, f"selected checkpoint predates saved task progress: {sorted(later)}")
    report_path = target / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _require(report.get("task_id") == TASK_ID and report.get("status") in {"VALID_COMPLETED", "VALID_FAILED"}, "terminal report is invalid")
        _require(arm == "B_two_evaluator" and completed == TOTAL_UPDATES, "terminal output requires the explicit final B checkpoint")
    return dict(checkpoint)


def _completed_resume_report(target: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any] | None:
    report_path = target / "report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("task_id") == TASK_ID and report.get("status") in {"VALID_COMPLETED", "VALID_FAILED"}, "saved terminal report is invalid")
    _require(checkpoint.get("arm") == "B_two_evaluator" and checkpoint.get("completed_updates") == TOTAL_UPDATES, "terminal recovery requires the explicit final B checkpoint")
    return report


def _restore_checkpoint(checkpoint: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer, arm: str) -> None:
    _require(checkpoint.get("arm") == arm, "resume checkpoint selects the wrong arm")
    parameters = capacity._parameters(consumer)
    with torch.no_grad():
        for name, parameter in parameters.items():
            value = checkpoint["trainable_parameter_state"][name]
            _require(tuple(value.shape) == tuple(parameter.shape) and value.dtype == parameter.dtype, f"resume parameter shape/dtype changed for {name}")
            parameter.copy_(value.to(parameter.device))
    _load_adam_state(optimizer, parameters["generator.phase"], checkpoint["optimizer_state"], expected_step=int(checkpoint["completed_updates"]))
    capacity._restore_rng_state(checkpoint["rng_state"])
    capacity._assert_state_equal(checkpoint["frozen_runtime_state"], capacity._frozen_runtime_state(consumer), "resumed frozen runtime")


def _gpu_snapshot(device: torch.device) -> dict[str, Any]:
    return diagnostic._resource_snapshot(device)


def _resource_telemetry(target: Path, device: torch.device, phase: str, progress: Mapping[str, Any], started: float) -> dict[str, Any]:
    record = {
        "task_id": TASK_ID,
        "phase": phase,
        "progress": dict(progress),
        "elapsed_seconds_observation_only": time.perf_counter() - started,
        "resource_observation": _gpu_snapshot(device),
    }
    with (target / RESOURCE_NAME).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
    return record


def _disk_probe(target: Path, source_checkpoint_paths: Mapping[str, Path]) -> dict[str, Any]:
    temporary_root = Path(tempfile.gettempdir()).resolve()
    _require(_root() not in temporary_root.parents and temporary_root != _root(), "runtime temporary directory must remain outside the repository")
    with tempfile.TemporaryDirectory(prefix="lg_accum_external_") as name:
        probe = Path(name) / "probe.bin"
        probe.write_bytes(b"lightgen-accumulation-check")
        _require(probe.read_bytes() == b"lightgen-accumulation-check", "external temporary directory write/read failed")
    existing = target
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free
    checkpoint_size = max(path.stat().st_size for path in source_checkpoint_paths.values())
    endpoint_bytes = 4 * TOTAL_ROWS * 200 * 200 * 4
    checkpoint_bytes = 14 * (checkpoint_size + capacity.PHASE_LARGE_COUNT * 12)
    estimate = endpoint_bytes + checkpoint_bytes + 600_000_000
    reserve = 1_610_612_736
    _require(free >= estimate + reserve, f"output volume has {free} bytes; estimated artifacts plus reserve require {estimate + reserve}")
    with tempfile.TemporaryDirectory(prefix="lg_accum_output_", dir=target.parent) as name:
        probe = Path(name) / "probe.pt"
        torch.save({"probe": torch.tensor([1.0, 2.0])}, probe)
        _require(torch.equal(torch.load(probe, map_location="cpu", weights_only=True)["probe"], torch.tensor([1.0, 2.0])), "output directory write/read failed")
    return {
        "external_temp_write_read": True,
        "output_temp_write_read": True,
        "output_volume": existing.as_posix(),
        "free_bytes_observed": free,
        "estimated_remaining_output_bytes": estimate,
        "reserve_bytes": reserve,
        "required_bytes": estimate + reserve,
    }


def _target_metrics(evaluators: Mapping[str, nn.Module], inputs: Mapping[str, Any]) -> dict[str, Any]:
    return capacity._target_metrics(evaluators, inputs)


def _target_control_passed(target: Mapping[str, Any]) -> bool:
    return capacity._target_control_passed(target)


def _canonical_semantic_gradient(
    gradients: Mapping[str, Mapping[str, torch.Tensor]], evaluator_names: Sequence[str]
) -> tuple[dict[str, torch.Tensor] | None, str | None]:
    if tuple(evaluator_names) not in {("E_S", "E_I", "E_C"), ("E_I", "E_C")}:
        return None, "unsupported semantic evaluator set"
    normalized: list[torch.Tensor] = []
    for name in evaluator_names:
        vector = diagnostic._flatten_group(gradients[name])
        norm = float(vector.detach().to(torch.float64).norm().cpu())
        if not math.isfinite(norm) or norm <= 0.0:
            return None, f"{name} gradient is zero or invalid"
        normalized.append(vector / norm)
    joint = torch.stack(normalized).sum(dim=0)
    joint_rms = float(joint.detach().to(torch.float64).square().mean().sqrt().cpu())
    if not math.isfinite(joint_rms) or joint_rms <= 0.0:
        return None, "equal-weight evaluator direction is zero or invalid"
    scaled = joint / joint_rms * capacity.semantic.DELTA_SIC
    shapes = {name: gradients[evaluator_names[0]][name].shape for name in ("encoder.phase", "generator.phase")}
    return diagnostic._split_group(scaled, shapes), None


def _independent_semantic_reference(
    gradients: Mapping[str, Mapping[str, torch.Tensor]], evaluator_names: Sequence[str]
) -> dict[str, torch.Tensor]:
    """Independent direct-vector reference for evaluator composition in preflight/tests."""
    vectors = []
    for name in evaluator_names:
        flattened = torch.cat(
            [gradients[name][parameter_name].detach().reshape(-1).to(torch.float64) for parameter_name in ("encoder.phase", "generator.phase")]
        )
        norm = torch.linalg.vector_norm(flattened)
        _require(bool(torch.isfinite(norm).item()) and float(norm.item()) > 0.0, f"{name} reference gradient is unavailable")
        vectors.append(flattened / norm)
    combined = torch.stack(vectors, dim=0).sum(dim=0)
    rms = torch.sqrt(torch.mean(combined.square()))
    _require(bool(torch.isfinite(rms).item()) and float(rms.item()) > 0.0, "independent semantic reference cancelled to zero")
    combined = combined / rms * capacity.semantic.DELTA_SIC
    sizes = {name: gradients[evaluator_names[0]][name].numel() for name in ("encoder.phase", "generator.phase")}
    first, second = combined.split((sizes["encoder.phase"], sizes["generator.phase"]))
    return {
        "encoder.phase": first.reshape_as(gradients[evaluator_names[0]]["encoder.phase"]),
        "generator.phase": second.reshape_as(gradients[evaluator_names[0]]["generator.phase"]),
    }


def _verify_evaluator_gradient_composition(
    inputs: Mapping[str, Any],
    base_state: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    evaluators: Mapping[str, nn.Module],
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    batch_cpu, labels_cpu, _positions = capacity.training_batch(inputs["images"], inputs["labels"], 0)
    batch, labels = batch_cpu.to(device), labels_cpu.to(device)
    results: dict[str, Any] = {}
    combined_by_arm = {}
    for arm in ARM_NAMES:
        consumer, optimizer, parameters = _load_arm(inputs, base_state, source_checkpoint, arm, device)
        with torch.no_grad():
            probe = consumer.runtime.forward_from_image(batch[:2])
            _require(bool(torch.isfinite(probe.raw_intensity).all().item()), f"{arm} no-grad production forward is nonfinite")
        del probe
        _planes, _losses, gradients, combined = _large_microbatch_loss_gradients(
            consumer, parameters, evaluators, batch, labels, arm
        )
        if arm == "A_three_evaluator":
            _ref_planes, _ref_losses, _ref_gradients, reference_combined = capacity._large_microbatch_loss_gradients(
                consumer, parameters, evaluators, batch, labels
            )
            differences = {}
            for name in ("encoder.phase", "generator.phase"):
                differences[name] = float((combined[name] - reference_combined[name]).abs().max().detach().cpu())
                _require(torch.allclose(combined[name], reference_combined[name], rtol=2e-5, atol=2e-7), f"A three-evaluator gradient differs from the production reference for {name}")
            results[arm] = {"passed": True, "reference": "existing three-evaluator microbatch producer", "max_abs_difference": differences}
            del _ref_planes, _ref_losses, _ref_gradients, reference_combined
        else:
            _require(set(gradients["semantic"]) == {"E_I", "E_C"}, "B semantic gradient producer included E_S or omitted a selected evaluator")
            independent = _independent_semantic_reference(gradients["semantic"], TRAINING_EVALUATORS[arm])
            production_semantic, unavailable = _canonical_semantic_gradient(gradients["semantic"], TRAINING_EVALUATORS[arm])
            _require(production_semantic is not None, f"B semantic direction is unavailable: {unavailable}")
            differences = {}
            for name in ("encoder.phase", "generator.phase"):
                differences[name] = float((production_semantic[name].to(torch.float64) - independent[name]).abs().max().cpu())
                _require(torch.allclose(production_semantic[name].to(torch.float64), independent[name], rtol=2e-5, atol=2e-7), f"B two-evaluator direction differs from the independent composition reference for {name}")
            results[arm] = {"passed": True, "reference": "independent direct-vector E_I/E_C composition", "max_abs_difference": differences, "included_evaluators": sorted(gradients["semantic"])}
            del independent, production_semantic
        combined_by_arm[arm] = {name: value.detach().cpu() for name, value in combined.items()}
        _require(all(parameter.grad is None for parameter in parameters.values()), f"{arm} gradient composition populated parameter .grad")
        del consumer, optimizer, parameters, _planes, _losses, gradients, combined
        gc.collect()
        torch.cuda.empty_cache()
    ablation_difference = {
        name: float((combined_by_arm["A_three_evaluator"][name] - combined_by_arm["B_two_evaluator"][name]).abs().max().item())
        for name in ("encoder.phase", "generator.phase")
    }
    _require(any(value > 0.0 for value in ablation_difference.values()), "removing E_S did not change the production combined direction")
    return {"passed": True, "by_arm": results, "A_vs_B_combined_gradient_max_abs_difference": ablation_difference}


def _large_microbatch_loss_gradients(
    consumer: Any,
    parameters: Mapping[str, nn.Parameter],
    evaluators: Mapping[str, nn.Module],
    images: torch.Tensor,
    labels: torch.Tensor,
    arm: str,
):
    """Compute one exact ten-row direction; only the selected evaluator set contributes gradients."""
    _require(images.shape[0] == BATCH_SIZE and labels.shape[0] == BATCH_SIZE, "logical update must contain ten fit rows")
    evaluator_names = TRAINING_EVALUATORS[arm]
    references = consumer.references
    _require(references is not None, "frozen BOGT references are missing")
    coefficient_parts = []
    with torch.no_grad():
        for start in range(0, BATCH_SIZE, capacity.LARGE_GRADIENT_MICROBATCH_SIZE):
            stop = min(BATCH_SIZE, start + capacity.LARGE_GRADIENT_MICROBATCH_SIZE)
            result = consumer.runtime.forward_from_image(images[start:stop])
            coefficient_parts.append(result.z_ols.coefficients.detach())
            del result
    coefficients = torch.cat(coefficient_parts, dim=0)
    l_dist, d_dist_d_ols = capacity._dist_loss_and_ols_gradient(coefficients, references.scale)
    parameter_names = ("encoder.phase", "generator.phase")
    dist_gradients = {name: torch.zeros_like(parameters[name]) for name in parameter_names}
    bogt_gradients = {name: torch.zeros_like(parameters[name]) for name in parameter_names}
    semantic_gradients = {
        name: {key: torch.zeros_like(parameters[key]) for key in parameter_names} for name in evaluator_names
    }
    loss_totals = {"L_rec": 0.0, "L_op": 0.0, **{f"{name}_ce": 0.0 for name in EVALUATOR_NAMES if name != "E_D"}}
    power_parts: list[torch.Tensor] = []
    extra_power_parts: list[torch.Tensor] = []
    for start in range(0, BATCH_SIZE, capacity.LARGE_GRADIENT_MICROBATCH_SIZE):
        stop = min(BATCH_SIZE, start + capacity.LARGE_GRADIENT_MICROBATCH_SIZE)
        weight = (stop - start) / BATCH_SIZE
        result = consumer.runtime.forward_from_image(images[start:stop])
        raw = result.raw_intensity
        target_raw = result.image_input.slm_intensity
        z = result.z_ols.coefficients
        d_dist_chunk = d_dist_d_ols[start:stop].to(dtype=z.dtype, device=z.device)
        l_dist_proxy = torch.real(torch.conj(d_dist_chunk) * z).sum()
        l_rec = (raw - target_raw).square().mean()
        planes = capacity.bogt.contracted_plane_powers(result)
        l_op = capacity.bogt.optical_power_loss_from_powers(planes, references.power_reference)
        view = capacity.canonical.cmos_view(raw)
        ce_losses: dict[str, torch.Tensor] = {}
        for name in EVALUATOR_NAMES:
            if name == "E_D":
                continue
            if name in evaluator_names:
                logits = capacity._evaluator_logits(evaluators[name], view, name)
                ce_losses[name] = F.cross_entropy(logits, labels[start:stop], reduction="mean")
            else:
                with torch.no_grad():
                    logits = capacity._evaluator_logits(evaluators[name], view.detach(), name)
                    ce_losses[name] = F.cross_entropy(logits, labels[start:stop], reduction="mean")
        grad_dist = capacity._gradient(l_dist_proxy, parameters, retain_graph=True)
        bogt_loss = capacity.semantic.BETA * l_rec + capacity.semantic.GAMMA * l_op
        grad_bogt = capacity._gradient(bogt_loss, parameters, retain_graph=True)
        grad_semantic = {}
        for index, name in enumerate(evaluator_names):
            grad_semantic[name] = capacity._gradient(
                ce_losses[name], parameters, retain_graph=index < len(evaluator_names) - 1
            )
        for name in parameter_names:
            dist_gradients[name] += grad_dist[name]
            bogt_gradients[name] += weight * grad_bogt[name]
            for evaluator_name in evaluator_names:
                semantic_gradients[evaluator_name][name] += weight * grad_semantic[evaluator_name][name]
        loss_totals["L_rec"] += weight * float(l_rec.detach().to(torch.float64).cpu())
        loss_totals["L_op"] += weight * float(l_op.detach().to(torch.float64).cpu())
        for name, loss in ce_losses.items():
            loss_totals[f"{name}_ce"] += weight * float(loss.detach().to(torch.float64).cpu())
        power_parts.append(planes.detach())
        generator = consumer.runtime.generator
        if isinstance(generator, capacity.ExpandedGenerator):
            _require(generator.last_extra_plane_powers is not None, "expanded Generator omitted new plane powers")
            extra_power_parts.append(generator.last_extra_plane_powers.detach())
        del result, raw, target_raw, z, d_dist_chunk, l_dist_proxy, l_rec, planes, l_op, view
        del ce_losses, grad_dist, bogt_loss, grad_bogt, grad_semantic
    semantic_gradient, unavailable = _canonical_semantic_gradient(semantic_gradients, evaluator_names)
    _require(semantic_gradient is not None, f"semantic direction is unavailable: {unavailable}")
    _require(capacity._vector_norm(dist_gradients["generator.phase"]) == 0.0, "L_dist produced a Generator gradient")
    combined = {
        name: capacity.semantic.ALPHA * dist_gradients[name]
        + bogt_gradients[name]
        + capacity.semantic.KAPPA * semantic_gradient[name]
        for name in parameter_names
    }
    _require(all(bool(torch.isfinite(value).all().item()) for value in combined.values()), "combined gradient is nonfinite")
    _require(all(parameter.grad is None for parameter in parameters.values()), "read-only autograd populated optical .grad")
    losses = {"L_dist": l_dist, **{name: torch.tensor(value, device=images.device) for name, value in loss_totals.items()}}
    if extra_power_parts:
        consumer.runtime.generator.last_extra_plane_powers = torch.cat(extra_power_parts, dim=0)
    return torch.cat(power_parts, dim=0), losses, {"L_dist": dist_gradients, "semantic": semantic_gradients}, combined


def _train_step(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, nn.Parameter],
    evaluators: Mapping[str, nn.Module],
    inputs: Mapping[str, Any],
    update_index: int,
    *,
    arm: str,
) -> dict[str, Any]:
    batch, labels_cpu, positions = capacity.training_batch(inputs["images"], inputs["labels"], update_index % BATCH_COUNT)
    labels = labels_cpu.to(consumer.device)
    planes, losses, gradients, combined = _large_microbatch_loss_gradients(
        consumer, parameters, evaluators, batch.to(consumer.device), labels, arm
    )
    phase = parameters["generator.phase"]
    generator_gradient = combined["generator.phase"]
    _require(tuple(generator_gradient.shape) == tuple(phase.shape), "Generator gradient shape differs")
    _require(capacity._vector_norm(gradients["L_dist"]["generator.phase"]) == 0.0, "L_dist Generator gradient is not structurally zero")
    before = phase.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    phase.grad = generator_gradient.detach().clone()
    _require(parameters["encoder.phase"].grad is None, "Encoder gradient entered the optimizer")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    step = int(float(optimizer.state[phase]["step"].item()))
    _require(step == update_index + 1, "continued Generator Adam step differs from cumulative update")
    delta = phase.detach() - before
    old_gradient, new_gradient = capacity._phase_parts(generator_gradient, "B_large")
    old_delta, new_delta = capacity._phase_parts(delta, "B_large")
    extra_power = consumer.runtime.generator.last_extra_plane_powers
    record = {
        "update": step,
        "arm": arm,
        "batch_index": int(positions[0].item() // BATCH_SIZE),
        "fit_positions": positions.tolist(),
        "source_row_indices": inputs["source_indices"].index_select(0, positions).tolist(),
        "labels": labels_cpu.tolist(),
        "training_evaluators": list(TRAINING_EVALUATORS[arm]),
        "ce_by_evaluator": {name: float(losses[f"{name}_ce"].detach().cpu()) for name in EVALUATOR_NAMES if name != "E_D"},
        "raw_mse": float(losses["L_rec"].detach().cpu()),
        "l_dist": float(losses["L_dist"].detach().cpu()),
        "l_op": float(losses["L_op"].detach().cpu()),
        "five_plane_power_mean_by_batch": planes.detach().to(torch.float64).mean(dim=0).cpu().tolist(),
        "new_phase_plane_power_mean_by_batch": extra_power.detach().to(torch.float64).mean(dim=0).cpu().tolist(),
        "generator_gradient_norm": capacity._vector_norm(generator_gradient),
        "generator_gradient_rms": capacity._vector_rms(generator_gradient),
        "old_100_plane_gradient_norm": capacity._vector_norm(old_gradient),
        "new_400_plane_gradient_norm": capacity._vector_norm(new_gradient),
        "old_100_plane_displacement_rms": capacity._vector_rms(old_delta),
        "new_400_plane_displacement_rms": capacity._vector_rms(new_delta),
        "generator_displacement_rms": capacity._vector_rms(delta),
        "generator_adam_step": step,
        "all_parameters_finite": bool(torch.isfinite(phase).all().item()),
    }
    _require(record["all_parameters_finite"], "Generator parameters became nonfinite")
    _require(bool(record["new_400_plane_gradient_norm"] > 0.0), "new Generator planes received no gradient")
    _require(all(parameter.grad is None for parameter in parameters.values()), "training left optical parameter gradients populated")
    del planes, losses, gradients, combined
    return record


def _trial_from_source(
    arm: str,
    inputs: Mapping[str, Any],
    base_state: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    evaluators: Mapping[str, nn.Module],
    *,
    label: str,
):
    device = torch.device("cuda:0")
    consumer, optimizer, parameters = _load_arm(inputs, base_state, source_checkpoint, arm, device)
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    runtime_before = capacity._frozen_runtime_state(consumer)
    evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
    records = []
    resources = []
    batch = inputs["images"][:BATCH_SIZE].to(device)
    with torch.no_grad():
        no_grad_result = consumer.runtime.forward_from_image(batch)
        _require(bool(torch.isfinite(no_grad_result.raw_intensity).all().item()), f"{label} no-grad forward is nonfinite")
    del no_grad_result
    for update_index in range(PARENT_UPDATES, PARENT_UPDATES + 3):
        gc.collect()
        torch.cuda.empty_cache()
        resources.append(_gpu_snapshot(device))
        record = _train_step(consumer, optimizer, parameters, evaluators, inputs, update_index, arm=arm)
        records.append(record)
        capacity._assert_disposable_frozen(consumer, parameters, encoder_before, runtime_before, evaluators, evaluator_before, label)
        gc.collect()
        torch.cuda.empty_cache()
    step = int(float(optimizer.state[parameters["generator.phase"]]["step"].item()))
    _require(step == PARENT_UPDATES + 3, f"{label} did not continue Adam from 1000 to 1003")
    _require(not torch.equal(parameters["generator.phase"].detach().cpu(), source_checkpoint["trainable_parameter_state"]["generator.phase"]), f"{label} did not update the Generator")
    _require(all(record["new_400_plane_gradient_norm"] > 0.0 for record in records), f"{label} did not update the added phase planes")
    result = {
        "arm": arm,
        "label": label,
        "updates": len(records),
        "records": records,
        "parameters": capacity._snapshot_parameters(parameters),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "resources": resources,
        "final_step": step,
    }
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _wrong_label_control(
    inputs: Mapping[str, Any],
    base_state: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    evaluators: Mapping[str, nn.Module],
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    images_cpu, labels_cpu, _ = capacity.training_batch(inputs["images"], inputs["labels"], 0)
    differences: dict[str, dict[str, float]] = {}
    references: dict[str, Any] = {}
    for arm in ARM_NAMES:
        consumer, optimizer, parameters = _load_arm(inputs, base_state, source_checkpoint, arm, device)
        images, labels = images_cpu.to(device), labels_cpu.to(device)
        _, true_losses, true_gradients, _ = _large_microbatch_loss_gradients(consumer, parameters, evaluators, images, labels, arm)
        wrong = labels.roll(1)
        _, wrong_losses, wrong_gradients, _ = _large_microbatch_loss_gradients(consumer, parameters, evaluators, images, wrong, arm)
        differences[arm] = {}
        with torch.no_grad():
            result = consumer.runtime.forward_from_image(images[:2])
            view = capacity.canonical.cmos_view(result.raw_intensity)
        for name in TRAINING_EVALUATORS[arm]:
            delta = true_gradients["semantic"][name]["generator.phase"] - wrong_gradients["semantic"][name]["generator.phase"]
            differences[arm][name] = capacity._vector_norm(delta)
            _require(differences[arm][name] > 0.0, f"wrong-label negative control did not change {arm}/{name} gradient")
            logits = capacity._evaluator_logits(evaluators[name], view, name)
            references[f"{arm}_{name}"] = capacity._check_ce_derivative(logits, labels[:2])
        _require(all(parameter.grad is None for parameter in parameters.values()), f"{arm} negative control populated parameter .grad")
        del consumer, optimizer, parameters, result, view, true_losses, true_gradients, wrong_losses, wrong_gradients
        gc.collect()
        torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    return {"passed": True, "source_step": PARENT_UPDATES, "wrong_label_generator_gradient_difference_norm": differences, "independent_ce_derivatives": references}


def _recovery_control(
    inputs: Mapping[str, Any],
    base_state: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    evaluators: Mapping[str, nn.Module],
    population: Mapping[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    consumer, optimizer, parameters = _load_arm(inputs, base_state, source_checkpoint, "A_three_evaluator", device)
    history = [dict(row) for row in source_checkpoint["training_history"]]
    first = _train_step(consumer, optimizer, parameters, evaluators, inputs, PARENT_UPDATES, arm="A_three_evaluator")
    history.append(first)
    payload = _checkpoint_payload(consumer, optimizer, "A_three_evaluator", PARENT_UPDATES + 1, history, population, source_checkpoint)
    path = workspace / "arm_a_step1001.pt"
    torch.save(payload, path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    continuous_records = [_train_step(consumer, optimizer, parameters, evaluators, inputs, idx, arm="A_three_evaluator") for idx in (1001, 1002)]
    continuous = {"parameters": capacity._snapshot_parameters(parameters), "optimizer": _cpu_tree(optimizer.state_dict()), "records": continuous_records}
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()

    consumer, optimizer, parameters = _load_arm(inputs, base_state, source_checkpoint, "A_three_evaluator", device)
    _restore_checkpoint(restored, consumer, optimizer, "A_three_evaluator")
    resumed_records = [_train_step(consumer, optimizer, parameters, evaluators, inputs, idx, arm="A_three_evaluator") for idx in (1001, 1002)]
    resumed = {"parameters": capacity._snapshot_parameters(parameters), "optimizer": _cpu_tree(optimizer.state_dict()), "records": resumed_records}
    passed, detail = capacity._repeatability(continuous, resumed)
    _require(passed, f"continuous and resumed 1001..1003 paths differ: {detail}")
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return {"passed": True, "source_step": PARENT_UPDATES, "saved_step": PARENT_UPDATES + 1, "final_step": PARENT_UPDATES + 3, "continuous_vs_resumed": detail}


def _preflight(
    inputs: Mapping[str, Any],
    population: Mapping[str, Any],
    paths: Mapping[str, Path],
    source_state: Mapping[str, Any],
    source_checkpoints: Mapping[str, Mapping[str, Any]],
    target: Path,
) -> dict[str, Any]:
    _require(torch.cuda.is_available() and torch.cuda.device_count() > 0, "CUDA device 0 is unavailable")
    device = torch.device("cuda:0")
    disk = _disk_probe(target, {arm: Path(path) for arm, path in population["source_checkpoint_paths"].items()})
    cuda_before = _gpu_snapshot(device)
    started = time.perf_counter()
    progress: dict[str, Any] = {}
    rng_before = capacity._rng_state()
    source_checkpoint_before = _cpu_tree(source_checkpoints["A_three_evaluator"])
    evaluators = None
    try:
        progress["large_arm_microbatch_independent_reference"] = {"executed": True, **capacity._verify_large_microbatch_gradient_identity()}
        evaluators = diagnostic.load_evaluators(paths, device)
        evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
        target_control = _target_metrics(evaluators, inputs)
        target_passed = _target_control_passed(target_control)
        progress["four_evaluator_target_positive_control"] = {
            "executed": True,
            "passed": target_passed,
            "metrics": {name: target_control[name]["metrics"]["by_role"]["confirmation"] for name in EVALUATOR_NAMES},
        }
        _require(target_passed, "one or more frozen evaluator target-image controls missed the declared threshold")
        progress["evaluator_gradient_composition_reference"] = {
            "executed": True,
            **_verify_evaluator_gradient_composition(inputs, source_state, source_checkpoints["A_three_evaluator"], evaluators),
        }

        trial_a = _trial_from_source("A_three_evaluator", inputs, source_state, source_checkpoints["A_three_evaluator"], evaluators, label="A_B1000_primary")
        trial_a_repeat = _trial_from_source("A_three_evaluator", inputs, source_state, source_checkpoints["A_three_evaluator"], evaluators, label="A_B1000_repeat")
        repeat_ok, repeat_detail = capacity._repeatability(
            {key: trial_a[key] for key in ("parameters", "optimizer", "records")},
            {key: trial_a_repeat[key] for key in ("parameters", "optimizer", "records")},
        )
        _require(repeat_ok, f"shared-state A/A repeatability failed: {repeat_detail}")
        progress["A_A_from_B1000_repeatability"] = {"executed": True, "passed": True, "detail": repeat_detail}

        trial_b = _trial_from_source("B_two_evaluator", inputs, source_state, source_checkpoints["B_two_evaluator"], evaluators, label="B_B1000")
        progress["three_real_updates_from_shared_1000_step_checkpoint"] = {
            "executed": True,
            "passed": True,
            "A_steps": [row["generator_adam_step"] for row in trial_a["records"]],
            "B_steps": [row["generator_adam_step"] for row in trial_b["records"]],
            "B_added_phase_gradients_nonzero": all(row["new_400_plane_gradient_norm"] > 0.0 for row in trial_b["records"]),
            "frozen_scope_checked_each_step": True,
            "A_resource_observations": trial_a["resources"],
            "B_resource_observations": trial_b["resources"],
        }
        del trial_a, trial_a_repeat, trial_b
        gc.collect()
        torch.cuda.empty_cache()

        progress["wrong_label_ce_negative_control"] = {"executed": True, **_wrong_label_control(inputs, source_state, source_checkpoints["A_three_evaluator"], evaluators)}
        with tempfile.TemporaryDirectory(prefix="lg_accum_recovery_", dir=Path(tempfile.gettempdir())) as temp_name:
            recovery = _recovery_control(inputs, source_state, source_checkpoints["A_three_evaluator"], evaluators, population, Path(temp_name))
        progress["interrupted_resume_equivalence"] = {"executed": True, **recovery}
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], _state_snapshot(model), f"preflight {name} evaluator")
            _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"preflight changed {name} evaluator freeze")
        progress["frozen_evaluator_state"] = {"executed": True, "passed": True, "evaluators": list(EVALUATOR_NAMES)}
        _require(_tree_equal(source_checkpoint_before, source_checkpoints["A_three_evaluator"]), "preflight mutated the shared B1000 checkpoint or Adam source state")
        progress["source_checkpoint_immutability"] = {"executed": True, "passed": True, "source_adam_step": PARENT_UPDATES}
        cuda_after = _gpu_snapshot(device)
        return {
            "schema": PREFLIGHT_SCHEMA,
            "task_id": TASK_ID,
            "status": "READY",
            "device": str(device),
            "numeric_mode": dict(NUMERIC_MODE),
            "propagation_grid": 2048,
            "source_generator_adam_step": PARENT_UPDATES,
            "target_generator_adam_step": TOTAL_UPDATES,
            "additional_updates_per_arm": ADDED_UPDATES,
            "optimizer_state_policy": "both arms clone the same B1000 Generator Adam moments and step",
            "disk_observation": disk,
            "cuda_before_observation": cuda_before,
            "cuda_after_observation": cuda_after,
            "checks": progress,
            "duration_seconds_observation_only": time.perf_counter() - started,
        }
    except Exception as error:
        _write_json_if_absent(
            target / "preflight_diagnostics.json",
            {
                "schema": PREFLIGHT_SCHEMA,
                "task_id": TASK_ID,
                "status": "INVALID_HARNESS",
                "checks": progress,
                "failure_type": type(error).__name__,
                "failure": str(error),
                "duration_seconds_observation_only": time.perf_counter() - started,
            },
        )
        raise
    finally:
        capacity._restore_rng_state(rng_before)
        evaluators = None
        gc.collect()
        torch.cuda.empty_cache()


def _write_json_if_absent(path: Path, payload: Mapping[str, Any]) -> None:
    if not path.exists():
        _write_json_new(path, payload, "cumulative task executed diagnostic evidence")


def _ready_receipt(control: ResearchRunControl, preflight: Mapping[str, Any]) -> PreflightReceipt:
    checks = preflight["checks"]
    checks_passed = bool(checks) and all(
        item.get("executed") is True and item.get("passed") is True for item in checks.values()
    )
    _require(checks_passed, "cannot create READY authorization from missing or failed preflight checks")
    return control.make_ready_receipt(
        correctness={
            "passed": checks_passed,
            "status": "PASS",
            "four_evaluator_target_positive_controls": checks["four_evaluator_target_positive_control"]["passed"],
            "three_vs_two_evaluator_gradient_reference": checks["evaluator_gradient_composition_reference"]["passed"],
            "retained_adam_1000_to_1003": checks["three_real_updates_from_shared_1000_step_checkpoint"]["passed"],
            "wrong_label_ce_negative_control": checks["wrong_label_ce_negative_control"]["passed"],
            "shared_source_checkpoint_immutability": checks["source_checkpoint_immutability"]["passed"],
            "frozen_encoder_ols_and_evaluators": checks["three_real_updates_from_shared_1000_step_checkpoint"]["frozen_scope_checked_each_step"],
        },
        resource={
            "passed": checks["four_evaluator_target_positive_control"]["passed"]
            and checks["three_real_updates_from_shared_1000_step_checkpoint"]["passed"],
            "status": "PASS",
            "cuda_runtime": True,
            "resource_observations": {
                "before": preflight["cuda_before_observation"],
                "after": preflight["cuda_after_observation"],
            },
            "disk_observation": preflight["disk_observation"],
        },
        implementation={
            "passed": checks["A_A_from_B1000_repeatability"]["passed"]
            and checks["interrupted_resume_equivalence"]["passed"]
            and checks["large_arm_microbatch_independent_reference"]["passed"]
            and checks["evaluator_gradient_composition_reference"]["passed"],
            "status": "PASS",
            "selected_engine": "cuda_2048_accumulated_generator_only",
            "optimizer_history_retained": True,
            "A_A_repeatability": checks["A_A_from_B1000_repeatability"]["passed"],
            "interrupted_resume_equivalence": checks["interrupted_resume_equivalence"]["passed"],
            "large_arm_microbatch_reference": checks["large_arm_microbatch_independent_reference"]["passed"],
            "strict_fp32": True,
        },
        selected_engine="cuda_2048_accumulated_generator_only",
        resource_envelope={"cuda": preflight["cuda_after_observation"], "disk": preflight["disk_observation"]},
    )


def _make_identity(population: Mapping[str, Any], source_state: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer) -> ResearchRunIdentity:
    parameters = capacity._parameters(consumer)
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-attempt-01",
        trajectory_id="fixed-large-generator-evaluator-ablation-1000-to-1700",
        semantic_authority_id=SOURCE_TASK_ID,
        data_role="canonical TRAIN fit=200 rows updated; historic=100 and confirmation=300 excluded from gradients",
        config_identity={
            "propagation_grid": 2048,
            "total_updates_per_arm": TOTAL_UPDATES,
            "source_updates_per_arm": PARENT_UPDATES,
            "additional_updates_per_arm": ADDED_UPDATES,
            "logical_batch_size": BATCH_SIZE,
            "large_arm_microbatch_size": capacity.LARGE_GRADIENT_MICROBATCH_SIZE,
            "optimizer_state": "both arms clone exact source Adam moments and step 1000",
            "numeric_mode": dict(NUMERIC_MODE),
            "alpha": capacity.semantic.ALPHA,
            "beta": capacity.semantic.BETA,
            "gamma": capacity.semantic.GAMMA,
            "delta_sic": capacity.semantic.DELTA_SIC,
            "kappa": capacity.semantic.KAPPA,
        },
        model_identity={
            "source_capacity_checkpoint_paths": dict(population["source_checkpoint_paths"]),
            "source_optimizer_steps": {"generator.phase": PARENT_UPDATES},
            "parameter_shapes": {name: list(parameter.shape) for name, parameter in parameters.items()},
            "parameter_counts": {name: parameter.numel() for name, parameter in parameters.items()},
        },
        optimizer_identity={
            "class": type(optimizer).__name__,
            "parameter_names": ["generator.phase"],
            "lr": capacity.GENERATOR_LEARNING_RATE,
            "betas": list(capacity.GENERATOR_BETAS),
            "eps": capacity.GENERATOR_EPS,
            "weight_decay": capacity.GENERATOR_WEIGHT_DECAY,
            "reset": False,
            "source_step": PARENT_UPDATES,
        },
        sampler_identity={
            "fit_rows": population["source_row_indices"][:FIT_ROWS],
            "batch_size": BATCH_SIZE,
            "batch_count": BATCH_COUNT,
            "additional_cycles": ADDED_UPDATES // BATCH_COUNT,
            "confirmation_rows_excluded": population["source_row_indices"][FIT_ROWS + HISTORIC_ROWS :],
        },
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )


def _new_run_control(population: Mapping[str, Any], source_state: Mapping[str, Any], preflight: Mapping[str, Any], inputs: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt]:
    consumer, optimizer, _ = capacity._new_endpoint(inputs, source_state, "B_large", torch.device("cpu"))
    identity = _make_identity(population, source_state, consumer, optimizer)
    del consumer, optimizer
    control = ResearchRunControl(identity=identity, authorization_id=f"{TASK_ID}:authorization:01")
    receipt = _ready_receipt(control, preflight)
    return control, receipt


def _forward_all(consumer: Any, evaluators: Mapping[str, nn.Module], inputs: Mapping[str, Any], arm: str, target: Path, phase: str, started: float) -> dict[str, Any]:
    rows = inputs["images"].shape[0]
    device = consumer.device
    z = torch.empty((rows, 100), dtype=torch.complex64)
    raw = torch.empty((rows, 1, 200, 200), dtype=torch.float32)
    view = torch.empty((rows, 1, 28, 28), dtype=torch.float32)
    target_raw = torch.empty_like(raw)
    power = torch.empty((rows, 5), dtype=torch.float64)
    extra_power = torch.empty((rows, 2), dtype=torch.float64)
    logits = {name: torch.empty((rows, 10), dtype=torch.float32) for name in EVALUATOR_NAMES}
    fields_metadata: dict[str, Any] | None = None
    for start in range(0, rows, BATCH_SIZE):
        stop = min(rows, start + BATCH_SIZE)
        batch = inputs["images"][start:stop].to(device)
        with torch.no_grad():
            result = consumer.runtime.forward_from_image(batch)
            current_view = capacity.canonical.cmos_view(result.raw_intensity)
            z[start:stop].copy_(result.z_ols.coefficients.detach().cpu())
            raw[start:stop].copy_(result.raw_intensity.detach().cpu())
            view[start:stop].copy_(current_view.detach().cpu())
            power[start:stop].copy_(capacity.bogt.contracted_plane_powers(result).detach().cpu())
            _require(result.image_input is not None, "full image path omitted its target image")
            target_raw[start:stop].copy_(result.image_input.slm_intensity.detach().cpu())
            generator = consumer.runtime.generator
            _require(isinstance(generator, capacity.ExpandedGenerator) and generator.last_extra_plane_powers is not None, "large Generator omitted added-plane powers")
            extra_power[start:stop].copy_(generator.last_extra_plane_powers.detach().to(torch.float64).cpu())
            for name in EVALUATOR_NAMES:
                logits[name][start:stop].copy_(capacity._evaluator_logits(evaluators[name], current_view, name).detach().cpu())
            if fields_metadata is None:
                fields_metadata = _fields_metadata(result, "B_large", consumer)
        del result, current_view
        if (start // BATCH_SIZE) % 10 == 0:
            _resource_telemetry(target, device, phase, {"arm": arm, "rows_completed": stop, "rows_total": rows}, started)
    payload: dict[str, Any] = {
        "z": z,
        "raw_cmos": raw,
        "evaluator_view": view,
        "target_slm_intensity": target_raw,
        "power_by_plane": power,
        "source_indices": inputs["source_indices"].clone(),
        "labels": inputs["labels"].clone(),
        **{f"{name.lower()}_logits": value for name, value in logits.items()},
    }
    if extra_power is not None:
        payload["new_phase_plane_power"] = extra_power
    for name, value in payload.items():
        if torch.is_tensor(value):
            diagnostic._finite_tensor(value.real if torch.is_complex(value) else value, f"{arm} {name}")
    _require(fields_metadata is not None, "endpoint omitted physical plane metadata")
    payload["plane_metadata"] = fields_metadata
    for name, model in evaluators.items():
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"measurement changed {name} evaluator isolation")
    return payload


def _fields_metadata(result: Any, arm: str, consumer: Any) -> dict[str, Any]:
    metadata = capacity._all_fields_metadata(result, "B_large", consumer)
    if arm in ARM_NAMES:
        entrance_z = float(result.generator_input_field.plane.z_m)
        metadata["new_phase_planes"] = [
            {
                "plane_id": "lightgen_generator_new_phase_075mm",
                "shape": [2048, 2048],
                "pixel_pitch_m": [2.0e-6, 2.0e-6],
                "z_m": entrance_z + 0.75e-3,
                "phase_support_shape": [400, 400],
                "phase_support_fov_m": [800.0e-6, 800.0e-6],
                "operation": "phase_only_modulation_centered_400_window",
                "outside_window_transmission": 1.0,
                "intermediate_crop": False,
            },
            {
                "plane_id": "lightgen_generator_new_phase_225mm",
                "shape": [2048, 2048],
                "pixel_pitch_m": [2.0e-6, 2.0e-6],
                "z_m": entrance_z + 2.25e-3,
                "phase_support_shape": [400, 400],
                "phase_support_fov_m": [800.0e-6, 800.0e-6],
                "operation": "phase_only_modulation_centered_400_window",
                "outside_window_transmission": 1.0,
                "intermediate_crop": False,
            },
        ]
    return metadata


def _compare_readout(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> dict[str, Any]:
    details = {}
    for key in ("z", "raw_cmos", "evaluator_view", "target_slm_intensity", "power_by_plane", "e_s_logits", "e_i_logits", "e_c_logits", "e_d_logits"):
        left, right = actual[key], expected[key]
        _require(left.shape == right.shape and left.dtype == right.dtype, f"{label} {key} shape/dtype changed")
        _require(torch.allclose(left, right, rtol=RTOL, atol=ATOL), f"{label} {key} failed source replay tolerance")
        details[key] = float((left - right).abs().max().item())
    if "new_phase_plane_power" in expected:
        _require(torch.allclose(actual["new_phase_plane_power"], expected["new_phase_plane_power"], rtol=RTOL, atol=ATOL), f"{label} added-plane powers failed source replay tolerance")
        details["new_phase_plane_power"] = float((actual["new_phase_plane_power"] - expected["new_phase_plane_power"]).abs().max().item())
    for key in ("source_indices", "labels"):
        _require(_tree_equal(actual[key], expected[key]), f"{label} {key} differs from the saved source")
    actual_metadata = dict(actual["plane_metadata"])
    expected_metadata = dict(expected["plane_metadata"])
    actual_new = actual_metadata.pop("new_phase_planes", [])
    expected_new = expected_metadata.pop("new_phase_planes", [])
    _require(_tree_equal(actual_metadata, expected_metadata), f"{label} optical plane metadata differs from the saved source")
    if actual_new:
        _require(len(actual_new) == 2, f"{label} does not record both expanded phase planes")
        entrance_z = float(actual_metadata["generator_input_field"]["plane"]["z_m"])
        expected_positions = [entrance_z + 0.75e-3, entrance_z + 2.25e-3]
        actual_positions = [float(item["z_m"]) for item in actual_new]
        _require(
            all(math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12) for actual, expected in zip(actual_positions, expected_positions, strict=True)),
            f"{label} added phase physical coordinates differ",
        )
    details["source_added_phase_metadata_entries"] = len(expected_new)
    details["measured_added_phase_metadata_entries"] = len(actual_new)
    return {"passed": True, "rtol": RTOL, "atol": ATOL, "max_abs_difference_by_tensor": details}


def _measure_checkpoint(
    arm: str,
    endpoint: str,
    source_checkpoint: Mapping[str, Any],
    task_checkpoint: Mapping[str, Any],
    base_state: Mapping[str, Any],
    inputs: Mapping[str, Any],
    evaluators: Mapping[str, nn.Module],
    target: Path,
    started: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    consumer, optimizer, _parameters = _load_arm(inputs, base_state, source_checkpoint, arm, torch.device("cuda:0"))
    _restore_checkpoint(task_checkpoint, consumer, optimizer, arm)
    parameters_before = capacity._snapshot_parameters(_parameters)
    optimizer_before = _cpu_tree(optimizer.state_dict())
    frozen_before = capacity._frozen_runtime_state(consumer)
    readouts = _forward_all(consumer, evaluators, inputs, arm, target, f"measure_{endpoint}", started)
    _require(_tree_equal(parameters_before, capacity._snapshot_parameters(_parameters)), f"{endpoint} forward changed optical parameters")
    _require(_tree_equal(optimizer_before, _cpu_tree(optimizer.state_dict())), f"{endpoint} forward changed Generator Adam state")
    capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), f"{endpoint} forward changed frozen runtime state")
    capacity._validate_readout(readouts, inputs, endpoint)
    _require(tuple(readouts["new_phase_plane_power"].shape) == (TOTAL_ROWS, 2), f"{endpoint} added-plane power shape changed")
    summary, rows = capacity._save_measurement(target, endpoint, readouts, inputs)
    del consumer, optimizer, _parameters
    gc.collect()
    torch.cuda.empty_cache()
    return readouts, summary, rows


def _ensure_measurement(
    arm: str,
    endpoint: str,
    checkpoint: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    source_readout: Mapping[str, Any] | None,
    base_state: Mapping[str, Any],
    inputs: Mapping[str, Any],
    evaluators: Mapping[str, nn.Module],
    target: Path,
    started: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    path = target / f"readouts_{endpoint}.pt"
    if path.is_file():
        readouts = torch.load(path, map_location="cpu", weights_only=True)
        capacity._validate_readout(readouts, inputs, endpoint)
        summary, rows = capacity._save_measurement(target, endpoint, readouts, inputs)
    else:
        readouts, summary, rows = _measure_checkpoint(arm, endpoint, source_checkpoint, checkpoint, base_state, inputs, evaluators, target, started)
    replay = None
    if source_readout is not None:
        replay = _compare_readout(readouts, source_readout, endpoint)
    return summary, {"readouts": readouts, "row_metrics": rows, "source_replay": replay}


def _final_decision(summaries: Mapping[str, Mapping[str, Any]], *, measurement_valid: bool) -> dict[str, Any]:
    a_metrics = capacity._endpoint_accuracy(summaries["A_step1700"])
    start_metrics = capacity._endpoint_accuracy(summaries["B_step1000"])
    b_metrics = capacity._endpoint_accuracy(summaries["B_step1700"])
    a_correct = capacity._correct_by_class(summaries["A_step1700"])
    start_correct = capacity._correct_by_class(summaries["B_step1000"])
    b_correct = capacity._correct_by_class(summaries["B_step1700"])
    focus_deltas = {
        str(class_id): {
            "vs_A_step1700": b_correct[class_id] - a_correct[class_id],
        "vs_B_step1000": b_correct[class_id] - start_correct[class_id],
        }
        for class_id in FOCUS_CLASSES
    }
    focus_gain = all(
        b_correct[class_id] >= a_correct[class_id] + 3 and b_correct[class_id] >= start_correct[class_id] + 3
        for class_id in FOCUS_CLASSES
    )
    a_macro = float(a_metrics["macro_recall"])
    start_macro = float(start_metrics["macro_recall"])
    b_macro = float(b_metrics["macro_recall"])
    class_guard = all(b_correct[c] >= a_correct[c] - 3 and b_correct[c] >= start_correct[c] - 3 for c in range(10))
    a_raw = float(summaries["A_step1700"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"])
    start_raw = float(summaries["B_step1000"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"])
    b_summary = summaries["B_step1700"]
    b_confirmation = b_summary["image_metrics_by_role"]["confirmation"]
    b_raw = float(b_confirmation["overall"]["raw_mse"])
    clip_guard = b_confirmation["overall"]["raw_fraction_above_one"] <= 0.01 and all(
        b_confirmation[str(c)]["raw_fraction_above_one"] <= 0.01 for c in FOCUS_CLASSES
    )
    class_8_guard = b_correct[8] >= 20
    image_guard = (
        b_raw <= 1.05 * a_raw
        and b_raw <= 1.05 * start_raw
        and clip_guard
        and not b_summary["has_zero_image"]
        and not b_summary["has_constant_image"]
        and b_summary["all_finite"]
    )
    conditions = {
        "class_1_and_7_each_gain_at_least_3_over_A1700_and_B1000": focus_gain,
        "macro_recall_at_least_A1700_plus_0_05_and_not_below_B1000": b_macro >= a_macro + 0.05 and b_macro >= start_macro,
        "class_8_at_least_20_of_30": class_8_guard,
        "no_class_loses_more_than_3_vs_A1700_or_B1000": class_guard,
        "raw_mse_and_clipping_image_guard": image_guard,
        "training_population_freeze_and_measurement_valid": measurement_valid,
    }
    if not measurement_valid:
        classification, status, recommendation = "INVALID_HARNESS", "INVALID_HARNESS", None
    elif all(conditions.values()):
        classification, status = "EVALUATOR_ABLATION_GAIN_SUPPORTED", "VALID_COMPLETED"
        recommendation = "CONSIDER_TWO_EVALUATOR_COMPOSITION_IN_A_SEPARATE_CONTROLLED_STUDY"
    elif any(b_correct[c] > a_correct[c] and b_correct[c] > start_correct[c] for c in FOCUS_CLASSES):
        classification, status, recommendation = "MIXED_OR_INSUFFICIENT", "VALID_FAILED", "NONE"
    else:
        classification, status, recommendation = "NOT_SUPPORTED_WITHIN_700_UPDATES", "VALID_FAILED", "NONE"
    return {
        "status": status,
        "classification": classification,
        "conditions": conditions,
        "focus_class_correct_deltas": focus_deltas,
        "macro_recall": {"A_step1700": a_macro, "B_step1000": start_macro, "B_step1700": b_macro},
        "confirmation_raw_mse": {"A_step1700": a_raw, "B_step1000": start_raw, "B_step1700": b_raw},
        "B_step1700_clipping": {
            "overall": b_confirmation["overall"]["raw_fraction_above_one"],
            "class_1": b_confirmation["1"]["raw_fraction_above_one"],
            "class_7": b_confirmation["7"]["raw_fraction_above_one"],
        },
        "class_correct": {"A_step1700": a_correct, "B_step1000": start_correct, "B_step1700": b_correct},
        "training_recommendation": recommendation,
        "next_action": "NONE",
        "stop": True,
    }


def _phase_class_render(view: torch.Tensor, scale: int = 4) -> Image.Image:
    image = view.detach().to(torch.float32).cpu().squeeze().clamp(0.0, 1.0)
    array = torch.round((1.0 - image) * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(array, mode="L").resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)


def _comparison_pages(target: Path, inputs: Mapping[str, Any], readouts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    names = ("B_step1000", "A_step1700", "B_step1700")
    headers = ("Target", "B1000", "A1700", "B1700")
    positions_by_name = {
        "classes_1_7_and_0_4_8": torch.nonzero(torch.isin(inputs["labels"], torch.tensor(DISPLAY_CLASSES)), as_tuple=False).flatten().tolist(),
        "class_9": torch.nonzero(inputs["labels"] == 9, as_tuple=False).flatten().tolist(),
    }
    page_names: dict[str, list[str]] = {}
    cell = 112
    row_height = cell + 38
    for group_name, positions in positions_by_name.items():
        page_names[group_name] = []
        for page_start in range(0, len(positions), 30):
            page_positions = positions[page_start : page_start + 30]
            canvas = Image.new("RGB", (800, 58 + len(page_positions) * row_height), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"{group_name}; rows {page_start + 1}-{page_start + len(page_positions)} of {len(positions)}; fixed [0,1]", fill="black")
            xs = (345, 455, 565, 675)
            for x, title in zip(xs, headers, strict=True):
                draw.text((x, 35), title, fill="black")
            for index, position in enumerate(page_positions):
                y = 55 + index * row_height
                role = "fit" if position < FIT_ROWS else ("history" if position < FIT_ROWS + HISTORIC_ROWS else "confirmation")
                draw.text((8, y + 8), f"row={int(inputs['source_indices'][position])} y={int(inputs['labels'][position])} {role}", fill="black")
                panels = [inputs["images"][position], *[readouts[name]["evaluator_view"][position] for name in names]]
                for col, panel in enumerate(panels):
                    canvas.paste(_phase_class_render(panel), (xs[col], y))
                for col, name in enumerate(names, start=1):
                    pred = int(readouts[name]["e_d_logits"][position].to(torch.float64).argmax().item())
                    draw.text((xs[col], y + cell + 2), f"E_D={pred}", fill="black")
            filename = f"comparison_{group_name}_page_{page_start // 30 + 1:02d}.png"
            path = target / filename
            if path.exists():
                with Image.open(path) as previous:
                    _require(previous.size == canvas.size and previous.convert("RGB").tobytes() == canvas.tobytes(), f"saved comparison page differs: {filename}")
            else:
                canvas.save(path, optimize=True)
            page_names[group_name].append(filename)
    return {
        "pages": page_names,
        "row_counts": {key: len(value) for key, value in positions_by_name.items()},
        "panel_order": ["target", *names],
        "fixed_display_range": [0.0, 1.0],
        "no_per_image_rescaling": True,
        "predictions": "E_D prediction displayed below each generated image",
    }


def _prepare_output(root: Path, target: Path, *, allow_ready: bool) -> bool:
    output_root = (root / "outputs").resolve()
    target = target.resolve()
    _require(target != output_root and output_root in target.parents, "output must be a named child of outputs/")
    parts = target.relative_to(output_root).parts
    _require(len(parts) == 3 and re.fullmatch(r"\d{8}_[A-Za-z0-9][A-Za-z0-9_-]*", parts[0]) and parts[1] == "runs", "output must be outputs/YYYYMMDD_<version>/runs/<attempt>")
    _require(not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in parts), "protected/temporary output namespace is forbidden")
    if not target.exists():
        target.mkdir(parents=True, exist_ok=False)
        return False
    _require(allow_ready and target.is_dir(), "output exists; use an explicit task checkpoint for resume")
    allowed = {"input_population.json", "preflight.json"}
    _require({path.name for path in target.iterdir()} == allowed, "existing output is not an untouched READY staging directory")
    return True


def _store_ready(target: Path, population: Mapping[str, Any], preflight: Mapping[str, Any], receipt: PreflightReceipt) -> None:
    _write_json_if_absent(target / "input_population.json", population)
    _write_json_if_absent(target / "preflight.json", {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": preflight})


def _read_ready(target: Path, population: Mapping[str, Any]) -> tuple[dict[str, Any], PreflightReceipt]:
    saved_population = json.loads((target / "input_population.json").read_text(encoding="utf-8"))
    _require(saved_population == dict(population), "READY population, parent paths, rows, roles, or labels changed")
    saved = json.loads((target / "preflight.json").read_text(encoding="utf-8"))
    _require(saved.get("task_id") == TASK_ID and saved.get("status") == "READY", "task READY evidence is missing")
    preflight = saved.get("preflight")
    checks = preflight.get("checks", {}) if isinstance(preflight, Mapping) else {}
    _require(
        isinstance(preflight, Mapping)
        and preflight.get("status") == "READY"
        and bool(checks)
        and all(item.get("executed") is True and item.get("passed") is True for item in checks.values()),
        "READY receipt does not contain executed passing checks",
    )
    return dict(preflight), PreflightReceipt.from_mapping(saved["receipt"])


def _measure_source_endpoint(
    arm: str,
    endpoint: str,
    inputs: Mapping[str, Any],
    base_state: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    evaluators: Mapping[str, nn.Module],
    target: Path,
    started: float,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    consumer, optimizer, parameters = _load_arm(inputs, base_state, source_checkpoint, arm, torch.device("cuda:0"))
    parameters_before = capacity._snapshot_parameters(parameters)
    optimizer_before = _cpu_tree(optimizer.state_dict())
    frozen_before = capacity._frozen_runtime_state(consumer)
    readouts = _forward_all(consumer, evaluators, inputs, arm, target, f"source_replay_{endpoint}", started)
    _require(_tree_equal(parameters_before, capacity._snapshot_parameters(parameters)), f"{endpoint} replay changed optical parameters")
    _require(_tree_equal(optimizer_before, _cpu_tree(optimizer.state_dict())), f"{endpoint} replay changed Generator Adam state")
    capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), f"{endpoint} replay changed frozen runtime state")
    capacity._validate_readout(readouts, inputs, endpoint)
    summary, _ = capacity._save_measurement(target, endpoint, readouts, inputs)
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return readouts, summary


def _train_arm(
    arm: str,
    inputs: Mapping[str, Any],
    base_state: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    evaluators: Mapping[str, nn.Module],
    target: Path,
    population: Mapping[str, Any],
    *,
    start_from: Mapping[str, Any],
    started: float,
) -> dict[str, Any]:
    consumer, optimizer, parameters = _load_arm(inputs, base_state, source_checkpoint, arm, torch.device("cuda:0"))
    _restore_checkpoint(start_from, consumer, optimizer, arm)
    completed = int(start_from["completed_updates"])
    history = [dict(record) for record in start_from["training_history"]]
    _require(len(history) == completed, f"{arm} continuation history/count differs")
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    frozen_before = capacity._frozen_runtime_state(consumer)
    evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
    signal_requested = False
    previous_handler = signal.getsignal(signal.SIGINT)

    def request_stop(_signal: int, _frame: Any) -> None:
        nonlocal signal_requested
        signal_requested = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        while completed < TOTAL_UPDATES:
            record = _train_step(consumer, optimizer, parameters, evaluators, inputs, completed, arm=arm)
            completed += 1
            history.append(record)
            if completed % CHECKPOINT_INTERVAL == 0 or signal_requested:
                checkpoint_path = _save_checkpoint(target, consumer, optimizer, arm, completed, history, population, source_checkpoint)
                _resource_telemetry(target, torch.device("cuda:0"), f"{arm}_checkpoint", {"completed_updates": completed, "additional_updates": completed - PARENT_UPDATES, "checkpoint": checkpoint_path.name}, started)
                _require(torch.equal(parameters["encoder.phase"].detach().cpu(), encoder_before), f"{arm} changed Encoder parameters")
                capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), f"{arm} OLS/geometry/scientific buffers")
                for name, model in evaluators.items():
                    capacity._assert_state_equal(evaluator_before[name], _state_snapshot(model), f"{arm} evaluator {name}")
                    _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"{name} evaluator isolation changed")
                gc.collect()
                torch.cuda.empty_cache()
            if signal_requested:
                raise TrainingInterrupted(f"saved explicit cumulative checkpoint {checkpoint_path.name}")
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    _require(completed == TOTAL_UPDATES and len(history) == TOTAL_UPDATES, f"{arm} did not finish exactly at cumulative update 1700")
    _require(torch.equal(parameters["encoder.phase"].detach().cpu(), encoder_before), f"{arm} changed frozen Encoder at completion")
    capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), f"{arm} frozen runtime at completion")
    for name, model in evaluators.items():
        capacity._assert_state_equal(evaluator_before[name], _state_snapshot(model), f"{arm} evaluator {name} at completion")
    cp_path = target / _checkpoint_name(arm, TOTAL_UPDATES)
    if not cp_path.exists():
        _save_checkpoint(target, consumer, optimizer, arm, TOTAL_UPDATES, history, population, source_checkpoint)
    history_payload = {"task_id": TASK_ID, "arm": arm, "parent_updates": PARENT_UPDATES, "additional_updates": ADDED_UPDATES, "total_updates": TOTAL_UPDATES, "records": history}
    history_path = target / f"training_history_{arm}.json"
    if history_path.exists():
        _require(json.loads(history_path.read_text(encoding="utf-8")) == history_payload, f"saved {arm} cumulative history differs")
    else:
        _write_json_new(history_path, history_payload, f"{arm} source 1000-step and added 700-step training history")
    result = torch.load(cp_path, map_location="cpu", weights_only=True)
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _ensure_history_artifact(target: Path, arm: str, checkpoint: Mapping[str, Any]) -> None:
    payload = {
        "task_id": TASK_ID,
        "arm": arm,
        "parent_updates": PARENT_UPDATES,
        "additional_updates": ADDED_UPDATES,
        "total_updates": TOTAL_UPDATES,
        "records": checkpoint["training_history"],
    }
    path = target / f"training_history_{arm}.json"
    if path.exists():
        _require(json.loads(path.read_text(encoding="utf-8")) == payload, f"{arm} completed history differs from its final checkpoint")
    else:
        _write_json_new(path, payload, f"{arm} complete cumulative training history")


def _ensure_task_start_checkpoints(
    target: Path,
    inputs: Mapping[str, Any],
    base_state: Mapping[str, Any],
    population: Mapping[str, Any],
    source_checkpoints: Mapping[str, Mapping[str, Any]],
) -> None:
    for arm in ARM_NAMES:
        consumer, optimizer, _parameters = _load_arm(inputs, base_state, source_checkpoints[arm], arm, torch.device("cuda:0"))
        path = target / _checkpoint_name(arm, PARENT_UPDATES)
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=True)
            source = source_checkpoints[arm]
            _require(existing.get("schema") == CHECKPOINT_SCHEMA and existing.get("task_id") == TASK_ID and existing.get("arm") == arm and existing.get("completed_updates") == PARENT_UPDATES, f"imported {arm} checkpoint is invalid")
            _require(existing.get("population") == dict(population) and existing.get("source_capacity_checkpoint") == population["source_checkpoint_paths"]["shared_B_step1000"], f"imported {arm} checkpoint input lineage differs")
            _require(_tree_equal(existing.get("trainable_parameter_state"), source["trainable_parameter_state"]), f"imported {arm} starting model differs from shared B1000")
            _require(_tree_equal(existing.get("optimizer_state"), source["optimizer_state"]), f"imported {arm} starting Adam differs from shared B1000")
            _require(_tree_equal(existing.get("training_history"), source["training_history"]), f"imported {arm} source history differs")
        else:
            _save_checkpoint(target, consumer, optimizer, arm, PARENT_UPDATES, source_checkpoints[arm]["training_history"], population, source_checkpoints[arm])
        del consumer, optimizer, _parameters
        gc.collect()
        torch.cuda.empty_cache()


def _source_replay_report(
    target: Path,
    inputs: Mapping[str, Any],
    base_state: Mapping[str, Any],
    source_checkpoints: Mapping[str, Mapping[str, Any]],
    parent_readouts: Mapping[str, Mapping[str, Any]],
    evaluators: Mapping[str, nn.Module],
    started: float,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], dict[str, Any]]:
    readouts: dict[str, Mapping[str, Any]] = {}
    summaries: dict[str, Mapping[str, Any]] = {}
    replay: dict[str, Any] = {}
    endpoint = "B_step1000"
    saved_path = target / f"readouts_{endpoint}.pt"
    if saved_path.is_file():
        measured = torch.load(saved_path, map_location="cpu", weights_only=True)
        capacity._validate_readout(measured, inputs, endpoint)
        summary, _ = capacity._save_measurement(target, endpoint, measured, inputs)
    else:
        measured, summary = _measure_source_endpoint(
            "B_two_evaluator", endpoint, inputs, base_state, source_checkpoints["B_two_evaluator"], evaluators, target, started
        )
    replay[endpoint] = _compare_readout(measured, parent_readouts[endpoint], endpoint)
    readouts[endpoint] = measured
    summaries[endpoint] = summary
    return readouts, summaries, replay


def _paired_metrics(readouts: Mapping[str, Mapping[str, Any]], labels: torch.Tensor) -> dict[str, Any]:
    return {
        "B1000_to_A1700": capacity._paired_correctness(readouts["B_step1000"], readouts["A_step1700"], labels),
        "B1000_to_B1700": capacity._paired_correctness(readouts["B_step1000"], readouts["B_step1700"], labels),
        "A1700_to_B1700": capacity._paired_correctness(readouts["A_step1700"], readouts["B_step1700"], labels),
    }


def _write_failure(target: Path, phase: str, error: BaseException, started: float, progress: Mapping[str, Any]) -> None:
    if not target.is_dir():
        return
    path = target / "invalid_harness.json"
    if not path.exists():
        _write_json_new(
            path,
            {
                "schema": REPORT_SCHEMA,
                "task_id": TASK_ID,
                "status": "INVALID_HARNESS",
                "classification": None,
                "phase": phase,
                "failure_type": type(error).__name__,
                "failure": str(error),
                "executed_checks": dict(progress),
                "duration_seconds_observation_only": time.perf_counter() - started,
                "scientific_conclusion": None,
                "next_action": "NONE",
            },
            "invalid harness incident without scientific classification",
        )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accumulation-dir", type=Path, default=DEFAULT_CAPACITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="explicit cumulative checkpoint required with --resume")
    return parser.parse_args()


def execute(arguments: argparse.Namespace | None = None) -> dict[str, Any]:
    arguments = arguments or _arguments()
    root = _root()
    capacity_dir = _resolve(root, arguments.accumulation_dir)
    target = _resolve(root, arguments.output_dir)
    started = time.perf_counter()
    _require(bool(arguments.resume) == (arguments.checkpoint is not None), "--resume requires an explicit --checkpoint path")
    _require(not (arguments.preflight_only and arguments.resume), "--preflight-only cannot be combined with --resume")
    progress: dict[str, Any] = {}
    phase = "input_validation"
    run_control: ResearchRunControl | None = None
    try:
        phase = "source_lineage_and_population"
        (
            inputs,
            population,
            paths,
            base_state,
            _source_population,
            source_report,
            source_checkpoints,
            parent_readouts,
            parent_summaries,
        ) = _load_parent_inputs(root, capacity_dir)
        _require(capacity_dir.is_dir(), f"the explicit source capacity directory is missing: {capacity_dir}")
        shared_source_checkpoint = source_checkpoints["A_three_evaluator"]
        shared_source_checkpoint_snapshot = _cpu_tree(shared_source_checkpoint)
        phase = "preflight_or_resume"
        if arguments.resume:
            _require(target.is_dir(), "resume output directory is missing")
            saved_population = json.loads((target / "input_population.json").read_text(encoding="utf-8"))
            _require(saved_population == population, "resume population or explicit source paths changed")
            preflight, receipt = _read_ready(target, population)
            selected_path = _resolve(root, arguments.checkpoint)
            selected = _read_checkpoint(selected_path, target, population, source_checkpoints)
            terminal = _completed_resume_report(target, selected)
            if terminal is not None:
                return terminal
        else:
            staged = _prepare_output(root, target, allow_ready=not arguments.preflight_only)
            if staged:
                preflight, receipt = _read_ready(target, population)
                selected = None
            else:
                with capacity.strict_fp32_mode():
                    preflight = _preflight(inputs, population, paths, base_state, source_checkpoints, target)
                progress["preflight_READY"] = True
                run_control, receipt = _new_run_control(population, base_state, preflight, inputs)
                run_control.mark_preflight_ready(receipt)
                _store_ready(target, population, preflight, receipt)
                selected = None
            if arguments.preflight_only:
                return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "training_updates": 0, "next_action": "NONE"}
        if run_control is None:
            run_control, _ = _new_run_control(population, base_state, preflight, inputs)
            run_control.mark_preflight_ready(receipt)
        run_control.start_execution()
        # The READY staging files already live in target. Reopen that exact
        # same attempt directory after consuming its receipt; model recovery
        # remains separately gated by the explicit checkpoint above.
        run_control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt, resume=True)

        device = torch.device("cuda:0")
        phase = "source_B1000_replay_and_training"
        with capacity.strict_fp32_mode() as numeric_mode:
            evaluators = diagnostic.load_evaluators(paths, device)
            evaluator_before = {name: _state_snapshot(model) for name, model in evaluators.items()}
            target_control = _target_metrics(evaluators, inputs)
            _require(_target_control_passed(target_control), "target positive controls changed after READY")
            target_control_payload = {name: {"logits": value["logits"], "metrics": value["metrics"]} for name, value in target_control.items()}
            target_control_path = target / "target_image_evaluator_positive_controls.pt"
            if target_control_path.exists():
                _require(_tree_equal(torch.load(target_control_path, map_location="cpu", weights_only=True), target_control_payload), "saved target controls changed during resume")
            else:
                _write_torch_new(target_control_path, target_control_payload, "four frozen evaluator target-image controls")

            _ensure_task_start_checkpoints(target, inputs, base_state, population, source_checkpoints)
            readouts, summaries, source_replays = _source_replay_report(
                target, inputs, base_state, source_checkpoints, parent_readouts, evaluators, started
            )
            progress["B1000_source_replay"] = {
                "executed": True,
                "passed": bool(source_replays["B_step1000"]["passed"]),
                **source_replays,
            }

            selected_arm = selected["arm"] if arguments.resume else None
            if arguments.resume and selected_arm == "B_two_evaluator":
                a_task_checkpoint = torch.load(target / _checkpoint_name("A_three_evaluator", TOTAL_UPDATES), map_location="cpu", weights_only=True)
                _require(a_task_checkpoint.get("completed_updates") == TOTAL_UPDATES, "A final checkpoint is missing before B continuation")
            else:
                if arguments.resume and selected_arm == "A_three_evaluator":
                    a_start = selected
                else:
                    a_start = torch.load(target / _checkpoint_name("A_three_evaluator", PARENT_UPDATES), map_location="cpu", weights_only=True)
                a_task_checkpoint = _train_arm(
                    "A_three_evaluator", inputs, base_state, source_checkpoints["A_three_evaluator"], evaluators, target, population,
                    start_from=a_start, started=started,
                )
            a_summary, a_data = _ensure_measurement(
                "A_three_evaluator", "A_step1700", a_task_checkpoint, source_checkpoints["A_three_evaluator"], None,
                base_state, inputs, evaluators, target, started,
            )
            readouts["A_step1700"] = a_data["readouts"]
            summaries["A_step1700"] = a_summary

            if selected_arm == "B_two_evaluator":
                b_start = selected
            else:
                b_start = torch.load(target / _checkpoint_name("B_two_evaluator", PARENT_UPDATES), map_location="cpu", weights_only=True)
            if selected_arm == "B_two_evaluator" and selected["completed_updates"] == TOTAL_UPDATES:
                b_task_checkpoint = selected
            else:
                b_task_checkpoint = _train_arm(
                    "B_two_evaluator", inputs, base_state, source_checkpoints["B_two_evaluator"], evaluators, target, population,
                    start_from=b_start, started=started,
                )
            _ensure_history_artifact(target, "A_three_evaluator", a_task_checkpoint)
            _ensure_history_artifact(target, "B_two_evaluator", b_task_checkpoint)
            b_summary, b_data = _ensure_measurement(
                "B_two_evaluator", "B_step1700", b_task_checkpoint, source_checkpoints["B_two_evaluator"], None,
                base_state, inputs, evaluators, target, started,
            )
            readouts["B_step1700"] = b_data["readouts"]
            summaries["B_step1700"] = b_summary

            for name, model in evaluators.items():
                capacity._assert_state_equal(evaluator_before[name], _state_snapshot(model), f"final {name} evaluator")
                _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"final {name} evaluator is not frozen")
            _require(_tree_equal(shared_source_checkpoint_snapshot, shared_source_checkpoint), "formal run mutated the shared B1000 model/Adam checkpoint")
            _require(set(readouts) >= {"B_step1000", "A_step1700", "B_step1700"}, "one or more required readouts are missing")
            final = _final_decision(summaries, measurement_valid=True)
            transitions = _paired_metrics(readouts, inputs["labels"])
            montage = _comparison_pages(target, inputs, readouts)
            incremental = {
                "A": {
                    "B1000_to_A1700_correct_by_class": {
                        "from": capacity._correct_by_class(parent_summaries["B_step1000"]),
                        "to": capacity._correct_by_class(summaries["A_step1700"]),
                    },
                    "confirmation_macro_recall_change": float(capacity._endpoint_accuracy(summaries["A_step1700"])["macro_recall"] - capacity._endpoint_accuracy(parent_summaries["B_step1000"])["macro_recall"]),
                    "raw_mse_change": float(summaries["A_step1700"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"] - parent_summaries["B_step1000"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"]),
                },
                "B": {
                    "B1000_to_B1700_correct_by_class": {
                        "from": capacity._correct_by_class(parent_summaries["B_step1000"]),
                        "to": capacity._correct_by_class(summaries["B_step1700"]),
                    },
                    "confirmation_macro_recall_change": float(capacity._endpoint_accuracy(summaries["B_step1700"])["macro_recall"] - capacity._endpoint_accuracy(parent_summaries["B_step1000"])["macro_recall"]),
                    "raw_mse_change": float(summaries["B_step1700"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"] - parent_summaries["B_step1000"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"]),
                    "class_8_correct_change": capacity._correct_by_class(summaries["B_step1700"])[8] - capacity._correct_by_class(parent_summaries["B_step1000"])[8],
                },
            }
            report = {
                "schema": REPORT_SCHEMA,
                "task_id": TASK_ID,
                **final,
                "scientific_question": "whether removing E_S from the semantic direction of the fixed large Generator improves E_D class 1/7 correctness after 700 additional updates while recovering class 8 and preserving other classes",
                "population": population,
                "source": {
                    "accumulation_dir": capacity_dir.as_posix(),
                    "task_id": source_report["task_id"],
                    "status": source_report["status"],
                    "classification": source_report["classification"],
                    "shared_B1000_checkpoint": population["source_checkpoint_paths"]["shared_B_step1000"],
                    "source_generator_adam_step": PARENT_UPDATES,
                },
                "training": {
                    "source_updates_per_arm": PARENT_UPDATES,
                    "additional_updates_per_arm": ADDED_UPDATES,
                    "cumulative_updates_per_arm": TOTAL_UPDATES,
                    "batch_size": BATCH_SIZE,
                    "batch_count": BATCH_COUNT,
                    "additional_cycles": ADDED_UPDATES // BATCH_COUNT,
                    "parameter_scope": "Generator phase only; Encoder/OLS/scientific buffers/evaluators frozen",
                    "optimizer_state": "both arms inherit the same B1000 Generator Adam moments; next update 1001, final step 1700",
                    "optimizer": {"lr": capacity.GENERATOR_LEARNING_RATE, "betas": list(capacity.GENERATOR_BETAS), "eps": capacity.GENERATOR_EPS, "weight_decay": capacity.GENERATOR_WEIGHT_DECAY},
                    "large_arm_microbatch_rows": capacity.LARGE_GRADIENT_MICROBATCH_SIZE,
                    "large_arm_full_batch_OLS_statistics_preserved": True,
                    "numeric_mode": numeric_mode,
                    "training_objective": "true-class CE from E_S/E_I/E_C in A and E_I/E_C in B; raw reconstruction MSE and BOGT unchanged; per-evaluator joint Encoder/Generator L2 normalization, equal-weight sum and joint RMS normalization",
                    "E_S_in_B": "forward-only diagnostic and logging; contributes no gradient",
                    "E_D": "forward-only confirmation evaluator in both arms",
                },
                "source_B1000_replay": source_replays,
                "shared_source_checkpoint_unchanged": True,
                "endpoint_metrics": summaries,
                "incremental_B1000_to_1700": incremental,
                "paired_correctness_transitions_confirmation": transitions,
                "readout_artifacts": {name: f"readouts_{name}.pt" for name in readouts},
                "row_metric_artifacts": {name: f"row_metrics_{name}.pt" for name in readouts},
                "metric_artifacts": {name: f"metrics_{name}.json" for name in summaries},
                "checkpoint_artifacts": sorted(path.name for path in target.glob("arm_*_step*.pt")),
                "training_history_artifacts": {arm: f"training_history_{arm}.json" for arm in ARM_NAMES},
                "resource_observation_artifact": RESOURCE_NAME,
                "comparison_images": montage,
                "preflight_artifact": "preflight.json",
                "elapsed_seconds_observation_only": time.perf_counter() - started,
                "next_action": "NONE",
                "stop": True,
            }
            if run_control is not None:
                if report["status"] == "VALID_COMPLETED":
                    run_control.complete()
                else:
                    run_control.valid_failed()
                report["research_run_state"] = run_control.state
            _write_json_new(target / "report.json", report, "final 700-update three-versus-two evaluator comparison decision")
            return report
    except TrainingInterrupted as error:
        incident = {
            "task_id": TASK_ID,
            "status": "INTERRUPTED",
            "classification": None,
            "failure": str(error),
            "next_action": "resume only from an explicit task checkpoint",
        }
        _write_json_if_absent(target / f"interrupted_{int(time.time())}.json", incident)
        return incident
    except Exception as error:
        _write_failure(target, phase, error, started, progress)
        raise


def main() -> None:
    result = execute()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
