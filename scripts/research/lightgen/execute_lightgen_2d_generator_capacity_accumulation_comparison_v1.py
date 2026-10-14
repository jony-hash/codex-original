"""Continue the fixed LightGen small/large Generator comparison from step 300 to 1000."""

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


TASK_ID = "M0_GENERATOR_CAPACITY_ACCUMULATION_COMPARISON_V1"
REPORT_SCHEMA = "m0_generator_capacity_accumulation_comparison_report.v1"
PREFLIGHT_SCHEMA = "m0_generator_capacity_accumulation_preflight.v1"
CHECKPOINT_SCHEMA = "m0_generator_capacity_accumulation_checkpoint.v1"
SOURCE_TASK_ID = "M0_GENERATOR_LARGE_CAPACITY_QUICK_COMPARISON_V1"
DEFAULT_CAPACITY = Path("outputs/20261007_m0_generator_large_capacity_quick_comparison_v1/runs/attempt_03")
DEFAULT_OUTPUT = Path("outputs/20261007_m0_generator_capacity_accumulation_comparison_v1/runs/attempt_01")
PARENT_UPDATES = 300
TOTAL_UPDATES = 1000
ADDED_UPDATES = TOTAL_UPDATES - PARENT_UPDATES
FIT_ROWS = capacity.FIT_ROWS
HISTORIC_ROWS = capacity.HISTORIC_ROWS
CONFIRMATION_ROWS = capacity.CONFIRMATION_ROWS
TOTAL_ROWS = capacity.TOTAL_ROWS
BATCH_COUNT = capacity.BATCH_COUNT
BATCH_SIZE = capacity.BATCH_SIZE
ARM_NAMES = capacity.ARM_NAMES
EVALUATOR_NAMES = capacity.EVALUATOR_NAMES
TRAINING_EVALUATORS = capacity.TRAINING_EVALUATORS
FOCUS_CLASSES = capacity.FOCUS_CLASSES
DISPLAY_CLASSES = capacity.DISPLAY_CLASSES
RTOL, ATOL = capacity.RTOL, capacity.ATOL
CHECKPOINT_INTERVAL = 100
CHECKPOINT_PATTERN = re.compile(r"arm_(a|b)_step(\d{4})\.pt$")
RESOURCE_NAME = "resource_telemetry_accumulation.jsonl"
NUMERIC_MODE = {
    "dtype": "float32",
    "autocast": False,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "float32_matmul_precision": "highest",
}


class AccumulationRunError(RuntimeError):
    """Invalid task input or violated training/recovery invariant."""


class TrainingInterrupted(RuntimeError):
    """An interrupt was recorded at a safe optimizer boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AccumulationRunError(message)


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
    _require(population_path.is_file() and report_path.is_file(), "the explicit 300-step capacity attempt is incomplete")
    source_population = json.loads(population_path.read_text(encoding="utf-8"))
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(source_population.get("task_id") == SOURCE_TASK_ID, "capacity input population is from another task")
    _require(source_report.get("task_id") == SOURCE_TASK_ID, "capacity report is from another task")
    _require(
        source_report.get("status") == "VALID_FAILED"
        and source_report.get("classification") == "MIXED_OR_INSUFFICIENT",
        "the named 300-step source attempt is not the completed mixed/insufficient result",
    )
    _require(source_population.get("updates_per_arm") == PARENT_UPDATES, "source population does not declare 300 updates per arm")
    parent_value = source_population.get("parent_dir")
    _require(isinstance(parent_value, str) and parent_value, "the source population has no explicit semantic-target parent")
    semantic_parent = _resolve(root, parent_value)
    inputs, expected_source_population, paths, source_state = capacity._load_inputs(root, semantic_parent)
    _require(source_population == expected_source_population, "capacity population rows, roles, labels, or paths differ from its declared parent")

    source_checkpoints: dict[str, dict[str, Any]] = {}
    for arm, letter in (("A_small", "a"), ("B_large", "b")):
        checkpoint_path = capacity_dir / f"arm_{letter}_step0300.pt"
        _require(checkpoint_path.is_file(), f"the explicit {arm} 300-step checkpoint is missing")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _validate_source_checkpoint(checkpoint, source_population, arm)
        source_checkpoints[arm] = dict(checkpoint)
        expected_history_path = capacity_dir / f"training_history_{arm}.json"
        _require(expected_history_path.is_file(), f"the source {arm} training history is missing")
        history_payload = json.loads(expected_history_path.read_text(encoding="utf-8"))
        _require(history_payload.get("task_id") == SOURCE_TASK_ID and history_payload.get("arm") == arm, f"the source {arm} history identity differs")
        _require(history_payload.get("updates") == PARENT_UPDATES, f"the source {arm} history does not contain 300 updates")
        _require(_tree_equal(history_payload.get("records"), checkpoint["training_history"]), f"the source {arm} checkpoint and history file differ")
        _validate_history_rows(checkpoint["training_history"], inputs, arm)

    a_state = source_checkpoints["A_small"]["trainable_parameter_state"]
    b_state = source_checkpoints["B_large"]["trainable_parameter_state"]
    _require(torch.equal(a_state["encoder.phase"], b_state["encoder.phase"]), "A/B source Encoders are not the same frozen state")
    _require(torch.count_nonzero(b_state["generator.phase"][capacity.PHASE_OLD_COUNT:]).item() > 0, "the large source has no learned added phase values")
    _require(_tree_equal(source_checkpoints["A_small"]["frozen_runtime_state"], source_checkpoints["B_large"]["frozen_runtime_state"]), "A/B frozen OLS/runtime buffers differ")
    for arm in ARM_NAMES:
        checkpoint = source_checkpoints[arm]
        _require(_tree_equal(checkpoint["references"], source_state["references"]), f"{arm} references differ from the parent checkpoint")
        _require(_tree_equal(checkpoint["weights"], source_state["weights"]), f"{arm} BOGT weights differ from the parent checkpoint")

    parent_readouts: dict[str, Mapping[str, Any]] = {}
    parent_summaries: dict[str, Mapping[str, Any]] = {}
    for name, arm in (
        ("A_start", "A_small"),
        ("B_start", "B_large"),
        ("A_step0300", "A_small"),
        ("B_step0300", "B_large"),
    ):
        path = capacity_dir / f"readouts_{name}.pt"
        _require(path.is_file(), f"source endpoint readout is missing: {name}")
        readouts = torch.load(path, map_location="cpu", weights_only=True)
        capacity._validate_readout(readouts, inputs, name)
        summary, _ = capacity._measure_statistics(readouts, inputs, name)
        source_summary = source_report.get("endpoint_metrics", {}).get(name)
        _require(isinstance(source_summary, Mapping), f"source report omitted the measured endpoint {name}")
        _require(
            source_summary["evaluators"]["E_D"]["by_role"]["confirmation"]["per_class_correct"]
            == summary["evaluators"]["E_D"]["by_role"]["confirmation"]["per_class_correct"],
            f"source report and saved {name} E_D confirmation counts disagree",
        )
        parent_readouts[name] = readouts
        parent_summaries[name] = summary
    _require(
        torch.allclose(parent_readouts["A_start"]["raw_cmos"], parent_readouts["B_start"]["raw_cmos"], rtol=RTOL, atol=ATOL)
        and torch.allclose(parent_readouts["A_start"]["z"], parent_readouts["B_start"]["z"], rtol=RTOL, atol=ATOL),
        "the shared zero-update A/B readout is inconsistent",
    )

    population = {
        "task_id": TASK_ID,
        "source_capacity_dir": capacity_dir.as_posix(),
        "source_capacity_task_id": SOURCE_TASK_ID,
        "source_capacity_status": source_report["status"],
        "source_capacity_classification": source_report["classification"],
        "source_checkpoint_paths": {
            "A_small_step0300": (capacity_dir / "arm_a_step0300.pt").as_posix(),
            "B_large_step0300": (capacity_dir / "arm_b_step0300.pt").as_posix(),
        },
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
        "source_input_paths": dict(source_population["input_paths"]),
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
    _require(checkpoint.get("architecture_layout") == ("two_original_100x100_phase_planes" if arm == "A_small" else "two_original_100x100_then_two_400x400_phase_planes_flattened"), f"{arm} source phase layout differs")
    _require(checkpoint.get("next_batch_index") == PARENT_UPDATES % BATCH_COUNT and checkpoint.get("next_batch_cursor") == PARENT_UPDATES % BATCH_COUNT, f"{arm} source next-batch cursor differs")
    _require(checkpoint.get("numeric_mode") == NUMERIC_MODE, f"{arm} source numeric mode differs")
    state = checkpoint.get("trainable_parameter_state")
    _require(isinstance(state, Mapping) and set(state) == {"encoder.phase", "generator.phase"}, f"{arm} source model state is incomplete")
    _require(tuple(state["encoder.phase"].shape) == (2, 1, 100, 100), f"{arm} source Encoder shape differs")
    expected_generator_shape = (2, 1, 100, 100) if arm == "A_small" else (capacity.PHASE_LARGE_COUNT,)
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
    consumer, optimizer, parameters = capacity._new_endpoint(inputs, base_state, arm, device)
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
    _require(PARENT_UPDATES <= cumulative_completed < TOTAL_UPDATES, "next update is outside the authorized 301..1000 range")
    return cumulative_completed % BATCH_COUNT


def _checkpoint_order(arm: str, completed: int) -> int:
    if completed == PARENT_UPDATES:
        return 0
    if arm == "A_small":
        return completed - PARENT_UPDATES
    return 10_000 + completed - PARENT_UPDATES


def _checkpoint_name(arm: str, completed: int) -> str:
    return capacity._checkpoint_name(arm, completed)


def _checkpoint_payload(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    arm: str,
    completed: int,
    history: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    _require(PARENT_UPDATES <= completed <= TOTAL_UPDATES, "checkpoint total update count is outside 300..1000")
    parameters = capacity._parameters(consumer)
    step = int(float(optimizer.state[parameters["generator.phase"]]["step"].item()))
    _require(step == completed, "checkpoint Generator Adam step differs from cumulative updates")
    _require(len(history) == completed, "checkpoint history does not contain cumulative updates")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "arm": arm,
        "architecture_layout": source_checkpoint["architecture_layout"],
        "source_capacity_checkpoint": population["source_checkpoint_paths"]["A_small_step0300" if arm == "A_small" else "B_large_step0300"],
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
    _write_torch_new(path, _checkpoint_payload(consumer, optimizer, arm, completed, history, population, source_checkpoint), "cumulative 300-to-1000 Generator Adam recovery checkpoint")
    return path


def _read_checkpoint(path: Path, target: Path, population: Mapping[str, Any], source_checkpoints: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    _require(path.resolve().parent == target.resolve() and path.is_file(), "resume requires an explicit checkpoint inside this task output")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    _require(checkpoint.get("schema") == CHECKPOINT_SCHEMA and checkpoint.get("task_id") == TASK_ID, "resume checkpoint is not from the cumulative task")
    arm, completed = checkpoint.get("arm"), checkpoint.get("completed_updates")
    _require(arm in ARM_NAMES and isinstance(completed, int) and PARENT_UPDATES <= completed <= TOTAL_UPDATES, "resume checkpoint arm/update count is invalid")
    _require(checkpoint.get("parent_updates") == PARENT_UPDATES and checkpoint.get("additional_updates") == completed - PARENT_UPDATES, "resume source/added update counts differ")
    _require(checkpoint.get("population") == dict(population), "resume checkpoint population, roles, labels, or inputs changed")
    _require(checkpoint.get("next_batch_index") == completed % BATCH_COUNT and checkpoint.get("next_batch_cursor") == completed % BATCH_COUNT, "resume batch cursor differs")
    _require(path.name == _checkpoint_name(arm, completed), "resume filename does not match checkpoint state")
    _require(checkpoint.get("optimizer_step") == completed and checkpoint.get("optimizer_parameter_names") == ["generator.phase"], "resume checkpoint does not retain cumulative Generator Adam state")
    _require(checkpoint.get("numeric_mode") == NUMERIC_MODE, "resume numeric mode differs")
    _require(checkpoint.get("source_capacity_checkpoint") == population["source_checkpoint_paths"]["A_small_step0300" if arm == "A_small" else "B_large_step0300"], "resume parent checkpoint lineage differs")
    history = checkpoint.get("training_history")
    source_history = source_checkpoints[arm]["training_history"]
    _require(isinstance(history, list) and len(history) == completed and _tree_equal(history[:PARENT_UPDATES], source_history), "resume history is not the source 300-step history plus continuation")
    for step, record in enumerate(history[PARENT_UPDATES:], start=PARENT_UPDATES + 1):
        _require(record.get("update") == step and record.get("arm") == arm, f"resume continuation history is invalid at step {step}")
    state = checkpoint.get("trainable_parameter_state")
    _require(isinstance(state, Mapping) and set(state) == {"encoder.phase", "generator.phase"}, "resume model state is incomplete")
    source_parameters = source_checkpoints[arm]["trainable_parameter_state"]
    for name, value in state.items():
        _require(tuple(value.shape) == tuple(source_parameters[name].shape) and value.dtype == source_parameters[name].dtype, f"resume parameter shape/dtype changed for {name}")
        _require(bool(torch.isfinite(value).all().item()), "resume model state is nonfinite")
    _validate_source_checkpoint(source_checkpoints[arm], source_checkpoints[arm]["population"], arm)
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
        candidate_arm = "A_small" if match.group(1) == "a" else "B_large"
        candidate_step = int(match.group(2))
        if _checkpoint_order(candidate_arm, candidate_step) > _checkpoint_order(arm, completed):
            later.append(candidate.name)
    _require(not later, f"selected checkpoint predates saved task progress: {sorted(later)}")
    report_path = target / "report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        _require(report.get("task_id") == TASK_ID and report.get("status") in {"VALID_COMPLETED", "VALID_FAILED"}, "terminal report is invalid")
        _require(arm == "B_large" and completed == TOTAL_UPDATES, "terminal output requires the explicit final B checkpoint")
    return dict(checkpoint)


def _completed_resume_report(target: Path, checkpoint: Mapping[str, Any]) -> dict[str, Any] | None:
    report_path = target / "report.json"
    if not report_path.is_file():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("task_id") == TASK_ID and report.get("status") in {"VALID_COMPLETED", "VALID_FAILED"}, "saved terminal report is invalid")
    _require(checkpoint.get("arm") == "B_large" and checkpoint.get("completed_updates") == TOTAL_UPDATES, "terminal recovery requires the explicit final B checkpoint")
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
    if arm == "A_small":
        batch = inputs["images"][:BATCH_SIZE].to(device)
        with torch.no_grad():
            no_grad_result = consumer.runtime.forward_from_image(batch)
            _require(bool(torch.isfinite(no_grad_result.raw_intensity).all().item()), f"{label} no-grad forward is nonfinite")
        del no_grad_result
    for update_index in range(PARENT_UPDATES, PARENT_UPDATES + 3):
        gc.collect()
        torch.cuda.empty_cache()
        resources.append(_gpu_snapshot(device))
        record = capacity._train_step(consumer, optimizer, parameters, evaluators, inputs, update_index, arm=arm)
        records.append(record)
        capacity._assert_disposable_frozen(consumer, parameters, encoder_before, runtime_before, evaluators, evaluator_before, label)
        gc.collect()
        torch.cuda.empty_cache()
    step = int(float(optimizer.state[parameters["generator.phase"]]["step"].item()))
    _require(step == PARENT_UPDATES + 3, f"{label} did not continue Adam from 300 to 303")
    _require(not torch.equal(parameters["generator.phase"].detach().cpu(), source_checkpoint["trainable_parameter_state"]["generator.phase"]), f"{label} did not update the Generator")
    if arm == "B_large":
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
    consumer, optimizer, parameters = _load_arm(inputs, base_state, source_checkpoint, "A_small", device)
    images_cpu, labels_cpu, _ = capacity.training_batch(inputs["images"], inputs["labels"], 0)
    images, labels = images_cpu.to(device), labels_cpu.to(device)
    result = consumer.runtime.forward_from_image(images)
    view = capacity.canonical.cmos_view(result.raw_intensity)
    wrong = labels.roll(1)
    differences: dict[str, float] = {}
    derivative_reference: dict[str, Any] | None = None
    for index, name in enumerate(TRAINING_EVALUATORS):
        logits = capacity._evaluator_logits(evaluators[name], view, name)
        if name == "E_S":
            derivative_reference = capacity._check_ce_derivative(logits, labels)
        correct_gradient = torch.autograd.grad(F.cross_entropy(logits, labels), parameters["generator.phase"], retain_graph=True)[0]
        wrong_gradient = torch.autograd.grad(F.cross_entropy(logits, wrong), parameters["generator.phase"], retain_graph=index < len(TRAINING_EVALUATORS) - 1)[0]
        differences[name] = capacity._vector_norm(correct_gradient - wrong_gradient)
        _require(differences[name] > 0.0, f"wrong-label negative control did not change {name} gradient")
    _require(derivative_reference is not None, "independent CE derivative reference did not execute")
    _require(all(parameter.grad is None for parameter in parameters.values()), "negative control populated parameter .grad")
    result = {
        "passed": True,
        "source_step": PARENT_UPDATES,
        "wrong_label_generator_gradient_difference_norm": differences,
        "independent_ce_derivative": derivative_reference,
    }
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _recovery_control(
    inputs: Mapping[str, Any],
    base_state: Mapping[str, Any],
    source_checkpoint: Mapping[str, Any],
    evaluators: Mapping[str, nn.Module],
    population: Mapping[str, Any],
    workspace: Path,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    consumer, optimizer, parameters = _load_arm(inputs, base_state, source_checkpoint, "A_small", device)
    history = [dict(row) for row in source_checkpoint["training_history"]]
    first = capacity._train_step(consumer, optimizer, parameters, evaluators, inputs, PARENT_UPDATES, arm="A_small")
    history.append(first)
    payload = _checkpoint_payload(consumer, optimizer, "A_small", PARENT_UPDATES + 1, history, population, source_checkpoint)
    path = workspace / "arm_a_step0301.pt"
    torch.save(payload, path)
    restored = torch.load(path, map_location="cpu", weights_only=True)
    continuous_records = [capacity._train_step(consumer, optimizer, parameters, evaluators, inputs, idx, arm="A_small") for idx in (301, 302)]
    continuous = {"parameters": capacity._snapshot_parameters(parameters), "optimizer": _cpu_tree(optimizer.state_dict()), "records": continuous_records}
    del consumer, optimizer, parameters
    gc.collect()
    torch.cuda.empty_cache()

    consumer, optimizer, parameters = _load_arm(inputs, base_state, source_checkpoint, "A_small", device)
    _restore_checkpoint(restored, consumer, optimizer, "A_small")
    resumed_records = [capacity._train_step(consumer, optimizer, parameters, evaluators, inputs, idx, arm="A_small") for idx in (301, 302)]
    resumed = {"parameters": capacity._snapshot_parameters(parameters), "optimizer": _cpu_tree(optimizer.state_dict()), "records": resumed_records}
    passed, detail = capacity._repeatability(continuous, resumed)
    _require(passed, f"continuous and resumed 301..303 paths differ: {detail}")
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

        trial_a = _trial_from_source("A_small", inputs, source_state, source_checkpoints["A_small"], evaluators, label="A_source_300_primary")
        trial_a_repeat = _trial_from_source("A_small", inputs, source_state, source_checkpoints["A_small"], evaluators, label="A_source_300_repeat")
        repeat_ok, repeat_detail = capacity._repeatability(
            {key: trial_a[key] for key in ("parameters", "optimizer", "records")},
            {key: trial_a_repeat[key] for key in ("parameters", "optimizer", "records")},
        )
        _require(repeat_ok, f"source-state A/A repeatability failed: {repeat_detail}")
        progress["A_A_from_300_repeatability"] = {"executed": True, "passed": True, "detail": repeat_detail}

        trial_b = _trial_from_source("B_large", inputs, source_state, source_checkpoints["B_large"], evaluators, label="B_source_300")
        progress["three_real_updates_from_each_300_step_checkpoint"] = {
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

        progress["wrong_label_ce_negative_control"] = {"executed": True, **_wrong_label_control(inputs, source_state, source_checkpoints["A_small"], evaluators)}
        with tempfile.TemporaryDirectory(prefix="lg_accum_recovery_", dir=Path(tempfile.gettempdir())) as temp_name:
            recovery = _recovery_control(inputs, source_state, source_checkpoints["A_small"], evaluators, population, Path(temp_name))
        progress["interrupted_resume_equivalence"] = {"executed": True, **recovery}
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], _state_snapshot(model), f"preflight {name} evaluator")
            _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"preflight changed {name} evaluator freeze")
        progress["frozen_evaluator_state"] = {"executed": True, "passed": True, "evaluators": list(EVALUATOR_NAMES)}
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
            "optimizer_state_policy": "retain_parent_generator_adam_moments_and_step_300",
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
    return control.make_ready_receipt(
        correctness={
            "passed": True,
            "status": "PASS",
            "four_evaluator_target_positive_controls": checks["four_evaluator_target_positive_control"]["passed"],
            "retained_adam_300_to_303": checks["three_real_updates_from_each_300_step_checkpoint"]["passed"],
            "wrong_label_ce_negative_control": checks["wrong_label_ce_negative_control"]["passed"],
            "frozen_encoder_ols_and_evaluators": checks["three_real_updates_from_each_300_step_checkpoint"]["frozen_scope_checked_each_step"],
        },
        resource={
            "passed": True,
            "status": "PASS",
            "cuda_runtime": True,
            "resource_observations": {
                "before": preflight["cuda_before_observation"],
                "after": preflight["cuda_after_observation"],
            },
            "disk_observation": preflight["disk_observation"],
        },
        implementation={
            "passed": True,
            "status": "PASS",
            "selected_engine": "cuda_2048_accumulated_generator_only",
            "optimizer_history_retained": True,
            "A_A_repeatability": checks["A_A_from_300_repeatability"]["passed"],
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
        trajectory_id="fixed-small-large-generator-cumulative-1000-updates",
        semantic_authority_id=SOURCE_TASK_ID,
        data_role="canonical TRAIN fit=200 rows updated; historic=100 and confirmation=300 excluded from gradients",
        config_identity={
            "propagation_grid": 2048,
            "total_updates_per_arm": TOTAL_UPDATES,
            "source_updates_per_arm": PARENT_UPDATES,
            "additional_updates_per_arm": ADDED_UPDATES,
            "logical_batch_size": BATCH_SIZE,
            "large_arm_microbatch_size": capacity.LARGE_GRADIENT_MICROBATCH_SIZE,
            "optimizer_state": "retain exact source Adam moments and step 300",
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
    consumer, optimizer, _ = capacity._new_endpoint(inputs, source_state, "A_small", torch.device("cpu"))
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
    extra_power = torch.empty((rows, 2), dtype=torch.float64) if arm == "B_large" else None
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
            if extra_power is not None:
                generator = consumer.runtime.generator
                _require(isinstance(generator, capacity.ExpandedGenerator) and generator.last_extra_plane_powers is not None, "large Generator omitted added-plane powers")
                extra_power[start:stop].copy_(generator.last_extra_plane_powers.detach().to(torch.float64).cpu())
            for name in EVALUATOR_NAMES:
                logits[name][start:stop].copy_(capacity._evaluator_logits(evaluators[name], current_view, name).detach().cpu())
            if fields_metadata is None:
                fields_metadata = _fields_metadata(result, arm, consumer)
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
    metadata = capacity._all_fields_metadata(result, arm, consumer)
    if arm == "B_large":
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
    readouts = _forward_all(consumer, evaluators, inputs, arm, target, f"measure_{endpoint}", started)
    capacity._validate_readout(readouts, inputs, endpoint)
    if arm == "B_large":
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
    a_metrics = capacity._endpoint_accuracy(summaries["A_step1000"])
    start_metrics = capacity._endpoint_accuracy(summaries["common_start"])
    b_metrics = capacity._endpoint_accuracy(summaries["B_step1000"])
    a_correct = capacity._correct_by_class(summaries["A_step1000"])
    start_correct = capacity._correct_by_class(summaries["common_start"])
    b_correct = capacity._correct_by_class(summaries["B_step1000"])
    focus_deltas = {
        str(class_id): {
            "vs_A_step1000": b_correct[class_id] - a_correct[class_id],
            "vs_common_zero_start": b_correct[class_id] - start_correct[class_id],
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
    a_raw = float(summaries["A_step1000"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"])
    start_raw = float(summaries["common_start"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"])
    b_summary = summaries["B_step1000"]
    b_confirmation = b_summary["image_metrics_by_role"]["confirmation"]
    b_raw = float(b_confirmation["overall"]["raw_mse"])
    clip_guard = b_confirmation["overall"]["raw_fraction_above_one"] <= 0.01 and all(
        b_confirmation[str(c)]["raw_fraction_above_one"] <= 0.01 for c in FOCUS_CLASSES
    )
    image_guard = (
        b_raw <= 1.05 * a_raw
        and b_raw <= 1.05 * start_raw
        and clip_guard
        and not b_summary["has_zero_image"]
        and not b_summary["has_constant_image"]
        and b_summary["all_finite"]
    )
    conditions = {
        "class_1_and_7_each_gain_at_least_3_over_A1000_and_common_start": focus_gain,
        "macro_recall_at_least_A1000_plus_0_05_and_not_below_common_start": b_macro >= a_macro + 0.05 and b_macro >= start_macro,
        "no_class_loses_more_than_3_vs_A1000_or_common_start": class_guard,
        "raw_mse_and_clipping_image_guard": image_guard,
        "training_population_freeze_and_measurement_valid": measurement_valid,
    }
    if not measurement_valid:
        classification, status, recommendation = "INVALID_HARNESS", "INVALID_HARNESS", None
    elif all(conditions.values()):
        classification, status = "CUMULATIVE_CAPACITY_GAIN_SUPPORTED", "VALID_COMPLETED"
        recommendation = "PRIORITIZE_LARGE_GENERATOR_FOR_NEXT_CONTROLLED_PRESENTATION_DESIGN"
    elif any(b_correct[c] > a_correct[c] and b_correct[c] > start_correct[c] for c in FOCUS_CLASSES):
        classification, status, recommendation = "MIXED_OR_INSUFFICIENT", "VALID_FAILED", "NONE"
    else:
        classification, status, recommendation = "NOT_SUPPORTED_WITHIN_1000_UPDATES", "VALID_FAILED", "NONE"
    return {
        "status": status,
        "classification": classification,
        "conditions": conditions,
        "focus_class_correct_deltas": focus_deltas,
        "macro_recall": {"A_step1000": a_macro, "common_start": start_macro, "B_step1000": b_macro},
        "confirmation_raw_mse": {"A_step1000": a_raw, "common_start": start_raw, "B_step1000": b_raw},
        "B_step1000_clipping": {
            "overall": b_confirmation["overall"]["raw_fraction_above_one"],
            "class_1": b_confirmation["1"]["raw_fraction_above_one"],
            "class_7": b_confirmation["7"]["raw_fraction_above_one"],
        },
        "class_correct": {"A_step1000": a_correct, "common_start": start_correct, "B_step1000": b_correct},
        "training_recommendation": recommendation,
        "next_action": "NONE",
        "stop": True,
    }


def _phase_class_render(view: torch.Tensor, scale: int = 4) -> Image.Image:
    image = view.detach().to(torch.float32).cpu().squeeze().clamp(0.0, 1.0)
    array = torch.round((1.0 - image) * 255.0).to(torch.uint8).numpy()
    return Image.fromarray(array, mode="L").resize((28 * scale, 28 * scale), Image.Resampling.NEAREST)


def _comparison_pages(target: Path, inputs: Mapping[str, Any], readouts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    names = ("A_step0300", "B_step0300", "A_step1000", "B_step1000")
    headers = ("Target", "A300", "B300", "A1000", "B1000")
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
            canvas = Image.new("RGB", (1080, 58 + len(page_positions) * row_height), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"{group_name}; rows {page_start + 1}-{page_start + len(page_positions)} of {len(positions)}; fixed [0,1]", fill="black")
            xs = (425, 535, 645, 755, 865)
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
            canvas.save(target / filename, optimize=True)
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
    _require(isinstance(preflight, Mapping) and preflight.get("status") == "READY" and all(item.get("executed") for item in preflight.get("checks", {}).values()), "READY receipt does not contain executed checks")
    return dict(preflight), PreflightReceipt.from_mapping(saved["receipt"])


def _persist_source_readouts(target: Path, inputs: Mapping[str, Any], parent_readouts: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    readouts: dict[str, Mapping[str, Any]] = {}
    for old_name, new_name in (("A_start", "common_start"),):
        path = target / f"readouts_{new_name}.pt"
        expected = parent_readouts[old_name]
        if path.exists():
            current = torch.load(path, map_location="cpu", weights_only=True)
            _require(_tree_equal(current, expected), f"saved parent readout {new_name} changed")
        else:
            _write_torch_new(path, expected, f"copied frozen parent 600-row {new_name} readout")
        capacity._validate_readout(expected, inputs, old_name if old_name != "common_start" else "A_start")
        summary, _ = capacity._save_measurement(target, new_name, expected, inputs)
        readouts[new_name] = expected
    return readouts


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
    readouts = _forward_all(consumer, evaluators, inputs, arm, target, f"source_replay_{endpoint}", started)
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
            record = capacity._train_step(consumer, optimizer, parameters, evaluators, inputs, completed, arm=arm)
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

    _require(completed == TOTAL_UPDATES and len(history) == TOTAL_UPDATES, f"{arm} did not finish exactly at cumulative update 1000")
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
        _write_json_new(history_path, history_payload, f"{arm} source 300-step and added 700-step training history")
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
            _require(existing.get("schema") == CHECKPOINT_SCHEMA and existing.get("arm") == arm and existing.get("completed_updates") == PARENT_UPDATES, f"imported {arm} checkpoint is invalid")
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
    readouts = _persist_source_readouts(target, inputs, parent_readouts)
    summaries: dict[str, Mapping[str, Any]] = {}
    replay: dict[str, Any] = {}
    for arm, endpoint, source_name in (
        ("A_small", "A_step0300", "A_step0300"),
        ("B_large", "B_step0300", "B_step0300"),
    ):
        saved_path = target / f"readouts_{endpoint}.pt"
        if saved_path.is_file():
            measured = torch.load(saved_path, map_location="cpu", weights_only=True)
            capacity._validate_readout(measured, inputs, endpoint)
            summary, _ = capacity._save_measurement(target, endpoint, measured, inputs)
        else:
            measured, summary = _measure_source_endpoint(arm, endpoint, inputs, base_state, source_checkpoints[arm], evaluators, target, started)
        replay[endpoint] = _compare_readout(measured, parent_readouts[source_name], endpoint)
        readouts[endpoint] = measured
        summaries[endpoint] = summary
    summaries["common_start"], _ = capacity._save_measurement(target, "common_start", readouts["common_start"], inputs)
    return readouts, summaries, replay


def _paired_metrics(readouts: Mapping[str, Mapping[str, Any]], labels: torch.Tensor) -> dict[str, Any]:
    return {
        "A300_to_A1000": capacity._paired_correctness(readouts["A_step0300"], readouts["A_step1000"], labels),
        "B300_to_B1000": capacity._paired_correctness(readouts["B_step0300"], readouts["B_step1000"], labels),
        "A1000_to_B1000": capacity._paired_correctness(readouts["A_step1000"], readouts["B_step1000"], labels),
        "common_start_to_B1000": capacity._paired_correctness(readouts["common_start"], readouts["B_step1000"], labels),
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
    parser.add_argument("--capacity-dir", type=Path, default=DEFAULT_CAPACITY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="explicit cumulative checkpoint required with --resume")
    return parser.parse_args()


def execute(arguments: argparse.Namespace | None = None) -> dict[str, Any]:
    arguments = arguments or _arguments()
    root = _root()
    capacity_dir = _resolve(root, arguments.capacity_dir)
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
        phase = "source_300_replay_and_training"
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
            source_replays = {}
            readouts: dict[str, Mapping[str, Any]] = {}
            summaries: dict[str, Mapping[str, Any]] = {}
            if arguments.resume and (target / "readouts_A_step0300.pt").is_file() and (target / "readouts_B_step0300.pt").is_file():
                readouts = dict(_persist_source_readouts(target, inputs, parent_readouts))
                for arm, endpoint in (("A_small", "A_step0300"), ("B_large", "B_step0300")):
                    readouts[endpoint] = torch.load(target / f"readouts_{endpoint}.pt", map_location="cpu", weights_only=True)
                    source_replays[endpoint] = _compare_readout(readouts[endpoint], parent_readouts[endpoint], endpoint)
                    summaries[endpoint], _ = capacity._save_measurement(target, endpoint, readouts[endpoint], inputs)
                summaries["common_start"], _ = capacity._save_measurement(target, "common_start", readouts["common_start"], inputs)
            else:
                readouts, summaries, source_replays = _source_replay_report(
                    target, inputs, base_state, source_checkpoints, parent_readouts, evaluators, started
                )
            progress["300_step_replays"] = {"executed": True, **source_replays}

            selected_arm = selected["arm"] if arguments.resume else None
            if arguments.resume and selected_arm == "B_large" and selected["completed_updates"] > PARENT_UPDATES:
                a_final_checkpoint = torch.load(target / _checkpoint_name("A_small", TOTAL_UPDATES), map_location="cpu", weights_only=True)
                a_history_payload = json.loads((target / "training_history_A_small.json").read_text(encoding="utf-8"))
                _require(a_history_payload.get("total_updates") == TOTAL_UPDATES, "A completed history is missing before B continuation")
                a_task_checkpoint = a_final_checkpoint
            elif arguments.resume and selected_arm == "A_small":
                a_task_checkpoint = _train_arm(
                    "A_small", inputs, base_state, source_checkpoints["A_small"], evaluators, target, population,
                    start_from=selected, started=started,
                )
            else:
                a_task_checkpoint = _train_arm(
                    "A_small", inputs, base_state, source_checkpoints["A_small"], evaluators, target, population,
                    start_from=torch.load(target / _checkpoint_name("A_small", PARENT_UPDATES), map_location="cpu", weights_only=True), started=started,
                )
            if "A_step1000" not in readouts:
                a_summary, a_data = _ensure_measurement(
                    "A_small", "A_step1000", a_task_checkpoint, source_checkpoints["A_small"], None,
                    base_state, inputs, evaluators, target, started,
                )
                readouts["A_step1000"] = a_data["readouts"]
                summaries["A_step1000"] = a_summary
            else:
                summaries["A_step1000"], _ = capacity._save_measurement(target, "A_step1000", readouts["A_step1000"], inputs)

            if selected_arm == "B_large":
                b_start = selected
            else:
                b_start = torch.load(target / _checkpoint_name("B_large", PARENT_UPDATES), map_location="cpu", weights_only=True)
            if selected_arm == "B_large" and selected["completed_updates"] == TOTAL_UPDATES:
                b_task_checkpoint = selected
            else:
                b_task_checkpoint = _train_arm(
                    "B_large", inputs, base_state, source_checkpoints["B_large"], evaluators, target, population,
                    start_from=b_start, started=started,
                )
            _ensure_history_artifact(target, "A_small", a_task_checkpoint)
            _ensure_history_artifact(target, "B_large", b_task_checkpoint)
            b_summary, b_data = _ensure_measurement(
                "B_large", "B_step1000", b_task_checkpoint, source_checkpoints["B_large"], None,
                base_state, inputs, evaluators, target, started,
            )
            readouts["B_step1000"] = b_data["readouts"]
            summaries["B_step1000"] = b_summary

            for name, model in evaluators.items():
                capacity._assert_state_equal(evaluator_before[name], _state_snapshot(model), f"final {name} evaluator")
                _require(all(parameter.grad is None and not parameter.requires_grad for parameter in model.parameters()), f"final {name} evaluator is not frozen")
            _require(set(readouts) >= {"common_start", "A_step0300", "B_step0300", "A_step1000", "B_step1000"}, "one or more required readouts are missing")
            final = _final_decision(summaries, measurement_valid=True)
            transitions = _paired_metrics(readouts, inputs["labels"])
            montage = _comparison_pages(target, inputs, readouts)
            a_source_three_hundred = parent_summaries["A_step0300"]
            b_source_three_hundred = parent_summaries["B_step0300"]
            incremental = {
                "A": {
                    "300_to_1000_correct_by_class": {
                        "from": capacity._correct_by_class(a_source_three_hundred),
                        "to": capacity._correct_by_class(summaries["A_step1000"]),
                    },
                    "confirmation_macro_recall_change": float(capacity._endpoint_accuracy(summaries["A_step1000"])["macro_recall"] - capacity._endpoint_accuracy(a_source_three_hundred)["macro_recall"]),
                    "raw_mse_change": float(summaries["A_step1000"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"] - a_source_three_hundred["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"]),
                },
                "B": {
                    "300_to_1000_correct_by_class": {
                        "from": capacity._correct_by_class(b_source_three_hundred),
                        "to": capacity._correct_by_class(summaries["B_step1000"]),
                    },
                    "confirmation_macro_recall_change": float(capacity._endpoint_accuracy(summaries["B_step1000"])["macro_recall"] - capacity._endpoint_accuracy(b_source_three_hundred)["macro_recall"]),
                    "raw_mse_change": float(summaries["B_step1000"]["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"] - b_source_three_hundred["image_metrics_by_role"]["confirmation"]["overall"]["raw_mse"]),
                    "class_8_correct_change": capacity._correct_by_class(summaries["B_step1000"])[8] - capacity._correct_by_class(b_source_three_hundred)[8],
                },
            }
            report = {
                "schema": REPORT_SCHEMA,
                "task_id": TASK_ID,
                **final,
                "scientific_question": "whether increasing the existing small and large Generator rules from 300 to 1000 cumulative updates improves large-Generator classes 1/7 against both small-Generator step1000 and common step0 while preserving other classes",
                "population": population,
                "source": {
                    "capacity_dir": capacity_dir.as_posix(),
                    "task_id": source_report["task_id"],
                    "status": source_report["status"],
                    "classification": source_report["classification"],
                    "A_checkpoint": population["source_checkpoint_paths"]["A_small_step0300"],
                    "B_checkpoint": population["source_checkpoint_paths"]["B_large_step0300"],
                    "both_generator_adam_steps_validated_at": PARENT_UPDATES,
                },
                "training": {
                    "parent_updates_per_arm": PARENT_UPDATES,
                    "additional_updates_per_arm": ADDED_UPDATES,
                    "cumulative_updates_per_arm": TOTAL_UPDATES,
                    "batch_size": BATCH_SIZE,
                    "batch_count": BATCH_COUNT,
                    "additional_cycles": ADDED_UPDATES // BATCH_COUNT,
                    "parameter_scope": "Generator phase only; Encoder/OLS/scientific buffers/evaluators frozen",
                    "optimizer_state": "source Adam moments and step retained; next update 301, final step 1000",
                    "optimizer": {"lr": capacity.GENERATOR_LEARNING_RATE, "betas": list(capacity.GENERATOR_BETAS), "eps": capacity.GENERATOR_EPS, "weight_decay": capacity.GENERATOR_WEIGHT_DECAY},
                    "large_arm_microbatch_rows": capacity.LARGE_GRADIENT_MICROBATCH_SIZE,
                    "large_arm_full_batch_OLS_statistics_preserved": True,
                    "numeric_mode": numeric_mode,
                    "training_objective": "existing three-evaluator true-class CE, raw reconstruction MSE and BOGT under unchanged normalized direction",
                },
                "source_step0300_replay": source_replays,
                "endpoint_metrics": summaries,
                "incremental_300_to_1000": incremental,
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
            _write_json_new(target / "report.json", report, "final cumulative 1000-update capacity comparison decision")
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
