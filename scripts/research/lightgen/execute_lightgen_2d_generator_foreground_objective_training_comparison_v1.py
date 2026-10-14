"""Run the fixed-weight A4700 foreground-objective training comparison."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import signal
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
)
from scripts.research.lightgen import audit_lightgen_2d_generator_cross_port_joint_structure_v1 as frozen
from scripts.research.lightgen import diagnose_lightgen_2d_generator_foreground_objective_gradient_v1 as gradient
from scripts.research.lightgen import diagnose_lightgen_2d_class17_mapping_v2 as diagnostic
from scripts.research.lightgen import execute_lightgen_2d_generator_energy_matched_target_comparison_v1 as parent_task
from scripts.research.lightgen import execute_lightgen_2d_generator_frozen_conditional_ols_generation_v1 as generation
from scripts.research.lightgen import execute_lightgen_2d_generator_large_capacity_quick_comparison_v1 as capacity
from scripts.research.lightgen import execute_lightgen_2d_generator_semantic_target_comparison_v1 as semantic
from scripts.research.lightgen import execute_lightgen_2d_generator_stroke_fidelity_comparison_v1 as stroke


TASK_ID = "M0_GENERATOR_FOREGROUND_OBJECTIVE_TRAINING_COMPARISON_V1"
SOURCE_STEP = 4_700
ADDED_UPDATES = 700
FINAL_STEP = SOURCE_STEP + ADDED_UPDATES
FIT_ROWS = 200
CONFIRM_ROWS = 300
TOTAL_ROWS = FIT_ROWS + CONFIRM_ROWS
BATCH_SIZE = 10
BATCH_COUNT = FIT_ROWS // BATCH_SIZE
FOCUS_CLASSES = (2, 4)
FIXED_LAMBDA = 0.11159457624315472
MEASUREMENT_STEPS = (0, 100, 300, 700)
CHUNK_SIZE = 50
REPLAY_RTOL = 2.0e-6
REPLAY_ATOL = 5.0e-8
GRADIENT_RTOL = 2.0e-5
GRADIENT_ATOL = 2.0e-7
FD_EPSILON = 1.0e-3
FD_RTOL = 0.1
FD_ATOL = 5.0e-7
DECISION_ATOL = 2.0e-7
DECISION_RTOL = 2.0e-5
NUMERIC_MODE = "strict_fp32_optics_float64_qfg_statistics"
STRICT_FP32_SETTINGS = {
    "dtype": "float32",
    "autocast": False,
    "cuda_matmul_tf32": False,
    "cudnn_tf32": False,
    "float32_matmul_precision": "highest",
}
CHECKPOINT_SCHEMA = "lightgen_foreground_objective_training_checkpoint_v1"
DEFAULT_ARTIFACT = Path("outputs/20261009_m0_generator_energy_matched_target_comparison_v1/runs/attempt_03")
DEFAULT_CHECKPOINT = DEFAULT_ARTIFACT / "arm_a_step4700.pt"
DEFAULT_GRADIENT_AUDIT = Path("outputs/20261009_m0_generator_foreground_objective_gradient_v1/runs/attempt_04")
DEFAULT_OUTPUT = Path("outputs/20261009_m0_generator_foreground_objective_training_comparison_v1/runs/attempt_01")
METRIC_FIELDS = (
    "scenario", "position", "source_row_index", "label", "role", "q_fg", "q_bg", "q_raw200",
    "raw_mse", "raw_mse_fg", "raw_mse_bg", "production_q_28", "ssim_28", "integral_intensity", "clipped_fraction",
    "e_d_prediction", "e_d_target_logit",
)


class TrainingComparisonError(RuntimeError):
    """An invalid input or failed production invariant prevents a scientific result."""


class TrainingInterrupted(RuntimeError):
    """A stop request was durably recorded at a training checkpoint boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingComparisonError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: Path) -> Path:
    return (value if value.is_absolute() else root / value).resolve()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        _require(_json(path) == value, f"refusing to overwrite different evidence: {path.name}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _write_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.exists(), f"refusing to overwrite saved tensor evidence: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str], *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        _require(_read_csv(path) == [dict(row) for row in rows], f"refusing to overwrite different CSV evidence: {path.name}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _tree_equal(left: Any, right: Any) -> bool:
    return capacity._tree_equal(left, right)


def _load_inputs(root: Path, artifact_dir: Path, checkpoint: Path, gradient_dir: Path) -> dict[str, Any]:
    _require(artifact_dir.is_dir() and gradient_dir.is_dir(), "the explicitly named source or gradient-audit directory is missing")
    _require(checkpoint == (artifact_dir / "arm_a_step4700.pt").resolve(), "checkpoint must be the A4700 copy inside the named energy-comparison attempt")
    inputs = frozen._load_inputs(root, artifact_dir, checkpoint)
    audit_report = _json(gradient_dir / "report.json")
    audit_population = _json(gradient_dir / "input_population.json")
    calibration = _json(gradient_dir / "fit_calibration.json")
    audit_preflight = _json(gradient_dir / "preflight.json")
    gradients_path = gradient_dir / "fit_gradients.pt"
    _require(audit_report.get("task_id") == gradient.TASK_ID and audit_report.get("status") == "VALID_COMPLETED", "named foreground-gradient audit is not a valid completed measurement")
    _require(audit_preflight.get("status") == "READY" and audit_preflight.get("task_id") == gradient.TASK_ID, "foreground-gradient audit lacks valid READY preflight evidence")
    _require(calibration.get("confirmation_rows_used_for_calibration") == 0 and calibration.get("fit_rows") == FIT_ROWS and calibration.get("fit_batches") == BATCH_COUNT, "gradient weight was not fixed from the 200-row fit population")
    _require(audit_population.get("source_checkpoint") == inputs["population"].get("source_checkpoint"), "gradient audit came from a different A4700 source checkpoint")
    _require(audit_population.get("source_row_indices") == inputs["population"].get("source_row_indices"), "gradient audit population rows differ from the training comparison")
    _require(audit_population.get("labels") == inputs["population"].get("labels") and audit_population.get("roles") == inputs["population"].get("roles"), "gradient audit row labels or roles differ")
    _require(gradients_path.is_file(), "gradient audit fit-gradient evidence is missing")
    fit_gradients = torch.load(gradients_path, map_location="cpu", weights_only=True)
    lambdas = (
        calibration.get("lambda"),
        audit_report.get("measurement", {}).get("lambda"),
        fit_gradients.get("lambda"),
    )
    _require(all(isinstance(value, (float, int)) and float(value) == FIXED_LAMBDA for value in lambdas), "saved fit calibration, report and gradient evidence do not match the prespecified lambda")
    _require(fit_gradients.get("task_id") == gradient.TASK_ID and float(fit_gradients.get("beta", -1.0)) == semantic.BETA, "saved gradient weight/evidence identity differs")
    source = inputs["generation_inputs"]["source"]
    _require(source.get("completed_updates") == SOURCE_STEP and source.get("optimizer_step") == SOURCE_STEP, "A4700 model and Adam are not at the fixed source step")
    _require(len(source.get("training_history", [])) == SOURCE_STEP, "A4700 source training history is incomplete")
    optimizer = source["optimizer_state"]
    groups = optimizer.get("param_groups", [])
    states = optimizer.get("state", {})
    _require(len(groups) == 1 and groups[0].get("params") == [0] and set(states) == {0}, "source Generator Adam parameter mapping differs")
    group = groups[0]
    _require(float(group.get("lr", -1.0)) == 1e-3 and tuple(group.get("betas", ())) == (0.9, 0.999) and float(group.get("eps", -1.0)) == 1e-8 and float(group.get("weight_decay", -1.0)) == 0.0, "source Adam configuration differs from the fixed plan")
    _require(int(float(states[0]["step"].item())) == SOURCE_STEP, "source Adam step differs from A4700")
    inputs["gradient_audit"] = {
        "directory": gradient_dir.as_posix(),
        "report_status": audit_report["status"],
        "lambda": FIXED_LAMBDA,
        "calibration": calibration,
        "fit_gradients_path": gradients_path.as_posix(),
    }
    return inputs


def _foreground_q(raw: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    return gradient._qfg_components(raw, target)


def _fit_batch(inputs: Mapping[str, Any], batch_index: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    start, stop = batch_index * BATCH_SIZE, (batch_index + 1) * BATCH_SIZE
    images = inputs["generation_inputs"]["images"][start:stop].to(device=device, dtype=torch.float32)
    labels = inputs["labels"][start:stop].to(device=device)
    _require(images.shape == (BATCH_SIZE, 1, 28, 28) and torch.bincount(labels.cpu(), minlength=10).tolist() == [1] * 10, "fixed fit batch is not a ten-class balanced microbatch")
    return images, labels


def _train_step(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, nn.Parameter],
    evaluators: Mapping[str, nn.Module],
    inputs: Mapping[str, Any],
    arm: str,
    local_update: int,
    *,
    qfg_lambda: float | None = None,
) -> dict[str, Any]:
    _require(arm in {"A", "B"}, "training arm must be A or B")
    _require(0 <= local_update < ADDED_UPDATES, "task-local update cursor is outside the fixed budget")
    batch_index = local_update % BATCH_COUNT
    images, labels = _fit_batch(inputs, batch_index, consumer.device)
    base_gradient, reconstruction_gradient, qfg_gradient, detail = gradient._single_microbatch_batch_gradients(
        consumer, parameters, evaluators, images, labels
    )
    effective_lambda = 0.0 if arm == "A" else FIXED_LAMBDA if qfg_lambda is None else float(qfg_lambda)
    _require(math.isfinite(effective_lambda) and effective_lambda >= 0.0, "effective foreground weight must be finite and nonnegative")
    generator_gradient = base_gradient if effective_lambda == 0.0 else gradient._candidate_generator_gradient(base_gradient, qfg_gradient, effective_lambda)
    _require(bool(torch.isfinite(generator_gradient).all()), "combined Generator gradient is nonfinite")
    phase = parameters["generator.phase"]
    before = phase.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    phase.grad = generator_gradient.detach().clone()
    _require(parameters["encoder.phase"].grad is None, "Encoder phase received a training gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = int(float(optimizer.state[phase]["step"].item()))
    _require(optimizer_step == SOURCE_STEP + local_update + 1, "Generator Adam step differs from A4700 plus the task-local update")
    displacement = phase.detach() - before
    _require(bool(torch.isfinite(phase).all() and torch.isfinite(displacement).all()), "training produced a nonfinite Generator phase/update")
    _require(bool((displacement != 0).any()), "Adam did not update the Generator phase")
    _require(all(parameter.grad is None for parameter in parameters.values()), "training left a parameter gradient populated")
    record = {
        "task_local_update": local_update + 1,
        "optimizer_step": optimizer_step,
        "arm": arm,
        "batch_index": batch_index,
        "source_row_indices": inputs["population"]["source_row_indices"][batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE],
        "labels": labels.detach().cpu().tolist(),
        "effective_qfg_lambda": effective_lambda,
        "semantic_objective": "preserved normalized E_S/E_I/E_C production gradient",
        "reconstruction_objective": "raw200 MSE + 0.1*(1-SSIM28)",
        "raw_mse": detail["metrics"]["raw_mse"],
        "ssim_loss": detail["metrics"]["ssim_loss"],
        "q_fg": detail["metrics"]["q_fg"],
        "l_op": detail["metrics"]["l_op"],
        "l_dist": detail["metrics"]["l_dist"],
        "ce_by_evaluator": {name: detail["metrics"][f"{name}_ce"] for name in ("E_S", "E_I", "E_C")},
        "base_gradient_rms": gradient._rms(base_gradient),
        "reconstruction_gradient_rms": gradient._rms(reconstruction_gradient),
        "qfg_gradient_rms": gradient._rms(qfg_gradient),
        "added_qfg_gradient_rms": gradient._rms(semantic.BETA * effective_lambda * qfg_gradient),
        "combined_gradient_rms": gradient._rms(generator_gradient),
        "phase_update_rms": gradient._rms(displacement),
        "finite": True,
    }
    return record


def _training_state(consumer: Any, optimizer: torch.optim.Optimizer, history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parameters = capacity._parameters(consumer)
    return {
        "phase": parameters["generator.phase"].detach().cpu().clone(),
        "optimizer": copy.deepcopy(capacity._cpu_tree(optimizer.state_dict())),
        "history": copy.deepcopy(list(history)),
    }


def _checkpoint_payload(
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    inputs: Mapping[str, Any],
    arm: str,
    local_completed: int,
    history: Sequence[Mapping[str, Any]],
    *,
    reason: str,
) -> dict[str, Any]:
    parameters = capacity._parameters(consumer)
    return {
        "schema": CHECKPOINT_SCHEMA,
        "task_id": TASK_ID,
        "arm": arm,
        "source_step": SOURCE_STEP,
        "source_history_count": SOURCE_STEP,
        "completed_updates": SOURCE_STEP + local_completed,
        "task_local_updates": local_completed,
        "optimizer_step": int(float(optimizer.state[parameters["generator.phase"]]["step"].item())),
        "next_batch_cursor": local_completed % BATCH_COUNT,
        "optimizer_config": {"lr": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "history_reset": False},
        "numeric_mode": NUMERIC_MODE,
        "lambda": FIXED_LAMBDA,
        "source_checkpoint": inputs["checkpoint_path"].as_posix(),
        "source_population": inputs["population"],
        "trainable_parameter_state": {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()},
        "optimizer_state": copy.deepcopy(capacity._cpu_tree(optimizer.state_dict())),
        "frozen_runtime_state": capacity._cpu_tree(capacity._frozen_runtime_state(consumer)),
        "rng_state": capacity._rng_state(),
        "source_rng_state": inputs["generation_inputs"]["source"]["rng_state"],
        "training_history": copy.deepcopy(list(history)),
        "checkpoint_reason": reason,
    }


def _checkpoint_path(target: Path, arm: str, local_completed: int) -> Path:
    return target / f"arm_{arm.lower()}_step{SOURCE_STEP + local_completed:04d}.pt"


def _save_checkpoint(
    target: Path,
    consumer: Any,
    optimizer: torch.optim.Optimizer,
    inputs: Mapping[str, Any],
    arm: str,
    local_completed: int,
    history: Sequence[Mapping[str, Any]],
    *,
    reason: str,
) -> Path:
    payload = _checkpoint_payload(consumer, optimizer, inputs, arm, local_completed, history, reason=reason)
    path = _checkpoint_path(target, arm, local_completed)
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        _require(existing.get("task_id") == TASK_ID and existing.get("arm") == arm and existing.get("task_local_updates") == local_completed, "checkpoint path is occupied by different training evidence")
        _require(all(_tree_equal(existing.get(key), payload.get(key)) for key in ("trainable_parameter_state", "optimizer_state", "training_history", "rng_state")), "existing checkpoint state differs from the recomputed training boundary")
    else:
        _write_torch(path, payload)
    _write_json(target / f"training_history_arm_{arm.lower()}.json", {"task_id": TASK_ID, "arm": arm, "additional_updates": local_completed, "records": list(history)}, replace=True)
    return path


def _validate_checkpoint(payload: Mapping[str, Any], path: Path, target: Path, inputs: Mapping[str, Any]) -> None:
    _require(path.is_file() and target.resolve() in path.resolve().parents, "resume checkpoint must be an explicit file inside this task output")
    _require(payload.get("schema") == CHECKPOINT_SCHEMA and payload.get("task_id") == TASK_ID, "resume checkpoint belongs to another task")
    _require(payload.get("source_checkpoint") == inputs["checkpoint_path"].as_posix() and payload.get("source_population") == inputs["population"], "resume checkpoint source identity/population differs")
    _require(payload.get("source_step") == SOURCE_STEP and payload.get("lambda") == FIXED_LAMBDA and payload.get("numeric_mode") == NUMERIC_MODE, "resume checkpoint source step, weight, or numeric mode differs")
    _require(payload.get("source_history_count") == SOURCE_STEP, "resume checkpoint source training-history count differs")
    arm = payload.get("arm")
    local = payload.get("task_local_updates")
    _require(arm in {"A", "B"} and type(local) is int and 0 <= local <= ADDED_UPDATES, "resume arm/update count is invalid")
    _require(payload.get("completed_updates") == SOURCE_STEP + local and payload.get("optimizer_step") == SOURCE_STEP + local and payload.get("next_batch_cursor") == local % BATCH_COUNT, "resume update count, Adam step or batch cursor differs")
    _require(path.name == _checkpoint_path(target, arm, local).name, "resume checkpoint filename does not match its arm and update count")
    history = payload.get("training_history", [])
    _require(len(history) == local and all(row.get("task_local_update") == index + 1 and row.get("optimizer_step") == SOURCE_STEP + index + 1 and row.get("arm") == arm for index, row in enumerate(history)), "resume training history differs from its update count or arm")
    params = payload.get("trainable_parameter_state", {})
    expected_shapes = inputs["population"]["source_parameter_shapes"]
    _require(set(params) == set(expected_shapes), "resume phase state is incomplete")
    for name, shape in expected_shapes.items():
        value = params[name]
        _require(tuple(value.shape) == tuple(shape) and value.dtype == torch.float32 and bool(torch.isfinite(value).all()), f"resume parameter state is invalid for {name}")
    optimizer = payload.get("optimizer_state", {})
    groups, states = optimizer.get("param_groups", []), optimizer.get("state", {})
    _require(len(groups) == 1 and groups[0].get("params") == [0] and set(states) == {0}, "resume Adam parameter mapping differs")
    group = groups[0]
    _require(float(group.get("lr", -1)) == 1e-3 and tuple(group.get("betas", ())) == (0.9, 0.999) and float(group.get("eps", -1)) == 1e-8 and float(group.get("weight_decay", -1)) == 0.0, "resume Adam configuration differs")
    state = states[0]
    _require(int(float(state["step"].item())) == SOURCE_STEP + local and tuple(state["exp_avg"].shape) == (340_000,) and tuple(state["exp_avg_sq"].shape) == (340_000,), "resume Adam step or moment geometry differs")
    _require(bool(torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()), "resume Adam moments are nonfinite")
    _require(_tree_equal(payload.get("source_rng_state"), inputs["generation_inputs"]["source"]["rng_state"]), "resume checkpoint names another source RNG state")


def _restore_checkpoint(payload: Mapping[str, Any], consumer: Any, optimizer: torch.optim.Optimizer) -> None:
    parameters = capacity._parameters(consumer)
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(payload["trainable_parameter_state"][name].to(device=parameter.device, dtype=parameter.dtype))
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == payload["optimizer_step"], "restored Adam step differs")
    capacity._assert_state_equal(payload["frozen_runtime_state"], capacity._frozen_runtime_state(consumer), "restored frozen OLS/runtime state")
    capacity._restore_rng_state(payload["rng_state"])


def _pilot(inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module], arm: str, *, qfg_lambda: float | None = None) -> dict[str, Any]:
    device = torch.device("cuda:0")
    consumer, optimizer, parameters = parent_task._new_endpoint(inputs["generation_inputs"], device)
    capacity._restore_rng_state(inputs["generation_inputs"]["source"]["rng_state"])
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    frozen_before = capacity._frozen_runtime_state(consumer)
    history = [_train_step(consumer, optimizer, parameters, evaluators, inputs, arm, index, qfg_lambda=qfg_lambda) for index in range(3)]
    _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), f"preflight {arm} changed the frozen Encoder")
    capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), f"preflight {arm} frozen OLS/runtime")
    _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == SOURCE_STEP + 3, f"preflight {arm} did not execute three real Adam updates")
    state = _training_state(consumer, optimizer, history)
    with torch.no_grad():
        result = consumer.runtime.forward_from_image(inputs["generation_inputs"]["images"][FIT_ROWS : FIT_ROWS + 1].to(device=device, dtype=torch.float32))
        view = diagnostic.canonical.cmos_view(result.raw_intensity)
        logits = capacity._evaluator_logits(evaluators["E_D"], view, "E_D")
        representative = {"raw": result.raw_intensity.detach().cpu(), "view": view.detach().cpu(), "e_d_logits": logits.detach().cpu()}
    del consumer, optimizer, parameters
    torch.cuda.empty_cache()
    return {"history": history, "state": state, "representative_readout": representative}


def _normalized_state(state: Mapping[str, Any]) -> dict[str, Any]:
    history = copy.deepcopy(state["history"])
    for row in history:
        row.pop("arm", None)
    return {"phase": state["phase"], "optimizer": state["optimizer"], "history": history}


def _recovery_check(inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module], target: Path, arm: str, continuous: Mapping[str, Any]) -> dict[str, Any]:
    device = torch.device("cuda:0")
    consumer_one, optimizer_one, parameters_one = parent_task._new_endpoint(inputs["generation_inputs"], device)
    capacity._restore_rng_state(inputs["generation_inputs"]["source"]["rng_state"])
    history_one = [_train_step(consumer_one, optimizer_one, parameters_one, evaluators, inputs, arm, 0)]
    payload = _checkpoint_payload(consumer_one, optimizer_one, inputs, arm, 1, history_one, reason="preflight_one_step_recovery_control")
    recovery_path = target / "preflight" / f"arm_{arm.lower()}_step{SOURCE_STEP + 1}.pt"
    if recovery_path.exists():
        existing = torch.load(recovery_path, map_location="cpu", weights_only=True)
        old_state = {"phase": existing["trainable_parameter_state"]["generator.phase"], "optimizer": existing["optimizer_state"], "history": existing["training_history"]}
        new_state = {"phase": payload["trainable_parameter_state"]["generator.phase"], "optimizer": payload["optimizer_state"], "history": payload["training_history"]}
        saved_recheck = stroke._trial_repeatability(_normalized_state(old_state), _normalized_state(new_state))
        _require(saved_recheck["passed"] and _tree_equal(existing.get("rng_state"), payload.get("rng_state")), "saved preflight recovery checkpoint differs on tolerance-based recheck")
    else:
        _write_torch(recovery_path, payload)
    _validate_checkpoint(payload, recovery_path, target, inputs)
    del consumer_one, optimizer_one, parameters_one
    torch.cuda.empty_cache()

    resumed_consumer, resumed_optimizer, resumed_parameters = parent_task._new_endpoint(inputs["generation_inputs"], device)
    saved = torch.load(recovery_path, map_location="cpu", weights_only=True)
    _require(saved.get("task_id") == TASK_ID and saved.get("arm") == arm and saved.get("task_local_updates") == 1, "preflight recovery checkpoint identity differs")
    _restore_checkpoint(saved, resumed_consumer, resumed_optimizer)
    resumed_history = list(saved["training_history"])
    resumed_history.extend(_train_step(resumed_consumer, resumed_optimizer, resumed_parameters, evaluators, inputs, arm, index) for index in range(1, 3))
    resumed_state = _training_state(resumed_consumer, resumed_optimizer, resumed_history)
    comparison = stroke._trial_repeatability(_normalized_state(continuous["state"]), _normalized_state(resumed_state))
    _require(comparison["passed"], f"{arm} one-step checkpoint recovery differs from three continuous updates: {comparison}")
    return {"executed": True, "passed": True, "arm": arm, "checkpoint": recovery_path.relative_to(target).as_posix(), "continuous_updates": 3, "resumed_updates": 2, "comparison": comparison}


def _preflight(inputs: Mapping[str, Any], target: Path) -> dict[str, Any]:
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is unavailable for the authorized production preflight")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    disk_free = shutil.disk_usage(target).free
    expected_readout_bytes = (1 + 2 * 3) * TOTAL_ROWS * (200 * 200 * 4 + 28 * 28 * 4 + 10 * 4 + 200 * 200 * 4)
    expected_checkpoint_bytes = 16 * 8_000_000
    _require(free_bytes > 0 and total_bytes > 0, "CUDA reported no usable device memory")
    _require(disk_free > expected_readout_bytes + expected_checkpoint_bytes, "observed free disk space is insufficient for planned readouts and checkpoints")
    dll = gradient._runtime_dll_probe()
    backend = gradient._cuda_backend_probe(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    evaluators = generation.diagnostic.load_evaluators(inputs["generation_paths"], device)
    evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
    source_rng = copy.deepcopy(inputs["generation_inputs"]["source"]["rng_state"])
    checks: dict[str, Any] = {}
    try:
        consumer, optimizer, parameters = parent_task._new_endpoint(inputs["generation_inputs"], device)
        _require(consumer.runtime.config.propagation_grid_size == 2_048, "production optical grid differs from A4700")
        _require(int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == SOURCE_STEP, "loaded A4700 Adam step differs")
        _require(not optimizer.state[parameters["encoder.phase"]], "Encoder phase unexpectedly has Adam history")
        fit_images, fit_labels = _fit_batch(inputs, 0, device)
        with torch.no_grad():
            replay = consumer.runtime.forward_from_image(fit_images)
            saved = inputs["readout"]
            raw = replay.raw_intensity.detach().cpu()
            view = diagnostic.canonical.cmos_view(replay.raw_intensity).detach().cpu()
            target_images = replay.image_input.slm_intensity.detach().cpu()
        _require(torch.allclose(raw, saved["raw_cmos"][:BATCH_SIZE], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "A4700 baseline raw does not replay against the saved ten-row microbatch")
        _require(torch.allclose(view, saved["evaluator_view"][:BATCH_SIZE], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "A4700 production view does not replay against saved evidence")
        _require(torch.allclose(target_images, saved["target_slm_intensity"][:BATCH_SIZE], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), "A4700 raw target differs from the saved ten-row microbatch")
        checks["saved_a4700_ten_row_replay"] = {"executed": True, "passed": True, "rows": BATCH_SIZE, "rtol": REPLAY_RTOL, "atol": REPLAY_ATOL}
        del replay, raw, view, target_images

        base, _reconstruction, qfg, details = gradient._single_microbatch_batch_gradients(consumer, parameters, evaluators, fit_images, fit_labels)
        _powers, _losses, _gradients, production = stroke._logical_batch_gradients(consumer, parameters, evaluators, fit_images, fit_labels, structural=True)
        production_gradient = production["generator.phase"]
        _require(torch.allclose(base, production_gradient, rtol=GRADIENT_RTOL, atol=GRADIENT_ATOL), "single-row accumulated A gradient differs from existing production logical-batch gradient")
        checks["single_row_production_gradient"] = {"executed": True, "passed": True, "rtol": GRADIENT_RTOL, "atol": GRADIENT_ATOL, "max_abs_difference": float((base.to(torch.float64) - production_gradient.to(torch.float64)).abs().max().cpu())}
        _require(float(details["g_q_rms"]) > 0.0 and gradient._rms(qfg) > 0.0, "Q_fg candidate gradient is zero")

        fd_records = []
        image = inputs["generation_inputs"]["images"][0:1].to(device=device, dtype=torch.float32)
        source_phase = parameters["generator.phase"].detach().clone()
        qfg_value, qfg_phase_gradient = gradient._qfg_phase_gradient(consumer, parameters, image)
        direction = gradient._unit_descent(qfg_phase_gradient, "preflight Q_fg")
        for epsilon in (FD_EPSILON, FD_EPSILON / 2):
            fd_records.append(gradient._check_fd(consumer, parameters, image, direction, epsilon=epsilon))
        _require(torch.equal(parameters["generator.phase"].detach(), source_phase), "Q_fg phase perturbation failed to restore A4700")
        checks["phase_restored_after_qfg_probe"] = {"executed": True, "passed": True, "parameter_gradients_empty": all(parameter.grad is None for parameter in parameters.values())}
        checks["qfg_image_derivative_and_phase_vjp"] = {"executed": True, "passed": all(record["passed"] for record in fd_records), "qfg_at_source": qfg_value, "epsilon_checks": fd_records, "epsilon_rad_rms": FD_EPSILON, "rtol": FD_RTOL, "atol": FD_ATOL}
        _require(checks["qfg_image_derivative_and_phase_vjp"]["passed"], "Q_fg derivative or phase VJP central-difference check failed")
        del consumer, optimizer, parameters
        capacity._restore_rng_state(source_rng)

        pilot_a = _pilot(inputs, evaluators, "A")
        checks["arm_a_three_real_updates"] = {"executed": True, "passed": True, "optimizer_step": SOURCE_STEP + 3, "generator_changed": True}
        pilot_b = _pilot(inputs, evaluators, "B")
        checks["arm_b_three_real_updates"] = {"executed": True, "passed": True, "optimizer_step": SOURCE_STEP + 3, "generator_changed": True, "effective_lambda": FIXED_LAMBDA}
        recovery_a = _recovery_check(inputs, evaluators, target, "A", pilot_a)
        recovery_b = _recovery_check(inputs, evaluators, target, "B", pilot_b)
        checks["save_restore_one_plus_two_updates"] = {"executed": True, "passed": True, "arms": {"A": recovery_a, "B": recovery_b}}

        zero_b = _pilot(inputs, evaluators, "B", qfg_lambda=0.0)
        zero_control = stroke._trial_repeatability(_normalized_state(pilot_a["state"]), _normalized_state(zero_b["state"]))
        zero_readout_a = pilot_a["representative_readout"]
        zero_readout_b = zero_b["representative_readout"]
        view_equal = torch.allclose(zero_readout_a["view"], zero_readout_b["view"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
        logits_equal = torch.allclose(zero_readout_a["e_d_logits"], zero_readout_b["e_d_logits"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
        raw_difference = float((zero_readout_a["raw"].to(torch.float64) - zero_readout_b["raw"].to(torch.float64)).abs().max())
        checks["zero_lambda_negative_control"] = {"executed": True, "passed": zero_control["passed"] and view_equal, "lambda": 0.0, "comparison": zero_control, "representative_production_view_equal": bool(view_equal), "e_d_logits_equal_observation": bool(logits_equal), "raw_readout_max_abs_difference_observation": raw_difference}
        _require(checks["zero_lambda_negative_control"]["passed"], "zero-weight candidate does not reproduce the A-arm updates and representative readout")
        for pilot in (pilot_a, pilot_b, zero_b):
            del pilot
        del pilot_a, pilot_b, zero_b
        torch.cuda.synchronize(device)
        evaluator_unchanged = all(_tree_equal(evaluator_before[name], capacity._state_snapshot(model)) for name, model in evaluators.items())
        checks["frozen_evaluators_and_empty_gradients"] = {"executed": True, "passed": evaluator_unchanged and all(parameter.grad is None for model in evaluators.values() for parameter in model.parameters()), "evaluators_unchanged": evaluator_unchanged}
        _require(checks["frozen_evaluators_and_empty_gradients"]["passed"], "frozen evaluator state changed during preflight")
        checks["cuda_and_disk"] = {"executed": True, "passed": True, "device": str(device), "cuda_free_bytes_observation": int(free_bytes), "cuda_total_bytes_observation": int(total_bytes), "disk_free_bytes_observation": int(disk_free), "estimated_readout_bytes": expected_readout_bytes, "estimated_checkpoint_bytes": expected_checkpoint_bytes}
        checks["strict_fp32_backend"] = {"executed": True, "passed": bool(backend["passed"]), "runtime_dll": dll, "backend": backend, "tf32_disabled": True}
        _require(checks["strict_fp32_backend"]["passed"], "strict CUDA FP32 backend preflight failed")
        _require(all(item.get("executed") is True and item.get("passed") is True for item in checks.values()), "preflight contains a missing or failed check")
        return {"task_id": TASK_ID, "status": "READY", "checks": checks, "fixed_lambda": FIXED_LAMBDA, "formal_measurement_started": False, "numeric_mode": NUMERIC_MODE}
    finally:
        capacity._restore_rng_state(source_rng)
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"preflight evaluator {name}")
        torch.cuda.empty_cache()


def _make_identity(inputs: Mapping[str, Any]) -> ResearchRunIdentity:
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=f"{TASK_ID}-attempt-01",
        trajectory_id="A4700-original-objective-vs-fixed-foreground-term-700-updates",
        semantic_authority_id=generation.TASK_ID,
        data_role="canonical TRAIN fit=[0,200); confirmation=[200,500); confirmation excluded from updates and endpoint selection",
        config_identity={"source_step": SOURCE_STEP, "updates_per_arm": ADDED_UPDATES, "lambda": FIXED_LAMBDA, "logical_batch": BATCH_SIZE, "batch_count": BATCH_COUNT, "measurement_steps": list(MEASUREMENT_STEPS), "strict_fp32_optics": True, "qfg_statistics": "float64"},
        model_identity={"source_checkpoint": inputs["checkpoint_path"].as_posix(), "source_checkpoint_step": SOURCE_STEP, "generator_parameter_shapes": inputs["population"]["source_parameter_shapes"], "finite": True},
        optimizer_identity={"class": "Adam", "parameter_names": ["generator.phase"], "source_step": SOURCE_STEP, "lr": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "history_reset": False},
        sampler_identity={"fit_source_rows": inputs["population"]["source_row_indices"][:FIT_ROWS], "fit_batch_count": BATCH_COUNT, "confirmation_source_rows": inputs["population"]["source_row_indices"][FIT_ROWS:]},
        execution_source_semantics_id=TASK_ID,
        scientific_attempt_id=f"{TASK_ID}:attempt-01",
    )


def _make_receipt(inputs: Mapping[str, Any], preflight: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt]:
    checks = preflight.get("checks", {})
    _require(checks and all(row.get("executed") is True and row.get("passed") is True for row in checks.values()), "READY receipt requires every executed preflight check to pass")
    control = ResearchRunControl(_make_identity(inputs), authorization_id=f"{TASK_ID}:authorization:01")
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "status": "PASS", "source_and_gradient_lineage": True, "saved_A4700_replay": checks["saved_a4700_ten_row_replay"]["passed"], "production_gradient_match": checks["single_row_production_gradient"]["passed"], "qfg_derivative_vjp": checks["qfg_image_derivative_and_phase_vjp"]["passed"], "phase_restoration": checks["phase_restored_after_qfg_probe"]["passed"], "three_updates_each_arm": checks["arm_a_three_real_updates"]["passed"] and checks["arm_b_three_real_updates"]["passed"], "checkpoint_recovery": checks["save_restore_one_plus_two_updates"]["passed"], "zero_lambda_control": checks["zero_lambda_negative_control"]["passed"], "evaluator_isolation": checks["frozen_evaluators_and_empty_gradients"]["passed"]},
        resource={"passed": checks["cuda_and_disk"]["passed"], "status": "PASS", "cuda": checks["cuda_and_disk"]},
        implementation={"passed": checks["strict_fp32_backend"]["passed"], "status": "PASS", "selected_engine": "cuda_2048_single_row_fp32_optics", "checkpoint_recovery": True, "fixed_lambda": inputs["gradient_audit"]["lambda"]},
        selected_engine="cuda_2048_single_row_fp32_optics",
        resource_envelope={"cuda_and_disk": checks["cuda_and_disk"]},
    )
    return control, receipt


def _stage_ready(target: Path, inputs: Mapping[str, Any], preflight: Mapping[str, Any], receipt: PreflightReceipt, *, resume: bool) -> None:
    expected_preflight = {"task_id": TASK_ID, "status": "READY", "receipt": receipt.to_dict(), "preflight": dict(preflight)}
    if resume:
        _require(_json(target / "input_population.json") == inputs["population"], "saved READY population differs")
        _require(_json(target / "preflight.json") == expected_preflight, "saved READY preflight or receipt differs")
        return
    _write_json(target / "input_population.json", inputs["population"])
    _write_json(target / "fixed_parameters.json", {"task_id": TASK_ID, "lambda": FIXED_LAMBDA, "source_step": SOURCE_STEP, "updates_per_arm": ADDED_UPDATES, "adam": {"lr": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0}, "measurement_steps": list(MEASUREMENT_STEPS)})
    _write_json(target / "preflight.json", expected_preflight)
    previous_failure = target / "invalid_harness.json"
    if previous_failure.is_file():
        preserved_failure = target / "engineering_preflight_failure_01.json"
        _require(not preserved_failure.exists(), "preserved preflight failure record already exists")
        os.replace(previous_failure, preserved_failure)


def _validate_ready(target: Path, inputs: Mapping[str, Any]) -> tuple[ResearchRunControl, PreflightReceipt, dict[str, Any]]:
    payload = _json(target / "preflight.json")
    _require(payload.get("task_id") == TASK_ID and payload.get("status") == "READY", "saved preflight is not READY for this task")
    _require(_json(target / "input_population.json") == inputs["population"], "input population changed after preflight")
    preflight = payload["preflight"]
    _require(preflight.get("status") == "READY" and all(value.get("executed") is True and value.get("passed") is True for value in preflight.get("checks", {}).values()), "READY consumer found a missing or failed check")
    control = ResearchRunControl(_make_identity(inputs), authorization_id=f"{TASK_ID}:authorization:01")
    receipt = PreflightReceipt.from_mapping(payload["receipt"])
    _require(receipt.source_semantics_id == TASK_ID and receipt.authorization_id == f"{TASK_ID}:authorization:01" and receipt.status == "READY", "saved preflight receipt identity/status differs")
    _require(all(group.get("passed") is True for group in receipt.checks.values()), "saved receipt has a failed or missing check group")
    control.mark_preflight_ready(receipt)
    return control, receipt, preflight


def _scenario(arm: str, local: int) -> str:
    return "A4700" if local == 0 else f"{arm}{SOURCE_STEP + local}"


def _row_metrics_from_readout(readout: Mapping[str, torch.Tensor], inputs: Mapping[str, Any], scenario: str) -> list[dict[str, Any]]:
    _summary, production_metrics = parent_task._energy_image_metrics(
        {
            "raw_cmos": readout["raw_cmos"],
            "evaluator_view": readout["evaluator_view"],
            "target_slm_intensity": readout["target_slm_intensity"],
            "labels": inputs["labels"],
            "plane_metadata": inputs["readout"]["plane_metadata"],
        },
        {"population": inputs["population"]},
    )
    production_q = production_metrics["brightness_normalized_stroke_error_Q"]
    _require(bool(torch.isfinite(production_q).all()), f"production Q is undefined for {scenario}")
    rows = []
    for position in range(TOTAL_ROWS):
        row = gradient._metric_row(
            readout["raw_cmos"][position : position + 1],
            readout["target_slm_intensity"][position : position + 1],
            readout["evaluator_view"][position : position + 1],
            readout["e_d_logits"][position : position + 1],
            scenario=scenario,
            position=position,
            source_row=int(inputs["population"]["source_row_indices"][position]),
            label=int(inputs["labels"][position]),
            role=inputs["roles"][position],
            pixel_area_m2=float(inputs["pixel_area_m2"]),
        )
        row["production_q_28"] = float(production_q[position])
        row["clipped_fraction"] = float(((readout["raw_cmos"][position : position + 1] < 0.0) | (readout["raw_cmos"][position : position + 1] > 1.0)).to(torch.float64).mean())
        rows.append(row)
    return rows


def _save_readout_chunks(target: Path, scenario: str, readout: Mapping[str, torch.Tensor], inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenario_dir = target / "readouts" / scenario
    rows: list[dict[str, Any]] = []
    for chunk_number, start in enumerate(range(0, TOTAL_ROWS, CHUNK_SIZE)):
        stop = min(start + CHUNK_SIZE, TOTAL_ROWS)
        path = scenario_dir / f"chunk_{chunk_number:04d}.pt"
        payload = {
            "task_id": TASK_ID,
            "scenario": scenario,
            "positions": torch.arange(start, stop, dtype=torch.long),
            "source_row_indices": torch.tensor(inputs["population"]["source_row_indices"][start:stop], dtype=torch.long),
            "labels": inputs["labels"][start:stop].clone(),
            "roles": list(inputs["roles"][start:stop]),
            **{key: readout[key][start:stop].detach().cpu().contiguous() for key in ("raw_cmos", "evaluator_view", "target_slm_intensity", "e_d_logits")},
        }
        if path.exists():
            existing = torch.load(path, map_location="cpu", weights_only=True)
            _require(existing.get("task_id") == TASK_ID and existing.get("scenario") == scenario and torch.equal(existing["positions"], payload["positions"]), f"existing readout chunk identity differs: {path.name}")
            for key in ("source_row_indices", "labels", "raw_cmos", "evaluator_view", "target_slm_intensity", "e_d_logits"):
                _require(torch.equal(existing[key], payload[key]), f"existing readout chunk does not match replay: {scenario}/{key}/{chunk_number}")
        else:
            _write_torch(path, payload)
    rows = _row_metrics_from_readout(readout, inputs, scenario)
    _write_csv(scenario_dir / "paired_rows.csv", rows, METRIC_FIELDS)
    return rows


def _measure_source(target: Path, inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = inputs["readout"]
    readout = {key: source[key].detach().cpu() for key in ("raw_cmos", "evaluator_view", "target_slm_intensity", "e_d_logits")}
    rows = _save_readout_chunks(target, "A4700", readout, inputs)
    return rows


def _measure_live(target: Path, inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module], arm: str, local: int, consumer: Any) -> list[dict[str, Any]]:
    scenario = _scenario(arm, local)
    scenario_dir = target / "readouts" / scenario
    if (scenario_dir / "paired_rows.csv").is_file():
        rows = _read_csv(scenario_dir / "paired_rows.csv")
        _require(len(rows) == TOTAL_ROWS, f"existing endpoint row count differs for {scenario}")
        return [{key: (int(value) if key in {"position", "source_row_index", "label", "e_d_prediction"} else float(value) if key not in {"scenario", "role"} else value) for key, value in row.items()} for row in rows]
    rng_before = capacity._rng_state()
    payload_parts: dict[str, list[torch.Tensor]] = {key: [] for key in ("raw_cmos", "evaluator_view", "target_slm_intensity", "e_d_logits")}
    try:
        with torch.no_grad():
            for start in range(0, TOTAL_ROWS, CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, TOTAL_ROWS)
                for position in range(start, stop):
                    image = inputs["generation_inputs"]["images"][position : position + 1].to(device="cuda:0", dtype=torch.float32)
                    result = consumer.runtime.forward_from_image(image)
                    view = diagnostic.canonical.cmos_view(result.raw_intensity)
                    logits = capacity._evaluator_logits(evaluators["E_D"], view, "E_D")
                    payload_parts["raw_cmos"].append(result.raw_intensity.detach().cpu())
                    payload_parts["evaluator_view"].append(view.detach().cpu())
                    payload_parts["target_slm_intensity"].append(result.image_input.slm_intensity.detach().cpu())
                    payload_parts["e_d_logits"].append(logits.detach().cpu())
                    del result, view, logits, image
                torch.cuda.synchronize()
        readout = {key: torch.cat(values, dim=0) for key, values in payload_parts.items()}
        rows = _save_readout_chunks(target, scenario, readout, inputs)
    finally:
        capacity._restore_rng_state(rng_before)
        torch.cuda.empty_cache()
    return rows


def _training_checkpoint(target: Path, arm: str, local: int) -> dict[str, Any]:
    path = _checkpoint_path(target, arm, local)
    _require(path.is_file(), f"measurement endpoint checkpoint is missing: {path.name}")
    return torch.load(path, map_location="cpu", weights_only=True)


def _run_arm(target: Path, inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module], arm: str, start_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    consumer, optimizer, parameters = parent_task._new_endpoint(inputs["generation_inputs"], torch.device("cuda:0"))
    if start_payload is None:
        local = 0
        history: list[dict[str, Any]] = []
        capacity._restore_rng_state(inputs["generation_inputs"]["source"]["rng_state"])
        _save_checkpoint(target, consumer, optimizer, inputs, arm, 0, history, reason="common_A4700_source_state")
    else:
        _restore_checkpoint(start_payload, consumer, optimizer)
        local = int(start_payload["task_local_updates"])
        history = copy.deepcopy(list(start_payload["training_history"]))
    _require(len(history) == local, f"{arm} resumed history length differs from its cursor")
    encoder_before = parameters["encoder.phase"].detach().cpu().clone()
    frozen_before = capacity._frozen_runtime_state(consumer)
    signal_state = {"stop": False}
    old_handler = signal.getsignal(signal.SIGINT)

    def request_stop(_signum, _frame):
        signal_state["stop"] = True

    signal.signal(signal.SIGINT, request_stop)
    try:
        if local in MEASUREMENT_STEPS:
            _measure_live(target, inputs, evaluators, arm, local, consumer)
        while local < ADDED_UPDATES:
            record = _train_step(consumer, optimizer, parameters, evaluators, inputs, arm, local)
            history.append(record)
            local += 1
            if local % 100 == 0 or signal_state["stop"]:
                _save_checkpoint(target, consumer, optimizer, inputs, arm, local, history, reason="interrupt_boundary" if signal_state["stop"] else "fixed_100_step_boundary")
                _write_json(target / "progress.json", {"task_id": TASK_ID, "status": "RUNNING", "arm": arm, "task_local_updates": local, "optimizer_step": SOURCE_STEP + local}, replace=True)
                if local in MEASUREMENT_STEPS:
                    _measure_live(target, inputs, evaluators, arm, local, consumer)
                if signal_state["stop"]:
                    raise TrainingInterrupted(f"{arm} checkpointed at task-local update {local}")
        _require(local == ADDED_UPDATES and int(float(optimizer.state[parameters["generator.phase"]]["step"].item())) == FINAL_STEP, f"{arm} failed to complete exactly 700 updates")
        _require(torch.equal(encoder_before, parameters["encoder.phase"].detach().cpu()), f"{arm} Encoder changed during training")
        capacity._assert_state_equal(frozen_before, capacity._frozen_runtime_state(consumer), f"{arm} frozen OLS/runtime")
        payload = _training_checkpoint(target, arm, local)
        _require(len(history) == ADDED_UPDATES, f"{arm} final history length differs")
        return payload
    finally:
        signal.signal(signal.SIGINT, old_handler)
        del consumer, optimizer
        torch.cuda.empty_cache()


def _run_training(target: Path, inputs: Mapping[str, Any], evaluators: Mapping[str, nn.Module], selected: Mapping[str, Any] | None) -> dict[str, Any]:
    if selected is None:
        final_a = _run_arm(target, inputs, evaluators, "A", None)
        b_start = _training_checkpoint(target, "B", 0) if _checkpoint_path(target, "B", 0).is_file() else None
        if b_start is not None:
            _validate_checkpoint(b_start, _checkpoint_path(target, "B", 0), target, inputs)
        final_b = _run_arm(target, inputs, evaluators, "B", b_start)
    elif selected["arm"] == "A":
        _require(selected["task_local_updates"] < ADDED_UPDATES or _checkpoint_path(target, "B", 0).is_file(), "completed A resume has no shared source for B")
        final_a = _run_arm(target, inputs, evaluators, "A", selected)
        b_path = _checkpoint_path(target, "B", 0)
        b_start = torch.load(b_path, map_location="cpu", weights_only=True) if b_path.is_file() else None
        final_b = _run_arm(target, inputs, evaluators, "B", b_start)
    else:
        _require(_checkpoint_path(target, "A", ADDED_UPDATES).is_file(), "B resume requires the completed A5400 control checkpoint")
        final_a = _training_checkpoint(target, "A", ADDED_UPDATES)
        final_b = _run_arm(target, inputs, evaluators, "B", selected)
    for arm, payload in (("A", final_a), ("B", final_b)):
        _validate_checkpoint(payload, _checkpoint_path(target, arm, ADDED_UPDATES), target, inputs)
    _require(final_a["optimizer_step"] == final_b["optimizer_step"] == FINAL_STEP, "both arms must finish at Adam step5400")
    return {"A": final_a, "B": final_b}


def _summaries(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fields = ("q_fg", "q_bg", "q_raw200", "raw_mse", "raw_mse_fg", "raw_mse_bg", "production_q_28", "ssim_28", "integral_intensity", "clipped_fraction")
    scenarios = ["A4700", *(f"A{SOURCE_STEP + step}" for step in (100, 300, 700)), *(f"B{SOURCE_STEP + step}" for step in (100, 300, 700))]
    summary_rows = []
    for scenario in scenarios:
        for role in ("fit", "confirmation"):
            for label in range(10):
                group = [row for row in rows if row["scenario"] == scenario and row["role"] == role and int(row["label"]) == label]
                expected = 20 if role == "fit" else 30
                _require(len(group) == expected, f"summary group has {len(group)} rows, expected {expected}: {scenario}/{role}/{label}")
                record: dict[str, Any] = {"scenario": scenario, "role": role, "label": label, "rows": len(group)}
                for field in fields:
                    record[f"mean_{field}"] = math.fsum(float(row[field]) for row in group) / len(group)
                record["e_d_correct"] = sum(int(row["e_d_prediction"]) == label for row in group)
                summary_rows.append(record)
    return summary_rows, _decision(rows)


def _decision(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def focus_mean(scenario: str) -> tuple[float, dict[str, float]]:
        by_class = {}
        for label in FOCUS_CLASSES:
            group = [row for row in rows if row["scenario"] == scenario and row["role"] == "confirmation" and int(row["label"]) == label]
            _require(len(group) == 30, f"primary decision requires exactly 30 confirmation rows for {scenario}/class {label}")
            by_class[str(label)] = math.fsum(float(row["q_fg"]) for row in group) / 30.0
        return math.fsum(by_class.values()) / len(FOCUS_CLASSES), by_class

    a, a_classes = focus_mean("A5400")
    b, b_classes = focus_mean("B5400")
    start, start_classes = focus_mean("A4700")
    difference = b - a
    tolerance = DECISION_ATOL + DECISION_RTOL * max(abs(a), abs(b))
    if difference < -tolerance:
        classification = "FOREGROUND_TRAINING_RELATIVE_IMPROVEMENT"
        interpretation = "固定前景项臂在类别2、4确认人口的终点前景误差低于原目标对照臂；收益限于该冻结起点和追加700步。"
    elif difference > tolerance:
        classification = "FOREGROUND_TRAINING_RELATIVE_REGRESSION"
        interpretation = "固定前景项臂在类别2、4确认人口的终点前景误差高于原目标对照臂；不推进这一固定权重候选。"
    else:
        classification = "NO_DIFFERENCE_BEYOND_NUMERICAL_TOLERANCE"
        interpretation = "两臂终点前景误差差异未超过预设数值容差；这不证明两臂等效。"
    return {
        "classification": classification,
        "difference_B5400_minus_A5400": difference,
        "tolerance": tolerance,
        "mean_q_fg": {"A4700": start, "A5400": a, "B5400": b},
        "means_by_focus_class": {"A4700": start_classes, "A5400": a_classes, "B5400": b_classes},
        "change_from_A4700": {"A5400": a - start, "B5400": b - start},
        "candidate_absolutely_lower_than_start": b < start,
        "interpretation": interpretation if b < start else f"{interpretation} 候选臂相对A4700仍未降低时，只能解释为相对对照臂改善/较少恶化，不能称为绝对前景改善。",
        "primary_population": "confirmation, classes 2 and 4, 30 rows per class, class-balanced mean",
        "fit_population_used_for_primary": False,
    }


def _recheck_chunks(target: Path, inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    scenarios = ["A4700", *(f"A{SOURCE_STEP + step}" for step in (100, 300, 700)), *(f"B{SOURCE_STEP + step}" for step in (100, 300, 700))]
    all_rows = []
    for scenario in scenarios:
        scenario_dir = target / "readouts" / scenario
        saved_rows = _read_csv(scenario_dir / "paired_rows.csv")
        _require(len(saved_rows) == TOTAL_ROWS, f"saved paired-row count differs for {scenario}")
        row_by_position = {int(row["position"]): row for row in saved_rows}
        _require(set(row_by_position) == set(range(TOTAL_ROWS)), f"saved positions are missing or duplicated for {scenario}")
        for chunk_number, start in enumerate(range(0, TOTAL_ROWS, CHUNK_SIZE)):
            stop = min(start + CHUNK_SIZE, TOTAL_ROWS)
            chunk = torch.load(scenario_dir / f"chunk_{chunk_number:04d}.pt", map_location="cpu", weights_only=True)
            _require(chunk.get("task_id") == TASK_ID and chunk.get("scenario") == scenario, f"saved chunk belongs to another task/scenario: {scenario}")
            _require(torch.equal(chunk["positions"], torch.arange(start, stop)), f"saved chunk positions differ for {scenario}")
            _require(torch.equal(chunk["source_row_indices"], torch.tensor(inputs["population"]["source_row_indices"][start:stop], dtype=torch.long)), f"saved source row identities differ for {scenario}")
            _require(torch.equal(chunk["labels"], inputs["labels"][start:stop]) and chunk["roles"] == inputs["roles"][start:stop], f"saved labels/roles differ for {scenario}")
            replay_view = diagnostic.canonical.cmos_view(chunk["raw_cmos"])
            _require(torch.allclose(replay_view, chunk["evaluator_view"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL), f"saved view does not replay from raw for {scenario}")
            _require(bool(torch.isfinite(chunk["raw_cmos"]).all() and torch.isfinite(chunk["target_slm_intensity"]).all() and torch.isfinite(chunk["e_d_logits"]).all()), f"saved readout is nonfinite for {scenario}")
            _require(bool((chunk["raw_cmos"] >= 0).all() and (chunk["target_slm_intensity"] >= 0).all()), f"saved intensity is negative for {scenario}")
            for position in range(start, stop):
                record = row_by_position[position]
                _require(int(record["source_row_index"]) == inputs["population"]["source_row_indices"][position] and int(record["label"]) == int(inputs["labels"][position]) and record["role"] == inputs["roles"][position], f"CSV row identity differs from its readout chunk: {scenario}/{position}")
                chunk_role_positions = {"fit": [0, len(chunk["labels"])], "confirmation": [len(chunk["labels"]), len(chunk["labels"])]} if stop <= FIT_ROWS else {"fit": [0, 0], "confirmation": [0, len(chunk["labels"])]}
                chunk_metrics = parent_task._energy_image_metrics(
                    {
                        "raw_cmos": chunk["raw_cmos"],
                        "evaluator_view": replay_view,
                        "target_slm_intensity": chunk["target_slm_intensity"],
                        "labels": chunk["labels"],
                        "plane_metadata": inputs["readout"]["plane_metadata"],
                    },
                    {"population": {"role_positions": chunk_role_positions}},
                )[1]
                recomputed = gradient._metric_row(chunk["raw_cmos"][position - start : position - start + 1], chunk["target_slm_intensity"][position - start : position - start + 1], replay_view[position - start : position - start + 1], chunk["e_d_logits"][position - start : position - start + 1], scenario=scenario, position=position, source_row=int(inputs["population"]["source_row_indices"][position]), label=int(inputs["labels"][position]), role=inputs["roles"][position], pixel_area_m2=float(inputs["pixel_area_m2"]))
                recomputed["production_q_28"] = float(chunk_metrics["brightness_normalized_stroke_error_Q"][position - start])
                recomputed["clipped_fraction"] = float(((chunk["raw_cmos"][position - start : position - start + 1] < 0.0) | (chunk["raw_cmos"][position - start : position - start + 1] > 1.0)).to(torch.float64).mean())
                for field in METRIC_FIELDS:
                    if field in {"scenario", "role"}:
                        _require(record[field] == str(recomputed[field]), f"CSV {field} differs from independent recomputation for {scenario}/{position}")
                    elif field in {"position", "source_row_index", "label", "e_d_prediction"}:
                        _require(int(record[field]) == int(recomputed[field]), f"CSV {field} differs from independent recomputation for {scenario}/{position}")
                    else:
                        _require(math.isclose(float(record[field]), float(recomputed[field]), rel_tol=1e-11, abs_tol=1e-13), f"CSV {field} differs from independent recomputation for {scenario}/{position}")
                all_rows.append({key: (int(value) if key in {"position", "source_row_index", "label", "e_d_prediction"} else float(value) if key not in {"scenario", "role"} else value) for key, value in record.items()})
    _require(len(all_rows) == 7 * TOTAL_ROWS, "formal measurement did not produce exactly 3,500 paired rows")
    return all_rows


def _report_markdown(report: Mapping[str, Any]) -> str:
    decision = report["decision"]
    means = decision["mean_q_fg"]
    return "\n".join([
        "# A4700 前景目标单因素训练对照",
        "",
        "## 结果与状态",
        "",
        f"状态：`{report['status']}`；判读：`{decision['classification']}`。两臂各完成700次Adam更新，A4700共同起点为4700步，终点为A5400/B5400。正式读出为3500行；唯一主量差值为 `{decision['difference_B5400_minus_A5400']:.10g}`，数值容差 `{decision['tolerance']:.10g}`。",
        "",
        "## 科学认识",
        "",
        f"类别2、4确认人口等权前景误差：A4700 `{means['A4700']:.8g}`，原目标A5400 `{means['A5400']:.8g}`，增加固定前景项的B5400 `{means['B5400']:.8g}`。{decision['interpretation']}",
        "",
        f"主效应分类：`{decision['classification']}`。拟合人口只用于更新；确认人口未参与训练、调权或终点选择。所有来源行均属canonical TRAIN。",
        "",
        "## 目标缺口",
        "",
        "本实验只检验固定A4700起点、固定lambda和700步下的相对前景误差。它不证明人眼可读性、分类代理效度、独立泛化、模型采用或完整生成成功。若候选仍比A4700起点差，只能称相对对照较好或较少恶化。",
        "",
        "## 工程完成与验证",
        "",
        f"保存读出、配对表、检查点和预检位于 `{report['output_dir']}`。READY预检含保存基线重放、生产梯度匹配、Q_fg导数/VJP差分、两臂三步真实Adam更新、每臂一步保存后续两步恢复、零lambda负控及冻结评价器核验。最终逐行CSV与读出块已独立重算。正式更新总数为 `{report['scientific_updates']}`。",
        "",
        "## 决策影响与下一步",
        "",
        f"训练建议：`{report['training_recommendation']}`；后续动作：`{report['next_action']}`。本次结果只决定该固定前景目标是否保留为进一步验证候选；后续采样、人审和独立人口确认需单独规划与授权。",
        "",
    ])


def _finalize(target: Path, inputs: Mapping[str, Any], control: ResearchRunControl, preflight: Mapping[str, Any], started: float) -> dict[str, Any]:
    rows = _recheck_chunks(target, inputs)
    summary_rows, decision = _summaries(rows)
    independent_decision = _decision(rows)
    _require(decision == independent_decision, "primary decision changed on independent CSV recomputation")
    _write_csv(target / "paired_rows.csv", rows, METRIC_FIELDS, replace=True)
    summary_fields = tuple(summary_rows[0].keys())
    _write_csv(target / "class_role_summary.csv", summary_rows, summary_fields, replace=True)
    control.complete()
    report = {
        "schema": "lightgen_foreground_objective_training_comparison_report_v1",
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "classification": decision["classification"],
        "decision": decision,
        "source_checkpoint_step": SOURCE_STEP,
        "final_generator_adam_step": FINAL_STEP,
        "additional_updates_per_arm": ADDED_UPDATES,
        "scientific_updates": 2 * ADDED_UPDATES,
        "lambda": FIXED_LAMBDA,
        "fit_rows": FIT_ROWS,
        "confirmation_rows": CONFIRM_ROWS,
        "training_arm_objectives": {"A": "R = raw200 MSE + 0.1*(1-SSIM28)", "B": f"R + {FIXED_LAMBDA:.17g}*Q_fg; Q_fg uses whole-image intensity normalization and target>0.1 mask"},
        "measurement_steps": list(MEASUREMENT_STEPS),
        "paired_rows": len(rows),
        "endpoint_summaries": summary_rows,
        "gradient_audit_source": inputs["gradient_audit"],
        "preflight": preflight,
        "research_run_state": control.state,
        "training_recommendation": "NONE",
        "next_action": "NONE",
        "stop": True,
        "output_dir": target.as_posix(),
        "elapsed_seconds_observation_only": time.perf_counter() - started,
    }
    _write_json(target / "validation.json", {"task_id": TASK_ID, "status": "VALID", "rows": len(rows), "decision_recomputed_from_saved_chunks": independent_decision, "csv_matches_chunks": True, "all_roles_and_classes_present": True})
    _write_json(target / "report.json", report)
    (target / "report.md").write_text(_report_markdown(report), encoding="utf-8", newline="\n")
    _write_json(target / "completed.json", {"task_id": TASK_ID, "status": "COMPLETED", "classification": decision["classification"], "scientific_updates": 2 * ADDED_UPDATES, "training_recommendation": "NONE", "next_action": "NONE"})
    _write_json(target / "progress.json", {"task_id": TASK_ID, "status": "COMPLETED", "arm_A_updates": ADDED_UPDATES, "arm_B_updates": ADDED_UPDATES}, replace=True)
    return report


def _prepare_target(root: Path, target: Path, *, resume_checkpoint: Path | None) -> None:
    outputs = (root / "outputs").resolve()
    expected = (root / DEFAULT_OUTPUT).resolve()
    _require(target == expected, "output directory must be the dedicated attempt_01 path from the plan")
    _require(outputs in target.parents and not any(part in {"final_pt", "_wsl_restore_metadata_v4", ".tmp"} for part in target.relative_to(outputs).parts), "protected/restored/temporary output namespace is forbidden")
    if not target.exists():
        _require(resume_checkpoint is None, "explicit resume requires the existing task output directory")
        target.mkdir(parents=True, exist_ok=False)
        (target / "plan.md").write_text(_plan_text(), encoding="utf-8", newline="\n")
        return
    _require(target.is_dir(), "output path is not a directory")
    if (target / "report.json").is_file():
        _require(resume_checkpoint is None, "completed run does not accept a resume checkpoint")
        return
    if not (target / "preflight.json").is_file():
        _require(resume_checkpoint is None, "resume requires an existing READY preflight")
        _require((target / "plan.md").is_file(), "existing partial output has no task plan")
        _require(not list(target.glob("arm_[ab]_step*.pt")), "formal checkpoint exists without READY preflight evidence")
        return
    if resume_checkpoint is not None:
        _require(resume_checkpoint.is_file() and target in resume_checkpoint.parents, "resume checkpoint must be an explicit file inside this task output")


def _preserve_prior_failure(target: Path) -> None:
    prior = target / "invalid_harness.json"
    if not prior.is_file():
        return
    index = 1
    while (target / f"engineering_failure_{index:02d}.json").exists():
        index += 1
    os.replace(prior, target / f"engineering_failure_{index:02d}.json")


def _plan_text() -> str:
    return """# A4700前景目标单因素训练对照\n\n唯一问题：在同一A4700与Adam状态、固定700步下，加入已由200行拟合人口确定的lambda=0.11159457624315472之Q_fg项，是否降低类别2、4确认人口的终点前景形态误差？\n\nA臂沿用生产重建目标R；B臂为R+lambda Q_fg。两臂各700步，保留光功率项、BOGT权重与E_S/E_I/E_C规范化语义梯度；真实Adam更新不归一化。Q_fg使用原始200×200未截断强度、整图质量归一化和固定目标前景T>0.1，统计为float64。\n\n固定输入为能量目标对照attempt_03的A4700/readouts_start/人口与父报告，以及有效梯度核验attempt_04的lambda记录。200行拟合人口用于更新；300行确认人口不参与训练或选择。所有人口来自canonical TRAIN。\n\n正式光学前向使用CUDA严格FP32、单行微批。每臂在100、300、700步读出全部500行，共含共同A4700起点3500行。唯一判读使用终点类别2、4各30行的等权Q_fg差值及预设数值容差。\n\n停止点：完成有效训练与报告后停止；training_recommendation=NONE，next_action=NONE。"""


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    root = _root()
    artifact = _resolve(root, args.artifact_dir)
    checkpoint = _resolve(root, args.checkpoint)
    gradient_dir = _resolve(root, args.gradient_audit_dir)
    target = _resolve(root, args.output_dir)
    resume_path = _resolve(root, args.resume_checkpoint) if args.resume_checkpoint is not None else None
    _require(not (args.preflight_only and resume_path is not None), "--preflight-only cannot be combined with --resume-checkpoint")
    _prepare_target(root, target, resume_checkpoint=resume_path)
    if (target / "report.json").is_file():
        return _json(target / "report.json")
    inputs = _load_inputs(root, artifact, checkpoint, gradient_dir)
    if (target / "input_population.json").is_file():
        _require(_json(target / "input_population.json") == inputs["population"], "staged input population differs from the explicitly named source")
    started = time.perf_counter()
    selected = None
    if resume_path is not None:
        selected = torch.load(resume_path, map_location="cpu", weights_only=True)
        _validate_checkpoint(selected, resume_path, target, inputs)
    with capacity.strict_fp32_mode() as numeric_mode:
        _require(numeric_mode == STRICT_FP32_SETTINGS, "strict FP32 runtime settings differ")
        execution_dll_probe = gradient._runtime_dll_probe()
        _write_json(target / "execution_runtime_environment.json", {"task_id": TASK_ID, "runtime_dll_probe": execution_dll_probe, "process_scope": "formal execution process"}, replace=True)
        if (target / "preflight.json").is_file():
            control, receipt, preflight = _validate_ready(target, inputs)
        else:
            preflight = _preflight(inputs, target)
            control, receipt = _make_receipt(inputs, preflight)
            _stage_ready(target, inputs, preflight, receipt, resume=False)
        if args.preflight_only:
            return {"task_id": TASK_ID, "status": "READY", "output_dir": target.as_posix(), "scientific_updates": 0, "checks": preflight["checks"], "next_action": "NONE"}
        control, receipt, preflight = _validate_ready(target, inputs)
        if not (target / "readouts" / "A4700" / "paired_rows.csv").is_file():
            _measure_source(target, inputs)
        else:
            _require(len(_read_csv(target / "readouts" / "A4700" / "paired_rows.csv")) == TOTAL_ROWS, "saved A4700 starting readout is incomplete")
        evaluators = generation.diagnostic.load_evaluators(inputs["generation_paths"], torch.device("cuda:0"))
        evaluator_before = {name: capacity._state_snapshot(model) for name, model in evaluators.items()}
        _preserve_prior_failure(target)
        control.start_execution()
        control.open_scientific_attempt(target, f"{TASK_ID}:attempt-01", receipt, resume=True)
        arms = _run_training(target, inputs, evaluators, selected)
        _require(all(payload["task_local_updates"] == ADDED_UPDATES for payload in arms.values()), "both arms must complete exactly 700 additional updates")
        for name, model in evaluators.items():
            capacity._assert_state_equal(evaluator_before[name], capacity._state_snapshot(model), f"final frozen evaluator {name}")
        torch.cuda.synchronize(torch.device("cuda:0"))
        return _finalize(target, inputs, control, preflight, started)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--gradient-audit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, help="explicit durable in-task checkpoint for interruption recovery")
    return parser.parse_args()


def execute(args: argparse.Namespace | None = None) -> dict[str, Any]:
    args = args or _arguments()
    try:
        return _execute(args)
    except TrainingInterrupted as error:
        return {"task_id": TASK_ID, "status": "INTERRUPTED", "failure": str(error), "next_action": "resume only with an explicit durable in-task checkpoint"}
    except Exception as error:
        target = _resolve(_root(), args.output_dir)
        if target.is_dir() and not (target / "report.json").is_file() and not (target / "invalid_harness.json").is_file():
            checkpoint_paths = sorted(target.glob("arm_[ab]_step*.pt"), key=lambda path: path.name)
            failure = {"task_id": TASK_ID, "status": "INVALID_HARNESS", "classification": None, "failure_type": type(error).__name__, "failure": str(error), "last_durable_checkpoint": checkpoint_paths[-1].name if checkpoint_paths else None, "scientific_conclusion": None, "next_action": "NONE"}
            try:
                _write_json(target / "invalid_harness.json", failure)
            except OSError:
                pass
        raise


def main() -> None:
    print(json.dumps(execute(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
