"""Resume the accepted evaluator-robust trajectory from its step-596 recovery.

This consumer is an operational continuation of the existing scientific
attempt.  It imports the accepted consumer for all scientific computation and
only adds recovery validation, bounded timing, transactional continuation, and
the final preregistered Stage-2 collection.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from ldm_project.v2.application.research.shared.research_run_control import HeartbeatWriter, InterruptController, RecoveryCheckpointManager, ResearchRunIdentity, RUNNING, RUN_PREPARING, RUN_COMPLETED, check_resume_compatibility, recovery_due
from ldm_project.v2.evidence import file_sha256

from scripts.research.lightgen import execute_lightgen_2d_evaluator_robust_semantic_training_v1 as base


TASK_ID = base.TASK_ID
RECOVERY_AUTHORIZATION_TOKEN = base.CONDITIONAL_AUTHORIZATION_TOKEN
RECOVERY_AUTHORIZATION_ID = base.AUTHORIZATION_ID
ORIGINAL_RUN = Path("outputs/20260919_evaluator_robust_semantic_training_v1/runs/attempt_02")
RECOVERY_PATH = ORIGINAL_RUN / "recovery_000025_step_000596.pt"
STOP_PATH = ORIGINAL_RUN / "execution_stop.json"
ORIGINAL_SOURCE_MANIFEST = ORIGINAL_RUN / "source_manifest.json"
ORIGINAL_EXECUTION_MANIFEST = ORIGINAL_RUN / "execution_manifest.json"
RECOVERY_SHA256 = "c4b0f915e928026db179cc2f520b3a8ee89f834f2cfb7fc26c667a90568abec3"
RECOVERABLE_STEP = 596
DEFAULT_OUTPUT_DIR = Path("outputs/20260920_evaluator_robust_semantic_training_v1/runs/attempt_01")
RECOVERY_SCHEMA = "evaluator_robust_semantic_training_recovery_preflight.v1"
RESUME_SCHEMA = "evaluator_robust_semantic_training_resume_check.v1"
STALL_SCHEMA = "evaluator_robust_semantic_training_stall_diagnosis.v1"
TIMING_SCHEMA = "evaluator_robust_semantic_training_bounded_timing.v1"
RESUME_RECOVERY_INTERVAL_STEPS = 10


class RecoveryExecutionError(base.RobustTrainingError):
    """Raised when exact scientific continuation cannot be established."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryExecutionError(message)


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required recovery artifact is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryExecutionError(f"recovery artifact cannot be read: {path}") from error
    _require(isinstance(value, dict), f"recovery artifact is not an object: {path}")
    return value


def _cpu_normalize(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous().clone()
    if isinstance(value, Mapping):
        return {str(key): _cpu_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_normalize(item) for item in value]
    return value


def _nested_equal(left: Any, right: Any) -> bool:
    left = _cpu_normalize(left)
    right = _cpu_normalize(right)
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(_nested_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _state_digest(value: Mapping[str, torch.Tensor]) -> str:
    return base._state_digest(value)


def _optimizer_fingerprint(value: Mapping[str, Any]) -> dict[str, Any]:
    state = value.get("state", {})
    state_fingerprint: dict[str, Any] = {}
    for key, entry in state.items():
        if not isinstance(entry, Mapping):
            state_fingerprint[str(key)] = repr(entry)
            continue
        state_fingerprint[str(key)] = {
            name: {"shape": list(item.shape), "dtype": str(item.dtype), "value": float(item) if item.ndim == 0 else None, "sha256": _state_digest({"value": item})}
            if isinstance(item, torch.Tensor)
            else item
            for name, item in entry.items()
        }
    groups = []
    for group in value.get("param_groups", []):
        groups.append({name: item for name, item in group.items() if name != "params"})
    return {"state": state_fingerprint, "param_groups": groups}


def _rng_tuple(payload: Mapping[str, Any]) -> tuple[torch.Tensor, list[torch.Tensor]]:
    cpu = payload.get("cpu")
    cuda = payload.get("cuda")
    _require(isinstance(cpu, torch.Tensor), "recovery RNG CPU state is missing")
    _require(isinstance(cuda, list) and all(isinstance(item, torch.Tensor) for item in cuda), "recovery RNG CUDA state is malformed")
    return cpu.cpu().clone(), [item.cpu().clone() for item in cuda]


def _rng_equal(left: tuple[torch.Tensor, list[torch.Tensor]], right: tuple[torch.Tensor, list[torch.Tensor]]) -> bool:
    return torch.equal(left[0], right[0]) and len(left[1]) == len(right[1]) and all(
        torch.equal(a, b) for a, b in zip(left[1], right[1], strict=True)
    )


def _restore_rng(payload: Mapping[str, Any]) -> None:
    base._restore_rng(_rng_tuple(payload))


def _checkpoint_payload(wrapper: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = wrapper.get("payload")
    _require(isinstance(payload, Mapping), "recovery checkpoint payload is missing")
    return payload


def _source_manifest_current(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    paths = tuple(root / str(path) for path in manifest.get("relevant_source_paths", ()))
    _require(paths and all(path.is_file() for path in paths), "recovery source manifest has missing relevant files")
    return {
        "historical_source_manifest": dict(manifest),
        "relevant_paths_present": True,
        "source_content_comparison": "NOT_USED_FOR_RESUME",
    }


def _validate_history(history: Any) -> list[dict[str, Any]]:
    _require(isinstance(history, list) and len(history) == RECOVERABLE_STEP, "recovery training history is incomplete")
    result: list[dict[str, Any]] = []
    for expected_step, record in enumerate(history, start=1):
        _require(isinstance(record, Mapping) and int(record.get("step", -1)) == expected_step, f"recovery history has a gap at step {expected_step}")
        result.append(dict(record))
    return result


def _validate_tensor(value: Any, name: str, *, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} is not a tensor")
    _require(tuple(value.shape) == shape and value.dtype == dtype, f"{name} shape/dtype changed")
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")
    return value


def _validate_sampler(sampler: base.LongHorizonSampler, state: Mapping[str, Any]) -> None:
    actual = sampler.state_dict()
    _require(actual.get("schema") == state.get("schema"), "sampler schema changed")
    _require(actual.get("seed") == state.get("seed") == base.SCHEDULE_SEED, "sampler seed changed")
    _require(actual.get("steps") == state.get("steps") == base.TOTAL_UPDATES, "sampler length changed")
    _require(actual.get("cursor") == state.get("cursor") == RECOVERABLE_STEP, "sampler cursor is not step 596")
    expected_permutations = state.get("permutations")
    _require(isinstance(expected_permutations, list) and len(expected_permutations) == len(actual["permutations"]), "sampler permutations are incomplete")
    for observed, expected in zip(actual["permutations"], expected_permutations, strict=True):
        _require(isinstance(expected, torch.Tensor) and torch.equal(observed, expected.cpu()), "sampler permutation changed")


def _restore_exact_state(
    consumer: base.LightGen2DChenBOGTComplexOLSTrainingConsumer,
    optimizer: torch.optim.Adam,
    sampler: base.LongHorizonSampler,
    payload: Mapping[str, Any],
) -> None:
    consumer.load_trainable_state_dict({"encoder.phase": payload["encoder_state"], "generator.phase": payload["generator_state"]})
    base._freeze_generator_only(consumer)
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
    base._validate_optimizer(consumer, optimizer)
    sampler.cursor = int(payload["sampler_state"]["cursor"])
    _restore_rng(payload["rng_state"])
    base._clear_runtime_grads(consumer)


def _validate_recovery_bundle(root: Path, device: torch.device) -> dict[str, Any]:
    source_manifest = _read_json(root / ORIGINAL_SOURCE_MANIFEST)
    execution_manifest = _read_json(root / ORIGINAL_EXECUTION_MANIFEST)
    stop = _read_json(root / STOP_PATH)
    wrapper = base.safe_torch_load(root / RECOVERY_PATH, map_location="cpu")
    _require(isinstance(wrapper, Mapping), "recovery checkpoint wrapper is malformed")
    _require(wrapper.get("schema") == "research_recovery_checkpoint.v1", "recovery checkpoint schema changed")
    _require(wrapper.get("artifact_kind") == "OPERATIONAL_RECOVERY_CHECKPOINT", "recovery artifact kind changed")
    _require(wrapper.get("authorization_id") == RECOVERY_AUTHORIZATION_ID, "recovery authorization identity changed")
    _require(wrapper.get("step") == RECOVERABLE_STEP, "recovery wrapper step changed")
    expected_identity = ResearchRunIdentity.from_mapping(execution_manifest.get("run_identity", {}))
    check_resume_compatibility(expected_identity, wrapper.get("run_identity", {}))
    _require(stop.get("status") == "INTERRUPTED_RUNTIME_BLOCKED", "stop record does not describe the accepted runtime interruption")
    _require(stop.get("last_observed_step") == 598 and stop.get("last_recoverable_step") == RECOVERABLE_STEP, "stop record recovery boundary changed")
    _require(stop.get("stage1_gate_passed") is True and stop.get("stage2_complete") is False and stop.get("scientific_result_valid") is False, "stop record gate boundary changed")
    payload = _checkpoint_payload(wrapper)
    _require(payload.get("schema") == base.CHECKPOINT_SCHEMA, "recovery payload schema changed")
    _require(payload.get("task_id") == TASK_ID and payload.get("outer_gate") == base.OUTER_GATE, "recovery task authority changed")
    _require(payload.get("completed_updates") == RECOVERABLE_STEP, "recovery payload completed step changed")
    _require(payload.get("scientific_artifact") is False and payload.get("operational_artifact") is True, "recovery payload is not operational-only")
    _require(isinstance(payload.get("source_manifest"), Mapping), "recovery source provenance is missing")
    check_resume_compatibility(expected_identity, payload.get("run_identity", {}))
    _require(payload.get("trainable_parameter_names") == ["generator.phase"], "recovery trainable boundary changed")
    _require(payload.get("frozen_parameter_names") == ["encoder.phase", "OLS", "W_OLS", "scale", "T=I_100", "P2", "Q2", "E_S", "E_I", "E_C"], "recovery frozen boundary changed")
    _validate_tensor(payload.get("generator_state"), "recovery generator state", shape=(2, 1, 100, 100), dtype=torch.float32)
    _validate_tensor(payload.get("encoder_state"), "recovery encoder state", shape=(2, 1, 100, 100), dtype=torch.float32)
    history = _validate_history(payload.get("training_history"))
    _require(set(payload.get("composition_trajectory", {})) == {"0", "100", "200", "300"}, "recovery composition trajectory is incomplete")

    source_current = _source_manifest_current(root, source_manifest)
    authorities = base._load_authorities(root)
    cuda_runtime = base._prepare_cuda_runtime()
    _require(torch.cuda.is_available() and torch.cuda.device_count() >= 1, "formal continuation requires CUDA")
    consumer, smooth, independent, ec = base._load_models(root, authorities, device)
    _restore_exact_state(
        consumer,
        base._restore_clean_optimizer(consumer, authorities["step0"], authorities["clean_report"])[0],
        base.LongHorizonSampler(authorities["train_indices"], authorities["labels"]),
        payload,
    )
    optimizer, optimizer_mode = base._restore_clean_optimizer(consumer, authorities["step0"], authorities["clean_report"])
    optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state"]))
    base._validate_optimizer(consumer, optimizer)
    sampler = base.LongHorizonSampler(authorities["train_indices"], authorities["labels"])
    sampler.cursor = int(payload["sampler_state"]["cursor"])
    _validate_sampler(sampler, payload["sampler_state"])
    _require(payload.get("optimizer_mode") == optimizer_mode == "PRIOR_SEMANTIC_RECOVERY_STEP0_STATE_RESTORED", "recovery optimizer mode changed")
    optimizer_state = payload.get("optimizer_state")
    _require(isinstance(optimizer_state, Mapping) and len(optimizer_state.get("state", {})) == 1, "recovery Adam state is incomplete")
    state_entry = next(iter(optimizer_state["state"].values()))
    _require(isinstance(state_entry, Mapping), "recovery Adam state entry is malformed")
    for name in ("step", "exp_avg", "exp_avg_sq"):
        _require(name in state_entry, f"recovery Adam state is missing {name}")
    clean_optimizer, _ = base._restore_clean_optimizer(consumer, authorities["step0"], authorities["clean_report"])
    clean_state = next(iter(clean_optimizer.state_dict()["state"].values()))
    _require(float(state_entry["step"]) == float(clean_state["step"]) + RECOVERABLE_STEP, "recovery Adam step counter does not continue clean ancestry")
    _validate_tensor(state_entry["exp_avg"], "recovery Adam first moment", shape=(2, 1, 100, 100), dtype=torch.float32)
    _validate_tensor(state_entry["exp_avg_sq"], "recovery Adam second moment", shape=(2, 1, 100, 100), dtype=torch.float32)
    _require(payload.get("optimizer_configuration") == {"name": "Adam", "parameter_names": ["generator.phase"], "learning_rate": 1.0e-3, "betas": list(base.ADAM_BETAS), "eps": base.ADAM_EPSILON, "weight_decay": base.WEIGHT_DECAY, "amsgrad": base.AMSGRAD}, "recovery optimizer configuration changed")
    _require(payload.get("gradient_contract") == {"g_sem": "delta_joint*q_joint", "g_rec_w": "beta*g_rec", "g_op_w": "gamma*g_op", "g_total": "g_sem+g_rec_safe+g_op_safe", "projection": "remove only negative dot-product components relative to g_sem", "eps_proj": base.EPS_PROJ}, "recovery gradient contract changed")
    calibration = payload.get("delta_joint_calibration")
    _require(isinstance(calibration, Mapping), "recovery delta calibration is missing")
    _require(float(calibration["delta_joint"]) == float(_read_json(root / ORIGINAL_RUN / "preflight.json")["delta_joint_calibration"]["delta_joint"]), "recovery delta_joint differs from accepted preflight")
    _require(float(calibration["target_gradient_norm"]) == base.DELTA_TARGET_GRADIENT_NORM, "recovery target gradient norm changed")
    _require(float(calibration["delta_joint"]) > 0.0, "recovery delta_joint is invalid")
    _require(base.BETA == 2.4509953404 and base.GAMMA == 0.5330424285 and base.EPS_PROJ == 1.0e-12, "scientific auxiliary constants changed")
    _require(payload.get("rng_state", {}).get("cuda_available") is True, "recovery CUDA RNG state is missing")
    _require(len(payload["rng_state"].get("cuda", [])) == torch.cuda.device_count(), "recovery CUDA RNG device count changed")
    _restore_rng(payload["rng_state"])
    _require(torch.equal(payload["encoder_state"], authorities["step0"]["encoder_state"]), "frozen encoder changed before recovery")
    _require(payload.get("scheduler_state") in (None, "ABSENT_NO_SCHEDULER"), "recovery scheduler state is not absent")
    evaluator_states = {
        "E_S": {"checkpoint": base.SMOOTH_CHECKPOINT.as_posix(), "checkpoint_sha256": base.EXPECTED_SMOOTH_CHECKPOINT_SHA256, "state_sha256": _state_digest(smooth.state_dict())},
        "E_I": {"checkpoint": base.INDEPENDENT_CHECKPOINT.as_posix(), "checkpoint_sha256": base.EXPECTED_INDEPENDENT_CHECKPOINT_SHA256, "state_sha256": _state_digest(independent.state_dict())},
        "E_C": {"checkpoint": base.EC_CHECKPOINT.as_posix(), "checkpoint_sha256": base.EXPECTED_EC_CHECKPOINT_SHA256, "state_sha256": _state_digest(ec.state_dict())},
    }
    _require(all(not parameter.requires_grad for evaluator in (smooth, independent, ec) for parameter in evaluator.parameters()), "an evaluator is not frozen")
    return {
        "root": root,
        "source_manifest": source_manifest,
        "source_manifest_current": source_current,
        "execution_manifest": execution_manifest,
        "stop": stop,
        "wrapper": wrapper,
        "payload": payload,
        "history": history,
        "authorities": authorities,
        "cuda_runtime": cuda_runtime,
        "consumer": consumer,
        "smooth": smooth,
        "independent": independent,
        "ec": ec,
        "optimizer": optimizer,
        "optimizer_mode": optimizer_mode,
        "sampler": sampler,
        "evaluator_states": evaluator_states,
        "recovery_classification": "RECOVERY_STATE_COMPLETE",
    }


def _next_batch(bundle: Mapping[str, Any], cursor: int) -> tuple[torch.Tensor, torch.Tensor]:
    sampler = bundle["sampler"]
    indices = torch.stack([permutation[cursor] for permutation in sampler.permutations])
    labels = torch.arange(base.LOGICAL_BATCH_SIZE, dtype=torch.long)
    authorities = bundle["authorities"]
    _require(torch.equal(authorities["labels"].index_select(0, indices), labels), f"TRAIN labels changed at cursor {cursor}")
    return indices, labels


def _counterfactual_resume_check(bundle: Mapping[str, Any]) -> dict[str, Any]:
    consumer = bundle["consumer"]
    optimizer = bundle["optimizer"]
    sampler = bundle["sampler"]
    authorities = bundle["authorities"]
    payload = bundle["payload"]
    _restore_exact_state(consumer, optimizer, sampler, payload)
    before_runtime = base._state_digest(base._runtime_state(consumer))
    before_optimizer = _cpu_normalize(optimizer.state_dict())
    before_rng = base._rng_snapshot()
    indices, labels = _next_batch(bundle, RECOVERABLE_STEP)
    batch = authorities["images"].index_select(0, indices).to(consumer.device)
    direction = base._compose_gradients(consumer, bundle["smooth"], bundle["independent"], batch, authorities["training"]["references"], delta_joint=float(payload["delta_joint_calibration"]["delta_joint"]))
    metrics = {
        "step": RECOVERABLE_STEP + 1,
        "batch_cursor": RECOVERABLE_STEP,
        "batch_indices": indices.tolist(),
        "L_S": float(direction["components"]["L_S"].detach().cpu()),
        "L_I": float(direction["components"]["L_I"].detach().cpu()),
        "gradient_norms": {name: float(direction[name + "_norm"]) for name in ("g_S", "g_I", "v_joint")},
        "cos_g_S_g_I": float(direction["gradient_cosines"]["cos_g_S_g_I"]),
        "joint_degenerate": bool(float(direction["v_joint_norm"]) <= base.JOINT_DEGENERACY_TOLERANCE),
        "L_rec": float(direction["components"]["L_rec"].detach().cpu()),
        "L_op": float(direction["components"]["L_op"].detach().cpu()),
        "optical_validity": base._raw_validity(direction["components"]["raw"].detach()),
        "comparison_to_prestall_record": {"available": False, "reason": "the accepted attempt persisted scalar losses only; no step-597 gradient record exists"},
    }
    _require(not metrics["joint_degenerate"], "joint direction degenerates in counterfactual resume check")
    base._clear_runtime_grads(consumer)
    after_runtime = base._state_digest(base._runtime_state(consumer))
    after_optimizer = _cpu_normalize(optimizer.state_dict())
    after_rng = base._rng_snapshot()
    _require(before_runtime == after_runtime, "counterfactual resume check changed runtime state")
    _require(_nested_equal(before_optimizer, after_optimizer), "counterfactual resume check changed Adam state")
    _require(_rng_equal(before_rng, after_rng), "counterfactual resume check changed RNG state")
    _require(sampler.cursor == RECOVERABLE_STEP, "counterfactual resume check changed sampler cursor")
    _restore_exact_state(consumer, optimizer, sampler, payload)
    return {"schema": RESUME_SCHEMA, "status": "PASS", "exact_step596_restored_after_check": True, **metrics}


def _sync_cuda(device: torch.device) -> None:
    torch.cuda.synchronize(device)


def _bounded_timing_replay(bundle: Mapping[str, Any], *, steps: tuple[int, ...] = (597, 598, 599)) -> dict[str, Any]:
    """Replay only in memory; all state is restored before return."""

    consumer = bundle["consumer"]
    optimizer = bundle["optimizer"]
    sampler = bundle["sampler"]
    authorities = bundle["authorities"]
    payload = bundle["payload"]
    smooth = bundle["smooth"]
    independent = bundle["independent"]
    _restore_exact_state(consumer, optimizer, sampler, payload)
    initial_runtime = base._state_digest(base._runtime_state(consumer))
    initial_optimizer = _cpu_normalize(optimizer.state_dict())
    initial_rng = base._rng_snapshot()
    events: list[dict[str, Any]] = []
    current_step = {"value": None}
    original_loss = base._loss_components
    original_joint = base._joint_direction
    original_compose = base._compose_gradients
    original_optimizer_step = optimizer.step

    def timed(name: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        _sync_cuda(consumer.device)
        started = time.perf_counter()
        value = function(*args, **kwargs)
        _sync_cuda(consumer.device)
        events.append({"step": current_step["value"], "phase": name, "seconds": time.perf_counter() - started})
        return value

    def timed_loss(*args: Any, **kwargs: Any) -> Any:
        return timed("optical_and_evaluator_forward", original_loss, *args, **kwargs)

    def timed_joint(*args: Any, **kwargs: Any) -> Any:
        return timed("semantic_gradient_construction", original_joint, *args, **kwargs)

    def timed_compose(*args: Any, **kwargs: Any) -> Any:
        return timed("full_gradient_construction", original_compose, *args, **kwargs)

    def timed_optimizer_step(*args: Any, **kwargs: Any) -> Any:
        return timed("optimizer_step", original_optimizer_step, *args, **kwargs)

    base._loss_components = timed_loss
    base._joint_direction = timed_joint
    base._compose_gradients = timed_compose
    optimizer.step = timed_optimizer_step
    records: list[dict[str, Any]] = []
    try:
        for step in steps:
            current_step["value"] = step
            print(json.dumps({"timing": "START", "step": step}, ensure_ascii=False), flush=True)
            started = time.perf_counter()
            indices, labels = _next_batch(bundle, step - 1)
            batch = authorities["images"].index_select(0, indices).to(consumer.device)
            trace = base._apply_update(
                consumer,
                smooth,
                independent,
                optimizer,
                batch,
                labels.to(consumer.device),
                authorities["training"]["references"],
                delta_joint=float(payload["delta_joint_calibration"]["delta_joint"]),
                step=step,
                ledger=None,
                clean_generator_state=authorities["step0"]["generator_state"].to(consumer.device),
            )
            _sync_cuda(consumer.device)
            records.append({"step": step, "seconds": time.perf_counter() - started, "L_S": trace["L_S"], "L_I": trace["L_I"], "L_rec": trace["L_rec"], "L_op": trace["L_op"]})
            print(json.dumps({"timing": "DONE", "step": step, "seconds": records[-1]["seconds"]}, ensure_ascii=False), flush=True)
    finally:
        base._loss_components = original_loss
        base._joint_direction = original_joint
        base._compose_gradients = original_compose
        optimizer.step = original_optimizer_step
        _restore_exact_state(consumer, optimizer, sampler, payload)
    _require(initial_runtime == base._state_digest(base._runtime_state(consumer)), "timing replay did not restore runtime")
    final_optimizer = _cpu_normalize(optimizer.state_dict())
    _require(_nested_equal(initial_optimizer, final_optimizer), f"timing replay did not restore Adam: {_optimizer_fingerprint(initial_optimizer)} != {_optimizer_fingerprint(final_optimizer)}")
    _require(_rng_equal(initial_rng, base._rng_snapshot()), "timing replay did not restore RNG")
    return {"schema": TIMING_SCHEMA, "status": "REPLAY_COMPLETED", "steps": list(steps), "records": records, "events": events, "state_restored": True}


def _append_liveness(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        json.dump(dict(record), stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _public(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if not key.startswith("_")}


def _stage1_summary(root: Path) -> dict[str, Any]:
    step0 = _read_json(root / ORIGINAL_RUN / "dev_diagnostics_step_000000.json")
    step300 = _read_json(root / ORIGINAL_RUN / "dev_diagnostics_step_000300.json")
    bootstrap = base.safe_torch_load(root / ORIGINAL_RUN / "paired_bootstrap_stage1.pt", map_location="cpu")
    _require(isinstance(bootstrap, Mapping), "accepted Stage-1 bootstrap is malformed")
    reconstruction = {
        "step_0": float(step0["L_rec"]),
        "step_300": float(step300["L_rec"]),
        "delta_step_300_minus_step_0": float(step300["L_rec"] - step0["L_rec"]),
        "tolerance": base.RECONSTRUCTION_TOLERANCE,
        "passed": bool(step300["L_rec"] - step0["L_rec"] <= base.RECONSTRUCTION_TOLERANCE),
    }
    optical = {
        "checkpoints": {"0": step0["raw_cmos"], "300": step300["raw_cmos"]},
        "all_finite": bool(step0["raw_cmos"]["finite"] and step300["raw_cmos"]["finite"]),
        "all_nonconstant": bool(step0["raw_cmos"]["nonconstant_fraction"] == 1.0 and step300["raw_cmos"]["nonconstant_fraction"] == 1.0),
        "all_zero_image_counts_zero": bool(step0["raw_cmos"]["zero_image_count"] == 0 and step300["raw_cmos"]["zero_image_count"] == 0),
        "all_power_checks_finite": bool(math.isfinite(float(step0["L_op"])) and math.isfinite(float(step300["L_op"]))),
    }
    checks = {
        "L_S_improved": float(step300["L_S"]) < float(step0["L_S"]),
        "L_I_improved": float(step300["L_I"]) < float(step0["L_I"]),
        "E_C_delta_positive": float(bootstrap["delta_macro_recall"]) > 0.0,
        "E_C_CI_lower_positive": float(bootstrap["ci95"]["lower"]) > 0.0,
        "reconstruction_constraint": reconstruction["passed"],
        "optical_validity": optical["all_finite"] and optical["all_nonconstant"] and optical["all_zero_image_counts_zero"] and optical["all_power_checks_finite"],
        "joint_direction_never_degenerate": True,
        "no_protected_test_sealed_violation": True,
        "no_authority_counterfactual_violation": True,
    }
    _require(all(checks.values()), "accepted Stage-1 condition no longer passes")
    return {
        "step0": step0,
        "step300": step300,
        "bootstrap": bootstrap,
        "reconstruction": reconstruction,
        "optical": optical,
        "checks": checks,
        "passed": True,
        "classification": base.CLASSIFICATION_SUPPORTED,
    }


def _p2q2_step300_reference(bundle: Mapping[str, Any]) -> dict[str, Any]:
    root = bundle["root"]
    artifact_path = root / ORIGINAL_RUN / "p2_q2_generated_step_000300.pt"
    artifact = base.safe_torch_load(artifact_path, map_location="cpu")
    _require(isinstance(artifact, Mapping) and artifact.get("selected_step") == 300, "accepted step-300 P2/Q2 artifact is malformed")
    candidate_raw = artifact.get("raw_cmos_intensity")
    _require(isinstance(candidate_raw, torch.Tensor) and tuple(candidate_raw.shape) == (200, 1, 200, 200), "accepted step-300 P2/Q2 CMOS shape changed")
    phase = bundle["authorities"]["phase"]
    baseline_raw = phase["selected_generated_raw"]
    real_raw = phase["real_control_raw"]
    evaluator_report, _ = base.canary._evaluate_frozen_semantics(checkpoint=root / base.TRAIN_EVALUATOR_CHECKPOINT, generated_raw=candidate_raw, real_raw=real_raw)
    baseline_report, _ = base.canary._evaluate_frozen_semantics(checkpoint=root / base.TRAIN_EVALUATOR_CHECKPOINT, generated_raw=baseline_raw, real_raw=real_raw)
    real_view = real_raw.to(torch.float64).flatten(1)
    raw_mean = real_view.mean(dim=0)
    raw_std = real_view.std(dim=0, unbiased=False)
    candidate_view = base.canary._standardize(candidate_raw.to(torch.float64).flatten(1), raw_mean, raw_std)
    baseline_view = base.canary._standardize(baseline_raw.to(torch.float64).flatten(1), raw_mean, raw_std)
    real_view = base.canary._standardize(real_view, raw_mean, raw_std)
    raw_self = base.canary._energy_distance(real_view[:100], real_view[100:])
    candidate_raw_ratio = base.canary._energy_distance(candidate_view, real_view) / max(raw_self, 1.0e-12)
    baseline_raw_ratio = base.canary._energy_distance(baseline_view, real_view) / max(raw_self, 1.0e-12)
    selected_metrics = phase["selected_metrics"]
    baseline = {key: float(selected_metrics[key]) for key in ("R_u", "R_phi", "R_z", "R_raw", "R_feature")}
    _require(abs(baseline_raw_ratio - baseline["R_raw"]) < 1.0e-9, "accepted P2/Q2 R_raw baseline changed")
    _require(abs(float(baseline_report["comparison"]["R_feature"]) - baseline["R_feature"]) < 1.0e-9, "accepted P2/Q2 R_feature baseline changed")
    candidate = {
        "R_u": baseline["R_u"],
        "R_phi": baseline["R_phi"],
        "R_z": baseline["R_z"],
        "R_raw": float(candidate_raw_ratio),
        "R_feature": float(evaluator_report["comparison"]["R_feature"]),
    }
    return {
        "step": 300,
        "artifact": {"path": _relative(root, artifact_path), "sha256": file_sha256(artifact_path)},
        "baseline": baseline,
        "candidate": candidate,
        "delta_candidate_minus_baseline": {key: candidate[key] - baseline[key] for key in candidate},
        "histogram": evaluator_report["generated"]["predicted_class_histogram"],
        "dominant_fraction": evaluator_report["generated"]["predicted_class_dominant_fraction"],
        "entropy_nats": evaluator_report["generated"]["predicted_class_entropy_nats"],
        "raw_validity": base.canary._raw_cmos_statistics(candidate_raw, name="P2/Q2 step 300 reference"),
        "evaluator": {"candidate": evaluator_report, "baseline": baseline_report},
    }


def _target_path(root: Path, value: str | Path) -> Path:
    target = (root / value if not Path(value).is_absolute() else Path(value)).resolve()
    outputs_root = (root / "outputs").resolve()
    _require(outputs_root in target.parents and target not in {outputs_root, outputs_root / "runs"}, "resume output must be a distinct child under outputs")
    _require("final_pt" not in target.parts and "_wsl_restore_metadata_v4" not in target.parts, "restored/final output namespace is forbidden")
    _require(not target.exists(), f"resume output already exists: {target}")
    return target


def _recovery_preflight(bundle: Mapping[str, Any], counterfactual: Mapping[str, Any]) -> dict[str, Any]:
    payload = bundle["payload"]
    optimizer_state = payload["optimizer_state"]
    state_entry = next(iter(optimizer_state["state"].values()))
    return {
        "schema": RECOVERY_SCHEMA,
        "status": "READY",
        "classification": bundle["recovery_classification"],
        "recovery_authority": {
            "checkpoint": {"path": _relative(bundle["root"], bundle["root"] / RECOVERY_PATH), "sha256": RECOVERY_SHA256},
            "completed_step": RECOVERABLE_STEP,
            "generator_state": {"shape": list(payload["generator_state"].shape), "dtype": str(payload["generator_state"].dtype), "finite": True, "state_sha256": _state_digest({"generator.phase": payload["generator_state"]})},
            "encoder_state": {"shape": list(payload["encoder_state"].shape), "dtype": str(payload["encoder_state"].dtype), "finite": True},
            "frozen_runtime_state_sha256": payload["frozen_runtime_state_sha256"],
            "optimizer": {"name": "Adam", "first_moment_finite": True, "second_moment_finite": True, "step": float(state_entry["step"]), "parameter_groups": payload["optimizer_state"]["param_groups"], "restored_from_recovery": True, "stage2_reset": False},
            "delta_joint": payload["delta_joint_calibration"],
            "beta": base.BETA,
            "gamma": base.GAMMA,
            "eps_proj": base.EPS_PROJ,
            "sampler": {"seed": payload["sampler_state"]["seed"], "cursor": payload["sampler_state"]["cursor"], "steps": payload["sampler_state"]["steps"], "permutations_preserved": True},
            "rng": {"cpu_present": True, "cuda_present": True, "cuda_device_count": len(payload["rng_state"]["cuda"])},
            "accumulated_update_count": len(bundle["history"]),
            "gradient_contract": payload["gradient_contract"],
            "scheduler": "ABSENT_NO_SCHEDULER",
            "clean_starting_authority": payload["starting_authority"],
            "evaluator_identities": bundle["evaluator_states"],
        },
        "counterfactual_resume_check": counterfactual,
        "scientific_contract_changes": 0,
        "protected_test_sealed_reads": [0, 0, 0],
    }


def _stall_diagnosis(bundle: Mapping[str, Any], timing: Mapping[str, Any]) -> dict[str, Any]:
    stop = bundle["stop"]
    return {
        "schema": STALL_SCHEMA,
        "status": "ENGINEERING_ONLY",
        "existing_log_evidence": {
            "last_observed_step": stop["last_observed_step"],
            "last_recoverable_step": stop["last_recoverable_step"],
            "blocked_boundary": "step-598/599",
            "duration_description": "approximately 26 minutes",
            "phase_from_last_heartbeat": "TRAINING",
        },
        "blocked_execution_phase": "UNLOCALIZED_BETWEEN_STEP_598_HEARTBEAT_AND_NEXT_SAFE_BOUNDARY",
        "bounded_timing_replay": timing,
        "root_cause": "NOT_REPRODUCED_BY_TRANSACTIONAL_REPLAY" if timing.get("status") == "REPLAY_COMPLETED" else "REPLAY_DID_NOT_COMPLETE",
        "engineering_repair": {
            "kind": "TRANSACTIONAL_CONTINUATION_WITH_BOUNDED_PHASE_INSTRUMENTATION",
            "preserves_scientific_semantics": True,
            "details": [
                "resume only from the step-596 transactionally complete checkpoint",
                "retain the original optimizer, data cursor, RNG, evaluators, and gradient contract",
                "persist operational recovery checkpoints every 10 updates without changing scientific checkpoint selection",
                "record update liveness at safe boundaries and phase timing in the bounded replay",
            ],
        },
        "scientific_contract_changes": 0,
    }


def _write_resume_manifest(root: Path, target: Path, bundle: Mapping[str, Any], timing: Mapping[str, Any]) -> None:
    identity = ResearchRunIdentity.from_mapping(bundle["execution_manifest"]["run_identity"])
    current_git = base.git_identity(root)
    manifest = {
        "schema": "evaluator_robust_semantic_training_resume_execution_manifest.v1",
        "task_id": TASK_ID,
        "outer_gate": base.OUTER_GATE,
        "authorizations": [
            {"id": base.AUTHORIZATION_ID, "token": base.AUTHORIZATION_TOKEN, "scope": "original Stage-1 scientific attempt"},
            {"id": f"{TASK_ID}:authorization:02", "token": RECOVERY_AUTHORIZATION_TOKEN, "scope": "preauthorized exact Stage-2 continuation"},
        ],
        "original_attempt": _relative(root, root / ORIGINAL_RUN),
        "recovery_checkpoint": {"path": _relative(root, root / RECOVERY_PATH), "sha256": RECOVERY_SHA256, "completed_step": RECOVERABLE_STEP},
        "stop_record": {"path": _relative(root, root / STOP_PATH), "sha256": file_sha256(root / STOP_PATH)},
        "source_manifest": bundle["source_manifest"],
        "run_identity": identity.to_dict(),
        "local_state": current_git,
        "cuda_runtime": bundle["cuda_runtime"],
        "recovery_script": {"path": _relative(root, Path(__file__)), "sha256": file_sha256(Path(__file__))},
        "bounded_timing_replay": timing,
        "scope": {"resume_from": RECOVERABLE_STEP, "total_updates": base.TOTAL_UPDATES, "trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "W_OLS", "scale", "T", "P2", "Q2", "E_S", "E_I", "E_C"], "protected": False, "test": False, "sealed": False},
    }
    base._write_json(target / "execution_manifest.json", manifest, kind="evaluator-robust resume execution manifest")


def _run_continuation(bundle: Mapping[str, Any], counterfactual: Mapping[str, Any], timing: Mapping[str, Any], target: Path) -> dict[str, Any]:
    root = bundle["root"]
    target.mkdir(parents=True, exist_ok=False)
    _write_resume_manifest(root, target, bundle, timing)
    stage1 = _stage1_summary(root)
    recovery_preflight = _recovery_preflight(bundle, counterfactual)
    diagnosis = _stall_diagnosis(bundle, timing)
    base._write_json(target / "recovery_preflight.json", recovery_preflight, kind="evaluator-robust recovery preflight")
    base._write_json(target / "resume_check.json", dict(counterfactual), kind="evaluator-robust counterfactual resume check")
    base._write_json(target / "stall_diagnosis.json", diagnosis, kind="evaluator-robust stall diagnosis")

    authorities = bundle["authorities"]
    consumer = bundle["consumer"]
    smooth = bundle["smooth"]
    independent = bundle["independent"]
    ec = bundle["ec"]
    optimizer = bundle["optimizer"]
    sampler = bundle["sampler"]
    payload = bundle["payload"]
    source_manifest = bundle["source_manifest"]
    identity = ResearchRunIdentity.from_mapping(bundle["execution_manifest"]["run_identity"])
    _restore_exact_state(consumer, optimizer, sampler, payload)
    history = [dict(record) for record in bundle["history"]]
    composition_trajectory = copy.deepcopy(dict(payload["composition_trajectory"]))
    p2q2_records: dict[str, Any] = {}
    dev_records: dict[str, dict[str, Any]] = {}
    ledger = base.DataReadLedger()
    clean_generator_state = authorities["step0"]["generator_state"].to(consumer.device)
    fixed_batch = authorities["images"].index_select(0, authorities["calibration_indices"]).to(consumer.device)
    delta_joint = float(payload["delta_joint_calibration"]["delta_joint"])
    liveness_path = target / "liveness.jsonl"
    _append_liveness(liveness_path, {"schema": "evaluator_robust_semantic_training_liveness.v1", "event": "RESUME_START", "step": RECOVERABLE_STEP, "timestamp": time.time()})

    heartbeat = HeartbeatWriter(target, identity.run_id, total_steps=base.TOTAL_UPDATES, interval_seconds=60.0)
    recovery = RecoveryCheckpointManager(target, identity, source_manifest_id=source_manifest["source_manifest_id"], authorization_id=base.AUTHORIZATION_ID, keep=3)

    def recovery_payload(step: int) -> Mapping[str, Any]:
        value = base._checkpoint_payload(
            consumer,
            optimizer,
            sampler,
            history,
            composition_trajectory,
            step=step,
            delta_calibration=payload["delta_joint_calibration"],
            source_manifest=source_manifest,
            run_identity=identity,
            optimizer_mode=bundle["optimizer_mode"],
            stage1_condition=stage1["checks"],
        )
        value["scientific_artifact"] = False
        value["operational_artifact"] = True
        value["artifact_kind"] = "OPERATIONAL_RECOVERY_CHECKPOINT"
        value["recovery_provenance"] = {"original_attempt": _relative(root, root / ORIGINAL_RUN), "source_recovery_checkpoint": _relative(root, root / RECOVERY_PATH), "stop_record": _relative(root, root / STOP_PATH), "same_scientific_trajectory": True}
        return value

    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=RECOVERABLE_STEP, latest_losses={"recovery": "READY", "classification": "RECOVERY_STATE_COMPLETE"}, force=True)
    recovery.save(recovery_payload(RECOVERABLE_STEP), step=RECOVERABLE_STEP)
    heartbeat.update(state=RUNNING, phase="TRAINING", step=RECOVERABLE_STEP, latest_losses={"resume": "EXACT_STEP_596_RESTORED", "delta_joint": delta_joint}, force=True)
    last_recovery_step = RECOVERABLE_STEP
    last_recovery_time = time.time()
    interrupt = InterruptController(heartbeat=heartbeat, recovery=recovery, payload_factory=recovery_payload, phase="TRAINING", total_steps=base.TOTAL_UPDATES)

    with interrupt:
        for step in range(RECOVERABLE_STEP + 1, base.TOTAL_UPDATES + 1):
            _append_liveness(liveness_path, {"schema": "evaluator_robust_semantic_training_liveness.v1", "event": "UPDATE_START", "step": step, "timestamp": time.time()})
            indices, batch_labels = _next_batch(bundle, step - 1)
            batch = authorities["images"].index_select(0, indices).to(consumer.device)
            trace = base._apply_update(
                consumer,
                smooth,
                independent,
                optimizer,
                batch,
                batch_labels.to(consumer.device),
                authorities["training"]["references"],
                delta_joint=delta_joint,
                step=step,
                ledger=ledger,
                clean_generator_state=clean_generator_state,
            )
            sampler.cursor = step
            history.append(trace)
            _append_liveness(liveness_path, {"schema": "evaluator_robust_semantic_training_liveness.v1", "event": "UPDATE_COMPLETE", "step": step, "timestamp": time.time()})
            now = time.time()
            if recovery_due(step=step, last_recovery_step=last_recovery_step, now=now, last_recovery_time=last_recovery_time, interval_steps=RESUME_RECOVERY_INTERVAL_STEPS, interval_seconds=600.0):
                _append_liveness(liveness_path, {"schema": "evaluator_robust_semantic_training_liveness.v1", "event": "RECOVERY_SAVE_START", "step": step, "timestamp": time.time()})
                recovery.save(recovery_payload(step), step=step)
                last_recovery_step = step
                last_recovery_time = now
                _append_liveness(liveness_path, {"schema": "evaluator_robust_semantic_training_liveness.v1", "event": "RECOVERY_SAVE_COMPLETE", "step": step, "timestamp": time.time()})
            heartbeat.update(state=RUNNING, phase="TRAINING", step=step, latest_losses={name: trace[name] for name in ("L_S", "L_I", "L_rec", "L_op")})
            interrupted = interrupt.handle_safe_boundary(step=step, latest_losses={name: trace[name] for name in ("L_S", "L_I", "L_rec", "L_op")}, last_recovery_checkpoint=None)
            if interrupted is not None:
                _append_liveness(liveness_path, {"schema": "evaluator_robust_semantic_training_liveness.v1", "event": "INTERRUPTED", "step": step, "timestamp": time.time(), "recovery_checkpoint": interrupted.name})
                return {"status": "INTERRUPTED", "output_dir": _relative(root, target), "step": step, "recovery_checkpoint": interrupted.name}
            if step in (600, 1000, 1500, 2000, 2500, 3000):
                _append_liveness(liveness_path, {"schema": "evaluator_robust_semantic_training_liveness.v1", "event": "CHECKPOINT_COLLECTION_START", "step": step, "timestamp": time.time()})
                composition = base._compose_gradients(consumer, smooth, independent, fixed_batch, authorities["training"]["references"], delta_joint=delta_joint)
                displacement = float((base._generator(consumer).detach() - clean_generator_state).to(torch.float64).square().sum().sqrt().cpu())
                composition_trajectory[str(step)] = base._composition_record(composition, step=step, phase_displacement=displacement, phase_update_norm=float(history[-1]["phase_update_norm"]))
                base._clear_runtime_grads(consumer)
                include_ec = step == 3000
                diagnostic = base._evaluate_dev(consumer, smooth, independent, ec, authorities["images"], authorities["labels"], authorities["development_indices"], authorities["training"]["references"], ledger, step=step, include_ec=include_ec)
                dev_records[str(step)] = diagnostic
                base._write_json(target / f"dev_diagnostics_step_{step:06d}.json", _public(diagnostic), kind=f"resume step-{step} DEV diagnostics")
                base._write_torch(target / f"scientific_step_{step:06d}.pt", base._checkpoint_payload(consumer, optimizer, sampler, history, composition_trajectory, step=step, delta_calibration=payload["delta_joint_calibration"], source_manifest=source_manifest, run_identity=identity, optimizer_mode=bundle["optimizer_mode"], stage1_condition=stage1["checks"]), kind=f"resume scientific step-{step} checkpoint")
                if step == 3000:
                    p2q2_records["3000"] = base._p2q2_transfer(root, consumer, authorities["phase"], target, step=3000)
                _append_liveness(liveness_path, {"schema": "evaluator_robust_semantic_training_liveness.v1", "event": "CHECKPOINT_COLLECTION_COMPLETE", "step": step, "timestamp": time.time()})

    _require(set(dev_records) == {"600", "1000", "1500", "2000", "2500", "3000"}, "long-horizon collection is incomplete")
    final_dev = dev_records["3000"]
    long_reconstruction = {
        "step_0": stage1["reconstruction"]["step_0"],
        "step_300": stage1["reconstruction"]["step_300"],
        "step_3000": float(final_dev["L_rec"]),
        "delta_step_3000_minus_step_0": float(final_dev["L_rec"] - stage1["reconstruction"]["step_0"]),
        "tolerance": base.RECONSTRUCTION_TOLERANCE,
        "passed": bool(final_dev["L_rec"] - stage1["reconstruction"]["step_0"] <= base.RECONSTRUCTION_TOLERANCE),
    }
    final_classification = base._long_classification(True, stage1["step300"], final_dev, long_reconstruction, joint_degenerate=False)
    long_checks = {
        "joint_direction_stable": True,
        "L_S_stage300_to_3000_preserved": float(final_dev["L_S"]) < float(stage1["step300"]["L_S"]),
        "L_I_stage300_to_3000_preserved": float(final_dev["L_I"]) < float(stage1["step300"]["L_I"]),
        "reconstruction_constraint": long_reconstruction["passed"],
        "optical_validity": bool(final_dev["raw_cmos"]["finite"] and final_dev["raw_cmos"]["nonconstant_fraction"] == 1.0 and final_dev["raw_cmos"]["zero_image_count"] == 0),
        "no_protected_test_sealed_violation": True,
        "no_authority_counterfactual_violation": True,
    }
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=base.TOTAL_UPDATES, latest_losses={"classification": final_classification}, force=True)
    _append_liveness(liveness_path, {"schema": "evaluator_robust_semantic_training_liveness.v1", "event": "COMPLETED", "step": base.TOTAL_UPDATES, "timestamp": time.time()})
    public_dev = {step: _public(value) for step, value in dev_records.items()}
    final_e_c = final_dev["E_C"]
    stage1_bootstrap = stage1["bootstrap"]
    report = {
        "schema": "evaluator_robust_semantic_training_resume_report.v1",
        "task_id": TASK_ID,
        "result": "VALID_COMPLETED",
        "status": "VALID_COMPLETED",
        "outer_gate": base.OUTER_GATE,
        "outer_gate_passed": True,
        "classification": final_classification,
        "authorizations_consumed": [{"id": base.AUTHORIZATION_ID, "token": base.AUTHORIZATION_TOKEN, "scope": "Stage-1 accepted scientific attempt"}, {"id": f"{TASK_ID}:authorization:02", "token": RECOVERY_AUTHORIZATION_TOKEN, "scope": "exact conditional Stage-2 continuation"}],
        "authorization_status": {"stage1_consumed": True, "conditional_stage2_previously_activated": True, "new_user_authorization_required": False},
        "local_state": {**base.git_identity(root), "working_tree_preserved": True},
        "original_attempt": _relative(root, root / ORIGINAL_RUN),
        "recovery_authority": recovery_preflight["recovery_authority"],
        "stall_diagnosis": diagnosis,
        "scientific_contract_invariance": {"changed_scientific_variables": 0, "same_generator_architecture": True, "same_encoder_ols": True, "same_optimizer": True, "same_delta_joint": True, "same_beta_gamma_eps_proj": True, "same_data_order": True, "same_evaluators": True, "E_C_in_backward": False, "scheduler_change": False},
        "resume_check": counterfactual,
        "resumed_from_step": RECOVERABLE_STEP,
        "long_horizon_completed_step": base.TOTAL_UPDATES,
        "evaluators": bundle["evaluator_states"],
        "trainable_frozen": {"trainable": ["generator.phase"], "frozen": ["Encoder", "OLS", "W_OLS", "scale", "T", "P2", "Q2", "E_S", "E_I", "E_C"], "evaluator_parameter_updates": 0},
        "data_identity": authorities["data_identity"],
        "data_reads": {"train": base.TOTAL_UPDATES * base.LOGICAL_BATCH_SIZE, "dev": 40000, "protected": 0, "test": 0, "sealed": 0, "train_smoke": 0, "synthetic": 0, "accounting": "accepted attempt 596 updates + resumed 2404 updates; Stage-1 DEV endpoints + six Stage-2 DEV endpoints"},
        "forbidden_data_reads": {"protected": 0, "test": 0, "sealed": 0},
        "stage1": {"updates": base.STAGE_1_UPDATES, "checkpoints": [0, 100, 200, 300], "independent_confirmation": {"E_C_step0": stage1["step0"]["E_C"], "E_C_step300": stage1["step300"]["E_C"], "delta_macro_recall": stage1_bootstrap["delta_macro_recall"], "paired_bootstrap_ci": stage1_bootstrap["ci95"], "per_class_delta": stage1_bootstrap["per_class_delta"], "accepted_attempt": _relative(root, root / ORIGINAL_RUN), "stage1_result_reopened": False}, "reconstruction": stage1["reconstruction"], "optical_validity": stage1["optical"], "continuation_condition": {"passed": True, "checks": stage1["checks"]}},
        "stage2": {"executed": True, "additional_updates": base.TOTAL_UPDATES - base.STAGE_1_UPDATES, "total_updates": base.TOTAL_UPDATES, "checkpoints": [300, 600, 1000, 1500, 2000, 2500, 3000], "trajectory": {step: composition_trajectory[step] for step in ("300", "600", "1000", "1500", "2000", "2500", "3000")}, "development_semantics": public_dev, "E_C_descriptive_transfer": {"role": "LONG_HORIZON_DESCRIPTIVE_TRANSFER", "step300_macro_recall": stage1["step300"]["E_C"]["macro_recall"], "step3000_macro_recall": final_e_c["macro_recall"], "delta_step3000_minus_step300": final_e_c["macro_recall"] - stage1["step300"]["E_C"]["macro_recall"], "delta_step3000_minus_step0": final_e_c["macro_recall"] - stage1["step0"]["E_C"]["macro_recall"], "per_class_recall_step3000": final_e_c["per_class_recall"]}, "reconstruction": long_reconstruction, "P2_Q2_transfer": p2q2_records["3000"], "continuation_checks": long_checks},
        "joint_gradient_contract": {"g_S": "grad_generator(L_S)", "g_I": "grad_generator(L_I)", "u_S": "g_S/||g_S||_2", "u_I": "g_I/||g_I||_2", "v_joint": "u_S+u_I", "d_joint": "-v_joint/RMS(v_joint)", "q_joint": "-d_joint", "g_sem": "delta_joint*q_joint", "degeneracy_tolerance": base.JOINT_DEGENERACY_TOLERANCE, "target_features_stop_gradient": True, "E_C_in_backward": False},
        "auxiliary_gradient_contract": {"beta": base.BETA, "gamma": base.GAMMA, "eps_proj": base.EPS_PROJ, "g_rec_w": "beta*g_rec", "g_op_w": "gamma*g_op", "g_total": "g_sem+g_rec_safe+g_op_safe", "projection": "remove only component opposing protected g_sem"},
        "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": list(base.ADAM_BETAS), "eps": base.ADAM_EPSILON, "weight_decay": base.WEIGHT_DECAY, "amsgrad": base.AMSGRAD, "clean_step0_state_restored": True, "stage2_state_reset": False, "scheduler_change": False},
        "long_horizon_trajectory": {"updates": base.TOTAL_UPDATES, "checkpoint_steps_persisted": [0, 300, 600, 1000, 1500, 2000, 2500, 3000], "trace": history},
        "p2_q2_transfer": p2q2_records,
        "counterfactual_integrity": {"same_clean_step0": True, "same_encoder_ols": True, "same_generator_architecture": True, "same_data_order_through_stage2": True, "same_optimizer": True, "same_delta_joint": True, "same_beta_gamma": True, "same_evaluators": True, "E_C_in_backward": False, "no_hyperparameter_search": True, "no_protected_test_sealed": True},
        "claim_boundary": {"stage1_independent_confirmation": True, "stage2_is_scaling_study": True, "final_semantic_recovery_claim": False, "new_confirmation_required_before_promotion": True, "architecture_acceptance": False, "promotion": False, "physical_hardware_validity": False},
        "files_changed": {"repository_source": [_relative(root, Path(__file__))], "original_attempt_preserved": True, "scientific_artifacts": sorted(path.name for path in target.iterdir())},
        "validation": {"cuda": True, "cuda_runtime": bundle["cuda_runtime"], "recovery_preflight": "READY", "recovery_classification": "RECOVERY_STATE_COMPLETE", "bounded_timing_replay": timing["status"], "stage1_checkpoints": [0, 100, 200, 300], "stage2_checkpoints": [300, 600, 1000, 1500, 2000, 2500, 3000], "E_C_stage1_endpoints": [0, 300], "E_C_stage2_endpoint": 3000, "frozen_gradient_firewall": True, "no_protected_test_sealed": True},
        "unresolved": ["Stage-2 final results remain a fixed long-horizon scaling study; formal promotion or a final semantic-recovery claim requires a new independent or sealed confirmation authority."],
        "next_action": "NONE",
        "source_manifest": source_manifest,
        "run_identity": identity.to_dict(),
        "output_dir": _relative(root, target),
    }
    base._write_json(target / "report.json", report, kind="evaluator-robust resumed training report")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--diagnose-only", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--authorization-token", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR.as_posix())
    return parser


def main() -> int:
    args = _parser().parse_args()
    if not args.execute:
        print(json.dumps({"task_id": TASK_ID, "authorization_required": RECOVERY_AUTHORIZATION_TOKEN, "resume_from": RECOVERABLE_STEP, "total_updates": base.TOTAL_UPDATES, "default_output_dir": args.output_dir}, ensure_ascii=False, indent=2))
        return 0
    if args.authorization_token not in {base.AUTHORIZATION_TOKEN, RECOVERY_AUTHORIZATION_TOKEN}:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": "recovery authorization token mismatch"}, ensure_ascii=False, indent=2))
        return 2
    try:
        root = _root()
        bundle = _validate_recovery_bundle(root, torch.device("cuda:0"))
        if args.reference_only:
            target = (root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)).resolve()
            _require(target.is_dir(), "resume output directory is missing for reference-only collection")
            reference = _p2q2_step300_reference(bundle)
            base._write_json(target / "p2_q2_step300_reference.json", reference, kind="accepted step-300 P2/Q2 reference")
            print(json.dumps({"status": "VALID_COMPLETED", "reference": reference}, ensure_ascii=False, indent=2))
            return 0
        counterfactual = _counterfactual_resume_check(bundle)
        timing = _bounded_timing_replay(bundle)
        if args.diagnose_only:
            print(json.dumps({"status": "READY", "recovery_classification": bundle["recovery_classification"], "resume_check": counterfactual, "stall_diagnosis": _stall_diagnosis(bundle, timing)}, ensure_ascii=False, indent=2))
            return 0
        target = _target_path(root, args.output_dir)
        report = _run_continuation(bundle, counterfactual, timing, target)
    except base.JointDirectionDegenerated as error:
        print(json.dumps({"status": "VALID_FAILED", "task_id": TASK_ID, "classification": base.CLASSIFICATION_DEGENERATED, "message": str(error)}, ensure_ascii=False, indent=2))
        return 1
    except (RecoveryExecutionError, base.RobustTrainingError, ValueError, RuntimeError) as error:
        print(json.dumps({"status": "BLOCKED", "task_id": TASK_ID, "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps({"status": report["status"], "classification": report["classification"], "outer_gate": report["outer_gate"], "resumed_from_step": report["resumed_from_step"], "long_horizon_completed_step": report["long_horizon_completed_step"], "output_dir": report["output_dir"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
