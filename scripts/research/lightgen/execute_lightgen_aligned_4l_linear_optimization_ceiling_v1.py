"""Run the bounded generator-only continuation of the valid 4L Linear arm.

The parent C2000 checkpoint is resolved from the immutable C scientific report
and lineage.  This successor keeps the 4L runtime, OLS interface, linear facet,
objective and data roles fixed, and asks only whether the generator-only
optimization still has material semantic headroom at local steps 3000 and 4000.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load research authority: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_module(
    Path(__file__).with_name("execute_lightgen_aligned_4l_relu_gain_v1.py"),
    "lightgen_4l_linear_continuation_code_authority",
)
N4 = BASE.N4
V10 = BASE.V10


TASK_ID = "RESOLVE_4L_LINEAR_OPTIMIZATION_CEILING_V1"
DESIGN_AUTHORITY = "LIGHTGEN_ALIGNED_4L_RELU_GAIN_V1_CONTRACT"
OUTER_GATE = "RESOLVE_4L_LINEAR_OPTIMIZATION_CEILING_V1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

PARENT_ROOT = Path("outputs/20260915_lightgen_aligned_4l_relu_gain_v1/runs")
PARENT_C_DIR = PARENT_ROOT / "lightgen_aligned_4l_relu_gain_v1_c_linear_attempt_01"
PARENT_REPORT_PATH = PARENT_C_DIR / "c_4l_linear_scientific_report.v1.json"
PARENT_LINEAGE_PATH = PARENT_C_DIR / "c_4l_linear_scientific_lineage.v1.json"
PARENT_REPORT_SHA256 = "83851bff4734a7091afdcd406353656aa2fd4055ca66af404735a8f140a3bdee"
PARENT_LINEAGE_SHA256 = "817ce3c70e75c71ffb2ea1a163cc46d67ae31e11891600ae9eb29ed9662220a8"

RUN_ROOT = Path("outputs/20260916_4l_linear_optimization_ceiling_v1/runs")
CONTRACT_DIR = RUN_ROOT / "4l_linear_optimization_ceiling_v1_contract_attempt_03"
CONTRACT_PATH = CONTRACT_DIR / "4l_linear_optimization_ceiling_v1_contract.v1.json"
PREFLIGHT_DIR = RUN_ROOT / "4l_linear_optimization_ceiling_v1_preflight_attempt_03"
PREFLIGHT_REPORT_PATH = PREFLIGHT_DIR / "4l_linear_optimization_ceiling_v1_preflight_report.v1.json"
PREFLIGHT_LINEAGE_PATH = PREFLIGHT_DIR / "4l_linear_optimization_ceiling_v1_preflight_lineage.v1.json"
SCIENTIFIC_DIR = RUN_ROOT / "4l_linear_optimization_ceiling_v1_c_linear_continuation_attempt_01"
AGGREGATE_DIR = RUN_ROOT / "4l_linear_optimization_ceiling_v1_aggregate_attempt_01"
AGGREGATE_REPORT_PATH = AGGREGATE_DIR / "4l_linear_optimization_ceiling_v1_aggregate_report.v1.json"
AGGREGATE_LINEAGE_PATH = AGGREGATE_DIR / "4l_linear_optimization_ceiling_v1_aggregate_lineage.v1.json"

PARENT_STEP = 2000
FIRST_TARGET_STEP = 3000
MAX_LOCAL_STEP = 4000
FIRST_SEGMENT_UPDATES = 1000
SECOND_SEGMENT_UPDATES = 1000
GLOBAL_STEP_OFFSET = 600
BLOCK_SIZE = 200
PROGRESS_EVAL_INTERVAL_ROWS = 500
LOGICAL_BATCH_SIZE = 10
MICRO_BATCH_SIZE = 2
SAMPLER_SEED = 1234
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
GENERATOR_LAYERS = 4
GENERATOR_LAYER_SPACING_M = 3.0e-3
PRODUCTION_GRID = 2048
ADAPTER_RANK = 9
LAMBDA_TRANSFER = 28.48464149920509
LAMBDA_IMAGE = 13.353282014834436
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
PARENT_FINAL_LR = 5.0e-5
FIRST_END_LR = 3.0e-5
SECOND_END_LR = 2.0e-5
STAGE_IDENTITY = "generator_only_continuation"
TRAINABLE_NAMES = ["generator.phase"]
FROZEN_NAMES = [
    "encoder.phase",
    "adapter.u_real",
    "adapter.u_imag",
    "adapter.v_real",
    "adapter.v_imag",
    "OLS",
    "W_OLS",
    "facet",
    "Smooth evaluator",
    "templates",
    "prototypes",
]
RECONSTRUCTION_MAX_DELTA = 0.005
CONTINUATION_MIN_GAIN = 0.010
CONTINUED_TOTAL_MIN_GAIN = 0.020
CONTINUED_LATE_MIN_GAIN = 0.005
HISTORICAL_2L_REFERENCE = 0.6170
RESOURCE_WARNING_SECONDS = 3.0 * 3600.0

CHECKPOINT_SCHEMA = "lightgen_aligned_4l_linear_optimization_ceiling_v1_checkpoint.v1"
SUFFICIENT_SCHEMA = "lightgen_aligned_4l_linear_optimization_ceiling_v1_sufficient_observations.v1"
DIAGNOSTIC_SCHEMA = "lightgen_aligned_4l_linear_optimization_ceiling_v1_diagnostics.v1"
RAW_SCHEMA = "lightgen_aligned_4l_linear_optimization_ceiling_v1_cyclic_raw_outputs.v1"
PROGRESS_SCHEMA = "lightgen_aligned_4l_linear_optimization_ceiling_v1_progress.v1"
BLOCK_SCHEMA = "lightgen_aligned_4l_linear_optimization_ceiling_v1_training_block.v1"
ENDPOINT_META_SCHEMA = "lightgen_aligned_4l_linear_optimization_ceiling_v1_endpoint_meta.v1"


class InvalidContinuation(RuntimeError):
    """A harness, lineage or process contract is invalid."""


class HardBlockedContinuation(RuntimeError):
    """A required authority or authorized resource boundary is unavailable."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.details = details


class ResourceWarningContinuation(RuntimeError):
    """The declared personal-PC resource boundary rejects the run."""

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidContinuation(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, value: str | Path) -> Path:
    relative = Path(value)
    _require(not relative.is_absolute(), f"absolute artifact path is not allowed: {relative}")
    resolved = (root / relative).resolve()
    _require(resolved == root.resolve() or root.resolve() in resolved.parents, f"artifact escapes repository root: {relative}")
    return resolved


def _relative(root: Path, value: Path) -> str:
    return value.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidContinuation(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = _sha256(path)
    _require(observed == expected.lower(), f"{name} SHA-256 mismatch: expected {expected}, observed {observed}")


def _cpu_tree(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return list(left) == list(right) and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _digest_tree(value: Any) -> str:
    return BASE._digest_tree(_cpu_tree(value))


def _finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _plain(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _save_json(root: Path, path: Path, payload: Mapping[str, Any], artifact_kind: str) -> dict[str, str]:
    _require(not path.exists(), f"refusing to overwrite {artifact_kind}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    N4.atomic_write_new_json(path, dict(payload), artifact_kind=artifact_kind)
    return {"path": _relative(root, path), "sha256": _sha256(path)}


def _save_tensor(root: Path, path: Path, payload: Mapping[str, Any], artifact_kind: str) -> dict[str, str]:
    _require(not path.exists(), f"refusing to overwrite {artifact_kind}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    N4.atomic_torch_save_new(path, dict(payload), artifact_kind=artifact_kind)
    digest = _sha256(path)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"{artifact_kind} reload")
    _require(loaded.get("schema") == payload.get("schema"), f"{artifact_kind} schema changed on reload")
    return {"path": _relative(root, path), "sha256": digest, "reloaded": True}


def _checkpoint_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"c_4l_linear_continuation_step_{step:05d}_checkpoint.v1.pt"


def _sufficient_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"c_4l_linear_continuation_step_{step:05d}_sufficient_observations.v1.pt"


def _diagnostic_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"c_4l_linear_continuation_step_{step:05d}_diagnostics.v1.json"


def _raw_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"c_4l_linear_continuation_step_{step:05d}_cyclic_raw_outputs.v1.pt"


def _endpoint_meta_path(output_dir: Path, step: int) -> Path:
    return output_dir / f"c_4l_linear_continuation_step_{step:05d}_endpoint_meta.v1.json"


def _schedule_lr(step: int) -> tuple[float, str]:
    _require(PARENT_STEP <= step <= MAX_LOCAL_STEP, f"continuation schedule step is outside [{PARENT_STEP},{MAX_LOCAL_STEP}]: {step}")
    if step <= FIRST_TARGET_STEP:
        fraction = (step - PARENT_STEP) / float(FIRST_TARGET_STEP - PARENT_STEP)
        return PARENT_FINAL_LR + fraction * (FIRST_END_LR - PARENT_FINAL_LR), STAGE_IDENTITY
    fraction = (step - FIRST_TARGET_STEP) / float(MAX_LOCAL_STEP - FIRST_TARGET_STEP)
    return FIRST_END_LR + fraction * (SECOND_END_LR - FIRST_END_LR), STAGE_IDENTITY


def _set_continuation_stage(consumer: Any, adapter: Any, *, coadapt: bool = False) -> str:
    _require(not coadapt, "continuation cannot enable adapter coadaptation")
    named = BASE._candidate_named(consumer, adapter)
    _require(list(named) == ["adapter.u_real", "adapter.u_imag", "adapter.v_real", "adapter.v_imag", "generator.phase"], "continuation parameter surface changed")
    for name, parameter in named.items():
        parameter.requires_grad_(name == "generator.phase")
        parameter.grad = None
    _require(all(not parameter.requires_grad for name, parameter in named.items() if name.startswith("adapter.")), "continuation adapter freeze failed")
    _require(named["generator.phase"].requires_grad, "continuation generator trainability failed")
    consumer.runtime.train()
    return STAGE_IDENTITY


def _patch_base_for_continuation() -> dict[str, Any]:
    keys = (
        "TASK_ID",
        "K",
        "SUFFICIENT_SCHEMA",
        "DIAGNOSTIC_SCHEMA",
        "RAW_SCHEMA",
        "PROGRESS_SCHEMA",
        "_schedule_lr",
        "_set_stage",
    )
    previous = {key: getattr(BASE, key) for key in keys}
    BASE.TASK_ID = TASK_ID
    BASE.K = MAX_LOCAL_STEP
    BASE.SUFFICIENT_SCHEMA = SUFFICIENT_SCHEMA
    BASE.DIAGNOSTIC_SCHEMA = DIAGNOSTIC_SCHEMA
    BASE.RAW_SCHEMA = RAW_SCHEMA
    BASE.PROGRESS_SCHEMA = PROGRESS_SCHEMA
    BASE._schedule_lr = _schedule_lr
    BASE._set_stage = _set_continuation_stage
    return previous


def _restore_base(previous: Mapping[str, Any]) -> None:
    for key, value in previous.items():
        setattr(BASE, key, value)


def _load_parent_authority(root: Path) -> dict[str, Any]:
    report_path = _path(root, PARENT_REPORT_PATH)
    lineage_path = _path(root, PARENT_LINEAGE_PATH)
    _validate_sha(report_path, PARENT_REPORT_SHA256, "parent C scientific report")
    _validate_sha(lineage_path, PARENT_LINEAGE_SHA256, "parent C scientific lineage")
    report = _read_json(report_path, "parent C scientific report")
    lineage = _read_json(lineage_path, "parent C scientific lineage")
    report_ref = {"path": _relative(root, report_path), "sha256": PARENT_REPORT_SHA256}
    _require(lineage.get("report") == report_ref, "parent C lineage does not bind the registered report")
    _require(
        report.get("result") == "VALID_COMPLETED"
        and report.get("status") == "valid_completed"
        and report.get("candidate") == "C_4L_LINEAR"
        and report.get("classification") == "C_4L_LINEAR_SCIENTIFIC_COMPLETED",
        "parent C report is not the valid C2000 authority",
    )
    _require(int(report.get("claim_boundary", {}).get("scientific_updates", -1)) == PARENT_STEP, "parent C update count changed")
    contract_ref = _mapping(report.get("contract"), "parent C contract reference")
    contract_path = _path(root, contract_ref["path"])
    _validate_sha(contract_path, str(contract_ref["sha256"]), "parent C contract")
    contract = _read_json(contract_path, "parent C contract")
    _require(contract.get("task_id") == "LIGHTGEN_ALIGNED_4L_RELU_GAIN_V1", "parent C contract task identity changed")
    _require(contract.get("status") == "READY_FOR_BOUNDED_LONG_SCIENTIFIC_EXECUTION", "parent C contract is not a frozen READY contract")
    geometry = _mapping(contract.get("geometry"), "parent C geometry")
    _require(geometry.get("variant") == "LIGHTGEN_TOPOLOGY_ALIGNED_4L", "parent C geometry variant changed")
    project_geometry = _mapping(geometry.get("project_used"), "parent C project geometry")
    _require(project_geometry.get("generator_layers") == GENERATOR_LAYERS, "parent C generator layer count changed")
    _require(float(project_geometry.get("generator_layer_spacing_m")) == GENERATOR_LAYER_SPACING_M, "parent C generator spacing changed")
    training = _mapping(contract.get("training"), "parent C training")
    _require(training.get("K") == PARENT_STEP, "parent C training horizon changed")
    _require(training.get("batch") == LOGICAL_BATCH_SIZE and training.get("micro_batch") == MICRO_BATCH_SIZE, "parent C batch contract changed")
    _require(training.get("sampler_seed") == SAMPLER_SEED and training.get("block_size") == BLOCK_SIZE, "parent C sampler contract changed")
    objective = _mapping(contract.get("objective"), "parent C objective")
    _require(float(objective.get("lambda_transfer")) == LAMBDA_TRANSFER and float(objective.get("lambda_image")) == LAMBDA_IMAGE, "parent C objective changed")

    checkpoint_ref = _mapping(report.get("checkpoints", {}).get(str(PARENT_STEP)), "parent C2000 checkpoint reference")
    checkpoint_path = _path(root, checkpoint_ref["path"])
    _require(int(checkpoint_ref.get("step", -1)) == PARENT_STEP and int(checkpoint_ref.get("global_step", -1)) == GLOBAL_STEP_OFFSET + PARENT_STEP, "parent C checkpoint step binding changed")
    _validate_sha(checkpoint_path, str(checkpoint_ref["sha256"]), "parent C2000 checkpoint")
    checkpoint = _mapping(
        N4.safe_torch_load(checkpoint_path, expected_sha256=str(checkpoint_ref["sha256"]), map_location="cpu"),
        "parent C2000 checkpoint",
    )
    _require(
        checkpoint.get("schema") == BASE.CHECKPOINT_SCHEMA
        and checkpoint.get("task_id") == "LIGHTGEN_ALIGNED_4L_RELU_GAIN_V1"
        and checkpoint.get("candidate") == "C_4L_LINEAR"
        and checkpoint.get("checkpoint_step") == PARENT_STEP
        and checkpoint.get("global_step") == GLOBAL_STEP_OFFSET + PARENT_STEP
        and checkpoint.get("total_steps") == PARENT_STEP,
        "parent C2000 checkpoint identity changed",
    )
    _require(checkpoint.get("facet_mode") == "linear" and float(checkpoint.get("facet_tau")) == 0.0, "parent C facet identity changed")
    _require(checkpoint.get("trainable_parameter_names") == TRAINABLE_NAMES, "parent C trainable surface changed")
    schedule = _mapping(checkpoint.get("schedule_state"), "parent C schedule state")
    _require(schedule.get("last_update") == PARENT_STEP and schedule.get("stage") == "decoder_only_refinement", "parent C stage identity changed")
    _require(math.isclose(float(schedule.get("learning_rate")), PARENT_FINAL_LR, rel_tol=0.0, abs_tol=1.0e-15), "parent C final learning rate changed")
    runtime_state = _mapping(checkpoint.get("runtime_state"), "parent C runtime state")
    _require(set(runtime_state) == {"encoder.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport", "generator.phase"}, "parent C runtime state surface changed")
    _require(tuple(runtime_state["encoder.phase"].shape) == (2, 1, 100, 100), "parent C encoder shape changed")
    _require(tuple(runtime_state["generator.phase"].shape) == (GENERATOR_LAYERS, 1, 100, 100), "parent C generator shape changed")
    adapter_state = _mapping(checkpoint.get("adapter_state"), "parent C adapter state")
    _require(set(adapter_state) == {"u_real", "u_imag", "v_real", "v_imag"}, "parent C adapter state surface changed")
    _require(tuple(adapter_state["u_real"].shape) == (100, ADAPTER_RANK), "parent C adapter rank changed")
    optimizer_state = _mapping(checkpoint.get("optimizer_state"), "parent C optimizer state")
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    _require(isinstance(groups, list) and len(groups) == 1 and list(groups[0].get("params", [])) == [0, 1, 2, 3, 4], "parent C optimizer parameter order changed")
    _require(isinstance(states, Mapping) and {int(key) for key in states} == {0, 1, 2, 3, 4}, "parent C Adam moments are incomplete")
    sampler_state = _mapping(checkpoint.get("sampler_state"), "parent C sampler state")
    _require(sampler_state.get("schema") == BASE.ENGINE.BalancedMultiEpochSampler.SCHEMA, "parent C sampler schema changed")
    _require(sampler_state.get("seed") == SAMPLER_SEED and sampler_state.get("steps") == PARENT_STEP and sampler_state.get("cursor") == PARENT_STEP, "parent C sampler continuation identity changed")
    _require(sampler_state.get("cursor") == sampler_state.get("epoch") * sampler_state.get("per_class") + sampler_state.get("position"), "parent C sampler cursor relation changed")
    rng_state = _mapping(checkpoint.get("rng_state"), "parent C RNG state")
    _require(
        set(rng_state) == {"cpu", "cuda"}
        and torch.is_tensor(rng_state["cpu"])
        and isinstance(rng_state["cuda"], list)
        and all(torch.is_tensor(value) for value in rng_state["cuda"]),
        "parent C RNG state is incomplete",
    )
    _require(checkpoint.get("generator_geometry", {}).get("layers") == GENERATOR_LAYERS, "parent C checkpoint generator geometry changed")
    _require(float(checkpoint.get("generator_geometry", {}).get("inter_layer_spacing_m")) == GENERATOR_LAYER_SPACING_M, "parent C checkpoint spacing changed")

    sufficient_ref = _mapping(report.get("sufficient_observations", {}).get(str(PARENT_STEP)), "parent C2000 sufficient observation reference")
    sufficient_path = _path(root, sufficient_ref["path"])
    _validate_sha(sufficient_path, str(sufficient_ref["sha256"]), "parent C2000 sufficient observations")
    sufficient = _mapping(
        N4.safe_torch_load(sufficient_path, expected_sha256=str(sufficient_ref["sha256"]), map_location="cpu"),
        "parent C2000 sufficient observations",
    )
    _require(
        sufficient.get("schema") == BASE.SUFFICIENT_SCHEMA
        and sufficient.get("candidate") == "C_4L_LINEAR"
        and sufficient.get("global_step") == GLOBAL_STEP_OFFSET + PARENT_STEP,
        "parent C2000 sufficient observation identity changed",
    )
    _require(torch.is_tensor(sufficient.get("dev_labels")) and tuple(sufficient["dev_labels"].shape) == (5_000,), "parent C2000 DEV labels changed")
    raw_ref = report.get("endpoint_raw_outputs")
    _require(isinstance(raw_ref, Mapping), "parent C2000 raw endpoint reference is missing")
    raw_path = _path(root, raw_ref["path"])
    _validate_sha(raw_path, str(raw_ref["sha256"]), "parent C2000 cyclic raw outputs")
    return {
        "report": report,
        "report_ref": report_ref,
        "lineage": lineage,
        "lineage_ref": {"path": _relative(root, lineage_path), "sha256": PARENT_LINEAGE_SHA256},
        "contract": contract,
        "contract_ref": {"path": _relative(root, contract_path), "sha256": str(contract_ref["sha256"])},
        "checkpoint": dict(checkpoint),
        "checkpoint_ref": {"path": _relative(root, checkpoint_path), "sha256": str(checkpoint_ref["sha256"]), "step": PARENT_STEP, "global_step": GLOBAL_STEP_OFFSET + PARENT_STEP},
        "sufficient": dict(sufficient),
        "sufficient_ref": {"path": _relative(root, sufficient_path), "sha256": str(sufficient_ref["sha256"])},
        "raw_ref": {"path": _relative(root, raw_path), "sha256": str(raw_ref["sha256"])},
    }


def _extend_sampler_state(state: Mapping[str, Any], *, step: int) -> dict[str, Any]:
    extended = _cpu_tree(state)
    _require(extended.get("schema") == BASE.ENGINE.BalancedMultiEpochSampler.SCHEMA, "continuation sampler schema changed")
    _require(int(extended.get("seed", -1)) == SAMPLER_SEED, "continuation sampler seed changed")
    _require(int(extended.get("cursor", -1)) == step, "continuation sampler cursor does not match checkpoint")
    _require(int(extended.get("steps", -1)) in {PARENT_STEP, MAX_LOCAL_STEP}, "continuation sampler horizon is unexpected")
    extended["steps"] = MAX_LOCAL_STEP
    return extended


def _make_context(
    root: Path,
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
    *,
    step: int,
    checkpoint: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _require(PARENT_STEP <= step <= MAX_LOCAL_STEP, f"context step outside continuation range: {step}")
    if checkpoint is None:
        _require(step == PARENT_STEP, "only the registered C2000 parent may initialize a continuation context")
        state = source["checkpoint"]
        runtime_state = state["runtime_state"]
        adapter_state = state["adapter_state"]
        optimizer_state = state["optimizer_state"]
        sampler_state = _extend_sampler_state(state["sampler_state"], step=step)
        rng_state = state["rng_state"]
        source_checkpoint = state
        initial_phase = runtime_state["generator.phase"]
    else:
        _require(int(checkpoint.get("checkpoint_step", -1)) == step, "continuation checkpoint step does not match context step")
        _require(checkpoint.get("candidate") == "C_4L_LINEAR_CONTINUATION", "continuation checkpoint candidate changed")
        runtime_state = checkpoint["runtime_state"]
        adapter_state = checkpoint["adapter_state"]
        optimizer_state = checkpoint["optimizer_state"]
        sampler_state = _extend_sampler_state(checkpoint["sampler_state"], step=step)
        rng_state = checkpoint["rng_state"]
        source_checkpoint = checkpoint
        initial_phase = checkpoint.get("continuation_start_generator_phase")
        _require(initial_phase is not None, "continuation checkpoint lost the C2000 generator-phase baseline")
    V10.V6._restore_rng(_cpu_tree(rng_state))
    consumer, adapter, probe = BASE._new_consumer(
        bundle,
        {"checkpoint": source_checkpoint},
        device,
        mode="linear",
        tau=0.0,
        trainable=True,
        runtime_state=runtime_state,
        adapter_state=adapter_state,
    )
    optimizer = BASE._optimizer(consumer, adapter)
    BASE._restore_optimizer(optimizer, optimizer_state)
    V10.V6._restore_rng(_cpu_tree(rng_state))
    sampler = BASE.ENGINE.BalancedMultiEpochSampler(train_indices, labels, seed=SAMPLER_SEED, steps=MAX_LOCAL_STEP)
    sampler.load_state_dict(sampler_state)
    _require(sampler.cursor == step, "continuation sampler cursor changed after reload")
    expected_lr, expected_stage = _schedule_lr(step)
    _set_continuation_stage(consumer, adapter)
    optimizer.param_groups[0]["lr"] = expected_lr
    return {
        "consumer": consumer,
        "adapter": adapter,
        "probe": probe,
        "optimizer": optimizer,
        "sampler": sampler,
        "checkpoint": dict(source_checkpoint),
        "initial_phase": _cpu_tree(initial_phase),
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "bundle": bundle,
        "source": source,
        "contract_ref": None,
        "device": device,
        "tau": 0.0,
        "facet_mode": "linear",
        "candidate": "C_4L_LINEAR_CONTINUATION",
        "output_dir": SCIENTIFIC_DIR,
        "step": step,
        "expected_stage": expected_stage,
    }


def _release_context(context: Mapping[str, Any], device: torch.device) -> None:
    for value in context.values():
        if torch.is_tensor(value):
            del value
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _run_update(context: Mapping[str, Any], step: int, z: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor) -> dict[str, Any]:
    indices, batch_labels = context["sampler"].next_batch()
    _require(int(indices.numel()) == LOGICAL_BATCH_SIZE and int(batch_labels.numel()) == LOGICAL_BATCH_SIZE, "continuation logical batch changed")
    batch = context["images"].index_select(0, indices).to(context["device"])
    generator = dict(context["consumer"].runtime.named_parameters())["generator.phase"]
    phase_before = generator.detach().cpu().clone()
    trace = BASE._train_step(
        context["consumer"],
        context["adapter"],
        context["optimizer"],
        z,
        batch,
        batch_labels.to(context["device"]),
        prototypes,
        templates,
        step=step,
    )
    phase_after = generator.detach().cpu().clone()
    _require(generator.grad is not None, "continuation generator gradient disappeared")
    gradient = generator.grad.detach().cpu()
    phase_delta = phase_after - phase_before
    trace["gradient_norms"]["generator.phase_per_layer"] = [
        float(value)
        for value in gradient.to(torch.float64).square().sum(dim=(1, 2, 3)).sqrt().tolist()
    ]
    trace["update_norms"]["generator.phase_per_layer"] = [
        float(value)
        for value in phase_delta.to(torch.float64).square().sum(dim=(1, 2, 3)).sqrt().tolist()
    ]
    trace["batch_indices_sha256"] = V10.V6.tensor_sha256(indices)
    trace["batch_labels_sha256"] = V10.V6.tensor_sha256(batch_labels)
    _finite(gradient, "continuation generator gradient")
    _finite(phase_after, "continuation generator phase")
    del batch
    return trace


def _checkpoint_payload(
    root: Path,
    contract_ref: Mapping[str, Any] | None,
    source: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    step: int,
    last_trace: Mapping[str, Any] | None,
) -> dict[str, Any]:
    lr, stage = _schedule_lr(step)
    named = BASE._candidate_named(context["consumer"], context["adapter"])
    _require([name for name, parameter in named.items() if parameter.requires_grad] == TRAINABLE_NAMES, "continuation checkpoint trainable surface changed")
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "candidate": "C_4L_LINEAR_CONTINUATION",
        "contract": dict(contract_ref) if contract_ref is not None else None,
        "source_c2000": {
            "report": dict(source["report_ref"]),
            "lineage": dict(source["lineage_ref"]),
            "checkpoint": dict(source["checkpoint_ref"]),
        },
        "checkpoint_step": step,
        "local_scientific_step": step,
        "global_step": GLOBAL_STEP_OFFSET + step,
        "total_steps": MAX_LOCAL_STEP,
        "runtime_state": _cpu_tree(context["consumer"].runtime.state_dict()),
        "adapter_state": _cpu_tree(context["adapter"].state_dict()),
        "optimizer_state": _cpu_tree(context["optimizer"].state_dict()),
        "sampler_state": _cpu_tree(context["sampler"].state_dict()),
        "rng_state": _cpu_tree(V10.V6._capture_rng()),
        "schedule_state": {
            "last_update": step,
            "stage": stage,
            "stage_identity": STAGE_IDENTITY,
            "learning_rate": lr,
            "segments": [
                {"start": PARENT_STEP, "end": FIRST_TARGET_STEP, "lr_start": PARENT_FINAL_LR, "lr_end": FIRST_END_LR},
                {"start": FIRST_TARGET_STEP, "end": MAX_LOCAL_STEP, "lr_start": FIRST_END_LR, "lr_end": SECOND_END_LR},
            ],
        },
        "trainable_parameter_names": TRAINABLE_NAMES,
        "frozen_parameter_names": FROZEN_NAMES,
        "continuation_start_generator_phase": _cpu_tree(context["initial_phase"]),
        "objective": {
            "formula": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image",
            "lambda_transfer": LAMBDA_TRANSFER,
            "lambda_image": LAMBDA_IMAGE,
        },
        "generator_geometry": {"layers": GENERATOR_LAYERS, "inter_layer_spacing_m": GENERATOR_LAYER_SPACING_M, "propagation_grid": PRODUCTION_GRID},
        "sampler": {"schema": BASE.ENGINE.BalancedMultiEpochSampler.SCHEMA, "seed": SAMPLER_SEED, "logical_batch": LOGICAL_BATCH_SIZE, "micro_batch": MICRO_BATCH_SIZE},
        "last_train_trace": dict(last_trace) if last_trace is not None else None,
        "stage_b_executed": False,
        "sampled_generation": False,
        "protected_accessed": False,
        "official_test_accessed": False,
        "sealed_accessed": False,
        "formal_promotion": False,
    }


def _save_checkpoint(
    root: Path,
    path: Path,
    payload: Mapping[str, Any],
    *,
    step: int,
    artifact_kind: str,
) -> dict[str, Any]:
    ref = _save_tensor(root, path, payload, artifact_kind)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=ref["sha256"], map_location="cpu"), f"{artifact_kind} reload")
    for key in ("runtime_state", "adapter_state", "optimizer_state", "sampler_state", "rng_state", "schedule_state"):
        _require(_tree_equal(loaded[key], payload[key]), f"{artifact_kind} reload changed {key}")
    _require(loaded.get("checkpoint_step") == step and loaded.get("global_step") == GLOBAL_STEP_OFFSET + step, f"{artifact_kind} step identity changed")
    _require(int(loaded["sampler_state"].get("cursor", -1)) == step, f"{artifact_kind} sampler cursor changed")
    return {**ref, "step": step, "global_step": GLOBAL_STEP_OFFSET + step, "reloaded": True}


def _save_progress(
    root: Path,
    output_dir: Path,
    *,
    phase: str,
    status: str,
    step: int,
    rows_done: int | None = None,
    total_rows: int | None = None,
    elapsed_seconds: float | None = None,
    checkpoint: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, str]:
    token = f"{time.time_ns():019d}"
    safe_phase = "".join(character if character.isalnum() else "_" for character in phase)
    path = output_dir / "progress" / f"c_4l_linear_continuation_{safe_phase}_{token}.v1.json"
    payload: dict[str, Any] = {
        "schema": PROGRESS_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "candidate": "C_4L_LINEAR_CONTINUATION",
        "phase": phase,
        "status": status,
        "step": step,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_done": rows_done,
        "total_rows": total_rows,
        "elapsed_seconds": elapsed_seconds,
        "checkpoint": dict(checkpoint) if checkpoint is not None else None,
    }
    if details:
        payload["details"] = dict(details)
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    return _save_json(root, path, payload, "continuation progress event")


def _trace_summary(traces: list[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(traces), "continuation block has no train traces")
    scalar_keys = ("L_anchor", "L_transfer", "L_image", "L_sem", "L_total", "learning_rate")
    means = {key: float(sum(float(trace[key]) for trace in traces) / len(traces)) for key in scalar_keys}
    generator_gradients = [
        float(trace["gradient_norms"]["generator.phase"])
        for trace in traces
    ]
    phase_updates = [
        float(trace["update_norms"]["generator.phase"])
        for trace in traces
    ]
    per_layer_grad = [
        float(sum(float(trace["gradient_norms"]["generator.phase_per_layer"][layer]) for trace in traces) / len(traces))
        for layer in range(GENERATOR_LAYERS)
    ]
    per_layer_update = [
        float(sum(float(trace["update_norms"]["generator.phase_per_layer"][layer]) for trace in traces) / len(traces))
        for layer in range(GENERATOR_LAYERS)
    ]
    return {
        "count": len(traces),
        "mean": means,
        "last": dict(traces[-1]),
        "generator_gradient_norm_mean": float(sum(generator_gradients) / len(generator_gradients)),
        "generator_gradient_norm_last": generator_gradients[-1],
        "generator_phase_update_norm_mean": float(sum(phase_updates) / len(phase_updates)),
        "generator_phase_update_norm_last": phase_updates[-1],
        "generator_gradient_norm_per_layer_mean": per_layer_grad,
        "generator_phase_update_norm_per_layer_mean": per_layer_update,
    }


def _run_training_block(
    root: Path,
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    contract_ref: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
    *,
    current_step: int,
    target_step: int,
    output_dir: Path,
) -> dict[str, Any]:
    _require(current_step >= PARENT_STEP and target_step > current_step and target_step <= MAX_LOCAL_STEP, "continuation block bounds are invalid")
    _require(current_step % BLOCK_SIZE == 0 and target_step % BLOCK_SIZE == 0, "continuation block is not aligned to the checkpoint cadence")
    context = _make_context(
        root,
        bundle,
        source,
        images,
        labels,
        train_indices,
        device,
        step=current_step,
        checkpoint=None if current_step == PARENT_STEP else _load_continuation_checkpoint(root, output_dir, current_step),
    )
    context["contract_ref"] = contract_ref
    prototypes = bundle["authorities"]["prototypes"].to(device)
    templates = bundle["authorities"]["templates"].to(device)
    z = torch.zeros((LOGICAL_BATCH_SIZE, BASE.OLS_PORT_COUNT), dtype=torch.complex64, device=device)
    started = time.perf_counter()
    _save_progress(root, output_dir, phase="training_block", status="started", step=current_step, rows_done=current_step, total_rows=target_step)
    traces: list[dict[str, Any]] = []
    try:
        for step in range(current_step + 1, target_step + 1):
            traces.append(_run_update(context, step, z, prototypes, templates))
    except BaseException as error:
        _save_progress(
            root,
            output_dir,
            phase="training_block",
            status="failed",
            step=current_step,
            rows_done=int(context["sampler"].cursor),
            total_rows=target_step,
            elapsed_seconds=time.perf_counter() - started,
            error=error,
        )
        _release_context(context, device)
        raise
    _require(context["sampler"].cursor == target_step, "continuation block sampler cursor did not reach target")
    checkpoint = _save_checkpoint(
        root,
        _checkpoint_path(output_dir, target_step),
        _checkpoint_payload(root, contract_ref, source, context, step=target_step, last_trace=traces[-1]),
        step=target_step,
        artifact_kind=f"C continuation checkpoint {target_step}",
    )
    elapsed = time.perf_counter() - started
    summary = _trace_summary(traces)
    block_payload = {
        "schema": BLOCK_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "candidate": "C_4L_LINEAR_CONTINUATION",
        "current_step": current_step,
        "target_step": target_step,
        "global_step_start": GLOBAL_STEP_OFFSET + current_step + 1,
        "global_step_end": GLOBAL_STEP_OFFSET + target_step,
        "training_seconds": elapsed,
        "seconds_per_update": elapsed / float(target_step - current_step),
        "summary": summary,
        "checkpoint": checkpoint,
        "finite": True,
        "frozen_groups_unchanged": True,
    }
    block_path = output_dir / "training_blocks" / f"c_4l_linear_continuation_{current_step + 1:05d}_{target_step:05d}.v1.json"
    block_ref = _save_json(root, block_path, block_payload, f"C continuation training block {target_step}")
    _save_progress(
        root,
        output_dir,
        phase="training_block",
        status="checkpointed",
        step=target_step,
        rows_done=target_step,
        total_rows=target_step,
        elapsed_seconds=elapsed,
        checkpoint=checkpoint,
        details={"block": block_ref, "seconds_per_update": elapsed / float(target_step - current_step)},
    )
    _release_context(context, device)
    del prototypes, templates, z
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "current_step": current_step,
        "target_step": target_step,
        "checkpoint": checkpoint,
        "block": block_ref,
        "summary": summary,
        "training_seconds": elapsed,
        "seconds_per_update": elapsed / float(target_step - current_step),
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
    }


def _load_continuation_checkpoint(root: Path, output_dir: Path, step: int) -> dict[str, Any]:
    path = _checkpoint_path(output_dir, step)
    _require(path.is_file(), f"continuation checkpoint {step} is unavailable")
    digest = _sha256(path)
    checkpoint = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"continuation checkpoint {step}")
    _require(
        checkpoint.get("schema") == CHECKPOINT_SCHEMA
        and checkpoint.get("candidate") == "C_4L_LINEAR_CONTINUATION"
        and checkpoint.get("checkpoint_step") == step
        and checkpoint.get("global_step") == GLOBAL_STEP_OFFSET + step
        and checkpoint.get("total_steps") == MAX_LOCAL_STEP,
        f"continuation checkpoint {step} identity changed",
    )
    _require(int(checkpoint["sampler_state"].get("cursor", -1)) == step, f"continuation checkpoint {step} sampler cursor changed")
    return {**dict(checkpoint), "_ref": {"path": _relative(root, path), "sha256": digest, "step": step, "global_step": GLOBAL_STEP_OFFSET + step}}


def _raw_power_summary(raw: torch.Tensor) -> dict[str, float]:
    _finite(raw, "raw CMOS power input")
    flat = raw.detach().to(torch.float64).flatten(1)
    power = flat.sum(dim=1)
    _finite(power, "raw CMOS power")
    return {
        "rows": int(power.numel()),
        "mean": float(power.mean()),
        "std": float(power.std(unbiased=False)),
        "min": float(power.min()),
        "max": float(power.max()),
        "total": float(power.sum()),
    }


def _source_raw_power(root: Path, source: Mapping[str, Any]) -> dict[str, float]:
    path = _path(root, source["raw_ref"]["path"])
    raw_payload = _mapping(N4.safe_torch_load(path, expected_sha256=source["raw_ref"]["sha256"], map_location="cpu"), "parent C raw outputs")
    raw = raw_payload.get("raw_base")
    _require(torch.is_tensor(raw), "parent C raw base output is missing")
    summary = _raw_power_summary(raw)
    del raw_payload, raw
    gc.collect()
    return summary


def _run_endpoint_from_checkpoint(
    root: Path,
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    contract_ref: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    dev_indices: torch.Tensor,
    device: torch.device,
    *,
    step: int,
    output_dir: Path,
) -> dict[str, Any]:
    checkpoint = _load_continuation_checkpoint(root, output_dir, step)
    context = _make_context(root, bundle, source, images, labels, train_indices, device, step=step, checkpoint=checkpoint)
    context["contract_ref"] = contract_ref
    captured: dict[str, Any] = {}
    original_diversity = BASE.ENGINE._direct_diversity

    def _capture_diversity(raw: torch.Tensor, raw_labels: torch.Tensor) -> dict[str, Any]:
        captured["raw_cmos_power"] = _raw_power_summary(raw)
        return original_diversity(raw, raw_labels)

    BASE.ENGINE._direct_diversity = _capture_diversity
    try:
        record = BASE._run_endpoint_for_context(root, bundle, context, dev_indices, step=step)
    finally:
        BASE.ENGINE._direct_diversity = original_diversity
    _require("raw_cmos_power" in captured, f"C continuation endpoint {step} raw CMOS power was not captured")
    record["raw_cmos_power"] = captured["raw_cmos_power"]
    endpoint_meta = {
        "schema": ENDPOINT_META_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "candidate": "C_4L_LINEAR_CONTINUATION",
        "step": step,
        "global_step": GLOBAL_STEP_OFFSET + step,
        "sufficient_observations": record["sufficient_ref"],
        "diagnostics": record["diagnostic_ref"],
        "checkpoint": {"path": _relative(root, _checkpoint_path(output_dir, step)), "sha256": _sha256(_checkpoint_path(output_dir, step)), "step": step, "global_step": GLOBAL_STEP_OFFSET + step},
        "raw_cmos_power": captured["raw_cmos_power"],
        "evaluation_raw_validity": record["evaluation_public"].get("raw_validity"),
        "cyclic_raw_validity": record["cyclic_public"].get("raw_validity"),
        "cyclic_transfer_pass": BASE._cyclic_pass(record["cyclic_public"]),
    }
    meta_ref = _save_json(root, _endpoint_meta_path(output_dir, step), endpoint_meta, f"C continuation endpoint meta {step}")
    record["endpoint_meta_ref"] = meta_ref
    _release_context(context, device)
    return record


def _load_endpoint(root: Path, output_dir: Path, step: int) -> dict[str, Any]:
    sufficient_path = _sufficient_path(output_dir, step)
    _require(sufficient_path.is_file(), f"C continuation sufficient observations {step} are unavailable")
    sufficient_digest = _sha256(sufficient_path)
    sufficient = _mapping(N4.safe_torch_load(sufficient_path, expected_sha256=sufficient_digest, map_location="cpu"), f"C continuation sufficient observations {step}")
    _require(sufficient.get("schema") == SUFFICIENT_SCHEMA and sufficient.get("candidate") == "C_4L_LINEAR_CONTINUATION" and sufficient.get("global_step") == GLOBAL_STEP_OFFSET + step, f"C continuation sufficient identity changed at {step}")
    diagnostic_path = _diagnostic_path(output_dir, step)
    diagnostic = _read_json(diagnostic_path, f"C continuation diagnostics {step}")
    _require(diagnostic.get("schema") == DIAGNOSTIC_SCHEMA and diagnostic.get("candidate") == "C_4L_LINEAR_CONTINUATION" and int(diagnostic.get("step", -1)) == step, f"C continuation diagnostic identity changed at {step}")
    meta = _read_json(_endpoint_meta_path(output_dir, step), f"C continuation endpoint meta {step}")
    _require(meta.get("schema") == ENDPOINT_META_SCHEMA and meta.get("candidate") == "C_4L_LINEAR_CONTINUATION" and meta.get("step") == step, f"C continuation endpoint meta identity changed at {step}")
    checkpoint_path = _checkpoint_path(output_dir, step)
    _require(checkpoint_path.is_file(), f"C continuation endpoint checkpoint {step} is unavailable")
    raw_ref = None
    if step == MAX_LOCAL_STEP:
        raw_path = _raw_path(output_dir, step)
        _require(raw_path.is_file(), "C4000 cyclic raw outputs are unavailable")
        raw_ref = {"path": _relative(root, raw_path), "sha256": _sha256(raw_path)}
    return {
        "step": step,
        "candidate": "C_4L_LINEAR_CONTINUATION",
        "dev_labels": sufficient["dev_labels"],
        "dev_correct": sufficient["dev_smooth_correct"],
        "evaluation_public": sufficient["evaluation_public"],
        "cyclic_public": sufficient["cyclic_public"],
        "dev_image_loss": float(sufficient["dev_image_loss"]),
        "train_metric": sufficient.get("train_metric", {}),
        "sufficient_ref": {"path": _relative(root, sufficient_path), "sha256": sufficient_digest},
        "diagnostic_ref": {"path": _relative(root, diagnostic_path), "sha256": _sha256(diagnostic_path)},
        "endpoint_meta_ref": {"path": _relative(root, _endpoint_meta_path(output_dir, step)), "sha256": _sha256(_endpoint_meta_path(output_dir, step))},
        "checkpoint_ref": {"path": _relative(root, checkpoint_path), "sha256": _sha256(checkpoint_path), "step": step, "global_step": GLOBAL_STEP_OFFSET + step},
        "raw_ref": raw_ref,
        "raw_cmos_power": meta["raw_cmos_power"],
        "generator_diagnostics": diagnostic.get("diagnostics", {}).get("generator", {}),
        "diagnostics": diagnostic.get("diagnostics", {}),
    }


def _source_endpoint(source: Mapping[str, Any], raw_power: Mapping[str, Any]) -> dict[str, Any]:
    report = source["report"]
    diagnostic_ref = _mapping(report.get("diagnostics", {}).get(str(PARENT_STEP)), "parent C diagnostic reference")
    return {
        "step": PARENT_STEP,
        "candidate": "C_4L_LINEAR",
        "dev_labels": source["sufficient"]["dev_labels"],
        "dev_correct": source["sufficient"]["dev_smooth_correct"],
        "evaluation_public": source["sufficient"]["evaluation_public"],
        "cyclic_public": source["sufficient"]["cyclic_public"],
        "dev_image_loss": float(source["sufficient"]["dev_image_loss"]),
        "train_metric": source["sufficient"].get("train_metric", {}),
        "sufficient_ref": dict(source["sufficient_ref"]),
        "diagnostic_ref": dict(diagnostic_ref),
        "checkpoint_ref": dict(source["checkpoint_ref"]),
        "raw_ref": dict(source["raw_ref"]),
        "raw_cmos_power": dict(raw_power),
        "generator_diagnostics": {},
        "diagnostics": {},
        "raw_validity_pass": bool(source["report"].get("raw_validity_pass")),
    }


def _raw_validity_pass(record: Mapping[str, Any]) -> bool:
    evaluation = _mapping(record.get("evaluation_public"), "endpoint evaluation")
    cyclic = _mapping(record.get("cyclic_public"), "endpoint cyclic transfer")
    raw = _mapping(evaluation.get("raw_validity"), "endpoint evaluation raw validity")
    cyclic_raw = _mapping(cyclic.get("raw_validity"), "endpoint cyclic raw validity")
    return bool(
        raw.get("finite")
        and raw.get("structured")
        and raw.get("nonconstant")
        and cyclic_raw.get("base", {}).get("finite")
        and cyclic_raw.get("shifted", {}).get("finite")
    )


def _class_summary(record: Mapping[str, Any]) -> dict[str, Any]:
    labels = record["dev_labels"].to(torch.long).cpu()
    correct = record["dev_correct"].to(torch.bool).cpu()
    recalls = []
    for digit in range(10):
        values = correct[labels == digit]
        _require(values.numel() > 0, f"endpoint class {digit} is empty")
        recalls.append(float(values.to(torch.float64).mean()))
    ordered = sorted(recalls)
    return {
        "class_recall": [{"class": digit, "recall": recalls[digit], "count": int((labels == digit).sum())} for digit in range(10)],
        "macro_recall": float(sum(recalls) / len(recalls)),
        "minimum_class_recall": float(min(recalls)),
        "worst_3_mean_recall": float(sum(ordered[:3]) / 3.0),
        "strongest_3_mean_recall": float(sum(ordered[-3:]) / 3.0),
        "weakest_classes": [digit for digit, value in enumerate(recalls) if value == min(recalls)],
    }


def _paired_bootstrap(delta: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    value = V10.V6._paired_bootstrap(delta.to(torch.float64), labels.to(torch.long), replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    return {key: item for key, item in value.items() if key != "delta_records"}


def _contrast(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    labels = parent["dev_labels"].to(torch.long)
    _require(torch.equal(labels, child["dev_labels"].to(torch.long)), f"DEV labels are not matched for step {child['step']}")
    delta = child["dev_correct"].to(torch.float64) - parent["dev_correct"].to(torch.float64)
    return {
        "from_step": int(parent["step"]),
        "to_step": int(child["step"]),
        "point": float(delta.mean()),
        "paired_bootstrap": _paired_bootstrap(delta, labels),
        "image_delta": float(child["dev_image_loss"] - parent["dev_image_loss"]),
        "class_delta": [
            {
                "class": digit,
                "delta": float(delta[labels == digit].mean()),
            }
            for digit in range(10)
        ],
    }


def _load_contract(root: Path) -> dict[str, Any]:
    report = _read_json(PREFLIGHT_REPORT_PATH if PREFLIGHT_REPORT_PATH.is_absolute() else root / PREFLIGHT_REPORT_PATH, "continuation preflight report")
    lineage = _read_json(root / PREFLIGHT_LINEAGE_PATH, "continuation preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed", "continuation preflight is not authorizable")
    report_path = root / PREFLIGHT_REPORT_PATH
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    _require(lineage.get("report") == report_ref, "continuation preflight lineage binding changed")
    contract_ref = _mapping(report.get("contract"), "continuation contract reference")
    contract_path = _path(root, contract_ref["path"])
    _validate_sha(contract_path, str(contract_ref["sha256"]), "continuation contract")
    contract = _read_json(contract_path, "continuation contract")
    _require(contract.get("task_id") == TASK_ID and contract.get("outer_gate") == OUTER_GATE, "continuation contract identity changed")
    _require(contract.get("status") == "READY_FOR_BOUNDED_CONTINUATION", "continuation contract is not READY")
    _require(contract.get("source_authority", {}).get("checkpoint", {}).get("path") == report.get("source_authority", {}).get("checkpoint", {}).get("path"), "continuation source checkpoint binding changed")
    _require(report.get("claim_boundary", {}).get("scientific_updates") == 0, "continuation preflight scientific boundary changed")
    return {
        "report": report,
        "report_ref": report_ref,
        "lineage_ref": {"path": _relative(root, root / PREFLIGHT_LINEAGE_PATH), "sha256": _sha256(root / PREFLIGHT_LINEAGE_PATH)},
        "contract": contract,
        "contract_ref": dict(contract_ref),
    }


def _preflight_contract(
    started: Mapping[str, Any],
    source: Mapping[str, Any],
    dataset: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    resume: Mapping[str, Any],
    negative: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "lightgen_aligned_4l_linear_optimization_ceiling_v1_contract.v1",
        "version": 1,
        "task_id": TASK_ID,
        "design_authority": DESIGN_AUTHORITY,
        "status": "READY_FOR_BOUNDED_CONTINUATION",
        "outer_gate": OUTER_GATE,
        "git_boundary": {"branch": started["branch"], "head": started["head"]},
        "source_authority": {
            "report": dict(source["report_ref"]),
            "lineage": dict(source["lineage_ref"]),
            "contract": dict(source["contract_ref"]),
            "checkpoint": dict(source["checkpoint_ref"]),
            "sufficient_observations": dict(source["sufficient_ref"]),
            "raw_outputs": dict(source["raw_ref"]),
        },
        "objective": {
            "formula": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image",
            "lambda_transfer": LAMBDA_TRANSFER,
            "lambda_image": LAMBDA_IMAGE,
            "tuning": False,
        },
        "architecture": {
            "variant": "LIGHTGEN_TOPOLOGY_ALIGNED_4L",
            "generator_layers": GENERATOR_LAYERS,
            "generator_layer_spacing_m": GENERATOR_LAYER_SPACING_M,
            "propagation_grid": PRODUCTION_GRID,
            "encoder": "frozen 2-layer encoder",
            "ols": "frozen complex 100-port OLS",
            "adapter": "frozen rank-9 adapter",
            "w_ols": "frozen modal re-emission",
            "facet": "LINEAR; frozen",
            "neuron_and_detector_conventions": "inherited exactly from parent C authority",
        },
        "continuation": {
            "source_local_step": PARENT_STEP,
            "source_global_step": GLOBAL_STEP_OFFSET + PARENT_STEP,
            "first_target_local_step": FIRST_TARGET_STEP,
            "first_target_global_step": GLOBAL_STEP_OFFSET + FIRST_TARGET_STEP,
            "maximum_local_step": MAX_LOCAL_STEP,
            "maximum_global_step": GLOBAL_STEP_OFFSET + MAX_LOCAL_STEP,
            "minimum_new_updates": FIRST_SEGMENT_UPDATES,
            "maximum_new_updates": FIRST_SEGMENT_UPDATES + SECOND_SEGMENT_UPDATES,
            "trainable": TRAINABLE_NAMES,
            "frozen": FROZEN_NAMES,
            "stage_identity": STAGE_IDENTITY,
            "lr_schedule": [
                {"local_start": PARENT_STEP, "local_end": FIRST_TARGET_STEP, "lr_start": PARENT_FINAL_LR, "lr_end": FIRST_END_LR},
                {"local_start": FIRST_TARGET_STEP, "local_end": MAX_LOCAL_STEP, "lr_start": FIRST_END_LR, "lr_end": SECOND_END_LR},
            ],
            "optimizer": {"name": "Adam", "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "restore_moments": True},
            "batch": LOGICAL_BATCH_SIZE,
            "micro_batch": MICRO_BATCH_SIZE,
            "sampler": {"schema": BASE.ENGINE.BalancedMultiEpochSampler.SCHEMA, "seed": SAMPLER_SEED, "resume_parent_cursor": PARENT_STEP, "horizon": MAX_LOCAL_STEP},
            "checkpoint_every_steps": BLOCK_SIZE,
        },
        "stopping_rule": {
            "c3000": {
                "delta_2k_3k_min": CONTINUATION_MIN_GAIN,
                "paired_ci_lower_gt": 0.0,
                "image_delta_max": RECONSTRUCTION_MAX_DELTA,
                "raw_validity_and_cyclic_transfer": True,
            },
            "c4000": {
                "delta_2k_4k_min": CONTINUED_TOTAL_MIN_GAIN,
                "paired_ci_lower_gt": 0.0,
                "late_gain_3k_4k_min": CONTINUED_LATE_MIN_GAIN,
                "image_delta_2k_4k_max": RECONSTRUCTION_MAX_DELTA,
                "raw_validity_and_cyclic_transfer": True,
            },
            "historical_2l_reference": HISTORICAL_2L_REFERENCE,
            "reference_comparison_is_descriptive_only": True,
        },
        "preflight": {
            "benchmark": benchmark,
            "checkpoint_reload": True,
            "optimizer_reload": True,
            "sampler_rng_reload": True,
            "three_step_real_update": True,
            "frozen_groups_unchanged": True,
            "negative_control": negative,
            "resume_equivalence": resume,
            "data_roles": _plain(dataset),
        },
        "claim_boundary": {
            "4l_linear_only": True,
            "relu_training": False,
            "physical_te_validation": False,
            "stage_b": False,
            "sampled_generation": False,
            "protected": False,
            "official_test": False,
            "sealed": False,
            "formal_promotion": False,
        },
    }


def _preflight_payload(
    started: Mapping[str, Any],
    *,
    result: str,
    status: str,
    classification: str,
    contract_ref: Mapping[str, Any] | None = None,
    details: Mapping[str, Any] | None = None,
    error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": "lightgen_aligned_4l_linear_optimization_ceiling_v1_preflight_report.v1",
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": result,
        "status": status,
        "classification": classification,
        "outer_gate": OUTER_GATE,
        "contract": dict(contract_ref) if contract_ref is not None else None,
        "provenance": {"git_start": dict(started)},
        "claim_boundary": {
            "engineering_preflight_executed": result == "AUTHORIZABLE",
            "scientific_training_executed": False,
            "scientific_updates": 0,
            "protected_accessed": False,
            "official_test_accessed": False,
            "sealed_accessed": False,
            "stage_b": False,
            "sampled_generation": False,
            "formal_promotion": False,
        },
    }
    if details:
        payload.update(dict(details))
    if error is not None:
        payload["error"] = dict(error)
    return payload


def _negative_control(
    root: Path,
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    context = _make_context(
        root,
        bundle,
        source,
        images,
        labels,
        train_indices,
        device,
        step=PARENT_STEP,
        checkpoint=None,
    )
    batch_indices = train_indices[:MICRO_BATCH_SIZE]
    batch = images.index_select(0, batch_indices).to(device)
    batch_labels = labels.index_select(0, batch_indices).to(device)
    z = torch.zeros((MICRO_BATCH_SIZE, BASE.OLS_PORT_COUNT), dtype=torch.complex64, device=device)
    components = BASE.ENGINE._components(
        context["consumer"],
        context["adapter"],
        z,
        batch,
        batch_labels,
        bundle["authorities"]["prototypes"].to(device),
        bundle["authorities"]["templates"].to(device),
        use_cache=False,
        require_grad=True,
    )
    generator = dict(context["consumer"].runtime.named_parameters())["generator.phase"]
    wrong_labels = (batch_labels + 1) % 10
    wrong_anchor = F.cross_entropy(components["anchor_logits"], wrong_labels)
    wrong_direction = F.softplus(components["delta"]).mean()
    anchor_grad = torch.autograd.grad(wrong_anchor, generator, retain_graph=True, allow_unused=False)[0]
    direction_grad = torch.autograd.grad(wrong_direction, generator, allow_unused=False)[0]
    _finite(anchor_grad, "wrong-label generator gradient")
    _finite(direction_grad, "wrong-direction generator gradient")
    result = {
        "wrong_label_generator_gradient_norm": float(anchor_grad.to(torch.float64).square().sum().sqrt().cpu()),
        "wrong_direction_generator_gradient_norm": float(direction_grad.to(torch.float64).square().sum().sqrt().cpu()),
        "runtime_gradients_none": all(parameter.grad is None for parameter in context["consumer"].runtime.parameters()),
    }
    _require(result["wrong_label_generator_gradient_norm"] > 0.0 and result["wrong_direction_generator_gradient_norm"] > 0.0, "continuation negative control has zero generator gradient")
    _require(result["runtime_gradients_none"], "continuation negative control touched frozen runtime gradients")
    _release_context(context, device)
    del batch, batch_labels, z, components
    return result


def _resume_preflight(
    root: Path,
    bundle: Mapping[str, Any],
    source: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    output_dir = PREFLIGHT_DIR
    context_a = _make_context(root, bundle, source, images, labels, train_indices, device, step=PARENT_STEP, checkpoint=None)
    prototypes = bundle["authorities"]["prototypes"].to(device)
    templates = bundle["authorities"]["templates"].to(device)
    z = torch.zeros((LOGICAL_BATCH_SIZE, BASE.OLS_PORT_COUNT), dtype=torch.complex64, device=device)
    traces_a: list[dict[str, Any]] = []
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    traces_a.append(_run_update(context_a, PARENT_STEP + 1, z, prototypes, templates))
    checkpoint_path = output_dir / "continuation_preflight_resume_step_02001.v1.pt"
    payload = _checkpoint_payload(root, None, source, context_a, step=PARENT_STEP + 1, last_trace=traces_a[-1])
    checkpoint_ref = _save_checkpoint(root, checkpoint_path, payload, step=PARENT_STEP + 1, artifact_kind="continuation preflight resume checkpoint")
    loaded_checkpoint = _mapping(
        N4.safe_torch_load(checkpoint_path, expected_sha256=checkpoint_ref["sha256"], map_location="cpu"),
        "continuation preflight resume checkpoint",
    )
    traces_a.append(_run_update(context_a, PARENT_STEP + 2, z, prototypes, templates))
    traces_a.append(_run_update(context_a, PARENT_STEP + 3, z, prototypes, templates))
    final_a = {
        "runtime": _cpu_tree(context_a["consumer"].runtime.state_dict()),
        "adapter": _cpu_tree(context_a["adapter"].state_dict()),
        "optimizer": _cpu_tree(context_a["optimizer"].state_dict()),
        "sampler": _cpu_tree(context_a["sampler"].state_dict()),
        "rng": _cpu_tree(V10.V6._capture_rng()),
    }
    context_b = _make_context(root, bundle, source, images, labels, train_indices, device, step=PARENT_STEP + 1, checkpoint=loaded_checkpoint)
    traces_b = [
        _run_update(context_b, PARENT_STEP + 2, z, prototypes, templates),
        _run_update(context_b, PARENT_STEP + 3, z, prototypes, templates),
    ]
    final_b = {
        "runtime": _cpu_tree(context_b["consumer"].runtime.state_dict()),
        "adapter": _cpu_tree(context_b["adapter"].state_dict()),
        "optimizer": _cpu_tree(context_b["optimizer"].state_dict()),
        "sampler": _cpu_tree(context_b["sampler"].state_dict()),
        "rng": _cpu_tree(V10.V6._capture_rng()),
    }
    _require(_tree_equal(final_a, final_b), "interrupted/resumed preflight state is not exact")
    for left, right in zip(traces_a[1:], traces_b, strict=True):
        for key in ("step", "global_step", "batch_indices_sha256", "batch_labels_sha256"):
            _require(left[key] == right[key], f"interrupted/resumed next-batch identity changed at {key}")
        for key in ("learning_rate", "L_anchor", "L_transfer", "L_image", "L_sem", "L_total"):
            _require(math.isclose(float(left[key]), float(right[key]), rel_tol=0.0, abs_tol=1.0e-12), f"interrupted/resumed trace changed at {key}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = int(torch.cuda.max_memory_allocated(device))
    else:
        peak_memory = 0
    elapsed = time.perf_counter() - started
    _release_context(context_a, device)
    _release_context(context_b, device)
    del prototypes, templates, z, loaded_checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "updates": [PARENT_STEP + 1, PARENT_STEP + 2, PARENT_STEP + 3],
        "checkpoint": checkpoint_ref,
        "exact_state_match": True,
        "exact_next_batch_match": True,
        "elapsed_seconds": elapsed,
        "seconds_per_update": elapsed / 3.0,
        "peak_memory_bytes": peak_memory,
        "traces": traces_a,
    }


def run_preflight(root: Path) -> dict[str, Any]:
    started = V10.V6.git_facts(root)
    status_before = V10.V6._status_lines(root)
    contract_ref: dict[str, Any] | None = None
    if PREFLIGHT_DIR.exists() or CONTRACT_DIR.exists() or SCIENTIFIC_DIR.exists() or AGGREGATE_DIR.exists():
        raise InvalidContinuation("continuation output boundary is not fresh; no artifact was overwritten")
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=False)
    old_base: dict[str, Any] | None = None
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from the registered continuation boundary")
        source = _load_parent_authority(root)
        old_base = _patch_base_for_continuation()
        BASE.V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = BASE.V10.V6._cuda_facts()
        _require(device.type == "cuda", "CUDA is required for the continuation preflight")
        authority = BASE._authorities(root)
        bundle = authority["bundle"]
        bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, dev_indices, dataset = BASE.V10.V6._load_roles(root)
        _require(dataset.get("TRAIN", {}).get("rows") == 55_000 and dataset.get("DEV", {}).get("rows") == 5_000, "continuation data roles changed")
        _require(dataset.get("protected", 0) in (0, [], None) and dataset.get("sealed", 0) in (0, [], None), "continuation protected or sealed role is nonempty")
        current_dev_labels = labels.index_select(0, dev_indices).cpu()
        _require(torch.equal(current_dev_labels, source["sufficient"]["dev_labels"].to(torch.long).cpu()), "current DEV role does not match parent C2000 DEV labels")
        negative = _negative_control(root, bundle, source, images, labels, train_indices, device)
        resume = _resume_preflight(root, bundle, source, images, labels, train_indices, device)
        estimate = float(resume["seconds_per_update"] * FIRST_SEGMENT_UPDATES)
        if estimate > RESOURCE_WARNING_SECONDS:
            raise ResourceWarningContinuation(
                "1000-step continuation exceeds the 3-hour active-GPU resource boundary",
                estimated_active_gpu_seconds=estimate,
                threshold_seconds=RESOURCE_WARNING_SECONDS,
            )
        benchmark = {
            "device": device_facts,
            "three_step_seconds_per_update": resume["seconds_per_update"],
            "estimated_active_gpu_seconds_c2000_to_c3000": estimate,
            "resource_warning_threshold_seconds": RESOURCE_WARNING_SECONDS,
            "peak_memory_bytes": resume["peak_memory_bytes"],
            "vram_bytes": int(device_facts.get("vram_bytes", 0)),
            "peak_memory_headroom": int(device_facts.get("vram_bytes", 0)) - resume["peak_memory_bytes"],
        }
        _require(benchmark["vram_bytes"] <= 0 or benchmark["peak_memory_bytes"] < benchmark["vram_bytes"], "continuation preflight peak VRAM exceeds device capacity")
        contract = _preflight_contract(started, source, dataset, benchmark, resume, negative)
        contract_ref = _save_json(root, CONTRACT_PATH, contract, "continuation contract")
        ending = V10.V6.git_facts(root)
        details = {
            "source_authority": {
                "report": source["report_ref"],
                "lineage": source["lineage_ref"],
                "contract": source["contract_ref"],
                "checkpoint": source["checkpoint_ref"],
                "sufficient_observations": source["sufficient_ref"],
                "raw_outputs": source["raw_ref"],
            },
            "runtime": {"device": device_facts, "production_grid": PRODUCTION_GRID},
            "data_roles": _plain(dataset),
            "benchmark": benchmark,
            "negative_control": negative,
            "resume_equivalence": resume,
            "contract": contract_ref,
            "provenance": {
                "git_start": started,
                "git_end": ending,
                "worktree_status_before": status_before,
                "worktree_status_after": V10.V6._status_lines(root),
                "branch_unchanged": started.get("branch") == ending.get("branch"),
                "head_unchanged": started.get("head") == ending.get("head"),
            },
        }
        payload = _preflight_payload(
            started,
            result="AUTHORIZABLE",
            status="valid_completed",
            classification="4L_LINEAR_CONTINUATION_PREFLIGHT_AUTHORIZABLE",
            contract_ref=contract_ref,
            details=details,
        )
    except HardBlockedContinuation as error:
        payload = _preflight_payload(
            started,
            result="BLOCKED",
            status="blocked",
            classification=error.code,
            error={"type": type(error).__name__, "message": str(error), "details": error.details},
        )
    except ResourceWarningContinuation as error:
        payload = _preflight_payload(
            started,
            result="RESOURCE_WARNING",
            status="resource_aborted",
            classification="4L_LINEAR_CONTINUATION_RESOURCE_WARNING",
            error={"type": type(error).__name__, "message": str(error), "details": error.details},
        )
    except Exception as error:
        payload = _preflight_payload(
            started,
            result="INVALID",
            status="invalid_harness",
            classification="INVALID_4L_LINEAR_CONTINUATION_PREFLIGHT",
            error={"type": type(error).__name__, "message": str(error)},
        )
    finally:
        ending = V10.V6.git_facts(root)
        payload.setdefault("provenance", {}).update(
            {
                "git_end": ending,
                "worktree_status_before": status_before,
                "worktree_status_after": V10.V6._status_lines(root),
                "branch_unchanged": started.get("branch") == ending.get("branch"),
                "head_unchanged": started.get("head") == ending.get("head"),
            }
        )
        if old_base is not None:
            _restore_base(old_base)
    report_ref = _save_json(root, PREFLIGHT_REPORT_PATH, payload, "continuation preflight report")
    lineage_ref = _save_json(
        root,
        PREFLIGHT_LINEAGE_PATH,
        {
            "schema": "lightgen_aligned_4l_linear_optimization_ceiling_v1_preflight_lineage.v1",
            "version": 1,
            "task_id": TASK_ID,
            "result": payload["result"],
            "status": payload["status"],
            "classification": payload["classification"],
            "report": report_ref,
            "contract": payload.get("contract"),
            "source_authority": payload.get("source_authority"),
            "claim_boundary": payload["claim_boundary"],
        },
        "continuation preflight lineage",
    )
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": lineage_ref, "contract": payload.get("contract")}


def _load_source_record(root: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    return _source_endpoint(source, _source_raw_power(root, source))


def _existing_block_refs(root: Path, output_dir: Path) -> list[dict[str, Any]]:
    refs = []
    for path in sorted((output_dir / "training_blocks").glob("*.json")):
        payload = _read_json(path, "continuation training block")
        _require(payload.get("schema") == BLOCK_SCHEMA and payload.get("candidate") == "C_4L_LINEAR_CONTINUATION", "continuation block identity changed")
        refs.append({"path": _relative(root, path), "sha256": _sha256(path), "current_step": payload.get("current_step"), "target_step": payload.get("target_step")})
    return refs


def _decision_c3000(parent: Mapping[str, Any], c3000: Mapping[str, Any]) -> dict[str, Any]:
    contrast = _contrast(parent, c3000)
    ci = contrast["paired_bootstrap"]
    conditions = {
        "delta_2k_3k_ge_0.010": contrast["point"] >= CONTINUATION_MIN_GAIN,
        "paired_ci_lower_gt_0": float(ci["ci_lower"]) > 0.0,
        "image_delta_le_0.005": contrast["image_delta"] <= RECONSTRUCTION_MAX_DELTA,
        "raw_validity_pass": _raw_validity_pass(c3000),
        "cyclic_transfer_pass": BASE._cyclic_pass(c3000["cyclic_public"]),
    }
    return {"contrast": contrast, "conditions": conditions, "continue_to_4000": all(conditions.values())}


def _build_aggregate(
    root: Path,
    preflight: Mapping[str, Any],
    source: Mapping[str, Any],
    parent: Mapping[str, Any],
    c3000: Mapping[str, Any],
    c4000: Mapping[str, Any] | None,
    *,
    contract_ref: Mapping[str, Any],
    started: Mapping[str, Any],
    status_before: list[str],
    status_after: list[str],
    block_refs: list[Mapping[str, Any]],
    training_seconds: float,
    peak_memory_bytes: int,
    pause_resume_count: int,
    device_facts: Mapping[str, Any],
) -> dict[str, Any]:
    decision = _decision_c3000(parent, c3000)
    trajectory: dict[str, Any] = {
        "R2000": float(parent["evaluation_public"]["smooth"]["macro_recall"]),
        "R3000": float(c3000["evaluation_public"]["smooth"]["macro_recall"]),
        "gap_3000_to_historical_2L_reference": float(c3000["evaluation_public"]["smooth"]["macro_recall"] - HISTORICAL_2L_REFERENCE),
        "Delta_2k_3k": decision["contrast"],
    }
    endpoint_records = {"2000": parent, "3000": c3000}
    comparisons: dict[str, Any] = {"Delta_2k_3k": decision["contrast"]}
    classification: str
    if c4000 is None:
        classification = "4L_LINEAR_CONTINUATION_PLATEAU_SUPPORTED"
        continue_to_4000 = False
    else:
        endpoint_records["4000"] = c4000
        contrast_3k_4k = _contrast(c3000, c4000)
        contrast_2k_4k = _contrast(parent, c4000)
        total_conditions = {
            "delta_2k_4k_ge_0.020": contrast_2k_4k["point"] >= CONTINUED_TOTAL_MIN_GAIN,
            "paired_ci_lower_gt_0": float(contrast_2k_4k["paired_bootstrap"]["ci_lower"]) > 0.0,
            "late_gain_3k_4k_ge_0.005": contrast_3k_4k["point"] >= CONTINUED_LATE_MIN_GAIN,
            "image_delta_2k_4k_le_0.005": contrast_2k_4k["image_delta"] <= RECONSTRUCTION_MAX_DELTA,
            "raw_validity_pass": _raw_validity_pass(c4000) and _raw_validity_pass(c3000),
            "cyclic_transfer_pass": BASE._cyclic_pass(c4000["cyclic_public"]) and BASE._cyclic_pass(c3000["cyclic_public"]),
        }
        classification = "4L_LINEAR_CONTINUED_OPTIMIZATION_SUPPORTED" if all(total_conditions.values()) else "4L_LINEAR_OPTIMIZATION_SATURATING"
        continue_to_4000 = True
        comparisons.update({"Delta_3k_4k": contrast_3k_4k, "Delta_2k_4k": contrast_2k_4k})
        trajectory.update(
            {
                "R4000": float(c4000["evaluation_public"]["smooth"]["macro_recall"]),
                "gap_4000_to_historical_2L_reference": float(c4000["evaluation_public"]["smooth"]["macro_recall"] - HISTORICAL_2L_REFERENCE),
            }
        )
        decision["c4000"] = {"conditions": total_conditions, "classification": classification}
    per_class = {}
    for step, record in endpoint_records.items():
        per_class[step] = _class_summary(record)
    per_class_deltas = {}
    for step in ("3000", "4000"):
        if step in endpoint_records:
            previous = parent if step == "3000" else c3000
            current = endpoint_records[step]
            per_class_deltas[f"Delta_{previous['step']}_{step}"] = _contrast(previous, current)["class_delta"]
    aggregate_payload = {
        "schema": "lightgen_aligned_4l_linear_optimization_ceiling_v1_aggregate.v1",
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "VALID_COMPLETED",
        "status": "valid_completed",
        "classification": classification,
        "outer_gate": OUTER_GATE,
        "contract": dict(contract_ref),
        "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"]},
        "source_authority": {
            "report": source["report_ref"],
            "lineage": source["lineage_ref"],
            "checkpoint": source["checkpoint_ref"],
            "sufficient_observations": source["sufficient_ref"],
            "raw_outputs": source["raw_ref"],
        },
        "candidates": {
            "C_4L_LINEAR_C2000_PARENT": {"R": trajectory["R2000"], "sufficient_observations": parent["sufficient_ref"], "checkpoint": parent["checkpoint_ref"]},
            "C_4L_LINEAR_C3000": {"R": trajectory["R3000"], "sufficient_observations": c3000["sufficient_ref"], "checkpoint": c3000["checkpoint_ref"], "diagnostics": c3000["diagnostic_ref"], "endpoint_meta": c3000["endpoint_meta_ref"]},
        },
        "continue_to_4000": bool(continue_to_4000),
        "decision": decision,
        "trajectory": trajectory,
        "comparisons": comparisons,
        "reference": {
            "historical_2L_linear_2000": HISTORICAL_2L_REFERENCE,
            "interpretation": "gap_to_reference is descriptive only; 4L and 2L horizons are not a matched architecture-superiority test",
        },
        "reconstruction": {
            "L_image_2000": float(parent["dev_image_loss"]),
            "L_image_3000": float(c3000["dev_image_loss"]),
            "Delta_L_image_2k_3k": float(c3000["dev_image_loss"] - parent["dev_image_loss"]),
            **(
                {
                    "L_image_4000": float(c4000["dev_image_loss"]),
                    "Delta_L_image_3k_4k": float(c4000["dev_image_loss"] - c3000["dev_image_loss"]),
                    "Delta_L_image_2k_4k": float(c4000["dev_image_loss"] - parent["dev_image_loss"]),
                }
                if c4000 is not None
                else {}
            ),
        },
        "per_class": {
            "endpoint_summaries": per_class,
            "deltas": per_class_deltas,
            "weak_class_review": {
                "persistent_near_zero_threshold_descriptive": 0.01,
                "persistent_near_zero_classes": [
                    digit
                    for digit in range(10)
                    if all(float(item["recall"]) <= 0.01 for record in per_class.values() for item in record["class_recall"] if item["class"] == digit)
                ],
            },
        },
        "raw_cmos_power": {str(step): record["raw_cmos_power"] for step, record in endpoint_records.items()},
        "validity": {
            str(step): {
                "raw_validity_pass": _raw_validity_pass(record) if step != 2000 else bool(record.get("raw_validity_pass")),
                "cyclic_transfer_pass": BASE._cyclic_pass(record["cyclic_public"]),
            }
            for step, record in endpoint_records.items()
        },
        "optimization": {
            "trainable": TRAINABLE_NAMES,
            "frozen": FROZEN_NAMES,
            "stage_identity": STAGE_IDENTITY,
            "generator_phase_change_relative_to_C2000": {
                str(step): record.get("generator_diagnostics", {})
                for step, record in endpoint_records.items()
                if step != 2000
            },
            "training_blocks": [dict(ref) for ref in block_refs],
            "late_trajectory_interpretation": "registered stopping rule is the decision consumer; gradient and phase-change traces are descriptive diagnostics",
        },
        "resource": {
            "device": dict(device_facts),
            "training_seconds_current_execution": training_seconds,
            "active_gpu_time_seconds": training_seconds,
            "seconds_per_update_observed": training_seconds / max(1, sum(int(ref.get("target_step", 0)) - int(ref.get("current_step", 0)) for ref in block_refs)),
            "peak_vram_bytes": peak_memory_bytes,
            "pause_resume_count": pause_resume_count,
        },
        "local": {
            "branch": started.get("branch"),
            "start_head": started.get("head"),
            "end_head": V10.V6.git_facts(root).get("head"),
            "worktree_before": status_before,
            "worktree_after": status_after,
            "commit": "NO",
        },
        "claim_boundary": {
            "4l_linear_only": True,
            "relu_training": False,
            "no_threshold_gain_slope_sweep": True,
            "no_architecture_change": True,
            "no_adapter_unfreeze": True,
            "physical_te_validation": False,
            "stage_b": False,
            "sampled_generation": False,
            "protected": False,
            "official_test": False,
            "sealed": False,
            "formal_promotion": False,
        },
        "governance": {
            "parent_C2000_immutable": True,
            "A_B_retrained": False,
            "D_relu_trained": False,
            "objective_changed": False,
            "optimizer_moments_reinitialized": False,
            "sampler_restarted": False,
            "dev_used_for_online_tuning": False,
            "maximum_local_step_respected": True,
        },
    }
    _require(not AGGREGATE_DIR.exists(), f"aggregate output already exists: {AGGREGATE_DIR}")
    AGGREGATE_DIR.mkdir(parents=True, exist_ok=False)
    report_ref = _save_json(root, AGGREGATE_REPORT_PATH, aggregate_payload, "continuation aggregate report")
    lineage_ref = _save_json(
        root,
        AGGREGATE_LINEAGE_PATH,
        {
            "schema": "lightgen_aligned_4l_linear_optimization_ceiling_v1_aggregate_lineage.v1",
            "version": 1,
            "task_id": TASK_ID,
            "result": aggregate_payload["result"],
            "status": aggregate_payload["status"],
            "classification": aggregate_payload["classification"],
            "report": report_ref,
            "contract": dict(contract_ref),
            "claim_boundary": aggregate_payload["claim_boundary"],
        },
        "continuation aggregate lineage",
    )
    return {"payload": aggregate_payload, "report": report_ref, "lineage": lineage_ref}


def run_scientific(root: Path, *, resume_step: int | None = None) -> dict[str, Any]:
    preflight = _load_contract(root)
    source = _load_parent_authority(root)
    started = V10.V6.git_facts(root)
    status_before = V10.V6._status_lines(root)
    _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "scientific continuation Git identity changed")
    BASE.V10.V6._prepare_cuda_runtime_dll_search_path()
    device, device_facts = BASE.V10.V6._cuda_facts()
    _require(device.type == "cuda", "CUDA is required for scientific continuation")
    authority = BASE._authorities(root)
    bundle = authority["bundle"]
    bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
    images, labels, train_indices, dev_indices, dataset = BASE.V10.V6._load_roles(root)
    _require(dataset.get("protected", 0) in (0, [], None) and dataset.get("sealed", 0) in (0, [], None), "scientific continuation protected or sealed role is nonempty")
    _require(torch.equal(labels.index_select(0, dev_indices).cpu(), source["sufficient"]["dev_labels"].to(torch.long).cpu()), "scientific continuation DEV identity changed")
    parent = _load_source_record(root, source)
    output_dir = SCIENTIFIC_DIR
    if resume_step is None:
        _require(not output_dir.exists(), f"scientific continuation output already exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
        current_step = PARENT_STEP
        pause_resume_count = 0
    else:
        _require(isinstance(resume_step, int) and PARENT_STEP <= resume_step <= MAX_LOCAL_STEP and resume_step % BLOCK_SIZE == 0, "explicit resume step is not a valid continuation checkpoint boundary")
        _require(output_dir.is_dir(), f"scientific continuation output directory is unavailable for resume: {output_dir}")
        current_step = resume_step
        pause_resume_count = 1
        if current_step > PARENT_STEP:
            _load_continuation_checkpoint(root, output_dir, current_step)
    old_base = _patch_base_for_continuation()
    block_results: list[dict[str, Any]] = []
    try:
        _save_progress(root, output_dir, phase="scientific_start", status="started", step=current_step, details={"resume_step": resume_step, "target_policy": "C3000 then registered stopping rule"})
        if current_step < FIRST_TARGET_STEP:
            while current_step < FIRST_TARGET_STEP:
                target = min(FIRST_TARGET_STEP, current_step + BLOCK_SIZE)
                block = _run_training_block(
                    root,
                    bundle,
                    source,
                    preflight["contract_ref"],
                    images,
                    labels,
                    train_indices,
                    device,
                    current_step=current_step,
                    target_step=target,
                    output_dir=output_dir,
                )
                block_results.append(block)
                current_step = target
                print(json.dumps({"event": "4l_linear_continuation_progress", "step": current_step, "target": FIRST_TARGET_STEP, "seconds_per_update": block["seconds_per_update"], "training_seconds_current_execution": sum(item["training_seconds"] for item in block_results)}, ensure_ascii=False), flush=True)
        if not (_sufficient_path(output_dir, FIRST_TARGET_STEP).is_file() and _endpoint_meta_path(output_dir, FIRST_TARGET_STEP).is_file()):
            _run_endpoint_from_checkpoint(root, bundle, source, preflight["contract_ref"], images, labels, train_indices, dev_indices, device, step=FIRST_TARGET_STEP, output_dir=output_dir)
        c3000 = _load_endpoint(root, output_dir, FIRST_TARGET_STEP)
        decision = _decision_c3000(parent, c3000)
        print(json.dumps({"event": "4l_linear_continuation_gate", "step": FIRST_TARGET_STEP, "delta_2k_3k": decision["contrast"]["point"], "ci_lower": decision["contrast"]["paired_bootstrap"]["ci_lower"], "image_delta": decision["contrast"]["image_delta"], "continue_to_4000": decision["continue_to_4000"]}, ensure_ascii=False), flush=True)
        c4000 = None
        if decision["continue_to_4000"]:
            if current_step < MAX_LOCAL_STEP:
                current_step = max(current_step, FIRST_TARGET_STEP)
                while current_step < MAX_LOCAL_STEP:
                    target = min(MAX_LOCAL_STEP, current_step + BLOCK_SIZE)
                    block = _run_training_block(
                        root,
                        bundle,
                        source,
                        preflight["contract_ref"],
                        images,
                        labels,
                        train_indices,
                        device,
                        current_step=current_step,
                        target_step=target,
                        output_dir=output_dir,
                    )
                    block_results.append(block)
                    current_step = target
                    print(json.dumps({"event": "4l_linear_continuation_progress", "step": current_step, "target": MAX_LOCAL_STEP, "seconds_per_update": block["seconds_per_update"], "training_seconds_current_execution": sum(item["training_seconds"] for item in block_results)}, ensure_ascii=False), flush=True)
            if not (_sufficient_path(output_dir, MAX_LOCAL_STEP).is_file() and _endpoint_meta_path(output_dir, MAX_LOCAL_STEP).is_file()):
                _run_endpoint_from_checkpoint(root, bundle, source, preflight["contract_ref"], images, labels, train_indices, dev_indices, device, step=MAX_LOCAL_STEP, output_dir=output_dir)
            c4000 = _load_endpoint(root, output_dir, MAX_LOCAL_STEP)
        status_after = V10.V6._status_lines(root)
        aggregate = _build_aggregate(
            root,
            preflight,
            source,
            parent,
            c3000,
            c4000,
            contract_ref=preflight["contract_ref"],
            started=started,
            status_before=status_before,
            status_after=status_after,
            block_refs=_existing_block_refs(root, output_dir),
            training_seconds=float(sum(item["training_seconds"] for item in block_results)),
            peak_memory_bytes=max([int(item["peak_memory_bytes"]) for item in block_results] + [0]),
            pause_resume_count=pause_resume_count,
            device_facts=device_facts,
        )
        _save_progress(root, output_dir, phase="scientific_complete", status="completed", step=MAX_LOCAL_STEP if c4000 is not None else FIRST_TARGET_STEP, details={"aggregate": aggregate["report"], "continue_to_4000": c4000 is not None})
        return aggregate
    finally:
        _restore_base(old_base)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--resume-step", type=int, default=None, help="explicit last continuation checkpoint step; never resolves latest automatically")
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    if args.action == "preflight":
        _require(args.resume_step is None, "preflight does not accept --resume-step")
        result = run_preflight(root)
    else:
        result = run_scientific(root, resume_step=args.resume_step)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_plain))
    return 0 if result.get("result") in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
