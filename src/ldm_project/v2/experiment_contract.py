"""算力感知的通用实验合同与低成本训练门。

该模块只负责跨进程、跨阶段的合同和状态校验；它不创建模型、不启动训练，
因此可在 optimizer 创建前被所有 v2 训练入口复用。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import torch

from .evidence import atomic_write_json, canonical_sha256, file_sha256


ExperimentLevel = Literal["L0", "L1", "L2", "L3"]
ExperimentState = Literal[
    "draft",
    "validated",
    "preflight_passed",
    "running",
    "checkpointed",
    "measured",
    "passed",
    "valid_failed",
    "invalid_harness",
    "resource_aborted",
]

_TRANSITIONS: dict[str, frozenset[str]] = {
    "draft": frozenset({"validated", "invalid_harness"}),
    "validated": frozenset({"preflight_passed", "invalid_harness"}),
    "preflight_passed": frozenset({"running", "invalid_harness"}),
    "running": frozenset({"checkpointed", "measured", "resource_aborted", "invalid_harness"}),
    "checkpointed": frozenset({"running", "measured", "resource_aborted", "invalid_harness"}),
    "measured": frozenset({"passed", "valid_failed", "invalid_harness"}),
    "passed": frozenset(),
    "valid_failed": frozenset(),
    "invalid_harness": frozenset(),
    "resource_aborted": frozenset({"running", "invalid_harness"}),
}


@dataclass(frozen=True)
class ExperimentContract:
    schema_version: int
    name: str
    level: ExperimentLevel
    hypothesis: str
    comparison: Mapping[str, Any]
    budget: Mapping[str, Any]
    inputs: Mapping[str, str]
    gates: Mapping[str, Any]
    stop_policy: Mapping[str, Any]
    runtime: Mapping[str, Any]
    runtime_closure_hash: str
    contract_hash: str | None = field(default=None, compare=False)

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "level": self.level,
            "hypothesis": self.hypothesis,
            "comparison": dict(self.comparison),
            "budget": dict(self.budget),
            "inputs": dict(self.inputs),
            "gates": dict(self.gates),
            "stop_policy": dict(self.stop_policy),
            "runtime": dict(self.runtime),
            "runtime_closure_hash": self.runtime_closure_hash,
        }

    @property
    def digest(self) -> str:
        return canonical_sha256(self.payload)

    @property
    def preflight_budget_seconds(self) -> float:
        configured = self.budget.get("preflight_seconds")
        if not isinstance(configured, (int, float)) or not math.isfinite(float(configured)) or float(configured) <= 0:
            raise ValueError("experiment contract requires a positive preflight_seconds budget")
        level_cap = {"L0": 0.0, "L1": 120.0, "L2": 300.0, "L3": 600.0}[self.level]
        if level_cap == 0.0:
            return 0.0
        planned_seconds = self.budget.get("planned_seconds")
        if isinstance(planned_seconds, (int, float)) and math.isfinite(float(planned_seconds)) and float(planned_seconds) > 0:
            fraction = {"L1": 1.0, "L2": 0.05, "L3": 0.10}[self.level]
            return min(float(configured), level_cap, float(planned_seconds) * fraction)
        return min(float(configured), level_cap)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported experiment contract schema")
        if self.level not in {"L0", "L1", "L2", "L3"}:
            raise ValueError("experiment contract level must be L0-L3")
        for name in ("hypothesis", "name"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"experiment contract {name} is required")
        if not isinstance(self.inputs, Mapping) or not self.inputs:
            raise ValueError("experiment contract requires immutable input hashes")
        if any(not isinstance(key, str) or not isinstance(value, str) or not value for key, value in self.inputs.items()):
            raise ValueError("experiment contract input hashes must be non-empty strings")
        if not isinstance(self.runtime_closure_hash, str) or self.runtime_closure_hash != canonical_sha256(self.runtime):
            raise ValueError("experiment runtime closure hash mismatch")
        if self.contract_hash is not None and self.contract_hash != self.digest:
            raise ValueError("experiment contract hash mismatch")
        if not isinstance(self.stop_policy, Mapping):
            raise ValueError("experiment stop policy is required")
        for state in ("invalid_harness", "valid_failed", "passed"):
            if state not in self.stop_policy:
                raise ValueError(f"experiment stop policy lacks {state}")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return self.payload | {"contract_hash": self.digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentContract":
        contract = cls(
            schema_version=int(value["schema_version"]),
            name=str(value["name"]),
            level=value["level"],
            hypothesis=str(value["hypothesis"]),
            comparison=dict(value["comparison"]),
            budget=dict(value["budget"]),
            inputs=dict(value["inputs"]),
            gates=dict(value["gates"]),
            stop_policy=dict(value["stop_policy"]),
            runtime=dict(value["runtime"]),
            runtime_closure_hash=str(value["runtime_closure_hash"]),
            contract_hash=value.get("contract_hash"),
        )
        contract.validate()
        return contract


def load_experiment_contract(path: str | Path) -> ExperimentContract:
    target = Path(path)
    if not target.is_absolute() or not target.is_file():
        raise ValueError("experiment contract must be an existing absolute path")
    return ExperimentContract.from_dict(json.loads(target.read_text(encoding="utf-8")))


def write_experiment_contract(path: str | Path, contract: ExperimentContract) -> None:
    atomic_write_json(Path(path), contract.to_dict())


def runtime_closure_hash(runtime: Mapping[str, Any]) -> str:
    return canonical_sha256(runtime)


def validate_contract_inputs(contract: ExperimentContract, expected: Mapping[str, str]) -> None:
    contract.validate()
    for name, value in expected.items():
        if contract.inputs.get(name) != value:
            raise ValueError(f"experiment contract input mismatch for {name}")


def validate_preflight_report(contract: ExperimentContract, report: Mapping[str, Any], *, elapsed_seconds: float) -> dict[str, Any]:
    """Validate a runtime-produced L1/L2 report before authorizing training."""
    contract.validate()
    if elapsed_seconds < 0 or elapsed_seconds > contract.preflight_budget_seconds:
        raise ValueError("experiment preflight exceeded its declared compute budget")
    required = ("finite", "all_parameters_updated", "evaluator_gradients_clean")
    if any(report.get(name) is not True for name in required):
        raise ValueError("experiment preflight did not pass finite/update/evaluator gates")
    if contract.level in {"L2", "L3"}:
        if report.get("independent_match") is not True or report.get("resume_match") is not True:
            raise ValueError("L2/L3 experiment preflight requires independent and resume matches")
    return {
        "state": "preflight_passed",
        "level": contract.level,
        "elapsed_seconds": float(elapsed_seconds),
        "contract_hash": contract.digest,
    }


def assert_transition(previous: ExperimentState, new: ExperimentState) -> None:
    if new not in _TRANSITIONS.get(previous, frozenset()):
        raise ValueError(f"invalid experiment state transition {previous} -> {new}")


def checkpoint_manifest(path: str | Path, *, contract: ExperimentContract, completed_step: int, candidate: str, seed: int, next_action: str | None = None) -> dict[str, Any]:
    target = Path(path)
    if not target.is_absolute() or not target.is_file():
        raise ValueError("checkpoint manifest requires an existing absolute checkpoint")
    if completed_step < 0 or seed < 0 or not candidate:
        raise ValueError("checkpoint identity is invalid")
    return {
        "schema_version": 1,
        "contract_hash": contract.digest,
        "checkpoint_sha256": file_sha256(target),
        "candidate": candidate,
        "seed": seed,
        "completed_step": completed_step,
        "formal_promotion_authorized": False,
        "permitted_next_actions": [] if next_action is None else [next_action],
    }


def validate_checkpoint_manifest(path: str | Path, manifest: Mapping[str, Any], *, contract: ExperimentContract, candidate: str, seed: int) -> int:
    target = Path(path)
    if not target.is_absolute() or not target.is_file():
        raise ValueError("checkpoint must be an existing absolute path")
    if manifest.get("schema_version") != 1 or manifest.get("contract_hash") != contract.digest or manifest.get("candidate") != candidate or manifest.get("seed") != seed:
        raise ValueError("checkpoint manifest context mismatch")
    if manifest.get("checkpoint_sha256") != file_sha256(target) or manifest.get("formal_promotion_authorized") is not False or manifest.get("permitted_next_actions") != []:
        raise ValueError("checkpoint manifest is tampered or has promotion authority")
    completed_step = manifest.get("completed_step")
    if not isinstance(completed_step, int) or completed_step < 0:
        raise ValueError("checkpoint completed_step is invalid")
    return completed_step


def checkpoint_interval_steps(total_steps: int, *, steps_per_second: float | None = None, max_minutes: float = 10.0) -> int:
    """Return the default bounded checkpoint cadence without forcing I/O frequency."""
    if total_steps < 1 or max_minutes <= 0:
        raise ValueError("checkpoint cadence inputs are invalid")
    step_bound = max(1, math.ceil(total_steps * 0.05))
    if steps_per_second is None:
        return step_bound
    if not math.isfinite(steps_per_second) or steps_per_second <= 0:
        raise ValueError("steps_per_second must be positive and finite")
    time_bound = max(1, math.floor(steps_per_second * max_minutes * 60.0))
    return min(step_bound, time_bound)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if "python" not in state or "torch_cpu" not in state:
        raise ValueError("checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_state = state.get("torch_cuda")
        if cuda_state is None:
            raise ValueError("checkpoint is missing CUDA RNG state")
        torch.cuda.set_rng_state_all(cuda_state)


def validate_history_steps(history: list[Mapping[str, Any]], completed_step: int) -> None:
    if completed_step < 0 or len(history) != completed_step:
        raise ValueError("checkpoint history length does not match completed step")
    actual = [row.get("step") for row in history]
    if actual != list(range(1, completed_step + 1)):
        raise ValueError("checkpoint history must cover each step exactly once")
