"""Execute the one-shot N5 true optical-depth comparison.

Candidate A continues the original N4 C1 endpoint.  Candidate B starts from
the identical endpoint and factorizes the second generator propagation
interval as ``P_(d/2) M_(E psi) P_(d/2)``.  The canonical v2 runtime is not
modified; the separated midpoint exists only in this bounded research
consumer.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import time
import traceback
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


N4 = _load_module(
    Path(__file__).with_name(
        "execute_lightgen_2d_ols_interface_synthesis_model_family_comparison_v1.py"
    ),
    "lightgen_n4_execution_authority_n5",
)
N4ATTR = _load_module(
    Path(__file__).with_name(
        "execute_lightgen_2d_n4_endpoint_attribution_evaluation_v1.py"
    ),
    "lightgen_n4_endpoint_attribution_authority_n5",
)
N5DEF = _load_module(
    Path(__file__).with_name(
        "define_lightgen_2d_true_optical_depth_comparison_contract_v1.py"
    ),
    "lightgen_n5_definition_authority",
)
V10 = N4.V10
V11 = N4.V11


TASK_ID = "EXECUTE_N5_TRUE_OPTICAL_DEPTH_COMPARISON_V1"
DESIGN_AUTHORITY = "DEFINE_N5_TRUE_OPTICAL_DEPTH_COMPARISON_CONTRACT_V1"
DESIGN_OUTER_GATE = "N5_TRUE_OPTICAL_DEPTH_COMPARISON_READY"
OUTER_GATE = "N5_TRUE_OPTICAL_DEPTH_COMPARISON_RESOLVED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

BASE_OUTPUT = Path("outputs/20260914_n5/runs")
CONTRACT_DIR = BASE_OUTPUT / "lightgen_2d_true_optical_depth_comparison_contract_v1_attempt_01"
CONTRACT_PATH = CONTRACT_DIR / "lightgen_2d_true_optical_depth_comparison_contract.v1.json"
CONTRACT_SHA256 = "8d305b59216ec5ccccc6af50d83dce5d30d40c231bca1e50da104ebb0406cb6d"
PREFLIGHT_DIR = BASE_OUTPUT / "lightgen_2d_true_optical_depth_comparison_preflight_v1_attempt_05"
CANDIDATE_A_DIR = BASE_OUTPUT / "lightgen_2d_true_optical_depth_comparison_candidate_a_attempt_01"
CANDIDATE_B_DIR = BASE_OUTPUT / "lightgen_2d_true_optical_depth_comparison_candidate_b_attempt_01"
AGGREGATE_DIR = BASE_OUTPUT / "lightgen_2d_true_optical_depth_comparison_aggregate_v1_attempt_01"

C1_CHECKPOINT_PATH = Path(
    "outputs/20260913_n4/runs/"
    "lightgen_2d_ols_interface_synthesis_model_family_c1_attempt_01/"
    "lightgen_2d_n4_c1_step_300_checkpoint.v1.pt"
)
C1_CHECKPOINT_SHA256 = "d7a5581df3fb460337e8655865a200886739cdeab0fb6074213cc4b2cc66c6e9"
N4_ATTRIBUTION_REPORT_PATH = Path(
    "outputs/20260914_n4_endpoint_attribution/runs/"
    "lightgen_2d_n4_endpoint_attribution_evaluation_v1_attempt_01/"
    "lightgen_2d_n4_endpoint_attribution_report.v1.json"
)
N4_ATTRIBUTION_REPORT_SHA256 = "18cef70e3ad1a2a7f81144663f6e55373d08265c4d387d2e39df2aec0c4a3388"
N4_ATTRIBUTION_SUFFICIENT_SHA256 = "59867343f90528951fcc90f6936f5a9cd2630f86aa6c9c6a167ad22a4c4c592a"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
ADAPTER_RANK = 9
LOGICAL_BATCH_SIZE = 10
PRODUCTION_GRID = 2048
PRECHECK_STEPS = 3
SCIENTIFIC_STEPS = 300
CONTINUATION_OFFSET = 300
CONTINUATION_TOTAL_STEPS = CONTINUATION_OFFSET + SCIENTIFIC_STEPS
CHECKPOINT_STEPS = (0, 100, 200, 300)
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
STAGE_A_FIREWALL = 0.937
MATERIAL_DELTA = 0.05
RECONSTRUCTION_MAX_DELTA = 0.005
SELECTION_ROWS = 1000
SELECTION_PER_CLASS = 100
SELECTION_SEED = 4321
SPLIT_SEED = 1234
S_TAU = 0.02919053753794475
LAMBDA_TRANSFER = 28.48464149920509
LAMBDA_IMAGE = 13.353282014834436
TOLERANCE = {"rtol": 1.0e-5, "atol": 1.0e-6}

ADAPTER_PARAMETER_NAMES = [
    "adapter.u_real",
    "adapter.u_imag",
    "adapter.v_real",
    "adapter.v_imag",
]
A_PARAMETER_NAMES = ADAPTER_PARAMETER_NAMES + ["generator.phase"]
B_PARAMETER_NAMES = ADAPTER_PARAMETER_NAMES + ["generator.phase", "generator.depth_phase"]

PREFLIGHT_REPORT_NAME = "lightgen_2d_true_optical_depth_comparison_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "lightgen_2d_true_optical_depth_comparison_preflight_lineage.v1.json"
PREFLIGHT_INPUT_NAME = "lightgen_2d_true_optical_depth_comparison_preflight_input.v1.json"
SCIENTIFIC_INPUT_NAME = "lightgen_2d_true_optical_depth_comparison_{candidate}_scientific_input.v1.json"
SCIENTIFIC_REPORT_NAME = "lightgen_2d_true_optical_depth_comparison_{candidate}_scientific_report.v1.json"
SCIENTIFIC_LINEAGE_NAME = "lightgen_2d_true_optical_depth_comparison_{candidate}_scientific_lineage.v1.json"
SUFFICIENT_NAME = "lightgen_2d_true_optical_depth_comparison_{candidate}_step_{step}_sufficient_observations.v1.pt"
RAW_NAME = "lightgen_2d_true_optical_depth_comparison_{candidate}_step_300_cyclic_raw_outputs.v1.pt"
CHECKPOINT_NAME = "lightgen_2d_true_optical_depth_comparison_{candidate}_step_{step}_checkpoint.v1.pt"
PREFLIGHT_CHECKPOINT_NAME = "lightgen_2d_true_optical_depth_comparison_{candidate}_preflight_checkpoint.v1.pt"
AGGREGATE_REPORT_NAME = "lightgen_2d_true_optical_depth_comparison_aggregate_report.v1.json"
AGGREGATE_LINEAGE_NAME = "lightgen_2d_true_optical_depth_comparison_aggregate_lineage.v1.json"


class InvalidN5(RuntimeError):
    """The N5 input or execution harness is invalid."""


class HardBlockedN5(RuntimeError):
    """A valid N5 boundary prevents continuation."""

    def __init__(self, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidN5(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, value: str | Path) -> Path:
    relative = Path(value)
    _require(not relative.is_absolute(), f"absolute path is not allowed: {relative}")
    return (root / relative).resolve()


def _relative(root: Path, value: Path) -> str:
    return value.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidN5(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _sha256(path: Path) -> str:
    return N4._file_sha256(path)


def _validate_sha(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = _sha256(path)
    _require(observed == expected, f"{name} SHA mismatch: expected {expected}, observed {observed}")


def _cpu_tree(value: Any) -> Any:
    return N4._cpu_tree(value)


def _tree_equal(left: Any, right: Any) -> bool:
    return N4._tree_equal(left, right)


def _digest_tree(value: Any) -> str:
    return N4._digest_tree(value)


def _tensor_finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _public(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _public(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (list, tuple)):
        return [_public(item) for item in value]
    if torch.is_tensor(value):
        raise InvalidN5("tensor leaked into JSON report")
    return value


def _status_lines(root: Path) -> list[str]:
    return V10.V6._status_lines(root)


def _validate_contract(root: Path) -> dict[str, Any]:
    _validate_sha(_path(root, CONTRACT_PATH), CONTRACT_SHA256, "N5 design contract")
    contract = _read_json(_path(root, CONTRACT_PATH), "N5 design contract")
    _require(
        contract.get("schema") == "lightgen_2d_true_optical_depth_comparison_contract.v1"
        and contract.get("result") == "READY"
        and contract.get("status") == "definition_ready"
        and contract.get("classification") == DESIGN_OUTER_GATE
        and contract.get("outer_gate") == DESIGN_OUTER_GATE,
        "N5 design contract is not the frozen READY authority",
    )
    _require(contract.get("task_id") == "DEFINE_LIGHTGEN_2D_TRUE_OPTICAL_DEPTH_COMPARISON_CONTRACT_V1", "N5 design task identity changed")
    definition = _mapping(contract.get("definition_entrypoint"), "N5 definition entrypoint")
    definition_path = _path(root, str(definition.get("path")))
    _validate_sha(definition_path, str(definition.get("sha256")), "N5 definition entrypoint")
    _require(str(definition.get("sha256")) == "437ff34819d10691b89c06d6853e3e396847fa29f07e5ed04b7e37237a1b61c5", "N5 definition entrypoint authority changed")
    common = _mapping(contract.get("common_execution"), "N5 common execution")
    _require(common.get("additional_updates") == SCIENTIFIC_STEPS and common.get("batch") == LOGICAL_BATCH_SIZE, "N5 update budget changed")
    _require(common.get("evaluation_steps_relative_to_continuation") == list(CHECKPOINT_STEPS), "N5 evaluation cadence changed")
    _require(common.get("sampler") == "fresh deterministic continuation view at global offset 300; no exhausted 300-step sampler reuse", "N5 sampler boundary changed")
    objective = _mapping(contract.get("common_objective"), "N5 common objective")
    _require(float(objective.get("lambda_transfer")) == LAMBDA_TRANSFER and float(objective.get("lambda_image")) == LAMBDA_IMAGE and objective.get("tuning") is False, "N5 objective changed")
    _require(contract.get("stage_a", {}).get("firewall") == "R300>=0.937", "N5 Stage-A firewall changed")
    _require(contract.get("claim_boundary", {}).get("protected") is False and contract.get("claim_boundary", {}).get("sealed") is False, "N5 data boundary changed")
    return contract


class ContinuationSampler:
    """Exact seed-1234 schedule with a fresh post-C1 global offset."""

    SCHEMA = "lightgen_n5_continuation_sampler.v1"

    def __init__(self, train_indices: torch.Tensor, labels: torch.Tensor) -> None:
        _require(train_indices.dtype == torch.long and labels.dtype == torch.long, "N5 sampler roles must be long tensors")
        _require(train_indices.ndim == 1 and labels.ndim == 1 and train_indices.numel() == 55_000 and labels.numel() == 60_000, "N5 sampler role sizes changed")
        self.seed = SPLIT_SEED
        self.global_offset = CONTINUATION_OFFSET
        self.steps = CONTINUATION_TOTAL_STEPS
        self._cursor = CONTINUATION_OFFSET
        generator = torch.Generator(device="cpu").manual_seed(SPLIT_SEED)
        self._permutations: list[torch.Tensor] = []
        role_labels = labels.index_select(0, train_indices).cpu()
        for digit in range(CLASS_COUNT):
            candidates = train_indices[role_labels == digit].cpu()
            _require(candidates.numel() >= self.steps, f"N5 TRAIN role cannot support continuation budget for class {digit}")
            self._permutations.append(candidates.index_select(0, torch.randperm(candidates.numel(), generator=generator)))

    @property
    def cursor(self) -> int:
        return self._cursor

    def batch_at(self, global_step: int) -> tuple[torch.Tensor, torch.Tensor]:
        _require(self.global_offset <= global_step < self.steps, "N5 continuation batch is outside the frozen continuation view")
        return torch.stack([permutation[global_step] for permutation in self._permutations]), torch.arange(CLASS_COUNT, dtype=torch.long)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cursor >= self.steps:
            raise StopIteration("N5 continuation sampler is exhausted")
        result = self.batch_at(self._cursor)
        self._cursor += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "seed": self.seed,
            "global_offset": self.global_offset,
            "steps": self.steps,
            "cursor": self._cursor,
            "class_count": CLASS_COUNT,
            "rows_per_class": [int(value.numel()) for value in self._permutations],
            "permutations": [value.clone() for value in self._permutations],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        _require(state.get("schema") == self.SCHEMA and state.get("seed") == self.seed, "N5 sampler schema/seed changed")
        _require(state.get("global_offset") == self.global_offset and state.get("steps") == self.steps, "N5 sampler continuation identity changed")
        cursor = state.get("cursor")
        _require(isinstance(cursor, int) and self.global_offset <= cursor <= self.steps, "N5 sampler cursor is invalid")
        permutations = state.get("permutations")
        _require(isinstance(permutations, list) and len(permutations) == CLASS_COUNT, "N5 sampler permutations are invalid")
        for index, value in enumerate(permutations):
            _require(isinstance(value, torch.Tensor) and value.dtype == torch.long and value.shape == self._permutations[index].shape, "N5 sampler permutation shape changed")
        self._permutations = [value.cpu().clone() for value in permutations]
        self._cursor = cursor


class TrueDepthGenerator(nn.Module):
    """Existing two phase screens with a separated second-segment midpoint."""

    def __init__(self, phase: torch.Tensor, device: torch.device) -> None:
        super().__init__()
        _require(tuple(phase.shape) == (2, 1, 100, 100), "N5 existing generator phase shape changed")
        self.phase = nn.Parameter(phase.detach().to(device=device, dtype=torch.float32).clone())
        self.depth_phase = nn.Parameter(torch.zeros((1, 1, 100, 100), device=device, dtype=torch.float32))
        self.propagation_shape = (PRODUCTION_GRID, PRODUCTION_GRID)
        self.spacing_m = float(N5DEF.PROPAGATION_DISTANCE_M)

    def forward(self, field: Any, prefix: str) -> tuple[Any, tuple[Any, ...], dict[str, torch.Tensor]]:
        _require(tuple(field.plane.shape) == tuple(N5DEF.ACTIVE_SHAPE), "N5 generator entrance shape changed")
        first_output, first_propagated = _direct_layer_with_trace(field, self.phase[0], f"{prefix}_layer_1")
        second_output, split_records = N5DEF._split_layer(first_output, self.phase[1], self.depth_phase, f"{prefix}_layer_2")
        first_active = first_output
        second_half = split_records["second_half"]
        first_retention = first_active.tensor.abs().square().sum(dim=(-3, -2, -1)) / first_propagated.tensor.abs().square().sum(dim=(-3, -2, -1)).clamp_min(1.0e-12)
        second_retention = second_output.tensor.abs().square().sum(dim=(-3, -2, -1)) / second_half.tensor.abs().square().sum(dim=(-3, -2, -1)).clamp_min(1.0e-12)
        return (
            second_output,
            (first_active, split_records["first_half"], split_records["midpoint"], second_half, second_output),
            {
                "crop_retention": torch.stack((first_retention, second_retention), dim=1),
                "phase": self.phase,
                "depth_phase": self.depth_phase,
            },
        )


def _direct_layer_with_trace(field: Any, phase: torch.Tensor, prefix: str) -> tuple[Any, Any]:
    modulated = N5DEF._phase_active(field, phase, f"{prefix}_phase")
    padded = N5DEF._pad_active(modulated, prefix)
    propagated = N5DEF.angular_spectrum_propagate(padded, N5DEF.PROPAGATION_DISTANCE_M, f"{prefix}_propagated")
    output = N5DEF._crop_active(propagated, f"{prefix}_active", field.plane.z_m + N5DEF.PROPAGATION_DISTANCE_M)
    return output, propagated


def _adapter_named(adapter: Any) -> dict[str, nn.Parameter]:
    return {f"adapter.{name}": parameter for name, parameter in adapter.named_parameters()}


def _runtime_trainable_names(candidate: str) -> list[str]:
    if candidate == "A_CONTINUED_C1":
        return ["generator.phase"]
    if candidate == "B_TRUE_DEPTH_C1":
        return ["generator.phase", "generator.depth_phase"]
    raise InvalidN5(f"unknown N5 candidate: {candidate}")


def _candidate_parameter_names(candidate: str) -> list[str]:
    return ADAPTER_PARAMETER_NAMES + _runtime_trainable_names(candidate)


def _candidate_named(consumer: Any, adapter: Any, candidate: str) -> dict[str, nn.Parameter]:
    runtime_named = dict(consumer.runtime.named_parameters())
    runtime_names = _runtime_trainable_names(candidate)
    _require(all(name in runtime_named for name in runtime_names), f"{candidate} generator parameter surface changed")
    named = _adapter_named(adapter)
    named.update({name: runtime_named[name] for name in runtime_names})
    expected = _candidate_parameter_names(candidate)
    _require(list(named) == expected, f"{candidate} trainable parameter order changed")
    return named


def _configure_candidate(consumer: Any, adapter: Any, candidate: str, *, trainable: bool) -> dict[str, Any]:
    V10.V8._configure_all_frozen(consumer)
    named = _candidate_named(consumer, adapter, candidate)
    runtime_trainable = set(_runtime_trainable_names(candidate))
    for parameter in named.values():
        parameter.requires_grad_(trainable)
        parameter.grad = None
    runtime_named = dict(consumer.runtime.named_parameters())
    _require(all(parameter.requires_grad is trainable for parameter in named.values()), f"{candidate} trainable firewall failed")
    _require(all(not parameter.requires_grad for name, parameter in runtime_named.items() if name not in runtime_trainable), f"{candidate} frozen runtime firewall failed")
    if trainable:
        consumer.runtime.train()
    else:
        consumer.runtime.eval()
    return {
        "candidate": candidate,
        "trainable": list(named) if trainable else [],
        "trainable_count": int(sum(parameter.numel() for parameter in named.values())) if trainable else 0,
        "parameter_counts": {name: int(parameter.numel()) for name, parameter in named.items()},
        "frozen_runtime": [name for name in runtime_named if name not in runtime_trainable],
        "runtime_parameter_names": list(runtime_named),
        "runtime_parameters_all_frozen_except_declared": True,
        "encoder_updated": False,
        "ols_updated": False,
        "smooth_updated": False,
    }


def _new_candidate(bundle: Mapping[str, Any], c1: Mapping[str, Any], candidate: str, device: torch.device) -> tuple[Any, Any]:
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V10.V8._load_runtime_state(consumer, dict(c1["runtime_state"]))
    adapter = V11.LowRankAdapter(device=device)
    adapter.load_state_dict(dict(c1["adapter_state"]), strict=True)
    if candidate == "B_TRUE_DEPTH_C1":
        base_phase = consumer.runtime.generator.phase.detach().clone()
        consumer.runtime.generator = TrueDepthGenerator(base_phase, device)
    _configure_candidate(consumer, adapter, candidate, trainable=True)
    return consumer, adapter


def _new_frozen_candidate(bundle: Mapping[str, Any], c1: Mapping[str, Any], candidate: str, device: torch.device) -> tuple[Any, Any]:
    consumer, adapter = _new_candidate(bundle, c1, candidate, device)
    _configure_candidate(consumer, adapter, candidate, trainable=False)
    return consumer, adapter


def _frozen_runtime_digest(consumer: Any, candidate: str) -> str:
    trainable = set(_runtime_trainable_names(candidate))
    state = {key: value for key, value in consumer.runtime.state_dict().items() if key not in trainable}
    return _digest_tree(_cpu_tree(state))


def _initial_shared_digests(consumer_a: Any, adapter_a: Any, consumer_b: Any, adapter_b: Any) -> dict[str, Any]:
    a_state = dict(consumer_a.runtime.state_dict())
    b_state = dict(consumer_b.runtime.state_dict())
    b_shared = {key: value for key, value in b_state.items() if key != "generator.depth_phase"}
    _require(set(a_state) == set(b_shared), "N5 A/B runtime state surfaces are not symmetric")
    _require(all(torch.equal(a_state[key], b_shared[key]) for key in a_state), "N5 A/B shared runtime initialization differs")
    a_adapter = dict(adapter_a.state_dict())
    b_adapter = dict(adapter_b.state_dict())
    _require(_tree_equal(_cpu_tree(a_adapter), _cpu_tree(b_adapter)), "N5 A/B adapter initialization differs")
    _require(torch.equal(b_state["generator.depth_phase"], torch.zeros_like(b_state["generator.depth_phase"])), "N5 B depth phase is not zero initialized")
    return {
        "A_runtime_digest": _digest_tree(_cpu_tree(a_state)),
        "B_runtime_shared_digest": _digest_tree(_cpu_tree(b_shared)),
        "B_depth_phase_digest": _digest_tree(_cpu_tree(b_state["generator.depth_phase"])),
        "adapter_digest": _digest_tree(_cpu_tree(a_adapter)),
        "shared_runtime_exact": True,
        "shared_adapter_exact": True,
        "depth_phase_exact_zero": True,
    }


def _optimizer(consumer: Any, adapter: Any, candidate: str) -> torch.optim.Adam:
    named = _candidate_named(consumer, adapter, candidate)
    parameters = list(named.values())
    _require(all(parameter.requires_grad for parameter in parameters), f"{candidate} optimizer received frozen parameter")
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    _require(optimizer.param_groups[0]["params"] == parameters, f"{candidate} optimizer parameter order changed")
    return optimizer


def _move_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_tree(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    return copy.deepcopy(value)


def _restore_optimizer(optimizer: torch.optim.Adam, saved: Mapping[str, Any], candidate: str) -> None:
    source_groups = saved.get("param_groups")
    source_state = saved.get("state")
    _require(isinstance(source_groups, list) and len(source_groups) == 1 and isinstance(source_state, Mapping), "C1 optimizer state is invalid")
    source_params = source_groups[0].get("params")
    target_params = list(optimizer.param_groups[0]["params"])
    _require(isinstance(source_params, list) and len(source_params) == len(A_PARAMETER_NAMES), "C1 shared optimizer surface changed")
    _require(len(target_params) == len(_candidate_parameter_names(candidate)), f"{candidate} optimizer surface changed")
    optimizer.param_groups[0].update({key: copy.deepcopy(value) for key, value in source_groups[0].items() if key != "params"})
    optimizer.param_groups[0]["params"] = target_params
    optimizer.state.clear()
    for index, parameter in enumerate(target_params[: len(source_params)]):
        source_key = source_params[index]
        state = source_state.get(source_key, {})
        _require(isinstance(state, Mapping), "C1 optimizer parameter state is invalid")
        optimizer.state[parameter] = _move_tree(state, parameter.device)
    if candidate == "B_TRUE_DEPTH_C1":
        extra = target_params[-1]
        source_step = optimizer.state[target_params[0]].get("step", torch.tensor(0.0, device=extra.device))
        step_value = torch.zeros_like(source_step) if torch.is_tensor(source_step) else 0
        optimizer.state[extra] = {
            "step": step_value,
            "exp_avg": torch.zeros_like(extra),
            "exp_avg_sq": torch.zeros_like(extra),
        }
    _require(len(optimizer.state) == len(target_params), f"{candidate} optimizer state restoration is incomplete")


def _reload_optimizer(optimizer: torch.optim.Adam, saved: Mapping[str, Any], candidate: str) -> None:
    """Reload either a same-candidate checkpoint or the original C1 state."""
    source_groups = saved.get("param_groups")
    target_params = list(optimizer.param_groups[0]["params"])
    _require(isinstance(source_groups, list) and len(source_groups) == 1, f"{candidate} saved optimizer groups are invalid")
    source_params = source_groups[0].get("params")
    _require(isinstance(source_params, list), f"{candidate} saved optimizer parameters are invalid")
    if len(source_params) == len(target_params):
        optimizer.load_state_dict(dict(saved))
        _require(len(optimizer.state) == len(target_params), f"{candidate} same-candidate optimizer reload is incomplete")
    else:
        _restore_optimizer(optimizer, saved, candidate)


def _objective_components(consumer: Any, adapter: Any, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, *, require_grad: bool) -> dict[str, Any]:
    return N4._objective_components(consumer, adapter, batch, labels, prototypes, templates, S_TAU, require_grad=require_grad)


def _gradient_norm(value: torch.Tensor, name: str) -> float:
    _tensor_finite(value, name)
    norm = float(value.detach().abs().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(norm), f"{name} norm is nonfinite")
    return norm


def _gradient_probe(bundle: Mapping[str, Any], c1: Mapping[str, Any], candidate: str, batch_indices: torch.Tensor, batch: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    consumer, adapter = _new_candidate(bundle, c1, candidate, device)
    components = _objective_components(consumer, adapter, batch, labels, bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), require_grad=True)
    named = _candidate_named(consumer, adapter, candidate)
    records: dict[str, Any] = {}
    for objective_name, loss in (("semantic", components["semantic_loss"]), ("image", components["image_loss"]), ("combined", components["total_loss"])):
        gradients = torch.autograd.grad(loss, tuple(named.values()), retain_graph=True, allow_unused=False)
        per_parameter = {name: _gradient_norm(gradient, f"N5 {candidate} {objective_name} {name}") for name, gradient in zip(named, gradients, strict=True)}
        total = math.sqrt(sum(value * value for value in per_parameter.values()))
        _require(math.isfinite(total) and total > 0.0, f"{candidate} {objective_name} gradient is unusable")
        records[objective_name] = {"per_parameter": per_parameter, "total": total}
    wrong_anchor = F.cross_entropy(components["anchor_logits"], (labels + 1) % CLASS_COUNT) + LAMBDA_IMAGE * components["image_loss"]
    wrong_transfer = LAMBDA_TRANSFER * F.softplus(components["delta"]).mean() + LAMBDA_IMAGE * components["image_loss"]
    negatives = {}
    for name, loss in (("wrong_label_anchor", wrong_anchor), ("wrong_direction_transfer", wrong_transfer)):
        gradients = torch.autograd.grad(loss, tuple(named.values()), retain_graph=name == "wrong_label_anchor", allow_unused=False)
        per_parameter = {parameter_name: _gradient_norm(gradient, f"N5 {candidate} {name} {parameter_name}") for parameter_name, gradient in zip(named, gradients, strict=True)}
        total = math.sqrt(sum(value * value for value in per_parameter.values()))
        _require(math.isfinite(total) and total > 0.0, f"{candidate} negative-control gradient is unusable")
        negatives[name] = {"per_parameter": per_parameter, "total": total, "scientific_meaning": "none"}
    trainable_ids = {id(parameter) for parameter in _candidate_named(consumer, adapter, candidate).values()}
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters() if id(parameter) not in trainable_ids), f"{candidate} frozen runtime received a gradient")
    _require(all(parameter.grad is None for parameter in bundle["authorities"]["smooth"].parameters()), "Smooth evaluator gradient firewall failed")
    return {
        "candidate": candidate,
        "batch_indices_sha256": V10.V6.tensor_sha256(batch_indices),
        "batch_size": int(batch.shape[0]),
        "semantic_image_gradient_defined": True,
        "gradients": records,
        "negative_controls": negatives,
    }


def _train_step(consumer: Any, adapter: Any, optimizer: torch.optim.Adam, candidate: str, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, step: int) -> dict[str, Any]:
    before_frozen = _frozen_runtime_digest(consumer, candidate)
    named = _candidate_named(consumer, adapter, candidate)
    before = {name: parameter.detach().cpu().clone() for name, parameter in named.items()}
    optimizer.zero_grad(set_to_none=True)
    components = _objective_components(consumer, adapter, batch, labels, prototypes, templates, require_grad=True)
    total_loss = components["total_loss"]
    _require(total_loss.requires_grad and components["semantic_loss"].requires_grad and components["image_loss"].requires_grad, f"{candidate} objective path is detached")
    total_loss.backward()
    gradients = {}
    for name, parameter in named.items():
        _require(parameter.grad is not None, f"{candidate} {name} gradient is missing")
        gradients[name] = _gradient_norm(parameter.grad, f"{candidate} {name} gradient")
    _require(any(value > 0.0 for value in gradients.values()), f"{candidate} total gradient is zero")
    _require(all(parameter.grad is None for name, parameter in consumer.runtime.named_parameters() if name not in _runtime_trainable_names(candidate)), f"{candidate} frozen runtime received a gradient")
    optimizer.step()
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    _require(before_frozen == _frozen_runtime_digest(consumer, candidate), f"{candidate} frozen runtime state changed")
    updates = {name: float((parameter.detach().cpu() - before[name]).to(torch.float64).square().sum().sqrt()) for name, parameter in named.items()}
    _require(all(math.isfinite(value) and value > 0.0 for value in updates.values()), f"{candidate} declared trainable group did not update")
    return {
        "relative_step": step,
        "global_step": CONTINUATION_OFFSET + step,
        "L_anchor": float(components["anchor_loss"].detach().cpu()),
        "L_transfer": float(components["transfer_loss"].detach().cpu()),
        "L_sem": float(components["semantic_loss"].detach().cpu()),
        "L_image": float(components["image_loss"].detach().cpu()),
        "L_total": float(total_loss.detach().cpu()),
        "Delta_template_mean": float(components["delta"].detach().to(torch.float64).mean().cpu()),
        "gradient_norms": gradients,
        "update_norms": updates,
        "trainable_parameter_names": list(named),
        "adapter_updated": True,
        "generator_updated": True,
        "encoder_updated": False,
        "ols_updated": False,
        "smooth_in_objective": False,
    }


def _measure_train_batch(consumer: Any, adapter: Any, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        components = _objective_components(consumer, adapter, batch, labels, prototypes, templates, require_grad=False)
    return {
        "L_anchor": float(components["anchor_loss"].cpu()),
        "L_transfer": float(components["transfer_loss"].cpu()),
        "L_sem": float(components["semantic_loss"].cpu()),
        "L_image": float(components["image_loss"].cpu()),
        "L_total": float(components["total_loss"].cpu()),
        "Delta_template_mean": float(components["delta"].to(torch.float64).mean().cpu()),
    }


def _generator_diagnostics(consumer: Any, candidate: str, initial_phase: torch.Tensor) -> dict[str, Any]:
    state = consumer.runtime.state_dict()
    phase = state["generator.phase"].detach().cpu().to(torch.float64)
    initial = initial_phase.detach().cpu().to(torch.float64)
    delta = phase - initial
    result: dict[str, Any] = {
        "generator_phase": {
            "phase_change_frobenius": float(torch.linalg.vector_norm(delta).item()),
            "phase_change_max_abs": float(delta.abs().max().item()),
            "phase_frobenius": float(torch.linalg.vector_norm(phase).item()),
            "phase_max_abs": float(phase.abs().max().item()),
            "phase_mean": float(phase.mean().item()),
            "phase_std": float(phase.std(unbiased=False).item()),
        }
    }
    if candidate == "B_TRUE_DEPTH_C1":
        depth = state["generator.depth_phase"].detach().cpu().to(torch.float64)
        result["midpoint_depth_phase"] = {
            "phase_change_frobenius": float(torch.linalg.vector_norm(depth).item()),
            "phase_change_max_abs": float(depth.abs().max().item()),
            "phase_frobenius": float(torch.linalg.vector_norm(depth).item()),
            "phase_max_abs": float(depth.abs().max().item()),
            "phase_mean": float(depth.mean().item()),
            "phase_std": float(depth.std(unbiased=False).item()),
            "shape": list(depth.shape),
            "propagation_distance_m": float(N5DEF.HALF_PROPAGATION_DISTANCE_M),
            "midpoint_crop": False,
            "outside_support": "fixed_transparent",
        }
    return result


def _adapter_diagnostics(adapter: Any) -> dict[str, Any]:
    with torch.no_grad():
        matrix = adapter.matrix().detach().cpu().to(torch.complex128)
    residual = matrix - torch.eye(OLS_PORT_COUNT, dtype=torch.complex128)
    _tensor_finite(residual.real, "N5 adapter residual real")
    _tensor_finite(residual.imag, "N5 adapter residual imag")
    matrix_singular = torch.linalg.svdvals(matrix)
    residual_singular = torch.linalg.svdvals(residual)
    return {
        **V10._adapter_observation(adapter),
        "factor_product_singular_values": [float(value) for value in matrix_singular.tolist()],
        "residual_singular_values": [float(value) for value in residual_singular.tolist()],
    }


def _evaluate_endpoint(consumer: Any, adapter: Any, bundle: Mapping[str, Any], dev_images: torch.Tensor, dev_labels: torch.Tensor, selection_images: torch.Tensor, selection_labels: torch.Tensor, selection_indices: torch.Tensor, step: int, train_trace: Mapping[str, Any] | None, train_metric: Mapping[str, Any], candidate: str, initial_phase: torch.Tensor) -> dict[str, Any]:
    runtime_before = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    adapter_before = _digest_tree(_cpu_tree(adapter.state_dict()))
    with torch.no_grad():
        evaluation = N4._evaluate(consumer, adapter, bundle["authorities"]["smooth"], dev_images, dev_labels, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"], step=step, train_trace=train_trace)
        cyclic = V10._cyclic_transfer_v10(consumer, adapter, bundle["authorities"]["smooth"], selection_images, selection_labels, selection_indices, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"])
    _require(runtime_before == _digest_tree(_cpu_tree(consumer.runtime.state_dict())), f"{candidate} evaluation changed runtime state")
    _require(adapter_before == _digest_tree(_cpu_tree(adapter.state_dict())), f"{candidate} evaluation changed adapter state")
    dev_logits = evaluation["_smooth_logits"].detach().cpu()
    dev_labels_cpu = dev_labels.detach().cpu()
    dev_correct = dev_logits.argmax(dim=1).eq(dev_labels_cpu)
    cyclic_observations = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in cyclic["_observations"].items()}
    return {
        "evaluation": evaluation,
        "cyclic": cyclic,
        "dev_logits": dev_logits,
        "dev_correct": dev_correct,
        "dev_labels": dev_labels_cpu,
        "cyclic_observations": cyclic_observations,
        "train_metric": dict(train_metric),
        "generator_diagnostics": _generator_diagnostics(consumer, candidate, initial_phase),
        "adapter_diagnostics": _adapter_diagnostics(adapter),
        "runtime_state_digest": runtime_before,
        "adapter_state_digest": adapter_before,
    }


def _sufficient_payload(root: Path, contract_ref: Mapping[str, Any], candidate: str, step: int, record: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = record["evaluation"]
    cyclic = record["cyclic"]
    cyclic_observations = {key: value for key, value in record["cyclic_observations"].items() if key not in {"raw_base", "raw_shifted"}}
    return {
        "schema": "lightgen_2d_true_optical_depth_comparison_sufficient_observations.v1",
        "version": 1,
        "task_id": TASK_ID,
        "contract": dict(contract_ref),
        "candidate": candidate,
        "relative_step": step,
        "global_step": CONTINUATION_OFFSET + step,
        "dev_labels": record["dev_labels"],
        "dev_smooth_logits": record["dev_logits"],
        "dev_smooth_correct": record["dev_correct"],
        "dev_ols_margins": evaluation["_margins"].detach().cpu(),
        "dev_anchor_distances": evaluation["_anchor_distances"].detach().cpu(),
        "dev_anchor_predictions": evaluation["_anchor_predictions"].detach().cpu(),
        "dev_anchor_correct_margin": evaluation["_anchor_correct_margin"].detach().cpu(),
        "dev_image_loss": float(evaluation["losses"]["L_image"]),
        "dev_image_count": int(record["dev_labels"].numel()),
        "evaluation_public": _public({key: value for key, value in evaluation.items() if not key.startswith("_")}),
        "cyclic_public": _public({key: value for key, value in cyclic.items() if not key.startswith("_")}),
        "cyclic_observations": cyclic_observations,
        "train_metric": dict(record["train_metric"]),
        "generator_diagnostics": record["generator_diagnostics"],
        "adapter_diagnostics": record["adapter_diagnostics"],
        "raw_validity": {"dev": evaluation["raw_validity"], "cyclic": cyclic["raw_validity"]},
        "collapse": evaluation["collapse"],
        "forward_mode": "torch.no_grad evaluation; sufficient observations persisted before continuation",
    }


def _save_tensor_artifact(root: Path, path: Path, payload: Mapping[str, Any], name: str) -> dict[str, str]:
    N4.atomic_torch_save_new(path, dict(payload), artifact_kind=name)
    digest = _sha256(path)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), name)
    _require(loaded.get("schema") == payload.get("schema"), f"{name} schema reload changed")
    return {"path": _relative(root, path), "sha256": digest}


def _save_sufficient(root: Path, output_dir: Path, contract_ref: Mapping[str, Any], candidate: str, step: int, record: Mapping[str, Any]) -> dict[str, str]:
    path = output_dir / SUFFICIENT_NAME.format(candidate=candidate.lower(), step=f"{step:03d}")
    return _save_tensor_artifact(root, path, _sufficient_payload(root, contract_ref, candidate, step, record), f"N5 {candidate} sufficient observations step {step}")


def _save_raw(root: Path, output_dir: Path, contract_ref: Mapping[str, Any], candidate: str, record: Mapping[str, Any]) -> dict[str, str]:
    observations = record["cyclic_observations"]
    payload = {
        "schema": "lightgen_2d_true_optical_depth_comparison_cyclic_raw_outputs.v1",
        "version": 1,
        "task_id": TASK_ID,
        "candidate": candidate,
        "contract": dict(contract_ref),
        "selection_indices": observations["selection_indices"],
        "labels": observations["labels"],
        "targets": observations["targets"],
        "raw_base": observations["raw_base"],
        "raw_shifted": observations["raw_shifted"],
    }
    path = output_dir / RAW_NAME.format(candidate=candidate.lower())
    return _save_tensor_artifact(root, path, payload, f"N5 {candidate} cyclic raw outputs")


def _checkpoint_payload(root: Path, contract_ref: Mapping[str, Any], c1: Mapping[str, Any], candidate: str, consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: ContinuationSampler, step: int, initial_phase: torch.Tensor) -> dict[str, Any]:
    return {
        "schema": "lightgen_2d_true_optical_depth_comparison_checkpoint.v1",
        "version": 1,
        "task_id": TASK_ID,
        "candidate": candidate,
        "contract": dict(contract_ref),
        "source_c1_step300_checkpoint": {"path": C1_CHECKPOINT_PATH.as_posix(), "sha256": C1_CHECKPOINT_SHA256},
        "relative_step": step,
        "global_step": CONTINUATION_OFFSET + step,
        "runtime_state": _cpu_tree(consumer.runtime.state_dict()),
        "adapter_state": _cpu_tree(adapter.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "sampler_state": _cpu_tree(sampler.state_dict()),
        "rng_state": _cpu_tree(V10.V6._capture_rng()),
        "initial_generator_phase": _cpu_tree(initial_phase),
        "objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image",
        "lambda_transfer": LAMBDA_TRANSFER,
        "lambda_image": LAMBDA_IMAGE,
        "rank": ADAPTER_RANK,
        "trainable_parameter_names": _candidate_parameter_names(candidate),
        "data_roles": {"TRAIN": True, "DEV": True, "protected": False, "official_test": False, "sealed": False},
        "stage_b_executed": False,
        "generation_executed": False,
        "formal_promotion": False,
        "terminal_sampler_rule": "step300 saves cursor=600 without requesting another batch",
        "source_c1_runtime_digest": _digest_tree(_cpu_tree(c1["runtime_state"])),
        "source_c1_adapter_digest": _digest_tree(_cpu_tree(c1["adapter_state"])),
    }


def _validate_checkpoint_reload(root: Path, path: Path, payload: Mapping[str, Any], bundle: Mapping[str, Any], c1: Mapping[str, Any], candidate: str, consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: ContinuationSampler) -> dict[str, Any]:
    digest = _sha256(path)
    loaded = _mapping(N4.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"N5 {candidate} checkpoint")
    _require(loaded.get("schema") == payload.get("schema") and loaded.get("candidate") == candidate and loaded.get("relative_step") == payload.get("relative_step"), f"N5 {candidate} checkpoint identity changed")
    clone, clone_adapter = _new_candidate(bundle, c1, candidate, consumer.device)
    clone.runtime.load_state_dict(dict(loaded["runtime_state"]), strict=True)
    clone_adapter.load_state_dict(dict(loaded["adapter_state"]), strict=True)
    _configure_candidate(clone, clone_adapter, candidate, trainable=True)
    clone_optimizer = _optimizer(clone, clone_adapter, candidate)
    _reload_optimizer(clone_optimizer, _mapping(loaded["optimizer_state"], "N5 checkpoint optimizer"), candidate)
    _require(_digest_tree(_cpu_tree(clone.runtime.state_dict())) == _digest_tree(_cpu_tree(loaded["runtime_state"])), f"N5 {candidate} runtime checkpoint reload changed")
    _require(_digest_tree(_cpu_tree(clone_adapter.state_dict())) == _digest_tree(_cpu_tree(loaded["adapter_state"])), f"N5 {candidate} adapter checkpoint reload changed")
    sampler_clone = ContinuationSampler(bundle["train_indices"], bundle["labels"])
    sampler_clone.load_state_dict(_mapping(loaded["sampler_state"], "N5 checkpoint sampler"))
    if int(loaded["relative_step"]) < SCIENTIFIC_STEPS:
        original_next, original_labels = sampler.next_batch()
        resumed_next, resumed_labels = sampler_clone.next_batch()
        _require(torch.equal(original_next, resumed_next) and torch.equal(original_labels, resumed_labels), f"N5 {candidate} sampler resume identity changed")
        # Restore the live sampler cursor after the non-scientific probe.
        sampler._cursor -= 1
    else:
        _require(sampler_clone.cursor == CONTINUATION_TOTAL_STEPS, "N5 terminal sampler cursor is not exhausted at the declared boundary")
    current_rng = V10.V6._capture_rng()
    V10.V6._restore_rng(_mapping(loaded["rng_state"], "N5 checkpoint RNG"))
    reloaded_rng = V10.V6._capture_rng()
    V10.V6._restore_rng(current_rng)
    _require(_tree_equal(reloaded_rng, loaded["rng_state"]), f"N5 {candidate} checkpoint RNG reload changed")
    return {"path": _relative(root, path), "sha256": digest, "reloaded": True, "resume_identity": True, "relative_step": int(payload["relative_step"]), "global_step": int(payload["global_step"]), "runtime_state_digest": _digest_tree(_cpu_tree(payload["runtime_state"])), "adapter_state_digest": _digest_tree(_cpu_tree(payload["adapter_state"]))}


def _save_checkpoint(root: Path, output_dir: Path, bundle: Mapping[str, Any], c1: Mapping[str, Any], contract_ref: Mapping[str, Any], candidate: str, consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: ContinuationSampler, step: int, initial_phase: torch.Tensor) -> dict[str, str]:
    payload = _checkpoint_payload(root, contract_ref, c1, candidate, consumer, adapter, optimizer, sampler, step, initial_phase)
    path = output_dir / CHECKPOINT_NAME.format(candidate=candidate.lower(), step=f"{step:03d}")
    N4.atomic_torch_save_new(path, payload, artifact_kind=f"N5 {candidate} checkpoint")
    return _validate_checkpoint_reload(root, path, payload, bundle, c1, candidate, consumer, adapter, optimizer, sampler)


def _step0_equivalence(bundle: Mapping[str, Any], c1: Mapping[str, Any], batch: torch.Tensor, device: torch.device) -> dict[str, Any]:
    consumer_a, adapter_a = _new_frozen_candidate(bundle, c1, "A_CONTINUED_C1", device)
    consumer_b, adapter_b = _new_frozen_candidate(bundle, c1, "B_TRUE_DEPTH_C1", device)
    with torch.no_grad():
        forward_a = V10._forward_adapter(consumer_a, adapter_a, batch, require_grad=False)
        forward_b = V10._forward_adapter(consumer_b, adapter_b, batch, require_grad=False)
    comparisons = {}
    for name, left, right in (("adapted_coefficients", forward_a["z_adapted"], forward_b["z_adapted"]), ("generator_input", forward_a["adapted_result"].generator_input_field.tensor, forward_b["adapted_result"].generator_input_field.tensor), ("raw_cmos", forward_a["raw_base"], forward_b["raw_base"])):
        comparisons[name] = {"allclose": bool(torch.allclose(left, right, **TOLERANCE)), "max_abs_diff": float((left - right).abs().max().item())}
    _require(all(value["allclose"] for value in comparisons.values()), "N5 A/B step0 real-path equivalence failed")
    _require(torch.equal(consumer_b.runtime.generator.depth_phase, torch.zeros_like(consumer_b.runtime.generator.depth_phase)), "N5 B step0 depth phase is not zero")
    return {"passed": True, "batch_size": int(batch.shape[0]), "tolerance": dict(TOLERANCE), "comparisons": comparisons, "B_depth_phase_exact_zero": True, "training_updates": 0}


def _preflight_smoke(root: Path, output_dir: Path, bundle: Mapping[str, Any], c1: Mapping[str, Any], contract_ref: Mapping[str, Any], candidate: str, images: torch.Tensor, labels: torch.Tensor, device: torch.device) -> dict[str, Any]:
    consumer, adapter = _new_candidate(bundle, c1, candidate, device)
    firewall = _configure_candidate(consumer, adapter, candidate, trainable=True)
    optimizer = _optimizer(consumer, adapter, candidate)
    _restore_optimizer(optimizer, _mapping(c1["checkpoint"]["optimizer_state"], "C1 optimizer state"), candidate)
    sampler = ContinuationSampler(bundle["train_indices"], bundle["labels"])
    initial_phase = c1["runtime_state"]["generator.phase"]
    traces = []
    prototypes = bundle["authorities"]["prototypes"].to(device)
    templates = bundle["authorities"]["templates"].to(device)
    for step in range(1, PRECHECK_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        traces.append(_train_step(consumer, adapter, optimizer, candidate, images.index_select(0, indices).to(device), batch_labels.to(device), prototypes, templates, step))
    path = output_dir / PREFLIGHT_CHECKPOINT_NAME.format(candidate=candidate.lower())
    payload = _checkpoint_payload(root, contract_ref, c1, candidate, consumer, adapter, optimizer, sampler, PRECHECK_STEPS, initial_phase)
    N4.atomic_torch_save_new(path, payload, artifact_kind=f"N5 {candidate} disposable preflight checkpoint")
    reload_ref = _validate_checkpoint_reload(root, path, payload, bundle, c1, candidate, consumer, adapter, optimizer, sampler)
    return {"firewall": firewall, "traces": traces, "disposable_updates": PRECHECK_STEPS, "scientific_updates": 0, "checkpoint_reload": reload_ref, "sampler_cursor_after_smoke": sampler.cursor, "scientific_meaning": "none"}


def _preflight_payload(root: Path, started: Mapping[str, Any], contract_ref: Mapping[str, Any], device_facts: Mapping[str, Any], authorities: Mapping[str, Any], step0: Mapping[str, Any], gradients: Mapping[str, Any], smokes: Mapping[str, Any], synthetic: Mapping[str, Any], schedule: Mapping[str, Any], c1: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "lightgen_2d_true_optical_depth_comparison_preflight_report.v1",
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "AUTHORIZABLE",
        "status": "valid_completed",
        "classification": "LIGHTGEN_2D_TRUE_OPTICAL_DEPTH_COMPARISON_AUTHORIZABLE",
        "outer_gate": OUTER_GATE,
        "design_authority": DESIGN_AUTHORITY,
        "provenance": {"git_start": dict(started)},
        "contract": dict(contract_ref),
        "authorities": dict(authorities),
        "runtime": {"device": dict(device_facts), "production_grid": PRODUCTION_GRID},
        "source_c1": {"path": C1_CHECKPOINT_PATH.as_posix(), "sha256": C1_CHECKPOINT_SHA256, "checkpoint_step": 300, "runtime_digest": _digest_tree(_cpu_tree(c1["runtime_state"])), "adapter_digest": _digest_tree(_cpu_tree(c1["adapter_state"]))},
        "continuation_schedule": schedule,
        "step0_equivalence": step0,
        "gradient_probes": gradients,
        "disposable_smokes": smokes,
        "synthetic_true_depth_checks": synthetic,
        "checks": {"authorities": True, "c1_endpoint": True, "shared_c1_initialization": True, "step0_real_path_equivalence": True, "production_true_depth_zero_phase_equivalence": True, "midpoint_geometry": True, "trainable_frozen_firewall": True, "finite_usable_gradients": True, "negative_controls": True, "three_step_real_disposable_updates": True, "checkpoint_reload_resume": True, "fresh_sampler_offset": True},
        "claim_boundary": {"engineering_preflight_real_updates": PRECHECK_STEPS * 2, "scientific_updates": 0, "scientific_forward_evaluations": 0, "scientific_training_executed": False, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False},
        "one_shot": {"preflight": 1, "candidate_a_scientific_attempts": 1, "candidate_b_scientific_attempts": 1, "retry": False, "second_arm": False},
        "next": "proceed to exactly one A and one B 300-update scientific comparison",
    }


def _validate_n5_authorities(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    contract = _validate_contract(root)
    authority_summary = N5DEF._validate_authorities(root)
    c1 = N4ATTR._validate_checkpoint(root, "C1", _path(root, C1_CHECKPOINT_PATH), C1_CHECKPOINT_SHA256)
    original_reports = N4ATTR._validate_original_reports(root)
    checkpoint = _mapping(c1["checkpoint"], "N5 C1 checkpoint")
    _require(checkpoint.get("trainable_parameter_names") == A_PARAMETER_NAMES, "C1 trainable parameter surface is not the frozen rank9/2-layer endpoint")
    _require(isinstance(checkpoint.get("optimizer_state"), Mapping), "C1 optimizer state is unavailable")
    c1_sampler = _mapping(checkpoint.get("sampler_state"), "C1 sampler state")
    _require(c1_sampler.get("steps") == SCIENTIFIC_STEPS and c1_sampler.get("cursor") == SCIENTIFIC_STEPS, "C1 source sampler boundary changed")
    _require(checkpoint.get("checkpoint_step") == SCIENTIFIC_STEPS, "C1 source checkpoint step changed")
    _require(checkpoint.get("stage_b_executed") is False and checkpoint.get("generation_executed") is False and checkpoint.get("formal_promotion") is False, "C1 claim boundary changed")
    n4_report = _read_json(_path(root, N4_ATTRIBUTION_REPORT_PATH), "N4 endpoint attribution report")
    sufficient_ref = _mapping(_mapping(n4_report.get("artifacts"), "N4 attribution artifacts").get("sufficient_observations"), "N4 attribution sufficient observations reference")
    sufficient_path = _path(root, str(sufficient_ref.get("path")))
    _validate_sha(sufficient_path, str(sufficient_ref.get("sha256")), "N4 attribution sufficient observations")
    _require(str(sufficient_ref.get("sha256")) == N4_ATTRIBUTION_SUFFICIENT_SHA256, "N4 attribution sufficient-observation identity changed")
    sufficient = _mapping(N4.safe_torch_load(sufficient_path, expected_sha256=N4_ATTRIBUTION_SUFFICIENT_SHA256, map_location="cpu"), "N4 attribution sufficient observations")
    _require(sufficient.get("schema") == "lightgen_2d_n4_endpoint_attribution_sufficient_observations.v1", "N4 attribution sufficient schema changed")
    _require(tuple(sufficient["dev_labels"].shape) == (5000,) and tuple(sufficient["endpoints"]["C1"]["dev_smooth_correct"].shape) == (5000,), "N4 C1 paired baseline observations changed")
    _require(float(n4_report["smooth"]["C1"]) == 0.4838, "N4 C1 baseline Smooth changed")
    authorities = {"n5_design": {"path": _relative(root, _path(root, CONTRACT_PATH)), "sha256": CONTRACT_SHA256}, "n5_definition": {"path": _relative(root, Path(__file__).with_name("define_lightgen_2d_true_optical_depth_comparison_contract_v1.py")), "sha256": str(contract["definition_entrypoint"]["sha256"])}, "n4_endpoint_report": {"path": _relative(root, _path(root, N4_ATTRIBUTION_REPORT_PATH)), "sha256": N4_ATTRIBUTION_REPORT_SHA256}, "c1_checkpoint": {"path": _relative(root, _path(root, C1_CHECKPOINT_PATH)), "sha256": C1_CHECKPOINT_SHA256}, "n3": authority_summary["n3"], "v11": authority_summary["v11"]}
    return contract, c1, {"n4_report": n4_report, "n4_sufficient": sufficient, "n4_sufficient_ref": {"path": _relative(root, sufficient_path), "sha256": N4_ATTRIBUTION_SUFFICIENT_SHA256}}, {"authorities": authorities, "original_reports": original_reports}


def run_preflight(root: Path) -> dict[str, Any]:
    _require(not PREFLIGHT_DIR.exists(), f"refusing to reuse N5 preflight directory: {_path(root, PREFLIGHT_DIR)}")
    for directory in (CANDIDATE_A_DIR, CANDIDATE_B_DIR, AGGREGATE_DIR):
        _require(not directory.exists(), f"N5 one-shot output boundary is not fresh: {_path(root, directory)}")
    PREFLIGHT_DIR.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = _status_lines(root)
    payload: dict[str, Any] = {"schema": "lightgen_2d_true_optical_depth_comparison_preflight_report.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N5_PREFLIGHT_HARNESS", "outer_gate": OUTER_GATE, "provenance": {"git_start": started}, "claim_boundary": {"engineering_preflight_real_updates": 0, "scientific_updates": 0, "scientific_training_executed": False}}
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from frozen N5 authorization")
        contract, c1, baseline, authority_context = _validate_n5_authorities(root)
        V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        _require(device.type == "cuda", "CUDA is required for N5 scientific continuation")
        bundle = N4._validate_v11_authorities(root)
        bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = V10.V6._load_roles(root)
        bundle.update({"images": images, "labels": labels, "train_indices": train_indices, "development_indices": development_indices})
        _require(len(dataset.get("protected", [])) == 0 if isinstance(dataset.get("protected"), list) else dataset.get("protected", 0) == 0, "protected data role is not empty")
        V10.V6._restore_rng(_mapping(c1["checkpoint"]["rng_state"], "C1 source RNG"))
        schedule = ContinuationSampler(train_indices, labels)
        first_indices, first_labels = schedule.batch_at(CONTINUATION_OFFSET)
        first_batch = images.index_select(0, first_indices).to(device)
        first_labels = first_labels.to(device)
        step0 = _step0_equivalence(bundle, c1, first_batch, device)
        initial_a, initial_adapter_a = _new_candidate(bundle, c1, "A_CONTINUED_C1", device)
        initial_b, initial_adapter_b = _new_candidate(bundle, c1, "B_TRUE_DEPTH_C1", device)
        shared = _initial_shared_digests(initial_a, initial_adapter_a, initial_b, initial_adapter_b)
        gradients: dict[str, Any] = {}
        smokes: dict[str, Any] = {}
        for candidate in ("A_CONTINUED_C1", "B_TRUE_DEPTH_C1"):
            V10.V6._restore_rng(_mapping(c1["checkpoint"]["rng_state"], "C1 source RNG"))
            gradients[candidate] = _gradient_probe(bundle, c1, candidate, first_indices, first_batch, first_labels, device)
            smokes[candidate] = _preflight_smoke(root, PREFLIGHT_DIR, bundle, c1, {"path": _relative(root, _path(root, CONTRACT_PATH)), "sha256": CONTRACT_SHA256}, candidate, images, labels, device)
        synthetic = N5DEF._synthetic_checks(device)
        _require(synthetic.get("status") == "PASS", "N5 production true-depth synthetic checks failed")
        schedule_state = ContinuationSampler(train_indices, labels).state_dict()
        schedule_summary = {"schema": ContinuationSampler.SCHEMA, "seed": SPLIT_SEED, "global_offset": CONTINUATION_OFFSET, "total_steps": CONTINUATION_TOTAL_STEPS, "initial_cursor": CONTINUATION_OFFSET, "first_batch_indices_sha256": V10.V6.tensor_sha256(first_indices), "first_batch_labels_sha256": V10.V6.tensor_sha256(first_labels), "schedule_digest": _digest_tree(_cpu_tree(schedule_state))}
        contract_ref = {"path": _relative(root, _path(root, CONTRACT_PATH)), "sha256": CONTRACT_SHA256}
        input_payload = {"schema": "lightgen_2d_true_optical_depth_comparison_preflight_input.v1", "version": 1, "task_id": TASK_ID, "design_authority": DESIGN_AUTHORITY, "outer_gate": OUTER_GATE, "contract": contract_ref, "candidates": ["A_CONTINUED_C1", "B_TRUE_DEPTH_C1"], "source_c1": {"path": C1_CHECKPOINT_PATH.as_posix(), "sha256": C1_CHECKPOINT_SHA256}, "objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "sampler": schedule_summary, "data_roles": {"TRAIN": 55000, "DEV": 5000, "protected": 0, "official_test": 0, "sealed": 0}, "scientific_updates": 0}
        input_path = PREFLIGHT_DIR / PREFLIGHT_INPUT_NAME
        N4.atomic_write_new_json(input_path, input_payload, artifact_kind="N5 preflight input")
        payload = _preflight_payload(root, started, contract_ref, device_facts, authority_context["authorities"], step0, gradients, smokes, synthetic, schedule_summary | {"shared_initialization": shared}, c1)
        payload["input"] = {"path": _relative(root, input_path), "sha256": _sha256(input_path)}
        payload["dataset"] = _public(dataset)
        payload["runtime"]["cuda_dll_search_path"] = V10.V6._prepare_cuda_runtime_dll_search_path()
        payload["claim_boundary"] = {"engineering_preflight_real_updates": PRECHECK_STEPS * 2, "scientific_updates": 0, "scientific_forward_evaluations": 0, "scientific_training_executed": False, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}
        payload["provenance"]["shared_initialization"] = shared
        payload["provenance"]["git_end"] = V10.V6.git_facts(root)
    except HardBlockedN5 as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": "N5_PREFLIGHT_HARD_BLOCKED", "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific execution"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N5_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR", "invalid_input": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12000:]}, "next": "stop; no scientific execution"})
    finally:
        ending = V10.V6.git_facts(root)
        payload.setdefault("provenance", {})["git_end"] = ending
        payload["provenance"]["branch_unchanged"] = started.get("branch") == ending.get("branch")
        payload["provenance"]["head_unchanged"] = started.get("head") == ending.get("head")
        payload["provenance"]["worktree_status_unchanged"] = status_before == _status_lines(root)
    report_path = PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME
    N4.atomic_write_new_json(report_path, payload, artifact_kind="N5 preflight report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_true_optical_depth_comparison_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "report": report_ref, "provenance": payload["provenance"], "claim_boundary": payload["claim_boundary"]}, artifact_kind="N5 preflight lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "scientific_updates": 0}


def _load_preflight(root: Path) -> dict[str, Any]:
    report_path = _path(root, PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = _path(root, PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    _require(report_path.is_file() and lineage_path.is_file(), "N5 preflight authority is unavailable")
    report = _read_json(report_path, "N5 preflight report")
    lineage = _read_json(lineage_path, "N5 preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed" and report.get("classification") == "LIGHTGEN_2D_TRUE_OPTICAL_DEPTH_COMPARISON_AUTHORIZABLE", "N5 preflight is not authorizable")
    _require(report.get("claim_boundary", {}).get("scientific_updates") == 0 and report.get("claim_boundary", {}).get("scientific_training_executed") is False, "N5 preflight scientific boundary changed")
    report_hash = _sha256(report_path)
    _require(lineage.get("report", {}).get("sha256") == report_hash, "N5 preflight lineage/report binding changed")
    return {"report": report, "report_ref": {"path": _relative(root, report_path), "sha256": report_hash}, "lineage_ref": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}}


def _scientific_payload(candidate: str, started: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "lightgen_2d_true_optical_depth_comparison_scientific_report.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N5_SCIENTIFIC_HARNESS", "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "provenance": {"git_start": dict(started)}, "claim_boundary": {"scientific_training_executed": False, "scientific_forward_evaluations": 0, "scientific_updates": 0, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {"candidate": candidate, "rank": ADAPTER_RANK, "retry": False, "second_arm": False, "rank_change": False, "lambda_change": False, "loss_change": False, "lr_change": False, "budget_change": False, "encoder_update": False, "ols_update": False, "stage_b": False, "generation": False, "promotion": False}}


def run_candidate(root: Path, candidate: str) -> dict[str, Any]:
    output_dir = CANDIDATE_A_DIR if candidate == "A_CONTINUED_C1" else CANDIDATE_B_DIR
    _require(not output_dir.exists(), f"refusing to reuse N5 {candidate} output directory: {_path(root, output_dir)}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = _status_lines(root)
    payload = _scientific_payload(candidate, started)
    scientific_started = False
    updates_completed = 0
    forward_evaluations = 0
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from frozen N5 authorization")
        preflight = _load_preflight(root)
        contract, c1, baseline, authority_context = _validate_n5_authorities(root)
        bundle = N4._validate_v11_authorities(root)
        V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        _require(device.type == "cuda", "CUDA is required for N5 scientific continuation")
        bundle["authorities"]["smooth"] = bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = V10.V6._load_roles(root)
        bundle.update({"images": images, "labels": labels, "train_indices": train_indices, "development_indices": development_indices})
        dev_images = images.index_select(0, development_indices).contiguous()
        dev_labels = labels.index_select(0, development_indices).contiguous()
        selection_images, selection_labels, selection_indices = V10.V6._load_selection(root, images, labels, development_indices)
        _require(V10.V6.tensor_sha256(selection_indices) == N4.SELECTION_INDICES_SHA256 and V10.V6.tensor_sha256(selection_labels) == N4.SELECTION_LABELS_SHA256, "N5 selection authority changed")
        continuation = ContinuationSampler(train_indices, labels)
        V10.V6._restore_rng(_mapping(c1["checkpoint"]["rng_state"], "C1 source RNG"))
        consumer, adapter = _new_candidate(bundle, c1, candidate, device)
        firewall = _configure_candidate(consumer, adapter, candidate, trainable=True)
        optimizer = _optimizer(consumer, adapter, candidate)
        _restore_optimizer(optimizer, _mapping(c1["checkpoint"]["optimizer_state"], "C1 source optimizer"), candidate)
        initial_phase = c1["runtime_state"]["generator.phase"].detach().cpu()
        first_indices, first_labels = continuation.batch_at(CONTINUATION_OFFSET)
        first_batch = images.index_select(0, first_indices).to(device)
        first_labels_device = first_labels.to(device)
        with torch.no_grad():
            initial_train_metric = _measure_train_batch(consumer, adapter, first_batch, first_labels_device, bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device))
        contract_ref = {"path": _relative(root, _path(root, CONTRACT_PATH)), "sha256": CONTRACT_SHA256}
        input_payload = {"schema": "lightgen_2d_true_optical_depth_comparison_scientific_input.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "design_authority": DESIGN_AUTHORITY, "outer_gate": OUTER_GATE, "contract": contract_ref, "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"], "result": "AUTHORIZABLE"}, "source_c1": {"path": _relative(root, _path(root, C1_CHECKPOINT_PATH)), "sha256": C1_CHECKPOINT_SHA256, "checkpoint_step": 300}, "protocol": {"objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE, "rank": ADAPTER_RANK, "optimizer": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "batch": LOGICAL_BATCH_SIZE, "additional_updates": SCIENTIFIC_STEPS, "relative_checkpoints": list(CHECKPOINT_STEPS), "global_offset": CONTINUATION_OFFSET, "continuation_total_steps": CONTINUATION_TOTAL_STEPS, "trainable": _candidate_parameter_names(candidate)}, "data_roles": {"TRAIN": 55000, "DEV": 5000, "protected": 0, "official_test": 0, "sealed": 0}, "one_shot": {"candidate_attempt": 1, "retry": False}, "claim_boundary": payload["claim_boundary"]}
        input_path = output_dir / SCIENTIFIC_INPUT_NAME.format(candidate=candidate.lower())
        N4.atomic_write_new_json(input_path, input_payload, artifact_kind=f"N5 {candidate} scientific input")
        scientific_started = True
        payload.update({"contract": contract_ref, "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"], "result": "AUTHORIZABLE"}, "source_c1": {"path": _relative(root, _path(root, C1_CHECKPOINT_PATH)), "sha256": C1_CHECKPOINT_SHA256, "checkpoint_step": 300}, "runtime": {"device": device_facts, "production_grid": PRODUCTION_GRID, "candidate": candidate}, "authorities": authority_context["authorities"], "dataset": _public(dataset), "firewall": firewall, "objective": input_payload["protocol"], "initial_state": {"source_c1_runtime_digest": _digest_tree(_cpu_tree(c1["runtime_state"])), "source_c1_adapter_digest": _digest_tree(_cpu_tree(c1["adapter_state"])), "candidate_frozen_runtime_digest": _frozen_runtime_digest(consumer, candidate), "generator_phase": _generator_diagnostics(consumer, candidate, initial_phase), "input": {"path": _relative(root, input_path), "sha256": _sha256(input_path)}}, "sampler": {"schema": ContinuationSampler.SCHEMA, "seed": SPLIT_SEED, "global_offset": CONTINUATION_OFFSET, "total_steps": CONTINUATION_TOTAL_STEPS, "first_batch_indices_sha256": V10.V6.tensor_sha256(first_indices), "first_batch_labels_sha256": V10.V6.tensor_sha256(first_labels)}, "data_roles": input_payload["data_roles"]})
        evaluations: dict[str, Any] = {}
        train_metrics: dict[str, Any] = {}
        sufficient_refs: dict[str, Any] = {}
        checkpoint_refs: dict[str, Any] = {}
        raw_ref: dict[str, Any] | None = None
        # Step 0 is a scientific endpoint observation, but no optimizer update.
        record = _evaluate_endpoint(consumer, adapter, bundle, dev_images, dev_labels, selection_images, selection_labels, selection_indices, 0, None, initial_train_metric, candidate, initial_phase)
        forward_evaluations += 2
        evaluations["0"] = record
        train_metrics["0"] = initial_train_metric
        sufficient_refs["0"] = _save_sufficient(root, output_dir, contract_ref, candidate, 0, record)
        checkpoint_refs["0"] = _save_checkpoint(root, output_dir, bundle, c1, contract_ref, candidate, consumer, adapter, optimizer, continuation, 0, initial_phase)
        training_start = time.perf_counter()
        for step in range(1, SCIENTIFIC_STEPS + 1):
            indices, batch_labels = continuation.next_batch()
            batch = images.index_select(0, indices).to(device)
            trace = _train_step(consumer, adapter, optimizer, candidate, batch, batch_labels.to(device), bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), step)
            updates_completed = step
            del batch
            if step in CHECKPOINT_STEPS[1:]:
                with torch.no_grad():
                    train_metric = _measure_train_batch(consumer, adapter, first_batch, first_labels_device, bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device))
                record = _evaluate_endpoint(consumer, adapter, bundle, dev_images, dev_labels, selection_images, selection_labels, selection_indices, step, trace, train_metric, candidate, initial_phase)
                forward_evaluations += 2
                evaluations[str(step)] = record
                train_metrics[str(step)] = train_metric
                # Sufficient observations are write-once and precede the checkpoint/next continuation.
                sufficient_refs[str(step)] = _save_sufficient(root, output_dir, contract_ref, candidate, step, record)
                checkpoint_refs[str(step)] = _save_checkpoint(root, output_dir, bundle, c1, contract_ref, candidate, consumer, adapter, optimizer, continuation, step, initial_phase)
                print(json.dumps({"event": "N5_scientific_progress", "candidate": candidate, "relative_step": step, "global_step": CONTINUATION_OFFSET + step, "elapsed_seconds": time.perf_counter() - training_start}, ensure_ascii=False), flush=True)
        _require(set(evaluations) == {"0", "100", "200", "300"} and set(sufficient_refs) == {"0", "100", "200", "300"} and set(checkpoint_refs) == {"0", "100", "200", "300"}, f"{candidate} checkpoint/evaluation set is incomplete")
        final_record = evaluations["300"]
        raw_ref = _save_raw(root, output_dir, contract_ref, candidate, final_record)
        baseline_correct = baseline["n4_sufficient"]["endpoints"]["C1"]["dev_smooth_correct"].to(torch.float64)
        step0_correct = evaluations["0"]["dev_correct"].to(torch.float64)
        final_correct = evaluations["300"]["dev_correct"].to(torch.float64)
        _require(torch.equal(baseline["n4_sufficient"]["dev_labels"], dev_labels.cpu()), "N4 paired C1 baseline DEV rows do not match N5 DEV rows")
        optimization_ci = V10.V6._paired_bootstrap(final_correct - baseline_correct, dev_labels.cpu(), replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        training_ci = V10.V6._paired_bootstrap(final_correct - step0_correct, dev_labels.cpu(), replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        reconstruction_delta = float(evaluations["300"]["evaluation"]["losses"]["L_image"] - evaluations["0"]["evaluation"]["losses"]["L_image"])
        raw_valid = bool(all(evaluations[str(step)]["evaluation"]["raw_validity"]["finite"] and evaluations[str(step)]["evaluation"]["raw_validity"]["structured"] and evaluations[str(step)]["evaluation"]["raw_validity"]["nonconstant"] for step in CHECKPOINT_STEPS) and final_record["cyclic"]["raw_validity"]["base"]["finite"] and final_record["cyclic"]["raw_validity"]["shifted"]["finite"])
        payload.update({"result": "VALID_COMPLETED", "status": "valid_completed", "classification": f"N5_{candidate}_SCIENTIFIC_COMPLETED", "scientific_updates": SCIENTIFIC_STEPS, "scientific_forward_evaluations": forward_evaluations, "evaluations": {step: {key: value for key, value in record["evaluation"].items() if not key.startswith("_")} for step, record in evaluations.items()}, "train_metrics": train_metrics, "checkpoints": checkpoint_refs, "sufficient_observations": sufficient_refs, "cyclic_raw_outputs": raw_ref, "optimization_horizon": {"baseline": "C1 step300 Smooth=0.4838", "R300": float(final_record["evaluation"]["smooth"]["macro_recall"]), "Delta_A": float(final_record["evaluation"]["smooth"]["macro_recall"] - 0.4838), "paired_bootstrap": {key: value for key, value in optimization_ci.items() if key != "delta_records"}}, "training_delta": {"R0": float(evaluations["0"]["evaluation"]["smooth"]["macro_recall"]), "R300": float(final_record["evaluation"]["smooth"]["macro_recall"]), "Delta_R": float(final_record["evaluation"]["smooth"]["macro_recall"] - evaluations["0"]["evaluation"]["smooth"]["macro_recall"]), "paired_bootstrap": {key: value for key, value in training_ci.items() if key != "delta_records"}}, "reconstruction": {"DEV_L_image_step0": float(evaluations["0"]["evaluation"]["losses"]["L_image"]), "DEV_L_image_step300": float(evaluations["300"]["evaluation"]["losses"]["L_image"]), "Delta_L_image": reconstruction_delta, "max_allowed_delta": RECONSTRUCTION_MAX_DELTA}, "cyclic_transfer": {key: value for key, value in final_record["cyclic"].items() if not key.startswith("_")}, "raw_diversity": {"step300": {"dev": final_record["evaluation"]["raw_validity"], "collapse": final_record["evaluation"]["collapse"], "cyclic": final_record["cyclic"]["raw_validity"]}}, "diagnostics": {"step0": {"adapter": evaluations["0"]["adapter_diagnostics"], "generator": evaluations["0"]["generator_diagnostics"]}, "step300": {"adapter": final_record["adapter_diagnostics"], "generator": final_record["generator_diagnostics"]}, "true_depth_geometry": {"candidate_b_only": candidate == "B_TRUE_DEPTH_C1", "midpoint_shape": [PRODUCTION_GRID, PRODUCTION_GRID], "active_support": [100, 100], "midpoint_crop": False, "outside_support": "fixed_transparent", "half_distance_m": float(N5DEF.HALF_PROPAGATION_DISTANCE_M)}}, "raw_validity_pass": raw_valid, "claim_boundary": {"scientific_training_executed": True, "scientific_forward_evaluations": forward_evaluations, "scientific_updates": SCIENTIFIC_STEPS, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {**payload["governance"], "scientific_attempt": 1, "adapter_update": True, "generator_phase_update": True, "depth_phase_update": candidate == "B_TRUE_DEPTH_C1", "retries": False, "no_adaptive_change": True}, "next": "candidate result is bounded input to the frozen N5 pairwise outer gate; Stage B and generation remain closed"})
    except HardBlockedN5 as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": "N5_SCIENTIFIC_HARD_BLOCKED", "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific retry"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N5_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "invalid_input": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12000:]}, "next": "stop; no scientific retry"})
    finally:
        ending = V10.V6.git_facts(root)
        payload.setdefault("provenance", {}).update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == _status_lines(root), "scientific_started": scientific_started})
        payload["claim_boundary"]["scientific_training_executed"] = bool(scientific_started)
        payload["claim_boundary"]["scientific_updates"] = updates_completed
        payload["claim_boundary"]["scientific_forward_evaluations"] = forward_evaluations
    report_path = output_dir / SCIENTIFIC_REPORT_NAME.format(candidate=candidate.lower())
    N4.atomic_write_new_json(report_path, payload, artifact_kind=f"N5 {candidate} scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = output_dir / SCIENTIFIC_LINEAGE_NAME.format(candidate=candidate.lower())
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_true_optical_depth_comparison_scientific_lineage.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "design_authority": DESIGN_AUTHORITY, "report": report_ref, "contract": payload.get("contract"), "preflight": payload.get("preflight"), "provenance": payload["provenance"], "claim_boundary": payload["claim_boundary"], "scientific_observations_new": bool(scientific_started), "retry": False}, artifact_kind=f"N5 {candidate} scientific lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "candidate": candidate, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "scientific_updates": updates_completed, "scientific_forward_evaluations": forward_evaluations, "smooth_R300": payload.get("evaluations", {}).get("300", {}).get("smooth", {}).get("macro_recall"), "next": payload.get("next")}


def _load_step_sufficient(root: Path, report: Mapping[str, Any], candidate: str, step: int) -> Mapping[str, Any]:
    ref = _mapping(report.get("sufficient_observations", {}).get(str(step)), f"N5 {candidate} sufficient step {step}")
    path = _path(root, ref["path"])
    _validate_sha(path, str(ref["sha256"]), f"N5 {candidate} sufficient step {step}")
    return _mapping(N4.safe_torch_load(path, expected_sha256=str(ref["sha256"]), map_location="cpu"), f"N5 {candidate} sufficient step {step}")


def _classification_from_reports(a_report: Mapping[str, Any], b_report: Mapping[str, Any], a_sufficient: Mapping[str, Any], b_sufficient: Mapping[str, Any], labels: torch.Tensor) -> dict[str, Any]:
    a300 = a_sufficient["dev_smooth_correct"].to(torch.float64)
    b300 = b_sufficient["dev_smooth_correct"].to(torch.float64)
    delta_depth = b300 - a300
    depth_ci = V10.V6._paired_bootstrap(delta_depth, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    a_material = float(a_report["optimization_horizon"]["Delta_A"]) >= MATERIAL_DELTA and float(a_report["optimization_horizon"]["paired_bootstrap"]["ci_lower"]) > 0.0
    depth_material = float(delta_depth.mean()) >= MATERIAL_DELTA and float(depth_ci["ci_lower"]) > 0.0
    b_reconstruction = float(b_report["reconstruction"]["Delta_L_image"])
    b_transfer = bool(b_report["cyclic_transfer"]["template"]["pass"] and b_report["cyclic_transfer"]["smooth"]["pass"] and b_report["cyclic_transfer"]["broad"] and not b_report["cyclic_transfer"]["pair_dominated"])
    raw = bool(b_report["raw_validity_pass"])
    primary_supported = bool(depth_material and b_reconstruction <= RECONSTRUCTION_MAX_DELTA and b_transfer and raw)
    if a_material and not depth_material:
        classification = "OPTIMIZATION_HORIZON_DOMINANT"
    elif depth_material:
        classification = "PROPAGATION_SEPARATED_SYNTHESIS_CAPACITY_SUPPORTED"
    elif not a_material and not depth_material:
        classification = "CURRENT_LINEAR_INTERFACE_OPTICAL_SYNTHESIS_FAMILY_PLATEAU_SUPPORTED"
    else:
        classification = "N5_TRUE_OPTICAL_DEPTH_COMPARISON_MIXED_OR_UNRESOLVED"
    return {"classification": classification, "primary_gate": {"name": OUTER_GATE, "supported": primary_supported, "Delta_depth": float(delta_depth.mean()), "paired_bootstrap": {key: value for key, value in depth_ci.items() if key != "delta_records"}, "B_reconstruction_pass": b_reconstruction <= RECONSTRUCTION_MAX_DELTA, "B_cyclic_transfer_pass": b_transfer, "B_raw_validity_pass": raw}, "optimization_horizon_material": a_material, "depth_material": depth_material}


def _aggregate(root: Path, contract: Mapping[str, Any], preflight: Mapping[str, Any], a_result: Mapping[str, Any], b_result: Mapping[str, Any], a_report: Mapping[str, Any], b_report: Mapping[str, Any], a_sufficient_steps: Mapping[str, Mapping[str, Any]], b_sufficient_steps: Mapping[str, Mapping[str, Any]], n4_sufficient: Mapping[str, Any]) -> dict[str, Any]:
    _require(not AGGREGATE_DIR.exists(), f"refusing to reuse N5 aggregate directory: {_path(root, AGGREGATE_DIR)}")
    _require(a_result["result"] == "VALID_COMPLETED" and b_result["result"] == "VALID_COMPLETED", "N5 aggregate requires both valid candidate reports")
    output_dir = _path(root, AGGREGATE_DIR)
    output_dir.mkdir(parents=True, exist_ok=False)
    labels = a_sufficient_steps["300"]["dev_labels"]
    _require(torch.equal(labels, b_sufficient_steps["300"]["dev_labels"]), "N5 A/B DEV labels are not symmetric")
    a300 = a_sufficient_steps["300"]["dev_smooth_correct"].to(torch.float64)
    b300 = b_sufficient_steps["300"]["dev_smooth_correct"].to(torch.float64)
    c1_correct = n4_sufficient["endpoints"]["C1"]["dev_smooth_correct"].to(torch.float64)
    _require(torch.equal(n4_sufficient["dev_labels"], labels), "N4 C1 paired baseline rows do not match N5 rows")
    delta_a = a300 - c1_correct
    delta_depth = b300 - a300
    delta_b = b300 - c1_correct
    ci_a = V10.V6._paired_bootstrap(delta_a, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    ci_depth = V10.V6._paired_bootstrap(delta_depth, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    ci_b = V10.V6._paired_bootstrap(delta_b, labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    classification = _classification_from_reports(a_report, b_report, a_sufficient_steps["300"], b_sufficient_steps["300"], labels)
    stage_a = {
        "A_CONTINUED_C1": {"R300": float(a_report["evaluations"]["300"]["smooth"]["macro_recall"]), "threshold": STAGE_A_FIREWALL, "status": "STAGE_A_FIREWALL_PASSED" if float(a_report["evaluations"]["300"]["smooth"]["macro_recall"]) >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False},
        "B_TRUE_DEPTH_C1": {"R300": float(b_report["evaluations"]["300"]["smooth"]["macro_recall"]), "threshold": STAGE_A_FIREWALL, "status": "STAGE_A_FIREWALL_PASSED" if float(b_report["evaluations"]["300"]["smooth"]["macro_recall"]) >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False},
    }
    pairwise = {
        "Delta_A_R_A300_minus_C1": {"observed_delta": float(delta_a.mean()), "paired_bootstrap": {key: value for key, value in ci_a.items() if key != "delta_records"}},
        "Delta_depth_R_B300_minus_A300": {"observed_delta": float(delta_depth.mean()), "paired_bootstrap": {key: value for key, value in ci_depth.items() if key != "delta_records"}},
        "Delta_B_R_B300_minus_C1": {"observed_delta": float(delta_b.mean()), "paired_bootstrap": {key: value for key, value in ci_b.items() if key != "delta_records"}},
    }
    payload = {
        "schema": "lightgen_2d_true_optical_depth_comparison_aggregate.v1",
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "VALID_COMPLETED",
        "status": "valid_completed",
        "classification": classification["classification"],
        "outer_gate": OUTER_GATE,
        "design_authority": DESIGN_AUTHORITY,
        "contract": {"path": _relative(root, _path(root, CONTRACT_PATH)), "sha256": CONTRACT_SHA256},
        "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"]},
        "endpoint_authorities": {"source_C1": {"path": _relative(root, _path(root, C1_CHECKPOINT_PATH)), "sha256": C1_CHECKPOINT_SHA256}, "candidate_A": a_result, "candidate_B": b_result},
        "candidates": {"A_CONTINUED_C1": {"report": a_result["report"], "Smooth_R300": float(a_report["evaluations"]["300"]["smooth"]["macro_recall"]), "DEV_L_image_step300": float(a_report["reconstruction"]["DEV_L_image_step300"]), "checkpoint": a_report["checkpoints"]["300"]}, "B_TRUE_DEPTH_C1": {"report": b_result["report"], "Smooth_R300": float(b_report["evaluations"]["300"]["smooth"]["macro_recall"]), "DEV_L_image_step300": float(b_report["reconstruction"]["DEV_L_image_step300"]), "checkpoint": b_report["checkpoints"]["300"]}},
        "common_step0_equivalence": preflight["report"].get("step0_equivalence"),
        "pairwise": pairwise,
        "primary_gate": classification["primary_gate"],
        "stage_a": stage_a,
        "reconstruction": {"A": a_report["reconstruction"], "B": b_report["reconstruction"]},
        "cyclic_transfer": {"A": a_report["cyclic_transfer"], "B": b_report["cyclic_transfer"]},
        "raw_diversity": {"A": a_report["raw_diversity"], "B": b_report["raw_diversity"]},
        "diagnostics": {"A": a_report["diagnostics"], "B": b_report["diagnostics"]},
        "original_n4_aggregate": "UNRESOLVED",
        "claim_boundary": {"training_updates": 600, "optimizer_updates": 600, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False, "promotion": False},
        "governance": {"rank": 9, "candidate_A": "one continuation arm", "candidate_B": "one true-depth arm", "retries": False, "rank_sweep": False, "loss_change": False, "lambda_change": False, "lr_change": False, "encoder_update": False, "W_OLS_change": False, "stage_b": False, "generation": False, "protected": False, "test": False, "sealed": False, "promotion": False},
        "next": "N5 result is bounded; Stage B and generation remain closed regardless of the primary Gate",
    }
    report_path = output_dir / AGGREGATE_REPORT_NAME
    N4.atomic_write_new_json(report_path, payload, artifact_kind="N5 aggregate report")
    report_ref = {"path": _relative(root, report_path), "sha256": _sha256(report_path)}
    lineage_path = output_dir / AGGREGATE_LINEAGE_NAME
    N4.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_true_optical_depth_comparison_aggregate_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "report": report_ref, "claim_boundary": payload["claim_boundary"], "original_n4_aggregate_rewritten": False}, artifact_kind="N5 aggregate lineage")
    return {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _sha256(lineage_path)}, "pairwise": pairwise, "stage_a": stage_a, "original_n4_aggregate": "UNRESOLVED"}


def run_scientific(root: Path) -> dict[str, Any]:
    preflight = _load_preflight(root)
    _require(preflight["report"].get("result") == "AUTHORIZABLE", "N5 preflight is not authorizable")
    a_result = run_candidate(root, "A_CONTINUED_C1")
    if a_result["result"] != "VALID_COMPLETED":
        return {"result": a_result["result"], "status": a_result["status"], "classification": a_result["classification"], "candidate_A": a_result, "candidate_B": "not executed because symmetric comparison could not be completed", "outer_gate": OUTER_GATE}
    b_result = run_candidate(root, "B_TRUE_DEPTH_C1")
    if b_result["result"] != "VALID_COMPLETED":
        return {"result": b_result["result"], "status": b_result["status"], "classification": b_result["classification"], "candidate_A": a_result, "candidate_B": b_result, "outer_gate": OUTER_GATE}
    a_report = _read_json(_path(root, a_result["report"]["path"]), "N5 candidate A report")
    b_report = _read_json(_path(root, b_result["report"]["path"]), "N5 candidate B report")
    a_steps = {str(step): _load_step_sufficient(root, a_report, "A_CONTINUED_C1", step) for step in CHECKPOINT_STEPS}
    b_steps = {str(step): _load_step_sufficient(root, b_report, "B_TRUE_DEPTH_C1", step) for step in CHECKPOINT_STEPS}
    _require(all("dev_smooth_correct" in a_steps[str(step)] and "dev_smooth_correct" in b_steps[str(step)] for step in CHECKPOINT_STEPS), "N5 paired sufficient observations are incomplete")
    _, _, baseline, _ = _validate_n5_authorities(root)
    return _aggregate(root, _validate_contract(root), preflight, a_result, b_result, a_report, b_report, a_steps, b_steps, baseline["n4_sufficient"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific", "all"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    if args.action == "preflight":
        result = run_preflight(root)
    elif args.action == "scientific":
        result = run_scientific(root)
    else:
        result = run_preflight(root)
        if result["result"] != "AUTHORIZABLE":
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 2
        result = run_scientific(root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("result") in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
