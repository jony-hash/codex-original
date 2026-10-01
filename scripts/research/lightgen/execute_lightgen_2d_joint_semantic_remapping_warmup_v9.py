"""Execute the one-shot V9 joint semantic-remapping warmup.

The V9 arm keeps the V8 absolute-anchor plus cyclic-transfer objective and
training conditions fixed, while making exactly ``encoder.phase`` and
``generator.phase`` trainable.  This file reuses the established V8 runtime
and evaluator helpers but owns the V9 firewall, preflight, evidence protocol,
and one-shot scientific execution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import DeterministicTrainRoleSampler, load_v2_training_contract
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load


def _load_v8_execution() -> Any:
    path = Path(__file__).with_name("execute_lightgen_2d_absolute_anchor_plus_transfer_warmup_v8.py")
    spec = importlib.util.spec_from_file_location("lightgen_v8_execution_authority_v9", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V8 execution authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V8 = _load_v8_execution()
V6 = V8.V6
V7 = V8.V7

TASK_ID = "EXECUTE_LIGHTGEN_2D_JOINT_SEMANTIC_REMAPPING_WARMUP_V9"
OUTER_GATE = "V9_JOINT_SEMANTIC_REMAPPING_SUPPORTED"
SUPPORTED_CLASSIFICATION = OUTER_GATE
NOT_SUPPORTED_CLASSIFICATION = "V9_JOINT_SEMANTIC_REMAPPING_NOT_SUPPORTED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

CONTRACT_PATH = Path(
    "outputs/20260912_v9/runs/"
    "lightgen_2d_joint_semantic_remapping_warmup_contract_v9_attempt_01/"
    "lightgen_2d_joint_semantic_remapping_warmup_contract.v9.json"
)
# Write-once V9 contract identity.
CONTRACT_SHA256 = "4d0fb81bfa6152995aad4484c83487c20335616283097b394757f4616bf41fca"

DEFAULT_PREFLIGHT_DIR = Path(
    "outputs/20260912_v9/runs/"
    "lightgen_2d_joint_semantic_remapping_warmup_preflight_v9_attempt_01"
)
DEFAULT_SCIENTIFIC_DIR = Path(
    "outputs/20260912_v9/runs/"
    "lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_attempt_01"
)

PREFLIGHT_INPUT_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_preflight_input.v1.json"
PREFLIGHT_REPORT_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_preflight_lineage.v1.json"
PREFLIGHT_CHECKPOINT_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_preflight_smoke_checkpoint.v1.pt"
PREFLIGHT_SYNTHETIC_OBS_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_synthetic_sufficient_observations.v1.pt"
PREFLIGHT_SYNTHETIC_REPORT_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_synthetic_report.v1.json"
PREFLIGHT_SYNTHETIC_LINEAGE_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_synthetic_lineage.v1.json"

SCIENTIFIC_INPUT_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_input.v1.json"
SCIENTIFIC_REPORT_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_report.v1.json"
SCIENTIFIC_LINEAGE_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_lineage.v1.json"
SCIENTIFIC_SUFFICIENT_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_sufficient_observations.v1.pt"
SCIENTIFIC_RAW_NAME = "lightgen_2d_joint_semantic_remapping_warmup_v9_step300_cyclic_raw_outputs.v1.pt"

REPORT_SCHEMA = "lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_lineage.v1"
INPUT_SCHEMA = "lightgen_2d_joint_semantic_remapping_warmup_v9_scientific_input.v1"
SUFFICIENT_SCHEMA = "lightgen_2d_joint_semantic_remapping_warmup_v9_sufficient_observations.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_joint_semantic_remapping_warmup_v9_checkpoint.v1"

CLASS_COUNT = 10
LOGICAL_BATCH_SIZE = 10
PRODUCTION_GRID = 2048
SMOKE_GRID = 128
PRECHECK_STEPS = 3
SCIENTIFIC_STEPS = 300
CHECKPOINT_STEPS = (0, 100, 200, 300)
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
BOOTSTRAP_REPLICATES = 999
BOOTSTRAP_SEED = 271828
SMOKE_BOOTSTRAP_REPLICATES = 9
STAGE_A_FIREWALL = 0.937
SMOOTH_REQUIRED_DELTA = 0.05
SELECTION_ROWS = 1000
SELECTION_PER_CLASS = 100
SELECTION_SEED = 4321
SPLIT_SEED = 1234
S_TAU = 0.02919053753794475
LAMBDA_TRANSFER = 28.48464149920509
V8_G_ANCHOR = 0.0082287618885706
V8_G_TRANSFER = 0.00028888416448265975
V3_ENCODER_SHA256 = "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca"
V4_GENERATOR_TENSOR_SHA256 = "83aa74b701049c09c5efb63dbd6d3a6d181184a964d3d3c0c06665ca97da5afe"
PROTOTYPE_SHA256 = "c02a728e8631e24918b018553958a44b25cdb358192c6b4b28239f55e93df912"
SELECTION_INDICES_SHA256 = "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8"
SELECTION_LABELS_SHA256 = "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"


class InvalidV9(RuntimeError):
    """The V9 harness or immutable authority is invalid."""


class HardBlocked(RuntimeError):
    """A valid hard boundary prevents V9 execution."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidV9(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    _require(not value.is_absolute(), f"absolute path is not allowed: {value}")
    return (root / value).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


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
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _digest_tree(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, prefix: str) -> None:
        if torch.is_tensor(item):
            digest.update(prefix.encode("utf-8"))
            digest.update(V6.tensor_sha256(item).encode("ascii"))
        elif isinstance(item, Mapping):
            for key in sorted(item, key=str):
                visit(item[key], f"{prefix}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{prefix}[{index}]")
        else:
            digest.update(prefix.encode("utf-8"))
            digest.update(repr(item).encode("utf-8"))

    visit(value, "root")
    return digest.hexdigest()


def _finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _norm(value: torch.Tensor, name: str) -> float:
    _finite(value, name)
    result = float(value.detach().to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(result), f"{name} norm is nonfinite")
    return result


def _status_lines(root: Path) -> list[str]:
    return V6._status_lines(root)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidV9(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _validate_hash(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = V7.file_sha256(path)
    _require(observed == expected, f"{name} SHA mismatch: expected {expected}, observed {observed}")


def _load_contract(root: Path) -> tuple[dict[str, Any], Any]:
    _require(CONTRACT_SHA256 != "__TO_BE_BOUND_AFTER_CONTRACT_MATERIALIZATION__", "V9 contract SHA is not bound")
    path = _path(root, CONTRACT_PATH)
    _validate_hash(path, CONTRACT_SHA256, "V9 contract")
    payload = _read_json(path, "V9 contract")
    _require(
        payload.get("schema") == "lightgen_2d_joint_semantic_remapping_warmup_contract.v9"
        and payload.get("result") == "READY"
        and payload.get("status") == "valid_completed"
        and payload.get("outer_gate") == OUTER_GATE,
        "V9 contract is not READY",
    )
    definition = _mapping(payload.get("definition_entrypoint"), "V9 definition entrypoint")
    _validate_hash(_path(root, str(definition["path"])), str(definition["sha256"]), "V9 definition entrypoint")
    git = _mapping(payload.get("git"), "V9 contract Git")
    _require(git.get("branch") == EXPECTED_BRANCH and git.get("head") == EXPECTED_HEAD, "V9 contract Git identity changed")
    parent = _mapping(payload.get("parent_contract"), "V8 parent contract")
    _require(parent.get("path") == "outputs/20260911_v8/runs/lightgen_2d_absolute_anchor_plus_transfer_warmup_contract_v8_attempt_01/lightgen_2d_absolute_anchor_plus_transfer_warmup_contract.v8.json", "V8 parent path changed")
    _require(parent.get("sha256") == "02cef2918ff89eafaa53cf52f7a7492b3091e84acba3c351b6142b94b703a85d", "V8 parent SHA changed")
    v2_contract = load_v2_training_contract(_path(root, V6.V2_CONTRACT_PATH), repo_root=root)
    _require(v2_contract.sha256 == V6.V2_CONTRACT_SHA256, "V2 contract identity changed")
    objective = _mapping(payload.get("objective"), "V9 objective")
    _require(float(objective.get("s_tau", 0.0)) == S_TAU, "V9 s_tau changed")
    _require(float(objective.get("lambda_transfer", 0.0)) == LAMBDA_TRANSFER, "V9 lambda_transfer changed")
    protocol = _mapping(payload.get("protocol"), "V9 protocol")
    _require(protocol.get("trainable") == ["encoder.phase", "generator.phase"], "V9 trainable protocol changed")
    return payload, v2_contract


def _resolve_authorities(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    authorities = V8._resolve_authorities(root, contract)
    _require(tuple(authorities["initial_state"]["encoder.phase"].shape) == (2, 1, 100, 100), "V9 encoder shape changed")
    _require(tuple(authorities["initial_state"]["generator.phase"].shape) == (2, 1, 100, 100), "V9 generator shape changed")
    _require(V6.tensor_sha256(authorities["initial_state"]["encoder.phase"]) == V3_ENCODER_SHA256, "V9 initial encoder is not exact V3 CONTROL")
    _require(V6.tensor_sha256(authorities["initial_state"]["generator.phase"]) == V4_GENERATOR_TENSOR_SHA256, "V9 initial generator is not exact V4")
    _require(V6.tensor_sha256(authorities["prototypes"]) == PROTOTYPE_SHA256, "V9 prototypes changed")
    _require(not authorities["prototypes"].requires_grad and not authorities["templates"].requires_grad, "V9 support tensors require gradients")
    _require(math.isclose(float(authorities["s_tau"]), S_TAU, rel_tol=0.0, abs_tol=1.0e-15), "V9 s_tau recomputation changed")
    return authorities


def _new_consumer(v2_contract: Any, device: torch.device, grid: int) -> Any:
    return V8._new_consumer(v2_contract, device, grid)


def _load_runtime_state(consumer: Any, state: Mapping[str, Any]) -> None:
    V8._load_runtime_state(consumer, state)


def _configure_joint(consumer: Any) -> dict[str, Any]:
    named = dict(consumer.runtime.named_parameters())
    _require(set(named) == {"encoder.phase", "generator.phase"}, "runtime parameter surface changed")
    for parameter in named.values():
        parameter.requires_grad_(True)
        parameter.grad = None
    consumer.runtime.train()
    _require(all(parameter.requires_grad for parameter in named.values()), "joint trainable firewall failed")
    return {
        "trainable": ["encoder.phase", "generator.phase"],
        "frozen": ["OLS", "T=I_100", "W_OLS", "all remaining runtime parameters"],
        "trainable_count": sum(int(parameter.numel()) for parameter in named.values()),
        "trainable_parameter_counts": {name: int(parameter.numel()) for name, parameter in named.items()},
        "frozen_parameter_count": 0,
    }


def _optimizer(consumer: Any) -> torch.optim.Adam:
    named = dict(consumer.runtime.named_parameters())
    parameters = [named["encoder.phase"], named["generator.phase"]]
    _require(all(parameter.requires_grad for parameter in parameters), "joint optimizer received frozen parameter")
    optimizer = torch.optim.Adam(
        parameters,
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    _require(len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["params"] == parameters, "joint optimizer firewall changed")
    return optimizer


def _joint_frozen_state(consumer: Any) -> dict[str, Any]:
    return {
        key: _cpu_tree(value)
        for key, value in consumer.runtime.state_dict().items()
        if key not in {"encoder.phase", "generator.phase"}
    }


def _objective_components(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
    return V8._objective_components(*args, **kwargs)


def _initial_gradient_balance(
    bundle: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    batch_indices, batch_labels = sampler.next_batch()
    batch = images.index_select(0, batch_indices).to(device)
    batch_labels_device = batch_labels.to(device)
    norms: dict[str, dict[str, float]] = {}
    for objective_name in ("anchor", "transfer"):
        consumer = _new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
        _load_runtime_state(consumer, bundle["initial_state"])
        _configure_joint(consumer)
        named = dict(consumer.runtime.named_parameters())
        components = _objective_components(
            consumer,
            batch,
            batch_labels_device,
            bundle["prototypes"].to(device),
            bundle["templates"].to(device),
            float(bundle["s_tau"]),
            require_grad=True,
        )
        loss = components[f"{objective_name}_loss"]
        gradients = torch.autograd.grad(loss, [named["encoder.phase"], named["generator.phase"]], retain_graph=False, allow_unused=False)
        norms[objective_name] = {
            "encoder.phase": _norm(gradients[0], f"initial {objective_name} encoder gradient"),
            "generator.phase": _norm(gradients[1], f"initial {objective_name} generator gradient"),
        }
        _require(all(value > 0.0 for value in norms[objective_name].values()), f"initial {objective_name} joint gradient is zero")
        del consumer, components, gradients
    _require(math.isclose(norms["anchor"]["encoder.phase"], V8_G_ANCHOR, rel_tol=1.0e-12, abs_tol=1.0e-12), "V8 anchor gradient authority changed")
    _require(math.isclose(norms["transfer"]["encoder.phase"], V8_G_TRANSFER, rel_tol=1.0e-12, abs_tol=1.0e-12), "V8 transfer gradient authority changed")
    return {
        "first_batch_indices_sha256": V6.tensor_sha256(batch_indices),
        "first_batch_labels_sha256": V6.tensor_sha256(batch_labels),
        "batch_size": int(batch.shape[0]),
        "seed": SPLIT_SEED,
        "g_anchor": norms["anchor"]["encoder.phase"],
        "g_transfer": norms["transfer"]["encoder.phase"],
        "lambda_transfer": LAMBDA_TRANSFER,
        "formula": "reuse exact frozen V8 lambda_transfer; no V9 recalibration",
        "gradient_norms": norms,
        "encoder_gradient_authority_match": True,
    }


def _train_step(
    consumer: Any,
    evaluator: Any,
    optimizer: torch.optim.Adam,
    batch: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    templates: torch.Tensor,
    s_tau: float,
    lambda_transfer: float,
    step: int,
) -> dict[str, Any]:
    named = dict(consumer.runtime.named_parameters())
    before_frozen = _joint_frozen_state(consumer)
    before_encoder = named["encoder.phase"].detach().cpu().clone()
    before_generator = named["generator.phase"].detach().cpu().clone()
    optimizer.zero_grad(set_to_none=True)
    for parameter in named.values():
        parameter.grad = None
    components = _objective_components(consumer, batch, labels, prototypes, templates, s_tau, require_grad=True)
    with torch.no_grad():
        train_image_loss = (components["raw_base"].detach() - components["target"].detach()).square().mean()
        train_shift_image_loss = (components["raw_shift"].detach() - components["target"].detach()).square().mean()
    _finite(train_image_loss, "V9 evaluation-only TRAIN image loss")
    _finite(train_shift_image_loss, "V9 evaluation-only shifted TRAIN image loss")
    total_loss = components["anchor_loss"] + lambda_transfer * components["transfer_loss"]
    _finite(total_loss, "V9 combined loss")
    _require(total_loss.requires_grad and components["anchor_loss"].requires_grad and components["transfer_loss"].requires_grad, "V9 objective gradient path is detached")
    _require(not train_image_loss.requires_grad, "V9 image loss entered differentiated objective")
    total_loss.backward()
    encoder_gradient = named["encoder.phase"].grad
    generator_gradient = named["generator.phase"].grad
    encoder_gradient_norm = _norm(encoder_gradient, "V9 combined encoder gradient")
    generator_gradient_norm = _norm(generator_gradient, "V9 combined generator gradient")
    _require(encoder_gradient_norm > 0.0 and generator_gradient_norm > 0.0, "V9 combined joint gradient is zero")
    _require(all(parameter.grad is None for parameter in evaluator.parameters()), "V9 Smooth evaluator received a gradient")
    optimizer.step()
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    after_frozen = _joint_frozen_state(consumer)
    after_encoder = named["encoder.phase"].detach().cpu().clone()
    after_generator = named["generator.phase"].detach().cpu().clone()
    encoder_update = float((after_encoder - before_encoder).to(torch.float64).square().sum().sqrt())
    generator_update = float((after_generator - before_generator).to(torch.float64).square().sum().sqrt())
    _require(math.isfinite(encoder_update) and encoder_update > 0.0, "V9 encoder update is zero or nonfinite")
    _require(math.isfinite(generator_update) and generator_update > 0.0, "V9 generator update is zero or nonfinite")
    _require(_tree_equal(before_frozen, after_frozen), "V9 excluded optical/runtime state changed")
    return {
        "step": step,
        "objective": "L_V9=L_anchor+lambda_transfer*L_transfer",
        "L_anchor": float(components["anchor_loss"].detach().cpu()),
        "L_transfer": float(components["transfer_loss"].detach().cpu()),
        "L_total": float(total_loss.detach().cpu()),
        "L_image_evaluation_only": float(train_image_loss.detach().cpu()),
        "L_image_shift_evaluation_only": float(train_shift_image_loss.detach().cpu()),
        "Delta_template_mean": float(components["delta"].detach().to(torch.float64).mean().cpu()),
        "gradient_norms": {
            "encoder.phase": encoder_gradient_norm,
            "generator.phase": generator_gradient_norm,
            "combined_objective": {"encoder.phase": encoder_gradient_norm, "generator.phase": generator_gradient_norm},
            "image_loss_contribution": 0.0,
            "prototype_loss_contribution": 0.0,
            "smooth_evaluator_contribution": 0.0,
        },
        "update_norms": {"encoder.phase": encoder_update, "generator.phase": generator_update},
        "image_loss_in_differentiated_objective": False,
        "prototype_loss_in_differentiated_objective": False,
        "smooth_in_differentiated_objective": False,
        "generator_updated": True,
        "ols_unchanged": True,
        "trainable_parameter_names": ["encoder.phase", "generator.phase"],
        "frozen_parameter_names": ["OLS", "T=I_100", "W_OLS", "TRAIN prototypes", "TRAIN templates", "Smooth evaluator"],
        "image_loss_gradient": False,
        "prototype_loss_gradient": False,
        "smooth_evaluator_gradient": False,
        "runtime_state_sha256_after": _digest_tree(_cpu_tree(consumer.runtime.state_dict())),
    }


def _measure_train_batch(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return V8._measure_train_batch(*args, **kwargs)


def _checkpoint_payload(
    consumer: Any,
    optimizer: torch.optim.Adam,
    sampler: Any,
    *,
    step: int,
    initial_state_digest: str,
    prototypes: torch.Tensor,
    templates: torch.Tensor,
    rng_state: Mapping[str, Any],
    s_tau: float,
    gradient_balance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        "checkpoint_step": step,
        "runtime_state": _cpu_tree(consumer.runtime.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "sampler_state": _cpu_tree(sampler.state_dict()),
        "rng_state": _cpu_tree(rng_state),
        "initial_runtime_state_sha256": initial_state_digest,
        "prototypes_sha256": V6.tensor_sha256(prototypes),
        "templates_sha256": V6.tensor_sha256(templates),
        "s_tau": s_tau,
        "g_anchor": gradient_balance["g_anchor"],
        "g_transfer": gradient_balance["g_transfer"],
        "lambda_transfer": gradient_balance["lambda_transfer"],
        "objective": "L_V9=L_anchor+lambda_transfer*L_transfer",
        "trainable_parameter_names": ["encoder.phase", "generator.phase"],
        "frozen_parameter_names": ["OLS", "T=I_100", "W_OLS", "TRAIN prototypes", "TRAIN templates", "Smooth evaluator"],
        "image_loss_gradient_contribution": 0.0,
        "prototype_loss_gradient_contribution": 0.0,
        "smooth_evaluator_gradient_contribution": 0.0,
        "data_roles": {"TRAIN": True, "DEV": True, "calibration": False, "protected": False, "official_test": False, "sealed": False},
        "stage_b_executed": False,
        "scale_extraction_executed": False,
        "prior_shaping_executed": False,
        "generation_executed": False,
        "formal_promotion": False,
    }


def _save_and_verify_checkpoint(
    root: Path,
    target: Path,
    payload: Mapping[str, Any],
    name: str,
    consumer: Any,
    v2_contract: Any,
) -> dict[str, Any]:
    path = target / name
    atomic_torch_save_new(path, payload, artifact_kind="V9 checkpoint")
    digest = V7.file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V9 checkpoint reload")
    _require(loaded.get("schema") == CHECKPOINT_SCHEMA and loaded.get("checkpoint_step") == payload.get("checkpoint_step"), "V9 checkpoint schema/step reload changed")
    _require(loaded.get("prototypes_sha256") == PROTOTYPE_SHA256 and loaded.get("templates_sha256") == V6.tensor_sha256(payload.get("templates", torch.empty(0))), "V9 checkpoint support identity changed")
    clone = _new_consumer(v2_contract, consumer.device, consumer.runtime.config.propagation_grid_size)
    _load_runtime_state(clone, _mapping(loaded.get("runtime_state"), "V9 checkpoint runtime state"))
    _configure_joint(clone)
    _require(_digest_tree(_cpu_tree(clone.runtime.state_dict())) == _digest_tree(_cpu_tree(loaded["runtime_state"])), "V9 checkpoint runtime reload changed")
    clone_optimizer = _optimizer(clone)
    clone_optimizer.load_state_dict(dict(_mapping(loaded.get("optimizer_state"), "V9 checkpoint optimizer state")))
    _require(_tree_equal(_cpu_tree(clone_optimizer.state_dict()), loaded["optimizer_state"]), "V9 checkpoint optimizer reload changed")
    current_rng = V6._capture_rng()
    V6._restore_rng(_mapping(loaded.get("rng_state"), "V9 checkpoint RNG state"))
    reloaded_rng = V6._capture_rng()
    V6._restore_rng(current_rng)
    _require(_tree_equal(reloaded_rng, loaded["rng_state"]), "V9 checkpoint RNG reload changed")
    _require(float(loaded["s_tau"]) == float(payload["s_tau"]) and float(loaded["lambda_transfer"]) == float(payload["lambda_transfer"]), "V9 checkpoint objective constants changed")
    _require(loaded.get("trainable_parameter_names") == ["encoder.phase", "generator.phase"], "V9 checkpoint trainable identity changed")
    return {"path": _relative(root, path), "sha256": digest, "step": int(payload["checkpoint_step"]), "reloaded": True, "resume_identity": True}


def _gradient_firewall_probe(
    bundle: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    consumer = _new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    _load_runtime_state(consumer, bundle["initial_state"])
    firewall = _configure_joint(consumer)
    indices = train_indices[:LOGICAL_BATCH_SIZE]
    batch = images.index_select(0, indices).to(device)
    batch_labels = labels.index_select(0, indices).to(device)
    named = dict(consumer.runtime.named_parameters())
    components = _objective_components(consumer, batch, batch_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], require_grad=True)
    with torch.no_grad():
        image_loss = (components["raw_base"].detach() - components["target"].detach()).square().mean()
    _require(not image_loss.requires_grad, "V9 image loss has a gradient path")
    wrong_anchor = F.cross_entropy(components["anchor_logits"], (batch_labels + 1) % CLASS_COUNT)
    wrong_anchor_grad = torch.autograd.grad(wrong_anchor, [named["encoder.phase"], named["generator.phase"]], retain_graph=True)
    wrong_transfer = F.softplus(components["delta"]).mean()
    wrong_transfer_grad = torch.autograd.grad(wrong_transfer, [named["encoder.phase"], named["generator.phase"]], retain_graph=True)
    total = components["anchor_loss"] + LAMBDA_TRANSFER * components["transfer_loss"]
    total.backward()
    combined = {
        "encoder.phase": _norm(named["encoder.phase"].grad, "V9 combined probe encoder gradient"),
        "generator.phase": _norm(named["generator.phase"].grad, "V9 combined probe generator gradient"),
    }
    negative = {
        "wrong_label_anchor": {
            "encoder.phase": _norm(wrong_anchor_grad[0], "V9 wrong-label encoder gradient"),
            "generator.phase": _norm(wrong_anchor_grad[1], "V9 wrong-label generator gradient"),
        },
        "wrong_direction_transfer": {
            "encoder.phase": _norm(wrong_transfer_grad[0], "V9 wrong-direction encoder gradient"),
            "generator.phase": _norm(wrong_transfer_grad[1], "V9 wrong-direction generator gradient"),
        },
    }
    _require(all(value > 0.0 for value in combined.values()), "V9 combined joint gradient is zero")
    _require(all(value > 0.0 for item in negative.values() for value in item.values()), "V9 negative-control gradient is zero")
    _require(all(parameter.grad is None for parameter in bundle["smooth"].parameters()), "V9 Smooth evaluator firewall failed")
    _require(not bundle["prototypes"].requires_grad and not bundle["templates"].requires_grad, "V9 support tensor firewall failed")
    return {
        "firewall": firewall,
        "objective": "L_V9=L_anchor+lambda_transfer*L_transfer",
        "image_loss_requires_grad": False,
        "image_loss_gradient_contribution": 0.0,
        "prototype_loss_gradient_contribution": 0.0,
        "smooth_evaluator_gradient_contribution": 0.0,
        "combined_gradient_norms": combined,
        "initial_gradient_norms": bundle["gradient_norms"],
        "negative_controls": negative,
        "generator_gradient": combined["generator.phase"],
        "prototype_loss_in_objective": False,
        "smooth_in_objective": False,
    }


def _three_step_smoke(
    root: Path,
    bundle: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
    target: Path,
) -> dict[str, Any]:
    consumer = _new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    _load_runtime_state(consumer, bundle["initial_state"])
    firewall = _configure_joint(consumer)
    optimizer = _optimizer(consumer)
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    traces = []
    for step in range(1, PRECHECK_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        batch = images.index_select(0, indices).to(device)
        traces.append(_train_step(consumer, bundle["smooth"], optimizer, batch, batch_labels.to(device), bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], bundle["lambda_transfer"], step))
    payload = _checkpoint_payload(
        consumer,
        optimizer,
        sampler,
        step=PRECHECK_STEPS,
        initial_state_digest=_digest_tree(bundle["initial_state"]),
        prototypes=bundle["prototypes"],
        templates=bundle["templates"],
        rng_state=V6._capture_rng(),
        s_tau=bundle["s_tau"],
        gradient_balance=bundle,
    )
    payload["templates"] = bundle["templates"]
    checkpoint = _save_and_verify_checkpoint(root, target, payload, PREFLIGHT_CHECKPOINT_NAME, consumer, bundle["v2_contract"])
    loaded = _mapping(safe_torch_load(target / PREFLIGHT_CHECKPOINT_NAME, expected_sha256=checkpoint["sha256"], map_location="cpu"), "V9 smoke checkpoint")
    resumed_sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    resumed_sampler.load_state_dict(dict(_mapping(loaded["sampler_state"], "V9 smoke sampler state")))
    next_original, next_original_labels = sampler.next_batch()
    next_resumed, next_resumed_labels = resumed_sampler.next_batch()
    _require(torch.equal(next_original, next_resumed) and torch.equal(next_original_labels, next_resumed_labels), "V9 sampler resume identity changed")
    return {
        "firewall": firewall,
        "traces": traces,
        "checkpoint": checkpoint,
        "resume_identity": {"runtime": True, "optimizer": True, "rng": True, "sampler_next_batch": True},
        "scientific_updates": 0,
        "scientific_meaning": "none",
    }


def _save_sufficient(root: Path, target: Path, observations: Mapping[str, Any], *, name: str, schema: str) -> dict[str, Any]:
    path = target / name
    payload = {
        "schema": schema,
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        **{key: value.cpu() if torch.is_tensor(value) else value for key, value in observations.items()},
    }
    atomic_torch_save_new(path, payload, artifact_kind="V9 sufficient observations")
    digest = V7.file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V9 sufficient observations reload")
    _require(loaded.get("schema") == schema, "V9 sufficient observation schema reload changed")
    for key, value in observations.items():
        if torch.is_tensor(value):
            _require(torch.equal(loaded[key], value.cpu()), f"V9 sufficient observation reload changed: {key}")
    return {"path": _relative(root, path), "sha256": digest, "schema": schema, "reloaded": True}


def _save_cyclic_raw(root: Path, target: Path, cyclic: Mapping[str, Any]) -> dict[str, Any]:
    observations = cyclic["_observations"]
    payload = {
        "schema": "lightgen_2d_joint_semantic_remapping_warmup_v9_cyclic_raw_outputs.v1",
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        **{key: value.cpu() if torch.is_tensor(value) else value for key, value in observations.items()},
    }
    path = target / SCIENTIFIC_RAW_NAME
    atomic_torch_save_new(path, payload, artifact_kind="V9 cyclic raw outputs")
    digest = V7.file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V9 cyclic raw reload")
    _require(loaded.get("schema") == payload["schema"] and torch.equal(loaded["selection_indices"], observations["selection_indices"].cpu()), "V9 cyclic raw reload changed")
    return {"path": _relative(root, path), "sha256": digest, "reloaded": True}


def _classify(r0: float, r300: float, smooth_bootstrap: Mapping[str, Any], cyclic: Mapping[str, Any], raw_valid: bool) -> dict[str, Any]:
    natural_pass = r300 - r0 >= SMOOTH_REQUIRED_DELTA and float(smooth_bootstrap["ci_lower"]) > 0.0
    cyclic_pass = bool(cyclic["known_cyclic_transfer_retained"])
    supported = bool(natural_pass and cyclic_pass and raw_valid)
    return {
        "classification": SUPPORTED_CLASSIFICATION if supported else NOT_SUPPORTED_CLASSIFICATION,
        "outer_gate": OUTER_GATE,
        "supported": supported,
        "criteria": {
            "natural_smooth_improvement": natural_pass,
            "known_cyclic_transfer_retained": cyclic_pass,
            "raw_validity": raw_valid,
        },
        "thresholds": {
            "smooth_delta": SMOOTH_REQUIRED_DELTA,
            "smooth_ci_lower": 0.0,
        },
    }


def _synthetic_acceptance(root: Path, bundle: Mapping[str, Any], device: torch.device, target: Path) -> dict[str, Any]:
    ramp = torch.linspace(0.0, 1.0, 28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)
    train_labels = torch.arange(CLASS_COUNT, dtype=torch.long)
    train_images = (ramp + train_labels.view(-1, 1, 1, 1).to(torch.float32) / 20.0).clamp(0.0, 1.0)
    eval_labels = train_labels.repeat_interleave(2)
    eval_images = train_images.index_select(0, eval_labels)
    consumer = _new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    _load_runtime_state(consumer, bundle["initial_state"])
    firewall = _configure_joint(consumer)
    optimizer = _optimizer(consumer)
    traces = []
    for step in range(1, PRECHECK_STEPS + 1):
        traces.append(_train_step(consumer, bundle["smooth"], optimizer, train_images.to(device), train_labels.to(device), bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], bundle["lambda_transfer"], step))
    evaluation = V8._evaluate_v8(consumer, bundle["smooth"], eval_images.to(device), eval_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], step=PRECHECK_STEPS, train_trace=traces[-1])
    cyclic = V6._cyclic_transfer(consumer, bundle["smooth"], eval_images.to(device), eval_labels, torch.arange(eval_images.shape[0], dtype=torch.long), bundle["prototypes"], bundle["templates"])
    observations = {
        "synthetic_anchor_distances": evaluation["_anchor_distances"],
        "synthetic_anchor_predictions": evaluation["_anchor_predictions"],
        "synthetic_anchor_margin": evaluation["_anchor_correct_margin"],
        "synthetic_ols_margin": evaluation["_margins"],
        "synthetic_smooth_logits": evaluation["_smooth_logits"],
        **cyclic["_observations"],
    }
    sufficient = _save_sufficient(root, target, observations, name=PREFLIGHT_SYNTHETIC_OBS_NAME, schema="lightgen_2d_joint_semantic_remapping_warmup_v9_synthetic_sufficient_observations.v1")
    synthetic_smooth = V6._paired_bootstrap(torch.zeros_like(eval_labels, dtype=torch.float64), eval_labels, replicates=SMOKE_BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    decision = _classify(0.0, 0.0, synthetic_smooth, cyclic, True)
    report_payload = {
        "schema": "lightgen_2d_joint_semantic_remapping_warmup_v9_synthetic_report.v1",
        "scientific_meaning": "none",
        "training_updates": PRECHECK_STEPS,
        "objective": "L_V9=L_anchor+lambda_transfer*L_transfer",
        "s_tau": bundle["s_tau"],
        "lambda_transfer": bundle["lambda_transfer"],
        "firewall": firewall,
        "traces": traces,
        "evaluation": {key: value for key, value in evaluation.items() if not key.startswith("_")},
        "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")},
        "collapse_diagnostics": evaluation["collapse"],
        "bootstrap": synthetic_smooth,
        "decision": decision,
        "sufficient_observations": sufficient,
    }
    report_path = target / PREFLIGHT_SYNTHETIC_REPORT_NAME
    atomic_write_new_json(report_path, report_payload, artifact_kind="V9 synthetic report")
    report_ref = {"path": _relative(root, report_path), "sha256": V7.file_sha256(report_path)}
    lineage_payload = {
        "schema": "lightgen_2d_joint_semantic_remapping_warmup_v9_synthetic_lineage.v1",
        "result": "VALID_COMPLETED",
        "report": report_ref,
        "sufficient_observations": sufficient,
        "claim_boundary": {"scientific_meaning": "none", "protected": False, "official_test": False, "sealed": False},
    }
    lineage_path = target / PREFLIGHT_SYNTHETIC_LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage_payload, artifact_kind="V9 synthetic lineage")
    _require(_read_json(report_path, "V9 synthetic report reload")["decision"] == decision, "V9 synthetic report reload changed")
    _require(_read_json(lineage_path, "V9 synthetic lineage reload")["report"]["sha256"] == report_ref["sha256"], "V9 synthetic lineage reload changed")
    return {
        "scientific_meaning": "none",
        "firewall": firewall,
        "traces": traces,
        "evaluation": {key: value for key, value in evaluation.items() if not key.startswith("_")},
        "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")},
        "collapse_diagnostics": evaluation["collapse"],
        "bootstrap": synthetic_smooth,
        "decision": decision,
        "sufficient_observations": sufficient,
        "report": report_ref,
        "lineage": {"path": _relative(root, lineage_path), "sha256": V7.file_sha256(lineage_path)},
        "report_lineage_reloaded": True,
    }


def _preflight_input_payload(contract_ref: Mapping[str, Any], device_facts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "lightgen_2d_joint_semantic_remapping_warmup_v9_preflight_input.v1",
        "version": 1,
        "task_id": TASK_ID,
        "contract": dict(contract_ref),
        "objective": "L_V9=L_anchor+lambda_transfer*L_transfer",
        "trainable_parameter_names": ["encoder.phase", "generator.phase"],
        "frozen_gradient_paths": ["L_image", "L_proto", "Smooth/Frozen evaluator"],
        "runtime": {"device": dict(device_facts), "grid": PRODUCTION_GRID},
        "one_shot": True,
        "scientific_300_step_executed": False,
        "protected": False,
        "official_test": False,
        "sealed": False,
    }


def run_preflight(root: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to overwrite V9 preflight output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V6.git_facts(root)
    status_before = _status_lines(root)
    payload: dict[str, Any] = {
        "schema": "lightgen_2d_joint_semantic_remapping_warmup_v9_preflight_report.v1",
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "INVALID",
        "status": "invalid_harness",
        "classification": "INVALID_V9_PREFLIGHT",
        "outer_gate": OUTER_GATE,
        "provenance": {"git_start": started},
        "claim_boundary": {"engineering_preflight_executed": False, "scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False},
    }
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from V9 authorization")
        contract, v2_contract = _load_contract(root)
        bundle = _resolve_authorities(root, contract)
        bundle["v2_contract"] = v2_contract
        dll = V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V6._cuda_facts()
        bundle["smooth"].to(device)
        images, labels, train_indices, _development_indices, dataset = V6._load_roles(root)
        V6._initial_rng()
        gradient_balance = _initial_gradient_balance(bundle, images, labels, train_indices, device)
        bundle.update(gradient_balance)
        firewall_probe = _gradient_firewall_probe(bundle, images, labels, train_indices, device)
        no_data = V6._production_no_data_smoke(bundle, device)
        smoke = _three_step_smoke(root, bundle, images, labels, train_indices, device, output_dir)
        synthetic = _synthetic_acceptance(root, bundle, device, output_dir)
        contract_ref = {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}
        input_payload = _preflight_input_payload(contract_ref, device_facts)
        input_path = output_dir / PREFLIGHT_INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V9 preflight input")
        payload.update({
            "contract": contract_ref,
            "authorities": bundle["authorities"],
            "runtime": {"cuda_dll_search_path": dll, "device": device_facts, "propagation_grid": PRODUCTION_GRID},
            "data_roles": dataset,
            "s_tau": bundle["s_tau"],
            "lambda_transfer": bundle["lambda_transfer"],
            "gradient_balance": gradient_balance,
            "firewall_probe": firewall_probe,
            "production_no_data_smoke": no_data,
            "three_step_smoke": smoke,
            "synthetic_acceptance": synthetic,
            "input": {"path": _relative(root, input_path), "sha256": V7.file_sha256(input_path)},
            "result": "AUTHORIZABLE",
            "status": "valid_completed",
            "classification": "LIGHTGEN_2D_JOINT_SEMANTIC_REMAPPING_WARMUP_V9_AUTHORIZABLE",
            "claim_boundary": {"engineering_preflight_executed": True, "scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False},
            "next": "proceed directly to the one-shot 300-step V9 scientific execution",
        })
    except HardBlocked as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": error.classification, "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific execution"})
    except (V6.HardBlocked, V7.HardBlocked) as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": "V9_HARD_BLOCKED", "hard_blocker": {"stage": getattr(error, "stage", "runtime"), "message": str(error)}, "next": "stop; no scientific execution"})
    except (InvalidV9, V8.InvalidV8, V7.InvalidV7, V6.InvalidV6) as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V9_PREFLIGHT_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no scientific execution"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V9_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no scientific execution"})
    finally:
        ending = V6.git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == _status_lines(root)})
    report_path = output_dir / PREFLIGHT_REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V9 preflight report")
    report_ref = {"path": _relative(root, report_path), "sha256": V7.file_sha256(report_path)}
    lineage = {"schema": "lightgen_2d_joint_semantic_remapping_warmup_v9_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "provenance": payload["provenance"], "report": report_ref, "claim_boundary": payload["claim_boundary"]}
    lineage_path = output_dir / PREFLIGHT_LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V9 preflight lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": V7.file_sha256(lineage_path)}, "scientific_300_step_executed": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _validate_preflight(root: Path) -> dict[str, Any]:
    report_path = _path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = _path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    _require(report_path.is_file() and lineage_path.is_file(), "V9 preflight authority is unavailable")
    report = _read_json(report_path, "V9 preflight report")
    lineage = _read_json(lineage_path, "V9 preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed" and report.get("classification") == "LIGHTGEN_2D_JOINT_SEMANTIC_REMAPPING_WARMUP_V9_AUTHORIZABLE", "V9 preflight is not AUTHORIZABLE")
    _require(lineage.get("result") == "AUTHORIZABLE" and lineage.get("classification") == report.get("classification"), "V9 preflight lineage changed")
    _require(report.get("claim_boundary", {}).get("scientific_300_step_executed") is False, "V9 preflight opened scientific execution")
    _require(float(report.get("s_tau", 0.0)) == S_TAU and float(report.get("lambda_transfer", 0.0)) == LAMBDA_TRANSFER, "V9 preflight objective constants changed")
    _require(report.get("gradient_balance", {}).get("gradient_norms", {}).get("anchor", {}).get("generator.phase", 0.0) > 0.0, "V9 preflight generator anchor gradient missing")
    _require(report.get("three_step_smoke", {}).get("resume_identity", {}).get("sampler_next_batch") is True, "V9 preflight resume identity missing")
    return {"report": report, "report_ref": {"path": _relative(root, report_path), "sha256": V7.file_sha256(report_path)}, "lineage_ref": {"path": _relative(root, lineage_path), "sha256": V7.file_sha256(lineage_path)}}


def run_scientific(root: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to overwrite V9 scientific output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V6.git_facts(root)
    status_before = _status_lines(root)
    payload: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "INVALID",
        "status": "invalid_harness",
        "classification": "INVALID_V9_SCIENTIFIC_HARNESS",
        "outer_gate": OUTER_GATE,
        "provenance": {"git_start": started},
        "claim_boundary": {"scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_shaping_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False},
        "governance": {"retry": False, "second_arm": False, "stage_b": False, "generation": False, "formal_promotion": False},
    }
    scientific_started = False
    updates_completed = 0
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from V9 authorization")
        contract, v2_contract = _load_contract(root)
        preflight = _validate_preflight(root)
        bundle = _resolve_authorities(root, contract)
        bundle["v2_contract"] = v2_contract
        dll = V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V6._cuda_facts()
        bundle["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = V6._load_roles(root)
        development_images = images.index_select(0, development_indices).contiguous()
        development_labels = labels.index_select(0, development_indices).contiguous()
        selection_images, selection_labels, selection_indices = V6._load_selection(root, images, labels, development_indices)
        V6._initial_rng()
        gradient_balance = _initial_gradient_balance(bundle, images, labels, train_indices, device)
        preflight_balance = preflight["report"]["gradient_balance"]
        for key in ("g_anchor", "g_transfer", "lambda_transfer"):
            _require(math.isclose(float(gradient_balance[key]), float(preflight_balance[key]), rel_tol=1.0e-12, abs_tol=1.0e-12), f"V9 {key} changed between preflight and scientific entry")
        bundle.update(gradient_balance)
        consumer = _new_consumer(v2_contract, device, PRODUCTION_GRID)
        _load_runtime_state(consumer, bundle["initial_state"])
        firewall = _configure_joint(consumer)
        optimizer = _optimizer(consumer)
        sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
        initial_state_digest = _digest_tree(bundle["initial_state"])
        metric_sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
        metric_indices, metric_labels = metric_sampler.next_batch()
        initial_train_metrics = _measure_train_batch(consumer, images.index_select(0, metric_indices).to(device), metric_labels.to(device), bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], bundle["lambda_transfer"])
        contract_ref = {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}
        payload.update({
            "contract": contract_ref,
            "preflight": {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"], "result": "AUTHORIZABLE"},
            "authorities": bundle["authorities"],
            "runtime": {"cuda_dll_search_path": dll, "device": device_facts, "propagation_grid": PRODUCTION_GRID, "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "trainable_parameter_names": ["encoder.phase", "generator.phase"]}},
            "data_roles": dataset,
            "initial_state": {"combined_runtime_state_sha256": initial_state_digest, "encoder_phase_sha256": V3_ENCODER_SHA256, "generator_phase_sha256": V4_GENERATOR_TENSOR_SHA256, "rng_sha256": _digest_tree(V6._capture_rng())},
            "firewall": firewall,
            "objective": {"formula": "L_V9=L_anchor+lambda_transfer*L_transfer", "s_tau": bundle["s_tau"], "s_tau_summary": bundle["s_tau_summary"], "g_anchor": bundle["g_anchor"], "g_transfer": bundle["g_transfer"], "lambda_transfer": bundle["lambda_transfer"], "lambda_rule": "reuse exact frozen V8 lambda_transfer", "image_loss_gradient_contribution": 0.0, "prototype_loss_gradient_contribution": 0.0, "smooth_evaluator_gradient_contribution": 0.0, "image_loss_role": "evaluation-only", "template_anchor_role": "differentiated natural output objective", "transfer_role": "differentiated cyclic latent-shift objective"},
            "selection": {"rows": SELECTION_ROWS, "per_class": SELECTION_PER_CLASS, "seed": SELECTION_SEED, "indices_sha256": SELECTION_INDICES_SHA256, "labels_sha256": SELECTION_LABELS_SHA256},
        })
        input_payload = {"schema": INPUT_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "contract": contract_ref, "preflight": payload["preflight"], "protocol": {"objective": "L_V9=L_anchor+lambda_transfer*L_transfer", "s_tau": bundle["s_tau"], "lambda_transfer": bundle["lambda_transfer"], "steps": SCIENTIFIC_STEPS, "checkpoints": list(CHECKPOINT_STEPS), "batch": LOGICAL_BATCH_SIZE, "grid": PRODUCTION_GRID, "optimizer": payload["runtime"]["optimizer"], "trainable": ["encoder.phase", "generator.phase"], "latent_shift": "z_shift=z-mu_y+mu_t", "cyclic_target": "t=(y+1) mod 10", "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}}, "data_roles": {"TRAIN": 55000, "DEV": 5000, "calibration": 0, "protected": 0, "official_test": 0, "sealed": 0}, "claim_boundary": payload["claim_boundary"]}
        input_path = output_dir / SCIENTIFIC_INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V9 scientific input")
        payload["input"] = {"path": _relative(root, input_path), "sha256": V7.file_sha256(input_path)}
        scientific_started = True
        evaluations: dict[str, dict[str, Any]] = {}
        checkpoints: dict[str, dict[str, Any]] = {}
        step0 = V8._evaluate_v8(consumer, bundle["smooth"], development_images, development_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], step=0, train_trace=None)
        checkpoint_payload = _checkpoint_payload(consumer, optimizer, sampler, step=0, initial_state_digest=initial_state_digest, prototypes=bundle["prototypes"], templates=bundle["templates"], rng_state=V6._capture_rng(), s_tau=bundle["s_tau"], gradient_balance=bundle)
        checkpoint_payload["templates"] = bundle["templates"]
        checkpoints["0"] = _save_and_verify_checkpoint(root, output_dir, checkpoint_payload, "lightgen_2d_joint_semantic_remapping_warmup_v9_step_0_checkpoint.v1.pt", consumer, v2_contract)
        evaluations["0"] = step0
        traces: list[dict[str, Any]] = []
        train_metrics = {"0": initial_train_metrics}
        started_training = time.perf_counter()
        for step in range(1, SCIENTIFIC_STEPS + 1):
            indices, batch_labels = sampler.next_batch()
            batch = images.index_select(0, indices).to(device)
            trace = _train_step(consumer, bundle["smooth"], optimizer, batch, batch_labels.to(device), bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], bundle["lambda_transfer"], step)
            traces.append(trace)
            updates_completed = step
            del batch
            if step in CHECKPOINT_STEPS[1:]:
                checkpoint_payload = _checkpoint_payload(consumer, optimizer, sampler, step=step, initial_state_digest=initial_state_digest, prototypes=bundle["prototypes"], templates=bundle["templates"], rng_state=V6._capture_rng(), s_tau=bundle["s_tau"], gradient_balance=bundle)
                checkpoint_payload["templates"] = bundle["templates"]
                checkpoints[str(step)] = _save_and_verify_checkpoint(root, output_dir, checkpoint_payload, f"lightgen_2d_joint_semantic_remapping_warmup_v9_step_{step}_checkpoint.v1.pt", consumer, v2_contract)
                evaluations[str(step)] = V8._evaluate_v8(consumer, bundle["smooth"], development_images, development_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], step=step, train_trace=trace)
                train_metrics[str(step)] = {key: trace[key] for key in ("L_anchor", "L_transfer", "L_total", "L_image_evaluation_only", "L_image_shift_evaluation_only", "Delta_template_mean")}
            if step % 25 == 0:
                print(json.dumps({"event": "scientific_progress", "step": step, "elapsed_seconds": time.perf_counter() - started_training}, ensure_ascii=False), flush=True)
        _require(set(evaluations) == {"0", "100", "200", "300"} and set(checkpoints) == {"0", "100", "200", "300"}, "V9 scientific checkpoint/evaluation set is incomplete")
        step0_record = evaluations["0"]
        step300_record = evaluations["300"]
        smooth_correct0 = step0_record["_smooth_logits"].argmax(dim=1).eq(development_labels)
        smooth_correct300 = step300_record["_smooth_logits"].argmax(dim=1).eq(development_labels)
        smooth_bootstrap = V6._paired_bootstrap(smooth_correct300.to(torch.float64) - smooth_correct0.to(torch.float64), development_labels)
        ols_bootstrap = V6._paired_bootstrap(step300_record["_margins"] - step0_record["_margins"], development_labels)
        cyclic = V6._cyclic_transfer(consumer, bundle["smooth"], selection_images, selection_labels, selection_indices, bundle["prototypes"], bundle["templates"])
        sufficient_obs = {"step0_smooth_logits": step0_record["_smooth_logits"], "step300_smooth_logits": step300_record["_smooth_logits"], "step0_ols_margin": step0_record["_margins"], "step300_ols_margin": step300_record["_margins"], "step0_anchor_distances": step0_record["_anchor_distances"], "step300_anchor_distances": step300_record["_anchor_distances"], "step0_anchor_predictions": step0_record["_anchor_predictions"], "step300_anchor_predictions": step300_record["_anchor_predictions"], "step0_anchor_correct_margin": step0_record["_anchor_correct_margin"], "step300_anchor_correct_margin": step300_record["_anchor_correct_margin"], **cyclic["_observations"]}
        sufficient_ref = _save_sufficient(root, output_dir, sufficient_obs, name=SCIENTIFIC_SUFFICIENT_NAME, schema=SUFFICIENT_SCHEMA)
        cyclic_raw_ref = _save_cyclic_raw(root, output_dir, cyclic)
        raw_valid = all(bool(evaluations[str(step)]["raw_validity"]["finite"] and evaluations[str(step)]["raw_validity"]["structured"] and evaluations[str(step)]["raw_validity"]["nonconstant"]) for step in CHECKPOINT_STEPS) and bool(cyclic["raw_validity"]["base"]["finite"] and cyclic["raw_validity"]["shifted"]["finite"])
        r0 = float(step0_record["smooth"]["macro_recall"])
        r300 = float(step300_record["smooth"]["macro_recall"])
        decision = _classify(r0, r300, smooth_bootstrap, cyclic, raw_valid)
        stage_firewall = {"metric": "step-300 DEV Smooth macro recall", "threshold": STAGE_A_FIREWALL, "R300": r300, "status": "STAGE_A_FIREWALL_PASSED" if r300 >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False}
        public_evaluations = {step: {key: value for key, value in record.items() if not key.startswith("_")} for step, record in evaluations.items()}
        payload.update({
            "result": "VALID_COMPLETED",
            "status": "valid_completed",
            "classification": decision["classification"],
            "evaluations": public_evaluations,
            "train_metrics": train_metrics,
            "traces": traces,
            "checkpoints": checkpoints,
            "scientific_updates": SCIENTIFIC_STEPS,
            "runtime_device": device_facts,
            "step300_minus_step0": {"Delta_R_Smooth": smooth_bootstrap, "Delta_R_observed": r300 - r0, "Delta_M_OLS": ols_bootstrap, "Delta_L_image_DEV": float(step300_record["losses"]["L_image"] - step0_record["losses"]["L_image"]), "Delta_L_anchor_TRAIN": train_metrics["300"]["L_anchor"] - train_metrics["0"]["L_anchor"], "Delta_L_transfer_TRAIN": train_metrics["300"]["L_transfer"] - train_metrics["0"]["L_transfer"], "Delta_L_total_TRAIN": train_metrics["300"]["L_total"] - train_metrics["0"]["L_total"]},
            "natural_smooth_semantics": {"R0": r0, "R300": r300, "Delta_R": r300 - r0, "bootstrap": smooth_bootstrap, "step0_per_class": step0_record["smooth"]["per_class_recall"], "step300_per_class": step300_record["smooth"]["per_class_recall"]},
            "ols_prototype_margin": {"step0": step0_record["ols_margin"], "step300": step300_record["ols_margin"], "bootstrap": ols_bootstrap},
            "template_anchor": {"step0": step0_record["template_anchor"], "step300": step300_record["template_anchor"], "role": "descriptive mechanism evidence; not a substitute for Smooth Gate"},
            "collapse_diagnostics": {"step0": step0_record["collapse"], "step300": step300_record["collapse"]},
            "initial_gradient_balance": gradient_balance,
            "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")},
            "artifacts": {"sufficient_observations": sufficient_ref, "cyclic_raw_outputs": cyclic_raw_ref},
            "decision": decision,
            "stage_a_firewall": stage_firewall,
            "claim_boundary": {**payload["claim_boundary"], "scientific_training_executed": True, "scientific_300_step_executed": True},
            "governance": {"optimizer": "Adam on encoder.phase and generator.phase", "backward": True, "encoder_update": True, "generator_update": True, "image_loss_gradient": False, "prototype_loss_gradient": False, "smooth_evaluator_gradient": False, "retry": False, "second_arm": False, "second_hyperparameter_attempt": False, "stage_b": False, "scale_extraction": False, "prior_shaping": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False, "training_authorized": False, "generation_readiness": "not_ready"},
            "next": "no_retry; V9 bounded result does not authorize Stage B or generation",
        })
    except HardBlocked as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": error.classification, "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific retry"})
    except (V6.HardBlocked, V7.HardBlocked) as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": "V9_HARD_BLOCKED", "hard_blocker": {"stage": getattr(error, "stage", "runtime"), "message": str(error)}, "next": "stop; no scientific retry"})
    except (InvalidV9, V8.InvalidV8, V7.InvalidV7, V6.InvalidV6) as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V9_SCIENTIFIC_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no scientific retry"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V9_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no scientific retry"})
    finally:
        ending = V6.git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == _status_lines(root), "unexpected_modified_paths": sorted(set(_status_lines(root)).symmetric_difference(status_before)), "scientific_started": scientific_started})
        payload["claim_boundary"]["scientific_training_executed"] = bool(scientific_started)
        payload["claim_boundary"]["scientific_300_step_executed"] = bool(updates_completed >= SCIENTIFIC_STEPS)
        payload["scientific_updates_completed"] = updates_completed
    report_path = output_dir / SCIENTIFIC_REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V9 scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": V7.file_sha256(report_path)}
    lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "preflight": payload.get("preflight"), "provenance": payload["provenance"], "artifacts": {**payload.get("artifacts", {}), "report": report_ref}, "claim_boundary": payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_self_reference"}
    lineage_path = output_dir / SCIENTIFIC_LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V9 scientific lineage")
    _require(_read_json(report_path, "V9 report reload").get("result") == payload["result"] and _read_json(lineage_path, "V9 lineage reload").get("result") == payload["result"], "V9 report/lineage reload changed")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": V7.file_sha256(lineage_path)}, "scientific_updates": int(payload.get("scientific_updates_completed", 0)), "stage_a_firewall": payload.get("stage_a_firewall"), "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    if args.action == "preflight":
        result = run_preflight(root, (args.output_dir or root / DEFAULT_PREFLIGHT_DIR).resolve())
    else:
        result = run_scientific(root, (args.output_dir or root / DEFAULT_SCIENTIFIC_DIR).resolve())
    return 0 if result["result"] in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
