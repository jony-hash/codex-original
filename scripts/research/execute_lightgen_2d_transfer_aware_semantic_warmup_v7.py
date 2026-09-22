"""Run the one-shot V7 transfer-aware semantic warmup.

V7 trains only encoder.phase with the frozen generator-aware template-transfer
objective.  Image reconstruction, prototype margins, and Smooth evaluation are
never part of the differentiated objective.  The script performs one bounded
preflight and, when valid, one 300-update scientific arm.
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
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training import (
    file_sha256,
)
from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 import (
    DeterministicTrainRoleSampler,
    load_v2_training_contract,
)
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load
from ldm_project.v2.student.evaluator import SmoothMNISTEvaluator


def _load_v6_execution() -> Any:
    path = Path(__file__).with_name("execute_lightgen_2d_proto_only_semantic_warmup_v6.py")
    spec = importlib.util.spec_from_file_location("lightgen_v6_execution_authority", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V6 execution authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V6 = _load_v6_execution()

TASK_ID = "EXECUTE_LIGHTGEN_2D_TRANSFER_AWARE_SEMANTIC_WARMUP_V7"
OUTER_GATE = "V7_TRANSFER_AWARE_SEMANTIC_WARMUP_SUPPORTED"
SUPPORTED_CLASSIFICATION = OUTER_GATE
NOT_SUPPORTED_CLASSIFICATION = "V7_TRANSFER_AWARE_SEMANTIC_WARMUP_NOT_SUPPORTED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

CONTRACT_PATH = Path(
    "outputs/20260911_v7/runs/"
    "lightgen_2d_transfer_aware_semantic_warmup_contract_v7_attempt_01/"
    "lightgen_2d_transfer_aware_semantic_warmup_contract.v7.json"
)
CONTRACT_SHA256 = "974d6dc560d2f4427cc75e5dc3d3404140d6e2a84439d800f6af10e299bc7766"

DEFAULT_PREFLIGHT_DIR = Path(
    "outputs/20260911_v7/runs/"
    "lightgen_2d_transfer_aware_semantic_warmup_preflight_v7_attempt_03"
)
DEFAULT_SCIENTIFIC_DIR = Path(
    "outputs/20260911_v7/runs/"
    "lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_attempt_01"
)

PREFLIGHT_INPUT_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_preflight_input.v1.json"
PREFLIGHT_REPORT_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_preflight_lineage.v1.json"
PREFLIGHT_CHECKPOINT_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_preflight_smoke_checkpoint.v1.pt"
PREFLIGHT_SYNTHETIC_OBS_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_synthetic_sufficient_observations.v1.pt"
PREFLIGHT_SYNTHETIC_REPORT_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_synthetic_report.v1.json"
PREFLIGHT_SYNTHETIC_LINEAGE_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_synthetic_lineage.v1.json"

SCIENTIFIC_INPUT_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_input.v1.json"
SCIENTIFIC_REPORT_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_report.v1.json"
SCIENTIFIC_LINEAGE_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_lineage.v1.json"
SCIENTIFIC_SUFFICIENT_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_sufficient_observations.v1.pt"
SCIENTIFIC_RAW_NAME = "lightgen_2d_transfer_aware_semantic_warmup_v7_step300_cyclic_raw_outputs.v1.pt"

REPORT_SCHEMA = "lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_lineage.v1"
INPUT_SCHEMA = "lightgen_2d_transfer_aware_semantic_warmup_v7_scientific_input.v1"
SUFFICIENT_SCHEMA = "lightgen_2d_transfer_aware_semantic_warmup_v7_sufficient_observations.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_transfer_aware_semantic_warmup_v7_checkpoint.v1"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
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
STAGE_A_FIREWALL = 0.937
SMOOTH_REQUIRED_DELTA = 0.05
SELECTION_ROWS = 1000
SELECTION_PER_CLASS = 100
SELECTION_SEED = 4321
SPLIT_SEED = 1234
V3_ENCODER_SHA256 = "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca"
V4_GENERATOR_TENSOR_SHA256 = "83aa74b701049c09c5efb63dbd6d3a6d181184a964d3d3c0c06665ca97da5afe"
PROTOTYPE_SCALE = 0.00555040966719389
PROTOTYPE_SHA256 = "c02a728e8631e24918b018553958a44b25cdb358192c6b4b28239f55e93df912"
SELECTION_INDICES_SHA256 = "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8"
SELECTION_LABELS_SHA256 = "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"


class InvalidV7(RuntimeError):
    """The V7 harness or immutable authority is invalid."""


class HardBlocked(RuntimeError):
    """A valid hard boundary prevents execution."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidV7(message)


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
        return len(left) == len(right) and all(
            _tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _digest_tree(value: Any) -> str:
    import hashlib

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
    import subprocess

    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidV7(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


def _validate_hash(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = file_sha256(path)
    _require(observed == expected, f"{name} SHA mismatch: expected {expected}, observed {observed}")


def _load_contract(root: Path) -> tuple[dict[str, Any], Any]:
    path = _path(root, CONTRACT_PATH)
    _validate_hash(path, CONTRACT_SHA256, "V7 contract")
    payload = _read_json(path, "V7 contract")
    _require(
        payload.get("schema") == "lightgen_2d_transfer_aware_semantic_warmup_contract.v7",
        "V7 contract schema changed",
    )
    _require(
        payload.get("result") == "READY"
        and payload.get("status") == "valid_completed"
        and payload.get("outer_gate") == OUTER_GATE,
        "V7 contract is not READY",
    )
    definition = _mapping(payload.get("definition_entrypoint"), "V7 definition entrypoint")
    _validate_hash(_path(root, definition.get("path")), str(definition.get("sha256")), "V7 definition entrypoint")
    git = _mapping(payload.get("git"), "V7 contract Git")
    _require(
        git.get("branch") == EXPECTED_BRANCH and git.get("head") == EXPECTED_HEAD,
        "V7 contract Git identity changed",
    )
    parent = _mapping(payload.get("parent_contract"), "V6 parent contract")
    _require(parent.get("sha256"), "V6 parent contract binding is missing")
    v2_contract = load_v2_training_contract(
        _path(root, V6.V2_CONTRACT_PATH),
        repo_root=root,
    )
    _require(v2_contract.sha256 == V6.V2_CONTRACT_SHA256, "V2 contract identity changed")
    return payload, v2_contract


def _resolve_authorities(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    authorities = V6._resolve_authorities(root, contract)
    _require(
        authorities["initial_state"]["encoder.phase"].shape == (2, 1, 100, 100),
        "V3 encoder shape changed",
    )
    _require(
        authorities["initial_state"]["generator.phase"].shape == (2, 1, 100, 100),
        "V4 generator shape changed",
    )
    _require(
        V6.tensor_sha256(authorities["initial_state"]["encoder.phase"]) == V3_ENCODER_SHA256,
        "V7 initial encoder is not exact V3 CONTROL",
    )
    _require(
        V6.tensor_sha256(authorities["initial_state"]["generator.phase"])
        == V4_GENERATOR_TENSOR_SHA256,
        "V7 generator is not exact V4",
    )
    _require(
        V6.tensor_sha256(authorities["prototypes"]) == PROTOTYPE_SHA256,
        "V7 prototypes changed",
    )
    _require(
        not authorities["prototypes"].requires_grad and not authorities["templates"].requires_grad,
        "V7 frozen support tensors unexpectedly require gradients",
    )
    return authorities


def _configure_encoder_only(consumer: Any) -> dict[str, Any]:
    return V6._configure_encoder_only(consumer)


def _configure_all_frozen(consumer: Any) -> None:
    V6._configure_all_frozen(consumer)


def _new_consumer(v2_contract: Any, device: torch.device, grid: int) -> Any:
    return V6._new_consumer(v2_contract, device, grid)


def _load_runtime_state(consumer: Any, state: Mapping[str, Any]) -> None:
    V6._load_runtime_state(consumer, state)


def _validate_result(result: Any, batch_size: int, *, require_grad: bool) -> torch.Tensor:
    return V6._validate_result(result, batch_size, require_grad=require_grad)


def _optimizer(consumer: Any) -> torch.optim.Adam:
    return V6._optimizer(consumer)


def _template_margins_differentiable(
    raw_base: torch.Tensor,
    raw_shifted: torch.Tensor,
    templates: torch.Tensor,
    labels: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    base_view = F.adaptive_avg_pool2d(raw_base.clamp(0.0, 1.0), (28, 28)).to(torch.float64)
    shifted_view = F.adaptive_avg_pool2d(raw_shifted.clamp(0.0, 1.0), (28, 28)).to(torch.float64)
    support = templates.to(device=raw_base.device, dtype=torch.float64)
    source_templates = support.index_select(0, labels)
    target_templates = support.index_select(0, targets)
    base_margin = (
        (base_view - source_templates).square().flatten(1).mean(dim=1)
        - (base_view - target_templates).square().flatten(1).mean(dim=1)
    )
    shifted_margin = (
        (shifted_view - source_templates).square().flatten(1).mean(dim=1)
        - (shifted_view - target_templates).square().flatten(1).mean(dim=1)
    )
    _finite(base_margin, "V7 differentiable template base margin")
    _finite(shifted_margin, "V7 differentiable template shifted margin")
    return base_margin, shifted_margin


def _transfer_components(
    consumer: Any,
    batch: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    templates: torch.Tensor,
    *,
    require_grad: bool,
) -> dict[str, torch.Tensor]:
    result = consumer.runtime.forward_from_image(batch)
    raw_base = _validate_result(result, int(batch.shape[0]), require_grad=require_grad)
    z = result.z_ols.coefficients
    targets = (labels + 1) % CLASS_COUNT
    support = prototypes.to(device=z.device)
    z_shift = z - support.index_select(0, labels) + support.index_select(0, targets)
    shifted_result = consumer.runtime.forward_from_ols(z_shift)
    raw_shift = shifted_result.raw_intensity
    _require(
        tuple(raw_shift.shape) == tuple(raw_base.shape)
        and raw_shift.dtype == torch.float32,
        "V7 shifted raw shape/dtype changed",
    )
    _finite(raw_shift, "V7 shifted raw")
    base_margin, shifted_margin = _template_margins_differentiable(
        raw_base,
        raw_shift,
        templates,
        labels,
        targets,
    )
    delta = shifted_margin - base_margin
    transfer_loss = F.softplus(-delta).mean()
    _finite(delta, "V7 template transfer delta")
    _finite(transfer_loss, "V7 transfer loss")
    return {
        "raw_base": raw_base,
        "raw_shift": raw_shift,
        "target": result.image_input.slm_intensity,
        "z": z,
        "z_shift": z_shift,
        "base_margin": base_margin,
        "shifted_margin": shifted_margin,
        "delta": delta,
        "transfer_loss": transfer_loss,
    }


def _train_step(
    consumer: Any,
    evaluator: SmoothMNISTEvaluator,
    optimizer: torch.optim.Adam,
    batch: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    templates: torch.Tensor,
    step: int,
) -> dict[str, Any]:
    named = dict(consumer.runtime.named_parameters())
    before_frozen = V6._frozen_state(consumer)
    before_encoder = named["encoder.phase"].detach().cpu().clone()
    optimizer.zero_grad(set_to_none=True)
    for parameter in named.values():
        parameter.grad = None
    _require(not prototypes.requires_grad and not templates.requires_grad, "V7 frozen support tensors require gradients")
    components = _transfer_components(
        consumer,
        batch,
        labels,
        prototypes,
        templates,
        require_grad=True,
    )
    with torch.no_grad():
        target = components["target"].detach()
        train_image_loss = (components["raw_base"].detach() - target).square().mean()
        shift_target = target
        train_shift_image_loss = (
            components["raw_shift"].detach() - shift_target
        ).square().mean()
    _finite(train_image_loss, "V7 evaluation-only TRAIN image loss")
    _finite(train_shift_image_loss, "V7 evaluation-only shifted TRAIN image loss")
    loss = components["transfer_loss"]
    _require(loss.requires_grad, "V7 transfer loss is detached")
    loss.backward()
    encoder_gradient = named["encoder.phase"].grad
    _require(encoder_gradient is not None, "V7 transfer gradient is missing")
    gradient_norm = _norm(encoder_gradient, "V7 transfer gradient")
    _require(gradient_norm > 0.0, "V7 transfer gradient is zero")
    _require(named["generator.phase"].grad is None, "V7 frozen generator received a gradient")
    _require(
        all(not parameter.requires_grad and parameter.grad is None for parameter in evaluator.parameters()),
        "V7 Smooth evaluator received a gradient",
    )
    optimizer.step()
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    after_frozen = V6._frozen_state(consumer)
    after_encoder = named["encoder.phase"].detach().cpu().clone()
    update_norm = float((after_encoder - before_encoder).to(torch.float64).square().sum().sqrt())
    _require(math.isfinite(update_norm) and update_norm > 0.0, "V7 encoder update is zero or nonfinite")
    _require(V6._tree_equal(before_frozen, after_frozen), "V7 frozen generator/OLS state changed")
    return {
        "step": step,
        "objective": "L_V7=L_transfer=mean(softplus(-Delta_template))",
        "L_transfer": float(loss.detach().cpu()),
        "L_image_evaluation_only": float(train_image_loss.detach().cpu()),
        "L_image_shift_evaluation_only": float(train_shift_image_loss.detach().cpu()),
        "Delta_template_mean": float(components["delta"].detach().to(torch.float64).mean().cpu()),
        "gradient_norms": {
            "encoder.phase": gradient_norm,
            "generator.phase": None,
            "image_loss_contribution": 0.0,
            "prototype_loss_contribution": 0.0,
            "smooth_evaluator_contribution": 0.0,
        },
        "update_norms": {"encoder.phase": update_norm, "generator.phase": 0.0},
        "image_loss_in_differentiated_objective": False,
        "prototype_loss_in_differentiated_objective": False,
        "smooth_in_differentiated_objective": False,
        "generator_unchanged": True,
        "ols_unchanged": True,
        "trainable_parameter_names": ["encoder.phase"],
        "frozen_parameter_names": ["generator.phase", "OLS", "T=I_100", "W_OLS"],
        "image_loss_gradient": False,
        "prototype_loss_gradient": False,
        "smooth_evaluator_gradient": False,
        "runtime_state_sha256_after": _digest_tree(V6._cpu_tree(consumer.runtime.state_dict())),
    }


def _measure_train_batch(
    consumer: Any,
    batch: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    templates: torch.Tensor,
) -> dict[str, Any]:
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    with torch.no_grad():
        components = _transfer_components(
            consumer,
            batch,
            labels,
            prototypes,
            templates,
            require_grad=False,
        )
        target = components["target"]
        image_loss = (components["raw_base"] - target).square().mean()
        shifted_image_loss = (components["raw_shift"] - target).square().mean()
    consumer.runtime.train(was_training)
    return {
        "L_transfer": float(components["transfer_loss"].cpu()),
        "L_image_evaluation_only": float(image_loss.cpu()),
        "L_image_shift_evaluation_only": float(shifted_image_loss.cpu()),
        "Delta_template_mean": float(components["delta"].to(torch.float64).mean().cpu()),
    }


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
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        "checkpoint_step": step,
        "runtime_state": V6._cpu_tree(consumer.runtime.state_dict()),
        "optimizer_state": V6._cpu_tree(optimizer.state_dict()),
        "sampler_state": V6._cpu_tree(sampler.state_dict()),
        "rng_state": V6._cpu_tree(rng_state),
        "initial_runtime_state_sha256": initial_state_digest,
        "prototypes_sha256": V6.tensor_sha256(prototypes),
        "templates_sha256": V6.tensor_sha256(templates),
        "objective": "L_V7=L_transfer=mean(softplus(-Delta_template))",
        "trainable_parameter_names": ["encoder.phase"],
        "frozen_parameter_names": [
            "generator.phase",
            "prototypes",
            "TRAIN templates",
            "OLS",
            "T=I_100",
            "W_OLS",
        ],
        "image_loss_gradient_contribution": 0.0,
        "prototype_loss_gradient_contribution": 0.0,
        "smooth_evaluator_gradient_contribution": 0.0,
        "data_roles": {"TRAIN": True, "DEV": True, "protected": False, "official_test": False, "sealed": False},
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
    atomic_torch_save_new(path, payload, artifact_kind="V7 checkpoint")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V7 checkpoint reload")
    _require(
        loaded.get("schema") == CHECKPOINT_SCHEMA
        and loaded.get("checkpoint_step") == payload.get("checkpoint_step"),
        "V7 checkpoint schema/step reload changed",
    )
    _require(
        loaded.get("prototypes_sha256") == PROTOTYPE_SHA256
        and loaded.get("templates_sha256") == V6.tensor_sha256(V6._load_ref(root, V6.HISTORICAL_RAW_PATH, V6.HISTORICAL_RAW_SHA256, "historical raw")["templates"]),
        "V7 checkpoint support identity changed",
    )
    clone = _new_consumer(v2_contract, consumer.device, consumer.runtime.config.propagation_grid_size)
    _load_runtime_state(clone, _mapping(loaded.get("runtime_state"), "V7 checkpoint runtime state"))
    _configure_encoder_only(clone)
    _require(
        _digest_tree(V6._cpu_tree(clone.runtime.state_dict()))
        == _digest_tree(V6._cpu_tree(loaded["runtime_state"])),
        "V7 checkpoint runtime reload changed",
    )
    clone_optimizer = _optimizer(clone)
    clone_optimizer.load_state_dict(dict(_mapping(loaded.get("optimizer_state"), "V7 checkpoint optimizer state")))
    _require(
        V6._tree_equal(V6._cpu_tree(clone_optimizer.state_dict()), loaded["optimizer_state"]),
        "V7 checkpoint optimizer reload changed",
    )
    current_rng = V6._capture_rng()
    V6._restore_rng(_mapping(loaded.get("rng_state"), "V7 checkpoint RNG state"))
    reloaded_rng = V6._capture_rng()
    V6._restore_rng(current_rng)
    _require(V6._tree_equal(reloaded_rng, loaded["rng_state"]), "V7 checkpoint RNG reload changed")
    return {
        "path": _relative(root, path),
        "sha256": digest,
        "step": int(payload["checkpoint_step"]),
        "reloaded": True,
        "resume_identity": True,
    }


def _synthetic_acceptance(
    root: Path,
    bundle: Mapping[str, Any],
    device: torch.device,
    target: Path,
) -> dict[str, Any]:
    consumer = _new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    _load_runtime_state(consumer, bundle["initial_state"])
    firewall = _configure_encoder_only(consumer)
    optimizer = _optimizer(consumer)
    ramp = torch.linspace(0.0, 1.0, 28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)
    train_labels = torch.arange(CLASS_COUNT, dtype=torch.long)
    train_images = (ramp + train_labels.view(-1, 1, 1, 1).to(torch.float32) / 20.0).clamp(0.0, 1.0)
    eval_labels = torch.arange(CLASS_COUNT, dtype=torch.long).repeat_interleave(2)
    eval_images = train_images.index_select(0, eval_labels)
    traces = []
    for step in range(1, 4):
        traces.append(
            _train_step(
                consumer,
                bundle["smooth"],
                optimizer,
                train_images.to(device),
                train_labels.to(device),
                bundle["prototypes"].to(device),
                bundle["templates"].to(device),
                step,
            )
        )
    evaluation = V6._evaluate(
        consumer,
        bundle["smooth"],
        eval_images.to(device),
        eval_labels,
        bundle["prototypes"].to(device),
        step=3,
        train_trace=traces[-1],
    )
    selection_indices = torch.arange(eval_images.shape[0], dtype=torch.long)
    cyclic = V6._cyclic_transfer(
        consumer,
        bundle["smooth"],
        eval_images.to(device),
        eval_labels,
        selection_indices,
        bundle["prototypes"],
        bundle["templates"],
    )
    observations = {
        "synthetic_step3_ols_margin": evaluation["_margins"],
        "synthetic_step3_smooth_logits": evaluation["_smooth_logits"],
        **cyclic["_observations"],
    }
    sufficient = _save_sufficient(
        root,
        target,
        observations,
        name=PREFLIGHT_SYNTHETIC_OBS_NAME,
        schema="lightgen_2d_transfer_aware_semantic_warmup_v7_synthetic_sufficient_observations.v1",
    )
    decision = _classify(
        float(evaluation["smooth"]["macro_recall"]),
        float(evaluation["smooth"]["macro_recall"]),
        V6._paired_bootstrap(
            cyclic["_observations"]["smooth_shifted_margin"] - cyclic["_observations"]["smooth_base_margin"],
            eval_labels,
            replicates=9,
        ),
        cyclic,
        True,
    )
    report_payload = {
        "schema": "lightgen_2d_transfer_aware_semantic_warmup_v7_synthetic_report.v1",
        "scientific_meaning": "none",
        "training_updates": 3,
        "firewall": firewall,
        "traces": traces,
        "evaluation": {key: value for key, value in evaluation.items() if not key.startswith("_")},
        "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")},
        "decision": decision,
        "sufficient_observations": sufficient,
    }
    report_path = target / PREFLIGHT_SYNTHETIC_REPORT_NAME
    atomic_write_new_json(report_path, report_payload, artifact_kind="V7 synthetic report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage_payload = {
        "schema": "lightgen_2d_transfer_aware_semantic_warmup_v7_synthetic_lineage.v1",
        "result": "VALID_COMPLETED",
        "report": report_ref,
        "sufficient_observations": sufficient,
        "claim_boundary": {"scientific_meaning": "none", "protected": False, "sealed": False},
    }
    lineage_path = target / PREFLIGHT_SYNTHETIC_LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage_payload, artifact_kind="V7 synthetic lineage")
    _require(
        _read_json(report_path, "V7 synthetic report reload").get("schema")
        == report_payload["schema"],
        "V7 synthetic report reload changed",
    )
    _require(
        _read_json(lineage_path, "V7 synthetic lineage reload").get("result")
        == "VALID_COMPLETED",
        "V7 synthetic lineage reload changed",
    )
    return {
        "scientific_meaning": "none",
        "firewall": firewall,
        "traces": traces,
        "cyclic_transfer": {
            key: value for key, value in cyclic.items() if not key.startswith("_")
        },
        "sufficient_observations": sufficient,
        "report": report_ref,
        "lineage": {
            "path": _relative(root, lineage_path),
            "sha256": file_sha256(lineage_path),
        },
        "report_lineage_reloaded": True,
    }


def _save_sufficient(
    root: Path,
    target: Path,
    observations: Mapping[str, Any],
    *,
    name: str,
    schema: str,
) -> dict[str, Any]:
    path = target / name
    payload = {
        "schema": schema,
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        **{key: value.cpu() if torch.is_tensor(value) else value for key, value in observations.items()},
    }
    atomic_torch_save_new(path, payload, artifact_kind="V7 sufficient observations")
    digest = file_sha256(path)
    loaded = _mapping(
        safe_torch_load(path, expected_sha256=digest, map_location="cpu"),
        "V7 sufficient observations reload",
    )
    _require(loaded.get("schema") == schema, "V7 sufficient observation schema reload changed")
    for key, value in observations.items():
        if torch.is_tensor(value):
            _require(torch.equal(loaded[key], value.cpu()), f"V7 sufficient observation reload changed: {key}")
    return {"path": _relative(root, path), "sha256": digest, "schema": schema, "reloaded": True}


def _save_cyclic_raw(root: Path, target: Path, cyclic: Mapping[str, Any]) -> dict[str, Any]:
    observations = cyclic["_observations"]
    payload = {
        "schema": "lightgen_2d_transfer_aware_semantic_warmup_v7_cyclic_raw_outputs.v1",
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        **{key: value.cpu() if torch.is_tensor(value) else value for key, value in observations.items()},
    }
    path = target / SCIENTIFIC_RAW_NAME
    atomic_torch_save_new(path, payload, artifact_kind="V7 cyclic raw outputs")
    digest = file_sha256(path)
    loaded = _mapping(
        safe_torch_load(path, expected_sha256=digest, map_location="cpu"),
        "V7 cyclic raw reload",
    )
    _require(
        loaded.get("schema") == payload["schema"]
        and torch.equal(loaded["selection_indices"], observations["selection_indices"].cpu()),
        "V7 cyclic raw reload changed",
    )
    return {"path": _relative(root, path), "sha256": digest, "reloaded": True}


def _classify(
    r0: float,
    r300: float,
    smooth_bootstrap: Mapping[str, Any],
    cyclic: Mapping[str, Any],
    raw_valid: bool,
) -> dict[str, Any]:
    smooth_delta = r300 - r0
    natural_pass = smooth_delta >= SMOOTH_REQUIRED_DELTA and smooth_bootstrap["ci_lower"] > 0.0
    cyclic_pass = bool(cyclic["known_cyclic_transfer_retained"])
    supported = bool(natural_pass and cyclic_pass and raw_valid)
    return {
        "classification": SUPPORTED_CLASSIFICATION if supported else NOT_SUPPORTED_CLASSIFICATION,
        "natural_smooth_pass": bool(natural_pass),
        "known_cyclic_transfer_retained": cyclic_pass,
        "raw_validity_pass": bool(raw_valid),
        "primary_gate": {
            "smooth_delta_threshold": SMOOTH_REQUIRED_DELTA,
            "smooth_delta": smooth_delta,
            "smooth_ci_lower": float(smooth_bootstrap["ci_lower"]),
            "cyclic_transfer": cyclic_pass,
            "raw_validity": bool(raw_valid),
        },
    }


def _gradient_firewall_probe(
    bundle: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    consumer = _new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    _load_runtime_state(consumer, bundle["initial_state"])
    firewall = _configure_encoder_only(consumer)
    indices = train_indices[:LOGICAL_BATCH_SIZE]
    batch = images.index_select(0, indices).to(device)
    batch_labels = labels.index_select(0, indices).to(device)
    optimizer = _optimizer(consumer)
    named = dict(consumer.runtime.named_parameters())
    optimizer.zero_grad(set_to_none=True)
    components = _transfer_components(
        consumer,
        batch,
        batch_labels,
        bundle["prototypes"].to(device),
        bundle["templates"].to(device),
        require_grad=True,
    )
    with torch.no_grad():
        image_loss = (
            components["raw_base"].detach()
            - components["target"].detach()
        ).square().mean()
    _require(not image_loss.requires_grad, "V7 image loss has a gradient path")
    _require(components["transfer_loss"].requires_grad, "V7 transfer loss has no gradient path")
    negative_control_loss = F.softplus(components["delta"]).mean()
    negative_control_gradient = torch.autograd.grad(
        negative_control_loss,
        named["encoder.phase"],
        retain_graph=True,
    )[0]
    negative_control_norm = _norm(negative_control_gradient, "V7 negative-control gradient")
    _require(negative_control_norm > 0.0, "V7 negative-control gradient is zero")
    components["transfer_loss"].backward()
    named_gradient = named["encoder.phase"].grad
    gradient_norm = _norm(named_gradient, "V7 probe encoder gradient")
    _require(gradient_norm > 0.0, "V7 probe transfer gradient is zero")
    _require(named["generator.phase"].grad is None, "V7 probe generator gradient is nonzero")
    _require(
        all(not parameter.requires_grad and parameter.grad is None for parameter in bundle["smooth"].parameters()),
        "V7 probe Smooth evaluator firewall failed",
    )
    for parameter in named.values():
        parameter.grad = None
    return {
        "firewall": firewall,
        "objective": "L_V7=L_transfer",
        "image_loss_requires_grad": False,
        "image_loss_gradient_contribution": 0.0,
        "prototype_loss_gradient_contribution": 0.0,
        "smooth_evaluator_gradient_contribution": 0.0,
        "transfer_gradient_norm": gradient_norm,
        "transfer_gradient_finite": True,
        "negative_control": {
            "objective": "mean(softplus(+Delta_template))",
            "would_penalize_positive_delta": True,
            "gradient_norm": negative_control_norm,
            "gradient_finite": True,
        },
        "generator_gradient": None,
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
    firewall = _configure_encoder_only(consumer)
    optimizer = _optimizer(consumer)
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    traces = []
    for step in range(1, PRECHECK_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        batch = images.index_select(0, indices).to(device)
        traces.append(
            _train_step(
                consumer,
                bundle["smooth"],
                optimizer,
                batch,
                batch_labels.to(device),
                bundle["prototypes"].to(device),
                bundle["templates"].to(device),
                step,
            )
        )
    payload = _checkpoint_payload(
        consumer,
        optimizer,
        sampler,
        step=PRECHECK_STEPS,
        initial_state_digest=_digest_tree(bundle["initial_state"]),
        prototypes=bundle["prototypes"],
        templates=bundle["templates"],
        rng_state=V6._capture_rng(),
    )
    checkpoint = _save_and_verify_checkpoint(
        root,
        target,
        payload,
        PREFLIGHT_CHECKPOINT_NAME,
        consumer,
        bundle["v2_contract"],
    )
    return {
        "firewall": firewall,
        "traces": traces,
        "checkpoint": checkpoint,
        "scientific_updates": 0,
        "scientific_meaning": "none",
    }


def _preflight_input_payload(
    payload: Mapping[str, Any],
    *,
    device_facts: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "lightgen_2d_transfer_aware_semantic_warmup_v7_preflight_input.v1",
        "version": 1,
        "task_id": TASK_ID,
        "contract": payload["contract"],
        "runtime": device_facts,
        "protocol": {
            "objective": "L_V7=L_transfer=mean(softplus(-Delta_template))",
            "image_loss_gradient_contribution": 0.0,
            "prototype_loss_gradient_contribution": 0.0,
            "smooth_evaluator_gradient_contribution": 0.0,
            "updates": SCIENTIFIC_STEPS,
            "checkpoints": list(CHECKPOINT_STEPS),
            "grid": PRODUCTION_GRID,
        },
        "data_roles": {
            "TRAIN_smoke_rows": LOGICAL_BATCH_SIZE * PRECHECK_STEPS,
            "DEV_scientific_values": 0,
            "protected": 0,
            "official_test": 0,
            "sealed": 0,
        },
    }


def run_preflight(root: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to overwrite V7 preflight output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V6.git_facts(root)
    status_before = _status_lines(root)
    payload: dict[str, Any] = {
        "schema": "lightgen_2d_transfer_aware_semantic_warmup_v7_preflight_report.v1",
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "INVALID",
        "status": "invalid_harness",
        "classification": "INVALID_V7_PREFLIGHT",
        "outer_gate": OUTER_GATE,
        "provenance": {"git_start": started},
        "claim_boundary": {
            "scientific_training_executed": False,
            "scientific_300_step_executed": False,
            "stage_b_executed": False,
            "generation_executed": False,
            "protected_accessed": False,
            "official_test_accessed": False,
            "sealed_accessed": False,
            "formal_promotion": False,
        },
    }
    try:
        _require(
            started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD,
            "active Git identity differs from V7 authorization",
        )
        contract, v2_contract = _load_contract(root)
        bundle = _resolve_authorities(root, contract)
        try:
            dll = V6._prepare_cuda_runtime_dll_search_path()
            device, device_facts = V6._cuda_facts()
        except V6.HardBlocked as error:
            raise HardBlocked("V7_CUDA_RUNTIME_BLOCKED", error.stage, str(error), **error.details) from error
        bundle["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = V6._load_roles(root)
        _ = development_indices
        production_smoke = V6._production_no_data_smoke(bundle, device)
        gradient_probe = _gradient_firewall_probe(bundle, images, labels, train_indices, device)
        smoke = _three_step_smoke(root, bundle, images, labels, train_indices, device, output_dir)
        synthetic = _synthetic_acceptance(root, bundle, device, output_dir)
        payload.update(
            {
                "result": "AUTHORIZABLE",
                "status": "valid_completed",
                "classification": "LIGHTGEN_2D_TRANSFER_AWARE_SEMANTIC_WARMUP_V7_AUTHORIZABLE",
                "contract": {
                    "path": CONTRACT_PATH.as_posix(),
                    "sha256": CONTRACT_SHA256,
                },
                "runtime": {"cuda_dll_search_path": dll, "device": device_facts},
                "authorities": bundle["authorities"],
                "data_roles": dataset,
                "production_grid_smoke": production_smoke,
                "gradient_firewall_probe": gradient_probe,
                "three_step_real_train_smoke": smoke,
                "synthetic_acceptance": synthetic,
                "next": "proceed directly to the one-shot 300-step V7 scientific execution",
            }
        )
        input_path = output_dir / PREFLIGHT_INPUT_NAME
        atomic_write_new_json(
            input_path,
            _preflight_input_payload(payload, device_facts=device_facts),
            artifact_kind="V7 preflight input",
        )
        payload["input"] = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
    except HardBlocked as blocked:
        payload.update(
            {
                "result": "BLOCKED",
                "status": "valid_failed",
                "classification": blocked.classification,
                "hard_blocker": {"stage": blocked.stage, "message": str(blocked)},
                "next": "stop; no scientific execution",
            }
        )
    except (InvalidV7, V6.InvalidV6) as error:
        payload.update(
            {
                "result": "INVALID",
                "status": "invalid_harness",
                "classification": "INVALID_V7_PREFLIGHT_HARNESS",
                "invalid_input": {"type": type(error).__name__, "message": str(error)},
                "next": "stop; repair harness before any scientific execution",
            }
        )
    except Exception as error:
        payload.update(
            {
                "result": "INVALID",
                "status": "invalid_harness",
                "classification": "INVALID_V7_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR",
                "unexpected_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc()[-12_000:],
                },
                "next": "stop; no scientific execution",
            }
        )
    finally:
        ending = V6.git_facts(root)
        payload["provenance"].update(
            {
                "git_end": ending,
                "branch_unchanged": started.get("branch") == ending.get("branch"),
                "head_unchanged": started.get("head") == ending.get("head"),
                "worktree_status_unchanged": status_before == _status_lines(root),
            }
        )
    report_path = output_dir / PREFLIGHT_REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V7 preflight report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {
        "schema": "lightgen_2d_transfer_aware_semantic_warmup_v7_preflight_lineage.v1",
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "contract": payload.get("contract"),
        "provenance": payload["provenance"],
        "report": report_ref,
        "claim_boundary": payload["claim_boundary"],
    }
    lineage_path = output_dir / PREFLIGHT_LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V7 preflight lineage")
    result = {
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "report": report_ref,
        "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)},
        "scientific_300_step_executed": False,
        "next": payload.get("next"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _validate_preflight(root: Path) -> dict[str, Any]:
    report_path = _path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = _path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    _require(report_path.is_file() and lineage_path.is_file(), "V7 preflight authority is unavailable")
    report = _read_json(report_path, "V7 preflight report")
    lineage = _read_json(lineage_path, "V7 preflight lineage")
    _require(
        report.get("result") == "AUTHORIZABLE"
        and report.get("status") == "valid_completed"
        and report.get("classification")
        == "LIGHTGEN_2D_TRANSFER_AWARE_SEMANTIC_WARMUP_V7_AUTHORIZABLE",
        "V7 preflight is not AUTHORIZABLE",
    )
    _require(
        lineage.get("result") == "AUTHORIZABLE"
        and lineage.get("classification") == report.get("classification"),
        "V7 preflight lineage changed",
    )
    _require(
        report.get("claim_boundary", {}).get("scientific_300_step_executed") is False,
        "V7 preflight opened scientific execution",
    )
    return {
        "report": report,
        "report_ref": {"path": _relative(root, report_path), "sha256": file_sha256(report_path)},
        "lineage_ref": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)},
    }


def _public_evaluation(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def run_scientific(root: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to overwrite V7 scientific output: {output_dir}")
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
        "classification": "INVALID_V7_SCIENTIFIC_HARNESS",
        "outer_gate": OUTER_GATE,
        "provenance": {"git_start": started},
        "claim_boundary": {
            "scientific_training_executed": False,
            "scientific_300_step_executed": False,
            "stage_b_executed": False,
            "scale_extraction_executed": False,
            "prior_shaping_executed": False,
            "generation_executed": False,
            "protected_accessed": False,
            "official_test_accessed": False,
            "sealed_accessed": False,
            "formal_promotion": False,
        },
        "governance": {"retry": False, "stage_b": False, "generation": False, "promotion": False},
    }
    scientific_started = False
    try:
        _require(
            started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD,
            "active Git identity differs from V7 authorization",
        )
        contract, v2_contract = _load_contract(root)
        preflight = _validate_preflight(root)
        bundle = _resolve_authorities(root, contract)
        try:
            dll = V6._prepare_cuda_runtime_dll_search_path()
            device, device_facts = V6._cuda_facts()
        except V6.HardBlocked as error:
            raise HardBlocked("V7_CUDA_RUNTIME_BLOCKED", error.stage, str(error), **error.details) from error
        bundle["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = V6._load_roles(root)
        development_images = images.index_select(0, development_indices).contiguous()
        development_labels = labels.index_select(0, development_indices).contiguous()
        selection_images, selection_labels, selection_indices = V6._load_selection(
            root, images, labels, development_indices
        )
        V6._initial_rng()
        consumer = _new_consumer(v2_contract, device, PRODUCTION_GRID)
        _load_runtime_state(consumer, bundle["initial_state"])
        firewall = _configure_encoder_only(consumer)
        optimizer = _optimizer(consumer)
        sampler = DeterministicTrainRoleSampler(
            train_indices,
            labels,
            seed=SPLIT_SEED,
            steps=SCIENTIFIC_STEPS,
        )
        initial_state_digest = _digest_tree(bundle["initial_state"])
        metric_sampler = DeterministicTrainRoleSampler(
            train_indices,
            labels,
            seed=SPLIT_SEED,
            steps=SCIENTIFIC_STEPS,
        )
        metric_indices, metric_labels = metric_sampler.next_batch()
        initial_train_metrics = _measure_train_batch(
            consumer,
            images.index_select(0, metric_indices).to(device),
            metric_labels.to(device),
            bundle["prototypes"].to(device),
            bundle["templates"].to(device),
        )
        payload.update(
            {
                "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
                "preflight": {
                    "report": preflight["report_ref"],
                    "lineage": preflight["lineage_ref"],
                    "result": "AUTHORIZABLE",
                },
                "authorities": bundle["authorities"],
                "runtime": {
                    "cuda_dll_search_path": dll,
                    "device": device_facts,
                    "propagation_grid": PRODUCTION_GRID,
                    "optimizer": {
                        "name": "Adam",
                        "learning_rate": LEARNING_RATE,
                        "betas": list(ADAM_BETAS),
                        "eps": ADAM_EPSILON,
                        "trainable_parameter_names": ["encoder.phase"],
                    },
                },
                "data_roles": dataset,
                "initial_state": {
                    "combined_runtime_state_sha256": initial_state_digest,
                    "encoder_phase_sha256": V3_ENCODER_SHA256,
                    "generator_phase_sha256": V4_GENERATOR_TENSOR_SHA256,
                    "rng_sha256": _digest_tree(V6._capture_rng()),
                },
                "firewall": firewall,
                "objective": {
                    "formula": "L_V7=L_transfer=mean(softplus(-Delta_template))",
                    "image_loss_gradient_contribution": 0.0,
                    "prototype_loss_gradient_contribution": 0.0,
                    "smooth_evaluator_gradient_contribution": 0.0,
                    "image_loss_role": "evaluation-only",
                    "prototype_margin_role": "evaluation-only/descriptive",
                    "smooth_role": "evaluation-only",
                },
                "selection": {
                    "rows": SELECTION_ROWS,
                    "per_class": SELECTION_PER_CLASS,
                    "seed": SELECTION_SEED,
                    "indices_sha256": V6.SELECTION_INDICES_SHA256,
                    "labels_sha256": V6.SELECTION_LABELS_SHA256,
                },
            }
        )
        input_payload = {
            "schema": INPUT_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "contract": payload["contract"],
            "preflight": payload["preflight"],
            "protocol": {
                "objective": "L_V7=L_transfer=mean(softplus(-Delta_template))",
                "steps": SCIENTIFIC_STEPS,
                "checkpoints": list(CHECKPOINT_STEPS),
                "batch": LOGICAL_BATCH_SIZE,
                "grid": PRODUCTION_GRID,
                "optimizer": payload["runtime"]["optimizer"],
                "latent_shift": "z_shift=z-mu_y+mu_t",
                "cyclic_target": "t=(y+1) mod 10",
                "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED},
            },
            "data_roles": {"TRAIN": 55_000, "DEV": 5_000, "protected": 0, "official_test": 0, "sealed": 0},
            "claim_boundary": payload["claim_boundary"],
        }
        input_path = output_dir / SCIENTIFIC_INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V7 scientific input")
        payload["input"] = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        scientific_started = True
        evaluations: dict[str, dict[str, Any]] = {}
        checkpoints: dict[str, dict[str, Any]] = {}
        step0 = V6._evaluate(
            consumer,
            bundle["smooth"],
            development_images,
            development_labels,
            bundle["prototypes"].to(device),
            step=0,
            train_trace=None,
        )
        checkpoints["0"] = _save_and_verify_checkpoint(
            root,
            output_dir,
            _checkpoint_payload(
                consumer,
                optimizer,
                sampler,
                step=0,
                initial_state_digest=initial_state_digest,
                prototypes=bundle["prototypes"],
                templates=bundle["templates"],
                rng_state=V6._capture_rng(),
            ),
            "lightgen_2d_transfer_aware_semantic_warmup_v7_step_0_checkpoint.v1.pt",
            consumer,
            v2_contract,
        )
        evaluations["0"] = step0
        traces: list[dict[str, Any]] = []
        train_metrics = {"0": initial_train_metrics}
        started_training = time.perf_counter()
        for step in range(1, SCIENTIFIC_STEPS + 1):
            indices, batch_labels = sampler.next_batch()
            batch = images.index_select(0, indices).to(device)
            trace = _train_step(
                consumer,
                bundle["smooth"],
                optimizer,
                batch,
                batch_labels.to(device),
                bundle["prototypes"].to(device),
                bundle["templates"].to(device),
                step,
            )
            traces.append(trace)
            del batch
            if step in CHECKPOINT_STEPS[1:]:
                checkpoints[str(step)] = _save_and_verify_checkpoint(
                    root,
                    output_dir,
                    _checkpoint_payload(
                        consumer,
                        optimizer,
                        sampler,
                        step=step,
                        initial_state_digest=initial_state_digest,
                        prototypes=bundle["prototypes"],
                        templates=bundle["templates"],
                        rng_state=V6._capture_rng(),
                    ),
                    f"lightgen_2d_transfer_aware_semantic_warmup_v7_step_{step}_checkpoint.v1.pt",
                    consumer,
                    v2_contract,
                )
                evaluations[str(step)] = V6._evaluate(
                    consumer,
                    bundle["smooth"],
                    development_images,
                    development_labels,
                    bundle["prototypes"].to(device),
                    step=step,
                    train_trace=trace,
                )
                train_metrics[str(step)] = {
                    key: trace[key]
                    for key in (
                        "L_transfer",
                        "L_image_evaluation_only",
                        "L_image_shift_evaluation_only",
                        "Delta_template_mean",
                    )
                }
            if step % 25 == 0:
                print(
                    json.dumps(
                        {
                            "event": "scientific_progress",
                            "step": step,
                            "elapsed_seconds": time.perf_counter() - started_training,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        _require(
            set(evaluations) == {"0", "100", "200", "300"}
            and set(checkpoints) == {"0", "100", "200", "300"},
            "V7 scientific checkpoint/evaluation set is incomplete",
        )
        step0_record = evaluations["0"]
        step300_record = evaluations["300"]
        smooth_correct0 = step0_record["_smooth_logits"].argmax(dim=1).eq(development_labels)
        smooth_correct300 = step300_record["_smooth_logits"].argmax(dim=1).eq(development_labels)
        smooth_bootstrap = V6._paired_bootstrap(
            smooth_correct300.to(torch.float64) - smooth_correct0.to(torch.float64),
            development_labels,
        )
        ols_bootstrap = V6._paired_bootstrap(
            step300_record["_margins"] - step0_record["_margins"],
            development_labels,
        )
        cyclic = V6._cyclic_transfer(
            consumer,
            bundle["smooth"],
            selection_images,
            selection_labels,
            selection_indices,
            bundle["prototypes"],
            bundle["templates"],
        )
        sufficient_obs = {
            "step0_smooth_logits": step0_record["_smooth_logits"],
            "step300_smooth_logits": step300_record["_smooth_logits"],
            "step0_ols_margin": step0_record["_margins"],
            "step300_ols_margin": step300_record["_margins"],
            **cyclic["_observations"],
        }
        sufficient_ref = _save_sufficient(
            root,
            output_dir,
            sufficient_obs,
            name=SCIENTIFIC_SUFFICIENT_NAME,
            schema=SUFFICIENT_SCHEMA,
        )
        cyclic_raw_ref = _save_cyclic_raw(root, output_dir, cyclic)
        raw_valid = all(
            bool(
                evaluations[str(step)]["raw_validity"]["finite"]
                and evaluations[str(step)]["raw_validity"]["structured"]
                and evaluations[str(step)]["raw_validity"]["nonconstant"]
            )
            for step in CHECKPOINT_STEPS
        ) and bool(
            cyclic["raw_validity"]["base"]["finite"]
            and cyclic["raw_validity"]["shifted"]["finite"]
        )
        r0 = float(step0_record["smooth"]["macro_recall"])
        r300 = float(step300_record["smooth"]["macro_recall"])
        decision = _classify(r0, r300, smooth_bootstrap, cyclic, raw_valid)
        stage_firewall = {
            "metric": "step-300 DEV Smooth macro recall",
            "threshold": STAGE_A_FIREWALL,
            "R300": r300,
            "status": "STAGE_A_FIREWALL_PASSED" if r300 >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED",
            "stage_b_authorized": False,
        }
        public_evaluations = {
            step: V6._public_evaluation(record) for step, record in evaluations.items()
        }
        payload.update(
            {
                "result": "VALID_COMPLETED",
                "status": "valid_completed",
                "classification": decision["classification"],
                "evaluations": public_evaluations,
                "train_metrics": train_metrics,
                "traces": traces,
                "checkpoints": checkpoints,
                "scientific_updates": SCIENTIFIC_STEPS,
                "runtime_device": device_facts,
                "step300_minus_step0": {
                    "Delta_R_Smooth": smooth_bootstrap,
                    "Delta_R_observed": r300 - r0,
                    "Delta_M_OLS": ols_bootstrap,
                    "Delta_L_image_DEV": float(
                        step300_record["losses"]["L_image"] - step0_record["losses"]["L_image"]
                    ),
                    "Delta_L_transfer_TRAIN": float(
                        train_metrics["300"]["L_transfer"] - train_metrics["0"]["L_transfer"]
                    ),
                },
                "natural_smooth_semantics": {
                    "R0": r0,
                    "R300": r300,
                    "Delta_R": r300 - r0,
                    "bootstrap": smooth_bootstrap,
                    "step0_per_class": step0_record["smooth"]["per_class_recall"],
                    "step300_per_class": step300_record["smooth"]["per_class_recall"],
                },
                "ols_prototype_margin": {
                    "step0": step0_record["ols_margin"],
                    "step300": step300_record["ols_margin"],
                    "bootstrap": ols_bootstrap,
                },
                "cyclic_transfer": {
                    key: value for key, value in cyclic.items() if not key.startswith("_")
                },
                "artifacts": {
                    "sufficient_observations": sufficient_ref,
                    "cyclic_raw_outputs": cyclic_raw_ref,
                },
                "decision": decision,
                "stage_a_firewall": stage_firewall,
                "claim_boundary": {
                    **payload["claim_boundary"],
                    "scientific_training_executed": True,
                    "scientific_300_step_executed": True,
                },
                "governance": {
                    "optimizer": "Adam on encoder.phase only",
                    "backward": True,
                    "image_loss_gradient": False,
                    "prototype_loss_gradient": False,
                    "smooth_evaluator_gradient": False,
                    "generator_update": False,
                    "optical_update": False,
                    "retry": False,
                    "second_hyperparameter_attempt": False,
                    "stage_b": False,
                    "scale_extraction": False,
                    "prior_shaping": False,
                    "generation": False,
                    "protected": False,
                    "official_test": False,
                    "sealed": False,
                    "formal_promotion": False,
                    "training_authorized": False,
                    "generation_readiness": "not_ready",
                },
                "next": "no_retry; V7 bounded result does not authorize Stage B",
            }
        )
    except HardBlocked as blocked:
        payload.update(
            {
                "result": "BLOCKED",
                "status": "valid_failed",
                "classification": blocked.classification,
                "hard_blocker": {"stage": blocked.stage, "message": str(blocked)},
                "next": "stop; no scientific retry",
            }
        )
    except (InvalidV7, V6.InvalidV6) as error:
        payload.update(
            {
                "result": "INVALID",
                "status": "invalid_harness",
                "classification": "INVALID_V7_SCIENTIFIC_HARNESS",
                "invalid_input": {"type": type(error).__name__, "message": str(error)},
                "next": "stop; no scientific retry",
            }
        )
    except Exception as error:
        payload.update(
            {
                "result": "INVALID",
                "status": "invalid_harness",
                "classification": "INVALID_V7_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR",
                "unexpected_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc()[-12_000:],
                },
                "next": "stop; no scientific retry",
            }
        )
    finally:
        ending = V6.git_facts(root)
        payload["provenance"].update(
            {
                "git_end": ending,
                "branch_unchanged": started.get("branch") == ending.get("branch"),
                "head_unchanged": started.get("head") == ending.get("head"),
                "worktree_status_unchanged": status_before == _status_lines(root),
                "scientific_started": scientific_started,
            }
        )
    report_path = output_dir / SCIENTIFIC_REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V7 scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "contract": payload.get("contract"),
        "preflight": payload.get("preflight"),
        "provenance": payload["provenance"],
        "artifacts": {**payload.get("artifacts", {}), "report": report_ref},
        "claim_boundary": payload["claim_boundary"],
        "lineage_self_hash": "not_embedded_to_avoid_self_reference",
    }
    lineage_path = output_dir / SCIENTIFIC_LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V7 scientific lineage")
    _require(
        _read_json(report_path, "V7 report reload").get("result") == payload["result"]
        and _read_json(lineage_path, "V7 lineage reload").get("result") == payload["result"],
        "V7 report/lineage reload changed",
    )
    result = {
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "outer_gate": OUTER_GATE,
        "report": report_ref,
        "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)},
        "scientific_updates": int(payload.get("scientific_updates", 0)),
        "stage_a_firewall": payload.get("stage_a_firewall"),
        "stage_b_authorized": False,
        "generation_readiness": "not_ready",
        "formal_promotion": False,
        "next": payload.get("next"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    if args.action == "preflight":
        target = (args.output_dir or root / DEFAULT_PREFLIGHT_DIR).resolve()
        result = run_preflight(root, target)
    else:
        target = (args.output_dir or root / DEFAULT_SCIENTIFIC_DIR).resolve()
        result = run_scientific(root, target)
    return 0 if result["result"] in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
