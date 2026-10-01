"""Execute the one-shot V11 semantic/reconstruction compatibility canary.

V11 reuses the V10.1 rank-nine adapter and real-path runtime.  Its only
scientific change is a frozen step-0 gradient-balanced reconstruction term in
the adapter objective.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F


def _load_v10_1_execution() -> Any:
    path = Path(__file__).with_name("execute_lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1.py")
    spec = importlib.util.spec_from_file_location("lightgen_v10_1_execution_authority_v11", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V10.1 execution authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V10_1 = _load_v10_1_execution()
V10 = V10_1.V10
_BASE_OBJECTIVE_COMPONENTS = V10._objective_components
_BASE_EVALUATE = V10._evaluate_v10
_BASE_ATOMIC_WRITE_JSON = V10.atomic_write_new_json

TASK_ID = "EXECUTE_LIGHTGEN_2D_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_V11"
OUTER_GATE = "V11_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_SUPPORTED"
SUPPORTED_CLASSIFICATION = OUTER_GATE
NOT_SUPPORTED_CLASSIFICATION = "V11_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_NOT_SUPPORTED"
PREFLIGHT_CLASSIFICATION = "LIGHTGEN_2D_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_V11_AUTHORIZABLE"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

CONTRACT_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract_v11_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract.v11.json"
)
CONTRACT_SHA256 = "6c49a0cff4e08622351438529cebafe9388537d713a5b78b9fabbc953904efff"
V10_1_CONTRACT_PATH = Path(
    "outputs/20260912_v10_1/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_1_attempt_01/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10_1.json"
)
V10_1_CONTRACT_SHA256 = "45210d92e00ba01255a4340dfc837c514c110640097aa20821d8d5b6e0efab3d"
V9_CONTRACT_PATH = V10.V9_CONTRACT_PATH
V9_CONTRACT_SHA256 = V10.V9_CONTRACT_SHA256

DEFAULT_PREFLIGHT_DIR = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_preflight_v11_attempt_01"
)
DEFAULT_SCIENTIFIC_DIR = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01"
)

PREFLIGHT_INPUT_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_preflight_input.v1.json"
PREFLIGHT_REPORT_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_preflight_lineage.v1.json"
PREFLIGHT_CHECKPOINT_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_preflight_smoke_checkpoint.v1.pt"
PREFLIGHT_SYNTHETIC_OBS_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_synthetic_sufficient_observations.v1.pt"
PREFLIGHT_SYNTHETIC_REPORT_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_synthetic_report.v1.json"
PREFLIGHT_SYNTHETIC_LINEAGE_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_synthetic_lineage.v1.json"

SCIENTIFIC_INPUT_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_input.v1.json"
SCIENTIFIC_REPORT_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_report.v1.json"
SCIENTIFIC_LINEAGE_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_lineage.v1.json"
SCIENTIFIC_SUFFICIENT_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_sufficient_observations.v1.pt"
SCIENTIFIC_RAW_NAME = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_step300_cyclic_raw_outputs.v1.pt"

REPORT_SCHEMA = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_lineage.v1"
INPUT_SCHEMA = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_input.v1"
SUFFICIENT_SCHEMA = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_sufficient_observations.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_checkpoint.v1"

CLASS_COUNT = V10_1.CLASS_COUNT
OLS_PORT_COUNT = V10_1.OLS_PORT_COUNT
ADAPTER_RANK = V10_1.ADAPTER_RANK
INIT_SCALE = V10_1.INIT_SCALE
LOGICAL_BATCH_SIZE = V10_1.LOGICAL_BATCH_SIZE
PRODUCTION_GRID = V10_1.PRODUCTION_GRID
SMOKE_GRID = V10_1.SMOKE_GRID
PRECHECK_STEPS = V10_1.PRECHECK_STEPS
SCIENTIFIC_STEPS = V10_1.SCIENTIFIC_STEPS
CHECKPOINT_STEPS = V10_1.CHECKPOINT_STEPS
LEARNING_RATE = V10_1.LEARNING_RATE
ADAM_BETAS = V10_1.ADAM_BETAS
ADAM_EPSILON = V10_1.ADAM_EPSILON
WEIGHT_DECAY = V10_1.WEIGHT_DECAY
AMSGRAD = V10_1.AMSGRAD
BOOTSTRAP_REPLICATES = V10_1.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = V10_1.BOOTSTRAP_SEED
SMOKE_BOOTSTRAP_REPLICATES = V10_1.SMOKE_BOOTSTRAP_REPLICATES
STAGE_A_FIREWALL = V10_1.STAGE_A_FIREWALL
SMOOTH_REQUIRED_DELTA = V10_1.SMOOTH_REQUIRED_DELTA
RECONSTRUCTION_MAX_DELTA = 0.005
SELECTION_ROWS = V10_1.SELECTION_ROWS
SELECTION_PER_CLASS = V10_1.SELECTION_PER_CLASS
SELECTION_SEED = V10_1.SELECTION_SEED
SPLIT_SEED = V10_1.SPLIT_SEED
S_TAU = V10_1.S_TAU
LAMBDA_TRANSFER = V10_1.LAMBDA_TRANSFER
V3_ENCODER_SHA256 = V10_1.V3_ENCODER_SHA256
V4_GENERATOR_TENSOR_SHA256 = V10_1.V4_GENERATOR_TENSOR_SHA256
PROTOTYPE_SHA256 = V10_1.PROTOTYPE_SHA256
SELECTION_INDICES_SHA256 = V10_1.SELECTION_INDICES_SHA256
SELECTION_LABELS_SHA256 = V10_1.SELECTION_LABELS_SHA256
ADAPTER_PARAMETER_NAMES = V10_1.ADAPTER_PARAMETER_NAMES

LowRankAdapter = V10_1.LowRankAdapter
LAMBDA_IMAGE: float | None = None
_SCIENTIFIC_DEV_IMAGE_LOSS: dict[int, float] = {}


def _patch_base_constants() -> None:
    names = (
        "TASK_ID", "OUTER_GATE", "SUPPORTED_CLASSIFICATION", "NOT_SUPPORTED_CLASSIFICATION",
        "EXPECTED_BRANCH", "EXPECTED_HEAD", "CONTRACT_PATH", "CONTRACT_SHA256", "V9_CONTRACT_PATH",
        "V9_CONTRACT_SHA256", "DEFAULT_PREFLIGHT_DIR", "DEFAULT_SCIENTIFIC_DIR", "PREFLIGHT_INPUT_NAME",
        "PREFLIGHT_REPORT_NAME", "PREFLIGHT_LINEAGE_NAME", "PREFLIGHT_CHECKPOINT_NAME", "PREFLIGHT_SYNTHETIC_OBS_NAME",
        "PREFLIGHT_SYNTHETIC_REPORT_NAME", "PREFLIGHT_SYNTHETIC_LINEAGE_NAME", "SCIENTIFIC_INPUT_NAME",
        "SCIENTIFIC_REPORT_NAME", "SCIENTIFIC_LINEAGE_NAME", "SCIENTIFIC_SUFFICIENT_NAME", "SCIENTIFIC_RAW_NAME",
        "REPORT_SCHEMA", "LINEAGE_SCHEMA", "INPUT_SCHEMA", "SUFFICIENT_SCHEMA", "CHECKPOINT_SCHEMA",
        "CLASS_COUNT", "OLS_PORT_COUNT", "ADAPTER_RANK", "INIT_SCALE", "LOGICAL_BATCH_SIZE", "PRODUCTION_GRID",
        "SMOKE_GRID", "PRECHECK_STEPS", "SCIENTIFIC_STEPS", "CHECKPOINT_STEPS", "LEARNING_RATE", "ADAM_BETAS",
        "ADAM_EPSILON", "WEIGHT_DECAY", "AMSGRAD", "BOOTSTRAP_REPLICATES", "BOOTSTRAP_SEED",
        "SMOKE_BOOTSTRAP_REPLICATES", "STAGE_A_FIREWALL", "SMOOTH_REQUIRED_DELTA", "SELECTION_ROWS",
        "SELECTION_PER_CLASS", "SELECTION_SEED", "SPLIT_SEED", "S_TAU", "LAMBDA_TRANSFER", "V3_ENCODER_SHA256",
        "V4_GENERATOR_TENSOR_SHA256", "PROTOTYPE_SHA256", "SELECTION_INDICES_SHA256", "SELECTION_LABELS_SHA256",
        "ADAPTER_PARAMETER_NAMES",
    )
    for name in names:
        setattr(V10, name, globals()[name])


def _load_contract(root: Path) -> tuple[dict[str, Any], Any]:
    V10._require(CONTRACT_SHA256 != "__TO_BE_BOUND_AFTER_CONTRACT_MATERIALIZATION__", "V11 contract SHA is not bound")
    path = V10._path(root, CONTRACT_PATH)
    V10._validate_hash(path, CONTRACT_SHA256, "V11 contract")
    payload = V10._read_json(path, "V11 contract")
    V10._require(
        payload.get("schema") == "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract.v11"
        and payload.get("result") == "READY"
        and payload.get("status") == "valid_completed"
        and payload.get("outer_gate") == OUTER_GATE,
        "V11 contract is not READY",
    )
    definition = V10._mapping(payload.get("definition_entrypoint"), "V11 definition entrypoint")
    V10._validate_hash(V10._path(root, str(definition["path"])), str(definition["sha256"]), "V11 definition entrypoint")
    git = V10._mapping(payload.get("git"), "V11 contract Git")
    V10._require(git.get("branch") == EXPECTED_BRANCH and git.get("head") == EXPECTED_HEAD, "V11 contract Git identity changed")
    supersedes = V10._mapping(payload.get("supersedes"), "V10.1 supersession")
    V10._require(
        supersedes.get("semantic_authority") == "DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10_1"
        and supersedes.get("path") == V10_1_CONTRACT_PATH.as_posix()
        and supersedes.get("sha256") == V10_1_CONTRACT_SHA256,
        "V11 supersession binding changed",
    )
    parent = V10._mapping(payload.get("parent_contract"), "V9 parent contract")
    V10._require(parent.get("path") == V9_CONTRACT_PATH.as_posix() and parent.get("sha256") == V9_CONTRACT_SHA256, "V9 parent binding changed")
    v2_contract = V10.load_v2_training_contract(V10._path(root, V10.V6.V2_CONTRACT_PATH), repo_root=root)
    V10._require(v2_contract.sha256 == V10.V6.V2_CONTRACT_SHA256, "V2 contract identity changed")
    objective = V10._mapping(payload.get("objective"), "V11 objective")
    V10._require(float(objective.get("lambda_transfer", 0.0)) == LAMBDA_TRANSFER, "V11 semantic objective constant changed")
    image_rule = V10._mapping(objective.get("lambda_image"), "V11 lambda_image rule")
    V10._require(image_rule.get("formula") == "g_sem/(g_image+1e-12)" and image_rule.get("freeze_step") == 0 and image_rule.get("clipping") is False and image_rule.get("tuning") is False and image_rule.get("sweep") is False, "V11 lambda_image rule changed")
    adapter = V10._mapping(payload.get("adapter"), "V11 adapter")
    V10._require(int(adapter.get("rank", 0)) == ADAPTER_RANK, "V11 adapter rank changed")
    initialization = V10._mapping(adapter.get("initialization"), "V10.1 initialization")
    V10._require(initialization.get("v_real") == "all zeros" and initialization.get("support_rows_V") == [], "V10.1 asymmetric initialization changed")
    protocol = V10._mapping(payload.get("protocol"), "V11 protocol")
    V10._require(protocol.get("trainable") == ADAPTER_PARAMETER_NAMES, "V11 trainable protocol changed")
    return payload, v2_contract


def _total_gradient_norm(norms: Mapping[str, Any]) -> float:
    values = [float(norms[f"adapter.{name}"]) for name in ("u_real", "u_imag", "v_real", "v_imag")]
    total = math.sqrt(sum(value * value for value in values))
    V10._require(math.isfinite(total), "V11 total adapter gradient is not finite")
    return total


def _gradient_norms(adapter: Any, gradients: Mapping[str, torch.Tensor | None], prefix: str) -> dict[str, Any]:
    norms = V10._factor_gradient_norms(adapter, gradients, prefix)
    norms["total_adapter"] = _total_gradient_norm(norms)
    return norms


def _gradient_map(loss: torch.Tensor, adapter: Any, *, retain_graph: bool) -> dict[str, torch.Tensor | None]:
    gradients = torch.autograd.grad(loss, tuple(adapter.parameters()), retain_graph=retain_graph, allow_unused=False)
    return {f"adapter.{name}": value for name, value in zip(("u_real", "u_imag", "v_real", "v_imag"), gradients, strict=True)}


def _initial_gradient_balance(bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    global LAMBDA_IMAGE
    sampler = V10.DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    batch_indices, batch_labels = sampler.next_batch()
    batch = images.index_select(0, batch_indices).to(device)
    batch_labels = batch_labels.to(device)
    adapter = LowRankAdapter(device=device)
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V10.V8._load_runtime_state(consumer, bundle["initial_state"])
    V10._configure_adapter(consumer, adapter)
    components = _BASE_OBJECTIVE_COMPONENTS(consumer, adapter, batch, batch_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), float(bundle["s_tau"]), require_grad=True)
    semantic_loss = components["anchor_loss"] + LAMBDA_TRANSFER * components["transfer_loss"]
    image_loss = (components["raw_base"] - components["target"]).square().mean()
    V10._finite(semantic_loss, "V11 semantic gradient loss")
    V10._finite(image_loss, "V11 image gradient loss")

    anchor_norms = _gradient_norms(adapter, _gradient_map(components["anchor_loss"], adapter, retain_graph=True), "V11 initial anchor")
    transfer_norms = _gradient_norms(adapter, _gradient_map(components["transfer_loss"], adapter, retain_graph=True), "V11 initial transfer")
    semantic_norms = _gradient_norms(adapter, _gradient_map(semantic_loss, adapter, retain_graph=True), "V11 initial semantic")
    image_norms = _gradient_norms(adapter, _gradient_map(image_loss, adapter, retain_graph=True), "V11 initial image")
    g_sem = semantic_norms["total_adapter"]
    g_image = image_norms["total_adapter"]
    V10._require(math.isfinite(g_sem) and math.isfinite(g_image) and g_sem > 0.0 and g_image > 0.0, "V11 semantic/image gradient balance is undefined or nonfinite")
    computed_lambda = g_sem / (g_image + 1.0e-12)
    V10._require(math.isfinite(computed_lambda) and computed_lambda > 0.0, "V11 lambda_image is undefined or nonfinite")
    if LAMBDA_IMAGE is None:
        LAMBDA_IMAGE = computed_lambda
    else:
        V10._require(math.isclose(LAMBDA_IMAGE, computed_lambda, rel_tol=1.0e-12, abs_tol=1.0e-12), "V11 lambda_image changed between preflight and scientific entry")
    total_loss = semantic_loss + LAMBDA_IMAGE * image_loss
    combined_norms = _gradient_norms(adapter, _gradient_map(total_loss, adapter, retain_graph=False), "V11 initial combined")
    V10._require(combined_norms["total_adapter"] > 0.0, "V11 total adapter gradient is zero")
    return {
        "first_batch_indices_sha256": V10.V6.tensor_sha256(batch_indices),
        "first_batch_labels_sha256": V10.V6.tensor_sha256(batch_labels),
        "batch_size": int(batch.shape[0]),
        "seed": SPLIT_SEED,
        "objective": "L_V11=L_sem+lambda_image*L_image",
        "semantic_objective": "L_sem=L_anchor+lambda_transfer*L_transfer",
        "g_sem": g_sem,
        "g_image": g_image,
        "lambda_image": LAMBDA_IMAGE,
        "anchor": {"gradient_norms": anchor_norms, "grad_U_may_be_zero": True},
        "transfer": {"gradient_norms": transfer_norms, "grad_U_may_be_zero": True},
        "semantic": {"gradient_norms": semantic_norms},
        "image": {"gradient_norms": image_norms},
        "combined": {"gradient_norms": combined_norms, "usable_direction": True},
    }


def _objective_components(consumer: Any, adapter: Any, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, s_tau: float, *, require_grad: bool) -> dict[str, Any]:
    V10._require(LAMBDA_IMAGE is not None and math.isfinite(LAMBDA_IMAGE), "V11 lambda_image is not bound")
    components = _BASE_OBJECTIVE_COMPONENTS(consumer, adapter, batch, labels, prototypes, templates, s_tau, require_grad=require_grad)
    image_loss = (components["raw_base"] - components["target"]).square().mean()
    semantic_loss = components["anchor_loss"] + LAMBDA_TRANSFER * components["transfer_loss"]
    total_loss = semantic_loss + LAMBDA_IMAGE * image_loss
    V10._finite(image_loss, "V11 image loss")
    V10._finite(semantic_loss, "V11 semantic loss")
    V10._finite(total_loss, "V11 total loss")
    return {**components, "image_loss": image_loss, "semantic_loss": semantic_loss, "total_loss": total_loss, "lambda_image": LAMBDA_IMAGE}


def _gradient_firewall_probe(bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    V10.V8._load_runtime_state(consumer, bundle["initial_state"])
    adapter = LowRankAdapter(device=device)
    firewall = V10._configure_adapter(consumer, adapter)
    indices = train_indices[:LOGICAL_BATCH_SIZE]
    batch = images.index_select(0, indices).to(device)
    batch_labels = labels.index_select(0, indices).to(device)
    components = _objective_components(consumer, adapter, batch, batch_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], require_grad=True)
    image_loss = components["image_loss"]
    V10._require(image_loss.requires_grad, "V11 image loss lost its adapter gradient path")
    image_norms = _gradient_norms(adapter, _gradient_map(image_loss, adapter, retain_graph=True), "V11 image firewall")
    semantic_norms = _gradient_norms(adapter, _gradient_map(components["semantic_loss"], adapter, retain_graph=True), "V11 semantic firewall")
    combined = _gradient_norms(adapter, _gradient_map(components["total_loss"], adapter, retain_graph=True), "V11 combined firewall")
    wrong_anchor = F.cross_entropy(components["anchor_logits"], (batch_labels + 1) % CLASS_COUNT) + LAMBDA_IMAGE * image_loss
    wrong_transfer = LAMBDA_TRANSFER * F.softplus(components["delta"]).mean() + LAMBDA_IMAGE * image_loss
    negative_anchor = _gradient_norms(adapter, _gradient_map(wrong_anchor, adapter, retain_graph=True), "V11 wrong-label firewall")
    negative_transfer = _gradient_norms(adapter, _gradient_map(wrong_transfer, adapter, retain_graph=False), "V11 wrong-direction firewall")
    V10._require(image_norms["total_adapter"] > 0.0 and semantic_norms["total_adapter"] > 0.0 and combined["total_adapter"] > 0.0 and negative_anchor["total_adapter"] > 0.0 and negative_transfer["total_adapter"] > 0.0, "V11 adapter gradient direction is unusable")
    V10._require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "V11 runtime gradient firewall failed")
    V10._require(all(parameter.grad is None for parameter in bundle["smooth"].parameters()), "V11 Smooth evaluator firewall failed")
    V10._require(not bundle["prototypes"].requires_grad and not bundle["templates"].requires_grad, "V11 support tensor firewall failed")
    return {"firewall": firewall, "objective": "L_V11=L_sem+lambda_image*L_image", "semantic_objective": "L_sem=L_anchor+lambda_transfer*L_transfer", "image_loss_requires_grad": True, "image_loss_gradient_contribution": "differentiated", "prototype_loss_gradient_contribution": 0.0, "smooth_evaluator_gradient_contribution": 0.0, "semantic_gradient_norms": semantic_norms, "image_gradient_norms": image_norms, "combined_gradient_norms": combined, "negative_controls": {"wrong_label_anchor": negative_anchor, "wrong_direction_transfer": negative_transfer}, "runtime_gradient_none": True, "prototype_loss_in_objective": False, "smooth_in_objective": False}


def _train_step(consumer: Any, adapter: Any, optimizer: torch.optim.Adam, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, s_tau: float, lambda_transfer: float, step: int) -> dict[str, Any]:
    runtime_before = V10._digest_tree(V10._cpu_tree(consumer.runtime.state_dict()))
    before = {name: parameter.detach().cpu().clone() for name, parameter in V10._adapter_named(adapter).items()}
    optimizer.zero_grad(set_to_none=True)
    for parameter in adapter.parameters():
        parameter.grad = None
    components = _objective_components(consumer, adapter, batch, labels, prototypes, templates, s_tau, require_grad=True)
    total_loss = components["total_loss"]
    V10._require(total_loss.requires_grad and components["semantic_loss"].requires_grad and components["image_loss"].requires_grad, "V11 differentiated objective path is detached")
    total_loss.backward()
    named = V10._adapter_named(adapter)
    gradients = {name: parameter.grad for name, parameter in named.items()}
    gradient_norms = _gradient_norms(adapter, gradients, "V11 train")
    V10._require(gradient_norms["total_adapter"] > 0.0, "V11 adapter total gradient is zero")
    V10._require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "V11 frozen runtime received a gradient")
    optimizer.step()
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    runtime_after = V10._digest_tree(V10._cpu_tree(consumer.runtime.state_dict()))
    V10._require(runtime_before == runtime_after, "V11 frozen runtime state changed")
    updates = {name: float((parameter.detach().cpu() - before[name]).to(torch.float64).square().sum().sqrt()) for name, parameter in named.items()}
    total_update = math.sqrt(sum(value * value for value in updates.values()))
    V10._require(total_update > 0.0, "V11 adapter update is zero")
    return {"step": step, "objective": "L_V11=L_sem+lambda_image*L_image", "semantic_objective": "L_sem=L_anchor+lambda_transfer*L_transfer", "lambda_image": LAMBDA_IMAGE, "L_anchor": float(components["anchor_loss"].detach().cpu()), "L_transfer": float(components["transfer_loss"].detach().cpu()), "L_sem": float(components["semantic_loss"].detach().cpu()), "L_image": float(components["image_loss"].detach().cpu()), "L_total": float(total_loss.detach().cpu()), "L_image_evaluation_only": float(components["image_loss"].detach().cpu()), "L_image_shift_evaluation_only": float((components["raw_shift"] - components["target"]).detach().square().mean().cpu()), "Delta_template_mean": float(components["delta"].detach().to(torch.float64).mean().cpu()), "gradient_norms": gradient_norms, "update_norms": updates, "total_update_norm": total_update, "image_loss_in_differentiated_objective": True, "prototype_loss_in_differentiated_objective": False, "smooth_in_differentiated_objective": False, "adapter_updated": True, "encoder_unchanged": True, "generator_unchanged": True, "ols_unchanged": True, "trainable_parameter_names": ADAPTER_PARAMETER_NAMES, "frozen_parameter_names": ["encoder.phase", "generator.phase", "OLS", "T=I_100", "W_OLS", "TRAIN prototypes", "TRAIN templates", "Smooth evaluator"], "runtime_state_sha256_after": runtime_after}


def _measure_train_batch(consumer: Any, adapter: Any, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, s_tau: float, lambda_transfer: float) -> dict[str, Any]:
    with torch.no_grad():
        components = _objective_components(consumer, adapter, batch, labels, prototypes, templates, s_tau, require_grad=False)
    image_shift = (components["raw_shift"] - components["target"]).square().mean()
    return {"L_anchor": float(components["anchor_loss"]), "L_transfer": float(components["transfer_loss"]), "L_sem": float(components["semantic_loss"]), "L_image": float(components["image_loss"]), "L_total": float(components["total_loss"]), "L_image_evaluation_only": float(components["image_loss"]), "L_image_shift_evaluation_only": float(image_shift), "Delta_template_mean": float(components["delta"].to(torch.float64).mean())}


def _checkpoint_payload(consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: Any, *, step: int, initial_runtime_digest: str, initial_adapter_digest: str, prototypes: torch.Tensor, templates: torch.Tensor, rng_state: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": CHECKPOINT_SCHEMA, "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "checkpoint_step": step, "runtime_state": V10._cpu_tree(consumer.runtime.state_dict()), "adapter_state": V10._cpu_tree(adapter.state_dict()), "optimizer_state": V10._cpu_tree(optimizer.state_dict()), "sampler_state": V10._cpu_tree(sampler.state_dict()), "rng_state": V10._cpu_tree(rng_state), "initial_runtime_state_sha256": initial_runtime_digest, "initial_adapter_state_sha256": initial_adapter_digest, "prototypes_sha256": V10.V6.tensor_sha256(prototypes), "templates_sha256": V10.V6.tensor_sha256(templates), "s_tau": S_TAU, "lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE, "objective": "L_V11=L_sem+lambda_image*L_image", "semantic_objective": "L_sem=L_anchor+lambda_transfer*L_transfer", "trainable_parameter_names": ADAPTER_PARAMETER_NAMES, "frozen_parameter_names": ["encoder.phase", "generator.phase", "OLS", "T=I_100", "W_OLS", "TRAIN prototypes", "TRAIN templates", "Smooth evaluator"], "image_loss_gradient_contribution": "differentiated", "prototype_loss_gradient_contribution": 0.0, "smooth_evaluator_gradient_contribution": 0.0, "data_roles": {"TRAIN": True, "DEV": True, "calibration": False, "protected": False, "official_test": False, "sealed": False}, "stage_b_executed": False, "scale_extraction_executed": False, "prior_shaping_executed": False, "generation_executed": False, "formal_promotion": False}


def _three_step_smoke(root: Path, bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device, target: Path) -> dict[str, Any]:
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V10.V8._load_runtime_state(consumer, bundle["initial_state"])
    adapter = LowRankAdapter(device=device)
    firewall = V10._configure_adapter(consumer, adapter)
    optimizer = V10._optimizer(adapter)
    initial_observation = V10._adapter_observation(adapter)
    sampler = V10.DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    traces = []
    observations = []
    for step in range(1, PRECHECK_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        batch = images.index_select(0, indices).to(device)
        traces.append(_train_step(consumer, adapter, optimizer, batch, batch_labels.to(device), bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], LAMBDA_TRANSFER, step))
        observations.append(V10._adapter_observation(adapter))
    one_step = {"initial_residual_frobenius": initial_observation["frobenius_norm_residual"], "after_step1_residual_frobenius": observations[0]["frobenius_norm_residual"], "after_step1_residual_max_abs": observations[0]["residual_max_abs"], "departed_from_identity": bool(observations[0]["frobenius_norm_residual"] > 0.0), "rank_after_step1": observations[0]["effective_residual_rank"]}
    V10._require(one_step["initial_residual_frobenius"] == 0.0 and one_step["departed_from_identity"], "V11 disposable step did not depart from identity")
    payload = _checkpoint_payload(consumer, adapter, optimizer, sampler, step=PRECHECK_STEPS, initial_runtime_digest=V10._digest_tree(bundle["initial_state"]), initial_adapter_digest=V10._digest_tree(V10._cpu_tree(LowRankAdapter(device=device).state_dict())), prototypes=bundle["prototypes"], templates=bundle["templates"], rng_state=V10.V6._capture_rng())
    payload["templates"] = bundle["templates"]
    checkpoint = V10._save_and_verify_checkpoint(root, target, payload, PREFLIGHT_CHECKPOINT_NAME, consumer, bundle["v2_contract"])
    loaded = V10._mapping(V10.safe_torch_load(target / PREFLIGHT_CHECKPOINT_NAME, expected_sha256=checkpoint["sha256"], map_location="cpu"), "V11 smoke checkpoint")
    resumed_sampler = V10.DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    resumed_sampler.load_state_dict(dict(V10._mapping(loaded["sampler_state"], "V11 smoke sampler state")))
    next_original, next_original_labels = sampler.next_batch()
    next_resumed, next_resumed_labels = resumed_sampler.next_batch()
    V10._require(torch.equal(next_original, next_resumed) and torch.equal(next_original_labels, next_resumed_labels), "V11 sampler resume identity changed")
    return {"firewall": firewall, "traces": traces, "adapter_observations": observations, "one_disposable_optimizer_step": one_step, "checkpoint": checkpoint, "resume_identity": {"runtime": True, "adapter": True, "optimizer": True, "rng": True, "sampler_next_batch": True}, "scientific_updates": 0, "scientific_meaning": "none"}


def _synthetic_acceptance(root: Path, bundle: Mapping[str, Any], device: torch.device, target: Path) -> dict[str, Any]:
    ramp = torch.linspace(0.0, 1.0, 28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)
    train_labels = torch.arange(CLASS_COUNT, dtype=torch.long)
    train_images = (ramp + train_labels.view(-1, 1, 1, 1).to(torch.float32) / 20.0).clamp(0.0, 1.0)
    eval_labels = train_labels.repeat_interleave(2)
    eval_images = train_images.index_select(0, eval_labels)
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    V10.V8._load_runtime_state(consumer, bundle["initial_state"])
    adapter = LowRankAdapter(device=device)
    firewall = V10._configure_adapter(consumer, adapter)
    optimizer = V10._optimizer(adapter)
    traces = []
    for step in range(1, PRECHECK_STEPS + 1):
        traces.append(_train_step(consumer, adapter, optimizer, train_images.to(device), train_labels.to(device), bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], LAMBDA_TRANSFER, step))
    evaluation = V10._evaluate_v10(consumer, adapter, bundle["smooth"], eval_images, eval_labels, bundle["prototypes"], bundle["templates"], bundle["s_tau"], step=PRECHECK_STEPS, train_trace=traces[-1])
    cyclic = V10._cyclic_transfer_v10(consumer, adapter, bundle["smooth"], eval_images, eval_labels, torch.arange(eval_images.shape[0], dtype=torch.long), bundle["prototypes"], bundle["templates"])
    observations = {"labels": eval_labels, "synthetic_anchor_distances": evaluation["_anchor_distances"], "synthetic_anchor_predictions": evaluation["_anchor_predictions"], "synthetic_anchor_margin": evaluation["_anchor_correct_margin"], "synthetic_ols_margin": evaluation["_margins"], "synthetic_smooth_logits": evaluation["_smooth_logits"], **cyclic["_observations"]}
    sufficient = V10._save_sufficient(root, target, observations, name=PREFLIGHT_SYNTHETIC_OBS_NAME, schema="lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_synthetic_sufficient_observations.v1")
    zero_delta = V10.V6._paired_bootstrap(torch.zeros_like(eval_labels, dtype=torch.float64), eval_labels, replicates=SMOKE_BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    report_payload = {"schema": "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_synthetic_report.v1", "scientific_meaning": "none", "training_updates": PRECHECK_STEPS, "objective": "L_V11=L_sem+lambda_image*L_image", "semantic_objective": "L_sem=L_anchor+lambda_transfer*L_transfer", "lambda_image": LAMBDA_IMAGE, "firewall": firewall, "identity": V10._identity_probe(consumer, LowRankAdapter(device=device), eval_images[:LOGICAL_BATCH_SIZE].to(device)), "traces": traces, "evaluation": {key: value for key, value in evaluation.items() if not key.startswith("_")}, "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")}, "zero_delta_bootstrap": zero_delta, "sufficient_observations": sufficient}
    report_path = target / PREFLIGHT_SYNTHETIC_REPORT_NAME
    V10.atomic_write_new_json(report_path, report_payload, artifact_kind="V11 synthetic report")
    report_ref = {"path": V10._relative(root, report_path), "sha256": V10.V7.file_sha256(report_path)}
    lineage_path = target / PREFLIGHT_SYNTHETIC_LINEAGE_NAME
    V10.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_synthetic_lineage.v1", "result": "VALID_COMPLETED", "report": report_ref, "sufficient_observations": sufficient, "claim_boundary": {"scientific_meaning": "none", "protected": False, "official_test": False, "sealed": False}}, artifact_kind="V11 synthetic lineage")
    V10._require(V10._read_json(report_path, "V11 synthetic report reload")["schema"] == report_payload["schema"], "V11 synthetic report reload changed")
    V10._require(V10._read_json(lineage_path, "V11 synthetic lineage reload")["report"]["sha256"] == report_ref["sha256"], "V11 synthetic lineage reload changed")
    return {"scientific_meaning": "none", "firewall": firewall, "traces": traces, "evaluation": {key: value for key, value in evaluation.items() if not key.startswith("_")}, "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")}, "identity": report_payload["identity"], "zero_delta_bootstrap": zero_delta, "sufficient_observations": sufficient, "report": report_ref, "lineage": {"path": V10._relative(root, lineage_path), "sha256": V10.V7.file_sha256(lineage_path)}, "report_lineage_reloaded": True}


def _production_identity(bundle: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V10.V8._load_runtime_state(consumer, bundle["initial_state"])
    adapter = LowRankAdapter(device=device)
    V10._configure_adapter(consumer, adapter, trainable=False)
    synthetic = torch.linspace(0.0, 1.0, LOGICAL_BATCH_SIZE * 28 * 28, device=device, dtype=torch.float32).reshape(LOGICAL_BATCH_SIZE, 1, 28, 28)
    with torch.no_grad():
        canonical_result = consumer.runtime.forward_from_image(synthetic)
        canonical_raw = V10.V8._validate_result(canonical_result, LOGICAL_BATCH_SIZE, require_grad=False)
        adapted = V10._forward_adapter(consumer, adapter, synthetic, require_grad=False)
        V10._require(torch.equal(canonical_result.z_ols.coefficients, adapted["z"]), "V11 production canonical coefficient identity changed")
        V10._require(torch.equal(adapted["z"], adapted["z_adapted"]), "V11 production coefficient identity changed")
        V10._require(torch.allclose(canonical_raw, adapted["raw_base"], atol=1.0e-6, rtol=1.0e-5), "V11 production canonical downstream identity changed")
    return {"executed": True, "grid": PRODUCTION_GRID, "canonical_vs_identity_adapter_raw_allclose": True, "coefficient_exact_equal": True, "raw_max_abs_diff": float((canonical_raw - adapted["raw_base"]).abs().max().item()), "runtime_unchanged": True, "adapter_unchanged": True}


def _save_cyclic_raw(root: Path, target: Path, cyclic: Mapping[str, Any]) -> dict[str, Any]:
    observations = cyclic["_observations"]
    path = target / SCIENTIFIC_RAW_NAME
    payload = {"schema": "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_cyclic_raw_outputs.v1", "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, **{key: value.cpu() if torch.is_tensor(value) else value for key, value in observations.items()}}
    V10.atomic_torch_save_new(path, payload, artifact_kind="V11 cyclic raw outputs")
    digest = V10.V7.file_sha256(path)
    loaded = V10._mapping(V10.safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V11 cyclic raw reload")
    V10._require(loaded.get("schema") == payload["schema"] and torch.equal(loaded["selection_indices"], observations["selection_indices"].cpu()), "V11 cyclic raw reload changed")
    return {"path": V10._relative(root, path), "sha256": digest, "reloaded": True}


def _evaluate_v10(consumer: Any, adapter: Any, evaluator: Any, images: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, s_tau: float, *, step: int, train_trace: Mapping[str, Any] | None) -> dict[str, Any]:
    result = _BASE_EVALUATE(consumer, adapter, evaluator, images, labels, prototypes, templates, s_tau, step=step, train_trace=train_trace)
    if step in (0, 300):
        _SCIENTIFIC_DEV_IMAGE_LOSS[step] = float(result["losses"]["L_image"])
    return result


def _classify(r0: float, r300: float, smooth_bootstrap: Mapping[str, Any], cyclic: Mapping[str, Any], raw_valid: bool) -> dict[str, Any]:
    V10._require(0 in _SCIENTIFIC_DEV_IMAGE_LOSS and 300 in _SCIENTIFIC_DEV_IMAGE_LOSS, "V11 DEV image statistic is undefined")
    image_delta = _SCIENTIFIC_DEV_IMAGE_LOSS[300] - _SCIENTIFIC_DEV_IMAGE_LOSS[0]
    natural_pass = r300 - r0 >= SMOOTH_REQUIRED_DELTA and float(smooth_bootstrap["ci_lower"]) > 0.0
    reconstruction_pass = image_delta <= RECONSTRUCTION_MAX_DELTA
    template_pass = bool(cyclic["template"]["pass"])
    smooth_pass = bool(cyclic["smooth"]["pass"])
    supported = bool(natural_pass and reconstruction_pass and template_pass and smooth_pass and raw_valid)
    return {"classification": SUPPORTED_CLASSIFICATION if supported else NOT_SUPPORTED_CLASSIFICATION, "outer_gate": OUTER_GATE, "supported": supported, "criteria": {"natural_smooth_improvement": natural_pass, "reconstruction_compatibility": reconstruction_pass, "known_cyclic_template_transfer_retained": template_pass, "known_cyclic_smooth_transfer_retained": smooth_pass, "raw_validity": raw_valid}, "observed": {"DEV_L_image_step0": _SCIENTIFIC_DEV_IMAGE_LOSS[0], "DEV_L_image_step300": _SCIENTIFIC_DEV_IMAGE_LOSS[300], "Delta_L_image_DEV": image_delta}, "thresholds": {"smooth_delta": SMOOTH_REQUIRED_DELTA, "smooth_ci_lower": 0.0, "DEV_image_delta": RECONSTRUCTION_MAX_DELTA}}


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    def transform(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("L_V10=L_anchor+lambda_transfer*L_transfer", "L_V11=L_sem+lambda_image*L_image")
        if isinstance(value, list):
            return [transform(item) for item in value]
        if isinstance(value, dict):
            transformed = {key: transform(item) for key, item in value.items()}
            if transformed.get("task_id") == TASK_ID:
                if isinstance(transformed.get("objective"), dict):
                    transformed["objective"]["formula"] = "L_V11=L_sem+lambda_image*L_image"
                    transformed["objective"]["semantic_objective"] = "L_sem=L_anchor+lambda_transfer*L_transfer"
                    transformed["objective"]["lambda_image"] = LAMBDA_IMAGE
                    transformed["objective"]["lambda_rule"] = "g_sem/(g_image+1e-12), frozen at step0 from the same initial TRAIN batch"
                    transformed["objective"]["image_loss_gradient_contribution"] = "differentiated"
                    transformed["objective"]["image_loss_role"] = "differentiated reconstruction compatibility term"
                if isinstance(transformed.get("protocol"), dict):
                    transformed["protocol"]["lambda_image"] = LAMBDA_IMAGE
                if isinstance(transformed.get("governance"), dict):
                    transformed["governance"]["image_loss_gradient"] = True
                traces = transformed.get("traces")
                train_metrics = transformed.get("train_metrics")
                if isinstance(traces, list) and isinstance(train_metrics, dict):
                    trace_by_step = {
                        str(item.get("step")): item
                        for item in traces
                        if isinstance(item, dict) and item.get("step") is not None
                    }
                    for step, record in train_metrics.items():
                        if not isinstance(record, dict):
                            continue
                        source = trace_by_step.get(str(step), {})
                        for key in ("L_sem", "L_image"):
                            if key not in record and key in source:
                                record[key] = source[key]
                        record.setdefault("lambda_image", LAMBDA_IMAGE)
                        record.setdefault("objective", "L_V11=L_sem+lambda_image*L_image")
                        record.setdefault("image_loss_in_differentiated_objective", True)
                if transformed.get("next") == "no_retry; V10 bounded canary does not authorize Stage B or generation":
                    transformed["next"] = "no_retry; V11 bounded canary does not authorize Stage B or generation"
            return transformed
        return value

    _BASE_ATOMIC_WRITE_JSON(path, transform(copy.deepcopy(dict(payload))), artifact_kind=artifact_kind)


def _preflight_input_payload(contract_ref: Mapping[str, Any], device_facts: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_preflight_input.v1", "version": 1, "task_id": TASK_ID, "contract": dict(contract_ref), "objective": "L_V11=L_sem+lambda_image*L_image", "semantic_objective": "L_sem=L_anchor+lambda_transfer*L_transfer", "lambda_image": LAMBDA_IMAGE, "trainable_parameter_names": ADAPTER_PARAMETER_NAMES, "frozen_gradient_paths": ["Smooth/frozen runtime", "prototypes as frozen shift support"], "adapter": {"rank": ADAPTER_RANK, "init_scale": INIT_SCALE, "initialization": "U0=INIT_SCALE*I_9; V0=0", "application": "z_adapted=z @ T_adapt.T"}, "runtime": {"device": dict(device_facts), "grid": PRODUCTION_GRID}, "one_shot": True, "scientific_300_step_executed": False, "protected": False, "official_test": False, "sealed": False}


def run_preflight(root: Path, output_dir: Path) -> dict[str, Any]:
    V10._require(not output_dir.exists(), f"refusing to overwrite V11 preflight output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = V10._status_lines(root)
    payload: dict[str, Any] = {"schema": "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_preflight_report.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V11_PREFLIGHT", "outer_gate": OUTER_GATE, "provenance": {"git_start": started}, "claim_boundary": {"engineering_preflight_executed": False, "scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}}
    try:
        V10._require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from V11 authorization")
        contract, v2_contract = _load_contract(root)
        bundle = V10._resolve_authorities(root, contract)
        bundle["v2_contract"] = v2_contract
        dll = V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        bundle["smooth"].to(device)
        images, labels, train_indices, _development_indices, dataset = V10.V6._load_roles(root)
        V10.V6._initial_rng()
        gradient_balance = _initial_gradient_balance(bundle, images, labels, train_indices, device)
        no_data = V10._production_no_data_smoke(bundle, device)
        production_identity = _production_identity(bundle, device)
        firewall_probe = _gradient_firewall_probe(bundle, images, labels, train_indices, device)
        identity_consumer = V10.V8._new_consumer(v2_contract, device, SMOKE_GRID)
        V10.V8._load_runtime_state(identity_consumer, bundle["initial_state"])
        identity_adapter = LowRankAdapter(device=device)
        V10._configure_adapter(identity_consumer, identity_adapter, trainable=False)
        identity = V10._identity_probe(identity_consumer, identity_adapter, images.index_select(0, train_indices[:LOGICAL_BATCH_SIZE]).to(device))
        smoke = _three_step_smoke(root, bundle, images, labels, train_indices, device, output_dir)
        synthetic = _synthetic_acceptance(root, bundle, device, output_dir)
        contract_ref = {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}
        input_path = output_dir / PREFLIGHT_INPUT_NAME
        V10.atomic_write_new_json(input_path, _preflight_input_payload(contract_ref, device_facts), artifact_kind="V11 preflight input")
        payload.update({"contract": contract_ref, "parent_contract": {"path": V9_CONTRACT_PATH.as_posix(), "sha256": V9_CONTRACT_SHA256}, "authorities": bundle["authorities"], "runtime": {"cuda_dll_search_path": dll, "device": device_facts, "propagation_grid": PRODUCTION_GRID}, "data_roles": dataset, "adapter": {"rank": ADAPTER_RANK, "initial": V10._adapter_observation(identity_adapter), "identity_probe": identity}, "objective": {"formula": "L_V11=L_sem+lambda_image*L_image", "semantic_objective": "L_sem=L_anchor+lambda_transfer*L_transfer", "s_tau": S_TAU, "lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE, "lambda_rule": "g_sem/(g_image+1e-12), frozen at step0 from the same initial TRAIN batch"}, "gradient_balance": gradient_balance, "production_no_data_smoke": no_data, "production_identity": production_identity, "gradient_firewall_probe": firewall_probe, "three_step_smoke": smoke, "synthetic_acceptance": synthetic, "input": {"path": V10._relative(root, input_path), "sha256": V10.V7.file_sha256(input_path)}, "preflight_checks": {"step0_identity": True, "semantic_image_gradient_balance": True, "usable_total_gradient": True, "one_step_departure": smoke["one_disposable_optimizer_step"]["departed_from_identity"], "frozen_encoder_generator": True, "checkpoint_resume": True, "exact_core_synthetic": True}, "result": "AUTHORIZABLE", "status": "valid_completed", "classification": PREFLIGHT_CLASSIFICATION, "claim_boundary": {"engineering_preflight_executed": True, "scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}, "next": "proceed directly to the one-shot 300-step V11 scientific execution"})
    except V10.HardBlocked as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": error.classification, "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific execution"})
    except (V10.InvalidV10, V10.V9.InvalidV9, V10.V8.InvalidV8, V10.V7.InvalidV7, V10.V6.InvalidV6) as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V11_PREFLIGHT_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no scientific execution"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V11_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no scientific execution"})
    finally:
        ending = V10.V6.git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == V10._status_lines(root)})
    report_path = output_dir / PREFLIGHT_REPORT_NAME
    V10.atomic_write_new_json(report_path, payload, artifact_kind="V11 preflight report")
    report_ref = {"path": V10._relative(root, report_path), "sha256": V10.V7.file_sha256(report_path)}
    lineage_path = output_dir / PREFLIGHT_LINEAGE_NAME
    V10.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "provenance": payload["provenance"], "report": report_ref, "claim_boundary": payload["claim_boundary"]}, artifact_kind="V11 preflight lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": {"path": V10._relative(root, lineage_path), "sha256": V10.V7.file_sha256(lineage_path)}, "scientific_300_step_executed": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _validate_preflight(root: Path) -> dict[str, Any]:
    report_path = V10._path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = V10._path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    V10._require(report_path.is_file() and lineage_path.is_file(), "V11 preflight authority is unavailable")
    report = V10._read_json(report_path, "V11 preflight report")
    lineage = V10._read_json(lineage_path, "V11 preflight lineage")
    V10._require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed" and report.get("classification") == PREFLIGHT_CLASSIFICATION, "V11 preflight is not AUTHORIZABLE")
    V10._require(report.get("claim_boundary", {}).get("scientific_300_step_executed") is False and report.get("preflight_checks", {}).get("semantic_image_gradient_balance") is True, "V11 preflight acceptance boundary changed")
    bound_lambda = report.get("objective", {}).get("lambda_image")
    V10._require(isinstance(bound_lambda, (int, float)) and math.isfinite(float(bound_lambda)) and float(bound_lambda) > 0.0, "V11 preflight lambda_image is unavailable")
    lineage_hash = V10.V7.file_sha256(report_path)
    V10._require(lineage.get("result") == "AUTHORIZABLE" and lineage.get("report", {}).get("sha256") == lineage_hash, "V11 preflight lineage changed")
    return {"report": report, "report_ref": {"path": V10._relative(root, report_path), "sha256": lineage_hash}, "lineage_ref": {"path": V10._relative(root, lineage_path), "sha256": V10.V7.file_sha256(lineage_path)}}


def _load_lambda_from_preflight(root: Path) -> None:
    global LAMBDA_IMAGE
    preflight = _validate_preflight(root)
    LAMBDA_IMAGE = float(preflight["report"]["objective"]["lambda_image"])


def _patch_base() -> None:
    V10_1._patch_base()
    _patch_base_constants()
    V10.LowRankAdapter = LowRankAdapter
    V10._load_contract = _load_contract
    V10._objective_components = _objective_components
    V10._initial_gradient_balance = _initial_gradient_balance
    V10._gradient_firewall_probe = _gradient_firewall_probe
    V10._train_step = _train_step
    V10._measure_train_batch = _measure_train_batch
    V10._checkpoint_payload = _checkpoint_payload
    V10._three_step_smoke = _three_step_smoke
    V10._synthetic_acceptance = _synthetic_acceptance
    V10._save_cyclic_raw = _save_cyclic_raw
    V10._evaluate_v10 = _evaluate_v10
    V10._classify = _classify
    V10._validate_preflight = _validate_preflight
    V10.atomic_write_new_json = _atomic_write_new_json


def main() -> int:
    parser = V10.argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    _patch_base()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    if args.action == "preflight":
        result = run_preflight(root, (args.output_dir or root / DEFAULT_PREFLIGHT_DIR).resolve())
    else:
        _load_lambda_from_preflight(root)
        _SCIENTIFIC_DEV_IMAGE_LOSS.clear()
        result = V10.run_scientific(root, (args.output_dir or root / DEFAULT_SCIENTIFIC_DIR).resolve())
    return 0 if result["result"] in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
