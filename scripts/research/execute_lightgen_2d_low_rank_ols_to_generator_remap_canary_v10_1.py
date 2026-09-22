"""Execute the one-shot V10.1 low-rank remap canary.

This entrypoint reuses the validated V10/V9 runtime and scientific path.  Its
only model correction is the asymmetric identity initialization U0 != 0,
V0 = 0; the preflight gradient checks allow the mathematically expected
grad_U=0 at step zero while requiring a finite nonzero total direction.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


def _load_v10_execution() -> Any:
    path = Path(__file__).with_name("execute_lightgen_2d_low_rank_ols_to_generator_remap_canary_v10.py")
    spec = importlib.util.spec_from_file_location("lightgen_v10_execution_authority_v10_1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V10 execution authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V10 = _load_v10_execution()

TASK_ID = "EXECUTE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_V10_1"
OUTER_GATE = "V10_LOW_RANK_OLS_TO_GENERATOR_REMAP_SUPPORTED"
SUPPORTED_CLASSIFICATION = OUTER_GATE
NOT_SUPPORTED_CLASSIFICATION = "V10_LOW_RANK_OLS_TO_GENERATOR_REMAP_NOT_SUPPORTED"
PREFLIGHT_CLASSIFICATION = "LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_V10_1_AUTHORIZABLE"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

CONTRACT_PATH = Path(
    "outputs/20260912_v10_1/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_1_attempt_01/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10_1.json"
)
CONTRACT_SHA256 = "45210d92e00ba01255a4340dfc837c514c110640097aa20821d8d5b6e0efab3d"
V10_CONTRACT_PATH = Path(
    "outputs/20260912_v10/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_attempt_01/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10.json"
)
V10_CONTRACT_SHA256 = "5efeb9d4b8e5e1e29fee3e3c445a52a201fbfabdd7f696a183284c71832ca0f1"
V9_CONTRACT_PATH = V10.V9_CONTRACT_PATH
V9_CONTRACT_SHA256 = V10.V9_CONTRACT_SHA256

DEFAULT_PREFLIGHT_DIR = Path(
    "outputs/20260912_v10_1/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_1_attempt_01"
)
DEFAULT_SCIENTIFIC_DIR = Path(
    "outputs/20260912_v10_1/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_attempt_01"
)

PREFLIGHT_INPUT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_preflight_input.v1.json"
PREFLIGHT_REPORT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_preflight_lineage.v1.json"
PREFLIGHT_CHECKPOINT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_preflight_smoke_checkpoint.v1.pt"
PREFLIGHT_SYNTHETIC_OBS_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_synthetic_sufficient_observations.v1.pt"
PREFLIGHT_SYNTHETIC_REPORT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_synthetic_report.v1.json"
PREFLIGHT_SYNTHETIC_LINEAGE_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_synthetic_lineage.v1.json"

SCIENTIFIC_INPUT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_input.v1.json"
SCIENTIFIC_REPORT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_report.v1.json"
SCIENTIFIC_LINEAGE_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_lineage.v1.json"
SCIENTIFIC_SUFFICIENT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_sufficient_observations.v1.pt"
SCIENTIFIC_RAW_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_step300_cyclic_raw_outputs.v1.pt"

REPORT_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_lineage.v1"
INPUT_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_scientific_input.v1"
SUFFICIENT_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_sufficient_observations.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_checkpoint.v1"

CLASS_COUNT = V10.CLASS_COUNT
OLS_PORT_COUNT = V10.OLS_PORT_COUNT
ADAPTER_RANK = V10.ADAPTER_RANK
INIT_SCALE = V10.INIT_SCALE
LOGICAL_BATCH_SIZE = V10.LOGICAL_BATCH_SIZE
PRODUCTION_GRID = V10.PRODUCTION_GRID
SMOKE_GRID = V10.SMOKE_GRID
PRECHECK_STEPS = V10.PRECHECK_STEPS
SCIENTIFIC_STEPS = V10.SCIENTIFIC_STEPS
CHECKPOINT_STEPS = V10.CHECKPOINT_STEPS
LEARNING_RATE = V10.LEARNING_RATE
ADAM_BETAS = V10.ADAM_BETAS
ADAM_EPSILON = V10.ADAM_EPSILON
WEIGHT_DECAY = V10.WEIGHT_DECAY
AMSGRAD = V10.AMSGRAD
BOOTSTRAP_REPLICATES = V10.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = V10.BOOTSTRAP_SEED
SMOKE_BOOTSTRAP_REPLICATES = V10.SMOKE_BOOTSTRAP_REPLICATES
STAGE_A_FIREWALL = V10.STAGE_A_FIREWALL
SMOOTH_REQUIRED_DELTA = V10.SMOOTH_REQUIRED_DELTA
SELECTION_ROWS = V10.SELECTION_ROWS
SELECTION_PER_CLASS = V10.SELECTION_PER_CLASS
SELECTION_SEED = V10.SELECTION_SEED
SPLIT_SEED = V10.SPLIT_SEED
S_TAU = V10.S_TAU
LAMBDA_TRANSFER = V10.LAMBDA_TRANSFER
V3_ENCODER_SHA256 = V10.V3_ENCODER_SHA256
V4_GENERATOR_TENSOR_SHA256 = V10.V4_GENERATOR_TENSOR_SHA256
PROTOTYPE_SHA256 = V10.PROTOTYPE_SHA256
SELECTION_INDICES_SHA256 = V10.SELECTION_INDICES_SHA256
SELECTION_LABELS_SHA256 = V10.SELECTION_LABELS_SHA256
ADAPTER_PARAMETER_NAMES = V10.ADAPTER_PARAMETER_NAMES


def _patch_base_constants() -> None:
    for name in (
        "TASK_ID", "OUTER_GATE", "SUPPORTED_CLASSIFICATION", "NOT_SUPPORTED_CLASSIFICATION",
        "EXPECTED_BRANCH", "EXPECTED_HEAD", "CONTRACT_PATH", "CONTRACT_SHA256",
        "V9_CONTRACT_PATH", "V9_CONTRACT_SHA256", "DEFAULT_PREFLIGHT_DIR", "DEFAULT_SCIENTIFIC_DIR",
        "PREFLIGHT_INPUT_NAME", "PREFLIGHT_REPORT_NAME", "PREFLIGHT_LINEAGE_NAME", "PREFLIGHT_CHECKPOINT_NAME",
        "PREFLIGHT_SYNTHETIC_OBS_NAME", "PREFLIGHT_SYNTHETIC_REPORT_NAME", "PREFLIGHT_SYNTHETIC_LINEAGE_NAME",
        "SCIENTIFIC_INPUT_NAME", "SCIENTIFIC_REPORT_NAME", "SCIENTIFIC_LINEAGE_NAME", "SCIENTIFIC_SUFFICIENT_NAME",
        "SCIENTIFIC_RAW_NAME", "REPORT_SCHEMA", "LINEAGE_SCHEMA", "INPUT_SCHEMA", "SUFFICIENT_SCHEMA",
        "CHECKPOINT_SCHEMA", "CLASS_COUNT", "OLS_PORT_COUNT", "ADAPTER_RANK", "INIT_SCALE",
        "LOGICAL_BATCH_SIZE", "PRODUCTION_GRID", "SMOKE_GRID", "PRECHECK_STEPS", "SCIENTIFIC_STEPS",
        "CHECKPOINT_STEPS", "LEARNING_RATE", "ADAM_BETAS", "ADAM_EPSILON", "WEIGHT_DECAY", "AMSGRAD",
        "BOOTSTRAP_REPLICATES", "BOOTSTRAP_SEED", "SMOKE_BOOTSTRAP_REPLICATES", "STAGE_A_FIREWALL",
        "SMOOTH_REQUIRED_DELTA", "SELECTION_ROWS", "SELECTION_PER_CLASS", "SELECTION_SEED", "SPLIT_SEED",
        "S_TAU", "LAMBDA_TRANSFER", "V3_ENCODER_SHA256", "V4_GENERATOR_TENSOR_SHA256", "PROTOTYPE_SHA256",
        "SELECTION_INDICES_SHA256", "SELECTION_LABELS_SHA256", "ADAPTER_PARAMETER_NAMES",
    ):
        setattr(V10, name, globals()[name])


class LowRankAdapter(nn.Module):
    """Complex rank-nine adapter with exact identity at step zero."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.u_real = nn.Parameter(torch.zeros((OLS_PORT_COUNT, ADAPTER_RANK), device=device, dtype=torch.float32))
        self.u_imag = nn.Parameter(torch.zeros_like(self.u_real))
        self.v_real = nn.Parameter(torch.zeros_like(self.u_real))
        self.v_imag = nn.Parameter(torch.zeros_like(self.u_real))
        with torch.no_grad():
            eye = torch.eye(ADAPTER_RANK, device=device, dtype=torch.float32) * INIT_SCALE
            self.u_real[:ADAPTER_RANK].copy_(eye)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.complex(self.u_real, self.u_imag), torch.complex(self.v_real, self.v_imag)

    def matrix(self) -> torch.Tensor:
        u, v = self.factors()
        identity = torch.eye(OLS_PORT_COUNT, device=u.device, dtype=torch.complex64)
        return identity + u @ v.conj().transpose(0, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        V10._require(z.ndim == 2 and tuple(z.shape[1:]) == (OLS_PORT_COUNT,) and z.dtype == torch.complex64, "V10.1 adapter input shape/dtype changed")
        return z @ self.matrix().transpose(0, 1)


def _load_contract(root: Path) -> tuple[dict[str, Any], Any]:
    V10._require(CONTRACT_SHA256 != "__TO_BE_BOUND_AFTER_CONTRACT_MATERIALIZATION__", "V10.1 contract SHA is not bound")
    path = V10._path(root, CONTRACT_PATH)
    V10._validate_hash(path, CONTRACT_SHA256, "V10.1 contract")
    payload = V10._read_json(path, "V10.1 contract")
    V10._require(
        payload.get("schema") == "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10_1"
        and payload.get("result") == "READY"
        and payload.get("status") == "valid_completed"
        and payload.get("outer_gate") == OUTER_GATE,
        "V10.1 contract is not READY",
    )
    definition = V10._mapping(payload.get("definition_entrypoint"), "V10.1 definition entrypoint")
    V10._validate_hash(V10._path(root, str(definition["path"])), str(definition["sha256"]), "V10.1 definition entrypoint")
    git = V10._mapping(payload.get("git"), "V10.1 contract Git")
    V10._require(git.get("branch") == EXPECTED_BRANCH and git.get("head") == EXPECTED_HEAD, "V10.1 contract Git identity changed")
    supersedes = V10._mapping(payload.get("supersedes"), "V10 supersession")
    V10._require(
        supersedes.get("semantic_authority") == "DEFINE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_CONTRACT_V10"
        and supersedes.get("path") == V10_CONTRACT_PATH.as_posix()
        and supersedes.get("sha256") == V10_CONTRACT_SHA256,
        "V10.1 supersession binding changed",
    )
    parent = V10._mapping(payload.get("parent_contract"), "V9 parent contract")
    V10._require(parent.get("path") == V9_CONTRACT_PATH.as_posix() and parent.get("sha256") == V9_CONTRACT_SHA256, "V9 parent binding changed")
    v2_contract = V10.load_v2_training_contract(V10._path(root, V10.V6.V2_CONTRACT_PATH), repo_root=root)
    V10._require(v2_contract.sha256 == V10.V6.V2_CONTRACT_SHA256, "V2 contract identity changed")
    objective = V10._mapping(payload.get("objective"), "V10.1 objective")
    V10._require(float(objective.get("s_tau", 0.0)) == S_TAU and float(objective.get("lambda_transfer", 0.0)) == LAMBDA_TRANSFER, "V10.1 objective constants changed")
    adapter = V10._mapping(payload.get("adapter"), "V10.1 adapter")
    V10._require(int(adapter.get("rank", 0)) == ADAPTER_RANK, "V10.1 adapter rank changed")
    initialization = V10._mapping(adapter.get("initialization"), "V10.1 initialization")
    V10._require(initialization.get("v_real") == "all zeros" and initialization.get("support_rows_V") == [], "V10.1 asymmetric initialization changed")
    protocol = V10._mapping(payload.get("protocol"), "V10.1 protocol")
    V10._require(protocol.get("trainable") == ADAPTER_PARAMETER_NAMES, "V10.1 trainable protocol changed")
    return payload, v2_contract


def _total_gradient_norm(norms: Mapping[str, Any]) -> float:
    values = [float(norms[f"adapter.{name}"]) for name in ("u_real", "u_imag", "v_real", "v_imag")]
    total = math.sqrt(sum(value * value for value in values))
    V10._require(math.isfinite(total), "V10.1 total adapter gradient is not finite")
    return total


def _gradient_norms(adapter: LowRankAdapter, gradients: Mapping[str, torch.Tensor | None], prefix: str) -> dict[str, Any]:
    norms = V10._factor_gradient_norms(adapter, gradients, prefix)
    norms["total_adapter"] = _total_gradient_norm(norms)
    return norms


def _initial_gradient_balance(bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    sampler = V10.DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    batch_indices, batch_labels = sampler.next_batch()
    batch = images.index_select(0, batch_indices).to(device)
    batch_labels = batch_labels.to(device)
    adapter = LowRankAdapter(device=device)
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V10.V8._load_runtime_state(consumer, bundle["initial_state"])
    V10._configure_adapter(consumer, adapter)
    components = V10._objective_components(consumer, adapter, batch, batch_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), float(bundle["s_tau"]), require_grad=True)
    balance: dict[str, Any] = {"first_batch_indices_sha256": V10.V6.tensor_sha256(batch_indices), "first_batch_labels_sha256": V10.V6.tensor_sha256(batch_labels), "batch_size": int(batch.shape[0]), "seed": SPLIT_SEED, "objective": "L_V10_1=L_anchor+lambda_transfer*L_transfer", "anchor": {}, "transfer": {}}
    gradients_by_objective: dict[str, dict[str, torch.Tensor | None]] = {}
    for objective_name, loss in (("anchor", components["anchor_loss"]), ("transfer", components["transfer_loss"])):
        gradients = torch.autograd.grad(loss, tuple(adapter.parameters()), retain_graph=True, allow_unused=False)
        gradient_map = {f"adapter.{name}": value for name, value in zip(("u_real", "u_imag", "v_real", "v_imag"), gradients, strict=True)}
        norms = _gradient_norms(adapter, gradient_map, f"V10.1 initial {objective_name}")
        balance[objective_name] = {"gradient_norms": norms, "grad_U_may_be_zero": True}
        gradients_by_objective[objective_name] = gradient_map
    total_loss = components["anchor_loss"] + LAMBDA_TRANSFER * components["transfer_loss"]
    gradients = torch.autograd.grad(total_loss, tuple(adapter.parameters()), allow_unused=False)
    gradient_map = {f"adapter.{name}": value for name, value in zip(("u_real", "u_imag", "v_real", "v_imag"), gradients, strict=True)}
    combined = _gradient_norms(adapter, gradient_map, "V10.1 initial combined")
    V10._require(combined["total_adapter"] > 0.0, "V10.1 total adapter gradient is zero")
    balance["combined"] = {"gradient_norms": combined, "usable_direction": True}
    return balance


def _gradient_firewall_probe(bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    V10.V8._load_runtime_state(consumer, bundle["initial_state"])
    adapter = LowRankAdapter(device=device)
    firewall = V10._configure_adapter(consumer, adapter)
    indices = train_indices[:LOGICAL_BATCH_SIZE]
    batch = images.index_select(0, indices).to(device)
    batch_labels = labels.index_select(0, indices).to(device)
    components = V10._objective_components(consumer, adapter, batch, batch_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], require_grad=True)
    image_loss = (components["raw_base"].detach() - components["target"].detach()).square().mean()
    V10._require(not image_loss.requires_grad, "V10.1 image loss has a gradient path")
    named = V10._adapter_named(adapter)
    wrong_anchor = F.cross_entropy(components["anchor_logits"], (batch_labels + 1) % CLASS_COUNT)
    wrong_transfer = F.softplus(components["delta"]).mean()
    wrong_anchor_grad = torch.autograd.grad(wrong_anchor, tuple(adapter.parameters()), retain_graph=True)
    wrong_transfer_grad = torch.autograd.grad(wrong_transfer, tuple(adapter.parameters()), retain_graph=True)
    total = components["anchor_loss"] + LAMBDA_TRANSFER * components["transfer_loss"]
    total.backward()
    combined = _gradient_norms(adapter, {name: value.grad for name, value in named.items()}, "V10.1 combined probe")
    negative_anchor = _gradient_norms(adapter, {f"adapter.{name}": value for name, value in zip(("u_real", "u_imag", "v_real", "v_imag"), wrong_anchor_grad, strict=True)}, "V10.1 wrong-label probe")
    negative_transfer = _gradient_norms(adapter, {f"adapter.{name}": value for name, value in zip(("u_real", "u_imag", "v_real", "v_imag"), wrong_transfer_grad, strict=True)}, "V10.1 wrong-direction probe")
    V10._require(combined["total_adapter"] > 0.0 and negative_anchor["total_adapter"] > 0.0 and negative_transfer["total_adapter"] > 0.0, "V10.1 adapter gradient direction is unusable")
    V10._require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "V10.1 runtime gradient firewall failed")
    V10._require(all(parameter.grad is None for parameter in bundle["smooth"].parameters()), "V10.1 Smooth evaluator firewall failed")
    V10._require(not bundle["prototypes"].requires_grad and not bundle["templates"].requires_grad, "V10.1 support tensor firewall failed")
    return {"firewall": firewall, "objective": "L_V10_1=L_anchor+lambda_transfer*L_transfer", "image_loss_requires_grad": False, "image_loss_gradient_contribution": 0.0, "prototype_loss_gradient_contribution": 0.0, "smooth_evaluator_gradient_contribution": 0.0, "combined_gradient_norms": combined, "negative_controls": {"wrong_label_anchor": negative_anchor, "wrong_direction_transfer": negative_transfer}, "runtime_gradient_none": True, "prototype_loss_in_objective": False, "smooth_in_objective": False}


def _train_step(consumer: Any, adapter: LowRankAdapter, optimizer: torch.optim.Adam, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, s_tau: float, lambda_transfer: float, step: int) -> dict[str, Any]:
    runtime_before = V10._digest_tree(V10._cpu_tree(consumer.runtime.state_dict()))
    before = {name: parameter.detach().cpu().clone() for name, parameter in V10._adapter_named(adapter).items()}
    optimizer.zero_grad(set_to_none=True)
    for parameter in adapter.parameters():
        parameter.grad = None
    components = V10._objective_components(consumer, adapter, batch, labels, prototypes, templates, s_tau, require_grad=True)
    with torch.no_grad():
        image_loss = (components["raw_base"].detach() - components["target"].detach()).square().mean()
        shifted_image_loss = (components["raw_shift"].detach() - components["target"].detach()).square().mean()
    total_loss = components["anchor_loss"] + lambda_transfer * components["transfer_loss"]
    V10._finite(image_loss, "V10.1 TRAIN image loss")
    V10._finite(shifted_image_loss, "V10.1 TRAIN shifted image loss")
    V10._finite(total_loss, "V10.1 total loss")
    V10._require(total_loss.requires_grad and components["anchor_loss"].requires_grad and components["transfer_loss"].requires_grad, "V10.1 objective gradient path is detached")
    V10._require(not image_loss.requires_grad and not shifted_image_loss.requires_grad, "V10.1 image loss entered differentiated objective")
    total_loss.backward()
    named = V10._adapter_named(adapter)
    gradients = {name: parameter.grad for name, parameter in named.items()}
    gradient_norms = _gradient_norms(adapter, gradients, "V10.1 train")
    V10._require(gradient_norms["total_adapter"] > 0.0, "V10.1 adapter total gradient is zero")
    V10._require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "V10.1 frozen runtime received a gradient")
    optimizer.step()
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    runtime_after = V10._digest_tree(V10._cpu_tree(consumer.runtime.state_dict()))
    V10._require(runtime_before == runtime_after, "V10.1 frozen runtime state changed")
    updates = {name: float((parameter.detach().cpu() - before[name]).to(torch.float64).square().sum().sqrt()) for name, parameter in named.items()}
    total_update = math.sqrt(sum(value * value for value in updates.values()))
    V10._require(total_update > 0.0, "V10.1 adapter update is zero")
    return {"step": step, "objective": "L_V10_1=L_anchor+lambda_transfer*L_transfer", "L_anchor": float(components["anchor_loss"].detach().cpu()), "L_transfer": float(components["transfer_loss"].detach().cpu()), "L_total": float(total_loss.detach().cpu()), "L_image_evaluation_only": float(image_loss.cpu()), "L_image_shift_evaluation_only": float(shifted_image_loss.cpu()), "Delta_template_mean": float(components["delta"].detach().to(torch.float64).mean().cpu()), "gradient_norms": gradient_norms, "update_norms": updates, "total_update_norm": total_update, "image_loss_in_differentiated_objective": False, "prototype_loss_in_differentiated_objective": False, "smooth_in_differentiated_objective": False, "adapter_updated": True, "encoder_unchanged": True, "generator_unchanged": True, "ols_unchanged": True, "trainable_parameter_names": ADAPTER_PARAMETER_NAMES, "frozen_parameter_names": ["encoder.phase", "generator.phase", "OLS", "T=I_100", "W_OLS", "TRAIN prototypes", "TRAIN templates", "Smooth evaluator"], "runtime_state_sha256_after": runtime_after}


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
    V10._require(one_step["initial_residual_frobenius"] == 0.0 and one_step["departed_from_identity"], "V10.1 disposable step did not depart from identity")
    payload = V10._checkpoint_payload(consumer, adapter, optimizer, sampler, step=PRECHECK_STEPS, initial_runtime_digest=V10._digest_tree(bundle["initial_state"]), initial_adapter_digest=V10._digest_tree(V10._cpu_tree(LowRankAdapter(device=device).state_dict())), prototypes=bundle["prototypes"], templates=bundle["templates"], rng_state=V10.V6._capture_rng())
    payload["templates"] = bundle["templates"]
    checkpoint = V10._save_and_verify_checkpoint(root, target, payload, PREFLIGHT_CHECKPOINT_NAME, consumer, bundle["v2_contract"])
    loaded = V10._mapping(V10.safe_torch_load(target / PREFLIGHT_CHECKPOINT_NAME, expected_sha256=checkpoint["sha256"], map_location="cpu"), "V10.1 smoke checkpoint")
    resumed_sampler = V10.DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    resumed_sampler.load_state_dict(dict(V10._mapping(loaded["sampler_state"], "V10.1 smoke sampler state")))
    next_original, next_original_labels = sampler.next_batch()
    next_resumed, next_resumed_labels = resumed_sampler.next_batch()
    V10._require(torch.equal(next_original, next_resumed) and torch.equal(next_original_labels, next_resumed_labels), "V10.1 sampler resume identity changed")
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
    sufficient = V10._save_sufficient(root, target, observations, name=PREFLIGHT_SYNTHETIC_OBS_NAME, schema="lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_synthetic_sufficient_observations.v1")
    zero_delta = V10.V6._paired_bootstrap(torch.zeros_like(eval_labels, dtype=torch.float64), eval_labels, replicates=SMOKE_BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    report_payload = {"schema": "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_synthetic_report.v1", "scientific_meaning": "none", "training_updates": PRECHECK_STEPS, "objective": "L_V10_1=L_anchor+lambda_transfer*L_transfer", "firewall": firewall, "identity": V10._identity_probe(consumer, LowRankAdapter(device=device), eval_images[:LOGICAL_BATCH_SIZE].to(device)), "traces": traces, "evaluation": {key: value for key, value in evaluation.items() if not key.startswith("_")}, "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")}, "zero_delta_bootstrap": zero_delta, "sufficient_observations": sufficient}
    report_path = target / PREFLIGHT_SYNTHETIC_REPORT_NAME
    V10.atomic_write_new_json(report_path, report_payload, artifact_kind="V10.1 synthetic report")
    report_ref = {"path": V10._relative(root, report_path), "sha256": V10.V7.file_sha256(report_path)}
    lineage_path = target / PREFLIGHT_SYNTHETIC_LINEAGE_NAME
    V10.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_synthetic_lineage.v1", "result": "VALID_COMPLETED", "report": report_ref, "sufficient_observations": sufficient, "claim_boundary": {"scientific_meaning": "none", "protected": False, "official_test": False, "sealed": False}}, artifact_kind="V10.1 synthetic lineage")
    V10._require(V10._read_json(report_path, "V10.1 synthetic report reload")["schema"] == report_payload["schema"], "V10.1 synthetic report reload changed")
    V10._require(V10._read_json(lineage_path, "V10.1 synthetic lineage reload")["report"]["sha256"] == report_ref["sha256"], "V10.1 synthetic lineage reload changed")
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
        V10._require(torch.equal(canonical_result.z_ols.coefficients, adapted["z"]), "V10.1 production canonical coefficient identity changed")
        V10._require(torch.equal(adapted["z"], adapted["z_adapted"]), "V10.1 production coefficient identity changed")
        raw_allclose = torch.allclose(canonical_raw, adapted["raw_base"], atol=1.0e-6, rtol=1.0e-5)
        V10._require(raw_allclose, "V10.1 production canonical downstream identity changed")
    return {"executed": True, "grid": PRODUCTION_GRID, "canonical_vs_identity_adapter_raw_allclose": True, "coefficient_exact_equal": True, "raw_max_abs_diff": float((canonical_raw - adapted["raw_base"]).abs().max().item()), "runtime_unchanged": True, "adapter_unchanged": True}


def _preflight_input_payload(contract_ref: Mapping[str, Any], device_facts: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_preflight_input.v1", "version": 1, "task_id": TASK_ID, "contract": dict(contract_ref), "objective": "L_V10_1=L_anchor+lambda_transfer*L_transfer", "trainable_parameter_names": ADAPTER_PARAMETER_NAMES, "frozen_gradient_paths": ["L_image", "L_proto", "Smooth/frozen runtime"], "adapter": {"rank": ADAPTER_RANK, "init_scale": INIT_SCALE, "initialization": "U0=INIT_SCALE*I_9; V0=0", "application": "z_adapted=z @ T_adapt.T"}, "runtime": {"device": dict(device_facts), "grid": PRODUCTION_GRID}, "one_shot": True, "scientific_300_step_executed": False, "protected": False, "official_test": False, "sealed": False}


def run_preflight(root: Path, output_dir: Path) -> dict[str, Any]:
    V10._require(not output_dir.exists(), f"refusing to overwrite V10.1 preflight output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = V10._status_lines(root)
    payload: dict[str, Any] = {"schema": "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_preflight_report.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": V10.datetime.now(V10.timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V10_1_PREFLIGHT", "outer_gate": OUTER_GATE, "provenance": {"git_start": started}, "claim_boundary": {"engineering_preflight_executed": False, "scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}}
    try:
        V10._require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from V10.1 authorization")
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
        V10.atomic_write_new_json(input_path, _preflight_input_payload(contract_ref, device_facts), artifact_kind="V10.1 preflight input")
        payload.update({"contract": contract_ref, "parent_contract": {"path": V9_CONTRACT_PATH.as_posix(), "sha256": V9_CONTRACT_SHA256}, "authorities": bundle["authorities"], "runtime": {"cuda_dll_search_path": dll, "device": device_facts, "propagation_grid": PRODUCTION_GRID}, "data_roles": dataset, "adapter": {"rank": ADAPTER_RANK, "initial": V10._adapter_observation(identity_adapter), "identity_probe": identity}, "objective": {"s_tau": S_TAU, "lambda_transfer": LAMBDA_TRANSFER, "formula": "L_V10_1=L_anchor+lambda_transfer*L_transfer"}, "gradient_balance": gradient_balance, "production_no_data_smoke": no_data, "production_identity": production_identity, "gradient_firewall_probe": firewall_probe, "three_step_smoke": smoke, "synthetic_acceptance": synthetic, "input": {"path": V10._relative(root, input_path), "sha256": V10.V7.file_sha256(input_path)}, "preflight_checks": {"step0_identity": True, "usable_total_gradient": True, "one_step_departure": smoke["one_disposable_optimizer_step"]["departed_from_identity"], "frozen_encoder_generator": True, "checkpoint_resume": True}, "result": "AUTHORIZABLE", "status": "valid_completed", "classification": PREFLIGHT_CLASSIFICATION, "claim_boundary": {"engineering_preflight_executed": True, "scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}, "next": "proceed directly to the one-shot 300-step V10.1 scientific execution"})
    except V10.HardBlocked as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": error.classification, "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific execution"})
    except (V10.InvalidV10, V10.V9.InvalidV9, V10.V8.InvalidV8, V10.V7.InvalidV7, V10.V6.InvalidV6) as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V10_1_PREFLIGHT_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no scientific execution"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V10_1_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no scientific execution"})
    finally:
        ending = V10.V6.git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == V10._status_lines(root)})
    report_path = output_dir / PREFLIGHT_REPORT_NAME
    V10.atomic_write_new_json(report_path, payload, artifact_kind="V10.1 preflight report")
    report_ref = {"path": V10._relative(root, report_path), "sha256": V10.V7.file_sha256(report_path)}
    lineage_path = output_dir / PREFLIGHT_LINEAGE_NAME
    V10.atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_1_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "provenance": payload["provenance"], "report": report_ref, "claim_boundary": payload["claim_boundary"]}, artifact_kind="V10.1 preflight lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": {"path": V10._relative(root, lineage_path), "sha256": V10.V7.file_sha256(lineage_path)}, "scientific_300_step_executed": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _validate_preflight(root: Path) -> dict[str, Any]:
    report_path = V10._path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = V10._path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    V10._require(report_path.is_file() and lineage_path.is_file(), "V10.1 preflight authority is unavailable")
    report = V10._read_json(report_path, "V10.1 preflight report")
    lineage = V10._read_json(lineage_path, "V10.1 preflight lineage")
    V10._require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed" and report.get("classification") == PREFLIGHT_CLASSIFICATION, "V10.1 preflight is not AUTHORIZABLE")
    V10._require(report.get("claim_boundary", {}).get("scientific_300_step_executed") is False and report.get("preflight_checks", {}).get("one_step_departure") is True, "V10.1 preflight acceptance boundary changed")
    V10._require(lineage.get("result") == "AUTHORIZABLE" and lineage.get("report", {}).get("sha256") == V10.V7.file_sha256(report_path), "V10.1 preflight lineage changed")
    return {"report": report, "report_ref": {"path": V10._relative(root, report_path), "sha256": V10.V7.file_sha256(report_path)}, "lineage_ref": {"path": V10._relative(root, lineage_path), "sha256": V10.V7.file_sha256(lineage_path)}}


def _patch_base() -> None:
    _patch_base_constants()
    V10.LowRankAdapter = LowRankAdapter
    V10._load_contract = _load_contract
    V10._initial_gradient_balance = _initial_gradient_balance
    V10._gradient_firewall_probe = _gradient_firewall_probe
    V10._train_step = _train_step
    V10._three_step_smoke = _three_step_smoke
    V10._synthetic_acceptance = _synthetic_acceptance
    V10._preflight_input_payload = _preflight_input_payload
    V10._validate_preflight = _validate_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    _patch_base()
    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    if args.action == "preflight":
        result = run_preflight(root, (args.output_dir or root / DEFAULT_PREFLIGHT_DIR).resolve())
    else:
        result = V10.run_scientific(root, (args.output_dir or root / DEFAULT_SCIENTIFIC_DIR).resolve())
    return 0 if result["result"] in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
