"""Execute the one-shot V10 low-rank OLS-to-generator remap canary.

The runtime is imported from the validated V9/V8 research chain.  V10 owns
only the external rank-nine adapter, its trainable/frozen firewall, bounded
preflight, evidence artifacts, and the single authorized 300-update arm.
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
from torch import nn
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import DeterministicTrainRoleSampler, load_v2_training_contract
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load


def _load_v9_execution() -> Any:
    path = Path(__file__).with_name("execute_lightgen_2d_joint_semantic_remapping_warmup_v9.py")
    spec = importlib.util.spec_from_file_location("lightgen_v9_execution_authority_v10", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V9 execution authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V9 = _load_v9_execution()
V8 = V9.V8
V6 = V9.V6
V7 = V9.V7

TASK_ID = "EXECUTE_LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_V10"
OUTER_GATE = "V10_LOW_RANK_OLS_TO_GENERATOR_REMAP_SUPPORTED"
SUPPORTED_CLASSIFICATION = OUTER_GATE
NOT_SUPPORTED_CLASSIFICATION = "V10_LOW_RANK_OLS_TO_GENERATOR_REMAP_NOT_SUPPORTED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

CONTRACT_PATH = Path(
    "outputs/20260912_v10/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract_v10_attempt_01/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10.json"
)
CONTRACT_SHA256 = "5efeb9d4b8e5e1e29fee3e3c445a52a201fbfabdd7f696a183284c71832ca0f1"
V9_CONTRACT_PATH = Path(
    "outputs/20260912_v9/runs/"
    "lightgen_2d_joint_semantic_remapping_warmup_contract_v9_attempt_01/"
    "lightgen_2d_joint_semantic_remapping_warmup_contract.v9.json"
)
V9_CONTRACT_SHA256 = "4d0fb81bfa6152995aad4484c83487c20335616283097b394757f4616bf41fca"

DEFAULT_PREFLIGHT_DIR = Path(
    "outputs/20260912_v10/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_preflight_v10_attempt_01"
)
DEFAULT_SCIENTIFIC_DIR = Path(
    "outputs/20260912_v10/runs/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_scientific_attempt_01"
)

PREFLIGHT_INPUT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_input.v1.json"
PREFLIGHT_REPORT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_lineage.v1.json"
PREFLIGHT_CHECKPOINT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_smoke_checkpoint.v1.pt"
PREFLIGHT_SYNTHETIC_OBS_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_synthetic_sufficient_observations.v1.pt"
PREFLIGHT_SYNTHETIC_REPORT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_synthetic_report.v1.json"
PREFLIGHT_SYNTHETIC_LINEAGE_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_synthetic_lineage.v1.json"

SCIENTIFIC_INPUT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_scientific_input.v1.json"
SCIENTIFIC_REPORT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_scientific_report.v1.json"
SCIENTIFIC_LINEAGE_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_scientific_lineage.v1.json"
SCIENTIFIC_SUFFICIENT_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_sufficient_observations.v1.pt"
SCIENTIFIC_RAW_NAME = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_step300_cyclic_raw_outputs.v1.pt"

REPORT_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_scientific_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_scientific_lineage.v1"
INPUT_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_scientific_input.v1"
SUFFICIENT_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_sufficient_observations.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_checkpoint.v1"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
ADAPTER_RANK = 9
INIT_SCALE = 1.0e-2
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
V3_ENCODER_SHA256 = "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca"
V4_GENERATOR_TENSOR_SHA256 = "83aa74b701049c09c5efb63dbd6d3a6d181184a964d3d3c0c06665ca97da5afe"
PROTOTYPE_SHA256 = "c02a728e8631e24918b018553958a44b25cdb358192c6b4b28239f55e93df912"
SELECTION_INDICES_SHA256 = "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8"
SELECTION_LABELS_SHA256 = "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"
ADAPTER_PARAMETER_NAMES = ["adapter.u_real", "adapter.u_imag", "adapter.v_real", "adapter.v_imag"]


class InvalidV10(RuntimeError):
    """The V10 harness or immutable authority is invalid."""


class HardBlocked(RuntimeError):
    """A valid hard boundary prevents V10 execution."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidV10(message)


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
    magnitude = value.detach().abs() if torch.is_complex(value) else value.detach()
    result = float(magnitude.to(torch.float64).square().sum().sqrt().cpu())
    _require(math.isfinite(result), f"{name} norm is nonfinite")
    return result


def _status_lines(root: Path) -> list[str]:
    return V6._status_lines(root)


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidV10(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


def _validate_hash(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = V7.file_sha256(path)
    _require(observed == expected, f"{name} SHA mismatch: expected {expected}, observed {observed}")


def _load_contract(root: Path) -> tuple[dict[str, Any], Any]:
    _require(CONTRACT_SHA256 != "__TO_BE_BOUND_AFTER_CONTRACT_MATERIALIZATION__", "V10 contract SHA is not bound")
    path = _path(root, CONTRACT_PATH)
    _validate_hash(path, CONTRACT_SHA256, "V10 contract")
    payload = _read_json(path, "V10 contract")
    _require(payload.get("schema") == "lightgen_2d_low_rank_ols_to_generator_remap_canary_contract.v10" and payload.get("result") == "READY" and payload.get("status") == "valid_completed" and payload.get("outer_gate") == OUTER_GATE, "V10 contract is not READY")
    definition = _mapping(payload.get("definition_entrypoint"), "V10 definition entrypoint")
    _validate_hash(_path(root, str(definition["path"])), str(definition["sha256"]), "V10 definition entrypoint")
    git = _mapping(payload.get("git"), "V10 contract Git")
    _require(git.get("branch") == EXPECTED_BRANCH and git.get("head") == EXPECTED_HEAD, "V10 contract Git identity changed")
    parent = _mapping(payload.get("parent_contract"), "V9 parent contract")
    _require(parent.get("path") == V9_CONTRACT_PATH.as_posix() and parent.get("sha256") == V9_CONTRACT_SHA256, "V9 parent binding changed")
    v2_contract = load_v2_training_contract(_path(root, V6.V2_CONTRACT_PATH), repo_root=root)
    _require(v2_contract.sha256 == V6.V2_CONTRACT_SHA256, "V2 contract identity changed")
    objective = _mapping(payload.get("objective"), "V10 objective")
    _require(float(objective.get("s_tau", 0.0)) == S_TAU and float(objective.get("lambda_transfer", 0.0)) == LAMBDA_TRANSFER, "V10 objective constants changed")
    adapter = _mapping(payload.get("adapter"), "V10 adapter")
    _require(int(adapter.get("rank", 0)) == ADAPTER_RANK, "V10 adapter rank changed")
    protocol = _mapping(payload.get("protocol"), "V10 protocol")
    _require(protocol.get("trainable") == ADAPTER_PARAMETER_NAMES, "V10 trainable protocol changed")
    return payload, v2_contract


def _resolve_authorities(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    authorities = V8._resolve_authorities(root, contract)
    _require(tuple(authorities["initial_state"]["encoder.phase"].shape) == (2, 1, 100, 100), "V10 encoder shape changed")
    _require(tuple(authorities["initial_state"]["generator.phase"].shape) == (2, 1, 100, 100), "V10 generator shape changed")
    _require(V6.tensor_sha256(authorities["initial_state"]["encoder.phase"]) == V3_ENCODER_SHA256, "V10 initial encoder is not exact V3 CONTROL")
    _require(V6.tensor_sha256(authorities["initial_state"]["generator.phase"]) == V4_GENERATOR_TENSOR_SHA256, "V10 initial generator is not exact V4")
    _require(V6.tensor_sha256(authorities["prototypes"]) == PROTOTYPE_SHA256, "V10 prototypes changed")
    _require(not authorities["prototypes"].requires_grad and not authorities["templates"].requires_grad, "V10 support tensors require gradients")
    _require(math.isclose(float(authorities["s_tau"]), S_TAU, rel_tol=0.0, abs_tol=1.0e-15), "V10 s_tau recomputation changed")
    authorities["smooth"].eval()
    for parameter in authorities["smooth"].parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return authorities


class LowRankAdapter(nn.Module):
    """Complex U/V adapter with an exactly identity step-0 construction."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        self.u_real = nn.Parameter(torch.zeros((OLS_PORT_COUNT, ADAPTER_RANK), device=device, dtype=torch.float32))
        self.u_imag = nn.Parameter(torch.zeros_like(self.u_real))
        self.v_real = nn.Parameter(torch.zeros_like(self.u_real))
        self.v_imag = nn.Parameter(torch.zeros_like(self.u_real))
        with torch.no_grad():
            eye = torch.eye(ADAPTER_RANK, device=device, dtype=torch.float32) * INIT_SCALE
            self.u_real[:ADAPTER_RANK].copy_(eye)
            self.v_real[ADAPTER_RANK : 2 * ADAPTER_RANK].copy_(eye)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.complex(self.u_real, self.u_imag), torch.complex(self.v_real, self.v_imag)

    def matrix(self) -> torch.Tensor:
        u, v = self.factors()
        identity = torch.eye(OLS_PORT_COUNT, device=u.device, dtype=torch.complex64)
        return identity + u @ v.conj().transpose(0, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _require(z.ndim == 2 and tuple(z.shape[1:]) == (OLS_PORT_COUNT,) and z.dtype == torch.complex64, "adapter input shape/dtype changed")
        return z @ self.matrix().transpose(0, 1)


def _adapter_named(adapter: LowRankAdapter) -> dict[str, nn.Parameter]:
    return {f"adapter.{name}": parameter for name, parameter in adapter.named_parameters()}


def _configure_adapter(consumer: Any, adapter: LowRankAdapter, *, trainable: bool = True) -> dict[str, Any]:
    V8._configure_all_frozen(consumer)
    named = _adapter_named(adapter)
    for parameter in named.values():
        parameter.requires_grad_(trainable)
        parameter.grad = None
    _require(set(named) == set(ADAPTER_PARAMETER_NAMES), "adapter parameter surface changed")
    if trainable:
        _require(all(parameter.requires_grad for parameter in named.values()), "adapter trainable firewall failed")
    else:
        _require(all(not parameter.requires_grad for parameter in named.values()), "adapter frozen firewall failed")
    return {"trainable": ADAPTER_PARAMETER_NAMES if trainable else [], "frozen": ["encoder.phase", "generator.phase", "OLS", "T=I_100", "W_OLS", "templates", "prototypes", "Smooth", "all other runtime state"] + ([] if trainable else ADAPTER_PARAMETER_NAMES), "trainable_count": int(sum(parameter.numel() for parameter in named.values())) if trainable else 0, "adapter_parameter_counts": {key: int(value.numel()) for key, value in named.items()}, "runtime_parameters_all_frozen": True}


def _optimizer(adapter: LowRankAdapter) -> torch.optim.Adam:
    parameters = list(adapter.parameters())
    _require(all(parameter.requires_grad for parameter in parameters), "adapter optimizer received frozen parameter")
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    _require(len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["params"] == parameters, "adapter optimizer firewall changed")
    return optimizer


def _validate_adapted_result(result: Any, batch_size: int, *, require_grad: bool) -> torch.Tensor:
    _require(result.z_ols.source == "external_generated", "adapted OLS state lost external adapter provenance")
    z = result.z_ols.coefficients
    _require(tuple(z.shape) == (batch_size, OLS_PORT_COUNT) and z.dtype == torch.complex64, "adapted OLS coefficient shape/dtype changed")
    if require_grad:
        _require(z.requires_grad, "adapted OLS coefficients detached from adapter")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "adapted W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "fixed T=I_100 provenance changed")
    raw = result.raw_intensity
    _require(tuple(raw.shape) == (batch_size, 1, 200, 200) and raw.dtype == torch.float32, "adapted raw CMOS shape/dtype changed")
    _finite(raw, "adapted raw CMOS")
    return raw


def _forward_adapter(consumer: Any, adapter: LowRankAdapter, batch: torch.Tensor, *, require_grad: bool) -> dict[str, torch.Tensor | Any]:
    encoded_result = consumer.runtime.forward_from_image(batch)
    identity_raw = V8._validate_result(encoded_result, int(batch.shape[0]), require_grad=False)
    z = encoded_result.z_ols.coefficients
    _require(not z.requires_grad, "frozen encoder unexpectedly has an adapter-independent gradient")
    z_adapted = adapter(z)
    adapted_result = consumer.runtime.forward_from_ols(z_adapted)
    raw_base = _validate_adapted_result(adapted_result, int(batch.shape[0]), require_grad=require_grad)
    return {"encoded_result": encoded_result, "adapted_result": adapted_result, "identity_raw": identity_raw, "z": z, "z_adapted": z_adapted, "raw_base": raw_base, "target": encoded_result.image_input.slm_intensity}


def _adapter_observation(adapter: LowRankAdapter) -> dict[str, Any]:
    with torch.no_grad():
        matrix = adapter.matrix().detach().cpu().to(torch.complex128)
    identity = torch.eye(OLS_PORT_COUNT, dtype=torch.complex128)
    residual = matrix - identity
    _finite(residual.real, "adapter residual real")
    _finite(residual.imag, "adapter residual imag")
    singular = torch.linalg.svdvals(residual)
    _finite(singular, "adapter residual singular values")
    return {"frobenius_norm_residual": float(torch.linalg.vector_norm(residual).item()), "spectral_norm_residual": float(singular.max().item()), "effective_residual_rank": int(torch.linalg.matrix_rank(residual).item()), "residual_singular_values": [float(value) for value in singular.tolist()], "residual_max_abs": float(residual.abs().max().item())}


def _latent_adapter_metrics(z: torch.Tensor, z_adapted: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    z_cpu = z.detach().cpu().to(torch.complex128)
    adapted_cpu = z_adapted.detach().cpu().to(torch.complex128)
    labels_cpu = labels.detach().cpu()
    centroids: list[torch.Tensor] = []
    adapted_centroids: list[torch.Tensor] = []
    displacement = []
    for digit in range(CLASS_COUNT):
        mask = labels_cpu == digit
        _require(bool(mask.any()), f"adapter centroid class {digit} absent")
        base = z_cpu[mask].mean(dim=0)
        changed = adapted_cpu[mask].mean(dim=0)
        centroids.append(base)
        adapted_centroids.append(changed)
        displacement.append({"class": digit, "displacement_l2": float(torch.linalg.vector_norm(changed - base).item())})
    pairwise = []
    distortions = []
    for left in range(CLASS_COUNT):
        for right in range(left + 1, CLASS_COUNT):
            base_distance = float(torch.linalg.vector_norm(centroids[left] - centroids[right]).item())
            adapted_distance = float(torch.linalg.vector_norm(adapted_centroids[left] - adapted_centroids[right]).item())
            distortion = adapted_distance - base_distance
            relative = distortion / max(base_distance, 1.0e-12)
            record = {"left": left, "right": right, "base_distance": base_distance, "adapted_distance": adapted_distance, "distance_distortion": distortion, "relative_distance_distortion": relative}
            pairwise.append(record)
            distortions.append(distortion)
    distortion_tensor = torch.tensor(distortions, dtype=torch.float64)
    return {"per_class_centroid_displacement": displacement, "pairwise_centroid_distance_distortion": pairwise, "pairwise_distortion_summary": {"mean": float(distortion_tensor.mean()), "median": float(distortion_tensor.median()), "min": float(distortion_tensor.min()), "max": float(distortion_tensor.max()), "pair_count": int(distortion_tensor.numel())}}


def _objective_components(consumer: Any, adapter: LowRankAdapter, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, s_tau: float, *, require_grad: bool) -> dict[str, Any]:
    forward = _forward_adapter(consumer, adapter, batch, require_grad=require_grad)
    raw_base = forward["raw_base"]
    z = forward["z"]
    targets = (labels + 1) % CLASS_COUNT
    support = prototypes.to(device=z.device)
    z_shift = z - support.index_select(0, labels) + support.index_select(0, targets)
    z_shift_adapted = adapter(z_shift)
    shifted_result = consumer.runtime.forward_from_ols(z_shift_adapted)
    raw_shift = _validate_adapted_result(shifted_result, int(batch.shape[0]), require_grad=require_grad)
    distances = V8._image_distances(raw_base, templates)
    anchor_logits = -distances / (s_tau + 1.0e-12)
    anchor_loss = F.cross_entropy(anchor_logits, labels)
    base_margin, shifted_margin = V7._template_margins_differentiable(raw_base, raw_shift, templates, labels, targets)
    delta = shifted_margin - base_margin
    transfer_loss = F.softplus(-delta).mean()
    for name, value in (("anchor_loss", anchor_loss), ("transfer_loss", transfer_loss), ("delta", delta), ("raw_shift", raw_shift)):
        _finite(value, f"V10 {name}")
    return {**forward, "raw_shift": raw_shift, "z_shift": z_shift, "z_shift_adapted": z_shift_adapted, "targets": targets, "distances": distances, "anchor_logits": anchor_logits, "anchor_loss": anchor_loss, "base_margin": base_margin, "shifted_margin": shifted_margin, "delta": delta, "transfer_loss": transfer_loss}


def _factor_gradient_norms(adapter: LowRankAdapter, gradients: Mapping[str, torch.Tensor | None], prefix: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("u_real", "u_imag", "v_real", "v_imag"):
        gradient = gradients.get(f"adapter.{name}")
        _require(gradient is not None, f"{prefix} {name} gradient missing")
        result[f"adapter.{name}"] = _norm(gradient, f"{prefix} {name} gradient")
    u = torch.complex(gradients["adapter.u_real"], gradients["adapter.u_imag"])
    v = torch.complex(gradients["adapter.v_real"], gradients["adapter.v_imag"])
    result["U_complex"] = _norm(u, f"{prefix} U complex gradient")
    result["V_complex"] = _norm(v, f"{prefix} V complex gradient")
    return result


def _initial_gradient_balance(bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    batch_indices, batch_labels = sampler.next_batch()
    batch = images.index_select(0, batch_indices).to(device)
    batch_labels = batch_labels.to(device)
    adapter = LowRankAdapter(device=device)
    consumer = V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V8._load_runtime_state(consumer, bundle["initial_state"])
    _configure_adapter(consumer, adapter)
    components = _objective_components(consumer, adapter, batch, batch_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), float(bundle["s_tau"]), require_grad=True)
    balance: dict[str, Any] = {"first_batch_indices_sha256": V6.tensor_sha256(batch_indices), "first_batch_labels_sha256": V6.tensor_sha256(batch_labels), "batch_size": int(batch.shape[0]), "seed": SPLIT_SEED, "objective": "L_V10=L_anchor+lambda_transfer*L_transfer", "anchor": {}, "transfer": {}}
    for objective_name, loss in (("anchor", components["anchor_loss"]), ("transfer", components["transfer_loss"])):
        gradients = torch.autograd.grad(loss, tuple(adapter.parameters()), retain_graph=True, allow_unused=False)
        gradient_map = {f"adapter.{name}": value for name, value in zip(("u_real", "u_imag", "v_real", "v_imag"), gradients, strict=True)}
        norms = _factor_gradient_norms(adapter, gradient_map, f"V10 initial {objective_name}")
        _require(norms["U_complex"] > 0.0 and norms["V_complex"] > 0.0, f"V10 initial {objective_name} gradient is zero")
        balance[objective_name] = {"gradient_norms": norms}
    return balance


def _measure_train_batch(consumer: Any, adapter: LowRankAdapter, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, s_tau: float, lambda_transfer: float) -> dict[str, Any]:
    with torch.no_grad():
        components = _objective_components(consumer, adapter, batch, labels, prototypes, templates, s_tau, require_grad=False)
        image_loss = (components["raw_base"] - components["target"]).square().mean()
        shifted_image_loss = (components["raw_shift"] - components["target"]).square().mean()
        total = components["anchor_loss"] + lambda_transfer * components["transfer_loss"]
    return {"L_anchor": float(components["anchor_loss"]), "L_transfer": float(components["transfer_loss"]), "L_total": float(total), "L_image_evaluation_only": float(image_loss), "L_image_shift_evaluation_only": float(shifted_image_loss), "Delta_template_mean": float(components["delta"].to(torch.float64).mean())}


def _train_step(consumer: Any, adapter: LowRankAdapter, optimizer: torch.optim.Adam, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, s_tau: float, lambda_transfer: float, step: int) -> dict[str, Any]:
    runtime_before = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    before = {name: parameter.detach().cpu().clone() for name, parameter in _adapter_named(adapter).items()}
    optimizer.zero_grad(set_to_none=True)
    for parameter in adapter.parameters():
        parameter.grad = None
    components = _objective_components(consumer, adapter, batch, labels, prototypes, templates, s_tau, require_grad=True)
    with torch.no_grad():
        image_loss = (components["raw_base"].detach() - components["target"].detach()).square().mean()
        shifted_image_loss = (components["raw_shift"].detach() - components["target"].detach()).square().mean()
    total_loss = components["anchor_loss"] + lambda_transfer * components["transfer_loss"]
    _finite(image_loss, "V10 TRAIN image loss")
    _finite(shifted_image_loss, "V10 TRAIN shifted image loss")
    _finite(total_loss, "V10 total loss")
    _require(total_loss.requires_grad and components["anchor_loss"].requires_grad and components["transfer_loss"].requires_grad, "V10 objective gradient path is detached")
    _require(not image_loss.requires_grad and not shifted_image_loss.requires_grad, "V10 image loss entered differentiated objective")
    total_loss.backward()
    named = _adapter_named(adapter)
    gradients = {name: parameter.grad for name, parameter in named.items()}
    gradient_norms = _factor_gradient_norms(adapter, gradients, "V10 train")
    _require(gradient_norms["U_complex"] > 0.0 and gradient_norms["V_complex"] > 0.0, "V10 adapter gradient is zero")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "V10 frozen runtime received a gradient")
    optimizer.step()
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    runtime_after = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    _require(runtime_before == runtime_after, "V10 frozen runtime state changed")
    updates: dict[str, float] = {}
    for name, parameter in named.items():
        updates[name] = float((parameter.detach().cpu() - before[name]).to(torch.float64).square().sum().sqrt())
    _require(updates["adapter.u_real"] + updates["adapter.u_imag"] > 0.0 and updates["adapter.v_real"] + updates["adapter.v_imag"] > 0.0, "V10 adapter update is zero")
    return {"step": step, "objective": "L_V10=L_anchor+lambda_transfer*L_transfer", "L_anchor": float(components["anchor_loss"].detach().cpu()), "L_transfer": float(components["transfer_loss"].detach().cpu()), "L_total": float(total_loss.detach().cpu()), "L_image_evaluation_only": float(image_loss.cpu()), "L_image_shift_evaluation_only": float(shifted_image_loss.cpu()), "Delta_template_mean": float(components["delta"].detach().to(torch.float64).mean().cpu()), "gradient_norms": gradient_norms, "update_norms": updates, "image_loss_in_differentiated_objective": False, "prototype_loss_in_differentiated_objective": False, "smooth_in_differentiated_objective": False, "adapter_updated": True, "encoder_unchanged": True, "generator_unchanged": True, "ols_unchanged": True, "trainable_parameter_names": ADAPTER_PARAMETER_NAMES, "frozen_parameter_names": ["encoder.phase", "generator.phase", "OLS", "T=I_100", "W_OLS", "TRAIN prototypes", "TRAIN templates", "Smooth evaluator"], "runtime_state_sha256_after": runtime_after}


def _evaluate_v10(consumer: Any, adapter: LowRankAdapter, evaluator: Any, images: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, s_tau: float, *, step: int, train_trace: Mapping[str, Any] | None) -> dict[str, Any]:
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    raw_summary = V6._raw_accumulator()
    smooth_parts: list[torch.Tensor] = []
    margin_parts: list[torch.Tensor] = []
    distance_parts: list[torch.Tensor] = []
    prediction_parts: list[torch.Tensor] = []
    anchor_margin_parts: list[torch.Tensor] = []
    z_parts: list[torch.Tensor] = []
    adapted_z_parts: list[torch.Tensor] = []
    image_sum = 0.0
    flat_size = 1 * 200 * 200
    class_sum = [torch.zeros(flat_size, dtype=torch.float64) for _ in range(CLASS_COUNT)]
    class_sum_sq = [0.0 for _ in range(CLASS_COUNT)]
    class_count = [0 for _ in range(CLASS_COUNT)]
    overall_sum = torch.zeros(flat_size, dtype=torch.float64)
    overall_sum_sq = 0.0
    duplicate_counts: dict[str, int] = {}
    try:
        with torch.no_grad():
            for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
                batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
                batch_labels = labels[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
                components = _objective_components(consumer, adapter, batch, batch_labels, prototypes.to(consumer.device), templates.to(consumer.device), s_tau, require_grad=False)
                raw = components["raw_base"]
                V6._accumulate_raw(raw_summary, raw)
                image_loss = (raw - components["target"]).square().mean()
                _finite(image_loss, "V10 DEV image loss")
                smooth_parts.append(V6._smooth_logits(evaluator, raw))
                margin_parts.append(V6._ols_margin(components["z"], batch_labels, prototypes.to(consumer.device)))
                raw_view = F.adaptive_avg_pool2d(raw.clamp(0.0, 1.0), (28, 28)).to(torch.float64)
                support = F.adaptive_avg_pool2d(templates.clamp(0.0, 1.0), (28, 28)).to(device=consumer.device, dtype=torch.float64)
                distances = (raw_view[:, None] - support[None]).square().flatten(2).mean(dim=2)
                predictions = distances.argmin(dim=1)
                rows = torch.arange(batch.shape[0], device=consumer.device)
                correct = distances[rows, batch_labels]
                others = distances.clone()
                others[rows, batch_labels] = torch.inf
                anchor_margin = others.min(dim=1).values - correct
                distance_parts.append(distances.cpu())
                prediction_parts.append(predictions.cpu())
                anchor_margin_parts.append(anchor_margin.cpu().to(torch.float64))
                image_sum += float(image_loss.cpu()) * int(batch.shape[0])
                z_parts.append(components["z"].detach().cpu().contiguous())
                adapted_z_parts.append(components["z_adapted"].detach().cpu().contiguous())
                flat = raw.detach().cpu().to(torch.float64).flatten(1)
                overall_sum += flat.sum(dim=0)
                overall_sum_sq += float(flat.square().sum())
                cpu_labels = batch_labels.detach().cpu()
                for digit in torch.unique(cpu_labels).tolist():
                    mask = cpu_labels == digit
                    values = flat[mask]
                    class_sum[digit] += values.sum(dim=0)
                    class_sum_sq[digit] += float(values.square().sum())
                    class_count[digit] += int(values.shape[0])
                for sample in raw.detach().cpu().contiguous():
                    key = hashlib.sha256(sample.numpy().tobytes()).hexdigest()
                    duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
    finally:
        consumer.runtime.train(was_training)
    smooth_logits = torch.cat(smooth_parts, dim=0)
    margins = torch.cat(margin_parts, dim=0)
    distances = torch.cat(distance_parts, dim=0)
    predictions = torch.cat(prediction_parts, dim=0)
    anchor_margins = torch.cat(anchor_margin_parts, dim=0)
    z = torch.cat(z_parts, dim=0)
    z_adapted = torch.cat(adapted_z_parts, dim=0)
    per_class = []
    for digit in range(CLASS_COUNT):
        mask = labels == digit
        _require(bool(mask.any()), f"V10 DEV class {digit} absent")
        per_class.append(float((smooth_logits[mask].argmax(dim=1) == digit).to(torch.float64).mean()))

    def vector_stats(total: torch.Tensor, total_sq: float, count: int) -> dict[str, Any]:
        _require(count > 0, "V10 collapse class count is zero")
        mean = total / float(count)
        variance = (total_sq / float(count) - mean.square()).clamp_min(0.0)
        return {"count": count, "mean": float(mean.mean()), "std": float(variance.mean().sqrt()), "pixel_variance_mean": float(variance.mean()), "pixel_variance_max": float(variance.max())}

    class_stats = []
    class_means = []
    for digit in range(CLASS_COUNT):
        stats = vector_stats(class_sum[digit], class_sum_sq[digit], class_count[digit])
        stats["class"] = digit
        stats["distance_to_class_template"] = float(distances[labels == digit, digit].mean())
        class_stats.append(stats)
        class_means.append(class_sum[digit] / float(class_count[digit]))
    pair_distances = torch.stack([(class_means[left] - class_means[right]).square().mean() for left in range(CLASS_COUNT) for right in range(left + 1, CLASS_COUNT)])
    anchor_summary = V8._summary(anchor_margins)
    anchor_summary["positive_fraction"] = float((anchor_margins > 0.0).to(torch.float64).mean())
    raw_validity = V6._finish_raw_summary(raw_summary, int(labels.shape[0]))
    return {"step": step, "losses": {"L_image": image_sum / float(images.shape[0]), "role": "evaluation-only"}, "smooth": {"macro_recall": float(sum(per_class) / CLASS_COUNT), "per_class_recall": per_class}, "ols_margin": {"mean": float(margins.mean()), "median": float(margins.median()), "positive_fraction": float((margins > 0.0).to(torch.float64).mean())}, "template_anchor": {"distance": "mean squared pixel distance after clamp [0,1] and adaptive_avg_pool2d to 28x28", "scale": "s_tau frozen in contract; logits=-distance/(s_tau+1e-12)", "macro_recall": float(sum(float((predictions[labels == digit] == digit).to(torch.float64).mean()) for digit in range(CLASS_COUNT)) / CLASS_COUNT), "per_class_recall": [float((predictions[labels == digit] == digit).to(torch.float64).mean()) for digit in range(CLASS_COUNT)], "correct_template_distance": V8._summary(distances[torch.arange(labels.shape[0]), labels]), "correct_template_margin": anchor_summary}, "collapse": {"overall_raw": vector_stats(overall_sum, overall_sum_sq, int(labels.numel())), "within_class_raw": class_stats, "between_class_mean_output_separation": {"mean": float(pair_distances.mean()), "median": float(pair_distances.median()), "min": float(pair_distances.min()), "max": float(pair_distances.max()), "pair_count": int(pair_distances.numel())}, "duplicate_or_near_identical": {"exact_duplicate_count": int(sum(max(count - 1, 0) for count in duplicate_counts.values())), "unique_output_count": int(len(duplicate_counts)), "near_identical_supported": False, "near_identical_reason": "no pre-existing threshold primitive was used"}}, "raw_validity": raw_validity, "adapter": {**_adapter_observation(adapter), **_latent_adapter_metrics(z, z_adapted, labels)}, "train_trace": dict(train_trace) if train_trace is not None else None, "_smooth_logits": smooth_logits, "_margins": margins, "_anchor_distances": distances, "_anchor_predictions": predictions, "_anchor_correct_margin": anchor_margins}


def _cyclic_transfer_v10(consumer: Any, adapter: LowRankAdapter, evaluator: Any, images: torch.Tensor, labels: torch.Tensor, selection_indices: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor) -> dict[str, Any]:
    consumer.runtime.eval()
    rows = int(images.shape[0])
    _require(rows > 0 and rows % CLASS_COUNT == 0 and labels.shape == (rows,), "V10 cyclic selection shape is invalid")
    samples_per_class = rows // CLASS_COUNT
    _require(torch.equal(torch.bincount(labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), samples_per_class, dtype=torch.long)), "V10 cyclic selection is not balanced")
    targets = (labels + 1) % CLASS_COUNT
    z_parts: list[torch.Tensor] = []
    z_adapted_parts: list[torch.Tensor] = []
    shift_parts: list[torch.Tensor] = []
    shift_adapted_parts: list[torch.Tensor] = []
    raw_base_parts: list[torch.Tensor] = []
    raw_shift_parts: list[torch.Tensor] = []
    smooth_base_parts: list[torch.Tensor] = []
    smooth_shift_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, rows, LOGICAL_BATCH_SIZE):
            batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            batch_labels = labels[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            batch_targets = targets[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            forward = _forward_adapter(consumer, adapter, batch, require_grad=False)
            raw_base = forward["raw_base"]
            z = forward["z"].detach()
            z_adapted = forward["z_adapted"].detach()
            z_shift = z - prototypes.to(consumer.device).index_select(0, batch_labels) + prototypes.to(consumer.device).index_select(0, batch_targets)
            z_shift_adapted = adapter(z_shift)
            shifted_result = consumer.runtime.forward_from_ols(z_shift_adapted)
            raw_shift = _validate_adapted_result(shifted_result, int(batch.shape[0]), require_grad=False)
            z_parts.append(z.cpu().contiguous())
            z_adapted_parts.append(z_adapted.cpu().contiguous())
            shift_parts.append(z_shift.cpu().contiguous())
            shift_adapted_parts.append(z_shift_adapted.cpu().contiguous())
            raw_base_parts.append(raw_base.cpu().contiguous())
            raw_shift_parts.append(raw_shift.cpu().contiguous())
            smooth_base_parts.append(V6._smooth_logits(evaluator, raw_base))
            smooth_shift_parts.append(V6._smooth_logits(evaluator, raw_shift))
    z_base = torch.cat(z_parts, dim=0)
    z_base_adapted = torch.cat(z_adapted_parts, dim=0)
    z_shift = torch.cat(shift_parts, dim=0)
    z_shift_adapted = torch.cat(shift_adapted_parts, dim=0)
    raw_base = torch.cat(raw_base_parts, dim=0)
    raw_shift = torch.cat(raw_shift_parts, dim=0)
    smooth_base = torch.cat(smooth_base_parts, dim=0)
    smooth_shift = torch.cat(smooth_shift_parts, dim=0)
    template_base, template_shift = V6._template_margins(raw_base, raw_shift, templates, labels, targets)
    smooth_base_margin, smooth_shift_margin = V6._smooth_transfer_margins(smooth_base, smooth_shift, labels, targets)
    delta_template = template_shift - template_base
    delta_smooth = smooth_shift_margin - smooth_base_margin
    template_bootstrap = V6._paired_bootstrap(delta_template, labels)
    smooth_bootstrap = V6._paired_bootstrap(delta_smooth, labels)
    pair_summaries = []
    positive_template_pairs = 0
    positive_smooth_pairs = 0
    for source in range(CLASS_COUNT):
        mask = labels == source
        template_values = delta_template[mask]
        smooth_values = delta_smooth[mask]
        template_mean = float(template_values.mean())
        smooth_mean = float(smooth_values.mean())
        positive_template_pairs += int(template_mean > 0.0)
        positive_smooth_pairs += int(smooth_mean > 0.0)
        pair_summaries.append({"source": source, "target": (source + 1) % CLASS_COUNT, "count": int(mask.sum()), "template_delta": V6._summary(template_values), "smooth_delta": V6._summary(smooth_values)})
    broad = positive_template_pairs >= 2 and positive_smooth_pairs >= 2
    template_pass = template_bootstrap["observed"]["mean"] > 0.0 and template_bootstrap["ci_lower"] > 0.0
    smooth_pass = smooth_bootstrap["observed"]["mean"] > 0.0 and smooth_bootstrap["ci_lower"] > 0.0
    return {"selection": {"rows": rows, "per_class": samples_per_class, "seed": SELECTION_SEED, "indices_sha256": V6.tensor_sha256(selection_indices), "labels_sha256": V6.tensor_sha256(labels), "target_rule": "t=(y+1) mod 10", "target_labels_sha256": V6.tensor_sha256(targets)}, "intervention": "z_shift=z-mu_y+mu_t; z_shift_adapted=T_adapt z_shift; t=(y+1) mod 10", "template": {"definition": "M_template=d(raw,tau_y)-d(raw,tau_t); Delta=M_shift-M_base", "distance": "mean squared pixel distance after clamp [0,1] and adaptive_avg_pool2d to 28x28", "bootstrap": template_bootstrap, "pass": bool(template_pass)}, "smooth": {"definition": "M_smooth=score_t-score_y; Delta=M_shift-M_base using frozen SmoothMNISTEvaluator v2.1", "bootstrap": smooth_bootstrap, "pass": bool(smooth_pass)}, "pair_summaries": pair_summaries, "positive_pair_counts": {"template": positive_template_pairs, "smooth": positive_smooth_pairs}, "broad": bool(broad), "pair_dominated": bool(positive_template_pairs == 1 or positive_smooth_pairs == 1), "known_cyclic_transfer_retained": bool(template_pass and smooth_pass and broad), "raw_validity": {"base": V6._finish_raw_summary(V6._raw_summary_from_tensor(raw_base), rows), "shifted": V6._finish_raw_summary(V6._raw_summary_from_tensor(raw_shift), rows)}, "adapter": {**_adapter_observation(adapter), **_latent_adapter_metrics(z_base, z_base_adapted, labels)}, "_observations": {"selection_indices": selection_indices.cpu(), "labels": labels.cpu(), "targets": targets.cpu(), "z_base": z_base, "z_base_adapted": z_base_adapted, "z_shift": z_shift, "z_shift_adapted": z_shift_adapted, "raw_base": raw_base, "raw_shifted": raw_shift, "smooth_base_logits": smooth_base, "smooth_shifted_logits": smooth_shift, "template_base_margin": template_base, "template_shifted_margin": template_shift, "smooth_base_margin": smooth_base_margin, "smooth_shifted_margin": smooth_shift_margin}}


def _identity_probe(consumer: Any, adapter: LowRankAdapter, batch: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        forward = _forward_adapter(consumer, adapter, batch, require_grad=False)
        identity = _adapter_observation(adapter)
        _require(torch.equal(forward["z"], forward["z_adapted"]), "V10 step-0 adapter changed encoded coefficients")
        _require(torch.allclose(forward["identity_raw"], forward["raw_base"], atol=1.0e-6, rtol=1.0e-5), "V10 step-0 adapted forward is not matched identity")
    identity["forward_raw_allclose"] = True
    identity["z_exact_equal"] = True
    identity["batch_size"] = int(batch.shape[0])
    return identity


def _gradient_firewall_probe(bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    consumer = V8._new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    V8._load_runtime_state(consumer, bundle["initial_state"])
    adapter = LowRankAdapter(device=device)
    firewall = _configure_adapter(consumer, adapter)
    indices = train_indices[:LOGICAL_BATCH_SIZE]
    batch = images.index_select(0, indices).to(device)
    batch_labels = labels.index_select(0, indices).to(device)
    components = _objective_components(consumer, adapter, batch, batch_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], require_grad=True)
    image_loss = (components["raw_base"].detach() - components["target"].detach()).square().mean()
    _require(not image_loss.requires_grad, "V10 image loss has a gradient path")
    named = _adapter_named(adapter)
    wrong_anchor = F.cross_entropy(components["anchor_logits"], (batch_labels + 1) % CLASS_COUNT)
    wrong_transfer = F.softplus(components["delta"]).mean()
    wrong_anchor_grad = torch.autograd.grad(wrong_anchor, tuple(adapter.parameters()), retain_graph=True)
    wrong_transfer_grad = torch.autograd.grad(wrong_transfer, tuple(adapter.parameters()), retain_graph=True)
    total = components["anchor_loss"] + LAMBDA_TRANSFER * components["transfer_loss"]
    total.backward()
    combined = _factor_gradient_norms(adapter, {name: value.grad for name, value in named.items()}, "V10 combined probe")
    negative_anchor = _factor_gradient_norms(adapter, {f"adapter.{name}": value for name, value in zip(("u_real", "u_imag", "v_real", "v_imag"), wrong_anchor_grad, strict=True)}, "V10 wrong-label probe")
    negative_transfer = _factor_gradient_norms(adapter, {f"adapter.{name}": value for name, value in zip(("u_real", "u_imag", "v_real", "v_imag"), wrong_transfer_grad, strict=True)}, "V10 wrong-direction probe")
    _require(combined["U_complex"] > 0.0 and combined["V_complex"] > 0.0, "V10 combined adapter gradient is zero")
    _require(negative_anchor["U_complex"] > 0.0 and negative_anchor["V_complex"] > 0.0 and negative_transfer["U_complex"] > 0.0 and negative_transfer["V_complex"] > 0.0, "V10 negative-control gradient is zero")
    _require(all(parameter.grad is None for parameter in consumer.runtime.parameters()), "V10 runtime gradient firewall failed")
    _require(all(parameter.grad is None for parameter in bundle["smooth"].parameters()), "V10 Smooth evaluator firewall failed")
    _require(not bundle["prototypes"].requires_grad and not bundle["templates"].requires_grad, "V10 support tensor firewall failed")
    return {"firewall": firewall, "objective": "L_V10=L_anchor+lambda_transfer*L_transfer", "image_loss_requires_grad": False, "image_loss_gradient_contribution": 0.0, "prototype_loss_gradient_contribution": 0.0, "smooth_evaluator_gradient_contribution": 0.0, "combined_gradient_norms": combined, "negative_controls": {"wrong_label_anchor": negative_anchor, "wrong_direction_transfer": negative_transfer}, "runtime_gradient_none": True, "prototype_loss_in_objective": False, "smooth_in_objective": False}


def _checkpoint_payload(consumer: Any, adapter: LowRankAdapter, optimizer: torch.optim.Adam, sampler: Any, *, step: int, initial_runtime_digest: str, initial_adapter_digest: str, prototypes: torch.Tensor, templates: torch.Tensor, rng_state: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": CHECKPOINT_SCHEMA, "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "checkpoint_step": step, "runtime_state": _cpu_tree(consumer.runtime.state_dict()), "adapter_state": _cpu_tree(adapter.state_dict()), "optimizer_state": _cpu_tree(optimizer.state_dict()), "sampler_state": _cpu_tree(sampler.state_dict()), "rng_state": _cpu_tree(rng_state), "initial_runtime_state_sha256": initial_runtime_digest, "initial_adapter_state_sha256": initial_adapter_digest, "prototypes_sha256": V6.tensor_sha256(prototypes), "templates_sha256": V6.tensor_sha256(templates), "s_tau": S_TAU, "lambda_transfer": LAMBDA_TRANSFER, "objective": "L_V10=L_anchor+lambda_transfer*L_transfer", "trainable_parameter_names": ADAPTER_PARAMETER_NAMES, "frozen_parameter_names": ["encoder.phase", "generator.phase", "OLS", "T=I_100", "W_OLS", "TRAIN prototypes", "TRAIN templates", "Smooth evaluator"], "image_loss_gradient_contribution": 0.0, "prototype_loss_gradient_contribution": 0.0, "smooth_evaluator_gradient_contribution": 0.0, "data_roles": {"TRAIN": True, "DEV": True, "calibration": False, "protected": False, "official_test": False, "sealed": False}, "stage_b_executed": False, "scale_extraction_executed": False, "prior_shaping_executed": False, "generation_executed": False, "formal_promotion": False}


def _save_and_verify_checkpoint(root: Path, target: Path, payload: Mapping[str, Any], name: str, consumer: Any, v2_contract: Any) -> dict[str, Any]:
    path = target / name
    atomic_torch_save_new(path, payload, artifact_kind="V10 checkpoint")
    digest = V7.file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V10 checkpoint reload")
    _require(loaded.get("schema") == CHECKPOINT_SCHEMA and loaded.get("checkpoint_step") == payload.get("checkpoint_step"), "V10 checkpoint schema/step reload changed")
    _require(loaded.get("prototypes_sha256") == PROTOTYPE_SHA256 and loaded.get("templates_sha256") == V6.tensor_sha256(payload.get("templates", torch.empty(0))), "V10 checkpoint support identity changed")
    clone = V8._new_consumer(v2_contract, consumer.device, consumer.runtime.config.propagation_grid_size)
    V8._load_runtime_state(clone, _mapping(loaded.get("runtime_state"), "V10 checkpoint runtime state"))
    clone_adapter = LowRankAdapter(device=consumer.device)
    clone_adapter.load_state_dict(dict(_mapping(loaded.get("adapter_state"), "V10 checkpoint adapter state")), strict=True)
    _configure_adapter(clone, clone_adapter)
    _require(_digest_tree(_cpu_tree(clone.runtime.state_dict())) == _digest_tree(_cpu_tree(loaded["runtime_state"])), "V10 checkpoint runtime reload changed")
    _require(_digest_tree(_cpu_tree(clone_adapter.state_dict())) == _digest_tree(_cpu_tree(loaded["adapter_state"])), "V10 checkpoint adapter reload changed")
    clone_optimizer = _optimizer(clone_adapter)
    clone_optimizer.load_state_dict(dict(_mapping(loaded.get("optimizer_state"), "V10 checkpoint optimizer state")))
    _require(_tree_equal(_cpu_tree(clone_optimizer.state_dict()), loaded["optimizer_state"]), "V10 checkpoint optimizer reload changed")
    current_rng = V6._capture_rng()
    V6._restore_rng(_mapping(loaded.get("rng_state"), "V10 checkpoint RNG state"))
    reloaded_rng = V6._capture_rng()
    V6._restore_rng(current_rng)
    _require(_tree_equal(reloaded_rng, loaded["rng_state"]), "V10 checkpoint RNG reload changed")
    _require(loaded.get("trainable_parameter_names") == ADAPTER_PARAMETER_NAMES and float(loaded["s_tau"]) == S_TAU and float(loaded["lambda_transfer"]) == LAMBDA_TRANSFER, "V10 checkpoint objective/trainable identity changed")
    return {"path": _relative(root, path), "sha256": digest, "step": int(payload["checkpoint_step"]), "reloaded": True, "resume_identity": True}


def _three_step_smoke(root: Path, bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device, target: Path) -> dict[str, Any]:
    consumer = V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V8._load_runtime_state(consumer, bundle["initial_state"])
    adapter = LowRankAdapter(device=device)
    firewall = _configure_adapter(consumer, adapter)
    optimizer = _optimizer(adapter)
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    traces = []
    for step in range(1, PRECHECK_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        batch = images.index_select(0, indices).to(device)
        traces.append(_train_step(consumer, adapter, optimizer, batch, batch_labels.to(device), bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], LAMBDA_TRANSFER, step))
    payload = _checkpoint_payload(consumer, adapter, optimizer, sampler, step=PRECHECK_STEPS, initial_runtime_digest=_digest_tree(bundle["initial_state"]), initial_adapter_digest=_digest_tree(_cpu_tree(LowRankAdapter(device=device).state_dict())), prototypes=bundle["prototypes"], templates=bundle["templates"], rng_state=V6._capture_rng())
    payload["templates"] = bundle["templates"]
    checkpoint = _save_and_verify_checkpoint(root, target, payload, PREFLIGHT_CHECKPOINT_NAME, consumer, bundle["v2_contract"])
    loaded = _mapping(safe_torch_load(target / PREFLIGHT_CHECKPOINT_NAME, expected_sha256=checkpoint["sha256"], map_location="cpu"), "V10 smoke checkpoint")
    resumed_sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    resumed_sampler.load_state_dict(dict(_mapping(loaded["sampler_state"], "V10 smoke sampler state")))
    next_original, next_original_labels = sampler.next_batch()
    next_resumed, next_resumed_labels = resumed_sampler.next_batch()
    _require(torch.equal(next_original, next_resumed) and torch.equal(next_original_labels, next_resumed_labels), "V10 sampler resume identity changed")
    return {"firewall": firewall, "traces": traces, "checkpoint": checkpoint, "resume_identity": {"runtime": True, "adapter": True, "optimizer": True, "rng": True, "sampler_next_batch": True}, "scientific_updates": 0, "scientific_meaning": "none"}


def _save_sufficient(root: Path, target: Path, observations: Mapping[str, Any], *, name: str, schema: str) -> dict[str, Any]:
    path = target / name
    payload = {"schema": schema, "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, **{key: value.cpu() if torch.is_tensor(value) else value for key, value in observations.items()}}
    atomic_torch_save_new(path, payload, artifact_kind="V10 sufficient observations")
    digest = V7.file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V10 sufficient observations reload")
    _require(loaded.get("schema") == schema, "V10 sufficient observation schema reload changed")
    for key, value in observations.items():
        if torch.is_tensor(value):
            _require(torch.equal(loaded[key], value.cpu()), f"V10 sufficient observation reload changed: {key}")
    return {"path": _relative(root, path), "sha256": digest, "schema": schema, "reloaded": True}


def _save_cyclic_raw(root: Path, target: Path, cyclic: Mapping[str, Any]) -> dict[str, Any]:
    observations = cyclic["_observations"]
    payload = {"schema": "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_cyclic_raw_outputs.v1", "version": 1, "task_id": TASK_ID, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, **{key: value.cpu() if torch.is_tensor(value) else value for key, value in observations.items()}}
    path = target / SCIENTIFIC_RAW_NAME
    atomic_torch_save_new(path, payload, artifact_kind="V10 cyclic raw outputs")
    digest = V7.file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), "V10 cyclic raw reload")
    _require(loaded.get("schema") == payload["schema"] and torch.equal(loaded["selection_indices"], observations["selection_indices"].cpu()), "V10 cyclic raw reload changed")
    return {"path": _relative(root, path), "sha256": digest, "reloaded": True}


def _synthetic_acceptance(root: Path, bundle: Mapping[str, Any], device: torch.device, target: Path) -> dict[str, Any]:
    ramp = torch.linspace(0.0, 1.0, 28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)
    train_labels = torch.arange(CLASS_COUNT, dtype=torch.long)
    train_images = (ramp + train_labels.view(-1, 1, 1, 1).to(torch.float32) / 20.0).clamp(0.0, 1.0)
    eval_labels = train_labels.repeat_interleave(2)
    eval_images = train_images.index_select(0, eval_labels)
    consumer = V8._new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    V8._load_runtime_state(consumer, bundle["initial_state"])
    adapter = LowRankAdapter(device=device)
    firewall = _configure_adapter(consumer, adapter)
    optimizer = _optimizer(adapter)
    traces = []
    for step in range(1, PRECHECK_STEPS + 1):
        traces.append(_train_step(consumer, adapter, optimizer, train_images.to(device), train_labels.to(device), bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], LAMBDA_TRANSFER, step))
    evaluation = _evaluate_v10(consumer, adapter, bundle["smooth"], eval_images, eval_labels, bundle["prototypes"], bundle["templates"], bundle["s_tau"], step=PRECHECK_STEPS, train_trace=traces[-1])
    cyclic = _cyclic_transfer_v10(consumer, adapter, bundle["smooth"], eval_images, eval_labels, torch.arange(eval_images.shape[0], dtype=torch.long), bundle["prototypes"], bundle["templates"])
    observations = {"labels": eval_labels, "synthetic_anchor_distances": evaluation["_anchor_distances"], "synthetic_anchor_predictions": evaluation["_anchor_predictions"], "synthetic_anchor_margin": evaluation["_anchor_correct_margin"], "synthetic_ols_margin": evaluation["_margins"], "synthetic_smooth_logits": evaluation["_smooth_logits"], **cyclic["_observations"]}
    sufficient = _save_sufficient(root, target, observations, name=PREFLIGHT_SYNTHETIC_OBS_NAME, schema="lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_synthetic_sufficient_observations.v1")
    zero_delta = V6._paired_bootstrap(torch.zeros_like(eval_labels, dtype=torch.float64), eval_labels, replicates=SMOKE_BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    report_payload = {"schema": "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_synthetic_report.v1", "scientific_meaning": "none", "training_updates": PRECHECK_STEPS, "objective": "L_V10=L_anchor+lambda_transfer*L_transfer", "firewall": firewall, "identity": _identity_probe(consumer, LowRankAdapter(device=device), eval_images[:LOGICAL_BATCH_SIZE].to(device)), "traces": traces, "evaluation": {key: value for key, value in evaluation.items() if not key.startswith("_")}, "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")}, "zero_delta_bootstrap": zero_delta, "sufficient_observations": sufficient}
    report_path = target / PREFLIGHT_SYNTHETIC_REPORT_NAME
    atomic_write_new_json(report_path, report_payload, artifact_kind="V10 synthetic report")
    report_ref = {"path": _relative(root, report_path), "sha256": V7.file_sha256(report_path)}
    lineage_path = target / PREFLIGHT_SYNTHETIC_LINEAGE_NAME
    atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_synthetic_lineage.v1", "result": "VALID_COMPLETED", "report": report_ref, "sufficient_observations": sufficient, "claim_boundary": {"scientific_meaning": "none", "protected": False, "official_test": False, "sealed": False}}, artifact_kind="V10 synthetic lineage")
    _require(_read_json(report_path, "V10 synthetic report reload")["schema"] == report_payload["schema"], "V10 synthetic report reload changed")
    _require(_read_json(lineage_path, "V10 synthetic lineage reload")["report"]["sha256"] == report_ref["sha256"], "V10 synthetic lineage reload changed")
    return {"scientific_meaning": "none", "firewall": firewall, "traces": traces, "evaluation": {key: value for key, value in evaluation.items() if not key.startswith("_")}, "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")}, "identity": report_payload["identity"], "zero_delta_bootstrap": zero_delta, "sufficient_observations": sufficient, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": V7.file_sha256(lineage_path)}, "report_lineage_reloaded": True}


def _production_no_data_smoke(bundle: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    consumer = V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V8._load_runtime_state(consumer, bundle["initial_state"])
    adapter = LowRankAdapter(device=device)
    _configure_adapter(consumer, adapter, trainable=False)
    before_runtime = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    before_adapter = _digest_tree(_cpu_tree(adapter.state_dict()))
    synthetic = torch.linspace(0.0, 1.0, LOGICAL_BATCH_SIZE * 28 * 28, device=device, dtype=torch.float32).reshape(LOGICAL_BATCH_SIZE, 1, 28, 28)
    with torch.no_grad():
        forward = _forward_adapter(consumer, adapter, synthetic, require_grad=False)
    after_runtime = _digest_tree(_cpu_tree(consumer.runtime.state_dict()))
    after_adapter = _digest_tree(_cpu_tree(adapter.state_dict()))
    _require(before_runtime == after_runtime and before_adapter == after_adapter, "V10 production no-data smoke changed frozen state")
    return {"executed": True, "grid": PRODUCTION_GRID, "raw_validity": V6._finish_raw_summary(V6._raw_summary_from_tensor(forward["raw_base"]), LOGICAL_BATCH_SIZE), "runtime_unchanged": True, "adapter_unchanged": True}


def _classify(r0: float, r300: float, smooth_bootstrap: Mapping[str, Any], cyclic: Mapping[str, Any], raw_valid: bool) -> dict[str, Any]:
    natural_pass = r300 - r0 >= SMOOTH_REQUIRED_DELTA and float(smooth_bootstrap["ci_lower"]) > 0.0
    cyclic_pass = bool(cyclic["known_cyclic_transfer_retained"])
    supported = bool(natural_pass and cyclic_pass and raw_valid)
    return {"classification": SUPPORTED_CLASSIFICATION if supported else NOT_SUPPORTED_CLASSIFICATION, "outer_gate": OUTER_GATE, "supported": supported, "criteria": {"natural_smooth_improvement": natural_pass, "known_cyclic_transfer_retained": cyclic_pass, "raw_validity": raw_valid}, "thresholds": {"smooth_delta": SMOOTH_REQUIRED_DELTA, "smooth_ci_lower": 0.0}}


def _preflight_input_payload(contract_ref: Mapping[str, Any], device_facts: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_input.v1", "version": 1, "task_id": TASK_ID, "contract": dict(contract_ref), "objective": "L_V10=L_anchor+lambda_transfer*L_transfer", "trainable_parameter_names": ADAPTER_PARAMETER_NAMES, "frozen_gradient_paths": ["L_image", "L_proto", "Smooth/frozen runtime"], "adapter": {"rank": ADAPTER_RANK, "init_scale": INIT_SCALE, "application": "z_adapted=z @ T_adapt.T"}, "runtime": {"device": dict(device_facts), "grid": PRODUCTION_GRID}, "one_shot": True, "scientific_300_step_executed": False, "protected": False, "official_test": False, "sealed": False}


def run_preflight(root: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to overwrite V10 preflight output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V6.git_facts(root)
    status_before = _status_lines(root)
    payload: dict[str, Any] = {"schema": "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_report.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V10_PREFLIGHT", "outer_gate": OUTER_GATE, "provenance": {"git_start": started}, "claim_boundary": {"engineering_preflight_executed": False, "scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}}
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from V10 authorization")
        contract, v2_contract = _load_contract(root)
        bundle = _resolve_authorities(root, contract)
        bundle["v2_contract"] = v2_contract
        dll = V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V6._cuda_facts()
        bundle["smooth"].to(device)
        images, labels, train_indices, _development_indices, dataset = V6._load_roles(root)
        V6._initial_rng()
        gradient_balance = _initial_gradient_balance(bundle, images, labels, train_indices, device)
        no_data = _production_no_data_smoke(bundle, device)
        firewall_probe = _gradient_firewall_probe(bundle, images, labels, train_indices, device)
        identity_consumer = V8._new_consumer(v2_contract, device, SMOKE_GRID)
        V8._load_runtime_state(identity_consumer, bundle["initial_state"])
        identity_adapter = LowRankAdapter(device=device)
        _configure_adapter(identity_consumer, identity_adapter, trainable=False)
        identity = _identity_probe(identity_consumer, identity_adapter, images.index_select(0, train_indices[:LOGICAL_BATCH_SIZE]).to(device))
        smoke = _three_step_smoke(root, bundle, images, labels, train_indices, device, output_dir)
        synthetic = _synthetic_acceptance(root, bundle, device, output_dir)
        contract_ref = {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}
        input_path = output_dir / PREFLIGHT_INPUT_NAME
        atomic_write_new_json(input_path, _preflight_input_payload(contract_ref, device_facts), artifact_kind="V10 preflight input")
        payload.update({"contract": contract_ref, "parent_contract": {"path": V9_CONTRACT_PATH.as_posix(), "sha256": V9_CONTRACT_SHA256}, "authorities": bundle["authorities"], "runtime": {"cuda_dll_search_path": dll, "device": device_facts, "propagation_grid": PRODUCTION_GRID}, "data_roles": dataset, "adapter": {"rank": ADAPTER_RANK, "initial": _adapter_observation(identity_adapter), "identity_probe": identity}, "objective": {"s_tau": S_TAU, "lambda_transfer": LAMBDA_TRANSFER, "formula": "L_V10=L_anchor+lambda_transfer*L_transfer"}, "gradient_balance": gradient_balance, "production_no_data_smoke": no_data, "gradient_firewall_probe": firewall_probe, "three_step_smoke": smoke, "synthetic_acceptance": synthetic, "input": {"path": _relative(root, input_path), "sha256": V7.file_sha256(input_path)}, "result": "AUTHORIZABLE", "status": "valid_completed", "classification": "LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_V10_AUTHORIZABLE", "claim_boundary": {"engineering_preflight_executed": True, "scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}, "next": "proceed directly to the one-shot 300-step V10 scientific execution"})
    except HardBlocked as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": error.classification, "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific execution"})
    except (InvalidV10, V9.InvalidV9, V8.InvalidV8, V7.InvalidV7, V6.InvalidV6) as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V10_PREFLIGHT_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no scientific execution"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V10_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no scientific execution"})
    finally:
        ending = V6.git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == _status_lines(root)})
    report_path = output_dir / PREFLIGHT_REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V10 preflight report")
    report_ref = {"path": _relative(root, report_path), "sha256": V7.file_sha256(report_path)}
    lineage_path = output_dir / PREFLIGHT_LINEAGE_NAME
    atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "provenance": payload["provenance"], "report": report_ref, "claim_boundary": payload["claim_boundary"]}, artifact_kind="V10 preflight lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": V7.file_sha256(lineage_path)}, "scientific_300_step_executed": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _validate_preflight(root: Path) -> dict[str, Any]:
    report_path = _path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = _path(root, DEFAULT_PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    _require(report_path.is_file() and lineage_path.is_file(), "V10 preflight authority is unavailable")
    report = _read_json(report_path, "V10 preflight report")
    lineage = _read_json(lineage_path, "V10 preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed" and report.get("classification") == "LIGHTGEN_2D_LOW_RANK_OLS_TO_GENERATOR_REMAP_CANARY_V10_AUTHORIZABLE", "V10 preflight is not AUTHORIZABLE")
    _require(lineage.get("result") == "AUTHORIZABLE" and lineage.get("classification") == report.get("classification"), "V10 preflight lineage changed")
    _require(report.get("claim_boundary", {}).get("scientific_300_step_executed") is False, "V10 preflight opened scientific execution")
    _require(float(report.get("objective", {}).get("s_tau", 0.0)) == S_TAU and float(report.get("objective", {}).get("lambda_transfer", 0.0)) == LAMBDA_TRANSFER, "V10 preflight objective constants changed")
    _require(report.get("adapter", {}).get("identity_probe", {}).get("forward_raw_allclose") is True, "V10 preflight adapter identity missing")
    _require(report.get("three_step_smoke", {}).get("resume_identity", {}).get("sampler_next_batch") is True, "V10 preflight sampler resume identity missing")
    return {"report": report, "report_ref": {"path": _relative(root, report_path), "sha256": V7.file_sha256(report_path)}, "lineage_ref": {"path": _relative(root, lineage_path), "sha256": V7.file_sha256(lineage_path)}}


def run_scientific(root: Path, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to overwrite V10 scientific output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V6.git_facts(root)
    status_before = _status_lines(root)
    payload: dict[str, Any] = {"schema": REPORT_SCHEMA, "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V10_SCIENTIFIC_HARNESS", "outer_gate": OUTER_GATE, "provenance": {"git_start": started}, "claim_boundary": {"scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_shaping_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}, "governance": {"retry": False, "second_arm": False, "rank_sweep": False, "hyperparameter_sweep": False, "stage_b": False, "generation": False, "formal_promotion": False}}
    scientific_started = False
    updates_completed = 0
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from V10 authorization")
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
        for objective_name in ("anchor", "transfer"):
            for key in ("U_complex", "V_complex"):
                _require(math.isclose(float(gradient_balance[objective_name]["gradient_norms"][key]), float(preflight_balance[objective_name]["gradient_norms"][key]), rel_tol=1.0e-12, abs_tol=1.0e-12), f"V10 initial {objective_name} gradient changed between preflight and scientific entry")
        consumer = V8._new_consumer(v2_contract, device, PRODUCTION_GRID)
        V8._load_runtime_state(consumer, bundle["initial_state"])
        adapter = LowRankAdapter(device=device)
        firewall = _configure_adapter(consumer, adapter)
        optimizer = _optimizer(adapter)
        sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
        initial_runtime_digest = _digest_tree(bundle["initial_state"])
        initial_adapter_digest = _digest_tree(_cpu_tree(adapter.state_dict()))
        metric_sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
        metric_indices, metric_labels = metric_sampler.next_batch()
        initial_train_metrics = _measure_train_batch(consumer, adapter, images.index_select(0, metric_indices).to(device), metric_labels.to(device), bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], LAMBDA_TRANSFER)
        contract_ref = {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}
        preflight_ref = {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"], "result": "AUTHORIZABLE"}
        payload.update({"contract": contract_ref, "parent_contract": {"path": V9_CONTRACT_PATH.as_posix(), "sha256": V9_CONTRACT_SHA256}, "preflight": preflight_ref, "authorities": bundle["authorities"], "runtime": {"cuda_dll_search_path": dll, "device": device_facts, "propagation_grid": PRODUCTION_GRID, "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "trainable_parameter_names": ADAPTER_PARAMETER_NAMES}}, "data_roles": dataset, "initial_state": {"combined_runtime_state_sha256": initial_runtime_digest, "adapter_state_sha256": initial_adapter_digest, "encoder_phase_sha256": V3_ENCODER_SHA256, "generator_phase_sha256": V4_GENERATOR_TENSOR_SHA256, "rng_sha256": _digest_tree(V6._capture_rng())}, "firewall": firewall, "adapter": {"rank": ADAPTER_RANK, "step0": _adapter_observation(adapter)}, "objective": {"formula": "L_V10=L_anchor+lambda_transfer*L_transfer", "s_tau": S_TAU, "lambda_transfer": LAMBDA_TRANSFER, "lambda_rule": "reuse exact frozen V9/V8 lambda_transfer", "image_loss_gradient_contribution": 0.0, "prototype_loss_gradient_contribution": 0.0, "smooth_evaluator_gradient_contribution": 0.0, "image_loss_role": "evaluation-only", "template_anchor_role": "differentiated natural adapted output objective", "transfer_role": "differentiated cyclic latent-shift adapted output objective"}, "selection": {"rows": SELECTION_ROWS, "per_class": SELECTION_PER_CLASS, "seed": SELECTION_SEED, "indices_sha256": SELECTION_INDICES_SHA256, "labels_sha256": SELECTION_LABELS_SHA256}})
        input_payload = {"schema": INPUT_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "contract": contract_ref, "preflight": preflight_ref, "protocol": {"objective": "L_V10=L_anchor+lambda_transfer*L_transfer", "s_tau": S_TAU, "lambda_transfer": LAMBDA_TRANSFER, "steps": SCIENTIFIC_STEPS, "checkpoints": list(CHECKPOINT_STEPS), "batch": LOGICAL_BATCH_SIZE, "grid": PRODUCTION_GRID, "optimizer": payload["runtime"]["optimizer"], "trainable": ADAPTER_PARAMETER_NAMES, "adapter": {"rank": ADAPTER_RANK, "application": "z_adapted=z @ T_adapt.T"}, "latent_shift": "z_shift=z-mu_y+mu_t", "cyclic_target": "t=(y+1) mod 10", "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED}}, "data_roles": {"TRAIN": 55000, "DEV": 5000, "calibration": 0, "protected": 0, "official_test": 0, "sealed": 0}, "claim_boundary": payload["claim_boundary"]}
        input_path = output_dir / SCIENTIFIC_INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V10 scientific input")
        payload["input"] = {"path": _relative(root, input_path), "sha256": V7.file_sha256(input_path)}
        scientific_started = True
        identity = _identity_probe(consumer, adapter, development_images[:LOGICAL_BATCH_SIZE].to(device))
        evaluations: dict[str, dict[str, Any]] = {}
        checkpoints: dict[str, dict[str, Any]] = {}
        step0 = _evaluate_v10(consumer, adapter, bundle["smooth"], development_images, development_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], step=0, train_trace=None)
        step0["identity_probe"] = identity
        checkpoint_payload = _checkpoint_payload(consumer, adapter, optimizer, sampler, step=0, initial_runtime_digest=initial_runtime_digest, initial_adapter_digest=initial_adapter_digest, prototypes=bundle["prototypes"], templates=bundle["templates"], rng_state=V6._capture_rng())
        checkpoint_payload["templates"] = bundle["templates"]
        checkpoints["0"] = _save_and_verify_checkpoint(root, output_dir, checkpoint_payload, "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_step_0_checkpoint.v1.pt", consumer, v2_contract)
        evaluations["0"] = step0
        traces: list[dict[str, Any]] = []
        train_metrics = {"0": initial_train_metrics}
        started_training = time.perf_counter()
        for step in range(1, SCIENTIFIC_STEPS + 1):
            indices, batch_labels = sampler.next_batch()
            batch = images.index_select(0, indices).to(device)
            trace = _train_step(consumer, adapter, optimizer, batch, batch_labels.to(device), bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], LAMBDA_TRANSFER, step)
            traces.append(trace)
            updates_completed = step
            del batch
            if step in CHECKPOINT_STEPS[1:]:
                checkpoint_payload = _checkpoint_payload(consumer, adapter, optimizer, sampler, step=step, initial_runtime_digest=initial_runtime_digest, initial_adapter_digest=initial_adapter_digest, prototypes=bundle["prototypes"], templates=bundle["templates"], rng_state=V6._capture_rng())
                checkpoint_payload["templates"] = bundle["templates"]
                checkpoints[str(step)] = _save_and_verify_checkpoint(root, output_dir, checkpoint_payload, f"lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_step_{step}_checkpoint.v1.pt", consumer, v2_contract)
                evaluations[str(step)] = _evaluate_v10(consumer, adapter, bundle["smooth"], development_images, development_labels, bundle["prototypes"].to(device), bundle["templates"].to(device), bundle["s_tau"], step=step, train_trace=trace)
                train_metrics[str(step)] = {key: trace[key] for key in ("L_anchor", "L_transfer", "L_total", "L_image_evaluation_only", "L_image_shift_evaluation_only", "Delta_template_mean")}
            if step % 25 == 0:
                print(json.dumps({"event": "scientific_progress", "step": step, "elapsed_seconds": time.perf_counter() - started_training}, ensure_ascii=False), flush=True)
        _require(set(evaluations) == {"0", "100", "200", "300"} and set(checkpoints) == {"0", "100", "200", "300"}, "V10 scientific checkpoint/evaluation set is incomplete")
        step0_record = evaluations["0"]
        step300_record = evaluations["300"]
        smooth_correct0 = step0_record["_smooth_logits"].argmax(dim=1).eq(development_labels)
        smooth_correct300 = step300_record["_smooth_logits"].argmax(dim=1).eq(development_labels)
        smooth_bootstrap = V6._paired_bootstrap(smooth_correct300.to(torch.float64) - smooth_correct0.to(torch.float64), development_labels)
        ols_bootstrap = V6._paired_bootstrap(step300_record["_margins"] - step0_record["_margins"], development_labels)
        cyclic = _cyclic_transfer_v10(consumer, adapter, bundle["smooth"], selection_images, selection_labels, selection_indices, bundle["prototypes"], bundle["templates"])
        sufficient_obs = {"step0_smooth_logits": step0_record["_smooth_logits"], "step300_smooth_logits": step300_record["_smooth_logits"], "step0_ols_margin": step0_record["_margins"], "step300_ols_margin": step300_record["_margins"], "step0_anchor_distances": step0_record["_anchor_distances"], "step300_anchor_distances": step300_record["_anchor_distances"], "step0_anchor_predictions": step0_record["_anchor_predictions"], "step300_anchor_predictions": step300_record["_anchor_predictions"], "step0_anchor_correct_margin": step0_record["_anchor_correct_margin"], "step300_anchor_correct_margin": step300_record["_anchor_correct_margin"], **cyclic["_observations"]}
        sufficient_ref = _save_sufficient(root, output_dir, sufficient_obs, name=SCIENTIFIC_SUFFICIENT_NAME, schema=SUFFICIENT_SCHEMA)
        cyclic_raw_ref = _save_cyclic_raw(root, output_dir, cyclic)
        raw_valid = all(bool(evaluations[str(step)]["raw_validity"]["finite"] and evaluations[str(step)]["raw_validity"]["structured"] and evaluations[str(step)]["raw_validity"]["nonconstant"]) for step in CHECKPOINT_STEPS) and bool(cyclic["raw_validity"]["base"]["finite"] and cyclic["raw_validity"]["shifted"]["finite"])
        r0 = float(step0_record["smooth"]["macro_recall"])
        r300 = float(step300_record["smooth"]["macro_recall"])
        decision = _classify(r0, r300, smooth_bootstrap, cyclic, raw_valid)
        stage_firewall = {"metric": "step-300 DEV Smooth macro recall", "threshold": STAGE_A_FIREWALL, "R300": r300, "status": "STAGE_A_FIREWALL_PASSED" if r300 >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False}
        public_evaluations = {step: {key: value for key, value in record.items() if not key.startswith("_")} for step, record in evaluations.items()}
        payload.update({"result": "VALID_COMPLETED", "status": "valid_completed", "classification": decision["classification"], "evaluations": public_evaluations, "train_metrics": train_metrics, "traces": traces, "checkpoints": checkpoints, "scientific_updates": SCIENTIFIC_STEPS, "runtime_device": device_facts, "step300_minus_step0": {"Delta_R_Smooth": smooth_bootstrap, "Delta_R_observed": r300 - r0, "Delta_M_OLS": ols_bootstrap, "Delta_L_image_DEV": float(step300_record["losses"]["L_image"] - step0_record["losses"]["L_image"]), "Delta_L_anchor_TRAIN": train_metrics["300"]["L_anchor"] - train_metrics["0"]["L_anchor"], "Delta_L_transfer_TRAIN": train_metrics["300"]["L_transfer"] - train_metrics["0"]["L_transfer"], "Delta_L_total_TRAIN": train_metrics["300"]["L_total"] - train_metrics["0"]["L_total"]}, "natural_smooth_semantics": {"R0": r0, "R300": r300, "Delta_R": r300 - r0, "bootstrap": smooth_bootstrap, "step0_per_class": step0_record["smooth"]["per_class_recall"], "step300_per_class": step300_record["smooth"]["per_class_recall"]}, "ols_prototype_margin": {"step0": step0_record["ols_margin"], "step300": step300_record["ols_margin"], "bootstrap": ols_bootstrap}, "template_anchor": {"step0": step0_record["template_anchor"], "step300": step300_record["template_anchor"], "role": "descriptive mechanism evidence; not a substitute for Smooth Gate"}, "collapse_diagnostics": {"step0": step0_record["collapse"], "step300": step300_record["collapse"]}, "adapter_diagnostics": {"step0": step0_record["adapter"], "step300": step300_record["adapter"], "role": "descriptive canary diagnostics; not an additional gate"}, "initial_gradient_balance": gradient_balance, "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")}, "artifacts": {"sufficient_observations": sufficient_ref, "cyclic_raw_outputs": cyclic_raw_ref}, "decision": decision, "stage_a_firewall": stage_firewall, "claim_boundary": {**payload["claim_boundary"], "scientific_training_executed": True, "scientific_300_step_executed": True}, "governance": {"optimizer": "Adam on low-rank adapter U/V real-imag parameters", "backward": True, "adapter_update": True, "encoder_update": False, "generator_update": False, "optical_update": False, "image_loss_gradient": False, "prototype_loss_gradient": False, "smooth_evaluator_gradient": False, "retry": False, "second_arm": False, "rank_sweep": False, "hyperparameter_sweep": False, "stage_b": False, "scale_extraction": False, "prior_shaping": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False, "training_authorized": False, "generation_readiness": "not_ready"}, "next": "no_retry; V10 bounded canary does not authorize Stage B or generation"})
    except HardBlocked as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": error.classification, "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific retry"})
    except (InvalidV10, V9.InvalidV9, V8.InvalidV8, V7.InvalidV7, V6.InvalidV6) as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V10_SCIENTIFIC_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no scientific retry"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V10_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no scientific retry"})
    finally:
        ending = V6.git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == _status_lines(root), "unexpected_modified_paths": sorted(set(_status_lines(root)).symmetric_difference(status_before)), "scientific_started": scientific_started})
        payload["claim_boundary"]["scientific_training_executed"] = bool(scientific_started)
        payload["claim_boundary"]["scientific_300_step_executed"] = bool(updates_completed >= SCIENTIFIC_STEPS)
        payload["scientific_updates_completed"] = updates_completed
    report_path = output_dir / SCIENTIFIC_REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V10 scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": V7.file_sha256(report_path)}
    lineage_path = output_dir / SCIENTIFIC_LINEAGE_NAME
    atomic_write_new_json(lineage_path, {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "preflight": payload.get("preflight"), "provenance": payload["provenance"], "artifacts": {**payload.get("artifacts", {}), "report": report_ref}, "claim_boundary": payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_self_reference"}, artifact_kind="V10 scientific lineage")
    _require(_read_json(report_path, "V10 report reload").get("result") == payload["result"] and _read_json(lineage_path, "V10 lineage reload").get("result") == payload["result"], "V10 report/lineage reload changed")
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
