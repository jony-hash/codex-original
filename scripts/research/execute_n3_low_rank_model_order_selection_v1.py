"""Execute the bounded N3 low-rank model-order selection consumer."""

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
from torch import nn


def _load_v11_execution() -> Any:
    path = Path(__file__).with_name("execute_lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11.py")
    spec = importlib.util.spec_from_file_location("lightgen_v11_execution_authority_n3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("V11 execution authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V11 = _load_v11_execution()
V10 = V11.V10
_BASE_ATOMIC_WRITE_JSON = V11._BASE_ATOMIC_WRITE_JSON
_BASE_SAVE_AND_VERIFY_CHECKPOINT = V10._save_and_verify_checkpoint

TASK_ID = "EXECUTE_N3_LOW_RANK_MODEL_ORDER_SELECTION_V1"
AGGREGATE_TASK_ID = "AGGREGATE_N3_LOW_RANK_MODEL_ORDER_SELECTION_V1"
OUTER_GATE = "N3_LOW_RANK_MODEL_ORDER_RESOLVED"
CANDIDATE_ADEQUATE = "N3_LOW_RANK_MODEL_ORDER_CANDIDATE_ADEQUATE"
CANDIDATE_NOT_ADEQUATE = "N3_LOW_RANK_MODEL_ORDER_CANDIDATE_NOT_ADEQUATE"
PREFLIGHT_CLASSIFICATION = "N3_LOW_RANK_MODEL_ORDER_AUTHORIZABLE"
NOT_RESOLVED_CLASSIFICATION = "N3_LOW_RANK_MODEL_ORDER_NOT_RESOLVED"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

CONTRACT_PATH = Path(
    "outputs/20260912_n3/runs/"
    "n3_low_rank_model_order_contract_v1_attempt_01/"
    "n3_low_rank_model_order_contract.v1.json"
)
CONTRACT_SHA256 = "d2b72ce978e4f28fdbb89441a0e84bc7e7792726def310beac129e94b9b4b181"

V11_CONTRACT_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract_v11_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract.v11.json"
)
V11_CONTRACT_SHA256 = "6c49a0cff4e08622351438529cebafe9388537d713a5b78b9fabbc953904efff"
V11_INPUT_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_input.v1.json"
)
V11_INPUT_SHA256 = "b0c5c6a80f0cd58be385762f5bf17b703e0f790a76aefb91586705874601b37a"
V11_REPORT_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_report.v1.json"
)
V11_REPORT_SHA256 = "bc603b7a3b7ef1590bc354cb0b62464f3ececcb2390834d864d58bfbe9158184"
V11_LINEAGE_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_lineage.v1.json"
)
V11_LINEAGE_SHA256 = "2c7d7c99a48c6a5a60d6524a38aa9facd6b7e0203138689f3830dcc7308ac9c8"

PREFLIGHT_DIR = Path(
    "outputs/20260912_n3/runs/"
    "n3_low_rank_model_order_preflight_v1_attempt_03"
)
PREFLIGHT_INPUT_NAME = "n3_low_rank_model_order_preflight_input.v1.json"
PREFLIGHT_REPORT_NAME = "n3_low_rank_model_order_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "n3_low_rank_model_order_preflight_lineage.v1.json"

AGGREGATE_DIR = Path(
    "outputs/20260912_n3/runs/"
    "n3_low_rank_model_order_aggregate_v1_attempt_01"
)
AGGREGATE_REPORT_NAME = "n3_low_rank_model_order_aggregate.v1.json"
AGGREGATE_LINEAGE_NAME = "n3_low_rank_model_order_aggregate_lineage.v1.json"

CANDIDATE_RANKS = tuple(range(1, 10))
NEW_SCIENTIFIC_RANKS = tuple(range(1, 9))
REUSED_RANKS = (9,)
OLS_PORT_COUNT = 100
CLASS_COUNT = 10
INIT_SCALE = 0.01
LOGICAL_BATCH_SIZE = 10
PRODUCTION_GRID = 2048
SMOKE_GRID = V11.SMOKE_GRID
PRECHECK_STEPS = V11.PRECHECK_STEPS
SCIENTIFIC_STEPS = 300
CHECKPOINT_STEPS = (0, 100, 200, 300)
LEARNING_RATE = 1e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
BOOTSTRAP_REPLICATES = V11.BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = V11.BOOTSTRAP_SEED
SMOKE_BOOTSTRAP_REPLICATES = V11.SMOKE_BOOTSTRAP_REPLICATES
STAGE_A_FIREWALL = 0.937
SMOOTH_REQUIRED_DELTA = 0.05
RECONSTRUCTION_MAX_DELTA = 0.005
SELECTION_ROWS = V11.SELECTION_ROWS
SELECTION_PER_CLASS = V11.SELECTION_PER_CLASS
SELECTION_SEED = V11.SELECTION_SEED
SPLIT_SEED = V11.SPLIT_SEED
S_TAU = V11.S_TAU
LAMBDA_TRANSFER = 28.48464149920509
LAMBDA_IMAGE = 13.353282014834436
ADAPTER_PARAMETER_NAMES = [
    "adapter.u_real",
    "adapter.u_imag",
    "adapter.v_real",
    "adapter.v_imag",
]

V3_ENCODER_SHA256 = V11.V3_ENCODER_SHA256
V4_GENERATOR_TENSOR_SHA256 = V11.V4_GENERATOR_TENSOR_SHA256
PROTOTYPE_SHA256 = V11.PROTOTYPE_SHA256
SELECTION_INDICES_SHA256 = V11.SELECTION_INDICES_SHA256
SELECTION_LABELS_SHA256 = V11.SELECTION_LABELS_SHA256

CURRENT_RANK: int | None = None


class LowRankAdapter(nn.Module):
    """Complex low-rank adapter with nested exact-identity initialization."""

    def __init__(self, *, device: torch.device) -> None:
        super().__init__()
        if CURRENT_RANK not in CANDIDATE_RANKS:
            raise RuntimeError("N3 candidate rank is not bound")
        rank = int(CURRENT_RANK)
        self.u_real = nn.Parameter(torch.zeros((OLS_PORT_COUNT, rank), device=device, dtype=torch.float32))
        self.u_imag = nn.Parameter(torch.zeros_like(self.u_real))
        self.v_real = nn.Parameter(torch.zeros_like(self.u_real))
        self.v_imag = nn.Parameter(torch.zeros_like(self.u_real))
        with torch.no_grad():
            eye = torch.eye(rank, device=device, dtype=torch.float32) * INIT_SCALE
            self.u_real[:rank].copy_(eye)

    def factors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.complex(self.u_real, self.u_imag), torch.complex(self.v_real, self.v_imag)

    def matrix(self) -> torch.Tensor:
        u, v = self.factors()
        identity = torch.eye(OLS_PORT_COUNT, device=u.device, dtype=torch.complex64)
        return identity + u @ v.conj().transpose(0, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        V10._require(z.ndim == 2 and tuple(z.shape[1:]) == (OLS_PORT_COUNT,) and z.dtype == torch.complex64, "N3 adapter input shape/dtype changed")
        return z @ self.matrix().transpose(0, 1)


def _set_module_value(name: str, value: Any) -> None:
    setattr(V11, name, value)
    setattr(V10, name, value)


def _candidate_prefix(rank: int) -> str:
    return f"n3_low_rank_model_order_candidate_r{rank}"


def _set_rank(rank: int, *, scientific: bool) -> None:
    global CURRENT_RANK
    V10._require(rank in CANDIDATE_RANKS, f"N3 rank {rank} is outside the frozen candidate domain")
    CURRENT_RANK = rank
    shared = {
        "TASK_ID": TASK_ID,
        "OUTER_GATE": OUTER_GATE,
        "SUPPORTED_CLASSIFICATION": CANDIDATE_ADEQUATE,
        "NOT_SUPPORTED_CLASSIFICATION": CANDIDATE_NOT_ADEQUATE,
        "EXPECTED_BRANCH": EXPECTED_BRANCH,
        "EXPECTED_HEAD": EXPECTED_HEAD,
        "CONTRACT_PATH": CONTRACT_PATH,
        "CONTRACT_SHA256": CONTRACT_SHA256,
        "V9_CONTRACT_PATH": V11.V9_CONTRACT_PATH,
        "V9_CONTRACT_SHA256": V11.V9_CONTRACT_SHA256,
        "CLASS_COUNT": CLASS_COUNT,
        "OLS_PORT_COUNT": OLS_PORT_COUNT,
        "ADAPTER_RANK": rank,
        "INIT_SCALE": INIT_SCALE,
        "LOGICAL_BATCH_SIZE": LOGICAL_BATCH_SIZE,
        "PRODUCTION_GRID": PRODUCTION_GRID,
        "SMOKE_GRID": SMOKE_GRID,
        "PRECHECK_STEPS": PRECHECK_STEPS,
        "SCIENTIFIC_STEPS": SCIENTIFIC_STEPS,
        "CHECKPOINT_STEPS": CHECKPOINT_STEPS,
        "LEARNING_RATE": LEARNING_RATE,
        "ADAM_BETAS": ADAM_BETAS,
        "ADAM_EPSILON": ADAM_EPSILON,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "AMSGRAD": AMSGRAD,
        "BOOTSTRAP_REPLICATES": BOOTSTRAP_REPLICATES,
        "BOOTSTRAP_SEED": BOOTSTRAP_SEED,
        "SMOKE_BOOTSTRAP_REPLICATES": SMOKE_BOOTSTRAP_REPLICATES,
        "STAGE_A_FIREWALL": STAGE_A_FIREWALL,
        "SMOOTH_REQUIRED_DELTA": SMOOTH_REQUIRED_DELTA,
        "RECONSTRUCTION_MAX_DELTA": RECONSTRUCTION_MAX_DELTA,
        "SELECTION_ROWS": SELECTION_ROWS,
        "SELECTION_PER_CLASS": SELECTION_PER_CLASS,
        "SELECTION_SEED": SELECTION_SEED,
        "SPLIT_SEED": SPLIT_SEED,
        "S_TAU": S_TAU,
        "LAMBDA_TRANSFER": LAMBDA_TRANSFER,
        "V3_ENCODER_SHA256": V3_ENCODER_SHA256,
        "V4_GENERATOR_TENSOR_SHA256": V4_GENERATOR_TENSOR_SHA256,
        "PROTOTYPE_SHA256": PROTOTYPE_SHA256,
        "SELECTION_INDICES_SHA256": SELECTION_INDICES_SHA256,
        "SELECTION_LABELS_SHA256": SELECTION_LABELS_SHA256,
        "ADAPTER_PARAMETER_NAMES": ADAPTER_PARAMETER_NAMES,
    }
    if scientific:
        prefix = _candidate_prefix(rank)
        shared.update(
            {
                "DEFAULT_SCIENTIFIC_DIR": Path("outputs/20260912_n3/runs") / f"{prefix}_attempt_01",
                "SCIENTIFIC_INPUT_NAME": f"{prefix}_scientific_input.v1.json",
                "SCIENTIFIC_REPORT_NAME": f"{prefix}_scientific_report.v1.json",
                "SCIENTIFIC_LINEAGE_NAME": f"{prefix}_scientific_lineage.v1.json",
                "SCIENTIFIC_SUFFICIENT_NAME": f"{prefix}_sufficient_observations.v1.pt",
                "SCIENTIFIC_RAW_NAME": f"{prefix}_step300_cyclic_raw_outputs.v1.pt",
                "REPORT_SCHEMA": "n3_low_rank_model_order_candidate_scientific_report.v1",
                "LINEAGE_SCHEMA": "n3_low_rank_model_order_candidate_scientific_lineage.v1",
                "INPUT_SCHEMA": "n3_low_rank_model_order_candidate_scientific_input.v1",
                "SUFFICIENT_SCHEMA": "n3_low_rank_model_order_candidate_sufficient_observations.v1",
                "CHECKPOINT_SCHEMA": "n3_low_rank_model_order_candidate_checkpoint.v1",
            }
        )
    else:
        prefix = f"n3_low_rank_model_order_preflight_r{rank}"
        shared.update(
            {
                "DEFAULT_SCIENTIFIC_DIR": Path("outputs/20260912_n3/runs") / f"{_candidate_prefix(rank)}_attempt_01",
                "PREFLIGHT_CHECKPOINT_NAME": f"{prefix}_smoke_checkpoint.v1.pt",
                "PREFLIGHT_SYNTHETIC_OBS_NAME": f"{prefix}_synthetic_sufficient_observations.v1.pt",
                "PREFLIGHT_SYNTHETIC_REPORT_NAME": f"{prefix}_synthetic_report.v1.json",
                "PREFLIGHT_SYNTHETIC_LINEAGE_NAME": f"{prefix}_synthetic_lineage.v1.json",
                "CHECKPOINT_SCHEMA": "n3_low_rank_model_order_preflight_checkpoint.v1",
            }
        )
    for name, value in shared.items():
        _set_module_value(name, value)
    V11.LAMBDA_IMAGE = LAMBDA_IMAGE
    V11.LowRankAdapter = LowRankAdapter
    V10.LowRankAdapter = LowRankAdapter


def _load_contract(root: Path) -> tuple[dict[str, Any], Any]:
    V10._require(CONTRACT_SHA256 != "__TO_BE_BOUND_AFTER_CONTRACT_MATERIALIZATION__", "N3 contract SHA is not bound")
    path = V10._path(root, CONTRACT_PATH)
    V10._validate_hash(path, CONTRACT_SHA256, "N3 contract")
    payload = V10._read_json(path, "N3 contract")
    V10._require(
        payload.get("schema") == "n3_low_rank_model_order_contract.v1"
        and payload.get("task_id") == "DEFINE_N3_LOW_RANK_MODEL_ORDER_CONTRACT_V1"
        and payload.get("result") == "READY"
        and payload.get("status") == "valid_completed"
        and payload.get("outer_gate") == OUTER_GATE,
        "N3 contract is not READY",
    )
    definition = V10._mapping(payload.get("definition_entrypoint"), "N3 definition entrypoint")
    V10._validate_hash(V10._path(root, str(definition["path"])), str(definition["sha256"]), "N3 definition entrypoint")
    git = V10._mapping(payload.get("git"), "N3 contract Git")
    V10._require(git.get("branch") == EXPECTED_BRANCH and git.get("head") == EXPECTED_HEAD, "N3 contract Git identity changed")
    parent = V10._mapping(payload.get("parent_contract"), "N3 parent V11 contract")
    V10._require(parent.get("path") == V11_CONTRACT_PATH.as_posix() and parent.get("sha256") == V11_CONTRACT_SHA256, "N3 V11 parent binding changed")
    model_order = V10._mapping(payload.get("model_order"), "N3 model-order contract")
    V10._require(model_order.get("candidate_ranks") == list(CANDIDATE_RANKS), "N3 candidate domain changed")
    V10._require(model_order.get("new_scientific_ranks") == list(NEW_SCIENTIFIC_RANKS) and model_order.get("reused_ranks") == list(REUSED_RANKS), "N3 rank reuse domain changed")
    objective = V10._mapping(payload.get("objective"), "N3 objective")
    V10._require(float(objective.get("lambda_transfer", 0.0)) == LAMBDA_TRANSFER and float(objective.get("lambda_image_value", 0.0)) == LAMBDA_IMAGE, "N3 objective constants changed")
    protocol = V10._mapping(payload.get("protocol"), "N3 protocol")
    V10._require(protocol.get("same_objective_across_ranks") is True and protocol.get("same_lambdas_across_ranks") is True, "N3 rank objective matching changed")
    V10._validate_hash(V10._path(root, V11_CONTRACT_PATH), V11_CONTRACT_SHA256, "V11 parent contract")
    v11_contract = V10._read_json(V10._path(root, V11_CONTRACT_PATH), "V11 parent contract")
    V10._require(v11_contract.get("result") == "READY" and v11_contract.get("status") == "valid_completed", "V11 parent contract is not READY")
    v2_contract = V10.load_v2_training_contract(V10._path(root, V10.V6.V2_CONTRACT_PATH), repo_root=root)
    V10._require(v2_contract.sha256 == V10.V6.V2_CONTRACT_SHA256, "V2 contract identity changed")
    return payload, v2_contract


def _verify_v11_reuse(root: Path) -> dict[str, Any]:
    contract_path = V10._path(root, V11_CONTRACT_PATH)
    input_path = V10._path(root, V11_INPUT_PATH)
    report_path = V10._path(root, V11_REPORT_PATH)
    lineage_path = V10._path(root, V11_LINEAGE_PATH)
    for path, digest, name in (
        (contract_path, V11_CONTRACT_SHA256, "V11 contract authority"),
        (input_path, V11_INPUT_SHA256, "V11 scientific input authority"),
        (report_path, V11_REPORT_SHA256, "V11 scientific report authority"),
        (lineage_path, V11_LINEAGE_SHA256, "V11 scientific lineage authority"),
    ):
        V10._validate_hash(path, digest, name)
    contract = V10._read_json(contract_path, "V11 contract authority")
    input_payload = V10._read_json(input_path, "V11 scientific input authority")
    report = V10._read_json(report_path, "V11 scientific report authority")
    lineage = V10._read_json(lineage_path, "V11 scientific lineage authority")
    protocol = V10._mapping(input_payload.get("protocol"), "V11 protocol")
    optimizer = V10._mapping(protocol.get("optimizer"), "V11 optimizer")
    report_runtime = V10._mapping(report.get("runtime"), "V11 runtime")
    report_optimizer = V10._mapping(report_runtime.get("optimizer"), "V11 report optimizer")
    adapter_authority = V10._mapping(V10._mapping(report.get("authorities"), "V11 authorities").get("v10_low_rank_adapter"), "V11 adapter authority")
    checks = {
        "contract_ready": contract.get("result") == "READY" and contract.get("status") == "valid_completed",
        "rank9_factorization": adapter_authority.get("rank") == 9 and adapter_authority.get("factor_shapes") == {"U": [100, 9], "V": [100, 9]},
        "rank9_initialization": V10._mapping(contract.get("adapter"), "V11 adapter").get("rank") == 9,
        "objective": protocol.get("objective") == "L_V11=L_sem+lambda_image*L_image" and float(protocol.get("lambda_transfer", 0.0)) == LAMBDA_TRANSFER and float(protocol.get("lambda_image", 0.0)) == LAMBDA_IMAGE,
        "optimizer": optimizer.get("name") == "Adam" and float(optimizer.get("learning_rate", 0.0)) == LEARNING_RATE and protocol.get("batch") == LOGICAL_BATCH_SIZE and protocol.get("steps") == SCIENTIFIC_STEPS,
        "trainable_set": protocol.get("trainable") == ADAPTER_PARAMETER_NAMES and report_optimizer.get("trainable_parameter_names") == ADAPTER_PARAMETER_NAMES,
        "frozen_runtime": report.get("firewall", {}).get("runtime_parameters_all_frozen") is True and report.get("governance", {}).get("encoder_update") is False and report.get("governance", {}).get("generator_update") is False,
        "data_roles": input_payload.get("data_roles") == {"TRAIN": 55000, "DEV": 5000, "calibration": 0, "protected": 0, "official_test": 0, "sealed": 0},
        "valid_scientific_result": report.get("result") == "VALID_COMPLETED" and report.get("status") == "valid_completed" and report.get("scientific_updates") == SCIENTIFIC_STEPS and report.get("decision", {}).get("supported") is True,
        "lineage_binds_report": lineage.get("artifacts", {}).get("report", {}).get("sha256") == V11_REPORT_SHA256,
        "no_downstream": report.get("claim_boundary", {}).get("stage_b_executed") is False and report.get("claim_boundary", {}).get("generation_executed") is False and report.get("claim_boundary", {}).get("protected_accessed") is False and report.get("claim_boundary", {}).get("official_test_accessed") is False and report.get("claim_boundary", {}).get("sealed_accessed") is False,
    }
    V10._require(all(checks.values()), "V11 rank-9 reuse equivalence failed")
    return {
        "status": "REUSED_VALID_AUTHORITY",
        "verified": True,
        "scientific_run_required": False,
        "rank": 9,
        "checks": checks,
        "contract": {"path": V11_CONTRACT_PATH.as_posix(), "sha256": V11_CONTRACT_SHA256},
        "input": {"path": V11_INPUT_PATH.as_posix(), "sha256": V11_INPUT_SHA256},
        "report": {"path": V11_REPORT_PATH.as_posix(), "sha256": V11_REPORT_SHA256},
        "lineage": {"path": V11_LINEAGE_PATH.as_posix(), "sha256": V11_LINEAGE_SHA256},
    }


def _initial_gradient_balance(bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device) -> dict[str, Any]:
    """Measure per-rank gradients without recalibrating the frozen V11 lambda."""

    sampler = V10.DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    batch_indices, batch_labels = sampler.next_batch()
    batch = images.index_select(0, batch_indices).to(device)
    batch_labels = batch_labels.to(device)
    adapter = LowRankAdapter(device=device)
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, PRODUCTION_GRID)
    V10.V8._load_runtime_state(consumer, bundle["initial_state"])
    V10._configure_adapter(consumer, adapter)
    components = V11._BASE_OBJECTIVE_COMPONENTS(
        consumer,
        adapter,
        batch,
        batch_labels,
        bundle["prototypes"].to(device),
        bundle["templates"].to(device),
        float(bundle["s_tau"]),
        require_grad=True,
    )
    semantic_loss = components["anchor_loss"] + LAMBDA_TRANSFER * components["transfer_loss"]
    image_loss = (components["raw_base"] - components["target"]).square().mean()
    V10._finite(semantic_loss, "N3 semantic gradient loss")
    V10._finite(image_loss, "N3 image gradient loss")

    anchor_norms = V11._gradient_norms(adapter, V11._gradient_map(components["anchor_loss"], adapter, retain_graph=True), "N3 initial anchor")
    transfer_norms = V11._gradient_norms(adapter, V11._gradient_map(components["transfer_loss"], adapter, retain_graph=True), "N3 initial transfer")
    semantic_norms = V11._gradient_norms(adapter, V11._gradient_map(semantic_loss, adapter, retain_graph=True), "N3 initial semantic")
    image_norms = V11._gradient_norms(adapter, V11._gradient_map(image_loss, adapter, retain_graph=True), "N3 initial image")
    g_sem = semantic_norms["total_adapter"]
    g_image = image_norms["total_adapter"]
    V10._require(math.isfinite(g_sem) and math.isfinite(g_image) and g_sem > 0.0 and g_image > 0.0, "N3 semantic/image gradient balance is undefined or nonfinite")
    diagnostic_ratio = g_sem / (g_image + 1.0e-12)
    V10._require(math.isfinite(diagnostic_ratio) and diagnostic_ratio > 0.0, "N3 semantic/image gradient diagnostic ratio is undefined or nonfinite")
    V10._require(math.isfinite(LAMBDA_IMAGE) and LAMBDA_IMAGE > 0.0, "N3 frozen V11 lambda_image is unavailable")
    total_loss = semantic_loss + LAMBDA_IMAGE * image_loss
    combined_norms = V11._gradient_norms(adapter, V11._gradient_map(total_loss, adapter, retain_graph=False), "N3 initial combined")
    V10._require(combined_norms["total_adapter"] > 0.0, "N3 total adapter gradient is zero")
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
        "initial_gradient_ratio_diagnostic": diagnostic_ratio,
        "lambda_recalibrated": False,
        "anchor": {"gradient_norms": anchor_norms, "grad_U_may_be_zero": True},
        "transfer": {"gradient_norms": transfer_norms, "grad_U_may_be_zero": True},
        "semantic": {"gradient_norms": semantic_norms},
        "image": {"gradient_norms": image_norms},
        "combined": {"gradient_norms": combined_norms, "usable_direction": True},
    }


def _save_and_verify_checkpoint(root: Path, target: Path, payload: Mapping[str, Any], name: str, consumer: Any, v2_contract: Any) -> dict[str, Any]:
    replacement = name.replace("lightgen_2d_low_rank_ols_to_generator_remap_canary_v10", _candidate_prefix(int(CURRENT_RANK)))
    return _BASE_SAVE_AND_VERIFY_CHECKPOINT(root, target, payload, replacement, consumer, v2_contract)


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any], *, artifact_kind: str) -> None:
    def transform(value: Any) -> Any:
        if isinstance(value, str):
            return value.replace("L_V10=L_anchor+lambda_transfer*L_transfer", "L_V11=L_sem+lambda_image*L_image").replace("L_V10_1=L_anchor+lambda_transfer*L_transfer", "L_V11=L_sem+lambda_image*L_image")
        if isinstance(value, list):
            return [transform(item) for item in value]
        if isinstance(value, dict):
            transformed = {key: transform(item) for key, item in value.items()}
            if transformed.get("task_id") == TASK_ID:
                transformed["candidate_rank"] = CURRENT_RANK
                transformed["candidate_domain"] = list(CANDIDATE_RANKS)
                transformed["model_order_selection"] = True
                if isinstance(transformed.get("objective"), dict):
                    transformed["objective"].update(
                        {
                            "formula": "L_V11=L_sem+lambda_image*L_image",
                            "semantic_objective": "L_sem=L_anchor+lambda_transfer*L_transfer",
                            "lambda_transfer": LAMBDA_TRANSFER,
                            "lambda_image": LAMBDA_IMAGE,
                            "lambda_rule": "reuse exact frozen V11 lambda_image; no rank-specific recalibration",
                            "image_loss_gradient_contribution": "differentiated",
                            "image_loss_role": "differentiated reconstruction compatibility term",
                            "rank": CURRENT_RANK,
                        }
                    )
                if isinstance(transformed.get("protocol"), dict):
                    transformed["protocol"].update(
                        {
                            "objective": "L_V11=L_sem+lambda_image*L_image",
                            "lambda_transfer": LAMBDA_TRANSFER,
                            "lambda_image": LAMBDA_IMAGE,
                            "rank": CURRENT_RANK,
                            "candidate_domain": list(CANDIDATE_RANKS),
                        }
                    )
                if isinstance(transformed.get("governance"), dict):
                    transformed["governance"].update(
                        {
                            "image_loss_gradient": True,
                            "model_order_selection": True,
                            "rank_domain_expansion": False,
                            "rank_specific_lambda_recalibration": False,
                            "rank9_reused": False,
                        }
                    )
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
                if isinstance(transformed.get("claim_boundary"), dict):
                    transformed["claim_boundary"].update({"model_order_candidate": True, "candidate_rank": CURRENT_RANK, "rank9_reused": False})
                if transformed.get("next") == "no_retry; V10 bounded canary does not authorize Stage B or generation":
                    transformed["next"] = "no_retry; N3 rank candidate does not authorize Stage B or generation"
            return transformed
        return value

    _BASE_ATOMIC_WRITE_JSON(path, transform(copy.deepcopy(dict(payload))), artifact_kind=artifact_kind)


def _classify(r0: float, r300: float, smooth_bootstrap: Mapping[str, Any], cyclic: Mapping[str, Any], raw_valid: bool) -> dict[str, Any]:
    V10._require(0 in V11._SCIENTIFIC_DEV_IMAGE_LOSS and 300 in V11._SCIENTIFIC_DEV_IMAGE_LOSS, "N3 DEV image statistic is undefined")
    image_delta = V11._SCIENTIFIC_DEV_IMAGE_LOSS[300] - V11._SCIENTIFIC_DEV_IMAGE_LOSS[0]
    semantic_pass = r300 - r0 >= SMOOTH_REQUIRED_DELTA and float(smooth_bootstrap["ci_lower"]) > 0.0
    broad = bool(cyclic.get("broad")) and not bool(cyclic.get("pair_dominated"))
    template_pass = bool(cyclic["template"]["pass"]) and broad
    smooth_pass = bool(cyclic["smooth"]["pass"]) and broad
    reconstruction_pass = image_delta <= RECONSTRUCTION_MAX_DELTA
    adequate = bool(semantic_pass and reconstruction_pass and template_pass and smooth_pass and raw_valid)
    return {
        "classification": CANDIDATE_ADEQUATE if adequate else CANDIDATE_NOT_ADEQUATE,
        "outer_gate": OUTER_GATE,
        "supported": adequate,
        "model_adequate": adequate,
        "criteria": {
            "semantic_pass": semantic_pass,
            "reconstruction_pass": reconstruction_pass,
            "transfer_pass": bool(template_pass and smooth_pass),
            "known_cyclic_template_transfer_retained": template_pass,
            "known_cyclic_smooth_transfer_retained": smooth_pass,
            "raw_validity": raw_valid,
        },
        "observed": {
            "rank": CURRENT_RANK,
            "DEV_L_image_step0": V11._SCIENTIFIC_DEV_IMAGE_LOSS[0],
            "DEV_L_image_step300": V11._SCIENTIFIC_DEV_IMAGE_LOSS[300],
            "Delta_L_image_DEV": image_delta,
        },
        "thresholds": {
            "smooth_delta": SMOOTH_REQUIRED_DELTA,
            "smooth_ci_lower": 0.0,
            "DEV_image_delta": RECONSTRUCTION_MAX_DELTA,
        },
    }


def _patch_base() -> None:
    V11._patch_base()
    V10.LowRankAdapter = LowRankAdapter
    V11.LowRankAdapter = LowRankAdapter
    V10._load_contract = _load_contract
    V10._initial_gradient_balance = _initial_gradient_balance
    V10._validate_preflight = _validate_preflight
    V10._classify = _classify
    V10._save_and_verify_checkpoint = _save_and_verify_checkpoint
    V10.atomic_write_new_json = _atomic_write_new_json


def _preflight_rank(root: Path, bundle: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device, target: Path, rank: int) -> dict[str, Any]:
    _set_rank(rank, scientific=False)
    rank_target = target / f"rank_{rank}"
    rank_target.mkdir(parents=True, exist_ok=False)
    identity_consumer = V10.V8._new_consumer(bundle["v2_contract"], device, SMOKE_GRID)
    V10.V8._load_runtime_state(identity_consumer, bundle["initial_state"])
    identity_adapter = LowRankAdapter(device=device)
    V10._configure_adapter(identity_consumer, identity_adapter, trainable=False)
    identity = V10._identity_probe(identity_consumer, identity_adapter, images.index_select(0, train_indices[:LOGICAL_BATCH_SIZE]).to(device))
    production_identity = V11._production_identity(bundle, device)
    V10.V6._initial_rng()
    gradient_balance = _initial_gradient_balance(bundle, images, labels, train_indices, device)
    firewall = V11._gradient_firewall_probe(bundle, images, labels, train_indices, device)
    smoke = V11._three_step_smoke(root, bundle, images, labels, train_indices, device, rank_target)
    checks = {
        "step0_identity": identity.get("forward_raw_allclose") is True and identity.get("z_exact_equal") is True and identity.get("frobenius_norm_residual") == 0.0,
        "production_identity": production_identity.get("canonical_vs_identity_adapter_raw_allclose") is True and production_identity.get("coefficient_exact_equal") is True,
        "finite_usable_gradient": bool(gradient_balance["combined"]["usable_direction"]),
        "adapter_only_firewall": firewall.get("runtime_gradient_none") is True and firewall.get("image_loss_gradient_contribution") == "differentiated",
        "one_disposable_step_departure": smoke["one_disposable_optimizer_step"]["departed_from_identity"] is True,
        "checkpoint_reload_resume": smoke["resume_identity"]["runtime"] is True and smoke["resume_identity"]["adapter"] is True and smoke["resume_identity"]["optimizer"] is True and smoke["resume_identity"]["rng"] is True and smoke["resume_identity"]["sampler_next_batch"] is True,
    }
    V10._require(all(checks.values()), f"N3 rank {rank} shared preflight check failed")
    return {
        "rank": rank,
        "checks": checks,
        "identity": identity,
        "production_identity": production_identity,
        "gradient_balance": gradient_balance,
        "firewall": firewall,
        "smoke": smoke,
        "preflight_pass": True,
        "scientific_meaning": "none",
    }


def _compact_synthetic(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scientific_meaning": result.get("scientific_meaning"),
        "identity": result.get("identity"),
        "zero_delta_bootstrap": result.get("zero_delta_bootstrap"),
        "checkpoint": result.get("sufficient_observations"),
        "report": result.get("report"),
        "lineage": result.get("lineage"),
        "report_lineage_reloaded": result.get("report_lineage_reloaded"),
    }


def run_preflight(root: Path, output_dir: Path) -> dict[str, Any]:
    V10._require(not output_dir.exists(), f"refusing to overwrite N3 preflight output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = V10._status_lines(root)
    payload: dict[str, Any] = {
        "schema": "n3_low_rank_model_order_preflight_report.v1",
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "INVALID",
        "status": "invalid_harness",
        "classification": "INVALID_N3_PREFLIGHT_HARNESS",
        "outer_gate": OUTER_GATE,
        "provenance": {"git_start": started},
        "claim_boundary": {
            "engineering_preflight_executed": False,
            "scientific_training_executed": False,
            "scientific_candidate_ranks_started": [],
            "stage_b_executed": False,
            "generation_executed": False,
            "protected_accessed": False,
            "official_test_accessed": False,
            "sealed_accessed": False,
            "formal_promotion": False,
        },
    }
    try:
        V10._require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from N3 authorization")
        _set_rank(1, scientific=False)
        contract, v2_contract = _load_contract(root)
        rank9_reuse = _verify_v11_reuse(root)
        bundle = V10._resolve_authorities(root, contract)
        bundle["v2_contract"] = v2_contract
        dll = V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        bundle["smooth"].to(device)
        images, labels, train_indices, _development_indices, dataset = V10.V6._load_roles(root)
        candidates: dict[str, Any] = {}
        for rank in NEW_SCIENTIFIC_RANKS:
            V10.V6._initial_rng()
            candidates[str(rank)] = _preflight_rank(root, bundle, images, labels, train_indices, device, output_dir, rank)
        _set_rank(1, scientific=False)
        low_synthetic = _compact_synthetic(V11._synthetic_acceptance(root, bundle, device, output_dir / "rank_1"))
        _set_rank(8, scientific=False)
        high_synthetic = _compact_synthetic(V11._synthetic_acceptance(root, bundle, device, output_dir / "rank_8"))
        objective_signatures = {
            str(rank): {
                "objective": candidates[str(rank)]["gradient_balance"]["objective"],
                "semantic_objective": candidates[str(rank)]["gradient_balance"]["semantic_objective"],
                "lambda_transfer": LAMBDA_TRANSFER,
                "lambda_image": LAMBDA_IMAGE,
            }
            for rank in NEW_SCIENTIFIC_RANKS
        }
        V10._require(len({json.dumps(value, sort_keys=True) for value in objective_signatures.values()}) == 1, "N3 objective/lambda signature differs across ranks")
        contract_ref = {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}
        input_payload = {
            "schema": "n3_low_rank_model_order_preflight_input.v1",
            "version": 1,
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "contract": contract_ref,
            "candidate_domain": list(CANDIDATE_RANKS),
            "new_scientific_ranks": list(NEW_SCIENTIFIC_RANKS),
            "reused_ranks": list(REUSED_RANKS),
            "objective": "L_V11=L_sem+lambda_image*L_image",
            "lambda_transfer": LAMBDA_TRANSFER,
            "lambda_image": LAMBDA_IMAGE,
            "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "batch": LOGICAL_BATCH_SIZE, "steps": SCIENTIFIC_STEPS},
            "same_train_order_rng": True,
            "same_evaluation_semantics": True,
            "data_roles": {"TRAIN": 55000, "DEV": 5000, "protected": 0, "official_test": 0, "sealed": 0},
            "one_shot": True,
            "scientific_candidate_ranks_started": [],
        }
        input_path = output_dir / PREFLIGHT_INPUT_NAME
        _atomic_write_new_json(input_path, input_payload, artifact_kind="N3 preflight input")
        payload.update(
            {
                "contract": contract_ref,
                "parent_contract": {"path": V11_CONTRACT_PATH.as_posix(), "sha256": V11_CONTRACT_SHA256},
                "rank9_reuse": rank9_reuse,
                "candidate_domain": list(CANDIDATE_RANKS),
                "new_scientific_ranks": list(NEW_SCIENTIFIC_RANKS),
                "reused_ranks": list(REUSED_RANKS),
                "authorities": bundle["authorities"],
                "runtime": {"cuda_dll_search_path": dll, "device": device_facts, "propagation_grid": PRODUCTION_GRID},
                "data_roles": dataset,
                "objective": {"formula": "L_V11=L_sem+lambda_image*L_image", "semantic_objective": "L_sem=L_anchor+lambda_transfer*L_transfer", "lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE, "rank_specific_recalibration": False},
                "candidates": candidates,
                "objective_signatures": objective_signatures,
                "exact_core_synthetic": {"rank_1": low_synthetic, "rank_8": high_synthetic, "representative_ranks": [1, 8]},
                "preflight_checks": {"v11_reuse": True, "all_candidate_identity": True, "all_candidate_gradient": True, "all_candidate_frozen_firewall": True, "all_candidate_checkpoint_resume": True, "objective_same_across_ranks": True, "exact_core_synthetic_low_high": True, "scientific_candidate_ranks_started": []},
                "input": {"path": V10._relative(root, input_path), "sha256": V10.V7.file_sha256(input_path)},
                "result": "AUTHORIZABLE",
                "status": "valid_completed",
                "classification": PREFLIGHT_CLASSIFICATION,
                "claim_boundary": {**payload["claim_boundary"], "engineering_preflight_executed": True},
                "next": "proceed exactly once through scientific ranks 1..8; reuse rank 9; stop on any hard blocker",
            }
        )
    except V10.HardBlocked as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": error.classification, "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific candidate execution"})
    except (V10.InvalidV10, V10.V9.InvalidV9, V10.V8.InvalidV8, V10.V7.InvalidV7, V10.V6.InvalidV6) as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N3_PREFLIGHT_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no scientific candidate execution"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N3_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12000:]}, "next": "stop; no scientific candidate execution"})
    finally:
        ending = V10.V6.git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == V10._status_lines(root)})
    report_path = output_dir / PREFLIGHT_REPORT_NAME
    _atomic_write_new_json(report_path, payload, artifact_kind="N3 preflight report")
    report_ref = {"path": V10._relative(root, report_path), "sha256": V10.V7.file_sha256(report_path)}
    lineage_path = output_dir / PREFLIGHT_LINEAGE_NAME
    lineage_payload = {"schema": "n3_low_rank_model_order_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "provenance": payload["provenance"], "report": report_ref, "claim_boundary": payload["claim_boundary"]}
    _atomic_write_new_json(lineage_path, lineage_payload, artifact_kind="N3 preflight lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": V10._relative(root, lineage_path), "sha256": V10.V7.file_sha256(lineage_path)}, "scientific_candidate_ranks_started": [], "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _validate_preflight(root: Path) -> dict[str, Any]:
    report_path = V10._path(root, PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = V10._path(root, PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    V10._require(report_path.is_file() and lineage_path.is_file(), "N3 shared preflight authority is unavailable")
    report = V10._read_json(report_path, "N3 shared preflight report")
    lineage = V10._read_json(lineage_path, "N3 shared preflight lineage")
    V10._require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed" and report.get("classification") == PREFLIGHT_CLASSIFICATION, "N3 shared preflight is not AUTHORIZABLE")
    V10._require(report.get("preflight_checks", {}).get("scientific_candidate_ranks_started") == [], "N3 preflight already consumed a scientific rank")
    V10._require(report.get("rank9_reuse", {}).get("verified") is True, "N3 rank-9 reuse is not verified")
    V10._require(lineage.get("report", {}).get("sha256") == V10.V7.file_sha256(report_path), "N3 preflight lineage changed")
    V10._require(CURRENT_RANK in NEW_SCIENTIFIC_RANKS, "N3 scientific candidate rank is not bound")
    candidate = report.get("candidates", {}).get(str(CURRENT_RANK))
    V10._require(isinstance(candidate, dict) and candidate.get("preflight_pass") is True, f"N3 rank {CURRENT_RANK} preflight is unavailable")
    report_for_runner = copy.deepcopy(report)
    report_for_runner["gradient_balance"] = candidate["gradient_balance"]
    report_for_runner["candidate_rank"] = CURRENT_RANK
    return {"report": report_for_runner, "report_ref": {"path": V10._relative(root, report_path), "sha256": V10.V7.file_sha256(report_path)}, "lineage_ref": {"path": V10._relative(root, lineage_path), "sha256": V10.V7.file_sha256(lineage_path)}}


def _rank_summary(root: Path, rank: int, report: Mapping[str, Any], report_ref: Mapping[str, Any], lineage_ref: Mapping[str, Any], *, reused: bool) -> dict[str, Any]:
    evaluations = report.get("evaluations", {})
    smooth = {step: evaluations[str(step)]["smooth"]["macro_recall"] for step in CHECKPOINT_STEPS}
    image = {step: evaluations[str(step)]["losses"]["L_image"] for step in CHECKPOINT_STEPS}
    anchor = {step: {"macro_recall": evaluations[str(step)]["template_anchor"]["macro_recall"], "correct_template_margin_mean": evaluations[str(step)]["template_anchor"]["correct_template_margin"]["mean"]} for step in CHECKPOINT_STEPS}
    cyclic = report["cyclic_transfer"]
    decision = report.get("decision", {})
    adapter = report.get("adapter_diagnostics", {})
    step0_adapter = adapter.get("step0", {})
    step300_adapter = adapter.get("step300", {})
    raw_validity = {
        "DEV_all_checkpoints": all(bool(evaluations[str(step)]["raw_validity"]["finite"] and evaluations[str(step)]["raw_validity"]["structured"] and evaluations[str(step)]["raw_validity"]["nonconstant"]) for step in CHECKPOINT_STEPS),
        "cyclic_base": bool(cyclic["raw_validity"]["base"]["finite"] and cyclic["raw_validity"]["base"]["structured"] and cyclic["raw_validity"]["base"]["nonconstant"]),
        "cyclic_shifted": bool(cyclic["raw_validity"]["shifted"]["finite"] and cyclic["raw_validity"]["shifted"]["structured"] and cyclic["raw_validity"]["shifted"]["nonconstant"]),
    }
    return {
        "rank": rank,
        "source": "REUSED_VALID_AUTHORITY" if reused else "SCIENTIFIC_ATTEMPT_01",
        "report": dict(report_ref),
        "lineage": dict(lineage_ref),
        "result": report.get("result"),
        "status": report.get("status"),
        "classification": report.get("classification"),
        "model_adequate": bool(decision.get("supported")),
        "smooth": {"R0": smooth[0], "R100": smooth[100], "R200": smooth[200], "R300": smooth[300], "Delta": report["natural_smooth_semantics"]["Delta_R"], "bootstrap_ci": {"lower": report["natural_smooth_semantics"]["bootstrap"]["ci_lower"], "upper": report["natural_smooth_semantics"]["bootstrap"]["ci_upper"]}},
        "image_loss_DEV": {str(step): image[step] for step in CHECKPOINT_STEPS},
        "image_loss_delta_DEV": report["step300_minus_step0"]["Delta_L_image_DEV"],
        "anchor_metric": anchor,
        "cyclic_transfer": {"template_mean": cyclic["template"]["bootstrap"]["observed"]["mean"], "template_ci": {"lower": cyclic["template"]["bootstrap"]["ci_lower"], "upper": cyclic["template"]["bootstrap"]["ci_upper"]}, "smooth_mean": cyclic["smooth"]["bootstrap"]["observed"]["mean"], "smooth_ci": {"lower": cyclic["smooth"]["bootstrap"]["ci_lower"], "upper": cyclic["smooth"]["bootstrap"]["ci_upper"]}, "positive_pair_counts": cyclic["positive_pair_counts"], "broad": cyclic["broad"], "pair_dominated": cyclic["pair_dominated"], "known_cyclic_transfer_retained": cyclic["known_cyclic_transfer_retained"]},
        "raw_validity": raw_validity,
        "residual": {"step0_frobenius": step0_adapter.get("frobenius_norm_residual"), "step0_spectral": step0_adapter.get("spectral_norm_residual"), "step300_frobenius": step300_adapter.get("frobenius_norm_residual"), "step300_spectral": step300_adapter.get("spectral_norm_residual"), "step300_max_abs": step300_adapter.get("residual_max_abs")},
        "factor_singular_values_step300": step300_adapter.get("residual_singular_values", []),
        "structural_max_rank": rank,
        "numerical_rank_step300": step300_adapter.get("effective_residual_rank"),
        "stage_a_firewall": report.get("stage_a_firewall"),
        "decision": decision,
    }


def _pareto_frontier(records: Mapping[str, Any]) -> list[int]:
    frontier: list[int] = []
    for rank in sorted(int(key) for key in records):
        candidate = records[str(rank)]
        dominated = False
        for other_rank in sorted(int(key) for key in records):
            if other_rank == rank:
                continue
            other = records[str(other_rank)]
            no_worse = other_rank <= rank and other["smooth"]["R300"] >= candidate["smooth"]["R300"] and other["image_loss_DEV"]["300"] <= candidate["image_loss_DEV"]["300"]
            strict = other_rank < rank or other["smooth"]["R300"] > candidate["smooth"]["R300"] or other["image_loss_DEV"]["300"] < candidate["image_loss_DEV"]["300"]
            if no_worse and strict:
                dominated = True
                break
        if not dominated:
            frontier.append(rank)
    return frontier


def _write_aggregate(root: Path, started: Mapping[str, Any], status_before: list[str], contract_ref: Mapping[str, Any], preflight: Mapping[str, Any], rank9_reuse: Mapping[str, Any], records: Mapping[str, Any]) -> dict[str, Any]:
    adequate = [int(rank) for rank in records if records[rank]["model_adequate"]]
    selected_rank = min(adequate) if adequate else None
    resolved = selected_rank is not None
    classification = OUTER_GATE if resolved else NOT_RESOLVED_CLASSIFICATION
    report_payload = {
        "schema": "n3_low_rank_model_order_aggregate.v1",
        "version": 1,
        "task_id": AGGREGATE_TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "VALID_COMPLETED",
        "status": "valid_completed",
        "classification": classification,
        "outer_gate": OUTER_GATE,
        "contract": dict(contract_ref),
        "preflight": {"report": dict(preflight["report_ref"]), "lineage": dict(preflight["lineage_ref"]), "result": "AUTHORIZABLE"},
        "rank9_reuse": dict(rank9_reuse),
        "candidate_domain": list(CANDIDATE_RANKS),
        "new_scientific_ranks": list(NEW_SCIENTIFIC_RANKS),
        "reused_ranks": list(REUSED_RANKS),
        "candidates": dict(records),
        "model_adequate_ranks": adequate,
        "selection": {"rule": "minimum integer r with MODEL_ADEQUATE_r=true", "selected_rank": selected_rank, "performance_maximization": False, "complexity_penalty": False, "resolved": resolved},
        "rank_performance_table": [{"rank": int(rank), "Smooth_R300": records[rank]["smooth"]["R300"], "Delta_Smooth": records[rank]["smooth"]["Delta"], "DEV_L_image_step300": records[rank]["image_loss_DEV"]["300"], "residual_frobenius_step300": records[rank]["residual"]["step300_frobenius"]} for rank in sorted(records, key=lambda value: int(value))],
        "descriptive_pareto_frontier": {"axes": ["rank complexity", "Smooth R300", "DEV L_image step300"], "ranks": _pareto_frontier(records), "selection_effect": "descriptive only; does not alter r_star"},
        "stage_a_firewall": {str(rank): records[rank]["stage_a_firewall"] for rank in sorted(records, key=lambda value: int(value))},
        "provenance": {"git_start": dict(started), "candidate_runs_completed": list(NEW_SCIENTIFIC_RANKS), "rank9_reused": True},
        "claim_boundary": {"scientific_training_executed": True, "scientific_candidate_ranks": list(NEW_SCIENTIFIC_RANKS), "rank9_scientific_rerun": False, "stage_b_executed": False, "generation_executed": False, "scale_extraction_executed": False, "prior_shaping_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False},
        "governance": {"adapter_update": True, "encoder_update": False, "generator_update": False, "optical_update": False, "rank_domain_expansion": False, "rank_specific_lambda_recalibration": False, "retry": False, "second_arm": False, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False, "generation_readiness": "not_ready"},
        "next": "no retry; model-order selection does not authorize Stage B or generation",
    }
    ending = V10.V6.git_facts(root)
    report_payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == V10._status_lines(root)})
    AGGREGATE_DIR_PATH = V10._path(root, AGGREGATE_DIR)
    V10._require(not AGGREGATE_DIR_PATH.exists(), f"refusing to overwrite N3 aggregate output: {AGGREGATE_DIR_PATH}")
    AGGREGATE_DIR_PATH.mkdir(parents=True, exist_ok=False)
    report_path = AGGREGATE_DIR_PATH / AGGREGATE_REPORT_NAME
    _BASE_ATOMIC_WRITE_JSON(report_path, report_payload, artifact_kind="N3 aggregate report")
    report_ref = {"path": V10._relative(root, report_path), "sha256": V10.V7.file_sha256(report_path)}
    lineage_path = AGGREGATE_DIR_PATH / AGGREGATE_LINEAGE_NAME
    lineage_payload = {"schema": "n3_low_rank_model_order_aggregate_lineage.v1", "version": 1, "task_id": AGGREGATE_TASK_ID, "result": report_payload["result"], "status": report_payload["status"], "classification": report_payload["classification"], "outer_gate": OUTER_GATE, "contract": contract_ref, "preflight": report_payload["preflight"], "report": report_ref, "candidate_count": len(records), "selected_rank": selected_rank, "claim_boundary": report_payload["claim_boundary"]}
    _BASE_ATOMIC_WRITE_JSON(lineage_path, lineage_payload, artifact_kind="N3 aggregate lineage")
    return {"result": report_payload["result"], "status": report_payload["status"], "classification": classification, "outer_gate": OUTER_GATE, "selected_rank": selected_rank, "report": report_ref, "lineage": {"path": V10._relative(root, lineage_path), "sha256": V10.V7.file_sha256(lineage_path)}, "candidate_count": len(records), "stage_b_authorized": False, "generation_readiness": "not_ready", "formal_promotion": False, "next": report_payload["next"]}


def _load_completed_candidate(root: Path, rank: int, candidate_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Consume a complete candidate without rerunning its scientific attempt."""

    report_path = candidate_dir / V10.SCIENTIFIC_REPORT_NAME
    lineage_path = candidate_dir / V10.SCIENTIFIC_LINEAGE_NAME
    V10._require(candidate_dir.is_dir() and report_path.is_file() and lineage_path.is_file(), f"N3 rank {rank} has an incomplete existing scientific output; no retry is allowed")
    report_ref = {"path": V10._relative(root, report_path), "sha256": V10.V7.file_sha256(report_path)}
    lineage_ref = {"path": V10._relative(root, lineage_path), "sha256": V10.V7.file_sha256(lineage_path)}
    report = V10._read_json(report_path, f"N3 rank {rank} scientific report")
    lineage = V10._read_json(lineage_path, f"N3 rank {rank} scientific lineage")
    V10._require(
        report.get("task_id") == TASK_ID
        and report.get("candidate_rank") == rank
        and report.get("candidate_domain") == list(CANDIDATE_RANKS)
        and report.get("result") == "VALID_COMPLETED"
        and report.get("status") == "valid_completed"
        and report.get("scientific_updates") == SCIENTIFIC_STEPS
        and report.get("decision", {}).get("model_adequate") is not None
        and report.get("claim_boundary", {}).get("scientific_300_step_executed") is True,
        f"N3 rank {rank} existing scientific output is not a complete valid attempt; no retry is allowed",
    )
    V10._require(
        lineage.get("task_id") == TASK_ID
        and lineage.get("result") == "VALID_COMPLETED"
        and lineage.get("status") == "valid_completed"
        and lineage.get("candidate_rank") == rank
        and lineage.get("artifacts", {}).get("report", {}).get("path") == report_ref["path"]
        and lineage.get("artifacts", {}).get("report", {}).get("sha256") == report_ref["sha256"],
        f"N3 rank {rank} existing lineage does not bind its report; no retry is allowed",
    )
    return report, lineage, report_ref, lineage_ref


def run_scientific(root: Path) -> dict[str, Any]:
    _set_rank(1, scientific=True)
    preflight = _validate_preflight(root)
    contract, _v2_contract = _load_contract(root)
    rank9_reuse = _verify_v11_reuse(root)
    started = V10.V6.git_facts(root)
    status_before = V10._status_lines(root)
    records: dict[str, Any] = {}
    for rank in NEW_SCIENTIFIC_RANKS:
        _set_rank(rank, scientific=True)
        V11._SCIENTIFIC_DEV_IMAGE_LOSS.clear()
        candidate_dir = V10._path(root, Path("outputs/20260912_n3/runs") / f"{_candidate_prefix(rank)}_attempt_01")
        recovered = candidate_dir.exists()
        try:
            if recovered:
                print(json.dumps({"event": "n3_rank_consume_completed", "rank": rank, "source": "existing_valid_scientific_attempt", "retry": False}, ensure_ascii=False), flush=True)
                report, _lineage, report_ref, lineage_ref = _load_completed_candidate(root, rank, candidate_dir)
            else:
                print(json.dumps({"event": "n3_rank_start", "rank": rank, "candidate_domain": list(CANDIDATE_RANKS)}, ensure_ascii=False), flush=True)
                result = V10.run_scientific(root, candidate_dir)
                if result.get("result") != "VALID_COMPLETED":
                    print(json.dumps({"event": "n3_rank_stop", "rank": rank, "result": result.get("result"), "classification": result.get("classification"), "reason": "scientific candidate did not complete validly; no retry"}, ensure_ascii=False), flush=True)
                    return {"result": result.get("result", "INVALID"), "status": result.get("status", "invalid_harness"), "classification": result.get("classification", "INVALID_N3_SCIENTIFIC_CANDIDATE"), "outer_gate": OUTER_GATE, "candidate_rank": rank, "candidate_result": result, "completed_candidate_ranks": sorted(int(key) for key in records), "rank9_reuse": rank9_reuse, "next": "stop; no scientific retry and no model-order resolution"}
                report, _lineage, report_ref, lineage_ref = _load_completed_candidate(root, rank, candidate_dir)
        except (V10.InvalidV10, V10.V9.InvalidV9, V10.V8.InvalidV8, V10.V7.InvalidV7, V10.V6.InvalidV6) as error:
            print(json.dumps({"event": "n3_rank_stop", "rank": rank, "result": "INVALID", "classification": "INVALID_N3_SCIENTIFIC_CANDIDATE_OUTPUT", "reason": str(error), "retry": False}, ensure_ascii=False), flush=True)
            return {"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N3_SCIENTIFIC_CANDIDATE_OUTPUT", "outer_gate": OUTER_GATE, "candidate_rank": rank, "completed_candidate_ranks": sorted(int(key) for key in records), "rank9_reuse": rank9_reuse, "next": "stop; no scientific retry and no model-order resolution"}
        records[str(rank)] = _rank_summary(root, rank, report, report_ref, lineage_ref, reused=False)
        print(json.dumps({"event": "n3_rank_complete", "rank": rank, "source": "existing_valid_scientific_attempt" if recovered else "scientific_attempt_01", "classification": report.get("classification"), "model_adequate": report.get("decision", {}).get("model_adequate"), "R300": report.get("evaluations", {}).get("300", {}).get("smooth", {}).get("macro_recall"), "DEV_L_image_delta": report.get("decision", {}).get("observed", {}).get("Delta_L_image_DEV")}, ensure_ascii=False), flush=True)
    reused_report = V10._read_json(V10._path(root, V11_REPORT_PATH), "V11 rank-9 report for reuse")
    records["9"] = _rank_summary(root, 9, reused_report, rank9_reuse["report"], rank9_reuse["lineage"], reused=True)
    contract_ref = {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}
    aggregate = _write_aggregate(root, started, status_before, contract_ref, preflight, rank9_reuse, records)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return aggregate


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()
    _patch_base()
    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    if args.action == "preflight":
        result = run_preflight(root, V10._path(root, PREFLIGHT_DIR))
    else:
        result = run_scientific(root)
    return 0 if result["result"] in {"AUTHORIZABLE", "VALID_COMPLETED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
