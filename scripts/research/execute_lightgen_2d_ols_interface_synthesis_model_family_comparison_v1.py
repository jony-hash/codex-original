"""Execute the one-shot N4 OLS/interface-synthesis model-family comparison.

The scientific family is frozen at rank nine and reuses the V11 endpoint.
C1 trains the existing two-layer generator jointly with the adapter.  C2 adds
one trainable co-planar phase-control screen at the generator entrance while
retaining the existing two propagation intervals.  The canonical v2 runtime
is not modified by this research entrypoint.
"""

from __future__ import annotations

import argparse
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

from ldm_project.v2.application.research.lightgen_2d_semantic_generation_input_ols_alignment import (
    ComplexOLSState,
    OLS_SHAPE,
)
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load
from ldm_project.v2.optics.d2nn import FiniteCropPhaseStack


def _load_v11_execution() -> Any:
    path = Path(__file__).with_name(
        "execute_lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11.py"
    )
    spec = importlib.util.spec_from_file_location(
        "lightgen_v11_execution_authority_n4", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("V11 execution authority cannot load")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V11 = _load_v11_execution()
V11._patch_base()
V10 = V11.V10

TASK_ID = "EXECUTE_N4_OLS_INTERFACE_SYNTHESIS_MODEL_FAMILY_COMPARISON_V1"
DESIGN_AUTHORITY = "DEFINE_N4_OLS_INTERFACE_SYNTHESIS_MODEL_FAMILY_CONTRACT_V1"
OUTER_GATE = "N4_OLS_INTERFACE_SYNTHESIS_COMPARISON_RESOLVED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

BASE_OUTPUT = Path("outputs/20260913_n4/runs")
CONTRACT_DIR = BASE_OUTPUT / "lightgen_2d_ols_interface_synthesis_model_family_contract_v1_attempt_04"
CONTRACT_PATH = CONTRACT_DIR / "lightgen_2d_ols_interface_synthesis_model_family_contract.v1.json"
PREFLIGHT_DIR = BASE_OUTPUT / "lightgen_2d_ols_interface_synthesis_model_family_preflight_n4_attempt_04"
CONTINUATION_CONTRACT_DIR = BASE_OUTPUT / "lightgen_2d_ols_interface_synthesis_model_family_contract_v1_attempt_05"
CONTINUATION_PREFLIGHT_DIR = BASE_OUTPUT / "lightgen_2d_ols_interface_synthesis_model_family_preflight_n4_attempt_05"
C1_DIR = BASE_OUTPUT / "lightgen_2d_ols_interface_synthesis_model_family_c1_attempt_01"
C2_DIR = BASE_OUTPUT / "lightgen_2d_ols_interface_synthesis_model_family_c2_attempt_01"
AGGREGATE_DIR = BASE_OUTPUT / "lightgen_2d_ols_interface_synthesis_model_family_aggregate_n4_attempt_01"

PREFLIGHT_INPUT_NAME = "lightgen_2d_n4_model_family_preflight_input.v1.json"
PREFLIGHT_REPORT_NAME = "lightgen_2d_n4_model_family_preflight_report.v1.json"
PREFLIGHT_LINEAGE_NAME = "lightgen_2d_n4_model_family_preflight_lineage.v1.json"
PREFLIGHT_C1_SMOKE_NAME = "lightgen_2d_n4_c1_preflight_smoke_checkpoint.v1.pt"
PREFLIGHT_C2_SMOKE_NAME = "lightgen_2d_n4_c2_preflight_smoke_checkpoint.v1.pt"
PREFLIGHT_SYNTHETIC_REPORT_NAME = "lightgen_2d_n4_{candidate}_synthetic_report.v1.json"
PREFLIGHT_SYNTHETIC_LINEAGE_NAME = "lightgen_2d_n4_{candidate}_synthetic_lineage.v1.json"
PREFLIGHT_SYNTHETIC_OBS_NAME = "lightgen_2d_n4_{candidate}_synthetic_sufficient_observations.v1.pt"

SCIENTIFIC_INPUT_NAME = "lightgen_2d_n4_{candidate}_scientific_input.v1.json"
SCIENTIFIC_REPORT_NAME = "lightgen_2d_n4_{candidate}_scientific_report.v1.json"
SCIENTIFIC_LINEAGE_NAME = "lightgen_2d_n4_{candidate}_scientific_lineage.v1.json"
SCIENTIFIC_SUFFICIENT_NAME = "lightgen_2d_n4_{candidate}_sufficient_observations.v1.pt"
SCIENTIFIC_RAW_NAME = "lightgen_2d_n4_{candidate}_step300_cyclic_raw_outputs.v1.pt"
CHECKPOINT_NAME = "lightgen_2d_n4_{candidate}_step_{step}_checkpoint.v1.pt"

REPORT_SCHEMA = "lightgen_2d_n4_model_family_{candidate}_scientific_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_n4_model_family_{candidate}_scientific_lineage.v1"
INPUT_SCHEMA = "lightgen_2d_n4_model_family_{candidate}_scientific_input.v1"
SUFFICIENT_SCHEMA = "lightgen_2d_n4_model_family_{candidate}_sufficient_observations.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_n4_model_family_{candidate}_checkpoint.v1"

V11_CONTRACT_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract_v11_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_contract.v11.json"
)
V11_CONTRACT_SHA256 = "6c49a0cff4e08622351438529cebafe9388537d713a5b78b9fabbc953904efff"
V11_REPORT_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_report.v1.json"
)
V11_REPORT_SHA256 = "bc603b7a3b7ef1590bc354cb0b62464f3ececcb2390834d864d58bfbe9158184"
V11_CHECKPOINT_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/"
    "lightgen_2d_low_rank_ols_to_generator_remap_canary_v10_step_300_checkpoint.v1.pt"
)
V11_CHECKPOINT_SHA256 = "9098019d03ae1e5a0afa2b77a864cea56ca220c6cfbde2df5a831352a72339c8"
V11_SUFFICIENT_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_scientific_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_sufficient_observations.v1.pt"
)
V11_SUFFICIENT_SHA256 = "2c4e623ce93c00dd8f865366cf7e4b3acfcea2bb203ac8f0e6b15532e61e7da9"
V11_PREFLIGHT_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_preflight_v11_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_preflight_report.v1.json"
)
V11_PREFLIGHT_LINEAGE_PATH = Path(
    "outputs/20260912_v11/runs/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_preflight_v11_attempt_01/"
    "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_preflight_lineage.v1.json"
)
V11_PREFLIGHT_REPORT_SHA256 = "b5078cdd61664f447ab880ad32f57b6f0d4287ea88f446c111414560f6743386"
V11_PREFLIGHT_LINEAGE_SHA256 = "4910a4de5b928f97c4c1ecf9af178ad66e3cf54ad94e22f5899a4f3f9fe4f695"
N3_AGGREGATE_PATH = Path(
    "outputs/20260912_n3/runs/"
    "n3_low_rank_model_order_aggregate_v1_attempt_01/n3_low_rank_model_order_aggregate.v1.json"
)
N3_AGGREGATE_SHA256 = "1b6bd6af18462f717fa19532e1a981e872a4e48fe6498ed561890f6570ae89de"

CLASS_COUNT = 10
OLS_PORT_COUNT = 100
ADAPTER_RANK = 9
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
ARCHITECTURE_BASELINE = 0.3056
ARCHITECTURE_REQUIRED_DELTA = 0.05
RECONSTRUCTION_MAX_DELTA = 0.005
SELECTION_ROWS = 1000
SELECTION_PER_CLASS = 100
SELECTION_SEED = 4321
SPLIT_SEED = 1234
S_TAU = 0.02919053753794475
LAMBDA_TRANSFER = 28.48464149920509
LAMBDA_IMAGE = 13.353282014834436
V3_ENCODER_SHA256 = "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca"
V4_GENERATOR_TENSOR_SHA256 = "83aa74b701049c09c5efb63dbd6d3a6d181184a964d3d3c0c06665ca97da5afe"
PROTOTYPE_SHA256 = "c02a728e8631e24918b018553958a44b25cdb358192c6b4b28239f55e93df912"
SELECTION_INDICES_SHA256 = "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8"
SELECTION_LABELS_SHA256 = "1f376234300ef9c7a60a5e895c2dabb52550292724c4b806adf5315e8ab5f251"
ADAPTER_PARAMETER_NAMES = ["adapter.u_real", "adapter.u_imag", "adapter.v_real", "adapter.v_imag"]
ADAPTER_STATE_NAMES = ["u_real", "u_imag", "v_real", "v_imag"]
TOLERANCE = {"rtol": 1.0e-5, "atol": 1.0e-6}


class InvalidN4(RuntimeError):
    """The N4 harness or a required authority is invalid."""


class HardBlockedN4(RuntimeError):
    """A valid N4 boundary prevents scientific execution."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidN4(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    _require(not value.is_absolute(), f"absolute path is not allowed: {value}")
    return (root / value).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidN4(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(value, dict), f"{name} must be a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    return V10.V7.file_sha256(path)


def _validate_hash(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    observed = _file_sha256(path)
    _require(observed == expected, f"{name} SHA mismatch: expected {expected}, observed {observed}")


def _cpu_tree(value: Any) -> Any:
    return V10._cpu_tree(value)


def _tree_equal(left: Any, right: Any) -> bool:
    return V10._tree_equal(left, right)


def _digest_tree(value: Any) -> str:
    return V10._digest_tree(value)


def _finite(value: torch.Tensor, name: str) -> None:
    V10._finite(value, name)


def _norm(value: torch.Tensor, name: str) -> float:
    return V10._norm(value, name)


def _status_lines(root: Path) -> list[str]:
    return V10._status_lines(root)


def _contract_payload(root: Path) -> dict[str, Any]:
    return {
        "schema": "lightgen_2d_ols_interface_synthesis_model_family_contract.v1",
        "version": 1,
        "result": "READY",
        "status": "valid_completed",
        "semantic_authority": DESIGN_AUTHORITY,
        "outer_gate": OUTER_GATE,
        "git": {"branch": EXPECTED_BRANCH, "head": EXPECTED_HEAD},
        "execution_entrypoint": {
            "path": _relative(root, Path(__file__).resolve()),
            "sha256": _file_sha256(Path(__file__).resolve()),
        },
        "authorities": {
            "v11_contract": {"path": V11_CONTRACT_PATH.as_posix(), "sha256": V11_CONTRACT_SHA256},
            "v11_scientific_report": {"path": V11_REPORT_PATH.as_posix(), "sha256": V11_REPORT_SHA256},
            "v11_step300_checkpoint": {"path": V11_CHECKPOINT_PATH.as_posix(), "sha256": V11_CHECKPOINT_SHA256},
            "v11_sufficient_observations": {"path": V11_SUFFICIENT_PATH.as_posix(), "sha256": V11_SUFFICIENT_SHA256},
            "n3_aggregate": {"path": N3_AGGREGATE_PATH.as_posix(), "sha256": N3_AGGREGATE_SHA256},
        },
        "family": {
            "fixed_rank": ADAPTER_RANK,
            "baseline": "C0=reused V11 rank9 step300 endpoint",
            "c0": {"trainable": [], "adapter_parameters": 3600, "generator_phase_parameters": 20000},
            "c1": {"trainable": ADAPTER_PARAMETER_NAMES + ["generator.phase"], "adapter_parameters": 3600, "generator_phase_parameters": 20000, "total_trainable_parameters": 23600},
            "c2": {"trainable": ADAPTER_PARAMETER_NAMES + ["generator.extra_phase", "generator.stack.phase"], "adapter_parameters": 3600, "generator_phase_parameters": 30000, "total_trainable_parameters": 33600, "label": "2-layer generator + additional co-planar trainable phase screen"},
        },
        "c2_neutral_initialization": {
            "extra_screen": "N_phi0(x)=x*exp(i*phi0)",
            "phi0": "all zeros",
            "propagation_intervals": 2,
            "additional_asm_or_crop": False,
            "step0_identity": "N_0=I and G_C2(step0)=G_C1(step0)=G_C0",
            "not_standard_three_propagation_layer_d2nn": True,
        },
        "objective": {
            "formula": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image",
            "lambda_transfer": LAMBDA_TRANSFER,
            "lambda_image": LAMBDA_IMAGE,
            "lambda_image_source": "V11 frozen step0 gradient balance; no N4 recalibration",
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "betas": list(ADAM_BETAS),
            "eps": ADAM_EPSILON,
            "weight_decay": WEIGHT_DECAY,
            "amsgrad": AMSGRAD,
            "batch": LOGICAL_BATCH_SIZE,
            "updates_per_new_candidate": SCIENTIFIC_STEPS,
            "evaluation_steps": list(CHECKPOINT_STEPS),
            "production_grid": PRODUCTION_GRID,
            "train_order_seed": SPLIT_SEED,
        },
        "data_roles": {"TRAIN": 55000, "DEV": 5000, "calibration": 0, "protected": 0, "official_test": 0, "sealed": 0},
        "selection": {"rows": SELECTION_ROWS, "per_class": SELECTION_PER_CLASS, "seed": SELECTION_SEED, "indices_sha256": SELECTION_INDICES_SHA256, "labels_sha256": SELECTION_LABELS_SHA256, "target_rule": "t=(y+1) mod 10"},
        "criteria": {
            "architecture_delta": "R_candidate(step300)-0.3056 >= 0.05",
            "architecture_paired_ci": "paired bootstrap CI lower > 0",
            "reconstruction": "DEV L_image(step300)-DEV L_image(step0) <= 0.005",
            "cyclic": "template mean/CI positive and Smooth mean/CI positive, broad and not pair-dominated",
            "raw": "finite, expected shape, structured, nonconstant",
            "stage_a": "Smooth R300 >= 0.937, independent and non-authorizing",
            "selection": "both adequate -> C1; only C2 adequate -> C2; neither -> N4_LINEAR_INTERFACE_SYNTHESIS_CAPACITY_NOT_RESOLVED",
        },
        "folding": {"target": "W_OLS_tilde(z)=W_OLS(Tz)", "basis": "v_tilde_k=sum_j T_jk v_j", "training": False, "physical_realizability_claim": False},
        "one_shot": {"c0_reuse": True, "c1_attempts": 1, "c2_attempts": 1, "retry": False, "adaptive_truncation": False, "post_hoc_criteria": False},
        "claim_boundary": {"stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False},
    }


def _materialize_contract(root: Path) -> dict[str, Any]:
    _require(not CONTRACT_DIR.exists(), f"refusing to reuse N4 contract directory: {CONTRACT_DIR}")
    CONTRACT_DIR.mkdir(parents=True, exist_ok=False)
    payload = _contract_payload(root)
    atomic_write_new_json(CONTRACT_PATH, payload, artifact_kind="N4 model-family contract")
    loaded = _read_json(CONTRACT_PATH, "N4 model-family contract reload")
    _require(loaded == payload, "N4 model-family contract reload changed")
    return {"path": _relative(root, CONTRACT_PATH), "sha256": _file_sha256(CONTRACT_PATH), "schema": payload["schema"]}


def _load_contract(root: Path, contract_ref: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _path(root, CONTRACT_PATH)
    _require(path.is_file(), f"N4 contract is unavailable: {path}")
    if contract_ref is not None:
        _validate_hash(path, str(contract_ref["sha256"]), "N4 contract")
    payload = _read_json(path, "N4 model-family contract")
    _require(payload == _contract_payload(root), "N4 contract content or execution identity changed")
    _require(payload["result"] == "READY" and payload["status"] == "valid_completed", "N4 contract is not READY")
    return payload, {"path": _relative(root, path), "sha256": _file_sha256(path), "schema": payload["schema"]}


def _validate_v11_authorities(root: Path) -> dict[str, Any]:
    _validate_hash(_path(root, V11_CONTRACT_PATH), V11_CONTRACT_SHA256, "V11 contract")
    _validate_hash(_path(root, V11_REPORT_PATH), V11_REPORT_SHA256, "V11 scientific report")
    _validate_hash(_path(root, V11_CHECKPOINT_PATH), V11_CHECKPOINT_SHA256, "V11 step300 checkpoint")
    _validate_hash(_path(root, V11_SUFFICIENT_PATH), V11_SUFFICIENT_SHA256, "V11 sufficient observations")
    _validate_hash(_path(root, V11_PREFLIGHT_PATH), V11_PREFLIGHT_REPORT_SHA256, "V11 preflight report")
    _validate_hash(_path(root, V11_PREFLIGHT_LINEAGE_PATH), V11_PREFLIGHT_LINEAGE_SHA256, "V11 preflight lineage")
    _validate_hash(_path(root, N3_AGGREGATE_PATH), N3_AGGREGATE_SHA256, "N3 aggregate")

    v11_report = _read_json(_path(root, V11_REPORT_PATH), "V11 scientific report")
    _require(v11_report.get("result") == "VALID_COMPLETED", "V11 report is not VALID_COMPLETED")
    _require(v11_report.get("classification") == "V11_LOW_RANK_SEMANTIC_RECONSTRUCTION_COMPATIBILITY_SUPPORTED", "V11 semantic authority changed")
    _require(int(v11_report.get("scientific_updates", 0)) == SCIENTIFIC_STEPS, "V11 update count changed")
    _require(math.isclose(float(v11_report["natural_smooth_semantics"]["R300"]), ARCHITECTURE_BASELINE, rel_tol=0.0, abs_tol=1.0e-15), "V11 baseline Smooth changed")
    _require(math.isclose(float(v11_report["evaluations"]["300"]["losses"]["L_image"]), 0.0531686320602894, rel_tol=0.0, abs_tol=1.0e-15), "V11 baseline DEV image loss changed")
    _require(v11_report.get("claim_boundary", {}).get("stage_b_executed") is False and v11_report.get("claim_boundary", {}).get("generation_executed") is False, "V11 claim boundary changed")
    _require(v11_report["checkpoints"]["300"]["path"] == V11_CHECKPOINT_PATH.as_posix(), "V11 endpoint checkpoint path changed")

    v11_preflight = _read_json(_path(root, V11_PREFLIGHT_PATH), "V11 preflight report")
    v11_lineage = _read_json(_path(root, V11_PREFLIGHT_LINEAGE_PATH), "V11 preflight lineage")
    _require(v11_preflight.get("result") == "AUTHORIZABLE" and v11_preflight.get("status") == "valid_completed", "V11 preflight is not authorizable")
    _require(v11_preflight.get("claim_boundary", {}).get("scientific_300_step_executed") is False, "V11 preflight scientific boundary changed")
    _require(v11_lineage.get("report", {}).get("sha256") == V11_PREFLIGHT_REPORT_SHA256, "V11 preflight lineage/report binding changed")
    _require(float(v11_preflight["objective"]["lambda_image"]) == LAMBDA_IMAGE, "V11 lambda_image authority changed")

    n3 = _read_json(_path(root, N3_AGGREGATE_PATH), "N3 aggregate")
    _require(n3.get("result") == "VALID_COMPLETED" and n3.get("classification") == "N3_LOW_RANK_MODEL_ORDER_RESOLVED", "N3 aggregate authority changed")
    _require(int(n3.get("selection", {}).get("selected_rank", 0)) == 1, "N3 selected rank changed")

    checkpoint = _mapping(safe_torch_load(_path(root, V11_CHECKPOINT_PATH), expected_sha256=V11_CHECKPOINT_SHA256, map_location="cpu"), "V11 checkpoint")
    _require(checkpoint.get("checkpoint_step") == 300 and checkpoint.get("schema") == "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_checkpoint.v1", "V11 checkpoint schema/step changed")
    _require(checkpoint.get("lambda_image") == LAMBDA_IMAGE and checkpoint.get("lambda_transfer") == LAMBDA_TRANSFER, "V11 checkpoint objective changed")
    runtime_state = dict(_mapping(checkpoint.get("runtime_state"), "V11 endpoint runtime state"))
    adapter_state = dict(_mapping(checkpoint.get("adapter_state"), "V11 endpoint adapter state"))
    _require(set(runtime_state) == {"encoder.phase", "generator.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}, "V11 endpoint runtime state surface changed")
    _require(tuple(runtime_state["encoder.phase"].shape) == (2, 1, 100, 100), "V11 endpoint encoder shape changed")
    _require(tuple(runtime_state["generator.phase"].shape) == (2, 1, 100, 100), "V11 endpoint generator shape changed")
    _require(V10.V6.tensor_sha256(runtime_state["encoder.phase"]) == V3_ENCODER_SHA256, "V11 endpoint encoder is not V3 CONTROL")
    _require(V10.V6.tensor_sha256(runtime_state["generator.phase"]) == V4_GENERATOR_TENSOR_SHA256, "V11 endpoint generator is not V4")
    _require(set(adapter_state) == set(ADAPTER_STATE_NAMES), "V11 endpoint adapter state surface changed")
    _require(all(tuple(adapter_state[name].shape) == (OLS_PORT_COUNT, ADAPTER_RANK) for name in ADAPTER_STATE_NAMES), "V11 endpoint adapter shape changed")
    _require(torch.equal(runtime_state["ols.operator.transport"], torch.eye(OLS_PORT_COUNT, dtype=torch.complex64)), "V11 endpoint T is not identity")

    sufficient = _mapping(safe_torch_load(_path(root, V11_SUFFICIENT_PATH), expected_sha256=V11_SUFFICIENT_SHA256, map_location="cpu"), "V11 sufficient observations")
    _require(sufficient.get("schema") == "lightgen_2d_low_rank_semantic_reconstruction_compatibility_v11_sufficient_observations.v1", "V11 sufficient observation schema changed")
    _require(tuple(sufficient["step300_smooth_logits"].shape) == (5000, CLASS_COUNT), "V11 endpoint DEV logits changed")

    v11_contract, v2_contract = V11._load_contract(root)
    authorities = V10._resolve_authorities(root, v11_contract)
    _require(V10.V6.tensor_sha256(authorities["prototypes"]) == PROTOTYPE_SHA256, "TRAIN prototypes changed")
    _require(not authorities["prototypes"].requires_grad and not authorities["templates"].requires_grad, "support tensors require gradients")
    _require(math.isclose(float(authorities["s_tau"]), S_TAU, rel_tol=0.0, abs_tol=1.0e-15), "s_tau changed")
    return {
        "v11_report": v11_report,
        "v11_preflight": v11_preflight,
        "v11_contract": v11_contract,
        "v2_contract": v2_contract,
        "authorities": authorities,
        "c0_runtime_state": runtime_state,
        "c0_adapter_state": adapter_state,
        "c0_smooth_logits": sufficient["step300_smooth_logits"].detach().cpu(),
        "c0_sufficient": sufficient,
        "lambda_image": LAMBDA_IMAGE,
        "c0_runtime_digest": _digest_tree(_cpu_tree(runtime_state)),
        "c0_adapter_digest": _digest_tree(_cpu_tree(adapter_state)),
    }


class CoPlanarPhaseControlGenerator(nn.Module):
    """Two propagated phase screens plus one trainable co-planar entrance screen."""

    def __init__(self, *, propagation_shape: tuple[int, int], spacing_m: float, device: torch.device) -> None:
        super().__init__()
        self.extra_phase = nn.Parameter(torch.zeros((1, 1, *OLS_SHAPE), device=device, dtype=torch.float32))
        self.stack = FiniteCropPhaseStack(
            layers=2,
            active_shape=OLS_SHAPE,
            propagation_shape=propagation_shape,
            spacing_m=spacing_m,
        ).to(device)

    def forward(self, field: Any, prefix: str) -> tuple[Any, tuple[Any, ...], dict[str, torch.Tensor]]:
        _require(tuple(field.plane.shape) == tuple(OLS_SHAPE), "C2 entrance plane shape changed")
        transfer = torch.polar(torch.ones_like(self.extra_phase), self.extra_phase).to(field.tensor.dtype)
        extra_tensor = field.tensor * transfer
        extra_plane = field.plane.with_plane(
            f"{prefix}_coplanar_phase_control",
            z_m=field.plane.z_m,
            coordinate_frame=field.plane.coordinate_frame,
        )
        extra_field = type(field)(
            extra_tensor,
            extra_plane,
            field.provenance.__class__(
                "n4_coplanar_phase_control",
                field.plane.plane_id,
                None,
                {"phase_screen": "additional_generator_entry", "neutral_initialization": "phi0=0", "propagation": False},
            ),
        )
        output, records, stats = self.stack(extra_field, prefix)
        return output, (extra_field, *records), {"crop_retention": stats["crop_retention"], "phase": torch.cat((self.extra_phase, self.stack.phase), dim=0)}


def _runtime_base_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(state)
    if "generator.stack.phase" in result:
        result["generator.phase"] = result.pop("generator.stack.phase")
        result.pop("generator.extra_phase", None)
    return result


def _install_c2_generator(consumer: Any) -> None:
    base_generator = consumer.runtime.generator
    _require(tuple(base_generator.phase.shape) == (2, 1, 100, 100), "C2 base generator phase shape changed")
    wrapper = CoPlanarPhaseControlGenerator(
        propagation_shape=tuple(base_generator.propagation_shape),
        spacing_m=float(base_generator.spacing_m),
        device=consumer.device,
    )
    with torch.no_grad():
        wrapper.stack.phase.copy_(base_generator.phase.detach())
        wrapper.extra_phase.zero_()
    consumer.runtime.generator = wrapper


def _new_candidate(bundle: Mapping[str, Any], candidate: str, device: torch.device, grid: int, *, runtime_state: Mapping[str, Any] | None = None, adapter_state: Mapping[str, Any] | None = None) -> tuple[Any, Any]:
    _require(candidate in {"C1", "C2"}, f"unknown N4 candidate: {candidate}")
    state = dict(runtime_state or bundle["c0_runtime_state"])
    consumer = V10.V8._new_consumer(bundle["v2_contract"], device, grid)
    if candidate == "C1":
        V10.V8._load_runtime_state(consumer, state)
    else:
        V10.V8._load_runtime_state(consumer, _runtime_base_state(state))
        _install_c2_generator(consumer)
        if runtime_state is not None:
            consumer.runtime.load_state_dict(state, strict=True)
    adapter = V11.LowRankAdapter(device=device)
    adapter.load_state_dict(dict(adapter_state or bundle["c0_adapter_state"]), strict=True)
    _configure_candidate(consumer, adapter, candidate)
    return consumer, adapter


def _runtime_trainable_names(candidate: str) -> list[str]:
    return ["generator.phase"] if candidate == "C1" else ["generator.extra_phase", "generator.stack.phase"]


def _candidate_named(consumer: Any, adapter: Any, candidate: str) -> dict[str, nn.Parameter]:
    named: dict[str, nn.Parameter] = {f"adapter.{name}": parameter for name, parameter in adapter.named_parameters()}
    runtime_names = _runtime_trainable_names(candidate)
    runtime_named = dict(consumer.runtime.named_parameters())
    _require(set(runtime_names).issubset(runtime_named), f"{candidate} generator parameter surface changed")
    named.update({name: runtime_named[name] for name in runtime_names})
    expected = ADAPTER_PARAMETER_NAMES + runtime_names
    _require(list(named) == expected, f"{candidate} trainable parameter order changed")
    return named


def _configure_candidate(consumer: Any, adapter: Any, candidate: str, *, trainable: bool = True) -> dict[str, Any]:
    V10.V8._configure_all_frozen(consumer)
    named = _candidate_named(consumer, adapter, candidate)
    runtime_trainable = set(_runtime_trainable_names(candidate))
    for name, parameter in named.items():
        parameter.requires_grad_(trainable)
        parameter.grad = None
    runtime_named = dict(consumer.runtime.named_parameters())
    _require(all((parameter.requires_grad is trainable) for name, parameter in named.items()), f"{candidate} trainable firewall failed")
    _require(all(not parameter.requires_grad for name, parameter in runtime_named.items() if name not in runtime_trainable), f"{candidate} frozen runtime firewall failed")
    if trainable:
        consumer.runtime.train()
    else:
        consumer.runtime.eval()
    return {
        "candidate": candidate,
        "trainable": list(named) if trainable else [],
        "frozen_runtime": [name for name in runtime_named if name not in runtime_trainable],
        "trainable_count": int(sum(parameter.numel() for parameter in named.values())) if trainable else 0,
        "parameter_counts": {name: int(parameter.numel()) for name, parameter in named.items()},
        "runtime_parameter_names": list(runtime_named),
        "runtime_parameters_all_frozen_except_declared": True,
        "ols_frozen": True,
        "smooth_evaluator_frozen": True,
    }


def _optimizer(consumer: Any, adapter: Any, candidate: str) -> torch.optim.Adam:
    named = _candidate_named(consumer, adapter, candidate)
    parameters = list(named.values())
    _require(all(parameter.requires_grad for parameter in parameters), f"{candidate} optimizer received frozen parameter")
    optimizer = torch.optim.Adam(parameters, lr=LEARNING_RATE, betas=ADAM_BETAS, eps=ADAM_EPSILON, weight_decay=WEIGHT_DECAY, amsgrad=AMSGRAD)
    _require(len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["params"] == parameters, f"{candidate} optimizer firewall changed")
    return optimizer


def _objective_components(consumer: Any, adapter: Any, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, s_tau: float, *, require_grad: bool = True) -> dict[str, Any]:
    base = V11._BASE_OBJECTIVE_COMPONENTS(
        consumer,
        adapter,
        batch,
        labels,
        prototypes,
        templates,
        s_tau,
        require_grad=require_grad,
    )
    image_loss = (base["raw_base"] - base["target"]).square().mean()
    semantic_loss = base["anchor_loss"] + LAMBDA_TRANSFER * base["transfer_loss"]
    total_loss = semantic_loss + LAMBDA_IMAGE * image_loss
    _finite(image_loss, "N4 image loss")
    _finite(semantic_loss, "N4 semantic loss")
    _finite(total_loss, "N4 total loss")
    return {**base, "image_loss": image_loss, "semantic_loss": semantic_loss, "total_loss": total_loss}


def _frozen_runtime_digest(consumer: Any, candidate: str) -> str:
    trainable = set(_runtime_trainable_names(candidate))
    return _digest_tree({key: value for key, value in _cpu_tree(consumer.runtime.state_dict()).items() if key not in trainable})


def _generator_observation(consumer: Any, candidate: str, initial_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    state = consumer.runtime.state_dict()
    keys = ["generator.phase"] if candidate == "C1" else ["generator.stack.phase", "generator.extra_phase"]
    result: dict[str, Any] = {}
    for key in keys:
        value = state[key].detach().cpu().to(torch.float64)
        initial = initial_state[key].detach().cpu().to(torch.float64)
        delta = value - initial
        result[key] = {"phase_change_frobenius": float(torch.linalg.vector_norm(delta).item()), "phase_change_max_abs": float(delta.abs().max().item()), "phase_frobenius": float(torch.linalg.vector_norm(value).item()), "phase_max_abs": float(value.abs().max().item()), "phase_mean": float(value.mean().item()), "phase_std": float(value.std(unbiased=False).item())}
    return result


def _candidate_gradients(bundle: Mapping[str, Any], candidate: str, images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device, *, grid: int) -> dict[str, Any]:
    sampler = V10.DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    indices, batch_labels = sampler.next_batch()
    batch = images.index_select(0, indices).to(device)
    batch_labels = batch_labels.to(device)
    consumer, adapter = _new_candidate(bundle, candidate, device, grid)
    components = _objective_components(consumer, adapter, batch, batch_labels, bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), S_TAU)
    named = _candidate_named(consumer, adapter, candidate)
    values: dict[str, dict[str, Any]] = {}
    for objective_name, loss in (("semantic", components["semantic_loss"]), ("image", components["image_loss"]), ("combined", components["total_loss"])):
        gradients = torch.autograd.grad(loss, tuple(named.values()), retain_graph=True, allow_unused=False)
        per_group = {name: _norm(gradient, f"N4 {candidate} {objective_name} {name} gradient") for name, gradient in zip(named, gradients, strict=True)}
        values[objective_name] = {"per_parameter": per_group, "total": float(math.sqrt(sum(value * value for value in per_group.values())))}
        _require(math.isfinite(values[objective_name]["total"]) and values[objective_name]["total"] > 0.0, f"{candidate} {objective_name} gradient is unusable")
    wrong_anchor = F.cross_entropy(components["anchor_logits"], (batch_labels + 1) % CLASS_COUNT) + LAMBDA_IMAGE * components["image_loss"]
    wrong_transfer = LAMBDA_TRANSFER * F.softplus(components["delta"]).mean() + LAMBDA_IMAGE * components["image_loss"]
    negative: dict[str, Any] = {}
    for name, loss in (("wrong_label_anchor", wrong_anchor), ("wrong_direction_transfer", wrong_transfer)):
        gradients = torch.autograd.grad(loss, tuple(named.values()), retain_graph=name == "wrong_label_anchor", allow_unused=False)
        per_group = {parameter_name: _norm(gradient, f"N4 {candidate} {name} gradient") for parameter_name, gradient in zip(named, gradients, strict=True)}
        total = float(math.sqrt(sum(value * value for value in per_group.values())))
        _require(math.isfinite(total) and total > 0.0, f"{candidate} negative control gradient is unusable")
        negative[name] = {"per_parameter": per_group, "total": total, "expected": "nonzero differentiated path; not scientific evidence"}
    _require(all(parameter.grad is None for parameter in bundle["authorities"]["smooth"].parameters()), "Smooth evaluator gradient firewall failed")
    _require(all(parameter.grad is None for name, parameter in consumer.runtime.named_parameters() if name not in _runtime_trainable_names(candidate)), f"{candidate} frozen runtime received a gradient")
    return {"candidate": candidate, "first_batch_indices_sha256": V10.V6.tensor_sha256(indices), "first_batch_labels_sha256": V10.V6.tensor_sha256(batch_labels), "batch_size": int(batch.shape[0]), "seed": SPLIT_SEED, "lambda_image": LAMBDA_IMAGE, "lambda_source": "V11 preflight/scientific authority", "gradients": values, "negative_controls": negative, "semantic_image_gradient_defined": True}


def _step0_snapshot(consumer: Any, adapter: Any, batch: torch.Tensor) -> dict[str, Any]:
    with torch.no_grad():
        forward = V10._forward_adapter(consumer, adapter, batch, require_grad=False)
    result = forward["adapted_result"]
    return {"z_adapted": forward["z_adapted"].detach().cpu(), "generator_input": result.generator_input_field.tensor.detach().cpu(), "generator_records": tuple(record.tensor.detach().cpu() for record in result.generator_records), "raw": forward["raw_base"].detach().cpu(), "plane": result.generator_input_field.plane}


def _step0_equivalence(bundle: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    synthetic = torch.linspace(0.0, 1.0, LOGICAL_BATCH_SIZE * 28 * 28, dtype=torch.float32).reshape(LOGICAL_BATCH_SIZE, 1, 28, 28).to(device)
    c0_consumer, c0_adapter = _new_candidate(bundle, "C1", device, SMOKE_GRID)
    _configure_candidate(c0_consumer, c0_adapter, "C1", trainable=False)
    c0 = _step0_snapshot(c0_consumer, c0_adapter, synthetic)
    result: dict[str, Any] = {"batch_size": LOGICAL_BATCH_SIZE, "tolerance": TOLERANCE, "c0_source": "V11 rank9 step300 endpoint", "candidates": {}}
    for candidate in ("C1", "C2"):
        consumer, adapter = _new_candidate(bundle, candidate, device, SMOKE_GRID)
        _configure_candidate(consumer, adapter, candidate, trainable=False)
        current = _step0_snapshot(consumer, adapter, synthetic)
        comparisons: dict[str, Any] = {}
        coefficient_diff = (current["z_adapted"] - c0["z_adapted"]).abs()
        input_diff = (current["generator_input"] - c0["generator_input"]).abs()
        raw_diff = (current["raw"] - c0["raw"]).abs()
        comparisons["adapted_coefficients"] = {"allclose": bool(torch.allclose(current["z_adapted"], c0["z_adapted"], **TOLERANCE)), "max_abs_diff": float(coefficient_diff.max())}
        comparisons["generator_entrance"] = {"allclose": bool(torch.allclose(current["generator_input"], c0["generator_input"], **TOLERANCE)), "max_abs_diff": float(input_diff.max())}
        comparisons["final_raw_cmos"] = {"allclose": bool(torch.allclose(current["raw"], c0["raw"], **TOLERANCE)), "max_abs_diff": float(raw_diff.max())}
        if candidate == "C1":
            _require(len(current["generator_records"]) == 2 and len(c0["generator_records"]) == 2, "C1 generator record count changed")
            comparisons["generator_intermediates"] = [{"allclose": bool(torch.allclose(left, right, **TOLERANCE)), "max_abs_diff": float((left - right).abs().max())} for left, right in zip(current["generator_records"], c0["generator_records"], strict=True)]
        else:
            _require(len(current["generator_records"]) == 3 and len(c0["generator_records"]) == 2, "C2 generator record count changed")
            comparisons["extra_screen_neutral"] = {"allclose": bool(torch.allclose(current["generator_records"][0], c0["generator_input"], **TOLERANCE)), "max_abs_diff": float((current["generator_records"][0] - c0["generator_input"]).abs().max())}
            comparisons["existing_generator_intermediates"] = [{"allclose": bool(torch.allclose(left, right, **TOLERANCE)), "max_abs_diff": float((left - right).abs().max())} for left, right in zip(current["generator_records"][1:], c0["generator_records"], strict=True)]
            extra = consumer.runtime.generator.extra_phase.detach()
            comparisons["extra_phase_exact_zero"] = bool(torch.equal(extra, torch.zeros_like(extra)))
        all_checks = [value["allclose"] for key, value in comparisons.items() if isinstance(value, dict) and "allclose" in value]
        if isinstance(comparisons.get("generator_intermediates"), list):
            all_checks.extend(item["allclose"] for item in comparisons["generator_intermediates"])
        if isinstance(comparisons.get("existing_generator_intermediates"), list):
            all_checks.extend(item["allclose"] for item in comparisons["existing_generator_intermediates"])
        _require(all(all_checks) and comparisons.get("extra_phase_exact_zero", True), f"{candidate} step0 equivalence failed")
        result["candidates"][candidate] = comparisons
    result["passed"] = True
    return result


def _train_step(consumer: Any, adapter: Any, optimizer: torch.optim.Adam, candidate: str, batch: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, step: int) -> dict[str, Any]:
    before_frozen = _frozen_runtime_digest(consumer, candidate)
    before = {name: parameter.detach().cpu().clone() for name, parameter in _candidate_named(consumer, adapter, candidate).items()}
    optimizer.zero_grad(set_to_none=True)
    components = _objective_components(consumer, adapter, batch, labels, prototypes, templates, S_TAU)
    total_loss = components["total_loss"]
    _require(total_loss.requires_grad and components["semantic_loss"].requires_grad and components["image_loss"].requires_grad, f"{candidate} objective path detached")
    total_loss.backward()
    named = _candidate_named(consumer, adapter, candidate)
    gradients = {}
    for name, parameter in named.items():
        _require(parameter.grad is not None, f"{candidate} {name} gradient missing")
        gradients[name] = _norm(parameter.grad, f"{candidate} {name} gradient")
    _require(all(parameter.grad is None for name, parameter in consumer.runtime.named_parameters() if name not in _runtime_trainable_names(candidate)), f"{candidate} frozen runtime received a gradient")
    optimizer.step()
    if consumer.device.type == "cuda":
        torch.cuda.synchronize(consumer.device)
    _require(before_frozen == _frozen_runtime_digest(consumer, candidate), f"{candidate} frozen runtime state changed")
    updates = {name: float((parameter.detach().cpu() - before[name]).to(torch.float64).square().sum().sqrt()) for name, parameter in named.items()}
    _require(all(math.isfinite(value) and value > 0.0 for value in updates.values()), f"{candidate} declared trainable group did not update")
    return {"step": step, "objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "L_anchor": float(components["anchor_loss"].detach().cpu()), "L_transfer": float(components["transfer_loss"].detach().cpu()), "L_sem": float(components["semantic_loss"].detach().cpu()), "L_image": float(components["image_loss"].detach().cpu()), "L_total": float(total_loss.detach().cpu()), "Delta_template_mean": float(components["delta"].detach().to(torch.float64).mean().cpu()), "gradient_norms": gradients, "update_norms": updates, "trainable_parameter_names": list(named), "frozen_runtime_parameter_names": [name for name, parameter in consumer.runtime.named_parameters() if name not in _runtime_trainable_names(candidate)], "image_loss_in_differentiated_objective": True, "smooth_evaluator_in_objective": False, "adapter_updated": True, "generator_updated": True, "encoder_updated": False, "ols_updated": False}


def _validate_sampler_resume(bundle: Mapping[str, Any], loaded_sampler_state: Mapping[str, Any], *, step: int) -> None:
    """Validate checkpoint resume without consuming the scientific sampler at its terminal step."""
    if step == SCIENTIFIC_STEPS:
        _require(int(loaded_sampler_state.get("cursor", -1)) == SCIENTIFIC_STEPS, "N4 final sampler cursor is not exhausted")
        _require(int(loaded_sampler_state.get("steps", -1)) == SCIENTIFIC_STEPS, "N4 final sampler budget changed")
        return
    _require(0 <= step < SCIENTIFIC_STEPS, "N4 checkpoint step is outside the sampler budget")
    sampler_clone = V10.DeterministicTrainRoleSampler(bundle["train_indices"], bundle["labels"], seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    sampler_clone.load_state_dict(dict(loaded_sampler_state))
    sampler_probe = V10.DeterministicTrainRoleSampler(bundle["train_indices"], bundle["labels"], seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    sampler_probe.load_state_dict(dict(loaded_sampler_state))
    next_a, labels_a = sampler_clone.next_batch()
    next_b, labels_b = sampler_probe.next_batch()
    _require(torch.equal(next_a, next_b) and torch.equal(labels_a, labels_b), "N4 sampler resume identity changed")


def _save_checkpoint(root: Path, target: Path, bundle: Mapping[str, Any], candidate: str, consumer: Any, adapter: Any, optimizer: torch.optim.Adam, sampler: Any, *, step: int, initial_frozen_digest: str, initial_generator_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    state = _cpu_tree(consumer.runtime.state_dict())
    adapter_state = _cpu_tree(adapter.state_dict())
    payload = {"schema": CHECKPOINT_SCHEMA.format(candidate=candidate.lower()), "version": 1, "task_id": TASK_ID, "candidate": candidate, "contract": {"path": _relative(root, CONTRACT_PATH), "sha256": _file_sha256(CONTRACT_PATH)}, "checkpoint_step": step, "runtime_state": state, "adapter_state": adapter_state, "optimizer_state": _cpu_tree(optimizer.state_dict()), "sampler_state": _cpu_tree(sampler.state_dict()), "rng_state": _cpu_tree(V10.V6._capture_rng()), "initial_c0_runtime_digest": bundle["c0_runtime_digest"], "initial_c0_adapter_digest": bundle["c0_adapter_digest"], "initial_frozen_runtime_digest": initial_frozen_digest, "prototypes_sha256": PROTOTYPE_SHA256, "templates_sha256": V10.V6.tensor_sha256(bundle["authorities"]["templates"]), "s_tau": S_TAU, "lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE, "objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "trainable_parameter_names": ADAPTER_PARAMETER_NAMES + _runtime_trainable_names(candidate), "data_roles": {"TRAIN": True, "DEV": True, "protected": False, "official_test": False, "sealed": False}, "stage_b_executed": False, "generation_executed": False, "formal_promotion": False}
    path = target / CHECKPOINT_NAME.format(candidate=candidate.lower(), step=step)
    atomic_torch_save_new(path, payload, artifact_kind=f"N4 {candidate} checkpoint")
    digest = _file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"N4 {candidate} checkpoint reload")
    _require(loaded.get("schema") == payload["schema"] and loaded.get("checkpoint_step") == step and loaded.get("candidate") == candidate, f"N4 {candidate} checkpoint schema changed")
    clone, clone_adapter = _new_candidate(bundle, candidate, consumer.device, consumer.runtime.config.propagation_grid_size, runtime_state=loaded["runtime_state"], adapter_state=loaded["adapter_state"])
    _require(_digest_tree(_cpu_tree(clone.runtime.state_dict())) == _digest_tree(_cpu_tree(loaded["runtime_state"])), f"N4 {candidate} runtime checkpoint reload changed")
    _require(_digest_tree(_cpu_tree(clone_adapter.state_dict())) == _digest_tree(_cpu_tree(loaded["adapter_state"])), f"N4 {candidate} adapter checkpoint reload changed")
    clone_optimizer = _optimizer(clone, clone_adapter, candidate)
    clone_optimizer.load_state_dict(dict(loaded["optimizer_state"]))
    _require(_tree_equal(_cpu_tree(clone_optimizer.state_dict()), loaded["optimizer_state"]), f"N4 {candidate} optimizer checkpoint reload changed")
    _validate_sampler_resume(bundle, loaded["sampler_state"], step=step)
    current_rng = V10.V6._capture_rng()
    V10.V6._restore_rng(_mapping(loaded["rng_state"], f"N4 {candidate} checkpoint RNG"))
    reloaded_rng = V10.V6._capture_rng()
    V10.V6._restore_rng(current_rng)
    _require(_tree_equal(reloaded_rng, loaded["rng_state"]), f"N4 {candidate} RNG checkpoint reload changed")
    return {"path": _relative(root, path), "sha256": digest, "step": step, "reloaded": True, "resume_identity": True, "runtime_state_digest": _digest_tree(state), "adapter_state_digest": _digest_tree(adapter_state)}


def _save_sufficient(root: Path, target: Path, candidate: str, observations: Mapping[str, Any]) -> dict[str, Any]:
    path = target / SCIENTIFIC_SUFFICIENT_NAME.format(candidate=candidate.lower())
    payload = {"schema": SUFFICIENT_SCHEMA.format(candidate=candidate.lower()), "version": 1, "task_id": TASK_ID, "candidate": candidate, "contract": {"path": _relative(root, CONTRACT_PATH), "sha256": _file_sha256(CONTRACT_PATH)}, **{key: value.cpu() if torch.is_tensor(value) else value for key, value in observations.items()}}
    atomic_torch_save_new(path, payload, artifact_kind=f"N4 {candidate} sufficient observations")
    digest = _file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"N4 {candidate} sufficient observations reload")
    _require(loaded.get("schema") == payload["schema"] and loaded.get("candidate") == candidate, f"N4 {candidate} sufficient observation reload changed")
    for key, value in observations.items():
        if torch.is_tensor(value):
            _require(torch.equal(loaded[key], value.cpu()), f"N4 {candidate} sufficient observation tensor changed: {key}")
    return {"path": _relative(root, path), "sha256": digest, "schema": payload["schema"], "reloaded": True}


def _save_raw(root: Path, target: Path, candidate: str, cyclic: Mapping[str, Any]) -> dict[str, Any]:
    path = target / SCIENTIFIC_RAW_NAME.format(candidate=candidate.lower())
    observations = cyclic["_observations"]
    payload = {"schema": f"lightgen_2d_n4_{candidate.lower()}_cyclic_raw_outputs.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, "contract": {"path": _relative(root, CONTRACT_PATH), "sha256": _file_sha256(CONTRACT_PATH)}, **{key: value.cpu() if torch.is_tensor(value) else value for key, value in observations.items()}}
    atomic_torch_save_new(path, payload, artifact_kind=f"N4 {candidate} cyclic raw outputs")
    digest = _file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"N4 {candidate} cyclic raw reload")
    _require(loaded.get("schema") == payload["schema"] and torch.equal(loaded["selection_indices"], observations["selection_indices"].cpu()), f"N4 {candidate} cyclic raw reload changed")
    return {"path": _relative(root, path), "sha256": digest, "reloaded": True}


def _evaluate(consumer: Any, adapter: Any, evaluator: Any, images: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, templates: torch.Tensor, *, step: int, train_trace: Mapping[str, Any] | None) -> dict[str, Any]:
    old_objective = V10._objective_components
    V10._objective_components = _objective_components
    try:
        result = V11._BASE_EVALUATE(consumer, adapter, evaluator, images, labels, prototypes, templates, S_TAU, step=step, train_trace=train_trace)
    finally:
        V10._objective_components = old_objective
    return result


def _generator_step_observation(consumer: Any, candidate: str, initial_state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return _generator_observation(consumer, candidate, initial_state)


def _synthetic_acceptance(root: Path, bundle: Mapping[str, Any], candidate: str, device: torch.device, target: Path) -> dict[str, Any]:
    ramp = torch.linspace(0.0, 1.0, 28 * 28, dtype=torch.float32).reshape(1, 1, 28, 28)
    labels = torch.arange(CLASS_COUNT, dtype=torch.long)
    images = (ramp + labels.view(-1, 1, 1, 1).to(torch.float32) / 20.0).clamp(0.0, 1.0)
    eval_labels = labels.repeat_interleave(2)
    eval_images = images.index_select(0, eval_labels)
    consumer, adapter = _new_candidate(bundle, candidate, device, SMOKE_GRID)
    optimizer = _optimizer(consumer, adapter, candidate)
    traces = []
    for step in range(1, PRECHECK_STEPS + 1):
        traces.append(_train_step(consumer, adapter, optimizer, candidate, images.to(device), labels.to(device), bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), step))
    evaluation = _evaluate(consumer, adapter, bundle["authorities"]["smooth"], eval_images, eval_labels, bundle["authorities"]["prototypes"], bundle["authorities"]["templates"], step=PRECHECK_STEPS, train_trace=traces[-1])
    cyclic = V10._cyclic_transfer_v10(consumer, adapter, bundle["authorities"]["smooth"], eval_images, eval_labels, torch.arange(eval_labels.numel()), bundle["authorities"]["prototypes"], bundle["authorities"]["templates"])
    obs = {"labels": eval_labels, "synthetic_smooth_logits": evaluation["_smooth_logits"], "synthetic_ols_margin": evaluation["_margins"], **cyclic["_observations"]}
    sufficient_path = target / PREFLIGHT_SYNTHETIC_OBS_NAME.format(candidate=candidate.lower())
    payload = {"schema": f"lightgen_2d_n4_{candidate.lower()}_synthetic_sufficient_observations.v1", "version": 1, "task_id": TASK_ID, "candidate": candidate, **{key: value.cpu() if torch.is_tensor(value) else value for key, value in obs.items()}}
    atomic_torch_save_new(sufficient_path, payload, artifact_kind=f"N4 {candidate} synthetic sufficient observations")
    digest = _file_sha256(sufficient_path)
    loaded = _mapping(safe_torch_load(sufficient_path, expected_sha256=digest, map_location="cpu"), f"N4 {candidate} synthetic sufficient reload")
    _require(loaded.get("schema") == payload["schema"], f"N4 {candidate} synthetic sufficient reload changed")
    report_payload = {"schema": f"lightgen_2d_n4_{candidate.lower()}_synthetic_report.v1", "scientific_meaning": "none", "candidate": candidate, "training_updates": PRECHECK_STEPS, "traces": traces, "evaluation": {key: value for key, value in evaluation.items() if not key.startswith("_")}, "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")}, "sufficient_observations": {"path": _relative(root, sufficient_path), "sha256": digest}, "classification": "SYNTHETIC_ONLY_NO_SCIENTIFIC_MEANING"}
    report_path = target / PREFLIGHT_SYNTHETIC_REPORT_NAME.format(candidate=candidate.lower())
    atomic_write_new_json(report_path, report_payload, artifact_kind=f"N4 {candidate} synthetic report")
    report_ref = {"path": _relative(root, report_path), "sha256": _file_sha256(report_path)}
    lineage_path = target / PREFLIGHT_SYNTHETIC_LINEAGE_NAME.format(candidate=candidate.lower())
    atomic_write_new_json(lineage_path, {"schema": f"lightgen_2d_n4_{candidate.lower()}_synthetic_lineage.v1", "result": "VALID_COMPLETED", "scientific_meaning": "none", "report": report_ref, "claim_boundary": {"scientific_training": False, "protected": False, "official_test": False, "sealed": False}}, artifact_kind=f"N4 {candidate} synthetic lineage")
    lineage = _read_json(lineage_path, f"N4 {candidate} synthetic lineage reload")
    _require(lineage.get("report", {}).get("sha256") == report_ref["sha256"], f"N4 {candidate} synthetic lineage reload changed")
    return {"scientific_meaning": "none", "training_updates": PRECHECK_STEPS, "sufficient_observations": {"path": _relative(root, sufficient_path), "sha256": digest, "reloaded": True}, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _file_sha256(lineage_path), "reloaded": True}, "classification": "SYNTHETIC_ONLY_NO_SCIENTIFIC_MEANING"}


def _three_step_smoke(root: Path, bundle: Mapping[str, Any], candidate: str, images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, device: torch.device, target: Path) -> dict[str, Any]:
    consumer, adapter = _new_candidate(bundle, candidate, device, PRODUCTION_GRID)
    firewall = _configure_candidate(consumer, adapter, candidate)
    optimizer = _optimizer(consumer, adapter, candidate)
    sampler = V10.DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
    initial_frozen = _frozen_runtime_digest(consumer, candidate)
    initial_generator = _cpu_tree(consumer.runtime.state_dict())
    traces = []
    for step in range(1, PRECHECK_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        traces.append(_train_step(consumer, adapter, optimizer, candidate, images.index_select(0, indices).to(device), batch_labels.to(device), bundle["authorities"]["prototypes"].to(device), bundle["authorities"]["templates"].to(device), step))
    checkpoint = _save_checkpoint(root, target, bundle, candidate, consumer, adapter, optimizer, sampler, step=PRECHECK_STEPS, initial_frozen_digest=initial_frozen, initial_generator_state=initial_generator)
    return {"firewall": firewall, "traces": traces, "checkpoint": checkpoint, "generator_step1": _generator_observation(consumer, candidate, initial_generator), "scientific_updates": 0, "scientific_meaning": "none", "every_declared_group_updated": True, "checkpoint_resume": True}


def _preflight_input_payload(root: Path, contract_ref: Mapping[str, Any], device_facts: Mapping[str, Any], equivalence: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": "lightgen_2d_n4_model_family_preflight_input.v1", "version": 1, "task_id": TASK_ID, "design_authority": DESIGN_AUTHORITY, "contract": dict(contract_ref), "outer_gate": OUTER_GATE, "candidates": ["C0", "C1", "C2"], "objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "lambda_image": LAMBDA_IMAGE, "rank": ADAPTER_RANK, "runtime": {"device": dict(device_facts), "production_grid": PRODUCTION_GRID, "smoke_grid": SMOKE_GRID}, "step0_equivalence": dict(equivalence), "one_shot": {"C0_reuse": True, "C1": 1, "C2": 1, "retry": False}, "data_roles": {"TRAIN": 55000, "DEV": 5000, "protected": 0, "official_test": 0, "sealed": 0}, "scientific_300_step_executed": False}


def _validate_known_c1_invalid(root: Path) -> dict[str, Any]:
    c1_report_path = _path(root, C1_DIR) / SCIENTIFIC_REPORT_NAME.format(candidate="c1")
    c1_report = _read_json(c1_report_path, "existing N4 C1 scientific report")
    _require(c1_report.get("result") == "INVALID", "continuation requires the existing C1 invalid report")
    _require(c1_report.get("classification") == "INVALID_N4_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "continuation requires the known C1 harness error")
    _require(int(c1_report.get("scientific_updates_completed", -1)) == SCIENTIFIC_STEPS, "continuation requires C1 to have completed 300 updates")
    _require("sampler is exhausted" in str(c1_report.get("unexpected_error", {}).get("message", "")), "continuation requires the final sampler-boundary error")
    return c1_report


def run_preflight(root: Path, output_dir: Path, *, allow_known_c1_invalid: bool = False) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to reuse N4 preflight directory: {output_dir}")
    if allow_known_c1_invalid:
        c1_report = _validate_known_c1_invalid(root)
        _require(not C2_DIR.exists() and not AGGREGATE_DIR.exists(), "N4 continuation output boundary is not fresh")
    else:
        _require(not C1_DIR.exists() and not C2_DIR.exists() and not AGGREGATE_DIR.exists(), "N4 scientific/aggregate output already exists; one-shot boundary is ambiguous")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = _status_lines(root)
    payload: dict[str, Any] = {"schema": "lightgen_2d_n4_model_family_preflight_report.v1", "version": 1, "task_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N4_PREFLIGHT", "outer_gate": OUTER_GATE, "provenance": {"git_start": started}, "claim_boundary": {"engineering_preflight_executed": False, "scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}}
    contract_ref: dict[str, Any] | None = None
    try:
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from N4 authorization")
        contract_ref = _materialize_contract(root)
        contract, _ = _load_contract(root, contract_ref)
        bundle = _validate_v11_authorities(root)
        dll = V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = V10.V6._load_roles(root)
        bundle.update({"images": images, "labels": labels, "train_indices": train_indices, "development_indices": development_indices})
        V10.V6._initial_rng()
        equivalence = _step0_equivalence(bundle, device)
        candidate_preflight: dict[str, Any] = {}
        for candidate in ("C1", "C2"):
            V10.V6._initial_rng()
            gradient = _candidate_gradients(bundle, candidate, images, labels, train_indices, device, grid=PRODUCTION_GRID)
            smoke = _three_step_smoke(root, bundle, candidate, images, labels, train_indices, device, output_dir)
            synthetic = _synthetic_acceptance(root, bundle, candidate, device, output_dir)
            candidate_preflight[candidate] = {"gradient_probe": gradient, "three_step_smoke": smoke, "synthetic_acceptance": synthetic, "trainable_parameter_names": ADAPTER_PARAMETER_NAMES + _runtime_trainable_names(candidate), "trainable_parameter_count": 23600 if candidate == "C1" else 33600}
        contract_ref = {"path": _relative(root, CONTRACT_PATH), "sha256": _file_sha256(CONTRACT_PATH), "schema": contract["schema"]}
        input_path = output_dir / PREFLIGHT_INPUT_NAME
        atomic_write_new_json(input_path, _preflight_input_payload(root, contract_ref, device_facts, equivalence), artifact_kind="N4 preflight input")
        payload.update({"design_authority": DESIGN_AUTHORITY, "contract": contract_ref, "authorities": {"v11_contract": {"path": V11_CONTRACT_PATH.as_posix(), "sha256": V11_CONTRACT_SHA256}, "v11_report": {"path": V11_REPORT_PATH.as_posix(), "sha256": V11_REPORT_SHA256}, "v11_checkpoint": {"path": V11_CHECKPOINT_PATH.as_posix(), "sha256": V11_CHECKPOINT_SHA256}, "n3_aggregate": {"path": N3_AGGREGATE_PATH.as_posix(), "sha256": N3_AGGREGATE_SHA256}}, "runtime": {"cuda_dll_search_path": dll, "device": device_facts, "production_grid": PRODUCTION_GRID}, "data_roles": dataset, "baseline": {"C0": "V11 rank9 step300 endpoint", "Smooth_R300": ARCHITECTURE_BASELINE, "DEV_L_image": 0.0531686320602894, "runtime_digest": bundle["c0_runtime_digest"], "adapter_digest": bundle["c0_adapter_digest"]}, "objective": {"formula": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE, "lambda_source": "V11 frozen step0 gradient balance"}, "step0_equivalence": equivalence, "candidates": candidate_preflight, "input": {"path": _relative(root, input_path), "sha256": _file_sha256(input_path)}, "preflight_checks": {"authorities": True, "c0_reuse": True, "c1_c2_step0_equivalence": True, "c2_neutral_identity": True, "trainable_frozen_firewall": True, "finite_usable_gradients": True, "negative_control_rejection": True, "three_step_real_updates": True, "checkpoint_reload_resume": True, "exact_core_synthetic": True}, "continuation": {"known_c1_invalid_harness": allow_known_c1_invalid, "known_c1_report": {"path": _relative(root, _path(root, C1_DIR) / SCIENTIFIC_REPORT_NAME.format(candidate="c1")), "scientific_updates_completed": int(c1_report.get("scientific_updates_completed", 0))} if allow_known_c1_invalid else None}, "result": "AUTHORIZABLE", "status": "valid_completed", "classification": "LIGHTGEN_2D_N4_MODEL_FAMILY_AUTHORIZABLE", "claim_boundary": {"engineering_preflight_executed": True, "scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}, "next": "proceed to exactly one C2 300-update scientific run; C1 was consumed and invalid harness" if allow_known_c1_invalid else "proceed directly to exactly one C1 and one C2 300-update scientific run"})
    except HardBlockedN4 as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": error.classification, "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific execution"})
    except (InvalidN4, V10.InvalidV10, V10.V9.InvalidV9, V10.V8.InvalidV8, V10.V7.InvalidV7, V10.V6.InvalidV6) as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N4_PREFLIGHT_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no scientific execution"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N4_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12000:]}, "next": "stop; no scientific execution"})
    finally:
        ending = V10.V6.git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == _status_lines(root)})
    report_path = output_dir / PREFLIGHT_REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="N4 preflight report")
    report_ref = {"path": _relative(root, report_path), "sha256": _file_sha256(report_path)}
    lineage_path = output_dir / PREFLIGHT_LINEAGE_NAME
    atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_n4_model_family_preflight_lineage.v1", "version": 1, "task_id": TASK_ID, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "report": report_ref, "provenance": payload["provenance"], "claim_boundary": payload["claim_boundary"]}, artifact_kind="N4 preflight lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _file_sha256(lineage_path)}, "scientific_300_step_executed": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _validate_preflight(root: Path) -> dict[str, Any]:
    report_path = _path(root, PREFLIGHT_DIR / PREFLIGHT_REPORT_NAME)
    lineage_path = _path(root, PREFLIGHT_DIR / PREFLIGHT_LINEAGE_NAME)
    _require(report_path.is_file() and lineage_path.is_file(), "N4 preflight authority is unavailable")
    report = _read_json(report_path, "N4 preflight report")
    lineage = _read_json(lineage_path, "N4 preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed" and report.get("classification") == "LIGHTGEN_2D_N4_MODEL_FAMILY_AUTHORIZABLE", "N4 preflight is not authorizable")
    _require(report.get("claim_boundary", {}).get("scientific_300_step_executed") is False, "N4 preflight scientific boundary changed")
    report_hash = _file_sha256(report_path)
    _require(lineage.get("report", {}).get("sha256") == report_hash, "N4 preflight lineage/report binding changed")
    _require(report.get("preflight_checks", {}).get("c1_c2_step0_equivalence") is True and report.get("preflight_checks", {}).get("checkpoint_reload_resume") is True, "N4 preflight checks incomplete")
    contract_ref = _mapping(report.get("contract"), "N4 preflight contract")
    contract, contract_ref_current = _load_contract(root, contract_ref)
    _require(contract_ref_current["sha256"] == contract_ref["sha256"], "N4 contract binding changed")
    return {"report": report, "report_ref": {"path": _relative(root, report_path), "sha256": report_hash}, "lineage_ref": {"path": _relative(root, lineage_path), "sha256": _file_sha256(lineage_path)}, "contract": contract, "contract_ref": contract_ref_current}


def _classify_candidate(candidate: str, r300: float, c0_logits: torch.Tensor, candidate0_logits: torch.Tensor, candidate300_logits: torch.Tensor, dev_labels: torch.Tensor, smooth_bootstrap_training: Mapping[str, Any], cyclic: Mapping[str, Any], evaluations: Mapping[str, Mapping[str, Any]], raw_valid: bool) -> dict[str, Any]:
    baseline_correct = c0_logits.argmax(dim=1).eq(dev_labels)
    candidate300_correct = candidate300_logits.argmax(dim=1).eq(dev_labels)
    architecture_bootstrap = V10.V6._paired_bootstrap(candidate300_correct.to(torch.float64) - baseline_correct.to(torch.float64), dev_labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
    architecture_delta = r300 - ARCHITECTURE_BASELINE
    reconstruction_delta = float(evaluations["300"]["losses"]["L_image"] - evaluations["0"]["losses"]["L_image"])
    architecture_pass = architecture_delta >= ARCHITECTURE_REQUIRED_DELTA and float(architecture_bootstrap["ci_lower"]) > 0.0
    reconstruction_pass = reconstruction_delta <= RECONSTRUCTION_MAX_DELTA
    transfer_pass = bool(cyclic["template"]["pass"] and cyclic["smooth"]["pass"] and cyclic["broad"] and not cyclic["pair_dominated"])
    criteria = {"architecture_gain": architecture_pass, "reconstruction": reconstruction_pass, "cyclic_template": bool(cyclic["template"]["pass"]), "cyclic_smooth": bool(cyclic["smooth"]["pass"]), "cyclic_broad": bool(cyclic["broad"]), "cyclic_not_pair_dominated": not bool(cyclic["pair_dominated"]), "raw_validity": bool(raw_valid)}
    adequate = bool(all(criteria.values()))
    return {"candidate": candidate, "adequate": adequate, "classification": "N4_CANDIDATE_ADEQUATE" if adequate else "N4_CANDIDATE_NOT_ADEQUATE", "criteria": criteria, "architecture": {"baseline_R": ARCHITECTURE_BASELINE, "R300": r300, "Delta_architecture": architecture_delta, "paired_bootstrap": architecture_bootstrap, "candidate_step0_vs_c0_max_logit_abs": float((candidate0_logits - c0_logits).abs().max())}, "reconstruction": {"DEV_L_image_step0": float(evaluations["0"]["losses"]["L_image"]), "DEV_L_image_step300": float(evaluations["300"]["losses"]["L_image"]), "Delta_L_image_DEV": reconstruction_delta, "threshold": RECONSTRUCTION_MAX_DELTA}, "training_smooth_bootstrap": smooth_bootstrap_training, "stage_a": {"R300": r300, "threshold": STAGE_A_FIREWALL, "status": "STAGE_A_FIREWALL_PASSED" if r300 >= STAGE_A_FIREWALL else "STAGE_A_FIREWALL_FAILED", "stage_b_authorized": False}, "cyclic_transfer_retained": transfer_pass}


def run_scientific_candidate(root: Path, bundle: Mapping[str, Any], preflight: Mapping[str, Any], candidate: str, output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to reuse N4 {candidate} output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = V10.V6.git_facts(root)
    status_before = _status_lines(root)
    payload: dict[str, Any] = {"schema": REPORT_SCHEMA.format(candidate=candidate.lower()), "version": 1, "task_id": TASK_ID, "candidate": candidate, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N4_SCIENTIFIC_HARNESS", "outer_gate": OUTER_GATE, "provenance": {"git_start": started}, "claim_boundary": {"scientific_training_executed": False, "scientific_300_step_executed": False, "stage_b_executed": False, "generation_executed": False, "protected_accessed": False, "official_test_accessed": False, "sealed_accessed": False, "formal_promotion": False}, "governance": {"retry": False, "second_arm": False, "rank_sweep": False, "lambda_tuning": False, "lr_tuning": False, "budget_extension": False, "encoder_update": False, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}}
    scientific_started = False
    updates_completed = 0
    try:
        _require(candidate in {"C1", "C2"}, "invalid N4 candidate")
        _require(started.get("branch") == EXPECTED_BRANCH and started.get("head") == EXPECTED_HEAD, "active Git identity differs from N4 authorization")
        _validate_preflight(root)
        fresh_bundle = _validate_v11_authorities(root)
        dll = V10.V6._prepare_cuda_runtime_dll_search_path()
        device, device_facts = V10.V6._cuda_facts()
        fresh_bundle["authorities"]["smooth"].to(device)
        images, labels, train_indices, development_indices, dataset = V10.V6._load_roles(root)
        fresh_bundle.update({"images": images, "labels": labels, "train_indices": train_indices, "development_indices": development_indices})
        V10.V6._initial_rng()
        consumer, adapter = _new_candidate(fresh_bundle, candidate, device, PRODUCTION_GRID)
        firewall = _configure_candidate(consumer, adapter, candidate)
        optimizer = _optimizer(consumer, adapter, candidate)
        sampler = V10.DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
        initial_frozen_digest = _frozen_runtime_digest(consumer, candidate)
        initial_generator_state = _cpu_tree(consumer.runtime.state_dict())
        dev_images = images.index_select(0, development_indices).contiguous()
        dev_labels = labels.index_select(0, development_indices).contiguous()
        selection_images, selection_labels, selection_indices = V10.V6._load_selection(root, images, labels, development_indices)
        metric_sampler = V10.DeterministicTrainRoleSampler(train_indices, labels, seed=SPLIT_SEED, steps=SCIENTIFIC_STEPS)
        metric_indices, metric_labels = metric_sampler.next_batch()
        metric_components = _objective_components(consumer, adapter, images.index_select(0, metric_indices).to(device), metric_labels.to(device), fresh_bundle["authorities"]["prototypes"].to(device), fresh_bundle["authorities"]["templates"].to(device), S_TAU)
        initial_train_metrics = {"L_anchor": float(metric_components["anchor_loss"].detach().cpu()), "L_transfer": float(metric_components["transfer_loss"].detach().cpu()), "L_sem": float(metric_components["semantic_loss"].detach().cpu()), "L_image": float(metric_components["image_loss"].detach().cpu()), "L_total": float(metric_components["total_loss"].detach().cpu())}
        contract_ref = _mapping(preflight["contract_ref"], "N4 contract reference")
        preflight_ref = {"report": preflight["report_ref"], "lineage": preflight["lineage_ref"], "result": "AUTHORIZABLE"}
        input_payload = {"schema": INPUT_SCHEMA.format(candidate=candidate.lower()), "version": 1, "task_id": TASK_ID, "candidate": candidate, "outer_gate": OUTER_GATE, "contract": dict(contract_ref), "preflight": preflight_ref, "protocol": {"objective": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE, "rank": ADAPTER_RANK, "steps": SCIENTIFIC_STEPS, "checkpoints": list(CHECKPOINT_STEPS), "batch": LOGICAL_BATCH_SIZE, "grid": PRODUCTION_GRID, "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON}, "trainable": ADAPTER_PARAMETER_NAMES + _runtime_trainable_names(candidate)}, "data_roles": {"TRAIN": 55000, "DEV": 5000, "protected": 0, "official_test": 0, "sealed": 0}, "claim_boundary": payload["claim_boundary"], "one_shot": True}
        input_path = output_dir / SCIENTIFIC_INPUT_NAME.format(candidate=candidate.lower())
        atomic_write_new_json(input_path, input_payload, artifact_kind=f"N4 {candidate} scientific input")
        scientific_started = True
        payload.update({"design_authority": DESIGN_AUTHORITY, "contract": dict(contract_ref), "preflight": preflight_ref, "authorities": {"v11_report": {"path": V11_REPORT_PATH.as_posix(), "sha256": V11_REPORT_SHA256}, "n3_aggregate": {"path": N3_AGGREGATE_PATH.as_posix(), "sha256": N3_AGGREGATE_SHA256}}, "runtime": {"cuda_dll_search_path": dll, "device": device_facts, "production_grid": PRODUCTION_GRID}, "data_roles": dataset, "initial_state": {"c0_runtime_digest": fresh_bundle["c0_runtime_digest"], "c0_adapter_digest": fresh_bundle["c0_adapter_digest"], "candidate_frozen_runtime_digest": initial_frozen_digest, "candidate_generator_state": _generator_observation(consumer, candidate, initial_generator_state), "input_sha256": _file_sha256(input_path)}, "firewall": firewall, "objective": {"formula": "L=L_anchor+28.48464149920509*L_transfer+13.353282014834436*L_image", "lambda_transfer": LAMBDA_TRANSFER, "lambda_image": LAMBDA_IMAGE, "lambda_source": "V11 frozen step0 gradient balance"}, "selection": {"rows": SELECTION_ROWS, "per_class": SELECTION_PER_CLASS, "seed": SELECTION_SEED, "indices_sha256": SELECTION_INDICES_SHA256, "labels_sha256": SELECTION_LABELS_SHA256}})
        evaluations: dict[str, dict[str, Any]] = {}
        checkpoints: dict[str, dict[str, Any]] = {}
        traces: list[dict[str, Any]] = []
        train_metrics: dict[str, Any] = {"0": initial_train_metrics}
        step0 = _evaluate(consumer, adapter, fresh_bundle["authorities"]["smooth"], dev_images, dev_labels, fresh_bundle["authorities"]["prototypes"].to(device), fresh_bundle["authorities"]["templates"].to(device), step=0, train_trace=None)
        step0["generator"] = _generator_step_observation(consumer, candidate, initial_generator_state)
        evaluations["0"] = step0
        checkpoints["0"] = _save_checkpoint(root, output_dir, fresh_bundle, candidate, consumer, adapter, optimizer, sampler, step=0, initial_frozen_digest=initial_frozen_digest, initial_generator_state=initial_generator_state)
        started_training = time.perf_counter()
        for step in range(1, SCIENTIFIC_STEPS + 1):
            indices, batch_labels = sampler.next_batch()
            batch = images.index_select(0, indices).to(device)
            trace = _train_step(consumer, adapter, optimizer, candidate, batch, batch_labels.to(device), fresh_bundle["authorities"]["prototypes"].to(device), fresh_bundle["authorities"]["templates"].to(device), step)
            traces.append(trace)
            updates_completed = step
            del batch
            if step in CHECKPOINT_STEPS[1:]:
                checkpoints[str(step)] = _save_checkpoint(root, output_dir, fresh_bundle, candidate, consumer, adapter, optimizer, sampler, step=step, initial_frozen_digest=initial_frozen_digest, initial_generator_state=initial_generator_state)
                evaluations[str(step)] = _evaluate(consumer, adapter, fresh_bundle["authorities"]["smooth"], dev_images, dev_labels, fresh_bundle["authorities"]["prototypes"].to(device), fresh_bundle["authorities"]["templates"].to(device), step=step, train_trace=trace)
                evaluations[str(step)]["generator"] = _generator_step_observation(consumer, candidate, initial_generator_state)
                train_metrics[str(step)] = {key: trace[key] for key in ("L_anchor", "L_transfer", "L_sem", "L_image", "L_total", "Delta_template_mean")}
            if step % 25 == 0:
                print(json.dumps({"event": "N4_scientific_progress", "candidate": candidate, "step": step, "elapsed_seconds": time.perf_counter() - started_training}, ensure_ascii=False), flush=True)
        _require(set(evaluations) == {"0", "100", "200", "300"} and set(checkpoints) == {"0", "100", "200", "300"}, f"N4 {candidate} checkpoint/evaluation set incomplete")
        step0_record, step300_record = evaluations["0"], evaluations["300"]
        smooth_training_bootstrap = V10.V6._paired_bootstrap(step300_record["_smooth_logits"].argmax(dim=1).eq(dev_labels).to(torch.float64) - step0_record["_smooth_logits"].argmax(dim=1).eq(dev_labels).to(torch.float64), dev_labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        ols_bootstrap = V10.V6._paired_bootstrap(step300_record["_margins"] - step0_record["_margins"], dev_labels, replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED)
        cyclic = V10._cyclic_transfer_v10(consumer, adapter, fresh_bundle["authorities"]["smooth"], selection_images, selection_labels, selection_indices, fresh_bundle["authorities"]["prototypes"], fresh_bundle["authorities"]["templates"])
        sufficient_obs = {"step0_smooth_logits": step0_record["_smooth_logits"], "step300_smooth_logits": step300_record["_smooth_logits"], "step0_ols_margin": step0_record["_margins"], "step300_ols_margin": step300_record["_margins"], "step0_anchor_distances": step0_record["_anchor_distances"], "step300_anchor_distances": step300_record["_anchor_distances"], "step0_anchor_predictions": step0_record["_anchor_predictions"], "step300_anchor_predictions": step300_record["_anchor_predictions"], "step0_anchor_correct_margin": step0_record["_anchor_correct_margin"], "step300_anchor_correct_margin": step300_record["_anchor_correct_margin"], **cyclic["_observations"]}
        sufficient_ref = _save_sufficient(root, output_dir, candidate, sufficient_obs)
        raw_ref = _save_raw(root, output_dir, candidate, cyclic)
        raw_valid = all(bool(evaluations[str(step)]["raw_validity"]["finite"] and evaluations[str(step)]["raw_validity"]["structured"] and evaluations[str(step)]["raw_validity"]["nonconstant"]) for step in CHECKPOINT_STEPS) and bool(cyclic["raw_validity"]["base"]["finite"] and cyclic["raw_validity"]["shifted"]["finite"])
        decision = _classify_candidate(candidate, float(step300_record["smooth"]["macro_recall"]), fresh_bundle["c0_smooth_logits"], step0_record["_smooth_logits"].detach().cpu(), step300_record["_smooth_logits"].detach().cpu(), dev_labels.cpu(), smooth_training_bootstrap, cyclic, evaluations, raw_valid)
        public_evaluations = {step: {key: value for key, value in record.items() if not key.startswith("_")} for step, record in evaluations.items()}
        payload.update({"result": "VALID_COMPLETED", "status": "valid_completed", "classification": decision["classification"], "evaluations": public_evaluations, "train_metrics": train_metrics, "traces": traces, "checkpoints": checkpoints, "scientific_updates": SCIENTIFIC_STEPS, "step300_minus_step0": {"Delta_R_Smooth_training": smooth_training_bootstrap, "Delta_M_OLS": ols_bootstrap, "Delta_L_image_DEV": decision["reconstruction"]["Delta_L_image_DEV"], "Delta_L_anchor_TRAIN": train_metrics["300"]["L_anchor"] - train_metrics["0"]["L_anchor"], "Delta_L_transfer_TRAIN": train_metrics["300"]["L_transfer"] - train_metrics["0"]["L_transfer"], "Delta_L_total_TRAIN": train_metrics["300"]["L_total"] - train_metrics["0"]["L_total"]}, "natural_smooth_semantics": {"R0": float(step0_record["smooth"]["macro_recall"]), "R300": float(step300_record["smooth"]["macro_recall"]), "Delta_R_training": float(step300_record["smooth"]["macro_recall"] - step0_record["smooth"]["macro_recall"]), "architecture_gain_vs_C0": decision["architecture"]}, "template_anchor": {"step0": step0_record["template_anchor"], "step300": step300_record["template_anchor"]}, "collapse_diagnostics": {"step0": step0_record["collapse"], "step300": step300_record["collapse"]}, "adapter_diagnostics": {"step0": step0_record["adapter"], "step300": step300_record["adapter"]}, "cyclic_transfer": {key: value for key, value in cyclic.items() if not key.startswith("_")}, "artifacts": {"sufficient_observations": sufficient_ref, "cyclic_raw_outputs": raw_ref}, "decision": decision, "preflight_candidate": preflight["report"].get("candidates", {}).get(candidate), "claim_boundary": {**payload["claim_boundary"], "scientific_training_executed": True, "scientific_300_step_executed": True}, "governance": {**payload["governance"], "adapter_update": True, "C1_generator_update": candidate == "C1", "C2_generator_update": candidate == "C2", "C2_extra_screen_update": candidate == "C2", "retry": False, "second_arm": False, "rank_sweep": False, "lambda_tuning": False, "lr_tuning": False, "encoder_update": False, "ols_update": False, "stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False, "training_authorized": False, "generation_readiness": "not_ready"}, "next": "no_retry; N4 comparison remains bounded and does not authorize Stage B or generation"})
    except HardBlockedN4 as error:
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": error.classification, "hard_blocker": {"stage": error.stage, "message": str(error), **error.details}, "next": "stop; no scientific retry"})
    except (InvalidN4, V10.InvalidV10, V10.V9.InvalidV9, V10.V8.InvalidV8, V10.V7.InvalidV7, V10.V6.InvalidV6) as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N4_SCIENTIFIC_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "stop; no scientific retry"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N4_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12000:]}, "next": "stop; no scientific retry"})
    finally:
        ending = V10.V6.git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "worktree_status_unchanged": status_before == _status_lines(root), "scientific_started": scientific_started})
        payload["claim_boundary"]["scientific_training_executed"] = bool(scientific_started)
        payload["claim_boundary"]["scientific_300_step_executed"] = bool(updates_completed >= SCIENTIFIC_STEPS)
        payload["scientific_updates_completed"] = updates_completed
    report_path = output_dir / SCIENTIFIC_REPORT_NAME.format(candidate=candidate.lower())
    atomic_write_new_json(report_path, payload, artifact_kind=f"N4 {candidate} scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": _file_sha256(report_path)}
    lineage_path = output_dir / SCIENTIFIC_LINEAGE_NAME.format(candidate=candidate.lower())
    atomic_write_new_json(lineage_path, {"schema": LINEAGE_SCHEMA.format(candidate=candidate.lower()), "version": 1, "task_id": TASK_ID, "candidate": candidate, "outer_gate": OUTER_GATE, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "preflight": payload.get("preflight"), "report": report_ref, "provenance": payload["provenance"], "claim_boundary": payload["claim_boundary"]}, artifact_kind=f"N4 {candidate} scientific lineage")
    _require(_read_json(report_path, f"N4 {candidate} report reload").get("result") == payload["result"], f"N4 {candidate} report reload changed")
    _require(_read_json(lineage_path, f"N4 {candidate} lineage reload").get("report", {}).get("sha256") == report_ref["sha256"], f"N4 {candidate} lineage reload changed")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "candidate": candidate, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _file_sha256(lineage_path)}, "scientific_updates": int(payload.get("scientific_updates_completed", 0)), "stage_a": payload.get("decision", {}).get("stage_a"), "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _run_folding_equivalence(root: Path, bundle: Mapping[str, Any], candidate: str, report: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_ref = _mapping(report.get("checkpoints", {}).get("300"), f"N4 {candidate} final checkpoint")
    checkpoint_path = _path(root, checkpoint_ref["path"])
    _validate_hash(checkpoint_path, str(checkpoint_ref["sha256"]), f"N4 {candidate} final checkpoint")
    checkpoint = _mapping(safe_torch_load(checkpoint_path, expected_sha256=str(checkpoint_ref["sha256"]), map_location="cpu"), f"N4 {candidate} final checkpoint reload")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    consumer, _adapter = _new_candidate(bundle, candidate, device, SMOKE_GRID, runtime_state=checkpoint["runtime_state"], adapter_state=checkpoint["adapter_state"])
    consumer.runtime.eval()
    T = _adapter.matrix().detach()
    generator = torch.Generator(device="cpu").manual_seed(20260913)
    z_cpu = torch.complex(torch.randn((4, OLS_PORT_COUNT), generator=generator), torch.randn((4, OLS_PORT_COUNT), generator=generator))
    z = z_cpu.to(device)
    explicit_coefficients = z @ T.transpose(0, 1)
    explicit_field = consumer.runtime.w_ols(ComplexOLSState.from_external(explicit_coefficients))
    modes = consumer.runtime.ols.operator.output_modes.to(device=T.device, dtype=T.dtype)
    folded_modes = torch.einsum("jk,jhw->khw", T, modes)
    folded_tensor = torch.einsum("bk,khw->bhw", z, folded_modes).unsqueeze(1)
    folded_field = type(explicit_field)(folded_tensor, consumer.runtime.generator_entrance_plane, explicit_field.provenance)
    _require(explicit_field.plane == folded_field.plane, "folded writer plane metadata changed")

    def raw_from_field(field: Any) -> tuple[torch.Tensor, Any]:
        generator_output, records, _stats = consumer.runtime.generator(field, "lightgen_folding_generator")
        output_field = type(generator_output)(generator_output.tensor, consumer.runtime.generator_exit_plane, generator_output.provenance)
        camera_field = consumer.runtime.output_relay(output_field)
        return camera_field.tensor.abs().square(), output_field

    with torch.no_grad():
        explicit_raw, explicit_output = raw_from_field(explicit_field)
        folded_raw, folded_output = raw_from_field(folded_field)
    field_diff = (explicit_field.tensor - folded_field.tensor).abs()
    output_diff = (explicit_output.tensor - folded_output.tensor).abs()
    raw_diff = (explicit_raw - folded_raw).abs()
    return {"executed": True, "candidate": candidate, "training": False, "explicit": "W_OLS(Tz)", "folded": "W_OLS_tilde(z)", "basis_formula": "v_tilde_k=sum_j T_jk v_j", "complex_field_allclose": bool(torch.allclose(explicit_field.tensor, folded_field.tensor, **TOLERANCE)), "generator_output_allclose": bool(torch.allclose(explicit_output.tensor, folded_output.tensor, **TOLERANCE)), "raw_cmos_allclose": bool(torch.allclose(explicit_raw, folded_raw, **TOLERANCE)), "field_max_abs_diff": float(field_diff.max()), "generator_output_max_abs_diff": float(output_diff.max()), "raw_cmos_max_abs_diff": float(raw_diff.max()), "plane_metadata_equal": True, "physical_realizability_claim": False}


def _aggregate(root: Path, bundle: Mapping[str, Any], preflight: Mapping[str, Any], c1_result: Mapping[str, Any], c2_result: Mapping[str, Any], c1_report: Mapping[str, Any], c2_report: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to reuse N4 aggregate directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    c1_decision = _mapping(c1_report.get("decision"), "N4 C1 decision")
    c2_decision = _mapping(c2_report.get("decision"), "N4 C2 decision")
    if bool(c1_decision["adequate"]) and bool(c2_decision["adequate"]):
        classification, selected = OUTER_GATE, "C1"
    elif not bool(c1_decision["adequate"]) and bool(c2_decision["adequate"]):
        classification, selected = OUTER_GATE, "C2"
    elif not bool(c1_decision["adequate"]) and not bool(c2_decision["adequate"]):
        classification, selected = "N4_LINEAR_INTERFACE_SYNTHESIS_CAPACITY_NOT_RESOLVED", None
    else:
        classification, selected = "N4_OLS_INTERFACE_SYNTHESIS_COMPARISON_MIXED_OR_UNRESOLVED", None
    folding_candidate = selected or "C2"
    folding = _run_folding_equivalence(root, bundle, folding_candidate, c2_report if folding_candidate == "C2" else c1_report)
    payload = {"schema": "lightgen_2d_n4_model_family_aggregate.v1", "version": 1, "task_id": TASK_ID, "design_authority": DESIGN_AUTHORITY, "outer_gate": OUTER_GATE, "result": "VALID_COMPLETED", "status": "valid_completed", "classification": classification, "selected_model": selected, "c0": {"authority": "V11 rank9 step300 endpoint", "Smooth_R300": ARCHITECTURE_BASELINE, "DEV_L_image": 0.0531686320602894, "report": {"path": V11_REPORT_PATH.as_posix(), "sha256": V11_REPORT_SHA256}, "checkpoint": {"path": V11_CHECKPOINT_PATH.as_posix(), "sha256": V11_CHECKPOINT_SHA256}}, "candidates": {"C1": {"report": c1_result["report"], "lineage": c1_result["lineage"], "adequate": bool(c1_decision["adequate"]), "decision": c1_decision}, "C2": {"report": c2_result["report"], "lineage": c2_result["lineage"], "adequate": bool(c2_decision["adequate"]), "decision": c2_decision}}, "common_baseline_verification": preflight["report"].get("step0_equivalence"), "model_selection": {"rule": "both -> C1; only C2 -> C2; neither -> unresolved", "selected": selected}, "folding_equivalence": folding, "claim_boundary": {"stage_b": False, "generation": False, "protected": False, "official_test": False, "sealed": False, "formal_promotion": False}, "governance": {"rank": 9, "rank_sweep": False, "C1_generator_update": True, "C2_generator_update": True, "C2_extra_screen_update": True, "retry": False, "stage_b": False, "generation": False, "promotion": False, "physical_realizability_claim": False}}
    report_path = output_dir / "lightgen_2d_n4_model_family_aggregate_report.v1.json"
    atomic_write_new_json(report_path, payload, artifact_kind="N4 aggregate report")
    report_ref = {"path": _relative(root, report_path), "sha256": _file_sha256(report_path)}
    lineage_path = output_dir / "lightgen_2d_n4_model_family_aggregate_lineage.v1.json"
    atomic_write_new_json(lineage_path, {"schema": "lightgen_2d_n4_model_family_aggregate_lineage.v1", "version": 1, "task_id": TASK_ID, "result": "VALID_COMPLETED", "status": "valid_completed", "classification": classification, "report": report_ref, "candidates": {"C1": c1_result, "C2": c2_result}, "claim_boundary": payload["claim_boundary"]}, artifact_kind="N4 aggregate lineage")
    _require(_read_json(report_path, "N4 aggregate report reload").get("classification") == classification, "N4 aggregate report reload changed")
    _require(_read_json(lineage_path, "N4 aggregate lineage reload").get("report", {}).get("sha256") == report_ref["sha256"], "N4 aggregate lineage reload changed")
    return {"result": "VALID_COMPLETED", "status": "valid_completed", "classification": classification, "outer_gate": OUTER_GATE, "selected_model": selected, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": _file_sha256(lineage_path)}, "folding_equivalence": folding}


def main() -> int:
    global CONTRACT_DIR, CONTRACT_PATH, PREFLIGHT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight", "scientific", "continue-c2"), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[2]).resolve()
    if args.action == "preflight":
        result = run_preflight(root, _path(root, PREFLIGHT_DIR))
        return 0 if result["result"] == "AUTHORIZABLE" else 2
    if args.action == "continue-c2":
        c1_report_path = _path(root, C1_DIR) / SCIENTIFIC_REPORT_NAME.format(candidate="c1")
        c1_report = _validate_known_c1_invalid(root)
        CONTRACT_DIR = CONTINUATION_CONTRACT_DIR
        CONTRACT_PATH = CONTRACT_DIR / "lightgen_2d_ols_interface_synthesis_model_family_contract.v1.json"
        PREFLIGHT_DIR = CONTINUATION_PREFLIGHT_DIR
        continuation_preflight = run_preflight(root, _path(root, PREFLIGHT_DIR), allow_known_c1_invalid=True)
        _require(continuation_preflight["result"] == "AUTHORIZABLE", "continuation preflight did not authorize C2")
        preflight = _validate_preflight(root)
        bundle = _validate_v11_authorities(root)
        c2_result = run_scientific_candidate(root, bundle, preflight, "C2", _path(root, C2_DIR))
        print(json.dumps({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_N4_C1_REPORT_BLOCKS_COMPARATIVE_AGGREGATE", "c1": {"report": _relative(root, c1_report_path), "result": c1_report.get("result"), "scientific_updates_completed": c1_report.get("scientific_updates_completed")}, "c2": c2_result, "next": "no retry; no aggregate; no Stage B or generation"}, indent=2, ensure_ascii=False))
        return 0 if c2_result["result"] == "VALID_COMPLETED" else 2
    preflight = _validate_preflight(root)
    bundle = _validate_v11_authorities(root)
    c1_result = run_scientific_candidate(root, bundle, preflight, "C1", _path(root, C1_DIR))
    _require(c1_result["result"] == "VALID_COMPLETED", "C1 scientific run did not complete validly; C2 execution is not permitted")
    c2_result = run_scientific_candidate(root, bundle, preflight, "C2", _path(root, C2_DIR))
    _require(c2_result["result"] == "VALID_COMPLETED", "C2 scientific run did not complete validly; aggregate is not permitted")
    c1_report = _read_json(_path(root, c1_result["report"]["path"]), "N4 C1 scientific report")
    c2_report = _read_json(_path(root, c2_result["report"]["path"]), "N4 C2 scientific report")
    aggregate = _aggregate(root, bundle, preflight, c1_result, c2_result, c1_report, c2_report, _path(root, AGGREGATE_DIR))
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
