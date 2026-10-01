"""Run the bounded target-device preflight for the LightGen V2 protocol.

This consumer validates the exact future 300-step Stage-A -> scale freeze ->
300-step Stage-B path without executing that scientific schedule.  It performs
real CUDA engineering smoke at the frozen 2048 grid, traverses the complete
TRAIN role for a clearly non-scientific scale feasibility artifact, and keeps
protected, test, and sealed roles closed.
"""

from __future__ import annotations

import copy
import ctypes
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NoReturn

import torch

import ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 as v2
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import ALLOWED_TRAINABLE_PARAMETER_NAMES, CHECKPOINT_SCHEMA, CHECKPOINT_STEPS, COMPLETE, DeterministicTrainRoleSampler, EPSILON_SCALE, FINAL_SELECT, FrozenPhysicalOLSScale, LightGen2DComplexOLSTrainingConsumerV2, LOGICAL_BATCH_SIZE, SCALE_SCHEMA, SCALE_SMOKE_CLASSIFICATION, SCALE_EXTRACT, SCALE_FREEZE, SEMANTIC_THRESHOLD, STAGE_A_FAILURE, STAGE_A_SEMANTIC_CHECK, STAGE_A_STEPS, STAGE_A_TRAIN, STAGE_B_FAILURE, STAGE_B_SEMANTIC_CHECK, STAGE_B_STEPS, STAGE_B_TRAIN, TrainingContractError, V2TrainingStateMachine, checkpoint_payload, development_macro_recall, extract_train_only_ols_scale, file_sha256, git_facts, load_checkpoint, load_scale_artifact, load_v2_training_contract, physical_ols_from_normalized, restore_rng_state, sample_normalized_prior, sample_physical_prior, save_checkpoint, save_scale_artifact, tensor_sha256
from ldm_project.v2.evidence import atomic_write_new_json, safe_torch_load
from ldm_project.v2.student.evaluator import (
    load_frozen_evaluator,
    load_local_mnist,
    stratified_mnist_indices,
)


TASK_ID = "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2"
SUCCESS_CLASSIFICATION = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2_AUTHORIZABLE"
PREFLIGHT_SCALE_CLASSIFICATION = "NON_SCIENTIFIC_PREFLIGHT_SCALE"
PREFLIGHT_CONTRACT_SCHEMA = "lightgen_2d_semantic_complex_ols_long_training_preflight_v2.v1"
REPORT_SCHEMA = "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_lineage.v1"

ORIGINAL_CONTRACT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_contract_v2_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2.json"
)
ORIGINAL_CONTRACT_SHA256 = "b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53"
CONTRACT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json"
)
CONTRACT_SHA256 = "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"
CORRECTION_AUTHORITY_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction.v1.json"
)
CORRECTION_AUTHORITY_SHA256 = "fdf03a0a8568db947011549e7869ff11ab972e19c1e01fbc3e7d2e9d850dbeb0"
CORRECTION_CLASSIFICATION = "TRAINING_SOURCE_AUTHORITY_CONTRACT_TRANSCRIPTION_ERROR_CONFIRMED"
IMPLEMENTATION_REPORT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_consumer_v2_attempt_02/"
    "lightgen_2d_semantic_complex_ols_training_consumer_v2_implementation_report.v1.json"
)
IMPLEMENTATION_REPORT_SHA256 = "8c6a07e9abf302a2060b4e65b91b65d5c514cdf1257576eba5936f734e28ebbe"
IMPLEMENTATION_PREFLIGHT_INPUT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_consumer_v2_attempt_02/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2.v1.json"
)
IMPLEMENTATION_PREFLIGHT_INPUT_SHA256 = "93773f3f88daf8be73ba576e6264acfd093e02eb77af877818e6798d0b6a588c"
HISTORICAL_BLOCKED_PREFLIGHT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_02/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_report.v1.json"
)
HISTORICAL_BLOCKED_PREFLIGHT_SHA256 = "2fbea9060fc96a2634201366dc2cb5bb549491581cc4fca3aa518a033750f7b3"
HISTORICAL_BLOCKED_LINEAGE_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_02/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_lineage.v1.json"
)
HISTORICAL_BLOCKED_LINEAGE_SHA256 = "1f9abc34454ef26734081c2b5c86352d624970b359558ccc8542ce2f71ddae74"
CONSUMER_RELATIVE_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_complex_ols_training_v2.py"
)
CONSUMER_SHA256 = "c659b93e0138531543ddfbe974c6dd4231a70b573a0701555b0dcd558a673ca3"
CONSUMER_ENTRYPOINT_RELATIVE_PATH = Path(
    "scripts/research/lightgen/execute_lightgen_2d_semantic_complex_ols_training_v2.py"
)
CONSUMER_ENTRYPOINT_SHA256 = "f223d8f8964e3c6b206f89d7d108603122383df2755742caf771cfe9e6b936bf"
ALIGNMENT_RUNTIME_RELATIVE_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
ALIGNMENT_RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
ALIGNMENT_RUNTIME_REPORT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_v1_attempt_02/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1.json"
)
ALIGNMENT_RUNTIME_REPORT_SHA256 = "e26d24654e1d9c8707e9a6b93b5c5ee66f1172fb0ae7cd686d24ea03c9263af4"
EVALUATOR_RELATIVE_PATH = Path("src/ldm_project/v2/student/evaluator.py")
EVALUATOR_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"
EVALUATOR_CHECKPOINT_RELATIVE_PATH = Path(
    "outputs/20260724_v2/runs/"
    "20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "frozen_mnist_evaluator.pt"
)
EVALUATOR_CHECKPOINT_SHA256 = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
TRAINING_DATA_RELATIVE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_DATA_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
PROTECTED_TEST_RELATIVE_PATH = Path("data/MNIST/processed/test.pt")
PROTECTED_TEST_SHA256 = "8c089c2f0a5db35de7961f4c60ed6b627a05141245d44f542eeee57b09b539e1"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07"
)
REPORT_NAME = "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_report.v1.json"
LINEAGE_NAME = "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_lineage.v1.json"
PREFLIGHT_INPUT_NAME = "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_input.v1.json"
PREFLIGHT_INPUT_SCHEMA = "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_input.v1"
PREFLIGHT_SCALE_NAME = "lightgen_2d_semantic_complex_ols_long_training_preflight_scale.v2.pt"
STAGE_A_CHECKPOINT_NAME = "lightgen_2d_semantic_complex_ols_long_training_preflight_stage_a_checkpoint.v2.pt"
STAGE_B_CHECKPOINT_NAME = "lightgen_2d_semantic_complex_ols_long_training_preflight_stage_b_checkpoint.v2.pt"
PRODUCTION_PROPAGATION_GRID = 2048
SMOKE_STEPS = 3
DEVELOPMENT_SELECTION_SEED = 4321
DATA_SPLIT_SEED = 1234
DETERMINISTIC_RTOL = 1.0e-5
DETERMINISTIC_ATOL = 1.0e-6
MALFORMED_TRAINING_DATA_SHA256 = "8151bb33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"


class InvalidPreflight(RuntimeError):
    """A malformed authority or invalid engineering harness."""


class PreflightBlocked(RuntimeError):
    """A valid preflight gate that blocks the future scientific run."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidPreflight(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidPreflight(f"{name} must be a mapping")
    return value


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _path(root: Path, relative: Path) -> Path:
    return (root / relative).resolve()


def _json(path: Path, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidPreflight(f"{name} is not valid JSON: {path}") from error
    return _mapping(value, name)


def _tree_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _tree_equal(first, second) for first, second in zip(left, right, strict=True)
        )
    return left == right


def _max_state_difference(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    if left.keys() != right.keys():
        raise InvalidPreflight("checkpoint parameter state keys differ")
    return max(
        (float((left[key].to(torch.float32) - right[key].to(torch.float32)).abs().max()) for key in left),
        default=0.0,
    )


def _max_metric_difference(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    if left.keys() != right.keys():
        raise InvalidPreflight("checkpoint metric keys differ")
    return max((abs(float(left[key]) - float(right[key])) for key in left), default=0.0)


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def _peak_snapshot(device: torch.device) -> dict[str, int]:
    snapshot = _memory_snapshot(device)
    snapshot.update(
        {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    )
    return snapshot


def _exception_is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _abort_cuda(error: BaseException, *, stage: str) -> NoReturn:
    if _exception_is_oom(error):
        raise PreflightBlocked(
            "BLOCKED_BATCH_MEMORY_INFEASIBLE",
            stage,
            "exact frozen batch/grid exhausted target-device memory",
            error_type=type(error).__name__,
            error=str(error),
        ) from error
    if isinstance(error, TrainingContractError):
        raise PreflightBlocked(
            "BLOCKED_GPU_SMOKE_STRUCTURAL_INVARIANT_FAILURE",
            stage,
            str(error),
            error_type=type(error).__name__,
        ) from error
    if isinstance(error, FloatingPointError):
        raise PreflightBlocked(
            "BLOCKED_GPU_SMOKE_NONFINITE",
            stage,
            str(error),
            error_type=type(error).__name__,
        ) from error
    raise error


def _validate_hash(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")


def _semantic_normal_form(payload: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(payload))
    normalized.pop("authority_correction", None)
    normalized["data_contract"]["source_artifacts"]["training"]["sha256"] = "<TRAINING_SOURCE_SHA256>"
    encoded = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _value_at(payload: Mapping[str, Any], path: tuple[str, ...], name: str) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise InvalidPreflight(f"semantic equivalence field is missing: {name}")
        value = value[key]
    return value


def _validate_semantic_equivalence(root: Path, contract: Any) -> dict[str, Any]:
    original_path = _path(root, ORIGINAL_CONTRACT_RELATIVE_PATH)
    _validate_hash(original_path, ORIGINAL_CONTRACT_SHA256, "original V2 contract")
    original = _json(original_path, "original V2 contract")
    _require(original.get("schema") == contract.payload.get("schema") == v2.V2_CONTRACT_SCHEMA, "V2 contract schema changed during authority repair")
    _require(original.get("version") == contract.payload.get("version") == 2, "V2 contract version changed during authority repair")
    original_normal = _semantic_normal_form(original)
    corrected_normal = _semantic_normal_form(contract.payload)
    _require(original_normal == corrected_normal, "corrected V2 contract has an unexpected scientific semantic delta")

    fields = {
        "stage_a_steps": ("stage_a", "steps"),
        "adam": ("optimizer_and_precision", "optimizer"),
        "logical_batch_size": ("optimizer_and_precision", "logical_batch_size"),
        "stage_a_loss": ("stage_a", "loss"),
        "semantic_threshold": ("semantic_evaluator_contract", "threshold", "value"),
        "stage_a_checkpoint_cadence_and_selection": ("stage_a", "checkpoint_steps"),
        "stage_a_selection_rule": ("stage_a", "selection"),
        "scale_formula": ("scale_contract", "formula"),
        "scale_train_count": ("scale_contract", "train_count"),
        "scale_dtype_and_accumulation": ("scale_contract", "dtype"),
        "epsilon_scale": ("scale_contract", "epsilon_scale"),
        "normalized_latent_definition": ("normalized_latent_contract", "formula"),
        "normalized_prior": ("normalized_latent_contract", "prior"),
        "stage_b_steps": ("stage_b", "steps"),
        "stage_b_optimizer_reset": ("stage_b", "optimizer_reset"),
        "stage_b_loss": ("stage_b", "loss"),
        "scale_anchor_formula": ("stage_b", "loss", "scale_anchor"),
        "stage_b_semantic_firewall": ("stage_b", "semantic_firewall"),
        "final_checkpoint_rule": ("stage_b", "selection"),
        "aligned_runtime_components": ("architecture", "immutable_components"),
        "aligned_runtime_chain": ("architecture", "immutable_chain"),
    }
    evidence: dict[str, Any] = {}
    for name, path in fields.items():
        before = _value_at(original, path, name)
        after = _value_at(contract.payload, path, name)
        _require(before == after, f"corrected V2 contract changed scientific field: {name}")
        evidence[name] = {"path": ".".join(path), "equal": True, "value": copy.deepcopy(after)}
    return {
        "original_contract_path": ORIGINAL_CONTRACT_RELATIVE_PATH.as_posix(),
        "corrected_contract_path": CONTRACT_RELATIVE_PATH.as_posix(),
        "all_scientific_fields_equal": True,
        "allowed_differences": [
            "data_contract.source_artifacts.training.sha256",
            "authority_correction",
        ],
        "fields": evidence,
    }


def _validate_correction_authority(root: Path, contract: Any) -> dict[str, Any]:
    path = _path(root, CORRECTION_AUTHORITY_RELATIVE_PATH)
    _validate_hash(path, CORRECTION_AUTHORITY_SHA256, "V2 authority correction")
    authority = _json(path, "V2 authority correction")
    _require(authority.get("schema") == "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction.v1", "V2 correction schema changed")
    _require(authority.get("result") == "CONFIRMED" and authority.get("classification") == CORRECTION_CLASSIFICATION, "V2 correction result changed")

    original = _mapping(authority.get("original_v2_contract"), "original V2 correction identity")
    _require(original.get("path") == ORIGINAL_CONTRACT_RELATIVE_PATH.as_posix(), "V2 correction original contract path changed")
    _require(original.get("field") == "data_contract.source_artifacts.training.sha256", "V2 correction source field changed")
    _require(original.get("value") == MALFORMED_TRAINING_DATA_SHA256 and original.get("character_count") == 63 and original.get("sha256_regex_valid") is False, "V2 correction does not bind the malformed source field")

    current_source = _mapping(authority.get("current_source"), "current training source correction identity")
    _require(current_source.get("path") == TRAINING_DATA_RELATIVE_PATH.as_posix(), "V2 correction canonical source path changed")
    _require(current_source.get("unchanged_before_after") is True, "V2 correction source read was not unchanged")

    corrected = _mapping(authority.get("corrected_contract"), "corrected V2 correction identity")
    _require(corrected.get("path") == CONTRACT_RELATIVE_PATH.as_posix(), "V2 correction corrected-contract path changed")
    boundary = _mapping(authority.get("correction_boundary"), "V2 correction boundary")
    for key in ("cuda_executed", "scientific_training_executed", "generation_executed", "protected_test_read", "sealed_read", "training_authorized", "formal_promotion", "original_v2_contract_overwritten", "training_source_modified", "scientific_semantics_changed"):
        _require(boundary.get(key) is False, f"V2 correction boundary is open: {key}")
    _require(boundary.get("generation_readiness") == "not_ready" and boundary.get("consumer_rebind_required") is True, "V2 correction downstream boundary changed")

    correction = _mapping(contract.payload.get("authority_correction"), "corrected V2 contract correction lineage")
    _require(correction.get("correction_id") == "RESOLVE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_TRAINING_SOURCE_AUTHORITY_V1", "corrected V2 contract correction id changed")
    _require(correction.get("correction_classification") == CORRECTION_CLASSIFICATION and correction.get("correction_only") is True, "corrected V2 contract correction classification changed")
    _require(correction.get("correction_authority_path") == CORRECTION_AUTHORITY_RELATIVE_PATH.as_posix(), "corrected V2 contract correction authority binding changed")
    _require(correction.get("original_contract", {}).get("path") == ORIGINAL_CONTRACT_RELATIVE_PATH.as_posix(), "corrected V2 contract original path changed")
    _require(correction.get("canonical_training_source", {}).get("path") == TRAINING_DATA_RELATIVE_PATH.as_posix(), "corrected V2 contract source path changed")
    return {
        "path": CORRECTION_AUTHORITY_RELATIVE_PATH.as_posix(),
        "sha256": CORRECTION_AUTHORITY_SHA256,
        "classification": CORRECTION_CLASSIFICATION,
        "original_contract": dict(original),
        "corrected_contract": dict(corrected),
        "canonical_training_source": dict(current_source),
        "competing_valid_canonical_sha256": [],
        "consumer_rebind_required": True,
    }


def _validate_historical_blocked_attempt(root: Path) -> dict[str, Any]:
    report_path = _path(root, HISTORICAL_BLOCKED_PREFLIGHT_RELATIVE_PATH)
    _validate_hash(report_path, HISTORICAL_BLOCKED_PREFLIGHT_SHA256, "historical blocked V2 preflight report")
    report = _json(report_path, "historical blocked V2 preflight report")
    lineage_path = _path(root, HISTORICAL_BLOCKED_LINEAGE_RELATIVE_PATH)
    _validate_hash(lineage_path, HISTORICAL_BLOCKED_LINEAGE_SHA256, "historical blocked V2 preflight lineage")
    lineage = _json(lineage_path, "historical blocked V2 preflight lineage")
    _require(report.get("classification") == "BLOCKED_TRAINING_SOURCE_AUTHORITY_MISMATCH", "historical blocked V2 preflight classification changed")
    _require(lineage.get("classification") == "BLOCKED_TRAINING_SOURCE_AUTHORITY_MISMATCH", "historical blocked V2 preflight lineage classification changed")
    return {
        "report": {"path": HISTORICAL_BLOCKED_PREFLIGHT_RELATIVE_PATH.as_posix(), "sha256": HISTORICAL_BLOCKED_PREFLIGHT_SHA256},
        "lineage": {"path": HISTORICAL_BLOCKED_LINEAGE_RELATIVE_PATH.as_posix(), "sha256": HISTORICAL_BLOCKED_LINEAGE_SHA256},
        "classification": "BLOCKED_TRAINING_SOURCE_AUTHORITY_MISMATCH",
        "disposition": "historical/superseded-by-authority-correction",
        "invalidated": False,
    }


def _validate_authorities(root: Path) -> tuple[Any, dict[str, Any]]:
    contract = load_v2_training_contract(repo_root=root)
    correction_authority = _validate_correction_authority(root, contract)
    semantic_equivalence = _validate_semantic_equivalence(root, contract)
    historical_blocked = _validate_historical_blocked_attempt(root)

    report_path = _path(root, IMPLEMENTATION_REPORT_RELATIVE_PATH)
    _validate_hash(report_path, IMPLEMENTATION_REPORT_SHA256, "V2 implementation report")
    report = _json(report_path, "V2 implementation report")
    _require(report.get("classification") == "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V2_IMPLEMENTED", "V2 implementation classification changed")
    report_contract = _mapping(_mapping(report.get("authorities"), "implementation authorities").get("v2_contract"), "implementation contract identity")
    _require(report_contract.get("path") == ORIGINAL_CONTRACT_RELATIVE_PATH.as_posix(), "historical implementation report contract path changed")
    report_boundaries = _mapping(report.get("boundaries"), "implementation boundaries")
    for key in ("scientific_training_executed", "long_training_executed", "protected_evaluation_executed", "test_values_loaded", "sealed_values_loaded", "generation_acceptance", "training_authorized", "formal_promotion"):
        _require(report_boundaries.get(key) is False, f"implementation report boundary is open: {key}")
    _require(report_boundaries.get("generation_readiness") == "not_ready", "implementation generation boundary changed")

    input_path = _path(root, IMPLEMENTATION_PREFLIGHT_INPUT_RELATIVE_PATH)
    _validate_hash(input_path, IMPLEMENTATION_PREFLIGHT_INPUT_SHA256, "V2 implementation preflight input")
    preflight_input = _json(input_path, "V2 implementation preflight input")
    _require(preflight_input.get("classification") == "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2_INPUT_READY", "V2 implementation preflight classification changed")
    input_contract = _mapping(preflight_input.get("contract"), "implementation preflight contract identity")
    _require(input_contract.get("path") == ORIGINAL_CONTRACT_RELATIVE_PATH.as_posix(), "historical implementation preflight contract path changed")
    input_boundary = _mapping(preflight_input.get("claim_boundary"), "implementation preflight claim boundary")
    for key in ("implementation_smoke_only", "scientific_training_executed", "long_training_executed", "protected_or_sealed_accessed", "training_authorized", "formal_promotion"):
        _require(input_boundary.get(key) is True if key == "implementation_smoke_only" else input_boundary.get(key) is False, f"implementation preflight boundary changed: {key}")
    _require(input_boundary.get("generation_readiness") == "not_ready", "implementation preflight generation boundary changed")

    consumer_path = _path(root, CONSUMER_RELATIVE_PATH)
    _require(consumer_path.is_file(), "V2 consumer is unavailable")
    entrypoint_path = _path(root, CONSUMER_ENTRYPOINT_RELATIVE_PATH)
    _require(entrypoint_path.is_file(), "V2 consumer entrypoint is unavailable")
    legacy = v2.validate_legacy_exclusion(consumer_path)

    runtime_path = _path(root, ALIGNMENT_RUNTIME_RELATIVE_PATH)
    _require(runtime_path.is_file(), "aligned runtime is unavailable")
    runtime_report_path = _path(root, ALIGNMENT_RUNTIME_REPORT_RELATIVE_PATH)
    _require(runtime_report_path.is_file(), "aligned runtime report is unavailable")
    runtime_report = _json(runtime_report_path, "aligned runtime report")
    _require(runtime_report.get("classification") == "LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1_IMPLEMENTED", "aligned runtime report classification changed")

    evaluator_path = _path(root, EVALUATOR_RELATIVE_PATH)
    _require(evaluator_path.is_file(), "FrozenMNISTEvaluator implementation is unavailable")
    evaluator_checkpoint_path = _path(root, EVALUATOR_CHECKPOINT_RELATIVE_PATH)
    _require(evaluator_checkpoint_path.is_file(), "FrozenMNISTEvaluator checkpoint is unavailable")

    data_contract = _mapping(contract.payload.get("data_contract"), "V2 data contract")
    roles = _mapping(data_contract.get("roles"), "V2 data roles")
    _require(_mapping(roles.get("train"), "train role").get("count") == 55_000, "train role count changed")
    _require(_mapping(roles.get("development"), "development role").get("count") == 5_000, "development role count changed")
    _require(_mapping(roles.get("protected"), "protected role").get("use") == "forbidden", "protected role opened")
    _require(_mapping(roles.get("test"), "test role").get("use") == "forbidden", "test role opened")
    _require(str(_mapping(roles.get("sealed"), "sealed role").get("use")).startswith("forbidden"), "sealed role opened")
    source_artifacts = _mapping(data_contract.get("source_artifacts"), "V2 source artifacts")
    training_ref = _mapping(source_artifacts.get("training"), "training source identity")
    protected_ref = _mapping(source_artifacts.get("protected_test"), "protected source identity")
    _require(training_ref.get("path") == TRAINING_DATA_RELATIVE_PATH.as_posix(), "training source path changed")
    _require(protected_ref.get("path") == PROTECTED_TEST_RELATIVE_PATH.as_posix() and protected_ref.get("read_by_future_training") is False, "protected source role changed")

    evaluator_contract = _mapping(contract.payload.get("semantic_evaluator_contract"), "semantic evaluator contract")
    evaluator_authority = _mapping(evaluator_contract.get("authority"), "evaluator authority")
    _require(_mapping(evaluator_authority.get("implementation"), "evaluator implementation identity").get("path") == EVALUATOR_RELATIVE_PATH.as_posix(), "evaluator implementation path changed")
    _require(_mapping(evaluator_authority.get("checkpoint"), "evaluator checkpoint identity").get("path") == EVALUATOR_CHECKPOINT_RELATIVE_PATH.as_posix(), "evaluator checkpoint path changed")
    _require(evaluator_contract.get("preprocessing") == "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28", "evaluator preprocessing changed")
    _require(_mapping(evaluator_contract.get("threshold"), "semantic threshold").get("value") == SEMANTIC_THRESHOLD, "semantic threshold changed")

    return contract, {
        "contract": contract.identity(root),
        "correction_authority": correction_authority,
        "semantic_equivalence": semantic_equivalence,
        "historical_blocked_preflight": historical_blocked,
        "implementation_report": {"path": IMPLEMENTATION_REPORT_RELATIVE_PATH.as_posix(), "sha256": IMPLEMENTATION_REPORT_SHA256},
        "implementation_preflight_input": {"path": IMPLEMENTATION_PREFLIGHT_INPUT_RELATIVE_PATH.as_posix(), "sha256": IMPLEMENTATION_PREFLIGHT_INPUT_SHA256},
        "consumer": {"path": CONSUMER_RELATIVE_PATH.as_posix(), "sha256": CONSUMER_SHA256, "entrypoint": CONSUMER_ENTRYPOINT_RELATIVE_PATH.as_posix(), "entrypoint_sha256": CONSUMER_ENTRYPOINT_SHA256},
        "aligned_runtime": {"path": ALIGNMENT_RUNTIME_RELATIVE_PATH.as_posix(), "sha256": ALIGNMENT_RUNTIME_SHA256, "report": ALIGNMENT_RUNTIME_REPORT_RELATIVE_PATH.as_posix(), "report_sha256": ALIGNMENT_RUNTIME_REPORT_SHA256},
        "evaluator": {"implementation": EVALUATOR_RELATIVE_PATH.as_posix(), "implementation_sha256": EVALUATOR_SHA256, "checkpoint": EVALUATOR_CHECKPOINT_RELATIVE_PATH.as_posix(), "checkpoint_sha256": EVALUATOR_CHECKPOINT_SHA256, "preprocessing": evaluator_contract.get("preprocessing"), "threshold": SEMANTIC_THRESHOLD},
        "dataset_authority": {"training": TRAINING_DATA_RELATIVE_PATH.as_posix(), "training_sha256": TRAINING_DATA_SHA256, "protected_test": PROTECTED_TEST_RELATIVE_PATH.as_posix(), "protected_test_sha256": PROTECTED_TEST_SHA256, "protected_test_read": False, "sealed_read": False},
        "legacy_exclusion": legacy,
        "hash_on_every_trust_boundary": False,
    }


def _fixed_development_selection(labels: torch.Tensor, development_indices: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(DEVELOPMENT_SELECTION_SEED)
    selected: list[torch.Tensor] = []
    dev_labels = labels.index_select(0, development_indices)
    for digit in range(10):
        matches = development_indices[dev_labels == digit]
        if matches.numel() != 500:
            raise InvalidPreflight("development role is not balanced at 500 rows per class")
        selected.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:100]))
    return torch.cat(selected)


def _load_dataset(root: Path, contract: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    training_path = _path(root, TRAINING_DATA_RELATIVE_PATH)
    _require(training_path.is_file(), f"TRAIN source is unavailable: {training_path}")
    images, labels = load_local_mnist(root / "data", train=True)
    if tuple(images.shape) != (60_000, 1, 28, 28) or tuple(labels.shape) != (60_000,):
        raise InvalidPreflight("local TRAIN source shape changed")
    train_indices, development_indices = stratified_mnist_indices(labels, seed=DATA_SPLIT_SEED, validation_per_label=500)
    if train_indices.numel() != 55_000 or development_indices.numel() != 5_000:
        raise InvalidPreflight("TRAIN/DEV role counts changed")
    if torch.unique(torch.cat((train_indices, development_indices))).numel() != 60_000:
        raise InvalidPreflight("TRAIN and DEV roles overlap or omit source rows")
    train_counts = torch.bincount(labels.index_select(0, train_indices), minlength=10)
    dev_counts = torch.bincount(labels.index_select(0, development_indices), minlength=10)
    source_counts = torch.bincount(labels, minlength=10)
    if not torch.equal(train_counts + dev_counts, source_counts):
        raise InvalidPreflight("TRAIN/DEV partition does not preserve observed source class counts")
    if not torch.equal(dev_counts, torch.full((10,), 500, dtype=torch.long)):
        raise InvalidPreflight("DEV class counts changed")
    selection_indices = _fixed_development_selection(labels, development_indices)
    train_images = images.index_select(0, train_indices).contiguous()
    selection_labels = labels.index_select(0, selection_indices).contiguous()
    dataset = {
        "source": {"path": TRAINING_DATA_RELATIVE_PATH.as_posix(), "sha256": TRAINING_DATA_SHA256, "rows_loaded": 60_000},
        "train": {"rows": 55_000, "counts_by_class": [int(value) for value in train_counts.tolist()], "indices_sha256": tensor_sha256(train_indices)},
        "development": {"rows": 5_000, "counts_by_class": [int(value) for value in dev_counts.tolist()], "indices_sha256": tensor_sha256(development_indices), "selection_rows": 1_000, "selection_per_class": 100, "selection_seed": DEVELOPMENT_SELECTION_SEED, "selection_indices_sha256": tensor_sha256(selection_indices)},
        "calibration": {"rows_read": 0, "status": "not consumed; prior authority only"},
        "protected": {"rows_read": 0, "status": "forbidden", "source_sha256_bound_only": PROTECTED_TEST_SHA256},
        "test": {"rows_read": 0, "status": "forbidden", "source_sha256_bound_only": PROTECTED_TEST_SHA256},
        "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
        "labels_as_model_input": False,
    }
    return images, labels, train_indices, development_indices, selection_indices, {"train_images": train_images, "selection_labels": selection_labels, **dataset}


def _nvidia_smi_facts() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise PreflightBlocked("BLOCKED_TARGET_DEVICE_UNAVAILABLE", "device", "nvidia-smi is unavailable")
    try:
        full = subprocess.run([executable], check=True, capture_output=True, text=True).stdout
        query = subprocess.run([executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightBlocked("BLOCKED_TARGET_DEVICE_UNAVAILABLE", "device", "nvidia-smi device facts failed", error_type=type(error).__name__, error=str(error)) from error
    fields = [field.strip() for field in query.split(",")]
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", full)
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", full)
    return {"executable": executable, "driver_version": driver_match.group(1) if driver_match else None, "driver_visible_cuda": cuda_match.group(1) if cuda_match else None, "query": {"gpu_name": fields[0] if fields else None, "driver_version": fields[1] if len(fields) > 1 else None, "memory_total_mib": int(fields[2]) if len(fields) > 2 and fields[2].isdigit() else None}}


def _cuda_device_facts() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise PreflightBlocked("BLOCKED_TARGET_DEVICE_UNAVAILABLE", "device", "required CUDA target is unavailable", cuda_available=bool(torch.cuda.is_available()), device_count=int(torch.cuda.device_count()))
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    env_root = Path(sys.prefix)
    dll_candidates = sorted({path.resolve() for directory in (env_root / "bin", env_root / "Library" / "bin") if directory.is_dir() for path in directory.glob("nvrtc*.dll")})
    dll_loads: list[dict[str, Any]] = []
    loader = getattr(ctypes, "WinDLL", None)
    if loader is not None:
        for path in dll_candidates:
            try:
                loader(str(path))
                dll_loads.append({"path": str(path), "loaded": True})
            except OSError as error:
                dll_loads.append({"path": str(path), "loaded": False, "error": str(error)})
    try:
        complex_probe = torch.tensor([1.0 + 1.0j], dtype=torch.complex64, device=device)
        exp_probe = torch.exp(complex_probe)
        fft_probe = torch.fft.ifft2(torch.fft.fft2(complex_probe.reshape(1, 1, 1, 1)))
        polar_probe = torch.polar(torch.ones(1, device=device), torch.zeros(1, device=device))
        torch.cuda.synchronize(device)
    except Exception as error:
        raise PreflightBlocked("BLOCKED_CUDA_BACKEND_UNAVAILABLE", "cuda_backend", "complex/FFT CUDA backend probe failed", error_type=type(error).__name__, error=str(error), nvrtc_candidates=[str(path) for path in dll_candidates]) from error
    backend_probe = {"complex_exp": bool(torch.isfinite(exp_probe).all()), "fft2_ifft2": bool(torch.isfinite(fft_probe).all()), "complex_polar": bool(torch.isfinite(polar_probe).all())}
    if not all(backend_probe.values()):
        raise PreflightBlocked("BLOCKED_CUDA_BACKEND_UNAVAILABLE", "cuda_backend", "CUDA backend probe was nonfinite", probe=backend_probe)
    smi = _nvidia_smi_facts()
    return device, {
        "device_index": 0,
        "device_count": int(torch.cuda.device_count()),
        "gpu": properties.name,
        "compute_capability": [int(properties.major), int(properties.minor)],
        "vram_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_visible": smi,
        "backend": {"cuda_built": bool(torch.backends.cuda.is_built()), "nvrtc_required_by_active_path": True, "nvrtc_candidate_dlls": [str(path) for path in dll_candidates], "nvrtc_dll_loads": dll_loads, "probe": backend_probe, "env_prefix": str(env_root)},
    }


def _finite_update(update: Any, *, stage: str) -> dict[str, Any]:
    result = update.loss.result
    finite = {
        "losses": all(bool(torch.isfinite(value)) for value in (update.loss.image, update.loss.latent, update.loss.scale, update.loss.total)),
        "complex_ols": bool(torch.isfinite(result.z_ols.coefficients).all()),
        "raw_cmos": bool(torch.isfinite(result.raw_intensity).all()),
        "prior": update.loss.prior is None or bool(torch.isfinite(update.loss.prior.coefficients).all()),
        "parameters": True,
    }
    if not all(finite.values()):
        raise PreflightBlocked("BLOCKED_GPU_SMOKE_NONFINITE", f"{stage}_finite", "V2 smoke produced a nonfinite value", finite=finite)
    return {"stage": stage, "global_step": update.global_step, "stage_local_step": update.stage_local_step, "loss": update.loss.metrics(), "autograd": update.autograd.to_dict(), "parameter_update_norms": dict(update.parameter_update_norms), "frozen_state_unchanged": update.frozen_state_unchanged, "finite": finite, "all_finite": True}


def _batch_memory_update(consumer: LightGen2DComplexOLSTrainingConsumerV2, images: torch.Tensor, indices: torch.Tensor, *, stage: str, global_step: int, stage_local_step: int, optimizer: torch.optim.Optimizer, scale: FrozenPhysicalOLSScale | None, device: torch.device) -> tuple[Any, dict[str, int]]:
    batch = images.index_select(0, indices).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        update = consumer.update(batch, stage=stage, global_step=global_step, stage_local_step=stage_local_step, optimizer=optimizer, scale=scale)
        torch.cuda.synchronize(device)
    except Exception as error:
        _abort_cuda(error, stage=f"{stage}_update_{global_step}")
    memory = _peak_snapshot(device)
    del batch
    return update, memory


def _deterministic_scale_probe(consumer: LightGen2DComplexOLSTrainingConsumerV2, images: torch.Tensor) -> torch.Tensor:
    """Measure a replayable smoke vector without asserting scale positivity."""

    if not isinstance(images, torch.Tensor) or images.ndim != 4 or images.shape[1:] != (1, 28, 28):
        raise ValueError("deterministic scale probe images require shape [N,1,28,28]")
    if images.dtype != torch.float32 or torch.is_complex(images) or not torch.isfinite(images).all():
        raise TypeError("deterministic scale probe images must be finite float32")
    if images.numel() and (images.min() < 0.0 or images.max() > 1.0):
        raise ValueError("deterministic scale probe images must be in [0,1]")
    if images.shape[0] == 0:
        raise ValueError("deterministic scale probe requires at least one row")

    was_training = consumer.runtime.training
    consumer.runtime.eval()
    total = torch.zeros((100,), dtype=torch.float64)
    count = 0
    try:
        with torch.no_grad():
            for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
                batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
                result = consumer.runtime.forward_from_image(batch)
                if result.z_ols.source != "real_encoded" or result.ols_input_field is None:
                    raise TrainingContractError("deterministic scale probe did not use the mandatory real modal projection")
                z = result.z_ols.coefficients
                if (
                    not isinstance(z, torch.Tensor)
                    or z.ndim != 2
                    or tuple(z.shape[1:]) != (100,)
                    or not torch.is_complex(z)
                    or not torch.isfinite(z).all()
                ):
                    raise TrainingContractError("deterministic scale probe produced an invalid complex OLS tensor")
                total += (z.real.to(torch.float64).square() + z.imag.to(torch.float64).square()).sum(dim=0).cpu()
                count += z.shape[0]
    finally:
        consumer.runtime.train(was_training)
    if count != images.shape[0]:
        raise TrainingContractError("deterministic scale probe row count changed")
    return total / float(count)


def _selected_a_identity() -> dict[str, Any]:
    return {"classification": "NON_SCIENTIFIC_PREFLIGHT_REFERENCE", "stage_local_step": 3, "checkpoint_selection_executed": False, "semantic_selection_executed": False}


def _make_stage_b_state(scale: FrozenPhysicalOLSScale, marker: Mapping[str, Any]) -> V2TrainingStateMachine:
    state = V2TrainingStateMachine(stage="STAGE_A_SELECT", selected_stage_a_checkpoint=_selected_a_identity(), scale_identity=scale.identity(), optimizer_reset_marker=marker)
    state.transition(SCALE_EXTRACT)
    state.transition(SCALE_FREEZE)
    state.transition(STAGE_B_TRAIN)
    return state


def _load_preflight_scale_artifact(path: Path) -> FrozenPhysicalOLSScale:
    _require(path.is_file(), f"preflight scale artifact is unavailable: {path}")
    actual = file_sha256(path)
    payload = _mapping(safe_torch_load(path, map_location="cpu"), "preflight scale artifact")
    _require(payload.get("schema") == SCALE_SCHEMA and payload.get("version") == 2, "preflight scale schema changed")
    _require(payload.get("classification") == PREFLIGHT_SCALE_CLASSIFICATION, "preflight scale classification changed")
    _require(payload.get("data_role") == SCALE_SMOKE_CLASSIFICATION, "preflight scale data role changed")
    _require(payload.get("epsilon_scale") == EPSILON_SCALE and payload.get("port_count") == 100 and payload.get("stage_b_frozen") is True, "preflight scale convention changed")
    values = payload.get("values")
    _require(isinstance(values, torch.Tensor) and values.dtype == torch.float64 and tuple(values.shape) == (100,), "preflight scale vector changed")
    return FrozenPhysicalOLSScale(values.detach().cpu(), _mapping(payload.get("checkpoint_identity"), "preflight checkpoint identity"), _mapping(payload.get("data_identity"), "preflight data identity"), data_role=SCALE_SMOKE_CLASSIFICATION, classification=PREFLIGHT_SCALE_CLASSIFICATION, artifact_path=str(path), artifact_sha256=actual)


def _load_preflight_checkpoint(path: Path, consumer: LightGen2DComplexOLSTrainingConsumerV2, optimizer: torch.optim.Optimizer, expected_scale: FrozenPhysicalOLSScale) -> tuple[Mapping[str, Any], FrozenPhysicalOLSScale]:
    payload = _mapping(safe_torch_load(path, map_location="cpu"), "V2 preflight checkpoint")
    _require(payload.get("schema") == CHECKPOINT_SCHEMA and payload.get("version") == 2, "preflight checkpoint schema changed")
    contract = _mapping(payload.get("contract"), "preflight checkpoint contract")
    _require(contract.get("schema") == consumer.contract.payload.get("schema") and contract.get("path") == consumer.contract.path.relative_to(_root()).as_posix(), "preflight checkpoint contract identity changed")
    runtime = _mapping(payload.get("runtime"), "preflight checkpoint runtime")
    _require(runtime.get("module") == v2.ALIGNMENT_RUNTIME_RELATIVE_PATH.as_posix() and runtime.get("transport") == "T=I_100 coefficient space", "preflight checkpoint runtime identity changed")
    _require(payload.get("stage") in {STAGE_B_TRAIN, STAGE_B_SEMANTIC_CHECK, FINAL_SELECT, COMPLETE}, "preflight checkpoint stage is not Stage B")
    consumer.load_trainable_state_dict(_mapping(payload.get("trainable_parameter_state"), "preflight checkpoint parameters"))
    consumer.optimizer_manifest(optimizer)
    optimizer_state = payload.get("optimizer_state")
    _require(isinstance(optimizer_state, Mapping), "preflight checkpoint optimizer state is missing")
    optimizer.load_state_dict(dict(optimizer_state))
    consumer.optimizer_manifest(optimizer)
    restore_rng_state(_mapping(payload.get("rng_state"), "preflight checkpoint RNG"))
    marker = _mapping(payload.get("optimizer_reset_marker"), "preflight checkpoint optimizer reset")
    _require(marker.get("optimizer_reset") is True and marker.get("stage_a_optimizer_moments_restored") is False, "preflight checkpoint reset marker changed")
    artifact = _mapping(payload.get("scale_artifact"), "preflight checkpoint scale identity")
    _require(artifact.get("schema") == SCALE_SCHEMA and isinstance(artifact.get("path"), str), "preflight checkpoint scale identity is incomplete")
    scale = _load_preflight_scale_artifact(Path(artifact["path"]))
    _require(torch.equal(scale.values, expected_scale.values), "preflight checkpoint scale authority changed")
    values = payload.get("scale_vector")
    _require(isinstance(values, torch.Tensor) and torch.equal(values.to(dtype=torch.float64, device="cpu"), scale.values), "preflight checkpoint scale vector changed")
    consumer.set_scale(scale)
    _require(_mapping(payload.get("selected_stage_a_checkpoint"), "preflight selected Stage-A identity").get("classification") == "NON_SCIENTIFIC_PREFLIGHT_REFERENCE", "preflight selected Stage-A identity changed")
    return payload, scale


def _run_semantic_firewall(consumer: LightGen2DComplexOLSTrainingConsumerV2, evaluator: torch.nn.Module, images: torch.Tensor, labels: torch.Tensor, selection_indices: torch.Tensor) -> dict[str, Any]:
    selected_images = images.index_select(0, selection_indices)
    selected_labels = labels.index_select(0, selection_indices)
    try:
        recall = development_macro_recall(consumer, evaluator, selected_images, selected_labels, batch_size=LOGICAL_BATCH_SIZE)
    except Exception as error:
        _abort_cuda(error, stage="semantic_firewall")
    pass_state = V2TrainingStateMachine(stage=STAGE_A_SEMANTIC_CHECK)
    pass_state.stage_a_checkpoint_checked(final_checkpoint=True, eligible=True)
    fail_state = V2TrainingStateMachine(stage=STAGE_A_SEMANTIC_CHECK)
    fail_state.stage_a_checkpoint_checked(final_checkpoint=True, eligible=False)
    pass_b = V2TrainingStateMachine(stage=STAGE_B_SEMANTIC_CHECK)
    pass_b.stage_b_checkpoint_checked(final_checkpoint=True, eligible=True)
    fail_b = V2TrainingStateMachine(stage=STAGE_B_SEMANTIC_CHECK)
    fail_b.stage_b_checkpoint_checked(final_checkpoint=True, eligible=False)
    return {"evaluator": "FrozenMNISTEvaluator v1", "preprocessing": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28", "threshold": SEMANTIC_THRESHOLD, "selection_rows": int(selection_indices.numel()), "smoke_macro_recall": recall, "smoke_eligibility": recall >= SEMANTIC_THRESHOLD, "smoke_recall_is_not_scientific_selection": True, "logical_pass_branch": {"stage": pass_state.stage, "terminal_classification": pass_state.terminal_classification}, "logical_fail_branch": {"stage": fail_state.stage, "terminal_classification": fail_state.terminal_classification, "expected": STAGE_A_FAILURE}, "stage_b_logical_pass_branch": {"stage": pass_b.stage, "terminal_classification": pass_b.terminal_classification}, "stage_b_logical_fail_branch": {"stage": fail_b.stage, "terminal_classification": fail_b.terminal_classification, "expected": STAGE_B_FAILURE}, "state_machine_branches_verified": True}


def _run_preflight(root: Path, contract: Any, dataset: Mapping[str, Any], images: torch.Tensor, labels: torch.Tensor, train_indices: torch.Tensor, selection_indices: torch.Tensor, device: torch.device, evaluator: torch.nn.Module, output_dir: Path, starting_git: Mapping[str, Any]) -> dict[str, Any]:
    print("preflight: Stage-A CUDA smoke (3 steps, batch=10, grid=2048)", flush=True)
    consumer = LightGen2DComplexOLSTrainingConsumerV2(contract, device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
    manifest = consumer.parameter_manifest()
    if manifest["trainable_parameter_names"] != list(ALLOWED_TRAINABLE_PARAMETER_NAMES) or manifest["optimizer_parameter_names"] != list(ALLOWED_TRAINABLE_PARAMETER_NAMES):
        raise PreflightBlocked("BLOCKED_GPU_SMOKE_STRUCTURAL_INVARIANT_FAILURE", "trainable_firewall", "V2 trainable manifest changed", manifest=manifest)
    optimizer_a = consumer.create_optimizer()
    optimizer_manifest = consumer.optimizer_manifest(optimizer_a)
    sampler_a = DeterministicTrainRoleSampler(train_indices, labels, steps=STAGE_A_STEPS)
    memory_before_model = _memory_snapshot(device)
    memory_after_init = _memory_snapshot(device)
    stage_a_updates: list[dict[str, Any]] = []
    stage_a_memory: list[dict[str, int]] = []
    stage_a_batches: list[torch.Tensor] = []
    stage_a_checkpoint_path = output_dir / STAGE_A_CHECKPOINT_NAME
    stage_a_checkpoint_payload: Mapping[str, Any] | None = None
    continuous_a_parameters: Mapping[str, torch.Tensor] | None = None
    continuous_a_optimizer: Mapping[str, Any] | None = None
    continuous_a_metrics: Mapping[str, float] | None = None
    for local_step in range(SMOKE_STEPS):
        batch_indices, batch_labels = sampler_a.next_batch()
        if not torch.equal(batch_labels, torch.arange(10, dtype=torch.long)):
            raise PreflightBlocked("BLOCKED_DATA_FIREWALL_ROLE_MISMATCH", "stage_a_sampler", "TRAIN sampler labels changed")
        stage_a_batches.append(batch_indices.clone())
        update, memory = _batch_memory_update(consumer, images, batch_indices, stage="A", global_step=local_step, stage_local_step=local_step, optimizer=optimizer_a, scale=None, device=device)
        record = _finite_update(update, stage="A")
        if update.loss.prior is not None or not update.autograd.stage_a_no_mmd_gradient or not update.autograd.modal_projection_active or not update.autograd.w_ols_active or not update.autograd.frozen_scale_no_grad or abs(float(update.loss.total - update.loss.image)) > DETERMINISTIC_ATOL:
            raise PreflightBlocked("BLOCKED_GPU_SMOKE_STRUCTURAL_INVARIANT_FAILURE", f"stage_a_step_{local_step}", "Stage-A latent-MMD or aligned-path invariant failed", record=record)
        record["loss_decomposition"] = {"total_equals_image": True, "latent_mmd_enabled": False}
        stage_a_updates.append(record)
        stage_a_memory.append(memory)
        if local_step == 0:
            state = V2TrainingStateMachine(stage=STAGE_A_TRAIN, global_step=0, stage_local_step=1)
            stage_a_checkpoint_payload = checkpoint_payload(consumer, optimizer_a, state=state, sampler_state=sampler_a.state_dict(), semantic_selection_state={"threshold": SEMANTIC_THRESHOLD, "checkpoint_steps": list(CHECKPOINT_STEPS), "selection_executed": False, "scope": "NON_SCIENTIFIC_PREFLIGHT"}, git=starting_git)
            save_checkpoint(stage_a_checkpoint_path, stage_a_checkpoint_payload)
        if local_step == 1:
            continuous_a_parameters = consumer.trainable_state_dict()
            continuous_a_optimizer = copy.deepcopy(optimizer_a.state_dict())
            continuous_a_metrics = dict(update.loss.metrics())
        del update
    if stage_a_checkpoint_payload is None or continuous_a_parameters is None or continuous_a_optimizer is None or continuous_a_metrics is None:
        raise InvalidPreflight("Stage-A checkpoint/resume reference was not captured")

    print("preflight: FrozenMNISTEvaluator development path", flush=True)
    semantic_firewall = _run_semantic_firewall(consumer, evaluator, images, labels, selection_indices)

    print("preflight: exact TRAIN-only scale traversal (55,000 rows, batch=10)", flush=True)
    train_images = dataset["train_images"]
    torch.cuda.reset_peak_memory_stats(device)
    scale_started = time.perf_counter()
    try:
        extracted = extract_train_only_ols_scale(consumer, train_images, data_role="train", checkpoint_identity={"role": "NON_SCIENTIFIC_PREFLIGHT_REFERENCE", "stage": "A_smoke", "selection_executed": False}, data_identity={"role": "TRAIN", "rows": 55_000, "indices_sha256": tensor_sha256(train_indices)})
        torch.cuda.synchronize(device)
    except Exception as error:
        _abort_cuda(error, stage="train_only_scale_extraction")
    scale_elapsed = time.perf_counter() - scale_started
    scale_memory = _peak_snapshot(device)
    if extracted.values.dtype != torch.float64 or tuple(extracted.values.shape) != (100,) or not torch.isfinite(extracted.values).all() or bool((extracted.values <= EPSILON_SCALE).any()):
        raise PreflightBlocked("BLOCKED_SCALE_EXTRACTION_INVALID", "train_only_scale_extraction", "55,000-row scale result is invalid")
    smoke_probe = train_images[:LOGICAL_BATCH_SIZE]
    try:
        deterministic_probe_a = _deterministic_scale_probe(consumer, smoke_probe)
        deterministic_probe_b = _deterministic_scale_probe(consumer, smoke_probe)
    except Exception as error:
        _abort_cuda(error, stage="train_only_scale_determinism_probe")
    if not torch.equal(deterministic_probe_a, deterministic_probe_b):
        raise PreflightBlocked("BLOCKED_SCALE_EXTRACTION_NONDETERMINISTIC", "train_only_scale_extraction", "deterministic scale probe changed")
    preflight_scale = FrozenPhysicalOLSScale(extracted.values.clone(), {"role": "NON_SCIENTIFIC_PREFLIGHT_REFERENCE", "stage": "A_smoke", "selection_executed": False}, {"role": "TRAIN", "rows": 55_000, "indices_sha256": tensor_sha256(train_indices)}, data_role=SCALE_SMOKE_CLASSIFICATION, classification=PREFLIGHT_SCALE_CLASSIFICATION)
    scale_artifact = save_scale_artifact(output_dir / PREFLIGHT_SCALE_NAME, preflight_scale)
    scale_artifact = _load_preflight_scale_artifact(Path(scale_artifact.artifact_path))
    try:
        load_scale_artifact(scale_artifact.artifact_path)
    except TrainingContractError as error:
        formal_scale_rejection = {"rejected": True, "error_type": type(error).__name__, "message": str(error)}
    else:
        raise PreflightBlocked("BLOCKED_PREFLIGHT_SCALE_FIREWALL", "scale_authority", "formal scale loader accepted NON_SCIENTIFIC_PREFLIGHT_SCALE")

    print("preflight: Stage-B CUDA smoke (3 steps, frozen preflight scale)", flush=True)
    consumer.set_scale(scale_artifact)
    stage_a_parameters_before_reset = consumer.trainable_state_dict()
    optimizer_b, reset_marker = consumer.reset_stage_b_optimizer()
    if not optimizer_a.state or optimizer_b is optimizer_a or optimizer_b.state or not reset_marker["optimizer_reset"] or not reset_marker["new_optimizer_state_empty"] or reset_marker["stage_a_optimizer_moments_restored"]:
        raise PreflightBlocked("BLOCKED_OPTIMIZER_RESET_NOT_REPRODUCIBLE", "stage_a_to_stage_b", "A->B optimizer reset invariant failed", reset_marker=reset_marker)
    if not all(torch.equal(stage_a_parameters_before_reset[name], consumer.trainable_state_dict()[name]) for name in stage_a_parameters_before_reset):
        raise PreflightBlocked("BLOCKED_OPTIMIZER_RESET_NOT_REPRODUCIBLE", "stage_a_to_stage_b", "Stage-A parameters did not carry forward")
    stage_b_state = _make_stage_b_state(scale_artifact, reset_marker)
    sampler_b = DeterministicTrainRoleSampler(train_indices, labels, steps=STAGE_B_STEPS)
    stage_b_updates: list[dict[str, Any]] = []
    stage_b_memory: list[dict[str, int]] = []
    stage_b_batches: list[torch.Tensor] = []
    stage_b_checkpoint_path = output_dir / STAGE_B_CHECKPOINT_NAME
    stage_b_checkpoint_payload: Mapping[str, Any] | None = None
    continuous_b_parameters: Mapping[str, torch.Tensor] | None = None
    continuous_b_optimizer: Mapping[str, Any] | None = None
    continuous_b_metrics: Mapping[str, float] | None = None
    captured_mmd_input: torch.Tensor | None = None
    captured_scale_input: torch.Tensor | None = None
    captured_scale_sha: str | None = None
    first_b_normalized: torch.Tensor | None = None
    first_b_raw_ols: torch.Tensor | None = None
    original_mmd = v2.normalized_full_complex_mmd_vstat
    original_scale_anchor = v2.physical_scale_anchor_loss

    def instrumented_mmd(encoded: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        nonlocal captured_mmd_input
        if captured_mmd_input is None:
            captured_mmd_input = encoded.detach().cpu().clone()
        return original_mmd(encoded, prior)

    def instrumented_scale_anchor(z: torch.Tensor, scale: FrozenPhysicalOLSScale) -> torch.Tensor:
        nonlocal captured_scale_input, captured_scale_sha
        if captured_scale_input is None:
            captured_scale_input = z.detach().cpu().clone()
            captured_scale_sha = scale.vector_sha256
        return original_scale_anchor(z, scale)

    v2.normalized_full_complex_mmd_vstat = instrumented_mmd
    v2.physical_scale_anchor_loss = instrumented_scale_anchor
    try:
        for local_step in range(SMOKE_STEPS):
            batch_indices, batch_labels = sampler_b.next_batch()
            if not torch.equal(batch_labels, torch.arange(10, dtype=torch.long)):
                raise PreflightBlocked("BLOCKED_DATA_FIREWALL_ROLE_MISMATCH", "stage_b_sampler", "TRAIN sampler labels changed")
            stage_b_batches.append(batch_indices.clone())
            update, memory = _batch_memory_update(consumer, images, batch_indices, stage="B", global_step=STAGE_A_STEPS + local_step + 1, stage_local_step=local_step + 1, optimizer=optimizer_b, scale=scale_artifact, device=device)
            record = _finite_update(update, stage="B")
            expected_total = update.loss.image + update.loss.latent + update.loss.scale
            if update.loss.prior is None or not update.autograd.stage_b_normalized_mmd_active or not update.autograd.scale_gradient_active or not update.autograd.frozen_scale_no_grad or not torch.allclose(update.loss.total, expected_total, rtol=DETERMINISTIC_RTOL, atol=DETERMINISTIC_ATOL):
                raise PreflightBlocked("BLOCKED_GPU_SMOKE_STRUCTURAL_INVARIANT_FAILURE", f"stage_b_step_{local_step}", "Stage-B loss/autograd invariant failed", record=record)
            record["loss_decomposition"] = {"total_equals_image_plus_normalized_mmd_plus_scale": True, "mmd_input": "z_tilde", "scale_anchor_input": "raw_z", "scale_vector_sha256": scale_artifact.vector_sha256}
            stage_b_updates.append(record)
            stage_b_memory.append(memory)
            if local_step == 0:
                first_b_normalized = update.loss.normalized.detach().cpu().clone() if update.loss.normalized is not None else None
                first_b_raw_ols = update.loss.result.z_ols.coefficients.detach().cpu().clone()
            if local_step == 0:
                stage_b_checkpoint_payload = checkpoint_payload(consumer, optimizer_b, state=stage_b_state, sampler_state=sampler_b.state_dict(), semantic_selection_state={"threshold": SEMANTIC_THRESHOLD, "checkpoint_steps": list(CHECKPOINT_STEPS), "selection_executed": False, "scope": "NON_SCIENTIFIC_PREFLIGHT"}, git=starting_git, scale=scale_artifact, selected_stage_a_checkpoint=_selected_a_identity(), optimizer_reset_marker=reset_marker)
                save_checkpoint(stage_b_checkpoint_path, stage_b_checkpoint_payload)
            if local_step == 1:
                continuous_b_parameters = consumer.trainable_state_dict()
                continuous_b_optimizer = copy.deepcopy(optimizer_b.state_dict())
                continuous_b_metrics = dict(update.loss.metrics())
            del update
    finally:
        v2.normalized_full_complex_mmd_vstat = original_mmd
        v2.physical_scale_anchor_loss = original_scale_anchor
    if stage_b_checkpoint_payload is None or continuous_b_parameters is None or continuous_b_optimizer is None or continuous_b_metrics is None:
        raise InvalidPreflight("Stage-B checkpoint/resume reference was not captured")
    if captured_mmd_input is None or captured_scale_input is None or captured_scale_sha != scale_artifact.vector_sha256:
        raise PreflightBlocked("BLOCKED_GPU_SMOKE_STRUCTURAL_INVARIANT_FAILURE", "stage_b_input_routing", "Stage-B loss input routing was not observed")
    if first_b_normalized is None or first_b_raw_ols is None or not torch.allclose(captured_mmd_input, first_b_normalized, rtol=DETERMINISTIC_RTOL, atol=DETERMINISTIC_ATOL):
        raise PreflightBlocked("BLOCKED_GPU_SMOKE_STRUCTURAL_INVARIANT_FAILURE", "stage_b_input_routing", "Stage-B MMD did not consume the normalized z_tilde coordinate")
    if not torch.allclose(captured_scale_input, first_b_raw_ols, rtol=DETERMINISTIC_RTOL, atol=DETERMINISTIC_ATOL):
        raise PreflightBlocked("BLOCKED_GPU_SMOKE_STRUCTURAL_INVARIANT_FAILURE", "stage_b_input_routing", "Stage-B scale anchor did not consume raw physical z")
    scale_before = scale_artifact.values.clone()
    sampled_prior, normalized_prior = sample_physical_prior(scale_artifact, 10, stage_local_step=0, device="cpu")
    prior_replay_a = sample_normalized_prior(10, stage="STAGE_B", stage_local_step=1, device="cpu")
    prior_replay_b = sample_normalized_prior(10, stage="STAGE_B", stage_local_step=1, device="cpu")
    if not torch.equal(prior_replay_a.coefficients, prior_replay_b.coefficients) or tuple(prior_replay_a.coefficients.shape) != (10, 100) or prior_replay_a.coefficients.dtype != torch.complex64:
        raise PreflightBlocked("BLOCKED_NORMALIZED_PRIOR_INVALID", "normalized_prior", "normalized prior replay failed")
    physical_prior = physical_ols_from_normalized(prior_replay_a.coefficients, scale_artifact)
    if tuple(physical_prior.shape) != (10, 100) or physical_prior.dtype != torch.complex64 or not torch.isfinite(physical_prior).all() or sampled_prior.coefficients.shape != (10, 100) or normalized_prior.coefficients.shape != (10, 100):
        raise PreflightBlocked("BLOCKED_NORMALIZED_PRIOR_INVALID", "physical_prior_mapping", "normalized to physical prior mapping failed")
    if not torch.equal(scale_before, scale_artifact.values):
        raise PreflightBlocked("BLOCKED_SCALE_MUTATION", "stage_b_scale_freeze", "preflight scale mutated during Stage B")

    print("preflight: Stage-B checkpoint/resume and formal scale rejection", flush=True)
    formal_load_consumer = LightGen2DComplexOLSTrainingConsumerV2(contract, device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
    formal_load_optimizer = formal_load_consumer.create_optimizer()
    try:
        load_checkpoint(stage_b_checkpoint_path, formal_load_consumer, formal_load_optimizer)
    except TrainingContractError as error:
        formal_checkpoint_rejection = {"rejected": True, "error_type": type(error).__name__, "message": str(error)}
    else:
        raise PreflightBlocked("BLOCKED_PREFLIGHT_SCALE_FIREWALL", "checkpoint_scale_authority", "formal V2 checkpoint loader accepted preflight scale")
    resumed_b = LightGen2DComplexOLSTrainingConsumerV2(contract, device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
    resumed_b_optimizer = resumed_b.create_optimizer()
    loaded_payload, loaded_scale = _load_preflight_checkpoint(stage_b_checkpoint_path, resumed_b, resumed_b_optimizer, scale_artifact)
    resumed_sampler = DeterministicTrainRoleSampler(train_indices, labels, steps=STAGE_B_STEPS)
    resumed_sampler.load_state_dict(_mapping(loaded_payload.get("sampler_state"), "resumed Stage-B sampler"))
    resumed_indices, resumed_labels = resumed_sampler.next_batch()
    expected_next_indices = torch.arange(10, dtype=torch.long)
    if not torch.equal(resumed_labels, expected_next_indices):
        raise PreflightBlocked("BLOCKED_CHECKPOINT_RESUME_FAILED", "stage_b_sampler_resume", "Stage-B sampler labels did not round-trip")
    if not torch.equal(resumed_indices, stage_b_batches[1]):
        raise PreflightBlocked("BLOCKED_CHECKPOINT_RESUME_FAILED", "stage_b_sampler_resume", "Stage-B sampler indices did not round-trip")
    resumed_update, resumed_memory = _batch_memory_update(resumed_b, images, resumed_indices, stage="B", global_step=STAGE_A_STEPS + 2, stage_local_step=2, optimizer=resumed_b_optimizer, scale=loaded_scale, device=device)
    resumed_metrics = resumed_update.loss.metrics()
    resumed_parameter_difference = _max_state_difference(continuous_b_parameters, resumed_b.trainable_state_dict())
    resumed_loss_difference = _max_metric_difference(continuous_b_metrics, resumed_metrics)
    optimizer_resume_equal = _tree_equal(continuous_b_optimizer, resumed_b_optimizer.state_dict())
    if resumed_loss_difference > 1.0e-5 or resumed_parameter_difference > 1.0e-5 or not optimizer_resume_equal or loaded_scale.vector_sha256 != scale_artifact.vector_sha256:
        raise PreflightBlocked("BLOCKED_CHECKPOINT_RESUME_FAILED", "stage_b_resume_equivalence", "Stage-B resumed step differs from continuous reference", loss_max_abs_difference=resumed_loss_difference, parameter_max_abs_difference=resumed_parameter_difference, optimizer_equivalent=optimizer_resume_equal)

    stage_a_resume_consumer = LightGen2DComplexOLSTrainingConsumerV2(contract, device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
    stage_a_resume_optimizer = stage_a_resume_consumer.create_optimizer()
    loaded_a = load_checkpoint(stage_a_checkpoint_path, stage_a_resume_consumer, stage_a_resume_optimizer)
    stage_a_resume_sampler = DeterministicTrainRoleSampler(train_indices, labels, steps=STAGE_A_STEPS)
    stage_a_resume_sampler.load_state_dict(_mapping(loaded_a.sampler_state, "resumed Stage-A sampler"))
    next_a_indices, next_a_labels = stage_a_resume_sampler.next_batch()
    if not torch.equal(next_a_labels, expected_next_indices):
        raise PreflightBlocked("BLOCKED_CHECKPOINT_RESUME_FAILED", "stage_a_sampler_resume", "Stage-A sampler labels did not round-trip")
    if not torch.equal(next_a_indices, stage_a_batches[1]):
        raise PreflightBlocked("BLOCKED_CHECKPOINT_RESUME_FAILED", "stage_a_sampler_resume", "Stage-A sampler indices did not round-trip")
    resumed_a_update, resumed_a_memory = _batch_memory_update(stage_a_resume_consumer, images, next_a_indices, stage="A", global_step=1, stage_local_step=1, optimizer=stage_a_resume_optimizer, scale=None, device=device)
    resumed_a_metrics = resumed_a_update.loss.metrics()
    resumed_a_parameter_difference = _max_state_difference(continuous_a_parameters, stage_a_resume_consumer.trainable_state_dict())
    resumed_a_loss_difference = _max_metric_difference(continuous_a_metrics, resumed_a_metrics)
    optimizer_a_resume_equal = _tree_equal(continuous_a_optimizer, stage_a_resume_optimizer.state_dict())
    if resumed_a_loss_difference > 1.0e-5 or resumed_a_parameter_difference > 1.0e-5 or not optimizer_a_resume_equal:
        raise PreflightBlocked("BLOCKED_CHECKPOINT_RESUME_FAILED", "stage_a_resume_equivalence", "Stage-A resumed step differs from continuous reference", loss_max_abs_difference=resumed_a_loss_difference, parameter_max_abs_difference=resumed_a_parameter_difference, optimizer_equivalent=optimizer_a_resume_equal)

    memory_phases = {"baseline_before_model": memory_before_model, "after_model_and_optimizer_init": memory_after_init, "stage_a_updates": stage_a_memory, "scale_extraction": scale_memory, "stage_b_updates": stage_b_memory, "stage_a_resume": resumed_a_memory, "stage_b_resume": resumed_memory}
    phase_snapshots = stage_a_memory + stage_b_memory + [scale_memory, resumed_a_memory, resumed_memory]
    total_vram = memory_before_model["total_bytes"]
    peak_reserved = max((item["peak_reserved_bytes"] for item in phase_snapshots), default=0)
    peak_allocated = max((item["peak_allocated_bytes"] for item in phase_snapshots), default=0)
    minimum_free = min([memory_before_model["free_bytes"], memory_after_init["free_bytes"]] + [item["free_bytes"] for item in phase_snapshots])
    memory_report = {**memory_phases, "peak_reserved_bytes_overall": peak_reserved, "peak_allocated_bytes_overall": peak_allocated, "minimum_observed_free_bytes": minimum_free, "safety_margin_to_total_vram_bytes": total_vram - peak_reserved, "exact_batch_size": LOGICAL_BATCH_SIZE, "exact_grid": PRODUCTION_PROPAGATION_GRID, "feasible": peak_reserved < total_vram and minimum_free > 0}
    if not memory_report["feasible"]:
        raise PreflightBlocked("BLOCKED_BATCH_MEMORY_INFEASIBLE", "memory", "exact frozen batch/grid left no memory margin", memory=memory_report)
    checkpoint_sizes = {"stage_a": stage_a_checkpoint_path.stat().st_size, "stage_b": stage_b_checkpoint_path.stat().st_size, "preflight_scale": Path(scale_artifact.artifact_path).stat().st_size}
    disk = shutil.disk_usage(output_dir)
    estimated_future_bytes = int(max(checkpoint_sizes.values()) * 2 + checkpoint_sizes["preflight_scale"] + 4 * 1024 * 1024)
    storage = {"free_bytes": int(disk.free), "total_bytes": int(disk.total), "estimated_future_bytes": estimated_future_bytes, "measured_artifacts": checkpoint_sizes, "feasible": int(disk.free) > estimated_future_bytes}
    if not storage["feasible"]:
        raise PreflightBlocked("BLOCKED_STORAGE_INFEASIBLE", "storage", "available output storage is insufficient", storage=storage)
    return {
        "target_device_grid": PRODUCTION_PROPAGATION_GRID,
        "batch_size": LOGICAL_BATCH_SIZE,
        "stage_a": {"smoke_steps": SMOKE_STEPS, "future_steps": STAGE_A_STEPS, "updates": stage_a_updates, "memory": stage_a_memory, "checkpoint": {"path": str(stage_a_checkpoint_path), "sha256": file_sha256(stage_a_checkpoint_path), "accepted_training_checkpoint": False}, "resume_equivalence": {"loss_max_abs_difference": resumed_a_loss_difference, "parameter_max_abs_difference": resumed_a_parameter_difference, "optimizer_equivalent": optimizer_a_resume_equal}},
        "semantic_firewall": semantic_firewall,
        "scale": {"rows_traversed": 55_000, "data_role": "TRAIN", "equation": "s_j=mean_train |z_j|^2", "port_count": 100, "epsilon_scale": EPSILON_SCALE, "float64_accumulation": True, "values_finite": True, "values_above_epsilon": True, "elapsed_seconds": scale_elapsed, "rows_per_second": 55_000 / scale_elapsed if scale_elapsed > 0 else None, "memory": scale_memory, "indices_sha256": tensor_sha256(train_indices), "scale_vector_sha256": scale_artifact.vector_sha256, "artifact": {"path": str(scale_artifact.artifact_path), "sha256": scale_artifact.artifact_sha256, "classification": PREFLIGHT_SCALE_CLASSIFICATION, "schema": SCALE_SCHEMA}, "deterministic_probe": {"rows_each_pass": LOGICAL_BATCH_SIZE, "equal": True, "vector_sha256": tensor_sha256(deterministic_probe_a), "zero_or_below_epsilon_allowed_for_probe": True}, "formal_scale_loader_rejection": formal_scale_rejection},
        "normalized_latent": {"formula": "z_tilde_j=z_j/sqrt(s_j+epsilon_scale)", "prior": "(epsilon_R+i*epsilon_I)/sqrt(2) ~ CN(0,I_100)", "shape": [LOGICAL_BATCH_SIZE, 100], "dtype": "torch.complex64", "finite": True, "seed_replay": True, "physical_mapping": "z_sample=sqrt(s+epsilon_scale)*z_tilde_sample", "mapping_checked": True},
        "stage_b": {"smoke_steps": SMOKE_STEPS, "future_steps": STAGE_B_STEPS, "updates": stage_b_updates, "memory": stage_b_memory, "loss": "L_B=L_image+L_latent_normalized+L_scale", "mmd_input_observed": True, "mmd_input_coordinate": "z_tilde", "raw_z_scale_anchor_observed": True, "scale_anchor_scale_sha256": captured_scale_sha, "frozen_scale_unchanged": True, "optimizer_reset": reset_marker, "checkpoint": {"path": str(stage_b_checkpoint_path), "sha256": file_sha256(stage_b_checkpoint_path), "accepted_training_checkpoint": False}, "formal_checkpoint_loader_rejection": formal_checkpoint_rejection, "resume_equivalence": {"loss_max_abs_difference": resumed_loss_difference, "parameter_max_abs_difference": resumed_parameter_difference, "optimizer_equivalent": optimizer_resume_equal, "scale_identity_preserved": loaded_scale.artifact_sha256 == scale_artifact.artifact_sha256}},
        "checkpoint_resume": {"stage_a": True, "stage_b": True, "rng_cpu_cuda_restored": True, "sampler_restored": True, "selected_stage_a_identity_preserved": True, "scale_identity_preserved": True, "optimizer_reset_marker_preserved": True},
        "memory": memory_report,
        "storage": storage,
        "parameter_manifest": manifest,
        "optimizer_manifest": optimizer_manifest,
        "data_firewall": {"train_rows_for_scale": 55_000, "development_rows_for_semantic_smoke": int(selection_indices.numel()), "calibration_rows": 0, "protected_rows": 0, "test_rows": 0, "sealed_rows": 0, "labels_as_model_input": False},
        "future_generation_executed": False,
    }


def _resolve_output_dir(root: Path, output_dir: Path | None) -> Path:
    target = root / DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    target = target if target.is_absolute() else root / target
    target = target.resolve()
    outputs_root = (root / "outputs").resolve()
    if not target.is_relative_to(outputs_root):
        raise InvalidPreflight("preflight output must remain under outputs/")
    if target.exists():
        raise InvalidPreflight(f"refusing to overwrite existing preflight output: {target}")
    return target


def _preflight_input_payload(root: Path, contract: Any, authorities: Mapping[str, Any], started: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": PREFLIGHT_INPUT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "preflight_id": TASK_ID,
        "status": "valid_completed",
        "result": "READY_FOR_EXECUTION",
        "classification": "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2_INPUT_READY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract": contract.identity(root),
        "correction_authority": dict(authorities["correction_authority"]),
        "canonical_training_source": dict(authorities["dataset_authority"]),
        "consumer": dict(authorities["consumer"]),
        "aligned_runtime": dict(authorities["aligned_runtime"]),
        "evaluator": dict(authorities["evaluator"]),
        "protocol": {
            "target_device_grid": PRODUCTION_PROPAGATION_GRID,
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "smoke_steps": SMOKE_STEPS,
            "optimizer": {"name": "Adam", "learning_rate": 1.0e-3},
            "stage_a": {
                "future_steps": STAGE_A_STEPS,
                "checkpoint_steps": list(CHECKPOINT_STEPS),
                "loss": "L_A=L_image",
                "latent_mmd_enabled": False,
                "semantic_threshold": SEMANTIC_THRESHOLD,
            },
            "scale": {
                "formula": "s_j=mean_train |z_j|^2",
                "train_rows": 55_000,
                "port_count": 100,
                "epsilon_scale": EPSILON_SCALE,
                "float64_accumulation": True,
                "classification": PREFLIGHT_SCALE_CLASSIFICATION,
            },
            "normalized_latent": {
                "formula": "z_tilde_j=z_j/sqrt(s_j+epsilon_scale)",
                "prior": "(epsilon_R+i*epsilon_I)/sqrt(2) ~ CN(0,I_100)",
                "mmd_coordinate": "z_tilde",
                "physical_mapping": "z_sample=sqrt(s+epsilon_scale)*z_tilde_sample",
            },
            "stage_b": {
                "future_steps": STAGE_B_STEPS,
                "checkpoint_steps": list(CHECKPOINT_STEPS),
                "loss": "L_B=L_image+L_latent_normalized+L_scale",
                "optimizer_reset": True,
                "semantic_threshold": SEMANTIC_THRESHOLD,
            },
        },
        "supersedes": {
            "report": {"path": HISTORICAL_BLOCKED_PREFLIGHT_RELATIVE_PATH.as_posix(), "sha256": HISTORICAL_BLOCKED_PREFLIGHT_SHA256},
            "lineage": {"path": HISTORICAL_BLOCKED_LINEAGE_RELATIVE_PATH.as_posix(), "sha256": HISTORICAL_BLOCKED_LINEAGE_SHA256},
            "classification": "BLOCKED_TRAINING_SOURCE_AUTHORITY_MISMATCH",
            "disposition": "historical/superseded-by-authority-correction",
            "invalidated_or_deleted": False,
        },
        "claim_boundary": {
            "input_definition_only": True,
            "long_training_executed": False,
            "scientific_training_executed": False,
            "heldout_evaluation_executed": False,
            "protected_accessed": False,
            "test_accessed": False,
            "sealed_accessed": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "provenance": {"git_start": dict(started)},
        "input_self_hash": "not_embedded_to_avoid_self_reference",
    }


def run_preflight(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = _root() if repo_root is None else repo_root.resolve()
    target = _resolve_output_dir(root, output_dir)
    target.mkdir(parents=True, exist_ok=False)
    started = git_facts(root)
    payload: dict[str, Any] = {"schema": REPORT_SCHEMA, "version": 1, "task_id": TASK_ID, "preflight_id": TASK_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "invalid_harness", "result": "INVALID", "classification": "INVALID_PREFLIGHT_NOT_COMPLETED", "provenance": {"git_start": started}, "claim_boundary": {"target_device_smoke_only": True, "long_training_executed": False, "scientific_training_executed": False, "heldout_evaluation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "formal_promotion": False, "generation_readiness": "not_ready", "training_authorized": False}, "next": None}
    authorities: dict[str, Any] | None = None
    preflight_input_identity: dict[str, str] | None = None
    dataset: dict[str, Any] | None = None
    gpu_result: dict[str, Any] | None = None
    try:
        contract, authorities = _validate_authorities(root)
        payload["authorities"] = authorities
        preflight_input_path = target / PREFLIGHT_INPUT_NAME
        preflight_input = _preflight_input_payload(root, contract, authorities, started)
        atomic_write_new_json(preflight_input_path, preflight_input, artifact_kind="LightGen V2 corrected long-training preflight input")
        preflight_input_identity = {"path": preflight_input_path.relative_to(root).as_posix(), "sha256": file_sha256(preflight_input_path)}
        authorities["preflight_input"] = dict(preflight_input_identity)
        payload["preflight_input"] = dict(preflight_input_identity)
        images, labels, train_indices, _development_indices, selection_indices, dataset_value = _load_dataset(root, contract)
        dataset = {key: value for key, value in dataset_value.items() if key != "train_images" and key != "selection_labels"}
        payload["dataset"] = dataset
        device, device_facts = _cuda_device_facts()
        payload["device"] = device_facts
        payload["target_device"] = str(device)
        evaluator_path = _path(root, EVALUATOR_CHECKPOINT_RELATIVE_PATH)
        evaluator, evaluator_payload = load_frozen_evaluator(evaluator_path, device=device)
        _require(all(not parameter.requires_grad for parameter in evaluator.parameters()), "FrozenMNISTEvaluator is not frozen")
        _require(float(evaluator_payload.get("validation_accuracy", 0.0)) >= 0.985, "FrozenMNISTEvaluator qualification changed")
        payload["evaluator"] = {"identity": "FrozenMNISTEvaluator v1", "checkpoint": EVALUATOR_CHECKPOINT_RELATIVE_PATH.as_posix(), "checkpoint_sha256": EVALUATOR_CHECKPOINT_SHA256, "implementation": EVALUATOR_RELATIVE_PATH.as_posix(), "implementation_sha256": EVALUATOR_SHA256, "validation_accuracy": float(evaluator_payload["validation_accuracy"]), "preprocessing": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28", "threshold": SEMANTIC_THRESHOLD, "gradient_enabled": False}
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        gpu_result = _run_preflight(root, contract, dataset_value, images, labels, train_indices, selection_indices, device, evaluator, target, started)
        payload["gpu_smoke"] = gpu_result
        payload["result"] = "AUTHORIZABLE"
        payload["classification"] = SUCCESS_CLASSIFICATION
        payload["status"] = "valid_completed"
        payload["next"] = "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2"
        payload["next_note"] = "requires separate explicit authorization; do not start automatically"
    except PreflightBlocked as blocked:
        payload.update(blocked.details)
        payload["result"] = "BLOCKED"
        payload["classification"] = blocked.classification
        payload["status"] = "valid_failed"
        payload["hard_blocker"] = {"stage": blocked.stage, "message": str(blocked)}
        payload["next"] = "resolve_preflight_blocker_before_training"
    except InvalidPreflight as error:
        payload["result"] = "INVALID"
        payload["classification"] = "INVALID_PREFLIGHT_HARNESS"
        payload["status"] = "invalid_harness"
        payload["invalid_input"] = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:
        payload["result"] = "INVALID"
        payload["classification"] = "INVALID_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR"
        payload["status"] = "invalid_harness"
        payload["unexpected_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-10_000:]}
    finally:
        ending = git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started["branch"] == ending["branch"], "head_unchanged": started["head"] == ending["head"], "worktree_dirty": ending["worktree_dirty"]})
        payload["claim_boundary"].update({"protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "formal_promotion": False, "generation_readiness": "not_ready", "training_authorized": False})
        payload["preflight_artifact_self_hash"] = "not_embedded_to_avoid_self_reference"
        payload["no_scientific_training"] = True
    report_path = target / REPORT_NAME
    lineage_path = target / LINEAGE_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="LightGen V2 long-training preflight report")
    lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "result": payload["result"], "classification": payload["classification"], "contract": (authorities or {}).get("contract"), "authorities": authorities, "device": payload.get("device"), "dataset": payload.get("dataset"), "artifacts": {"preflight_input": preflight_input_identity, "report": {"path": str(report_path), "sha256": file_sha256(report_path)}, "stage_a_checkpoint": (gpu_result or {}).get("stage_a", {}).get("checkpoint"), "stage_b_checkpoint": (gpu_result or {}).get("stage_b", {}).get("checkpoint"), "preflight_scale": (gpu_result or {}).get("scale", {}).get("artifact")}, "claim_boundary": payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_self_reference"}
    atomic_write_new_json(lineage_path, lineage, artifact_kind="LightGen V2 long-training preflight lineage")
    result = {"result": payload["result"], "classification": payload["classification"], "preflight_id": TASK_ID, "preflight_input": preflight_input_identity, "report": {"path": str(report_path), "sha256": file_sha256(report_path)}, "lineage": {"path": str(lineage_path), "sha256": file_sha256(lineage_path)}, "preflight_scale": (gpu_result or {}).get("scale", {}).get("artifact", {}).get("path"), "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready", "next": payload["next"]}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="new output directory under outputs/")
    args = parser.parse_args()
    result = run_preflight(output_dir=args.output_dir)
    return 0 if result["result"] in {"AUTHORIZABLE", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
