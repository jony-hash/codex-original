"""Resolve the bounded V3 semantic-objective failure attribution.

This is a read-only, no-update consumer of the immutable V3 step-300
control/candidate checkpoints.  It measures the matched DEV output paths,
the encoder/generator cross swaps, the existing full-complex OLS geometry,
and a fixed TRAIN-centroid diagnostic.  It never creates an optimizer,
performs a backward pass, updates an optical parameter, reads protected/test/
sealed data, or opens Stage B.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

import ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training as gauge_source
import ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 as v2
from ldm_project.v2.evidence import atomic_write_new_json, file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.student.evaluator import FrozenMNISTEvaluator, SmoothMNISTEvaluator, stratified_mnist_indices


TASK_ID = "RESOLVE_V3_OBJECTIVE_CORRECTION_FAILURE_ATTRIBUTION_V1"
OUTER_GATE = "V3_OBJECTIVE_FAILURE_ATTRIBUTION_RESOLVED"
REPORT_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_lineage.v1"
INPUT_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_input.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_checkpoint.v1"
PANEL_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_raw_cmos_panel.v1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "383197a930dd48f98294120f8cf867444f994f85"
V3_CONTRACT_PATH = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_contract_v3_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_contract.v3.json"
)
V3_CONTRACT_SHA256 = "e537fbf4b856519a71e48c16a48c149fadd327e2ad12db84f64c953413fb53f6"
PREFLIGHT_PATH = Path("scripts/research/execute_lightgen_2d_semantic_complex_ols_objective_correction_v3.py")
PREFLIGHT_SHA256 = "64c45782ad9e9d23a26c8c1e419996322ed2f84fc74846be918fabc795a01988"
PREFLIGHT_REPORT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_report.v1.json"
)
PREFLIGHT_REPORT_SHA256 = "265e9b80582161872a207041cd5edc2279b6f72c5729aa6f5406f9646e354468"
PREFLIGHT_INPUT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_input.v1.json"
)
PREFLIGHT_INPUT_SHA256 = "631eb85792062f00e2e59344d091a6aac203225e05c56e1a4c27150859772d36"
PREFLIGHT_LINEAGE_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_lineage.v1.json"
)
PREFLIGHT_LINEAGE_SHA256 = "f600bf924c3494f0b348fccde19aa6d49d5b33923cff622d5a5722394cb08e31"

SCIENTIFIC_REPORT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_report.v1.json"
)
SCIENTIFIC_REPORT_SHA256 = "8d7930449a1872e685ad39dafa499d89fdcc1d6bb5a9317e7ba06c554f54f31a"
SCIENTIFIC_ENTRYPOINT_PATH = Path(
    "scripts/research/execute_lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific.py"
)
SCIENTIFIC_ENTRYPOINT_SHA256 = "ca9eb8db3c7360f31da8d08ed6d0f61fa3bc61e8f0ce3de0f5a952803ff44ae4"

CONTROL_CHECKPOINT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_control_step_300_checkpoint.v1.pt"
)
CONTROL_CHECKPOINT_SHA256 = "a42a43fdf99cf12d9abd1917a6d025fc2fcb9ecee39f0c05429dd8d7a8870079"
CANDIDATE_CHECKPOINT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_candidate_step_300_checkpoint.v1.pt"
)
CANDIDATE_CHECKPOINT_SHA256 = "56ad62e41e0776f89d6c6d50fa5475b330659607e6630d28dc737ac6337e39d2"
CONTROL_PANEL_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_control_step_300_raw_cmos_panel.v1.pt"
)
CONTROL_PANEL_SHA256 = "dcca8fdc1ac39b2a712d21e15441b3cb5728d496a71494b8e758ae3a004e648b"
CONTROL_PANEL_TENSOR_SHA256 = "56c6cc2394eb62d46d582d763b5ca2d8ba766f2c56807b012a0c64dafef16716"
CANDIDATE_PANEL_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_candidate_step_300_raw_cmos_panel.v1.pt"
)
CANDIDATE_PANEL_SHA256 = "3ec46c070d18b0c76dfe5a4990434c2aa2f7eef2d6d51514ccac508315eff433"
CANDIDATE_PANEL_TENSOR_SHA256 = "0fb1b331a59ff90af1df487cf25885f0dd709f60021a8df7907f6da9a9ae274f"

ALIGNMENT_RUNTIME_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py"
)
ALIGNMENT_RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
GAUGE_HELPER_PATH = Path("src/ldm_project/v2/application/research/lightgen_2d_semantic_complex_ols_training.py")
GAUGE_HELPER_SHA256 = "fae7ffcd5f724aa4a07966e0f75dbe77653eaf85cf9907093606ceb884517d1c"
EVALUATOR_IMPLEMENTATION_PATH = Path("src/ldm_project/v2/student/evaluator.py")
EVALUATOR_IMPLEMENTATION_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"

TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
DEV_SELECTION_SHA256 = "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8"
V2_CONTRACT_SHA256 = "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_v1_attempt_02"
)
INPUT_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_input.v1.json"
REPORT_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_report.v1.json"
LINEAGE_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_failure_attribution_lineage.v1.json"

PRODUCTION_PROPAGATION_GRID = 2048
INITIALIZATION_SEED = 1234
DEVELOPMENT_SELECTION_SEED = 4321
LOGICAL_BATCH_SIZE = 10
BOOTSTRAP_SEED = 271828
BOOTSTRAP_REPLICATES = 999
CENTROID_BOOTSTRAP_SEED = 271829
CENTROID_ROWS_PER_CLASS = 100
SEMANTIC_FEATURE_DIMENSION = 128
STAGE_A_FIREWALL = 0.937
RECALL_MATERIAL_MARGIN = 0.05
GAUGE_NUMERICAL_RANK_SCALE = 1.0e-6
PANEL_RTOL = 1.0e-5
PANEL_ATOL = 1.0e-6

CLASSIFICATIONS = {
    "provider": "PROVIDER_SPECIFIC_ALIGNMENT_SUPPORTED",
    "generator": "OLS_SEMANTIC_GAIN_GENERATOR_TRANSFER_FAILURE_SUPPORTED",
    "ineffective": "SEMANTIC_OBJECTIVE_INEFFECTIVE_SUPPORTED",
    "mixed": "V3_FAILURE_ATTRIBUTION_MIXED_OR_UNRESOLVED",
}


class InvalidDiagnosis(RuntimeError):
    """The frozen diagnosis harness or an authority is invalid."""


class DiagnosisBlocked(RuntimeError):
    """A valid external boundary blocks this one-shot diagnostic."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidDiagnosis(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _path(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    _require(not value.is_absolute(), f"absolute authority path is not allowed: {value}")
    return (root / value).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidDiagnosis(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


def _validate_hash(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    actual = file_sha256(path)
    _require(actual == expected, f"{name} SHA mismatch: expected {expected}, observed {actual}")


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
            digest.update(tensor_sha256(item).encode("ascii"))
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


def _load_preflight_module(root: Path) -> Any:
    path = _path(root, PREFLIGHT_PATH)
    _validate_hash(path, PREFLIGHT_SHA256, "accepted V3 preflight entrypoint")
    spec = importlib.util.spec_from_file_location("accepted_v3_preflight_for_attribution", path)
    _require(spec is not None and spec.loader is not None, "accepted V3 preflight module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_scientific_authority(root: Path, preflight: Any) -> tuple[v2.FrozenV2TrainingContract, dict[str, Any], dict[str, Any]]:
    contract, v2_contract, authorities = preflight._validate_v3_authority(root)
    _validate_hash(_path(root, V3_CONTRACT_PATH), V3_CONTRACT_SHA256, "V3 contract")
    _validate_hash(_path(root, PREFLIGHT_INPUT_PATH), PREFLIGHT_INPUT_SHA256, "accepted preflight input")
    _validate_hash(_path(root, PREFLIGHT_REPORT_PATH), PREFLIGHT_REPORT_SHA256, "accepted preflight report")
    _validate_hash(_path(root, PREFLIGHT_LINEAGE_PATH), PREFLIGHT_LINEAGE_SHA256, "accepted preflight lineage")
    preflight_report = _read_json(_path(root, PREFLIGHT_REPORT_PATH), "accepted preflight report")
    _require(preflight_report.get("result") == "AUTHORIZABLE" and preflight_report.get("status") == "valid_completed", "accepted preflight is not AUTHORIZABLE")

    _validate_hash(_path(root, SCIENTIFIC_REPORT_PATH), SCIENTIFIC_REPORT_SHA256, "V3 scientific report")
    _validate_hash(_path(root, SCIENTIFIC_ENTRYPOINT_PATH), SCIENTIFIC_ENTRYPOINT_SHA256, "V3 scientific entrypoint")
    scientific = _read_json(_path(root, SCIENTIFIC_REPORT_PATH), "V3 scientific report")
    _require(scientific.get("result") == "VALID_COMPLETED" and scientific.get("status") == "valid_completed", "V3 scientific report is not valid_completed")
    _require(scientific.get("classification") == "V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED", "V3 scientific result is not the accepted failed comparison")
    _require(scientific.get("task_id") == "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_V3", "V3 scientific task identity changed")
    boundary = _mapping(scientific.get("claim_boundary"), "V3 scientific claim boundary")
    for key in ("stage_b_executed", "scale_extraction_executed", "prior_generation_executed", "protected_accessed", "test_accessed", "sealed_accessed", "formal_promotion"):
        _require(boundary.get(key) is False, f"V3 scientific claim boundary is open: {key}")

    scientific_arms = _mapping(scientific.get("arms"), "scientific arms")
    scientific_control_arm = _mapping(scientific_arms.get("control"), "scientific control arm")
    scientific_candidate_arm = _mapping(scientific_arms.get("candidate"), "scientific candidate arm")
    report_checkpoint_refs = {
        "control": _mapping(_mapping(scientific_control_arm.get("checkpoints"), "scientific control checkpoints").get("300"), "scientific control step-300 checkpoint"),
        "candidate": _mapping(_mapping(scientific_candidate_arm.get("checkpoints"), "scientific candidate checkpoints").get("300"), "scientific candidate step-300 checkpoint"),
    }
    expected_checkpoint_refs = {
        "control": {"path": CONTROL_CHECKPOINT_PATH.as_posix(), "sha256": CONTROL_CHECKPOINT_SHA256, "step": 300},
        "candidate": {"path": CANDIDATE_CHECKPOINT_PATH.as_posix(), "sha256": CANDIDATE_CHECKPOINT_SHA256, "step": 300},
    }
    for arm in ("control", "candidate"):
        observed = _mapping(report_checkpoint_refs[arm], f"scientific {arm} step-300 checkpoint ref")
        for key in ("path", "sha256", "step"):
            _require(observed.get(key) == expected_checkpoint_refs[arm][key], f"scientific {arm} checkpoint authority changed: {key}")

    control_endpoint = _mapping(_mapping(scientific_control_arm.get("evaluations"), "scientific control evaluations").get("300"), "scientific control endpoint")
    candidate_endpoint = _mapping(_mapping(scientific_candidate_arm.get("evaluations"), "scientific candidate evaluations").get("300"), "scientific candidate endpoint")
    report_panel_refs = {
        "control": _mapping(_mapping(control_endpoint.get("raw_panel"), "scientific control raw panel").get("artifact"), "scientific control raw panel artifact"),
        "candidate": _mapping(_mapping(candidate_endpoint.get("raw_panel"), "scientific candidate raw panel").get("artifact"), "scientific candidate raw panel artifact"),
    }
    expected_panel_refs = {
        "control": {"path": CONTROL_PANEL_PATH.as_posix(), "sha256": CONTROL_PANEL_SHA256},
        "candidate": {"path": CANDIDATE_PANEL_PATH.as_posix(), "sha256": CANDIDATE_PANEL_SHA256},
    }
    for arm in ("control", "candidate"):
        observed = _mapping(report_panel_refs[arm], f"scientific {arm} endpoint panel ref")
        for key in ("path", "sha256"):
            _require(observed.get(key) == expected_panel_refs[arm][key], f"scientific {arm} panel authority changed: {key}")

    _validate_hash(_path(root, ALIGNMENT_RUNTIME_PATH), ALIGNMENT_RUNTIME_SHA256, "LightGen aligned runtime")
    _validate_hash(_path(root, GAUGE_HELPER_PATH), GAUGE_HELPER_SHA256, "project gauge helper")
    _validate_hash(_path(root, EVALUATOR_IMPLEMENTATION_PATH), EVALUATOR_IMPLEMENTATION_SHA256, "evaluator implementation")
    _require(v2_contract.sha256 == V2_CONTRACT_SHA256, "corrected V2 contract hash changed")
    _require(v2_contract.payload.get("data_contract", {}).get("source_artifacts", {}).get("training", {}).get("sha256") == TRAINING_SOURCE_SHA256, "training source authority changed")

    return v2_contract, dict(authorities), {
        "v3_scientific": {
            "report": {"path": SCIENTIFIC_REPORT_PATH.as_posix(), "sha256": SCIENTIFIC_REPORT_SHA256},
            "entrypoint": {"path": SCIENTIFIC_ENTRYPOINT_PATH.as_posix(), "sha256": SCIENTIFIC_ENTRYPOINT_SHA256},
            "classification": scientific.get("classification"),
            "checkpoint_refs": expected_checkpoint_refs,
            "panel_refs": expected_panel_refs,
        },
        "preflight": {
            "entrypoint": {"path": PREFLIGHT_PATH.as_posix(), "sha256": PREFLIGHT_SHA256},
            "input": {"path": PREFLIGHT_INPUT_PATH.as_posix(), "sha256": PREFLIGHT_INPUT_SHA256},
            "report": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256},
            "lineage": {"path": PREFLIGHT_LINEAGE_PATH.as_posix(), "sha256": PREFLIGHT_LINEAGE_SHA256},
            "result": preflight_report.get("result"),
        },
        "runtime": {"path": ALIGNMENT_RUNTIME_PATH.as_posix(), "sha256": ALIGNMENT_RUNTIME_SHA256},
        "gauge_helper": {"path": GAUGE_HELPER_PATH.as_posix(), "sha256": GAUGE_HELPER_SHA256},
        "evaluator_implementation": {"path": EVALUATOR_IMPLEMENTATION_PATH.as_posix(), "sha256": EVALUATOR_IMPLEMENTATION_SHA256},
        "corrected_v2_contract": {"path": v2.V2_CONTRACT_RELATIVE_PATH.as_posix(), "sha256": V2_CONTRACT_SHA256},
        "training_source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256},
        "hash_on_trust_boundary": True,
    }


def _load_checkpoint(root: Path, path: Path, digest: str, arm: str, v2_contract: v2.FrozenV2TrainingContract) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_hash(_path(root, path), digest, f"{arm} scientific checkpoint")
    payload = _mapping(safe_torch_load(_path(root, path), expected_sha256=digest, map_location="cpu"), f"{arm} scientific checkpoint payload")
    _require(payload.get("schema") == CHECKPOINT_SCHEMA and payload.get("arm") == arm and payload.get("checkpoint_step") == 300, f"{arm} checkpoint identity changed")
    _require(payload.get("primary_endpoint") is True, f"{arm} checkpoint is not the primary endpoint")
    _require(payload.get("contract") == {"path": V3_CONTRACT_PATH.as_posix(), "sha256": V3_CONTRACT_SHA256}, f"{arm} checkpoint V3 contract identity changed")
    _require(payload.get("v2_contract") == v2_contract.identity(), f"{arm} checkpoint V2 contract identity changed")
    _require(payload.get("preflight") == {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256}, f"{arm} checkpoint preflight identity changed")
    roles = _mapping(payload.get("data_roles"), f"{arm} checkpoint data roles")
    for role in ("protected", "test", "sealed"):
        _require(roles.get(role) is False, f"{arm} checkpoint role boundary is open: {role}")
    _require(payload.get("training_authorized") is False and payload.get("stage_b_executed") is False and payload.get("generation_executed") is False, f"{arm} checkpoint claim boundary changed")
    state = _mapping(payload.get("runtime_state"), f"{arm} checkpoint runtime state")
    expected_keys = {"encoder.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport", "generator.phase"}
    _require(set(state) == expected_keys, f"{arm} checkpoint runtime state keys changed")
    _require(tuple(state["encoder.phase"].shape) == (2, 1, 100, 100) and tuple(state["generator.phase"].shape) == (2, 1, 100, 100), f"{arm} phase shape changed")
    _require(state["encoder.phase"].dtype == torch.float32 and state["generator.phase"].dtype == torch.float32, f"{arm} phase dtype changed")
    _finite(state["encoder.phase"], f"{arm} encoder phase")
    _finite(state["generator.phase"], f"{arm} generator phase")
    return dict(payload), _cpu_tree(state)


def _load_panel(root: Path, path: Path, digest: str, arm: str, v2_contract: v2.FrozenV2TrainingContract) -> tuple[torch.Tensor, dict[str, Any]]:
    _validate_hash(_path(root, path), digest, f"{arm} raw-CMOS panel")
    payload = _mapping(safe_torch_load(_path(root, path), expected_sha256=digest, map_location="cpu"), f"{arm} raw-CMOS panel payload")
    _require(payload.get("schema") == PANEL_SCHEMA and payload.get("arm") == arm and payload.get("checkpoint_step") == 300, f"{arm} panel identity changed")
    _require(payload.get("contract") == {"path": V3_CONTRACT_PATH.as_posix(), "sha256": V3_CONTRACT_SHA256}, f"{arm} panel contract identity changed")
    _require(payload.get("v2_contract") == v2_contract.identity(), f"{arm} panel V2 contract identity changed")
    _require(payload.get("selection_indices_sha256") == DEV_SELECTION_SHA256, f"{arm} panel DEV selection identity changed")
    raw = payload.get("raw_cmos")
    _require(torch.is_tensor(raw) and tuple(raw.shape) == (1000, 1, 200, 200) and raw.dtype == torch.float32, f"{arm} raw-CMOS panel shape/dtype changed")
    _finite(raw, f"{arm} stored raw-CMOS panel")
    _require(tensor_sha256(raw) == (CONTROL_PANEL_TENSOR_SHA256 if arm == "control" else CANDIDATE_PANEL_TENSOR_SHA256), f"{arm} stored panel tensor identity changed")
    return raw.contiguous(), {"path": path.as_posix(), "sha256": digest, "tensor_sha256": tensor_sha256(raw), "shape": list(raw.shape), "role": "fixed DEV selection only"}


def _load_roles(root: Path, preflight: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    images, labels, train_indices, source_metadata = preflight._load_train_role(root)
    _validate_hash(_path(root, TRAINING_SOURCE_PATH), TRAINING_SOURCE_SHA256, "canonical TRAIN source")
    expected_train, development_indices = stratified_mnist_indices(labels, seed=INITIALIZATION_SEED, validation_per_label=500)
    _require(torch.equal(train_indices, expected_train), "TRAIN split differs from the frozen V3 split")
    development_labels = labels.index_select(0, development_indices)
    selection: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu").manual_seed(DEVELOPMENT_SELECTION_SEED)
    for digit in range(10):
        matches = development_indices[development_labels == digit]
        _require(matches.numel() == 500, f"DEV role is not balanced at 500 rows for class {digit}")
        selection.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:100]))
    selection_indices = torch.cat(selection).contiguous()
    _require(tensor_sha256(selection_indices) == DEV_SELECTION_SHA256, "fixed DEV selection identity changed")
    selection_labels = labels.index_select(0, selection_indices).contiguous()
    _require(torch.equal(torch.bincount(selection_labels, minlength=10), torch.full((10,), 100, dtype=torch.long)), "fixed DEV selection class counts changed")
    centroid_parts: list[torch.Tensor] = []
    for digit in range(10):
        candidates = train_indices[labels.index_select(0, train_indices) == digit]
        sorted_candidates = torch.sort(candidates).values
        _require(sorted_candidates.numel() >= CENTROID_ROWS_PER_CLASS, f"TRAIN centroid class {digit} is too small")
        centroid_parts.append(sorted_candidates[:CENTROID_ROWS_PER_CLASS])
    train_centroid_indices = torch.cat(centroid_parts).contiguous()
    train_centroid_labels = labels.index_select(0, train_centroid_indices).contiguous()
    _require(torch.equal(torch.bincount(train_centroid_labels, minlength=10), torch.full((10,), CENTROID_ROWS_PER_CLASS, dtype=torch.long)), "TRAIN centroid subset is not balanced")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "TRAIN/DEV roles overlap or omit source rows")
    metadata = {
        "source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000, "values_read_from_this_train_role_source": True},
        "train": {"contract_rows": 55_000, "counts_by_class": source_metadata["train"]["counts_by_class"], "indices_sha256": source_metadata["train"]["indices_sha256"]},
        "development": {"contract_rows": 5_000, "selection_rows": 1_000, "selection_per_class": 100, "selection_seed": DEVELOPMENT_SELECTION_SEED, "indices_sha256": tensor_sha256(development_indices), "selection_indices_sha256": tensor_sha256(selection_indices)},
        "train_centroid_subset": {"rule": "within_train_class_source_index_ascending_first_100_v1", "rows_used": 1_000, "rows_per_class": CENTROID_ROWS_PER_CLASS, "indices_sha256": tensor_sha256(train_centroid_indices), "labels_sha256": tensor_sha256(train_centroid_labels), "source_role": "TRAIN only; diagnostic centroid construction"},
        "calibration": {"rows_read": 0, "status": "forbidden"},
        "protected": {"rows_read": 0, "status": "forbidden; path not opened"},
        "test": {"rows_read": 0, "status": "forbidden; path not opened"},
        "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
        "labels_as_model_input": False,
    }
    return images, labels, train_indices, development_indices, selection_indices, train_centroid_indices, {**metadata, "selection_labels_sha256": tensor_sha256(selection_labels)}


def _new_consumer(contract: v2.FrozenV2TrainingContract, device: torch.device, preflight: Any) -> Any:
    return preflight._new_consumer(contract, device)


def _load_state(consumer: Any, state: Mapping[str, Any], preflight: Any) -> None:
    preflight._load_runtime_state(consumer, state)
    consumer.runtime.eval()
    consumer._enforce_parameter_firewall()


def _combined_state(control: Mapping[str, Any], candidate: Mapping[str, Any], encoder: str, generator: str) -> dict[str, Any]:
    _require(encoder in {"control", "candidate"} and generator in {"control", "candidate"}, "invalid cross-swap arm")
    source_encoder = control if encoder == "control" else candidate
    source_generator = control if generator == "control" else candidate
    result = _cpu_tree(control)
    result["encoder.phase"] = source_encoder["encoder.phase"].clone()
    result["generator.phase"] = source_generator["generator.phase"].clone()
    return result


def _check_direct_result(consumer: Any, result: Any, batch_size: int, preflight: Any) -> None:
    _require(result.image_input is not None and result.ols_input_field is not None, "real-path image/OLS boundaries are missing")
    _require(result.z_ols.source == "real_encoded" and tuple(result.z_ols.coefficients.shape) == (batch_size, 100), "real-path z shape/source changed")
    _require(result.z_ols.coefficients.dtype == torch.complex64, "real-path z dtype changed")
    _require(result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "real-path modal projection boundary is inactive")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "real-path W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "real-path transport changed")
    _require(torch.equal(consumer.runtime.ols.transport.detach().cpu(), torch.eye(100, dtype=torch.complex64)), "real-path transport is not T=I_100")
    _require(tuple(result.raw_intensity.shape) == (batch_size, 1, 200, 200) and result.raw_intensity.dtype == torch.float32, "real-path raw CMOS shape/dtype changed")
    _finite(result.raw_intensity, "real-path raw CMOS")
    _finite(result.image_input.slm_intensity, "real-path target SLM intensity")


def _check_external_result(consumer: Any, result: Any, batch_size: int) -> None:
    _require(result.image_input is None and result.ols_input_field is None and result.encoder_output_field is None, "cross-swap unexpectedly re-entered image encoder")
    _require(tuple(result.z_ols.coefficients.shape) == (batch_size, 100) and result.z_ols.coefficients.dtype == torch.complex64, "cross-swap z shape/dtype changed")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "cross-swap W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "cross-swap transport changed")
    _require(torch.equal(consumer.runtime.ols.transport.detach().cpu(), torch.eye(100, dtype=torch.complex64)), "cross-swap transport is not T=I_100")
    _require(tuple(result.raw_intensity.shape) == (batch_size, 1, 200, 200) and result.raw_intensity.dtype == torch.float32, "cross-swap raw CMOS shape/dtype changed")
    _finite(result.raw_intensity, "cross-swap raw CMOS")


def _new_accumulator() -> dict[str, Any]:
    return {
        "count": 0,
        "image_loss_sum": 0.0,
        "feature_loss_sum": 0.0,
        "feature_distances": [],
        "frozen_correct": [],
        "smooth_correct": [],
        "raw_finite": True,
        "raw_structural_valid": True,
        "raw_min": math.inf,
        "raw_max": -math.inf,
        "raw_sum": torch.zeros(40_000, dtype=torch.float64),
        "raw_square_sum": torch.zeros(40_000, dtype=torch.float64),
        "per_image_variance_sum": 0.0,
    }


def _record_output(
    accumulator: dict[str, Any],
    raw: torch.Tensor,
    target_slm: torch.Tensor,
    target_images: torch.Tensor,
    labels: torch.Tensor,
    provider: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
    preflight: Any,
) -> None:
    _require(tuple(raw.shape) == tuple(target_slm.shape) and raw.dtype == torch.float32, "raw/target optical shape or dtype changed")
    _finite(raw, "raw CMOS")
    _finite(target_slm, "target SLM intensity")
    batch_size = int(raw.shape[0])
    image_loss = (raw - target_slm).square().mean()
    view = preflight._prepare_evaluator_view(raw)
    target_view = preflight._prepare_evaluator_view(target_images)
    candidate_embedding, frozen_logits = provider(view)
    target_embedding, _ = provider(target_view)
    candidate_feature = F.normalize(candidate_embedding, p=2.0, dim=1, eps=1.0e-12)
    target_feature = F.normalize(target_embedding, p=2.0, dim=1, eps=1.0e-12)
    per_sample_feature = (candidate_feature - target_feature).square().sum(dim=1)
    _, smooth_logits = smooth(view)
    _finite(image_loss, "image loss")
    _finite(per_sample_feature, "Frozen 128D feature consistency distance")
    _finite(frozen_logits, "Frozen evaluator logits")
    _finite(smooth_logits, "Smooth evaluator logits")
    raw_cpu = raw.detach().cpu().to(torch.float64).flatten(1)
    accumulator["count"] += batch_size
    accumulator["image_loss_sum"] += float(image_loss.detach().cpu()) * batch_size
    accumulator["feature_loss_sum"] += float(per_sample_feature.mean().detach().cpu()) * batch_size
    accumulator["feature_distances"].append(per_sample_feature.detach().cpu().to(torch.float64))
    accumulator["frozen_correct"].append((frozen_logits.argmax(dim=1).detach().cpu() == labels.cpu()).contiguous())
    accumulator["smooth_correct"].append((smooth_logits.argmax(dim=1).detach().cpu() == labels.cpu()).contiguous())
    accumulator["raw_finite"] = accumulator["raw_finite"] and bool(torch.isfinite(raw).all())
    accumulator["raw_structural_valid"] = accumulator["raw_structural_valid"] and tuple(raw.shape[1:]) == (1, 200, 200) and raw.dtype == torch.float32
    accumulator["raw_min"] = min(accumulator["raw_min"], float(raw.min().detach().cpu()))
    accumulator["raw_max"] = max(accumulator["raw_max"], float(raw.max().detach().cpu()))
    accumulator["raw_sum"] += raw_cpu.sum(dim=0)
    accumulator["raw_square_sum"] += raw_cpu.square().sum(dim=0)
    accumulator["per_image_variance_sum"] += float(raw_cpu.var(dim=1, unbiased=False).sum())


def _recall_summary(correct: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    values: list[float] = []
    for digit in range(10):
        mask = labels == digit
        _require(int(mask.sum()) == 100, f"DEV class {digit} count changed")
        values.append(float(correct[mask].to(torch.float64).mean()))
    return {"macro_recall": float(sum(values) / 10.0), "per_class_recall": values}


def _finalize_accumulator(accumulator: dict[str, Any], labels: torch.Tensor) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    count = int(accumulator["count"])
    _require(count == 1000, f"fixed DEV measurement count changed: {count}")
    feature_distances = torch.cat(accumulator["feature_distances"], dim=0).to(torch.float64)
    frozen_correct = torch.cat(accumulator["frozen_correct"], dim=0).to(torch.bool)
    smooth_correct = torch.cat(accumulator["smooth_correct"], dim=0).to(torch.bool)
    raw_sum = accumulator["raw_sum"]
    raw_square_sum = accumulator["raw_square_sum"]
    raw_mean_per_pixel = raw_sum / count
    raw_variance_per_pixel = (raw_square_sum / count - raw_mean_per_pixel.square()).clamp_min(0.0)
    per_image_variance_mean = accumulator["per_image_variance_sum"] / count
    across_sample_variance_mean = float(raw_variance_per_pixel.mean())
    raw_mean = float(raw_mean_per_pixel.mean())
    raw_std = float((raw_square_sum.sum() / (count * 40_000) - raw_mean**2).clamp_min(0.0).sqrt())
    structural_valid = bool(accumulator["raw_structural_valid"])
    nonconstant = per_image_variance_mean > 0.0 and across_sample_variance_mean > 0.0
    public = {
        "sample_count": count,
        "losses": {"L_image": accumulator["image_loss_sum"] / count, "L_feature": accumulator["feature_loss_sum"] / count},
        "frozen_evaluator": _recall_summary(frozen_correct, labels),
        "smooth_evaluator": _recall_summary(smooth_correct, labels),
        "frozen_feature_consistency": {"representation": "L2-normalized FrozenMNISTEvaluator embedding in R^128", "mean_squared_euclidean_distance": float(feature_distances.mean()), "per_sample_distance_tensor_sha256": tensor_sha256(feature_distances)},
        "raw_cmos": {"shape": [count, 1, 200, 200], "dtype": "torch.float32", "finite": bool(accumulator["raw_finite"]), "structural_valid": structural_valid, "nonconstant": nonconstant, "mean": raw_mean, "std": raw_std, "min": accumulator["raw_min"], "max": accumulator["raw_max"], "per_image_variance_mean": per_image_variance_mean, "across_sample_variance_mean": across_sample_variance_mean},
    }
    public["raw_cmos"]["valid_for_diagnostic"] = bool(public["raw_cmos"]["finite"] and structural_valid and nonconstant)
    return public, {"frozen_correct": frozen_correct, "smooth_correct": smooth_correct, "feature_distance": feature_distances}


def _measure_direct(
    consumer: Any,
    images: torch.Tensor,
    labels: torch.Tensor,
    provider: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
    preflight: Any,
    expected_panel: torch.Tensor,
    arm: str,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    before_state = _cpu_tree(consumer.runtime.state_dict())
    accumulator = _new_accumulator()
    z_parts: list[torch.Tensor] = []
    panel_max_abs = 0.0
    panel_allclose = True
    with torch.no_grad():
        for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
            batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            batch_labels = labels[start : start + LOGICAL_BATCH_SIZE]
            result = consumer.runtime.forward_from_image(batch)
            _check_direct_result(consumer, result, int(batch.shape[0]), preflight)
            z_parts.append(result.z_ols.coefficients.detach().cpu().contiguous())
            _record_output(accumulator, result.raw_intensity, result.image_input.slm_intensity, batch, batch_labels, provider, smooth, preflight)
            raw_cpu = result.raw_intensity.detach().cpu()
            difference = (raw_cpu - expected_panel[start : start + LOGICAL_BATCH_SIZE]).abs()
            panel_max_abs = max(panel_max_abs, float(difference.max()))
            panel_allclose = panel_allclose and bool(torch.allclose(raw_cpu, expected_panel[start : start + LOGICAL_BATCH_SIZE], rtol=PANEL_RTOL, atol=PANEL_ATOL))
            del batch, result
    after_state = _cpu_tree(consumer.runtime.state_dict())
    _require(_tree_equal(before_state, after_state), f"{arm} direct forward changed runtime state")
    _require(panel_allclose, f"{arm} direct forward no longer reproduces the immutable V3 endpoint panel")
    public, tensors = _finalize_accumulator(accumulator, labels)
    return torch.cat(z_parts, dim=0), public, tensors, {"panel_allclose": panel_allclose, "panel_max_abs_difference": panel_max_abs, "runtime_state_unchanged": True}


def _measure_cross_generator(
    consumer: Any,
    z_cpu: torch.Tensor,
    images: torch.Tensor,
    labels: torch.Tensor,
    provider: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
    preflight: Any,
    encoder: str,
    generator: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, Any]]:
    before_state = _cpu_tree(consumer.runtime.state_dict())
    accumulator = _new_accumulator()
    with torch.no_grad():
        for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
            batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            batch_labels = labels[start : start + LOGICAL_BATCH_SIZE]
            z_batch = z_cpu[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            result = consumer.runtime.forward_from_ols(z_batch)
            _check_external_result(consumer, result, int(batch.shape[0]))
            target_slm = consumer.runtime.input_adapter(batch).slm_intensity
            _record_output(accumulator, result.raw_intensity, target_slm, batch, batch_labels, provider, smooth, preflight)
            del batch, batch_labels, z_batch, result, target_slm
    after_state = _cpu_tree(consumer.runtime.state_dict())
    _require(_tree_equal(before_state, after_state), f"cross-swap {encoder}+{generator} changed runtime state")
    public, tensors = _finalize_accumulator(accumulator, labels)
    return public, tensors, {"runtime_state_unchanged": True, "encoder": encoder, "generator": generator, "route": "forward_from_ols with encoder-produced full complex z"}


def _bootstrap_deltas(control: Mapping[str, torch.Tensor], candidate: Mapping[str, torch.Tensor], labels: torch.Tensor, seed: int) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    deltas = {"frozen_macro_recall": [], "smooth_macro_recall": [], "feature_distance": []}
    per_class = {name: [] for name in deltas}
    for digit in range(10):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == 100, f"bootstrap class {digit} count changed")
        draw = torch.randint(0, matches.numel(), (BOOTSTRAP_REPLICATES, matches.numel()), generator=generator)
        for name in deltas:
            control_values = control[name].index_select(0, matches)[draw]
            candidate_values = candidate[name].index_select(0, matches)[draw]
            delta = (candidate_values.to(torch.float64) - control_values.to(torch.float64)).mean(dim=1)
            deltas[name].append(delta)
            per_class[name].append(float(candidate[name].index_select(0, matches).to(torch.float64).mean() - control[name].index_select(0, matches).to(torch.float64).mean()))
    result: dict[str, Any] = {"type": "paired, class-stratified bootstrap over the same 100 DEV examples per class", "replicates": BOOTSTRAP_REPLICATES, "seed": seed, "quantile_convention": "torch.quantile interpolation=linear", "metrics": {}, "per_class_delta": per_class}
    for name in deltas:
        tensor = torch.stack(deltas[name], dim=0).mean(dim=0)
        quantiles = torch.quantile(tensor, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
        result["metrics"][name] = {"delta_mean": float(tensor.mean()), "ci_lower": float(quantiles[0]), "ci_upper": float(quantiles[1]), "bootstrap_delta_tensor_sha256": tensor_sha256(tensor)}
    return result


def _geometry(z: torch.Tensor, labels: torch.Tensor, arm: str) -> dict[str, Any]:
    z = z.detach().cpu().to(torch.complex64)
    _require(tuple(z.shape) == (1000, 100), f"{arm} complex z shape changed")
    _finite(z.real, f"{arm} z real")
    _finite(z.imag, f"{arm} z imaginary")
    representation = gauge_source.gauge_invariant_ols_representation(z).detach().cpu().to(torch.float64)
    _finite(representation, f"{arm} gauge representation")
    centered = representation - representation.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1).mm(centered) / float(max(representation.shape[0] - 1, 1))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    maximum = float(eigenvalues.max()) if eigenvalues.numel() else 0.0
    denominator = float(eigenvalues.square().sum())
    effective_rank = float(eigenvalues.sum().square() / denominator) if denominator > 0.0 else 0.0
    numerical_rank = int((eigenvalues > maximum * GAUGE_NUMERICAL_RANK_SCALE).sum()) if maximum > 0.0 else 0
    centroids: list[torch.Tensor] = []
    within_values: list[torch.Tensor] = []
    for digit in range(10):
        group = representation[labels == digit]
        _require(group.shape[0] == 100, f"{arm} geometry class {digit} count changed")
        centroid = group.mean(dim=0)
        centroids.append(centroid)
        within_values.append((group - centroid).square().mean(dim=1).sqrt())
    centroid_tensor = torch.stack(centroids)
    distance_matrix = (centroid_tensor[:, None, :] - centroid_tensor[None, :, :]).square().mean(dim=2).sqrt()
    upper = distance_matrix[torch.triu(torch.ones_like(distance_matrix, dtype=torch.bool), diagonal=1)]
    within = torch.cat(within_values)
    between_mean = float(upper.mean())
    within_mean = float(within.mean())
    _require(torch.isfinite(distance_matrix).all() and torch.isfinite(within).all(), f"{arm} geometry is nonfinite")
    return {
        "arm": arm,
        "authoritative_latent": "full complex z in C^100 from the real image encoder and mandatory OLS modal projection",
        "representation": "S(z)=gauge_invariant_ols_representation(z) in R^200, using the project helper",
        "gauge": {"epsilon": 1.0e-6, "sigma": float(gauge_source.PRIOR_SIGMA), "anchor": "z.sum(dim=1)/sqrt(100)", "phase_convention": "anchor.conj()/sqrt(|anchor|^2+epsilon^2)"},
        "distance_definition": "RMS over the 200 gauge-representation coordinates, matching the existing warmup class-separation diagnostic",
        "sample_count": 1000,
        "representation_tensor_sha256": tensor_sha256(representation),
        "covariance_spectrum_gauge_real_200d": {"effective_rank_participation_ratio": effective_rank, "numerical_rank_threshold": "1e-6 * maximum eigenvalue", "numerical_rank": numerical_rank, "maximum_eigenvalue": maximum, "eigenvalue_sum": float(eigenvalues.sum())},
        "within_class_dispersion_rms": {"mean": within_mean, "std": float(within.std(unbiased=False)), "min": float(within.min()), "max": float(within.max())},
        "between_class_centroid_distance_rms": {"mean": between_mean, "std": float(upper.std(unbiased=False)), "min": float(upper.min()), "max": float(upper.max())},
        "between_to_within_mean_ratio": between_mean / within_mean if within_mean > 0.0 else None,
        "centroid_distance_matrix": distance_matrix.tolist(),
    }


def _extract_z(consumer: Any, images: torch.Tensor, preflight: Any, role: str) -> torch.Tensor:
    before_state = _cpu_tree(consumer.runtime.state_dict())
    parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
            batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            result = consumer.runtime.forward_from_image(batch)
            _check_direct_result(consumer, result, int(batch.shape[0]), preflight)
            parts.append(result.z_ols.coefficients.detach().cpu().contiguous())
            del batch, result
    after_state = _cpu_tree(consumer.runtime.state_dict())
    _require(_tree_equal(before_state, after_state), f"{role} TRAIN centroid forward changed runtime state")
    result = torch.cat(parts, dim=0)
    _require(tuple(result.shape) == (1000, 100) and result.dtype == torch.complex64, f"{role} centroid z shape/dtype changed")
    return result


def _centroid_metrics(control_z: torch.Tensor, candidate_z: torch.Tensor, train_control_z: torch.Tensor, train_candidate_z: torch.Tensor, dev_labels: torch.Tensor, train_labels: torch.Tensor) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    def centroids(values: torch.Tensor) -> torch.Tensor:
        representation = gauge_source.gauge_invariant_ols_representation(values).detach().cpu().to(torch.float64)
        return torch.stack([representation[train_labels == digit].mean(dim=0) for digit in range(10)])

    control_centroids = centroids(train_control_z)
    candidate_centroids = centroids(train_candidate_z)
    control_rep = gauge_source.gauge_invariant_ols_representation(control_z).detach().cpu().to(torch.float64)
    candidate_rep = gauge_source.gauge_invariant_ols_representation(candidate_z).detach().cpu().to(torch.float64)

    def classify(rep: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        distance = (rep[:, None, :] - bank[None, :, :]).square().sum(dim=2)
        return distance.argmin(dim=1)

    control_correct = classify(control_rep, control_centroids) == dev_labels
    candidate_correct = classify(candidate_rep, candidate_centroids) == dev_labels
    control_summary = _recall_summary(control_correct, dev_labels)
    candidate_summary = _recall_summary(candidate_correct, dev_labels)
    bootstrap = _bootstrap_deltas(
        {"frozen_macro_recall": control_correct, "smooth_macro_recall": control_correct, "feature_distance": control_correct.to(torch.float64)},
        {"frozen_macro_recall": candidate_correct, "smooth_macro_recall": candidate_correct, "feature_distance": candidate_correct.to(torch.float64)},
        dev_labels,
        CENTROID_BOOTSTRAP_SEED,
    )
    # The generic bootstrap above is intentionally not reused for the report: it
    # is a boolean nearest-centroid metric, so expose only its macro-recall CI.
    centroid_ci = bootstrap["metrics"]["frozen_macro_recall"]
    public = {
        "role": "diagnostic-only TRAIN-centroid bank to fixed DEV nearest-centroid evaluation",
        "train_subset_rule": "within_train_class_source_index_ascending_first_100_v1",
        "train_rows_used": 1000,
        "train_rows_per_class": CENTROID_ROWS_PER_CLASS,
        "control": control_summary,
        "candidate": candidate_summary,
        "candidate_minus_control": {"macro_recall": candidate_summary["macro_recall"] - control_summary["macro_recall"], "per_class_recall": [candidate_summary["per_class_recall"][digit] - control_summary["per_class_recall"][digit] for digit in range(10)]},
        "paired_bootstrap": {"type": "paired, class-stratified nearest-centroid recall bootstrap", "replicates": BOOTSTRAP_REPLICATES, "seed": CENTROID_BOOTSTRAP_SEED, "quantile_convention": "torch.quantile interpolation=linear", "macro_delta": centroid_ci},
        "diagnostic_boundary": "not electronic decoder evidence, not generation evidence, not a Stage-A-to-B eligibility gate",
    }
    return public, {"control_correct": control_correct, "candidate_correct": candidate_correct}


def _delta(control: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frozen_macro_recall": candidate["frozen_evaluator"]["macro_recall"] - control["frozen_evaluator"]["macro_recall"],
        "smooth_macro_recall": candidate["smooth_evaluator"]["macro_recall"] - control["smooth_evaluator"]["macro_recall"],
        "frozen_feature_consistency_distance": candidate["frozen_feature_consistency"]["mean_squared_euclidean_distance"] - control["frozen_feature_consistency"]["mean_squared_euclidean_distance"],
        "image_loss": candidate["losses"]["L_image"] - control["losses"]["L_image"],
        "frozen_per_class_recall": [candidate["frozen_evaluator"]["per_class_recall"][digit] - control["frozen_evaluator"]["per_class_recall"][digit] for digit in range(10)],
        "smooth_per_class_recall": [candidate["smooth_evaluator"]["per_class_recall"][digit] - control["smooth_evaluator"]["per_class_recall"][digit] for digit in range(10)],
    }


def _decision(
    matched_delta: Mapping[str, Any],
    matched_bootstrap: Mapping[str, Any],
    centroid: Mapping[str, Any],
    geometry: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = matched_delta["frozen_macro_recall"] >= RECALL_MATERIAL_MARGIN and float(matched_bootstrap["metrics"]["frozen_macro_recall"]["ci_lower"]) > 0.0
    smooth = matched_delta["smooth_macro_recall"] >= RECALL_MATERIAL_MARGIN and float(matched_bootstrap["metrics"]["smooth_macro_recall"]["ci_lower"]) > 0.0
    feature_improvement = matched_delta["frozen_feature_consistency_distance"] < 0.0 and float(matched_bootstrap["metrics"]["feature_distance"]["ci_upper"]) < 0.0
    centroid_delta = float(centroid["candidate_minus_control"]["macro_recall"])
    centroid_gain = centroid_delta >= RECALL_MATERIAL_MARGIN and float(centroid["paired_bootstrap"]["macro_delta"]["ci_lower"]) > 0.0
    ratio_control = geometry["control"]["between_to_within_mean_ratio"]
    ratio_candidate = geometry["candidate"]["between_to_within_mean_ratio"]
    geometry_nonworse = ratio_control is not None and ratio_candidate is not None and float(ratio_candidate) >= float(ratio_control)
    latent_gain = centroid_gain and feature_improvement and geometry_nonworse
    candidate_containing_smooth_unsupported = all(
        not (float(matrix[key]["smooth_evaluator"]["macro_recall"]) - float(matrix["control+control"]["smooth_evaluator"]["macro_recall"]) >= RECALL_MATERIAL_MARGIN)
        for key in ("candidate+control", "control+candidate", "candidate+candidate")
    )
    rules = {
        "material_recall_margin": RECALL_MATERIAL_MARGIN,
        "frozen_provider_material_improvement": frozen,
        "independent_smooth_material_improvement": smooth,
        "frozen_feature_distance_improves_with_bootstrap_ci_upper_lt_zero": feature_improvement,
        "train_centroid_dev_material_improvement": centroid_gain,
        "gauge_geometry_between_within_ratio_nonworse": geometry_nonworse,
        "latent_semantic_gain_supported": latent_gain,
        "candidate_containing_cross_swap_smooth_not_materially_above_control": candidate_containing_smooth_unsupported,
        "stage_a_firewall_threshold": STAGE_A_FIREWALL,
        "stage_a_firewall_unchanged": True,
    }
    if frozen and not smooth and not latent_gain:
        classification = CLASSIFICATIONS["provider"]
        rationale = "Frozen-provider improvement does not transfer to the independent Smooth evaluator, and the fixed TRAIN-centroid/gauge-geometry diagnostic does not establish a matched latent semantic gain."
    elif latent_gain and not smooth and candidate_containing_smooth_unsupported:
        classification = CLASSIFICATIONS["generator"]
        rationale = "The fixed TRAIN-centroid DEV diagnostic, provider feature distance, and non-worsened gauge geometry support a latent semantic gain, while no candidate-containing encoder/generator path transfers a material gain to the independent raw-CMOS evaluator."
    elif not frozen and not latent_gain:
        classification = CLASSIFICATIONS["ineffective"]
        rationale = "Neither the matched Frozen provider nor the fixed TRAIN-centroid/gauge-geometry diagnostic establishes a material semantic gain."
    else:
        classification = CLASSIFICATIONS["mixed"]
        rationale = "The matched provider, independent evaluator, latent geometry, centroid diagnostic, and cross-swap evidence do not satisfy one unique attribution rule."
    return {"classification": classification, "rationale": rationale, "rules": rules, "secondary_unresolved": ["Independent evaluator transfer and provider alignment are not interchangeable evidence."], "no_stage_b_implication": True}


def _resolve_output_dir(root: Path, value: Path | None) -> Path:
    target = DEFAULT_OUTPUT_DIR if value is None else value
    target = target if target.is_absolute() else root / target
    target = target.resolve()
    outputs_root = (root / "outputs").resolve()
    _require(target.is_relative_to(outputs_root), "diagnosis output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite existing diagnosis output: {target}")
    return target


def _base_payload(root: Path, started: Mapping[str, Any], entrypoint_sha: str) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "outer_gate": OUTER_GATE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "INVALID",
        "status": "invalid_harness",
        "classification": "INVALID_V3_FAILURE_ATTRIBUTION_NOT_COMPLETED",
        "entrypoint": {"path": _relative(root, Path(__file__)), "sha256": entrypoint_sha},
        "provenance": {"git_start": dict(started)},
        "claim_boundary": {
            "optimizer_created": False,
            "optimizer_steps": 0,
            "backward_executed": False,
            "optical_parameter_updates": 0,
            "scientific_retraining": False,
            "v3_retry": False,
            "stage_b_executed": False,
            "scale_extraction_executed": False,
            "prior_generation_executed": False,
            "generation_executed": False,
            "protected_accessed": False,
            "test_accessed": False,
            "sealed_accessed": False,
            "training_authorized": False,
            "stage_b_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "next": None,
    }


def run(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    started = v2.git_facts(root)
    entrypoint_sha = file_sha256(Path(__file__).resolve())
    target = _resolve_output_dir(root, output_dir)
    target.mkdir(parents=True, exist_ok=False)
    payload = _base_payload(root, started, entrypoint_sha)
    input_ref: dict[str, Any] | None = None
    scientific_started = False
    try:
        _require(started["branch"] == EXPECTED_BRANCH and started["head"] == EXPECTED_HEAD, f"active Git state differs from frozen attribution authority: {started['branch']}@{started['head']}")
        preflight = _load_preflight_module(root)
        v2_contract, authorities, authority_extension = _validate_scientific_authority(root, preflight)
        device, device_facts = preflight._cuda_device_facts()
        _require(device.type == "cuda" and torch.cuda.get_device_properties(device).major == 8 and torch.cuda.get_device_properties(device).minor == 6, "target CUDA device differs from the accepted V3 environment")
        images, labels, train_indices, development_indices, selection_indices, train_centroid_indices, dataset = _load_roles(root, preflight)
        selection_images = images.index_select(0, selection_indices).contiguous()
        selection_labels = labels.index_select(0, selection_indices).contiguous()
        centroid_images = images.index_select(0, train_centroid_indices).contiguous()
        centroid_labels = labels.index_select(0, train_centroid_indices).contiguous()
        _require(torch.equal(torch.bincount(centroid_labels, minlength=10), torch.full((10,), 100, dtype=torch.long)), "fixed TRAIN centroid labels changed")

        control_checkpoint, control_state = _load_checkpoint(root, CONTROL_CHECKPOINT_PATH, CONTROL_CHECKPOINT_SHA256, "control", v2_contract)
        candidate_checkpoint, candidate_state = _load_checkpoint(root, CANDIDATE_CHECKPOINT_PATH, CANDIDATE_CHECKPOINT_SHA256, "candidate", v2_contract)
        _require(torch.equal(control_state["ols.operator.input_modes"], candidate_state["ols.operator.input_modes"]), "control/candidate OLS input modes differ")
        _require(torch.equal(control_state["ols.operator.output_modes"], candidate_state["ols.operator.output_modes"]), "control/candidate OLS output modes differ")
        _require(torch.equal(control_state["ols.operator.transport"], candidate_state["ols.operator.transport"]), "control/candidate OLS transport differs")

        control_panel, control_panel_ref = _load_panel(root, CONTROL_PANEL_PATH, CONTROL_PANEL_SHA256, "control", v2_contract)
        candidate_panel, candidate_panel_ref = _load_panel(root, CANDIDATE_PANEL_PATH, CANDIDATE_PANEL_SHA256, "candidate", v2_contract)
        frozen, smooth = _load_evaluators(root, authorities, device)
        evaluator_state_before = {"frozen": _cpu_tree(frozen.state_dict()), "smooth": _cpu_tree(smooth.state_dict())}
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in frozen.parameters()), "Frozen evaluator firewall failed")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in smooth.parameters()), "Smooth evaluator firewall failed")

        input_payload = {
            "schema": INPUT_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "entrypoint": {"path": _relative(root, Path(__file__)), "sha256": entrypoint_sha},
            "frozen_contracts": {"v3_contract": {"path": V3_CONTRACT_PATH.as_posix(), "sha256": V3_CONTRACT_SHA256}, "v2_contract": {"path": v2.V2_CONTRACT_RELATIVE_PATH.as_posix(), "sha256": V2_CONTRACT_SHA256}},
            "scientific_failure_authority": authority_extension["v3_scientific"],
            "protocol": {
                "consumer": "immutable V3 step-300 control/candidate runtime states; forward-only",
                "matched_dev": {"rows": 1000, "selection_indices_sha256": DEV_SELECTION_SHA256, "selection_labels_sha256": tensor_sha256(selection_labels)},
                "train_centroid_subset": {"rule": "within_train_class_source_index_ascending_first_100_v1", "rows": 1000, "rows_per_class": CENTROID_ROWS_PER_CLASS, "indices_sha256": tensor_sha256(train_centroid_indices), "labels_sha256": tensor_sha256(centroid_labels)},
                "cross_swaps": ["E_control+G_control", "E_candidate+G_candidate", "E_candidate+G_control", "E_control+G_candidate"],
                "gauge": {"helper": GAUGE_HELPER_PATH.as_posix(), "helper_sha256": GAUGE_HELPER_SHA256, "representation": "full complex z -> global-phase quotient S(z) in R^200", "epsilon": 1.0e-6, "sigma": float(gauge_source.PRIOR_SIGMA)},
                "bootstrap": {"matched_metrics_seed": BOOTSTRAP_SEED, "centroid_seed": CENTROID_BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPLICATES},
                "decision_rule_frozen_before_measurement": {"material_recall_margin": RECALL_MATERIAL_MARGIN, "feature_improvement_requires_ci_upper_lt_zero": True, "latent_gain_requires_centroid_gain_feature_gain_and_nonworse_geometry": True, "stage_a_firewall": STAGE_A_FIREWALL},
            },
            "data_roles": dataset,
            "claim_boundary": payload["claim_boundary"],
            "input_self_hash": "not_embedded_to_avoid_self_reference",
        }
        input_path = target / INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V3 failure-attribution input")
        input_ref = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        scientific_started = True

        control_consumer = _new_consumer(v2_contract, device, preflight)
        candidate_consumer = _new_consumer(v2_contract, device, preflight)
        _load_state(control_consumer, control_state, preflight)
        _load_state(candidate_consumer, candidate_state, preflight)
        control_z, control_metrics, control_tensors, control_reproduction = _measure_direct(control_consumer, selection_images, selection_labels, frozen, smooth, preflight, control_panel, "control")
        candidate_z, candidate_metrics, candidate_tensors, candidate_reproduction = _measure_direct(candidate_consumer, selection_images, selection_labels, frozen, smooth, preflight, candidate_panel, "candidate")

        matched = {"control": control_metrics, "candidate": candidate_metrics, "candidate_minus_control": _delta(control_metrics, candidate_metrics)}
        matched_bootstrap = _bootstrap_deltas(
            {"frozen_macro_recall": control_tensors["frozen_correct"].to(torch.float64), "smooth_macro_recall": control_tensors["smooth_correct"].to(torch.float64), "feature_distance": control_tensors["feature_distance"]},
            {"frozen_macro_recall": candidate_tensors["frozen_correct"].to(torch.float64), "smooth_macro_recall": candidate_tensors["smooth_correct"].to(torch.float64), "feature_distance": candidate_tensors["feature_distance"]},
            selection_labels,
            BOOTSTRAP_SEED,
        )

        cross_consumers: dict[str, Any] = {}
        matrix: dict[str, Any] = {
            "control+control": {**control_metrics, "measurement_route": "forward_from_image with matched control checkpoint"},
            "candidate+candidate": {**candidate_metrics, "measurement_route": "forward_from_image with matched candidate checkpoint"},
        }
        matrix_tensors: dict[str, dict[str, torch.Tensor]] = {"control+control": control_tensors, "candidate+candidate": candidate_tensors}
        for encoder, generator in (("candidate", "control"), ("control", "candidate")):
            key = f"{encoder}+{generator}"
            consumer = _new_consumer(v2_contract, device, preflight)
            _load_state(consumer, _combined_state(control_state, candidate_state, encoder, generator), preflight)
            cross_consumers[key] = consumer
            matrix[key], matrix_tensors[key], matrix_route = _measure_cross_generator(consumer, candidate_z if encoder == "candidate" else control_z, selection_images, selection_labels, frozen, smooth, preflight, encoder, generator)
            matrix[key]["measurement_route"] = matrix_route

        control_geometry = _geometry(control_z, selection_labels, "control")
        candidate_geometry = _geometry(candidate_z, selection_labels, "candidate")
        geometry = {"control": control_geometry, "candidate": candidate_geometry, "candidate_minus_control": {"effective_rank_participation_ratio": candidate_geometry["covariance_spectrum_gauge_real_200d"]["effective_rank_participation_ratio"] - control_geometry["covariance_spectrum_gauge_real_200d"]["effective_rank_participation_ratio"], "numerical_rank": candidate_geometry["covariance_spectrum_gauge_real_200d"]["numerical_rank"] - control_geometry["covariance_spectrum_gauge_real_200d"]["numerical_rank"], "between_to_within_mean_ratio": candidate_geometry["between_to_within_mean_ratio"] - control_geometry["between_to_within_mean_ratio"]}}

        train_control_z = _extract_z(control_consumer, centroid_images, preflight, "control")
        train_candidate_z = _extract_z(candidate_consumer, centroid_images, preflight, "candidate")
        centroid_metrics, centroid_tensors = _centroid_metrics(control_z, candidate_z, train_control_z, train_candidate_z, selection_labels, centroid_labels)

        _require(_tree_equal(evaluator_state_before["frozen"], _cpu_tree(frozen.state_dict())), "Frozen evaluator state changed")
        _require(_tree_equal(evaluator_state_before["smooth"], _cpu_tree(smooth.state_dict())), "Smooth evaluator state changed")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in frozen.parameters()), "Frozen evaluator firewall changed")
        _require(all(parameter.grad is None and not parameter.requires_grad for parameter in smooth.parameters()), "Smooth evaluator firewall changed")

        decision = _decision(matched["candidate_minus_control"], matched_bootstrap, centroid_metrics, geometry, matrix)
        end = v2.git_facts(root)
        report = {
            **payload,
            "result": "VALID_COMPLETED",
            "status": "valid_completed",
            "classification": decision["classification"],
            "provenance": {"git_start": started, "git_end_before_document_sync": end},
            "authorities": {**authorities, **authority_extension, "control_checkpoint": {"path": CONTROL_CHECKPOINT_PATH.as_posix(), "sha256": CONTROL_CHECKPOINT_SHA256, "step": 300}, "candidate_checkpoint": {"path": CANDIDATE_CHECKPOINT_PATH.as_posix(), "sha256": CANDIDATE_CHECKPOINT_SHA256, "step": 300}, "control_panel": control_panel_ref, "candidate_panel": candidate_panel_ref},
            "device": device_facts,
            "data_roles": dataset,
            "input": input_ref,
            "control_candidate": {"matched_dev": matched, "paired_bootstrap": matched_bootstrap, "direct_panel_reproduction": {"control": control_reproduction, "candidate": candidate_reproduction}},
            "cross_swap_matrix": {"same_modal_projection": True, "transport": "T=I_100_coefficient_space", "shared_W_OLS": True, "all_paths": matrix},
            "complex_ols_semantic_geometry": geometry,
            "train_centroid_to_dev": centroid_metrics,
            "attribution_decision": decision,
            "stage_a_firewall": {"threshold": STAGE_A_FIREWALL, "metric": "Frozen evaluator macro recall on the fixed 1000-row DEV selection", "control_macro_recall": control_metrics["frozen_evaluator"]["macro_recall"], "candidate_macro_recall": candidate_metrics["frozen_evaluator"]["macro_recall"], "candidate_passed": candidate_metrics["frozen_evaluator"]["macro_recall"] >= STAGE_A_FIREWALL, "stage_b_authorized": False},
            "no_update_audit": {"optimizer_created": False, "optimizer_steps": 0, "backward_executed": False, "optical_parameter_updates": 0, "runtime_state_unchanged_after_all_forwards": True, "checkpoint_payload_optimizer_state_loaded": False, "checkpoint_payload_optimizer_state_present": {"control": "optimizer_state" in control_checkpoint, "candidate": "optimizer_state" in candidate_checkpoint}, "v3_retry": False},
            "claim_boundary": {**payload["claim_boundary"], "diagnosis_executed": True, "stage_b_authorized": False},
            "next": "No automatic training, retry, Stage B, scale extraction, prior generation, generation, or promotion follows this diagnosis.",
        }
        report_path = target / REPORT_NAME
        atomic_write_new_json(report_path, report, artifact_kind="V3 failure-attribution report")
        report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
        lineage = {
            "schema": LINEAGE_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "result": "VALID_COMPLETED",
            "status": "valid_completed",
            "classification": decision["classification"],
            "git_start": started,
            "git_end_before_document_sync": end,
            "input": input_ref,
            "report": report_ref,
            "authorities": report["authorities"],
            "measurement_order": ["authority validation", "input freeze", "matched DEV forward-only comparison", "2x2 cross swaps", "complex-OLS geometry", "fixed TRAIN-centroid -> DEV diagnostic", "classification"],
            "read_counts": {"canonical_source_rows_loaded": 60_000, "train_centroid_rows_used": 1_000, "development_selection_rows_used": 1_000, "calibration_rows_read": 0, "protected_rows_read": 0, "test_rows_read": 0, "sealed_rows_read": 0},
            "no_update": report["no_update_audit"],
            "stage_b_authorized": False,
            "generation_ready": False,
            "formal_promotion": False,
        }
        lineage_path = target / LINEAGE_NAME
        atomic_write_new_json(lineage_path, lineage, artifact_kind="V3 failure-attribution lineage")
        report["lineage"] = {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)}
        print(json.dumps({"result": report["result"], "classification": report["classification"], "report": report_ref, "lineage": report["lineage"]}, ensure_ascii=False), flush=True)
        return report
    except DiagnosisBlocked as error:
        payload.update({"result": "BLOCKED", "status": "blocked", "classification": error.classification, "blocked_at": error.stage, "message": str(error), "details": error.details, "provenance": {"git_start": started, "git_end_before_document_sync": v2.git_facts(root)}, "input": input_ref, "scientific_execution_started": scientific_started})
        report_path = target / REPORT_NAME
        atomic_write_new_json(report_path, payload, artifact_kind="blocked V3 failure-attribution report")
        return payload
    except InvalidDiagnosis as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V3_FAILURE_ATTRIBUTION_NOT_COMPLETED", "message": str(error), "provenance": {"git_start": started, "git_end_before_document_sync": v2.git_facts(root)}, "input": input_ref, "scientific_execution_started": scientific_started})
        report_path = target / REPORT_NAME
        atomic_write_new_json(report_path, payload, artifact_kind="invalid V3 failure-attribution report")
        return payload


def _load_evaluators(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[FrozenMNISTEvaluator, SmoothMNISTEvaluator]:
    frozen_ref = _mapping(_mapping(authorities["frozen_evaluator"], "Frozen evaluator authority").get("checkpoint"), "Frozen evaluator checkpoint")
    frozen_payload = _mapping(safe_torch_load(_path(root, frozen_ref["path"]), expected_sha256=frozen_ref["sha256"], map_location="cpu"), "Frozen evaluator payload")
    _require(frozen_payload.get("architecture_kind") == "frozen_mnist_evaluator_v1" and float(frozen_payload.get("validation_accuracy", 0.0)) >= 0.985, "Frozen evaluator qualification changed")
    frozen = FrozenMNISTEvaluator()
    frozen.load_state_dict(_mapping(frozen_payload.get("state_dict"), "Frozen evaluator state"), strict=True)
    frozen.to(device).freeze()
    smooth_ref = _mapping(_mapping(authorities["smooth_evaluator"], "Smooth evaluator authority").get("checkpoint"), "Smooth evaluator checkpoint")
    smooth_payload = _mapping(safe_torch_load(_path(root, smooth_ref["path"]), expected_sha256=smooth_ref["sha256"], map_location="cpu"), "Smooth evaluator payload")
    _require(smooth_payload.get("architecture_kind") == "smooth_mnist_evaluator_v2", "Smooth evaluator architecture changed")
    smooth = SmoothMNISTEvaluator()
    smooth.load_state_dict(_mapping(smooth_payload.get("state_dict"), "Smooth evaluator state"), strict=True)
    smooth.to(device).freeze()
    return frozen, smooth


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    report = run(repo_root=args.repo_root, output_dir=args.output_dir)
    return 0 if report.get("result") == "VALID_COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
