"""Execute the explicitly authorized LightGen V2 scientific training run.

The implementation consumer intentionally has no long-training command.  This
entry point is the separate, bounded scientific executor for the already
authorized Stage-A -> train-only scale -> Stage-B protocol.  It does not
perform generation, held-out evaluation, promotion, or readiness updates.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

import ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 as v2
from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 import (
    CHECKPOINT_STEPS,
    COMPLETE,
    DeterministicTrainRoleSampler,
    EPSILON_SCALE,
    FrozenPhysicalOLSScale,
    LightGen2DComplexOLSTrainingConsumerV2,
    LOGICAL_BATCH_SIZE,
    SCALE_EXTRACT,
    SCALE_FREEZE,
    SCALE_SCHEMA,
    SEMANTIC_THRESHOLD,
    STAGE_A_CHECKPOINT_RULE,
    STAGE_A_FAILURE,
    STAGE_A_SEMANTIC_CHECK,
    STAGE_A_STEPS,
    STAGE_A_TRAIN,
    STAGE_B_CHECKPOINT_RULE,
    STAGE_B_FAILURE,
    STAGE_B_SEMANTIC_CHECK,
    STAGE_B_STEPS,
    STAGE_B_TRAIN,
    V2TrainingStateMachine,
    checkpoint_payload,
    development_macro_recall,
    extract_train_only_ols_scale,
    file_sha256,
    git_facts,
    load_checkpoint,
    load_scale_artifact,
    load_v2_training_contract,
    save_checkpoint,
    save_scale_artifact,
    select_stage_a_checkpoint,
    select_stage_b_checkpoint,
    tensor_sha256,
)
from ldm_project.v2.evidence import atomic_write_new_json
from ldm_project.v2.student.evaluator import load_frozen_evaluator, load_local_mnist, stratified_mnist_indices


TASK_ID = "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2"
SUCCESS_CLASSIFICATION = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2_VALID_COMPLETED"
REPORT_SCHEMA = "lightgen_2d_semantic_complex_ols_training_v2_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_semantic_complex_ols_training_v2_lineage.v1"
STAGE_A_FAILURE_CLASSIFICATION = STAGE_A_FAILURE
STAGE_B_FAILURE_CLASSIFICATION = STAGE_B_FAILURE

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
ORIGINAL_CONTRACT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_contract_v2_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2.json"
)
ORIGINAL_CONTRACT_SHA256 = "b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53"
TRAINING_DATA_RELATIVE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_DATA_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
PROTECTED_TEST_RELATIVE_PATH = Path("data/MNIST/processed/test.pt")
PROTECTED_TEST_SHA256 = "8c089c2f0a5db35de7961f4c60ed6b627a05141245d44f542eeee57b09b539e1"
CONSUMER_RELATIVE_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen_2d_semantic_complex_ols_training_v2.py"
)
CONSUMER_SHA256 = "c659b93e0138531543ddfbe974c6dd4231a70b573a0701555b0dcd558a673ca3"
ALIGNMENT_RUNTIME_RELATIVE_PATH = Path(
    "src/ldm_project/v2/application/research/"
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

PREFLIGHT_REPORT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_report.v1.json"
)
PREFLIGHT_REPORT_SHA256 = "b7bc2482d55af70ad8fc7a1b204c636425e9bca64448feda9e11e0111b5d64a4"
PREFLIGHT_INPUT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_input.v1.json"
)
PREFLIGHT_INPUT_SHA256 = "2e090926b057f70ed9cacc63473d16786336aa39b9371da2dcfdbccfead94b1d"
PREFLIGHT_LINEAGE_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_lineage.v1.json"
)
PREFLIGHT_LINEAGE_SHA256 = "30913cd07ad318fbd94d2e5a4a346da6ca8390087e47a0b145d2c08b54b2d3bd"
PREFLIGHT_SCALE_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_scale.v2.pt"
)
PREFLIGHT_SCALE_SHA256 = "bb83ecbb02bf54e31967eff7056c9ad4b8213319816c97ee7be5a21bfef3ec12"
PREFLIGHT_STAGE_A_CHECKPOINT_SHA256 = "7c8dd3f65ab598fa56d20dd449a1ced0672464c776726255a888c5ee403bcea1"
PREFLIGHT_STAGE_B_CHECKPOINT_SHA256 = "811faefa39e0e4bb90b63c12afabc69215c9fb239b1dad70a3da3144be263d0f"

PRODUCTION_PROPAGATION_GRID = 2048
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_MIN_VRAM_BYTES = 6_441_926_656
DEVELOPMENT_SELECTION_SEED = 4321
INITIALIZATION_SEED = 1234
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_v2_attempt_01"
)
REPORT_NAME = "lightgen_2d_semantic_complex_ols_training_v2_report.v1.json"
LINEAGE_NAME = "lightgen_2d_semantic_complex_ols_training_v2_lineage.v1.json"
FORMAL_SCALE_NAME = "lightgen_2d_semantic_complex_ols_training_v2_formal_scale.v2.pt"


class ExecutionBlocked(RuntimeError):
    """A valid external condition prevents this bounded execution."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


class ScientificGateFailure(RuntimeError):
    """A complete scientific measurement failed its declared semantic gate."""

    def __init__(self, classification: str, stage: str, **details: Any) -> None:
        super().__init__(classification)
        self.classification = classification
        self.stage = stage
        self.details = details


class InvalidExecution(RuntimeError):
    """The execution harness or an authority input is malformed."""


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: Path) -> Path:
    target = value if value.is_absolute() else root / value
    return target.resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidExecution(f"{name} must be a mapping")
    return value


def _expect(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise ExecutionBlocked("BLOCKED_EXECUTION_AUTHORITY_MISMATCH", "authority", message, expected=expected, actual=actual)


def _read_hashed_json(root: Path, relative_path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    path = _resolve(root, relative_path)
    if not path.is_file():
        raise ExecutionBlocked("BLOCKED_EXECUTION_AUTHORITY_UNAVAILABLE", "authority", f"{name} is unavailable", path=relative_path.as_posix())
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ExecutionBlocked(
            "BLOCKED_EXECUTION_AUTHORITY_MISMATCH",
            "authority",
            f"{name} SHA mismatch",
            path=relative_path.as_posix(),
            expected_sha256=expected_sha256,
            actual_sha256=actual,
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidExecution(f"{name} is not valid JSON") from error
    return dict(_mapping(value, name))


def _identity(relative_path: Path, sha256: str, *, schema: str | None = None) -> dict[str, str]:
    value = {"path": relative_path.as_posix(), "sha256": sha256}
    if schema is not None:
        value["schema"] = schema
    return value


def _validate_preflight_authority(root: Path) -> tuple[v2.FrozenV2TrainingContract, dict[str, Any]]:
    contract = load_v2_training_contract(repo_root=root)
    _expect(contract.sha256, CONTRACT_SHA256, "corrected V2 training contract identity changed")
    _expect(contract.path.resolve(), _resolve(root, CONTRACT_RELATIVE_PATH), "corrected V2 training contract path changed")

    correction = _read_hashed_json(root, CORRECTION_AUTHORITY_RELATIVE_PATH, CORRECTION_AUTHORITY_SHA256, "V2 authority correction")
    _expect(correction.get("result"), "CONFIRMED", "V2 authority correction result changed")
    _expect(correction.get("classification"), CORRECTION_CLASSIFICATION, "V2 authority correction classification changed")
    original = _mapping(correction.get("original_v2_contract"), "original V2 contract correction identity")
    _expect(original.get("path"), ORIGINAL_CONTRACT_RELATIVE_PATH.as_posix(), "original V2 contract path changed")
    _expect(original.get("sha256"), ORIGINAL_CONTRACT_SHA256, "original V2 contract digest changed")
    _expect(original.get("character_count"), 63, "malformed source authority character count changed")
    corrected = _mapping(correction.get("corrected_contract"), "corrected V2 contract correction identity")
    _expect(corrected.get("path"), CONTRACT_RELATIVE_PATH.as_posix(), "corrected contract correction path changed")
    _expect(corrected.get("sha256"), CONTRACT_SHA256, "corrected contract correction digest changed")
    current_source = _mapping(correction.get("current_source"), "canonical source correction identity")
    _expect(current_source.get("path"), TRAINING_DATA_RELATIVE_PATH.as_posix(), "canonical training source path changed")
    _expect(current_source.get("sha256"), TRAINING_DATA_SHA256, "canonical training source digest changed")
    _expect(current_source.get("unchanged_before_after"), True, "canonical training source change boundary is not closed")
    boundary = _mapping(correction.get("correction_boundary"), "correction boundary")
    for key in (
        "cuda_executed",
        "scientific_training_executed",
        "generation_executed",
        "protected_test_read",
        "sealed_read",
        "training_authorized",
        "formal_promotion",
        "original_v2_contract_overwritten",
        "training_source_modified",
        "scientific_semantics_changed",
    ):
        _expect(boundary.get(key), False, f"correction boundary opened: {key}")
    _expect(boundary.get("generation_readiness"), "not_ready", "correction generation boundary changed")

    preflight_input = _read_hashed_json(root, PREFLIGHT_INPUT_RELATIVE_PATH, PREFLIGHT_INPUT_SHA256, "attempt-07 preflight input")
    preflight_report = _read_hashed_json(root, PREFLIGHT_REPORT_RELATIVE_PATH, PREFLIGHT_REPORT_SHA256, "attempt-07 preflight report")
    preflight_lineage = _read_hashed_json(root, PREFLIGHT_LINEAGE_RELATIVE_PATH, PREFLIGHT_LINEAGE_SHA256, "attempt-07 preflight lineage")
    for name, value in (("preflight input", preflight_input), ("preflight report", preflight_report), ("preflight lineage", preflight_lineage)):
        _expect(value.get("task_id"), "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2", f"{name} task identity changed")
    _expect(preflight_input.get("result"), "READY_FOR_EXECUTION", "attempt-07 input is not ready for execution")
    _expect(preflight_report.get("result"), "AUTHORIZABLE", "attempt-07 report is not authorizable")
    _expect(preflight_lineage.get("result"), "AUTHORIZABLE", "attempt-07 lineage is not authorizable")
    _expect(preflight_input.get("classification"), "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2_INPUT_READY", "attempt-07 input classification changed")
    _expect(preflight_report.get("classification"), "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2_AUTHORIZABLE", "attempt-07 report classification changed")
    _expect(preflight_lineage.get("classification"), "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V2_AUTHORIZABLE", "attempt-07 lineage classification changed")
    _expect(preflight_report.get("no_scientific_training"), True, "attempt-07 preflight training boundary changed")
    report_boundary = _mapping(preflight_report.get("claim_boundary"), "attempt-07 report claim boundary")
    input_boundary = _mapping(preflight_input.get("claim_boundary"), "attempt-07 input claim boundary")
    lineage_boundary = _mapping(preflight_lineage.get("claim_boundary"), "attempt-07 lineage claim boundary")
    for name, boundary_value in (("report", report_boundary), ("input", input_boundary), ("lineage", lineage_boundary)):
        for key in ("scientific_training_executed", "long_training_executed", "heldout_evaluation_executed", "protected_accessed", "test_accessed", "sealed_accessed", "training_authorized", "formal_promotion"):
            _expect(boundary_value.get(key), False, f"attempt-07 {name} boundary changed: {key}")
        _expect(boundary_value.get("generation_readiness"), "not_ready", f"attempt-07 {name} generation boundary changed")
    _expect(preflight_report.get("next"), TASK_ID, "attempt-07 next action changed")
    preflight_input_contract = _mapping(preflight_input.get("contract"), "attempt-07 input contract")
    _expect(preflight_input_contract.get("path"), CONTRACT_RELATIVE_PATH.as_posix(), "attempt-07 input contract path changed")
    _expect(preflight_input_contract.get("sha256"), CONTRACT_SHA256, "attempt-07 input contract digest changed")
    report_authorities = _mapping(preflight_report.get("authorities"), "attempt-07 report authorities")
    report_contract = _mapping(report_authorities.get("contract"), "attempt-07 report contract authority")
    _expect(report_contract.get("path"), CONTRACT_RELATIVE_PATH.as_posix(), "attempt-07 report contract path changed")
    _expect(report_contract.get("sha256"), CONTRACT_SHA256, "attempt-07 report contract digest changed")

    canonical_source = _mapping(preflight_input.get("canonical_training_source"), "attempt-07 input canonical source")
    _expect(canonical_source.get("training"), TRAINING_DATA_RELATIVE_PATH.as_posix(), "attempt-07 training source path changed")
    _expect(canonical_source.get("training_sha256"), TRAINING_DATA_SHA256, "attempt-07 training source digest changed")
    _expect(canonical_source.get("protected_test_read"), False, "attempt-07 protected source boundary changed")
    _expect(canonical_source.get("sealed_read"), False, "attempt-07 sealed source boundary changed")
    input_correction = _mapping(preflight_input.get("correction_authority"), "attempt-07 input correction authority")
    _expect(input_correction.get("path"), CORRECTION_AUTHORITY_RELATIVE_PATH.as_posix(), "attempt-07 correction authority path changed")
    _expect(input_correction.get("sha256"), CORRECTION_AUTHORITY_SHA256, "attempt-07 correction authority digest changed")

    consumer_ref = _mapping(preflight_input.get("consumer"), "attempt-07 input consumer")
    _expect(consumer_ref.get("path"), CONSUMER_RELATIVE_PATH.as_posix(), "attempt-07 consumer path changed")
    _expect(consumer_ref.get("sha256"), CONSUMER_SHA256, "attempt-07 consumer digest changed")
    runtime_ref = _mapping(preflight_input.get("aligned_runtime"), "attempt-07 input aligned runtime")
    _expect(runtime_ref.get("path"), ALIGNMENT_RUNTIME_RELATIVE_PATH.as_posix(), "attempt-07 runtime path changed")
    _expect(runtime_ref.get("sha256"), ALIGNMENT_RUNTIME_SHA256, "attempt-07 runtime digest changed")
    evaluator_ref = _mapping(preflight_input.get("evaluator"), "attempt-07 input evaluator")
    _expect(evaluator_ref.get("implementation"), EVALUATOR_RELATIVE_PATH.as_posix(), "attempt-07 evaluator implementation path changed")
    _expect(evaluator_ref.get("implementation_sha256"), EVALUATOR_SHA256, "attempt-07 evaluator implementation digest changed")
    _expect(evaluator_ref.get("checkpoint"), EVALUATOR_CHECKPOINT_RELATIVE_PATH.as_posix(), "attempt-07 evaluator checkpoint path changed")
    _expect(evaluator_ref.get("checkpoint_sha256"), EVALUATOR_CHECKPOINT_SHA256, "attempt-07 evaluator checkpoint digest changed")
    _expect(preflight_report.get("gpu_smoke", {}).get("target_device_grid"), PRODUCTION_PROPAGATION_GRID, "attempt-07 target grid changed")
    _expect(preflight_report.get("gpu_smoke", {}).get("batch_size"), LOGICAL_BATCH_SIZE, "attempt-07 logical batch changed")
    scale_ref = _mapping(preflight_report.get("gpu_smoke", {}).get("scale", {}).get("artifact"), "attempt-07 scale artifact")
    _expect(scale_ref.get("classification"), "NON_SCIENTIFIC_PREFLIGHT_SCALE", "attempt-07 scale classification changed")
    _expect(scale_ref.get("sha256"), PREFLIGHT_SCALE_SHA256, "attempt-07 scale digest changed")
    _expect(scale_ref.get("path"), str(_resolve(root, PREFLIGHT_SCALE_RELATIVE_PATH)), "attempt-07 scale path changed")

    consumer_path = _resolve(root, CONSUMER_RELATIVE_PATH)
    runtime_path = _resolve(root, ALIGNMENT_RUNTIME_RELATIVE_PATH)
    evaluator_path = _resolve(root, EVALUATOR_RELATIVE_PATH)
    evaluator_checkpoint_path = _resolve(root, EVALUATOR_CHECKPOINT_RELATIVE_PATH)
    for path, expected, name in (
        (consumer_path, CONSUMER_SHA256, "active V2 training consumer"),
        (runtime_path, ALIGNMENT_RUNTIME_SHA256, "active aligned runtime"),
        (evaluator_path, EVALUATOR_SHA256, "frozen evaluator implementation"),
        (evaluator_checkpoint_path, EVALUATOR_CHECKPOINT_SHA256, "frozen evaluator checkpoint"),
    ):
        if not path.is_file() or file_sha256(path) != expected:
            actual = file_sha256(path) if path.is_file() else None
            raise ExecutionBlocked("BLOCKED_EXECUTION_AUTHORITY_MISMATCH", "authority", f"{name} identity changed", expected_sha256=expected, actual_sha256=actual)
    legacy = v2.validate_legacy_exclusion(consumer_path)
    runtime_report = _read_hashed_json(root, ALIGNMENT_RUNTIME_REPORT_RELATIVE_PATH, ALIGNMENT_RUNTIME_REPORT_SHA256, "aligned runtime report")
    _expect(runtime_report.get("classification"), "LIGHTGEN_2D_SEMANTIC_GENERATION_INPUT_OLS_ALIGNMENT_RUNTIME_V1_IMPLEMENTED", "aligned runtime report classification changed")

    authorities = {
        "contract": contract.identity(root),
        "correction_authority": _identity(CORRECTION_AUTHORITY_RELATIVE_PATH, CORRECTION_AUTHORITY_SHA256),
        "original_contract": _identity(ORIGINAL_CONTRACT_RELATIVE_PATH, ORIGINAL_CONTRACT_SHA256),
        "preflight_input": _identity(PREFLIGHT_INPUT_RELATIVE_PATH, PREFLIGHT_INPUT_SHA256),
        "preflight_report": _identity(PREFLIGHT_REPORT_RELATIVE_PATH, PREFLIGHT_REPORT_SHA256),
        "preflight_lineage": _identity(PREFLIGHT_LINEAGE_RELATIVE_PATH, PREFLIGHT_LINEAGE_SHA256),
        "preflight_scale_not_reused": _identity(PREFLIGHT_SCALE_RELATIVE_PATH, PREFLIGHT_SCALE_SHA256),
        "preflight_stage_a_checkpoint_not_reused": {"sha256": PREFLIGHT_STAGE_A_CHECKPOINT_SHA256, "accepted_training_checkpoint": False},
        "preflight_stage_b_checkpoint_not_reused": {"sha256": PREFLIGHT_STAGE_B_CHECKPOINT_SHA256, "accepted_training_checkpoint": False},
        "consumer": _identity(CONSUMER_RELATIVE_PATH, CONSUMER_SHA256),
        "aligned_runtime": {**_identity(ALIGNMENT_RUNTIME_RELATIVE_PATH, ALIGNMENT_RUNTIME_SHA256), "report": _identity(ALIGNMENT_RUNTIME_REPORT_RELATIVE_PATH, ALIGNMENT_RUNTIME_REPORT_SHA256)},
        "evaluator": {"implementation": _identity(EVALUATOR_RELATIVE_PATH, EVALUATOR_SHA256), "checkpoint": _identity(EVALUATOR_CHECKPOINT_RELATIVE_PATH, EVALUATOR_CHECKPOINT_SHA256), "preprocessing": evaluator_ref.get("preprocessing"), "threshold": SEMANTIC_THRESHOLD},
        "dataset": {"training": _identity(TRAINING_DATA_RELATIVE_PATH, TRAINING_DATA_SHA256), "protected_test": {"path": PROTECTED_TEST_RELATIVE_PATH.as_posix(), "sha256": PROTECTED_TEST_SHA256, "read": False}, "sealed_read": False},
        "legacy_exclusion": legacy,
        "preflight_reexecuted": False,
    }
    return contract, authorities


def _fixed_development_selection(labels: torch.Tensor, development_indices: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(DEVELOPMENT_SELECTION_SEED)
    development_labels = labels.index_select(0, development_indices)
    selected: list[torch.Tensor] = []
    for digit in range(10):
        matches = development_indices[development_labels == digit]
        if matches.numel() != 500:
            raise InvalidExecution("development role is not balanced at 500 rows per class")
        selected.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:100]))
    return torch.cat(selected)


def _load_dataset(root: Path) -> tuple[
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    source_path = _resolve(root, TRAINING_DATA_RELATIVE_PATH)
    if not source_path.is_file():
        raise ExecutionBlocked("BLOCKED_TRAINING_SOURCE_UNAVAILABLE", "dataset", "TRAIN source is unavailable", path=TRAINING_DATA_RELATIVE_PATH.as_posix())
    actual = file_sha256(source_path)
    if actual != TRAINING_DATA_SHA256:
        raise ExecutionBlocked("BLOCKED_TRAINING_SOURCE_AUTHORITY_MISMATCH", "dataset", "TRAIN source SHA does not match authority", expected_sha256=TRAINING_DATA_SHA256, actual_sha256=actual)
    images, labels = load_local_mnist(root / "data", train=True)
    if tuple(images.shape) != (60_000, 1, 28, 28) or tuple(labels.shape) != (60_000,) or images.dtype != torch.float32 or labels.dtype != torch.long:
        raise InvalidExecution("local TRAIN source shape or dtype changed")
    train_indices, development_indices = stratified_mnist_indices(labels, seed=INITIALIZATION_SEED, validation_per_label=500)
    if train_indices.numel() != 55_000 or development_indices.numel() != 5_000:
        raise InvalidExecution("TRAIN/DEV role counts changed")
    if torch.unique(torch.cat((train_indices, development_indices))).numel() != 60_000:
        raise InvalidExecution("TRAIN and DEV roles overlap or omit source rows")
    train_counts = torch.bincount(labels.index_select(0, train_indices), minlength=10)
    development_counts = torch.bincount(labels.index_select(0, development_indices), minlength=10)
    if not torch.equal(development_counts, torch.full((10,), 500, dtype=torch.long)):
        raise InvalidExecution("DEV role is not balanced at 500 rows per class")
    if not torch.equal(train_counts + development_counts, torch.bincount(labels, minlength=10)):
        raise InvalidExecution("TRAIN/DEV partition does not preserve observed source counts")
    selection_indices = _fixed_development_selection(labels, development_indices)
    train_images = images.index_select(0, train_indices).contiguous()
    selection_images = images.index_select(0, selection_indices).contiguous()
    selection_labels = labels.index_select(0, selection_indices).contiguous()
    metadata = {
        "source": {"path": TRAINING_DATA_RELATIVE_PATH.as_posix(), "sha256": TRAINING_DATA_SHA256, "size_bytes": source_path.stat().st_size, "rows_loaded": 60_000},
        "train": {"rows": 55_000, "counts_by_class": [int(value) for value in train_counts.tolist()], "indices_sha256": tensor_sha256(train_indices)},
        "development": {"rows": 5_000, "counts_by_class": [int(value) for value in development_counts.tolist()], "indices_sha256": tensor_sha256(development_indices), "selection_rows": 1_000, "selection_per_class": 100, "selection_seed": DEVELOPMENT_SELECTION_SEED, "selection_indices_sha256": tensor_sha256(selection_indices)},
        "calibration": {"rows_read": 0, "status": "not consumed; prior authority only"},
        "protected": {"rows_read": 0, "status": "forbidden", "source_sha256_bound_only": PROTECTED_TEST_SHA256},
        "test": {"rows_read": 0, "status": "forbidden", "source_sha256_bound_only": PROTECTED_TEST_SHA256},
        "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
        "labels_as_model_input": False,
    }
    return metadata, images, labels, train_indices, selection_indices, train_images, selection_images, selection_labels


def _device_facts() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise ExecutionBlocked("BLOCKED_TARGET_DEVICE_UNAVAILABLE", "device", "required CUDA target is unavailable", cuda_available=bool(torch.cuda.is_available()), device_count=int(torch.cuda.device_count()))
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    capability = (int(properties.major), int(properties.minor))
    if capability != EXPECTED_COMPUTE_CAPABILITY or int(properties.total_memory) < EXPECTED_MIN_VRAM_BYTES:
        raise ExecutionBlocked("BLOCKED_TARGET_DEVICE_CHANGED", "device", "target device differs from authorized preflight target", expected_compute_capability=list(EXPECTED_COMPUTE_CAPABILITY), actual_compute_capability=list(capability), expected_min_vram_bytes=EXPECTED_MIN_VRAM_BYTES, actual_vram_bytes=int(properties.total_memory))
    env_root = Path(sys.prefix).resolve()
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
        raise ExecutionBlocked("BLOCKED_CUDA_BACKEND_UNAVAILABLE", "cuda_backend", "authorized CUDA complex/FFT backend is unavailable", error_type=type(error).__name__, error=str(error)) from error
    probe = {"complex_exp": bool(torch.isfinite(exp_probe).all()), "fft2_ifft2": bool(torch.isfinite(fft_probe).all()), "complex_polar": bool(torch.isfinite(polar_probe).all())}
    if not all(probe.values()):
        raise ExecutionBlocked("BLOCKED_CUDA_BACKEND_UNAVAILABLE", "cuda_backend", "authorized CUDA backend probe was nonfinite", probe=probe)
    smi = shutil.which("nvidia-smi")
    return device, {
        "device": str(device),
        "device_index": 0,
        "device_count": int(torch.cuda.device_count()),
        "gpu": properties.name,
        "compute_capability": list(capability),
        "vram_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "nvidia_smi": smi,
        "backend": {"cuda_built": bool(torch.backends.cuda.is_built()), "nvrtc_required_by_active_path": True, "nvrtc_candidate_dlls": [str(path) for path in dll_candidates], "nvrtc_dll_loads": dll_loads, "probe": probe, "env_prefix": str(env_root)},
    }


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }


def _validate_update(update: Any, stage: str, scale: FrozenPhysicalOLSScale | None) -> dict[str, Any]:
    result = update.loss.result
    finite = {
        "losses": all(bool(torch.isfinite(value)) for value in (update.loss.image, update.loss.latent, update.loss.scale, update.loss.total)),
        "complex_ols": bool(torch.isfinite(result.z_ols.coefficients).all()),
        "raw_cmos": bool(torch.isfinite(result.raw_intensity).all()),
        "generator_input": bool(torch.isfinite(result.generator_input_field.tensor).all()),
        "prior": update.loss.prior is None or bool(torch.isfinite(update.loss.prior.coefficients).all()),
    }
    if not all(finite.values()):
        raise InvalidExecution(f"{stage} update produced a nonfinite value")
    if not update.autograd.modal_projection_active or not update.autograd.w_ols_active or not update.autograd.frozen_scale_no_grad or not update.frozen_state_unchanged:
        raise InvalidExecution(f"{stage} aligned-path or frozen-state invariant failed")
    if stage == "A":
        if update.loss.prior is not None or not update.autograd.stage_a_no_mmd_gradient or abs(float(update.loss.total - update.loss.image)) > 1.0e-6:
            raise InvalidExecution("Stage-A loss firewall failed")
    else:
        if scale is None or update.loss.prior is None or not update.autograd.stage_b_normalized_mmd_active or not update.autograd.scale_gradient_active:
            raise InvalidExecution("Stage-B normalized MMD/scale-gradient firewall failed")
        expected_total = update.loss.image + update.loss.latent + update.loss.scale
        if not torch.allclose(update.loss.total, expected_total, rtol=1.0e-5, atol=1.0e-6):
            raise InvalidExecution("Stage-B loss decomposition changed")
    return {"finite": finite, "all_finite": True, "loss_decomposition": {"stage": stage, "scale_vector_sha256": scale.vector_sha256 if scale is not None else None}}


def _run_update(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    images: torch.Tensor,
    indices: torch.Tensor,
    *,
    stage: str,
    global_step: int,
    stage_local_step: int,
    optimizer: torch.optim.Optimizer,
    scale: FrozenPhysicalOLSScale | None,
    device: torch.device,
) -> dict[str, Any]:
    batch = images.index_select(0, indices).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        update = consumer.update(batch, stage=stage, global_step=global_step, stage_local_step=stage_local_step, optimizer=optimizer, scale=scale)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        memory = _memory_snapshot(device)
        record = update.to_dict()
        record.update(_validate_update(update, stage, scale))
        record["elapsed_seconds"] = elapsed
        record["memory"] = memory
        return record
    except torch.cuda.OutOfMemoryError as error:
        raise ExecutionBlocked("BLOCKED_RUNTIME_OUT_OF_MEMORY", f"{stage}_update_{global_step}", "authorized frozen batch/grid exhausted target-device memory", error_type=type(error).__name__, error=str(error), memory=_memory_snapshot(device)) from error
    finally:
        del batch


@torch.no_grad()
def _development_metrics(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    evaluator: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    stage: str,
    stage_local_step: int,
    scale: FrozenPhysicalOLSScale | None,
) -> dict[str, Any]:
    recall = development_macro_recall(consumer, evaluator, images, labels, batch_size=LOGICAL_BATCH_SIZE)
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    sums = {"L_image": 0.0, "L_latent_normalized": 0.0, "L_scale": 0.0, "L_total": 0.0}
    raw_power_sum = torch.zeros(100, dtype=torch.float64)
    normalized_power_sum = torch.zeros(100, dtype=torch.float64)
    count = 0
    try:
        for start in range(0, images.shape[0], LOGICAL_BATCH_SIZE):
            batch = images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            loss = consumer.forward_loss(batch, stage=stage, stage_local_step=stage_local_step, scale=scale)
            metrics = loss.metrics()
            for name in sums:
                sums[name] += metrics[name] * batch.shape[0]
            z = loss.result.z_ols.coefficients
            raw_power_sum += (z.real.to(torch.float64).square() + z.imag.to(torch.float64).square()).sum(dim=0).cpu()
            if loss.normalized is not None:
                normalized_power_sum += (loss.normalized.real.to(torch.float64).square() + loss.normalized.imag.to(torch.float64).square()).sum(dim=0).cpu()
            count += int(batch.shape[0])
    finally:
        consumer.runtime.train(was_training)
    if count != images.shape[0]:
        raise InvalidExecution("development metric row count changed")
    record = {"development_macro_recall": recall}
    record.update({f"development_{name}": value / count for name, value in sums.items()})
    record["development_raw_ols_power_mean"] = (raw_power_sum / count).mean().item()
    record["development_normalized_ols_power_mean"] = (normalized_power_sum / count).mean().item()
    record["development_selection_rows"] = int(images.shape[0])
    record["development_semantic_eligible"] = recall >= SEMANTIC_THRESHOLD
    if not all(torch.isfinite(torch.tensor(value)) for value in record.values() if isinstance(value, (float, int))):
        raise InvalidExecution("development metrics are nonfinite")
    return record


def _checkpoint_identity(root: Path, path: Path, *, stage: str, local_step: int) -> dict[str, Any]:
    return {"path": _relative(root, path), "sha256": file_sha256(path), "stage": stage, "stage_local_step": local_step, "accepted_training_checkpoint": False}


def _progress_update(progress: dict[str, Any], stage: str, record: dict[str, Any]) -> None:
    progress.setdefault(stage, {}).setdefault("updates", []).append(record)
    progress[stage]["updates_completed"] = len(progress[stage]["updates"])


def _run_stage_a(
    root: Path,
    target: Path,
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    evaluator: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    selection_images: torch.Tensor,
    selection_labels: torch.Tensor,
    *,
    device: torch.device,
    git: Mapping[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    optimizer = consumer.create_optimizer()
    sampler = DeterministicTrainRoleSampler(train_indices, labels, steps=STAGE_A_STEPS)
    state = V2TrainingStateMachine(stage=STAGE_A_TRAIN)
    stage = progress.setdefault("stage_a", {"updates": [], "checkpoint_records": []})
    stage.setdefault("updates", [])
    stage.setdefault("checkpoint_records", [])
    stage["protocol"] = {"steps": STAGE_A_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "loss": "L_A=L_image", "selection_rule": STAGE_A_CHECKPOINT_RULE}
    stage["parameter_manifest"] = consumer.parameter_manifest()
    stage["optimizer_manifest"] = consumer.optimizer_manifest(optimizer)
    started = time.perf_counter()
    for local_step in range(1, STAGE_A_STEPS + 1):
        indices, sampler_labels = sampler.next_batch()
        if not torch.equal(sampler_labels, torch.arange(10, dtype=torch.long)):
            raise InvalidExecution("TRAIN sampler labels changed")
        record = _run_update(consumer, images, indices, stage="A", global_step=local_step, stage_local_step=local_step, optimizer=optimizer, scale=None, device=device)
        _progress_update(progress, "stage_a", record)
        if local_step in CHECKPOINT_STEPS:
            semantic = _development_metrics(consumer, evaluator, selection_images, selection_labels, stage="A", stage_local_step=local_step, scale=None)
            state.global_step = local_step
            state.stage_local_step = local_step
            state.transition(STAGE_A_SEMANTIC_CHECK)
            checkpoint_record = {"stage_local_step": local_step, **semantic, "checkpoint_selection_executed": False}
            checkpoint_path = target / f"lightgen_2d_semantic_complex_ols_training_v2_stage_a_checkpoint_step_{local_step:03d}.v2.pt"
            payload = checkpoint_payload(
                consumer,
                optimizer,
                state=state,
                sampler_state=sampler.state_dict(),
                semantic_selection_state={"threshold": SEMANTIC_THRESHOLD, "checkpoint_steps": list(CHECKPOINT_STEPS), "selection_executed": False, "rule": STAGE_A_CHECKPOINT_RULE, "record": checkpoint_record},
                git=git,
            )
            save_checkpoint(checkpoint_path, payload)
            checkpoint_record["checkpoint"] = _checkpoint_identity(root, checkpoint_path, stage="A", local_step=local_step)
            stage["checkpoint_records"].append(checkpoint_record)
            eligible = bool(semantic["development_semantic_eligible"])
            state.stage_a_checkpoint_checked(final_checkpoint=local_step == STAGE_A_STEPS, eligible=eligible)
            stage["state"] = state.to_dict()
            print(f"scientific: Stage A checkpoint {local_step}/300 recall={semantic['development_macro_recall']:.6f} L_image={semantic['development_L_image']:.8f} eligible={eligible}", flush=True)
        elif local_step % 25 == 0:
            print(f"scientific: Stage A step {local_step}/300 L_image={record['loss']['L_image']:.8f}", flush=True)
    selected = None if state.terminal_classification is not None else select_stage_a_checkpoint(stage["checkpoint_records"])
    stage["elapsed_seconds"] = time.perf_counter() - started
    stage["selection"] = {"threshold": SEMANTIC_THRESHOLD, "rule": STAGE_A_CHECKPOINT_RULE, "selected": dict(selected) if selected is not None else None}
    stage["state"] = state.to_dict()
    if selected is None:
        raise ScientificGateFailure(STAGE_A_FAILURE_CLASSIFICATION, STAGE_A_SEMANTIC_CHECK, stage_a=stage)
    state.selected_stage_a_checkpoint = dict(selected["checkpoint"])
    stage["selected_checkpoint"] = dict(selected["checkpoint"])
    stage["state"] = state.to_dict()
    return {"state": state, "stage": stage, "selected": dict(selected)}


def _extract_formal_scale(
    root: Path,
    target: Path,
    contract: v2.FrozenV2TrainingContract,
    selected_a: Mapping[str, Any],
    train_images: torch.Tensor,
    train_indices: torch.Tensor,
    *,
    device: torch.device,
    progress: dict[str, Any],
) -> tuple[LightGen2DComplexOLSTrainingConsumerV2, V2TrainingStateMachine, FrozenPhysicalOLSScale, dict[str, Any]]:
    state = progress["stage_a"]["state"]
    stage_state = V2TrainingStateMachine.from_dict(state)
    stage_state.transition(SCALE_EXTRACT)
    selected_identity = _mapping(selected_a.get("checkpoint"), "selected Stage-A checkpoint")
    selected_path = _resolve(root, Path(str(selected_identity["path"])))
    selected_consumer = LightGen2DComplexOLSTrainingConsumerV2(contract, device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
    selected_optimizer = selected_consumer.create_optimizer()
    loaded = load_checkpoint(selected_path, selected_consumer, selected_optimizer)
    if loaded.global_step != int(selected_a["stage_local_step"]) or loaded.stage_local_step != int(selected_a["stage_local_step"]):
        raise InvalidExecution("selected Stage-A checkpoint coordinates changed")
    checkpoint_identity = {"path": _relative(root, selected_path), "sha256": file_sha256(selected_path), "stage": "A", "stage_local_step": int(selected_a["stage_local_step"]), "accepted_training_checkpoint": False}
    data_identity = {"role": "TRAIN", "path": TRAINING_DATA_RELATIVE_PATH.as_posix(), "sha256": TRAINING_DATA_SHA256, "rows": 55_000, "indices_sha256": tensor_sha256(train_indices)}
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    try:
        formal_scale = extract_train_only_ols_scale(selected_consumer, train_images, data_role="train", checkpoint_identity=checkpoint_identity, data_identity=data_identity)
        torch.cuda.synchronize(device)
    except torch.cuda.OutOfMemoryError as error:
        raise ExecutionBlocked("BLOCKED_RUNTIME_OUT_OF_MEMORY", "formal_scale_extraction", "formal train-only scale extraction exhausted target-device memory", error_type=type(error).__name__, error=str(error), memory=_memory_snapshot(device)) from error
    elapsed = time.perf_counter() - started
    if formal_scale.classification != "SCIENTIFIC_TRAIN_ONLY" or formal_scale.values.dtype != torch.float64 or tuple(formal_scale.values.shape) != (100,) or not torch.isfinite(formal_scale.values).all() or bool((formal_scale.values <= EPSILON_SCALE).any()):
        raise InvalidExecution("formal train-only scale is invalid")
    scale_path = target / FORMAL_SCALE_NAME
    published_scale = save_scale_artifact(scale_path, formal_scale)
    reloaded_scale = load_scale_artifact(scale_path, expected_sha256=published_scale.artifact_sha256)
    if reloaded_scale.classification != "SCIENTIFIC_TRAIN_ONLY" or reloaded_scale.vector_sha256 != published_scale.vector_sha256:
        raise InvalidExecution("formal scale artifact reload changed authority")
    stage_state.scale_identity = reloaded_scale.identity()
    stage_state.transition(SCALE_FREEZE)
    progress["scale"] = {
        "data_role": "TRAIN",
        "classification": reloaded_scale.classification,
        "rows_traversed": 55_000,
        "formula": "s_j=mean_train |z_j|^2",
        "port_count": 100,
        "epsilon_scale": EPSILON_SCALE,
        "float64_accumulation": True,
        "values_finite": True,
        "values_above_epsilon": True,
        "elapsed_seconds": elapsed,
        "rows_per_second": 55_000 / elapsed if elapsed > 0 else None,
        "memory": _memory_snapshot(device),
        "indices_sha256": tensor_sha256(train_indices),
        "scale_vector_sha256": reloaded_scale.vector_sha256,
        "artifact": {"path": _relative(root, scale_path), "sha256": reloaded_scale.artifact_sha256, "vector_sha256": reloaded_scale.vector_sha256, "schema": SCALE_SCHEMA, "classification": reloaded_scale.classification},
        "preflight_scale_reused": False,
        "preflight_scale_authority": {"path": PREFLIGHT_SCALE_RELATIVE_PATH.as_posix(), "sha256": PREFLIGHT_SCALE_SHA256, "classification": "NON_SCIENTIFIC_PREFLIGHT_SCALE", "consumed": False},
    }
    progress["stage_a"]["state_after_scale_freeze"] = stage_state.to_dict()
    return selected_consumer, stage_state, reloaded_scale, progress["scale"]


def _run_stage_b(
    root: Path,
    target: Path,
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    state: V2TrainingStateMachine,
    scale: FrozenPhysicalOLSScale,
    evaluator: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    selection_images: torch.Tensor,
    selection_labels: torch.Tensor,
    *,
    device: torch.device,
    git: Mapping[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    consumer.set_scale(scale)
    optimizer_b, reset_marker = consumer.reset_stage_b_optimizer()
    if optimizer_b.state or not reset_marker.get("optimizer_reset") or not reset_marker.get("new_optimizer_state_empty") or reset_marker.get("stage_a_optimizer_moments_restored"):
        raise InvalidExecution("Stage-B optimizer reset invariant failed")
    state.transition(STAGE_B_TRAIN)
    state.optimizer_reset_marker = reset_marker
    sampler = DeterministicTrainRoleSampler(train_indices, labels, steps=STAGE_B_STEPS)
    stage = progress.setdefault("stage_b", {"updates": [], "checkpoint_records": []})
    stage.setdefault("updates", [])
    stage.setdefault("checkpoint_records", [])
    stage["protocol"] = {"steps": STAGE_B_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "loss": "L_B=L_image+L_latent_normalized+L_scale", "selection_rule": STAGE_B_CHECKPOINT_RULE, "optimizer_reset": reset_marker}
    stage["parameter_manifest"] = consumer.parameter_manifest()
    stage["optimizer_manifest"] = consumer.optimizer_manifest(optimizer_b)
    scale_before = scale.values.clone()
    started = time.perf_counter()
    for local_step in range(1, STAGE_B_STEPS + 1):
        indices, sampler_labels = sampler.next_batch()
        if not torch.equal(sampler_labels, torch.arange(10, dtype=torch.long)):
            raise InvalidExecution("TRAIN sampler labels changed in Stage B")
        record = _run_update(consumer, images, indices, stage="B", global_step=STAGE_A_STEPS + local_step, stage_local_step=local_step, optimizer=optimizer_b, scale=scale, device=device)
        _progress_update(progress, "stage_b", record)
        if not torch.equal(scale.values, scale_before):
            raise InvalidExecution("formal physical scale mutated during Stage B")
        if local_step in CHECKPOINT_STEPS:
            semantic = _development_metrics(consumer, evaluator, selection_images, selection_labels, stage="B", stage_local_step=local_step, scale=scale)
            state.global_step = STAGE_A_STEPS + local_step
            state.stage_local_step = local_step
            state.transition(STAGE_B_SEMANTIC_CHECK)
            checkpoint_record = {"stage_local_step": local_step, **semantic, "checkpoint_selection_executed": False}
            checkpoint_path = target / f"lightgen_2d_semantic_complex_ols_training_v2_stage_b_checkpoint_step_{local_step:03d}.v2.pt"
            payload = checkpoint_payload(
                consumer,
                optimizer_b,
                state=state,
                sampler_state=sampler.state_dict(),
                semantic_selection_state={"threshold": SEMANTIC_THRESHOLD, "checkpoint_steps": list(CHECKPOINT_STEPS), "selection_executed": False, "rule": STAGE_B_CHECKPOINT_RULE, "record": checkpoint_record},
                git=git,
                scale=scale,
                selected_stage_a_checkpoint=state.selected_stage_a_checkpoint,
                optimizer_reset_marker=reset_marker,
            )
            save_checkpoint(checkpoint_path, payload)
            checkpoint_record["checkpoint"] = _checkpoint_identity(root, checkpoint_path, stage="B", local_step=local_step)
            stage["checkpoint_records"].append(checkpoint_record)
            eligible = bool(semantic["development_semantic_eligible"])
            state.stage_b_checkpoint_checked(final_checkpoint=local_step == STAGE_B_STEPS, eligible=eligible)
            stage["state"] = state.to_dict()
            print(f"scientific: Stage B checkpoint {local_step}/300 recall={semantic['development_macro_recall']:.6f} L_total={semantic['development_L_total']:.8f} eligible={eligible}", flush=True)
        elif local_step % 25 == 0:
            print(f"scientific: Stage B step {local_step}/300 L_total={record['loss']['L_total']:.8f}", flush=True)
    selected = None if state.terminal_classification is not None else select_stage_b_checkpoint(stage["checkpoint_records"])
    stage["elapsed_seconds"] = time.perf_counter() - started
    stage["selection"] = {"threshold": SEMANTIC_THRESHOLD, "rule": STAGE_B_CHECKPOINT_RULE, "selected": dict(selected) if selected is not None else None}
    stage["state"] = state.to_dict()
    if selected is None:
        raise ScientificGateFailure(STAGE_B_FAILURE_CLASSIFICATION, STAGE_B_SEMANTIC_CHECK, stage_b=stage)
    state.transition(COMPLETE)
    stage["selected_checkpoint"] = dict(selected["checkpoint"])
    final_optimizer = consumer.create_optimizer()
    loaded = load_checkpoint(_resolve(root, Path(str(selected["checkpoint"]["path"]))), consumer, final_optimizer, expected_scale=scale)
    if loaded.stage != STAGE_B_SEMANTIC_CHECK or loaded.scale is None or loaded.scale.vector_sha256 != scale.vector_sha256 or loaded.scale.artifact_sha256 != scale.artifact_sha256:
        raise InvalidExecution("selected final Stage-B checkpoint failed safe reload")
    stage["final_selected_checkpoint_reload"] = {"passed": True, "stage": loaded.stage, "global_step": loaded.global_step, "stage_local_step": loaded.stage_local_step, "scale_vector_sha256": loaded.scale.vector_sha256, "scale_artifact_sha256": loaded.scale.artifact_sha256}
    stage["state_after_complete"] = state.to_dict()
    return {"stage": stage, "selected": dict(selected), "state": state, "optimizer": final_optimizer, "sampler": sampler}


def _resolve_output_dir(root: Path, output_dir: Path | None) -> Path:
    target = _resolve(root, DEFAULT_OUTPUT_DIR if output_dir is None else output_dir)
    outputs_root = (root / "outputs").resolve()
    if not target.is_relative_to(outputs_root):
        raise InvalidExecution("scientific output must remain under outputs/")
    if target.exists():
        raise ExecutionBlocked("BLOCKED_OUTPUT_ALREADY_EXISTS", "output", "refusing to overwrite an existing scientific run output", path=_relative(root, target))
    return target


def _initial_report(started: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "status": "invalid_harness",
        "result": "INVALID",
        "classification": "INVALID_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2_NOT_COMPLETED",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": {"explicit_user_authorization_consumed": True, "scope": "exactly one bounded Stage-A/scale/Stage-B scientific execution", "no_automatic_follow_up": True},
        "provenance": {"git_start": dict(started)},
        "claim_boundary": {"scientific_training_executed": False, "long_training_executed": False, "heldout_evaluation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "generation_executed": False, "generation_acceptance": False, "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"},
        "execution": {"stage_a": {"updates": [], "checkpoint_records": [], "updates_completed": 0}, "stage_b": {"updates": [], "checkpoint_records": [], "updates_completed": 0}},
        "next": None,
    }


def run_training(root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    repo_root = _root() if root is None else root.resolve()
    started_git = git_facts(repo_root)
    report = _initial_report(started_git)
    progress = report["execution"]
    target: Path | None = None
    authorities: dict[str, Any] | None = None
    result_summary: dict[str, Any] | None = None
    try:
        target = _resolve_output_dir(repo_root, output_dir)
        contract, authorities = _validate_preflight_authority(repo_root)
        report["authorities"] = authorities
        target.mkdir(parents=True, exist_ok=False)
        metadata, images, labels, train_indices, selection_indices, train_images, selection_images, selection_labels = _load_dataset(repo_root)
        report["dataset"] = metadata
        device, device_facts = _device_facts()
        report["device"] = device_facts
        evaluator_path = _resolve(repo_root, EVALUATOR_CHECKPOINT_RELATIVE_PATH)
        evaluator, evaluator_payload = load_frozen_evaluator(evaluator_path, device=device)
        if any(parameter.requires_grad for parameter in evaluator.parameters()):
            raise InvalidExecution("FrozenMNISTEvaluator is not frozen")
        validation_accuracy = float(evaluator_payload.get("validation_accuracy", 0.0))
        if validation_accuracy < 0.985:
            raise ExecutionBlocked("BLOCKED_EVALUATOR_QUALIFICATION_CHANGED", "evaluator", "frozen evaluator qualification changed", validation_accuracy=validation_accuracy)
        report["evaluator"] = {"identity": "FrozenMNISTEvaluator v1", "implementation": EVALUATOR_RELATIVE_PATH.as_posix(), "implementation_sha256": EVALUATOR_SHA256, "checkpoint": EVALUATOR_CHECKPOINT_RELATIVE_PATH.as_posix(), "checkpoint_sha256": EVALUATOR_CHECKPOINT_SHA256, "validation_accuracy": validation_accuracy, "preprocessing": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28", "threshold": SEMANTIC_THRESHOLD, "gradient_enabled": False}
        report["protocol"] = {"propagation_grid": PRODUCTION_PROPAGATION_GRID, "logical_batch_size": LOGICAL_BATCH_SIZE, "initialization_seed": INITIALIZATION_SEED, "schedule_seed": INITIALIZATION_SEED, "development_selection_seed": DEVELOPMENT_SELECTION_SEED, "stage_a_steps": STAGE_A_STEPS, "stage_b_steps": STAGE_B_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "semantic_threshold": SEMANTIC_THRESHOLD, "scale_schema": SCALE_SCHEMA}
        torch.manual_seed(INITIALIZATION_SEED)
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
        consumer = LightGen2DComplexOLSTrainingConsumerV2(contract, device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)
        report["trainable_parameter_manifest"] = consumer.parameter_manifest()
        stage_a = _run_stage_a(repo_root, target, consumer, evaluator, images, labels, train_indices, selection_images, selection_labels, device=device, git=started_git, progress=progress)
        report["stage_a"] = stage_a["stage"]
        report["claim_boundary"]["scientific_training_executed"] = True
        report["claim_boundary"]["long_training_executed"] = True
        del consumer
        del evaluator
        torch.cuda.empty_cache()
        selected_consumer, state, formal_scale, scale_record = _extract_formal_scale(repo_root, target, contract, stage_a["selected"], train_images, train_indices, device=device, progress=progress)
        report["scale"] = scale_record
        stage_b = _run_stage_b(repo_root, target, selected_consumer, state, formal_scale, load_frozen_evaluator(evaluator_path, device=device)[0], images, labels, train_indices, selection_images, selection_labels, device=device, git=started_git, progress=progress)
        report["stage_b"] = stage_b["stage"]
        report["claim_boundary"]["scientific_training_executed"] = True
        report["claim_boundary"]["long_training_executed"] = True
        report["result"] = "VALID_COMPLETED"
        report["status"] = "valid_completed"
        report["classification"] = SUCCESS_CLASSIFICATION
        report["next"] = "STOP_AFTER_V2_TRAINING_RESULT"
        result_summary = {"selected_stage_a_checkpoint": stage_a["selected"]["checkpoint"], "selected_stage_b_checkpoint": stage_b["selected"]["checkpoint"], "scale": scale_record["artifact"]}
    except ScientificGateFailure as failure:
        report.update({"result": "VALID_COMPLETED", "status": "valid_failed", "classification": failure.classification, "scientific_gate_failure": {"stage": failure.stage, **failure.details}, "next": "STOP_AFTER_V2_TRAINING_RESULT"})
        report["claim_boundary"]["scientific_training_executed"] = bool(progress.get("stage_a", {}).get("updates_completed", 0) or progress.get("stage_b", {}).get("updates_completed", 0))
        report["claim_boundary"]["long_training_executed"] = report["claim_boundary"]["scientific_training_executed"]
    except ExecutionBlocked as blocked:
        report.update({"result": "BLOCKED", "status": "valid_failed", "classification": blocked.classification, "hard_blocker": {"stage": blocked.stage, "message": str(blocked), **blocked.details}, "next": "resolve_execution_blocker_before_downstream_stage"})
    except InvalidExecution as error:
        report.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "repair_execution_harness_before_reuse"})
    except torch.cuda.OutOfMemoryError as error:
        report.update({"result": "BLOCKED", "status": "resource_aborted", "classification": "BLOCKED_RUNTIME_OUT_OF_MEMORY", "hard_blocker": {"stage": "runtime", "message": str(error), "error_type": type(error).__name__}, "next": "resolve_execution_resource_blocker_before_reuse"})
    except Exception as error:
        report.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V2_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-10_000:]}, "next": "repair_execution_harness_before_reuse"})
    finally:
        ending_git = git_facts(repo_root)
        report.setdefault("provenance", {})["git_end"] = ending_git
        report["provenance"]["branch_unchanged"] = started_git["branch"] == ending_git["branch"]
        report["provenance"]["head_unchanged"] = started_git["head"] == ending_git["head"]
        report["claim_boundary"].update({"heldout_evaluation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "generation_executed": False, "generation_acceptance": False, "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"})
        report["artifact_self_hash"] = "not_embedded_to_avoid_self_reference"
        if target is not None and target.is_dir():
            report_path = target / REPORT_NAME
            atomic_write_new_json(report_path, report, artifact_kind="LightGen V2 scientific training report")
            lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "result": report["result"], "status": report["status"], "classification": report["classification"], "contract": (authorities or {}).get("contract"), "authorities": authorities, "device": report.get("device"), "dataset": report.get("dataset"), "execution": report.get("execution"), "stage_a": report.get("stage_a"), "scale": report.get("scale"), "stage_b": report.get("stage_b"), "artifacts": {"report": {"path": _relative(repo_root, report_path), "sha256": file_sha256(report_path)}}, "claim_boundary": report["claim_boundary"], "provenance": report["provenance"], "lineage_self_hash": "not_embedded_to_avoid_self_reference"}
            lineage_path = target / LINEAGE_NAME
            atomic_write_new_json(lineage_path, lineage, artifact_kind="LightGen V2 scientific training lineage")
            result_summary = {**(result_summary or {}), "result": report["result"], "status": report["status"], "classification": report["classification"], "report": {"path": _relative(repo_root, report_path), "sha256": file_sha256(report_path)}, "lineage": {"path": _relative(repo_root, lineage_path), "sha256": file_sha256(lineage_path)}, "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready", "next": report["next"]}
    if result_summary is None:
        result_summary = {"result": report["result"], "status": report["status"], "classification": report["classification"], "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready", "next": report["next"]}
    print(json.dumps(result_summary, indent=2, ensure_ascii=False, default=str), flush=True)
    return result_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="new output directory under outputs/")
    return parser.parse_args()


def main() -> int:
    result = run_training(output_dir=_parse_args().output_dir)
    return 0 if result.get("result") in {"VALID_COMPLETED", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
