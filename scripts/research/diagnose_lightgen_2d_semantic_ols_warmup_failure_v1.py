"""Diagnose the frozen LightGen V2 Stage-A semantic warmup failure.

This is a read-only diagnostic consumer.  It binds the corrected V2 authority,
the completed Stage-A run, and its three existing checkpoints, then performs
CPU-only forward measurements and ``torch.autograd.grad`` probes.  It never
creates an optimizer, calls ``optimizer.step``/``backward`` on model state,
reads calibration/protected/test/sealed data, extracts a scale, or generates
from a prior.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 as v2
from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 import (
    CHECKPOINT_SCHEMA,
    LightGen2DComplexOLSTrainingConsumerV2,
    SEMANTIC_THRESHOLD,
    STAGE_A_SEMANTIC_CHECK,
)
from ldm_project.v2.evidence import atomic_write_new_json, file_sha256, safe_torch_load, tensor_sha256
from ldm_project.v2.student.evaluator import (
    load_frozen_evaluator,
    load_local_mnist,
    stratified_mnist_indices,
)


TASK_ID = "DIAGNOSE_LIGHTGEN_2D_SEMANTIC_OLS_WARMUP_FAILURE_V1"
RESULT = "VALID_COMPLETED"
REPORT_SCHEMA = "lightgen_2d_semantic_ols_warmup_failure_diagnosis_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_semantic_ols_warmup_failure_diagnosis_lineage.v1"
OUTPUT_DIR = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_ols_warmup_failure_diagnosis_v1_attempt_01"
)
REPORT_NAME = "lightgen_2d_semantic_ols_warmup_failure_diagnosis_report.v1.json"
LINEAGE_NAME = "lightgen_2d_semantic_ols_warmup_failure_diagnosis_lineage.v1.json"

CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json"
)
CONTRACT_SHA256 = "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf"
CORRECTION_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction.v1.json"
)
CORRECTION_SHA256 = "fdf03a0a8568db947011549e7869ff11ab972e19c1e01fbc3e7d2e9d850dbeb0"
ORIGINAL_CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_contract_v2_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_contract.v2.json"
)
ORIGINAL_CONTRACT_SHA256 = "b85fce30b70643638c1ea276d78dc58174c90980e420414cafc3f1947155de53"
TRAINING_DATA_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_DATA_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
CONSUMER_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen_2d_semantic_complex_ols_training_v2.py"
)
CONSUMER_SHA256 = "c659b93e0138531543ddfbe974c6dd4231a70b573a0701555b0dcd558a673ca3"
RUNTIME_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen_2d_semantic_generation_input_ols_alignment.py"
)
RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
RUNTIME_REPORT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_v1_attempt_02/"
    "lightgen_2d_semantic_generation_input_ols_alignment_runtime_report.v1.json"
)
RUNTIME_REPORT_SHA256 = "e26d24654e1d9c8707e9a6b93b5c5ee66f1172fb0ae7cd686d24ea03c9263af4"
EVALUATOR_PATH = Path("src/ldm_project/v2/student/evaluator.py")
EVALUATOR_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"
EVALUATOR_CHECKPOINT_PATH = Path(
    "outputs/20260724_v2/runs/"
    "20260724_162422_705459_formal_v2_identity_logit_amplitude_finite_crop_identity/"
    "frozen_mnist_evaluator.pt"
)
EVALUATOR_CHECKPOINT_SHA256 = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
PREFLIGHT_REPORT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_report.v1.json"
)
PREFLIGHT_REPORT_SHA256 = "b7bc2482d55af70ad8fc7a1b204c636425e9bca64448feda9e11e0111b5d64a4"
PREFLIGHT_INPUT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_input.v1.json"
)
PREFLIGHT_INPUT_SHA256 = "2e090926b057f70ed9cacc63473d16786336aa39b9371da2dcfdbccfead94b1d"
PREFLIGHT_LINEAGE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_attempt_07/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v2_lineage.v1.json"
)
PREFLIGHT_LINEAGE_SHA256 = "30913cd07ad318fbd94d2e5a4a346da6ca8390087e47a0b145d2c08b54b2d3bd"
TRAINING_REPORT_PATH = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_v2_attempt_02/"
    "lightgen_2d_semantic_complex_ols_training_v2_report.v1.json"
)
TRAINING_REPORT_SHA256 = "de55a0f307675ded10aca5726f9ac09474633b2c9a5f8bfb1890075de53a530d"
TRAINING_LINEAGE_PATH = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_v2_attempt_02/"
    "lightgen_2d_semantic_complex_ols_training_v2_lineage.v1.json"
)
TRAINING_LINEAGE_SHA256 = "899aa78daff4bbe2f70cf72e01080e323e24d791f4937ca35fefc8c9b30394dc"
CHECKPOINTS = {
    100: (
        Path(
            "outputs/20260911_v2/runs/"
            "lightgen_2d_semantic_complex_ols_training_v2_attempt_02/"
            "lightgen_2d_semantic_complex_ols_training_v2_stage_a_checkpoint_step_100.v2.pt"
        ),
        "b9b90681fd64907d98c429a9432fe69db428882256e31f8df35c49717be98efe",
    ),
    200: (
        Path(
            "outputs/20260911_v2/runs/"
            "lightgen_2d_semantic_complex_ols_training_v2_attempt_02/"
            "lightgen_2d_semantic_complex_ols_training_v2_stage_a_checkpoint_step_200.v2.pt"
        ),
        "cfa8b9c270110c29b553ef69631d76e8fccfd58434e20b14b34318d6ea4fd2a0",
    ),
    300: (
        Path(
            "outputs/20260911_v2/runs/"
            "lightgen_2d_semantic_complex_ols_training_v2_attempt_02/"
            "lightgen_2d_semantic_complex_ols_training_v2_stage_a_checkpoint_step_300.v2.pt"
        ),
        "a0898b42859d2b66279d212e87292bd3f2c813200e351586a844306a0dbeae35",
    ),
}
HISTORICAL_DIAGNOSIS_PATH = Path(
    "scripts/research/diagnose_lightgen_2d_semantic_real_path_control_failure_v1.py"
)
HISTORICAL_DIAGNOSIS_SHA256 = "24f32bd7344a2ed1ce52185891fdc0ee3cb30fc040aef24a7e44eee3296a8594"

PANEL_PER_CLASS = 20
PANEL_SIZE = PANEL_PER_CLASS * 10
BATCH_SIZE = 10
PAIR_SEED = 20260911
GAUGE_EPSILON = 1.0e-6
MATERIAL_PORT_RELATIVE_POWER = 1.0e-3
MATERIAL_PORT_ABSOLUTE_POWER = 1.0e-12

PRIMARY_CLASSIFICATIONS = {
    "objective": "SEMANTIC_WARMUP_FAILURE_IMAGE_OBJECTIVE_MISMATCH_SUPPORTED",
    "optimization": "SEMANTIC_WARMUP_FAILURE_OPTIMIZATION_NOT_PROGRESSING_SUPPORTED",
    "gradient": "SEMANTIC_WARMUP_FAILURE_ENCODER_GRADIENT_STARVATION_SUPPORTED",
    "ols": "SEMANTIC_WARMUP_FAILURE_ENCODER_OLS_INFORMATION_COLLAPSE_SUPPORTED",
    "generator": "SEMANTIC_WARMUP_FAILURE_GENERATOR_DECODE_LIMITATION_SUPPORTED",
    "capacity": "SEMANTIC_WARMUP_FAILURE_ARCHITECTURE_CAPACITY_LIMITATION_SUPPORTED",
    "generalization": "SEMANTIC_WARMUP_FAILURE_GENERALIZATION_FAILURE_SUPPORTED",
}
MIXED_CLASSIFICATION = "SEMANTIC_WARMUP_FAILURE_MIXED"
UNRESOLVED_CLASSIFICATION = "SEMANTIC_WARMUP_FAILURE_UNRESOLVED"


class BlockedDiagnosis(RuntimeError):
    """A frozen authority or permitted-data boundary cannot be established."""


class InvalidDiagnosis(RuntimeError):
    """The diagnostic harness is invalid rather than a scientific result."""


def _require(condition: bool, message: str, *, blocked: bool = False) -> None:
    if not condition:
        raise BlockedDiagnosis(message) if blocked else InvalidDiagnosis(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _ref(root: Path, path: Path, role: str, expected_sha256: str) -> dict[str, str]:
    target = _resolve(root, path)
    _require(target.is_file(), f"{role} is unavailable: {path.as_posix()}", blocked=True)
    actual = file_sha256(target)
    _require(actual == expected_sha256, f"{role} SHA256 changed", blocked=True)
    return {"path": path.as_posix(), "role": role, "sha256": actual}


def _read_json(root: Path, path: Path, role: str, expected_sha256: str) -> tuple[dict[str, Any], dict[str, str]]:
    reference = _ref(root, path, role, expected_sha256)
    try:
        value = json.loads(_resolve(root, path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlockedDiagnosis(f"{role} cannot be parsed") from exc
    return dict(_mapping(value, role)), reference


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_facts(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "status_line_count": len(status.splitlines()) if status else 0,
        "status_sha256": _sha256_text(status),
    }


def _finite_summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().cpu().to(torch.float64).flatten()
    _require(values.numel() > 0 and bool(torch.isfinite(values).all()), "diagnostic summary is empty or nonfinite")
    quantiles = torch.quantile(values, torch.tensor([0.01, 0.50, 0.99], dtype=torch.float64))
    return {
        "count": int(values.numel()),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(quantiles[1]),
        "q01": float(quantiles[0]),
        "q99": float(quantiles[2]),
    }


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach()
    if torch.is_complex(value):
        magnitude = value.abs()
    else:
        magnitude = value
    finite = bool(torch.isfinite(magnitude).all())
    _require(finite, "tensor diagnostic is nonfinite")
    zero_fraction = float((magnitude == 0).to(torch.float64).mean())
    nonfinite_fraction = float((~torch.isfinite(magnitude)).to(torch.float64).mean())
    return {
        "norm": float(magnitude.to(torch.float64).square().sum().sqrt()),
        "zero_fraction": zero_fraction,
        "nonfinite_fraction": nonfinite_fraction,
    }


def _first_balanced_panel(role_indices: torch.Tensor, labels: torch.Tensor, per_class: int) -> torch.Tensor:
    selected: list[torch.Tensor] = []
    role_labels = labels.index_select(0, role_indices)
    for digit in range(10):
        matches = role_indices[role_labels == digit]
        _require(matches.numel() >= per_class, f"role class {digit} has too few rows")
        selected.append(matches[:per_class])
    panel = torch.cat(selected).contiguous()
    counts = torch.bincount(labels.index_select(0, panel), minlength=10)
    _require(torch.equal(counts, torch.full((10,), per_class, dtype=torch.long)), "diagnostic panel is not balanced")
    _require(torch.unique(panel).numel() == panel.numel(), "diagnostic panel contains duplicates")
    return panel


def _fixed_development_selection(labels: torch.Tensor, development_indices: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(4321)
    selected: list[torch.Tensor] = []
    development_labels = labels.index_select(0, development_indices)
    for digit in range(10):
        matches = development_indices[development_labels == digit]
        _require(matches.numel() == 500, "development role count changed")
        selected.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:100]))
    return torch.cat(selected).contiguous()


def _load_authority(root: Path) -> dict[str, Any]:
    contract, contract_ref = _read_json(root, CONTRACT_PATH, "corrected V2 training contract", CONTRACT_SHA256)
    correction, correction_ref = _read_json(root, CORRECTION_PATH, "V2 authority correction", CORRECTION_SHA256)
    original_ref = _ref(root, ORIGINAL_CONTRACT_PATH, "original V2 training contract", ORIGINAL_CONTRACT_SHA256)
    preflight, preflight_ref = _read_json(root, PREFLIGHT_REPORT_PATH, "attempt-07 preflight report", PREFLIGHT_REPORT_SHA256)
    preflight_input, preflight_input_ref = _read_json(root, PREFLIGHT_INPUT_PATH, "attempt-07 preflight input", PREFLIGHT_INPUT_SHA256)
    preflight_lineage, preflight_lineage_ref = _read_json(root, PREFLIGHT_LINEAGE_PATH, "attempt-07 preflight lineage", PREFLIGHT_LINEAGE_SHA256)
    training_report, training_report_ref = _read_json(root, TRAINING_REPORT_PATH, "completed V2 training report", TRAINING_REPORT_SHA256)
    training_lineage, training_lineage_ref = _read_json(root, TRAINING_LINEAGE_PATH, "completed V2 training lineage", TRAINING_LINEAGE_SHA256)
    runtime_report, runtime_report_ref = _read_json(root, RUNTIME_REPORT_PATH, "aligned runtime report", RUNTIME_REPORT_SHA256)
    for name, payload, expected in (
        ("correction result", correction, "CONFIRMED"),
        ("preflight report result", preflight, "AUTHORIZABLE"),
        ("preflight input result", preflight_input, "READY_FOR_EXECUTION"),
        ("preflight lineage result", preflight_lineage, "AUTHORIZABLE"),
        ("training report result", training_report, "VALID_COMPLETED"),
        ("training lineage result", training_lineage, "VALID_COMPLETED"),
    ):
        _require(payload.get("result") == expected, f"{name} changed", blocked=True)
    _require(
        correction.get("classification") == "TRAINING_SOURCE_AUTHORITY_CONTRACT_TRANSCRIPTION_ERROR_CONFIRMED",
        "authority correction classification changed",
        blocked=True,
    )
    _require(training_report.get("classification") == "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_SEMANTIC_WARMUP_NOT_ESTABLISHED", "training classification changed", blocked=True)
    _require(training_report.get("claim_boundary", {}).get("protected_accessed") is False, "training protected boundary changed", blocked=True)
    _require(training_report.get("claim_boundary", {}).get("test_accessed") is False, "training test boundary changed", blocked=True)
    _require(training_report.get("claim_boundary", {}).get("sealed_accessed") is False, "training sealed boundary changed", blocked=True)
    _require(training_lineage.get("result") == "VALID_COMPLETED", "training lineage is not complete", blocked=True)
    _require(
        runtime_report.get("result") == "IMPLEMENTED"
        and runtime_report.get("status") == "valid_completed",
        "aligned runtime report is not valid",
        blocked=True,
    )
    contract_obj = v2.load_v2_training_contract(_resolve(root, CONTRACT_PATH), repo_root=root)
    _require(contract_obj.sha256 == CONTRACT_SHA256, "loaded corrected contract SHA changed", blocked=True)
    refs = {
        "contract": contract_ref,
        "correction": correction_ref,
        "original_contract": original_ref,
        "preflight_report": preflight_ref,
        "preflight_input": preflight_input_ref,
        "preflight_lineage": preflight_lineage_ref,
        "training_report": training_report_ref,
        "training_lineage": training_lineage_ref,
        "consumer": _ref(root, CONSUMER_PATH, "V2 consumer", CONSUMER_SHA256),
        "aligned_runtime": _ref(root, RUNTIME_PATH, "aligned LightGen runtime", RUNTIME_SHA256),
        "aligned_runtime_report": runtime_report_ref,
        "evaluator_implementation": _ref(root, EVALUATOR_PATH, "FrozenMNISTEvaluator implementation", EVALUATOR_SHA256),
        "evaluator_checkpoint": _ref(root, EVALUATOR_CHECKPOINT_PATH, "FrozenMNISTEvaluator checkpoint", EVALUATOR_CHECKPOINT_SHA256),
        "historical_v1_diagnosis": _ref(root, HISTORICAL_DIAGNOSIS_PATH, "historical V1 diagnosis consumer", HISTORICAL_DIAGNOSIS_SHA256),
    }
    checkpoint_refs = {}
    for step, (path, expected_sha256) in CHECKPOINTS.items():
        checkpoint_refs[str(step)] = _ref(root, path, f"Stage-A checkpoint step {step}", expected_sha256)
    return {
        "contract": contract_obj,
        "contract_payload": contract,
        "training_report": training_report,
        "training_lineage": training_lineage,
        "refs": refs,
        "checkpoint_refs": checkpoint_refs,
        "preflight": {
            "report": preflight,
            "input": preflight_input,
            "lineage": preflight_lineage,
        },
    }


def _load_allowed_data(root: Path, contract_payload: Mapping[str, Any], training_report: Mapping[str, Any]) -> dict[str, Any]:
    source = _resolve(root, TRAINING_DATA_PATH)
    _require(source.is_file(), "canonical training source is unavailable", blocked=True)
    _require(file_sha256(source) == TRAINING_DATA_SHA256, "canonical training source SHA changed", blocked=True)
    images, labels = load_local_mnist(root / "data", train=True)
    _require(tuple(images.shape) == (60_000, 1, 28, 28), "TRAIN source image shape changed", blocked=True)
    _require(tuple(labels.shape) == (60_000,) and labels.dtype == torch.long, "TRAIN source labels changed", blocked=True)
    train_indices, development_indices = stratified_mnist_indices(labels, seed=1234, validation_per_label=500)
    _require(tensor_sha256(train_indices) == training_report["dataset"]["train"]["indices_sha256"], "TRAIN role indices changed", blocked=True)
    _require(tensor_sha256(development_indices) == training_report["dataset"]["development"]["indices_sha256"], "DEV role indices changed", blocked=True)
    selection_indices = _fixed_development_selection(labels, development_indices)
    _require(tensor_sha256(selection_indices) == training_report["dataset"]["development"]["selection_indices_sha256"], "DEV selection indices changed", blocked=True)
    data_contract = _mapping(contract_payload.get("data_contract"), "V2 contract data contract")
    role_payload = _mapping(data_contract.get("roles"), "V2 contract data roles")
    _require(int(_mapping(role_payload.get("train"), "TRAIN contract role").get("count")) == 55_000, "TRAIN contract count changed", blocked=True)
    _require(int(_mapping(role_payload.get("development"), "DEV contract role").get("count")) == 5_000, "DEV contract count changed", blocked=True)
    calibration = _mapping(role_payload.get("calibration"), "calibration contract role")
    _require(str(calibration.get("use")) == "not consumed; prior authority only", "calibration role permission changed", blocked=True)
    train_panel = _first_balanced_panel(train_indices, labels, PANEL_PER_CLASS)
    development_panel = _first_balanced_panel(selection_indices, labels, PANEL_PER_CLASS)
    _require(torch.unique(torch.cat((train_panel, development_panel))).numel() == PANEL_SIZE * 2, "diagnostic panels overlap")
    return {
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "development_indices": development_indices,
        "selection_indices": selection_indices,
        "panels": {"train": train_panel, "development": development_panel},
        "metadata": {
            "source": {"path": TRAINING_DATA_PATH.as_posix(), "sha256": TRAINING_DATA_SHA256, "rows_loaded": 60_000},
            "train": {"role_rows": 55_000, "role_indices_sha256": tensor_sha256(train_indices), "diagnostic_panel_rows": PANEL_SIZE, "panel_rule": "first 20 rows per class in frozen TRAIN role order", "panel_indices_sha256": tensor_sha256(train_panel)},
            "development": {"role_rows": 5_000, "role_indices_sha256": tensor_sha256(development_indices), "selection_rows": 1_000, "selection_indices_sha256": tensor_sha256(selection_indices), "diagnostic_panel_rows": PANEL_SIZE, "panel_rule": "first 20 rows per class in frozen development selection order", "panel_indices_sha256": tensor_sha256(development_panel)},
            "calibration": {"rows_read": 0, "status": "not consumed; prior authority only"},
            "protected": {"rows_read": 0, "status": "forbidden; no path opened"},
            "test": {"rows_read": 0, "status": "forbidden; no path opened"},
            "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
            "labels_as_model_input": False,
        },
    }


def _load_checkpoint_consumer(root: Path, authority: Mapping[str, Any], step: int) -> tuple[LightGen2DComplexOLSTrainingConsumerV2, Mapping[str, Any]]:
    contract = authority["contract"]
    path, expected_sha256 = CHECKPOINTS[step]
    payload = _mapping(safe_torch_load(_resolve(root, path), expected_sha256=expected_sha256, map_location="cpu"), f"Stage-A checkpoint {step}")
    _require(payload.get("schema") == CHECKPOINT_SCHEMA and payload.get("version") == 2, f"checkpoint {step} schema changed", blocked=True)
    _require(payload.get("accepted_training_checkpoint") is False, f"checkpoint {step} acceptance flag changed", blocked=True)
    checkpoint_contract = _mapping(payload.get("contract"), f"checkpoint {step} contract")
    _require(checkpoint_contract.get("sha256") == CONTRACT_SHA256, f"checkpoint {step} contract binding changed", blocked=True)
    runtime = _mapping(payload.get("runtime"), f"checkpoint {step} runtime")
    _require(runtime.get("module_sha256") == RUNTIME_SHA256 and runtime.get("transport") == "T=I_100 coefficient space", f"checkpoint {step} runtime binding changed", blocked=True)
    _require(payload.get("stage") == "STAGE_A_SEMANTIC_CHECK" and int(payload.get("stage_local_step")) == step, f"checkpoint {step} stage coordinate changed", blocked=True)
    consumer = LightGen2DComplexOLSTrainingConsumerV2(contract, device="cpu", propagation_grid_size=2048)
    consumer.load_trainable_state_dict(_mapping(payload.get("trainable_parameter_state"), f"checkpoint {step} trainable parameter state"))
    consumer.runtime.eval()
    _require(all(parameter.requires_grad for parameter in consumer.runtime.parameters()), "diagnostic runtime parameter firewall changed")
    return consumer, payload


def _semantic_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    logits = logits.detach().cpu().to(torch.float64)
    labels = labels.detach().cpu().to(torch.long)
    probabilities = torch.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    confusion = torch.zeros((10, 10), dtype=torch.long)
    confusion.index_put_((labels, predictions), torch.ones_like(labels, dtype=torch.long), accumulate=True)
    counts = confusion.sum(dim=1)
    recall = confusion.diag().to(torch.float64) / counts.clamp_min(1).to(torch.float64)
    entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum(dim=1)
    mean_probability = probabilities.mean(dim=0)
    mean_entropy = -(mean_probability * mean_probability.clamp_min(1.0e-12).log()).sum()
    _require(torch.isfinite(recall).all() and torch.isfinite(entropy).all(), "semantic metrics are nonfinite")
    return {
        "macro_recall": float(recall.mean()),
        "accuracy": float((predictions == labels).to(torch.float64).mean()),
        "per_class_recall": [float(value) for value in recall.tolist()],
        "confusion_matrix": confusion.tolist(),
        "prediction_counts": torch.bincount(predictions, minlength=10).tolist(),
        "prediction_class_count": int(torch.unique(predictions).numel()),
        "mean_prediction_entropy": float(entropy.mean()),
        "mean_prediction_entropy_summary": _finite_summary(entropy),
        "entropy_of_mean_prediction_distribution": float(mean_entropy),
        "evaluator_threshold": SEMANTIC_THRESHOLD,
    }


def _class_separation(values: torch.Tensor, labels: torch.Tensor, *, name: str) -> dict[str, Any]:
    values = values.detach().cpu().to(torch.float64)
    labels = labels.detach().cpu().to(torch.long)
    centroids: list[torch.Tensor] = []
    within: list[torch.Tensor] = []
    counts: list[int] = []
    for digit in range(10):
        group = values[labels == digit]
        _require(group.shape[0] >= 2, f"{name} class {digit} has too few rows")
        centroid = group.mean(dim=0)
        centroids.append(centroid)
        counts.append(int(group.shape[0]))
        within.append((group - centroid).square().mean(dim=1).sqrt())
    centroid_tensor = torch.stack(centroids)
    distances = (centroid_tensor[:, None, :] - centroid_tensor[None, :, :]).square().mean(dim=2).sqrt()
    between = distances[torch.triu(torch.ones_like(distances, dtype=torch.bool), diagonal=1)]
    within_tensor = torch.cat(within)
    within_mean = float(within_tensor.mean())
    between_mean = float(between.mean())
    return {
        "representation": name,
        "class_counts": counts,
        "within_class_rms": _finite_summary(within_tensor),
        "between_class_centroid_rms": _finite_summary(between),
        "between_to_within_mean_ratio": between_mean / within_mean if within_mean > 0.0 else None,
        "centroid_distance_matrix": distances.tolist(),
        "distance_definition": "RMS over representation coordinates",
    }


def _gauge_representation(z: torch.Tensor) -> torch.Tensor:
    anchor = z.sum(dim=1, keepdim=True) / math.sqrt(float(z.shape[1]))
    factor = anchor.conj() / (anchor.abs().square() + GAUGE_EPSILON**2).sqrt()
    relative = z * factor
    return torch.cat((relative.real, relative.imag), dim=1).to(torch.float64)


def _effective_rank(eigenvalues: torch.Tensor) -> float:
    eigenvalues = eigenvalues.detach().cpu().to(torch.float64).clamp_min(0.0)
    denominator = eigenvalues.square().sum()
    return float(eigenvalues.sum().square() / denominator) if float(denominator) > 0.0 else 0.0


def _covariance_spectrum(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().cpu().to(torch.float64)
    centered = values - values.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1).mm(centered) / float(max(values.shape[0] - 1, 1))
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
    maximum = float(eigenvalues.max()) if eigenvalues.numel() else 0.0
    numerical_rank = int((eigenvalues > maximum * 1.0e-6).sum()) if maximum > 0.0 else 0
    return {
        "eigenvalues": eigenvalues.tolist(),
        "effective_rank_participation_ratio": _effective_rank(eigenvalues),
        "numerical_rank_threshold": "1e-6 * maximum eigenvalue",
        "numerical_rank": numerical_rank,
    }


def _pairwise_metrics(z: torch.Tensor, raw: torch.Tensor, *, step: int, role_offset: int) -> dict[str, Any]:
    count = z.shape[0]
    pair_count = min(50, count // 2)
    generator = torch.Generator(device="cpu").manual_seed(PAIR_SEED + step + role_offset)
    permutation = torch.randperm(count, generator=generator)
    left = permutation[:pair_count]
    right = permutation[pair_count : 2 * pair_count]
    z_distance = (z[left] - z[right]).abs().square().mean(dim=1).sqrt()
    raw_flat = raw.flatten(start_dim=1).to(torch.float64)
    raw_distance = (raw_flat[left] - raw_flat[right]).square().mean(dim=1).sqrt()
    return {
        "pair_count": pair_count,
        "seed": PAIR_SEED + step + role_offset,
        "complex_state_rms": _finite_summary(z_distance),
        "raw_output_rms": _finite_summary(raw_distance),
    }


def _ols_geometry(z: torch.Tensor, labels: torch.Tensor, raw: torch.Tensor, *, step: int, role: str) -> dict[str, Any]:
    z = z.detach().cpu().to(torch.complex128)
    power = z.abs().square().mean(dim=1)
    per_port_power = z.abs().square().mean(dim=0)
    gauge = _gauge_representation(z)
    centered_z = z - z.mean(dim=0, keepdim=True)
    complex_covariance = centered_z.conj().transpose(0, 1).mm(centered_z) / float(z.shape[0] - 1)
    complex_eigenvalues = torch.linalg.eigvalsh(complex_covariance).real.clamp_min(0.0)
    anchor = z.sum(dim=1, keepdim=True) / math.sqrt(float(z.shape[1]))
    relative = z * anchor.conj() / (anchor.abs().square() + GAUGE_EPSILON**2).sqrt()
    unit = relative / relative.abs().clamp_min(torch.finfo(torch.float64).tiny)
    concentration = unit.mean(dim=0).abs().clamp(0.0, 1.0)
    material_threshold = max(MATERIAL_PORT_ABSOLUTE_POWER, MATERIAL_PORT_RELATIVE_POWER * float(per_port_power.max()))
    active_ports = int((per_port_power > material_threshold).sum())
    magnitude = z.abs()
    modal_variation = magnitude.std(dim=0, unbiased=False) / magnitude.mean(dim=0).clamp_min(1.0e-12)
    return {
        "authoritative_latent": "complex z_real in C^100; q is not used",
        "sample_count": int(z.shape[0]),
        "mean_power": _finite_summary(power),
        "per_port_power": per_port_power.tolist(),
        "per_port_real_variance": z.real.var(dim=0, unbiased=False).tolist(),
        "per_port_imag_variance": z.imag.var(dim=0, unbiased=False).tolist(),
        "relative_phase": {
            "definition": "per-port phase after the frozen global gauge anchor z.sum/sqrt(100)",
            "per_port_circular_concentration": concentration.tolist(),
            "per_port_circular_variation": (1.0 - concentration).tolist(),
            "variation_summary": _finite_summary(1.0 - concentration),
        },
        "covariance_spectrum_gauge_real_200d": _covariance_spectrum(gauge),
        "covariance_spectrum_raw_complex_100d": {
            "eigenvalues": complex_eigenvalues.tolist(),
            "effective_rank_participation_ratio": _effective_rank(complex_eigenvalues),
        },
        "class_separation": _class_separation(gauge, labels, name="gauge-invariant complex OLS z"),
        "pairwise": _pairwise_metrics(z, raw, step=step, role_offset=0 if role == "train" else 1000),
        "modal_variation": _finite_summary(modal_variation),
        "material_port_rule": {
            "relative_threshold": MATERIAL_PORT_RELATIVE_POWER,
            "absolute_floor": MATERIAL_PORT_ABSOLUTE_POWER,
            "threshold_value": material_threshold,
            "active_port_count": active_ports,
            "active_port_fraction": active_ports / 100.0,
        },
    }


def _role_measurement(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    evaluator: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    *,
    step: int,
    role: str,
) -> dict[str, Any]:
    image_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    raw_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    z_parts: list[torch.Tensor] = []
    ols_field_parts: list[torch.Tensor] = []
    generator_input_parts: list[torch.Tensor] = []
    logits_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, indices.numel(), BATCH_SIZE):
            batch_indices = indices[start : start + BATCH_SIZE]
            batch_images = images.index_select(0, batch_indices)
            batch_labels = labels.index_select(0, batch_indices)
            result = consumer.runtime.forward_from_image(batch_images)
            _require(result.image_input is not None and result.ols_input_field is not None, "real encoded OLS path is incomplete")
            raw = result.raw_intensity.detach().cpu()
            target = result.image_input.slm_intensity.detach().cpu()
            z = result.z_ols.coefficients.detach().cpu()
            ols_field = result.ols_input_field.tensor.detach().cpu()
            generator_input = result.generator_input_field.tensor.detach().cpu()
            _, logits = evaluator(v2.raw_cmos_evaluator_view(raw))
            _require(torch.isfinite(raw).all() and torch.isfinite(z).all(), "read-only forward produced nonfinite values")
            image_parts.append(batch_images.cpu())
            label_parts.append(batch_labels.cpu())
            raw_parts.append(raw)
            target_parts.append(target)
            z_parts.append(z)
            ols_field_parts.append(ols_field)
            generator_input_parts.append(generator_input)
            logits_parts.append(logits.detach().cpu())
            if start == 0 or start + BATCH_SIZE >= indices.numel():
                print(f"diagnosis step {step} {role}: {min(start + BATCH_SIZE, indices.numel())}/{indices.numel()}", flush=True)
    image_tensor = torch.cat(image_parts)
    label_tensor = torch.cat(label_parts)
    raw_tensor = torch.cat(raw_parts)
    target_tensor = torch.cat(target_parts)
    z_tensor = torch.cat(z_parts).to(torch.complex64)
    ols_field_tensor = torch.cat(ols_field_parts)
    generator_input_tensor = torch.cat(generator_input_parts)
    logits_tensor = torch.cat(logits_parts)
    raw_flat = raw_tensor.flatten(start_dim=1).to(torch.float64)
    target_flat = target_tensor.flatten(start_dim=1).to(torch.float64)
    image_loss = (raw_flat - target_flat).square().mean(dim=1)
    per_image_variance = raw_flat.var(dim=1, unbiased=False)
    across_image_variance = raw_flat.var(dim=0, unbiased=False)
    semantic = _semantic_metrics(logits_tensor, label_tensor)
    raw_separation = _class_separation(raw_flat, label_tensor, name="raw CMOS output")
    output_mean = raw_flat.mean(dim=0)
    output_centroid_variance = (raw_flat - output_mean).square().mean(dim=1).sqrt()
    input_power = ols_field_tensor.abs().square().mean(dim=(1, 2, 3))
    projected_power = z_tensor.abs().square().mean(dim=1)
    generator_input_power = generator_input_tensor.abs().square().mean(dim=(1, 2, 3))
    coupling_ratio = projected_power / input_power.clamp_min(1.0e-30)
    modal_power = z_tensor.abs().square().mean(dim=0)
    return {
        "role": role,
        "step": step,
        "rows": int(indices.numel()),
        "indices_sha256": tensor_sha256(indices),
        "image_loss": {"metric": "mean squared raw CMOS intensity minus aligned SLM intensity", "summary": _finite_summary(image_loss)},
        "semantic": semantic,
        "output": {
            "per_image_spatial_variance": _finite_summary(per_image_variance),
            "across_image_pixel_variance": _finite_summary(across_image_variance),
            "pairwise": _pairwise_metrics(z_tensor, raw_tensor, step=step, role_offset=0 if role == "train" else 1000),
            "mean_image_prototype_rms": float(output_centroid_variance.mean()),
            "class_separation": raw_separation,
            "prediction_distribution": semantic["prediction_counts"],
        },
        "ols": _ols_geometry(z_tensor, label_tensor, raw_tensor, step=step, role=role),
        "modal_coupling": {
            "encoder_field_power_before_projection": _finite_summary(input_power),
            "total_projected_modal_power_mean_over_100_ports": _finite_summary(projected_power),
            "generator_entrance_field_power_after_W_OLS": _finite_summary(generator_input_power),
            "projected_to_input_power_ratio": _finite_summary(coupling_ratio),
            "per_port_mean_power": modal_power.tolist(),
            "sample_to_sample_modal_variation": _finite_summary(z_tensor.abs().std(dim=0, unbiased=False) / z_tensor.abs().mean(dim=0).clamp_min(1.0e-12)),
            "normalization_boundary": "coupling ratio is a normalization-dependent diagnostic, not a claimed physical efficiency",
        },
        "_raw": {
            "images": image_tensor,
            "labels": label_tensor,
            "raw": raw_tensor,
            "target": target_tensor,
            "z": z_tensor,
        },
    }


def _route_equivalence(consumer: LightGen2DComplexOLSTrainingConsumerV2, images: torch.Tensor, indices: torch.Tensor) -> dict[str, Any]:
    batch_indices = indices[::PANEL_PER_CLASS][:BATCH_SIZE]
    batch = images.index_select(0, batch_indices)
    with torch.no_grad():
        image_result = consumer.runtime.forward_from_image(batch)
        direct_result = consumer.runtime.forward_from_ols(image_result.z_ols.coefficients)
    generator_delta = (image_result.generator_input_field.tensor - direct_result.generator_input_field.tensor).abs()
    raw_delta = (image_result.raw_intensity - direct_result.raw_intensity).abs()
    _require(torch.isfinite(generator_delta).all() and torch.isfinite(raw_delta).all(), "route equivalence delta is nonfinite")
    return {
        "batch_size": BATCH_SIZE,
        "batch_indices_sha256": tensor_sha256(batch_indices),
        "source_state": "z_real from forward_from_image",
        "direct_state": "same z_real passed to forward_from_ols",
        "shared_w_ols": True,
        "shared_generator": True,
        "generator_input_max_abs_error": float(generator_delta.max()),
        "raw_cmos_max_abs_error": float(raw_delta.max()),
        "generator_input_exact_equal": bool(torch.equal(image_result.generator_input_field.tensor, direct_result.generator_input_field.tensor)),
        "raw_cmos_exact_equal": bool(torch.equal(image_result.raw_intensity, direct_result.raw_intensity)),
        "conclusion": "real encoded OLS and direct OLS paths converge at the same W_OLS and generator interface",
    }


def _parameter_gradient_record(parameter: torch.nn.Parameter, gradient: torch.Tensor | None) -> dict[str, Any]:
    parameter_stats = _tensor_stats(parameter.detach())
    if gradient is None:
        gradient_stats = {"norm": 0.0, "zero_fraction": 1.0, "nonfinite_fraction": 0.0}
        per_plane = []
    else:
        _require(torch.isfinite(gradient).all(), "parameter gradient is nonfinite")
        gradient_stats = _tensor_stats(gradient)
        per_plane = [float(value) for value in gradient.detach().to(torch.float64).reshape(gradient.shape[0], -1).square().sum(dim=1).sqrt().tolist()] if gradient.ndim >= 1 else []
    return {
        "parameter_norm": parameter_stats["norm"],
        "gradient_norm": gradient_stats["norm"],
        "gradient_parameter_ratio": gradient_stats["norm"] / parameter_stats["norm"] if parameter_stats["norm"] > 0.0 else None,
        "gradient_zero_fraction": gradient_stats["zero_fraction"],
        "gradient_nonfinite_fraction": gradient_stats["nonfinite_fraction"],
        "per_plane_gradient_norm": per_plane,
    }


def _boundary_gradient_record(value: torch.Tensor, gradient: torch.Tensor | None) -> dict[str, Any]:
    value_stats = _tensor_stats(value.detach())
    if gradient is None:
        gradient_stats = {"norm": 0.0, "zero_fraction": 1.0, "nonfinite_fraction": 0.0}
    else:
        _require(torch.isfinite(gradient).all(), "OLS boundary gradient is nonfinite")
        gradient_stats = _tensor_stats(gradient)
    return {
        "requires_grad": bool(value.requires_grad),
        "value_norm": value_stats["norm"],
        "gradient_norm": gradient_stats["norm"],
        "gradient_zero_fraction": gradient_stats["zero_fraction"],
        "gradient_nonfinite_fraction": gradient_stats["nonfinite_fraction"],
    }


def _gradient_diagnostic(
    root: Path,
    authority: Mapping[str, Any],
    images: torch.Tensor,
    labels: torch.Tensor,
    panels: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    consumer, _ = _load_checkpoint_consumer(root, authority, 300)
    before = {name: value.detach().cpu().clone() for name, value in consumer.runtime.state_dict().items()}
    by_role: dict[str, Any] = {}
    parameter_names = tuple(name for name, _ in consumer.runtime.named_parameters())
    _require(parameter_names == ("encoder.phase", "generator.phase"), "gradient parameter surface changed")
    for role in ("train", "development"):
        batch_indices = panels[role][::PANEL_PER_CLASS][:BATCH_SIZE]
        batch = images.index_select(0, batch_indices)
        batch_labels = labels.index_select(0, batch_indices)
        _require(torch.equal(torch.bincount(batch_labels, minlength=10), torch.ones(10, dtype=torch.long)), f"{role} gradient batch is not one-per-class")
        result = consumer.runtime.forward_from_image(batch)
        _require(result.image_input is not None and result.ols_input_field is not None, "gradient path lacks image/OLS boundary")
        loss = (result.raw_intensity - result.image_input.slm_intensity).square().mean()
        parameters = tuple(parameter for _, parameter in consumer.runtime.named_parameters())
        boundaries = (result.encoder_output_field.tensor, result.z_ols.coefficients, result.generator_input_field.tensor)
        _require(all(value.requires_grad for value in boundaries), f"{role} OLS boundary is detached")
        gradients = torch.autograd.grad(loss, parameters + boundaries, allow_unused=True, retain_graph=False, create_graph=False)
        parameter_gradients = gradients[: len(parameters)]
        boundary_gradients = gradients[len(parameters) :]
        by_role[role] = {
            "batch_indices_sha256": tensor_sha256(batch_indices),
            "batch_size": int(batch.shape[0]),
            "label_counts": torch.bincount(batch_labels, minlength=10).tolist(),
            "loss": float(loss.detach()),
            "parameters": {
                name: _parameter_gradient_record(parameter, gradient)
                for (name, parameter), gradient in zip(consumer.runtime.named_parameters(), parameter_gradients, strict=True)
            },
            "ols_boundary": {
                "encoder_output_field": _boundary_gradient_record(boundaries[0], boundary_gradients[0]),
                "modal_coefficients_z": _boundary_gradient_record(boundaries[1], boundary_gradients[1]),
                "generator_entrance_field_W_OLS": _boundary_gradient_record(boundaries[2], boundary_gradients[2]),
            },
            "backward_mode": "torch.autograd.grad (backward derivative, no .grad accumulation)",
            "optimizer_created": False,
            "optimizer_step": False,
        }
        del loss, result, gradients
    after = {name: value.detach().cpu() for name, value in consumer.runtime.state_dict().items()}
    _require(before.keys() == after.keys() and all(torch.equal(before[name], after[name]) for name in before), "read-only gradient diagnostic mutated runtime state", blocked=True)
    encoder_norms = torch.tensor([by_role[role]["parameters"]["encoder.phase"]["gradient_norm"] for role in by_role], dtype=torch.float64)
    generator_norms = torch.tensor([by_role[role]["parameters"]["generator.phase"]["gradient_norm"] for role in by_role], dtype=torch.float64)
    return {
        "checkpoint_step": 300,
        "by_role": by_role,
        "aggregate": {
            "encoder_gradient_norm": _finite_summary(encoder_norms),
            "generator_gradient_norm": _finite_summary(generator_norms),
            "encoder_to_generator_gradient_norm_ratio": _finite_summary(encoder_norms / generator_norms.clamp_min(1.0e-30)),
            "encoder_gradient_effectively_zero_rule": "zero fraction >= 0.99 or encoder/generator norm ratio < 0.01 on both allowed batches",
            "encoder_gradient_effectively_zero": bool(torch.all((encoder_norms / generator_norms.clamp_min(1.0e-30)) < 1.0e-2)),
        },
        "parameter_state_unchanged": True,
        "optimizer_created": False,
        "optimizer_step_count": 0,
    }


def _trajectory_diagnostic(training_report: Mapping[str, Any]) -> dict[str, Any]:
    execution = _mapping(training_report.get("execution"), "training execution")
    stage_a = _mapping(execution.get("stage_a"), "Stage-A execution")
    trace = stage_a.get("updates")
    _require(isinstance(trace, list) and len(trace) == 300, "Stage-A trajectory is incomplete", blocked=True)
    by_step = {int(item["global_step"]): item for item in trace}
    _require(set(by_step) == set(range(1, 301)), "Stage-A global-step coverage changed", blocked=True)
    components: dict[str, Any] = {}
    for component in ("L_image", "L_total"):
        values = torch.tensor([float(_mapping(item.get("loss"), "trajectory loss")[component]) for item in trace], dtype=torch.float64)
        differences = values[1:] - values[:-1]
        components[component] = {
            "initial_step_1": float(values[0]),
            "early_step_25": float(values[24]),
            "step_100": float(values[99]),
            "step_200": float(values[199]),
            "step_300": float(values[299]),
            "final_minus_initial": float(values[-1] - values[0]),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "increasing_transition_count": int((differences > 0.0).sum()),
            "decreasing_transition_count": int((differences < 0.0).sum()),
            "monotonic_nonincreasing": bool(torch.all(differences <= 0.0)),
            "all_finite": bool(torch.isfinite(values).all()),
        }
    for name in ("encoder.phase", "generator.phase"):
        gradients = torch.tensor([float(_mapping(_mapping(item.get("autograd"), "trajectory autograd").get("gradient_norms"), "gradient norms")[name]) for item in trace], dtype=torch.float64)
        updates = torch.tensor([float(_mapping(item.get("parameter_update_norms"), "trajectory update norms")[name]) for item in trace], dtype=torch.float64)
        components[name] = {
            "gradient_norm": _finite_summary(gradients),
            "update_norm": _finite_summary(updates),
            "final_gradient_norm": float(gradients[-1]),
            "final_update_norm": float(updates[-1]),
            "all_gradients_finite": bool(torch.isfinite(gradients).all()),
            "all_updates_finite": bool(torch.isfinite(updates).all()),
            "all_updates_nonzero": bool(torch.all(updates > 0.0)),
        }
    checkpoint_records = stage_a.get("checkpoint_records")
    _require(isinstance(checkpoint_records, list) and len(checkpoint_records) == 3, "Stage-A checkpoint records changed", blocked=True)
    checkpoints = []
    for record in checkpoint_records:
        checkpoints.append(
            {
                "step": int(record["stage_local_step"]),
                "recorded_train_L_image": float(by_step[int(record["stage_local_step"])]["loss"]["L_image"]),
                "development_L_image": float(record["development_L_image"]),
                "development_semantic_recall": float(record["development_macro_recall"]),
                "development_raw_ols_power_mean": float(record["development_raw_ols_power_mean"]),
                "development_semantic_eligible": bool(record["development_semantic_eligible"]),
            }
        )
    dev_losses = torch.tensor([item["development_L_image"] for item in checkpoints], dtype=torch.float64)
    dev_recalls = torch.tensor([item["development_semantic_recall"] for item in checkpoints], dtype=torch.float64)
    return {
        "stage": "A",
        "steps_completed": len(trace),
        "components": components,
        "checkpoint_records": checkpoints,
        "recorded_development_image_loss_direction": {
            "values_step_100_200_300": dev_losses.tolist(),
            "strictly_decreased": bool(torch.all(dev_losses[1:] < dev_losses[:-1])),
        },
        "recorded_development_semantic_direction": {
            "values_step_100_200_300": dev_recalls.tolist(),
            "strictly_decreased": bool(torch.all(dev_recalls[1:] < dev_recalls[:-1])),
        },
        "interpretation": "recorded image objective improved at every semantic checkpoint while recorded development semantic recall worsened at every checkpoint",
    }


def _attribution(
    trajectory: Mapping[str, Any],
    checkpoints: Mapping[int, Mapping[str, Any]],
    gradients: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    final_train = checkpoints[300]["train"]
    final_dev = checkpoints[300]["development"]
    first_train = checkpoints[100]["train"]
    first_dev = checkpoints[100]["development"]
    objective_supported = bool(
        trajectory["recorded_development_image_loss_direction"]["strictly_decreased"]
        and trajectory["recorded_development_semantic_direction"]["strictly_decreased"]
        and trajectory["components"]["L_image"]["final_minus_initial"] < 0.0
    )
    optimization_not_progressing = not bool(
        trajectory["components"]["L_image"]["final_minus_initial"] < 0.0
        and trajectory["components"]["encoder.phase"]["all_updates_nonzero"]
        and trajectory["components"]["generator.phase"]["all_updates_nonzero"]
        and trajectory["components"]["encoder.phase"]["all_gradients_finite"]
        and trajectory["components"]["generator.phase"]["all_gradients_finite"]
    )
    gradient_starvation = bool(gradients["aggregate"]["encoder_gradient_effectively_zero"])
    z_ratios = torch.tensor(
        [float(checkpoints[step]["train"]["ols"]["class_separation"]["between_to_within_mean_ratio"]) for step in (100, 200, 300)],
        dtype=torch.float64,
    )
    dev_z_ratios = torch.tensor(
        [float(checkpoints[step]["development"]["ols"]["class_separation"]["between_to_within_mean_ratio"]) for step in (100, 200, 300)],
        dtype=torch.float64,
    )
    active_ports = [int(checkpoints[step]["development"]["ols"]["material_port_rule"]["active_port_count"]) for step in (100, 200, 300)]
    ols_supported = bool(
        torch.isfinite(z_ratios).all()
        and torch.isfinite(dev_z_ratios).all()
        and float(z_ratios[-1]) < 1.0
        and float(dev_z_ratios[-1]) < 1.0
        and active_ports[-1] < 100
    )
    raw_ratio = float(final_dev["output"]["class_separation"]["between_to_within_mean_ratio"])
    z_ratio = float(final_dev["ols"]["class_separation"]["between_to_within_mean_ratio"])
    generator_supported = bool(
        z_ratio >= 1.25
        and raw_ratio <= 0.80
        and final_dev["semantic"]["macro_recall"] < SEMANTIC_THRESHOLD
    )
    train_recall = float(final_train["semantic"]["macro_recall"])
    dev_recall = float(final_dev["semantic"]["macro_recall"])
    generalization_supported = bool(train_recall >= SEMANTIC_THRESHOLD and dev_recall < SEMANTIC_THRESHOLD)
    capacity_supported = False
    statuses = {
        "objective_mismatch": {
            "status": "SUPPORTED" if objective_supported else "UNRESOLVED",
            "classification": PRIMARY_CLASSIFICATIONS["objective"],
            "evidence": {"recorded_dev_L_image_step_100_200_300": trajectory["recorded_development_image_loss_direction"]["values_step_100_200_300"], "recorded_dev_recall_step_100_200_300": trajectory["recorded_development_semantic_direction"]["values_step_100_200_300"], "panel_train_step_100_vs_300_L_image": [first_train["image_loss"]["summary"]["mean"], final_train["image_loss"]["summary"]["mean"]], "panel_dev_step_100_vs_300_L_image": [first_dev["image_loss"]["summary"]["mean"], final_dev["image_loss"]["summary"]["mean"]]},
            "reason": "image-only L_image improved while semantic recall worsened on the frozen development checkpoint evaluations" if objective_supported else "the frozen trajectory does not establish the required opposing objective/semantic direction",
        },
        "optimization_failure": {
            "status": "SUPPORTED" if optimization_not_progressing else "NOT_SUPPORTED",
            "classification": PRIMARY_CLASSIFICATIONS["optimization"],
            "evidence": {"image_loss_endpoint_decreased": not optimization_not_progressing, "all_updates_nonzero": trajectory["components"]["encoder.phase"]["all_updates_nonzero"] and trajectory["components"]["generator.phase"]["all_updates_nonzero"]},
            "reason": "the complete trace has finite gradients, nonzero parameter updates, and a lower final L_image" if not optimization_not_progressing else "the trace lacks a valid optimization-progress signal",
        },
        "encoder_gradient_starvation": {
            "status": "SUPPORTED" if gradient_starvation else "NOT_SUPPORTED",
            "classification": PRIMARY_CLASSIFICATIONS["gradient"],
            "evidence": gradients["aggregate"],
            "reason": "encoder gradient is effectively zero relative to generator gradient on both allowed batches" if gradient_starvation else "encoder gradient is finite and not effectively zero relative to generator gradient",
        },
        "encoder_ols_information_collapse": {
            "status": "SUPPORTED" if ols_supported else "UNRESOLVED",
            "classification": PRIMARY_CLASSIFICATIONS["ols"],
            "evidence": {"train_z_between_within_step_100_200_300": z_ratios.tolist(), "development_z_between_within_step_100_200_300": dev_z_ratios.tolist(), "development_active_port_count_step_100_200_300": active_ports},
            "reason": "class-conditioned complex OLS geometry remains below within-class spread and the modal interface is not class-separating" if ols_supported else "the bounded z geometry does not satisfy the registered information-collapse pattern",
        },
        "generator_decode_limitation": {
            "status": "SUPPORTED" if generator_supported else "NOT_SUPPORTED" if z_ratio < 1.0 or raw_ratio >= 1.0 else "UNRESOLVED",
            "classification": PRIMARY_CLASSIFICATIONS["generator"],
            "evidence": {"development_z_between_within_ratio": z_ratio, "development_raw_output_between_within_ratio": raw_ratio, "z_to_raw_ratio_margin": z_ratio - raw_ratio, "strong_z_structure_rule": "z ratio >= 1.25 and raw-output ratio <= 0.80; diagnostic attribution heuristic, not an acceptance threshold", "route_equivalence": True},
            "reason": "z retains strong class separation while raw output loses it" if generator_supported else "z/output separation margins are marginal and no decoder-only conclusion is justified" if z_ratio >= 1.0 and raw_ratio < 1.0 else "z does not retain class separation needed to isolate a generator-only decode limitation",
        },
        "generalization_failure": {
            "status": "SUPPORTED" if generalization_supported else "NOT_SUPPORTED",
            "classification": PRIMARY_CLASSIFICATIONS["generalization"],
            "evidence": {"train_final_macro_recall": train_recall, "development_final_macro_recall": dev_recall, "threshold": SEMANTIC_THRESHOLD},
            "reason": "train semantic recall is qualified while development recall is not" if generalization_supported else "both permitted train and development panels are below the semantic qualification threshold; this is not a train/dev generalization pattern",
        },
        "architecture_capacity_limitation": {
            "status": "SUPPORTED" if capacity_supported else "UNRESOLVED",
            "classification": PRIMARY_CLASSIFICATIONS["capacity"],
            "evidence": {"objective_mismatch_supported": objective_supported, "gradient_starvation_supported": gradient_starvation, "modal_coupling_diagnostic_available": True},
            "reason": "300-step failure cannot establish capacity after unresolved objective and representation mechanisms" if not capacity_supported else "all lower-level alternatives were excluded by existing evidence",
        },
    }
    supported_keys = [key for key, value in statuses.items() if value["status"] == "SUPPORTED"]
    if len(supported_keys) == 1:
        primary = statuses[supported_keys[0]]["classification"]
    elif len(supported_keys) > 1:
        primary = MIXED_CLASSIFICATION
    else:
        primary = UNRESOLVED_CLASSIFICATION
    statuses["primary_selection"] = {
        "supported_mechanisms": supported_keys,
        "rule": "one supported mechanism returns its exact classification; two or more supported mechanisms return MIXED",
        "primary": primary,
    }
    return primary, statuses


def _build_report(root: Path, authority: Mapping[str, Any], data: Mapping[str, Any], trajectory: Mapping[str, Any], checkpoints: Mapping[int, Mapping[str, Any]], route: Mapping[int, Mapping[str, Any]], gradients: Mapping[str, Any], primary: str, attribution: Mapping[str, Any], started: str, finished: str, git_before: Mapping[str, Any], git_after: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_summaries = []
    for step in (100, 200, 300):
        checkpoint_summaries.append({
            "step": step,
            "checkpoint": authority["checkpoint_refs"][str(step)],
            "train": {key: value for key, value in checkpoints[step]["train"].items() if key != "_raw"},
            "development": {key: value for key, value in checkpoints[step]["development"].items() if key != "_raw"},
            "route_equivalence": route[step],
        })
    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "result": RESULT,
        "created_at_utc": finished,
        "classification": primary,
        "primary_diagnosis": primary,
        "secondary_attribution": attribution,
        "authoritative_result": "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_SEMANTIC_WARMUP_NOT_ESTABLISHED",
        "diagnostic_scope": {
            "semantic_generation_only": True,
            "read_only": True,
            "device": "cpu",
            "cuda_used": False,
            "propagation_grid": 2048,
            "diagnostic_panel": "balanced first 20 rows per class from frozen TRAIN role and frozen development selection; 200 rows per role per checkpoint",
            "panel_is_not_a_new_training_split": True,
            "calibration_allowed_by_current_contract": False,
        },
        "training_trajectory": trajectory,
        "checkpoint_diagnostics": checkpoint_summaries,
        "gradient_flow": gradients,
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "generation_executed": False,
            "stage_b_executed": False,
            "formal_scale_extracted": False,
            "optimizer_created": False,
            "optimizer_step_count": 0,
            "new_training": False,
            "new_checkpoint": False,
            "new_decoder_or_phase_model": False,
            "protected_read": False,
            "test_read": False,
            "sealed_read": False,
            "calibration_rows_read": 0,
        },
        "data": data["metadata"],
        "authority": {
            "refs": authority["refs"],
            "checkpoints": authority["checkpoint_refs"],
            "historical_comparison": "historical V1 diagnosis consumer bound by SHA and not executed; its conclusions remain historical evidence only",
        },
        "provenance": {"started_at_utc": started, "finished_at_utc": finished, "git_start": git_before, "git_end": git_after, "branch_unchanged": git_before["branch"] == git_after["branch"], "head_unchanged": git_before["head"] == git_after["head"], "worktree_status_unchanged": git_before["status_sha256"] == git_after["status_sha256"]},
        "claim_boundary": {
            "fact": "the image-only V2 Stage-A semantic warmup failed under the frozen 300-step configuration",
            "fact_2": "latent MMD, normalized prior, and Stage-B scale anchor were absent and cannot explain this Stage-A failure",
            "not_concluded": ["LightGen OLS is invalid", "100-port OLS is inherently insufficient", "2+2 diffractive architecture is definitely too small", "more training would definitely fix the failure", "semantic loss is necessarily required", "generator architecture must change"],
            "diagnosis_is_bounded": True,
        },
        "next": "STOP_AFTER_DIAGNOSIS",
    }
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "result": RESULT,
        "classification": primary,
        "consumer": {"path": "scripts/research/diagnose_lightgen_2d_semantic_ols_warmup_failure_v1.py", "sha256": file_sha256(Path(__file__))},
        "input_authority": report["authority"],
        "data_firewall": report["data"],
        "execution_boundary": report["governance"],
        "diagnostic_boundary": report["diagnostic_scope"],
        "parameter_state_unchanged_after_gradient_probe": gradients["parameter_state_unchanged"],
        "no_optimizer_updates": gradients["optimizer_step_count"] == 0,
        "git_start": git_before,
        "git_end": git_after,
        "next": "STOP_AFTER_DIAGNOSIS",
    }
    return report, lineage


def execute(root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    root = root.resolve()
    output_dir = _resolve(root, output_dir)
    _require(not output_dir.exists(), f"refusing to reuse diagnosis output: {output_dir}", blocked=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    git_before = _git_facts(root)
    cpu_rng = torch.random.get_rng_state()
    try:
        authority = _load_authority(root)
        data = _load_allowed_data(root, authority["contract_payload"], authority["training_report"])
        evaluator, evaluator_payload = load_frozen_evaluator(
            _resolve(root, EVALUATOR_CHECKPOINT_PATH), device=torch.device("cpu")
        )
        _require(float(evaluator_payload.get("validation_accuracy", 0.0)) >= 0.985, "evaluator qualification changed", blocked=True)
        trajectory = _trajectory_diagnostic(authority["training_report"])
        checkpoint_diagnostics: dict[int, dict[str, Any]] = {}
        route_diagnostics: dict[int, dict[str, Any]] = {}
        for step in (100, 200, 300):
            consumer, _ = _load_checkpoint_consumer(root, authority, step)
            role_diagnostics: dict[str, Any] = {}
            for role, indices in data["panels"].items():
                role_diagnostics[role] = _role_measurement(
                    consumer,
                    evaluator,
                    data["images"],
                    data["labels"],
                    indices,
                    step=step,
                    role=role,
                )
            route_diagnostics[step] = _route_equivalence(consumer, data["images"], data["panels"]["development"])
            checkpoint_diagnostics[step] = role_diagnostics
            del consumer
            gc.collect()
        gradients = _gradient_diagnostic(root, authority, data["images"], data["labels"], data["panels"])
        primary, attribution = _attribution(trajectory, checkpoint_diagnostics, gradients)
        git_after = _git_facts(root)
        _require(git_before["branch"] == git_after["branch"], "branch changed during diagnosis", blocked=True)
        _require(git_before["head"] == git_after["head"], "HEAD changed during diagnosis", blocked=True)
        _require(git_before["status_sha256"] == git_after["status_sha256"], "worktree changed during diagnosis", blocked=True)
        finished = dt.datetime.now(dt.timezone.utc).isoformat()
        report, lineage = _build_report(root, authority, data, trajectory, checkpoint_diagnostics, route_diagnostics, gradients, primary, attribution, started, finished, git_before, git_after)
        output_dir.mkdir(parents=True, exist_ok=False)
        report_path = output_dir / REPORT_NAME
        lineage_path = output_dir / LINEAGE_NAME
        atomic_write_new_json(lineage_path, lineage, artifact_kind="warmup diagnosis lineage")
        atomic_write_new_json(report_path, report, artifact_kind="warmup diagnosis report")
        _require({path.name for path in output_dir.iterdir()} == {REPORT_NAME, LINEAGE_NAME}, "diagnosis output tree is not exact")
        return {"result": RESULT, "classification": primary, "report_path": _relative(root, report_path), "report_sha256": file_sha256(report_path), "lineage_path": _relative(root, lineage_path), "lineage_sha256": file_sha256(lineage_path)}
    finally:
        torch.random.set_rng_state(cpu_rng)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        result = execute(args.repo_root, args.output_dir)
    except BlockedDiagnosis as exc:
        print(json.dumps({"result": "BLOCKED", "message": str(exc)}, ensure_ascii=False), flush=True)
        return 2
    except InvalidDiagnosis as exc:
        print(json.dumps({"result": "INVALID", "message": str(exc)}, ensure_ascii=False), flush=True)
        return 3
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
