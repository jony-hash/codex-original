"""Build the frozen, output-domain-matched MNIST semantic evaluator.

This task consumes the registered step-5000 Scaled-Chen optical system only as
a frozen forward instrument.  It builds an evaluator from disjoint MNIST
identities and never enters optical training, prior generation, or S2 semantic
measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from torch import nn

from ldm_project.v2.application.research import optical_manifold as base
from ldm_project.v2.application.research import optical_manifold_smf_bogt as bogt
from ldm_project.v2.application.research import (
    optical_manifold_smf_bogt_manifold_readiness as mra,
)
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.conditional import fixed_exposure_view
from ldm_project.v2.student.evaluator import FrozenMNISTEvaluator
from scripts.research import audit_scaled_chen_ols_semantic_representation as s1
from scripts.research import replay_scaled_chen_repaired_coordinate_mra as rcr


TASK_ID = "H-LOMA-GEN-S2-FOSEA-BUILD"
HYPOTHESIS = TASK_ID
BRANCH = "research/v2-lightgen-optical-manifold-scaled-chen-bogt-v1"
CONFIG_SCHEMA = "lightgen_scaled_chen_frozen_output_semantic_evaluator_v1"
ARTIFACT_SCHEMA = "lightgen_scaled_chen_frozen_output_semantic_evaluator_manifest.v1"
CORPUS_SCHEMA = "lightgen_scaled_chen_frozen_output_domain_corpus.v1"
DATA_IDENTITY_SCHEMA = "lightgen_scaled_chen_frozen_output_data_identity.v1"
OUTPUT_CONTRACT_SCHEMA = "lightgen_scaled_chen_frozen_output_domain_contract.v1"
EXPOSURE_SCHEMA = "lightgen_scaled_chen_frozen_output_exposure_provenance.v1"
TRAINING_SCHEMA = "lightgen_scaled_chen_frozen_output_evaluator_training.v1"
TRAINING_METRICS_SCHEMA = "lightgen_scaled_chen_frozen_output_evaluator_training_metrics.v1"
QUALIFICATION_SCHEMA = "lightgen_scaled_chen_frozen_output_evaluator_qualification.v1"
LEAKAGE_SCHEMA = "lightgen_scaled_chen_frozen_output_evaluator_leakage.v1"
ISOLATION_SCHEMA = "lightgen_scaled_chen_frozen_output_optical_isolation.v1"
REPLAY_SCHEMA = "lightgen_scaled_chen_frozen_output_evaluator_freeze_replay.v1"
REPORT_SCHEMA = "lightgen_scaled_chen_frozen_output_evaluator_compact_report.v1"
CHECKPOINT_SCHEMA = "lightgen_scaled_chen_frozen_output_evaluator_checkpoint.v1"
CHECKPOINT_SHA256 = "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de"
S1_CONFIG_SHA256 = "60d57cd134a567c17602775ec16f11fff05f56ffe900ac730774abde3432c521"
S1_CONFIG_CANONICAL_SHA256 = "719526b8d6e82da20bdc9040314084b1192ce376d413eecdf0db07420ad2ab13"
S1_MANIFEST_SHA256 = "b0199040926618b17373edc4f167cc8bc4788f39f64de174e1a3fb225781833e"
R2_MANIFEST_SHA256 = "0cf707fe4f146c1a1c3190256fad8ec8e4bc72e33674d374e75f40b161b3fce6"
R2_PAYLOAD_SHA256 = "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07"
MRA_MANIFEST_SHA256 = "520f7743bb317fabf891734ad899c84f2cec7639dbe02f65642670a8a1e64ebb"
EXPOSURE_SCALE = 0.4913053810596466
EXPOSURE_CONFIG_SHA256 = "0c42035c1a525e27ff787d013ca27b0f5ae91338329eb0607c7042ebc5ae8905"
CONDITIONAL_SOURCE_SHA256 = "382558c75bf9b69502f2dedfd8f9522d4b8f063cdc94dfe39a91084c7cd1cb8f"
FIT_INDICES_SHA256 = "6ed6d2824a0cf6d98ab7ad6a35a5b0941dc33abc3194e2291d3b3b116cd881f8"
VALIDATION_INDICES_SHA256 = "5370b35f304a779cde7f978df6392b9974faaf4296e9edf75f9edf8202db02fa"
CONFIRMATION_INDICES_SHA256 = "ff7ab50b95d96e727c240edc2baa916d1e6c11402b8f4f9c2ce6e2e7e5e2218a"
SELECTION_INDICES_SHA256 = "11c076affa8511e883f595a0d1ee91e40f6d94747a351e21269520344d3f594f"
EVALUATOR_THRESHOLD = 0.985
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 128
TRAIN_SEED = 1234
OPTICAL_BATCH_SIZE = 10
NEXT_ACTION_READY = "IMPLEMENT_S2_DIRECT_OLS_SEMANTIC_DECODE_CANARY"
NEXT_ACTION_NOT_QUALIFIED = "REVIEW_FROZEN_OUTPUT_SEMANTIC_EVALUATOR_FAILURE"
FROZEN_OUTPUT_SEMANTIC_EVALUATOR_READY = "FROZEN_OUTPUT_SEMANTIC_EVALUATOR_READY"
FROZEN_OUTPUT_SEMANTIC_EVALUATOR_NOT_QUALIFIED = (
    "FROZEN_OUTPUT_SEMANTIC_EVALUATOR_NOT_QUALIFIED"
)
BLOCKED_EVALUATOR_IDENTITY_LEAKAGE = "BLOCKED_EVALUATOR_IDENTITY_LEAKAGE"
BLOCKED_EXPOSURE_SCALE_PROVENANCE = "BLOCKED_EXPOSURE_SCALE_PROVENANCE"
BLOCKED_OPTICAL_STATE_ISOLATION = "BLOCKED_OPTICAL_STATE_ISOLATION"
BLOCKED_EVALUATOR_FREEZE_REPLAY = "BLOCKED_EVALUATOR_FREEZE_REPLAY"
BLOCKED_EVALUATOR_CONTRACT = "BLOCKED_EVALUATOR_CONTRACT"
BLOCKED_EVALUATOR_METRICS = "BLOCKED_EVALUATOR_METRICS_INVALID"
DEFAULT_CONFIG_PATH = Path(
    "configs/research/lightgen_scaled_chen_frozen_output_semantic_evaluator_v1.yaml"
)
DEFAULT_OUTPUT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_frozen_output_semantic_evaluator_v1_attempt_01"
)
CONFIG_PATH = DEFAULT_CONFIG_PATH
ALLOWED_DIRTY_PATHS = {
    "scripts/research/build_scaled_chen_frozen_output_semantic_evaluator.py",
    "configs/research/lightgen_scaled_chen_frozen_output_semantic_evaluator_v1.yaml",
    "tests/v2/test_scaled_chen_frozen_output_semantic_evaluator.py",
}


@dataclass(frozen=True)
class OutputCorpus:
    fit_views: torch.Tensor
    validation_views: torch.Tensor
    fit_labels: torch.Tensor
    validation_labels: torch.Tensor
    fit_indices: torch.Tensor
    validation_indices: torch.Tensor
    identity_records: list[dict[str, Any]]


def _resolve(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return base._repo_root() / candidate


def _relative_path(value: str | Path) -> str:
    target = Path(value).resolve()
    try:
        return target.relative_to(base._repo_root()).as_posix()
    except ValueError:
        return str(target)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _assert_config(payload: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "task_id",
        "hypothesis",
        "s1_authority",
        "r2_calibration",
        "checkpoint",
        "output",
        "dataset",
        "exposure",
        "output_domain",
        "evaluator",
        "training",
        "execution",
        "authorization",
        "governance",
        "forbidden_operations",
    }
    public_payload = {key: value for key, value in payload.items() if not key.startswith("_")}
    if set(public_payload) != required:
        raise ValueError("FOSEA config fields are invalid")
    if payload.get("schema") != CONFIG_SCHEMA:
        raise ValueError("FOSEA config schema is invalid")
    if payload.get("task_id") != TASK_ID or payload.get("hypothesis") != HYPOTHESIS:
        raise ValueError("FOSEA config task identity is invalid")

    s1_authority = _require_mapping(payload["s1_authority"], "s1_authority")
    expected_s1 = {
        "config": "configs/research/lightgen_scaled_chen_ols_semantic_representation_audit_v1.yaml",
        "config_sha256": S1_CONFIG_SHA256,
        "canonical_sha256": S1_CONFIG_CANONICAL_SHA256,
        "manifest": "outputs/20260904_v2/runs/lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01/semantic_manifest.json",
        "manifest_sha256": S1_MANIFEST_SHA256,
        "gate": "S1_POWER_SEMANTICS_SUPPORTED",
    }
    if dict(s1_authority) != expected_s1:
        raise ValueError("FOSEA S1 authority binding is invalid")

    r2_authority = _require_mapping(payload["r2_calibration"], "r2_calibration")
    expected_r2 = {
        "manifest": "outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/calibration_manifest.json",
        "manifest_sha256": R2_MANIFEST_SHA256,
        "payload": "outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/repaired_calibration.pt",
        "payload_sha256": R2_PAYLOAD_SHA256,
        "schema": "scaled_chen_bogt_calibration.v2",
        "contract_version": "unit_aware_raw_mean_v2",
        "zero_phase_reference": "independent_train_only_zero_phase",
    }
    if dict(r2_authority) != expected_r2:
        raise ValueError("FOSEA repaired calibration binding is invalid")

    checkpoint = _require_mapping(payload["checkpoint"], "checkpoint")
    expected_checkpoint = {
        "path": "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt",
        "sha256": CHECKPOINT_SHA256,
        "schema": "scaled_chen_bogt_horizon_checkpoint.v1",
        "global_step": 5000,
        "selection": "final_registered_step_5000_parent",
    }
    if dict(checkpoint) != expected_checkpoint:
        raise ValueError("FOSEA checkpoint binding is invalid")

    output = _require_mapping(payload["output"], "output")
    if dict(output) != {"path": str(DEFAULT_OUTPUT_PATH).replace("\\", "/")}:
        raise ValueError("FOSEA output path is invalid")

    dataset = _require_mapping(payload["dataset"], "dataset")
    expected_dataset = {
        "dataset_root": "data",
        "mra_manifest_sha256": MRA_MANIFEST_SHA256,
        "fit": {
            "dataset": "mnist_train",
            "start": 0,
            "stop": 50000,
            "count": 50000,
            "indices_sha256": FIT_INDICES_SHA256,
        },
        "validation": {
            "dataset": "mnist_train",
            "start": 50000,
            "stop": 55000,
            "count": 5000,
            "indices_sha256": VALIDATION_INDICES_SHA256,
        },
        "s2_confirmation": {
            "dataset": "mnist_train",
            "start": 55000,
            "stop": 60000,
            "count": 5000,
            "indices_sha256": CONFIRMATION_INDICES_SHA256,
        },
        "s2_selection": {
            "dataset": "mnist_test",
            "source": "mnist_test_indices_5000_9999",
            "count": 1000,
            "indices_sha256": SELECTION_INDICES_SHA256,
        },
        "evaluator_consumed": ["fit", "validation"],
        "evaluator_excluded": ["s2_confirmation", "s2_selection"],
    }
    if dict(dataset) != expected_dataset:
        raise ValueError("FOSEA dataset split contract is invalid")

    exposure = _require_mapping(payload["exposure"], "exposure")
    expected_exposure = {
        "scale": EXPOSURE_SCALE,
        "source_config": "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml",
        "source_config_sha256": EXPOSURE_CONFIG_SHA256,
        "source_field": "conditional_generation.fixed_exposure_scale",
        "provenance": "fixed_pre_s2_formal_configuration",
        "fit_from_s2_confirmation": False,
        "fit_from_s2_selection": False,
        "tuned_on_s2_performance": False,
    }
    if dict(exposure) != expected_exposure:
        raise ValueError("FOSEA exposure contract is invalid")

    output_domain = _require_mapping(payload["output_domain"], "output_domain")
    expected_output_domain = {
        "raw_tensor_path": "ScaledChenBOGTModel.forward.output.camera",
        "raw_shape": ["N", 1, 200, 200],
        "raw_dtype": "torch.float32",
        "processed_tensor_path": "fixed_exposure_view(output.camera)",
        "processed_shape": ["N", 1, 28, 28],
        "processed_dtype": "torch.float32",
        "clamp_range": [0.0, 1.0],
        "pooling": "adaptive_avg_pool2d_28x28",
        "normalization": "one_global_exposure_scale",
        "per_sample_normalization": False,
        "input_encoding": "mnist_intensity_to_sqrt_amplitude",
        "output_class_map": "mnist_0_to_9",
    }
    if dict(output_domain) != expected_output_domain:
        raise ValueError("FOSEA output-domain contract is invalid")

    evaluator = _require_mapping(payload["evaluator"], "evaluator")
    if dict(evaluator) != {
        "architecture": "FrozenMNISTEvaluator",
        "schema": "frozen_mnist_evaluator_v1",
        "class_count": 10,
        "class_mapping": "mnist_label_identity_0_to_9",
        "selection": "final_epoch_only",
    }:
        raise ValueError("FOSEA evaluator architecture contract is invalid")

    training = _require_mapping(payload["training"], "training")
    if dict(training) != {
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": TRAIN_BATCH_SIZE,
        "epochs": TRAIN_EPOCHS,
        "initialization_seed": TRAIN_SEED,
        "epoch_order_seed": "1234_plus_epoch",
        "loss": "cross_entropy",
        "preflight_steps": 3,
        "validation_model_selection": False,
        "checkpoint_selection": "final_epoch",
    }:
        raise ValueError("FOSEA training contract is invalid")

    execution = _require_mapping(payload["execution"], "execution")
    if dict(execution) != {
        "device": "cuda",
        "optical_batch_size": OPTICAL_BATCH_SIZE,
        "min_free_cuda_bytes": 4294967296,
        "max_peak_cuda_bytes": 5368709120,
        "min_disk_free_bytes": 10737418240,
        "max_storage_bytes": 805306368,
        "require_cublas_workspace_config": ":4096:8",
    }:
        raise ValueError("FOSEA execution contract is invalid")

    authorization = _require_mapping(payload["authorization"], "authorization")
    if authorization.get("s2_preview_forbidden") is not True:
        raise ValueError("FOSEA S2 preview is forbidden during evaluator build")
    if dict(authorization) != {
        "optical_training_authorized": False,
        "evaluator_build_authorized": "bounded_only",
        "after_build_evaluator_build_authorized": False,
        "training_authorized": False,
        "formal_promotion": False,
        "s2_execution": False,
        "s2_preview_forbidden": True,
        "s2_confirmation_consumed": False,
    }:
        raise ValueError("FOSEA authorization boundary is invalid")

    governance = _require_mapping(payload["governance"], "governance")
    if dict(governance) != {
        "mainline_state": "closed_at_current_evidence_boundary",
        "reconstruction_manifold_supported": False,
        "generative_manifold_supported": False,
        "generation_readiness": "not_ready",
        "training_authorized": False,
        "formal_promotion": False,
    }:
        raise ValueError("FOSEA governance boundary is invalid")

    if list(payload["forbidden_operations"]) != [
        "optical_training",
        "optical_backward",
        "optical_phase_update",
        "prior_sampling",
        "prior_generation",
        "s2_direct_replay",
        "s2_semantic_measurement",
        "architecture_sweep",
        "seed_sweep",
        "hyperparameter_sweep",
        "formal_promotion",
    ]:
        raise ValueError("FOSEA forbidden-operation boundary is invalid")

    for name, value in {
        "s1 config": s1_authority["config_sha256"],
        "s1 canonical config": s1_authority["canonical_sha256"],
        "s1 manifest": s1_authority["manifest_sha256"],
        "checkpoint": checkpoint["sha256"],
        "R2 manifest": r2_authority["manifest_sha256"],
        "R2 payload": r2_authority["payload_sha256"],
        "MRA manifest": dataset["mra_manifest_sha256"],
        "fit indices": dataset["fit"]["indices_sha256"],
        "validation indices": dataset["validation"]["indices_sha256"],
        "confirmation indices": dataset["s2_confirmation"]["indices_sha256"],
        "selection indices": dataset["s2_selection"]["indices_sha256"],
        "exposure config": exposure["source_config_sha256"],
    }.items():
        _require_sha(value, name)


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    target = _resolve(path)
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("FOSEA config must be a YAML mapping")
    _assert_config(payload)
    return payload


def next_write_once_output(base_path: str | Path) -> Path:
    """Return the first unused numbered attempt without touching the disk."""
    requested = Path(base_path)
    match = re.fullmatch(r"(.+)_attempt_(\d+)", requested.name)
    if match is None:
        raise ValueError("write-once output must end with _attempt_<number>")
    prefix, number_text = match.groups()
    number = int(number_text)
    while True:
        candidate = requested.parent / f"{prefix}_attempt_{number:02d}"
        if not candidate.exists():
            return candidate
        number += 1


def _range_indices(start: int, stop: int) -> torch.Tensor:
    return torch.arange(start, stop, dtype=torch.long)


def registered_split_indices(
    config: Mapping[str, Any], selection_indices: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Return the fixed split identities; selection is supplied by the MRA loader."""
    _assert_config(config)
    dataset = config["dataset"]
    fit = _range_indices(dataset["fit"]["start"], dataset["fit"]["stop"])
    validation = _range_indices(
        dataset["validation"]["start"], dataset["validation"]["stop"]
    )
    confirmation = _range_indices(
        dataset["s2_confirmation"]["start"], dataset["s2_confirmation"]["stop"]
    )
    for name, indices, expected in (
        ("fit", fit, FIT_INDICES_SHA256),
        ("validation", validation, VALIDATION_INDICES_SHA256),
        ("s2_confirmation", confirmation, CONFIRMATION_INDICES_SHA256),
    ):
        if tensor_sha256(indices) != expected:
            raise ValueError(f"registered {name} indices changed")
    if selection_indices is None:
        selection = _range_indices(0, int(dataset["s2_selection"]["count"]))
    else:
        selection = selection_indices.detach().cpu().to(torch.long).contiguous()
        if tensor_sha256(selection) != SELECTION_INDICES_SHA256:
            raise ValueError("registered S2 selection indices changed")
    if selection.ndim != 1 or selection.numel() != 1000:
        raise ValueError("registered S2 selection count is invalid")
    if torch.unique(selection).numel() != selection.numel():
        raise ValueError("registered S2 selection contains duplicate identities")
    return {
        "fit": fit,
        "validation": validation,
        "s2_confirmation": confirmation,
        "s2_selection": selection,
    }


def _source_identity_set(source: str, indices: torch.Tensor) -> set[tuple[str, int]]:
    if indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError("split indices must be one-dimensional int64 tensors")
    if torch.unique(indices).numel() != indices.numel():
        raise ValueError(f"{source} split contains duplicate identities")
    return {(source, int(index)) for index in indices.tolist()}


def audit_identity_leakage(
    fit_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    confirmation_indices: torch.Tensor,
    selection_indices: torch.Tensor,
) -> dict[str, Any]:
    """Audit identity overlap using dataset namespace, never raw integer coincidence."""
    groups = {
        "fit": ("mnist_train", fit_indices),
        "validation": ("mnist_train", validation_indices),
        "s2_confirmation": ("mnist_train", confirmation_indices),
        "s2_selection": ("mnist_test", selection_indices),
    }
    identity_sets = {
        name: _source_identity_set(source, indices)
        for name, (source, indices) in groups.items()
    }
    intersections: dict[str, int] = {}
    same_source_index_intersections: dict[str, int] = {}
    for left, right in combinations(groups, 2):
        key = f"{left}__{right}"
        intersections[key] = len(identity_sets[left] & identity_sets[right])
        left_source, left_indices = groups[left]
        right_source, right_indices = groups[right]
        same_source_index_intersections[key] = (
            int(torch.isin(left_indices, right_indices).sum())
            if left_source == right_source
            else 0
        )
    all_zero = all(value == 0 for value in intersections.values())
    same_source_zero = all(
        value == 0 for value in same_source_index_intersections.values()
    )
    return {
        "schema": LEAKAGE_SCHEMA,
        "identity_namespace": {
            "fit": "mnist_train",
            "validation": "mnist_train",
            "s2_confirmation": "mnist_train",
            "s2_selection": "mnist_test",
        },
        "evaluator_consumed_splits": ["fit", "validation"],
        "evaluator_excluded_splits": ["s2_confirmation", "s2_selection"],
        "pairwise_identity_intersections": intersections,
        "pairwise_same_source_index_intersections": same_source_index_intersections,
        "all_pairwise_index_intersections_zero": same_source_zero,
        "underlying_source_identity_overlap_zero": all_zero,
        "fit_validation_overlap_zero": intersections["fit__validation"] == 0,
        "fit_confirmation_overlap_zero": intersections["fit__s2_confirmation"] == 0,
        "validation_confirmation_overlap_zero": intersections[
            "validation__s2_confirmation"
        ]
        == 0,
        "fit_selection_overlap_zero": intersections["fit__s2_selection"] == 0,
        "validation_selection_overlap_zero": intersections[
            "validation__s2_selection"
        ]
        == 0,
        "s2_confirmation_consumed_by_evaluator": False,
        "s2_selection_consumed_by_evaluator": False,
        "valid": all_zero and same_source_zero,
    }


def fixed_output_view(
    raw_camera: torch.Tensor, exposure_scale: float, *, output_size: int = 28
) -> torch.Tensor:
    """Apply the registered global exposure and fixed 28x28 pooling contract."""
    if (
        not torch.is_tensor(raw_camera)
        or raw_camera.ndim != 4
        or raw_camera.shape[1] != 1
        or raw_camera.dtype != torch.float32
        or torch.is_complex(raw_camera)
        or not torch.isfinite(raw_camera).all()
        or exposure_scale != EXPOSURE_SCALE
    ):
        raise ValueError("raw CMOS does not satisfy the fixed output-domain contract")
    viewed = fixed_exposure_view(raw_camera, exposure_scale, output_size=output_size)
    if (
        viewed.ndim != 4
        or viewed.shape[1] != 1
        or viewed.dtype != torch.float32
        or not torch.isfinite(viewed).all()
        or viewed.min() < 0.0
        or viewed.max() > 1.0
    ):
        raise ValueError("fixed output view is invalid")
    return viewed


def training_order(count: int, *, epoch: int, seed: int = TRAIN_SEED) -> torch.Tensor:
    if count <= 0 or epoch < 0:
        raise ValueError("training order requires a positive count and nonnegative epoch")
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    return torch.randperm(count, generator=generator)


def validate_checkpoint_identity(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    for key in ("path", "sha256", "global_step"):
        if actual.get(key) != expected.get(key):
            raise ValueError(f"checkpoint identity mismatch for {key}")
    if expected.get("schema") is not None and actual.get("schema") != expected["schema"]:
        raise ValueError("checkpoint identity mismatch for schema")
    return True


def qualification_classification(
    *,
    validation_accuracy: float,
    validation_metrics_finite: bool,
    contract_valid: bool,
    leakage_free: bool,
    optical_isolated: bool,
    freeze_replay: bool,
) -> str:
    if not contract_valid:
        return BLOCKED_EVALUATOR_CONTRACT
    if not leakage_free:
        return BLOCKED_EVALUATOR_IDENTITY_LEAKAGE
    if not optical_isolated:
        return BLOCKED_OPTICAL_STATE_ISOLATION
    if not freeze_replay:
        return BLOCKED_EVALUATOR_FREEZE_REPLAY
    if not validation_metrics_finite:
        return BLOCKED_EVALUATOR_METRICS
    if not math.isfinite(validation_accuracy):
        return BLOCKED_EVALUATOR_METRICS
    return (
        FROZEN_OUTPUT_SEMANTIC_EVALUATOR_READY
        if validation_accuracy >= EVALUATOR_THRESHOLD
        else FROZEN_OUTPUT_SEMANTIC_EVALUATOR_NOT_QUALIFIED
    )


def _classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    if (
        logits.ndim != 2
        or logits.shape[1] != 10
        or labels.ndim != 1
        or logits.shape[0] != labels.shape[0]
        or labels.dtype != torch.long
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("evaluator classification tensors are invalid")
    predictions = logits.argmax(dim=1)
    confusion = torch.zeros((10, 10), dtype=torch.long)
    confusion.index_put_((labels.cpu(), predictions.cpu()), torch.ones_like(labels.cpu()), accumulate=True)
    counts = torch.bincount(labels.cpu(), minlength=10)
    correct = confusion.diag()
    if (counts == 0).any():
        raise ValueError("validation split is missing a class")
    per_class = (correct.to(torch.float64) / counts.to(torch.float64)).tolist()
    accuracy = float((predictions == labels).to(torch.float64).mean().item())
    return {
        "sample_count": int(labels.numel()),
        "overall_accuracy": accuracy,
        "per_class_accuracy": [float(value) for value in per_class],
        "class_counts": [int(value) for value in counts.tolist()],
        "confusion_matrix": confusion.tolist(),
        "finite": True,
    }


def _parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _parameter_update_norm(
    before: Mapping[str, torch.Tensor], model: nn.Module
) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for name, parameter in model.named_parameters():
        delta = parameter.detach().cpu().to(torch.float64) - before[name].cpu().to(torch.float64)
        total += delta.square().sum()
    return float(total.sqrt().item())


def _evaluator_training_step(
    model: FrozenMNISTEvaluator,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, int]:
    optimizer.zero_grad(set_to_none=True)
    _, logits = model(images)
    loss = nn.functional.cross_entropy(logits, labels)
    if not torch.isfinite(loss):
        raise RuntimeError("evaluator training produced a non-finite loss")
    loss.backward()
    optimizer.step()
    if any(
        parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    ):
        raise RuntimeError("evaluator training produced a non-finite gradient")
    return float(loss.detach().item()), int((logits.argmax(dim=1) == labels).sum().item())


def _run_three_step_preflight(
    views: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if views.shape[0] < TRAIN_BATCH_SIZE:
        raise ValueError("fit corpus is too small for evaluator preflight")

    def run_case(case_labels: torch.Tensor, case_name: str) -> dict[str, Any]:
        torch.manual_seed(TRAIN_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(TRAIN_SEED)
        model = FrozenMNISTEvaluator().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(config["training"]["learning_rate"])
        )
        losses: list[float] = []
        update_norms: list[float] = []
        for step in range(int(config["training"]["preflight_steps"])):
            order = training_order(views.shape[0], epoch=step, seed=TRAIN_SEED)
            batch_indices = order[:TRAIN_BATCH_SIZE]
            before = _parameter_snapshot(model)
            loss, _ = _evaluator_training_step(
                model,
                optimizer,
                views.index_select(0, batch_indices).to(device),
                case_labels.index_select(0, batch_indices).to(device),
            )
            losses.append(loss)
            update_norms.append(_parameter_update_norm(before, model))
        finite = all(math.isfinite(value) for value in losses + update_norms)
        result = {
            "case": case_name,
            "steps": len(losses),
            "losses": losses,
            "update_norms": update_norms,
            "finite": finite,
            "parameters_updated": all(value > 0.0 for value in update_norms),
        }
        del optimizer, model
        return result

    real = run_case(labels, "registered_labels")
    negative_labels = torch.remainder(torch.roll(labels, shifts=1), 10)
    negative = run_case(negative_labels, "fixed_label_shift_negative_control")
    if not real["finite"] or not real["parameters_updated"]:
        raise RuntimeError("evaluator three-step preflight did not produce real updates")
    if not negative["finite"] or not negative["parameters_updated"]:
        raise RuntimeError("evaluator negative-control preflight is invalid")
    return {
        "schema": "lightgen_scaled_chen_frozen_output_evaluator_preflight.v1",
        "same_entrypoint": True,
        "real": real,
        "negative_control": negative,
        "optical_optimizer_created": False,
        "optical_backward_called": False,
        "optical_training_steps": 0,
    }


def _train_evaluator(
    fit_views: torch.Tensor,
    fit_labels: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
) -> tuple[FrozenMNISTEvaluator, dict[str, Any]]:
    torch.manual_seed(TRAIN_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TRAIN_SEED)
    evaluator = FrozenMNISTEvaluator().to(device)
    optimizer = torch.optim.Adam(
        evaluator.parameters(), lr=float(config["training"]["learning_rate"])
    )
    epochs: list[dict[str, Any]] = []
    batch_size = int(config["training"]["batch_size"])
    train_steps = 0
    evaluator.train()
    for epoch in range(int(config["training"]["epochs"])):
        order = training_order(fit_views.shape[0], epoch=epoch, seed=TRAIN_SEED)
        loss_sum = 0.0
        correct = 0
        sample_count = 0
        for start in range(0, fit_views.shape[0], batch_size):
            indices = order[start : start + batch_size]
            batch_views = fit_views.index_select(0, indices).to(device)
            batch_labels = fit_labels.index_select(0, indices).to(device)
            loss, batch_correct = _evaluator_training_step(
                evaluator, optimizer, batch_views, batch_labels
            )
            count = int(indices.numel())
            loss_sum += loss * count
            correct += batch_correct
            sample_count += count
            train_steps += 1
        epoch_loss = loss_sum / sample_count
        epoch_accuracy = correct / sample_count
        if not math.isfinite(epoch_loss) or not math.isfinite(epoch_accuracy):
            raise RuntimeError("evaluator training metrics are non-finite")
        epochs.append(
            {
                "epoch": epoch + 1,
                "loss": epoch_loss,
                "accuracy": epoch_accuracy,
                "sample_count": sample_count,
                "order_sha256": tensor_sha256(order),
            }
        )
    evaluator.eval()
    return evaluator, {
        "schema": TRAINING_METRICS_SCHEMA,
        "optimizer": "Adam",
        "learning_rate": float(config["training"]["learning_rate"]),
        "batch_size": batch_size,
        "epochs": epochs,
        "train_steps": train_steps,
        "initialization_seed": TRAIN_SEED,
        "epoch_order_seed": "1234_plus_epoch",
        "validation_model_selection": False,
        "checkpoint_selection": "final_epoch",
    }


def _evaluate_evaluator(
    evaluator: FrozenMNISTEvaluator,
    views: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    evaluator.eval()
    logits_parts: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, views.shape[0], TRAIN_BATCH_SIZE):
            _, logits = evaluator(views[start : start + TRAIN_BATCH_SIZE].to(device))
            logits_parts.append(logits.detach().cpu())
    return _classification_metrics(torch.cat(logits_parts), labels.cpu())


def _validate_exposure_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    exposure = config["exposure"]
    path = _resolve(exposure["source_config"])
    if not path.is_file() or file_sha256(path) != exposure["source_config_sha256"]:
        raise RuntimeError(f"{BLOCKED_EXPOSURE_SCALE_PROVENANCE}: source config hash")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        value = payload["conditional_generation"]["fixed_exposure_scale"]
    except (KeyError, TypeError):
        raise RuntimeError(
            f"{BLOCKED_EXPOSURE_SCALE_PROVENANCE}: source field missing"
        ) from None
    if float(value) != EXPOSURE_SCALE or float(exposure["scale"]) != EXPOSURE_SCALE:
        raise RuntimeError(f"{BLOCKED_EXPOSURE_SCALE_PROVENANCE}: scale changed")
    if (
        exposure["fit_from_s2_confirmation"]
        or exposure["fit_from_s2_selection"]
        or exposure["tuned_on_s2_performance"]
    ):
        raise RuntimeError(
            f"{BLOCKED_EXPOSURE_SCALE_PROVENANCE}: S2-dependent provenance"
        )
    return {
        "schema": EXPOSURE_SCHEMA,
        "scale": EXPOSURE_SCALE,
        "source_config": _relative_path(path),
        "source_config_sha256": file_sha256(path),
        "source_field": exposure["source_field"],
        "provenance": exposure["provenance"],
        "fit_from_s2_confirmation": False,
        "fit_from_s2_selection": False,
        "tuned_on_s2_performance": False,
        "status": "EXPOSURE_SCALE_AUTHORITY_PASS",
    }


def _load_upstream(config: Mapping[str, Any]) -> dict[str, Any]:
    s1_ref = config["s1_authority"]
    s1_path = _resolve(s1_ref["config"])
    if file_sha256(s1_path) != S1_CONFIG_SHA256:
        raise ValueError("FOSEA S1 config file hash is stale")
    loaded_path, s1_config, s1_file_hash, s1_canonical_hash = s1.load_config(s1_path)
    if loaded_path != s1_path or s1_file_hash != S1_CONFIG_SHA256:
        raise ValueError("FOSEA S1 config identity is invalid")
    if s1_canonical_hash != S1_CONFIG_CANONICAL_SHA256:
        raise ValueError("FOSEA S1 canonical config identity is invalid")
    s1_manifest_path = _resolve(s1_ref["manifest"])
    if file_sha256(s1_manifest_path) != S1_MANIFEST_SHA256:
        raise ValueError("FOSEA S1 manifest hash is stale")
    s1_manifest = json.loads(s1_manifest_path.read_text(encoding="utf-8"))
    if (
        s1_manifest.get("schema") != s1.AUDIT_SCHEMA
        or s1_manifest.get("task_id") != s1.TASK_ID
        or s1_manifest.get("gate") != s1.S1_POWER_SEMANTICS_SUPPORTED
        or s1_manifest.get("status") != "valid_completed"
    ):
        raise ValueError("FOSEA S1 authority status is invalid")
    parent = rcr._validate_parent_mra(s1_config)
    dataset_root = _resolve(config["dataset"]["dataset_root"])
    data = mra._load_data(dataset_root)
    if data.manifest.get("manifest_hash") != MRA_MANIFEST_SHA256:
        raise ValueError("FOSEA dataset manifest hash is stale")
    rcr._validate_parent_panel(parent["panel"], data, parent["source"])
    r2_info = rcr._validate_r2(s1_config)
    checkpoint, checkpoint_path = rcr._load_checkpoint(s1_config, parent["source"])
    expected_checkpoint = config["checkpoint"]
    actual_checkpoint = {
        "path": _relative_path(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "schema": checkpoint.get("schema"),
        "global_step": checkpoint.get("global_step"),
    }
    validate_checkpoint_identity(actual_checkpoint, expected_checkpoint)
    return {
        "s1_path": s1_path,
        "s1_config": s1_config,
        "s1_manifest_path": s1_manifest_path,
        "s1_manifest": s1_manifest,
        "parent": parent,
        "data": data,
        "r2": r2_info,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
    }


def _resource_preflight(config: Mapping[str, Any], output_path: Path) -> dict[str, int]:
    execution = config["execution"]
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != execution[
        "require_cublas_workspace_config"
    ]:
        raise RuntimeError(
            "BLOCKED_EVALUATOR_ENVIRONMENT: CUBLAS_WORKSPACE_CONFIG must be :4096:8"
        )
    device = torch.device(execution["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_CUDA_UNAVAILABLE")
    free_cuda, total_cuda = torch.cuda.mem_get_info(device)
    if free_cuda < int(execution["min_free_cuda_bytes"]):
        raise RuntimeError("BLOCKED_EVALUATOR_RESOURCE: less than 4 GiB free CUDA")
    disk = shutil.disk_usage(base._repo_root().anchor or str(base._repo_root()))
    if disk.free < int(execution["min_disk_free_bytes"]):
        raise RuntimeError("BLOCKED_EVALUATOR_RESOURCE: less than 10 GiB free disk")
    if output_path.exists():
        raise FileExistsError(f"write-once output already exists: {_relative_path(output_path)}")
    return {
        "free_cuda_bytes_before": int(free_cuda),
        "total_cuda_bytes": int(total_cuda),
        "free_disk_bytes_before": int(disk.free),
    }


def _require_expected_worktree(git_state: Mapping[str, Any]) -> None:
    dirty_paths = {
        line[3:]
        for line in git_state.get("status", [])
        if len(line) >= 4
    }
    unexpected = dirty_paths - ALLOWED_DIRTY_PATHS
    if git_state.get("branch") != BRANCH or unexpected:
        raise RuntimeError(
            "BLOCKED_UNEXPECTED_DIRTY_PATH: "
            f"branch={git_state.get('branch')!r}, paths={sorted(unexpected or dirty_paths)}"
        )


def _controller_contract_digest() -> str:
    return canonical_sha256(
        {
            "kind": "frozen_mnist_input_to_amplitude_slm_command",
            "controller_parameters": "none",
            "boundary": "MNIST intensity -> sqrt amplitude -> frozen Scaled-Chen forward",
        }
    )


def _optical_state_snapshot(model: nn.Module) -> dict[str, Any]:
    return {
        "checkpoint_digest": CHECKPOINT_SHA256,
        "model_state_digest": rcr.state_digest(model),
        "encoder_phase_digest": tensor_sha256(model.system.encoder.phase),
        "decoder_phase_digest": tensor_sha256(model.system.decoder.phase),
        "ols_state_config_digest": bogt.mode_bank_sha256(model),
        "controller_config_digest": _controller_contract_digest(),
    }


def _optical_state_isolation(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    compared = (
        "checkpoint_digest",
        "model_state_digest",
        "encoder_phase_digest",
        "decoder_phase_digest",
        "ols_state_config_digest",
        "controller_config_digest",
    )
    unchanged = all(before[name] == after[name] for name in compared)
    return {
        "schema": ISOLATION_SCHEMA,
        "before": dict(before),
        "after": dict(after),
        "state_unchanged": unchanged,
        "optical_optimizer_created": False,
        "optical_backward_called": False,
        "optical_training_steps": 0,
        "phase_updated": False,
        "s2_preview_executed": False,
        "status": "OPTICAL_STATE_ISOLATION_PASS" if unchanged else BLOCKED_OPTICAL_STATE_ISOLATION,
    }


def _build_output_corpus(
    model: bogt.ScaledChenBOGTModel,
    data: mra.AuditData,
    splits: Mapping[str, torch.Tensor],
    device: torch.device,
    exposure_scale: float,
    batch_size: int,
) -> OutputCorpus:
    views: dict[str, list[torch.Tensor]] = {"fit": [], "validation": []}
    labels: dict[str, list[torch.Tensor]] = {"fit": [], "validation": []}
    identity_records: list[dict[str, Any]] = []
    model.eval()
    for split_name in ("fit", "validation"):
        split_indices = splits[split_name]
        split_labels = data.labels.index_select(0, split_indices)
        for start in range(0, split_indices.numel(), batch_size):
            batch_indices = split_indices[start : start + batch_size]
            batch_labels = split_labels[start : start + batch_size]
            rendered = base.render_intensity_batch(
                data.images.index_select(0, batch_indices)
            )
            with torch.inference_mode():
                output = model(rendered.amplitude.to(device), return_trace=False)
            raw_camera = output.camera
            if (
                raw_camera.shape != (batch_indices.numel(), 1, 200, 200)
                or raw_camera.dtype != torch.float32
                or not torch.isfinite(raw_camera).all()
                or raw_camera.min() < 0.0
            ):
                raise RuntimeError("frozen optical output camera contract is invalid")
            processed = fixed_output_view(raw_camera, exposure_scale)
            raw_cpu = raw_camera.detach().cpu().contiguous()
            processed_cpu = processed.detach().cpu().contiguous()
            views[split_name].append(processed_cpu)
            labels[split_name].append(batch_labels.clone())
            for offset, (source_index, label) in enumerate(
                zip(batch_indices.tolist(), batch_labels.tolist())
            ):
                identity_records.append(
                    {
                        "split": split_name,
                        "source_dataset": "mnist_train",
                        "source_index": int(source_index),
                        "label": int(label),
                        "raw_camera_sha256": tensor_sha256(raw_cpu[offset]),
                        "processed_view_sha256": tensor_sha256(processed_cpu[offset]),
                    }
                )
            del output, raw_camera, processed, raw_cpu, processed_cpu, rendered
    fit_views = torch.cat(views["fit"], dim=0)
    validation_views = torch.cat(views["validation"], dim=0)
    fit_label_tensor = torch.cat(labels["fit"], dim=0)
    validation_label_tensor = torch.cat(labels["validation"], dim=0)
    if (
        fit_views.shape != (50000, 1, 28, 28)
        or validation_views.shape != (5000, 1, 28, 28)
        or fit_label_tensor.shape != (50000,)
        or validation_label_tensor.shape != (5000,)
        or not torch.isfinite(fit_views).all()
        or not torch.isfinite(validation_views).all()
    ):
        raise RuntimeError("output-domain corpus shape or finiteness is invalid")
    if fit_views.min() < 0.0 or fit_views.max() > 1.0 or validation_views.min() < 0.0 or validation_views.max() > 1.0:
        raise RuntimeError("output-domain corpus is outside [0,1]")
    return OutputCorpus(
        fit_views,
        validation_views,
        fit_label_tensor,
        validation_label_tensor,
        splits["fit"].clone(),
        splits["validation"].clone(),
        identity_records,
    )


def _class_counts(labels: torch.Tensor) -> list[int]:
    return [int(value) for value in torch.bincount(labels, minlength=10).tolist()]


def _data_identity_manifest(
    config: Mapping[str, Any],
    data: mra.AuditData,
    splits: Mapping[str, torch.Tensor],
    leakage: Mapping[str, Any],
    corpus_hash: str,
) -> dict[str, Any]:
    split_payload: dict[str, Any] = {}
    for name, source, labels in (
        ("fit", "mnist_train", data.labels.index_select(0, splits["fit"])),
        (
            "validation",
            "mnist_train",
            data.labels.index_select(0, splits["validation"]),
        ),
        (
            "s2_confirmation",
            "mnist_train",
            data.labels.index_select(0, splits["s2_confirmation"]),
        ),
        (
            "s2_selection",
            "mnist_test",
            data.test_labels.index_select(0, splits["s2_selection"]),
        ),
    ):
        split_payload[name] = {
            "dataset": source,
            "count": int(labels.numel()),
            "indices_sha256": tensor_sha256(splits[name]),
            "labels_sha256": tensor_sha256(labels),
            "class_counts": _class_counts(labels),
        }
    payload: dict[str, Any] = {
        "schema": DATA_IDENTITY_SCHEMA,
        "dataset_root": config["dataset"]["dataset_root"],
        "mra_manifest_sha256": data.manifest["manifest_hash"],
        "source_dataset_hashes": {
            "train_images_sha256": data.manifest["train_images_sha256"],
            "train_labels_sha256": data.manifest["train_labels_sha256"],
            "test_images_sha256": data.manifest["test_images_sha256"],
            "test_labels_sha256": data.manifest["test_labels_sha256"],
        },
        "splits": split_payload,
        "evaluator_consumed_splits": ["fit", "validation"],
        "reserved_s2_splits": ["s2_confirmation", "s2_selection"],
        "s2_confirmation_consumed": False,
        "s2_selection_consumed": False,
        "reselection": False,
        "leakage_audit_valid": bool(leakage["valid"]),
        "output_domain_corpus_sha256": corpus_hash,
    }
    payload["identity_manifest_hash"] = canonical_sha256(payload)
    return payload


def _output_domain_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": OUTPUT_CONTRACT_SCHEMA,
        "input": {
            "source": "mnist_train_image",
            "shape": ["N", 1, 28, 28],
            "dtype": "torch.float32",
            "range": [0.0, 1.0],
            "encoding": "intensity_to_sqrt_amplitude",
        },
        "frozen_optical_forward": {
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "raw_path": config["output_domain"]["raw_tensor_path"],
            "raw_shape": config["output_domain"]["raw_shape"],
            "raw_dtype": config["output_domain"]["raw_dtype"],
        },
        "fixed_preprocessing": {
            "function": "ldm_project.v2.student.conditional.fixed_exposure_view",
            "source_sha256": CONDITIONAL_SOURCE_SHA256,
            "exposure_scale": EXPOSURE_SCALE,
            "clamp_range": [0.0, 1.0],
            "pooling": "adaptive_avg_pool2d_28x28",
            "processed_shape": ["N", 1, 28, 28],
            "processed_dtype": "torch.float32",
            "per_sample_normalization": False,
        },
        "storage_policy": {
            "processed_views_saved": True,
            "raw_camera_saved": False,
            "raw_camera_hashes_only": True,
            "s2_confirmation_forwarded": False,
            "s2_selection_forwarded": False,
        },
    }


def _training_contract(config: Mapping[str, Any], preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": TRAINING_SCHEMA,
        "architecture": "FrozenMNISTEvaluator",
        "architecture_schema": "frozen_mnist_evaluator_v1",
        "optimizer": config["training"]["optimizer"],
        "learning_rate": config["training"]["learning_rate"],
        "batch_size": config["training"]["batch_size"],
        "epochs": config["training"]["epochs"],
        "initialization_seed": config["training"]["initialization_seed"],
        "epoch_order_seed": config["training"]["epoch_order_seed"],
        "loss": config["training"]["loss"],
        "validation_model_selection": False,
        "checkpoint_selection": "final_epoch",
        "preflight_steps": config["training"]["preflight_steps"],
        "preflight_schema": preflight["schema"],
        "fit_only_parameter_updates": True,
        "validation_used_for_training": False,
        "s2_confirmation_used_for_training": False,
    }


def _checkpoint_payload(
    evaluator: FrozenMNISTEvaluator,
    validation_accuracy: float,
    data_identity_hash: str,
    output_contract_hash: str,
    exposure_hash: str,
    training_contract_hash: str,
    qualification: str,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "architecture_kind": "frozen_mnist_evaluator_v1",
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in evaluator.state_dict().items()
        },
        "validation_accuracy": validation_accuracy,
        "checkpoint_selection": "final_epoch",
        "data_identity_manifest_sha256": data_identity_hash,
        "output_domain_contract_sha256": output_contract_hash,
        "exposure_provenance_sha256": exposure_hash,
        "training_contract_sha256": training_contract_hash,
        "qualification": qualification,
        "fit_split_only": True,
        "s2_confirmation_used": False,
        "s2_selection_used": False,
    }


def _freeze_replay(
    checkpoint_path: Path,
    checkpoint_hash: str,
    evaluator: FrozenMNISTEvaluator,
    validation_views: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    payload = safe_torch_load(
        checkpoint_path, expected_sha256=checkpoint_hash, map_location="cpu"
    )
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("architecture_kind") != "frozen_mnist_evaluator_v1"
    ):
        raise RuntimeError(f"{BLOCKED_EVALUATOR_FREEZE_REPLAY}: checkpoint schema")
    replay = FrozenMNISTEvaluator().to(device)
    replay.load_state_dict(payload["state_dict"], strict=True)
    replay.freeze()
    evaluator.eval()
    panel = validation_views[: min(128, validation_views.shape[0])].to(device)
    with torch.inference_mode():
        _, expected_logits = evaluator(panel)
        _, replay_logits = replay(panel)
    difference = (expected_logits - replay_logits).abs()
    max_abs = float(difference.max().item()) if difference.numel() else 0.0
    same = bool(torch.equal(expected_logits, replay_logits))
    if not same and max_abs > 1.0e-6:
        raise RuntimeError(f"{BLOCKED_EVALUATOR_FREEZE_REPLAY}: logits changed")
    if any(parameter.requires_grad for parameter in replay.parameters()):
        raise RuntimeError(f"{BLOCKED_EVALUATOR_FREEZE_REPLAY}: parameters unfrozen")
    return {
        "schema": REPLAY_SCHEMA,
        "panel_count": int(panel.shape[0]),
        "checkpoint_sha256": checkpoint_hash,
        "logits_exact_equal": same,
        "max_abs_logit_difference": max_abs,
        "reloaded_parameters_frozen": True,
        "deterministic_replay": True,
        "status": "EVALUATOR_FREEZE_REPLAY_PASS",
    }


def _artifact_refs(root: Path) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "evaluator_manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        refs[relative] = {
            "path": relative,
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
            "role": "checkpoint" if path.suffix == ".pt" else "metadata",
        }
    return refs


def _assert_final_tree(root: Path) -> None:
    expected = {
        "compact_report.json",
        "data_identity_manifest.json",
        "evaluator_freeze_replay.json",
        "evaluator_manifest.json",
        "exposure_provenance.json",
        "frozen_output_semantic_evaluator.pt",
        "leakage_audit.json",
        "optical_state_isolation.json",
        "output_domain_contract.json",
        "output_domain_corpus.pt",
        "output_identity_manifest.json",
        "qualification_metrics.json",
        "training_contract.json",
        "training_metrics.json",
    }
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise RuntimeError(f"FOSEA output tree mismatch: {sorted(actual)}")
    for relative in actual:
        path = root / relative
        if path.name.startswith(".") or path.suffix.lower() in {".tmp", ".tif", ".tiff"}:
            raise RuntimeError(f"FOSEA output contains forbidden residue: {relative}")


def _publish_artifacts(
    output_path: Path,
    config: Mapping[str, Any],
    upstream: Mapping[str, Any],
    splits: Mapping[str, torch.Tensor],
    corpus: OutputCorpus,
    leakage: Mapping[str, Any],
    exposure: Mapping[str, Any],
    preflight: Mapping[str, Any],
    training_metrics: Mapping[str, Any],
    qualification_metrics: Mapping[str, Any],
    isolation: Mapping[str, Any],
    replay: Mapping[str, Any],
    resource: Mapping[str, Any],
    started_at: float,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.", dir=str(output_path.parent))
    )
    try:
        corpus_payload = {
            "schema": CORPUS_SCHEMA,
            "fit_views": corpus.fit_views,
            "validation_views": corpus.validation_views,
            "fit_labels": corpus.fit_labels,
            "validation_labels": corpus.validation_labels,
            "fit_indices": corpus.fit_indices,
            "validation_indices": corpus.validation_indices,
        }
        corpus_path = temporary / "output_domain_corpus.pt"
        atomic_torch_save_new(corpus_path, corpus_payload, artifact_kind="output corpus")
        corpus_hash = file_sha256(corpus_path)

        identity_payload = _data_identity_manifest(
            config, upstream["data"], splits, leakage, corpus_hash
        )
        identity_path = temporary / "data_identity_manifest.json"
        atomic_write_new_json(identity_path, identity_payload, artifact_kind="data identity")
        identity_hash = file_sha256(identity_path)

        output_identity_payload = {
            "schema": "lightgen_scaled_chen_frozen_output_identity_manifest.v1",
            "corpus_sha256": corpus_hash,
            "record_count": len(corpus.identity_records),
            "records": corpus.identity_records,
            "raw_camera_persisted": False,
            "s2_confirmation_records": 0,
            "s2_selection_records": 0,
        }
        atomic_write_new_json(
            temporary / "output_identity_manifest.json",
            output_identity_payload,
            artifact_kind="output identity",
        )

        output_contract_payload = _output_domain_contract(config)
        atomic_write_new_json(
            temporary / "output_domain_contract.json",
            output_contract_payload,
            artifact_kind="output contract",
        )
        output_contract_hash = file_sha256(temporary / "output_domain_contract.json")

        atomic_write_new_json(
            temporary / "exposure_provenance.json",
            exposure,
            artifact_kind="exposure provenance",
        )
        exposure_hash = file_sha256(temporary / "exposure_provenance.json")

        training_contract_payload = _training_contract(config, preflight)
        atomic_write_new_json(
            temporary / "training_contract.json",
            training_contract_payload,
            artifact_kind="training contract",
        )
        training_contract_hash = file_sha256(temporary / "training_contract.json")

        evaluator = qualification_metrics["_evaluator"]
        checkpoint_payload = _checkpoint_payload(
            evaluator,
            float(qualification_metrics["validation"]["overall_accuracy"]),
            identity_hash,
            output_contract_hash,
            exposure_hash,
            training_contract_hash,
            qualification_metrics["classification"],
        )
        checkpoint_path = temporary / "frozen_output_semantic_evaluator.pt"
        atomic_torch_save_new(
            checkpoint_path, checkpoint_payload, artifact_kind="evaluator checkpoint"
        )
        checkpoint_hash = file_sha256(checkpoint_path)

        training_payload = dict(training_metrics)
        training_payload["preflight"] = preflight
        training_payload["optical_optimizer_created"] = False
        training_payload["optical_backward_called"] = False
        training_payload["optical_training_steps"] = 0
        atomic_write_new_json(
            temporary / "training_metrics.json",
            training_payload,
            artifact_kind="training metrics",
        )

        qualification_payload = {
            key: value
            for key, value in qualification_metrics.items()
            if key != "_evaluator"
        }
        qualification_payload["evaluator_checkpoint_sha256"] = checkpoint_hash
        atomic_write_new_json(
            temporary / "qualification_metrics.json",
            qualification_payload,
            artifact_kind="qualification metrics",
        )
        atomic_write_new_json(
            temporary / "leakage_audit.json",
            leakage,
            artifact_kind="leakage audit",
        )
        atomic_write_new_json(
            temporary / "optical_state_isolation.json",
            isolation,
            artifact_kind="optical isolation",
        )
        atomic_write_new_json(
            temporary / "evaluator_freeze_replay.json",
            replay,
            artifact_kind="freeze replay",
        )

        classification = qualification_metrics["classification"]
        compact_report = {
            "schema": REPORT_SCHEMA,
            "task_id": TASK_ID,
            "classification": classification,
            "purpose": "clean frozen output-domain semantic evaluator only",
            "validation_accuracy": qualification_metrics["validation"][
                "overall_accuracy"
            ],
            "threshold": EVALUATOR_THRESHOLD,
            "s2_execution": False,
            "s2_preview_executed": False,
            "optical_training": False,
            "generation_authority": False,
            "next_action": (
                NEXT_ACTION_READY
                if classification == FROZEN_OUTPUT_SEMANTIC_EVALUATOR_READY
                else NEXT_ACTION_NOT_QUALIFIED
            ),
        }
        atomic_write_new_json(
            temporary / "compact_report.json",
            compact_report,
            artifact_kind="compact report",
        )

        refs = _artifact_refs(temporary)
        manifest = {
            "schema": ARTIFACT_SCHEMA,
            "task_id": TASK_ID,
            "hypothesis": HYPOTHESIS,
            "status": "valid_completed",
            "classification": classification,
            "configuration": {
                "path": _relative_path(config["_path"]),
                "sha256": config["_file_sha256"],
                "canonical_sha256": config["_canonical_sha256"],
            },
            "upstream": {
                "s1_manifest_sha256": S1_MANIFEST_SHA256,
                "s1_gate": upstream["s1_manifest"]["gate"],
                "r2_manifest_sha256": R2_MANIFEST_SHA256,
                "r2_payload_sha256": R2_PAYLOAD_SHA256,
                "checkpoint_path": _relative_path(upstream["checkpoint_path"]),
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "checkpoint_global_step": 5000,
            },
            "dataset": {
                "mra_manifest_sha256": MRA_MANIFEST_SHA256,
                "fit_count": 50000,
                "validation_count": 5000,
                "s2_confirmation_reserved_count": 5000,
                "s2_selection_reserved_count": 1000,
                "s2_confirmation_consumed": False,
                "s2_selection_consumed": False,
            },
            "output_domain": {
                "raw_camera_shape": ["N", 1, 200, 200],
                "processed_view_shape": ["N", 1, 28, 28],
                "processed_view_dtype": "torch.float32",
                "processed_view_range": [0.0, 1.0],
                "per_sample_normalization": False,
                "raw_camera_persisted": False,
                "output_domain_contract_sha256": output_contract_hash,
            },
            "exposure": dict(exposure),
            "evaluator": {
                "architecture": "FrozenMNISTEvaluator",
                "schema": "frozen_mnist_evaluator_v1",
                "checkpoint": "frozen_output_semantic_evaluator.pt",
                "checkpoint_sha256": checkpoint_hash,
                "checkpoint_selection": "final_epoch",
            },
            "training": {
                "epochs": TRAIN_EPOCHS,
                "batch_size": TRAIN_BATCH_SIZE,
                "optimizer": "Adam",
                "learning_rate": 0.001,
                "initialization_seed": TRAIN_SEED,
                "validation_model_selection": False,
                "training_contract_sha256": training_contract_hash,
            },
            "qualification": qualification_payload,
            "leakage_audit": {
                "path": "leakage_audit.json",
                "sha256": file_sha256(temporary / "leakage_audit.json"),
                "valid": bool(leakage["valid"]),
            },
            "optical_state_isolation": {
                "path": "optical_state_isolation.json",
                "sha256": file_sha256(temporary / "optical_state_isolation.json"),
                "status": isolation["status"],
            },
            "freeze_replay": {
                "path": "evaluator_freeze_replay.json",
                "sha256": file_sha256(temporary / "evaluator_freeze_replay.json"),
                "status": replay["status"],
            },
            "resource": {
                **dict(resource),
                "peak_cuda_bytes": int(resource["peak_cuda_bytes"]),
                "free_disk_bytes_after": int(resource["free_disk_bytes_after"]),
                "wall_clock_seconds": time.monotonic() - started_at,
            },
            "authorization": {
                "optical_training_authorized": False,
                "evaluator_build_authorized": "bounded_only",
                "after_build_evaluator_build_authorized": False,
                "training_authorized": False,
                "formal_promotion": False,
                "s2_execution": False,
                "s2_preview_executed": False,
            },
            "governance": dict(config["governance"]),
            "next_action": (
                NEXT_ACTION_READY
                if classification == FROZEN_OUTPUT_SEMANTIC_EVALUATOR_READY
                else NEXT_ACTION_NOT_QUALIFIED
            ),
            "layout": {
                "write_once": True,
                "manifest_self_hash_excluded": True,
                "all_paths_relative": True,
                "raw_camera_not_duplicated": True,
                "artifacts": refs,
            },
            "artifact_storage": {
                "corpus_sha256": corpus_hash,
                "corpus_size_bytes": (temporary / "output_domain_corpus.pt").stat().st_size,
                "identity_record_count": len(corpus.identity_records),
            },
        }
        atomic_write_new_json(
            temporary / "evaluator_manifest.json",
            manifest,
            artifact_kind="FOSEA manifest",
        )
        _assert_final_tree(temporary)
        if output_path.exists():
            raise FileExistsError(f"write-once output already exists: {output_path}")
        os.replace(temporary, output_path)
        return output_path
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def run(
    config_path: str | Path = CONFIG_PATH,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    started_at = time.monotonic()
    git_state = base._git_state()
    _require_expected_worktree(git_state)
    config_file = _resolve(config_path)
    config = load_config(config_file)
    config = dict(config)
    config["_path"] = config_file
    config["_file_sha256"] = file_sha256(config_file)
    config["_canonical_sha256"] = canonical_sha256(
        {key: value for key, value in config.items() if not key.startswith("_")}
    )
    requested = _resolve(output_path or config["output"]["path"])
    actual_output = next_write_once_output(requested)
    resource = _resource_preflight(config, actual_output)
    exposure = _validate_exposure_provenance(config)
    upstream = _load_upstream(config)
    data = upstream["data"]
    splits = registered_split_indices(config, data.selection_indices)
    if not torch.equal(splits["s2_confirmation"], data.confirmation_indices):
        raise ValueError("FOSEA S2 confirmation identity does not match registered horizon")
    leakage = audit_identity_leakage(
        splits["fit"],
        splits["validation"],
        splits["s2_confirmation"],
        splits["s2_selection"],
    )
    if not leakage["valid"]:
        raise RuntimeError(f"{BLOCKED_EVALUATOR_IDENTITY_LEAKAGE}: split audit")

    device = torch.device(config["execution"]["device"])
    model = mra._load_scaled_model(
        upstream["parent"]["formal"], upstream["checkpoint"], device
    )
    before = _optical_state_snapshot(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    corpus = _build_output_corpus(
        model,
        data,
        splits,
        device,
        EXPOSURE_SCALE,
        int(config["execution"]["optical_batch_size"]),
    )
    after = _optical_state_snapshot(model)
    isolation = _optical_state_isolation(before, after)
    if not isolation["state_unchanged"]:
        raise RuntimeError(f"{BLOCKED_OPTICAL_STATE_ISOLATION}: optical state changed")

    preflight = _run_three_step_preflight(
        corpus.fit_views, corpus.fit_labels, device, config
    )
    evaluator, training_metrics = _train_evaluator(
        corpus.fit_views, corpus.fit_labels, device, config
    )
    validation_metrics = _evaluate_evaluator(
        evaluator, corpus.validation_views, corpus.validation_labels, device
    )
    evaluator.freeze()
    resource["peak_cuda_bytes"] = int(torch.cuda.max_memory_allocated(device))
    if resource["peak_cuda_bytes"] > int(config["execution"]["max_peak_cuda_bytes"]):
        raise RuntimeError("BLOCKED_EVALUATOR_RESOURCE: peak CUDA memory exceeded")
    resource["free_disk_bytes_after"] = int(
        shutil.disk_usage(base._repo_root().anchor or str(base._repo_root())).free
    )
    if resource["free_disk_bytes_after"] < int(config["execution"]["min_disk_free_bytes"]):
        raise RuntimeError("BLOCKED_EVALUATOR_RESOURCE: disk floor after build")

    qualification = qualification_classification(
        validation_accuracy=float(validation_metrics["overall_accuracy"]),
        validation_metrics_finite=bool(validation_metrics["finite"]),
        contract_valid=True,
        leakage_free=bool(leakage["valid"]),
        optical_isolated=bool(isolation["state_unchanged"]),
        freeze_replay=True,
    )
    checkpoint_fd, checkpoint_name = tempfile.mkstemp(suffix=".pt")
    os.close(checkpoint_fd)
    temporary_checkpoint = Path(checkpoint_name)
    try:
        torch.save(
            {
                "schema": CHECKPOINT_SCHEMA,
                "architecture_kind": "frozen_mnist_evaluator_v1",
                "state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in evaluator.state_dict().items()
                },
            },
            temporary_checkpoint,
        )
        replay = _freeze_replay(
            temporary_checkpoint,
            file_sha256(temporary_checkpoint),
            evaluator,
            corpus.validation_views,
            device,
        )
    finally:
        temporary_checkpoint.unlink(missing_ok=True)
    qualification_metrics: dict[str, Any] = {
        "schema": QUALIFICATION_SCHEMA,
        "validation": validation_metrics,
        "threshold": EVALUATOR_THRESHOLD,
        "gates": {
            "validation_metrics_finite": bool(validation_metrics["finite"]),
            "contract_valid": True,
            "leakage_free": bool(leakage["valid"]),
            "optical_isolated": bool(isolation["state_unchanged"]),
            "freeze_replay": replay["status"] == "EVALUATOR_FREEZE_REPLAY_PASS",
        },
        "classification": qualification,
        "validation_model_selection": False,
        "s2_confirmation_evaluated": False,
        "s2_selection_evaluated": False,
        "_evaluator": evaluator,
    }
    if qualification_metrics["gates"]["freeze_replay"] is not True:
        raise RuntimeError(f"{BLOCKED_EVALUATOR_FREEZE_REPLAY}: replay status")
    output = _publish_artifacts(
        actual_output,
        config,
        upstream,
        splits,
        corpus,
        leakage,
        exposure,
        preflight,
        training_metrics,
        qualification_metrics,
        isolation,
        replay,
        resource,
        started_at,
    )
    manifest = json.loads((output / "evaluator_manifest.json").read_text(encoding="utf-8"))
    return {
        "task_id": TASK_ID,
        "classification": manifest["classification"],
        "output": _relative_path(output),
        "validation_accuracy": manifest["qualification"]["validation"][
            "overall_accuracy"
        ],
        "output_manifest_sha256": file_sha256(output / "evaluator_manifest.json"),
        "s2_executed": False,
        "optical_training": False,
        "next_action": manifest["next_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
