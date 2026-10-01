"""Execute the frozen one-shot LightGen complex-OLS post-training evaluation.

The consumer has one protected-data entry point and a fail-closed stage order:
real-path control, latent compatibility, direct prior generation, semantic
distribution, diversity, novelty, and one final classification.  It never
creates an optimizer, calls backward, searches checkpoints, changes an
evaluator/prior/threshold, or loads the consumed N3 sealed values.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training import GAUGE_EPSILON, LATENT_DIMENSION, LightGen2DComplexOLSTrainingConsumer, OLS_PORT_COUNT, PRIOR_SIGMA, PRODUCTION_PROPAGATION_GRID, derive_prior_seed, load_frozen_training_contract, gauge_invariant_ols_representation
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import (
    load_frozen_evaluator,
    stratified_mnist_indices,
)
from scripts.research.lightgen import define_lightgen_2d_semantic_complex_ols_post_training_evaluation_contract_v1 as contract_definition


TASK_ID = "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_V1"
GATE = TASK_ID
CONTRACT_PATH = contract_definition.OUTPUT_DIR / contract_definition.CONTRACT_NAME
EXPECTED_CONTRACT_SHA256 = "b750bafd9c3866707a7d492c55c12ee08ab4775da9f6878805317feccb937a1d"

OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_post_training_evaluation_v1_attempt_01"
)
CONTROL_ARTIFACT_NAME = "lightgen_2d_semantic_complex_ols_post_training_evaluation_control.v1.pt"
GENERATED_ARTIFACT_NAME = "lightgen_2d_semantic_complex_ols_post_training_evaluation_generated_banks.v1.pt"
STATISTICS_ARTIFACT_NAME = "lightgen_2d_semantic_complex_ols_post_training_evaluation_statistics.v1.pt"
LINEAGE_NAME = "lightgen_2d_semantic_complex_ols_post_training_evaluation_lineage.v1.json"
REPORT_NAME = "lightgen_2d_semantic_complex_ols_post_training_evaluation_report.v1.json"

CONTROL_ARTIFACT_SCHEMA = "lightgen_2d_semantic_complex_ols_post_training_evaluation_control.v1"
GENERATED_ARTIFACT_SCHEMA = "lightgen_2d_semantic_complex_ols_post_training_evaluation_generated_banks.v1"
STATISTICS_ARTIFACT_SCHEMA = "lightgen_2d_semantic_complex_ols_post_training_evaluation_statistics.v1"
LINEAGE_SCHEMA = "lightgen_2d_semantic_complex_ols_post_training_evaluation_lineage.v1"
REPORT_SCHEMA = "lightgen_2d_semantic_complex_ols_post_training_evaluation_report.v1"

EXPECTED_BRANCH = contract_definition.EXPECTED_BRANCH
EXPECTED_STARTING_HEAD = contract_definition.EXPECTED_STARTING_HEAD
CHECKPOINT_PATH = contract_definition.CHECKPOINT_PATH
CHECKPOINT_SHA256 = contract_definition.CHECKPOINT_SHA256
TRAINING_SOURCE_PATH = contract_definition.TRAINING_SOURCE_PATH
TRAINING_SOURCE_SHA256 = contract_definition.TRAINING_SOURCE_SHA256
PROTECTED_TEST_PATH = contract_definition.PROTECTED_TEST_PATH
PROTECTED_TEST_SHA256 = contract_definition.PROTECTED_TEST_SHA256
EVALUATOR_PATH = contract_definition.EVALUATOR_PATH
EVALUATOR_SHA256 = contract_definition.EVALUATOR_SHA256
EVALUATOR_SOURCE_PATH = contract_definition.EVALUATOR_SOURCE_PATH
EVALUATOR_SOURCE_SHA256 = contract_definition.EVALUATOR_SOURCE_SHA256
ALIGNMENT_RUNTIME_PATH = contract_definition.ALIGNMENT_RUNTIME_PATH
ALIGNMENT_RUNTIME_SHA256 = contract_definition.ALIGNMENT_RUNTIME_SHA256
TRAINING_CONSUMER_PATH = contract_definition.TRAINING_CONSUMER_PATH
TRAINING_CONSUMER_SHA256 = contract_definition.TRAINING_CONSUMER_SHA256
TRAINING_CONTRACT_PATH = contract_definition.TRAINING_CONTRACT_PATH
TRAINING_CONTRACT_SHA256 = contract_definition.TRAINING_CONTRACT_SHA256
TRAINING_REPORT_PATH = contract_definition.TRAINING_REPORT_PATH
TRAINING_REPORT_SHA256 = contract_definition.TRAINING_REPORT_SHA256
TRAINING_LINEAGE_PATH = contract_definition.TRAINING_LINEAGE_PATH
TRAINING_LINEAGE_SHA256 = contract_definition.TRAINING_LINEAGE_SHA256
ALIGNMENT_CONTRACT_PATH = contract_definition.ALIGNMENT_CONTRACT_PATH
ALIGNMENT_CONTRACT_SHA256 = contract_definition.ALIGNMENT_CONTRACT_SHA256
ALIGNMENT_RUNTIME_REPORT_PATH = contract_definition.ALIGNMENT_RUNTIME_REPORT_PATH
ALIGNMENT_RUNTIME_REPORT_SHA256 = contract_definition.ALIGNMENT_RUNTIME_REPORT_SHA256
EVALUATOR_QUALIFICATION_PATH = contract_definition.EVALUATOR_QUALIFICATION_PATH
EVALUATOR_QUALIFICATION_SHA256 = contract_definition.EVALUATOR_QUALIFICATION_SHA256
EVALUATOR_SPLIT_MANIFEST_PATH = contract_definition.EVALUATOR_SPLIT_MANIFEST_PATH
EVALUATOR_SPLIT_MANIFEST_SHA256 = contract_definition.EVALUATOR_SPLIT_MANIFEST_SHA256
EVALUATOR_QUALIFICATION_THRESHOLD = contract_definition.EVALUATOR_QUALIFICATION_THRESHOLD
DEVELOPMENT_CALIBRATION_INDICES_SHA256 = contract_definition.DEVELOPMENT_CALIBRATION_INDICES_SHA256
PROTECTED_PANEL_SEED = contract_definition.PROTECTED_PANEL_SEED
PROTECTED_PANEL_COUNT = contract_definition.PROTECTED_PANEL_COUNT
PROTECTED_PANEL_PER_CLASS = contract_definition.PROTECTED_PANEL_PER_CLASS
PROTECTED_EVALUATION_COUNT = contract_definition.PROTECTED_EVALUATION_COUNT
DEVELOPMENT_CALIBRATION_COUNT = contract_definition.DEVELOPMENT_CALIBRATION_COUNT
TRAIN_REFERENCE_COUNT = 55_000
CLASS_COUNT = contract_definition.CLASS_COUNT
IMAGE_SHAPE = (1, 28, 28)
RAW_OUTPUT_SHAPE = (1, 200, 200)
BATCH_SIZE = 10
MONTE_CARLO_REPLICATES = contract_definition.MONTE_CARLO_REPLICATES
MONTE_CARLO_ALPHA = contract_definition.MONTE_CARLO_ALPHA
LATENT_BANDWIDTH_SQUARED = contract_definition.LATENT_MMD_BANDWIDTH_SQUARED
SEMANTIC_BANDWIDTH_SQUARED = contract_definition.SEMANTIC_MMD_BANDWIDTH_SQUARED
PRIOR_NAMESPACE = contract_definition.PRIOR_NAMESPACE
PRIOR_GLOBAL_STEP = contract_definition.PRIOR_GLOBAL_STEP
LATENT_MODEL_ROLE = contract_definition.LATENT_MODEL_ROLE
GENERATION_BANK_COUNT = contract_definition.GENERATION_BANK_COUNT
GENERATION_BANK_SIZE = contract_definition.GENERATION_BANK_SIZE
GENERATION_TOTAL_COUNT = GENERATION_BANK_COUNT * GENERATION_BANK_SIZE
GENERATION_ROLE_PREFIX = contract_definition.GENERATION_ROLE_PREFIX
METRIC_NULL_NAMESPACE = contract_definition.METRIC_NULL_NAMESPACE
CONTROL_FLOOR = contract_definition.CONTROL_FLOOR
CONTROL_ALLOWED_DROP = contract_definition.CONTROL_ALLOWED_DROP

CONTROL_FAILURE_CLASSIFICATION = contract_definition.CONTROL_NOT_ESTABLISHED_CLASSIFICATION
NOT_SUPPORTED_CLASSIFICATION = contract_definition.NOT_SUPPORTED_CLASSIFICATION
SUPPORTED_CLASSIFICATION = contract_definition.SUPPORTED_CLASSIFICATION


class BlockedExecution(RuntimeError):
    """A frozen authority is unavailable or stale."""


class InvalidHarnessError(RuntimeError):
    """A complete measurement cannot be trusted because the harness is invalid."""


def _require(condition: bool, message: str, *, blocked: bool = False) -> None:
    if condition:
        return
    error = BlockedExecution if blocked else InvalidHarnessError
    raise error(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    status = result.stdout.splitlines()
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in status],
        "worktree_dirty": bool(status),
    }


def _authority_ref(repo_root: Path, path: Path, role: str, expected_sha256: str) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{role} is missing: {path.as_posix()}", blocked=True)
    actual = file_sha256(target)
    _require(actual == expected_sha256, f"{role} SHA256 changed", blocked=True)
    return {"path": _relative(repo_root, target), "sha256": actual, "role": role}


def _safe_float(value: torch.Tensor, name: str) -> float:
    _require(torch.is_tensor(value) and value.numel() == 1, f"{name} must be scalar")
    _require(torch.isfinite(value).all().item(), f"{name} is nonfinite")
    return float(value.detach().cpu())


def _validate_image_tensor(images: torch.Tensor, name: str, *, count: int | None = None) -> None:
    _require(torch.is_tensor(images), f"{name} is not a tensor")
    _require(images.ndim == 4 and tuple(images.shape[1:]) == IMAGE_SHAPE, f"{name} shape changed")
    if count is not None:
        _require(images.shape[0] == count, f"{name} count changed")
    _require(images.dtype == torch.float32 and not torch.is_complex(images), f"{name} dtype changed")
    _require(torch.isfinite(images).all().item(), f"{name} is nonfinite")
    _require(bool((images >= 0.0).all()) and bool((images <= 1.0).all()), f"{name} is outside [0,1]")


def _load_mnist_file(repo_root: Path, path: Path, expected_sha256: str, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    target = repo_root / path
    _require(target.is_file(), f"{name} is missing", blocked=True)
    _require(file_sha256(target) == expected_sha256, f"{name} SHA256 changed", blocked=True)
    try:
        payload = torch.load(target, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError) as exc:
        raise BlockedExecution(f"{name} cannot be read") from exc
    _require(isinstance(payload, (tuple, list)) and len(payload) == 2, f"{name} layout changed", blocked=True)
    images, labels = payload
    _require(torch.is_tensor(images) and torch.is_tensor(labels), f"{name} tensors are missing", blocked=True)
    _require(images.ndim == 3 and tuple(images.shape[-2:]) == (28, 28), f"{name} image shape changed", blocked=True)
    _require(labels.ndim == 1 and labels.shape[0] == images.shape[0], f"{name} label shape changed", blocked=True)
    if images.dtype == torch.uint8:
        image_values = images.unsqueeze(1).to(torch.float32).div_(255.0)
    else:
        image_values = images.unsqueeze(1).to(torch.float32)
    label_values = labels.to(torch.long)
    _validate_image_tensor(image_values, name)
    _require(bool(((label_values >= 0) & (label_values < CLASS_COUNT)).all()), f"{name} labels changed", blocked=True)
    return image_values.contiguous(), label_values.contiguous()


def _derive_training_roles(repo_root: Path) -> dict[str, torch.Tensor]:
    images, labels = _load_mnist_file(repo_root, TRAINING_SOURCE_PATH, TRAINING_SOURCE_SHA256, "training source")
    _require(labels.numel() == 60_000, "training source count changed", blocked=True)
    train_indices, development_indices = stratified_mnist_indices(labels, seed=1234, validation_per_label=500)
    generator = torch.Generator(device="cpu").manual_seed(4321)
    development_labels = labels.index_select(0, development_indices)
    selected: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        matches = development_indices[development_labels == digit]
        _require(matches.numel() == 500, f"development class {digit} count changed", blocked=True)
        selected.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:100]))
    selection_indices = torch.cat(selected).contiguous()
    calibration_indices = development_indices[~torch.isin(development_indices, selection_indices)].contiguous()
    _require(train_indices.numel() == TRAIN_REFERENCE_COUNT, "train role count changed", blocked=True)
    _require(development_indices.numel() == 5_000, "development role count changed", blocked=True)
    _require(selection_indices.numel() == 1_000, "checkpoint selection role count changed", blocked=True)
    _require(calibration_indices.numel() == DEVELOPMENT_CALIBRATION_COUNT, "calibration role count changed", blocked=True)
    _require(tensor_sha256(train_indices) == "42980fdbe39cafeb89ae86e21fd818dc2e2404bb7f89193f09bd6c530f4a846c", "train role identity changed", blocked=True)
    _require(tensor_sha256(development_indices) == "5c31afc027134a797ffb022fcbfcd1a95fec53c1119ceaf8ca8d01806e5f017e", "development role identity changed", blocked=True)
    _require(tensor_sha256(selection_indices) == "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8", "selection role identity changed", blocked=True)
    _require(tensor_sha256(calibration_indices) == DEVELOPMENT_CALIBRATION_INDICES_SHA256, "calibration role identity changed", blocked=True)
    calibration_counts = torch.bincount(labels.index_select(0, calibration_indices), minlength=CLASS_COUNT)
    _require(torch.equal(calibration_counts, torch.full((CLASS_COUNT,), 400)), "calibration class counts changed", blocked=True)
    return {
        "images": images,
        "labels": labels,
        "train_indices": train_indices,
        "development_indices": development_indices,
        "selection_indices": selection_indices,
        "calibration_indices": calibration_indices,
    }


def _protected_panel_indices(labels: torch.Tensor) -> torch.Tensor:
    _require(labels.ndim == 1 and labels.numel() == PROTECTED_EVALUATION_COUNT, "protected label role changed", blocked=True)
    generator = torch.Generator(device="cpu").manual_seed(PROTECTED_PANEL_SEED)
    selected: list[torch.Tensor] = []
    for digit in range(CLASS_COUNT):
        matches = torch.nonzero(labels == digit, as_tuple=False).flatten()
        _require(matches.numel() >= PROTECTED_PANEL_PER_CLASS, f"protected class {digit} is too small", blocked=True)
        selected.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:PROTECTED_PANEL_PER_CLASS]))
    panel = torch.cat(selected).contiguous()
    _require(panel.numel() == PROTECTED_PANEL_COUNT and torch.unique(panel).numel() == PROTECTED_PANEL_COUNT, "protected panel identity is invalid", blocked=True)
    counts = torch.bincount(labels.index_select(0, panel), minlength=CLASS_COUNT)
    _require(torch.equal(counts, torch.full((CLASS_COUNT,), PROTECTED_PANEL_PER_CLASS)), "protected panel balance changed", blocked=True)
    return panel


def _preflight(repo_root: Path) -> dict[str, Any]:
    try:
        git_before = contract_definition._validate_git_boundary(repo_root)
        contract = contract_definition._read_json(
            repo_root,
            CONTRACT_PATH,
            "post-training evaluation contract",
            EXPECTED_CONTRACT_SHA256,
        )
        alignment_payload, alignment_authority = contract_definition._validate_alignment_authority(repo_root)
        training_authority = contract_definition._validate_training_authority(repo_root)
        evaluator_authority = contract_definition._validate_evaluator_authority(repo_root)
        dataset_authority = contract_definition._validate_dataset_authority(repo_root, training_authority)
    except contract_definition.DefinitionBlocked as exc:
        raise BlockedExecution(str(exc)) from exc

    _require(contract.get("schema") == contract_definition.CONTRACT_SCHEMA, "evaluation contract schema changed", blocked=True)
    _require(contract.get("status") == "valid_completed" and contract.get("result") == "READY", "evaluation contract is not READY", blocked=True)
    _require(contract.get("classification") == contract_definition.READY_CLASSIFICATION, "evaluation contract classification changed", blocked=True)
    _require(contract.get("training_authorized") is False, "evaluation contract opened training", blocked=True)
    _require(contract.get("formal_promotion") is False, "evaluation contract opened promotion", blocked=True)
    _require(contract.get("generation_readiness") == "not_ready", "evaluation contract opened readiness", blocked=True)
    protocol = _mapping(contract.get("evaluation_protocol"), "evaluation protocol")
    runtime = _mapping(protocol.get("runtime"), "evaluation runtime protocol")
    _require(runtime.get("device", "").startswith("cuda:0"), "evaluation device protocol changed", blocked=True)
    _require(runtime.get("autocast") is False and runtime.get("batch_size") == BATCH_SIZE, "evaluation precision/batch protocol changed", blocked=True)
    _require(runtime.get("raw_output_shape") == list(RAW_OUTPUT_SHAPE), "raw output protocol changed", blocked=True)
    _require(runtime.get("metric_accumulation_dtype") == "torch.float64", "metric dtype protocol changed", blocked=True)
    common = _mapping(protocol.get("common_statistic_rules"), "common statistic rules")
    _require(common.get("monte_carlo_replicates") == MONTE_CARLO_REPLICATES, "Monte Carlo replicate count changed", blocked=True)
    _require(common.get("alpha") == MONTE_CARLO_ALPHA and common.get("p_value") == "p_MC=(1 + #{b: T_b >= T_obs})/(B+1)", "Monte Carlo decision rule changed", blocked=True)
    _require(common.get("metric_search") is False and common.get("protected_retries") is False, "metric retry/search protocol changed", blocked=True)
    generation = _mapping(protocol.get("stage_3_direct_prior_generation"), "generation protocol")
    banks = _mapping(generation.get("banks"), "generation banks")
    _require(banks.get("count") == GENERATION_BANK_COUNT and banks.get("samples_per_bank") == GENERATION_BANK_SIZE, "generation bank protocol changed", blocked=True)
    _require(banks.get("total_samples") == GENERATION_BANK_COUNT * GENERATION_BANK_SIZE, "generation sample count changed", blocked=True)
    _require(contract.get("generator_input_writer", {}).get("same_writer") is True, "real/generated writer identity changed", blocked=True)
    _require(contract.get("generator_input_writer", {}).get("same_generator") is True, "real/generated generator identity changed", blocked=True)
    totality = _mapping(contract.get("totality_checks"), "evaluation contract totality")
    _require(totality.get("sealed_values_loaded") is False and totality.get("sealed_values_reused") is False, "sealed boundary changed", blocked=True)
    _require(totality.get("same_real_and_generated_w_ols") is True and totality.get("same_downstream_generator") is True, "shared path proof changed", blocked=True)

    checkpoint = safe_torch_load(repo_root / CHECKPOINT_PATH, expected_sha256=CHECKPOINT_SHA256, map_location="cpu")
    _require(isinstance(checkpoint, Mapping), "selected checkpoint is not a mapping", blocked=True)
    _require(checkpoint.get("schema") == "lightgen_2d_semantic_complex_ols_training_checkpoint.v1", "checkpoint schema changed", blocked=True)
    _require(checkpoint.get("valid_training_checkpoint") is True and checkpoint.get("accepted_training_checkpoint") is False, "checkpoint validity boundary changed", blocked=True)
    _require(
        checkpoint.get("checkpoint_selection_candidate") is True
        and checkpoint.get("global_step") == 99
        and checkpoint.get("completed_steps") == 100
        and checkpoint.get("development_evaluation", {}).get("checkpoint_step") == 100,
        "step-100 checkpoint identity changed",
        blocked=True,
    )
    _require(checkpoint.get("contract", {}).get("sha256") == TRAINING_CONTRACT_SHA256, "checkpoint training contract binding changed", blocked=True)
    _require(checkpoint.get("runtime", {}).get("module_sha256") == ALIGNMENT_RUNTIME_SHA256, "checkpoint runtime binding changed", blocked=True)
    _require(checkpoint.get("runtime", {}).get("transport") == "T=I_100 coefficient space", "checkpoint transport binding changed", blocked=True)
    parameter_state = checkpoint.get("trainable_parameter_state")
    _require(isinstance(parameter_state, Mapping) and set(parameter_state) == {"encoder.phase", "generator.phase"}, "checkpoint trainable state changed", blocked=True)

    protected_path = repo_root / PROTECTED_TEST_PATH
    _require(protected_path.is_file(), "protected test source is missing", blocked=True)
    protected_sha_before_values = file_sha256(protected_path)
    _require(protected_sha_before_values == PROTECTED_TEST_SHA256, "protected test source SHA256 changed", blocked=True)
    _require(torch.cuda.is_available(), "CUDA is required by the frozen 2048-grid evaluation", blocked=True)
    _require(torch.cuda.device_count() >= 1, "cuda:0 is unavailable", blocked=True)
    return {
        "git_before": git_before,
        "contract": contract,
        "alignment_payload": alignment_payload,
        "alignment_authority": alignment_authority,
        "training_authority": training_authority,
        "evaluator_authority": evaluator_authority,
        "dataset_authority": dataset_authority,
        "checkpoint": checkpoint,
        "checkpoint_safe_load_verified": True,
        "protected_sha_before_values": protected_sha_before_values,
        "protected_value_load_count": 0,
    }


def _load_evaluator_authority(repo_root: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    _authority_ref(repo_root, EVALUATOR_PATH, "semantic evaluator checkpoint", EVALUATOR_SHA256)
    _authority_ref(repo_root, EVALUATOR_SOURCE_PATH, "semantic evaluator implementation", EVALUATOR_SOURCE_SHA256)
    try:
        evaluator, payload = load_frozen_evaluator(
            repo_root / EVALUATOR_PATH,
            device=device,
            min_validation_accuracy=EVALUATOR_QUALIFICATION_THRESHOLD,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise BlockedExecution("frozen semantic evaluator cannot be loaded") from exc
    _require(payload.get("architecture_kind") == "frozen_mnist_evaluator_v1", "semantic evaluator architecture changed", blocked=True)
    _require(float(payload.get("validation_accuracy", 0.0)) >= EVALUATOR_QUALIFICATION_THRESHOLD, "semantic evaluator qualification changed", blocked=True)
    return evaluator, payload


def _evaluator_view(raw_cmos: torch.Tensor) -> torch.Tensor:
    _require(raw_cmos.ndim == 4 and tuple(raw_cmos.shape[1:]) == RAW_OUTPUT_SHAPE, "raw CMOS shape changed")
    _require(raw_cmos.dtype == torch.float32 and not torch.is_complex(raw_cmos), "raw CMOS dtype changed")
    _require(torch.isfinite(raw_cmos).all().item(), "raw CMOS is nonfinite")
    view = F.adaptive_avg_pool2d(raw_cmos.clamp(0.0, 1.0), (28, 28))
    _validate_image_tensor(view, "evaluator view")
    return view


def _evaluate_views(evaluator: torch.nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_image_tensor(images, "evaluator images")
    features, logits = evaluator(images)
    _require(features.ndim == 2 and features.shape[1] == LATENT_DIMENSION - 72, "semantic feature dimension changed")
    _require(logits.ndim == 2 and logits.shape[1] == CLASS_COUNT, "semantic logits dimension changed")
    _require(torch.isfinite(features).all().item() and torch.isfinite(logits).all().item(), "semantic evaluator output is nonfinite")
    norms = features.norm(p=2, dim=1)
    _require(bool((norms > 1.0e-12).all()), "semantic feature normalization is undefined")
    normalized = F.normalize(features, p=2, dim=1)
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    _require(torch.isfinite(normalized).all().item() and torch.isfinite(probabilities).all().item(), "semantic evaluator normalization is nonfinite")
    return normalized, features, logits, probabilities, predictions


def _clean_evaluator_bank(
    evaluator: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor | float]:
    selected_images = images.index_select(0, indices)
    selected_labels = labels.index_select(0, indices)
    _validate_image_tensor(selected_images, "clean evaluator bank")
    normalized_chunks: list[torch.Tensor] = []
    feature_chunks: list[torch.Tensor] = []
    logit_chunks: list[torch.Tensor] = []
    probability_chunks: list[torch.Tensor] = []
    prediction_chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, selected_images.shape[0], BATCH_SIZE):
            batch = selected_images[start : start + BATCH_SIZE].to(device)
            normalized, features, logits, probabilities, predictions = _evaluate_views(evaluator, batch)
            normalized_chunks.append(normalized.cpu())
            feature_chunks.append(features.cpu())
            logit_chunks.append(logits.cpu())
            probability_chunks.append(probabilities.cpu())
            prediction_chunks.append(predictions.cpu())
    normalized = torch.cat(normalized_chunks).contiguous()
    features = torch.cat(feature_chunks).contiguous()
    logits = torch.cat(logit_chunks).contiguous()
    probabilities = torch.cat(probability_chunks).contiguous()
    predictions = torch.cat(prediction_chunks).contiguous()
    counts = torch.bincount(selected_labels, minlength=CLASS_COUNT)
    correct = torch.bincount(selected_labels[predictions == selected_labels], minlength=CLASS_COUNT)
    recall = correct.to(torch.float64) / counts.to(torch.float64)
    macro_recall = recall.mean()
    _require(torch.isfinite(macro_recall).item(), "clean calibration macro recall is nonfinite")
    return {
        "indices": indices.cpu().contiguous(),
        "labels": selected_labels.cpu().contiguous(),
        "features_l2": normalized,
        "features_raw": features,
        "logits": logits,
        "probabilities": probabilities,
        "predictions": predictions,
        "per_class_recall": recall.cpu(),
        "macro_recall": _safe_float(macro_recall, "clean calibration macro recall"),
    }


def _panel_rank_map(panel_indices: torch.Tensor) -> dict[int, int]:
    return {int(source_index): rank for rank, source_index in enumerate(panel_indices.tolist())}


def _validate_real_path_result(result: Any, *, first: bool) -> None:
    if not first:
        return
    _require(result.image_input is not None, "real path omitted image optical input")
    _require(result.encoder_output_field is not None and result.ols_input_field is not None, "real path omitted encoder/OLS boundary")
    _require(result.z_ols.source == "real_encoded", "real path OLS source changed")
    _require(result.z_ols.coefficients.dtype == torch.complex64 and tuple(result.z_ols.coefficients.shape[1:]) == (OLS_PORT_COUNT,), "real OLS shape/dtype changed")
    _require(result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "modal projection boundary is absent")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "real path W_OLS operation changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "real path transport changed")


def _stage_one_control(
    runtime: LightGen2DComplexOLSTrainingConsumer,
    evaluator: torch.nn.Module,
    protected_images: torch.Tensor,
    protected_labels: torch.Tensor,
    panel_indices: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    count = protected_images.shape[0]
    _validate_image_tensor(protected_images, "protected images", count=PROTECTED_EVALUATION_COUNT)
    panel_ranks = _panel_rank_map(panel_indices)
    protected_features = torch.empty((count, 128), dtype=torch.float32)
    protected_raw_features = torch.empty((count, 128), dtype=torch.float32)
    protected_logits = torch.empty((count, CLASS_COUNT), dtype=torch.float32)
    protected_probabilities = torch.empty((count, CLASS_COUNT), dtype=torch.float32)
    protected_predictions = torch.empty((count,), dtype=torch.long)
    z_panel = torch.empty((PROTECTED_PANEL_COUNT, OLS_PORT_COUNT), dtype=torch.complex64)
    panel_features = torch.empty((PROTECTED_PANEL_COUNT, 128), dtype=torch.float32)
    panel_raw_features = torch.empty((PROTECTED_PANEL_COUNT, 128), dtype=torch.float32)
    panel_logits = torch.empty((PROTECTED_PANEL_COUNT, CLASS_COUNT), dtype=torch.float32)
    panel_probabilities = torch.empty((PROTECTED_PANEL_COUNT, CLASS_COUNT), dtype=torch.float32)
    panel_predictions = torch.empty((PROTECTED_PANEL_COUNT,), dtype=torch.long)
    correct_by_class = torch.zeros((CLASS_COUNT,), dtype=torch.long)
    label_counts = torch.bincount(protected_labels, minlength=CLASS_COUNT)
    first = True
    with torch.inference_mode():
        for start in range(0, count, BATCH_SIZE):
            stop = start + BATCH_SIZE
            image_batch = protected_images[start:stop].to(device)
            result = runtime.runtime.forward_from_image(image_batch)
            _validate_real_path_result(result, first=first)
            first = False
            normalized, raw_features, logits, probabilities, predictions = _evaluate_views(evaluator, _evaluator_view(result.raw_intensity))
            batch_slice = slice(start, stop)
            protected_features[batch_slice] = normalized.cpu()
            protected_raw_features[batch_slice] = raw_features.cpu()
            protected_logits[batch_slice] = logits.cpu()
            protected_probabilities[batch_slice] = probabilities.cpu()
            protected_predictions[batch_slice] = predictions.cpu()
            correct_by_class += torch.bincount(
                protected_labels[start:stop].to(torch.long)[predictions.cpu() == protected_labels[start:stop]],
                minlength=CLASS_COUNT,
            )
            for local, source_index in enumerate(range(start, stop)):
                rank = panel_ranks.get(source_index)
                if rank is not None:
                    z_panel[rank] = result.z_ols.coefficients[local].cpu()
                    panel_features[rank] = normalized[local].cpu()
                    panel_raw_features[rank] = raw_features[local].cpu()
                    panel_logits[rank] = logits[local].cpu()
                    panel_probabilities[rank] = probabilities[local].cpu()
                    panel_predictions[rank] = predictions[local].cpu()
            if (stop % 1000 == 0) or stop == count:
                print(f"stage1 protected control: {stop}/{count}", flush=True)
    _require(torch.isfinite(z_panel.real).all().item() and torch.isfinite(z_panel.imag).all().item(), "protected panel OLS is nonfinite")
    recalls = correct_by_class.to(torch.float64) / label_counts.to(torch.float64)
    macro_recall = recalls.mean()
    return {
        "protected_labels": protected_labels.cpu().contiguous(),
        "protected_features_l2": protected_features,
        "protected_features_raw": protected_raw_features,
        "protected_logits": protected_logits,
        "protected_probabilities": protected_probabilities,
        "protected_predictions": protected_predictions,
        "protected_label_counts": label_counts.cpu().contiguous(),
        "protected_correct_by_class": correct_by_class,
        "protected_per_class_recall": recalls.cpu(),
        "protected_macro_recall": _safe_float(macro_recall, "protected macro recall"),
        "panel_indices": panel_indices.cpu().contiguous(),
        "panel_index_sha256": tensor_sha256(panel_indices),
        "panel_labels": protected_labels.index_select(0, panel_indices).cpu().contiguous(),
        "panel_z_ols": z_panel,
        "panel_features_l2": panel_features,
        "panel_features_raw": panel_raw_features,
        "panel_logits": panel_logits,
        "panel_probabilities": panel_probabilities,
        "panel_predictions": panel_predictions,
        "path_proof": {
            "image_adapter": True,
            "input_relay": True,
            "trained_encoder": True,
            "mandatory_modal_projection": True,
            "coefficient_space_transport": "T=I_100",
            "shared_w_ols": True,
            "shared_generator": True,
            "raw_cmos_output": True,
        },
    }


def _sample_prior_exact(
    sample_count: int,
    *,
    namespace: str,
    role: str,
    global_step: int,
    device: torch.device,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    _require(sample_count > 0 and role and namespace and global_step >= 0, "prior coordinates are invalid")
    seed = derive_prior_seed(namespace=namespace, role=role, global_step=global_step, sample_count=sample_count)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    real = torch.randn((sample_count, OLS_PORT_COUNT), generator=generator, dtype=torch.float32)
    imag = torch.randn((sample_count, OLS_PORT_COUNT), generator=generator, dtype=torch.float32)
    coefficients = torch.complex(real, imag) * (PRIOR_SIGMA / math.sqrt(2.0))
    _require(torch.isfinite(coefficients.real).all().item() and torch.isfinite(coefficients.imag).all().item(), "prior sample is nonfinite")
    return coefficients.to(device), seed, generator.get_state().clone()


def _validate_direct_path_result(result: Any, expected_count: int) -> None:
    _require(result.image_input is None and result.encoder_output_field is None and result.ols_input_field is None, "direct prior path used the image encoder")
    _require(result.encoder_records == (), "direct prior path emitted encoder records")
    _require(result.z_ols.source == "external_generated", "direct prior OLS source changed")
    _require(tuple(result.z_ols.coefficients.shape) == (expected_count, OLS_PORT_COUNT), "direct prior OLS shape changed")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "direct prior W_OLS operation changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "direct prior transport changed")


def _run_generation_bank(
    runtime: LightGen2DComplexOLSTrainingConsumer,
    z: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    raw_outputs = torch.empty((z.shape[0],) + RAW_OUTPUT_SHAPE, dtype=torch.float32)
    first = True
    with torch.inference_mode():
        for start in range(0, z.shape[0], BATCH_SIZE):
            stop = min(start + BATCH_SIZE, z.shape[0])
            result = runtime.runtime.forward_from_ols(z[start:stop].to(device))
            _validate_direct_path_result(result, stop - start)
            if first:
                first = False
            raw = result.raw_intensity.detach()
            _require(tuple(raw.shape[1:]) == RAW_OUTPUT_SHAPE and raw.dtype == torch.float32, "generated raw output shape/dtype changed")
            _require(torch.isfinite(raw).all().item() and bool((raw >= 0.0).all()), "generated raw output is invalid")
            raw_outputs[start:stop] = raw.cpu()
    return raw_outputs


def _metric_features(features: torch.Tensor, device: torch.device) -> torch.Tensor:
    _require(features.ndim == 2 and features.shape[1] == 128, "semantic feature shape changed")
    _require(torch.isfinite(features).all().item(), "metric features are nonfinite")
    return features.to(device=device, dtype=torch.float64)


def _rbf_mean(left: torch.Tensor, right: torch.Tensor, bandwidth_squared: float, *, row_chunk: int = 128) -> torch.Tensor:
    _require(left.ndim == 2 and right.ndim == 2 and left.shape[1] == right.shape[1], "kernel feature dimensions differ")
    _require(left.shape[0] > 0 and right.shape[0] > 0, "kernel banks must be nonempty")
    x = left.to(dtype=torch.float64)
    y = right.to(dtype=torch.float64)
    x_squared = x.square().sum(dim=1)
    y_squared = y.square().sum(dim=1)
    y_transposed = y.transpose(0, 1).contiguous()
    total = torch.zeros((), dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], row_chunk):
        stop = min(start + row_chunk, x.shape[0])
        squared_distance = x_squared[start:stop, None] + y_squared[None, :] - 2.0 * torch.mm(x[start:stop], y_transposed)
        squared_distance = squared_distance.clamp_min(0.0)
        total = total + torch.exp(-squared_distance / (2.0 * bandwidth_squared)).sum(dtype=torch.float64)
    value = total / float(x.shape[0] * y.shape[0])
    _require(torch.isfinite(value).item(), "Gaussian kernel mean is nonfinite")
    return value


def _mmd_vstat(left: torch.Tensor, right: torch.Tensor, bandwidth_squared: float) -> torch.Tensor:
    value = _rbf_mean(left, left, bandwidth_squared) + _rbf_mean(right, right, bandwidth_squared) - 2.0 * _rbf_mean(left, right, bandwidth_squared)
    _require(torch.isfinite(value).item(), "MMD V-statistic is nonfinite")
    return value


def _full_complex_mmd(left: torch.Tensor, right: torch.Tensor, device: torch.device) -> torch.Tensor:
    _require(left.ndim == 2 and right.ndim == 2 and left.shape == right.shape, "full-complex latent banks must have equal shape")
    left_device = left.to(device)
    right_device = right.to(device)
    left_representation = gauge_invariant_ols_representation(left_device, sigma=PRIOR_SIGMA, epsilon=GAUGE_EPSILON)
    right_representation = gauge_invariant_ols_representation(right_device, sigma=PRIOR_SIGMA, epsilon=GAUGE_EPSILON)
    _require(left_representation.shape[1] == LATENT_DIMENSION and right_representation.shape[1] == LATENT_DIMENSION, "full-complex representation dimension changed")
    return _mmd_vstat(left_representation, right_representation, LATENT_BANDWIDTH_SQUARED)


def _seed_record(namespace: str, role: str, global_step: int, sample_count: int, seed: int) -> dict[str, Any]:
    return {
        "namespace": namespace,
        "role": role,
        "global_step": global_step,
        "sample_count": sample_count,
        "seed": seed,
    }


def _seed_records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    return canonical_sha256([dict(record) for record in records])


def _summary(values: torch.Tensor) -> dict[str, Any]:
    values = values.detach().cpu().to(torch.float64).flatten()
    _require(values.numel() > 0 and torch.isfinite(values).all().item(), "summary values are invalid")
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


def _mc_decision(observed: torch.Tensor, null_values: torch.Tensor) -> dict[str, Any]:
    observed = observed.detach().to(torch.float64).cpu().reshape(())
    null_values = null_values.detach().to(torch.float64).cpu().flatten()
    _require(null_values.numel() == MONTE_CARLO_REPLICATES, "Monte Carlo null count changed")
    _require(torch.isfinite(observed).item() and torch.isfinite(null_values).all().item(), "Monte Carlo statistic is nonfinite")
    exceedance = int((null_values >= observed).sum())
    p_value = (1 + exceedance) / float(MONTE_CARLO_REPLICATES + 1)
    return {
        "observed": float(observed),
        "null_summary": _summary(null_values),
        "exceedance_count_ge": exceedance,
        "p_mc": p_value,
        "alpha": MONTE_CARLO_ALPHA,
        "pass": bool(p_value >= MONTE_CARLO_ALPHA),
        "tail": "upper; T_b >= T_obs",
    }


def _draw_metric_indices(role: str, replicate: int, source_count: int, sample_count: int) -> tuple[torch.Tensor, dict[str, Any]]:
    seed = derive_prior_seed(
        namespace=METRIC_NULL_NAMESPACE,
        role=role,
        global_step=replicate,
        sample_count=sample_count,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randint(source_count, (sample_count,), generator=generator, dtype=torch.long)
    return indices, _seed_record(METRIC_NULL_NAMESPACE, role, replicate, sample_count, seed)


def _latent_compatibility(z_eval: torch.Tensor, device: torch.device) -> tuple[dict[str, Any], dict[str, Any]]:
    z_prior, prior_seed, prior_generator_state = _sample_prior_exact(
        PROTECTED_PANEL_COUNT,
        namespace=PRIOR_NAMESPACE,
        role=LATENT_MODEL_ROLE,
        global_step=PRIOR_GLOBAL_STEP,
        device=device,
    )
    z_prior_cpu = z_prior.cpu().contiguous()
    observed = _full_complex_mmd(z_eval, z_prior_cpu, device)
    null_values = torch.empty((MONTE_CARLO_REPLICATES,), dtype=torch.float64)
    seed_records: list[dict[str, Any]] = []
    for replicate in range(MONTE_CARLO_REPLICATES):
        data, data_seed, _ = _sample_prior_exact(
            PROTECTED_PANEL_COUNT,
            namespace=PRIOR_NAMESPACE,
            role=f"post_training_latent_null_data_{replicate}",
            global_step=replicate,
            device=device,
        )
        model, model_seed, _ = _sample_prior_exact(
            PROTECTED_PANEL_COUNT,
            namespace=PRIOR_NAMESPACE,
            role=f"post_training_latent_null_model_{replicate}",
            global_step=replicate,
            device=device,
        )
        null_values[replicate] = _full_complex_mmd(data, model, device).detach().cpu()
        seed_records.extend(
            [
                _seed_record(PRIOR_NAMESPACE, f"post_training_latent_null_data_{replicate}", replicate, PROTECTED_PANEL_COUNT, data_seed),
                _seed_record(PRIOR_NAMESPACE, f"post_training_latent_null_model_{replicate}", replicate, PROTECTED_PANEL_COUNT, model_seed),
            ]
        )
        if (replicate + 1) % 100 == 0 or replicate + 1 == MONTE_CARLO_REPLICATES:
            print(f"stage2 latent null: {replicate + 1}/{MONTE_CARLO_REPLICATES}", flush=True)
    decision = _mc_decision(observed, null_values)
    return (
        {
            "status": "COMPLETED",
            "observed_model_role": LATENT_MODEL_ROLE,
            "prior_seed": prior_seed,
            "prior_generator_state_sha256": tensor_sha256(prior_generator_state),
            "representation": "S(z)=concat(Re(G(z)/sigma_z), Im(G(z)/sigma_z)) in R^200; global-phase regularized anchor",
            "sigma_z": PRIOR_SIGMA,
            "gauge_epsilon": GAUGE_EPSILON,
            "bandwidth_squared": LATENT_BANDWIDTH_SQUARED,
            "sample_count": PROTECTED_PANEL_COUNT,
            "decision": decision,
        },
        {
            "z_prior": z_prior_cpu,
            "null_values": null_values,
            "seed_records": seed_records,
            "seed_records_sha256": _seed_records_sha256(seed_records),
        },
    )


def _class_diagnostics(probabilities: torch.Tensor, predictions: torch.Tensor) -> dict[str, Any]:
    probabilities = probabilities.to(torch.float64).cpu()
    predictions = predictions.cpu()
    entropy = -(probabilities.clamp_min(1.0e-30) * probabilities.clamp_min(1.0e-30).log()).sum(dim=1)
    confidence = probabilities.max(dim=1).values
    histogram = torch.bincount(predictions, minlength=CLASS_COUNT)
    return {
        "count": int(predictions.numel()),
        "class_histogram": histogram.tolist(),
        "class_coverage": int((histogram > 0).sum()),
        "confidence": _summary(confidence),
        "entropy": _summary(entropy),
        "descriptive_only": True,
    }


def _semantic_distribution(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    calibration_features: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    real_metric = _metric_features(real_features, device)
    generated_metric = _metric_features(generated_features, device)
    calibration_metric = _metric_features(calibration_features, device)
    observed = _mmd_vstat(real_metric, generated_metric, SEMANTIC_BANDWIDTH_SQUARED)
    null_values = torch.empty((MONTE_CARLO_REPLICATES,), dtype=torch.float64)
    seed_records: list[dict[str, Any]] = []
    for replicate in range(MONTE_CARLO_REPLICATES):
        left_indices, left_record = _draw_metric_indices("semantic_null_real_a", replicate, calibration_features.shape[0], PROTECTED_PANEL_COUNT)
        right_indices, right_record = _draw_metric_indices("semantic_null_real_b", replicate, calibration_features.shape[0], GENERATION_TOTAL_COUNT)
        left = calibration_metric.index_select(0, left_indices.to(device))
        right = calibration_metric.index_select(0, right_indices.to(device))
        null_values[replicate] = _mmd_vstat(left, right, SEMANTIC_BANDWIDTH_SQUARED).detach().cpu()
        seed_records.extend([left_record, right_record])
        if (replicate + 1) % 100 == 0 or replicate + 1 == MONTE_CARLO_REPLICATES:
            print(f"stage4 semantic null: {replicate + 1}/{MONTE_CARLO_REPLICATES}", flush=True)
    decision = _mc_decision(observed, null_values)
    return (
        {
            "status": "COMPLETED",
            "feature_definition": "L2-normalized FrozenMNISTEvaluator 128D embedding",
            "bandwidth_squared": SEMANTIC_BANDWIDTH_SQUARED,
            "calibration_role": "development_calibration",
            "sample_counts": {"protected": int(real_features.shape[0]), "generated": int(generated_features.shape[0]), "calibration": int(calibration_features.shape[0])},
            "decision": decision,
        },
        {
            "null_values": null_values,
            "seed_records": seed_records,
            "seed_records_sha256": _seed_records_sha256(seed_records),
        },
    )


def _median_pairwise_distance(features: torch.Tensor) -> torch.Tensor:
    _require(features.ndim == 2 and features.shape[0] >= 2, "pairwise bank is too small")
    x = features.to(dtype=torch.float64)
    squared_norm = x.square().sum(dim=1)
    squared_distance = squared_norm[:, None] + squared_norm[None, :] - 2.0 * torch.mm(x, x.transpose(0, 1))
    upper = torch.triu_indices(x.shape[0], x.shape[0], offset=1, device=x.device)
    squared_distance = squared_distance[upper[0], upper[1]].clamp_min(0.0)
    value = squared_distance.median().sqrt()
    _require(torch.isfinite(value).item(), "pairwise median distance is nonfinite")
    return value


def _diversity(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    calibration_features: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    real_metric = _metric_features(real_features, device)
    generated_metric = _metric_features(generated_features, device)
    calibration_metric = _metric_features(calibration_features, device)
    real_median = _median_pairwise_distance(real_metric)
    generated_median = _median_pairwise_distance(generated_metric)
    observed = real_median - generated_median
    null_values = torch.empty((MONTE_CARLO_REPLICATES,), dtype=torch.float64)
    seed_records: list[dict[str, Any]] = []
    for replicate in range(MONTE_CARLO_REPLICATES):
        left_indices, left_record = _draw_metric_indices("diversity_null_real_a", replicate, calibration_features.shape[0], PROTECTED_PANEL_COUNT)
        right_indices, right_record = _draw_metric_indices("diversity_null_real_b", replicate, calibration_features.shape[0], GENERATION_TOTAL_COUNT)
        left = calibration_metric.index_select(0, left_indices.to(device))
        right = calibration_metric.index_select(0, right_indices.to(device))
        null_values[replicate] = (_median_pairwise_distance(left) - _median_pairwise_distance(right)).detach().cpu()
        seed_records.extend([left_record, right_record])
        if (replicate + 1) % 100 == 0 or replicate + 1 == MONTE_CARLO_REPLICATES:
            print(f"stage6 diversity null: {replicate + 1}/{MONTE_CARLO_REPLICATES}", flush=True)
    decision = _mc_decision(observed, null_values)
    return (
        {
            "status": "COMPLETED",
            "feature_definition": "same L2-normalized FrozenMNISTEvaluator 128D embedding",
            "observed_real_median_pairwise_distance": _safe_float(real_median, "real median pairwise distance"),
            "observed_generated_median_pairwise_distance": _safe_float(generated_median, "generated median pairwise distance"),
            "hard_criterion": "upper-tail real_median - generated_median; generated lower dispersion is a failure",
            "decision": decision,
        },
        {
            "null_values": null_values,
            "seed_records": seed_records,
            "seed_records_sha256": _seed_records_sha256(seed_records),
        },
    )


def _nearest_training_distances(
    queries: torch.Tensor,
    references: torch.Tensor,
    device: torch.device,
    *,
    query_chunk: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = _metric_features(queries, device)
    reference = _metric_features(references, device)
    reference_squared = reference.square().sum(dim=1)
    reference_transposed = reference.transpose(0, 1).contiguous()
    best_distance = torch.full((query.shape[0],), float("inf"), dtype=torch.float64, device=device)
    best_index = torch.full((query.shape[0],), -1, dtype=torch.long, device=device)
    for start in range(0, query.shape[0], query_chunk):
        stop = min(start + query_chunk, query.shape[0])
        query_chunk_tensor = query[start:stop]
        distance_squared = (
            query_chunk_tensor.square().sum(dim=1)[:, None]
            + reference_squared[None, :]
            - 2.0 * torch.mm(query_chunk_tensor, reference_transposed)
        ).clamp_min(0.0)
        distances, indices = distance_squared.min(dim=1)
        best_distance[start:stop] = distances.sqrt()
        best_index[start:stop] = indices
    _require(torch.isfinite(best_distance).all().item() and bool((best_index >= 0).all()), "nearest-training relation is invalid")
    return best_distance.cpu(), best_index.cpu()


def _view_digest(view: torch.Tensor) -> str:
    view = view.detach().cpu().contiguous()
    _validate_image_tensor(view, "exact-duplicate view")
    return hashlib.sha256(view.numpy().tobytes()).hexdigest()


def _view_digest_set(views: torch.Tensor) -> set[str]:
    _validate_image_tensor(views, "train reference evaluator views")
    return {_view_digest(views[index : index + 1]) for index in range(views.shape[0])}


def _novelty(
    protected_features: torch.Tensor,
    generated_features: torch.Tensor,
    calibration_features: torch.Tensor,
    train_features: torch.Tensor,
    train_indices: torch.Tensor,
    generated_views: torch.Tensor,
    train_views: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protected_distances, protected_nearest = _nearest_training_distances(protected_features, train_features, device)
    generated_distances, generated_nearest = _nearest_training_distances(generated_features, train_features, device)
    calibration_distances, _ = _nearest_training_distances(calibration_features, train_features, device)
    observed = torch.median(protected_distances) - torch.median(generated_distances)
    null_values = torch.empty((MONTE_CARLO_REPLICATES,), dtype=torch.float64)
    seed_records: list[dict[str, Any]] = []
    for replicate in range(MONTE_CARLO_REPLICATES):
        left_indices, left_record = _draw_metric_indices("novelty_null_real_a", replicate, calibration_features.shape[0], PROTECTED_PANEL_COUNT)
        right_indices, right_record = _draw_metric_indices("novelty_null_real_b", replicate, calibration_features.shape[0], GENERATION_TOTAL_COUNT)
        null_values[replicate] = (
            calibration_distances.index_select(0, left_indices).median()
            - calibration_distances.index_select(0, right_indices).median()
        )
        seed_records.extend([left_record, right_record])
        if (replicate + 1) % 100 == 0 or replicate + 1 == MONTE_CARLO_REPLICATES:
            print(f"stage7 novelty null: {replicate + 1}/{MONTE_CARLO_REPLICATES}", flush=True)
    duplicate_reference = _view_digest_set(train_views)
    generated_duplicate_count = sum(_view_digest(generated_views[index : index + 1]) in duplicate_reference for index in range(generated_views.shape[0]))
    decision = _mc_decision(observed, null_values)
    return (
        {
            "status": "COMPLETED",
            "reference_role": "train_reference",
            "reference_count": int(train_features.shape[0]),
            "observed_protected_median_nearest_distance": float(torch.median(protected_distances)),
            "observed_generated_median_nearest_distance": float(torch.median(generated_distances)),
            "protected_nearest_training_summary": _summary(protected_distances),
            "generated_nearest_training_summary": _summary(generated_distances),
            "generated_exact_duplicate_count": int(generated_duplicate_count),
            "exact_duplicate_definition": "SHA256 of exact float32 [1,28,28] evaluator-view bytes; no sample removal",
            "hard_criterion": "upper-tail protected_median_nearest - generated_median_nearest; generated closer proximity is a failure",
            "decision": decision,
        },
        {
            "protected_nearest_distances": protected_distances,
            "protected_nearest_train_positions": protected_nearest,
            "protected_nearest_train_indices": train_indices.index_select(0, protected_nearest),
            "generated_nearest_distances": generated_distances,
            "generated_nearest_train_positions": generated_nearest,
            "generated_nearest_train_indices": train_indices.index_select(0, generated_nearest),
            "calibration_nearest_distances": calibration_distances,
            "null_values": null_values,
            "seed_records": seed_records,
            "seed_records_sha256": _seed_records_sha256(seed_records),
        },
    )


def _artifact_ref(repo_root: Path, output_dir: Path, temporary: Path, path: Path, role: str, schema: str) -> dict[str, Any]:
    relative = path.relative_to(temporary).as_posix()
    return {
        "path": _relative(repo_root, output_dir / relative),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "role": role,
        "schema": schema,
    }


def _write_and_verify_torch(path: Path, payload: Any, schema: str, artifact_kind: str) -> None:
    atomic_torch_save_new(path, payload, artifact_kind=artifact_kind)
    digest = file_sha256(path)
    loaded = safe_torch_load(path, expected_sha256=digest, map_location="cpu")
    _require(isinstance(loaded, Mapping) and loaded.get("schema") == schema, f"{artifact_kind} reload/schema validation failed")


def _build_control_payload(
    control: Mapping[str, Any],
    clean_calibration: Mapping[str, Any],
    *,
    protected_source: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema": CONTROL_ARTIFACT_SCHEMA,
        "version": 1,
        "protected_source": dict(protected_source),
        "protected_consumption": {"values_read_once": True, "labels_used_for_control_and_fixed_panel_only": True},
        "panel": {
            "indices": control["panel_indices"],
            "indices_sha256": control["panel_index_sha256"],
            "selection_seed": PROTECTED_PANEL_SEED,
            "count": PROTECTED_PANEL_COUNT,
            "per_class": PROTECTED_PANEL_PER_CLASS,
        },
        "control": {
            "labels": control["protected_labels"],
            "features_l2": control["protected_features_l2"],
            "features_raw": control["protected_features_raw"],
            "logits": control["protected_logits"],
            "probabilities": control["protected_probabilities"],
            "predictions": control["protected_predictions"],
            "label_counts": control["protected_label_counts"],
            "correct_by_class": control["protected_correct_by_class"],
            "per_class_recall": control["protected_per_class_recall"],
        },
        "clean_calibration": {
            "indices": clean_calibration["indices"],
            "indices_sha256": tensor_sha256(clean_calibration["indices"]),
            "labels": clean_calibration["labels"],
            "features_l2": clean_calibration["features_l2"],
            "features_raw": clean_calibration["features_raw"],
            "logits": clean_calibration["logits"],
            "probabilities": clean_calibration["probabilities"],
            "predictions": clean_calibration["predictions"],
            "per_class_recall": clean_calibration["per_class_recall"],
        },
        "path_proof": control["path_proof"],
    }


def _build_generated_payload(
    z_generated: torch.Tensor,
    raw_generated: torch.Tensor,
    bank_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(tuple(z_generated.shape) == (GENERATION_TOTAL_COUNT, OLS_PORT_COUNT), "generated OLS bank shape changed")
    _require(tuple(raw_generated.shape) == (GENERATION_TOTAL_COUNT,) + RAW_OUTPUT_SHAPE, "generated raw bank shape changed")
    return {
        "schema": GENERATED_ARTIFACT_SCHEMA,
        "version": 1,
        "z_generated": z_generated,
        "raw_cmos": raw_generated,
        "bank_records": [dict(item) for item in bank_records],
        "concatenation": "ascending bank index; all samples retained; no sorting or selection",
        "path_proof": {
            "encoder_used": False,
            "labels_used_as_generator_input": False,
            "q_used": False,
            "phase_repair_used": False,
            "latent_optimization_used": False,
            "rejection_or_retry_used": False,
            "shared_w_ols": True,
            "shared_generator": True,
        },
    }


def _build_statistics_payload(
    control: Mapping[str, Any],
    latent: Mapping[str, Any],
    latent_data: Mapping[str, Any],
    semantic: Mapping[str, Any],
    semantic_data: Mapping[str, Any],
    diversity: Mapping[str, Any],
    diversity_data: Mapping[str, Any],
    novelty: Mapping[str, Any],
    novelty_data: Mapping[str, Any],
    generated: Mapping[str, Any],
    training_roles: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    return {
        "schema": STATISTICS_ARTIFACT_SCHEMA,
        "version": 1,
        "latent_compatibility": {
            "z_eval_panel": control["panel_z_ols"],
            "z_prior_panel": latent_data["z_prior"],
            "null_statistics": latent_data["null_values"],
            "seed_records": latent_data["seed_records"],
            "seed_records_sha256": latent_data["seed_records_sha256"],
            "decision": latent["decision"],
        },
        "semantic_distribution": {
            "protected_panel_features_l2": control["panel_features_l2"],
            "generated_features_l2": generated["features_l2"],
            "calibration_features_l2": training_roles["calibration_features_l2"],
            "protected_panel_logits": control["panel_logits"],
            "generated_logits": generated["logits"],
            "null_statistics": semantic_data["null_values"],
            "seed_records": semantic_data["seed_records"],
            "seed_records_sha256": semantic_data["seed_records_sha256"],
            "decision": semantic["decision"],
        },
        "diversity": {
            "null_statistics": diversity_data["null_values"],
            "seed_records": diversity_data["seed_records"],
            "seed_records_sha256": diversity_data["seed_records_sha256"],
            "decision": diversity["decision"],
        },
        "novelty": {
            "train_reference_features_l2": training_roles["train_features_l2"],
            "train_reference_indices": training_roles["train_indices"],
            "protected_nearest_distances": novelty_data["protected_nearest_distances"],
            "protected_nearest_train_positions": novelty_data["protected_nearest_train_positions"],
            "protected_nearest_train_indices": novelty_data["protected_nearest_train_indices"],
            "generated_nearest_distances": novelty_data["generated_nearest_distances"],
            "generated_nearest_train_positions": novelty_data["generated_nearest_train_positions"],
            "generated_nearest_train_indices": novelty_data["generated_nearest_train_indices"],
            "calibration_nearest_distances": novelty_data["calibration_nearest_distances"],
            "null_statistics": novelty_data["null_values"],
            "seed_records": novelty_data["seed_records"],
            "seed_records_sha256": novelty_data["seed_records_sha256"],
            "decision": novelty["decision"],
        },
    }


def _final_classification(control: Mapping[str, Any], latent: Mapping[str, Any] | None, semantic: Mapping[str, Any] | None, diversity: Mapping[str, Any] | None, novelty: Mapping[str, Any] | None) -> tuple[str, str | None, list[str]]:
    sequence = ["authority_precheck", "stage_1_real_image_positive_control"]
    if not bool(control["pass"]):
        return CONTROL_FAILURE_CLASSIFICATION, "stage_1_real_image_positive_control", sequence
    _require(latent is not None, "latent decision state is incomplete")
    sequence.append("stage_2_full_complex_latent_compatibility")
    if not bool(latent["decision"]["pass"]):
        return NOT_SUPPORTED_CLASSIFICATION, "stage_2_full_complex_latent_compatibility", sequence
    _require(semantic is not None and diversity is not None and novelty is not None, "downstream decision state is incomplete")
    sequence.extend(["stage_3_direct_target_prior_generation", "stage_4_semantic_distribution", "stage_5_diversity", "stage_6_novelty", "one_final_classification"])
    gates = [("stage_2_full_complex_latent_compatibility", latent), ("stage_4_semantic_distribution", semantic), ("stage_5_diversity", diversity), ("stage_6_novelty", novelty)]
    for name, result in gates:
        if not bool(result["decision"]["pass"]):
            return NOT_SUPPORTED_CLASSIFICATION, name, sequence
    return SUPPORTED_CLASSIFICATION, None, sequence


def _assert_output_tree(temporary: Path, *, generated: bool) -> None:
    expected = {CONTROL_ARTIFACT_NAME, LINEAGE_NAME, REPORT_NAME}
    if generated:
        expected.update({GENERATED_ARTIFACT_NAME, STATISTICS_ARTIFACT_NAME})
    actual = {path.relative_to(temporary).as_posix() for path in temporary.rglob("*") if path.is_file()}
    _require(actual == expected, f"evaluation output tree is not exact: {sorted(actual)}")


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve() if output_dir.is_absolute() else (repo_root / output_dir).resolve()
    _require(not output_dir.exists(), f"refusing to reuse write-once evaluation output: {output_dir}", blocked=True)
    preflight = _preflight(repo_root)
    git_before = preflight["git_before"]
    device = torch.device("cuda:0")
    training_contract = load_frozen_training_contract(repo_root / TRAINING_CONTRACT_PATH, repo_root=repo_root)
    checkpoint = preflight["checkpoint"]
    consumer = LightGen2DComplexOLSTrainingConsumer(
        training_contract,
        device=device,
        propagation_grid_size=PRODUCTION_PROPAGATION_GRID,
    )
    consumer.load_trainable_state_dict(_mapping(checkpoint["trainable_parameter_state"], "checkpoint trainable state"))
    consumer.runtime.eval()
    for parameter in consumer.runtime.parameters():
        parameter.requires_grad_(False)
    evaluator, evaluator_payload = _load_evaluator_authority(repo_root, device)
    training_roles = _derive_training_roles(repo_root)
    calibration = _clean_evaluator_bank(
        evaluator,
        training_roles["images"],
        training_roles["labels"],
        training_roles["calibration_indices"],
        device,
    )
    train_clean = _clean_evaluator_bank(
        evaluator,
        training_roles["images"],
        training_roles["labels"],
        training_roles["train_indices"],
        device,
    )
    training_roles = {
        **training_roles,
        "calibration_features_l2": calibration["features_l2"],
        "train_features_l2": train_clean["features_l2"],
        "train_indices": training_roles["train_indices"].cpu().contiguous(),
    }
    protected_images, protected_labels = _load_mnist_file(repo_root, PROTECTED_TEST_PATH, PROTECTED_TEST_SHA256, "protected test source")
    preflight["protected_value_load_count"] = 1
    _require(preflight["protected_sha_before_values"] == file_sha256(repo_root / PROTECTED_TEST_PATH), "protected test changed after authority precheck", blocked=True)
    panel_indices = _protected_panel_indices(protected_labels)
    clean_threshold = max(CONTROL_FLOOR, float(calibration["macro_recall"]) - CONTROL_ALLOWED_DROP)
    control_data = _stage_one_control(consumer, evaluator, protected_images, protected_labels, panel_indices, device)
    control_pass = bool(control_data["protected_macro_recall"] >= clean_threshold)
    control = {
        "status": "PASS" if control_pass else "FAIL",
        "statistic": "macro_recall over full protected test role",
        "clean_calibration_macro_recall": calibration["macro_recall"],
        "threshold": clean_threshold,
        "protected_macro_recall": control_data["protected_macro_recall"],
        "protected_per_class_recall": control_data["protected_per_class_recall"].tolist(),
        "protected_label_counts": control_data["protected_label_counts"].tolist(),
        "protected_correct_by_class": control_data["protected_correct_by_class"].tolist(),
        "pass": control_pass,
        "evaluation_order": 1,
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        protected_source_ref = _authority_ref(repo_root, PROTECTED_TEST_PATH, "protected evaluation source", PROTECTED_TEST_SHA256)
        control_path = temporary / CONTROL_ARTIFACT_NAME
        control_payload = _build_control_payload(control_data, calibration, protected_source=protected_source_ref)
        _write_and_verify_torch(control_path, control_payload, CONTROL_ARTIFACT_SCHEMA, "protected real-path control artifact")
        control_ref = _artifact_ref(repo_root, output_dir, temporary, control_path, "protected real-path control and evaluator outputs", CONTROL_ARTIFACT_SCHEMA)

        latent_result: dict[str, Any] | None = None
        latent_data: dict[str, Any] | None = None
        semantic_result: dict[str, Any] | None = None
        semantic_data: dict[str, Any] | None = None
        diversity_result: dict[str, Any] | None = None
        diversity_data: dict[str, Any] | None = None
        novelty_result: dict[str, Any] | None = None
        novelty_data: dict[str, Any] | None = None
        generated_ref: dict[str, Any] | None = None
        statistics_ref: dict[str, Any] | None = None
        bank_records: list[dict[str, Any]] = []

        if control_pass:
            latent_result, latent_data = _latent_compatibility(control_data["panel_z_ols"], device)
            if latent_result["decision"]["pass"]:
                z_generated_parts: list[torch.Tensor] = []
                raw_generated_parts: list[torch.Tensor] = []
                generated_feature_parts: list[torch.Tensor] = []
                generated_raw_feature_parts: list[torch.Tensor] = []
                generated_logit_parts: list[torch.Tensor] = []
                generated_probability_parts: list[torch.Tensor] = []
                generated_prediction_parts: list[torch.Tensor] = []
                for bank_index in range(GENERATION_BANK_COUNT):
                    role = f"{GENERATION_ROLE_PREFIX}{bank_index}"
                    z_bank_device, seed, generator_state = _sample_prior_exact(
                        GENERATION_BANK_SIZE,
                        namespace=PRIOR_NAMESPACE,
                        role=role,
                        global_step=PRIOR_GLOBAL_STEP,
                        device=device,
                    )
                    z_bank = z_bank_device.cpu().contiguous()
                    raw_bank = _run_generation_bank(consumer, z_bank, device)
                    with torch.inference_mode():
                        generated_view = _evaluator_view(raw_bank.to(device))
                        normalized, raw_features, logits, probabilities, predictions = _evaluate_views(evaluator, generated_view)
                    z_generated_parts.append(z_bank)
                    raw_generated_parts.append(raw_bank)
                    generated_feature_parts.append(normalized.cpu())
                    generated_raw_feature_parts.append(raw_features.cpu())
                    generated_logit_parts.append(logits.cpu())
                    generated_probability_parts.append(probabilities.cpu())
                    generated_prediction_parts.append(predictions.cpu())
                    bank_records.append(
                        {
                            "bank_index": bank_index,
                            "role": role,
                            "global_step": PRIOR_GLOBAL_STEP,
                            "sample_count": GENERATION_BANK_SIZE,
                            "seed": seed,
                            "generator_state_sha256": tensor_sha256(generator_state),
                        }
                    )
                    print(f"stage3 direct prior generation: bank {bank_index + 1}/{GENERATION_BANK_COUNT}", flush=True)
                generated = {
                    "z_generated": torch.cat(z_generated_parts).contiguous(),
                    "raw_cmos": torch.cat(raw_generated_parts).contiguous(),
                    "features_l2": torch.cat(generated_feature_parts).contiguous(),
                    "features_raw": torch.cat(generated_raw_feature_parts).contiguous(),
                    "logits": torch.cat(generated_logit_parts).contiguous(),
                    "probabilities": torch.cat(generated_probability_parts).contiguous(),
                    "predictions": torch.cat(generated_prediction_parts).contiguous(),
                }
                _require(generated["z_generated"].shape[0] == GENERATION_BANK_COUNT * GENERATION_BANK_SIZE, "generated concatenation count changed")
                generated_path = temporary / GENERATED_ARTIFACT_NAME
                generated_payload = _build_generated_payload(generated["z_generated"], generated["raw_cmos"], bank_records)
                _write_and_verify_torch(generated_path, generated_payload, GENERATED_ARTIFACT_SCHEMA, "generated OLS and raw CMOS bank artifact")
                generated_ref = _artifact_ref(repo_root, output_dir, temporary, generated_path, "generated OLS and raw CMOS banks", GENERATED_ARTIFACT_SCHEMA)
                semantic_result, semantic_data = _semantic_distribution(
                    control_data["panel_features_l2"],
                    generated["features_l2"],
                    training_roles["calibration_features_l2"],
                    device,
                )
                diversity_result, diversity_data = _diversity(
                    control_data["panel_features_l2"],
                    generated["features_l2"],
                    training_roles["calibration_features_l2"],
                    device,
                )
                # Novelty uses the retained generated raw-output bank for the exact
                # evaluator-view duplicate check and the frozen train-role view.
                train_views = training_roles["images"].index_select(0, training_roles["train_indices"])
                generated_views = _evaluator_view(generated["raw_cmos"])
                novelty_result, novelty_data = _novelty(
                    control_data["panel_features_l2"],
                    generated["features_l2"],
                    training_roles["calibration_features_l2"],
                    training_roles["train_features_l2"],
                    training_roles["train_indices"],
                    generated_views,
                    train_views,
                    device,
                )
                statistics_payload = _build_statistics_payload(
                    control_data,
                    latent_result,
                    latent_data,
                    semantic_result,
                    semantic_data,
                    diversity_result,
                    diversity_data,
                    novelty_result,
                    novelty_data,
                    generated,
                    training_roles,
                )
                statistics_path = temporary / STATISTICS_ARTIFACT_NAME
                _write_and_verify_torch(statistics_path, statistics_payload, STATISTICS_ARTIFACT_SCHEMA, "post-training evaluation statistics artifact")
                statistics_ref = _artifact_ref(repo_root, output_dir, temporary, statistics_path, "latent/semantic/diversity/novelty statistics", STATISTICS_ARTIFACT_SCHEMA)

        classification, first_failure, decision_sequence = _final_classification(control, latent_result, semantic_result, diversity_result, novelty_result)
        stages = {
            "stage_1_real_image_positive_control": control,
            "stage_2_full_complex_latent_compatibility": latent_result or {"status": "NOT_EXECUTED_CONTROL_FAILED"},
            "stage_3_direct_target_prior_generation": {
                "status": "COMPLETED" if generated_ref is not None else ("NOT_EXECUTED_CONTROL_FAILED" if not control_pass else "NOT_EXECUTED_LATENT_FAILED"),
                "bank_count": GENERATION_BANK_COUNT if generated_ref is not None else 0,
                "samples_per_bank": GENERATION_BANK_SIZE if generated_ref is not None else 0,
                "bank_records": bank_records,
                "encoder_used": False if generated_ref is not None else None,
            },
            "stage_4_semantic_distribution": semantic_result or ({"status": "NOT_EXECUTED_CONTROL_FAILED"} if not control_pass else {"status": "NOT_EXECUTED_LATENT_FAILED"}),
            "stage_5_diversity": diversity_result or ({"status": "NOT_EXECUTED_CONTROL_FAILED"} if not control_pass else {"status": "NOT_EXECUTED_LATENT_FAILED"}),
            "stage_6_novelty": novelty_result or ({"status": "NOT_EXECUTED_CONTROL_FAILED"} if not control_pass else {"status": "NOT_EXECUTED_LATENT_FAILED"}),
        }
        git_after = _git_state(repo_root)
        _require(git_after["branch"] == git_before["branch"] and git_after["head"] == git_before["head"], "branch or committed HEAD changed during evaluation", blocked=True)
        created_refs = {"control": control_ref}
        if generated_ref is not None:
            created_refs["generated"] = generated_ref
        if statistics_ref is not None:
            created_refs["statistics"] = statistics_ref
        lineage = {
            "schema": LINEAGE_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "gate": GATE,
            "status": "valid_completed",
            "result": "VALID_COMPLETED",
            "classification": classification,
            "created_at_utc": created_at,
            "contract": {"path": _relative(repo_root, repo_root / CONTRACT_PATH), "sha256": EXPECTED_CONTRACT_SHA256, "classification": contract_definition.READY_CLASSIFICATION},
            "checkpoint": {"path": _relative(repo_root, repo_root / CHECKPOINT_PATH), "sha256": CHECKPOINT_SHA256, "global_step": 100, "safe_load_verified": True, "other_checkpoints_scientifically_evaluated": False},
            "authorities": {
                "training_contract": {"path": _relative(repo_root, repo_root / TRAINING_CONTRACT_PATH), "sha256": TRAINING_CONTRACT_SHA256},
                "training_report": {"path": _relative(repo_root, repo_root / TRAINING_REPORT_PATH), "sha256": TRAINING_REPORT_SHA256},
                "training_lineage": {"path": _relative(repo_root, repo_root / TRAINING_LINEAGE_PATH), "sha256": TRAINING_LINEAGE_SHA256},
                "aligned_runtime": {"path": _relative(repo_root, repo_root / ALIGNMENT_RUNTIME_PATH), "sha256": ALIGNMENT_RUNTIME_SHA256},
                "aligned_contract": {"path": _relative(repo_root, repo_root / ALIGNMENT_CONTRACT_PATH), "sha256": ALIGNMENT_CONTRACT_SHA256},
                "aligned_runtime_report": {"path": _relative(repo_root, repo_root / ALIGNMENT_RUNTIME_REPORT_PATH), "sha256": ALIGNMENT_RUNTIME_REPORT_SHA256},
                "training_consumer": {"path": _relative(repo_root, repo_root / TRAINING_CONSUMER_PATH), "sha256": TRAINING_CONSUMER_SHA256},
                "semantic_evaluator": {"path": _relative(repo_root, repo_root / EVALUATOR_PATH), "sha256": EVALUATOR_SHA256, "implementation_sha256": EVALUATOR_SOURCE_SHA256, "validation_accuracy": float(evaluator_payload["validation_accuracy"])},
                "evaluator_qualification": {"path": _relative(repo_root, repo_root / EVALUATOR_QUALIFICATION_PATH), "sha256": EVALUATOR_QUALIFICATION_SHA256},
                "evaluator_split_manifest": {"path": _relative(repo_root, repo_root / EVALUATOR_SPLIT_MANIFEST_PATH), "sha256": EVALUATOR_SPLIT_MANIFEST_SHA256},
            },
            "data_firewall": {
                "training_source": {"path": _relative(repo_root, repo_root / TRAINING_SOURCE_PATH), "sha256": TRAINING_SOURCE_SHA256, "use": "train reference and calibration only"},
                "protected_source": {"path": _relative(repo_root, repo_root / PROTECTED_TEST_PATH), "sha256": PROTECTED_TEST_SHA256, "value_load_count": preflight["protected_value_load_count"], "panel_index_sha256": control_data["panel_index_sha256"]},
                "sealed_values_loaded": False,
                "sealed_values_reused": False,
                "roles_moved": False,
            },
            "prior": {
                "name": "PROJECT_COMPLEX_ISOTROPIC_GAUSSIAN_NATIVE_OLS_V1",
                "namespace": PRIOR_NAMESPACE,
                "formula": "z_j=sigma_z/sqrt(2)*(epsilon_j_real+i*epsilon_j_imag)",
                "sigma_z": PRIOR_SIGMA,
                "sample_order": "real quadratures then imaginary quadratures using CPU torch.Generator",
                "seed_derivation": "SHA256(namespace|role|global_step|sample_count), first 8 bytes masked to 63 bits",
                "generation_bank_records": bank_records,
            },
            "stages": stages,
            "decision_sequence": decision_sequence,
            "artifacts": created_refs,
            "execution": {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(0),
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "propagation_grid_size": PRODUCTION_PROPAGATION_GRID,
                "batch_size": BATCH_SIZE,
                "autocast": False,
                "optimizer_created": False,
                "backward_called": False,
                "training_steps": 0,
                "checkpoint_search": False,
                "threshold_tuning": False,
                "sample_retry_or_cherry_pick": False,
            },
            "governance": {"training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"},
            "git": {"branch": git_before["branch"], "head_before": git_before["head"], "head_after": git_after["head"], "worktree_before": git_before, "worktree_after": git_after, "commit_status": "not_committed", "output_artifacts_write_once": True},
            "report": {"path": _relative(repo_root, output_dir / REPORT_NAME), "sha256": "not_embedded_to_avoid_cycle"},
            "next_action": "STOP_AFTER_FROZEN_SCIENTIFIC_RESULT",
        }
        lineage_path = temporary / LINEAGE_NAME
        atomic_write_new_json(lineage_path, lineage, artifact_kind="post-training evaluation lineage")
        lineage_ref = _artifact_ref(repo_root, output_dir, temporary, lineage_path, "post-training evaluation lineage", LINEAGE_SCHEMA)
        report = {
            "schema": REPORT_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "gate": GATE,
            "status": "valid_completed",
            "result": "VALID_COMPLETED",
            "classification": classification,
            "created_at_utc": created_at,
            "authorization": {"explicit_user_authorization_consumed": True, "scope_expansion": False, "protected_evaluation_count": 1},
            "contract": {"path": _relative(repo_root, repo_root / CONTRACT_PATH), "sha256": EXPECTED_CONTRACT_SHA256, "result": "READY", "classification": contract_definition.READY_CLASSIFICATION},
            "checkpoint": lineage["checkpoint"],
            "control": control,
            "latent_compatibility": latent_result or {"status": "NOT_EXECUTED_CONTROL_FAILED"},
            "generation": stages["stage_3_direct_target_prior_generation"],
            "semantic_distribution": semantic_result or {"status": "NOT_EXECUTED_CONTROL_FAILED"},
            "diversity": diversity_result or {"status": "NOT_EXECUTED_CONTROL_FAILED"},
            "novelty_memorization": novelty_result or {"status": "NOT_EXECUTED_CONTROL_FAILED"},
            "aggregate": {"decision_sequence": decision_sequence, "final_classification": classification, "first_decisive_failure": first_failure},
            "provenance": {"authorities": lineage["authorities"], "data_firewall": lineage["data_firewall"], "prior": lineage["prior"], "lineage": lineage_ref},
            "artifacts": created_refs | {"lineage": lineage_ref},
            "local_state": lineage["git"],
            "boundaries": {
                "training": False,
                "optimizer_created": False,
                "backward_called": False,
                "checkpoint_search": False,
                "evaluator_modified": False,
                "prior_modified": False,
                "threshold_tuned": False,
                "sealed_values_loaded": False,
                "sealed_values_reused": False,
                "formal_promotion": False,
                "generation_readiness": "not_ready",
            },
            "claim_boundary": {
                "classification_scope": "one frozen protected 2D semantic-generation evaluation only",
                "supported_does_not_mean": ["exact Chen hardware reproduction", "paper-equivalent FID/KID", "universal image-generation readiness", "formal promotion"],
                "not_supported_does_not_mean": ["aligned architecture impossible", "all complex priors fail", "all joint training fails"],
            },
            "next_action": "STOP_AFTER_FROZEN_SCIENTIFIC_RESULT",
        }
        report_path = temporary / REPORT_NAME
        atomic_write_new_json(report_path, report, artifact_kind="post-training evaluation report")
        _assert_output_tree(temporary, generated=control_pass)
        _require(not output_dir.exists(), "evaluation output appeared during execution", blocked=True)
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    report_sha = file_sha256(output_dir / REPORT_NAME)
    lineage_sha = file_sha256(output_dir / LINEAGE_NAME)
    _require(report_sha and lineage_sha, "published evaluation identities are missing")
    return {
        "result": "VALID_COMPLETED",
        "classification": classification,
        "first_decisive_failure": first_failure,
        "output_directory": _relative(repo_root, output_dir),
        "report": _relative(repo_root, output_dir / REPORT_NAME),
        "report_sha256": report_sha,
        "lineage": _relative(repo_root, output_dir / LINEAGE_NAME),
        "lineage_sha256": lineage_sha,
        "generated_artifact": _relative(repo_root, output_dir / GENERATED_ARTIFACT_NAME) if generated_ref is not None else None,
        "statistics_artifact": _relative(repo_root, output_dir / STATISTICS_ARTIFACT_NAME) if statistics_ref is not None else None,
        "starting_head": git_before["head"],
        "ending_head": _git_state(repo_root)["head"],
        "protected_value_load_count": preflight["protected_value_load_count"],
        "training_steps": 0,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "next_action": "STOP_AFTER_FROZEN_SCIENTIFIC_RESULT",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        result = execute(args.repo_root, args.output_dir)
    except BlockedExecution as exc:
        print(json.dumps({"result": "BLOCKED", "classification": None, "reason": str(exc)}, indent=2, sort_keys=True))
        return 3
    except InvalidHarnessError as exc:
        print(json.dumps({"result": "INVALID", "classification": "INVALID_HARNESS", "reason": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
