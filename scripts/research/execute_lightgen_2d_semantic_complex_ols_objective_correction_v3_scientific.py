"""Execute the one-shot V3 Stage-A semantic-objective comparison.

This entry point consumes the already accepted V3 engineering preflight.  It
does not calibrate a new objective, select a checkpoint, enter Stage B, read a
protected/test/sealed role, or retry after a scientific execution starts.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

import ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 as v2
from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training_v2 import (
    ALLOWED_TRAINABLE_PARAMETER_NAMES,
    DeterministicTrainRoleSampler,
    LightGen2DComplexOLSTrainingConsumerV2,
    LOGICAL_BATCH_SIZE,
    file_sha256,
    git_facts,
    load_v2_training_contract,
    tensor_sha256,
)
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load
from ldm_project.v2.student.evaluator import FrozenMNISTEvaluator, SmoothMNISTEvaluator, stratified_mnist_indices


TASK_ID = "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_V3"
OUTER_GATE = "V3_SEMANTIC_OBJECTIVE_CORRECTION_SUPPORTED"
SUPPORTED_CLASSIFICATION = OUTER_GATE
NOT_SUPPORTED_CLASSIFICATION = "V3_SEMANTIC_OBJECTIVE_CORRECTION_NOT_SUPPORTED"
REPORT_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_lineage.v1"
INPUT_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_input.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_checkpoint.v1"
PANEL_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_raw_cmos_panel.v1"
BOOTSTRAP_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_bootstrap.v1"

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
PREFLIGHT_DIR = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_09"
)
PREFLIGHT_INPUT_PATH = PREFLIGHT_DIR / "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_input.v1.json"
PREFLIGHT_INPUT_SHA256 = "631eb85792062f00e2e59344d091a6aac203225e05c56e1a4c27150859772d36"
PREFLIGHT_REPORT_PATH = PREFLIGHT_DIR / "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_report.v1.json"
PREFLIGHT_REPORT_SHA256 = "265e9b80582161872a207041cd5edc2279b6f72c5729aa6f5406f9646e354468"
PREFLIGHT_LINEAGE_PATH = PREFLIGHT_DIR / "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_lineage.v1.json"
PREFLIGHT_LINEAGE_SHA256 = "f600bf924c3494f0b348fccde19aa6d49d5b33923cff622d5a5722394cb08e31"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01"
)
INPUT_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_input.v1.json"
REPORT_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_report.v1.json"
LINEAGE_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_lineage.v1.json"
BOOTSTRAP_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_v3_paired_bootstrap.v1.pt"

PRODUCTION_PROPAGATION_GRID = 2048
STAGE_A_STEPS = 300
CHECKPOINT_STEPS = (100, 200, 300)
INITIALIZATION_SEED = 1234
SCHEDULE_SEED = 1234
DEVELOPMENT_SELECTION_SEED = 4321
BOOTSTRAP_SEED = 271828
BOOTSTRAP_REPLICATES = 999
SEMANTIC_FEATURE_DIMENSION = 128
STAGE_A_FIREWALL = 0.937
SMOOTH_MARGIN = 0.05
SMOOTH_PER_CLASS_DROP = 0.10
IMAGE_LOSS_TOLERANCE = 0.005
LAMBDA_FEATURE = 0.1
LAMBDA_G_IMAGE = 0.0012350937723848974
LAMBDA_G_FEATURE = 0.0631487965185555
LAMBDA_RAW_RATIO = 0.01955846889373387
TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bb33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_VRAM_BYTES = 6_441_926_656
MIN_FREE_BYTES_FOR_OUTPUT = 1_500_000_000


class InvalidScientific(RuntimeError):
    """The frozen scientific harness or authority is invalid."""


class ScientificBlocked(RuntimeError):
    """A valid external boundary prevents this one-shot execution."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidScientific(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _path(root: Path, relative: str | Path) -> Path:
    value = Path(relative)
    _require(not value.is_absolute(), f"absolute authority path is not allowed: {value}")
    return (root / value).resolve()


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidScientific(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


def _validate_hash(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    actual = file_sha256(path)
    _require(actual == expected, f"{name} SHA mismatch: expected {expected}, observed {actual}")


def _load_preflight_module(root: Path) -> Any:
    path = _path(root, PREFLIGHT_PATH)
    _validate_hash(path, PREFLIGHT_SHA256, "accepted V3 preflight entrypoint")
    spec = importlib.util.spec_from_file_location("accepted_v3_preflight", path)
    _require(spec is not None and spec.loader is not None, "accepted V3 preflight module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _norm(value: torch.Tensor) -> float:
    _finite(value, "gradient")
    return float(value.detach().to(torch.float64).square().sum().sqrt().cpu())


def _resolve_output_dir(root: Path, value: Path | None) -> Path:
    target = DEFAULT_OUTPUT_DIR if value is None else value
    target = target if target.is_absolute() else root / target
    target = target.resolve()
    outputs_root = (root / "outputs").resolve()
    _require(target.is_relative_to(outputs_root), "scientific output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite existing scientific output: {target}")
    return target


def _validate_preflight_authority(root: Path, preflight: Any) -> tuple[dict[str, Any], v2.FrozenV2TrainingContract, dict[str, Any]]:
    contract, v2_contract, authorities = preflight._validate_v3_authority(root)
    report_path = _path(root, PREFLIGHT_REPORT_PATH)
    lineage_path = _path(root, PREFLIGHT_LINEAGE_PATH)
    input_path = _path(root, PREFLIGHT_INPUT_PATH)
    _validate_hash(input_path, PREFLIGHT_INPUT_SHA256, "accepted V3 preflight input")
    _validate_hash(report_path, PREFLIGHT_REPORT_SHA256, "accepted V3 preflight report")
    _validate_hash(lineage_path, PREFLIGHT_LINEAGE_SHA256, "accepted V3 preflight lineage")
    report = _read_json(report_path, "accepted V3 preflight report")
    lineage = _read_json(lineage_path, "accepted V3 preflight lineage")
    _require(report.get("result") == "AUTHORIZABLE" and report.get("status") == "valid_completed", "accepted V3 preflight is not AUTHORIZABLE")
    _require(report.get("classification") == "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_V3_AUTHORIZABLE", "accepted V3 preflight classification changed")
    _require(lineage.get("result") == "AUTHORIZABLE" and lineage.get("status") == "valid_completed", "accepted V3 preflight lineage changed")
    boundary = _mapping(report.get("claim_boundary"), "accepted preflight claim boundary")
    for key in ("scientific_training_executed", "scientific_comparison_executed", "stage_b_executed", "scale_extraction_executed", "prior_generation_executed", "protected_accessed", "test_accessed", "sealed_accessed"):
        _require(boundary.get(key) is False, f"accepted preflight claim boundary is open: {key}")
    _require(report.get("preflight_input") == {"path": PREFLIGHT_INPUT_PATH.as_posix(), "sha256": PREFLIGHT_INPUT_SHA256}, "accepted preflight input identity changed")
    entrypoint = _mapping(report.get("entrypoint"), "accepted preflight entrypoint")
    _require(entrypoint.get("path") == PREFLIGHT_PATH.as_posix() and entrypoint.get("sha256") == PREFLIGHT_SHA256, "accepted preflight entrypoint identity changed")
    calibration = _mapping(report.get("lambda_calibration"), "accepted lambda calibration")
    _require(calibration.get("lambda_feature") == LAMBDA_FEATURE, "accepted lambda changed")
    _require(math.isclose(float(calibration.get("g_image")), LAMBDA_G_IMAGE, rel_tol=0.0, abs_tol=1.0e-15), "accepted g_image changed")
    _require(math.isclose(float(calibration.get("g_feature")), LAMBDA_G_FEATURE, rel_tol=0.0, abs_tol=1.0e-15), "accepted g_feature changed")
    _require(math.isclose(float(calibration.get("raw_ratio")), LAMBDA_RAW_RATIO, rel_tol=0.0, abs_tol=1.0e-15), "accepted lambda ratio changed")
    _require(report.get("training_authorized") is False or boundary.get("training_authorized") is False, "accepted preflight opened standing training authorization")
    authorities = dict(authorities)
    authorities["accepted_preflight"] = {
        "input": {"path": PREFLIGHT_INPUT_PATH.as_posix(), "sha256": PREFLIGHT_INPUT_SHA256},
        "report": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256},
        "lineage": {"path": PREFLIGHT_LINEAGE_PATH.as_posix(), "sha256": PREFLIGHT_LINEAGE_SHA256},
        "entrypoint": {"path": PREFLIGHT_PATH.as_posix(), "sha256": PREFLIGHT_SHA256},
        "result": report.get("result"),
        "classification": report.get("classification"),
        "lambda_calibration": {"g_image": LAMBDA_G_IMAGE, "g_feature": LAMBDA_G_FEATURE, "raw_ratio": LAMBDA_RAW_RATIO, "lambda_feature": LAMBDA_FEATURE},
        "recovery_candidate_optimizer_equal": _mapping(_mapping(report.get("arms"), "accepted preflight arms").get("candidate"), "accepted candidate arm").get("checkpoint_resume", {}).get("optimizer_equal"),
    }
    return contract, v2_contract, authorities


def _load_roles(root: Path, preflight: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    images, labels, train_indices, source_metadata = preflight._load_train_role(root)
    expected_train, development_indices = stratified_mnist_indices(labels, seed=INITIALIZATION_SEED, validation_per_label=500)
    _require(torch.equal(train_indices, expected_train), "TRAIN split differs from the frozen stratified split")
    development_labels = labels.index_select(0, development_indices)
    selection: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu").manual_seed(DEVELOPMENT_SELECTION_SEED)
    for digit in range(10):
        matches = development_indices[development_labels == digit]
        _require(matches.numel() == 500, "DEV role is not balanced at 500 rows per class")
        selection.append(matches.index_select(0, torch.randperm(matches.numel(), generator=generator)[:100]))
    selection_indices = torch.cat(selection).contiguous()
    _require(tensor_sha256(selection_indices) == "4acbae34a68778753d770f2cac955a8e109e438fb4df9f41e83b15df46890bb8", "fixed DEV selection identity changed")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "TRAIN/DEV roles overlap or omit rows")
    _require(torch.equal(torch.bincount(labels.index_select(0, development_indices), minlength=10), torch.full((10,), 500, dtype=torch.long)), "DEV role counts changed")
    metadata = {
        "source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000, "values_read_from_this_train_role_source": True},
        "train": {"rows": 55_000, "counts_by_class": source_metadata["train"]["counts_by_class"], "indices_sha256": source_metadata["train"]["indices_sha256"]},
        "development": {"rows": 5_000, "selection_rows": 1_000, "selection_per_class": 100, "selection_seed": DEVELOPMENT_SELECTION_SEED, "indices_sha256": tensor_sha256(development_indices), "selection_indices_sha256": tensor_sha256(selection_indices)},
        "calibration": {"rows_read": 0, "status": "forbidden in V3"},
        "protected": {"rows_read": 0, "status": "forbidden; path not opened"},
        "test": {"rows_read": 0, "status": "forbidden; path not opened"},
        "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
        "labels_as_model_input": False,
    }
    return images, labels, train_indices, development_indices, selection_indices, metadata


def _materialize_schedule(train_indices: torch.Tensor, labels: torch.Tensor, preflight: Any) -> tuple[dict[str, Any], list[tuple[torch.Tensor, torch.Tensor]], str]:
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=STAGE_A_STEPS)
    initial = _cpu_tree(sampler.state_dict())
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for step in range(1, STAGE_A_STEPS + 1):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(batch_labels, torch.arange(10, dtype=torch.long)), f"schedule labels changed at step {step}")
        batches.append((indices.cpu().clone(), batch_labels.cpu().clone()))
    schedule_identity = {"initial": initial, "batch_indices": [item[0] for item in batches], "batch_labels": [item[1] for item in batches]}
    return initial, batches, preflight._digest_tree(schedule_identity)


def _load_evaluators(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[FrozenMNISTEvaluator, SmoothMNISTEvaluator]:
    frozen_ref = _mapping(_mapping(authorities["frozen_evaluator"], "frozen evaluator authority").get("checkpoint"), "frozen evaluator checkpoint")
    frozen_payload = _mapping(safe_torch_load(_path(root, frozen_ref["path"]), expected_sha256=frozen_ref["sha256"], map_location="cpu"), "frozen evaluator checkpoint payload")
    _require(frozen_payload.get("architecture_kind") == "frozen_mnist_evaluator_v1", "frozen evaluator architecture changed")
    _require(float(frozen_payload.get("validation_accuracy", 0.0)) >= 0.985, "frozen evaluator qualification changed")
    frozen = FrozenMNISTEvaluator()
    frozen.load_state_dict(_mapping(frozen_payload.get("state_dict"), "frozen evaluator state"), strict=True)
    frozen.to(device).freeze()

    smooth_ref = _mapping(_mapping(authorities["smooth_evaluator"], "smooth evaluator authority").get("checkpoint"), "smooth evaluator checkpoint")
    smooth_payload = _mapping(safe_torch_load(_path(root, smooth_ref["path"]), expected_sha256=smooth_ref["sha256"], map_location="cpu"), "smooth evaluator checkpoint payload")
    _require(smooth_payload.get("architecture_kind") == "smooth_mnist_evaluator_v2", "smooth evaluator architecture changed")
    smooth = SmoothMNISTEvaluator()
    smooth.load_state_dict(_mapping(smooth_payload.get("state_dict"), "smooth evaluator state"), strict=True)
    smooth.to(device).freeze()
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in frozen.parameters()), "frozen evaluator firewall failed")
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in smooth.parameters()), "smooth evaluator firewall failed")
    return frozen, smooth


def _make_consumer(contract: v2.FrozenV2TrainingContract, device: torch.device, preflight: Any) -> LightGen2DComplexOLSTrainingConsumerV2:
    return preflight._new_consumer(contract, device)


def _runtime_state(consumer: LightGen2DComplexOLSTrainingConsumerV2, preflight: Any) -> dict[str, Any]:
    return preflight._runtime_state(consumer)


def _load_runtime_state(consumer: LightGen2DComplexOLSTrainingConsumerV2, state: Mapping[str, Any], preflight: Any) -> None:
    preflight._load_runtime_state(consumer, state)


def _capture_rng(preflight: Any) -> dict[str, Any]:
    return preflight._capture_rng_state()


def _restore_rng(value: Mapping[str, Any], preflight: Any) -> None:
    preflight._restore_rng_state(value)


def _prepare_view(images: torch.Tensor, preflight: Any) -> torch.Tensor:
    return preflight._prepare_evaluator_view(images)


def _feature_loss(provider: nn.Module, raw_cmos: torch.Tensor, target_images: torch.Tensor, preflight: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
    return preflight._feature_loss(provider, raw_cmos, target_images, detached_provider=False)


def _check_eval_forward_path(consumer: LightGen2DComplexOLSTrainingConsumerV2, result: Any, batch_size: int, preflight: Any) -> None:
    _require(result.image_input is not None and result.ols_input_field is not None, "evaluation image path did not expose image/OLS boundaries")
    _require(result.z_ols.source == "real_encoded" and tuple(result.z_ols.coefficients.shape) == (batch_size, 100), "evaluation OLS state changed")
    _require(result.z_ols.coefficients.dtype == torch.complex64, "evaluation z dtype changed")
    _require(result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "evaluation modal projection boundary is inactive")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "evaluation W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "evaluation T=I_100 transport changed")
    _require(torch.equal(consumer.runtime.ols.transport.detach().cpu(), torch.eye(100, dtype=torch.complex64)), "evaluation OLS transport is not T=I_100")
    _require(tuple(result.raw_intensity.shape) == (batch_size, 1, 200, 200) and result.raw_intensity.dtype == torch.float32, "evaluation raw CMOS shape/dtype changed")
    _finite(result.raw_intensity, "evaluation raw CMOS")
    _finite(result.image_input.slm_intensity, "evaluation target")


def _gradient_norms(gradients: Mapping[str, torch.Tensor | None], preflight: Any) -> dict[str, float]:
    result: dict[str, float] = {}
    for name, gradient in gradients.items():
        result[name] = 0.0 if gradient is None else preflight._norm(gradient)
    return result


def _run_scientific_step(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    provider: FrozenMNISTEvaluator,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    arm: str,
    step: int,
    preflight: Any,
) -> dict[str, Any]:
    _require(arm in {"control", "candidate"}, "scientific arm identity is invalid")
    consumer._enforce_parameter_firewall()
    named = dict(consumer.runtime.named_parameters())
    _require(list(named) == list(ALLOWED_TRAINABLE_PARAMETER_NAMES), "scientific trainable parameter surface changed")
    parameters = [named[name] for name in ALLOWED_TRAINABLE_PARAMETER_NAMES]
    before_parameters = {name: parameter.detach().clone() for name, parameter in named.items()}
    before_frozen = consumer._frozen_state_snapshot()
    before_provider = _cpu_tree(provider.state_dict())
    optimizer.zero_grad(set_to_none=True)
    result = consumer.runtime.forward_from_image(images)
    preflight._check_forward_path(consumer, result)
    raw_cmos = result.raw_intensity
    image_loss = (raw_cmos - result.image_input.slm_intensity).square().mean()
    _finite(image_loss, "scientific image loss")
    if arm == "control":
        with torch.no_grad():
            feature_loss, semantic_input_gradient = preflight._feature_loss(provider, raw_cmos, images, detached_provider=True)
    else:
        feature_loss, semantic_input_gradient = _feature_loss(provider, raw_cmos, images, preflight)
    total_loss = image_loss if arm == "control" else image_loss + LAMBDA_FEATURE * feature_loss
    _finite(total_loss, "scientific total loss")
    boundary_tensors = {
        "raw_cmos": raw_cmos,
        "ols_input_field": result.ols_input_field.tensor,
        "z_ols": result.z_ols.coefficients,
        "generator_input_field": result.generator_input_field.tensor,
    }
    for tensor in boundary_tensors.values():
        _require(tensor.requires_grad, "scientific autograd boundary tensor is detached")
        tensor.retain_grad()
    image_gradients = dict(zip(ALLOWED_TRAINABLE_PARAMETER_NAMES, torch.autograd.grad(image_loss, parameters, retain_graph=True, allow_unused=False), strict=True))
    if feature_loss.requires_grad:
        feature_gradients = dict(zip(ALLOWED_TRAINABLE_PARAMETER_NAMES, torch.autograd.grad(feature_loss, parameters, retain_graph=True, allow_unused=True), strict=True))
    else:
        feature_gradients = dict.fromkeys(ALLOWED_TRAINABLE_PARAMETER_NAMES)
    total_loss.backward()
    total_gradients = {name: parameter.grad.detach().clone() if parameter.grad is not None else None for name, parameter in named.items()}
    _require(all(parameter.grad is not None for parameter in parameters), "scientific trainable gradient is missing")
    _require(all(parameter.grad is None for name, parameter in named.items() if name not in ALLOWED_TRAINABLE_PARAMETER_NAMES), "forbidden runtime parameter received a gradient")
    if arm == "candidate":
        _require(semantic_input_gradient is not None and _norm(semantic_input_gradient) > 0.0, "candidate semantic input gradient is absent or zero")
        _finite(semantic_input_gradient, "candidate semantic input gradient")
    else:
        _require(semantic_input_gradient is None, "control unexpectedly retained semantic input gradient")
    optimizer.step()
    torch.cuda.synchronize(consumer.device)
    after_parameters = {name: parameter.detach().clone() for name, parameter in named.items()}
    for name, parameter in named.items():
        _finite(parameter, f"updated parameter {name}")
    update_norms = {name: _norm(after_parameters[name] - before_parameters[name]) for name in named}
    _require(all(update_norms[name] > 0.0 for name in ALLOWED_TRAINABLE_PARAMETER_NAMES), "scientific optimizer did not update both trainable groups")
    frozen_unchanged = _tree_equal(before_frozen, consumer._frozen_state_snapshot())
    _require(frozen_unchanged, "scientific frozen runtime state changed")
    provider_unchanged = _tree_equal(before_provider, _cpu_tree(provider.state_dict()))
    _require(provider_unchanged and all(parameter.grad is None for parameter in provider.parameters()), "scientific provider firewall changed")
    zero_count = 0
    gradient_count = 0
    nonfinite_count = 0
    for gradient in total_gradients.values():
        if gradient is None:
            continue
        gradient_count += gradient.numel()
        zero_count += int((gradient == 0).sum().item())
        nonfinite_count += int((~torch.isfinite(gradient)).sum().item())
    boundary_norms = {name: _norm(tensor.grad) if tensor.grad is not None else 0.0 for name, tensor in boundary_tensors.items()}
    trace = {
        "step": step,
        "arm": arm,
        "objective": "PURE_IMAGE_CONTROL" if arm == "control" else "FROZEN_FEATURE_CORRECTED",
        "lambda_feature": 0.0 if arm == "control" else LAMBDA_FEATURE,
        "losses": {"L_image": float(image_loss.detach().cpu()), "L_feature": float(feature_loss.detach().cpu()), "L_total": float(total_loss.detach().cpu())},
        "gradient_norms": {"image": _gradient_norms(image_gradients, preflight), "feature": _gradient_norms(feature_gradients, preflight), "total": _gradient_norms(total_gradients, preflight), "semantic_input": None if semantic_input_gradient is None else _norm(semantic_input_gradient)},
        "boundary_gradient_norms": boundary_norms,
        "parameter_update_norms": update_norms,
        "zero_gradient_fraction": zero_count / gradient_count if gradient_count else 1.0,
        "nonfinite_gradient_fraction": nonfinite_count / gradient_count if gradient_count else 0.0,
        "frozen_runtime_unchanged": frozen_unchanged,
        "provider_unchanged_and_frozen": provider_unchanged and all(parameter.grad is None for parameter in provider.parameters()),
        "modal_projection_active": True,
        "w_ols_active": True,
        "ols_transport": "T=I_100_coefficient_space",
        "ols_provenance": {"modal_projection_boundary": True, "transport": "T=I_100_coefficient_space", "shared_w_ols_writer": True},
    }
    return {"trace": trace}


@torch.no_grad()
def _evaluate_checkpoint(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    provider: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
    selection_images: torch.Tensor,
    selection_labels: torch.Tensor,
    *,
    arm: str,
    step: int,
    preflight: Any,
) -> dict[str, Any]:
    was_training = consumer.runtime.training
    consumer.runtime.eval()
    raw_parts: list[torch.Tensor] = []
    image_sum = 0.0
    feature_sum = 0.0
    frozen_correct: list[torch.Tensor] = []
    smooth_correct: list[torch.Tensor] = []
    try:
        for start in range(0, selection_images.shape[0], LOGICAL_BATCH_SIZE):
            batch = selection_images[start : start + LOGICAL_BATCH_SIZE].to(consumer.device)
            labels = selection_labels[start : start + LOGICAL_BATCH_SIZE]
            result = consumer.runtime.forward_from_image(batch)
            _check_eval_forward_path(consumer, result, int(batch.shape[0]), preflight)
            raw = result.raw_intensity
            target = result.image_input.slm_intensity
            image_loss = (raw - target).square().mean()
            candidate_embedding, frozen_logits = provider(_prepare_view(raw, preflight))
            target_embedding, _ = provider(_prepare_view(batch, preflight))
            candidate_feature = F.normalize(candidate_embedding, p=2.0, dim=1, eps=1.0e-12)
            target_feature = F.normalize(target_embedding, p=2.0, dim=1, eps=1.0e-12)
            feature_loss = (candidate_feature - target_feature).square().sum(dim=1).mean()
            _finite(image_loss, "evaluation image loss")
            _finite(feature_loss, "evaluation feature loss")
            _, smooth_logits = smooth(_prepare_view(raw, preflight))
            frozen_correct.append((frozen_logits.argmax(dim=1).cpu() == labels).contiguous())
            smooth_correct.append((smooth_logits.argmax(dim=1).cpu() == labels).contiguous())
            raw_parts.append(raw.detach().cpu().contiguous())
            count = int(batch.shape[0])
            image_sum += float(image_loss.cpu()) * count
            feature_sum += float(feature_loss.cpu()) * count
            del batch, result, raw, candidate_embedding, frozen_logits, target_embedding, smooth_logits
    finally:
        consumer.runtime.train(was_training)
    raw_cpu = torch.cat(raw_parts, dim=0)
    frozen_correct_cpu = torch.cat(frozen_correct, dim=0)
    smooth_correct_cpu = torch.cat(smooth_correct, dim=0)
    _require(tuple(raw_cpu.shape) == (1000, 1, 200, 200) and raw_cpu.dtype == torch.float32, "fixed raw-CMOS panel shape/dtype changed")
    finite = bool(torch.isfinite(raw_cpu).all())
    flat = raw_cpu.flatten(1)
    per_image_variance = flat.var(dim=1, unbiased=False)
    across_sample_variance = flat.var(dim=0, unbiased=False)
    per_image_variance_mean = float(per_image_variance.mean())
    across_sample_variance_mean = float(across_sample_variance.mean())
    structural_valid = finite and tuple(raw_cpu.shape) == (1000, 1, 200, 200) and raw_cpu.dtype == torch.float32
    nonconstant = per_image_variance_mean > 0.0 and across_sample_variance_mean > 0.0
    _require(structural_valid and nonconstant, f"raw-CMOS panel failed finite/structural/nonconstant check at step {step} ({arm})")

    def recalls(correct: torch.Tensor) -> tuple[list[float], float]:
        values = []
        for digit in range(10):
            mask = selection_labels == digit
            _require(int(mask.sum()) == 100, f"fixed DEV selection class count changed for {digit}")
            values.append(float(correct[mask].to(torch.float32).mean()))
        return values, float(sum(values) / 10.0)

    frozen_per_class, frozen_macro = recalls(frozen_correct_cpu)
    smooth_per_class, smooth_macro = recalls(smooth_correct_cpu)
    return {
        "step": step,
        "arm": arm,
        "losses": {"L_image": image_sum / 1000.0, "L_feature": feature_sum / 1000.0, "L_total": image_sum / 1000.0 if arm == "control" else image_sum / 1000.0 + LAMBDA_FEATURE * feature_sum / 1000.0},
        "frozen_evaluator": {"macro_recall": frozen_macro, "per_class_recall": frozen_per_class},
        "smooth_evaluator": {"macro_recall": smooth_macro, "per_class_recall": smooth_per_class},
        "raw_panel": {"shape": list(raw_cpu.shape), "dtype": str(raw_cpu.dtype), "finite": finite, "structural_valid": structural_valid, "nonconstant": nonconstant, "mean": float(raw_cpu.mean()), "std": float(raw_cpu.std(unbiased=False)), "per_image_variance_mean": per_image_variance_mean, "across_sample_variance_mean": across_sample_variance_mean, "tensor_sha256": tensor_sha256(raw_cpu)},
        "_raw_cpu": raw_cpu,
        "_frozen_correct": frozen_correct_cpu,
        "_smooth_correct": smooth_correct_cpu,
    }


def _save_panel(root: Path, target: Path, record: Mapping[str, Any], selection_indices: torch.Tensor, selection_labels: torch.Tensor, contract: v2.FrozenV2TrainingContract) -> dict[str, Any]:
    arm = str(record["arm"])
    step = int(record["step"])
    path = target / f"lightgen_2d_semantic_complex_ols_objective_correction_v3_{arm}_step_{step}_raw_cmos_panel.v1.pt"
    payload = {"schema": PANEL_SCHEMA, "version": 1, "contract": {"path": V3_CONTRACT_PATH.as_posix(), "sha256": V3_CONTRACT_SHA256}, "v2_contract": contract.identity(), "arm": arm, "checkpoint_step": step, "role": "development selection only", "selection_indices_sha256": tensor_sha256(selection_indices), "selection_labels_sha256": tensor_sha256(selection_labels), "raw_cmos": record["_raw_cpu"]}
    atomic_torch_save_new(path, payload, artifact_kind=f"V3 {arm} raw-CMOS panel")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"{arm} step-{step} raw-CMOS panel")
    loaded_tensor = loaded.get("raw_cmos")
    _require(torch.is_tensor(loaded_tensor) and tensor_sha256(loaded_tensor) == record["raw_panel"]["tensor_sha256"], f"{arm} step-{step} raw-CMOS panel reload changed")
    return {"path": _relative(root, path), "sha256": digest, "tensor_sha256": record["raw_panel"]["tensor_sha256"], "shape": list(record["_raw_cpu"].shape), "role": "development selection only"}


def _checkpoint_payload(
    root: Path,
    contract: v2.FrozenV2TrainingContract,
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    optimizer: torch.optim.Optimizer,
    sampler: DeterministicTrainRoleSampler,
    *,
    arm: str,
    step: int,
    base_state_digest: str,
    initial_rng_digest: str,
    schedule_digest: str,
    preflight: Any,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": 1,
        "contract": {"path": V3_CONTRACT_PATH.as_posix(), "sha256": V3_CONTRACT_SHA256},
        "v2_contract": contract.identity(),
        "preflight": {"path": PREFLIGHT_REPORT_PATH.as_posix(), "sha256": PREFLIGHT_REPORT_SHA256},
        "arm": arm,
        "objective": "PURE_IMAGE_CONTROL" if arm == "control" else "FROZEN_FEATURE_CORRECTED",
        "lambda_feature": 0.0 if arm == "control" else LAMBDA_FEATURE,
        "checkpoint_step": step,
        "primary_endpoint": step == 300,
        "base_initial_state_digest": base_state_digest,
        "initial_rng_digest": initial_rng_digest,
        "schedule_digest": schedule_digest,
        "runtime_state": _runtime_state(consumer, preflight),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "optimizer_manifest": consumer.optimizer_manifest(optimizer),
        "sampler_state": _cpu_tree(sampler.state_dict()),
        "rng_state": _cpu_tree(_capture_rng(preflight)),
        "trainable_parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
        "data_roles": {"train": True, "development": False, "calibration": False, "protected": False, "test": False, "sealed": False},
        "scientific_training_checkpoint": True,
        "training_authorized": False,
        "stage_b_executed": False,
        "generation_executed": False,
        "formal_promotion": False,
        "repo_root": str(root),
    }


def _save_checkpoint(root: Path, target: Path, payload: Mapping[str, Any], arm: str, step: int) -> dict[str, Any]:
    path = target / f"lightgen_2d_semantic_complex_ols_objective_correction_v3_{arm}_step_{step}_checkpoint.v1.pt"
    atomic_torch_save_new(path, payload, artifact_kind=f"V3 {arm} scientific checkpoint")
    digest = file_sha256(path)
    loaded = _mapping(safe_torch_load(path, expected_sha256=digest, map_location="cpu"), f"{arm} step-{step} checkpoint")
    _require(loaded.get("schema") == CHECKPOINT_SCHEMA and loaded.get("arm") == arm and loaded.get("checkpoint_step") == step, f"{arm} step-{step} checkpoint identity changed")
    return {"path": _relative(root, path), "sha256": digest, "step": step, "loaded": True, "scientific_training_checkpoint": True}


def _public_evaluation(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _run_arm(
    root: Path,
    target: Path,
    contract: v2.FrozenV2TrainingContract,
    provider: FrozenMNISTEvaluator,
    smooth: SmoothMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    selection_images: torch.Tensor,
    selection_labels: torch.Tensor,
    schedule: Sequence[tuple[torch.Tensor, torch.Tensor]],
    sampler_initial_state: Mapping[str, Any],
    base_state: Mapping[str, Any],
    initial_rng: Mapping[str, Any],
    base_state_digest: str,
    schedule_digest: str,
    selection_indices: torch.Tensor,
    device: torch.device,
    *,
    arm: str,
    preflight: Any,
) -> dict[str, Any]:
    _restore_rng(initial_rng, preflight)
    consumer = _make_consumer(contract, device, preflight)
    _load_runtime_state(consumer, base_state, preflight)
    _restore_rng(initial_rng, preflight)
    initial_state_digest = preflight._digest_tree(_runtime_state(consumer, preflight))
    initial_rng_digest = preflight._digest_tree(_capture_rng(preflight))
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=STAGE_A_STEPS)
    sampler.load_state_dict(sampler_initial_state)
    optimizer = consumer.create_optimizer()
    optimizer_manifest = consumer.optimizer_manifest(optimizer)
    traces: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    evaluations: dict[str, Any] = {}
    started = time.perf_counter()
    for step, expected in enumerate(schedule, start=1):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(indices, expected[0]) and torch.equal(batch_labels, expected[1]), f"{arm} schedule diverged at step {step}")
        batch_images = images.index_select(0, indices).to(device)
        try:
            result = _run_scientific_step(consumer, provider, batch_images, optimizer, arm=arm, step=step, preflight=preflight)
        except torch.cuda.OutOfMemoryError as error:
            raise ScientificBlocked("BLOCKED_V3_SCIENTIFIC_TARGET_OUT_OF_MEMORY", f"{arm}_step_{step}", "scientific target device ran out of memory", error_type=type(error).__name__, error=str(error)) from error
        traces.append(result["trace"])
        del result, batch_images
        if step in CHECKPOINT_STEPS:
            checkpoint_payload = _checkpoint_payload(root, contract, consumer, optimizer, sampler, arm=arm, step=step, base_state_digest=base_state_digest, initial_rng_digest=initial_rng_digest, schedule_digest=schedule_digest, preflight=preflight)
            checkpoints[str(step)] = _save_checkpoint(root, target, checkpoint_payload, arm, step)
            evaluation = _evaluate_checkpoint(consumer, provider, smooth, selection_images, selection_labels, arm=arm, step=step, preflight=preflight)
            evaluation["raw_panel"]["selection_indices_sha256"] = tensor_sha256(selection_indices)
            panel_ref = _save_panel(root, target, evaluation, selection_indices=selection_indices, selection_labels=selection_labels, contract=contract)
            evaluation["raw_panel"]["artifact"] = panel_ref
            evaluation["train_step_trace"] = traces[-1]
            del evaluation["_raw_cpu"]
            evaluations[str(step)] = evaluation
            del evaluation
            torch.cuda.empty_cache()
        if step % 25 == 0:
            print(json.dumps({"event": "scientific_progress", "arm": arm, "step": step, "elapsed_seconds": time.perf_counter() - started}, ensure_ascii=False), flush=True)
    _require(set(checkpoints) == {str(step) for step in CHECKPOINT_STEPS}, f"{arm} checkpoint set is incomplete")
    _require(set(evaluations) == {str(step) for step in CHECKPOINT_STEPS}, f"{arm} evaluation set is incomplete")
    return {"arm": arm, "objective": "PURE_IMAGE_CONTROL" if arm == "control" else "FROZEN_FEATURE_CORRECTED", "lambda_feature": 0.0 if arm == "control" else LAMBDA_FEATURE, "initial_state_digest": initial_state_digest, "initial_rng_digest": initial_rng_digest, "sampler_initial_state_digest": preflight._digest_tree(sampler_initial_state), "optimizer_manifest": optimizer_manifest, "traces": traces, "checkpoints": checkpoints, "evaluations": evaluations, "elapsed_seconds": time.perf_counter() - started, "updates": STAGE_A_STEPS}


def _paired_bootstrap(control: Mapping[str, Any], candidate: Mapping[str, Any], selection_labels: torch.Tensor, target: Path, root: Path, contract: v2.FrozenV2TrainingContract) -> tuple[dict[str, Any], dict[str, Any]]:
    control_correct = control["evaluations"]["300"]["_smooth_correct"]
    candidate_correct = candidate["evaluations"]["300"]["_smooth_correct"]
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    class_deltas: list[torch.Tensor] = []
    draws: list[torch.Tensor] = []
    per_class_delta: list[float] = []
    for digit in range(10):
        matches = torch.nonzero(selection_labels == digit, as_tuple=False).flatten()
        _require(matches.numel() == 100, f"bootstrap class count changed for {digit}")
        draw = torch.randint(0, matches.numel(), (BOOTSTRAP_REPLICATES, matches.numel()), generator=generator)
        draws.append(draw)
        control_values = control_correct.index_select(0, matches)[draw]
        candidate_values = candidate_correct.index_select(0, matches)[draw]
        deltas = (candidate_values.to(torch.float64) - control_values.to(torch.float64)).mean(dim=1)
        class_deltas.append(deltas)
        per_class_delta.append(float(candidate_correct.index_select(0, matches).to(torch.float64).mean() - control_correct.index_select(0, matches).to(torch.float64).mean()))
    bootstrap_delta = torch.stack(class_deltas, dim=0).mean(dim=0)
    quantiles = torch.quantile(bootstrap_delta, torch.tensor([0.025, 0.975], dtype=torch.float64), interpolation="linear")
    artifact_path = target / BOOTSTRAP_NAME
    artifact_payload = {"schema": BOOTSTRAP_SCHEMA, "version": 1, "contract": {"path": V3_CONTRACT_PATH.as_posix(), "sha256": V3_CONTRACT_SHA256}, "v2_contract": contract.identity(), "endpoint_step": 300, "selection_indices_sha256": control["evaluations"]["300"]["raw_panel"].get("selection_indices_sha256"), "bootstrap": {"type": "paired, class-stratified bootstrap over the same 100 examples per class", "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "quantile_convention": "torch.quantile interpolation=linear", "class_draws": draws, "macro_delta": bootstrap_delta}, "control_smooth_correct": control_correct, "candidate_smooth_correct": candidate_correct}
    atomic_torch_save_new(artifact_path, artifact_payload, artifact_kind="V3 paired bootstrap artifact")
    digest = file_sha256(artifact_path)
    loaded = _mapping(safe_torch_load(artifact_path, expected_sha256=digest, map_location="cpu"), "V3 paired bootstrap artifact")
    _require(torch.is_tensor(loaded.get("bootstrap", {}).get("macro_delta")) and tensor_sha256(loaded["bootstrap"]["macro_delta"]) == tensor_sha256(bootstrap_delta), "bootstrap artifact reload changed")
    summary = {"path": _relative(root, artifact_path), "sha256": digest, "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED, "delta_smooth": float(bootstrap_delta.mean()), "ci_lower": float(quantiles[0]), "ci_upper": float(quantiles[1]), "per_class_delta": per_class_delta, "macro_delta_tensor_sha256": tensor_sha256(bootstrap_delta)}
    return summary, {"bootstrap_delta": bootstrap_delta, "control_correct": control_correct, "candidate_correct": candidate_correct}


def _decision(control: Mapping[str, Any], candidate: Mapping[str, Any], bootstrap: Mapping[str, Any]) -> dict[str, Any]:
    c_eval = control["evaluations"]["300"]
    k_eval = candidate["evaluations"]["300"]
    image_delta = float(k_eval["losses"]["L_image"] - c_eval["losses"]["L_image"])
    smooth_delta = float(k_eval["smooth_evaluator"]["macro_recall"] - c_eval["smooth_evaluator"]["macro_recall"])
    per_class_delta = [float(candidate["evaluations"]["300"]["smooth_evaluator"]["per_class_recall"][digit] - control["evaluations"]["300"]["smooth_evaluator"]["per_class_recall"][digit]) for digit in range(10)]
    raw_valid = all(bool(record["evaluations"]["300"]["raw_panel"][key]) for record in (control, candidate) for key in ("finite", "structural_valid", "nonconstant"))
    checks = {"matched_300_step_traces": control["updates"] == candidate["updates"] == STAGE_A_STEPS and control["initial_state_digest"] == candidate["initial_state_digest"] and control["initial_rng_digest"] == candidate["initial_rng_digest"] and control["sampler_initial_state_digest"] == candidate["sampler_initial_state_digest"] and control["optimizer_manifest"] == candidate["optimizer_manifest"], "smooth_delta_at_least_0_05": smooth_delta >= SMOOTH_MARGIN, "paired_bootstrap_ci_lower_gt_zero": float(bootstrap["ci_lower"]) > 0.0, "no_smooth_per_class_drop_gt_0_10": min(per_class_delta) >= -SMOOTH_PER_CLASS_DROP, "candidate_image_loss_within_tolerance": image_delta <= IMAGE_LOSS_TOLERANCE, "raw_cmos_endpoint_panels_valid_and_nonconstant": raw_valid}
    return {"primary_endpoint_step": 300, "delta_smooth": smooth_delta, "image_loss_delta_candidate_minus_control": image_delta, "smooth_per_class_delta": per_class_delta, "checks": checks, "all_support_conditions": all(checks.values()), "classification": SUPPORTED_CLASSIFICATION if all(checks.values()) else NOT_SUPPORTED_CLASSIFICATION}


def _base_payload(root: Path, started: Mapping[str, Any], entrypoint_sha256: str) -> dict[str, Any]:
    return {"schema": REPORT_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "created_at_utc": datetime.now(timezone.utc).isoformat(), "result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V3_SCIENTIFIC_NOT_COMPLETED", "entrypoint": {"path": _relative(root, Path(__file__)), "sha256": entrypoint_sha256}, "provenance": {"git_start": dict(started)}, "claim_boundary": {"scientific_execution_authorization_consumed": True, "scientific_training_executed": False, "scientific_comparison_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False, "stage_b_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"}, "next": None}


def run_scientific(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    target = _resolve_output_dir(root, output_dir)
    entrypoint_path = Path(__file__).resolve()
    entrypoint_sha256 = file_sha256(entrypoint_path)
    target.mkdir(parents=True, exist_ok=False)
    started = git_facts(root)
    payload = _base_payload(root, started, entrypoint_sha256)
    authorities: dict[str, Any] | None = None
    input_ref: dict[str, Any] | None = None
    scientific_started = False
    try:
        preflight = _load_preflight_module(root)
        _require(started["branch"] == EXPECTED_BRANCH and started["head"] == EXPECTED_HEAD, f"active Git state differs from frozen V3 execution: {started['branch']}@{started['head']}")
        contract_payload, v2_contract, authorities = _validate_preflight_authority(root, preflight)
        device, device_facts = preflight._cuda_device_facts()
        properties = torch.cuda.get_device_properties(device)
        _require((int(properties.major), int(properties.minor)) == EXPECTED_COMPUTE_CAPABILITY, "target compute capability differs from accepted preflight")
        _require(int(properties.total_memory) >= EXPECTED_VRAM_BYTES, "target VRAM is below accepted preflight target")
        free_disk = shutil.disk_usage(target.parent).free
        _require(free_disk >= MIN_FREE_BYTES_FOR_OUTPUT, f"insufficient output storage headroom: {free_disk}")
        images, labels, train_indices, development_indices, selection_indices, dataset = _load_roles(root, preflight)
        selection_images = images.index_select(0, selection_indices).contiguous()
        selection_labels = labels.index_select(0, selection_indices).contiguous()
        sampler_initial_state, schedule, schedule_digest = _materialize_schedule(train_indices, labels, preflight)
        torch.manual_seed(INITIALIZATION_SEED)
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
        base_consumer = _make_consumer(v2_contract, device, preflight)
        base_state = _runtime_state(base_consumer, preflight)
        base_state_digest = preflight._digest_tree(base_state)
        _require(all(torch.count_nonzero(parameter.detach()).item() == 0 for parameter in base_consumer.runtime.parameters()), "fresh V3 runtime is not zero-phase initialized")
        initial_rng = _capture_rng(preflight)
        initial_rng_digest = preflight._digest_tree(initial_rng)
        initial_manifest = base_consumer.parameter_manifest()
        del base_consumer
        torch.cuda.empty_cache()
        frozen, smooth = _load_evaluators(root, authorities, device)
        input_payload = {"schema": INPUT_SCHEMA, "version": 1, "task_id": TASK_ID, "contract": {"path": V3_CONTRACT_PATH.as_posix(), "sha256": V3_CONTRACT_SHA256}, "preflight": authorities["accepted_preflight"], "entrypoint": payload["entrypoint"], "protocol": {"control": "L_control=L_image", "candidate": "L_candidate=L_image+lambda_feature*L_feature", "lambda_feature": LAMBDA_FEATURE, "stage_a_steps": STAGE_A_STEPS, "checkpoint_steps": list(CHECKPOINT_STEPS), "logical_batch_size": LOGICAL_BATCH_SIZE, "propagation_grid": PRODUCTION_PROPAGATION_GRID, "initialization_seed": INITIALIZATION_SEED, "schedule_seed": SCHEDULE_SEED, "development_selection_seed": DEVELOPMENT_SELECTION_SEED, "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_replicates": BOOTSTRAP_REPLICATES}, "claim_boundary": payload["claim_boundary"], "input_self_hash": "not_embedded_to_avoid_self_reference"}
        input_path = target / INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V3 scientific input")
        input_ref = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        payload.update({"contract": {"path": V3_CONTRACT_PATH.as_posix(), "sha256": V3_CONTRACT_SHA256}, "v2_contract": v2_contract.identity(), "preflight": authorities["accepted_preflight"], "authorities": authorities, "device": device_facts, "storage": {"free_bytes_before_run": int(free_disk), "minimum_required_bytes": MIN_FREE_BYTES_FOR_OUTPUT}, "dataset": dataset, "schedule": {"seed": SCHEDULE_SEED, "steps": STAGE_A_STEPS, "initial_state_sha256": preflight._digest_tree(sampler_initial_state), "schedule_digest": schedule_digest, "first_batch_indices_sha256": tensor_sha256(schedule[0][0]), "last_batch_indices_sha256": tensor_sha256(schedule[-1][0])}, "initial_state": {"runtime_state_sha256": base_state_digest, "rng_state_sha256": initial_rng_digest, "parameter_manifest": initial_manifest, "zero_phase": True}, "lambda_calibration": {"source": "accepted V3 preflight attempt-09; scientific run does not recompute lambda", "g_image": LAMBDA_G_IMAGE, "g_feature": LAMBDA_G_FEATURE, "raw_ratio": LAMBDA_RAW_RATIO, "lambda_feature": LAMBDA_FEATURE, "clipped": True}, "input": input_ref})
        scientific_started = True
        control = _run_arm(root, target, v2_contract, frozen, smooth, images, labels, train_indices, selection_images, selection_labels, schedule, sampler_initial_state, base_state, initial_rng, base_state_digest, schedule_digest, selection_indices, device, arm="control", preflight=preflight)
        candidate = _run_arm(root, target, v2_contract, frozen, smooth, images, labels, train_indices, selection_images, selection_labels, schedule, sampler_initial_state, base_state, initial_rng, base_state_digest, schedule_digest, selection_indices, device, arm="candidate", preflight=preflight)
        _require(control["initial_state_digest"] == candidate["initial_state_digest"] == base_state_digest, "control/candidate initial state mismatch")
        _require(control["initial_rng_digest"] == candidate["initial_rng_digest"] == initial_rng_digest, "control/candidate initial RNG mismatch")
        _require(control["sampler_initial_state_digest"] == candidate["sampler_initial_state_digest"] == preflight._digest_tree(sampler_initial_state), "control/candidate sampler initial state mismatch")
        _require(control["optimizer_manifest"] == candidate["optimizer_manifest"], "control/candidate optimizer manifest mismatch")
        bootstrap_summary, _ = _paired_bootstrap(control, candidate, selection_labels, target, root, v2_contract)
        decision = _decision(control, candidate, bootstrap_summary)
        checkpoint_frozen = {str(step): candidate["evaluations"][str(step)]["frozen_evaluator"]["macro_recall"] for step in CHECKPOINT_STEPS}
        firewall = {"threshold": STAGE_A_FIREWALL, "metric": "FrozenMNISTEvaluator macro recall on fixed 1000-row DEV selection", "checkpoint_macro_recall": checkpoint_frozen, "passed_at_recorded_checkpoint": any(value >= STAGE_A_FIREWALL for value in checkpoint_frozen.values()), "step_300_passed": checkpoint_frozen["300"] >= STAGE_A_FIREWALL, "stage_b_authorized": False}
        payload.update({"result": "VALID_COMPLETED", "status": "valid_completed", "classification": decision["classification"], "claim_boundary": {**payload["claim_boundary"], "scientific_training_executed": True, "scientific_comparison_executed": True}, "arms": {"control": {key: value for key, value in control.items() if key != "evaluations"} | {"evaluations": {step: _public_evaluation(record) for step, record in control["evaluations"].items()}}, "candidate": {key: value for key, value in candidate.items() if key != "evaluations"} | {"evaluations": {step: _public_evaluation(record) for step, record in candidate["evaluations"].items()}}}, "bootstrap": bootstrap_summary, "decision": decision, "stage_a_firewall": firewall, "next": "no_stage_b_authorization; retain_v3_result_as_bounded_scientific_evidence"})
    except ScientificBlocked as blocked:
        payload.update(blocked.details)
        payload.update({"result": "BLOCKED", "status": "valid_failed", "classification": blocked.classification, "hard_blocker": {"stage": blocked.stage, "message": str(blocked)}, "next": "stop; no_retry_without_new_authority"})
    except InvalidScientific as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V3_SCIENTIFIC_HARNESS", "invalid_input": {"type": type(error).__name__, "message": str(error)}, "next": "repair_harness_or_authority; do_not_interpret_scientific_quality"})
    except Exception as error:
        payload.update({"result": "INVALID", "status": "invalid_harness", "classification": "INVALID_V3_SCIENTIFIC_UNEXPECTED_EXECUTION_ERROR", "unexpected_error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-12_000:]}, "next": "stop; no_retry_without_new_authority"})
    finally:
        try:
            ending = git_facts(root)
            payload["provenance"].update({"git_end": ending, "branch_unchanged": started.get("branch") == ending.get("branch"), "head_unchanged": started.get("head") == ending.get("head"), "status_lines_unchanged": started.get("status_lines") == ending.get("status_lines")})
        except Exception as error:
            payload["provenance"]["git_end_error"] = {"type": type(error).__name__, "message": str(error)}
        payload["claim_boundary"].update({"scientific_training_executed": bool(payload["claim_boundary"].get("scientific_training_executed") or scientific_started), "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False, "stage_b_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"})
        payload.setdefault("artifacts", {})
        payload["artifacts"]["input"] = input_ref
        payload["artifacts"]["report_self_hash"] = "not_embedded_to_avoid_self_reference"

    report_path = target / REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V3 scientific report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "outer_gate": OUTER_GATE, "result": payload["result"], "status": payload["status"], "classification": payload["classification"], "contract": payload.get("contract"), "entrypoint": payload["entrypoint"], "authorities": payload.get("authorities"), "provenance": payload["provenance"], "artifacts": {**payload.get("artifacts", {}), "report": report_ref}, "claim_boundary": payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_self_reference"}
    lineage_path = target / LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V3 scientific lineage")
    result = {"result": payload["result"], "status": payload["status"], "classification": payload["classification"], "task_id": TASK_ID, "outer_gate": OUTER_GATE, "report": report_ref, "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)}, "scientific_training_executed": payload["claim_boundary"]["scientific_training_executed"], "scientific_comparison_executed": payload["claim_boundary"]["scientific_comparison_executed"], "stage_b_authorized": False, "stage_b_executed": False, "generation_readiness": "not_ready", "formal_promotion": False, "next": payload.get("next")}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("scientific",), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_scientific(repo_root=args.repo_root, output_dir=args.output_dir)
    return 0 if result["result"] == "VALID_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
