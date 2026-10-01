"""Run one bounded engineering preflight for the frozen V3 Stage-A pair.

This entry point is deliberately preflight-only.  It executes three real CUDA
updates for the pure-image control and frozen-feature candidate, plus the
minimum counterfactual and recovery checks needed to establish that a future
300-step paired experiment can be reconstructed faithfully.  It never runs a
scientific comparison, Stage B, scale extraction, prior generation, or
protected/test/sealed evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NoReturn, Sequence

import torch
from torch import nn
from torch.nn import functional as F

import ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 as v2
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import ALLOWED_TRAINABLE_PARAMETER_NAMES, DeterministicTrainRoleSampler, LightGen2DComplexOLSTrainingConsumerV2, LOGICAL_BATCH_SIZE, file_sha256, git_facts, load_v2_training_contract, tensor_sha256
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load
from ldm_project.v2.student.evaluator import (
    FrozenMNISTEvaluator,
    SmoothMNISTEvaluator,
    load_local_mnist,
    stratified_mnist_indices,
)


TASK_ID = "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_V3"
SUCCESS_CLASSIFICATION = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_V3_AUTHORIZABLE"
PREFLIGHT_INPUT_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_input.v1"
REPORT_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_lineage.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_checkpoint.v1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "383197a930dd48f98294120f8cf867444f994f85"

V3_CONTRACT_PATH = Path(
    "outputs/20260911_v2/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_contract_v3_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_contract.v3.json"
)
V3_CONTRACT_SHA256 = "e537fbf4b856519a71e48c16a48c149fadd327e2ad12db84f64c953413fb53f6"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_attempt_01"
)
INPUT_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_input.v1.json"
REPORT_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_report.v1.json"
LINEAGE_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_v3_preflight_lineage.v1.json"
CONTROL_CHECKPOINT_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_v3_control_checkpoint.v1.pt"
CANDIDATE_CHECKPOINT_NAME = "lightgen_2d_semantic_complex_ols_objective_correction_v3_candidate_checkpoint.v1.pt"

PRODUCTION_PROPAGATION_GRID = 2048
PREFLIGHT_STEPS = 3
CHECKPOINT_STEP = 2
FUTURE_STAGE_A_STEPS = 300
FUTURE_CHECKPOINT_STEPS = (100, 200, 300)
INITIALIZATION_SEED = 1234
SCHEDULE_SEED = 1234
DETERMINISTIC_RTOL = 1.0e-5
DETERMINISTIC_ATOL = 1.0e-6
LAMBDA_LOWER = 0.1
LAMBDA_UPPER = 10.0
SEMANTIC_FEATURE_DIMENSION = 128
TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"


class InvalidPreflight(RuntimeError):
    """An authority, implementation, or engineering invariant is invalid."""


class PreflightBlocked(RuntimeError):
    """A valid environmental boundary prevents this preflight from running."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidPreflight(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _path(root: Path, relative: str | Path) -> Path:
    target = Path(relative)
    _require(not target.is_absolute(), f"absolute authority path is not allowed: {target}")
    return (root / target).resolve()


def _validate_hash(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"{name} is unavailable: {path}")
    actual = file_sha256(path)
    _require(actual == expected, f"{name} SHA mismatch: expected {expected}, observed {actual}")


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidPreflight(f"{name} is not valid JSON: {path}") from error
    _require(isinstance(payload, dict), f"{name} must be a JSON object")
    return payload


def _authority_ref(root: Path, value: object, name: str) -> dict[str, Any]:
    ref = _mapping(value, name)
    path_value = ref.get("path")
    sha_value = ref.get("sha256")
    _require(isinstance(path_value, str) and isinstance(sha_value, str) and len(sha_value) == 64, f"{name} lacks a complete path/SHA binding")
    path = Path(path_value)
    _validate_hash(_path(root, path), sha_value, name)
    return {"path": path.as_posix(), "sha256": sha_value, "role": ref.get("role")}


def _git_state(root: Path) -> dict[str, Any]:
    state = git_facts(root)
    _require(state["branch"] == EXPECTED_BRANCH, f"active branch differs from frozen V3 plan: {state['branch']}")
    _require(state["head"] == EXPECTED_HEAD, f"active HEAD differs from frozen V3 plan: {state['head']}")
    return state


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


def _max_state_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    _require(left.keys() == right.keys(), "state keys differ")
    maximum = 0.0
    for key in left:
        first, second = left[key], right[key]
        if torch.is_tensor(first) and torch.is_tensor(second):
            _require(first.shape == second.shape, f"state shape differs at {key}")
            if first.numel():
                maximum = max(maximum, float((first.to(torch.float64) - second.to(torch.float64)).abs().max()))
        elif torch.is_tensor(first) or torch.is_tensor(second):
            return math.inf
        elif isinstance(first, Mapping) and isinstance(second, Mapping):
            maximum = max(maximum, _max_state_difference(first, second))
        elif isinstance(first, (list, tuple)) and isinstance(second, (list, tuple)):
            _require(len(first) == len(second), f"state sequence length differs at {key}")
            maximum = max(maximum, max((_max_state_difference({"item": a}, {"item": b}) for a, b in zip(first, second, strict=True)), default=0.0))
        elif first != second:
            return math.inf
    return maximum


def _max_numeric_difference(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0 if left is right else math.inf
    if isinstance(left, bool) or isinstance(right, bool):
        return 0.0 if left == right else math.inf
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            return math.inf
        return max((_max_numeric_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return math.inf
        return max((_max_numeric_difference(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    return 0.0 if left == right else math.inf


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


def _capture_rng_state() -> dict[str, Any]:
    return {
        "cpu": torch.random.get_rng_state().cpu().clone(),
        "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
        "cuda_available": True,
    }


def _restore_rng_state(value: Mapping[str, Any]) -> None:
    cpu = value.get("cpu")
    cuda = value.get("cuda")
    _require(torch.is_tensor(cpu) and isinstance(cuda, list), "RNG state is malformed")
    torch.random.set_rng_state(cpu.cpu())
    _require(all(torch.is_tensor(state) for state in cuda), "CUDA RNG state is malformed")
    torch.cuda.set_rng_state_all([state.cpu() for state in cuda])


def _finite_tensor(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _norm(value: torch.Tensor) -> float:
    _finite_tensor(value, "gradient")
    return float(value.detach().to(torch.float64).square().sum().sqrt().cpu())


def _gradient_summary(gradients: Mapping[str, torch.Tensor | None]) -> tuple[dict[str, float], float, float, float]:
    norms: dict[str, float] = {}
    zero = 0
    count = 0
    nonfinite = 0
    for name, gradient in gradients.items():
        if gradient is None:
            norms[name] = 0.0
            continue
        count += gradient.numel()
        nonfinite += int((~torch.isfinite(gradient)).sum().item())
        zero += int((gradient == 0).sum().item())
        _finite_tensor(gradient, f"gradient {name}")
        norms[name] = _norm(gradient)
    return norms, zero / count if count else 1.0, nonfinite / count if count else 0.0, max(norms.values(), default=0.0)


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "free_bytes": int(free),
        "total_bytes": int(total),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _nvidia_smi_facts() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise PreflightBlocked("BLOCKED_V3_PREFLIGHT_TARGET_DEVICE_UNAVAILABLE", "device", "nvidia-smi is unavailable")
    try:
        full = subprocess.run([executable], check=True, capture_output=True, text=True).stdout
        query = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightBlocked("BLOCKED_V3_PREFLIGHT_TARGET_DEVICE_UNAVAILABLE", "device", "nvidia-smi query failed", error_type=type(error).__name__, error=str(error)) from error
    fields = [field.strip() for field in query.split(",")]
    driver = re.search(r"Driver Version:\s*([0-9.]+)", full)
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", full)
    return {
        "executable": executable,
        "driver_version": driver.group(1) if driver else None,
        "driver_visible_cuda": cuda.group(1) if cuda else None,
        "query": {
            "gpu_name": fields[0] if fields else None,
            "driver_version": fields[1] if len(fields) > 1 else None,
            "memory_total_mib": int(fields[2]) if len(fields) > 2 and fields[2].isdigit() else None,
        },
    }


def _cuda_device_facts() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise PreflightBlocked(
            "BLOCKED_V3_PREFLIGHT_TARGET_DEVICE_UNAVAILABLE",
            "device",
            "required CUDA target is unavailable",
            cuda_available=bool(torch.cuda.is_available()),
            device_count=int(torch.cuda.device_count()),
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    try:
        complex_probe = torch.tensor([1.0 + 1.0j], dtype=torch.complex64, device=device)
        exp_probe = torch.exp(complex_probe)
        fft_probe = torch.fft.ifft2(torch.fft.fft2(complex_probe.reshape(1, 1, 1, 1)))
        polar_probe = torch.polar(torch.ones(1, device=device), torch.zeros(1, device=device))
        torch.cuda.synchronize(device)
    except Exception as error:
        raise PreflightBlocked(
            "BLOCKED_V3_PREFLIGHT_CUDA_BACKEND_UNAVAILABLE",
            "cuda_backend",
            "complex/FFT CUDA backend probe failed",
            error_type=type(error).__name__,
            error=str(error),
        ) from error
    backend = {
        "complex_exp": bool(torch.isfinite(exp_probe).all()),
        "fft2_ifft2": bool(torch.isfinite(fft_probe).all()),
        "complex_polar": bool(torch.isfinite(polar_probe).all()),
    }
    if not all(backend.values()):
        raise PreflightBlocked("BLOCKED_V3_PREFLIGHT_CUDA_BACKEND_UNAVAILABLE", "cuda_backend", "CUDA backend probe was nonfinite", probe=backend)
    smi = _nvidia_smi_facts()
    return device, {
        "device": str(device),
        "device_index": 0,
        "device_count": int(torch.cuda.device_count()),
        "gpu": properties.name,
        "compute_capability": [int(properties.major), int(properties.minor)],
        "vram_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_visible": smi,
        "backend": backend,
        "environment_prefix": sys.prefix,
    }


def _validate_v3_authority(root: Path) -> tuple[dict[str, Any], v2.FrozenV2TrainingContract, dict[str, Any]]:
    contract_path = _path(root, V3_CONTRACT_PATH)
    _validate_hash(contract_path, V3_CONTRACT_SHA256, "V3 objective-correction contract")
    contract = _read_json(contract_path, "V3 objective-correction contract")
    _require(contract.get("task_id") == "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_OBJECTIVE_CORRECTION_CONTRACT_V3", "V3 contract task identity changed")
    _require(contract.get("version") == 3 and contract.get("status") == "valid_completed" and contract.get("result") == "READY", "V3 contract is not READY")
    _require(contract.get("training_authorized") is False, "V3 contract authorized training")
    _require(contract.get("gate") == "V3_SEMANTIC_OBJECTIVE_CORRECTION_SUPPORTED", "V3 scientific gate identity changed")
    governance = _mapping(contract.get("governance"), "V3 governance")
    for key in ("scientific_training_executed", "stage_b_authorized", "stage_b_executed", "generation_executed", "protected_read", "test_read", "sealed_read", "formal_promotion"):
        _require(governance.get(key) is False, f"V3 governance boundary is open: {key}")
    _require(governance.get("generation_readiness") == "not_ready", "V3 generation boundary changed")

    source = _mapping(contract.get("source_authority"), "V3 source authority")
    source_refs: dict[str, Any] = {}
    for name in (
        "aligned_input_ols_contract",
        "aligned_runtime",
        "aligned_runtime_report",
        "corrected_v2_contract",
        "v2_training_report",
        "v2_training_lineage",
        "warmup_diagnosis_report",
        "warmup_diagnosis_lineage",
        "training_source",
    ):
        source_refs[name] = _authority_ref(root, source.get(name), f"V3 source authority {name}")

    alignment = _read_json(_path(root, source_refs["aligned_input_ols_contract"]["path"]), "aligned input/OLS contract")
    _require(alignment.get("result") == "READY", "aligned input/OLS contract is not READY")
    scope = _mapping(alignment.get("scope"), "alignment scope")
    _require(scope.get("task") == "2D semantic image generation only", "alignment scope changed")
    ols = _mapping(alignment.get("ols_contract"), "alignment OLS contract")
    latent = _mapping(ols.get("authoritative_latent"), "alignment latent")
    projection = _mapping(ols.get("modal_projection"), "alignment modal projection")
    reemission = _mapping(ols.get("modal_reemission"), "alignment modal re-emission")
    _require(latent.get("domain") == "C^100", "alignment latent is not C^100")
    _require(projection.get("mandatory") is True and reemission.get("mandatory") is True, "mandatory modal boundaries changed")
    _require(reemission.get("same_writer_for_real_and_generated") is True, "real/generated writer is not shared")
    runtime_text = _path(root, source_refs["aligned_runtime"]["path"]).read_text(encoding="utf-8")
    for marker in ("forward_from_image", "forward_from_ols", "modal_projection", "W_OLS"):
        _require(marker in runtime_text, f"aligned runtime marker missing: {marker}")
    runtime_report = _read_json(_path(root, source_refs["aligned_runtime_report"]["path"]), "aligned runtime report")
    _require(runtime_report.get("result") == "IMPLEMENTED", "aligned runtime report is not IMPLEMENTED")

    v2_ref = source_refs["corrected_v2_contract"]
    v2_contract = load_v2_training_contract(_path(root, v2_ref["path"]), repo_root=root)
    _require(v2_contract.sha256 == v2_ref["sha256"], "corrected V2 contract hash binding changed")
    v2_data = _mapping(v2_contract.payload.get("data_contract"), "corrected V2 data contract")
    v2_source = _mapping(_mapping(v2_data.get("source_artifacts"), "corrected V2 source artifacts").get("training"), "corrected V2 training source")
    _require(v2_source.get("path") == TRAINING_SOURCE_PATH.as_posix() and v2_source.get("sha256") == source_refs["training_source"]["sha256"], "V2 training source differs from V3 authority")
    v2_stage_a = _mapping(v2_contract.payload.get("stage_a"), "corrected V2 Stage-A")
    _require(v2_stage_a.get("steps") == FUTURE_STAGE_A_STEPS and v2_stage_a.get("checkpoint_steps") == list(FUTURE_CHECKPOINT_STEPS), "future V2 Stage-A budget changed")
    _require(_mapping(v2_stage_a.get("loss"), "corrected V2 Stage-A loss").get("formula") == "L_A = 1.0*L_image + 0.0*L_optical", "pure-image V2 baseline changed")

    v2_report = _read_json(_path(root, source_refs["v2_training_report"]["path"]), "V2 training report")
    _require(v2_report.get("result") == "VALID_COMPLETED" and v2_report.get("status") == "valid_failed", "V2 training authority result changed")
    _require(v2_report.get("classification") == "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_V2_SEMANTIC_WARMUP_NOT_ESTABLISHED", "V2 training classification changed")
    v2_lineage = _read_json(_path(root, source_refs["v2_training_lineage"]["path"]), "V2 training lineage")
    _require(v2_lineage.get("result") == "VALID_COMPLETED", "V2 training lineage result changed")
    diagnosis = _read_json(_path(root, source_refs["warmup_diagnosis_report"]["path"]), "warmup diagnosis report")
    _require(diagnosis.get("result") == "VALID_COMPLETED" and diagnosis.get("primary_diagnosis") == "SEMANTIC_WARMUP_FAILURE_IMAGE_OBJECTIVE_MISMATCH_SUPPORTED", "warmup diagnosis authority changed")
    diagnosis_boundary = _mapping(diagnosis.get("governance"), "warmup diagnosis governance")
    for key in ("new_training", "stage_b_executed", "generation_executed", "protected_read", "test_read", "sealed_read"):
        _require(diagnosis_boundary.get(key) is False, f"warmup diagnosis boundary is open: {key}")
    diagnosis_lineage = _read_json(_path(root, source_refs["warmup_diagnosis_lineage"]["path"]), "warmup diagnosis lineage")
    _require(diagnosis_lineage.get("result") == "VALID_COMPLETED", "warmup diagnosis lineage result changed")

    evaluator_source = _mapping(_mapping(source.get("frozen_evaluator"), "frozen evaluator authority").get("implementation"), "frozen evaluator implementation authority")
    evaluator_ref = _authority_ref(root, evaluator_source, "FrozenMNISTEvaluator implementation")
    evaluator_text = _path(root, evaluator_ref["path"]).read_text(encoding="utf-8")
    for marker in ("class FrozenMNISTEvaluator", "class SmoothMNISTEvaluator", "return features, self.classifier(features)"):
        _require(marker in evaluator_text, f"evaluator implementation marker missing: {marker}")
    frozen_authority = _mapping(source.get("frozen_evaluator"), "frozen evaluator authority")
    frozen_checkpoint = _authority_ref(root, frozen_authority.get("checkpoint"), "FrozenMNISTEvaluator checkpoint")
    frozen_qualification = _authority_ref(root, frozen_authority.get("qualification"), "FrozenMNISTEvaluator qualification")
    qualification_payload = _read_json(_path(root, frozen_qualification["path"]), "FrozenMNISTEvaluator qualification")
    _require(qualification_payload.get("status") == "passed" and qualification_payload.get("tier") == "acceptance", "FrozenMNISTEvaluator qualification changed")
    qualification_gates = qualification_payload.get("gates")
    _require(isinstance(qualification_gates, list) and qualification_gates, "FrozenMNISTEvaluator qualification gates are missing")
    validation_accuracy = float(_mapping(qualification_gates[0], "FrozenMNISTEvaluator accuracy gate").get("metrics", {}).get("validation_accuracy", 0.0))
    _require(validation_accuracy >= 0.985, "FrozenMNISTEvaluator qualification is below 0.985")

    smooth_authority = _mapping(source.get("smooth_evaluator"), "Smooth evaluator authority")
    smooth_checkpoint = _authority_ref(root, smooth_authority.get("checkpoint"), "SmoothMNISTEvaluator checkpoint")
    smooth_bundle = _authority_ref(root, smooth_authority.get("bundle"), "Smooth evaluator bundle")
    smooth_qualification = _authority_ref(root, smooth_authority.get("qualification"), "Smooth evaluator qualification")
    smooth_bundle_payload = _read_json(_path(root, smooth_bundle["path"]), "Smooth evaluator bundle")
    _require(smooth_bundle_payload.get("feature_dim") == SEMANTIC_FEATURE_DIMENSION, "Smooth feature dimension changed")
    _require(smooth_bundle_payload.get("primary_checkpoint") == frozen_checkpoint["sha256"] and smooth_bundle_payload.get("smooth_checkpoint") == smooth_checkpoint["sha256"], "Smooth bundle authority changed")
    smooth_qualification_payload = _read_json(_path(root, smooth_qualification["path"]), "Smooth evaluator qualification")
    _require(smooth_qualification_payload.get("status") == "passed" and smooth_qualification_payload.get("tier") == "diagnostic", "Smooth evaluator qualification changed")
    smooth_hashes = _mapping(smooth_qualification_payload.get("input_hashes"), "Smooth evaluator input hashes")
    _require(smooth_hashes.get("smooth_evaluator") == smooth_checkpoint["sha256"], "Smooth evaluator input authority changed")

    objective = _mapping(contract.get("objective"), "V3 objective")
    control = _mapping(objective.get("control"), "V3 control objective")
    candidate = _mapping(objective.get("candidate"), "V3 candidate objective")
    _require(control.get("formula") == "L_control = L_image" and control.get("semantic_term_weight") == 0.0, "V3 control objective changed")
    _require(candidate.get("formula") == "L_candidate = L_image + lambda_feature * L_feature", "V3 candidate objective changed")
    _require(candidate.get("name") == "FROZEN_FEATURE_CORRECTED", "V3 candidate identity changed")
    provider = _mapping(contract.get("semantic_provider"), "V3 semantic provider")
    _require(provider.get("identity") == "FrozenMNISTEvaluator v1" and provider.get("feature_dimension") == SEMANTIC_FEATURE_DIMENSION, "V3 provider identity changed")
    _require(provider.get("layer") == "embedding output after the evaluator's second ReLU and before classifier", "V3 provider layer changed")
    _require(provider.get("preprocessing") == "clamp image to [0,1], then adaptive_avg_pool2d to 28x28", "V3 provider preprocessing changed")
    _require(provider.get("representation") == "f = normalize_l2(embedding(preprocessed_image), dim=1, eps=1e-12) in R^128", "V3 provider representation changed")
    _require(provider.get("distance") == "per-sample squared Euclidean distance in normalized feature space", "V3 provider distance changed")
    lambda_rule = _mapping(contract.get("lambda_rule"), "V3 lambda rule")
    _require(lambda_rule.get("formula") == "lambda_feature = min(10.0, max(0.1, g_image / g_feature))", "V3 lambda formula changed")
    _require(lambda_rule.get("lower_bound") == LAMBDA_LOWER and lambda_rule.get("upper_bound") == LAMBDA_UPPER and lambda_rule.get("no_outcome_tuning") is True, "V3 lambda bounds changed")
    optimizer = _mapping(contract.get("optimizer_and_precision"), "V3 optimizer")
    _require(optimizer.get("logical_batch_size") == LOGICAL_BATCH_SIZE and optimizer.get("stage_a_steps") == FUTURE_STAGE_A_STEPS, "V3 optimizer/budget changed")
    _require(optimizer.get("initialization_seed") == INITIALIZATION_SEED and optimizer.get("schedule_seed") == SCHEDULE_SEED, "V3 seed authority changed")
    _require(optimizer.get("trainable_parameters") == list(ALLOWED_TRAINABLE_PARAMETER_NAMES), "V3 trainable surface changed")
    _require(_mapping(optimizer.get("precision"), "V3 precision").get("phase") == "float32" and _mapping(optimizer.get("precision"), "V3 precision").get("field") == "complex64", "V3 precision changed")
    _require(_mapping(optimizer.get("optimizer"), "V3 Adam").get("name") == "Adam" and _mapping(optimizer.get("optimizer"), "V3 Adam").get("learning_rate") == 1.0e-3, "V3 Adam configuration changed")
    paired = _mapping(contract.get("paired_protocol"), "V3 paired protocol")
    _require(paired.get("primary_endpoint") == FUTURE_STAGE_A_STEPS and paired.get("warm_start") == "forbidden; failed V2 checkpoints are comparison evidence only", "V3 paired endpoint or warm-start changed")
    preflight = _mapping(contract.get("preflight"), "V3 preflight contract")
    required = preflight.get("required_before_any_scientific_run")
    _require(isinstance(required, list) and any("detached-provider" in str(item) for item in required), "V3 preflight negative control is missing")

    return contract, v2_contract, {
        "v3_contract": {"path": V3_CONTRACT_PATH.as_posix(), "sha256": V3_CONTRACT_SHA256},
        "source_authority": source_refs,
        "frozen_evaluator": {
            "implementation": evaluator_ref,
            "checkpoint": frozen_checkpoint,
            "qualification": frozen_qualification,
            "validation_accuracy": validation_accuracy,
        },
        "smooth_evaluator": {
            "checkpoint": smooth_checkpoint,
            "bundle": smooth_bundle,
            "qualification": smooth_qualification,
        },
        "hash_on_every_trust_boundary": True,
    }


def _load_train_role(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = _path(root, TRAINING_SOURCE_PATH)
    _validate_hash(path, TRAINING_SOURCE_SHA256, "canonical TRAIN source")
    images, labels = load_local_mnist(root / "data", train=True)
    _require(tuple(images.shape) == (60_000, 1, 28, 28) and images.dtype == torch.float32, "TRAIN source image shape/dtype changed")
    _require(tuple(labels.shape) == (60_000,) and labels.dtype == torch.long, "TRAIN source label shape/dtype changed")
    train_indices, development_indices = stratified_mnist_indices(labels, seed=INITIALIZATION_SEED, validation_per_label=500)
    _require(train_indices.numel() == 55_000 and development_indices.numel() == 5_000, "TRAIN/DEV role counts changed")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "TRAIN/DEV partition overlaps or omits source rows")
    counts = torch.bincount(labels.index_select(0, train_indices), minlength=10)
    dataset = {
        "source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000, "values_read_from_this_train_role_source": True},
        "train": {"contract_rows": 55_000, "rows_consumed_by_three_step_schedule": 30, "counts_by_class": [int(value) for value in counts.tolist()], "indices_sha256": tensor_sha256(train_indices)},
        "development": {"contract_rows": 5_000, "values_consumed": 0, "status": "not consumed by engineering preflight"},
        "lambda_calibration": {"role": "TRAIN", "rows_consumed": LOGICAL_BATCH_SIZE},
        "calibration": {"rows_read": 0},
        "protected": {"rows_read": 0, "status": "forbidden; path not opened"},
        "test": {"rows_read": 0, "status": "forbidden; path not opened"},
        "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
        "labels_as_model_input": False,
    }
    return images, labels, train_indices, dataset


def _make_schedule(train_indices: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, Any], list[tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=FUTURE_STAGE_A_STEPS)
    initial = _cpu_tree(sampler.state_dict())
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(PREFLIGHT_STEPS):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(batch_labels, torch.arange(10, dtype=torch.long)), "V3 sampler labels changed")
        batches.append((indices.cpu().clone(), batch_labels.cpu().clone()))
    after = _cpu_tree(sampler.state_dict())
    return initial, batches, {
        "schema": initial["schema"],
        "seed": SCHEDULE_SEED,
        "future_steps": FUTURE_STAGE_A_STEPS,
        "preflight_steps": PREFLIGHT_STEPS,
        "initial_state_sha256": _digest_tree(initial),
        "after_preflight_state_sha256": _digest_tree(after),
        "batch_indices_sha256": [tensor_sha256(batch[0]) for batch in batches],
        "batch_label_sha256": [tensor_sha256(batch[1]) for batch in batches],
        "unique_rows_consumed": int(torch.unique(torch.cat([batch[0] for batch in batches])).numel()),
    }


def _new_consumer(contract: v2.FrozenV2TrainingContract, device: torch.device) -> LightGen2DComplexOLSTrainingConsumerV2:
    return LightGen2DComplexOLSTrainingConsumerV2(contract, device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID)


def _runtime_state(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> dict[str, Any]:
    return _cpu_tree(consumer.runtime.state_dict())


def _load_runtime_state(consumer: LightGen2DComplexOLSTrainingConsumerV2, state: Mapping[str, Any]) -> None:
    consumer.runtime.load_state_dict(dict(state), strict=True)
    consumer._enforce_parameter_firewall()


def _prepare_evaluator_view(images: torch.Tensor) -> torch.Tensor:
    _require(images.ndim == 4 and images.shape[1] == 1 and not torch.is_complex(images), "evaluator view input shape/type changed")
    _finite_tensor(images, "evaluator view input")
    return F.adaptive_avg_pool2d(images.clamp(0.0, 1.0), (28, 28))


def _feature_loss(provider: nn.Module, raw_cmos: torch.Tensor, target_images: torch.Tensor, *, detached_provider: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
    raw_input = raw_cmos.detach() if detached_provider else raw_cmos
    candidate_view = _prepare_evaluator_view(raw_input)
    target_view = _prepare_evaluator_view(target_images)
    candidate_embedding, _ = provider(candidate_view)
    with torch.no_grad():
        target_embedding, _ = provider(target_view)
    _require(tuple(candidate_embedding.shape) == (raw_cmos.shape[0], SEMANTIC_FEATURE_DIMENSION), "provider candidate embedding shape changed")
    _require(tuple(target_embedding.shape) == (target_images.shape[0], SEMANTIC_FEATURE_DIMENSION), "provider target embedding shape changed")
    candidate_feature = F.normalize(candidate_embedding, p=2.0, dim=1, eps=1.0e-12)
    target_feature = F.normalize(target_embedding, p=2.0, dim=1, eps=1.0e-12)
    loss = (candidate_feature - target_feature).square().sum(dim=1).mean()
    _finite_tensor(loss, "V3 feature loss")
    semantic_input = None
    if loss.requires_grad:
        semantic_input = torch.autograd.grad(loss, raw_cmos, retain_graph=True, allow_unused=True)[0]
    return loss, semantic_input


def _check_forward_path(consumer: LightGen2DComplexOLSTrainingConsumerV2, result: Any) -> None:
    _require(result.image_input is not None and result.ols_input_field is not None, "V3 real image path did not expose image/OLS boundaries")
    _require(result.z_ols.source == "real_encoded" and tuple(result.z_ols.coefficients.shape) == (LOGICAL_BATCH_SIZE, 100), "V3 real OLS state changed")
    _require(result.z_ols.coefficients.dtype == torch.complex64 and result.z_ols.coefficients.requires_grad, "V3 z state is not differentiable complex64")
    _require(result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "V3 modal projection boundary is inactive")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "V3 W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "V3 T=I_100 transport changed")
    _require(torch.equal(consumer.runtime.ols.transport.detach().cpu(), torch.eye(100, dtype=torch.complex64)), "V3 OLS transport is not T=I_100")
    _require(tuple(result.raw_intensity.shape) == (LOGICAL_BATCH_SIZE, 1, 200, 200) and result.raw_intensity.dtype == torch.float32, "V3 raw CMOS shape/dtype changed")
    _finite_tensor(result.raw_intensity, "V3 raw CMOS")
    _finite_tensor(result.image_input.slm_intensity, "V3 image target")


def _raw_summary(raw_cmos: torch.Tensor) -> dict[str, Any]:
    detached = raw_cmos.detach()
    per_image = detached.flatten(1)
    variance = per_image.var(dim=1, unbiased=False)
    return {
        "shape": list(detached.shape),
        "finite": bool(torch.isfinite(detached).all()),
        "mean": float(detached.mean().cpu()),
        "std": float(detached.std(unbiased=False).cpu()),
        "per_image_variance_mean": float(variance.mean().cpu()),
        "nonconstant": bool((variance > 0.0).any()),
    }


def _run_step(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    provider: FrozenMNISTEvaluator,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    arm: str,
    lambda_feature: float,
    step: int,
    detached_provider: bool = False,
) -> dict[str, Any]:
    _require(arm in {"control", "candidate"}, "V3 arm identity is invalid")
    consumer._enforce_parameter_firewall()
    named = dict(consumer.runtime.named_parameters())
    _require(list(named) == list(ALLOWED_TRAINABLE_PARAMETER_NAMES), "V3 trainable parameter surface changed")
    parameters = [named[name] for name in ALLOWED_TRAINABLE_PARAMETER_NAMES]
    before_parameters = {name: parameter.detach().clone() for name, parameter in named.items()}
    before_frozen = consumer._frozen_state_snapshot()
    before_provider = _cpu_tree(provider.state_dict())
    optimizer.zero_grad(set_to_none=True)
    result = consumer.runtime.forward_from_image(images)
    _check_forward_path(consumer, result)
    raw_cmos = result.raw_intensity
    target = result.image_input.slm_intensity
    image_loss = (raw_cmos - target).square().mean()
    _finite_tensor(image_loss, "V3 image loss")
    if arm == "control":
        with torch.no_grad():
            feature_loss, semantic_input_gradient = _feature_loss(provider, raw_cmos, images, detached_provider=True)
    else:
        feature_loss, semantic_input_gradient = _feature_loss(provider, raw_cmos, images, detached_provider=detached_provider)
    total_loss = image_loss if arm == "control" else image_loss + float(lambda_feature) * feature_loss
    _finite_tensor(total_loss, "V3 total loss")
    for tensor in (raw_cmos, result.ols_input_field.tensor, result.z_ols.coefficients, result.generator_input_field.tensor):
        _require(tensor.requires_grad, "V3 autograd boundary tensor is detached")
        tensor.retain_grad()
    image_gradients = dict(zip(ALLOWED_TRAINABLE_PARAMETER_NAMES, torch.autograd.grad(image_loss, parameters, retain_graph=True, allow_unused=False), strict=True))
    if feature_loss.requires_grad:
        feature_gradients = dict(zip(ALLOWED_TRAINABLE_PARAMETER_NAMES, torch.autograd.grad(feature_loss, parameters, retain_graph=True, allow_unused=True), strict=True))
    else:
        feature_gradients = dict.fromkeys(ALLOWED_TRAINABLE_PARAMETER_NAMES)
    total_loss.backward()
    total_gradients = {name: parameter.grad.detach().clone() if parameter.grad is not None else None for name, parameter in named.items()}
    image_gradient_norms, _, _, _ = _gradient_summary(image_gradients)
    feature_gradient_norms, _, _, _ = _gradient_summary(feature_gradients)
    total_gradient_norms, total_zero_fraction, total_nonfinite_fraction, _ = _gradient_summary(total_gradients)
    _require(all(parameter.grad is not None for parameter in parameters), "V3 trainable parameter gradient is missing")
    _require(all(parameter.grad is None for name, parameter in named.items() if name not in ALLOWED_TRAINABLE_PARAMETER_NAMES), "forbidden runtime parameter received a gradient")
    if arm == "candidate" and not detached_provider:
        _require(semantic_input_gradient is not None, "V3 candidate semantic input gradient is absent")
        _finite_tensor(semantic_input_gradient, "V3 candidate semantic input gradient")
        _require(_norm(semantic_input_gradient) > 0.0, "V3 candidate semantic input gradient is zero")
    if detached_provider:
        _require(semantic_input_gradient is None and not feature_loss.requires_grad, "detached-provider negative control retained a semantic gradient")
    optimizer.step()
    torch.cuda.synchronize(consumer.device)
    after_parameters = {name: parameter.detach().clone() for name, parameter in named.items()}
    for name, parameter in named.items():
        _finite_tensor(parameter, f"V3 updated parameter {name}")
    update_norms = {name: _norm(after_parameters[name] - before_parameters[name]) for name in named}
    _require(all(update_norms[name] > 0.0 for name in ALLOWED_TRAINABLE_PARAMETER_NAMES), "V3 optimizer did not update both trainable groups")
    frozen_unchanged = _tree_equal(before_frozen, consumer._frozen_state_snapshot())
    _require(frozen_unchanged, "V3 frozen runtime state changed")
    provider_unchanged = _tree_equal(before_provider, _cpu_tree(provider.state_dict()))
    _require(provider_unchanged and all(parameter.grad is None for parameter in provider.parameters()), "V3 provider parameter firewall changed")
    if semantic_input_gradient is None:
        semantic_input_norm = None
    else:
        semantic_input_norm = _norm(semantic_input_gradient)
    trace = {
        "step": step,
        "arm": arm,
        "detached_provider": detached_provider,
        "lambda_feature": float(lambda_feature),
        "losses": {
            "L_image": float(image_loss.detach().cpu()),
            "L_feature": float(feature_loss.detach().cpu()),
            "L_total": float(total_loss.detach().cpu()),
        },
        "gradient_norms": {
            "image": image_gradient_norms,
            "feature": feature_gradient_norms,
            "total": total_gradient_norms,
            "semantic_input": semantic_input_norm,
        },
        "parameter_update_norms": update_norms,
        "zero_gradient_fraction": total_zero_fraction,
        "nonfinite_gradient_fraction": total_nonfinite_fraction,
        "frozen_runtime_unchanged": frozen_unchanged,
        "provider_unchanged_and_frozen": provider_unchanged and all(parameter.grad is None for parameter in provider.parameters()),
        "modal_projection_active": True,
        "w_ols_active": True,
        "ols_transport": "T=I_100_coefficient_space",
        "raw_cmos": _raw_summary(raw_cmos),
        "memory": _memory_snapshot(consumer.device),
    }
    return {
        "trace": trace,
        "trainable_state": _cpu_tree(consumer.trainable_state_dict()),
        "runtime_state": _runtime_state(consumer),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "gradient_state": _cpu_tree(total_gradients),
        "image_gradient_state": _cpu_tree(image_gradients),
        "feature_gradient_state": _cpu_tree(feature_gradients),
    }


def _calibrate_lambda(
    contract: v2.FrozenV2TrainingContract,
    provider: FrozenMNISTEvaluator,
    images: torch.Tensor,
    base_state: Mapping[str, Any],
    initial_rng: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    _restore_rng_state(initial_rng)
    consumer = _new_consumer(contract, device)
    _load_runtime_state(consumer, base_state)
    _restore_rng_state(initial_rng)
    batch = images.to(device)
    before_state = _runtime_state(consumer)
    before_provider = _cpu_tree(provider.state_dict())
    result = consumer.runtime.forward_from_image(batch)
    _check_forward_path(consumer, result)
    raw_cmos = result.raw_intensity
    image_loss = (raw_cmos - result.image_input.slm_intensity).square().mean()
    feature_loss, semantic_input = _feature_loss(provider, raw_cmos, batch, detached_provider=False)
    _require(feature_loss.requires_grad, "V3 feature loss is not differentiable")
    _require(semantic_input is not None and _norm(semantic_input) > 0.0, "V3 calibration semantic input gradient is undefined or zero")
    named = dict(consumer.runtime.named_parameters())
    parameters = [named[name] for name in ALLOWED_TRAINABLE_PARAMETER_NAMES]
    image_gradients = torch.autograd.grad(image_loss, parameters, retain_graph=True, allow_unused=False)
    feature_gradients = torch.autograd.grad(feature_loss, parameters, retain_graph=True, allow_unused=True)
    g_image = math.sqrt(sum(float(gradient.detach().to(torch.float64).square().sum().cpu()) for gradient in image_gradients))
    g_feature = math.sqrt(sum(float(gradient.detach().to(torch.float64).square().sum().cpu()) for gradient in feature_gradients if gradient is not None))
    _require(math.isfinite(g_image) and math.isfinite(g_feature) and g_image > 0.0 and g_feature > 0.0, "V3 lambda gradients are zero, nonfinite, or undefined")
    raw_ratio = g_image / g_feature
    lambda_feature = min(LAMBDA_UPPER, max(LAMBDA_LOWER, raw_ratio))
    _require(lambda_feature == min(LAMBDA_UPPER, max(LAMBDA_LOWER, raw_ratio)), "V3 lambda formula was not applied exactly")
    _require(_digest_tree(before_state) == _digest_tree(_runtime_state(consumer)), "V3 lambda calibration changed runtime state")
    _require(_tree_equal(before_provider, _cpu_tree(provider.state_dict())), "V3 lambda calibration changed provider state")
    _require(all(parameter.grad is None for parameter in provider.parameters()), "V3 provider accumulated a calibration gradient")
    return {
        "pre_update": True,
        "batch_role": "TRAIN",
        "batch_size": LOGICAL_BATCH_SIZE,
        "g_image": g_image,
        "g_feature": g_feature,
        "raw_ratio": raw_ratio,
        "lambda_feature": lambda_feature,
        "clipped": bool(raw_ratio < LAMBDA_LOWER or raw_ratio > LAMBDA_UPPER),
        "semantic_input_gradient_norm": _norm(semantic_input),
        "state_unchanged": True,
        "provider_unchanged_and_frozen": True,
    }


def _checkpoint_payload(
    contract: v2.FrozenV2TrainingContract,
    *,
    arm: str,
    lambda_feature: float,
    base_state_digest: str,
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    optimizer: torch.optim.Optimizer,
    sampler: DeterministicTrainRoleSampler,
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": 1,
        "contract": {"path": V3_CONTRACT_PATH.as_posix(), "sha256": V3_CONTRACT_SHA256},
        "v2_contract": contract.identity(),
        "arm": arm,
        "objective": "PURE_IMAGE_CONTROL" if arm == "control" else "FROZEN_FEATURE_CORRECTED",
        "lambda_feature": float(lambda_feature),
        "checkpoint_step": CHECKPOINT_STEP,
        "base_initial_state_digest": base_state_digest,
        "runtime_state": _runtime_state(consumer),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "optimizer_manifest": consumer.optimizer_manifest(optimizer),
        "sampler_state": _cpu_tree(sampler.state_dict()),
        "rng_state": _cpu_tree(_capture_rng_state()),
        "trainable_parameter_names": list(ALLOWED_TRAINABLE_PARAMETER_NAMES),
        "data_roles": {"train": True, "development": False, "calibration": False, "protected": False, "test": False, "sealed": False},
        "training_authorized": False,
        "scientific_training_executed": False,
    }


def _run_arm(
    root: Path,
    contract: v2.FrozenV2TrainingContract,
    provider: FrozenMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    sampler_initial_state: Mapping[str, Any],
    base_state: Mapping[str, Any],
    initial_rng: Mapping[str, Any],
    lambda_feature: float,
    device: torch.device,
    *,
    arm: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    _restore_rng_state(initial_rng)
    consumer = _new_consumer(contract, device)
    _load_runtime_state(consumer, base_state)
    _restore_rng_state(initial_rng)
    start_state = _runtime_state(consumer)
    start_rng = _capture_rng_state()
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=FUTURE_STAGE_A_STEPS)
    sampler.load_state_dict(sampler_initial_state)
    optimizer = consumer.create_optimizer()
    optimizer_manifest = consumer.optimizer_manifest(optimizer)
    traces: list[dict[str, Any]] = []
    step_results: dict[int, dict[str, Any]] = {}
    checkpoint_hash: str | None = None
    for step, expected in enumerate(batches, start=1):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(indices, expected[0]) and torch.equal(batch_labels, expected[1]), f"{arm} sampler schedule diverged at step {step}")
        batch_images = images.index_select(0, indices).to(device)
        result = _run_step(consumer, provider, batch_images, optimizer, arm=arm, lambda_feature=lambda_feature, step=step)
        traces.append(result["trace"])
        step_results[step] = result
        if step == CHECKPOINT_STEP:
            payload = _checkpoint_payload(
                contract,
                arm=arm,
                lambda_feature=lambda_feature,
                base_state_digest=_digest_tree(base_state),
                consumer=consumer,
                optimizer=optimizer,
                sampler=sampler,
            )
            atomic_torch_save_new(checkpoint_path, payload, artifact_kind=f"V3 {arm} preflight checkpoint")
            checkpoint_hash = file_sha256(checkpoint_path)
    torch.cuda.synchronize(device)
    continuous_step = step_results[PREFLIGHT_STEPS]
    continuous_final_state = _cpu_tree(consumer.trainable_state_dict())
    continuous_optimizer_state = _cpu_tree(optimizer.state_dict())
    continuous_sampler_state = _cpu_tree(sampler.state_dict())
    continuous_rng = _cpu_tree(_capture_rng_state())

    _require(checkpoint_hash is not None, f"{arm} checkpoint was not written")
    loaded = _mapping(safe_torch_load(checkpoint_path, expected_sha256=checkpoint_hash, map_location="cpu"), f"{arm} checkpoint")
    _require(loaded.get("schema") == CHECKPOINT_SCHEMA and loaded.get("arm") == arm and loaded.get("checkpoint_step") == CHECKPOINT_STEP, f"{arm} checkpoint identity changed")
    _require(loaded.get("lambda_feature") == float(lambda_feature) and loaded.get("base_initial_state_digest") == _digest_tree(base_state), f"{arm} checkpoint lambda/state identity changed")
    _restore_rng_state(_mapping(loaded.get("rng_state"), f"{arm} checkpoint RNG"))
    resumed = _new_consumer(contract, device)
    _load_runtime_state(resumed, _mapping(loaded.get("runtime_state"), f"{arm} checkpoint runtime state"))
    resumed_optimizer = resumed.create_optimizer()
    resumed_optimizer.load_state_dict(dict(_mapping(loaded.get("optimizer_state"), f"{arm} checkpoint optimizer state")))
    resumed.optimizer_manifest(resumed_optimizer)
    resumed_sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=FUTURE_STAGE_A_STEPS)
    resumed_sampler.load_state_dict(_mapping(loaded.get("sampler_state"), f"{arm} checkpoint sampler state"))
    resumed_indices, resumed_labels = resumed_sampler.next_batch()
    _require(torch.equal(resumed_indices, batches[CHECKPOINT_STEP][0]) and torch.equal(resumed_labels, batches[CHECKPOINT_STEP][1]), f"{arm} checkpoint next batch changed")
    _restore_rng_state(_mapping(loaded.get("rng_state"), f"{arm} checkpoint RNG"))
    resumed_result = _run_step(
        resumed,
        provider,
        images.index_select(0, resumed_indices).to(device),
        resumed_optimizer,
        arm=arm,
        lambda_feature=lambda_feature,
        step=PREFLIGHT_STEPS,
    )
    continuous_recovery_trace = {key: value for key, value in continuous_step["trace"].items() if key != "memory"}
    resumed_recovery_trace = {key: value for key, value in resumed_result["trace"].items() if key != "memory"}
    resume_trace_difference = _max_numeric_difference(continuous_recovery_trace, resumed_recovery_trace)
    resume_parameter_difference = _max_state_difference(continuous_final_state, resumed_result["trainable_state"])
    resume_optimizer_equal = _tree_equal(continuous_optimizer_state, resumed_result["optimizer_state"])
    resume_optimizer_difference = _max_state_difference(continuous_optimizer_state, resumed_result["optimizer_state"])
    resumed_sampler_state = _cpu_tree(resumed_sampler.state_dict())
    resumed_rng = _cpu_tree(_capture_rng_state())
    resume_sampler_equal = _tree_equal(continuous_sampler_state, resumed_sampler_state)
    resume_rng_equal = _tree_equal(continuous_rng, resumed_rng)
    resume_sampler_difference = _max_state_difference(continuous_sampler_state, resumed_sampler_state)
    resume_rng_difference = _max_state_difference(continuous_rng, resumed_rng)
    _require(resume_trace_difference <= 1.0e-5, f"{arm} checkpoint/resume trace differs: {resume_trace_difference}")
    _require(
        resume_parameter_difference <= 1.0e-5 and resume_optimizer_difference <= 1.0e-5 and resume_sampler_equal and resume_rng_equal,
        f"{arm} checkpoint/resume state differs: parameter={resume_parameter_difference}, optimizer={resume_optimizer_equal}/{resume_optimizer_difference}, sampler={resume_sampler_equal}/{resume_sampler_difference}, rng={resume_rng_equal}/{resume_rng_difference}",
    )
    return {
        "arm": arm,
        "objective": "PURE_IMAGE_CONTROL" if arm == "control" else "FROZEN_FEATURE_CORRECTED",
        "initial_state_digest": _digest_tree(start_state),
        "initial_rng_digest": _digest_tree(start_rng),
        "sampler_initial_state_digest": _digest_tree(sampler_initial_state),
        "optimizer_manifest": optimizer_manifest,
        "traces": traces,
        "checkpoint": {"path": _relative(root, checkpoint_path), "sha256": checkpoint_hash, "step": CHECKPOINT_STEP, "loaded": True},
        "checkpoint_resume": {
            "next_batch_equal": True,
            "trace_max_abs_difference": resume_trace_difference,
            "parameter_max_abs_difference": resume_parameter_difference,
            "optimizer_equal": resume_optimizer_equal,
            "sampler_equal": resume_sampler_equal,
            "rng_equal": resume_rng_equal,
            "lambda_equal": loaded.get("lambda_feature") == float(lambda_feature),
            "identity_equal": loaded.get("contract", {}).get("sha256") == V3_CONTRACT_SHA256 and loaded.get("arm") == arm,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "updates": PREFLIGHT_STEPS,
    }


def _single_step_variant(
    contract: v2.FrozenV2TrainingContract,
    provider: FrozenMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    first_batch: tuple[torch.Tensor, torch.Tensor],
    sampler_initial_state: Mapping[str, Any],
    base_state: Mapping[str, Any],
    initial_rng: Mapping[str, Any],
    device: torch.device,
    *,
    arm: str,
    lambda_feature: float,
    detached_provider: bool = False,
) -> dict[str, Any]:
    _restore_rng_state(initial_rng)
    consumer = _new_consumer(contract, device)
    _load_runtime_state(consumer, base_state)
    _restore_rng_state(initial_rng)
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=FUTURE_STAGE_A_STEPS)
    sampler.load_state_dict(sampler_initial_state)
    indices, labels = sampler.next_batch()
    _require(torch.equal(indices, first_batch[0]) and torch.equal(labels, first_batch[1]), "V3 counterfactual sampler diverged")
    optimizer = consumer.create_optimizer()
    return _run_step(
        consumer,
        provider,
        images.index_select(0, indices).to(device),
        optimizer,
        arm=arm,
        lambda_feature=lambda_feature,
        step=1,
        detached_provider=detached_provider,
    ) | {"optimizer_manifest": consumer.optimizer_manifest(optimizer)}


def _disabled_term_control(
    contract: v2.FrozenV2TrainingContract,
    provider: FrozenMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    first_batch: tuple[torch.Tensor, torch.Tensor],
    sampler_initial_state: Mapping[str, Any],
    base_state: Mapping[str, Any],
    initial_rng: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    control = _single_step_variant(contract, provider, images, labels, train_indices, first_batch, sampler_initial_state, base_state, initial_rng, device, arm="control", lambda_feature=0.0)
    disabled = _single_step_variant(contract, provider, images, labels, train_indices, first_batch, sampler_initial_state, base_state, initial_rng, device, arm="candidate", lambda_feature=0.0)
    loss_difference = _max_numeric_difference(control["trace"]["losses"], disabled["trace"]["losses"])
    gradient_difference = _max_state_difference(control["gradient_state"], disabled["gradient_state"])
    image_gradient_difference = _max_state_difference(control["image_gradient_state"], disabled["image_gradient_state"])
    update_difference = _max_numeric_difference(control["trace"]["parameter_update_norms"], disabled["trace"]["parameter_update_norms"])
    parameter_difference = _max_state_difference(control["trainable_state"], disabled["trainable_state"])
    optimizer_equal = _tree_equal(control["optimizer_state"], disabled["optimizer_state"])
    passed = all(value <= 1.0e-5 for value in (loss_difference, gradient_difference, image_gradient_difference, update_difference, parameter_difference)) and optimizer_equal
    _require(passed, "disabled-term candidate does not reproduce pure-image control")
    return {
        "expected": "lambda_feature=0 candidate reproduces pure-image control",
        "passed": True,
        "tolerance": {"rtol": DETERMINISTIC_RTOL, "atol": DETERMINISTIC_ATOL},
        "loss_max_abs_difference": loss_difference,
        "gradient_max_abs_difference": gradient_difference,
        "image_gradient_max_abs_difference": image_gradient_difference,
        "update_norm_max_abs_difference": update_difference,
        "parameter_max_abs_difference": parameter_difference,
        "optimizer_equal": optimizer_equal,
    }


def _detached_provider_negative_control(
    contract: v2.FrozenV2TrainingContract,
    provider: FrozenMNISTEvaluator,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    first_batch: tuple[torch.Tensor, torch.Tensor],
    sampler_initial_state: Mapping[str, Any],
    base_state: Mapping[str, Any],
    initial_rng: Mapping[str, Any],
    device: torch.device,
    lambda_feature: float,
) -> dict[str, Any]:
    pure = _single_step_variant(contract, provider, images, labels, train_indices, first_batch, sampler_initial_state, base_state, initial_rng, device, arm="control", lambda_feature=0.0)
    detached = _single_step_variant(contract, provider, images, labels, train_indices, first_batch, sampler_initial_state, base_state, initial_rng, device, arm="candidate", lambda_feature=lambda_feature, detached_provider=True)
    gradient_difference = _max_state_difference(pure["gradient_state"], detached["gradient_state"])
    parameter_difference = _max_state_difference(pure["trainable_state"], detached["trainable_state"])
    feature_forward_finite = bool(math.isfinite(detached["trace"]["losses"]["L_feature"]))
    semantic_gradient_present = detached["trace"]["gradient_norms"]["semantic_input"] is not None
    _require(feature_forward_finite and not semantic_gradient_present, "detached-provider negative control did not eliminate semantic input gradient")
    _require(gradient_difference <= 1.0e-5 and parameter_difference <= 1.0e-5, "detached-provider path did not reduce to image-only update")
    return {
        "expected_rejection": "invalid_harness",
        "rejected_as_invalid_harness": True,
        "forward_feature_loss_preserved_and_finite": feature_forward_finite,
        "semantic_input_gradient_present": False,
        "semantic_gradient_contribution_eliminated": True,
        "pure_image_gradient_max_abs_difference": gradient_difference,
        "pure_image_parameter_max_abs_difference": parameter_difference,
    }


def _base_payload(root: Path, started: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "preflight_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "result": "INVALID",
        "status": "invalid_harness",
        "classification": "INVALID_V3_PREFLIGHT_NOT_COMPLETED",
        "provenance": {"git_start": dict(started)},
        "claim_boundary": {
            "engineering_smoke_only": True,
            "scientific_training_executed": False,
            "scientific_comparison_executed": False,
            "stage_b_executed": False,
            "scale_extraction_executed": False,
            "prior_generation_executed": False,
            "protected_accessed": False,
            "test_accessed": False,
            "sealed_accessed": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "next": None,
    }


def _resolve_output_dir(root: Path, output_dir: Path | None) -> Path:
    target = root / DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    target = target if target.is_absolute() else root / target
    target = target.resolve()
    outputs_root = (root / "outputs").resolve()
    _require(target.is_relative_to(outputs_root), "V3 preflight output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite existing V3 preflight output: {target}")
    return target


def run_preflight(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    target = _resolve_output_dir(root, output_dir)
    target.mkdir(parents=True, exist_ok=False)
    started = _git_state(root)
    payload = _base_payload(root, started)
    payload["entrypoint"] = {"path": _relative(root, Path(__file__)), "sha256": file_sha256(Path(__file__))}
    authorities: dict[str, Any] | None = None
    input_identity: dict[str, str] | None = None
    checkpoint_refs: dict[str, Any] = {}
    try:
        contract_payload, v2_contract, authorities = _validate_v3_authority(root)
        input_payload = {
            "schema": PREFLIGHT_INPUT_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "preflight_id": TASK_ID,
            "contract": {"path": V3_CONTRACT_PATH.as_posix(), "sha256": V3_CONTRACT_SHA256},
            "entrypoint": dict(payload["entrypoint"]),
            "protocol": {
                "target_device_grid": PRODUCTION_PROPAGATION_GRID,
                "logical_batch_size": LOGICAL_BATCH_SIZE,
                "preflight_updates_per_arm": PREFLIGHT_STEPS,
                "future_stage_a_steps": FUTURE_STAGE_A_STEPS,
                "future_checkpoint_steps": list(FUTURE_CHECKPOINT_STEPS),
                "checkpoint_step": CHECKPOINT_STEP,
                "control": "L_control=L_image",
                "candidate": "L_candidate=L_image+lambda_feature*L_feature",
                "lambda_formula": "min(10.0, max(0.1, g_image/g_feature))",
                "optimizer": {"name": "Adam", "learning_rate": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8, "weight_decay": 0.0, "amsgrad": False},
            },
            "claim_boundary": {"scientific_comparison_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False},
            "input_self_hash": "not_embedded_to_avoid_self_reference",
        }
        input_path = target / INPUT_NAME
        atomic_write_new_json(input_path, input_payload, artifact_kind="V3 objective-correction preflight input")
        input_identity = {"path": _relative(root, input_path), "sha256": file_sha256(input_path)}
        payload["preflight_input"] = dict(input_identity)
        payload["authorities"] = authorities

        images, labels, train_indices, dataset = _load_train_role(root)
        payload["dataset"] = dataset
        initial_sampler_state, batches, schedule = _make_schedule(train_indices, labels)
        payload["schedule"] = schedule
        device, device_facts = _cuda_device_facts()
        payload["device"] = device_facts
        torch.manual_seed(INITIALIZATION_SEED)
        torch.cuda.manual_seed_all(INITIALIZATION_SEED)
        frozen_checkpoint_path = _path(root, authorities["frozen_evaluator"]["checkpoint"]["path"])
        frozen_payload = _mapping(safe_torch_load(frozen_checkpoint_path, expected_sha256=authorities["frozen_evaluator"]["checkpoint"]["sha256"], map_location="cpu"), "FrozenMNISTEvaluator checkpoint")
        _require(frozen_payload.get("architecture_kind") == "frozen_mnist_evaluator_v1", "FrozenMNISTEvaluator checkpoint architecture changed")
        _require(float(frozen_payload.get("validation_accuracy", 0.0)) >= 0.985, "FrozenMNISTEvaluator checkpoint qualification changed")
        primary = FrozenMNISTEvaluator()
        primary.load_state_dict(_mapping(frozen_payload.get("state_dict"), "FrozenMNISTEvaluator state"), strict=True)
        primary.to(device).freeze()
        smooth_checkpoint_path = _path(root, authorities["smooth_evaluator"]["checkpoint"]["path"])
        smooth_payload = _mapping(safe_torch_load(smooth_checkpoint_path, expected_sha256=authorities["smooth_evaluator"]["checkpoint"]["sha256"], map_location="cpu"), "SmoothMNISTEvaluator checkpoint")
        _require(smooth_payload.get("architecture_kind") == "smooth_mnist_evaluator_v2", "SmoothMNISTEvaluator checkpoint architecture changed")
        smooth = SmoothMNISTEvaluator()
        smooth.load_state_dict(_mapping(smooth_payload.get("state_dict"), "SmoothMNISTEvaluator state"), strict=True)
        smooth.to(device).freeze()
        _require(all(not parameter.requires_grad for parameter in primary.parameters()) and all(not parameter.requires_grad for parameter in smooth.parameters()), "evaluator freeze boundary failed before preflight")

        base_consumer = _new_consumer(v2_contract, device)
        base_state = _runtime_state(base_consumer)
        base_state_digest = _digest_tree(base_state)
        _require(all(torch.count_nonzero(parameter.detach()).item() == 0 for parameter in base_consumer.runtime.parameters()), "V3 fresh runtime is not zero-phase initialized")
        initial_rng = _capture_rng_state()
        payload["initial_state"] = {"runtime_state_sha256": base_state_digest, "zero_phase": True, "rng_state_sha256": _digest_tree(initial_rng), "parameter_manifest": base_consumer.parameter_manifest()}
        del base_consumer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = _memory_snapshot(device)

        calibration_batch = images.index_select(0, batches[0][0]).to(device)
        calibration = _calibrate_lambda(v2_contract, primary, calibration_batch, base_state, initial_rng, device)
        payload["lambda_calibration"] = calibration

        control_checkpoint_path = target / CONTROL_CHECKPOINT_NAME
        candidate_checkpoint_path = target / CANDIDATE_CHECKPOINT_NAME
        control_result = _run_arm(root, v2_contract, primary, images, labels, train_indices, batches, initial_sampler_state, base_state, initial_rng, 0.0, device, arm="control", checkpoint_path=control_checkpoint_path)
        candidate_result = _run_arm(root, v2_contract, primary, images, labels, train_indices, batches, initial_sampler_state, base_state, initial_rng, calibration["lambda_feature"], device, arm="candidate", checkpoint_path=candidate_checkpoint_path)
        checkpoint_refs = {"control": dict(control_result["checkpoint"]), "candidate": dict(candidate_result["checkpoint"])}
        payload["arms"] = {"control": control_result, "candidate": candidate_result}

        _require(control_result["initial_state_digest"] == candidate_result["initial_state_digest"] == base_state_digest, "control/candidate initial runtime states differ")
        _require(control_result["initial_rng_digest"] == candidate_result["initial_rng_digest"] == _digest_tree(initial_rng), "control/candidate initial RNG states differ")
        _require(control_result["sampler_initial_state_digest"] == candidate_result["sampler_initial_state_digest"] == _digest_tree(initial_sampler_state), "control/candidate sampler states differ")
        _require(control_result["optimizer_manifest"] == candidate_result["optimizer_manifest"], "control/candidate optimizer construction differs")
        payload["shared_state_matching"] = {"initial_runtime_equal": True, "initial_rng_equal": True, "sampler_equal": True, "optimizer_manifest_equal": True, "base_initial_state_digest": base_state_digest}

        payload["disabled_term_control"] = _disabled_term_control(v2_contract, primary, images, labels, train_indices, batches[0], initial_sampler_state, base_state, initial_rng, device)
        payload["detached_provider_negative_control"] = _detached_provider_negative_control(v2_contract, primary, images, labels, train_indices, batches[0], initial_sampler_state, base_state, initial_rng, device, calibration["lambda_feature"])

        payload["frozen_parameter_firewall"] = {
            "provider": all(parameter.grad is None and not parameter.requires_grad for parameter in primary.parameters()),
            "independent_evaluator": all(parameter.grad is None and not parameter.requires_grad for parameter in smooth.parameters()),
            "runtime_nonphase_state": all(trace["frozen_runtime_unchanged"] for result in (control_result, candidate_result) for trace in result["traces"]),
            "ols_transport": "T=I_100_coefficient_space",
        }
        _require(all(payload["frozen_parameter_firewall"].values() if isinstance(payload["frozen_parameter_firewall"], list) else [payload["frozen_parameter_firewall"][key] for key in ("provider", "independent_evaluator", "runtime_nonphase_state")]), "V3 frozen-parameter firewall failed")
        memory_after = _memory_snapshot(device)
        payload["memory"] = {
            "before": memory_before,
            "after": memory_after,
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "headroom_bytes": int(memory_after["total_bytes"] - torch.cuda.max_memory_reserved(device)),
            "exact_grid": PRODUCTION_PROPAGATION_GRID,
            "exact_batch_size": LOGICAL_BATCH_SIZE,
            "feasible": int(torch.cuda.max_memory_reserved(device)) < memory_after["total_bytes"] and memory_after["free_bytes"] > 0,
        }
        _require(payload["memory"]["feasible"], "V3 exact-grid memory headroom is unavailable")
        payload["result"] = "AUTHORIZABLE"
        payload["status"] = "valid_completed"
        payload["classification"] = SUCCESS_CLASSIFICATION
        payload["next"] = "explicit_authorization_required_for_future_v3_scientific_stage_a"
        payload["next_note"] = "AUTHORIZABLE means only that the frozen paired Stage-A experiment is technically executable; no scientific quality claim is made"
    except PreflightBlocked as blocked:
        payload.update(blocked.details)
        payload["result"] = "BLOCKED"
        payload["status"] = "valid_failed"
        payload["classification"] = blocked.classification
        payload["hard_blocker"] = {"stage": blocked.stage, "message": str(blocked)}
        payload["next"] = "resolve_preflight_blocker_before_any_future_scientific_run"
    except InvalidPreflight as error:
        payload["result"] = "INVALID"
        payload["status"] = "invalid_harness"
        payload["classification"] = "INVALID_V3_PREFLIGHT_HARNESS"
        payload["invalid_input"] = {"type": type(error).__name__, "message": str(error)}
    except Exception as error:
        payload["result"] = "INVALID"
        payload["status"] = "invalid_harness"
        payload["classification"] = "INVALID_V3_PREFLIGHT_UNEXPECTED_EXECUTION_ERROR"
        payload["unexpected_error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-10_000:]}
    finally:
        ending = git_facts(root)
        payload["provenance"].update({"git_end": ending, "branch_unchanged": started["branch"] == ending["branch"], "head_unchanged": started["head"] == ending["head"], "worktree_dirty": ending["worktree_dirty"]})
        payload["claim_boundary"].update({"scientific_training_executed": False, "scientific_comparison_executed": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "protected_accessed": False, "test_accessed": False, "sealed_accessed": False, "training_authorized": False, "formal_promotion": False, "generation_readiness": "not_ready"})
        payload["artifacts"] = {"preflight_input": input_identity, "control_checkpoint": checkpoint_refs.get("control"), "candidate_checkpoint": checkpoint_refs.get("candidate"), "report_self_hash": "not_embedded_to_avoid_self_reference"}

    report_path = target / REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V3 objective-correction preflight report")
    report_ref = {"path": _relative(root, report_path), "sha256": file_sha256(report_path)}
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "contract": (authorities or {}).get("v3_contract"),
        "entrypoint": payload["entrypoint"],
        "authorities": authorities,
        "provenance": payload["provenance"],
        "artifacts": {"preflight_input": input_identity, "control_checkpoint": checkpoint_refs.get("control"), "candidate_checkpoint": checkpoint_refs.get("candidate"), "report": report_ref},
        "claim_boundary": payload["claim_boundary"],
        "lineage_self_hash": "not_embedded_to_avoid_self_reference",
    }
    lineage_path = target / LINEAGE_NAME
    atomic_write_new_json(lineage_path, lineage, artifact_kind="V3 objective-correction preflight lineage")
    result = {
        "result": payload["result"],
        "status": payload["status"],
        "classification": payload["classification"],
        "preflight_id": TASK_ID,
        "preflight_input": input_identity,
        "report": report_ref,
        "lineage": {"path": _relative(root, lineage_path), "sha256": file_sha256(lineage_path)},
        "lambda_calibration": payload.get("lambda_calibration"),
        "three_step_traces": {name: value.get("traces") for name, value in payload.get("arms", {}).items()} if isinstance(payload.get("arms"), Mapping) else None,
        "training_authorized": False,
        "stage_b_authorized": False,
        "generation_readiness": "not_ready",
        "formal_promotion": False,
        "scientific_comparison_executed": False,
        "next": payload["next"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("preflight",), required=True)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_preflight(repo_root=args.repo_root, output_dir=args.output_dir)
    return 0 if result["result"] in {"AUTHORIZABLE", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
