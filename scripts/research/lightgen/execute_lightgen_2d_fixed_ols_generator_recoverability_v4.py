"""Run the bounded three-update V4 fixed-OLS generator preflight.

The preflight is an engineering consumer of the V4 contract.  It performs
exactly three real CUDA generator-only updates through the production
LightGen-aligned runtime, then stops.  It never performs the 300-update
scientific comparison, Stage B, scale extraction, prior generation, protected
or test reads, or formal promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.nn import functional as F

from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_complex_ols_training_v2 import DeterministicTrainRoleSampler, LightGen2DComplexOLSTrainingConsumerV2, LOGICAL_BATCH_SIZE, PRODUCTION_PROPAGATION_GRID, file_sha256, git_facts, load_v2_training_contract, tensor_sha256
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_new_json, safe_torch_load
from ldm_project.v2.student.evaluator import (
    FrozenMNISTEvaluator,
    SmoothMNISTEvaluator,
    load_local_mnist,
    stratified_mnist_indices,
)


TASK_ID = "EXECUTE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_PREFLIGHT_V4"
SUCCESS_CLASSIFICATION = "FIXED_OLS_GENERATOR_RECOVERABILITY_PREFLIGHT_AUTHORIZABLE"
REPORT_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_preflight_report.v1"
INPUT_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_preflight_input.v1"
LINEAGE_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_preflight_lineage.v1"
CHECKPOINT_SCHEMA = "lightgen_2d_fixed_ols_generator_recoverability_preflight_checkpoint.v1"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "383197a930dd48f98294120f8cf867444f994f85"
CONTRACT_PATH = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_contract_v4_attempt_01/"
    "lightgen_2d_fixed_ols_generator_recoverability_contract.v4.json"
)
CONTRACT_SHA256 = "d46740ded4db58d7d9f0caa8acaf40942d03dc3407d9611d24efcaa12d3036b8"
DEFINITION_PATH = Path("scripts/research/lightgen/define_lightgen_2d_fixed_ols_generator_recoverability_contract_v4.py")
DEFINITION_SHA256 = "e2bf1f1f7ceae4e48d0c03d3146b60643c34ee6af82e89b38bcf6102cb2572cc"
V3_CONTROL_CHECKPOINT_PATH = Path(
    "outputs/20260911_v3/runs/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_scientific_attempt_01/"
    "lightgen_2d_semantic_complex_ols_objective_correction_v3_control_step_300_checkpoint.v1.pt"
)
V3_CONTROL_CHECKPOINT_SHA256 = "a42a43fdf99cf12d9abd1917a6d025fc2fcb9ecee39f0c05429dd8d7a8870079"
TRAINING_SOURCE_PATH = Path("data/MNIST/processed/training.pt")
TRAINING_SOURCE_SHA256 = "8151bbd33f70ee47d9557adaaa667bb878b38cf4015dcac206d1411fb0d0effd"
ALIGNED_RUNTIME_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen/"
    "lightgen_2d_semantic_generation_input_ols_alignment.py"
)
ALIGNED_RUNTIME_SHA256 = "d01a234f6486a99c7ac2307b1b4021ed2ffccb53812a0a858675617a9023c3ce"
FROZEN_CHECKPOINT_SHA256 = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"
SMOOTH_CHECKPOINT_SHA256 = "45879962cf8a6dc51e141a2f2a9892356ede97dd45063f38d830299aa877e26c"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260911_v4/runs/"
    "lightgen_2d_fixed_ols_generator_recoverability_preflight_v4_attempt_01"
)
INPUT_NAME = "lightgen_2d_fixed_ols_generator_recoverability_preflight_input.v1.json"
REPORT_NAME = "lightgen_2d_fixed_ols_generator_recoverability_preflight_report.v1.json"
LINEAGE_NAME = "lightgen_2d_fixed_ols_generator_recoverability_preflight_lineage.v1.json"
CHECKPOINT_NAME = "lightgen_2d_fixed_ols_generator_recoverability_preflight_step_2_checkpoint.v1.pt"

PREFLIGHT_STEPS = 3
CHECKPOINT_STEP = 2
SCHEDULE_SEED = 1234
SCIENTIFIC_STEPS = 300
PRODUCTION_GRID = 2048
LEARNING_RATE = 1.0e-3
ADAM_BETAS = (0.9, 0.999)
ADAM_EPSILON = 1.0e-8
WEIGHT_DECAY = 0.0
AMSGRAD = False
DETERMINISTIC_RTOL = 1.0e-5
DETERMINISTIC_ATOL = 1.0e-6
INITIAL_RUNTIME_STATE_SHA256 = "0f90086cf26a179905a27a5a519f7599de256e4f4225f7798f24aeed086eb813"
PHASE_SHAPE = (2, 1, 100, 100)
PHASE_DTYPE = torch.float32
OLS_TRANSPORT = torch.eye(100, dtype=torch.complex64)


class InvalidPreflight(RuntimeError):
    """An implementation, authority, or invariant is invalid."""


class PreflightBlocked(RuntimeError):
    """A valid boundary prevents the requested preflight from running."""

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


def _path(root: Path, relative: str | Path) -> Path:
    target = Path(relative)
    _require(not target.is_absolute(), f"absolute authority path is not allowed: {target}")
    return (root / target).resolve()


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


def _tree_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_tree_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_tree_equal(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def _max_difference(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 0.0 if left is right else math.inf
    if torch.is_tensor(left) and torch.is_tensor(right):
        if left.shape != right.shape:
            return math.inf
        if left.numel() == 0:
            return 0.0
        comparison_dtype = torch.complex128 if (torch.is_complex(left) or torch.is_complex(right)) else torch.float64
        return float((left.to(comparison_dtype) - right.to(comparison_dtype)).abs().max())
    if torch.is_tensor(left) or torch.is_tensor(right):
        return math.inf
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            return math.inf
        return max((_max_difference(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return math.inf
        return max((_max_difference(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    return 0.0 if left == right else math.inf


def _finite(value: torch.Tensor, name: str) -> None:
    _require(torch.is_tensor(value) and bool(torch.isfinite(value).all()), f"{name} is nonfinite")


def _norm(value: torch.Tensor) -> float:
    _finite(value, "norm input")
    return float(value.detach().to(torch.float64).square().sum().sqrt().cpu())


def _ref(path: str | Path, sha256: str, role: str, **extra: Any) -> dict[str, Any]:
    return {"path": Path(path).as_posix(), "sha256": sha256, "role": role, **extra}


def _load_definition_module(root: Path) -> Any:
    path = _path(root, DEFINITION_PATH)
    _require(path.is_file() and file_sha256(path) == DEFINITION_SHA256, "V4 definition entrypoint hash changed")
    spec = importlib.util.spec_from_file_location("lightgen_v4_contract_definition", path)
    _require(spec is not None and spec.loader is not None, "V4 definition module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_contract(root: Path) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    path = _path(root, CONTRACT_PATH)
    _require(path.is_file() and file_sha256(path) == CONTRACT_SHA256, "V4 contract is unavailable or hash-mismatched")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "V4 contract must be a JSON object")
    _require(payload.get("schema") == "lightgen_2d_fixed_ols_generator_recoverability_contract.v4", "V4 contract schema changed")
    _require(payload.get("version") == 4 and payload.get("status") == "valid_completed" and payload.get("result") == "READY", "V4 contract is not READY")
    _require(payload.get("task_id") == "DEFINE_LIGHTGEN_2D_FIXED_OLS_GENERATOR_RECOVERABILITY_CONTRACT_V4", "V4 contract task identity changed")
    _require(payload.get("outer_gate") == "FIXED_OLS_GENERATOR_RECOVERABILITY_SUPPORTED", "V4 outer gate changed")
    _require(payload.get("governance", {}).get("training_authorized") is False, "V4 contract opened scientific training")
    definition_ref = _mapping(payload.get("definition_entrypoint"), "V4 definition entrypoint")
    _require(definition_ref.get("path") == DEFINITION_PATH.as_posix() and definition_ref.get("sha256") == DEFINITION_SHA256, "V4 definition identity changed")
    module = _load_definition_module(root)
    current_authorities = module.validate_v4_authorities(root)
    return payload, module, current_authorities


def _load_v3_control_state(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = _path(root, V3_CONTROL_CHECKPOINT_PATH)
    _require(path.is_file() and file_sha256(path) == V3_CONTROL_CHECKPOINT_SHA256, "V3 CONTROL checkpoint unavailable or hash-mismatched")
    payload = _mapping(safe_torch_load(path, expected_sha256=V3_CONTROL_CHECKPOINT_SHA256, map_location="cpu"), "V3 CONTROL checkpoint")
    _require(payload.get("checkpoint_step") == 300 and payload.get("arm") == "control", "V3 CONTROL endpoint changed")
    state = _mapping(payload.get("runtime_state"), "V3 CONTROL runtime state")
    _require(set(state) == {"encoder.phase", "generator.phase", "ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport"}, "V3 CONTROL state keys changed")
    _require(tensor_sha256(state["encoder.phase"]) == "61648ce609ce4e62f890112faf3f3f91ffa5c018e630ec2d451a3d19c11ed9ca", "V3 CONTROL encoder tensor changed")
    _require(tensor_sha256(state["ols.operator.transport"]) == "fae1f7bb57af089f9eb6b3d3f4996a290ad62ec129195f4b9b507e08affc5293", "V3 CONTROL OLS transport changed")
    _require(torch.equal(state["ols.operator.transport"], OLS_TRANSPORT), "V3 CONTROL transport is not T=I_100")
    return dict(payload), {key: value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in state.items()}


def _cuda_device_facts() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise PreflightBlocked(
            "BLOCKED_V4_PREFLIGHT_TARGET_DEVICE_UNAVAILABLE",
            "device",
            "required CUDA target is unavailable",
            cuda_available=bool(torch.cuda.is_available()),
            device_count=int(torch.cuda.device_count()),
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    try:
        complex_probe = torch.tensor([1.0 + 1.0j], dtype=torch.complex64, device=device)
        probes = {
            "complex_exp": torch.exp(complex_probe),
            "fft2_ifft2": torch.fft.ifft2(torch.fft.fft2(complex_probe.reshape(1, 1, 1, 1))),
            "complex_polar": torch.polar(torch.ones(1, device=device), torch.zeros(1, device=device)),
        }
        torch.cuda.synchronize(device)
    except Exception as error:
        raise PreflightBlocked("BLOCKED_V4_PREFLIGHT_CUDA_BACKEND_UNAVAILABLE", "cuda_backend", "CUDA complex/FFT probe failed", error_type=type(error).__name__, error=str(error)) from error
    backend = {name: bool(torch.isfinite(value).all()) for name, value in probes.items()}
    if not all(backend.values()):
        raise PreflightBlocked("BLOCKED_V4_PREFLIGHT_CUDA_BACKEND_UNAVAILABLE", "cuda_backend", "CUDA backend probe was nonfinite", probe=backend)
    smi = shutil.which("nvidia-smi")
    smi_facts: dict[str, Any] = {"executable": smi}
    if smi:
        try:
            query = subprocess.run([smi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True).stdout.strip()
            fields = [field.strip() for field in query.split(",")]
            smi_facts["query"] = {"gpu_name": fields[0] if fields else None, "driver_version": fields[1] if len(fields) > 1 else None, "memory_total_mib": int(fields[2]) if len(fields) > 2 and fields[2].isdigit() else None}
        except (OSError, subprocess.CalledProcessError) as error:
            smi_facts["query_error"] = f"{type(error).__name__}: {error}"
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
        "backend": backend,
        "nvidia_smi": smi_facts,
        "environment_prefix": sys.prefix,
    }


def _load_train_role(root: Path) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    source = _path(root, TRAINING_SOURCE_PATH)
    _require(source.is_file() and file_sha256(source) == TRAINING_SOURCE_SHA256, "canonical TRAIN source changed")
    images, labels = load_local_mnist(root / "data", train=True)
    _require(tuple(images.shape) == (60_000, 1, 28, 28) and images.dtype == torch.float32, "TRAIN image source shape/dtype changed")
    _require(tuple(labels.shape) == (60_000,) and labels.dtype == torch.long, "TRAIN label source shape/dtype changed")
    train_indices, development_indices = stratified_mnist_indices(labels, seed=SCHEDULE_SEED, validation_per_label=500)
    _require(train_indices.numel() == 55_000 and development_indices.numel() == 5_000, "TRAIN/DEV role counts changed")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "TRAIN/DEV split overlaps or omits rows")
    counts = torch.bincount(labels.index_select(0, train_indices), minlength=10)
    return images, labels, train_indices, {
        "source": {"path": TRAINING_SOURCE_PATH.as_posix(), "sha256": TRAINING_SOURCE_SHA256, "rows_loaded": 60_000},
        "train": {"contract_rows": 55_000, "rows_consumed": 30, "counts_by_class": [int(value) for value in counts.tolist()], "indices_sha256": tensor_sha256(train_indices)},
        "development": {"contract_rows": 5_000, "values_consumed": 0, "status": "not consumed by engineering preflight"},
        "protected": {"rows_read": 0, "status": "forbidden; path not opened"},
        "test": {"rows_read": 0, "status": "forbidden; path not opened"},
        "sealed": {"rows_read": 0, "status": "forbidden and unavailable"},
        "labels_as_model_input": False,
    }


def _make_schedule(train_indices: torch.Tensor, labels: torch.Tensor) -> tuple[dict[str, Any], list[tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    sampler = DeterministicTrainRoleSampler(train_indices, labels, seed=SCHEDULE_SEED, steps=SCIENTIFIC_STEPS)
    initial = _cpu_tree(sampler.state_dict())
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(PREFLIGHT_STEPS):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(batch_labels, torch.arange(10, dtype=torch.long)), "TRAIN sampler labels changed")
        batches.append((indices.cpu().clone(), batch_labels.cpu().clone()))
    after = _cpu_tree(sampler.state_dict())
    return initial, batches, {
        "seed": SCHEDULE_SEED,
        "future_steps": SCIENTIFIC_STEPS,
        "preflight_steps": PREFLIGHT_STEPS,
        "initial_state_sha256": _digest_tree(initial),
        "after_preflight_state_sha256": _digest_tree(after),
        "batch_indices_sha256": [tensor_sha256(indices) for indices, _ in batches],
        "batch_label_sha256": [tensor_sha256(batch_labels) for _, batch_labels in batches],
        "unique_rows_consumed": int(torch.unique(torch.cat([indices for indices, _ in batches])).numel()),
    }


def _load_v2_consumer(root: Path, device: torch.device) -> LightGen2DComplexOLSTrainingConsumerV2:
    contract_payload = json.loads(_path(root, Path("outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json")).read_text(encoding="utf-8"))
    _require(isinstance(contract_payload, dict), "corrected V2 contract is malformed")
    v2_contract = load_v2_training_contract(_path(root, Path("outputs/20260910_v2/runs/lightgen_2d_semantic_complex_ols_v2_training_source_authority_correction_v1_attempt_01/lightgen_2d_semantic_complex_ols_training_contract.v2_corrected.v1.json")), repo_root=root)
    _require(v2_contract.sha256 == "aaa3cc409a9a4775be4bf48ef566fce8615f75b72277ca79194d36bd1f56c9cf", "corrected V2 contract hash changed")
    return LightGen2DComplexOLSTrainingConsumerV2(v2_contract, device=device, propagation_grid_size=PRODUCTION_GRID)


def _runtime_state(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> dict[str, Any]:
    return _cpu_tree(consumer.runtime.state_dict())


def _configure_fixed_generator_consumer(consumer: LightGen2DComplexOLSTrainingConsumerV2, state: Mapping[str, Any]) -> None:
    consumer.runtime.load_state_dict(dict(state), strict=True)
    named = dict(consumer.runtime.named_parameters())
    _require(set(named) == {"encoder.phase", "generator.phase"}, "runtime parameter surface changed")
    for name, parameter in named.items():
        parameter.requires_grad_(name == "generator.phase")
        parameter.grad = None
    _require(named["generator.phase"].requires_grad and not named["encoder.phase"].requires_grad, "generator-only firewall is inactive")
    _require(torch.equal(consumer.runtime.ols.transport.detach().cpu(), OLS_TRANSPORT), "fixed OLS transport is not T=I_100")
    consumer.runtime.train()


def _create_generator_optimizer(consumer: LightGen2DComplexOLSTrainingConsumerV2) -> torch.optim.Adam:
    generator = dict(consumer.runtime.named_parameters()).get("generator.phase")
    _require(generator is not None and generator.requires_grad, "generator.phase is not the sole trainable parameter")
    optimizer = torch.optim.Adam(
        [generator],
        lr=LEARNING_RATE,
        betas=ADAM_BETAS,
        eps=ADAM_EPSILON,
        weight_decay=WEIGHT_DECAY,
        amsgrad=AMSGRAD,
    )
    _require(len(optimizer.param_groups) == 1 and optimizer.param_groups[0]["params"] == [generator], "generator-only optimizer parameter identity changed")
    _require(optimizer.defaults["lr"] == LEARNING_RATE and tuple(optimizer.defaults["betas"]) == ADAM_BETAS and optimizer.defaults["eps"] == ADAM_EPSILON and optimizer.defaults["weight_decay"] == WEIGHT_DECAY and optimizer.defaults["amsgrad"] is AMSGRAD, "generator-only Adam configuration changed")
    return optimizer


def _check_canonical_result(result: Any, consumer: LightGen2DComplexOLSTrainingConsumerV2, batch_size: int) -> None:
    _require(result.image_input is not None and result.ols_input_field is not None and result.z_ols.source == "real_encoded", "canonical frozen-encoder path did not expose real OLS boundaries")
    _require(tuple(result.z_ols.coefficients.shape) == (batch_size, 100) and result.z_ols.coefficients.dtype == torch.complex64, "canonical z shape/dtype changed")
    _require(result.ols_input_field.provenance.parameters.get("modal_projection_boundary") is True, "mandatory modal projection boundary is inactive")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "shared W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "T=I_100 provenance changed")
    _require(torch.equal(consumer.runtime.ols.transport.detach().cpu(), OLS_TRANSPORT), "runtime OLS transport changed")
    _require(tuple(result.raw_intensity.shape) == (batch_size, 1, 200, 200) and result.raw_intensity.dtype == torch.float32, "canonical raw CMOS shape/dtype changed")
    _finite(result.raw_intensity, "canonical raw CMOS")
    _finite(result.image_input.slm_intensity, "canonical SLM target")


def _raw_summary(raw: torch.Tensor) -> dict[str, Any]:
    detached = raw.detach()
    per_image = detached.flatten(1)
    variance = per_image.var(dim=1, unbiased=False)
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "finite": bool(torch.isfinite(detached).all()),
        "mean": float(detached.mean().cpu()),
        "std": float(detached.std(unbiased=False).cpu()),
        "per_image_variance_min": float(variance.min().cpu()),
        "per_image_variance_max": float(variance.max().cpu()),
        "structured": bool((variance > 0.0).all()),
        "nonconstant": bool(float(detached.std(unbiased=False).cpu()) > 0.0),
        "tensor_sha256": tensor_sha256(detached.cpu()),
    }


def _materialize_fixed_z_cache(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    images: torch.Tensor,
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache: list[dict[str, Any]] = []
    z_differences: list[float] = []
    raw_differences: list[float] = []
    canonical_records: list[dict[str, Any]] = []
    consumer.runtime.eval()
    with torch.no_grad():
        for batch_number, (indices, batch_labels) in enumerate(batches, start=1):
            batch = images.index_select(0, indices).to(device, non_blocking=False)
            canonical = consumer.runtime.forward_from_image(batch)
            _check_canonical_result(canonical, consumer, LOGICAL_BATCH_SIZE)
            z = canonical.z_ols.coefficients.detach()
            target = canonical.image_input.slm_intensity.detach()
            replay = consumer.runtime.forward_from_ols(z)
            _require(replay.z_ols.source == "external_generated", "fixed-z replay source identity changed")
            z_difference = float((replay.z_ols.coefficients - z).abs().max().cpu())
            raw_difference = float((replay.raw_intensity - canonical.raw_intensity).abs().max().cpu())
            _require(z_difference <= DETERMINISTIC_ATOL, f"canonical/replay z mismatch at preflight batch {batch_number}")
            _require(torch.allclose(replay.raw_intensity, canonical.raw_intensity, rtol=DETERMINISTIC_RTOL, atol=DETERMINISTIC_ATOL), f"canonical/replay raw mismatch at preflight batch {batch_number}")
            z_cpu = z.cpu().clone()
            target_cpu = target.cpu().clone()
            cache.append({"indices": indices.clone(), "labels": batch_labels.clone(), "z": z_cpu, "target": target_cpu})
            z_differences.append(z_difference)
            raw_differences.append(raw_difference)
            canonical_records.append({"batch": batch_number, "indices_sha256": tensor_sha256(indices), "labels_sha256": tensor_sha256(batch_labels), "z_sha256": tensor_sha256(z_cpu), "target_sha256": tensor_sha256(target_cpu), "raw": _raw_summary(canonical.raw_intensity.cpu())})
    consumer.runtime.train()
    return cache, {
        "canonical_path": "image -> frozen encoder -> modal projection -> z_real -> T=I_100 -> W_OLS",
        "replay_path": "cached z_real -> T=I_100 -> W_OLS -> same generator",
        "z_equivalent": True,
        "max_abs_z_difference": max(z_differences, default=0.0),
        "max_abs_canonical_replay_raw_difference": max(raw_differences, default=0.0),
        "tolerance": {"rtol": DETERMINISTIC_RTOL, "atol": DETERMINISTIC_ATOL},
        "batches": canonical_records,
    }


def _run_update(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    optimizer: torch.optim.Adam,
    cached: Mapping[str, Any],
    step: int,
    device: torch.device,
) -> tuple[dict[str, Any], torch.Tensor]:
    _require(step in {1, 2, 3}, "preflight step escaped the bounded three-update budget")
    consumer.runtime.train()
    named = dict(consumer.runtime.named_parameters())
    for parameter in named.values():
        parameter.grad = None
    before_state = _runtime_state(consumer)
    before_encoder = before_state["encoder.phase"]
    before_ols = {key: before_state[key] for key in ("ols.operator.input_modes", "ols.operator.output_modes", "ols.operator.transport")}
    before_generator = before_state["generator.phase"]
    z = cached["z"].to(device)
    target = cached["target"].to(device)
    optimizer.zero_grad(set_to_none=True)
    result = consumer.runtime.forward_from_ols(z)
    _require(result.image_input is None and result.ols_input_field is None and result.z_ols.source == "external_generated", "generator-only path crossed back into an image/encoder path")
    _require(result.generator_input_field.provenance.operation == "lightgen_ols_coefficient_transmission_and_modal_reemission", "generator-only W_OLS provenance changed")
    _require(result.generator_input_field.provenance.parameters.get("transport") == "T=I_100_coefficient_space", "generator-only T=I_100 provenance changed")
    _require(tuple(result.raw_intensity.shape) == (LOGICAL_BATCH_SIZE, 1, 200, 200) and result.raw_intensity.dtype == torch.float32, "generator-only raw CMOS shape/dtype changed")
    _finite(result.raw_intensity, "generator-only raw CMOS")
    _finite(target, "cached SLM target")
    loss = (result.raw_intensity - target).square().mean()
    _finite(loss, "generator image loss")
    loss.backward()
    generator_gradient = named["generator.phase"].grad
    encoder_gradient = named["encoder.phase"].grad
    _require(generator_gradient is not None, "generator.phase did not receive an image gradient")
    _finite(generator_gradient, "generator.phase gradient")
    _require(float(generator_gradient.abs().sum().cpu()) > 0.0, "generator.phase gradient is identically zero")
    _require(encoder_gradient is None, "encoder.phase received a gradient in the negative control")
    _require(all(parameter.grad is None for name, parameter in named.items() if name != "generator.phase"), "a frozen runtime parameter received a gradient")
    generator_gradient_norm = _norm(generator_gradient)
    optimizer.step()
    after_state = _runtime_state(consumer)
    after_generator = after_state["generator.phase"]
    generator_update_norm = float((after_generator - before_generator).to(torch.float64).square().sum().sqrt())
    _require(generator_update_norm > 0.0, "generator-only optimizer step did not update generator.phase")
    _require(torch.equal(before_encoder, after_state["encoder.phase"]), "encoder.phase changed under generator-only optimizer")
    _require(all(torch.equal(before_ols[key], after_state[key]) for key in before_ols), "frozen OLS state changed under generator-only optimizer")
    _require(torch.isfinite(after_generator).all(), "generator.phase became nonfinite")
    raw_cpu = result.raw_intensity.detach().cpu().clone()
    raw_summary = _raw_summary(raw_cpu)
    _require(raw_summary["finite"] and raw_summary["structured"] and raw_summary["nonconstant"], "generator-only raw CMOS is invalid or constant")
    return {
        "step": step,
        "objective": "L_generator=L_image",
        "batch_indices_sha256": tensor_sha256(cached["indices"]),
        "batch_labels_sha256": tensor_sha256(cached["labels"]),
        "z_sha256": tensor_sha256(cached["z"]),
        "target_sha256": tensor_sha256(cached["target"]),
        "loss_image": float(loss.detach().cpu()),
        "gradient_norm_generator_phase": generator_gradient_norm,
        "gradient_encoder_phase": None,
        "generator_update_norm": generator_update_norm,
        "encoder_unchanged": True,
        "ols_unchanged": True,
        "trainable_parameter_names": ["generator.phase"],
        "frozen_parameter_names": ["encoder.phase"],
        "w_ols_active": True,
        "transport": "T=I_100_coefficient_space",
        "raw_cmos": raw_summary,
        "generator_phase_before_sha256": tensor_sha256(before_generator),
        "generator_phase_after_sha256": tensor_sha256(after_generator),
    }, raw_cpu


def _capture_rng_state() -> dict[str, Any]:
    return {"cpu": torch.random.get_rng_state().cpu().clone(), "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]}


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    _require(torch.is_tensor(state.get("cpu")) and isinstance(state.get("cuda"), list), "checkpoint RNG state is malformed")
    torch.random.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def _checkpoint_payload(
    consumer: LightGen2DComplexOLSTrainingConsumerV2,
    optimizer: torch.optim.Adam,
    schedule: Mapping[str, Any],
    cache: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        "checkpoint_step": CHECKPOINT_STEP,
        "next_step": CHECKPOINT_STEP + 1,
        "scientific_training_checkpoint": False,
        "training_authorized": False,
        "stage_b_executed": False,
        "generation_executed": False,
        "formal_promotion": False,
        "runtime_state": _runtime_state(consumer),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "rng_state": _capture_rng_state(),
        "schedule": {"seed": SCHEDULE_SEED, "future_steps": SCIENTIFIC_STEPS, "next_batch_indices_sha256": tensor_sha256(cache[CHECKPOINT_STEP]["indices"]), "next_batch_labels_sha256": tensor_sha256(cache[CHECKPOINT_STEP]["labels"])},
        "trainable_parameter_names": ["generator.phase"],
    }


def _load_evaluators(root: Path, authorities: Mapping[str, Any], device: torch.device) -> tuple[FrozenMNISTEvaluator, SmoothMNISTEvaluator, dict[str, Any]]:
    eval_authorities = _mapping(authorities.get("evaluators"), "V4 evaluator authorities")
    frozen_ref = _mapping(eval_authorities.get("frozen_checkpoint"), "Frozen evaluator ref")
    smooth_ref = _mapping(eval_authorities.get("smooth_checkpoint"), "Smooth evaluator ref")
    _require(frozen_ref.get("sha256") == FROZEN_CHECKPOINT_SHA256 and smooth_ref.get("sha256") == SMOOTH_CHECKPOINT_SHA256, "evaluator authority hash changed")
    frozen_payload = _mapping(safe_torch_load(_path(root, frozen_ref["path"]), expected_sha256=frozen_ref["sha256"], map_location="cpu"), "Frozen evaluator payload")
    _require(frozen_payload.get("architecture_kind") == "frozen_mnist_evaluator_v1" and float(frozen_payload.get("validation_accuracy", 0.0)) >= 0.985, "Frozen evaluator qualification changed")
    frozen = FrozenMNISTEvaluator()
    frozen.load_state_dict(_mapping(frozen_payload.get("state_dict"), "Frozen evaluator state"), strict=True)
    frozen.to(device).freeze()
    smooth_payload = _mapping(safe_torch_load(_path(root, smooth_ref["path"]), expected_sha256=smooth_ref["sha256"], map_location="cpu"), "Smooth evaluator payload")
    _require(smooth_payload.get("architecture_kind") == "smooth_mnist_evaluator_v2", "Smooth evaluator architecture changed")
    smooth = SmoothMNISTEvaluator()
    smooth.load_state_dict(_mapping(smooth_payload.get("state_dict"), "Smooth evaluator state"), strict=True)
    smooth.to(device).freeze()
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in frozen.parameters()), "Frozen evaluator gradient firewall failed")
    _require(all(not parameter.requires_grad and parameter.grad is None for parameter in smooth.parameters()), "Smooth evaluator gradient firewall failed")
    return frozen, smooth, {"frozen_state_sha256": _digest_tree(_cpu_tree(frozen.state_dict())), "smooth_state_sha256": _digest_tree(_cpu_tree(smooth.state_dict())), "gradient_enabled": False, "training_role": "evaluation-only"}


def _evaluator_smoke(frozen: FrozenMNISTEvaluator, smooth: SmoothMNISTEvaluator, raw: torch.Tensor, before: Mapping[str, Any]) -> dict[str, Any]:
    view = F.adaptive_avg_pool2d(raw.to(next(frozen.parameters()).device).clamp(0.0, 1.0), (28, 28))
    with torch.no_grad():
        frozen_features, frozen_logits = frozen(view)
        smooth_features, smooth_logits = smooth(view)
    _finite(frozen_features, "Frozen evaluator smoke features")
    _finite(frozen_logits, "Frozen evaluator smoke logits")
    _finite(smooth_features, "Smooth evaluator smoke features")
    _finite(smooth_logits, "Smooth evaluator smoke logits")
    after = {"frozen_state_sha256": _digest_tree(_cpu_tree(frozen.state_dict())), "smooth_state_sha256": _digest_tree(_cpu_tree(smooth.state_dict()))}
    _require(after == dict(before)["states"], "semantic evaluator state changed during smoke evaluation")
    _require(all(parameter.grad is None for evaluator in (frozen, smooth) for parameter in evaluator.parameters()), "semantic evaluator received a gradient")
    return {"executed": True, "data_role": "TRAIN smoke output only", "feature_shapes": {"frozen": list(frozen_features.shape), "smooth": list(smooth_features.shape)}, "logit_shapes": {"frozen": list(frozen_logits.shape), "smooth": list(smooth_logits.shape)}, "states_unchanged": True, "gradient_none": True}


def _base_input(root: Path, contract_payload: Mapping[str, Any], authorities: Mapping[str, Any], dataset: Mapping[str, Any], schedule: Mapping[str, Any], device_facts: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": INPUT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
        "entrypoint": {"path": "scripts/research/lightgen/execute_lightgen_2d_fixed_ols_generator_recoverability_v4.py", "sha256": file_sha256(Path(__file__))},
        "protocol": {"objective": "L_generator=L_image", "trainable": ["generator.phase"], "frozen": ["encoder.phase", "OLS", "evaluators"], "grid": PRODUCTION_GRID, "logical_batch_size": LOGICAL_BATCH_SIZE, "preflight_updates": PREFLIGHT_STEPS, "future_scientific_updates": SCIENTIFIC_STEPS, "checkpoint_step": CHECKPOINT_STEP, "seed": SCHEDULE_SEED, "rtol": DETERMINISTIC_RTOL, "atol": DETERMINISTIC_ATOL},
        "dataset": dataset,
        "schedule": schedule,
        "device": device_facts,
        "data_isolation": {"TRAIN_updates": True, "DEV_evaluation": False, "protected_read": False, "test_read": False, "sealed_read": False},
        "claim_boundary": "engineering preflight only; no scientific gate, Stage-B authorization, generation readiness, or formal promotion",
        "input_self_hash": "not_embedded_to_avoid_self_reference",
    }


def _write_report(root: Path, output: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    report_path = output / REPORT_NAME
    atomic_write_new_json(report_path, payload, artifact_kind="V4 generator preflight report")
    return {"path": report_path.relative_to(root).as_posix(), "sha256": file_sha256(report_path)}


def run_preflight(repo_root: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    output = (output_dir or (root / DEFAULT_OUTPUT_DIR)).resolve()
    started = datetime.now(timezone.utc).isoformat()
    output.mkdir(parents=True, exist_ok=True)
    contract_payload: dict[str, Any] | None = None
    authorities: dict[str, Any] | None = None
    input_path = output / INPUT_NAME
    try:
        contract_payload, definition_module, current_authorities = _load_contract(root)
        authorities = _mapping(contract_payload.get("authorities"), "V4 contract authorities")
        _require(authorities.get("aligned_runtime", {}).get("sha256") == ALIGNED_RUNTIME_SHA256, "aligned runtime authority changed")
        _require(current_authorities.get("aligned_runtime", {}).get("sha256") == ALIGNED_RUNTIME_SHA256, "current aligned runtime authority changed")
        _require(git_facts(root).get("branch") == EXPECTED_BRANCH and git_facts(root).get("head") == EXPECTED_HEAD, "branch or HEAD changed")
        device, device_facts = _cuda_device_facts()
        images, labels, train_indices, dataset = _load_train_role(root)
        sampler_initial, batches, schedule = _make_schedule(train_indices, labels)
        input_payload = _base_input(root, contract_payload, authorities, dataset, schedule, device_facts)
        atomic_write_new_json(input_path, input_payload, artifact_kind="V4 generator preflight input")

        _, control_state = _load_v3_control_state(root)
        initial_consumer = _load_v2_consumer(root, device)
        fresh_state = _runtime_state(initial_consumer)
        _require(_digest_tree(fresh_state) == INITIAL_RUNTIME_STATE_SHA256, "fresh CUDA runtime does not reproduce original zero-phase state")
        _require(torch.equal(fresh_state["generator.phase"], torch.zeros(PHASE_SHAPE, dtype=PHASE_DTYPE)), "fresh CUDA generator is not zero phase")
        fixed_state = dict(control_state)
        fixed_state["generator.phase"] = fresh_state["generator.phase"].clone()
        fixed_initial_state_sha256 = _digest_tree(fixed_state)
        _configure_fixed_generator_consumer(initial_consumer, fixed_state)
        parameter_manifest = {
            "trainable_names": [name for name, parameter in initial_consumer.runtime.named_parameters() if parameter.requires_grad],
            "frozen_names": [name for name, parameter in initial_consumer.runtime.named_parameters() if not parameter.requires_grad],
            "trainable_count": sum(parameter.numel() for name, parameter in initial_consumer.runtime.named_parameters() if parameter.requires_grad),
            "encoder_frozen_count": sum(parameter.numel() for name, parameter in initial_consumer.runtime.named_parameters() if name == "encoder.phase"),
            "ols_frozen_state_numel": sum(value.numel() for name, value in initial_consumer.runtime.state_dict().items() if name.startswith("ols.")),
        }
        _require(parameter_manifest["trainable_names"] == ["generator.phase"] and parameter_manifest["frozen_names"] == ["encoder.phase"], "generator-only parameter firewall changed")
        _require(parameter_manifest["trainable_count"] == 20_000 and parameter_manifest["encoder_frozen_count"] == 20_000 and parameter_manifest["ols_frozen_state_numel"] == 2_010_000, "generator-only parameter counts changed")
        z_cache, z_equivalence = _materialize_fixed_z_cache(initial_consumer, images, batches, device)
        del initial_consumer
        torch.cuda.empty_cache()

        full_consumer = _load_v2_consumer(root, device)
        _configure_fixed_generator_consumer(full_consumer, fixed_state)
        full_optimizer = _create_generator_optimizer(full_consumer)
        full_traces: list[dict[str, Any]] = []
        full_raw = None
        for step in range(1, PREFLIGHT_STEPS + 1):
            trace, full_raw = _run_update(full_consumer, full_optimizer, z_cache[step - 1], step, device)
            full_traces.append(trace)
        full_final_state = _runtime_state(full_consumer)
        full_optimizer_state = _cpu_tree(full_optimizer.state_dict())
        del full_optimizer
        del full_consumer
        torch.cuda.empty_cache()

        partial_consumer = _load_v2_consumer(root, device)
        _configure_fixed_generator_consumer(partial_consumer, fixed_state)
        partial_optimizer = _create_generator_optimizer(partial_consumer)
        partial_traces: list[dict[str, Any]] = []
        for step in range(1, CHECKPOINT_STEP + 1):
            trace, _ = _run_update(partial_consumer, partial_optimizer, z_cache[step - 1], step, device)
            partial_traces.append(trace)
        checkpoint_payload = _checkpoint_payload(partial_consumer, partial_optimizer, schedule, z_cache)
        checkpoint_path = output / CHECKPOINT_NAME
        atomic_torch_save_new(checkpoint_path, checkpoint_payload, artifact_kind="V4 generator preflight checkpoint")
        checkpoint_sha256 = file_sha256(checkpoint_path)
        loaded_checkpoint = _mapping(safe_torch_load(checkpoint_path, expected_sha256=checkpoint_sha256, map_location="cpu"), "V4 preflight checkpoint")
        _require(loaded_checkpoint.get("schema") == CHECKPOINT_SCHEMA and loaded_checkpoint.get("checkpoint_step") == CHECKPOINT_STEP and loaded_checkpoint.get("next_step") == 3, "V4 checkpoint identity changed")
        del partial_optimizer
        del partial_consumer
        torch.cuda.empty_cache()

        resumed_consumer = _load_v2_consumer(root, device)
        _configure_fixed_generator_consumer(resumed_consumer, _mapping(loaded_checkpoint.get("runtime_state"), "checkpoint runtime state"))
        resumed_optimizer = _create_generator_optimizer(resumed_consumer)
        resumed_optimizer.load_state_dict(_mapping(loaded_checkpoint.get("optimizer_state"), "checkpoint optimizer state"))
        _restore_rng_state(_mapping(loaded_checkpoint.get("rng_state"), "checkpoint RNG state"))
        resumed_trace, resumed_raw = _run_update(resumed_consumer, resumed_optimizer, z_cache[2], 3, device)
        resumed_final_state = _runtime_state(resumed_consumer)
        full_step3 = full_traces[2]
        resume_identity = {
            "checkpoint_saved_and_reloaded": True,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_step": CHECKPOINT_STEP,
            "next_step": 3,
            "step_3_loss_difference": abs(float(full_step3["loss_image"]) - float(resumed_trace["loss_image"])),
            "step_3_generator_gradient_difference": abs(float(full_step3["gradient_norm_generator_phase"]) - float(resumed_trace["gradient_norm_generator_phase"])),
            "step_3_update_difference": abs(float(full_step3["generator_update_norm"]) - float(resumed_trace["generator_update_norm"])),
            "raw_step_3_max_abs_difference": float((full_raw - resumed_raw).abs().max()),
            "runtime_state_max_abs_difference": _max_difference(full_final_state, resumed_final_state),
            "optimizer_state_max_difference": _max_difference(full_optimizer_state, _cpu_tree(resumed_optimizer.state_dict())),
            "within_tolerance": True,
        }
        # The full optimizer was intentionally released after its run; compare
        # the resumed result against the full trace/state within the declared
        # tolerance rather than inventing a second optimizer identity.
        _require(resume_identity["step_3_loss_difference"] <= DETERMINISTIC_ATOL, "checkpoint/resume step-3 loss is not reproducible")
        _require(resume_identity["step_3_generator_gradient_difference"] <= DETERMINISTIC_ATOL, "checkpoint/resume gradient is not reproducible")
        _require(resume_identity["step_3_update_difference"] <= DETERMINISTIC_ATOL, "checkpoint/resume update is not reproducible")
        _require(resume_identity["raw_step_3_max_abs_difference"] <= DETERMINISTIC_ATOL, "checkpoint/resume raw output is not reproducible")
        _require(resume_identity["runtime_state_max_abs_difference"] <= DETERMINISTIC_ATOL, "checkpoint/resume runtime state is not reproducible")

        frozen, smooth, evaluator_states = _load_evaluators(root, authorities, device)
        provider_smoke = _evaluator_smoke(frozen, smooth, resumed_raw, {"states": {"frozen_state_sha256": evaluator_states["frozen_state_sha256"], "smooth_state_sha256": evaluator_states["smooth_state_sha256"]}})
        del resumed_optimizer
        del resumed_consumer
        del frozen
        del smooth
        torch.cuda.empty_cache()

        report_payload: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "status": "valid_completed",
            "result": "AUTHORIZABLE",
            "classification": SUCCESS_CLASSIFICATION,
            "outer_gate": "FIXED_OLS_GENERATOR_RECOVERABILITY_SUPPORTED",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "started_at_utc": started,
            "git": {"contract": contract_payload.get("git"), "start": git_facts(root), "end": git_facts(root), "worktree_intentionally_dirty": True},
            "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256},
            "entrypoint": {"path": "scripts/research/lightgen/execute_lightgen_2d_fixed_ols_generator_recoverability_v4.py", "sha256": file_sha256(Path(__file__))},
            "authorities": {"v4_contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "v3_control_checkpoint": {"path": V3_CONTROL_CHECKPOINT_PATH.as_posix(), "sha256": V3_CONTROL_CHECKPOINT_SHA256, "step": 300}, "aligned_runtime": {"path": ALIGNED_RUNTIME_PATH.as_posix(), "sha256": ALIGNED_RUNTIME_SHA256}, "v2_contract": authorities.get("v2_contract"), "evaluators": authorities.get("evaluators")},
            "device": device_facts,
            "data": dataset,
            "schedule": {"initial_state_sha256": _digest_tree(sampler_initial), **schedule, "batches": [{"indices_sha256": tensor_sha256(item["indices"]), "labels_sha256": tensor_sha256(item["labels"])} for item in z_cache]},
            "initialization": {"v3_control_encoder_phase_sha256": tensor_sha256(fixed_state["encoder.phase"]), "original_zero_generator_phase_sha256": tensor_sha256(fixed_state["generator.phase"]), "fixed_initial_runtime_state_sha256": fixed_initial_state_sha256, "fresh_runtime_state_sha256": INITIAL_RUNTIME_STATE_SHA256, "control_generator_not_reused": True},
            "parameter_firewall": parameter_manifest,
            "objective": {"formula": "L_generator=L_image", "semantic_provider_in_optimization": False, "semantic_loss_weight": None, "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "betas": list(ADAM_BETAS), "eps": ADAM_EPSILON, "weight_decay": WEIGHT_DECAY, "amsgrad": AMSGRAD, "parameter_names": ["generator.phase"]}},
            "z_equivalence": z_equivalence,
            "traces": full_traces,
            "checkpoint_resume": resume_identity,
            "semantic_provider_firewall": {"authorities": evaluator_states, "smoke": provider_smoke},
            "governance": {"engineering_preflight_executed": True, "scientific_training_executed": False, "scientific_updates_executed": 0, "stage_b_authorized": False, "stage_b_executed": False, "scale_extraction_executed": False, "prior_generation_executed": False, "generation_executed": False, "protected_read": False, "test_read": False, "sealed_read": False, "formal_promotion": False, "training_authorized": False, "generation_readiness": "not_ready"},
            "claim_boundary": "V4 engineering preflight is AUTHORIZABLE for a future bounded scientific run; no scientific run or downstream stage was authorized or executed.",
        }
        report_ref = _write_report(root, output, report_payload)
        lineage_payload = {"schema": LINEAGE_SCHEMA, "version": 1, "task_id": TASK_ID, "status": report_payload["status"], "result": report_payload["result"], "classification": report_payload["classification"], "outer_gate": report_payload["outer_gate"], "contract": report_payload["contract"], "entrypoint": report_payload["entrypoint"], "authorities": report_payload["authorities"], "artifacts": {"input": {"path": input_path.relative_to(root).as_posix(), "sha256": file_sha256(input_path)}, "checkpoint": {"path": checkpoint_path.relative_to(root).as_posix(), "sha256": checkpoint_sha256}, "report": report_ref}, "governance": report_payload["governance"], "claim_boundary": report_payload["claim_boundary"], "lineage_self_hash": "not_embedded_to_avoid_self_reference"}
        lineage_path = output / LINEAGE_NAME
        atomic_write_new_json(lineage_path, lineage_payload, artifact_kind="V4 generator preflight lineage")
        report_payload["artifacts"] = {"input": {"path": input_path.relative_to(root).as_posix(), "sha256": file_sha256(input_path)}, "checkpoint": {"path": checkpoint_path.relative_to(root).as_posix(), "sha256": checkpoint_sha256}, "report": report_ref, "lineage": {"path": lineage_path.relative_to(root).as_posix(), "sha256": file_sha256(lineage_path)}}
        return report_payload
    except PreflightBlocked as error:
        blocked = {"schema": REPORT_SCHEMA, "version": 1, "task_id": TASK_ID, "status": "blocked", "result": "BLOCKED", "classification": error.classification, "outer_gate": "FIXED_OLS_GENERATOR_RECOVERABILITY_SUPPORTED", "started_at_utc": started, "stage": error.stage, "error": str(error), "details": error.details, "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "governance": {"engineering_preflight_executed": False, "scientific_training_executed": False, "stage_b_authorized": False, "stage_b_executed": False, "generation_executed": False, "protected_read": False, "test_read": False, "sealed_read": False, "formal_promotion": False, "training_authorized": False}}
        if not (output / REPORT_NAME).exists():
            _write_report(root, output, blocked)
        return blocked
    except Exception as error:
        failed = {"schema": REPORT_SCHEMA, "version": 1, "task_id": TASK_ID, "status": "valid_failed", "result": "INVALID_HARNESS", "classification": "V4_PREFLIGHT_INVALID_HARNESS", "outer_gate": "FIXED_OLS_GENERATOR_RECOVERABILITY_SUPPORTED", "started_at_utc": started, "error": f"{type(error).__name__}: {error}", "contract": {"path": CONTRACT_PATH.as_posix(), "sha256": CONTRACT_SHA256}, "governance": {"engineering_preflight_executed": False, "scientific_training_executed": False, "stage_b_authorized": False, "stage_b_executed": False, "generation_executed": False, "protected_read": False, "test_read": False, "sealed_read": False, "formal_promotion": False, "training_authorized": False}}
        if not (output / REPORT_NAME).exists():
            _write_report(root, output, failed)
        return failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    payload = run_preflight(args.repo_root, args.output_dir)
    print(json.dumps({key: payload.get(key) for key in ("task_id", "status", "result", "classification", "outer_gate", "error", "artifacts")}, indent=2, sort_keys=True))
    return 0 if payload.get("result") == "AUTHORIZABLE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
