"""Execute the one-shot LightGen 2-D semantic complex-OLS training run.

The frozen training consumer owns the optical path, loss, prior, and parameter
firewall.  This entry point only supplies the explicitly authorized CUDA
schedule, checkpoint cadence, development selection, and provenance.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import json
import math
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training import (
    ALIGNMENT_RUNTIME_RELATIVE_PATH,
    ALIGNMENT_RUNTIME_SHA256,
    ALLOWED_TRAINABLE_PARAMETER_NAMES,
    LOGICAL_BATCH_SIZE,
    PRIOR_NAMESPACE,
    PRIOR_SAMPLE_COUNT,
    PRIOR_SIGMA,
    PRODUCTION_PROPAGATION_GRID,
    SCHEDULE_SEED,
    TRAINING_STEPS,
    TRAINING_CONTRACT_SHA256,
    TrainingContractError,
    BalancedTrainRoleSampler,
    DatasetRoleMetadata,
    FrozenTrainingContract,
    LightGen2DComplexOLSTrainingConsumer,
    PriorSample,
    full_complex_mmd_vstat,
    git_facts,
    load_frozen_training_contract,
    restore_rng_state,
    validate_legacy_exclusion,
)
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import load_local_mnist, stratified_mnist_indices


TASK_ID = "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1"
SUCCESS_CLASSIFICATION = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1_VALID_COMPLETED"
FORMAL_CHECKPOINT_SCHEMA = "lightgen_2d_semantic_complex_ols_training_checkpoint.v1"
REPORT_SCHEMA = "lightgen_2d_semantic_complex_ols_training_report.v1"
LINEAGE_SCHEMA = "lightgen_2d_semantic_complex_ols_training_lineage.v1"
PREFLIGHT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight.v1.json"
)
PREFLIGHT_SHA256 = "365573edb0a3b4525bb571e31241bba6ba4243b9b2a3f591a8d1aea722c48d8a"
PREFLIGHT_CLASSIFICATION = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_AUTHORIZABLE"
PREVIOUS_CONSUMER_PREFLIGHT_SHA256 = "c16dc2b5a1ed1c8a4f0ba0fc2e5a3e90154620e3f053d54d79342cc434e85aa7"
PREVIOUS_CONSUMER_ENTRYPOINT = Path(
    "scripts/research/execute_lightgen_2d_semantic_complex_ols_training_v1.py"
)
PREVIOUS_CONSUMER_ENTRYPOINT_SHA256 = "c3b8d69b1cc387778414d3e53c895639af2318cb7c16ac2e94107aafcc05f0be"
CONSUMER_RELATIVE_PATH = Path(
    "src/ldm_project/v2/application/research/lightgen_2d_semantic_complex_ols_training.py"
)
CONSUMER_SHA256 = "fae7ffcd5f724aa4a07966e0f75dbe77653eaf85cf9907093606ceb884517d1c"
ALIGNMENT_CONTRACT_SHA256 = "6f3859d87e9e3f38d141943ae3fac23607cc446fe2c58b99c9b5b4c379a3a72e"
REPORT_NAME = "lightgen_2d_semantic_complex_ols_training_report.v1.json"
LINEAGE_NAME = "lightgen_2d_semantic_complex_ols_training_lineage.v1.json"
CHECKPOINT_NAMES = {
    50: "lightgen_2d_semantic_complex_ols_training_checkpoint_step_050.v1.pt",
    100: "lightgen_2d_semantic_complex_ols_training_checkpoint_step_100.v1.pt",
}
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_v1_attempt_01"
)
DEVELOPMENT_SELECTION_SEED = 4321
DEVELOPMENT_TARGET_SEED = 20260911
TRAINING_PRIOR_SEED = 20260910
DEVELOPMENT_COUNT = 1000
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 3060 Laptop GPU"
EXPECTED_COMPUTE_CAPABILITY = (8, 6)
EXPECTED_TORCH_VERSION = "2.5.1"
EXPECTED_CUDA_BUILD = "12.1"


class InvalidTrainingExecution(RuntimeError):
    """A malformed authority or invalid execution harness."""


class BlockedTrainingExecution(RuntimeError):
    """A valid hard stop that prevents the bounded run from continuing."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidTrainingExecution(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _resolve_output_dir(root: Path, output_dir: Path | None, resume_checkpoint: Path | None) -> Path:
    if resume_checkpoint is not None:
        checkpoint = resume_checkpoint if resume_checkpoint.is_absolute() else root / resume_checkpoint
        checkpoint = checkpoint.resolve()
        outputs_root = (root / "outputs").resolve()
        _require(checkpoint.is_relative_to(outputs_root), "resume checkpoint must remain under outputs/")
        _require(checkpoint.is_file(), f"resume checkpoint is unavailable: {checkpoint}")
        target = (root / output_dir if output_dir is not None and not output_dir.is_absolute() else output_dir)
        target = checkpoint.parent if target is None else target
        target = target.resolve()
        _require(target == checkpoint.parent, "resume output directory must be the checkpoint parent")
        _require(target.is_relative_to(outputs_root), "resume output must remain under outputs/")
        _require(not (target / REPORT_NAME).exists(), "resume output already has a final training report")
        _require(not (target / LINEAGE_NAME).exists(), "resume output already has a final training lineage")
        return target
    target = root / DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    target = target if target.is_absolute() else root / target
    target = target.resolve()
    outputs_root = (root / "outputs").resolve()
    _require(target.is_relative_to(outputs_root), "training output must remain under outputs/")
    _require(not target.exists(), f"refusing to overwrite training output directory: {target}")
    return target


def _read_hashed_json(root: Path, relative_path: Path, expected_sha256: str, name: str) -> Mapping[str, Any]:
    path = root / relative_path
    _require(path.is_file(), f"{name} is unavailable: {path}")
    actual = file_sha256(path)
    _require(actual == expected_sha256, f"{name} SHA mismatch: {actual}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidTrainingExecution(f"{name} is not valid JSON") from error
    return _mapping(value, name)


def _validate_authorities(root: Path) -> tuple[FrozenTrainingContract, dict[str, Any]]:
    contract = load_frozen_training_contract(repo_root=root)
    _require(contract.sha256 == TRAINING_CONTRACT_SHA256, "training contract identity changed")
    preflight = _read_hashed_json(root, PREFLIGHT_RELATIVE_PATH, PREFLIGHT_SHA256, "long-training preflight")
    _require(preflight.get("status") == "valid_completed", "long-training preflight is not valid_completed")
    _require(preflight.get("result") == "AUTHORIZABLE", "long-training preflight is not AUTHORIZABLE")
    _require(preflight.get("classification") == PREFLIGHT_CLASSIFICATION, "long-training preflight classification changed")
    boundary = _mapping(preflight.get("claim_boundary"), "long-training preflight claim boundary")
    for key in ("long_training_executed", "scientific_training_executed", "heldout_evaluation_executed", "sealed_accessed", "sealed_reused", "formal_promotion"):
        _require(boundary.get(key) is False, f"preflight crossed protected boundary: {key}")
    _require(boundary.get("generation_readiness") == "not_ready", "preflight generation boundary changed")
    _require(boundary.get("training_authorized") is False, "preflight unexpectedly authorized training")
    _require(preflight.get("next") == TASK_ID, "preflight next action is not this training task")

    authorities = _mapping(preflight.get("authorities"), "preflight authorities")
    previous = _mapping(authorities.get("implemented_consumer_preflight"), "implemented consumer preflight authority")
    _require(previous.get("sha256") == PREVIOUS_CONSUMER_PREFLIGHT_SHA256, "implemented consumer preflight identity changed")
    consumer = _mapping(authorities.get("consumer"), "consumer authority")
    consumer_path = root / CONSUMER_RELATIVE_PATH
    entrypoint_path = root / PREVIOUS_CONSUMER_ENTRYPOINT
    _require(consumer_path.is_file(), "training consumer module is unavailable")
    _require(entrypoint_path.is_file(), "implemented consumer entrypoint is unavailable")
    consumer_sha = file_sha256(consumer_path)
    entrypoint_sha = file_sha256(entrypoint_path)
    _require(consumer_sha == CONSUMER_SHA256 == consumer.get("module_sha256"), "training consumer module identity changed")
    _require(entrypoint_sha == PREVIOUS_CONSUMER_ENTRYPOINT_SHA256 == consumer.get("entrypoint_sha256"), "consumer preflight entrypoint identity changed")
    runtime_path = root / ALIGNMENT_RUNTIME_RELATIVE_PATH
    _require(runtime_path.is_file(), "aligned runtime is unavailable")
    runtime_sha = file_sha256(runtime_path)
    _require(runtime_sha == ALIGNMENT_RUNTIME_SHA256, "aligned runtime identity changed")
    _require(preflight.get("gpu_smoke", {}).get("runtime_grid") == PRODUCTION_PROPAGATION_GRID, "preflight production grid changed")
    _require(preflight.get("gpu_smoke", {}).get("three_step_real_update") is True, "three-step GPU preflight evidence is missing")
    _require(preflight.get("gpu_smoke", {}).get("memory", {}).get("feasible") is True, "GPU memory preflight was not feasible")
    _require(validate_legacy_exclusion().get("passed") is True, "legacy exclusion validation failed")

    return contract, {
        "training_contract": contract.identity(root),
        "long_training_preflight": {
            "path": PREFLIGHT_RELATIVE_PATH.as_posix(),
            "sha256": PREFLIGHT_SHA256,
            "classification": PREFLIGHT_CLASSIFICATION,
        },
        "implemented_consumer_preflight": {
            "sha256": PREVIOUS_CONSUMER_PREFLIGHT_SHA256,
        },
        "consumer": {
            "module": CONSUMER_RELATIVE_PATH.as_posix(),
            "module_sha256": consumer_sha,
            "entrypoint": PREVIOUS_CONSUMER_ENTRYPOINT.as_posix(),
            "entrypoint_sha256": entrypoint_sha,
        },
        "aligned_runtime": {
            "path": ALIGNMENT_RUNTIME_RELATIVE_PATH.as_posix(),
            "module_sha256": runtime_sha,
            "alignment_contract_sha256": ALIGNMENT_CONTRACT_SHA256,
        },
        "hash_on_trust_boundary": True,
    }


def _load_dataset_roles(
    root: Path,
    contract: FrozenTrainingContract,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    data_contract = _mapping(contract.payload.get("data_contract"), "data contract")
    source_artifacts = _mapping(data_contract.get("source_artifacts"), "source artifact manifest")
    training_ref = _mapping(source_artifacts.get("data/MNIST/processed/training.pt"), "training source artifact")
    training_path = root / Path(str(training_ref["path"]))
    actual_training_sha = file_sha256(training_path)
    _require(actual_training_sha == training_ref.get("sha256"), "local MNIST training source changed")
    dataset_root = root / Path(str(data_contract["root"]))
    loader_root = dataset_root.parent if dataset_root.name == "MNIST" else dataset_root
    _require(loader_root / "MNIST" == dataset_root, "frozen MNIST root cannot bind to the canonical loader")
    images, labels = load_local_mnist(loader_root, train=True)
    train_indices, development_indices = stratified_mnist_indices(labels, seed=SCHEDULE_SEED, validation_per_label=500)
    _require(tuple(images.shape) == (60_000, 1, 28, 28), "local MNIST image shape changed")
    _require(tuple(labels.shape) == (60_000,), "local MNIST label shape changed")
    _require(train_indices.numel() == 55_000 and development_indices.numel() == 5_000, "train/development counts changed")
    _require(torch.unique(torch.cat((train_indices, development_indices))).numel() == 60_000, "train/development roles overlap")
    train_counts = torch.bincount(labels.index_select(0, train_indices), minlength=10)
    development_counts = torch.bincount(labels.index_select(0, development_indices), minlength=10)
    _require(torch.equal(development_counts, torch.full((10,), 500, dtype=torch.long)), "development role is not 500/class")
    _require(int(train_counts.min()) >= TRAINING_STEPS, "train role cannot support the frozen schedule")
    selection_indices = _fixed_development_selection(labels, development_indices)
    test_ref = _mapping(source_artifacts.get("data/MNIST/processed/test.pt"), "protected test artifact")
    roles = DatasetRoleMetadata().to_dict()
    dataset = {
        **roles,
        "loader": "ldm_project.v2.student.evaluator.load_local_mnist",
        "source_artifacts": {
            "training": {
                "path": str(training_ref["path"]),
                "expected_sha256": str(training_ref["sha256"]),
                "actual_sha256": actual_training_sha,
                "read": True,
            },
            "official_test": {
                "path": str(test_ref["path"]),
                "contract_sha256": str(test_ref["sha256"]),
                "read": False,
                "role": "protected heldout; not consumed",
            },
        },
        "split": {
            "algorithm": "stratified_mnist_indices(labels, seed=1234, validation_per_label=500)",
            "seed": SCHEDULE_SEED,
            "train_count": int(train_indices.numel()),
            "development_count": int(development_indices.numel()),
            "train_counts_by_class": [int(value) for value in train_counts.tolist()],
            "development_counts_by_class": [int(value) for value in development_counts.tolist()],
            "train_indices_sha256": tensor_sha256(train_indices),
            "development_indices_sha256": tensor_sha256(development_indices),
            "selection_subset": {
                "count": int(selection_indices.numel()),
                "per_class": 100,
                "seed": DEVELOPMENT_SELECTION_SEED,
                "indices_sha256": tensor_sha256(selection_indices),
            },
        },
        "heldout_consumed": False,
        "sealed_consumed": False,
        "labels_used_as_model_input": False,
    }
    return images, labels, train_indices, development_indices, selection_indices, dataset


def _fixed_development_selection(labels: torch.Tensor, development_indices: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(DEVELOPMENT_SELECTION_SEED)
    development_labels = labels.index_select(0, development_indices)
    selected: list[torch.Tensor] = []
    for digit in range(10):
        matches = development_indices[development_labels == digit]
        _require(matches.numel() == 500, "development split is not exactly 500 rows per class")
        order = torch.randperm(matches.numel(), generator=generator)
        selected.append(matches.index_select(0, order[:100]))
    result = torch.cat(selected)
    _require(result.numel() == DEVELOPMENT_COUNT, "development selection count changed")
    return result


def _nvidia_smi_facts() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise BlockedTrainingExecution("BLOCKED_TARGET_DEVICE_UNAVAILABLE", "device", "nvidia-smi is unavailable")
    try:
        full = subprocess.run([executable], check=True, capture_output=True, text=True).stdout
        query = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise BlockedTrainingExecution(
            "BLOCKED_TARGET_DEVICE_UNAVAILABLE",
            "device",
            "nvidia-smi could not resolve the target device",
            error_type=type(error).__name__,
            error=str(error),
        ) from error
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", full)
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", full)
    fields = [field.strip() for field in query.split(",")]
    return {
        "executable": executable,
        "driver_version": driver_match.group(1) if driver_match else None,
        "driver_visible_cuda": cuda_match.group(1) if cuda_match else None,
        "query": {
            "gpu_name": fields[0] if len(fields) > 0 else None,
            "driver_version": fields[1] if len(fields) > 1 else None,
            "memory_total_mib": int(fields[2]) if len(fields) > 2 and fields[2].isdigit() else None,
        },
    }


def _target_device() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise BlockedTrainingExecution(
            "BLOCKED_TARGET_DEVICE_UNAVAILABLE",
            "device",
            "the authorized single CUDA device is unavailable",
            cuda_available=bool(torch.cuda.is_available()),
            device_count=int(torch.cuda.device_count()),
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    capability = (int(properties.major), int(properties.minor))
    if properties.name != EXPECTED_GPU_NAME or capability != EXPECTED_COMPUTE_CAPABILITY:
        raise BlockedTrainingExecution(
            "BLOCKED_TARGET_DEVICE_MISMATCH",
            "device",
            "the available GPU does not match the authorized target",
            actual_name=properties.name,
            actual_compute_capability=list(capability),
            expected_name=EXPECTED_GPU_NAME,
            expected_compute_capability=list(EXPECTED_COMPUTE_CAPABILITY),
        )
    if torch.__version__.split("+")[0] != EXPECTED_TORCH_VERSION or torch.version.cuda != EXPECTED_CUDA_BUILD:
        raise BlockedTrainingExecution(
            "BLOCKED_TARGET_BACKEND_MISMATCH",
            "device",
            "the validated PyTorch/CUDA backend identity changed",
            torch_version=torch.__version__,
            pytorch_cuda_build=torch.version.cuda,
        )
    env_root = Path(sys.prefix)
    candidates = sorted(
        {
            path.resolve()
            for directory in (env_root / "bin", env_root / "Library" / "bin")
            if directory.is_dir()
            for path in directory.glob("nvrtc*.dll")
        }
    )
    loads: list[dict[str, Any]] = []
    loader = getattr(ctypes, "WinDLL", None)
    if loader is not None:
        for path in candidates:
            try:
                loader(str(path))
                loads.append({"path": str(path), "loaded": True})
            except OSError as error:
                loads.append({"path": str(path), "loaded": False, "error": str(error)})
    try:
        complex_probe = torch.tensor([1.0 + 1.0j], dtype=torch.complex64, device=device)
        exp_probe = torch.exp(complex_probe)
        fft_probe = torch.fft.ifft2(torch.fft.fft2(complex_probe.reshape(1, 1, 1, 1)))
        polar_probe = torch.polar(torch.ones(1, device=device), torch.zeros(1, device=device))
        torch.cuda.synchronize(device)
    except Exception as error:
        raise BlockedTrainingExecution(
            "BLOCKED_CUDA_BACKEND_UNAVAILABLE",
            "cuda_backend",
            "the required CUDA complex backend probe failed",
            error_type=type(error).__name__,
            error=str(error),
        ) from error
    probes = {
        "complex_exp": bool(torch.isfinite(exp_probe).all()),
        "fft2_ifft2": bool(torch.isfinite(fft_probe).all()),
        "complex_polar": bool(torch.isfinite(polar_probe).all()),
    }
    if not all(probes.values()):
        raise BlockedTrainingExecution("BLOCKED_CUDA_BACKEND_UNAVAILABLE", "cuda_backend", "a CUDA backend probe was nonfinite", probes=probes)
    return device, {
        "device_index": 0,
        "device_count": 1,
        "gpu": properties.name,
        "compute_capability": list(capability),
        "vram_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "pytorch_compiled_cuda_version_int": getattr(torch._C, "_cuda_getCompiledVersion", lambda: None)(),
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_visible": _nvidia_smi_facts(),
        "backend": {
            "cuda_built": bool(torch.backends.cuda.is_built()),
            "nvrtc_required_by_active_path": True,
            "nvrtc_candidate_dlls": [str(path) for path in candidates],
            "nvrtc_dll_loads": loads,
            "probe": probes,
            "path_prefix": [str(env_root / "bin"), str(env_root / "Library" / "bin")],
        },
    }


def _memory_snapshot(device: torch.device) -> dict[str, int]:
    free, total = torch.cuda.mem_get_info(device)
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "free_bytes": int(free),
        "total_bytes": int(total),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


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


def _rng_state_payload() -> dict[str, Any]:
    return {
        "cpu": torch.random.get_rng_state().cpu().clone(),
        "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()],
        "cuda_available": True,
    }


def _sample_development_target() -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(DEVELOPMENT_TARGET_SEED)
    real = torch.randn((DEVELOPMENT_COUNT, 100), generator=generator, dtype=torch.float32)
    imag = torch.randn((DEVELOPMENT_COUNT, 100), generator=generator, dtype=torch.float32)
    coefficients = torch.complex(real, imag) * (PRIOR_SIGMA / math.sqrt(2.0))
    _require(torch.isfinite(coefficients).all(), "development prior target is nonfinite")
    return coefficients, {
        "seed": DEVELOPMENT_TARGET_SEED,
        "sample_count": DEVELOPMENT_COUNT,
        "namespace": PRIOR_NAMESPACE,
        "role": "development_target",
        "dtype": str(coefficients.dtype),
        "shape": list(coefficients.shape),
        "tensor_sha256": tensor_sha256(coefficients),
        "generator_state_sha256": tensor_sha256(generator.get_state()),
    }


def _complex_summary(value: torch.Tensor) -> dict[str, float]:
    _require(torch.is_complex(value) and torch.isfinite(value).all(), "complex diagnostic is invalid")
    amplitude = value.abs()
    power = amplitude.square()
    return {
        "mean_real": float(value.real.mean().detach().cpu()),
        "mean_imag": float(value.imag.mean().detach().cpu()),
        "mean_abs": float(amplitude.mean().detach().cpu()),
        "mean_power": float(power.mean().detach().cpu()),
        "max_abs": float(amplitude.max().detach().cpu()),
    }


def _development_evaluation(
    consumer: LightGen2DComplexOLSTrainingConsumer,
    images: torch.Tensor,
    selection_indices: torch.Tensor,
    development_target: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    _require(selection_indices.numel() == DEVELOPMENT_COUNT, "development selection count changed")
    before_trainable = consumer.trainable_state_dict()
    before_frozen = consumer._frozen_state_snapshot()
    consumer.runtime.eval()
    z_rows: list[torch.Tensor] = []
    image_squared_sum = 0.0
    image_element_count = 0
    try:
        with torch.no_grad():
            for chunk in selection_indices.split(LOGICAL_BATCH_SIZE):
                batch = images.index_select(0, chunk).to(device)
                result = consumer.runtime.forward_from_image(batch)
                raw_cmos = result.raw_intensity
                _require(result.image_input is not None, "development path did not expose image input")
                target = result.image_input.slm_intensity
                _require(raw_cmos.shape == target.shape, "development raw CMOS and target shapes differ")
                _require(bool(torch.isfinite(raw_cmos).all()), "development raw CMOS is nonfinite")
                _require(bool(torch.isfinite(result.z_ols.coefficients).all()), "development complex OLS is nonfinite")
                image_squared_sum += float((raw_cmos - target).square().sum().cpu())
                image_element_count += raw_cmos.numel()
                z_rows.append(result.z_ols.coefficients.detach())
                del result, batch
        encoded_z = torch.cat(z_rows, dim=0)
        target_z = development_target.to(device)
        latent_loss = full_complex_mmd_vstat(encoded_z, target_z)
        image_loss = image_squared_sum / float(image_element_count)
        optical_loss = 0.0
        total_loss = image_loss + float(latent_loss.detach().cpu())
        _require(math.isfinite(image_loss) and math.isfinite(total_loss), "development loss is nonfinite")
        torch.cuda.synchronize(device)
        metrics = {
            "sample_count": DEVELOPMENT_COUNT,
            "batch_size": LOGICAL_BATCH_SIZE,
            "selection_seed": DEVELOPMENT_SELECTION_SEED,
            "target_seed": DEVELOPMENT_TARGET_SEED,
            "L_image": image_loss,
            "L_latent": float(latent_loss.detach().cpu()),
            "L_optical": optical_loss,
            "L_total": total_loss,
            "encoded_z": _complex_summary(encoded_z),
            "target_z": _complex_summary(target_z),
            "target_z_sha256": tensor_sha256(development_target),
            "memory": _memory_snapshot(device),
        }
    finally:
        consumer.runtime.train()
    after_trainable = consumer.trainable_state_dict()
    _require(all(torch.equal(before_trainable[name], after_trainable[name]) for name in before_trainable), "development validation changed trainable parameters")
    after_frozen = consumer._frozen_state_snapshot()
    _require(
        before_frozen.keys() == after_frozen.keys()
        and all(torch.equal(before_frozen[name], after_frozen[name]) for name in before_frozen),
        "development validation changed frozen state",
    )
    del z_rows
    gc.collect()
    return metrics


def _update_record(
    update: Any,
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    torch.cuda.synchronize(device)
    result = update.loss.result
    z = result.z_ols.coefficients.detach()
    prior = update.loss.prior.state.coefficients.detach()
    raw_cmos = result.raw_intensity.detach()
    _require(bool(torch.isfinite(z).all()), "encoded complex OLS is nonfinite")
    _require(bool(torch.isfinite(prior).all()), "prior complex OLS is nonfinite")
    _require(bool(torch.isfinite(raw_cmos).all()), "raw CMOS output is nonfinite")
    return {
        **update.to_dict(),
        "batch_indices_sha256": tensor_sha256(indices),
        "batch_labels": list(range(10)),
        "finite": {
            "complex_ols": True,
            "prior": True,
            "raw_cmos": True,
            "component_losses": True,
        },
        "ols": {
            "shape": list(z.shape),
            "dtype": str(z.dtype),
            "encoded": _complex_summary(z),
            "prior": _complex_summary(prior),
            "amplitude_only_diagnostic": _complex_summary(torch.complex(z.abs(), torch.zeros_like(z.real))),
        },
        "raw_cmos": {
            "shape": list(raw_cmos.shape),
            "mean": float(raw_cmos.mean().cpu()),
            "mean_power": float(raw_cmos.square().mean().cpu()),
            "max": float(raw_cmos.max().cpu()),
        },
        "memory": _memory_snapshot(device),
    }


def _formal_checkpoint_payload(
    *,
    root: Path,
    contract: FrozenTrainingContract,
    authorities: Mapping[str, Any],
    consumer: LightGen2DComplexOLSTrainingConsumer,
    optimizer: torch.optim.Optimizer,
    sampler: BalancedTrainRoleSampler,
    dataset: Mapping[str, Any],
    last_prior: PriorSample,
    completed_steps: int,
    development: Mapping[str, Any],
    entrypoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    _require(completed_steps in (50, 100), "checkpoint cadence is outside the frozen schedule")
    return {
        "schema": FORMAL_CHECKPOINT_SCHEMA,
        "version": 1,
        "valid_training_checkpoint": True,
        "selected": False,
        "accepted_training_checkpoint": False,
        "contract": contract.identity(root),
        "authorities": dict(authorities),
        "runtime": consumer.runtime_identity,
        "consumer": {
            "module": CONSUMER_RELATIVE_PATH.as_posix(),
            "module_sha256": CONSUMER_SHA256,
            "formal_entrypoint": dict(entrypoint_identity),
        },
        "architecture_identity": {
            "scope": "2D semantic image generation only",
            "ols_interface": "lightgen_10x10_complex_ols_v1",
            "ols_shape": [100],
            "port_order": "row_major j=10*r+c",
            "modal_projection": "z_j=<E_encoder,u_j>",
            "transport": "T=I_100 coefficient space",
            "writer": "W_OLS(z)=sum_j (Tz)_j v_j",
            "raw_output": "raw CMOS intensity |E_out|^2",
        },
        "training_configuration": {
            "device": "cuda:0",
            "propagation_grid": PRODUCTION_PROPAGATION_GRID,
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "steps": TRAINING_STEPS,
            "schedule_seed": SCHEDULE_SEED,
            "training_prior_seed": TRAINING_PRIOR_SEED,
            "development_selection_seed": DEVELOPMENT_SELECTION_SEED,
            "development_target_seed": DEVELOPMENT_TARGET_SEED,
            "autocast": False,
            "grad_scaler": False,
            "gradient_clipping": None,
        },
        "global_step": completed_steps - 1,
        "completed_steps": completed_steps,
        "epoch": 0,
        "trainable_parameter_state": _cpu_tree(consumer.trainable_state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()),
        "scheduler_state": None,
        "rng_state": _cpu_tree(_rng_state_payload()),
        "prior_rng": {
            "namespace": last_prior.namespace,
            "role": last_prior.role,
            "global_step": last_prior.global_step,
            "sample_count": PRIOR_SAMPLE_COUNT,
            "seed": last_prior.seed,
            "state": last_prior.generator_state.cpu().clone(),
        },
        "sampler_state": _cpu_tree(sampler.state_dict()),
        "dataset_roles": dict(dataset),
        "development_evaluation": dict(development),
        "loss_configuration": consumer.loss_configuration(),
        "optimizer_configuration": consumer.optimizer_manifest(optimizer),
        "checkpoint_selection_candidate": True,
        "promotion": {"formal_promotion": False, "generation_readiness": "not_ready"},
    }


def _validate_formal_checkpoint(
    path: Path,
    expected_sha256: str,
    contract: FrozenTrainingContract,
    completed_steps: int,
) -> dict[str, Any]:
    payload = safe_torch_load(path, expected_sha256=expected_sha256, map_location="cpu")
    _require(isinstance(payload, Mapping), "formal checkpoint is not a mapping")
    _require(payload.get("schema") == FORMAL_CHECKPOINT_SCHEMA, "formal checkpoint schema mismatch")
    _require(payload.get("valid_training_checkpoint") is True, "formal checkpoint validity flag is missing")
    _require(payload.get("selected") is False, "intermediate checkpoint was marked selected")
    _require(payload.get("accepted_training_checkpoint") is False, "checkpoint crossed promotion boundary")
    checkpoint_contract = _mapping(payload.get("contract"), "checkpoint contract")
    _require(checkpoint_contract.get("sha256") == contract.sha256, "formal checkpoint contract identity mismatch")
    _require(payload.get("completed_steps") == completed_steps, "formal checkpoint step count mismatch")
    state = _mapping(payload.get("trainable_parameter_state"), "checkpoint trainable state")
    _require(set(state) == set(ALLOWED_TRAINABLE_PARAMETER_NAMES), "formal checkpoint trainable state keys changed")
    expected_shapes = {"encoder.phase": (2, 1, 100, 100), "generator.phase": (2, 1, 100, 100)}
    for name in ALLOWED_TRAINABLE_PARAMETER_NAMES:
        value = state[name]
        _require(isinstance(value, torch.Tensor), f"formal checkpoint parameter is not a tensor: {name}")
        _require(tuple(value.shape) == expected_shapes[name] and value.dtype == torch.float32, f"formal checkpoint parameter changed: {name}")
        _require(bool(torch.isfinite(value).all()), f"formal checkpoint parameter is nonfinite: {name}")
    _require(isinstance(payload.get("optimizer_state"), Mapping), "formal checkpoint optimizer state is missing")
    _require(isinstance(payload.get("rng_state"), Mapping), "formal checkpoint RNG state is missing")
    _require(isinstance(payload.get("prior_rng"), Mapping), "formal checkpoint prior RNG state is missing")
    _require(isinstance(payload.get("sampler_state"), Mapping), "formal checkpoint sampler state is missing")
    return dict(payload)


def _save_formal_checkpoint(
    *,
    path: Path,
    payload: Mapping[str, Any],
    contract: FrozenTrainingContract,
    completed_steps: int,
) -> dict[str, Any]:
    atomic_torch_save_new(path, payload, artifact_kind="LightGen formal training checkpoint")
    sha256 = file_sha256(path)
    _validate_formal_checkpoint(path, sha256, contract, completed_steps)
    return {
        "path": str(path),
        "relative_path": _relative_path(path, _repo_root()),
        "sha256": sha256,
        "size_bytes": path.stat().st_size,
        "completed_steps": completed_steps,
        "valid_training_checkpoint": True,
        "selected": False,
        "accepted_training_checkpoint": False,
        "safe_load_verified": True,
    }


def _exception_to_failure(error: BaseException, stage: str) -> tuple[str, str, dict[str, Any]]:
    if isinstance(error, BlockedTrainingExecution):
        return "BLOCKED", error.classification, {"stage": error.stage, "message": str(error), **error.details}
    if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
        return "BLOCKED", "BLOCKED_BATCH_MEMORY_INFEASIBLE", {"stage": stage, "message": str(error), "error_type": type(error).__name__}
    if isinstance(error, (InvalidTrainingExecution, TrainingContractError, FloatingPointError)):
        return "INVALID", "INVALID_EXECUTION_VALIDITY_FAILURE", {"stage": stage, "message": str(error), "error_type": type(error).__name__}
    return "INVALID", "INVALID_UNEXPECTED_EXECUTION_ERROR", {
        "stage": stage,
        "message": str(error),
        "error_type": type(error).__name__,
    }


def _execute(
    *,
    root: Path,
    output_dir: Path,
    resume_checkpoint: Path | None,
    contract: FrozenTrainingContract,
    authorities: Mapping[str, Any],
    starting_git: Mapping[str, Any],
) -> dict[str, Any]:
    entrypoint_path = Path(__file__).resolve()
    entrypoint_identity = {
        "path": _relative_path(entrypoint_path, root),
        "sha256": file_sha256(entrypoint_path),
        "task_id": TASK_ID,
    }
    images, labels, train_indices, development_indices, selection_indices, dataset = _load_dataset_roles(root, contract)
    device, device_facts = _target_device()
    torch.cuda.set_device(device)
    torch.manual_seed(SCHEDULE_SEED)
    torch.cuda.manual_seed_all(SCHEDULE_SEED)
    development_target, development_target_identity = _sample_development_target()
    consumer = LightGen2DComplexOLSTrainingConsumer(
        contract,
        device=device,
        propagation_grid_size=PRODUCTION_PROPAGATION_GRID,
    )
    optimizer = consumer.create_optimizer()
    parameter_manifest = consumer.parameter_manifest()
    _require(parameter_manifest["trainable_parameter_count"] == 40_000, "trainable parameter count changed")
    _require(parameter_manifest["trainable_parameter_names"] == list(ALLOWED_TRAINABLE_PARAMETER_NAMES), "trainable parameter names changed")
    _require(all(torch.equal(value, torch.zeros_like(value)) for value in consumer.trainable_state_dict().values()), "zero-phase initialization changed")
    optimizer_manifest = consumer.optimizer_manifest(optimizer)
    sampler = BalancedTrainRoleSampler(train_indices, labels)
    resumed_from = None
    completed_before_run = 0
    last_prior: PriorSample | None = None
    if resume_checkpoint is not None:
        checkpoint_path = resume_checkpoint if resume_checkpoint.is_absolute() else root / resume_checkpoint
        checkpoint_path = checkpoint_path.resolve()
        checkpoint_sha = file_sha256(checkpoint_path)
        payload = safe_torch_load(checkpoint_path, expected_sha256=checkpoint_sha, map_location="cpu")
        _require(isinstance(payload, Mapping), "resume checkpoint is not a mapping")
        _require(payload.get("schema") == FORMAL_CHECKPOINT_SCHEMA, "resume checkpoint schema mismatch")
        _require(payload.get("completed_steps") in (50,), "resume requires the valid step-50 checkpoint")
        _require(_mapping(payload.get("contract"), "resume checkpoint contract").get("sha256") == contract.sha256, "resume contract mismatch")
        consumer.load_trainable_state_dict(_mapping(payload.get("trainable_parameter_state"), "resume trainable state"))
        optimizer_state = payload.get("optimizer_state")
        _require(isinstance(optimizer_state, Mapping), "resume optimizer state is missing")
        optimizer.load_state_dict(dict(optimizer_state))
        consumer.optimizer_manifest(optimizer)
        rng_state = _mapping(payload.get("rng_state"), "resume RNG state")
        restore_rng_state(rng_state)
        sampler.load_state_dict(_mapping(payload.get("sampler_state"), "resume sampler state"))
        completed_before_run = int(payload["completed_steps"])
        resumed_from = {"path": str(checkpoint_path), "sha256": checkpoint_sha, "completed_steps": completed_before_run}

    update_records: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    development_records: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for global_step in range(completed_before_run, TRAINING_STEPS):
        indices, batch_labels = sampler.next_batch()
        _require(torch.equal(batch_labels, torch.arange(10, dtype=torch.long)), "sampler label metadata order changed")
        batch = images.index_select(0, indices).to(device)
        try:
            update = consumer.update(batch, global_step=global_step, optimizer=optimizer)
        except Exception as error:
            if isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower():
                raise BlockedTrainingExecution("BLOCKED_BATCH_MEMORY_INFEASIBLE", f"training_step_{global_step}", "frozen batch exhausted target-device memory", error=str(error)) from error
            raise InvalidTrainingExecution(f"training step {global_step} violated a frozen validity invariant: {error}") from error
        update_record = _update_record(update, indices, device)
        update_records.append(update_record)
        last_prior = update.loss.prior
        del update, batch
        completed_steps = global_step + 1
        if completed_steps in CHECKPOINT_NAMES:
            gc.collect()
            torch.cuda.empty_cache()
            development = _development_evaluation(consumer, images, selection_indices, development_target, device)
            development["checkpoint_step"] = completed_steps
            development_records.append(development)
            checkpoint_payload = _formal_checkpoint_payload(
                root=root,
                contract=contract,
                authorities=authorities,
                consumer=consumer,
                optimizer=optimizer,
                sampler=sampler,
                dataset=dataset,
                last_prior=last_prior,
                completed_steps=completed_steps,
                development=development,
                entrypoint_identity=entrypoint_identity,
            )
            checkpoint_path = output_dir / CHECKPOINT_NAMES[completed_steps]
            checkpoint_record = _save_formal_checkpoint(
                path=checkpoint_path,
                payload=checkpoint_payload,
                contract=contract,
                completed_steps=completed_steps,
            )
            checkpoint_record["development"] = development
            checkpoint_records.append(checkpoint_record)
            print(
                json.dumps(
                    {"event": "checkpoint", "completed_steps": completed_steps, "development_L_total": development["L_total"], "path": str(checkpoint_path)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if completed_steps % 10 == 0 or completed_steps == TRAINING_STEPS:
            print(
                json.dumps(
                    {"event": "progress", "completed_steps": completed_steps, "L_total": update_record["loss"]["L_total"], "elapsed_seconds": time.perf_counter() - start_time},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        del indices, batch_labels
        if global_step % 5 == 4:
            gc.collect()

    _require(last_prior is not None, "training produced no prior state")
    _require(len(update_records) == TRAINING_STEPS - completed_before_run, "training update count changed")
    _require(len(checkpoint_records) == 2 if completed_before_run == 0 else len(checkpoint_records) == 1, "checkpoint count changed")
    selected = min(checkpoint_records, key=lambda record: (float(record["development"]["L_total"]), int(record["completed_steps"])))
    for record in checkpoint_records:
        record["selected"] = record["path"] == selected["path"]
    ending_git = git_facts(root)
    _require(ending_git["branch"] == starting_git["branch"] and ending_git["head"] == starting_git["head"], "branch or HEAD changed during training")
    duration = time.perf_counter() - start_time
    return {
        "status": "valid_completed",
        "result": "VALID_COMPLETED",
        "classification": SUCCESS_CLASSIFICATION,
        "authorities": dict(authorities),
        "entrypoint": entrypoint_identity,
        "device": device_facts,
        "target_device": str(device),
        "training": {
            "device": str(device),
            "propagation_grid": PRODUCTION_PROPAGATION_GRID,
            "logical_batch_size": LOGICAL_BATCH_SIZE,
            "planned_steps": TRAINING_STEPS,
            "completed_steps": TRAINING_STEPS,
            "optimizer": optimizer_manifest,
            "scheduler": None,
            "precision": {"phase": "float32", "field": "complex64", "autocast": False, "grad_scaler": False},
            "seeds": {
                "initialization": SCHEDULE_SEED,
                "schedule": SCHEDULE_SEED,
                "training_prior": TRAINING_PRIOR_SEED,
                "development_selection": DEVELOPMENT_SELECTION_SEED,
                "development_target": DEVELOPMENT_TARGET_SEED,
            },
            "duration_seconds": duration,
            "started_at_utc": started_at,
            "resumed_from": resumed_from,
        },
        "parameter_manifest": parameter_manifest,
        "loss": {
            "initial": update_records[0]["loss"],
            "final": update_records[-1]["loss"],
            "checkpoint_development": development_records,
            "selected_development": selected["development"],
        },
        "complex_ols": {
            "authoritative_shape": [100],
            "authoritative_dtype": "torch.complex64",
            "interface": "lightgen_10x10_complex_ols_v1",
            "port_order": "row_major j=10*r+c",
            "prior": {
                "name": "PROJECT_COMPLEX_ISOTROPIC_GAUSSIAN_NATIVE_OLS_V1",
                "sigma_z": PRIOR_SIGMA,
                "namespace": PRIOR_NAMESPACE,
                "training_sample_count": PRIOR_SAMPLE_COUNT,
                "development_target": development_target_identity,
            },
            "q_phase_repair_used": False,
            "historical_producer_used": False,
        },
        "autograd_structure": {
            "modal_projection_active": all(record["autograd"]["modal_projection_active"] for record in update_records),
            "w_ols_active": all(record["autograd"]["w_ols_active"] for record in update_records),
            "trainable_gradients_finite": all(record["autograd"]["trainable_gradients_finite"] for record in update_records),
            "forbidden_gradients_absent": all(record["autograd"]["forbidden_gradients_absent"] for record in update_records),
            "frozen_state_unchanged": all(record["frozen_state_unchanged"] for record in update_records),
        },
        "data": dict(dataset),
        "checkpoint_records": checkpoint_records,
        "selected_checkpoint": selected,
        "training_trace": update_records,
        "development_target_identity": development_target_identity,
        "provenance": {
            "git_start": dict(starting_git),
            "git_end": ending_git,
            "branch_unchanged": True,
            "head_unchanged": True,
            "worktree_dirty": ending_git["worktree_dirty"],
        },
        "claim_boundary": {
            "explicit_user_authorization_consumed": True,
            "authorization_scope_expanded": False,
            "scientific_training_executed": True,
            "heldout_evaluation_executed": False,
            "test_split_read": False,
            "sealed_accessed": False,
            "sealed_reused": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "post_training_evaluation_executed": False,
        },
        "next": "DEFINE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_POST_TRAINING_EVALUATION_CONTRACT_V1",
        "next_note": "requires a separate task and must not execute automatically",
    }


def run_training(root: Path, output_dir: Path | None = None, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    target_dir = _resolve_output_dir(root, output_dir, resume_checkpoint)
    target_dir.mkdir(parents=True, exist_ok=False if resume_checkpoint is None else True)
    starting_git = git_facts(root)
    result: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(target_dir),
        "status": "invalid_harness",
        "result": "INVALID",
        "classification": "INVALID_TRAINING_NOT_COMPLETED",
        "authorization": {
            "explicit_user_authorization_consumed": True,
            "scope": TASK_ID,
            "scope_expanded": False,
            "training_authorized_for_this_run": True,
            "formal_promotion_authorized": False,
            "generation_readiness_promotion_authorized": False,
        },
        "provenance": {"git_start": starting_git},
        "claim_boundary": {
            "scientific_training_executed": False,
            "heldout_evaluation_executed": False,
            "test_split_read": False,
            "sealed_accessed": False,
            "sealed_reused": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "next": None,
    }
    lineage: dict[str, Any] = {
        "schema": LINEAGE_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "created_at_utc": result["created_at_utc"],
        "training_report": {"path": _relative_path(target_dir / REPORT_NAME, root), "sha256": "not_embedded_to_avoid_cycle"},
        "provenance": {"git_start": starting_git},
        "execution": {},
    }
    try:
        contract, authorities = _validate_authorities(root)
        execution = _execute(
            root=root,
            output_dir=target_dir,
            resume_checkpoint=resume_checkpoint,
            contract=contract,
            authorities=authorities,
            starting_git=starting_git,
        )
        result.update(execution)
        lineage["execution"] = execution
    except (BlockedTrainingExecution, InvalidTrainingExecution, TrainingContractError, FloatingPointError, OSError, RuntimeError) as error:
        status, classification, details = _exception_to_failure(error, "training_execution")
        result.update({"status": "blocked" if status == "BLOCKED" else "invalid_harness", "result": status, "classification": classification, "failure": details})
        lineage["execution"] = {"failure": details}
    except Exception as error:
        status, classification, details = _exception_to_failure(error, "unexpected")
        result.update({"status": "invalid_harness", "result": status, "classification": classification, "failure": details})
        lineage["execution"] = {"failure": details}
    finally:
        ending_git = git_facts(root)
        result.setdefault("provenance", {})["git_end"] = ending_git
        result["provenance"]["branch_unchanged"] = starting_git["branch"] == ending_git["branch"]
        result["provenance"]["head_unchanged"] = starting_git["head"] == ending_git["head"]
        result["provenance"]["worktree_dirty"] = ending_git["worktree_dirty"]
        lineage["provenance"]["git_end"] = ending_git
        lineage["provenance"]["branch_unchanged"] = result["provenance"]["branch_unchanged"]
        lineage["provenance"]["head_unchanged"] = result["provenance"]["head_unchanged"]
        lineage["provenance"]["worktree_dirty"] = ending_git["worktree_dirty"]
        if result["provenance"]["branch_unchanged"] is not True or result["provenance"]["head_unchanged"] is not True:
            result["result"] = "INVALID"
            result["status"] = "invalid_harness"
            result["classification"] = "INVALID_REPOSITORY_IDENTITY_CHANGED"
        lineage_path = target_dir / LINEAGE_NAME
        atomic_write_new_json(lineage_path, lineage, artifact_kind="LightGen training lineage")
        lineage_sha256 = file_sha256(lineage_path)
        result["lineage"] = {"path": str(lineage_path), "sha256": lineage_sha256}
        result["report_self_hash"] = "not_embedded_to_avoid_self_reference"
        report_path = target_dir / REPORT_NAME
        atomic_write_new_json(report_path, result, artifact_kind="LightGen training report")
        result["report"] = {"path": str(report_path), "sha256": file_sha256(report_path)}
    print(
        json.dumps(
            {
                "result": result["result"],
                "classification": result["classification"],
                "task_id": TASK_ID,
                "report": result.get("report", {}).get("path"),
                "report_sha256": result.get("report", {}).get("sha256"),
                "lineage": result.get("lineage", {}).get("path"),
                "lineage_sha256": result.get("lineage", {}).get("sha256"),
                "selected_checkpoint": result.get("selected_checkpoint", {}).get("path"),
                "formal_promotion": False,
                "generation_readiness": "not_ready",
                "next": result.get("next"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="new output directory under outputs/")
    parser.add_argument("--resume-checkpoint", type=Path, default=None, help="explicit step-50 checkpoint for same-run resume")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_training(_repo_root(), args.output_dir, args.resume_checkpoint)
    return 0 if result["result"] == "VALID_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
