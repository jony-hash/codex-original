"""Run the bounded LightGen complex-OLS long-training preflight.

This command validates the frozen target-device path only.  It performs three
real CUDA updates, a target-device smoke checkpoint/resume, and operational
memory/storage checks; it never starts the 100-step training run.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import gc
import json
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, NoReturn

import torch

from ldm_project.v2.application.research.lightgen_2d_semantic_complex_ols_training import (
    ALIGNMENT_RUNTIME_RELATIVE_PATH,
    ALIGNMENT_RUNTIME_SHA256,
    BalancedTrainRoleSampler,
    DatasetRoleMetadata,
    LightGen2DComplexOLSTrainingConsumer,
    PRODUCTION_PROPAGATION_GRID,
    TRAINING_CONTRACT_SHA256,
    TRAINING_STEPS,
    TrainingContractError,
    checkpoint_payload,
    file_sha256,
    git_facts,
    load_frozen_training_contract,
    load_smoke_checkpoint,
    save_smoke_checkpoint,
    validate_legacy_exclusion,
)
from ldm_project.v2.evidence import atomic_write_new_json, tensor_sha256
from ldm_project.v2.student.evaluator import load_local_mnist, stratified_mnist_indices


TASK_ID = "PREFLIGHT_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_V1"
SUCCESS_CLASSIFICATION = "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_LONG_TRAINING_AUTHORIZABLE"
PREVIOUS_PREFLIGHT_ARTIFACT_SHA256 = "c16dc2b5a1ed1c8a4f0ba0fc2e5a3e90154620e3f053d54d79342cc434e85aa7"
PREVIOUS_PREFLIGHT_ARTIFACT_RELATIVE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_training_consumer_v1_attempt_01/"
    "lightgen_2d_semantic_complex_ols_training_preflight_input.v1.json"
)
PREVIOUS_CONSUMER_ENTRYPOINT_RELATIVE_PATH = Path(
    "scripts/research/execute_lightgen_2d_semantic_complex_ols_training_v1.py"
)
PREVIOUS_CONSUMER_ENTRYPOINT_SHA256 = "c3b8d69b1cc387778414d3e53c895639af2318cb7c16ac2e94107aafcc05f0be"
CONSUMER_MODULE_RELATIVE_PATH = Path(
    "src/ldm_project/v2/application/research/"
    "lightgen_2d_semantic_complex_ols_training.py"
)
CONSUMER_MODULE_SHA256 = "fae7ffcd5f724aa4a07966e0f75dbe77653eaf85cf9907093606ceb884517d1c"
PREFLIGHT_ARTIFACT_NAME = "lightgen_2d_semantic_complex_ols_long_training_preflight.v1.json"
SMOKE_CHECKPOINT_NAME = "lightgen_2d_semantic_complex_ols_long_training_preflight_smoke_checkpoint.v1.pt"
DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "lightgen_2d_semantic_complex_ols_long_training_preflight_v1_attempt_01"
)
LOGICAL_BATCH_SIZE = 10
DEV_SELECTION_SEED = 4321
TRAINING_SEED = 20260910


class PreflightAbort(RuntimeError):
    """A valid preflight decision that stops the downstream training path."""

    def __init__(self, classification: str, stage: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.classification = classification
        self.stage = stage
        self.details = details


class InvalidPreflight(RuntimeError):
    """A malformed authority or execution harness, not a scientific failure."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InvalidPreflight(f"{name} must be a mapping")
    return value


def _require(condition: bool, message: str, *, classification: str, stage: str, **details: Any) -> None:
    if not condition:
        raise PreflightAbort(classification, stage, message, **details)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
        raise InvalidPreflight("checkpoint loss metric keys differ")
    return max((abs(float(left[key]) - float(right[key])) for key in left), default=0.0)


def _exception_is_oom(error: BaseException) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def _abort_gpu_error(error: BaseException, *, stage: str) -> NoReturn:
    if _exception_is_oom(error):
        raise PreflightAbort(
            "BLOCKED_BATCH_MEMORY_INFEASIBLE",
            stage,
            "exact frozen batch exhausted target-device memory",
            error_type=type(error).__name__,
            error=str(error),
        ) from error
    if isinstance(error, TrainingContractError):
        raise PreflightAbort(
            "BLOCKED_GPU_SMOKE_STRUCTURAL_INVARIANT_FAILURE",
            stage,
            str(error),
            error_type=type(error).__name__,
        ) from error
    if isinstance(error, FloatingPointError):
        raise PreflightAbort(
            "BLOCKED_GPU_SMOKE_NONFINITE",
            stage,
            str(error),
            error_type=type(error).__name__,
        ) from error
    raise error


def _validate_authorities(root: Path) -> tuple[Any, dict[str, Any]]:
    contract = load_frozen_training_contract(repo_root=root)
    previous_path = root / PREVIOUS_PREFLIGHT_ARTIFACT_RELATIVE_PATH
    if not previous_path.is_file():
        raise InvalidPreflight(f"implemented consumer preflight artifact is unavailable: {previous_path}")
    previous_sha = file_sha256(previous_path)
    if previous_sha != PREVIOUS_PREFLIGHT_ARTIFACT_SHA256:
        raise InvalidPreflight(
            f"implemented consumer preflight artifact SHA mismatch: {previous_sha}"
        )
    try:
        previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InvalidPreflight("implemented consumer preflight artifact is not valid JSON") from error
    previous = _mapping(previous_payload, "implemented consumer preflight artifact")
    if previous.get("classification") != "LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_CONSUMER_V1_IMPLEMENTED":
        raise InvalidPreflight("implemented consumer preflight classification changed")
    previous_contract = _mapping(previous.get("contract"), "implemented consumer contract")
    if previous_contract.get("sha256") != TRAINING_CONTRACT_SHA256:
        raise InvalidPreflight("implemented consumer preflight is bound to a different training contract")
    boundary = _mapping(previous.get("claim_boundary"), "implemented consumer claim boundary")
    if any(boundary.get(key) is not False for key in ("long_training_executed", "heldout_evaluation_executed", "sealed_reused")):
        raise InvalidPreflight("implemented consumer preflight crossed the training or protected-data boundary")
    consumer = _mapping(previous.get("consumer"), "implemented consumer identity")
    module_path = root / CONSUMER_MODULE_RELATIVE_PATH
    entrypoint_path = root / PREVIOUS_CONSUMER_ENTRYPOINT_RELATIVE_PATH
    module_sha = file_sha256(module_path)
    entrypoint_sha = file_sha256(entrypoint_path)
    if module_sha != CONSUMER_MODULE_SHA256 or consumer.get("module_sha256") != module_sha:
        raise InvalidPreflight("training consumer module identity changed after implementation preflight")
    if entrypoint_sha != PREVIOUS_CONSUMER_ENTRYPOINT_SHA256 or consumer.get("entrypoint_sha256") != entrypoint_sha:
        raise InvalidPreflight("training consumer entrypoint identity changed after implementation preflight")
    runtime_path = root / ALIGNMENT_RUNTIME_RELATIVE_PATH
    runtime_sha = file_sha256(runtime_path)
    if runtime_sha != ALIGNMENT_RUNTIME_SHA256:
        raise InvalidPreflight("aligned runtime identity changed")
    return contract, {
        "training_contract": contract.identity(root),
        "implemented_consumer_preflight": {
            "path": PREVIOUS_PREFLIGHT_ARTIFACT_RELATIVE_PATH.as_posix(),
            "sha256": previous_sha,
        },
        "consumer": {
            "module": CONSUMER_MODULE_RELATIVE_PATH.as_posix(),
            "module_sha256": module_sha,
            "entrypoint": PREVIOUS_CONSUMER_ENTRYPOINT_RELATIVE_PATH.as_posix(),
            "entrypoint_sha256": entrypoint_sha,
        },
        "aligned_runtime": {
            "path": ALIGNMENT_RUNTIME_RELATIVE_PATH.as_posix(),
            "sha256": runtime_sha,
        },
        "hash_on_trust_boundary": True,
    }


def _fixed_development_selection(labels: torch.Tensor, development_indices: torch.Tensor) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(DEV_SELECTION_SEED)
    selected: list[torch.Tensor] = []
    dev_labels = labels.index_select(0, development_indices)
    for label in range(10):
        matches = development_indices[dev_labels == label]
        if matches.numel() != 500:
            raise InvalidPreflight("development split is not exactly 500 rows per class")
        order = torch.randperm(matches.numel(), generator=generator)
        selected.append(matches.index_select(0, order[:100]))
    return torch.cat(selected)


def _load_dataset_roles(
    root: Path,
    contract_payload: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    data_contract = _mapping(contract_payload.get("data_contract"), "data contract")
    source_artifacts = _mapping(data_contract.get("source_artifacts"), "source artifact manifest")
    training_ref = _mapping(source_artifacts.get("data/MNIST/processed/training.pt"), "training source artifact")
    training_path = root / Path(str(training_ref["path"]))
    actual_training_sha = file_sha256(training_path)
    if actual_training_sha != training_ref.get("sha256"):
        raise InvalidPreflight("local MNIST training source does not match the frozen contract")
    dataset_root = root / Path(str(data_contract["root"]))
    loader_root = dataset_root.parent if dataset_root.name == "MNIST" else dataset_root
    if loader_root / "MNIST" != dataset_root:
        raise InvalidPreflight("frozen MNIST root cannot be bound to the canonical loader")
    images, labels = load_local_mnist(loader_root, train=True)
    train_indices, development_indices = stratified_mnist_indices(labels, seed=1234, validation_per_label=500)
    if tuple(images.shape) != (60_000, 1, 28, 28) or tuple(labels.shape) != (60_000,):
        raise InvalidPreflight("local MNIST source shape is outside the frozen contract")
    if train_indices.numel() != 55_000 or development_indices.numel() != 5_000:
        raise InvalidPreflight("frozen train/development role counts are not present")
    if torch.unique(torch.cat((train_indices, development_indices))).numel() != 60_000:
        raise InvalidPreflight("train and development roles overlap or omit source rows")
    train_counts = torch.bincount(labels.index_select(0, train_indices), minlength=10)
    development_counts = torch.bincount(labels.index_select(0, development_indices), minlength=10)
    if not torch.equal(development_counts, torch.full((10,), 500, dtype=torch.long)):
        raise InvalidPreflight("development role is not balanced at 500 rows per class")
    if int(train_counts.min()) < TRAINING_STEPS:
        raise InvalidPreflight("train role cannot support the frozen 100-step no-replacement sampler")
    selection_indices = _fixed_development_selection(labels, development_indices)
    test_ref = _mapping(source_artifacts.get("data/MNIST/processed/test.pt"), "protected test artifact")
    base_roles = DatasetRoleMetadata().to_dict()
    dataset = {
        **base_roles,
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
            "seed": 1234,
            "train_count": int(train_indices.numel()),
            "development_count": int(development_indices.numel()),
            "train_counts_by_class": [int(value) for value in train_counts.tolist()],
            "development_counts_by_class": [int(value) for value in development_counts.tolist()],
            "train_indices_sha256": tensor_sha256(train_indices),
            "development_indices_sha256": tensor_sha256(development_indices),
            "selection_subset": {
                "count": int(selection_indices.numel()),
                "per_class": 100,
                "seed": DEV_SELECTION_SEED,
                "indices_sha256": tensor_sha256(selection_indices),
            },
        },
        "heldout_consumed": False,
        "sealed_consumed": False,
        "labels_used_as_model_input": False,
    }
    return images, labels, train_indices, development_indices, selection_indices, dataset


def _nvidia_smi_facts() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise PreflightAbort(
            "BLOCKED_TARGET_DEVICE_UNAVAILABLE",
            "device",
            "nvidia-smi is unavailable; driver-visible CUDA cannot be resolved",
        )
    try:
        full = subprocess.run([executable], check=True, capture_output=True, text=True).stdout
        query = subprocess.run(
            [executable, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise PreflightAbort(
            "BLOCKED_TARGET_DEVICE_UNAVAILABLE",
            "device",
            "nvidia-smi could not resolve driver-visible device facts",
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


def _cuda_device_facts() -> tuple[torch.device, dict[str, Any]]:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise PreflightAbort(
            "BLOCKED_TARGET_DEVICE_UNAVAILABLE",
            "device",
            "the required CUDA target device is unavailable",
            cuda_available=bool(torch.cuda.is_available()),
            device_count=int(torch.cuda.device_count()),
        )
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    smi = _nvidia_smi_facts()
    env_root = Path(sys.prefix)
    dll_candidates = sorted(
        {
            path.resolve()
            for directory in (env_root / "bin", env_root / "Library" / "bin")
            if directory.is_dir()
            for path in directory.glob("nvrtc*.dll")
        }
    )
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
        backend_probe = {
            "complex_exp": bool(torch.isfinite(exp_probe).all()),
            "fft2_ifft2": bool(torch.isfinite(fft_probe).all()),
            "complex_polar": bool(torch.isfinite(polar_probe).all()),
        }
    except Exception as error:
        raise PreflightAbort(
            "BLOCKED_CUDA_BACKEND_UNAVAILABLE",
            "cuda_backend",
            "required CUDA complex/FFT backend probe failed",
            error_type=type(error).__name__,
            error=str(error),
            nvrtc_required_by_active_path=True,
        ) from error
    if not all(backend_probe.values()):
        raise PreflightAbort(
            "BLOCKED_CUDA_BACKEND_UNAVAILABLE",
            "cuda_backend",
            "required CUDA backend probe produced a nonfinite value",
            backend_probe=backend_probe,
        )
    facts = {
        "device_index": 0,
        "device_count": int(torch.cuda.device_count()),
        "gpu": properties.name,
        "compute_capability": [int(properties.major), int(properties.minor)],
        "vram_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "pytorch_cuda_build": torch.version.cuda,
        "pytorch_compiled_cuda_version_int": getattr(torch._C, "_cuda_getCompiledVersion", lambda: None)(),
        "cudnn_version": torch.backends.cudnn.version(),
        "driver_visible": smi,
        "backend": {
            "cuda_built": bool(torch.backends.cuda.is_built()),
            "nvrtc_required_by_active_path": True,
            "nvrtc_candidate_dlls": [str(path) for path in dll_candidates],
            "nvrtc_dll_loads": dll_loads,
            "probe": backend_probe,
            "path_prefix": [str(env_root / "bin"), str(env_root / "Library" / "bin")],
        },
    }
    return device, facts


def _measured_update(
    consumer: LightGen2DComplexOLSTrainingConsumer,
    optimizer: torch.optim.Optimizer,
    batch: torch.Tensor,
    *,
    global_step: int,
    device: torch.device,
) -> tuple[Any, dict[str, Any]]:
    memory: dict[str, Any] = {}
    original_forward = consumer.forward_and_loss
    original_retain = consumer._retain_autograd_probes
    original_validate = consumer._validate_autograd
    original_step = optimizer.step

    def measured_forward(images: torch.Tensor, *, global_step: int) -> Any:
        result = original_forward(images, global_step=global_step)
        torch.cuda.synchronize(device)
        memory["forward"] = _peak_snapshot(device)
        torch.cuda.reset_peak_memory_stats(device)
        return result

    def measured_retain(loss: Any) -> None:
        original_retain(loss)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    def measured_validate(loss: Any) -> Any:
        result = original_validate(loss)
        torch.cuda.synchronize(device)
        memory["backward"] = _peak_snapshot(device)
        torch.cuda.reset_peak_memory_stats(device)
        return result

    def measured_step(*args: Any, **kwargs: Any) -> Any:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        result = original_step(*args, **kwargs)
        torch.cuda.synchronize(device)
        memory["optimizer_step"] = _peak_snapshot(device)
        return result

    consumer.__dict__["forward_and_loss"] = measured_forward
    consumer.__dict__["_retain_autograd_probes"] = measured_retain
    consumer.__dict__["_validate_autograd"] = measured_validate
    optimizer.__dict__["step"] = measured_step
    try:
        update = consumer.update(batch, global_step=global_step, optimizer=optimizer)
        torch.cuda.synchronize(device)
        memory["post_update"] = _memory_snapshot(device)
    except Exception as error:
        _abort_gpu_error(error, stage=f"gpu_update_step_{global_step}")
    finally:
        consumer.__dict__.pop("forward_and_loss", None)
        consumer.__dict__.pop("_retain_autograd_probes", None)
        consumer.__dict__.pop("_validate_autograd", None)
        optimizer.__dict__.pop("step", None)
    if "forward" not in memory or "backward" not in memory or "optimizer_step" not in memory:
        raise InvalidPreflight("GPU memory instrumentation did not observe all update phases")
    return update, memory


def _update_record(update: Any) -> dict[str, Any]:
    result = update.loss.result
    finite = {
        "component_losses": all(
            bool(torch.isfinite(value))
            for value in (update.loss.image, update.loss.latent, update.loss.optical, update.loss.total)
        ),
        "complex_ols": bool(torch.isfinite(result.z_ols.coefficients).all()),
        "prior": bool(torch.isfinite(update.loss.prior.state.coefficients).all()),
        "raw_cmos": bool(torch.isfinite(result.raw_intensity).all()),
    }
    return {
        "global_step": update.global_step,
        "batch_size": LOGICAL_BATCH_SIZE,
        "loss": update.loss.metrics(),
        "autograd": update.autograd.to_dict(),
        "parameter_update_norms": dict(update.parameter_update_norms),
        "frozen_state_unchanged": update.frozen_state_unchanged,
        "finite": finite,
        "all_finite": all(finite.values()),
    }


def _run_negative_transport_control(contract: Any, device: torch.device) -> dict[str, Any]:
    control = LightGen2DComplexOLSTrainingConsumer(
        contract, device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID
    )
    with torch.no_grad():
        bad_transport = torch.roll(
            torch.eye(100, dtype=torch.complex64, device=device), shifts=1, dims=1
        )
        control.runtime.ols.transport.copy_(bad_transport)
    try:
        control._enforce_parameter_firewall()
    except TrainingContractError as error:
        del control
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "control": "nonidentity coefficient-space transport",
            "expected_rejection": True,
            "rejection_type": type(error).__name__,
            "rejection": str(error),
            "optimizer_step_executed": False,
        }
    raise PreflightAbort(
        "BLOCKED_GPU_NEGATIVE_CONTROL_FAILED",
        "negative_control",
        "the OLS coefficient-transport firewall accepted a nonidentity transport",
    )


def _run_gpu_smoke(
    *,
    contract: Any,
    images: torch.Tensor,
    labels: torch.Tensor,
    train_indices: torch.Tensor,
    development_indices: torch.Tensor,
    selection_indices: torch.Tensor,
    dataset_roles: Mapping[str, Any],
    output_dir: Path,
    starting_git: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    sampler = BalancedTrainRoleSampler(train_indices, labels)
    torch.cuda.synchronize(device)
    memory_before_model = _memory_snapshot(device)
    consumer = LightGen2DComplexOLSTrainingConsumer(
        contract, device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID
    )
    manifest = consumer.parameter_manifest()
    optimizer = consumer.create_optimizer()
    optimizer_manifest = consumer.optimizer_manifest(optimizer)
    initial_parameters = consumer.trainable_state_dict()
    initial_zero = all(bool(torch.equal(value, torch.zeros_like(value))) for value in initial_parameters.values())
    if not initial_zero:
        raise PreflightAbort(
            "BLOCKED_GPU_SMOKE_STRUCTURAL_INVARIANT_FAILURE",
            "gpu_initialization",
            "frozen zero-phase initialization was not preserved",
        )
    torch.cuda.synchronize(device)
    updates: list[dict[str, Any]] = []
    memory_phases: dict[str, Any] = {}
    batch_indices_history: list[torch.Tensor] = []
    checkpoint_path = output_dir / SMOKE_CHECKPOINT_NAME
    checkpoint_payload_value: Mapping[str, Any] | None = None
    continuous_step1_parameters: dict[str, torch.Tensor] | None = None
    continuous_step1_optimizer: Mapping[str, Any] | None = None
    continuous_step1_metrics: Mapping[str, float] | None = None
    after_init = _memory_snapshot(device)

    for global_step in range(3):
        batch_indices, batch_labels = sampler.next_batch()
        if not torch.equal(batch_labels, torch.arange(10, dtype=torch.long)):
            raise PreflightAbort(
                "BLOCKED_DATA_FIREWALL_ROLE_MISMATCH",
                "gpu_smoke",
                "train sampler labels are not metadata-only 0..9 rows",
            )
        batch_indices_history.append(batch_indices.clone())
        batch = images.index_select(0, batch_indices).to(device)
        try:
            if global_step == 0:
                update, measured_memory = _measured_update(
                    consumer, optimizer, batch, global_step=global_step, device=device
                )
                memory_phases = measured_memory
            else:
                update = consumer.update(batch, global_step=global_step, optimizer=optimizer)
                torch.cuda.synchronize(device)
        except Exception as error:
            _abort_gpu_error(error, stage=f"gpu_update_step_{global_step}")
        record = _update_record(update)
        if not record["all_finite"]:
            raise PreflightAbort(
                "BLOCKED_GPU_SMOKE_NONFINITE",
                f"gpu_update_step_{global_step}",
                "a component, complex OLS, prior, or raw CMOS value is nonfinite",
                record=record,
            )
        updates.append(record)
        if global_step == 0:
            torch.cuda.reset_peak_memory_stats(device)
            checkpoint_payload_value = checkpoint_payload(
                consumer,
                optimizer,
                global_step=global_step,
                sampler_state=sampler.state_dict(),
                prior=update.loss.prior,
                dataset_roles=dataset_roles,
                git=starting_git,
            )
            save_smoke_checkpoint(checkpoint_path, checkpoint_payload_value)
            torch.cuda.synchronize(device)
            memory_phases["checkpoint_serialization"] = _peak_snapshot(device)
        if global_step == 1:
            continuous_step1_parameters = consumer.trainable_state_dict()
            continuous_step1_optimizer = copy.deepcopy(optimizer.state_dict())
            continuous_step1_metrics = dict(update.loss.metrics())
        del update, batch
        gc.collect()
        torch.cuda.synchronize(device)
        memory_phases.setdefault("post_release", []).append(_memory_snapshot(device))

    if checkpoint_payload_value is None or continuous_step1_parameters is None or continuous_step1_optimizer is None or continuous_step1_metrics is None:
        raise InvalidPreflight("GPU smoke did not create the required checkpoint/resume reference state")

    dev_batch_indices = selection_indices.reshape(10, 100)[:, 0]
    dev_batch = images.index_select(0, dev_batch_indices).to(device)
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.no_grad():
            dev_loss = consumer.forward_and_loss(dev_batch, global_step=0)
        torch.cuda.synchronize(device)
    except Exception as error:
        _abort_gpu_error(error, stage="development_validation_memory")
    validation_finite = all(
        bool(torch.isfinite(value))
        for value in (dev_loss.image, dev_loss.latent, dev_loss.optical, dev_loss.total)
    )
    if not validation_finite:
        raise PreflightAbort(
            "BLOCKED_GPU_SMOKE_NONFINITE",
            "development_validation_memory",
            "development validation memory smoke produced a nonfinite loss",
        )
    memory_phases["development_validation"] = _peak_snapshot(device)
    del dev_loss, dev_batch, consumer, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    resumed = LightGen2DComplexOLSTrainingConsumer(
        contract, device=device, propagation_grid_size=PRODUCTION_PROPAGATION_GRID
    )
    resumed_optimizer = resumed.create_optimizer()
    try:
        loaded = load_smoke_checkpoint(checkpoint_path, resumed, resumed_optimizer)
    except Exception as error:
        raise PreflightAbort(
            "BLOCKED_CHECKPOINT_RESUME_FAILED",
            "checkpoint_reload",
            "target-device checkpoint reload failed",
            error_type=type(error).__name__,
            error=str(error),
        ) from error
    saved_parameter_state = checkpoint_payload_value["trainable_parameter_state"]
    loaded_parameter_state = resumed.trainable_state_dict()
    if not isinstance(saved_parameter_state, Mapping) or not _tree_equal(saved_parameter_state, loaded_parameter_state):
        raise PreflightAbort(
            "BLOCKED_CHECKPOINT_RESUME_FAILED",
            "checkpoint_reload",
            "trainable parameter state did not round-trip",
        )
    if not _tree_equal(checkpoint_payload_value["optimizer_state"], resumed_optimizer.state_dict()):
        raise PreflightAbort(
            "BLOCKED_CHECKPOINT_RESUME_FAILED",
            "checkpoint_reload",
            "optimizer state did not round-trip",
        )
    saved_rng = checkpoint_payload_value["rng_state"]
    cuda_rng_equal = _tree_equal(saved_rng["cuda"], torch.cuda.get_rng_state_all())
    cpu_rng_equal = torch.equal(saved_rng["cpu"], torch.random.get_rng_state())
    if not cpu_rng_equal or not cuda_rng_equal:
        raise PreflightAbort(
            "BLOCKED_CHECKPOINT_RESUME_FAILED",
            "checkpoint_reload",
            "CPU/CUDA RNG state did not round-trip",
            cpu_rng_equal=cpu_rng_equal,
            cuda_rng_equal=cuda_rng_equal,
        )
    resumed_sampler = BalancedTrainRoleSampler(train_indices, labels)
    resumed_sampler.load_state_dict(loaded.sampler_state)
    resumed_indices, resumed_labels = resumed_sampler.next_batch()
    if not torch.equal(resumed_indices, batch_indices_history[1]) or not torch.equal(resumed_labels, torch.arange(10)):
        raise PreflightAbort(
            "BLOCKED_CHECKPOINT_RESUME_FAILED",
            "checkpoint_sampler",
            "sampler continuation did not round-trip",
        )
    resumed_batch = images.index_select(0, resumed_indices).to(device)
    try:
        resumed_update = resumed.update(resumed_batch, global_step=1, optimizer=resumed_optimizer)
        torch.cuda.synchronize(device)
    except Exception as error:
        raise PreflightAbort(
            "BLOCKED_CHECKPOINT_RESUME_FAILED",
            "checkpoint_continue_step",
            "target-device continuation update failed",
            error_type=type(error).__name__,
            error=str(error),
        ) from error
    resumed_metrics = resumed_update.loss.metrics()
    resumed_parameter_difference = _max_state_difference(
        continuous_step1_parameters, resumed.trainable_state_dict()
    )
    resumed_loss_difference = _max_metric_difference(continuous_step1_metrics, resumed_metrics)
    optimizer_roundtrip_after_continue = _tree_equal(
        continuous_step1_optimizer, resumed_optimizer.state_dict()
    )
    if resumed_loss_difference > 1.0e-5 or resumed_parameter_difference > 1.0e-5 or not optimizer_roundtrip_after_continue:
        raise PreflightAbort(
            "BLOCKED_CHECKPOINT_RESUME_FAILED",
            "checkpoint_resume_equivalence",
            "target-device continuation differs from the continuous reference",
            loss_max_abs_difference=resumed_loss_difference,
            parameter_max_abs_difference=resumed_parameter_difference,
            optimizer_roundtrip_after_continue=optimizer_roundtrip_after_continue,
        )
    checkpoint = {
        "path": str(checkpoint_path),
        "sha256": file_sha256(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
        "schema": "lightgen_2d_semantic_complex_ols_training_smoke_checkpoint.v1",
        "target_device_save_reload": True,
        "smoke_only": True,
        "accepted_training_checkpoint": False,
        "global_step_saved": 0,
        "completed_steps_saved": 1,
        "epoch_saved": 0,
        "parameter_roundtrip": True,
        "optimizer_roundtrip": True,
        "scheduler_roundtrip": loaded.epoch == 0,
        "cpu_rng_roundtrip": cpu_rng_equal,
        "cuda_rng_roundtrip": cuda_rng_equal,
        "prior_rng_roundtrip": _tree_equal(checkpoint_payload_value["prior_rng"], loaded.prior_rng),
        "sampler_roundtrip": True,
        "runtime_identity_validated": True,
        "training_contract_identity_validated": True,
        "resume_equivalence": True,
        "next_step_loss_max_abs_difference": resumed_loss_difference,
        "next_step_parameter_max_abs_difference": resumed_parameter_difference,
        "next_step_optimizer_equivalent": optimizer_roundtrip_after_continue,
    }
    del resumed_update, resumed_batch, resumed, resumed_optimizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)

    total_vram = memory_before_model["total_bytes"]
    phase_peaks = [
        value
        for key, value in memory_phases.items()
        if key in {"forward", "backward", "optimizer_step", "checkpoint_serialization", "development_validation"}
    ]
    peak_reserved = max((int(value["peak_reserved_bytes"]) for value in phase_peaks), default=0)
    peak_allocated = max((int(value["peak_allocated_bytes"]) for value in phase_peaks), default=0)
    minimum_free = min(
        [memory_before_model["free_bytes"], after_init["free_bytes"]]
        + [int(value["free_bytes"]) for value in phase_peaks]
    )
    memory_report = {
        "baseline_before_model": memory_before_model,
        "after_model_and_optimizer_init": after_init,
        "phases": memory_phases,
        "peak_forward_allocation": memory_phases["forward"],
        "peak_backward_allocation": memory_phases["backward"],
        "peak_optimizer_step_allocation": memory_phases["optimizer_step"],
        "peak_reserved_bytes_overall": peak_reserved,
        "peak_allocated_bytes_overall": peak_allocated,
        "minimum_observed_free_bytes": minimum_free,
        "safety_margin_to_total_vram_bytes": total_vram - peak_reserved,
        "exact_batch_size": LOGICAL_BATCH_SIZE,
        "feasible": peak_reserved < total_vram and minimum_free > 0,
    }
    if not memory_report["feasible"]:
        raise PreflightAbort(
            "BLOCKED_BATCH_MEMORY_INFEASIBLE",
            "memory",
            "measured exact-batch peak leaves no device-memory safety margin",
            memory=memory_report,
        )
    return {
        "device_memory_before_model": memory_before_model,
        "parameter_manifest": manifest,
        "optimizer_manifest": optimizer_manifest,
        "initial_zero_phase": initial_zero,
        "runtime_grid": PRODUCTION_PROPAGATION_GRID,
        "updates": updates,
        "three_step_real_update": len(updates) == 3,
        "memory": memory_report,
        "checkpoint": checkpoint,
        "data_batches": {
            "count": len(batch_indices_history),
            "batch_size": LOGICAL_BATCH_SIZE,
            "indices_sha256": [tensor_sha256(value) for value in batch_indices_history],
            "development_validation_batch_size": int(dev_batch_indices.numel()),
            "development_source_indices_sha256": tensor_sha256(development_indices),
        },
    }


def _storage_report(output_dir: Path, checkpoint_size: int) -> dict[str, Any]:
    usage = shutil.disk_usage(output_dir)
    estimated_future_bytes = int(checkpoint_size * 2 + 2 * 1024 * 1024)
    feasible = int(usage.free) > estimated_future_bytes
    report = {
        "output_root": str(output_dir),
        "free_bytes": int(usage.free),
        "total_bytes": int(usage.total),
        "checkpoint_count_required": 2,
        "measured_smoke_checkpoint_bytes": int(checkpoint_size),
        "estimated_future_checkpoint_and_metadata_bytes": estimated_future_bytes,
        "feasible": feasible,
    }
    if not feasible:
        raise PreflightAbort(
            "BLOCKED_STORAGE_INFEASIBLE",
            "storage",
            "available output storage is insufficient for the frozen checkpoint count",
            storage=report,
        )
    return report


def _frozen_schedule(contract_payload: Mapping[str, Any]) -> dict[str, Any]:
    schedule = _mapping(contract_payload.get("training_schedule"), "training schedule")
    return copy.deepcopy(dict(schedule))


def _resolve_output_dir(root: Path, output_dir: Path | None) -> Path:
    target = root / DEFAULT_OUTPUT_DIR if output_dir is None else output_dir
    target = target if target.is_absolute() else root / target
    target = target.resolve()
    outputs_root = (root / "outputs").resolve()
    if not target.is_relative_to(outputs_root):
        raise InvalidPreflight("preflight output must remain under the repository outputs directory")
    if target.exists():
        raise InvalidPreflight(f"refusing to overwrite existing preflight output: {target}")
    return target


def run_preflight(repo_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    root = repo_root.resolve()
    target_dir = _resolve_output_dir(root, output_dir)
    target_dir.mkdir(parents=True, exist_ok=False)
    started = git_facts(root)
    payload: dict[str, Any] = {
        "schema": "lightgen_2d_semantic_complex_ols_long_training_preflight.v1",
        "version": 1,
        "task_id": TASK_ID,
        "preflight_id": TASK_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(target_dir),
        "status": "invalid_harness",
        "result": "INVALID",
        "classification": "INVALID_PREFLIGHT_NOT_COMPLETED",
        "provenance": {"git_start": started},
        "claim_boundary": {
            "target_device_smoke_only": True,
            "long_training_executed": False,
            "scientific_training_executed": False,
            "heldout_evaluation_executed": False,
            "sealed_accessed": False,
            "sealed_reused": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "training_authorized": False,
        },
        "next": None,
    }
    try:
        contract, authorities = _validate_authorities(root)
        payload["authorities"] = authorities
        payload["frozen_schedule"] = _frozen_schedule(contract.payload)
        payload["legacy_exclusion"] = validate_legacy_exclusion()
        images, labels, train_indices, development_indices, selection_indices, dataset = _load_dataset_roles(
            root, contract.payload
        )
        payload["dataset"] = dataset
        device, device_facts = _cuda_device_facts()
        payload["device"] = device_facts
        payload["target_device"] = str(device)
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        gpu = _run_gpu_smoke(
            contract=contract,
            images=images,
            labels=labels,
            train_indices=train_indices,
            development_indices=development_indices,
            selection_indices=selection_indices,
            dataset_roles=dataset,
            output_dir=target_dir,
            starting_git=started,
            device=device,
        )
        payload["gpu_smoke"] = gpu
        payload["negative_control"] = _run_negative_transport_control(contract, device)
        payload["storage"] = _storage_report(target_dir, gpu["checkpoint"]["size_bytes"])
        payload["checkpoint"] = gpu["checkpoint"]
        payload["memory"] = gpu["memory"]
        payload["result"] = "AUTHORIZABLE"
        payload["classification"] = SUCCESS_CLASSIFICATION
        payload["status"] = "valid_completed"
        payload["next"] = "EXECUTE_LIGHTGEN_2D_SEMANTIC_COMPLEX_OLS_TRAINING_V1"
        payload["next_note"] = "requires separate explicit authorization; do not start automatically"
    except PreflightAbort as abort:
        payload.update(abort.details)
        payload["result"] = "BLOCKED"
        payload["classification"] = abort.classification
        payload["status"] = "valid_failed"
        payload["hard_blocker"] = {"stage": abort.stage, "message": str(abort)}
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
        payload["unexpected_error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc()[-8000:],
        }
    finally:
        ending = git_facts(root)
        payload["provenance"]["git_end"] = ending
        payload["provenance"]["branch_unchanged"] = started["branch"] == ending["branch"]
        payload["provenance"]["head_unchanged"] = started["head"] == ending["head"]
        payload["provenance"]["worktree_dirty"] = ending["worktree_dirty"]
        payload["preflight_artifact_self_hash"] = "not_embedded_to_avoid_self_reference"
    artifact_path = target_dir / PREFLIGHT_ARTIFACT_NAME
    atomic_write_new_json(artifact_path, payload, artifact_kind="LightGen long-training preflight")
    result = {
        "result": payload["result"],
        "classification": payload["classification"],
        "preflight_id": TASK_ID,
        "artifact": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "checkpoint": payload.get("checkpoint", {}).get("path"),
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "next": payload["next"],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None, help="new output directory under outputs/")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_preflight(_repo_root(), args.output_dir)
    return 0 if result["result"] in {"AUTHORIZABLE", "BLOCKED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
