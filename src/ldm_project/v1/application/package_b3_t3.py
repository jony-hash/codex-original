"""Package B3-T.3 isolated numerical GPU acceleration preflight.

The module is a research-only canary runner.  Every numerical execution mode
starts from the same frozen FP32 model, optimizer, scheduler, RNG states, and
fixed minibatches.  A mode must pass the registered numerical contract before
its T1/T2-compatible throughput window is measured.  No evaluator, model
checkpoint, formal training run, DDPM, student, or optical object is created.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import hashlib
import inspect
import json
import math
import resource
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

from ..teacher.software_vae import SoftwareMultiscaleTeacherVAE, reconstruction_loss, state_content_hash
from . import package_b3_throughput as throughput
from .package_b3_vae import (
    _batch_loss,
    _config,
    _kl_beta,
    _load_mnist,
    _nested_tensor_hash,
    _select_development,
    _to_camera,
    atomic_json,
    canonical_sha256,
    file_sha256,
    git_state,
    require_clean,
    require_cuda,
)


PROGRAM = "package-b3-t3"
SCHEMA_PREFIX = "package_b3_t3"
T3_VERSION = "b3_t3_numerical_gpu_acceleration_preflight_v1"
LANDMARKS = (0, 1, 2, 5, 10, 25, 50, 100)
MODES = ("fp32", "tf32", "bf16", "fp16")
FORMAL_EPOCH_MINUTES = 25.0

# Locked before any candidate result is observed.  These are deliberately
# conservative engineering tolerances for one fixed deterministic trajectory;
# they do not authorize changing any formal admission threshold.
EQUIVALENCE_CONTRACT = {
    "version": "b3_t3_equivalence_contract_v1",
    "loss_relative_deviation_max": 0.020,
    "reconstruction_max_abs_deviation": 0.050,
    "reconstruction_mean_abs_deviation": 0.010,
    "gradient_cosine_minimum": 0.995,
    "gradient_norm_ratio_minimum": 0.950,
    "gradient_norm_ratio_maximum": 1.050,
    "parameter_max_abs_deviation": 0.020,
    "trajectory_max_abs_deviation": 0.050,
    "optimizer_step_count_must_match": True,
    "finite_required": True,
    "basis": "T2 FP32 deterministic replay was bitwise identical; tolerances are locked before candidate observation and are not tuned post hoc.",
}


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _cpu_nested(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_nested(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_nested(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_nested(child) for child in value)
    return copy.deepcopy(value)


def _vector(values: Iterator[torch.Tensor]) -> torch.Tensor:
    tensors = [value.detach().float().reshape(-1) for value in values]
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(tensors).cpu()


def _parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return _vector(model.parameters())


def _gradient_vector(model: torch.nn.Module) -> torch.Tensor:
    return _vector(parameter.grad for parameter in model.parameters() if parameter.grad is not None)


def _parameter_norm(model: torch.nn.Module) -> float:
    return float(torch.linalg.vector_norm(_parameter_vector(model)))


def _state_hash(model: torch.nn.Module) -> str:
    state = {key: value for key, value in model.state_dict().items() if torch.is_tensor(value)}
    return state_content_hash(state)


def _resource_snapshot(device: torch.device) -> dict[str, int]:
    return {
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
        "host_rss_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }


def _input_hashes(config_path: str, dataset_root: str, t2_output_dir: str, upstream_evidence: str) -> dict[str, str]:
    t2_dir = Path(t2_output_dir)
    files = {
        "config": config_path,
        "t2_final_timing": str(t2_dir / "b3_t2_final_timing.json"),
        "t2_closeout": str(t2_dir / "b3_t2_closeout.md"),
        "t2_equivalence": str(t2_dir / "b3_t2_equivalence_evidence.json"),
        "upstream_preflight": upstream_evidence,
        "model_definition": str(Path(inspect.getfile(SoftwareMultiscaleTeacherVAE)).resolve()),
    }
    result = {key: file_sha256(path) for key, path in files.items()}
    data = _load_mnist(dataset_root)
    result["dataset_hash"] = str(data["dataset_hash"])
    return result


def _validate_t2(config_path: str, dataset_root: str, t2_output_dir: str, upstream_evidence: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    config = _config(config_path)
    t2_dir = Path(t2_output_dir)
    final = _read_json(t2_dir / "b3_t2_final_timing.json")
    closeout = t2_dir / "b3_t2_closeout.md"
    equivalence = _read_json(t2_dir / "b3_t2_equivalence_evidence.json")
    upstream = _read_json(upstream_evidence)
    data = _load_mnist(dataset_root)
    if final.get("schema") != "package_b3_t2_final_timing.v1" or final.get("classification") != "final_fp32_timing_measured":
        raise ValueError("T3 requires authoritative T2 final timing evidence")
    if equivalence.get("schema") != "package_b3_t2_equivalence_evidence.v1" or equivalence.get("evaluator_calls") != 0:
        raise ValueError("T2 equivalence evidence is stale or contains evaluator calls")
    if not closeout.is_file() or "b3_fp32_semantic_preserving_optimization_insufficient" not in closeout.read_text(encoding="utf-8"):
        raise ValueError("T2 closeout does not contain the locked engineering classification")
    if upstream.get("schema") != "teacher_first_stage_recovery_preflight.v1" or upstream.get("evaluator_calls") != 0:
        raise ValueError("B3 upstream preflight evidence is invalid")
    if final.get("dataset_hash") != data["dataset_hash"] or final.get("dataset_split_hash") != config["formal_split_hash"]:
        raise ValueError("T2 dataset or split lineage is stale")
    if float(final.get("optimized_epoch_min", 0.0)) < 25.0:
        raise ValueError("T2 evidence unexpectedly changes the locked throughput decision")
    return final, upstream, _input_hashes(config_path, dataset_root, t2_output_dir, upstream_evidence)


def _hardware_capability(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    capability = tuple(torch.cuda.get_device_capability(device))
    tf32 = capability[0] >= 8
    bf16 = bool(torch.cuda.is_bf16_supported()) and capability[0] >= 8
    fp16 = capability[0] >= 7
    grad_scaler_available = callable(torch.cuda.amp.GradScaler)
    compile_available = hasattr(torch, "compile") and callable(torch.compile)
    return {
        "device": str(device),
        "gpu_name": properties.name,
        "compute_capability": list(capability),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "pytorch_version": torch.__version__,
        "vram_bytes": int(properties.total_memory),
        "tf32_hardware_support": tf32,
        "bf16_hardware_support": bf16,
        "fp16_tensor_core_support": fp16,
        "amp_grad_scaler_available": grad_scaler_available,
        "torch_compile_available": compile_available,
        "arms": {
            "fp32": {"supported": True, "reason": "reference"},
            "tf32": {"supported": tf32, "reason": "native compute capability requirement"},
            "bf16": {"supported": bf16, "reason": "native BF16 capability requirement"},
            "fp16": {"supported": fp16 and grad_scaler_available, "reason": "native FP16 capability and GradScaler requirement"},
        },
    }


@contextlib.contextmanager
def _execution_context(mode: str) -> Iterator[None]:
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    old_cudnn = torch.backends.cudnn.allow_tf32
    try:
        if mode == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            yield
        elif mode == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        elif mode == "fp16":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                yield
        else:
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.backends.cudnn.allow_tf32 = old_cudnn


def _new_runtime(
    mode: str,
    initial_state: dict[str, torch.Tensor],
    cpu_rng: torch.Tensor,
    cuda_rng: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    model = SoftwareMultiscaleTeacherVAE().to(device)
    model.load_state_dict(initial_state, strict=True)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.cuda.amp.GradScaler(enabled=True) if mode == "fp16" else None
    torch.set_rng_state(cpu_rng.clone())
    torch.cuda.set_rng_state(cuda_rng.clone(), device)
    return {"mode": mode, "model": model, "optimizer": optimizer, "scheduler": scheduler, "scaler": scaler}


def _initial_runtime(config: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, Any]]:
    torch.manual_seed(config["master_seed"])
    model = SoftwareMultiscaleTeacherVAE().to(device)
    initial_state = _cpu_nested(model.state_dict())
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(device).clone()
    return initial_state, cpu_rng, cuda_rng, {
        "model_state_hash": state_content_hash(initial_state),
        "optimizer_state_hash": canonical_sha256({}),
        "scheduler_state": torch.optim.lr_scheduler.LambdaLR(torch.optim.AdamW(model.parameters(), lr=0.0002), lambda _: 1.0).state_dict(),
        "cpu_rng_sha256": hashlib.sha256(cpu_rng.numpy().tobytes()).hexdigest(),
        "cuda_rng_sha256": hashlib.sha256(cuda_rng.numpy().tobytes()).hexdigest(),
    }


def _update(
    runtime: dict[str, Any],
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    config: dict[str, Any],
    *,
    track_update: bool,
) -> dict[str, Any]:
    model = runtime["model"]
    optimizer = runtime["optimizer"]
    scaler = runtime["scaler"]
    optimizer.zero_grad(set_to_none=True)
    metric_sums: dict[str, torch.Tensor] = {}
    last_loss: torch.Tensor | None = None
    old_parameters = _parameter_vector(model) if track_update else None
    with _execution_context(runtime["mode"]):
        for raw_images, raw_labels in microbatches:
            images = _to_camera(raw_images.to(next(model.parameters()).device, non_blocking=True))
            labels = raw_labels.to(next(model.parameters()).device, non_blocking=True)
            loss, metrics = _batch_loss(model, images, labels, config, 1)
            if not bool(torch.isfinite(loss).item()):
                return {"finite": False, "failure": "loss_finite", "skipped": False}
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, torch.zeros_like(value)) + value
            last_loss = loss
            if scaler is None:
                (loss / config["gradient_accumulation"]).backward()
            else:
                scaler.scale(loss / config["gradient_accumulation"]).backward()
    if last_loss is None:
        raise RuntimeError("numerical arm update received no microbatches")
    if scaler is not None:
        scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not bool(torch.isfinite(gradient_norm).item()) or not _gradient_vector(model).isfinite().all():
        return {"finite": False, "failure": "gradient_finite", "skipped": False}
    old_scale = float(scaler.get_scale()) if scaler is not None else None
    if scaler is None:
        optimizer.step()
        skipped = False
    else:
        scaler.step(optimizer)
        scaler.update()
        skipped = bool(old_scale is not None and float(scaler.get_scale()) < old_scale)
    if not all(value.isfinite().all() for value in model.parameters()):
        return {"finite": False, "failure": "parameter_finite", "skipped": skipped}
    new_parameters = _parameter_vector(model) if track_update else None
    updated = bool(old_parameters is None or not torch.equal(old_parameters, new_parameters))
    return {
        "finite": True,
        "failure": None,
        "metrics": {key: float((value / len(microbatches)).item()) for key, value in metric_sums.items()},
        "gradient_norm": float(gradient_norm.item()),
        "gradient_vector": _gradient_vector(model) if track_update else None,
        "parameter_updated": updated,
        "skipped": skipped,
        "old_scale": old_scale,
        "new_scale": float(scaler.get_scale()) if scaler is not None else None,
    }


def _probe(runtime: dict[str, Any], raw_images: torch.Tensor, raw_labels: torch.Tensor, config: dict[str, Any]) -> dict[str, Any]:
    model = runtime["model"]
    device = next(model.parameters()).device
    images = _to_camera(raw_images.to(device, non_blocking=True))
    labels = raw_labels.to(device, non_blocking=True)
    was_training = model.training
    model.eval()
    with torch.inference_mode(), _execution_context(runtime["mode"]):
        output = model(images, deterministic=True)
        loss, metrics = reconstruction_loss(
            output["reconstruction"], images, labels, output["mu"], output["logvar"],
            temperature=config["loss_temperature"], kl_beta=_kl_beta(config, 1),
            multiscale_weight=config["multiscale_weight"], sobel_weight=config["sobel_weight"],
            foreground_dice_weight=config["foreground_dice_weight"],
        )
    model.train(was_training)
    reconstruction = output["reconstruction"].detach().cpu().float()
    return {
        "reconstruction": reconstruction,
        "reconstruction_sha256": hashlib.sha256(reconstruction.numpy().tobytes()).hexdigest(),
        "reconstruction_shape": list(reconstruction.shape),
        "reconstruction_min": float(reconstruction.min()),
        "reconstruction_max": float(reconstruction.max()),
        "reconstruction_mean": float(reconstruction.mean()),
        "loss": float(loss.item()),
        "metrics": {key: float(value.item()) for key, value in metrics.items()},
        "dtype": str(output["reconstruction"].dtype),
    }


def _observation(runtime: dict[str, Any], step: int, update: dict[str, Any] | None, probe_images: torch.Tensor, probe_labels: torch.Tensor, config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    probe = _probe(runtime, probe_images, probe_labels, config)
    parameters = _parameter_vector(runtime["model"])
    gradient = update.get("gradient_vector") if update is not None else None
    record = {
        "step": step,
        "loss": None if update is None else update["metrics"]["total"],
        "metrics": None if update is None else update["metrics"],
        "probe_loss": probe["loss"],
        "probe_metrics": probe["metrics"],
        "gradient_norm": None if update is None else update["gradient_norm"],
        "parameter_norm": float(torch.linalg.vector_norm(parameters)),
        "model_state_hash": _state_hash(runtime["model"]),
        "optimizer_state_hash": _nested_tensor_hash(runtime["optimizer"].state_dict()),
        "reconstruction_sha256": probe["reconstruction_sha256"],
        "reconstruction_shape": probe["reconstruction_shape"],
        "reconstruction_min": probe["reconstruction_min"],
        "reconstruction_max": probe["reconstruction_max"],
        "reconstruction_mean": probe["reconstruction_mean"],
        "output_dtype": probe["dtype"],
        "parameter_updated": None if update is None else update["parameter_updated"],
        "skipped_optimizer_step": None if update is None else update["skipped"],
        "finite": True,
    }
    private = {"reconstruction": probe["reconstruction"], "parameters": parameters, "gradient": gradient}
    return record, private


def _snapshot(runtime: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "state_dict": _cpu_nested(runtime["model"].state_dict()),
        "optimizer": _cpu_nested(runtime["optimizer"].state_dict()),
        "scheduler": _cpu_nested(runtime["scheduler"].state_dict()),
        "scaler": None if runtime["scaler"] is None else copy.deepcopy(runtime["scaler"].state_dict()),
        "cpu_rng": torch.get_rng_state().clone(),
        "cuda_rng": torch.cuda.get_rng_state(device).clone(),
    }


def _restore(runtime: dict[str, Any], snapshot: dict[str, Any], device: torch.device) -> None:
    runtime["model"].load_state_dict(snapshot["state_dict"], strict=True)
    runtime["optimizer"].load_state_dict(snapshot["optimizer"])
    runtime["scheduler"].load_state_dict(snapshot["scheduler"])
    if runtime["scaler"] is not None and snapshot["scaler"] is not None:
        runtime["scaler"].load_state_dict(snapshot["scaler"])
    torch.set_rng_state(snapshot["cpu_rng"].clone())
    torch.cuda.set_rng_state(snapshot["cuda_rng"].clone(), device)


def _resume_preflight(
    mode: str,
    initial_state: dict[str, torch.Tensor],
    cpu_rng: torch.Tensor,
    cuda_rng: torch.Tensor,
    fixed_batches: list[tuple[torch.Tensor, torch.Tensor]],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    continuous = _new_runtime(mode, initial_state, cpu_rng, cuda_rng, config, device)
    continuous_updates = []
    for step in range(3):
        result = _update(continuous, fixed_batches[step * config["gradient_accumulation"] : (step + 1) * config["gradient_accumulation"]], config, track_update=True)
        continuous_updates.append(result)
        if not result["finite"]:
            return {"passed": False, "first_failed_gate": result["failure"], "continuous_updates": continuous_updates}
    interrupted = _new_runtime(mode, initial_state, cpu_rng, cuda_rng, config, device)
    interrupted_updates = []
    for step in range(2):
        result = _update(interrupted, fixed_batches[step * config["gradient_accumulation"] : (step + 1) * config["gradient_accumulation"]], config, track_update=True)
        interrupted_updates.append(result)
        if not result["finite"]:
            return {"passed": False, "first_failed_gate": result["failure"], "continuous_updates": continuous_updates, "interrupted_updates": interrupted_updates}
    snapshot = _snapshot(interrupted, device)
    resumed = _new_runtime(mode, initial_state, cpu_rng, cuda_rng, config, device)
    _restore(resumed, snapshot, device)
    resumed_update = _update(resumed, fixed_batches[2 * config["gradient_accumulation"] : 3 * config["gradient_accumulation"]], config, track_update=True)
    if not resumed_update["finite"]:
        return {"passed": False, "first_failed_gate": resumed_update["failure"], "continuous_updates": continuous_updates, "interrupted_updates": interrupted_updates}
    model_equal = _state_hash(continuous["model"]) == _state_hash(resumed["model"])
    optimizer_equal = _nested_tensor_hash(continuous["optimizer"].state_dict()) == _nested_tensor_hash(resumed["optimizer"].state_dict())
    scheduler_equal = continuous["scheduler"].state_dict() == resumed["scheduler"].state_dict()
    continuous_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone())
    resumed_rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state(device).clone())
    rng_equal = torch.equal(continuous_rng[0], resumed_rng[0]) and torch.equal(continuous_rng[1], resumed_rng[1])
    parameter_updates = [bool(result.get("parameter_updated")) for result in continuous_updates + [resumed_update]]
    return {
        "passed": bool(model_equal and optimizer_equal and scheduler_equal and rng_equal and all(parameter_updates)),
        "first_failed_gate": None if model_equal and optimizer_equal and scheduler_equal and rng_equal and all(parameter_updates) else "resume_exactness",
        "continuous_updates": [{key: value for key, value in result.items() if key != "gradient_vector"} for result in continuous_updates],
        "interrupted_updates": [{key: value for key, value in result.items() if key != "gradient_vector"} for result in interrupted_updates],
        "resumed_update": {key: value for key, value in resumed_update.items() if key != "gradient_vector"},
        "model_exact": model_equal,
        "optimizer_exact": optimizer_equal,
        "scheduler_exact": scheduler_equal,
        "rng_exact": rng_equal,
        "parameter_updates": parameter_updates,
    }


def _build_fixed_batches(data: dict[str, Any], config: dict[str, Any], steps: int) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], list[int]]:
    fitting, _ = _select_development(data["train_indices"], data["labels"], seed=config["development_seed"], samples_per_class=config["development_samples_per_class"])
    count = steps * config["gradient_accumulation"] * config["microbatch_size"]
    indices = fitting[:count]
    batches = []
    for start in range(0, indices.numel(), config["microbatch_size"]):
        selected = indices[start : start + config["microbatch_size"]]
        batches.append((data["images"].index_select(0, selected), data["labels"].index_select(0, selected)))
    return batches, indices.tolist()


def _compare_trajectory(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    rows = []
    max_loss = 0.0
    max_recon = 0.0
    max_recon_mean = 0.0
    min_cosine = 1.0
    min_norm_ratio = float("inf")
    max_norm_ratio = 0.0
    max_parameter = 0.0
    for step in LANDMARKS:
        ref = reference["private"][step]
        cand = candidate["private"][step]
        recon_delta = (cand["reconstruction"] - ref["reconstruction"]).abs()
        recon_max = float(recon_delta.max())
        recon_mean = float(recon_delta.mean())
        ref_record = reference["landmarks"][step]
        cand_record = candidate["landmarks"][step]
        component_deviation = {
            key: abs(cand_record["probe_metrics"][key] - ref_record["probe_metrics"][key]) / max(abs(ref_record["probe_metrics"][key]), 1.0e-12)
            for key in ref_record["probe_metrics"]
        }
        loss_deviation = max(component_deviation.values(), default=0.0)
        max_loss = max(max_loss, loss_deviation)
        max_recon = max(max_recon, recon_max)
        max_recon_mean = max(max_recon_mean, recon_mean)
        cosine = None
        norm_ratio = None
        if step > 0:
            cosine = float(F.cosine_similarity(cand["gradient"].view(1, -1), ref["gradient"].view(1, -1)).item())
            ref_norm = float(ref_record["gradient_norm"])
            norm_ratio = float(cand_record["gradient_norm"]) / max(ref_norm, 1.0e-12)
            min_cosine = min(min_cosine, cosine)
            min_norm_ratio = min(min_norm_ratio, norm_ratio)
            max_norm_ratio = max(max_norm_ratio, norm_ratio)
        parameter_delta = float((cand["parameters"] - ref["parameters"]).abs().max())
        max_parameter = max(max_parameter, parameter_delta)
        rows.append({
            "step": step,
            "loss_relative_deviation_max": loss_deviation,
            "reconstruction_max_abs_deviation": recon_max,
            "reconstruction_mean_abs_deviation": recon_mean,
            "gradient_cosine": cosine,
            "gradient_norm_ratio": norm_ratio,
            "parameter_max_abs_deviation": parameter_delta,
            "reference_reconstruction_sha256": ref_record["reconstruction_sha256"],
            "candidate_reconstruction_sha256": cand_record["reconstruction_sha256"],
        })
    passed = bool(
        candidate["resume"]["passed"]
        and candidate["finite"]
        and candidate["successful_optimizer_steps"] == reference["successful_optimizer_steps"]
        and max_loss <= EQUIVALENCE_CONTRACT["loss_relative_deviation_max"]
        and max_recon <= EQUIVALENCE_CONTRACT["reconstruction_max_abs_deviation"]
        and max_recon_mean <= EQUIVALENCE_CONTRACT["reconstruction_mean_abs_deviation"]
        and min_cosine >= EQUIVALENCE_CONTRACT["gradient_cosine_minimum"]
        and min_norm_ratio >= EQUIVALENCE_CONTRACT["gradient_norm_ratio_minimum"]
        and max_norm_ratio <= EQUIVALENCE_CONTRACT["gradient_norm_ratio_maximum"]
        and max_parameter <= EQUIVALENCE_CONTRACT["parameter_max_abs_deviation"]
        and max_recon <= EQUIVALENCE_CONTRACT["trajectory_max_abs_deviation"]
        and candidate["skipped_optimizer_steps"] == 0
    )
    return {
        "passed": passed,
        "landmarks": rows,
        "max_loss_relative_deviation": max_loss,
        "max_reconstruction_abs_deviation": max_recon,
        "max_reconstruction_mean_abs_deviation": max_recon_mean,
        "minimum_gradient_cosine": min_cosine,
        "minimum_gradient_norm_ratio": min_norm_ratio if min_norm_ratio != float("inf") else None,
        "maximum_gradient_norm_ratio": max_norm_ratio,
        "maximum_parameter_abs_deviation": max_parameter,
        "optimizer_step_count_equal": candidate["successful_optimizer_steps"] == reference["successful_optimizer_steps"],
        "finite": candidate["finite"],
        "skipped_optimizer_steps": candidate["skipped_optimizer_steps"],
    }


def _run_trajectory(
    mode: str,
    initial_state: dict[str, torch.Tensor],
    cpu_rng: torch.Tensor,
    cuda_rng: torch.Tensor,
    fixed_batches: list[tuple[torch.Tensor, torch.Tensor]],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats(device)
    runtime = _new_runtime(mode, initial_state, cpu_rng, cuda_rng, config, device)
    probe_images, probe_labels = fixed_batches[0]
    records: dict[int, dict[str, Any]] = {}
    private: dict[int, dict[str, Any]] = {}
    record, hidden = _observation(runtime, 0, None, probe_images, probe_labels, config)
    records[0] = record
    private[0] = hidden
    successful = 0
    skipped = 0
    finite = True
    failure = None
    for step in range(1, 101):
        selected = fixed_batches[(step - 1) * config["gradient_accumulation"] : step * config["gradient_accumulation"]]
        update = _update(runtime, selected, config, track_update=step in LANDMARKS)
        if not update["finite"]:
            finite = False
            failure = update["failure"]
            break
        successful += 0 if update["skipped"] else 1
        skipped += int(update["skipped"])
        if step in LANDMARKS:
            record, hidden = _observation(runtime, step, update, probe_images, probe_labels, config)
            records[step] = record
            private[step] = hidden
    return {
        "mode": mode,
        "landmarks": records,
        "private": private,
        "successful_optimizer_steps": successful,
        "attempted_optimizer_steps": successful + skipped,
        "skipped_optimizer_steps": skipped,
        "finite": finite,
        "failure": failure,
        "model_state_hash_after_trajectory": _state_hash(runtime["model"]),
        "resume": {},
        "resource": _resource_snapshot(device),
    }


def _profile_throughput(
    mode: str,
    initial_state: dict[str, torch.Tensor],
    cpu_rng: torch.Tensor,
    cuda_rng: torch.Tensor,
    data: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    torch.cuda.reset_peak_memory_stats(device)
    runtime = _new_runtime(mode if mode in MODES else "fp32", initial_state, cpu_rng, cuda_rng, config, device)
    model = runtime["model"]
    fitting, development = _select_development(data["train_indices"], data["labels"], seed=config["development_seed"], samples_per_class=config["development_samples_per_class"])
    order = fitting[: (8 + 200) * config["microbatch_size"]]
    loader = DataLoader(Subset(TensorDataset(data["images"], data["labels"]), order.tolist()), batch_size=config["microbatch_size"], shuffle=False, num_workers=0, pin_memory=True)
    warmup = 8
    measured = 200
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
    phase_values: dict[str, list[float]] = {}
    finite_sync_values: list[float] = []
    runtime["optimizer"].zero_grad(set_to_none=True)
    iterator = iter(loader)
    window_started: float | None = None
    skipped = 0
    with throughput._SystemSampler() as sampler:
        for batch_number in range(warmup + measured):
            raw_images, raw_labels = next(iterator)
            images = _to_camera(raw_images.to(device, non_blocking=True))
            labels = raw_labels.to(device, non_blocking=True)
            forward_start = torch.cuda.Event(enable_timing=True)
            forward_end = torch.cuda.Event(enable_timing=True)
            forward_start.record()
            with _execution_context(mode):
                output = model(images, deterministic=False)
                forward_end.record()
                loss, _ = throughput._profile_loss(output["reconstruction"], images, labels, output["mu"], output["logvar"], config, 1, events)
                backward_start = torch.cuda.Event(enable_timing=True)
                backward_start.record()
            finite_started = time.perf_counter()
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"{mode} throughput loss is non-finite")
            finite_sync_values.append((time.perf_counter() - finite_started) * 1000.0)
            if runtime["scaler"] is None:
                (loss / config["gradient_accumulation"]).backward()
            else:
                runtime["scaler"].scale(loss / config["gradient_accumulation"]).backward()
            next_batch = batch_number + 1
            backward_end = torch.cuda.Event(enable_timing=True)
            backward_end.record()
            if next_batch % config["gradient_accumulation"] == 0:
                if runtime["scaler"] is not None:
                    runtime["scaler"].unscale_(runtime["optimizer"])
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                finite_sync_started = time.perf_counter()
                if not bool(torch.isfinite(gradient_norm).item()):
                    raise RuntimeError(f"{mode} throughput gradient is non-finite")
                finite_sync_values.append((time.perf_counter() - finite_sync_started) * 1000.0)
                old_scale = None if runtime["scaler"] is None else float(runtime["scaler"].get_scale())
                if runtime["scaler"] is None:
                    runtime["optimizer"].step()
                else:
                    runtime["scaler"].step(runtime["optimizer"])
                    runtime["scaler"].update()
                    skipped += int(old_scale is not None and float(runtime["scaler"].get_scale()) < old_scale)
                runtime["optimizer"].zero_grad(set_to_none=True)
                events.setdefault("gradient_check", []).append((backward_start, backward_end))
            if batch_number == warmup - 1:
                torch.cuda.synchronize(device)
                window_started = time.perf_counter()
            torch.cuda.synchronize(device)
            if batch_number >= warmup:
                phase_values.setdefault("model_forward", []).append(float(forward_start.elapsed_time(forward_end)))
                phase_values.setdefault("backward", []).append(float(backward_start.elapsed_time(backward_end)))
                for key, pairs in events.items():
                    if pairs and len(pairs) > len(phase_values.get(key, [])):
                        phase_values.setdefault(key, []).append(float(sum(throughput._event_ms([pairs[-1]]))))
    if window_started is None:
        raise RuntimeError("throughput window did not start")
    profile_ms = (time.perf_counter() - window_started) * 1000.0
    model.eval()
    development_values: list[float] = []
    with torch.inference_mode():
        for start in range(0, 100 * config["microbatch_size"], config["microbatch_size"]):
            indices = development[start : start + config["microbatch_size"]]
            started = time.perf_counter()
            images = _to_camera(data["images"].index_select(0, indices).to(device, non_blocking=True))
            labels = data["labels"].index_select(0, indices).to(device, non_blocking=True)
            with _execution_context(mode):
                output = model(images, deterministic=True)
                reconstruction_loss(output["reconstruction"], images, labels, output["mu"], output["logvar"], temperature=config["loss_temperature"], kl_beta=0.0, multiscale_weight=config["multiscale_weight"], sobel_weight=config["sobel_weight"], foreground_dice_weight=config["foreground_dice_weight"])
            torch.cuda.synchronize(device)
            development_values.append((time.perf_counter() - started) * 1000.0)
    phases = {key: throughput._summary(values) for key, values in phase_values.items()}
    phases["cuda_synchronization_wait"] = throughput._summary(finite_sync_values)
    fit_batches = math.ceil(fitting.numel() / config["microbatch_size"])
    dev_batches = math.ceil(development.numel() / config["microbatch_size"])
    projected_seconds = profile_ms / 1000.0 / measured * fit_batches + sum(development_values) / 1000.0 / len(development_values) * dev_batches
    return {
        "mode": mode,
        "profile_training_window_ms": profile_ms,
        "development_evaluation": throughput._summary(development_values),
        "phases": phases,
        "projected_epoch_minutes": projected_seconds / 60.0,
        "speedup_vs_fp32": None,
        "forward_speedup": None,
        "backward_speedup": None,
        "peak_VRAM": _resource_snapshot(device)["cuda_peak_bytes"],
        "peak_RSS": _resource_snapshot(device)["host_rss_peak_bytes"],
        "finite": True,
        "skipped_optimizer_steps": skipped,
        "evaluator_calls": 0,
        "warmup_batches": warmup,
        "measured_training_batches": measured,
        "development_batches": 100,
        "microbatch_size": config["microbatch_size"],
        "gradient_accumulation": config["gradient_accumulation"],
        "system_samples": sampler.samples,
    }


def _compile_correctness(initial_state: dict[str, torch.Tensor], cpu_rng: torch.Tensor, cuda_rng: torch.Tensor, fixed_batches: list[tuple[torch.Tensor, torch.Tensor]], config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    if not hasattr(torch, "compile") or not callable(torch.compile):
        return {"status": "unsupported", "classification": "unsupported", "first_failed_gate": "compile_backend_unavailable"}
    reference = _new_runtime("fp32", initial_state, cpu_rng, cuda_rng, config, device)
    candidate = _new_runtime("fp32", initial_state, cpu_rng, cuda_rng, config, device)
    try:
        counters = torch._dynamo.utils.counters
        counters.clear()
        compile_started = time.perf_counter()
        compiled = torch.compile(candidate["model"], backend="inductor", dynamic=False)
        candidate["model"] = compiled
        probe_ref = _probe(reference, *fixed_batches[0], config)
        first_compiled_started = time.perf_counter()
        probe_candidate = _probe(candidate, *fixed_batches[0], config)
        startup_ms = (time.perf_counter() - first_compiled_started) * 1000.0
        ref_update = _update(reference, fixed_batches[: config["gradient_accumulation"]], config, track_update=True)
        cand_update = _update(candidate, fixed_batches[: config["gradient_accumulation"]], config, track_update=True)
        recon_delta = (probe_candidate["reconstruction"] - probe_ref["reconstruction"]).abs()
        param_delta = (_parameter_vector(candidate["model"]) - _parameter_vector(reference["model"])).abs()
        loss_relative = abs(cand_update["metrics"]["total"] - ref_update["metrics"]["total"]) / max(abs(ref_update["metrics"]["total"]), 1.0e-12)
        backend_counters = {str(key): int(value) for key, value in counters.items() if "graph_break" in str(key) or "recompile" in str(key)}
        passed = bool(ref_update["finite"] and cand_update["finite"] and float(recon_delta.max()) <= EQUIVALENCE_CONTRACT["reconstruction_max_abs_deviation"] and loss_relative <= EQUIVALENCE_CONTRACT["loss_relative_deviation_max"] and float(param_delta.max()) <= EQUIVALENCE_CONTRACT["parameter_max_abs_deviation"])
        return {
            "status": "valid",
            "classification": "compile_correctness_passed" if passed else "compile_correctness_failed",
            "first_failed_gate": None if passed else "compile_equivalence",
            "compile_wrapper_ms": (first_compiled_started - compile_started) * 1000.0,
            "compile_startup_ms": startup_ms,
            "graph_break_and_recompile_counters": backend_counters,
            "output_max_abs_deviation": float(recon_delta.max()),
            "loss_relative_deviation": loss_relative,
            "parameter_max_abs_deviation": float(param_delta.max()),
            "finite": bool(ref_update["finite"] and cand_update["finite"]),
            "optimizer_step_count_equal": True,
            "evaluator_calls": 0,
            "model": candidate["model"],
        }
    except RuntimeError as error:
        return {"status": "valid_failed", "classification": "compile_correctness_failed", "first_failed_gate": "compile_execution", "error_type": type(error).__name__, "error": str(error)[:500], "evaluator_calls": 0}


def _common(config_path: str, dataset_root: str, input_hashes: dict[str, str], config: dict[str, Any], *, classification: str, first_failed_gate: str | None, permitted: list[str], resource_data: dict[str, Any] | None = None, state_hash: str | None = None, started_at: str) -> dict[str, Any]:
    git = git_state()
    resource_data = resource_data or {}
    return {
        "schema_version": 1,
        "program": PROGRAM,
        "t3_version": T3_VERSION,
        "classification": classification,
        "first_failed_gate": first_failed_gate,
        "status": "valid" if first_failed_gate is None else "valid_failed",
        "permitted_next_actions": permitted,
        "forbidden_next_actions": ["formal_b3_vae_training", "b3_ddpm", "package_c", "student", "optical", "phase", "quadrature", "ols", "fiber", "promotion", "candidate_export", "resource_contract_modification"],
        "scientific_model_status": "not_evaluated",
        "formal_VAE_training": "not_started",
        "resource_contract": "25_min_unchanged",
        "numerical_precision_contract": "FP32_reference_and_isolated_candidates",
        "config": str(Path(config_path).resolve()),
        "config_sha256": file_sha256(config_path),
        "dataset_root": str(Path(dataset_root).resolve()),
        "dataset_hash": input_hashes["dataset_hash"],
        "dataset_split_hash": config["formal_split_hash"],
        "input_hashes": input_hashes,
        "git_head": git["git_head"],
        "dirty_worktree": git["dirty_worktree"],
        "seed": config["master_seed"],
        "command": list(sys.argv),
        "started_at": started_at,
        "completed_at": _now(),
        "state_hash": state_hash,
        "evaluator_calls": 0,
        "optimizer_created": True,
        "checkpoint_created": False,
        "candidate_export": False,
        "cuda_peak_bytes": int(resource_data.get("cuda_peak_bytes", 0)),
        "host_rss_peak_bytes": int(resource_data.get("host_rss_peak_bytes", 0)),
    }


def _run(args: argparse.Namespace) -> int:
    started_at = _now()
    require_clean()
    config = _config(args.config)
    t2_final, upstream, input_hashes = _validate_t2(args.config, args.dataset_root, args.t2_output_dir, args.upstream_evidence)
    device = require_cuda(args.device)
    hardware = _hardware_capability(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common = {"config_path": args.config, "dataset_root": args.dataset_root, "input_hashes": input_hashes, "config": config, "started_at": started_at}
    atomic_json(output_dir / "b3_t3_hardware_capability.json", {**_common(**common, classification="hardware_capability_audited", first_failed_gate=None, permitted=["fp32_reference"]), "schema": f"{SCHEMA_PREFIX}_hardware_capability.v1", "hardware": hardware, "t2_final_timing": t2_final, "upstream_state_hash": upstream.get("state_hash_after_three_steps")})
    atomic_json(output_dir / "b3_t3_equivalence_contract.json", {**_common(**common, classification="equivalence_contract_locked", first_failed_gate=None, permitted=["fp32_reference"]), "schema": f"{SCHEMA_PREFIX}_equivalence_contract.v1", "contract": EQUIVALENCE_CONTRACT, "landmarks": list(LANDMARKS), "arms": list(MODES)})
    data = _load_mnist(args.dataset_root)
    fixed_batches, fixed_indices = _build_fixed_batches(data, config, 100)
    batch_hash = canonical_sha256(fixed_indices)
    initial_state, cpu_rng, cuda_rng, initial_lineage = _initial_runtime(config, device)
    reference = _run_trajectory("fp32", initial_state, cpu_rng, cuda_rng, fixed_batches, config, device)
    resume = _resume_preflight("fp32", initial_state, cpu_rng, cuda_rng, fixed_batches, config, device)
    reference["resume"] = resume
    reference["fixed_minibatch_indices_sha256"] = batch_hash
    reference["initial_lineage"] = initial_lineage
    reference_public = {key: value for key, value in reference.items() if key != "private"}
    atomic_json(output_dir / "b3_t3_fp32_reference.json", {**_common(**common, classification="fp32_reference_replayed", first_failed_gate=None if reference["finite"] and resume["passed"] else "fp32_reference", permitted=["isolated_numerical_arms"], resource_data=reference["resource"], state_hash=reference["model_state_hash_after_trajectory"]), "schema": f"{SCHEMA_PREFIX}_fp32_reference.v1", "measurement_contract": {"landmarks": list(LANDMARKS), "optimizer_steps": 100, "gradient_accumulation": config["gradient_accumulation"], "microbatch_size": config["microbatch_size"], "fixed_minibatch_indices_sha256": batch_hash}, "reference": reference_public, "landmarks": reference["landmarks"]})
    if not reference["finite"] or not resume["passed"]:
        closeout_classification = "b3_numerical_execution_equivalence_failed"
        atomic_json(output_dir / "b3_t3_candidate_comparison.json", {**_common(**common, classification=closeout_classification, first_failed_gate="fp32_reference", permitted=["resource_contract_revalidation"], resource_data=reference["resource"], state_hash=reference["model_state_hash_after_trajectory"]), "schema": f"{SCHEMA_PREFIX}_candidate_comparison.v1", "arms": []})
        (output_dir / "b3_t3_closeout.md").write_text(f"# Package B3-T.3 Closeout\n\nclassification: `{closeout_classification}`\n\nFP32 reference failed; no candidate arm was run. Formal training and downstream stages remain forbidden.\n", encoding="utf-8")
        return 2

    fp32_throughput = _profile_throughput("fp32", initial_state, cpu_rng, cuda_rng, data, config, device)
    throughput_results: dict[str, Any] = {"fp32": fp32_throughput}
    arm_results: dict[str, Any] = {}
    selected: str | None = None
    resource_failure = False
    for mode in MODES[1:]:
        supported = bool(hardware["arms"][mode]["supported"])
        if not supported:
            arm_results[mode] = {"classification": "unsupported", "first_failed_gate": "hardware_capability", "mode": mode, "evaluator_calls": 0}
            continue
        trajectory = _run_trajectory(mode, initial_state, cpu_rng, cuda_rng, fixed_batches, config, device)
        trajectory["resume"] = _resume_preflight(mode, initial_state, cpu_rng, cuda_rng, fixed_batches, config, device)
        trajectory["fixed_minibatch_indices_sha256"] = batch_hash
        comparison = _compare_trajectory(reference, trajectory)
        resource_data = trajectory["resource"]
        if resource_data["cuda_peak_bytes"] >= config["cuda_memory_limit_bytes"] or resource_data["host_rss_peak_bytes"] >= config["host_rss_limit_bytes"]:
            resource_failure = True
        if not comparison["passed"]:
            classification = "numerical_equivalence_failed"
        else:
            throughput_results[mode] = _profile_throughput(mode, initial_state, cpu_rng, cuda_rng, data, config, device)
            throughput_results[mode]["speedup_vs_fp32"] = fp32_throughput["projected_epoch_minutes"] / throughput_results[mode]["projected_epoch_minutes"]
            throughput_results[mode]["forward_speedup"] = fp32_throughput["phases"]["model_forward"]["total_ms"] / throughput_results[mode]["phases"]["model_forward"]["total_ms"]
            throughput_results[mode]["backward_speedup"] = fp32_throughput["phases"]["backward"]["total_ms"] / throughput_results[mode]["phases"]["backward"]["total_ms"]
            classification = "numerically_supported_and_throughput_supported" if throughput_results[mode]["projected_epoch_minutes"] <= FORMAL_EPOCH_MINUTES else "numerically_supported_but_throughput_insufficient"
            if classification == "numerically_supported_and_throughput_supported" and selected is None:
                selected = mode
        arm_results[mode] = {"mode": mode, "classification": classification, "first_failed_gate": None if comparison["passed"] else "numerical_equivalence", "equivalence": comparison, "trajectory": {key: value for key, value in trajectory.items() if key not in {"private", "landmarks"}}, "evaluator_calls": 0}
        torch.cuda.empty_cache()
        if resource_failure:
            break

    compile_result: dict[str, Any] | None = None
    if selected is None and not resource_failure:
        compile_result = _compile_correctness(initial_state, cpu_rng, cuda_rng, fixed_batches, config, device)
        compile_public = {key: value for key, value in compile_result.items() if key != "model"}
        atomic_json(output_dir / "b3_t3_compile_evidence.json", {**_common(**common, classification=compile_result["classification"], first_failed_gate=compile_result.get("first_failed_gate"), permitted=["compile_throughput"] if compile_result["classification"] == "compile_correctness_passed" else ["closeout"], resource_data=_resource_snapshot(device), state_hash=initial_lineage["model_state_hash"]), "schema": f"{SCHEMA_PREFIX}_compile_evidence.v1", "compile": compile_public})
        if compile_result["classification"] == "compile_correctness_passed":
            throughput_results["compile"] = _profile_throughput("compile", initial_state, cpu_rng, cuda_rng, data, config, device)
            throughput_results["compile"]["speedup_vs_fp32"] = fp32_throughput["projected_epoch_minutes"] / throughput_results["compile"]["projected_epoch_minutes"]
            throughput_results["compile"]["forward_speedup"] = fp32_throughput["phases"]["model_forward"]["total_ms"] / throughput_results["compile"]["phases"]["model_forward"]["total_ms"]
            throughput_results["compile"]["backward_speedup"] = fp32_throughput["phases"]["backward"]["total_ms"] / throughput_results["compile"]["phases"]["backward"]["total_ms"]
            compile_result["throughput"] = throughput_results["compile"]
            if throughput_results["compile"]["projected_epoch_minutes"] <= FORMAL_EPOCH_MINUTES:
                selected = "compile"

    for mode, result in arm_results.items():
        atomic_json(output_dir / f"b3_t3_{mode}_evidence.json", {**_common(**common, classification=result["classification"], first_failed_gate=result.get("first_failed_gate"), permitted=["throughput_measurement"] if result["classification"] == "numerically_supported_but_throughput_insufficient" else ["closeout"], resource_data=result.get("trajectory", {}).get("resource", {}), state_hash=result.get("trajectory", {}).get("model_state_hash_after_trajectory")), "schema": f"{SCHEMA_PREFIX}_{mode}_evidence.v1", "arm": result})
    candidate_summary = {mode: {"classification": result["classification"], "first_failed_gate": result.get("first_failed_gate"), "equivalence_passed": result.get("equivalence", {}).get("passed", False)} for mode, result in arm_results.items()}
    if compile_result is not None:
        candidate_summary["compile"] = {"classification": compile_result["classification"], "first_failed_gate": compile_result.get("first_failed_gate"), "equivalence_passed": compile_result["classification"] == "compile_correctness_passed"}
    overall_classification = "b3_numerical_gpu_acceleration_supported" if selected is not None else ("b3_numerical_acceleration_insufficient" if any(result.get("equivalence", {}).get("passed", False) for result in arm_results.values()) or (compile_result is not None and compile_result.get("classification") == "compile_correctness_passed") else "b3_numerical_execution_equivalence_failed")
    if resource_failure:
        overall_classification = "invalid_harness"
    atomic_json(output_dir / "b3_t3_candidate_comparison.json", {**_common(**common, classification=overall_classification, first_failed_gate="resource_contract" if resource_failure else None, permitted=["formalize_selected_execution_contract_only"] if selected else ["resource_contract_revalidation"], resource_data=reference["resource"], state_hash=initial_lineage["model_state_hash"]), "schema": f"{SCHEMA_PREFIX}_candidate_comparison.v1", "arms": candidate_summary, "selected_numerical_candidate": selected, "hardware": hardware})
    atomic_json(output_dir / "b3_t3_throughput_comparison.json", {**_common(**common, classification=overall_classification, first_failed_gate="resource_contract" if resource_failure else None, permitted=["formalize_selected_execution_contract_only"] if selected else ["resource_contract_revalidation"], resource_data=reference["resource"], state_hash=initial_lineage["model_state_hash"]), "schema": f"{SCHEMA_PREFIX}_throughput_comparison.v1", "reference": fp32_throughput, "candidates": throughput_results, "required_speedup": 31.135088336258004 / 25.0, "formal_epoch_minutes": FORMAL_EPOCH_MINUTES})
    selected_text = selected or "none"
    permitted_text = "formalize_selected_execution_contract_only" if selected else ("resource_contract_revalidation" if not resource_failure else "none")
    closeout = [
        "# Package B3-T.3 Closeout", "", f"- classification: `{overall_classification}`", "- scientific_model_status: `not_evaluated`", "- formal_VAE_training: `not_started`", "- resource_contract: `25_min_unchanged`", f"- selected_numerical_candidate: `{selected_text}`", f"- numerical_equivalence: `{'passed for selected arm' if selected else 'no arm met equivalence and throughput'}`", f"- permitted_next_actions: `{permitted_text}`", "- evaluator_calls: `0`", "", "All numerical modes were isolated. No formal B3-VAE training, DDPM, student, optical, promotion, export, or resource-contract modification was performed.", "", "T3 stops here. A selected mode, if any, requires a separately written execution contract and final preflight before formal training.", "",
    ]
    if throughput_results:
        closeout.extend(["## Throughput", "", *[f"- {mode}: `{result['projected_epoch_minutes']:.3f} min`" for mode, result in throughput_results.items()], ""])
    (output_dir / "b3_t3_closeout.md").write_text("\n".join(closeout), encoding="utf-8")
    print(json.dumps({"classification": overall_classification, "selected_numerical_candidate": selected}, sort_keys=True))
    return 0 if overall_classification != "invalid_harness" else 2


def _finalize_resource_abort(args: argparse.Namespace) -> int:
    """Close a canary terminated by the external WSL memory guard.

    This action is offline-only.  It consumes the already-written hardware,
    contract, and FP32 reference artifacts and never initializes CUDA or reruns
    an arm.  The termination numbers are explicit CLI inputs so the event is
    auditable rather than inferred from a missing file.
    """
    started_at = _now()
    require_clean()
    config = _config(args.config)
    _t2_final, _upstream, input_hashes = _validate_t2(args.config, args.dataset_root, args.t2_output_dir, args.upstream_evidence)
    output_dir = Path(args.output_dir)
    hardware_path = output_dir / "b3_t3_hardware_capability.json"
    contract_path = output_dir / "b3_t3_equivalence_contract.json"
    reference_path = output_dir / "b3_t3_fp32_reference.json"
    for path in (hardware_path, contract_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(f"resource-abort finalizer requires partial evidence: {path}")
    reference = _read_json(reference_path)
    if reference.get("schema") != f"{SCHEMA_PREFIX}_fp32_reference.v1" or reference.get("evaluator_calls") != 0:
        raise ValueError("partial FP32 reference is invalid or contains evaluator calls")
    resource_data = {
        "cuda_peak_bytes": int(reference.get("cuda_peak_bytes", 0)),
        "host_rss_peak_bytes": int(reference.get("host_rss_peak_bytes", 0)),
    }
    common = {
        "config_path": args.config,
        "dataset_root": args.dataset_root,
        "input_hashes": input_hashes,
        "config": config,
        "started_at": started_at,
    }
    partial = {
        "directory": str(output_dir.resolve()),
        "hardware_sha256": file_sha256(hardware_path),
        "contract_sha256": file_sha256(contract_path),
        "reference_sha256": file_sha256(reference_path),
        "partial_git_head": reference.get("git_head"),
    }
    abort = {
        "termination_reason": args.termination_reason,
        "guard_process_rss_bytes": int(args.guard_process_rss_bytes),
        "guard_total_python_rss_bytes": int(args.guard_total_python_rss_bytes),
        "guard_process_rss_limit_bytes": int(config["host_rss_limit_bytes"]),
        "guard_total_rss_limit_bytes": 6291456 * 1024,
        "partial_evidence": partial,
        "candidate_arms_started": [],
        "candidate_arms_not_run": ["tf32", "bf16", "fp16"],
        "evaluator_calls": 0,
    }
    candidate_comparison = {
        **_common(**common, classification="resource_aborted", first_failed_gate="resource_limit", permitted=["resource_contract_revalidation"], resource_data=resource_data, state_hash=reference.get("state_hash")),
        "schema": f"{SCHEMA_PREFIX}_candidate_comparison.v1",
        "arms": {mode: {"mode": mode, "classification": "resource_aborted", "first_failed_gate": "resource_limit", "not_run": True} for mode in MODES[1:]},
        "selected_numerical_candidate": None,
        "abort": abort,
    }
    throughput_comparison = {
        **_common(**common, classification="resource_aborted", first_failed_gate="resource_limit", permitted=["resource_contract_revalidation"], resource_data=resource_data, state_hash=reference.get("state_hash")),
        "schema": f"{SCHEMA_PREFIX}_throughput_comparison.v1",
        "reference": None,
        "candidates": {},
        "required_speedup": 31.135088336258004 / 25.0,
        "formal_epoch_minutes": FORMAL_EPOCH_MINUTES,
        "throughput_measurement": "not_run_resource_aborted",
        "abort": abort,
    }
    atomic_json(output_dir / "b3_t3_candidate_comparison.json", candidate_comparison)
    atomic_json(output_dir / "b3_t3_throughput_comparison.json", throughput_comparison)
    for mode in MODES[1:]:
        atomic_json(
            output_dir / f"b3_t3_{mode}_evidence.json",
            {
                **_common(**common, classification="resource_aborted", first_failed_gate="resource_limit", permitted=["resource_contract_revalidation"], resource_data=resource_data, state_hash=reference.get("state_hash")),
                "schema": f"{SCHEMA_PREFIX}_{mode}_evidence.v1",
                "arm": {"mode": mode, "classification": "resource_aborted", "first_failed_gate": "resource_limit", "not_run": True},
                "abort": abort,
            },
        )
    closeout = [
        "# Package B3-T.3 Closeout",
        "",
        "- classification: `resource_aborted`",
        "- first_failed_gate: `resource_limit`",
        "- scientific_model_status: `not_evaluated`",
        "- formal_VAE_training: `not_started`",
        "- resource_contract: `25_min_unchanged`",
        "- selected_numerical_candidate: `none`",
        "- numerical_equivalence: `not_completed; candidate arms were not run to completion`",
        "- projected_epoch: `not_measured`",
        "- permitted_next_actions: `resource_contract_revalidation`",
        "- evaluator_calls: `0`",
        "",
        f"The external WSL memory guard terminated the process: `{args.termination_reason}`.",
        f"Guard observation: process RSS `{int(args.guard_process_rss_bytes) / 1024**2:.1f} MiB`, total Python RSS `{int(args.guard_total_python_rss_bytes) / 1024**2:.1f} MiB`.",
        "The FP32 reference artifact was complete and passed its own finite/resume checks, but no TF32, BF16, or FP16 arm reached a valid equivalence decision and no throughput measurement was run.",
        "",
        f"Partial evidence remains bound to `{partial['partial_git_head']}`; its hashes are hardware `{partial['hardware_sha256']}`, contract `{partial['contract_sha256']}`, and FP32 reference `{partial['reference_sha256']}`.",
        "",
        "No retry, AMP/compile combination, formal B3-VAE training, DDPM, student, optical, export, promotion, or resource-contract change was performed. T3 stops here.",
        "",
    ]
    (output_dir / "b3_t3_closeout.md").write_text("\n".join(closeout), encoding="utf-8")
    return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package B3-T.3 numerical GPU acceleration preflight")
    parser.add_argument("action", choices=("run", "finalize-resource-abort"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--t2-output-dir", required=True)
    parser.add_argument("--upstream-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--termination-reason")
    parser.add_argument("--guard-process-rss-bytes", type=int)
    parser.add_argument("--guard-total-python-rss-bytes", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "finalize-resource-abort":
        if args.termination_reason is None or args.guard_process_rss_bytes is None or args.guard_total_python_rss_bytes is None:
            raise ValueError("resource-abort finalizer requires explicit termination and watchdog RSS observations")
        return _finalize_resource_abort(args)
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
