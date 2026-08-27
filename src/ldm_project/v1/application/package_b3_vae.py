"""Package B3 producer: train and freeze an offline software first-stage VAE.

The producer never imports an evaluator and never constructs an optical or
diffusion model.  Formal evaluator calls belong to the v2 consumer after the
reconstruction banks and their hashes are fixed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset

from ..teacher.software_vae import (
    IMAGE_SIZE,
    LATENT_SHAPE,
    MODEL_VERSION,
    SoftwareMultiscaleTeacherVAE,
    reconstruction_loss,
    state_content_hash,
)


PROGRAM = "teacher-first-stage-recovery"
CONFIG_NAME = "teacher_first_stage_recovery_v1"
SCHEMA_PREFIX = "teacher_first_stage_recovery"
STATE_SCHEMA = f"{SCHEMA_PREFIX}_state.v2"
CAMERA_PREPROCESSING = "bilinear_28_to_200_align_corners_false"
CLASS_COUNT = 10
FORMAL_SPLIT_HASH = "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912"
FORBIDDEN_ACTIONS = (
    "ddpm_training",
    "student_training",
    "optical_training",
    "phase_training",
    "quadrature_training",
    "ols_training",
    "fiber_training",
    "candidate_export",
    "promotion",
)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(repr(tuple(value.shape)).encode())
    digest.update(str(value.dtype).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def atomic_torch(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def git_state() -> dict[str, Any]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"git_head": head, "dirty_worktree": bool(status), "git_status": status}


def require_clean() -> dict[str, Any]:
    state = git_state()
    if state["dirty_worktree"]:
        raise RuntimeError("Package B3 scientific stages require a clean worktree")
    return state


def require_cuda(device: str) -> torch.device:
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Package B3 formal producer stages require CUDA")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if os.environ["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8")
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    return torch.device(device)


def _config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("name") != CONFIG_NAME:
        raise ValueError("Package B3 config name is invalid")
    expected = {
        "schema_version": 1,
        "name": CONFIG_NAME,
        "version": 20260827,
        "program": PROGRAM,
        "model_version": MODEL_VERSION,
        "image_size": IMAGE_SIZE,
        "latent_shape": list(LATENT_SHAPE),
        "camera_preprocessing": CAMERA_PREPROCESSING,
        "class_count": 10,
        "formal_split_hash": FORMAL_SPLIT_HASH,
        "development_samples_per_class": 500,
        "development_seed": 49979687,
        "master_seed": 49979687,
        "formal_bank_partition_seed": 32452843,
        "microbatch_size": 16,
        "gradient_accumulation": 4,
        "num_workers": 0,
        "learning_rate": 0.0002,
        "weight_decay": 0.0001,
        "maximum_epochs": 40,
        "minimum_epochs": 8,
        "early_stopping_patience": 3,
        "early_stopping_min_relative_improvement": 0.001,
        "checkpoint_interval_optimizer_steps": 100,
        "projected_epoch_seconds_maximum": 1500,
        "warmup_epochs": 5,
        "kl_beta_maximum": 1.0e-6,
        "kl_warmup_fraction": 0.20,
        "loss_temperature": 0.1,
        "multiscale_weight": 1.0,
        "sobel_weight": 0.5,
        "foreground_dice_weight": 0.5,
        "host_rss_limit_bytes": 5905580032,
        "cuda_memory_limit_bytes": 6442450944,
        "primary_accuracy_minimum": 0.95,
        "minimum_recall_minimum": 0.90,
        "evaluator_agreement_minimum": 0.95,
        "diversity_ratio_minimum": 0.70,
        "diversity_ratio_maximum": 1.30,
        "coverage_ratio_maximum": 1.50,
        "memorization_anomaly_maximum": 0.05,
        "ddpm_training_authorized": False,
        "student_training_authorized": False,
        "optical_training_authorized": False,
        "phase_training_authorized": False,
        "quadrature_training_authorized": False,
        "ols_training_authorized": False,
        "fiber_training_authorized": False,
        "candidate_export_authorized": False,
        "promotion_authorized": False,
        "formal_evaluator_in_producer": False,
        "synthetic_diagnostic_only": True,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ValueError(f"Package B3 config field {key!r} is not registered")
    for key in (
        "package_a_gate_audit",
        "package_a_closeout",
        "package_b2_closeout",
        "primary_evaluator",
        "secondary_evaluator",
        "primary_qualification",
        "secondary_qualification",
    ):
        if not isinstance(value.get(key), str) or not value[key]:
            raise ValueError(f"Package B3 config field {key!r} is missing")
    return value


def _stratified_split(labels: torch.Tensor, *, seed: int = 1234) -> tuple[torch.Tensor, torch.Tensor]:
    if labels.ndim != 1 or labels.numel() != 60000:
        raise ValueError("Package B3 requires the complete 60,000-example MNIST train split")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    train, validation = [], []
    for label in range(CLASS_COUNT):
        matches = torch.nonzero(labels == label, as_tuple=False).flatten()
        shuffled = matches.index_select(0, torch.randperm(matches.numel(), generator=generator))
        validation.append(shuffled[:500])
        train.append(shuffled[500:])
    return torch.cat(train), torch.cat(validation)


def _load_mnist(dataset_root: str) -> dict[str, torch.Tensor | str]:
    from torchvision.datasets import MNIST

    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"MNIST root does not exist: {root}")
    dataset = MNIST(root=str(root), train=True, download=False)
    raw_images = dataset.data.unsqueeze(1).to(torch.float32).div_(255.0)
    labels = dataset.targets.to(torch.long)
    train_indices, formal_indices = _stratified_split(labels)
    if FORMAL_SPLIT_HASH != "8a51217b1ed6833250226255930ab598bf1539e27772ade0188592e5bc0b3912":
        raise AssertionError("registered Package A split hash changed")
    return {
        "images": raw_images,
        "labels": labels,
        "train_indices": train_indices,
        "formal_indices": formal_indices,
        "dataset_hash": canonical_sha256({"images": tensor_sha256(raw_images), "labels": tensor_sha256(labels)}),
        "split_hash": FORMAL_SPLIT_HASH,
    }


def _to_camera(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or tuple(images.shape[1:]) != (1, 28, 28):
        raise ValueError(f"MNIST images must be [N,1,28,28], got {tuple(images.shape)}")
    return F.interpolate(images, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)


def _select_development(
    train_indices: torch.Tensor, labels: torch.Tensor, *, seed: int, samples_per_class: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    development, fitting = [], []
    for label in range(CLASS_COUNT):
        matches = train_indices[labels.index_select(0, train_indices) == label]
        if matches.numel() < samples_per_class:
            raise ValueError(f"not enough class {label} examples for development bank")
        order = matches.index_select(0, torch.randperm(matches.numel(), generator=generator))
        development.append(order[:samples_per_class])
        fitting.append(order[samples_per_class:])
    return torch.cat(fitting), torch.cat(development)


def _max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1024 if sys.platform != "darwin" else 1))


def _base(
    stage: str,
    config: dict[str, Any],
    config_path: str,
    git: dict[str, Any],
    *,
    status: str,
    passed: bool,
    classification: str,
    first_failed_gate: str | None,
    permitted: list[str],
    evaluator_calls: int = 0,
) -> dict[str, Any]:
    return {
        "schema": f"{SCHEMA_PREFIX}_{stage}.v1",
        "stage": stage,
        "status": status,
        "passed": passed,
        "classification": classification,
        "first_failed_gate": first_failed_gate,
        "permitted_next_actions": permitted,
        "forbidden_next_actions": list(FORBIDDEN_ACTIONS),
        "config_sha256": file_sha256(config_path),
        "input_artifact_hashes": {
            key: config[f"{key}_sha256"]
            for key in (
                "package_a_gate_audit",
                "package_a_closeout",
                "package_b2_closeout",
                "primary_evaluator",
                "secondary_evaluator",
                "primary_qualification",
                "secondary_qualification",
            )
        },
        "dataset_split_hash": config["formal_split_hash"],
        "git_head": git["git_head"],
        "dirty_worktree": git["dirty_worktree"],
        "seed": config["master_seed"],
        "evaluator_calls": evaluator_calls,
        "optimizer_created": False,
        "checkpoint_created": False,
        "candidate_export": False,
        "host_rss_peak_bytes": 0,
        "cuda_peak_bytes": 0,
        "synthetic_diagnostic_only": True,
        "termination_reason": None,
        "early_stopping_state": None,
        "actual_last_epoch": None,
        "optimizer_steps": 0,
        "checkpoint_interval_optimizer_steps": config.get("checkpoint_interval_optimizer_steps"),
        "projected_epoch_seconds": None,
    }


def _validate_inputs(config: dict[str, Any], dataset_root: str) -> dict[str, Any]:
    paths = {
        "package_a_gate_audit": config["package_a_gate_audit"],
        "package_a_closeout": config["package_a_closeout"],
        "package_b2_closeout": config["package_b2_closeout"],
        "primary_evaluator": config["primary_evaluator"],
        "secondary_evaluator": config["secondary_evaluator"],
        "primary_qualification": config["primary_qualification"],
        "secondary_qualification": config["secondary_qualification"],
    }
    for role, path in paths.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f"Package B3 lineage input is missing: {role}: {path}")
        expected = config[f"{role}_sha256"]
        if file_sha256(path) != expected:
            raise ValueError(f"Package B3 lineage hash mismatch: {role}")
    gate = read_json(config["package_a_gate_audit"])
    closeout = read_json(config["package_a_closeout"])
    b2 = read_json(config["package_b2_closeout"])
    if gate.get("schema") != "electronic_distribution_recovery_gate_audit.v1" or gate.get("classification") != "gate_feasibility_supported" or not gate.get("passed"):
        raise ValueError("Package A H-GFA evidence is not a passed positive control")
    if closeout.get("schema") != "electronic_distribution_recovery_closeout.v1" or closeout.get("status") != "valid":
        raise ValueError("Package A closeout evidence is invalid")
    if b2.get("schema") != "teacher_failure_attribution_closeout.v1" or b2.get("classification") != "teacher_first_stage_distribution_failed":
        raise ValueError("Package B2 closeout is not the locked first-stage failure")
    data = _load_mnist(dataset_root)
    if data["split_hash"] != config["formal_split_hash"]:
        raise ValueError("MNIST split does not match Package A")
    return {"paths": paths, **data}


def _model(device: torch.device) -> SoftwareMultiscaleTeacherVAE:
    model = SoftwareMultiscaleTeacherVAE().to(device)
    model.train()
    return model


def _kl_beta(config: dict[str, Any], epoch: int) -> float:
    fraction = min(1.0, max(0.0, epoch / max(1, config["maximum_epochs"] * config["kl_warmup_fraction"])))
    return config["kl_beta_maximum"] * fraction


def _batch_loss(model: SoftwareMultiscaleTeacherVAE, images: torch.Tensor, labels: torch.Tensor, config: dict[str, Any], epoch: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(images, deterministic=False)
    loss, metrics = reconstruction_loss(
        output["reconstruction"],
        images,
        labels,
        output["mu"],
        output["logvar"],
        temperature=config["loss_temperature"],
        kl_beta=_kl_beta(config, epoch),
        multiscale_weight=config["multiscale_weight"],
        sobel_weight=config["sobel_weight"],
        foreground_dice_weight=config["foreground_dice_weight"],
    )
    return loss, metrics


def _metric_scalars(metrics: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def _nested_tensor_hash(value: Any) -> str:
    digest = hashlib.sha256()

    def visit(item: Any, path: str) -> None:
        digest.update(path.encode())
        if torch.is_tensor(item):
            digest.update(tensor_sha256(item).encode())
            return
        if isinstance(item, dict):
            for key in sorted(item, key=str):
                visit(item[key], f"{path}.{key}")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}.{index}")
            return
        digest.update(repr(item).encode())

    visit(value, "root")
    return digest.hexdigest()


def _gradient_norm_is_finite(model: SoftwareMultiscaleTeacherVAE) -> bool:
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    return bool(torch.isfinite(gradient_norm).item())


def _early_stopping_update(
    *,
    best_loss: float,
    best_epoch: int,
    no_improvement_epochs: int,
    development_loss: float,
    epoch: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    is_best = development_loss < best_loss
    relative_improvement = (
        float("inf")
        if best_loss == float("inf")
        else (best_loss - development_loss) / max(abs(best_loss), 1.0e-12)
    )
    if is_best:
        best_epoch = epoch
        best_loss = development_loss
        if relative_improvement >= config["early_stopping_min_relative_improvement"]:
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
    else:
        relative_improvement = 0.0
        no_improvement_epochs += 1
    return {
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "no_improvement_epochs": no_improvement_epochs,
        "relative_improvement": relative_improvement,
        "is_best": is_best,
        "should_stop": bool(
            epoch >= config["minimum_epochs"]
            and no_improvement_epochs >= config["early_stopping_patience"]
        ),
    }


def _preflight_update(
    model: SoftwareMultiscaleTeacherVAE,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    config: dict[str, Any],
    epoch: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    optimizer.zero_grad(set_to_none=True)
    last_loss: torch.Tensor | None = None
    last_metrics: dict[str, torch.Tensor] | None = None
    for _ in range(config["gradient_accumulation"]):
        last_loss, last_metrics = _batch_loss(model, images, labels, config, epoch)
        if not bool(torch.isfinite(last_loss).item()):
            raise ValueError("Package B3 preflight loss is non-finite")
        (last_loss / config["gradient_accumulation"]).backward()
    if not _gradient_norm_is_finite(model):
        raise ValueError("Package B3 preflight gradient is non-finite")
    optimizer.step()
    if last_loss is None or last_metrics is None:
        raise RuntimeError("Package B3 preflight update did not execute")
    return last_loss, last_metrics


def _preflight(args: argparse.Namespace, config: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    device = require_cuda(args.device)
    torch.manual_seed(config["master_seed"])
    model = _model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    fitting, _ = _select_development(data["train_indices"], data["labels"], seed=config["development_seed"], samples_per_class=config["development_samples_per_class"])
    batch_indices = fitting[: config["microbatch_size"]]
    images = _to_camera(data["images"].index_select(0, batch_indices).to(device))
    labels = data["labels"].index_select(0, batch_indices).to(device)
    before = state_content_hash(model.state_dict())
    losses = []
    updates = []
    for step in range(2):
        _, metrics = _preflight_update(model, optimizer, images, labels, config, step + 1)
        scheduler.step()
        losses.append(_metric_scalars(metrics))
        updates.append(state_content_hash(model.state_dict()) != before)
        before = state_content_hash(model.state_dict())
    checkpoint = {
        "schema": STATE_SCHEMA,
        "model_version": MODEL_VERSION,
        "config_sha256": file_sha256(args.config),
        "git_head": git_state()["git_head"],
        "phase": "train",
        "epoch": 1,
        "next_batch": 2 * config["gradient_accumulation"],
        "epoch_order": fitting[: 3 * config["gradient_accumulation"] * config["microbatch_size"]].clone(),
        "train_loss_sum": 0.0,
        "train_loss_count": 2,
        "optimizer_steps": 2,
        "state_dict": {key: value.detach().clone() for key, value in model.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "history": [],
        "early_stopping_state": {"best_loss": float("inf"), "best_epoch": 0, "no_improvement_epochs": 0},
        "loader_generator_state": torch.Generator(device="cpu").get_state(),
        "cpu_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
    }
    with tempfile.TemporaryDirectory(prefix="package_b3_preflight_") as temporary:
        checkpoint_path = Path(temporary) / "resume.pt"
        atomic_torch(checkpoint_path, checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    cpu_rng_state = checkpoint["cpu_rng_state"]
    cuda_rng_state = checkpoint["cuda_rng_state"]
    continuation = _model(device)
    continuation.load_state_dict(checkpoint["state_dict"])
    continuation_optimizer = torch.optim.AdamW(continuation.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    continuation_optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    continuation_scheduler = torch.optim.lr_scheduler.LambdaLR(continuation_optimizer, lambda _: 1.0)
    continuation_scheduler.load_state_dict(copy.deepcopy(checkpoint["scheduler"]))
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device)
    resumed_loss, _ = _preflight_update(continuation, continuation_optimizer, images, labels, config, 3)
    continuation_scheduler.step()
    continuous = _model(device)
    continuous.load_state_dict(checkpoint["state_dict"])
    continuous_optimizer = torch.optim.AdamW(continuous.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    continuous_optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    continuous_scheduler = torch.optim.lr_scheduler.LambdaLR(continuous_optimizer, lambda _: 1.0)
    continuous_scheduler.load_state_dict(copy.deepcopy(checkpoint["scheduler"]))
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device)
    continuous_loss, continuous_metrics = _preflight_update(continuous, continuous_optimizer, images, labels, config, 3)
    continuous_scheduler.step()
    losses.append(_metric_scalars(continuous_metrics))
    updates.append(state_content_hash(continuous.state_dict()) != state_content_hash(checkpoint["state_dict"]))
    resume_exact = state_content_hash(continuation.state_dict()) == state_content_hash(continuous.state_dict())
    resume_model_hash = state_content_hash(continuation.state_dict())
    continuous_model_hash = state_content_hash(continuous.state_dict())
    resume_optimizer_hash = _nested_tensor_hash(continuation_optimizer.state_dict())
    continuous_optimizer_hash = _nested_tensor_hash(continuous_optimizer.state_dict())
    resume_scheduler_hash = _nested_tensor_hash(continuation_scheduler.state_dict())
    continuous_scheduler_hash = _nested_tensor_hash(continuous_scheduler.state_dict())
    resume_exact = bool(
        resume_model_hash == continuous_model_hash
        and resume_optimizer_hash == continuous_optimizer_hash
        and resume_scheduler_hash == continuous_scheduler_hash
    )
    after = continuous_model_hash

    del model, optimizer, scheduler, continuation, continuation_optimizer, continuation_scheduler, continuous, continuous_optimizer, continuous_scheduler
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    benchmark_model = _model(device)
    benchmark_optimizer = torch.optim.AdamW(benchmark_model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    benchmark_warmup_batches = 10
    benchmark_batch_count = benchmark_warmup_batches + 200
    benchmark_indices = fitting[: benchmark_batch_count * config["microbatch_size"]]
    for start in range(0, benchmark_warmup_batches * config["microbatch_size"], config["microbatch_size"]):
        batch = benchmark_indices[start : start + config["microbatch_size"]]
        benchmark_images = _to_camera(data["images"].index_select(0, batch).to(device, non_blocking=True))
        benchmark_labels = data["labels"].index_select(0, batch).to(device, non_blocking=True)
        _preflight_update(benchmark_model, benchmark_optimizer, benchmark_images, benchmark_labels, config, 1)
    torch.cuda.synchronize(device)
    train_started = time.perf_counter()
    for start in range(benchmark_warmup_batches * config["microbatch_size"], benchmark_batch_count * config["microbatch_size"], config["microbatch_size"]):
        batch = benchmark_indices[start : start + config["microbatch_size"]]
        benchmark_images = _to_camera(data["images"].index_select(0, batch).to(device, non_blocking=True))
        benchmark_labels = data["labels"].index_select(0, batch).to(device, non_blocking=True)
        _preflight_update(benchmark_model, benchmark_optimizer, benchmark_images, benchmark_labels, config, 1)
    torch.cuda.synchronize(device)
    train_elapsed = time.perf_counter() - train_started

    benchmark_model.eval()
    development_indices = _select_development(data["train_indices"], data["labels"], seed=config["development_seed"], samples_per_class=config["development_samples_per_class"])[1]
    development_batch_count = 100
    development_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, development_batch_count * config["microbatch_size"], config["microbatch_size"]):
            batch = development_indices[start : start + config["microbatch_size"]]
            benchmark_images = _to_camera(data["images"].index_select(0, batch).to(device, non_blocking=True))
            benchmark_labels = data["labels"].index_select(0, batch).to(device, non_blocking=True)
            output = benchmark_model(benchmark_images, deterministic=True)
            reconstruction_loss(output["reconstruction"], benchmark_images, benchmark_labels, output["mu"], output["logvar"], temperature=config["loss_temperature"], kl_beta=_kl_beta(config, 1), multiscale_weight=config["multiscale_weight"], sobel_weight=config["sobel_weight"], foreground_dice_weight=config["foreground_dice_weight"])
    torch.cuda.synchronize(device)
    development_elapsed = time.perf_counter() - development_started
    memory = torch.cuda.max_memory_allocated(device)
    rss = _max_rss_bytes()
    projected_train_seconds = train_elapsed / 200 * math.ceil(fitting.numel() / config["microbatch_size"])
    projected_development_seconds = development_elapsed / development_batch_count * math.ceil(development_indices.numel() / config["microbatch_size"])
    projected_epoch_seconds = projected_train_seconds + projected_development_seconds
    throughput_passed = projected_epoch_seconds <= config["projected_epoch_seconds_maximum"]
    del benchmark_model, benchmark_optimizer
    torch.cuda.empty_cache()
    passed = bool(
        all(updates)
        and bool(torch.isfinite(resumed_loss).item())
        and bool(torch.isfinite(continuous_loss).item())
        and resume_exact
        and throughput_passed
        and memory < config["cuda_memory_limit_bytes"]
        and rss < config["host_rss_limit_bytes"]
        and after != ""
    )
    return {
        "passed": passed,
        "classification": "preflight_passed" if passed else "invalid_harness",
        "first_failed_gate": None if passed else "preflight_contract",
        "permitted_next_actions": ["train"] if passed else [],
        "losses": losses,
        "parameter_updates": updates,
        "resume_exact": resume_exact,
        "state_hash_after_three_steps": after,
        "resume_model_exact": resume_model_hash == continuous_model_hash,
        "resume_optimizer_exact": resume_optimizer_hash == continuous_optimizer_hash,
        "resume_scheduler_exact": resume_scheduler_hash == continuous_scheduler_hash,
        "benchmark_training_batches": benchmark_batch_count,
        "benchmark_development_batches": development_batch_count,
        "benchmark_training_seconds": train_elapsed,
        "benchmark_development_seconds": development_elapsed,
        "projected_train_seconds": projected_train_seconds,
        "projected_development_seconds": projected_development_seconds,
        "projected_epoch_seconds": projected_epoch_seconds,
        "throughput_gate_passed": throughput_passed,
        "cuda_peak_bytes": int(memory),
        "host_rss_peak_bytes": rss,
        "evaluator_calls": 0,
        "optimizer_created": True,
        "checkpoint_created": False,
        "optical_objects_created": False,
        "diffusion_objects_created": False,
        "student_objects_created": False,
    }


def _train(args: argparse.Namespace, config: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    device = require_cuda(args.device)
    torch.manual_seed(config["master_seed"])
    model = _model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    def lr_schedule(epoch: int) -> float:
        if epoch <= config["warmup_epochs"]:
            return max(1.0 / config["warmup_epochs"], epoch / config["warmup_epochs"])
        progress = (epoch - config["warmup_epochs"]) / max(1, config["maximum_epochs"] - config["warmup_epochs"])
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    fitting_indices, development_indices = _select_development(
        data["train_indices"],
        data["labels"],
        seed=config["development_seed"],
        samples_per_class=config["development_samples_per_class"],
    )
    fitting_dataset = TensorDataset(data["images"], data["labels"])
    loader_generator = torch.Generator(device="cpu").manual_seed(config["master_seed"])
    best_loss = float("inf")
    best_epoch = 0
    history = []
    run_dir = Path(args.run_dir)
    state_dir = run_dir / "states"
    config_sha = file_sha256(args.config)
    git_head = git_state()["git_head"]
    current_epoch = 1
    phase = "train"
    next_batch = 0
    epoch_order = torch.empty(0, dtype=torch.long)
    train_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    train_loss_count = 0
    optimizer_steps = 0
    no_improvement_epochs = 0
    resume_source = args.resume_checkpoint

    def ordered_loader(indices: torch.Tensor) -> DataLoader:
        return DataLoader(
            Subset(fitting_dataset, indices.tolist()),
            batch_size=config["microbatch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

    def state_payload(
        *,
        state_phase: str,
        state_epoch: int,
        state_next_batch: int,
        state_epoch_order: torch.Tensor,
        state_train_loss_sum: float,
        state_train_loss_count: int,
    ) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "model_version": MODEL_VERSION,
            "config_sha256": config_sha,
            "git_head": git_head,
            "phase": state_phase,
            "epoch": state_epoch,
            "next_batch": state_next_batch,
            "epoch_order": state_epoch_order.detach().cpu().clone(),
            "train_loss_sum": state_train_loss_sum,
            "train_loss_count": state_train_loss_count,
            "optimizer_steps": optimizer_steps,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "early_stopping_state": {
                "best_loss": best_loss,
                "best_epoch": best_epoch,
                "no_improvement_epochs": no_improvement_epochs,
            },
            "loader_generator_state": loader_generator.get_state(),
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(device),
            "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
            "host_rss_peak_bytes": _max_rss_bytes(),
        }

    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
        if (
            not isinstance(resume, dict)
            or resume.get("schema") != STATE_SCHEMA
            or resume.get("model_version") != MODEL_VERSION
            or resume.get("config_sha256") != config_sha
            or resume.get("git_head") != git_head
        ):
            raise ValueError("Package B3 resume checkpoint lineage is invalid")
        if resume.get("phase") not in {"train", "development"}:
            raise ValueError("Package B3 resume checkpoint phase is invalid")
        model.load_state_dict(resume["state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        if not isinstance(resume.get("history"), list) or not isinstance(resume.get("epoch"), int):
            raise ValueError("Package B3 resume checkpoint metadata is invalid")
        history = resume["history"]
        early_state = resume.get("early_stopping_state")
        if not isinstance(early_state, dict):
            raise ValueError("Package B3 resume checkpoint is missing early-stopping state")
        best_loss = float(early_state["best_loss"])
        best_epoch = int(early_state["best_epoch"])
        no_improvement_epochs = int(early_state["no_improvement_epochs"])
        phase = resume["phase"]
        current_epoch = int(resume["epoch"])
        if current_epoch < 1 or current_epoch > config["maximum_epochs"]:
            raise ValueError("Package B3 resume checkpoint is beyond the registered epoch limit")
        if not isinstance(resume.get("next_batch"), int) or resume["next_batch"] < 0:
            raise ValueError("Package B3 resume checkpoint batch cursor is invalid")
        next_batch = int(resume["next_batch"])
        train_loss_sum = torch.tensor(float(resume["train_loss_sum"]), dtype=torch.float64, device=device)
        train_loss_count = int(resume["train_loss_count"])
        optimizer_steps = int(resume["optimizer_steps"])
        if not torch.is_tensor(resume.get("epoch_order")):
            raise ValueError("Package B3 resume checkpoint is missing epoch order")
        epoch_order = resume["epoch_order"].to(device="cpu", dtype=torch.long)
        if phase == "train":
            expected_batches = math.ceil(fitting_indices.numel() / config["microbatch_size"])
            if epoch_order.numel() not in {0, fitting_indices.numel()} or next_batch > expected_batches:
                raise ValueError("Package B3 resume checkpoint epoch cursor is invalid")
            if epoch_order.numel() and next_batch % config["gradient_accumulation"]:
                raise ValueError("Package B3 resume checkpoint is not at an accumulation boundary")
            if not epoch_order.numel() and next_batch:
                raise ValueError("Package B3 empty epoch order has a nonzero cursor")
        elif next_batch != 0 or epoch_order.numel() != 0:
            raise ValueError("Package B3 development checkpoint cursor is invalid")
        if not torch.is_tensor(resume.get("loader_generator_state")) or not torch.is_tensor(resume.get("cpu_rng_state")) or not torch.is_tensor(resume.get("cuda_rng_state")):
            raise ValueError("Package B3 resume checkpoint is missing RNG state")
        loader_generator.set_state(resume["loader_generator_state"])
        torch.set_rng_state(resume["cpu_rng_state"])
        torch.cuda.set_rng_state(resume["cuda_rng_state"], device)

    def memory_gate() -> None:
        if (
            torch.cuda.max_memory_allocated(device) >= config["cuda_memory_limit_bytes"]
            or _max_rss_bytes() >= config["host_rss_limit_bytes"]
        ):
            raise RuntimeError("Package B3 training exceeded the registered memory safety limit")

    while phase in {"train", "development"} and current_epoch <= config["maximum_epochs"]:
        if phase == "train":
            if epoch_order.numel() == 0:
                epoch_order = fitting_indices.index_select(
                    0,
                    torch.randperm(fitting_indices.numel(), generator=loader_generator),
                )
                next_batch = 0
                train_loss_sum.zero_()
                train_loss_count = 0
            total_batches = math.ceil(epoch_order.numel() / config["microbatch_size"])
            if next_batch > total_batches:
                raise ValueError("Package B3 training cursor exceeds epoch length")
            loader = ordered_loader(epoch_order[next_batch * config["microbatch_size"] :])
            model.train()
            for local_batch_number, (batch_images, batch_labels) in enumerate(loader):
                batch_number = next_batch + local_batch_number
                batch_images = _to_camera(batch_images.to(device, non_blocking=True))
                batch_labels = batch_labels.to(device, non_blocking=True)
                loss, metrics = _batch_loss(model, batch_images, batch_labels, config, current_epoch)
                if not bool(torch.isfinite(loss).item()):
                    raise ValueError("Package B3 training loss is non-finite")
                (loss / config["gradient_accumulation"]).backward()
                train_loss_sum.add_(metrics["total"].double())
                train_loss_count += 1
                next_batch = batch_number + 1
                is_last_batch = next_batch == total_batches
                if next_batch % config["gradient_accumulation"] == 0 or is_last_batch:
                    if not _gradient_norm_is_finite(model):
                        raise ValueError("Package B3 training gradient is non-finite")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_steps += 1
                    if optimizer_steps % config["checkpoint_interval_optimizer_steps"] == 0:
                        memory_gate()
                        atomic_torch(
                            state_dir / "latest.pt",
                            state_payload(
                                state_phase="train",
                                state_epoch=current_epoch,
                                state_next_batch=next_batch,
                                state_epoch_order=epoch_order,
                                state_train_loss_sum=float(train_loss_sum.cpu()),
                                state_train_loss_count=train_loss_count,
                            ),
                        )
            if next_batch != total_batches:
                raise RuntimeError("Package B3 training did not consume the complete epoch")
            scheduler.step()
            memory_gate()
            atomic_torch(
                state_dir / "latest.pt",
                state_payload(
                    state_phase="development",
                    state_epoch=current_epoch,
                    state_next_batch=0,
                    state_epoch_order=torch.empty(0, dtype=torch.long),
                    state_train_loss_sum=float(train_loss_sum.cpu()),
                    state_train_loss_count=train_loss_count,
                ),
            )
            phase = "development"

        model.train()
        model.eval()
        development_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        development_batch_count = 0
        with torch.inference_mode():
            for start in range(0, development_indices.numel(), config["microbatch_size"]):
                indices = development_indices[start : start + config["microbatch_size"]]
                images = _to_camera(data["images"].index_select(0, indices).to(device, non_blocking=True))
                labels = data["labels"].index_select(0, indices).to(device, non_blocking=True)
                output = model(images, deterministic=True)
                loss, _ = reconstruction_loss(output["reconstruction"], images, labels, output["mu"], output["logvar"], temperature=config["loss_temperature"], kl_beta=_kl_beta(config, current_epoch), multiscale_weight=config["multiscale_weight"], sobel_weight=config["sobel_weight"], foreground_dice_weight=config["foreground_dice_weight"])
                if not bool(torch.isfinite(loss).item()):
                    raise ValueError("Package B3 development loss is non-finite")
                development_loss_sum.add_(loss.detach().double())
                development_batch_count += 1
        dev_loss = float((development_loss_sum / development_batch_count).cpu())
        train_loss = float((train_loss_sum / train_loss_count).cpu())
        row = {"epoch": current_epoch, "train_loss": train_loss, "development_loss": dev_loss, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        early_state = _early_stopping_update(
            best_loss=best_loss,
            best_epoch=best_epoch,
            no_improvement_epochs=no_improvement_epochs,
            development_loss=dev_loss,
            epoch=current_epoch,
            config=config,
        )
        best_loss = early_state["best_loss"]
        best_epoch = early_state["best_epoch"]
        no_improvement_epochs = early_state["no_improvement_epochs"]
        relative_improvement = early_state["relative_improvement"]
        is_best = early_state["is_best"]
        should_stop = early_state["should_stop"]
        memory_gate()
        next_phase = "complete" if should_stop or current_epoch == config["maximum_epochs"] else "train"
        checkpoint_payload = state_payload(
            state_phase=next_phase,
            state_epoch=current_epoch if next_phase == "complete" else current_epoch + 1,
            state_next_batch=0,
            state_epoch_order=torch.empty(0, dtype=torch.long),
            state_train_loss_sum=0.0,
            state_train_loss_count=0,
        )
        checkpoint_payload["early_stopping_state"]["relative_improvement"] = relative_improvement
        checkpoint_payload["early_stopping_state"]["stopped"] = should_stop
        atomic_torch(state_dir / "latest.pt", checkpoint_payload)
        if is_best:
            atomic_torch(
                run_dir / "first_stage_best.pt",
                {**checkpoint_payload, "selection": "development_composite_loss", "best_development_loss": best_loss},
            )
        stopped_by_early_stopping = should_stop and current_epoch < config["maximum_epochs"]
        if should_stop or current_epoch == config["maximum_epochs"]:
            phase = "complete"
        else:
            current_epoch += 1
            epoch_order = torch.empty(0, dtype=torch.long)
            next_batch = 0
            train_loss_sum.zero_()
            train_loss_count = 0
            phase = "train"
    if best_epoch == 0:
        raise RuntimeError("Package B3 did not produce a development checkpoint")
    return {
        "best_epoch": best_epoch,
        "best_development_loss": best_loss,
        "history": history,
        "checkpoint": str(run_dir / "first_stage_best.pt"),
        "evaluator_calls": 0,
        "host_rss_peak_bytes": _max_rss_bytes(),
        "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
        "start_epoch": int(history[0]["epoch"]) if history else current_epoch,
        "actual_last_epoch": history[-1]["epoch"],
        "optimizer_steps": optimizer_steps,
        "termination_reason": "early_stopping" if stopped_by_early_stopping else "maximum_epochs",
        "early_stopping_state": {
            "minimum_epochs": config["minimum_epochs"],
            "patience": config["early_stopping_patience"],
            "minimum_relative_improvement": config["early_stopping_min_relative_improvement"],
            "best_epoch": best_epoch,
            "best_development_loss": best_loss,
            "no_improvement_epochs": no_improvement_epochs,
        },
        "checkpoint_interval_optimizer_steps": config["checkpoint_interval_optimizer_steps"],
        "resume_checkpoint": resume_source,
        "resume_contract": STATE_SCHEMA,
    }


def _load_checkpoint(path: str | Path, device: torch.device) -> tuple[SoftwareMultiscaleTeacherVAE, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("model_version") != MODEL_VERSION or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Package B3 checkpoint payload is invalid")
    model = _model(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def _latent_statistics(
    model: SoftwareMultiscaleTeacherVAE,
    data: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    fitting_indices, _ = _select_development(
        data["train_indices"],
        data["labels"],
        seed=config["development_seed"],
        samples_per_class=config["development_samples_per_class"],
    )
    total = 0
    sum_value = torch.zeros(LATENT_SHAPE[0], dtype=torch.float64, device=device)
    sum_square = torch.zeros_like(sum_value)
    with torch.inference_mode():
        for start in range(0, fitting_indices.numel(), config["microbatch_size"]):
            indices = fitting_indices[start : start + config["microbatch_size"]]
            batch = _to_camera(data["images"].index_select(0, indices).to(device))
            output = model.encode(batch)
            values = output["mu"].double().flatten(2)
            sum_value += values.sum(dim=(0, 2))
            sum_square += values.square().sum(dim=(0, 2))
            total += values.shape[0] * values.shape[2]
    if total <= 1:
        raise ValueError("Package B3 latent statistics require at least two values")
    mean = sum_value / total
    variance = (sum_square / total - mean.square()).clamp_min(1.0e-12)
    return mean.cpu().float(), variance.sqrt().cpu().float(), total


def _formal_banks(data: dict[str, Any], config: dict[str, Any]) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    generator = torch.Generator(device="cpu").manual_seed(config["formal_bank_partition_seed"])
    images, labels = data["images"], data["labels"]
    banks = [[[], []] for _ in range(2)]
    for label in range(CLASS_COUNT):
        matches = data["formal_indices"][labels.index_select(0, data["formal_indices"]) == label]
        order = matches.index_select(0, torch.randperm(matches.numel(), generator=generator))
        for bank, selected in enumerate((order[:250], order[250:])):
            banks[bank][0].append(selected)
            banks[bank][1].append(torch.full((250,), label, dtype=torch.long))
    result = []
    for bank, (index_rows, label_rows) in enumerate(banks):
        indices = torch.cat(index_rows)
        bank_images = images.index_select(0, indices)
        bank_labels = torch.cat(label_rows)
        result.append((bank, bank_images, bank_labels, indices))
    return result


def _reconstruct(args: argparse.Namespace, config: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    device = require_cuda(args.device)
    checkpoint = Path(args.checkpoint)
    expected = config.get("candidate_checkpoint_sha256")
    if expected is not None and file_sha256(checkpoint) != expected:
        raise ValueError("candidate checkpoint hash does not match locked config")
    model, _checkpoint_payload = _load_checkpoint(checkpoint, device)
    before = state_content_hash(model.state_dict())
    run_dir = Path(args.run_dir)
    shard_dir = run_dir / "reconstruction_shards"
    shards = []
    latent_rows = []
    for bank, images, labels, source_indices in _formal_banks(data, config):
        for label in range(CLASS_COUNT):
            selected = torch.nonzero(labels == label, as_tuple=False).flatten()
            for start in range(0, selected.numel(), config["microbatch_size"]):
                chosen = selected[start : start + config["microbatch_size"]]
                with torch.inference_mode():
                    output = model(_to_camera(images.index_select(0, chosen).to(device)), deterministic=True)
                reconstruction = output["reconstruction"].cpu().float()
                mu = output["mu"].cpu().float()
                if not torch.isfinite(reconstruction).all() or not torch.isfinite(mu).all():
                    raise ValueError("Package B3 reconstruction is non-finite")
                path = shard_dir / f"bank_{bank}_class_{label}_start_{start}.pt"
                atomic_torch(path, {"schema": f"{SCHEMA_PREFIX}_reconstruction_shard.v1", "bank": bank, "label": label, "start": start, "end": start + chosen.numel(), "images": reconstruction, "labels": labels.index_select(0, chosen), "source_indices": source_indices.index_select(0, chosen).tolist(), "mu": mu})
                shards.append({"bank": bank, "label": label, "start": start, "end": start + chosen.numel(), "path": str(path), "sha256": file_sha256(path)})
                latent_rows.append(mu)
    after = state_content_hash(model.state_dict())
    if before != after:
        raise ValueError("frozen Package B3 candidate changed during reconstruction")
    latent = torch.cat(latent_rows)
    latent_mean, latent_std, latent_count = _latent_statistics(model, data, config, device)
    stats_path = run_dir / "latent_stats.pt"
    atomic_torch(stats_path, {"schema": f"{SCHEMA_PREFIX}_latent_stats.v1", "model_version": MODEL_VERSION, "latent_shape": list(LATENT_SHAPE), "mean": latent_mean, "std": latent_std, "count": latent_count})
    manifest = {"schema": f"{SCHEMA_PREFIX}_reconstruction.v1", "status": "first_stage_reconstruction_bank", "model_version": MODEL_VERSION, "camera_preprocessing": CAMERA_PREPROCESSING, "checkpoint_sha256": file_sha256(checkpoint), "checkpoint": str(checkpoint), "dataset_hash": data["dataset_hash"], "dataset_split_hash": data["split_hash"], "formal_bank_partition_seed": config["formal_bank_partition_seed"], "bank_count": 2, "samples_per_class": 250, "image_shape": [1, IMAGE_SIZE, IMAGE_SIZE], "latent_shape": list(LATENT_SHAPE), "decoder_input": "encoded_real_mu", "latent_standardization": "training_fit_mu_channelwise", "latent_stats_path": str(stats_path), "latent_stats_sha256": file_sha256(stats_path), "latent_stats_count": latent_count, "evaluator_calls": 0, "state_hash_before": before, "state_hash_after": after, "shards": shards, "latent_mu_sha256": tensor_sha256(latent)}
    atomic_json(args.output, manifest)
    return {"manifest": manifest, "evaluator_calls": 0, "host_rss_peak_bytes": _max_rss_bytes(), "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device)), "state_hash_unchanged": before == after}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package B3 software first-stage producer")
    parser.add_argument("action", choices=("lineage", "preflight", "train", "reconstruct"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--upstream-evidence")
    parser.add_argument("--resume-checkpoint")
    return parser


def dispatch(args: argparse.Namespace) -> int:
    config = _config(args.config)
    git = require_clean()
    data = _validate_inputs(config, args.dataset_root)
    if args.action == "lineage":
        payload = _base("lineage", config, args.config, git, status="valid", passed=True, classification="lineage_validated", first_failed_gate=None, permitted=["preflight"])
        payload.update({"model_version": MODEL_VERSION, "image_size": IMAGE_SIZE, "latent_shape": list(LATENT_SHAPE), "camera_preprocessing": CAMERA_PREPROCESSING, "dataset_hash": data["dataset_hash"], "dataset_split_hash": data["split_hash"], "formal_indices_hash": tensor_sha256(data["formal_indices"]), "train_indices_hash": tensor_sha256(data["train_indices"]), "evaluator_calls": 0})
        atomic_json(args.output, payload)
        return 0
    if args.action == "preflight":
        if not args.upstream_evidence:
            raise ValueError("preflight requires --upstream-evidence")
        upstream = read_json(args.upstream_evidence)
        if upstream.get("schema") != f"{SCHEMA_PREFIX}_lineage.v1" or not upstream.get("passed"):
            raise ValueError("preflight requires passed lineage evidence")
        result = _preflight(args, config, data)
        payload = _base("preflight", config, args.config, git, status="valid" if result["passed"] else "invalid_harness", passed=result["passed"], classification=result["classification"], first_failed_gate=result["first_failed_gate"], permitted=result["permitted_next_actions"])
        payload.update(result)
        payload["upstream_evidence_sha256"] = file_sha256(args.upstream_evidence)
        atomic_json(args.output, payload)
        return 0
    if args.action == "train":
        if not args.upstream_evidence:
            raise ValueError("train requires --upstream-evidence")
        upstream = read_json(args.upstream_evidence)
        if upstream.get("schema") != f"{SCHEMA_PREFIX}_preflight.v1" or not upstream.get("passed"):
            raise ValueError("train requires passed preflight evidence")
        result = _train(args, config, data)
        payload = _base("training", config, args.config, git, status="valid", passed=True, classification="candidate_trained", first_failed_gate=None, permitted=["reconstruct"])
        payload.update(result)
        payload["checkpoint_created"] = True
        payload["upstream_evidence_sha256"] = file_sha256(args.upstream_evidence)
        atomic_json(args.output, payload)
        return 0
    if args.action == "reconstruct":
        if not args.upstream_evidence or not args.checkpoint:
            raise ValueError("reconstruct requires --upstream-evidence and --checkpoint")
        upstream = read_json(args.upstream_evidence)
        if upstream.get("schema") != f"{SCHEMA_PREFIX}_training.v1" or upstream.get("classification") != "candidate_trained":
            raise ValueError("reconstruct requires training evidence")
        result = _reconstruct(args, config, data)
        payload = _base("reconstruction", config, args.config, git, status="valid", passed=True, classification="reconstruction_bank_fixed", first_failed_gate=None, permitted=["audit"])
        payload.update(result)
        payload["checkpoint_created"] = False
        payload["upstream_evidence_sha256"] = file_sha256(args.upstream_evidence)
        atomic_json(args.output, payload)
        return 0
    raise ValueError(f"unknown Package B3 producer action: {args.action}")


def main(argv: list[str] | None = None) -> int:
    return dispatch(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
