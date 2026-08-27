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
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

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
    return F.interpolate(images, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False, antialias=True)


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


def _batch_loss(model: SoftwareMultiscaleTeacherVAE, images: torch.Tensor, labels: torch.Tensor, config: dict[str, Any], epoch: int) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(images, deterministic=False)
    loss, metrics = reconstruction_loss(
        output["reconstruction"],
        images,
        labels,
        output["mu"],
        output["logvar"],
        temperature=config["loss_temperature"],
        kl_beta=_kl_beta(config, epoch),
    )
    return loss, metrics


def _preflight(args: argparse.Namespace, config: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    device = require_cuda(args.device)
    torch.manual_seed(config["master_seed"])
    model = _model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    fitting, _ = _select_development(data["train_indices"], data["labels"], seed=config["development_seed"], samples_per_class=config["development_samples_per_class"])
    batch_indices = fitting[: config["microbatch_size"]]
    images = _to_camera(data["images"].index_select(0, batch_indices)).to(device)
    labels = data["labels"].index_select(0, batch_indices).to(device)
    before = state_content_hash(model.state_dict())
    losses = []
    updates = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = _batch_loss(model, images, labels, config, step + 1)
        if not torch.isfinite(loss):
            raise ValueError("Package B3 preflight loss is non-finite")
        loss.backward()
        finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        if not finite_gradients:
            raise ValueError("Package B3 preflight gradient is non-finite")
        optimizer.step()
        scheduler.step()
        losses.append(metrics)
        updates.append(state_content_hash(model.state_dict()) != before)
        before = state_content_hash(model.state_dict())
    after = state_content_hash(model.state_dict())
    checkpoint = {
        "schema": f"{SCHEMA_PREFIX}_preflight_state.v1",
        "model_version": MODEL_VERSION,
        "state_dict": {key: value.detach().clone() for key, value in model.state_dict().items()},
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "next_step": 3,
    }
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state(device)
    continuation = _model(device)
    continuation.load_state_dict(checkpoint["state_dict"])
    continuation_optimizer = torch.optim.AdamW(continuation.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    continuation_optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    continuation_scheduler = torch.optim.lr_scheduler.LambdaLR(continuation_optimizer, lambda _: 1.0)
    continuation_scheduler.load_state_dict(copy.deepcopy(checkpoint["scheduler"]))
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device)
    continuation_optimizer.zero_grad(set_to_none=True)
    resumed_loss, _ = _batch_loss(continuation, images, labels, config, 4)
    resumed_loss.backward()
    continuation_optimizer.step()
    continuation_scheduler.step()
    continuous = _model(device)
    continuous.load_state_dict(checkpoint["state_dict"])
    continuous_optimizer = torch.optim.AdamW(continuous.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    continuous_optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    continuous_scheduler = torch.optim.lr_scheduler.LambdaLR(continuous_optimizer, lambda _: 1.0)
    continuous_scheduler.load_state_dict(copy.deepcopy(checkpoint["scheduler"]))
    torch.set_rng_state(cpu_rng_state)
    torch.cuda.set_rng_state(cuda_rng_state, device)
    continuous_optimizer.zero_grad(set_to_none=True)
    continuous_loss, _ = _batch_loss(continuous, images, labels, config, 4)
    continuous_loss.backward()
    continuous_optimizer.step()
    continuous_scheduler.step()
    resume_exact = state_content_hash(continuation.state_dict()) == state_content_hash(continuous.state_dict())
    memory = torch.cuda.max_memory_allocated(device)
    rss = _max_rss_bytes()
    passed = bool(
        all(updates)
        and torch.isfinite(resumed_loss)
        and resume_exact
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
    fitting_indices, development_indices = _select_development(data["train_indices"], data["labels"], seed=config["development_seed"], samples_per_class=config["development_samples_per_class"])
    fitting_images = data["images"].index_select(0, fitting_indices)
    fitting_labels = data["labels"].index_select(0, fitting_indices)
    development_images = data["images"].index_select(0, development_indices)
    development_labels = data["labels"].index_select(0, development_indices)
    loader = DataLoader(TensorDataset(fitting_images, fitting_labels), batch_size=config["microbatch_size"], shuffle=True, generator=torch.Generator().manual_seed(config["master_seed"]), num_workers=0, pin_memory=True)
    best_loss = float("inf")
    best_epoch = 0
    history = []
    run_dir = Path(args.run_dir)
    state_dir = run_dir / "states"
    for epoch in range(1, config["maximum_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_losses = []
        for batch_number, (batch_images, batch_labels) in enumerate(loader):
            batch_images = _to_camera(batch_images).to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            loss, metrics = _batch_loss(model, batch_images, batch_labels, config, epoch)
            (loss / config["gradient_accumulation"]).backward()
            if (batch_number + 1) % config["gradient_accumulation"] == 0 or batch_number + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_losses.append(metrics["total"])
        scheduler.step()
        model.eval()
        development_losses = []
        with torch.inference_mode():
            for start in range(0, len(development_images), config["microbatch_size"]):
                images = _to_camera(development_images[start : start + config["microbatch_size"]]).to(device)
                labels = development_labels[start : start + config["microbatch_size"]].to(device)
                output = model(images, deterministic=True)
                loss, _ = reconstruction_loss(output["reconstruction"], images, labels, output["mu"], output["logvar"], temperature=config["loss_temperature"], kl_beta=_kl_beta(config, epoch))
                development_losses.append(float(loss.cpu()))
        dev_loss = sum(development_losses) / len(development_losses)
        row = {"epoch": epoch, "train_loss": sum(train_losses) / len(train_losses), "development_loss": dev_loss, "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        state_payload = {"schema": f"{SCHEMA_PREFIX}_state.v1", "model_version": MODEL_VERSION, "epoch": epoch, "state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history}
        atomic_torch(state_dir / "latest.pt", state_payload)
        if dev_loss < best_loss:
            best_loss, best_epoch = dev_loss, epoch
            atomic_torch(run_dir / "first_stage_best.pt", {**state_payload, "selection": "development_composite_loss", "best_development_loss": best_loss})
    return {"best_epoch": best_epoch, "best_development_loss": best_loss, "history": history, "checkpoint": str(run_dir / "first_stage_best.pt"), "evaluator_calls": 0, "host_rss_peak_bytes": _max_rss_bytes(), "cuda_peak_bytes": int(torch.cuda.max_memory_allocated(device))}


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
            output = model.encode(_to_camera(data["images"].index_select(0, indices)).to(device))
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
                    output = model(_to_camera(images.index_select(0, chosen)).to(device), deterministic=True)
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
    manifest = {"schema": f"{SCHEMA_PREFIX}_reconstruction.v1", "status": "first_stage_reconstruction_bank", "model_version": MODEL_VERSION, "checkpoint_sha256": file_sha256(checkpoint), "checkpoint": str(checkpoint), "dataset_hash": data["dataset_hash"], "dataset_split_hash": data["split_hash"], "formal_bank_partition_seed": config["formal_bank_partition_seed"], "bank_count": 2, "samples_per_class": 250, "image_shape": [1, IMAGE_SIZE, IMAGE_SIZE], "latent_shape": list(LATENT_SHAPE), "decoder_input": "encoded_real_mu", "latent_standardization": "training_fit_mu_channelwise", "latent_stats_path": str(stats_path), "latent_stats_sha256": file_sha256(stats_path), "latent_stats_count": latent_count, "evaluator_calls": 0, "state_hash_before": before, "state_hash_after": after, "shards": shards, "latent_mu_sha256": tensor_sha256(latent)}
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
    return parser


def dispatch(args: argparse.Namespace) -> int:
    config = _config(args.config)
    git = require_clean()
    data = _validate_inputs(config, args.dataset_root)
    if args.action == "lineage":
        payload = _base("lineage", config, args.config, git, status="valid", passed=True, classification="lineage_validated", first_failed_gate=None, permitted=["preflight"])
        payload.update({"model_version": MODEL_VERSION, "image_size": IMAGE_SIZE, "latent_shape": list(LATENT_SHAPE), "dataset_hash": data["dataset_hash"], "dataset_split_hash": data["split_hash"], "formal_indices_hash": tensor_sha256(data["formal_indices"]), "train_indices_hash": tensor_sha256(data["train_indices"]), "evaluator_calls": 0})
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
