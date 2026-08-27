"""Package B3-T throughput attribution and safe-acceleration profiler.

This is a research-only diagnostic.  It uses throwaway models and temporary
serialization files; it never produces a B3 candidate checkpoint and never
constructs an evaluator, diffusion, student, or optical object.
"""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import inspect
import json
import math
import os
import resource
import statistics
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset, default_collate

from ..teacher.software_vae import (
    MODEL_VERSION,
    SoftwareMultiscaleTeacherVAE,
    _foreground_dice_per_sample,
    reconstruction_loss,
    sobel,
)
from .package_b3_vae import (
    CLASS_COUNT,
    FORMAL_SPLIT_HASH,
    STATE_SCHEMA,
    _config,
    _gradient_norm_is_finite,
    _kl_beta,
    _load_mnist,
    _select_development,
    _to_camera,
    atomic_json,
    file_sha256,
    require_clean,
)


PROGRAM = "package-b3-throughput"
SCHEMA_PREFIX = "package_b3_throughput"
PROFILER_VERSION = "b3_t1_wall_clock_attribution_v1"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _summary(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "total_ms": sum(values),
        "mean_ms": statistics.fmean(values) if values else 0.0,
        "median_ms": statistics.median(values) if values else 0.0,
        "p95_ms": _percentile(values, 95.0),
    }


def _read_meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    path = Path("/proc/meminfo")
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition(":")
        if value.strip().endswith(" kB"):
            result[key] = int(value.strip().split()[0]) * 1024
    return result


def _read_swaps() -> dict[str, Any]:
    path = Path("/proc/swaps")
    if not path.is_file():
        return {"available": False, "entries": []}
    lines = path.read_text(encoding="utf-8").splitlines()
    entries = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) >= 5:
            entries.append({"path": fields[0], "size_bytes": int(fields[2]) * 1024, "used_bytes": int(fields[3]) * 1024})
    return {"available": True, "entries": entries}


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _gpu_snapshot() -> dict[str, Any]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if not output:
        return {"available": False}
    fields = [field.strip() for field in output.splitlines()[0].split(",")]
    if len(fields) != 5:
        return {"available": False, "raw": output}
    return {
        "available": True,
        "name": fields[0],
        "memory_total_mib": int(fields[1]),
        "memory_used_mib": int(fields[2]),
        "utilization_gpu_percent": float(fields[3]),
        "driver_version": fields[4],
    }


def _watchdog_snapshot() -> dict[str, Any]:
    paths = [
        Path("/etc/systemd/system/wsl-python-memory-guard.service"),
        Path("/usr/local/sbin/wsl-python-memory-guard"),
    ]
    result: dict[str, Any] = {}
    for path in paths:
        result[str(path)] = {"exists": path.is_file(), "sha256": file_sha256(path) if path.is_file() else None}
    result["systemctl_status"] = _command_output(["systemctl", "is-enabled", "wsl-python-memory-guard.service"])
    return result


def _environment() -> dict[str, Any]:
    torch_cuda = torch.version.cuda
    return {
        "recorded_at": _now(),
        "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        "pytorch": torch.__version__,
        "torch_cuda": torch_cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "gpu": _gpu_snapshot(),
        "memory": _read_meminfo(),
        "swap": _read_swaps(),
        "wsl_python_memory_guard": _watchdog_snapshot(),
        "cpu_count": os.cpu_count(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def _validate_upstream(config: dict[str, Any], path: str, *, head: str, config_sha: str) -> dict[str, Any]:
    evidence = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(evidence, dict) or evidence.get("schema") != "teacher_first_stage_recovery_preflight.v1":
        raise ValueError("B3 throughput profiler requires Package B3 preflight evidence")
    if not evidence.get("passed") or evidence.get("git_head") != head or evidence.get("config_sha256") != config_sha:
        raise ValueError("B3 throughput profiler upstream preflight is stale or failed")
    if evidence.get("evaluator_calls") != 0:
        raise ValueError("B3 throughput profiler upstream evidence has evaluator calls")
    return evidence


def _training_contract(config: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "model_version",
        "image_size",
        "latent_shape",
        "camera_preprocessing",
        "microbatch_size",
        "gradient_accumulation",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "maximum_epochs",
        "minimum_epochs",
        "early_stopping_patience",
        "early_stopping_min_relative_improvement",
        "warmup_epochs",
        "kl_beta_maximum",
        "kl_warmup_fraction",
        "loss_temperature",
        "multiscale_weight",
        "sobel_weight",
        "foreground_dice_weight",
    )
    return {key: config[key] for key in keys}


def _baseline(args: argparse.Namespace) -> int:
    git = require_clean()
    config = _config(args.config)
    config_sha = file_sha256(args.config)
    data = _load_mnist(args.dataset_root)
    upstream = _validate_upstream(config, args.upstream_evidence, head=git["git_head"], config_sha=config_sha)
    model_file = Path(inspect.getfile(SoftwareMultiscaleTeacherVAE)).resolve()
    manifest = {
        "schema": f"{SCHEMA_PREFIX}_baseline_manifest.v1",
        "profiler_version": PROFILER_VERSION,
        "status": "valid",
        "classification": "baseline_locked",
        "first_failed_gate": None,
        "recorded_at": _now(),
        "git_head": git["git_head"],
        "dirty_worktree": git["dirty_worktree"],
        "config": str(Path(args.config).resolve()),
        "config_sha256": config_sha,
        "model_definition": str(model_file),
        "model_definition_sha256": file_sha256(model_file),
        "model_version": MODEL_VERSION,
        "training_state_schema": STATE_SCHEMA,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "dataset_hash": data["dataset_hash"],
        "dataset_split_hash": data["split_hash"],
        "formal_split_hash": FORMAL_SPLIT_HASH,
        "upstream_preflight": str(Path(args.upstream_evidence).resolve()),
        "upstream_preflight_sha256": file_sha256(args.upstream_evidence),
        "upstream_preflight_git_head": upstream["git_head"],
        "seed": config["master_seed"],
        "training_contract": _training_contract(config),
        "permitted_next_actions": ["profile"],
        "forbidden_next_actions": ["formal_training", "b3_ddpm", "student", "optical", "amp", "promotion", "candidate_export"],
        "evaluator_calls": 0,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "baseline_manifest.json", manifest)
    atomic_json(
        output_dir / "baseline_environment.json",
        {
            "schema": f"{SCHEMA_PREFIX}_baseline_environment.v1",
            "status": "valid",
            "classification": "environment_locked",
            "first_failed_gate": None,
            "git_head": git["git_head"],
            "dirty_worktree": git["dirty_worktree"],
            "config_sha256": config_sha,
            "dataset_hash": data["dataset_hash"],
            "dataset_split_hash": data["split_hash"],
            "environment": _environment(),
            "permitted_next_actions": ["profile"],
            "forbidden_next_actions": manifest["forbidden_next_actions"],
            "evaluator_calls": 0,
        },
    )
    return 0


class _TimedDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, dataset: Dataset[tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.dataset = dataset
        self.batch_get_ms: list[float] = []

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        started = time.perf_counter()
        value = self.dataset[index]
        self.batch_get_ms.append((time.perf_counter() - started) * 1000.0)
        return value


def _profile_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    labels: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    config: dict[str, Any],
    epoch: int,
    events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    def timed(label: str, operation: Callable[[], Any]) -> Any:
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record()
        result = operation()
        finished.record()
        events.setdefault(label, []).append((started, finished))
        return result

    recon_x, target_x = timed("sobel", lambda: (sobel(reconstruction), sobel(target)))

    def multiscale() -> torch.Tensor:
        terms = [
            F.smooth_l1_loss(reconstruction, target, reduction="none").flatten(1).mean(1),
            F.smooth_l1_loss(F.avg_pool2d(reconstruction, 2), F.avg_pool2d(target, 2), reduction="none").flatten(1).mean(1),
            F.smooth_l1_loss(F.avg_pool2d(reconstruction, 4), F.avg_pool2d(target, 4), reduction="none").flatten(1).mean(1),
        ]
        return torch.stack(terms, dim=1).mean(dim=1)

    multiscale_value = timed("multiscale", multiscale)

    def sobel_value() -> torch.Tensor:
        return torch.stack(
            [
                F.smooth_l1_loss(recon_x[0], target_x[0], reduction="none").flatten(1).mean(1),
                F.smooth_l1_loss(recon_x[1], target_x[1], reduction="none").flatten(1).mean(1),
            ],
            dim=1,
        ).mean(dim=1)

    sobel_loss = timed("sobel", sobel_value)
    dice = timed("dice", lambda: _foreground_dice_per_sample(reconstruction, target))

    def aggregate() -> torch.Tensor:
        sample_loss = (
            config["multiscale_weight"] * multiscale_value
            + config["sobel_weight"] * sobel_loss
            + config["foreground_dice_weight"] * dice
        )
        class_losses = []
        for label in range(CLASS_COUNT):
            selected = sample_loss[labels == label]
            if selected.numel():
                class_losses.append(selected.mean())
        if not class_losses:
            raise ValueError("profile loss requires at least one known MNIST class")
        return config["loss_temperature"] * torch.logsumexp(torch.stack(class_losses) / config["loss_temperature"], dim=0)

    reconstruction_part = timed("loss_aggregation", aggregate)
    kl = timed("kl", lambda: -0.5 * (1.0 + logvar - mu.square() - logvar.exp()).mean())
    total = reconstruction_part + _kl_beta(config, epoch) * kl
    metrics = {
        "total": total.detach(),
        "worst_class_reconstruction": reconstruction_part.detach(),
        "foreground_dice": dice.mean().detach(),
        "kl": kl.detach(),
    }
    return total, metrics


def _event_ms(events: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> list[float]:
    return [float(start.elapsed_time(end)) for start, end in events]


def _source_sync_counts() -> dict[str, int]:
    source_file = Path(inspect.getfile(SoftwareMultiscaleTeacherVAE)).resolve().parent.parent / "application" / "package_b3_vae.py"
    source = source_file.read_text(encoding="utf-8")
    tree = ast.parse(source)
    counts = {"item_calls": 0, "cpu_calls": 0, "numpy_calls": 0, "cuda_synchronize_calls": 0}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            key = {"item": "item_calls", "cpu": "cpu_calls", "numpy": "numpy_calls"}.get(node.func.attr)
            if key is not None:
                counts[key] += 1
            if node.func.attr == "synchronize":
                counts["cuda_synchronize_calls"] += 1
    return {"source_file": str(source_file), **counts}


class _SystemSampler:
    def __init__(self) -> None:
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="b3-throughput-system-sampler", daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = _gpu_snapshot()
            sample["rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            sample["time"] = _now()
            self.samples.append(sample)
            self._stop.wait(1.0)

    def __enter__(self) -> "_SystemSampler":
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)


def _profile(args: argparse.Namespace) -> int:
    git = require_clean()
    config = _config(args.config)
    config_sha = file_sha256(args.config)
    baseline = json.loads(Path(args.baseline_manifest).read_text(encoding="utf-8"))
    if (
        baseline.get("schema") != f"{SCHEMA_PREFIX}_baseline_manifest.v1"
        or baseline.get("git_head") != git["git_head"]
        or baseline.get("config_sha256") != config_sha
        or baseline.get("dataset_split_hash") != FORMAL_SPLIT_HASH
    ):
        raise ValueError("B3 throughput baseline manifest is stale or mismatched")
    device = torch.device("cuda") if args.device == "cuda" else None
    if device is None or not torch.cuda.is_available():
        raise RuntimeError("B3 throughput profiler requires CUDA")
    from .package_b3_vae import require_cuda

    device = require_cuda(args.device)
    data = _load_mnist(args.dataset_root)
    if data["dataset_hash"] != baseline.get("dataset_hash"):
        raise ValueError("B3 throughput dataset hash changed after baseline lock")
    fitting, development = _select_development(
        data["train_indices"],
        data["labels"],
        seed=config["development_seed"],
        samples_per_class=config["development_samples_per_class"],
    )
    torch.manual_seed(config["master_seed"])
    torch.cuda.reset_peak_memory_stats(device)
    model: SoftwareMultiscaleTeacherVAE = SoftwareMultiscaleTeacherVAE().to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    order = fitting[: (8 + 200) * config["microbatch_size"]]
    source_dataset = TensorDataset(data["images"], data["labels"])
    timed_dataset = _TimedDataset(Subset(source_dataset, order.tolist()))
    collate_ms: list[float] = []

    def collate(batch: list[Any]) -> Any:
        started = time.perf_counter()
        result = default_collate(batch)
        collate_ms.append((time.perf_counter() - started) * 1000.0)
        return result

    loader = DataLoader(
        timed_dataset,
        batch_size=config["microbatch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        collate_fn=collate,
    )
    loader_wait_values: list[float] = []
    raw_loader_wait_values: list[float] = []
    data_index_values: list[float] = []
    collate_values: list[float] = []
    cpu_preprocess_values: list[float] = []
    finite_sync_values: list[float] = []
    event_phase_values: dict[str, list[float]] = {}
    window_started: float | None = None
    warmup_batches = 8
    measured_batches = 200
    total_batches = warmup_batches + measured_batches
    loader_iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    equivalence: dict[str, Any] = {"checked": False, "bitwise_identical": False}

    with _SystemSampler() as sampler:
        for batch_number in range(total_batches):
            before_get = len(timed_dataset.batch_get_ms)
            before_collate = len(collate_ms)
            wait_started = time.perf_counter()
            batch_images, batch_labels = next(loader_iterator)
            wait_ms = (time.perf_counter() - wait_started) * 1000.0
            if batch_number >= warmup_batches:
                index_ms = sum(timed_dataset.batch_get_ms[before_get:])
                collate_ms_value = sum(collate_ms[before_collate:])
                raw_loader_wait_values.append(wait_ms)
                data_index_values.append(index_ms)
                collate_values.append(collate_ms_value)
                loader_wait_values.append(max(0.0, wait_ms - index_ms - collate_ms_value))
                cpu_preprocess_values.append(0.0)
            events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {}
            h2d_started = torch.cuda.Event(enable_timing=True)
            h2d_finished = torch.cuda.Event(enable_timing=True)
            h2d_started.record()
            batch_images = batch_images.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            h2d_finished.record()
            forward_started = torch.cuda.Event(enable_timing=True)
            forward_finished = torch.cuda.Event(enable_timing=True)
            forward_started.record()
            output = model(batch_images, deterministic=False)
            forward_finished.record()
            loss, metrics = _profile_loss(
                output["reconstruction"],
                batch_images,
                batch_labels,
                output["mu"],
                output["logvar"],
                config,
                1,
                events,
            )
            if not equivalence["checked"]:
                reference_loss, _ = reconstruction_loss(
                    output["reconstruction"], batch_images, batch_labels, output["mu"], output["logvar"],
                    temperature=config["loss_temperature"], kl_beta=_kl_beta(config, 1),
                    multiscale_weight=config["multiscale_weight"], sobel_weight=config["sobel_weight"],
                    foreground_dice_weight=config["foreground_dice_weight"],
                )
                equivalence = {
                    "checked": True,
                    "bitwise_identical": bool(torch.equal(loss, reference_loss)),
                    "max_abs_deviation": float((loss - reference_loss).abs().detach().cpu()),
                    "max_relative_deviation": float(((loss - reference_loss).abs() / reference_loss.abs().clamp_min(1.0e-12)).detach().cpu()),
                }
                if not equivalence["bitwise_identical"]:
                    raise RuntimeError("profiler loss attribution is not numerically identical to registered loss")
            finite_started = time.perf_counter()
            if not bool(torch.isfinite(loss).item()):
                raise ValueError("B3 throughput profiler loss is non-finite")
            finite_sync_values.append((time.perf_counter() - finite_started) * 1000.0)
            backward_started = torch.cuda.Event(enable_timing=True)
            backward_finished = torch.cuda.Event(enable_timing=True)
            backward_started.record()
            (loss / config["gradient_accumulation"]).backward()
            backward_finished.record()
            next_batch = batch_number + 1
            if next_batch % config["gradient_accumulation"] == 0:
                grad_started = time.perf_counter()
                gradient_started = torch.cuda.Event(enable_timing=True)
                gradient_finished = torch.cuda.Event(enable_timing=True)
                gradient_started.record()
                if not _gradient_norm_is_finite(model):
                    raise ValueError("B3 throughput profiler gradient is non-finite")
                gradient_finished.record()
                finite_sync_values.append((time.perf_counter() - grad_started) * 1000.0)
                optimizer_started = torch.cuda.Event(enable_timing=True)
                optimizer_finished = torch.cuda.Event(enable_timing=True)
                optimizer_started.record()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_finished.record()
                events.setdefault("gradient_check", []).append((gradient_started, gradient_finished))
                events.setdefault("optimizer_step", []).append((optimizer_started, optimizer_finished))
            if batch_number == warmup_batches - 1:
                torch.cuda.synchronize(device)
                window_started = time.perf_counter()
            torch.cuda.synchronize(device)
            if batch_number >= warmup_batches:
                event_phase_values.setdefault("host_to_device", []).append(float(h2d_started.elapsed_time(h2d_finished)))
                event_phase_values.setdefault("model_forward", []).append(float(forward_started.elapsed_time(forward_finished)))
                event_phase_values.setdefault("backward", []).append(float(backward_started.elapsed_time(backward_finished)))
                for label, pairs in events.items():
                    if pairs:
                        event_phase_values.setdefault(label, []).append(sum(_event_ms(pairs)))
    if window_started is None:
        raise RuntimeError("B3 profiler did not enter measured window")
    training_window_seconds = time.perf_counter() - window_started
    scheduler_started = time.perf_counter()
    scheduler.step()
    scheduler_seconds = time.perf_counter() - scheduler_started
    with tempfile.TemporaryDirectory(prefix="b3_throughput_checkpoint_") as temporary:
        checkpoint_path = Path(temporary) / "checkpoint.pt"
        checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
        serialization_started = time.perf_counter()
        torch.save(checkpoint, checkpoint_path)
        serialization_seconds = time.perf_counter() - serialization_started

    model.eval()
    development_batch_values: list[float] = []
    development_batch_count = 100
    with torch.inference_mode():
        for start in range(0, development_batch_count * config["microbatch_size"], config["microbatch_size"]):
            indices = development[start : start + config["microbatch_size"]]
            started = time.perf_counter()
            images = _to_camera(data["images"].index_select(0, indices).to(device, non_blocking=True))
            labels = data["labels"].index_select(0, indices).to(device, non_blocking=True)
            output = model(images, deterministic=True)
            reconstruction_loss(
                output["reconstruction"], images, labels, output["mu"], output["logvar"],
                temperature=config["loss_temperature"], kl_beta=_kl_beta(config, 1),
                multiscale_weight=config["multiscale_weight"], sobel_weight=config["sobel_weight"],
                foreground_dice_weight=config["foreground_dice_weight"],
            )
            torch.cuda.synchronize(device)
            development_batch_values.append((time.perf_counter() - started) * 1000.0)

    cuda_peak = int(torch.cuda.max_memory_allocated(device))
    rss_peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    phase_values: dict[str, list[float]] = {
        "data_index_preparation": data_index_values,
        "dataloader_wait": loader_wait_values,
        "dataloader_collate": collate_values,
        "cpu_preprocessing": cpu_preprocess_values,
        "host_to_device": event_phase_values.get("host_to_device", []),
        "model_forward": event_phase_values.get("model_forward", []),
        "reconstruction_loss": [
            sum(values)
            for values in zip(
                event_phase_values.get("multiscale", []),
                event_phase_values.get("sobel", []),
                event_phase_values.get("dice", []),
                event_phase_values.get("loss_aggregation", []),
                event_phase_values.get("kl", []),
                strict=False,
            )
        ],
        "multiscale_loss": event_phase_values.get("multiscale", []),
        "sobel_loss": event_phase_values.get("sobel", []),
        "dice_loss": event_phase_values.get("dice", []),
        "kl_loss": event_phase_values.get("kl", []),
        "backward": event_phase_values.get("backward", []),
        "gradient_finite_norm_check": event_phase_values.get("gradient_check", []),
        "optimizer_step": event_phase_values.get("optimizer_step", []),
        "cuda_synchronization_wait": finite_sync_values,
    }
    phase_summaries = {key: _summary(values) for key, values in phase_values.items()}
    phase_summaries["scheduler_overhead"] = _summary([scheduler_seconds * 1000.0])
    phase_summaries["logging"] = _summary([])
    phase_summaries["checkpoint_serialization"] = _summary([serialization_seconds * 1000.0])
    profile_total_ms = training_window_seconds * 1000.0
    attributed_ms = sum(float(value["total_ms"]) for value in phase_summaries.values())
    phase_summaries["misc_unattributed"] = _summary([max(0.0, profile_total_ms - attributed_ms)])
    percentages = {
        key: (float(value["total_ms"]) / profile_total_ms * 100.0 if profile_total_ms else 0.0)
        for key, value in phase_summaries.items()
    }
    ranked = sorted(
        ((key, percentage) for key, percentage in percentages.items() if key != "misc_unattributed"),
        key=lambda item: item[1],
        reverse=True,
    )
    non_compute = sum(percentages.get(key, 0.0) for key in ("data_index_preparation", "dataloader_wait", "cpu_preprocessing", "host_to_device", "cuda_synchronization_wait", "misc_unattributed")) / 100.0
    safe_speedup = 1.0 / max(1.0e-9, 1.0 - min(0.95, non_compute))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    breakdown = {
        "schema": f"{SCHEMA_PREFIX}_timing_breakdown.v1",
        "profiler_version": PROFILER_VERSION,
        "status": "valid",
        "classification": "timing_attribution_complete",
        "first_failed_gate": None,
        "recorded_at": _now(),
        "git_head": git["git_head"],
        "dirty_worktree": git["dirty_worktree"],
        "config_sha256": config_sha,
        "dataset_hash": data["dataset_hash"],
        "dataset_split_hash": data["split_hash"],
        "seed": config["master_seed"],
        "warmup_batches": warmup_batches,
        "measured_training_batches": measured_batches,
        "development_batches": development_batch_count,
        "microbatch_size": config["microbatch_size"],
        "gradient_accumulation": config["gradient_accumulation"],
        "profile_training_window_ms": profile_total_ms,
        "phases": phase_summaries,
        "phase_percentages": percentages,
        "development_evaluation": _summary(development_batch_values),
        "top_1_bottleneck": ranked[0][0] if ranked else None,
        "top_2_bottleneck": ranked[1][0] if len(ranked) > 1 else None,
        "top_3_bottleneck": ranked[2][0] if len(ranked) > 2 else None,
        "estimated_max_safe_speedup": safe_speedup,
        "equivalence": equivalence,
        "resource": {
            "cuda_peak_bytes": cuda_peak,
            "host_rss_peak_bytes": rss_peak,
            "system_samples": sampler.samples,
        },
        "dataloader_starvation": {
            "num_workers": config["num_workers"],
            "queue_available": False,
            "wait_over_1ms_count": sum(value > 1.0 for value in loader_wait_values),
            "wait_summary": _summary(loader_wait_values),
            "raw_wait_summary": _summary(raw_loader_wait_values),
        },
        "source_synchronization_counts": _source_sync_counts(),
        "evaluator_calls": 0,
        "optimizer_created": True,
        "candidate_export": False,
        "permitted_next_actions": ["analyze_timing_before_optimization"],
        "forbidden_next_actions": ["formal_training", "b3_ddpm", "student", "optical", "amp", "promotion", "candidate_export"],
    }
    atomic_json(output_dir / "b3_timing_breakdown.json", breakdown)
    report = _render_report(baseline, breakdown)
    (output_dir / "b3_timing_report.md").write_text(report, encoding="utf-8")
    return 0


def _render_report(baseline: dict[str, Any], breakdown: dict[str, Any]) -> str:
    phases = breakdown["phases"]
    percentages = breakdown["phase_percentages"]
    lines = [
        "# Package B3-T1 Timing Attribution",
        "",
        "This is a diagnostic-only profiler result. No B3 candidate was created and no downstream stage was authorized.",
        "",
        f"- HEAD: `{breakdown['git_head']}`",
        f"- Baseline manifest: `{baseline.get('schema')}` / `{baseline.get('config_sha256')}`",
        f"- Training window: {breakdown['profile_training_window_ms'] / 60000.0:.3f} minutes",
        f"- Measured batches: {breakdown['measured_training_batches']} after {breakdown['warmup_batches']} warmup batches",
        f"- CUDA peak: {breakdown['resource']['cuda_peak_bytes'] / 1024**3:.3f} GiB",
        f"- RSS peak: {breakdown['resource']['host_rss_peak_bytes'] / 1024**3:.3f} GiB",
        "",
        "## Attribution",
        "",
        "| phase | total ms | mean ms | median ms | p95 ms | share |\n|---|---:|---:|---:|---:|---:|",
    ]
    for key, value in sorted(phases.items(), key=lambda item: float(item[1]["total_ms"]), reverse=True):
        lines.append(f"| {key} | {value['total_ms']:.1f} | {value['mean_ms']:.3f} | {value['median_ms']:.3f} | {value['p95_ms']:.3f} | {percentages[key]:.2f}% |")
    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- top_1_bottleneck: `{breakdown['top_1_bottleneck']}`",
            f"- top_2_bottleneck: `{breakdown['top_2_bottleneck']}`",
            f"- top_3_bottleneck: `{breakdown['top_3_bottleneck']}`",
            f"- estimated_max_safe_speedup: `{breakdown['estimated_max_safe_speedup']:.3f}x`",
            f"- loss attribution bitwise equivalence: `{breakdown['equivalence']['bitwise_identical']}`",
            f"- evaluator calls: `{breakdown['evaluator_calls']}`",
            "",
            "The profiler result authorizes timing analysis only. It does not authorize modifying the formal B3 contract, enabling AMP, resuming formal training, or entering DDPM/Student/optical stages.",
            "",
        ]
    )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package B3-T throughput attribution profiler")
    parser.add_argument("action", choices=("baseline", "profile"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--upstream-evidence", required=True)
    parser.add_argument("--baseline-manifest")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.action == "baseline":
        return _baseline(args)
    if not args.baseline_manifest:
        raise ValueError("profile requires --baseline-manifest")
    return _profile(args)


if __name__ == "__main__":
    raise SystemExit(main())
