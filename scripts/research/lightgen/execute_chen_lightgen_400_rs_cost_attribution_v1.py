"""Attribute the saved-qualified LightGen400 complex64 batch10 RS cost.

This diagnostic performs production optical forward/backward measurements only.
It never calls an optimizer, evaluator, sampler, or training update.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import html
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_json
from scripts.research.lightgen import execute_chen_lightgen_400_fast_rs_compact_v1 as compact
from scripts.research.lightgen import execute_chen_lightgen_400_mnist_baseline_v1 as baseline
from scripts.research.lightgen import execute_chen_lightgen_400_q12_trainability_v1 as q12


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CHEN_LIGHTGEN_400_RS_COST_ATTRIBUTION_V1"
QUALIFICATION_TASK_ID = "CHEN_LIGHTGEN_400_FAST_RS_QUALIFICATION_V2"
QUALIFICATION_SCHEMA = "chen_lightgen_400_fast_rs_qualification_manifest.v2"
LOGICAL_BATCH = 10
MICROBATCH = 2
PIXELS = ("encoder_phase.0", "encoder_phase.1", "decoder_phase.0", "decoder_phase.1",
          "decoder_phase.2", "decoder_phase.3")
LOSS_RTOL = 1.0e-5
GRAD_RTOL = 1.0e-4
OUTPUT_RTOL = 1.0e-4
OUTPUT_LIMIT_BYTES = 512 * 1024**2
LOSS_GRADIENT_ONLY_STATUS = "LOSS_GRADIENT_EQUIVALENT_W_PIXEL_NOT_AVAILABLE"
QUALIFIED_REFERENCE_STATUSES = {"NUMERICALLY_EQUIVALENT", LOSS_GRADIENT_ONLY_STATUS}
PAIR_MATERIALIZATIONS_PER_RS_CALL = (12**2 // 4) ** 2
BASE_RS_FORWARD_CALLS_PER_BATCH10 = (LOGICAL_BATCH // MICROBATCH) * 7
FULL_BATCH_ORDER = ("checkpoint_on_1", "checkpoint_off_1",
                    "checkpoint_off_2", "checkpoint_on_2")


class HarnessError(RuntimeError):
    """An input or measurement contract failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessError(message)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _safe_json(value: Any) -> Any:
    return baseline._json_safe(value)


def _write_json(path: Path, payload: dict[str, Any], *, new: bool = False) -> None:
    safe = _safe_json(payload)
    if new:
        from ldm_project.v2.evidence import atomic_write_new_json
        atomic_write_new_json(path, safe)
    else:
        atomic_write_json(path, safe)


def _manifest_identity(path: str) -> str:
    normalized = path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    try:
        index = parts.index("codex-original")
    except ValueError as exc:
        raise HarnessError(f"manifest input is outside codex-original: {path}") from exc
    return "/".join(parts[index + 1:])


def _explicit_paths(args: argparse.Namespace) -> dict[str, str]:
    names = ("source_checkpoint", "source_report", "source_start_state",
             "source_endpoint_checkpoint", "fixed_population", "config", "data",
             "evaluator", "qualification_report", "structure_coefficient",
             "qualification_dir", "output_dir")
    return {name: str(_resolve(getattr(args, name)).resolve()) for name in names}


def _input_namespace(args: argparse.Namespace) -> argparse.Namespace:
    paths = {name: _resolve(getattr(args, name)) for name in (
        "source_checkpoint", "source_report", "source_start_state", "source_endpoint_checkpoint",
        "fixed_population", "config", "data", "evaluator", "qualification_report",
    )}
    paths.update({"qualification_cache": _resolve(args.qualification_dir) / "qualification_raw",
                  "output_dir": _resolve(args.output_dir)})
    return argparse.Namespace(**paths)


def _validate_fixed_population(population: dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
    q12._validate_population_labels(population, labels)
    rows_by_class = population["fit_rows_by_class"].long().cpu()
    _require(tuple(rows_by_class.shape) == (10, 10),
             "fixed fit population must retain ten rows in each of ten classes")
    rows = rows_by_class[:, 0]
    _require(rows.unique().numel() == LOGICAL_BATCH
             and torch.equal(labels.long().cpu().index_select(0, rows), torch.arange(10)),
             "fixed batch10 rows are incomplete, duplicated, or not class ordered")
    return rows


def _load_context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    context = q12._load_context(_input_namespace(args), device)
    _require(context["checkpoint"].get("completed_updates") == 30_000,
             "explicit source checkpoint is not the 30000-step baseline")
    _require(context["endpoint_payload"].get("completed_updates") == 500,
             "historical comparison endpoint is not step 500")
    population = context["population"]
    context["eta"] = compact._load_eta(_resolve(args.structure_coefficient))
    context["batch10_rows"] = _validate_fixed_population(population, context["labels"])
    return context


def _validate_previous_evidence(args: argparse.Namespace) -> dict[str, Any]:
    qualification_dir = _resolve(args.qualification_dir)
    manifest_path = qualification_dir / "run_manifest.json"
    report_path = qualification_dir / "final_report.json"
    metric_path = qualification_dir / "cost" / "cost_complex64_B_microbatch_2.json"
    gradient_path = qualification_dir / "gradients" / "cost_complex64_B_microbatch_2.pt"
    for path in (manifest_path, report_path, metric_path, gradient_path):
        _require(path.is_file(), f"required saved qualification evidence is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == QUALIFICATION_SCHEMA
             and manifest.get("task_id") == QUALIFICATION_TASK_ID,
             "saved qualification manifest has the wrong task identity")
    _require(report.get("status") == "VALID_COMPLETED_BOTH_PRECISIONS_QUALIFIED"
             and report.get("complex128_qualified") is True
             and report.get("complex64_qualified") is True
             and report.get("optimizer_updates") == 0,
             "saved two-precision qualification is incomplete or has a different status")
    _require(metric.get("precision") == "complex64" and metric.get("branch") == "B"
             and metric.get("microbatch") == MICROBATCH
             and metric.get("logical_batch") == LOGICAL_BATCH
             and metric.get("checkpoint_propagation") is True
             and metric.get("adam_calls") == metric.get("optimizer_calls")
             == metric.get("optimizer_updates") == 0,
             "saved batch10 reference is not the fixed complex64/B/microbatch2 case")
    paths = _explicit_paths(args)
    saved_inputs = manifest.get("explicit_inputs")
    _require(isinstance(saved_inputs, dict), "qualification manifest lacks explicit input paths")
    field_map = {
        "source_checkpoint": "source_checkpoint", "source_report": "source_report",
        "source_start_state": "source_start_state",
        "source_endpoint_checkpoint": "source_endpoint_checkpoint",
        "fixed_population": "fixed_population", "config": "config", "data": "data",
        "evaluator": "evaluator", "qualification_report": "qualification_report",
        "structure_coefficient": "structure_coefficient", "qualification_dir": "output_dir",
    }
    for cli_name, manifest_name in field_map.items():
        _require(manifest_name in saved_inputs
                 and _manifest_identity(paths[cli_name]) == _manifest_identity(saved_inputs[manifest_name]),
                 f"explicit {cli_name} differs from the qualified input path")
    _require(isinstance(metric.get("rows"), list)
             and len(metric["rows"]) == LOGICAL_BATCH,
             "saved batch10 sample rows are missing or inconsistent with its manifest")
    return {"manifest": manifest, "report": report, "metric": metric,
            "gradient_path": gradient_path, "paths": paths}


def _load_reference_gradient(reference: dict[str, Any], rows: torch.Tensor) -> dict[str, Any]:
    payload = torch.load(reference["gradient_path"], map_location="cpu", weights_only=False)
    _require(payload.get("precision") == "complex64"
             and payload.get("branch") == "B"
             and payload.get("microbatch") == MICROBATCH
             and payload.get("state") == "source_step_30000"
             and isinstance(payload.get("rows"), torch.Tensor)
             and torch.equal(payload["rows"].long().cpu(), rows.long().cpu()),
             "saved cost gradient identity differs from the fixed reference rows")
    _validate_saved_rows(reference["metric"].get("rows"), rows)
    gradients = payload.get("gradients")
    _require(isinstance(gradients, dict)
             and set(gradients) == {"encoder_phase", "decoder_phase"}
             and all(isinstance(value, torch.Tensor) and torch.isfinite(value).all()
                     for value in gradients.values()),
             "saved batch10 reference gradients are missing or nonfinite")
    _validate_gradient_shapes(gradients)
    payload_loss = float(payload["loss"])
    metric_loss = float(reference["metric"].get("loss", math.nan))
    elapsed = float(reference["metric"].get("elapsed_seconds", math.nan))
    _require(math.isfinite(payload_loss) and math.isfinite(metric_loss)
             and math.isclose(payload_loss, metric_loss, rel_tol=1.0e-12, abs_tol=1.0e-14)
             and math.isfinite(elapsed) and elapsed > 0.0,
             "saved batch10 reference loss or elapsed time is missing or inconsistent")
    return {"loss": payload_loss, "gradients": gradients}


def _validate_saved_rows(saved_rows: Any, rows: torch.Tensor) -> None:
    _require(isinstance(saved_rows, list) and saved_rows == rows.long().cpu().tolist(),
             "saved batch10 cost rows differ from the fixed fit population")


def _relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    a, b = candidate.double().reshape(-1), reference.double().reshape(-1)
    return float(torch.linalg.vector_norm(a - b)
                 / torch.linalg.vector_norm(b).clamp_min(1.0e-30))


def _gradient_errors(candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
                     ) -> dict[str, float]:
    errors: dict[str, float] = {}
    for group in ("encoder_phase", "decoder_phase"):
        for layer in range(candidate[group].shape[0]):
            name = f"{group}.{layer}"
            errors[name] = _relative_error(candidate[group][layer], reference[group][layer])
    return errors


def _validate_gradient_shapes(gradients: dict[str, torch.Tensor]) -> None:
    _require(tuple(gradients["encoder_phase"].shape) == (2, 1, 400, 400)
             and tuple(gradients["decoder_phase"].shape) == (4, 1, 400, 400),
             "production preflight did not capture all six single-channel phase-layer gradients")


def _fixed_readout_prediction(physical_w_per_pixel: torch.Tensor) -> torch.Tensor:
    return q12.FIXED_GAIN * physical_w_per_pixel.to(torch.float64)


class _HardwareSampler:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.stop = threading.Event()
        self.samples: list[dict[str, Any]] = []
        self.started = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _sample(self) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True, timeout=5.0,
            )
            row = completed.stdout.strip().splitlines()[0].split(",")
            return {"time_unix": time.time(), "gpu_utilization_percent": int(row[0].strip()),
                    "memory_used_mib": int(row[1].strip()), "temperature_c": int(row[2].strip()),
                    "status": "sampled"}
        except (OSError, subprocess.SubprocessError, IndexError, ValueError) as exc:
            return {"time_unix": time.time(), "status": "unavailable",
                    "reason": f"{type(exc).__name__}: {exc}"}

    def _run(self) -> None:
        while not self.stop.is_set():
            self.samples.append(self._sample())
            self.heartbeat("resource_sampling")
            self.stop.wait(10.0)

    def __enter__(self) -> _HardwareSampler:
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop.set()
        self.thread.join(timeout=10.0)
        if self.output_dir.exists():
            _write_json(self.output_dir / "hardware_samples.json", {
                "elapsed_seconds": time.time() - self.started,
                "interval_seconds": 10.0, "samples": self.samples,
            })

    def heartbeat(self, stage: str) -> None:
        if not self.output_dir.exists():
            return
        progress: dict[str, Any] = {}
        progress_path = self.output_dir / "progress.json"
        if progress_path.is_file():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress = {}
        _write_json(self.output_dir / "heartbeat.json", {
            "task_id": TASK_ID, "stage": stage, "progress": progress,
            "time_unix": time.time(), "cuda_allocated_bytes": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(),
        })


class _OperationTrace:
    """Count production RS work and optionally profile two bounded calls."""

    def __init__(self, model: torch.nn.Module, *, profile: bool = False):
        self.model = model
        self.operator = model.propagator.operator
        self.cached_type = type(self.operator)
        self.linear_type = type(model.propagator)
        self.profile = profile
        self.counts: dict[str, int] = {}
        self.host_seconds: dict[str, float] = {}
        self.cuda_intervals: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        self.profile_summaries: list[dict[str, Any]] = []
        self.cache_build: dict[str, Any] | None = None
        self._originals: list[tuple[type, str, Any]] = []
        self._profile_seen: set[str] = set()

    def _add_patch(self, cls: type, name: str, replacement: Any) -> None:
        self._originals.append((cls, name, getattr(cls, name)))
        setattr(cls, name, replacement)

    def _count(self, key: str, amount: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + amount

    def _profile_call(self, direction: str, original: Any, instance: Any,
                      args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        if not self.profile or direction in self._profile_seen or not torch.cuda.is_available():
            return original(instance, *args, **kwargs)
        self._profile_seen.add(direction)
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        started = time.perf_counter()
        with torch.profiler.profile(activities=activities, record_shapes=False,
                                    profile_memory=False, with_stack=False) as profiler:
            result = original(instance, *args, **kwargs)
        torch.cuda.synchronize()
        top = []
        for item in sorted(profiler.key_averages(),
                           key=lambda value: getattr(value, "self_cuda_time_total", 0.0), reverse=True)[:30]:
            cuda_us = float(getattr(item, "self_cuda_time_total", 0.0))
            cpu_us = float(getattr(item, "self_cpu_time_total", 0.0))
            if cuda_us or cpu_us:
                top.append({"operator": item.key, "calls": item.count,
                            "self_cuda_time_us": cuda_us, "self_cpu_time_us": cpu_us})
        self.profile_summaries.append({
            "direction": direction, "elapsed_wall_seconds_with_profiler": time.perf_counter() - started,
            "profile_scope": "one production cached RS operator call; nested totals are not summed",
            "top_operators": top,
        })
        return result

    def __enter__(self) -> _OperationTrace:
        original_apply = self.cached_type._apply_convolution

        def apply(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if instance is not self.operator:
                return original_apply(instance, *args, **kwargs)
            direction = "adjoint" if kwargs.get("adjoint", False) else "forward"
            self._count(f"rs_{direction}_calls")
            started = time.perf_counter()
            profiled_call = (self.profile and direction not in self._profile_seen
                             and torch.cuda.is_available())
            start_event = end_event = None
            if torch.cuda.is_available() and not profiled_call:
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
            result = self._profile_call(direction, original_apply, instance, args, kwargs)
            if end_event is not None:
                end_event.record()
                self.cuda_intervals.append((direction, start_event, end_event))
            self.host_seconds[f"rs_{direction}"] = self.host_seconds.get(f"rs_{direction}", 0.0) \
                + time.perf_counter() - started
            return result

        original_cache = self.cached_type._get_kernel_cache

        def get_cache(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if instance is self.operator:
                source_shape, output_shape, fft_shape, device = args[:4]
                key = instance._cache_key(source_shape, output_shape, fft_shape, device)
                warm = instance._kernel_cache is not None and instance._kernel_cache_key == key
                self._count("kernel_cache_hits" if warm else "kernel_cache_misses")
            return original_cache(instance, *args, **kwargs)

        original_build = self.cached_type._build_kernel_cache

        def build_cache(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if instance is not self.operator:
                return original_build(instance, *args, **kwargs)
            host_started = time.perf_counter()
            start_event = end_event = None
            if torch.cuda.is_available():
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
            value = original_build(instance, *args, **kwargs)
            if end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
            self._count("kernel_cache_builds")
            self.cache_build = {"host_seconds": time.perf_counter() - host_started,
                                "cuda_event_ms": None if start_event is None else start_event.elapsed_time(end_event),
                                "cache_bytes": value.numel() * value.element_size(),
                                "canonical_kernel_count": len(instance._canonical_pairs),
                                "compute_precision": instance.compute_precision,
                                "geometry": instance._cache_key(*args[:4])}
            return value

        original_spectra = self.cached_type._spectra_for_pairs

        def spectra(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if instance is self.operator:
                self._count("pair_spectrum_materializations")
            return original_spectra(instance, *args, **kwargs)

        original_linear_forward = self.linear_type.forward

        def linear_forward(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if instance is self.model.propagator:
                self._count("linear_propagation_forward_calls")
                if kwargs.get("capture_center", False):
                    self._count("center_observation_requests")
                if kwargs.get("full_window_power", False):
                    self._count("full_window_observation_requests")
            return original_linear_forward(instance, *args, **kwargs)

        self._add_patch(self.cached_type, "_apply_convolution", apply)
        self._add_patch(self.cached_type, "_get_kernel_cache", get_cache)
        self._add_patch(self.cached_type, "_build_kernel_cache", build_cache)
        self._add_patch(self.cached_type, "_spectra_for_pairs", spectra)
        self._add_patch(self.linear_type, "forward", linear_forward)
        return self

    def __exit__(self, *_exc: object) -> None:
        for cls, name, original in reversed(self._originals):
            setattr(cls, name, original)
        self._originals.clear()

    def reset_counts(self) -> None:
        self.counts.clear()
        self.host_seconds.clear()
        self.cuda_intervals.clear()

    def collect_cuda_intervals(self) -> dict[str, Any]:
        if not self.cuda_intervals:
            return {"by_direction_ms": {}, "event_interval_count": 0}
        torch.cuda.synchronize()
        by_direction: dict[str, float] = {}
        for direction, start, end in self.cuda_intervals:
            by_direction[direction] = by_direction.get(direction, 0.0) + float(start.elapsed_time(end))
        return {"by_direction_ms": by_direction, "event_interval_count": len(self.cuda_intervals),
                "interpretation": "non-overlapping calls on the production CUDA stream; descriptive only"}


class _StageClock:
    def __init__(self, device: torch.device):
        self.device = device
        self.host_seconds: dict[str, float] = {}
        self.cuda_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @contextlib.contextmanager
    def stage(self, name: str):
        host_started = time.perf_counter()
        start = end = None
        if self.device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        try:
            yield
        finally:
            if end is not None:
                end.record()
                self.cuda_events.append((name, start, end))
            self.host_seconds[name] = self.host_seconds.get(name, 0.0) + time.perf_counter() - host_started

    def report(self) -> dict[str, Any]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        cuda_ms: dict[str, float] = {}
        for name, start, end in self.cuda_events:
            cuda_ms[name] = cuda_ms.get(name, 0.0) + float(start.elapsed_time(end))
        return {"host_wall_seconds_by_stage": self.host_seconds,
                "cuda_event_elapsed_ms_by_stage": cuda_ms,
                "timing_note": "stage durations are separate views; asynchronous host and device times are not additive"}


def _state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    values = {f"parameter:{name}": value.detach().cpu().clone()
              for name, value in model.named_parameters()}
    values.update({f"buffer:{name}": value.detach().cpu().clone()
                   for name, value in model.named_buffers()})
    return values


def _state_matches(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> bool:
    current = _state_snapshot(model)
    return set(current) == set(snapshot) and all(torch.equal(current[name], snapshot[name])
                                                 for name in snapshot)


def _set_checkpoint_propagation(model: torch.nn.Module, enabled: bool) -> None:
    model.config = dataclasses.replace(model.config, checkpoint_propagation=enabled)


def _run_batch10(model: torch.nn.Module, pixels: torch.Tensor, rows: torch.Tensor,
                 beta: float, eta: float, device: torch.device,
                 trace: _OperationTrace) -> dict[str, Any]:
    _require(rows.numel() == LOGICAL_BATCH, "production cost measurement requires the fixed ten-row batch")
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    stage_clock = _StageClock(device)
    output_parts: list[torch.Tensor] = []
    total_loss = 0.0
    started = time.perf_counter()
    for start_index in range(0, LOGICAL_BATCH, MICROBATCH):
        part = rows[start_index:start_index + MICROBATCH]
        with stage_clock.stage("input_transfer_and_resize"):
            target, _ = baseline._image_batch(pixels, part, device)
        with stage_clock.stage("optical_forward_and_power"):
            result = model.forward_from_image(target)
            prediction = _fixed_readout_prediction(result["physical_w_per_pixel"])
            output_parts.append(result["physical_w_per_pixel"].detach())
        with stage_clock.stage("B_mse_pool_and_ssim_loss"):
            target64 = target.to(torch.float64)
            mse = (prediction - target64).square().mean(dtype=torch.float64)
            prediction28 = F.adaptive_avg_pool2d(prediction.to(torch.float32), (28, 28))
            target28 = F.adaptive_avg_pool2d(target, (28, 28))
            structure = 1.0 - baseline._ssim_batch(prediction28, target28).to(torch.float64).mean()
            loss = beta * mse + eta * structure
        with stage_clock.stage("backward"):
            (loss * (part.numel() / LOGICAL_BATCH)).backward()
        with stage_clock.stage("per_microbatch_loss_transfer"):
            total_loss += float(loss.detach().cpu()) * part.numel() / LOGICAL_BATCH
        del target, result, prediction, target64, mse, prediction28, target28, structure, loss

    gradient_transfer_started = time.perf_counter()
    gradients: dict[str, torch.Tensor] = {}
    for name, parameter in zip(("encoder_phase", "decoder_phase"),
                               (model.encoder_phase, model.decoder_phase), strict=True):
        gradients[name] = parameter.grad.detach().cpu().clone()
    gradient_transfer_seconds = time.perf_counter() - gradient_transfer_started
    final_sync_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    final_sync_seconds = time.perf_counter() - final_sync_started
    if device.type == "cuda":
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_reserved(device)
    else:
        elapsed = time.perf_counter() - started
        peak = 0
    stage_report = stage_clock.report()
    operator_intervals = trace.collect_cuda_intervals()

    export_started = time.perf_counter()
    camera = torch.cat([part.detach().cpu().to(torch.float32) for part in output_parts], dim=0)
    export_seconds = time.perf_counter() - export_started
    _require(tuple(camera.shape) == (LOGICAL_BATCH, 1, 400, 400)
             and torch.isfinite(camera).all(), "saved W/pixel output is malformed")
    _require(math.isfinite(total_loss) and all(torch.isfinite(value).all() for value in gradients.values()),
             "production loss or gradient is nonfinite")
    return {"loss": total_loss, "gradients": gradients, "w_per_pixel": camera,
            "elapsed_seconds": elapsed, "artifact_export_seconds": export_seconds,
            "gradient_transfer_wall_seconds": gradient_transfer_seconds,
            "final_cuda_sync_wall_seconds": final_sync_seconds,
            "peak_reserved_vram_bytes": peak, "stage_timing": stage_report,
            "operator_cuda_intervals": operator_intervals,
            "operator_host_wall_seconds": trace.host_seconds.copy(),
            "operator_call_counts": trace.counts.copy()}


def _run_partial(model: torch.nn.Module, pixels: torch.Tensor, rows: torch.Tensor,
                 beta: float, eta: float, device: torch.device) -> dict[str, Any]:
    """Run the first production microbatch as the checkpoint-off resource probe."""
    _require(rows.numel() >= MICROBATCH, "fixed batch does not contain a complete microbatch2")
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    target, _ = baseline._image_batch(pixels, rows[:MICROBATCH], device)
    result = model.forward_from_image(target)
    prediction = _fixed_readout_prediction(result["physical_w_per_pixel"])
    mse = (prediction - target.to(torch.float64)).square().mean(dtype=torch.float64)
    prediction28 = F.adaptive_avg_pool2d(prediction.to(torch.float32), (28, 28))
    target28 = F.adaptive_avg_pool2d(target, (28, 28))
    structure = 1.0 - baseline._ssim_batch(prediction28, target28).to(torch.float64).mean()
    loss = beta * mse + eta * structure
    (loss * (MICROBATCH / LOGICAL_BATCH)).backward()
    gradients = {name: parameter.grad.detach().cpu().clone()
                 for name, parameter in zip(("encoder_phase", "decoder_phase"),
                                            (model.encoder_phase, model.decoder_phase), strict=True)}
    camera = result["physical_w_per_pixel"].detach().cpu().to(torch.float32)
    _require(torch.isfinite(camera).all() and math.isfinite(float(loss.detach().cpu()))
             and all(torch.isfinite(value).all() for value in gradients.values()),
             "checkpoint-off microbatch2 probe produced nonfinite values")
    return {"loss_weighted_into_batch10": float(loss.detach().cpu()) * MICROBATCH / LOGICAL_BATCH,
            "gradients": gradients, "w_per_pixel": camera}


def _write_case(output_dir: Path, case: str, result: dict[str, Any], rows: torch.Tensor,
                checkpoint: bool, *, include_gradients: bool = True) -> dict[str, Any]:
    tensor_path = output_dir / "measurements" / f"{case}.pt"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "chen_lightgen_400_rs_cost_case.v1", "task_id": TASK_ID,
               "case": case, "rows": rows.cpu().long(),
               "w_per_pixel": result["w_per_pixel"].cpu().to(torch.float32),
               "gradients": result["gradients"] if include_gradients else None,
               "loss": float(result["loss"]), "checkpoint_propagation": checkpoint,
               "elapsed_seconds": float(result["elapsed_seconds"]),
               "peak_reserved_vram_bytes": int(result["peak_reserved_vram_bytes"]),
               "operator_call_counts": result["operator_call_counts"],
               "stage_timing": result["stage_timing"]}
    atomic_torch_save_new(tensor_path, payload)
    loaded = torch.load(tensor_path, map_location="cpu", weights_only=False)
    _require(loaded.get("schema") == payload["schema"] and loaded.get("case") == case
             and torch.equal(loaded["rows"], rows.cpu().long())
             and torch.equal(loaded["w_per_pixel"], payload["w_per_pixel"])
             and loaded["operator_call_counts"] == payload["operator_call_counts"]
             and loaded["stage_timing"] == payload["stage_timing"]
             and loaded["elapsed_seconds"] == payload["elapsed_seconds"]
             and isinstance(loaded.get("gradients"), dict)
             and set(loaded["gradients"]) == {"encoder_phase", "decoder_phase"}
             and all(torch.equal(loaded["gradients"][name], payload["gradients"][name])
                     for name in payload["gradients"]),
             f"measurement write/readback failed: {tensor_path}")
    return {key: value for key, value in result.items()
            if key not in ("gradients", "w_per_pixel")} | {
                "case": case, "checkpoint_propagation": checkpoint,
                "tensor_payload": str(tensor_path), "loss": float(result["loss"]),
                "rows": rows.cpu().tolist(),
                "w_per_pixel_relative_error_vs_checkpoint_on_1": None}


def _compare_case(result: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    loss_error = abs(float(result["loss"]) - float(reference["loss"])) \
        / max(abs(float(reference["loss"])), 1.0e-30)
    gradient_errors = _gradient_errors(result["gradients"], reference["gradients"])
    has_output_reference = isinstance(reference.get("w_per_pixel"), torch.Tensor)
    output_error = (_relative_error(result["w_per_pixel"], reference["w_per_pixel"])
                    if has_output_reference else None)
    loss_gradient_pass = loss_error <= LOSS_RTOL and max(gradient_errors.values()) <= GRAD_RTOL
    output_pass = not has_output_reference or output_error <= OUTPUT_RTOL
    return {"loss_relative_error": loss_error,
            "gradient_relative_error_by_layer": gradient_errors,
            "max_gradient_relative_error": max(gradient_errors.values()),
            "w_per_pixel_relative_error": output_error,
            "w_per_pixel_reference_available": has_output_reference,
            "status": ("NUMERICALLY_EQUIVALENT" if has_output_reference else
                       LOSS_GRADIENT_ONLY_STATUS) if loss_gradient_pass and output_pass
            else "NUMERICAL_NOT_EQUIVALENT"}


def _safe_speed_bound(total_seconds: float, component_seconds: float) -> dict[str, Any]:
    _require(math.isfinite(total_seconds) and total_seconds > 0.0
             and math.isfinite(component_seconds) and 0.0 <= component_seconds <= total_seconds,
             "component timing overlaps or was double-counted beyond the measured wall time")
    share = component_seconds / total_seconds
    if share >= 1.0:
        speedup = None
    else:
        speedup = 1.0 / (1.0 - share)
    return {"component_seconds": component_seconds, "share_of_wall": share,
            "optimistic_max_speedup_if_component_were_free": speedup,
            "assumption": "component intervals are non-overlapping; upper bound only"}


def _validate_profile_summaries(summaries: list[dict[str, Any]]) -> None:
    _require(len(summaries) == 2
             and {record.get("direction") for record in summaries} == {"forward", "adjoint"},
             "bounded profiler must contain exactly one forward and one adjoint RS call")
    for record in summaries:
        _require(isinstance(record.get("top_operators"), list)
                 and record.get("profile_scope", "").startswith("one production cached RS operator call")
                 and "total_cuda_time_us" not in record,
                 "profiler data is incomplete or contains an inclusive aggregate that could be double-counted")


def _validate_call_counts(counts: dict[str, int], *, checkpoint: bool) -> None:
    forward = counts.get("rs_forward_calls", 0)
    adjoint = counts.get("rs_adjoint_calls", 0)
    linear_forward = counts.get("linear_propagation_forward_calls", 0)
    allowed_forward_counts = ({BASE_RS_FORWARD_CALLS_PER_BATCH10,
                               2 * BASE_RS_FORWARD_CALLS_PER_BATCH10} if checkpoint else
                              {BASE_RS_FORWARD_CALLS_PER_BATCH10})
    _require(forward in allowed_forward_counts and adjoint == BASE_RS_FORWARD_CALLS_PER_BATCH10
             and linear_forward == forward,
             f"production RS forward/adjoint/recomputation count is incomplete: {counts}")
    expected_total_calls = forward + adjoint
    _require(counts.get("center_observation_requests", 0) == 0
             and counts.get("full_window_observation_requests", 0) == 0,
             "normal production batch unexpectedly requested diagnostic observations")
    _require(counts.get("kernel_cache_misses", 0) == 0
             and counts.get("kernel_cache_hits", 0) == expected_total_calls
             and counts.get("pair_spectrum_materializations", 0)
             == expected_total_calls * PAIR_MATERIALIZATIONS_PER_RS_CALL,
             f"measured batch did not use a warm cache for every RS forward and adjoint: {counts}")


def _validate_measurement_order(order: list[str], *, checkpoint_off_eligible: bool) -> None:
    expected = list(FULL_BATCH_ORDER) if checkpoint_off_eligible else [
        "checkpoint_on_1", "checkpoint_on_2"]
    _require(order == expected, "full batch10 measurements are not in the fixed predeclared order")


def _load_saved_case(output_dir: Path, case: str, rows: torch.Tensor,
                     reference: dict[str, Any], *, checkpoint: bool,
                     qualification_reference: dict[str, Any] | None = None
                     ) -> tuple[dict[str, Any], dict[str, Any]]:
    summary_path = output_dir / f"{case}_summary.json"
    tensor_path = output_dir / "measurements" / f"{case}.pt"
    _require(summary_path.is_file() and tensor_path.is_file(),
             f"saved cost measurement is incomplete: {case}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = torch.load(tensor_path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "chen_lightgen_400_rs_cost_case.v1"
             and payload.get("task_id") == TASK_ID and payload.get("case") == case
             and payload.get("checkpoint_propagation") is checkpoint,
             f"saved cost measurement identity is invalid: {case}")
    _require(isinstance(payload.get("rows"), torch.Tensor)
             and torch.equal(payload["rows"].long().cpu(), rows.long().cpu()),
             f"saved measurement rows differ from the fixed batch10: {case}")
    gradients = payload.get("gradients")
    camera = payload.get("w_per_pixel")
    _require(isinstance(gradients, dict) and isinstance(camera, torch.Tensor)
             and tuple(camera.shape) == (LOGICAL_BATCH, 1, 400, 400)
             and torch.isfinite(camera).all()
             and all(isinstance(value, torch.Tensor) and torch.isfinite(value).all()
                     for value in gradients.values()),
             f"saved production outputs or gradients are missing or nonfinite: {case}")
    _validate_gradient_shapes(gradients)
    loss = float(payload.get("loss", math.nan))
    elapsed = float(payload.get("elapsed_seconds", math.nan))
    peak = int(payload.get("peak_reserved_vram_bytes", 0))
    _require(math.isfinite(loss) and math.isfinite(elapsed) and elapsed > 0.0 and peak > 0,
             f"saved timing, loss, or memory measurement is invalid: {case}")
    counts = payload.get("operator_call_counts")
    _require(isinstance(counts, dict), f"saved production call counts are missing: {case}")
    _validate_call_counts(counts, checkpoint=checkpoint)
    stage_timing = payload.get("stage_timing")
    _require(isinstance(stage_timing, dict)
             and isinstance(stage_timing.get("host_wall_seconds_by_stage"), dict)
             and isinstance(stage_timing.get("cuda_event_elapsed_ms_by_stage"), dict),
             f"saved stage timings are incomplete: {case}")
    _require(summary.get("case") == case
             and summary.get("checkpoint_propagation") is checkpoint
             and summary.get("rows") == rows.long().cpu().tolist()
             and summary.get("loss") == loss
             and summary.get("elapsed_seconds") == elapsed
             and summary.get("peak_reserved_vram_bytes") == peak
             and summary.get("operator_call_counts") == counts
             and summary.get("stage_timing") == stage_timing
             and summary.get("tensor_payload", "").endswith(f"/measurements/{case}.pt"),
             f"saved cost summary does not match its production tensor payload: {case}")
    _require(summary.get("cache_hit_ratio") == 1.0
             and summary.get("cache_cold_build") is None
             and isinstance(summary.get("cache_info_before_repetition"), dict)
             and summary["cache_info_before_repetition"].get("cache_bytes", 0) > 0,
             f"saved cost measurement did not use an established warm cache: {case}")

    run = {"loss": loss, "gradients": gradients, "w_per_pixel": camera,
           "operator_call_counts": counts, "elapsed_seconds": elapsed,
           "stage_timing": stage_timing}
    comparison = _validate_checkpoint_qualification(run, reference)
    for name, value in (("loss_relative_error", comparison["loss_relative_error"]),
                        ("max_gradient_relative_error", comparison["max_gradient_relative_error"]),
                        ("w_per_pixel_relative_error", comparison["w_per_pixel_relative_error"])):
        saved = summary.get(name)
        _require(saved is not None and math.isclose(float(saved), float(value),
                 rel_tol=1.0e-6, abs_tol=1.0e-12),
                 f"saved cost summary comparison no longer matches its tensor payload: {case}/{name}")
    _require(summary.get("status") == comparison["status"] == "NUMERICALLY_EQUIVALENT",
             f"saved cost measurement is not numerically qualified: {case}")
    if qualification_reference is not None:
        qualification_comparison = _compare_case(run, qualification_reference)
        _require(qualification_comparison["status"] in QUALIFIED_REFERENCE_STATUSES,
                 f"saved first production case differs from its qualification reference: {case}")
        summary["saved_qualification_comparison"] = {
            "status": qualification_comparison["status"],
            "loss_relative_error": qualification_comparison["loss_relative_error"],
            "max_gradient_relative_error": qualification_comparison[
                "max_gradient_relative_error"],
            "w_per_pixel_reference_available": qualification_comparison[
                "w_per_pixel_reference_available"],
        }
    return summary, run


def _build_cost_report(*, args: argparse.Namespace, preflight: dict[str, Any],
                       rows: torch.Tensor, resource_probe: dict[str, Any],
                       cases: dict[str, Any], execution_cache_build: dict[str, Any] | None,
                       profile_summaries: list[dict[str, Any]],
                       execution_elapsed_seconds: float,
                       parameter_and_buffer_state_unchanged: bool,
                       finalization_mode: str,
                       recovery_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    checkpoint_off_eligible = resource_probe.get("status") == "NUMERICALLY_EQUIVALENT"
    measurement_order = list(cases)
    _validate_measurement_order(measurement_order, checkpoint_off_eligible=checkpoint_off_eligible)
    measurements = [value["elapsed_seconds"] for value in cases.values()]
    enabled_times = [cases[name]["elapsed_seconds"] for name in cases
                     if name.startswith("checkpoint_on")]
    off_times = [cases[name]["elapsed_seconds"] for name in cases
                 if name.startswith("checkpoint_off")]
    _require(enabled_times and math.isfinite(execution_elapsed_seconds)
             and execution_elapsed_seconds > 0.0,
             "final cost report lacks completed execution timing")
    speed_ratio = (sum(enabled_times) / len(enabled_times)
                   / (sum(off_times) / len(off_times))) if checkpoint_off_eligible else None
    checkpoint_on_mean = sum(enabled_times) / len(enabled_times)
    status = ("VALID_COMPLETED_COST_ATTRIBUTION" if checkpoint_off_eligible else
              "VALID_COMPLETED_CHECKPOINT_OFF_RESOURCE_BLOCKED"
              if resource_probe.get("status") == "PREFLIGHT_BLOCKED_RESOURCE" else
              "VALID_COMPLETED_CHECKPOINT_OFF_NUMERICALLY_INELIGIBLE")
    preflight_receipt = preflight["receipt"]
    beta = float(preflight_receipt["beta"])
    eta = float(preflight_receipt["eta"])
    cache_info = cases["checkpoint_on_1"].get("cache_info_before_repetition")
    _require(isinstance(cache_info, dict) and cache_info.get("cache_bytes", 0) > 0,
             "execution warmup cache evidence is missing from the first production case")
    report = {
        "status": status,
        "task_id": TASK_ID,
        "elapsed_seconds": execution_elapsed_seconds,
        "execution_elapsed_seconds": execution_elapsed_seconds,
        "preflight_elapsed_seconds": float(preflight_receipt["elapsed_seconds"]),
        "total_diagnostic_elapsed_seconds": (execution_elapsed_seconds
                                              + float(preflight_receipt["elapsed_seconds"])),
        "result_role": "engineering_cost_attribution_only",
        "explicit_inputs": preflight_receipt["explicit_inputs"],
        "input_identity": {"loaded_checkpoint_step": 30_000, "historical_endpoint_step": 500,
                           "precision": "complex64", "branch": "B",
                           "logical_batch": LOGICAL_BATCH, "microbatch": MICROBATCH,
                           "beta": beta, "eta": eta, "rows": rows.tolist(),
                           "labels": list(range(10)),
                           "data_role": "fixed_fit100_first_row_per_class"},
        "preflight_status": preflight_receipt["status"],
        "checkpoint_off_microbatch2_resource_probe": resource_probe,
        "checkpoint_off_eligible": checkpoint_off_eligible,
        "measurement_order": measurement_order,
        "cases": cases,
        "cache_cold_build_preflight": preflight_receipt["cache_cold_build"],
        "cache_cold_build_execution_warmup": execution_cache_build,
        "checkpoint_on_mean_seconds": checkpoint_on_mean,
        "checkpoint_on_range_seconds": [min(enabled_times), max(enabled_times)],
        "checkpoint_off_mean_seconds": None if not off_times else
            sum(off_times) / len(off_times),
        "checkpoint_off_range_seconds": None if not off_times else
            [min(off_times), max(off_times)],
        "checkpoint_off_speed_ratio_vs_on": speed_ratio,
        "mean_across_measured_settings_seconds": sum(measurements) / len(measurements),
        "checkpoint_on_uninstrumented_preflight_seconds": preflight_receipt[
            "checkpoint_on_uninstrumented_production_cost_seconds"],
        "saved_qualification_batch10_seconds": preflight_receipt[
            "saved_qualification_cost_seconds"],
        "production_checkpoint_on_batch10_mean_seconds": checkpoint_on_mean,
        "production_checkpoint_on_batch10_range_seconds": [min(enabled_times), max(enabled_times)],
        "production_checkpoint_on_compute_only_estimate_500_steps_seconds":
            500 * checkpoint_on_mean,
        "production_checkpoint_on_compute_only_estimate_30000_steps_seconds":
            30_000 * checkpoint_on_mean,
        "production_checkpoint_on_compute_only_estimate_30000_steps_days":
            30_000 * checkpoint_on_mean / 86_400,
        "checkpoint_off_compute_only_estimate_500_steps_seconds": None if not off_times else
            500 * sum(off_times) / len(off_times),
        "checkpoint_off_compute_only_estimate_30000_steps_seconds": None if not off_times else
            30_000 * sum(off_times) / len(off_times),
        "checkpoint_off_compute_only_estimate_30000_steps_days": None if not off_times else
            30_000 * sum(off_times) / len(off_times) / 86_400,
        "six_hour_30000_step_goal_seconds_per_step": 6 * 3600 / 30_000,
        "six_hour_required_speedup_vs_observed": checkpoint_on_mean / (6 * 3600 / 30_000),
        "rs_event_interval_cost_bound_checkpoint_on": _safe_speed_bound(
            checkpoint_on_mean,
            sum(sum(value.get("operator_cuda_intervals", {}).get("by_direction_ms", {}).values())
                / 1000.0 for name, value in cases.items() if name.startswith("checkpoint_on"))
            / len(enabled_times)),
        "rs_event_interval_cost_bound_checkpoint_off": None if not off_times else
            _safe_speed_bound(
                sum(off_times) / len(off_times),
                sum(sum(value.get("operator_cuda_intervals", {}).get("by_direction_ms", {}).values())
                    / 1000.0 for name, value in cases.items()
                    if name.startswith("checkpoint_off")) / len(off_times)),
        "operator_profile_scope": "one cached RS forward plus one cached RS adjoint; excluded from throughput",
        "operator_profile_summaries": profile_summaries,
        "diagnostic_center_calls": 0,
        "diagnostic_full_window_calls": 0,
        "optimizer_updates": 0,
        "optimizer_calls": 0,
        "adam_calls": 0,
        "generation_samples": 0,
        "evaluator_forward_calls": 0,
        "parameter_and_buffer_state_unchanged": parameter_and_buffer_state_unchanged,
        "hardware_samples_path": str(_resolve(args.output_dir) / "hardware_samples.json"),
        "inclusive_and_asynchronous_timing_not_additive": True,
        "finalization_mode": finalization_mode,
        "recovery_receipt": recovery_receipt,
        "next_action": "recommend_one_exact_acceleration_only_after_reviewing_measured_dominant_cost",
    }
    _require(parameter_and_buffer_state_unchanged,
             "parameter or buffer state integrity was not verified")
    return report


def _write_reports(output_dir: Path, report: dict[str, Any]) -> None:
    measurement_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    for case, result in report["cases"].items():
        counts = result["operator_call_counts"]
        timing = result["stage_timing"]
        measurement_rows.append({
            "case": case, "checkpoint_propagation": result["checkpoint_propagation"],
            "elapsed_seconds": result["elapsed_seconds"],
            "artifact_export_seconds": result["artifact_export_seconds"],
            "gradient_transfer_wall_seconds": result["gradient_transfer_wall_seconds"],
            "final_cuda_sync_wall_seconds": result["final_cuda_sync_wall_seconds"],
            "peak_reserved_vram_bytes": result["peak_reserved_vram_bytes"],
            "loss": result["loss"], "loss_relative_error": result.get("loss_relative_error"),
            "max_gradient_relative_error": result.get("max_gradient_relative_error"),
            "w_per_pixel_relative_error": result.get("w_per_pixel_relative_error"),
            "forward_calls": counts.get("rs_forward_calls", 0),
            "adjoint_calls": counts.get("rs_adjoint_calls", 0),
            "cache_hits": counts.get("kernel_cache_hits", 0),
            "cache_misses": counts.get("kernel_cache_misses", 0),
            "pair_spectrum_materializations": counts.get("pair_spectrum_materializations", 0),
            "center_observation_requests": counts.get("center_observation_requests", 0),
            "full_window_observation_requests": counts.get("full_window_observation_requests", 0),
            "cuda_event_elapsed_ms_by_stage": json.dumps(timing["cuda_event_elapsed_ms_by_stage"]),
            "host_wall_seconds_by_stage": json.dumps(timing["host_wall_seconds_by_stage"]),
        })
        for stage, seconds in timing["host_wall_seconds_by_stage"].items():
            stage_rows.append({"case": case, "stage": stage, "host_wall_seconds": seconds,
                               "cuda_event_elapsed_ms": timing["cuda_event_elapsed_ms_by_stage"].get(stage)})
    for filename, records in (("batch10_cost.csv", measurement_rows),
                              ("stage_cost.csv", stage_rows)):
        fields = list(records[0]) if records else ["case", "stage"]
        with (output_dir / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)
        with (output_dir / filename).open("r", encoding="utf-8-sig", newline="") as handle:
            _require(sum(1 for _ in csv.DictReader(handle)) == len(records),
                     f"CSV write/readback count differs: {filename}")

    rows = []
    for name, item in report["cases"].items():
        rows.append("<tr>" + "".join(
            f"<td>{html.escape(str(value))}</td>" for value in (
                name, item["checkpoint_propagation"], f"{item['elapsed_seconds']:.3f}",
                f"{item.get('loss_relative_error', 0.0):.3e}",
                f"{item.get('max_gradient_relative_error', 0.0):.3e}",
                f"{item.get('w_per_pixel_relative_error', 0.0):.3e}")) + "</tr>")
    page = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>LightGen400 RS成本归因</title>"
            "<style>body{font:15px system-ui;max-width:1200px;margin:32px auto;padding:0 18px}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:7px;text-align:left}"
            "th{background:#f1f3f5}</style><body><h1>LightGen400 RS零更新成本归因</h1>"
            f"<p>状态：{html.escape(report['status'])}；Adam调用、优化更新、采样与E_D前向均为0。</p>"
            "<p>Profiler仅追踪一个缓存RS前向和一个伴随调用；其运行时间不进入吞吐量比较。"
            "主机时间与CUDA事件时间是并列观察，不相加。</p>"
            "<table><thead><tr><th>运行</th><th>重计算</th><th>秒／batch10</th>"
            "<th>损失差</th><th>梯度最大差</th><th>W/pixel差</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "<p><a href='final_report.json'>机器报告</a> · <a href='batch10_cost.csv'>成本</a> · "
            "<a href='stage_cost.csv'>阶段计时</a> · <a href='operator_profile.json'>算子追踪</a> · "
            "<a href='validation_receipt.json'>验证收据</a></p></body></html>")
    (output_dir / "comparison.html").write_text(page, encoding="utf-8")


def _validate_checkpoint_qualification(report: dict[str, Any], reference: dict[str, Any]
                                       ) -> dict[str, Any]:
    _require(tuple(report["w_per_pixel"].shape) == (LOGICAL_BATCH, 1, 400, 400)
             and torch.isfinite(report["w_per_pixel"]).all(),
             "production preflight did not capture a finite batch10 W/pixel output")
    _validate_gradient_shapes(report["gradients"])
    comparison = _compare_case(report, reference)
    _require(comparison["status"] in QUALIFIED_REFERENCE_STATUSES,
             "production batch10 re-render differs from saved qualified loss/gradient/W-per-pixel")
    return comparison


def _run_preflight(args: argparse.Namespace, output_dir: Path, device: torch.device) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to overwrite existing output directory: {output_dir}")
    reference_info = _validate_previous_evidence(args)
    context = _load_context(args, device)
    _require(_manifest_identity(str(context["paths"]["source_checkpoint"]))
             == _manifest_identity(reference_info["paths"]["source_checkpoint"]),
             "loaded model checkpoint path differs from saved qualification")
    reference = _load_reference_gradient(reference_info, context["batch10_rows"])
    free = torch.cuda.mem_get_info(device)[0]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _require(torch.cuda.is_available(), "CUDA is required for this cost preflight")
    _require(os.statvfs(output_dir.parent).f_bavail * os.statvfs(output_dir.parent).f_frsize
             >= OUTPUT_LIMIT_BYTES,
             "cloud output filesystem has less than the 512 MiB result budget free")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "progress.json", {
        "task_id": TASK_ID, "stage": "checkpoint_on_uninstrumented_production_batch10_preflight",
        "optimizer_updates": 0, "adam_calls": 0,
    })

    model = compact._make_model(context["state_phases"]["source_step_30000"], device,
                                "complex64", checkpoint_propagation=True)
    compact.prior._set_trainable(model)
    state = _state_snapshot(model)
    with _OperationTrace(model) as trace:
        _run_partial(model, context["pixels"], context["batch10_rows"],
                     float(context["beta"]), float(context["eta"]), device)
        _require(trace.counts.get("kernel_cache_builds", 0) == 1
                 and trace.counts.get("kernel_cache_hits", 0) > 0,
                 "representative production microbatch did not build and reuse the cached RS kernel")
    output_parts: list[torch.Tensor] = []
    original_forward = model.forward_from_image

    def capture_output(target: torch.Tensor) -> dict[str, Any]:
        result = original_forward(target)
        output_parts.append(result["physical_w_per_pixel"].detach())
        return result

    model.forward_from_image = capture_output
    try:
        loss, gradients, elapsed, peak = q12._run_microbatch(
            model, context["pixels"], context["batch10_rows"], float(context["beta"]),
            float(context["eta"]), "B", MICROBATCH, device)
    finally:
        del model.forward_from_image
    export_started = time.perf_counter()
    camera = torch.cat(output_parts, dim=0).detach().cpu().to(torch.float32)
    export_seconds = time.perf_counter() - export_started
    run = {"loss": loss, "gradients": gradients, "w_per_pixel": camera,
           "elapsed_seconds": elapsed, "peak_reserved_vram_bytes": peak,
           "artifact_export_seconds": export_seconds}
    equality = _validate_checkpoint_qualification(run, reference)
    _require(_state_matches(model, state), "model parameters or buffers changed during zero-update preflight")
    report = {"status": "PREFLIGHT_PASSED", "task_id": TASK_ID,
              "elapsed_seconds": time.perf_counter() - args.started_at,
              "explicit_inputs": reference_info["paths"],
              "loaded_checkpoint_step": 30_000, "historical_endpoint_step": 500,
              "rows": context["batch10_rows"].tolist(), "labels": list(range(10)),
              "data_role": "fixed_fit100_first_row_per_class",
        "saved_qualification_cost_seconds": float(reference_info["metric"]["elapsed_seconds"]),
              "precision": "complex64", "loss": "B=beta*MSE+eta*(1-SSIM(pool28(pred),pool28(target)))",
              "beta": float(context["beta"]), "eta": float(context["eta"]),
              "saved_reference_comparison": equality,
              "checkpoint_on_uninstrumented_production_cost_seconds": run["elapsed_seconds"],
              "cache_cold_build": trace.cache_build,
              "cache_after_preflight": model.propagator.operator.cache_info,
              "free_cuda_bytes_before_call": free,
              "peak_reserved_vram_bytes": run["peak_reserved_vram_bytes"],
              "optimizer_calls": 0, "optimizer_updates": 0, "adam_calls": 0,
              "generation_samples": 0, "evaluator_forward_calls": 0,
              "input_state_unchanged": True}
    atomic_torch_save_new(output_dir / "preflight_reference.pt", {
        "schema": "chen_lightgen_400_rs_cost_preflight.v1", "loss": run["loss"],
        "gradients": run["gradients"], "rows": context["batch10_rows"],
        "w_per_pixel": run["w_per_pixel"], "comparison": equality,
        "explicit_inputs": reference_info["paths"],
    })
    _write_json(output_dir / "preflight_receipt.json", report, new=True)
    _write_json(output_dir / "progress.json", {
        "task_id": TASK_ID, "stage": "preflight_complete", "status": report["status"],
        "elapsed_seconds": report["elapsed_seconds"], "optimizer_updates": 0, "adam_calls": 0,
    })
    return report


def _load_preflight(output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    receipt_path = output_dir / "preflight_receipt.json"
    reference_path = output_dir / "preflight_reference.pt"
    _require(output_dir.is_dir() and receipt_path.is_file() and reference_path.is_file(),
             "execution requires a complete saved --preflight-only receipt")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt.get("task_id") == TASK_ID and receipt.get("status") == "PREFLIGHT_PASSED"
             and receipt.get("explicit_inputs") == _explicit_paths(args)
             and receipt.get("optimizer_updates") == receipt.get("adam_calls") == 0,
             "preflight receipt is incomplete or bound to different explicit inputs")
    payload = torch.load(reference_path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "chen_lightgen_400_rs_cost_preflight.v1"
             and payload.get("explicit_inputs") == _explicit_paths(args)
             and payload.get("comparison", {}).get("status") in QUALIFIED_REFERENCE_STATUSES,
             "saved preflight values are incomplete or do not qualify the production output")
    _require(not (output_dir / "final_report.json").exists(),
             "cost attribution is already complete; refusing to overwrite its final report")
    return {"receipt": receipt, "reference": payload}


def _run_execution(args: argparse.Namespace, output_dir: Path, device: torch.device) -> dict[str, Any]:
    previous = _validate_previous_evidence(args)
    preflight = _load_preflight(output_dir, args)
    context = _load_context(args, device)
    rows = context["batch10_rows"]
    old_reference = _load_reference_gradient(previous, rows)
    reference = preflight["reference"]
    preflight_rows = reference.get("rows")
    _require(isinstance(preflight_rows, torch.Tensor), "saved preflight rows are missing")
    _validate_saved_rows(preflight_rows.long().cpu().tolist(), rows)
    _validate_checkpoint_qualification({"loss": reference["loss"],
                                        "gradients": reference["gradients"],
                                        "w_per_pixel": reference["w_per_pixel"]},
                                       {"loss": reference["loss"],
                                        "gradients": reference["gradients"],
                                        "w_per_pixel": reference["w_per_pixel"]})
    reference_case = {"loss": float(reference["loss"]), "gradients": reference["gradients"],
                      "w_per_pixel": reference["w_per_pixel"]}
    phases = context["state_phases"]["source_step_30000"]
    beta, eta = float(context["beta"]), float(context["eta"])
    model = compact._make_model(phases, device, "complex64", checkpoint_propagation=True)
    compact.prior._set_trainable(model)
    original_state = _state_snapshot(model)

    progress = {"task_id": TASK_ID, "stage": "checkpoint_off_microbatch2_preflight",
                "optimizer_updates": 0, "adam_calls": 0}
    _write_json(output_dir / "progress.json", progress)
    resource_probe: dict[str, Any]
    with _OperationTrace(model) as trace:
        # The cache is built by the saved preflight call; this model is new, so build it
        # with an actual production microbatch before either timed checkpoint setting.
        warm = _run_partial(model, context["pixels"], rows, beta, eta, device)
        _require(model.propagator.operator.cache_info["cache_bytes"] > 0,
                 "production warmup did not build the complex64 kernel cache")
        trace.reset_counts()
        _set_checkpoint_propagation(model, False)
        try:
            candidate_probe = _run_partial(model, context["pixels"], rows, beta, eta, device)
            loss_error = abs(candidate_probe["loss_weighted_into_batch10"]
                             - warm["loss_weighted_into_batch10"]) \
                / max(abs(warm["loss_weighted_into_batch10"]), 1.0e-30)
            resource_probe = {"status": "COMPLETED", "loss_relative_error": loss_error,
                "w_per_pixel_relative_error": _relative_error(
                candidate_probe["w_per_pixel"], warm["w_per_pixel"]),
                "gradient_relative_error_by_layer": _gradient_errors(
                    candidate_probe["gradients"], warm["gradients"]),
                "cuda_free_bytes_after_probe": torch.cuda.mem_get_info(device)[0],
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)}
            resource_probe["max_gradient_relative_error"] = max(
                resource_probe["gradient_relative_error_by_layer"].values())
            resource_probe["status"] = "NUMERICALLY_EQUIVALENT" if (
                resource_probe["loss_relative_error"] <= LOSS_RTOL
                and resource_probe["w_per_pixel_relative_error"] <= OUTPUT_RTOL
                and resource_probe["max_gradient_relative_error"] <= GRAD_RTOL
            ) else "NUMERICAL_NOT_EQUIVALENT"
        except torch.cuda.OutOfMemoryError as exc:
            model.zero_grad(set_to_none=True)
            del exc
            torch.cuda.empty_cache()
            resource_probe = {"status": "PREFLIGHT_BLOCKED_RESOURCE",
                              "reason": "checkpoint-off microbatch2 CUDA allocation failed",
                              "cuda_free_bytes_after_failure": torch.cuda.mem_get_info(device)[0]}
        _require(_state_matches(model, original_state),
                 "model parameters or buffers changed during checkpoint schedule probe")

        del warm
        checkpoint_off_eligible = resource_probe["status"] == "NUMERICALLY_EQUIVALENT"
        _write_json(output_dir / "checkpoint_off_resource_probe.json", resource_probe, new=True)

        cases: dict[str, Any] = {}
        if checkpoint_off_eligible:
            run_order = FULL_BATCH_ORDER
        else:
            run_order = ("checkpoint_on_1", "checkpoint_on_2")
        for index, case_name in enumerate(run_order):
            enabled = case_name.startswith("checkpoint_on")
            _set_checkpoint_propagation(model, enabled)
            model.zero_grad(set_to_none=True)
            trace.reset_counts()
            progress.update({"stage": case_name, "completed_cases": list(cases),
                             "optimizer_updates": 0, "adam_calls": 0})
            _write_json(output_dir / "progress.json", progress)
            run = _run_batch10(model, context["pixels"], rows, beta, eta, device, trace)
            if not enabled:
                comparison = _compare_case(run, reference_case)
                _require(comparison["status"] == "NUMERICALLY_EQUIVALENT",
                         "checkpoint-off batch10 differs from the saved checkpoint-on preflight")
                run.update(comparison)
            else:
                comparison = _compare_case(run, reference_case)
                run.update(comparison)
            run["checkpoint_propagation"] = enabled
            run["cache_info_before_repetition"] = model.propagator.operator.cache_info
            run["cache_cold_build"] = None
            run["cache_hit_ratio"] = trace.counts.get("kernel_cache_hits", 0) / max(
                trace.counts.get("kernel_cache_hits", 0) + trace.counts.get("kernel_cache_misses", 0), 1)
            run["profile_summaries"] = trace.profile_summaries
            summary = _write_case(output_dir, case_name, run, rows, enabled)
            cases[case_name] = summary
            _write_json(output_dir / f"{case_name}_summary.json", summary, new=True)
            _validate_call_counts(run["operator_call_counts"], checkpoint=enabled)
            if index == 0 and case_name == "checkpoint_on_1":
                _require(_validate_checkpoint_qualification(run, old_reference)["status"]
                         in QUALIFIED_REFERENCE_STATUSES,
                         "first measured batch10 fails saved qualification reference")
                reference_case = {"loss": run["loss"], "gradients": run["gradients"],
                                  "w_per_pixel": run["w_per_pixel"]}
            del run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # A distinct full production batch10 trace profiles only its first cached
        # RS forward and first adjoint. It is excluded from all throughput values.
        _set_checkpoint_propagation(model, True)
        model.zero_grad(set_to_none=True)
        trace.profile = True
        trace.reset_counts()
        profile_run = _run_batch10(model, context["pixels"], rows, beta, eta, device, trace)
        profile_comparison = _compare_case(profile_run, reference_case)
        _require(profile_comparison["status"] == "NUMERICALLY_EQUIVALENT",
                 "profiled production batch changed the measured values")
        _write_json(output_dir / "operator_profile.json", {
            "status": "PROFILED_TWO_CALLS",
            "scope": "only the first cached RS forward and first cached RS adjoint in one batch10 call",
            "profile_summaries": trace.profile_summaries,
            "operator_call_counts": trace.counts,
            "profiled_batch_elapsed_seconds_excluded_from_cost_comparison": profile_run["elapsed_seconds"],
            "nested_cuda_operator_totals_must_not_be_added": True,
        }, new=True)
        _validate_profile_summaries(trace.profile_summaries)
        _validate_call_counts(trace.counts, checkpoint=True)
        _require(_state_matches(model, original_state),
                 "model parameters or buffers changed during final profiling call")

    execution_cache_build = trace.cache_build
    execution_elapsed = time.perf_counter() - args.started_at
    state_unchanged = _state_matches(model, original_state)
    report = _build_cost_report(
        args=args, preflight=preflight, rows=rows, resource_probe=resource_probe,
        cases=cases, execution_cache_build=execution_cache_build,
        profile_summaries=trace.profile_summaries,
        execution_elapsed_seconds=execution_elapsed,
        parameter_and_buffer_state_unchanged=state_unchanged,
        finalization_mode="LIVE_MEASUREMENT")
    _require(report["parameter_and_buffer_state_unchanged"],
             "model parameters or buffers changed after the final diagnostic")
    _write_json(output_dir / "cost_attribution_report.json", report, new=True)
    _write_reports(output_dir, report)
    receipt = {"task_id": TASK_ID, "status": report["status"],
               "case_count": len(cases), "measurement_order": report["measurement_order"],
               "checkpoint_off_eligible": checkpoint_off_eligible,
               "optimizer_updates": 0, "adam_calls": 0, "generation_samples": 0,
               "evaluator_forward_calls": 0,
               "parameter_and_buffer_state_unchanged": report["parameter_and_buffer_state_unchanged"],
               "all_cases_written_and_read_back": True,
               "all_cases_used_warm_cache": True,
               "profile_scope_verified": len(trace.profile_summaries) == 2}
    _write_json(output_dir / "validation_receipt.json", receipt, new=True)
    _write_json(output_dir / "final_report.json", report, new=True)
    _write_json(output_dir / "progress.json", {
        "task_id": TASK_ID, "stage": "complete", "status": report["status"],
        "elapsed_seconds": report["elapsed_seconds"], "optimizer_updates": 0, "adam_calls": 0,
        "completed_cases": list(cases),
    })
    _require(sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
             <= OUTPUT_LIMIT_BYTES, "formal results exceed the 512 MiB output budget")
    return report


def _finalize_saved_execution(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    finalization_started = time.perf_counter()
    previous = _validate_previous_evidence(args)
    preflight = _load_preflight(output_dir, args)
    recovery_path = output_dir / "finalization_recovery_receipt.json"
    _require(recovery_path.is_file(),
             "saved-measurement finalization requires an explicit recovery receipt")
    recovery = json.loads(recovery_path.read_text(encoding="utf-8"))
    _require(recovery.get("schema") == "chen_lightgen_400_rs_finalization_recovery.v1"
             and recovery.get("task_id") == TASK_ID
             and isinstance(recovery.get("run_id"), str)
             and len(recovery["run_id"]) == 32
             and all(character in "0123456789abcdef" for character in recovery["run_id"])
             and recovery.get("run_state") == "COMMAND_EXITED_NONZERO"
             and recovery.get("exit_code") == 1
             and recovery.get("failure_kind") == "KeyError"
             and recovery.get("failure_key")
             == "checkpoint_on_uninstrumented_production_batch10_cost_seconds"
             and recovery.get("completed_cases") == list(FULL_BATCH_ORDER)
             and recovery.get("operator_profile_completed") is True
             and recovery.get("state_unchanged_assertion_passed_before_failure") is True,
             "recovery receipt does not identify the verified report-assembly-only failure")
    _require(not args.preflight_only, "saved-measurement finalization is not a preflight")
    existing_outputs = ("cost_attribution_report.json", "batch10_cost.csv", "stage_cost.csv",
                        "comparison.html", "validation_receipt.json", "final_report.json")
    _require(not any((output_dir / name).exists() for name in existing_outputs),
             "refusing to overwrite partially finalized result artifacts")

    reference_payload = preflight["reference"]
    rows = reference_payload.get("rows")
    _require(isinstance(rows, torch.Tensor), "saved preflight reference rows are missing")
    rows = rows.long().cpu()
    _validate_saved_rows(preflight["receipt"].get("rows"), rows)
    reference = {"loss": float(reference_payload["loss"]),
                 "gradients": reference_payload["gradients"],
                 "w_per_pixel": reference_payload["w_per_pixel"]}
    qualification_reference = _load_reference_gradient(previous, rows)
    cases: dict[str, Any] = {}
    for case in FULL_BATCH_ORDER:
        summary, _payload = _load_saved_case(
            output_dir, case, rows, reference,
            checkpoint=case.startswith("checkpoint_on"),
            qualification_reference=(qualification_reference if case == "checkpoint_on_1" else None))
        cases[case] = summary

    resource_path = output_dir / "checkpoint_off_resource_probe.json"
    profile_path = output_dir / "operator_profile.json"
    hardware_path = output_dir / "hardware_samples.json"
    for path in (resource_path, profile_path, hardware_path):
        _require(path.is_file(), f"saved execution evidence is incomplete: {path.name}")
    resource_probe = json.loads(resource_path.read_text(encoding="utf-8"))
    _require(resource_probe.get("status") == "NUMERICALLY_EQUIVALENT"
             and resource_probe.get("loss_relative_error", math.inf) <= LOSS_RTOL
             and resource_probe.get("w_per_pixel_relative_error", math.inf) <= OUTPUT_RTOL
             and resource_probe.get("max_gradient_relative_error", math.inf) <= GRAD_RTOL,
             "saved checkpoint-off microbatch2 resource probe is not numerically equivalent")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    _require(profile.get("status") == "PROFILED_TWO_CALLS"
             and profile.get("nested_cuda_operator_totals_must_not_be_added") is True
             and float(profile.get("profiled_batch_elapsed_seconds_excluded_from_cost_comparison", 0.0)) > 0.0,
             "saved bounded operator profile is incomplete or includes throughput timing")
    profile_summaries = profile.get("profile_summaries")
    _validate_profile_summaries(profile_summaries)
    _validate_call_counts(profile.get("operator_call_counts", {}), checkpoint=True)
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    samples = hardware.get("samples")
    _require(isinstance(samples, list) and len(samples) > 0
             and all(sample.get("status") == "sampled" for sample in samples)
             and math.isfinite(float(hardware.get("elapsed_seconds", math.nan)))
             and float(hardware["elapsed_seconds"]) > 0.0,
             "saved execution hardware telemetry is incomplete or has failed samples")

    cache_info = cases["checkpoint_on_1"].get("cache_info_before_repetition", {})
    execution_cache_build = {
        "host_seconds": float(cache_info.get("cache_build_seconds", math.nan)),
        "cuda_event_ms": None,
        "cache_bytes": int(cache_info.get("cache_bytes", 0)),
        "canonical_kernel_count": int(cache_info.get("canonical_kernel_count", 0)),
        "compute_precision": cache_info.get("compute_precision"),
        "geometry": cache_info.get("cached_geometry"),
        "recovered_from": "first_saved_production_case_cache_metadata",
    }
    _require(math.isfinite(execution_cache_build["host_seconds"])
             and execution_cache_build["host_seconds"] > 0.0
             and execution_cache_build["cache_bytes"] > 0,
             "execution cold-cache build evidence is incomplete")
    _require(all(bool(cases[name]["checkpoint_propagation"]) for name in
                 ("checkpoint_on_1", "checkpoint_on_2"))
             and all(not bool(cases[name]["checkpoint_propagation"]) for name in
                     ("checkpoint_off_1", "checkpoint_off_2")),
             "saved ABBA case settings are not aligned")

    report = _build_cost_report(
        args=args, preflight=preflight, rows=rows, resource_probe=resource_probe,
        cases=cases, execution_cache_build=execution_cache_build,
        profile_summaries=profile_summaries,
        execution_elapsed_seconds=float(hardware["elapsed_seconds"]),
        parameter_and_buffer_state_unchanged=True,
        finalization_mode="SAVED_MEASUREMENT_FINALIZATION_AFTER_REPORT_ASSEMBLY_ERROR",
        recovery_receipt=recovery)
    report["finalization_elapsed_seconds"] = time.perf_counter() - finalization_started
    report["measurement_runtime_source"] = "saved hardware_samples.json elapsed_seconds"
    report["state_invariance_evidence"] = (
        "The original execution reached its explicit _require(_state_matches(...)) check; "
        "the recorded exception occurred later while assembling the missing preflight receipt key.")
    _write_json(output_dir / "cost_attribution_report.json", report, new=True)
    _write_reports(output_dir, report)
    receipt = {"task_id": TASK_ID, "status": report["status"],
               "case_count": len(cases), "measurement_order": report["measurement_order"],
               "checkpoint_off_eligible": report["checkpoint_off_eligible"],
               "optimizer_updates": 0, "adam_calls": 0, "generation_samples": 0,
               "evaluator_forward_calls": 0,
               "parameter_and_buffer_state_unchanged": True,
               "all_cases_written_and_read_back": True,
               "all_cases_revalidated_from_saved_tensor_payloads": True,
               "all_cases_used_warm_cache": True,
               "profile_scope_verified": len(profile_summaries) == 2,
               "finalization_mode": report["finalization_mode"],
               "recovery_receipt": recovery_path.name,
               "new_optical_forward_or_backward_calls": 0}
    _write_json(output_dir / "validation_receipt.json", receipt, new=True)
    _write_json(output_dir / "final_report.json", report, new=True)
    _write_json(output_dir / "progress.json", {
        "task_id": TASK_ID, "stage": "complete", "status": report["status"],
        "elapsed_seconds": report["elapsed_seconds"], "optimizer_updates": 0, "adam_calls": 0,
        "completed_cases": report["measurement_order"],
        "finalization_mode": report["finalization_mode"],
    })
    _write_json(output_dir / "heartbeat.json", {
        "task_id": TASK_ID, "stage": "finalization_complete",
        "progress": json.loads((output_dir / "progress.json").read_text(encoding="utf-8")),
        "new_optical_forward_or_backward_calls": 0,
    })
    _require(sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
             <= OUTPUT_LIMIT_BYTES, "formal results exceed the 512 MiB output budget")
    return report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--source-start-state", type=Path, required=True)
    parser.add_argument("--source-endpoint-checkpoint", type=Path, required=True)
    parser.add_argument("--fixed-population", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--qualification-report", type=Path, required=True)
    parser.add_argument("--structure-coefficient", type=Path, required=True)
    parser.add_argument("--qualification-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--finalize-saved-measurements-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    args.started_at = time.perf_counter()
    output_dir = _resolve(args.output_dir)
    _require(not (args.preflight_only and args.finalize_saved_measurements_only),
             "preflight and saved-measurement finalization are separate entry modes")
    if args.finalize_saved_measurements_only:
        result = _finalize_saved_execution(args, output_dir)
    elif args.preflight_only:
        device = torch.device("cuda")
        _require(torch.cuda.is_available(),
                 "CUDA is required; CPU is not a substitute for this cost measurement")
        with _HardwareSampler(output_dir):
            result = _run_preflight(args, output_dir, device)
    else:
        device = torch.device("cuda")
        _require(torch.cuda.is_available(),
                 "CUDA is required; CPU is not a substitute for this cost measurement")
        _require(output_dir.is_dir(), "run --preflight-only first with this exact output directory")
        with _HardwareSampler(output_dir):
            result = _run_execution(args, output_dir, device)
    if output_dir.exists():
        _require(sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file())
                 <= OUTPUT_LIMIT_BYTES, "formal results exceed the 512 MiB output budget")
    print(json.dumps(_safe_json(result), ensure_ascii=False, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
