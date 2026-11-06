"""Benchmark cached finite-pixel RS propagation and compact LightGen400 evidence.

This engineering-only entry point reuses the fixed q12/q16 population and
saved observations. It performs no scientific model selection or training.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    PIXEL_PITCH_M,
)
from ldm_project.v2.application.research.lightgen.chen_lightgen_400_finite_pixel import (
    FinitePixelChenLightGen400,
    FinitePixelConfig,
    FinitePixelRSLinear,
)
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_json
from scripts.research.lightgen import execute_chen_lightgen_400_finite_pixel_reconstruction_ab_v1 as prior
from scripts.research.lightgen import execute_chen_lightgen_400_mnist_baseline_v1 as baseline
from scripts.research.lightgen import execute_chen_lightgen_400_q12_trainability_v1 as q12


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CHEN_LIGHTGEN_400_FAST_RS_COMPACT_V1"
QUALIFICATION_TASK_ID = "CHEN_LIGHTGEN_400_FAST_RS_QUALIFICATION_V2"
ORDER = 12
REFERENCE_ORDER = 16
WINDOW = 4096
FIELD_TOLERANCE = 1.0e-4
LOGICAL_BATCH = 10
OUTPUT_LIMIT_BYTES = 512 * 1024**2
STATES = q12.STATES


class HarnessError(RuntimeError):
    """An input, storage, heartbeat, or recovery contract was invalid."""


class NumericalQualificationError(RuntimeError):
    """A complete numerical comparison exceeded its frozen tolerance."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessError(message)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _safe_json(value: Any) -> Any:
    return baseline._json_safe(value)


def _write_json(path: Path, value: dict[str, Any], *, new: bool = False) -> None:
    safe = _safe_json(value)
    if new:
        from ldm_project.v2.evidence import atomic_write_new_json
        atomic_write_new_json(path, safe)
    else:
        atomic_write_json(path, safe)


def _write_torch(path: Path, value: Any, *, new: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if new:
        atomic_torch_save_new(path, value)
        loaded = torch.load(path, map_location="cpu", weights_only=False)
        _require(type(loaded) is type(value), f"write/readback type differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _require(type(loaded) is type(value), f"write/readback type differs: {path}")


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _require_output_budget(path: Path) -> int:
    size = _tree_bytes(path)
    _require(size <= OUTPUT_LIMIT_BYTES,
             f"permanent result directory exceeded its 512 MiB budget: {size} bytes")
    return size


class Heartbeat:
    def __init__(self, output_dir: Path, progress: dict[str, Any], interval_s: float = 60.0):
        self.output_dir = output_dir
        self.progress = progress
        self.interval_s = interval_s
        self.stop = threading.Event()
        self.failure: str | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(self.interval_s):
            try:
                device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
                _write_json(self.output_dir / "heartbeat.json", {
                    "task_id": self.progress.get("task_id", TASK_ID),
                    "stage": self.progress.get("stage"),
                    "last_progress": self.progress.copy(),
                    "unix_time": time.time(),
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else None,
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(device) if device.type == "cuda" else None,
                })
            except Exception as exc:  # fail at the next safe boundary
                self.failure = f"{type(exc).__name__}: {exc}"
                self.stop.set()

    def __enter__(self) -> "Heartbeat":
        self.thread.start()
        return self

    def check(self) -> None:
        if self.failure is not None:
            raise HarnessError(f"heartbeat failed; stop at safe boundary: {self.failure}")

    def __exit__(self, *_exc: object) -> None:
        self.stop.set()
        self.thread.join(timeout=2.0)


def _input_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**{
        "source_checkpoint": _resolve(args.source_checkpoint),
        "source_report": _resolve(args.source_report),
        "source_start_state": _resolve(args.source_start_state),
        "source_endpoint_checkpoint": _resolve(args.source_endpoint_checkpoint),
        "fixed_population": _resolve(args.fixed_population),
        "config": _resolve(args.config),
        "data": _resolve(args.data),
        "evaluator": _resolve(args.evaluator),
        "qualification_report": _resolve(args.qualification_report),
        "qualification_cache": _resolve(args.reference_run) / "qualification_raw",
        "output_dir": _resolve(args.output_dir),
    })


def _load_context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    loaded = q12._load_context(_input_namespace(args), device)
    _require(loaded["paths"]["qualification_report"].is_file(), "prior qualification report is missing")
    return loaded


def _load_eta(path: Path) -> float:
    try:
        payload = json.loads(_resolve(path).read_text(encoding="utf-8"))
        eta = float(payload["eta"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise HarnessError(f"explicit structure coefficient is unreadable: {_resolve(path)}") from exc
    _require(math.isfinite(eta) and eta > 0.0, "structure coefficient eta must be finite and positive")
    return eta


def _make_model(phases: dict[str, torch.Tensor], device: torch.device, precision: str,
                *, checkpoint_propagation: bool) -> FinitePixelChenLightGen400:
    model = FinitePixelChenLightGen400(FinitePixelConfig(
        quadrature_order=ORDER,
        window_cells=WINDOW,
        distance_m=3.0e-3,
        checkpoint_propagation=checkpoint_propagation,
        propagation_backend="rs_linear_cached",
        rs_compute_precision=precision,
        rs_pair_block=4,
    )).to(device)
    with torch.no_grad():
        model.encoder_phase.copy_(phases["encoder_phase"].to(device))
        model.decoder_phase.copy_(phases["decoder_phase"].to(device))
    return model


def _set_phases(model: FinitePixelChenLightGen400, phases: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        model.encoder_phase.copy_(phases["encoder_phase"].to(model.encoder_phase.device))
        model.decoder_phase.copy_(phases["decoder_phase"].to(model.decoder_phase.device))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_info(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "cuda_available": torch.cuda.is_available()}
    properties = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        "device": torch.cuda.get_device_name(device),
        "total_memory_bytes": properties.total_memory,
        "free_memory_bytes": free,
        "mem_total_bytes": total,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def _operator_probe(context: dict[str, Any], output_dir: Path, device: torch.device) -> dict[str, Any]:
    row = int(context["population"]["confirm_rows_by_class"][0, 0])
    target, _ = baseline._image_batch(context["pixels"], torch.tensor([row]), device)
    seed_model = _make_model(context["state_phases"][STATES[0]], device, "complex128",
                             checkpoint_propagation=False)
    intensity = seed_model._initial_node_intensities(target)
    density_scale = math.sqrt(
        baseline.SOURCE_POWER_W /
        ((400 * PIXEL_PITCH_M) * (400 * PIXEL_PITCH_M))
    )
    field_seed = torch.sqrt(intensity).to(torch.complex128) * density_scale
    field_seed = seed_model._phase(field_seed, seed_model.encoder_phase[0])
    _sync(device)

    records: dict[str, Any] = {}
    reference = None
    for name, precision in (("old_uncached_complex128", None),
                            ("cached_complex128", "complex128"),
                            ("cached_complex64", "complex64")):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if precision is None:
            operator = FinitePixelRSLinear(ORDER, (400, 400), FinitePixelConfig(
                quadrature_order=ORDER, window_cells=WINDOW, checkpoint_propagation=False,
                propagation_backend="rs_linear",
            )).to(device)
            cache_info: dict[str, Any] = {"cache_bytes": 0, "cache_build_seconds": 0.0}
            field = field_seed
        else:
            operator = FinitePixelRSLinear(ORDER, (400, 400), FinitePixelConfig(
                quadrature_order=ORDER, window_cells=WINDOW, checkpoint_propagation=False,
                propagation_backend="rs_linear_cached", rs_compute_precision=precision,
                rs_pair_block=4,
            )).to(device)
            cache_info = {}
            field = field_seed if precision == "complex128" else field_seed.to(torch.complex64)
        _sync(device)
        started = time.perf_counter()
        with torch.no_grad():
            nodes, power, center = operator(
                field, output_shape=(400, 400), capture_center=True, full_window_power=False
            )
        _sync(device)
        elapsed = time.perf_counter() - started
        _require(tuple(nodes.shape) == (1, ORDER**2, 400, 400)
                 and bool(torch.isfinite(nodes).all()) and bool(torch.isfinite(power).all())
                 and bool(torch.isfinite(center).all()) and math.isfinite(elapsed) and elapsed > 0.0,
                 f"{name} actual 400-pixel single-hop output is invalid")
        if precision is not None:
            cache_info = operator.operator.cache_info
        record = {
            "elapsed_seconds_including_cold_build": elapsed,
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
            "node_field_shape": list(nodes.shape),
            "node_field_dtype": str(nodes.dtype),
            "input_field_dtype": str(field.dtype),
            "cache": cache_info,
        }
        if reference is None:
            reference = nodes.detach().cpu()
        else:
            base = reference.to(torch.complex128)
            candidate = nodes.detach().cpu().to(torch.complex128)
            record["node_field_relative_l2_vs_uncached"] = float(
                torch.linalg.vector_norm(candidate - base) /
                torch.linalg.vector_norm(base).clamp_min(1.0e-300)
            )
            record["node_field_max_abs_vs_uncached"] = float((candidate - base).abs().max())
        records[name] = record
        del operator, nodes, power, center
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "status": "MEASURED",
        "row": row,
        "label": 0,
        "operator_scope": "one production first-encoder propagation from a fixed MNIST field",
        "records": records,
        "c64_one_hop_pass": records["cached_complex64"].get("node_field_relative_l2_vs_uncached", math.inf)
                                 <= FIELD_TOLERANCE,
        "c128_one_hop_pass": records["cached_complex128"].get("node_field_relative_l2_vs_uncached", math.inf)
                                  <= FIELD_TOLERANCE,
    }
    _write_json(output_dir / "operator_benchmark.json", result, new=True)
    return result


def _estimate_remaining_work(operator: dict[str, Any]) -> dict[str, Any]:
    records = operator["records"]
    precision_record = {name: records[f"cached_{name}"]
                        for name in ("complex128", "complex64")}
    map_seconds: dict[str, float] = {}
    for precision, record in precision_record.items():
        elapsed = float(record["elapsed_seconds_including_cold_build"])
        build = float(record["cache"].get("cache_build_seconds", 0.0))
        _require(math.isfinite(elapsed) and elapsed > 0.0
                 and math.isfinite(build) and 0.0 <= build <= elapsed,
                 f"{precision} one-hop timing cannot support a remaining-work estimate")
        # The production probe performs one node-field map and one center-field
        # map. Divide the measured post-build time between those two maps.
        map_seconds[precision] = (elapsed - build) / 2.0

    # This qualification contains no recovery updates. Counts include both
    # precision paths, four real batch10 cases per precision, and B-loss FD.
    map_equivalents = {"complex128": {
        "compact_full_chain": 20 * 7 * 2,
        "candidate_gradients_forward_backward": 4 * 7 * 2,
        "b_total_loss_gradients_forward_backward": 2 * 7 * 2,
        "finite_difference_perturbed_forwards": 44 * 7,
        "batch10_cost_cases": 4 * 10 * 7 * 2,
    }, "complex64": {
        "compact_full_chain": 20 * 7 * 2,
        "candidate_gradients_forward_backward": 4 * 7 * 2,
        "b_total_loss_gradients_forward_backward": 2 * 7 * 2,
        "finite_difference_perturbed_forwards": 44 * 7,
        "batch10_cost_cases": 4 * 10 * 7 * 2,
    }}
    cache_build_counts = {"complex128": {"400_roi": 5, "500_roi": 4},
                          "complex64": {"400_roi": 5, "500_roi": 4}}
    cache_build_scale_500_roi = 1.266
    estimated_maps = {
        precision: map_seconds[precision] * sum(map_equivalents[precision].values())
        for precision in map_equivalents
    }
    estimated_cache_builds = {
        "complex128": float(precision_record["complex128"]["cache"]["cache_build_seconds"])
        * (cache_build_counts["complex128"]["400_roi"]
           + cache_build_counts["complex128"]["500_roi"] * cache_build_scale_500_roi),
        "complex64": float(precision_record["complex64"]["cache"]["cache_build_seconds"])
        * (cache_build_counts["complex64"]["400_roi"]
           + cache_build_counts["complex64"]["500_roi"] * cache_build_scale_500_roi),
    }
    return {
        "status": "MEASURED_OPERATION_COUNT_ESTIMATE",
        "map_seconds_from_one_hop_after_cache_build": map_seconds,
        "map_equivalents": map_equivalents,
        "estimated_map_seconds": estimated_maps,
        "estimated_cache_rebuild_seconds": estimated_cache_builds,
        "estimated_remaining_seconds": sum(estimated_maps.values()) + sum(estimated_cache_builds.values()),
        "assumptions": [
            "one-hop probe contains one node map and one center map",
            "500x500 cache-build cost scales by FFT area ratio 900^2/800^2",
            "map counts include both precision paths, B-loss directions and batch10 cost cases",
            "estimate excludes Python setup, file read/write and checkpoint overhead",
            "estimated duration is advisory and never truncates a required measurement",
        ],
    }


def _load_prior_raw(reference_run: Path, stage: str, state: str,
                    row: int, label: int, order: int,
                    context: dict[str, Any] | None = None) -> dict[str, Any]:
    if context is not None:
        cache = context.get("_raw_reference_cache", {})
        cached = cache.get((stage, state, row, label, order))
        if cached is not None:
            return cached
    path = prior._raw_observation_path(reference_run, stage, state, row, order)
    _require(path.is_file(), f"explicit archived numerical reference is missing: {path}")
    return prior._load_full_chain_raw(reference_run, stage, state, row, label, order)


def _validate_reference_bundle(args: argparse.Namespace,
                               context: dict[str, Any]) -> dict[str, Any]:
    reference_run = _resolve(args.reference_run)
    raw_root = reference_run / "qualification_raw"
    gradient_root = _resolve(args.gradient_reference_dir)
    rows = context["population"]["confirm_rows_by_class"][:, 0].long()
    raw_requests = [
        (stage, state, int(row), label, order)
        for state in STATES
        for label, row in enumerate(rows)
        for stage, order in (("candidate", ORDER), ("reference", REFERENCE_ORDER))
    ]
    expected_raw = {
        prior._raw_observation_path(reference_run, stage, state, row, order).name
        for stage, state, row, _label, order in raw_requests
    }
    actual_raw = {path.name for path in raw_root.glob("*.pt")} if raw_root.is_dir() else set()
    _require(raw_root.is_dir() and actual_raw == expected_raw,
             f"reference bundle raw files differ: expected {len(expected_raw)}, found {len(actual_raw)}")
    raw_checks = []
    raw_cache = {}
    for stage, state, row, label, order in raw_requests:
        payload = _load_prior_raw(reference_run, stage, state, row, label, order)
        result = payload["result"]
        planes = result["finite_pixel_plane_values"]
        _require(len(planes) == 7 and all(
            set(plane) == {"pixel_center", "pixel_area_average", "pixel_intensity_average"}
            and tuple(plane["pixel_center"].shape) == (1, 500, 500)
            and tuple(plane["pixel_area_average"].shape) == (1, 1, 500, 500)
            and tuple(plane["pixel_intensity_average"].shape) == (1, 1, 500, 500)
            and all(isinstance(plane[name], torch.Tensor) and bool(torch.isfinite(plane[name]).all())
                    for name in ("pixel_center", "pixel_area_average", "pixel_intensity_average"))
            for plane in planes),
            f"reference plane values are incomplete or nonfinite: {payload['raw_path']}")
        raw_checks.append({"stage": stage, "state": state, "row": row,
                           "label": label, "order": order,
                           "path": payload["raw_path"]})
        raw_cache[(stage, state, row, label, order)] = payload
        del result, planes

    gradient_requests = [
        (state, int(rows[label]), label)
        for state in STATES for label in (0, 1)
    ]
    expected_gradients = {
        f"gradients_{state}_row{row}_q12_q16.pt"
        for state, row, _label in gradient_requests
    }
    actual_gradients = {path.name for path in gradient_root.glob("gradients_*_q12_q16.pt")} \
        if gradient_root.is_dir() else set()
    _require(gradient_root.is_dir() and actual_gradients == expected_gradients,
             f"gradient reference files differ: expected {len(expected_gradients)}, found {len(actual_gradients)}")
    gradient_checks = []
    gradient_cache = {}
    for state, row, label in gradient_requests:
        path = gradient_root / f"gradients_{state}_row{row}_q12_q16.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        _require(payload.get("state") == state and payload.get("row") == row
                 and payload.get("label") == label,
                 f"gradient reference identity mismatch: {path}")
        for name in ("q12_gradient_by_layer", "q16_gradient_by_layer"):
            gradients = payload.get(name)
            _require(isinstance(gradients, (tuple, list)) and len(gradients) == 6
                     and all(isinstance(gradient, torch.Tensor)
                             and tuple(gradient.shape) == (400, 400)
                             and bool(torch.isfinite(gradient).all()) for gradient in gradients),
                     f"gradient reference {name} is incomplete or nonfinite: {path}")
        gradient_checks.append({"state": state, "row": row, "label": label,
                                "path": str(path)})
        gradient_cache[(state, row, label)] = payload
    context["_raw_reference_cache"] = raw_cache
    context["_gradient_reference_cache"] = gradient_cache
    return {"status": "PASSED", "raw_expected": len(expected_raw),
            "raw_completed": len(raw_checks), "raw_checks": raw_checks,
            "gradient_expected": len(expected_gradients),
            "gradient_completed": len(gradient_checks),
            "gradient_checks": gradient_checks,
            "raw_geometry": "seven 4096-window q12/q16 full-chain observations with 500-pixel camera ROI",
            "gradient_geometry": "six finite phase-gradient layers per state and class 0/1"}


def _compact_result(result: dict[str, Any], row: int, label: int, state: str,
                    precision: str) -> dict[str, Any]:
    return {
        "schema": "chen_lightgen_400_fast_rs_compact_observation.v1",
        "state": state,
        "row": row,
        "label": label,
        "data_role": "canonical_MNIST_TRAIN_fixed_confirmation_row",
        "propagation_backend": "rs_linear_cached",
        "compute_precision": precision,
        "physical_units": {
            "physical_w_per_pixel": "W/pixel",
            "ols_coefficients": "sqrt(W)",
            "powers": "W",
        },
        "physical_w_per_pixel": result["physical_w_per_pixel"].detach().cpu().to(torch.float32),
        "ols_coefficients": result["ols_coefficients"].detach().cpu().to(torch.complex64),
        "input_power_w": result["input_power_w"].detach().cpu().to(torch.float64),
        "ols_input_power_w": result["ols_input_power_w"].detach().cpu().to(torch.float64),
        "ols_output_power_w": result["ols_output_power_w"].detach().cpu().to(torch.float64),
        "camera_roi_power_w": result["camera_roi_power_w"].detach().cpu().to(torch.float64),
        "plane_powers_w": result["plane_powers_w"].detach().cpu().to(torch.float64),
    }


def _compact_task_metrics(physical_w_per_pixel: torch.Tensor, target: torch.Tensor,
                          original_physical_w_per_pixel: torch.Tensor | None = None) -> dict[str, Any]:
    stored = physical_w_per_pixel.detach().cpu().to(torch.float64)
    target_cpu = target.detach().cpu().to(torch.float64)
    _require(stored.shape == target_cpu.shape and bool(torch.isfinite(stored).all())
             and bool(torch.isfinite(target_cpu).all()),
             "compact physical output or target has invalid shape/value")
    prediction = q12.FIXED_GAIN * stored
    mse = float((prediction - target_cpu).square().mean(dtype=torch.float64))
    result: dict[str, Any] = {
        "reader_gain_w_inverse": q12.FIXED_GAIN,
        "mse400_from_saved_float32_w_per_pixel": mse,
        "task_output_mean": float(prediction.mean(dtype=torch.float64)),
        "task_output_min": float(prediction.min()),
        "task_output_max": float(prediction.max()),
        "out_of_range_pixel_count": int(((prediction < 0.0) | (prediction > 1.0)).sum()),
        "pixel_count": prediction.numel(),
    }
    result["out_of_range_pixel_fraction"] = result["out_of_range_pixel_count"] / result["pixel_count"]
    if original_physical_w_per_pixel is not None:
        original = original_physical_w_per_pixel.detach().cpu().to(torch.float64)
        _require(original.shape == stored.shape and bool(torch.isfinite(original).all()),
                 "original physical output differs from the compact image layout")
        original_prediction = q12.FIXED_GAIN * original
        original_mse = float((original_prediction - target_cpu).square().mean(dtype=torch.float64))
        result["original_float64_mse400"] = original_mse
        result["task_output_relative_l2_from_original_float64"] = float(
            torch.linalg.vector_norm(prediction - original_prediction)
            / torch.linalg.vector_norm(original_prediction).clamp_min(1.0e-300)
        )
        result["mse_relative_error_from_original_float64"] = abs(mse - original_mse) \
            / max(abs(original_mse), 1.0e-30)
    return result


def _total_loss(result: dict[str, torch.Tensor], target: torch.Tensor, beta: float,
                eta: float, branch: str) -> torch.Tensor:
    prediction = q12.FIXED_GAIN * result["physical_w_per_pixel"].to(torch.float64)
    mse = (prediction - target.to(torch.float64)).square().mean(dtype=torch.float64)
    if branch == "A":
        return beta * mse
    _require(branch == "B", f"unsupported loss branch: {branch}")
    prediction28 = F.adaptive_avg_pool2d(prediction.to(torch.float32), (28, 28))
    target28 = F.adaptive_avg_pool2d(target, (28, 28))
    structure = 1.0 - baseline._ssim_batch(prediction28, target28).to(torch.float64).mean()
    return beta * mse + eta * structure


def _six_layer_gradients(gradients: tuple[torch.Tensor, torch.Tensor]) -> list[torch.Tensor]:
    by_layer = [group[index, 0].detach().cpu().to(torch.float64).clone()
                for group, count in zip(gradients, (2, 4), strict=True) for index in range(count)]
    _require(len(by_layer) == 6 and all(bool(torch.isfinite(value).all())
                                        and float(torch.linalg.vector_norm(value)) > 0.0
                                        for value in by_layer),
             "six-layer gradient is disconnected or nonfinite")
    return by_layer


def _state_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _require_state_restored(model: torch.nn.Module, snapshot: dict[str, torch.Tensor]) -> None:
    current = model.state_dict()
    _require(set(current) == set(snapshot)
             and all(torch.equal(current[name].detach().cpu(), value)
                     for name, value in snapshot.items()),
             "model parameters or buffers changed during the zero-update qualification")


def _measure_compact_full_chain(context: dict[str, Any], args: argparse.Namespace,
                                output_dir: Path, device: torch.device,
                                heartbeat: Heartbeat, progress: dict[str, Any]) -> dict[str, Any]:
    reference_run = _resolve(args.reference_run)
    measurements_dir = output_dir / "measurements"
    measurements_dir.mkdir(exist_ok=True)
    model = _make_model(context["state_phases"][STATES[0]], device, "complex128",
                         checkpoint_propagation=False)
    records: list[dict[str, Any]] = []
    worst_by_state: dict[str, tuple[float, dict[str, Any]]] = {}
    rows = context["population"]["confirm_rows_by_class"][:, 0].long()
    expected = len(STATES) * len(rows)
    for state in STATES:
        prior_metrics = []
        for metric_path in sorted(measurements_dir.glob(f"{state}_row*_label*.json")):
            prior_record = json.loads(metric_path.read_text(encoding="utf-8"))
            if prior_record.get("state") == state:
                prior_metrics.append(prior_record)
        if prior_metrics:
            worst = max(prior_metrics, key=lambda item: item["max_q12_q16_field_relative_l2"])
            worst_path = output_dir / "worst_fields" / f"{state}_row{worst['row']}.pt"
            if worst_path.is_file():
                worst_by_state[state] = (worst["max_q12_q16_field_relative_l2"],
                                         torch.load(worst_path, map_location="cpu", weights_only=False))
    for state in STATES:
        _set_phases(model, context["state_phases"][state])
        for label, row_value in enumerate(rows):
            heartbeat.check()
            row = int(row_value)
            sample_path = measurements_dir / f"{state}_row{row}_label{label}.pt"
            metric_path = measurements_dir / f"{state}_row{row}_label{label}.json"
            if sample_path.is_file():
                compact_payload = torch.load(sample_path, map_location="cpu", weights_only=False)
                record = compact_payload.get("metrics")
                _require(isinstance(record, dict) and record.get("row") == row
                         and record.get("label") == label and record.get("state") == state
                         and record.get("precision") == "complex128"
                         and compact_payload.get("physical_units", {}).get("physical_w_per_pixel") == "W/pixel",
                         f"saved compact measurement identity mismatch: {sample_path}")
                if not metric_path.is_file():
                    _write_json(metric_path, record, new=True)
                else:
                    existing_record = json.loads(metric_path.read_text(encoding="utf-8"))
                    _require(existing_record == _safe_json(record),
                             f"compact metric does not match saved producer: {metric_path}")
                target, _ = baseline._image_batch(context["pixels"], torch.tensor([row]), device)
                replay_metrics = _compact_task_metrics(compact_payload["physical_w_per_pixel"], target)
                saved_replay = record.get("compact_task_replay")
                _require(isinstance(saved_replay, dict)
                         and all(math.isclose(float(replay_metrics[key]), float(saved_replay[key]),
                                              rel_tol=1.0e-12, abs_tol=1.0e-14)
                                 for key in replay_metrics),
                         f"compact output does not reproduce its saved task readout: {sample_path}")
                del target
                records.append(record)
                expected_worst_path = output_dir / "worst_fields" / f"{state}_row{record['row']}.pt"
                current_best = max((item["max_q12_q16_field_relative_l2"] for item in records
                                    if item.get("state") == state), default=-1.0)
                if record["max_q12_q16_field_relative_l2"] >= current_best:
                    any_worst = list((output_dir / "worst_fields").glob(f"{state}_row*.pt"))
                    _require(bool(any_worst),
                             f"the worst candidate full field is missing for state={state}: {expected_worst_path}")
                continue
            if metric_path.is_file():
                raise HarnessError(f"orphan metric without its compact producer: {metric_path}")
            target, _ = baseline._image_batch(context["pixels"], torch.tensor([row]), device)
            model.eval()
            _sync(device)
            started = time.perf_counter()
            with torch.no_grad():
                result = model.forward_from_image(target, capture_fields=True)
            _sync(device)
            elapsed = time.perf_counter() - started
            candidate = {"result": result, "row": row, "label": label}
            old_q12 = _load_prior_raw(reference_run, "candidate", state, row, label, ORDER, context)
            q16 = _load_prior_raw(reference_run, "reference", state, row, label,
                                  REFERENCE_ORDER, context)
            residual_replay = prior._full_chain_residual(candidate, old_q12)
            residual_reference = prior._full_chain_residual(candidate, q16)
            values = {**residual_reference["per_plane_field_relative_l2"],
                      "ols_complex_relative_l2": residual_reference["ols_complex_relative_l2"],
                      "camera_roi_w_per_pixel_relative_l2": residual_reference["camera_roi_w_per_pixel_relative_l2"]}
            replay_values = {**residual_replay["per_plane_field_relative_l2"],
                             "ols_complex_relative_l2": residual_replay["ols_complex_relative_l2"],
                             "camera_roi_w_per_pixel_relative_l2": residual_replay["camera_roi_w_per_pixel_relative_l2"]}
            max_error = max(values.values())
            record = {
                "state": state, "row": row, "label": label, "precision": "complex128",
                "q12_vs_q16": residual_reference,
                "q12_cached_vs_historical_q12": residual_replay,
                "max_q12_q16_field_relative_l2": max_error,
                "max_q12_q16_value_relative_l2": max(values.values()),
                "max_cached_vs_historical_q12_relative_l2": max(replay_values.values()),
                "elapsed_seconds": elapsed,
                "status": "NUMERICALLY_QUALIFIED" if max(values.values()) <= FIELD_TOLERANCE
                          and max(replay_values.values()) <= FIELD_TOLERANCE else "NUMERICAL_NOT_QUALIFIED",
                "compact_path": str(sample_path),
            }
            if max_error > worst_by_state.get(state, (-1.0, {}))[0]:
                worst_by_state[state] = (max_error, {
                    "schema": "chen_lightgen_400_fast_rs_worst_field.v1",
                    "state": state, "row": row, "label": label,
                    "q12_candidate_fields": tuple({name: value.detach().cpu().clone()
                                                    for name, value in plane.items()}
                                                   for plane in result["finite_pixel_plane_values"]),
                    "q12_vs_q16_max_error": max_error,
                })
                worst_path = output_dir / "worst_fields" / f"{state}_row{row}.pt"
                _write_torch(worst_path, worst_by_state[state][1])
                for old in (output_dir / "worst_fields").glob(f"{state}_row*.pt"):
                    if old != worst_path:
                        old.unlink()
            compact_payload = _compact_result(result, row, label, state, "complex128")
            compact_task = _compact_task_metrics(
                compact_payload["physical_w_per_pixel"], target, result["physical_w_per_pixel"]
            )
            compact_readback_status = (
                "PASSED" if compact_task["task_output_relative_l2_from_original_float64"]
                <= FIELD_TOLERANCE
                and compact_task["mse_relative_error_from_original_float64"] <= FIELD_TOLERANCE
                else "FAILED"
            )
            record["compact_task_replay"] = compact_task
            record["compact_readback_status"] = compact_readback_status
            if compact_readback_status != "PASSED":
                record["status"] = "INVALID_HARNESS"
            compact_payload["metrics"] = record
            _write_torch(sample_path, compact_payload, new=True)
            compact_readback = torch.load(sample_path, map_location="cpu", weights_only=False)
            _require(torch.equal(compact_readback["physical_w_per_pixel"],
                                 compact_payload["physical_w_per_pixel"])
                     and torch.equal(compact_readback["ols_coefficients"],
                                     compact_payload["ols_coefficients"]),
                     f"compact tensor readback changed stored measurements: {sample_path}")
            replay_after_write = _compact_task_metrics(compact_readback["physical_w_per_pixel"], target)
            _require(all(math.isclose(float(replay_after_write[key]), float(compact_task[key]),
                                      rel_tol=1.0e-12, abs_tol=1.0e-14)
                           for key in replay_after_write),
                     f"compact readback changed derived task output/metrics: {sample_path}")
            _write_json(metric_path, record, new=True)
            _require_output_budget(output_dir)
            records.append(record)
            progress.update({"stage": "compact_full_chain", "state": state,
                             "row": row, "label": label, "completed": len(records),
                             "expected": expected, "optimizer_updates": 0, "adam_calls": 0})
            _write_json(output_dir / "progress.json", progress.copy())
            heartbeat.check()
            del result, candidate, old_q12, q16, target
            if device.type == "cuda":
                torch.cuda.empty_cache()
    statuses = [item["status"] for item in records]
    status = ("INVALID_HARNESS" if len(records) != expected or "INVALID_HARNESS" in statuses else
              "NUMERICALLY_QUALIFIED" if all(value == "NUMERICALLY_QUALIFIED" for value in statuses)
              else "NUMERICAL_NOT_QUALIFIED")
    return {"status": status, "records": records, "expected": expected,
            "completed": len(records), "tolerance": FIELD_TOLERANCE,
            "output_bytes_so_far": _tree_bytes(output_dir)}


def _measure_complex64_full_chain(context: dict[str, Any], args: argparse.Namespace,
                                  output_dir: Path, device: torch.device,
                                  heartbeat: Heartbeat, progress: dict[str, Any]) -> dict[str, Any]:
    reference_run = _resolve(args.reference_run)
    records: list[dict[str, Any]] = []
    rows = context["population"]["confirm_rows_by_class"][:, 0].long()
    model = _make_model(context["state_phases"][STATES[0]], device, "complex64",
                         checkpoint_propagation=False)
    expected = len(STATES) * len(rows)
    for state in STATES:
        _set_phases(model, context["state_phases"][state])
        for label, row_value in enumerate(rows):
            heartbeat.check()
            row = int(row_value)
            path = output_dir / "precision_c64" / f"{state}_row{row}_label{label}.json"
            sample_path = output_dir / "precision_c64" / f"{state}_row{row}_label{label}.pt"
            _require(path.exists() == sample_path.exists(),
                     f"partial complex64 compact evidence cannot be resumed: {sample_path}")
            if path.is_file():
                record = json.loads(path.read_text(encoding="utf-8"))
                _require(record.get("state") == state and record.get("row") == row
                         and record.get("label") == label and record.get("precision") == "complex64"
                         and sample_path.is_file(),
                         f"complex64 resume record identity mismatch: {path}")
                saved = torch.load(sample_path, map_location="cpu", weights_only=False)
                physical = saved.get("physical_w_per_pixel")
                ols = saved.get("ols_coefficients")
                _require(saved.get("state") == state and saved.get("row") == row
                         and saved.get("label") == label and saved.get("precision") == "complex64"
                         and saved.get("physical_units", {}).get("physical_w_per_pixel") == "W/pixel"
                         and isinstance(physical, torch.Tensor)
                         and tuple(physical.shape) == (1, 1, 400, 400)
                         and isinstance(ols, torch.Tensor) and ols.numel() > 0
                         and saved.get("metrics") == record
                         and bool(torch.isfinite(physical).all())
                         and bool(torch.isfinite(ols).all()),
                         f"complex64 compact values are invalid: {sample_path}")
                records.append(record)
                del saved
                continue
            target, _ = baseline._image_batch(context["pixels"], torch.tensor([row]), device)
            _sync(device)
            started = time.perf_counter()
            with torch.no_grad():
                result = model.forward_from_image(target, capture_fields=True)
            _sync(device)
            elapsed = time.perf_counter() - started
            candidate = {"result": result, "row": row, "label": label}
            old_q12 = _load_prior_raw(reference_run, "candidate", state, row, label, ORDER, context)
            q16 = _load_prior_raw(reference_run, "reference", state, row, label,
                                  REFERENCE_ORDER, context)
            replay = prior._full_chain_residual(candidate, old_q12)
            reference = prior._full_chain_residual(candidate, q16)
            errors = {**reference["per_plane_field_relative_l2"],
                      "ols_complex_relative_l2": reference["ols_complex_relative_l2"],
                      "camera_roi_w_per_pixel_relative_l2": reference["camera_roi_w_per_pixel_relative_l2"]}
            replay_errors = {**replay["per_plane_field_relative_l2"],
                             "ols_complex_relative_l2": replay["ols_complex_relative_l2"],
                             "camera_roi_w_per_pixel_relative_l2": replay["camera_roi_w_per_pixel_relative_l2"]}
            c128_path = output_dir / "measurements" / f"{state}_row{row}_label{label}.pt"
            _require(c128_path.is_file(), f"qualified complex128 compact output is missing: {c128_path}")
            c128 = torch.load(c128_path, map_location="cpu", weights_only=False)
            candidate_w = result["physical_w_per_pixel"].detach().cpu().to(torch.float64)
            reference_w = c128["physical_w_per_pixel"].to(torch.float64)
            w_error = float(torch.linalg.vector_norm(candidate_w - reference_w)
                            / torch.linalg.vector_norm(reference_w).clamp_min(1.0e-300))
            candidate_z = result["ols_coefficients"].detach().cpu().to(torch.complex128)
            reference_z = c128["ols_coefficients"].to(torch.complex128)
            overlap = torch.vdot(reference_z.reshape(-1), candidate_z.reshape(-1))
            phase = torch.exp(-1j * torch.angle(overlap))
            z_error = float(torch.linalg.vector_norm(candidate_z * phase - reference_z)
                            / torch.linalg.vector_norm(reference_z).clamp_min(1.0e-300))
            c128_error = max(w_error, z_error)
            record = {
                "state": state, "row": row, "label": label, "precision": "complex64",
                "max_value_relative_l2_vs_q16": max(errors.values()),
                "max_value_relative_l2_vs_historical_q12": max(replay_errors.values()),
                "physical_w_per_pixel_relative_l2_vs_complex128": w_error,
                "ols_phase_aligned_relative_l2_vs_complex128": z_error,
                "max_final_readout_relative_l2_vs_complex128": c128_error,
                "q16_errors": errors, "historical_q12_errors": replay_errors,
                "elapsed_seconds": elapsed, "compact_path": str(sample_path),
                "status": "NUMERICALLY_QUALIFIED" if max(errors.values()) <= FIELD_TOLERANCE
                          and max(replay_errors.values()) <= FIELD_TOLERANCE
                          and c128_error <= FIELD_TOLERANCE
                          else "NUMERICAL_NOT_QUALIFIED",
            }
            _write_torch(sample_path, {
                "schema": "chen_lightgen_400_fast_rs_compact_observation.v1",
                "state": state, "row": row, "label": label,
                "data_role": "canonical_MNIST_TRAIN_fixed_confirmation_row",
                "propagation_backend": "rs_linear_cached", "compute_precision": "complex64",
                "physical_units": {
                    "physical_w_per_pixel": "W/pixel", "ols_coefficients": "sqrt(W)",
                    "powers": "W",
                },
                "physical_w_per_pixel": result["physical_w_per_pixel"].detach().cpu().to(torch.float32),
                "ols_coefficients": result["ols_coefficients"].detach().cpu().to(torch.complex64),
                "input_power_w": result["input_power_w"].detach().cpu().to(torch.float64),
                "ols_input_power_w": result["ols_input_power_w"].detach().cpu().to(torch.float64),
                "ols_output_power_w": result["ols_output_power_w"].detach().cpu().to(torch.float64),
                "camera_roi_power_w": result["camera_roi_power_w"].detach().cpu().to(torch.float64),
                "plane_powers_w": result["plane_powers_w"].detach().cpu().to(torch.float64),
                "metrics": record,
            }, new=True)
            _write_json(path, record, new=True)
            _require_output_budget(output_dir)
            records.append(record)
            progress.update({"stage": "complex64_full_chain_screen", "state": state,
                             "row": row, "label": label, "completed": len(records),
                             "expected": expected})
            _write_json(output_dir / "progress.json", progress.copy())
            heartbeat.check()
            del target, result, candidate, old_q12, q16, c128, candidate_w, reference_w
            if device.type == "cuda":
                torch.cuda.empty_cache()
    qualified = len(records) == expected and all(
        item["status"] == "NUMERICALLY_QUALIFIED" for item in records)
    status = ("NUMERICALLY_QUALIFIED" if qualified else
              "INVALID_HARNESS" if len(records) != expected else "NUMERICAL_NOT_QUALIFIED")
    return {"status": status, "records": records, "expected": expected,
            "completed": len(records), "tolerance": FIELD_TOLERANCE,
            "retained_for_training": False}


def _candidate_gradients(context: dict[str, Any], reference_dir: Path, output_dir: Path,
                         device: torch.device, precision: str,
                         heartbeat: Heartbeat) -> dict[str, Any]:
    phases = context["state_phases"][STATES[0]]
    model = _make_model(phases, device, precision, checkpoint_propagation=True)
    prior._set_trainable(model)
    original_state = _state_snapshot(model)
    rows = context["population"]["confirm_rows_by_class"][:2, 0].long()
    records: list[dict[str, Any]] = []
    for state in STATES:
        _set_phases(model, context["state_phases"][state])
        for label, row_value in enumerate(rows):
            heartbeat.check()
            row = int(row_value)
            ref_path = reference_dir / f"gradients_{state}_row{row}_q12_q16.pt"
            _require(ref_path.is_file(), f"archived full-gradient reference is missing: {ref_path}")
            old = context.get("_gradient_reference_cache", {}).get((state, row, label))
            if old is None:
                old = torch.load(ref_path, map_location="cpu", weights_only=False)
            _require(old.get("state") == state and old.get("row") == row and old.get("label") == label,
                     f"archived gradient reference identity mismatch: {ref_path}")
            grad_path = output_dir / "gradients" / f"{precision}_{state}_row{row}.pt"
            metric_path = output_dir / "gradients" / f"{precision}_{state}_row{row}.json"
            _require(grad_path.exists() == metric_path.exists(),
                     f"partial candidate gradient evidence cannot be resumed: {grad_path}")
            if grad_path.is_file() and metric_path.is_file():
                record = json.loads(metric_path.read_text(encoding="utf-8"))
                payload = torch.load(grad_path, map_location="cpu", weights_only=False)
                _require(record.get("state") == state and record.get("row") == row
                         and record.get("label") == label and record.get("precision") == precision
                         and payload.get("state") == state and payload.get("row") == row
                         and payload.get("label") == label and payload.get("precision") == precision
                         and isinstance(payload.get("gradients_by_layer"), list)
                         and len(payload["gradients_by_layer"]) == 6
                         and all(torch.isfinite(value).all() for value in payload["gradients_by_layer"]),
                         f"candidate gradient resume identity mismatch: {metric_path}")
                errors_q12 = q12._compare_gradients(payload["gradients_by_layer"],
                                                    old["q12_gradient_by_layer"])
                errors_q16 = q12._compare_gradients(payload["gradients_by_layer"],
                                                    old["q16_gradient_by_layer"])
                _require(math.isclose(float(record.get("max_relative_l2_vs_historical_q12", math.nan)),
                                      errors_q12["max_relative_l2"], rel_tol=1.0e-12, abs_tol=1.0e-14)
                         and math.isclose(float(record.get("max_relative_l2_vs_historical_q16", math.nan)),
                                          errors_q16["max_relative_l2"], rel_tol=1.0e-12, abs_tol=1.0e-14),
                         f"saved candidate gradient report does not match its measured tensor: {metric_path}")
                records.append(record)
                del payload
                continue
            target, _ = baseline._image_batch(context["pixels"], torch.tensor([row]), device)
            model.zero_grad(set_to_none=True)
            result = model.forward_from_image(target)
            prediction = q12.FIXED_GAIN * result["physical_w_per_pixel"].to(torch.float64)
            loss = context["beta"] * (prediction - target.to(torch.float64)).square().mean(dtype=torch.float64)
            grads = torch.autograd.grad(loss, (model.encoder_phase, model.decoder_phase))
            by_layer = _six_layer_gradients(grads)
            _require(len(by_layer) == 6 and all(torch.isfinite(value).all() for value in by_layer)
                     and all(float(torch.linalg.vector_norm(value)) > 0.0 for value in by_layer),
                     f"{precision} q12 gradient has a disconnected/nonfinite layer for {state} row {row}")
            historical_q12 = old["q12_gradient_by_layer"]
            historical_q16 = old["q16_gradient_by_layer"]
            errors_q12 = q12._compare_gradients(by_layer, historical_q12)
            errors_q16 = q12._compare_gradients(by_layer, historical_q16)
            record = {
                "state": state, "row": row, "label": label, "precision": precision,
                "relative_l2_vs_historical_q12_by_layer": errors_q12["per_layer_relative_l2"],
                "relative_l2_vs_historical_q16_by_layer": errors_q16["per_layer_relative_l2"],
                "max_relative_l2_vs_historical_q12": errors_q12["max_relative_l2"],
                "max_relative_l2_vs_historical_q16": errors_q16["max_relative_l2"],
                "gradient_payload": str(grad_path),
                "status": "NUMERICALLY_QUALIFIED" if max(
                    errors_q12["max_relative_l2"], errors_q16["max_relative_l2"]) <= FIELD_TOLERANCE
                    else "NUMERICAL_NOT_QUALIFIED",
            }
            _write_torch(grad_path, {"state": state, "row": row, "label": label,
                                    "precision": precision, "gradients_by_layer": by_layer}, new=True)
            _write_json(metric_path, record, new=True)
            _require_output_budget(output_dir)
            records.append(record)
            del target, result, prediction, loss, grads, by_layer
            if device.type == "cuda":
                torch.cuda.empty_cache()
    status = ("NUMERICALLY_QUALIFIED" if len(records) == 4 and all(
        item["status"] == "NUMERICALLY_QUALIFIED" for item in records) else
        "INVALID_HARNESS" if len(records) != 4 else "NUMERICAL_NOT_QUALIFIED")
    _set_phases(model, context["state_phases"][STATES[0]])
    _require_state_restored(model, original_state)
    return {"status": status, "precision": precision, "records": records,
            "expected": 4, "tolerance": FIELD_TOLERANCE}


def _gradient_qualification(context: dict[str, Any], reference_dir: Path,
                            output_dir: Path, device: torch.device,
                            heartbeat: Heartbeat, precision: str) -> dict[str, Any]:
    current = _candidate_gradients(context, reference_dir, output_dir, device, precision, heartbeat)
    comparisons = []
    if precision == "complex64":
        for item in current["records"]:
            low_path = Path(item["gradient_payload"])
            high_path = output_dir / "gradients" / f"complex128_{item['state']}_row{item['row']}.pt"
            _require(high_path.is_file(), f"qualified complex128 gradient is missing: {high_path}")
            low_payload = torch.load(low_path, map_location="cpu", weights_only=False)
            high_payload = torch.load(high_path, map_location="cpu", weights_only=False)
            _require(low_payload.get("state") == high_payload.get("state") == item["state"]
                     and low_payload.get("row") == high_payload.get("row") == item["row"]
                     and low_payload.get("label") == high_payload.get("label") == item["label"],
                     "complex64 and complex128 gradient identities differ")
            compared = q12._compare_gradients(low_payload["gradients_by_layer"],
                                               high_payload["gradients_by_layer"])
            comparisons.append({"state": item["state"], "row": item["row"],
                                "label": item["label"],
                                "max_relative_l2": compared["max_relative_l2"],
                                "per_layer_relative_l2": compared["per_layer_relative_l2"]})
            del low_payload, high_payload
        if current["status"] == "NUMERICALLY_QUALIFIED":
            current["status"] = ("NUMERICALLY_QUALIFIED" if len(comparisons) == 4 and all(
                item["max_relative_l2"] <= FIELD_TOLERANCE for item in comparisons)
                else "NUMERICAL_NOT_QUALIFIED")
    result = {"status": current["status"], "precision": precision,
              "records": current["records"], "expected": current["expected"],
              "tolerance": FIELD_TOLERANCE,
              "comparison_to_complex128": comparisons}
    _write_json(output_dir / f"gradient_qualification_{precision}.json", result)
    return result


def _finite_difference_record_passes(checks: list[dict[str, Any]]) -> bool:
    return len(checks) == len(q12.FD_STEPS) and all(
        not item.get("degenerate") and item.get("same_sign")
        and math.isfinite(float(item.get("relative_error", math.nan)))
        and float(item["relative_error"]) <= q12.FINITE_DIFFERENCE_TOLERANCE
        for item in checks
    )


def _b_loss_gradient(context: dict[str, Any], model: FinitePixelChenLightGen400,
                     state: str, row: int, label: int, precision: str,
                     output_dir: Path, device: torch.device) -> list[torch.Tensor]:
    path = output_dir / "gradients" / f"{precision}_{state}_row{row}_branchB.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        _require(saved.get("state") == state and saved.get("row") == row
                 and saved.get("label") == label and saved.get("precision") == precision
                 and saved.get("branch") == "B"
                 and saved.get("beta") == float(context["beta"])
                 and saved.get("eta") == float(context["eta"]),
                 f"saved B-loss gradient identity differs: {path}")
        gradients = saved.get("gradients_by_layer")
        _require(isinstance(gradients, list) and len(gradients) == 6
                 and math.isfinite(float(saved.get("loss", math.nan)))
                 and all(isinstance(value, torch.Tensor) and tuple(value.shape) == (400, 400)
                         and bool(torch.isfinite(value).all()) for value in gradients),
                 f"saved B-loss gradient is incomplete or nonfinite: {path}")
    else:
        _set_phases(model, context["state_phases"][state])
        target, _ = baseline._image_batch(context["pixels"], torch.tensor([row]), device)
        model.zero_grad(set_to_none=True)
        result = model.forward_from_image(target)
        loss = _total_loss(result, target, float(context["beta"]),
                           float(context["eta"]), "B")
        gradients = _six_layer_gradients(
            torch.autograd.grad(loss, (model.encoder_phase, model.decoder_phase))
        )
        payload = {"schema": "chen_lightgen_400_fast_rs_total_loss_gradient.v1",
                   "state": state, "row": row, "label": label, "precision": precision,
                   "branch": "B", "loss": float(loss.detach().cpu()),
                   "beta": float(context["beta"]), "eta": float(context["eta"]),
                   "gradients_by_layer": gradients}
        _write_torch(path, payload, new=True)
        del target, result, loss, payload
    if precision == "complex64":
        reference_path = output_dir / "gradients" / f"complex128_{state}_row{row}_branchB.pt"
        _require(reference_path.is_file(),
                 f"complex128 B-loss gradient is missing: {reference_path}")
        high = torch.load(reference_path, map_location="cpu", weights_only=False)
        comparison = q12._compare_gradients(gradients, high["gradients_by_layer"])
        _require(high.get("state") == state and high.get("row") == row
                 and high.get("label") == label
                 and high.get("beta") == float(context["beta"])
                 and high.get("eta") == float(context["eta"])
                 and comparison["max_relative_l2"] <= FIELD_TOLERANCE,
                 f"complex64 B-loss gradient differs from complex128: {path}")
        del high
    return gradients


def _finite_difference_qualification(context: dict[str, Any], output_dir: Path,
                                     device: torch.device, heartbeat: Heartbeat,
                                     precision: str) -> dict[str, Any]:
    phases_by_state = context["state_phases"]
    rows = context["population"]["confirm_rows_by_class"][:2, 0].long()
    model = _make_model(phases_by_state[STATES[0]], device, precision,
                         checkpoint_propagation=False)
    prior._set_trainable(model)
    model.eval()
    original_state = _state_snapshot(model)
    records: list[dict[str, Any]] = []
    old_backend = q12.BACKEND
    q12.BACKEND = "rs_linear_cached"
    try:
        for state_index, state in enumerate(STATES):
            row = int(rows[0])
            b_gradients = _b_loss_gradient(
                context, model, state, row, 0, precision, output_dir, device
            )
            b_norm = torch.sqrt(sum(item.square().sum() for item in b_gradients))
            _require(float(b_norm) > 0.0, f"{precision} B-loss joint gradient is zero")
            b_direction = [item / b_norm for item in b_gradients]
            directions_by_key: dict[tuple[str, int, str], tuple[str, list[torch.Tensor], list[torch.Tensor]]] = {}
            for label, row_value in enumerate(rows):
                current_row = int(row_value)
                gradient_path = output_dir / "gradients" / f"{precision}_{state}_row{current_row}.pt"
                _require(gradient_path.is_file(), f"six-layer MSE gradient is missing: {gradient_path}")
                payload = torch.load(gradient_path, map_location="cpu", weights_only=False)
                _require(payload.get("state") == state and payload.get("row") == current_row
                         and payload.get("label") == label and payload.get("precision") == precision,
                         f"six-layer MSE gradient identity mismatch: {gradient_path}")
                gradients = payload["gradients_by_layer"]
                if state_index == 0 and label == 0:
                    for layer_index, gradient in enumerate(gradients):
                        norm = torch.linalg.vector_norm(gradient)
                        basis = [torch.zeros_like(item) for item in gradients]
                        basis[layer_index] = gradient / norm
                        directions_by_key[("A", current_row, f"layer_{layer_index + 1}")] = (
                            f"layer_{layer_index + 1}", basis, gradients)
                else:
                    norm = torch.sqrt(sum(item.square().sum() for item in gradients))
                    _require(float(norm) > 0.0, f"{precision} MSE joint gradient is zero")
                    directions_by_key[("A", current_row, "joint_six_layer")] = (
                        "joint_six_layer", [item / norm for item in gradients], gradients)
                del payload
            directions_by_key[("B", row, "joint_six_layer")] = (
                "joint_six_layer", b_direction, b_gradients
            )
            for (branch, current_row, _direction_key), (direction_name, direction, gradients) in \
                    directions_by_key.items():
                heartbeat.check()
                label = 0 if current_row == int(rows[0]) else 1
                loss_path = output_dir / "finite_difference" / (
                    f"{precision}_{state}_row{current_row}_{branch}_{direction_name}.json"
                )
                if loss_path.is_file():
                    saved = json.loads(loss_path.read_text(encoding="utf-8"))
                    _require(saved.get("schema") == "chen_lightgen_400_fast_rs_fd_direction.v1"
                             and saved.get("state") == state and saved.get("row") == current_row
                             and saved.get("label") == label and saved.get("direction") == direction_name
                             and saved.get("branch") == branch and saved.get("precision") == precision
                             and saved.get("steps_rad") == list(q12.FD_STEPS)
                             and _finite_difference_record_passes(saved.get("checks", []))
                             == (saved.get("status") == "NUMERICALLY_QUALIFIED"),
                             f"finite-difference resume record identity or values differ: {loss_path}")
                    records.append(saved)
                    continue
                autodiff = sum(float((gradient * direction_item).sum())
                               for gradient, direction_item in zip(gradients, direction, strict=True))
                checks = []
                for epsilon in q12.FD_STEPS:
                    losses: dict[str, float] = {}
                    projections: dict[str, float] = {}
                    for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
                        phases, projection, _delta_l2 = q12._apply_represented_phase_delta(
                            phases_by_state[state], direction, epsilon, sign, device
                        )
                        _set_phases(model, phases)
                        target, _ = baseline._image_batch(
                            context["pixels"], torch.tensor([current_row]), device
                        )
                        with torch.no_grad():
                            result = model.forward_from_image(target)
                            loss = _total_loss(result, target, float(context["beta"]),
                                               float(context["eta"]), branch)
                            losses[sign_name] = float(loss.detach().cpu())
                        projections[sign_name] = projection
                        del phases, target, result, loss
                    denominator = projections["plus"] - projections["minus"]
                    _require(math.isfinite(denominator) and denominator != 0.0,
                             "phase storage erased finite-difference displacement")
                    derivative = (losses["plus"] - losses["minus"]) / denominator
                    degenerate = not math.isfinite(autodiff) or abs(autodiff) <= 1.0e-14
                    relative = (abs(derivative - autodiff)
                                / max(abs(derivative), abs(autodiff), 1.0e-14)
                                if not degenerate else math.nan)
                    checks.append({"nominal_step_rad": epsilon,
                                   "plus_loss": losses["plus"], "minus_loss": losses["minus"],
                                   "plus_directional_displacement_rad": projections["plus"],
                                   "minus_directional_displacement_rad": projections["minus"],
                                   "represented_central_step_rad": denominator,
                                   "central_difference": derivative,
                                   "autodiff_directional_derivative": autodiff,
                                   "degenerate": degenerate,
                                   "same_sign": bool(not degenerate and derivative * autodiff > 0.0),
                                   "relative_error": relative})
                status = ("NUMERICALLY_QUALIFIED" if _finite_difference_record_passes(checks)
                          else "NUMERICAL_NOT_QUALIFIED")
                record = {"schema": "chen_lightgen_400_fast_rs_fd_direction.v1",
                          "state": state, "row": current_row, "label": label,
                          "direction": direction_name, "branch": branch,
                          "loss_definition": "beta*MSE" if branch == "A"
                          else "beta*MSE+eta*(1-SSIM(pool28(prediction),pool28(target)))",
                          "precision": precision, "checks": checks, "steps_rad": list(q12.FD_STEPS),
                          "status": status}
                _write_json(loss_path, record, new=True)
                records.append(record)
                _require_output_budget(output_dir)
                progress = {"stage": f"{precision}_finite_difference",
                            "completed_directions": len(records), "expected_directions": 11,
                            "state": state, "row": current_row, "label": label,
                            "branch": branch, "optimizer_updates": 0, "adam_calls": 0}
                _write_json(output_dir / "progress.json", progress)
                heartbeat.check()
            del b_gradients
        status = ("NUMERICALLY_QUALIFIED" if len(records) == 11 and all(
            item["status"] == "NUMERICALLY_QUALIFIED" for item in records) else
            "INVALID_HARNESS" if len(records) != 11 else "NUMERICAL_NOT_QUALIFIED")
        result = {"status": status, "precision": precision, "records": records,
                  "expected_directions": 11,
                  "executed_perturbed_forwards": sum(len(item["checks"]) * 2 for item in records),
                  "steps_rad": list(q12.FD_STEPS),
                  "relative_error_tolerance": q12.FINITE_DIFFERENCE_TOLERANCE,
                  "model_state_restored": True}
        _write_json(output_dir / f"finite_difference_qualification_{precision}.json", result)
        return result
    finally:
        q12.BACKEND = old_backend
        _set_phases(model, phases_by_state[STATES[0]])
        _require_state_restored(model, original_state)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _paths_from_namespace(args: argparse.Namespace) -> dict[str, str]:
    return {name: str(_resolve(getattr(args, name)).resolve()) for name in (
        "source_checkpoint", "source_report", "source_start_state", "source_endpoint_checkpoint",
        "fixed_population", "config", "data", "evaluator", "qualification_report",
        "reference_run", "structure_coefficient", "gradient_reference_dir", "output_dir")}



def _cost_probe(context: dict[str, Any], output_dir: Path, device: torch.device,
                heartbeat: Heartbeat, structure_coefficient: Path,
                precision: str) -> dict[str, Any]:
    phases = context["state_phases"][STATES[0]]
    rows = context["population"]["fit_rows_by_class"][:, 0].long()
    eta = _load_eta(structure_coefficient)
    beta = float(context["beta"])
    _require(precision in ("complex128", "complex64"), "cost probe precision is invalid")
    model = _make_model(phases, device, precision, checkpoint_propagation=True)
    prior._set_trainable(model)
    original_state = _state_snapshot(model)
    records: dict[str, Any] = {}
    for branch in ("A", "B"):
        for microbatch in (1, 2):
            heartbeat.check()
            key = f"{branch}_microbatch_{microbatch}"
            payload_path = output_dir / "gradients" / f"cost_{precision}_{key}.pt"
            metric_path = output_dir / "cost" / f"cost_{precision}_{key}.json"
            if payload_path.is_file() and metric_path.is_file():
                metric = json.loads(metric_path.read_text(encoding="utf-8"))
                payload = torch.load(payload_path, map_location="cpu", weights_only=False)
                _require(metric.get("branch") == branch and metric.get("microbatch") == microbatch
                         and metric.get("precision") == precision
                         and metric.get("rows") == rows.cpu().tolist()
                         and metric.get("state") == STATES[0]
                         and payload.get("branch") == branch and payload.get("microbatch") == microbatch
                         and payload.get("precision") == precision
                         and isinstance(payload.get("rows"), torch.Tensor)
                         and torch.equal(payload["rows"], rows.cpu()),
                         f"saved batch cost metadata differs: {metric_path}")
                gradients = payload.get("gradients")
                _require(isinstance(gradients, dict) and set(gradients) == {"encoder_phase", "decoder_phase"}
                         and all(torch.isfinite(value).all() for value in gradients.values()),
                         f"saved batch cost gradient is invalid: {payload_path}")
                _require(math.isfinite(float(metric.get("loss", math.nan)))
                         and math.isfinite(float(metric.get("elapsed_seconds", math.nan)))
                         and float(metric["elapsed_seconds"]) > 0.0
                         and metric.get("gradient_payload") == str(payload_path)
                         and math.isclose(float(payload.get("loss", math.nan)), float(metric["loss"]),
                                          rel_tol=1.0e-12, abs_tol=1.0e-14),
                         f"saved batch cost values are incomplete: {metric_path}")
                records[key] = metric
                del payload, gradients
                continue
            if payload_path.is_file():
                payload = torch.load(payload_path, map_location="cpu", weights_only=False)
                metric = payload.get("metric")
                _require(isinstance(metric, dict) and metric.get("branch") == branch
                         and metric.get("microbatch") == microbatch
                         and metric.get("precision") == precision
                         and metric.get("rows") == rows.cpu().tolist()
                         and metric.get("state") == STATES[0]
                         and payload.get("precision") == precision
                         and payload.get("state") == STATES[0]
                         and isinstance(payload.get("rows"), torch.Tensor)
                         and torch.equal(payload["rows"], rows.cpu())
                         and math.isfinite(float(metric.get("loss", math.nan)))
                         and math.isfinite(float(metric.get("elapsed_seconds", math.nan)))
                         and float(metric["elapsed_seconds"]) > 0.0,
                         f"saved gradient lacks its exact measurement metadata: {payload_path}")
                _write_json(metric_path, metric, new=True)
                records[key] = metric
                del payload
                continue
            _require(not metric_path.exists(),
                     f"cost metadata exists without its measured gradient payload: {metric_path}")
            cached_before = model.propagator.operator.cache_info.get("cached_geometry") is not None
            loss, gradients, elapsed, peak = q12._run_microbatch(
                model, context["pixels"], rows, beta, eta, branch, microbatch, device
            )
            _sync(device)
            cache_info = model.propagator.operator.cache_info
            metric = {
                "branch": branch, "microbatch": microbatch, "loss": loss,
                "precision": precision, "state": STATES[0], "rows": rows.cpu().tolist(),
                "gradient_l2_by_group": {name: float(torch.linalg.vector_norm(value.double()))
                                          for name, value in gradients.items()},
                "elapsed_seconds": elapsed,
                "peak_reserved_vram_bytes": peak,
                "logical_batch": LOGICAL_BATCH,
                "optimizer_calls": 0,
                "optimizer_updates": 0,
                "adam_calls": 0,
                "checkpoint_propagation": True,
                "kernel_cache_was_warm_before_call": cached_before,
                "kernel_cache": cache_info,
                "gradient_payload": str(payload_path),
            }
            _write_torch(payload_path, {"branch": branch, "microbatch": microbatch,
                                        "precision": precision, "state": STATES[0],
                                        "rows": rows.cpu(), "loss": loss, "gradients": gradients,
                                        "metric": metric}, new=True)
            _write_json(metric_path, metric, new=True)
            _require_output_budget(output_dir)
            records[key] = metric
            del gradients
            heartbeat.check()
    comparisons: dict[str, Any] = {}
    for branch in ("A", "B"):
        m1, m2 = records[f"{branch}_microbatch_1"], records[f"{branch}_microbatch_2"]
        p1 = torch.load(m1["gradient_payload"], map_location="cpu", weights_only=False)
        p2 = torch.load(m2["gradient_payload"], map_location="cpu", weights_only=False)
        errors = {
            name: float(torch.linalg.vector_norm(p1["gradients"][name].double()
                                                 - p2["gradients"][name].double())
                        / torch.linalg.vector_norm(p1["gradients"][name].double()).clamp_min(1.0e-30))
            for name in p1["gradients"]
        }
        loss_error = abs(m1["loss"] - m2["loss"]) / max(abs(m1["loss"]), 1.0e-30)
        selected = min((1, 2), key=lambda mb: records[f"{branch}_microbatch_{mb}"]["elapsed_seconds"])
        comparisons[branch] = {
            "loss_relative_error": loss_error,
            "gradient_relative_l2_by_group": errors,
            "selected_microbatch_by_observed_time": selected,
            "status": "NUMERICALLY_QUALIFIED" if loss_error <= 1.0e-5 and max(errors.values()) <= 1.0e-4
                      else "NUMERICAL_NOT_QUALIFIED",
        }
    _require_state_restored(model, original_state)
    result = {"status": "NUMERICALLY_QUALIFIED" if all(
        item["status"] == "NUMERICALLY_QUALIFIED" for item in comparisons.values())
        else "NUMERICAL_NOT_QUALIFIED", "records": records, "comparisons": comparisons,
        "precision": precision, "source_beta": beta, "fixed_eta": eta,
        "logical_batch": LOGICAL_BATCH, "gradient_recomputed": False,
        "optimizer_calls": 0, "optimizer_updates": 0, "adam_calls": 0}
    _write_json(output_dir / f"batch10_cost_{precision}.json", result)
    return result



def _qualification_input_snapshot(args: argparse.Namespace,
                                  context: dict[str, Any]) -> dict[str, Any]:
    eta = _load_eta(args.structure_coefficient)
    population = context["population"]
    metadata = {
        "task_id": QUALIFICATION_TASK_ID,
        "explicit_inputs": _paths_from_namespace(args),
        "target_hours_estimate_reference_only": args.target_hours,
        "source_checkpoint_step": context["checkpoint"].get("completed_updates"),
        "historical_endpoint_step": context["endpoint_payload"].get("completed_updates"),
        "configuration": _safe_json(context["config"]),
        "beta": float(context["beta"]), "eta": eta,
        "fit_rows_by_class": population["fit_rows_by_class"].cpu().tolist(),
        "confirmation_rows_by_class": population["confirm_rows_by_class"].cpu().tolist(),
        "fit_labels": context["labels"].detach().cpu()[population["fit_rows"].long().cpu()].tolist(),
        "confirmation_labels": context["labels"].detach().cpu()[
            population["confirm_rows"].long().cpu()].tolist(),
        "data_roles": {"fit": "canonical_MNIST_TRAIN_fit100",
                       "confirmation": "canonical_MNIST_TRAIN_confirmation200"},
        "optimizer_updates": 0, "adam_calls": 0,
    }
    phases = {state: {name: value.detach().cpu().clone() for name, value in values.items()}
              for state, values in context["state_phases"].items()}
    return {"metadata": metadata, "state_phases": phases}


def _verify_qualification_input_snapshot(saved: dict[str, Any],
                                         current: dict[str, Any]) -> None:
    _require(saved.get("metadata") == current["metadata"],
             "resume model/configuration, steps, input paths, population, or data roles differ")
    saved_phases = saved.get("state_phases")
    _require(isinstance(saved_phases, dict) and set(saved_phases) == set(STATES),
             "resume state snapshot is incomplete")
    for state in STATES:
        _require(set(saved_phases[state]) == set(current["state_phases"][state])
                 and all(torch.equal(saved_phases[state][name], current["state_phases"][state][name])
                         for name in current["state_phases"][state]),
                 f"resume model phases differ from saved input state: {state}")


def _write_qualification_closeout(output_dir: Path, report: dict[str, Any],
                                   stages: dict[str, Any]) -> dict[str, int]:
    value_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    fd_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    precision_rows: list[dict[str, Any]] = []
    for precision in ("complex128", "complex64"):
        stage = stages.get(precision, {})
        full_chain = stage.get("full_chain")
        if isinstance(full_chain, dict):
            for item in full_chain.get("records", []):
                value_rows.append({
                    "precision": precision, "state": item.get("state"), "row": item.get("row"),
                    "label": item.get("label"),
                    "max_relative_l2_vs_q16": item.get("max_value_relative_l2_vs_q16",
                                                        item.get("max_q12_q16_value_relative_l2")),
                    "max_relative_l2_vs_historical_q12": item.get(
                        "max_value_relative_l2_vs_historical_q12",
                        item.get("max_cached_vs_historical_q12_relative_l2")),
                    "max_final_readout_relative_l2_vs_complex128": item.get(
                        "max_final_readout_relative_l2_vs_complex128"),
                    "elapsed_seconds": item.get("elapsed_seconds"),
                    "status": item.get("status"), "compact_path": item.get("compact_path"),
                })
        gradient = stage.get("gradients")
        if isinstance(gradient, dict):
            compared = {(item["state"], item["row"]): item
                        for item in gradient.get("comparison_to_complex128", [])}
            for item in gradient.get("records", []):
                payload = torch.load(item["gradient_payload"], map_location="cpu", weights_only=False)
                compare = compared.get((item.get("state"), item.get("row")), {})
                for layer, values in enumerate(payload["gradients_by_layer"], start=1):
                    gradient_rows.append({
                        "precision": precision, "branch": "A",
                        "state": item.get("state"), "row": item.get("row"),
                        "label": item.get("label"), "layer": layer,
                        "gradient_l2": float(torch.linalg.vector_norm(values.to(torch.float64))),
                        "relative_l2_vs_historical_q12": item.get(
                            "relative_l2_vs_historical_q12_by_layer", [None] * 6)[layer - 1],
                        "relative_l2_vs_historical_q16": item.get(
                            "relative_l2_vs_historical_q16_by_layer", [None] * 6)[layer - 1],
                        "relative_l2_vs_complex128": compare.get(
                            "per_layer_relative_l2", [None] * 6)[layer - 1],
                        "status": item.get("status"),
                    })
                del payload
            for item in gradient.get("records", []):
                if item.get("label") != 0:
                    continue
                path = output_dir / "gradients" / (
                    f"{precision}_{item['state']}_row{item['row']}_branchB.pt"
                )
                if not path.is_file():
                    continue
                payload = torch.load(path, map_location="cpu", weights_only=False)
                high_path = output_dir / "gradients" / (
                    f"complex128_{item['state']}_row{item['row']}_branchB.pt"
                )
                compare = {}
                if precision == "complex64" and high_path.is_file():
                    high = torch.load(high_path, map_location="cpu", weights_only=False)
                    compare = q12._compare_gradients(
                        payload["gradients_by_layer"], high["gradients_by_layer"]
                    )
                    del high
                for layer, values in enumerate(payload["gradients_by_layer"], start=1):
                    gradient_rows.append({
                        "precision": precision, "branch": "B", "state": item.get("state"),
                        "row": item.get("row"), "label": 0, "layer": layer,
                        "gradient_l2": float(torch.linalg.vector_norm(values.to(torch.float64))),
                        "relative_l2_vs_historical_q12": None,
                        "relative_l2_vs_historical_q16": None,
                        "relative_l2_vs_complex128": compare.get(
                            "per_layer_relative_l2", [None] * 6)[layer - 1],
                        "status": "NUMERICALLY_QUALIFIED" if precision == "complex128"
                        or compare.get("max_relative_l2", math.inf) <= FIELD_TOLERANCE
                        else "NUMERICAL_NOT_QUALIFIED",
                    })
                del payload
        fd = stage.get("finite_difference")
        if isinstance(fd, dict):
            for record in fd.get("records", []):
                for check in record.get("checks", []):
                    fd_rows.append({
                        "precision": precision, "state": record.get("state"), "row": record.get("row"),
                        "label": record.get("label"), "branch": record.get("branch"),
                        "direction": record.get("direction"),
                        "nominal_step_rad": check.get("nominal_step_rad"),
                        "plus_loss": check.get("plus_loss"), "minus_loss": check.get("minus_loss"),
                        "represented_central_step_rad": check.get("represented_central_step_rad"),
                        "central_difference": check.get("central_difference"),
                        "autodiff_directional_derivative": check.get("autodiff_directional_derivative"),
                        "same_sign": check.get("same_sign"), "relative_error": check.get("relative_error"),
                        "status": record.get("status"),
                    })
        cost = stage.get("batch10_cost")
        if isinstance(cost, dict):
            for key, item in cost.get("records", {}).items():
                cost_rows.append({
                    "precision": precision, "case": key, "branch": item.get("branch"),
                    "microbatch": item.get("microbatch"), "loss": item.get("loss"),
                    "elapsed_seconds": item.get("elapsed_seconds"),
                    "peak_reserved_vram_bytes": item.get("peak_reserved_vram_bytes"),
                    "kernel_cache_was_warm_before_call": item.get("kernel_cache_was_warm_before_call"),
                    "adam_calls": item.get("adam_calls"),
                })
        selected = {}
        if isinstance(cost, dict):
            for branch, comparison in cost.get("comparisons", {}).items():
                microbatch = comparison.get("selected_microbatch_by_observed_time")
                if microbatch in (1, 2):
                    seconds = cost["records"][f"{branch}_microbatch_{microbatch}"]["elapsed_seconds"]
                    selected[branch] = {
                        "measured_logical_batch_seconds": seconds,
                        "500_updates_seconds_compute_only": 500 * seconds,
                        "30000_updates_seconds_compute_only": 30_000 * seconds,
                        "500_updates_days_compute_only": 500 * seconds / 86_400,
                        "30000_updates_days_compute_only": 30_000 * seconds / 86_400,
                        "checkpoint_evaluation_and_restart_overhead_included": False,
                    }
        stage["compute_only_training_estimates"] = selected
        precision_rows.append({
            "precision": precision, "status": stage.get("status", "NOT_RUN"),
            "single_hop": stage.get("single_hop_status", "NOT_RUN"),
            "full_chain": stage.get("full_chain", {}).get("status", "NOT_RUN"),
            "gradients": stage.get("gradients", {}).get("status", "NOT_RUN"),
            "finite_difference": stage.get("finite_difference", {}).get("status", "NOT_RUN"),
            "batch10_cost": stage.get("batch10_cost", {}).get("status", "NOT_RUN"),
            "compute_only_training_estimates": selected,
        })

    tables = {
        "precision_summary.csv": (precision_rows,
            ["precision", "status", "single_hop", "full_chain", "gradients",
             "finite_difference", "batch10_cost", "compute_only_training_estimates"]),
        "per_sample_errors.csv": (value_rows,
            ["precision", "state", "row", "label", "max_relative_l2_vs_q16",
             "max_relative_l2_vs_historical_q12",
             "max_final_readout_relative_l2_vs_complex128", "elapsed_seconds", "status", "compact_path"]),
        "per_layer_gradients.csv": (gradient_rows,
            ["precision", "branch", "state", "row", "label", "layer", "gradient_l2",
             "relative_l2_vs_historical_q12", "relative_l2_vs_historical_q16",
             "relative_l2_vs_complex128", "status"]),
        "finite_difference.csv": (fd_rows,
            ["precision", "state", "row", "label", "branch", "direction", "nominal_step_rad",
             "plus_loss", "minus_loss", "represented_central_step_rad", "central_difference",
             "autodiff_directional_derivative", "same_sign", "relative_error", "status"]),
        "batch10_cost.csv": (cost_rows,
            ["precision", "case", "branch", "microbatch", "loss", "elapsed_seconds",
             "peak_reserved_vram_bytes", "kernel_cache_was_warm_before_call", "adam_calls"]),
    }
    counts = {}
    for filename, (rows, fields) in tables.items():
        with (output_dir / filename).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        counts[filename.removesuffix(".csv")] = len(rows)
        with (output_dir / filename).open("r", encoding="utf-8-sig", newline="") as handle:
            readback_count = sum(1 for _ in csv.DictReader(handle))
        _require(readback_count == len(rows), f"CSV write/readback row count differs: {filename}")
    report["precision_results"] = precision_rows
    report["artifact_counts"] = counts
    html_rows = "".join("<tr>" + "".join(
        f"<td>{html.escape(str(row[key]))}</td>" for key in (
            "precision", "status", "single_hop", "full_chain", "gradients",
            "finite_difference", "batch10_cost")) + "</tr>" for row in precision_rows)
    page = ("<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>LightGen400 RS资格</title>"
            "<style>body{font:15px system-ui;max-width:1100px;margin:36px auto;padding:0 18px}"
            "table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:8px;text-align:left}"
            "th{background:#f1f3f5}.note{background:#f8f8f8;padding:12px}</style><body>"
            "<h1>LightGen400快速RS零更新资格</h1>"
            f"<p>状态：{html.escape(str(report['status']))}</p>"
            "<p class='note'>光学前后向和有限差分已测；Adam更新、E_D前向及生成采样均为0。</p>"
            "<table><thead><tr><th>精度</th><th>状态</th><th>单跳</th><th>整链</th>"
            "<th>六层梯度</th><th>有限差分</th><th>batch10成本</th></tr></thead>"
            f"<tbody>{html_rows}</tbody></table><p><a href='final_report.json'>机器报告</a> · "
            "<a href='precision_summary.csv'>精度汇总</a> · "
            "<a href='per_sample_errors.csv'>逐样本误差</a> · "
            "<a href='per_layer_gradients.csv'>逐层梯度</a> · "
            "<a href='finite_difference.csv'>有限差分</a> · "
            "<a href='batch10_cost.csv'>batch10成本</a></p></body></html>")
    (output_dir / "comparison.html").write_text(page, encoding="utf-8")
    return counts


def _finish_qualification(output_dir: Path, state_path: Path,
                          input_snapshot: dict[str, Any], progress: dict[str, Any],
                          stages: dict[str, Any], operator: dict[str, Any] | None,
                          reference_validation: dict[str, Any] | None,
                          direct_reference: dict[str, Any] | None,
                          target_hours: float, started: float,
                          device: torch.device) -> dict[str, Any]:
    c128 = stages.get("complex128", {})
    c64 = stages.get("complex64", {})
    c128_pass = bool(operator and operator.get("c128_one_hop_pass")
                     and all(_qualified(c128.get(key, {}).get("status", ""))
                             for key in ("batch10_cost", "full_chain", "gradients", "finite_difference")))
    c64_ran = any(key in c64 for key in ("full_chain", "gradients", "batch10_cost", "finite_difference"))
    c64_pass = bool(c128_pass and operator and operator.get("c64_one_hop_pass")
                    and all(_qualified(c64.get(key, {}).get("status", ""))
                            for key in ("full_chain", "gradients", "batch10_cost", "finite_difference")))
    direct_pass = bool(direct_reference
                       and direct_reference.get("status") == "NUMERICALLY_QUALIFIED")
    if not direct_pass:
        status = "VALID_COMPLETED_INDEPENDENT_RS_REFERENCE_NOT_QUALIFIED"
    elif c128_pass and c64_pass:
        status = "VALID_COMPLETED_BOTH_PRECISIONS_QUALIFIED"
    elif c128_pass and c64_ran:
        status = "VALID_COMPLETED_COMPLEX128_ONLY"
    else:
        status = "VALID_COMPLETED_COMPLEX128_NOT_QUALIFIED_C64_SKIPPED"
    report: dict[str, Any] = {
        "status": status, "task_id": QUALIFICATION_TASK_ID,
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer_updates": 0, "optimizer_calls": 0, "adam_calls": 0,
        "generation_samples": 0, "evaluator_forward_calls": 0,
        "input_validation": reference_validation,
        "independent_rs_reference": direct_reference,
        "operator_benchmark": operator, "precision_stages": stages,
        "complex128_qualified": c128_pass, "complex64_qualified": c64_pass,
        "complex64_status": c64.get("status", "SKIPPED_AFTER_COMPLEX128_NOT_QUALIFIED"),
        "target_hours_estimate_reference_only": target_hours,
        "time_budget_stop_applied": False,
        "resources_at_end": _device_info(device),
        "input_state_restored": True,
        "optimizer_or_recovery_path_called": False,
        "scientific_effect": "none; no model updates or generation samples",
    }
    counts = _write_qualification_closeout(output_dir, report, stages)
    receipt = {
        "task_id": QUALIFICATION_TASK_ID, "status": report["status"],
        "expected_counts": {
            "raw_references": 40, "gradient_references": 4,
            "full_chain_per_precision": 20, "gradient_samples_per_precision": 4,
            "finite_difference_directions_per_precision": 11,
            "finite_difference_perturbed_forwards_per_precision": 44,
            "batch10_cost_cases_per_precision": 4,
        },
        "input_reference_validation": reference_validation,
        "direct_rs_reference_status": None if direct_reference is None else direct_reference.get("status"),
        "one_hop_status": None if operator is None else {
            "complex128": operator.get("c128_one_hop_pass"),
            "complex64": operator.get("c64_one_hop_pass"),
        },
        "precision_statuses": {precision: item.get("status", "NOT_RUN")
                                for precision, item in stages.items()},
        "optimizer_updates": 0, "optimizer_calls": 0, "adam_calls": 0,
        "generation_samples": 0, "evaluator_forward_calls": 0,
        "input_state_restored": True, "csv_rows": counts,
    }
    _write_json(output_dir / "validation_receipt.json", receipt)
    _write_json(output_dir / "final_report.json", report)
    _write_json(output_dir / "progress.json", {
        "task_id": QUALIFICATION_TASK_ID, "stage": "complete", "status": status,
        "elapsed_seconds": report["elapsed_seconds"],
        "completed_stages": progress.get("completed_stages", []),
        "optimizer_updates": 0, "adam_calls": 0,
    })
    _write_torch(state_path, {
        "schema": "chen_lightgen_400_fast_rs_qualification_state.v1",
        "task_id": QUALIFICATION_TASK_ID, "input_metadata": input_snapshot["metadata"],
        "input_state_phases": input_snapshot["state_phases"],
        "completed_stages": progress.get("completed_stages", []),
        "stage": "complete", "status": status,
    })
    _require_output_budget(output_dir)
    report["output_bytes"] = _tree_bytes(output_dir)
    _write_json(output_dir / "final_report.json", report)
    closeout = (
        "# LightGen400 快速 RS 零更新资格收尾\n\n"
        f"结果与状态：`{status}`。complex128 资格：{c128_pass}；complex64 资格：{c64_pass}。"
        "本结果是工程数值资格，不改变既有大架构基线的科学判定。\n\n"
        "科学认识：数值资格只覆盖固定30000步起点与历史500步端点、fit100/confirmation200、"
        "原损失及固定读出；不支持完整生成、架构容量或真实硬件结论。\n\n"
        "目标缺口：工程更新／恢复、效果短训、生成采样及模型采用尚未验证。原基线 `VALID_FAILED`、"
        "九通道类别2/8功率约束和身份CPU重放缺口保留。\n\n"
        "工程完成与验证：保存了单跳、整链、六层梯度、有限差分、batch10成本、逐样本/逐层CSV、"
        "独立参考、验证收据及统一比较页；更新、Adam、E_D前向和生成采样均为0。\n\n"
        "决策影响与下一步：资格通过时只支持另行设计固定预算工程更新／恢复验收；资格未通过时保留"
        "对应精度缺口，不安排效果短训。上述是下一步建议，不是本轮授权。\n"
    )
    (output_dir / "closeout_report.md").write_text(closeout, encoding="utf-8")
    report["elapsed_seconds"] = time.perf_counter() - started
    _require_output_budget(output_dir)
    report["output_bytes"] = _tree_bytes(output_dir)
    _write_json(output_dir / "final_report.json", report)
    return report


def _qualified(status: str) -> bool:
    return status == "NUMERICALLY_QUALIFIED"


def _qualification_preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve(args.output_dir)
    state_path = output_dir / "preflight_state.pt"
    if args.mode == "preflight":
        _require(not output_dir.exists(), f"refusing to overwrite existing run directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_torch(state_path, {
            "schema": "chen_lightgen_400_fast_rs_qualification_state.v1",
            "task_id": QUALIFICATION_TASK_ID,
            "explicit_inputs": _paths_from_namespace(args),
            "completed_stages": [], "stage": "starting"}, new=True)
    else:
        _require(output_dir.is_dir(), "explicit resume output directory does not exist")
        _require(args.resume_checkpoint.resolve() == state_path.resolve(),
                 "resume must use this run's explicit qualification state")
        _require(state_path.is_file() and not (output_dir / "final_report.json").exists(),
                 "resume state is missing or qualification is already complete")
        saved_state = torch.load(state_path, map_location="cpu", weights_only=False)
        _require(saved_state.get("schema") == "chen_lightgen_400_fast_rs_qualification_state.v1"
                 and saved_state.get("task_id") == QUALIFICATION_TASK_ID
                 and saved_state.get("explicit_inputs") == _paths_from_namespace(args),
                 "resume state does not match this explicit task and input paths")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = torch.device("cuda")
    explicit_inputs = _paths_from_namespace(args)
    progress: dict[str, Any] = {
        "task_id": QUALIFICATION_TASK_ID, "stage": "input_load",
        "completed_stages": [], "optimizer_updates": 0, "adam_calls": 0,
    }
    if not torch.cuda.is_available():
        failure = {"status": "PREFLIGHT_BLOCKED_RESOURCE", "task_id": QUALIFICATION_TASK_ID,
                   "stage": "input_load", "reason": "CUDA unavailable",
                   "optimizer_updates": 0, "adam_calls": 0}
        _write_json(output_dir / "failure_record.json", failure)
        return failure
    started_device = _device_info(device)
    with Heartbeat(output_dir, progress) as heartbeat:
        try:
            file_inputs = (
                "source_checkpoint", "source_report", "source_start_state",
                "source_endpoint_checkpoint", "fixed_population", "config", "data",
                "evaluator", "qualification_report", "structure_coefficient",
            )
            for name in file_inputs:
                path = _resolve(getattr(args, name))
                _require(path.is_file(), f"explicit input file is missing: --{name.replace('_', '-')}: {path}")
            for name in ("reference_run", "gradient_reference_dir"):
                path = _resolve(getattr(args, name))
                _require(path.is_dir(), f"explicit input directory is missing: --{name.replace('_', '-')}: {path}")
            context = _load_context(args, device)
            _require(context["checkpoint"].get("completed_updates") == 30_000
                     and context["endpoint_payload"].get("completed_updates") == 500,
                     "explicit model states are not the fixed 30000/500 step pair")
            _require(context["population"]["fit_rows"].numel() == 100
                     and context["population"]["confirm_rows"].numel() == 200,
                     "fixed population differs from fit100/confirmation200")
            input_snapshot = _qualification_input_snapshot(args, context)
            context["eta"] = input_snapshot["metadata"]["eta"]
            snapshot_path = output_dir / "input_identity.pt"
            if args.mode == "preflight":
                _write_torch(snapshot_path, input_snapshot, new=True)
            else:
                _require(snapshot_path.is_file(), "resume input identity snapshot is missing")
                saved_snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
                _verify_qualification_input_snapshot(saved_snapshot, input_snapshot)

            reference_validation = _validate_reference_bundle(args, context)
            _write_json(output_dir / "reference_validation.json", reference_validation)
            progress["completed_stages"].append("input_and_reference_validation")
            progress.update({"stage": "independent_rs_reference", "resources_at_start": started_device})
            _write_json(output_dir / "progress.json", progress.copy())
            free_bytes = shutil.disk_usage(output_dir).free
            _require(free_bytes >= OUTPUT_LIMIT_BYTES,
                     f"output filesystem has less than the 512 MiB result budget free: {free_bytes}")
            _write_json(output_dir / "resource_preflight.json", {
                "device": started_device, "output_free_bytes": free_bytes,
                "output_limit_bytes": OUTPUT_LIMIT_BYTES,
                "reference_raw_bytes": sum(path.stat().st_size for path in
                    (_resolve(args.reference_run) / "qualification_raw").glob("*.pt")),
                "gradient_reference_bytes": sum(path.stat().st_size for path in
                    _resolve(args.gradient_reference_dir).glob("gradients_*_q12_q16.pt")),
                "status": "PASSED",
            })

            direct_path = output_dir / "independent_rs_reference.json"
            if direct_path.is_file():
                direct_reference = json.loads(direct_path.read_text(encoding="utf-8"))
                _require(direct_reference.get("status") == "NUMERICALLY_QUALIFIED"
                         and direct_reference.get("reference_status") == "NUMERICALLY_QUALIFIED"
                         and direct_reference.get("production_status") == "NUMERICALLY_QUALIFIED",
                         "saved independent RS reference is incomplete or unqualified")
            else:
                direct_reference = q12._direct_rs_qualification(device, output_dir, progress)
                _write_json(direct_path, direct_reference, new=True)
            progress["completed_stages"].append("independent_rs_reference")
            heartbeat.check()
            stages: dict[str, Any] = {"complex128": {}, "complex64": {}}
            operator = None
            if direct_reference.get("status") != "NUMERICALLY_QUALIFIED":
                return _finish_qualification(
                    output_dir, state_path, input_snapshot, progress, stages, operator,
                    reference_validation, direct_reference, args.target_hours, started, device,
                )

            progress["stage"] = "actual_400_pixel_single_hop"
            _write_json(output_dir / "progress.json", progress.copy())
            operator_path = output_dir / "operator_benchmark.json"
            if operator_path.is_file():
                operator = json.loads(operator_path.read_text(encoding="utf-8"))
                _require(operator.get("row") == int(context["population"]["confirm_rows_by_class"][0, 0])
                         and operator.get("label") == 0
                         and set(operator.get("records", {})) == {
                             "old_uncached_complex128", "cached_complex128", "cached_complex64"}
                         and math.isfinite(float(operator["records"]["cached_complex128"][
                             "node_field_relative_l2_vs_uncached"]))
                         and math.isfinite(float(operator["records"]["cached_complex64"][
                             "node_field_relative_l2_vs_uncached"])),
                         "saved one-hop measurement is incomplete or has the wrong sample")
            else:
                operator = _operator_probe(context, output_dir, device)
            operator["remaining_work_estimate"] = _estimate_remaining_work(operator)
            operator["target_hours_estimate_reference_only"] = args.target_hours
            operator["time_budget_stop_applied"] = False
            _write_json(operator_path, operator)
            progress["completed_stages"].append("actual_400_pixel_single_hop")
            heartbeat.check()

            if operator.get("c128_one_hop_pass"):
                progress.update({"stage": "complex128_batch10_cost", "precision": "complex128"})
                _write_json(output_dir / "progress.json", progress.copy())
                stages["complex128"]["batch10_cost"] = _cost_probe(
                    context, output_dir, device, heartbeat, args.structure_coefficient, "complex128"
                )
                progress["completed_stages"].append("complex128_batch10_cost")
                progress.update({"stage": "complex128_full_chain", "precision": "complex128"})
                _write_json(output_dir / "progress.json", progress.copy())
                stages["complex128"]["full_chain"] = _measure_compact_full_chain(
                    context, args, output_dir, device, heartbeat, progress
                )
                _write_json(output_dir / "full_chain_compact.json", stages["complex128"]["full_chain"])
                progress["completed_stages"].append("complex128_full_chain")
                progress.update({"stage": "complex128_six_layer_gradients"})
                _write_json(output_dir / "progress.json", progress.copy())
                stages["complex128"]["gradients"] = _gradient_qualification(
                    context, args.gradient_reference_dir, output_dir, device, heartbeat, "complex128"
                )
                progress["completed_stages"].append("complex128_six_layer_gradients")
                progress.update({"stage": "complex128_finite_difference"})
                _write_json(output_dir / "progress.json", progress.copy())
                stages["complex128"]["finite_difference"] = _finite_difference_qualification(
                    context, output_dir, device, heartbeat, "complex128"
                )
                progress["completed_stages"].append("complex128_finite_difference")
            else:
                stages["complex128"]["batch10_cost"] = {"status": "SKIPPED_AFTER_ONE_HOP_FAILURE"}
                stages["complex128"]["full_chain"] = {"status": "SKIPPED_AFTER_ONE_HOP_FAILURE"}
                stages["complex128"]["gradients"] = {"status": "SKIPPED_AFTER_ONE_HOP_FAILURE"}
                stages["complex128"]["finite_difference"] = {"status": "SKIPPED_AFTER_ONE_HOP_FAILURE"}
            stages["complex128"]["single_hop_status"] = (
                "NUMERICALLY_QUALIFIED" if operator.get("c128_one_hop_pass")
                else "NUMERICAL_NOT_QUALIFIED")
            c128_pass = bool(operator.get("c128_one_hop_pass")
                             and all(_qualified(stages["complex128"].get(name, {}).get("status", ""))
                                     for name in ("batch10_cost", "full_chain", "gradients",
                                                  "finite_difference")))
            stages["complex128"]["status"] = (
                "NUMERICALLY_QUALIFIED" if c128_pass else "NUMERICAL_NOT_QUALIFIED")
            progress.update({"stage": "complex128_completed", "status": stages["complex128"]["status"]})
            _write_json(output_dir / "progress.json", progress.copy())
            heartbeat.check()

            if c128_pass:
                progress.update({"stage": "complex64_full_chain", "precision": "complex64"})
                _write_json(output_dir / "progress.json", progress.copy())
                stages["complex64"]["full_chain"] = _measure_complex64_full_chain(
                    context, args, output_dir, device, heartbeat, progress
                )
                _write_json(output_dir / "complex64_full_chain.json", stages["complex64"]["full_chain"])
                progress["completed_stages"].append("complex64_full_chain")
                progress.update({"stage": "complex64_six_layer_gradients"})
                _write_json(output_dir / "progress.json", progress.copy())
                stages["complex64"]["gradients"] = _gradient_qualification(
                    context, args.gradient_reference_dir, output_dir, device, heartbeat, "complex64"
                )
                progress["completed_stages"].append("complex64_six_layer_gradients")
                stages["complex64"]["single_hop_status"] = (
                    "NUMERICALLY_QUALIFIED" if operator.get("c64_one_hop_pass")
                    else "NUMERICAL_NOT_QUALIFIED")
                c64_screen_pass = bool(operator.get("c64_one_hop_pass")
                                       and _qualified(stages["complex64"]["full_chain"]["status"])
                                       and _qualified(stages["complex64"]["gradients"]["status"]))
                if c64_screen_pass:
                    progress.update({"stage": "complex64_batch10_cost", "precision": "complex64"})
                    _write_json(output_dir / "progress.json", progress.copy())
                    stages["complex64"]["batch10_cost"] = _cost_probe(
                        context, output_dir, device, heartbeat, args.structure_coefficient, "complex64"
                    )
                    progress["completed_stages"].append("complex64_batch10_cost")
                    progress.update({"stage": "complex64_finite_difference"})
                    _write_json(output_dir / "progress.json", progress.copy())
                    stages["complex64"]["finite_difference"] = _finite_difference_qualification(
                        context, output_dir, device, heartbeat, "complex64"
                    )
                    progress["completed_stages"].append("complex64_finite_difference")
                    c64_pass = bool(all(_qualified(stages["complex64"].get(name, {}).get("status", ""))
                                        for name in ("batch10_cost", "finite_difference")))
                    stages["complex64"]["status"] = (
                        "NUMERICALLY_QUALIFIED" if c64_pass else "NUMERICAL_NOT_QUALIFIED")
                else:
                    stages["complex64"]["batch10_cost"] = {"status": "SKIPPED_AFTER_VALUE_OR_GRADIENT_SCREEN"}
                    stages["complex64"]["finite_difference"] = {"status": "SKIPPED_AFTER_VALUE_OR_GRADIENT_SCREEN"}
                    stages["complex64"]["status"] = "NUMERICAL_NOT_QUALIFIED"
            else:
                stages["complex64"]["status"] = "SKIPPED_AFTER_COMPLEX128_NOT_QUALIFIED"

            _verify_qualification_input_snapshot(input_snapshot, _qualification_input_snapshot(args, context))
            progress["completed_stages"].append("model_state_and_input_restoration")
            _write_json(output_dir / "run_manifest.json", {
                "schema": "chen_lightgen_400_fast_rs_qualification_manifest.v2",
                "task_id": QUALIFICATION_TASK_ID, "explicit_inputs": explicit_inputs,
                "states": list(STATES), "candidate_order": ORDER,
                "reference_order": REFERENCE_ORDER, "window_cells": WINDOW,
                "pixel_pitch_m": PIXEL_PITCH_M,
                "data_role": "fixed_fit100_confirmation200_from_canonical_TRAIN",
                "precision_sequence": ["complex128", "complex64"],
                "science_optimizer_updates": 0, "engineering_adam_calls": 0,
                "generation_samples": 0, "evaluator_forward_calls": 0,
                "beta": context["beta"], "eta": input_snapshot["metadata"]["eta"],
                "target_hours_estimate_reference_only": args.target_hours,
                "output_limit_bytes": OUTPUT_LIMIT_BYTES,
            }, new=args.mode == "preflight")
            _require_output_budget(output_dir)
            heartbeat.check()
            return _finish_qualification(
                output_dir, state_path, input_snapshot, progress, stages, operator,
                reference_validation, direct_reference, args.target_hours, started, device,
            )
        except NumericalQualificationError as exc:
            failure = {"status": "NUMERICAL_NOT_QUALIFIED", "task_id": QUALIFICATION_TASK_ID,
                       "stage": progress.get("stage"), "error": str(exc),
                       "optimizer_updates": 0, "adam_calls": 0,
                       "elapsed_seconds": time.perf_counter() - started,
                       "completed_stages": progress.get("completed_stages", [])}
        except torch.cuda.OutOfMemoryError as exc:
            failure = {"status": "PREFLIGHT_BLOCKED_RESOURCE", "task_id": QUALIFICATION_TASK_ID,
                       "stage": progress.get("stage"), "error": str(exc),
                       "optimizer_updates": 0, "adam_calls": 0,
                       "elapsed_seconds": time.perf_counter() - started,
                       "completed_stages": progress.get("completed_stages", [])}
        except HarnessError as exc:
            failure = {"status": "INVALID_HARNESS", "task_id": QUALIFICATION_TASK_ID,
                       "stage": progress.get("stage"), "error": str(exc),
                       "optimizer_updates": 0, "adam_calls": 0,
                       "elapsed_seconds": time.perf_counter() - started,
                       "completed_stages": progress.get("completed_stages", [])}
        except (q12.HarnessCheckError, prior.HarnessCheckError) as exc:
            failure = {"status": "INVALID_HARNESS", "task_id": QUALIFICATION_TASK_ID,
                       "stage": progress.get("stage"), "error": str(exc),
                       "optimizer_updates": 0, "adam_calls": 0,
                       "elapsed_seconds": time.perf_counter() - started,
                       "completed_stages": progress.get("completed_stages", [])}
        _write_json(output_dir / "failure_record.json", failure)
        return failure


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require(args.qualification_only,
             "this task requires --qualification-only; optimizer and recovery modes are disabled")
    _require(args.worker_payload is None,
             "worker mode is disabled for the zero-update qualification")
    return _qualification_preflight(args)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-payload", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--mode", choices=("preflight", "resume"))
    for name in ("source-checkpoint", "source-report", "source-start-state",
                 "source-endpoint-checkpoint", "fixed-population", "config", "data",
                 "evaluator", "qualification-report", "reference-run",
                 "structure-coefficient", "gradient-reference-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--target-hours", type=float, choices=(1.0, 2.0, 3.0), default=3.0)
    args = parser.parse_args(argv)
    if args.worker_payload is not None:
        raise HarnessError("worker mode is disabled for the zero-update qualification")
    _require(args.mode is not None, "normal execution requires an explicit --mode")
    _require(args.qualification_only,
             "this reviewed task requires --qualification-only; update and recovery modes are disabled")
    required_paths = (
        "source_checkpoint", "source_report", "source_start_state",
        "source_endpoint_checkpoint", "fixed_population", "config", "data",
        "evaluator", "qualification_report", "reference_run", "structure_coefficient",
        "gradient_reference_dir", "output_dir",
    )
    missing = [name for name in required_paths if getattr(args, name) is None]
    _require(not missing, f"normal execution is missing explicit paths: {missing}")
    if args.mode == "preflight":
        _require(args.resume_checkpoint is None, "--resume-checkpoint only applies to resume mode")
    else:
        _require(args.resume_checkpoint is not None, "resume mode requires an explicit checkpoint")
    return args


def main() -> None:
    args = _parse_args()
    result = _preflight(args)
    print(json.dumps(_safe_json(result), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
