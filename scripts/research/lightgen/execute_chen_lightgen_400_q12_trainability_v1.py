"""Qualify the q12 LightGen400 path and its recoverable engineering updates."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    PIXEL_PITCH_M,
    WAVELENGTH_M,
)
from ldm_project.v2.application.research.lightgen.chen_lightgen_400_finite_pixel import (
    FinitePixelChenLightGen400,
    FinitePixelConfig,
    FinitePixelRSLinear,
    _legendre_rule,
)
from ldm_project.v2.application.research.lightgen.chen_lightgen_400_rs_convolution import (
    FinitePixelRayleighSommerfeld,
)
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_json, atomic_write_new_json
from scripts.research.lightgen import diagnose_chen_lightgen_400_sampling_closure_v1 as sampling
from scripts.research.lightgen import execute_chen_lightgen_400_finite_pixel_reconstruction_ab_v1 as prior
from scripts.research.lightgen import execute_chen_lightgen_400_mnist_baseline_v1 as baseline


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CHEN_LIGHTGEN_400_Q12_TRAINABILITY_V1"
AUTHORITY = "USER_APPROVED_CHEN_LIGHTGEN_400_Q12_TRAINABILITY_20261018"
BACKEND = "rs_linear"
ORDER = 12
REFERENCE_ORDER = 16
WINDOW = 4096
COMMON_ROI_PIXELS = 500
FIELD_TOLERANCE = 1.0e-4
FINITE_DIFFERENCE_TOLERANCE = 1.0e-2
FD_STEPS = (2.5e-3, 1.25e-3)
FIXED_GAIN = prior.FIXED_GAIN
LOGICAL_BATCH = 10
FIT_CLASSES = 10
TASK_SIZE = 28
STATES = ("source_step_30000", "historical_500_step_endpoint")
CONVOLUTION_SCRIPT = "scripts/research/lightgen/execute_chen_lightgen_400_q12_trainability_v1.py"


class HarnessCheckError(RuntimeError):
    """An explicit input, numerical, or engineering check rejected the path."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessCheckError(message)


def _validate_population_labels(population: dict[str, torch.Tensor], labels: torch.Tensor) -> None:
    fit_rows = population["fit_rows"].long().cpu()
    confirm_rows = population["confirm_rows"].long().cpu()
    _require(fit_rows.numel() == 100 and confirm_rows.numel() == 200,
             "fixed population must contain 100 fit and 200 confirmation rows")
    all_rows = torch.cat((fit_rows, confirm_rows))
    _require(bool((all_rows >= 0).all()) and bool((all_rows < labels.numel()).all()),
             "fixed population contains a row outside canonical TRAIN labels")
    expected_fit_labels = torch.arange(10).repeat_interleave(10)
    expected_confirm_labels = torch.arange(10).repeat_interleave(20)
    _require(torch.equal(labels[fit_rows], expected_fit_labels)
             and torch.equal(labels[confirm_rows], expected_confirm_labels),
             "fixed population labels differ from canonical TRAIN labels or class order")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _qualification_cache_run_dir(raw_dir: Path) -> Path:
    _require(raw_dir.name == "qualification_raw" and raw_dir.is_dir(),
             "qualification cache must be the explicit qualification_raw directory")
    return raw_dir.parent


def _relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    a = candidate.to(torch.complex128).reshape(-1)
    b = reference.to(torch.complex128).reshape(-1)
    norm = torch.linalg.vector_norm(b)
    if float(norm) == 0.0:
        return float(torch.linalg.vector_norm(a - b))
    phase = torch.exp(-1j * torch.angle(torch.vdot(b, a)))
    return float(torch.linalg.vector_norm(a * phase - b) / norm)


def _status(measurements: dict[str, float], required: set[str], tolerance: float) -> str:
    if not required or not required.issubset(measurements):
        return "INVALID_HARNESS"
    values = [measurements[key] for key in required]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        return "INVALID_HARNESS"
    return "NUMERICALLY_QUALIFIED" if max(values) <= tolerance else "NUMERICAL_NOT_QUALIFIED"


def _write_checkpoint(path: Path, explicit_inputs: dict[str, str], completed: set[str], stage: str) -> None:
    state = {"schema": "chen_lightgen_400_q12_preflight_state.v1", "task_id": TASK_ID,
             "authorization_id": AUTHORITY, "explicit_inputs": explicit_inputs,
             "completed_stages": sorted(completed), "stage": stage,
             "updated_at_unix": time.time()}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _require(loaded.get("schema") == state["schema"] and loaded.get("task_id") == TASK_ID
             and loaded.get("explicit_inputs") == explicit_inputs
             and loaded.get("completed_stages") == sorted(completed),
             "q12 preflight checkpoint failed its write/readback check")


def _save_json(path: Path, value: dict[str, Any], *, new: bool = False) -> None:
    safe = baseline._json_safe(value)
    if new:
        atomic_write_new_json(path, safe)
    else:
        atomic_write_json(path, safe)


def _load_context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    paths = {name: _resolve(getattr(args, name)) for name in (
        "source_checkpoint", "source_report", "source_start_state", "source_endpoint_checkpoint",
        "fixed_population", "config", "data", "evaluator", "qualification_report",
        "qualification_cache", "output_dir")}
    qualification_text = paths["qualification_report"].read_text(encoding="utf-8")
    _require(len(qualification_text.strip()) >= 100,
             "explicit previous qualification report is empty or incomplete")
    source_args = argparse.Namespace(**{
        **{name: paths[name] for name in (
            "source_checkpoint", "source_report", "source_start_state", "source_endpoint_checkpoint",
            "fixed_population", "config", "data", "evaluator")},
    })
    loaded = prior._load_inputs(source_args, device)
    (loaded_paths, config, checkpoint, report, source_model, references, start_state, endpoint_payload,
     pixels, labels, population, evaluator) = loaded
    _validate_population_labels(population, labels)
    _require(config["optics"]["active_shape"] == [400, 400]
             and math.isclose(config["optics"]["pixel_pitch_m"], PIXEL_PITCH_M)
             and math.isclose(config["optics"]["wavelength_m"], WAVELENGTH_M),
             "loaded LightGen400 geometry differs from the fixed q12 task")
    phases = prior._source_phase_states(source_model, start_state, endpoint_payload)
    _require(tuple(phases) == STATES, "explicit phase states do not match the registered start and endpoint")
    return {"paths": paths, "config": config, "checkpoint": checkpoint, "source_report": report,
            "qualification_report_characters": len(qualification_text),
            "source_model": source_model, "references": references, "start_state": start_state,
            "endpoint_payload": endpoint_payload, "pixels": pixels, "labels": labels,
            "population": population, "evaluator": evaluator, "state_phases": phases,
            "beta": float(checkpoint["bogt_weights"]["rec"])}


class _Heartbeat:
    def __init__(self, output_dir: Path, progress: dict[str, Any]):
        self.output_dir = output_dir
        self.progress = progress
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(60.0):
            atomic_write_json(self.output_dir / "heartbeat.json", {
                "task_id": TASK_ID, "stage": self.progress.get("stage"),
                "last_progress": self.progress.copy(), "unix_time": time.time(),
                "cuda_allocated_bytes": (torch.cuda.memory_allocated()
                                          if torch.cuda.is_available() else None),
                "cuda_reserved_bytes": (torch.cuda.memory_reserved()
                                        if torch.cuda.is_available() else None),
            })

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop.set()
        self.thread.join(timeout=2.0)


def _single_pixel_rs_integral(order: int, target_offset_cells: tuple[float, float],
                              numerical_aperture_x: float) -> complex:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    coords = nodes * PIXEL_PITCH_M / 2.0
    weight2 = np.outer(weights, weights).reshape(-1) * PIXEL_PITCH_M**2 / 4.0
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    source_x = xx.reshape(-1)
    source_y = yy.reshape(-1)
    wave_number = 2.0 * math.pi / WAVELENGTH_M
    source_field = np.exp(1j * wave_number * numerical_aperture_x * source_x)
    target_x = target_offset_cells[1] * PIXEL_PITCH_M
    target_y = target_offset_cells[0] * PIXEL_PITCH_M
    radius = np.sqrt(3.0e-3**2 + (target_y - source_y)**2 + (target_x - source_x)**2)
    kernel = (3.0e-3 / radius**2) * (1.0 / (2.0 * math.pi * radius)
              + 1.0 / (1j * WAVELENGTH_M)) * np.exp(1j * wave_number * radius)
    return complex(np.sum(kernel * source_field * weight2, dtype=np.complex128))


def _single_pixel_stress_check(device: torch.device) -> dict[str, Any]:
    cases = []
    offsets = ((0.0, 0.0), (0.0, 200.0), (0.0, 400.0), (0.0, 450.0))
    for tilt in (0.0, -0.4):
        for offset in offsets:
            references = {order: _single_pixel_rs_integral(order, offset, tilt)
                          for order in (24, 32, 48)}
            convergence = abs(references[32] - references[48]) / max(abs(references[48]), 1.0e-300)
            production: dict[str, float] = {}
            for order in (12, 16):
                nodes, _ = _legendre_rule(order, device=device)
                yy, xx = torch.meshgrid(nodes, nodes, indexing="ij")
                source = torch.exp(1j * (2.0 * math.pi / WAVELENGTH_M) * tilt
                                   * xx * PIXEL_PITCH_M / 2.0).reshape(1, order**2, 1, 1).to(torch.complex128)
                op = FinitePixelRayleighSommerfeld(order, distance_m=3.0e-3).to(device)
                with torch.no_grad():
                    actual = op(source, output_shape=(1, 1), centered=True,
                                output_offset_cells=offset)[0, 0, 0, 0]
                ref = torch.tensor(references[48], dtype=torch.complex128, device=device)
                production[str(order)] = float((actual - ref).abs() / ref.abs().clamp_min(1.0e-300))
            cases.append({"numerical_aperture_x": tilt, "target_offset_cells": list(offset),
                          "direct_gl24": [references[24].real, references[24].imag],
                          "direct_gl32": [references[32].real, references[32].imag],
                          "direct_gl48": [references[48].real, references[48].imag],
                          "reference_gl24_vs_gl32_relative": abs(references[24] - references[32])
                          / max(abs(references[32]), 1.0e-300),
                          "reference_gl32_vs_gl48_relative": convergence,
                          "reference_status": ("NUMERICALLY_QUALIFIED" if convergence <= FIELD_TOLERANCE
                                               else "NUMERICAL_NOT_QUALIFIED"),
                          "q12_relative_to_gl48": production["12"],
                          "q16_relative_to_gl48": production["16"],
                          "scope": "non_gating_stress_boundary_probe"})
    reference_status = ("NUMERICALLY_QUALIFIED" if all(
        case["reference_status"] == "NUMERICALLY_QUALIFIED" for case in cases
    ) else "NUMERICAL_NOT_QUALIFIED")
    return {"status": "MEASURED_NON_GATING", "reference_status": reference_status,
            "orders": [24, 32, 48],
            "cases": cases, "gate_role": "labels q12 applicability boundary; fixed MNIST path remains the candidate gate"}


def _direct_rs_qualification(device: torch.device, output_dir: Path,
                             progress: dict[str, Any]) -> dict[str, Any]:
    patterns = prior._make_rs_patterns()
    records: dict[str, Any] = {}
    measurements: dict[str, float] = {}
    required: set[str] = set()
    for name, pattern in patterns.items():
        source_pattern = np.asarray(pattern, dtype=np.complex64).astype(np.complex128)
        references: dict[int, np.ndarray] = {}
        derivatives: dict[int, complex] = {}
        for order in (24, 32, 48):
            references[order], derivatives[order] = sampling._rayleigh_sommerfeld_pixel_integral(
                source_pattern, order)
        ref48 = references[48]
        norm48 = max(float(np.linalg.norm(ref48)), 1.0e-300)
        reference_value_error_24_32 = float(np.linalg.norm(references[24] - references[32])
                                            / max(float(np.linalg.norm(references[32])), 1.0e-300))
        reference_value_error_32_48 = float(np.linalg.norm(references[32] - ref48) / norm48)
        reference_derivative_error = abs(derivatives[32] - derivatives[48]) \
            / max(abs(derivatives[48]), 1.0e-14)
        records[name] = {"reference_gl24_vs_gl32_relative_l2": reference_value_error_24_32,
                         "reference_gl32_vs_gl48_relative_l2": reference_value_error_32_48,
                         "reference_derivative_gl32_vs_gl48_relative": reference_derivative_error,
                         "reference_norm": float(np.linalg.norm(ref48)),
                         "reference_derivative_gl48": [derivatives[48].real, derivatives[48].imag]}
        for key, value in ((f"{name}:reference_value_24_32", reference_value_error_24_32),
                           (f"{name}:reference_value_32_48", reference_value_error_32_48),
                           (f"{name}:reference_derivative_32_48", reference_derivative_error)):
            required.add(key)
            measurements[key] = value
        source = torch.as_tensor(source_pattern, dtype=torch.complex64, device=device)
        base = source[None, None].expand(1, ORDER**2, -1, -1)
        mask = torch.zeros((1, 1, 32, 32), dtype=torch.float32, device=device)
        mask[0, 0, 16, 16] = 1.0
        for order in (ORDER, REFERENCE_ORDER):
            model_config = FinitePixelConfig(
                quadrature_order=order, window_cells=1536, distance_m=3.0e-3,
                checkpoint_propagation=False, propagation_backend=BACKEND)
            propagator = FinitePixelRSLinear(order, (32, 32), model_config).to(device)
            source_q = source[None, None].expand(1, order**2, -1, -1)
            with torch.no_grad():
                _, _, center = propagator(source_q, output_shape=(32, 32), capture_center=True)
                _, _, derivative_center = propagator(1j * source_q * mask, output_shape=(32, 32),
                                                     capture_center=True)
            actual = prior._select_rs_center_probes(center).detach().cpu().numpy().astype(np.complex128)
            derivative_actual = complex(prior._select_rs_center_probes(derivative_center)[12].detach().cpu())
            overlap = np.vdot(ref48, actual)
            phase = np.exp(-1j * np.angle(overlap))
            value_error = float(np.linalg.norm(actual * phase - ref48) / norm48)
            derivative_error = float(abs(derivative_actual * phase - derivatives[48])
                                     / max(abs(derivatives[48]), 1.0e-14))
            records[name][f"q{order}"] = {"value_relative_l2_to_gl48": value_error,
                                          "derivative_relative_to_gl48": derivative_error,
                                          "global_phase_alignment_rad": float(np.angle(overlap))}
            for suffix, value in (("value", value_error), ("derivative", derivative_error)):
                key = f"{name}:q{order}:{suffix}"
                required.add(key)
                measurements[key] = value
            del propagator, source_q, center, derivative_center
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
        progress.update({"stage": "independent_GL24_GL32_GL48_reference", "pattern": name,
                         "completed_patterns": list(records)})
        atomic_write_json(output_dir / "progress.json", progress.copy())
        del source, base, mask
    ref_keys = {key for key in required if "reference_" in key}
    production_keys = {key for key in required if ":q12:" in key or ":q16:" in key}
    reference_status = _status(measurements, ref_keys, FIELD_TOLERANCE)
    production_status = _status(measurements, production_keys, FIELD_TOLERANCE)
    stress = _single_pixel_stress_check(device)
    stress_reference_status = stress["reference_status"]
    result = {"status": "NUMERICALLY_QUALIFIED" if reference_status == production_status ==
              stress_reference_status ==
              "NUMERICALLY_QUALIFIED" else "NUMERICAL_NOT_QUALIFIED",
              "reference_status": reference_status,
              "stress_reference_status": stress_reference_status,
              "production_status": production_status,
              "orders": [24, 32, 48], "candidate_order": ORDER,
              "reference_order": REFERENCE_ORDER, "measurements": measurements,
              "required_measurements": sorted(required), "patterns": records,
              "single_pixel_stress": stress, "tolerance": FIELD_TOLERANCE,
              "reference": "independent direct real-space Rayleigh-Sommerfeld Gauss integration"}
    return result


def _run_microbatch(
    model: FinitePixelChenLightGen400, pixels: torch.Tensor, rows: torch.Tensor,
    beta: float, eta: float, branch: str, microbatch: int, device: torch.device,
) -> tuple[float, dict[str, torch.Tensor], float, int]:
    _require(microbatch in (1, 2), "logical batch10 measurement only accepts microbatch1 or2")
    _require(rows.numel() == LOGICAL_BATCH, "logical batch measurement must contain ten fixed fit rows")
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    total_loss = 0.0
    for start in range(0, LOGICAL_BATCH, microbatch):
        part = rows[start:start + microbatch]
        target, _ = baseline._image_batch(pixels, part, device)
        result = model.forward_from_image(target)
        prediction = FIXED_GAIN * result["physical_w_per_pixel"].to(torch.float64)
        target64 = target.to(torch.float64)
        mse = (prediction - target64).square().mean(dtype=torch.float64)
        if branch == "B":
            prediction28 = F.adaptive_avg_pool2d(prediction.to(torch.float32), (TASK_SIZE, TASK_SIZE))
            target28 = F.adaptive_avg_pool2d(target, (TASK_SIZE, TASK_SIZE))
            structure = 1.0 - baseline._ssim_batch(prediction28, target28).to(torch.float64).mean()
            loss = beta * mse + eta * structure
        else:
            prediction28 = target28 = structure = None
            loss = beta * mse
        (loss * (part.numel() / LOGICAL_BATCH)).backward()
        total_loss += float(loss.detach().cpu()) * part.numel() / LOGICAL_BATCH
        del target, result, prediction, target64, mse, prediction28, target28, structure, loss
    gradients = {name: parameter.grad.detach().cpu().clone()
                 for name, parameter in zip(("encoder_phase", "decoder_phase"),
                                            (model.encoder_phase, model.decoder_phase), strict=True)}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        peak = torch.cuda.max_memory_reserved(device)
    else:
        elapsed = time.perf_counter() - started
        peak = 0
    return total_loss, gradients, elapsed, peak


def _cost_probe(
    phases: dict[str, torch.Tensor], population: dict[str, torch.Tensor], pixels: torch.Tensor,
    beta: float, eta: float, branch: str, device: torch.device, output_dir: Path,
) -> dict[str, Any]:
    rows = population["fit_rows_by_class"][:, 0].long()
    results: dict[str, Any] = {}
    for microbatch in (1, 2):
        result_path = output_dir / f"cost_{branch}_microbatch_{microbatch}.json"
        if result_path.is_file():
            results[str(microbatch)] = json.loads(result_path.read_text(encoding="utf-8"))
            continue
        model = prior._model_from_phases(phases, ORDER, WINDOW, device, BACKEND)
        prior._set_trainable(model)
        loss, gradients, elapsed, peak = _run_microbatch(
            model, pixels, rows, beta, eta, branch, microbatch, device)
        result = {"branch": branch, "microbatch": microbatch, "loss": loss,
                  "gradient_l2": {key: float(torch.linalg.vector_norm(value.double()))
                                  for key, value in gradients.items()},
                  "elapsed_seconds": elapsed, "peak_reserved_vram_bytes": peak,
                  "logical_batch": LOGICAL_BATCH, "optimizer_calls": 0,
                  "status": "MEASURED"}
        _save_json(result_path, result, new=True)
        results[str(microbatch)] = result
        del model, gradients
        if device.type == "cuda":
            torch.cuda.empty_cache()
    m1, m2 = results["1"], results["2"]
    loss_error = abs(m1["loss"] - m2["loss"]) / max(abs(m1["loss"]), 1.0e-30)
    gradient_errors = {}
    # Recompute gradient tensors only for the two microbatch comparisons and save their exact values.
    for microbatch in (1, 2):
        model = prior._model_from_phases(phases, ORDER, WINDOW, device, BACKEND)
        prior._set_trainable(model)
        _, gradient, _, _ = _run_microbatch(model, pixels, rows, beta, eta, branch, microbatch, device)
        raw_path = output_dir / f"cost_{branch}_microbatch_{microbatch}_gradients.pt"
        if not raw_path.is_file():
            atomic_torch_save_new(raw_path, {"branch": branch, "microbatch": microbatch,
                                             "rows": rows.cpu(), "gradients": gradient})
        del model
        results[str(microbatch)]["gradient_payload"] = str(raw_path)
    g1 = torch.load(results["1"]["gradient_payload"], map_location="cpu", weights_only=False)["gradients"]
    g2 = torch.load(results["2"]["gradient_payload"], map_location="cpu", weights_only=False)["gradients"]
    for name in g1:
        gradient_errors[name] = float(torch.linalg.vector_norm(g1[name].double() - g2[name].double())
                                      / torch.linalg.vector_norm(g1[name].double()).clamp_min(1.0e-30))
    passed = loss_error <= 1.0e-5 and max(gradient_errors.values()) <= 1.0e-4
    order = sorted((1, 2), key=lambda value: results[str(value)]["elapsed_seconds"])[0]
    summary = {"status": "NUMERICALLY_QUALIFIED" if passed else "NUMERICAL_NOT_QUALIFIED",
               "branch": branch, "microbatch_results": results,
               "loss_relative_error_microbatch1_vs2": loss_error,
               "gradient_relative_l2_error_by_group": gradient_errors,
               "tolerance": {"loss": 1.0e-5, "gradient": 1.0e-4},
               "selected_microbatch_by_observed_wall_time": order,
               "timing_scope": "one real CUDA logical batch10 forward/backward; no optimizer update"}
    _save_json(output_dir / f"cost_{branch}_summary.json", summary)
    return summary


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _raw_full_chain(out_dir: Path, stage: str, state: str, row: int, label: int, order: int,
                    phases: dict[str, torch.Tensor] | None, pixels: torch.Tensor,
                    device: torch.device, cache_dir: Path | None = None) -> dict[str, Any]:
    raw_path = prior._raw_observation_path(out_dir, stage, state, row, order)
    if raw_path.is_file():
        return prior._load_full_chain_raw(out_dir, stage, state, row, label, order)
    if cache_dir is not None:
        cache_run_dir = _qualification_cache_run_dir(cache_dir)
        cached_path = prior._raw_observation_path(cache_run_dir, "reference", state, row, order)
        if cached_path.is_file():
            cached = prior._load_full_chain_raw(cache_run_dir, "reference", state, row, label, order)
            prior._save_full_chain_raw(out_dir, stage, state, {
                "result": cached["result"], "row": row, "label": label,
            }, order)
            return prior._load_full_chain_raw(out_dir, stage, state, row, label, order)
    _require(phases is not None, f"missing {stage} q{order} sample and no explicit phases to produce it")
    comparison = prior._full_chain_comparison(phases, row, label, pixels, order, device)[0]
    prior._save_full_chain_raw(out_dir, stage, state, comparison, order)
    del comparison
    _sync(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prior._load_full_chain_raw(out_dir, stage, state, row, label, order)


def _validate_cache_replay(context: dict[str, Any], output_dir: Path,
                           device: torch.device) -> dict[str, Any]:
    cache_run_dir = _qualification_cache_run_dir(context["paths"]["qualification_cache"])
    rows = context["population"]["confirm_rows_by_class"][:, 0].long()
    cache_inventory: list[dict[str, Any]] = []
    cached: dict[tuple[str, int, int], dict[str, Any]] = {}
    cache_validation_error: str | None = None
    try:
        for state in STATES:
            for label, row_value in enumerate(rows):
                row = int(row_value)
                for order in (ORDER, REFERENCE_ORDER):
                    path = prior._raw_observation_path(cache_run_dir, "reference", state, row, order)
                    if not path.is_file():
                        continue
                    raw = prior._load_full_chain_raw(cache_run_dir, "reference", state, row, label, order)
                    cached[(state, row, order)] = raw
                    cache_inventory.append({"state": state, "row": row, "label": label,
                                            "order": order, "path": str(path),
                                            "status": "METADATA_AND_TENSOR_VALIDATED"})
    except (HarnessCheckError, OSError, EOFError, RuntimeError) as exc:
        cache_validation_error = f"{type(exc).__name__}: {exc}"

    expected_cache_counts = {str(ORDER): len(STATES) * rows.numel(),
                             str(REFERENCE_ORDER): 4}
    actual_cache_counts = {str(order): sum(key[2] == order for key in cached)
                           for order in (ORDER, REFERENCE_ORDER)}
    replay_row = int(rows[0])
    replay_label = 0
    replay_errors: dict[str, Any] = {}
    if cache_validation_error is None:
        for order in (ORDER, REFERENCE_ORDER):
            replay_key = (STATES[0], replay_row, order)
            if replay_key not in cached:
                cache_validation_error = f"missing replay source cache for state={STATES[0]}, row={replay_row}, q{order}"
                break
            replay = prior._full_chain_comparison(
                context["state_phases"][STATES[0]], replay_row, replay_label,
                context["pixels"], order, device
            )[0]
            residual = prior._full_chain_residual(replay, cached[replay_key])
            measured = {**residual["per_plane_field_relative_l2"],
                        "ols_complex_relative_l2": residual["ols_complex_relative_l2"],
                        "camera_roi_w_per_pixel_relative_l2": residual["camera_roi_w_per_pixel_relative_l2"]}
            replay_errors[str(order)] = {"status": _status(measured, set(measured), FIELD_TOLERANCE),
                                         "errors": residual}
            del replay
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
            if replay_errors[str(order)]["status"] != "NUMERICALLY_QUALIFIED":
                cache_validation_error = f"production replay rejected the q{order} source cache"
                break

    # Missing cache members are safe to regenerate, but the available cache set must include
    # a fully replayed q12/q16 pair and each loaded item must match its declared identity.
    accepted = cache_validation_error is None and all(
        replay_errors.get(str(order), {}).get("status") == "NUMERICALLY_QUALIFIED"
        for order in (ORDER, REFERENCE_ORDER)
    )
    return {"status": "NUMERICALLY_QUALIFIED" if accepted else "CACHE_REJECTED_REGENERATE_ALL",
            "cache_accepted": accepted,
            "source_cache": str(context["paths"]["qualification_cache"]), "replay_state": STATES[0],
            "replay_row": replay_row, "replay_label": replay_label,
            "replayed_orders": [ORDER, REFERENCE_ORDER], "replay_errors": replay_errors,
            "cache_inventory": cache_inventory,
            "expected_cache_counts": expected_cache_counts,
            "actual_cache_counts": actual_cache_counts,
            "cache_validation_error": cache_validation_error,
            "interpretation": "both quadrature caches are accepted only after one same-row production replay at q12 and q16; any loaded member identity/content failure rejects both cache families; missing members are regenerated"}


def _compare_node_fields(candidate: tuple[torch.Tensor, ...], reference: tuple[torch.Tensor, ...]
                         ) -> dict[str, Any]:
    _require(len(candidate) == len(reference) == 7,
             "mixed-order node-field comparison requires all seven unique planes")
    overlap = torch.zeros((), dtype=torch.complex128)
    for a, b in zip(candidate, reference, strict=True):
        _require(a.shape == b.shape, "mixed-order node-field coordinate arrays have different shapes")
        overlap += torch.vdot(b.to(torch.complex128).reshape(-1), a.to(torch.complex128).reshape(-1))
    phase = torch.exp(-1j * torch.angle(overlap))
    errors: dict[str, float] = {}
    samples: list[dict[str, Any]] = []
    for index, (a, b) in enumerate(zip(candidate, reference, strict=True)):
        aa, bb = a.to(torch.complex128), b.to(torch.complex128)
        errors[f"plane_{index + 1}"] = float(torch.linalg.vector_norm(aa * phase - bb)
                                              / torch.linalg.vector_norm(bb).clamp_min(1.0e-300))
        h, w = a.shape[-2:]
        points = ((h // 2, w // 2), (h // 4, w // 4), (h // 4, 3 * w // 4),
                  (3 * h // 4, w // 4), (3 * h // 4, 3 * w // 4))
        node_count = a.shape[1]
        node_ids = (0, node_count // 4, node_count // 2, 3 * node_count // 4, node_count - 1)
        samples.append({"plane": index + 1,
                        "node_indices": list(node_ids),
                        "pixel_coordinates_yx": [list(p) for p in points],
                        "candidate_q12": [[float((a[0, node, y, x] * phase).real),
                                            float((a[0, node, y, x] * phase).imag)]
                                           for node, (y, x) in zip(node_ids, points, strict=True)],
                        "reference_q16_source_to_q12_nodes": [[float(b[0, node, y, x].real),
                                                                float(b[0, node, y, x].imag)]
                                                               for node, (y, x) in zip(node_ids, points, strict=True)]})
    return {"global_phase_alignment_rad": float(torch.angle(overlap)),
            "per_plane_node_field_relative_l2": errors,
            "max_node_field_relative_l2": max(errors.values()),
            "fixed_node_samples": samples,
            "node_coordinates": "production q12 Gauss-Legendre observer nodes; q16 source integrated directly at those coordinates"}


def _node_field_pair(context: dict[str, Any], state: str, row: int, label: int,
                     device: torch.device) -> dict[str, Any]:
    target, _ = baseline._image_batch(context["pixels"], torch.tensor([row]), device)
    outputs: list[tuple[torch.Tensor, ...]] = []
    for order in (ORDER, REFERENCE_ORDER):
        model = prior._model_from_phases(context["state_phases"][state], order, WINDOW, device, BACKEND)
        model.eval()
        with torch.no_grad():
            result = model.forward_from_image(
                target, capture_fields=True, diagnostic_observer_order=ORDER
            )
        nodes = tuple(value.detach().cpu() for value in result["diagnostic_node_fields_at_order"])
        _require(len(nodes) == 7, "production mixed-order observer omitted a propagation plane")
        outputs.append(nodes)
        del model, result
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    record = _compare_node_fields(outputs[0], outputs[1])
    record.update({"state": state, "row": row, "label": label,
                   "source_order_candidate": ORDER, "source_order_reference": REFERENCE_ORDER,
                   "observer_order": ORDER})
    del outputs, target
    return record


def _gradient_for_order(context: dict[str, Any], phases: dict[str, torch.Tensor], row: int,
                        order: int, device: torch.device) -> list[torch.Tensor]:
    target, _ = baseline._image_batch(context["pixels"], torch.tensor([row]), device)
    model = prior._model_from_phases(phases, order, WINDOW, device, BACKEND)
    prior._set_trainable(model)
    result = model.forward_from_image(target)
    prediction = FIXED_GAIN * result["physical_w_per_pixel"].to(torch.float64)
    mse = (prediction - target.to(torch.float64)).square().mean(dtype=torch.float64)
    loss = context["beta"] * mse
    gradients = torch.autograd.grad(loss, (model.encoder_phase, model.decoder_phase))
    flat = [group[index, 0].detach().cpu().to(torch.float64).clone()
            for group, count in zip(gradients, (2, 4), strict=True) for index in range(count)]
    _require(len(flat) == 6 and all(torch.isfinite(value).all() for value in flat),
             "six-layer q12/q16 gradient measurement is incomplete or nonfinite")
    _require(all(float(torch.linalg.vector_norm(value)) > 0.0 for value in flat),
             "a physical phase layer has zero reconstruction gradient")
    del target, model, result, prediction, mse, loss, gradients
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return flat


def _compare_gradients(candidate: list[torch.Tensor], reference: list[torch.Tensor]) -> dict[str, Any]:
    _require(len(candidate) == len(reference) == 6, "gradient comparison does not cover six physical layers")
    errors = [float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b).clamp_min(1.0e-300))
              for a, b in zip(candidate, reference, strict=True)]
    return {"per_layer_relative_l2": errors, "max_relative_l2": max(errors),
            "candidate_layer_l2": [float(torch.linalg.vector_norm(value)) for value in candidate],
            "reference_layer_l2": [float(torch.linalg.vector_norm(value)) for value in reference]}


def _loss_at_phases(context: dict[str, Any], state_phases: dict[str, torch.Tensor], row: int,
                    order: int, directions: list[torch.Tensor], epsilon: float,
                    sign: float, device: torch.device) -> tuple[float, float, float]:
    phases, actual_projection, actual_delta_l2 = _apply_represented_phase_delta(
        state_phases, directions, epsilon, sign, device
    )
    target, _ = baseline._image_batch(context["pixels"], torch.tensor([row]), device)
    model = prior._model_from_phases(phases, order, WINDOW, device, BACKEND)
    model.eval()
    with torch.no_grad():
        result = model.forward_from_image(target)
        prediction = FIXED_GAIN * result["physical_w_per_pixel"].to(torch.float64)
        loss = context["beta"] * (prediction - target.to(torch.float64)).square().mean(dtype=torch.float64)
        value = float(loss.cpu())
    del target, phases, model, result, prediction, loss
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return value, actual_projection, actual_delta_l2


def _apply_represented_phase_delta(
    state_phases: dict[str, torch.Tensor], directions: list[torch.Tensor],
    epsilon: float, sign: float, device: torch.device,
) -> tuple[dict[str, torch.Tensor], float, float]:
    _require(len(directions) == 6, "phase perturbation must declare all six layers")
    _require(epsilon > 0.0 and sign in (-1.0, 1.0), "phase perturbation step is invalid")
    phases = {name: value.to(device).clone() for name, value in state_phases.items()}
    actual_projection = 0.0
    actual_delta_squared = 0.0
    layer_index = 0
    for group_name, count in (("encoder_phase", 2), ("decoder_phase", 4)):
        for layer in range(count):
            direction = directions[layer_index].to(device=device, dtype=torch.float64)
            if bool(torch.count_nonzero(direction)):
                before = phases[group_name][layer, 0].to(torch.float64)
                proposed = (before + sign * epsilon * direction).to(phases[group_name].dtype)
                phases[group_name][layer, 0].copy_(proposed)
                actual = proposed.to(torch.float64) - before
                actual_projection += float((actual * direction).sum().cpu())
                actual_delta_squared += float(actual.square().sum().cpu())
            layer_index += 1
    return phases, actual_projection, math.sqrt(actual_delta_squared)


def _finite_difference_check(context: dict[str, Any], state: str, row: int, label: int,
                             candidate_gradients: list[torch.Tensor],
                             reference_gradients: list[torch.Tensor], device: torch.device,
                             progress: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    directions: list[tuple[str, list[torch.Tensor]]]
    if state == STATES[0] and label == 0:
        directions = []
        for index, gradient in enumerate(candidate_gradients):
            norm = torch.linalg.vector_norm(gradient)
            unit = [torch.zeros_like(value) for value in candidate_gradients]
            unit[index] = gradient / norm
            directions.append((f"layer_{index + 1}", unit))
    else:
        norm = torch.sqrt(sum(value.square().sum() for value in candidate_gradients))
        directions = [("joint_six_layer", [value / norm for value in candidate_gradients])]
    records: list[dict[str, Any]] = []
    required: set[str] = set()
    measurements: dict[str, float] = {}
    completed_forwards = 0
    executed_forwards = 0
    reused_forwards = 0
    for direction_name, direction in directions:
        direction_path = output_dir / f"finite_difference_{state}_row{row}_{direction_name}.json"
        if direction_path.is_file():
            try:
                saved = json.loads(direction_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise HarnessCheckError(f"saved finite-difference direction is unreadable: {direction_path}") from exc
            _require(saved.get("schema") == "chen_lightgen_400_q12_fd_direction.v1"
                     and saved.get("state") == state and saved.get("row") == row
                     and saved.get("label") == label and saved.get("direction") == direction_name
                     and saved.get("steps_rad") == list(FD_STEPS),
                     f"saved finite-difference direction identity differs: {direction_path}")
            path_records = saved.get("checks")
            _require(isinstance(path_records, dict) and set(path_records) == {"q12", "q16"}
                     and all(isinstance(path_records[name], list)
                             and len(path_records[name]) == len(FD_STEPS) for name in ("q12", "q16")),
                     f"saved finite-difference direction is incomplete: {direction_path}")
            for path_name in ("q12", "q16"):
                for epsilon, check in zip(FD_STEPS, path_records[path_name], strict=True):
                    _require(isinstance(check, dict) and check.get("nominal_epsilon_rad") == epsilon
                             and isinstance(check.get("degenerate"), bool)
                             and isinstance(check.get("same_sign"), bool),
                             f"saved finite-difference check is malformed: {direction_path}")
                    numeric_fields = ("plus_loss", "minus_loss", "plus_represented_directional_rad",
                                      "minus_represented_directional_rad", "plus_represented_phase_delta_l2_rad",
                                      "minus_represented_phase_delta_l2_rad", "effective_central_step_rad",
                                      "central_difference", "autodiff_directional_derivative")
                    _require(all(isinstance(check.get(name), (int, float))
                                 and math.isfinite(float(check[name])) for name in numeric_fields),
                             f"saved finite-difference values are nonfinite: {direction_path}")
                    relative = check.get("relative_error")
                    _require((check["degenerate"] and isinstance(relative, (int, float))
                              and not math.isfinite(float(relative)))
                             or (not check["degenerate"] and isinstance(relative, (int, float))
                                 and math.isfinite(float(relative))),
                             f"saved finite-difference error value is invalid: {direction_path}")
            reused_forwards += 2 * 2 * len(FD_STEPS)
        else:
            path_records = {}
            autodiff_by_order = {
                "q12": sum(float((g * d).sum()) for g, d in zip(candidate_gradients, direction, strict=True)),
                "q16": sum(float((g * d).sum()) for g, d in zip(reference_gradients, direction, strict=True)),
            }
            for order, path_name in ((ORDER, "q12"), (REFERENCE_ORDER, "q16")):
                checks = []
                for epsilon in FD_STEPS:
                    plus, plus_displacement, plus_delta_l2 = _loss_at_phases(
                        context, context["state_phases"][state], row, order, direction, epsilon, 1.0, device
                    )
                    minus, minus_displacement, minus_delta_l2 = _loss_at_phases(
                        context, context["state_phases"][state], row, order, direction, epsilon, -1.0, device
                    )
                    executed_forwards += 2
                    denominator = plus_displacement - minus_displacement
                    _require(abs(denominator) > 0.0,
                             "float32 phase storage erased the requested finite-difference displacement")
                    derivative = (plus - minus) / denominator
                    autodiff = autodiff_by_order[path_name]
                    degenerate = abs(autodiff) <= 1.0e-14
                    relative = (abs(derivative - autodiff) / max(abs(derivative), abs(autodiff), 1.0e-14)
                                if not degenerate else float("nan"))
                    checks.append({"nominal_epsilon_rad": epsilon, "plus_loss": plus, "minus_loss": minus,
                                   "plus_represented_directional_rad": plus_displacement,
                                   "minus_represented_directional_rad": minus_displacement,
                                   "plus_represented_phase_delta_l2_rad": plus_delta_l2,
                                   "minus_represented_phase_delta_l2_rad": minus_delta_l2,
                                   "effective_central_step_rad": denominator,
                                   "central_difference": derivative,
                                   "autodiff_directional_derivative": autodiff,
                                   "degenerate": degenerate,
                                   "same_sign": bool(not degenerate and derivative * autodiff > 0.0),
                                   "relative_error": relative})
                path_records[path_name] = checks
            _save_json(direction_path, {"schema": "chen_lightgen_400_q12_fd_direction.v1",
                                        "state": state, "row": row, "label": label,
                                        "direction": direction_name, "steps_rad": FD_STEPS,
                                        "checks": path_records}, new=True)
        records.append({"direction": direction_name, "paths": path_records})
        completed_forwards += 2 * 2 * len(FD_STEPS)
        for path_name in ("q12", "q16"):
            for check in path_records[path_name]:
                epsilon = check["nominal_epsilon_rad"]
                relative = float(check["relative_error"])
                key = f"{state}:{row}:{direction_name}:{path_name}:{epsilon}"
                required.add(key)
                measurements[key] = relative
        completed_direction_files = list(output_dir.glob("finite_difference_*.json"))
        total_completed_forwards = len(completed_direction_files) * 2 * 2 * len(FD_STEPS)
        expected_probe_forwards = len(directions) * 2 * 2 * len(FD_STEPS)
        progress.update({"stage": "finite_difference_72_forward_check", "state": state,
                         "row": row, "direction": direction_name,
                         "completed_perturbation_forwards": total_completed_forwards,
                         "expected_perturbation_forwards": 72,
                         "probe_completed_perturbation_forwards": completed_forwards,
                         "probe_expected_perturbation_forwards": expected_probe_forwards,
                         "executed_forwards_this_probe_this_resume": executed_forwards,
                         "reused_completed_forwards_this_probe": reused_forwards})
        atomic_write_json(output_dir / "progress.json", progress.copy())
    expected_checks = len(directions) * 2 * len(FD_STEPS)
    expected_forwards = expected_checks * 2
    _require(completed_forwards == expected_forwards,
             f"finite-difference direction covered {completed_forwards} forwards; expected {expected_forwards}")
    finite_difference_values = [measurements[key] for key in required]
    passed = (len(finite_difference_values) == expected_checks and
              all(math.isfinite(value) and value <= FINITE_DIFFERENCE_TOLERANCE for value in finite_difference_values) and
              all(entry["same_sign"] for record in records for path in record["paths"].values()
                  for entry in path))
    return {"state": state, "row": row, "label": label, "directions": records,
            "completed_forwards": completed_forwards, "expected_forwards": expected_forwards,
            "executed_forwards_this_resume": executed_forwards,
            "reused_completed_forwards": reused_forwards,
            "measurements": measurements,
            "status": "NUMERICALLY_QUALIFIED" if passed else "NUMERICAL_NOT_QUALIFIED",
            "tolerance": FINITE_DIFFERENCE_TOLERANCE}


def _fit_endpoint_raws(context: dict[str, Any], output_dir: Path, device: torch.device,
                       cache_accepted: bool) -> dict[str, Any]:
    cache_dir = context["paths"]["qualification_cache"] if cache_accepted else None
    cache_run_dir = (_qualification_cache_run_dir(cache_dir) if cache_dir is not None else None)
    rows = context["population"]["confirm_rows_by_class"][:, 0].long()
    records: list[dict[str, Any]] = []
    sources: dict[tuple[str, int, int], str] = {}
    for state in STATES:
        for row_tensor in rows:
            row = int(row_tensor)
            for stage, order in (("candidate", ORDER), ("reference", REFERENCE_ORDER)):
                local_path = prior._raw_observation_path(output_dir, stage, state, row, order)
                cache_path = (prior._raw_observation_path(cache_run_dir, "reference", state, row, order)
                              if cache_run_dir is not None else None)
                sources[(state, row, order)] = (
                    "local_resume" if local_path.is_file() else
                    "source_cache" if cache_path is not None and cache_path.is_file() else
                    "generated"
                )
    expected_forwards = sum(value == "generated" for value in sources.values())
    generated_forwards = 0
    cache_reused = 0
    local_reused = 0
    for state in STATES:
        phases = context["state_phases"][state]
        for label, row_tensor in enumerate(rows):
            row = int(row_tensor)
            candidate = _raw_full_chain(output_dir, "candidate", state, row, label, ORDER,
                                        phases, context["pixels"], device,
                                        cache_dir if cache_accepted else None)
            reference = _raw_full_chain(output_dir, "reference", state, row, label, REFERENCE_ORDER,
                                        phases, context["pixels"], device,
                                        cache_dir if cache_accepted else None)
            for order in (ORDER, REFERENCE_ORDER):
                source = sources[(state, row, order)]
                if source == "generated":
                    generated_forwards += 1
                elif source == "source_cache":
                    cache_reused += 1
                else:
                    local_reused += 1
            residual = prior._full_chain_residual(candidate, reference)
            measured = {**residual["per_plane_field_relative_l2"],
                        "ols_complex_relative_l2": residual["ols_complex_relative_l2"],
                        "camera_roi_w_per_pixel_relative_l2": residual["camera_roi_w_per_pixel_relative_l2"]}
            records.append({"state": state, "row": row, "label": label,
                            "q12_raw_path": candidate["raw_path"],
                            "q16_raw_path": reference["raw_path"], "errors": residual,
                            "status": _status(measured, set(measured), FIELD_TOLERANCE)})
            del candidate, reference
            if device.type == "cuda":
                torch.cuda.empty_cache()
            atomic_write_json(output_dir / "progress.json", {
                "stage": "full_chain_q12_vs_q16", "completed_samples": len(records),
                "expected_samples": len(STATES) * 10, "state": state, "row": row,
                "completed_generated_forwards": generated_forwards,
                "expected_generated_forwards": expected_forwards,
                "source_cache_observations_reused": cache_reused,
                "local_resume_observations_reused": local_reused})
    statuses = [record["status"] for record in records]
    status = ("NUMERICALLY_QUALIFIED" if len(records) == 20 and all(
        value == "NUMERICALLY_QUALIFIED" for value in statuses) else
        "INVALID_HARNESS" if len(records) != 20 else "NUMERICAL_NOT_QUALIFIED")
    return {"status": status,
        "states": list(STATES), "labels": list(range(10)), "records": records,
        "expected_pairs": 20, "tolerance": FIELD_TOLERANCE,
        "expected_generated_forwards": expected_forwards,
        "generated_forwards": generated_forwards,
        "source_cache_observations_reused": cache_reused,
        "local_resume_observations_reused": local_reused}


def _node_gradient_qualification(context: dict[str, Any], output_dir: Path,
                                 device: torch.device, progress: dict[str, Any]) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    for state in STATES:
        for label in (0, 1):
            row = int(context["population"]["confirm_rows_by_class"][label, 0])
            node = _node_field_pair(context, state, row, label, device)
            candidate_gradient = _gradient_for_order(
                context, context["state_phases"][state], row, ORDER, device
            )
            reference_gradient = _gradient_for_order(
                context, context["state_phases"][state], row, REFERENCE_ORDER, device
            )
            gradient = _compare_gradients(candidate_gradient, reference_gradient)
            payload_path = output_dir / f"gradients_{state}_row{row}_q12_q16.pt"
            atomic_torch_save_new(payload_path, {
                "state": state, "row": row, "label": label,
                "q12_gradient_by_layer": candidate_gradient,
                "q16_gradient_by_layer": reference_gradient,
            })
            probe = {"state": state, "row": row, "label": label,
                     "node_fields": node, "gradients": gradient,
                     "gradient_payload": str(payload_path),
                     "node_status": "NUMERICALLY_QUALIFIED" if node["max_node_field_relative_l2"] <= FIELD_TOLERANCE else "NUMERICAL_NOT_QUALIFIED",
                     "gradient_status": "NUMERICALLY_QUALIFIED" if gradient["max_relative_l2"] <= FIELD_TOLERANCE else "NUMERICAL_NOT_QUALIFIED"}
            probes.append(probe)
            _save_json(output_dir / f"node_gradient_{state}_row{row}.json", probe)
            progress.update({"stage": "mixed_node_and_six_layer_gradient", "state": state,
                             "row": row, "completed_probes": len(probes), "expected_probes": 4})
            atomic_write_json(output_dir / "progress.json", progress.copy())
    status = "NUMERICALLY_QUALIFIED" if len(probes) == 4 and all(
        probe["node_status"] == probe["gradient_status"] == "NUMERICALLY_QUALIFIED"
        for probe in probes) else "INVALID_HARNESS" if len(probes) != 4 else "NUMERICAL_NOT_QUALIFIED"
    return {"status": status, "probes": probes, "expected_probes": 4,
            "node_field_tolerance": FIELD_TOLERANCE, "gradient_tolerance": FIELD_TOLERANCE}


def _finite_difference_qualification(context: dict[str, Any], output_dir: Path,
                                     device: torch.device, progress: dict[str, Any]) -> dict[str, Any]:
    probes: list[dict[str, Any]] = []
    total_forwards = 0
    executed_forwards = 0
    reused_forwards = 0
    for state in STATES:
        for label in (0, 1):
            row = int(context["population"]["confirm_rows_by_class"][label, 0])
            gradient_path = output_dir / f"gradients_{state}_row{row}_q12_q16.pt"
            gradient_payload = torch.load(gradient_path, map_location="cpu", weights_only=False)
            result = _finite_difference_check(
                context, state, row, label, gradient_payload["q12_gradient_by_layer"],
                gradient_payload["q16_gradient_by_layer"], device, progress, output_dir
            )
            total_forwards += int(result["completed_forwards"])
            executed_forwards += int(result["executed_forwards_this_resume"])
            reused_forwards += int(result["reused_completed_forwards"])
            probes.append(result)
    _require(total_forwards == 72 and executed_forwards + reused_forwards == 72,
             f"finite-difference path covered {total_forwards} perturbation forwards; expected 72")
    status = ("NUMERICALLY_QUALIFIED" if len(probes) == 4 and all(
        value["status"] == "NUMERICALLY_QUALIFIED" for value in probes) else
        "INVALID_HARNESS" if len(probes) != 4 else "NUMERICAL_NOT_QUALIFIED")
    return {"status": status, "probes": probes, "completed_forwards": total_forwards,
            "executed_forwards_this_resume": executed_forwards,
            "reused_completed_forwards": reused_forwards,
            "expected_forwards": 72, "steps_rad": list(FD_STEPS),
            "tolerance": FINITE_DIFFERENCE_TOLERANCE}


def _calibrate_eta_q12(context: dict[str, Any], device: torch.device) -> dict[str, Any]:
    rows = context["population"]["fit_rows_by_class"][:, 0].long()
    model = prior._model_from_phases(context["state_phases"][STATES[0]], ORDER, WINDOW, device, BACKEND)
    prior._set_trainable(model)
    params = (model.encoder_phase, model.decoder_phase)
    mse_acc = [torch.zeros_like(value, dtype=torch.float64, device="cpu") for value in params]
    ssim_acc = [torch.zeros_like(value, dtype=torch.float64, device="cpu") for value in params]
    for index, row_value in enumerate(rows):
        row = int(row_value)
        target, _ = baseline._image_batch(context["pixels"], torch.tensor([row]), device)
        result = model.forward_from_image(target)
        prediction = FIXED_GAIN * result["physical_w_per_pixel"].to(torch.float64)
        mse = (prediction - target.to(torch.float64)).square().mean(dtype=torch.float64)
        prediction28 = F.adaptive_avg_pool2d(prediction.to(torch.float32), (TASK_SIZE, TASK_SIZE))
        target28 = F.adaptive_avg_pool2d(target, (TASK_SIZE, TASK_SIZE))
        structure = 1.0 - baseline._ssim_batch(prediction28, target28).to(torch.float64).mean()
        mse_grads = torch.autograd.grad(context["beta"] * mse, params, retain_graph=True)
        structure_grads = torch.autograd.grad(structure, params)
        for group_index in range(2):
            mse_acc[group_index] += mse_grads[group_index].detach().to(torch.float64).cpu() / LOGICAL_BATCH
            ssim_acc[group_index] += structure_grads[group_index].detach().to(torch.float64).cpu() / LOGICAL_BATCH
        del target, result, prediction, mse, prediction28, target28, structure, mse_grads, structure_grads
        atomic_write_json(context["paths"]["output_dir"] / "progress.json", {
            "stage": "q12_structure_weight_calibration", "completed_fit_rows": index + 1,
            "expected_fit_rows": LOGICAL_BATCH})
    mse_norm = math.sqrt(sum(float(value.square().sum()) for value in mse_acc))
    structure_norm = math.sqrt(sum(float(value.square().sum()) for value in ssim_acc))
    _require(math.isfinite(mse_norm) and mse_norm > 0.0 and math.isfinite(structure_norm)
             and structure_norm > 0.0, "q12 six-layer MSE/SSIM calibration gradient is zero or invalid")
    eta = mse_norm / structure_norm
    _require(math.isfinite(eta) and eta > 0.0, "q12 calibrated structure coefficient is invalid")
    del model
    return {"eta": eta, "beta_mse_gradient_l2": mse_norm,
            "structure_gradient_l2": structure_norm, "first_fit_batch_rows": rows.tolist(),
            "estimator": "q12_start_fit_batch_summed_gradient_equalization",
            "structure_loss": "1 - SSIM28(unclipped_camera_output, target)"}


def _save_engineering_endpoint(output_dir: Path, branch: str, result: dict[str, Any]) -> None:
    path = output_dir / f"engineering_{branch}_three_step_endpoint.pt"
    if path.exists():
        existing = torch.load(path, map_location="cpu", weights_only=False)
        _require(baseline._assert_state_close(existing, result,
                                              f"{branch}_q12_endpoint_repeat") == 0.0,
                 "existing q12 engineering endpoint differs from repeated state")
        return
    atomic_torch_save_new(path, result)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _require(baseline._assert_state_close(loaded, result,
                                          f"{branch}_q12_endpoint_readback") == 0.0,
             "q12 engineering endpoint failed write/readback")


def _branch_signal_test(context: dict[str, Any], branch: str, eta: float,
                        output_dir: Path, start_rng: dict[str, Any], device: torch.device,
                        progress: dict[str, Any], microbatch: int) -> dict[str, Any]:
    phases = context["state_phases"][STATES[0]]
    _require(all(torch.equal(phases[name], context["source_model"].state_dict()[name].cpu())
                 for name in phases), "engineering branch start phases differ from source step 30000")
    prior._set_rng(start_rng["cpu"], start_rng["cuda"])
    recovery_signal = signal.SIGINT if branch == "A" else signal.SIGTERM
    recovery_dir = output_dir / f"recovery_{branch}_signal_test"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    endpoint_path = output_dir / f"engineering_{branch}_three_step_endpoint.pt"
    progress.update({"stage": "engineering_independent_signal_resume", "branch": branch,
                     "signal": int(recovery_signal), "expected_engineering_adam_calls": 6})
    atomic_write_json(output_dir / "progress.json", progress.copy())
    evaluator_before = prior._cpu_state(context["evaluator"].state_dict())
    result = prior._branch_preflight(
        phases, context["population"], context["pixels"], context["paths"]["data"],
        branch, ORDER, context["beta"], eta, device, recovery_dir,
        endpoint_path=endpoint_path, microbatch=microbatch,
    )
    _require(result["independent_signal_recovery"]["requested_signal"] == recovery_signal
             and result["independent_signal_recovery"]["rng_equal"],
             f"q12 {branch} did not complete the required signal recovery check")
    endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
    _save_engineering_endpoint(output_dir, branch, endpoint)
    evaluator_error = baseline._assert_state_close(
        evaluator_before, prior._cpu_state(context["evaluator"].state_dict()), f"q12_{branch}_evaluator"
    )
    _require(evaluator_error == 0.0, f"q12 {branch} updated the frozen evaluator")
    result.update({"branch": branch, "signal": int(recovery_signal),
                   "endpoint_checkpoint": str(output_dir / f"engineering_{branch}_three_step_endpoint.pt"),
                   "evaluator_max_abs_difference": evaluator_error,
                   "engineering_adam_calls": 6,
                   "selected_microbatch": microbatch})
    return result


def _endpoint_numeric_qualification(context: dict[str, Any], output_dir: Path,
                                     branch_results: dict[str, Any], device: torch.device,
                                     progress: dict[str, Any]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    rows = context["population"]["confirm_rows_by_class"][[0, 1], 0].long()
    for branch, branch_result in branch_results.items():
        payload = torch.load(branch_result["endpoint_checkpoint"], map_location="cpu", weights_only=False)
        phases = {name: payload["model"][name].detach().cpu().clone()
                  for name in ("encoder_phase", "decoder_phase")}
        endpoint_state = f"engineering_{branch}_three_step_endpoint"
        for label, row_value in zip((0, 1), rows, strict=True):
            row = int(row_value)
            saved_record_path = output_dir / f"endpoint_{branch}_row{row}_q12_q16.json"
            if saved_record_path.is_file():
                saved_record = json.loads(saved_record_path.read_text(encoding="utf-8"))
                _require(saved_record.get("branch") == branch and saved_record.get("row") == row
                         and saved_record.get("label") == label,
                         f"saved engineering endpoint probe identity differs: {saved_record_path}")
                records.append(saved_record)
                continue
            q12 = _raw_full_chain(output_dir, "candidate", endpoint_state, row, label, ORDER,
                                  phases, context["pixels"], device)
            q16 = _raw_full_chain(output_dir, "reference", endpoint_state, row, label, REFERENCE_ORDER,
                                  phases, context["pixels"], device)
            residual = prior._full_chain_residual(q12, q16)
            q12_gradient = _gradient_for_order(context, phases, row, ORDER, device)
            q16_gradient = _gradient_for_order(context, phases, row, REFERENCE_ORDER, device)
            gradients = _compare_gradients(
                q12_gradient, q16_gradient)
            gradient_payload = output_dir / f"endpoint_gradients_{branch}_row{row}_q12_q16.pt"
            if not gradient_payload.is_file():
                atomic_torch_save_new(gradient_payload, {
                    "branch": branch, "row": row, "label": label,
                    "q12_gradient_by_layer": q12_gradient, "q16_gradient_by_layer": q16_gradient,
                })
            value_set = {**residual["per_plane_field_relative_l2"],
                         "ols_complex_relative_l2": residual["ols_complex_relative_l2"],
                         "camera_roi_w_per_pixel_relative_l2": residual["camera_roi_w_per_pixel_relative_l2"]}
            record = {"branch": branch, "state": "three_step_endpoint", "row": row, "label": label,
                      "value_errors": value_set, "max_value_relative_l2": max(value_set.values()),
                      "gradient_errors": gradients, "gradient_payload": str(gradient_payload),
                      "q12_raw_path": q12["raw_path"], "q16_raw_path": q16["raw_path"]}
            record["status"] = ("NUMERICALLY_QUALIFIED"
                                if record["max_value_relative_l2"] <= FIELD_TOLERANCE
                                and gradients["max_relative_l2"] <= FIELD_TOLERANCE
                                else "NUMERICAL_NOT_QUALIFIED")
            records.append(record)
            _save_json(saved_record_path, record, new=True)
            del q12, q16, q12_gradient, q16_gradient
            if device.type == "cuda":
                torch.cuda.empty_cache()
            progress.update({"stage": "engineering_endpoint_q12_q16_recheck", "branch": branch,
                             "row": row, "completed_checks": len(records), "expected_checks": 8})
            atomic_write_json(output_dir / "progress.json", progress.copy())
    status = ("NUMERICALLY_QUALIFIED" if len(records) == 8 and all(
        item["status"] == "NUMERICALLY_QUALIFIED" for item in records) else
        "INVALID_HARNESS" if len(records) != 8 else "NUMERICAL_NOT_QUALIFIED")
    return {"status": status, "records": records, "expected_checks": 8,
            "value_tolerance": FIELD_TOLERANCE, "gradient_tolerance": FIELD_TOLERANCE}


def _atomic_stage(output_dir: Path, name: str, result: dict[str, Any],
                  progress: dict[str, Any], completed: set[str], explicit_inputs: dict[str, str]) -> None:
    path = output_dir / f"{name}.json"
    if not path.exists():
        _save_json(path, result, new=True)
    completed.add(name)
    _write_checkpoint(output_dir / "preflight_state.pt", explicit_inputs, completed, name)
    progress.update({"stage": name, "status": result.get("status", "MEASURED"),
                     "completed_stages": sorted(completed)})
    atomic_write_json(output_dir / "progress.json", progress.copy())


def _read_stage(output_dir: Path, name: str) -> dict[str, Any] | None:
    path = output_dir / f"{name}.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"saved q12 stage result is invalid: {path}")
    return value


def _set_terminal(output_dir: Path, status: str, stage: str, started: float,
                  details: dict[str, Any], *, final: bool) -> dict[str, Any]:
    report = {"schema": "chen_lightgen_400_q12_trainability_preflight.v1", "task_id": TASK_ID,
              "authorization_id": AUTHORITY, "status": status, "terminal_stage": stage,
              "optimizer_updates": 0, "engineering_adam_calls": details.get("engineering_adam_calls", 0),
              "official_test_reads": 0, "elapsed_seconds": time.perf_counter() - started,
              "candidate_order": ORDER, "reference_order": REFERENCE_ORDER,
              "decision": "q12 engineering basis only; no 500-step A/B was executed",
              "evidence": details}
    if final:
        atomic_write_json(output_dir / "preflight.json", baseline._json_safe(report))
        _write_closeout(output_dir / "closeout_report.md", report)
    else:
        atomic_write_json(output_dir / "failure_record.json", baseline._json_safe(report))
    return report


def _write_closeout(path: Path, report: dict[str, Any]) -> None:
    evidence = report["evidence"]
    status = report["status"]
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# LightGen400 q12 trainability preflight\n\n")
        stream.write("## 结果与状态\n\n")
        stream.write(f"状态为 **{status}**，终止阶段为 `{report['terminal_stage']}`。"
                     f"科学训练更新为0；工程 Adam 调用为 {report['engineering_adam_calls']}/12；"
                     "没有执行500步A/B。\n\n")
        stream.write("## 科学认识\n\n")
        stream.write("本结果只评价固定起点/历史终点、q12有限像素RS候选、q16参考和本轮真实batch10更新/恢复路径。"
                     "即使工程验收通过，也不证明400×400架构已达到重建基线、生成合格、光纤硬件耦合或人眼可读；"
                     "原30,000步基线 `VALID_FAILED` 保持不变。\n\n")
        stream.write("## 目标缺口\n\n")
        stream.write("完整TRAIN训练、DEV重建验收、生成验收、实物MFD/传输参数校准和人眼识读评审仍未完成。"
                     "q12只对本轮固定状态、样本和三步工程更新终点作资格判断。\n\n")
        stream.write("## 工程完成与验证\n\n")
        stream.write("实际完成阶段：" + ", ".join(evidence.get("completed_stages", []))
                     + f"；最终阶段为 `{report['terminal_stage']}`。"
                     "独立积分、七平面中心/面积场、q16到q12节点场、OLS、CMOS、梯度、有限差分、"
                     "微批次batch10成本和SIGINT/SIGTERM恢复读数均按阶段保存；"
                     "缺项或未执行检查不作为通过。\n\n")
        stream.write("## 决策影响与下一步\n\n")
        stream.write("只有独立参考收敛、整链与梯度门槛、实际batch10等价、冻结隔离及恢复全部通过，"
                     "才支持另行设计科学A/B。数值问题记为 `NUMERICAL_NOT_QUALIFIED`，实现/恢复问题记为"
                     " `INVALID_HARNESS`，资源问题记为 `PREFLIGHT_BLOCKED_RESOURCE`。"
                     "本轮授权止于本预检，不自动启动500步A/B。\n")


def _device_info(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device), "cuda_available": torch.cuda.is_available()}
    props = torch.cuda.get_device_properties(device)
    return {"device": torch.cuda.get_device_name(device), "cuda_available": True,
            "total_memory_bytes": props.total_memory,
            "free_memory_bytes": torch.cuda.mem_get_info(device)[0],
            "allocated_memory_bytes": torch.cuda.memory_allocated(device),
            "reserved_memory_bytes": torch.cuda.memory_reserved(device),
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda}


def _load_or_make(output_dir: Path, stage: str, action: Any) -> dict[str, Any]:
    prior_value = _read_stage(output_dir, stage)
    if prior_value is not None:
        return prior_value
    return action()


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve(args.output_dir)
    if args.mode == "preflight":
        _require(not output_dir.exists(), f"refusing to overwrite existing q12 evidence directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
    else:
        _require(output_dir.is_dir(), f"explicit q12 resume directory does not exist: {output_dir}")
    device = torch.device("cuda")
    started = time.perf_counter()
    progress: dict[str, Any] = {"status": "RUNNING", "stage": "input_load", "optimizer_updates": 0}
    explicit_inputs = {name: str(_resolve(getattr(args, name)).resolve())
                       for name in ("source_checkpoint", "source_report", "source_start_state",
                                    "source_endpoint_checkpoint", "fixed_population", "config", "data",
                                    "evaluator", "qualification_report", "qualification_cache", "output_dir")}
    completed: set[str] = set()
    prior_state = None
    prior_failure_record = None
    if args.mode == "preflight":
        _require(not (output_dir / "preflight_state.pt").exists(),
                 "preflight state already exists; use explicit --mode resume and its checkpoint")
        _require(not (output_dir / "preflight.json").exists(), "refusing to overwrite a prior q12 report")
        _write_checkpoint(output_dir / "preflight_state.pt", explicit_inputs, completed, "starting")
    else:
        _require(args.resume_checkpoint.is_file(), "explicit q12 resume checkpoint is missing")
        _require(args.resume_checkpoint.resolve() == (output_dir / "preflight_state.pt").resolve(),
                 "resume checkpoint must be this output directory's explicit preflight_state.pt")
        _require(not (output_dir / "preflight.json").exists(), "completed q12 preflight cannot be resumed")
        prior_state = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
        _require(prior_state.get("schema") == "chen_lightgen_400_q12_preflight_state.v1"
                 and prior_state.get("task_id") == TASK_ID
                 and prior_state.get("authorization_id") == AUTHORITY
                 and prior_state.get("explicit_inputs") == explicit_inputs,
                 "explicit q12 checkpoint does not belong to this input set and output path")
        completed = set(prior_state.get("completed_stages", []))
        prior_failure_path = output_dir / "failure_record.json"
        if prior_failure_path.is_file():
            loaded_failure = json.loads(prior_failure_path.read_text(encoding="utf-8"))
            _require(loaded_failure.get("task_id") == TASK_ID
                     and loaded_failure.get("schema") == "chen_lightgen_400_q12_trainability_preflight.v1",
                     "q12 resume failure record belongs to a different task")
            prior_failure_record = {
                "status": loaded_failure.get("status"),
                "terminal_stage": loaded_failure.get("terminal_stage"),
                "evidence": loaded_failure.get("evidence"),
            }
    if not torch.cuda.is_available():
        return _set_terminal(output_dir, "PREFLIGHT_BLOCKED_RESOURCE", "cuda_device_check", started,
                             {"error": "CUDA is unavailable on the required AutoDL execution host",
                              "completed_stages": sorted(completed)}, final=False)
    try:
        context = _load_context(args, device)
    except torch.cuda.OutOfMemoryError as exc:
        return _set_terminal(output_dir, "PREFLIGHT_BLOCKED_RESOURCE", "explicit_input_load", started,
                             {"error": str(exc), "completed_stages": sorted(completed)}, final=False)
    except Exception as exc:
        return _set_terminal(output_dir, "INVALID_HARNESS", "explicit_input_load", started,
                             {"exception_type": type(exc).__name__, "error": str(exc),
                              "completed_stages": sorted(completed)}, final=False)
    context["paths"]["output_dir"] = output_dir
    initial_rng = {"cpu": torch.random.get_rng_state().cpu(),
                   "cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()]}
    progress["resources_at_start"] = _device_info(device)
    with _Heartbeat(output_dir, progress):
        try:
            _require(device.type == "cuda" and torch.cuda.is_available(),
                     "q12 full-model qualification requires CUDA")
            manifest_payload = {
                "schema": "chen_lightgen_400_q12_trainability_manifest.v1",
                "task_id": TASK_ID, "authorization_id": AUTHORITY,
                "explicit_inputs": explicit_inputs, "candidate_order": ORDER,
                "reference_order": REFERENCE_ORDER, "window_cells": WINDOW,
                "pixel_pitch_m": PIXEL_PITCH_M, "backend": BACKEND,
                "fixed_gain_w_inverse": FIXED_GAIN, "source_beta_rec": context["beta"],
                "fixed_fit_rows": context["population"]["fit_rows"].tolist(),
                "fixed_confirmation_rows": context["population"]["confirm_rows"].tolist(),
                "states": list(STATES), "science_updates_authorized": 0,
                "previous_qualification_report_characters": context["qualification_report_characters"],
                "engineering_adam_calls_authorized": 12,
                "optimizer_contract": {"name": "Adam", "lr": 1.0e-3,
                                       "betas": [0.9, 0.999], "eps": 1.0e-8},
            }
            manifest_path = output_dir / "run_manifest.json"
            if manifest_path.is_file():
                existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                _require(existing_manifest == baseline._json_safe(manifest_payload),
                         "existing q12 run manifest differs from explicit resume inputs")
            else:
                _save_json(manifest_path, manifest_payload, new=True)
            resource = _load_or_make(output_dir, "resource_snapshot", lambda: _device_info(device))
            _atomic_stage(output_dir, "resource_snapshot", resource, progress, completed, explicit_inputs)

            progress.update({"stage": "q12_batch10_cost_probe_A"})
            cost_a = _load_or_make(output_dir, "q12_batch10_cost_A", lambda: _cost_probe(
                context["state_phases"][STATES[0]], context["population"], context["pixels"],
                context["beta"], 0.0, "A", device, output_dir))
            _atomic_stage(output_dir, "q12_batch10_cost_A", cost_a, progress, completed, explicit_inputs)
            if cost_a["status"] != "NUMERICALLY_QUALIFIED":
                return _set_terminal(output_dir, "INVALID_HARNESS", "q12_batch10_cost_A", started,
                                     {"cost_A": cost_a, "completed_stages": sorted(completed)}, final=True)

            direct = _load_or_make(output_dir, "independent_direct_rs", lambda: _direct_rs_qualification(device, output_dir, progress))
            _atomic_stage(output_dir, "independent_direct_rs", direct, progress, completed, explicit_inputs)
            if direct["status"] != "NUMERICALLY_QUALIFIED":
                return _set_terminal(output_dir, direct["status"], "independent_direct_rs", started,
                                     {"direct_rs": direct, "completed_stages": sorted(completed)}, final=True)

            boundary = _load_or_make(output_dir, "production_boundary", lambda: prior._production_boundary_checks(ORDER, device))
            _atomic_stage(output_dir, "production_boundary", boundary, progress, completed, explicit_inputs)

            cache_replay = _load_or_make(output_dir, "cache_production_replay", lambda: _validate_cache_replay(context, output_dir, device))
            _atomic_stage(output_dir, "cache_production_replay", cache_replay, progress, completed, explicit_inputs)
            cache_accepted = bool(cache_replay["cache_accepted"])
            if not cache_accepted:
                progress.update({"cache_rejected": True,
                                 "reason": "source q12 cache differs from current corrected production replay; regenerate candidate and reference raws"})
                atomic_write_json(output_dir / "progress.json", progress.copy())

            full_chain = _load_or_make(output_dir, "full_chain_q12_vs_q16", lambda: _fit_endpoint_raws(
                context, output_dir, device, cache_accepted))
            _atomic_stage(output_dir, "full_chain_q12_vs_q16", full_chain, progress, completed, explicit_inputs)
            if full_chain["status"] != "NUMERICALLY_QUALIFIED":
                return _set_terminal(output_dir, full_chain["status"], "full_chain_q12_vs_q16", started,
                                     {"cost_A": cost_a, "direct_rs": direct, "boundary": boundary,
                                      "cache_replay": cache_replay, "full_chain": full_chain,
                                      "completed_stages": sorted(completed)}, final=True)

            mixed_cache_accepted = cache_accepted
            if mixed_cache_accepted:
                try:
                    embedding = _load_or_make(output_dir, "fft_embedding_q12", lambda: prior._fft_embedding_qualification(
                        context["state_phases"], context["population"], context["pixels"], ORDER, device, output_dir))
                except torch.cuda.OutOfMemoryError as exc:
                    return _set_terminal(output_dir, "PREFLIGHT_BLOCKED_RESOURCE", "fft_embedding_q12", started,
                                         {"error": str(exc), "completed_stages": sorted(completed)}, final=True)
            else:
                embedding = _load_or_make(output_dir, "fft_embedding_q12", lambda: prior._fft_embedding_qualification(
                    context["state_phases"], context["population"], context["pixels"], ORDER, device, output_dir))
            _atomic_stage(output_dir, "fft_embedding_q12", embedding, progress, completed, explicit_inputs)
            if embedding["status"] != "NUMERICALLY_QUALIFIED":
                return _set_terminal(output_dir, embedding["status"], "fft_embedding_q12", started,
                                     {"full_chain": full_chain, "fft_embedding": embedding,
                                      "completed_stages": sorted(completed)}, final=True)

            node_gradients = _load_or_make(output_dir, "node_gradient_q12_q16", lambda: _node_gradient_qualification(
                context, output_dir, device, progress))
            _atomic_stage(output_dir, "node_gradient_q12_q16", node_gradients, progress, completed, explicit_inputs)
            if node_gradients["status"] != "NUMERICALLY_QUALIFIED":
                return _set_terminal(output_dir, node_gradients["status"], "node_gradient_q12_q16", started,
                                     {"node_gradients": node_gradients,
                                      "completed_stages": sorted(completed)}, final=True)

            finite_diff = _load_or_make(output_dir, "finite_difference_72", lambda: _finite_difference_qualification(
                context, output_dir, device, progress))
            _atomic_stage(output_dir, "finite_difference_72", finite_diff, progress, completed, explicit_inputs)
            if finite_diff["status"] != "NUMERICALLY_QUALIFIED":
                return _set_terminal(output_dir, finite_diff["status"], "finite_difference_72", started,
                                     {"finite_difference": finite_diff,
                                      "completed_stages": sorted(completed)}, final=True)

            eta_info = _load_or_make(output_dir, "structure_coefficient", lambda: _calibrate_eta_q12(context, device))
            _atomic_stage(output_dir, "structure_coefficient", eta_info, progress, completed, explicit_inputs)
            cost_b = _load_or_make(output_dir, "q12_batch10_cost_B", lambda: _cost_probe(
                context["state_phases"][STATES[0]], context["population"], context["pixels"],
                context["beta"], eta_info["eta"], "B", device, output_dir))
            _atomic_stage(output_dir, "q12_batch10_cost_B", cost_b, progress, completed, explicit_inputs)
            if cost_b["status"] != "NUMERICALLY_QUALIFIED":
                return _set_terminal(output_dir, "INVALID_HARNESS", "q12_batch10_cost_B", started,
                                     {"cost_B": cost_b, "completed_stages": sorted(completed)}, final=True)
            combined_costs = {
                microbatch: cost_a["microbatch_results"][str(microbatch)]["elapsed_seconds"]
                + cost_b["microbatch_results"][str(microbatch)]["elapsed_seconds"]
                for microbatch in (1, 2)
            }
            selected_microbatch = min(combined_costs, key=combined_costs.__getitem__)

            def run_negative_control() -> dict[str, Any]:
                negative_model = prior._model_from_phases(
                    context["state_phases"][STATES[0]], ORDER, WINDOW, device, BACKEND
                )
                try:
                    return prior._pairing_negative_control(
                        negative_model, context["pixels"], context["population"], device
                    )
                finally:
                    del negative_model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            negative = _load_or_make(output_dir, "pairing_negative_control", run_negative_control)
            _atomic_stage(output_dir, "pairing_negative_control", negative, progress, completed, explicit_inputs)

            start_rng = initial_rng
            branch_results: dict[str, Any] = {}
            for branch in ("A", "B"):
                branch_stage = f"engineering_branch_{branch}"
                saved_branch = _read_stage(output_dir, branch_stage)
                if saved_branch is not None:
                    branch_results[branch] = saved_branch
                    continue
                result = _branch_signal_test(context, branch, eta_info["eta"], output_dir,
                                             start_rng, device, progress, selected_microbatch)
                branch_results[branch] = result
                _atomic_stage(output_dir, branch_stage, result, progress, completed, explicit_inputs)
            endpoint_numeric = _load_or_make(output_dir, "engineering_endpoint_numeric", lambda: _endpoint_numeric_qualification(
                context, output_dir, branch_results, device, progress))
            _atomic_stage(output_dir, "engineering_endpoint_numeric", endpoint_numeric, progress, completed, explicit_inputs)
            status = "ENGINEERING_ACCEPTED" if endpoint_numeric["status"] == "NUMERICALLY_QUALIFIED" else endpoint_numeric["status"]
            evidence = {"cost_A": cost_a, "direct_rs": direct, "production_boundary": boundary,
                        "cache_replay": cache_replay, "full_chain_q12_vs_q16": full_chain,
                        "fft_embedding": embedding, "node_gradient": node_gradients,
                        "finite_difference": finite_diff, "structure_coefficient": eta_info,
                        "cost_B": cost_b, "selected_microbatch": selected_microbatch,
                        "pairing_negative_control": negative, "engineering_branches": branch_results,
                        "endpoint_numeric": endpoint_numeric,
                        "resumed_from_failure_record": prior_failure_record,
                        "completed_stages": sorted(completed),
                        "resources_at_end": _device_info(device),
                        "engineering_adam_calls": sum(item["engineering_adam_calls"] for item in branch_results.values())}
            return _set_terminal(output_dir, status, "engineering_endpoint_numeric", started, evidence, final=True)
        except torch.cuda.OutOfMemoryError as exc:
            return _set_terminal(output_dir, "PREFLIGHT_BLOCKED_RESOURCE", progress.get("stage", "unknown"),
                                 started, {"error": str(exc), "progress": progress,
                                           "completed_stages": sorted(completed)}, final=False)
        except HarnessCheckError as exc:
            return _set_terminal(output_dir, "INVALID_HARNESS", progress.get("stage", "unknown"),
                                 started, {"error": str(exc), "progress": progress,
                                           "completed_stages": sorted(completed)}, final=True)
        except Exception as exc:
            atomic_write_json(output_dir / "execution_error.json", {
                "status": "INVALID_HARNESS", "stage": progress.get("stage"),
                "exception_type": type(exc).__name__, "error": str(exc),
                "completed_stages": sorted(completed), "time": time.time()})
            raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "resume"), required=True)
    parser.add_argument("--propagation-backend", choices=("rs_linear",), required=True)
    for name in ("source-checkpoint", "source-report", "source-start-state",
                 "source-endpoint-checkpoint", "fixed-population", "config", "data", "evaluator",
                 "qualification-report", "qualification-cache", "output-dir", "resume-checkpoint"):
        parser.add_argument(f"--{name}", type=Path, required=name != "resume-checkpoint")
    args = parser.parse_args(argv)
    if args.mode == "preflight":
        _require(args.resume_checkpoint is None, "--resume-checkpoint is valid only with --mode resume")
    else:
        _require(args.resume_checkpoint is not None, "resume mode requires an explicit --resume-checkpoint")
    return args


def main() -> None:
    args = _parse_args()
    _require(args.propagation_backend == BACKEND, "only rs_linear is authorized for q12 trainability")
    result = _preflight(args)
    print(json.dumps(baseline._json_safe(result), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()

