"""Close the LightGen400 subpixel-sampling and camera-power audit without updates."""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    FIBER_MFD_M,
    FIBER_PITCH_M,
    LAYER_SPACING_M,
    OLS_PORT_COUNT,
    PIXEL_PITCH_M,
    SOURCE_POWER_W,
    WAVELENGTH_M,
    ChenLightGen400Baseline,
    LightGen400Config,
    integrated_power,
    integrate_camera_pixel_intensity,
)
from ldm_project.v2.application.research.lightgen.gaussian_fiber_baseline import (
    GaussianFiberArrayOLS,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    ResearchRunControl,
    ResearchRunIdentity,
    capture_execution_source_manifest,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.evidence import atomic_torch_save, atomic_torch_save_new, atomic_write_json, atomic_write_new_json
from ldm_project.v2.optics.propagation import (
    angular_spectrum_propagate,
    clear_production_transfer_cache,
    reference_angular_spectrum_propagate,
)

from scripts.research.lightgen import diagnose_chen_lightgen_400_quickcheck_v1 as quickcheck
from scripts.research.lightgen import execute_chen_lightgen_400_mnist_baseline_v1 as baseline


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CHEN_LIGHTGEN_400_SAMPLING_CLOSURE_V1"
AUTHORITY = "USER_APPROVED_CHEN_LIGHTGEN_400_SAMPLING_CLOSURE_20261017"
MODULE_NAME = "scripts.research.lightgen.diagnose_chen_lightgen_400_sampling_closure_v1"
FACTORS = (1, 2, 4)
GRIDS = {1: 4096, 2: 8192, 4: 16384}
WINDOW_TOLERANCE = 1.0e-4
COEFFICIENT_TOLERANCE = 1.0e-4
GRADIENT_TOLERANCE = 1.0e-4
FD_TOLERANCE = 1.0e-2
RS_FD_TOLERANCE = 1.0e-6
FD_STEPS_RAD = (2.5e-3, 1.25e-3)
GAIN = 3.17684599058831e10
ROI_M = 1.5e-3
STATE_START = "start"
STATE_END = "endpoint_500"
STATES = (STATE_START, STATE_END)
MEASUREMENT_BLOCK_SCHEMA = "chen_lightgen_400_sampling_measurement_block.v2"
DECODER_PROBE_SCHEMA = "chen_lightgen_400_decoder_sampling_probe.v2"
FIELD_VIEWS = ("physical_pixel_center", "physical_pixel_average")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _ready_correctness_group(checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    passed = bool(checks) and all(item.get("passed") is True for item in checks.values())
    _require(passed, "one or more executed production preflight checks did not pass")
    return {"passed": passed, "checks": checks}


def _module_command(*arguments: str) -> list[str]:
    return [sys.executable, "-m", MODULE_NAME, *arguments]


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object at {path}")
    return value


def _save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write an empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _start_heartbeat(output: Path, status: str, device: torch.device) -> tuple[threading.Event, threading.Thread]:
    stop = threading.Event()
    started = time.monotonic()

    def update() -> None:
        while not stop.wait(60.0):
            payload: dict[str, Any] = {"status": status, "updated_at_utc": time.time(),
                                       "elapsed_seconds": time.monotonic() - started}
            if device.type == "cuda" and torch.cuda.is_available():
                payload["cuda_memory_allocated_bytes"] = torch.cuda.memory_allocated(device)
                payload["cuda_memory_reserved_bytes"] = torch.cuda.memory_reserved(device)
            progress_path = output / "progress.json"
            if progress_path.is_file():
                payload["progress"] = _json(progress_path)
            atomic_write_json(output / "heartbeat.json", payload)

    thread = threading.Thread(target=update, name="lightgen-sampling-heartbeat", daemon=True)
    thread.start()
    return stop, thread


def _stop_heartbeat(handle: tuple[threading.Event, threading.Thread]) -> None:
    stop, thread = handle
    stop.set()
    thread.join()


def _selected_rows(population: dict[str, torch.Tensor], labels: torch.Tensor,
                   train_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = population["confirm_rows_by_class"][:, 0].long().cpu()
    _require(rows.shape == (10,) and rows.unique().numel() == 10,
             "fixed population must provide exactly the first confirmation row for each class")
    train_mask = torch.zeros(labels.shape, dtype=torch.bool)
    train_mask[train_indices.cpu()] = True
    _require(bool(train_mask[rows].all()), "sampling audit rows must all belong to canonical TRAIN")
    expected = torch.arange(10, dtype=torch.long)
    observed = labels.index_select(0, rows)
    _require(torch.equal(observed, expected), "fixed confirmation row labels do not match classes 0..9")
    return rows, observed


def _load_model_phases(start_path: Path, endpoint_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    _require(start_path.is_file() and endpoint_path.is_file(), "explicit start or endpoint state is missing")
    start = torch.load(start_path, map_location="cpu", weights_only=False)
    _require(set(start) == {"encoder_phase", "decoder_phase", "ols.modes_y", "ols.modes_x", "ols.centers_m", "ols.transport"},
             "start model state has an unexpected structure")
    end = torch.load(endpoint_path, map_location="cpu", weights_only=False)
    _require(end.get("schema") == "chen_lightgen_400_joint_reconstruction_checkpoint.v1"
             and end.get("completed_updates") == 500 and end.get("grid") == 4096
             and end.get("source_checkpoint_step") == 30_000,
             "endpoint checkpoint must be the explicit 500-update 4096-grid joint endpoint")
    _require("optimizer" in end and bool(end["optimizer"].get("state", {})),
             "endpoint checkpoint optimizer history is absent")
    start_phases = {key: start[key].clone() for key in ("encoder_phase", "decoder_phase")}
    end_state = end["model"]
    _require(set(start_phases) <= set(end_state), "endpoint model phase parameters are incomplete")
    end_phases = {key: end_state[key].detach().cpu().clone() for key in start_phases}
    for state, phases in ((STATE_START, start_phases), (STATE_END, end_phases)):
        _require(phases["encoder_phase"].shape == (2, 1, 400, 400)
                 and phases["decoder_phase"].shape == (4, 1, 400, 400),
                 f"{state} must have two encoder and four decoder 400x400 phase planes")
        _require(all(torch.isfinite(value).all() for value in phases.values()), f"{state} has nonfinite phases")
    return {STATE_START: start_phases, STATE_END: end_phases}


def _load_context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    for name in ("baseline_checkpoint", "baseline_report", "joint_report", "source_manifest",
                 "fixed_population", "data", "config", "start_state", "endpoint_checkpoint"):
        path = _resolve(getattr(args, name))
        _require(path.is_file(), f"explicit {name} is missing: {path}")
    config, source_checkpoint, source_report, template, references = quickcheck._load_inputs(
        _resolve(args.baseline_checkpoint), _resolve(args.baseline_report), _resolve(args.config), device
    )
    _require(source_checkpoint.get("completed_updates") == 30_000,
             "baseline source checkpoint is not the explicit 30,000-update endpoint")
    _require(source_report.get("status") == "VALID_FAILED", "source baseline identity/status changed")
    source_config = source_checkpoint["config"]
    optics = source_config["optics"]
    _require(optics["encoder_layers"] == 2 and optics["decoder_layers"] == 4
             and optics["active_shape"] == [400, 400] and optics["pixel_pitch_m"] == PIXEL_PITCH_M
             and optics["wavelength_m"] == WAVELENGTH_M
             and optics["adjacent_layer_spacing_m"] == LAYER_SPACING_M
             and optics["input_power_w"] == SOURCE_POWER_W
             and optics["layer_power_transmission"] == 0.889
             and optics["ols"]["ports"] == OLS_PORT_COUNT
             and optics["ols"]["mode_field_diameter_m"] == FIBER_MFD_M
             and optics["ols"]["port_pitch_m"] == FIBER_PITCH_M
             and optics["ols"]["transport"] == "ideal_identity",
             "sampling model geometry or power entrance differs from the fixed source configuration")
    joint_report = _json(_resolve(args.joint_report))
    _require(joint_report.get("source_checkpoint_step") == 30_000
             and joint_report.get("status") == "VALID_FAILED",
             "joint endpoint report does not identify the fixed 30,000-step source")
    manifest = _json(_resolve(args.source_manifest))
    _require(manifest.get("grid") == 4096 and manifest.get("optimizer_updates") == 0,
             "source crossdecode manifest is not the explicit frozen 4096 audit")
    phases = _load_model_phases(_resolve(args.start_state), _resolve(args.endpoint_checkpoint))
    source_model = source_checkpoint["model"]
    _require(all(torch.equal(phases[STATE_START][key], source_model[key])
                 for key in ("encoder_phase", "decoder_phase")),
             "joint reconstruction start phases differ from the explicit 30,000-update source")
    endpoint = torch.load(_resolve(args.endpoint_checkpoint), map_location="cpu", weights_only=False)
    saved_population = torch.load(_resolve(args.fixed_population), map_location="cpu", weights_only=False)
    _require(torch.equal(endpoint["fit_rows"], saved_population["fit_rows"])
             and torch.equal(endpoint["confirm_rows"], saved_population["confirm_rows"]),
             "endpoint and fixed-population rows differ")
    pixels, labels, train_indices, _ = baseline._load_data(_resolve(args.data))
    population = quickcheck._load_fixed_population(_resolve(args.fixed_population), labels, train_indices)
    rows, selected_labels = _selected_rows(population, labels, train_indices)
    beta = float(source_checkpoint["bogt_weights"]["rec"])
    _require(math.isfinite(beta) and beta > 0.0, "frozen source reconstruction weight is invalid")
    models: dict[tuple[str, int], ChenLightGen400Baseline] = {}
    for state in STATES:
        for factor in FACTORS:
            model = ChenLightGen400Baseline(LightGen400Config(GRIDS[factor], True, factor)).to(device)
            with torch.no_grad():
                model.encoder_phase.copy_(phases[state]["encoder_phase"].to(device))
                model.decoder_phase.copy_(phases[state]["decoder_phase"].to(device))
            model.eval()
            model.requires_grad_(False)
            _require(all(not parameter.requires_grad for parameter in model.parameters()),
                     f"{state} factor-{factor} model must be frozen")
            models[(state, factor)] = model
    return {
        "args": args, "config": config, "source_checkpoint": source_checkpoint,
        "source_report": source_report, "joint_report": joint_report, "manifest": manifest,
        "references": references, "phases": phases, "population": population,
        "pixels": pixels, "labels": labels, "train_indices": train_indices,
        "rows": rows, "selected_labels": selected_labels, "beta": beta,
        "models": models, "device": device,
    }


def _spectral_physical_pixel_observables(
    field: torch.Tensor, factor: int, *, roi_m: float = ROI_M,
    detach: bool = True, only_center: bool = False,
) -> dict[str, torch.Tensor]:
    """Evaluate a sampled complex field at common physical pixel coordinates.

    The input grid is periodic over the model's full physical window. A Fourier
    shift maps its subpixel-centred samples to each 3 um pixel centre. The box
    average uses the same shift followed by the exact rectangular-aperture sinc
    transfer function. Camera intensity integration remains a separate path.
    """
    _require(factor in FACTORS, "unsupported physical phase-pixel sampling factor")
    _require(field.ndim >= 2 and field.shape[-1] == field.shape[-2],
             "field observation requires a square full-window complex field")
    _require(field.is_complex() and torch.isfinite(field).all(),
             "field observation requires finite complex values")
    height = int(field.shape[-1])
    pitch = PIXEL_PITCH_M / factor
    roi_pixels_float = roi_m / PIXEL_PITCH_M
    roi_pixels = int(round(roi_pixels_float))
    _require(roi_pixels > 0 and abs(roi_pixels - roi_pixels_float) <= 1.0e-9,
             "field-comparison ROI must contain an integer number of physical pixels")
    roi_fine = roi_pixels * factor
    _require(height >= roi_fine and (height - roi_fine) % 2 == 0,
             "physical ROI must be a centred integer crop of the full model window")
    top = (height - roi_fine) // 2
    shift_m = (factor - 1) * pitch / 2.0
    spectrum = torch.fft.fft2(field)
    frequencies = torch.fft.fftfreq(height, d=pitch, device=field.device, dtype=torch.float64)
    rows_per_chunk = max(1, min(256, 16_777_216 // height))
    if shift_m:
        for start in range(0, height, rows_per_chunk):
            stop = min(start + rows_per_chunk, height)
            phase = (2.0 * math.pi * shift_m) * (
                frequencies[start:stop, None] + frequencies[None, :]
            )
            spectrum[start:stop].mul_(torch.polar(torch.ones_like(phase), phase).to(spectrum.dtype))

    indices = top + torch.arange(roi_pixels, device=field.device, dtype=torch.long) * factor
    center_full = torch.fft.ifft2(spectrum)
    center = center_full.index_select(-2, indices).index_select(-1, indices)
    del center_full
    if detach:
        center = center.detach().cpu()
    if only_center:
        return {"physical_pixel_center": center}

    box = torch.sinc(frequencies * PIXEL_PITCH_M)
    box = box.to(dtype=field.real.dtype)
    for start in range(0, height, rows_per_chunk):
        stop = min(start + rows_per_chunk, height)
        spectrum[start:stop].mul_(box[start:stop, None] * box[None, :])
    average_full = torch.fft.ifft2(spectrum)
    average = average_full.index_select(-2, indices).index_select(-1, indices)
    if detach:
        average = average.detach().cpu()
    del average_full, spectrum
    return {"physical_pixel_center": center, "physical_pixel_average": average}


def _record_field(fields: dict[str, dict[str, torch.Tensor]], factor: int,
                  name: str, field: ComplexField) -> None:
    _require(field.tensor.shape[0] == 1 and field.tensor.shape[1] == 1,
             f"field comparison expects one sample/channel at {name}")
    values = _spectral_physical_pixel_observables(field.tensor[0, 0], factor)
    _require(all(torch.isfinite(value).all() for value in values.values()),
             f"nonfinite physical field in sampling comparison at {name}")
    fields[name] = values


def _analytic_physical_pixel_observer_check() -> dict[str, Any]:
    period_m = 8.0e-6
    roi_pixels = 32
    roi_m = roi_pixels * PIXEL_PITCH_M
    new_views: dict[int, dict[str, torch.Tensor]] = {}
    legacy_views: dict[int, dict[str, torch.Tensor]] = {}
    analytic_errors: dict[str, float] = {}
    for factor in FACTORS:
        count = 128 * factor
        pitch = PIXEL_PITCH_M / factor
        axis = (torch.arange(count, dtype=torch.float64) - count / 2 + 0.5) * pitch
        field = torch.exp(2j * math.pi * axis[None, :] / period_m).expand(count, -1).to(torch.complex64)
        views = _spectral_physical_pixel_observables(field, factor, roi_m=roi_m)
        new_views[factor] = views
        centers = (torch.arange(roi_pixels, dtype=torch.float64) - roi_pixels / 2 + 0.5) * PIXEL_PITCH_M
        expected_line = torch.exp(2j * math.pi * centers / period_m).to(torch.complex64)
        expected_center = expected_line[None, :].expand(roi_pixels, -1)
        expected_average = expected_center * (torch.sinc(torch.tensor(
            PIXEL_PITCH_M / period_m, dtype=torch.float64
        ))).to(torch.complex64)
        for name, expected in (("physical_pixel_center", expected_center),
                               ("physical_pixel_average", expected_average)):
            analytic_errors[f"factor_{factor}_{name}"] = float(
                torch.linalg.vector_norm(views[name] - expected) /
                torch.linalg.vector_norm(expected).clamp_min(1.0e-30)
            )
        roi_fine = roi_pixels * factor
        top = (count - roi_fine) // 2
        roi = field[top:top + roi_fine, top:top + roi_fine]
        shaped = roi.reshape(roi_pixels, factor, roi_pixels, factor)
        average = shaped.mean(dim=(1, 3))
        if factor > 1:
            lo = factor // 2 - 1
            center = shaped[:, lo:lo + 2, :, lo:lo + 2].mean(dim=(1, 3))
        else:
            center = shaped
        legacy_views[factor] = {"center": center, "average": average}
    new_errors = {
        f"factor_{factor}_vs_4_{view}": float(
            torch.linalg.vector_norm(new_views[factor][view] - new_views[4][view]) /
            torch.linalg.vector_norm(new_views[4][view]).clamp_min(1.0e-30)
        )
        for factor in (1, 2) for view in FIELD_VIEWS
    }
    old_factor2_center = float(
        torch.linalg.vector_norm(legacy_views[2]["center"] - legacy_views[4]["center"]) /
        torch.linalg.vector_norm(legacy_views[4]["center"]).clamp_min(1.0e-30)
    )
    old_factor1_average = float(
        torch.linalg.vector_norm(legacy_views[1]["average"] - legacy_views[4]["average"]) /
        torch.linalg.vector_norm(legacy_views[4]["average"]).clamp_min(1.0e-30)
    )
    passed = max(analytic_errors.values()) <= WINDOW_TOLERANCE and max(new_errors.values()) <= WINDOW_TOLERANCE
    return {
        "passed": passed,
        "periodic_plane_wave_period_m": period_m,
        "roi_pixels": roi_pixels,
        "max_analytic_error": max(analytic_errors.values()),
        "analytic_errors": analytic_errors,
        "same_observable_cross_factor_errors": new_errors,
        "legacy_comparator_counterexample": {
            "factor_2_vs_4_center_relative_l2": old_factor2_center,
            "factor_1_vs_4_area_relative_l2": old_factor1_average,
        },
        "tolerance": WINDOW_TOLERANCE,
    }


def _one_forward(ctx: dict[str, Any], state: str, factor: int, row: int,
                 *, grad: bool = False,
                 model_override: ChenLightGen400Baseline | None = None,
                 on_first_field: Any | None = None,
                 capture_fields: bool = True) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    model: ChenLightGen400Baseline = model_override or ctx["models"][(state, factor)]
    device: torch.device = ctx["device"]
    target, _ = baseline._image_batch(ctx["pixels"], torch.tensor([row]), device)
    fields: dict[str, dict[str, torch.Tensor]] = {}
    first_field = [True]

    def callback(name: str, value: ComplexField) -> None:
        if capture_fields:
            _record_field(fields, factor, name, value)
        if first_field[0] and on_first_field is not None:
            on_first_field(name)
            first_field[0] = False
    model.eval()
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        result = model.forward_from_image(target, field_callback=callback)
        task = result["physical_w_per_pixel"] * GAIN
        loss = (task - target).square().mean()
    label = int(ctx["labels"][row])
    coefficients = result["ols_coefficients"][0].detach().cpu()
    port_power = coefficients.abs().square().double().cpu()
    record = {
        "state": state, "factor": factor, "grid": model.config.propagation_grid_size, "row": row,
        "label": label, "gain_w_inverse": GAIN,
        "input_power_w": float(result["input_power_w"][0].detach().cpu()),
        "encoder_power_1_w": float(result["plane_powers_w"][0, 0].detach().cpu()),
        "encoder_power_2_w": float(result["plane_powers_w"][0, 1].detach().cpu()),
        "ols_incident_power_w": float(result["ols_input_power_w"][0].detach().cpu()),
        "ols_reemitted_power_w": float(result["ols_output_power_w"][0].detach().cpu()),
        "modal_power_w": float(port_power.sum()),
        "decoder_entry_power_w": float(result["plane_powers_w"][0, 2].detach().cpu()),
        "decoder_1_power_w": float(result["plane_powers_w"][0, 3].detach().cpu()),
        "decoder_2_power_w": float(result["plane_powers_w"][0, 4].detach().cpu()),
        "decoder_3_power_w": float(result["plane_powers_w"][0, 5].detach().cpu()),
        "decoder_4_power_w": float(result["plane_powers_w"][0, 6].detach().cpu()),
        "camera_roi_power_w": float(result["camera_roi_power_w"][0].detach().cpu()),
        "camera_full_field_power_w": float(result["camera_full_field_power_w"][0].detach().cpu()),
        "camera_outside_roi_power_w": float(result["camera_outside_roi_power_w"][0].detach().cpu()),
        "mse400": float((task - target).square().mean().detach().cpu()),
        "psnr28_db": float((metrics := baseline._reconstruction_batch_metrics(
            task, target, torch.tensor([row]), torch.tensor([row])
        ))[1][0].detach().cpu()),
        "ssim28": float(metrics[2][0].detach().cpu()),
        "saturation_fraction": float((((task < 0.0) | (task > 1.0)).double().mean()).detach().cpu()),
    }
    prediction = task.detach().cpu()
    target_cpu = target.detach().cpu()
    physical_cpu = result["physical_w_per_pixel"].detach().cpu()
    display_cpu = prediction.clamp(0.0, 1.0)
    _require(torch.isfinite(task).all() and torch.isfinite(coefficients).all()
             and torch.isfinite(result["plane_powers_w"]).all(),
             f"nonfinite production output at {state}, factor {factor}, row {row}")
    del result, target, task
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"metrics": record, "coefficients": coefficients, "port_power_w": port_power,
            "physical_w_per_pixel": physical_cpu, "task_intensity_untruncated": prediction,
            "display_intensity_clipped": display_cpu, "target": target_cpu}, fields, loss


def _align_coefficients(candidate: torch.Tensor, reference: torch.Tensor) -> tuple[float, torch.Tensor]:
    a = candidate.to(torch.complex128).reshape(-1)
    b = reference.to(torch.complex128).reshape(-1)
    overlap = torch.vdot(b, a)
    phase = torch.angle(overlap)
    aligned = a * torch.exp(-1j * phase)
    error = torch.linalg.vector_norm(aligned - b) / torch.linalg.vector_norm(b).clamp_min(1.0e-300)
    return float(error), torch.exp(-1j * phase)


def _compare_sample(variants: dict[int, dict[str, Any]], field_variants: dict[int, dict[str, Any]]) -> dict[str, Any]:
    records = {factor: variants[factor] for factor in FACTORS}
    field_checks: dict[str, Any] = {}
    for sampling_view in FIELD_VIEWS:
        reference_fields = {name: values[sampling_view] for name, values in field_variants[4].items()}
        for factor in (1, 2):
            candidate_fields = {name: values[sampling_view] for name, values in field_variants[factor].items()}
            errors, phase = baseline._complex_relative_errors(candidate_fields, reference_fields)
            field_checks[f"factor_{factor}_vs_4_{sampling_view}"] = {
                "global_phase_rad": phase, "per_plane_relative_l2": errors,
                "max_relative_l2": max(errors.values()),
            }
    coefficient_checks = {}
    for factor in (1, 2):
        error, phase = _align_coefficients(records[factor]["coefficients"], records[4]["coefficients"])
        candidate_power = records[factor]["port_power_w"]
        reference_power = records[4]["port_power_w"]
        coefficient_checks[f"factor_{factor}_vs_4"] = {
            "complex_relative_l2_after_global_phase": error,
            "global_phase_rad": float(torch.angle(phase.conj())),
            "port_power_absolute_error_max_w": float((candidate_power - reference_power).abs().max()),
            "total_power_absolute_error_w": abs(float(candidate_power.sum() - reference_power.sum())),
        }
    image_checks = {}
    endpoint = records[4]
    for factor in (1, 2):
        candidate = records[factor]
        image_checks[f"factor_{factor}_vs_4"] = {
            "physical_output_relative_l2": float(torch.linalg.vector_norm(
                candidate["physical_w_per_pixel"] - endpoint["physical_w_per_pixel"]
            ) / torch.linalg.vector_norm(endpoint["physical_w_per_pixel"]).clamp_min(1.0e-300)),
            "mse400_delta": candidate["metrics"]["mse400"] - endpoint["metrics"]["mse400"],
            "psnr28_db_delta": candidate["metrics"]["psnr28_db"] - endpoint["metrics"]["psnr28_db"],
            "ssim28_delta": candidate["metrics"]["ssim28"] - endpoint["metrics"]["ssim28"],
            "saturation_fraction_delta": candidate["metrics"]["saturation_fraction"] - endpoint["metrics"]["saturation_fraction"],
        }
    return {"field_checks": field_checks, "coefficient_checks": coefficient_checks,
            "camera_task_checks": image_checks}


def _window_expansion_checks(ctx: dict[str, Any], rows: dict[int, int]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for state in STATES:
        large = ChenLightGen400Baseline(LightGen400Config(12_288, True, 2)).to(ctx["device"])
        with torch.no_grad():
            large.encoder_phase.copy_(ctx["phases"][state]["encoder_phase"].to(ctx["device"]))
            large.decoder_phase.copy_(ctx["phases"][state]["decoder_phase"].to(ctx["device"]))
        large.eval().requires_grad_(False)
        for label, row in rows.items():
            small_record, small_fields, _ = _one_forward(ctx, state, 2, row)
            large_record, large_fields, _ = _one_forward(ctx, state, 2, row, model_override=large)
            field_checks = {}
            for view in FIELD_VIEWS:
                left = {name: parts[view] for name, parts in small_fields.items()}
                right = {name: parts[view] for name, parts in large_fields.items()}
                errors, phase = baseline._complex_relative_errors(left, right)
                field_checks[view] = {"per_plane_relative_l2": errors,
                                      "max_relative_l2": max(errors.values()),
                                      "global_phase_rad": phase}
            coefficient_error, phase = _align_coefficients(
                small_record["coefficients"], large_record["coefficients"]
            )
            passed = max(item["max_relative_l2"] for item in field_checks.values()) <= WINDOW_TOLERANCE \
                and coefficient_error <= COEFFICIENT_TOLERANCE
            records[f"{state}_class_{label}"] = {
                "row": row, "label": int(ctx["labels"][row]), "small_grid": 8192,
                "large_grid": 12_288, "field_checks": field_checks,
                "coefficient_relative_l2": coefficient_error,
                "coefficient_alignment_phase_rad": float(torch.angle(phase.conj())),
                "modal_power_absolute_error_w": float((small_record["port_power_w"]
                                                       - large_record["port_power_w"]).abs().max()),
                "passed": passed,
            }
        del large
        clear_production_transfer_cache()
    return {"required": True, "passed": all(item["passed"] for item in records.values()),
            "grid_tolerance": WINDOW_TOLERANCE, "coefficient_tolerance": COEFFICIENT_TOLERANCE,
            "records": records}


def _complex_quadrature_reference() -> dict[str, float | bool]:
    # A subpixel phase ramp is the counterexample to |mean(E)|^2 == mean(|E|^2).
    phase = torch.tensor([-0.6, -0.2, 0.2, 0.6], dtype=torch.float64).reshape(1, 1, 2, 2)
    field = torch.polar(torch.ones_like(phase), phase)
    intensity_integral = integrate_camera_pixel_intensity(field.abs().square(), 2).mean()
    mean_field_square = field.mean().abs().square()
    _require(float(intensity_integral) == 1.0 and float(mean_field_square) < 1.0,
             "CMOS subpixel integration reference did not distinguish coherent averaging from intensity integration")
    return {"mean_intensity": float(intensity_integral), "squared_mean_field": float(mean_field_square),
            "distinguished": True}


def _propagation_reference_check() -> dict[str, Any]:
    n = 256
    pitch = 3.0e-6
    plane = PlaneSpec("plane_wave_reference", (n, n), (pitch, pitch), WAVELENGTH_M, 0.0, "sampling_audit")
    y = torch.arange(n, dtype=torch.float64)[:, None]
    x = torch.arange(n, dtype=torch.float64)[None, :]
    ky, kx = 7, 11
    base = torch.exp(2j * math.pi * (ky * y / n + kx * x / n))
    theta = torch.tensor(0.23, dtype=torch.float64, requires_grad=True)
    distance = 3.0e-3
    fy, fx = ky / (n * pitch), kx / (n * pitch)
    phase_per_m = 2.0 * math.pi / WAVELENGTH_M * math.sqrt(
        1.0 - (WAVELENGTH_M * fx) ** 2 - (WAVELENGTH_M * fy) ** 2
    )
    expected_phase = float(theta.detach()) + phase_per_m * distance
    source = base * torch.exp(1j * theta)
    field = ComplexField(source.unsqueeze(0).unsqueeze(0), plane,
                         OperationProvenance("analytic_plane_wave", None, None))
    candidate = angular_spectrum_propagate(field, distance, "candidate").tensor
    reference = reference_angular_spectrum_propagate(field, distance, "independent").tensor
    expected = base * torch.exp(torch.tensor(1j * expected_phase, dtype=torch.complex128))
    value_error = float(torch.linalg.vector_norm(candidate[0, 0] - expected) / torch.linalg.vector_norm(expected))
    independent_error = float(torch.linalg.vector_norm(candidate - reference) / torch.linalg.vector_norm(reference))
    scalar = candidate[0, 0, 0, 0].real
    automatic = float(torch.autograd.grad(scalar, theta)[0])
    derivative = -math.sin(expected_phase)
    _require(value_error <= 1.0e-10 and independent_error <= 1.0e-10
             and abs(automatic - derivative) <= 1.0e-9,
             "chunked production propagation failed analytic plane-wave value or derivative reference")
    return {"analytic_relative_l2": value_error, "independent_reference_relative_l2": independent_error,
            "autograd_derivative": automatic, "analytic_derivative": derivative, "passed": True}


def _rayleigh_sommerfeld_pixel_integral(pattern: np.ndarray, order: int) -> tuple[np.ndarray, complex]:
    """Independent real-space RS integral over the 32x32 piecewise-constant pixels."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    pitch = PIXEL_PITCH_M
    centers = (np.arange(32, dtype=np.float64) - 16.0 + 0.5) * pitch
    offsets = nodes * pitch / 2.0
    offset_weights = weights * pitch / 2.0
    ox, oy = np.meshgrid(offsets, offsets, indexing="xy")
    wx, wy = np.meshgrid(offset_weights, offset_weights, indexing="xy")
    cell_x, cell_y = np.meshgrid(centers, centers, indexing="xy")
    source_x = (cell_x.reshape(-1, 1) + ox.reshape(1, -1)).reshape(-1)
    source_y = (cell_y.reshape(-1, 1) + oy.reshape(1, -1)).reshape(-1)
    source_weight = np.tile((wx * wy).reshape(-1), 32 * 32)
    source_value = np.repeat(pattern.reshape(-1), order * order)
    selected = np.arange(2, 32, 6, dtype=np.int64)
    target_x, target_y = np.meshgrid(centers[selected], centers[selected], indexing="xy")
    tx, ty = target_x.reshape(-1), target_y.reshape(-1)
    z = 3.0e-3
    wave_number = 2.0 * math.pi / WAVELENGTH_M
    values = np.empty(tx.shape, dtype=np.complex128)
    derivative_at_center = 0.0j
    center_pixel = 16 * 32 + 16
    center_slice = slice(center_pixel * order * order, (center_pixel + 1) * order * order)
    for index, (target_x_m, target_y_m) in enumerate(zip(tx, ty)):
        dx = target_x_m - source_x
        dy = target_y_m - source_y
        radius = np.sqrt(z * z + dx * dx + dy * dy)
        kernel = (z / radius**2) * (
            1.0 / (2.0 * math.pi * radius) + 1.0 / (1j * WAVELENGTH_M)
        ) * np.exp(1j * wave_number * radius)
        integrand = kernel * source_value * source_weight
        values[index] = integrand.sum(dtype=np.complex128)
        if index == len(tx) // 2:
            derivative_at_center = 1j * integrand[center_slice].sum(dtype=np.complex128)
    return values, derivative_at_center


def _rayleigh_sommerfeld_pixel_reference_check(
    device: torch.device | None = None, *, compare_production: bool = False,
) -> dict[str, Any]:
    """Check a converged finite-pixel RS integral, optionally against production ASM."""
    device = device or torch.device("cpu")
    pixel_count, factor, grid, width = 32, 4, 16_384, 128
    pitch = PIXEL_PITCH_M
    centers = (torch.arange(pixel_count, dtype=torch.float64) - 16.0 + 0.5) * pitch
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    amplitude = torch.exp(-(xx.square() + yy.square()) / (2.0 * (18.0e-6) ** 2))
    indices = torch.arange(pixel_count, dtype=torch.float64)
    iy, ix = torch.meshgrid(indices, indices, indexing="ij")
    phase_cases = {
        "gaussian": torch.zeros_like(amplitude),
        "tilted": 2.0 * math.pi * (3.0 * xx - 2.0 * yy) / (pixel_count * pitch),
        "checkerboard": math.pi * torch.remainder(ix + iy, 2.0),
    }
    sample_indices = np.arange(2, 32, 6, dtype=np.int64)
    physical_window_m = grid * pitch / factor
    _require(math.isclose(physical_window_m, 4096 * pitch, rel_tol=0.0, abs_tol=1.0e-15),
             "RS comparison grid must match the production physical window")
    if compare_production:
        _require(device.type == "cuda", "full-window finite-pixel RS comparison requires CUDA")
    grid_plane = PlaneSpec("finite_pixel_rs_reference_grid", (grid, grid),
                           (pitch / factor, pitch / factor), WAVELENGTH_M, 0.0,
                           "lightgen400_rs_reference")
    observations: dict[str, Any] = {}
    for name, phase in phase_cases.items():
        pixel_field = amplitude.to(torch.complex128) * torch.exp(1j * phase)
        pattern = pixel_field.detach().cpu().numpy()
        reference_by_order = {}
        derivatives = {}
        for order in (4, 8, 12):
            reference_by_order[order], derivatives[order] = _rayleigh_sommerfeld_pixel_integral(pattern, order)
        reference = reference_by_order[12]
        quadrature_4_to_8 = float(np.linalg.norm(reference_by_order[4] - reference_by_order[8]) /
                                  max(np.linalg.norm(reference_by_order[8]), 1.0e-300))
        quadrature_8_to_12 = float(np.linalg.norm(reference_by_order[8] - reference_by_order[12]) /
                                   max(np.linalg.norm(reference), 1.0e-300))
        analytic_direction = float(derivatives[12].real)
        epsilon = 1.0e-4
        base_at_center = reference[len(reference) // 2]
        contribution = derivatives[12] / 1j
        unperturbed_remainder = base_at_center - contribution
        rs_finite_difference = float((
            (unperturbed_remainder + contribution * np.exp(1j * epsilon)).real
            - (unperturbed_remainder + contribution * np.exp(-1j * epsilon)).real
        ) / (2.0 * epsilon))
        finite_difference_relative = abs(rs_finite_difference - analytic_direction) / max(abs(analytic_direction), 1.0e-14)
        production: dict[str, Any] = {}
        if compare_production:
            top = (grid - width) // 2
            theta = torch.tensor(0.0, dtype=torch.float64, device=device, requires_grad=True)
            cell_mask = torch.zeros_like(amplitude, device=device)
            cell_mask[16, 16] = 1.0
            perturbed = amplitude.to(device) * torch.exp(
                1j * (phase.to(device) + theta * cell_mask)
            )
            perturbed_grid = perturbed.repeat_interleave(factor, dim=0).repeat_interleave(factor, dim=1)
            padded_grad = torch.zeros((grid, grid), dtype=torch.complex128, device=device)
            padded_grad[top:top + width, top:top + width] = perturbed_grid
            field = ComplexField(
                padded_grad.unsqueeze(0).unsqueeze(0), grid_plane,
                OperationProvenance("piecewise_constant_phase_pixels", None, None),
            )
            propagated = angular_spectrum_propagate(field, 3.0e-3, f"finite_pixel_rs_{name}").tensor[0, 0]
            observed = _spectral_physical_pixel_observables(
                propagated, factor, roi_m=pixel_count * pitch, detach=False, only_center=True
            )["physical_pixel_center"]
            selected_values = observed[sample_indices[:, None], sample_indices[None, :]].reshape(-1)
            asm = selected_values.detach().cpu().numpy().astype(np.complex128)
            overlap = np.vdot(reference, asm)
            alignment = np.exp(-1j * np.angle(overlap))
            aligned = asm * alignment
            reference_error = float(np.linalg.norm(aligned - reference) / max(np.linalg.norm(reference), 1.0e-300))
            center_field = selected_values[len(selected_values) // 2]
            real_derivative = torch.autograd.grad(center_field.real, theta, retain_graph=True)[0]
            imag_derivative = torch.autograd.grad(center_field.imag, theta, retain_graph=False)[0]
            automatic_complex = complex(float(real_derivative.detach().cpu()), float(imag_derivative.detach().cpu()))
            aligned_automatic = automatic_complex * alignment
            automatic_aligned_real = float(aligned_automatic.real)
            phase_derivative_relative = abs(aligned_automatic - derivatives[12]) / max(abs(derivatives[12]), 1.0e-14)
            production = {
                "asm_vs_real_space_rs_relative_l2_after_global_phase": reference_error,
                "asm_phase_derivative_before_global_alignment": [automatic_complex.real, automatic_complex.imag],
                "asm_phase_derivative_after_global_alignment": [aligned_automatic.real, aligned_automatic.imag],
                "phase_derivative_relative_error": float(phase_derivative_relative),
                "aligned_real_direction_agrees": automatic_aligned_real * analytic_direction > 0.0,
                "global_phase_alignment_rad": float(np.angle(overlap)),
                "asm_value_l2": float(np.linalg.norm(asm)),
                "rs_value_l2": float(np.linalg.norm(reference)),
            }
            del field, propagated, observed, selected_values, padded_grad
        observations[name] = {
            "gauss_order_4_vs_8_relative_l2": quadrature_4_to_8,
            "gauss_order_8_vs_12_relative_l2": quadrature_8_to_12,
            "rs_phase_derivative": analytic_direction,
            "rs_complex_phase_derivative": [derivatives[12].real, derivatives[12].imag],
            "rs_center_difference_relative_error": finite_difference_relative,
            "rs_value_l2": float(np.linalg.norm(reference)),
            **production,
            "passed": quadrature_8_to_12 <= WINDOW_TOLERANCE
            and finite_difference_relative <= RS_FD_TOLERANCE
            and (not compare_production or (
                production["asm_vs_real_space_rs_relative_l2_after_global_phase"] <= WINDOW_TOLERANCE
                and production["phase_derivative_relative_error"] <= FD_TOLERANCE
                and production["aligned_real_direction_agrees"]
            )),
        }
        if compare_production:
            clear_production_transfer_cache()
            torch.cuda.empty_cache()
    passed = all(item["passed"] for item in observations.values())
    return {"patterns": observations, "tolerance": WINDOW_TOLERANCE, "passed": passed,
            "source_pixels": [pixel_count, pixel_count], "subpixel_sampling_factor": factor,
            "quadrature_orders": [4, 8, 12], "reference": "direct_real_space_Rayleigh_Sommerfeld",
            "production_comparison_executed": compare_production,
            "grid": [grid, grid], "grid_pitch_m": pitch / factor,
            "physical_window_m": physical_window_m,
            "physical_window_matches_production": True}


def _gaussian_check(model: ChenLightGen400Baseline) -> dict[str, Any]:
    ols: GaussianFiberArrayOLS = model.ols
    mode = torch.outer(ols.modes_y[4], ols.modes_x[4]).to(torch.complex128)[None, None]
    field = ComplexField(mode, model.ols_input_plane,
                         OperationProvenance("analytic_center_mode", None, None))
    coeff = ols.project_coefficients(field)
    index = 4 * 10 + 4
    expected = torch.zeros_like(coeff)
    expected[0, index] = 1.0
    error = float(torch.linalg.vector_norm(coeff - expected))
    input_power = float(integrated_power(field)[0])
    output = ols(field)
    output_power = float(integrated_power(output)[0])
    diag = ols.geometry_diagnostics
    _require(error <= 1.0e-10 and abs(input_power - 1.0) <= 1.0e-8
             and output_power <= input_power * (1.0 + 1.0e-5)
             and diag["mode_norm_max_absolute_error"] <= 5.0e-6,
             "sampling-specific Gaussian projection, normalization or passivity check failed")
    return {"matched_mode_coefficient_l2_error": error, "input_power_w_for_unit_mode": input_power,
            "reemitted_power_w": output_power, **diag, "passed": True}


def _white_gray_black_integral() -> dict[str, Any]:
    outcomes = {}
    for factor in FACTORS:
        dy = dx = PIXEL_PITCH_M / factor
        for name, value in (("black", 0.0), ("half", 0.5), ("white", 1.0)):
            image = torch.full((400 * factor, 400 * factor), value, dtype=torch.float64)
            power = float((image * (SOURCE_POWER_W / (1.2e-3**2)) * dy * dx).sum())
            expected = SOURCE_POWER_W * value
            _require(math.isclose(power, expected, rel_tol=1.0e-12, abs_tol=1.0e-18),
                     f"{factor}x {name} input power integral is inconsistent")
            outcomes[f"factor_{factor}_{name}"] = power
    return outcomes


def _gradient_for(ctx: dict[str, Any], state: str, factor: int, row: int) -> dict[str, Any]:
    model: ChenLightGen400Baseline = ctx["models"][(state, factor)]
    for parameter in (model.encoder_phase, model.decoder_phase):
        parameter.requires_grad_(True)
    try:
        if ctx["device"].type == "cuda":
            clear_production_transfer_cache()
            torch.cuda.empty_cache()
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            _, _, loss = _one_forward(ctx, state, factor, row, grad=True, capture_fields=False)
            gradients = torch.autograd.grad(loss * ctx["beta"],
                                            (model.encoder_phase, model.decoder_phase),
                                            allow_unused=False)
        _require(all(torch.isfinite(value).all() for value in gradients),
                 f"nonfinite phase gradient at {state}, factor {factor}, row {row}")
        return {"encoder_phase": gradients[0].detach().cpu(), "decoder_phase": gradients[1].detach().cpu()}
    finally:
        for parameter in (model.encoder_phase, model.decoder_phase):
            parameter.grad = None
            parameter.requires_grad_(False)
        model.eval()
        if ctx["device"].type == "cuda":
            clear_production_transfer_cache()
            torch.cuda.empty_cache()


def _gradient_checks(ctx: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    selected = {int(label): int(row) for label, row in zip(
        ctx["selected_labels"].tolist(), ctx["rows"].tolist(), strict=True
    )}
    probes = [selected[0], selected[1]]
    for state in STATES:
        for row in probes:
            by_factor = {factor: _gradient_for(ctx, state, factor, row) for factor in FACTORS}
            comparisons = {}
            for group in ("encoder_phase", "decoder_phase"):
                for i in range(by_factor[1][group].shape[0]):
                    a = by_factor[1][group][i].double().flatten()
                    b = by_factor[2][group][i].double().flatten()
                    c = by_factor[4][group][i].double().flatten()
                    def norm(value):
                        return torch.linalg.vector_norm(value).clamp_min(1.0e-30)

                    pair = {}
                    for factor, candidate in ((2, b), (4, c)):
                        pair[f"factor_{factor}"] = {
                            "gradient_norm": float(torch.linalg.vector_norm(candidate)),
                            "relative_l2_vs_factor_1": float(norm(candidate - a) / norm(a)),
                            "cosine_vs_factor_1": float(torch.dot(candidate, a) / (norm(candidate) * norm(a))),
                        }
                    pair["factor_2_vs_factor_4"] = {
                        "relative_l2": float(norm(b - c) / norm(c)),
                        "cosine": float(torch.dot(b, c) / (norm(b) * norm(c))),
                    }
                    comparisons[f"{group}_layer_{i + 1}"] = pair
            result[f"{state}_row_{row}"] = {"label": int(ctx["labels"][row]), "comparisons": comparisons,
                                                "tolerance": GRADIENT_TOLERANCE, "passed": True}
    return result


def _finite_difference_checks(ctx: dict[str, Any]) -> dict[str, Any]:
    row = int(ctx["rows"][0])
    outputs = {}
    for state in STATES:
        model: ChenLightGen400Baseline = ctx["models"][(state, 1)]
        gradient_data = _gradient_for(ctx, state, 1, row)
        state_results = []
        for group, parameter, gradient in (
            ("encoder_phase", model.encoder_phase, gradient_data["encoder_phase"]),
            ("decoder_phase", model.decoder_phase, gradient_data["decoder_phase"]),
        ):
            for layer in range(parameter.shape[0]):
                grad = gradient[layer, 0].to(device=ctx["device"], dtype=parameter.dtype)
                rms = float(grad.double().square().mean().sqrt())
                if not math.isfinite(rms) or rms <= 1.0e-18:
                    state_results.append({"group": group, "layer": layer + 1,
                                          "status": "DEGENERATE_DIRECTION", "gradient_rms": rms})
                    continue
                direction = grad / rms
                base_phase = parameter[layer, 0].detach().clone()

                def value_at(step: float) -> float:
                    with torch.no_grad():
                        parameter[layer, 0].copy_(base_phase + step * direction)
                    _, _, loss = _one_forward(ctx, state, 1, row, grad=False, capture_fields=False)
                    return float(loss.detach().cpu()) * ctx["beta"]

                try:
                    for step in FD_STEPS_RAD:
                        plus = value_at(step)
                        minus = value_at(-step)
                        analytic = float((gradient[layer, 0].to(torch.float64).flatten()
                                          * direction.detach().cpu().double().flatten()).sum())
                        finite = (plus - minus) / (2.0 * step)
                        relative = abs(finite - analytic) / max(abs(finite), abs(analytic), 1.0e-30)
                        _require(finite * analytic > 0.0 and relative <= FD_TOLERANCE,
                                 f"finite difference disagrees at {state}/{group}/layer{layer + 1}/step{step}: "
                                 f"analytic={analytic}, central={finite}, relative_error={relative}")
                        state_results.append({"group": group, "layer": layer + 1, "step_rad": step,
                                              "gradient_rms": rms, "analytic_directional_derivative": analytic,
                                              "central_difference": finite, "relative_error": relative,
                                              "direction_agrees": True})
                finally:
                    with torch.no_grad():
                        parameter[layer, 0].copy_(base_phase)
        model.zero_grad(set_to_none=True)
        _require(all(item.get("status") != "DEGENERATE_DIRECTION" for item in state_results),
                 f"a selected finite-difference direction is degenerate at {state}")
        outputs[state] = {"row": row, "label": int(ctx["labels"][row]), "directions": state_results,
                          "relative_tolerance": FD_TOLERANCE, "passed": True}
    return outputs


def _measurement_identity(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "authorization_id": AUTHORITY,
        "start_checkpoint": str(_resolve(ctx["args"].start_state)),
        "endpoint_checkpoint": str(_resolve(ctx["args"].endpoint_checkpoint)),
        "joint_updates": 500,
        "fixed_rows": [int(v) for v in ctx["rows"].tolist()],
        "fixed_labels": [int(v) for v in ctx["selected_labels"].tolist()],
        "factors": list(FACTORS),
        "grids": {str(k): v for k, v in GRIDS.items()},
        "gain_w_inverse": GAIN,
    }


def _validate_block(block: dict[str, Any], ctx: dict[str, Any], index: int) -> None:
    row = int(ctx["rows"][index])
    label = int(ctx["selected_labels"][index])
    _require(block.get("schema") == MEASUREMENT_BLOCK_SCHEMA
             and block.get("identity") == _measurement_identity(ctx)
             and block.get("row_index") == index
             and block.get("row") == row and block.get("label") == label,
             f"saved sampling block {index} does not match fixed input row and model identity")
    _require(set(block["by_factor"]) == set(STATES),
             f"saved sampling block {index} lacks a model state")
    for state in STATES:
        _require(set(block["by_factor"][state]) == {str(value) for value in FACTORS},
                 f"saved sampling block {index} lacks a sampling factor for {state}")
        for factor in FACTORS:
            entry = block["by_factor"][state][str(factor)]
            _require(entry["metrics"]["state"] == state and entry["metrics"]["row"] == row
                     and entry["metrics"]["label"] == label and entry["metrics"]["factor"] == factor,
                     f"saved sampling block {index} contains mismatched metrics")
            physical = entry.get("physical_w_per_pixel")
            task = entry.get("task_intensity_untruncated")
            display = entry.get("display_intensity_clipped")
            target = entry.get("target")
            expected_shape = (1, 1, 400, 400)
            _require(torch.isfinite(entry["coefficients"]).all()
                     and all(isinstance(value, torch.Tensor) and value.shape == expected_shape
                             and torch.isfinite(value).all() for value in (physical, task, display, target)),
                     f"saved sampling block {index} has invalid camera tensors")
            _require(torch.allclose(task, physical * GAIN, rtol=2.0e-6, atol=1.0e-7)
                     and torch.equal(display, task.clamp(0.0, 1.0)),
                     f"saved sampling block {index} camera units or display clipping are inconsistent")
            metric = entry["metrics"]
            expected_mse = float((task - target).square().mean())
            _require(abs(expected_mse - float(metric["mse400"])) <= 1.0e-7
                     and abs(float(physical.sum()) - float(metric["camera_roi_power_w"]))
                     <= max(1.0e-14, 1.0e-5 * abs(float(metric["camera_roi_power_w"]))),
                     f"saved sampling block {index} disagrees with its MSE or physical ROI power")


def _decode_saved_coefficients(ctx: dict[str, Any], state: str, row: int, coefficients: torch.Tensor,
                               factor: int) -> tuple[dict[str, Any], dict[str, Any]]:
    model: ChenLightGen400Baseline = ctx["models"][(state, factor)]
    device = ctx["device"]
    target, _ = baseline._image_batch(ctx["pixels"], torch.tensor([row]), device)
    fields: dict[str, dict[str, torch.Tensor]] = {}
    def callback(name, value):
        _record_field(fields, factor, name, value)

    with torch.no_grad():
        result = model.forward_from_ols(coefficients.to(device).unsqueeze(0), return_diagnostics=True,
                                        field_callback=callback)
        task = result["physical_w_per_pixel"] * GAIN
        summary, psnr, ssim = baseline._reconstruction_batch_metrics(
            task, target, torch.tensor([row]), torch.tensor([row])
        )
        metrics = {
            "state": state, "row": row, "label": int(ctx["labels"][row]), "factor": factor,
            "grid": GRIDS[factor], "mse400": float((task - target).square().mean().cpu()),
            "psnr28_db": float(psnr[0].cpu()), "ssim28": float(ssim[0].cpu()),
            "saturation_fraction": float((((task < 0.0) | (task > 1.0)).double().mean()).cpu()),
            "modal_power_w": float(result["ols_modal_power_w"][0].cpu()),
            "ols_reemitted_power_w": float(result["ols_reemitted_power_w"][0].cpu()),
            "camera_roi_power_w": float(result["camera_roi_power_w"][0].cpu()),
            "camera_full_field_power_w": float(result["camera_full_field_power_w"][0].cpu()),
            "camera_outside_roi_power_w": float(result["camera_outside_roi_power_w"][0].cpu()),
        }
        record = {"metrics": metrics,
                  "physical_w_per_pixel": result["physical_w_per_pixel"].detach().cpu(),
                  "task_intensity_untruncated": task.detach().cpu(),
                  "display_intensity_clipped": task.detach().cpu().clamp(0.0, 1.0),
                  "port_power_w": coefficients.abs().square().double().cpu(),
                  "decoder_plane_powers_w": result["plane_powers_w"][0].detach().cpu()}
    _require(all(torch.isfinite(value).all() for value in (record["physical_w_per_pixel"],
                                                            record["task_intensity_untruncated"],
                                                            record["display_intensity_clipped"],
                                                            record["port_power_w"],
                                                            record["decoder_plane_powers_w"])),
             f"fixed-coefficient Decoder output is nonfinite at {state}, row {row}, factor {factor}")
    return record, fields


def _decoder_probe_identity(ctx: dict[str, Any]) -> dict[str, Any]:
    return {"sampling_identity": _measurement_identity(ctx),
            "rows": [int(ctx["rows"][0]), int(ctx["rows"][1])],
            "states": list(STATES), "source_coefficients": "same-row-factor-1-full-chain"}


def _validate_decoder_probe(block: dict[str, Any], ctx: dict[str, Any], state: str, row_index: int) -> None:
    row = int(ctx["rows"][row_index])
    label = int(ctx["selected_labels"][row_index])
    _require(block.get("schema") == DECODER_PROBE_SCHEMA
             and block.get("identity") == _decoder_probe_identity(ctx)
             and block.get("state") == state and block.get("row_index") == row_index
             and block.get("row") == row and block.get("label") == label
             and set(block.get("by_factor", {})) == {str(factor) for factor in FACTORS},
             "fixed-coefficient Decoder probe identity or coverage is invalid")
    coefficients = block.get("source_coefficients")
    _require(isinstance(coefficients, torch.Tensor) and coefficients.shape == (100,)
             and coefficients.is_complex() and torch.isfinite(coefficients).all(),
             "fixed-coefficient Decoder probe source coefficients are invalid")
    for factor in FACTORS:
        item = block["by_factor"][str(factor)]
        _require(item["metrics"]["state"] == state and item["metrics"]["row"] == row
                 and item["metrics"]["factor"] == factor
                 and item["physical_w_per_pixel"].shape == (1, 1, 400, 400)
                 and item["task_intensity_untruncated"].shape == (1, 1, 400, 400)
                 and item["display_intensity_clipped"].shape == (1, 1, 400, 400)
                 and torch.isfinite(item["physical_w_per_pixel"]).all()
                 and torch.isfinite(item["task_intensity_untruncated"]).all()
                 and torch.allclose(item["task_intensity_untruncated"],
                                    item["physical_w_per_pixel"] * GAIN, rtol=2.0e-6, atol=1.0e-7)
                 and torch.equal(item["display_intensity_clipped"],
                                 item["task_intensity_untruncated"].clamp(0.0, 1.0)),
                 "fixed-coefficient Decoder probe output is invalid")


def _run_fixed_coefficient_decoder_probes(ctx: dict[str, Any], output: Path) -> dict[str, Any] | None:
    identity = _decoder_probe_identity(ctx)
    cases = [(state, index) for state in STATES for index in (0, 1)]
    completed = []
    for state, row_index in cases:
        case_path = output / f"decoder_probe_{state}_{row_index:02d}.pt"
        if case_path.is_file():
            block = torch.load(case_path, map_location="cpu", weights_only=False)
            _validate_decoder_probe(block, ctx, state, row_index)
            completed.append((state, row_index))
    flag = [None]
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda signum, _frame, state=flag: state.__setitem__(0, signum))
    try:
        for state, row_index in cases:
            if (state, row_index) in completed:
                continue
            case_path = output / f"decoder_probe_{state}_{row_index:02d}.pt"
            row = int(ctx["rows"][row_index])
            source_block = torch.load(output / f"sample_{row_index:02d}.pt", map_location="cpu", weights_only=False)
            _validate_block(source_block, ctx, row_index)
            coefficients = source_block["by_factor"][state]["1"]["coefficients"]
            by_factor, fields_by_factor = {}, {}
            for factor in FACTORS:
                by_factor[str(factor)], fields_by_factor[factor] = _decode_saved_coefficients(
                    ctx, state, row, coefficients, factor
                )
            field_checks = {}
            for view in FIELD_VIEWS:
                reference = {name: values[view] for name, values in fields_by_factor[4].items()}
                for factor in (1, 2):
                    candidate = {name: values[view] for name, values in fields_by_factor[factor].items()}
                    errors, phase = baseline._complex_relative_errors(candidate, reference)
                    field_checks[f"factor_{factor}_vs_4_{view}"] = {
                        "per_plane_relative_l2": errors, "max_relative_l2": max(errors.values()),
                        "global_phase_rad": phase}
            block = {"schema": DECODER_PROBE_SCHEMA, "identity": identity,
                     "state": state, "row_index": row_index, "row": row,
                     "label": int(ctx["labels"][row]), "source_coefficients": coefficients,
                     "by_factor": by_factor, "field_checks": field_checks}
            _validate_decoder_probe(block, ctx, state, row_index)
            atomic_torch_save_new(case_path, block)
            completed.append((state, row_index))
            atomic_torch_save(output / "decoder_probe_checkpoint.pt", {
                "schema": "chen_lightgen_400_decoder_sampling_recovery.v1", "identity": identity,
                "sampling_checkpoint": str(output / "checkpoint_sample_10.pt"),
                "completed_cases": [[item[0], item[1]] for item in completed],
                "next_case": len(completed), "signal_received": flag[0]})
            atomic_write_json(output / "decoder_probe_progress.json", {
                "status": "INTERRUPTED" if flag[0] is not None else "RUNNING",
                "completed_cases": [[item[0], item[1]] for item in completed],
                "total_cases": len(cases), "signal_received": flag[0], "updated_at_utc": time.time()})
            if flag[0] is not None:
                return {"status": "INTERRUPTED", "resume_checkpoint": str(output / "decoder_probe_checkpoint.pt"),
                        "completed_cases": len(completed)}
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    atomic_write_json(output / "decoder_probe_progress.json", {
        "status": "DECODER_PROBES_COMPLETE", "completed_cases": [[item[0], item[1]] for item in completed],
        "total_cases": len(cases), "signal_received": flag[0], "updated_at_utc": time.time()})
    return {"status": "DECODER_PROBES_COMPLETE", "completed_cases": len(completed)}


def _run_samples(ctx: dict[str, Any], output: Path, *, resume_checkpoint: Path | None) -> dict[str, Any] | None:
    identity = _measurement_identity(ctx)
    completed = 0
    if resume_checkpoint is not None:
        _require(resume_checkpoint.is_file() and output.is_dir(), "resume requires an explicit existing checkpoint and run directory")
        checkpoint = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        _require(checkpoint.get("schema") == "chen_lightgen_400_sampling_recovery.v1"
                 and checkpoint.get("identity") == identity,
                 "sampling recovery checkpoint belongs to different inputs")
        completed = int(checkpoint["completed_samples"])
        _require(0 <= completed <= len(ctx["rows"]), "sampling checkpoint cursor is invalid")
        for index in range(completed):
            path = output / f"sample_{index:02d}.pt"
            _require(path.is_file(), f"completed measurement block is missing: {path}")
            _validate_block(torch.load(path, map_location="cpu", weights_only=False), ctx, index)
    else:
        _require(not output.exists(), f"refusing to overwrite run directory: {output}")
        output.mkdir(parents=True, exist_ok=False)
        atomic_write_new_json(output / "run_manifest.json", {
            "schema": "chen_lightgen_400_sampling_run.v2", "identity": identity,
            "baseline_checkpoint": str(_resolve(ctx["args"].baseline_checkpoint)),
            "start_state": str(_resolve(ctx["args"].start_state)),
            "endpoint_checkpoint": str(_resolve(ctx["args"].endpoint_checkpoint)),
            "fixed_population": str(_resolve(ctx["args"].fixed_population)),
            "source_manifest": str(_resolve(ctx["args"].source_manifest)),
            "optimizer_updates": 0, "measurement_blocks": len(ctx["rows"]),
        })
    flag = [None]
    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        previous[sig] = signal.signal(sig, lambda signum, _frame, state=flag: state.__setitem__(0, signum))
    try:
        for index in range(completed, len(ctx["rows"])):
            row = int(ctx["rows"][index])
            by_factor: dict[int, Any] = {}
            field_variants: dict[int, Any] = {}
            for state in STATES:
                variants: dict[int, Any] = {}
                state_fields: dict[int, Any] = {}
                for factor in FACTORS:
                    value, fields, _ = _one_forward(ctx, state, factor, row)
                    variants[factor] = value
                    state_fields[factor] = fields
                    del value, fields
                by_factor[state] = variants
                field_variants[state] = state_fields
            comparisons = {state: _compare_sample(by_factor[state], field_variants[state]) for state in STATES}
            block = {"schema": MEASUREMENT_BLOCK_SCHEMA, "identity": identity,
                     "row_index": index, "row": row, "label": int(ctx["selected_labels"][index]),
                     "by_factor": {state: {str(factor): value for factor, value in factors.items()}
                                   for state, factors in by_factor.items()},
                     "sampling_comparisons": comparisons}
            _validate_block(block, ctx, index)
            atomic_torch_save_new(output / f"sample_{index:02d}.pt", block)
            completed = index + 1
            checkpoint = {"schema": "chen_lightgen_400_sampling_recovery.v1", "identity": identity,
                          "completed_samples": completed, "next_row_index": completed,
                          "completed_rows": [int(v) for v in ctx["rows"][:completed].tolist()],
                          "signal_received": flag[0]}
            checkpoint_path = output / f"checkpoint_sample_{completed:02d}.pt"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            atomic_torch_save(checkpoint_path, checkpoint)
            atomic_write_json(output / "progress.json", {
                "status": "INTERRUPTED" if flag[0] is not None else "RUNNING",
                "completed_samples": completed, "total_samples": len(ctx["rows"]),
                "next_row_index": completed, "signal_received": flag[0], "updated_at_utc": time.time(),
            })
            if flag[0] is not None:
                return {"status": "INTERRUPTED", "resume_checkpoint": str(checkpoint_path),
                        "completed_samples": completed}
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    atomic_write_json(output / "progress.json", {
        "status": "MEASUREMENT_BLOCKS_COMPLETE", "completed_samples": completed,
        "total_samples": len(ctx["rows"]), "next_row_index": completed,
        "signal_received": flag[0], "updated_at_utc": time.time(),
    })
    return {"status": "MEASUREMENT_BLOCKS_COMPLETE", "completed_samples": completed,
            "resume_checkpoint": str(output / f"checkpoint_sample_{completed:02d}.pt")}


def _aggregate(ctx: dict[str, Any], output: Path, checks: dict[str, Any]) -> dict[str, Any]:
    blocks = []
    for index in range(10):
        block = torch.load(output / f"sample_{index:02d}.pt", map_location="cpu", weights_only=False)
        _validate_block(block, ctx, index)
        blocks.append(block)
    decoder_probe_rows: list[dict[str, Any]] = []
    decoder_field_rows: list[dict[str, Any]] = []
    for state in STATES:
        for row_index in (0, 1):
            probe = torch.load(output / f"decoder_probe_{state}_{row_index:02d}.pt",
                               map_location="cpu", weights_only=False)
            _validate_decoder_probe(probe, ctx, state, row_index)
            for factor in FACTORS:
                decoder_probe_rows.append(probe["by_factor"][str(factor)]["metrics"])
            for comparison, details in probe["field_checks"].items():
                for plane, error in details["per_plane_relative_l2"].items():
                    decoder_field_rows.append({"state": state, "row": probe["row"], "label": probe["label"],
                                               "comparison": comparison, "plane": plane,
                                               "relative_l2": error,
                                               "global_phase_rad": details["global_phase_rad"]})
    sample_rows: list[dict[str, Any]] = []
    port_rows: list[dict[str, Any]] = []
    plane_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    examples: dict[int, dict[str, torch.Tensor]] = {}
    for block in blocks:
        row, label = block["row"], block["label"]
        examples[row] = {}
        for state in STATES:
            for factor in FACTORS:
                item = block["by_factor"][state][str(factor)]
                metric = dict(item["metrics"])
                metric["modal_roots_ge_1nw"] = int((item["port_power_w"] >= 1e-9).sum())
                metric["weakest_port_power_w"] = float(item["port_power_w"].min())
                metric["modal_n_eff"] = float(item["port_power_w"].sum().square()
                                               / item["port_power_w"].square().sum().clamp_min(1e-300))
                metric["system_coupling_efficiency"] = metric["modal_power_w"] / max(metric["input_power_w"], 1e-300)
                metric["modal_capture_fraction"] = metric["modal_power_w"] / max(metric["ols_incident_power_w"], 1e-300)
                sample_rows.append(metric)
                for port, value in enumerate(item["port_power_w"].tolist()):
                    port_rows.append({"state": state, "row": row, "label": label, "factor": factor,
                                      "port": port, "array_row": port // 10, "array_column": port % 10,
                                      "modal_power_w": value})
                for plane, key in enumerate(("encoder_power_1_w", "encoder_power_2_w", "decoder_entry_power_w",
                                             "decoder_1_power_w", "decoder_2_power_w", "decoder_3_power_w",
                                             "decoder_4_power_w")):
                    plane_rows.append({"state": state, "row": row, "label": label, "factor": factor,
                                       "plane_index": plane + 1, "plane": key, "power_w": metric[key]})
                image = item["task_intensity_untruncated"]
                examples[row][f"{state}_f{factor}"] = F.adaptive_avg_pool2d(image, (28, 28))[0, 0]
            comparison = block["sampling_comparisons"][state]
            for kind, group in comparison.items():
                for pair, detail in group.items():
                    if kind == "field_checks":
                        for plane, error in detail["per_plane_relative_l2"].items():
                            field_rows.append({"state": state, "row": row, "label": label,
                                               "comparison": pair, "plane": plane,
                                               "relative_l2": error, "global_phase_rad": detail["global_phase_rad"]})
    decision = _sampling_decision(blocks, sample_rows, field_rows, checks)
    by_state = {}
    for state in STATES:
        by_factor = {}
        for factor in FACTORS:
            subset = [x for x in sample_rows if x["state"] == state and x["factor"] == factor]
            by_factor[str(factor)] = {
                "count": len(subset),
                **{f"mean_{key}": sum(x[key] for x in subset) / len(subset)
                   for key in ("mse400", "psnr28_db", "ssim28", "saturation_fraction", "input_power_w",
                               "ols_incident_power_w", "modal_power_w", "camera_roi_power_w",
                               "camera_full_field_power_w", "camera_outside_roi_power_w",
                               "system_coupling_efficiency", "modal_capture_fraction", "modal_n_eff",
                               "modal_roots_ge_1nw")},
            }
        by_state[state] = by_factor
    result = {
        "schema": "chen_lightgen_400_sampling_closure_report.v3", "status": decision["status"],
        "decision": decision["decision"], "identity": _measurement_identity(ctx),
        "source_statuses": {"original_baseline": ctx["source_report"]["status"],
                             "joint_500_step": ctx["joint_report"]["status"],
                             "crossdecode": ctx["manifest"]["analysis_mode"]},
        "data_roles": {"sample_count": 10, "all_canonical_train": True, "official_test_reads": 0,
                       "rows": ctx["rows"].tolist(), "labels": ctx["selected_labels"].tolist()},
        "optimizer_updates": 0, "formal_full_chain_forwards": 60, "fixed_coefficient_decoder_forwards": 12,
        "gradient_forwards": 12, "finite_difference_forwards": 48,
        "common_camera_gain_w_inverse": GAIN, "sampling_factors": list(FACTORS),
        "grid_by_factor": {str(k): v for k, v in GRIDS.items()}, "comparison_roi_m": ROI_M,
        "field_observation_semantics": {
            "physical_pixel_center": "full_window_spectral_evaluation_at_shared_3um_pixel_centres",
            "physical_pixel_average": "full_window_spectral_evaluation_with_3um_box_sinc_integral",
            "camera": "intensity_then_area_integrate; complex-field averaging is not used for detection",
        },
        "window_tolerance": WINDOW_TOLERANCE, "coefficient_tolerance": COEFFICIENT_TOLERANCE,
        "gradient_tolerance": GRADIENT_TOLERANCE, "finite_difference_tolerance": FD_TOLERANCE,
        "summary_by_state_and_factor": by_state, "sample_rows": sample_rows,
        "fixed_coefficient_decoder_probe_rows": decoder_probe_rows,
        "fixed_coefficient_decoder_field_errors": decoder_field_rows,
        "window_and_field_errors": field_rows, "comparison_reference": {
            "camera_quadrature_counterexample": _complex_quadrature_reference(),
            "angular_spectrum_plane_wave_and_gradient": _propagation_reference_check(),
            "finite_pixel_rayleigh_sommerfeld": decision["finite_pixel_rayleigh_sommerfeld"],
            "analytic_physical_pixel_observer": checks["analytic_physical_pixel_observer"],
            "white_gray_black_power_w": _white_gray_black_integral(),
            "gaussian_by_factor": decision["gaussian_checks"],
            "gradient_checks": decision["gradient_checks"],
            "finite_difference_checks": decision["finite_difference_checks"],
        },
        "sampling_decision_detail": decision,
        "official_test_reads": 0,
    }
    _save_csv(output / "per_sample_sampling_metrics.csv", sample_rows)
    _save_csv(output / "per_port_power.csv", port_rows)
    _save_csv(output / "plane_power_budget.csv", plane_rows)
    _save_csv(output / "fixed_coefficient_decoder_metrics.csv", decoder_probe_rows)
    _save_csv(output / "fixed_coefficient_decoder_field_errors.csv", decoder_field_rows)
    if field_rows:
        _save_csv(output / "per_plane_field_errors.csv", field_rows)
    atlas = _write_atlas(output / "fixed_sampling_atlas.png", blocks, examples)
    result["artifacts"] = {"per_sample": "per_sample_sampling_metrics.csv", "per_port": "per_port_power.csv",
                           "plane_power_budget": "plane_power_budget.csv",
                           "fixed_coefficient_decoder_metrics": "fixed_coefficient_decoder_metrics.csv",
                           "fixed_coefficient_decoder_field_errors": "fixed_coefficient_decoder_field_errors.csv",
                           "field_errors": "per_plane_field_errors.csv", "atlas": atlas}
    _write_report(output / "results_analysis.md", result)
    return result


def _migrate_cached_sample_block(block: dict[str, Any], ctx: dict[str, Any], index: int,
                                source_identity: dict[str, Any]) -> dict[str, Any]:
    row = int(ctx["rows"][index])
    label = int(ctx["selected_labels"][index])
    _require(block.get("schema") == "chen_lightgen_400_sampling_measurement_block.v1"
             and block.get("identity") == source_identity and block.get("row_index") == index
             and block.get("row") == row and block.get("label") == label,
             f"cached sample block {index} does not match its recorded source identity")
    _require(set(block.get("by_factor", {})) == set(STATES),
             f"cached sample block {index} lacks a model state")
    converted = dict(block)
    converted.update({"schema": MEASUREMENT_BLOCK_SCHEMA, "identity": _measurement_identity(ctx)})
    converted_states = {}
    for state in STATES:
        _require(set(block["by_factor"][state]) == {str(factor) for factor in FACTORS},
                 f"cached sample block {index} lacks a sampling factor for {state}")
        converted_factors = {}
        for factor in FACTORS:
            source = block["by_factor"][state][str(factor)]
            old_field = source["physical_w_per_pixel"]
            target = source["target"]
            metric = source["metrics"]
            _require(metric["state"] == state and metric["row"] == row and metric["label"] == label
                     and metric["factor"] == factor and metric["gain_w_inverse"] == GAIN,
                     f"cached sample block {index} contains mismatched metrics")
            legacy_mse = float((old_field - target).square().mean())
            _require(abs(legacy_mse - float(metric["mse400"])) <= 1.0e-7,
                     f"cached sample block {index} does not store the measured task intensity")
            physical = old_field / GAIN
            roi_power = float(physical.sum())
            expected_roi = float(metric["camera_roi_power_w"])
            _require(abs(roi_power - expected_roi) <= max(1.0e-14, 1.0e-5 * abs(expected_roi)),
                     f"cached sample block {index} cannot recover its physical camera power")
            converted_item = dict(source)
            converted_item.update({
                "physical_w_per_pixel": physical,
                "task_intensity_untruncated": old_field,
                "display_intensity_clipped": old_field.clamp(0.0, 1.0),
            })
            converted_factors[str(factor)] = converted_item
        converted_states[state] = converted_factors
    converted["by_factor"] = converted_states
    converted["artifact_unit_migration"] = "v1_task_intensity_to_v2_physical_and_task_views"
    _validate_block(converted, ctx, index)
    return converted


def _migrate_cached_decoder_probe(block: dict[str, Any], ctx: dict[str, Any], state: str,
                                 row_index: int, source_identity: dict[str, Any]) -> dict[str, Any]:
    row = int(ctx["rows"][row_index])
    label = int(ctx["selected_labels"][row_index])
    source_probe_identity = {
        "sampling_identity": source_identity,
        "rows": [int(value) for value in source_identity["fixed_rows"][:2]],
        "states": list(STATES),
        "source_coefficients": "same-row-factor-1-full-chain",
    }
    _require(block.get("schema") == "chen_lightgen_400_decoder_sampling_probe.v1"
             and block.get("identity") == source_probe_identity and block.get("state") == state
             and block.get("row_index") == row_index and block.get("row") == row and block.get("label") == label,
             f"cached Decoder probe {state}/{row_index} does not match its recorded source identity")
    _require(set(block.get("by_factor", {})) == {str(factor) for factor in FACTORS},
             f"cached Decoder probe {state}/{row_index} lacks a sampling factor")
    converted = dict(block)
    converted.update({"schema": DECODER_PROBE_SCHEMA, "identity": _decoder_probe_identity(ctx)})
    converted_factors = {}
    for factor in FACTORS:
        source = block["by_factor"][str(factor)]
        physical = source["physical_w_per_pixel"]
        metric = source["metrics"]
        _require(metric["state"] == state and metric["row"] == row and metric["label"] == label
                 and metric["factor"] == factor,
                 f"cached Decoder probe {state}/{row_index} contains mismatched metrics")
        expected_roi = float(metric["camera_roi_power_w"])
        _require(abs(float(physical.sum()) - expected_roi) <= max(1.0e-14, 1.0e-5 * abs(expected_roi)),
                 f"cached Decoder probe {state}/{row_index} physical camera power is inconsistent")
        task = physical * GAIN
        item = dict(source)
        item.update({"physical_w_per_pixel": physical, "task_intensity_untruncated": task,
                     "display_intensity_clipped": task.clamp(0.0, 1.0)})
        converted_factors[str(factor)] = item
    converted["by_factor"] = converted_factors
    _validate_decoder_probe(converted, ctx, state, row_index)
    return converted


def _reanalyze_cached_run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    source_dir = _resolve(args.source_run_dir)
    output = _resolve(args.output_dir)
    preflight_dir = _resolve(args.preflight_dir)
    _require(source_dir.is_dir() and not output.exists(),
             "cached reanalysis requires an explicit source run and a new output directory")
    for name in ("baseline_checkpoint", "baseline_report", "joint_report", "source_manifest",
                 "fixed_population", "data", "config", "start_state", "endpoint_checkpoint"):
        _require(_resolve(getattr(args, name)).is_file(), f"explicit {name} is missing")
    preflight = _json(preflight_dir / "preflight.json")
    receipt = _json(preflight_dir / "preflight_ready.json")
    _require(preflight.get("status") == "READY" and preflight.get("task_id") == TASK_ID
             and preflight.get("authorization_id") == AUTHORITY and preflight.get("receipt") == receipt
             and receipt.get("status") == "READY" and receipt.get("authorization_id") == AUTHORITY
             and all(receipt.get("checks", {}).get(group, {}).get("passed") is True for group in (
                 "CORRECTNESS_INVARIANTS", "IMPLEMENTATION_SELECTION", "RESOURCE_ENVELOPE")),
             "cached reanalysis requires the original complete READY receipt")
    source_manifest = _json(source_dir / "run_manifest.json")
    source_report = _json(source_dir / "report.json")
    source_identity = source_manifest.get("identity")
    _require(source_manifest.get("schema") == "chen_lightgen_400_sampling_run.v1"
             and source_manifest.get("optimizer_updates") == 0 and source_manifest.get("measurement_blocks") == 10
             and source_report.get("identity") == source_identity
             and source_report.get("formal_full_chain_forwards") == 60
             and source_report.get("fixed_coefficient_decoder_forwards") == 12
             and source_report.get("optimizer_updates") == 0 and source_report.get("official_test_reads") == 0,
             "cached source run is incomplete or does not match this zero-update measurement")
    source_baseline_report = _json(_resolve(args.baseline_report))
    joint_report = _json(_resolve(args.joint_report))
    manifest = _json(_resolve(args.source_manifest))
    pixels, labels, train_indices, _ = baseline._load_data(_resolve(args.data))
    population = quickcheck._load_fixed_population(_resolve(args.fixed_population), labels, train_indices)
    rows, selected_labels = _selected_rows(population, labels, train_indices)
    _require(source_identity.get("fixed_rows") == rows.tolist()
             and source_identity.get("fixed_labels") == selected_labels.tolist()
             and source_identity.get("task_id") == TASK_ID
             and source_identity.get("authorization_id") == AUTHORITY
             and source_identity.get("factors") == list(FACTORS)
             and source_identity.get("grids") == {str(key): value for key, value in GRIDS.items()}
             and source_identity.get("gain_w_inverse") == GAIN
             and Path(source_identity.get("start_checkpoint", "")).name == _resolve(args.start_state).name
             and Path(source_identity.get("endpoint_checkpoint", "")).name == _resolve(args.endpoint_checkpoint).name,
             "cached measurement identity differs from the explicit local source, rows, or gain")
    _require(source_baseline_report.get("status") == "VALID_FAILED"
             and joint_report.get("status") == "VALID_FAILED"
             and manifest.get("grid") == 4096 and manifest.get("optimizer_updates") == 0,
             "explicit baseline, joint endpoint, or crossdecode source identity is invalid")
    ctx = {"args": args, "rows": rows, "selected_labels": selected_labels, "labels": labels,
           "pixels": pixels, "source_report": source_baseline_report, "joint_report": joint_report,
           "manifest": manifest}
    identity = _measurement_identity(ctx)
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_new_json(output / "run_manifest.json", {
        "schema": "chen_lightgen_400_sampling_reanalysis.v1", "identity": identity,
        "source_run_dir": str(source_dir), "source_identity": source_identity,
        "preflight_dir": str(preflight_dir), "optimizer_updates": 0,
        "new_optical_forwards": 0, "source_full_chain_forwards": 60,
        "source_decoder_probe_forwards": 12,
    })
    for index in range(10):
        cached = torch.load(source_dir / f"sample_{index:02d}.pt", map_location="cpu", weights_only=False)
        corrected = _migrate_cached_sample_block(cached, ctx, index, source_identity)
        atomic_torch_save_new(output / f"sample_{index:02d}.pt", corrected)
    for state in STATES:
        for row_index in (0, 1):
            cached = torch.load(source_dir / f"decoder_probe_{state}_{row_index:02d}.pt",
                                map_location="cpu", weights_only=False)
            corrected = _migrate_cached_decoder_probe(cached, ctx, state, row_index, source_identity)
            atomic_torch_save_new(output / f"decoder_probe_{state}_{row_index:02d}.pt", corrected)
    migration = {
        "status": "UNIT_CLOSED_FROM_CACHED_FORWARD_OUTPUTS", "source_attempt": str(source_dir),
        "source_sample_schema": "chen_lightgen_400_sampling_measurement_block.v1",
        "source_field_semantics": "gain_scaled_untruncated_task_intensity",
        "corrected_physical_field": "legacy_stored_task_intensity / frozen_gain",
        "task_field": "legacy_stored_value_preserved_exactly",
        "display_field": "task_intensity_clipped_to_0_1",
        "sample_blocks_migrated": 10, "decoder_probe_blocks_migrated": 4,
        "new_optical_forwards": 0, "all_sample_mse_and_camera_power_checks_passed": True,
        "decoder_probe_camera_power_checks_passed": True,
    }
    atomic_write_new_json(output / "artifact_unit_correction.json", migration)
    checks = preflight["checks"]
    report = _aggregate(ctx, output, checks)
    report["preflight_status"] = "READY"
    report["preflight_dir"] = str(preflight_dir)
    report["source_measurement_run_dir"] = str(source_dir)
    report["cached_reanalysis_elapsed_seconds"] = time.perf_counter() - started
    report["new_optical_forwards_during_reanalysis"] = 0
    report["artifact_unit_correction"] = migration
    _write_report(output / "results_analysis.md", report)
    atomic_write_new_json(output / "report.json", report)
    atomic_write_new_json(output / "reanalysis_complete.json", {
        "status": "COMPLETE", "source_run_dir": str(source_dir), "optimizer_updates": 0,
        "new_optical_forwards": 0, "report_status": report["status"], "decision": report["decision"],
    })
    return report


def _sampling_decision(blocks: list[dict[str, Any]], sample_rows: list[dict[str, Any]],
                       field_rows: list[dict[str, Any]], checks: dict[str, Any]) -> dict[str, Any]:
    gaussian_checks = checks["gaussian_by_factor"]
    gradient_checks = checks["gradient_checks"]
    finite_difference_checks = checks["finite_difference_checks"]
    errors: dict[str, float] = {}
    for row in field_rows:
        errors[f"{row['state']}_{row['row']}_{row['comparison']}_{row['plane']}"] = float(row["relative_l2"])
    max_factor1 = max((r["relative_l2"] for r in field_rows if "factor_1_vs_4" in r["comparison"]), default=math.inf)
    max_factor2 = max((r["relative_l2"] for r in field_rows if "factor_2_vs_4" in r["comparison"]), default=math.inf)
    max_coeff1 = max(float(b["sampling_comparisons"][state]["coefficient_checks"]["factor_1_vs_4"]["complex_relative_l2_after_global_phase"])
                     for b in blocks for state in STATES)
    max_coeff2 = max(float(b["sampling_comparisons"][state]["coefficient_checks"]["factor_2_vs_4"]["complex_relative_l2_after_global_phase"])
                     for b in blocks for state in STATES)
    max_grad2 = max(value["relative_l2_vs_factor_1"]
                    for record in gradient_checks.values()
                    for by_factor in record["comparisons"].values()
                    for factor, value in by_factor.items() if factor == "factor_2")
    max_grad4 = max(value["relative_l2_vs_factor_1"]
                    for record in gradient_checks.values()
                    for by_factor in record["comparisons"].values()
                    for factor, value in by_factor.items() if factor == "factor_4")
    max_grad2_vs4 = max(value["relative_l2"]
                        for record in gradient_checks.values()
                        for by_factor in record["comparisons"].values()
                        for factor, value in by_factor.items() if factor == "factor_2_vs_factor_4")
    independent_reference = checks.get("finite_pixel_rayleigh_sommerfeld", {})
    independent_reference_passed = independent_reference.get("passed") is True
    pass_factor1 = max_factor1 <= WINDOW_TOLERANCE and max_coeff1 <= COEFFICIENT_TOLERANCE \
        and max_grad2 <= GRADIENT_TOLERANCE and max_grad4 <= GRADIENT_TOLERANCE \
        and independent_reference_passed
    window_expansion = checks.get("factor_2_window_expansion", {"required": False, "passed": False})
    pass_factor2 = max_factor2 <= WINDOW_TOLERANCE and max_coeff2 <= COEFFICIENT_TOLERANCE \
        and max_grad2_vs4 <= GRADIENT_TOLERANCE and window_expansion.get("passed") is True \
        and independent_reference_passed
    if pass_factor1:
        decision, status = "THREE_MICRON_NUMERICALLY_SUPPORTED_FOR_FIXED_PROBES", "VALID_COMPLETED"
    elif pass_factor2:
        decision, status = "ONE_POINT_FIVE_MICRON_NUMERICALLY_SUPPORTED_FOR_FIXED_PROBES", "VALID_COMPLETED"
    else:
        decision, status = "NO_NUMERICALLY_QUALIFIED_SAMPLING_CANDIDATE", "NUMERICAL_NOT_QUALIFIED"
    return {"decision": decision, "status": status, "max_complex_error_factor1_vs4": max_factor1,
            "max_complex_error_factor2_vs4": max_factor2, "max_coefficient_error_factor1_vs4": max_coeff1,
            "max_coefficient_error_factor2_vs4": max_coeff2, "max_gradient_error_factor2": max_grad2,
            "max_gradient_error_factor4": max_grad4, "max_gradient_error_factor2_vs4": max_grad2_vs4,
            "factor1_pass": pass_factor1,
            "factor2_pass": pass_factor2, "gaussian_checks": gaussian_checks,
            "finite_pixel_rayleigh_sommerfeld_passed": independent_reference_passed,
            "finite_pixel_rayleigh_sommerfeld": independent_reference,
            "gradient_checks": gradient_checks, "finite_difference_checks": finite_difference_checks,
            "factor_2_window_expansion": window_expansion,
            "field_error_records": errors}


def _write_atlas(path: Path, blocks: list[dict[str, Any]], examples: dict[int, dict[str, torch.Tensor]]) -> str:
    columns = ("target", "start 3um", "start 1.5um", "start .75um", "end 3um", "end 1.5um", "end .75um")
    tile, row_h, label_w = 84, 96, 25
    image = Image.new("L", (label_w + tile * len(columns), row_h * len(blocks)), color=255)
    draw = ImageDraw.Draw(image)
    for i, block in enumerate(blocks):
        row, label = block["row"], block["label"]
        y = i * row_h
        draw.text((1, y + 4), f"{label}:{row}", fill=0)
        target = block["by_factor"][STATE_START]["1"]["target"][0, 0]
        for col, name in enumerate(columns):
            if name == "target":
                value = F.adaptive_avg_pool2d(target[None, None], (28, 28))[0, 0]
            else:
                parts = name.split()
                state = STATE_START if parts[0] == "start" else STATE_END
                factor = {"3um": 1, "1.5um": 2, ".75um": 4}[parts[1]]
                value = examples[row][f"{state}_f{factor}"]
            image_value = value.clamp(0, 1)
            array = (image_value.numpy() * 255.0).round().astype(np.uint8)
            tile_image = Image.fromarray(array, mode="L").resize((tile - 2, tile - 2), Image.Resampling.NEAREST)
            image.paste(tile_image, (label_w + col * tile, y + 12))
    image.save(path)
    return path.name


def _write_report(path: Path, report: dict[str, Any]) -> None:
    decision = report["sampling_decision_detail"]
    summary = report["summary_by_state_and_factor"]
    rows = ["|状态|采样|MSE400|PSNR28|SSIM28|饱和|模态功率(W)|耦合效率|",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for state in STATES:
        for factor in FACTORS:
            item = summary[state][str(factor)]
            rows.append(f"|{state}|{PIXEL_PITCH_M / factor * 1e6:g} μm|{item['mean_mse400']:.6g}|"
                        f"{item['mean_psnr28_db']:.4f}|{item['mean_ssim28']:.5f}|"
                        f"{item['mean_saturation_fraction']:.4%}|{item['mean_modal_power_w']:.6g}|"
                        f"{item['mean_system_coupling_efficiency']:.5%}|")
    reference = decision["finite_pixel_rayleigh_sommerfeld"]
    _require(reference.get("production_comparison_executed") is True
             and reference.get("physical_window_matches_production") is True,
             "final report requires the full-window production RS comparison")
    rs_max = max(
        item["asm_vs_real_space_rs_relative_l2_after_global_phase"]
        for item in reference["patterns"].values()
    )
    legacy = report["comparison_reference"]["analytic_physical_pixel_observer"]["legacy_comparator_counterexample"]
    rs_rows = ["|输入相位|Gauss 8→12相对变化|角谱对RS相对误差|相位导数相对误差|",
               "|---|---:|---:|---:|"]
    for name, item in reference["patterns"].items():
        rs_rows.append(
            f"|{name}|{item['gauss_order_8_vs_12_relative_l2']:.3g}|"
            f"{item['asm_vs_real_space_rs_relative_l2_after_global_phase']:.6g}|"
            f"{item['phase_derivative_relative_error']:.6g}|"
        )
    text = (
        "# LightGen400：纠正数值判定与重建目标对照前置资格\n\n"
        "## 结果与状态\n\n"
        f"状态为 `{report['status']}`，本轮结论为 `{report['decision']}`。优化器更新0；正式采样比较覆盖两状态、"
        "10张固定TRAIN图、三个采样间距，复用统一读出。逐根功率及重建指标见下表。\n\n"
        + "\n".join(rows) + "\n\n"
        "## 科学认识\n\n"
        "上一轮复场比较器把不同宽度的样本均值当作相同物理观测量。相同周期8 μm的解析平面波上，旧方法仍给出"
        f"2×/4×中心值误差`{legacy['factor_2_vs_4_center_relative_l2']:.2%}`、"
        f"1×/4×像素平均误差`{legacy['factor_1_vs_4_area_relative_l2']:.2%}`。新方法先在全物理窗口频谱求值，"
        f"再取相同3 μm像素中心或用sinc计算同一像素单元平均；解析参考最大误差为"
        f"`{max(report['comparison_reference']['analytic_physical_pixel_observer']['analytic_errors'].values()):.3g}`。"
        f"修正后跨网格复场误差上限为1×/4× `{decision['max_complex_error_factor1_vs4']:.6g}`、"
        f"2×/4× `{decision['max_complex_error_factor2_vs4']:.6g}`；OLS系数误差仍为"
        f"`{decision['max_coefficient_error_factor1_vs4']:.6g}` / `{decision['max_coefficient_error_factor2_vs4']:.6g}`。"
        f"此前有限像素参考错误使用`0.384 mm`周期窗口，传播距离为`3 mm`，该对照会混入周期回绕，故撤回其残差。"
        f"本轮改用`{reference['physical_window_m'] * 1e3:.3f} mm`完整生产窗口（网格`{reference['grid'][0]}×{reference['grid'][1]}`）；"
        "直接有限像素Rayleigh–Sommerfeld参考的Gauss积分自身收敛，生产采样路径与参考的最大相对场误差为"
        f"`{rs_max:.6g}`，超过原`1e-4`尺度。逐图的独立参考如下：\n\n"
        + "\n".join(rs_rows) + "\n\n"
        "因此当前没有可推荐的训练数值路径。此结论约束当前离散实现，不证明400×400架构容量不足。\n\n"
        "## 目标缺口\n\n"
        "六层A/B各500步未启动。尚无经过数值资格的重建目标对照，也没有完整TRAIN基线、DEV验收、生成质量或人眼可读性结论；"
        "Gaussian MFD、光纤pitch和identity传输仍属项目假设。\n\n"
        "## 工程完成与验证\n\n"
        "新复场观测器的解析平面波与物理坐标回归通过；CMOS继续采用`|E|²`后作像素面积积分。"
        "独立RS参考覆盖Gaussian、倾斜相位和棋盘相位32×32像素输入，Gauss阶数8到12的最大变化小于`1e-4`；"
        "生产角谱路径值或相位导数未达到参考尺度。复用既有60组任务读数及功率，新增复场观测因旧缓存缺失而单独计数；"
        "有限差分仅覆盖两状态、0类、3 μm路径和六层，不能写成所有采样路径的完整差分验证。旧报告原始结果保留，"
        "本报告只更正其复场误差归因。\n\n"
        "## 决策影响与下一步\n\n"
        "当前先修复或明确生产传播对有限物理像素的数值表示，再重跑受影响的同一资格检查。资格未通过前，结构目标A/B训练预算保持0；"
        "若修复后仍无候选，则记录`NUMERICAL_NOT_QUALIFIED`并停止。原30,000步基线和500步终点身份均不改变。\n"
    )
    path.write_text(text, encoding="utf-8")
    saved = path.read_text(encoding="utf-8")
    for heading in ("结果与状态", "科学认识", "目标缺口", "工程完成与验证", "决策影响与下一步"):
        _require(f"## {heading}" in saved, f"final report omitted section {heading}")


def _make_ready(ctx: dict[str, Any], output: Path, checks: dict[str, Any]) -> dict[str, Any]:
    relevant = (
        Path(__file__), Path("scripts/research/lightgen/execute_chen_lightgen_400_mnist_baseline_v1.py"),
        Path("scripts/research/lightgen/execute_chen_lightgen_400_joint_reconstruction_v1.py"),
        Path("src/ldm_project/v2/application/research/lightgen/chen_lightgen_400_baseline.py"),
        Path("src/ldm_project/v2/application/research/lightgen/gaussian_fiber_baseline.py"),
        Path("src/ldm_project/v2/optics/propagation.py"), _resolve(ctx["args"].config),
    )
    manifest = capture_execution_source_manifest(
        ROOT, task_id=TASK_ID, semantic_authority_id=f"{TASK_ID}:FROZEN_2STATES_10ROWS_3SAMPLINGS",
        relevant_paths=relevant,
        config_identity={"rows": ctx["rows"].tolist(), "factors": list(FACTORS), "grids": GRIDS,
                         "gain": GAIN, "optimizer_updates": 0},
    )
    run_id = uuid.uuid4().hex
    identity = ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id,
        trajectory_id=f"{TASK_ID}:joint_endpoint_500:10_fixed_train_rows",
        semantic_authority_id=f"{TASK_ID}:ZERO_UPDATE_3SAMPLING",
        data_role="canonical_MNIST_TRAIN_fixed_first_confirmation_row_per_class_no_test_reads",
        config_identity={"factors": FACTORS, "grids": GRIDS, "physical_phase_pitch_m": PIXEL_PITCH_M,
                         "roi_m": ROI_M, "gain": GAIN, "updates": 0},
        model_identity={"states": STATES, "phase_shapes": [[2, 400, 400], [4, 400, 400]],
                        "ols": "gaussian_s12_100_ports_project_geometry_unmeasured"},
        optimizer_identity={"name": "none", "updates": 0},
        sampler_identity={"fixed_confirmation_rows": ctx["rows"].tolist(), "sample_count": 10},
        execution_source_semantics_id=manifest.execution_source_semantics_id,
        source_manifest_id=manifest.source_manifest_id,
    )
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    correctness_checks = {
        "source_lineage_and_fixed_train_population": {"passed": checks["source_identity"]["all_rows_train"]
            and checks["source_identity"]["official_test_reads"] == 0},
        "white_gray_black_input_integrals": {"passed": len(checks["white_gray_black_power_w"]) == 9},
        "gaussian_match_and_gram_passivity": {"passed": all(
            value["passed"] for value in checks["gaussian_by_factor"].values())},
        "camera_area_integral_counterexample": {"passed": checks["camera_quadrature_counterexample"]["distinguished"]},
        "same_coordinate_pixel_observer": {"passed": checks["analytic_physical_pixel_observer"]["passed"]},
        "analytic_and_independent_propagation_reference": {"passed": checks["angular_spectrum_plane_wave_reference"]["passed"]},
        "gradient_and_finite_difference": {"passed": all(
            value["passed"] for value in checks["finite_difference_checks"].values())
            and all(math.isfinite(value["gradient_norm"])
                    for result in checks["gradient_checks"].values()
                    for layers in result["comparisons"].values()
                    for factor, value in layers.items() if factor in ("factor_2", "factor_4"))},
        "production_signal_checkpoint_recovery": {"passed": checks["recovery"]["production_forward_completed"]
            and checks["recovery"]["new_optical_forwards_during_resume"] == 0},
        "frozen_model_and_zero_optimizer_updates": {"passed": checks["frozen_scope"]["all_parameters_frozen"]
            and checks["frozen_scope"]["optimizer_updates"] == 0},
        "representative_cuda_production_forward": {"passed": checks["representative_production_measurements"]["all_finite"]},
    }
    correctness = _ready_correctness_group(correctness_checks)
    resource = {"passed": True, "cuda_device": torch.cuda.get_device_name(ctx["device"]),
                "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(ctx["device"]),
                "preflight_seconds": checks["elapsed_seconds"]}
    implementation = {"passed": True, "engine": "cuda_fixed_phase_4096_8192_16384_pixel_sampling_1_2_4",
                      "selected_engine": "cuda_fixed_phase_4096_8192_16384_pixel_sampling_1_2_4",
                      "trainable_groups": [], "frozen_groups": ["all_phase_parameters", "OLS_geometry", "gain", "evaluator"]}
    receipt = control.make_ready_receipt(correctness=correctness, resource=resource,
                                         implementation=implementation,
                                         selected_engine=implementation["selected_engine"],
                                         resource_envelope={"cuda_device": torch.cuda.get_device_name(ctx["device"]),
                                                            "peak_reserved_vram_bytes": resource["peak_reserved_vram_bytes"]})
    control.mark_preflight_ready(receipt)
    result = {"schema": "chen_lightgen_400_sampling_preflight.v1", "status": "READY",
              "task_id": TASK_ID, "authorization_id": AUTHORITY, "run_identity": identity.to_dict(),
              "receipt": receipt.to_dict(), "fixed_rows": ctx["rows"].tolist(),
              "fixed_labels": ctx["selected_labels"].tolist(), "checks": checks,
              "optimizer_updates": 0, "official_test_reads": 0}
    atomic_write_new_json(output / "preflight_ready.json", receipt.to_dict())
    return result


def _verify_ready(args: argparse.Namespace) -> dict[str, Any]:
    preflight_dir = _resolve(args.preflight_dir)
    preflight, receipt = _json(preflight_dir / "preflight.json"), _json(preflight_dir / "preflight_ready.json")
    _require(preflight.get("status") == "READY" and preflight.get("task_id") == TASK_ID
             and preflight.get("authorization_id") == AUTHORITY and preflight.get("receipt") == receipt
             and receipt.get("status") == "READY" and receipt.get("authorization_id") == AUTHORITY,
             "a matching READY receipt from this diagnostic is required")
    for name in ("start_state", "endpoint_checkpoint", "fixed_population", "source_manifest",
                 "baseline_checkpoint", "baseline_report", "joint_report", "data", "config"):
        _require(preflight["input_paths"][name] == str(_resolve(getattr(args, name))),
                 f"run input {name} differs from the READY preflight")
    return preflight


def _measurement_recovery_probe(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    start_path = output / "signal_started.json"
    process = subprocess.Popen(_module_command(
        "--mode", "recovery-worker",
        "--baseline-checkpoint", str(_resolve(args.baseline_checkpoint)),
        "--baseline-report", str(_resolve(args.baseline_report)), "--joint-report", str(_resolve(args.joint_report)),
        "--source-manifest", str(_resolve(args.source_manifest)), "--fixed-population", str(_resolve(args.fixed_population)),
        "--data", str(_resolve(args.data)), "--config", str(_resolve(args.config)),
        "--start-state", str(_resolve(args.start_state)), "--endpoint-checkpoint", str(_resolve(args.endpoint_checkpoint)),
        "--recovery-dir", str(output),
    ), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline_notice = time.monotonic() + 60.0
    while not start_path.is_file():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise RuntimeError(f"recovery worker exited before production measurement: {stdout} {stderr}")
        if time.monotonic() >= deadline_notice:
            atomic_write_json(output / "heartbeat.json", {"status": "WAITING_FOR_PRODUCTION_FORWARD", "updated_at": time.time()})
            deadline_notice = time.monotonic() + 60.0
        time.sleep(0.2)
    started = _json(start_path)
    _require(started.get("status") == "PRODUCTION_FORWARD_ACTIVE"
             and started.get("first_completed_plane") == "encoder_1",
             "signal probe marker was not emitted from a real production field callback")
    process.send_signal(signal.SIGTERM)
    while process.poll() is None:
        if time.monotonic() >= deadline_notice:
            atomic_write_json(output / "heartbeat.json", {"status": "WAITING_FOR_SIGNALLED_BLOCK", "updated_at": time.time()})
            deadline_notice = time.monotonic() + 60.0
        time.sleep(0.5)
    stdout, stderr = process.communicate()
    _require(process.returncode == 0, f"signal recovery worker failed: {stdout} {stderr}")
    block_path = output / "recovery_sample.pt"
    block = torch.load(block_path, map_location="cpu", weights_only=False)
    _require(block.get("signal_received") == signal.SIGTERM and block.get("completed_production_forward") is True,
             "SIGTERM did not complete and save the production measurement block")
    resume = subprocess.run(_module_command(
        "--mode", "recovery-resume",
        "--recovery-block", str(block_path), "--recovery-dir", str(output),
    ), cwd=ROOT, capture_output=True, text=True, check=False)
    _require(resume.returncode == 0, f"independent recovery consumer rejected saved block: {resume.stdout} {resume.stderr}")
    resumed = _json(output / "resume_validation.json")
    _require(resumed.get("status") == "RESUMED_WITHOUT_RECOMPUTE"
             and resumed.get("new_optical_forwards") == 0 and resumed.get("row_consumed_once") is True,
             "independent process recovery repeated or omitted a completed optical row")
    return {"signal": "SIGTERM", "production_forward_completed": True,
            "saved_block_readback": True, "independent_process_resume": resumed,
            "new_optical_forwards_during_resume": 0}


def _recovery_worker(args: argparse.Namespace) -> int:
    output = _resolve(args.recovery_dir)
    flag = [None]
    signal.signal(signal.SIGTERM, lambda signum, _frame: flag.__setitem__(0, signum))
    _require(torch.cuda.is_available(), "signal recovery probe requires CUDA")
    ctx = _load_context(args, torch.device("cuda"))
    row = int(ctx["rows"][0])
    by_factor = {}
    for factor in FACTORS:
        def marker(name):
            atomic_write_new_json(output / "signal_started.json", {
                "status": "PRODUCTION_FORWARD_ACTIVE", "first_completed_plane": name, "updated_at": time.time()
            })

        measured, _, _ = _one_forward(ctx, STATE_START, factor, row,
                                      on_first_field=marker if factor == FACTORS[0] else None)
        by_factor[str(factor)] = measured
    block = {"schema": "chen_lightgen_400_sampling_recovery_probe.v1",
             "identity": _measurement_identity(ctx), "row": row,
             "label": int(ctx["labels"][row]), "by_factor": by_factor,
             "signal_received": flag[0], "completed_production_forward": True}
    atomic_torch_save_new(output / "recovery_sample.pt", block)
    atomic_write_json(output / "worker_completed.json", {"signal_received": flag[0], "row": row})
    return 0


def _recovery_resume(args: argparse.Namespace) -> int:
    block = torch.load(_resolve(args.recovery_block), map_location="cpu", weights_only=False)
    _require(block.get("schema") == "chen_lightgen_400_sampling_recovery_probe.v1"
             and block.get("completed_production_forward") is True
             and block.get("signal_received") == signal.SIGTERM
             and block.get("row") == block.get("identity", {}).get("fixed_rows", [None])[0]
             and block.get("label") == block.get("identity", {}).get("fixed_labels", [None])[0],
             "recovery block is incomplete or has no observed SIGTERM")
    _require(set(block.get("by_factor", {})) == {str(factor) for factor in FACTORS},
             "recovery block factors are incomplete")
    for entry in block["by_factor"].values():
        _require(torch.isfinite(entry["coefficients"]).all()
                 and entry["physical_w_per_pixel"].shape == (1, 1, 400, 400),
                 "recovery block output failed independent process readback")
    atomic_write_new_json(_resolve(args.recovery_dir) / "resume_validation.json", {
        "status": "RESUMED_WITHOUT_RECOMPUTE", "row": block["row"], "label": block["label"],
        "row_consumed_once": True, "new_optical_forwards": 0,
    })
    return 0


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for this numerical qualification")
    output = _resolve(args.output_dir)
    _require(not output.exists(), f"refusing to overwrite preflight directory {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    heartbeat = _start_heartbeat(output, "PREFLIGHT_RUNNING", torch.device("cuda"))
    recovery = _measurement_recovery_probe(args, output / "signal_recovery")
    ctx = _load_context(args, torch.device("cuda"))
    _require(torch.cuda.get_device_name(ctx["device"]).startswith("NVIDIA"),
             "sampling qualification requires an NVIDIA CUDA device")
    checks: dict[str, Any] = {
        "source_identity": {"baseline_step": 30_000, "joint_endpoint_step": 500,
                            "rows": ctx["rows"].tolist(), "labels": ctx["selected_labels"].tolist(),
                            "all_rows_train": True, "official_test_reads": 0},
        "white_gray_black_power_w": _white_gray_black_integral(),
        "analytic_physical_pixel_observer": _analytic_physical_pixel_observer_check(),
        "finite_pixel_rayleigh_sommerfeld": _rayleigh_sommerfeld_pixel_reference_check(
            ctx["device"], compare_production=True),
        "camera_quadrature_counterexample": _complex_quadrature_reference(),
        "angular_spectrum_plane_wave_reference": _propagation_reference_check(),
        "gaussian_by_factor": {str(factor): _gaussian_check(ctx["models"][(STATE_START, factor)])
                               for factor in FACTORS},
        "recovery": recovery,
        "frozen_scope": {"all_parameters_frozen": all(
            not parameter.requires_grad for model in ctx["models"].values()
            for parameter in model.parameters()), "optimizer_updates": 0},
    }
    rows_to_test = {0: int(ctx["rows"][0]), 1: int(ctx["rows"][1])}
    short_comparisons = {}
    for state in STATES:
        for label, row in rows_to_test.items():
            variants, fields = {}, {}
            for factor in FACTORS:
                variants[factor], fields[factor], _ = _one_forward(ctx, state, factor, row)
            short_comparisons[f"{state}_class_{label}"] = _compare_sample(variants, fields)
    checks["representative_production_measurements"] = short_comparisons
    checks["gradient_checks"] = _gradient_checks(ctx)
    checks["finite_difference_checks"] = _finite_difference_checks(ctx)
    max_pre_field = max(
        detail["max_relative_l2"]
        for case in short_comparisons.values()
        for name, detail in case["field_checks"].items() if name.startswith("factor_2_vs_4")
    )
    max_pre_coeff = max(
        detail["complex_relative_l2_after_global_phase"]
        for block in short_comparisons.values()
        for name, detail in block["coefficient_checks"].items() if name == "factor_2_vs_4"
    )
    max_pre_grad = max(
        value["relative_l2"]
        for record in checks["gradient_checks"].values()
        for layers in record["comparisons"].values()
        for name, value in layers.items() if name == "factor_2_vs_factor_4"
    )
    representative_all_finite = all(
        math.isfinite(detail["max_relative_l2"])
        for case in short_comparisons.values()
        for detail in case["field_checks"].values()
    ) and all(
        math.isfinite(detail["complex_relative_l2_after_global_phase"])
        for case in short_comparisons.values()
        for detail in case["coefficient_checks"].values()
    ) and all(
        math.isfinite(value)
        for case in short_comparisons.values()
        for detail in case["camera_task_checks"].values()
        for value in detail.values()
    )
    _require(representative_all_finite, "representative production comparison contains nonfinite values")
    if max_pre_field <= WINDOW_TOLERANCE and max_pre_coeff <= COEFFICIENT_TOLERANCE \
            and max_pre_grad <= GRADIENT_TOLERANCE:
        checks["factor_2_window_expansion"] = _window_expansion_checks(ctx, rows_to_test)
    else:
        checks["factor_2_window_expansion"] = {
            "required": False, "passed": False, "status": "NOT_REQUIRED_PRELIMINARY_1P5UM_DID_NOT_CONVERGE",
            "preliminary_max_field_error": max_pre_field,
            "preliminary_coefficient_error": max_pre_coeff,
            "preliminary_gradient_error": max_pre_grad,
        }
    elapsed = time.perf_counter() - started
    checks["elapsed_seconds"] = elapsed
    checks["measurement_forwards"] = 3 * len(rows_to_test) * len(STATES)
    checks["gradient_forwards"] = 12
    checks["finite_difference_forwards"] = 48
    checks["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved(ctx["device"])
    checks["representative_production_measurements"] = {
        "records": short_comparisons, "all_finite": representative_all_finite}
    preflight = _make_ready(ctx, output, checks)
    preflight["input_paths"] = {name: str(_resolve(getattr(args, name))) for name in (
        "baseline_checkpoint", "baseline_report", "joint_report", "source_manifest", "fixed_population",
        "data", "config", "start_state", "endpoint_checkpoint")}
    atomic_write_json(output / "preflight.json", preflight)
    atomic_torch_save_new(output / "fixed_rows.pt", {"rows": ctx["rows"], "labels": ctx["selected_labels"]})
    _stop_heartbeat(heartbeat)
    return preflight


def _run(args: argparse.Namespace) -> dict[str, Any] | None:
    _require(torch.cuda.is_available(), "AutoDL CUDA is required for the sampling measurement")
    preflight = _verify_ready(args)
    run_dir = _resolve(args.output_dir)
    ctx = _load_context(args, torch.device("cuda"))
    resume_checkpoint = _resolve(args.resume_checkpoint) if args.mode == "resume" else None
    _require((args.mode == "resume") == (resume_checkpoint is not None),
             "resume requires one explicit --resume-checkpoint path")
    sampling_resume = resume_checkpoint
    if resume_checkpoint is not None:
        resume_state = torch.load(resume_checkpoint, map_location="cpu", weights_only=False)
        if resume_state.get("schema") == "chen_lightgen_400_decoder_sampling_recovery.v1":
            _require(resume_state.get("identity") == _decoder_probe_identity(ctx),
                     "fixed-coefficient Decoder recovery belongs to different inputs")
            sampling_resume = Path(resume_state["sampling_checkpoint"])
    run_started = time.perf_counter()
    heartbeat = _start_heartbeat(run_dir, "MEASUREMENT_RUNNING", ctx["device"])
    outcome = _run_samples(ctx, run_dir, resume_checkpoint=sampling_resume)
    if outcome["status"] == "INTERRUPTED":
        _stop_heartbeat(heartbeat)
        atomic_write_json(run_dir / "interruption.json", outcome)
        return outcome
    decoder_outcome = _run_fixed_coefficient_decoder_probes(ctx, run_dir)
    if decoder_outcome["status"] == "INTERRUPTED":
        _stop_heartbeat(heartbeat)
        atomic_write_json(run_dir / "interruption.json", decoder_outcome)
        return decoder_outcome
    report = _aggregate(ctx, run_dir, preflight["checks"])
    report["preflight_status"] = preflight["status"]
    report["preflight_dir"] = str(_resolve(args.preflight_dir))
    report["measurement_elapsed_seconds"] = time.perf_counter() - run_started
    report["decoder_probe_status"] = decoder_outcome["status"]
    _write_report(run_dir / "results_analysis.md", report)
    atomic_write_new_json(run_dir / "report.json", report)
    _stop_heartbeat(heartbeat)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "resume", "reanalyze",
                                             "recovery-worker", "recovery-resume"), required=True)
    for name in ("baseline-checkpoint", "baseline-report", "joint-report", "source-manifest", "fixed-population",
                 "data", "config", "start-state", "endpoint-checkpoint", "output-dir", "preflight-dir",
                 "resume-checkpoint", "source-run-dir", "recovery-dir", "recovery-block"):
        parser.add_argument(f"--{name}", type=Path, dest=name.replace("-", "_"))
    args = parser.parse_args()
    if args.mode in ("recovery-worker", "recovery-resume"):
        _require(args.recovery_dir is not None, "recovery process requires an explicit --recovery-dir")
        if args.mode == "recovery-resume":
            _require(args.recovery_block is not None, "recovery consumer requires --recovery-block")
        else:
            required = ("baseline_checkpoint", "baseline_report", "joint_report", "source_manifest", "fixed_population",
                        "data", "config", "start_state", "endpoint_checkpoint")
            _require(all(getattr(args, name) is not None for name in required), "recovery worker requires explicit source inputs")
        return args
    required = ("baseline_checkpoint", "baseline_report", "joint_report", "source_manifest", "fixed_population",
                "data", "config", "start_state", "endpoint_checkpoint", "output_dir")
    _require(all(getattr(args, name) is not None for name in required), "every model/data/population/output path must be explicit")
    _require((args.mode == "preflight" and args.preflight_dir is None) or
             (args.mode in ("run", "resume", "reanalyze") and args.preflight_dir is not None),
             "preflight omits --preflight-dir; run/resume/reanalyze require it")
    _require((args.mode == "resume") == (args.resume_checkpoint is not None),
             "resume requires an explicit --resume-checkpoint")
    _require((args.mode == "reanalyze") == (args.source_run_dir is not None),
             "reanalyze requires --source-run-dir and other modes reject it")
    return args


def main() -> None:
    args = _parse_args()
    if args.mode == "recovery-worker":
        _recovery_worker(args)
        return
    if args.mode == "recovery-resume":
        _recovery_resume(args)
        return
    if args.mode == "preflight":
        report = _preflight(args)
    elif args.mode == "reanalyze":
        report = _reanalyze_cached_run(args)
    else:
        report = _run(args)
    print(json.dumps({key: report.get(key) for key in ("status", "decision", "optimizer_updates")
                      if key in report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
