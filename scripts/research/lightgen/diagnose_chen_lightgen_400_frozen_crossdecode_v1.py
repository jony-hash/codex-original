"""Frozen four-way Encoder/Decoder cross-decode audit for LightGen400."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt,
    ResearchRunControl,
    ResearchRunIdentity,
    capture_execution_source_manifest,
)
from ldm_project.v2.evidence import (
    atomic_torch_save,
    atomic_torch_save_new,
    atomic_write_json,
    atomic_write_new_json,
)
from scripts.research.lightgen import (
    diagnose_chen_lightgen_400_readout_structure_audit_v1 as readout,
)
from scripts.research.lightgen import (
    diagnose_chen_lightgen_400_quickcheck_v1 as quickcheck,
)
from scripts.research.lightgen import (
    execute_chen_lightgen_400_mnist_baseline_v1 as baseline,
)

ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CHEN_LIGHTGEN_400_FROZEN_CROSSDECODE_V1"
AUTHORITY = "USER_APPROVED_CHEN_LIGHTGEN_400_FROZEN_CROSSDECODE_20261017"
GRID = 4096
REFERENCE_GRID = 6144
WINDOW_TOLERANCE = 1.0e-4
TARGET_REPLAY_ATOL = 2.0e-6
BLOCK_SIZE = 10
SATURATION_MAX = 0.01
MSE_IMPROVEMENT_MIN = 0.10
STATE0 = "start_4096"
STATE1 = "endpoint_500"
COMBOS = {
    "R00": {"encoder": "E0", "decoder": "D0", "coefficient_state": STATE0},
    "R01": {"encoder": "E0", "decoder": "D1", "coefficient_state": STATE0},
    "R10": {"encoder": "E1", "decoder": "D0", "coefficient_state": STATE1},
    "R11": {"encoder": "E1", "decoder": "D1", "coefficient_state": STATE1},
}
CROSS_COMBOS = ("R01", "R10")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"expected JSON object at {path}")
    return value


def _input_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint=args.checkpoint,
        source_report=args.baseline_report,
        start_state=args.start_state,
        endpoint_checkpoint=args.endpoint_checkpoint,
        joint_report=args.joint_report,
        fixed_population=args.fixed_population,
        config=args.config,
        data=args.data,
        evaluator=args.evaluator,
    )


def _load_context(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    source_dir = _resolve(args.source_audit_dir)
    source_manifest = _json(source_dir / "run_manifest.json")
    source_preflight_dir = _resolve(args.source_preflight_dir)
    source_preflight = _json(source_preflight_dir / "preflight.json")
    source_ready = _json(source_preflight_dir / "preflight_ready.json")
    readout_report = _json(_resolve(args.readout_report))
    _require(
        source_manifest.get("task_id") == readout.TASK_ID
        and source_manifest.get("sample_count") == 300
        and source_manifest.get("grid") == GRID
        and source_manifest.get("optimizer_updates") == 0,
        "source audit manifest is not the explicit 300-row, zero-update, grid-4096 audit",
    )
    _require(
        str(source_manifest.get("start_state", "")).endswith("/start_model_state.pt")
        and str(source_manifest.get("endpoint_checkpoint", "")).endswith("/checkpoint_step_000500.pt")
        and str(source_manifest.get("fixed_population", "")).endswith("/fixed_population.pt"),
        "source audit manifest does not identify the named start, endpoint and population artifacts",
    )
    _require(
        source_preflight.get("status") == "READY"
        and source_preflight.get("task_id") == readout.TASK_ID
        and source_ready.get("status") == "READY",
        "source audit preflight lineage is incomplete",
    )
    _require(
        readout_report.get("status") == "VALID_COMPLETED"
        and readout_report.get("task_id") == readout.TASK_ID
        and readout_report.get("completed_updates") == 0,
        "readout audit report is not the completed zero-update source report",
    )

    loaded = readout._load_inputs(_input_args(args), device)
    population = loaded["population"]
    expected_rows = population["all_rows"].long()
    states: dict[str, dict[str, torch.Tensor]] = {}
    target_preprocessing_max_abs_error = 0.0
    for state in (STATE0, STATE1):
        state_dir = source_dir / state
        paths = sorted(state_dir.glob("block_*.pt"))
        _require(len(paths) == 30, f"{state} source must contain 30 ten-row blocks")
        blocks = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
        for index, block in enumerate(blocks):
            start = index * BLOCK_SIZE
            end = start + BLOCK_SIZE
            _require(
                torch.equal(block["rows"], expected_rows[start:end])
                and torch.equal(block["labels"], population["all_labels"][start:end]),
                f"{state} source block row or label mismatch: {paths[index]}",
            )
            canonical_target = readout._entry_pixels(
                loaded["pixels"], expected_rows[start:end], torch.device("cpu")
            )
            target_error = float((block["target"] - canonical_target).abs().max())
            target_preprocessing_max_abs_error = max(
                target_preprocessing_max_abs_error, target_error
            )
            _require(
                torch.allclose(
                    block["target"], canonical_target, rtol=0.0, atol=TARGET_REPLAY_ATOL
                ),
                f"{state} source block target differs from canonical TRAIN rows beyond CPU/CUDA interpolation tolerance",
            )
            _require(
                block["ols_coefficients"].shape == (BLOCK_SIZE, 100)
                and torch.is_complex(block["ols_coefficients"])
                and torch.isfinite(block["ols_coefficients"]).all()
                and torch.isfinite(block["physical_w_per_pixel"]).all(),
                f"{state} source block has invalid physical fields or coefficients",
            )
        keys = (
            "rows", "labels", "target", "physical_w_per_pixel", "ols_coefficients",
            "modal_power_w", "slm_before_modulation_power_w",
            "slm_after_modulation_power_w", "ols_incident_power_w", "ols_modal_sum_w",
            "camera_roi_power_w", "seven_plane_power_w",
        )
        state_data = {key: torch.cat([block[key] for block in blocks], dim=0) for key in keys}
        evaluation = torch.load(
            source_dir / f"evaluation_{state}.pt", map_location="cpu", weights_only=False
        )
        _require(
            torch.equal(evaluation["coefficients"], state_data["ols_coefficients"])
            and torch.equal(
                evaluation["physical_w_per_pixel"], state_data["physical_w_per_pixel"]
            ),
            f"{state} cached evaluation differs from production measurement blocks",
        )
        states[state] = state_data
        del blocks, evaluation

    _require(
        torch.equal(states[STATE0]["rows"], population["all_rows"])
        and torch.equal(states[STATE1]["rows"], population["all_rows"])
        and torch.equal(states[STATE0]["target"], states[STATE1]["target"])
        and torch.equal(states[STATE0]["labels"], states[STATE1]["labels"]),
        "diagonal source states do not share the same fixed rows, labels and targets",
    )
    _require(
        torch.equal(
            torch.load(
                _resolve(args.fixed_population), map_location="cpu", weights_only=False
            )["all_rows"],
            population["all_rows"],
        ),
        "explicit fixed population differs from the source audit population",
    )
    loaded.update(
        {
            "source_audit_dir": source_dir,
            "source_preflight_dir": source_preflight_dir,
            "source_manifest": source_manifest,
            "source_preflight": source_preflight,
            "readout_report": readout_report,
            "states": states,
            "target_preprocessing_max_abs_error": target_preprocessing_max_abs_error,
        }
    )
    return loaded


def _current_models(loaded: dict[str, Any]) -> dict[str, torch.nn.Module]:
    models = loaded["models"]
    return {"D0": models[STATE0], "D1": models[STATE1]}


def _common_field_errors(
    candidate: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]
) -> tuple[dict[str, float], float]:
    return baseline._complex_relative_errors(candidate, reference)


def _cross_window_probe(
    decoder: torch.nn.Module,
    coefficients: torch.Tensor,
    roi_m: float,
) -> dict[str, Any]:
    per_grid: dict[int, dict[str, torch.Tensor]] = {}
    for grid in (GRID, REFERENCE_GRID):
        model = quickcheck._diagnostic_model(decoder, grid)
        model.requires_grad_(False)
        with torch.no_grad():
            result = model.forward_from_ols(coefficients.to(next(model.parameters()).device), return_diagnostics=True)
        roi_pixels = int(round(roi_m / model.config.pixel_pitch_m))
        fields = {
            "decoder_entry": result["generator_entry_field"].tensor,
            **{
                f"decoder_{index + 1}": field.tensor
                for index, field in enumerate(result["decoder_fields"])
            },
        }
        per_grid[grid] = {
            name: baseline._common_roi(field, roi_pixels).detach().cpu()
            for name, field in fields.items()
        }
        del result, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    errors, phase = _common_field_errors(per_grid[GRID], per_grid[REFERENCE_GRID])
    maximum = max(errors.values())
    return {
        "per_plane_errors": errors,
        "aligned_global_phase_rad": phase,
        "maximum_error": maximum,
        "passed": maximum <= WINDOW_TOLERANCE,
    }


def _ssim_parts(image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, ...]:
    coordinate = torch.arange(-3, 4, dtype=torch.float64, device=image.device)
    one = torch.exp(-coordinate.square() / (2.0 * 1.5**2))
    one /= one.sum()
    kernel = (one[:, None] * one[None, :]).to(image.dtype)[None, None]
    mu_x = F.conv2d(image, kernel, padding=3)
    mu_y = F.conv2d(target, kernel, padding=3)
    var_x = F.conv2d(image.square(), kernel, padding=3) - mu_x.square()
    var_y = F.conv2d(target.square(), kernel, padding=3) - mu_y.square()
    cov_xy = F.conv2d(image * target, kernel, padding=3) - mu_x * mu_y
    luminance = (2.0 * mu_x * mu_y + 0.01**2) / (
        mu_x.square() + mu_y.square() + 0.01**2
    )
    contrast_structure = (2.0 * cov_xy + 0.03**2) / (var_x + var_y + 0.03**2)
    return luminance, contrast_structure, luminance * contrast_structure


def _ssim_symmetric_decomposition(
    luminance0: torch.Tensor, contrast0: torch.Tensor,
    luminance1: torch.Tensor, contrast1: torch.Tensor,
) -> dict[str, float]:
    luminance_term = float(((luminance1 - luminance0) * (contrast1 + contrast0) / 2.0).mean())
    contrast_term = float(((contrast1 - contrast0) * (luminance1 + luminance0) / 2.0).mean())
    delta = float((luminance1 * contrast1).mean() - (luminance0 * contrast0).mean())
    error = abs(luminance_term + contrast_term - delta)
    _require(error <= 1.0e-9, "SSIM symmetric decomposition does not reconstruct the map mean")
    return {
        "delta_ssim": delta,
        "luminance_contribution": luminance_term,
        "contrast_structure_contribution": contrast_term,
        "reconstructed_delta": luminance_term + contrast_term,
        "absolute_error": error,
    }


def _normalized_shape_tv(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_mass = prediction.sum((-2, -1), keepdim=True)
    target_mass = target.sum((-2, -1), keepdim=True)
    _require(bool((prediction_mass > 0).all()) and bool((target_mass > 0).all()), "shape TV needs positive image mass")
    return 0.5 * (
        prediction / prediction_mass - target / target_mass
    ).abs().sum((-2, -1))


def _sobel_metrics(image: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, ...]:
    gx = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8.0
    gy = gx.T
    kernels = torch.stack((gx, gy)).unsqueeze(1)
    image_gradient = F.conv2d(image, kernels, padding=1)
    target_gradient = F.conv2d(target, kernels, padding=1)
    image_energy = image_gradient.square().sum(1)
    target_energy = target_gradient.square().sum(1)
    gradient_error = (image_gradient - target_gradient).square().sum(1)
    return image_energy, target_energy, gradient_error


def _independent_gain_reference(physical: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    import numpy as np

    w = physical.detach().cpu().double().numpy().ravel()
    y = target.detach().cpu().double().numpy().ravel()
    _require(w.size == y.size and np.isfinite(w).all() and np.isfinite(y).all(), "invalid independent gain values")
    denom = float(np.dot(w, w))
    _require(denom > 0 and math.isfinite(denom), "independent gain denominator is zero")
    ls = float(np.dot(w, y)) / denom
    allowed = math.floor(SATURATION_MAX * w.size)
    index = w.size - allowed - 1
    threshold = float(np.partition(w, index)[index])
    cap = math.inf if threshold <= 0 else 1.0 / threshold
    return {"ls": ls, "cap": cap, "constrained": min(ls, cap), "allowed_exceedances": allowed, "threshold": threshold}


def _verify_metric_references(loaded: dict[str, Any]) -> dict[str, Any]:
    device = next(loaded["models"][STATE0].parameters()).device
    generator = torch.Generator(device=device).manual_seed(2026101701)
    target = torch.rand((1, 1, 28, 28), device=device, dtype=torch.float64, generator=generator)
    cases = {
        "identical": target,
        "brightness_0_8": target * 0.8,
        "blurred": F.avg_pool2d(target, 3, stride=1, padding=1),
    }
    ssim_checks = {}
    for name, image in cases.items():
        luminance, contrast, product = _ssim_parts(image, target)
        reference = baseline._ssim_batch(image, target)
        error = abs(float(product.mean()) - float(reference[0]))
        _require(error <= 1.0e-8, f"SSIM decomposition differs from production SSIM for {name}")
        ssim_checks[name] = {
            "production_ssim": float(reference[0]),
            "luminance_mean": float(luminance.mean()),
            "contrast_structure_mean": float(contrast.mean()),
            "product_mean": float(product.mean()),
            "absolute_error": error,
        }
    identical = cases["identical"]
    image_energy, target_energy, gradient_error = _sobel_metrics(identical, identical)
    _require(
        torch.equal(gradient_error, torch.zeros_like(gradient_error))
        and torch.equal(image_energy, target_energy),
        "Sobel reference fails identical-image control",
    )
    ramp = torch.arange(28, dtype=torch.float64, device=device)[None, None, None, :].expand(1, 1, 28, 28) / 27.0
    ramp_energy, _, ramp_error = _sobel_metrics(ramp, ramp)
    _require(
        torch.isclose(ramp_energy[0, 14, 14], torch.tensor(1.0 / 729.0, dtype=torch.float64, device=device), rtol=1.0e-12, atol=1.0e-12)
        and torch.equal(ramp_error, torch.zeros_like(ramp_error)),
        "Sobel analytic ramp reference failed",
    )
    shape = torch.rand((2, 1, 28, 28), device=device, dtype=torch.float64, generator=generator)
    target_shape = torch.rand((2, 1, 28, 28), device=device, dtype=torch.float64, generator=generator)
    tv = _normalized_shape_tv(shape, target_shape)
    scaled_tv = _normalized_shape_tv(shape * 7.25, target_shape * 0.31)
    _require(torch.allclose(tv, scaled_tv, rtol=1.0e-12, atol=1.0e-12), "shape TV is not invariant to positive scalar gain")
    z = torch.randn((7, 100), dtype=torch.complex128, device=device, generator=generator)
    common_phase = torch.exp(1j * torch.linspace(-2, 2, 7, dtype=torch.float64, device=device))
    kernel = _phase_power_kernel(z)
    per_sample_scale = torch.linspace(0.3, 8.0, 7, dtype=torch.float64, device=device)
    transformed_kernel = _phase_power_kernel(z * common_phase[:, None] * per_sample_scale[:, None])
    _require(torch.allclose(kernel, transformed_kernel, rtol=1.0e-12, atol=1.0e-12), "coefficient similarity kernel changed under phase/power gauge")
    tie_values = torch.tensor([0.0] * 97 + [1.0] * 3, dtype=torch.float64)
    tie_ref = _cap_reference_ties(tie_values)
    _require(tie_ref["at_cap_exceedances"] <= tie_ref["allowed"] and tie_ref["above_cap_exceedances"] > tie_ref["allowed"], "strict saturation cap tie reference is invalid")
    saturation_values = torch.tensor([-1.0e-6, 0.0, 1.0, 1.0, 1.0 + 1.0e-6], dtype=torch.float64)
    saturation_count = int(((saturation_values < 0.0) | (saturation_values > 1.0)).sum())
    _require(saturation_count == 2, "strict saturation pixel count differs at boundary values")
    # A non-empty target is used for the error partition identity.
    target400 = F.interpolate(target, size=(400, 400), mode="bilinear", align_corners=False)
    regions = readout._target_regions(target400)
    prediction = target400 * 0.75
    residual2 = (prediction - target400).square()
    region_sse = sum(float((residual2 * regions[name]).sum()) for name in ("edge", "stroke_interior", "background"))
    _require(math.isclose(region_sse, float(residual2.sum()), rel_tol=1.0e-10, abs_tol=1.0e-10), "regional SSE reference does not sum to full SSE")
    _require(torch.isfinite(regions["target28"]).all(), "target region reference is not finite")
    return {
        "ssim_cases": ssim_checks,
        "sobel_identical_image": True,
        "sobel_analytic_ramp": True,
        "shape_tv_positive_scale_invariance": True,
        "phase_power_kernel_invariance": True,
        "strict_saturation_tie_reference": tie_ref,
        "strict_saturation_boundary_count": saturation_count,
        "regional_sse_partition": True,
    }


def _phase_power_kernel(coefficients: torch.Tensor) -> torch.Tensor:
    z = coefficients.to(torch.complex128)
    power = z.abs().square().sum(1)
    _require(bool((power > 0).all()), "coefficient kernel requires nonzero samples")
    overlap = z @ z.conj().T
    kernel = overlap.abs().square() / (power[:, None] * power[None, :])
    return kernel.clamp(0.0, 1.0)


def _cap_reference_ties(values: torch.Tensor) -> dict[str, int | float]:
    values = values.detach().cpu().double()
    allowed = math.floor(SATURATION_MAX * values.numel())
    index = values.numel() - allowed - 1
    threshold = float(torch.sort(values).values[index])
    cap = math.inf if threshold <= 0 else 1.0 / threshold
    at = int(((values * cap) > 1.0).sum()) if math.isfinite(cap) else 0
    above = int(((values * math.nextafter(cap, math.inf)) > 1.0).sum()) if math.isfinite(cap) else 0
    return {"allowed": allowed, "threshold": threshold, "cap": cap, "at_cap_exceedances": at, "above_cap_exceedances": above}


def _validate_cache_payload(
    payload: dict[str, Any], expected_rows: torch.Tensor, expected_labels: torch.Tensor,
    expected_coefficients: torch.Tensor, expected_targets: torch.Tensor, combo: str,
) -> None:
    _require(payload.get("combo") == combo and payload.get("grid") == GRID, "cached block combo or grid mismatch")
    _require(torch.equal(payload["rows"], expected_rows), "cached block row order mismatch")
    _require(torch.equal(payload["labels"], expected_labels), "cached block labels mismatch")
    _require(torch.equal(payload["coefficients"], expected_coefficients), "cached block OLS coefficients mismatch")
    _require(torch.equal(payload["target"], expected_targets), "cached block target mismatch")
    physical = payload["physical_w_per_pixel"]
    _require(tuple(physical.shape) == (expected_rows.numel(), 1, 400, 400) and torch.isfinite(physical).all() and bool((physical >= 0).all()), "cached block physical output is invalid")


def _cache_validation_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "expected_rows": payload["rows"],
        "expected_labels": payload["labels"],
        "expected_coefficients": payload["coefficients"],
        "expected_targets": payload["target"],
        "combo": payload["combo"],
    }


def _state_snapshot_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)


def _expect_rejected(action: Any, name: str) -> bool:
    try:
        action()
    except (RuntimeError, ValueError):
        return True
    raise RuntimeError(f"consumer accepted invalid {name}")


def _cache_arrays(loaded: dict[str, Any]) -> dict[str, dict[str, torch.Tensor]]:
    arrays = {}
    for combo, spec in COMBOS.items():
        source = loaded["states"][spec["coefficient_state"]]
        arrays[combo] = {
            "rows": source["rows"],
            "labels": source["labels"],
            "target": source["target"],
            "coefficients": source["ols_coefficients"],
            "physical_w_per_pixel": source["physical_w_per_pixel"],
            "modal_power_w": source["modal_power_w"],
            "slm_after_modulation_power_w": source["slm_after_modulation_power_w"],
            "ols_incident_power_w": source["ols_incident_power_w"],
            "ols_modal_sum_w": source["ols_modal_sum_w"],
            "camera_roi_power_w": source["camera_roi_power_w"],
            "seven_plane_power_w": source["seven_plane_power_w"],
        }
        if combo in ("R01", "R10"):
            arrays[combo]["physical_w_per_pixel"] = torch.empty((0, 1, 400, 400), dtype=torch.float32)
    return arrays


def _signal_flag() -> tuple[list[int | None], dict[int, Any]]:
    flag: list[int | None] = [None]
    previous: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        del frame
        flag[0] = signum

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, handler)
    return flag, previous


def _restore_signals(previous: dict[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _recovery_subset(loaded: dict[str, Any], combo: str) -> tuple[dict[str, Any], torch.Tensor]:
    population = loaded["population"]
    rows = torch.cat((population["fit_rows_by_class"][0, :10], population["fit_rows_by_class"][1, :10])).long()
    row_to_index = {int(row): index for index, row in enumerate(population["all_rows"].tolist())}
    indices = torch.tensor([row_to_index[int(row)] for row in rows], dtype=torch.long)
    state_name = COMBOS[combo]["coefficient_state"]
    state = loaded["states"][state_name]
    loaded["states"][state_name] = {
        key: value.index_select(0, indices) if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 300 else value
        for key, value in state.items()
    }
    loaded["population"] = {
        **population,
        "all_rows": rows,
        "all_labels": loaded["labels"][rows].cpu(),
    }
    return loaded, rows


def _recovery_evaluation_blocks(
    combo: str, measured: dict[str, torch.Tensor], loaded: dict[str, Any], directory: Path,
    progress: dict[str, Any], stop_requested: list[int | None], gain: float,
) -> bool:
    directory.mkdir(parents=True, exist_ok=True)
    for start in range(0, measured["rows"].numel(), BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, measured["rows"].numel())
        path = directory / f"metrics_{start:03d}_{end:03d}.pt"
        expected_rows = measured["rows"][start:end]
        if path.is_file():
            payload = torch.load(path, map_location="cpu", weights_only=False)
            _require(torch.equal(payload["rows"], expected_rows), "saved evaluation block rows differ on recovery")
            _require(payload["combo"] == combo and len(payload["records"]) == end - start, "saved evaluation block identity differs")
            progress["completed_rows"] = max(int(progress.get("completed_rows", 0)), end)
            continue
        if stop_requested[0] is not None:
            return False
        if progress.get("signal_recovery_probe"):
            atomic_write_json(directory / "active_evaluation_block.json", {"combo": combo, "start": start, "end": end, "next_index": start, "stage": "active"})
        records = []
        with torch.no_grad():
            for index in range(start, end):
                record, _ = _detailed_row(combo, index, measured, gain, loaded["evaluator"], set())
                records.append(record)
                if progress.get("signal_recovery_probe"):
                    atomic_write_json(directory / "active_evaluation_block.json", {"combo": combo, "start": start, "end": end, "next_index": index + 1, "stage": "active"})
        atomic_torch_save_new(path, {"combo": combo, "rows": expected_rows.clone(), "records": records, "decoder_forwards": 0})
        readback = torch.load(path, map_location="cpu", weights_only=False)
        _require(torch.equal(readback["rows"], expected_rows) and readback["records"] == records, "evaluation recovery block readback differs")
        progress["completed_rows"] = end
        progress["updated_at"] = time.time()
        if stop_requested[0] is not None:
            return False
    return True


def _recovery_worker(args: argparse.Namespace) -> int:
    directory = _resolve(args.recovery_dir)
    _require(args.recovery_stage in {"measurement", "evaluation"} and args.recovery_combo in CROSS_COMBOS, "invalid production recovery probe stage or pairing")
    loaded = _load_context(args, torch.device("cuda"))
    population = loaded["population"]
    fit = population["fit_rows_by_class"].long()
    fit_indices = torch.tensor(
        [{int(row): index for index, row in enumerate(population["all_rows"].tolist())}[int(row)] for row in fit.flatten()],
        dtype=torch.long,
    )
    gain = _independent_gain_reference(
        loaded["states"][STATE0]["physical_w_per_pixel"][fit_indices],
        loaded["states"][STATE0]["target"][fit_indices],
    )["constrained"]
    loaded, expected_rows = _recovery_subset(loaded, args.recovery_combo)
    output = directory / "measurement"
    if args.recovery_stage == "evaluation":
        measurement_output = directory.parent / "measurement" / "measurement"
        _require(measurement_output.is_dir(), "evaluation recovery requires completed production measurement blocks")
        measured = _read_cross_combo(args.recovery_combo, loaded, measurement_output)
        output = directory / "evaluation"
    if args.resume_checkpoint is None:
        _require(not output.exists(), f"production recovery probe output already exists: {output}")
        output.mkdir(parents=True, exist_ok=False)
        progress: dict[str, Any] = {
            "signal_recovery_probe": True,
            "stage": args.recovery_stage,
            "combo": args.recovery_combo,
            "completed_rows": 0,
            "cross_forward_calls": 0,
            "updated_at": time.time(),
        }
    else:
        checkpoint = torch.load(_resolve(args.resume_checkpoint), map_location="cpu", weights_only=False)
        _require(checkpoint.get("task_id") == TASK_ID and checkpoint.get("stage") == args.recovery_stage and checkpoint.get("combo") == args.recovery_combo, "production recovery checkpoint identity differs")
        _require(torch.equal(checkpoint["expected_rows"], expected_rows), "production recovery checkpoint rows differ")
        progress = checkpoint["progress"]
        progress["signal_recovery_probe"] = True
    stop_requested, previous_handlers = _signal_flag()
    try:
        if args.recovery_stage == "measurement":
            decoder = _current_models(loaded)[COMBOS[args.recovery_combo]["decoder"]]
            progress.setdefault("completed_rows_by_combo", {args.recovery_combo: 0})
            completed = _measure_cross_combo(args.recovery_combo, decoder, loaded, output, progress, stop_requested)
            saved_rows = []
            for start in range(0, expected_rows.numel(), BLOCK_SIZE):
                end = min(start + BLOCK_SIZE, expected_rows.numel())
                path = output / args.recovery_combo / f"block_{start:03d}_{end:03d}.pt"
                if path.is_file():
                    saved_rows.extend(_validate_cross_block(path, args.recovery_combo, loaded)["rows"].tolist())
            _require(saved_rows == expected_rows.tolist()[: len(saved_rows)] and len(set(saved_rows)) == len(saved_rows), "production measurement recovery blocks omit, duplicate or reorder rows")
            if args.resume_checkpoint is None:
                _require(stop_requested[0] is not None and not completed, "production measurement probe was not interrupted")
                checkpoint_path = directory / "measurement_resume.pt"
                atomic_torch_save(checkpoint_path, {"task_id": TASK_ID, "stage": args.recovery_stage, "combo": args.recovery_combo, "signal": stop_requested[0], "expected_rows": expected_rows, "progress": progress})
                _require(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["progress"] == progress, "production measurement recovery checkpoint readback differs")
                return 75
            _require(completed and len(saved_rows) == 20 and progress["cross_forward_calls"] == 20, "independent production measurement resume did not validate all rows")
            atomic_write_json(directory / "completion.json", {"rows": saved_rows, "unique_rows": len(set(saved_rows)), "decoder_forwards_total": progress["cross_forward_calls"], "decoder_forwards_on_resume": 0, "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()})
            return 0

        progress.setdefault("completed_rows", 0)
        completed = _recovery_evaluation_blocks(args.recovery_combo, measured, loaded, output, progress, stop_requested, gain)
        saved_rows = []
        for start in range(0, expected_rows.numel(), BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, expected_rows.numel())
            path = output / f"metrics_{start:03d}_{end:03d}.pt"
            if path.is_file():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                _require(torch.equal(payload["rows"], expected_rows[start:end]) and payload["decoder_forwards"] == 0, "production evaluation recovery block is invalid")
                saved_rows.extend(payload["rows"].tolist())
        _require(saved_rows == expected_rows.tolist()[: len(saved_rows)] and len(set(saved_rows)) == len(saved_rows), "production evaluation recovery blocks omit, duplicate or reorder rows")
        if args.resume_checkpoint is None:
            _require(stop_requested[0] is not None and not completed, "production evaluation probe was not interrupted")
            checkpoint_path = directory / "evaluation_resume.pt"
            atomic_torch_save(checkpoint_path, {"task_id": TASK_ID, "stage": args.recovery_stage, "combo": args.recovery_combo, "signal": stop_requested[0], "expected_rows": expected_rows, "progress": progress})
            _require(torch.load(checkpoint_path, map_location="cpu", weights_only=False)["progress"] == progress, "production evaluation recovery checkpoint readback differs")
            return 75
        _require(completed and len(saved_rows) == 20 and progress["completed_rows"] == 20, "independent production evaluation resume did not validate all rows")
        atomic_write_json(directory / "completion.json", {"rows": saved_rows, "unique_rows": len(set(saved_rows)), "decoder_forwards_total": 0, "decoder_forwards_on_resume": 0, "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()})
        return 0
    finally:
        _restore_signals(previous_handlers)


def _recovery_cli(args: argparse.Namespace, directory: Path, stage: str, combo: str, resume: Path | None = None) -> list[str]:
    command = [
        sys.executable, "-m", "scripts.research.lightgen.diagnose_chen_lightgen_400_frozen_crossdecode_v1",
        "--mode", "recovery-worker",
        "--checkpoint", str(_resolve(args.checkpoint)),
        "--baseline-report", str(_resolve(args.baseline_report)),
        "--source-audit-dir", str(_resolve(args.source_audit_dir)),
        "--source-preflight-dir", str(_resolve(args.source_preflight_dir)),
        "--readout-report", str(_resolve(args.readout_report)),
        "--start-state", str(_resolve(args.start_state)),
        "--endpoint-checkpoint", str(_resolve(args.endpoint_checkpoint)),
        "--joint-report", str(_resolve(args.joint_report)),
        "--fixed-population", str(_resolve(args.fixed_population)),
        "--config", str(_resolve(args.config)),
        "--data", str(_resolve(args.data)),
        "--output-dir", str(_resolve(args.output_dir)),
        "--recovery-dir", str(directory),
        "--recovery-stage", stage,
        "--recovery-combo", combo,
    ]
    if args.evaluator is not None:
        command.extend(("--evaluator", str(_resolve(args.evaluator))))
    if resume is not None:
        command.extend(("--resume-checkpoint", str(resume)))
    return command


def _run_signalled_child(
    args: argparse.Namespace, directory: Path, stage: str, combo: str, signum: int, probe_root: Path,
) -> dict[str, Any]:
    directory.mkdir(parents=True, exist_ok=False)
    process = subprocess.Popen(_recovery_cli(args, directory, stage, combo), cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    marker = directory / ("measurement" if stage == "measurement" else "evaluation") / (
        "active_measurement_block.json" if stage == "measurement" else "active_evaluation_block.json"
    )
    last_heartbeat = time.monotonic()
    try:
        while process.poll() is None:
            if marker.is_file():
                state = _json(marker)
                if int(state.get("next_index", -1)) > int(state.get("start", -1)):
                    os.kill(process.pid, signum)
                    break
            if time.monotonic() - last_heartbeat >= 60.0:
                atomic_write_json(probe_root / "heartbeat.json", {"alive": True, "time": time.time(), "signal": signum, "stage": stage, "combo": combo})
                last_heartbeat = time.monotonic()
            time.sleep(0.001)
        exit_code = process.wait()
        stdout, stderr = process.communicate()
        _require(exit_code == 75, f"{stage} production worker did not exit at an observed signal: exit={exit_code}; stderr={stderr[-2000:]}; stdout={stdout[-1000:]}")
        checkpoint = directory / f"{stage}_resume.pt"
        _require(checkpoint.is_file(), f"{stage} production worker omitted its recovery checkpoint")
        resume = subprocess.run(_recovery_cli(args, directory, stage, combo, checkpoint), cwd=ROOT, capture_output=True, text=True, check=False)
        _require(resume.returncode == 0, f"independent {stage} production resume failed: {resume.stderr[-2000:]}")
        completion = _json(directory / "completion.json")
        expected_count = 20
        _require(completion.get("unique_rows") == expected_count and len(completion.get("rows", [])) == expected_count, f"{stage} production recovery did not consume each row once")
        if stage == "measurement":
            _require(completion["decoder_forwards_total"] == expected_count and completion["decoder_forwards_on_resume"] == 0, "measurement recovery forward accounting differs")
        else:
            _require(completion["decoder_forwards_total"] == 0 and completion["decoder_forwards_on_resume"] == 0, "evaluation resume repeated an optical forward")
        return {"signal": signum, "stage": stage, "combo": combo, "signal_exit_code": exit_code, "resume_exit_code": resume.returncode, "rows_consumed_once": expected_count, "decoder_forwards_total": completion["decoder_forwards_total"], "decoder_forwards_on_resume": completion["decoder_forwards_on_resume"], "peak_reserved_vram_bytes": completion["peak_reserved_vram_bytes"]}
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def _verify_signal_recovery(args: argparse.Namespace, directory: Path) -> dict[str, Any]:
    scenarios = []
    total_decoder_forwards = 0
    for signum, combo in ((signal.SIGINT, "R01"), (signal.SIGTERM, "R10")):
        scenario = directory / f"signal_{signum}"
        measurement = _run_signalled_child(args, scenario / "measurement", "measurement", combo, signum, directory)
        evaluation = _run_signalled_child(args, scenario / "evaluation", "evaluation", combo, signum, directory)
        scenarios.extend((measurement, evaluation))
        total_decoder_forwards += measurement["decoder_forwards_total"]
    _require(total_decoder_forwards == 40, "production signal recovery did not account for exactly 40 Decoder forwards")
    return {
        "passed": True,
        "scenarios": scenarios,
        "measurement_decoder_forwards": total_decoder_forwards,
        "evaluation_decoder_forwards": 0,
        "peak_reserved_vram_bytes": max(item["peak_reserved_vram_bytes"] for item in scenarios),
        "scope": "production cross-Decoder saver, block recovery, and offline metric recovery in independent processes",
    }


def _previous_preflight_evidence(args: argparse.Namespace, loaded: dict[str, Any]) -> dict[str, Any]:
    previous_dir = _resolve(args.previous_cross_preflight_dir)
    previous = _json(previous_dir / "preflight.json")
    previous_ready = _json(previous_dir / "preflight_ready.json")
    _require(previous.get("task_id") == TASK_ID and previous.get("status") == "READY", "explicit previous preflight is not the expected completed cross-decode preflight")
    _require(previous_ready.get("status") == "READY" and previous_ready.get("authorization_id") == AUTHORITY, "previous cross-decode receipt is incomplete")
    for key, value in (
        ("source_audit_dir", str(loaded["source_audit_dir"])),
        ("start_state", str(loaded["start_path"])),
        ("endpoint_checkpoint", str(loaded["endpoint_path"])),
        ("fixed_population", str(loaded["fixed_path"])),
        ("config", str(_resolve(args.config))),
        ("data", str(_resolve(args.data))),
        ("grid", GRID),
        ("reference_grid", REFERENCE_GRID),
    ):
        _require(previous.get(key) == value, f"historical preflight input differs: {key}")
    population = loaded["population"]
    expected_class_rows = {int(population["confirm_rows_by_class"][label, 0]): label for label in range(10)}
    old_windows = previous.get("cross_window_checks", [])
    old_cross = {(record.get("combo"), int(record.get("row", -1))): record for record in old_windows}
    expected_old_cross = {
        (combo, int(population["confirm_rows_by_class"][label, 0]))
        for combo in CROSS_COMBOS
        for label in (0, 1)
    }
    _require(len(old_cross) == 4 and set(old_cross) == expected_old_cross, "historical cross-window evidence is not the exact original 0/1-class coverage")
    old_cross_records = []
    for key, record in old_cross.items():
        label = expected_class_rows[key[1]]
        errors = record.get("per_plane_errors", {})
        _require(label in (0, 1) and set(errors) == {"decoder_entry", "decoder_1", "decoder_2", "decoder_3", "decoder_4"}, "historical cross-window plane coverage is incomplete")
        maximum = max(float(value) for value in errors.values())
        _require(record.get("passed") is True and maximum <= WINDOW_TOLERANCE and math.isclose(maximum, float(record["maximum_error"]), rel_tol=1.0e-10, abs_tol=1.0e-12), "historical cross-window result does not satisfy the original tolerance")
        old_cross_records.append({**record, "evidence_source": str(previous_dir / "preflight.json"), "evidence_status": "revalidated_saved_result"})

    old_diagonal = previous.get("diagonal_replays", [])
    expected_diagonal = {
        (combo, int(population["confirm_rows_by_class"][label, 0]))
        for combo in ("R00", "R11")
        for label in (0, 1)
    }
    diagonal_map = {(item.get("combo"), int(item.get("row", -1))): item for item in old_diagonal}
    _require(len(diagonal_map) == 4 and set(diagonal_map) == expected_diagonal, "historical diagonal replay evidence is incomplete")
    checked_diagonal = []
    for (combo, row), record in diagonal_map.items():
        _require(record.get("passed") is True and float(record["relative_l2_error"]) <= 1.0e-5, "historical diagonal replay exceeds tolerance")
        path = previous_dir / f"diagonal_{combo}_row_{row}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = COMBOS[combo]["coefficient_state"]
        index = int(torch.where(loaded["states"][state]["rows"] == row)[0][0])
        _require(payload.get("combo") == combo and int(payload["row"]) == row, "historical diagonal saved replay identity differs")
        _require(torch.equal(payload["coefficients"], loaded["states"][state]["ols_coefficients"][index : index + 1]), "historical diagonal replay coefficient differs")
        _require(torch.allclose(payload["physical_w_per_pixel"], loaded["states"][state]["physical_w_per_pixel"][index : index + 1], rtol=0.0, atol=1.0e-7), "historical diagonal replay output differs")
        checked_diagonal.append({**record, "evidence_source": str(path), "evidence_status": "revalidated_saved_result"})
    return {"previous_preflight_dir": str(previous_dir), "cross_windows": old_cross_records, "diagonal_replays": checked_diagonal}


def _validate_source_cross_run(args: argparse.Namespace, loaded: dict[str, Any]) -> dict[str, Any]:
    source_dir = _resolve(args.source_cross_dir)
    manifest = _json(source_dir / "run_manifest.json")
    report = _json(source_dir / "report.json")
    _require(
        manifest.get("task_id") == TASK_ID
        and manifest.get("cross_forward_budget") == 600
        and manifest.get("optimizer_updates") == 0
        and manifest.get("combos") == COMBOS,
        "source cross-decode run manifest is not the explicit four-combination 600-forward audit",
    )
    _require(
        report.get("task_id") == TASK_ID
        and report.get("status") == "VALID_COMPLETED"
        and report.get("formal_cross_decoder_forwards") == 600
        and report.get("completed_updates") == 0,
        "source cross-decode report is not the completed zero-update measurement",
    )
    for key, value in (
        ("source_audit_dir", str(loaded["source_audit_dir"])),
        ("start_state", str(loaded["start_path"])),
        ("endpoint_checkpoint", str(loaded["endpoint_path"])),
        ("fixed_population", str(loaded["fixed_path"])),
    ):
        _require(manifest.get(key) == value, f"source cross-decode run input differs: {key}")
    counts = {}
    for combo in CROSS_COMBOS:
        paths = sorted((source_dir / combo).glob("block_*.pt"))
        _require(len(paths) == 30, f"source cross-decode cache is incomplete for {combo}")
        counts[combo] = len(paths)
    return {"source_cross_dir": str(source_dir), "source_run_id": manifest.get("run_id"), "cached_blocks_by_combo": counts}


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is required for frozen cross-decode preflight")
    output = _resolve(args.output_dir)
    _require(not output.exists(), f"refusing to overwrite preflight directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    loaded = _load_context(args, torch.device("cuda"))
    source_cross = _validate_source_cross_run(args, loaded)
    models = _current_models(loaded)
    states = loaded["states"]
    population = loaded["population"]
    roi_m = float(loaded["config"]["optics"]["common_complex_comparison_roi_m"])
    before = {name: baseline._state_snapshot(model) for name, model in models.items()}
    evaluator_before = baseline._state_snapshot(loaded["evaluator"])
    previous_evidence = _previous_preflight_evidence(args, loaded)
    checks: list[dict[str, Any]] = previous_evidence["diagonal_replays"]

    cross_windows = list(previous_evidence["cross_windows"])
    for combo in CROSS_COMBOS:
        coefficient_state = COMBOS[combo]["coefficient_state"]
        decoder = models[COMBOS[combo]["decoder"]]
        index_by_row = {int(row): i for i, row in enumerate(states[coefficient_state]["rows"].tolist())}
        for label in range(2, 10):
            row = int(population["confirm_rows_by_class"][label, 0])
            index = index_by_row[row]
            probe = _cross_window_probe(
                decoder,
                states[coefficient_state]["ols_coefficients"][index : index + 1],
                roi_m,
            )
            _require(probe["passed"], f"{combo} 4096/6144 comparison failed at row {row}")
            cross_windows.append({"combo": combo, "row": row, "label": label, "evidence_source": "new_production_cross_window_probe", **probe})

    mismatch_rejected = False
    try:
        baseline._reconstruction_batch_metrics(
            torch.zeros((1, 1, 400, 400), device="cuda"),
            torch.zeros((1, 1, 400, 400), device="cuda"),
            torch.tensor([1]),
            torch.tensor([2]),
        )
    except ValueError:
        mismatch_rejected = True
    _require(mismatch_rejected, "production metric consumer accepted mismatched row ids")
    sample_source = states[STATE0]
    expected_payload = {
        "combo": "R00", "grid": GRID,
        "rows": sample_source["rows"][:1], "labels": sample_source["labels"][:1],
        "coefficients": sample_source["ols_coefficients"][:1],
        "target": sample_source["target"][:1],
        "physical_w_per_pixel": sample_source["physical_w_per_pixel"][:1],
    }
    wrong_label = expected_payload["labels"].clone()
    wrong_label[0] = (wrong_label[0] + 1) % 10
    mismatch_consumers = {
        "row": _expect_rejected(
            lambda: _validate_cache_payload(
                {**expected_payload, "rows": expected_payload["rows"] + 1},
                **_cache_validation_kwargs(expected_payload),
            ),
            "cache row mismatch",
        ),
        "label": _expect_rejected(
            lambda: _validate_cache_payload(
                {**expected_payload, "labels": wrong_label},
                **_cache_validation_kwargs(expected_payload),
            ),
            "cache label mismatch",
        ),
        "coefficient": _expect_rejected(
            lambda: _validate_cache_payload(
                {**expected_payload, "coefficients": expected_payload["coefficients"] + (1.0e-8 + 0.0j)},
                **_cache_validation_kwargs(expected_payload),
            ),
            "cache coefficient mismatch",
        ),
    }
    metric_references = _verify_metric_references(loaded)

    fit_mask = torch.isin(states[STATE0]["rows"], population["fit_rows"])
    reference = _independent_gain_reference(
        states[STATE0]["physical_w_per_pixel"][fit_mask], states[STATE0]["target"][fit_mask]
    )
    implemented = readout._gain_values(
        states[STATE0]["physical_w_per_pixel"][fit_mask],
        states[STATE0]["target"][fit_mask],
        float(loaded["references"]["camera_gain"]),
        float(_json(_resolve(args.joint_report))["fixed_gain"]),
    )
    _require(
        math.isclose(reference["constrained"], implemented["fit_only_constrained_gain_w_inverse"], rel_tol=1.0e-12),
        "shared baseline gain differs from independent NumPy reference",
    )
    for name, model in models.items():
        _require(all(not parameter.requires_grad for parameter in model.parameters()) and model.training is False, f"{name} is not frozen/eval")
        _require(_state_snapshot_equal(baseline._state_snapshot(model), before[name]), f"{name} changed during preflight")
    _require(_state_snapshot_equal(baseline._state_snapshot(loaded["evaluator"]), evaluator_before), "frozen evaluator changed during preflight")
    signals = _verify_signal_recovery(args, output / "signal_recovery")
    g0 = reference["constrained"]
    _require(math.isclose(g0, 3.17684599058831e10, rel_tol=2.0e-6), "R00 common readout gain does not reproduce the planned fit-only value")

    report = {
        "schema": "chen_lightgen_400_frozen_crossdecode_preflight.v1",
        "status": "READY",
        "task_id": TASK_ID,
        "authorization_id": AUTHORITY,
        "source_audit_dir": str(loaded["source_audit_dir"]),
        "source_cross_dir": source_cross["source_cross_dir"],
        "source_cross_run": source_cross,
        "previous_cross_preflight_dir": previous_evidence["previous_preflight_dir"],
        "canonical_target_cpu_cuda_max_abs_error": loaded["target_preprocessing_max_abs_error"],
        "canonical_target_cpu_cuda_absolute_tolerance": TARGET_REPLAY_ATOL,
        "source_audit_manifest": str(loaded["source_audit_dir"] / "run_manifest.json"),
        "source_readout_report": str(_resolve(args.readout_report)),
        "baseline_checkpoint": str(loaded["source_checkpoint_path"]),
        "start_state": str(loaded["start_path"]),
        "endpoint_checkpoint": str(loaded["endpoint_path"]),
        "joint_report": str(loaded["joint_report_path"]),
        "fixed_population": str(loaded["fixed_path"]),
        "source_preflight_dir": str(loaded["source_preflight_dir"]),
        "config": str(_resolve(args.config)),
        "data": str(_resolve(args.data)),
        "evaluator": str(loaded["evaluator_path"]),
        "grid": GRID,
        "reference_grid": REFERENCE_GRID,
        "sample_count": 300,
        "fit_rows": 100,
        "confirmation_rows": 200,
        "cross_forwards_planned": 600,
        "preflight_optical_forwards": 72,
        "window_decoder_forwards_new": 32,
        "signal_recovery_decoder_forwards_new": 40,
        "optimizer_updates": 0,
        "common_gain_w_inverse": g0,
        "independent_gain_reference": reference,
        "implemented_gain_reference": implemented,
        "diagonal_replays": checks,
        "cross_window_checks": cross_windows,
        "row_mismatch_rejected": mismatch_rejected,
        "cache_row_label_coefficient_mismatch_rejected": mismatch_consumers,
        "metric_references": metric_references,
        "signal_recovery": signals,
        "frozen_models_evaluator_passed": True,
        "cuda_device": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_reserved_vram_bytes": max(torch.cuda.max_memory_reserved(), signals["peak_reserved_vram_bytes"]),
    }
    cross_coverage = {(record["combo"], int(record["row"])) for record in cross_windows}
    expected_cross_coverage = {
        (combo, int(population["confirm_rows_by_class"][label, 0]))
        for combo in CROSS_COMBOS
        for label in range(10)
    }
    _require(len(checks) == 4 and len(cross_windows) == 20 and cross_coverage == expected_cross_coverage, "cross-window preflight does not cover both cross decoders and all ten classes")
    _require(signals["measurement_decoder_forwards"] == 40 and signals["evaluation_decoder_forwards"] == 0, "production recovery evidence has incorrect optical forward counts")
    correctness = {
        "passed": True,
        "source_and_population_cache_valid": True,
        "diagonal_replay_passed": True,
        "cross_windows_passed": True,
        "cross_window_pair_count": 20,
        "metrics_independent_references_passed": True,
        "row_label_coefficient_rejections_passed": all(mismatch_consumers.values()),
        "signal_recovery_passed": signals["passed"],
        "models_and_evaluator_frozen": True,
        "optimizer_updates": 0,
        "official_dev_test_reads": 0,
    }
    resource = {"passed": True, "cuda_device": report["cuda_device"], "peak_reserved_vram_bytes": report["peak_reserved_vram_bytes"]}
    implementation = {"passed": True, "engine": "cuda_4096_frozen_decoder_crossdecode", "selected_engine": "cuda_4096_frozen_decoder_crossdecode", "trainable_groups": [], "frozen_groups": ["encoder", "decoder", "ols", "gain", "references", "evaluator"]}
    source_manifest = capture_execution_source_manifest(
        ROOT,
        task_id=TASK_ID,
        semantic_authority_id=f"{TASK_ID}:E0D0_E0D1_E1D0_E1D1:ZERO_UPDATES",
        relevant_paths=(
            Path(__file__),
            Path("scripts/research/lightgen/diagnose_chen_lightgen_400_readout_structure_audit_v1.py"),
            Path("scripts/research/lightgen/diagnose_chen_lightgen_400_quickcheck_v1.py"),
            Path("scripts/research/lightgen/execute_chen_lightgen_400_mnist_baseline_v1.py"),
            Path("src/ldm_project/v2/application/research/lightgen/chen_lightgen_400_baseline.py"),
            _resolve(args.config),
            loaded["fixed_path"],
            _resolve(args.source_cross_dir) / "run_manifest.json",
            _resolve(args.source_cross_dir) / "report.json",
            _resolve(args.previous_cross_preflight_dir) / "preflight.json",
        ),
            config_identity={"combos": list(COMBOS), "grid": GRID, "gain": "R00_fit_constrained", "forwards": 600, "preflight_window_pairs": 20},
    )
    identity = ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=uuid.uuid4().hex,
        trajectory_id=f"{TASK_ID}:start4096:endpoint500:fixed300",
        semantic_authority_id=f"{TASK_ID}:COMMON_R00_GAIN:600_CROSS_DECODER_FORWARDS",
        data_role="canonical_MNIST_TRAIN_fixed_fit100_confirmation200_no_dev_test_reads",
        config_identity={"grid": GRID, "reference_grid": REFERENCE_GRID, "combos": list(COMBOS), "cross_forwards": 600, "block_size": BLOCK_SIZE, "window_pairs": 20},
        model_identity={"encoder": [2, 400, 400], "decoder": [4, 400, 400], "ols": "gaussian_s12_v1_100_ports", "source_step": 30_000, "endpoint_updates": 500, "grid": GRID, "trainable": []},
        optimizer_identity={"name": "none", "updates": 0},
        sampler_identity={"fit_rows": 100, "confirmation_rows": 200, "source": str(loaded["fixed_path"]), "new_samples": 0},
        execution_source_semantics_id=source_manifest.execution_source_semantics_id,
        source_manifest_id=source_manifest.source_manifest_id,
    )
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    receipt = control.make_ready_receipt(
        correctness=correctness,
        resource=resource,
        implementation=implementation,
        selected_engine="cuda_4096_frozen_decoder_crossdecode",
        resource_envelope=resource,
    )
    control.mark_preflight_ready(receipt)
    report["run_identity"] = identity.to_dict()
    report["receipt"] = receipt.to_dict()
    atomic_write_new_json(output / "preflight.json", report)
    atomic_write_new_json(output / "preflight_ready.json", receipt.to_dict())
    atomic_torch_save_new(output / "fixed_population.pt", population)
    atomic_write_new_json(output / "cache_validation.json", {"status": "PASSED", "states": [STATE0, STATE1], "rows": population["all_rows"].tolist(), "source_cache_dir": str(loaded["source_audit_dir"])})
    return report


def _ready(args: argparse.Namespace) -> tuple[dict[str, Any], PreflightReceipt]:
    preflight_dir = _resolve(args.preflight_dir)
    report = _json(preflight_dir / "preflight.json")
    receipt = PreflightReceipt.from_mapping(_json(preflight_dir / "preflight_ready.json"))
    _require(report.get("status") == "READY" and report.get("task_id") == TASK_ID and report.get("authorization_id") == AUTHORITY, "this task requires its own explicit READY preflight")
    _require(receipt.status == "READY" and receipt.authorization_id == AUTHORITY, "cross-decode READY receipt is invalid")
    for key, path in (("source_audit_dir", args.source_audit_dir), ("source_cross_dir", args.source_cross_dir), ("previous_cross_preflight_dir", args.previous_cross_preflight_dir), ("source_readout_report", args.readout_report), ("baseline_checkpoint", args.checkpoint), ("start_state", args.start_state), ("endpoint_checkpoint", args.endpoint_checkpoint), ("joint_report", args.joint_report), ("fixed_population", args.fixed_population), ("source_preflight_dir", args.source_preflight_dir), ("config", args.config), ("data", args.data)):
        _require(report[key] == str(_resolve(path)), f"run input differs from this task READY preflight: {key}")
    if args.evaluator is not None:
        _require(report["evaluator"] == str(_resolve(args.evaluator)), "run evaluator differs from READY preflight")
    return report, receipt


def _save_recovery(output: Path, progress: dict[str, Any], identity: ResearchRunIdentity) -> Path:
    recovery = {"schema": "chen_lightgen_400_frozen_crossdecode_resume.v1", "task_id": TASK_ID, "run_id": identity.run_id, "attempt_id": output.name, "progress": progress}
    path = output / f"resume_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.pt"
    atomic_torch_save_new(path, recovery)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _require(loaded == recovery, "recovery checkpoint readback differs")
    return path


def _validate_cross_block(path: Path, combo: str, loaded: dict[str, Any]) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = COMBOS[combo]["coefficient_state"]
    rows = loaded["population"]["all_rows"].long()
    start = int(path.stem.split("_")[1])
    end = int(path.stem.split("_")[2])
    _validate_cache_payload(
        payload,
        rows[start:end],
        loaded["labels"][rows[start:end]].long(),
        loaded["states"][state]["ols_coefficients"][start:end],
        loaded["states"][state]["target"][start:end],
        combo,
    )
    _require(payload.get("encoder") == COMBOS[combo]["encoder"] and payload.get("decoder") == COMBOS[combo]["decoder"], "cross block model pairing mismatch")
    source = loaded["states"][state]
    for key in ("modal_power_w", "slm_after_modulation_power_w", "ols_incident_power_w", "ols_modal_sum_w"):
        _require(torch.equal(payload[key], source[key][start:end]), f"cross block source power differs: {key}")
    _require(
        tuple(payload["decoder_plane_power_w"].shape) == (end - start, 5)
        and torch.isfinite(payload["decoder_plane_power_w"]).all()
        and payload.get("forward_calls") == end - start,
        "cross block decoder powers or forward count are invalid",
    )
    expected_camera = payload["physical_w_per_pixel"].double().sum((1, 2, 3))
    expected_planes = torch.cat(
        (source["seven_plane_power_w"][start:end, :2], payload["decoder_plane_power_w"]),
        dim=1,
    )
    legacy_camera = source["camera_roi_power_w"][start:end].reshape(-1)
    legacy_planes = source["seven_plane_power_w"][start:end]
    _require(
        torch.allclose(payload["camera_roi_power_w"].reshape(-1).double(), expected_camera, rtol=2.0e-6, atol=1.0e-15)
        or torch.equal(payload["camera_roi_power_w"].reshape(-1), legacy_camera),
        "cross block camera power is neither current output power nor the known historical diagonal copy",
    )
    _require(
        torch.allclose(payload["seven_plane_power_w"].double(), expected_planes.double(), rtol=2.0e-6, atol=1.0e-15)
        or torch.equal(payload["seven_plane_power_w"], legacy_planes),
        "cross block seven-plane powers are neither current Decoder powers nor the known historical diagonal copy",
    )
    return payload


def _save_cross_block(path: Path, payload: dict[str, Any]) -> None:
    atomic_torch_save_new(path, payload)
    readback = torch.load(path, map_location="cpu", weights_only=False)
    _require(
        readback["combo"] == payload["combo"]
        and torch.equal(readback["rows"], payload["rows"])
        and torch.equal(readback["labels"], payload["labels"])
        and torch.equal(readback["coefficients"], payload["coefficients"])
        and torch.equal(readback["target"], payload["target"])
        and torch.equal(readback["physical_w_per_pixel"], payload["physical_w_per_pixel"]),
        f"cross measurement block readback mismatch: {path}",
    )


def _measure_cross_combo(
    combo: str,
    decoder: torch.nn.Module,
    loaded: dict[str, Any],
    output: Path,
    progress: dict[str, Any],
    stop_requested: list[int | None],
) -> bool:
    destination = output / combo
    destination.mkdir(parents=True, exist_ok=True)
    state = COMBOS[combo]["coefficient_state"]
    source = loaded["states"][state]
    rows = loaded["population"]["all_rows"].long()
    for start in range(0, rows.numel(), BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, rows.numel())
        block_path = destination / f"block_{start:03d}_{end:03d}.pt"
        if block_path.exists():
            _validate_cross_block(block_path, combo, loaded)
            progress["completed_rows_by_combo"][combo] = max(progress["completed_rows_by_combo"][combo], end)
            continue
        if stop_requested[0] is not None:
            return False
        if progress.get("signal_recovery_probe"):
            atomic_write_json(
                output / "active_measurement_block.json",
                {"combo": combo, "start": start, "end": end, "next_index": start, "stage": "active"},
            )
        physical_rows = []
        decoder_power_rows = []
        with torch.no_grad():
            for index in range(start, end):
                coefficients = source["ols_coefficients"][index : index + 1].to("cuda")
                result = decoder.forward_from_ols(coefficients, return_diagnostics=True)
                physical_rows.append(result["physical_w_per_pixel"].detach().cpu())
                decoder_power_rows.append(result["plane_powers_w"].detach().cpu())
                progress["cross_forward_calls"] += 1
                del result, coefficients
                if progress.get("signal_recovery_probe"):
                    atomic_write_json(
                        output / "active_measurement_block.json",
                        {"combo": combo, "start": start, "end": end, "next_index": index + 1, "stage": "active"},
                    )
        decoder_planes = torch.cat(decoder_power_rows)
        physical = torch.cat(physical_rows)
        source_seven = source["seven_plane_power_w"][start:end].clone()
        current_seven = torch.cat((source_seven[:, :2], decoder_planes), dim=1)
        payload = {
            "schema": "chen_lightgen_400_frozen_crossdecode_block.v1",
            "combo": combo,
            "encoder": COMBOS[combo]["encoder"],
            "decoder": COMBOS[combo]["decoder"],
            "grid": GRID,
            "rows": rows[start:end].clone(),
            "labels": source["labels"][start:end].clone(),
            "target": source["target"][start:end].clone(),
            "coefficients": source["ols_coefficients"][start:end].clone(),
            "physical_w_per_pixel": physical,
            "modal_power_w": source["modal_power_w"][start:end].clone(),
            "slm_after_modulation_power_w": source["slm_after_modulation_power_w"][start:end].clone(),
            "ols_incident_power_w": source["ols_incident_power_w"][start:end].clone(),
            "ols_modal_sum_w": source["ols_modal_sum_w"][start:end].clone(),
            "camera_roi_power_w": physical.double().sum((1, 2, 3)),
            "seven_plane_power_w": current_seven,
            "decoder_plane_power_w": decoder_planes,
            "forward_calls": end - start,
        }
        _save_cross_block(block_path, payload)
        _validate_cross_block(block_path, combo, loaded)
        progress["completed_rows_by_combo"][combo] = end
        progress["last_saved_block"] = str(block_path)
        progress["updated_at"] = time.time()
        progress["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved()
        atomic_write_json(output / "progress.json", progress)
        if stop_requested[0] is not None:
            return False
    return True


def _read_cross_combo(combo: str, loaded: dict[str, Any], output: Path) -> dict[str, torch.Tensor]:
    source = loaded["states"][COMBOS[combo]["coefficient_state"]]
    blocks = []
    power_overlay = []
    row_count = int(loaded["population"]["all_rows"].numel())
    for start in range(0, row_count, BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, row_count)
        path = output / combo / f"block_{start:03d}_{end:03d}.pt"
        _require(path.is_file(), f"missing completed cross block: {path}")
        block = _validate_cross_block(path, combo, loaded)
        corrected_camera = block["physical_w_per_pixel"].double().sum((1, 2, 3))
        corrected_seven = torch.cat(
            (source["seven_plane_power_w"][start:end, :2], block["decoder_plane_power_w"]),
            dim=1,
        )
        old_camera = block["camera_roi_power_w"].reshape(-1).double()
        old_seven = block["seven_plane_power_w"].double()
        for offset, row in enumerate(block["rows"].tolist()):
            power_overlay.append({
                "combo": combo,
                "row": int(row),
                "camera_power_w_original": float(old_camera[offset]),
                "camera_power_w_corrected": float(corrected_camera[offset]),
                "camera_power_relative_correction": float((corrected_camera[offset] - old_camera[offset]) / max(abs(corrected_camera[offset]), 1.0e-300)),
                "decoder_planes_w_original": json.dumps(old_seven[offset, 2:].tolist()),
                "decoder_planes_w_corrected": json.dumps(block["decoder_plane_power_w"][offset].double().tolist()),
            })
        block["camera_roi_power_w"] = corrected_camera
        block["seven_plane_power_w"] = corrected_seven
        blocks.append(block)
    fields = (
        "rows", "labels", "target", "coefficients", "physical_w_per_pixel", "modal_power_w",
        "slm_after_modulation_power_w", "ols_incident_power_w", "ols_modal_sum_w",
        "camera_roi_power_w", "seven_plane_power_w", "decoder_plane_power_w",
    )
    combined = {key: torch.cat([block[key] for block in blocks], dim=0) for key in fields}
    _require(torch.equal(combined["coefficients"], source["ols_coefficients"]), f"{combo} combined coefficients differ from source encoder")
    _require(torch.equal(combined["rows"], loaded["population"]["all_rows"]), f"{combo} rows omit, duplicate or reorder samples")
    loaded.setdefault("power_overlay", []).extend(power_overlay)
    return combined


def _power_summary(measured: dict[str, torch.Tensor]) -> dict[str, Any]:
    power = measured["modal_power_w"].double()
    total = power.sum(1)
    n_eff = total.square() / power.square().sum(1).clamp_min(1.0e-300)
    root_count = (power >= 1.0e-9).sum(1)
    slm = measured["slm_after_modulation_power_w"].double().reshape(-1)
    incident = measured["ols_incident_power_w"].double().reshape(-1)
    modal = measured["ols_modal_sum_w"].double().reshape(-1)
    _require(torch.allclose(total, modal, rtol=1.0e-5, atol=1.0e-15), "coefficient power sum differs from saved OLS modal power")
    return {
        "mean_roots_ge_1nw": float(root_count.double().mean()),
        "median_roots_ge_1nw": float(root_count.double().median()),
        "all_100_roots_rows": int((root_count == 100).sum()),
        "weakest_port_power_w_median": float(power.min(1).values.median()),
        "weakest_port_power_w_minimum": float(power.min()),
        "mean_modal_power_w": float(modal.mean()),
        "mean_slm_after_modulation_power_w": float(slm.mean()),
        "mean_ols_incident_power_w": float(incident.mean()),
        "mean_system_coupling_efficiency": float((modal / slm.clamp_min(1.0e-300)).mean()),
        "mean_ols_modal_capture_fraction": float((modal / incident.clamp_min(1.0e-300)).mean()),
        "median_n_eff": float(n_eff.median()),
        "mean_n_eff": float(n_eff.mean()),
        "port_power_share_mean": (power / total[:, None].clamp_min(1.0e-300)).mean(0).tolist(),
    }


def _covariance_and_kernel(coefficients: torch.Tensor) -> dict[str, Any]:
    z = coefficients.detach().cpu().to(torch.complex128)
    pivots = {
        "raw": torch.ones(z.shape[0], dtype=torch.complex128),
        "max_magnitude_port": z[torch.arange(z.shape[0]), z.abs().argmax(1)],
        "fixed_port_45": z[:, 44],
    }
    result = {}
    for name, pivot in pivots.items():
        _require(bool((pivot.abs() > 0).all()), f"{name} phase gauge has a zero pivot")
        aligned = z * (pivot.conj() / pivot.abs())[:, None]
        features = torch.cat((aligned.real, aligned.imag), 1)
        centered = features - features.mean(0, keepdim=True)
        covariance = centered.T @ centered / max(features.shape[0] - 1, 1)
        eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
        total = eigenvalues.sum()
        result[name] = {
            "effective_covariance_rank": float(total.square() / eigenvalues.square().sum().clamp_min(1.0e-300)),
            "top_10_variance_fraction": float(eigenvalues[:10].sum() / total.clamp_min(1.0e-300)),
            "eigenvalues": eigenvalues.tolist(),
        }
    kernel = _phase_power_kernel(z)
    off_diagonal = kernel[~torch.eye(kernel.shape[0], dtype=torch.bool)]
    eigenvalues = torch.linalg.eigvalsh(kernel).clamp_min(0).flip(0)
    total = eigenvalues.sum()
    result["phase_power_invariant_kernel"] = {
        "off_diagonal_mean": float(off_diagonal.mean()),
        "off_diagonal_median": float(off_diagonal.median()),
        "off_diagonal_p95": float(torch.quantile(off_diagonal, 0.95)),
        "effective_kernel_rank": float(total.square() / eigenvalues.square().sum().clamp_min(1.0e-300)),
        "eigenvalues": eigenvalues.tolist(),
    }
    return result


def _detailed_row(
    combo: str, index: int, measured: dict[str, torch.Tensor], gain: float,
    evaluator: torch.nn.Module, atlas_rows: set[int],
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    device = next(evaluator.parameters()).device
    row = int(measured["rows"][index])
    label = int(measured["labels"][index])
    physical = measured["physical_w_per_pixel"][index : index + 1].to(device=device, dtype=torch.float64)
    target = measured["target"][index : index + 1].to(device=device, dtype=torch.float64)
    task = physical * gain
    rows = torch.tensor([row], dtype=torch.long)
    view, psnr, ssim = baseline._reconstruction_batch_metrics(task, target, rows, rows)
    raw28 = F.adaptive_avg_pool2d(task, (28, 28))
    target28 = F.adaptive_avg_pool2d(target, (28, 28))
    luminance, contrast_structure, ssim_map = _ssim_parts(view, target28)
    _require(torch.allclose(ssim_map.mean().reshape(1), ssim, rtol=1.0e-9, atol=1.0e-9), "SSIM local decomposition differs from production metric")
    _, _, raw_ssim_map = _ssim_parts(raw28, target28)
    shape_tv = float(_normalized_shape_tv(raw28, target28)[0])
    region = readout._target_regions(target)
    residual2 = (task - target).square()
    total_sse = float(residual2.sum())
    regions: dict[str, dict[str, float]] = {}
    for name in ("edge", "stroke_interior", "background"):
        mask = region[name].to(device)
        sse = float((residual2 * mask).sum())
        regions[name] = {"sse": sse, "mse": sse / int(mask.sum()), "sse_share": sse / max(total_sse, 1.0e-300)}
    _require(math.isclose(sum(item["sse"] for item in regions.values()), total_sse, rel_tol=1.0e-9, abs_tol=1.0e-12), "regional SSE does not sum to total")
    image_energy, target_energy, gradient_error = _sobel_metrics(view, target28)
    saturation = (task < 0.0) | (task > 1.0)
    excess = (task - 1.0).clamp_min(0.0) + (-task).clamp_min(0.0)
    prediction = int(evaluator(view.float()).argmax(1)[0].cpu())
    power = measured["modal_power_w"][index].double()
    total_power = float(power.sum())
    slm_power = float(measured["slm_after_modulation_power_w"][index])
    ols_incident = float(measured["ols_incident_power_w"][index])
    n_eff = total_power**2 / float(power.square().sum().clamp_min(1.0e-300))
    record: dict[str, Any] = {
        "combo": combo,
        "row": row,
        "label": label,
        "mse400_unclipped": float((task - target).square().mean()),
        "mse400_clipped": float((task.clamp(0.0, 1.0) - target).square().mean()),
        "mse28_unclipped": float((raw28 - target28).square().mean()),
        "mse28_clipped": float((view - target28).square().mean()),
        "ssim28_unclipped": float(raw_ssim_map.mean()),
        "psnr28_db": float(psnr[0]),
        "ssim28": float(ssim[0]),
        "ssim_luminance_mean": float(luminance.mean()),
        "ssim_contrast_structure_mean": float(contrast_structure.mean()),
        "ssim_map_mean": float(ssim_map.mean()),
        "shape_tv_normalized": shape_tv,
        "saturation_fraction": float(saturation.double().mean()),
        "mean_excess_magnitude": float(excess.mean()),
        "saturation_excess_sum": float(excess.sum()),
        "edge_band_target_intensity_sum28": float((target28 * region["edge28"]).sum()),
        "edge_band_prediction_intensity_sum28": float((view * region["edge28"]).sum()),
        "sobel_prediction_gradient_energy": float(image_energy.mean()),
        "sobel_target_gradient_energy": float(target_energy.mean()),
        "sobel_gradient_squared_error": float(gradient_error.mean()),
        "ed_prediction_descriptive": prediction,
        "ols_modal_power_w": total_power,
        "slm_after_modulation_power_w": slm_power,
        "ols_incident_power_w": ols_incident,
        "system_coupling_efficiency": total_power / max(slm_power, 1.0e-300),
        "ols_modal_capture_fraction": total_power / max(ols_incident, 1.0e-300),
        "ports_at_least_1nw": int((power >= 1.0e-9).sum()),
        "weakest_port_power_w": float(power.min()),
        "n_eff": n_eff,
        **{f"{name}_sse": value["sse"] for name, value in regions.items()},
        **{f"{name}_sse_share": value["sse_share"] for name, value in regions.items()},
    }
    del atlas_rows
    heat = {
        "target28": target28.detach().cpu()[0, 0],
        "prediction28": view.detach().cpu()[0, 0],
        "absolute_error28": (view - target28).abs().detach().cpu()[0, 0],
        "ssim_map28": ssim_map.detach().cpu()[0, 0],
        "luminance_map28": luminance.detach().cpu()[0, 0],
        "contrast_structure_map28": contrast_structure.detach().cpu()[0, 0],
        "saturation28": F.adaptive_max_pool2d(saturation.float(), (28, 28)).cpu()[0, 0],
        "saturation400_indices_yx": torch.nonzero(saturation[0, 0], as_tuple=False).to(torch.int32).cpu(),
    }
    return record, heat


def _gain_curve_sample_metrics(
    measured: dict[str, torch.Tensor], gain: float,
) -> list[dict[str, float | int]]:
    device = torch.device("cuda")
    sample_metrics: list[dict[str, float | int]] = []
    rows = measured["rows"]
    for start in range(0, rows.numel(), BLOCK_SIZE):
        end = min(start + BLOCK_SIZE, rows.numel())
        physical = measured["physical_w_per_pixel"][start:end].to(device=device, dtype=torch.float64)
        target = measured["target"][start:end].to(device=device, dtype=torch.float64)
        task = physical * gain
        view, batch_psnr, batch_ssim = baseline._reconstruction_batch_metrics(task, target, rows[start:end], rows[start:end])
        target28 = F.adaptive_avg_pool2d(target, (28, 28))
        raw28 = F.adaptive_avg_pool2d(task, (28, 28))
        sat = (task < 0.0) | (task > 1.0)
        excess = (task - 1.0).clamp_min(0.0) + (-task).clamp_min(0.0)
        raw_ssim = _ssim_parts(raw28, target28)[2].mean((1, 2, 3))
        mse400_raw = (task - target).square().mean((1, 2, 3))
        mse400_clipped = (task.clamp(0.0, 1.0) - target).square().mean((1, 2, 3))
        mse28_raw = (raw28 - target28).square().mean((1, 2, 3))
        mse28_clipped = (view - target28).square().mean((1, 2, 3))
        shape_tv = _normalized_shape_tv(raw28, target28).reshape(-1)
        sat_count = sat.sum((1, 2, 3))
        excess_mean = excess.mean((1, 2, 3))
        for local_index in range(end - start):
            sample_metrics.append({
                "row": int(rows[start + local_index]),
                "mse400_unclipped": float(mse400_raw[local_index]),
                "mse400_clipped": float(mse400_clipped[local_index]),
                "mse28_unclipped": float(mse28_raw[local_index]),
                "mse28_clipped": float(mse28_clipped[local_index]),
                "psnr28_db": float(batch_psnr[local_index]),
                "ssim28": float(batch_ssim[local_index]),
                "ssim28_unclipped": float(raw_ssim[local_index]),
                "shape_tv_normalized": float(shape_tv[local_index]),
                "saturation_count": int(sat_count[local_index]),
                "pixel_count": int(sat[local_index].numel()),
                "mean_excess_magnitude": float(excess_mean[local_index]),
            })
    return sample_metrics


def _gain_curve_record(
    combo: str, gain: float, gain_name: str, population: str,
    sample_metrics: list[dict[str, float | int]], selected_indices: torch.Tensor,
) -> dict[str, Any]:
    _require(population in {"fit100", "confirmation200", "all300"}, f"unknown gain-curve population: {population}")
    indices = selected_indices.long().tolist()
    _require(bool(indices), f"gain curve has an empty {population} population")
    selected = [sample_metrics[index] for index in indices]
    pixel_count = sum(int(item["pixel_count"]) for item in selected)
    _require(pixel_count > 0, "gain-curve pixel population is empty")
    return {
        "combo": combo,
        "gain_name": gain_name,
        "gain_w_inverse": gain,
        "population": population,
        "sample_count": len(selected),
        "mse400_unclipped_mean": statistics.mean(float(item["mse400_unclipped"]) for item in selected),
        "mse400_clipped_mean": statistics.mean(float(item["mse400_clipped"]) for item in selected),
        "mse28_unclipped_mean": statistics.mean(float(item["mse28_unclipped"]) for item in selected),
        "mse28_clipped_mean": statistics.mean(float(item["mse28_clipped"]) for item in selected),
        "psnr28_db_mean": statistics.mean(float(item["psnr28_db"]) for item in selected),
        "ssim28_mean": statistics.mean(float(item["ssim28"]) for item in selected),
        "ssim28_unclipped_mean": statistics.mean(float(item["ssim28_unclipped"]) for item in selected),
        "shape_tv_normalized_mean": statistics.mean(float(item["shape_tv_normalized"]) for item in selected),
        "saturation_fraction": sum(int(item["saturation_count"]) for item in selected) / pixel_count,
        "mean_excess_magnitude": sum(float(item["mean_excess_magnitude"]) * int(item["pixel_count"]) for item in selected) / pixel_count,
    }


def _finalize_gain_curve_records(
    curve_rows: list[dict[str, Any]], gains: dict[str, float],
) -> list[dict[str, Any]]:
    populations = {"fit100": 100, "confirmation200": 200, "all300": 300}
    expected_keys = {
        (population, gain_name, combo)
        for population in populations
        for gain_name in gains
        for combo in COMBOS
    }
    actual_keys = [(row.get("population"), row.get("gain_name"), row.get("combo")) for row in curve_rows]
    _require(len(actual_keys) == len(set(actual_keys)), "gain curve contains duplicate population/gain/combination rows")
    _require(set(actual_keys) == expected_keys, "gain curve population, gain or combination coverage is incomplete")
    index = {(row["population"], row["gain_name"], row["combo"]): row for row in curve_rows}
    for population, expected_count in populations.items():
        rows = [row for row in curve_rows if row["population"] == population]
        _require(len(rows) == len(gains) * len(COMBOS), f"gain curve has missing or mixed rows for {population}")
        _require(all(int(row["sample_count"]) == expected_count for row in rows), f"gain curve count does not match {population}")
        for gain_name in gains:
            baseline_same = index[(population, gain_name, "R00")]
            baseline_fixed = index[(population, "R00_shared_constrained", "R00")]
            _require(
                math.isfinite(float(baseline_same["mse400_unclipped_mean"]))
                and float(baseline_same["mse400_unclipped_mean"]) > 0.0
                and math.isfinite(float(baseline_fixed["mse400_unclipped_mean"]))
                and float(baseline_fixed["mse400_unclipped_mean"]) > 0.0,
                f"gain curve has an invalid R00 denominator for {population}/{gain_name}",
            )
            for combo in COMBOS:
                candidate = index[(population, gain_name, combo)]
                candidate["mse_improvement_vs_same_gain_R00"] = 1.0 - candidate["mse400_unclipped_mean"] / baseline_same["mse400_unclipped_mean"]
                candidate["ssim_delta_vs_same_gain_R00"] = candidate["ssim28_mean"] - baseline_same["ssim28_mean"]
                candidate["same_gain_R00_saturation_fraction"] = baseline_same["saturation_fraction"]
                candidate["same_gain_gate_shaped"] = bool(
                    candidate["mse_improvement_vs_same_gain_R00"] >= MSE_IMPROVEMENT_MIN
                    and candidate["ssim_delta_vs_same_gain_R00"] >= 0.0
                    and candidate["saturation_fraction"] <= SATURATION_MAX
                )
                candidate["mse_improvement_vs_R00_g0"] = 1.0 - candidate["mse400_unclipped_mean"] / baseline_fixed["mse400_unclipped_mean"]
                candidate["ssim_delta_vs_R00_g0"] = candidate["ssim28_mean"] - baseline_fixed["ssim28_mean"]
                candidate["R00_g0_saturation_fraction"] = baseline_fixed["saturation_fraction"]
                candidate["fixed_R00_g0_gate_shaped"] = bool(
                    candidate["mse_improvement_vs_R00_g0"] >= MSE_IMPROVEMENT_MIN
                    and candidate["ssim_delta_vs_R00_g0"] >= 0.0
                    and candidate["saturation_fraction"] <= SATURATION_MAX
                )
    return curve_rows


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    _require(bool(records), f"cannot write empty CSV: {path}")
    columns = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def _compare_historical_main_metrics(source_dir: Path, output_dir: Path) -> dict[str, float]:
    with (source_dir / "per_sample_metrics_g0.csv").open(encoding="utf-8", newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    with (output_dir / "per_sample_metrics_g0.csv").open(encoding="utf-8", newline="") as stream:
        output_rows = list(csv.DictReader(stream))
    def key(row: dict[str, str]) -> tuple[str, int]:
        return row["combo"], int(row["row"])

    source_map = {key(row): row for row in source_rows}
    output_map = {key(row): row for row in output_rows}
    _require(len(source_map) == len(source_rows) == 1200 and set(source_map) == set(output_map), "historical main metric rows are incomplete or mispaired")
    fields = ("mse400_unclipped", "ssim28", "saturation_fraction")
    maximum = {field: 0.0 for field in fields}
    for identity, old in source_map.items():
        current = output_map[identity]
        for field in fields:
            difference = abs(float(old[field]) - float(current[field]))
            maximum[field] = max(maximum[field], difference)
            _require(difference <= 1.0e-12, f"reanalysis changed historical main metric {field} for {identity}")
    return maximum


def _evaluate_all(
    loaded: dict[str, Any], arrays: dict[str, dict[str, torch.Tensor]],
    output: Path, progress: dict[str, Any], stop_requested: list[int | None],
) -> dict[str, Any] | None:
    progress["evaluation_gain_points_completed"] = 0
    progress["detailed_rows_completed"] = 0
    population = loaded["population"]
    fit_rows = population["fit_rows"].long()
    confirm_rows = population["confirm_rows"].long()
    row_to_index = {int(row): i for i, row in enumerate(population["all_rows"].tolist())}
    fit_indices = torch.tensor([row_to_index[int(row)] for row in fit_rows], dtype=torch.long)
    confirm_indices = torch.tensor([row_to_index[int(row)] for row in confirm_rows], dtype=torch.long)
    r00 = arrays["R00"]
    independent_gain = _independent_gain_reference(r00["physical_w_per_pixel"][fit_indices], r00["target"][fit_indices])
    g0 = independent_gain["constrained"]
    previous_gain = float(_json(_resolve(loaded["joint_report_path"]))["fixed_gain"])
    original_gain = float(loaded["references"]["camera_gain"].cpu())
    gains: dict[str, float] = {"R00_shared_constrained": g0, "original_gain": original_gain, "previous_frozen_gain": previous_gain}
    combo_gain_refs = {}
    for combo, measured in arrays.items():
        reference = _independent_gain_reference(measured["physical_w_per_pixel"][fit_indices], measured["target"][fit_indices])
        combo_gain_refs[combo] = reference
        gains[f"{combo}_fit_ls"] = reference["ls"]
        gains[f"{combo}_fit_constrained"] = reference["constrained"]
    for index in range(41):
        ratio = 0.25 + 0.025 * index
        gains[f"curve_{index:02d}_x{ratio:.3f}"] = g0 * ratio

    population_indices = {
        "fit100": fit_indices,
        "confirmation200": confirm_indices,
        "all300": torch.arange(population["all_rows"].numel(), dtype=torch.long),
    }
    curve_rows: list[dict[str, Any]] = []
    for gain_name, gain in gains.items():
        for combo, measured in arrays.items():
            sample_metrics = _gain_curve_sample_metrics(measured, gain)
            _require(len(sample_metrics) == 300, f"gain curve {gain_name}/{combo} did not measure all 300 samples")
            for population_name, indices in population_indices.items():
                curve_rows.append(_gain_curve_record(combo, gain, gain_name, population_name, sample_metrics, indices))
        progress["evaluation_gain_points_completed"] += 1
        progress["updated_at"] = time.time()
        atomic_write_json(output / "progress.json", progress)
        if stop_requested[0] is not None:
            return None

    expected_curve_rows = len(gains) * len(COMBOS) * 3
    _require(len(curve_rows) == expected_curve_rows, "gain curve population/combination coverage is incomplete")
    curve_rows = _finalize_gain_curve_records(curve_rows, gains)

    metric_records: list[dict[str, Any]] = []
    atlas_rows = {
        int(population["confirm_rows_by_class"][label, offset])
        for label in range(10)
        for offset in range(3)
    }
    atlas: dict[str, dict[int, dict[str, torch.Tensor]]] = {combo: {} for combo in COMBOS}
    fit_set = set(fit_rows.tolist())
    row_count = int(population["all_rows"].numel())
    for combo_index, (combo, measured) in enumerate(arrays.items()):
        for index in range(row_count):
            record, heat = _detailed_row(combo, index, measured, g0, loaded["evaluator"], atlas_rows)
            record["population"] = "fit" if int(record["row"]) in fit_set else "confirmation"
            metric_records.append(record)
            if heat is not None:
                atlas[combo][record["row"]] = heat
            if (index + 1) % BLOCK_SIZE == 0:
                progress["detailed_rows_completed"] = combo_index * row_count + index + 1
                progress["updated_at"] = time.time()
                atomic_write_json(output / "progress.json", progress)
                if stop_requested[0] is not None:
                    return None

    metric_by_combo: dict[str, list[dict[str, Any]]] = {
        combo: [row for row in metric_records if row["combo"] == combo] for combo in COMBOS
    }
    row_records = {
        combo: {int(row["row"]): row for row in rows}
        for combo, rows in metric_by_combo.items()
    }
    ssim_factorial_records = []
    for row in sorted(row_records["R00"]):
        base_map = atlas["R00"][row]
        base_record = row_records["R00"][row]
        for combo in ("R01", "R10", "R11"):
            candidate_map = atlas[combo][row]
            candidate_record = row_records[combo][row]
            decomposition = _ssim_symmetric_decomposition(
                base_map["luminance_map28"], base_map["contrast_structure_map28"],
                candidate_map["luminance_map28"], candidate_map["contrast_structure_map28"],
            )
            delta = candidate_record["ssim28"] - base_record["ssim28"]
            _require(abs(decomposition["delta_ssim"] - delta) <= 1.0e-9, f"SSIM summary and maps differ at row {row}, {combo}")
            ssim_factorial_records.append({
                "combo": combo,
                "row": row,
                "label": candidate_record["label"],
                "population": candidate_record["population"],
                "delta_ssim_vs_R00": delta,
                "luminance_symmetric_contribution": decomposition["luminance_contribution"],
                "contrast_structure_symmetric_contribution": decomposition["contrast_structure_contribution"],
                "reconstructed_delta_ssim": decomposition["reconstructed_delta"],
                "identity_absolute_error": decomposition["absolute_error"],
            })
    effects = []
    for row in sorted(row_records["R00"]):
        r00m, r01m, r10m, r11m = (row_records[name][row] for name in ("R00", "R01", "R10", "R11"))
        _require(len({r00m["label"], r01m["label"], r10m["label"], r11m["label"]}) == 1, f"cross-combo labels differ for row {row}")
        effects.append({
            "row": row,
            "label": r00m["label"],
            "encoder_effect_at_D0_delta_mse": r10m["mse400_unclipped"] - r00m["mse400_unclipped"],
            "encoder_effect_at_D1_delta_mse": r11m["mse400_unclipped"] - r01m["mse400_unclipped"],
            "decoder_effect_at_E0_delta_mse": r01m["mse400_unclipped"] - r00m["mse400_unclipped"],
            "decoder_effect_at_E1_delta_mse": r11m["mse400_unclipped"] - r10m["mse400_unclipped"],
            "mse_interaction_R11_minus_R10_minus_R01_plus_R00": r11m["mse400_unclipped"] - r10m["mse400_unclipped"] - r01m["mse400_unclipped"] + r00m["mse400_unclipped"],
            **{f"{name}_mse400": row_records[name][row]["mse400_unclipped"] for name in COMBOS},
            **{f"{name}_ssim28": row_records[name][row]["ssim28"] for name in COMBOS},
        })

    summary_by_combo: dict[str, Any] = {}
    local_gates = {}
    for combo in COMBOS:
        records = metric_by_combo[combo]
        by_population = {}
        for population_name, selected in (("fit", [r for r in records if r["population"] == "fit"]), ("confirmation", [r for r in records if r["population"] == "confirmation"])):
            by_population[population_name] = {
                "count": len(selected),
                "mse400_unclipped_mean": statistics.mean(r["mse400_unclipped"] for r in selected),
                "mse400_clipped_mean": statistics.mean(r["mse400_clipped"] for r in selected),
                "mse28_unclipped_mean": statistics.mean(r["mse28_unclipped"] for r in selected),
                "mse28_clipped_mean": statistics.mean(r["mse28_clipped"] for r in selected),
                "ssim28_unclipped_mean": statistics.mean(r["ssim28_unclipped"] for r in selected),
                "psnr28_db_mean": statistics.mean(r["psnr28_db"] for r in selected),
                "ssim28_mean": statistics.mean(r["ssim28"] for r in selected),
                "shape_tv_normalized_mean": statistics.mean(r["shape_tv_normalized"] for r in selected),
                "saturation_fraction": sum(r["saturation_fraction"] for r in selected) / len(selected),
                "mean_excess_magnitude": statistics.mean(r["mean_excess_magnitude"] for r in selected),
                "ed_macro_recall_descriptive": sum(1 for r in selected if r["ed_prediction_descriptive"] == r["label"]) / len(selected),
            }
        power_state = COMBOS[combo]["coefficient_state"]
        summary_by_combo[combo] = {
            "encoder": COMBOS[combo]["encoder"],
            "decoder": COMBOS[combo]["decoder"],
            "coefficient_state": power_state,
            "g0_metrics": by_population,
            "fit_gain_reference": combo_gain_refs[combo],
            "power": {
                "all300": _power_summary(arrays[combo]),
                "fit100": _power_summary({key: value[fit_indices] for key, value in arrays[combo].items() if isinstance(value, torch.Tensor) and value.shape[:1] == (300,)}),
                "confirmation200": _power_summary({key: value[confirm_indices] for key, value in arrays[combo].items() if isinstance(value, torch.Tensor) and value.shape[:1] == (300,)}),
            },
            "coefficient_diversity": _covariance_and_kernel(arrays[combo]["coefficients"]),
        }
    r00c = summary_by_combo["R00"]["g0_metrics"]["confirmation"]
    for combo in ("R01", "R10", "R11"):
        candidate = summary_by_combo[combo]["g0_metrics"]["confirmation"]
        improvement = 1.0 - candidate["mse400_unclipped_mean"] / r00c["mse400_unclipped_mean"]
        gate = {
            "mse_improvement_fraction_vs_R00": improvement,
            "ssim_delta_vs_R00": candidate["ssim28_mean"] - r00c["ssim28_mean"],
            "confirmation_saturation_fraction": candidate["saturation_fraction"],
            "criteria": {"mse_improvement_min": MSE_IMPROVEMENT_MIN, "ssim_not_lower": True, "saturation_fraction_max": SATURATION_MAX},
            "passed": improvement >= MSE_IMPROVEMENT_MIN and candidate["ssim28_mean"] >= r00c["ssim28_mean"] and candidate["saturation_fraction"] <= SATURATION_MAX,
        }
        local_gates[combo] = gate
    passed_combos = [combo for combo, gate in local_gates.items() if gate["passed"]]
    per_port_records = []
    for combo, measured in arrays.items():
        power = measured["modal_power_w"].double()
        share = power / power.sum(1, keepdim=True).clamp_min(1.0e-300)
        for index, row in enumerate(measured["rows"].tolist()):
            for port in range(100):
                per_port_records.append({"combo": combo, "encoder": COMBOS[combo]["encoder"], "row": row, "label": int(measured["labels"][index]), "port": port, "array_row": port // 10, "array_col": port % 10, "power_w": float(power[index, port]), "power_share": float(share[index, port])})
    _write_csv(output / "per_sample_metrics_g0.csv", metric_records)
    _write_csv(output / "gain_sensitivity.csv", curve_rows)
    _write_csv(output / "paired_factorial_effects.csv", effects)
    _write_csv(output / "per_port_power.csv", per_port_records)
    _write_csv(output / "per_class_summary.csv", _class_summary(metric_records))
    _write_csv(output / "ssim_factorial_decomposition.csv", ssim_factorial_records)
    if loaded.get("power_overlay"):
        _write_csv(output / "corrected_cross_power_overlay.csv", loaded["power_overlay"])
    atomic_torch_save(
        output / "local_metric_maps.pt",
        {combo: {row: values for row, values in combo_maps.items()} for combo, combo_maps in atlas.items()},
    )
    _write_atlas(output / "fixed_confirmation_atlas.png", loaded["labels"], atlas, atlas_rows, g0)
    covariance = {state: _covariance_and_kernel(arrays["R00"]["coefficients"] if state == STATE0 else arrays["R10"]["coefficients"]) for state in (STATE0, STATE1)}
    return {
        "common_gain_w_inverse": g0,
        "common_gain_definition": "min(R00 fit-only least-squares gain, R00 fit-only exact strict-saturation cap)",
        "independent_gain_reference": independent_gain,
        "gain_sensitivity_points": len(gains),
        "gain_sensitivity": {"rows": len(curve_rows), "curve_range_ratio": [0.25, 1.25], "curve_points": 41, "anchors": gains, "populations": {name: int(index.numel()) for name, index in population_indices.items()}, "comparisons": ["same_gain_R00", "fixed_R00_g0"]},
        "combos": summary_by_combo,
        "local_gates_vs_R00": local_gates,
        "decision": "AT_LEAST_ONE_FIXED_COMBINATION_PASSES" if passed_combos else "NO_FIXED_COMBINATION_PASSES",
        "passed_combinations": passed_combos,
        "coefficient_diversity_by_encoder": covariance,
        "ed_confusion_by_combo": _ed_confusions(metric_records),
        "ssim_symmetric_decomposition_rows": len(ssim_factorial_records),
        "local_metric_map_rows": sum(len(values) for values in atlas.values()),
    }


def _class_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for combo in COMBOS:
        for population in ("fit", "confirmation"):
            for label in range(10):
                selected = [r for r in records if r["combo"] == combo and r["population"] == population and r["label"] == label]
                result.append({"combo": combo, "population": population, "label": label, "count": len(selected), "mse400_mean": statistics.mean(r["mse400_unclipped"] for r in selected), "ssim28_mean": statistics.mean(r["ssim28"] for r in selected), "shape_tv_mean": statistics.mean(r["shape_tv_normalized"] for r in selected), "saturation_fraction": sum(r["saturation_fraction"] for r in selected) / len(selected)})
    return result


def _ed_confusions(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for combo in COMBOS:
        selected = [r for r in records if r["combo"] == combo and r["population"] == "confirmation"]
        matrix = [[0 for _ in range(10)] for _ in range(10)]
        for row in selected:
            matrix[int(row["label"])][int(row["ed_prediction_descriptive"])] += 1
        result[combo] = {"confusion_matrix": matrix, "macro_recall": sum(matrix[i][i] / max(sum(matrix[i]), 1) for i in range(10)) / 10, "recall_by_class": [matrix[i][i] / max(sum(matrix[i]), 1) for i in range(10)]}
    return result


def _write_atlas(path: Path, labels: torch.Tensor, atlas: dict[str, dict[int, dict[str, torch.Tensor]]], rows: set[int], gain: float) -> None:
    tile = 112
    columns = ("target", "R00", "R01", "R10", "R11", "R00 error", "R01 error", "R10 error", "R11 error")
    selected = sorted(rows, key=lambda row: (int(labels[row]), row))
    canvas = Image.new("RGB", (tile * len(columns), 24 + tile * len(selected)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, name in enumerate(columns):
        draw.text((col * tile + 2, 2), name, fill="black")
    for row_index, row in enumerate(selected):
        y = 24 + row_index * tile
        draw.text((1, y + 2), f"{int(labels[row])}:{row}", fill="black")
        for col, name in enumerate(columns):
            if name == "target":
                tensor = atlas["R00"][row]["target28"]
                high = False
            elif " error" in name:
                combo = name.split()[0]
                tensor = atlas[combo][row]["absolute_error28"].clamp(0, 1)
                high = True
            else:
                combo = name
                tensor = atlas[combo][row]["prediction28"].clamp(0, 1)
                high = False
            array = (tensor * 255).round().byte().numpy()
            image = Image.fromarray(array, mode="L").resize((tile - 4, tile - 4), Image.Resampling.NEAREST)
            if high:
                image = Image.eval(image, lambda value: 255 - value)
            canvas.paste(image.convert("RGB"), (col * tile + 2, y + 16))
    canvas.save(path)


def _write_report(path: Path, report: dict[str, Any]) -> None:
    analysis = report["analysis"]
    gate = analysis["local_gates_vs_R00"]
    lines = [
        "# LightGen400 冻结交叉解码定位",
        "",
        "## 结果与状态",
        "",
        f"测量状态：`{report['status']}`；判定：`{report['decision']}`。优化器更新为0；本轮新增交叉 Decoder 前向 `{report['formal_cross_decoder_forwards']}` 次，复用旧缓存的历史前向 `{report.get('source_cached_cross_decoder_forwards', 0)}` 次。共同 gain 为 `{report['common_gain_w_inverse']:.8g} W^-1`，只由 R00 拟合集计算。",
        "",
        "|组合|拟合集MSE400|确认MSE400|确认SSIM28|确认饱和|相对R00 MSE改善|Gate|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for combo in COMBOS:
        c = analysis["combos"][combo]["g0_metrics"]["confirmation"]
        improvement = 0.0 if combo == "R00" else gate[combo]["mse_improvement_fraction_vs_R00"]
        status = "基线" if combo == "R00" else ("通过" if gate[combo]["passed"] else "未通过")
        lines.append(f"|{combo}|{analysis['combos'][combo]['g0_metrics']['fit']['mse400_unclipped_mean']:.6f}|{c['mse400_unclipped_mean']:.6f}|{c['ssim28_mean']:.4f}|{c['saturation_fraction']:.3%}|{improvement:.1%}|{status}|")
    passed_combinations = analysis["passed_combinations"]
    lines.extend(["", "## 科学认识", "", f"统一读出下达到预设局部门槛的组合：{', '.join(passed_combinations) if passed_combinations else '无'}。gain 曲线现按 fit100、confirmation200、all300 分开，并同时比较同 gain 的 R00 与固定 R00-g0；曲线只作敏感性分析。交叉功率修正只影响相机总功率和七平面元数据，不改变保存的像素输出、OLS系数或主表重建指标。SSIM亮度与对比度—结构项使用对称恒等式分解，不作训练因果归因。", "", "## 目标缺口", "", "本结果限于固定300张TRAIN、两组冻结相位和4096离散传播；没有新潜变量、DEV/TEST或生成验收，不能推断硬件可读性。", "", "## 工程完成与验证", "", f"新预检状态 `{report['preflight_status']}`；交叉4096/6144窗口覆盖 {len(report['window_checks'])} 组，真实Decoder保存与独立进程恢复记录见 `preflight.json`。纠正功率覆盖、分人口gain曲线、全1200条局部图及SSIM分解见报告所列产物。旧 attempt_01 报告和缓存未覆盖。", "", "## 决策影响与下一步", "", "本轮到此停止。若R11达标，统一读出下的联合端点收益获得支持；若某交叉组合达标，后续可围绕对应冻结匹配关系另行规划；若均未达标，仅保留固定R00-g0条件下无合格组合的结论，gain敏感性结果不能替代该Gate。本报告不授权训练或模型采用。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    saved = path.read_text(encoding="utf-8")
    for heading in ("结果与状态", "科学认识", "目标缺口", "工程完成与验证", "决策影响与下一步"):
        _require(f"## {heading}" in saved, f"final report omitted required section: {heading}")


def _read_cross_outputs(loaded: dict[str, Any], output: Path) -> dict[str, dict[str, torch.Tensor]]:
    arrays = _cache_arrays(loaded)
    for combo in ("R00", "R11"):
        source = loaded["states"][COMBOS[combo]["coefficient_state"]]
        arrays[combo].update({key: source[key] for key in ("rows", "labels", "target", "ols_coefficients", "physical_w_per_pixel", "modal_power_w", "slm_after_modulation_power_w", "ols_incident_power_w", "ols_modal_sum_w", "camera_roi_power_w", "seven_plane_power_w")})
        arrays[combo]["coefficients"] = source["ols_coefficients"]
    for combo in CROSS_COMBOS:
        arrays[combo] = _read_cross_combo(combo, loaded, output)
    return arrays


def _run(args: argparse.Namespace) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "CUDA is required for cross-decode measurement")
    preflight, receipt = _ready(args)
    output = _resolve(args.output_dir)
    if args.mode in ("run", "reanalyze"):
        _require(not output.exists() and args.resume_checkpoint is None, "fresh run requires a new output directory")
    else:
        _require(output.is_dir() and args.resume_checkpoint is not None, "resume requires an existing output directory and explicit checkpoint")
    loaded = _load_context(args, torch.device("cuda"))
    if args.mode == "reanalyze":
        _validate_source_cross_run(args, loaded)
    saved_population = torch.load(_resolve(args.preflight_dir) / "fixed_population.pt", map_location="cpu", weights_only=False)
    _require(baseline._assert_state_close(loaded["population"], saved_population, "crossdecode_fixed_population") == 0.0, "fixed population differs from cross-decode READY")
    identity = ResearchRunIdentity.from_mapping(preflight["run_identity"])
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    if args.mode in ("run", "reanalyze"):
        control.open_scientific_attempt(output, attempt_id=output.name, receipt=receipt)
        manifest = {
            "schema": "chen_lightgen_400_frozen_crossdecode_run.v1", "task_id": TASK_ID,
            "run_id": identity.run_id, "attempt_id": output.name,
            "source_audit_dir": str(loaded["source_audit_dir"]),
            "source_readout_report": str(_resolve(args.readout_report)),
            "baseline_checkpoint": str(loaded["source_checkpoint_path"]),
            "start_state": str(loaded["start_path"]), "endpoint_checkpoint": str(loaded["endpoint_path"]),
            "joint_report": str(loaded["joint_report_path"]), "fixed_population": str(loaded["fixed_path"]),
            "preflight_dir": str(_resolve(args.preflight_dir)), "previous_cross_preflight_dir": str(_resolve(args.previous_cross_preflight_dir)),
            "source_cross_dir": str(_resolve(args.source_cross_dir)), "grid": GRID,
            "combos": COMBOS, "analysis_mode": "reanalyze_cached_cross_outputs" if args.mode == "reanalyze" else "cross_forward",
            "cross_forward_budget": 0 if args.mode == "reanalyze" else 600, "optimizer_updates": 0,
        }
        atomic_write_new_json(output / "run_manifest.json", manifest)
        atomic_torch_save_new(output / "fixed_population.pt", loaded["population"])
        progress = {
            "schema": "chen_lightgen_400_frozen_crossdecode_progress.v1",
            "completed_rows_by_combo": {combo: 0 for combo in CROSS_COMBOS},
            "cross_forward_calls": 0, "source_cached_cross_forward_calls": 600 if args.mode == "reanalyze" else 0,
            "evaluation_gain_points_completed": 0,
            "detailed_rows_completed": 0, "started_at": time.time(), "elapsed_before_resume": 0.0,
        }
    else:
        control.open_scientific_attempt(output, attempt_id=output.name, receipt=receipt, resume=True)
        manifest = _json(output / "run_manifest.json")
        expected_analysis_mode = manifest.get("analysis_mode")
        expected_forward_budget = 0 if expected_analysis_mode == "reanalyze_cached_cross_outputs" else 600
        _require(manifest.get("task_id") == TASK_ID and manifest.get("run_id") == identity.run_id and manifest.get("cross_forward_budget") == expected_forward_budget and manifest.get("optimizer_updates") == 0 and manifest.get("combos") == COMBOS, "resume manifest differs from the approved task")
        checkpoint = torch.load(_resolve(args.resume_checkpoint), map_location="cpu", weights_only=False)
        _require(checkpoint.get("task_id") == TASK_ID and checkpoint.get("run_id") == identity.run_id and checkpoint.get("attempt_id") == output.name, "resume checkpoint belongs to a different run")
        progress = checkpoint["progress"]
        expected_source = {"source_audit_dir": str(loaded["source_audit_dir"]), "source_readout_report": str(_resolve(args.readout_report)), "baseline_checkpoint": str(loaded["source_checkpoint_path"]), "start_state": str(loaded["start_path"]), "endpoint_checkpoint": str(loaded["endpoint_path"]), "joint_report": str(loaded["joint_report_path"]), "fixed_population": str(loaded["fixed_path"]), "preflight_dir": str(_resolve(args.preflight_dir)), "previous_cross_preflight_dir": str(_resolve(args.previous_cross_preflight_dir)), "source_cross_dir": str(_resolve(args.source_cross_dir))}
        _require(all(manifest.get(key) == value for key, value in expected_source.items()), "resume source paths differ")
    progress["resumed_at"] = time.time() if args.mode == "resume" else None
    with (output / "segments.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "run_started" if args.mode in ("run", "reanalyze") else "resumed", "analysis_mode": manifest.get("analysis_mode"), "time": time.time(), "resume_checkpoint": str(_resolve(args.resume_checkpoint)) if args.resume_checkpoint else None}) + "\n")
    stop_requested, previous_handlers = _signal_flag()
    heartbeat_stop = threading.Event()

    def heartbeat_worker() -> None:
        while not heartbeat_stop.wait(60.0):
            atomic_write_json(output / "heartbeat.json", {"time": time.time(), "alive": True, "stage": progress.get("stage"), "completed_rows_by_combo": progress.get("completed_rows_by_combo"), "cross_forward_calls": progress.get("cross_forward_calls"), "elapsed_seconds": progress.get("elapsed_before_resume", 0.0) + time.time() - progress.get("started_at", time.time())})

    thread = threading.Thread(target=heartbeat_worker, name="crossdecode-heartbeat", daemon=True)
    thread.start()
    start_time = time.perf_counter()
    try:
        if manifest.get("analysis_mode") == "reanalyze_cached_cross_outputs":
            progress["stage"] = "cached_reanalysis"
            source_cross_dir = _resolve(args.source_cross_dir)
            arrays = _read_cross_outputs(loaded, source_cross_dir)
            progress["cross_forward_calls"] = 0
            progress["source_cached_cross_forward_calls"] = 600
        else:
            progress["stage"] = "measurement"
            models = _current_models(loaded)
            for combo in CROSS_COMBOS:
                completed = _measure_cross_combo(combo, models[COMBOS[combo]["decoder"]], loaded, output, progress, stop_requested)
                atomic_write_json(output / "progress.json", progress)
                if not completed or stop_requested[0] is not None:
                    checkpoint_path = _save_recovery(output, progress, identity)
                    with (output / "segments.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"event": "interrupted", "signal": stop_requested[0], "resume_checkpoint": checkpoint_path.name, "time": time.time()}) + "\n")
                    atomic_write_json(output / "heartbeat.json", {"time": time.time(), "alive": False, "status": "INTERRUPTED", "resume_checkpoint": str(checkpoint_path), "cross_forward_calls": progress["cross_forward_calls"]})
                    return {"status": "INTERRUPTED", "task_id": TASK_ID, "resume_checkpoint": str(checkpoint_path), "cross_forward_calls": progress["cross_forward_calls"]}
            expected_calls = sum(
                int(_validate_cross_block(output / combo / f"block_{start:03d}_{min(start + BLOCK_SIZE, 300):03d}.pt", combo, loaded)["forward_calls"])
                for combo in CROSS_COMBOS
                for start in range(0, 300, BLOCK_SIZE)
            )
            _require(expected_calls == 600, "completed cross blocks do not contain exactly 600 new Decoder forwards")
            progress["cross_forward_calls"] = expected_calls
            arrays = _read_cross_outputs(loaded, output)
        progress["stage"] = "evaluation"
        atomic_write_json(output / "progress.json", progress)
        analysis = _evaluate_all(loaded, arrays, output, progress, stop_requested)
        if analysis is None or stop_requested[0] is not None:
            checkpoint_path = _save_recovery(output, progress, identity)
            with (output / "segments.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"event": "evaluation_interrupted", "signal": stop_requested[0], "resume_checkpoint": checkpoint_path.name, "time": time.time(), "cross_forward_calls": progress["cross_forward_calls"]}) + "\n")
            return {"status": "INTERRUPTED", "task_id": TASK_ID, "resume_checkpoint": str(checkpoint_path), "cross_forward_calls": progress["cross_forward_calls"], "note": "resume will reuse all validated optical blocks and perform no Decoder forwards"}
        historical_metric_comparison = None
        if manifest.get("analysis_mode") == "reanalyze_cached_cross_outputs":
            historical_metric_comparison = _compare_historical_main_metrics(_resolve(args.source_cross_dir), output)
        report = {
            "schema": "chen_lightgen_400_frozen_crossdecode_report.v1", "status": "VALID_COMPLETED",
            "task_id": TASK_ID, "authorization_id": AUTHORITY, "completed_updates": 0,
            "analysis_mode": manifest.get("analysis_mode"),
            "formal_cross_decoder_forwards": progress["cross_forward_calls"],
            "source_cached_cross_decoder_forwards": progress.get("source_cached_cross_forward_calls", 0),
            "historical_main_metric_max_abs_difference": historical_metric_comparison,
            "preflight_optical_forwards": preflight["preflight_optical_forwards"],
            "data_roles": {"canonical_train": True, "fit": 100, "confirmation": 200, "dev_reads": 0, "official_test_reads": 0, "new_latent_samples": 0},
            "source_audit_report_identity": {"source_audit_dir": str(loaded["source_audit_dir"]), "source_run_id": loaded["source_manifest"]["run_id"], "source_decision": loaded["readout_report"]["decision"]},
            "preflight_status": "READY", "window_checks": preflight["cross_window_checks"],
            "common_gain_w_inverse": analysis["common_gain_w_inverse"],
            "decision": analysis["decision"],
            "analysis": analysis,
            "artifacts": {"source_cross_outputs": str(_resolve(args.source_cross_dir)), "metrics": "per_sample_metrics_g0.csv", "gain_sensitivity": "gain_sensitivity.csv", "corrected_power_overlay": "corrected_cross_power_overlay.csv", "ssim_decomposition": "ssim_factorial_decomposition.csv", "local_maps": "local_metric_maps.pt", "power": "per_port_power.csv", "effects": "paired_factorial_effects.csv", "atlas": "fixed_confirmation_atlas.png"},
            "elapsed_seconds": progress.get("elapsed_before_resume", 0.0) + time.perf_counter() - start_time,
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(),
            "resource_telemetry": "resource_telemetry.csv (AutoDL watcher)",
        }
        required_outputs = ("per_sample_metrics_g0.csv", "gain_sensitivity.csv", "per_port_power.csv", "paired_factorial_effects.csv", "ssim_factorial_decomposition.csv", "local_metric_maps.pt", "fixed_confirmation_atlas.png")
        _require(all((output / name).is_file() and (output / name).stat().st_size > 0 for name in required_outputs), "analysis output readback is incomplete")
        curve_rows_read = list(csv.DictReader((output / "gain_sensitivity.csv").open(encoding="utf-8", newline="")))
        _require(len(curve_rows_read) == 624 and {row["population"] for row in curve_rows_read} == {"fit100", "confirmation200", "all300"}, "saved gain curve population coverage is incomplete")
        metrics_read = list(csv.DictReader((output / "per_sample_metrics_g0.csv").open(encoding="utf-8", newline="")))
        _require(len(metrics_read) == 1200, "saved four-combination per-sample metric coverage is incomplete")
        ssim_rows_read = list(csv.DictReader((output / "ssim_factorial_decomposition.csv").open(encoding="utf-8", newline="")))
        _require(len(ssim_rows_read) == 900 and max(float(row["identity_absolute_error"]) for row in ssim_rows_read) <= 1.0e-9, "saved SSIM symmetric decomposition is incomplete or invalid")
        maps_read = torch.load(output / "local_metric_maps.pt", map_location="cpu", weights_only=False)
        _require(set(maps_read) == set(COMBOS) and all(len(maps_read[combo]) == 300 for combo in COMBOS), "saved local SSIM map coverage is incomplete")
        if loaded.get("power_overlay"):
            overlay_rows = list(csv.DictReader((output / "corrected_cross_power_overlay.csv").open(encoding="utf-8", newline="")))
            _require(len(overlay_rows) == 600, "corrected cross power overlay is incomplete")
        atomic_write_json(output / "progress.json", {**progress, "stage": "complete", "updated_at": time.time()})
        with (output / "segments.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": "completed", "time": time.time(), "decision": analysis["decision"]}) + "\n")
        _write_report(output / "results_analysis.md", report)
        _require((output / "results_analysis.md").is_file() and (output / "results_analysis.md").stat().st_size > 0, "final Markdown report was not saved")
        atomic_write_new_json(output / "report.json", report)
        _require(_json(output / "report.json") == report, "final report readback differs")
        return report
    finally:
        heartbeat_stop.set()
        thread.join()
        _restore_signals(previous_handlers)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "resume", "reanalyze", "recovery-worker"), required=True)
    parser.add_argument("--recovery-dir", type=Path)
    parser.add_argument("--recovery-stage", choices=("measurement", "evaluation"))
    parser.add_argument("--recovery-combo", choices=CROSS_COMBOS)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline-report", type=Path)
    parser.add_argument("--source-audit-dir", type=Path)
    parser.add_argument("--source-cross-dir", type=Path)
    parser.add_argument("--previous-cross-preflight-dir", type=Path)
    parser.add_argument("--source-preflight-dir", type=Path)
    parser.add_argument("--readout-report", type=Path)
    parser.add_argument("--start-state", type=Path)
    parser.add_argument("--endpoint-checkpoint", type=Path)
    parser.add_argument("--joint-report", type=Path)
    parser.add_argument("--fixed-population", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/research/lightgen/chen_lightgen_400_mnist_baseline_v1.yaml"))
    parser.add_argument("--data", type=Path, default=Path("data/MNIST/processed/training.pt"))
    parser.add_argument("--evaluator", type=Path)
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.mode == "recovery-worker":
        _require(args.recovery_dir is not None and args.recovery_stage is not None and args.recovery_combo is not None, "recovery-worker requires an explicit directory, stage and pairing")
        context_required = ("checkpoint", "baseline_report", "source_audit_dir", "source_preflight_dir", "readout_report", "start_state", "endpoint_checkpoint", "joint_report", "fixed_population")
        _require(all(getattr(args, name) is not None for name in context_required), "recovery-worker requires explicit model, report, data and population inputs")
        return args
    required = ("checkpoint", "baseline_report", "source_audit_dir", "source_cross_dir", "previous_cross_preflight_dir", "source_preflight_dir", "readout_report", "start_state", "endpoint_checkpoint", "joint_report", "fixed_population", "output_dir")
    _require(all(getattr(args, name) is not None for name in required), "explicit checkpoint, reports, source audit, population and output paths are required")
    _require((args.mode == "preflight" and args.preflight_dir is None) or (args.mode in ("run", "resume", "reanalyze") and args.preflight_dir is not None), "preflight omits --preflight-dir; run/resume/reanalyze require it")
    _require((args.mode == "resume") == (args.resume_checkpoint is not None), "resume mode requires an explicit recovery checkpoint")
    return args


def main() -> None:
    args = _parse_args()
    if args.mode == "recovery-worker":
        raise SystemExit(_recovery_worker(args))
    try:
        report = _preflight(args) if args.mode == "preflight" else _run(args)
        print(json.dumps({key: report.get(key) for key in ("status", "task_id", "decision", "common_gain_w_inverse", "resume_checkpoint")}, ensure_ascii=False))
    except BaseException as error:
        output = _resolve(args.output_dir)
        failure = {"status": "INVALID_HARNESS", "task_id": TASK_ID, "error_type": type(error).__name__, "error": str(error), "completed_updates": 0}
        if args.mode == "preflight" and output.is_dir() and not (output / "preflight_failure.json").exists():
            atomic_write_new_json(output / "preflight_failure.json", failure)
        elif args.mode in ("run", "resume", "reanalyze") and output.is_dir() and not (output / "report.json").exists():
            manifest_path = output / "run_manifest.json"
            progress_path = output / "progress.json"
            if manifest_path.is_file() and progress_path.is_file():
                manifest = _json(manifest_path)
                preflight = _json(_resolve(args.preflight_dir) / "preflight.json")
                identity = ResearchRunIdentity.from_mapping(preflight["run_identity"])
                _require(identity.run_id == manifest.get("run_id"), "cannot save recovery for a different run identity")
                recovery_path = _save_recovery(output, _json(progress_path), identity)
                failure["resume_checkpoint"] = str(recovery_path)
            with (output / "execution_errors.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
        raise


if __name__ == "__main__":
    main()
