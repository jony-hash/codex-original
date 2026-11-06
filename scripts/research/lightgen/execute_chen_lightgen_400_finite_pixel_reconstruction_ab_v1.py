"""Qualify a finite-pixel LightGen400 path before its bounded MSE/SSIM A/B."""

from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    PIXEL_PITCH_M,
    SOURCE_POWER_W,
    WAVELENGTH_M,
)
from ldm_project.v2.application.research.lightgen.chen_lightgen_400_finite_pixel import (
    FinitePixelChenLightGen400,
    FinitePixelConfig,
    FinitePixelRSLinear,
    _legendre_rule,
    _relative_l2,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    ResearchRunControl,
    ResearchRunIdentity,
    capture_execution_source_manifest,
)
from ldm_project.v2.evidence import atomic_torch_save_new, atomic_write_json, atomic_write_new_json
from scripts.research.lightgen import diagnose_chen_lightgen_400_quickcheck_v1 as quickcheck
from scripts.research.lightgen import diagnose_chen_lightgen_400_sampling_closure_v1 as sampling
from scripts.research.lightgen import execute_chen_lightgen_400_mnist_baseline_v1 as baseline


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "CHEN_LIGHTGEN_400_RS_CLOSURE_RECONSTRUCTION_AB_V1"
AUTHORITY = "USER_APPROVED_CHEN_LIGHTGEN_400_RS_CLOSURE_RECONSTRUCTION_AB_20261018"
PROPAGATION_BACKEND = "rs_linear"
ORDERS = (2, 4, 8, 12)
CANDIDATE_ORDERS = ORDERS[:-1]
DIRECT_RS_ORDERS = (2, 4, 8, 12, 16)
COMMON_ROI_PIXELS = 500
WINDOW_TOLERANCE = 1.0e-4
FD_TOLERANCE = 1.0e-2
FD_STEPS_RAD = (2.5e-3, 1.25e-3)
DIRECT_RS_WINDOW_CELLS = 1536
FIXED_GAIN = 3.17684599058831e10
UPDATES = 500
LOGICAL_BATCH = 10
CHECKPOINT_INTERVAL = 100
TASK_SIZE = 28


class HarnessCheckError(RuntimeError):
    """A predeclared contract check rejected an input or measurement path."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HarnessCheckError(message)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _numerical_qualification_status(
    measurements: dict[str, float], required_keys: set[str], tolerance: float
) -> str:
    """Compute the status from executed residuals; absent checks are invalid."""
    if not required_keys or not required_keys.issubset(measurements):
        return "INVALID_HARNESS"
    values = [measurements[key] for key in required_keys]
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        return "INVALID_HARNESS"
    return "NUMERICALLY_QUALIFIED" if max(values) <= tolerance else "NUMERICAL_NOT_QUALIFIED"


def _load_inputs(args: argparse.Namespace, device: torch.device):
    paths = {
        name: _resolve(getattr(args, name))
        for name in ("source_checkpoint", "source_report", "source_start_state", "source_endpoint_checkpoint",
                     "fixed_population", "config", "data", "evaluator")
    }
    config, checkpoint, report, source_model, references = quickcheck._load_inputs(
        paths["source_checkpoint"], paths["source_report"], paths["config"], device
    )
    _require(paths["source_endpoint_checkpoint"].is_file(), "explicit historical endpoint checkpoint is missing")
    start_state = torch.load(paths["source_start_state"], map_location="cpu", weights_only=False)
    endpoint_payload = torch.load(paths["source_endpoint_checkpoint"], map_location="cpu", weights_only=False)
    _require(isinstance(start_state, dict) and "encoder_phase" in start_state and "decoder_phase" in start_state,
             "explicit source start state does not contain both phase groups")
    _require(endpoint_payload.get("completed_updates") == 500 and endpoint_payload.get("grid") == 4096
             and endpoint_payload.get("source_checkpoint_step") == 30_000,
             "explicit historical endpoint is not the intended 500-step 4096 run")
    _require(endpoint_payload.get("task_id") == "CHEN_LIGHTGEN_400_JOINT_RECONSTRUCTION_V1",
             "historical endpoint belongs to a different experiment")
    for name in ("encoder_phase", "decoder_phase"):
        _require(torch.equal(start_state[name], source_model.state_dict()[name].cpu()),
                 f"source start {name} differs from the explicit 30000-step checkpoint")
        _require(tuple(endpoint_payload["model"][name].shape) == tuple(start_state[name].shape),
                 f"historical endpoint {name} shape changed")
    pixels, labels, train_indices, dev_indices = baseline._load_data(paths["data"])
    _require(dev_indices.numel() == 5_000, "canonical data split is not the expected TRAIN/DEV split")
    population = quickcheck._load_fixed_population(paths["fixed_population"], labels, train_indices)
    evaluator = baseline._load_evaluator(paths["evaluator"], device)
    _require(config["optics"]["active_shape"] == [400, 400]
             and math.isclose(config["optics"]["pixel_pitch_m"], PIXEL_PITCH_M)
             and math.isclose(config["optics"]["wavelength_m"], WAVELENGTH_M),
             "source optics configuration differs from the fixed LightGen400 geometry")
    _require(float(checkpoint["bogt_weights"]["rec"]) > 0.0,
             "source checkpoint reconstruction coefficient is invalid")
    return paths, config, checkpoint, report, source_model, references, start_state, endpoint_payload, \
        pixels, labels, population, evaluator


def _model_from_phases(
    phases: dict[str, torch.Tensor], order: int, window: int, device: torch.device,
    propagation_backend: str = PROPAGATION_BACKEND,
    fft_embedding_extra_cells: int = 0,
) -> FinitePixelChenLightGen400:
    model = FinitePixelChenLightGen400(FinitePixelConfig(
        quadrature_order=order, window_cells=window, distance_m=3.0e-3,
        checkpoint_propagation=True, propagation_backend=propagation_backend,
        fft_embedding_extra_cells=fft_embedding_extra_cells,
    )).to(device)
    with torch.no_grad():
        model.encoder_phase.copy_(phases["encoder_phase"].to(device))
        model.decoder_phase.copy_(phases["decoder_phase"].to(device))
    model.requires_grad_(False)
    return model


def _phase_state(model: FinitePixelChenLightGen400) -> dict[str, torch.Tensor]:
    return {name: getattr(model, name).detach().cpu().clone()
            for name in ("encoder_phase", "decoder_phase")}


def _make_rs_patterns() -> dict[str, np.ndarray]:
    pixel_count = 32
    centers = (np.arange(pixel_count, dtype=np.float64) - 16.0 + 0.5) * PIXEL_PITCH_M
    yy, xx = np.meshgrid(centers, centers, indexing="ij")
    amplitude = np.exp(-(xx**2 + yy**2) / (2.0 * (18.0e-6)**2))
    iy, ix = np.meshgrid(np.arange(pixel_count), np.arange(pixel_count), indexing="ij")
    return {
        "gaussian": amplitude.astype(np.complex128),
        "tilted": (amplitude * np.exp(2j * math.pi * (3.0 * xx - 2.0 * yy)
                                         / (pixel_count * PIXEL_PITCH_M))).astype(np.complex128),
        "checkerboard": (amplitude * np.exp(1j * math.pi * ((ix + iy) % 2))).astype(np.complex128),
    }


def _select_rs_center_probes(center_field: torch.Tensor | None) -> torch.Tensor:
    _require(center_field is not None, "independent RS observation omitted pixel centers")
    _require(center_field.ndim == 3 and tuple(center_field.shape[-2:]) == (32, 32),
             "independent RS center field must have [batch, height, width] layout")
    return center_field[0, 2:32:6, 2:32:6].reshape(-1)


def _direct_rs_path_check(device: torch.device, outdir: Path) -> dict[str, Any]:
    patterns = _make_rs_patterns()
    records: dict[str, Any] = {}
    required: set[str] = set()
    residuals: dict[str, float] = {}
    execution_devices: dict[str, str] = {}
    for name, pattern in patterns.items():
        production_pattern = np.asarray(pattern, dtype=np.complex64).astype(np.complex128)
        source_quantization_max_abs = float(np.max(np.abs(production_pattern - pattern)))
        direct_values: dict[int, np.ndarray] = {}
        direct_derivatives: dict[int, complex] = {}
        for order in (8, 12, 16):
            direct_values[order], direct_derivatives[order] = \
                sampling._rayleigh_sommerfeld_pixel_integral(production_pattern, order)
        reference = direct_values[16]
        reference_derivative = direct_derivatives[16]
        convergence_8_12 = float(np.linalg.norm(direct_values[8] - direct_values[12])
                                 / max(np.linalg.norm(direct_values[12]), 1.0e-300))
        convergence_12_16 = float(np.linalg.norm(direct_values[12] - reference)
                                   / max(np.linalg.norm(reference), 1.0e-300))
        required.update({f"{name}:reference_8_vs_12", f"{name}:reference_12_vs_16"})
        residuals[f"{name}:reference_8_vs_12"] = convergence_8_12
        residuals[f"{name}:reference_12_vs_16"] = convergence_12_16
        derivative_convergence = float(abs(direct_derivatives[12] - direct_derivatives[16])
                                       / max(abs(direct_derivatives[16]), 1.0e-14))
        required.add(f"{name}:reference_derivative_12_vs_16")
        residuals[f"{name}:reference_derivative_12_vs_16"] = derivative_convergence
        atomic_write_json(outdir / "direct_rs_progress.json", {
            "status": "RUNNING", "pattern": name, "stage": "reference_convergence",
            "completed_residuals": residuals.copy(), "completed_patterns": records.copy()})
        order_rows: dict[str, Any] = {}
        for order in DIRECT_RS_ORDERS:
            order_device = device
            execution_devices[f"{name}:order_{order}"] = order_device.type
            config = FinitePixelConfig(
                quadrature_order=order, window_cells=DIRECT_RS_WINDOW_CELLS,
                distance_m=3.0e-3, checkpoint_propagation=False,
                propagation_backend=PROPAGATION_BACKEND,
            )
            propagator = FinitePixelRSLinear(order, (32, 32), config).to(order_device)
            source = torch.as_tensor(production_pattern, dtype=torch.complex64, device=order_device)
            base = source.unsqueeze(0).unsqueeze(0).expand(1, order**2, -1, -1)
            center_mask = torch.zeros((1, 1, 32, 32), dtype=torch.float32, device=order_device)
            center_mask[0, 0, 16, 16] = 1.0
            fields, _, center_field = propagator(base, output_shape=(32, 32), capture_center=True)
            selected = _select_rs_center_probes(center_field)
            values = selected.detach().cpu().numpy().astype(np.complex128)
            overlap = np.vdot(reference, values)
            phase = np.exp(-1j * np.angle(overlap))
            value_error = float(np.linalg.norm(phase * values - reference) /
                                max(np.linalg.norm(reference), 1.0e-300))
            required.add(f"{name}:order_{order}:value")
            residuals[f"{name}:order_{order}:value"] = value_error
            derivative_source = 1j * base * center_mask
            _, _, derivative_field = propagator(
                derivative_source, output_shape=(32, 32), capture_center=True
            )
            derivative_probe = _select_rs_center_probes(derivative_field)[12]
            propagated_derivative = complex(derivative_probe.detach().cpu()) * phase
            derivative_error = float(abs(propagated_derivative - reference_derivative)
                                      / max(abs(direct_derivatives[16]), 1.0e-14))
            required.add(f"{name}:order_{order}:derivative")
            residuals[f"{name}:order_{order}:derivative"] = derivative_error
            order_rows[str(order)] = {"value_relative_l2_after_global_phase": value_error,
                                      "phase_alignment_rad": float(np.angle(overlap)),
                                      "derivative_relative_error": derivative_error}
            records[name] = {"reference_8_vs_12_relative_l2": convergence_8_12,
                             "reference_12_vs_16_relative_l2": convergence_12_16,
                             "reference_derivative_12_vs_16_relative_error": derivative_convergence,
                             "orders": order_rows.copy(), "reference_norm": float(np.linalg.norm(reference)),
                             "reference_derivative": [float(direct_derivatives[16].real),
                                                       float(direct_derivatives[16].imag)]}
            atomic_write_json(outdir / "direct_rs_progress.json", {
                "status": "RUNNING", "pattern": name, "stage": f"order_{order}",
                "completed_residuals": residuals.copy(), "completed_patterns": records.copy()})
            del propagator, fields, center_field, selected, base, source
            if order_device.type == "cuda":
                torch.cuda.synchronize(order_device)
                torch.cuda.empty_cache()
        records[name] = {"source_quantization_max_abs": source_quantization_max_abs,
                         "reference_8_vs_12_relative_l2": convergence_8_12,
                         "reference_12_vs_16_relative_l2": convergence_12_16,
                         "reference_derivative_12_vs_16_relative_error": derivative_convergence,
                         "orders": order_rows,
                         "reference_norm": float(np.linalg.norm(reference)),
                         "reference_derivative": [float(reference_derivative.real),
                                                   float(reference_derivative.imag)]}
    reference_keys = {
        key for key in required
        if key.endswith(":reference_8_vs_12") or key.endswith(":reference_12_vs_16")
        or key.endswith(":order_16:value")
    }
    reference_derivative_keys = {
        key for key in required if key.endswith(":reference_derivative_12_vs_16")
        or key.endswith(":order_16:derivative")
    }
    reference_status = _numerical_qualification_status(residuals, reference_keys, WINDOW_TOLERANCE)
    derivative_status = _numerical_qualification_status(
        residuals, reference_derivative_keys, WINDOW_TOLERANCE
    )
    candidate_status = {}
    for order in CANDIDATE_ORDERS:
        candidate_keys = {
            key for pattern in patterns
            for key in (f"{pattern}:order_{order}:value", f"{pattern}:order_{order}:derivative")
        }
        candidate_status[str(order)] = _numerical_qualification_status(
            residuals, candidate_keys, WINDOW_TOLERANCE)
    return {"patterns": records, "measurements": residuals, "required_measurements": sorted(required),
            "reference_status": reference_status, "reference_derivative_status": derivative_status,
            "candidate_status": candidate_status,
            "status": "NUMERICALLY_QUALIFIED" if reference_status == derivative_status == "NUMERICALLY_QUALIFIED"
            else "NUMERICAL_NOT_QUALIFIED",
            "tolerance": WINDOW_TOLERANCE, "fft_window_cells": DIRECT_RS_WINDOW_CELLS,
            "derivative_tolerance": WINDOW_TOLERANCE,
            "execution_devices": execution_devices,
            "derivative_method": "linear superposition: propagate i times the perturbed physical-pixel field",
            "reference": "independent direct real-space Rayleigh-Sommerfeld pixel integral, GL orders 8/12/16",
            "candidate_backend": PROPAGATION_BACKEND}


def _align_many(candidate: list[torch.Tensor], reference: list[torch.Tensor]) -> tuple[float, dict[str, float]]:
    _require(len(candidate) == len(reference) and bool(candidate), "field comparison set is incomplete")
    overlap = torch.zeros((), dtype=torch.complex128)
    for a, b in zip(candidate, reference, strict=True):
        overlap += torch.vdot(b.to(torch.complex128).reshape(-1), a.to(torch.complex128).reshape(-1)).cpu()
    phase = torch.exp(-1j * torch.angle(overlap))
    errors = {}
    for index, (a, b) in enumerate(zip(candidate, reference, strict=True)):
        errors[str(index)] = _relative_l2(a * phase, b)
    return float(torch.angle(overlap)), errors


def _full_chain_comparison(
    phases: dict[str, torch.Tensor], row: int, label: int, pixels: torch.Tensor,
    order: int, device: torch.device, reference_result: dict[str, Any] | None = None,
    fft_embedding_extra_cells: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target, _ = baseline._image_batch(pixels, torch.tensor([row]), device)
    model = _model_from_phases(phases, order, 4096, device,
                               fft_embedding_extra_cells=fft_embedding_extra_cells)
    model.eval()
    with torch.no_grad():
        result = model.forward_from_image(target, capture_fields=True)
    return {"model": model, "target": target, "result": result, "row": row, "label": label}, {}


def _full_chain_residual(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    a_result, b_result = candidate["result"], reference["result"]
    a_planes, b_planes = a_result["finite_pixel_plane_values"], b_result["finite_pixel_plane_values"]
    _require(len(a_planes) == len(b_planes) == 7, "finite-pixel full chain did not record the seven unique planes")
    a_fields: list[torch.Tensor] = []
    b_fields: list[torch.Tensor] = []
    names: list[str] = []
    for index, (a_plane, b_plane) in enumerate(zip(a_planes, b_planes, strict=True)):
        for view in ("pixel_center", "pixel_area_average"):
            names.append(f"plane_{index + 1}_{view}")
            a_fields.append(a_plane[view].detach().cpu())
            b_fields.append(b_plane[view].detach().cpu())
    # One scalar phase is selected jointly across every observed plane and view.
    overlap = sum(torch.vdot(b.to(torch.complex128).reshape(-1), a.to(torch.complex128).reshape(-1))
                  for a, b in zip(a_fields, b_fields, strict=True))
    phase = torch.exp(-1j * torch.angle(overlap))
    field_errors = {name: float(torch.linalg.vector_norm((a.to(torch.complex128) * phase) - b.to(torch.complex128)) /
                                 torch.linalg.vector_norm(b.to(torch.complex128)).clamp_min(1.0e-300))
                    for name, a, b in zip(names, a_fields, b_fields, strict=True)}
    z_a = a_result["ols_coefficients"].detach().cpu()
    z_b = b_result["ols_coefficients"].detach().cpu()
    ols_error = float(torch.linalg.vector_norm(z_a.to(torch.complex128) * phase - z_b.to(torch.complex128)) /
                      torch.linalg.vector_norm(z_b.to(torch.complex128)).clamp_min(1.0e-300))
    per_port_power_error = float((z_a.abs().square() - z_b.abs().square()).abs().max())
    camera_error = _relative_l2(a_result["camera_roi_w_per_pixel"].detach().cpu(),
                                b_result["camera_roi_w_per_pixel"].detach().cpu())
    return {"global_phase_alignment_rad": float(torch.angle(overlap)),
            "per_plane_field_relative_l2": field_errors,
            "max_plane_field_relative_l2": max(field_errors.values()),
            "ols_complex_relative_l2": ols_error,
            "max_port_power_absolute_error_w": per_port_power_error,
            "camera_roi_w_per_pixel_relative_l2": camera_error,
            "camera_roi_w_per_pixel_gate_passed": camera_error <= WINDOW_TOLERANCE}


def _record_full_chain_residual(
    measurements: dict[str, float], required: set[str], prefix: str, residual: dict[str, Any],
) -> None:
    for plane_name, value in residual["per_plane_field_relative_l2"].items():
        key = f"{prefix}:{plane_name}"
        measurements[key] = value
        required.add(key)
    for name in ("ols_complex_relative_l2", "camera_roi_w_per_pixel_relative_l2"):
        key = f"{prefix}:{name}"
        measurements[key] = residual[name]
        required.add(key)


def _raw_observation_path(outdir: Path, stage: str, state: str, row: int, order: int,
                         embedding_extra_cells: int = 0) -> Path:
    return outdir / "qualification_raw" / (
        f"{stage}_{state}_row{row}_q{order}_pad{embedding_extra_cells}.pt"
    )


def _save_full_chain_raw(
    outdir: Path, stage: str, state: str, comparison: dict[str, Any], order: int,
    embedding_extra_cells: int = 0,
) -> Path:
    result = comparison["result"]
    plane_values = tuple({name: value.detach().cpu().clone() for name, value in plane.items()}
                         for plane in result["finite_pixel_plane_values"])
    saved_result = {
        "finite_pixel_plane_values": plane_values,
        "ols_coefficients": result["ols_coefficients"].detach().cpu().clone(),
        "camera_roi_w_per_pixel": result["camera_roi_w_per_pixel"].detach().cpu().clone(),
        "physical_w_per_pixel": result["physical_w_per_pixel"].detach().cpu().clone(),
        "plane_powers_w": result["plane_powers_w"].detach().cpu().clone(),
        "input_power_w": result["input_power_w"].detach().cpu().clone(),
    }
    path = _raw_observation_path(outdir, stage, state, int(comparison["row"]), order,
                                 embedding_extra_cells)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save_new(path, {
        "schema": "chen_lightgen_400_rs_qualification_raw.v1",
        "stage": stage, "state": state, "row": int(comparison["row"]),
        "label": int(comparison["label"]), "order": order, "window_cells": 4096,
        "common_roi_pixels": COMMON_ROI_PIXELS,
        "embedding_extra_cells": embedding_extra_cells,
        "propagation_backend": PROPAGATION_BACKEND,
        "physical_units": {"fields": "sqrt(W)/m", "coefficients": "sqrt(W)",
                            "camera": "W/pixel", "plane_powers": "W"},
        "data_role": "canonical_MNIST_TRAIN_fixed_confirmation_row",
        "result": saved_result,
    })
    return path


def _load_full_chain_raw(
    outdir: Path, stage: str, state: str, row: int, label: int, order: int,
    embedding_extra_cells: int = 0,
) -> dict[str, Any]:
    path = _raw_observation_path(outdir, stage, state, row, order, embedding_extra_cells)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {"schema": "chen_lightgen_400_rs_qualification_raw.v1", "stage": stage,
                "state": state, "row": row, "label": label, "order": order,
                "window_cells": 4096, "common_roi_pixels": COMMON_ROI_PIXELS,
                "embedding_extra_cells": embedding_extra_cells,
                "propagation_backend": PROPAGATION_BACKEND,
                "data_role": "canonical_MNIST_TRAIN_fixed_confirmation_row"}
    _require(all(payload.get(key) == value for key, value in expected.items()),
             f"saved numerical observation metadata differs: {path}")
    result = payload.get("result")
    _require(isinstance(result, dict)
             and len(result.get("finite_pixel_plane_values", ())) == 7
             and tuple(result.get("ols_coefficients", torch.empty(0)).shape) == (1, 100)
             and tuple(result.get("camera_roi_w_per_pixel", torch.empty(0)).shape) == (1, 1, 500, 500),
             f"saved numerical observation payload is incomplete: {path}")
    tensors = [result["ols_coefficients"], result["camera_roi_w_per_pixel"],
               result["physical_w_per_pixel"], result["plane_powers_w"], result["input_power_w"]]
    tensors.extend(value for plane in result["finite_pixel_plane_values"] for value in plane.values())
    _require(all(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all())
                 for value in tensors), f"saved numerical observation contains invalid tensors: {path}")
    return {"result": result, "state": state, "row": row, "label": label,
            "raw_path": str(path)}


def _import_qualification_cache(
    source_dir: Path | None, outdir: Path, state_phases: dict[str, dict[str, torch.Tensor]],
    population: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Import only explicitly requested raw observations after validating their identities."""
    if source_dir is None:
        return {"source_dir": None, "files": []}
    source_dir = _resolve(source_dir)
    _require(source_dir.is_dir(), f"explicit qualification cache directory is missing: {source_dir}")
    labels = (0, 2)
    confirm_rows = population["confirm_rows_by_class"][:, 0].long()
    requests = [
        ("reference", state_name, int(confirm_rows[label]), label, order)
        for state_name in state_phases for label in labels for order in (12, 16)
    ]
    requests.extend(
        ("candidate", state_name, int(row), label, 8)
        for state_name in state_phases for label, row in enumerate(confirm_rows)
    )
    imported: list[dict[str, Any]] = []
    for stage, state, row, label, order in requests:
        source = _raw_observation_path(source_dir, stage, state, row, order)
        if not source.is_file():
            continue
        _load_full_chain_raw(source_dir, stage, state, row, label, order)
        destination = _raw_observation_path(outdir, stage, state, row, order)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _require(not destination.exists(), f"qualification cache import would overwrite: {destination}")
        shutil.copy2(source, destination)
        _load_full_chain_raw(outdir, stage, state, row, label, order)
        imported.append({"stage": stage, "state": state, "row": row, "label": label,
                         "order": order, "source": str(source),
                         "destination": str(destination), "bytes": destination.stat().st_size})
    result = {"source_dir": str(source_dir), "files": imported,
              "imported_count": len(imported)}
    atomic_write_new_json(outdir / "qualification_cache_import.json", baseline._json_safe(result))
    return result


def _reference_convergence_qualification(
    state_phases: dict[str, dict[str, torch.Tensor]], population: dict[str, torch.Tensor],
    pixels: torch.Tensor, device: torch.device, outdir: Path,
) -> dict[str, Any]:
    """Establish q12 as a reference on fixed worst rows, reusing valid saved observations."""
    rows = population["confirm_rows_by_class"][[0, 2], 0].long()
    records: list[dict[str, Any]] = []
    measurements: dict[str, float] = {}
    required: set[str] = set()
    started = time.perf_counter()
    expected = sum(
        not _raw_observation_path(outdir, "reference", state_name, int(row), order).is_file()
        for state_name in state_phases for row in rows for order in (12, 16)
    )
    completed = 0
    for state_name, phases in state_phases.items():
        for label, row_tensor in zip((0, 2), rows, strict=True):
            row = int(row_tensor)
            observations: dict[int, dict[str, Any]] = {}
            paths: dict[int, str] = {}
            for order in (12, 16):
                raw_path = _raw_observation_path(outdir, "reference", state_name, row, order)
                if raw_path.is_file():
                    observations[order] = _load_full_chain_raw(
                        outdir, "reference", state_name, row, label, order
                    )
                    paths[order] = str(raw_path)
                    continue
                comparison = _full_chain_comparison(phases, row, label, pixels, order, device)[0]
                paths[order] = str(_save_full_chain_raw(
                    outdir, "reference", state_name, comparison, order
                ))
                observations[order] = comparison
                completed += 1
                atomic_write_json(outdir / "progress.json", {
                    "stage": "full_chain_reference_convergence", "completed_forwards": completed,
                    "expected_forwards": expected, "state": state_name, "row": row,
                    "order": order, "elapsed_seconds": time.perf_counter() - started,
                    "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                })
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
            q12, q16 = observations[12], observations[16]
            residual = _full_chain_residual(q12, q16)
            prefix = f"{state_name}:{row}:q12_vs_q16"
            _record_full_chain_residual(measurements, required, prefix, residual)
            records.append({"state": state_name, "label": label, "row": row, "errors": residual,
                            "q12_raw_path": paths[12], "q16_raw_path": paths[16]})
            atomic_write_json(outdir / "progress.json", {
                "stage": "full_chain_reference_convergence", "completed_forwards": completed,
                "expected_forwards": expected, "state": state_name, "row": row,
                "elapsed_seconds": time.perf_counter() - started,
                "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            })
            del observations, q12, q16
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    status = _numerical_qualification_status(measurements, required, WINDOW_TOLERANCE)
    return {"status": status, "candidate_order": 12, "reference_order": 16,
            "labels": [0, 2], "rows": rows.tolist(), "records": records,
            "measurements": measurements, "required_measurements": sorted(required),
            "tolerance": WINDOW_TOLERANCE, "elapsed_seconds": time.perf_counter() - started}


def _gradient_for_row(
    phases: dict[str, torch.Tensor], row: int, pixels: torch.Tensor, order: int,
    beta: float, device: torch.device,
) -> tuple[list[torch.Tensor], float]:
    target, _ = baseline._image_batch(pixels, torch.tensor([row]), device)
    model = _model_from_phases(phases, order, 4096, device)
    model.requires_grad_(False)
    model.encoder_phase.requires_grad_(True)
    model.decoder_phase.requires_grad_(True)
    result = model.forward_from_image(target)
    loss = beta * (FIXED_GAIN * result["physical_w_per_pixel"] - target).square().mean()
    gradients = torch.autograd.grad(loss, (model.encoder_phase, model.decoder_phase))
    flat = [layer.detach().cpu().double().clone()
            for group in gradients for layer in group[:, 0]]
    norms = [float(torch.linalg.vector_norm(value)) for value in flat]
    _require(len(flat) == 6 and all(math.isfinite(value) and value > 0.0 for value in norms),
             "a finite-pixel phase plane has a zero or nonfinite reconstruction gradient")
    del model, result, loss, gradients
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    return flat, float(torch.linalg.vector_norm(torch.stack([torch.tensor(v) for v in norms])))


def _finite_difference_direction(
    phases: dict[str, torch.Tensor], row: int, pixels: torch.Tensor, order: int,
    beta: float, layer_directions: list[torch.Tensor], autodiff_gradients: list[torch.Tensor],
    device: torch.device,
) -> dict[str, Any]:
    target, _ = baseline._image_batch(pixels, torch.tensor([row]), device)
    checks: list[dict[str, Any]] = []
    layer_index = 0
    for group_name, group_layers in (("encoder_phase", 2), ("decoder_phase", 4)):
        for index in range(group_layers):
            direction = layer_directions[layer_index].to(device=device, dtype=torch.float32)
            norm = torch.linalg.vector_norm(direction)
            _require(float(norm) > 0.0 and torch.isfinite(norm), "finite-difference direction is invalid")
            direction = direction / norm
            base_phase = phases[group_name].to(device).clone()
            local_index = index
            autodiff = float((autodiff_gradients[layer_index].to(device=device, dtype=torch.float32)
                              * direction).sum().detach().cpu())
            derivatives = []
            for epsilon in FD_STEPS_RAD:
                losses = []
                for sign in (1.0, -1.0):
                    model = _model_from_phases(phases, order, 4096, device)
                    parameter = getattr(model, group_name)
                    with torch.no_grad():
                        parameter[local_index, 0].copy_(base_phase[local_index, 0] + sign * epsilon * direction)
                    with torch.no_grad():
                        result = model.forward_from_image(target)
                        value = beta * (FIXED_GAIN * result["physical_w_per_pixel"] - target).square().mean()
                    losses.append(float(value.cpu()))
                    del model, result
                derivative = (losses[0] - losses[1]) / (2.0 * epsilon)
                relative = abs(derivative - autodiff) / max(abs(derivative), abs(autodiff), 1.0e-14)
                derivatives.append({"epsilon_rad": epsilon, "central_difference": derivative,
                                    "autodiff_directional_derivative": autodiff,
                                    "relative_error": relative,
                                    "same_sign": derivative * autodiff > 0.0})
            checks.append({"group": group_name, "layer": index, "direction_norm": float(norm),
                           "autodiff_directional_derivative": autodiff,
                           "degenerate": abs(autodiff) <= 1.0e-14,
                           "central_differences": derivatives})
            layer_index += 1
    return {"checks": checks, "steps_rad": list(FD_STEPS_RAD)}


def _full_chain_qualification(
    state_phases: dict[str, dict[str, torch.Tensor]], population: dict[str, torch.Tensor],
    labels: torch.Tensor, pixels: torch.Tensor, candidate_orders: list[int], device: torch.device,
    outdir: Path,
) -> dict[str, Any]:
    rows = population["confirm_rows_by_class"][:, 0].long()
    records: dict[str, list[dict[str, Any]]] = {str(order): [] for order in candidate_orders}
    measurements: dict[str, dict[str, float]] = {str(order): {} for order in candidate_orders}
    required: dict[str, set[str]] = {str(order): set() for order in candidate_orders}
    started = time.perf_counter()
    total = 0
    total_expected = sum(
        not _raw_observation_path(outdir, "reference", state_name, int(row), 12).is_file()
        for state_name in state_phases for row in rows
    ) + sum(
        not _raw_observation_path(outdir, "candidate", state_name, int(row), order).is_file()
        for state_name in state_phases for row in rows for order in candidate_orders
    )
    for state_name, phases in state_phases.items():
        for label, row_tensor in enumerate(rows):
            row = int(row_tensor)
            ref_path = _raw_observation_path(outdir, "reference", state_name, row, 12)
            if ref_path.is_file():
                ref = _load_full_chain_raw(outdir, "reference", state_name, row, label, 12)
            else:
                reference_comparison = _full_chain_comparison(
                    phases, row, label, pixels, 12, device
                )[0]
                saved_ref_path = _save_full_chain_raw(
                    outdir, "reference", state_name, reference_comparison, 12
                )
                ref = {"result": reference_comparison["result"], "state": state_name,
                       "row": row, "label": label, "raw_path": str(saved_ref_path)}
                total += 1
                atomic_write_json(outdir / "progress.json", {
                    "stage": "full_chain_comparison", "completed_forwards": total,
                    "expected_forwards": total_expected, "state": state_name,
                    "row": row, "order": 12, "elapsed_seconds": time.perf_counter() - started,
                    "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                })
                del reference_comparison
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
            for order in candidate_orders:
                candidate_path = _raw_observation_path(outdir, "candidate", state_name, row, order)
                if candidate_path.is_file():
                    candidate = _load_full_chain_raw(
                        outdir, "candidate", state_name, row, label, order
                    )
                else:
                    comparison = _full_chain_comparison(
                        phases, row, label, pixels, order, device
                    )[0]
                    saved_candidate_path = _save_full_chain_raw(
                        outdir, "candidate", state_name, comparison, order
                    )
                    candidate = {"result": comparison["result"], "state": state_name,
                                 "row": row, "label": label,
                                 "raw_path": str(saved_candidate_path)}
                    total += 1
                    atomic_write_json(outdir / "progress.json", {
                        "stage": "full_chain_comparison", "completed_forwards": total,
                        "expected_forwards": total_expected, "state": state_name,
                        "row": row, "order": order,
                        "elapsed_seconds": time.perf_counter() - started,
                        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
                    })
                    del comparison
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        torch.cuda.empty_cache()
                residual = _full_chain_residual(candidate, ref)
                record = {"state": state_name, "label": label, "row": row, "order": order,
                          "candidate_raw_path": candidate["raw_path"],
                          "reference_raw_path": ref["raw_path"], **residual}
                records[str(order)].append(record)
                _record_full_chain_residual(
                    measurements[str(order)], required[str(order)], f"{state_name}:{row}", residual
                )
                del candidate
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.empty_cache()
            del ref
            if device.type == "cuda":
                torch.cuda.empty_cache()
    statuses = {
        order: _numerical_qualification_status(measurements[order], required[order], WINDOW_TOLERANCE)
        for order in measurements
    }
    return {"samples_per_state": rows.tolist(), "states": list(state_phases),
            "forwards_completed": total, "forwards_expected": total_expected,
            "elapsed_seconds": time.perf_counter() - started,
            "orders": {order: {"status": statuses[order], "records": records[order],
                               "measurements": measurements[order],
                               "max_field_relative_l2": max(
                                   (record["max_plane_field_relative_l2"] for record in records[order]),
                                   default=math.inf),
                               "max_ols_relative_l2": max(
                                   (record["ols_complex_relative_l2"] for record in records[order]),
                                   default=math.inf)}
                       for order in records}}


def _fft_embedding_qualification(
    state_phases: dict[str, dict[str, torch.Tensor]], population: dict[str, torch.Tensor],
    pixels: torch.Tensor, order: int, device: torch.device, outdir: Path,
) -> dict[str, Any]:
    """Change the actual RS FFT embedding while holding source and observer ROI fixed."""
    labels = (0, 2)
    rows = population["confirm_rows_by_class"][list(labels), 0].long()
    extra_cells = 64
    records: list[dict[str, Any]] = []
    measurements: dict[str, float] = {}
    required: set[str] = set()
    started = time.perf_counter()
    for state_name, phases in state_phases.items():
        for label, row_tensor in zip(labels, rows, strict=True):
            row = int(row_tensor)
            base = _load_full_chain_raw(outdir, "candidate", state_name, row, label, order)
            padded = _full_chain_comparison(
                phases, row, label, pixels, order, device,
                fft_embedding_extra_cells=extra_cells,
            )[0]
            padded_path = _save_full_chain_raw(outdir, "embedding", state_name,
                                               padded, order, extra_cells)
            residual = _full_chain_residual(base, padded)
            prefix = f"{state_name}:{row}:embedding_extra_{extra_cells}"
            _record_full_chain_residual(measurements, required, prefix, residual)
            records.append({"state": state_name, "label": label, "row": row,
                            "base_raw_path": base["raw_path"],
                            "padded_raw_path": str(padded_path), "errors": residual})
            atomic_write_json(outdir / "progress.json", {
                "stage": "actual_fft_embedding_check", "state": state_name,
                "row": row, "order": order, "extra_embedding_cells": extra_cells,
                "completed_checks": len(records), "expected_checks": len(state_phases) * len(rows),
                "elapsed_seconds": time.perf_counter() - started,
            })
            del base, padded
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    status = _numerical_qualification_status(measurements, required, WINDOW_TOLERANCE)
    return {"status": status, "candidate_order": order, "reference_order": order,
            "physical_window_cells": 4096, "common_roi_pixels": COMMON_ROI_PIXELS,
            "fft_embedding_extra_cells": extra_cells, "labels": list(labels),
            "rows": rows.tolist(), "records": records, "measurements": measurements,
            "required_measurements": sorted(required), "tolerance": WINDOW_TOLERANCE,
            "elapsed_seconds": time.perf_counter() - started}


def _gradient_qualification(
    state_phases: dict[str, dict[str, torch.Tensor]], population: dict[str, torch.Tensor],
    pixels: torch.Tensor, order: int, beta: float, device: torch.device, outdir: Path,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    measurements: dict[str, float] = {}
    required: set[str] = set()
    derivative_checks: list[dict[str, Any]] = []
    rows = [int(population["confirm_rows_by_class"][label, 0]) for label in (0, 1)]
    for state_name, phases in state_phases.items():
        for row in rows:
            candidate_gradient, _ = _gradient_for_row(phases, row, pixels, order, beta, device)
            reference_gradient, _ = _gradient_for_row(phases, row, pixels, 12, beta, device)
            layer_errors = []
            for layer, (candidate_layer, reference_layer) in enumerate(
                zip(candidate_gradient, reference_gradient, strict=True)
            ):
                relative = float(torch.linalg.vector_norm(candidate_layer - reference_layer) /
                                 torch.linalg.vector_norm(reference_layer).clamp_min(1.0e-30))
                key = f"{state_name}:{row}:layer_{layer + 1}"
                measurements[key] = relative
                required.add(key)
                layer_errors.append(relative)
            candidate_fd = _finite_difference_direction(
                phases, row, pixels, order, beta, candidate_gradient, candidate_gradient, device
            )
            reference_fd = _finite_difference_direction(
                phases, row, pixels, 12, beta, candidate_gradient, reference_gradient, device
            )
            for path_name, fd in (("candidate", candidate_fd), ("reference", reference_fd)):
                for check in fd["checks"]:
                    for entry in check["central_differences"]:
                        key = f"{state_name}:{row}:{path_name}:{check['group']}:{check['layer']}:{entry['epsilon_rad']}"
                        required.add(key)
                        measurements[key] = entry["relative_error"] if entry["same_sign"] else math.inf
            derivative_checks.append({"state": state_name, "row": row,
                                      "candidate": candidate_fd, "reference": reference_fd})
            records.append({"state": state_name, "row": row,
                            "per_layer_gradient_relative_l2": layer_errors,
                            "max_gradient_relative_l2": max(layer_errors)})
            atomic_write_json(outdir / "progress.json", {
                "stage": "gradient_and_finite_difference", "state": state_name,
                "row": row, "completed_rows": len(records), "expected_rows": len(state_phases) * 2,
            })
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    status = _numerical_qualification_status(measurements, required, max(WINDOW_TOLERANCE, FD_TOLERANCE))
    # Gradient agreement has a 1e-4 limit; directional finite differences have a 1% limit.
    gradient_values = [value for key, value in measurements.items() if ":layer_" in key]
    finite_difference_values = [value for key, value in measurements.items() if ":layer_" not in key]
    status = "NUMERICALLY_QUALIFIED" if (
        len(gradient_values) == len(state_phases) * 2 * 6
        and len(finite_difference_values) == len(state_phases) * 2 * 6 * 2 * len(FD_STEPS_RAD)
        and all(math.isfinite(value) and value <= WINDOW_TOLERANCE for value in gradient_values)
        and all(math.isfinite(value) and value <= FD_TOLERANCE for value in finite_difference_values)
    ) else "NUMERICAL_NOT_QUALIFIED" if required.issubset(measurements) else "INVALID_HARNESS"
    return {"status": status, "candidate_order": order, "reference_order": 12,
            "records": records, "measurements": measurements,
            "required_measurements": sorted(required), "finite_difference_checks": derivative_checks,
            "gradient_tolerance": WINDOW_TOLERANCE, "finite_difference_tolerance": FD_TOLERANCE,
            "finite_difference_steps_rad": list(FD_STEPS_RAD)}


def _loss_parts(result: dict[str, Any], target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    task = FIXED_GAIN * result["physical_w_per_pixel"]
    mse = (task - target).square().mean()
    prediction_28 = F.adaptive_avg_pool2d(task, (TASK_SIZE, TASK_SIZE))
    target_28 = F.adaptive_avg_pool2d(target, (TASK_SIZE, TASK_SIZE))
    structure = 1.0 - baseline._ssim_batch(prediction_28, target_28).mean()
    return mse, structure


def _trainable_parameters(model: FinitePixelChenLightGen400) -> tuple[torch.Tensor, ...]:
    return model.encoder_phase, model.decoder_phase


def _optimizer(model: FinitePixelChenLightGen400) -> torch.optim.Adam:
    return torch.optim.Adam(_trainable_parameters(model), lr=1.0e-3, betas=(0.9, 0.999),
                            eps=1.0e-8, weight_decay=0.0)


def _set_trainable(model: FinitePixelChenLightGen400) -> None:
    model.requires_grad_(False)
    model.encoder_phase.requires_grad_(True)
    model.decoder_phase.requires_grad_(True)


def _cpu_state(value: Any) -> Any:
    return baseline._cpu_snapshot(value)


def _set_rng(cpu_state: torch.Tensor, cuda_states: list[torch.Tensor]) -> None:
    torch.random.set_rng_state(cpu_state.cpu())
    torch.cuda.set_rng_state_all([value.cpu() for value in cuda_states])


def _batch_gradients(
    model: FinitePixelChenLightGen400, pixels: torch.Tensor, rows: torch.Tensor,
    branch: str, beta: float, eta: float, microbatch: int, device: torch.device,
) -> tuple[float, dict[str, torch.Tensor]]:
    _require(microbatch in (1, 2), "preflight microbatch must be 1 or 2")
    _require(rows.numel() == LOGICAL_BATCH, "gradient probe requires the fixed logical batch of ten")
    model.zero_grad(set_to_none=True)
    weighted_loss = 0.0
    for start in range(0, LOGICAL_BATCH, microbatch):
        part = rows[start:start + microbatch]
        target, _ = baseline._image_batch(pixels, part, device)
        result = model.forward_from_image(target)
        mse, structure = _loss_parts(result, target)
        loss = beta * mse + (eta * structure if branch == "B" else 0.0)
        (loss * (part.numel() / LOGICAL_BATCH)).backward()
        weighted_loss += float(loss.detach().cpu()) * part.numel() / LOGICAL_BATCH
        del target, result, mse, structure, loss
    gradients = {name: parameter.grad.detach().cpu().clone()
                 for name, parameter in zip(("encoder_phase", "decoder_phase"),
                                            _trainable_parameters(model), strict=True)}
    return weighted_loss, gradients


def _calibrate_eta(
    model: FinitePixelChenLightGen400, pixels: torch.Tensor, rows: torch.Tensor,
    beta: float, device: torch.device,
) -> dict[str, float]:
    _require(rows.numel() == LOGICAL_BATCH, "structure balance batch must contain ten fit rows")
    mse_gradients = [torch.zeros_like(parameter, dtype=torch.float64, device="cpu")
                     for parameter in _trainable_parameters(model)]
    structure_gradients = [torch.zeros_like(value) for value in mse_gradients]
    for row_tensor in rows:
        target, _ = baseline._image_batch(pixels, row_tensor.reshape(1), device)
        result = model.forward_from_image(target)
        mse, structure = _loss_parts(result, target)
        grad_mse = torch.autograd.grad(beta * mse, _trainable_parameters(model))
        grad_structure = torch.autograd.grad(structure, _trainable_parameters(model))
        for index in range(2):
            mse_gradients[index] += grad_mse[index].detach().double().cpu() / LOGICAL_BATCH
            structure_gradients[index] += grad_structure[index].detach().double().cpu() / LOGICAL_BATCH
        del target, result, mse, structure, grad_mse, grad_structure
    norm_mse = math.sqrt(sum(float(value.square().sum()) for value in mse_gradients))
    norm_structure = math.sqrt(sum(float(value.square().sum()) for value in structure_gradients))
    _require(math.isfinite(norm_mse) and norm_mse > 0.0
             and math.isfinite(norm_structure) and norm_structure > 0.0,
             "MSE or structure term has a zero/nonfinite six-layer gradient")
    eta = norm_mse / norm_structure
    _require(math.isfinite(eta) and eta > 0.0, "one-time structural coefficient is invalid")
    return {"eta": eta, "beta_mse_gradient_l2": norm_mse,
            "structure_gradient_l2": norm_structure,
            "first_fit_batch_rows": rows.tolist(),
            "estimator": "single_initial_fit_batch_equal_six_layer_gradient_l2"}


def _train_update(
    model: FinitePixelChenLightGen400, optimizer: torch.optim.Optimizer, pixels: torch.Tensor,
    rows: torch.Tensor, branch: str, beta: float, eta: float, microbatch: int,
    device: torch.device,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss, gradients = _batch_gradients(model, pixels, rows, branch, beta, eta, microbatch, device)
    for name, parameter in zip(("encoder_phase", "decoder_phase"), _trainable_parameters(model), strict=True):
        gradient = parameter.grad
        _require(gradient is not None and torch.isfinite(gradient).all().item(),
                 f"{branch} {name} gradient is missing or nonfinite")
        layer_norms = [float(torch.linalg.vector_norm(layer.detach().double()).cpu())
                       for layer in gradient[:, 0]]
        _require(all(math.isfinite(value) and value > 0.0 for value in layer_norms),
                 f"{branch} {name} contains a disconnected phase plane")
    before = {name: getattr(model, name).detach().clone()
              for name in ("encoder_phase", "decoder_phase")}
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _require(all(torch.isfinite(getattr(model, name)).all().item()
                 and not torch.equal(before[name], getattr(model, name)) for name in before),
             f"{branch} failed to update all six phase planes")
    max_updates = {name: float((getattr(model, name) - before[name]).abs().max().cpu()) for name in before}
    return {"branch": branch, "weighted_loss": loss,
            "gradient_l2": {key: float(torch.linalg.vector_norm(value.double()))
                            for key, value in gradients.items()},
            "phase_max_update": max_updates,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_reserved_vram_bytes": (torch.cuda.max_memory_reserved(device)
                                         if device.type == "cuda" else None)}


def _worker_update(payload_path: Path, output_path: Path, ready_path: Path,
                   requested_signal: int | None) -> None:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    if requested_signal is None:
        requested_signal = payload.get("requested_signal")
    _require(payload.get("propagation_backend") == PROPAGATION_BACKEND,
             "recovery worker propagation backend differs from the authorized RS path")
    device = torch.device("cuda")
    model = FinitePixelChenLightGen400(FinitePixelConfig(
        quadrature_order=int(payload["order"]), window_cells=int(payload["window"]),
        distance_m=3.0e-3, checkpoint_propagation=True,
        propagation_backend=str(payload["propagation_backend"]),
    )).to(device)
    _set_trainable(model)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = _optimizer(model)
    optimizer.load_state_dict(payload["optimizer"])
    _set_rng(payload["rng_cpu"], payload["rng_cuda"])
    signal_received: list[int] = []
    old_handlers: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        signal_received.append(signum)

    if requested_signal is not None:
        for signum in (signal.SIGINT, signal.SIGTERM):
            old_handlers[signum] = signal.signal(signum, handler)
        atomic_write_new_json(ready_path, {"ready": True, "completed_updates": payload["completed_updates"]})
        while not signal_received:
            time.sleep(0.05)
    pixels, _, _, _ = baseline._load_data(Path(payload["data_path"]))
    rows_by_update = payload["rows_by_update"].long()
    latest_updates = []
    for rows in rows_by_update:
        latest_updates.append(_train_update(model, optimizer, pixels, rows, payload["branch"],
                                            float(payload["beta"]), float(payload["eta"]),
                                            int(payload["microbatch"]), device))
    for signum, old_handler in old_handlers.items():
        signal.signal(signum, old_handler)
    state = {
        "model": _cpu_state(model.state_dict()), "optimizer": _cpu_state(optimizer.state_dict()),
        "rng_cpu": torch.random.get_rng_state().cpu(),
        "rng_cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()],
        "completed_updates": int(payload["completed_updates"]) + len(latest_updates),
        "next_batch_cursor": (int(payload["completed_updates"]) + len(latest_updates)) % 10,
        "requested_signal": requested_signal, "received_signals": signal_received,
        "latest_updates": latest_updates,
    }
    atomic_torch_save_new(output_path, state)


def _worker_probe(
    model: FinitePixelChenLightGen400, optimizer: torch.optim.Optimizer, rows_by_update: torch.Tensor,
    data_path: Path, branch: str, beta: float, eta: float, completed: int, order: int,
    window: int, outdir: Path, label: str, requested_signal: int | None, microbatch: int = 1,
) -> dict[str, Any]:
    stem = f"worker_{branch}_{label}"
    payload_path, output_path, ready_path = (outdir / f"{stem}_{suffix}.pt"
                                             for suffix in ("input", "output", "ready"))
    payload = {"schema": "chen_lightgen_400_finite_pixel_worker.v1", "branch": branch,
               "completed_updates": completed, "next_batch_cursor": completed % 10,
               "order": order, "window": window, "model": _cpu_state(model.state_dict()),
               "optimizer": _cpu_state(optimizer.state_dict()),
               "rng_cpu": torch.random.get_rng_state().cpu(),
               "rng_cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()],
               "rows_by_update": rows_by_update.cpu(), "data_path": str(data_path), "beta": beta, "eta": eta,
               "requested_signal": requested_signal, "propagation_backend": PROPAGATION_BACKEND,
               "microbatch": microbatch}
    atomic_torch_save_new(payload_path, payload)
    command = [sys.executable, "-m",
               "scripts.research.lightgen.execute_chen_lightgen_400_finite_pixel_reconstruction_ab_v1",
               "--worker-payload", str(payload_path),
               "--worker-output", str(output_path), "--worker-ready", str(ready_path),
               "--propagation-backend", PROPAGATION_BACKEND]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if requested_signal is not None:
        while not ready_path.is_file():
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise RuntimeError(f"{label} recovery worker exited before ready: {stdout[-1000:]} {stderr[-2500:]}")
            time.sleep(0.1)
        process.send_signal(requested_signal)
    stdout, stderr = process.communicate()
    _require(process.returncode == 0,
             f"{label} independent worker failed: {stdout[-1000:]} {stderr[-2500:]}")
    result = torch.load(output_path, map_location="cpu", weights_only=False)
    expected_completed = completed + rows_by_update.shape[0]
    _require(result["completed_updates"] == expected_completed
             and result["next_batch_cursor"] == expected_completed % 10,
             f"{label} worker skipped or duplicated a logical update")
    if requested_signal is not None:
        _require(result["requested_signal"] == requested_signal
                 and result["received_signals"] == [requested_signal],
                 f"{label} worker did not catch the requested signal")
    return result


def _branch_preflight(
    phases: dict[str, torch.Tensor], population: dict[str, torch.Tensor], pixels: torch.Tensor,
    data_path: Path, branch: str, order: int, beta: float, eta: float, device: torch.device,
    outdir: Path, endpoint_path: Path | None = None, microbatch: int = 1,
) -> dict[str, Any]:
    _require(microbatch in (1, 2), "branch preflight must use a qualified microbatch of one or two")
    model = _model_from_phases(phases, order, 4096, device)
    _set_trainable(model)
    optimizer = _optimizer(model)
    first_rows = population["fit_rows_by_class"][:, 0].long()
    before = _cpu_state(model.state_dict())
    loss_1, grad_1 = _batch_gradients(model, pixels, first_rows, branch, beta, eta, 1, device)
    model.load_state_dict(before, strict=True)
    loss_2, grad_2 = _batch_gradients(model, pixels, first_rows, branch, beta, eta, 2, device)
    loss_error = abs(loss_1 - loss_2) / max(abs(loss_1), 1.0e-30)
    grad_errors = {name: float(torch.linalg.vector_norm(grad_1[name].double() - grad_2[name].double()) /
                               torch.linalg.vector_norm(grad_1[name].double()).clamp_min(1.0e-30))
                   for name in grad_1}
    _require(loss_error <= 1.0e-5 and max(grad_errors.values()) <= 1.0e-4,
             f"{branch} microbatch 1/2 logical batch is not equivalent")
    model.load_state_dict(before, strict=True)
    optimizer = _optimizer(model)
    initial_rng_cpu = torch.random.get_rng_state().cpu()
    initial_rng_cuda = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    updates: list[dict[str, Any]] = []
    for cursor in range(3):
        rows = population["fit_rows_by_class"][:, cursor].long()
        updates.append(_train_update(model, optimizer, pixels, rows, branch, beta, eta, microbatch, device))
    expected_model = _cpu_state(model.state_dict())
    expected_optimizer = _cpu_state(optimizer.state_dict())
    expected_rng_cpu = torch.random.get_rng_state().cpu()
    expected_rng_cuda = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    if endpoint_path is not None:
        endpoint_payload = {
            "schema": "chen_lightgen_400_finite_pixel_engineering_endpoint.v1",
            "branch": branch, "quadrature_order": order, "window_cells": 4096,
            "propagation_backend": PROPAGATION_BACKEND,
            "completed_updates": 3, "next_batch_cursor": 3,
            "model": expected_model, "optimizer": expected_optimizer,
            "rng_cpu": expected_rng_cpu, "rng_cuda": expected_rng_cuda,
        }
        if endpoint_path.is_file():
            saved_endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
            _require(baseline._assert_state_close(saved_endpoint, endpoint_payload,
                                                  f"{branch}_engineering_endpoint") == 0.0,
                     "existing finite-pixel engineering endpoint differs from the reproduced three-step state")
        else:
            atomic_torch_save_new(endpoint_path, endpoint_payload)
            saved_endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=False)
            _require(baseline._assert_state_close(saved_endpoint, endpoint_payload,
                                                  f"{branch}_engineering_endpoint_readback") == 0.0,
                     "finite-pixel engineering endpoint failed write/readback")
    model.load_state_dict(before, strict=True)
    optimizer = _optimizer(model)
    _set_rng(initial_rng_cpu, initial_rng_cuda)
    first_rows = population["fit_rows_by_class"][:, 0].long()
    first_update = _train_update(model, optimizer, pixels, first_rows, branch, beta, eta, microbatch, device)
    state_after_one = _cpu_state(model.state_dict())
    optimizer_after_one = _cpu_state(optimizer.state_dict())
    rng_cpu_after_one = torch.random.get_rng_state().cpu()
    rng_cuda_after_one = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    recovery_signal = signal.SIGINT if branch == "A" else signal.SIGTERM
    resume_rows = population["fit_rows_by_class"][:, 1:3].long()
    model.load_state_dict(state_after_one, strict=True)
    optimizer.load_state_dict(optimizer_after_one)
    _set_rng(rng_cpu_after_one, rng_cuda_after_one)
    recovery = _worker_probe(model, optimizer, resume_rows, data_path, branch, beta, eta, 1,
                             order, 4096, outdir, f"{branch.lower()}_signal", recovery_signal,
                             microbatch=microbatch)
    model_error = baseline._assert_state_close(expected_model, recovery["model"],
                                               f"{branch}_recovery_model")
    optimizer_error = baseline._assert_state_close(expected_optimizer, recovery["optimizer"],
                                                   f"{branch}_recovery_optimizer")
    rng_equal = (torch.equal(expected_rng_cpu, recovery["rng_cpu"])
                 and all(torch.equal(a, b) for a, b in
                         zip(expected_rng_cuda, recovery["rng_cuda"], strict=True)))
    _require(model_error <= 5.0e-8 and optimizer_error <= 5.0e-8 and rng_equal,
             f"{branch} independent signal recovery differs from the continuous three-update path")
    recovery_record = {"worker": f"{branch.lower()}_signal", "requested_signal": recovery_signal,
                       "model_max_abs_error": model_error,
                       "optimizer_max_abs_error": optimizer_error,
                       "rng_equal": rng_equal,
                       "completed_updates": recovery["completed_updates"],
                       "next_batch_cursor": recovery["next_batch_cursor"],
                       "updates_after_resume": len(recovery["latest_updates"])}
    frozen_buffers = {key: value for key, value in expected_model.items()
                      if key not in ("encoder_phase", "decoder_phase")}
    initial_buffers = {key: value for key, value in before.items()
                       if key not in ("encoder_phase", "decoder_phase")}
    _require(baseline._assert_state_close(initial_buffers, frozen_buffers,
                                          f"{branch}_frozen_buffers") == 0.0,
             f"{branch} preflight changed frozen optical geometry or modes")
    restored_model = _model_from_phases(phases, order, 4096, device)
    restored_initial_model = baseline._assert_state_close(
        before, _cpu_state(restored_model.state_dict()), f"{branch}_restored_initial"
    ) == 0.0
    del restored_model
    _require(sum(update["branch"] == branch for update in updates) == 3,
             f"{branch} continuous path did not execute three engineering updates")
    timings = sorted(update["elapsed_seconds"] for update in
                     (*updates, first_update, *recovery["latest_updates"]))
    median_seconds = timings[len(timings) // 2]
    mean_seconds = sum(timings) / len(timings)
    return {"branch": branch, "three_cuda_updates": 3, "engineering_adam_calls": 6,
            "microbatch_equivalence": {
                "loss_relative_error": loss_error, "gradient_relative_l2_error": grad_errors,
                "tolerance": {"loss": 1.0e-5, "gradient": 1.0e-4}},
            "continuous_updates": updates, "interrupted_first_update": first_update,
            "independent_signal_recovery": recovery_record,
            "logical_batch_10_speed": {"observed_update_seconds": timings,
                                       "median_seconds": median_seconds,
                                       "mean_seconds": mean_seconds,
                                       "estimated_500_updates_seconds_median": median_seconds * 500,
                                       "estimated_30000_updates_seconds_median": median_seconds * 30_000,
                                       "measurement_scope": "six engineering Adam calls per branch; no scientific updates"},
            "frozen_buffers_unchanged": True,
            "restored_initial_model": restored_initial_model,
            "endpoint_path": str(endpoint_path.resolve()) if endpoint_path is not None else None,
            "selected_microbatch": microbatch}


def _production_boundary_checks(order: int, device: torch.device) -> dict[str, Any]:
    model = FinitePixelChenLightGen400(FinitePixelConfig(
        quadrature_order=order, window_cells=4096, distance_m=3.0e-3,
        checkpoint_propagation=True, propagation_backend=PROPAGATION_BACKEND,
    )).to(device)
    diagonal = model.gram.diag()
    gram_max_eigenvalue = float(torch.linalg.eigvalsh(model.gram.cpu())[-1])
    _require(float((diagonal - 1.0).abs().max()) <= 2.0e-6 and gram_max_eigenvalue <= 1.0 + 1.0e-8,
             "production Gaussian OLS mode norm or passivity check failed")
    input_records = {}
    density_scale = math.sqrt(SOURCE_POWER_W / (400 * PIXEL_PITCH_M)**2)
    for value, expected in ((0.0, 0.0), (0.5, SOURCE_POWER_W * 0.5), (1.0, SOURCE_POWER_W)):
        intensity = torch.full((1, 1, 400, 400), value, dtype=torch.float32, device=device)
        field = torch.sqrt(intensity[:, 0]).to(torch.complex64).unsqueeze(1).expand(
            -1, order**2, -1, -1
        ) * density_scale
        measured = float(model._weighted_power(field)[0].detach().cpu())
        relative = abs(measured - expected) / max(expected, SOURCE_POWER_W * 1.0e-12)
        _require(relative <= 2.0e-6, f"production source-plane intensity {value} power integral failed")
        input_records[str(value)] = {"measured_w": measured, "expected_w": expected,
                                    "relative_error": relative}
    mode_y, mode_x = model.modes_y[4], model.modes_x[4]
    matching = torch.einsum("yi,xj->ijyx", mode_y, mode_x).reshape(1, order**2, 400, 400)
    coefficient = model._project_coefficients(matching.to(torch.complex64))
    coefficient_error = float(torch.linalg.vector_norm(
        coefficient.cpu() - torch.nn.functional.one_hot(torch.tensor([44]), 100).to(torch.complex64)
    ))
    _require(coefficient_error <= 1.0e-5, "production Gaussian OLS failed the matched-mode projection")
    test_coefficients = torch.arange(100, dtype=torch.float32, device=device).to(torch.complex64).unsqueeze(0)
    reemitted = model._reemit(test_coefficients)
    measured_reemitted = model._weighted_power(reemitted)
    expected_reemitted = torch.einsum("i,ij,j->", test_coefficients[0].conj(),
                                      model.gram.to(torch.complex64), test_coefficients[0]).real.unsqueeze(0)
    ols_power_error = float(torch.linalg.vector_norm(measured_reemitted - expected_reemitted).cpu()
                            / torch.linalg.vector_norm(expected_reemitted).clamp_min(1.0e-30).cpu())
    _require(ols_power_error <= 2.0e-5, "production OLS output power does not equal its Gram quadratic")
    anti_phase = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.complex64, device=device)
    if order != 2:
        anti_phase = torch.zeros(order**2, dtype=torch.complex64, device=device)
        anti_phase[0], anti_phase[-1] = 1.0, -1.0
    weights = model.node_weight_2d.to(device=device, dtype=torch.float32)
    coherent = (anti_phase * weights).sum().abs().square()
    detector = (anti_phase.abs().square() * weights).sum()
    _require(float(detector) > 0.0 and float(detector) > float(coherent) + 1.0e-6,
             "production CMOS area integration failed to distinguish it from coherent amplitude averaging")
    far_axis_tile = _far_axis_tile_check(order, device)
    result = {"input_power_integrals": input_records,
              "gaussian_mode_max_norm_error": float((diagonal - 1.0).abs().max().cpu()),
              "gram_max_eigenvalue": gram_max_eigenvalue,
              "ols_matched_mode_coefficient_l2_error": coefficient_error,
              "ols_output_power_gram_relative_error": ols_power_error,
              "far_axis_full_window_tile": far_axis_tile,
              "cmos_node_intensity_integral": float(detector.cpu()),
              "cmos_coherent_average_counterexample": float(coherent.cpu()),
              "passed": True}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _far_axis_tile_check(order: int, device: torch.device) -> dict[str, Any]:
    """Check one production-window edge tile against an independent source sum."""
    nodes, weights = _legendre_rule(order, device=device)
    wy, wx = torch.meshgrid(weights, weights, indexing="ij")
    ny, nx = torch.meshgrid(nodes, nodes, indexing="ij")
    source = torch.ones((1, order**2, 400, 400), dtype=torch.complex64, device=device)
    config = FinitePixelConfig(
        quadrature_order=order, window_cells=4096, checkpoint_propagation=False,
        propagation_backend=PROPAGATION_BACKEND, window_power_source_chunk=4,
        window_power_target_chunk=4,
    )
    propagator = FinitePixelRSLinear(order, (400, 400), config).to(device)
    offset = (-2047.5, -2047.5)
    with torch.no_grad():
        measured = propagator.operator(
            source, output_shape=(1, 1), target_indices=(0,),
            output_offset_cells=offset, source_chunk_size=config.window_power_source_chunk,
        )[0, 0, 0, 0].to(torch.complex128)

    coordinates = (torch.arange(400, device=device, dtype=torch.float64) + 0.5 - 200.0) * PIXEL_PITCH_M
    source_y, source_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    observer_y = offset[0] * PIXEL_PITCH_M + nodes[0] * PIXEL_PITCH_M / 2.0
    observer_x = offset[1] * PIXEL_PITCH_M + nodes[0] * PIXEL_PITCH_M / 2.0
    areas = (wy * wx).reshape(-1) * PIXEL_PITCH_M**2
    ny_flat, nx_flat = ny.reshape(-1), nx.reshape(-1)
    reference = torch.zeros((), dtype=torch.complex128, device=device)
    for source_index in range(order**2):
        dy = observer_y - (source_y + ny_flat[source_index] * PIXEL_PITCH_M / 2.0)
        dx = observer_x - (source_x + nx_flat[source_index] * PIXEL_PITCH_M / 2.0)
        radius2 = 3.0e-3**2 + dy.square() + dx.square()
        radius = radius2.sqrt()
        kernel = 3.0e-3 * torch.exp(2j * math.pi * radius / WAVELENGTH_M) / (2 * math.pi * radius**3) \
            * (1 - 2j * math.pi * radius / WAVELENGTH_M)
        reference += (kernel * areas[source_index]).sum().to(torch.complex128)
    relative_error = float((measured - reference).abs() / reference.abs().clamp_min(1.0e-300))
    _require(relative_error <= 2.0e-5,
             "production RS far-axis spatial tile differs from the independent coordinate sum")
    return {"status": "passed", "window_cells": 4096, "tile_shape": [1, 1],
            "target_node": 0, "source_chunk_size": config.window_power_source_chunk,
            "target_offset_cells": list(offset), "measured": [float(measured.real), float(measured.imag)],
            "reference": [float(reference.real), float(reference.imag)],
            "relative_error": relative_error, "tolerance": 2.0e-5}


def _pairing_negative_control(
    model: FinitePixelChenLightGen400, pixels: torch.Tensor,
    population: dict[str, torch.Tensor], device: torch.device,
) -> dict[str, Any]:
    rows = population["fit_rows_by_class"][:, 0].long()
    target, _ = baseline._image_batch(pixels, rows, device)
    model.eval()
    with torch.no_grad():
        result = model.forward_from_image(target)
        task = FIXED_GAIN * result["physical_w_per_pixel"]
    wrong_rows = rows.roll(1)
    mismatch_rejected = False
    try:
        baseline._reconstruction_batch_metrics(task, target, rows, wrong_rows)
    except ValueError:
        mismatch_rejected = True
    _require(mismatch_rejected, "production evaluator accepted mismatched target row identifiers")
    paired = float((task - target).square().mean().cpu())
    shuffled = float((task - target.roll(1, dims=0)).square().mean().cpu())
    content_changed = not math.isclose(paired, shuffled, rel_tol=1.0e-8, abs_tol=1.0e-12)
    _require(content_changed, "target-content permutation did not affect the production reconstruction metric")
    return {"fit_rows": rows.tolist(), "target_row_mismatch_rejected": mismatch_rejected,
            "paired_mse": paired, "shuffled_target_mse": shuffled,
            "target_content_permutation_changed_metric": content_changed}


def _source_phase_states(
    source_model: torch.nn.Module, start_state: dict[str, torch.Tensor], endpoint_payload: dict[str, Any]
) -> dict[str, dict[str, torch.Tensor]]:
    phases = {name: source_model.state_dict()[name].detach().cpu().clone()
              for name in ("encoder_phase", "decoder_phase")}
    _require(all(torch.equal(phases[name], start_state[name]) for name in phases),
             "historical 4096 start state is not the source 30000-step model")
    endpoint = {name: endpoint_payload["model"][name].detach().cpu().clone() for name in phases}
    return {"source_step_30000": phases, "historical_500_step_endpoint": endpoint}


def _save_checkpoint(
    path: Path, model: FinitePixelChenLightGen400, optimizer: torch.optim.Optimizer,
    branch: str, completed: int, order: int, eta: float, beta: float,
    population: dict[str, torch.Tensor], preflight_dir: Path, run_dir: Path,
    latest: dict[str, Any],
) -> dict[str, Any]:
    payload = {"schema": "chen_lightgen_400_finite_pixel_reconstruction_checkpoint.v1",
               "task_id": TASK_ID, "authorization_id": AUTHORITY, "branch": branch,
               "completed_updates": completed, "next_batch_cursor": completed % 10,
               "quadrature_order": order, "window_cells": 4096,
               "propagation_backend": PROPAGATION_BACKEND,
               "source_checkpoint_step": 30_000, "model": _cpu_state(model.state_dict()),
               "optimizer": _cpu_state(optimizer.state_dict()),
               "rng_cpu": torch.random.get_rng_state().cpu(),
               "rng_cuda": [value.cpu() for value in torch.cuda.get_rng_state_all()],
               "fixed_gain_w_inverse": FIXED_GAIN, "beta_rec": beta, "eta": eta,
               "fit_rows": population["fit_rows"].clone(),
               "confirm_rows": population["confirm_rows"].clone(),
               "preflight_dir": str(preflight_dir.resolve()), "run_dir": str(run_dir.resolve()),
               "latest": latest}
    if path.exists():
        prior = torch.load(path, map_location="cpu", weights_only=False)
        _require(prior.get("completed_updates") == completed and prior.get("branch") == branch
                 and prior.get("propagation_backend") == PROPAGATION_BACKEND
                 and baseline._assert_state_close(payload["model"], prior["model"], "checkpoint_repeat") == 0.0
                 and baseline._assert_state_close(payload["optimizer"], prior["optimizer"],
                                                  "checkpoint_optimizer_repeat") == 0.0,
                 "same-step checkpoint path already contains a different state")
        return prior
    atomic_torch_save_new(path, payload)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    _require(loaded["completed_updates"] == completed and loaded["next_batch_cursor"] == completed % 10
             and loaded["propagation_backend"] == PROPAGATION_BACKEND
             and baseline._assert_state_close(payload["model"], loaded["model"], "checkpoint_readback") == 0.0
             and baseline._assert_state_close(payload["optimizer"], loaded["optimizer"],
                                              "checkpoint_optimizer_readback") == 0.0,
             "finite-pixel checkpoint readback differs")
    return loaded


def _retain_checkpoints(branch_dir: Path) -> None:
    files = sorted(branch_dir.glob("checkpoint_step_*.pt"))
    keep = {branch_dir / "checkpoint_step_000500.pt", *files[-2:]}
    for path in files:
        if path not in keep and path.parent.resolve() == branch_dir.resolve():
            path.unlink()


def _load_resume_checkpoint(
    path: Path, model: FinitePixelChenLightGen400, optimizer: torch.optim.Optimizer,
    preflight_dir: Path, run_dir: Path, population: dict[str, torch.Tensor], order: int,
) -> dict[str, Any]:
    _require(path.is_file(), f"explicit resume checkpoint is missing: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    _require(payload.get("schema") == "chen_lightgen_400_finite_pixel_reconstruction_checkpoint.v1"
             and payload.get("task_id") == TASK_ID and payload.get("authorization_id") == AUTHORITY,
             "resume checkpoint belongs to another finite-pixel task")
    _require(payload.get("quadrature_order") == order and payload.get("window_cells") == 4096
             and payload.get("propagation_backend") == PROPAGATION_BACKEND
             and payload.get("source_checkpoint_step") == 30_000
             and payload.get("preflight_dir") == str(preflight_dir.resolve())
             and payload.get("run_dir") == str(run_dir.resolve()),
             "resume checkpoint order, source step, or explicit run path differs")
    _require(torch.equal(payload["fit_rows"], population["fit_rows"])
             and torch.equal(payload["confirm_rows"], population["confirm_rows"]),
             "resume checkpoint fixed TRAIN rows differ")
    completed = int(payload["completed_updates"])
    _require(payload["branch"] in ("A", "B") and 0 <= completed <= UPDATES
             and payload["next_batch_cursor"] == completed % 10,
             "resume checkpoint branch, update count, or next-batch cursor is invalid")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if completed:
        baseline._validate_adam_state(optimizer, completed_updates=completed)
    _set_rng(payload["rng_cpu"], payload["rng_cuda"])
    return payload


def _rows_at_cursor(population: dict[str, torch.Tensor], cursor: int) -> torch.Tensor:
    _require(0 <= cursor < 10, "fixed fit-cycle cursor is outside 0..9")
    return population["fit_rows_by_class"][:, cursor].long()


def _train_branch(
    branch: str, model: FinitePixelChenLightGen400, optimizer: torch.optim.Optimizer,
    start: int, population: dict[str, torch.Tensor], pixels: torch.Tensor,
    branch_dir: Path, run_dir: Path, preflight_dir: Path, order: int,
    beta: float, eta: float, latest: dict[str, Any],
) -> dict[str, Any]:
    interrupted: list[int] = []
    old_handlers: dict[int, Any] = {}

    def handler(signum: int, frame: Any) -> None:
        interrupted.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, handler)
    started = time.perf_counter()
    last_heartbeat = started
    try:
        for step0 in range(start, UPDATES):
            cursor = step0 % 10
            rows = _rows_at_cursor(population, cursor)
            latest = _train_update(model, optimizer, pixels, rows, branch, beta, eta, 1,
                                   torch.device("cuda"))
            completed = step0 + 1
            now = time.perf_counter()
            record = {"branch": branch, "completed_updates": completed, "rows": rows.tolist(),
                      "latest": latest, "elapsed_seconds": now - started,
                      "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()}
            if completed % 100 == 0:
                with (branch_dir / "training_history.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(baseline._json_safe(record), ensure_ascii=False) + "\n")
            if completed % CHECKPOINT_INTERVAL == 0 or interrupted:
                checkpoint_path = branch_dir / f"checkpoint_step_{completed:06d}.pt"
                _save_checkpoint(checkpoint_path, model, optimizer, branch, completed, order, eta, beta,
                                 population, preflight_dir, run_dir, latest)
                _retain_checkpoints(branch_dir)
                atomic_write_json(run_dir / "progress.json", {
                    "status": "INTERRUPTED" if interrupted else "RUNNING", "branch": branch,
                    "completed_updates": completed, "next_batch_cursor": completed % 10,
                    "checkpoint": str(checkpoint_path), "elapsed_seconds": now - started})
            if now - last_heartbeat >= 60.0:
                atomic_write_json(run_dir / "heartbeat.json", {
                    "time": time.time(), "branch": branch, "completed_updates": completed,
                    "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved()})
                last_heartbeat = now
            if interrupted:
                _require(checkpoint_path.is_file(), "signal-handled recovery checkpoint was not written")
                with (run_dir / "segments.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps({"event": "interrupted", "branch": branch,
                                             "completed_updates": completed,
                                             "signal": interrupted[-1], "time": time.time()}) + "\n")
                return {"status": "INTERRUPTED", "branch": branch,
                        "completed_updates": completed, "checkpoint": str(checkpoint_path)}
    finally:
        for signum, old_handler in old_handlers.items():
            signal.signal(signum, old_handler)
    terminal = branch_dir / f"checkpoint_step_{UPDATES:06d}.pt"
    _require(terminal.is_file(), f"{branch} 500-step checkpoint is missing")
    return {"status": "COMPLETE", "branch": branch, "completed_updates": UPDATES,
            "checkpoint": str(terminal), "latest": latest,
            "elapsed_seconds": time.perf_counter() - started}


def _measure_state(
    state_name: str, phases: dict[str, torch.Tensor], order: int,
    population: dict[str, torch.Tensor], pixels: torch.Tensor, labels: torch.Tensor,
    evaluator: torch.nn.Module, run_dir: Path, device: torch.device,
) -> dict[str, Any]:
    output_dir = run_dir / "measurements" / state_name
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = population["all_rows"].long()
    expected_labels = population["all_labels"].long()
    fit_set = set(population["fit_rows"].tolist())
    records: list[dict[str, Any]] = []
    port_records: list[dict[str, Any]] = []
    predictions: list[int] = []
    image_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    forward_count = 0
    interrupted: list[int] = []
    old_handlers: dict[int, Any] = {}

    def record_signal(signum: int, frame: Any) -> None:
        interrupted.append(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, record_signal)
    model = _model_from_phases(phases, order, 4096, device)
    model.eval()
    evaluator.eval()
    started = time.perf_counter()
    block_size = 10
    for block_index, start in enumerate(range(0, all_rows.numel(), block_size)):
        rows = all_rows[start:start + block_size]
        expected_block_labels = expected_labels[start:start + block_size]
        cache_path = output_dir / f"block_{block_index:03d}.pt"
        if cache_path.is_file():
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            _require(payload.get("state") == state_name and payload.get("order") == order
                     and torch.equal(payload.get("rows"), rows)
                     and torch.equal(payload.get("labels"), expected_block_labels),
                     f"saved measurement block {cache_path} does not match fixed rows/order")
            records.extend(payload["records"])
            port_records.extend(payload["ports"])
            predictions.extend(payload["predictions"])
            for row, pair in payload["image_cache"].items():
                image_cache[int(row)] = pair
            if interrupted:
                for signum, old_handler in old_handlers.items():
                    signal.signal(signum, old_handler)
                return {"status": "INTERRUPTED", "state": state_name,
                        "completed_rows": len(records), "signal": interrupted[-1]}
            continue
        block_records: list[dict[str, Any]] = []
        block_ports: list[dict[str, Any]] = []
        block_predictions: list[int] = []
        block_images: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for row_tensor, label_tensor in zip(rows, expected_block_labels, strict=True):
            row, label = int(row_tensor), int(label_tensor)
            target, _ = baseline._image_batch(pixels, row_tensor.reshape(1), device)
            with torch.no_grad():
                result = model.forward_from_image(
                    target, capture_fields=True, measure_full_window_power=True
                )
                forward_count += 1
                physical = result["physical_w_per_pixel"]
                task = FIXED_GAIN * physical
                view, psnr, ssim = baseline._reconstruction_batch_metrics(
                    task, target, row_tensor.reshape(1), row_tensor.reshape(1))
                logits = evaluator(view)
            prediction = int(logits.argmax(1)[0].cpu())
            block_predictions.append(prediction)
            coefficients = result["ols_coefficients"][0].detach().cpu()
            port_power = coefficients.abs().square().double()
            total_modal = float(port_power.sum())
            denominator = float(port_power.square().sum())
            modal_input = float(result["ols_input_power_w"][0].cpu())
            slm_output_power = float(result["input_power_w"][0].cpu())
            record = {"state": state_name, "row": row, "label": label,
                      "population": "fit" if row in fit_set else "confirmation",
                      "mse400_unclipped": float((task - target).square().mean().cpu()),
                      "psnr28_db": float(psnr[0].cpu()), "ssim28": float(ssim[0].cpu()),
                      "saturation_fraction": float((((task < 0.0) | (task > 1.0)).float().mean()).cpu()),
                      "saturation_excess_mean": float((task.clamp_min(0.0) - task.clamp_max(1.0)).abs().mean().cpu()),
                      "ed_prediction_descriptive": prediction,
                      "slm_output_power_w": slm_output_power,
                      "ols_incident_power_w": modal_input, "total_ols_modal_power_w": total_modal,
                      "system_coupling_efficiency": total_modal / max(slm_output_power, 1.0e-300),
                      "modal_capture_fraction": total_modal / max(modal_input, 1.0e-300),
                      "weakest_port_power_w": float(port_power.min()),
                      "ports_at_least_1nw": int((port_power >= 1.0e-9).sum()),
                      "n_eff": total_modal**2 / max(denominator, 1.0e-300),
                      "camera_roi_power_w": float(result["camera_roi_power_w"][0].cpu()),
                      "camera_observed_power_w": float(result["camera_observed_power_w"][0].cpu()),
                      "plane_power_scope": result["plane_power_scope"],
                      "seven_plane_power_w": result["plane_powers_w"][0].detach().cpu().tolist()}
            block_records.append(record)
            for port, value in enumerate(port_power.tolist()):
                block_ports.append({"state": state_name, "row": row, "label": label,
                                    "population": record["population"], "port": port,
                                    "row_index": port // 10, "column_index": port % 10,
                                    "modal_power_w": value})
            if bool(population["confirm_rows_by_class"][label, :3].eq(row).any()):
                target28 = F.adaptive_avg_pool2d(target, (TASK_SIZE, TASK_SIZE))[0, 0].cpu()
                view28 = view[0, 0].detach().cpu()
                block_images[row] = (target28, view28)
            del target, result, task, physical, view, logits
        payload = {"schema": "chen_lightgen_400_finite_pixel_measurement_block.v1",
                   "state": state_name, "order": order, "rows": rows.clone(),
                   "labels": expected_block_labels.clone(), "records": block_records,
                   "ports": block_ports, "predictions": block_predictions,
                   "image_cache": block_images}
        atomic_torch_save_new(cache_path, payload)
        loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
        _require(torch.equal(loaded["rows"], rows) and torch.equal(loaded["labels"], expected_block_labels),
                 "measurement block row/label readback differs")
        records.extend(block_records)
        port_records.extend(block_ports)
        predictions.extend(block_predictions)
        image_cache.update(block_images)
        atomic_write_json(run_dir / "progress.json", {
            "status": "INTERRUPTED_EVALUATION" if interrupted else "MEASURING",
            "state": state_name, "completed_rows": min(start + block_size, 300),
            "expected_rows": 300, "elapsed_seconds": time.perf_counter() - started,
            "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
            "signal": interrupted[-1] if interrupted else None})
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if interrupted:
            for signum, old_handler in old_handlers.items():
                signal.signal(signum, old_handler)
            return {"status": "INTERRUPTED", "state": state_name,
                    "completed_rows": min(start + block_size, 300), "signal": interrupted[-1],
                    "forward_count": forward_count}
    for signum, old_handler in old_handlers.items():
        signal.signal(signum, old_handler)
    _require(len(records) == 300 and len({record["row"] for record in records}) == 300
             and len(port_records) == 30_000,
             f"{state_name} final measurement does not cover the fixed population exactly once")
    requested = torch.tensor([record["label"] for record in records])
    predicted = torch.tensor(predictions)
    summaries = {}
    for name in ("fit", "confirmation"):
        subset = [record for record in records if record["population"] == name]
        mask = torch.tensor([record["population"] == name for record in records])
        classification = baseline._classification_summary(requested[mask], predicted[mask])
        summaries[name] = {
            "count": len(subset),
            "mse400_unclipped_mean": float(np.mean([item["mse400_unclipped"] for item in subset])),
            "psnr28_db_mean": float(np.mean([item["psnr28_db"] for item in subset])),
            "ssim28_mean": float(np.mean([item["ssim28"] for item in subset])),
            "saturation_fraction": float(np.mean([item["saturation_fraction"] for item in subset])),
            "ed_macro_recall_descriptive": classification["macro_recall"],
            "ed_recall_by_class_descriptive": classification["recall_by_class"].tolist(),
            "ed_confusion_matrix_descriptive": classification["confusion_matrix"].tolist(),
            "mean_total_ols_modal_power_w": float(np.mean([item["total_ols_modal_power_w"] for item in subset])),
            "mean_ports_at_least_1nw": float(np.mean([item["ports_at_least_1nw"] for item in subset])),
            "mean_n_eff": float(np.mean([item["n_eff"] for item in subset])),
            "mean_system_coupling_efficiency": float(np.mean([item["system_coupling_efficiency"] for item in subset])),
            "mean_modal_capture_fraction": float(np.mean([item["modal_capture_fraction"] for item in subset])),
        }
    return {"records": records, "ports": port_records, "summaries": summaries,
            "image_cache": image_cache, "elapsed_seconds": time.perf_counter() - started,
            "forward_count": forward_count, "blocks_readback": 30}


def _write_atlas(path: Path, labels: torch.Tensor, population: dict[str, torch.Tensor],
                 images: dict[str, dict[int, tuple[torch.Tensor, torch.Tensor]]]) -> None:
    selected = [int(value) for value in population["confirm_rows_by_class"][:, :3].flatten()]
    states = ("source_step_30000", "A500", "B500")
    tile = 28
    margin = 4
    label_width = 22
    canvas = Image.new("L", (label_width + margin + len(states + ("target",)) * (tile + margin),
                             margin + len(selected) * (tile + margin)), color=255)
    draw = ImageDraw.Draw(canvas)
    title_names = ("target",) + states
    for column, name in enumerate(title_names):
        draw.text((label_width + margin + column * (tile + margin), 1), name, fill=0)
    for index, row in enumerate(selected):
        label = int(labels[row])
        y = margin + index * (tile + margin)
        draw.text((1, y + 8), str(label), fill=0)
        target_image = images["source_step_30000"][row][0]
        values = [target_image] + [images[state][row][1] for state in states]
        for column, value in enumerate(values):
            image = Image.fromarray((value.clamp(0, 1).numpy() * 255).round().astype("uint8"), mode="L")
            canvas.paste(image, (label_width + margin + column * (tile + margin), y))
    canvas.save(path, format="PNG")


def _read_ready(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    preflight_dir = _resolve(args.preflight_dir)
    preflight = json.loads((preflight_dir / "preflight.json").read_text(encoding="utf-8"))
    ready = json.loads((preflight_dir / "preflight_ready.json").read_text(encoding="utf-8"))
    _require(preflight.get("status") == "READY" and preflight.get("task_id") == TASK_ID
             and preflight.get("authorization_id") == AUTHORITY
             and preflight.get("propagation_backend") == PROPAGATION_BACKEND,
             "an explicit numerically qualified READY preflight is required")
    _require(preflight.get("run_identity", {}).get("config_identity", {}).get("propagation_backend")
             == PROPAGATION_BACKEND,
             "READY run identity does not bind the production RS backend")
    _require(preflight.get("receipt") == ready and ready.get("status") == "READY"
             and ready.get("authorization_id") == AUTHORITY,
             "preflight READY receipt is missing or inconsistent")
    _require(all((preflight_dir / name).is_file() for name in
                 ("fixed_population.pt", "start_model_state.pt", "start_rng.pt")),
             "READY preflight support state is incomplete")
    supplied = {name: str(_resolve(getattr(args, name))) for name in (
        "source_checkpoint", "source_report", "source_start_state", "source_endpoint_checkpoint",
        "fixed_population", "config", "data", "evaluator")}
    _require(preflight.get("inputs") == supplied, "run inputs differ from explicit READY preflight inputs")
    return preflight, ready


def _load_data_again(args: argparse.Namespace):
    return baseline._load_data(_resolve(args.data))


def _report_closeout(path: Path, report: dict[str, Any]) -> None:
    decision = report["decision"]
    fit_start = report["measurement"]["source_step_30000"]["fit"]["mse400_unclipped_mean"]
    fit_a = report["measurement"]["A500"]["fit"]["mse400_unclipped_mean"]
    fit_b = report["measurement"]["B500"]["fit"]["mse400_unclipped_mean"]
    fit_gap = (f"拟合集MSE从 {fit_start:.6g} 变为 A 的 {fit_a:.6g}、B 的 {fit_b:.6g}；"
               "这只描述固定拟合样本，不替代确认集门槛。")
    md = (
        "# LightGen400 有限像素传播与重建目标 A/B\n\n"
        "## 结果与状态\n\n"
        f"状态：`{report['status']}`。最终判定：`{decision}`。"
        f"确认集相对起点的 MSE 改善，A 为 {report['local_gate']['A']['confirm_mse_improvement_fraction']:.2%}，"
        f"B 为 {report['local_gate']['B']['confirm_mse_improvement_fraction']:.2%}；"
        f"SSIM变化分别为 {report['local_gate']['A']['confirm_ssim_delta']:+.5f}、"
        f"{report['local_gate']['B']['confirm_ssim_delta']:+.5f}。\n\n"
        "## 科学认识\n\n"
        f"本轮在数值资格通过的有限像素离散路径、固定 400×400 相位参数和当前 OLS 项目假设下比较纯 MSE 与 MSE+结构项。"
        f"{report['scientific_interpretation']}\n\n"
        "## 目标缺口\n\n"
        "结论限于100张拟合和200张确认TRAIN样本、固定传播窗口及500步预算；尚未形成完整TRAIN基线、DEV/TEST验收、生成验证或人眼可读性结论。\n\n"
        "## 工程完成与验证\n\n"
        f"数值候选为 Gauss–Legendre 阶数 {report['preflight']['selected_quadrature_order']}，"
        f"4096物理窗口；A/B各完成500步。预检覆盖独立有限像素RS、OLS/功率、相机面积积分、"
        "六层梯度、跨窗口、批次等价、错配负控及真实独立进程SIGINT/SIGTERM恢复。"
        f"训练与测量文件位于 `{report['run_dir']}`。\n\n"
        "## 决策影响与下一步\n\n"
        f"{fit_gap}本轮到此停止，不自动追加步数、改功率、重拟生成分布或采用模型。"
        "完整基线、DEV与生成验收需另行设计和授权。\n"
    )
    path.write_text(md, encoding="utf-8")


def _summarize_and_report(
    run_dir: Path, preflight: dict[str, Any], state_results: dict[str, dict[str, Any]],
    labels: torch.Tensor, population: dict[str, torch.Tensor], training: dict[str, Any],
    elapsed: float, peak_memory: int,
) -> dict[str, Any]:
    start_confirm = state_results["source_step_30000"]["summaries"]["confirmation"]
    gates: dict[str, Any] = {}
    all_records = []
    all_ports = []
    for state_name, result in state_results.items():
        all_records.extend(result["records"])
        all_ports.extend(result["ports"])
        summary = result["summaries"]["confirmation"]
        if state_name != "source_step_30000":
            improvement = 1.0 - summary["mse400_unclipped_mean"] / start_confirm["mse400_unclipped_mean"]
            gates[state_name[0]] = {
                "passed": improvement >= 0.10
                and summary["ssim28_mean"] >= start_confirm["ssim28_mean"]
                and summary["saturation_fraction"] <= 0.01,
                "confirm_mse_improvement_fraction": improvement,
                "confirm_ssim_delta": summary["ssim28_mean"] - start_confirm["ssim28_mean"],
                "confirm_saturation_fraction": summary["saturation_fraction"],
                "start_confirmation": start_confirm,
                "endpoint_confirmation": summary,
            }
    passed_branches = [branch for branch in ("A", "B") if gates[branch]["passed"]]
    status = "VALID_COMPLETED" if passed_branches else "VALID_FAILED"
    if passed_branches == ["A"]:
        decision = "PURE_MSE_LOCAL_IMPROVEMENT_SUPPORTED"
        scientific = "A满足预设局部标准，支持该数值路径上的纯重建目标具有有限预算改善能力；B未满足时，不支持结构项在本次配比下改善该候选。"
    elif passed_branches == ["B"]:
        decision = "STRUCTURE_AUGMENTED_LOCAL_IMPROVEMENT_SUPPORTED"
        scientific = "只有B满足预设局部标准，支持本轮固定结构权重值得在后续基线设计中进一步检验；不能推广为结构损失普遍更优。"
    elif passed_branches == ["A", "B"]:
        decision = "BOTH_OBJECTIVES_LOCAL_IMPROVEMENT_SUPPORTED"
        scientific = "两臂均满足预设局部标准，支持该数值路径与预算内两种目标均存在局部重建改善能力。"
    else:
        decision = "FIXED_MSE_STRUCTURE_INTERVENTIONS_NOT_SUPPORTED"
        scientific = "两臂都未满足预设局部标准，说明本次固定起点、数值路径、目标配比和500步干预未获支持，不构成架构容量不足的证据。"
    _write_csv(run_dir / "per_sample_metrics.csv", all_records)
    _write_csv(run_dir / "per_port_power.csv", all_ports)
    images = {state: result["image_cache"] for state, result in state_results.items()}
    _write_atlas(run_dir / "fixed_index_atlas.png", labels, population, images)
    report = {"schema": "chen_lightgen_400_finite_pixel_reconstruction_ab_report.v1",
              "task_id": TASK_ID, "authorization_id": AUTHORITY, "status": status,
              "decision": decision, "scientific_interpretation": scientific,
              "run_dir": str(run_dir.resolve()), "preflight": {
                  "status": preflight["status"], "selected_quadrature_order": preflight["selected_quadrature_order"],
                  "direct_rs_status": preflight["numerical_qualification"]["direct_rs"]["status"],
                  "fft_embedding_status": preflight["numerical_qualification"]["fft_embedding_checks"]["status"],
                  "gradient_status": preflight["numerical_qualification"]["gradient_checks"]["status"]},
              "data_roles": {"fit": 100, "confirmation": 200, "both_canonical_train": True,
                             "dev_reads": 0, "official_test_reads": 0},
              "physical_units": {"camera": "W/pixel", "task": "unclipped g0 times physical W/pixel",
                                 "display": "clipped to [0,1] only for visualization and evaluator"},
              "fixed_gain_w_inverse": FIXED_GAIN,
              "training": training, "measurement": {
                  state: {"forward_count": result["forward_count"],
                          "blocks_readback": result["blocks_readback"],
                          "elapsed_seconds": result["elapsed_seconds"],
                          "fit": result["summaries"]["fit"],
                          "confirmation": result["summaries"]["confirmation"]}
                  for state, result in state_results.items()},
              "local_gate": {"criteria": {"confirm_mse_improvement_min": 0.10,
                                           "ssim_not_lower": True, "saturation_fraction_max": 0.01},
                             "A": gates["A"], "B": gates["B"]},
              "evaluation_artifacts": {"per_sample_metrics": "per_sample_metrics.csv",
                                       "per_port_power": "per_port_power.csv",
                                       "fixed_index_atlas": "fixed_index_atlas.png"},
              "elapsed_seconds": elapsed, "peak_reserved_vram_bytes": peak_memory,
              "official_test_reads": 0}
    atomic_write_new_json(run_dir / "report.json", baseline._json_safe(report))
    _report_closeout(run_dir / "results_analysis.md", report)
    return report


def _run_or_resume(args: argparse.Namespace) -> dict[str, Any]:
    _require(torch.cuda.is_available(), "finite-pixel LightGen400 A/B requires CUDA")
    preflight, receipt = _read_ready(args)
    run_dir = _resolve(args.output_dir)
    preflight_dir = _resolve(args.preflight_dir)
    if args.mode == "run":
        _require(not run_dir.exists(), f"refusing to overwrite attempt directory: {run_dir}")
        _require(args.resume_checkpoint is None, "fresh A/B run cannot consume a resume checkpoint")
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        _require(run_dir.is_dir() and args.resume_checkpoint is not None,
                 "resume requires its existing attempt directory and an explicit checkpoint")
        _require(not (run_dir / "report.json").is_file(), "completed A/B report cannot be resumed")
    device = torch.device("cuda")
    (paths, config, source_checkpoint, source_report, source_model, references, source_start_state,
     historical_endpoint, pixels, labels, population, evaluator) = _load_inputs(args, device)
    saved_population = torch.load(preflight_dir / "fixed_population.pt", map_location="cpu", weights_only=False)
    _require(baseline._assert_state_close(saved_population, population, "run_fixed_population") == 0.0,
             "run fixed population differs from READY preflight")
    order = int(preflight["selected_quadrature_order"])
    beta = float(preflight["source_bogt_beta_rec"])
    eta = float(preflight["structure_coefficient"]["eta"])
    source_phases = {name: source_model.state_dict()[name].detach().cpu().clone()
                     for name in ("encoder_phase", "decoder_phase")}
    _require(all(torch.equal(source_phases[name], source_start_state[name]) for name in source_phases),
             "run source 30000 phases differ from explicit preflight start state")
    saved_start_state = torch.load(preflight_dir / "start_model_state.pt", map_location="cpu", weights_only=False)
    _require(all(torch.equal(source_phases[name], saved_start_state[name]) for name in source_phases),
             "run source phases differ from the saved READY preflight state")
    identity = ResearchRunIdentity.from_mapping(preflight["run_identity"])
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    manifest_path = run_dir / "run_manifest.json"
    if args.mode == "run":
        control.open_scientific_attempt(run_dir, attempt_id=run_dir.name, receipt=receipt)
        manifest = {"schema": "chen_lightgen_400_finite_pixel_reconstruction_ab_run.v1",
                    "task_id": TASK_ID, "run_id": identity.run_id, "attempt_id": run_dir.name,
                    "preflight_dir": str(preflight_dir.resolve()),
                    "explicit_inputs": {key: str(value.resolve()) for key, value in paths.items()},
                    "selected_quadrature_order": order, "propagation_backend": PROPAGATION_BACKEND,
                    "window_cells": 4096,
                    "linear_convolution_fft_shape": list(FinitePixelRSLinear._fft_shape((400, 400), (4096, 4096))),
                    "plane_power_scope": "area-integrated RS field over the configured full physical window",
                    "updates_per_branch": UPDATES, "fixed_gain_w_inverse": FIXED_GAIN,
                    "fit_rows": population["fit_rows"].tolist(),
                    "confirmation_rows": population["confirm_rows"].tolist(),
                    "evaluator": str(paths["evaluator"].resolve())}
        atomic_write_new_json(manifest_path, manifest)
        atomic_torch_save_new(run_dir / "fixed_population.pt", population)
        atomic_torch_save_new(run_dir / "start_phase_state.pt", source_phases)
        rng = torch.load(preflight_dir / "start_rng.pt", map_location="cpu", weights_only=False)
        start_rng_cpu, start_rng_cuda = rng["cpu"], rng["cuda"]
        _set_rng(start_rng_cpu, start_rng_cuda)
        resume_payload = None
        start_payload = None
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(manifest.get("task_id") == TASK_ID and manifest.get("run_id") == identity.run_id
                 and manifest.get("attempt_id") == run_dir.name
                 and manifest.get("preflight_dir") == str(preflight_dir.resolve())
                 and manifest.get("explicit_inputs") == {key: str(value.resolve()) for key, value in paths.items()}
                 and manifest.get("selected_quadrature_order") == order
                 and manifest.get("propagation_backend") == PROPAGATION_BACKEND,
                 "resume inputs differ from the explicit original run manifest")
        control.resume_scientific_attempt(run_dir, attempt_id=run_dir.name, receipt=receipt)
        start_payload = torch.load(_resolve(args.resume_checkpoint), map_location="cpu", weights_only=False)
        resume_payload = start_payload
        rng = torch.load(preflight_dir / "start_rng.pt", map_location="cpu", weights_only=False)
        start_rng_cpu, start_rng_cuda = rng["cpu"], rng["cuda"]
        if start_payload["branch"] == "A" and start_payload["completed_updates"] == 500:
            _require(not (run_dir / "B" / "checkpoint_step_000100.pt").exists(),
                     "B already has a recovery checkpoint; resume from that explicit B path")
    if args.mode == "run":
        branch_a_checkpoint = run_dir / "A" / "checkpoint_step_000500.pt"
        branch_b_checkpoint = run_dir / "B" / "checkpoint_step_000500.pt"
        start_branch = "A"
    else:
        branch_a_checkpoint = run_dir / "A" / "checkpoint_step_000500.pt"
        branch_b_checkpoint = run_dir / "B" / "checkpoint_step_000500.pt"
        start_branch = resume_payload["branch"]
    training: dict[str, Any] = {}
    for branch in ("A", "B"):
        branch_dir = run_dir / branch
        branch_dir.mkdir(parents=True, exist_ok=True)
        if branch == "A" and start_branch == "B":
            if not branch_a_checkpoint.is_file():
                _require(resume_payload["completed_updates"] == 500,
                         "B resume requires completed A500 checkpoint")
            else:
                training["A"] = {"status": "COMPLETE", "checkpoint": str(branch_a_checkpoint)}
            continue
        if branch == "B" and start_branch == "A" and resume_payload is not None:
            if resume_payload["completed_updates"] < 500:
                break
            start_branch = "B"
            _set_rng(start_rng_cpu, start_rng_cuda)
            model = _model_from_phases(source_phases, order, 4096, device)
            _set_trainable(model)
            optimizer = _optimizer(model)
            started_updates = 0
            latest = {}
        elif branch == "A" and resume_payload is not None and start_branch == "A":
            model = _model_from_phases(source_phases, order, 4096, device)
            _set_trainable(model)
            optimizer = _optimizer(model)
            loaded = _load_resume_checkpoint(_resolve(args.resume_checkpoint), model, optimizer,
                                             preflight_dir, run_dir, population, order)
            _require(loaded["branch"] == "A", "resume checkpoint branch changed")
            started_updates = int(loaded["completed_updates"])
            latest = loaded.get("latest", {})
        elif branch == "B" and resume_payload is not None and start_branch == "B":
            model = _model_from_phases(source_phases, order, 4096, device)
            _set_trainable(model)
            optimizer = _optimizer(model)
            loaded = _load_resume_checkpoint(_resolve(args.resume_checkpoint), model, optimizer,
                                             preflight_dir, run_dir, population, order)
            _require(loaded["branch"] == "B", "resume checkpoint branch changed")
            started_updates = int(loaded["completed_updates"])
            latest = loaded.get("latest", {})
        else:
            _set_rng(start_rng_cpu, start_rng_cuda)
            model = _model_from_phases(source_phases, order, 4096, device)
            _set_trainable(model)
            optimizer = _optimizer(model)
            started_updates = 0
            latest = {}
        outcome = _train_branch(branch, model, optimizer, started_updates, population, pixels,
                                branch_dir, run_dir, preflight_dir, order, beta, eta, latest)
        training[branch] = outcome
        if outcome["status"] == "INTERRUPTED":
            return {"status": "INTERRUPTED", "branch": branch,
                    "completed_updates": outcome["completed_updates"],
                    "resume_checkpoint": outcome["checkpoint"]}
        _require(outcome["status"] == "COMPLETE", f"{branch} did not reach its fixed 500-step endpoint")
        if branch == "A":
            _require(branch_a_checkpoint.is_file(), "A500 checkpoint missing after branch completion")
            start_branch = "B"
            resume_payload = None
        else:
            _require(branch_b_checkpoint.is_file(), "B500 checkpoint missing after branch completion")
        del model, optimizer
        torch.cuda.empty_cache()
    if not branch_a_checkpoint.is_file():
        _require(start_payload is not None and start_payload["branch"] == "A"
                 and start_payload["completed_updates"] == 500,
                 "A500 completion evidence is missing")
    if not branch_b_checkpoint.is_file():
        _require(start_payload is not None and start_payload["branch"] == "B"
                 and start_payload["completed_updates"] == 500,
                 "B500 completion evidence is missing")
    checkpoint_a = torch.load(branch_a_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_b = torch.load(branch_b_checkpoint, map_location="cpu", weights_only=False)
    _require(checkpoint_a["completed_updates"] == checkpoint_b["completed_updates"] == 500
             and checkpoint_a["branch"] == "A" and checkpoint_b["branch"] == "B",
             "A/B terminal checkpoint identity or step is wrong")
    measurement_started = time.perf_counter()
    initial_result = _measure_state("source_step_30000", source_phases, order, population,
                                    pixels, labels, evaluator, run_dir, device)
    if initial_result.get("status") == "INTERRUPTED":
        atomic_write_json(run_dir / "progress.json", {
            "status": "INTERRUPTED_EVALUATION", "state": "source_step_30000",
            "completed_rows": initial_result["completed_rows"],
            "resume_checkpoint": str(branch_b_checkpoint.resolve()),
            "signal": initial_result["signal"]})
        return {"status": "INTERRUPTED_EVALUATION", **initial_result,
                "resume_checkpoint": str(branch_b_checkpoint.resolve())}
    phase_a = {name: checkpoint_a["model"][name] for name in source_phases}
    result_a = _measure_state("A500", phase_a, order, population, pixels, labels,
                              evaluator, run_dir, device)
    if result_a.get("status") == "INTERRUPTED":
        atomic_write_json(run_dir / "progress.json", {
            "status": "INTERRUPTED_EVALUATION", "state": "A500",
            "completed_rows": result_a["completed_rows"],
            "resume_checkpoint": str(branch_b_checkpoint.resolve()),
            "signal": result_a["signal"]})
        return {"status": "INTERRUPTED_EVALUATION", **result_a,
                "resume_checkpoint": str(branch_b_checkpoint.resolve())}
    phase_b = {name: checkpoint_b["model"][name] for name in source_phases}
    result_b = _measure_state("B500", phase_b, order, population, pixels, labels,
                              evaluator, run_dir, device)
    if result_b.get("status") == "INTERRUPTED":
        atomic_write_json(run_dir / "progress.json", {
            "status": "INTERRUPTED_EVALUATION", "state": "B500",
            "completed_rows": result_b["completed_rows"],
            "resume_checkpoint": str(branch_b_checkpoint.resolve()),
            "signal": result_b["signal"]})
        return {"status": "INTERRUPTED_EVALUATION", **result_b,
                "resume_checkpoint": str(branch_b_checkpoint.resolve())}
    evaluator_after = _cpu_state(evaluator.state_dict())
    evaluator_before = _cpu_state(baseline._load_evaluator(paths["evaluator"], device).state_dict())
    _require(baseline._assert_state_close(evaluator_before, evaluator_after, "finite_pixel_evaluator_run") == 0.0,
             "measurement modified the frozen E_D evaluator")
    report = _summarize_and_report(
        run_dir, preflight, {"source_step_30000": initial_result, "A500": result_a, "B500": result_b},
        labels, population, {"A": training.get("A", {"checkpoint": str(branch_a_checkpoint)}),
                             "B": training.get("B", {"checkpoint": str(branch_b_checkpoint)})},
        time.perf_counter() - measurement_started, torch.cuda.max_memory_reserved(device))
    atomic_write_json(run_dir / "progress.json", {"status": report["status"],
                        "completed_updates_per_branch": UPDATES, "evaluation_complete": True})
    with (run_dir / "segments.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": "completed", "time": time.time(),
                                 "decision": report["decision"]}) + "\n")
    if report["status"] == "VALID_COMPLETED":
        control.complete()
    else:
        control.valid_failed()
    return report


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    _require(bool(rows), f"refusing to write an empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _preflight_terminal(
    output_dir: Path, started: float, status: str, stage: str, **details: Any,
) -> dict[str, Any]:
    report = {"schema": "chen_lightgen_400_finite_pixel_preflight.v1", "status": status,
              "task_id": TASK_ID, "stage": stage, "training_updates": 0,
              "official_test_reads": 0, "elapsed_seconds": time.perf_counter() - started,
              **details}
    atomic_write_new_json(output_dir / "preflight.json", baseline._json_safe(report))
    atomic_write_json(output_dir / "progress.json", {"status": status, "stage": stage})
    return report


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    _require(args.propagation_backend == PROPAGATION_BACKEND,
             "the approved finite-pixel training path requires --propagation-backend rs_linear")
    output_dir = _resolve(args.output_dir)
    _require(not output_dir.exists(), f"refusing to overwrite preflight output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if not torch.cuda.is_available():
        return _preflight_terminal(output_dir, started, "PREFLIGHT_BLOCKED_RESOURCE", "cuda_unavailable",
                                  error="finite-pixel numerical qualification requires CUDA")
    device = torch.device("cuda")
    try:
        (paths, config, checkpoint, source_report, source_model, references, start_state,
         endpoint_payload, pixels, labels, population, evaluator) = _load_inputs(args, device)
    except torch.cuda.OutOfMemoryError as exc:
        return _preflight_terminal(output_dir, started, "PREFLIGHT_BLOCKED_RESOURCE", "input_load",
                                   error=str(exc))
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        return _preflight_terminal(output_dir, started, "INVALID_HARNESS", "explicit_input_validation",
                                   error=str(exc))
    state_phases = _source_phase_states(source_model, start_state, endpoint_payload)
    try:
        cache_import = _import_qualification_cache(
            args.reuse_qualification_cache, output_dir, state_phases, population
        )
    except (HarnessCheckError, FileNotFoundError, KeyError, RuntimeError, ValueError, OSError, EOFError) as exc:
        return _preflight_terminal(output_dir, started, "INVALID_HARNESS", "qualification_cache_import",
                                   error=str(exc))
    source_model.to("cpu")
    del references
    torch.cuda.empty_cache()
    beta = float(checkpoint["bogt_weights"]["rec"])
    torch.manual_seed(2026101701)
    torch.cuda.manual_seed_all(2026101701)
    initial_rng_cpu = torch.random.get_rng_state().cpu()
    initial_rng_cuda = [value.cpu() for value in torch.cuda.get_rng_state_all()]
    evaluator_before = _cpu_state(evaluator.state_dict())
    torch.cuda.reset_peak_memory_stats(device)
    resource = {"device_name": torch.cuda.get_device_name(device),
                "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory}
    atomic_write_json(output_dir / "progress.json", {"stage": "independent_rs", "status": "RUNNING"})
    try:
        rs_check = _direct_rs_path_check(device, output_dir)
    except torch.cuda.OutOfMemoryError as exc:
        partial_path = output_dir / "direct_rs_progress.json"
        partial = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.is_file() else None
        return _preflight_terminal(output_dir, started, "PREFLIGHT_BLOCKED_RESOURCE",
                                   "independent_RS_reference", error=str(exc), direct_rs_partial=partial)
    except HarnessCheckError as exc:
        return _preflight_terminal(output_dir, started, "INVALID_HARNESS",
                                   "independent_RS_reference", error=str(exc))
    atomic_write_new_json(output_dir / "direct_rs_reference.json", baseline._json_safe(rs_check))
    progress = {"status": "IN_PROGRESS", "stage": "independent_rs_complete",
                "direct_rs_status": rs_check["status"], "optimizer_updates": 0}
    atomic_write_json(output_dir / "preflight_progress.json", progress)
    rs_statuses = (rs_check["reference_status"], rs_check["reference_derivative_status"])
    if "INVALID_HARNESS" in rs_statuses:
        return _preflight_terminal(output_dir, started, "INVALID_HARNESS",
                                   "independent_RS_reference", direct_rs=rs_check)
    if any(status != "NUMERICALLY_QUALIFIED" for status in rs_statuses):
        return _preflight_terminal(output_dir, started, "NUMERICAL_NOT_QUALIFIED",
                                   "independent_RS_reference", direct_rs=rs_check)
    # This run is authorized to qualify q=8 specifically.  A lower quadrature
    # order passing its local probe is not a substitute candidate.
    candidate_orders = ([8] if rs_check["candidate_status"].get("8") == "NUMERICALLY_QUALIFIED"
                        else [])
    if not candidate_orders:
        status = ("INVALID_HARNESS" if "INVALID_HARNESS" in rs_check["candidate_status"].values()
                  else "NUMERICAL_NOT_QUALIFIED")
        return _preflight_terminal(output_dir, started, status, "candidate_RS_comparison",
                                   direct_rs=rs_check)
    try:
        boundary_checks = _production_boundary_checks(8, device)
    except torch.cuda.OutOfMemoryError as exc:
        return _preflight_terminal(output_dir, started, "PREFLIGHT_BLOCKED_RESOURCE",
                                   "production_boundary", candidate_orders=candidate_orders,
                                   error=str(exc))
    except HarnessCheckError as exc:
        return _preflight_terminal(output_dir, started, "INVALID_HARNESS",
                                   "production_boundary", candidate_orders=candidate_orders,
                                   error=str(exc))
    try:
        reference_convergence = _reference_convergence_qualification(
            state_phases, population, pixels, device, output_dir
        )
        if reference_convergence["status"] != "NUMERICALLY_QUALIFIED":
            return _preflight_terminal(
                output_dir, started, reference_convergence["status"], "full_chain_reference_convergence",
                direct_rs=rs_check, production_boundary_checks=boundary_checks,
                reference_convergence=reference_convergence,
            )
        full_chain = _full_chain_qualification(
            state_phases, population, labels, pixels, candidate_orders, device, output_dir
        )
    except torch.cuda.OutOfMemoryError as exc:
        return _preflight_terminal(output_dir, started, "PREFLIGHT_BLOCKED_RESOURCE",
                                   "full_chain_reference_or_candidate", candidate_orders=candidate_orders,
                                   production_boundary_checks=boundary_checks, error=str(exc))
    except HarnessCheckError as exc:
        return _preflight_terminal(output_dir, started, "INVALID_HARNESS",
                                   "full_chain_reference_or_candidate", candidate_orders=candidate_orders,
                                   production_boundary_checks=boundary_checks, error=str(exc))
    atomic_write_new_json(output_dir / "full_chain_qualification.json", baseline._json_safe(full_chain))
    atomic_write_new_json(output_dir / "reference_convergence.json",
                          baseline._json_safe(reference_convergence))
    field_qualified = [order for order in candidate_orders
                       if full_chain["orders"][str(order)]["status"] == "NUMERICALLY_QUALIFIED"]
    if not field_qualified:
        statuses = [entry["status"] for entry in full_chain["orders"].values()]
        status = "INVALID_HARNESS" if "INVALID_HARNESS" in statuses else "NUMERICAL_NOT_QUALIFIED"
        return _preflight_terminal(output_dir, started, status, "full_chain_field_and_OLS",
                                   direct_rs=rs_check, production_boundary_checks=boundary_checks,
                                   full_chain=full_chain)
    gradient_results: dict[str, Any] = {}
    embedding_results: dict[str, Any] = {}
    order_candidates = []
    for order in field_qualified:
        try:
            window = _fft_embedding_qualification(state_phases, population, pixels, order, device, output_dir)
        except torch.cuda.OutOfMemoryError as exc:
            window = {"status": "PREFLIGHT_BLOCKED_RESOURCE", "error": str(exc), "candidate_order": order}
        except HarnessCheckError as exc:
            window = {"status": "INVALID_HARNESS", "error": str(exc), "candidate_order": order}
        embedding_results[str(order)] = window
        if window["status"] != "NUMERICALLY_QUALIFIED":
            continue
        try:
            gradient = _gradient_qualification(state_phases, population, pixels, order, beta, device, output_dir)
        except torch.cuda.OutOfMemoryError as exc:
            gradient = {"status": "PREFLIGHT_BLOCKED_RESOURCE", "error": str(exc), "candidate_order": order}
        except HarnessCheckError as exc:
            gradient = {"status": "INVALID_HARNESS", "error": str(exc), "candidate_order": order}
        gradient_results[str(order)] = gradient
        if gradient["status"] == "NUMERICALLY_QUALIFIED":
            order_candidates.append(order)
    selected_order = min(order_candidates) if order_candidates else None
    if selected_order is None:
        statuses = [entry["status"] for entry in (*embedding_results.values(), *gradient_results.values())]
        if "PREFLIGHT_BLOCKED_RESOURCE" in statuses:
            status = "PREFLIGHT_BLOCKED_RESOURCE"
        elif "INVALID_HARNESS" in statuses:
            status = "INVALID_HARNESS"
        else:
            status = "NUMERICAL_NOT_QUALIFIED"
        return _preflight_terminal(output_dir, started, status, "window_or_gradient_qualification",
                                   direct_rs=rs_check, production_boundary_checks=boundary_checks,
                                   reference_convergence=reference_convergence, full_chain=full_chain,
                                   fft_embedding_checks=embedding_results,
                                   gradient_checks=gradient_results)
    eta_info: dict[str, float] = {}
    training_preflight: dict[str, Any] = {}
    initial_phases = state_phases["source_step_30000"]
    try:
        eta_model = _model_from_phases(initial_phases, selected_order, 4096, device)
        _set_trainable(eta_model)
        fit_first = population["fit_rows_by_class"][:, 0].long()
        eta_info = _calibrate_eta(eta_model, pixels, fit_first, beta, device)
        negative_control = _pairing_negative_control(eta_model, pixels, population, device)
        del eta_model
        for branch in ("A", "B"):
            _set_rng(initial_rng_cpu, initial_rng_cuda)
            training_preflight[branch] = _branch_preflight(
                initial_phases, population, pixels, paths["data"], branch, selected_order,
                beta, eta_info["eta"], device, output_dir
            )
    except torch.cuda.OutOfMemoryError as exc:
        report = {"schema": "chen_lightgen_400_finite_pixel_preflight.v1",
                  "status": "PREFLIGHT_BLOCKED_RESOURCE", "task_id": TASK_ID,
                  "stage": "conditional_training_preflight", "selected_quadrature_order": selected_order,
                  "error": str(exc), "training_updates": 0, "official_test_reads": 0,
                  "elapsed_seconds": time.perf_counter() - started}
        atomic_write_new_json(output_dir / "preflight.json", baseline._json_safe(report))
        return report
    except HarnessCheckError as exc:
        return _preflight_terminal(output_dir, started, "INVALID_HARNESS",
                                   "conditional_training_preflight", selected_quadrature_order=selected_order,
                                   error=str(exc), direct_rs=rs_check, full_chain=full_chain,
                                   reference_convergence=reference_convergence,
                                   fft_embedding_checks=embedding_results,
                                   gradient_checks=gradient_results)
    _set_rng(initial_rng_cpu, initial_rng_cuda)
    _require(baseline._assert_state_close(evaluator_before, _cpu_state(evaluator.state_dict()),
                                          "finite_pixel_evaluator_frozen") == 0.0,
             "preflight modified the frozen evaluator")
    manifest = capture_execution_source_manifest(
        ROOT, task_id=TASK_ID, semantic_authority_id=f"{TASK_ID}:FINITE_PIXEL_Q{selected_order}:A_B500",
        relevant_paths=(Path(__file__), Path("src/ldm_project/v2/application/research/lightgen/chen_lightgen_400_finite_pixel.py"),
                        Path("src/ldm_project/v2/application/research/lightgen/chen_lightgen_400_rs_convolution.py"),
                        Path("scripts/research/lightgen/diagnose_chen_lightgen_400_quickcheck_v1.py"),
                        Path("scripts/research/lightgen/execute_chen_lightgen_400_mnist_baseline_v1.py"),
                        Path("scripts/research/lightgen/diagnose_chen_lightgen_400_sampling_closure_v1.py"),
                        paths["source_checkpoint"], paths["source_report"], paths["source_start_state"],
                        paths["source_endpoint_checkpoint"], paths["fixed_population"], paths["config"],
                        paths["data"], paths["evaluator"]),
        config_identity={"quadrature_order": selected_order, "window": 4096,
                         "propagation_backend": PROPAGATION_BACKEND,
                         "pixel_pitch_m": PIXEL_PITCH_M,
                         "fit_rows": population["fit_rows"].tolist(),
                         "confirmation_rows": population["confirm_rows"].tolist()})
    run_id = uuid.uuid4().hex
    identity = ResearchRunIdentity(
        task_id=TASK_ID, run_id=run_id,
        trajectory_id=f"{TASK_ID}:step30000:q{selected_order}:seed20261017",
        semantic_authority_id=f"{TASK_ID}:FINITE_PIXEL_Q{selected_order}:A_B500",
        data_role="canonical_MNIST_TRAIN_fixed_fit_100_confirmation_200_no_dev_or_test_reads",
        config_identity={"quadrature_order": selected_order,
                         "propagation_backend": PROPAGATION_BACKEND,
                         "pixel_pitch_m": PIXEL_PITCH_M,
                         "window_cells": 4096, "window_m": 4096 * PIXEL_PITCH_M,
                         "common_roi_m": 1.5e-3,
                         "linear_convolution_fft_shapes": {
                             "train_active_400": list(FinitePixelRSLinear._fft_shape((400, 400), (400, 400))),
                             "measurement_roi_500": list(FinitePixelRSLinear._fft_shape((400, 400), (500, 500))),
                             "full_window_4096": list(FinitePixelRSLinear._fft_shape((400, 400), (4096, 4096)))},
                         "gain_w_inverse": FIXED_GAIN, "beta_rec": beta, "eta": eta_info["eta"]},
        model_identity={"encoder_layers": [2, 400, 400], "decoder_layers": [4, 400, 400],
                        "trainable_groups": ["encoder_phase", "decoder_phase"],
                        "ols": "gaussian_s12_100_ports_project_geometry"},
        optimizer_identity={"name": "Adam", "lr": 1.0e-3, "betas": [0.9, 0.999], "eps": 1.0e-8,
                            "objective_A": "beta_times_unclipped_MSE400",
                            "objective_B": "beta_times_unclipped_MSE400_plus_eta_times_unclipped_SSIM28"},
        sampler_identity={"logical_batch": LOGICAL_BATCH, "fit_rows_by_class": population["fit_rows_by_class"].tolist(),
                          "updates_per_branch": UPDATES},
        execution_source_semantics_id=manifest.execution_source_semantics_id,
        source_manifest_id=manifest.source_manifest_id,
    )
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "finite_pixel_independent_RS_passed": True,
                    "all_20_fixed_confirmation_rows_passed": True,
                    "full_chain_reference_q12_vs_q16_passed": True,
                    "actual_fft_embedding_invariance_passed": True,
                    "all_six_layer_gradient_and_direction_checks_passed": True,
                    "source_role_and_checkpoint_checks_passed": True,
                    "microbatch_equivalence_passed": True,
                    "target_mismatch_and_shuffle_negative_control_passed": True,
                    "engineering_adam_calls_total": 12,
                    "sigint_sigterm_independent_recovery_passed": True,
                    "official_test_reads": 0},
        resource={"passed": True, "cuda_device": resource["device_name"],
                  "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device),
                  "preflight_seconds": time.perf_counter() - started},
        implementation={"passed": True, "engine": f"rs_linear_q{selected_order}_4096",
                        "selected_engine": f"rs_linear_q{selected_order}_4096",
                        "trainable_groups": ["encoder_phase", "decoder_phase"],
                        "frozen_groups": ["OLS", "fixed_gain", "references", "evaluator"]},
        selected_engine=f"rs_linear_q{selected_order}_4096",
        resource_envelope={"cuda_device": resource["device_name"],
                           "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)},
    )
    control.mark_preflight_ready(receipt)
    elapsed = time.perf_counter() - started
    preflight = {"schema": "chen_lightgen_400_finite_pixel_preflight.v1", "status": "READY",
                 "task_id": TASK_ID, "authorization_id": AUTHORITY, "run_identity": identity.to_dict(),
                 "receipt": receipt.to_dict(), "inputs": {key: str(value) for key, value in paths.items()},
                 "propagation_backend": PROPAGATION_BACKEND,
                 "source_checkpoint_step": 30_000, "source_report_status": source_report["status"],
                 "historical_endpoint_updates": 500, "data_roles": {"fit": 100, "confirmation": 200,
                                                                          "both_canonical_train": True,
                                                                          "dev_reads": 0, "test_reads": 0},
                 "fixed_population_rows": {"fit": population["fit_rows"].tolist(),
                                           "confirmation": population["confirm_rows"].tolist()},
                 "source_bogt_beta_rec": beta, "fixed_gain_w_inverse": FIXED_GAIN,
                 "selected_quadrature_order": selected_order,
                 "propagation_grid": {"backend": PROPAGATION_BACKEND,
                    "window_cells": 4096, "window_m": 4096 * PIXEL_PITCH_M,
                    "physical_pixel_pitch_m": PIXEL_PITCH_M,
                    "quadrature_order_per_axis": selected_order,
                    "linear_convolution_fft_shapes": {
                        "train_active_400": list(FinitePixelRSLinear._fft_shape((400, 400), (400, 400))),
                        "measurement_roi_500": list(FinitePixelRSLinear._fft_shape((400, 400), (500, 500))),
                    "full_window_4096_base": list(FinitePixelRSLinear._fft_shape((400, 400), (4096, 4096))),
                    "window_power_tile_cells": 512, "window_power_source_chunk": 4,
                    "window_power_target_chunk": 4},
                    "full_window_observation_power_includes_all_gauss_nodes": True},
                 "plane_power_scope": "area-integrated RS observer intensity over the configured full physical window using spatial and node chunks; camera ROI power is reported separately",
                 "numerical_qualification": {"direct_rs": rs_check,
                    "production_boundary_checks": boundary_checks, "reference_convergence": reference_convergence,
                    "full_chain": full_chain,
                    "fft_embedding_checks": embedding_results[str(selected_order)],
                    "gradient_checks": gradient_results[str(selected_order)]},
                 "qualification_cache_import": cache_import,
                 "structure_coefficient": eta_info, "training_preflight": training_preflight,
                 "negative_control": negative_control,
                 "resources": {**resource, "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)},
                 "elapsed_seconds": elapsed, "training_updates": 0, "official_test_reads": 0}
    atomic_torch_save_new(output_dir / "fixed_population.pt", population)
    atomic_torch_save_new(output_dir / "start_model_state.pt", source_model.state_dict())
    atomic_torch_save_new(output_dir / "start_rng.pt", {
        "cpu": initial_rng_cpu,
        "cuda": initial_rng_cuda})
    atomic_write_new_json(output_dir / "preflight.json", baseline._json_safe(preflight))
    atomic_write_new_json(output_dir / "preflight_ready.json", baseline._json_safe(receipt.to_dict()))
    atomic_write_json(output_dir / "progress.json", {"status": "READY", "selected_order": selected_order,
                        "preflight_seconds": elapsed})
    return preflight


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run", "resume"))
    parser.add_argument("--propagation-backend", choices=("rs_linear",), required=True)
    for name in ("source-checkpoint", "source-report", "source-start-state",
                 "source-endpoint-checkpoint", "fixed-population", "config", "data", "evaluator",
                 "output-dir", "preflight-dir", "resume-checkpoint", "reuse-qualification-cache"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--worker-payload", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-ready", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_payload is not None:
        _require(args.worker_output is not None and args.worker_ready is not None,
                 "independent recovery worker requires explicit output and ready paths")
        _require(args.mode is None, "worker invocation cannot also select a research mode")
        return args
    _require(args.mode is not None, "--mode is required")
    required_paths = ("source_checkpoint", "source_report", "source_start_state",
                      "source_endpoint_checkpoint", "fixed_population", "config", "data", "evaluator",
                      "output_dir")
    for name in required_paths:
        _require(getattr(args, name) is not None, f"explicit --{name.replace('_', '-')} is required")
    if args.mode in ("run", "resume"):
        _require(args.preflight_dir is not None, "run/resume mode requires explicit --preflight-dir")
        _require(args.reuse_qualification_cache is None,
                 "--reuse-qualification-cache is valid only with --mode preflight")
    if args.mode == "resume":
        _require(args.resume_checkpoint is not None, "resume mode requires explicit --resume-checkpoint")
    else:
        _require(args.resume_checkpoint is None, "--resume-checkpoint is valid only with --mode resume")
    return args


def main() -> None:
    args = _parse_args()
    _require(args.propagation_backend == PROPAGATION_BACKEND,
             "this research task accepts only the explicitly selected RS linear-convolution backend")
    if args.worker_payload is not None:
        _worker_update(_resolve(args.worker_payload), _resolve(args.worker_output),
                       _resolve(args.worker_ready), None)
        return
    result = _preflight(args) if args.mode == "preflight" else _run_or_resume(args)
    print(json.dumps(baseline._json_safe(result), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()


