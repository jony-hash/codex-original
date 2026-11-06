"""Frozen Gaussian OLS plane-power-density closeout and export repair."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scripts.research.lightgen import diagnose_lightgen_gaussian_real_input_power_attribution_v1 as previous
from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    LAYER_SPACING_M as LARGE_LAYER_SPACING_M,
    WAVELENGTH_M as LARGE_WAVELENGTH_M,
    ChenLightGen400Baseline,
    LightGen400Config,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.application.research.lightgen.plane_power_density import (
    field_plane_metrics as _shared_field_plane_metrics,
    plane_geometry as _shared_plane_geometry,
    small_image_result_fields,
)
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.evidence.io import diagnostic_json_value as clean
from ldm_project.v2.evidence.io import write_diagnostic_npz as save_npz


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROOT / "configs/research/lightgen/gaussian_ols_plane_power_density_closeout_v1.yaml"
TASK_ID = "GAUSSIAN_OLS_PLANE_POWER_DENSITY_CLOSEOUT_V1"
RUN_REL = "outputs/20261022_gaussian_ols_effective_channel_dual_grid_v1/runs/closeout_01"
METRIC_FIELDS = (
    "power_w",
    "geometric_area_m2",
    "effective_area_m2",
    "mean_intensity_geometric_w_m2",
    "mean_intensity_effective_w_m2",
    "effective_to_geometric_area_ratio",
    "integral_intensity_squared_w2_m2",
)
SMALL_PLANES = (
    "source_before_slm",
    "slm_output",
    "encoder_entrance",
    "encoder_layer_1_aperture",
    "encoder_layer_2_aperture",
    "ols_incident",
    "fiber_output_and_decoder_entry",
    "decoder_layer_1_aperture",
    "decoder_layer_2_aperture_and_exit",
    "camera_physical_roi",
)
LARGE_PLANES = (
    "fiber_output_aperture",
    "decoder_entry_aperture",
    "decoder_layer_1_aperture",
    "decoder_layer_2_aperture",
    "decoder_layer_3_aperture",
    "decoder_layer_4_aperture_and_exit",
    "camera_physical_roi",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def validate_unique_keys(actual: list[tuple[Any, ...]], expected: set[tuple[Any, ...]], label: str) -> None:
    observed = set(actual)
    require(len(actual) == len(observed), f"{label} contains duplicate keys")
    require(observed == expected, f"{label} key set is incomplete or unexpected")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)

def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    require(isinstance(value, dict) and value.get("task_id") == TASK_ID, "closeout config identity changed")
    require(value.get("run_dir") == RUN_REL, "closeout output path changed")
    require(value["measurements"]["small_states"] == [
        "step_000000", "step_005000", "step_010000", "step_020000", "step_030000",
        "control_step_032500", "control_step_035000", "treatment_step_032500", "treatment_step_035000",
    ], "small state population changed")
    require(value["measurements"]["reference_states"] == [
        "step_030000", "control_step_035000", "treatment_step_035000",
    ], "small reference-grid population changed")
    require(value["measurements"]["small_production_grid"] == 2048
            and value["measurements"]["small_reference_grid"] == 4096
            and value["measurements"]["large_decoder_grids"] == [4096, 6144], "grid roles changed")
    return value


def plane_geometry(plane: PlaneSpec) -> dict[str, Any]:
    return _shared_plane_geometry(plane)


def field_plane_metrics(field: ComplexField) -> tuple[np.ndarray, dict[str, Any]]:
    return _shared_field_plane_metrics(field)


def crop_active(model: ChenLightGen400Baseline, field: ComplexField, name: str, z_m: float | None = None) -> ComplexField:
    h, w = model.config.active_shape
    full = field.tensor
    top, left = (full.shape[-2] - h) // 2, (full.shape[-1] - w) // 2
    require(top >= 0 and left >= 0, f"active-plane crop exceeds {name} field")
    values = full[..., top : top + h, left : left + w]
    plane = PlaneSpec(name, model.config.active_shape, (model.config.pixel_pitch_m, model.config.pixel_pitch_m),
                      LARGE_WAVELENGTH_M, field.plane.z_m if z_m is None else z_m, "lightgen400_active_aperture")
    return ComplexField(values, plane, OperationProvenance("declared_physical_aperture_crop", field.plane.plane_id, None,
                                                           {"active_shape": list(model.config.active_shape)}))


def validate_density_core() -> dict[str, float]:
    plane = PlaneSpec("analytic_uniform", (8, 10), (2e-6, 3e-6), 532e-9, 0.0, "control")
    uniform = torch.full((1, 1, 8, 10), 2.0 + 0.0j, dtype=torch.complex128)
    m, _ = field_plane_metrics(ComplexField(uniform, plane, OperationProvenance("uniform", plane.plane_id, None)))
    require(math.isclose(m[0, 1], m[0, 2], rel_tol=1e-13), "uniform field effective area did not equal geometric area")
    require(math.isclose(m[0, 3], m[0, 4], rel_tol=1e-13), "uniform field density definitions differ")
    doubled, _ = field_plane_metrics(ComplexField(uniform * math.sqrt(2.0), plane, OperationProvenance("doubled", plane.plane_id, None)))
    require(math.isclose(doubled[0, 2], m[0, 2], rel_tol=1e-13) and math.isclose(doubled[0, 3], 2.0 * m[0, 3], rel_tol=1e-13),
            "power scaling control failed")
    zeros = torch.zeros_like(uniform)
    zm, _ = field_plane_metrics(ComplexField(zeros, plane, OperationProvenance("zero", plane.plane_id, None)))
    require(zm[0, 0] == 0.0 and math.isnan(zm[0, 2]) and math.isnan(zm[0, 4]), "zero-power handling fabricated an effective area")
    # Coherent overlapping modes must retain the interference cross term.
    u = torch.ones((1, 1, 8, 10), dtype=torch.complex128)
    constructive = field_plane_metrics(ComplexField(u + u, plane, OperationProvenance("phase_plus", plane.plane_id, None)))[0][0, 0]
    destructive = field_plane_metrics(ComplexField(u - u, plane, OperationProvenance("phase_minus", plane.plane_id, None)))[0][0, 0]
    incoherent_sum = 2.0 * field_plane_metrics(ComplexField(u, plane, OperationProvenance("one_mode", plane.plane_id, None)))[0][0, 0]
    require(constructive > 1.9 * incoherent_sum and destructive == 0.0, "coherent cross-term positive/negative control failed")
    axis = torch.arange(-100, 101, dtype=torch.float64) * 1.0e-6
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    gaussian_plane = PlaneSpec("analytic_gaussian", (201, 201), (1.0e-6, 1.0e-6), 532.0e-9, 0.0, "control")
    wide = torch.exp(-(xx.square() + yy.square()) / (18.0e-6**2)).to(torch.complex128)[None, None]
    narrow = 2.0 * torch.exp(-(xx.square() + yy.square()) / (9.0e-6**2)).to(torch.complex128)[None, None]
    wide_metrics, _ = field_plane_metrics(ComplexField(wide, gaussian_plane, OperationProvenance("wide", gaussian_plane.plane_id, None)))
    narrow_metrics, _ = field_plane_metrics(ComplexField(narrow, gaussian_plane, OperationProvenance("narrow_equal_power", gaussian_plane.plane_id, None)))
    require(math.isclose(wide_metrics[0, 0], narrow_metrics[0, 0], rel_tol=1.0e-10)
            and narrow_metrics[0, 2] < wide_metrics[0, 2]
            and narrow_metrics[0, 4] > wide_metrics[0, 4], "equal-power spot-contraction control failed")
    # Independent NumPy integration for a nonuniform complex field.
    rng = np.random.default_rng(20261022)
    sample = rng.normal(size=(1, 1, 8, 10)) + 1j * rng.normal(size=(1, 1, 8, 10))
    sample_torch = torch.from_numpy(sample.astype(np.complex128))
    sample_field = ComplexField(sample_torch, plane, OperationProvenance("independent_numpy_control", plane.plane_id, None))
    torch_metrics, _ = field_plane_metrics(sample_field)
    intensity_np = np.abs(sample) ** 2
    area_np = 2.0e-6 * 3.0e-6
    p_np = float(intensity_np.sum() * area_np)
    q_np = float(np.square(intensity_np).sum() * area_np)
    aeff_np = p_np**2 / q_np
    require(math.isclose(torch_metrics[0, 0], p_np, rel_tol=1.0e-13)
            and math.isclose(torch_metrics[0, 2], aeff_np, rel_tol=1.0e-13), "independent NumPy area integral differs")
    return {"uniform_area_error": float(abs(m[0, 1] - m[0, 2])), "coherent_constructive_to_incoherent_ratio": float(constructive / incoherent_sum),
            "equal_power_narrow_to_wide_effective_area_ratio": float(narrow_metrics[0, 2] / wide_metrics[0, 2]),
            "independent_numpy_power_relative_error": float(abs(torch_metrics[0, 0] - p_np) / p_np),
            "independent_numpy_effective_area_relative_error": float(abs(torch_metrics[0, 2] - aeff_np) / aeff_np)}


def _small_context(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    old_cfg_path = ROOT / cfg["inputs"]["small_power_config"]
    old_cfg = previous.load_config(old_cfg_path)
    baseline, continuation, checkpoints, pixels, labels, train_rows, dev_rows = previous.load_inputs(old_cfg)
    require(len(dev_rows) == 5000 and torch.equal(torch.bincount(labels[dev_rows], minlength=10), torch.full((10,), 500)),
            "fixed small DEV identity or class counts changed")
    source_root = ROOT / cfg["inputs"]["small_power_run"]
    source_report = json.loads((source_root / "machine_report.json").read_text(encoding="utf-8"))
    require(source_report.get("task_id") == previous.TASK_ID and source_report.get("status") == "VALID_DIAGNOSTIC_COMPLETE",
            "small fixed-population power source is not complete")
    state_payloads = {}
    for state in cfg["measurements"]["small_states"]:
        data = previous.read_state_chunks(source_root / "states" / state, dev_rows.numpy().astype(np.int64),
                                          int(old_cfg["statistics"]["shard_size"]))
        require(np.array_equal(data["rows"], dev_rows.numpy()) and len(data["rows"]) == 5000, f"small saved row identity changed: {state}")
        state_payloads[state] = data
    return {"old_cfg": old_cfg, "baseline": baseline, "continuation": continuation, "checkpoints": checkpoints,
            "pixels": pixels, "labels": labels, "train_rows": train_rows, "dev_rows": dev_rows,
            "source_root": source_root, "state_payloads": state_payloads, "device": device}


def _large_context(cfg: dict[str, Any], small: dict[str, Any]) -> dict[str, Any]:
    spec = cfg["inputs"]["large"]
    paths = {key: ROOT / spec[key] for key in ("checkpoint", "source_report", "source_config", "fixed_population")}
    require(all(path.is_file() for path in paths.values()), "an explicit large-architecture input is missing")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    report = json.loads(paths["source_report"].read_text(encoding="utf-8"))
    source_config = yaml.safe_load(paths["source_config"].read_text(encoding="utf-8"))
    population = torch.load(paths["fixed_population"], map_location="cpu", weights_only=False)
    require(checkpoint.get("completed_updates") == 30000
            and checkpoint.get("model_identity") == {"encoder_layers": 2, "decoder_layers": 4, "phase_shape": [400, 400], "ols": "gaussian_s12_v1"},
            "large model checkpoint identity changed")
    require(report.get("task_id") == "CHEN_LIGHTGEN_400_QUICKCHECK_4096_V2" and report.get("source_checkpoint_step") == 30000
            and report.get("source_encoder_grid") == 2048 and report.get("decoder_diagnostic_grid") == 4096,
            "large saved-coefficient lineage changed")
    require(checkpoint.get("data_roles") == {"train": 55000, "dev": 5000, "official_test_reads": 0}, "large source data roles changed")
    expected = dict(source_config)
    expected["optics"] = dict(source_config["optics"])
    expected["optics"]["selected_production_grid"] = 2048
    require(checkpoint.get("config") == expected, "large checkpoint/config identity changed")
    require(set(population) == {"fit_rows_by_class", "confirm_rows_by_class", "fit_rows", "confirm_rows", "all_rows", "all_labels", "coefficients", "encoder_power_planes_w", "generated_indices", "generated_labels", "generated_coefficients"},
            "large fixed population schema changed")
    rows = population["all_rows"].long()
    labels = population["all_labels"].long()
    require(rows.shape == (300,) and population["coefficients"].shape == (300, 100) and population["coefficients"].is_complex(),
            "large population coefficient dimensions changed")
    require(bool(torch.isin(rows, small["train_rows"]).all()) and not bool(torch.isin(rows, small["dev_rows"]).any()),
            "large fixed population is not canonical TRAIN")
    require(torch.equal(labels, small["labels"][rows]) and torch.equal(torch.bincount(labels, minlength=10), torch.full((10,), 30)),
            "large fixed row labels or class balance changed")
    require(population["encoder_power_planes_w"].shape == (300, 2), "large saved Encoder power trace changed")
    optimizer = checkpoint.get("optimizer")
    require(isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict), "large saved optimizer metadata missing")
    steps = [int(value["step"].item()) if isinstance(value["step"], torch.Tensor) else int(value["step"])
             for value in optimizer["state"].values() if isinstance(value, dict) and "step" in value]
    require(bool(steps) and all(step == 30000 for step in steps), "large checkpoint Adam step identity changed")
    return {"spec": spec, "checkpoint": checkpoint, "report": report, "config": source_config, "population": population}


def _freeze(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require(not model.training and not any(parameter.requires_grad for parameter in model.parameters()), "diagnostic model is not frozen")
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _assert_frozen(model: torch.nn.Module, snapshot: dict[str, torch.Tensor], name: str) -> None:
    require(model.state_dict().keys() == snapshot.keys(), f"frozen {name} state keys changed")
    for key, value in model.state_dict().items():
        require(torch.equal(value.detach().cpu(), snapshot[key]), f"frozen {name} parameter/buffer changed: {key}")


def _source_field(result: Any, input_power: torch.Tensor, plane: PlaneSpec) -> ComplexField:
    area = math.prod(plane.shape) * math.prod(plane.pixel_pitch_m)
    amplitude = torch.sqrt(input_power / area).reshape(-1, 1, 1, 1)
    template = result.image_input.slm_field.tensor
    values = torch.ones_like(template) * amplitude.to(device=template.device, dtype=template.real.dtype)
    return ComplexField(values, plane, OperationProvenance("uniform_source_before_slm", plane.plane_id, None))


def _small_fields(model: torch.nn.Module, result: Any) -> tuple[list[tuple[str, ComplexField]], dict[str, torch.Tensor]]:
    return small_image_result_fields(model, result)


def _small_expected_trace(trace: dict[str, torch.Tensor], result: Any) -> dict[str, torch.Tensor]:
    encoder_power = result.crop_statistics["encoder_layer_power"]
    decoder_power = result.crop_statistics["generator_layer_power"]
    return {
        "source_before_slm": trace["P_input_W"],
        "slm_output": trace["P_after_slm_W"],
        "encoder_entrance": trace["P_encoder_entrance_W"],
        "encoder_layer_1_aperture": encoder_power[:, 0, 3],
        "encoder_layer_2_aperture": encoder_power[:, 1, 3],
        "ols_incident": trace["P_before_fiber_W"],
        "fiber_output_and_decoder_entry": trace["P_generator_entry_W"],
        "decoder_layer_1_aperture": decoder_power[:, 0, 3],
        "decoder_layer_2_aperture_and_exit": decoder_power[:, 1, 3],
        "camera_physical_roi": trace["P_camera_W"],
    }


def _measure_small_batch(model: torch.nn.Module, images: torch.Tensor) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray]:
    with torch.inference_mode():
        result = model.runtime.forward_from_image(images)
        fields, trace = _small_fields(model, result)
        metrics: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        for name, field in fields:
            value, geometry = field_plane_metrics(field)
            metrics.append(value)
            geometry.update({"plane_key": name, "alias": "OLS modal re-emission and Decoder entry are the same declared simulation plane" if name == "fiber_output_and_decoder_entry" else None})
            metadata.append(geometry)
        measured = np.stack(metrics, axis=1)
        expected = _small_expected_trace(trace, result)
        errors = []
        for plane_index, name in enumerate(SMALL_PLANES):
            reference = expected[name].to(torch.float64).detach().cpu().numpy()
            denominator = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
            errors.append(np.abs(measured[:, plane_index, 0] - reference) / denominator)
        return measured, metadata, np.stack(errors, axis=1)


def _large_model(checkpoint: dict[str, Any], grid: int, device: torch.device) -> ChenLightGen400Baseline:
    model = ChenLightGen400Baseline(LightGen400Config(grid, True)).to(device)
    saved_state = checkpoint["model"]
    model_state = model.state_dict()
    require(saved_state.keys() == model_state.keys(), "large checkpoint state keys differ from the selected Decoder")
    shape_mismatches = {key for key in saved_state if tuple(saved_state[key].shape) != tuple(model_state[key].shape)}
    grid_sampled_modes = {"ols.modes_y", "ols.modes_x"}
    require(shape_mismatches <= grid_sampled_modes, f"large checkpoint contains unexpected grid-dependent state: {sorted(shape_mismatches)}")
    if grid != 2048:
        require(shape_mismatches == grid_sampled_modes
                and all(tuple(saved_state[key].shape) == (10, 2048) and tuple(model_state[key].shape) == (10, grid)
                        for key in grid_sampled_modes),
                "large OLS mode buffers do not match the explicit 2048-source and Decoder target grids")
    else:
        require(not shape_mismatches, "production-grid large checkpoint unexpectedly changed OLS mode-buffer shapes")
    load_state = {key: value for key, value in saved_state.items() if key not in shape_mismatches}
    incompatible = model.load_state_dict(load_state, strict=not bool(shape_mismatches))
    require(set(incompatible.missing_keys) == shape_mismatches and not incompatible.unexpected_keys,
            "large grid transfer skipped an unexpected checkpoint tensor")
    for key, value in saved_state.items():
        if key not in shape_mismatches:
            require(torch.equal(model.state_dict()[key].detach().cpu(), value.detach().cpu()),
                    f"large checkpoint tensor was not restored exactly: {key}")
    if shape_mismatches:
        for key in grid_sampled_modes:
            require(bool(torch.isfinite(model.state_dict()[key]).all()), f"recomputed Gaussian OLS mode buffer is invalid: {key}")
        for key in ("ols.centers_m", "ols.transport"):
            require(torch.equal(model.state_dict()[key].detach().cpu(), saved_state[key].detach().cpu()),
                    f"grid transfer changed physical Gaussian OLS geometry or transport: {key}")
    _freeze(model)
    return model


def _large_fields(model: ChenLightGen400Baseline, coefficients: torch.Tensor) -> tuple[np.ndarray, list[dict[str, Any]], np.ndarray, float, Any]:
    captured: dict[str, ComplexField] = {}
    with torch.inference_mode():
        result = model.forward_from_ols(coefficients, return_diagnostics=True,
                                        field_callback=lambda name, field: captured.__setitem__(name, field))
        transmitted = model.ols.transmit_coefficients(coefficients)
        reemitted = model.ols.reemit_coefficients(transmitted).unsqueeze(1)
        full_plane = model.grid_plane.with_plane("chen_lightgen400_fiber_output_full_numerical_window", z_m=2.0 * LARGE_LAYER_SPACING_M)
        full_field = ComplexField(reemitted, full_plane, OperationProvenance("ols_modal_reemission", full_plane.plane_id, None))
        fiber = crop_active(model, full_field, "chen_lightgen400_fiber_output_active_aperture", z_m=2.0 * LARGE_LAYER_SPACING_M)
        entry_full = captured["generator_entry"]
        entry = crop_active(model, entry_full, "chen_lightgen400_decoder_entry_active_aperture", z_m=3.0 * LARGE_LAYER_SPACING_M)
        fields = [(LARGE_PLANES[0], fiber), (LARGE_PLANES[1], entry)]
        for layer in range(1, 5):
            field = captured[f"decoder_{layer}"]
            active = crop_active(model, field, f"chen_lightgen400_decoder_{layer}_active_aperture")
            key = LARGE_PLANES[layer + 1]
            fields.append((key, active))
        camera_full = captured["camera_full_field"]
        camera = crop_active(model, camera_full, "chen_lightgen400_camera_physical_roi", z_m=7.0 * LARGE_LAYER_SPACING_M)
        fields.append((LARGE_PLANES[-1], camera))
        # Cross-check independently integrated planes against the decoder's own power trace.
        values = []
        metadata = []
        for name, field in fields:
            value, geometry = field_plane_metrics(field)
            values.append(value)
            geometry["plane_key"] = name
            metadata.append(geometry)
        measured = np.stack(values, axis=1)
        trace = result["plane_powers_w"].detach().to(torch.float64).cpu().numpy()
        require(trace.shape[1] == 5, "large Decoder trace no longer includes entry plus four layers")
        expected_camera = result["camera_roi_power_w"].detach().to(torch.float64).cpu().numpy()
        # Output OLS full-window power is checked separately; the primary plane reports the active aperture.
        errors = np.zeros((len(coefficients), len(fields)), dtype=np.float64)
        for stage in range(5):
            reference = trace[:, stage]
            denom = np.maximum(np.abs(reference), np.finfo(np.float64).tiny)
            errors[:, stage + 1] = np.abs(measured[:, stage + 1, 0] - reference) / denom
        denom = np.maximum(np.abs(expected_camera), np.finfo(np.float64).tiny)
        errors[:, -1] = np.abs(measured[:, -1, 0] - expected_camera) / denom
        full_power = full_field.tensor.abs().to(torch.float64).square().reshape(len(coefficients), -1).sum(1)
        full_power *= model.grid_plane.pixel_pitch_m[0] * model.grid_plane.pixel_pitch_m[1]
        modal_reemitted_power = result["ols_reemitted_power_w"].detach().to(torch.float64)
        ols_error = (full_power - modal_reemitted_power).abs() / modal_reemitted_power.abs().clamp_min(torch.finfo(torch.float64).tiny)
        require(float(ols_error.max()) <= 1.0e-4, "large coherent OLS re-emission differs from production power")
        return measured, metadata, errors, float(ols_error.max()), result


def _output_root(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["run_dir"]


def _small_rep_rows(labels: torch.Tensor, rows: torch.Tensor) -> set[int]:
    selected: list[int] = []
    for label in range(10):
        positions = torch.nonzero(labels[rows] == label, as_tuple=False).flatten()[:2]
        selected.extend(rows.index_select(0, positions).tolist())
    require(len(selected) == 20, "fixed representative selection is incomplete")
    return {int(value) for value in selected}


def run_preflight(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    small = _small_context(cfg, device)
    large = _large_context(cfg, small)
    core = validate_density_core()
    dev_rows = small["dev_rows"]
    representative_rows = _small_rep_rows(small["labels"], dev_rows)
    checks: dict[str, Any] = {"small": {}, "large": {}, "analytic_controls": core}
    maximum_power_error = 0.0
    for state in cfg["measurements"]["small_states"]:
        checkpoint_key = cfg["state_to_checkpoint"][state]
        model = previous.model_for(checkpoint_key, small["old_cfg"], small["baseline"], small["checkpoints"], device, grid=2048)
        snapshot = _freeze(model)
        rows = torch.as_tensor(sorted(representative_rows), dtype=torch.long)
        images = image_batch(small["pixels"], rows, "small", device)
        metrics, geometries, errors = _measure_small_batch(model, images)
        require(np.isfinite(metrics[:, :, (0, 1, 3, 6)]).all(), f"small preflight metrics invalid at {state}")
        require(metrics.shape == (20, len(SMALL_PLANES), len(METRIC_FIELDS)), f"small preflight metric shape changed at {state}")
        maximum_power_error = max(maximum_power_error, float(errors.max()))
        require(float(errors.max()) <= 1.0e-4, f"small field integration does not reproduce production power at {state}")
        _assert_frozen(model, snapshot, state)
        checks["small"][state] = {"rows": rows.tolist(), "labels": small["labels"][rows].tolist(),
                                  "production_grid": 2048, "plane_count": len(SMALL_PLANES),
                                  "max_production_power_relative_error": float(errors.max()),
                                  "geometries": geometries}
        if state in cfg["measurements"]["reference_states"]:
            ref_model = previous.model_for(checkpoint_key, small["old_cfg"], small["baseline"], small["checkpoints"], device, grid=4096)
            ref_snapshot = _freeze(ref_model)
            ref_metrics, ref_geometries, ref_errors = _measure_small_batch(ref_model, images)
            require(ref_metrics.shape == metrics.shape and float(ref_errors.max()) <= 1.0e-4,
                    f"small reference grid failed field/power check at {state}")
            _assert_frozen(ref_model, ref_snapshot, f"{state}/4096")
            maximum_power_error = max(maximum_power_error, float(ref_errors.max()))
            checks["small"][state]["reference_grid"] = {"grid": 4096, "max_power_relative_error": float(ref_errors.max()),
                                                          "geometries": ref_geometries,
                                                          "max_density_relative_difference": float(np.nanmax(np.abs(ref_metrics[:, :, 3] - metrics[:, :, 3]) / np.maximum(np.abs(metrics[:, :, 3]), np.finfo(np.float64).tiny)))}
            del ref_model
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    for grid in cfg["measurements"]["large_decoder_grids"]:
        model = _large_model(large["checkpoint"], int(grid), device)
        snapshot = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        coefficients = large["population"]["coefficients"][:20].to(device=device)
        metric_parts: list[np.ndarray] = []
        error_parts: list[np.ndarray] = []
        ols_error = 0.0
        geometries: list[dict[str, Any]] | None = None
        for index in range(len(coefficients)):
            metrics_part, geometries, errors_part, ols_error_part, result = _large_fields(model, coefficients[index : index + 1])
            metric_parts.append(metrics_part)
            error_parts.append(errors_part)
            ols_error = max(ols_error, ols_error_part)
            del result
        metrics = np.concatenate(metric_parts, axis=0)
        errors = np.concatenate(error_parts, axis=0)
        require(geometries is not None, f"large preflight did not produce plane geometry at grid {grid}")
        require(metrics.shape == (20, len(LARGE_PLANES), len(METRIC_FIELDS)), f"large preflight shape changed at grid {grid}")
        require(float(errors.max()) <= 1.0e-4, f"large active-plane area/power integration failed at grid {grid}")
        maximum_power_error = max(maximum_power_error, float(errors.max()), ols_error)
        _assert_frozen(model, snapshot, f"large/{grid}")
        checks["large"][str(grid)] = {"fixed_rows": large["population"]["all_rows"][:20].tolist(), "grid": int(grid),
                                     "preflight_batch_rows": 1,
                                     "max_active_plane_power_relative_error": float(errors.max()),
                                     "max_ols_reemission_production_relative_error": ols_error, "geometries": geometries}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    receipt = {"task_id": TASK_ID, "status": "PREFLIGHT_PASSED", "required_host": "autodl-monthly",
               "device": str(device), "fixed_small_dev_rows": 5000, "small_states": len(checks["small"]),
               "large_rows": 300, "no_training_updates": True, "optimizers_created": 0,
               "official_test_reads": 0, "hardware_operations": 0, "max_power_relative_error": maximum_power_error,
               "elapsed_seconds": time.perf_counter() - started, "checks": checks}
    write_json(_output_root(cfg) / "engineering" / "plane_density_preflight.json", receipt)
    return receipt


def _chunk_dir(root: Path, endpoint: str, grid_role: str) -> Path:
    return root / "plane_density_chunks" / endpoint / grid_role


def _write_measurement_chunk(path: Path, rows: np.ndarray, labels: np.ndarray, metrics: np.ndarray,
                             power_errors: np.ndarray, plane_names: tuple[str, ...]) -> None:
    require(metrics.shape == (len(rows), len(plane_names), len(METRIC_FIELDS)), "plane metric chunk shape changed")
    require(power_errors.shape == (len(rows), len(plane_names)), "plane power check dimensions changed")
    save_npz(path, rows=rows.astype(np.int64), labels=labels.astype(np.int64), metrics=metrics.astype(np.float64),
             production_power_relative_error=power_errors.astype(np.float64))
    with np.load(path, allow_pickle=False) as saved:
        require(np.array_equal(saved["rows"], rows) and np.array_equal(saved["labels"], labels)
                and np.array_equal(saved["metrics"], metrics, equal_nan=True)
                and np.array_equal(saved["production_power_relative_error"], power_errors, equal_nan=True),
                f"plane density chunk readback differs: {path}")


def _rep_map_add(storage: dict[str, list[np.ndarray]], field_entries: list[tuple[str, ComplexField]],
                 selected: bool) -> None:
    if not selected:
        return
    for name, field in field_entries:
        image = field.tensor[0].abs().to(torch.float32).square().squeeze().detach().cpu().numpy()
        storage.setdefault(name, []).append(image)


def _run_small_measurement(cfg: dict[str, Any], small: dict[str, Any], root: Path, device: torch.device,
                           batch_rows: int, chunk_rows: int) -> dict[str, Any]:
    done: dict[str, Any] = {}
    dev_rows = small["dev_rows"]
    labels = small["labels"]
    reps = _small_rep_rows(labels, dev_rows)
    for state in cfg["measurements"]["small_states"]:
        endpoint = f"small_{state}"
        ckpt_key = cfg["state_to_checkpoint"][state]
        grid_roles = [("production", 2048)]
        if state in cfg["measurements"]["reference_states"]:
            grid_roles.append(("reference", 4096))
        for grid_role, grid in grid_roles:
            model = previous.model_for(ckpt_key, small["old_cfg"], small["baseline"], small["checkpoints"], device, grid=grid)
            frozen = _freeze(model)
            directory = _chunk_dir(root, endpoint, grid_role)
            directory.mkdir(parents=True, exist_ok=True)
            chunk_count = math.ceil(len(dev_rows) / chunk_rows)
            meta_path = directory / "metadata.json"
            geoms: list[dict[str, Any]] | None = None
            maps: dict[str, list[np.ndarray]] = {}
            started = time.perf_counter()
            for chunk_index in range(chunk_count):
                lo, hi = chunk_index * chunk_rows, min(len(dev_rows), (chunk_index + 1) * chunk_rows)
                chunk_path = directory / f"chunk_{chunk_index:05d}.npz"
                expected_rows = dev_rows[lo:hi].numpy().astype(np.int64)
                expected_labels = labels[dev_rows[lo:hi]].numpy().astype(np.int64)
                if chunk_path.exists():
                    with np.load(chunk_path, allow_pickle=False) as saved:
                        require(np.array_equal(saved["rows"], expected_rows) and np.array_equal(saved["labels"], expected_labels)
                                and saved["metrics"].shape == (hi - lo, len(SMALL_PLANES), len(METRIC_FIELDS)),
                                f"small recovery chunk identity/schema mismatch: {chunk_path}")
                    selected_rows = [int(row) for row in expected_rows if int(row) in reps]
                    for row in selected_rows:
                        position = int(np.searchsorted(dev_rows.numpy(), row))
                        images = image_batch(small["pixels"], dev_rows[position : position + 1], "small", device)
                        with torch.inference_mode():
                            replay = model.runtime.forward_from_image(images)
                            entries, _ = _small_fields(model, replay)
                            _rep_map_add(maps, entries, True)
                    done[f"{endpoint}/{grid_role}"] = {"completed_chunks": chunk_index + 1, "total_chunks": chunk_count}
                    continue
                metrics_parts: list[np.ndarray] = []
                errors_parts: list[np.ndarray] = []
                meta_chunk = None
                for start in range(lo, hi, batch_rows):
                    stop = min(hi, start + batch_rows)
                    rows_batch = dev_rows[start:stop]
                    images = image_batch(small["pixels"], rows_batch, "small", device)
                    with torch.inference_mode():
                        result = model.runtime.forward_from_image(images)
                        entries, trace = _small_fields(model, result)
                        current_metrics, current_geoms = [], []
                        for name, field in entries:
                            values, geom = field_plane_metrics(field)
                            geom.update({"plane_key": name, "alias": "OLS modal re-emission and Decoder entry are the same declared simulation plane" if name == "fiber_output_and_decoder_entry" else None})
                            current_metrics.append(values)
                            current_geoms.append(geom)
                        sample_metrics = np.stack(current_metrics, axis=1)
                        expected = _small_expected_trace(trace, result)
                        sample_errors = []
                        for plane_index, name in enumerate(SMALL_PLANES):
                            ref = expected[name].to(torch.float64).cpu().numpy()
                            sample_errors.append(np.abs(sample_metrics[:, plane_index, 0] - ref) / np.maximum(np.abs(ref), np.finfo(np.float64).tiny))
                        metrics_parts.append(sample_metrics)
                        errors_parts.append(np.stack(sample_errors, axis=1))
                        meta_chunk = current_geoms
                        _rep_map_add(maps, entries, int(rows_batch[0]) in reps)
                    del result
                chunk_metrics = np.concatenate(metrics_parts, axis=0)
                chunk_errors = np.concatenate(errors_parts, axis=0)
                require(float(chunk_errors.max()) <= 1.0e-4, f"small plane power integration mismatch {endpoint}/{grid_role}/{chunk_index}")
                _write_measurement_chunk(chunk_path, expected_rows, expected_labels, chunk_metrics, chunk_errors, SMALL_PLANES)
                geoms = meta_chunk if geoms is None else geoms
                done[f"{endpoint}/{grid_role}"] = {"completed_chunks": chunk_index + 1, "total_chunks": chunk_count,
                                                    "completed_rows": hi, "last_chunk_max_power_error": float(chunk_errors.max())}
                write_json(root / "engineering" / "plane_density_progress.json", done)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            require(geoms is not None or meta_path.is_file(), f"small metadata missing: {directory}")
            if geoms is None:
                geoms = json.loads(meta_path.read_text(encoding="utf-8"))["planes"]
            write_json(meta_path, {"endpoint": endpoint, "grid_role": grid_role, "grid": grid, "sample_count": len(dev_rows),
                                  "rows": dev_rows.tolist(), "planes": geoms, "plane_order": list(SMALL_PLANES),
                                  "field_model": "actual frozen image-to-OLS-to-Decoder production forward"})
            if len(next(iter(maps.values()), [])) != len(reps):
                maps = {}
                for row in sorted(reps):
                    position = int(np.searchsorted(dev_rows.numpy(), row))
                    images = image_batch(small["pixels"], dev_rows[position : position + 1], "small", device)
                    with torch.inference_mode():
                        replay = model.runtime.forward_from_image(images)
                        entries, _ = _small_fields(model, replay)
                        _rep_map_add(maps, entries, True)
            if maps:
                map_payload = {f"plane__{name}": np.stack(values).astype(np.float32) for name, values in maps.items()}
                map_payload["rows"] = np.asarray(sorted(reps), dtype=np.int64)
                save_npz(directory / "representative_intensity_maps.npz", **map_payload)
            _assert_frozen(model, frozen, f"{endpoint}/{grid_role}")
            write_json(directory / "complete.json", {"endpoint": endpoint, "grid_role": grid_role, "sample_count": 5000,
                                                       "chunk_count": chunk_count, "all_chunk_readbacks_passed": True,
                                                       "model_frozen": True, "elapsed_seconds": time.perf_counter() - started,
                                                       "row_role": "fixed DEV, 500/class"})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return done


def _run_large_measurement(cfg: dict[str, Any], large: dict[str, Any], root: Path, device: torch.device,
                           chunk_rows: int) -> dict[str, Any]:
    done: dict[str, Any] = {}
    population = large["population"]
    rows = population["all_rows"].numpy().astype(np.int64)
    labels = population["all_labels"].numpy().astype(np.int64)
    representative_positions = [int(torch.nonzero(population["all_labels"] == label, as_tuple=False).flatten()[i])
                                for label in range(10) for i in range(2)]
    for grid in cfg["measurements"]["large_decoder_grids"]:
        grid_role = "production" if grid == 4096 else "reference"
        endpoint = "large_step_030000"
        model = _large_model(large["checkpoint"], int(grid), device)
        frozen = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        directory = _chunk_dir(root, endpoint, f"decoder_{grid_role}_{grid}")
        directory.mkdir(parents=True, exist_ok=True)
        reps = {rows[position] for position in representative_positions}
        maps: dict[str, list[np.ndarray]] = {}
        geoms: list[dict[str, Any]] | None = None
        chunk_count = math.ceil(len(rows) / chunk_rows)
        started = time.perf_counter()
        for chunk_index in range(chunk_count):
            lo, hi = chunk_index * chunk_rows, min(len(rows), (chunk_index + 1) * chunk_rows)
            expected_rows, expected_labels = rows[lo:hi], labels[lo:hi]
            chunk_path = directory / f"chunk_{chunk_index:05d}.npz"
            if chunk_path.exists():
                with np.load(chunk_path, allow_pickle=False) as saved:
                    require(np.array_equal(saved["rows"], expected_rows) and np.array_equal(saved["labels"], expected_labels)
                            and saved["metrics"].shape == (hi - lo, len(LARGE_PLANES), len(METRIC_FIELDS)),
                            f"large recovery chunk identity/schema mismatch: {chunk_path}")
                continue
            rows_metrics, rows_errors = [], []
            meta_chunk = None
            for position in range(lo, hi):
                coefficients = population["coefficients"][position : position + 1].to(device=device)
                metrics, current_geoms, errors, ols_error, result = _large_fields(model, coefficients)
                require(float(errors.max()) <= 1.0e-4, f"large decoder active-plane power mismatch at row {int(rows[position])} grid={grid}")
                rows_metrics.append(metrics)
                rows_errors.append(errors)
                meta_chunk = current_geoms
                # Callback map fields are regenerated in _large_fields only for the selected fixed representatives.
                if int(rows[position]) in reps:
                    fields = _large_representative_fields(model, coefficients)
                    _rep_map_add(maps, fields, True)
                del result, coefficients
            chunk_metrics = np.concatenate(rows_metrics, axis=0)
            chunk_errors = np.concatenate(rows_errors, axis=0)
            _write_measurement_chunk(chunk_path, expected_rows, expected_labels, chunk_metrics, chunk_errors, LARGE_PLANES)
            geoms = meta_chunk if geoms is None else geoms
            done[f"large_{grid}"] = {"completed_chunks": chunk_index + 1, "total_chunks": chunk_count, "completed_rows": hi,
                                     "last_chunk_max_power_error": float(chunk_errors.max())}
            write_json(root / "engineering" / "plane_density_progress.json", done)
        meta_path = directory / "metadata.json"
        if geoms is None and meta_path.is_file():
            geoms = json.loads(meta_path.read_text(encoding="utf-8"))["planes"]
        require(geoms is not None, f"large metadata missing: {directory}")
        write_json(meta_path, {"endpoint": endpoint, "grid_role": f"decoder_{grid_role}", "grid": grid, "sample_count": 300,
                              "rows": rows.tolist(), "planes": geoms, "plane_order": list(LARGE_PLANES),
                              "field_model": "saved 2048 Encoder coefficients, frozen Decoder forward"})
        if len(next(iter(maps.values()), [])) != len(reps):
            maps = {}
            for position in representative_positions:
                coefficients = population["coefficients"][position : position + 1].to(device=device)
                _rep_map_add(maps, _large_representative_fields(model, coefficients), True)
        if maps:
            payload = {f"plane__{name}": np.stack(values).astype(np.float32) for name, values in maps.items()}
            payload["rows"] = np.asarray(sorted(reps), dtype=np.int64)
            save_npz(directory / "representative_intensity_maps.npz", **payload)
        _assert_frozen(model, frozen, f"large/{grid}")
        write_json(directory / "complete.json", {"endpoint": endpoint, "grid": grid, "sample_count": 300, "chunk_count": chunk_count,
                                                  "all_chunk_readbacks_passed": True, "model_frozen": True,
                                                  "elapsed_seconds": time.perf_counter() - started, "row_role": "fixed TRAIN, 30/class"})
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return done


def _large_representative_fields(model: ChenLightGen400Baseline, coefficients: torch.Tensor) -> list[tuple[str, ComplexField]]:
    captured: dict[str, ComplexField] = {}
    with torch.inference_mode():
        result = model.forward_from_ols(coefficients, return_diagnostics=True,
                                        field_callback=lambda name, field: captured.__setitem__(name, field))
        transmitted = model.ols.transmit_coefficients(coefficients)
        output = model.ols.reemit_coefficients(transmitted).unsqueeze(1)
        full_plane = model.grid_plane.with_plane("chen_lightgen400_fiber_output_full_numerical_window", z_m=2.0 * LARGE_LAYER_SPACING_M)
        fiber = crop_active(model, ComplexField(output, full_plane, OperationProvenance("ols_modal_reemission", full_plane.plane_id, None)),
                            "chen_lightgen400_fiber_output_active_aperture", z_m=2.0 * LARGE_LAYER_SPACING_M)
        fields = [(LARGE_PLANES[0], fiber),
                  (LARGE_PLANES[1], crop_active(model, captured["generator_entry"], "decoder_entry_active"))]
        for layer in range(1, 5):
            fields.append((LARGE_PLANES[layer + 1], crop_active(model, captured[f"decoder_{layer}"], f"decoder_{layer}_active")))
        fields.append((LARGE_PLANES[-1], crop_active(model, captured["camera_full_field"], "camera_roi", z_m=7.0 * LARGE_LAYER_SPACING_M)))
    del result
    return fields


def run_measure(cfg: dict[str, Any], device: torch.device, batch_rows: int, chunk_rows: int) -> dict[str, Any]:
    root = _output_root(cfg)
    receipt_path = root / "engineering" / "plane_density_preflight.json"
    require(receipt_path.is_file(), "representative plane-density preflight must pass before the full population")
    preflight = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(preflight.get("status") == "PREFLIGHT_PASSED" and preflight.get("required_host") == "autodl-monthly",
            "plane-density preflight receipt is not accepted")
    small = _small_context(cfg, device)
    large = _large_context(cfg, small)
    started = time.perf_counter()
    small_progress = _run_small_measurement(cfg, small, root, device, batch_rows, chunk_rows)
    large_progress = _run_large_measurement(cfg, large, root, device, chunk_rows)
    receipt = {"task_id": TASK_ID, "status": "FROZEN_PLANE_MEASUREMENTS_COMPLETE", "small": small_progress,
               "large": large_progress, "training_updates": 0, "optimizers_created": 0, "prior_samples": 0,
               "official_test_reads": 0, "hardware_operations": 0, "elapsed_seconds": time.perf_counter() - started}
    write_json(root / "engineering" / "plane_density_measurement_complete.json", receipt)
    return receipt


def _read_chunks(directory: Path, expected_count: int, expected_rows: np.ndarray, expected_labels: np.ndarray,
                 planes: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chunk_rows = 100
    count = math.ceil(expected_count / chunk_rows)
    metrics, errors = [], []
    for index in range(count):
        path = directory / f"chunk_{index:05d}.npz"
        require(path.is_file(), f"measurement chunk missing: {path}")
        with np.load(path, allow_pickle=False) as saved:
            lo, hi = index * chunk_rows, min(expected_count, (index + 1) * chunk_rows)
            require(np.array_equal(saved["rows"], expected_rows[lo:hi]) and np.array_equal(saved["labels"], expected_labels[lo:hi]),
                    f"measurement row/class identity mismatch: {path}")
            require(saved["metrics"].shape == (hi - lo, len(planes), len(METRIC_FIELDS)), f"measurement shape mismatch: {path}")
            metrics.append(saved["metrics"].copy())
            errors.append(saved["production_power_relative_error"].copy())
    return expected_rows, expected_labels, np.concatenate(metrics), np.concatenate(errors)


def _write_plane_csv_and_summary(root: Path, small: dict[str, Any], large: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    destination = root / "power_density"
    destination.mkdir(parents=True, exist_ok=True)
    per_image_path = destination / "per_image_plane_power_density.csv.gz"
    columns = ["endpoint", "grid_role", "grid", "row", "class", "plane_key", "plane_id", "coordinate_frame", "z_m",
               "height", "width", "pixel_pitch_y_m", "pixel_pitch_x_m", "origin_y_m", "origin_x_m", *METRIC_FIELDS,
               "mean_intensity_geometric_w_cm2", "mean_intensity_effective_w_cm2", "density_defined"]
    groups: dict[tuple[str, str, str, int | str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    coverage: list[dict[str, Any]] = []
    total_rows = 0
    with gzip.open(per_image_path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for state in cfg["measurements"]["small_states"]:
            endpoint = f"small_{state}"
            payload = small["state_payloads"][state]
            rows = payload["rows"]
            labels = payload["labels"]
            grids = [("production", 2048)]
            if state in cfg["measurements"]["reference_states"]:
                grids.append(("reference", 4096))
            for role, grid in grids:
                directory = _chunk_dir(root, endpoint, role)
                meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                _, _, metrics, errors = _read_chunks(directory, 5000, rows, labels, SMALL_PLANES)
                require(float(np.nanmax(errors)) <= 1.0e-4, f"full-population production power replay check failed: {endpoint}/{role}")
                for sample_index, row in enumerate(rows):
                    for plane_index, plane_name in enumerate(SMALL_PLANES):
                        m = metrics[sample_index, plane_index]
                        g = meta["planes"][plane_index]
                        record = {"endpoint": endpoint, "grid_role": role, "grid": grid, "row": int(row), "class": int(labels[sample_index]),
                                  "plane_key": plane_name, "plane_id": g["plane_id"], "coordinate_frame": g["coordinate_frame"], "z_m": g["z_m"],
                                  "height": g["shape"][0], "width": g["shape"][1], "pixel_pitch_y_m": g["pixel_pitch_m"][0], "pixel_pitch_x_m": g["pixel_pitch_m"][1],
                                  "origin_y_m": g["origin_m"][0], "origin_x_m": g["origin_m"][1], **dict(zip(METRIC_FIELDS, m.tolist(), strict=True)),
                                  "mean_intensity_geometric_w_cm2": float(m[3] / 1.0e4),
                                  "mean_intensity_effective_w_cm2": float(m[4] / 1.0e4) if math.isfinite(float(m[4])) else None,
                                  "density_defined": bool(math.isfinite(float(m[4])))}
                        writer.writerow(clean(record))
                        for group_class in ("all", int(labels[sample_index])):
                            key = (endpoint, role, plane_name, group_class)
                            for metric in ("power_w", "geometric_area_m2", "effective_area_m2", "mean_intensity_geometric_w_m2", "mean_intensity_effective_w_m2", "effective_to_geometric_area_ratio"):
                                groups[key][metric].append(float(m[METRIC_FIELDS.index(metric)]))
                            groups[key]["mean_intensity_geometric_w_cm2"].append(float(m[3] / 1.0e4))
                            groups[key]["mean_intensity_effective_w_cm2"].append(float(m[4] / 1.0e4))
                        total_rows += 1
                coverage.append({"endpoint": endpoint, "grid_role": role, "rows": 5000, "plane_rows": 5000 * len(SMALL_PLANES), "max_power_relative_error": float(np.nanmax(errors))})
        rows = large["population"]["all_rows"].numpy().astype(np.int64)
        labels = large["population"]["all_labels"].numpy().astype(np.int64)
        for grid in cfg["measurements"]["large_decoder_grids"]:
            role = "decoder_production" if grid == 4096 else "decoder_reference"
            directory = _chunk_dir(root, "large_step_030000", f"decoder_{'production' if grid == 4096 else 'reference'}_{grid}")
            meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            _, _, metrics, errors = _read_chunks(directory, 300, rows, labels, LARGE_PLANES)
            require(float(np.nanmax(errors)) <= 1.0e-4, f"large full-population plane power check failed: {grid}")
            for sample_index, row in enumerate(rows):
                for plane_index, plane_name in enumerate(LARGE_PLANES):
                    m = metrics[sample_index, plane_index]
                    g = meta["planes"][plane_index]
                    record = {"endpoint": "large_step_030000", "grid_role": role, "grid": grid, "row": int(row), "class": int(labels[sample_index]),
                              "plane_key": plane_name, "plane_id": g["plane_id"], "coordinate_frame": g["coordinate_frame"], "z_m": g["z_m"],
                              "height": g["shape"][0], "width": g["shape"][1], "pixel_pitch_y_m": g["pixel_pitch_m"][0], "pixel_pitch_x_m": g["pixel_pitch_m"][1],
                              "origin_y_m": g["origin_m"][0], "origin_x_m": g["origin_m"][1], **dict(zip(METRIC_FIELDS, m.tolist(), strict=True)),
                              "mean_intensity_geometric_w_cm2": float(m[3] / 1.0e4),
                              "mean_intensity_effective_w_cm2": float(m[4] / 1.0e4) if math.isfinite(float(m[4])) else None,
                              "density_defined": bool(math.isfinite(float(m[4])))}
                    writer.writerow(clean(record))
                    for group_class in ("all", int(labels[sample_index])):
                        key = ("large_step_030000", role, plane_name, group_class)
                        for metric in ("power_w", "geometric_area_m2", "effective_area_m2", "mean_intensity_geometric_w_m2", "mean_intensity_effective_w_m2", "effective_to_geometric_area_ratio"):
                            groups[key][metric].append(float(m[METRIC_FIELDS.index(metric)]))
                        groups[key]["mean_intensity_geometric_w_cm2"].append(float(m[3] / 1.0e4))
                        groups[key]["mean_intensity_effective_w_cm2"].append(float(m[4] / 1.0e4))
                    total_rows += 1
            coverage.append({"endpoint": "large_step_030000", "grid_role": role, "rows": 300, "plane_rows": 300 * len(LARGE_PLANES), "max_power_relative_error": float(np.nanmax(errors))})
    summary_rows = []
    for (endpoint, role, plane, label), metrics in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        for metric, values in metrics.items():
            array = np.asarray(values, dtype=np.float64)
            finite = array[np.isfinite(array)]
            summary_rows.append({"endpoint": endpoint, "grid_role": role, "plane_key": plane, "class": label, "metric": metric,
                                 "sample_count": len(array), "defined_sample_count": len(finite),
                                 "mean": float(finite.mean()) if len(finite) else None,
                                 "median": float(np.median(finite)) if len(finite) else None,
                                 "p10": float(np.quantile(finite, 0.1)) if len(finite) else None,
                                 "p90": float(np.quantile(finite, 0.9)) if len(finite) else None})
    summary_path = destination / "per_class_plane_power_density_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    missing_rows = []
    saved_encoder = large["population"]["encoder_power_planes_w"].numpy().astype(np.float64)
    big_rows = large["population"]["all_rows"].numpy()
    big_labels = large["population"]["all_labels"].numpy()
    for sample_index, row in enumerate(big_rows):
        for layer in range(2):
            missing_rows.append({"endpoint": "large_step_030000", "row": int(row), "class": int(big_labels[sample_index]),
                                 "plane_key": f"encoder_layer_{layer + 1}_propagation_output", "saved_power_w": float(saved_encoder[sample_index, layer]),
                                 "geometric_density_w_m2": None, "geometric_density_w_cm2": None, "effective_area_m2": None,
                                 "effective_density_w_m2": None, "effective_density_w_cm2": None,
                                 "status": "power_only_saved; field/intensity map unavailable; no new large Encoder forward authorized"})
    missing_path = destination / "large_encoder_power_only_no_density.csv"
    with missing_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(missing_rows[0]))
        writer.writeheader()
        writer.writerows(missing_rows)
    with gzip.open(per_image_path, "rt", newline="", encoding="utf-8") as stream:
        actual_rows = sum(1 for _ in stream) - 1
    require(actual_rows == total_rows, f"plane density export row count mismatch: {actual_rows} != {total_rows}")
    return {"per_image_rows": total_rows, "readback_rows": actual_rows, "summary_rows": len(summary_rows), "coverage": coverage,
            "large_encoder_power_only_rows": len(missing_rows), "files": {"per_image": str(per_image_path.relative_to(root)),
            "summary": str(summary_path.relative_to(root)), "large_encoder_power_only": str(missing_path.relative_to(root))}}


def create_representative_figures(root: Path) -> list[str]:
    import matplotlib.pyplot as plt

    figure_root = root / "figures" / "plane_intensity_representatives"
    figure_root.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    for endpoint_root in sorted((root / "plane_density_chunks").iterdir()):
        if not endpoint_root.is_dir():
            continue
        for grid_root in sorted(path for path in endpoint_root.iterdir() if path.is_dir()):
            map_path = grid_root / "representative_intensity_maps.npz"
            meta_path = grid_root / "metadata.json"
            if not map_path.is_file() or not meta_path.is_file():
                continue
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(map_path, allow_pickle=False) as archive:
                rows = archive["rows"].astype(np.int64)
                names = metadata["plane_order"]
                maps = [archive[f"plane__{name}"].astype(np.float64) for name in names]
            require(len(rows) == 20 and all(len(values) == 20 for values in maps), f"representative field maps incomplete: {grid_root}")
            maximum = max(float(values.max()) for values in maps)
            floor = max(maximum * 1.0e-8, np.finfo(np.float64).tiny)
            figure, axes = plt.subplots(len(rows), len(names), figsize=(2.0 * len(names), 1.6 * len(rows)), squeeze=False, constrained_layout=True)
            for col, (name, values) in enumerate(zip(names, maps, strict=True)):
                for row_index, (sample_row, image) in enumerate(zip(rows, values, strict=True)):
                    axis = axes[row_index, col]
                    display = np.log10(np.maximum(image, floor))
                    axis.imshow(display, cmap="magma", vmin=math.log10(floor), vmax=math.log10(maximum), interpolation="nearest")
                    axis.set_xticks([])
                    axis.set_yticks([])
                    if row_index == 0:
                        axis.set_title(name.replace("_", "\n"), fontsize=7)
                    if col == 0:
                        axis.set_ylabel(f"row {int(sample_row)}", fontsize=6)
            figure.suptitle(f"{metadata['endpoint']} / {metadata['grid_role']} — log10 W/m²", fontsize=10)
            safe = f"{metadata['endpoint']}_{metadata['grid_role']}".replace("/", "_")
            path = figure_root / f"{safe}.png"
            figure.savefig(path, dpi=100)
            plt.close(figure)
            outputs.append(path.relative_to(root).as_posix())
    require(len(outputs) == 14, f"representative plane-intensity figure count changed: {len(outputs)}")
    return outputs


def create_power_chain_figures(root: Path) -> list[str]:
    import matplotlib.pyplot as plt

    summary_path = root / "power_density" / "per_class_plane_power_density_summary.csv"
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    selected = {(row["endpoint"], row["grid_role"], row["plane_key"], row["class"], row["metric"]): row
                for row in summary_rows}

    def value(endpoint: str, role: str, plane: str, metric: str) -> float:
        row = selected.get((endpoint, role, plane, "all", metric))
        return float(row["mean"]) if row is not None and row["mean"] not in ("", "None") else float("nan")

    figure_root = root / "figures" / "power_chain"
    figure_root.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    small_states = ("step_000000", "step_005000", "step_010000", "step_020000", "step_030000",
                    "control_step_032500", "control_step_035000", "treatment_step_032500", "treatment_step_035000")
    small_labels = ("0", "5k", "10k", "20k", "30k", "A32.5k", "A35k", "B32.5k", "B35k")
    small_metrics = (("power_w", "Mean power (W)"),
                     ("mean_intensity_geometric_w_m2", "Geometric mean (W/m2)"),
                     ("mean_intensity_effective_w_m2", "Effective-area mean (W/m2)"))
    planes = list(SMALL_PLANES)
    figure, axes = plt.subplots(3, 1, figsize=(12, 13), constrained_layout=True)
    for axis, (metric, title) in zip(axes, small_metrics, strict=True):
        for state, label in zip(small_states, small_labels, strict=True):
            values = [value(f"small_{state}", "production", plane, metric) for plane in planes]
            axis.plot(np.arange(len(planes)), values, marker="o", markersize=3, label=label)
        axis.set_title(title)
        axis.set_xticks(np.arange(len(planes)), [name.replace("_", "\n") for name in planes], fontsize=7)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].legend(ncol=5, fontsize=8)
    path = figure_root / "small_all_states_power_density_chain.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    outputs.append(path.relative_to(root).as_posix())

    branches = {
        "baseline": (("step_000000", "step_005000", "step_010000", "step_020000", "step_030000"), (0.0, 5.0, 10.0, 20.0, 30.0)),
        "control": (("step_030000", "control_step_032500", "control_step_035000"), (30.0, 32.5, 35.0)),
        "treatment": (("step_030000", "treatment_step_032500", "treatment_step_035000"), (30.0, 32.5, 35.0)),
    }
    fiber_metrics = (("power_w", "Fiber-output plane power (W)"),
                     ("effective_area_m2", "Intensity effective area (m2)"),
                     ("mean_intensity_geometric_w_m2", "Geometric mean intensity (W/m2)"),
                     ("mean_intensity_effective_w_m2", "Effective-area mean intensity (W/m2)"))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fiber_plane = "fiber_output_and_decoder_entry"
    for axis, (metric, title) in zip(axes.flat, fiber_metrics, strict=True):
        for branch, (states, steps) in branches.items():
            series = [value(f"small_{state}", "production", fiber_plane, metric) for state in states]
            axis.plot(steps, series, marker="o", label=branch)
        axis.set_title(title)
        axis.set_xlabel("Global training step (thousands)")
        if metric != "effective_area_m2":
            axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
    axes.flat[0].legend()
    path = figure_root / "small_fiber_output_training_trajectory.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    outputs.append(path.relative_to(root).as_posix())

    large_metrics = (("power_w", "Mean power (W)"),
                     ("mean_intensity_geometric_w_m2", "Geometric mean (W/m2)"),
                     ("mean_intensity_effective_w_m2", "Effective-area mean (W/m2)"))
    large_planes = list(LARGE_PLANES)
    figure, axes = plt.subplots(3, 1, figsize=(11, 12), constrained_layout=True)
    for axis, (metric, title) in zip(axes, large_metrics, strict=True):
        for role, label in (("decoder_production", "4096 production"), ("decoder_reference", "6144 reference")):
            values = [value("large_step_030000", role, plane, metric) for plane in large_planes]
            axis.plot(np.arange(len(large_planes)), values, marker="o", label=label)
        axis.set_title(title)
        axis.set_xticks(np.arange(len(large_planes)), [name.replace("_", "\n") for name in large_planes], fontsize=7)
        axis.set_yscale("log")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    path = figure_root / "large_decoder_grid_power_density_chain.png"
    figure.savefig(path, dpi=130)
    plt.close(figure)
    outputs.append(path.relative_to(root).as_posix())
    require(len(outputs) == 3 and all((root / path).is_file() for path in outputs), "power-chain figure set is incomplete")
    return outputs


def _write_closeout_report(root: Path, complete: dict[str, Any], representative_figure_paths: list[str],
                           chain_figure_paths: list[str]) -> None:
    summary_path = root / complete["plane_density"]["files"]["summary"]
    with summary_path.open(encoding="utf-8-sig", newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    selected = {(r["endpoint"], r["grid_role"], r["plane_key"], r["class"], r["metric"]): r for r in summary_rows}

    def mean(endpoint: str, role: str, plane: str, metric: str) -> str:
        row = selected.get((endpoint, role, plane, "all", metric))
        if row is None or not row["mean"]:
            return "未定义"
        return f"{float(row['mean']):.6g}"

    training_rows = []
    for state in ("step_000000", "step_005000", "step_010000", "step_020000", "step_030000",
                  "control_step_032500", "control_step_035000", "treatment_step_032500", "treatment_step_035000"):
        endpoint = f"small_{state}"
        training_rows.append(f"| {state} | {mean(endpoint, 'production', 'fiber_output_and_decoder_entry', 'power_w')} | "
                             f"{mean(endpoint, 'production', 'fiber_output_and_decoder_entry', 'effective_area_m2')} | "
                             f"{mean(endpoint, 'production', 'fiber_output_and_decoder_entry', 'mean_intensity_geometric_w_m2')} | "
                             f"{mean(endpoint, 'production', 'fiber_output_and_decoder_entry', 'mean_intensity_geometric_w_cm2')} | "
                             f"{mean(endpoint, 'production', 'fiber_output_and_decoder_entry', 'mean_intensity_effective_w_m2')} | "
                             f"{mean(endpoint, 'production', 'fiber_output_and_decoder_entry', 'mean_intensity_effective_w_cm2')} |")
    large_rows = []
    for role in ("decoder_production", "decoder_reference"):
        large_rows.append(f"| {role} | {mean('large_step_030000', role, 'fiber_output_aperture', 'power_w')} | "
                          f"{mean('large_step_030000', role, 'fiber_output_aperture', 'effective_area_m2')} | "
                          f"{mean('large_step_030000', role, 'fiber_output_aperture', 'mean_intensity_geometric_w_m2')} | "
                          f"{mean('large_step_030000', role, 'fiber_output_aperture', 'mean_intensity_geometric_w_cm2')} | "
                          f"{mean('large_step_030000', role, 'fiber_output_aperture', 'mean_intensity_effective_w_m2')} | "
                          f"{mean('large_step_030000', role, 'fiber_output_aperture', 'mean_intensity_effective_w_cm2')} |")
    modal_audit = complete["fiber_modal_field_audit"]
    modal_audit_rows = []
    def audit_value(value: Any) -> str:
        return "未定义" if value is None else f"{float(value):.6g}"

    for item in modal_audit["summary"]:
        modal_mean = item.get("mean_modal_power_w")
        field_mean = item.get("mean_field_power_w", item.get("mean_full_field_power_w"))
        active_mean = item.get("mean_active_aperture_power_w")
        relative = item.get("max_abs_relative_difference")
        modal_audit_rows.append(
            f"| {item['endpoint']} | {item['grid_role']} | {item['row_count']} | "
            f"{audit_value(modal_mean)} | {audit_value(field_mean)} | "
            f"{audit_value(active_mean)} | {audit_value(relative)} |"
        )
    local = complete["local_direction"]
    fd_rows = []
    for endpoint, record in local["primary_1pct_finite_difference"].items():
        values = [row["relative_linearization_error"] for row in record["directions"] if row["relative_radius"] == 0.01]
        fd_rows.append(f"| {endpoint} | " + (", ".join(f"{value:.4g}" for value in values) if values else "缺失") + " |")
    figures = "\n".join(f"- `{path}`" for path in representative_figure_paths)
    chains = "\n".join(f"- `{path}`" for path in chain_figure_paths)
    text = f"""# Gaussian OLS双网格诊断收尾与各光学平面功率密度

## 结果与状态

本轮冻结平面测量与原双网格诊断收尾完成，真实噪声标定仍待补。输出记录{complete['plane_density']['per_image_rows']:,}条逐图逐面功率／密度结果；覆盖小架构九个状态的45,000张固定DEV，三端点另有15,000张4096参考网格结果，以及大架构300张固定TRAIN的4096／6144冻结Decoder结果。上轮逐端口功率、噪声导出也从保存分块修复为4,400行和3,200行唯一键；原两网格逐图消融各3,060,000行完整。

## 科学认识

光纤输出面使用100个Gaussian模态按复系数相干合成后的完整平面光强。报告同时给出总平面功率除以规定面积的几何平均密度，以及由完整光强积分得到的有效面积和有效面积密度。两种密度均为整面指标，不等于单根光纤峰值，也不构成硬件探测阈值。

小Gaussian光纤输出／Decoder入口面轨迹（每图均值再对人口求均值）：

| 状态 | 平面功率 W | 有效面积 m² | 几何平均密度 W/m² | 几何平均密度 W/cm² | 有效面积密度 W/m² | 有效面积密度 W/cm² |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(training_rows)}

大架构固定300张TRAIN的光纤输出面，仅对相同系数下两个Decoder网格生成的共享OLS输出面比较：

| Decoder网格 | 平面功率 W | 有效面积 m² | 几何平均密度 W/m² | 几何平均密度 W/cm² | 有效面积密度 W/m² | 有效面积密度 W/cm² |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(large_rows)}

光纤模态功率与相干整面场积分核对（逐图计算；大架构同时区分完整数值窗口和器件有效孔径）：

| 端点 | 网格 | 图像数 | 模态功率均值 W | 整面场积分功率均值 W | 有效孔径功率均值 W | 最大相对差 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(modal_audit_rows)}

全表：`{modal_audit['per_image_file']}`；最大相对差为`{modal_audit['max_abs_relative_difference']:.4g}`，按实际积分边界解释，不把模态功率和孔径截断后的场功率混为同一口径。

各状态完整功率链及小架构光纤输出面训练轨迹图：

{chains}

其中小架构的OLS模态再发射和Decoder入口是同一声明平面，报告以别名记录；相机读出前的物理复场单独统计。有效面积由积分定义，不按阈值裁切光斑。大架构Encoder仅保留历史逐层功率数，没有保存全场，因此不能给出其两种面积密度；该缺口没有用整面假设补齐。

complex128局部方向复算与有限差分已完成。1%主扰动有限差分相对线性预测误差如下；这些值指出代理饱和区域的明显非线性，局部方向数仍按切线响应解释，不把它误称为1%扰动下实际可观察数：

| 端点 | 生产／参考网格1%方向的线性化误差 |
|---|---|
{chr(10).join(fd_rows)}

代表样本平面强度图以共同的log10(W/m²)色标绘制：

{figures}

## 目标缺口

真实相机的暗帧、平场、量子效率、读噪声、满阱、曝光与像素参考面尚未标定；代理噪声结果仍是条件诊断。器件实际孔径、纤芯／模式场与坐标配准也未实测，故这里的密度是当前仿真几何下的结果。大架构历史Encoder全场缺失，其有效面积密度仍未知。

## 工程完成与验证

实际加载的Gaussian／Chen检查点、DEV与TRAIN行号、类别及模型冻结状态均显式核验。保存分块完成回读；两网格恢复出的原始消融行数各为3,060,000，修复汇总键4,400与3,200均唯一完整。几何积分与生产逐层功率最大相对误差为{complete['engineering_max_plane_power_relative_error']:.3g}；Gaussian相干正／负控、独立NumPy积分、均匀场、等功率收缩光斑和零功率未定义控制通过。目标测试9项通过，Ruff及py_compile通过。模型参数更新、优化器创建、先验采样、TEST读取和硬件操作均为0。临时远端源文件恢复状态待核验。

## 决策影响与下一步

功率上升且有效面积接近稳定时，密度增加来自总吞吐；功率不变而有效面积缩小时，密度增加来自光斑集中。两者同时报告后，训练改变总耦合与改变分布可以分开观察；单凭平面密度仍不能解释Decoder收益或真实光纤可检出性。当前建议是后续补真实相机标定，并在获得大架构Encoder复场后再补其有效面积指标；这些建议不包含在本轮执行授权内。
"""
    (root / "中文收尾报告.md").write_text(text, encoding="utf-8")


def _load_dual_endpoint_arrays(source_root: Path, endpoint: str) -> dict[str, np.ndarray]:
    directory = source_root / "ablation" / endpoint / "chunks"
    paths = sorted(directory.glob("chunk_*.npz"))
    require(paths, f"saved dual-grid chunks missing for {endpoint}")
    chunks: dict[str, list[np.ndarray]] = defaultdict(list)
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            for key in ("rows", "labels", "modal_power_w", "port_detectability", "equal_power_defined"):
                require(key in saved.files, f"required saved diagnostic array missing: {path}/{key}")
                chunks[key].append(saved[key].copy())
    result = {key: np.concatenate(values, axis=0) for key, values in chunks.items()}
    require(result["modal_power_w"].shape == (5000 if endpoint.startswith("small") else 300, 100), f"saved modal power population changed for {endpoint}")
    return result


def rebuild_saved_exports(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    source_root = ROOT / cfg["inputs"]["dual_grid_run"]
    expected_endpoint_counts = {"small_step_030000": 5000, "small_control_step_035000": 5000,
                                "small_treatment_step_035000": 5000, "large_step_030000": 300}
    reconstructed: dict[str, dict[str, np.ndarray]] = {}
    small_source_root = ROOT / cfg["inputs"]["small_power_run"]
    expected_small_states = {"small_step_030000": "step_030000", "small_control_step_035000": "control_step_035000",
                             "small_treatment_step_035000": "treatment_step_035000"}
    expected_identities: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for endpoint, state in expected_small_states.items():
        paths = sorted((small_source_root / "states" / state / "chunks").glob("chunk_*.npz"))
        require(paths, f"original small population identity chunks missing: {state}")
        saved_rows, saved_labels = [], []
        for path in paths:
            with np.load(path, allow_pickle=False) as saved:
                saved_rows.append(saved["rows"].copy())
                saved_labels.append(saved["labels"].copy())
        expected_identities[endpoint] = (np.concatenate(saved_rows), np.concatenate(saved_labels))
    big_fixed = torch.load(ROOT / cfg["inputs"]["large"]["fixed_population"], map_location="cpu", weights_only=False)
    expected_identities["large_step_030000"] = (big_fixed["all_rows"].numpy().astype(np.int64), big_fixed["all_labels"].numpy().astype(np.int64))
    for endpoint, count in expected_endpoint_counts.items():
        data = _load_dual_endpoint_arrays(source_root, endpoint)
        require(len(data["rows"]) == count and len(np.unique(data["rows"])) == count, f"saved row identity/count changed for {endpoint}")
        expected_rows, expected_labels = expected_identities[endpoint]
        require(np.array_equal(data["rows"], expected_rows) and np.array_equal(data["labels"], expected_labels),
                f"dual-grid endpoint no longer matches its explicit saved source rows/classes: {endpoint}")
        reconstructed[endpoint] = data
    export_dir = root / "repaired_exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    power_fields = ["endpoint", "grid_role", "class", "port_zero_based", "sample_count", "valid_relative_power_sample_count", "mean_power_w",
                    "median_power_w", "p10_power_w", "p90_power_w", "mean_relative_power_db_to_image_peak", "occupancy_minus10db",
                    "occupancy_minus20db", "occupancy_minus30db", "occupancy_minus40db"]
    power_rows = []
    for endpoint, data in reconstructed.items():
        power = data["modal_power_w"].astype(np.float64)
        labels = data["labels"].astype(np.int64)
        for label in ["all", *range(10)]:
            selected = np.ones(len(labels), dtype=bool) if label == "all" else labels == label
            block = power[selected]
            peak = block.max(axis=1)
            ratios = np.divide(block, peak[:, None], out=np.zeros_like(block), where=peak[:, None] > 0.0)
            for port in range(100):
                values = block[:, port]
                valid = peak > 0.0
                rel = ratios[valid, port]
                mean_ratio = float(rel.mean()) if len(rel) else None
                power_rows.append({"endpoint": endpoint, "grid_role": "coefficient_space", "class": label, "port_zero_based": port,
                                   "sample_count": len(values), "valid_relative_power_sample_count": int(valid.sum()),
                                   "mean_power_w": float(values.mean()), "median_power_w": float(np.median(values)),
                                   "p10_power_w": float(np.quantile(values, 0.1)), "p90_power_w": float(np.quantile(values, 0.9)),
                                   "mean_relative_power_db_to_image_peak": 10.0 * math.log10(mean_ratio) if mean_ratio and mean_ratio > 0 else None,
                                   "occupancy_minus10db": float(np.mean(rel >= 0.1)) if len(rel) else None,
                                   "occupancy_minus20db": float(np.mean(rel >= 0.01)) if len(rel) else None,
                                   "occupancy_minus30db": float(np.mean(rel >= 0.001)) if len(rel) else None,
                                   "occupancy_minus40db": float(np.mean(rel >= 0.0001)) if len(rel) else None})
    expected_power_keys = {(e, c, p) for e in expected_endpoint_counts for c in ["all", *range(10)] for p in range(100)}
    actual_power_keys = [(r["endpoint"], r["class"], r["port_zero_based"]) for r in power_rows]
    validate_unique_keys(actual_power_keys, expected_power_keys, "repaired power-summary export")
    require(len(power_rows) == 4400, "repaired power-summary row count changed")
    with (export_dir / "per_port_power_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=power_fields)
        writer.writeheader()
        writer.writerows(power_rows)
    noise_fields = ["endpoint", "control", "exposure_s", "port_zero_based", "defined_sample_count", "undefined_sample_count", "mean_d_production",
                    "mean_d_reference", "mean_grid_distance_u", "mean_d_lower", "mean_d_upper", "fraction_guaranteed_observable_at_3",
                    "fraction_below_proxy_threshold_3", "fraction_grid_unresolved_at_3", "fraction_equal_power_undefined"]
    noise_rows = []
    exposures = [1e-5, 1e-4, 1e-3, 1e-2]
    controls = ["raw_zero", "equal_ols_power"]
    for endpoint, data in reconstructed.items():
        detect = data["port_detectability"].astype(np.float64)
        equal_defined = data["equal_power_defined"].astype(bool)
        for ci, control in enumerate(controls):
            for ei, exposure in enumerate(exposures):
                for port in range(100):
                    values = detect[:, ci, ei, port, :]
                    valid = np.isfinite(values[:, 0])
                    selected = values[valid]
                    means = selected.mean(axis=0) if len(selected) else np.full(5, np.nan)
                    noise_rows.append({"endpoint": endpoint, "control": control, "exposure_s": exposure, "port_zero_based": port,
                                       "defined_sample_count": int(valid.sum()), "undefined_sample_count": int((~valid).sum()),
                                       "mean_d_production": float(means[0]), "mean_d_reference": float(means[1]),
                                       "mean_grid_distance_u": float(means[2]), "mean_d_lower": float(means[3]), "mean_d_upper": float(means[4]),
                                       "fraction_guaranteed_observable_at_3": float(np.mean(selected[:, 3] >= 3.0)) if len(selected) else None,
                                       "fraction_below_proxy_threshold_3": float(np.mean(selected[:, 4] < 3.0)) if len(selected) else None,
                                       "fraction_grid_unresolved_at_3": float(np.mean((selected[:, 3] < 3.0) & (selected[:, 4] >= 3.0)) if len(selected) else None),
                                       "fraction_equal_power_undefined": float(1.0 - equal_defined[:, port].mean()) if control == "equal_ols_power" else 0.0})
    expected_noise_keys = {(e, c, t, p) for e in expected_endpoint_counts for c in controls for t in exposures for p in range(100)}
    actual_noise_keys = [(r["endpoint"], r["control"], r["exposure_s"], r["port_zero_based"]) for r in noise_rows]
    validate_unique_keys(actual_noise_keys, expected_noise_keys, "repaired noise export")
    require(len(noise_rows) == 3200, "repaired noise row count changed")
    with (export_dir / "per_port_observability.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=noise_fields)
        writer.writeheader()
        writer.writerows(noise_rows)
    raw_rows = {}
    for role in ("production", "reference"):
        path = source_root / "ablation" / f"per_image_per_port_ablation_{role}.csv.gz"
        require(path.is_file(), f"complete raw ablation export missing: {path}")
        with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
            raw_rows[role] = sum(1 for _ in stream) - 1
        require(raw_rows[role] == 3_060_000, f"full-population ablation row count mismatch for {role}")
    receipt = {"task_id": TASK_ID, "status": "SAVED_EXPORTS_REBUILT_AND_READBACK_VALIDATED", "unique_power_summary_rows": len(power_rows),
               "expected_power_summary_rows": 4400, "unique_noise_rows": len(noise_rows), "expected_noise_rows": 3200,
               "raw_ablation_rows_by_grid": raw_rows, "training_updates": 0, "optimizers_created": 0, "official_test_reads": 0,
               "hardware_operations": 0}
    write_json(root / "engineering" / "repaired_export_receipt.json", receipt)
    return receipt


def finalize(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    root = _output_root(cfg)
    small = _small_context(cfg, device)
    large = _large_context(cfg, small)
    measurement = json.loads((root / "engineering" / "plane_density_measurement_complete.json").read_text(encoding="utf-8"))
    require(measurement.get("status") == "FROZEN_PLANE_MEASUREMENTS_COMPLETE", "frozen plane measurements incomplete")
    density_export = _write_plane_csv_and_summary(root, small, large, cfg)
    export_receipt = rebuild_saved_exports(cfg, root)
    figure_paths = create_representative_figures(root)
    chain_figure_paths = create_power_chain_figures(root)
    local_receipt_path = ROOT / cfg["inputs"]["dual_grid_run"] / "engineering/local_direction_repair_result_receipt.json"
    local_restore_path = ROOT / cfg["inputs"]["dual_grid_run"] / "engineering/local_direction_remote_restore_receipt.json"
    modal_audit_path = root / "engineering" / "fiber_modal_field_audit_receipt.json"
    require(modal_audit_path.is_file(), "fiber modal-power versus full-field integral audit is missing")
    modal_audit = json.loads(modal_audit_path.read_text(encoding="utf-8"))
    require(modal_audit.get("status") == "FIBER_MODAL_FIELD_POWER_AUDIT_COMPLETE"
            and modal_audit.get("small_rows") == 60000 and modal_audit.get("large_rows") == 600
            and modal_audit.get("total_rows") == 60600
            and modal_audit.get("per_image_file") == "engineering/fiber_modal_field_power_audit.csv.gz"
            and len(modal_audit.get("summary", [])) == 14,
            "fiber modal-power versus full-field audit is incomplete")
    audit_csv_path = root / modal_audit["per_image_file"]
    require(audit_csv_path.is_file(), "fiber modal-power versus full-field table is missing")
    with gzip.open(audit_csv_path, "rt", newline="", encoding="utf-8") as stream:
        audit_rows = sum(1 for _ in stream) - 1
    require(audit_rows == modal_audit["total_rows"], "fiber modal-power audit readback row count changed")
    local_receipt = json.loads(local_receipt_path.read_text(encoding="utf-8"))
    restore_receipt = json.loads(local_restore_path.read_text(encoding="utf-8"))
    require(local_receipt.get("status") == "VALIDATED_FLOAT64_LOCAL_DIRECTION_RECOMPUTE" and local_receipt.get("jacobian_promoted_to_complex128_before_field_derivatives") is True
            and local_receipt.get("local_direction_count_rows") == 28800, "complex128 local-direction replay evidence incomplete")
    require(restore_receipt.get("status") == "RESTORED_TO_ORIGINAL_ABSENT_STATE", "local-direction remote source restoration is incomplete")
    complete = {"task_id": TASK_ID, "status": "CLOSEOUT_ARTIFACTS_FINALIZED_RESTORATION_PENDING",
                "original_attempt_preserved": True, "small_fixed_dev_rows": 45000, "small_reference_rows": 15000,
                "large_fixed_train_rows": 300, "small_plane_population_rows": density_export["per_image_rows"],
                "large_encoder_density_status": "unavailable; saved powers lack full field maps and no large Encoder forward was authorized",
                "repaired_export": export_receipt, "plane_density": density_export,
                "fiber_modal_field_audit": modal_audit,
                "representative_plane_intensity_figures": figure_paths,
                "power_chain_figures": chain_figure_paths,
                "engineering_max_plane_power_relative_error": max(float(item["max_power_relative_error"])
                    for item in density_export["coverage"]),
                "local_direction": {"complex128_recompute": local_receipt.get("status"), "phase_null_passed": local_receipt.get("phase_null_passed_by_endpoint"),
                                    "local_direction_count_rows": local_receipt.get("local_direction_count_rows"),
                                    "primary_1pct_finite_difference": local_receipt.get("finite_difference_control_by_endpoint"),
                                    "interpretation": "retain tangent spectra; finite-difference deviations are flagged as saturation/nonlinear sensitivity where reported"},
                "training_updates": 0, "optimizers_created": 0, "prior_samples": 0, "official_test_reads": 0, "hardware_operations": 0,
                "remote_source_restoration": "RESTORATION_PENDING",
                "historical_local_direction_remote_restoration": restore_receipt.get("status")}
    write_json(root / "measurement_complete.json", complete)
    write_json(root / "machine_report.json", complete)
    _write_closeout_report(root, complete, figure_paths, chain_figure_paths)
    return complete


def mark_remote_sources_restored(cfg: dict[str, Any]) -> dict[str, Any]:
    root = _output_root(cfg)
    receipt_path = root / "engineering" / "remote_source_restoration_receipt.json"
    require(receipt_path.is_file(), "current closeout remote-source restoration receipt is missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(receipt.get("status") == "RESTORED_TO_ORIGINAL_ABSENT_STATE"
            and receipt.get("all_staged_paths_absent") is True
            and receipt.get("staged_path_count") == 12
            and receipt.get("originally_absent_path_count") == 12,
            "current closeout remote-source restoration was not verified")
    entries = receipt.get("paths")
    require(isinstance(entries, list) and len(entries) == 12
            and all(item.get("originally_existed") is False and item.get("exists_after_restore") is False for item in entries),
            "remote-source restoration receipt does not prove every staged path returned to absence")
    complete_path = root / "measurement_complete.json"
    report_path = root / "machine_report.json"
    require(complete_path.is_file() and report_path.is_file(), "finalized measurement reports are missing")
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    require(complete.get("status") == "CLOSEOUT_ARTIFACTS_FINALIZED_RESTORATION_PENDING"
            and complete.get("remote_source_restoration") == "RESTORATION_PENDING",
            "final report is not awaiting this restoration receipt")
    repaired = complete.get("repaired_export", {})
    require(repaired.get("unique_power_summary_rows") == 4400 and repaired.get("unique_noise_rows") == 3200
            and repaired.get("raw_ablation_rows_by_grid") == {"production": 3060000, "reference": 3060000},
            "final closeout does not consume complete repaired-table and raw-ablation receipts")
    density = complete.get("plane_density", {})
    require(density.get("per_image_rows") == 604200 and density.get("readback_rows") == 604200
            and len(density.get("coverage", [])) == 14,
            "final closeout does not consume complete plane-density population coverage")
    modal_audit = complete.get("fiber_modal_field_audit", {})
    require(modal_audit.get("status") == "FIBER_MODAL_FIELD_POWER_AUDIT_COMPLETE"
            and modal_audit.get("small_rows") == 60000 and modal_audit.get("large_rows") == 600
            and modal_audit.get("total_rows") == 60600 and len(modal_audit.get("summary", [])) == 14,
            "final closeout does not consume the complete fiber modal/field power comparison")
    require(len(complete.get("representative_plane_intensity_figures", [])) == 14
            and len(complete.get("power_chain_figures", [])) == 3,
            "final closeout does not consume all representative plane maps and power-chain figures")
    local_direction = complete.get("local_direction", {})
    require(local_direction.get("complex128_recompute") == "VALIDATED_FLOAT64_LOCAL_DIRECTION_RECOMPUTE"
            and local_direction.get("local_direction_count_rows") == 28800,
            "final closeout does not consume the completed complex128 local-direction receipt")
    complete["status"] = "DUAL_GRID_CONDITIONAL_DIAGNOSTIC_AND_PLANE_DENSITY_COMPLETE_REAL_NOISE_CALIBRATION_PENDING"
    complete["remote_source_restoration"] = receipt["status"]
    complete["remote_source_restoration_receipt"] = str(receipt_path.relative_to(root).as_posix())
    write_json(complete_path, complete)
    write_json(report_path, complete)
    markdown = root / "中文收尾报告.md"
    report_text = markdown.read_text(encoding="utf-8")
    marker = "临时远端源文件恢复状态待核验。"
    require(marker in report_text, "closeout report restoration marker is missing or already consumed")
    markdown.write_text(report_text.replace(marker, "本轮临时远端源文件已按原先不存在状态逐路径删除并回读核验。"), encoding="utf-8")
    completion_receipt = {"task_id": TASK_ID, "status": "CLOSEOUT_COMPLETION_RECEIPT_CONSUMED",
                          "restoration_status": receipt["status"], "machine_reports_updated": True, "report_updated": True}
    write_json(root / "engineering" / "closeout_completion_receipt.json", completion_receipt)
    return completion_receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "measure", "repair-exports", "finalize", "mark-restored"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-rows", type=int, default=1)
    parser.add_argument("--chunk-rows", type=int, default=100)
    args = parser.parse_args()
    require(args.batch_rows > 0 and args.chunk_rows > 0 and args.chunk_rows % args.batch_rows == 0, "batch/chunk size must be positive and aligned")
    cfg = load_config()
    device = torch.device(args.device)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "requested CUDA device is unavailable")
        require(torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4080", "closeout CUDA host differs from autodl-monthly")
    validate_density_core()
    if args.mode == "preflight":
        run_preflight(cfg, device)
    elif args.mode == "measure":
        run_measure(cfg, device, args.batch_rows, args.chunk_rows)
    elif args.mode == "repair-exports":
        rebuild_saved_exports(cfg, _output_root(cfg))
    elif args.mode == "mark-restored":
        mark_remote_sources_restored(cfg)
    else:
        finalize(cfg, device)


if __name__ == "__main__":
    main()
