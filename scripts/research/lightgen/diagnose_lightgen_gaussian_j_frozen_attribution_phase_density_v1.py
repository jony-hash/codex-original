"""Frozen J attribution, whole-plane density, G4 compatibility, and phase response."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base
import scripts.research.lightgen.execute_lightgen_small_decoder_native_gradient_ab_v1 as decoder_ref
import scripts.research.lightgen.execute_lightgen_small_joint_update_readability_pilot_v1 as joint
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.application.research.lightgen.image_readout import resize_view, ssim_batch
from ldm_project.v2.application.research.lightgen.plane_power_density import (
    field_intensity_distribution, field_plane_metrics, plane_geometry, small_image_result_fields,
)
from ldm_project.v2.application.research.shared.research_run_control import (
    PreflightReceipt, ResearchRunControl, ResearchRunIdentity,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.evidence import atomic_write_json


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "GAUSSIAN_J_FROZEN_ATTRIBUTION_PHASE_DENSITY_V1"
AUTHORITY = "USER_APPROVED_GAUSSIAN_J_FROZEN_ATTRIBUTION_PHASE_DENSITY_20261023"
GRIDS = (("production", 2048), ("reference", 4096))
ENDPOINTS = ("D050000", "J50000")
DECODER_PLANES = (
    "fiber_output_and_decoder_entry", "decoder_layer_1_aperture",
    "decoder_layer_2_aperture_and_exit", "camera_physical_roi",
)
DEV_PLANES = (
    "source_before_slm", "slm_output", "encoder_entrance", "encoder_layer_1_aperture",
    "encoder_layer_2_aperture", "ols_incident", *DECODER_PLANES,
)
METRIC_FIELDS = (
    "power_w", "geometric_area_m2", "effective_area_m2",
    "mean_intensity_geometric_w_m2", "mean_intensity_effective_w_m2",
    "effective_to_geometric_area_ratio", "integral_intensity_squared_w2_m2",
)
DIST_FIELDS = (
    "maximum_intensity_w_m2", "area_weighted_q50_w_m2", "area_weighted_q90_w_m2",
    "area_weighted_q99_w_m2", "power_weighted_q50_w_m2", "power_weighted_q90_w_m2",
    "power_weighted_q99_w_m2",
)
REC_FIELDS = ("mse", "psnr_db", "ssim", "clipping_fraction", "camera_power_w")
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 2026102301
CHUNK_ROWS = 200
BATCH_ROWS = 10


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _resolve(path: str | Path, root: Path = ROOT) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict) and value.get("schema") == "gaussian_j_frozen_attribution_phase_density.v1"
            and value.get("task_id") == TASK_ID and value.get("semantic_authority_id") == AUTHORITY,
            "J frozen attribution config identity changed")
    require(value["run_dir"] == "outputs/20261023_gaussian_j_frozen_attribution_phase_density_v1/runs/attempt_01"
            and value["grids"] == {"production": 2048, "reference": 4096},
            "registered output path or grids changed")
    require(value["population"] == {"split_seed": 1234, "dev_count": 5000, "dev_per_class": 500,
                                    "g4_count": 10000, "g4_per_class": 1000, "official_test_reads": 0},
            "fixed diagnostic populations changed")
    require(value["statistics"] == {"bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                                    "bootstrap_seed": BOOTSTRAP_SEED},
            "paired bootstrap contract changed")
    require(value["phase"] == {"relative_rms_rad": [0.01, 0.03, 0.1, 0.3],
                               "families": ["independent_port", "linear_ramp"],
                               "repeats_per_class": 8, "seed": 2026102303,
                               "representative_per_class": 20},
            "phase sensitivity scope changed")
    execution = value["execution"]
    require(execution["frozen_only"] is True and all(execution[key] == 0 for key in (
        "scientific_updates", "optimizer_creation", "prior_fitting", "prior_sampling",
        "official_test_reads", "material_deployment"))
        and execution["batch_rows"] == BATCH_ROWS and execution["chunk_rows"] == CHUNK_ROWS,
        "zero-update boundary or registered chunking changed")
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def _write_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)
    with np.load(path, allow_pickle=False) as saved:
        require(saved.files == list(payload), f"array readback key changed: {path}")
        for key, value in payload.items():
            expected = np.asarray(value)
            actual = saved[key]
            equal = np.array_equal(actual, expected, equal_nan=True) if actual.dtype.kind in "fc" else np.array_equal(
                actual, expected,
            )
            require(actual.dtype == expected.dtype and equal, f"array readback changed: {path}:{key}")


def _load_fixed(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    inp = cfg["inputs"]
    task_cfg = joint._read_config(_resolve(inp["joint_task_config"]))
    optical_dir = _resolve(inp["joint_optical_dir"])
    control_dir = _resolve(inp["control_evidence_dir"])
    g4_path = _resolve(inp["g4_coefficients"])
    source_checkpoint_path = _resolve(task_cfg["source"]["checkpoint"])
    registered_cohort_path = _resolve(task_cfg["generation"]["cohort"])
    fixed = joint._load_fixed_inputs(
        task_cfg, source_checkpoint_path, control_dir, registered_cohort_path,
        _resolve(inp["readability_reference_dir"]),
    )
    pixels, labels, train_rows, dev_rows = base._load_data(_resolve(inp["data_file"]))
    require(len(train_rows) == 55_000 and len(dev_rows) == 5_000
            and torch.equal(torch.bincount(labels[dev_rows], minlength=10), torch.full((10,), 500))
            and not bool(torch.isin(train_rows, dev_rows).any()), "fixed DEV split or data roles changed")
    checkpoints = {"D050000": fixed["control_checkpoint"]}
    for name, updates in zip(inp["joint_checkpoint_names"], (0, 2500, 5000), strict=True):
        cp = torch.load(optical_dir / name, map_location="cpu", weights_only=False)
        require(cp.get("schema") == joint.CHECKPOINT_SCHEMA and cp.get("task_id") == joint.TASK_ID
                and cp.get("completed_updates") == updates
                and cp.get("completed_global_step") == joint.SOURCE_STEP + updates
                and cp.get("model_identity") == fixed["source"]["checkpoint"]["model_identity"]
                and cp.get("propagation_grid_size") == 2048,
                f"J checkpoint identity changed: {name}")
        joint._validate_checkpoint(cp, cp["run_id"], fixed["source"], task_cfg, cp["schedule"])
        adam_steps = [int(torch.as_tensor(state["step"]).item()) for state in cp["optimizer"]["state"].values()]
        require(len(adam_steps) == 2 and adam_steps == [joint.SOURCE_STEP + updates] * 2,
                f"saved J Adam counters changed: {name}")
        checkpoints[f"J{joint.SOURCE_STEP + updates}"] = cp
    cp0, d0 = checkpoints["J45000"]["model"], checkpoints["D050000"]["model"]
    require(cp0.keys() == d0.keys() and all(torch.equal(cp0[k], d0[k]) for k in cp0
            if k != "runtime.generator.phase"),
            "J45k and D0 do not share the verified C source state and encoder")
    require(checkpoints["J45000"]["source_run_id"] == checkpoints["D050000"]["source_run_id"]
            == fixed["source"]["checkpoint"]["run_id"], "J and D0 source lineage differs")
    g4 = torch.load(g4_path, map_location="cpu", weights_only=False)
    old = fixed["control_generation"]
    require(torch.equal(g4["ols_coefficients"], old["ols_coefficients"])
            and torch.equal(g4["proposal_ids"], old["proposal_ids"])
            and torch.equal(g4["requested_labels"], old["requested_labels"]),
            "old G4 coefficients, labels, or proposal IDs differ from the D0 cohort")
    require(g4["ols_coefficients"].shape == (10_000, 100) and g4["ols_coefficients"].is_complex()
            and torch.equal(torch.bincount(g4["requested_labels"].long(), minlength=10),
                            torch.full((10,), 1000)), "old G4 cohort is incomplete")
    source = fixed["source"]["checkpoint"]
    return {
        "task_config": task_cfg, "checkpoints": checkpoints, "pixels": pixels, "labels": labels,
        "train_rows": train_rows, "dev_rows": dev_rows, "device": device, "g4": g4, "g4_path": g4_path,
        "baseline_cfg": fixed["previous_config"], "source": fixed["source"], "control_dir": control_dir,
        "camera_gain": source["camera_gain"].to(device), "scale": source["scale"].to(device),
        "power_reference": source["power_reference"].to(device), "weights": source["weights"],
        "structural_weight": source["structural_weight"], "config": cfg,
    }


def _model(key: str, grid: int, fixed: dict[str, Any]) -> torch.nn.Module:
    model = base._model("small", grid, fixed["device"], fixed["baseline_cfg"])
    model.load_state_dict(fixed["checkpoints"][key]["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require(not model.training and not any(parameter.requires_grad for parameter in model.parameters()),
            f"model is not frozen: {key}/{grid}")
    return model


def _external_ols_state(model: torch.nn.Module, coefficients: torch.Tensor) -> base.ComplexOLSState:
    return base.ComplexOLSState(
        coefficients=coefficients, source="matched_empirical_prior",
        basis_id=model.runtime.ols.basis_id, mode_geometry=model.runtime.ols.mode_identity(),
    )


def _decoder_fields(model: torch.nn.Module, result: Any) -> list[tuple[str, ComplexField]]:
    camera = model.runtime.output_relay(result.generator_output_field)
    return [(DECODER_PLANES[0], result.generator_input_field),
            (DECODER_PLANES[1], result.generator_records[0]),
            (DECODER_PLANES[2], result.generator_records[1]),
            (DECODER_PLANES[3], camera)]


def _decoder_fields_from_coefficients(
    model: torch.nn.Module, coefficients: torch.Tensor,
) -> list[tuple[str, ComplexField]]:
    """Expose production generated-decoder planes and verify its camera intensity."""
    state = _external_ols_state(model, coefficients)
    generator_input = model.runtime.ols.write_to_generator_input(
        state, model.runtime.generator_entrance_plane,
    )
    generator_output, records, _statistics = model.runtime.generator(
        generator_input, "matched_generated_generator",
    )
    require(len(records) == 2, "generated decoder layer record count changed")
    generator_output = ComplexField(
        generator_output.tensor, model.runtime.generator_exit_plane,
        OperationProvenance("lightgen_generator_exit", generator_output.plane.plane_id, None,
                            {"generator_layers": 2}),
    )
    camera = model.runtime.output_relay(generator_output)
    fields = [(DECODER_PLANES[0], generator_input), (DECODER_PLANES[1], records[0]),
              (DECODER_PLANES[2], records[1]), (DECODER_PLANES[3], camera)]
    production = model.forward_from_ols(coefficients)["physical_w_per_pixel"]
    area = math.prod(float(value) for value in camera.plane.pixel_pitch_m)
    direct = camera.tensor.abs().square() * area
    difference = torch.linalg.vector_norm((production-direct).to(torch.float64))
    reference = torch.linalg.vector_norm(production.to(torch.float64))
    relative = float(difference / reference) if float(reference) > 0.0 else float(difference)
    require(math.isfinite(relative) and relative <= 1e-6,
            f"generated decoder field path differs from production camera intensity: {relative}")
    return fields


def _measure_fields(fields: list[tuple[str, ComplexField]]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    require(tuple(name for name, _ in fields) == DECODER_PLANES, "decoder plane order changed")
    metrics, geometry, distribution = [], [], None
    for i, (name, field) in enumerate(fields):
        values, geo = field_plane_metrics(field)
        metrics.append(np.asarray(values, dtype=np.float64))
        geometry.append({"plane_key": name, **geo})
        if i == 0:
            distribution = field_intensity_distribution(field)
    require(distribution is not None, "OLS output intensity distribution is missing")
    dist = np.asarray([[float(row.get(key, math.nan)) for key in DIST_FIELDS] for row in distribution],
                      dtype=np.float64)
    return np.stack(metrics, axis=1), dist, geometry


def _plane_specs(geometry: list[dict[str, Any]]) -> dict[str, PlaneSpec]:
    return {item["plane_key"]: PlaneSpec(
        plane_id=item["plane_id"], shape=tuple(item["shape"]),
        pixel_pitch_m=tuple(item["pixel_pitch_m"]), wavelength_m=float(item["wavelength_m"]),
        z_m=float(item["z_m"]), coordinate_frame=item["coordinate_frame"],
        origin_m=tuple(item["origin_m"]),
    ) for item in geometry}


def _build_basis(key: str, grid: int, root: Path, fixed: dict[str, Any]
                 ) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    path = root / "bases" / f"{key}_{grid}_decoder_planes.npz"
    if key == "D050000":
        old = _resolve(fixed["config"]["inputs"]["reusable_d0_bases"]) / f"D050000_{grid}_decoder_planes.npz"
        source = path if path.is_file() else old
        require(source.is_file(), f"reusable D0 basis is missing: {source}")
    else:
        source = path
    if source.is_file():
        with np.load(source, allow_pickle=False) as saved:
            require(str(saved["endpoint"].item()) == key and int(saved["grid"].item()) == grid
                    and int(saved["global_step"].item()) == 50_000, f"saved Decoder basis identity changed: {source}")
            arrays = {name: saved[f"field_{i}"].copy() for i, name in enumerate(DECODER_PLANES)}
            geometry = json.loads(str(saved["geometry_json"].item()))
    else:
        model = _model(key, grid, fixed)
        pieces = {name: [] for name in DECODER_PLANES}
        geometry = None
        with torch.inference_mode():
            for start in range(0, 100, 8):
                ports = torch.arange(start, min(100, start + 8), device=fixed["device"])
                onehot = torch.zeros((len(ports), 100), dtype=torch.complex64, device=fixed["device"])
                onehot[torch.arange(len(ports), device=fixed["device"]), ports] = 1.0 + 0j
                fields = _decoder_fields_from_coefficients(model, onehot)
                current = [{"plane_key": name, **plane_geometry(field.plane),
                            "wavelength_m": float(field.plane.wavelength_m)} for name, field in fields]
                geometry = current if geometry is None else geometry
                for name, field in fields:
                    pieces[name].append(field.tensor.detach().to(torch.complex64).cpu().numpy()[:, 0])
        arrays = {name: np.concatenate(pieces[name], axis=0).astype(np.complex64) for name in DECODER_PLANES}
        require(geometry is not None, "new J Decoder basis geometry is missing")
        _write_npz(path, endpoint=np.asarray(key), grid=np.asarray(grid, dtype=np.int64),
                   global_step=np.asarray(50_000, dtype=np.int64),
                   **{f"field_{i}": arrays[name] for i, name in enumerate(DECODER_PLANES)},
                   geometry_json=np.asarray(json.dumps(geometry, ensure_ascii=False)))
        del model
    require(all(arrays[name].shape[0] == 100 and arrays[name].dtype == np.complex64
                and np.isfinite(arrays[name]).all() for name in DECODER_PLANES),
            f"Decoder basis is malformed: {key}/{grid}")
    if key == "D050000" and not path.exists():
        _write_npz(path, endpoint=np.asarray(key), grid=np.asarray(grid, dtype=np.int64),
                   global_step=np.asarray(50_000, dtype=np.int64),
                   **{f"field_{i}": arrays[name] for i, name in enumerate(DECODER_PLANES)},
                   geometry_json=np.asarray(json.dumps(geometry, ensure_ascii=False)))
    return arrays, geometry


def _synthesize(key: str, coefficients: np.ndarray, basis: dict[str, np.ndarray],
                geometry: list[dict[str, Any]], device: torch.device
                ) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    metrics = np.empty((len(coefficients), 4, 7), dtype=np.float64)
    dist = np.empty((len(coefficients), len(DIST_FIELDS)), dtype=np.float64)
    specs = _plane_specs(geometry)
    basis_t = {name: torch.as_tensor(basis[name], dtype=torch.complex64, device=device) for name in DECODER_PLANES}
    coeff = torch.as_tensor(coefficients, dtype=torch.complex64, device=device)
    camera_parts = []
    with torch.inference_mode():
        for start in range(0, len(coefficients), BATCH_ROWS):
            stop = min(len(coefficients), start + BATCH_ROWS)
            z = coeff[start:stop]
            for pi, name in enumerate(DECODER_PLANES):
                tensor = torch.einsum("bi,ihw->bhw", z, basis_t[name]).unsqueeze(1)
                field = ComplexField(tensor, specs[name],
                    OperationProvenance("coherent_saved_port_response_sum", specs[name].plane_id, None))
                value, _ = field_plane_metrics(field)
                metrics[start:stop, pi] = np.asarray(value, dtype=np.float64)
                if pi == 0:
                    distribution = field_intensity_distribution(field)
                    dist[start:stop] = np.asarray(
                        [[float(row.get(k, math.nan)) for k in DIST_FIELDS] for row in distribution],
                        dtype=np.float64)
                if pi == 3:
                    camera_parts.append(tensor[:, 0].detach().clone())
    camera = torch.cat(camera_parts, dim=0)
    require(np.isfinite(metrics[:, :, (0, 1, 3, 6)]).all(), f"nonfinite synthesis: {key}")
    return metrics, dist, camera


def _camera_readout(camera: torch.Tensor, geometry: dict[str, Any], gain: float,
                    pixels: torch.Tensor | None = None, rows: np.ndarray | None = None,
                    intensity_scale: torch.Tensor | None = None) -> tuple[np.ndarray, torch.Tensor]:
    area = math.prod(float(value) for value in geometry["pixel_pitch_m"])
    raw = camera.abs().square().unsqueeze(1) * area
    power = raw.double().sum((1, 2, 3))
    if intensity_scale is not None:
        scale = intensity_scale.to(device=raw.device, dtype=raw.dtype).reshape(-1, 1, 1, 1)
        raw = raw * scale
        power = power * scale.reshape(-1).double()
    gained = raw * float(gain)
    view = resize_view(gained)
    clipping = ((gained < 0) | (gained > 1)).double().mean((1, 2, 3))
    if pixels is None:
        nan = torch.full_like(power, math.nan, dtype=torch.float64)
        values = torch.stack((nan, nan, nan, clipping.double(), power), dim=1)
        return values.cpu().numpy(), view
    require(rows is not None, "camera target rows are missing")
    target = pixels.index_select(0, torch.as_tensor(rows)).unsqueeze(1).to(
        device=view.device, dtype=view.dtype) / 255.0
    mse = (view - target).square().mean((1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    ssim = ssim_batch(view, target)
    return torch.stack((mse, psnr, ssim, clipping, power.to(mse.dtype)), dim=1).cpu().numpy(), view


def _equal_camera_readout(camera: torch.Tensor, geometry: dict[str, Any], gain: float,
                          target_power: np.ndarray, pixels: torch.Tensor | None,
                          rows: np.ndarray | None) -> tuple[np.ndarray, torch.Tensor, np.ndarray]:
    """Match total physical camera power per image; zero-source ratios stay undefined."""
    require(torch.is_complex(camera),
            "equal-camera-power matching requires a complex camera field, not an intensity frame")
    target = torch.as_tensor(target_power, device=camera.device, dtype=torch.float64).reshape(-1)
    require(len(target) == len(camera) and bool(torch.isfinite(target).all())
            and bool((target >= 0).all()), "camera-power targets must be finite, nonnegative, and row-aligned")
    area = math.prod(float(value) for value in geometry["pixel_pitch_m"])
    current = camera.abs().square().double().sum((1, 2)) * area
    valid_t = torch.isfinite(target) & (target >= 0) & (current > 0)
    valid = valid_t.cpu().numpy()
    require(rows is None or len(rows) == len(camera), "camera-power target row count differs from fields")
    row_count = len(camera)
    values = np.full((row_count, len(REC_FIELDS)), np.nan, dtype=np.float64)
    view = torch.full((row_count, 1, 28, 28), math.nan, device=camera.device, dtype=torch.float32)
    if bool(valid_t.any()):
        scale = (target[valid_t] / current[valid_t]).to(torch.float32)
        subset_camera = camera[valid_t]
        if pixels is None:
            output, resized = _camera_readout(subset_camera, geometry, gain, intensity_scale=scale)
        else:
            require(rows is not None, "equal-power DEV readout requires sample rows")
            output, resized = _camera_readout(subset_camera, geometry, gain, pixels, rows[valid], scale)
        values[valid] = output
        view[valid_t] = resized
    return values, view, valid


def _cache_dir(root: Path, endpoint: str, role: str) -> Path:
    return root / "populations" / "dev" / endpoint / role


def _load_chunks(directory: Path, rows: np.ndarray, labels: torch.Tensor) -> dict[str, np.ndarray]:
    paths = sorted(directory.glob("chunk_*.npz"))
    require(len(paths) == math.ceil(len(rows) / CHUNK_ROWS), f"population chunks are incomplete: {directory}")
    arrays: dict[str, list[np.ndarray]] = {}
    for index, path in enumerate(paths):
        expected = rows[index * CHUNK_ROWS:min(len(rows), (index + 1) * CHUNK_ROWS)]
        expected_labels = labels[torch.as_tensor(expected)].numpy()
        with np.load(path, allow_pickle=False) as saved:
            require(np.array_equal(saved["rows"], expected) and np.array_equal(saved["labels"], expected_labels),
                    f"cached row or label mismatch: {path}")
            require(saved["metrics"].shape == (len(expected), len(DEV_PLANES), 7)
                    and saved["distribution"].shape == (len(expected), len(DIST_FIELDS))
                    and saved["coefficients"].shape == (len(expected), 100)
                    and saved["coefficients"].dtype == np.complex64
                    and saved["reconstruction"].shape == (len(expected), len(REC_FIELDS))
                    and saved["equal_camera_power_reconstruction"].shape == (len(expected), len(REC_FIELDS)),
                    f"cached DEV payload shape or dtype changed: {path}")
            for key in saved.files:
                arrays.setdefault(key, []).append(saved[key].copy())
    return {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}


def _save_chunk(path: Path, rows: np.ndarray, labels: np.ndarray, **values: np.ndarray) -> None:
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            require(np.array_equal(saved["rows"], rows) and np.array_equal(saved["labels"], labels),
                    f"recovery identity changed: {path}")
            for key, value in values.items():
                require(key in saved.files and np.array_equal(saved[key], value, equal_nan=True),
                        f"recovery payload changed: {path}:{key}")
        return
    _write_npz(path, rows=rows, labels=labels, **values)


def _paired_bootstrap(values: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    require(len(values) == len(labels) and np.isfinite(values).all(), "bootstrap values are incomplete")
    samples = values[indices].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(x) for x in np.quantile(samples, [0.025, 0.975])],
            "n": len(values), "bootstrap_resamples": int(indices.shape[0]), "bootstrap_seed": BOOTSTRAP_SEED}


def _bootstrap_indices(labels: np.ndarray, seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == label) for label in range(10)]
    require(all(len(group) for group in groups), "paired population misses a class")
    output = np.empty((BOOTSTRAP_RESAMPLES, len(labels)), dtype=np.int64)
    for i in range(BOOTSTRAP_RESAMPLES):
        output[i] = np.concatenate([group[rng.integers(0, len(group), len(group))] for group in groups])
    return output


def _coherent_replay(fixed: dict[str, Any], bases: dict[tuple[str, int], tuple[dict[str, np.ndarray], list[dict[str, Any]]]]
                     ) -> dict[str, Any]:
    selected = [int(x) for label in range(10)
                for x in fixed["dev_rows"][fixed["labels"].index_select(0, fixed["dev_rows"]) == label][:2]]
    rows = np.asarray(selected, dtype=np.int64)
    images = image_batch(fixed["pixels"], torch.as_tensor(rows), "small", fixed["device"])
    coefficients: dict[tuple[str, int], np.ndarray] = {}
    direct_encoder_metrics: dict[tuple[str, int], np.ndarray] = {}
    cache_checks = []
    dev_rows_all = fixed["dev_rows"].numpy().astype(np.int64)
    cache_positions = np.asarray([int(np.flatnonzero(dev_rows_all == row)[0]) for row in rows], dtype=np.int64)
    for role, grid in GRIDS:
        for encoder in ENDPOINTS:
            model = _model(encoder, grid, fixed)
            with torch.inference_mode():
                result = model.runtime.forward_from_image(images)
                coeff = result.z_ols.coefficients.detach().cpu().numpy().astype(np.complex64)
                fields, _trace = small_image_result_fields(model, result)
                plane_values = []
                for _name, field in fields:
                    value, _geo = field_plane_metrics(field)
                    plane_values.append(np.asarray(value, dtype=np.float64))
            coefficients[(encoder, grid)] = coeff
            direct_encoder_metrics[(encoder, grid)] = np.stack(plane_values, axis=1)
            if encoder == "D050000":
                source = _resolve(fixed["config"]["inputs"]["reusable_density_run"])
                source = source / "populations" / "dev" / encoder / role
                meta_path = source / "metadata.json"
                require(meta_path.is_file(), f"reusable D0 DEV cache metadata is missing: {meta_path}")
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                require(meta.get("endpoint") == "D050000" and meta.get("global_step") == 50_000
                        and meta.get("grid") == grid and meta.get("grid_role") == role
                        and meta.get("sample_count") == 5_000 and meta.get("plane_order") == list(DEV_PLANES),
                        f"reusable D0 DEV cache metadata identity changed: {meta_path}")
                cached = _load_chunks(source, dev_rows_all, fixed["labels"])
                direct_coeff = coeff.astype(np.complex128)
                cached_coeff = cached["coefficients"][cache_positions].astype(np.complex128)
                coeff_den = float(np.linalg.norm(direct_coeff))
                coeff_rel = float(np.linalg.norm(direct_coeff-cached_coeff) / coeff_den) if coeff_den else None
                direct_metric = direct_encoder_metrics[(encoder, grid)]
                cached_metric = cached["metrics"][cache_positions]
                direct_power = direct_metric[:, :, 0]
                cached_power = cached_metric[:, :, 0]
                power_rel = float(np.max(np.abs(direct_power-cached_power)
                                         / np.maximum(np.abs(direct_power), np.finfo(np.float64).tiny)))
                raw = result.physical_sensor_output_w_per_pixel
                gain = float(fixed["checkpoints"][encoder]["camera_gain"].reshape(-1)[0])
                view = resize_view(raw * gain)
                target = fixed["pixels"].index_select(0, torch.as_tensor(rows)).unsqueeze(1).to(
                    device=view.device, dtype=view.dtype) / 255.0
                mse = (view-target).square().mean((1, 2, 3))
                psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
                ssim = ssim_batch(view, target)
                clipping = ((raw * gain < 0) | (raw * gain > 1)).double().mean((1, 2, 3))
                camera_power = result.physical_power_trace["P_camera_W"].to(torch.float64)
                direct_rec = torch.stack((mse, psnr, ssim, clipping, camera_power), dim=1).cpu().numpy()
                cached_rec = cached["reconstruction"][cache_positions]
                readout_errors = np.max(np.abs(direct_rec-cached_rec), axis=0)
                mse_abs = float(readout_errors[0])
                require(coeff_rel is not None and coeff_rel <= 1e-4 and power_rel <= 1e-4
                        and mse_abs <= 1e-6,
                        f"D0 reusable cache failed same-grid direct replay: {role}, coeff={coeff_rel}, power={power_rel}, MSE={mse_abs}")
                cache_checks.append({"grid": grid, "grid_role": role,
                    "cache": str(source), "rows": len(rows), "coefficient_relative_error": coeff_rel,
                    "plane_power_relative_error_max": power_rel,
                    "MSE_absolute_error_max": mse_abs,
                    "readout_metric_absolute_error_by_name": dict(zip(REC_FIELDS, readout_errors.tolist(), strict=True)),
                    "status": "PASSED"})
            del model
    records = []
    for role, grid in GRIDS:
        for decoder in ENDPOINTS:
            model = _model(decoder, grid, fixed)
            basis, geometry = bases[(decoder, grid)]
            basis_t = {name: torch.as_tensor(basis[name], device=fixed["device"]) for name in DECODER_PLANES}
            maximum = 0.0
            maximum_mse = 0.0
            camera_geo = next(item for item in geometry if item["plane_key"] == DECODER_PLANES[3])
            gain = float(fixed["checkpoints"][decoder]["camera_gain"].reshape(-1)[0])
            for encoder in ENDPOINTS:
                z = torch.as_tensor(coefficients[(encoder, grid)], dtype=torch.complex64, device=fixed["device"])
                for start in range(0, len(rows), BATCH_ROWS):
                    stop = min(len(rows), start + BATCH_ROWS)
                    block = z[start:stop]
                    direct_fields = _decoder_fields_from_coefficients(model, block)
                    direct_metrics = []
                    direct_camera = None
                    for name, field in direct_fields:
                        value, _geo = field_plane_metrics(field)
                        direct_metrics.append(np.asarray(value, dtype=np.float64))
                        predicted = torch.einsum("bi,ihw->bhw", block, basis_t[name]).unsqueeze(1)
                        diff = torch.linalg.vector_norm((predicted-field.tensor).to(torch.complex128))
                        denom = torch.linalg.vector_norm(field.tensor.to(torch.complex128))
                        if float(denom) == 0.0:
                            require(float(diff) <= 1e-12, "zero field produced a nonzero synthesized field")
                            rel = 0.0
                        else:
                            rel = float(diff / denom)
                        maximum = max(maximum, rel)
                        require(math.isfinite(rel) and rel <= 1e-4,
                                f"direct/synth complex-field check failed: {decoder}/{encoder}/{grid}/{name}")
                        if name == DECODER_PLANES[3]:
                            direct_camera = field.tensor[:, 0]
                    synth_metrics, _dist, synth_camera = _synthesize(
                        decoder, block.detach().cpu().numpy(), basis, geometry, fixed["device"])
                    direct_metric_np = np.stack(direct_metrics, axis=1)
                    rel_power = np.abs(direct_metric_np[:, :, 0] - synth_metrics[:, :, 0]) / np.maximum(
                        np.abs(direct_metric_np[:, :, 0]), np.finfo(np.float64).tiny)
                    require(float(np.max(rel_power)) <= 1e-4,
                            f"direct/synth plane power mismatch: {decoder}/{encoder}/{grid}")
                    require(direct_camera is not None, "direct camera field is missing")
                    direct_readout, _ = _camera_readout(direct_camera, camera_geo, gain, fixed["pixels"], rows[start:stop])
                    synth_readout, _ = _camera_readout(synth_camera, camera_geo, gain, fixed["pixels"], rows[start:stop])
                    mse_diff = float(np.max(np.abs(direct_readout[:, 0] - synth_readout[:, 0])))
                    maximum_mse = max(maximum_mse, mse_diff)
                    require(mse_diff <= 1e-6,
                            f"direct/synth reconstruction MSE mismatch: {decoder}/{encoder}/{grid}: {mse_diff}")
            records.append({"decoder": decoder, "grid": grid,
                            "maximum_complex_relative_error": maximum,
                            "maximum_reconstruction_mse_absolute_error": maximum_mse})
            del model
    # Phase RMS and global phase are separately controlled.
    rng = np.random.default_rng(2026102303)
    vector = rng.normal(size=100)
    vector -= vector.mean()
    vector *= 0.1 / np.sqrt(np.mean(vector**2))
    require(abs(np.sqrt(np.mean(vector**2)) - 0.1) < 1e-12, "phase RMS normalization failed")
    return {"status": "PASSED", "fixed_rows": rows.tolist(), "decoder_replays": records,
            "reusable_d0_cache_replays": cache_checks,
            "complex_relative_tolerance": 1e-4, "mse_absolute_tolerance": 1e-6,
            "phase_rms_control": "PASSED"}


def _preflight(cfg: dict[str, Any], staging: Path, device: torch.device) -> dict[str, Any]:
    require(device.type == "cuda" and torch.cuda.is_available(), "frozen J diagnostic requires CUDA")
    require(not (staging / "preflight.json").exists(), "READY preflight already exists; read it before reusing")
    staging.mkdir(parents=True, exist_ok=True)
    fixed = _load_fixed(cfg, device)
    bases = {(key, grid): _build_basis(key, grid, staging, fixed)
             for _role, grid in GRIDS for key in ENDPOINTS}
    replay = _coherent_replay(fixed, bases)
    identity = ResearchRunIdentity(
        task_id=TASK_ID, run_id=uuid.uuid4().hex, trajectory_id=f"{TASK_ID}:D0_50k_J_45to50k",
        semantic_authority_id=AUTHORITY,
        data_role="fixed_DEV_5000_plus_saved_G4_10000; official_TEST_unread",
        config_identity=cfg,
        model_identity={key: fixed["checkpoints"][key]["model_identity"] for key in ENDPOINTS},
        optimizer_identity={"optimizer_creation": 0, "scientific_updates": 0},
        sampler_identity={"DEV_split_seed": 1234, "saved_G4_coefficients": str(fixed["g4_path"]),
                          "new_prior_samples": 0},
        execution_source_semantics_id="gaussian_j_frozen_attribution_phase_density_v1",
    )
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    receipt = control.make_ready_receipt(
        correctness={"passed": True, "status": "PASSED", "checks": {
            "explicit_checkpoint_steps_and_Adam_counters": [45_000, 47_500, 50_000],
            "D0_J_common_source_and_encoder_at_45k": "PASSED",
            "fixed_DEV_and_old_G4_row_label_proposal_identity": "PASSED",
            "all_four_decoder_grid_bases_direct_replay": replay,
            "training_updates": 0, "optimizers_created": 0, "official_test_reads": 0,
        }},
        resource={"passed": True, "status": "PASSED", "device": torch.cuda.get_device_name(device),
                  "free_bytes_observed": int(torch.cuda.mem_get_info(device)[0])},
        implementation={"passed": True, "status": "PASSED",
                        "engine": "cuda_gaussian_s12_2048_4096_frozen_coherent_basis",
                        "selected_engine": "cuda_gaussian_s12_2048_4096_frozen_coherent_basis"},
        selected_engine="cuda_gaussian_s12_2048_4096_frozen_coherent_basis",
    )
    result = {"schema": "gaussian_j_frozen_attribution_preflight.v1", "task_id": TASK_ID,
              "status": "READY", "identity": identity.to_dict(), "receipt": receipt.to_dict(),
              "checks": replay, "device": torch.cuda.get_device_name(device),
              "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device)}
    _write_json(staging / "preflight.json", result)
    _write_json(staging / "run_identity.json", identity.to_dict())
    _write_json(staging / "config.json", cfg)
    return result


def _open_attempt(root: Path, staging: Path) -> tuple[ResearchRunControl, ResearchRunIdentity]:
    preflight_path = root / "preflight.json" if root.is_dir() else staging / "preflight.json"
    record = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(record.get("task_id") == TASK_ID and record.get("status") == "READY",
            "the same task's READY preflight is required")
    identity = ResearchRunIdentity.from_mapping(record["identity"])
    receipt = PreflightReceipt.from_mapping(record["receipt"])
    control = ResearchRunControl(identity, authorization_id=AUTHORITY)
    control.mark_preflight_ready(receipt)
    if root.exists():
        control.resume_scientific_attempt(root, "attempt_01", receipt)
    else:
        control.open_scientific_attempt(root, "attempt_01", receipt)
        import shutil
        shutil.copy2(preflight_path, root / "preflight.json")
        shutil.copy2(staging / "config.json", root / "config.json")
        shutil.copytree(staging / "bases", root / "bases", dirs_exist_ok=True)
    _write_json(root / "run_identity.json", control.identity.to_dict())
    return control, control.identity


def _measure_dev(key: str, role: str, grid: int, root: Path, fixed: dict[str, Any],
                d0: dict[str, np.ndarray] | None, reuse_d0: bool = False) -> dict[str, Any]:
    rows_all = fixed["dev_rows"].numpy().astype(np.int64)
    directory = _cache_dir(root, key, role)
    directory.mkdir(parents=True, exist_ok=True)
    if key == "D050000" and reuse_d0 and not list(directory.glob("chunk_*.npz")):
        source = _resolve(fixed["config"]["inputs"]["reusable_density_run"])
        source = source / "populations" / "dev" / key / role
        require(source.is_dir(), f"verified reusable D0 cache disappeared: {source}")
        shutil.copytree(source, directory, dirs_exist_ok=True)
        _load_chunks(directory, rows_all, fixed["labels"])
    model = _model(key, grid, fixed)
    gain = float(fixed["checkpoints"][key]["camera_gain"].reshape(-1)[0])
    all_geometry = None
    for chunk_index, start in enumerate(range(0, len(rows_all), CHUNK_ROWS)):
        stop = min(len(rows_all), start + CHUNK_ROWS)
        rows = rows_all[start:stop]
        labels = fixed["labels"][torch.as_tensor(rows)].numpy().astype(np.int64)
        path = directory / f"chunk_{chunk_index:04d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                require(np.array_equal(saved["rows"], rows) and np.array_equal(saved["labels"], labels)
                        and saved["metrics"].shape == (len(rows), len(DEV_PLANES), 7)
                        and saved["distribution"].shape == (len(rows), len(DIST_FIELDS))
                        and saved["coefficients"].shape == (len(rows), 100)
                        and saved["coefficients"].dtype == np.complex64
                        and saved["reconstruction"].shape == (len(rows), len(REC_FIELDS))
                        and saved["equal_camera_power_reconstruction"].shape == (len(rows), len(REC_FIELDS)),
                        f"J measurement recovery identity changed: {path}")
            continue
        metrics_parts, distributions, coeff_parts, rec_parts, equal_parts = [], [], [], [], []
        for batch_start in range(start, stop, BATCH_ROWS):
            batch_stop = min(stop, batch_start + BATCH_ROWS)
            batch_rows = rows_all[batch_start:batch_stop]
            image = image_batch(fixed["pixels"], torch.as_tensor(batch_rows), "small", fixed["device"])
            with torch.inference_mode():
                result = model.runtime.forward_from_image(image)
                fields, _trace = small_image_result_fields(model, result)
                plane_metrics, current_geometry, current_distribution = [], [], None
                for plane_index, (name, field) in enumerate(fields):
                    values, geometry = field_plane_metrics(field)
                    plane_metrics.append(np.asarray(values, dtype=np.float64))
                    current_geometry.append({"plane_key": name, **geometry})
                    if plane_index == 6:
                        current_distribution = field_intensity_distribution(field)
                metrics_parts.append(np.stack(plane_metrics, axis=1))
                distributions.append(np.asarray(
                    [[float(row.get(name, math.nan)) for name in DIST_FIELDS] for row in current_distribution],
                    dtype=np.float64))
                coeff = result.z_ols.coefficients.detach().cpu().numpy().astype(np.complex64)
                coeff_parts.append(coeff)
                raw = result.physical_sensor_output_w_per_pixel
                gained = raw * gain
                view = resize_view(gained)
                target = fixed["pixels"].index_select(0, torch.as_tensor(batch_rows)).unsqueeze(1).to(
                    device=view.device, dtype=view.dtype) / 255.0
                mse = (view - target).square().mean((1, 2, 3))
                psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
                ssim = ssim_batch(view, target)
                clipping = ((gained < 0) | (gained > 1)).double().mean((1, 2, 3))
                camera_power = result.physical_power_trace["P_camera_W"].to(torch.float64)
                native = torch.stack((mse, psnr, ssim, clipping, camera_power), dim=1)
                rec_parts.append(native.cpu().numpy().astype(np.float64))
                if key == "J50000":
                    require(d0 is not None, "D0 camera power is required for the equal-power control")
                    target_power = d0["metrics"][batch_start:batch_stop, 9, 0]
                    require(fields[-1][0] == DECODER_PLANES[-1],
                            "camera field is missing from the production plane list")
                    equal, _equal_view, _defined = _equal_camera_readout(
                        fields[-1][1].tensor[:, 0], current_geometry[9], gain,
                        target_power, fixed["pixels"], batch_rows,
                    )
                    equal_parts.append(equal)
                else:
                    equal_parts.append(native.cpu().numpy().astype(np.float64))
                all_geometry = current_geometry if all_geometry is None else all_geometry
        _save_chunk(path, rows, labels, metrics=np.concatenate(metrics_parts),
                    distribution=np.concatenate(distributions), coefficients=np.concatenate(coeff_parts),
                    reconstruction=np.concatenate(rec_parts),
                    equal_camera_power_reconstruction=np.concatenate(equal_parts))
        _write_json(root / "progress.json", {"stage": "DEV", "endpoint": key, "grid": grid,
                    "completed_rows": stop, "total_rows": len(rows_all)})
    metadata = directory / "metadata.json"
    if not metadata.exists():
        if all_geometry is None:
            one = image_batch(fixed["pixels"], torch.as_tensor(rows_all[:1]), "small", fixed["device"])
            with torch.inference_mode():
                fields, _ = small_image_result_fields(model, model.runtime.forward_from_image(one))
            all_geometry = [{"plane_key": name, **plane_geometry(field.plane)} for name, field in fields]
        _write_json(metadata, {"endpoint": key, "global_step": 50_000, "grid_role": role, "grid": grid,
                               "data_role": "fixed_DEV_500_per_class", "plane_order": list(DEV_PLANES),
                               "planes": all_geometry})
    del model
    return {"endpoint": key, "grid": grid, "grid_role": role, "rows": len(rows_all),
            "chunks": math.ceil(len(rows_all) / CHUNK_ROWS)}


def _posthoc_j_probes(root: Path, fixed: dict[str, Any]) -> list[dict[str, Any]]:
    """Replay the three saved J states on the fixed 20-image DEV probe."""
    selected = [int(x) for label in range(10)
                for x in fixed["dev_rows"][fixed["labels"].index_select(0, fixed["dev_rows"]) == label][:2]]
    rows = np.asarray(selected, dtype=np.int64)
    labels = fixed["labels"][torch.as_tensor(rows)].numpy().astype(np.int64)
    images = image_batch(fixed["pixels"], torch.as_tensor(rows), "small", fixed["device"])
    records = []
    for step in (45_000, 47_500, 50_000):
        endpoint = f"J{step}"
        cp = fixed["checkpoints"][endpoint]
        require(cp["completed_global_step"] == step, f"posthoc J probe checkpoint step changed: {endpoint}")
        for role, grid in GRIDS:
            model = _model(endpoint, grid, fixed)
            with torch.inference_mode():
                result = model.runtime.forward_from_image(images)
                fields, _trace = small_image_result_fields(model, result)
                metrics, geometry, distributions = [], [], None
                for index, (name, field) in enumerate(fields):
                    values, geo = field_plane_metrics(field)
                    metrics.append(np.asarray(values, dtype=np.float64))
                    geometry.append({"plane_key": name, **plane_geometry(field.plane)})
                    if index == 6:
                        distributions = field_intensity_distribution(field)
                coefficients = result.z_ols.coefficients.detach().cpu().numpy().astype(np.complex64)
            distribution = np.asarray([[float(item.get(name, math.nan)) for name in DIST_FIELDS]
                                       for item in (distributions or [])], dtype=np.float64)
            require(distribution.shape == (20, len(DIST_FIELDS)), "posthoc J OLS distribution is incomplete")
            path = root / "populations" / "posthoc_probes" / f"{endpoint}_{grid}.npz"
            payload = dict(endpoint=np.asarray(endpoint), global_step=np.asarray(step, dtype=np.int64),
                grid=np.asarray(grid, dtype=np.int64), rows=rows, labels=labels,
                coefficients=coefficients, metrics=np.stack(metrics, axis=1), distribution=distribution,
                geometry_json=np.asarray(json.dumps(geometry, ensure_ascii=False)),
                replay_scope=np.asarray("posthoc_frozen_replay"))
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    require(str(saved["endpoint"].item()) == endpoint and int(saved["global_step"]) == step
                            and int(saved["grid"]) == grid and np.array_equal(saved["rows"], rows)
                            and np.array_equal(saved["labels"], labels)
                            and saved["metrics"].shape == (20, 10, 7),
                            f"posthoc J probe recovery identity changed: {path}")
                    old_coeff = saved["coefficients"].astype(np.complex128)
                    new_coeff = coefficients.astype(np.complex128)
                    coeff_scale = max(float(np.linalg.norm(new_coeff)), np.finfo(np.float64).tiny)
                    coeff_error = float(np.linalg.norm(old_coeff-new_coeff)/coeff_scale)
                    require(coeff_error <= 1e-4
                            and np.allclose(saved["metrics"], payload["metrics"], rtol=1e-4, atol=1e-12)
                            and np.allclose(saved["distribution"], distribution, rtol=1e-4, atol=1e-12)
                            and str(saved["geometry_json"].item()) == str(payload["geometry_json"].item()),
                            f"posthoc J probe cached values differ from current frozen replay: {path}")
            else:
                _write_npz(path, **payload)
            records.append({"endpoint": endpoint, "global_step": step, "grid_role": role,
                            "grid": grid, "rows": 20, "planes": 10,
                            "recording_type": "posthoc_frozen_replay",
                            "path": str(path.relative_to(root))})
            del model
    return records


def _production_loss(
    coefficients: np.ndarray, powers: np.ndarray, camera: torch.Tensor, geometry: dict[str, Any],
    encoder_metrics: np.ndarray, pixels: torch.Tensor, rows: np.ndarray,
    fixed: dict[str, Any], gain: float,
) -> dict[str, float]:
    require(len(rows) == BATCH_ROWS, "production objective is evaluated in fixed groups of ten")
    device = fixed["device"]
    z = torch.as_tensor(coefficients, dtype=torch.complex64, device=device)
    mode_power = z.real.double().square() + z.imag.double().square()
    u = mode_power / (fixed["scale"].to(device=device, dtype=torch.float64).unsqueeze(0) + base.EPS_S)
    dist = base.truncated_normal_positive_kl(base.q_from_batch(u), base.fixed_prior(device=device)).mean()
    plane_power = np.stack((encoder_metrics[:, 3, 0], encoder_metrics[:, 4, 0],
                            powers[:, 0, 0], powers[:, 1, 0], powers[:, 2, 0]), axis=1)
    plane_power_t = torch.as_tensor(plane_power, dtype=torch.float64, device=device)
    power_reference = fixed["power_reference"].to(device=device, dtype=torch.float64).unsqueeze(0)
    op = torch.log((plane_power_t + base.EPS_OP) / (power_reference + base.EPS_OP)).square().mean()
    pixel_area = math.prod(float(value) for value in geometry["pixel_pitch_m"])
    raw = camera.abs().square().unsqueeze(1) * pixel_area
    target_native = image_batch(pixels, torch.as_tensor(rows), "small", device)
    rec = (raw * gain - target_native).square().mean()
    view = resize_view(raw * gain)
    target_28 = pixels.index_select(0, torch.as_tensor(rows)).unsqueeze(1).to(
        device=device, dtype=torch.float32) / 255.0
    ssim_loss = 1.0 - ssim_batch(view, target_28).mean()
    total = sum(float(fixed["weights"][name]) * value for name, value in
                (("dist", dist), ("rec", rec), ("op", op)))
    total = total + float(fixed["structural_weight"]) * ssim_loss
    return {name: float(value.detach().double().cpu()) for name, value in
            {"total": total, "dist": dist, "rec": rec, "op": op, "ssim_loss": ssim_loss}.items()}


def _measure_crossover(root: Path, fixed: dict[str, Any], bases: dict[Any, Any],
                       dev: dict[Any, dict[str, np.ndarray]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows_out: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    combos = (("L00", "D050000", "D050000"), ("L01", "D050000", "J50000"),
              ("L10", "J50000", "D050000"), ("L11", "J50000", "J50000"))
    for role, grid in GRIDS:
        endpoint_arrays = {key: dev[(key, grid)] for key in ENDPOINTS}
        d0_power = endpoint_arrays["D050000"]["metrics"][:, 9, 0]
        for combo, decoder, encoder in combos:
            path = root / "populations" / "crossover" / f"{combo}_{grid}.npz"
            source = endpoint_arrays[encoder]
            recovered = path.is_file()
            if recovered:
                with np.load(path, allow_pickle=False) as saved:
                    require(np.array_equal(saved["rows"], source["rows"])
                            and np.array_equal(saved["labels"], source["labels"])
                            and saved["metrics"].shape == (5_000, 4, 7)
                            and saved["readout"].shape == (5_000, len(REC_FIELDS))
                            and saved["equal"].shape == (5_000, len(REC_FIELDS))
                            and saved["equal_defined"].shape == (5_000,)
                            and saved["production_components"].shape == (500, 5),
                            f"crossover recovery population changed: {path}")
                    values = {key: saved[key].copy() for key in (
                        "rows", "labels", "metrics", "readout", "equal", "equal_defined", "coeff")}
                    components = saved["production_components"].copy()
            else:
                basis, geometry = bases[(decoder, grid)]
                camera_geometry = next(item for item in geometry if item["plane_key"] == DECODER_PLANES[3])
                blocks = []
                for start in range(0, 5000, CHUNK_ROWS):
                    stop = min(5000, start + CHUNK_ROWS)
                    rows = source["rows"][start:stop]
                    labels = source["labels"][start:stop]
                    coeff = source["coefficients"][start:stop]
                    metrics, _dist, camera = _synthesize(decoder, coeff, basis, geometry, fixed["device"])
                    gain = float(fixed["checkpoints"][decoder]["camera_gain"].reshape(-1)[0])
                    readout, _view = _camera_readout(camera, camera_geometry, gain, fixed["pixels"], rows)
                    equal, _eq_view, equal_defined = _equal_camera_readout(
                        camera, camera_geometry, gain, d0_power[start:stop], fixed["pixels"], rows)
                    production = []
                    for local in range(0, len(rows), BATCH_ROWS):
                        sl = slice(local, local + BATCH_ROWS)
                        value = _production_loss(coeff[sl], metrics[sl], camera[sl], camera_geometry,
                                                 source["metrics"][start + local:start + local + BATCH_ROWS],
                                                 fixed["pixels"], rows[sl], fixed, gain)
                        production.append(value)
                        loss_rows.append({"grid": grid, "combo": combo, "batch_index": (start + local) // BATCH_ROWS,
                                          **value})
                    blocks.append({"rows": rows, "labels": labels, "metrics": metrics, "readout": readout,
                                   "equal": equal, "equal_defined": equal_defined, "coeff": coeff,
                                   "production": production})
                values = {name: np.concatenate([block[name] for block in blocks], axis=0)
                          for name in ("rows", "labels", "metrics", "readout", "equal", "equal_defined", "coeff")}
                components = np.asarray([[row[name] for name in ("total", "dist", "rec", "op", "ssim_loss")]
                                          for block in blocks for row in block["production"]], dtype=np.float64)
                _write_npz(path, **values, production_components=components)
            if recovered:
                # A recovered shard carries all five exact production objective terms.
                components = np.asarray(components, dtype=np.float64)
                for index, component in enumerate(components):
                    loss_rows.append({"grid": grid, "combo": combo, "batch_index": index,
                        **dict(zip(("total", "dist", "rec", "op", "ssim_loss"), component.tolist(), strict=True))})
            for i in range(5000):
                rows_out.append({"grid": grid, "grid_role": role, "combo": combo,
                    "row": int(values["rows"][i]), "label": int(values["labels"][i]),
                    "mse": float(values["readout"][i, 0]), "psnr_db": float(values["readout"][i, 1]),
                    "ssim": float(values["readout"][i, 2]), "clipping_fraction": float(values["readout"][i, 3]),
                    "camera_power_w": float(values["readout"][i, 4]),
                    "equal_power_defined": bool(values["equal_defined"][i]),
                    "equal_power_mse": float(values["equal"][i, 0]),
                    "equal_power_ssim": float(values["equal"][i, 2]),
                    "equal_power_clipping_fraction": float(values["equal"][i, 3]),
                    "equal_camera_power_w": float(values["equal"][i, 4])})
    require(len(rows_out) == 40_000 and len(loss_rows) == 4_000,
            "full factorial DEV or production-loss population is incomplete")
    return rows_out, loss_rows


def _phase_conditions(fixed: dict[str, Any], cfg: dict[str, Any], labels: np.ndarray
                      ) -> list[dict[str, Any]]:
    model = _model("D050000", 2048, fixed)
    centers = model.runtime.ols.operator.centers_m.detach().cpu().numpy().astype(np.float64)
    del model
    centers -= centers.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(cfg["phase"]["seed"])
    conditions = [{"condition_id": "zero", "family": "zero", "sigma": 0.0, "repeat": 0,
                   "phase": np.zeros((len(labels), 100), dtype=np.float64)}]
    label_array = np.asarray(labels, dtype=np.int64)
    for sigma in cfg["phase"]["relative_rms_rad"]:
        for family in cfg["phase"]["families"]:
            vectors = {}
            for label in range(10):
                for repeat in range(cfg["phase"]["repeats_per_class"]):
                    if family == "independent_port":
                        direction = rng.normal(size=100)
                    else:
                        angle = rng.uniform(0.0, 2.0 * math.pi)
                        direction = centers @ np.asarray((math.cos(angle), math.sin(angle)))
                    direction -= direction.mean()
                    norm = float(np.sqrt(np.mean(direction**2)))
                    require(norm > 0.0, "phase perturbation direction has zero RMS")
                    vectors[(label, repeat)] = direction * (float(sigma) / norm)
            for repeat in range(cfg["phase"]["repeats_per_class"]):
                phase = np.stack([vectors[(int(label), repeat)] for label in label_array])
                require(np.allclose(np.sqrt(np.mean(phase**2, axis=1)), sigma, atol=1e-12, rtol=0.0),
                        "per-image phase RMS differs from the registered perturbation")
                conditions.append({"condition_id": f"{family}_{sigma:g}_{repeat:02d}",
                                   "family": family, "sigma": float(sigma), "repeat": repeat, "phase": phase})
    conditions.append({"condition_id": "global_common_0.3", "family": "common_phase_control",
                       "sigma": 0.3, "repeat": 0, "phase": np.full((len(labels), 100), 0.3)})
    require(len(conditions) == 66, "phase condition count changed")
    return conditions


def _phase_measure(root: Path, cfg: dict[str, Any], fixed: dict[str, Any], bases: dict[Any, Any],
                   dev: dict[Any, dict[str, np.ndarray]]) -> dict[str, Any]:
    evaluator = base._load_evaluator(_resolve(fixed["task_config"]["evaluation"]["evaluator_checkpoint"]),
                                     fixed["device"])
    ec_eval, _ = decoder_ref._load_ec_evaluator(
        _resolve(fixed["task_config"]["evaluation"]["ec_checkpoint"]), fixed["device"])
    routes = (
        ("D0_DEV", "D050000", "D050000", "dev"),
        ("J_DEV", "J50000", "J50000", "dev"),
        ("G4_to_D0", "G4", "D050000", "g4"),
        ("G4_to_J", "G4", "J50000", "g4"),
    )
    total = 0
    for role, grid in GRIDS:
        for route, encoder, decoder, kind in routes:
            if kind == "dev":
                data = dev[(encoder, grid)]
                positions = [int(i) for label in range(10)
                             for i in np.flatnonzero(data["labels"] == label)[:20]]
                coefficients = data["coefficients"][positions]
                rows = data["rows"][positions]
                labels = data["labels"][positions]
                proposals = np.full(200, -1, dtype=np.int64)
            else:
                g4 = fixed["g4"]
                g4_labels = g4["requested_labels"].numpy().astype(np.int64)
                positions = [int(i) for label in range(10) for i in np.flatnonzero(g4_labels == label)[:20]]
                coefficients = g4["ols_coefficients"].numpy()[positions]
                labels = g4_labels[positions]
                rows = np.asarray(positions, dtype=np.int64)
                proposals = g4["proposal_ids"].numpy()[positions].astype(np.int64)
            conditions = _phase_conditions(fixed, cfg, labels)
            directory = root / "populations" / "phase" / route / str(grid)
            directory.mkdir(parents=True, exist_ok=True)
            basis, geometry = bases[(decoder, grid)]
            camera_geo = next(item for item in geometry if item["plane_key"] == DECODER_PLANES[3])
            gain = float(fixed["checkpoints"][decoder]["camera_gain"].reshape(-1)[0])
            for condition_index, condition in enumerate(conditions):
                path = directory / f"condition_{condition_index:02d}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        require(str(saved["condition_id"].item()) == condition["condition_id"]
                                and np.array_equal(saved["rows"], rows)
                                and np.array_equal(saved["labels"], labels)
                                and np.array_equal(saved["proposal_ids"], proposals)
                                and str(saved["family"].item()) == condition["family"]
                                and float(saved["sigma"]) == float(condition["sigma"])
                                and int(saved["repeat"]) == int(condition["repeat"])
                                and saved["metrics"].shape == (200, 4, 7)
                                and saved["readout"].shape == (200, len(REC_FIELDS)),
                                f"phase recovery identity changed: {path}")
                        if condition["family"] == "common_phase_control":
                            with np.load(directory / "condition_00.npz", allow_pickle=False) as baseline:
                                rel = np.abs(saved["metrics"][:, :, 0]-baseline["metrics"][:, :, 0]) / np.maximum(
                                    np.abs(baseline["metrics"][:, :, 0]), np.finfo(np.float64).tiny)
                                require(float(np.max(rel)) <= 1e-4
                                        and np.max(np.abs(saved["readout"][:, 4]-baseline["readout"][:, 4]))
                                            <= max(1e-12, float(np.max(baseline["readout"][:, 4]))*1e-4)
                                        and np.array_equal(saved["ed_prediction"], baseline["ed_prediction"])
                                        and np.array_equal(saved["ec_prediction"], baseline["ec_prediction"]),
                                        "cached common-phase negative control changed physical outputs")
                                if kind == "dev":
                                    require(np.max(np.abs(saved["readout"][:, 0]-baseline["readout"][:, 0])) <= 1e-6
                                            and np.max(np.abs(saved["readout"][:, 2]-baseline["readout"][:, 2])) <= 1e-6,
                                            "cached common-phase control changed DEV reconstruction")
                        total += len(rows)
                    continue
                phases = condition["phase"]
                perturbed = coefficients * np.exp(1j * phases).astype(np.complex64)
                metrics, dist, camera = _synthesize(decoder, perturbed, basis, geometry, fixed["device"])
                readout, view = _camera_readout(camera, camera_geo, gain,
                    fixed["pixels"] if kind == "dev" else None, rows if kind == "dev" else None)
                ed_prediction = evaluator(view).argmax(1).cpu().numpy().astype(np.int64)
                ec_prediction = decoder_ref._ec_logits(ec_eval, view).argmax(1).cpu().numpy().astype(np.int64)
                baseline_path = directory / "condition_00.npz"
                if condition_index == 0:
                    baseline_readout = readout.copy()
                    baseline_metrics = metrics.copy()
                    baseline_mse = readout[:, 0].copy()
                    baseline_ssim = readout[:, 2].copy()
                    baseline_ed, baseline_ec = ed_prediction.copy(), ec_prediction.copy()
                else:
                    with np.load(baseline_path, allow_pickle=False) as baseline:
                        baseline_readout = baseline["readout"].copy()
                        baseline_metrics = baseline["metrics"].copy()
                        baseline_mse = baseline["readout"][:, 0].copy()
                        baseline_ssim = baseline["readout"][:, 2].copy()
                        baseline_ed = baseline["ed_prediction"].copy()
                        baseline_ec = baseline["ec_prediction"].copy()
                if condition["family"] == "common_phase_control":
                    plane_power = metrics[:, :, 0]
                    base_plane_power = baseline_metrics[:, :, 0]
                    rel_power = np.abs(plane_power-base_plane_power) / np.maximum(
                        np.abs(base_plane_power), np.finfo(np.float64).tiny)
                    require(float(np.max(rel_power)) <= 1e-4
                            and np.max(np.abs(readout[:, 4] - baseline_readout[:, 4])) <=
                            max(1e-12, float(np.max(baseline_readout[:, 4])) * 1e-4)
                            and np.array_equal(ed_prediction, baseline_ed)
                            and np.array_equal(ec_prediction, baseline_ec),
                            f"common phase negative control changed intensity: {route}/{grid}")
                    if kind == "dev":
                        require(np.max(np.abs(readout[:, 0] - baseline_mse)) <= 1e-6
                                and np.max(np.abs(readout[:, 2] - baseline_ssim)) <= 1e-6,
                                f"common phase control changed DEV reconstruction: {route}/{grid}")
                _write_npz(path, condition_id=np.asarray(condition["condition_id"]), rows=rows,
                    labels=labels, proposal_ids=proposals, metrics=metrics, distribution=dist,
                    readout=readout, ed_prediction=ed_prediction, ec_prediction=ec_prediction,
                    baseline_ed_prediction=baseline_ed, baseline_ec_prediction=baseline_ec,
                    baseline_mse=baseline_mse, baseline_ssim=baseline_ssim,
                    family=np.asarray(condition["family"]), sigma=np.asarray(condition["sigma"]),
                    repeat=np.asarray(condition["repeat"]))
                total += len(rows)
    expected = 105_600
    require(total == expected, f"phase condition population {total} != {expected}")
    return {"status": "VERIFIED", "rows": total, "conditions_per_route_grid": 66,
            "fixed_images_per_route": 200, "path": "populations/phase"}


def _phase_to_csv(root: Path) -> int:
    output = root / "tables" / "phase_sensitivity_per_image.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["route", "data_role", "grid_role", "grid", "sample_row", "label", "proposal_id",
              "condition_id", "family", "relative_phase_rms_rad", "repeat", "ols_power_w",
              "ols_effective_area_m2", "ols_geometric_density_w_cm2", "ols_effective_density_w_cm2",
              "camera_power_w", "camera_clipping_fraction", "mse", "ssim", "baseline_mse", "baseline_ssim",
              "ed_prediction", "ec_prediction", "baseline_ed_prediction", "baseline_ec_prediction",
              "ed_correct", "ec_correct", "baseline_ed_correct", "baseline_ec_correct"]
    count = 0
    with gzip.open(output, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for role, grid in GRIDS:
            for route in ("D0_DEV", "J_DEV", "G4_to_D0", "G4_to_J"):
                directory = root / "populations" / "phase" / route / str(grid)
                for index in range(66):
                    with np.load(directory / f"condition_{index:02d}.npz", allow_pickle=False) as saved:
                        for i in range(len(saved["rows"])):
                            readout = saved["readout"][i]
                            metric = saved["metrics"][i]
                            row = {
                                "route": route, "data_role": "fixed_DEV" if route.endswith("DEV") else "saved_G4",
                                "grid_role": role, "grid": grid, "sample_row": int(saved["rows"][i]),
                                "label": int(saved["labels"][i]), "proposal_id": int(saved["proposal_ids"][i]),
                                "condition_id": str(saved["condition_id"].item()),
                                "family": str(saved["family"].item()), "relative_phase_rms_rad": float(saved["sigma"]),
                                "repeat": int(saved["repeat"]), "ols_power_w": float(metric[0, 0]),
                                "ols_effective_area_m2": float(metric[0, 2]),
                                "ols_geometric_density_w_cm2": float(metric[0, 3] / 1e4),
                                "ols_effective_density_w_cm2": float(metric[0, 4] / 1e4),
                                "camera_power_w": float(readout[4]),
                                "camera_clipping_fraction": float(readout[3]),
                                "mse": float(readout[0]), "ssim": float(readout[2]),
                                "baseline_mse": float(saved["baseline_mse"][i]),
                                "baseline_ssim": float(saved["baseline_ssim"][i]),
                                "ed_prediction": int(saved["ed_prediction"][i]),
                                "ec_prediction": int(saved["ec_prediction"][i]),
                                "baseline_ed_prediction": int(saved["baseline_ed_prediction"][i]),
                                "baseline_ec_prediction": int(saved["baseline_ec_prediction"][i]),
                                "ed_correct": bool(saved["ed_prediction"][i] == saved["labels"][i]),
                                "ec_correct": bool(saved["ec_prediction"][i] == saved["labels"][i]),
                                "baseline_ed_correct": bool(saved["baseline_ed_prediction"][i] == saved["labels"][i]),
                                "baseline_ec_correct": bool(saved["baseline_ec_prediction"][i] == saved["labels"][i]),
                            }
                            writer.writerow(row); count += 1
    require(count == 105_600, f"phase CSV keys {count} != 105600")
    return count


def _measure_g4(root: Path, fixed: dict[str, Any], bases: dict[Any, Any]
                ) -> list[dict[str, Any]]:
    evaluator = base._load_evaluator(_resolve(fixed["task_config"]["evaluation"]["evaluator_checkpoint"]),
                                     fixed["device"])
    ec_eval, _ = decoder_ref._load_ec_evaluator(
        _resolve(fixed["task_config"]["evaluation"]["ec_checkpoint"]), fixed["device"])
    coefficients = fixed["g4"]["ols_coefficients"].numpy().astype(np.complex64)
    labels = fixed["g4"]["requested_labels"].numpy().astype(np.int64)
    proposals = fixed["g4"]["proposal_ids"].numpy().astype(np.int64)
    output_rows = []
    for role, grid in GRIDS:
        for endpoint in ENDPOINTS:
            basis, geometry = bases[(endpoint, grid)]
            metadata_path = root / "populations" / "generation" / endpoint / role / "metadata.json"
            metadata = {"endpoint": endpoint, "global_step": 50_000, "grid_role": role, "grid": grid,
                        "data_role": "saved_old_G4_coefficients_same_absolute_complex_values",
                        "sample_count": 10_000, "plane_order": list(DECODER_PLANES),
                        "planes": geometry, "source_g4": str(fixed["g4_path"]),
                        "OLS_field_synthesis": "coherent complex field sum before intensity"}
            if metadata_path.exists():
                saved_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
                require(saved_meta == metadata, f"G4 metadata identity changed: {metadata_path}")
            else:
                _write_json(metadata_path, metadata)
            camera_geo = next(item for item in geometry if item["plane_key"] == DECODER_PLANES[3])
            gain = float(fixed["checkpoints"][endpoint]["camera_gain"].reshape(-1)[0])
            for chunk_index, start in enumerate(range(0, 10_000, CHUNK_ROWS)):
                stop = min(10_000, start + CHUNK_ROWS)
                rows = np.arange(start, stop, dtype=np.int64)
                path = root / "populations" / "generation" / endpoint / role / f"chunk_{chunk_index:04d}.npz"
                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        require(np.array_equal(saved["rows"], rows)
                                and np.array_equal(saved["labels"], labels[start:stop])
                                and np.array_equal(saved["proposal_ids"], proposals[start:stop]),
                                f"G4 recovery proposal identity changed: {path}")
                        metrics = saved["metrics"].copy(); dist = saved["distribution"].copy()
                        readout = saved["readout"].copy(); equal = saved["equal_readout"].copy()
                        ed = saved["ed_prediction"].copy(); ec = saved["ec_prediction"].copy()
                        ed_eq = saved["equal_ed_prediction"].copy(); ec_eq = saved["equal_ec_prediction"].copy()
                        equal_defined = saved["equal_power_defined"].copy()
                        require(metrics.shape == (len(rows), 4, 7)
                                and dist.shape == (len(rows), len(DIST_FIELDS))
                                and readout.shape == equal.shape == (len(rows), len(REC_FIELDS))
                                and equal_defined.shape == (len(rows),),
                                f"G4 recovery payload is incomplete: {path}")
                else:
                    metrics, dist, camera = _synthesize(endpoint, coefficients[start:stop], basis, geometry,
                                                        fixed["device"])
                    readout, view = _camera_readout(camera, camera_geo, gain)
                    ed = evaluator(view).argmax(1).cpu().numpy().astype(np.int64)
                    ec = decoder_ref._ec_logits(ec_eval, view).argmax(1).cpu().numpy().astype(np.int64)
                    if endpoint == "D050000":
                        target_power = np.asarray(metrics[:, 3, 0], dtype=np.float64)
                    else:
                        d0_path = root / "populations" / "generation" / "D050000" / role / f"chunk_{chunk_index:04d}.npz"
                        with np.load(d0_path, allow_pickle=False) as d0:
                            target_power = torch.as_tensor(d0["metrics"][:, 3, 0],
                                device=fixed["device"], dtype=torch.float64).cpu().numpy()
                    equal, equal_view, equal_defined = _equal_camera_readout(
                        camera, camera_geo, gain, target_power, None, None)
                    ed_eq = np.full(stop-start, -1, dtype=np.int64)
                    ec_eq = np.full(stop-start, -1, dtype=np.int64)
                    valid = torch.as_tensor(equal_defined, device=fixed["device"], dtype=torch.bool)
                    if bool(valid.any()):
                        ed_eq[equal_defined] = evaluator(equal_view[valid]).argmax(1).cpu().numpy().astype(np.int64)
                        ec_eq[equal_defined] = decoder_ref._ec_logits(ec_eval, equal_view[valid]).argmax(1).cpu().numpy().astype(np.int64)
                    _write_npz(path, rows=rows, labels=labels[start:stop], proposal_ids=proposals[start:stop],
                        metrics=metrics, distribution=dist, readout=readout, equal_readout=equal,
                        ed_prediction=ed, ec_prediction=ec, equal_ed_prediction=ed_eq, equal_ec_prediction=ec_eq,
                        equal_power_defined=equal_defined)
                for i, row in enumerate(rows):
                    output_rows.append({
                        "endpoint": endpoint, "grid_role": role, "grid": grid, "cohort_row": int(row),
                        "requested_label": int(labels[start+i]), "proposal_id": int(proposals[start+i]),
                        "ols_output_power_w": float(metrics[i, 0, 0]),
                        "ols_effective_area_m2": float(metrics[i, 0, 2]),
                        "ols_mean_intensity_geometric_w_cm2": float(metrics[i, 0, 3] / 1e4),
                        "ols_mean_intensity_effective_w_cm2": float(metrics[i, 0, 4] / 1e4),
                        "ed_prediction": int(ed[i]), "ec_prediction": int(ec[i]),
                        "ed_correct": bool(ed[i] == labels[start+i]), "ec_correct": bool(ec[i] == labels[start+i]),
                        "proxy_clipping_fraction": float(readout[i, 3]), "camera_power_w": float(readout[i, 4]),
                        "equal_camera_power_ed_prediction": int(ed_eq[i]),
                        "equal_camera_power_ec_prediction": int(ec_eq[i]),
                        "equal_camera_power_defined": bool(equal_defined[i]),
                        "equal_camera_power_ed_correct": bool(ed_eq[i] == labels[start+i]),
                        "equal_camera_power_ec_correct": bool(ec_eq[i] == labels[start+i]),
                        "equal_camera_power_w": float(equal[i, 4]),
                    })
    require(len(output_rows) == 40_000, "saved G4 Decoder population is incomplete")
    return output_rows


def _density_record(population: str, endpoint: str, role: str, grid: int, row: int, label: int,
                    proposal_id: int, step: int, plane_index: int, plane: dict[str, Any],
                    metrics: np.ndarray, distribution: np.ndarray | None = None,
                    recording_type: str = "frozen_measurement") -> dict[str, Any]:
    value = np.asarray(metrics, dtype=np.float64)
    require(value.shape == (7,), "plane metric vector must contain seven values")
    dist = np.full(len(DIST_FIELDS), math.nan) if distribution is None else np.asarray(distribution, dtype=np.float64)
    require(dist.shape == (len(DIST_FIELDS),), "plane intensity distribution vector changed")
    return {
        "population": population, "endpoint": endpoint, "data_role": population,
        "recording_type": recording_type, "global_step": step, "grid_role": role, "grid": grid,
        "sample_row": row, "label": label, "proposal_id": proposal_id,
        "plane_index": plane_index, "plane_key": plane["plane_key"], "plane_id": plane["plane_id"],
        "coordinate_frame": plane["coordinate_frame"], "height": plane["shape"][0],
        "width": plane["shape"][1], "pixel_pitch_y_m": plane["pixel_pitch_m"][0],
        "pixel_pitch_x_m": plane["pixel_pitch_m"][1], "origin_y_m": plane["origin_m"][0],
        "origin_x_m": plane["origin_m"][1], "z_m": plane["z_m"],
        "measurement_boundary": plane["measurement_boundary"],
        "power_w": value[0], "geometric_area_m2": value[1], "effective_area_m2": value[2],
        "mean_intensity_geometric_w_m2": value[3], "mean_intensity_geometric_w_cm2": value[3] / 1.0e4,
        "mean_intensity_effective_w_m2": value[4], "mean_intensity_effective_w_cm2": value[4] / 1.0e4,
        "effective_to_geometric_area_ratio": value[5], "integral_intensity_squared_w2_m2": value[6],
        **{name: float(dist[index]) for index, name in enumerate(DIST_FIELDS)},
    }


def _write_rows(path: Path, rows: list[dict[str, Any]], *, compressed: bool = False) -> None:
    require(bool(rows), f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if compressed else open
    mode = "wt" if compressed else "w"
    with opener(path, mode, newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validated_plane_order(metadata: dict[str, Any], expected: tuple[str, ...], path: Path) -> dict[str, Any]:
    planes = metadata.get("planes")
    require(isinstance(planes, list) and all(isinstance(plane, dict) for plane in planes),
            f"plane geometry is missing or malformed: {path}")
    actual = [plane.get("plane_key") for plane in planes]
    require(actual == list(expected), f"plane geometry order does not match the registered path: {path}")
    saved = metadata.get("plane_order")
    if saved is None:
        # Recover the omitted index from the independently stored per-plane geometry.
        metadata["plane_order"] = actual
        _write_json(path, metadata)
    else:
        require(saved == list(expected), f"saved plane-order index changed: {path}")
    return metadata


def _write_density_tables(root: Path, fixed: dict[str, Any], posthoc: list[dict[str, Any]]) -> dict[str, int]:
    fields = None
    output = root / "tables" / "whole_plane_density_per_image.csv.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(output, "wt", newline="", encoding="utf-8") as stream:
        writer = None
        for endpoint in ENDPOINTS:
            for role, grid in GRIDS:
                directory = _cache_dir(root, endpoint, role)
                metadata_path = directory / "metadata.json"
                metadata = _validated_plane_order(
                    json.loads(metadata_path.read_text(encoding="utf-8")), DEV_PLANES, metadata_path)
                require(metadata.get("endpoint") == endpoint and metadata.get("grid") == grid
                        and metadata.get("plane_order") == list(DEV_PLANES),
                        f"DEV density plane metadata changed: {directory}")
                planes = metadata["planes"]
                for chunk_path in sorted(directory.glob("chunk_*.npz")):
                    with np.load(chunk_path, allow_pickle=False) as saved:
                        for i, sample_row in enumerate(saved["rows"]):
                            for pi, plane in enumerate(planes):
                                record = _density_record("fixed_DEV", endpoint, role, grid,
                                    int(sample_row), int(saved["labels"][i]), -1, 50_000, pi, plane,
                                    saved["metrics"][i, pi], saved["distribution"][i] if pi == 6 else None)
                                if writer is None:
                                    fields = list(record)
                                    writer = csv.DictWriter(stream, fieldnames=fields)
                                    writer.writeheader()
                                writer.writerow(record); count += 1
        for endpoint in ENDPOINTS:
            for role, grid in GRIDS:
                directory = root / "populations" / "generation" / endpoint / role
                metadata_path = directory / "metadata.json"
                metadata = _validated_plane_order(
                    json.loads(metadata_path.read_text(encoding="utf-8")), DECODER_PLANES, metadata_path)
                require(metadata.get("endpoint") == endpoint and metadata.get("grid") == grid
                        and metadata.get("plane_order") == list(DECODER_PLANES),
                        f"G4 density plane metadata changed: {directory}")
                planes = metadata["planes"]
                for chunk_path in sorted(directory.glob("chunk_*.npz")):
                    with np.load(chunk_path, allow_pickle=False) as saved:
                        for i, cohort_row in enumerate(saved["rows"]):
                            for pi, plane in enumerate(planes):
                                record = _density_record("saved_old_G4", endpoint, role, grid,
                                    int(cohort_row), int(saved["labels"][i]), int(saved["proposal_ids"][i]),
                                    50_000, pi, plane, saved["metrics"][i, pi],
                                    saved["distribution"][i] if pi == 0 else None)
                                require(list(record) == fields,
                                        "fixed-DEV and G4 density table schemas diverged")
                                writer.writerow(record); count += 1
    require(count == 360_000, f"full-path density rows {count} != 360000")

    probe_rows = []
    for item in posthoc:
        endpoint, grid = item["endpoint"], item["grid"]
        path = root / item["path"]
        with np.load(path, allow_pickle=False) as saved:
            planes = json.loads(str(saved["geometry_json"].item()))
            for i, row in enumerate(saved["rows"]):
                for pi, plane in enumerate(planes):
                    probe_rows.append(_density_record("fixed_DEV_probe", endpoint, item["grid_role"], grid,
                        int(row), int(saved["labels"][i]), -1, int(saved["global_step"]), pi, plane,
                        saved["metrics"][i, pi], saved["distribution"][i] if pi == 6 else None,
                        "posthoc_frozen_replay"))
    require(len(probe_rows) == 1_200, f"posthoc J plane probe rows {len(probe_rows)} != 1200")
    _write_rows(root / "tables" / "j_historical_checkpoint_density_probe.csv", probe_rows)
    return {"native_full_population_rows": count, "j_posthoc_probe_rows": len(probe_rows)}


def _summarize_density(root: Path) -> dict[str, int]:
    path = root / "tables" / "whole_plane_density_per_image.csv.gz"
    groups: dict[tuple[Any, ...], dict[str, list[float]]] = {}
    summary_fields = ("power_w", "effective_area_m2", "mean_intensity_geometric_w_m2",
                      "mean_intensity_effective_w_m2", "effective_to_geometric_area_ratio")
    count = 0
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            count += 1
            base_key = (row["population"], row["endpoint"], int(row["grid"]), row["plane_key"])
            for label in (-1, int(row["label"])):
                key = (*base_key, label)
                target = groups.setdefault(key, {name: [] for name in summary_fields})
                for name in summary_fields:
                    target[name].append(float(row[name]))
    require(count == 360_000, "density summary consumer received an incomplete full table")
    rows = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        population, endpoint, grid, plane, label = key
        output: dict[str, Any] = {"population": population, "endpoint": endpoint, "grid": grid,
                                  "plane_key": plane, "label": label, "n": len(values["power_w"])}
        for name, raw in values.items():
            data = np.asarray(raw, dtype=np.float64)
            for stat, value in (("mean", np.mean(data)), ("median", np.median(data)),
                                ("q10", np.quantile(data, 0.1)), ("q90", np.quantile(data, 0.9))):
                output[f"{name}_{stat}"] = float(value)
        rows.append(output)
    _write_rows(root / "tables" / "whole_plane_density_summary_by_class.csv", rows)
    return {"source_rows": count, "summary_rows": len(rows)}


def _complex_moment_matrices(coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(coefficients, dtype=np.complex128)
    require(z.ndim == 2 and z.shape[0] > 0 and np.isfinite(z).all(),
            "complex coefficient matrix must be a finite nonempty [samples,ports] array")
    pivot = z[np.arange(len(z)), np.argmax(np.abs(z), axis=1)]
    rotations = np.ones(len(z), dtype=np.complex128)
    nonzero = np.abs(pivot) > 0.0
    rotations[nonzero] = np.conj(pivot[nonzero]) / np.abs(pivot[nonzero])
    gauged = z * rotations[:, None]
    mean = gauged.mean(axis=0)
    second = gauged.T @ gauged.conj() / len(gauged)
    covariance = second - np.outer(mean, mean.conj())
    covariance = (covariance + covariance.conj().T) * 0.5
    require(np.max(np.abs(second-second.conj().T)) < 1e-10
            and np.max(np.abs(covariance-covariance.conj().T)) < 1e-10,
            "complex second moment or covariance is not Hermitian")
    return second, covariance, gauged


def _coefficient_moments(root: Path, fixed: dict[str, Any],
                         dev: dict[tuple[str, int], dict[str, np.ndarray]]) -> dict[str, Any]:
    g4_coeff = fixed["g4"]["ols_coefficients"].numpy().astype(np.complex128)
    g4_labels = fixed["g4"]["requested_labels"].numpy().astype(np.int64)
    summaries: list[dict[str, Any]] = []
    seconds: list[np.ndarray] = []
    covariances: list[np.ndarray] = []
    eigen_spectra: list[np.ndarray] = []
    class_vectors: dict[tuple[int, str, int], np.ndarray] = {}
    for role, grid in GRIDS:
        populations = {
            "E0_DEV": (dev[("D050000", grid)]["coefficients"].astype(np.complex128),
                       dev[("D050000", grid)]["labels"]),
            "EJ_DEV": (dev[("J50000", grid)]["coefficients"].astype(np.complex128),
                       dev[("J50000", grid)]["labels"]),
            "old_G4": (g4_coeff, g4_labels),
        }
        for source, (coeff, labels) in populations.items():
            groups = [(-1, np.arange(len(labels), dtype=np.int64))]
            groups.extend((label, np.flatnonzero(labels == label)) for label in range(10))
            for label, indices in groups:
                z = coeff[indices]
                require(len(z) > 0 and np.isfinite(z).all(), f"coefficient moment population is empty: {source}/{label}")
                powers = np.abs(z)**2
                second, covariance, gauged = _complex_moment_matrices(z)
                mean = gauged.mean(axis=0)
                eigenvalues = np.maximum(np.linalg.eigvalsh(covariance).real[::-1], 0.0)
                total = float(eigenvalues.sum())
                squared = float(np.square(eigenvalues).sum())
                rank = total*total/squared if squared > 0 else 0.0
                cumulative = np.cumsum(eigenvalues) / total if total > 0 else np.zeros_like(eigenvalues)
                directions95 = int(np.searchsorted(cumulative, 0.95) + 1) if total > 0 else 0
                directions99 = int(np.searchsorted(cumulative, 0.99) + 1) if total > 0 else 0
                mean_power = powers.mean(axis=0)
                mean_total = float(powers.sum(axis=1).mean())
                sample_neff = np.divide(np.square(powers.sum(axis=1)), np.square(powers).sum(axis=1),
                                        out=np.zeros(len(powers)), where=np.square(powers).sum(axis=1) > 0)
                row_id = len(summaries)
                class_vectors[(grid, source, label)] = mean_power
                seconds.append(second); covariances.append(covariance); eigen_spectra.append(eigenvalues)
                summaries.append({"array_index": row_id, "source": source, "grid_role": role, "grid": grid,
                    "label": label, "n": len(z), "mean_total_ols_power": mean_total,
                    "mean_per_sample_power_participation": float(sample_neff.mean()),
                    "participation_of_mean_port_power": float(mean_power.sum()**2 / np.square(mean_power).sum())
                        if np.square(mean_power).sum() > 0 else 0.0,
                    "covariance_trace": total, "covariance_effective_rank": rank,
                    "covariance_directions_95": directions95, "covariance_directions_99": directions99,
                    "phase_gauge": "per-sample maximum-amplitude port made real-positive for moments only",
                    "coefficients_modified_for_decoder": False})
    for row in summaries:
        grid, label = row["grid"], row["label"]
        current = class_vectors[(grid, row["source"], label)]
        for target in ("E0_DEV", "EJ_DEV"):
            target_vector = class_vectors[(grid, target, label)]
            denominator = float(np.linalg.norm(target_vector))
            row[f"relative_mean_port_power_distance_to_{target}"] = (
                float(np.linalg.norm(current-target_vector)/denominator) if denominator > 0 else None)
    _write_rows(root / "tables" / "coefficient_moments_by_class.csv", summaries)
    _write_npz(root / "tables" / "coefficient_complex_moments.npz",
        complex_second_moment=np.stack(seconds).astype(np.complex128),
        complex_covariance=np.stack(covariances).astype(np.complex128),
        covariance_eigenvalues=np.stack(eigen_spectra).astype(np.float64),
        row_index=np.arange(len(summaries), dtype=np.int64))
    _write_json(root / "tables" / "coefficient_moment_index.json", {
        "entries": summaries, "complex_matrix_order": "E[a_i conjugate(a_j)]",
        "covariance": "Hermitian complex covariance after per-sample max-amplitude-port phase gauge",
        "old_G4_coefficients_reused_identically_across_grids": True,
        "interpretation_limit": "coefficient moments describe distribution differences, not causal decoder compatibility",
    })
    return {"groups": len(summaries), "matrix_rows": len(seconds),
            "complex_moment_dtype": "complex128"}


def _class_bootstrap(values: np.ndarray, labels: np.ndarray, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    require(values.shape == labels.shape and np.isfinite(values).all(), "class bootstrap input is incomplete")
    groups = [np.flatnonzero(labels == label) for label in range(10)]
    require(all(len(group) > 0 for group in groups), "class bootstrap population misses a digit")
    class_means = [float(values[index].mean()) for index in groups]
    point = float(np.mean(class_means))
    rng = np.random.default_rng(seed)
    draws = np.zeros(BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for group in groups:
        sampled = rng.integers(0, len(group), size=(BOOTSTRAP_RESAMPLES, len(group)))
        draws += values[group[sampled]].mean(axis=1) / 10.0
    return {"mean_equal_class": point, "ci95": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
            "n": len(values), "class_means": class_means,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": seed}


def _factorial_mse_effects(l00: np.ndarray, l01: np.ndarray,
                           l10: np.ndarray, l11: np.ndarray) -> dict[str, np.ndarray]:
    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in (l00, l01, l10, l11)]
    require(len({value.shape for value in arrays}) == 1 and all(np.isfinite(value).all() for value in arrays),
            "factorial MSE vectors must be finite and row-paired")
    old_encoder_old_decoder, old_encoder_new_decoder, new_encoder_old_decoder, new_encoder_new_decoder = arrays
    effects = {
        "encoder_average_mse_improvement": 0.5*((old_encoder_old_decoder-new_encoder_old_decoder)
                                                 +(old_encoder_new_decoder-new_encoder_new_decoder)),
        "decoder_average_mse_improvement": 0.5*((old_encoder_old_decoder-old_encoder_new_decoder)
                                                 +(new_encoder_old_decoder-new_encoder_new_decoder)),
        "encoder_improvement_with_D0_decoder": old_encoder_old_decoder-new_encoder_old_decoder,
        "encoder_improvement_with_J_decoder": old_encoder_new_decoder-new_encoder_new_decoder,
        "decoder_improvement_with_E0_encoder": old_encoder_old_decoder-old_encoder_new_decoder,
        "decoder_improvement_with_EJ_encoder": new_encoder_old_decoder-new_encoder_new_decoder,
        "interaction_L00_minus_L01_minus_L10_plus_L11":
            old_encoder_old_decoder-old_encoder_new_decoder-new_encoder_old_decoder+new_encoder_new_decoder,
        "native_mse_improvement_L00_minus_L11": old_encoder_old_decoder-new_encoder_new_decoder,
    }
    require(np.max(np.abs(effects["encoder_average_mse_improvement"]
                         + effects["decoder_average_mse_improvement"]
                         - effects["native_mse_improvement_L00_minus_L11"])) <= 1e-12,
            "factorial Encoder/Decoder contributions do not sum to the native MSE change")
    return effects


def _validate_native_equal_power_replay(
    dev: dict[tuple[str, int], dict[str, np.ndarray]], crossover_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_combo: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in crossover_rows:
        by_combo.setdefault((int(row["grid"]), str(row["combo"])), []).append(row)
    maximum_mse_error = 0.0
    maximum_ssim_error = 0.0
    maximum_clipping_error = 0.0
    maximum_camera_power_relative_error = 0.0
    compared_rows = 0
    for grid in (2048, 4096):
        d0 = dev[("D050000", grid)]
        j = dev[("J50000", grid)]
        for combo, endpoint, actual in (
            ("L00", "D050000", d0["reconstruction"]),
            ("L11", "J50000", j["equal_camera_power_reconstruction"]),
        ):
            current = sorted(by_combo.get((grid, combo), []), key=lambda item: int(item["row"]))
            require(len(current) == 5_000, f"equal-camera-power replay is incomplete: {combo}/{grid}")
            rows = np.asarray([int(item["row"]) for item in current], dtype=np.int64)
            labels = np.asarray([int(item["label"]) for item in current], dtype=np.int64)
            population = dev[(endpoint, grid)]
            require(np.array_equal(rows, population["rows"])
                    and np.array_equal(labels, population["labels"]),
                    f"equal-camera-power replay identity differs: {combo}/{grid}")
            expected = np.column_stack((
                [float(item["equal_power_mse"]) for item in current],
                [float(item["equal_power_ssim"]) for item in current],
                [float(item["equal_power_clipping_fraction"]) for item in current],
                [float(item["equal_camera_power_w"]) for item in current],
            ))
            require(actual.shape[0] == len(expected) and np.isfinite(actual[:, (0, 2, 3, 4)]).all(),
                    f"native equal-camera-power replay is invalid: {endpoint}/{grid}")
            target_power = d0["reconstruction"][:, 4]
            require(bool((target_power > 0.0).all()), "fixed DEV camera-power target must be positive")
            mse_error = float(np.max(np.abs(actual[:, 0] - expected[:, 0])))
            ssim_error = float(np.max(np.abs(actual[:, 2] - expected[:, 1])))
            clipping_error = float(np.max(np.abs(actual[:, 3] - expected[:, 2])))
            camera_error = float(np.max(np.abs(actual[:, 4] - expected[:, 3]) / target_power))
            maximum_mse_error = max(maximum_mse_error, mse_error)
            maximum_ssim_error = max(maximum_ssim_error, ssim_error)
            maximum_clipping_error = max(maximum_clipping_error, clipping_error)
            maximum_camera_power_relative_error = max(maximum_camera_power_relative_error, camera_error)
            compared_rows += len(current)
    require(maximum_mse_error <= 1.0e-6 and maximum_ssim_error <= 1.0e-6
            and maximum_clipping_error <= 1.0e-6
            and maximum_camera_power_relative_error <= 1.0e-4,
            "native equal-camera-power replay differs from the verified L00/L11 crossover")
    return {"status": "VERIFIED", "rows": compared_rows,
            "mse_absolute_tolerance": 1.0e-6, "maximum_mse_absolute_error": maximum_mse_error,
            "ssim_absolute_tolerance": 1.0e-6, "maximum_ssim_absolute_error": maximum_ssim_error,
            "clipping_absolute_tolerance": 1.0e-6, "maximum_clipping_absolute_error": maximum_clipping_error,
            "camera_power_relative_tolerance": 1.0e-4,
            "maximum_camera_power_relative_error": maximum_camera_power_relative_error,
            "negative_control": "real-valued raw_camera intensity is rejected by _equal_camera_readout"}


def _summarize_crossover(rows: list[dict[str, Any]], loss_rows: list[dict[str, Any]],
                         root: Path) -> dict[str, Any]:
    combos = ("L00", "L01", "L10", "L11")
    metrics = ("mse", "psnr_db", "ssim", "clipping_fraction", "camera_power_w",
               "equal_power_mse", "equal_power_ssim", "equal_power_clipping_fraction")
    organized: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        organized.setdefault((int(row["grid"]), row["combo"]), []).append(row)
    all_labels = None
    endpoint_summary: dict[str, Any] = {}
    grid_effects: dict[int, dict[str, np.ndarray]] = {}
    grid_rows: dict[int, np.ndarray] = {}
    class_rows, combo_rows = [], []
    for grid in (2048, 4096):
        data = {}
        shared_indices = None
        for combo in combos:
            current = sorted(organized[(grid, combo)], key=lambda item: int(item["row"]))
            require(len(current) == 5_000 and len({int(x["row"]) for x in current}) == 5_000,
                    f"crossover table is incomplete for {combo}/{grid}")
            labels = np.asarray([int(x["label"]) for x in current], dtype=np.int64)
            row_ids = np.asarray([int(x["row"]) for x in current], dtype=np.int64)
            grid_rows.setdefault(grid, row_ids)
            if all_labels is None:
                all_labels = labels
            else:
                require(np.array_equal(labels, all_labels) and np.array_equal(row_ids,
                    np.asarray([int(x["row"]) for x in sorted(organized[(grid, "L00")], key=lambda v: int(v["row"]))])),
                    "crossover row-label pairing changed")
            if shared_indices is None:
                shared_indices = _bootstrap_indices(labels)
            data[combo] = {name: np.asarray([float(x[name]) for x in current], dtype=np.float64)
                           for name in metrics}
            for name in metrics:
                values = data[combo][name]
                finite = np.isfinite(values)
                if finite.all():
                    stats = _paired_bootstrap(values, labels, shared_indices)
                else:
                    stats = {"mean": float(np.nanmean(values)), "n_defined": int(finite.sum()),
                             "undefined": int((~finite).sum()), "ci95": None}
                combo_rows.append({"grid": grid, "combo": combo, "metric": name,
                                   "mean": stats.get("mean"), "ci95": stats.get("ci95"),
                                   "n_defined": stats.get("n", stats.get("n_defined")),
                                   "undefined": stats.get("undefined", 0)})
            for label in range(10):
                idx = labels == label
                class_rows.append({"grid": grid, "combo": combo, "label": label, "n": int(idx.sum()),
                    **{name: float(np.nanmean(data[combo][name][idx])) for name in metrics}})
        effects = _factorial_mse_effects(*(data[name]["mse"] for name in combos))
        effects["equal_camera_power_mse_improvement_L00_minus_L11"] = (
            data["L00"]["equal_power_mse"]-data["L11"]["equal_power_mse"])
        shared_indices = _bootstrap_indices(labels)
        endpoint_summary[str(grid)] = {
            name: _paired_bootstrap(value, labels, shared_indices)
            for name, value in effects.items()
        }
        grid_effects[grid] = effects
    _write_rows(root / "tables" / "crossover_metrics_by_combo.csv", combo_rows)
    _write_rows(root / "tables" / "crossover_metrics_by_class.csv", class_rows)
    objective_rows = []
    for grid in (2048, 4096):
        for combo in combos:
            subset = [row for row in loss_rows if int(row["grid"]) == grid and row["combo"] == combo]
            require(len(subset) == 500, f"production objective batches incomplete: {grid}/{combo}")
            objective_rows.append({"grid": grid, "combo": combo, "batch_count": len(subset),
                **{name: float(np.mean([float(row[name]) for row in subset]))
                   for name in ("total", "dist", "rec", "op", "ssim_loss")}})
    _write_rows(root / "tables" / "crossover_production_objective_summary.csv", objective_rows)
    require(np.array_equal(grid_rows[2048], grid_rows[4096]), "DEV row identities differ across grids")
    shared_grid_indices = _bootstrap_indices(all_labels)
    diff = {name: _paired_bootstrap(grid_effects[2048][name]-grid_effects[4096][name],
                                    all_labels, shared_grid_indices)
            for name in grid_effects[2048]}
    return {"estimand": "frozen 2x2 Encoder/Decoder crossing on fixed DEV rows",
            "factorial_effects_by_grid": endpoint_summary,
            "production_minus_reference_grid_effect_sensitivity": diff,
            "decomposition_identity": "G_E + G_D = L00 - L11, verified per image on each grid",
            "rows": len(rows), "production_objective_batches": len(loss_rows),
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED}


def _summarize_g4(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["grid"]), row["endpoint"]), []).append(row)
    class_rows = []
    comparisons: dict[str, Any] = {}
    grid_row_ids: dict[int, list[int]] = {}
    for grid in (2048, 4096):
        data = {}
        for endpoint in ENDPOINTS:
            current = sorted(grouped[(grid, endpoint)], key=lambda item: int(item["cohort_row"]))
            require(len(current) == 10_000, f"saved G4 full population missing for {endpoint}/{grid}")
            ids = [int(row["cohort_row"]) for row in current]
            grid_row_ids.setdefault(grid, ids)
            data[endpoint] = current
            for label in range(10):
                subset = [row for row in current if int(row["requested_label"]) == label]
                require(len(subset) == 1000, f"G4 class {label} population changed for {endpoint}/{grid}")
                record: dict[str, Any] = {"grid": grid, "endpoint": endpoint, "label": label, "n": len(subset)}
                for name in ("ed_correct", "ec_correct", "camera_power_w", "ols_output_power_w",
                             "ols_mean_intensity_geometric_w_cm2", "ols_mean_intensity_effective_w_cm2",
                             "proxy_clipping_fraction"):
                    record[name] = float(np.mean([float(item[name]) for item in subset]))
                valid = [item for item in subset if item["equal_camera_power_defined"]]
                record["equal_camera_power_defined_n"] = len(valid)
                for name in ("equal_camera_power_ed_correct", "equal_camera_power_ec_correct"):
                    record[name] = float(np.mean([float(item[name]) for item in valid])) if valid else math.nan
                class_rows.append(record)
        d0, jj = data["D050000"], data["J50000"]
        require(all(a["cohort_row"] == b["cohort_row"] and a["requested_label"] == b["requested_label"]
                    and a["proposal_id"] == b["proposal_id"] for a, b in zip(d0, jj, strict=True)),
                "G4 D0/J proposal pairing changed")
        labels = np.asarray([int(row["requested_label"]) for row in d0], dtype=np.int64)
        sample_indices = _bootstrap_indices(labels)
        endpoints = {}
        for endpoint, current in (("D050000", d0), ("J50000", jj)):
            endpoint_result = {}
            for classifier in ("ed", "ec"):
                values = np.asarray([float(row[f"{classifier}_correct"]) for row in current])
                endpoint_result[f"{classifier}_macro_recall"] = _paired_bootstrap(values, labels, sample_indices)
                defined = np.asarray([bool(row["equal_camera_power_defined"]) for row in current])
                equal_values = np.asarray([float(row[f"equal_camera_power_{classifier}_correct"])
                                           for row in current if row["equal_camera_power_defined"]])
                if defined.all():
                    endpoint_result[f"{classifier}_equal_power_macro_recall"] = _paired_bootstrap(
                        equal_values, labels, sample_indices)
                else:
                    endpoint_result[f"{classifier}_equal_power_macro_recall"] = {
                        "status": "PARTIALLY_UNDEFINED", "n_defined": int(defined.sum()),
                        "mean_defined": float(equal_values.mean()) if len(equal_values) else None, "ci95": None}
            endpoints[endpoint] = endpoint_result
        paired = {}
        for metric in ("ed_correct", "ec_correct", "equal_camera_power_ed_correct",
                       "equal_camera_power_ec_correct", "camera_power_w", "ols_output_power_w",
                       "ols_mean_intensity_effective_w_cm2", "proxy_clipping_fraction"):
            if metric.startswith("equal_"):
                valid = np.asarray([a["equal_camera_power_defined"] and b["equal_camera_power_defined"]
                                    for a, b in zip(d0, jj, strict=True)], dtype=bool)
                left = np.asarray([float(a[metric]) for a, ok in zip(d0, valid, strict=True) if ok])
                right = np.asarray([float(b[metric]) for b, ok in zip(jj, valid, strict=True) if ok])
                paired_labels = labels[valid]
                if not valid.all():
                    paired[metric] = {"status": "PARTIALLY_UNDEFINED", "n_defined": int(valid.sum()),
                                      "effect_J_minus_D0": float(np.mean(right-left)) if len(left) else None,
                                      "ci95": None}
                    continue
            else:
                left = np.asarray([float(row[metric]) for row in d0])
                right = np.asarray([float(row[metric]) for row in jj])
                paired_labels = labels
            paired[metric] = _paired_bootstrap(right-left, paired_labels,
                                               sample_indices)
            paired[metric]["effect"] = "J_minus_D0"
        comparisons[str(grid)] = {"per_endpoint": endpoints, "paired_effects": paired,
            "interpretation": "same saved G4 coefficients and proposal IDs entered the two frozen Decoders"}
    require(grid_row_ids[2048] == grid_row_ids[4096], "G4 row order differs across grids")
    _write_rows(root / "tables" / "old_g4_metrics_by_class.csv", class_rows)
    return {"rows": len(rows), "per_grid": comparisons, "all_proposal_identities_paired": True,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED}


def _summarize_phase(root: Path) -> dict[str, Any]:
    routes = ("D0_DEV", "J_DEV", "G4_to_D0", "G4_to_J")
    primary_metrics = ("delta_mse", "delta_ssim", "delta_ed_correct", "delta_ec_correct",
                      "delta_camera_power_w", "delta_ols_output_power_w",
                      "delta_ols_effective_density_w_m2", "delta_clipping_fraction")
    summaries, class_rows = [], []
    image_effects: dict[tuple[str, int, str, float, str], tuple[np.ndarray, np.ndarray]] = {}

    def mean_repeats(values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values)
        total = np.where(valid, values, 0.0).sum(axis=0)
        count = valid.sum(axis=0)
        return np.divide(total, count, out=np.full(values.shape[1], np.nan), where=count > 0)

    for route in routes:
        for role, grid in GRIDS:
            directory = root / "populations" / "phase" / route / str(grid)
            with np.load(directory / "condition_00.npz", allow_pickle=False) as base_npz:
                rows = base_npz["rows"].copy(); labels = base_npz["labels"].copy()
                proposals = base_npz["proposal_ids"].copy(); baseline_m = base_npz["metrics"].copy()
                baseline_r = base_npz["readout"].copy(); baseline_ed = base_npz["ed_prediction"].copy()
                baseline_ec = base_npz["ec_prediction"].copy()
            conditions: dict[tuple[str, float], list[dict[str, np.ndarray]]] = {}
            for index in range(1, 65):
                with np.load(directory / f"condition_{index:02d}.npz", allow_pickle=False) as current:
                    require(np.array_equal(current["rows"], rows) and np.array_equal(current["labels"], labels)
                            and np.array_equal(current["proposal_ids"], proposals),
                            f"phase condition identity changed during summary: {route}/{grid}/{index}")
                    family = str(current["family"].item()); sigma = float(current["sigma"])
                    conditions.setdefault((family, sigma), []).append({
                        "readout": current["readout"].copy(), "metrics": current["metrics"].copy(),
                        "ed": current["ed_prediction"].copy(), "ec": current["ec_prediction"].copy(),
                    })
            require(len(conditions) == 8 and all(len(items) == 8 for items in conditions.values()),
                    f"phase repeated-condition matrix changed: {route}/{grid}")
            for (family, sigma), reps in sorted(conditions.items()):
                readout = np.stack([item["readout"] for item in reps])
                metric = np.stack([item["metrics"] for item in reps])
                ed = np.stack([item["ed"] for item in reps])
                ec = np.stack([item["ec"] for item in reps])
                ed_correct = (ed == labels[None, :]).astype(np.float64)
                ec_correct = (ec == labels[None, :]).astype(np.float64)
                values = {
                    "delta_mse": mean_repeats(readout[:, :, 0]-baseline_r[None, :, 0]),
                    "delta_ssim": mean_repeats(readout[:, :, 2]-baseline_r[None, :, 2]),
                    "delta_ed_correct": np.mean(ed_correct, axis=0) - (baseline_ed == labels),
                    "delta_ec_correct": np.mean(ec_correct, axis=0) - (baseline_ec == labels),
                    "delta_camera_power_w": np.mean(readout[:, :, 4], axis=0)-baseline_r[:, 4],
                    "delta_ols_output_power_w": np.mean(metric[:, :, 0, 0], axis=0)-baseline_m[:, 0, 0],
                    "delta_ols_effective_density_w_m2": np.mean(metric[:, :, 0, 4], axis=0)-baseline_m[:, 0, 4],
                    "delta_clipping_fraction": np.mean(readout[:, :, 3], axis=0)-baseline_r[:, 3],
                }
                for name in primary_metrics:
                    current = values[name]
                    if np.isfinite(current).all():
                        stats = _class_bootstrap(current, labels)
                        mean = stats["mean_equal_class"]
                        interval = stats["ci95"]
                    else:
                        defined = np.isfinite(current)
                        stats = None; mean = float(np.nanmean(current)) if defined.any() else None; interval = None
                    summaries.append({"route": route, "data_role": "fixed_DEV" if route.endswith("DEV") else "saved_G4",
                        "grid_role": role, "grid": grid, "family": family, "relative_phase_rms_rad": sigma,
                        "metric": name, "mean_change": mean, "ci95": interval,
                        "n_defined": int(np.isfinite(current).sum()), "repeats_per_image": len(reps),
                        "bootstrap_resamples": BOOTSTRAP_RESAMPLES if stats else 0,
                        "bootstrap_seed": BOOTSTRAP_SEED if stats else None})
                    if name in {"delta_mse", "delta_ssim", "delta_ed_correct", "delta_ec_correct"}:
                        for label in range(10):
                            mask = labels == label
                            class_rows.append({"route": route, "grid": grid, "family": family,
                                "relative_phase_rms_rad": sigma, "metric": name, "label": label,
                                "n": int(mask.sum()),
                                "mean_change": float(np.nanmean(current[mask])) if np.isfinite(current[mask]).any() else math.nan})
                    image_effects[(route, grid, family, sigma, name)] = (current, labels)
    _write_rows(root / "tables" / "phase_sensitivity_summary.csv", summaries)
    _write_rows(root / "tables" / "phase_sensitivity_by_class.csv", class_rows)
    cross_grid = []
    for route in routes:
        for family in ("independent_port", "linear_ramp"):
            for sigma in (0.01, 0.03, 0.1, 0.3):
                for name in primary_metrics:
                    prod, labels = image_effects[(route, 2048, family, sigma, name)]
                    ref, ref_labels = image_effects[(route, 4096, family, sigma, name)]
                    require(np.array_equal(labels, ref_labels), "phase grids use different sample identities")
                    diff = prod-ref
                    if np.isfinite(diff).all():
                        stats = _class_bootstrap(diff, labels)
                        cross_grid.append({"route": route, "family": family, "relative_phase_rms_rad": sigma,
                            "metric": name, "production_minus_reference_change": stats["mean_equal_class"],
                            "ci95": json.dumps(stats["ci95"]), "n": len(diff)})
                    else:
                        cross_grid.append({"route": route, "family": family, "relative_phase_rms_rad": sigma,
                            "metric": name, "production_minus_reference_change": math.nan,
                            "ci95": None, "n": int(np.isfinite(diff).sum())})
    _write_rows(root / "tables" / "phase_sensitivity_cross_grid.csv", cross_grid)
    return {"summary_rows": len(summaries), "class_rows": len(class_rows),
            "cross_grid_rows": len(cross_grid), "representative_images_per_route": 200,
            "repeats_aggregated_within_image": 8}


def _summarize_native_dev(dev: dict[tuple[str, int], dict[str, np.ndarray]], root: Path
                          ) -> dict[str, Any]:
    state_rows, class_rows = [], []
    effects: dict[int, dict[str, np.ndarray]] = {}
    effect_summary_by_grid: dict[str, dict[str, Any]] = {}
    grid_rows: dict[int, np.ndarray] = {}
    metric_names = ("mse", "psnr_db", "ssim", "clipping_fraction", "camera_power_w")
    for role, grid in GRIDS:
        d0, jj = dev[("D050000", grid)], dev[("J50000", grid)]
        require(np.array_equal(d0["rows"], jj["rows"]) and np.array_equal(d0["labels"], jj["labels"]),
                "D0/J fixed DEV rows or labels differ")
        rows = d0["rows"].astype(np.int64); labels = d0["labels"].astype(np.int64)
        grid_rows[grid] = rows
        indices = _bootstrap_indices(labels)
        for endpoint, data in (("D050000", d0), ("J50000", jj)):
            values = data["reconstruction"]
            for index, name in enumerate(metric_names):
                stats = _paired_bootstrap(values[:, index], labels, indices)
                state_rows.append({"grid": grid, "endpoint": endpoint, "metric": name,
                    "mean": stats["mean"], "ci95": json.dumps(stats["ci95"]), "n": len(values)})
            for label in range(10):
                mask = labels == label
                class_rows.append({"grid": grid, "endpoint": endpoint, "label": label, "n": int(mask.sum()),
                    **{name: float(np.mean(values[mask, index])) for index, name in enumerate(metric_names)},
                    "ols_output_power_w": float(data["metrics"][mask, 6, 0].mean()),
                    "ols_effective_area_m2": float(data["metrics"][mask, 6, 2].mean()),
                    "ols_effective_density_w_m2": float(data["metrics"][mask, 6, 4].mean())})
        delta = {name: jj["reconstruction"][:, index]-d0["reconstruction"][:, index]
                 for index, name in enumerate(metric_names)}
        delta["equal_camera_power_mse"] = jj["equal_camera_power_reconstruction"][:, 0]-d0["reconstruction"][:, 0]
        delta["equal_camera_power_ssim"] = jj["equal_camera_power_reconstruction"][:, 2]-d0["reconstruction"][:, 2]
        for name, values in delta.items():
            stats = _paired_bootstrap(values, labels, indices) if np.isfinite(values).all() else {
                "mean": float(np.nanmean(values)), "ci95": None, "n": int(np.isfinite(values).sum())}
            state_rows.append({"grid": grid, "endpoint": "J_minus_D0", "metric": name,
                "mean": stats["mean"], "ci95": json.dumps(stats["ci95"]) if stats["ci95"] else None,
                "n": stats["n"]})
        for plane_index, plane in enumerate(DEV_PLANES):
            for metric_index, metric_name in ((0, "power_w"), (2, "effective_area_m2"),
                                               (3, "mean_intensity_geometric_w_m2"),
                                               (4, "mean_intensity_effective_w_m2"),
                                               (5, "effective_to_geometric_area_ratio")):
                difference = jj["metrics"][:, plane_index, metric_index]-d0["metrics"][:, plane_index, metric_index]
                stats = _paired_bootstrap(difference, labels, indices)
                state_rows.append({"grid": grid, "endpoint": "J_minus_D0", "metric": f"{plane}:{metric_name}",
                    "mean": stats["mean"], "ci95": json.dumps(stats["ci95"]), "n": stats["n"]})
        pos = ((jj["metrics"][:, 6, 0] > 0) & (d0["metrics"][:, 6, 0] > 0)
               & (jj["metrics"][:, 6, 2] > 0) & (d0["metrics"][:, 6, 2] > 0))
        delta_log_i = np.log(jj["metrics"][pos, 6, 4]/d0["metrics"][pos, 6, 4])
        delta_log_p = np.log(jj["metrics"][pos, 6, 0]/d0["metrics"][pos, 6, 0])
        delta_log_a = np.log(jj["metrics"][pos, 6, 2]/d0["metrics"][pos, 6, 2])
        require(np.max(np.abs(delta_log_i-(delta_log_p-delta_log_a))) < 1e-10,
                "OLS output log-density decomposition failed")
        pos_labels = labels[pos]
        log_decomposition = {
            "positive_power_samples": int(pos.sum()),
            "delta_log_effective_density": _class_bootstrap(delta_log_i, pos_labels),
            "delta_log_power": _class_bootstrap(delta_log_p, pos_labels),
            "minus_delta_log_effective_area": _class_bootstrap(-delta_log_a, pos_labels),
            "max_identity_residual": float(np.max(np.abs(delta_log_i-(delta_log_p-delta_log_a)))),
        }
        effects[grid] = {**delta, "log_density": delta_log_i, "log_power": delta_log_p,
                         "minus_log_area": -delta_log_a}
        effect_summary_by_grid[str(grid)] = {
            name: (_paired_bootstrap(values, labels, indices) if np.isfinite(values).all() else {
                "mean": float(np.nanmean(values)), "ci95": None, "n": int(np.isfinite(values).sum())})
            for name, values in delta.items()}
        state_rows.append({"grid": grid, "endpoint": "J_minus_D0", "metric": "ols_log_density_decomposition",
                           "mean": log_decomposition["delta_log_effective_density"]["mean_equal_class"],
                           "ci95": json.dumps(log_decomposition["delta_log_effective_density"]["ci95"]),
                           "n": int(pos.sum())})
        effects[grid]["log_decomposition"] = log_decomposition
    require(np.array_equal(grid_rows[2048], grid_rows[4096]), "DEV row identities differ between grids")
    _write_rows(root / "tables" / "native_dev_metrics_summary.csv", state_rows)
    _write_rows(root / "tables" / "native_dev_metrics_by_class.csv", class_rows)
    index_2048 = _bootstrap_indices(dev[("D050000", 2048)]["labels"])
    labels = dev[("D050000", 2048)]["labels"]
    grid_sensitivity = {}
    for name in metric_names:
        diff = effects[2048][name]-effects[4096][name]
        grid_sensitivity[name] = _paired_bootstrap(diff, labels, index_2048)
    return {"rows_per_endpoint_grid": 5_000, "fixed_dev_paired": True,
            "effect_J_minus_D0_by_grid": {str(grid): {**effect_summary_by_grid[str(grid)],
                "ols_log_density_decomposition": effects[grid]["log_decomposition"]}
                for grid in (2048, 4096)},
            "production_minus_reference_effect_sensitivity": grid_sensitivity,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES, "bootstrap_seed": BOOTSTRAP_SEED}


def _historical_j_quality(fixed: dict[str, Any]) -> dict[str, Any]:
    path = _resolve(fixed["config"]["inputs"]["joint_optical_dir"]) / "machine_report.json"
    require(path.is_file(), f"historical J report is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    require(report.get("task_id") == joint.TASK_ID and report.get("training", {}).get("global_step") == 50_000,
            "historical J report identity changed")
    j = report["endpoints"]["J"]
    generation, reconstruction = j["generation"], j["reconstruction"]
    thresholds = report["historical_system_quality_checklist"]["thresholds"]
    ed_per_class = generation["ed_recall_by_class"]
    ec_per_class = generation["ec_recall_by_class"]
    measured = {
        "E_D_macro_recall": generation["ed_macro_recall"],
        "E_C_macro_recall": generation["ec_macro_recall"],
        "E_D_min_class_recall": min(ed_per_class),
        "E_D_min_class_label": int(np.argmin(ed_per_class)),
        "E_C_min_class_recall": min(ec_per_class),
        "E_C_min_class_label": int(np.argmin(ec_per_class)),
        "proxy_clipping_fraction_macro": generation["proxy_clipping_area_fraction_macro"],
        "reconstruction_psnr_db_mean": reconstruction["psnr_db_mean"],
        "reconstruction_ssim_mean": reconstruction["ssim_mean"],
        "generated_to_dev_tv_ratio_min": generation["generated_to_dev_tv_ratio_min"],
    }
    correct = {
        "E_D_macro_recall": measured["E_D_macro_recall"] >= thresholds["generation_macro_recall_min"],
        "E_C_macro_recall": measured["E_C_macro_recall"] >= thresholds["generation_macro_recall_min"],
        "E_D_min_class_recall": measured["E_D_min_class_recall"] >= thresholds["generation_recall_per_class_min"],
        "E_C_min_class_recall": measured["E_C_min_class_recall"] >= thresholds["generation_recall_per_class_min"],
        "proxy_clipping_fraction_macro": measured["proxy_clipping_fraction_macro"] <= thresholds["proxy_clipping_fraction_max"],
        "reconstruction_psnr_db_mean": measured["reconstruction_psnr_db_mean"] >= thresholds["reconstruction_psnr_db_min"],
        "reconstruction_ssim_mean": measured["reconstruction_ssim_mean"] >= thresholds["reconstruction_ssim_min"],
        "generated_to_dev_tv_ratio_min": measured["generated_to_dev_tv_ratio_min"] >= thresholds["generated_to_dev_tv_ratio_min"],
    }
    return {"source_report": str(path), "historical_status_preserved": report["historical_system_quality_checklist"]["status"],
            "measured": measured, "thresholds": thresholds, "meets_threshold": correct,
            "unmet_items": [key for key, passed in correct.items() if not passed],
            "correction": "E_D macro recall 0.8873 is above 0.80; the prior prose claim that it was below 0.80 was incorrect.",
            "source_file_modified": False}


def _export_machine_tables(root: Path, crossover: list[dict[str, Any]], loss_rows: list[dict[str, Any]],
                           g4_rows: list[dict[str, Any]]) -> dict[str, int]:
    require(len(crossover) == 40_000 and len(loss_rows) == 4_000 and len(g4_rows) == 40_000,
            "machine table population is incomplete before export")
    _write_rows(root / "tables" / "encoder_decoder_crossover_per_image.csv.gz", crossover, compressed=True)
    _write_rows(root / "tables" / "crossover_production_loss_batches.csv", loss_rows)
    _write_rows(root / "tables" / "old_g4_decoder_metrics_per_image.csv.gz", g4_rows, compressed=True)
    return {"crossover_per_image_rows": len(crossover), "production_loss_batch_rows": len(loss_rows),
            "old_g4_decoder_rows": len(g4_rows)}


def _validate_complete(root: Path, fixed: dict[str, Any], density_counts: dict[str, int],
                       table_counts: dict[str, int], phase_receipt: dict[str, Any],
                       posthoc: list[dict[str, Any]], dev: dict[Any, Any],
                       crossover_summary: dict[str, Any], g4_summary: dict[str, Any],
                       equal_power_replay: dict[str, Any],
                       evaluator_before: tuple[dict[str, Any], dict[str, Any]],
                       evaluator_after: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
    require(density_counts == {"native_full_population_rows": 360_000, "j_posthoc_probe_rows": 1_200},
            f"whole-plane density table counts incomplete: {density_counts}")
    require(table_counts == {"crossover_per_image_rows": 40_000, "production_loss_batch_rows": 4_000,
                             "old_g4_decoder_rows": 40_000}, f"exported table counts incomplete: {table_counts}")
    require(phase_receipt.get("status") == "VERIFIED" and phase_receipt.get("rows") == 105_600,
            "phase measurement population is incomplete")
    require(len(posthoc) == 6 and sum(item["rows"]*item["planes"] for item in posthoc) == 1_200,
            "J historical posthoc probe coverage is incomplete")
    require(all(len(value.get("rows", [])) == 5_000 for value in dev.values()),
            "frozen DEV endpoint population incomplete")
    require(crossover_summary["rows"] == 40_000 and g4_summary["rows"] == 40_000,
            "crossed or G4 summary population is incomplete")
    require(equal_power_replay.get("status") == "VERIFIED" and equal_power_replay.get("rows") == 20_000,
            "native equal-camera-power crossover verification is incomplete")
    require(base._assert_close(evaluator_before[0], evaluator_after[0], "J frozen E_D") == 0.0
            and base._assert_close(evaluator_before[1], evaluator_after[1], "J frozen E_C") == 0.0,
            "a frozen confirmation evaluator changed during diagnostic")
    require(len(list((root / "populations" / "phase").glob("*/*/condition_*.npz"))) == 528,
            "phase chunk inventory is incomplete")
    for endpoint in ENDPOINTS:
        for role, grid in GRIDS:
            arrays = _load_chunks(_cache_dir(root, endpoint, role), fixed["dev_rows"].numpy(), fixed["labels"])
            require(len(arrays["rows"]) == 5_000 and arrays["metrics"].shape == (5_000, 10, 7),
                    f"DEV table recovery consumer failed: {endpoint}/{grid}")
    return {"status": "VERIFIED", "whole_plane_density_rows": 360_000,
            "j_posthoc_probe_rows": 1_200, "crossover_rows": 40_000,
            "production_loss_batches": 4_000, "old_g4_rows": 40_000,
            "phase_condition_rows": 105_600, "phase_chunks": 528,
            "fixed_dev_endpoint_grid_populations": 4,
            "evaluator_freeze": "PASSED", "official_test_reads": 0,
            "scientific_updates": 0, "optimizers_created": 0,
            "native_equal_camera_power_rows_verified": equal_power_replay["rows"],
            "native_equal_camera_power_crosscheck": equal_power_replay["status"]}


def _write_report(root: Path, cfg: dict[str, Any], preflight: dict[str, Any], fixed: dict[str, Any],
                  density_counts: dict[str, int], density_summary: dict[str, int],
                  posthoc: list[dict[str, Any]], native: dict[str, Any], crossover: dict[str, Any],
                  g4: dict[str, Any], phase: dict[str, Any], coefficient: dict[str, Any],
                  table_counts: dict[str, int], complete: dict[str, Any], probe_coverage: dict[str, Any]) -> dict[str, Any]:
    historical_quality = _historical_j_quality(fixed)
    report = {
        "schema": "gaussian_j_frozen_attribution_phase_density_report.v1", "task_id": TASK_ID,
        "status": "COMPLETE_FROZEN_DIAGNOSTIC", "measurement_complete": True,
        "semantic_authority_id": AUTHORITY, "input_identity": {
            "checkpoints": {key: {"global_step": int(value["completed_global_step"]),
                                  "run_id": value.get("run_id"),
                                  "saved_adam_steps": [int(torch.as_tensor(item["step"]).item())
                                      for item in value.get("optimizer", {}).get("state", {}).values()]}
                             for key, value in fixed["checkpoints"].items()},
            "fixed_DEV_rows": 5_000, "DEV_rows_per_class": 500,
            "saved_G4_coefficients": 10_000, "G4_rows_per_class": 1_000,
            "G4_proposal_identity_preserved": True, "grids": {"production": 2048, "reference": 4096},
            "new_prior_samples": 0, "official_test_reads": 0,
        },
        "frozen_boundary": {"scientific_updates": 0, "optimizer_creation": 0, "prior_fitting": 0,
            "prior_sampling": 0, "official_test_reads": 0, "material_deployment": 0,
            "human_review": "not performed by plan"},
        "preflight": preflight["checks"], "completion_consumer": complete,
        "native_equal_camera_power_crosscheck": native.get("equal_camera_power_crosscheck"),
        "whole_plane_density": {**density_counts, "summary": density_summary,
            "units": ["W/m2", "W/cm2"], "surfaces": DEV_PLANES,
            "reconstruction_quality": native},
        "posthoc_J_density_probes": {"records": posthoc, "rows": 1_200,
            "missing_historical_500_step_checkpoints": "not reconstructed; only saved 45k, 47.5k, and 50k J states were replayed"},
        "future_J_training_probe_integration": {"coverage_contract": probe_coverage,
            "runtime_training_updates_in_this_task": 0,
            "probe_frequency": "source/resume checkpoint and every 500-update saved checkpoint",
            "historical_logs_rewritten": False},
        "encoder_decoder_frozen_attribution": crossover,
        "old_G4_compatibility": g4,
        "coefficient_distribution": coefficient,
        "phase_sensitivity": phase,
        "historical_quality_report_correction": historical_quality,
        "tables": table_counts,
        "interpretation_limits": [
            "D0/J encoder-decoder effects are frozen factorial contrasts on one fixed DEV split, not independently trained interventions.",
            "Old G4 compatibility compares a fixed saved coefficient cohort; coefficient-moment distance alone is not causal evidence.",
            "Phase amplitudes are deterministic conditional perturbations, not a measured hardware drift distribution.",
            "2048/4096 differences are reported as grid sensitivity and do not establish continuous-grid convergence.",
            "No material response, physical camera calibration, official TEST result, or new human readability result is measured.",
            "Project geometry remains PROJECT_GEOMETRY_UNMEASURED; this numerical diagnosis does not qualify hardware deployment.",
        ],
    }
    _write_json(root / "machine_report.json", report)
    report_lines = [
        "# Gaussian J冻结归因、整面功率密度与相位敏感性诊断",
        "",
        "## 结果与状态",
        "",
        f"状态：{report['status']}。D0/J固定DEV、旧G4、相位扰动和整面功率密度记录均通过完整性消费者。训练更新、优化器创建、先验采样、官方TEST读取与材料部署均为0。",
        "",
        f"整面密度表包含{density_counts['native_full_population_rows']:,}条完整输入记录，J历史检查点事后冻结探针包含{density_counts['j_posthoc_probe_rows']:,}条；全部单位同时保留W/m²与W/cm²。",
        "",
        f"旧报告纠正：J的E_D宏召回率为{historical_quality['measured']['E_D_macro_recall']:.4f}，高于0.80；E_D最低类别为{historical_quality['measured']['E_D_min_class_label']}类、{historical_quality['measured']['E_D_min_class_recall']:.4f}。历史VALID_FAILED状态未改。",
        "",
        "## 科学认识",
        "",
        "冻结2×2归因、等相机功率对照、旧G4相容性、复系数矩与相位条件曲线详见machine_report.json及tables/。跨网格差异作为敏感性报告，不从较有利网格单独挑选结论。各端口和整面功率的变化分开记录。",
        "",
        "## 目标缺口",
        "",
        "历史J只保存45k、47.5k、50k检查点，因此未保存的500步历史状态没有密度记录；本轮只对存在的检查点回放。真实硬件相位漂移、相机噪声标定、材料响应曲线和材料面物理照明阈值仍缺。",
        "",
        "## 工程完成与验证",
        "",
        "预检逐格比较同网格复场、功率与直算/合成重构MSE；D0旧缓存只在行号、标签、系数、功率与读出重放通过后复用。完成消费者核对360,000条整面记录、1,200条历史探针、40,000条交叉记录、4,000个生产损失批次、40,000条G4记录和105,600条相位记录。",
        "",
        "## 决策影响与下一步",
        "",
        "结果用于选择下一项冻结研究，不构成模型采用或自动训练授权。J训练入口已接入起点/恢复点/每500步固定DEV探针与完成覆盖消费者；本轮未运行训练。旧G4、Decoder和Encoder贡献按实测保留，不能把相关性表述成独立干预因果。",
        "",
        "历史质量清单实际结果及所有逐图/逐类结果见machine_report.json、tables/whole_plane_density_per_image.csv.gz、tables/encoder_decoder_crossover_per_image.csv.gz、tables/old_g4_decoder_metrics_per_image.csv.gz与tables/phase_sensitivity_per_image.csv.gz。",
    ]
    (root / "中文诊断报告.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return report


def _run(cfg: dict[str, Any], root: Path, preflight_dir: Path) -> dict[str, Any]:
    require(torch.cuda.is_available(), "frozen J full measurement requires CUDA")
    require(preflight_dir.is_dir(), f"explicit READY preflight directory is missing: {preflight_dir}")
    fixed = _load_fixed(cfg, torch.device("cuda"))
    if (root / "machine_report.json").is_file():
        completed_report = json.loads((root / "machine_report.json").read_text(encoding="utf-8"))
        require(completed_report.get("status") == "COMPLETE_FROZEN_DIAGNOSTIC"
                and completed_report.get("completion_consumer", {}).get("status") == "VERIFIED",
                "existing output is not a complete recoverable result")
        return completed_report
    control, identity = _open_attempt(root, preflight_dir)
    preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    require(preflight.get("status") == "READY" and preflight.get("checks", {}).get("status") == "PASSED",
            "same-plan verified preflight is required before the full frozen measurement")
    _write_json(root / "task_config.json", cfg)
    progress_path = root / "progress.json"
    _write_json(progress_path, {"task_id": TASK_ID, "run_id": identity.run_id,
        "stage": "FIXED_INPUTS_LOADED", "scientific_updates": 0, "official_test_reads": 0})
    bases = {(key, grid): _build_basis(key, grid, root, fixed)
             for _role, grid in GRIDS for key in ENDPOINTS}
    evaluator = base._load_evaluator(_resolve(fixed["task_config"]["evaluation"]["evaluator_checkpoint"]), fixed["device"])
    ec_evaluator, _qualification = decoder_ref._load_ec_evaluator(
        _resolve(fixed["task_config"]["evaluation"]["ec_checkpoint"]), fixed["device"])
    evaluator_before = (base._state_snapshot(evaluator), base._state_snapshot(ec_evaluator))

    posthoc = _posthoc_j_probes(root, fixed)
    _write_json(progress_path, {"task_id": TASK_ID, "run_id": identity.run_id,
        "stage": "POSTHOC_J_DENSITY_PROBES_COMPLETE", "rows": 1_200,
        "scientific_updates": 0, "official_test_reads": 0})
    dev: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    reuse_rows = preflight["checks"].get("reusable_d0_cache_replays", [])
    reuse_by_grid = {int(row["grid"]): row.get("status") == "PASSED" for row in reuse_rows}
    for role, grid in GRIDS:
        d0_meta = _measure_dev("D050000", role, grid, root, fixed, None, reuse_d0=reuse_by_grid.get(grid, False))
        d0 = _load_chunks(_cache_dir(root, "D050000", role), fixed["dev_rows"].numpy(), fixed["labels"])
        dev[("D050000", grid)] = d0_meta | d0
        _measure_dev("J50000", role, grid, root, fixed, d0)
        jdata = _load_chunks(_cache_dir(root, "J50000", role), fixed["dev_rows"].numpy(), fixed["labels"])
        dev[("J50000", grid)] = {"endpoint": "J50000", "grid": grid, "grid_role": role,
                                  "rows": 5_000, "chunks": 25} | jdata
        _write_json(progress_path, {"task_id": TASK_ID, "run_id": identity.run_id,
            "stage": "NATIVE_DEV", "grid": grid, "completed_endpoints": ["D050000", "J50000"],
            "scientific_updates": 0, "official_test_reads": 0})
    crossover_rows, loss_rows = _measure_crossover(root, fixed, bases, dev)
    equal_power_replay = _validate_native_equal_power_replay(dev, crossover_rows)
    native_summary = _summarize_native_dev(dev, root)
    native_summary["equal_camera_power_crosscheck"] = equal_power_replay
    crossover_summary = _summarize_crossover(crossover_rows, loss_rows, root)
    g4_rows = _measure_g4(root, fixed, bases)
    g4_summary = _summarize_g4(g4_rows, root)
    density_counts = _write_density_tables(root, fixed, posthoc)
    density_summary = _summarize_density(root)
    coefficient_summary = _coefficient_moments(root, fixed, dev)
    phase_receipt = _phase_measure(root, cfg, fixed, bases, dev)
    phase_csv_rows = _phase_to_csv(root)
    require(phase_csv_rows == phase_receipt["rows"], "phase CSV consumer row count differs from chunks")
    phase_summary = _summarize_phase(root)
    table_counts = _export_machine_tables(root, crossover_rows, loss_rows, g4_rows)
    evaluator_after = (base._state_snapshot(evaluator), base._state_snapshot(ec_evaluator))
    complete = _validate_complete(root, fixed, density_counts, table_counts, phase_receipt,
                                  posthoc, dev, crossover_summary, g4_summary,
                                  equal_power_replay,
                                  evaluator_before, evaluator_after)
    coverage = {
        "status": "CODE_PATH_INTEGRATED_NOT_RUN_IN_THIS_FROZEN_TASK",
        "consumer": "future J machine report requires the 45,000-to-50,000 step coverage contract",
        "expected_future_training_steps": list(range(45_000, 50_001, 500)),
        "helper": "base._record_gaussian_plane_density_probe",
        "current_task_scientific_updates": 0,
    }
    report = _write_report(root, cfg, preflight, fixed, density_counts, density_summary,
        posthoc, native_summary, crossover_summary, g4_summary, phase_summary,
        coefficient_summary, table_counts, complete, coverage)
    _write_json(root / "run_state.json", {"status": "COMPLETE_FROZEN_DIAGNOSTIC",
        "task_id": TASK_ID, "run_id": identity.run_id, "scientific_updates": 0,
        "optimizer_creation": 0, "official_test_reads": 0, "completion_consumer": complete})
    control.complete()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    parser.add_argument("--preflight-dir", type=Path)
    args = parser.parse_args()
    cfg_path = _resolve(args.config)
    cfg = _read_config(cfg_path)
    root = _resolve(cfg["run_dir"])
    if args.mode == "preflight":
        require(args.preflight_dir is None, "preflight mode does not accept a prior preflight directory")
        stage = root.parent / f"{root.name}_preflight"
        if stage.exists():
            stage = root.parent / f"{root.name}_preflight_{uuid.uuid4().hex[:8]}"
        preflight = _preflight(cfg, stage, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(json.dumps({"status": preflight["status"], "directory": str(stage),
                          "device": preflight["device"]}, ensure_ascii=False), flush=True)
        return
    require(args.preflight_dir is not None, "run mode requires the explicit READY preflight directory")
    report = _run(cfg, root, _resolve(args.preflight_dir))
    print(json.dumps({"status": report["status"], "report": str(root / "machine_report.json"),
                      "density_rows": report["whole_plane_density"]["native_full_population_rows"],
                      "crossover_rows": report["tables"]["crossover_per_image_rows"],
                      "phase_rows": report["phase_sensitivity"]["summary_rows"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()



