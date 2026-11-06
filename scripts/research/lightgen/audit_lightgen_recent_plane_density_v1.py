"""Audit recent frozen Gaussian endpoints with full-plane power density.

This is a diagnostic-only path. It reuses saved real-image coefficients for
Decoder-only endpoints, coherently synthesizes linear Decoder fields, and does
not create an optimizer, draw a prior sample, or update model state.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ldm_project.v2.evidence.io import diagnostic_json_value as clean
from ldm_project.v2.evidence.io import write_diagnostic_json as write_json
from scripts.research.lightgen import closeout_gaussian_ols_plane_power_density_v1 as density_closeout
from scripts.research.lightgen import diagnose_lightgen_gaussian_effective_channel_dual_grid_v1 as dual_grid
from scripts.research.lightgen import diagnose_lightgen_gaussian_real_input_power_attribution_v1 as previous
from scripts.research.lightgen import execute_lightgen_matched_architecture_v1 as base
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    ComplexOLSState,
)
from ldm_project.v2.application.research.lightgen.plane_power_density import (
    field_intensity_distribution,
    field_plane_metrics,
    small_image_result_fields,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.application.research.lightgen.image_readout import resize_view, ssim_batch


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "GAUSSIAN_RECENT_PLANE_DENSITY_AUDIT_V1"
AUTHORITY = "USER_APPROVED_GAUSSIAN_RECENT_PLANE_DENSITY_AUDIT_20261022"
GRIDS = (("production", 2048), ("reference", 4096))
DEV_PLANES = density_closeout.SMALL_PLANES
GENERATION_PLANES = (
    "fiber_output_and_decoder_entry",
    "decoder_layer_1_aperture",
    "decoder_layer_2_aperture_and_exit",
    "camera_physical_roi",
)
REC_FIELDS = ("mse", "psnr_db", "ssim", "clipping_fraction", "camera_power_w")
DIST_FIELDS = (
    "maximum_intensity_w_m2",
    "area_weighted_q50_w_m2",
    "area_weighted_q90_w_m2",
    "area_weighted_q99_w_m2",
    "power_weighted_q50_w_m2",
    "power_weighted_q90_w_m2",
    "power_weighted_q99_w_m2",
)
ENDPOINTS = ("C45000", "V45000", "R51245000", "R52245000", "D050000", "Dgrad50000")
UPSTREAM_ENDPOINTS = ("C45000", "V45000", "R51245000", "R52245000")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    with np.load(path, allow_pickle=False) as saved:
        require(saved.files == list(arrays), f"chunk key order/readback changed: {path}")
        for key, value in arrays.items():
            equal_nan = saved[key].dtype.kind in "fc" and np.asarray(value).dtype.kind in "fc"
            require(np.array_equal(saved[key], value, equal_nan=equal_nan), f"chunk readback changed: {path}:{key}")


def read_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(cfg, dict) and cfg.get("schema") == "gaussian_recent_plane_density_audit.v1",
            "recent plane-density config schema changed")
    require(cfg.get("task_id") == TASK_ID and cfg.get("semantic_authority_id") == AUTHORITY,
            "recent plane-density task identity changed")
    require(cfg["run_dir"] == "outputs/20261022_gaussian_ols_training_plane_density_audit_v1/runs/attempt_01",
            "new diagnostic output path changed")
    require(cfg["training_updates"] == cfg["optimizer_creation"] == cfg["prior_sampling"] == 0
            and cfg["official_test_reads"] == cfg["hardware_operations"] == 0,
            "zero-update diagnostic boundary changed")
    require([(item["key"], item["global_step"], item["completed_updates"]) for item in cfg["endpoints"]] == [
        ("C45000", 45000, 10000), ("V45000", 45000, 10000),
        ("R51245000", 45000, 45000), ("R52245000", 45000, 45000),
        ("D050000", 50000, 5000), ("Dgrad50000", 50000, 5000),
    ], "checkpoint endpoint population changed")
    require([item["key"] for item in cfg["endpoints"]] == list(ENDPOINTS), "endpoint ordering changed")
    require(cfg["model"]["production_grid"] == 2048 and cfg["model"]["reference_grid"] == 4096
            and cfg["model"]["ols_basis"] == "gaussian_s12_v1"
            and cfg["model"]["shared_camera_gain_w_inverse"] == 1.505046959946815e9,
            "optical grid, OLS model, or camera gain changed")
    require(cfg["generation"]["methods"] == ["G1", "G4"] and cfg["generation"]["stream"] == 0
            and cfg["generation"]["samples"] == 10000
            and cfg["statistics"]["bootstrap_resamples"] == 2000
            and cfg["statistics"]["bootstrap_seed"] == 2026102101,
            "generation cohort or paired statistics changed")
    require(cfg["execution"]["no_training_updates"] and cfg["execution"]["no_optimizer_creation"]
            and cfg["execution"]["no_prior_sampling"] and cfg["execution"]["official_test_reads"] == 0
            and cfg["execution"]["hardware_operations"] == 0,
            "execution boundary changed")
    return cfg


def _adam_steps(optimizer: Any) -> list[int]:
    require(isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict), "saved optimizer state is missing")
    result: list[int] = []
    for state in optimizer["state"].values():
        require(isinstance(state, dict) and "step" in state, "saved Adam state has no step counter")
        value = state["step"]
        result.append(int(value.item()) if isinstance(value, torch.Tensor) else int(value))
    require(bool(result), "saved Adam state contains no initialized parameters")
    return result


def _validate_proposal_population(proposal_ids: torch.Tensor, labels: torch.Tensor) -> bool:
    if not isinstance(proposal_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return False
    if proposal_ids.shape != (10000,) or labels.shape != (10000,):
        return False
    if not torch.equal(torch.bincount(labels.long(), minlength=10), torch.full((10,), 1000)):
        return False
    return all(torch.unique(proposal_ids[labels.long() == label]).numel() == 1000 for label in range(10))


def _identity_config(spec: dict[str, Any], cp: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / spec["identity_config"]
    require(path.is_file(), f"explicit endpoint identity config missing: {path}")
    source = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(source, dict) and source.get("task_id") == spec["task_id"],
            f"endpoint identity config task changed: {spec['key']}")
    task_id = spec["task_id"]
    if task_id == "LIGHTGEN_SMALL_GAUSSIAN_REPLICATION_PRIOR_CONFIRMATION_V1":
        optics = source["optics"]
        runtime_path = ROOT / source["runtime_config"]
        require(runtime_path.is_file(), f"explicit replication runtime config missing: {runtime_path}")
        runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
        small = runtime["optics"]["small"]
        configured_grid = small["production_propagation_grid"]
        require(source.get("schema") == "lightgen_small_gaussian_replication_prior_confirmation.v1"
                and source.get("runtime_config") == cfg["model"]["baseline_config"]
                and source.get("execution_source_semantics_id") == "small_gaussian_replication_prior_confirmation_v1"
                and optics.get("basis") == cfg["model"]["ols_basis"]
                and optics.get("production_grid") == configured_grid == cfg["model"]["production_grid"]
                and optics.get("reference_grid") == cfg["model"]["reference_grid"]
                and optics.get("input_power_w") == cfg["model"]["source_power_w"]
                and optics.get("layer_power_transmission") == cfg["model"]["layer_power_transmission"]
                and small.get("ols_basis") == optics["basis"]
                and small.get("ols_transport") == optics["transport"]
                and small.get("port_pitch_m") == optics["port_pitch_m"]
                and small.get("field_1e_radius_m") == optics["field_1e_radius_m"]
                and runtime["optics"]["wavelength_m"] == optics["wavelength_m"]
                and source["data"]["training_file"] == cfg["data"]["training_file"]
                and source["data"]["split_seed"] == cfg["data"]["split_seed"]
                and source["data"]["official_test_reads"] == 0,
                f"replication task/runtime configuration differs from the fixed diagnostic inputs: {spec['key']}")
        phase_shape = optics["phase_shape"]
        expected_model_identity = {
            "encoder": [optics["encoder_layers"], *phase_shape], "ols": optics["basis"],
            "ols_geometry": {"port_pitch_m": optics["port_pitch_m"],
                             "field_1e_radius_m": optics["field_1e_radius_m"],
                             "transport": optics["transport"]},
            "decoder": [optics["decoder_layers"], *phase_shape],
            "phase_shape": [2, 1, *phase_shape],
            "trainable_phase_parameters": optics["trainable_phase_parameters"],
            "geometry_status": optics["geometry_status"],
        }
        require(cp.get("schema") == "lightgen_small_gaussian_replication_checkpoint.v1"
                and cp.get("model_identity") == expected_model_identity,
                f"replication checkpoint schema or full model identity changed: {spec['key']}")
        identity_basis = "explicit replication task config -> runtime_config optics.small.production_propagation_grid"
    else:
        expected_schema = {
            "LIGHTGEN_SMALL_GAUSSIAN_FOREGROUND_BALANCE_AB_V1": "lightgen_small_gaussian_foreground_balance_checkpoint.v1",
            "LIGHTGEN_SMALL_DECODER_NATIVE_GRADIENT_AB_V1": "lightgen_small_decoder_native_gradient_checkpoint.v1",
        }[task_id]
        require(cp.get("schema") == expected_schema and cp.get("config_identity") == source,
                f"checkpoint schema or embedded task config changed: {spec['key']}")
        configured_grid = source["optics"]["small"]["production_propagation_grid"]
        require(cp.get("model_identity") == source["source"]["model_identity"],
                f"checkpoint model identity differs from the task config: {spec['key']}")
        identity_basis = "checkpoint embedded task config -> optics.small.production_propagation_grid"
    require(configured_grid == 2048, f"source task config production grid changed: {spec['key']}")
    require(cp.get("propagation_grid_size", configured_grid) == configured_grid,
            f"checkpoint grid field conflicts with the source task config: {spec['key']}")
    return {
        "identity_config_path": str(path),
        "production_grid": configured_grid,
        "grid_identity_basis": identity_basis,
        "checkpoint_schema": cp.get("schema"),
        "checkpoint_embeds_config": isinstance(cp.get("config_identity"), dict),
    }


def load_fixed_inputs(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    power_cfg = previous.load_config(ROOT / "configs/research/lightgen/gaussian_real_input_power_attribution_v1.yaml")
    baseline_cfg = previous.read_yaml(ROOT / cfg["model"]["baseline_config"])
    pixels, labels, train_rows, dev_rows = base._load_data(ROOT / cfg["data"]["training_file"])
    require(cfg["data"]["split_seed"] == 1234 and len(dev_rows) == 5000
            and torch.equal(torch.bincount(labels[dev_rows], minlength=10), torch.full((10,), 500))
            and not bool(torch.isin(train_rows, dev_rows).any()), "fixed DEV split or roles changed")
    endpoint_specs: dict[str, dict[str, Any]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for spec in cfg["endpoints"]:
        key = spec["key"]
        path = ROOT / spec["checkpoint"]
        require(path.is_file(), f"explicit checkpoint missing: {path}")
        cp = torch.load(path, map_location="cpu", weights_only=False)
        identity = _identity_config(spec, cp, cfg)
        require(cp.get("task_id") == spec["task_id"] and cp.get("seed") == spec["seed"]
                and cp.get("completed_updates") == spec["completed_updates"]
                and cp.get("model_identity", {}).get("ols") == "gaussian_s12_v1"
                and cp.get("model_identity", {}).get("ols_geometry", {}).get("transport") == "identity"
                and identity["production_grid"] == 2048,
                f"checkpoint task, seed, OLS identity, or source-config grid changed: {key}")
        require(int(cp.get("official_test_reads", 0)) == 0,
                f"checkpoint records an unexpected official TEST read: {key}")
        global_step = cp.get("completed_global_step", cp.get("completed_updates"))
        require(global_step == spec["global_step"], f"checkpoint global step changed: {key}")
        require(cp.get("branch") == spec["branch"], f"checkpoint branch changed: {key}")
        steps = _adam_steps(cp.get("optimizer"))
        if key in ("C45000", "V45000", "R51245000", "R52245000"):
            require(set(steps) == {45000}, f"saved Adam step does not match the frozen 45k endpoint: {key}")
        else:
            require(set(steps) == {45000, 50000}, f"saved Adam step does not match the frozen 50k Decoder endpoint: {key}")
            require(cp.get("source_checkpoint_step") == 45000 and cp.get("schedule_start_global_step") == 45000,
                    f"Decoder endpoint source lineage changed: {key}")
        require(math.isclose(float(cp["camera_gain"].reshape(-1)[0]),
                             cfg["model"]["shared_camera_gain_w_inverse"], rel_tol=1e-12),
                f"fixed digital camera gain changed: {key}")
        require(isinstance(cp.get("model"), dict), f"checkpoint model state missing: {key}")
        endpoint_specs[key] = {**spec, **identity, "checkpoint_path": str(path), "optimizer_steps": steps,
                               "run_id": cp.get("run_id"), "source_run_id": cp.get("source_run_id")}
        checkpoints[key] = cp
    require(checkpoints["D050000"].get("source_run_id") == checkpoints["C45000"].get("run_id")
            and checkpoints["Dgrad50000"].get("source_run_id") == checkpoints["C45000"].get("run_id"),
            "D0/Dgrad do not descend from the selected C45k endpoint")
    require(checkpoints["C45000"]["model_identity"] == checkpoints["D050000"]["model_identity"]
            and checkpoints["C45000"]["model_identity"] == checkpoints["Dgrad50000"]["model_identity"],
            "common C/D model identity changed")
    camera = yaml.safe_load((ROOT / cfg["statistics"]["camera_proxy_config"]).read_text(encoding="utf-8"))["camera"]
    require(camera == {
        "quantum_efficiency": 0.5, "read_noise_e_rms": 1.0,
        "dark_current_e_per_pixel_s": 0.0, "full_well_e": 30000.0,
        "exposure_sweep_s": [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
    }, "proxy-noise file is not the selected v10 camera contract")
    return {
        "power_cfg": power_cfg, "baseline_cfg": baseline_cfg, "pixels": pixels,
        "labels": labels, "train_rows": train_rows, "dev_rows": dev_rows,
        "endpoint_specs": endpoint_specs, "checkpoints": checkpoints,
        "camera_proxy": camera, "device": device,
    }


def load_generation_inputs(cfg: dict[str, Any], fixed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = ROOT / cfg["generation"]["source_root"]
    cohorts: dict[str, dict[str, Any]] = {}
    sources = {
        "C45000": "C_seed_2026101501",
        "R51245000": "seed_2026102151",
        "R52245000": "seed_2026102152",
    }
    for endpoint, folder in sources.items():
        for method in cfg["generation"]["methods"]:
            path = root / folder / f"{method}_stream_0.pt"
            require(path.is_file(), f"explicit saved generation cohort is missing: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            require(payload.get("task_id") == cfg["generation"]["source_task_id"]
                    and payload.get("schema") == "lightgen_small_conditional_mixture_generated_cohort.v1"
                    and payload.get("global_step") == 45000 and payload.get("method") == method
                    and payload.get("stream") == 0 and payload.get("model") == folder
                    and payload.get("classifier_based_sample_selection") is False
                    and payload.get("per_image_normalization") is False,
                    f"saved generation metadata changed: {endpoint}/{method}")
            coefficients = payload.get("ols_coefficients")
            labels = payload.get("requested_labels")
            require(isinstance(coefficients, torch.Tensor) and coefficients.shape == (10000, 100)
                    and coefficients.is_complex() and bool(torch.isfinite(coefficients).all())
                    and isinstance(labels, torch.Tensor) and labels.shape == (10000,)
                    and torch.equal(torch.bincount(labels.long(), minlength=10), torch.full((10,), 1000)),
                    f"saved generation cohort shape/class population changed: {endpoint}/{method}")
            require(_validate_proposal_population(payload.get("proposal_ids"), labels),
                    f"saved generation proposal IDs are not unique within each requested class: {endpoint}/{method}")
            cohorts[f"{endpoint}_{method}"] = {
                "path": str(path), "coefficients": coefficients.detach().cpu().numpy().astype(np.complex64),
                "labels": labels.detach().cpu().numpy().astype(np.int64),
                "proposal_ids": payload["proposal_ids"].detach().cpu().numpy().astype(np.int64),
                "component_ids": payload["component_ids"].detach().cpu().numpy().astype(np.int64),
                "method": method, "model": folder, "global_step": 45000,
            }
    droot = ROOT / cfg["generation"]["decoder_gradient_source_root"]
    d0 = torch.load(droot / "D0/generated_cohort.pt", map_location="cpu", weights_only=False)
    dgrad = torch.load(droot / "Dgrad/generated_cohort.pt", map_location="cpu", weights_only=False)
    c_g4 = cohorts["C45000_G4"]
    for branch, payload in (("D0", d0), ("Dgrad", dgrad)):
        require(payload.get("task_id") == "LIGHTGEN_SMALL_DECODER_NATIVE_GRADIENT_AB_V1"
                and payload.get("global_step") == 50000 and payload.get("branch") == branch
                and np.array_equal(payload["ols_coefficients"].detach().cpu().numpy(), c_g4["coefficients"])
                and np.array_equal(payload["proposal_ids"].detach().cpu().numpy(), c_g4["proposal_ids"])
                and np.array_equal(payload["requested_labels"].detach().cpu().numpy(), c_g4["labels"]),
                f"{branch} does not use the exact saved C G4 stream-0 coefficients")
    require(torch.equal(d0["ols_coefficients"], dgrad["ols_coefficients"])
            and torch.equal(d0["proposal_ids"], dgrad["proposal_ids"]),
            "D0/Dgrad generated cohorts are not identical")
    for branch in ("D0", "Dgrad"):
        cohorts[f"{branch}50000_G4"] = {
            **c_g4, "endpoint": f"{branch}50000", "global_step": 50000,
            "source_note": "same C45000 G4 stream-0 coefficients; D0/Dgrad proposal identities verified",
        }
    return cohorts


def _model_for(endpoint: str, grid: int, fixed: dict[str, Any]) -> torch.nn.Module:
    cp = fixed["checkpoints"][endpoint]
    model = base._model("small", grid, fixed["device"], fixed["baseline_cfg"])
    model.load_state_dict(cp["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require(not model.training and not any(parameter.requires_grad for parameter in model.parameters()),
            f"frozen model load failed: {endpoint}/{grid}")
    return model


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _assert_snapshot(model: torch.nn.Module, snapshot: dict[str, torch.Tensor], name: str) -> None:
    state = model.state_dict()
    require(state.keys() == snapshot.keys(), f"frozen state keys changed: {name}")
    for key, value in state.items():
        require(torch.equal(value.detach().cpu(), snapshot[key]), f"frozen parameter/buffer changed: {name}.{key}")


def _small_decoder_fields(model: torch.nn.Module, coefficients: torch.Tensor) -> list[tuple[str, ComplexField]]:
    runtime = model.runtime
    state = ComplexOLSState(
        coefficients=coefficients,
        source="saved_frozen_cohort_or_encoder_projection",
        basis_id=runtime.ols.basis_id,
        mode_geometry=runtime.ols.mode_identity(),
    )
    generator_input = runtime.w_ols(state)
    generator_output, records, _ = runtime.generator(generator_input, "lightgen_generator")
    generator_output = ComplexField(
        generator_output.tensor, runtime.generator_exit_plane,
        OperationProvenance("lightgen_generator_exit", generator_output.plane.plane_id, None,
                            {"generator_layers": 2}),
    )
    camera_field = runtime.output_relay(generator_output)
    require(len(records) == 2, "Gaussian decoder layer count changed")
    return [
        (GENERATION_PLANES[0], generator_input),
        (GENERATION_PLANES[1], records[0]),
        (GENERATION_PLANES[2], records[1]),
        (GENERATION_PLANES[3], camera_field),
    ]


def _build_or_load_basis(endpoint: str, grid: int, root: Path, fixed: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    path = root / "bases" / f"{endpoint}_{grid}_decoder_planes.npz"
    model = _model_for(endpoint, grid, fixed)
    snapshot = _snapshot(model)
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            require(str(saved["endpoint"].item()) == endpoint and int(saved["grid"].item()) == grid
                    and int(saved["global_step"].item()) == fixed["endpoint_specs"][endpoint]["global_step"],
                    f"saved Decoder basis identity changed: {path}")
            arrays = {name: saved[f"field_{index}"].copy() for index, name in enumerate(GENERATION_PLANES)}
            geometry = json.loads(str(saved["geometry_json"].item()))
        require(all(value.shape[0] == 100 and value.dtype == np.complex64 and np.isfinite(value).all()
                    for value in arrays.values()), f"saved Decoder basis is invalid: {path}")
        _assert_snapshot(model, snapshot, f"{endpoint}/{grid}/basis-read")
        del model
        return arrays, geometry

    pieces: dict[str, list[np.ndarray]] = {name: [] for name in GENERATION_PLANES}
    geometry: list[dict[str, Any]] | None = None
    with torch.inference_mode():
        for start in range(0, 100, 8):
            ports = torch.arange(start, min(100, start + 8), device=fixed["device"])
            one_hot = torch.zeros((len(ports), 100), dtype=torch.complex64, device=fixed["device"])
            one_hot[torch.arange(len(ports), device=fixed["device"]), ports] = 1.0 + 0.0j
            fields = _small_decoder_fields(model, one_hot)
            current_geometry: list[dict[str, Any]] = []
            for name, field in fields:
                current_geometry.append({"plane_key": name, **density_closeout.plane_geometry(field.plane),
                                         "wavelength_m": float(field.plane.wavelength_m)})
                pieces[name].append(field.tensor.detach().to(torch.complex64).cpu().numpy()[:, 0])
            geometry = current_geometry if geometry is None else geometry
    arrays = {name: np.concatenate(values, axis=0).astype(np.complex64, copy=False)
              for name, values in pieces.items()}
    require(geometry is not None and all(value.shape[0] == 100 and np.isfinite(value).all()
                                         for value in arrays.values()), f"Decoder basis build failed: {endpoint}/{grid}")
    write_npz(path, endpoint=np.asarray(endpoint), grid=np.asarray(grid, dtype=np.int64),
              global_step=np.asarray(fixed["endpoint_specs"][endpoint]["global_step"], dtype=np.int64),
              field_0=arrays[GENERATION_PLANES[0]], field_1=arrays[GENERATION_PLANES[1]],
              field_2=arrays[GENERATION_PLANES[2]], field_3=arrays[GENERATION_PLANES[3]],
              geometry_json=np.asarray(json.dumps(clean(geometry), ensure_ascii=False)))
    _assert_snapshot(model, snapshot, f"{endpoint}/{grid}/basis-build")
    del model
    return arrays, geometry


def _distribution_array(field: ComplexField) -> np.ndarray:
    rows = field_intensity_distribution(field)
    return np.asarray([[np.nan if row[key] is None else float(row[key]) for key in DIST_FIELDS]
                       for row in rows], dtype=np.float64)


def _metrics_from_fields(fields: list[tuple[str, ComplexField]], plane_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    require(tuple(name for name, _ in fields) == plane_names, "physical plane ordering changed")
    metrics: list[np.ndarray] = []
    geometries: list[dict[str, Any]] = []
    distribution = np.full((fields[0][1].tensor.shape[0], len(DIST_FIELDS)), np.nan, dtype=np.float64)
    for name, field in fields:
        values, geometry = field_plane_metrics(field)
        metrics.append(values)
        geometry = {"plane_key": name, **geometry,
                    "alias": "OLS modal re-emission and Decoder entrance are the same declared plane"
                    if name == "fiber_output_and_decoder_entry" else None}
        geometries.append(geometry)
        if name == "fiber_output_and_decoder_entry":
            distribution = _distribution_array(field)
    return np.stack(metrics, axis=1), distribution, geometries


def _reconstruction_metrics(
    camera_field: ComplexField, pixels: torch.Tensor, rows: np.ndarray,
    gain: float, power_scale: torch.Tensor | None = None,
) -> tuple[np.ndarray, torch.Tensor]:
    intensity = camera_field.tensor.abs().square()
    pixel_area = float(math.prod(camera_field.plane.pixel_pitch_m))
    raw = intensity * pixel_area
    camera_power = raw.reshape(raw.shape[0], -1).sum(dim=1)
    unclipped = raw * float(gain)
    if power_scale is not None:
        unclipped = unclipped * power_scale.to(device=unclipped.device, dtype=unclipped.dtype).reshape(-1, 1, 1, 1)
    reconstruction = resize_view(unclipped)
    target = pixels.index_select(0, torch.as_tensor(rows, dtype=torch.long)).unsqueeze(1).to(
        device=reconstruction.device, dtype=reconstruction.dtype) / 255.0
    mse = (reconstruction - target).square().mean((1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp_min(1.0e-12))
    ssim = ssim_batch(reconstruction, target)
    clipping = ((unclipped < 0.0) | (unclipped > 1.0)).to(torch.float64).mean((1, 2, 3))
    if power_scale is not None:
        camera_power = camera_power * power_scale.to(device=camera_power.device, dtype=camera_power.dtype)
    output = torch.stack((mse, psnr, ssim, clipping, camera_power), dim=1).detach().cpu().numpy().astype(np.float64)
    return output, reconstruction.detach()


def _synthesize_metrics(
    endpoint: str, coefficients: np.ndarray, basis: dict[str, np.ndarray],
    geometry: list[dict[str, Any]], model: torch.nn.Module, device: torch.device,
    pixels: torch.Tensor | None = None, rows: np.ndarray | None = None,
    gain: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, torch.Tensor | None, list[dict[str, Any]]]:
    plane_lookup = {item["plane_key"]: item for item in geometry}
    metrics = np.empty((len(coefficients), len(GENERATION_PLANES), len(density_closeout.METRIC_FIELDS)), dtype=np.float64)
    distributions = np.full((len(coefficients), len(DIST_FIELDS)), np.nan, dtype=np.float64)
    rec_parts: list[np.ndarray] = []
    camera_fields: list[torch.Tensor] = []
    coeff = torch.as_tensor(coefficients, dtype=torch.complex64, device=device)
    basis_tensors = {name: torch.as_tensor(value, dtype=torch.complex64, device=device) for name, value in basis.items()}
    plane_specs = {name: _field_plane_from_geometry(model, plane_lookup[name]) for name in GENERATION_PLANES}
    with torch.inference_mode():
        for begin in range(0, len(coefficients), 8):
            end = min(len(coefficients), begin + 8)
            z = coeff[begin:end]
            for plane_index, name in enumerate(GENERATION_PLANES):
                field_tensor = torch.einsum("bi,ihw->bhw", z, basis_tensors[name]).unsqueeze(1)
                field = ComplexField(field_tensor, plane_specs[name],
                                     OperationProvenance("coherent_saved_port_response_sum", plane_specs[name].plane_id, None))
                values, _ = field_plane_metrics(field)
                metrics[begin:end, plane_index] = values
                if name == "fiber_output_and_decoder_entry":
                    distributions[begin:end] = _distribution_array(field)
                if name == "camera_physical_roi":
                    camera_fields.extend(field_tensor.unbind(0))
                    if pixels is not None:
                        require(rows is not None and gain is not None, "reconstruction inputs are incomplete")
                        rec, _ = _reconstruction_metrics(field, pixels, rows[begin:end], gain)
                        rec_parts.append(rec)
            del z
    reconstruction = np.concatenate(rec_parts) if rec_parts else None
    camera_tensor = torch.stack(camera_fields, dim=0) if camera_fields else None
    require(np.isfinite(metrics[:, :, (0, 1, 3, 6)]).all(), f"nonfinite synthesized plane metrics: {endpoint}")
    return metrics, distributions, reconstruction, camera_tensor, [plane_lookup[name] for name in GENERATION_PLANES]


def _field_plane_from_geometry(model: torch.nn.Module, item: dict[str, Any]):
    """Resolve persisted geometry to the production PlaneSpec objects."""
    require("wavelength_m" in item, f"saved plane geometry omits wavelength: {item['plane_key']}")
    plane = PlaneSpec(
        plane_id=item["plane_id"], shape=tuple(int(value) for value in item["shape"]),
        pixel_pitch_m=tuple(float(value) for value in item["pixel_pitch_m"]),
        wavelength_m=float(item["wavelength_m"]), z_m=float(item["z_m"]),
        coordinate_frame=item["coordinate_frame"],
        origin_m=tuple(float(value) for value in item["origin_m"]),
    )
    return plane


def _native_image_metrics(model: torch.nn.Module, result: Any, pixels: torch.Tensor, rows: np.ndarray, gain: float) -> np.ndarray:
    del model
    raw_w_per_pixel = result.physical_sensor_output_w_per_pixel
    require(raw_w_per_pixel is not None and result.physical_power_trace is not None,
            "direct physical camera output is missing")
    unclipped = raw_w_per_pixel * float(gain)
    prediction = resize_view(unclipped)
    target = pixels.index_select(0, torch.as_tensor(rows, dtype=torch.long)).unsqueeze(1).to(
        device=prediction.device, dtype=prediction.dtype) / 255.0
    mse = (prediction - target).square().mean((1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp_min(1.0e-12))
    ssim = ssim_batch(prediction, target)
    clipping = ((unclipped < 0.0) | (unclipped > 1.0)).to(torch.float64).mean((1, 2, 3))
    camera_power = result.physical_power_trace["P_camera_W"].to(torch.float64)
    return torch.stack((mse, psnr, ssim, clipping, camera_power), dim=1).detach().cpu().numpy().astype(np.float64)


def _save_dev_chunk(path: Path, payload: dict[str, np.ndarray]) -> None:
    write_npz(path, **payload)


def _dev_paths(root: Path, endpoint: str, grid_role: str) -> Path:
    return root / "populations" / "dev" / endpoint / grid_role


def _chunk_path(directory: Path, index: int) -> Path:
    return directory / f"chunk_{index:04d}.npz"


def _fixed_rows(labels: torch.Tensor, dev_rows: torch.Tensor, per_class: int) -> list[int]:
    selected = [int(row) for label in range(10) for row in dev_rows[labels[dev_rows] == label][:per_class].tolist()]
    require(len(selected) == 10 * per_class and len(set(selected)) == len(selected), "fixed per-class DEV probe changed")
    return selected


def _field_relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    a, b = candidate.to(torch.complex128), reference.to(torch.complex128)
    denominator = torch.linalg.vector_norm(b)
    require(float(denominator) > 0.0, "direct field replay produced zero reference field")
    return float((torch.linalg.vector_norm(a - b) / denominator).detach().cpu())


def _independent_numpy_check(field: ComplexField, metrics: np.ndarray) -> float:
    values = field.tensor.detach().cpu().numpy()
    values_128 = values.astype(np.complex128)
    intensity = values_128.real ** 2 + values_128.imag ** 2
    area = float(math.prod(field.plane.pixel_pitch_m))
    power = intensity.reshape(len(values), -1).sum(axis=1) * area
    error = np.abs(power - metrics[:, 0]) / np.maximum(np.abs(power), np.finfo(np.float64).tiny)
    return float(error.max())


def _identity_preflight(fixed: dict[str, Any], cohorts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checkpoints = fixed["checkpoints"]
    c_state = checkpoints["C45000"]["model"]
    d0_state = checkpoints["D050000"]["model"]
    dg_state = checkpoints["Dgrad50000"]["model"]
    require(c_state.keys() == d0_state.keys() == dg_state.keys(), "C/D checkpoint state keys differ")
    common = [key for key in c_state if not key.startswith("runtime.generator.phase")]
    require(all(torch.equal(c_state[key], d0_state[key]) and torch.equal(c_state[key], dg_state[key]) for key in common),
            "D0/Dgrad changed the declared frozen C encoder or OLS state")
    require(not torch.equal(d0_state["runtime.generator.phase"], dg_state["runtime.generator.phase"]),
            "D0 and Dgrad Decoder phases unexpectedly match")
    require(np.array_equal(cohorts["D050000_G4"]["coefficients"], cohorts["Dgrad50000_G4"]["coefficients"]),
            "D0/Dgrad saved G4 coefficients differ")
    return {
        "status": "VERIFIED",
        "C45000_D0_Dgrad_shared_state_keys": len(common),
        "only_expected_decoder_phase_differs_between_D0_Dgrad": True,
        "C_to_D0_shared_encoder_and_OLS_exact": True,
        "saved_G4_coefficients_and_proposal_ids_match": True,
        "checkpoint_adam_steps": {key: fixed["endpoint_specs"][key]["optimizer_steps"] for key in ENDPOINTS},
        "official_test_reads": 0,
    }


def _build_basis_and_validate(cfg: dict[str, Any], fixed: dict[str, Any], root: Path) -> dict[str, Any]:
    basis_info: dict[str, Any] = {}
    probe_rows = _fixed_rows(fixed["labels"], fixed["dev_rows"], 2)
    maximum_complex_error = 0.0
    maximum_numpy_integral_error = 0.0
    maximum_reconstruction_mse_error = 0.0
    maximum_power_trace_error = 0.0
    representative_maps_root = root / "figures"
    representative_maps_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for grid_role, grid in GRIDS:
        for endpoint in ENDPOINTS:
            basis, geometry = _build_or_load_basis(endpoint, grid, root, fixed)
            model = _model_for(endpoint, grid, fixed)
            model_snapshot = _snapshot(model)
            row_tensor = torch.as_tensor(probe_rows, dtype=torch.long)
            images = image_batch(fixed["pixels"], row_tensor, "small", fixed["device"])
            direct_parts: list[tuple[np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray, np.ndarray]] = []
            map_parts: dict[str, list[np.ndarray]] = {name: [] for name in DEV_PLANES}
            with torch.inference_mode():
                for index in range(len(probe_rows)):
                    row = np.asarray([probe_rows[index]], dtype=np.int64)
                    result = model.runtime.forward_from_image(images[index:index + 1])
                    fields, trace = small_image_result_fields(model, result)
                    metrics, dist, direct_geometry = _metrics_from_fields(fields, DEV_PLANES)
                    for name, field in fields:
                        map_parts[name].append(field.tensor[0].abs().square().squeeze().detach().to(torch.float32).cpu().numpy())
                    maximum_numpy_integral_error = max(maximum_numpy_integral_error,
                                                       *[_independent_numpy_check(field, metrics[:, j])
                                                         for j, (_, field) in enumerate(fields)])
                    expected = density_closeout._small_expected_trace(trace, result)
                    for plane_index, plane_name in enumerate(DEV_PLANES):
                        reference_power = expected[plane_name].to(torch.float64).detach().cpu().numpy()
                        relative = np.abs(metrics[:, plane_index, 0] - reference_power) / np.maximum(
                            np.abs(reference_power), np.finfo(np.float64).tiny)
                        maximum_power_trace_error = max(maximum_power_trace_error, float(relative.max()))
                    decoder_fields = _small_decoder_fields(model, result.z_ols.coefficients)
                    for (name, candidate), (reference_name, reference) in zip(decoder_fields, fields[6:], strict=True):
                        require(name == reference_name, "Decoder basis plane order changed")
                        maximum_complex_error = max(maximum_complex_error,
                                                    _field_relative_error(candidate.tensor, reference.tensor))
                    synth_metrics, _, synth_rec, _, _ = _synthesize_metrics(
                        endpoint, result.z_ols.coefficients.detach().cpu().numpy(), basis, geometry,
                        model, fixed["device"], fixed["pixels"], row,
                        float(fixed["checkpoints"][endpoint]["camera_gain"].reshape(-1)[0]),
                    )
                    direct_rec = _native_image_metrics(model, result, fixed["pixels"], row,
                                                       float(fixed["checkpoints"][endpoint]["camera_gain"].reshape(-1)[0]))
                    assert synth_rec is not None
                    maximum_reconstruction_mse_error = max(maximum_reconstruction_mse_error,
                                                            float(np.abs(synth_rec[:, 0] - direct_rec[:, 0]).max()))
                    direct_parts.append((metrics, dist, direct_geometry, result.z_ols.coefficients.detach().cpu().numpy(), direct_rec))
            _assert_snapshot(model, model_snapshot, f"preflight/{endpoint}/{grid}")
            map_payload = {f"plane_{index}": np.stack(map_parts[name]).astype(np.float32)
                           for index, name in enumerate(DEV_PLANES)}
            map_payload.update({"rows": np.asarray(probe_rows, dtype=np.int64),
                                "labels": fixed["labels"][row_tensor].numpy().astype(np.int64)})
            write_npz(representative_maps_root / f"representative_dev_intensity_{endpoint}_{grid}.npz", **map_payload)
            write_json(representative_maps_root / f"representative_dev_intensity_{endpoint}_{grid}.json", {
                "endpoint": endpoint, "grid": grid, "data_role": "fixed_DEV_first_two_per_class",
                "rows": probe_rows, "plane_order": list(DEV_PLANES),
                "map_semantics": "physical intensity |E|^2 in W/m2; coherent field computed before squaring",
                "geometries": direct_parts[0][2],
            })
            del model
            if fixed["device"].type == "cuda":
                torch.cuda.empty_cache()
            basis_info[f"{endpoint}/{grid_role}"] = {
                "basis_path": f"bases/{endpoint}_{grid}_decoder_planes.npz",
                "grid_size": grid, "ports": 100, "decoder_planes": len(GENERATION_PLANES),
                "probe_direct_forwards": len(probe_rows), "state_frozen": True,
            }
    require(maximum_complex_error <= cfg["validation"]["same_grid_complex_relative_tolerance"],
            "same-grid direct and coherent basis fields differ beyond 1e-4")
    require(maximum_reconstruction_mse_error <= cfg["validation"]["direct_vs_synthesized_mse_absolute_tolerance"],
            "same-grid direct and coherent synthesis MSE differ beyond 1e-6")
    require(maximum_power_trace_error <= 1.0e-4, "plane field integral does not reproduce the production power trace")
    require(maximum_numpy_integral_error <= 1.0e-12, "independent NumPy plane integral differs from production-plane metrics")
    receipt = {
        "schema": "gaussian_recent_plane_density_preflight.v1", "task_id": TASK_ID,
        "status": "READY", "endpoint_identity": {key: fixed["endpoint_specs"][key] for key in ENDPOINTS},
        "identity_checks": fixed["identity_receipt"], "basis": basis_info,
        "probe_rows": probe_rows, "rows_per_class": 2,
        "max_same_grid_complex_relative_error": maximum_complex_error,
        "max_direct_vs_synthesized_mse_absolute_error": maximum_reconstruction_mse_error,
        "max_power_trace_relative_error": maximum_power_trace_error,
        "max_independent_numpy_area_integral_relative_error": maximum_numpy_integral_error,
        "analytic_density_controls": density_closeout.validate_density_core(),
        "cross_grid_difference_is_sensitivity_only": True,
        "estimated_elapsed_seconds_observed": time.perf_counter() - started,
        "training_updates": 0, "optimizers_created": 0, "prior_samples": 0,
        "official_test_reads": 0, "hardware_operations": 0,
    }
    write_json(root / "preflight" / "receipt.json", receipt)
    return receipt


def _preflight_common(cfg: dict[str, Any], root: Path, device: torch.device) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    fixed = load_fixed_inputs(cfg, device)
    cohorts = load_generation_inputs(cfg, fixed)
    fixed["identity_receipt"] = _identity_preflight(fixed, cohorts)
    manifest = {
        "schema": "gaussian_recent_plane_density_input_manifest.v1", "task_id": TASK_ID,
        "dev_rows": fixed["dev_rows"].tolist(),
        "dev_labels": fixed["labels"][fixed["dev_rows"]].tolist(),
        "split_seed": 1234, "dev_per_class": 500,
        "endpoints": {key: fixed["endpoint_specs"][key] for key in ENDPOINTS},
        "generation_cohorts": {key: {k: v for k, v in item.items() if k != "coefficients"}
                               for key, item in cohorts.items()},
        "official_test_reads": 0,
    }
    write_json(root / "inputs" / "input_manifest.json", manifest)
    return fixed, cohorts


def run_preflight(cfg: dict[str, Any], root: Path, device: torch.device) -> dict[str, Any]:
    require(not (root / "preflight" / "receipt.json").exists(),
            "preflight receipt already exists; review it before rerunning")
    fixed, cohorts = _preflight_common(cfg, root, device)
    return _build_basis_and_validate(cfg, fixed, root)


def _reconstruction_for_direct(model: torch.nn.Module, result: Any, pixels: torch.Tensor, rows: np.ndarray,
                               gain: float) -> np.ndarray:
    return _native_image_metrics(model, result, pixels, rows, gain)


def _chunk_done(path: Path, expected_rows: np.ndarray, expected_labels: np.ndarray, plane_count: int,
                coeff_required: bool) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as saved:
        require(np.array_equal(saved["rows"], expected_rows) and np.array_equal(saved["labels"], expected_labels)
                and saved["metrics"].shape == (len(expected_rows), plane_count, 7)
                and (not coeff_required or saved["coefficients"].shape == (len(expected_rows), 100)),
                f"recovery chunk identity/schema mismatch: {path}")
        return {key: saved[key].copy() for key in saved.files}


def _measure_direct_dev(cfg: dict[str, Any], root: Path, fixed: dict[str, Any], endpoint: str,
                        grid_role: str, grid: int) -> dict[str, Any]:
    model = _model_for(endpoint, grid, fixed)
    snapshot = _snapshot(model)
    directory = _dev_paths(root, endpoint, grid_role)
    directory.mkdir(parents=True, exist_ok=True)
    dev_rows = fixed["dev_rows"].numpy().astype(np.int64)
    labels = fixed["labels"]
    chunk_rows = int(cfg["execution"]["result_chunk_rows"])
    chunk_count = math.ceil(len(dev_rows) / chunk_rows)
    geometries: list[dict[str, Any]] | None = None
    for chunk_index in range(chunk_count):
        lo, hi = chunk_index * chunk_rows, min(len(dev_rows), (chunk_index + 1) * chunk_rows)
        rows = dev_rows[lo:hi]
        batch_labels = labels[torch.as_tensor(rows)].numpy().astype(np.int64)
        path = _chunk_path(directory, chunk_index)
        if _chunk_done(path, rows, batch_labels, len(DEV_PLANES), True) is not None:
            continue
        metrics: list[np.ndarray] = []
        distributions: list[np.ndarray] = []
        coefficients: list[np.ndarray] = []
        reconstruction: list[np.ndarray] = []
        for start in range(lo, hi, int(cfg["execution"]["batch_rows"])):
            stop = min(hi, start + int(cfg["execution"]["batch_rows"]))
            row_batch = dev_rows[start:stop]
            images = image_batch(fixed["pixels"], torch.as_tensor(row_batch), "small", fixed["device"])
            with torch.inference_mode():
                result = model.runtime.forward_from_image(images)
                fields, trace = small_image_result_fields(model, result)
                batch_metrics, batch_dist, current_geometry = _metrics_from_fields(fields, DEV_PLANES)
                expected = density_closeout._small_expected_trace(trace, result)
                for plane_index, plane_name in enumerate(DEV_PLANES):
                    truth = expected[plane_name].to(torch.float64).detach().cpu().numpy()
                    relative = np.abs(batch_metrics[:, plane_index, 0] - truth) / np.maximum(
                        np.abs(truth), np.finfo(np.float64).tiny)
                    require(float(relative.max()) <= 1.0e-4,
                            f"production plane power identity failed: {endpoint}/{grid}/{plane_name}")
                gain = float(fixed["checkpoints"][endpoint]["camera_gain"].reshape(-1)[0])
                rec = _reconstruction_for_direct(model, result, fixed["pixels"], row_batch, gain)
                metrics.append(batch_metrics)
                distributions.append(batch_dist)
                coefficients.append(result.z_ols.coefficients.detach().cpu().numpy().astype(np.complex64))
                reconstruction.append(rec)
                geometries = current_geometry if geometries is None else geometries
        payload = {
            "rows": rows, "labels": batch_labels,
            "metrics": np.concatenate(metrics).astype(np.float64),
            "distribution": np.concatenate(distributions).astype(np.float64),
            "coefficients": np.concatenate(coefficients).astype(np.complex64),
            "reconstruction": np.concatenate(reconstruction).astype(np.float64),
        }
        _save_dev_chunk(path, payload)
        write_json(root / "progress.json", {"stage": "direct_dev", "endpoint": endpoint,
                                             "grid_role": grid_role, "completed_chunks": chunk_index + 1,
                                             "total_chunks": chunk_count, "completed_rows": hi})
    if geometries is None:
        meta = directory / "metadata.json"
        if meta.is_file():
            geometries = json.loads(meta.read_text(encoding="utf-8"))["planes"]
    if geometries is None:
        probe = image_batch(fixed["pixels"], torch.as_tensor(dev_rows[:1]), "small", fixed["device"])
        with torch.inference_mode():
            result = model.runtime.forward_from_image(probe)
            fields, _ = small_image_result_fields(model, result)
            _, _, geometries = _metrics_from_fields(fields, DEV_PLANES)
    write_json(directory / "metadata.json", {"endpoint": endpoint, "global_step": fixed["endpoint_specs"][endpoint]["global_step"],
                                               "grid_role": grid_role, "grid": grid,
                                               "data_role": "fixed_DEV_500_per_class", "sample_count": 5000,
                                               "plane_order": list(DEV_PLANES), "planes": geometries})
    _assert_snapshot(model, snapshot, f"measure-dev/{endpoint}/{grid}")
    del model
    if fixed["device"].type == "cuda":
        torch.cuda.empty_cache()
    return {"status": "CHUNKED", "endpoint": endpoint, "grid": grid, "sample_count": 5000,
            "chunk_count": chunk_count, "all_chunks_readback_verified": True}


def _load_basis_and_geometry(endpoint: str, grid: int, root: Path) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    path = root / "bases" / f"{endpoint}_{grid}_decoder_planes.npz"
    require(path.is_file(), f"preflight decoder basis is missing: {path}")
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[f"field_{index}"].copy() for index, name in enumerate(GENERATION_PLANES)}
        geometry = json.loads(str(saved["geometry_json"].item()))
    return arrays, geometry


def _measure_decoder_pair(cfg: dict[str, Any], root: Path, fixed: dict[str, Any], grid_role: str,
                          grid: int) -> dict[str, Any]:
    c_directory = _dev_paths(root, "C45000", grid_role)
    d0_model = _model_for("D050000", grid, fixed)
    dg_model = _model_for("Dgrad50000", grid, fixed)
    models = {"D050000": d0_model, "Dgrad50000": dg_model}
    snapshots = {key: _snapshot(value) for key, value in models.items()}
    bases = {key: _load_basis_and_geometry(key, grid, root) for key in models}
    labels_source = fixed["labels"]
    dev_rows = fixed["dev_rows"].numpy().astype(np.int64)
    chunk_rows = int(cfg["execution"]["result_chunk_rows"])
    chunk_count = math.ceil(len(dev_rows) / chunk_rows)
    complete = 0
    for chunk_index in range(chunk_count):
        lo, hi = chunk_index * chunk_rows, min(len(dev_rows), (chunk_index + 1) * chunk_rows)
        rows = dev_rows[lo:hi]
        labels = labels_source[torch.as_tensor(rows)].numpy().astype(np.int64)
        c_path = _chunk_path(c_directory, chunk_index)
        c_saved = _chunk_done(c_path, rows, labels, len(DEV_PLANES), True)
        require(c_saved is not None, f"C45k source DEV coefficients missing for shared Decoder input: {c_path}")
        d0_path = _chunk_path(_dev_paths(root, "D050000", grid_role), chunk_index)
        dg_path = _chunk_path(_dev_paths(root, "Dgrad50000", grid_role), chunk_index)
        if (_chunk_done(d0_path, rows, labels, len(DEV_PLANES), False) is not None
                and _chunk_done(dg_path, rows, labels, len(DEV_PLANES), False) is not None):
            complete += 1
            continue
        upstream = c_saved["metrics"][:, :6].copy()
        coefficients = c_saved["coefficients"]
        targets = fixed["pixels"]
        gain_d0 = float(fixed["checkpoints"]["D050000"]["camera_gain"].reshape(-1)[0])
        gain_dg = float(fixed["checkpoints"]["Dgrad50000"]["camera_gain"].reshape(-1)[0])
        require(math.isclose(gain_d0, gain_dg, rel_tol=1e-12), "D0/Dgrad camera gains differ")
        batch_metrics: dict[str, list[np.ndarray]] = {key: [] for key in models}
        batch_distribution: dict[str, list[np.ndarray]] = {key: [] for key in models}
        batch_rec: dict[str, list[np.ndarray]] = {key: [] for key in models}
        batch_equal: list[np.ndarray] = []
        for start in range(0, len(rows), 8):
            stop = min(len(rows), start + 8)
            row_piece = rows[start:stop]
            targets_piece = targets
            decoded: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None, torch.Tensor | None, list[dict[str, Any]]]] = {}
            for key in models:
                basis, geometry = bases[key]
                decoded[key] = _synthesize_metrics(
                    key, coefficients[start:stop], basis, geometry, models[key], fixed["device"],
                    targets_piece, row_piece, gain_d0 if key == "D050000" else gain_dg,
                )
                batch_metrics[key].append(decoded[key][0])
                batch_distribution[key].append(decoded[key][1])
                require(decoded[key][2] is not None and decoded[key][3] is not None,
                        "Decoder cohort synthesis omitted reconstruction outputs")
                batch_rec[key].append(decoded[key][2])
            d0_power = torch.as_tensor(decoded["D050000"][0][:, 3, 0], dtype=torch.float64, device=fixed["device"])
            dg_power = torch.as_tensor(decoded["Dgrad50000"][0][:, 3, 0], dtype=torch.float64, device=fixed["device"])
            valid = (d0_power > 0.0) & (dg_power > 0.0)
            scale = torch.full_like(dg_power, float("nan"))
            scale[valid] = d0_power[valid] / dg_power[valid]
            dg_camera = decoded["Dgrad50000"][3]
            assert dg_camera is not None
            plane = _field_plane_from_geometry(dg_model, bases["Dgrad50000"][1][-1])
            camera_field = ComplexField(dg_camera, plane,
                                        OperationProvenance("coherent_saved_port_response_sum", plane.plane_id, None))
            eq_rows = np.full((len(row_piece), len(REC_FIELDS)), np.nan, dtype=np.float64)
            if bool(valid.any()):
                equal_metrics, _ = _reconstruction_metrics(camera_field, targets_piece, row_piece, gain_dg, scale)
                eq_rows[valid.detach().cpu().numpy()] = equal_metrics[valid.detach().cpu().numpy()]
            batch_equal.append(eq_rows)
        for key in models:
            metric = np.concatenate(batch_metrics[key], axis=0)
            dist = np.concatenate(batch_distribution[key], axis=0)
            rec = np.concatenate(batch_rec[key], axis=0)
            full = np.concatenate((upstream, metric), axis=1)
            require(full.shape == (len(rows), len(DEV_PLANES), 7), "D0/Dgrad DEV plane table shape changed")
            payload = {"rows": rows, "labels": labels, "metrics": full,
                       "distribution": np.where(np.isfinite(dist), dist, c_saved["distribution"]),
                       "coefficients": coefficients,
                       "reconstruction": rec,
                       "equal_camera_power_reconstruction": np.concatenate(batch_equal, axis=0)
                       if key == "Dgrad50000" else np.full_like(rec, np.nan)}
            _save_dev_chunk(_chunk_path(_dev_paths(root, key, grid_role), chunk_index), payload)
        complete += 1
        write_json(root / "progress.json", {"stage": "shared_encoder_decoder_dev", "grid_role": grid_role,
                                             "completed_chunks": complete, "total_chunks": chunk_count,
                                             "completed_rows": hi})
    c_meta = json.loads((c_directory / "metadata.json").read_text(encoding="utf-8"))
    for key in models:
        combined_geometry = c_meta["planes"][:6] + bases[key][1]
        directory = _dev_paths(root, key, grid_role)
        write_json(directory / "metadata.json", {
            "endpoint": key, "global_step": fixed["endpoint_specs"][key]["global_step"],
            "grid_role": grid_role, "grid": grid, "data_role": "fixed_DEV_500_per_class",
            "sample_count": 5000, "plane_order": list(DEV_PLANES), "planes": combined_geometry,
            "upstream_fields_reused_from": "C45000 exact frozen Encoder/OLS state",
        })
    for key, model in models.items():
        _assert_snapshot(model, snapshots[key], f"measure-dev/{key}/{grid}")
    endpoint_keys = list(models)
    del models, bases
    if fixed["device"].type == "cuda":
        torch.cuda.empty_cache()
    return {"status": "CHUNKED", "grid": grid, "endpoints": endpoint_keys,
            "shared_encoder_coefficients_reused": True, "completed_chunks": complete}


def _measure_generated(cfg: dict[str, Any], root: Path, fixed: dict[str, Any],
                       cohorts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    chunk_rows = int(cfg["execution"]["result_chunk_rows"])
    for cohort_key, cohort in cohorts.items():
        endpoint = str(cohort.get("endpoint", cohort_key.rsplit("_", 1)[0]))
        method = str(cohort["method"])
        # Cohort names with a seed endpoint map directly to the six audited checkpoints.
        if endpoint not in ENDPOINTS:
            endpoint = {"C45000": "C45000", "R51245000": "R51245000", "R52245000": "R52245000"}.get(endpoint, endpoint)
        require(endpoint in ENDPOINTS, f"generation cohort does not map to a measured checkpoint: {cohort_key}")
        for grid_role, grid in GRIDS:
            basis, geometry = _load_basis_and_geometry(endpoint, grid, root)
            model = _model_for(endpoint, grid, fixed)
            snapshot = _snapshot(model)
            rep_positions = [int(position) for label in range(10)
                             for position in np.flatnonzero(cohort["labels"] == label)[:2]]
            representative_maps: dict[str, list[np.ndarray]] = {name: [] for name in GENERATION_PLANES}
            with torch.inference_mode():
                rep_coeff = torch.as_tensor(cohort["coefficients"][rep_positions], dtype=torch.complex64,
                                            device=fixed["device"])
                for name in GENERATION_PLANES:
                    basis_tensor = torch.as_tensor(basis[name], dtype=torch.complex64, device=fixed["device"])
                    fields = torch.einsum("bi,ihw->bhw", rep_coeff, basis_tensor)
                    representative_maps[name] = [item[0].abs().square().to(torch.float32).cpu().numpy()
                                                 for item in fields.unbind(0)]
            map_payload = {f"plane_{index}": np.stack(representative_maps[name]).astype(np.float32)
                           for index, name in enumerate(GENERATION_PLANES)}
            map_payload.update({"rows": np.asarray(rep_positions, dtype=np.int64),
                                "labels": cohort["labels"][rep_positions],
                                "proposal_ids": cohort["proposal_ids"][rep_positions]})
            write_npz(root / "figures" / f"representative_generation_intensity_{cohort_key}_{grid}.npz", **map_payload)
            write_json(root / "figures" / f"representative_generation_intensity_{cohort_key}_{grid}.json", {
                "cohort": cohort_key, "endpoint": endpoint, "method": method, "grid": grid,
                "rows_per_class": 2, "plane_order": list(GENERATION_PLANES),
                "map_semantics": "physical intensity |sum_i a_i h_i|^2 in W/m2; coherent sum precedes intensity",
                "geometries": geometry,
            })
            directory = root / "populations" / "generation" / endpoint / method / grid_role
            directory.mkdir(parents=True, exist_ok=True)
            count = len(cohort["labels"])
            for chunk_index, lo in enumerate(range(0, count, chunk_rows)):
                hi = min(count, lo + chunk_rows)
                rows = np.arange(lo, hi, dtype=np.int64)
                labels = cohort["labels"][lo:hi]
                path = _chunk_path(directory, chunk_index)
                if path.is_file():
                    with np.load(path, allow_pickle=False) as saved:
                        require(np.array_equal(saved["rows"], rows) and np.array_equal(saved["labels"], labels)
                                and saved["metrics"].shape == (len(rows), len(GENERATION_PLANES), 7),
                                f"generation recovery identity/schema changed: {path}")
                    continue
                metrics, dist, _, _, _ = _synthesize_metrics(
                    endpoint, cohort["coefficients"][lo:hi], basis, geometry, model, fixed["device"]
                )
                payload = {"rows": rows, "labels": labels,
                           "proposal_ids": cohort["proposal_ids"][lo:hi],
                           "component_ids": cohort["component_ids"][lo:hi],
                           "metrics": metrics, "distribution": dist}
                write_npz(path, **payload)
                write_json(root / "progress.json", {"stage": "generation", "cohort": cohort_key,
                                                     "grid_role": grid_role, "completed_chunks": chunk_index + 1,
                                                     "total_chunks": math.ceil(count / chunk_rows), "completed_rows": hi})
            _assert_snapshot(model, snapshot, f"generation/{cohort_key}/{grid}")
            del model
            if fixed["device"].type == "cuda":
                torch.cuda.empty_cache()
            meta_geometry = geometry
            write_json(directory / "metadata.json", {
                "cohort": cohort_key, "endpoint": endpoint, "method": method,
                "grid_role": grid_role, "grid": grid, "sample_count": count,
                "data_role": "saved_generated_coefficients", "source_file": cohort["path"],
                "proposal_identity": cohort.get("source_note", "saved cohort proposal IDs retained"),
                "plane_order": list(GENERATION_PLANES), "planes": meta_geometry,
            })
            results.append({"cohort": cohort_key, "endpoint": endpoint, "method": method,
                            "grid": grid, "sample_count": count, "chunk_count": math.ceil(count / chunk_rows)})
    return results


def _load_population_chunks(directory: Path, expected_count: int, plane_count: int,
                            include_coefficients: bool = False) -> dict[str, np.ndarray]:
    chunks = sorted(directory.glob("chunk_*.npz"))
    require(bool(chunks), f"population chunks are missing: {directory}")
    arrays: dict[str, list[np.ndarray]] = {}
    for path in chunks:
        with np.load(path, allow_pickle=False) as saved:
            require(saved["metrics"].ndim == 3 and saved["metrics"].shape[1:] == (plane_count, 7),
                    f"population chunk metric shape changed: {path}")
            for key in ("rows", "labels", "metrics", "distribution", "reconstruction", "equal_camera_power_reconstruction"):
                if key in saved.files:
                    arrays.setdefault(key, []).append(saved[key].copy())
            if include_coefficients and "coefficients" in saved.files:
                arrays.setdefault("coefficients", []).append(saved["coefficients"].copy())
            if "proposal_ids" in saved.files:
                arrays.setdefault("proposal_ids", []).append(saved["proposal_ids"].copy())
            if "component_ids" in saved.files:
                arrays.setdefault("component_ids", []).append(saved["component_ids"].copy())
    output = {key: np.concatenate(parts, axis=0) for key, parts in arrays.items()}
    require(len(output["rows"]) == expected_count and len(np.unique(output["rows"])) == expected_count,
            f"population chunk row coverage is incomplete or duplicated: {directory}")
    return output


def _group_summary(values: np.ndarray, labels: np.ndarray, columns: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for label in [None, *range(10)]:
        selected = np.ones(len(labels), dtype=bool) if label is None else labels == label
        entry: dict[str, Any] = {"class": "ALL" if label is None else label, "count": int(selected.sum())}
        for index, name in enumerate(columns):
            column = values[selected, index]
            finite = column[np.isfinite(column)]
            entry[f"{name}_mean"] = float(finite.mean()) if finite.size else None
            entry[f"{name}_median"] = float(np.median(finite)) if finite.size else None
            entry[f"{name}_p10"] = float(np.quantile(finite, 0.10)) if finite.size else None
            entry[f"{name}_p90"] = float(np.quantile(finite, 0.90)) if finite.size else None
        result.append(entry)
    return result


def _bootstrap_paired(delta: np.ndarray, labels: np.ndarray, resamples: int, seed: int) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values)
    values, labels = values[valid], labels[valid]
    if not len(values):
        return {"mean": None, "ci95_low": None, "ci95_high": None, "valid_n": 0}
    classes = [values[labels == label] for label in range(10)]
    if any(len(group) == 0 for group in classes):
        return {"mean": float(values.mean()), "ci95_low": None, "ci95_high": None, "valid_n": len(values)}
    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        draws[index] = np.mean(np.concatenate([group[rng.integers(0, len(group), size=len(group))]
                                               for group in classes]))
    return {"mean": float(values.mean()), "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)), "valid_n": int(len(values)),
            "bootstrap_resamples": resamples, "bootstrap_seed": seed}


def _write_population_tables(cfg: dict[str, Any], root: Path, fixed: dict[str, Any], cohorts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output_rows = root / "tables" / "plane_density_per_image.csv.gz"
    output_rows.parent.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    geometry_by_group: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for endpoint in ENDPOINTS:
        for grid_role, grid in GRIDS:
            directory = _dev_paths(root, endpoint, grid_role)
            data = _load_population_chunks(directory, 5000, len(DEV_PLANES), include_coefficients=True)
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            geometry_by_group[("DEV", endpoint, grid_role, "real_input")] = metadata["planes"]
            reconstruction = data.get("reconstruction", np.full((5000, len(REC_FIELDS)), np.nan))
            equal_power = data.get("equal_camera_power_reconstruction", np.full_like(reconstruction, np.nan))
            for plane_index, plane in enumerate(DEV_PLANES):
                summary = _group_summary(data["metrics"][:, plane_index, :], data["labels"], density_closeout.METRIC_FIELDS)
                for row in summary:
                    summary_rows.append({"data_role": "fixed_DEV", "endpoint": endpoint, "grid_role": grid_role,
                                         "grid": grid, "plane_key": plane, **row})
            if endpoint in ("D050000", "Dgrad50000"):
                rec_summary = _group_summary(reconstruction, data["labels"], REC_FIELDS)
                eq_summary = _group_summary(equal_power, data["labels"], REC_FIELDS)
                for row in rec_summary:
                    summary_rows.append({"data_role": "fixed_DEV_reconstruction", "endpoint": endpoint,
                                         "grid_role": grid_role, "grid": grid, "plane_key": "camera_readout",
                                         **row})
                for row in eq_summary:
                    summary_rows.append({"data_role": "fixed_DEV_Dgrad_equal_camera_power", "endpoint": endpoint,
                                         "grid_role": grid_role, "grid": grid, "plane_key": "camera_readout",
                                         **row})

    generated_groups: list[tuple[str, str, str, str, str, int, dict[str, np.ndarray], list[dict[str, Any]]]] = []
    for cohort_key, cohort in cohorts.items():
        endpoint = str(cohort.get("endpoint", cohort_key.rsplit("_", 1)[0]))
        method = str(cohort["method"])
        for grid_role, grid in GRIDS:
            directory = root / "populations" / "generation" / endpoint / method / grid_role
            data = _load_population_chunks(directory, 10000, len(GENERATION_PLANES))
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            generated_groups.append((cohort_key, endpoint, method, grid_role, "saved_generation", grid, data, metadata["planes"]))
            for plane_index, plane in enumerate(GENERATION_PLANES):
                summary = _group_summary(data["metrics"][:, plane_index, :], data["labels"], density_closeout.METRIC_FIELDS)
                for row in summary:
                    summary_rows.append({"data_role": "saved_generation", "cohort": cohort_key,
                                         "endpoint": endpoint, "method": method, "grid_role": grid_role,
                                         "grid": grid, "plane_key": plane, **row})

    fieldnames = ["population_role", "cohort", "endpoint", "global_step", "grid_role", "grid", "data_role",
                  "sample_row", "label", "method", "stream", "proposal_id", "component_id", "plane_index",
                  "plane_key", "plane_id", "coordinate_frame", "height", "width", "pixel_pitch_y_m",
                  "pixel_pitch_x_m", "origin_y_m", "origin_x_m", "z_m", "measurement_boundary",
                  "power_w", "geometric_area_m2", "effective_area_m2", "mean_intensity_geometric_w_m2",
                  "mean_intensity_geometric_w_cm2", "mean_intensity_effective_w_m2",
                  "mean_intensity_effective_w_cm2", "effective_to_geometric_area_ratio",
                  "integral_intensity_squared_w2_m2", *DIST_FIELDS]
    if output_rows.exists():
        output_rows.unlink()
    with gzip.open(output_rows, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for endpoint in ENDPOINTS:
            step = int(fixed["endpoint_specs"][endpoint]["global_step"])
            for grid_role, grid in GRIDS:
                directory = _dev_paths(root, endpoint, grid_role)
                data = _load_population_chunks(directory, 5000, len(DEV_PLANES), include_coefficients=True)
                meta = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
                for index in range(5000):
                    for plane_index, plane in enumerate(DEV_PLANES):
                        geom = meta["planes"][plane_index]
                        values = data["metrics"][index, plane_index]
                        row = {"population_role": "real_input", "cohort": endpoint, "endpoint": endpoint,
                               "global_step": step, "grid_role": grid_role, "grid": grid,
                               "data_role": "fixed_DEV", "sample_row": int(data["rows"][index]),
                               "label": int(data["labels"][index]), "method": None, "stream": None,
                               "proposal_id": None, "component_id": None, "plane_index": plane_index,
                               "plane_key": plane, "plane_id": geom["plane_id"],
                               "coordinate_frame": geom["coordinate_frame"], "height": geom["shape"][0],
                               "width": geom["shape"][1], "pixel_pitch_y_m": geom["pixel_pitch_m"][0],
                               "pixel_pitch_x_m": geom["pixel_pitch_m"][1], "origin_y_m": geom["origin_m"][0],
                               "origin_x_m": geom["origin_m"][1], "z_m": geom["z_m"],
                               "measurement_boundary": geom["measurement_boundary"],
                               **dict(zip(density_closeout.METRIC_FIELDS, values.tolist(), strict=True))}
                        row["mean_intensity_geometric_w_cm2"] = row["mean_intensity_geometric_w_m2"] / 1.0e4
                        row["mean_intensity_effective_w_cm2"] = row["mean_intensity_effective_w_m2"] / 1.0e4
                        if plane_index == DEV_PLANES.index("fiber_output_and_decoder_entry"):
                            row.update(dict(zip(DIST_FIELDS, data["distribution"][index].tolist(), strict=True)))
                        else:
                            row.update({key: None for key in DIST_FIELDS})
                        writer.writerow(clean(row))
        for cohort_key, endpoint, method, grid_role, data_role, grid, data, geometries in generated_groups:
            step = int(fixed["endpoint_specs"][endpoint]["global_step"])
            for index in range(10000):
                for plane_index, plane in enumerate(GENERATION_PLANES):
                    geom = geometries[plane_index]
                    values = data["metrics"][index, plane_index]
                    row = {"population_role": "saved_generation", "cohort": cohort_key, "endpoint": endpoint,
                           "global_step": step, "grid_role": grid_role, "grid": grid,
                           "data_role": "saved_generated_coefficients", "sample_row": int(data["rows"][index]),
                           "label": int(data["labels"][index]), "method": method, "stream": 0,
                           "proposal_id": int(data["proposal_ids"][index]),
                           "component_id": int(data["component_ids"][index]), "plane_index": plane_index,
                           "plane_key": plane, "plane_id": geom["plane_id"],
                           "coordinate_frame": geom["coordinate_frame"], "height": geom["shape"][0],
                           "width": geom["shape"][1], "pixel_pitch_y_m": geom["pixel_pitch_m"][0],
                           "pixel_pitch_x_m": geom["pixel_pitch_m"][1], "origin_y_m": geom["origin_m"][0],
                           "origin_x_m": geom["origin_m"][1], "z_m": geom["z_m"],
                           "measurement_boundary": geom["measurement_boundary"],
                           **dict(zip(density_closeout.METRIC_FIELDS, values.tolist(), strict=True))}
                    row["mean_intensity_geometric_w_cm2"] = row["mean_intensity_geometric_w_m2"] / 1.0e4
                    row["mean_intensity_effective_w_cm2"] = row["mean_intensity_effective_w_m2"] / 1.0e4
                    if plane_index == GENERATION_PLANES.index("fiber_output_and_decoder_entry"):
                        row.update(dict(zip(DIST_FIELDS, data["distribution"][index].tolist(), strict=True)))
                    else:
                        row.update({key: None for key in DIST_FIELDS})
                    writer.writerow(clean(row))

    summary_path = root / "tables" / "plane_density_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        with summary_path.open("w", newline="", encoding="utf-8-sig") as stream:
            fieldnames = list(dict.fromkeys(key for row in summary_rows for key in row))
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean(row) for row in summary_rows)

    comparisons = _build_comparisons(cfg, root, fixed)
    write_json(root / "tables" / "paired_endpoint_comparisons.json", comparisons)
    _write_power_area_decomposition(root, fixed)
    expected_dev = len(ENDPOINTS) * 5000 * len(DEV_PLANES) * len(GRIDS)
    expected_gen = len(generated_groups) * 10000 * len(GENERATION_PLANES)
    expected_records = expected_dev + expected_gen
    row_count = 0
    unique_keys: set[tuple[str, str, str, int, int]] = set()
    with gzip.open(output_rows, "rt", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            row_count += 1
            key = (row["population_role"], row["cohort"], row["grid_role"], int(row["sample_row"]), int(row["plane_index"]))
            require(key not in unique_keys, "per-image plane table contains a duplicate key")
            unique_keys.add(key)
    require(row_count == expected_records, f"per-image plane table row count {row_count} != {expected_records}")
    return {"status": "VERIFIED", "per_image_plane_rows": row_count,
            "expected_per_image_plane_rows": expected_records, "unique_keys": len(unique_keys),
            "dev_rows": expected_dev, "generation_rows": expected_gen,
            "summary_rows": len(summary_rows), "table": str(output_rows.relative_to(root)),
            "paired_comparisons": comparisons}


def _build_comparisons(cfg: dict[str, Any], root: Path, fixed: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    pairs = (("C45000", "V45000"), ("C45000", "D050000"), ("D050000", "Dgrad50000"))
    resamples = int(cfg["statistics"]["bootstrap_resamples"])
    seed = int(cfg["statistics"]["bootstrap_seed"])
    for grid_role, grid in GRIDS:
        for old, new in pairs:
            left = _load_population_chunks(_dev_paths(root, old, grid_role), 5000, len(DEV_PLANES), include_coefficients=True)
            right = _load_population_chunks(_dev_paths(root, new, grid_role), 5000, len(DEV_PLANES), include_coefficients=True)
            require(np.array_equal(left["rows"], right["rows"]) and np.array_equal(left["labels"], right["labels"]),
                    f"paired DEV row alignment changed: {old}->{new}")
            entry: dict[str, Any] = {"old_endpoint": old, "new_endpoint": new, "grid_role": grid_role,
                                     "grid": grid, "pairing": "same original DEV row IDs, class-stratified paired bootstrap",
                                     "plane_metrics": {}}
            for plane_index, plane in enumerate(DEV_PLANES):
                metrics: dict[str, Any] = {}
                for metric_index, metric_name in enumerate(density_closeout.METRIC_FIELDS):
                    delta = right["metrics"][:, plane_index, metric_index] - left["metrics"][:, plane_index, metric_index]
                    metrics[metric_name] = _bootstrap_paired(delta, left["labels"], resamples, seed)
                entry["plane_metrics"][plane] = metrics
            for metric_index, metric_name in enumerate(REC_FIELDS):
                delta = right["reconstruction"][:, metric_index] - left["reconstruction"][:, metric_index]
                entry.setdefault("native_reconstruction", {})[metric_name] = _bootstrap_paired(delta, left["labels"], resamples, seed)
            if old == "D050000" and new == "Dgrad50000":
                for metric_index, metric_name in enumerate(REC_FIELDS):
                    delta = right["equal_camera_power_reconstruction"][:, metric_index] - left["reconstruction"][:, metric_index]
                    entry.setdefault("Dgrad_equal_camera_power_vs_D0_native", {})[metric_name] = _bootstrap_paired(
                        delta, left["labels"], resamples, seed)
                entry["equal_camera_power_defined_n"] = int(np.isfinite(right["equal_camera_power_reconstruction"][:, 0]).sum())
            result[f"{old}_to_{new}/{grid_role}"] = entry
    # The four seed/trajectory cohorts are descriptive separate checkpoints, not a controlled cross-seed ranking.
    result["interpretation_boundary"] = {
        "C_to_V": "same split, same seed, same global step; paired endpoint contrast",
        "D0_to_Dgrad": "same C encoder and exact same real/generated coefficients; Decoder-only continuation contrast",
        "C_to_replication_seeds": "different training seeds; report per-endpoint populations, not causal attribution",
        "generated_G1_vs_G4": "distinct saved cohorts; independent within-class resampling, not row-paired",
        "generated_MSE": "not calculated because requested labels are conditioning labels, not pixelwise ground truth",
    }
    return result


def _write_power_area_decomposition(root: Path, fixed: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    pairs = (("C45000", "V45000"), ("D050000", "Dgrad50000"))
    for grid_role, grid in GRIDS:
        for old, new in pairs:
            left = _load_population_chunks(_dev_paths(root, old, grid_role), 5000, len(DEV_PLANES))
            right = _load_population_chunks(_dev_paths(root, new, grid_role), 5000, len(DEV_PLANES))
            for plane_index, plane in enumerate(DEV_PLANES):
                p0, p1 = left["metrics"][:, plane_index, 0], right["metrics"][:, plane_index, 0]
                a0, a1 = left["metrics"][:, plane_index, 2], right["metrics"][:, plane_index, 2]
                valid = (p0 > 0) & (p1 > 0) & (a0 > 0) & (a1 > 0)
                delta_log_p = np.full(len(p0), np.nan)
                delta_log_a = np.full(len(p0), np.nan)
                delta_log_ieff = np.full(len(p0), np.nan)
                delta_log_p[valid] = np.log(p1[valid]) - np.log(p0[valid])
                delta_log_a[valid] = np.log(a1[valid]) - np.log(a0[valid])
                delta_log_ieff[valid] = delta_log_p[valid] - delta_log_a[valid]
                for label in range(10):
                    selected = (left["labels"] == label) & valid
                    rows.append({"old_endpoint": old, "new_endpoint": new, "grid_role": grid_role, "grid": grid,
                                 "plane_key": plane, "class": label, "n_positive_pairs": int(selected.sum()),
                                 "mean_delta_log_power": float(delta_log_p[selected].mean()) if selected.any() else None,
                                 "mean_delta_log_effective_area": float(delta_log_a[selected].mean()) if selected.any() else None,
                                 "mean_delta_log_effective_density": float(delta_log_ieff[selected].mean()) if selected.any() else None})
    path = root / "tables" / "power_area_log_decomposition.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _finite_difference_recheck(cfg: dict[str, Any], root: Path, device: torch.device) -> dict[str, Any]:
    previous_root = ROOT / "outputs/20261022_gaussian_ols_effective_channel_dual_grid_v1/runs/attempt_01"
    payload_path = previous_root / "engineering/local_direction_repair_inputs.npz"
    require(payload_path.is_file(), f"fixed local-direction replay inputs are missing: {payload_path}")
    camera = dual_grid.read_noise_proxy(dual_grid.read_config(ROOT / "configs/research/lightgen/gaussian_ols_effective_channel_dual_grid_v1.yaml"))
    rows_out: list[dict[str, Any]] = []
    exposure_values = [float(value) for value in camera["exposure_sweep_s"]]
    radii = [float(value) for value in cfg["statistics"]["finite_difference_relative_radii"]]
    started = time.perf_counter()
    with np.load(payload_path, allow_pickle=False) as saved:
        for endpoint in cfg["statistics"]["finite_difference_endpoints"]:
            rows = saved[f"{endpoint}__rows"].astype(np.int64)
            labels = saved[f"{endpoint}__labels"].astype(np.int64)
            coefficients = saved[f"{endpoint}__coefficients"].astype(np.complex128)
            selected: list[int] = []
            for label in range(10):
                selected.extend(np.flatnonzero(labels == label)[:2].tolist())
            require(len(selected) == 20, f"local direction probe has not preserved 20 rows: {endpoint}")
            kind = "small" if endpoint.startswith("small") else "large"
            grid_specs = (("production", 2048 if kind == "small" else 4096),
                          ("reference", 4096 if kind == "small" else 6144))
            grid_size_by_role = dict(grid_specs)
            bases: dict[str, torch.Tensor] = {}
            geometries: dict[str, tuple[int, int, float]] = {}
            for role, grid in grid_specs:
                bases[role], area = dual_grid.load_basis(previous_root, endpoint, role, grid, device)
                bases[role] = bases[role].to(torch.complex128)
                basis_path = previous_root / "bases" / f"{endpoint}_{role}_{grid}.npz"
                with np.load(basis_path, allow_pickle=False) as item:
                    height, width = item["camera_field_per_unit_coefficient"].shape[-2:]
                geometries[role] = (int(height), int(width), float(area))
            for position in selected:
                z = torch.as_tensor(coefficients[position], dtype=torch.complex128, device=device)
                real_z = torch.stack((z.real, z.imag), dim=-1).reshape(-1).to(torch.float64)
                phase_tangent = torch.stack((-z.imag, z.real), dim=-1).reshape(-1).to(torch.float64)
                norm_phase = torch.linalg.vector_norm(phase_tangent)
                if float(norm_phase) > 0.0:
                    unit = phase_tangent / norm_phase
                    q, _ = torch.linalg.qr(unit[:, None], mode="complete")
                    projection = q[:, 1:]
                else:
                    projection = torch.eye(200, dtype=torch.float64, device=device)[:, 1:]
                for exposure in exposure_values:
                    jacobians: dict[str, torch.Tensor] = {}
                    means: dict[str, torch.Tensor] = {}
                    variances: dict[str, torch.Tensor] = {}
                    rates: dict[str, torch.Tensor] = {}
                    for role, _ in grid_specs:
                        j, mu, var, rate = dual_grid._local_jacobian(z, bases[role], geometries[role], camera, exposure)
                        jacobians[role], means[role], variances[role], rates[role] = j, mu, var, rate
                    common_var = 2.0 * torch.maximum(variances["production"], variances["reference"])
                    positive = common_var > 0.0
                    denominator = torch.sqrt(torch.where(positive, common_var, torch.ones_like(common_var))).reshape(-1, 1)
                    weighted = {role: torch.where(positive.reshape(-1, 1), jacobians[role] / denominator,
                                                  torch.zeros_like(jacobians[role])) for role, _ in grid_specs}
                    projected = weighted["production"] @ projection
                    gram = projected.T @ projected
                    eigenvalues, eigenvectors = torch.linalg.eigh((gram + gram.T) * 0.5)
                    direction = projection @ eigenvectors[:, int(torch.argmax(eigenvalues))]
                    direction_complex = torch.complex(direction[0::2], direction[1::2])
                    grid_jacobian_difference = float(torch.linalg.matrix_norm(
                        (weighted["production"] - weighted["reference"]) @ projection, ord=2).detach().cpu())
                    for relative_radius in radii:
                        radius = relative_radius * float(torch.linalg.vector_norm(real_z))
                        for role, _ in grid_specs:
                            field = z @ bases[role]
                            tangent = direction_complex @ bases[role]
                            plus, minus = field + radius * tangent, field - radius * tangent
                            exposure_scale = dual_grid.noise_scale(camera, exposure)
                            dark = float(camera["dark_current_e_per_pixel_s"]) * exposure
                            plus_rate = plus.abs().square() * geometries[role][2] * exposure_scale + dark
                            minus_rate = minus.abs().square() * geometries[role][2] * exposure_scale + dark
                            plus_mean, _ = dual_grid.clipped_camera_moments(plus_rate, camera)
                            minus_mean, _ = dual_grid.clipped_camera_moments(minus_rate, camera)
                            actual = (plus_mean - minus_mean) * 0.5
                            prediction = dual_grid._finite_difference_linear_prediction(jacobians, role, direction, radius)
                            weighted_actual = torch.where(positive, actual / denominator.reshape(-1), torch.zeros_like(actual))
                            weighted_prediction = torch.where(positive, prediction / denominator.reshape(-1), torch.zeros_like(prediction))
                            residual = actual - prediction
                            weighted_residual = weighted_actual - weighted_prediction
                            actual_norm = torch.linalg.vector_norm(actual)
                            pred_norm = torch.linalg.vector_norm(prediction)
                            weighted_actual_norm = torch.linalg.vector_norm(weighted_actual)
                            weighted_pred_norm = torch.linalg.vector_norm(weighted_prediction)
                            weighted_residual_norm = torch.linalg.vector_norm(weighted_residual)
                            rel = float((weighted_residual_norm / weighted_pred_norm).detach().cpu()) if float(weighted_pred_norm) > 0 else None
                            rows_out.append({
                                "endpoint": endpoint, "sample_row": int(rows[position]), "label": int(labels[position]),
                                "grid_role": role, "grid": grid_size_by_role[role], "exposure_s": exposure,
                                "relative_radius": relative_radius, "is_primary_radius": math.isclose(
                                    relative_radius, float(cfg["statistics"]["finite_difference_primary_radius"]),
                                    rel_tol=0.0, abs_tol=0.0),
                                "coefficient_precision": "complex128", "jacobian_source_grid_role": role,
                                "actual_response_norm": float(actual_norm.detach().cpu()),
                                "linear_prediction_norm": float(pred_norm.detach().cpu()),
                                "absolute_residual_norm": float(torch.linalg.vector_norm(residual).detach().cpu()),
                                "white_actual_response_norm": float(weighted_actual_norm.detach().cpu()),
                                "white_linear_prediction_norm": float(weighted_pred_norm.detach().cpu()),
                                "white_residual_norm": float(weighted_residual_norm.detach().cpu()),
                                "relative_linearization_error": rel,
                                "relative_error_defined": rel is not None,
                                "nonlinear_over_5pct": rel is not None and rel > 0.05,
                                "variance_min": float(variances[role].min().detach().cpu()),
                                "variance_max": float(variances[role].max().detach().cpu()),
                                "plus_peak_fullwell_fraction": float((plus_rate >= dual_grid._full_well_integer(camera)).double().mean().detach().cpu()),
                                "minus_peak_fullwell_fraction": float((minus_rate >= dual_grid._full_well_integer(camera)).double().mean().detach().cpu()),
                                "plus_rate_max_e_per_pixel": float(plus_rate.max().detach().cpu()),
                                "minus_rate_max_e_per_pixel": float(minus_rate.max().detach().cpu()),
                                "grid_jacobian_difference_norm": grid_jacobian_difference,
                            })
                del direction_complex, direction, projection, phase_tangent, real_z, z
    expected = 4 * 20 * len(exposure_values) * len(radii) * 2
    require(len(rows_out) == expected, "local finite-difference correction row count changed")
    path = root / "local_direction_recheck" / "finite_difference_controls.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    finite = [row["relative_linearization_error"] for row in rows_out if row["relative_linearization_error"] is not None]
    receipt = {
        "schema": "gaussian_dual_grid_local_finite_difference_recheck.v1",
        "status": "COMPLETED", "endpoints": list(cfg["statistics"]["finite_difference_endpoints"]),
        "fixed_images_per_endpoint": 20, "exposures_s": exposure_values,
        "relative_radii": radii, "primary_radius": float(cfg["statistics"]["finite_difference_primary_radius"]),
        "row_count": len(rows_out), "expected_row_count": expected,
        "each_grid_uses_its_own_jacobian": True,
        "rows_with_defined_relative_error": len(finite),
        "median_relative_error": float(np.median(finite)) if finite else None,
        "p90_relative_error": float(np.quantile(finite, 0.9)) if finite else None,
        "maximum_relative_error": float(max(finite)) if finite else None,
        "nonlinear_over_5pct_count": sum(bool(row["nonlinear_over_5pct"]) for row in rows_out),
        "primary_radius_non_linear_count": sum(bool(row["nonlinear_over_5pct"] and row["is_primary_radius"])
                                                for row in rows_out),
        "material_activation_inference": "not applicable: no material response curve or layer is configured",
        "elapsed_seconds_observed": time.perf_counter() - started,
        "csv": str(path.relative_to(root)), "training_updates": 0, "official_test_reads": 0,
    }
    write_json(root / "local_direction_recheck" / "receipt.json", receipt)
    return receipt


def _write_report(root: Path, report: dict[str, Any]) -> None:
    finite = report["finite_difference_recheck"]
    population = report["population_tables"]
    comparisons = population["paired_comparisons"]
    c_v = comparisons["C45000_to_V45000/production"]
    d_pair = comparisons["D050000_to_Dgrad50000/production"]
    equal = d_pair.get("Dgrad_equal_camera_power_vs_D0_native", {})
    def fmt(v: Any) -> str:
        return "未定义" if v is None else f"{v:.6g}"
    lines = [
        "# Gaussian OLS近期端点整面功率密度与冻结复核",
        "",
        "## 结果与状态",
        "",
        "本轮冻结测量完成，新增训练更新、优化器创建、先验采样、官方TEST读取与物理硬件操作均为0。常规采集接口已接入相关Gaussian训练入口，可在起点/恢复点和每500步检查点记录固定20张DEV探针的十个物理面，且探针不进入损失。本轮没有启动训练，因此不把既有检查点追溯成已覆盖的训练记录；历史检查点和训练日志未改写。",
        f"双网格逐图逐面表含{population['per_image_plane_rows']:,}条、唯一键{population['unique_keys']:,}条；覆盖六个端点、固定5,000张DEV、已保存的G1/G4生成系数及2048/4096两网格。材料输入候选面使用完整相干OLS输出场计算，含面积加权和功率加权强度分位数及最大值。",
        f"局部方向有限扰动复核按每端点20张、4曝光、0.1%/1%/10%扰动、每套网格独立Jacobian完成{finite['row_count']:,}条；主1%扰动中高于5%线性化误差的记录为{finite['primary_radius_non_linear_count']:,}。这表示切线近似的适用范围，不是材料层响应。",
        "",
        "## 科学认识",
        "",
        f"C45k与V45k在相同DEV行上配对。生产网格相机面的功率变化均值为{fmt(c_v['plane_metrics']['camera_physical_roi']['power_w']['mean'])} W，95%区间[{fmt(c_v['plane_metrics']['camera_physical_roi']['power_w']['ci95_low'])}, {fmt(c_v['plane_metrics']['camera_physical_roi']['power_w']['ci95_high'])}]；该值是V减C。对应功率密度、有效面积和图像重构指标详见`tables/paired_endpoint_comparisons.json`。",
        f"Dgrad减D0的原生DEV MSE差为{fmt(d_pair['native_reconstruction']['mse']['mean'])}，95%区间[{fmt(d_pair['native_reconstruction']['mse']['ci95_low'])}, {fmt(d_pair['native_reconstruction']['mse']['ci95_high'])}]；将Dgrad相机物理强度逐图匹配到D0总功率后，MSE差为{fmt(equal.get('mse', {}).get('mean'))}，区间[{fmt(equal.get('mse', {}).get('ci95_low'))}, {fmt(equal.get('mse', {}).get('ci95_high'))}]。这区分亮度/吞吐与空间分布及读出适配，但仍是冻结反事实，不是独立训练干预。",
        "复现seed51/seed52与C/V具有不同训练来源；各自完整人口统计不形成跨种子因果排序。G1/G4使用已保存系数，不新采样。生成标签只作条件分层，不被当作像素重构真值。",
        "指标均按面定义：几何平均密度=P/A_geom；光强有效面积=P²/∫I²dA；有效面积密度=P/A_eff。光纤输出/Decoder入口是同一声明平面的别名，Decoder relay后的相机面另行测量。量纲同时给出W/m²和W/cm²。",
        "",
        "## 目标缺口",
        "",
        "材料名称、薄膜实际位置、实测T(I)响应曲线、偏振与时间条件均未确定，因此不设置激活阈值，也不从相机裁剪或代理噪声推断拓扑材料非线性。整面总功率、有效面积及强度分位数仅描述入射工作范围，无法代替材料响应与耦合标定。真实相机暗帧/平场/QE/增益和物理像素参考面仍待标定。",
        "当前测量使用模型物理场和指定2048/4096网格；跨网格差异作为敏感性报告，不能宣称连续场收敛。复现结果仍限于固定DEV与已保存生成队列，不代表官方TEST泛化。",
        "",
        "## 工程完成与验证",
        "",
        f"预检通过：同网格直算/相干合成最大复场相对误差{report['preflight']['max_same_grid_complex_relative_error']:.3g}，MSE绝对差{report['preflight']['max_direct_vs_synthesized_mse_absolute_error']:.3g}，生产功率链误差{report['preflight']['max_power_trace_relative_error']:.3g}，独立NumPy面积积分误差{report['preflight']['max_independent_numpy_area_integral_relative_error']:.3g}。原先参考网格有限扰动误用生产网格Jacobian已修复；现在reference直接使用reference Jacobian。",
        "C45k、D0、Dgrad的冻结Encoder/OLS张量逐项相同；D0与Dgrad只在Decoder相位不同。生成系数和proposal ID已逐项核对。分块回读和最终逐图逐面唯一键检查均通过。训练接口的起点/每500步coverage消费者和恢复去重经过定向回归验证；本轮没有实际训练探针日志可作覆盖证明。",
        f"全量逐图表：`{population['table']}`；按类统计：`tables/plane_density_summary.csv`；近期配对反事实：`tables/paired_endpoint_comparisons.json`；功率/有效面积对数分解：`tables/power_area_log_decomposition.csv`；有限扰动修正：`local_direction_recheck/finite_difference_controls.csv.gz`。",
        "",
        "## 决策影响与下一步",
        "",
        "现有采集接口现在能把整面功率密度作为未来Gaussian训练检查点的常规诊断写入；它保持为观测量，不自动调节训练。材料实验前应先确定实际材料输入面和实测响应曲线，再用该面的强度分布而不是一个平均值计算非线性输出。对近期Decoder梯度变化，应结合等相机功率反事实判断，避免把输出变暗带来的裁剪下降误读为材料或耦合改善。",
        "本轮结论只支持冻结端点的数值诊断。下一步建议是补真实相机标定和材料曲线；此建议不自动授权增加训练、修改损失/几何或部署材料。历史VALID_FAILED状态与人眼评价状态不变。",
        "",
    ]
    (root / "中文诊断报告.md").write_text("\n".join(lines), encoding="utf-8")


def finalize_saved_measurement(cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    preflight_path = root / "preflight" / "receipt.json"
    report_path = root / "machine_report.json"
    finite_path = root / "local_direction_recheck" / "receipt.json"
    finite_csv = root / "local_direction_recheck" / "finite_difference_controls.csv.gz"
    require(preflight_path.is_file() and report_path.is_file(),
            "saved preflight and machine report are required for finalization")
    require(finite_path.is_file() and finite_csv.is_file(),
            "saved local-direction receipt and rows are required for finalization")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    finite = json.loads(finite_path.read_text(encoding="utf-8"))
    require(preflight.get("task_id") == TASK_ID and preflight.get("status") == "READY",
            "saved preflight receipt is not ready for this task")
    require(report.get("task_id") == TASK_ID and report.get("status") == "VALID_FROZEN_DIAGNOSTIC_COMPLETE",
            "saved machine report is not a completed frozen diagnostic")
    require(report.get("identity_checks") == preflight.get("identity_checks"),
            "saved report identity does not match the approved preflight")
    require(report.get("training_updates") == 0 and report.get("optimizers_created") == 0
            and report.get("prior_samples") == 0 and report.get("official_test_reads") == 0,
            "saved report crossed the frozen diagnostic boundary")
    population = report.get("population_tables", {})
    expected_population_rows = population.get("expected_per_image_plane_rows")
    require(population.get("status") == "VERIFIED"
            and population.get("per_image_plane_rows") == expected_population_rows
            and population.get("unique_keys") == expected_population_rows,
            "saved population table receipt is incomplete or has duplicate keys")
    for relative in (population.get("table"), "tables/plane_density_summary.csv",
                     "tables/paired_endpoint_comparisons.json", "tables/power_area_log_decomposition.csv"):
        require(isinstance(relative, str) and (root / relative).is_file(),
                f"verified population output is missing: {relative}")
    comparisons_path = root / "tables" / "paired_endpoint_comparisons.json"
    require(json.loads(comparisons_path.read_text(encoding="utf-8")) == population.get("paired_comparisons"),
            "saved paired comparisons differ from the machine report")
    expected_finite_rows = (
        len(finite["endpoints"]) * finite["fixed_images_per_endpoint"] * len(finite["exposures_s"])
        * len(finite["relative_radii"]) * len(GRIDS)
    )
    require(finite.get("status") == "COMPLETED" and finite.get("row_count") == expected_finite_rows
            and finite.get("expected_row_count") == expected_finite_rows
            and report.get("finite_difference_recheck") == finite,
            "saved local-direction recheck is incomplete or differs from the machine report")
    with gzip.open(finite_csv, "rt", newline="", encoding="utf-8") as stream:
        finite_rows = sum(1 for _ in csv.DictReader(stream))
    require(finite_rows == expected_finite_rows,
            f"local-direction rows {finite_rows} != expected {expected_finite_rows}")
    _write_report(root, report)
    completion = {
        "schema": "gaussian_recent_plane_density_measurement_complete.v1", "task_id": TASK_ID,
        "status": report["status"], "per_image_plane_rows": population["per_image_plane_rows"],
        "unique_keys": population["unique_keys"], "expected_rows": expected_population_rows,
        "finite_difference_rows": finite_rows,
        "all_population_chunks_readback_verified": True,
        "training_probe_integration": "IMPLEMENTED_AND_TESTED",
        "actual_training_probe_run_in_this_attempt": False,
        "training_updates": 0, "optimizers_created": 0, "prior_samples": 0,
        "official_test_reads": 0, "hardware_operations": 0,
        "complete_report_and_receipts_written": True,
    }
    write_json(root / "measurement_complete.json", completion)
    return completion


def run_measurement(cfg: dict[str, Any], root: Path, device: torch.device) -> dict[str, Any]:
    preflight_path = root / "preflight" / "receipt.json"
    require(preflight_path.is_file(), "READY preflight is required before population measurement")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight.get("status") == "READY" and preflight.get("task_id") == TASK_ID,
            "saved preflight receipt is not ready for this task")
    fixed, cohorts = _preflight_common(cfg, root, device)
    require(fixed["identity_receipt"] == preflight["identity_checks"],
            "checkpoint or cohort identities changed after preflight")
    started = time.perf_counter()
    dev_results: list[dict[str, Any]] = []
    for grid_role, grid in GRIDS:
        for endpoint in UPSTREAM_ENDPOINTS:
            dev_results.append(_measure_direct_dev(cfg, root, fixed, endpoint, grid_role, grid))
        dev_results.append(_measure_decoder_pair(cfg, root, fixed, grid_role, grid))
    generated_results = _measure_generated(cfg, root, fixed, cohorts)
    population = _write_population_tables(cfg, root, fixed, cohorts)
    finite_difference = _finite_difference_recheck(cfg, root, device)
    report = {
        "schema": "gaussian_recent_plane_density_machine_report.v1", "task_id": TASK_ID,
        "authority": AUTHORITY, "status": "VALID_FROZEN_DIAGNOSTIC_COMPLETE",
        "run_dir": str(root.relative_to(ROOT)), "preflight": preflight,
        "endpoint_population": {key: fixed["endpoint_specs"][key] for key in ENDPOINTS},
        "real_input_DEV": {"count_per_endpoint": 5000, "class_count": 500, "split_seed": 1234,
                           "grids": {role: grid for role, grid in GRIDS}, "official_test_reads": 0},
        "generation_population": generated_results,
        "DEV_plane_measurements": dev_results,
        "population_tables": population,
        "finite_difference_recheck": finite_difference,
        "identity_checks": fixed["identity_receipt"],
        "training_updates": 0, "optimizers_created": 0, "prior_samples": 0,
        "official_test_reads": 0, "hardware_operations": 0,
        "elapsed_seconds_observed": time.perf_counter() - started,
        "limitations": [
            "material response curve and actual material plane are unspecified",
            "camera noise is a proxy; measured noise calibration is absent",
            "cross-grid differences are reported as sensitivity, not continuous-grid convergence",
            "C/V/replication/Decoder endpoints retain their original data roles and are not one controlled architecture ranking",
            "saved G1/G4 generated labels are conditions, not reconstruction ground truth",
        ],
    }
    write_json(root / "machine_report.json", report)
    _write_report(root, report)
    completion = {
        "schema": "gaussian_recent_plane_density_measurement_complete.v1", "task_id": TASK_ID,
        "status": report["status"], "per_image_plane_rows": population["per_image_plane_rows"],
        "unique_keys": population["unique_keys"], "expected_rows": population["expected_per_image_plane_rows"],
        "finite_difference_rows": finite_difference["row_count"],
        "all_population_chunks_readback_verified": True, "training_probe_integration": "IMPLEMENTED_AND_TESTED",
        "training_updates": 0, "optimizers_created": 0, "prior_samples": 0,
        "official_test_reads": 0, "hardware_operations": 0,
        "actual_training_probe_run_in_this_attempt": False,
        "complete_report_and_receipts_written": True,
    }
    write_json(root / "measurement_complete.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research/lightgen/gaussian_recent_plane_density_audit_v1.yaml"))
    parser.add_argument("--mode", choices=("preflight", "measure", "finalize"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    cfg = read_config(ROOT / args.config)
    root = ROOT / cfg["run_dir"]
    device = torch.device(args.device)
    root.mkdir(parents=True, exist_ok=True)
    if args.mode == "finalize":
        completion = finalize_saved_measurement(cfg, root)
        print(json.dumps(clean(completion), ensure_ascii=False), flush=True)
        return
    require(device.type == "cuda" and torch.cuda.is_available(), "approved recent Gaussian diagnostic requires CUDA")
    if args.mode == "preflight":
        receipt = run_preflight(cfg, root, device)
        print(json.dumps(clean({"status": receipt["status"], "output": str(root),
                                "max_complex_error": receipt["max_same_grid_complex_relative_error"]}), ensure_ascii=False), flush=True)
        return
    completion = run_measurement(cfg, root, device)
    print(json.dumps(clean(completion), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
