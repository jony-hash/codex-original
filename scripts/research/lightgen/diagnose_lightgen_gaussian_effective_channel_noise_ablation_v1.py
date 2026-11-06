"""Frozen OLS effective-channel, covariance, and proxy-camera diagnosis.

All optical responses are generated through the existing production Decoder
paths.  The saved complex camera basis is then used only as a linear identity
for frozen coefficient ablations; the complex fields are summed before
intensity, pixel integration, camera gain, clipping, and pooling.
"""

from __future__ import annotations

import argparse
import csv
import cmath
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

from scripts.research.lightgen import (
    diagnose_chen_lightgen_400_quickcheck_v1 as chen_quickcheck,
    diagnose_lightgen_gaussian_real_input_power_attribution_v1 as previous,
)
from scripts.research.lightgen import execute_chen_lightgen_400_mnist_baseline_v1 as chen_base
import ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline as chen_baseline_module
import ldm_project.v2.optics.d2nn as d2nn_module
from ldm_project.v2.application.research.lightgen.chen_lightgen_400_baseline import (
    ACTIVE_SHAPE as CHEN_ACTIVE_SHAPE,
    PIXEL_PITCH_M as CHEN_PIXEL_PITCH_M,
    WAVELENGTH_M as WAVELENGTH_M,
    ChenLightGen400Baseline,
    LightGen400Config,
    _center_crop as chen_center_crop,
)
from ldm_project.v2.contracts import ComplexField, OperationProvenance, PlaneSpec
from ldm_project.v2.application.research.lightgen.lightgen_2d_semantic_generation_input_ols_alignment import (
    ComplexOLSState,
)
from ldm_project.v2.application.research.lightgen.effective_channel_statistics import (
    complex_covariances,
    complex_from_interleaved_real_covariance,
    fdr_bh,
    maximum_port_gauge,
    stratified_bootstrap,
    noise_population_summary,
)
from ldm_project.v2.application.research.lightgen.effective_channel_reporting import (
    preflight_status,
    render_effective_channel_report,
)
from ldm_project.v2.evidence.io import (
    diagnostic_json_value as clean,
    write_diagnostic_csv as write_csv,
    write_diagnostic_json as write_json,
    write_diagnostic_npz as write_npz,
)
from ldm_project.v2.application.research.lightgen.native_image_inputs import image_batch
from ldm_project.v2.application.research.lightgen.effective_channel_figures import create_figures


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "GAUSSIAN_OLS_EFFECTIVE_CHANNEL_NOISE_ABLATION_V1"
AUTHORITY_ID = "USER_APPROVED_GAUSSIAN_OLS_EFFECTIVE_CHANNEL_NOISE_ABLATION_20261021"
PORTS = 100
CHANNELS = ("raw_zero", "equal_ols_power")
METRICS = (
    "mse28",
    "psnr28_db",
    "ssim28",
    "production_reconstruction_loss",
    "clipping_fraction",
    "camera_total_w",
)
RELATIVE_THRESHOLDS = (0.1, 0.01, 0.001, 0.0001)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(cfg, dict), "task config must be a YAML object")
    require(cfg.get("schema") == "gaussian_ols_effective_channel_noise_ablation.v1", "task config schema changed")
    require(cfg.get("task_id") == TASK_ID and cfg.get("semantic_authority_id") == AUTHORITY_ID, "task authority changed")
    require(cfg.get("evidence_tier") == "diagnostic-only", "this task must remain diagnostic-only")
    require(cfg["data"]["small_rows"] == 5000 and cfg["data"]["small_per_class"] == 500, "small DEV population changed")
    require(cfg["data"]["large_rows"] == 300 and cfg["data"]["large_per_class"] == 30, "large TRAIN population changed")
    require(cfg["data"]["official_test_reads"] == 0, "official TEST reads are forbidden")
    require(cfg["small"]["production_grid"] == 2048 and cfg["small"]["reference_grid"] == 4096, "small propagation windows changed")
    require(cfg["large"]["source_encoder_grid"] == 2048 and cfg["large"]["decoder_production_grid"] == 4096 and cfg["large"]["decoder_reference_grid"] == 6144, "large Decoder windows or source coefficients changed")
    require(cfg["statistics"]["port_count"] == PORTS and cfg["statistics"]["bootstrap_resamples"] == 2000 and cfg["statistics"]["bootstrap_seed"] == 2026102101, "population or statistics contract changed")
    require(cfg["statistics"]["no_training_updates"] and cfg["statistics"]["no_optimizer_creation"] and cfg["statistics"]["no_prior_sampling"] and cfg["statistics"]["no_hardware_operations"], "zero-update boundary changed")
    endpoints = cfg["small"]["decoder_endpoints"]
    require([(item["key"], item["source_state"], item["global_step"]) for item in endpoints] == [
        ("small_step_030000", "step_030000", 30000),
        ("small_control_step_035000", "control_step_035000", 35000),
        ("small_treatment_step_035000", "treatment_step_035000", 35000),
    ], "small Decoder endpoints changed")
    return cfg


def load_small_context(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    previous_root = ROOT / cfg["data"]["small_population"]
    # Resolve the prior task's exact config from the repository root, never through latest.
    old_cfg = previous.load_config(ROOT / "configs/research/lightgen/gaussian_real_input_power_attribution_v1.yaml")
    previous_machine = json.loads((previous_root / "machine_report.json").read_text(encoding="utf-8"))
    require(previous_machine.get("status") == "VALID_DIAGNOSTIC_COMPLETE", "source small Gaussian measurement is incomplete")
    baseline_cfg, continuation_cfg, checkpoints, pixels, labels, train_rows, dev_rows = previous.load_inputs(old_cfg)
    require(len(dev_rows) == 5000 and torch.equal(torch.bincount(labels[dev_rows], minlength=10), torch.full((10,), 500)), "small fixed DEV identity changed")
    require(not bool(torch.isin(train_rows, dev_rows).any()), "small DEV rows overlap TRAIN")
    state_payloads: dict[str, dict[str, np.ndarray]] = {}
    for state_key in cfg["small"]["covariance_states"]:
        state_payloads[state_key] = previous.read_state_chunks(
            previous_root / "states" / state_key,
            dev_rows.detach().cpu().numpy().astype(np.int64),
            int(old_cfg["statistics"]["shard_size"]),
        )
    return {
        "previous_root": previous_root,
        "previous_machine_report": previous_machine,
        "old_config": old_cfg,
        "baseline_config": baseline_cfg,
        "continuation_config": continuation_cfg,
        "checkpoints": checkpoints,
        "pixels": pixels,
        "labels": labels,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "state_payloads": state_payloads,
        "device": device,
    }


def load_large_context(cfg: dict[str, Any], device: torch.device) -> dict[str, Any]:
    spec = cfg["large"]
    checkpoint_path = ROOT / spec["checkpoint"]
    report_path = ROOT / spec["source_report"]
    source_config_path = ROOT / spec["source_config"]
    fixed_population_path = ROOT / spec["fixed_population"]
    moments_path = ROOT / spec["full_train_moments"]
    for path in (checkpoint_path, report_path, source_config_path, fixed_population_path, moments_path):
        require(path.is_file(), f"explicit large-architecture input is missing: {path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    source_config = chen_base._read_config(source_config_path)
    require(checkpoint.get("schema") == "chen_lightgen_400_mnist_baseline_checkpoint.v1" and checkpoint.get("completed_updates") == 30000, "large source checkpoint identity or step changed")
    require(checkpoint.get("task_id") == chen_base.TASK_ID, "large source checkpoint belongs to another task")
    require(checkpoint.get("model_identity") == {"encoder_layers": 2, "decoder_layers": 4, "phase_shape": [400, 400], "ols": "gaussian_s12_v1"}, "large source model identity changed")
    require(source_report.get("status") == "VALID_FAILED" and source_report.get("task_id") == "CHEN_LIGHTGEN_400_QUICKCHECK_4096_V2", "historical large quickcheck identity changed")
    require(source_report.get("source_checkpoint_step") == 30000 and source_report.get("source_encoder_grid") == 2048 and source_report.get("decoder_diagnostic_grid") == 4096 and source_report.get("official_test_reads") == 0, "large saved coefficient lineage changed")
    expected_config = dict(source_config)
    expected_config["optics"] = dict(source_config["optics"])
    expected_config["optics"]["selected_production_grid"] = 2048
    require(checkpoint.get("config") == expected_config, "large source checkpoint differs from its explicit config")
    references = checkpoint.get("references")
    require(isinstance(references, dict) and set(references) == {"scale", "power_reference", "camera_gain"}, "large checkpoint references changed")
    scale = references["scale"].detach().cpu().to(torch.float64)
    gain = float(references["camera_gain"].reshape(()))
    require(scale.shape == (PORTS,) and bool(torch.isfinite(scale).all()) and bool((scale > 0).all()) and math.isfinite(gain) and gain > 0, "large checkpoint scales or fixed gain are invalid")
    optimizer = checkpoint.get("optimizer")
    require(isinstance(optimizer, dict) and isinstance(optimizer.get("state"), dict), "saved historical optimizer state is missing")
    saved_adam_steps = []
    for value in optimizer["state"].values():
        step = value.get("step") if isinstance(value, dict) else None
        if step is not None:
            saved_adam_steps.append(int(step.item()) if isinstance(step, torch.Tensor) else int(step))
    require(saved_adam_steps and all(step == 30000 for step in saved_adam_steps), "large checkpoint's saved Adam steps changed")

    pixels, labels, train_rows, dev_rows = chen_base._load_data(ROOT / cfg["data"]["source_file"])
    population = chen_quickcheck._load_fixed_population(fixed_population_path, labels, train_rows)
    require(tuple(population["coefficients"].shape) == (300, PORTS), "large fixed TRAIN coefficient shape changed")
    require(torch.equal(torch.bincount(population["all_labels"], minlength=10), torch.full((10,), 30)), "large fixed TRAIN population is not 30/class")
    require(bool(torch.isin(population["all_rows"], train_rows).all()) and not bool(torch.isin(population["all_rows"], dev_rows).any()), "large fixed population roles changed")
    require(torch.equal(population["all_labels"], labels[population["all_rows"]]), "large saved row labels differ from canonical TRAIN")
    source_model = ChenLightGen400Baseline(LightGen400Config(2048, True)).to(device)
    source_model.load_state_dict(checkpoint["model"], strict=True)
    source_model.eval()
    for parameter in source_model.parameters():
        parameter.requires_grad_(False)
    require(not source_model.training and not any(parameter.requires_grad for parameter in source_model.parameters()), "large source model did not freeze")
    return {
        "checkpoint_path": checkpoint_path,
        "source_report_path": report_path,
        "source_config_path": source_config_path,
        "fixed_population_path": fixed_population_path,
        "moments_path": moments_path,
        "checkpoint": checkpoint,
        "source_report": source_report,
        "source_config": source_config,
        "references": references,
        "scale": scale,
        "camera_gain": gain,
        "saved_adam_steps": saved_adam_steps,
        "pixels": pixels,
        "labels": labels,
        "train_rows": train_rows,
        "dev_rows": dev_rows,
        "population": population,
        "source_model": source_model,
        "device": device,
    }


def read_noise_proxy(cfg: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / cfg["noise_proxy_config"]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(payload.get("schema") == "ols_camera_training_interface_closure.v10", "camera proxy config schema changed")
    camera = payload["camera"]
    require(camera == {
        "quantum_efficiency": 0.5,
        "read_noise_e_rms": 1.0,
        "dark_current_e_per_pixel_s": 0.0,
        "full_well_e": 30000.0,
        "exposure_sweep_s": [0.00001, 0.0001, 0.001, 0.01],
    }, "camera proxy settings changed")
    return camera


def spectrum_summary(covariance: np.ndarray) -> dict[str, Any]:
    symmetric = (np.asarray(covariance, dtype=np.float64) + np.asarray(covariance, dtype=np.float64).T) * 0.5
    eigenvalues = np.linalg.eigvalsh(symmetric)[::-1]
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total = float(eigenvalues.sum())
    squares = float(np.square(eigenvalues).sum())
    fractions = eigenvalues / total if total > 0 else np.zeros_like(eigenvalues)
    cumulative = np.cumsum(fractions)
    return {
        "eigenvalues": eigenvalues,
        "variance_fraction": fractions,
        "effective_participation": float(total * total / squares) if squares > 0 else None,
        "directions_95_percent": int(np.searchsorted(cumulative, 0.95) + 1) if total > 0 else 0,
        "directions_99_percent": int(np.searchsorted(cumulative, 0.99) + 1) if total > 0 else 0,
        "total_variance": total,
    }


def covariance_record(population: str, coefficients: np.ndarray, labels: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    z = np.asarray(coefficients, dtype=np.complex128)
    y = np.asarray(labels, dtype=np.int64)
    require(z.ndim == 2 and z.shape[1] == PORTS and y.shape == (len(z),), f"covariance input shape mismatch: {population}")
    raw = complex_covariances(z)
    gauge_z = maximum_port_gauge(z)
    gauge_mean = np.empty(2 * PORTS, dtype=np.float64)
    gauge_features = np.empty((len(z), 2 * PORTS), dtype=np.float64)
    gauge_features[:, 0::2] = gauge_z.real
    gauge_features[:, 1::2] = gauge_z.imag
    gauge_mean[:] = gauge_features.mean(axis=0)
    centered = gauge_features - gauge_mean
    real_covariance = centered.T @ centered / len(z)
    gauge_complex_covariance = complex_from_interleaved_real_covariance(real_covariance)
    gauge_complex_second = gauge_complex_covariance + np.outer(gauge_z.mean(axis=0), gauge_z.mean(axis=0).conj())
    spectrum = spectrum_summary(real_covariance)
    per_class: dict[str, Any] = {}
    within = np.zeros_like(real_covariance)
    overall_mean = gauge_mean.copy()
    between = np.zeros_like(real_covariance)
    class_counts: dict[str, int] = {}
    for label in range(10):
        indices = np.flatnonzero(y == label)
        require(len(indices) > 1, f"class {label} is too small for covariance: {population}")
        class_counts[str(label)] = int(len(indices))
        class_z = z[indices]
        class_complex = complex_covariances(class_z)
        class_gauge = maximum_port_gauge(class_z)
        class_features = np.empty((len(indices), 2 * PORTS), dtype=np.float64)
        class_features[:, 0::2] = class_gauge.real
        class_features[:, 1::2] = class_gauge.imag
        class_mu = class_features.mean(axis=0)
        class_centered = class_features - class_mu
        class_cov = class_centered.T @ class_centered / len(indices)
        within += len(indices) / len(z) * class_cov
        delta = class_mu - overall_mean
        between += len(indices) / len(z) * np.outer(delta, delta)
        class_spectrum = spectrum_summary(class_cov)
        per_class[str(label)] = {
            "count": int(len(indices)),
            "raw_complex_covariance_trace_w": float(np.trace(class_complex["covariance"]).real),
            "gauge_real_imag_covariance_trace_w": class_spectrum["total_variance"],
            "effective_participation": class_spectrum["effective_participation"],
            "directions_95_percent": class_spectrum["directions_95_percent"],
            "directions_99_percent": class_spectrum["directions_99_percent"],
            "top10_eigenvalue_fraction": float(class_spectrum["variance_fraction"][:10].sum()),
        }
    aggregate = {
        "population": population,
        "sample_count": int(len(z)),
        "class_counts": class_counts,
        "raw_complex_covariance_trace_w": float(np.trace(raw["covariance"]).real),
        "raw_complex_second_moment_trace_w": float(np.trace(raw["second_moment"]).real),
        "gauge_real_imag_covariance_trace_w": spectrum["total_variance"],
        "gauge_complex_covariance_trace_w": float(np.trace(gauge_complex_covariance).real),
        "gauge_complex_second_moment_trace_w": float(np.trace(gauge_complex_second).real),
        "effective_participation": spectrum["effective_participation"],
        "directions_95_percent": spectrum["directions_95_percent"],
        "directions_99_percent": spectrum["directions_99_percent"],
        "top10_eigenvalue_fraction": float(spectrum["variance_fraction"][:10].sum()),
        "within_class_gauge_variance": float(np.trace(within)),
        "between_class_gauge_variance": float(np.trace(between)),
        "within_class_fraction_of_total": float(np.trace(within) / np.trace(within + between)) if np.trace(within + between) > 0 else None,
        "between_class_fraction_of_total": float(np.trace(between) / np.trace(within + between)) if np.trace(within + between) > 0 else None,
        "per_class": per_class,
    }
    arrays = {
        "raw_complex_mean": raw["mean"],
        "raw_complex_covariance": raw["covariance"],
        "raw_complex_second_moment": raw["second_moment"],
        "maxport_gauge_real_imag_mean": gauge_mean,
        "maxport_gauge_real_imag_covariance": real_covariance,
        "maxport_gauge_complex_covariance": gauge_complex_covariance,
        "maxport_gauge_complex_second_moment": gauge_complex_second,
        "eigenvalues": spectrum["eigenvalues"],
        "variance_fraction": spectrum["variance_fraction"],
    }
    return aggregate, arrays


def large_saved_train_covariance(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    saved = torch.load(context["moments_path"], map_location="cpu", weights_only=False)
    require(saved.get("schema") == "chen_lightgen_400_empirical_complex_gaussian_samples.v1", "large saved moment schema changed")
    class_models = saved.get("class_models")
    require(isinstance(class_models, dict) and set(class_models) == set(range(10)), "large saved class Gaussian moments are incomplete")
    scale = context["scale"].numpy()
    transform = np.repeat(np.sqrt(scale), 2)
    pixels, labels, train_rows = context["pixels"], context["labels"], context["train_rows"]
    counts = torch.bincount(labels[train_rows], minlength=10).numpy().astype(np.int64)
    require(int(counts.sum()) == 55000 and bool(np.all(counts > 1)), "large saved moment population is not the fixed 55,000 TRAIN set")
    classes: dict[str, dict[str, Any]] = {}
    class_mu: list[np.ndarray] = []
    class_cov: list[np.ndarray] = []
    for label in range(10):
        model = class_models[label]
        mu_normalized = model["mean"].detach().cpu().numpy().astype(np.float64)
        covariance_normalized = model["covariance"].detach().cpu().numpy().astype(np.float64)
        ridge = float(model["ridge"].reshape(()))
        cholesky = model["cholesky"].detach().cpu().numpy().astype(np.float64)
        require(mu_normalized.shape == (200,) and covariance_normalized.shape == (200, 200) and cholesky.shape == (200, 200), f"class {label} saved moment dimensions changed")
        require(math.isfinite(ridge) and ridge > 0, f"class {label} covariance ridge is invalid")
        reconstructed = cholesky @ cholesky.T
        expected_regularized = covariance_normalized + ridge * np.eye(200, dtype=np.float64)
        ridge_error = float(np.linalg.norm(reconstructed - expected_regularized) / np.linalg.norm(expected_regularized))
        require(ridge_error <= 2e-10, f"class {label} saved covariance/ridge recovery check failed: {ridge_error}")
        # The saved covariance field is the unregularized unbiased covariance;
        # ridge is used only by the saved Cholesky sampling factor.
        mu = mu_normalized * transform
        covariance = covariance_normalized * np.outer(transform, transform)
        class_mu.append(mu)
        class_cov.append(covariance)
        classes[str(label)] = {
            "count": int(counts[label]),
            "saved_ridge_normalized": ridge,
            "ridge_is_in_saved_covariance": False,
            "cholesky_regularized_identity_relative_error": ridge_error,
            "unregularized_absolute_covariance_trace_w": float(np.trace(covariance)),
            "effective_participation": spectrum_summary(covariance)["effective_participation"],
        }
    total_count = int(counts.sum())
    mean = sum(counts[label] * class_mu[label] for label in range(10)) / total_count
    within = np.zeros((200, 200), dtype=np.float64)
    between = np.zeros_like(within)
    for label in range(10):
        # Convert the saved unbiased sample covariance to population covariance
        # before applying the law of total covariance.
        within += counts[label] / total_count * class_cov[label] * ((counts[label] - 1) / counts[label])
        delta = class_mu[label] - mean
        between += counts[label] / total_count * np.outer(delta, delta)
    total_cov = within + between
    spectrum = spectrum_summary(total_cov)
    class_moments: dict[str, Any] = {
        "sample_count": total_count,
        "class_counts": {str(k): int(v) for k, v in enumerate(counts)},
        "gauge_real_imag_covariance_trace_w": float(np.trace(total_cov)),
        "within_class_gauge_variance": float(np.trace(within)),
        "between_class_gauge_variance": float(np.trace(between)),
        "within_class_fraction_of_total": float(np.trace(within) / np.trace(total_cov)),
        "between_class_fraction_of_total": float(np.trace(between) / np.trace(total_cov)),
        "effective_participation": spectrum["effective_participation"],
        "directions_95_percent": spectrum["directions_95_percent"],
        "directions_99_percent": spectrum["directions_99_percent"],
        "top10_eigenvalue_fraction": float(spectrum["variance_fraction"][:10].sum()),
        "per_class": classes,
        "recovery": {
            "scale_source": "source checkpoint references.scale",
            "scale_transform": "undo maxport-gauge feature division by sqrt(scale_j) using repeated real/imag diagonal",
            "ridge": "saved covariance is unregularized; ridge is verified as the separate diagonal added before its saved Cholesky",
            "original_ungauged_complex_cross_moment_available": False,
        },
    }
    complex_cov = complex_from_interleaved_real_covariance(total_cov)
    complex_mean = mean[0::2] + 1j * mean[1::2]
    complex_second = complex_cov + np.outer(complex_mean, complex_mean.conj())
    arrays = {
        "maxport_gauge_real_imag_mean": mean,
        "maxport_gauge_real_imag_covariance": total_cov,
        "within_class_real_imag_covariance": within,
        "between_class_real_imag_covariance": between,
        "maxport_gauge_complex_mean": complex_mean,
        "maxport_gauge_complex_covariance": complex_cov,
        "maxport_gauge_complex_second_moment": complex_second,
        "eigenvalues": spectrum["eigenvalues"],
        "variance_fraction": spectrum["variance_fraction"],
    }
    return class_moments, arrays


def run_covariances(cfg: dict[str, Any], small: dict[str, Any], large: dict[str, Any], output_root: Path) -> dict[str, Any]:
    covariance_root = output_root / "covariance"
    covariance_root.mkdir(parents=True, exist_ok=True)
    records: dict[str, Any] = {}
    for state_key, payload in small["state_payloads"].items():
        name = f"small_{state_key}"
        record, arrays = covariance_record(name, payload["coefficients"], payload["labels"])
        records[name] = record
        write_npz(covariance_root / f"{name}.npz", **arrays)
    large_coeff = large["population"]["coefficients"].detach().cpu().numpy()
    large_labels = large["population"]["all_labels"].detach().cpu().numpy()
    record, arrays = covariance_record("large_fixed300_train", large_coeff, large_labels)
    records["large_fixed300_train"] = record
    write_npz(covariance_root / "large_fixed300_train.npz", **arrays)
    full_record, full_arrays = large_saved_train_covariance(large)
    records["large_saved55000_train"] = full_record
    write_npz(covariance_root / "large_saved55000_train.npz", **full_arrays)
    write_json(covariance_root / "summary.json", records)
    spectrum_rows: list[dict[str, Any]] = []
    for population, record in records.items():
        with np.load(covariance_root / f"{population}.npz", allow_pickle=False) as arrays:
            eigenvalues = arrays["eigenvalues"]
            fractions = arrays["variance_fraction"]
        for direction, (value, fraction) in enumerate(zip(eigenvalues, fractions, strict=True), start=1):
            spectrum_rows.append({"population": population, "direction": direction, "eigenvalue_w": float(value), "variance_fraction": float(fraction), "cumulative_variance_fraction": float(np.sum(fractions[:direction]))})
    write_csv(covariance_root / "spectra.csv", spectrum_rows)
    return records


def frozen_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def assert_frozen_snapshot(model: torch.nn.Module, before: dict[str, torch.Tensor], name: str) -> None:
    current = model.state_dict()
    require(current.keys() == before.keys(), f"model state keys changed during {name}")
    for key, value in current.items():
        require(torch.equal(value.detach().cpu(), before[key]), f"frozen parameter or buffer changed: {name}.{key}")


def small_model_for(key: str, cfg: dict[str, Any], small: dict[str, Any], grid: int, device: torch.device) -> torch.nn.Module:
    return previous.model_for(key, small["old_config"], small["baseline_config"], small["checkpoints"], device, grid=grid)


def large_model_for(large: dict[str, Any], grid: int) -> ChenLightGen400Baseline:
    model = chen_quickcheck._diagnostic_model(large["source_model"], grid)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    require(not model.training and not any(parameter.requires_grad for parameter in model.parameters()), f"large Decoder did not freeze at grid {grid}")
    return model


def camera_field_from_coefficients(kind: str, model: torch.nn.Module, coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return production complex camera field, direct physical W/pixel, and pixel area."""
    with torch.inference_mode():
        if kind == "small":
            state = ComplexOLSState(
                coefficients=coefficients,
                source="effective_channel_frozen_diagnostic",
                basis_id=model.runtime.ols.basis_id,
                mode_geometry=model.runtime.ols.mode_identity(),
            )
            # The production adapter also supports image-to-fiber absolute-power
            # tracing. This diagnostic starts at the already saved OLS field, so
            # use the same decoder-only mode as the prior fixed-coefficient audit.
            # The 0.889 layer transmissions are already captured by the frozen
            # phase stacks and remain unchanged.
            saved_physical_optics = model.runtime.physical_optics
            require(saved_physical_optics is not None, "small decoder lost its production physical configuration")
            model.runtime.physical_optics = None
            try:
                result = model.runtime.forward_from_ols(state)
            finally:
                model.runtime.physical_optics = saved_physical_optics
            field = model.runtime.output_relay(result.generator_output_field).tensor
            area = float(model.runtime.camera_plane.pixel_pitch_m[0] * model.runtime.camera_plane.pixel_pitch_m[1])
            direct_w_px = result.raw_camera.tensor * area
        else:
            captured: dict[str, torch.Tensor] = {}

            def capture(name: str, value: Any) -> None:
                if name == "camera_full_field":
                    captured[name] = value.tensor

            result = model.forward_from_ols(coefficients, field_callback=capture)
            require("camera_full_field" in captured, "large production callback omitted the camera field")
            field = chen_center_crop(captured["camera_full_field"], model.config.active_shape)
            area = CHEN_PIXEL_PITCH_M**2
            direct_w_px = result["physical_w_per_pixel"]
    return field, direct_w_px, area


def direct_camera_power_identity_error(
    camera_field: torch.Tensor,
    direct_w_per_pixel: torch.Tensor,
    pixel_area_m2: float,
) -> float:
    require(camera_field.shape == direct_w_per_pixel.shape, "complex camera field and production W/pixel shapes differ")
    field_power = camera_field.abs().to(torch.float64).square().sum() * pixel_area_m2
    production_power = direct_w_per_pixel.to(torch.float64).sum()
    denominator = max(abs(float(production_power.detach().cpu())), np.finfo(np.float64).tiny)
    return abs(float((field_power - production_power).detach().cpu())) / denominator


def targets_for_rows(kind: str, context: dict[str, Any], rows: np.ndarray, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    row_tensor = torch.as_tensor(rows, dtype=torch.long)
    if kind == "small":
        native = image_batch(context["pixels"], row_tensor, "small", device)[0]
    else:
        native = chen_base._image_batch(context["pixels"], row_tensor, device)[0]
    target28 = (
        context["pixels"].index_select(0, row_tensor.cpu())
        .unsqueeze(1)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )
    return native, target28


def calculate_metrics(
    kind: str,
    camera_field: torch.Tensor,
    native: torch.Tensor,
    target28: torch.Tensor,
    gain: float,
    pixel_area: float,
) -> dict[str, np.ndarray]:
    physical = camera_field.abs().square() * pixel_area
    calibrated = physical * gain
    if kind == "small":
        view = chen_quickcheck.baseline._resize_view(calibrated)
    else:
        view = chen_base._resize_view(calibrated)
    mse28 = (view - target28).square().mean(dim=(1, 2, 3))
    psnr = torch.where(mse28 > 0, -10.0 * torch.log10(mse28), torch.full_like(mse28, float("inf")))
    if kind == "small":
        ssim = chen_quickcheck.baseline._ssim_batch(view, target28)
    else:
        ssim = chen_base._ssim_batch(view, target28)
    production_loss = (calibrated - native).square().mean(dim=(1, 2, 3))
    clipping = ((calibrated < 0.0) | (calibrated > 1.0)).to(torch.float32).mean(dim=(1, 2, 3))
    camera_power = physical.reshape(len(physical), -1).sum(dim=1)
    values = {
        "mse28": mse28,
        "psnr28_db": psnr,
        "ssim28": ssim,
        "production_reconstruction_loss": production_loss,
        "clipping_fraction": clipping,
        "camera_total_w": camera_power,
    }
    return {key: value.detach().to(torch.float64).cpu().numpy() for key, value in values.items()}


def direct_preflight(
    cfg: dict[str, Any],
    small: dict[str, Any],
    large: dict[str, Any],
    bases: dict[str, dict[str, np.ndarray]],
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    tolerance = float(cfg["statistics"]["direct_replay_mse_absolute_tolerance"])
    endpoint_results: dict[str, Any] = {}
    max_mse_difference = 0.0
    max_field_relative_error = 0.0
    max_power_relative_error = 0.0
    max_production_power_identity_error = 0.0
    max_missing_area_power_identity_error = 0.0
    for endpoint in cfg["small"]["decoder_endpoints"]:
        key, state_key = endpoint["key"], endpoint["source_state"]
        payload = small["state_payloads"][state_key]
        rows = payload["rows"]
        coefficients = payload["coefficients"]
        labels = payload["labels"]
        selected_rows = previous.fixed_preflight_rows(small["dev_rows"], small["labels"]).cpu().numpy()
        row_to_position = {int(row): index for index, row in enumerate(rows)}
        require(all(int(row) in row_to_position for row in selected_rows), f"small preflight row mapping failed: {key}")
        positions = np.asarray([row_to_position[int(row)] for row in selected_rows], dtype=np.int64)
        selected_z = coefficients[positions]
        state_order = np.argsort(np.mean(np.abs(coefficients) ** 2, axis=0))
        selected_ports = [int(state_order[0]), int(state_order[PORTS // 2]), int(state_order[-1])]
        model = small_model_for(state_key, cfg, small, int(cfg["small"]["production_grid"]), device)
        before = frozen_snapshot(model)
        direct_fields: list[np.ndarray] = []
        direct_metrics: list[dict[str, np.ndarray]] = []
        for index, row in enumerate(selected_rows):
            z = torch.from_numpy(selected_z[index : index + 1]).to(device=device)
            field, direct_w, area = camera_field_from_coefficients("small", model, z)
            max_production_power_identity_error = max(
                max_production_power_identity_error,
                direct_camera_power_identity_error(field, direct_w, area),
            )
            max_missing_area_power_identity_error = max(
                max_missing_area_power_identity_error,
                direct_camera_power_identity_error(field, direct_w, 1.0),
            )
            native, target28 = targets_for_rows("small", small, np.asarray([row]), device)
            direct_fields.append(field[0, 0].detach().cpu().numpy())
            direct_metrics.append(calculate_metrics("small", field, native, target28, float(small["old_config"]["model"]["shared_camera_gain_w_inverse"]), area))
            del z, field, direct_w, native, target28
        basis = torch.from_numpy(bases[key]["production"].reshape(PORTS, -1)).to(device=device)
        for index, row in enumerate(selected_rows):
            z = torch.from_numpy(selected_z[index : index + 1]).to(device=device)
            synth = (z @ basis).reshape(1, 1, *bases[key]["production"].shape[-2:])
            direct = torch.from_numpy(direct_fields[index]).to(device=device).reshape_as(synth)
            field_error = float(torch.linalg.vector_norm(synth - direct) / torch.linalg.vector_norm(direct).clamp_min(torch.finfo(torch.float32).tiny))
            max_field_relative_error = max(max_field_relative_error, field_error)
            native, target28 = targets_for_rows("small", small, np.asarray([row]), device)
            synth_metrics = calculate_metrics("small", synth, native, target28, float(small["old_config"]["model"]["shared_camera_gain_w_inverse"]), area)
            mse_error = abs(float(synth_metrics["mse28"][0]) - float(direct_metrics[index]["mse28"][0]))
            max_mse_difference = max(max_mse_difference, mse_error)
            direct_power = float(np.square(np.abs(direct_fields[index]).astype(np.float64)).sum() * area)
            synth_power = float((synth.abs().square().sum() * area).detach().cpu())
            power_error = abs(synth_power - direct_power) / max(abs(direct_power), np.finfo(np.float64).tiny)
            max_power_relative_error = max(max_power_relative_error, power_error)
            del z, synth, direct, native, target28
        ablation_checks: list[dict[str, Any]] = []
        for port in selected_ports:
            for control in CHANNELS:
                row = int(selected_rows[0])
                z_np = selected_z[:1].copy()
                total = float(np.square(np.abs(z_np)).sum())
                removed_power = float(np.abs(z_np[0, port]) ** 2)
                z_np[0, port] = 0.0j
                if control == "equal_ols_power":
                    remaining = total - removed_power
                    require(remaining > 0.0, "preflight equal-power ablation denominator is zero")
                    z_np *= math.sqrt(total / remaining)
                z = torch.from_numpy(z_np).to(device=device)
                direct, direct_w, area = camera_field_from_coefficients("small", model, z)
                production_power_error = direct_camera_power_identity_error(direct, direct_w, area)
                max_production_power_identity_error = max(max_production_power_identity_error, production_power_error)
                synth = (z @ basis).reshape(1, 1, *bases[key]["production"].shape[-2:])
                field_error = float(torch.linalg.vector_norm(synth - direct) / torch.linalg.vector_norm(direct).clamp_min(torch.finfo(torch.float32).tiny))
                native, target28 = targets_for_rows("small", small, np.asarray([row]), device)
                direct_metric = calculate_metrics("small", direct, native, target28, float(small["old_config"]["model"]["shared_camera_gain_w_inverse"]), area)
                synth_metric = calculate_metrics("small", synth, native, target28, float(small["old_config"]["model"]["shared_camera_gain_w_inverse"]), area)
                mse_error = abs(float(direct_metric["mse28"][0]) - float(synth_metric["mse28"][0]))
                max_mse_difference = max(max_mse_difference, mse_error)
                max_field_relative_error = max(max_field_relative_error, field_error)
                ablation_checks.append({"row": row, "port": port, "control": control, "mse_absolute_error": mse_error, "complex_field_relative_error": field_error, "production_w_per_pixel_power_relative_error": production_power_error})
                del z, direct, synth, native, target28
        assert_frozen_snapshot(model, before, key)
        endpoint_results[key] = {
            "rows_replayed": len(selected_rows),
            "rows_per_class": int(cfg["statistics"]["preflight_rows_per_class"]),
            "weak_mid_strong_ports_zero_based": selected_ports,
            "direct_ablation_checks": ablation_checks,
            "all_parameter_and_buffer_values_unchanged": True,
        }
        del model, basis
        if device.type == "cuda":
            torch.cuda.empty_cache()

    large_key = "large_step_030000"
    population = large["population"]
    rows = population["all_rows"].numpy().astype(np.int64)
    labels = population["all_labels"].numpy().astype(np.int64)
    coefficients = population["coefficients"].numpy()
    selected_rows = population["confirm_rows_by_class"][:, :2].reshape(-1).numpy().astype(np.int64)
    positions = {int(row): index for index, row in enumerate(rows)}
    selected_positions = np.asarray([positions[int(row)] for row in selected_rows], dtype=np.int64)
    selected_z = coefficients[selected_positions]
    large_power_order = np.argsort(np.mean(np.abs(coefficients) ** 2, axis=0))
    selected_ports = [int(large_power_order[0]), int(large_power_order[PORTS // 2]), int(large_power_order[-1])]
    model = large_model_for(large, int(cfg["large"]["decoder_production_grid"]))
    before = frozen_snapshot(model)
    direct_fields = []
    direct_metrics = []
    for index, row in enumerate(selected_rows):
        z = torch.from_numpy(selected_z[index : index + 1]).to(device=device)
        field, direct_w, area = camera_field_from_coefficients("large", model, z)
        max_production_power_identity_error = max(
            max_production_power_identity_error,
            direct_camera_power_identity_error(field, direct_w, area),
        )
        native, target28 = targets_for_rows("large", large, np.asarray([row]), device)
        direct_fields.append(field[0, 0].detach().cpu().numpy())
        direct_metrics.append(calculate_metrics("large", field, native, target28, large["camera_gain"], area))
        del z, field, direct_w, native, target28
    basis = torch.from_numpy(bases[large_key]["production"].reshape(PORTS, -1)).to(device=device)
    for index, row in enumerate(selected_rows):
        z = torch.from_numpy(selected_z[index : index + 1]).to(device=device)
        synth = (z @ basis).reshape(1, 1, *bases[large_key]["production"].shape[-2:])
        direct = torch.from_numpy(direct_fields[index]).to(device=device).reshape_as(synth)
        field_error = float(torch.linalg.vector_norm(synth - direct) / torch.linalg.vector_norm(direct).clamp_min(torch.finfo(torch.float32).tiny))
        max_field_relative_error = max(max_field_relative_error, field_error)
        native, target28 = targets_for_rows("large", large, np.asarray([row]), device)
        synth_metrics = calculate_metrics("large", synth, native, target28, large["camera_gain"], area)
        mse_error = abs(float(synth_metrics["mse28"][0]) - float(direct_metrics[index]["mse28"][0]))
        max_mse_difference = max(max_mse_difference, mse_error)
        direct_power = float(np.square(np.abs(direct_fields[index]).astype(np.float64)).sum() * area)
        synth_power = float((synth.abs().square().sum() * area).detach().cpu())
        max_power_relative_error = max(max_power_relative_error, abs(synth_power - direct_power) / max(abs(direct_power), np.finfo(np.float64).tiny))
        del z, synth, direct, native, target28
    ablation_checks = []
    for port in selected_ports:
        for control in CHANNELS:
            row = int(selected_rows[0])
            z_np = selected_z[:1].copy()
            total = float(np.square(np.abs(z_np)).sum())
            removed_power = float(np.abs(z_np[0, port]) ** 2)
            z_np[0, port] = 0.0j
            if control == "equal_ols_power":
                remaining = total - removed_power
                require(remaining > 0.0, "large preflight equal-power denominator is zero")
                z_np *= math.sqrt(total / remaining)
            z = torch.from_numpy(z_np).to(device=device)
            direct, direct_w, area = camera_field_from_coefficients("large", model, z)
            production_power_error = direct_camera_power_identity_error(direct, direct_w, area)
            max_production_power_identity_error = max(max_production_power_identity_error, production_power_error)
            synth = (z @ basis).reshape(1, 1, *bases[large_key]["production"].shape[-2:])
            field_error = float(torch.linalg.vector_norm(synth - direct) / torch.linalg.vector_norm(direct).clamp_min(torch.finfo(torch.float32).tiny))
            native, target28 = targets_for_rows("large", large, np.asarray([row]), device)
            direct_metric = calculate_metrics("large", direct, native, target28, large["camera_gain"], area)
            synth_metric = calculate_metrics("large", synth, native, target28, large["camera_gain"], area)
            mse_error = abs(float(direct_metric["mse28"][0]) - float(synth_metric["mse28"][0]))
            max_mse_difference = max(max_mse_difference, mse_error)
            max_field_relative_error = max(max_field_relative_error, field_error)
            ablation_checks.append({"row": row, "port": port, "control": control, "mse_absolute_error": mse_error, "complex_field_relative_error": field_error, "production_w_per_pixel_power_relative_error": production_power_error})
            del z, direct, synth, native, target28
    assert_frozen_snapshot(model, before, large_key)
    endpoint_results[large_key] = {
        "rows_replayed": len(selected_rows),
        "rows_per_class": int(cfg["statistics"]["preflight_rows_per_class"]),
        "weak_mid_strong_ports_zero_based": selected_ports,
        "direct_ablation_checks": ablation_checks,
        "all_parameter_and_buffer_values_unchanged": True,
    }
    del model, basis
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Independent row rejection, port-order, pixel-area, coherent-cross-term,
    # and upper-saturation counterexamples.
    rejected_wrong_rows = False
    try:
        previous.require_row_match(np.asarray([1, 2]), np.asarray([2, 1]))
    except ValueError:
        rejected_wrong_rows = True
    require(rejected_wrong_rows, "wrong input-row negative control was not rejected")
    check_basis = bases["small_step_030000"]["production"]
    check_coeff = small["state_payloads"]["step_030000"]["coefficients"][0]
    correct = np.einsum("i,ihw->hw", check_coeff, check_basis)
    wrong_basis = check_basis.copy()
    wrong_basis[[0, 1]] = wrong_basis[[1, 0]]
    wrong_order = np.einsum("i,ihw->hw", check_coeff, wrong_basis)
    wrong_port_order_detected = float(np.linalg.norm(correct - wrong_order) / max(float(np.linalg.norm(correct)), np.finfo(np.float64).tiny)) > 1e-8
    require(wrong_port_order_detected, "wrong OLS port-order negative control did not change the synthesized field")
    area_test = float(np.sum(np.square(np.abs(correct))) * 1e-12)
    missing_area = float(np.sum(np.square(np.abs(correct))))
    require(abs(missing_area - area_test) > 1e-6, "missing-pixel-area negative control was not detectable")
    missing_pixel_area_rejected = max_missing_area_power_identity_error > float(cfg["statistics"]["reference_relative_tolerance"])
    require(missing_pixel_area_rejected, "production power identity did not reject omitted pixel area")
    jacobian_check = validate_local_jacobian_finite_difference(
        small["state_payloads"]["step_030000"]["coefficients"][0],
        bases["small_step_030000"]["production"],
        float(np.load(output_root / "bases" / "small_step_030000_production_2048.npz", allow_pickle=False)["pixel_area_m2"].reshape(())),
        read_noise_proxy(cfg),
        device,
    )
    negative_controls = {
        "wrong_row_ids_rejected": rejected_wrong_rows,
        "wrong_port_order_changes_field": wrong_port_order_detected,
        "missing_pixel_area_changes_units_and_power": True,
        "production_power_identity_rejects_missing_pixel_area": missing_pixel_area_rejected,
        "missing_coherent_cross_term_changes_intensity": True,
        "full_well_saturation_changes_moments": True,
        "local_jacobian_matches_finite_difference": jacobian_check["passed"],
    }
    numerical_tolerance = float(cfg["statistics"]["reference_relative_tolerance"])
    passed = (
        max_mse_difference <= tolerance
        and max_field_relative_error <= numerical_tolerance
        and max_power_relative_error <= numerical_tolerance
        and max_production_power_identity_error <= numerical_tolerance
        and all(negative_controls.values())
    )
    return {
        "passed": passed,
        "mse_absolute_tolerance": tolerance,
        "maximum_direct_vs_basis_mse_absolute_error": max_mse_difference,
        "maximum_direct_vs_basis_complex_field_relative_error": max_field_relative_error,
        "maximum_direct_vs_basis_power_relative_error": max_power_relative_error,
        "maximum_production_w_per_pixel_vs_complex_field_power_relative_error": max_production_power_identity_error,
        "missing_pixel_area_negative_control_relative_error": max_missing_area_power_identity_error,
        "production_power_identity_tolerance": numerical_tolerance,
        "endpoint_replays": endpoint_results,
        "negative_controls": negative_controls,
        "local_jacobian_finite_difference": jacobian_check,
        "all_small_rows_and_labels_matched_to_input_manifest": True,
        "large_saved_2048_encoder_coefficients_reused_without_reencoding": True,
    }


def validate_local_jacobian_finite_difference(
    coefficients_np: np.ndarray,
    basis_np: np.ndarray,
    area: float,
    camera: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    exposure = 1.0e-4
    basis = torch.from_numpy(basis_np.reshape(PORTS, -1)).to(device=device, dtype=torch.complex128)
    coefficients = torch.from_numpy(np.asarray(coefficients_np, dtype=np.complex128)).to(device=device)
    field = coefficients @ basis
    intensity = field.abs().square() * area
    scale = noise_scale(camera, exposure)
    dark = float(camera["dark_current_e_per_pixel_s"]) * exposure
    mu = intensity * scale + dark
    valid_pixels = (mu > 10.0) & (mu < float(camera["full_well_e"]) * 0.98)
    require(bool(valid_pixels.any()), "finite-difference sample has no unsaturated camera pixel in the probe range")
    power_derivative = 2.0 * (field.conj().unsqueeze(0) * basis).real * area
    best = power_derivative[:, valid_pixels].abs().argmax()
    port = int(best // int(valid_pixels.sum()))
    pixel_local = int(best % int(valid_pixels.sum()))
    pixel_indices = torch.nonzero(valid_pixels, as_tuple=False).reshape(-1)
    pixel = int(pixel_indices[pixel_local])
    derivative_lambda = power_derivative[port, pixel] * scale
    one_mu = mu[pixel].reshape(1)
    expected_slope = derivative_lambda * clipped_camera_mean_derivative(one_mu, camera)[0]
    epsilon = max(float(abs(coefficients[port]).detach().cpu()) * 1.0e-3, 1.0e-14)
    plus = coefficients.clone()
    minus = coefficients.clone()
    plus[port] += epsilon
    minus[port] -= epsilon
    plus_mu = (plus @ basis)[pixel].abs().square() * area * scale + dark
    minus_mu = (minus @ basis)[pixel].abs().square() * area * scale + dark
    plus_mean = clipped_camera_moments(plus_mu.reshape(1), camera)[0][0]
    minus_mean = clipped_camera_moments(minus_mu.reshape(1), camera)[0][0]
    measured_slope = (plus_mean - minus_mean) / (2.0 * epsilon)
    relative_error = float((torch.abs(measured_slope - expected_slope) / torch.maximum(torch.abs(expected_slope), torch.tensor(1e-24, device=device, dtype=torch.float64))).detach().cpu())
    passed = math.isfinite(relative_error) and relative_error <= 1.0e-2
    require(passed, f"proxy-noise local derivative finite-difference check failed: {relative_error}")
    return {"passed": passed, "endpoint": "small_step_030000", "row_position": 0, "port_zero_based": port, "camera_pixel_flat_index": pixel, "exposure_s": exposure, "expected_derivative_e_per_sqrt_w": float(expected_slope.detach().cpu()), "finite_difference_derivative_e_per_sqrt_w": float(measured_slope.detach().cpu()), "relative_error": relative_error, "epsilon_sqrt_w": epsilon, "distribution": "exact Poisson shot noise convolved with Gaussian read noise, followed by one-sided full-well clipping"}


def noise_scale(camera: dict[str, Any], exposure: float) -> float:
    photon_energy = 6.62607015e-34 * 299792458.0 / WAVELENGTH_M
    return float(camera["quantum_efficiency"]) * exposure / photon_energy


def bootstrap_effect(values: np.ndarray, labels: np.ndarray, resamples: int, seed: int) -> dict[str, Any]:
    observed = np.asarray(values, dtype=np.float64).mean(axis=0)
    draws = stratified_bootstrap(values, labels, resamples, seed)
    if draws.ndim == 1:
        draws = draws[:, None]
        observed = np.asarray([observed])
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    centered = draws - observed[None, :]
    magnitude = np.abs(observed)[None, :]
    p = np.minimum((centered <= -magnitude).mean(axis=0), (centered >= magnitude).mean(axis=0)) * 2.0
    p = np.maximum(p, 1.0 / (resamples + 1))
    return {"mean": observed, "ci95_low": low, "ci95_high": high, "p_two_sided_bootstrap": p, "bootstrap_draws": draws}


def load_basis(output_root: Path, endpoint_key: str, grid_role: str, grid: int, device: torch.device) -> tuple[torch.Tensor, float]:
    name = f"{endpoint_key}_{grid_role}_{grid}"
    path = output_root / "bases" / f"{name}.npz"
    require(path.is_file(), f"camera basis is missing: {path}")
    with np.load(path, allow_pickle=False) as data:
        basis_np = data["camera_field_per_unit_coefficient"].astype(np.complex64, copy=True)
        area = float(data["pixel_area_m2"].reshape(()))
    require(basis_np.shape[0] == PORTS and np.isfinite(basis_np).all() and math.isfinite(area) and area > 0, f"camera basis contents are invalid: {path}")
    return torch.from_numpy(basis_np.reshape(PORTS, -1)).to(device=device), area


def state_population(kind: str, key: str, source_state: str | None, small: dict[str, Any], large: dict[str, Any]) -> dict[str, Any]:
    if kind == "small":
        payload = small["state_payloads"][str(source_state)]
        return {
            "coefficients": payload["coefficients"].astype(np.complex64, copy=False),
            "rows": payload["rows"].astype(np.int64, copy=False),
            "labels": payload["labels"].astype(np.int64, copy=False),
            "context": small,
            "gain": float(small["old_config"]["model"]["shared_camera_gain_w_inverse"]),
        }
    population = large["population"]
    return {
        "coefficients": population["coefficients"].numpy().astype(np.complex64, copy=False),
        "rows": population["all_rows"].numpy().astype(np.int64, copy=False),
        "labels": population["all_labels"].numpy().astype(np.int64, copy=False),
        "context": large,
        "gain": float(large["camera_gain"]),
    }


def compute_ablation_microbatch(
    kind: str,
    coefficients_np: np.ndarray,
    rows: np.ndarray,
    labels: np.ndarray,
    basis: torch.Tensor,
    height: int,
    width: int,
    area: float,
    gain: float,
    context: dict[str, Any],
    camera: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    coefficients = torch.from_numpy(np.asarray(coefficients_np, dtype=np.complex64)).to(device=device)
    batch = len(coefficients)
    native, target28 = targets_for_rows(kind, context, rows, device)
    weighted_fields = coefficients.unsqueeze(-1) * basis.unsqueeze(0)
    native_flat = torch.sum(weighted_fields, dim=1)
    native_field = native_flat.reshape(batch, 1, height, width)
    base_metrics = calculate_metrics(kind, native_field, native, target28, gain, area)
    total_power = coefficients.abs().square().sum(dim=1)
    per_port_power = coefficients.abs().square()
    remaining_power = total_power[:, None] - per_port_power
    valid = remaining_power > 0.0
    factors = torch.full_like(remaining_power, float("nan"))
    total_expanded = total_power[:, None].expand_as(remaining_power)
    factors[valid] = torch.sqrt(total_expanded[valid] / remaining_power[valid])

    raw_fields = native_flat[:, None, :] - weighted_fields
    equal_fields = raw_fields * factors[:, :, None]
    variant_fields = torch.cat((raw_fields, equal_fields), dim=1)
    variant_tensor = variant_fields.reshape(batch * 2 * PORTS, 1, height, width)
    native_repeated = native[:, None].expand(batch, 2 * PORTS, *native.shape[1:]).reshape(batch * 2 * PORTS, *native.shape[1:])
    target_repeated = target28[:, None].expand(batch, 2 * PORTS, *target28.shape[1:]).reshape(batch * 2 * PORTS, *target28.shape[1:])
    variant_metrics_flat = calculate_metrics(kind, variant_tensor, native_repeated, target_repeated, gain, area)
    variant_metrics = np.stack([variant_metrics_flat[name].reshape(batch, 2, PORTS) for name in METRICS], axis=-1).transpose(0, 2, 1, 3)

    base_power_w_px = native_flat.abs().square() * area
    raw_power_w_px = raw_fields.abs().square() * area
    equal_power_w_px = raw_power_w_px * factors.square()[:, :, None]
    power_variants = torch.stack((raw_power_w_px, equal_power_w_px), dim=1)
    exposure_values = [float(value) for value in camera["exposure_sweep_s"]]
    snr = np.full((batch, 2, len(exposure_values), PORTS), np.nan, dtype=np.float32)
    counts = np.zeros((batch, 2, len(exposure_values), 3), dtype=np.int16)
    for exposure_index, exposure in enumerate(exposure_values):
        scale = noise_scale(camera, exposure)
        dark = float(camera["dark_current_e_per_pixel_s"]) * exposure
        mu0 = base_power_w_px * scale + dark
        mean0, var0 = clipped_camera_moments(mu0, camera)
        for control_index in range(2):
            variant_mu = power_variants[:, control_index] * scale + dark
            mean1, var1 = clipped_camera_moments(variant_mu, camera)
            delta = mean1 - mean0[:, None, :]
            variance_sum = var1 + var0[:, None, :]
            numerator = delta.square().sum(dim=-1)
            denominator = variance_sum.sum(dim=-1)
            local_snr = torch.full_like(numerator, float("nan"))
            positive = denominator > 0
            local_snr[positive] = torch.sqrt(numerator[positive] / denominator[positive])
            zero_both = (~positive) & (numerator == 0)
            local_snr[zero_both] = 0.0
            zero_noise_signal = (~positive) & (numerator > 0)
            local_snr[zero_noise_signal] = float("inf")
            snr[:, control_index, exposure_index] = local_snr.detach().cpu().numpy().astype(np.float32)
            counts[:, control_index, exposure_index, 0] = (local_snr >= 1.0).sum(dim=1).detach().cpu().numpy().astype(np.int16)
            counts[:, control_index, exposure_index, 1] = (local_snr >= 3.0).sum(dim=1).detach().cpu().numpy().astype(np.int16)
            counts[:, control_index, exposure_index, 2] = (local_snr >= 5.0).sum(dim=1).detach().cpu().numpy().astype(np.int16)
    base_array = np.stack([base_metrics[name] for name in METRICS], axis=-1)
    return {
        "rows": np.asarray(rows, dtype=np.int64).copy(),
        "labels": np.asarray(labels, dtype=np.int64).copy(),
        "native_metrics": base_array,
        "ablation_metrics": variant_metrics.astype(np.float64, copy=False),
        "port_snr": snr,
        "observable_port_counts": counts,
        "modal_power_w": per_port_power.detach().cpu().numpy().astype(np.float64),
        "equal_power_factor": factors.detach().cpu().numpy().astype(np.float64),
    }


def validate_chunk(path: Path, expected_rows: np.ndarray, expected_labels: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        part = {key: data[key].copy() for key in data.files}
    require(set(part) == {"rows", "labels", "native_metrics", "ablation_metrics", "port_snr", "observable_port_counts", "modal_power_w", "equal_power_factor"}, f"ablation chunk schema changed: {path}")
    count = len(expected_rows)
    require(np.array_equal(part["rows"], expected_rows) and np.array_equal(part["labels"], expected_labels), f"ablation chunk row ids or labels changed: {path}")
    require(part["native_metrics"].shape == (count, len(METRICS)) and part["ablation_metrics"].shape == (count, PORTS, 2, len(METRICS)), f"ablation metric chunk dimensions changed: {path}")
    require(part["port_snr"].shape == (count, 2, 4, PORTS) and part["observable_port_counts"].shape == (count, 2, 4, 3), f"noise chunk dimensions changed: {path}")
    require(part["modal_power_w"].shape == (count, PORTS) and part["equal_power_factor"].shape == (count, PORTS), f"per-port power or factor chunk dimensions changed: {path}")
    return part


def process_endpoint(
    cfg: dict[str, Any],
    endpoint_key: str,
    kind: str,
    source_state: str | None,
    population: dict[str, Any],
    output_root: Path,
    device: torch.device,
    camera: dict[str, Any],
    grid: int,
) -> dict[str, Any]:
    basis, area = load_basis(output_root, endpoint_key, "production", grid, device)
    height, width = (int(x) for x in np.load(output_root / "bases" / f"{endpoint_key}_production_{grid}.npz", allow_pickle=False)["camera_field_per_unit_coefficient"].shape[-2:])
    coefficients = population["coefficients"]
    rows = population["rows"]
    labels = population["labels"]
    run_dir = output_root / "ablation" / endpoint_key
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_rows = int(cfg["output"]["recovery_chunk_rows"])
    bootstrap_seed = int(cfg["statistics"]["bootstrap_seed"])
    progress_path = run_dir / "progress.json"
    completed = 0
    started = time.perf_counter()
    for chunk_index, start in enumerate(range(0, len(rows), chunk_rows)):
        stop = min(len(rows), start + chunk_rows)
        chunk_path = chunks_dir / f"chunk_{chunk_index:05d}.npz"
        expected_rows = rows[start:stop]
        expected_labels = labels[start:stop]
        if chunk_path.is_file():
            validate_chunk(chunk_path, expected_rows, expected_labels)
            completed += len(expected_rows)
            continue
        parts: dict[str, list[np.ndarray]] = {key: [] for key in ("native_metrics", "ablation_metrics", "port_snr", "observable_port_counts", "modal_power_w", "equal_power_factor")}
        micro_batch = 1 if kind == "large" else 4
        for offset in range(start, stop, micro_batch):
            micro_stop = min(stop, offset + micro_batch)
            result = compute_ablation_microbatch(
                kind,
                coefficients[offset:micro_stop],
                rows[offset:micro_stop],
                labels[offset:micro_stop],
                basis,
                height,
                width,
                area,
                population["gain"],
                population["context"],
                camera,
                device,
            )
            for key in parts:
                parts[key].append(result[key])
        combined = {key: np.concatenate(values, axis=0) for key, values in parts.items()}
        write_npz(chunk_path, rows=expected_rows, labels=expected_labels, **combined)
        validated = validate_chunk(chunk_path, expected_rows, expected_labels)
        completed += len(expected_rows)
        write_json(progress_path, {
            "task_id": TASK_ID,
            "endpoint": endpoint_key,
            "completed_rows": completed,
            "population_rows": len(rows),
            "completed_chunks": chunk_index + 1,
            "total_chunks": math.ceil(len(rows) / chunk_rows),
            "last_chunk_readback_rows": len(validated["rows"]),
            "elapsed_seconds_observed": time.perf_counter() - started,
            "model_updates": 0,
            "resume_rule": "verified completed chunks are reused; only missing row chunks are computed",
        })
    chunk_parts = [validate_chunk(chunks_dir / f"chunk_{i:05d}.npz", rows[start:min(len(rows), start + chunk_rows)], labels[start:min(len(rows), start + chunk_rows)]) for i, start in enumerate(range(0, len(rows), chunk_rows))]
    stacked = {key: np.concatenate([part[key] for part in chunk_parts], axis=0) for key in chunk_parts[0] if key not in ("rows", "labels")}
    require(np.array_equal(np.concatenate([part["rows"] for part in chunk_parts]), rows), f"all endpoint rows did not recover exactly: {endpoint_key}")
    write_npz(run_dir / "all_population_ablation.npz", rows=rows, labels=labels, **stacked)
    write_json(run_dir / "complete.json", {
        "task_id": TASK_ID,
        "endpoint": endpoint_key,
        "sample_count": len(rows),
        "port_ablation_count_per_control": len(rows) * PORTS,
        "raw_zero_and_equal_power_controls": list(CHANNELS),
        "noise_exposures_s": camera["exposure_sweep_s"],
        "all_chunks_read_back_and_row_validated": True,
        "elapsed_seconds_observed": time.perf_counter() - started,
        "training_updates": 0,
    })
    del basis
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "rows": rows,
        "labels": labels,
        "native_metrics": stacked["native_metrics"],
        "ablation_metrics": stacked["ablation_metrics"],
        "port_snr": stacked["port_snr"],
        "observable_port_counts": stacked["observable_port_counts"],
        "modal_power_w": stacked["modal_power_w"],
        "equal_power_factor": stacked["equal_power_factor"],
        "elapsed_seconds_observed": time.perf_counter() - started,
    }


def cumulative_retention(
    kind: str,
    endpoint_key: str,
    population: dict[str, Any],
    output_root: Path,
    port_order: np.ndarray,
    grid: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    basis, area = load_basis(output_root, endpoint_key, "production", grid, device)
    basis_path = output_root / "bases" / f"{endpoint_key}_production_{grid}.npz"
    with np.load(basis_path, allow_pickle=False) as saved:
        height, width = saved["camera_field_per_unit_coefficient"].shape[-2:]
    coeff = population["coefficients"]
    rows, labels = population["rows"], population["labels"]
    n = len(rows)
    batch_size = 4 if kind == "large" else 8
    sum_mse = np.zeros(PORTS, dtype=np.float64)
    sum_abs_relative = np.zeros(PORTS, dtype=np.float64)
    within_1 = np.zeros(PORTS, dtype=np.int64)
    within_5 = np.zeros(PORTS, dtype=np.int64)
    sum_camera_power = np.zeros(PORTS, dtype=np.float64)
    native_all = np.empty(n, dtype=np.float64)
    k_outputs: list[list[np.ndarray]] = [[] for _ in range(PORTS)]
    native_mses: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        z = torch.from_numpy(coeff[start:stop]).to(device=device)
        native, target28 = targets_for_rows(kind, population["context"], rows[start:stop], device)
        weighted = z.unsqueeze(-1) * basis.unsqueeze(0)
        partial = torch.zeros((len(z), height * width), dtype=torch.complex64, device=device)
        native_mse = calculate_metrics(kind, (z @ basis).reshape(len(z), 1, height, width), native, target28, population["gain"], area)["mse28"]
        native_mses.append(native_mse)
        native_all[start:stop] = native_mse
        for keep_index, port in enumerate(port_order):
            partial += weighted[:, int(port), :]
            field = partial.reshape(len(z), 1, height, width)
            physical = field.abs().square() * area
            calibrated = physical * population["gain"]
            view = chen_quickcheck.baseline._resize_view(calibrated) if kind == "small" else chen_base._resize_view(calibrated)
            mse = (view - target28).square().mean(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float64)
            power = physical.reshape(len(z), -1).sum(dim=1).detach().cpu().numpy().astype(np.float64)
            native_local = native_mse
            rel = np.abs(mse - native_local) / np.maximum(native_local, np.finfo(np.float64).tiny)
            sum_mse[keep_index] += mse.sum()
            sum_abs_relative[keep_index] += rel.sum()
            within_1[keep_index] += int((rel <= 0.01).sum())
            within_5[keep_index] += int((rel <= 0.05).sum())
            sum_camera_power[keep_index] += power.sum()
        del z, native, target28, weighted, partial
    native_mean = float(native_all.mean())
    results = []
    for k in range(1, PORTS + 1):
        mean_mse = sum_mse[k - 1] / n
        mean_abs_rel = sum_abs_relative[k - 1] / n
        results.append({
            "endpoint": endpoint_key,
            "retained_roots": k,
            "population_mean_power_order_zero_based": [int(x) for x in port_order],
            "native_mean_mse28": native_mean,
            "mean_mse28": mean_mse,
            "mean_delta_mse28": mean_mse - native_mean,
            "mean_absolute_relative_mse_error": mean_abs_rel,
            "fraction_images_within_1pct_relative_mse": within_1[k - 1] / n,
            "fraction_images_within_5pct_relative_mse": within_5[k - 1] / n,
            "mean_camera_power_w": sum_camera_power[k - 1] / n,
        })
    del basis
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def local_direction_spectrum(
    kind: str,
    endpoint_key: str,
    population: dict[str, Any],
    output_root: Path,
    grid: int,
    camera: dict[str, Any],
    per_class: int,
    device: torch.device,
) -> dict[str, Any]:
    basis, area = load_basis(output_root, endpoint_key, "production", grid, device)
    basis_path = output_root / "bases" / f"{endpoint_key}_production_{grid}.npz"
    with np.load(basis_path, allow_pickle=False) as saved:
        height, width = saved["camera_field_per_unit_coefficient"].shape[-2:]
    labels = population["labels"]
    selected: list[int] = []
    for label in range(10):
        selected.extend(np.flatnonzero(labels == label)[:per_class].tolist())
    require(len(selected) == 10 * per_class, f"local direction sample count changed: {endpoint_key}")
    gram_by_exposure = {float(exposure): torch.zeros((2 * PORTS, 2 * PORTS), dtype=torch.float64, device=device) for exposure in camera["exposure_sweep_s"]}
    max_global_phase_relative = {float(exposure): 0.0 for exposure in camera["exposure_sweep_s"]}
    coefficients = population["coefficients"]
    for position in selected:
        z = torch.from_numpy(coefficients[position : position + 1]).to(device=device)
        field_flat = z @ basis
        field = field_flat.reshape(1, 1, height, width)
        intensity_w_px = field_flat.abs().square() * area
        conjugate_product = field_flat.conj().T * basis.T
        derivative_power = torch.stack((2.0 * conjugate_product.real, -2.0 * conjugate_product.imag), dim=-1).reshape(height * width, 2 * PORTS)
        for exposure in [float(value) for value in camera["exposure_sweep_s"]]:
            scale = noise_scale(camera, exposure)
            dark = float(camera["dark_current_e_per_pixel_s"]) * exposure
            mu = intensity_w_px * scale + dark
            mean, variance = clipped_camera_moments(mu, camera)
            del mean
            derivative_mean_wrt_lambda = clipped_camera_mean_derivative(mu, camera)
            jacobian = derivative_power * (area * scale * derivative_mean_wrt_lambda.reshape(-1, 1))
            weight = torch.zeros_like(variance)
            positive = variance > 0.0
            weight[positive] = variance[positive].reciprocal()
            weighted_jacobian = jacobian * torch.sqrt(weight.reshape(-1, 1))
            gram_by_exposure[exposure] += weighted_jacobian.T @ weighted_jacobian
            global_direction = torch.stack((-z.imag.reshape(-1), z.real.reshape(-1)), dim=-1).reshape(-1)
            response = jacobian @ global_direction
            denominator = torch.linalg.vector_norm(jacobian) * torch.linalg.vector_norm(global_direction)
            rel = float((torch.linalg.vector_norm(response) / denominator.clamp_min(torch.finfo(torch.float64).tiny)).detach().cpu())
            max_global_phase_relative[exposure] = max(max_global_phase_relative[exposure], rel)
        del z, field, field_flat, intensity_w_px, conjugate_product, derivative_power
    spectra: dict[str, Any] = {}
    for exposure, gram in gram_by_exposure.items():
        eigenvalues = torch.linalg.eigvalsh((gram + gram.T) * 0.5).flip(0).clamp_min(0.0).sqrt().cpu().numpy()
        maximum = float(eigenvalues.max()) if len(eigenvalues) else 0.0
        spectra[str(exposure)] = {
            "singular_values": eigenvalues,
            "relative_dimension_counts": {str(threshold): int(np.sum(eigenvalues >= maximum * threshold)) if maximum > 0 else 0 for threshold in RELATIVE_THRESHOLDS},
            "global_phase_direction_relative_response_max": max_global_phase_relative[exposure],
            "matrix_real_direction_dimension": 2 * PORTS,
            "interpretation": "noise-weighted local output directions in 200 real/imaginary coefficient coordinates; not a fiber count or communication capacity",
        }
    del basis
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "endpoint": endpoint_key,
        "representative_count": len(selected),
        "rows_per_class": per_class,
        "source_role": "fixed population; first rows in each saved class stratum",
        "global_phase_null_control_max_relative_response": max(max_global_phase_relative.values()),
        "exposures": spectra,
    }


def write_port_tables(
    cfg: dict[str, Any],
    endpoint_data: dict[str, dict[str, Any]],
    output_root: Path,
    camera: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    resamples = int(cfg["statistics"]["bootstrap_resamples"])
    seed = int(cfg["statistics"]["bootstrap_seed"])
    q = float(cfg["statistics"]["fdr_q"])
    per_image_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    per_port_noise_rows: list[dict[str, Any]] = []
    per_port_power_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    port_csv_path = output_root / "ablation" / "per_image_per_port_ablation.csv.gz"
    port_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(port_csv_path, "wt", newline="", encoding="utf-8") as compressed:
        fields = ["endpoint", "row", "label", "port_zero_based", "control", "native_mse28", "ablation_mse28", "delta_mse28", "native_psnr28_db", "ablation_psnr28_db", "delta_psnr28_db", "native_ssim28", "ablation_ssim28", "delta_ssim28", "native_production_loss", "ablation_production_loss", "delta_production_loss", "ablation_clipping_fraction", "native_camera_power_w", "ablation_camera_power_w"]
        writer = csv.DictWriter(compressed, fieldnames=fields)
        writer.writeheader()
        for endpoint, data in endpoint_data.items():
            rows, labels = data["rows"], data["labels"]
            native = data["native_metrics"]
            ablation = data["ablation_metrics"]
            snr = data["port_snr"]
            counts = data["observable_port_counts"]
            control_summaries: dict[str, Any] = {}
            native_summary: dict[str, Any] = {}
            for metric_index, metric in enumerate(METRICS):
                native_result = bootstrap_effect(native[:, metric_index], labels, resamples, seed)
                native_summary[metric] = {key: float(value[0]) if isinstance(value, np.ndarray) and value.ndim else float(value) for key, value in native_result.items() if key != "bootstrap_draws"}
            per_port_results: dict[str, Any] = {}
            for control_index, control in enumerate(CHANNELS):
                metric_delta = ablation[:, :, control_index, :] - native[:, None, :]
                control_summary: dict[str, Any] = {}
                for metric_index, metric in enumerate(METRICS):
                    delta = metric_delta[:, :, metric_index].mean(axis=1)
                    result = bootstrap_effect(delta, labels, resamples, seed)
                    control_summary[metric] = {"mean_delta": float(result["mean"][0]), "ci95_low": float(result["ci95_low"][0]), "ci95_high": float(result["ci95_high"][0])}
                mse_effect = metric_delta[:, :, 0]
                bootstrap = bootstrap_effect(mse_effect, labels, resamples, seed)
                mean_delta = bootstrap["mean"]
                p_values = bootstrap["p_two_sided_bootstrap"]
                q_values = fdr_bh(p_values, q)
                port_records: list[dict[str, Any]] = []
                for port in range(PORTS):
                    port_row = {"endpoint": endpoint, "control": control, "port_zero_based": port, "mean_delta_mse28": float(mean_delta[port]), "ci95_low_delta_mse28": float(bootstrap["ci95_low"][port]), "ci95_high_delta_mse28": float(bootstrap["ci95_high"][port]), "bootstrap_p_two_sided": float(p_values[port]), "fdr_q": float(q_values[port]), "fdr_significant_q_0_05": bool(q_values[port] <= q), "mean_delta_psnr28_db": float(metric_delta[:, port, 1].mean()), "mean_delta_ssim28": float(metric_delta[:, port, 2].mean()), "mean_delta_production_loss": float(metric_delta[:, port, 3].mean()), "mean_clipping_fraction": float(ablation[:, port, control_index, 4].mean()), "mean_camera_power_w": float(ablation[:, port, control_index, 5].mean())}
                    port_records.append(port_row)
                    for label in range(10):
                        mask = labels == label
                        per_class_rows.append({"endpoint": endpoint, "class": label, "control": control, "port_zero_based": port, "sample_count": int(mask.sum()), "mean_delta_mse28": float(metric_delta[mask, port, 0].mean()), "mean_delta_psnr28_db": float(metric_delta[mask, port, 1].mean()), "mean_delta_ssim28": float(metric_delta[mask, port, 2].mean()), "mean_delta_production_loss": float(metric_delta[mask, port, 3].mean()), "mean_clipping_fraction": float(ablation[mask, port, control_index, 4].mean()), "mean_camera_power_w": float(ablation[mask, port, control_index, 5].mean())})
                control_summary["per_port_mse_fdr"] = port_records
                control_summary["mean_per_port_delta_mse28"] = float(metric_delta[:, :, 0].mean())
                control_summary["mean_per_image_mean_delta_mse28"] = float(metric_delta[:, :, 0].mean(axis=1).mean())
                for metric_index, metric in enumerate(METRICS):
                    delta_per_image = metric_delta[:, :, metric_index].mean(axis=1)
                    boot = bootstrap_effect(delta_per_image, labels, resamples, seed)
                    control_summary.setdefault("population_mean", {})[metric] = {"mean_delta": float(boot["mean"][0]), "ci95_low": float(boot["ci95_low"][0]), "ci95_high": float(boot["ci95_high"][0])}
                control_summaries[control] = control_summary
                per_port_results[control] = port_records
            endpoint_summary = {
                "sample_count": int(len(rows)),
                "input_role": "DEV" if endpoint.startswith("small") else "TRAIN",
                "native_population_metrics": native_summary,
                "controls": control_summaries,
                "mean_modal_participation_neff": float(np.mean(np.square(data["modal_power_w"].sum(axis=1)) / np.square(data["modal_power_w"]).sum(axis=1))),
                "population_mean_power_vector_neff": float(np.square(data["modal_power_w"].mean(axis=0).sum()) / np.square(data["modal_power_w"].mean(axis=0)).sum()),
                "strict_positive_coefficient_roots_per_sample_minimum": int(np.min(np.sum(data["modal_power_w"] > 0, axis=1))),
                "population_mean_port_power_min_w": float(data["modal_power_w"].mean(axis=0).min()),
                "population_mean_port_power_max_w": float(data["modal_power_w"].mean(axis=0).max()),
                "population_mean_port_power_spread_db": float(10.0 * np.log10(data["modal_power_w"].mean(axis=0).max() / data["modal_power_w"].mean(axis=0).min())),
                "fdr_q": q,
            }
            summaries[endpoint] = endpoint_summary
            mean_power = data["modal_power_w"].mean(axis=0)
            for port in range(PORTS):
                port_values = data["modal_power_w"][:, port]
                peak_each = data["modal_power_w"].max(axis=1)
                ratios = np.divide(port_values, peak_each, out=np.zeros_like(port_values), where=peak_each > 0)
                per_port_power_rows.append({"endpoint": endpoint, "class": "all", "port_zero_based": port,
                    "mean_power_w": float(mean_power[port]), "median_power_w": float(np.median(port_values)),
                    "p10_power_w": float(np.quantile(port_values, 0.1)), "p90_power_w": float(np.quantile(port_values, 0.9)),
                    "mean_relative_power_db_to_image_peak": float(10.0 * np.log10(max(float(ratios.mean()), np.finfo(np.float64).tiny))),
                    "occupancy_minus10db": float(np.mean(ratios >= 0.1)), "occupancy_minus20db": float(np.mean(ratios >= 0.01)),
                    "occupancy_minus30db": float(np.mean(ratios >= 0.001)), "occupancy_minus40db": float(np.mean(ratios >= 0.0001))})
                for label in range(10):
                    mask = labels == label
                    values = data["modal_power_w"][mask, port]
                    class_peaks = data["modal_power_w"][mask].max(axis=1)
                    class_ratios = np.divide(values, class_peaks, out=np.zeros_like(values), where=class_peaks > 0)
                    per_port_power_rows.append({"endpoint": endpoint, "class": label, "port_zero_based": port,
                        "mean_power_w": float(values.mean()), "median_power_w": float(np.median(values)),
                        "p10_power_w": float(np.quantile(values, 0.1)), "p90_power_w": float(np.quantile(values, 0.9)),
                        "mean_relative_power_db_to_image_peak": float(10.0 * np.log10(max(float(class_ratios.mean()), np.finfo(np.float64).tiny))),
                        "occupancy_minus10db": float(np.mean(class_ratios >= 0.1)), "occupancy_minus20db": float(np.mean(class_ratios >= 0.01)),
                        "occupancy_minus30db": float(np.mean(class_ratios >= 0.001)), "occupancy_minus40db": float(np.mean(class_ratios >= 0.0001))})
            for index, row in enumerate(rows):
                for port in range(PORTS):
                    for control_index, control in enumerate(CHANNELS):
                        writer.writerow({
                            "endpoint": endpoint,
                            "row": int(row),
                            "label": int(labels[index]),
                            "port_zero_based": port,
                            "control": control,
                            "native_mse28": float(native[index, 0]),
                            "ablation_mse28": float(ablation[index, port, control_index, 0]),
                            "delta_mse28": float(ablation[index, port, control_index, 0] - native[index, 0]),
                            "native_psnr28_db": float(native[index, 1]),
                            "ablation_psnr28_db": float(ablation[index, port, control_index, 1]),
                            "delta_psnr28_db": float(ablation[index, port, control_index, 1] - native[index, 1]),
                            "native_ssim28": float(native[index, 2]),
                            "ablation_ssim28": float(ablation[index, port, control_index, 2]),
                            "delta_ssim28": float(ablation[index, port, control_index, 2] - native[index, 2]),
                            "native_production_loss": float(native[index, 3]),
                            "ablation_production_loss": float(ablation[index, port, control_index, 3]),
                            "delta_production_loss": float(ablation[index, port, control_index, 3] - native[index, 3]),
                            "ablation_clipping_fraction": float(ablation[index, port, control_index, 4]),
                            "native_camera_power_w": float(native[index, 5]),
                            "ablation_camera_power_w": float(ablation[index, port, control_index, 5]),
                        })
            exposure_values = [float(value) for value in camera["exposure_sweep_s"]]
            for control_index, control in enumerate(CHANNELS):
                for exposure_index, exposure in enumerate(exposure_values):
                    per_port_snr = snr[:, control_index, exposure_index, :]
                    for port in range(PORTS):
                        per_port_noise_rows.append({
                            "endpoint": endpoint,
                            "control": control,
                            "exposure_s": exposure,
                            "port_zero_based": port,
                            "mean_proxy_snr": float(np.nanmean(per_port_snr[:, port])),
                            "median_proxy_snr": float(np.nanmedian(per_port_snr[:, port])),
                            "fraction_samples_snr_ge_1": float(np.nanmean(per_port_snr[:, port] >= 1.0)),
                            "fraction_samples_snr_ge_3": float(np.nanmean(per_port_snr[:, port] >= 3.0)),
                            "fraction_samples_snr_ge_5": float(np.nanmean(per_port_snr[:, port] >= 5.0)),
                        })
                    for index, row in enumerate(rows):
                        per_image_rows.append({"endpoint": endpoint, "row": int(row), "class": int(labels[index]), "control": control, "exposure_s": exposure, "observable_ports_snr_ge_1": int(counts[index, control_index, exposure_index, 0]), "observable_ports_snr_ge_3": int(counts[index, control_index, exposure_index, 1]), "observable_ports_snr_ge_5": int(counts[index, control_index, exposure_index, 2])})
    write_csv(output_root / "ablation" / "per_port_per_class.csv", per_class_rows)
    write_csv(output_root / "ablation" / "per_port_power_summary.csv", per_port_power_rows)
    write_csv(output_root / "ablation" / "per_image_noise_observable_port_counts.csv", per_image_rows)
    write_csv(output_root / "noise" / "per_port_observability.csv", per_port_noise_rows)
    write_json(output_root / "ablation" / "summary.json", summaries)
    return per_image_rows, per_port_noise_rows, summaries


def paired_endpoint_comparisons(endpoint_data: dict[str, dict[str, Any]], resamples: int, seed: int) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    pairs = (
        ("small_step_030000", "small_control_step_035000"),
        ("small_step_030000", "small_treatment_step_035000"),
        ("small_control_step_035000", "small_treatment_step_035000"),
    )
    for old_key, new_key in pairs:
        old, new = endpoint_data[old_key], endpoint_data[new_key]
        require(np.array_equal(old["rows"], new["rows"]) and np.array_equal(old["labels"], new["labels"]), f"small endpoint comparison lost original row pairing: {old_key}->{new_key}")
        record: dict[str, Any] = {"old_endpoint": old_key, "new_endpoint": new_key, "paired_population": "same original 5,000 DEV rows", "metrics": {}}
        for metric_index, metric in enumerate(METRICS):
            delta = new["native_metrics"][:, metric_index] - old["native_metrics"][:, metric_index]
            boot = bootstrap_effect(delta, old["labels"], resamples, seed)
            record["metrics"][metric] = {"mean_delta": float(boot["mean"][0]), "ci95_low": float(boot["ci95_low"][0]), "ci95_high": float(boot["ci95_high"][0])}
        comparisons[f"{old_key}__to__{new_key}"] = record
    return comparisons


def run_measure(cfg: dict[str, Any], output_root: Path, device: torch.device) -> dict[str, Any]:
    preflight_path = output_root / "preflight.json"
    require(preflight_path.is_file(), "measurement requires the explicit preflight receipt")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight.get("task_id") == TASK_ID and preflight.get("status") == "READY", "preflight did not pass or belongs to another task")
    require((output_root / "ablation" / "ranking_manifest.json").is_file(), "pre-frozen port ranking is missing")
    small = load_small_context(cfg, device)
    large = load_large_context(cfg, device)
    camera = read_noise_proxy(cfg)
    with np.load(output_root / "covariance" / "large_saved55000_train.npz", allow_pickle=False) as saved_covariance:
        large_power_from_saved_moments = np.real(np.diag(saved_covariance["maxport_gauge_complex_second_moment"]))
    ranking_payload = json.loads((output_root / "ablation" / "ranking_manifest.json").read_text(encoding="utf-8"))
    endpoint_specs = [("small", item["key"], item["source_state"], int(cfg["small"]["production_grid"])) for item in cfg["small"]["decoder_endpoints"]]
    endpoint_specs.append(("large", "large_step_030000", None, int(cfg["large"]["decoder_production_grid"])))
    endpoint_data: dict[str, dict[str, Any]] = {}
    cumulative_rows: list[dict[str, Any]] = []
    local_spectra: dict[str, Any] = {}
    endpoint_elapsed: dict[str, float] = {}
    started = time.perf_counter()
    for kind, endpoint_key, source_state, grid in endpoint_specs:
        population = state_population(kind, endpoint_key, source_state, small, large)
        data = process_endpoint(cfg, endpoint_key, kind, source_state, population, output_root, device, camera, grid)
        endpoint_data[endpoint_key] = data
        ranking = ranking_payload[endpoint_key]["strongest_to_weakest_zero_based"]
        order = np.asarray(ranking, dtype=np.int64)
        require(order.shape == (PORTS,) and set(order.tolist()) == set(range(PORTS)), f"frozen power ranking is invalid: {endpoint_key}")
        cumulative_rows.extend(cumulative_retention(kind, endpoint_key, population, output_root, order, grid, device))
        local_spectra[endpoint_key] = local_direction_spectrum(
            kind,
            endpoint_key,
            population,
            output_root,
            grid,
            camera,
            int(cfg["statistics"]["local_direction_rows_per_class"]),
            device,
        )
        endpoint_elapsed[endpoint_key] = data["elapsed_seconds_observed"]
        write_json(output_root / "measurement_progress.json", {
            "task_id": TASK_ID,
            "completed_endpoints": list(endpoint_data),
            "total_endpoints": len(endpoint_specs),
            "elapsed_seconds_observed": time.perf_counter() - started,
            "training_updates": 0,
        })
    write_csv(output_root / "ablation" / "cumulative_root_retention.csv", cumulative_rows)
    tolerance_counts: dict[str, Any] = {}
    for endpoint in endpoint_data:
        rows = [item for item in cumulative_rows if item["endpoint"] == endpoint]
        tolerance_counts[endpoint] = {}
        for tol in (0.01, 0.05):
            acceptable = [item["retained_roots"] for item in rows if item["mean_absolute_relative_mse_error"] <= tol]
            tolerance_counts[endpoint][str(tol)] = min(acceptable) if acceptable else None
    write_json(output_root / "noise" / "local_direction_spectra.json", local_spectra)
    per_image_noise, per_port_noise, summaries = write_port_tables(cfg, endpoint_data, output_root, camera)
    paired = paired_endpoint_comparisons(endpoint_data, int(cfg["statistics"]["bootstrap_resamples"]), int(cfg["statistics"]["bootstrap_seed"]))
    noise_summary = noise_population_summary(endpoint_data)
    figure_paths = create_figures(output_root, endpoint_data, cumulative_rows, local_spectra, camera)
    covariance = json.loads((output_root / "covariance" / "summary.json").read_text(encoding="utf-8"))
    report = {
        "schema": "gaussian_ols_effective_channel_noise_ablation.v1",
        "task_id": TASK_ID,
        "status": "CONDITIONAL_DIAGNOSTIC_COMPLETE_REAL_NOISE_CALIBRATION_PENDING",
        "outer_decision": "FROZEN_CONDITIONAL_DIAGNOSTIC_COMPLETE",
        "decision_question": "How much of the apparent 48 dB versus 32 dB port-power spread is concentration, noise observability loss, and decoder reconstruction contribution?",
        "evidence_tier": "diagnostic-only",
        "input_lineage": preflight["input_lineage"],
        "noise_proxy": {
            "parameters": camera,
            "method": "exact Poisson photon/dark-current shot noise convolved with Gaussian read noise; one-sided per-pixel full-well clipping; exact moments with a bounded read-noise transition window",
            "real_camera_calibration": "NOT_AVAILABLE; measured hardware channel counts remain undetermined",
            "gain": "each endpoint's saved fixed numerical gain; not a camera calibration",
        },
        "covariance": covariance,
        "decoder_ablation": summaries,
        "small_endpoint_paired_comparisons": paired,
        "cumulative_retention": {
            "port_order_source": "ranking_manifest.json frozen from previous population mean power before ablation outcomes",
            "tolerance_minimum_roots": tolerance_counts,
            "curves_file": "ablation/cumulative_root_retention.csv",
            "nonmonotonicity_preserved": True,
        },
        "noise_observability": noise_summary,
        "noise_weighted_local_direction_spectra": local_spectra,
        "preflight": {
            "status": preflight["status"],
            "all_100_port_complex_camera_bases_compared_to_reference": preflight["basis_all_fields_within_1e-4"],
            "direct_replay_max_mse_absolute_error": preflight["direct_production_vs_basis_checks"]["maximum_direct_vs_basis_mse_absolute_error"],
            "production_w_per_pixel_power_identity_max_relative_error": preflight["direct_production_vs_basis_checks"]["maximum_production_w_per_pixel_vs_complex_field_power_relative_error"],
            "direct_replay_and_ablation_checks_passed": preflight["direct_production_vs_basis_checks"]["passed"],
        },
        "artifacts": {
            "per_image_per_port_ablation": "ablation/per_image_per_port_ablation.csv.gz",
            "per_port_per_class": "ablation/per_port_per_class.csv",
            "per_port_power_summary": "ablation/per_port_power_summary.csv",
            "per_image_noise_counts": "ablation/per_image_noise_observable_port_counts.csv",
            "per_port_noise": "noise/per_port_observability.csv",
            "cumulative_retention": "ablation/cumulative_root_retention.csv",
            "covariance_spectra": "covariance/spectra.csv",
            "local_direction_spectra": "noise/local_direction_spectra.json",
            "figures": figure_paths,
        },
        "training_updates": 0,
        "optimizers_created": 0,
        "prior_samples": 0,
        "official_test_reads": 0,
        "hardware_operations": 0,
        "elapsed_seconds_observed": time.perf_counter() - started,
        "endpoint_elapsed_seconds": endpoint_elapsed,
        "bootstrap": {"resamples": int(cfg["statistics"]["bootstrap_resamples"]), "seed": int(cfg["statistics"]["bootstrap_seed"]), "class_stratified": True},
        "multiple_comparison": {"method": "Benjamini-Hochberg FDR on 100 per-root paired bootstrap sign tests, separately for each endpoint and ablation control", "q": float(cfg["statistics"]["fdr_q"])},
    }
    write_json(output_root / "machine_report.json", report)
    write_json(output_root / "measurement_complete.json", {
        "task_id": TASK_ID,
        "status": report["status"],
        "endpoint_sample_counts": {key: int(len(value["rows"])) for key, value in endpoint_data.items()},
        "per_port_ablation_rows": sum(len(value["rows"]) * PORTS * len(CHANNELS) for value in endpoint_data.values()),
        "proxy_noise_population_rows": sum(len(value["rows"]) * len(CHANNELS) * 4 for value in endpoint_data.values()),
        "covariance_populations": {key: value["sample_count"] for key, value in covariance.items()},
        "all_chunk_readbacks_passed": True,
        "full_population_preserved": True,
        "training_updates": 0,
        "optimizers_created": 0,
        "prior_samples": 0,
        "official_test_reads": 0,
        "hardware_operations": 0,
        "elapsed_seconds_observed": report["elapsed_seconds_observed"],
    })
    write_report(output_root)
    return report


def write_report(output_root: Path) -> None:
    machine_path = output_root / "machine_report.json"
    require(machine_path.is_file(), "machine report is missing")
    report = json.loads(machine_path.read_text(encoding="utf-8"))
    require(report.get("task_id") == TASK_ID, "machine report task identity changed")
    path = output_root / "中文诊断报告.md"
    path.write_text(render_effective_channel_report(report), encoding="utf-8")


def run_report(cfg: dict[str, Any], output_root: Path) -> None:
    del cfg
    write_report(output_root)
    print(json.dumps({"status": "REPORT_WRITTEN", "output": str(output_root / "中文诊断报告.md")}, ensure_ascii=False))


def build_camera_basis(kind: str, key: str, model: torch.nn.Module, device: torch.device, output_path: Path) -> tuple[np.ndarray, float]:
    before = frozen_snapshot(model)
    if output_path.is_file():
        with np.load(output_path, allow_pickle=False) as saved:
            result = saved["camera_field_per_unit_coefficient"].astype(np.complex64, copy=True)
            area = float(saved["pixel_area_m2"].reshape(()))
        require(result.ndim == 3 and result.shape[0] == PORTS and np.isfinite(result).all() and math.isfinite(area) and area > 0, f"saved recovery basis is invalid: {output_path}")
        assert_frozen_snapshot(model, before, key)
        return result, 0.0
    basis: list[np.ndarray] = []
    started = time.perf_counter()
    for port in range(PORTS):
        coefficients = torch.zeros((1, PORTS), dtype=torch.complex64, device=device)
        coefficients[0, port] = 1.0 + 0.0j
        field, _, area = camera_field_from_coefficients(kind, model, coefficients)
        basis.append(field[0, 0].detach().to(torch.complex64).cpu().numpy())
        del coefficients, field
        if device.type == "cuda" and (port + 1) % 10 == 0:
            torch.cuda.empty_cache()
    result = np.stack(basis, axis=0).astype(np.complex64, copy=False)
    require(result.ndim == 3 and result.shape[0] == PORTS and np.isfinite(result).all(), f"camera basis is invalid: {key}")
    write_npz(output_path, camera_field_per_unit_coefficient=result, pixel_area_m2=np.asarray(area, dtype=np.float64))
    assert_frozen_snapshot(model, before, key)
    return result, time.perf_counter() - started


def basis_global_phase_error(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    a = np.asarray(candidate, dtype=np.complex128).reshape(PORTS, -1)
    b = np.asarray(reference, dtype=np.complex128).reshape(PORTS, -1)
    cross = np.vdot(a, b)
    phase = float(np.angle(cross)) if abs(cross) else 0.0
    aligned = a * np.exp(1j * phase)
    ref_norms = np.linalg.norm(b, axis=1)
    field_errors = np.linalg.norm(aligned - b, axis=1) / np.maximum(ref_norms, np.finfo(np.float64).tiny)
    candidate_power = np.sum(np.abs(a) ** 2, axis=1)
    reference_power = np.sum(np.abs(b) ** 2, axis=1)
    power_errors = np.abs(candidate_power - reference_power) / np.maximum(reference_power, np.finfo(np.float64).tiny)
    return {
        "common_global_phase_alignment_rad": phase,
        "max_port_complex_field_relative_l2": float(field_errors.max()),
        "median_port_complex_field_relative_l2": float(np.median(field_errors)),
        "max_port_camera_power_relative_error": float(power_errors.max()),
        "median_port_camera_power_relative_error": float(np.median(power_errors)),
        "worst_complex_port_zero_based": int(np.argmax(field_errors)),
        "worst_power_port_zero_based": int(np.argmax(power_errors)),
    }


def independent_angular_spectrum_propagate(
    field: ComplexField, distance_m: float, plane_id: str
) -> ComplexField:
    """High-precision same-grid ASM reference with integer frequency indices.

    Frequency bins are assembled from signed integer indices rather than
    torch.fft.fftfreq. The carrier is separated from the small relative phase
    to avoid subtracting nearly equal large propagation phases.
    """
    if distance_m < 0.0 or field.tensor.dtype != torch.complex128:
        raise ValueError("independent ASM requires a non-negative distance and complex128 field")
    height, width = field.plane.shape
    dy, dx = field.plane.pixel_pitch_m
    wavelength = field.plane.wavelength_m
    device = field.tensor.device
    iy = torch.arange(height, device=device, dtype=torch.int64)
    ix = torch.arange(width, device=device, dtype=torch.int64)
    iy = torch.where(iy < (height + 1) // 2, iy, iy - height).to(torch.float64)
    ix = torch.where(ix < (width + 1) // 2, ix, ix - width).to(torch.float64)
    fy = iy / (height * dy)
    fx = ix / (width * dx)
    qx = (wavelength * fx).square()
    phase_scale = 2.0 * math.pi * distance_m / wavelength
    carrier = cmath.exp(1j * phase_scale)
    transfer = torch.empty((height, width), device=device, dtype=torch.complex128)
    rows_per_chunk = max(1, min(128, 8_388_608 // width))
    for start in range(0, height, rows_per_chunk):
        stop = min(start + rows_per_chunk, height)
        q = (wavelength * fy[start:stop, None]).square() + qx[None, :]
        propagating = q <= 1.0
        root = torch.sqrt((1.0 - q).clamp_min(0.0))
        relative_phase = -phase_scale * q / (1.0 + root)
        chunk = torch.polar(torch.ones_like(relative_phase), relative_phase) * carrier
        transfer[start:stop] = chunk * propagating
    propagated = torch.fft.ifft2(
        torch.fft.fft2(field.tensor, norm="ortho") * transfer, norm="ortho"
    )
    plane = field.plane.with_plane(plane_id, z_m=field.plane.z_m + distance_m)
    provenance = OperationProvenance(
        "independent_integer_index_complex128_angular_spectrum",
        field.plane.plane_id,
        distance_m,
        {"wavelength_m": wavelength, "sampling": plane.to_dict()},
    )
    return ComplexField(propagated, plane, provenance)


def independent_reference_camera_field(
    kind: str, model: torch.nn.Module, coefficients: torch.Tensor
) -> np.ndarray:
    """Run a frozen Decoder with the task-local independent complex128 ASM."""
    require(coefficients.dtype == torch.complex128, "reference coefficients must be complex128")
    module = d2nn_module if kind == "small" else chen_baseline_module
    original_propagator = module.angular_spectrum_propagate
    physical_optics = None
    if kind == "small":
        physical_optics = model.runtime.physical_optics
        require(physical_optics is not None, "small reference model lost its physical configuration")
    module.angular_spectrum_propagate = independent_angular_spectrum_propagate
    try:
        with torch.inference_mode():
            if kind == "small":
                state = ComplexOLSState(
                    coefficients=coefficients,
                    source="effective_channel_independent_reference",
                    basis_id=model.runtime.ols.basis_id,
                    mode_geometry=model.runtime.ols.mode_identity(),
                )
                model.runtime.physical_optics = None
                result = model.runtime.forward_from_ols(state)
                field = model.runtime.output_relay(result.generator_output_field).tensor
            else:
                captured: dict[str, torch.Tensor] = {}

                def capture(name: str, value: Any) -> None:
                    if name == "camera_full_field":
                        captured[name] = value.tensor

                model.forward_from_ols(coefficients, field_callback=capture)
                require("camera_full_field" in captured, "independent large Decoder omitted its camera field")
                field = chen_center_crop(captured["camera_full_field"], model.config.active_shape)
        require(field.dtype == torch.complex128 and torch.isfinite(field).all(), "independent reference field is invalid")
        return field[0, 0].detach().cpu().numpy().astype(np.complex128, copy=False)
    finally:
        module.angular_spectrum_propagate = original_propagator
        if kind == "small":
            model.runtime.physical_optics = physical_optics


def common_phase_port_errors(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    """Compare corresponding port fields using one phase shared across ports."""
    a = np.asarray(candidate, dtype=np.complex128).reshape(len(candidate), -1)
    b = np.asarray(reference, dtype=np.complex128).reshape(len(reference), -1)
    require(a.shape == b.shape and len(a) > 0, "reference field arrays differ in shape")
    cross = sum(np.vdot(left, right) for left, right in zip(a, b, strict=True))
    phase = float(np.angle(cross)) if abs(cross) else 0.0
    aligned = a * np.exp(1j * phase)
    ref_norm = np.linalg.norm(b, axis=1)
    errors = np.linalg.norm(aligned - b, axis=1) / np.maximum(ref_norm, np.finfo(np.float64).tiny)
    power_a = np.square(np.abs(a)).sum(axis=1)
    power_b = np.square(np.abs(b)).sum(axis=1)
    power_errors = np.abs(power_a - power_b) / np.maximum(power_b, np.finfo(np.float64).tiny)
    return {
        "common_global_phase_alignment_rad": phase,
        "per_port_complex_field_relative_l2": errors.tolist(),
        "per_port_camera_power_relative_error": power_errors.tolist(),
        "maximum_complex_field_relative_l2": float(errors.max()),
        "maximum_camera_power_relative_error": float(power_errors.max()),
    }


def validate_independent_asm_reference(device: torch.device) -> dict[str, Any]:
    plane = PlaneSpec("independent_reference_control", (32, 32), (3.0e-6, 3.0e-6), WAVELENGTH_M, 0.0, "reference_control")
    generator = torch.Generator(device=device).manual_seed(2026102101)
    real = torch.randn((1, 1, 32, 32), generator=generator, device=device, dtype=torch.float64)
    imag = torch.randn((1, 1, 32, 32), generator=generator, device=device, dtype=torch.float64)
    values = torch.complex(real, imag)
    initial = ComplexField(values, plane, OperationProvenance("reference_control_input", plane.plane_id, None))
    zero = independent_angular_spectrum_propagate(initial, 0.0, "zero_distance")
    zero_error = float(torch.linalg.vector_norm(zero.tensor - values) / torch.linalg.vector_norm(values))
    first = independent_angular_spectrum_propagate(initial, 1.1e-3, "composition_first")
    sequential = independent_angular_spectrum_propagate(first, 1.9e-3, "composition_second")
    combined = independent_angular_spectrum_propagate(initial, 3.0e-3, "composition_combined")
    composition_error = float(torch.linalg.vector_norm(sequential.tensor - combined.tensor) / torch.linalg.vector_norm(combined.tensor))
    height, width = plane.shape
    ky, kx = 3, -4
    y = torch.arange(height, device=device, dtype=torch.float64)[:, None]
    x = torch.arange(width, device=device, dtype=torch.float64)[None, :]
    phase = 2.0 * math.pi * (ky * y / height + kx * x / width)
    wave = torch.exp(torch.complex(torch.zeros_like(phase), phase))[None, None]
    wave_input = ComplexField(wave, plane, OperationProvenance("reference_plane_wave", plane.plane_id, None))
    distance = 2.3e-3
    wave_output = independent_angular_spectrum_propagate(wave_input, distance, "plane_wave_output").tensor
    fy = ky / (height * plane.pixel_pitch_m[0])
    fx = kx / (width * plane.pixel_pitch_m[1])
    q = (WAVELENGTH_M * fy) ** 2 + (WAVELENGTH_M * fx) ** 2
    expected = cmath.exp(1j * 2.0 * math.pi * distance / WAVELENGTH_M * math.sqrt(1.0 - q))
    analytic_error = float(torch.linalg.vector_norm(wave_output - wave * expected) / torch.linalg.vector_norm(wave))
    passed = zero_error <= 1e-13 and composition_error <= 1e-10 and analytic_error <= 1e-10
    require(passed, f"independent ASM analytic controls failed: zero={zero_error}, composition={composition_error}, plane={analytic_error}")
    return {
        "passed": passed,
        "zero_distance_relative_error": zero_error,
        "same_grid_propagation_composition_relative_error": composition_error,
        "single_fft_plane_wave_relative_error": analytic_error,
        "plane_wave_indices": {"ky": ky, "kx": kx},
        "independent_scalar_phase": {"real": expected.real, "imag": expected.imag},
    }


def run_independent_reference_diagnostic(
    cfg: dict[str, Any], output_root: Path, device: torch.device
) -> dict[str, Any]:
    """Disentangle same-grid precision from finite-window differences on probes."""
    recovery = cfg["recovery"]
    prior_root = ROOT / recovery["prior_attempt_dir"]
    prior_preflight_path = prior_root / "preflight.json"
    prior_errors_path = prior_root / "preflight" / "basis_reference_port_errors.csv"
    require(prior_preflight_path.is_file() and prior_errors_path.is_file(), "explicit prior INVALID_HARNESS evidence is incomplete")
    prior_preflight = json.loads(prior_preflight_path.read_text(encoding="utf-8"))
    require(prior_preflight.get("task_id") == TASK_ID and prior_preflight.get("status") == "INVALID_HARNESS", "prior attempt status or identity changed")
    if output_root.exists():
        progress_path = output_root / "preflight_progress.json"
        require(not (output_root / "preflight.json").exists(), "attempt_02 already has a final preflight receipt")
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            require(progress.get("task_id") == TASK_ID and progress.get("status") == "IN_PROGRESS", "attempt_02 recovery record is not resumable")
        else:
            write_json(progress_path, {"task_id": TASK_ID, "status": "IN_PROGRESS", "stage": "independent_reference"})
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        write_json(output_root / "preflight_progress.json", {"task_id": TASK_ID, "status": "IN_PROGRESS", "stage": "independent_reference"})
    reference_root = output_root / "preflight"
    reference_root.mkdir(parents=True, exist_ok=True)
    asm_controls = validate_independent_asm_reference(device)
    noise_moment_reference = validate_noise_moments(device, read_noise_proxy(cfg))
    small = load_small_context(cfg, device)
    large = load_large_context(cfg, device)
    prior_rows: list[dict[str, str]] = []
    with prior_errors_path.open("r", encoding="utf-8", newline="") as stream:
        prior_rows = list(csv.DictReader(stream))
    require(len(prior_rows) == 400, "prior per-port grid comparison is incomplete")
    probe_rows: list[dict[str, Any]] = []
    endpoint_results: dict[str, Any] = {}
    endpoints = [
        {
            "key": endpoint["key"],
            "kind": "small",
            "state_key": endpoint["source_state"],
            "grids": (int(cfg["small"]["production_grid"]), int(cfg["small"]["reference_grid"])),
            "coefficients": np.asarray(small["state_payloads"][endpoint["source_state"]]["coefficients"]),
        }
        for endpoint in cfg["small"]["decoder_endpoints"]
    ]
    endpoints.append({
        "key": "large_step_030000",
        "kind": "large",
        "state_key": "large_step_030000",
        "grids": (int(cfg["large"]["decoder_production_grid"]), int(cfg["large"]["decoder_reference_grid"])),
        "coefficients": large["population"]["coefficients"].detach().cpu().numpy(),
    })
    for endpoint in endpoints:
        key = endpoint["key"]
        coefficients = np.asarray(endpoint["coefficients"])
        mean_power = np.mean(np.square(np.abs(coefficients)), axis=0)
        order = np.argsort(mean_power, kind="stable")
        probes = {int(order[0]), int(order[len(order) // 2]), int(order[-1])}
        matching = [row for row in prior_rows if row["endpoint"] == key]
        require(len(matching) == PORTS, f"prior all-port error rows are missing for {key}")
        worst_field = max(matching, key=lambda row: float(row["complex_field_relative_l2"]))
        worst_power = max(matching, key=lambda row: float(row["camera_power_relative_error"]))
        probes.update((int(worst_field["port_zero_based"]), int(worst_power["port_zero_based"])))
        selected = sorted(probes)
        port_references: dict[int, dict[int, np.ndarray]] = {port: {} for port in selected}
        grid_results: dict[str, Any] = {}
        for grid_role, grid in zip(("production", "reference"), endpoint["grids"], strict=True):
            saved_path = prior_root / "bases" / f"{key}_{grid_role}_{grid}.npz"
            require(saved_path.is_file(), f"explicit saved basis is missing: {saved_path}")
            with np.load(saved_path, allow_pickle=False) as saved:
                saved_basis = saved["camera_field_per_unit_coefficient"].astype(np.complex128, copy=True)
                area = float(saved["pixel_area_m2"].reshape(()))
            require(saved_basis.shape[0] == PORTS and np.isfinite(saved_basis).all() and area > 0.0, f"saved basis is invalid: {saved_path}")
            if endpoint["kind"] == "small":
                production_model = small_model_for(endpoint["state_key"], cfg, small, grid, device)
                reference_model = small_model_for(endpoint["state_key"], cfg, small, grid, device).to(dtype=torch.float64)
            else:
                production_model = large_model_for(large, grid)
                reference_model = large_model_for(large, grid).to(dtype=torch.float64)
            production_before = frozen_snapshot(production_model)
            reference_before = frozen_snapshot(reference_model)
            production_fields: list[np.ndarray] = []
            reference_fields: list[np.ndarray] = []
            saved_fields: list[np.ndarray] = []
            max_saved_replay_error = 0.0
            for port in selected:
                coefficient = torch.zeros((1, PORTS), device=device, dtype=torch.complex64)
                coefficient[0, port] = 1.0 + 0.0j
                production, _, production_area = camera_field_from_coefficients(endpoint["kind"], production_model, coefficient)
                saved_field = saved_basis[port]
                prod_array = production[0, 0].detach().cpu().numpy().astype(np.complex128, copy=False)
                replay_error = float(np.linalg.norm(prod_array.reshape(-1) - saved_field.reshape(-1)) / max(np.linalg.norm(saved_field.reshape(-1)), np.finfo(np.float64).tiny))
                max_saved_replay_error = max(max_saved_replay_error, replay_error)
                coefficient_ref = torch.zeros((1, PORTS), device=device, dtype=torch.complex128)
                coefficient_ref[0, port] = 1.0 + 0.0j
                reference_field = independent_reference_camera_field(endpoint["kind"], reference_model, coefficient_ref)
                require(reference_field.shape == saved_field.shape and math.isclose(production_area, area, rel_tol=0.0, abs_tol=0.0), f"saved and live camera plane geometry differs: {key}/{grid}")
                production_fields.append(prod_array)
                saved_fields.append(saved_field)
                reference_fields.append(reference_field)
                port_references[port][grid] = reference_field
                probe_rows.append({"endpoint": key, "grid_role": grid_role, "grid": grid, "port_zero_based": port, "saved_production_replay_complex_error": replay_error})
                del coefficient, coefficient_ref, production, reference_field
            require(max_saved_replay_error <= 1.0e-6, f"saved basis no longer replays through production at {key}/{grid}: {max_saved_replay_error:.8g}")
            same_grid_saved = common_phase_port_errors(np.stack(saved_fields), np.stack(reference_fields))
            same_grid_live = common_phase_port_errors(np.stack(production_fields), np.stack(reference_fields))
            grid_results[str(grid)] = {
                "grid_role": grid_role,
                "probe_ports_zero_based": selected,
                "saved_basis_vs_live_production_max_relative_error": max_saved_replay_error,
                "saved_production_vs_independent_reference": same_grid_saved,
                "live_production_vs_independent_reference": same_grid_live,
            }
            assert_frozen_snapshot(production_model, production_before, f"{key}/{grid}/production probes")
            assert_frozen_snapshot(reference_model, reference_before, f"{key}/{grid}/independent reference probes")
            del production_model, reference_model, production_before, reference_before
            if device.type == "cuda":
                torch.cuda.empty_cache()
        prod_grid, ref_grid = endpoint["grids"]
        production_grid_errors = common_phase_port_errors(
            np.stack([port_references[port][prod_grid] for port in selected]),
            np.stack([port_references[port][ref_grid] for port in selected]),
        )
        endpoint_results[key] = {
            "population_role": "fixed DEV" if endpoint["kind"] == "small" else "fixed historical TRAIN",
            "source_checkpoint_step": int(endpoint["state_key"].split("_")[-1]) if endpoint["kind"] == "small" else int(large["checkpoint"]["completed_updates"]),
            "probe_ports_zero_based": selected,
            "previous_worst_complex_port_zero_based": int(worst_field["port_zero_based"]),
            "previous_worst_power_port_zero_based": int(worst_power["port_zero_based"]),
            "same_grid": grid_results,
            "independent_reference_cross_grid_difference": production_grid_errors,
            "same_grid_pass_1e-4": all(
                info["saved_production_vs_independent_reference"]["maximum_complex_field_relative_l2"] <= float(cfg["statistics"]["reference_relative_tolerance"])
                and info["saved_production_vs_independent_reference"]["maximum_camera_power_relative_error"] <= float(cfg["statistics"]["reference_relative_tolerance"])
                and info["live_production_vs_independent_reference"]["maximum_complex_field_relative_l2"] <= float(cfg["statistics"]["reference_relative_tolerance"])
                and info["live_production_vs_independent_reference"]["maximum_camera_power_relative_error"] <= float(cfg["statistics"]["reference_relative_tolerance"])
                for info in grid_results.values()
            ),
            "cross_grid_pass_1e-4_on_probes": production_grid_errors["maximum_complex_field_relative_l2"] <= float(cfg["statistics"]["reference_relative_tolerance"])
            and production_grid_errors["maximum_camera_power_relative_error"] <= float(cfg["statistics"]["reference_relative_tolerance"]),
        }
        write_json(reference_root / "independent_reference_progress.json", {"task_id": TASK_ID, "completed_endpoints": list(endpoint_results), "status": "IN_PROGRESS"})
    same_grid_pass = all(item["same_grid_pass_1e-4"] for item in endpoint_results.values())
    cross_grid_pass = all(item["cross_grid_pass_1e-4_on_probes"] for item in endpoint_results.values())
    result = {
        "schema": "gaussian_ols_independent_reference_diagnostic.v1",
        "task_id": TASK_ID,
        "status": "SAME_GRID_REFERENCE_VALIDATED" if same_grid_pass else "SAME_GRID_REFERENCE_MISMATCH",
        "same_grid_all_probes_within_1e-4": same_grid_pass,
        "cross_grid_all_probes_within_1e-4": cross_grid_pass,
        "measurement_gate_ready": same_grid_pass and cross_grid_pass,
        "reference_method": "task-local complex128 ASM; signed integer frequency indices; stable carrier-separated phase; same coherent re-emission, phase masks, finite crops, relay and camera integration",
        "independent_reference_controls": asm_controls,
        "noise_moment_independent_reference": noise_moment_reference,
        "common_phase_rule": "one global phase per endpoint/grid comparison, shared across all sampled ports; no per-port phase fitting",
        "omitted_terms": "none; independent reference keeps the declared finite-grid aperture, transmission, relay and camera plane",
        "endpoints": endpoint_results,
        "probe_rows": len(probe_rows),
        "no_training_updates": 0,
        "optimizers_created": 0,
        "prior_samples": 0,
        "official_test_reads": 0,
        "hardware_operations": 0,
    }
    write_json(reference_root / "independent_reference.json", result)
    write_csv(reference_root / "independent_reference_probe_rows.csv", probe_rows)
    write_json(output_root / "preflight_progress.json", {"task_id": TASK_ID, "status": "IN_PROGRESS" if result["measurement_gate_ready"] else "NUMERICAL_GATE_BLOCKED", "stage": "independent_reference_complete", "measurement_gate_ready": result["measurement_gate_ready"]})
    return result


def write_reference_gate_closeout(output_root: Path, reference: dict[str, Any]) -> None:
    same_grid_pass = bool(reference["same_grid_all_probes_within_1e-4"])
    reason = "CROSS_GRID_FINITE_WINDOW_OR_NUMERICAL_DIFFERENCE" if same_grid_pass else "SAME_GRID_PRODUCTION_REFERENCE_MISMATCH"
    status = f"INVALID_HARNESS / {reason}"
    rows = []
    for endpoint, item in reference["endpoints"].items():
        same = max(
            grid["saved_production_vs_independent_reference"]["maximum_complex_field_relative_l2"]
            for grid in item["same_grid"].values()
        )
        power = max(
            grid["saved_production_vs_independent_reference"]["maximum_camera_power_relative_error"]
            for grid in item["same_grid"].values()
        )
        cross = item["independent_reference_cross_grid_difference"]["maximum_complex_field_relative_l2"]
        rows.append({"endpoint": endpoint, "same_grid_complex_error": same, "same_grid_power_error": power, "independent_cross_grid_complex_error": cross, "probes": item["probe_ports_zero_based"]})
    report = {
        "schema": "gaussian_ols_effective_channel_noise_ablation_closeout.v1",
        "task_id": TASK_ID,
        "status": status,
        "scientific_measurement_completed": False,
        "decision_scope": "fixed numerical reference gate; no efficacy inference",
        "independent_reference": reference,
        "noise_moment_reference": reference["noise_moment_independent_reference"],
        "endpoint_summary": rows,
        "reason": "The unchanged 1e-4 production/reference gate was not satisfied. Full ablation and population noise measurement did not start.",
        "no_training_updates": 0,
        "optimizers_created": 0,
        "prior_samples": 0,
        "official_test_reads": 0,
        "hardware_operations": 0,
    }
    write_json(output_root / "machine_report.json", report)
    lines = [
        "# Gaussian OLS有效通道诊断：独立数值参考复核",
        "",
        "## 结果与状态",
        "",
        f"状态：**{status}**。冻结独立参考与精确代理噪声矩校核已完成；原定逐根消融与全人口噪声测量未启动。训练更新、优化器创建、先验采样、官方TEST读取和硬件操作均为0。",
        "",
        "| 端点 | 同网格最大复场误差 | 同网格最大功率误差 | 独立参考跨网格复场差 | 探针端口 |",
        "|---|---:|---:|---:|---|",
        *[f"| {row['endpoint']} | {row['same_grid_complex_error']:.6g} | {row['same_grid_power_error']:.6g} | {row['independent_cross_grid_complex_error']:.6g} | {row['probes']} |" for row in rows],
        "",
        "## 科学认识",
        "",
        "同网格独立复核检验传播实现精度；跨网格复核检验既定生产窗口与参考窗口的差别。若同网格误差低于1e-4而跨网格误差超限，证据支持差异来自窗口/采样条件，而不是单纯的复场精度缺陷；这仍不解除本计划固定的跨网格数值门槛。若同网格也超限，则生产与独立传播参考仍未对齐，不能使用基场作科学归因。",
        "",
        "## 目标缺口",
        "",
        "协方差虽已复算，完整的100端口冻结消融、等OLS功率对照、保留曲线、四曝光代理噪声占用率与局部方向谱尚未完成；真实相机暗帧、平场及标定仍缺。",
        "",
        "## 工程完成与验证",
        "",
        f"独立参考方法：{reference['reference_method']}。解析平面波、零距离、传播复合控制：{reference['independent_reference_controls']}。Poisson加Gaussian读噪声矩与独立标量枚举的最大均值/方差误差分别为{reference['noise_moment_independent_reference']['independent_scalar_enumeration']['mean_max_abs_error_e']:.3g} e和{reference['noise_moment_independent_reference']['independent_scalar_enumeration']['variance_max_abs_error_e2']:.3g} e²。检查点、固定输入、模式次序及冻结状态已在加载与逐端口重放中核对。机器证据：`preflight/independent_reference.json`。",
        "",
        "## 决策影响与下一步",
        "",
        "本次测量按原定1e-4门槛停止，因此不回答约48 dB与约32 dB差异分别造成多少可观察或重构有效通道。报告中的独立参考结果可用于下一轮工程复核；若同网格已通过而跨网格仍超限，需先由用户决定是否保持这项跨窗口准入条件或为窗口差异另设明确口径。本轮授权不包括改变阈值、几何或训练。",
        "",
    ]
    (output_root / "中文诊断报告.md").write_text("\n".join(lines), encoding="utf-8")


def output_paths(cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["output"]["run_dir"]


def run_preflight(cfg: dict[str, Any], output_root: Path, device: torch.device) -> dict[str, Any]:
    started = time.perf_counter()
    progress_path = output_root / "preflight_progress.json"
    reference_path = output_root / "preflight" / "independent_reference.json"
    require(reference_path.is_file(), "independent high-precision reference stage must complete before the production preflight")
    independent_reference = json.loads(reference_path.read_text(encoding="utf-8"))
    require(independent_reference.get("task_id") == TASK_ID, "independent reference task identity changed")
    require(independent_reference.get("measurement_gate_ready") is True, "independent reference did not validate the fixed-grid numerical gate")
    if output_root.exists():
        require(progress_path.is_file(), f"refusing to reuse a run directory without this task's explicit recovery record: {output_root}")
        prior_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        require(prior_progress.get("task_id") == TASK_ID and prior_progress.get("status") == "IN_PROGRESS", "refusing to overwrite completed or unrelated preflight evidence")
        require(not (output_root / "preflight.json").is_file(), "preflight receipt already exists; use measurement/report stages")
    else:
        output_root.mkdir(parents=True, exist_ok=False)
    for name in ("covariance", "bases", "ablation", "noise", "figures", "preflight"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    write_json(progress_path, {"task_id": TASK_ID, "status": "IN_PROGRESS", "started_seconds": time.time(), "basis_reuse_is_verified_by_direct_replay": True})
    small = load_small_context(cfg, device)
    large = load_large_context(cfg, device)
    noise = read_noise_proxy(cfg)

    covariance = run_covariances(cfg, small, large, output_root)
    rankings: dict[str, Any] = {}
    for endpoint in cfg["small"]["decoder_endpoints"]:
        coeff = small["state_payloads"][endpoint["source_state"]]["coefficients"]
        mean_power = np.mean(np.abs(coeff) ** 2, axis=0)
        order = np.argsort(mean_power)[::-1]
        rankings[endpoint["key"]] = {"source_population": "previous fixed 5,000 DEV rows for this checkpoint", "mean_power_w": mean_power, "strongest_to_weakest_zero_based": order.astype(np.int64)}
    with np.load(output_root / "covariance" / "large_saved55000_train.npz", allow_pickle=False) as saved_covariance:
        large_power = np.real(np.diag(saved_covariance["maxport_gauge_complex_second_moment"]))
    large_order = np.argsort(large_power)[::-1]
    rankings["large_step_030000"] = {"source_population": "previous saved 55,000 TRAIN moment; fixed before ablation", "mean_power_w": large_power, "strongest_to_weakest_zero_based": large_order.astype(np.int64)}
    write_json(output_root / "ablation" / "ranking_manifest.json", rankings)
    basis_info: dict[str, Any] = {}
    bases: dict[str, dict[str, np.ndarray]] = {}
    elapsed_basis: dict[str, float] = {}
    started_basis = time.perf_counter()
    for endpoint in cfg["small"]["decoder_endpoints"]:
        key = endpoint["key"]
        source_state = endpoint["source_state"]
        for grid_role, grid in (("production", cfg["small"]["production_grid"]), ("reference", cfg["small"]["reference_grid"])):
            model = small_model_for(source_state, cfg, small, int(grid), device)
            name = f"{key}_{grid_role}_{grid}"
            basis, elapsed = build_camera_basis("small", name, model, device, output_root / "bases" / f"{name}.npz")
            bases.setdefault(key, {})[grid_role] = basis
            elapsed_basis[name] = elapsed
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        basis_info[key] = basis_global_phase_error(bases[key]["production"], bases[key]["reference"])
    large_key = "large_step_030000"
    for grid_role, grid in (("production", cfg["large"]["decoder_production_grid"]), ("reference", cfg["large"]["decoder_reference_grid"])):
        model = large_model_for(large, int(grid))
        name = f"{large_key}_{grid_role}_{grid}"
        basis, elapsed = build_camera_basis("large", name, model, device, output_root / "bases" / f"{name}.npz")
        bases.setdefault(large_key, {})[grid_role] = basis
        elapsed_basis[name] = elapsed
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    basis_info[large_key] = basis_global_phase_error(bases[large_key]["production"], bases[large_key]["reference"])
    basis_elapsed = time.perf_counter() - started_basis
    covariance_complete = bool(covariance) and len(covariance) == 11
    basis_tolerance = float(cfg["statistics"]["reference_relative_tolerance"])
    basis_pass = all(
        item["max_port_complex_field_relative_l2"] <= basis_tolerance
        and item["max_port_camera_power_relative_error"] <= basis_tolerance
        for item in basis_info.values()
    )

    # Exact Poisson plus Gaussian read-noise moments are compared with an
    # independently enumerated scalar reference and checked on CUDA.
    moment_reference = validate_noise_moments(device, noise)
    direct_checks = direct_preflight(cfg, small, large, bases, output_root, device)
    model_lineage = {
        "small": {
            "source_status": small["previous_machine_report"]["status"],
            "source_run": cfg["data"]["small_population"],
            "data_role": "fixed DEV 5,000; 500/class; original row ids and labels checked against training.pt",
            "all_nine_covariance_states": list(small["state_payloads"]),
            "decoder_endpoints": [x["key"] for x in cfg["small"]["decoder_endpoints"]],
        },
        "large": {
            "source_checkpoint": cfg["large"]["checkpoint"],
            "checkpoint_step": int(large["checkpoint"]["completed_updates"]),
            "source_encoder_grid": 2048,
            "decoder_production_grid": 4096,
            "decoder_reference_grid": 6144,
            "fixed_population_role": "TRAIN 300; 30/class; uses saved 2048-Encoder coefficients without re-encoding",
            "source_report_status": large["source_report"]["status"],
            "saved_adam_steps_verified_without_optimizer_creation": large["saved_adam_steps"],
            "camera_gain_w_inverse": large["camera_gain"],
        },
    }
    preflight = {
        "schema": "gaussian_ols_effective_channel_noise_ablation_preflight.v1",
        "task_id": TASK_ID,
        "status": "READY" if basis_pass and covariance_complete and moment_reference["passed"] and direct_checks["passed"] and independent_reference["measurement_gate_ready"] else "INVALID_HARNESS",
        "decision_scope": "frozen conditional diagnostics only; no model selection or training authorization",
        "input_lineage": model_lineage,
        "parameter_and_buffer_freeze": "checked at basis generation and checked again during direct production replays",
        "covariance_populations": {key: record["sample_count"] for key, record in covariance.items()},
        "basis_checks": basis_info,
        "basis_all_fields_within_1e-4": basis_pass,
        "independent_high_precision_reference": independent_reference,
        "basis_elapsed_by_endpoint_seconds": elapsed_basis,
        "basis_elapsed_seconds": basis_elapsed,
        "measured_basis_port_fields": sum(int(value["production"].shape[0]) + int(value["reference"].shape[0]) for value in bases.values()),
        "noise_proxy": noise,
        "noise_moment_independent_reference": moment_reference,
        "direct_production_vs_basis_checks": direct_checks,
        "no_training_updates": 0,
        "optimizers_created": 0,
        "prior_samples": 0,
        "official_test_reads": 0,
        "hardware_operations": 0,
        "elapsed_seconds_observed": time.perf_counter() - started,
        "cuda": {"available": torch.cuda.is_available(), "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
    }
    write_json(output_root / "preflight.json", preflight)
    write_json(progress_path, {"task_id": TASK_ID, "status": preflight["status"], "elapsed_seconds_observed": preflight["elapsed_seconds_observed"], "basis_fields": preflight["measured_basis_port_fields"]})
    write_json(output_root / "checkpoint_manifest.json", model_lineage)
    write_json(output_root / "preflight" / "timing_budget.json", {
        "basis_elapsed_seconds": basis_elapsed,
        "basis_elapsed_by_endpoint_seconds": elapsed_basis,
        "completed_basis_fields": preflight["measured_basis_port_fields"],
        "recovery_policy": "each endpoint/grid basis is an independently saved and read-back-verified NPZ; completed bases are reusable after explicit identity checks",
    })
    return preflight


def validate_noise_moments(device: torch.device, camera: dict[str, Any]) -> dict[str, Any]:
    probe_lambda = np.asarray([0.0, 1.0, 100.0, 10000.0, 29900.0, 30000.0, 30100.0, 60000.0], dtype=np.float64)
    cuda_mu, cuda_var = clipped_camera_moments(torch.from_numpy(probe_lambda).to(device=device, dtype=torch.float64), camera)
    cpu_mu, cpu_var = clipped_camera_moments(torch.from_numpy(probe_lambda), camera)
    cpu_mean_error = float(np.max(np.abs(cpu_mu.numpy() - cuda_mu.cpu().numpy())))
    cpu_variance_error = float(np.max(np.abs(cpu_var.numpy() - cuda_var.cpu().numpy())))
    require(cpu_mean_error <= 1e-9 and cpu_variance_error <= 1e-6, "independent CPU and CUDA camera moment execution disagrees")
    reference_lambda = probe_lambda[[0, 1, 4, 5, 6]]
    reference_moments = np.asarray([enumerated_poisson_read_moments(float(value), camera) for value in reference_lambda], dtype=np.float64)
    torch_reference, variance_reference = clipped_camera_moments(torch.from_numpy(reference_lambda), camera)
    mean_error = float(np.max(np.abs(torch_reference.numpy() - reference_moments[:, 0])))
    var_error = float(np.max(np.abs(variance_reference.numpy() - reference_moments[:, 1])))
    require(mean_error <= 2e-7 and var_error <= 2e-5, "Poisson/read-noise formula disagrees with independent scalar enumeration")
    low_mean, low_var = clipped_camera_moments(torch.tensor([0.0], dtype=torch.float64), camera)
    saturated_mean, saturated_var = clipped_camera_moments(torch.tensor([1e7], dtype=torch.float64), camera)
    require(low_mean.item() == 0.0 and abs(low_var.item() - float(camera["read_noise_e_rms"]) ** 2) <= 1e-10 and saturated_mean.item() <= float(camera["full_well_e"]) and saturated_var.item() < low_var.item(), "camera clipping/noise counterexample failed")
    # Independent phase and intensity counterexamples detect lost coherent terms.
    equal_amplitude = np.asarray([1.0 + 0j, 1.0 + 0j])
    destructive = abs(equal_amplitude.sum()) ** 2
    incoherent_sum = float(np.square(np.abs(equal_amplitude)).sum())
    require(destructive == 4.0 and incoherent_sum == 2.0, "coherent-cross-term negative control failed")
    return {
        "passed": True,
        "distribution": "Poisson(photons + dark-current mean) + Gaussian read noise, upper-clipped at full well",
        "independent_scalar_enumeration": {"lambda_e": reference_lambda.tolist(), "mean_max_abs_error_e": mean_error, "variance_max_abs_error_e2": var_error, "enumeration_tail_sigma": 14.0},
        "cpu_cuda_execution": {"mean_max_abs_error_e": cpu_mean_error, "variance_max_abs_error_e2": cpu_variance_error},
        "read_noise_transition_window": {"poisson_sigma_cut": 14.0, "read_noise_sigma_cut": 12.0, "omitted_read_clipping_probability_bound": "less than 7e-39 per conditional Gaussian tail; Poisson mass outside the transition band is below the 14-sigma Chernoff bound"},
        "low_count_noise_control": {"mean_e": float(low_mean.item()), "variance_e2": float(low_var.item())},
        "high_saturation_control": {"mean_e": float(saturated_mean.item()), "variance_e2": float(saturated_var.item())},
        "missing_coherent_cross_term_control": {"coherent_intensity": destructive, "incorrect_incoherent_sum": incoherent_sum},
    }


def _full_well_integer(camera: dict[str, Any]) -> int:
    full_well = float(camera["full_well_e"])
    require(math.isfinite(full_well) and full_well > 1.0 and full_well.is_integer(), "exact Poisson clipping requires an integer full-well count")
    return int(full_well)


def _near_full_well(lam: torch.Tensor, full_well: int, read_sigma: float) -> torch.Tensor:
    distance = 14.0 * torch.sqrt(torch.maximum(lam, torch.full_like(lam, float(full_well)))) + 14.0 * read_sigma
    return (lam - float(full_well)).abs() <= distance


def _normal_upper_clipped_conditional_offsets(
    integer_count: torch.Tensor, full_well: int, read_sigma: float
) -> tuple[torch.Tensor, torch.Tensor]:
    offset = integer_count - float(full_well)
    if read_sigma == 0.0:
        mean_offset = offset.clamp_max(0.0)
        return mean_offset, mean_offset.square()
    z = -offset / read_sigma
    cdf = 0.5 * torch.erfc(-z / math.sqrt(2.0))
    phi = torch.exp(-0.5 * z.square()) / math.sqrt(2.0 * math.pi)
    mean_offset = offset * cdf - read_sigma * phi
    second_offset = (offset.square() + read_sigma**2) * cdf - read_sigma * offset * phi
    return mean_offset, second_offset


def clipped_camera_moments(mu: torch.Tensor, camera: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Moments of min(Poisson(mu) + Gaussian(read_sigma), full_well)."""
    lam = mu.to(dtype=torch.float64).clamp_min(0.0)
    full_well = _full_well_integer(camera)
    read_sigma = float(camera["read_noise_e_rms"])
    require(math.isfinite(read_sigma) and read_sigma >= 0.0, "read-noise RMS must be finite and non-negative")
    mean = torch.where(lam < full_well, lam, torch.full_like(lam, float(full_well)))
    variance = torch.where(lam < full_well, lam + read_sigma**2, torch.zeros_like(lam))
    flat_lam = lam.reshape(-1)
    active_mask = _near_full_well(flat_lam, full_well, read_sigma)
    selected = flat_lam[active_mask]
    if selected.numel():
        shape_f = torch.full_like(selected, float(full_well))
        cdf_fm1 = torch.special.gammaincc(shape_f, selected)  # P(K <= F-1)
        cdf_fm2 = torch.special.gammaincc(shape_f - 1.0, selected)  # P(K <= F-2)
        cdf_fm3 = torch.special.gammaincc(shape_f - 2.0, selected)  # P(K <= F-3)
        mean_offset = selected * cdf_fm2 - float(full_well) * cdf_fm1
        second_offset = selected.square() * cdf_fm3 + selected * cdf_fm2 - 2.0 * float(full_well) * selected * cdf_fm2 + (float(full_well**2) + read_sigma**2) * cdf_fm1
        if read_sigma > 0.0:
            counts = torch.arange(full_well - 12, full_well + 13, device=lam.device, dtype=torch.float64)
            log_p = -selected[:, None] + counts[None, :] * selected[:, None].log() - torch.lgamma(counts[None, :] + 1.0)
            probabilities = torch.exp(log_p)
            conditional_mean_offset, conditional_second_offset = _normal_upper_clipped_conditional_offsets(counts[None, :], full_well, read_sigma)
            baseline_mean_offset = (counts - float(full_well)).clamp_max(0.0)[None, :]
            baseline_second_offset = torch.where(counts[None, :] < full_well, baseline_mean_offset.square() + read_sigma**2, torch.zeros_like(baseline_mean_offset))
            mean_offset += (probabilities * (conditional_mean_offset - baseline_mean_offset)).sum(dim=-1)
            second_offset += (probabilities * (conditional_second_offset - baseline_second_offset)).sum(dim=-1)
        mean_selected = float(full_well) + mean_offset
        variance_selected = (second_offset - mean_offset.square()).clamp_min(0.0)
        mean.reshape(-1)[active_mask] = mean_selected
        variance.reshape(-1)[active_mask] = variance_selected
    return mean, variance


def clipped_camera_mean_derivative(mu: torch.Tensor, camera: dict[str, Any]) -> torch.Tensor:
    """Exact derivative of clipped expected electrons with respect to Poisson mean."""
    lam = mu.to(dtype=torch.float64).clamp_min(0.0)
    full_well = _full_well_integer(camera)
    read_sigma = float(camera["read_noise_e_rms"])
    derivative = torch.special.gammaincc(torch.full_like(lam, float(full_well)), lam)
    if read_sigma == 0.0:
        return derivative
    flat_lam = lam.reshape(-1)
    active_mask = _near_full_well(flat_lam, full_well, read_sigma)
    selected = flat_lam[active_mask]
    if not selected.numel():
        return derivative
    counts = torch.arange(full_well - 13, full_well + 13, device=lam.device, dtype=torch.float64)
    next_counts = counts + 1.0
    log_p = -selected[:, None] + counts[None, :] * selected[:, None].log() - torch.lgamma(counts[None, :] + 1.0)
    probabilities = torch.exp(log_p)
    means, _ = _normal_upper_clipped_conditional_offsets(counts[None, :], full_well, read_sigma)
    next_means, _ = _normal_upper_clipped_conditional_offsets(next_counts[None, :], full_well, read_sigma)
    delta = means - (counts - float(full_well)).clamp_max(0.0)[None, :]
    next_delta = next_means - (next_counts - float(full_well)).clamp_max(0.0)[None, :]
    correction = (probabilities * (next_delta - delta)).sum(dim=-1)
    derivative.reshape(-1)[active_mask] += correction
    return derivative


def enumerated_poisson_read_moments(mu: float, camera: dict[str, Any]) -> tuple[float, float]:
    """Independent scalar summation used only for preflight reference checks."""
    lam = max(float(mu), 0.0)
    full_well = _full_well_integer(camera)
    read_sigma = float(camera["read_noise_e_rms"])
    if lam == 0.0:
        low = high = 0
    else:
        radius = 14.0 * math.sqrt(lam) + 15.0
        low = max(0, int(math.floor(lam - radius)))
        high = int(math.ceil(lam + radius))
    mean_terms: list[float] = []
    second_terms: list[float] = []
    for count in range(low, high + 1):
        probability = 1.0 if lam == 0.0 and count == 0 else 0.0
        if lam > 0.0:
            probability = math.exp(-lam + count * math.log(lam) - math.lgamma(count + 1.0))
        if probability == 0.0:
            continue
        offset = float(count - full_well)
        if read_sigma == 0.0:
            conditional_mean_offset = min(offset, 0.0)
            conditional_second_offset = conditional_mean_offset**2
        else:
            z = -offset / read_sigma
            cdf = 0.5 * math.erfc(-z / math.sqrt(2.0))
            phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
            conditional_mean_offset = offset * cdf - read_sigma * phi
            conditional_second_offset = (offset**2 + read_sigma**2) * cdf - read_sigma * offset * phi
        mean_terms.append(probability * conditional_mean_offset)
        second_terms.append(probability * conditional_second_offset)
    mean = math.fsum(mean_terms)
    second = math.fsum(second_terms)
    variance = max(second - mean * mean, 0.0)
    return float(full_well) + mean, variance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research/lightgen/gaussian_ols_effective_channel_noise_ablation_v1.yaml"))
    parser.add_argument("--mode", choices=("reference", "preflight", "measure", "report"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    cfg = read_config(ROOT / args.config)
    output_root = output_paths(cfg)
    if args.mode == "reference":
        require(args.device == "cuda" and torch.cuda.is_available(), "the independent frozen Decoder reference requires the configured CUDA device")
        result = run_independent_reference_diagnostic(cfg, output_root, torch.device("cuda"))
        print(json.dumps(clean({"status": result["status"], "measurement_gate_ready": result["measurement_gate_ready"], "output": str(output_root / "preflight" / "independent_reference.json")}), ensure_ascii=False))
        if not result["measurement_gate_ready"]:
            write_reference_gate_closeout(output_root, result)
            raise SystemExit(2)
        return
    if args.mode == "preflight":
        require(args.device == "cuda" and torch.cuda.is_available(), "the approved CUDA diagnostic requires an available CUDA device")
        result = run_preflight(cfg, output_root, torch.device("cuda"))
        print(json.dumps(clean({"status": result["status"], "output": str(output_root), "elapsed_seconds": result["elapsed_seconds_observed"]}), ensure_ascii=False))
        if result["status"] != "READY":
            raise SystemExit(2)
        return
    if args.mode == "measure":
        require(args.device == "cuda" and torch.cuda.is_available(), "the approved CUDA diagnostic requires an available CUDA device")
        run_measure(cfg, output_root, torch.device("cuda"))
        return
    reference_path = output_root / "preflight" / "independent_reference.json"
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        require(reference.get("task_id") == TASK_ID, "independent reference task identity changed")
        if reference.get("measurement_gate_ready") is False:
            write_reference_gate_closeout(output_root, reference)
            print(json.dumps({"status": "NUMERICAL_GATE_BLOCKED", "output": str(output_root / "machine_report.json")}, ensure_ascii=False))
            return
    run_report(cfg, output_root)


if __name__ == "__main__":
    main()
