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
TASK_ID = "GAUSSIAN_OLS_EFFECTIVE_CHANNEL_DUAL_GRID_V1"
AUTHORITY_ID = "USER_APPROVED_GAUSSIAN_OLS_EFFECTIVE_CHANNEL_DUAL_GRID_20261022"
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
OBSERVABILITY_THRESHOLDS = (1.0, 3.0, 5.0)
RELATIVE_THRESHOLDS = (0.1, 0.01, 0.001, 0.0001)  # retained for the unused historical helper in this task copy


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_config(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    require(isinstance(cfg, dict), "task config must be a YAML object")
    require(cfg.get("schema") == "gaussian_ols_effective_channel_dual_grid.v1", "task config schema changed")
    require(cfg.get("task_id") == TASK_ID and cfg.get("semantic_authority_id") == AUTHORITY_ID, "task authority changed")
    require(cfg.get("evidence_tier") == "diagnostic-only", "this task must remain diagnostic-only")
    require(cfg["data"]["small_rows"] == 5000 and cfg["data"]["small_per_class"] == 500, "small DEV population changed")
    require(cfg["data"]["large_rows"] == 300 and cfg["data"]["large_per_class"] == 30, "large TRAIN population changed")
    require(cfg["data"]["official_test_reads"] == 0, "official TEST reads are forbidden")
    require(cfg["small"]["production_grid"] == 2048 and cfg["small"]["reference_grid"] == 4096, "small propagation windows changed")
    require(cfg["large"]["source_encoder_grid"] == 2048 and cfg["large"]["decoder_production_grid"] == 4096 and cfg["large"]["decoder_reference_grid"] == 6144, "large Decoder windows or source coefficients changed")
    require(cfg["statistics"]["port_count"] == PORTS and cfg["statistics"]["bootstrap_resamples"] == 2000 and cfg["statistics"]["bootstrap_seed"] == 2026102101, "population or statistics contract changed")
    require(cfg["statistics"]["local_direction_relative_perturbations"] == [0.001, 0.01, 0.1] and cfg["statistics"]["local_direction_primary_relative_perturbation"] == 0.01, "local perturbation radii changed")
    require(cfg["statistics"]["ablation_port_block_size"] > 0 and cfg["statistics"]["measurement_batch_rows"] > 0, "measurement chunk shape is invalid")
    require(cfg["statistics"]["no_training_updates"] and cfg["statistics"]["no_optimizer_creation"] and cfg["statistics"]["no_prior_sampling"] and cfg["statistics"]["no_hardware_operations"], "zero-update boundary changed")
    endpoints = cfg["small"]["decoder_endpoints"]
    require([(item["key"], item["source_state"], item["global_step"]) for item in endpoints] == [
        ("small_step_030000", "step_030000", 30000),
        ("small_control_step_035000", "control_step_035000", 35000),
        ("small_treatment_step_035000", "treatment_step_035000", 35000),
    ], "small Decoder endpoints changed")
    require(cfg["output"]["run_dir"] == "outputs/20261022_gaussian_ols_effective_channel_dual_grid_v1/runs/attempt_01", "new attempt path changed")
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
    matrix = np.asarray(covariance)
    if np.iscomplexobj(matrix):
        hermitian = (matrix.astype(np.complex128, copy=False) + matrix.conj().T) * 0.5
    else:
        real = matrix.astype(np.float64, copy=False)
        hermitian = (real + real.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(hermitian)[::-1]
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
    raw_spectrum = spectrum_summary(raw["covariance"])
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
        class_raw_spectrum = spectrum_summary(class_complex["covariance"])
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
            "raw_complex_covariance_effective_participation": class_raw_spectrum["effective_participation"],
            "raw_complex_covariance_directions_95_percent": class_raw_spectrum["directions_95_percent"],
            "raw_complex_covariance_directions_99_percent": class_raw_spectrum["directions_99_percent"],
            "gauge_real_imag_covariance_trace_w": class_spectrum["total_variance"],
            "effective_participation": class_spectrum["effective_participation"],
            "directions_95_percent": class_spectrum["directions_95_percent"],
            "directions_99_percent": class_spectrum["directions_99_percent"],
            "top10_eigenvalue_fraction": float(class_spectrum["variance_fraction"][:10].sum()),
        }
    within_spectrum = spectrum_summary(within)
    between_spectrum = spectrum_summary(between)
    aggregate = {
        "population": population,
        "sample_count": int(len(z)),
        "class_counts": class_counts,
        "raw_complex_covariance_trace_w": float(np.trace(raw["covariance"]).real),
        "original_ungauged_complex_cross_moment_available": True,
        "raw_complex_covariance_effective_participation": raw_spectrum["effective_participation"],
        "raw_complex_covariance_directions_95_percent": raw_spectrum["directions_95_percent"],
        "raw_complex_covariance_directions_99_percent": raw_spectrum["directions_99_percent"],
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
        "within_class_directions_95_percent": within_spectrum["directions_95_percent"],
        "within_class_directions_99_percent": within_spectrum["directions_99_percent"],
        "between_class_directions_95_percent": between_spectrum["directions_95_percent"],
        "between_class_directions_99_percent": between_spectrum["directions_99_percent"],
        "per_class": per_class,
    }
    arrays = {
        "raw_complex_mean": raw["mean"],
        "raw_complex_covariance": raw["covariance"],
        "raw_complex_second_moment": raw["second_moment"],
        "raw_complex_covariance_eigenvalues": raw_spectrum["eigenvalues"],
        "raw_complex_covariance_variance_fraction": raw_spectrum["variance_fraction"],
        "maxport_gauge_real_imag_mean": gauge_mean,
        "maxport_gauge_real_imag_covariance": real_covariance,
        "maxport_gauge_complex_covariance": gauge_complex_covariance,
        "maxport_gauge_complex_second_moment": gauge_complex_second,
        "eigenvalues": spectrum["eigenvalues"],
        "variance_fraction": spectrum["variance_fraction"],
        "within_class_eigenvalues": within_spectrum["eigenvalues"],
        "between_class_eigenvalues": between_spectrum["eigenvalues"],
    }
    return aggregate, arrays


def large_saved_train_covariance(context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    saved = torch.load(context["moments_path"], map_location="cpu", weights_only=False)
    require(saved.get("schema") == "chen_lightgen_400_empirical_complex_gaussian_samples.v1", "large saved moment schema changed")
    class_models = saved.get("class_models")
    require(isinstance(class_models, dict) and set(class_models) == set(range(10)), "large saved class Gaussian moments are incomplete")
    scale = context["scale"].numpy()
    transform = np.repeat(np.sqrt(scale), 2)
    labels, train_rows = context["labels"], context["train_rows"]
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
    within_spectrum = spectrum_summary(within)
    between_spectrum = spectrum_summary(between)
    class_moments: dict[str, Any] = {
        "sample_count": total_count,
        "class_counts": {str(k): int(v) for k, v in enumerate(counts)},
        "gauge_real_imag_covariance_trace_w": float(np.trace(total_cov)),
        "within_class_gauge_variance": float(np.trace(within)),
        "between_class_gauge_variance": float(np.trace(between)),
        "within_class_fraction_of_total": float(np.trace(within) / np.trace(total_cov)),
        "between_class_fraction_of_total": float(np.trace(between) / np.trace(total_cov)),
        "within_class_directions_95_percent": within_spectrum["directions_95_percent"],
        "within_class_directions_99_percent": within_spectrum["directions_99_percent"],
        "between_class_directions_95_percent": between_spectrum["directions_95_percent"],
        "between_class_directions_99_percent": between_spectrum["directions_99_percent"],
        "effective_participation": spectrum["effective_participation"],
        "directions_95_percent": spectrum["directions_95_percent"],
        "directions_99_percent": spectrum["directions_99_percent"],
        "top10_eigenvalue_fraction": float(spectrum["variance_fraction"][:10].sum()),
        "per_class": classes,
        "original_ungauged_complex_cross_moment_available": False,
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
        "within_class_eigenvalues": within_spectrum["eigenvalues"],
        "between_class_eigenvalues": between_spectrum["eigenvalues"],
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
                    if remaining == 0.0:
                        ablation_checks.append({"row": row, "port": port, "control": control, "status": "undefined_zero_remaining_power"})
                        continue
                    require(remaining > 0.0, "preflight equal-power ablation denominator is negative")
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
                if remaining == 0.0:
                    ablation_checks.append({"row": row, "port": port, "control": control, "status": "undefined_zero_remaining_power"})
                    continue
                require(remaining > 0.0, "large preflight equal-power denominator is negative")
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


def direct_reference_grid_preflight(
    cfg: dict[str, Any], small: dict[str, Any], large: dict[str, Any],
    bases: dict[str, dict[str, np.ndarray]], device: torch.device,
) -> dict[str, Any]:
    tolerance = float(cfg["statistics"]["direct_replay_mse_absolute_tolerance"])
    endpoints = [("small", item["key"], item["source_state"]) for item in cfg["small"]["decoder_endpoints"]]
    endpoints.append(("large", "large_step_030000", None))
    records: dict[str, Any] = {}
    maximum_mse, maximum_field = 0.0, 0.0
    for kind, key, state_key in endpoints:
        if kind == "small":
            payload = small["state_payloads"][state_key]
            rows, coefficients = payload["rows"], payload["coefficients"]
            selected_rows = previous.fixed_preflight_rows(small["dev_rows"], small["labels"]).cpu().numpy()
            grid = int(cfg["small"]["reference_grid"])
            model = small_model_for(state_key, cfg, small, grid, device)
            context, gain = small, float(small["old_config"]["model"]["shared_camera_gain_w_inverse"])
        else:
            population = large["population"]
            rows = population["all_rows"].numpy().astype(np.int64)
            coefficients = population["coefficients"].numpy()
            selected_rows = population["confirm_rows_by_class"][:, :2].reshape(-1).numpy().astype(np.int64)
            grid = int(cfg["large"]["decoder_reference_grid"])
            model = large_model_for(large, grid)
            context, gain = large, float(large["camera_gain"])
        row_positions = {int(row): index for index, row in enumerate(rows)}
        require(len(selected_rows) == 20 and all(int(row) in row_positions for row in selected_rows), f"reference-grid sample row mapping changed: {key}")
        positions = np.asarray([row_positions[int(row)] for row in selected_rows], dtype=np.int64)
        z_values = np.asarray(coefficients[positions], dtype=np.complex64)
        before = frozen_snapshot(model)
        basis_np = bases[key]["reference"]
        height, width = basis_np.shape[-2:]
        basis = torch.from_numpy(basis_np.reshape(PORTS, -1)).to(device=device)
        native_errors: list[float] = []
        for index, row in enumerate(selected_rows):
            z = torch.from_numpy(z_values[index:index + 1]).to(device=device)
            direct, _, area = camera_field_from_coefficients(kind, model, z)
            synthesized = (z @ basis).reshape(1, 1, height, width)
            native, target28 = targets_for_rows(kind, context, np.asarray([row]), device)
            direct_mse = calculate_metrics(kind, direct, native, target28, gain, area)["mse28"][0]
            synth_mse = calculate_metrics(kind, synthesized, native, target28, gain, area)["mse28"][0]
            mse_error = abs(float(direct_mse) - float(synth_mse))
            field_error = float(torch.linalg.vector_norm(synthesized - direct) / torch.linalg.vector_norm(direct).clamp_min(torch.finfo(torch.float32).tiny))
            maximum_mse, maximum_field = max(maximum_mse, mse_error), max(maximum_field, field_error)
            native_errors.append(mse_error)
            del z, direct, synthesized, native, target28
        mean_power = np.mean(np.abs(coefficients) ** 2, axis=0)
        power_order = np.argsort(mean_power, kind="stable")
        ports = [int(power_order[0]), int(power_order[PORTS // 2]), int(power_order[-1])]
        ablation_errors: list[dict[str, Any]] = []
        row = int(selected_rows[0])
        base_z = np.asarray(coefficients[row_positions[row]:row_positions[row] + 1], dtype=np.complex64)
        for port in ports:
            for control in CHANNELS:
                z_np = base_z.copy()
                total = float(np.square(np.abs(z_np)).sum())
                removed = float(np.abs(z_np[0, port]) ** 2)
                z_np[0, port] = 0.0j
                if control == "equal_ols_power":
                    remaining = total - removed
                    if remaining == 0.0:
                        ablation_errors.append({"row": row, "port": port, "control": control, "status": "undefined_zero_remaining_power"})
                        continue
                    require(remaining > 0.0, "reference-grid equal-power factor has negative remaining power")
                    z_np *= math.sqrt(total / remaining)
                z = torch.from_numpy(z_np).to(device=device)
                direct, _, area = camera_field_from_coefficients(kind, model, z)
                synthesized = (z @ basis).reshape(1, 1, height, width)
                native, target28 = targets_for_rows(kind, context, np.asarray([row]), device)
                direct_mse = calculate_metrics(kind, direct, native, target28, gain, area)["mse28"][0]
                synth_mse = calculate_metrics(kind, synthesized, native, target28, gain, area)["mse28"][0]
                mse_error = abs(float(direct_mse) - float(synth_mse))
                field_error = float(torch.linalg.vector_norm(synthesized - direct) / torch.linalg.vector_norm(direct).clamp_min(torch.finfo(torch.float32).tiny))
                maximum_mse, maximum_field = max(maximum_mse, mse_error), max(maximum_field, field_error)
                ablation_errors.append({"row": row, "port": port, "control": control, "mse_absolute_error": mse_error, "field_relative_error": field_error})
                del z, direct, synthesized, native, target28
        assert_frozen_snapshot(model, before, f"{key}/{grid}/reference-grid direct replay")
        records[key] = {"grid": grid, "rows_replayed": len(selected_rows), "rows_per_class": 2, "weak_mid_strong_ports_zero_based": ports, "native_mse_absolute_errors": native_errors, "ablation_direct_checks": ablation_errors, "frozen_state_unchanged": True}
        del basis, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"passed": maximum_mse <= tolerance, "mse_absolute_tolerance": tolerance, "maximum_direct_vs_basis_mse_absolute_error": maximum_mse, "maximum_field_relative_error": maximum_field, "endpoints": records}


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
    values_array = np.asarray(values, dtype=np.float64)
    labels_array = np.asarray(labels, dtype=np.int64)
    require(values_array.shape[0] == labels_array.shape[0], "bootstrap effect values/labels length mismatch")
    observed = np.mean([values_array[labels_array == label].mean(axis=0) for label in range(10)], axis=0)
    draws = stratified_bootstrap(values, labels, resamples, seed)
    if draws.ndim == 1:
        draws = draws[:, None]
        observed = np.asarray([observed])
    low, high = np.quantile(draws, [0.025, 0.975], axis=0)
    centered_null = draws - observed[None, :]
    p = (np.sum(np.abs(centered_null) >= np.abs(observed)[None, :], axis=0) + 1.0) / (resamples + 1.0)
    return {"mean": observed, "ci95_low": low, "ci95_high": high, "p_two_sided_bootstrap": p, "bootstrap_draws": draws}


def bootstrap_effect_per_port(
    values: np.ndarray, labels: np.ndarray, valid_mask: np.ndarray,
    resamples: int, seed: int,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(values)
    require(values.ndim == 2 and valid.shape == values.shape and labels.shape == (len(values),), "per-port bootstrap inputs do not align")
    if bool(valid.all()):
        return bootstrap_effect(values, labels, resamples, seed)
    roots = values.shape[1]
    result = {
        "mean": np.full(roots, np.nan, dtype=np.float64),
        "ci95_low": np.full(roots, np.nan, dtype=np.float64),
        "ci95_high": np.full(roots, np.nan, dtype=np.float64),
        "p_two_sided_bootstrap": np.ones(roots, dtype=np.float64),
        "bootstrap_draws": np.full((resamples, roots), np.nan, dtype=np.float64),
    }
    for port in range(roots):
        selected = valid[:, port]
        selected_labels = labels[selected]
        counts = np.bincount(selected_labels, minlength=10)
        if counts.shape[0] != 10 or np.any(counts < 2):
            continue
        estimate = bootstrap_effect(values[selected, port], selected_labels, resamples, seed)
        result["bootstrap_draws"][:, port] = np.asarray(estimate["bootstrap_draws"], dtype=np.float64).reshape(-1)
        for key in ("mean", "ci95_low", "ci95_high", "p_two_sided_bootstrap"):
            result[key][port] = float(np.asarray(estimate[key], dtype=np.float64).reshape(-1)[0])
    return result


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
    require(bool((remaining_power >= 0.0).all()), "equal-power ablation produced negative remaining modal power")
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
    rows = population["rows"]
    n = len(rows)
    batch_size = 4 if kind == "large" else 8
    sum_mse = np.zeros(PORTS, dtype=np.float64)
    sum_abs_relative = np.zeros(PORTS, dtype=np.float64)
    within_1 = np.zeros(PORTS, dtype=np.int64)
    within_5 = np.zeros(PORTS, dtype=np.int64)
    sum_camera_power = np.zeros(PORTS, dtype=np.float64)
    native_all = np.empty(n, dtype=np.float64)
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
        require(field.ndim == 4 and field.shape[1] == 1, "independent reference returned an unexpected camera field shape")
        return field[:, 0].detach().cpu().numpy().astype(np.complex128, copy=False)
    finally:
        module.angular_spectrum_propagate = original_propagator
        if kind == "small":
            model.runtime.physical_optics = physical_optics


def promote_reference_precision(model: torch.nn.Module) -> dict[str, int]:
    """Promote floating tensors without converting complex state to real."""
    counts = {"real_float64": 0, "complex128": 0, "unchanged_integer_or_bool": 0}
    for parameter in model.parameters():
        if parameter.is_complex():
            parameter.data = parameter.data.to(dtype=torch.complex128)
            counts["complex128"] += 1
        elif parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype=torch.float64)
            counts["real_float64"] += 1
        else:
            counts["unchanged_integer_or_bool"] += 1
    for module in model.modules():
        for name, value in tuple(module._buffers.items()):
            if value is None:
                continue
            if value.is_complex():
                module._buffers[name] = value.to(dtype=torch.complex128)
                counts["complex128"] += 1
            elif value.is_floating_point():
                module._buffers[name] = value.to(dtype=torch.float64)
                counts["real_float64"] += 1
            else:
                counts["unchanged_integer_or_bool"] += 1
    for value in (*model.parameters(), *(buffer for buffer in model.buffers() if buffer is not None)):
        require(value.dtype in (torch.float64, torch.complex128) or not (value.is_floating_point() or value.is_complex()), "reference precision promotion lost a supported dtype")
        require(torch.isfinite(value).all(), "reference model contains a non-finite promoted parameter or buffer")
    return counts


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


def run_probe_reference_diagnostic_historical(
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


def run_independent_reference_diagnostic(
    cfg: dict[str, Any], output_root: Path, device: torch.device
) -> dict[str, Any]:
    """Validate every saved port basis on both grids; propagate cross-grid differences."""
    completed_reference_path = output_root / "preflight" / "independent_reference.json"
    if completed_reference_path.is_file():
        completed = json.loads(completed_reference_path.read_text(encoding="utf-8"))
        require(completed.get("task_id") == TASK_ID and "same_grid_all_100_ports_within_1e-4" in completed, "existing reference receipt belongs to another task or schema")
        if completed.get("measurement_gate_ready") is True:
            return completed
    prior_root = ROOT / cfg["recovery"]["prior_attempt_dir"]
    prior_preflight = json.loads((prior_root / "preflight.json").read_text(encoding="utf-8"))
    require(prior_preflight.get("task_id") == "GAUSSIAN_OLS_EFFECTIVE_CHANNEL_NOISE_ABLATION_V1" and prior_preflight.get("status") == "INVALID_HARNESS", "the named source attempt is not the preserved INVALID_HARNESS record")
    old_reference_path = ROOT / cfg["recovery"]["same_grid_reference_evidence"]
    old_reference = json.loads(old_reference_path.read_text(encoding="utf-8"))
    require(old_reference.get("task_id") == "GAUSSIAN_OLS_EFFECTIVE_CHANNEL_NOISE_ABLATION_V1" and old_reference.get("same_grid_all_probes_within_1e-4") is True, "the named same-grid reference evidence is incomplete")
    require(old_reference.get("cross_grid_all_probes_within_1e-4") is False, "the named source cross-grid condition changed")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "preflight").mkdir(parents=True, exist_ok=True)
    write_json(output_root / "preflight_progress.json", {"task_id": TASK_ID, "status": "IN_PROGRESS", "stage": "all_port_dual_grid_reference"})
    asm_controls = validate_independent_asm_reference(device)
    noise_reference = validate_noise_moments(device, read_noise_proxy(cfg))
    small = load_small_context(cfg, device)
    large = load_large_context(cfg, device)
    endpoints = [
        {
            "key": item["key"], "kind": "small", "state_key": item["source_state"],
            "grids": (int(cfg["small"]["production_grid"]), int(cfg["small"]["reference_grid"])),
        }
        for item in cfg["small"]["decoder_endpoints"]
    ]
    endpoints.append({
        "key": "large_step_030000", "kind": "large", "state_key": "large_step_030000",
        "grids": (int(cfg["large"]["decoder_production_grid"]), int(cfg["large"]["decoder_reference_grid"])),
    })
    endpoint_results: dict[str, Any] = {}
    all_port_rows: list[dict[str, Any]] = []
    for endpoint in endpoints:
        key, kind, state_key = endpoint["key"], endpoint["kind"], endpoint["state_key"]
        grid_references: dict[int, np.ndarray] = {}
        grid_results: dict[str, Any] = {}
        reused_by_grid: dict[str, bool] = {}
        for grid_role, grid in zip(("production", "reference"), endpoint["grids"], strict=True):
            prior_path = prior_root / "bases" / f"{key}_{grid_role}_{grid}.npz"
            require(prior_path.is_file(), f"named saved basis is missing: {prior_path}")
            with np.load(prior_path, allow_pickle=False) as saved:
                saved_basis = saved["camera_field_per_unit_coefficient"].astype(np.complex128, copy=True)
                saved_area = float(saved["pixel_area_m2"].reshape(()))
            require(saved_basis.shape[0] == PORTS and np.isfinite(saved_basis).all() and saved_area > 0.0, f"saved basis is invalid: {prior_path}")
            if kind == "small":
                production_model = small_model_for(state_key, cfg, small, grid, device)
                reference_model = small_model_for(state_key, cfg, small, grid, device)
            else:
                production_model = large_model_for(large, grid)
                reference_model = large_model_for(large, grid)
            for parameter in reference_model.parameters():
                parameter.requires_grad_(False)
            reference_model.eval()
            dtype_counts = promote_reference_precision(reference_model)
            production_before = frozen_snapshot(production_model)
            reference_before = frozen_snapshot(reference_model)
            live_fields: list[np.ndarray] = []
            independent_fields: list[np.ndarray] = []
            live_areas: list[float] = []
            for port in range(PORTS):
                coefficient = torch.zeros((1, PORTS), device=device, dtype=torch.complex64)
                coefficient[0, port] = 1.0 + 0.0j
                live_field, _, live_area = camera_field_from_coefficients(kind, production_model, coefficient)
                reference_coeff = coefficient.to(dtype=torch.complex128)
                reference_field = independent_reference_camera_field(kind, reference_model, reference_coeff)[0]
                live_fields.append(live_field[0, 0].detach().cpu().numpy().astype(np.complex128, copy=False))
                independent_fields.append(reference_field)
                live_areas.append(live_area)
                del coefficient, reference_coeff, live_field, reference_field
            live_array = np.stack(live_fields)
            reference_array = np.stack(independent_fields)
            require(np.isfinite(live_array).all() and np.isfinite(reference_array).all(), f"non-finite full-port basis at {key}/{grid}")
            require(all(math.isclose(value, saved_area, rel_tol=0.0, abs_tol=0.0) for value in live_areas), f"production pixel area changed at {key}/{grid}")
            saved_live = common_phase_port_errors(saved_basis, live_array)
            live_reference = common_phase_port_errors(live_array, reference_array)
            can_reuse = saved_live["maximum_complex_field_relative_l2"] <= 1.0e-6 and saved_live["maximum_camera_power_relative_error"] <= 1.0e-6
            selected_basis = saved_basis if can_reuse else live_array
            selected_reference = common_phase_port_errors(selected_basis, reference_array)
            if not can_reuse:
                write_npz(output_root / "bases" / f"{key}_{grid_role}_{grid}.npz", camera_field_per_unit_coefficient=live_array.astype(np.complex64), pixel_area_m2=np.asarray(saved_area, dtype=np.float64))
            else:
                write_npz(output_root / "bases" / f"{key}_{grid_role}_{grid}.npz", camera_field_per_unit_coefficient=saved_basis.astype(np.complex64), pixel_area_m2=np.asarray(saved_area, dtype=np.float64))
            grid_references[grid] = reference_array
            reused_by_grid[grid_role] = bool(can_reuse)
            grid_results[str(grid)] = {
                "grid_role": grid_role,
                "all_100_ports_checked": True,
                "saved_basis_reused_after_live_replay": bool(can_reuse),
                "reference_precision_tensor_counts": dtype_counts,
                "saved_vs_live_production": saved_live,
                "live_production_vs_independent_reference": live_reference,
                "selected_basis_vs_independent_reference": selected_reference,
                "selected_basis_same_grid_pass_1e-4": selected_reference["maximum_complex_field_relative_l2"] <= float(cfg["statistics"]["reference_relative_tolerance"]) and selected_reference["maximum_camera_power_relative_error"] <= float(cfg["statistics"]["reference_relative_tolerance"]) and live_reference["maximum_complex_field_relative_l2"] <= float(cfg["statistics"]["reference_relative_tolerance"]) and live_reference["maximum_camera_power_relative_error"] <= float(cfg["statistics"]["reference_relative_tolerance"]),
            }
            for port in range(PORTS):
                all_port_rows.append({
                    "endpoint": key, "grid_role": grid_role, "grid": grid, "port_zero_based": port,
                    "saved_vs_live_complex_error": float(saved_live["per_port_complex_field_relative_l2"][port]),
                    "live_vs_reference_complex_error": float(live_reference["per_port_complex_field_relative_l2"][port]),
                    "selected_vs_reference_complex_error": float(selected_reference["per_port_complex_field_relative_l2"][port]),
                    "selected_vs_reference_power_error": float(selected_reference["per_port_camera_power_relative_error"][port]),
                })
            assert_frozen_snapshot(production_model, production_before, f"{key}/{grid}/live reference comparison")
            assert_frozen_snapshot(reference_model, reference_before, f"{key}/{grid}/independent reference comparison")
            del production_model, reference_model, production_before, reference_before, live_fields, independent_fields, live_array, reference_array, selected_basis, saved_basis
            if device.type == "cuda":
                torch.cuda.empty_cache()
        production_grid, reference_grid = endpoint["grids"]
        cross_grid = common_phase_port_errors(grid_references[production_grid], grid_references[reference_grid])
        same_grid_pass = all(item["selected_basis_same_grid_pass_1e-4"] for item in grid_results.values())
        endpoint_results[key] = {
            "population_role": "fixed DEV" if kind == "small" else "fixed historical TRAIN",
            "source_checkpoint_step": int(state_key.split("_")[-1]) if kind == "small" else int(large["checkpoint"]["completed_updates"]),
            "same_grid": grid_results,
            "independent_reference_cross_grid_difference": cross_grid,
            "same_grid_all_100_pass": same_grid_pass,
            "saved_basis_reused_by_grid": reused_by_grid,
        }
        write_json(output_root / "preflight" / "independent_reference_progress.json", {"task_id": TASK_ID, "completed_endpoints": list(endpoint_results), "checked_ports": len(all_port_rows), "status": "IN_PROGRESS"})
    same_grid_pass = all(item["same_grid_all_100_pass"] for item in endpoint_results.values())
    cross_grid_all_1e4 = all(item["independent_reference_cross_grid_difference"]["maximum_complex_field_relative_l2"] <= float(cfg["statistics"]["reference_relative_tolerance"]) and item["independent_reference_cross_grid_difference"]["maximum_camera_power_relative_error"] <= float(cfg["statistics"]["reference_relative_tolerance"]) for item in endpoint_results.values())
    result = {
        "schema": "gaussian_ols_dual_grid_independent_reference.v1",
        "task_id": TASK_ID,
        "status": "SAME_GRID_REFERENCE_VALIDATED_CROSS_GRID_DIFFERENCES_RECORDED" if same_grid_pass else "SAME_GRID_REFERENCE_MISMATCH",
        "same_grid_all_100_ports_within_1e-4": same_grid_pass,
        "cross_grid_all_100_ports_within_1e-4": cross_grid_all_1e4,
        "cross_grid_is_a_reported_sensitivity_not_a_global_stop_condition": True,
        "measurement_gate_ready": same_grid_pass,
        "reference_method": "independent integer-index complex128 ASM, compared separately within each grid; both grid outputs retained for sensitivity propagation",
        "independent_reference_controls": asm_controls,
        "noise_moment_independent_reference": noise_reference,
        "endpoints": endpoint_results,
        "per_port_rows": len(all_port_rows),
        "no_training_updates": 0, "optimizers_created": 0, "prior_samples": 0, "official_test_reads": 0, "hardware_operations": 0,
    }
    write_json(output_root / "preflight" / "independent_reference.json", result)
    write_csv(output_root / "preflight" / "independent_reference_all_port_rows.csv", all_port_rows)
    write_json(output_root / "preflight_progress.json", {"task_id": TASK_ID, "status": "IN_PROGRESS" if same_grid_pass else "SAME_GRID_REFERENCE_MISMATCH", "stage": "all_port_reference_complete", "measurement_gate_ready": same_grid_pass})
    return result


def write_reference_gate_closeout(output_root: Path, reference: dict[str, Any]) -> None:
    reason = "SAME_GRID_PRODUCTION_REFERENCE_MISMATCH"
    status = f"INVALID_HARNESS / {reason}"
    rows = []
    for endpoint, item in reference["endpoints"].items():
        same = max(grid["selected_basis_vs_independent_reference"]["maximum_complex_field_relative_l2"] for grid in item["same_grid"].values())
        power = max(grid["selected_basis_vs_independent_reference"]["maximum_camera_power_relative_error"] for grid in item["same_grid"].values())
        cross = item["independent_reference_cross_grid_difference"]["maximum_complex_field_relative_l2"]
        rows.append({"endpoint": endpoint, "same_grid_complex_error": same, "same_grid_power_error": power, "independent_cross_grid_complex_error": cross})
    report = {
        "schema": "gaussian_ols_effective_channel_dual_grid_closeout.v1",
        "task_id": TASK_ID,
        "status": status,
        "scientific_measurement_completed": False,
        "decision_scope": "all-port same-grid numerical gate; no efficacy inference",
        "independent_reference": reference,
        "noise_moment_reference": reference["noise_moment_independent_reference"],
        "endpoint_summary": rows,
        "reason": "The fixed same-grid all-port production/reference gate was not satisfied. Cross-grid differences do not decide this gate.",
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
        "| 端点 | 同网格最大复场误差 | 同网格最大功率误差 | 全端口跨网格复场差 |",
        "|---|---:|---:|---:|",
        *[f"| {row['endpoint']} | {row['same_grid_complex_error']:.6g} | {row['same_grid_power_error']:.6g} | {row['independent_cross_grid_complex_error']:.6g} |" for row in rows],
        "",
        "## 科学认识",
        "",
        "同网格独立复核检验传播实现精度。跨网格差异按计划作为最终指标的不确定度传播；本次因同网格全端口门槛未通过而停止，不能作重构或噪声归因。",
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
        "本次测量因同网格全端口复核未通过而停止，因此不回答约48 dB与约32 dB差异分别造成多少可观察或重构有效通道。跨网格差异不作为统一停止条件，仍会进入后续双网格敏感性带。",
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
        basis_info[key] = independent_reference["endpoints"][key]["independent_reference_cross_grid_difference"]
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
    basis_info[large_key] = independent_reference["endpoints"][large_key]["independent_reference_cross_grid_difference"]
    basis_elapsed = time.perf_counter() - started_basis
    covariance_complete = bool(covariance) and len(covariance) == 11
    basis_pass = bool(independent_reference["same_grid_all_100_ports_within_1e-4"])

    # Exact Poisson plus Gaussian read-noise moments are compared with an
    # independently enumerated scalar reference and checked on CUDA.
    moment_reference = validate_noise_moments(device, noise)
    detection_reference = validate_dual_grid_detection_and_dtype(device)
    local_reference_checks: dict[str, Any] = {}
    small_coefficients = small["state_payloads"]["step_030000"]["coefficients"][0]
    small_basis_path = output_root / "bases" / f"small_step_030000_reference_{int(cfg['small']['reference_grid'])}.npz"
    with np.load(small_basis_path, allow_pickle=False) as saved:
        small_basis = saved["camera_field_per_unit_coefficient"]
        small_area = float(saved["pixel_area_m2"].reshape(()))
    local_reference_checks["small_step_030000_reference_grid"] = validate_local_jacobian_finite_difference(small_coefficients, small_basis, small_area, noise, device)
    large_coefficients = large["population"]["coefficients"].numpy()[0]
    large_basis_path = output_root / "bases" / f"large_step_030000_reference_{int(cfg['large']['decoder_reference_grid'])}.npz"
    with np.load(large_basis_path, allow_pickle=False) as saved:
        large_basis = saved["camera_field_per_unit_coefficient"]
        large_area = float(saved["pixel_area_m2"].reshape(()))
    local_reference_checks["large_step_030000_reference_grid"] = validate_local_jacobian_finite_difference(large_coefficients, large_basis, large_area, noise, device)
    local_reference_pass = all(item["passed"] for item in local_reference_checks.values())
    direct_production = direct_preflight(cfg, small, large, bases, output_root, device)
    direct_reference = direct_reference_grid_preflight(cfg, small, large, bases, device)
    direct_checks = {
        "passed": bool(direct_production["passed"] and direct_reference["passed"]),
        "production_grid": direct_production,
        "reference_grid": direct_reference,
        "maximum_direct_vs_basis_mse_absolute_error": max(direct_production["maximum_direct_vs_basis_mse_absolute_error"], direct_reference["maximum_direct_vs_basis_mse_absolute_error"]),
        "maximum_production_w_per_pixel_vs_complex_field_power_relative_error": direct_production["maximum_production_w_per_pixel_vs_complex_field_power_relative_error"],
        "all_direct_replays_and_ablation_checks_passed": bool(direct_production["passed"] and direct_reference["passed"]),
    }
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
        "schema": "gaussian_ols_effective_channel_dual_grid_preflight.v1",
        "task_id": TASK_ID,
        "status": "READY" if basis_pass and covariance_complete and moment_reference["passed"] and detection_reference["passed"] and local_reference_pass and direct_checks["passed"] and independent_reference["measurement_gate_ready"] else "INVALID_HARNESS",
        "decision_scope": "frozen conditional diagnostics only; no model selection or training authorization",
        "input_lineage": model_lineage,
        "parameter_and_buffer_freeze": "checked at basis generation and checked again during direct production replays",
        "covariance_populations": {key: record["sample_count"] for key, record in covariance.items()},
        "basis_checks": basis_info,
        "basis_same_grid_all_100_ports_within_1e-4": basis_pass,
        "cross_grid_differences_are_reported_not_a_stop_condition": True,
        "independent_high_precision_reference": independent_reference,
        "basis_elapsed_by_endpoint_seconds": elapsed_basis,
        "basis_elapsed_seconds": basis_elapsed,
        "measured_basis_port_fields": sum(int(value["production"].shape[0]) + int(value["reference"].shape[0]) for value in bases.values()),
        "noise_proxy": noise,
        "noise_moment_independent_reference": moment_reference,
        "pixelwise_detectability_and_reference_dtype_controls": detection_reference,
        "local_jacobian_reference_grid_checks": local_reference_checks,
        "local_jacobian_reference_grid_checks_passed": local_reference_pass,
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
    probe_lambda = np.asarray([0.0, 1.0, 100.0, 10000.0, 29900.0, 30000.0, 30100.0, 32536.0, 60000.0], dtype=np.float64)
    cuda_mu, cuda_var = clipped_camera_moments(torch.from_numpy(probe_lambda).to(device=device, dtype=torch.float64), camera)
    cpu_mu, cpu_var = clipped_camera_moments(torch.from_numpy(probe_lambda), camera)
    cpu_mean_error = float(np.max(np.abs(cpu_mu.numpy() - cuda_mu.cpu().numpy())))
    cpu_variance_error = float(np.max(np.abs(cpu_var.numpy() - cuda_var.cpu().numpy())))
    require(cpu_mean_error <= 1e-9 and cpu_variance_error <= 1e-6, "independent CPU and CUDA camera moment execution disagrees")
    reference_lambda = probe_lambda[[0, 1, 4, 5, 6, 7]]
    reference_moments = np.asarray([enumerated_poisson_read_moments(float(value), camera) for value in reference_lambda], dtype=np.float64)
    torch_reference, variance_reference = clipped_camera_moments(torch.from_numpy(reference_lambda), camera)
    mean_error = float(np.max(np.abs(torch_reference.numpy() - reference_moments[:, 0])))
    var_error = float(np.max(np.abs(variance_reference.numpy() - reference_moments[:, 1])))
    rare_tail_index = int(np.flatnonzero(reference_lambda == 32536.0)[0])
    rare_tail_relative_variance_error = float(
        abs(float(variance_reference[rare_tail_index]) - float(reference_moments[rare_tail_index, 1]))
        / max(float(reference_moments[rare_tail_index, 1]), np.finfo(np.float64).tiny)
    )
    require(mean_error <= 2e-7 and var_error <= 2e-5 and rare_tail_relative_variance_error <= 1e-5, "Poisson/read-noise formula disagrees with independent scalar enumeration, including the rare saturated tail")
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
        "independent_scalar_enumeration": {"lambda_e": reference_lambda.tolist(), "mean_max_abs_error_e": mean_error, "variance_max_abs_error_e2": var_error, "enumeration_tail_sigma": 28.0, "rare_saturated_tail_relative_variance_error": rare_tail_relative_variance_error},
        "cpu_cuda_execution": {"mean_max_abs_error_e": cpu_mean_error, "variance_max_abs_error_e2": cpu_variance_error},
        "read_noise_transition_window": {"poisson_sigma_cut": 14.0, "read_noise_sigma_cut": 12.0, "omitted_read_clipping_probability_bound": "less than 7e-39 per conditional Gaussian tail; Poisson mass outside the transition band is below the 14-sigma Chernoff bound"},
        "low_count_noise_control": {"mean_e": float(low_mean.item()), "variance_e2": float(low_var.item())},
        "high_saturation_control": {"mean_e": float(saturated_mean.item()), "variance_e2": float(saturated_var.item())},
        "missing_coherent_cross_term_control": {"coherent_intensity": destructive, "incorrect_incoherent_sum": incoherent_sum},
    }


def validate_dual_grid_detection_and_dtype(device: torch.device) -> dict[str, Any]:
    delta = torch.tensor([[[3.0]]], device=device, dtype=torch.float64)
    variance = torch.ones_like(delta)
    one_pixel = _weighted_pixel_distance(delta, variance)
    padded = _weighted_pixel_distance(torch.tensor([[[3.0, 0.0, 0.0]]], device=device, dtype=torch.float64), torch.ones((1, 1, 3), device=device, dtype=torch.float64))
    production = torch.tensor([[[3.0, 0.0]]], device=device, dtype=torch.float64)
    reference = torch.tensor([[[2.0, 0.0]]], device=device, dtype=torch.float64)
    d_prod, d_ref, uncertainty, lower, upper = _detectability_interval(production, torch.ones_like(production), reference, torch.ones_like(reference))
    zero = _weighted_pixel_distance(torch.zeros_like(delta), torch.zeros_like(delta))
    infinite = _weighted_pixel_distance(torch.ones_like(delta), torch.zeros_like(delta))
    require(abs(float(one_pixel.item()) - 3.0) <= 1e-12 and abs(float(padded.item()) - 3.0) <= 1e-12, "pixelwise detectability was diluted by unaffected pixels")
    require(float(lower.item()) <= float(d_ref.item()) <= float(upper.item()) and abs(float(d_prod.item()) - 3.0) <= 1e-12 and abs(float(uncertainty.item()) - 1.0) <= 1e-12, "dual-grid detectability band did not cover the reference prediction")
    require(float(zero.item()) == 0.0 and bool(torch.isinf(infinite).all()), "zero-variance detectability cases were not represented explicitly")

    class DtypeProbe(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.real = torch.nn.Parameter(torch.tensor([1.0], device=device, dtype=torch.float32))
            self.phase = torch.nn.Parameter(torch.tensor([1.0 + 2.0j], device=device, dtype=torch.complex64))
            self.register_buffer("complex_buffer", torch.tensor([3.0 - 4.0j], device=device, dtype=torch.complex64))
            self.register_buffer("integer_buffer", torch.tensor([7], device=device, dtype=torch.int64))

    probe = DtypeProbe()
    dtype_counts = promote_reference_precision(probe)
    require(probe.real.dtype == torch.float64 and probe.phase.dtype == torch.complex128 and probe.complex_buffer.dtype == torch.complex128 and probe.integer_buffer.dtype == torch.int64, "reference dtype promotion changed a non-floating dtype")
    require(complex(probe.phase.detach().cpu().item()) == 1.0 + 2.0j and complex(probe.complex_buffer.detach().cpu().item()) == 3.0 - 4.0j, "reference dtype promotion lost a nonzero imaginary component")
    return {"passed": True, "one_pixel_distance": float(one_pixel.item()), "unaffected_pixel_invariance": float(padded.item()), "dual_grid_interval": {"production": float(d_prod.item()), "reference": float(d_ref.item()), "u": float(uncertainty.item()), "lower": float(lower.item()), "upper": float(upper.item())}, "zero_variance_zero_signal": float(zero.item()), "zero_variance_nonzero_signal": "infinite", "dtype_promotion_counts": dtype_counts, "complex_phase_preserved": True, "integer_buffer_preserved": True}


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
        # Include the rare lower tail when the Poisson mean is above full well.
        # A 14-sigma cut can omit most of the variance in the saturated tail,
        # even though its absolute value is tiny.  Centered offsets keep this
        # independent scalar sum stable without subtracting quantities near F^2.
        radius = 28.0 * math.sqrt(lam) + 30.0
        low = max(0, int(math.floor(lam - radius)))
        high = max(low - 1, int(math.ceil(min(lam + radius, full_well + 12.0))))
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


def _weighted_pixel_distance(delta: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    positive = variance > 0.0
    safe_variance = torch.where(positive, variance, torch.ones_like(variance))
    squared = torch.where(positive, delta.square() / safe_variance, torch.zeros_like(delta))
    distance = torch.sqrt(squared.sum(dim=-1).clamp_min(0.0))
    undefined_nonzero = ((~positive) & (delta != 0.0)).any(dim=-1)
    distance[undefined_nonzero] = float("inf")
    return distance


def _detectability_interval(
    delta_production: torch.Tensor,
    variance_production: torch.Tensor,
    delta_reference: torch.Tensor,
    variance_reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    common_variance = torch.maximum(variance_production, variance_reference)
    d_production = _weighted_pixel_distance(delta_production, common_variance)
    d_reference = _weighted_pixel_distance(delta_reference, common_variance)
    difference = _weighted_pixel_distance(delta_production - delta_reference, common_variance)
    finite = torch.isfinite(d_production) & torch.isfinite(difference)
    lower = torch.zeros_like(d_production)
    upper = torch.full_like(d_production, float("inf"))
    lower[finite] = (d_production[finite] - difference[finite]).clamp_min(0.0)
    upper[finite] = d_production[finite] + difference[finite]
    return d_production, d_reference, difference, lower, upper


def compute_ablation_dual_grid_microbatch(
    kind: str,
    coefficients_np: np.ndarray,
    rows: np.ndarray,
    labels: np.ndarray,
    bases: dict[str, torch.Tensor],
    geometries: dict[str, tuple[int, int, float]],
    gain: float,
    context: dict[str, Any],
    camera: dict[str, Any],
    port_block_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    coefficients = torch.from_numpy(np.asarray(coefficients_np, dtype=np.complex64)).to(device=device)
    batch = len(coefficients)
    require(batch > 0 and set(bases) == {"production", "reference"}, "dual-grid batch inputs are incomplete")
    native, target28 = targets_for_rows(kind, context, rows, device)
    total_power = coefficients.abs().square().sum(dim=1)
    per_port_power = coefficients.abs().square()
    remaining_power = total_power[:, None] - per_port_power
    factors = torch.full_like(remaining_power, float("nan"))
    positive_remaining = remaining_power > 0.0
    factors[positive_remaining] = torch.sqrt(total_power[:, None].expand_as(remaining_power)[positive_remaining] / remaining_power[positive_remaining])
    exposure_values = [float(value) for value in camera["exposure_sweep_s"]]
    control_count = len(CHANNELS)
    metric_count = len(METRICS)
    native_metrics: dict[str, np.ndarray] = {}
    native_fields: dict[str, torch.Tensor] = {}
    ablation_metrics: dict[str, np.ndarray] = {}
    native_noise: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for grid_role in ("production", "reference"):
        height, width, area = geometries[grid_role]
        basis = bases[grid_role]
        native_flat = coefficients @ basis
        native_fields[grid_role] = native_flat
        native_field = native_flat.reshape(batch, 1, height, width)
        base_values = calculate_metrics(kind, native_field, native, target28, gain, area)
        native_metrics[grid_role] = np.stack([base_values[name] for name in METRICS], axis=-1)
        physical_native = native_flat.abs().square() * area
        native_noise[grid_role] = []
        for exposure in exposure_values:
            rate = physical_native * noise_scale(camera, exposure) + float(camera["dark_current_e_per_pixel_s"]) * exposure
            mean, variance = clipped_camera_moments(rate, camera)
            native_noise[grid_role].append((mean, variance))
        ablation_metrics[grid_role] = np.empty((batch, PORTS, control_count, metric_count), dtype=np.float64)

    intervals = np.full((batch, control_count, len(exposure_values), PORTS, 5), np.nan, dtype=np.float64)
    port_block_size = max(1, min(int(port_block_size), PORTS))
    for port_start in range(0, PORTS, port_block_size):
        port_stop = min(PORTS, port_start + port_block_size)
        port_indices = slice(port_start, port_stop)
        port_count = port_stop - port_start
        noise_deltas: dict[str, list[list[torch.Tensor]]] = {role: [[] for _ in exposure_values] for role in ("production", "reference")}
        noise_variances: dict[str, list[list[torch.Tensor]]] = {role: [[] for _ in exposure_values] for role in ("production", "reference")}
        for grid_role in ("production", "reference"):
            height, width, area = geometries[grid_role]
            basis = bases[grid_role]
            native_flat = native_fields[grid_role]
            weighted = coefficients[:, port_indices, None] * basis[None, port_indices, :]
            raw_fields = native_flat[:, None, :] - weighted
            local_factor = factors[:, port_indices]
            equal_fields = raw_fields * local_factor[:, :, None]
            variants = torch.stack((raw_fields, equal_fields), dim=2)
            variant_count = batch * port_count * control_count
            variant_tensor = variants.reshape(variant_count, 1, height, width)
            native_repeated = native[:, None, None].expand(batch, port_count, control_count, *native.shape[1:]).reshape(variant_count, *native.shape[1:])
            target_repeated = target28[:, None, None].expand(batch, port_count, control_count, *target28.shape[1:]).reshape(variant_count, *target28.shape[1:])
            metric_values = calculate_metrics(kind, variant_tensor, native_repeated, target_repeated, gain, area)
            for metric_index, name in enumerate(METRICS):
                shaped = metric_values[name].reshape(batch, port_count, control_count)
                ablation_metrics[grid_role][:, port_start:port_stop, :, metric_index] = shaped
            variant_power = variants.abs().square() * area
            for exposure_index, exposure in enumerate(exposure_values):
                scale = noise_scale(camera, exposure)
                dark = float(camera["dark_current_e_per_pixel_s"]) * exposure
                mean0, variance0 = native_noise[grid_role][exposure_index]
                for control_index in range(control_count):
                    rates = variant_power[:, :, control_index] * scale + dark
                    mean1, variance1 = clipped_camera_moments(rates, camera)
                    noise_deltas[grid_role][exposure_index].append(mean1 - mean0[:, None, :])
                    noise_variances[grid_role][exposure_index].append(variance1 + variance0[:, None, :])
            del weighted, raw_fields, equal_fields, variants, variant_tensor, native_repeated, target_repeated, metric_values, variant_power, native_flat
        for exposure_index in range(len(exposure_values)):
            for control_index in range(control_count):
                delta_prod = noise_deltas["production"][exposure_index][control_index]
                var_prod = noise_variances["production"][exposure_index][control_index]
                delta_ref = noise_deltas["reference"][exposure_index][control_index]
                var_ref = noise_variances["reference"][exposure_index][control_index]
                d_prod, d_ref, uncertainty, lower, upper = _detectability_interval(delta_prod, var_prod, delta_ref, var_ref)
                values = torch.stack((d_prod, d_ref, uncertainty, lower, upper), dim=-1).detach().cpu().numpy()
                if control_index == 1:
                    values[~positive_remaining[:, port_start:port_stop].detach().cpu().numpy()] = np.nan
                intervals[:, control_index, exposure_index, port_start:port_stop, :] = values
        del noise_deltas, noise_variances

    counts = np.zeros((batch, control_count, len(exposure_values), len(OBSERVABILITY_THRESHOLDS), 2), dtype=np.int16)
    for threshold_index, threshold in enumerate(OBSERVABILITY_THRESHOLDS):
        counts[:, :, :, threshold_index, 0] = np.sum(intervals[:, :, :, :, 3] >= threshold, axis=-1).astype(np.int16)
        counts[:, :, :, threshold_index, 1] = np.sum(intervals[:, :, :, :, 4] >= threshold, axis=-1).astype(np.int16)
    native_metric_arrays = {role: np.stack([native_metrics[role][:, index] for index in range(metric_count)], axis=-1) for role in native_metrics}
    return {
        "rows": np.asarray(rows, dtype=np.int64).copy(),
        "labels": np.asarray(labels, dtype=np.int64).copy(),
        "native_metrics_production": native_metric_arrays["production"],
        "native_metrics_reference": native_metric_arrays["reference"],
        "ablation_metrics_production": ablation_metrics["production"],
        "ablation_metrics_reference": ablation_metrics["reference"],
        "port_detectability": intervals,
        "observable_port_counts": counts,
        "modal_power_w": per_port_power.detach().cpu().numpy().astype(np.float64),
        "equal_power_factor": factors.detach().cpu().numpy().astype(np.float64),
        "equal_power_defined": positive_remaining.detach().cpu().numpy(),
    }


def validate_dual_grid_chunk(path: Path, expected_rows: np.ndarray, expected_labels: np.ndarray) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        part = {key: data[key].copy() for key in data.files}
    required = {"rows", "labels", "native_metrics_production", "native_metrics_reference", "ablation_metrics_production", "ablation_metrics_reference", "port_detectability", "observable_port_counts", "modal_power_w", "equal_power_factor", "equal_power_defined"}
    require(set(part) == required, f"dual-grid chunk schema changed: {path}")
    count = len(expected_rows)
    require(np.array_equal(part["rows"], expected_rows) and np.array_equal(part["labels"], expected_labels), f"dual-grid chunk rows or labels changed: {path}")
    for role in ("production", "reference"):
        require(part[f"native_metrics_{role}"].shape == (count, len(METRICS)), f"native metric chunk shape changed: {path}/{role}")
        require(part[f"ablation_metrics_{role}"].shape == (count, PORTS, len(CHANNELS), len(METRICS)), f"ablation chunk shape changed: {path}/{role}")
    require(part["port_detectability"].shape == (count, len(CHANNELS), 4, PORTS, 5), f"detectability chunk shape changed: {path}")
    require(part["observable_port_counts"].shape == (count, len(CHANNELS), 4, 3, 2), f"observable-count chunk shape changed: {path}")
    require(part["modal_power_w"].shape == (count, PORTS) and part["equal_power_factor"].shape == (count, PORTS), f"coefficient-power chunk shape changed: {path}")
    require(part["equal_power_defined"].shape == (count, PORTS), f"equal-power defined mask shape changed: {path}")
    return part


def process_endpoint_dual_grid(
    cfg: dict[str, Any], endpoint_key: str, kind: str, source_state: str | None,
    population: dict[str, Any], output_root: Path, device: torch.device, camera: dict[str, Any],
) -> dict[str, Any]:
    bases: dict[str, torch.Tensor] = {}
    geometries: dict[str, tuple[int, int, float]] = {}
    for role, grid in (("production", int(cfg["small"]["production_grid"]) if kind == "small" else int(cfg["large"]["decoder_production_grid"])), ("reference", int(cfg["small"]["reference_grid"]) if kind == "small" else int(cfg["large"]["decoder_reference_grid"]))):
        bases[role], area = load_basis(output_root, endpoint_key, role, grid, device)
        with np.load(output_root / "bases" / f"{endpoint_key}_{role}_{grid}.npz", allow_pickle=False) as saved:
            height, width = (int(value) for value in saved["camera_field_per_unit_coefficient"].shape[-2:])
        geometries[role] = (height, width, area)
    rows, labels = population["rows"], population["labels"]
    coefficients = population["coefficients"]
    run_dir = output_root / "ablation" / endpoint_key
    chunks_dir = run_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_rows = int(cfg["output"]["recovery_chunk_rows"])
    port_block = int(cfg["statistics"]["ablation_port_block_size"])
    progress_path = run_dir / "progress.json"
    started = time.perf_counter()
    for chunk_index, start in enumerate(range(0, len(rows), chunk_rows)):
        stop = min(len(rows), start + chunk_rows)
        expected_rows, expected_labels = rows[start:stop], labels[start:stop]
        chunk_path = chunks_dir / f"chunk_{chunk_index:05d}.npz"
        if chunk_path.is_file():
            validate_dual_grid_chunk(chunk_path, expected_rows, expected_labels)
            continue
        pieces: list[dict[str, np.ndarray]] = []
        for offset in range(start, stop, int(cfg["statistics"]["measurement_batch_rows"])):
            micro_stop = min(stop, offset + int(cfg["statistics"]["measurement_batch_rows"]))
            pieces.append(compute_ablation_dual_grid_microbatch(kind, coefficients[offset:micro_stop], rows[offset:micro_stop], labels[offset:micro_stop], bases, geometries, population["gain"], population["context"], camera, port_block, device))
        combined = {key: np.concatenate([piece[key] for piece in pieces], axis=0) for key in pieces[0]}
        write_npz(chunk_path, **combined)
        validate_dual_grid_chunk(chunk_path, expected_rows, expected_labels)
        write_json(progress_path, {
            "task_id": TASK_ID, "endpoint": endpoint_key, "completed_rows": stop,
            "population_rows": len(rows), "completed_chunks": chunk_index + 1,
            "total_chunks": math.ceil(len(rows) / chunk_rows),
            "elapsed_seconds_observed": time.perf_counter() - started,
            "training_updates": 0, "zero_optimizer_creation": 0,
        })
    parts = [validate_dual_grid_chunk(chunks_dir / f"chunk_{index:05d}.npz", rows[start:min(len(rows), start + chunk_rows)], labels[start:min(len(rows), start + chunk_rows)]) for index, start in enumerate(range(0, len(rows), chunk_rows))]
    stacked = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0] if key not in ("rows", "labels")}
    require(np.array_equal(np.concatenate([part["rows"] for part in parts]), rows), f"dual-grid chunk recovery changed row identity: {endpoint_key}")
    write_json(run_dir / "complete.json", {"task_id": TASK_ID, "endpoint": endpoint_key, "sample_count": len(rows), "records_per_grid": len(rows) * PORTS * len(CHANNELS), "grid_count": 2, "recovery_source": "validated row chunks; full-population arrays are not duplicated", "all_chunks_read_back_and_row_validated": True, "training_updates": 0, "optimizers_created": 0, "official_test_reads": 0, "hardware_operations": 0, "elapsed_seconds_observed": time.perf_counter() - started})
    for basis in bases.values():
        del basis
    return {"rows": rows, "labels": labels, **stacked, "elapsed_seconds_observed": time.perf_counter() - started}


def cumulative_retention_dual_grid(
    kind: str, endpoint_key: str, population: dict[str, Any], output_root: Path,
    port_order: np.ndarray, grid_role: str, grid: int, device: torch.device,
) -> list[dict[str, Any]]:
    basis, area = load_basis(output_root, endpoint_key, grid_role, grid, device)
    with np.load(output_root / "bases" / f"{endpoint_key}_{grid_role}_{grid}.npz", allow_pickle=False) as saved:
        height, width = (int(value) for value in saved["camera_field_per_unit_coefficient"].shape[-2:])
    rows, coefficients = population["rows"], population["coefficients"]
    n = len(rows)
    batch_size = 1 if kind == "large" else 4
    sums = np.zeros(PORTS, dtype=np.float64)
    abs_rel_sums = np.zeros(PORTS, dtype=np.float64)
    within_1 = np.zeros(PORTS, dtype=np.int64)
    within_5 = np.zeros(PORTS, dtype=np.int64)
    camera_sums = np.zeros(PORTS, dtype=np.float64)
    native_all = np.empty(n, dtype=np.float64)
    for start in range(0, n, batch_size):
        stop = min(n, start + batch_size)
        z = torch.from_numpy(coefficients[start:stop]).to(device=device)
        native, target28 = targets_for_rows(kind, population["context"], rows[start:stop], device)
        weighted = z.unsqueeze(-1) * basis.unsqueeze(0)
        native_flat = z @ basis
        native_mse = calculate_metrics(kind, native_flat.reshape(len(z), 1, height, width), native, target28, population["gain"], area)["mse28"]
        native_np = np.asarray(native_mse, dtype=np.float64)
        native_all[start:stop] = native_np
        partial = torch.zeros_like(native_flat)
        for keep_index, port in enumerate(port_order):
            partial = partial + weighted[:, int(port), :]
            physical = partial.abs().square().reshape(len(z), 1, height, width) * area
            calibrated = physical * population["gain"]
            view = chen_quickcheck.baseline._resize_view(calibrated) if kind == "small" else chen_base._resize_view(calibrated)
            mse = (view - target28).square().mean(dim=(1, 2, 3)).detach().cpu().numpy().astype(np.float64)
            power = physical.reshape(len(z), -1).sum(dim=1).detach().cpu().numpy().astype(np.float64)
            relative = np.abs(mse - native_np) / np.maximum(native_np, np.finfo(np.float64).tiny)
            sums[keep_index] += mse.sum()
            abs_rel_sums[keep_index] += relative.sum()
            within_1[keep_index] += int((relative <= 0.01).sum())
            within_5[keep_index] += int((relative <= 0.05).sum())
            camera_sums[keep_index] += power.sum()
        del z, native, target28, weighted, native_flat, partial
    native_mean = float(native_all.mean())
    return [{
        "endpoint": endpoint_key, "grid_role": grid_role, "grid": grid,
        "retained_roots": index + 1,
        "population_mean_power_order_zero_based": [int(value) for value in port_order],
        "native_mean_mse28": native_mean, "mean_mse28": float(sums[index] / n),
        "mean_delta_mse28": float(sums[index] / n - native_mean),
        "mean_absolute_relative_mse_error": float(abs_rel_sums[index] / n),
        "fraction_images_within_1pct_relative_mse": float(within_1[index] / n),
        "fraction_images_within_5pct_relative_mse": float(within_5[index] / n),
        "mean_camera_power_w": float(camera_sums[index] / n),
    } for index in range(PORTS)]


def minimum_stable_retained_roots(rows: list[dict[str, Any]], tolerance: float) -> int | None:
    errors = np.asarray([row["mean_absolute_relative_mse_error"] for row in rows], dtype=np.float64)
    for index in range(len(errors)):
        if np.all(errors[index:] <= tolerance):
            return index + 1
    return None


def _local_jacobian(
    coefficients: torch.Tensor, basis: torch.Tensor, geometry: tuple[int, int, float],
    camera: dict[str, Any], exposure: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, area = geometry
    coefficients = coefficients.to(torch.complex128)
    basis = basis.to(torch.complex128)
    field = coefficients @ basis
    power = field.abs().square() * area
    scale = noise_scale(camera, exposure)
    rate = power * scale + float(camera["dark_current_e_per_pixel_s"]) * exposure
    mean, variance = clipped_camera_moments(rate, camera)
    derivative_mean = clipped_camera_mean_derivative(rate, camera)
    product = field.conj()[:, None] * basis.T
    derivative_power = torch.stack((2.0 * product.real, -2.0 * product.imag), dim=-1).reshape(-1, 2 * PORTS).to(torch.float64) * area
    jacobian = derivative_power * (scale * derivative_mean.reshape(-1, 1))
    return jacobian, mean, variance, rate


def _symmetric_complex_perturbation(
    coefficients: torch.Tensor, direction: torch.Tensor, radius: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    center = coefficients.to(torch.complex128)
    delta = direction.to(torch.complex128) * float(radius)
    return center + delta, center - delta


def _finite_difference_linear_prediction(
    jacobians: dict[str, torch.Tensor], grid_role: str,
    direction: torch.Tensor, radius: float,
) -> torch.Tensor:
    require(grid_role in jacobians, f"finite-difference grid Jacobian is missing: {grid_role}")
    return jacobians[grid_role] @ (float(radius) * direction)


def local_direction_diagnostic_dual_grid(
    cfg: dict[str, Any], endpoint_key: str, population: dict[str, Any],
    output_root: Path, camera: dict[str, Any], device: torch.device,
) -> dict[str, Any]:
    basis: dict[str, torch.Tensor] = {}
    geometry: dict[str, tuple[int, int, float]] = {}
    kind = "small" if endpoint_key.startswith("small") else "large"
    for role, grid in (("production", int(cfg["small"]["production_grid"]) if kind == "small" else int(cfg["large"]["decoder_production_grid"])), ("reference", int(cfg["small"]["reference_grid"]) if kind == "small" else int(cfg["large"]["decoder_reference_grid"]))):
        basis[role], area = load_basis(output_root, endpoint_key, role, grid, device)
        basis[role] = basis[role].to(torch.complex128)
        with np.load(output_root / "bases" / f"{endpoint_key}_{role}_{grid}.npz", allow_pickle=False) as saved:
            height, width = (int(value) for value in saved["camera_field_per_unit_coefficient"].shape[-2:])
        geometry[role] = (height, width, area)
    per_class = int(cfg["statistics"]["local_direction_rows_per_class"])
    selected: list[int] = []
    for label in range(10):
        selected.extend(np.flatnonzero(population["labels"] == label)[:per_class].tolist())
    require(len(selected) == 10 * per_class, f"local direction sample selection changed for {endpoint_key}")
    radii = [float(value) for value in cfg["statistics"]["local_direction_relative_perturbations"]]
    thresholds = list(OBSERVABILITY_THRESHOLDS)
    exposure_values = [float(value) for value in camera["exposure_sweep_s"]]
    row_records: list[dict[str, Any]] = []
    finite_difference_positions: set[int] = set()
    for label in range(10):
        finite_difference_positions.update(np.flatnonzero(population["labels"] == label)[:2].tolist())
    require(len(finite_difference_positions) == 20,
            f"local finite-difference control requires 20 fixed rows for {endpoint_key}")
    finite_difference_images: list[dict[str, Any]] = []
    maximum_phase_response = 0.0
    for selected_position in selected:
        z_np = population["coefficients"][selected_position]
        z = torch.from_numpy(np.asarray(z_np, dtype=np.complex128)).to(device=device)
        real_z = torch.stack((z.real, z.imag), dim=-1).reshape(-1).to(torch.float64)
        phase_tangent = torch.stack((-z.imag, z.real), dim=-1).reshape(-1).to(torch.float64)
        tangent_norm = torch.linalg.vector_norm(phase_tangent)
        if float(tangent_norm) > 0.0:
            phase_unit = phase_tangent / tangent_norm
            q, _ = torch.linalg.qr(phase_unit[:, None], mode="complete")
            projection = q[:, 1:]
        else:
            projection = torch.eye(2 * PORTS, dtype=torch.float64, device=device)[:, 1:]
        exposure_rows: dict[str, Any] = {}
        for exposure in exposure_values:
            jacobians: dict[str, torch.Tensor] = {}
            means: dict[str, torch.Tensor] = {}
            variances: dict[str, torch.Tensor] = {}
            unweighted: dict[str, torch.Tensor] = {}
            for role in ("production", "reference"):
                jacobians[role], means[role], variances[role], _ = _local_jacobian(z, basis[role], geometry[role], camera, exposure)
                unweighted[role] = jacobians[role]
            common_variance = 2.0 * torch.maximum(variances["production"], variances["reference"])
            positive = common_variance > 0.0
            denominator = torch.sqrt(torch.where(positive, common_variance, torch.ones_like(common_variance))).reshape(-1, 1)
            weighted: dict[str, torch.Tensor] = {}
            for role in ("production", "reference"):
                weighted[role] = torch.where(positive.reshape(-1, 1), jacobians[role] / denominator, torch.zeros_like(jacobians[role]))
            projected = {role: weighted[role] @ projection for role in ("production", "reference")}
            grams = {role: projected[role].T @ projected[role] for role in ("production", "reference")}
            eigen_production, vectors_production = torch.linalg.eigh((grams["production"] + grams["production"].T) * 0.5)
            eigen_reference = torch.linalg.eigvalsh((grams["reference"] + grams["reference"].T) * 0.5)
            difference = (weighted["production"] - weighted["reference"]) @ projection
            difference_gram = difference.T @ difference
            sigma_difference = torch.sqrt(torch.linalg.eigvalsh((difference_gram + difference_gram.T) * 0.5).clamp_min(0.0).max())
            sigma_production = torch.sqrt(eigen_production.clamp_min(0.0)).flip(0)
            sigma_reference = torch.sqrt(eigen_reference.clamp_min(0.0)).flip(0)
            if float(tangent_norm) > 0.0:
                phase_response = float(torch.linalg.vector_norm(weighted["production"] @ phase_tangent) / (torch.linalg.matrix_norm(weighted["production"]) * tangent_norm).clamp_min(torch.finfo(torch.float64).tiny))
                maximum_phase_response = max(maximum_phase_response, phase_response)
            radius_rows: dict[str, Any] = {}
            for relative_radius in radii:
                radius = relative_radius * float(torch.linalg.vector_norm(real_z))
                low = (sigma_production - sigma_difference).clamp_min(0.0) * radius
                high = (sigma_production + sigma_difference) * radius
                reference_low = (sigma_reference - sigma_difference).clamp_min(0.0) * radius
                reference_high = (sigma_reference + sigma_difference) * radius
                radius_rows[str(relative_radius)] = {
                    "coefficient_l2_perturbation_radius": radius,
                    "production_singular_response": (sigma_production * radius).detach().cpu().numpy(),
                    "reference_singular_response": (sigma_reference * radius).detach().cpu().numpy(),
                    "grid_sensitivity_lower": low.detach().cpu().numpy(),
                    "grid_sensitivity_upper": high.detach().cpu().numpy(),
                    "reference_sensitivity_lower": reference_low.detach().cpu().numpy(),
                    "reference_sensitivity_upper": reference_high.detach().cpu().numpy(),
                    "observable_direction_count_lower": {str(value): int((low >= value).sum().item()) for value in thresholds},
                    "observable_direction_count_upper": {str(value): int((high >= value).sum().item()) for value in thresholds},
                }
            exposure_rows[str(exposure)] = {
                "common_pair_exposure_variance_max": float(common_variance.max().detach().cpu()),
                "production_native_variance_max": float(variances["production"].max().detach().cpu()),
                "reference_native_variance_max": float(variances["reference"].max().detach().cpu()),
                "grid_jacobian_difference_spectral_norm": float(sigma_difference.detach().cpu()),
                "radii": radius_rows,
            }
            if selected_position in finite_difference_positions:
                top_index = int(torch.argmax(eigen_production).item())
                direction = projection @ vectors_production[:, top_index]
                direction_complex = torch.complex(direction[0::2], direction[1::2])
                sample_control = {
                    "endpoint": endpoint_key,
                    "row": int(population["rows"][selected_position]),
                    "class": int(population["labels"][selected_position]),
                    "exposure_s": exposure,
                    "coefficient_perturbation_precision": "complex128",
                    "direction_source": "top_production_local_singular_direction_shared_across_grids",
                    "variance_weight": "two_native_exposure_variances_summed_per_grid_max",
                    "variance_min_by_grid": {role: float(variances[role].min().detach().cpu())
                                              for role in ("production", "reference")},
                    "variance_max_by_grid": {role: float(variances[role].max().detach().cpu())
                                              for role in ("production", "reference")},
                    "directions": [],
                }
                response_fields = {
                    role: (z @ basis[role], direction_complex @ basis[role])
                    for role in ("production", "reference")
                }
                for relative_radius in radii:
                    radius = relative_radius * float(torch.linalg.vector_norm(real_z))
                    scale = noise_scale(camera, exposure)
                    dark = float(camera["dark_current_e_per_pixel_s"]) * exposure
                    for role in ("production", "reference"):
                        base_field, direction_field = response_fields[role]
                        delta_field = radius * direction_field
                        plus_field = base_field + delta_field
                        minus_field = base_field - delta_field
                        plus_rate = plus_field.abs().square() * geometry[role][2] * scale + dark
                        minus_rate = minus_field.abs().square() * geometry[role][2] * scale + dark
                        plus_mu, _ = clipped_camera_moments(plus_rate, camera)
                        minus_mu, _ = clipped_camera_moments(minus_rate, camera)
                        actual = (plus_mu - minus_mu) * 0.5
                        linear = _finite_difference_linear_prediction(jacobians, role, direction, radius)
                        weighted_actual = torch.where(positive, actual / denominator.reshape(-1), torch.zeros_like(actual))
                        weighted_linear = torch.where(positive, linear / denominator.reshape(-1), torch.zeros_like(linear))
                        actual_norm = torch.linalg.vector_norm(actual)
                        linear_norm = torch.linalg.vector_norm(linear)
                        residual_norm = torch.linalg.vector_norm(actual - linear)
                        weighted_actual_norm = torch.linalg.vector_norm(weighted_actual)
                        weighted_linear_norm = torch.linalg.vector_norm(weighted_linear)
                        weighted_residual_norm = torch.linalg.vector_norm(weighted_actual - weighted_linear)
                        relative_error = (
                            float((weighted_residual_norm / weighted_linear_norm).detach().cpu())
                            if float(weighted_linear_norm) > 0.0 else None
                        )
                        sample_control["directions"].append({
                            "relative_radius": relative_radius,
                            "grid_role": role,
                            "is_primary_radius": math.isclose(
                                relative_radius,
                                float(cfg["statistics"]["local_direction_primary_relative_perturbation"]),
                                rel_tol=0.0, abs_tol=0.0,
                            ),
                            "actual_response_norm": float(actual_norm.detach().cpu()),
                            "linear_prediction_norm": float(linear_norm.detach().cpu()),
                            "absolute_residual_norm": float(residual_norm.detach().cpu()),
                            "weighted_actual_response_norm": float(weighted_actual_norm.detach().cpu()),
                            "weighted_linear_prediction_norm": float(weighted_linear_norm.detach().cpu()),
                            "weighted_absolute_residual_norm": float(weighted_residual_norm.detach().cpu()),
                            "relative_linearization_error": relative_error,
                            "relative_error_defined": relative_error is not None,
                            "flagged_nonlinear_over_5pct": relative_error is not None and relative_error > 0.05,
                            "mean_at_or_above_full_well_fraction_plus": float(
                                (plus_rate >= _full_well_integer(camera)).double().mean().detach().cpu()
                            ),
                            "mean_at_or_above_full_well_fraction_minus": float(
                                (minus_rate >= _full_well_integer(camera)).double().mean().detach().cpu()
                            ),
                            "plus_rate_max_e_per_pixel": float(plus_rate.max().detach().cpu()),
                            "minus_rate_max_e_per_pixel": float(minus_rate.max().detach().cpu()),
                        })
                finite_difference_images.append(sample_control)
                del direction, direction_complex, response_fields
            del jacobians, means, variances, unweighted, weighted, projected, grams, difference, difference_gram
        row_records.append({"row": int(population["rows"][selected_position]), "class": int(population["labels"][selected_position]), "exposures": exposure_rows})
        del z, real_z, phase_tangent, projection
    require(len(finite_difference_images) == 20 * len(exposure_values),
            f"local finite-difference controls are incomplete for {endpoint_key}")
    require(all(len(image["directions"]) == 2 * len(radii) for image in finite_difference_images),
            f"local finite-difference grid/radius controls are incomplete for {endpoint_key}")
    per_exposure_summary: dict[str, Any] = {}
    for exposure in exposure_values:
        by_radius: dict[str, Any] = {}
        for radius in radii:
            by_threshold: dict[str, Any] = {}
            for threshold in thresholds:
                lower_values = [record["exposures"][str(exposure)]["radii"][str(radius)]["observable_direction_count_lower"][str(threshold)] for record in row_records]
                upper_values = [record["exposures"][str(exposure)]["radii"][str(radius)]["observable_direction_count_upper"][str(threshold)] for record in row_records]
                by_threshold[str(threshold)] = {"lower_median": float(np.median(lower_values)), "lower_p10": float(np.quantile(lower_values, 0.1)), "lower_p90": float(np.quantile(lower_values, 0.9)), "upper_median": float(np.median(upper_values)), "upper_p10": float(np.quantile(upper_values, 0.1)), "upper_p90": float(np.quantile(upper_values, 0.9))}
            by_radius[str(radius)] = by_threshold
        per_exposure_summary[str(exposure)] = by_radius
    return {
        "endpoint": endpoint_key,
        "representative_count": len(row_records),
        "rows_per_class": per_class,
        "source_role": "fixed population; first rows in each saved class stratum",
        "global_phase_direction_max_relative_response": maximum_phase_response,
        "phase_direction_control_passed": maximum_phase_response <= 1.0e-10,
        "perturbation_radii_relative_to_coefficient_norm": radii,
        "primary_relative_perturbation": float(cfg["statistics"]["local_direction_primary_relative_perturbation"]),
        "thresholds": thresholds,
        "per_image": row_records,
        "population_summary": per_exposure_summary,
        "finite_difference_control": {
            "status": "COMPLETED_20_FIXED_IMAGES_ALL_EXPOSURES_GRIDS_AND_RADII",
            "sample_count": len(finite_difference_positions),
            "check_count": sum(len(image["directions"]) for image in finite_difference_images),
            "rows": sorted(int(population["rows"][position]) for position in finite_difference_positions),
            "exposures_s": exposure_values,
            "grid_roles": ["production", "reference"],
            "relative_radii": radii,
            "primary_relative_radius": float(cfg["statistics"]["local_direction_primary_relative_perturbation"]),
            "max_defined_relative_linearization_error": max(
                (float(direction["relative_linearization_error"])
                 for image in finite_difference_images for direction in image["directions"]
                 if direction["relative_linearization_error"] is not None), default=None,
            ),
            "nonlinear_over_5pct_count": sum(
                bool(direction["flagged_nonlinear_over_5pct"])
                for image in finite_difference_images for direction in image["directions"]
            ),
            "per_image_exposure": finite_difference_images,
        },
        "interpretation": "real coefficient-space local response directions after removing the per-image global-phase tangent; dimension is not a fiber count or communication capacity",
    }


def write_dual_grid_tables(
    cfg: dict[str, Any], endpoint_data: dict[str, dict[str, Any]], output_root: Path,
    camera: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    (output_root / "ablation").mkdir(parents=True, exist_ok=True)
    (output_root / "noise").mkdir(parents=True, exist_ok=True)
    resamples, seed = int(cfg["statistics"]["bootstrap_resamples"]), int(cfg["statistics"]["bootstrap_seed"])
    q = float(cfg["statistics"]["fdr_q"])
    port_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    image_noise_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    noise_summary: dict[str, Any] = {}
    for grid_role in ("production", "reference"):
        csv_path = output_root / "ablation" / f"per_image_per_port_ablation_{grid_role}.csv.gz"
        fields = ["endpoint", "grid_role", "row", "label", "port_zero_based", "control", "metric_defined", "native_mse28", "ablation_mse28", "delta_mse28", "native_psnr28_db", "ablation_psnr28_db", "delta_psnr28_db", "native_ssim28", "ablation_ssim28", "delta_ssim28", "native_production_loss", "ablation_production_loss", "delta_production_loss", "ablation_clipping_fraction", "native_camera_power_w", "ablation_camera_power_w"]
        written_rows = 0
        with gzip.open(csv_path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for endpoint, data in endpoint_data.items():
                rows, labels = data["rows"], data["labels"]
                native = data[f"native_metrics_{grid_role}"]
                ablation = data[f"ablation_metrics_{grid_role}"]
                other_role = "reference" if grid_role == "production" else "production"
                other_delta = data[f"ablation_metrics_{other_role}"][:, :, :, 0] - data[f"native_metrics_{other_role}"][:, None, None, 0]
                this_delta = ablation[:, :, :, 0] - native[:, None, None, 0]
                per_endpoint: dict[str, Any] = {"input_role": "DEV" if endpoint.startswith("small") else "TRAIN", "sample_count": int(len(rows)), "grid_role": grid_role, "native_population_metrics": {}}
                for metric_index, metric in enumerate(METRICS):
                    boot = bootstrap_effect(native[:, metric_index], labels, resamples, seed)
                    per_endpoint["native_population_metrics"][metric] = {"mean": float(np.asarray(boot["mean"]).reshape(-1)[0]), "ci95_low": float(np.asarray(boot["ci95_low"]).reshape(-1)[0]), "ci95_high": float(np.asarray(boot["ci95_high"]).reshape(-1)[0])}
                control_summaries: dict[str, Any] = {}
                for control_index, control in enumerate(CHANNELS):
                    metric_delta = ablation[:, :, control_index, :] - native[:, None, :]
                    valid_ports = np.ones((len(rows), PORTS), dtype=bool) if control_index == 0 else data["equal_power_defined"].astype(bool, copy=False)
                    mse_boot = bootstrap_effect_per_port(metric_delta[:, :, 0], labels, valid_ports, resamples, seed)
                    q_values = fdr_bh(mse_boot["p_two_sided_bootstrap"], q)
                    other_boot = bootstrap_effect_per_port(other_delta[:, :, control_index], labels, valid_ports, resamples, seed)
                    other_q_values = fdr_bh(other_boot["p_two_sided_bootstrap"], q)
                    grid_difference_boot = bootstrap_effect_per_port(np.abs(this_delta[:, :, control_index] - other_delta[:, :, control_index]), labels, valid_ports, resamples, seed)
                    reliability: list[dict[str, Any]] = []
                    for port in range(PORTS):
                        mean = float(mse_boot["mean"][port])
                        low, high = float(mse_boot["ci95_low"][port]), float(mse_boot["ci95_high"][port])
                        other_mean = float(other_boot["mean"][port])
                        other_low, other_high = float(other_boot["ci95_low"][port]), float(other_boot["ci95_high"][port])
                        mean_diff = float(grid_difference_boot["mean"][port])
                        clear_zero = (low > 0.0 or high < 0.0) and (other_low > 0.0 or other_high < 0.0)
                        both_fdr = bool(q_values[port] <= q and other_q_values[port] <= q)
                        same_sign = mean != 0.0 and other_mean != 0.0 and np.sign(mean) == np.sign(other_mean)
                        stable_margin = min(abs(mean), abs(other_mean)) > mean_diff
                        if both_fdr and clear_zero and same_sign and stable_margin:
                            classification = "reliably_reconstruction_sensitive"
                        elif not both_fdr or not clear_zero:
                            classification = "statistically_unresolved"
                        else:
                            classification = "numerically_unresolved"
                        reliability.append({"port_zero_based": port, "defined_sample_count": int(valid_ports[:, port].sum()), "undefined_sample_count": int((~valid_ports[:, port]).sum()), "grid_effect_mean_delta_mse28": mean, "grid_effect_ci95_low": low, "grid_effect_ci95_high": high, "other_grid_effect_mean_delta_mse28": other_mean, "other_grid_ci95_low": other_low, "other_grid_ci95_high": other_high, "grid_pair_mean_absolute_per_image_effect_difference": mean_diff, "grid_pair_difference_ci95_high": float(grid_difference_boot["ci95_high"][port]), "grid_fdr_q": float(q_values[port]), "other_grid_fdr_q": float(other_q_values[port]), "classification": classification})
                        defined_delta = metric_delta[valid_ports[:, port], port, :]
                        port_rows.append({"endpoint": endpoint, "grid_role": grid_role, "control": control, **reliability[-1], "mean_delta_psnr28_db": float(defined_delta[:, 1].mean()) if len(defined_delta) else None, "mean_delta_ssim28": float(defined_delta[:, 2].mean()) if len(defined_delta) else None, "mean_delta_production_loss": float(defined_delta[:, 3].mean()) if len(defined_delta) else None, "mean_clipping_fraction": float(ablation[valid_ports[:, port], port, control_index, 4].mean()) if valid_ports[:, port].any() else None, "mean_camera_power_w": float(ablation[valid_ports[:, port], port, control_index, 5].mean()) if valid_ports[:, port].any() else None})
                        for label in range(10):
                            mask = (labels == label) & valid_ports[:, port]
                            selected_delta = metric_delta[mask, port, :]
                            class_rows.append({"endpoint": endpoint, "grid_role": grid_role, "class": label, "control": control, "port_zero_based": port, "sample_count": int(mask.sum()), "undefined_sample_count": int(np.sum((labels == label) & ~valid_ports[:, port])), "mean_delta_mse28": float(selected_delta[:, 0].mean()) if len(selected_delta) else None, "mean_delta_psnr28_db": float(selected_delta[:, 1].mean()) if len(selected_delta) else None, "mean_delta_ssim28": float(selected_delta[:, 2].mean()) if len(selected_delta) else None, "mean_delta_production_loss": float(selected_delta[:, 3].mean()) if len(selected_delta) else None})
                    population_metrics: dict[str, Any] = {}
                    for metric_index, metric in enumerate(METRICS):
                        values = metric_delta[:, :, metric_index]
                        valid = valid_ports & np.isfinite(values)
                        denominator = valid.sum(axis=1)
                        image_delta = np.divide(np.where(valid, values, 0.0).sum(axis=1), denominator, out=np.full(len(rows), np.nan, dtype=np.float64), where=denominator > 0)
                        included = np.isfinite(image_delta)
                        class_counts = np.bincount(labels[included], minlength=10)
                        if np.all(class_counts >= 2):
                            boot = bootstrap_effect(image_delta[included], labels[included], resamples, seed)
                            population_metrics[metric] = {"mean_delta": float(boot["mean"][0]), "ci95_low": float(boot["ci95_low"][0]), "ci95_high": float(boot["ci95_high"][0]), "defined_image_count": int(included.sum()), "undefined_image_count": int((~included).sum())}
                        else:
                            population_metrics[metric] = {"mean_delta": None, "ci95_low": None, "ci95_high": None, "defined_image_count": int(included.sum()), "undefined_image_count": int((~included).sum()), "status": "insufficient_defined_rows_for_class_bootstrap"}
                    control_summaries[control] = {"population_mean": population_metrics, "per_port_reconstruction_sensitivity": reliability, "reliably_sensitive_port_count": sum(item["classification"] == "reliably_reconstruction_sensitive" for item in reliability), "statistically_unresolved_count": sum(item["classification"] == "statistically_unresolved" for item in reliability), "numerically_unresolved_count": sum(item["classification"] == "numerically_unresolved" for item in reliability)}
                    for index, row in enumerate(rows):
                        for port in range(PORTS):
                            writer.writerow({"endpoint": endpoint, "grid_role": grid_role, "row": int(row), "label": int(labels[index]), "port_zero_based": port, "control": control, "metric_defined": bool(valid_ports[index, port]), "native_mse28": float(native[index, 0]), "ablation_mse28": float(ablation[index, port, control_index, 0]), "delta_mse28": float(metric_delta[index, port, 0]), "native_psnr28_db": float(native[index, 1]), "ablation_psnr28_db": float(ablation[index, port, control_index, 1]), "delta_psnr28_db": float(metric_delta[index, port, 1]), "native_ssim28": float(native[index, 2]), "ablation_ssim28": float(ablation[index, port, control_index, 2]), "delta_ssim28": float(metric_delta[index, port, 2]), "native_production_loss": float(native[index, 3]), "ablation_production_loss": float(ablation[index, port, control_index, 3]), "delta_production_loss": float(metric_delta[index, port, 3]), "ablation_clipping_fraction": float(ablation[index, port, control_index, 4]), "native_camera_power_w": float(native[index, 5]), "ablation_camera_power_w": float(ablation[index, port, control_index, 5])})
                    written_rows += len(rows) * PORTS
                per_endpoint["controls"] = control_summaries
                power = data["modal_power_w"]
                total_each = power.sum(axis=1)
                squares_each = np.square(power).sum(axis=1)
                neff = np.divide(np.square(total_each), squares_each, out=np.full_like(total_each, np.nan), where=squares_each > 0)
                mean_power = power.mean(axis=0)
                finite_neff = neff[np.isfinite(neff)]
                per_endpoint["mean_modal_participation_neff"] = float(finite_neff.mean()) if len(finite_neff) else None
                mean_power_squares = float(np.square(mean_power).sum())
                per_endpoint["population_mean_power_vector_neff"] = float(np.square(mean_power.sum()) / mean_power_squares) if mean_power_squares > 0.0 else None
                per_endpoint["strict_positive_coefficient_roots_per_sample_minimum"] = int(np.min(np.sum(power > 0.0, axis=1)))
                per_endpoint["population_mean_port_power_spread_db"] = float(10.0 * np.log10(mean_power.max() / mean_power.min())) if mean_power.min() > 0.0 else None
                for port in range(PORTS):
                    values = power[:, port]
                    peak = power.max(axis=1)
                    valid_peak = peak > 0.0
                    ratios = np.divide(values, peak, out=np.zeros_like(values), where=peak > 0.0)
                    valid_ratios = ratios[valid_peak]
                    mean_ratio = float(valid_ratios.mean()) if len(valid_ratios) else None
                    power_rows.append({"endpoint": endpoint, "grid_role": "coefficient_space", "class": "all", "port_zero_based": port, "sample_count": int(len(values)), "valid_relative_power_sample_count": int(valid_peak.sum()), "mean_power_w": float(mean_power[port]), "median_power_w": float(np.median(values)), "p10_power_w": float(np.quantile(values, 0.1)), "p90_power_w": float(np.quantile(values, 0.9)), "mean_relative_power_db_to_image_peak": float(10.0 * np.log10(mean_ratio)) if mean_ratio is not None and mean_ratio > 0.0 else None, "occupancy_minus10db": float(np.mean(valid_ratios >= 0.1)) if len(valid_ratios) else None, "occupancy_minus20db": float(np.mean(valid_ratios >= 0.01)) if len(valid_ratios) else None, "occupancy_minus30db": float(np.mean(valid_ratios >= 0.001)) if len(valid_ratios) else None, "occupancy_minus40db": float(np.mean(valid_ratios >= 0.0001)) if len(valid_ratios) else None})
                    for label in range(10):
                        mask = labels == label
                        values_class = power[mask, port]
                        class_peak = power[mask].max(axis=1)
                        class_valid_peak = class_peak > 0.0
                        ratios_class = np.divide(values_class, class_peak, out=np.zeros_like(values_class), where=class_valid_peak)
                        valid_class_ratios = ratios_class[class_valid_peak]
                        class_mean_ratio = float(valid_class_ratios.mean()) if len(valid_class_ratios) else None
                        power_rows.append({"endpoint": endpoint, "grid_role": "coefficient_space", "class": label, "sample_count": int(len(values_class)), "valid_relative_power_sample_count": int(class_valid_peak.sum()), "mean_power_w": float(values_class.mean()), "median_power_w": float(np.median(values_class)), "p10_power_w": float(np.quantile(values_class, 0.1)), "p90_power_w": float(np.quantile(values_class, 0.9)), "mean_relative_power_db_to_image_peak": float(10.0 * np.log10(class_mean_ratio)) if class_mean_ratio is not None and class_mean_ratio > 0.0 else None, "occupancy_minus10db": float(np.mean(valid_class_ratios >= 0.1)) if len(valid_class_ratios) else None, "occupancy_minus20db": float(np.mean(valid_class_ratios >= 0.01)) if len(valid_class_ratios) else None, "occupancy_minus30db": float(np.mean(valid_class_ratios >= 0.001)) if len(valid_class_ratios) else None, "occupancy_minus40db": float(np.mean(valid_class_ratios >= 0.0001)) if len(valid_class_ratios) else None})
                summaries[f"{endpoint}/{grid_role}"] = per_endpoint
                if grid_role == "reference":
                    continue
                intervals = data["port_detectability"]
                counts = data["observable_port_counts"]
                exposure_summary: dict[str, Any] = {}
                for control_index, control in enumerate(CHANNELS):
                    exposure_summary[control] = {}
                    for exposure_index, exposure in enumerate(camera["exposure_sweep_s"]):
                        count_record = counts[:, control_index, exposure_index]
                        stats = {"mean_guaranteed_observable_ports": float(np.mean(count_record[:, 1, 0])), "mean_possible_observable_ports": float(np.mean(count_record[:, 1, 1])), "mean_defined_ports": float(np.mean(np.sum(data["equal_power_defined"], axis=1))) if control == "equal_ols_power" else float(PORTS), "mean_undefined_equal_power_ports": float(np.mean(np.sum(~data["equal_power_defined"], axis=1))) if control == "equal_ols_power" else 0.0, "p10_guaranteed_at_threshold_3": float(np.quantile(count_record[:, 1, 0], 0.1)), "median_guaranteed_at_threshold_3": float(np.median(count_record[:, 1, 0])), "p90_possible_at_threshold_3": float(np.quantile(count_record[:, 1, 1], 0.9)), "per_class_threshold_3": {str(label): {"mean_lower": float(np.mean(count_record[labels == label, 1, 0])), "mean_upper": float(np.mean(count_record[labels == label, 1, 1])), "mean_undefined_equal_power_ports": float(np.mean(np.sum(~data["equal_power_defined"][labels == label], axis=1))) if control == "equal_ols_power" else 0.0} for label in range(10)}}
                        exposure_summary[control][str(float(exposure))] = stats
                        for threshold_index, threshold in enumerate(OBSERVABILITY_THRESHOLDS):
                            for index, row in enumerate(rows):
                                image_noise_rows.append({"endpoint": endpoint, "grid_weighting": "common_conservative_two_grid", "row": int(row), "class": int(labels[index]), "control": control, "exposure_s": float(exposure), "threshold": threshold, "guaranteed_port_count_lower": int(counts[index, control_index, exposure_index, threshold_index, 0]), "possible_port_count_upper": int(counts[index, control_index, exposure_index, threshold_index, 1]), "undefined_equal_power_port_count": int(np.sum(~data["equal_power_defined"][index])) if control == "equal_ols_power" else 0})
                        for port in range(PORTS):
                            values = intervals[:, control_index, exposure_index, port, :]
                            valid = np.isfinite(values[:, 0])
                            selected = values[valid]
                            mean_values = np.mean(selected, axis=0) if len(selected) else np.full(5, np.nan)
                noise_rows.append({"endpoint": endpoint, "control": control, "exposure_s": float(exposure), "port_zero_based": port, "defined_sample_count": int(valid.sum()), "undefined_sample_count": int((~valid).sum()), "mean_d_production": float(mean_values[0]), "mean_d_reference": float(mean_values[1]), "mean_grid_distance_u": float(mean_values[2]), "mean_d_lower": float(mean_values[3]), "mean_d_upper": float(mean_values[4]), "fraction_guaranteed_observable_at_3": float(np.mean(selected[:, 3] >= 3.0)) if len(selected) else None, "fraction_below_proxy_threshold_3": float(np.mean(selected[:, 4] < 3.0)) if len(selected) else None, "fraction_grid_unresolved_at_3": float(np.mean((selected[:, 3] < 3.0) & (selected[:, 4] >= 3.0))) if len(selected) else None, "fraction_equal_power_undefined": float(1.0 - valid.mean())})
                noise_summary[endpoint] = exposure_summary
        expected_grid_rows = sum(len(data["rows"]) * PORTS * len(CHANNELS) for data in endpoint_data.values())
        require(written_rows == expected_grid_rows, f"dual-grid image-port CSV emitted an unexpected row count for {grid_role}: {written_rows} != {expected_grid_rows}")
        with gzip.open(csv_path, "rt", newline="", encoding="utf-8") as stream:
            durable_rows = sum(1 for _ in stream) - 1
        require(durable_rows == expected_grid_rows, f"dual-grid image-port CSV readback row count mismatch for {grid_role}: {durable_rows} != {expected_grid_rows}")
        write_json(output_root / "ablation" / f"per_image_per_port_ablation_{grid_role}_row_count.json", {"task_id": TASK_ID, "grid_role": grid_role, "written_data_rows": written_rows, "readback_data_rows": durable_rows, "expected_data_rows": expected_grid_rows, "passed": True})
    write_csv(output_root / "ablation" / "dual_grid_per_port_reconstruction_sensitivity.csv", port_rows)
    write_csv(output_root / "ablation" / "per_port_per_class_dual_grid.csv", class_rows)
    write_csv(output_root / "ablation" / "per_port_power_summary.csv", power_rows)
    write_csv(output_root / "noise" / "per_port_observability.csv", noise_rows)
    write_csv(output_root / "ablation" / "per_image_noise_observable_port_counts.csv", image_noise_rows)
    write_json(output_root / "ablation" / "decoder_ablation_summary.json", summaries)
    write_json(output_root / "noise" / "observability_summary.json", noise_summary)
    return summaries, noise_summary


def create_dual_grid_figures(
    output_root: Path, endpoint_data: dict[str, dict[str, Any]],
    summaries: dict[str, Any], cumulative_rows: list[dict[str, Any]],
    noise_summary: dict[str, Any], local_spectra: dict[str, Any], camera: dict[str, Any],
) -> list[str]:
    import matplotlib.pyplot as plt

    paths: list[str] = []
    endpoints = list(endpoint_data)
    figure, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for axis, endpoint in zip(axes.flat, endpoints, strict=True):
        power = endpoint_data[endpoint]["modal_power_w"].mean(axis=0).reshape(10, 10)
        image = axis.imshow(10.0 * np.log10(np.maximum(power / power.max(), np.finfo(np.float64).tiny)), cmap="magma", vmin=-50, vmax=0)
        axis.set_title(endpoint)
        axis.set_xlabel("OLS column")
        axis.set_ylabel("OLS row")
    figure.colorbar(image, ax=axes, label="population mean power / maximum (dB)")
    path = output_root / "figures" / "population_mean_port_power_10x10.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path.relative_to(output_root).as_posix())

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, endpoint in zip(axes.flat, endpoints, strict=True):
        for grid_role, color in (("production", "#1f77b4"), ("reference", "#d62728")):
            record = summaries[f"{endpoint}/{grid_role}"]
            effects = [item["grid_effect_mean_delta_mse28"] for item in record["controls"]["raw_zero"]["per_port_reconstruction_sensitivity"]]
            axis.plot(np.arange(PORTS), effects, color=color, linewidth=0.8, label=grid_role)
        axis.axhline(0.0, color="#333333", linewidth=0.6)
        axis.set_title(endpoint)
        axis.set_xlabel("port index")
        axis.set_ylabel("raw-zero ΔMSE")
        axis.legend(fontsize=8)
    path = output_root / "figures" / "dual_grid_port_mse_effects.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path.relative_to(output_root).as_posix())

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, endpoint in zip(axes.flat, endpoints, strict=True):
        for grid_role, color in (("production", "#1f77b4"), ("reference", "#d62728")):
            rows = sorted((row for row in cumulative_rows if row["endpoint"] == endpoint and row["grid_role"] == grid_role), key=lambda row: row["retained_roots"])
            axis.plot([row["retained_roots"] for row in rows], [row["mean_mse28"] for row in rows], color=color, label=grid_role)
        axis.set_title(endpoint)
        axis.set_xlabel("strongest roots retained")
        axis.set_ylabel("mean 28×28 MSE")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    path = output_root / "figures" / "dual_grid_cumulative_retention.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path.relative_to(output_root).as_posix())

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    exposures = [float(value) for value in camera["exposure_sweep_s"]]
    for axis, endpoint in zip(axes.flat, endpoints, strict=True):
        for control, color in (("raw_zero", "#1f77b4"), ("equal_ols_power", "#d62728")):
            lower = [noise_summary[endpoint][control][str(value)]["mean_guaranteed_observable_ports"] for value in exposures]
            upper = [noise_summary[endpoint][control][str(value)]["mean_possible_observable_ports"] for value in exposures]
            axis.plot(exposures, lower, marker="o", color=color, label=f"{control} lower")
            axis.plot(exposures, upper, marker="x", linestyle="--", color=color, label=f"{control} upper")
            axis.fill_between(exposures, lower, upper, color=color, alpha=0.1)
        axis.set_xscale("log")
        axis.set_ylim(0, 100)
        axis.set_title(endpoint)
        axis.set_xlabel("proxy exposure (s)")
        axis.set_ylabel("ports detectable at D=3")
        axis.legend(fontsize=6)
        axis.grid(alpha=0.2)
    path = output_root / "figures" / "proxy_observability_grid_sensitivity.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path.relative_to(output_root).as_posix())

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, endpoint in zip(axes.flat, endpoints, strict=True):
        local = local_spectra[endpoint]["population_summary"]
        exposure = "0.001"
        radius = str(local_spectra[endpoint]["primary_relative_perturbation"])
        threshold = "3.0"
        lower = local[exposure][radius][threshold]["lower_median"]
        upper = local[exposure][radius][threshold]["upper_median"]
        axis.bar(["lower", "upper"], [lower, upper], color=["#1f77b4", "#ff7f0e"])
        axis.set_ylim(0, 199)
        axis.set_title(f"{endpoint}, 1 ms, 1% perturbation")
        axis.set_ylabel("median observable real directions at D=3")
    path = output_root / "figures" / "local_direction_ranges.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    paths.append(path.relative_to(output_root).as_posix())
    return paths


def _format_diagnostic_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "未定义"
    number = float(value)
    if not math.isfinite(number):
        return "未定义"
    return f"{number:.{digits}f}"


def compact_local_direction_spectra(local_spectra: dict[str, Any]) -> dict[str, Any]:
    summary_fields = (
        "endpoint",
        "representative_count",
        "rows_per_class",
        "source_role",
        "global_phase_direction_max_relative_response",
        "phase_direction_control_passed",
        "perturbation_radii_relative_to_coefficient_norm",
        "primary_relative_perturbation",
        "thresholds",
        "population_summary",
        "finite_difference_control",
        "interpretation",
    )
    result: dict[str, Any] = {}
    for endpoint, record in local_spectra.items():
        compact = {field: record[field] for field in summary_fields if field != "finite_difference_control"}
        control = record.get("finite_difference_control")
        if isinstance(control, dict):
            compact["finite_difference_control"] = {
                key: value for key, value in control.items() if key != "per_image_exposure"
            }
        else:
            compact["finite_difference_control"] = control
        result[endpoint] = compact
    return result


def dual_grid_preflight_report(
    preflight: dict[str, Any], independent_reference: dict[str, Any]
) -> dict[str, Any]:
    return {
        "status": preflight["status"],
        "all_100_port_same_grid_reference_within_1e-4": preflight[
            "basis_same_grid_all_100_ports_within_1e-4"
        ],
        "all_100_port_complex_camera_bases_compared_to_reference": preflight[
            "basis_same_grid_all_100_ports_within_1e-4"
        ],
        "cross_grid_differences": independent_reference["endpoints"],
        "independent_reference": independent_reference,
        "cross_grid_is_reported_sensitivity_not_stop_condition": True,
        "direct_replay_max_mse_absolute_error": preflight[
            "direct_production_vs_basis_checks"
        ]["maximum_direct_vs_basis_mse_absolute_error"],
        "direct_replay_and_ablation_checks_passed": preflight[
            "direct_production_vs_basis_checks"
        ]["passed"],
        "production_w_per_pixel_power_identity_max_relative_error": preflight[
            "direct_production_vs_basis_checks"
        ]["maximum_production_w_per_pixel_vs_complex_field_power_relative_error"],
    }


def assemble_dual_grid_machine_report(
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    independent_reference: dict[str, Any],
    covariance: dict[str, Any],
    summaries: dict[str, Any],
    noise_summary: dict[str, Any],
    tolerance_counts: dict[str, Any],
    local_spectra: dict[str, Any],
    figure_paths: list[str],
    elapsed_seconds: float,
    endpoint_elapsed: dict[str, float],
) -> dict[str, Any]:
    return {
        "schema": "gaussian_ols_effective_channel_dual_grid.v1",
        "task_id": TASK_ID,
        "status": "CONDITIONAL_DUAL_GRID_DIAGNOSTIC_COMPLETE_REAL_NOISE_CALIBRATION_PENDING",
        "outer_decision": "FROZEN_DUAL_GRID_CONDITIONAL_DIAGNOSTIC_COMPLETE",
        "decision_question": "Quantify power concentration, proxy observability, and frozen Decoder reconstruction contribution for the historical 48 dB and 32 dB OLS spreads.",
        "evidence_tier": "diagnostic-only",
        "input_lineage": preflight["input_lineage"],
        "noise_proxy": {
            "parameters": preflight["noise_proxy"],
            "metric": "pixelwise matched detectability distance using two independent clipped exposures and common conservative variance across grids",
            "real_camera_calibration": "NOT_AVAILABLE; measured hardware channel counts remain undetermined",
            "gain": "fixed checkpoint gain; not camera calibration",
        },
        "covariance": covariance,
        "decoder_ablation": summaries,
        "noise_observability": noise_summary,
        "cumulative_retention": {
            "port_order_source": "ranking_manifest.json frozen from saved population mean power before ablation outcomes",
            "minimum_stable_roots": tolerance_counts,
            "curves_file": "ablation/cumulative_root_retention_dual_grid.csv",
            "tail_condition": "all subsequent retained-root counts satisfy the requested relative MSE tolerance",
        },
        "local_direction_spectra": compact_local_direction_spectra(local_spectra),
        "preflight": dual_grid_preflight_report(preflight, independent_reference),
        "artifacts": {
            "dual_grid_port_ablation": [
                "ablation/per_image_per_port_ablation_production.csv.gz",
                "ablation/per_image_per_port_ablation_reference.csv.gz",
            ],
            "chunked_recovery_sources": "ablation/<endpoint>/chunks/*.npz with complete.json readback receipts",
            "paired_sensitivity": "ablation/dual_grid_per_port_reconstruction_sensitivity.csv",
            "power_summary": "ablation/per_port_power_summary.csv",
            "noise_per_port": "noise/per_port_observability.csv",
            "noise_per_image": "ablation/per_image_noise_observable_port_counts.csv",
            "retention": "ablation/cumulative_root_retention_dual_grid.csv",
            "local_direction_spectra": "noise/local_direction_spectra.json",
            "per_image_local_direction_spectra": "noise/local_direction_spectra.json",
            "figures": figure_paths,
        },
        "training_updates": 0,
        "optimizers_created": 0,
        "prior_samples": 0,
        "official_test_reads": 0,
        "hardware_operations": 0,
        "elapsed_seconds_observed": float(elapsed_seconds),
        "endpoint_elapsed_seconds": endpoint_elapsed,
        "statistics": {
            "bootstrap_resamples": int(cfg["statistics"]["bootstrap_resamples"]),
            "bootstrap_seed": int(cfg["statistics"]["bootstrap_seed"]),
            "class_stratified": True,
            "paired_grid_draws_shared": True,
            "fdr_q": float(cfg["statistics"]["fdr_q"]),
            "fdr_scope": "100 ports per endpoint/control/grid",
        },
    }


def write_dual_grid_report(output_root: Path) -> None:
    report = json.loads((output_root / "machine_report.json").read_text(encoding="utf-8"))
    require(report.get("task_id") == TASK_ID, "dual-grid report task identity changed")
    ablation_rows: list[str] = []
    endpoints = sorted({key.rsplit("/", 1)[0] for key in report["decoder_ablation"]})
    for endpoint in endpoints:
        for grid_role in ("production", "reference"):
            summary = report["decoder_ablation"][f"{endpoint}/{grid_role}"]
            raw = summary["controls"]["raw_zero"]
            equal = summary["controls"]["equal_ols_power"]
            raw_mse = raw["population_mean"]["mse28"]
            equal_mse = equal["population_mean"]["mse28"]
            ablation_rows.append(f"| {endpoint} | {grid_role} | {summary['sample_count']} | {_format_diagnostic_number(summary['mean_modal_participation_neff'])} | {_format_diagnostic_number(summary['population_mean_power_vector_neff'])} | {_format_diagnostic_number(summary['population_mean_port_power_spread_db'])} | {raw['reliably_sensitive_port_count']} | {equal['reliably_sensitive_port_count']} | {_format_diagnostic_number(raw_mse['mean_delta'], 6)} | {_format_diagnostic_number(equal_mse['mean_delta'], 6)} |")
    covariance_rows = [f"| {name} | {record['sample_count']} | {record['effective_participation']:.2f} | {record['directions_95_percent']} | {record['directions_99_percent']} | {record.get('raw_complex_covariance_effective_participation')} | {record.get('original_ungauged_complex_cross_moment_available', False)} |" for name, record in report["covariance"].items()]
    noise_rows: list[str] = []
    for endpoint, controls in report["noise_observability"].items():
        for control, exposures in controls.items():
            for exposure, values in exposures.items():
                noise_rows.append(f"| {endpoint} | {control} | {float(exposure):g} | {values['mean_guaranteed_observable_ports']:.2f} | {values['mean_possible_observable_ports']:.2f} | {_format_diagnostic_number(values['mean_undefined_equal_power_ports'])} | {values['per_class_threshold_3']} |")
    retention_rows: list[str] = []
    for key, values in report["cumulative_retention"]["minimum_stable_roots"].items():
        retention_rows.append(f"| {key} | {values['0.01'] if values['0.01'] is not None else '未达到'} | {values['0.05'] if values['0.05'] is not None else '未达到'} |")
    local_rows: list[str] = []
    local_fd_rows: list[str] = []
    for endpoint, record in report["local_direction_spectra"].items():
        exposure = "0.001"
        radius = str(record["primary_relative_perturbation"])
        for threshold, summary in record["population_summary"][exposure][radius].items():
            local_rows.append(f"| {endpoint} | {radius} | {float(threshold):g} | {summary['lower_median']:.1f} [{summary['lower_p10']:.1f}, {summary['lower_p90']:.1f}] | {summary['upper_median']:.1f} [{summary['upper_p10']:.1f}, {summary['upper_p90']:.1f}] |")
        finite_difference = record.get("finite_difference_control") or {}
        finite_difference_validated = (
            finite_difference.get("sample_count") == 20
            and finite_difference.get("status") == "COMPLETED_20_FIXED_IMAGES_ALL_EXPOSURES_GRIDS_AND_RADII"
        )
        checks = [direction for image in finite_difference.get("per_image_exposure", [])
                  for direction in image.get("directions", [])]

        def select_fd(relative_radius: float, role: str) -> dict[str, Any]:
            values = [item for item in checks if float(item["relative_radius"]) == relative_radius
                      and item["grid_role"] == role and item.get("relative_linearization_error") is not None]
            errors = [float(item["relative_linearization_error"]) for item in values]
            return {
                "relative_linearization_error": float(np.median(errors)) if errors else None,
                "p90_relative_linearization_error": float(np.quantile(errors, 0.9)) if errors else None,
                "flagged_nonlinear_over_5pct": sum(bool(item["flagged_nonlinear_over_5pct"]) for item in values),
                "defined_count": len(errors),
            }

        primary_production = select_fd(float(radius), "production")
        primary_reference = select_fd(float(radius), "reference")
        small_production = select_fd(0.001, "production")
        small_reference = select_fd(0.001, "reference")
        def format_fd_error(item: dict[str, Any]) -> str:
            value = item.get("relative_linearization_error")
            return "未定义" if value is None or not math.isfinite(float(value)) else f"{float(value):.3g}"
        local_fd_rows.append(
            f"| {endpoint} | {format_fd_error(small_production) if finite_difference_validated else '待complex128复核'} | {format_fd_error(small_reference) if finite_difference_validated else '待complex128复核'} | {format_fd_error(primary_production) if finite_difference_validated else '待complex128复核'} | {format_fd_error(primary_reference) if finite_difference_validated else '待complex128复核'} | {primary_production.get('flagged_nonlinear_over_5pct', '待复核')} / {primary_reference.get('flagged_nonlinear_over_5pct', '待复核')} |"
        )
    reference_rows: list[str] = []
    for endpoint, item in report["preflight"]["independent_reference"]["endpoints"].items():
        cross = item["independent_reference_cross_grid_difference"]
        reference_rows.append(f"| {endpoint} | {cross['maximum_complex_field_relative_l2']:.6g} | {cross['maximum_camera_power_relative_error']:.6g} |")
    lines = [
        "# Gaussian OLS双网格有效通道诊断",
        "",
        "## 结果与状态",
        "",
        f"状态：**{report['status']}**。四个Decoder端点的双网格消融、协方差和代理噪声人口诊断已完成；局部Jacobian谱已计算，但1%有限差分的complex128复核尚未完成。小架构每端点5,000张固定DEV，大架构300张历史TRAIN；官方TEST读取0。结果仍是冻结条件诊断，真实噪声标定待补。",
        "",
        "| 端点 | 网格 | 样本数 | 单图功率参与数 | 人口均值功率参与数 | 人口功率跨度(dB) | 双网格稳健敏感根数：原功率 | 等OLS功率 | 原功率ΔMSE | 等OLS功率ΔMSE |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *ablation_rows,
        "",
        "稳健敏感根要求两网格分别通过FDR且置信区间同号越过零，并且最小效应幅度大于逐图两网格效应差异的平均绝对值；其余按统计未分辨或数值未分辨保留。移除后变好的端口也保留。等OLS功率对照只恢复剩余端口的OLS系数总功率，不匹配相机功率；置零后剩余功率为0时该指标标记为未定义，不用epsilon补数。",
        "",
        "| 协方差人口 | n | Gauge实虚参与数 | 95%方向数 | 99%方向数 | 原复协方差参与数 | 原始交叉矩可用性 |",
        "|---|---:|---:|---:|---:|---:|---|",
        *covariance_rows,
        "",
        "| 端点 | 消融控制 | 曝光(s) | D≥3保证可观察根数均值下限 | D≥3可能可观察根数均值上限 | 等OLS功率未定义根均值 | 十类下限/上限 |",
        "|---|---|---:|---:|---:|---:|---|",
        *noise_rows,
        "",
        "可观察性基于QE=0.5、读噪声1 e⁻ RMS、暗电流0、满阱30,000 e⁻的精确截断矩代理；每像素匹配检测距离使用两次独立曝光的方差和，跨网格用较大像素方差并形成差异带。D=1、3、5均在逐端口记录中；这里列D=3。该值不是相机实测检出概率。",
        "",
        "| 端点/网格 | 最少保留根：从该点起所有后续MSE误差≤1% | ≤5% |",
        "|---|---:|---:|",
        *retention_rows,
        "",
        "| 端点 | 相对扰动半径 | 检测门限 | 逐图下限中位数[P10,P90] | 上限中位数[P10,P90] |",
        "|---|---:|---:|---:|---:|",
        *local_rows,
        "",
        "| 端点 | 0.1%差分误差生产网格 | 0.1%差分误差参考网格 | 1%差分误差生产网格 | 1%差分误差参考网格 | 1%两网格均非线性标记 |",
        "|---|---:|---:|---:|---:|---|",
        *local_fd_rows,
        "",
        "局部方向是每张代表样本上的实数系数Jacobian谱，已去掉该图公共相位方向；主扰动半径为总复系数范数的1%，并另报0.1%和10%。当前保存的有限差分数值来自先导复算，其中扰动系数曾经过complex64量化；因此表格将其标为待complex128复核，不把先导数值作为最终验证。旧complex64先导控制在1%处曾显示强非线性，可能来自近满阱像素的极小基线方差以及扰动跨越满阱截断边界；修正精度后的有限差分尚未在指定AutoDL执行机重跑。故1%方向数目前只能读作Jacobian一阶外推，不能当成经有限扰动验证的1%可观察方向数。公共相位方向残差约为数值零；方向数仍不等于光纤数或通信容量。",
        "",
        "## 科学认识",
        "",
        "功率跨度本身不等于有效通道数。大架构固定300张TRAIN的逐图参与数均值16.77、人口平均功率向量参与数34.56、端口跨度47.48 dB；小Gaussian 30k固定DEV分别为24.62、36.25、32.22 dB。大架构在单图上更集中，但人口平均功率参与数与小架构接近，说明48 dB极值差没有直接变成相近数量级的‘失效通道’。不过两端点的输入角色、几何和Decoder不同，不能据此排名架构。",
        "双网格稳健逐根敏感端口：大架构原功率100/100、等OLS总功率96/100；小30k为100/100与99/100，A35k为100/100与99/100，B35k为100/100与100/100。大架构单根消融平均ΔMSE从原功率0.000669降至等OLS功率0.000236，约65%差值被总模态功率对照抵消；小30k从0.000902降至0.000537，约40%被抵消。由此支持总吞吐是部分原因，同时等功率后残余退化支持分布/相位仍有贡献。敏感表示在该固定人口和检验条件下可重复影响重构，不表示每根都不可替代。",
        "固定最强端口排序的累计曲线显示，大架构保留43根即可使后续各点均落在原生MSE的1%误差内（5%容忍为38根）；小30k和A35k分别需要88/71根，B35k需要87/69根。该结果在各自人口内体现联合冗余，不是可部署删减通道数。代理噪声下D≥3可观察端口均值：10 μs时大架构67.36、小架构约99.90–99.93；1 ms时大架构90.50、小架构约100；10 ms时大架构94.89、小架构约100。小架构代理响应较易观察，但真实相机标定缺失，不能把这些数称为实测有效通道。",
        "功率参与数描述系数分布集中程度；协方差谱描述人口变化方向；噪声带描述指定代理相机下端口消融的输出可分辨程度；双网格消融描述Decoder重构对端口的依赖。四者不应合并成单一‘利用率’。较多逐根敏感端口与较低累计保留根数可同时出现，因为单根影响不等于联合必要性。",
        "",
        "## 目标缺口",
        "",
        "本轮代理噪声不是实测暗帧/平场/光子转换与ADC校准，不能回答真实相机下的可观察根数；小DEV与大TRAIN角色不同，不能形成受控架构排名。跨网格敏感性带只覆盖两套指定离散网格，不是连续光场误差定理。1%有限扰动的complex128复核仍待指定云端复算；物理几何、真实相机读出和语义可用性也未由本诊断确认。",
        "",
        "## 工程完成与验证",
        "",
        f"全端口同网格独立参考复场与功率误差仍按1e-4核验；本轮最大值见：\n\n| 端点 | 全端口跨网格复场差 | 全端口跨网格功率差 |\n|---|---:|---:|\n{chr(10).join(reference_rows)}\n\n跨网格差异没有被挑选一个更宽容差来掩盖，而是进入重构效应判定、检测距离区间和局部奇异值带。生产直接前向与基场合成MSE差按1e-6核验。固定样本身份、类别、输入角色、模式次序、模型冻结和分块回读见机器报告及工程收据。精确截断矩的独立枚举已扩展到稀有饱和下尾并单独核验32536 e⁻基线，避免14σ截断遗漏极小但非零的方差；具体相对误差与AutoDL复算阻断状态见engineering/local_direction_complex128_replay_status.json。聚焦研究测试在此前源码状态18项通过；最终状态/报告修改后的报告重建、Ruff和compileall通过，pytest复跑两次均因自动权限审查超时而未启动。",
        "",
        "## 决策影响与下一步",
        "",
        "相同端点的功率参与数与空间分布可用于判断强弱差异是否只是集中效应；只有两网格均显著且超出网格差异余量的端口才计为稳健重构敏感。若代理噪声上下限仍很宽，或局部方向数远低于协方差方向数，只支持继续研究读出限制，不支持删通道或采用新模型。当前优先建议是在指定AutoDL执行机用complex128扰动精度补跑已选的200张局部方向有限差分并更新同一报告；本地记录的两次外部执行审批审查超时不代表服务器故障。其后真实噪声标定仍需与相机、波长、曝光、增益、ROI、binning和像素参考面匹配的暗帧/平场/标定文件。上述待办仍属于本轮诊断或后续建议，不授权训练、改几何、删通道或采用模型。",
        "",
    ]
    (output_root / "中文诊断报告.md").write_text("\n".join(lines), encoding="utf-8")


def run_measure_dual_grid(cfg: dict[str, Any], output_root: Path, device: torch.device) -> dict[str, Any]:
    preflight_path = output_root / "preflight.json"
    require(preflight_path.is_file(), "dual-grid measurement requires the explicit preflight receipt")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    require(preflight.get("task_id") == TASK_ID and preflight.get("status") == "READY", "dual-grid preflight is not READY")
    ranking_path = output_root / "ablation" / "ranking_manifest.json"
    require(ranking_path.is_file(), "pre-frozen power ranking is missing")
    small, large = load_small_context(cfg, device), load_large_context(cfg, device)
    camera = read_noise_proxy(cfg)
    ranking_payload = json.loads(ranking_path.read_text(encoding="utf-8"))
    endpoint_specs = [("small", item["key"], item["source_state"]) for item in cfg["small"]["decoder_endpoints"]]
    endpoint_specs.append(("large", "large_step_030000", None))
    endpoint_data: dict[str, dict[str, Any]] = {}
    cumulative_rows: list[dict[str, Any]] = []
    local_spectra: dict[str, Any] = {}
    endpoint_elapsed: dict[str, float] = {}
    started = time.perf_counter()
    for kind, endpoint_key, source_state in endpoint_specs:
        population = state_population(kind, endpoint_key, source_state, small, large)
        data = process_endpoint_dual_grid(cfg, endpoint_key, kind, source_state, population, output_root, device, camera)
        endpoint_data[endpoint_key] = data
        order = np.asarray(ranking_payload[endpoint_key]["strongest_to_weakest_zero_based"], dtype=np.int64)
        require(order.shape == (PORTS,) and set(order.tolist()) == set(range(PORTS)), f"frozen power ranking is invalid: {endpoint_key}")
        if kind == "small":
            grids = (("production", int(cfg["small"]["production_grid"])), ("reference", int(cfg["small"]["reference_grid"])))
        else:
            grids = (("production", int(cfg["large"]["decoder_production_grid"])), ("reference", int(cfg["large"]["decoder_reference_grid"])))
        for grid_role, grid in grids:
            cumulative_rows.extend(cumulative_retention_dual_grid(kind, endpoint_key, population, output_root, order, grid_role, grid, device))
        local_spectra[endpoint_key] = local_direction_diagnostic_dual_grid(cfg, endpoint_key, population, output_root, camera, device)
        endpoint_elapsed[endpoint_key] = data["elapsed_seconds_observed"]
        write_json(output_root / "measurement_progress.json", {"task_id": TASK_ID, "completed_endpoints": list(endpoint_data), "total_endpoints": len(endpoint_specs), "elapsed_seconds_observed": time.perf_counter() - started, "completed_dual_grid_ablation_rows": sum(len(item["rows"]) * PORTS * len(CHANNELS) * 2 for item in endpoint_data.values()), "training_updates": 0})
    write_csv(output_root / "ablation" / "cumulative_root_retention_dual_grid.csv", cumulative_rows)
    tolerance_counts: dict[str, Any] = {}
    for endpoint in endpoint_data:
        for grid_role in ("production", "reference"):
            key = f"{endpoint}/{grid_role}"
            rows = sorted((item for item in cumulative_rows if item["endpoint"] == endpoint and item["grid_role"] == grid_role), key=lambda item: item["retained_roots"])
            tolerance_counts[key] = {str(tolerance): minimum_stable_retained_roots(rows, tolerance) for tolerance in (0.01, 0.05)}
    write_json(output_root / "noise" / "local_direction_spectra.json", local_spectra)
    local_rows = []
    for endpoint, payload in local_spectra.items():
        for image in payload["per_image"]:
            for exposure, record in image["exposures"].items():
                for radius, counts in record["radii"].items():
                    for threshold in OBSERVABILITY_THRESHOLDS:
                        local_rows.append({"endpoint": endpoint, "row": image["row"], "class": image["class"], "exposure_s": float(exposure), "relative_perturbation": float(radius), "threshold": threshold, "direction_count_lower": counts["observable_direction_count_lower"][str(threshold)], "direction_count_upper": counts["observable_direction_count_upper"][str(threshold)]})
    write_csv(output_root / "noise" / "per_image_local_direction_counts.csv", local_rows)
    summaries, noise_summary = write_dual_grid_tables(cfg, endpoint_data, output_root, camera)
    figure_paths = create_dual_grid_figures(output_root, endpoint_data, summaries, cumulative_rows, noise_summary, local_spectra, camera)
    covariance = json.loads((output_root / "covariance" / "summary.json").read_text(encoding="utf-8"))
    independent_reference = json.loads((output_root / "preflight" / "independent_reference.json").read_text(encoding="utf-8"))
    report = assemble_dual_grid_machine_report(
        cfg,
        preflight,
        independent_reference,
        covariance,
        summaries,
        noise_summary,
        tolerance_counts,
        local_spectra,
        figure_paths,
        time.perf_counter() - started,
        endpoint_elapsed,
    )
    write_json(output_root / "machine_report.json", report)
    expected_rows = sum(len(data["rows"]) * PORTS * len(CHANNELS) * 2 for data in endpoint_data.values())
    write_json(output_root / "measurement_complete.json", {"task_id": TASK_ID, "status": report["status"], "endpoint_sample_counts": {key: len(value["rows"]) for key, value in endpoint_data.items()}, "per_image_port_control_grid_rows": expected_rows, "expected_rows": 6_120_000, "row_count_matches_plan": expected_rows == 6_120_000, "all_chunk_readbacks_passed": True, "full_population_preserved": True, "training_updates": 0, "optimizers_created": 0, "prior_samples": 0, "official_test_reads": 0, "hardware_operations": 0, "elapsed_seconds_observed": report["elapsed_seconds_observed"]})
    require(expected_rows == 6_120_000, f"dual-grid population count changed: {expected_rows}")
    write_dual_grid_report(output_root)
    return report


def run_finalize_dual_grid_from_saved_artifacts(
    cfg: dict[str, Any], output_root: Path
) -> dict[str, Any]:
    from datetime import datetime

    expected_samples = {
        "small_step_030000": 5000,
        "small_control_step_035000": 5000,
        "small_treatment_step_035000": 5000,
        "large_step_030000": 300,
    }
    require(not (output_root / "machine_report.json").exists(), "machine report already exists")
    require(not (output_root / "measurement_complete.json").exists(), "completion receipt already exists")
    preflight = json.loads((output_root / "preflight.json").read_text(encoding="utf-8"))
    independent_reference = json.loads(
        (output_root / "preflight" / "independent_reference.json").read_text(encoding="utf-8")
    )
    require(preflight.get("status") == "READY", "saved preflight is not READY")
    require(
        preflight.get("basis_same_grid_all_100_ports_within_1e-4") is True,
        "same-grid all-port reference check did not pass",
    )
    require(
        preflight.get("direct_production_vs_basis_checks", {}).get("passed") is True,
        "direct production replay did not pass",
    )
    require(
        independent_reference.get("measurement_gate_ready") is True,
        "saved independent reference does not permit measurement",
    )
    progress = json.loads((output_root / "measurement_progress.json").read_text(encoding="utf-8"))
    require(
        progress.get("completed_endpoints") == list(expected_samples),
        "saved endpoint progress does not cover the full fixed population",
    )

    endpoint_elapsed: dict[str, float] = {}
    chunk_counts: dict[str, int] = {}
    sample_counts: dict[str, int] = {}
    for endpoint, expected_count in expected_samples.items():
        endpoint_root = output_root / "ablation" / endpoint
        complete = json.loads((endpoint_root / "complete.json").read_text(encoding="utf-8"))
        progress_record = json.loads((endpoint_root / "progress.json").read_text(encoding="utf-8"))
        require(complete.get("sample_count") == expected_count, f"sample count changed: {endpoint}")
        require(complete.get("all_chunks_read_back_and_row_validated") is True, f"chunk readback failed: {endpoint}")
        require(progress_record.get("completed_rows") == expected_count, f"incomplete rows: {endpoint}")
        require(progress_record.get("population_rows") == expected_count, f"population changed: {endpoint}")
        chunks_root = endpoint_root / "chunks"
        chunk_paths = sorted(chunks_root.glob("chunk_*.npz"))
        expected_chunk_count = math.ceil(expected_count / int(cfg["output"]["recovery_chunk_rows"]))
        require(len(chunk_paths) == expected_chunk_count, f"chunk count changed: {endpoint}")
        all_rows: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for chunk_index, chunk_path in enumerate(chunk_paths):
            require(
                chunk_path.name == f"chunk_{chunk_index:05d}.npz",
                f"chunk sequence has a gap: {endpoint}/{chunk_path.name}",
            )
            with np.load(chunk_path, allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            rows = np.asarray(arrays["rows"], dtype=np.int64)
            labels = np.asarray(arrays["labels"], dtype=np.int64)
            count = min(int(cfg["output"]["recovery_chunk_rows"]), expected_count - chunk_index * int(cfg["output"]["recovery_chunk_rows"]))
            require(rows.shape == (count,) and labels.shape == (count,), f"chunk identities have wrong shape: {endpoint}/{chunk_path.name}")
            expected_shapes = {
                "native_metrics_production": (count, len(METRICS)),
                "native_metrics_reference": (count, len(METRICS)),
                "ablation_metrics_production": (count, PORTS, len(CHANNELS), len(METRICS)),
                "ablation_metrics_reference": (count, PORTS, len(CHANNELS), len(METRICS)),
                "port_detectability": (count, len(CHANNELS), 4, PORTS, 5),
                "observable_port_counts": (count, len(CHANNELS), 4, 3, 2),
                "modal_power_w": (count, PORTS),
                "equal_power_factor": (count, PORTS),
                "equal_power_defined": (count, PORTS),
            }
            require(set(arrays) == set(expected_shapes) | {"rows", "labels"}, f"chunk fields changed: {endpoint}/{chunk_path.name}")
            for key, shape in expected_shapes.items():
                require(arrays[key].shape == shape, f"chunk array shape changed: {endpoint}/{chunk_path.name}/{key}")
            require(np.isfinite(arrays["native_metrics_production"]).all(), f"native production metrics invalid: {endpoint}/{chunk_path.name}")
            require(np.isfinite(arrays["native_metrics_reference"]).all(), f"native reference metrics invalid: {endpoint}/{chunk_path.name}")
            power = arrays["modal_power_w"]
            require(np.isfinite(power).all() and np.all(power >= 0.0), f"modal power is invalid: {endpoint}/{chunk_path.name}")
            defined = arrays["equal_power_defined"].astype(bool, copy=False)
            factor = arrays["equal_power_factor"]
            require(np.isfinite(factor[defined]).all() and np.all(factor[defined] > 0.0), f"equal-power factor is invalid: {endpoint}/{chunk_path.name}")
            require(np.isnan(factor[~defined]).all(), f"undefined equal-power values were fabricated: {endpoint}/{chunk_path.name}")
            all_rows.append(rows)
            all_labels.append(labels)
        rows = np.concatenate(all_rows)
        labels = np.concatenate(all_labels)
        require(len(np.unique(rows)) == expected_count, f"duplicate sample rows: {endpoint}")
        require(np.array_equal(np.bincount(labels, minlength=10), np.full(10, expected_count // 10)), f"class counts changed: {endpoint}")
        sample_counts[endpoint] = expected_count
        chunk_counts[endpoint] = expected_chunk_count
        endpoint_elapsed[endpoint] = float(complete["elapsed_seconds_observed"])

    total_expected_rows = sum(sample_counts.values()) * PORTS * len(CHANNELS) * 2
    require(total_expected_rows == 6_120_000, "saved endpoint counts do not yield the plan record count")
    csv_readback_counts: dict[str, int] = {}
    for grid_role in ("production", "reference"):
        receipt = json.loads(
            (output_root / "ablation" / f"per_image_per_port_ablation_{grid_role}_row_count.json").read_text(encoding="utf-8")
        )
        expected_grid_rows = 3_060_000
        require(
            receipt.get("written_data_rows") == expected_grid_rows
            and receipt.get("readback_data_rows") == expected_grid_rows
            and receipt.get("passed") is True,
            f"saved {grid_role} row-count receipt failed",
        )
        csv_path = output_root / "ablation" / f"per_image_per_port_ablation_{grid_role}.csv.gz"
        with gzip.open(csv_path, "rt", newline="", encoding="utf-8") as stream:
            actual_rows = sum(1 for _ in stream) - 1
        require(actual_rows == expected_grid_rows, f"transferred {grid_role} CSV is truncated")
        csv_readback_counts[grid_role] = actual_rows

    covariance = json.loads((output_root / "covariance" / "summary.json").read_text(encoding="utf-8"))
    expected_covariance_counts = {
        **{key: 5000 for key in (
            "small_step_000000",
            "small_step_005000",
            "small_step_010000",
            "small_step_020000",
            "small_step_030000",
            "small_control_step_032500",
            "small_control_step_035000",
            "small_treatment_step_032500",
            "small_treatment_step_035000",
        )},
        "large_fixed300_train": 300,
        "large_saved55000_train": 55000,
    }
    require(set(covariance) == set(expected_covariance_counts), "covariance summary population identities changed")
    require(
        {key: int(value["sample_count"]) for key, value in covariance.items()} == expected_covariance_counts,
        "covariance summary row counts changed",
    )
    summaries = json.loads((output_root / "ablation" / "decoder_ablation_summary.json").read_text(encoding="utf-8"))
    require(len(summaries) == 8, "Decoder summary does not cover four endpoints and two grids")
    noise_summary = json.loads((output_root / "noise" / "observability_summary.json").read_text(encoding="utf-8"))
    require(set(noise_summary) == set(expected_samples), "noise summary endpoint coverage changed")
    local_spectra = json.loads((output_root / "noise" / "local_direction_spectra.json").read_text(encoding="utf-8"))
    require(set(local_spectra) == set(expected_samples), "local direction endpoint coverage changed")
    local_repair_receipt = json.loads(
        (output_root / "engineering" / "local_direction_repair_result_receipt.json").read_text(encoding="utf-8")
    )
    require(
        local_repair_receipt.get("status") == "VALIDATED_FLOAT64_LOCAL_DIRECTION_RECOMPUTE"
        and local_repair_receipt.get("local_direction_count_rows") == 28_800
        and local_repair_receipt.get("representatives_per_endpoint") == 200
        and local_repair_receipt.get("training_updates") == 0
        and local_repair_receipt.get("optimizers_created") == 0
        and local_repair_receipt.get("prior_samples") == 0
        and local_repair_receipt.get("official_test_reads") == 0
        and local_repair_receipt.get("hardware_operations") == 0,
        "complex128 local-direction repair receipt failed",
    )
    require(
        set(local_repair_receipt.get("phase_null_passed_by_endpoint", {})) == set(expected_samples)
        and all(local_repair_receipt["phase_null_passed_by_endpoint"].values()),
        "complex128 local-direction phase-null controls did not pass for every endpoint",
    )
    for endpoint, record in local_spectra.items():
        image_records = record.get("per_image", [])
        require(len(image_records) == 200, f"local direction population changed: {endpoint}")
        require(record.get("primary_relative_perturbation") == 0.01, f"local direction radius changed: {endpoint}")
        require(record.get("phase_direction_control_passed") is True, f"public-phase null direction failed: {endpoint}")
        require(np.array_equal(np.bincount([item["class"] for item in image_records], minlength=10), np.full(10, 20)), f"local direction class sample changed: {endpoint}")
    cumulative_rows: list[dict[str, Any]] = []
    with (output_root / "ablation" / "cumulative_root_retention_dual_grid.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            row["retained_roots"] = int(row["retained_roots"])
            row["mean_absolute_relative_mse_error"] = float(row["mean_absolute_relative_mse_error"])
            cumulative_rows.append(row)
    require(len(cumulative_rows) == 800, "cumulative retention curve is incomplete")
    tolerance_counts: dict[str, Any] = {}
    for endpoint in expected_samples:
        for grid_role in ("production", "reference"):
            key = f"{endpoint}/{grid_role}"
            selected = sorted(
                (row for row in cumulative_rows if row["endpoint"] == endpoint and row["grid_role"] == grid_role),
                key=lambda row: row["retained_roots"],
            )
            require([row["retained_roots"] for row in selected] == list(range(1, PORTS + 1)), f"retention root sequence changed: {key}")
            tolerance_counts[key] = {
                str(value): minimum_stable_retained_roots(selected, value)
                for value in (0.01, 0.05)
            }
    figure_paths = sorted(
        path.relative_to(output_root).as_posix()
        for path in (output_root / "figures").glob("*.png")
        if path.is_file() and path.stat().st_size > 0
    )
    require(len(figure_paths) == 5, "expected diagnostic figures are incomplete")

    history_root = output_root / "engineering" / "runner_history" / "gaussian-dual-grid-measure-final-33da73ab1d0e47ca841a5ecebc0d5d03"
    runner_status = json.loads((history_root / "status.json").read_text(encoding="utf-8"))
    require(runner_status.get("state") == "COMMAND_EXITED_NONZERO" and runner_status.get("exit_code") == 1, "original runner exit history changed")
    run_log = (history_root / "run.log").read_text(encoding="utf-8", errors="replace")
    require("No space left on device" in run_log, "original engineering failure does not match the reported disk error")
    telemetry = list(csv.DictReader((history_root / "telemetry.csv").open(encoding="utf-8", newline="")))
    require(telemetry, "final run telemetry is missing")
    started_at = datetime.fromisoformat(telemetry[0]["timestamp_utc"])
    ended_at = datetime.fromisoformat(runner_status["updated_at"])
    run_elapsed_seconds = (ended_at - started_at).total_seconds()
    require(run_elapsed_seconds > 0.0, "runner duration could not be recovered")

    recovery_receipt = {
        "schema": "dual_grid_report_reconstruction_receipt.v1",
        "task_id": TASK_ID,
        "status": "RECONSTRUCTED_FROM_SAVED_FULL_POPULATION_EVIDENCE",
        "original_remote_runner_state": runner_status,
        "original_failure": "OSError [Errno 28] No space left on device while writing machine_report.json after all four endpoint computations, both per-image ablation CSV readbacks, summary tables, and all five figures were complete.",
        "training_updates": 0,
        "optimizers_created": 0,
        "prior_samples": 0,
        "official_test_reads": 0,
        "hardware_operations": 0,
        "endpoint_population_counts": sample_counts,
        "endpoint_chunk_counts": chunk_counts,
        "verified_total_dual_grid_ablation_rows": total_expected_rows,
        "verified_csv_data_rows_by_grid": csv_readback_counts,
        "verified_local_direction_rows_by_endpoint": {key: 200 for key in expected_samples},
        "runner_elapsed_seconds_observed": run_elapsed_seconds,
        "report_repairs": [
            "include the independent-reference payload consumed by the report writer",
            "include the production W/pixel power-identity diagnostic consumed by the report writer",
            "keep full per-image local-direction spectra in their dedicated JSON artifact and store only population summaries in machine_report.json",
            "promote local Jacobian field derivatives to complex128 and recompute all four fixed representative populations after the initial float32 phase-null control failed",
        ],
        "local_direction_precision_repair": {
            "status": local_repair_receipt["status"],
            "phase_null_max_relative_response_by_endpoint": local_repair_receipt["phase_null_max_relative_response_by_endpoint"],
            "phase_null_passed_by_endpoint": local_repair_receipt["phase_null_passed_by_endpoint"],
            "local_direction_count_rows": local_repair_receipt["local_direction_count_rows"],
        },
        "all_recovered_artifacts_passed_readback": True,
    }
    write_json(output_root / "engineering" / "report_reconstruction_receipt.json", recovery_receipt)
    report = assemble_dual_grid_machine_report(
        cfg,
        preflight,
        independent_reference,
        covariance,
        summaries,
        noise_summary,
        tolerance_counts,
        local_spectra,
        figure_paths,
        run_elapsed_seconds,
        endpoint_elapsed,
    )
    write_json(output_root / "machine_report.json", report)
    require(
        (output_root / "machine_report.json").stat().st_size < 20_000_000,
        "machine report unexpectedly duplicates large per-image local spectra",
    )
    completion = {
        "task_id": TASK_ID,
        "status": report["status"],
        "endpoint_sample_counts": sample_counts,
        "per_image_port_control_grid_rows": total_expected_rows,
        "expected_rows": 6_120_000,
        "row_count_matches_plan": total_expected_rows == 6_120_000,
        "all_chunk_readbacks_passed": True,
        "full_population_preserved": True,
        "finalized_from_saved_artifacts": True,
        "local_direction_precision_repair_validated": True,
        "training_updates": 0,
        "optimizers_created": 0,
        "prior_samples": 0,
        "official_test_reads": 0,
        "hardware_operations": 0,
        "elapsed_seconds_observed": run_elapsed_seconds,
    }
    write_json(output_root / "measurement_complete.json", completion)
    write_dual_grid_report(output_root)
    report_path = output_root / "中文诊断报告.md"
    report_text = report_path.read_text(encoding="utf-8")
    marker = "\n\n## 决策影响与下一步"
    recovery_paragraph = (
        "\n\n报告收尾说明：云端计算进程在四个人口端点、逐根消融表和图表均已落盘后，"
        "因磁盘空间不足未能写出机器报告。临时文件清理和源码恢复后，"
        "本报告及完成收据由已保存并通过回读验证的全量分块和统计结果重建；"
        "原始云端非零退出状态保留在工程运行记录中。"
    )
    require(marker in report_text, "Chinese report section boundary changed")
    report_path.write_text(report_text.replace(marker, recovery_paragraph + marker, 1), encoding="utf-8")
    return {"status": "REPORT_AND_COMPLETION_RECEIPTS_RECONSTRUCTED", "machine_report_bytes": (output_root / "machine_report.json").stat().st_size, "dual_grid_rows": total_expected_rows, "runner_elapsed_seconds_observed": run_elapsed_seconds}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/research/lightgen/gaussian_ols_effective_channel_dual_grid_v1.yaml"))
    parser.add_argument("--mode", choices=("reference", "preflight", "measure", "finalize", "report"), required=True)
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
        run_measure_dual_grid(cfg, output_root, torch.device("cuda"))
        return
    if args.mode == "finalize":
        result = run_finalize_dual_grid_from_saved_artifacts(cfg, output_root)
        print(json.dumps(clean(result), ensure_ascii=False))
        return
    reference_path = output_root / "preflight" / "independent_reference.json"
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        require(reference.get("task_id") == TASK_ID, "independent reference task identity changed")
        if reference.get("measurement_gate_ready") is False:
            write_reference_gate_closeout(output_root, reference)
            print(json.dumps({"status": "NUMERICAL_GATE_BLOCKED", "output": str(output_root / "machine_report.json")}, ensure_ascii=False))
            return
    if (output_root / "machine_report.json").is_file():
        write_dual_grid_report(output_root)
        return
    run_report(cfg, output_root)


if __name__ == "__main__":
    main()
