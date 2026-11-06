"""Run a frozen-C phase-chart G1/G4 machine comparison on CUDA."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

import scripts.research.lightgen.audit_lightgen_g4_phase_reference_representation_v1 as audit
import scripts.research.lightgen.execute_lightgen_matched_architecture_v1 as base
import scripts.research.lightgen.execute_lightgen_small_conditional_mixture_prior_ab_v1 as prior
import scripts.research.lightgen.execute_lightgen_small_gaussian_replication_prior_confirmation_v1 as replication
from ldm_project.v2.application.research.lightgen.image_readout import resize_view


ROOT = Path(__file__).resolve().parents[3]
TASK_ID = "LIGHTGEN_C45K_PHASE_CHART_AB_V1"
SOURCE_TASK_ID = "LIGHTGEN_SMALL_GAUSSIAN_FOREGROUND_BALANCE_AB_V1"
SOURCE_STEP = 45_000
SOURCE_RUN_ID = "0a0ea0e11b334e0ebfb05674abe4149b"
SOURCE_SEED = 2026101501
TRAIN_COUNT = 55_000
TRAIN_LABEL_COUNTS = (5423, 6242, 5458, 5631, 5342, 4921, 5418, 5765, 5351, 5449)
N_CHANNELS = 100
FEATURE_DIM = 200
COMPONENTS = 4
STREAMS = 3
SAMPLE_COUNT = 1_000
MAX_PROPOSALS = 1_000_000
KMEANS_SEED_BASE = prior.KMEANS_SEED_BASE
RIDGE_RELATIVE = prior.POWER_RIDGE
G1_NORMAL_SEED_BASE = replication.G1_NORMAL_SEED_BASE
G4_NORMAL_SEED_BASE = replication.G4_NORMAL_SEED_BASE
G4_COMPONENT_SEED_BASE = replication.G4_COMPONENT_SEED_BASE
BOOTSTRAP_SEED = 2026102305
BOOTSTRAP_RESAMPLES = 2_000
SHAPE_PAIR_SEED_BASE = 2026102501
SHAPE_RATIO_MIN = 0.25
REPLAY_RTOL = replication.REPLAY_RTOL
REPLAY_ATOL = replication.REPLAY_ATOL
NUMPY_RTOL = 1e-10
NUMPY_ATOL = 1e-12
PROJECTION_MIN = 1e-8
REPORT_NAME = "report.json"
GENERATION_FIELDS = (
    "representation", "prior", "stream", "requested_label", "component_id", "proposal_id",
    "E_D_prediction", "E_C_prediction", "E_D_correct", "E_C_correct",
    "capture_power_w", "raw_mean_w_per_pixel", "raw_peak_w_per_pixel", "view_mean_intensity",
    "proxy_clipping_area_fraction", "proxy_clipped_excess_fraction",
)


class ExperimentError(RuntimeError):
    """The fixed input or measurement contract was not satisfied."""


def _require(condition: bool, message: str, error_type: type[Exception] = ExperimentError) -> None:
    if not condition:
        raise error_type(message)


def _path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _save_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _read_csv(path: Path, fields: tuple[str, ...]) -> list[dict[str, str]]:
    _require(path.is_file(), f"required CSV is missing: {path}", FileNotFoundError)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == fields, f"CSV fields changed: {path}")
        return list(reader)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _complex(features: torch.Tensor) -> torch.Tensor:
    _require(features.ndim == 2 and features.shape[1] == FEATURE_DIM and features.dtype == torch.float64,
             "normalized TRAIN features must be float64 with 200 real columns")
    return torch.view_as_complex(features.reshape(-1, N_CHANNELS, 2).contiguous())


def _canonicalize_template(reference: torch.Tensor) -> tuple[torch.Tensor, int, bool]:
    _require(reference.ndim == 1 and reference.numel() == N_CHANNELS and torch.is_complex(reference),
             "phase template must be a 100-port complex vector")
    magnitudes = reference.abs()
    _require(bool(torch.isfinite(magnitudes).all()) and float(magnitudes.max()) > 0.0,
             "phase template has no finite nonzero port")
    channel = int(torch.argmax(magnitudes))
    tied = int((magnitudes == magnitudes.max()).sum()) > 1
    phase = torch.angle(reference[channel])
    canonical = reference * torch.polar(torch.ones((), dtype=torch.float64, device=reference.device), -phase)
    _require(float(canonical[channel].real) > 0.0 and abs(float(canonical[channel].imag)) <= 1e-12,
             "canonical phase template did not make its first maximum port positive real")
    return canonical, channel, tied


def _template_and_transform(features: torch.Tensor, device: torch.device) -> tuple[dict[str, Any], torch.Tensor]:
    values = _complex(features).to(device=device, dtype=torch.complex128)
    hessian = values.T @ values.conj() / float(values.shape[0])
    hessian = (hessian + hessian.mH) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
    reference, channel, tied = _canonicalize_template(eigenvectors[:, -1])
    top = float(eigenvalues[-1].real)
    gap = float((eigenvalues[-1] - eigenvalues[-2]).real / max(abs(top), 1e-300))
    _require(math.isfinite(gap) and gap > PROJECTION_MIN,
             f"class template top eigenvalue relative gap is too small: {gap}")
    projection = values @ reference.conj()
    norm = torch.linalg.vector_norm(values, dim=1).clamp_min(1e-30)
    ratios = projection.abs() / norm
    minimum_projection = float(ratios.min())
    _require(math.isfinite(minimum_projection) and minimum_projection > PROJECTION_MIN,
             f"class template projection approaches zero: {minimum_projection}")
    unit_phase = projection / projection.abs()
    transformed = values * unit_phase.conj().unsqueeze(1)
    fixed_projection = transformed @ reference.conj()
    _require(float(fixed_projection.real.min()) > 0.0
             and float(fixed_projection.imag.abs().max()) <= 1e-12,
             "template transformation failed to put the reference projection on positive real axis")
    record = {
        "reference": reference.detach().cpu(),
        "eigenvalues": eigenvalues.real.detach().cpu(),
        "top_eigenvalue_relative_gap": gap,
        "minimum_normalized_reference_projection": minimum_projection,
        "canonical_reference_channel": channel,
        "canonical_reference_exact_tie": tied,
        "exact_maximum_port_ties": int(tied),
    }
    return record, transformed


def _transform_with_template(features: torch.Tensor, reference: torch.Tensor, device: torch.device) -> torch.Tensor:
    values = _complex(features).to(device=device, dtype=torch.complex128)
    r = reference.to(device=device, dtype=torch.complex128)
    projection = values @ r.conj()
    norm = torch.linalg.vector_norm(values, dim=1).clamp_min(1e-30)
    ratios = projection.abs() / norm
    _require(bool(torch.isfinite(ratios).all()) and float(ratios.min()) > PROJECTION_MIN,
             "TRAIN row has zero or near-zero template projection")
    transformed = values * (projection / projection.abs()).conj().unsqueeze(1)
    return torch.view_as_real(transformed).reshape(features.shape[0], FEATURE_DIM)


def _require_template_label(row_label: int, template_label: int) -> None:
    _require(row_label == template_label,
             f"class {row_label} row was supplied class {template_label} phase template")


def _numpy_template_and_transform(features: torch.Tensor) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    raw = features.detach().cpu().numpy().reshape(-1, N_CHANNELS, 2)
    values = raw[..., 0] + 1j * raw[..., 1]
    hessian = values.T @ values.conj() / float(values.shape[0])
    hessian = (hessian + hessian.conj().T) * 0.5
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    reference = eigenvectors[:, -1].astype(np.complex128, copy=True)
    magnitude = np.abs(reference)
    channel = int(np.flatnonzero(magnitude == magnitude.max())[0])
    tied = int(np.count_nonzero(magnitude == magnitude.max())) > 1
    reference *= np.exp(-1j * np.angle(reference[channel]))
    projection = values @ reference.conj()
    ratios = np.abs(projection) / np.maximum(np.linalg.norm(values, axis=1), 1e-30)
    _require(np.isfinite(ratios).all() and float(ratios.min()) > PROJECTION_MIN,
             "NumPy reference has zero or near-zero template projection")
    transformed = values * np.conj(projection / np.abs(projection))[:, None]
    top = float(eigenvalues[-1])
    return reference, transformed, {
        "top_eigenvalue_relative_gap": float((eigenvalues[-1] - eigenvalues[-2]) / max(abs(top), 1e-300)),
        "minimum_normalized_reference_projection": float(ratios.min()),
        "canonical_reference_channel": channel,
        "canonical_reference_exact_tie": tied,
    }


def _numpy_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    covariance = centered.T @ centered / float(values.shape[0] - 1)
    ridge = float(np.diag(covariance).mean() * RIDGE_RELATIVE)
    cholesky = np.linalg.cholesky(covariance + ridge * np.eye(values.shape[1], dtype=np.float64))
    return mean, covariance, ridge, cholesky


def _fit_gaussian_cuda(values: torch.Tensor) -> dict[str, torch.Tensor]:
    _require(values.is_cuda and values.dtype == torch.float64 and values.ndim == 2,
             "formal prior moments must be computed from CUDA float64 features")
    mean = values.mean(0)
    centered = values - mean
    covariance = centered.T @ centered / float(values.shape[0] - 1)
    ridge = covariance.diag().mean() * RIDGE_RELATIVE
    _require(bool(torch.isfinite(covariance).all()) and float(ridge) > 0.0,
             "Gaussian covariance or ridge is invalid")
    regularized = covariance + ridge * torch.eye(values.shape[1], dtype=torch.float64, device=values.device)
    cholesky, info = torch.linalg.cholesky_ex(regularized)
    _require(int(info.max()) == 0 and bool(torch.isfinite(cholesky).all()),
             "CUDA Gaussian covariance Cholesky factorization failed")
    return {"mean": mean, "covariance": covariance, "ridge": ridge,
            "regularized_covariance": regularized, "cholesky": cholesky}


def _squared_distances_cuda(values: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    return (values.square().sum(1, keepdim=True) + centers.square().sum(1)[None, :]
            - 2.0 * values @ centers.T).clamp_min_(0.0)


def _kmeans_cuda(
    values: torch.Tensor, components: int, seed: int, maximum_iterations: int = prior.KMEANS_MAX_ITERATIONS,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    _require(values.is_cuda and values.dtype == torch.float64 and values.ndim == 2
             and values.shape[0] >= components, "fixed K-means input or device is invalid")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(values.shape[0], (1,), generator=generator).item())
    centers = [values[first]]
    nearest = _squared_distances_cuda(values, torch.stack(centers)).squeeze(1)
    for _ in range(1, components):
        mass = nearest.sum()
        _require(bool(torch.isfinite(mass)) and float(mass) > 0.0,
                 "K-means++ cannot select a distinct center from zero remaining distance")
        selected = int(torch.multinomial(nearest.detach().cpu(), 1, replacement=True, generator=generator).item())
        centers.append(values[selected])
        new_distance = _squared_distances_cuda(values, values[selected:selected + 1]).squeeze(1)
        nearest = torch.minimum(nearest, new_distance)
    center_tensor = torch.stack(centers)
    previous: torch.Tensor | None = None
    converged = False
    labels = torch.empty(values.shape[0], dtype=torch.long, device=values.device)
    for iteration in range(1, maximum_iterations + 1):
        labels = _squared_distances_cuda(values, center_tensor).argmin(1)
        counts = torch.bincount(labels, minlength=components)
        _require(bool((counts > 0).all()), "K-means produced an empty cluster")
        next_centers = torch.stack([values[labels == component].mean(0) for component in range(components)])
        if previous is not None and torch.equal(labels, previous):
            center_tensor = next_centers
            converged = True
            break
        previous = labels.clone()
        center_tensor = next_centers
    _require(converged, f"CUDA K-means failed to converge within {maximum_iterations} iterations")
    final_labels = _squared_distances_cuda(values, center_tensor).argmin(1)
    _require(torch.equal(final_labels, labels), "CUDA K-means assignment changed after convergence")
    return final_labels, center_tensor, iteration


def _fit_template_priors(
    transformed_by_class: dict[int, torch.Tensor], device: torch.device,
) -> tuple[dict[str, Any], dict[int, torch.Tensor]]:
    g1: dict[int, dict[str, Any]] = {}
    g4: dict[int, dict[str, Any]] = {}
    assignments: dict[int, torch.Tensor] = {}
    parity: list[dict[str, Any]] = []
    for label in range(10):
        features = transformed_by_class[label].to(device=device, dtype=torch.float64)
        _require(tuple(features.shape) == (TRAIN_LABEL_COUNTS[label], FEATURE_DIM),
                 f"template TRAIN class {label} population is wrong")
        fit1 = _fit_gaussian_cuda(features)
        cluster_ids, means, iterations = _kmeans_cuda(
            features, COMPONENTS, KMEANS_SEED_BASE + label, prior.KMEANS_MAX_ITERATIONS,
        )
        counts = torch.bincount(cluster_ids, minlength=COMPONENTS)
        _require(bool((counts >= 2).all()), f"template G4 class {label} has a cluster with fewer than two rows")
        covariances: list[torch.Tensor] = []
        regularized: list[torch.Tensor] = []
        choleskies: list[torch.Tensor] = []
        identity = torch.eye(FEATURE_DIM, dtype=torch.float64, device=device)
        for component in range(COMPONENTS):
            selected = features[cluster_ids == component]
            centered = selected - means[component]
            covariance = centered.T @ centered / float(selected.shape[0] - 1)
            cov = covariance + fit1["ridge"] * identity
            chol, info = torch.linalg.cholesky_ex(cov)
            _require(int(info.max()) == 0 and bool(torch.isfinite(covariance).all())
                     and bool(torch.isfinite(chol).all()),
                     f"template G4 class {label} component {component} covariance is invalid")
            covariances.append(covariance)
            regularized.append(cov)
            choleskies.append(chol)
        fit4 = {
            "weights": counts.to(torch.float64) / float(features.shape[0]),
            "means": means,
            "covariances": torch.stack(covariances),
            "regularized_covariances": torch.stack(regularized),
            "choleskies": torch.stack(choleskies),
            "ridge": fit1["ridge"], "counts": counts,
            "assignments": cluster_ids, "iterations": iterations,
            "seed": KMEANS_SEED_BASE + label,
        }
        g1[label] = fit1
        g4[label] = fit4
        assignments[label] = cluster_ids.detach().cpu()

        # Independent NumPy moment references consume the same transformed rows and fixed CUDA assignments.
        cpu_values = features.detach().cpu().numpy()
        mean_np, cov_np, ridge_np, _ = _numpy_moments(cpu_values)
        mean1_cuda = fit1["mean"].detach().cpu().numpy()
        cov1_cuda = fit1["covariance"].detach().cpu().numpy()
        chol1_cuda = fit1["cholesky"].detach().cpu().numpy()
        ridge1_cuda = float(fit1["ridge"].detach().cpu())
        g1_factor_reconstructs = np.allclose(
            chol1_cuda @ chol1_cuda.T, cov_np + ridge_np * np.eye(FEATURE_DIM),
            rtol=NUMPY_RTOL, atol=NUMPY_ATOL,
        )
        _require(np.allclose(mean1_cuda, mean_np, rtol=NUMPY_RTOL, atol=NUMPY_ATOL)
                 and np.allclose(cov1_cuda, cov_np, rtol=NUMPY_RTOL, atol=NUMPY_ATOL)
                 and math.isclose(ridge1_cuda, ridge_np, rel_tol=NUMPY_RTOL, abs_tol=NUMPY_ATOL)
                 and g1_factor_reconstructs,
                 f"CUDA/NumPy G1 mean/covariance/ridge parity failed for class {label}; "
                 f"max_mean_error={np.max(np.abs(mean1_cuda - mean_np)):.6g}, "
                 f"max_covariance_error={np.max(np.abs(cov1_cuda - cov_np)):.6g}, "
                 f"ridge_error={abs(ridge1_cuda - ridge_np):.6g}, "
                 f"cholesky_reconstruction={g1_factor_reconstructs}")
        for component in range(COMPONENTS):
            values_np = cpu_values[cluster_ids.detach().cpu().numpy() == component]
            mean4_np, cov4_np, _, _ = _numpy_moments(values_np)
            mean4_cuda = means[component].detach().cpu().numpy()
            cov4_cuda = covariances[component].detach().cpu().numpy()
            chol4_cuda = choleskies[component].detach().cpu().numpy()
            ridge4_np = ridge1_cuda
            g4_factor_reconstructs = np.allclose(
                chol4_cuda @ chol4_cuda.T, cov4_np + ridge4_np * np.eye(FEATURE_DIM),
                rtol=NUMPY_RTOL, atol=NUMPY_ATOL,
            )
            _require(np.allclose(mean4_cuda, mean4_np, rtol=NUMPY_RTOL, atol=NUMPY_ATOL)
                     and np.allclose(cov4_cuda, cov4_np, rtol=NUMPY_RTOL, atol=NUMPY_ATOL)
                     and g4_factor_reconstructs,
                     f"CUDA/NumPy G4 mean/covariance parity failed for class {label}, "
                     f"component {component}; max_mean_error={np.max(np.abs(mean4_cuda - mean4_np)):.6g}, "
                     f"max_covariance_error={np.max(np.abs(cov4_cuda - cov4_np)):.6g}, "
                     f"cholesky_reconstruction={g4_factor_reconstructs}")
        parity.append({"label": label, "g1_mean_covariance_ridge_cholesky": True,
                       "g4_component_means_covariances_choleskies": True,
                       "kmeans_iterations": iterations, "component_counts": counts.detach().cpu().tolist()})
    _require(set(g1) == set(g4) == set(range(10)) and len(parity) == 10,
             "template prior fitting omitted a class")
    return {"G1": g1, "G4": g4, "numpy_moment_parity": parity}, assignments


def _seeded_sample_class(
    method: str, fit: dict[str, Any], scale: torch.Tensor, label: int, stream: int,
    sample_count: int, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    if method == "G1":
        normal_seed = G1_NORMAL_SEED_BASE + 100 * stream + label
        component_seed = None
        normal_generator = torch.Generator(device="cpu").manual_seed(normal_seed)
        component_generator = None
    elif method == "G4":
        normal_seed = G4_NORMAL_SEED_BASE + 100 * stream
        component_seed = G4_COMPONENT_SEED_BASE + 100 * stream + label
        normal_generator = torch.Generator(device="cpu").manual_seed(normal_seed + label)
        component_generator = torch.Generator(device="cpu").manual_seed(component_seed)
    else:
        raise ExperimentError(f"unknown prior method: {method}")

    accepted_coefficients: list[torch.Tensor] = []
    accepted_components: list[torch.Tensor] = []
    accepted_proposals: list[torch.Tensor] = []
    evaluated = 0
    retained = 0
    rejected = 0
    proposed_by_component = torch.zeros(COMPONENTS, dtype=torch.long)
    rejected_by_component = torch.zeros(COMPONENTS, dtype=torch.long)
    accepted_by_component = torch.zeros(COMPONENTS, dtype=torch.long)
    scale_device = scale.to(device=device, dtype=torch.float64).sqrt()
    while retained < sample_count:
        budget = MAX_PROPOSALS - evaluated
        _require(budget > 0, f"{method} class {label} stream {stream} exceeded its proposal cap")
        count = min(256, budget)
        noise_cpu = torch.randn((count, FEATURE_DIM), generator=normal_generator, dtype=torch.float64)
        noise = noise_cpu.to(device=device)
        if method == "G1":
            draws = fit["mean"].to(device) + noise @ fit["cholesky"].to(device).T
            components = torch.zeros(count, dtype=torch.long, device=device)
        else:
            weights_cpu = fit["weights"].detach().cpu().double()
            components = torch.multinomial(weights_cpu, count, replacement=True, generator=component_generator).to(device)
            draws = torch.empty_like(noise)
            means = fit["means"].to(device)
            choleskies = fit["choleskies"].to(device)
            for component in range(COMPONENTS):
                mask = components == component
                if bool(mask.any()):
                    draws[mask] = means[component] + noise[mask] @ choleskies[component].T
        normalized = torch.view_as_complex(draws.reshape(count, N_CHANNELS, 2).contiguous())
        coefficients = normalized * scale_device.unsqueeze(0)
        powers = coefficients.abs().square().sum(1)
        valid = powers <= base.MAX_MODAL_POWER_W
        valid_positions = torch.nonzero(valid, as_tuple=False).flatten()
        if method == "G4":
            proposed_by_component += torch.bincount(components.detach().cpu(), minlength=COMPONENTS)
            rejected_by_component += torch.bincount(components[~valid].detach().cpu(), minlength=COMPONENTS)
        need = sample_count - retained
        selected = valid_positions[:need]
        if selected.numel():
            selected_components = components[selected]
            accepted_coefficients.append(coefficients[selected].to(torch.complex64).detach().cpu())
            accepted_components.append(selected_components.detach().cpu().long())
            accepted_proposals.append((selected + evaluated).detach().cpu().long())
            if method == "G4":
                accepted_by_component += torch.bincount(selected_components.detach().cpu(), minlength=COMPONENTS)
            retained += int(selected.numel())
        rejected += int((~valid).sum())
        evaluated += count
    coefficients_out = torch.cat(accepted_coefficients)
    components_out = torch.cat(accepted_components)
    proposals_out = torch.cat(accepted_proposals)
    _require(tuple(coefficients_out.shape) == (sample_count, N_CHANNELS)
             and torch.unique(proposals_out).numel() == sample_count
             and bool((coefficients_out.abs().square().sum(1) <= base.MAX_MODAL_POWER_W).all()),
             f"{method} generated coefficients failed power or proposal validation")
    record = {
        "label": label, "stream": stream, "method": method,
        "normal_seed": normal_seed if method == "G1" else normal_seed + label,
        "component_seed": component_seed,
        "requested_samples": sample_count, "retained_samples": retained,
        "proposals_evaluated": evaluated, "proposal_limit": MAX_PROPOSALS,
        "power_rejected_candidates": rejected,
        "rejection_fraction": rejected / evaluated,
        "proposed_by_component": proposed_by_component.tolist() if method == "G4" else None,
        "power_rejected_by_component": rejected_by_component.tolist() if method == "G4" else None,
        "retained_by_component": accepted_by_component.tolist() if method == "G4" else None,
        "power_bound_w": base.MAX_MODAL_POWER_W,
    }
    return coefficients_out, components_out, proposals_out, record


def _validate_source_cohort(payload: dict[str, Any], method: str, stream: int) -> None:
    _require(payload.get("schema") == "lightgen_small_conditional_mixture_generated_cohort.v1"
             and payload.get("task_id") == replication.TASK_ID
             and payload.get("model") == "C_seed_2026101501"
             and payload.get("method") == method and payload.get("stream") == stream
             and payload.get("global_step") == SOURCE_STEP,
             f"saved original {method} stream {stream} has the wrong endpoint or task identity")
    fields = {
        "requested_labels": (10_000,), "component_ids": (10_000,), "proposal_ids": (10_000,),
        "ols_coefficients": (10_000, 100), "ed_logits": (10_000, 10), "ec_logits": (10_000, 10),
        "proxy_clipping_area_fraction": (10_000,), "proxy_clipped_excess_fraction": (10_000,),
    }
    for name, shape in fields.items():
        value = payload.get(name)
        _require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
                 f"saved original {method} stream {stream} omitted or malformed {name}")
    labels = payload["requested_labels"]
    _require(torch.equal(torch.bincount(labels, minlength=10), torch.full((10,), SAMPLE_COUNT))
             and bool(torch.isfinite(payload["ols_coefficients"].real).all())
             and bool(torch.isfinite(payload["ols_coefficients"].imag).all())
             and bool(torch.isfinite(payload["ed_logits"]).all())
             and bool(torch.isfinite(payload["ec_logits"]).all()),
             f"saved original {method} stream {stream} has invalid rows")
    for label in range(10):
        selected = labels == label
        _require(torch.unique(payload["proposal_ids"][selected]).numel() == SAMPLE_COUNT,
                 f"saved original {method} stream {stream} class {label} repeats proposal IDs")
    if method == "G4":
        _require(bool(((payload["component_ids"] >= 0) & (payload["component_ids"] < COMPONENTS)).all()),
                 f"saved original G4 stream {stream} has invalid component IDs")
    else:
        _require(bool((payload["component_ids"] == 0).all()),
                 f"saved original G1 stream {stream} has nonzero component IDs")


def _load_source_cohorts(cohort_dir: Path, prior_report: Path) -> tuple[dict[tuple[str, int], dict[str, Any]], str]:
    _require(prior_report.is_file(), f"historical prior report is missing: {prior_report}", FileNotFoundError)
    report = json.loads(prior_report.read_text(encoding="utf-8"))
    run_identity_path = prior_report.parent / "run_identity.json"
    row_report_path = prior_report.parent / "generation_per_image_metrics.csv"
    _require(run_identity_path.is_file() and row_report_path.is_file(),
             "historical generation run identity or row-level report is missing", FileNotFoundError)
    run_identity = json.loads(run_identity_path.read_text(encoding="utf-8"))
    _require(report.get("task_id") == replication.TASK_ID and report.get("status") == "COMPLETE_DIAGNOSTIC"
             and report.get("decision", {}).get("complete_valid_measurement") is True
             and run_identity.get("task_id") == replication.TASK_ID
             and run_identity.get("run_id")
             and report.get("endpoints", {}).get("C_seed_2026101501", {}).get("cumulative_step") == SOURCE_STEP
             and report.get("endpoints", {}).get("C_seed_2026101501", {}).get("source_status") == "VALID_FAILED",
             "historical C G1/G4 prior report is incomplete or mismatched")
    prior_run_id = str(run_identity["run_id"])
    input_paths = run_identity.get("input_paths", {})
    source_checkpoint_record = str(input_paths.get("source_checkpoint", "")).replace("\\", "/")
    prior_evidence_record = str(input_paths.get("prior_evidence_dir", "")).replace("\\", "/")
    _require(source_checkpoint_record.endswith(
                 "outputs/20261021_lightgen_small_gaussian_foreground_balance_ab_v1/"
                 "runs/autodl_backup_attempt_01/C/checkpoint_step_010000.pt")
             and prior_evidence_record.endswith(
                 "outputs/20261021_lightgen_small_conditional_mixture_prior_ab_v1/"
                 "runs/autodl_backup_attempt_01"),
             "historical run identity does not cite the explicit C checkpoint and saved prior evidence")
    _require(cohort_dir.resolve() == (prior_report.parent / "generation" / "C_seed_2026101501").resolve(),
             "historical source cohort directory does not belong to the explicit completed report")
    row_fields = audit.GENERATION_FIELDS
    saved_rows = _read_csv(row_report_path, row_fields)
    c_rows = [row for row in saved_rows if row["model"] == "C_seed_2026101501"]
    _require(len(saved_rows) == 180_000 and len(c_rows) == 60_000,
             "historical generation CSV does not contain the complete 60,000-row C source population")
    row_lookup: dict[tuple[str, int, str, int, int], dict[str, str]] = {}
    for row in c_rows:
        key = (row["model"], int(row["stream"]), row["method"],
               int(row["requested_label"]), int(row["proposal_id"]))
        _require(key not in row_lookup, "historical C generation CSV repeats a row identity")
        row_lookup[key] = row
    cohorts: dict[tuple[str, int], dict[str, Any]] = {}
    for method in ("G1", "G4"):
        for stream in range(STREAMS):
            path = cohort_dir / f"{method}_stream_{stream}.pt"
            _require(path.is_file(), f"saved original source cohort is missing: {path}", FileNotFoundError)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            _require(isinstance(payload, dict), f"saved source cohort is malformed: {path}")
            _validate_source_cohort(payload, method, stream)
            arrays = _saved_metric_arrays(payload)
            for index, label in enumerate(arrays["label"]):
                key = ("C_seed_2026101501", stream, method, int(label), int(arrays["proposal"][index]))
                row = row_lookup.get(key)
                _require(row is not None, f"historical row report omits saved {method} stream {stream} row {key}")
                expected = {
                    "component_id": int(arrays["component"][index]),
                    "E_D_prediction": int(arrays["ed_prediction"][index]),
                    "E_C_prediction": int(arrays["ec_prediction"][index]),
                    "E_D_correct": int(arrays["ed_correct"][index]),
                    "E_C_correct": int(arrays["ec_correct"][index]),
                }
                for field, value in expected.items():
                    _require(int(row[field]) == value,
                             f"saved cohort and historical row report disagree for {field} at {key}")
                for field in ("capture_power_w", "raw_mean_w_per_pixel", "raw_peak_w_per_pixel",
                              "view_mean_intensity", "proxy_clipping_area_fraction",
                              "proxy_clipped_excess_fraction"):
                    _require(math.isclose(float(row[field]), float(arrays[field][index]),
                                          rel_tol=1e-12, abs_tol=1e-15),
                             f"saved cohort and historical row report disagree for {field} at {key}")
            cohorts[(method, stream)] = payload
    return cohorts, prior_run_id


def _metric_arrays(payload: dict[str, Any]) -> dict[str, np.ndarray]:
    labels = payload["requested_labels"].detach().cpu().numpy().astype(np.int64, copy=False)
    ed_logits = payload["ed_logits"].detach().cpu().numpy()
    ec_logits = payload["ec_logits"].detach().cpu().numpy()
    return {
        "label": labels,
        "component": payload["component_ids"].detach().cpu().numpy().astype(np.int64, copy=False),
        "proposal": payload["proposal_ids"].detach().cpu().numpy().astype(np.int64, copy=False),
        "ed_prediction": ed_logits.argmax(1).astype(np.int64),
        "ec_prediction": ec_logits.argmax(1).astype(np.int64),
        "ed_correct": (ed_logits.argmax(1) == labels).astype(np.float64),
        "ec_correct": (ec_logits.argmax(1) == labels).astype(np.float64),
        "capture_power_w": payload["capture_power_w"].detach().cpu().numpy().astype(np.float64),
        "raw_mean_w_per_pixel": payload["raw_mean_w_per_pixel"].detach().cpu().numpy().astype(np.float64),
        "raw_peak_w_per_pixel": payload["raw_peak_w_per_pixel"].detach().cpu().numpy().astype(np.float64),
        "view_mean_intensity": payload["view_mean_intensity"].detach().cpu().numpy().astype(np.float64),
        "proxy_clipping_area_fraction": payload["proxy_clipping_area_fraction"].detach().cpu().numpy().astype(np.float64),
        "proxy_clipped_excess_fraction": payload["proxy_clipped_excess_fraction"].detach().cpu().numpy().astype(np.float64),
    }


_saved_metric_arrays = _metric_arrays


def _metrics_from_predictions(
    labels: torch.Tensor, ed_logits: torch.Tensor, ec_logits: torch.Tensor,
    coefficients: torch.Tensor, raw: torch.Tensor, scaled: torch.Tensor, views: torch.Tensor,
) -> dict[str, torch.Tensor]:
    ed_prediction = ed_logits.argmax(1)
    ec_prediction = ec_logits.argmax(1)
    return {
        "ed_prediction": ed_prediction.detach().cpu(), "ec_prediction": ec_prediction.detach().cpu(),
        "ed_correct": (ed_prediction == labels.to(ed_prediction.device)).detach().cpu(),
        "ec_correct": (ec_prediction == labels.to(ec_prediction.device)).detach().cpu(),
        "capture_power_w": raw.double().sum((1, 2, 3)).detach().cpu(),
        "raw_mean_w_per_pixel": raw.double().mean((1, 2, 3)).detach().cpu(),
        "raw_peak_w_per_pixel": raw.double().amax((1, 2, 3)).detach().cpu(),
        "view_mean_intensity": views.double().mean((1, 2, 3)).detach().cpu(),
        "proxy_clipping_area_fraction": ((scaled < 0.0) | (scaled > 1.0)).double().mean((1, 2, 3)).detach().cpu(),
        "proxy_clipped_excess_fraction": (scaled.double().sub(1.0).clamp_min(0.0).sum((1, 2, 3))
                                           / scaled.double().sum((1, 2, 3)).clamp_min(1e-30)).detach().cpu(),
    }


def _decode_coefficients(
    model: torch.nn.Module, evaluator_d: torch.nn.Module, evaluator_c: torch.nn.Module,
    coefficients: torch.Tensor, labels: torch.Tensor, gain: torch.Tensor,
    device: torch.device, capture_first_per_class: int = 10,
) -> dict[str, Any]:
    _require(torch.is_complex(coefficients) and tuple(coefficients.shape) == (labels.numel(), N_CHANNELS),
             "Decoder input coefficients and requested labels are misaligned")
    _require(bool(torch.isfinite(coefficients.real).all()) and bool(torch.isfinite(coefficients.imag).all()),
             "Decoder input coefficients are nonfinite")
    count = labels.numel()
    _require(torch.equal(torch.bincount(labels, minlength=10), torch.full((10,), count // 10)),
             "Decoder population is incomplete or class-imbalanced")
    image_parts: list[torch.Tensor] = []
    ed_parts: list[torch.Tensor] = []
    ec_parts: list[torch.Tensor] = []
    metric_parts: dict[str, list[torch.Tensor]] = {
        name: [] for name in ("capture_power_w", "raw_mean_w_per_pixel", "raw_peak_w_per_pixel",
                              "view_mean_intensity", "proxy_clipping_area_fraction",
                              "proxy_clipped_excess_fraction")
    }
    native_captures: dict[int, torch.Tensor] = {}
    capture_indices: list[int] = []
    capture_counts = [0] * 10
    model.eval()
    evaluator_d.eval()
    evaluator_c.eval()
    with torch.inference_mode():
        for start in range(0, count, base.BATCH_SIZE):
            stop = min(start + base.BATCH_SIZE, count)
            output = model.forward_from_ols(coefficients[start:stop].to(device))
            raw = output["physical_w_per_pixel"].float()
            scaled = raw * gain.to(device=device, dtype=raw.dtype)
            views = resize_view(scaled)
            ed_logits = evaluator_d(views)
            ec_logits = prior._ec_logits(evaluator_c, views)
            _require(torch.isfinite(views).all().item() and torch.isfinite(ed_logits).all().item()
                     and torch.isfinite(ec_logits).all().item(),
                     "frozen Decoder/E_D/E_C returned nonfinite values")
            image_parts.append(views.detach().cpu())
            ed_parts.append(ed_logits.detach().float().cpu())
            ec_parts.append(ec_logits.detach().float().cpu())
            scalar = _metrics_from_predictions(labels[start:stop], ed_logits, ec_logits,
                                               coefficients[start:stop], raw, scaled, views)
            for name in metric_parts:
                metric_parts[name].append(scalar[name])
            labels_cpu = labels[start:stop]
            for local_index, label_value in enumerate(labels_cpu.tolist()):
                label = int(label_value)
                if capture_counts[label] < capture_first_per_class:
                    row = start + local_index
                    native_captures[row] = scaled[local_index].clamp(0.0, 1.0).detach().float().cpu()
                    capture_indices.append(row)
                    capture_counts[label] += 1
            if stop % 1_000 == 0 or stop == count:
                torch.cuda.synchronize(device)
                print(json.dumps({"phase": "CUDA_DECODER", "rows": int(stop), "total_rows": int(count)}),
                      flush=True)
    images = torch.cat(image_parts)
    ed_logits_all = torch.cat(ed_parts)
    ec_logits_all = torch.cat(ec_parts)
    metrics = {name: torch.cat(parts) for name, parts in metric_parts.items()}
    metrics["ed_prediction"] = ed_logits_all.argmax(1)
    metrics["ec_prediction"] = ec_logits_all.argmax(1)
    metrics["ed_correct"] = metrics["ed_prediction"].eq(labels)
    metrics["ec_correct"] = metrics["ec_prediction"].eq(labels)
    return {
        "images": images, "ed_logits": ed_logits_all, "ec_logits": ec_logits_all,
        "metrics": metrics,
        "native_capture_indices": torch.tensor(capture_indices, dtype=torch.long),
        "native_capture_images": (torch.stack([native_captures[index] for index in sorted(capture_indices)])
                                  if capture_indices else torch.empty((0, 1, 200, 200), dtype=torch.float32)),
    }


def _decode_new_group(
    method: str, stream: int, fits: dict[int, dict[str, Any]], scale: torch.Tensor,
    model: torch.nn.Module, evaluator_d: torch.nn.Module, evaluator_c: torch.nn.Module,
    gain: torch.Tensor, device: torch.device,
) -> tuple[dict[str, Any], dict[str, Any]]:
    coefficients: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    components: list[torch.Tensor] = []
    proposal_ids: list[torch.Tensor] = []
    accounting: list[dict[str, Any]] = []
    for label in range(10):
        fit = fits[label][method]
        values, assigned, proposals, sampling = _seeded_sample_class(
            method, fit, scale, label, stream, SAMPLE_COUNT, device,
        )
        coefficients.append(values)
        components.append(assigned)
        proposal_ids.append(proposals)
        labels.append(torch.full((SAMPLE_COUNT,), label, dtype=torch.long))
        accounting.append(sampling)
    coeff = torch.cat(coefficients)
    requested = torch.cat(labels)
    component_ids = torch.cat(components)
    proposal = torch.cat(proposal_ids)
    decoded = _decode_coefficients(model, evaluator_d, evaluator_c, coeff, requested, gain, device)
    metrics = decoded["metrics"]
    cohort = {
        "schema": "lightgen_c45k_phase_chart_generated_cohort.v1",
        "task_id": TASK_ID, "source_endpoint": "C_seed_2026101501",
        "source_run_id": SOURCE_RUN_ID, "source_global_step": SOURCE_STEP,
        "representation": "class_fixed_principal_reference_template",
        "method": method, "stream": stream, "rows": SAMPLE_COUNT * 10,
        "requested_labels": requested,
        "component_ids": component_ids,
        "proposal_ids": proposal,
        "ols_coefficients": coeff,
        "images": decoded["images"],
        "ed_logits": decoded["ed_logits"], "ec_logits": decoded["ec_logits"],
        "ed_predictions": metrics["ed_prediction"], "ec_predictions": metrics["ec_prediction"],
        "ed_correct": metrics["ed_correct"], "ec_correct": metrics["ec_correct"],
        "capture_power_w": metrics["capture_power_w"],
        "raw_mean_w_per_pixel": metrics["raw_mean_w_per_pixel"],
        "raw_peak_w_per_pixel": metrics["raw_peak_w_per_pixel"],
        "view_mean_intensity": metrics["view_mean_intensity"],
        "proxy_clipping_area_fraction": metrics["proxy_clipping_area_fraction"],
        "proxy_clipped_excess_fraction": metrics["proxy_clipped_excess_fraction"],
        "native_capture_indices": decoded["native_capture_indices"],
        "native_capture_images": decoded["native_capture_images"],
        "sampling_accounting": accounting,
        "classifier_based_sample_selection": False,
        "per_image_normalization": False,
        "modal_power_bound_w": base.MAX_MODAL_POWER_W,
    }
    _validate_new_cohort(cohort, method, stream)
    return cohort, {"method": method, "stream": stream, "rows": len(requested),
                    "sampling_accounting": accounting}


def _validate_new_cohort(
    payload: dict[str, Any], method: str, stream: int, *, require_cuda_event: bool = False,
) -> None:
    _require(payload.get("schema") == "lightgen_c45k_phase_chart_generated_cohort.v1"
             and payload.get("task_id") == TASK_ID
             and payload.get("source_endpoint") == "C_seed_2026101501"
             and payload.get("source_global_step") == SOURCE_STEP
             and payload.get("representation") == "class_fixed_principal_reference_template"
             and payload.get("method") == method and payload.get("stream") == stream
             and payload.get("rows") == 10_000,
             "generated cohort endpoint, representation, method or stream identity is mixed")
    tensors = {
        "requested_labels": (10_000,), "component_ids": (10_000,), "proposal_ids": (10_000,),
        "ols_coefficients": (10_000, 100), "images": (10_000, 1, 28, 28),
        "ed_logits": (10_000, 10), "ec_logits": (10_000, 10),
        "ed_correct": (10_000,), "ec_correct": (10_000,),
        "capture_power_w": (10_000,), "raw_mean_w_per_pixel": (10_000,),
        "raw_peak_w_per_pixel": (10_000,), "view_mean_intensity": (10_000,),
        "proxy_clipping_area_fraction": (10_000,), "proxy_clipped_excess_fraction": (10_000,),
    }
    for name, shape in tensors.items():
        value = payload.get(name)
        _require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
                 f"generated cohort omitted or malformed {name}")
    if require_cuda_event:
        event_ms = payload.get("cuda_event_ms")
        _require(isinstance(event_ms, (int, float)) and math.isfinite(float(event_ms)) and float(event_ms) > 0.0,
                 "saved generated cohort is missing its positive CUDA event timing")
    labels = payload["requested_labels"]
    for name in ("capture_power_w", "raw_mean_w_per_pixel", "raw_peak_w_per_pixel",
                 "view_mean_intensity", "proxy_clipping_area_fraction", "proxy_clipped_excess_fraction"):
        _require(bool(torch.isfinite(payload[name]).all()), f"generated cohort {name} contains nonfinite values")
    _require(torch.equal(torch.bincount(labels, minlength=10), torch.full((10,), 1_000))
             and bool(torch.isfinite(payload["images"]).all())
             and bool(torch.isfinite(payload["ed_logits"]).all())
             and bool(torch.isfinite(payload["ec_logits"]).all())
             and bool(torch.isfinite(payload["ols_coefficients"].real).all())
             and bool(torch.isfinite(payload["ols_coefficients"].imag).all())
             and bool((payload["ols_coefficients"].abs().square().sum(1) <= base.MAX_MODAL_POWER_W).all()),
             "generated cohort contains nonfinite data, wrong class counts or over-bound power")
    _require(torch.equal(payload["ed_predictions"], payload["ed_logits"].argmax(1))
             and torch.equal(payload["ec_predictions"], payload["ec_logits"].argmax(1))
             and torch.equal(payload["ed_correct"], payload["ed_predictions"].eq(labels))
             and torch.equal(payload["ec_correct"], payload["ec_predictions"].eq(labels)),
             "generated cohort prediction and correctness records disagree with saved logits")
    for label in range(10):
        selected = labels == label
        _require(torch.unique(payload["proposal_ids"][selected]).numel() == 1_000,
                 f"generated cohort repeats proposal IDs for class {label}")
    if method == "G4":
        _require(bool(((payload["component_ids"] >= 0) & (payload["component_ids"] < COMPONENTS)).all()),
                 "generated G4 cohort has a component ID outside 0..3")
    else:
        _require(bool((payload["component_ids"] == 0).all()),
                 "generated G1 cohort has nonzero component IDs")
    _require(payload.get("classifier_based_sample_selection") is False
             and payload.get("per_image_normalization") is False,
             "generated cohort used a forbidden classifier selection or per-image normalization")
    captures = payload.get("native_capture_images")
    indices = payload.get("native_capture_indices")
    _require(isinstance(captures, torch.Tensor) and tuple(captures.shape) == (100, 1, 200, 200)
             and bool(torch.isfinite(captures).all())
             and isinstance(indices, torch.Tensor) and tuple(indices.shape) == (100,)
             and torch.unique(indices).numel() == 100
             and bool(((indices >= 0) & (indices < 10_000)).all())
             and torch.equal(torch.bincount(labels[indices], minlength=10), torch.full((10,), 10)),
             "native preview set is incomplete")


def _rows_from_cohort(representation: str, method: str, stream: int,
                      payload: dict[str, Any]) -> list[dict[str, Any]]:
    arrays = _metric_arrays(payload)
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(arrays["label"]):
        row = {"representation": representation, "prior": method, "stream": stream,
               "requested_label": int(label), "component_id": int(arrays["component"][index]),
               "proposal_id": int(arrays["proposal"][index]),
               "E_D_prediction": int(arrays["ed_prediction"][index]),
               "E_C_prediction": int(arrays["ec_prediction"][index]),
               "E_D_correct": int(arrays["ed_correct"][index]),
               "E_C_correct": int(arrays["ec_correct"][index])}
        for name in ("capture_power_w", "raw_mean_w_per_pixel", "raw_peak_w_per_pixel",
                     "view_mean_intensity", "proxy_clipping_area_fraction", "proxy_clipped_excess_fraction"):
            row[name] = float(arrays[name][index])
        rows.append(row)
    return rows


def _cell_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    _require(len(rows) == 30_000, "four-cell consumer requires exactly 30,000 rows per cell")
    per_class: list[dict[str, Any]] = []
    for label in range(10):
        selected = [row for row in rows if int(row["requested_label"]) == label]
        _require(len(selected) == 3_000, f"four-cell class {label} must have 3,000 rows")
        per_class.append({
            "label": label, "count": len(selected),
            "E_D_macro_recall": float(np.mean([int(row["E_D_correct"]) for row in selected])),
            "E_C_macro_recall": float(np.mean([int(row["E_C_correct"]) for row in selected])),
            "proxy_clipping_area_fraction": float(np.mean([float(row["proxy_clipping_area_fraction"]) for row in selected])),
            "proxy_clipped_excess_fraction": float(np.mean([float(row["proxy_clipped_excess_fraction"]) for row in selected])),
        })
    return {"population": len(rows),
            "E_D_macro_recall": float(np.mean([item["E_D_macro_recall"] for item in per_class])),
            "E_C_macro_recall": float(np.mean([item["E_C_macro_recall"] for item in per_class])),
            "macro_proxy_clipping_area_fraction": float(np.mean([item["proxy_clipping_area_fraction"] for item in per_class])),
            "macro_proxy_clipped_excess_fraction": float(np.mean([item["proxy_clipped_excess_fraction"] for item in per_class])),
            "per_class": per_class}


def _strata(rows: list[dict[str, Any]], representation: str, method: str) -> dict[tuple[int, int], dict[str, np.ndarray]]:
    result: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    selected = [row for row in rows if row["representation"] == representation and row["prior"] == method]
    for stream in range(STREAMS):
        for label in range(10):
            group = [row for row in selected if int(row["stream"]) == stream
                     and int(row["requested_label"]) == label]
            _require(len(group) == SAMPLE_COUNT, f"bootstrap stratum {representation}/{method}/{stream}/{label} is incomplete")
            result[(stream, label)] = {
                "ed": np.asarray([int(row["E_D_correct"]) for row in group], dtype=np.float64),
                "ec": np.asarray([int(row["E_C_correct"]) for row in group], dtype=np.float64),
                "clip": np.asarray([float(row["proxy_clipping_area_fraction"]) for row in group], dtype=np.float64),
            }
    return result


def _bootstrap_g4(rows: list[dict[str, Any]], seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    old = _strata(rows, "original", "G4")
    new = _strata(rows, "template", "G4")
    rng = np.random.default_rng(seed)
    distributions = {name: np.empty(BOOTSTRAP_RESAMPLES, dtype=np.float64) for name in ("E_D", "E_C", "clip_area")}
    point = {name: 0.0 for name in distributions}
    for name, key in (("E_D", "ed"), ("E_C", "ec"), ("clip_area", "clip")):
        point[name] = float(np.mean([new[stratum][key].mean() - old[stratum][key].mean()
                                     for stratum in sorted(old)]))
    for iteration in range(BOOTSTRAP_RESAMPLES):
        differences: dict[str, list[float]] = {name: [] for name in distributions}
        for stratum in sorted(old):
            old_indices = rng.integers(0, SAMPLE_COUNT, size=SAMPLE_COUNT)
            new_indices = rng.integers(0, SAMPLE_COUNT, size=SAMPLE_COUNT)
            for name, key in (("E_D", "ed"), ("E_C", "ec"), ("clip_area", "clip")):
                differences[name].append(float(new[stratum][key][new_indices].mean()
                                              - old[stratum][key][old_indices].mean()))
        for name in distributions:
            distributions[name][iteration] = float(np.mean(differences[name]))
    output: dict[str, Any] = {"resamples": BOOTSTRAP_RESAMPLES, "seed": seed,
                              "strata": "independent original/template draws within each stream/class cell; equal cell weights",
                              "proposal_ids_paired": False, "differences": {}}
    for name, values in distributions.items():
        output["differences"][name] = {
            "point_estimate": point[name],
            "ci_95": [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))],
            "bootstrap_mean": float(values.mean()),
        }
    output["bootstrap_rows"] = [{"replicate": index + 1,
                                 "delta_E_D_macro_recall": float(distributions["E_D"][index]),
                                 "delta_E_C_macro_recall": float(distributions["E_C"][index]),
                                 "delta_proxy_clipping_area_fraction": float(distributions["clip_area"][index])}
                                for index in range(BOOTSTRAP_RESAMPLES)]
    return output


def _nmi(assignments: torch.Tensor, categories: torch.Tensor, category_count: int) -> float:
    left = assignments.detach().cpu().numpy().astype(np.int64)
    right = categories.detach().cpu().numpy().astype(np.int64)
    _require(left.shape == right.shape and left.size > 0, "NMI input arrays are not aligned")
    table = np.zeros((int(left.max()) + 1, category_count), dtype=np.int64)
    np.add.at(table, (left, right), 1)
    pxy = table / table.sum()
    px = pxy.sum(1)
    py = pxy.sum(0)
    nz = pxy > 0
    expected = px[:, None] * py[None, :]
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / expected[nz])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    return 0.0 if hx + hy == 0.0 else 2.0 * mi / (hx + hy)


def _build_generation_rows(
    old_cohorts: dict[tuple[str, int], dict[str, Any]],
    new_cohorts: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ("G1", "G4"):
        for stream in range(STREAMS):
            rows.extend(_rows_from_cohort("original", method, stream, old_cohorts[(method, stream)]))
            rows.extend(_rows_from_cohort("template", method, stream, new_cohorts[(method, stream)]))
    _require(len(rows) == 120_000, "per-image metric consumer requires 120,000 declared old and new rows")
    return rows


def _replay_and_chart_checks(
    model: torch.nn.Module, evaluator_d: torch.nn.Module, evaluator_c: torch.nn.Module,
    old_cohorts: dict[tuple[str, int], dict[str, Any]], templates: dict[int, dict[str, Any]],
    scale: torch.Tensor, gain: torch.Tensor, device: torch.device,
) -> dict[str, Any]:
    replay_rows = 0
    maximum_view_error = 0.0
    maximum_ed_error = 0.0
    maximum_ec_error = 0.0
    maximum_power_delta = 0.0
    maximum_chart_decode_error = 0.0
    maximum_chart_ed_error = 0.0
    maximum_chart_ec_error = 0.0
    for method in ("G1", "G4"):
        for stream in range(STREAMS):
            payload = old_cohorts[(method, stream)]
            labels = payload["requested_labels"]
            selected_indices = torch.cat([torch.nonzero(labels == label, as_tuple=False).flatten()[:2]
                                          for label in range(10)]).long()
            coefficients = payload["ols_coefficients"][selected_indices]
            requested = labels[selected_indices]
            original = _decode_coefficients(model, evaluator_d, evaluator_c, coefficients,
                                            requested, gain, device, capture_first_per_class=0)
            saved_ed = payload["ed_logits"][selected_indices]
            saved_ec = payload["ec_logits"][selected_indices]
            _require(torch.allclose(original["images"], payload["images"][selected_indices]
                                    if payload["images"].shape[0] == 10_000 else original["images"],
                                    rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                     f"original {method} stream {stream} saved view replay failed")
            _require(torch.allclose(original["ed_logits"], saved_ed, rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
                     and torch.allclose(original["ec_logits"], saved_ec, rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                     f"original {method} stream {stream} saved evaluator replay failed")
            maximum_ed_error = max(maximum_ed_error, float((original["ed_logits"] - saved_ed).abs().max()))
            maximum_ec_error = max(maximum_ec_error, float((original["ec_logits"] - saved_ec).abs().max()))
            if payload["images"].shape[0] == 10_000:
                maximum_view_error = max(maximum_view_error,
                                         float((original["images"] - payload["images"][selected_indices]).abs().max()))
            replay_rows += len(selected_indices)

            charted: list[torch.Tensor] = []
            for coeff, label in zip(coefficients, requested, strict=True):
                r = templates[int(label)]["reference"].to(torch.complex128)
                normalized = coeff.to(torch.complex128) / scale.sqrt()
                projection = torch.dot(r.conj(), normalized)
                _require(float(projection.abs()) > PROJECTION_MIN * float(torch.linalg.vector_norm(normalized)),
                         f"saved row has near-zero template projection for class {int(label)}")
                charted.append(coeff.to(torch.complex128) * (projection / projection.abs()).conj())
            changed_high_precision = torch.stack(charted)
            original_power = coefficients.to(torch.complex128).abs().square().sum(1)
            changed_power = changed_high_precision.abs().square().sum(1)
            _require(torch.allclose(original_power, changed_power, rtol=1e-12, atol=1e-15),
                     "common phase chart changed recovered modal power")
            changed = changed_high_precision.to(torch.complex64)
            rotated = _decode_coefficients(model, evaluator_d, evaluator_c, changed,
                                           requested, gain, device, capture_first_per_class=0)
            image_delta = (rotated["images"] - original["images"]).abs()
            ed_delta = (rotated["ed_logits"] - original["ed_logits"]).abs()
            ec_delta = (rotated["ec_logits"] - original["ec_logits"]).abs()
            error = float(image_delta.max())
            maximum_chart_decode_error = max(maximum_chart_decode_error, error)
            maximum_chart_ed_error = max(maximum_chart_ed_error, float(ed_delta.max()))
            maximum_chart_ec_error = max(maximum_chart_ec_error, float(ec_delta.max()))
            chart_decoder_equivalent = torch.allclose(
                rotated["images"], original["images"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL,
            )
            if not chart_decoder_equivalent:
                print(json.dumps({
                    "phase": "COMMON_PHASE_REPLAY_MISMATCH", "method": method, "stream": stream,
                    "decoder_image_equivalent": chart_decoder_equivalent,
                    "max_abs_error": {"images": float(image_delta.max()),
                                      "E_D": float(ed_delta.max()), "E_C": float(ec_delta.max())},
                    "max_reference_magnitude": {"images": float(original["images"].abs().max()),
                                                 "E_D": float(original["ed_logits"].abs().max()),
                                                 "E_C": float(original["ec_logits"].abs().max())},
                    "max_tolerance_ratio": {
                        "images": float((image_delta / (REPLAY_ATOL + REPLAY_RTOL
                                                         * original["images"].abs())).max()),
                        "E_D": float((ed_delta / (REPLAY_ATOL + REPLAY_RTOL
                                                   * original["ed_logits"].abs())).max()),
                        "E_C": float((ec_delta / (REPLAY_ATOL + REPLAY_RTOL
                                                   * original["ec_logits"].abs())).max()),
                    },
                    "evaluator_logits_are_diagnostic_only": True,
                }), flush=True)
            _require(chart_decoder_equivalent,
                     f"class {method}/{stream} common-phase chart changed frozen Decoder output")

    global_phase = torch.polar(torch.ones((), dtype=torch.float64), torch.tensor(0.731, dtype=torch.float64))
    probe = old_cohorts[("G4", 0)]["ols_coefficients"][:10]
    reference = _decode_coefficients(model, evaluator_d, evaluator_c, probe, torch.arange(10), gain, device, 0)
    rotated_coeff = (probe * global_phase.to(torch.complex64)).contiguous()
    rotated = _decode_coefficients(model, evaluator_d, evaluator_c, rotated_coeff,
                                   torch.arange(10), gain, device, 0)
    _require(torch.allclose(reference["images"], rotated["images"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
             "common global phase changed frozen Decoder output")
    return {"saved_rows_replayed": replay_rows,
            "saved_view_max_absolute_error": maximum_view_error,
            "saved_E_D_logit_max_absolute_error": maximum_ed_error,
            "saved_E_C_logit_max_absolute_error": maximum_ec_error,
            "chart_decoder_max_absolute_error": maximum_chart_decode_error,
            "chart_E_D_logit_max_absolute_error_diagnostic_only": maximum_chart_ed_error,
            "chart_E_C_logit_max_absolute_error_diagnostic_only": maximum_chart_ec_error,
            "global_phase_invariance": True,
            "recovered_modal_power_preserved": True,
            "rtol": REPLAY_RTOL, "atol": REPLAY_ATOL,
            "optimizer_calls": 0, "encoder_forward_images": 0, "official_test_reads": 0}


def _new_sample_preflight(
    fits: dict[int, dict[str, Any]], scale: torch.Tensor, model: torch.nn.Module,
    evaluator_d: torch.nn.Module, evaluator_c: torch.nn.Module, gain: torch.Tensor,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in ("G1", "G4"):
        for stream in range(STREAMS):
            coeff_parts: list[torch.Tensor] = []
            label_parts: list[torch.Tensor] = []
            component_parts: list[torch.Tensor] = []
            proposal_parts: list[torch.Tensor] = []
            for label in range(10):
                coefficients, components, proposals, _ = _seeded_sample_class(
                    method, fits[label][method], scale, label, stream, 2, device,
                )
                coeff_parts.append(coefficients)
                component_parts.append(components)
                proposal_parts.append(proposals)
                label_parts.append(torch.full((2,), label, dtype=torch.long))
            coefficients = torch.cat(coeff_parts)
            labels = torch.cat(label_parts)
            _require(torch.equal(torch.bincount(labels, minlength=10), torch.full((10,), 2)),
                     f"new {method} stream {stream} preflight class counts are wrong")
            decoded = _decode_coefficients(model, evaluator_d, evaluator_c, coefficients, labels,
                                           gain, device, capture_first_per_class=0)
            for index, label in enumerate(labels.tolist()):
                rows.append({
                    "method": method, "stream": stream, "label": int(label), "sample_index": index % 2,
                    "proposal_id": int(proposal_parts[index // 2][index % 2]),
                    "component_id": int(component_parts[index // 2][index % 2]),
                    "image": decoded["images"][index],
                    "ed_logits": decoded["ed_logits"][index],
                    "ec_logits": decoded["ec_logits"][index],
                    "coefficient": coefficients[index],
                })
    _require(len(rows) == 120, "new-sample preflight must cover 2 rows × 2 priors × 3 streams × 10 classes")
    return rows


def _compare_preflight_to_full(preflight: list[dict[str, Any]], cohorts: dict[tuple[str, int], dict[str, Any]],
                               method: str | None = None, stream: int | None = None) -> None:
    for row in preflight:
        if method is not None and (row["method"] != method or row["stream"] != stream):
            continue
        cohort = cohorts[(row["method"], row["stream"])]
        label_positions = torch.nonzero(cohort["requested_labels"] == row["label"], as_tuple=False).flatten()
        full_index = int(label_positions[row["sample_index"]])
        _require(int(cohort["proposal_ids"][full_index]) == row["proposal_id"]
                 and int(cohort["component_ids"][full_index]) == row["component_id"]
                 and torch.equal(cohort["ols_coefficients"][full_index], row["coefficient"]),
                 "formal sampler first-two rows do not reproduce the complete production preflight")
        _require(torch.allclose(cohort["images"][full_index], row["image"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
                 and torch.allclose(cohort["ed_logits"][full_index], row["ed_logits"],
                                    rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
                 and torch.allclose(cohort["ec_logits"][full_index], row["ec_logits"],
                                    rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                 "formal Decoder/evaluator rows do not reproduce the complete production preflight")


def _make_rows_for_preflight(preflight: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in preflight:
        groups.setdefault((row["method"], row["stream"]), []).append(row)
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for key, records in groups.items():
        result[key] = {"rows": records}
    return result


def _save_previews(cohort: dict[str, Any], output_dir: Path, method: str, stream: int) -> int:
    root = output_dir / "native_previews" / method / f"stream_{stream}"
    root.mkdir(parents=True, exist_ok=True)
    captures = cohort["native_capture_images"]
    indices = cohort["native_capture_indices"]
    labels = cohort["requested_labels"]
    written = 0
    seen = [0] * 10
    for capture_index, image in zip(indices.tolist(), captures, strict=True):
        label = int(labels[capture_index])
        rank = seen[label]
        if rank >= 10:
            continue
        array = np.rint(image.squeeze().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(array, mode="L").save(root / f"class_{label}_sample_{rank:02d}.png", optimize=True)
        seen[label] += 1
        written += 1
    _require(seen == [10] * 10, f"native previews for {method} stream {stream} omit a class")
    return written


def _shape_ratios(cohorts: dict[tuple[str, int], dict[str, Any]], dev_reference: list[float]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method in ("G1", "G4"):
        per_class: list[dict[str, Any]] = []
        for label in range(10):
            views = torch.cat([cohorts[(method, stream)]["images"][
                cohorts[(method, stream)]["requested_labels"] == label] for stream in range(STREAMS)])
            _require(tuple(views.shape) == (3_000, 1, 28, 28),
                     f"template {method} class {label} shape sample count is wrong")
            distance = float(base._pair_distances(views, SHAPE_PAIR_SEED_BASE + label).mean())
            ratio = distance / max(float(dev_reference[label]), 1e-12)
            per_class.append({"label": label, "count": int(views.shape[0]),
                              "generated_tv_mean": distance,
                              "fixed_dev_tv_reference": float(dev_reference[label]),
                              "generated_to_dev_tv_ratio": ratio})
        output[method] = {"per_class": per_class,
                          "minimum_ratio": min(item["generated_to_dev_tv_ratio"] for item in per_class)}
    return output


def _nmi_summary(
    original_features: torch.Tensor, labels: torch.Tensor, original_assignments: torch.Tensor,
    template_assignments: dict[int, torch.Tensor], scale: torch.Tensor,
) -> dict[str, Any]:
    normalized = _complex(original_features)
    raw_magnitude = (normalized * scale.sqrt().unsqueeze(0)).abs()
    raw_reference = raw_magnitude.argmax(1)
    standardized_max = normalized.abs().argmax(1)
    per_class = []
    for label in range(10):
        mask = labels == label
        old = original_assignments[mask]
        new = template_assignments[label]
        _require(old.numel() == new.numel() == TRAIN_LABEL_COUNTS[label],
                 f"NMI class {label} TRAIN assignments are incomplete")
        per_class.append({
            "label": label, "rows": int(mask.sum()),
            "original_g4_actual_phase_reference_nmi": _nmi(old, raw_reference[mask], N_CHANNELS),
            "original_g4_standardized_max_nmi": _nmi(old, standardized_max[mask], N_CHANNELS),
            "template_g4_actual_phase_reference_nmi": _nmi(new, raw_reference[mask], N_CHANNELS),
            "template_g4_standardized_max_nmi": _nmi(new, standardized_max[mask], N_CHANNELS),
        })
    keys = [key for key in per_class[0] if key.endswith("_nmi")]
    macro = {key: float(np.mean([item[key] for item in per_class])) for key in keys}
    return {"per_class": per_class, "macro_mean_class_nmi": macro,
            "interpretation": "descriptive; original and template G4 component IDs are representation-local"}


def _load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = _path(args.source_checkpoint)
    feature_cache_path = _path(args.feature_cache)
    source_prior_path = _path(args.source_priors)
    baseline_dir = _path(args.baseline_dir)
    prior_report_path = _path(args.source_prior_report)
    cohort_dir = _path(args.source_cohort_dir)
    endpoint_summary_path = _path(args.endpoint_summary)
    training_data_path = _path(args.training_data)
    runtime_config_path = _path(args.runtime_config)
    baseline_config_path = baseline_dir / "config.json"
    for path in (checkpoint_path, feature_cache_path, source_prior_path, prior_report_path,
                 endpoint_summary_path, training_data_path, runtime_config_path, baseline_config_path):
        _require(path.is_file(), f"fixed explicit input is missing: {path}", FileNotFoundError)
    _require(checkpoint_path.resolve() == (baseline_dir / "C" / "checkpoint_step_010000.pt").resolve(),
             "C checkpoint path does not resolve to the explicitly registered local-10k/global-45k endpoint")
    endpoint = audit._load_endpoint(audit.EndpointSpec(
        "C_seed_2026101501", SOURCE_SEED, checkpoint_path, feature_cache_path, source_prior_path,
    ))
    _require(endpoint.checkpoint.get("completed_global_step") == SOURCE_STEP
             and endpoint.checkpoint.get("run_id") == SOURCE_RUN_ID
             and endpoint.cache.get("source_checkpoint_step") == SOURCE_STEP
             and endpoint.cache.get("source_run_id") == SOURCE_RUN_ID,
             "loaded C checkpoint/cache lineage does not resolve to global step 45,000")
    baseline_config = json.loads(baseline_config_path.read_text(encoding="utf-8"))
    endpoint_summary = json.loads(endpoint_summary_path.read_text(encoding="utf-8"))
    _require(endpoint_summary.get("branch") == "C" and endpoint_summary.get("global_step") == SOURCE_STEP
             and endpoint_summary.get("status") == "VALID_FAILED"
             and endpoint_summary.get("measurement_complete") is True,
             "fixed C endpoint summary identity or status changed")
    runtime_config = yaml.safe_load(runtime_config_path.read_text(encoding="utf-8"))
    _require(runtime_config.get("schema") == "lightgen_matched_small_vs_chen400.v1"
             and runtime_config.get("optics", {}).get("small", {}).get("ols_basis") == "gaussian_s12_v1"
             and runtime_config["optics"]["small"].get("production_propagation_grid") == 2048,
             "runtime Gaussian geometry or production grid changed")
    dev_reference = endpoint_summary.get("generation", {}).get("dev_within_class_tv_mean")
    _require(isinstance(dev_reference, list) and len(dev_reference) == 10
             and all(math.isfinite(float(value)) and float(value) > 0.0 for value in dev_reference),
             "fixed saved DEV shape reference is incomplete")
    pixels, labels, train_rows, dev_rows = base._load_data(training_data_path)
    train_labels = labels.index_select(0, train_rows)
    _require(torch.equal(train_rows, endpoint.rows)
             and torch.equal(train_labels, endpoint.labels)
             and tuple(train_rows.shape) == (TRAIN_COUNT,)
             and torch.equal(torch.bincount(train_labels, minlength=10), torch.tensor(TRAIN_LABEL_COUNTS))
             and torch.equal(torch.bincount(labels.index_select(0, dev_rows), minlength=10), torch.full((10,), 500))
             and not bool(torch.isin(train_rows, dev_rows).any()),
             "explicit TRAIN rows or labels do not match the fixed TRAIN/DEV split")
    old_cohorts, prior_run_id = _load_source_cohorts(cohort_dir, prior_report_path)
    old_assignments = endpoint.priors["G4_train_assignments_in_cache_order"].detach().cpu().long()
    _require(tuple(old_assignments.shape) == (TRAIN_COUNT,)
             and bool(((old_assignments >= 0) & (old_assignments < COMPONENTS)).all()),
             "original C G4 assignments are incomplete")
    _require(baseline_config.get("task_id") == SOURCE_TASK_ID
             and baseline_config.get("schema") == "lightgen_small_gaussian_foreground_balance_ab.v1"
             and baseline_config.get("source", {}).get("global_step") in (None, SOURCE_STEP),
             "saved C runtime config belongs to another endpoint")
    ed_checkpoint_path = _path(args.ed_checkpoint)
    ec_checkpoint_path = _path(args.ec_checkpoint)
    _require(ed_checkpoint_path.is_file() and ec_checkpoint_path.is_file(),
             "frozen E_D or E_C checkpoint is missing", FileNotFoundError)
    return {
        "checkpoint_path": checkpoint_path, "feature_cache_path": feature_cache_path,
        "source_prior_path": source_prior_path, "baseline_dir": baseline_dir,
        "prior_report_path": prior_report_path, "cohort_dir": cohort_dir,
        "endpoint_summary_path": endpoint_summary_path, "training_data_path": training_data_path,
        "runtime_config_path": runtime_config_path, "runtime_config": runtime_config,
        "baseline_config": baseline_config, "endpoint_summary": endpoint_summary,
        "dev_reference": [float(value) for value in dev_reference],
        "checkpoint": endpoint.checkpoint, "cache": endpoint.cache, "source_priors": endpoint.priors,
        "features": endpoint.features.detach().cpu().double(), "rows": endpoint.rows,
        "labels": endpoint.labels, "scale": endpoint.scale.detach().cpu().double(),
        "gain": endpoint.checkpoint["camera_gain"].detach().cpu(),
        "old_assignments": old_assignments, "old_cohorts": old_cohorts,
        "source_prior_run_id": prior_run_id,
        "ed_checkpoint_path": ed_checkpoint_path, "ec_checkpoint_path": ec_checkpoint_path,
        "pixels": pixels, "labels_all": labels, "train_rows": train_rows,
        "dev_rows": dev_rows,
    }


def _semantic_config(args: argparse.Namespace, data: dict[str, Any]) -> dict[str, Any]:
    path_fields = ("source_checkpoint", "feature_cache", "source_priors", "baseline_dir",
                   "source_prior_report", "source_cohort_dir", "endpoint_summary", "training_data",
                   "runtime_config", "ed_checkpoint", "ec_checkpoint")
    return {
        "task_id": TASK_ID,
        "inputs": {key: str(_path(getattr(args, key))) for key in path_fields},
        "source_endpoint": {"branch": "C", "seed": SOURCE_SEED,
                            "run_id": SOURCE_RUN_ID, "global_step": SOURCE_STEP,
                            "train_rows": TRAIN_COUNT, "labels": list(TRAIN_LABEL_COUNTS)},
        "representation": {"kind": "class_fixed_principal_reference_template",
                           "formula": "H_y=mean(u u^H), top eigenvector, canonical max port positive real; u'=u*conj((r^H u)/abs(r^H u))",
                           "projection_min": PROJECTION_MIN,
                           "eigenspace_relative_gap_min": PROJECTION_MIN},
        "prior": {"methods": ["G1", "G4"], "feature_dimensions": FEATURE_DIM,
                  "g4_components": COMPONENTS, "kmeans_seed_base": KMEANS_SEED_BASE,
                  "kmeans_max_iterations": prior.KMEANS_MAX_ITERATIONS,
                  "relative_covariance_ridge": RIDGE_RELATIVE},
        "sampling": {"streams": STREAMS, "samples_per_class_per_stream": SAMPLE_COUNT,
                     "maximum_proposals_per_class": MAX_PROPOSALS,
                     "g1_normal_seed_base": G1_NORMAL_SEED_BASE,
                     "g4_normal_seed_base": G4_NORMAL_SEED_BASE,
                     "g4_component_seed_base": G4_COMPONENT_SEED_BASE,
                     "power_bound_w": base.MAX_MODAL_POWER_W},
        "measurement": {"decoder_grid": 2048, "reference_grid": 4096,
                         "evaluator_roles": ["E_D_frozen", "E_C_frozen"],
                         "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
                         "bootstrap_seed": BOOTSTRAP_SEED,
                         "shape_pair_seed_base": SHAPE_PAIR_SEED_BASE,
                         "minimum_shape_ratio": SHAPE_RATIO_MIN,
                         "numpy_rtol": NUMPY_RTOL, "numpy_atol": NUMPY_ATOL,
                         "replay_rtol": REPLAY_RTOL, "replay_atol": REPLAY_ATOL},
        "populations": {"new_generation_rows": 60_000, "old_saved_generation_rows": 60_000,
                        "total_metric_rows": 120_000, "native_previews": 600},
        "counts": {"training_updates": 0, "optimizer_creations": 0, "encoder_forward_images": 0,
                   "test_reads": 0, "lpips_evaluations": 0, "human_reviews": 0},
        "device_required": "CUDA; no CPU formal-measurement fallback",
        "source_prior_run_id": data["source_prior_run_id"],
    }


def _preflight(
    data: dict[str, Any], output_dir: Path, semantic: dict[str, Any], device: torch.device,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    torch.cuda.reset_peak_memory_stats(device)
    template_started = torch.cuda.Event(enable_timing=True)
    template_finished = torch.cuda.Event(enable_timing=True)
    template_started.record()
    templates: dict[int, dict[str, Any]] = {}
    transformed_by_class: dict[int, torch.Tensor] = {}
    template_numpy_checks: list[dict[str, Any]] = []
    features = data["features"]
    labels = data["labels"]
    for label in range(10):
        class_features = features[labels == label]
        record, transformed = _template_and_transform(class_features, device)
        reference_np, transformed_np, independent = _numpy_template_and_transform(class_features)
        gpu_reference = record["reference"].numpy()
        gpu_transformed = transformed.detach().cpu().numpy()
        _require(np.allclose(gpu_reference, reference_np, rtol=NUMPY_RTOL, atol=NUMPY_ATOL)
                 and np.allclose(gpu_transformed, transformed_np,
                                 rtol=NUMPY_RTOL, atol=NUMPY_ATOL),
                 f"CUDA/NumPy template or transformed TRAIN parity failed for class {label}")
        _require(math.isclose(record["top_eigenvalue_relative_gap"],
                              independent["top_eigenvalue_relative_gap"], rel_tol=NUMPY_RTOL, abs_tol=NUMPY_ATOL)
                 and math.isclose(record["minimum_normalized_reference_projection"],
                                  independent["minimum_normalized_reference_projection"],
                                  rel_tol=NUMPY_RTOL, abs_tol=NUMPY_ATOL),
                 f"CUDA/NumPy template safety statistics changed for class {label}")
        # Common-phase invariance is tested against the independent transformation producer.
        source_values = _complex(class_features).to(device=device, dtype=torch.complex128)
        rotation = torch.polar(torch.ones((), dtype=torch.float64, device=device),
                               torch.tensor(0.731, dtype=torch.float64, device=device))
        rotated = source_values * rotation
        rotated_features = torch.view_as_real(rotated).reshape(-1, FEATURE_DIM)
        transformed_rotated = _complex(
            _transform_with_template(rotated_features, record["reference"], device)
        )
        _require(torch.allclose(transformed, transformed_rotated, rtol=NUMPY_RTOL, atol=NUMPY_ATOL),
                 f"class {label} template changed under a common phase rotation")
        templates[label] = record
        transformed_by_class[label] = torch.view_as_real(transformed).reshape(-1, FEATURE_DIM).contiguous()
        template_numpy_checks.append({
            "label": label, "rows": int(class_features.shape[0]),
            "template_max_abs_error": float(np.max(np.abs(gpu_reference - reference_np))),
            "transformed_max_abs_error": float(np.max(np.abs(gpu_transformed - transformed_np))),
            "relative_eigenvalue_gap": record["top_eigenvalue_relative_gap"],
            "minimum_normalized_projection": record["minimum_normalized_reference_projection"],
            "common_phase_invariance": True,
            "canonical_reference_channel": record["canonical_reference_channel"],
            "canonical_reference_exact_tie": record["canonical_reference_exact_tie"],
        })
    template_finished.record()
    torch.cuda.synchronize(device)
    priors_started = torch.cuda.Event(enable_timing=True)
    priors_finished = torch.cuda.Event(enable_timing=True)
    priors_started.record()
    fitted, assignments = _fit_template_priors(transformed_by_class, device)
    priors_finished.record()
    torch.cuda.synchronize(device)
    template_path = output_dir / "phase_templates_and_assignments.pt"
    prior_path = output_dir / "template_priors_G1_G4.pt"
    existing_fit_artifacts = template_path.is_file() or prior_path.is_file()
    _require(template_path.is_file() == prior_path.is_file(),
             "recovery found only one of the paired completed template/prior artifacts")
    template_record = {"schema": "lightgen_c45k_phase_chart_templates.v1",
                       "task_id": TASK_ID, "source_run_id": SOURCE_RUN_ID,
                       "source_global_step": SOURCE_STEP, "templates": templates,
                       "train_assignments": assignments}
    prior_record = {"schema": "lightgen_c45k_phase_chart_priors.v1",
                    "task_id": TASK_ID, "source_run_id": SOURCE_RUN_ID,
                    "source_global_step": SOURCE_STEP, "G1": fitted["G1"], "G4": fitted["G4"],
                    "G4_train_assignments_by_class": assignments}
    if existing_fit_artifacts:
        saved_templates, saved_fits = _fit_from_saved(output_dir, device)
        _assert_saved_fits_match(templates, fitted=fitted, assignments=assignments,
                                 saved_templates=saved_templates, saved_fits=saved_fits)
    else:
        _save_torch(template_path, template_record)
        _save_torch(prior_path, prior_record)
    fits: dict[int, dict[str, Any]] = {}
    for label in range(10):
        fits[label] = {"G1": fitted["G1"][label], "G4": fitted["G4"][label]}
    model = _load_model(data, device)
    evaluator_d, evaluator_c = _load_evaluators(data, device)
    _validate_zero_and_wrong_label_controls(templates, device)
    saved_control = _replay_and_chart_checks(model, evaluator_d, evaluator_c,
                                             data["old_cohorts"], templates,
                                             data["scale"], data["gain"], device)
    new_preflight_rows = _new_sample_preflight(fits, data["scale"], model, evaluator_d, evaluator_c,
                                               data["gain"], device)
    sample_path = output_dir / "new_sample_preflight.pt"
    sample_payload = {"task_id": TASK_ID, "source_run_id": SOURCE_RUN_ID,
                      "source_global_step": SOURCE_STEP, "rows": new_preflight_rows}
    if sample_path.is_file():
        saved_sample_payload = torch.load(sample_path, map_location="cpu", weights_only=False)
        _assert_preflight_rows_match(new_preflight_rows, saved_sample_payload)
    else:
        _save_torch(sample_path, sample_payload)
    window = replication._generated_window_check(
        model, data["old_cohorts"][("G4", 0)]["ols_coefficients"][:10],
        data["runtime_config"], device,
    )
    check_receipt = {
        "status": "PASSED", "source_endpoint": "C_seed_2026101501",
        "checkpoint_run_id": data["checkpoint"].get("run_id"),
        "checkpoint_completed_global_step": data["checkpoint"].get("completed_global_step"),
        "feature_rows": int(features.shape[0]), "train_label_counts": list(TRAIN_LABEL_COUNTS),
        "saved_original_rows": 60_000,
        "template_numpy_parity": template_numpy_checks,
        "fit_numpy_parity": fitted["numpy_moment_parity"],
        "saved_control_replay": saved_control,
        "new_generated_record_preflight_rows": 120,
        "new_generated_record_preflight_groups": [
            {"method": method, "stream": stream, "rows_per_class": 2, "classes": list(range(10))}
            for method in ("G1", "G4") for stream in range(STREAMS)
        ],
        "original_control_2048_4096_common_roi": window,
        "rejected_negative_controls": {"zero_reference_projection": True,
                                        "wrong_label_template": True},
        "official_test_reads": 0, "optimizer_creations": 0, "encoder_forward_images": 0,
    }
    check_receipt["cuda_timing"] = {
        "template_fit_ms": float(template_started.elapsed_time(template_finished)),
        "prior_fit_and_numpy_reference_ms": float(priors_started.elapsed_time(priors_finished)),
    }
    _write_json(output_dir / "preflight_receipt.json", check_receipt)
    return templates, fits, check_receipt, template_numpy_checks


def _load_model(data: dict[str, Any], device: torch.device) -> torch.nn.Module:
    model = base._model("small", 2048, device, data["runtime_config"])
    model.load_state_dict(data["checkpoint"]["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    _require(all(not parameter.requires_grad for parameter in model.parameters()),
             "frozen C model has a trainable parameter")
    return model


def _load_evaluators(data: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.nn.Module]:
    evaluator_d = base._load_evaluator(data["ed_checkpoint_path"], device)
    evaluator_c, _ = prior._load_ec(data["ec_checkpoint_path"], device)
    evaluator_d.eval()
    evaluator_c.eval()
    for evaluator in (evaluator_d, evaluator_c):
        for parameter in evaluator.parameters():
            parameter.requires_grad_(False)
    return evaluator_d, evaluator_c


def _validate_zero_and_wrong_label_controls(templates: dict[int, dict[str, Any]], device: torch.device) -> None:
    zero = torch.zeros(N_CHANNELS, dtype=torch.complex128, device=device)
    with torch.no_grad():
        rejected_zero = False
        try:
            _transform_with_template(torch.view_as_real(zero).reshape(1, FEATURE_DIM),
                                     templates[0]["reference"], device)
        except ExperimentError:
            rejected_zero = True
    _require(rejected_zero, "zero template projection negative control was not rejected")
    rejected_wrong_label = False
    try:
        _require_template_label(row_label=0, template_label=1)
    except ExperimentError:
        rejected_wrong_label = True
    _require(rejected_wrong_label, "wrong-label template negative control was not rejected")


def _cuda_receipt(device: torch.device, phase_times: dict[str, float], row_counts: dict[str, int]) -> dict[str, Any]:
    _require(device.type == "cuda" and torch.cuda.is_available(), "CUDA execution evidence is incomplete")
    name = torch.cuda.get_device_name(device)
    _require(name and all(value > 0 for value in phase_times.values()),
             "CUDA events are missing or show no production compute")
    return {
        "schema": "lightgen_c45k_phase_chart_cuda_execution.v1", "status": "PASSED",
        "device": name, "device_index": device.index, "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "major_computation_device": "cuda",
        "major_cuda_stages": ["template_eigendecomposition", "feature_transform", "G1_G4_fit",
                              "covariance_and_cholesky", "sampling_tensor_math", "frozen_decoder", "E_D", "E_C"],
        "cpu_stages": ["file_parsing", "seeded random number generation", "independent NumPy references",
                       "CSV reporting", "bootstrap and shape-distance summaries"],
        "phase_event_ms": phase_times,
        "phase_row_counts": row_counts,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "cuda_unavailable_fallback": False,
    }


def _fit_from_saved(output_dir: Path, device: torch.device) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    template_payload = torch.load(output_dir / "phase_templates_and_assignments.pt",
                                  map_location="cpu", weights_only=False)
    prior_payload = torch.load(output_dir / "template_priors_G1_G4.pt", map_location="cpu", weights_only=False)
    _require(template_payload.get("schema") == "lightgen_c45k_phase_chart_templates.v1"
             and prior_payload.get("schema") == "lightgen_c45k_phase_chart_priors.v1"
             and template_payload.get("source_run_id") == SOURCE_RUN_ID
             and prior_payload.get("source_run_id") == SOURCE_RUN_ID
             and template_payload.get("source_global_step") == SOURCE_STEP
             and prior_payload.get("source_global_step") == SOURCE_STEP,
             "saved template fit belongs to another endpoint")
    templates = template_payload["templates"]
    fits: dict[int, dict[str, Any]] = {}
    assignments = template_payload.get("train_assignments", {})
    for label in range(10):
        template = templates.get(label, templates.get(str(label)))
        g1_record = prior_payload["G1"].get(label, prior_payload["G1"].get(str(label)))
        g4_record = prior_payload["G4"].get(label, prior_payload["G4"].get(str(label)))
        _require(isinstance(template, dict) and isinstance(g1_record, dict) and isinstance(g4_record, dict),
                 f"saved template or prior is missing class {label}")
        reference = template.get("reference")
        eigenvalues = template.get("eigenvalues")
        template_assignment = assignments.get(label, assignments.get(str(label)))
        g4_assignment = g4_record.get("assignments")
        _require(isinstance(reference, torch.Tensor) and reference.dtype == torch.complex128
                 and tuple(reference.shape) == (N_CHANNELS,)
                 and bool(torch.isfinite(reference.real).all()) and bool(torch.isfinite(reference.imag).all())
                 and math.isclose(float(reference.abs().square().sum()), 1.0, rel_tol=NUMPY_RTOL, abs_tol=NUMPY_ATOL)
                 and isinstance(eigenvalues, torch.Tensor) and eigenvalues.dtype == torch.float64
                 and tuple(eigenvalues.shape) == (N_CHANNELS,),
                 f"saved phase template is malformed for class {label}")
        _require(isinstance(template_assignment, torch.Tensor)
                 and template_assignment.dtype == torch.long
                 and tuple(template_assignment.shape) == (TRAIN_LABEL_COUNTS[label],)
                 and isinstance(g4_assignment, torch.Tensor)
                 and torch.equal(template_assignment, g4_assignment)
                 and bool(((template_assignment >= 0) & (template_assignment < COMPONENTS)).all()),
                 f"saved G4 assignments are missing, malformed, or inconsistent for class {label}")
        tensor_shapes = {
            "G1": {"mean": (FEATURE_DIM,), "covariance": (FEATURE_DIM, FEATURE_DIM),
                   "regularized_covariance": (FEATURE_DIM, FEATURE_DIM),
                   "cholesky": (FEATURE_DIM, FEATURE_DIM)},
            "G4": {"weights": (COMPONENTS,), "means": (COMPONENTS, FEATURE_DIM),
                   "covariances": (COMPONENTS, FEATURE_DIM, FEATURE_DIM),
                   "regularized_covariances": (COMPONENTS, FEATURE_DIM, FEATURE_DIM),
                   "choleskies": (COMPONENTS, FEATURE_DIM, FEATURE_DIM),
                   "counts": (COMPONENTS,), "assignments": (TRAIN_LABEL_COUNTS[label],)},
        }
        for method, record in (("G1", g1_record), ("G4", g4_record)):
            for name, shape in tensor_shapes[method].items():
                value = record.get(name)
                _require(isinstance(value, torch.Tensor) and tuple(value.shape) == shape,
                         f"saved {method} prior class {label} has malformed {name}")
                if value.is_floating_point():
                    _require(bool(torch.isfinite(value).all()),
                             f"saved {method} prior class {label} has nonfinite {name}")
            _require(isinstance(record.get("ridge"), torch.Tensor)
                     and record["ridge"].numel() == 1
                     and bool(torch.isfinite(record["ridge"]))
                     and float(record["ridge"]) > 0.0,
                     f"saved {method} prior class {label} has an invalid ridge")
        _require(torch.equal(torch.bincount(g4_assignment, minlength=COMPONENTS), g4_record["counts"])
                 and int(g4_record["counts"].sum()) == TRAIN_LABEL_COUNTS[label]
                 and torch.allclose(g4_record["weights"].sum(), torch.tensor(1.0, dtype=torch.float64),
                                    rtol=NUMPY_RTOL, atol=NUMPY_ATOL),
                 f"saved G4 prior counts or weights are inconsistent for class {label}")
        templates[label] = template
        fits[label] = {"G1": g1_record, "G4": g4_record}
    return templates, fits


def _assert_saved_fits_match(
    templates: dict[int, dict[str, Any]], fitted: dict[str, Any], assignments: dict[int, torch.Tensor],
    saved_templates: dict[int, dict[str, Any]], saved_fits: dict[int, dict[str, Any]],
) -> None:
    for label in range(10):
        current_template = templates[label]
        saved_template = saved_templates[label]
        for name in ("reference", "eigenvalues"):
            _require(torch.allclose(current_template[name].detach().cpu(), saved_template[name].detach().cpu(),
                                    rtol=NUMPY_RTOL, atol=NUMPY_ATOL),
                     f"saved template recovery changed class {label} {name}")
        for name in ("top_eigenvalue_relative_gap", "minimum_normalized_reference_projection"):
            _require(math.isclose(float(current_template[name]), float(saved_template[name]),
                                  rel_tol=NUMPY_RTOL, abs_tol=NUMPY_ATOL),
                     f"saved template recovery changed class {label} {name}")
        _require(current_template["canonical_reference_channel"]
                 == saved_template["canonical_reference_channel"]
                 and current_template["canonical_reference_exact_tie"]
                 == saved_template["canonical_reference_exact_tie"]
                 and torch.equal(assignments[label].detach().cpu(),
                                 saved_fits[label]["G4"]["assignments"].detach().cpu()),
                 f"saved template recovery changed class {label} canonical channel or assignments")
        for method in ("G1", "G4"):
            current_fit = fitted[method][label]
            saved_fit = saved_fits[label][method]
            _require(set(current_fit) == set(saved_fit),
                     f"saved {method} recovery fields changed for class {label}")
            for name, current_value in current_fit.items():
                saved_value = saved_fit[name]
                if torch.is_tensor(current_value):
                    current_cpu = current_value.detach().cpu()
                    saved_cpu = saved_value.detach().cpu()
                    same = (torch.equal(current_cpu, saved_cpu) if not current_cpu.is_floating_point()
                            else torch.allclose(current_cpu, saved_cpu, rtol=NUMPY_RTOL, atol=NUMPY_ATOL))
                else:
                    same = current_value == saved_value
                _require(bool(same), f"saved {method} recovery changed class {label} {name}")


def _assert_preflight_rows_match(rows: list[dict[str, Any]], payload: dict[str, Any]) -> None:
    _require(payload.get("task_id") == TASK_ID and payload.get("source_run_id") == SOURCE_RUN_ID
             and payload.get("source_global_step") == SOURCE_STEP
             and isinstance(payload.get("rows"), list) and len(payload["rows"]) == len(rows),
             "saved representative sample preflight is missing or malformed")
    for current, saved in zip(rows, payload["rows"], strict=True):
        _require((current["method"], current["stream"], current["label"], current["sample_index"],
                  current["proposal_id"], current["component_id"])
                 == (saved["method"], saved["stream"], saved["label"], saved["sample_index"],
                     saved["proposal_id"], saved["component_id"])
                 and torch.equal(current["coefficient"], saved["coefficient"])
                 and torch.allclose(current["image"], saved["image"], rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
                 and torch.allclose(current["ed_logits"], saved["ed_logits"],
                                    rtol=REPLAY_RTOL, atol=REPLAY_ATOL)
                 and torch.allclose(current["ec_logits"], saved["ec_logits"],
                                    rtol=REPLAY_RTOL, atol=REPLAY_ATOL),
                 "representative rows changed between CUDA preflight and formal generation")


def _preflight_saved_samples(fits: dict[int, dict[str, Any]], data: dict[str, Any], output_dir: Path,
                             templates: dict[int, dict[str, Any]], device: torch.device) -> dict[str, Any]:
    model = _load_model(data, device)
    evaluator_d, evaluator_c = _load_evaluators(data, device)
    rows = _new_sample_preflight(fits, data["scale"], model, evaluator_d, evaluator_c,
                                 data["gain"], device)
    saved_preflight = torch.load(output_dir / "new_sample_preflight.pt", map_location="cpu", weights_only=False)
    _assert_preflight_rows_match(rows, saved_preflight)
    _validate_zero_and_wrong_label_controls(templates, device)
    window = replication._generated_window_check(
        model, data["old_cohorts"][("G4", 0)]["ols_coefficients"][:10],
        data["runtime_config"], device,
    )
    return {"new_rows": len(rows), "zero_and_wrong_label_negative_controls": "PASSED",
            "original_control_2048_4096_common_roi": window,
            "frozen_model_state_fingerprint": len(base._state_snapshot(model)),
            "frozen_evaluator_state_fingerprints": [len(base._state_snapshot(evaluator_d)),
                                                       len(base._state_snapshot(evaluator_c))]}


def _output_identity(
    output_dir: Path, semantic: dict[str, Any], resume: bool, *, recover_incomplete_preflight: bool = False,
) -> dict[str, Any] | None:
    config_path = output_dir / "run_config.json"
    if resume:
        receipt_path = output_dir / "preflight_receipt.json"
        _require(output_dir.is_dir() and config_path.is_file(),
                 "resume requires an existing task output directory and fixed run config")
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        _require(existing == semantic,
                 "resume paths or fixed task semantics differ from the completed preflight")
        if recover_incomplete_preflight:
            error_path = output_dir / "engineering_error.json"
            _require(not receipt_path.exists() and error_path.is_file(),
                     "preflight recovery requires a recorded engineering failure and no completed preflight")
            error = json.loads(error_path.read_text(encoding="utf-8"))
            _require(error.get("task_id") == TASK_ID and error.get("status") == "ENGINEERING_INVALID"
                     and error.get("source_global_step") == SOURCE_STEP
                     and error.get("scientific_result") is None,
                     "preflight recovery refused a non-engineering or mismatched failure")
            existing_files = {path.name for path in output_dir.iterdir() if path.is_file()}
            error_files = {name for name in existing_files
                           if name == "engineering_error.json"
                           or re.fullmatch(r"engineering_error_[0-9]{4}\.json", name)}
            fit_artifacts = {"phase_templates_and_assignments.pt", "template_priors_G1_G4.pt"}
            fit_present = existing_files & fit_artifacts
            sample_present = "new_sample_preflight.pt" in existing_files
            base_files = {"run_config.json", "resource_telemetry.csv", *error_files}
            allowed_file_sets = [base_files]
            if fit_present == fit_artifacts:
                allowed_file_sets.append(base_files | fit_artifacts)
                if sample_present:
                    allowed_file_sets.append(base_files | fit_artifacts | {"new_sample_preflight.pt"})
            _require("engineering_error.json" in error_files
                     and existing_files in allowed_file_sets
                     and not any(path.is_dir() for path in output_dir.iterdir()),
                     "preflight recovery found partial scientific artifacts; preserving this attempt for review")
            for name in sorted(error_files):
                previous = json.loads((output_dir / name).read_text(encoding="utf-8"))
                _require(previous.get("task_id") == TASK_ID
                         and previous.get("status") == "ENGINEERING_INVALID"
                         and previous.get("source_global_step") == SOURCE_STEP
                         and previous.get("scientific_result") is None,
                         f"preflight recovery refused non-engineering or mismatched failure {name}")
            return error
        _require(receipt_path.is_file(), "resume requires an existing completed preflight receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        _require(receipt.get("status") == "PASSED" and receipt.get("source_endpoint") == "C_seed_2026101501"
                 and receipt.get("checkpoint_completed_global_step") == SOURCE_STEP,
                 "resume preflight receipt is incomplete or bound to another endpoint")
        return None
    else:
        _require(not recover_incomplete_preflight,
                 "incomplete preflight recovery requires resume semantics", ValueError)
        _require(not output_dir.exists(), f"refusing to overwrite existing output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_json(config_path, semantic)
        return None


def _run_full(
    data: dict[str, Any], semantic: dict[str, Any], output_dir: Path,
    templates: dict[int, dict[str, Any]], fits: dict[int, dict[str, Any]], device: torch.device,
) -> dict[str, Any]:
    progress_path = output_dir / "progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        _require(progress.get("task_id") == TASK_ID and progress.get("source_global_step") == SOURCE_STEP,
                 "progress file belongs to another task or endpoint")
    else:
        progress = {"task_id": TASK_ID, "source_global_step": SOURCE_STEP,
                    "completed_blocks": [], "status": "RUNNING"}
    completed = set(progress.get("completed_blocks", []))
    model = _load_model(data, device)
    evaluator_d, evaluator_c = _load_evaluators(data, device)
    model_before = base._state_snapshot(model)
    ed_before = base._state_snapshot(evaluator_d)
    ec_before = base._state_snapshot(evaluator_c)
    torch.cuda.reset_peak_memory_stats(device)
    preflight_receipt = json.loads((output_dir / "preflight_receipt.json").read_text(encoding="utf-8"))
    timings: dict[str, float] = {
        "template_fit": float(preflight_receipt["cuda_timing"]["template_fit_ms"]),
        "prior_fit": float(preflight_receipt["cuda_timing"]["prior_fit_and_numpy_reference_ms"]),
        "sampling_decoder": 0.0,
    }
    preflight_payload = torch.load(output_dir / "new_sample_preflight.pt", map_location="cpu", weights_only=False)
    _require(preflight_payload.get("task_id") == TASK_ID and len(preflight_payload.get("rows", [])) == 120,
             "formal generation requires the complete 120-row saved preflight")
    new_cohorts: dict[tuple[str, int], dict[str, Any]] = {}
    accounting: list[dict[str, Any]] = []
    preview_count = 0
    for method in ("G1", "G4"):
        for stream in range(STREAMS):
            key = f"{method}_stream_{stream}"
            path = output_dir / "cohorts" / f"{key}.pt"
            if key in completed:
                _require(path.is_file(), f"progress marks {key} complete but cohort file is missing")
                payload = torch.load(path, map_location="cpu", weights_only=False)
                _validate_new_cohort(payload, method, stream, require_cuda_event=True)
                new_cohorts[(method, stream)] = payload
                timings["sampling_decoder"] += float(payload.get("cuda_event_ms", 0.0))
                preview_count += _save_previews(payload, output_dir, method, stream)
                continue
            if path.exists():
                # A process can stop after atomically saving a cohort but before
                # recording it in progress.json. Adopt only a complete, validated
                # file; malformed artifacts remain untouched for diagnosis.
                payload = torch.load(path, map_location="cpu", weights_only=False)
                _validate_new_cohort(payload, method, stream, require_cuda_event=True)
                new_cohorts[(method, stream)] = payload
                timings["sampling_decoder"] += float(payload["cuda_event_ms"])
                preview_count += _save_previews(payload, output_dir, method, stream)
                completed.add(key)
                progress.update({"completed_blocks": sorted(completed), "status": "RUNNING",
                                 "current_block": key, "completed_new_rows": len(completed) * 10_000,
                                 "recovered_complete_block": key,
                                 "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                _write_json(progress_path, progress)
                continue
            torch.cuda.synchronize(device)
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            cohort, cohort_record = _decode_new_group(
                method, stream, fits, data["scale"], model, evaluator_d,
                evaluator_c, data["gain"], device,
            )
            finished.record()
            torch.cuda.synchronize(device)
            block_event_ms = float(started.elapsed_time(finished))
            timings["sampling_decoder"] += block_event_ms
            cohort["cuda_event_ms"] = block_event_ms
            _save_torch(path, cohort)
            reread = torch.load(path, map_location="cpu", weights_only=False)
            _validate_new_cohort(reread, method, stream, require_cuda_event=True)
            _require(torch.equal(reread["ols_coefficients"], cohort["ols_coefficients"])
                     and torch.equal(reread["ed_logits"], cohort["ed_logits"]),
                     f"saved cohort consumer altered {key}")
            new_cohorts[(method, stream)] = reread
            _compare_preflight_to_full(preflight_payload["rows"], {(method, stream): reread}, method, stream)
            accounting.append(cohort_record)
            preview_count += _save_previews(reread, output_dir, method, stream)
            completed.add(key)
            progress.update({"completed_blocks": sorted(completed), "status": "RUNNING",
                             "current_block": key, "completed_new_rows": len(completed) * 10_000,
                             "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            _write_json(progress_path, progress)
            del cohort, reread
            torch.cuda.empty_cache()
    _require(len(new_cohorts) == 6 and completed == {f"{method}_stream_{stream}"
             for method in ("G1", "G4") for stream in range(STREAMS)},
             "new generation consumer did not complete six method/stream blocks")

    old_rows: list[dict[str, Any]] = []
    for method in ("G1", "G4"):
        for stream in range(STREAMS):
            old_rows.extend(_rows_from_cohort("original", method, stream,
                                              data["old_cohorts"][(method, stream)]))
    new_rows: list[dict[str, Any]] = []
    for method in ("G1", "G4"):
        for stream in range(STREAMS):
            new_rows.extend(_rows_from_cohort("template", method, stream, new_cohorts[(method, stream)]))
    rows = old_rows + new_rows
    _require(len(old_rows) == len(new_rows) == 60_000 and len(rows) == 120_000,
             "old/new generation row populations are inconsistent")
    _write_csv(output_dir / "generation_per_image_metrics.csv", GENERATION_FIELDS, rows)
    cell_summary: dict[str, Any] = {}
    for representation in ("original", "template"):
        for method in ("G1", "G4"):
            cell = [row for row in rows if row["representation"] == representation and row["prior"] == method]
            cell_summary[f"{representation}_{method}"] = _cell_metrics(cell)
    _write_csv(output_dir / "four_cell_summary.csv", (
        "representation", "prior", "population", "E_D_macro_recall", "E_C_macro_recall",
        "macro_proxy_clipping_area_fraction", "macro_proxy_clipped_excess_fraction",
    ), [{"representation": key.split("_")[0], "prior": key.split("_")[1], **{
        metric: value for metric, value in summary.items() if metric != "per_class"},
         "population": summary["population"]}
        for key, summary in cell_summary.items()])
    bootstrap = _bootstrap_g4(rows)
    shape = _shape_ratios(new_cohorts, data["dev_reference"])
    nmi = _nmi_summary(data["features"], data["labels"], data["old_assignments"],
                       {label: torch.as_tensor(fits[label]["G4"]["assignments"]).cpu()
                        for label in range(10)}, data["scale"])
    gate = {
        "comparison": "template_G4_minus_original_G4",
        "E_D_macro_recall_ci_lower_gt_zero": bootstrap["differences"]["E_D"]["ci_95"][0] > 0.0,
        "E_C_macro_recall_ci_lower_gt_zero": bootstrap["differences"]["E_C"]["ci_95"][0] > 0.0,
        "proxy_clipping_area_ci_upper_le_zero": bootstrap["differences"]["clip_area"]["ci_95"][1] <= 0.0,
        "all_template_G4_class_shape_ratios_at_least_0_25": all(
            item["generated_to_dev_tv_ratio"] >= SHAPE_RATIO_MIN for item in shape["G4"]["per_class"]
        ),
    }
    gate["passed"] = all(gate[key] for key in (
        "E_D_macro_recall_ci_lower_gt_zero", "E_C_macro_recall_ci_lower_gt_zero",
        "proxy_clipping_area_ci_upper_le_zero", "all_template_G4_class_shape_ratios_at_least_0_25",
    ))
    bootstrap_rows = bootstrap.pop("bootstrap_rows")
    _write_csv(output_dir / "primary_bootstrap_draws.csv", (
        "replicate", "delta_E_D_macro_recall", "delta_E_C_macro_recall",
        "delta_proxy_clipping_area_fraction",
    ), bootstrap_rows)
    primary = {"gate": gate, "bootstrap": bootstrap,
               "old_G4": cell_summary["original_G4"],
               "template_G4": cell_summary["template_G4"],
               "shape_diversity": shape["G4"]}
    _write_json(output_dir / "primary_comparison.json", primary)
    _write_csv(output_dir / "template_g4_shape_ratios.csv",
               ("label", "count", "generated_tv_mean", "fixed_dev_tv_reference", "generated_to_dev_tv_ratio"),
               shape["G4"]["per_class"])
    _write_csv(output_dir / "template_g1_shape_ratios_descriptive.csv",
               ("label", "count", "generated_tv_mean", "fixed_dev_tv_reference", "generated_to_dev_tv_ratio"),
               shape["G1"]["per_class"])
    _write_csv(output_dir / "bootstrap_descriptive_g1.csv", (
        "comparison", "delta_E_D_macro_recall", "delta_E_C_macro_recall",
        "delta_proxy_clipping_area_fraction",
    ), [{"comparison": "template_G1_minus_original_G1",
         "delta_E_D_macro_recall": cell_summary["template_G1"]["E_D_macro_recall"]
                                   - cell_summary["original_G1"]["E_D_macro_recall"],
         "delta_E_C_macro_recall": cell_summary["template_G1"]["E_C_macro_recall"]
                                   - cell_summary["original_G1"]["E_C_macro_recall"],
         "delta_proxy_clipping_area_fraction": cell_summary["template_G1"]["macro_proxy_clipping_area_fraction"]
                                               - cell_summary["original_G1"]["macro_proxy_clipping_area_fraction"]}])
    _write_json(output_dir / "phase_reference_nmi_descriptive.json", nmi)
    accounting_fields = (
        "method", "stream", "label", "normal_seed", "component_seed", "requested_samples",
        "retained_samples", "proposals_evaluated", "power_rejected_candidates", "rejection_fraction",
    )
    accounting_rows = [
        {name: record[name] for name in accounting_fields}
        for cohort in new_cohorts.values() for record in cohort["sampling_accounting"]
    ]
    _write_csv(output_dir / "cuda_sampling_accounting.csv", accounting_fields, accounting_rows)
    _require(preview_count == 600, f"expected exactly 600 native preview images, wrote {preview_count}")
    _require(base._assert_close(model_before, base._state_snapshot(model), "C_model_frozen_end") == 0.0
             and base._assert_close(ed_before, base._state_snapshot(evaluator_d), "E_D_frozen_end") == 0.0
             and base._assert_close(ec_before, base._state_snapshot(evaluator_c), "E_C_frozen_end") == 0.0,
             "frozen model or evaluator changed during generation")
    cuda = _cuda_receipt(device, timings,
                         {"train_template_rows": TRAIN_COUNT, "template_transform_rows": TRAIN_COUNT,
                          "prior_fit_rows": TRAIN_COUNT, "kmeans_rows": TRAIN_COUNT,
                          "new_generated_rows": 60_000, "decoder_rows": 60_000,
                          "E_D_rows": 60_000, "E_C_rows": 60_000,
                          "old_saved_rows_consumed": 60_000})
    _write_json(output_dir / "cuda_execution_receipt.json", cuda)
    return {"rows": rows, "cell_summary": cell_summary, "bootstrap": bootstrap,
            "shape": shape, "nmi": nmi, "gate": gate, "cuda": cuda,
            "new_cohorts": new_cohorts, "preview_count": preview_count,
            "timing_ms": timings}


def _validate_report_consumer(output_dir: Path, rows: list[dict[str, Any]], report: dict[str, Any]) -> None:
    _require(report.get("task_id") == TASK_ID and report.get("measurement_complete") is True
             and report.get("source_endpoint") == "C_seed_2026101501"
             and report.get("source_run_id") == SOURCE_RUN_ID
             and report.get("source_global_step") == SOURCE_STEP
             and report.get("source_status") == "VALID_FAILED"
             and report.get("new_generation_rows") == 60_000
             and report.get("reused_original_generation_rows") == 60_000
             and report.get("generation_metric_rows") == 120_000,
             "final report consumer rejected wrong task or endpoint")
    _require(len(rows) == 120_000, "final report consumer rejected missing generation rows")
    _validate_generation_rows(rows)
    group_counts: dict[tuple[str, str, int, int], int] = {}
    for row in rows:
        group = (row["representation"], row["prior"], int(row["stream"]), int(row["requested_label"]))
        group_counts[group] = group_counts.get(group, 0) + 1
    _require(len(group_counts) == 120 and set(group_counts.values()) == {1_000},
             "final report consumer rejected incomplete method/stream/class strata")
    cuda = json.loads((output_dir / "cuda_execution_receipt.json").read_text(encoding="utf-8"))
    _require(cuda.get("status") == "PASSED" and cuda.get("major_computation_device") == "cuda"
             and cuda.get("cuda_unavailable_fallback") is False
             and all(float(value) > 0 for value in cuda.get("phase_event_ms", {}).values())
             and cuda.get("phase_row_counts", {}).get("decoder_rows") == 60_000
             and report.get("cuda_execution", {}).get("status") == "PASSED"
             and report.get("cuda_execution", {}).get("major_computation_device") == "cuda"
             and report.get("cuda_execution", {}).get("phase_event_ms") == cuda.get("phase_event_ms"),
             "final report consumer rejected missing or incomplete CUDA evidence")


def _validate_generation_rows(rows: list[dict[str, Any]]) -> None:
    key_rows: set[tuple[str, str, int, int, int]] = set()
    numeric_fields = ("capture_power_w", "raw_mean_w_per_pixel", "raw_peak_w_per_pixel",
                      "view_mean_intensity", "proxy_clipping_area_fraction", "proxy_clipped_excess_fraction")
    for row in rows:
        _require(set(row) == set(GENERATION_FIELDS), "generation row has missing, extra, or mixed statistic fields")
        _require(row["representation"] in {"original", "template"} and row["prior"] in {"G1", "G4"}
                 and int(row["stream"]) in range(STREAMS)
                 and int(row["requested_label"]) in range(10)
                 and int(row["component_id"]) in range(COMPONENTS)
                 and int(row["proposal_id"]) >= 0,
                 "generation row has an unknown endpoint category or invalid identity")
        key = (row["representation"], row["prior"], int(row["stream"]),
               int(row["requested_label"]), int(row["proposal_id"]))
        _require(key not in key_rows, "final report consumer rejected duplicate generation row")
        key_rows.add(key)
        _require(int(row["E_D_correct"]) in (0, 1) and int(row["E_C_correct"]) in (0, 1),
                 "generation correctness value is not binary")
        for name in numeric_fields:
            _require(math.isfinite(float(row[name])), f"generation row has nonfinite {name}")


def _assert_generation_rows_match_expected(
    expected: list[dict[str, Any]], actual: list[dict[str, Any]],
) -> None:
    _require(len(expected) == len(actual) == 120_000,
             "local full readback disagrees with expected source and generated populations")
    expected_by_key = {_generation_row_key(row): row for row in expected}
    actual_by_key = {_generation_row_key(row): row for row in actual}
    _require(len(expected_by_key) == len(expected) and expected_by_key.keys() == actual_by_key.keys(),
             "local full readback found a missing, extra, duplicate, or wrong-endpoint row")
    integer_fields = ("stream", "requested_label", "component_id", "proposal_id", "E_D_prediction",
                      "E_C_prediction", "E_D_correct", "E_C_correct")
    for key, expected_row in expected_by_key.items():
        actual_row = actual_by_key[key]
        for name in GENERATION_FIELDS:
            if name in {"representation", "prior"}:
                _require(actual_row[name] == expected_row[name], f"local readback changed {name} for {key}")
            elif name in integer_fields:
                _require(int(actual_row[name]) == int(expected_row[name]),
                         f"local readback changed {name} for {key}")
            else:
                _require(math.isfinite(float(actual_row[name]))
                         and float(actual_row[name]) == float(expected_row[name]),
                         f"local readback changed {name} for {key}")


def _generation_row_key(row: dict[str, Any]) -> tuple[str, str, int, int, int]:
    return (str(row["representation"]), str(row["prior"]), int(row["stream"]),
            int(row["requested_label"]), int(row["proposal_id"]))


def _typed_generation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    integer_fields = {"stream", "requested_label", "component_id", "proposal_id", "E_D_prediction",
                      "E_C_prediction", "E_D_correct", "E_C_correct"}
    return [{name: (int(value) if name in integer_fields else
                    float(value) if name not in {"representation", "prior"} else value)
             for name, value in row.items()} for row in rows]


def _assert_json_equal(actual: Any, expected: Any, path: str = "report") -> None:
    if isinstance(expected, dict):
        _require(isinstance(actual, dict) and actual.keys() == expected.keys(), f"{path} fields differ")
        for key, value in expected.items():
            _assert_json_equal(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        _require(isinstance(actual, list) and len(actual) == len(expected), f"{path} list length differs")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_json_equal(left, right, f"{path}[{index}]")
    elif isinstance(expected, (float, np.floating)):
        _require(isinstance(actual, (int, float)) and math.isclose(float(actual), float(expected),
                 rel_tol=1e-12, abs_tol=1e-12), f"{path} numeric result differs")
    else:
        _require(actual == expected, f"{path} value differs")


def _write_markdown_report(output_dir: Path, report: dict[str, Any]) -> None:
    gate = report["primary_comparison"]["gate"]
    differences = report["primary_comparison"]["bootstrap"]["differences"]
    result = "主要比较通过，模板G4成为机器候选。" if gate["passed"] else "测量有效但主要比较未通过，保留原C＋G4。"
    text = f"""# 固定C端点的GPU相位表示对照

## 结果与状态

{result} 本轮仅比较冻结C全局45,000步端点上的模板G4与原表示G4；四格表和逐行表覆盖原表示及模板表示的G1/G4。测量完整，系统状态`VALID_FAILED`保持。

唯一主要判定：E_D差={differences['E_D']['point_estimate']:.6f}，95%区间[{differences['E_D']['ci_95'][0]:.6f}, {differences['E_D']['ci_95'][1]:.6f}]；E_C差={differences['E_C']['point_estimate']:.6f}，95%区间[{differences['E_C']['ci_95'][0]:.6f}, {differences['E_C']['ci_95'][1]:.6f}]；代理截断面积差={differences['clip_area']['point_estimate']:.6f}，95%区间[{differences['clip_area']['ci_95'][0]:.6f}, {differences['clip_area']['ci_95'][1]:.6f}]。模板G4十类形态距离比最低值={report['primary_comparison']['shape_diversity']['minimum_ratio']:.6f}。完整阈值字段见`primary_comparison.json`。

## 科学认识

结果仅解释为同一冻结光学端点、同一尺度和同一G4容量下，改变复系数公共相位图表后的机器代理差异。G1、表示×K、逐类、NMI与截断超额比例均为次要描述。G4分量编号是各自表示内的局部编号。

模板最大特征值相对间隔范围={min(x['relative_eigenvalue_gap'] for x in report['template_checks']):.6f}至{max(x['relative_eigenvalue_gap'] for x in report['template_checks']):.6f}；最小归一化参考投影={min(x['minimum_normalized_projection'] for x in report['template_checks']):.6f}。这些量只说明本端点模板可计算，不代表硬件相位稳定或物理可达。

## 目标缺口

本轮未做人工可读性、官方TEST、LPIPS、实际光纤输入可达性或物理传播测量。代理改善不能升级系统状态，也不能单独证明语义生成成功或模型可采用。

## 工程完成与验证

CUDA设备：{report['cuda_execution']['device']}，CUDA {report['cuda_execution']['cuda_runtime']}。实际新生成并评价60,000条记录；另复用原G1/G4保存60,000条，逐行表共120,000行。GPU事件、峰值显存和每阶段行数见`cuda_execution_receipt.json`。模板/变换、均值、协方差和Cholesky与独立NumPy参考按rtol={NUMPY_RTOL:g}、atol={NUMPY_ATOL:g}核对；保存控制重放和2048/4096共同ROI检查见`preflight_receipt.json`。

训练更新、优化器创建、Encoder前向、TEST读取、LPIPS和人工评阅均为0。600张原生200×200预览按方法／流／类别各取前10张；全体200×200浮点帧未保存。

## 决策影响与下一步

{"保留模板G4作为机器候选，可另行设计独立质量确认；本报告不授权该确认。" if gate['passed'] else "保留原C＋G4，降低该模板表示方向优先级；本报告不授权继续实验。"} G1只提供描述性简化先验线索，不改变主要判定。停止于机器对照、证据解释和后继问题排序。
"""
    (output_dir / "report.md").write_text(text, encoding="utf-8")


def _final_readback(output_dir: Path, source_data: dict[str, Any] | None = None) -> dict[str, Any]:
    report_path = output_dir / REPORT_NAME
    _require(report_path.is_file(), f"completed machine report is missing: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows_text = _read_csv(output_dir / "generation_per_image_metrics.csv", GENERATION_FIELDS)
    _require(len(rows_text) == 120_000, "local consumer did not read all 120,000 declared metric rows")
    keyset: set[tuple[str, str, int, int, int]] = set()
    group_counts: dict[tuple[str, str, int, int], int] = {}
    for row in rows_text:
        key = (row["representation"], row["prior"], int(row["stream"]),
               int(row["requested_label"]), int(row["proposal_id"]))
        _require(key not in keyset, "local readback found a duplicate row")
        keyset.add(key)
        group = (row["representation"], row["prior"], int(row["stream"]), int(row["requested_label"]))
        group_counts[group] = group_counts.get(group, 0) + 1
        _require(row["representation"] in {"original", "template"} and row["prior"] in {"G1", "G4"},
                 "local readback found mixed row semantics")
    _require(len(group_counts) == 120 and set(group_counts.values()) == {1_000},
             "local readback found a missing or extra method/stream/class group")
    typed_rows = _typed_generation_rows(rows_text)
    new_files = sorted((output_dir / "cohorts").glob("*.pt"))
    _require(len(new_files) == 6, "local readback expected six complete new generation cohort files")
    for method in ("G1", "G4"):
        for stream in range(STREAMS):
            payload = torch.load(output_dir / "cohorts" / f"{method}_stream_{stream}.pt",
                                 map_location="cpu", weights_only=False)
            _validate_new_cohort(payload, method, stream, require_cuda_event=True)
    _validate_report_consumer(output_dir, rows_text, report)
    cuda_path = output_dir / "cuda_execution_receipt.json"
    cuda = json.loads(cuda_path.read_text(encoding="utf-8"))
    receipt = {
        "schema": "lightgen_c45k_phase_chart_local_readback.v1", "task_id": TASK_ID,
        "status": "PASSED", "verified_output_dir": str(output_dir.resolve()),
        "generation_metric_rows": len(rows_text), "new_generation_rows": 60_000,
        "original_saved_rows": 60_000, "method_stream_class_groups": len(group_counts),
        "new_cohort_files": len(new_files), "cuda_device": cuda["device"],
        "cuda_major_computation_device": cuda["major_computation_device"],
        "report_primary_gate": report["primary_comparison"]["gate"],
        "input_lineage_rechecked_locally": source_data is not None,
        "official_test_reads": 0,
    }
    if source_data is not None:
        new_cohorts = {
            (method, stream): torch.load(output_dir / "cohorts" / f"{method}_stream_{stream}.pt",
                                        map_location="cpu", weights_only=False)
            for method in ("G1", "G4") for stream in range(STREAMS)
        }
        expected = _build_generation_rows(source_data["old_cohorts"], new_cohorts)
        _assert_generation_rows_match_expected(expected, rows_text)
        summaries: dict[str, Any] = {}
        for representation in ("original", "template"):
            for method in ("G1", "G4"):
                key = f"{representation}_{method}"
                summaries[key] = _cell_metrics([row for row in typed_rows
                                                 if row["representation"] == representation
                                                 and row["prior"] == method])
        _assert_json_equal(report.get("four_cell_summary"), summaries, "report.four_cell_summary")
        bootstrap = _bootstrap_g4(typed_rows)
        bootstrap.pop("bootstrap_rows")
        primary = report.get("primary_comparison", {})
        _assert_json_equal(primary.get("bootstrap"), bootstrap, "report.primary_comparison.bootstrap")
        gate = {
            "comparison": "template_G4_minus_original_G4",
            "E_D_macro_recall_ci_lower_gt_zero": bootstrap["differences"]["E_D"]["ci_95"][0] > 0.0,
            "E_C_macro_recall_ci_lower_gt_zero": bootstrap["differences"]["E_C"]["ci_95"][0] > 0.0,
            "proxy_clipping_area_ci_upper_le_zero": bootstrap["differences"]["clip_area"]["ci_95"][1] <= 0.0,
            "all_template_G4_class_shape_ratios_at_least_0_25": len(
                primary.get("shape_diversity", {}).get("per_class", [])) == 10 and all(
                float(item["generated_to_dev_tv_ratio"]) >= SHAPE_RATIO_MIN
                for item in primary.get("shape_diversity", {}).get("per_class", [])
            ),
        }
        gate["passed"] = all(gate[name] for name in (
            "E_D_macro_recall_ci_lower_gt_zero", "E_C_macro_recall_ci_lower_gt_zero",
            "proxy_clipping_area_ci_upper_le_zero", "all_template_G4_class_shape_ratios_at_least_0_25",
        ))
        _assert_json_equal(primary.get("gate"), gate, "report.primary_comparison.gate")
        saved_primary = json.loads((output_dir / "primary_comparison.json").read_text(encoding="utf-8"))
        for name in ("gate", "bootstrap", "shape_diversity"):
            _assert_json_equal(saved_primary.get(name), primary.get(name), f"primary_comparison.json.{name}")
        _assert_json_equal(saved_primary.get("old_G4"), summaries["original_G4"],
                           "primary_comparison.json.old_G4")
        _assert_json_equal(saved_primary.get("template_G4"), summaries["template_G4"],
                           "primary_comparison.json.template_G4")
        shape = _shape_ratios(new_cohorts, source_data["dev_reference"])
        _assert_json_equal(primary.get("shape_diversity"), shape["G4"],
                           "report.primary_comparison.shape_diversity")
        expected_decision = ("PHASE_TEMPLATE_G4_MACHINE_CANDIDATE_SUPPORTED" if gate["passed"]
                             else "PHASE_TEMPLATE_G4_PROXY_GAIN_NOT_SUPPORTED")
        _require(report.get("scientific_status") == expected_decision,
                 "final report scientific decision does not consume the verified primary gate")
        receipt["explicit_source_lineage"] = {"checkpoint_run_id": source_data["checkpoint"].get("run_id"),
                                               "checkpoint_global_step": source_data["checkpoint"].get("completed_global_step"),
                                               "cache_rows": int(source_data["features"].shape[0]),
                                               "source_prior_run_id": source_data["source_prior_run_id"]}
    _write_json(output_dir / "local_readback_receipt.json", receipt)
    return receipt


def _run(args: argparse.Namespace) -> None:
    output_dir = _path(args.output_dir)
    if args.readback_only:
        data = _load_inputs(args)
        receipt = _final_readback(output_dir, data)
        print(json.dumps(receipt, ensure_ascii=False), flush=True)
        return
    _require(torch.cuda.is_available(), "CUDA is unavailable; formal measurement cannot fall back to CPU")
    device = torch.device("cuda")
    data = _load_inputs(args)
    semantic = _semantic_config(args, data)
    if args.recover_preflight:
        prior_error = _output_identity(output_dir, semantic, resume=True, recover_incomplete_preflight=True)
        templates, fits, preflight, _ = _preflight(data, output_dir, semantic, device)
        _write_json(output_dir / "engineering_recovery_receipt.json", {
            "schema": "lightgen_c45k_phase_chart_preflight_recovery.v1",
            "task_id": TASK_ID, "status": "RECOVERED",
            "original_error_file": "engineering_error.json",
            "prior_engineering_error_files": sorted(
                path.name for path in output_dir.glob("engineering_error*.json")
            ),
            "original_error_type": prior_error["error_type"],
            "original_error": prior_error["error"],
            "preflight_status": preflight["status"],
            "source_endpoint": "C_seed_2026101501", "source_global_step": SOURCE_STEP,
            "new_generated_record_preflight_rows": preflight["new_generated_record_preflight_rows"],
            "scientific_result": None,
        })
        _require(set(templates) == set(fits) == set(range(10)),
                 "recovered preflight did not produce all fixed class templates and priors")
        print(json.dumps({"task_id": TASK_ID, "status": "PREFLIGHT_RECOVERED",
                          "output_dir": str(output_dir),
                          "rows": preflight["new_generated_record_preflight_rows"]},
                         ensure_ascii=False), flush=True)
        return
    _output_identity(output_dir, semantic, args.resume)
    if args.resume:
        templates, fits = _fit_from_saved(output_dir, device)
        preflight = json.loads((output_dir / "preflight_receipt.json").read_text(encoding="utf-8"))
        validation = _preflight_saved_samples(fits, data, output_dir, templates, device)
        _require(preflight.get("new_generated_record_preflight_rows") == 120,
                 "saved preflight omitted required new samples")
        _write_json(output_dir / "preflight_resume_recheck.json", validation)
    else:
        templates, fits, preflight, template_checks = _preflight(data, output_dir, semantic, device)
        if args.preflight_only:
            print(json.dumps({"task_id": TASK_ID, "status": "PREFLIGHT_PASSED",
                              "output_dir": str(output_dir), "rows": preflight["new_generated_record_preflight_rows"]},
                             ensure_ascii=False), flush=True)
            return
    if args.preflight_only:
        _require(args.resume, "--preflight-only should not be combined with --resume")
        return
    _require(args.resume, "full generation requires --resume after a completed representative preflight")
    result = _run_full(data, semantic, output_dir, templates, fits, device)
    decision = "PHASE_TEMPLATE_G4_MACHINE_CANDIDATE_SUPPORTED" if result["gate"]["passed"] \
        else "PHASE_TEMPLATE_G4_PROXY_GAIN_NOT_SUPPORTED"
    report = {
        "task_id": TASK_ID, "status": "COMPLETE_DIAGNOSTIC", "scientific_status": decision,
        "measurement_complete": True,
        "scientific_question": "With frozen C at global step 45,000 and the same G4 capacity, does a class-fixed phase reference template improve saved generation proxies over the original phase chart?",
        "source_endpoint": "C_seed_2026101501", "source_run_id": SOURCE_RUN_ID,
        "source_global_step": SOURCE_STEP, "source_status": "VALID_FAILED",
        "training_updates": 0, "optimizer_creations": 0, "encoder_forward_images": 0,
        "official_test_reads": 0, "lpips_evaluations": 0, "human_reviews": 0,
        "new_generation_rows": 60_000, "reused_original_generation_rows": 60_000,
        "generation_metric_rows": 120_000, "native_preview_images": result["preview_count"],
        "four_cell_summary": result["cell_summary"],
        "primary_comparison": {"gate": result["gate"], "bootstrap": result["bootstrap"],
                               "shape_diversity": result["shape"]["G4"]},
        "secondary_descriptive": {"template_G1_shape_diversity": result["shape"]["G1"],
                                   "template_G1_minus_original_G1": {
                                       key: result["cell_summary"]["template_G1"][key]
                                       - result["cell_summary"]["original_G1"][key]
                                       for key in ("E_D_macro_recall", "E_C_macro_recall",
                                                   "macro_proxy_clipping_area_fraction",
                                                   "macro_proxy_clipped_excess_fraction")},
                                   "phase_reference_nmi": result["nmi"]},
        "template_checks": json.loads((output_dir / "preflight_receipt.json").read_text(encoding="utf-8"))[
            "template_numpy_parity"],
        "cuda_execution": result["cuda"],
        "decision_boundary": "A passing machine gate identifies only a candidate for separate independent quality confirmation; it is not a readability, physical reachability, hardware or model-adoption result.",
        "next_action_authorization": "NONE",
        "system_scientific_status_changed": False,
    }
    _validate_report_consumer(output_dir, result["rows"], report)
    _write_json(output_dir / REPORT_NAME, report)
    _write_markdown_report(output_dir, report)
    _write_json(output_dir / "remote_completion_receipt.json", {
        "schema": "lightgen_c45k_phase_chart_remote_completion.v1",
        "task_id": TASK_ID, "status": "COMPLETE_DIAGNOSTIC",
        "measurement_complete": True, "report": REPORT_NAME,
        "generation_metric_rows": 120_000, "new_generation_rows": 60_000,
        "new_cohort_files": 6, "native_preview_images": 600,
        "cuda_status": result["cuda"]["status"],
        "source_global_step": SOURCE_STEP, "scientific_status": decision,
    })
    receipt = _final_readback(output_dir, data)
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    progress.update({"status": "COMPLETE", "completed_blocks": sorted(progress["completed_blocks"]),
                     "result": decision, "local_style_readback_rows": receipt["generation_metric_rows"]})
    _write_json(output_dir / "progress.json", progress)
    print(json.dumps({"task_id": TASK_ID, "status": report["status"],
                      "scientific_status": decision, "primary_gate": result["gate"],
                      "output_dir": str(output_dir)}, ensure_ascii=False), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--source-priors", type=Path)
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--source-prior-report", type=Path)
    parser.add_argument("--source-cohort-dir", type=Path)
    parser.add_argument("--endpoint-summary", type=Path)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--ed-checkpoint", type=Path)
    parser.add_argument("--ec-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--recover-preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--readback-only", action="store_true")
    args = parser.parse_args()
    required = ("source_checkpoint", "feature_cache", "source_priors", "baseline_dir",
                "source_prior_report", "source_cohort_dir", "endpoint_summary", "training_data",
                "runtime_config", "ed_checkpoint", "ec_checkpoint")
    missing = [name for name in required if getattr(args, name) is None]
    _require(not missing, f"missing explicit input paths: {', '.join(missing)}", ValueError)
    if not args.readback_only:
        if args.recover_preflight:
            _require(args.preflight_only and not args.resume,
                     "recover-preflight requires preflight-only and cannot run formal generation", ValueError)
        else:
            _require(not (args.preflight_only and args.resume),
                     "preflight-only cannot resume an existing completed attempt", ValueError)
            _require(args.preflight_only or args.resume,
                     "select --preflight-only for preflight or --resume after a completed preflight", ValueError)
    return args


def main() -> None:
    args = _parse_args()
    try:
        _run(args)
    except Exception as error:
        if not args.readback_only:
            output_dir = _path(args.output_dir)
            if output_dir.is_dir() and not (output_dir / REPORT_NAME).is_file():
                error_path = output_dir / "engineering_error.json"
                suffix = 2
                while error_path.exists():
                    error_path = output_dir / f"engineering_error_{suffix:04d}.json"
                    suffix += 1
                _write_json(error_path, {
                    "task_id": TASK_ID, "status": "ENGINEERING_INVALID",
                    "error_type": type(error).__name__, "error": str(error),
                    "source_global_step": SOURCE_STEP,
                    "scientific_result": None,
                    "scientific_updates": 0, "encoder_forward_images": 0,
                    "official_test_reads": 0,
                })
        raise


if __name__ == "__main__":
    main()
