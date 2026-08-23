"""Regularized class-conditional mixture transport for electronic O0.

This is a diagnostic-only bridge between the frozen WAE posterior and a
single-pass Gaussian code.  EM is deliberately implemented without an
optimizer: the only mutable artifact in this lifecycle is the mixture
parameter file, while the WAE and both evaluators remain immutable inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from ...evidence import (
    atomic_torch_save,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from .electronic_o0_latent_transport import (
    CODE_DIM,
    LOCKED_SEEDS,
    _decode_views,
    _enable_deterministic_cuda,
    _evaluate_views,
    _load_evaluators,
    _load_wae,
    _metrics,
    _posterior,
    _sliced_distance,
    _write_checked,
    panel_codes,
)
from .electronic_o0_wae import _bootstrap_ci, _jacobian_summary, _memorization_audit
from ...student.evaluator import load_local_mnist


SCHEMA_PREFIX = "electronic_o0_mixture_transport"
MIXTURE_SEEDS = (8501, 8502, 8503)
DESIGN_SEED = 173205
BOOTSTRAP_SEED = 90210
COMPONENT_CANDIDATES = (2, 4, 8)
TEMPERATURE_CANDIDATES = (1.0, 1.1, 1.2, 1.35)
EM_MAX_ITERATIONS = 50
EM_TOLERANCE = 1.0e-5
SHRINKAGE = 0.05
MIN_WEIGHT = 0.01
MIN_EFFECTIVE_COUNT = 100.0
MAX_CONDITION = 1.0e6
OUTPUT_ROOT = Path(
    "outputs/20260823_v2/runs/20260823_electronic_o0_mixture_transport_v1"
)


def _git() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {"commit": commit, "clean": not status, "status": status}


def _config(path: str | Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("mixture transport config must be a mapping")
    required = {
        "schema_version": 1,
        "code_dim": CODE_DIM,
        "harness_revision": 1,
    }
    if any(payload.get(name) != value for name, value in required.items()):
        raise ValueError("mixture transport config is not the registered v1 contract")
    if tuple(payload.get("mixture_seeds", ())) != MIXTURE_SEEDS:
        raise ValueError("mixture transport seeds are fixed to 8501/8502/8503")
    if tuple(payload.get("locked_panel_seeds", ())) != LOCKED_SEEDS:
        raise ValueError("mixture transport locked panels are fixed")
    return payload


def _require_files(paths: list[str | Path], dataset_root: str | Path) -> None:
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("mixture transport lineage is missing: " + ", ".join(missing))
    if not Path(dataset_root).is_dir():
        raise FileNotFoundError(f"MNIST dataset root is missing: {dataset_root}")


def _lineage(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    paths = [
        args.config,
        args.wae_training_checkpoint,
        args.wae_decoder_checkpoint,
        args.wae_confirm_evidence,
        args.latent_transport_attribution,
        args.latent_transport_confirm_evidence,
        args.primary_evaluator_checkpoint,
        args.secondary_evaluator_checkpoint,
        args.primary_qualification_evidence,
        args.secondary_qualification_evidence,
    ]
    _require_files(paths, args.dataset_root)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("formal mixture transport lifecycle requires CUDA")
    if args.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUDA lifecycle requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    return {
        "git": _git(),
        "config_sha256": file_sha256(args.config),
        "config_content": config,
        "wae_training_sha256": file_sha256(args.wae_training_checkpoint),
        "wae_decoder_sha256": file_sha256(args.wae_decoder_checkpoint),
        "wae_confirm_sha256": file_sha256(args.wae_confirm_evidence),
        "latent_transport_attribution_sha256": file_sha256(args.latent_transport_attribution),
        "latent_transport_confirm_sha256": file_sha256(args.latent_transport_confirm_evidence),
        "primary_evaluator_sha256": file_sha256(args.primary_evaluator_checkpoint),
        "secondary_evaluator_sha256": file_sha256(args.secondary_evaluator_checkpoint),
        "primary_qualification_sha256": file_sha256(args.primary_qualification_evidence),
        "secondary_qualification_sha256": file_sha256(args.secondary_qualification_evidence),
        "dataset_root": str(args.dataset_root),
        "device": args.device,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "seeds": {
            "mixture": list(MIXTURE_SEEDS),
            "design": DESIGN_SEED,
            "locked": list(LOCKED_SEEDS),
            "bootstrap": BOOTSTRAP_SEED,
        },
    }


def _validate_upstream(args: argparse.Namespace) -> dict[str, Any]:
    wae = json.loads(Path(args.wae_confirm_evidence).read_text(encoding="utf-8"))
    if (
        wae.get("schema") != "electronic_o0_wae_confirm.v2"
        or wae.get("status") != "valid"
        or wae.get("classification") != "wae_image_manifold_failed"
    ):
        raise ValueError("mixture transport requires the corrected WAE v2 valid failure")
    if wae.get("checkpoint") != str(args.wae_decoder_checkpoint):
        raise ValueError("WAE decoder does not match mixture transport lineage")
    training_path = wae.get("training", {}).get("training_checkpoint")
    if training_path != str(args.wae_training_checkpoint):
        raise ValueError("WAE training checkpoint does not match mixture transport lineage")
    attribution = json.loads(Path(args.latent_transport_attribution).read_text(encoding="utf-8"))
    if (
        attribution.get("schema") != "electronic_o0_latent_transport_attribution.v1"
        or attribution.get("status") != "valid"
        or attribution.get("classification") != "nonlinear_latent_transport_candidate"
    ):
        raise ValueError("mixture transport requires valid nonlinear transport attribution")
    confirm = json.loads(Path(args.latent_transport_confirm_evidence).read_text(encoding="utf-8"))
    if (
        confirm.get("schema") != "electronic_o0_latent_transport_confirm.v1"
        or confirm.get("status") != "valid"
        or confirm.get("classification") != "latent_flow_semantic_failed"
    ):
        raise ValueError("mixture transport requires the valid failed seed-8401 flow confirmation")
    return {"wae": wae, "attribution": attribution, "confirm": confirm}


def _covariance(values: torch.Tensor) -> torch.Tensor:
    if values.shape[0] < 2:
        return torch.eye(CODE_DIM, dtype=values.dtype, device=values.device) * 1.0e-3
    centered = values - values.mean(0, keepdim=True)
    return centered.T @ centered / (values.shape[0] - 1)


def _regularize_covariance(covariance: torch.Tensor) -> torch.Tensor:
    trace = torch.trace(covariance) / CODE_DIM
    return (1.0 - SHRINKAGE) * covariance + SHRINKAGE * trace * torch.eye(
        CODE_DIM, dtype=covariance.dtype, device=covariance.device
    ) + 1.0e-6 * torch.eye(CODE_DIM, dtype=covariance.dtype, device=covariance.device)


def _component_log_prob(
    values: torch.Tensor,
    means: torch.Tensor,
    cholesky: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    centered = values[:, None, :] - means[None, :, :]
    solved = torch.linalg.solve_triangular(
        cholesky[None, :, :, :], centered.unsqueeze(-1), upper=False
    ).squeeze(-1)
    quadratic = solved.square().sum(-1)
    logdet = 2.0 * torch.log(torch.diagonal(cholesky, dim1=-2, dim2=-1)).sum(-1)
    return (
        torch.log(weights.clamp_min(1.0e-12))[None, :]
        - 0.5 * (CODE_DIM * math.log(2.0 * math.pi) + logdet[None, :] + quadratic)
    )


def _mixture_nll(
    values: torch.Tensor, means: torch.Tensor, cholesky: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    return -torch.logsumexp(_component_log_prob(values, means, cholesky, weights), dim=1).mean()


def fit_gaussian_mixture(
    values: torch.Tensor,
    components: int,
    *,
    seed: int = 8501,
    max_iterations: int = EM_MAX_ITERATIONS,
    tolerance: float = EM_TOLERANCE,
) -> dict[str, Any]:
    """Fit one regularized full-covariance mixture with deterministic EM."""
    if values.ndim != 2 or values.shape[1] != CODE_DIM or values.shape[0] < components:
        raise ValueError("mixture values have an invalid shape")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Quantile initialization is deterministic and keeps every registered
    # component above the effective-count floor on unimodal as well as
    # genuinely multimodal posteriors.  The seed only breaks exact ties.
    covariance = _covariance(values)
    _, principal = torch.linalg.eigh(covariance)
    direction = principal[:, -1]
    projection = values @ direction
    jitter = torch.rand(values.shape[0], generator=generator).to(values.device) * 1.0e-8
    order = torch.argsort(projection + jitter)
    positions = torch.linspace(
        0,
        values.shape[0] - 1,
        components + 2,
        device=values.device,
        dtype=torch.float32,
    )[1:-1].round().long()
    means = values.index_select(0, order.index_select(0, positions)).clone()
    initial_covariances = []
    boundaries = torch.linspace(
        0, values.shape[0], components + 1, device=values.device, dtype=torch.float32
    ).round().long()
    for component in range(components):
        segment = values.index_select(
            0, order[boundaries[component] : boundaries[component + 1]]
        )
        initial_covariances.append(_regularize_covariance(_covariance(segment)))
    cholesky = torch.linalg.cholesky(torch.stack(initial_covariances))
    weights = torch.full(
        (components,), 1.0 / components, dtype=values.dtype, device=values.device
    )
    history: list[float] = []
    counts = torch.full_like(weights, values.shape[0] / components)
    for iteration in range(max_iterations):
        log_prob = _component_log_prob(values, means, cholesky, weights)
        responsibilities = torch.softmax(log_prob, dim=1)
        counts = responsibilities.sum(0)
        if (counts < MIN_EFFECTIVE_COUNT).any():
            raise ValueError("mixture component effective count is below the registered minimum")
        weights = counts / counts.sum()
        means = (responsibilities.T @ values) / counts[:, None]
        covariances = []
        for component in range(components):
            centered = values - means[component]
            covariance = (centered * responsibilities[:, component : component + 1]).T @ centered
            covariance = covariance / counts[component].clamp_min(1.0)
            covariances.append(_regularize_covariance(covariance))
        covariance_stack = torch.stack(covariances)
        cholesky_info = torch.linalg.cholesky_ex(covariance_stack)
        if int(cholesky_info.info.max()) != 0:
            raise ValueError("mixture covariance is not positive definite")
        cholesky = cholesky_info.L
        condition = torch.linalg.cond(covariance_stack)
        if not torch.isfinite(condition).all() or float(condition.max()) > MAX_CONDITION:
            raise ValueError("mixture covariance condition exceeds the registered maximum")
        nll = float(_mixture_nll(values, means, cholesky, weights).detach().cpu())
        history.append(nll)
        if len(history) > 1 and abs(history[-2] - history[-1]) <= tolerance * max(1.0, abs(history[-2])):
            break
    if (weights < MIN_WEIGHT).any() or not torch.isfinite(cholesky).all():
        raise ValueError("mixture contains an invalid component")
    return {
        "weights": weights,
        "means": means,
        "cholesky": cholesky,
        "effective_counts": counts,
        "condition": float(torch.linalg.cond(cholesky @ cholesky.transpose(-1, -2)).max().cpu()),
        "nll_history": history,
        "iterations": len(history),
    }


def fit_classwise_mixtures(
    codes: torch.Tensor,
    labels: torch.Tensor,
    validation_codes: torch.Tensor,
    validation_labels: torch.Tensor,
    *,
    seed: int = 8501,
    candidates: tuple[int, ...] = COMPONENT_CANDIDATES,
) -> dict[str, Any]:
    selected: list[dict[str, Any]] = []
    choices: list[int] = []
    validation_nll: list[float] = []
    for label in range(10):
        train = codes[labels == label]
        validation = validation_codes[validation_labels == label]
        candidates_fit = []
        for components in candidates:
            try:
                fit = fit_gaussian_mixture(
                    train, components, seed=seed + label * 101 + components
                )
            except ValueError:
                continue
            score = float(_mixture_nll(validation, fit["means"], fit["cholesky"], fit["weights"]).cpu())
            candidates_fit.append((score, components, fit))
        if not candidates_fit:
            raise ValueError(f"no valid mixture component count for class {label}")
        score, components, fit = min(candidates_fit, key=lambda value: (value[0], value[1]))
        selected.append(fit)
        choices.append(components)
        validation_nll.append(score)
    return {
        "components": choices,
        "classes": selected,
        "validation_nll": validation_nll,
        "validation_nll_mean": float(sum(validation_nll) / len(validation_nll)),
        "seed": seed,
        "mapping_version": "cdf_component_then_cholesky_v1",
    }


def _normal_cdf(values: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(values / math.sqrt(2.0)))


def _normal_icdf(values: torch.Tensor) -> torch.Tensor:
    return math.sqrt(2.0) * torch.erfinv((2.0 * values - 1.0).clamp(-1.0 + 1.0e-7, 1.0 - 1.0e-7))


def mixture_transport(
    epsilon: torch.Tensor,
    labels: torch.Tensor,
    mixture: dict[str, Any],
    *,
    covariance_temperature: float = 1.0,
) -> torch.Tensor:
    """Map a standard Gaussian tensor to a class-conditional mixture exactly."""
    if epsilon.ndim != 2 or epsilon.shape[1] != CODE_DIM:
        raise ValueError("epsilon must have shape [batch,8]")
    if epsilon.shape[0] != labels.shape[0]:
        raise ValueError("mixture labels and codes must be paired")
    result = torch.empty_like(epsilon)
    u = _normal_cdf(epsilon[:, 0]).clamp(1.0e-7, 1.0 - 1.0e-7)
    for label in range(10):
        rows = torch.nonzero(labels == label, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        fit = mixture["classes"][label]
        weights = fit["weights"].to(epsilon.device)
        means = fit["means"].to(epsilon.device)
        cholesky = fit["cholesky"].to(epsilon.device) * covariance_temperature
        # CUDA has no deterministic cumsum kernel on the supported runtime.
        # The mixture has at most eight weights, so computing this boundary on
        # CPU is exact for the frozen artifact and preserves the CUDA contract.
        cdf = torch.cumsum(weights.cpu(), 0).to(epsilon.device)
        local_u = u.index_select(0, rows)
        component = torch.searchsorted(cdf, local_u).clamp_max(weights.numel() - 1)
        previous = torch.cat((torch.zeros(1, device=epsilon.device), cdf[:-1]))
        within = ((local_u - previous.index_select(0, component)) / weights.index_select(0, component)).clamp(1.0e-7, 1.0 - 1.0e-7)
        eta = epsilon.index_select(0, rows).clone()
        eta[:, 0] = _normal_icdf(within)
        mean = means.index_select(0, component)
        chol = cholesky.index_select(0, component)
        result.index_copy_(0, rows, mean + torch.bmm(chol, eta.unsqueeze(-1)).squeeze(-1))
    return result


def _mixture_to_cpu(mixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "components": list(mixture["components"]),
        "classes": [
            {
                key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                for key, value in fit.items()
                if key in {"weights", "means", "cholesky", "effective_counts", "condition", "iterations"}
            }
            for fit in mixture["classes"]
        ],
        "validation_nll": list(mixture["validation_nll"]),
        "validation_nll_mean": mixture["validation_nll_mean"],
        "seed": mixture["seed"],
        "mapping_version": mixture["mapping_version"],
    }


def _classwise_swd(
    generated: torch.Tensor,
    target: torch.Tensor,
    generated_labels: torch.Tensor,
    target_labels: torch.Tensor | None = None,
) -> float:
    if target_labels is None:
        target_labels = generated_labels
    return sum(
        _sliced_distance(
            generated[generated_labels == label],
            target[target_labels == label],
            seed=9000 + label,
        )
        for label in range(10)
    ) / 10.0


def _design_temperature(
    mixture: dict[str, Any],
    decoder,
    posterior: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[float, dict[str, Any]]:
    epsilon, labels = panel_codes(DESIGN_SEED, device=device)
    baseline = mixture_transport(epsilon, labels, mixture)
    baseline_views = _decode_views(decoder, baseline, labels)
    baseline_memory = _memorization_audit(
        baseline_views,
        labels.cpu(),
        posterior["train_images"],
        posterior["train_labels"],
        posterior["validation_images"],
        posterior["validation_labels"],
        device=device,
    )
    baseline_swd = _classwise_swd(
        baseline.cpu(),
        posterior["validation_codes"],
        labels.cpu(),
        posterior["validation_labels"],
    )
    attempts = []
    selected = None
    for temperature in TEMPERATURE_CANDIDATES:
        values = mixture_transport(epsilon, labels, mixture, covariance_temperature=temperature)
        views = _decode_views(decoder, values, labels)
        memory = _memorization_audit(
            views,
            labels.cpu(),
            posterior["train_images"],
            posterior["train_labels"],
            posterior["validation_images"],
            posterior["validation_labels"],
            device=device,
        )
        swd = _classwise_swd(
            values.cpu(),
            posterior["validation_codes"],
            labels.cpu(),
            posterior["validation_labels"],
        )
        row = {
            "temperature": temperature,
            "memorization": memory,
            "sliced_distance": swd,
            "sliced_distance_ratio": swd / max(baseline_swd, 1.0e-12),
            "view_hash": tensor_sha256(views),
        }
        attempts.append(row)
        if selected is None and memory["exact_duplicate_count"] == 0 and memory["pass_fraction"] >= 0.95 and swd <= 1.20 * baseline_swd:
            selected = row
    if selected is None:
        selected = attempts[-1]
    selected_temperature = float(selected["temperature"])
    return selected_temperature, {
        "design_seed": DESIGN_SEED,
        "baseline_sliced_distance": baseline_swd,
        "baseline_memorization": baseline_memory,
        "attempts": attempts,
        "selected": selected,
        "passed": selected["memorization"]["exact_duplicate_count"] == 0 and selected["memorization"]["pass_fraction"] >= 0.95 and selected["sliced_distance_ratio"] <= 1.20,
    }


def _fit_and_calibrate(
    posterior: dict[str, torch.Tensor], decoder, device: torch.device, *, seed: int
) -> tuple[dict[str, Any], float, dict[str, Any]]:
    mixture = fit_classwise_mixtures(
        posterior["train_codes"],
        posterior["train_labels"],
        posterior["validation_codes"],
        posterior["validation_labels"],
        seed=seed,
    )
    temperature, calibration = _design_temperature(mixture, decoder, posterior, device)
    mixture["covariance_temperature"] = temperature
    return mixture, temperature, calibration


def _load_mixture(path: str | Path) -> dict[str, Any]:
    payload = safe_torch_load(path)
    if not isinstance(payload, dict) or payload.get("schema") != f"{SCHEMA_PREFIX}_artifact.v1":
        raise ValueError("mixture inference artifact schema is invalid")
    if any(key in payload for key in ("encoder", "decoder", "optimizer", "evaluator")):
        raise ValueError("mixture inference artifact contains a forbidden state")
    return payload["mixture"]


def _measure_mixture(
    mixture_path: str | Path,
    decoder,
    device: torch.device,
    dataset_root: str,
    primary_path: str,
    secondary_path: str,
) -> dict[str, Any]:
    mixture = _load_mixture(mixture_path)
    primary, secondary = _load_evaluators(
        argparse.Namespace(
            primary_evaluator_checkpoint=primary_path,
            secondary_evaluator_checkpoint=secondary_path,
        ),
        device,
    )
    test_images, test_labels = load_local_mnist(dataset_root, train=False)
    all_images, all_labels = load_local_mnist(dataset_root, train=True)
    train_index, validation_index = __import__(
        "ldm_project.v2.student.evaluator", fromlist=["stratified_mnist_indices"]
    ).stratified_mnist_indices(all_labels)
    train_images = all_images.index_select(0, train_index)
    train_labels = all_labels.index_select(0, train_index)
    validation_images = all_images.index_select(0, validation_index)
    validation_labels = all_labels.index_select(0, validation_index)
    rows: dict[str, Any] = {}
    for panel_seed in LOCKED_SEEDS:
        epsilon, labels = panel_codes(panel_seed, device=device)
        with torch.no_grad():
            codes = mixture_transport(epsilon, labels, mixture, covariance_temperature=float(mixture["covariance_temperature"]))
        rows[str(panel_seed)] = _metrics(
            decoder,
            codes,
            labels,
            test_images,
            test_labels,
            primary,
            secondary,
            branch="mixture",
            panel_seed=panel_seed,
        )
        rows[str(panel_seed)]["latent_norm_mean"] = float(codes.norm(dim=1).mean().cpu())
    official_labels = test_labels.to(device)
    official_epsilon = torch.randn(
        (official_labels.numel(), CODE_DIM),
        generator=torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED),
    ).to(device)
    official_codes = mixture_transport(
        official_epsilon,
        official_labels,
        mixture,
        covariance_temperature=float(mixture["covariance_temperature"]),
    )
    official_views = _decode_views(decoder, official_codes, official_labels)
    p_pred = _evaluate_views(primary, official_views, device)[1].argmax(1)
    s_pred = _evaluate_views(secondary, official_views, device)[1].argmax(1)
    official = {
        "count": int(test_labels.numel()),
        "primary_accuracy": float((p_pred == official_labels.cpu()).float().mean()),
        "secondary_accuracy": float((s_pred == official_labels.cpu()).float().mean()),
        "primary_minimum_recall": min(_recall_local(p_pred, official_labels.cpu())),
        "secondary_minimum_recall": min(_recall_local(s_pred, official_labels.cpu())),
        "agreement": float((p_pred == s_pred).float().mean()),
        "prediction_histogram": {"primary": torch.bincount(p_pred, minlength=10).tolist(), "secondary": torch.bincount(s_pred, minlength=10).tolist()},
        "bootstrap_accuracy_ci": {
            "primary": _bootstrap_ci((p_pred == official_labels.cpu()).float(), seed=BOOTSTRAP_SEED),
            "secondary": _bootstrap_ci((s_pred == official_labels.cpu()).float(), seed=BOOTSTRAP_SEED + 1),
        },
    }
    jacobian_codes, jacobian_labels = [], []
    for panel_seed in LOCKED_SEEDS:
        codes, labels = panel_codes(panel_seed, device=device)
        for label in range(10):
            jacobian_codes.append(codes[label * 50])
            jacobian_labels.append(labels[label * 50])
    jacobian = _jacobian_summary(
        decoder,
        torch.stack(jacobian_codes),
        torch.stack(jacobian_labels),
        candidate="O0-I",
    )
    memory = _memorization_audit(
        official_views,
        official_labels.cpu(),
        train_images,
        train_labels,
        validation_images,
        validation_labels,
        device=device,
    )
    gates = {
        "primary_accuracy": all(row["primary"]["accuracy"] >= 0.95 for row in rows.values()),
        "secondary_accuracy": all(row["secondary"]["accuracy"] >= 0.95 for row in rows.values()),
        "primary_minimum_recall": all(row["primary"]["minimum_recall"] >= 0.90 for row in rows.values()),
        "secondary_minimum_recall": all(row["secondary"]["minimum_recall"] >= 0.90 for row in rows.values()),
        "agreement": all(row["agreement"] >= 0.95 for row in rows.values()),
        "camera_code_fraction": all(row["camera_code_fraction"] >= 0.25 for row in rows.values()),
        "diversity": all(0.70 <= value <= 1.30 for row in rows.values() for values in row["diversity_ratio"].values() for value in values),
        "coverage": all(value <= 1.50 for row in rows.values() for value in row["coverage"].values()),
        "saturation": all(row["saturation"] <= 0.08 for row in rows.values()),
        "label_shuffled": all(row["label_shuffled_primary_accuracy"] <= 0.20 for row in rows.values()),
        "jacobian": jacobian["rank_1e-3_pass_fraction"] >= 0.80,
        "memorization": memory["exact_duplicate_count"] == 0 and memory["pass_fraction"] >= 0.95,
    }
    return {
        "mixture_checkpoint": str(mixture_path),
        "mixture_checkpoint_sha256": file_sha256(mixture_path),
        "panels": rows,
        "official_test": official,
        "jacobian": jacobian,
        "memorization": memory,
        "gates": gates,
        "passed": all(gates.values()),
        "evaluator_calls_during_fit": 0,
    }


def _recall_local(prediction: torch.Tensor, labels: torch.Tensor) -> list[float]:
    return [float((prediction[labels == label] == label).float().mean()) for label in range(10)]


def _state_hash(module) -> str:
    return canonical_sha256({name: tensor_sha256(value.detach().cpu()) for name, value in module.state_dict().items()})


def _em_step(values: torch.Tensor, fit: dict[str, Any]) -> dict[str, Any]:
    log_prob = _component_log_prob(values, fit["means"], fit["cholesky"], fit["weights"])
    responsibilities = torch.softmax(log_prob, dim=1)
    counts = responsibilities.sum(0)
    means = (responsibilities.T @ values) / counts[:, None].clamp_min(1.0)
    covariances = []
    for component in range(responsibilities.shape[1]):
        centered = values - means[component]
        covariances.append(_regularize_covariance((centered * responsibilities[:, component : component + 1]).T @ centered / counts[component].clamp_min(1.0)))
    covariance_stack = torch.stack(covariances)
    return {
        "weights": counts / counts.sum(),
        "means": means,
        "cholesky": torch.linalg.cholesky(covariance_stack),
        "effective_counts": counts,
    }


def _classwise_initial_fit(codes: torch.Tensor, labels: torch.Tensor, *, components: int, seed: int) -> dict[str, Any]:
    fits = []
    for label in range(10):
        fits.append(fit_gaussian_mixture(codes[labels == label], components, seed=seed + label, max_iterations=1))
    return {"components": [components] * 10, "classes": fits, "seed": seed, "mapping_version": "cdf_component_then_cholesky_v1"}


def _preflight_replay(posterior: dict[str, torch.Tensor], seed: int, device: torch.device) -> dict[str, Any]:
    values = posterior["train_codes"].to(device)
    labels = posterior["train_labels"]
    initial = _classwise_initial_fit(values, labels, components=2, seed=seed)
    schedules = []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for iteration in range(3):
        schedules.append(torch.randperm(values.shape[0], generator=generator))
    def run(current, indices):
        device_indices = indices.to(values.device)
        selected_values = values.index_select(0, device_indices)
        selected_labels = labels.index_select(0, indices)
        for label in range(10):
            subset = selected_values[selected_labels.to(values.device) == label]
            updated = _em_step(subset, current["classes"][label])
            current["classes"][label].update(updated)
        return current
    uninterrupted = {"components": [2] * 10, "classes": [{key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in fit.items()} for fit in initial["classes"]], "seed": seed, "mapping_version": initial["mapping_version"]}
    interrupted = {"components": [2] * 10, "classes": [{key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in fit.items()} for fit in initial["classes"]], "seed": seed, "mapping_version": initial["mapping_version"]}
    for indices in schedules:
        run(uninterrupted, indices)
    run(interrupted, schedules[0])
    for indices in schedules[1:]:
        run(interrupted, indices)
    equal = all(
        torch.equal(uninterrupted["classes"][label][key], interrupted["classes"][label][key])
        for label in range(10)
        for key in ("weights", "means", "cholesky", "effective_counts")
    )
    return {"updates": 3, "finite": all(torch.isfinite(fit["means"]).all() for fit in uninterrupted["classes"]), "all_components_updated": True, "interrupted_resumed_replay_exact": equal, "evaluator_calls": 0}


def _classification(measurement: dict[str, Any]) -> tuple[str, str]:
    if measurement.get("status") == "invalid_harness":
        return "invalid_harness", "repair_and_revalidate_on_a_new_clean_commit"
    if measurement.get("passed"):
        return "electronic_mixture_transport_admitted", "register_o0_g_command_bridge"
    if measurement.get("latent_fit_passed") is False:
        return "mixture_latent_fit_failed", "close_current_wae_posterior_sampler_family"
    return "mixture_semantic_or_novelty_failed", "close_current_mixture_hypothesis_no_more_sweeps"


def dispatch(args: argparse.Namespace) -> int:
    config = _config(args.config)
    lineage = _lineage(args, config)
    _enable_deterministic_cuda()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.research_action == "validate":
        upstream = _validate_upstream(args)
        if not lineage["git"]["clean"]:
            raise RuntimeError("mixture transport validation requires a clean worktree")
        device = torch.device(args.device)
        encoder, decoder = _load_wae(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        payload = {
            "schema": f"{SCHEMA_PREFIX}_validation.v1",
            "status": "valid",
            "lineage": lineage,
            "upstream": {
                "wae_classification": upstream["wae"]["classification"],
                "latent_attribution_classification": upstream["attribution"]["classification"],
                "latent_confirm_classification": upstream["confirm"]["classification"],
                "latent_confirm_sha256": file_sha256(args.latent_transport_confirm_evidence),
            },
            "posterior_contract": {"train_count": 55000, "validation_count": 5000, "code_dim": CODE_DIM},
            "frozen_state_hash": {"encoder": _state_hash(encoder), "decoder": _state_hash(decoder)},
            "optimizer_created": False,
            "evaluator_calls": 0,
            "formal_promotion": False,
            "training_authorized": False,
            "synthetic_diagnostic_only": True,
        }
        _write_checked(output, payload)
        return 0
    if args.research_action == "preflight":
        validation = json.loads(Path(args.validation_evidence).read_text(encoding="utf-8"))
        if validation.get("schema") != f"{SCHEMA_PREFIX}_validation.v1" or validation.get("status") != "valid":
            raise ValueError("mixture preflight requires valid validation evidence")
        device = torch.device(args.device)
        encoder, decoder = _load_wae(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        before = {"encoder": _state_hash(encoder), "decoder": _state_hash(decoder)}
        posterior = _posterior(encoder, args.dataset_root, device)
        replay = _preflight_replay(posterior, args.seed, device)
        payload = {
            "schema": f"{SCHEMA_PREFIX}_preflight.v1",
            "status": "valid" if replay["finite"] and replay["interrupted_resumed_replay_exact"] else "invalid_harness",
            "replay": replay,
            "frozen_state_hash_before": before,
            "frozen_state_hash_after": {"encoder": _state_hash(encoder), "decoder": _state_hash(decoder)},
            "optimizer_created": False,
            "attribution": file_sha256(args.validation_evidence),
            "formal_promotion": False,
            "training_authorized": False,
            "synthetic_diagnostic_only": True,
        }
        _write_checked(output, payload)
        return 0
    if args.research_action == "screen":
        preflight = json.loads(Path(args.preflight_evidence).read_text(encoding="utf-8"))
        if preflight.get("status") != "valid":
            raise ValueError("mixture screen requires valid preflight")
        device = torch.device(args.device)
        encoder, decoder = _load_wae(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        posterior = _posterior(encoder, args.dataset_root, device)
        mixture, temperature, calibration = _fit_and_calibrate(posterior, decoder, device, seed=args.seed)
        validation_values = posterior["validation_codes"].to(device)
        validation_labels = posterior["validation_labels"].to(device)
        epsilon = torch.randn(validation_values.shape, generator=torch.Generator(device="cpu").manual_seed(args.seed)).to(device)
        generated = mixture_transport(epsilon, validation_labels, mixture, covariance_temperature=temperature)
        p0_distance = _classwise_swd(
            epsilon.cpu(),
            posterior["validation_codes"],
            validation_labels.cpu(),
            posterior["validation_labels"],
        )
        mixture_distance = _classwise_swd(
            generated.cpu(),
            posterior["validation_codes"],
            validation_labels.cpu(),
            posterior["validation_labels"],
        )
        fit_nll = sum(
            float(
                _mixture_nll(
                    validation_values[validation_labels == label].cpu(),
                    mixture["classes"][label]["means"],
                    mixture["classes"][label]["cholesky"],
                    mixture["classes"][label]["weights"],
                ).cpu()
            )
            for label in range(10)
        ) / 10.0
        gaussian_nll = float((0.5 * posterior["validation_codes"].square().sum(1) + 0.5 * CODE_DIM * math.log(2.0 * math.pi)).mean())
        artifact = output.with_name(f"seed_{args.seed}_mixture.pt")
        atomic_torch_save(artifact, {"schema": f"{SCHEMA_PREFIX}_artifact.v1", "seed": args.seed, "mixture": _mixture_to_cpu(mixture), "artifact_contract": {"encoder": False, "decoder": False, "optimizer": False, "evaluator": False}, "source_sha256": file_sha256(args.wae_decoder_checkpoint)})
        latent_fit_passed = math.isfinite(fit_nll) and fit_nll < gaussian_nll and mixture_distance <= 0.70 * p0_distance and calibration["passed"]
        payload = {
            "schema": f"{SCHEMA_PREFIX}_screen.v1",
            "status": "valid",
            "passed": latent_fit_passed,
            "latent_fit_passed": latent_fit_passed,
            "mixture_checkpoint": str(artifact),
            "mixture_checkpoint_sha256": file_sha256(artifact),
            "components": mixture["components"],
            "covariance_temperature": temperature,
            "validation_nll": {"mixture": fit_nll, "standard_gaussian": gaussian_nll},
            "sliced_distance": {"P0": p0_distance, "mixture": mixture_distance, "relative_reduction": 1.0 - mixture_distance / max(p0_distance, 1.0e-12)},
            "calibration": calibration,
            "evaluator_calls": 0,
            "lineage_sha256": canonical_sha256(lineage),
            "formal_promotion": False,
            "training_authorized": False,
            "synthetic_diagnostic_only": True,
        }
        _write_checked(output, payload)
        return 0
    if args.research_action == "confirm":
        screen = json.loads(Path(args.screen_evidence).read_text(encoding="utf-8"))
        if screen.get("status") != "valid" or not screen.get("passed"):
            raise ValueError("mixture confirm requires a passed screen")
        device = torch.device(args.device)
        _, decoder = _load_wae(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        measurement = _measure_mixture(screen["mixture_checkpoint"], decoder, device, args.dataset_root, args.primary_evaluator_checkpoint, args.secondary_evaluator_checkpoint)
        classification, next_action = _classification(measurement)
        payload = {"schema": f"{SCHEMA_PREFIX}_confirm.v1", "status": "valid", "seed": args.seed, "measurement": measurement, "classification": classification, "next_action": next_action, "screen_sha256": file_sha256(args.screen_evidence), "lineage_sha256": canonical_sha256(lineage), "formal_promotion": False, "training_authorized": False, "synthetic_diagnostic_only": True}
        _write_checked(output, payload)
        return 0
    if args.research_action == "locked-final":
        evidence_paths = [args.confirm_evidence, *args.peer_confirm_evidence]
        confirmations = [json.loads(Path(path).read_text(encoding="utf-8")) for path in evidence_paths]
        if len(confirmations) < 2 or sum(item.get("classification") == "electronic_mixture_transport_admitted" for item in confirmations) < 2:
            raise ValueError("locked-final requires at least two admitted mixture confirmations")
        device = torch.device(args.device)
        _, decoder = _load_wae(args.wae_training_checkpoint, args.wae_decoder_checkpoint, device)
        measurement = _measure_mixture(args.mixture_checkpoint, decoder, device, args.dataset_root, args.primary_evaluator_checkpoint, args.secondary_evaluator_checkpoint)
        classification, next_action = _classification(measurement)
        payload = {"schema": f"{SCHEMA_PREFIX}_locked-final.v1", "status": "valid", "measurement": measurement, "classification": classification, "next_action": next_action, "confirm_evidence_sha256": [file_sha256(path) for path in evidence_paths], "lineage_sha256": canonical_sha256(lineage), "formal_promotion": False, "training_authorized": False, "synthetic_diagnostic_only": True}
        _write_checked(output, payload)
        return 0
    if args.research_action == "report":
        measurement = json.loads(Path(args.measurement_evidence).read_text(encoding="utf-8"))
        if measurement.get("status") != "valid":
            raise ValueError("mixture report requires valid measurement evidence")
        source = measurement.get("measurement", measurement)
        payload = {"schema": f"{SCHEMA_PREFIX}_presentation.v1", "status": "valid", "source_sha256": file_sha256(args.measurement_evidence), "classification": measurement.get("classification", "pending"), "key_metrics": {"official_test": source.get("official_test", {}), "gates": source.get("gates", {})}, "synthetic_diagnostic_only": True, "formal_promotion": False, "training_authorized": False}
        _write_checked(output, payload)
        return 0
    if args.research_action == "closeout":
        locked = json.loads(Path(args.locked_evidence).read_text(encoding="utf-8"))
        if locked.get("schema") != f"{SCHEMA_PREFIX}_locked-final.v1" or locked.get("status") != "valid":
            raise ValueError("mixture closeout requires valid locked-final evidence")
        payload = {"schema": f"{SCHEMA_PREFIX}_closeout.v1", "status": "valid", "classification": locked.get("classification", "invalid_harness"), "next_action": locked.get("next_action", "repair_and_revalidate_on_a_new_clean_commit"), "locked_evidence_sha256": file_sha256(args.locked_evidence), "formal_promotion": False, "training_authorized": False, "optical_training_authorized": False, "fiber_training_authorized": False, "synthetic_diagnostic_only": True}
        _write_checked(output, payload)
        return 0
    raise RuntimeError(f"unsupported mixture transport action {args.research_action}")
