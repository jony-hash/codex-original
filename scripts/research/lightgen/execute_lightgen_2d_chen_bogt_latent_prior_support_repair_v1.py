"""Decompose the frozen Chen BOGT amplitude-prior support mismatch on CPU.

This consumer reads only the frozen final TRAIN amplitude collection, the
completed sampled-generation canary report, its latent bank, and the frozen
generation bundle identity.  It never loads images or labels, constructs an
optical runtime, uses CUDA, or changes the frozen bundle.  P1 and P2 are
post-train sampling candidates, not replacements for the historical training
objective.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.application.research.shared.research_run_control import HeartbeatWriter, RUN_COMPLETED, RUN_PREPARING, RUNNING, capture_execution_source_manifest
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research.lightgen import execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1 as canary


TASK_ID = "DECOMPOSE_AND_REPAIR_CHEN_BOGT_LATENT_PRIOR_SUPPORT_V1"
OUTER_GATE = "CHEN_BOGT_LATENT_PRIOR_SUPPORT_REPAIR_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "chen-bogt-latent-prior-support-repair-v1"
TRAJECTORY_ID = "chen-bogt-latent-prior-support-repair-trajectory-v1"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"

TRAIN_RUN = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01")
BUNDLE_PATH = TRAIN_RUN / "generation_ready_bundle.pt"
TRAINING_REPORT_PATH = TRAIN_RUN / "report.json"
FINAL_COLLECTION_PATH = TRAIN_RUN / "final_train_ols.pt"

CANARY_RUN = Path(
    "outputs/20260918_chen_bogt_complex_ols_sampled_generation_canary_v1/runs/attempt_01"
)
CANARY_REPORT_PATH = CANARY_RUN / "sampled_generation_canary_report.json"
LATENT_BANK_PATH = CANARY_RUN / "sampled_latent_bank.pt"

EXPECTED_BUNDLE_SHA256 = "4c9a8d10ab1694d69aeaf9e58c5637c2695088a5804b15f4b976e15cc489439f"
EXPECTED_TRAINING_REPORT_SHA256 = "e2e87a9aece563616f9e64b329411a4bcd9a0a3f6dcbcab7f220e690bee15dfc"
EXPECTED_FINAL_COLLECTION_SHA256 = "2d1264219faa0964467d40f2e4882ec3767d3a1aafd30e143bf406aa39be6dcc"
EXPECTED_CANARY_REPORT_SHA256 = "60e9a8852dd1f03839a5d88940c15242f6324591ae79f521bae5e41da3c0c7fa"
EXPECTED_LATENT_BANK_SHA256 = "09f8ccd01738862c79f9a3ed8792b155fffb5dad907b36c1e9a1bfab691cce6d"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260918_chen_bogt_latent_prior_support_repair_v1/runs/attempt_01"
)

N_TOTAL = 55_000
DIMENSION = 100
FIT_COUNT = 45_000
EVAL_COUNT = 5_000
COMPARISON_COUNT = 1_000

SPLIT_SEED = 20_260_922
P1_SEED = 20_260_923
P2_SEED = 20_260_924
P3_SEED = 20_260_925

LOG_EPS = 1.0e-12
STANDARD_DEVIATION_EPS = 1.0e-12
PAIRWISE_CHUNK_SIZE = 128
DIAGONAL_ATOL = float(torch.finfo(torch.float64).eps * 256.0)

REPORT_SCHEMA = "chen_bogt_latent_prior_support_decomposition_report.v1"
P1_SCHEMA = "chen_bogt_empirical_independent_u_bank.v1"
P2_SCHEMA = "chen_bogt_empirical_gaussian_copula_u_bank.v1"
P3_SCHEMA = "chen_bogt_real_row_support_control_u_bank.v1"
SPLIT_SCHEMA = "chen_bogt_latent_prior_support_split_manifest.v1"

CORE_SCALAR_METRICS = (
    "R_u",
    "covariance_relative_frobenius",
    "correlation_relative_frobenius",
    "norm_distribution_energy_distance",
    "mean_abs_off_diagonal_correlation_difference",
    "mean_abs_per_port_mean_difference",
    "mean_abs_per_port_std_difference",
)


class RepairError(RuntimeError):
    """Raised when the frozen latent-only repair boundary is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is non-finite")


def _float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _git_state(root: Path) -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "-C", str(root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    status_sha256 = hashlib.sha256("\n".join(status_lines).encode("utf-8")).hexdigest()
    return {
        "branch": branch,
        "head": head,
        "dirty": bool(status_lines),
        "status_sha256": status_sha256,
    }


def _relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    return {
        "path": _relative_path(root, path),
        "sha256": file_sha256(path),
        "role": role,
    }


def _ensure_output(root: Path, output_dir: Path) -> Path:
    target = output_dir if output_dir.is_absolute() else root / output_dir
    target = target.resolve()
    _require(root.resolve() in target.parents, "output directory must remain inside the repository")
    _require(target.relative_to(root).as_posix().startswith("outputs/"), "output must remain below outputs/")
    _require(not target.exists(), f"refusing to reuse write-once output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    probe = target / ".write_probe"
    probe.write_text("latent prior support repair\n", encoding="utf-8")
    probe.unlink()
    return target


def _load_json(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    _require(path.is_file(), f"{name} is missing: {path}")
    observed = file_sha256(path)
    _require(observed == expected_sha256, f"{name} hash changed: {path}")
    return dict(_mapping(json.loads(path.read_text(encoding="utf-8")), name))


def _load_authorities(root: Path) -> dict[str, Any]:
    bundle_path = root / BUNDLE_PATH
    training_report_path = root / TRAINING_REPORT_PATH
    final_path = root / FINAL_COLLECTION_PATH
    canary_report_path = root / CANARY_REPORT_PATH
    latent_path = root / LATENT_BANK_PATH

    bundle = safe_torch_load(bundle_path, expected_sha256=EXPECTED_BUNDLE_SHA256, map_location="cpu")
    _require(isinstance(bundle, Mapping), "generation-ready bundle is not a mapping")
    training_report = _load_json(training_report_path, EXPECTED_TRAINING_REPORT_SHA256, "frozen training report")
    canary_report = _load_json(canary_report_path, EXPECTED_CANARY_REPORT_SHA256, "sampled-generation canary report")
    final_payload = safe_torch_load(final_path, expected_sha256=EXPECTED_FINAL_COLLECTION_SHA256, map_location="cpu")
    latent = safe_torch_load(latent_path, expected_sha256=EXPECTED_LATENT_BANK_SHA256, map_location="cpu")

    _require(training_report.get("result") == "VALID_COMPLETED_GENERATION_BUNDLE_READY", "upstream training result changed")
    _require(training_report.get("project_mainline") == SEMANTIC_AUTHORITY_ID, "upstream mainline changed")
    _require(training_report.get("current_authorized_next_action") == "NONE", "upstream next action changed")
    _require(canary_report.get("status") == "VALID_COMPLETED", "sampled-generation canary is not complete")
    _require(canary_report.get("outer_gate") == "CHEN_BOGT_SAMPLED_GENERATION_CANARY_V1_RESOLVED", "sampled-generation canary gate changed")
    _require(canary_report.get("project_mainline") == SEMANTIC_AUTHORITY_ID, "canary mainline changed")
    _require(canary_report.get("current_authorized_next_action") == "NONE", "canary next action changed")
    _require(canary_report.get("training_executed") is False, "canary training flag changed")
    data_reads = _mapping(canary_report.get("data_reads"), "canary data-read boundary")
    _require(data_reads.get("labels_read") == 0 and data_reads.get("raw_images_read") == 0, "canary image/label boundary changed")
    _require(data_reads.get("protected_read") == 0 and data_reads.get("test_read") == 0 and data_reads.get("sealed_read") == 0, "canary protected/test/sealed boundary changed")
    _require(canary_report.get("stage_f_frozen_evaluator", {}).get("labels_consumed") is False, "canary evaluator label boundary changed")
    _require(canary_report.get("scientific_facts_modified") is False, "canary scientific-fact boundary changed")

    _require(isinstance(final_payload, Mapping), "final TRAIN collection is not a mapping")
    _require(final_payload.get("schema") == "chen_bogt_final_train_ols_collection.v1", "final collection schema changed")
    u_final = final_payload.get("u_final")
    source_indices = final_payload.get("source_indices")
    _require(isinstance(u_final, torch.Tensor) and isinstance(source_indices, torch.Tensor), "final collection tensors are missing")
    _require(tuple(u_final.shape) == (N_TOTAL, DIMENSION) and u_final.dtype == torch.float64, "final u shape or dtype changed")
    _require(tuple(source_indices.shape) == (N_TOTAL,) and source_indices.dtype == torch.int64, "final source index shape changed")
    _finite(u_final, "final TRAIN u")
    _require(bool(u_final.ge(0.0).all().item()), "final TRAIN u is negative")
    _require(torch.unique(source_indices).numel() == N_TOTAL, "final TRAIN source indices are not unique")

    _require(isinstance(latent, Mapping), "sampled latent bank is not a mapping")
    _require(latent.get("schema") == "chen_bogt_sampled_latent_bank.v1", "sampled latent bank schema changed")
    u_sample = latent.get("u_sample")
    _require(isinstance(u_sample, torch.Tensor) and tuple(u_sample.shape) == (1_000, DIMENSION), "frozen u_sample shape changed")
    _require(u_sample.dtype == torch.float64, "frozen u_sample dtype changed")
    _finite(u_sample, "frozen u_sample")
    _require(bool(u_sample.ge(0.0).all().item()), "frozen u_sample is negative")
    _require(latent.get("selected_step") == 300, "sampled latent selected step changed")
    _require(latent.get("source_bundle_sha256") == EXPECTED_BUNDLE_SHA256, "sampled latent bundle identity changed")

    return {
        "bundle": bundle,
        "training_report": training_report,
        "canary_report": canary_report,
        "final_payload": final_payload,
        "latent": latent,
        "u_final": u_final.to(dtype=torch.float64, device="cpu").contiguous(),
        "source_indices": source_indices.to(dtype=torch.int64, device="cpu").contiguous(),
        "u_sample": u_sample.to(dtype=torch.float64, device="cpu").contiguous(),
        "input_hashes": {
            "bundle": EXPECTED_BUNDLE_SHA256,
            "training_report": EXPECTED_TRAINING_REPORT_SHA256,
            "final_collection": EXPECTED_FINAL_COLLECTION_SHA256,
            "canary_report": EXPECTED_CANARY_REPORT_SHA256,
            "latent_bank": EXPECTED_LATENT_BANK_SHA256,
        },
    }


def _make_split(u_final: torch.Tensor, source_indices: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SPLIT_SEED)
    permutation = torch.randperm(N_TOTAL, generator=generator)
    fit_positions = permutation[:FIT_COUNT]
    eval_a_positions = permutation[FIT_COUNT : FIT_COUNT + EVAL_COUNT]
    eval_b_positions = permutation[FIT_COUNT + EVAL_COUNT :]
    union = torch.cat([fit_positions, eval_a_positions, eval_b_positions])
    _require(torch.unique(union).numel() == N_TOTAL, "FIT/EVAL split is not mutually exclusive")
    split = {
        "schema": SPLIT_SCHEMA,
        "seed": SPLIT_SEED,
        "source_row_count": N_TOTAL,
        "fit_count": FIT_COUNT,
        "eval_a_count": EVAL_COUNT,
        "eval_b_count": EVAL_COUNT,
        "comparison_count_per_eval": COMPARISON_COUNT,
        "comparison_selection": "first 1000 rows of each deterministic EVAL partition",
        "permutation_sha256": tensor_sha256(permutation),
        "fit_positions_sha256": tensor_sha256(fit_positions),
        "eval_a_positions_sha256": tensor_sha256(eval_a_positions),
        "eval_b_positions_sha256": tensor_sha256(eval_b_positions),
        "fit_source_indices": [int(value) for value in source_indices.index_select(0, fit_positions).tolist()],
        "eval_a_source_indices": [int(value) for value in source_indices.index_select(0, eval_a_positions).tolist()],
        "eval_b_source_indices": [int(value) for value in source_indices.index_select(0, eval_b_positions).tolist()],
    }
    return (
        split,
        u_final.index_select(0, fit_positions).contiguous(),
        u_final.index_select(0, eval_a_positions[:COMPARISON_COUNT]).contiguous(),
        u_final.index_select(0, eval_b_positions[:COMPARISON_COUNT]).contiguous(),
    )


def _standardization(fit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h_fit = torch.log(fit + LOG_EPS)
    mean_h = h_fit.mean(dim=0)
    std_h = h_fit.std(dim=0, unbiased=False)
    _require(bool(torch.isfinite(mean_h).all()) and bool(torch.isfinite(std_h).all()), "FIT log-u statistics are non-finite")
    _require(bool(std_h.gt(0.0).all()), "FIT log-u standard deviation is zero")
    return mean_h, std_h, h_fit


def _standardize(value: torch.Tensor, mean_h: torch.Tensor, std_h: torch.Tensor) -> torch.Tensor:
    result = (torch.log(value + LOG_EPS) - mean_h) / (std_h + STANDARD_DEVIATION_EPS)
    _finite(result, "standardized log-u")
    return result


def _fit_gaussian_dependence(scores: torch.Tensor) -> dict[str, Any]:
    n = int(scores.shape[0])
    _require(tuple(scores.shape[1:]) == (DIMENSION,), "Gaussianized score shape is invalid")
    _finite(scores, "Gaussianized scores")
    mean_fit = scores.mean(dim=0)
    centered = scores - mean_fit
    standard_deviation = torch.sqrt(centered.square().sum(dim=0).div(float(n - 1)))
    _require(bool(standard_deviation.gt(0.0).all()), "Gaussianized score standard deviation is zero")
    standardized = centered / standard_deviation
    empirical = standardized.transpose(0, 1).mm(standardized).div(float(n - 1))
    tau = torch.trace(empirical).div(float(DIMENSION))
    target = tau * torch.eye(DIMENSION, dtype=torch.float64)
    quadratic = (standardized.mm(empirical) * standardized).sum(dim=1)
    beta = (
        standardized.square().sum(dim=1).square().sum()
        - 2.0 * quadratic.sum()
        + float(n) * empirical.square().sum()
    ).div(float(n * n))
    beta = beta.clamp_min(0.0)
    delta = (empirical - target).square().sum()
    alpha = torch.zeros((), dtype=torch.float64) if float(delta) == 0.0 else (beta / delta).clamp(0.0, 1.0)
    shrunk = ((1.0 - alpha) * empirical + alpha * target)
    shrunk = (shrunk + shrunk.transpose(0, 1)).mul(0.5).contiguous()
    _finite(shrunk, "R_shrunk")
    cholesky, info = torch.linalg.cholesky_ex(shrunk, check_errors=False)
    _require(int(info.item()) == 0, "R_shrunk Cholesky decomposition failed")
    eigenvalues = torch.linalg.eigvalsh(shrunk)
    _require(bool(torch.isfinite(eigenvalues).all()) and bool(eigenvalues.gt(0.0).all()), "R_shrunk is not positive definite")
    diagonal_error = (torch.diagonal(shrunk) - 1.0).abs().max()
    _require(float(diagonal_error) <= DIAGONAL_ATOL, "R_shrunk unit diagonal invariant failed")
    return {
        "R_emp": empirical,
        "R_shrunk": shrunk,
        "cholesky": cholesky,
        "eigenvalues": eigenvalues,
        "tau": tau,
        "beta": beta,
        "delta": delta,
        "alpha": alpha,
        "diagonal_error": diagonal_error,
        "mean_fit": mean_fit,
        "standard_deviation": standard_deviation,
    }


def _fit_empirical_marginals(fit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    values = fit.to(dtype=torch.float64).contiguous()
    _require(values.ndim == 2 and tuple(values.shape[1:]) == (DIMENSION,), "FIT marginal shape is invalid")
    _finite(values, "FIT marginal values")
    n = int(values.shape[0])
    positions = torch.empty_like(values)
    sorted_values = torch.empty_like(values)
    tie_blocks = 0
    tied_rows = 0
    maximum_tie_block = 1
    coordinates_with_ties = 0
    for coordinate in range(DIMENSION):
        order = torch.argsort(values[:, coordinate], stable=True)
        ordered = values.index_select(0, order)[:, coordinate]
        sorted_values[:, coordinate] = ordered
        starts = torch.cat(
            [
                torch.zeros(1, dtype=torch.long),
                torch.nonzero(ordered[1:] != ordered[:-1], as_tuple=False).reshape(-1) + 1,
            ]
        )
        ends = torch.cat([starts[1:], torch.tensor([n], dtype=torch.long)])
        block_sizes = ends - starts
        block_positions = ((starts.to(torch.float64) + 1.0) + ends.to(torch.float64)).div(2.0)
        plotting_positions = (block_positions - 0.5) / float(n)
        positions[order, coordinate] = torch.repeat_interleave(plotting_positions, block_sizes)
        tied = block_sizes.gt(1)
        if bool(tied.any()):
            coordinates_with_ties += 1
            tie_blocks += int(tied.sum())
            tied_rows += int(block_sizes[tied].sum())
            maximum_tie_block = max(maximum_tie_block, int(block_sizes.max()))
    _require(bool(positions.gt(0.0).all()) and bool(positions.lt(1.0).all()), "FIT plotting positions are not strictly interior")
    return positions, sorted_values, {
        "sorting": "stable value-then-original-row-index order",
        "fit_rows": n,
        "coordinates": DIMENSION,
        "coordinates_with_ties": coordinates_with_ties,
        "tie_block_count": tie_blocks,
        "tied_row_count": tied_rows,
        "maximum_tie_block": maximum_tie_block,
        "strictly_interior": True,
        "epsilon_clamp": False,
        "implementation": "vectorized stable-tie equivalent of the existing N3 midrank primitive",
    }


def _fit_marginals_and_copula(fit: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    positions, sorted_values, tie_summary = _fit_empirical_marginals(fit)
    gaussian_scores = torch.special.ndtri(positions)
    _finite(gaussian_scores, "Gaussianized empirical ranks")
    dependence = _fit_gaussian_dependence(gaussian_scores)
    dependence["gaussian_scores"] = gaussian_scores
    return sorted_values, {
        "positions": positions,
        "sorted_values": sorted_values,
        "tie_summary": tie_summary,
        "dependence": dependence,
    }


def _empirical_quantile_batch(sorted_values: torch.Tensor, probabilities: torch.Tensor) -> torch.Tensor:
    values = sorted_values.to(dtype=torch.float64).contiguous()
    probs = probabilities.to(dtype=torch.float64).contiguous()
    _require(values.ndim == 2 and values.shape[1] == DIMENSION, "quantile table shape is invalid")
    _require(probs.ndim == 2 and probs.shape[1] == DIMENSION, "quantile probability shape is invalid")
    _finite(values, "quantile table")
    _finite(probs, "quantile probabilities")
    count = int(values.shape[0])
    _require(count >= 2, "quantile table is too small")
    positions = (torch.arange(count, dtype=torch.float64) + 0.5) / float(count)
    result = torch.empty_like(probs)
    for coordinate in range(DIMENSION):
        probability = probs[:, coordinate].contiguous()
        low = probability.le(positions[0])
        high = probability.ge(positions[-1])
        upper = torch.searchsorted(positions, probability, right=True).clamp(min=1, max=count - 1)
        lower = upper - 1
        lower_position = positions.index_select(0, lower)
        upper_position = positions.index_select(0, upper)
        fraction = (probability - lower_position) / (upper_position - lower_position)
        interpolated = values[:-1, coordinate].index_select(0, lower) + fraction * (
            values[1:, coordinate].index_select(0, lower)
            - values[:-1, coordinate].index_select(0, lower)
        )
        result[:, coordinate] = torch.where(
            low,
            values[0, coordinate],
            torch.where(high, values[-1, coordinate], interpolated),
        )
    _finite(result, "empirical quantile output")
    _require(bool(result.ge(0.0).all()), "empirical quantile output is negative")
    return result


def _sample_empirical_independent(sorted_values: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = torch.Generator(device="cpu").manual_seed(P1_SEED)
    probabilities = torch.rand((COMPARISON_COUNT, DIMENSION), generator=generator, dtype=torch.float64)
    sample = _empirical_quantile_batch(sorted_values, probabilities)
    return sample, {
        "seed": P1_SEED,
        "rng": "torch.Generator(device='cpu')",
        "probability_source": "independent Uniform(0,1) per port",
        "quantile_primitive": "local._empirical_quantile_batch",
    }


def _sample_empirical_gaussian_copula(fit_state: Mapping[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    dependence = _mapping(fit_state["dependence"], "Gaussian-copula dependence")
    generator = torch.Generator(device="cpu").manual_seed(P2_SEED)
    gaussian_noise = torch.randn((COMPARISON_COUNT, DIMENSION), generator=generator, dtype=torch.float64)
    gaussian_sample = gaussian_noise.mm(dependence["cholesky"].transpose(0, 1))
    probabilities = torch.special.ndtr(gaussian_sample)
    sample = _empirical_quantile_batch(fit_state["sorted_values"], probabilities)
    return sample, {
        "seed": P2_SEED,
        "rng": "torch.Generator(device='cpu')",
        "gaussian_sampling": "x~N(0,R_shrunk); u=Phi(x); q=fit empirical inverse marginal",
        "gaussian_noise": gaussian_noise,
        "gaussian_sample": gaussian_sample,
        "probabilities": probabilities,
    }


def _sample_real_row_control(fit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(P3_SEED)
    positions = torch.randperm(FIT_COUNT, generator=generator)[:COMPARISON_COUNT]
    return fit.index_select(0, positions).contiguous(), positions


def _covariance(value: torch.Tensor) -> torch.Tensor:
    centered = value - value.mean(dim=0, keepdim=True)
    return centered.transpose(0, 1).mm(centered).div(float(value.shape[0]))


def _correlation(covariance: torch.Tensor) -> torch.Tensor:
    scale = covariance.diagonal().clamp_min(0.0).sqrt()
    denominator = scale[:, None] * scale[None, :]
    return torch.where(denominator > STANDARD_DEVIATION_EPS, covariance / denominator, torch.zeros_like(covariance))


def _relative_matrix(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).norm() / right.norm().clamp_min(STANDARD_DEVIATION_EPS))


def _correlation_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    left_corr = _correlation(_covariance(left))
    right_corr = _correlation(_covariance(right))
    off_diagonal = ~torch.eye(DIMENSION, dtype=torch.bool)
    return float((left_corr - right_corr)[off_diagonal].abs().mean())


def _port_summary(candidate: torch.Tensor, eval_a: torch.Tensor, fit: torch.Tensor) -> dict[str, Any]:
    levels = torch.tensor([0.005, 0.05, 0.50, 0.95, 0.995], dtype=torch.float64)
    candidate_quantiles = torch.quantile(candidate, levels, dim=0)
    eval_quantiles = torch.quantile(eval_a, levels, dim=0)
    fit_quantiles = torch.quantile(fit, torch.tensor([0.005, 0.995], dtype=torch.float64), dim=0)
    candidate_mean = candidate.mean(dim=0)
    eval_mean = eval_a.mean(dim=0)
    candidate_std = candidate.std(dim=0, unbiased=False)
    eval_std = eval_a.std(dim=0, unbiased=False)
    mean_difference = candidate_mean - eval_mean
    std_difference = candidate_std - eval_std
    return {
        "candidate_mean": _float_list(candidate_mean),
        "eval_a_mean": _float_list(eval_mean),
        "mean_difference_candidate_minus_eval_a": _float_list(mean_difference),
        "candidate_std_population": _float_list(candidate_std),
        "eval_a_std_population": _float_list(eval_std),
        "std_difference_candidate_minus_eval_a": _float_list(std_difference),
        "candidate_quantiles_0.005_0.05_0.50_0.95_0.995": [_float_list(row) for row in candidate_quantiles],
        "eval_a_quantiles_0.005_0.05_0.50_0.95_0.995": [_float_list(row) for row in eval_quantiles],
        "quantile_difference_candidate_minus_eval_a": [_float_list(row) for row in (candidate_quantiles - eval_quantiles)],
        "fit_quantiles_0.005_0.995": [_float_list(row) for row in fit_quantiles],
        "outside_fit_q005_fraction_by_port": _float_list((candidate < fit_quantiles[0].unsqueeze(0)).to(torch.float64).mean(dim=0)),
        "outside_fit_q995_fraction_by_port": _float_list((candidate > fit_quantiles[1].unsqueeze(0)).to(torch.float64).mean(dim=0)),
        "mean_abs_mean_difference": float(mean_difference.abs().mean()),
        "mean_abs_std_difference": float(std_difference.abs().mean()),
        "mean_outside_fit_q005_fraction": float((candidate < fit_quantiles[0].unsqueeze(0)).to(torch.float64).mean()),
        "mean_outside_fit_q995_fraction": float((candidate > fit_quantiles[1].unsqueeze(0)).to(torch.float64).mean()),
    }


def _support_metrics(candidate: torch.Tensor, eval_a_h: torch.Tensor, eval_b_h: torch.Tensor, fit: torch.Tensor, eval_a: torch.Tensor) -> dict[str, Any]:
    candidate_h = _standardize(candidate, FIT_MEAN_H, FIT_STD_H)
    ed = canary._energy_distance(candidate_h, eval_a_h)
    covariance = _covariance(candidate_h)
    eval_covariance = _covariance(eval_a_h)
    correlation = _correlation(covariance)
    eval_correlation = _correlation(eval_covariance)
    reference_ed = canary._energy_distance(eval_a_h, eval_b_h)
    norm_ed = canary._energy_distance(candidate_h.norm(dim=1, keepdim=True), eval_a_h.norm(dim=1, keepdim=True))
    per_port = _port_summary(candidate, eval_a, fit)
    metrics = {
        "energy_distance_candidate_vs_eval_a": ed,
        "energy_distance_eval_a_vs_eval_b": reference_ed,
        "R_u": ed / max(reference_ed, STANDARD_DEVIATION_EPS),
        "covariance_relative_frobenius": _relative_matrix(covariance, eval_covariance),
        "correlation_relative_frobenius": _relative_matrix(correlation, eval_correlation),
        "norm_distribution_energy_distance": norm_ed,
        "mean_abs_off_diagonal_correlation_difference": _correlation_difference(candidate_h, eval_a_h),
    }
    metrics["per_port"] = per_port
    metrics["mean_abs_per_port_mean_difference"] = per_port["mean_abs_mean_difference"]
    metrics["mean_abs_per_port_std_difference"] = per_port["mean_abs_std_difference"]
    return metrics


def _canary_identity(authorities: Mapping[str, Any]) -> dict[str, Any]:
    final_u = authorities["u_final"]
    latent = authorities["latent"]
    canary_report = authorities["canary_report"]
    h_all = torch.log(final_u + LOG_EPS)
    mean_h = h_all.mean(dim=0)
    std_h = h_all.std(dim=0, unbiased=False).clamp_min(STANDARD_DEVIATION_EPS)
    real_a_h = (torch.log(latent["real_u_a"] + LOG_EPS) - mean_h) / std_h
    real_b_h = (torch.log(latent["real_u_b"] + LOG_EPS) - mean_h) / std_h
    sample_h = (torch.log(latent["u_sample"] + LOG_EPS) - mean_h) / std_h
    ed_real = canary._energy_distance(real_a_h, real_b_h)
    ed_sample = canary._energy_distance(sample_h, real_a_h)
    observed = ed_sample / max(ed_real, STANDARD_DEVIATION_EPS)
    expected = float(canary_report["stage_a_prior_support"]["joint_support"]["R_u"])
    difference = abs(observed - expected)
    _require(math.isclose(observed, expected, rel_tol=1.0e-10, abs_tol=1.0e-10), "current canary R_u identity cannot be reproduced")
    return {
        "reproduced": True,
        "expected_R_u": expected,
        "observed_R_u": observed,
        "absolute_difference": difference,
        "energy_distance_real_a_real_b": ed_real,
        "energy_distance_sample_real_a": ed_sample,
        "reference_selection": "frozen sampled-generation canary real_u_a/real_u_b bank",
        "standardization": "all 55,000 frozen TRAIN u_final rows, matching canary",
        "metric_implementation": "scripts.research.lightgen.execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1._energy_distance",
    }


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(metrics[name]) for name in CORE_SCALAR_METRICS}


def _strictly_dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_values = _scalar_metrics(left)
    right_values = _scalar_metrics(right)
    return all(left_values[name] <= right_values[name] for name in CORE_SCALAR_METRICS) and any(
        left_values[name] < right_values[name] for name in CORE_SCALAR_METRICS
    )


def _decompose(p0: Mapping[str, Any], p1: Mapping[str, Any], p2: Mapping[str, Any], p3: Mapping[str, Any]) -> dict[str, Any]:
    marginal_dominates = _strictly_dominates(p1, p0)
    dependence_dominates = _strictly_dominates(p2, p1)
    control_dominates_copula = _strictly_dominates(p3, p2)
    if marginal_dominates and dependence_dominates:
        diagnosis = "BOTH_MARGINAL_AND_DEPENDENCE_MISMATCH"
        candidate = "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA"
    elif marginal_dominates:
        diagnosis = "MARGINAL_PRIOR_MISMATCH_DOMINANT"
        candidate = "P1_EMPIRICAL_MARGINAL_INDEPENDENT"
    elif dependence_dominates:
        diagnosis = "JOINT_DEPENDENCE_MISMATCH_DOMINANT"
        candidate = "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA"
    elif control_dominates_copula:
        diagnosis = "GAUSSIAN_COPULA_INSUFFICIENT"
        candidate = "NONE"
    else:
        diagnosis = "UNRESOLVED"
        candidate = "NONE"
    return {
        "dominance_rule": "strict Pareto improvement across all listed descriptive scalar metrics; no absolute pass/fail threshold",
        "marginal_repair_fraction": (float(p0["R_u"]) - float(p1["R_u"])) / max(float(p0["R_u"]) - 1.0, STANDARD_DEVIATION_EPS),
        "dependence_repair_fraction": (float(p1["R_u"]) - float(p2["R_u"])) / max(float(p1["R_u"]) - 1.0, STANDARD_DEVIATION_EPS),
        "R_TN_common_split": float(p0["R_u"]),
        "R_independent": float(p1["R_u"]),
        "R_copula": float(p2["R_u"]),
        "R_real_row_control": float(p3["R_u"]),
        "marginal_candidate_strictly_dominates_current": marginal_dominates,
        "copula_candidate_strictly_dominates_independent": dependence_dominates,
        "real_row_control_strictly_dominates_copula": control_dominates_copula,
        "primary_diagnosis": diagnosis,
        "minimal_prior_repair_candidate": candidate,
    }


def _bank_payload(schema: str, candidate: str, sample: torch.Tensor, metadata: Mapping[str, Any]) -> dict[str, Any]:
    _finite(sample, f"{candidate} sample")
    _require(bool(sample.ge(0.0).all().item()), f"{candidate} sample is negative")
    return {
        "schema": schema,
        "candidate_type": candidate,
        "u": sample.to(dtype=torch.float64, device="cpu").contiguous(),
        "sample_shape": list(sample.shape),
        "dtype": str(sample.dtype),
        "finite": True,
        "nonnegative": True,
        **dict(metadata),
    }


def _artifact_paths(root: Path, target: Path, candidate_path: Path | None) -> dict[str, dict[str, Any]]:
    names = {
        "P1_empirical_independent_u.pt": "P1 empirical marginal independent candidate bank",
        "P2_empirical_gaussian_copula_u.pt": "P2 empirical Gaussian-copula candidate bank",
        "P3_real_row_support_control_u.pt": "P3 real-row joint-support control bank",
        "split_manifest.json": "deterministic FIT/EVAL row-index manifest",
    }
    if candidate_path is not None:
        names[candidate_path.name] = "optional latent prior repair candidate"
    return {
        name: _artifact_ref(root, target / name, role=role)
        for name, role in names.items()
    }


def run(output_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    git_before = _git_state(root)
    authorities = _load_authorities(root)
    identity_check = _canary_identity(authorities)
    split, fit, eval_a, eval_b = _make_split(authorities["u_final"], authorities["source_indices"])
    mean_h, std_h, _h_fit = _standardization(fit)
    global FIT_MEAN_H, FIT_STD_H
    FIT_MEAN_H = mean_h
    FIT_STD_H = std_h

    config_identity = {
        "task_id": TASK_ID,
        "sizes": {"N_TOTAL": N_TOTAL, "FIT": FIT_COUNT, "EVAL_A": EVAL_COUNT, "EVAL_B": EVAL_COUNT, "N_CANDIDATE": COMPARISON_COUNT},
        "seeds": {"split": SPLIT_SEED, "P1": P1_SEED, "P2": P2_SEED, "P3": P3_SEED},
        "coordinate": "h(u)=log(u+1e-12), FIT-derived mean/std",
        "no_optical_forward": True,
    }
    source_manifest = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            root / CANARY_RUN.parent.parent / "runs" / "attempt_01" / "sampled_generation_canary_report.json",
            root / Path("scripts/research/lightgen/execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1.py"),
        ),
        config_identity=config_identity,
    ).to_dict()
    target = _ensure_output(root, output_dir)
    heartbeat = HeartbeatWriter(target, RUN_ID, total_steps=6, interval_seconds=60.0)
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=0, latest_losses={"task": "INPUTS_VALIDATED"}, force=True)
    atomic_write_new_json(target / "source_manifest.json", source_manifest, artifact_kind="latent support repair source manifest")
    atomic_write_new_json(
        target / "execution_manifest.json",
        {
            "schema": "chen_bogt_latent_prior_support_repair_execution_manifest.v1",
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "run_id": RUN_ID,
            "trajectory_id": TRAJECTORY_ID,
            "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
            "source_manifest_id": source_manifest["source_manifest_id"],
            "git_before": git_before,
            "input_hashes": authorities["input_hashes"],
            "config_identity": config_identity,
            "forbidden_paths": {"images": False, "labels": False, "encoder": False, "generator": False, "cuda": False},
        },
        artifact_kind="latent support repair execution manifest",
    )
    heartbeat.update(state=RUNNING, phase="STATISTICS", step=1, latest_losses={"task": "SPLIT_AND_STANDARDIZATION_READY"}, force=True)
    atomic_write_new_json(target / "split_manifest.json", split, artifact_kind="latent support repair split manifest")

    sorted_values, fit_state = _fit_marginals_and_copula(fit)
    p1, p1_sampling = _sample_empirical_independent(sorted_values)
    p2, p2_sampling = _sample_empirical_gaussian_copula(fit_state)
    p3, p3_positions = _sample_real_row_control(fit)
    heartbeat.update(state=RUNNING, phase="POST_TRAIN_FIT", step=2, latest_losses={"task": "P1_P2_P3_SAMPLED"}, force=True)

    eval_a_h = _standardize(eval_a, mean_h, std_h)
    eval_b_h = _standardize(eval_b, mean_h, std_h)
    p0 = _support_metrics(authorities["u_sample"], eval_a_h, eval_b_h, fit, eval_a)
    p1_metrics = _support_metrics(p1, eval_a_h, eval_b_h, fit, eval_a)
    p2_metrics = _support_metrics(p2, eval_a_h, eval_b_h, fit, eval_a)
    p3_metrics = _support_metrics(p3, eval_a_h, eval_b_h, fit, eval_a)
    heartbeat.update(state=RUNNING, phase="STATISTICS", step=3, latest_losses={"task": "COMMON_METRICS_COMPLETE"}, force=True)

    p1_payload = _bank_payload(
        P1_SCHEMA,
        "P1_EMPIRICAL_MARGINAL_INDEPENDENT",
        p1,
        {
            "fit_count": FIT_COUNT,
            "fit_source_indices_sha256": tensor_sha256(torch.tensor(split["fit_source_indices"], dtype=torch.int64)),
            "seed": P1_SEED,
            "marginal_representation": "FIT-only piecewise-linear inverse empirical quantile from stable midrank plotting positions",
            "normalization_identity": "candidate metrics use the single FIT-derived h=log(u+1e-12) coordinate",
            "sampling_api": p1_sampling,
        },
    )
    dependence = fit_state["dependence"]
    p2_payload = _bank_payload(
        P2_SCHEMA,
        "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA",
        p2,
        {
            "fit_count": FIT_COUNT,
            "fit_source_indices_sha256": tensor_sha256(torch.tensor(split["fit_source_indices"], dtype=torch.int64)),
            "seed": P2_SEED,
            "marginal_representation": "same FIT-only piecewise-linear inverse empirical quantile as P1",
            "dependence_model": "R_shrunk fitted from FIT Gaussianized midrank scores",
            "R_shrunk": dependence["R_shrunk"],
            "cholesky": dependence["cholesky"],
            "sampling_api": {key: value for key, value in p2_sampling.items() if isinstance(value, (str, int, float, bool))},
        },
    )
    p3_payload = _bank_payload(
        P3_SCHEMA,
        "P3_REAL_ROW_JOINT_SUPPORT_CONTROL",
        p3,
        {
            "fit_count": FIT_COUNT,
            "seed": P3_SEED,
            "selected_fit_positions": p3_positions,
            "selected_fit_positions_sha256": tensor_sha256(p3_positions),
            "sampling_api": "deterministic without-replacement whole-row selection from FIT",
            "generative_candidate": False,
        },
    )
    atomic_torch_save_new(target / "P1_empirical_independent_u.pt", p1_payload, artifact_kind="P1 empirical independent u bank")
    atomic_torch_save_new(target / "P2_empirical_gaussian_copula_u.pt", p2_payload, artifact_kind="P2 empirical Gaussian-copula u bank")
    atomic_torch_save_new(target / "P3_real_row_support_control_u.pt", p3_payload, artifact_kind="P3 real-row support control u bank")

    decomposition = _decompose(p0, p1_metrics, p2_metrics, p3_metrics)
    candidate_path: Path | None = None
    if decomposition["minimal_prior_repair_candidate"] != "NONE":
        candidate_path = target / "latent_prior_repair_candidate.pt"
        candidate_payload: dict[str, Any] = {
            "schema": "chen_bogt_latent_prior_repair_candidate.v1",
            "candidate_type": decomposition["minimal_prior_repair_candidate"],
            "claim": "LATENT_PRIOR_REPAIR_CANDIDATE",
            "fit_split_identity": {
                "split_manifest": "split_manifest.json",
                "fit_count": FIT_COUNT,
                "fit_source_indices_sha256": tensor_sha256(torch.tensor(split["fit_source_indices"], dtype=torch.int64)),
            },
            "marginal_representation": {
                "sorted_fit_values": sorted_values,
                "primitive": "local empirical-quantile interpolation",
                "plotting_positions": fit_state["positions"],
                "tie_summary": fit_state["tie_summary"],
            },
            "seed_semantics": {"P1": P1_SEED, "P2": P2_SEED, "P3": P3_SEED},
            "normalization_identity": {
                "formula": "h=log(u+1e-12); h_std=(h-mu_fit)/(sigma_fit+1e-12)",
                "mu_fit": mean_h,
                "sigma_fit_population": std_h,
            },
            "sampling_api_metadata": {
                "P1": p1_sampling,
                "P2": {key: value for key, value in p2_sampling.items() if isinstance(value, (str, int, float, bool))},
            },
            "training_boundary": "post-train sampling producer only; BOGT historical objective remains TN_+(1,0.25)",
        }
        if decomposition["minimal_prior_repair_candidate"] == "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA":
            candidate_payload["dependence_parameters"] = {
                "R_emp": dependence["R_emp"],
                "R_shrunk": dependence["R_shrunk"],
                "cholesky": dependence["cholesky"],
                "tau": dependence["tau"],
                "beta": dependence["beta"],
                "delta": dependence["delta"],
                "alpha": dependence["alpha"],
                "eigenvalues": dependence["eigenvalues"],
                "diagonal_error": dependence["diagonal_error"],
                "shrinkage_convention": "analytic fit-only alpha=clip(beta/delta,0,1); R_shrunk=(1-alpha)R_emp+alpha*tau*I; exact symmetrization; no unit-diagonal renormalization",
            }
        atomic_torch_save_new(candidate_path, candidate_payload, artifact_kind="latent prior repair candidate")

    heartbeat.update(state=RUNNING, phase="FINALIZING", step=5, latest_losses={"task": "REPORT_ASSEMBLY"}, force=True)
    artifacts = _artifact_paths(root, target, candidate_path)
    run_identity = {
        "schema": "research_run_identity.v1",
        "task_id": TASK_ID,
        "run_id": RUN_ID,
        "trajectory_id": TRAJECTORY_ID,
        "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
        "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
        "data_role": "frozen_train_u_only",
        "source_manifest_id": source_manifest["source_manifest_id"],
        "config_identity": config_identity,
    }
    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "result": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "project_mainline": SEMANTIC_AUTHORITY_ID,
        "current_authorized_next_action": "NONE",
        "git": {
            "branch": git_before["branch"],
            "head": git_before["head"],
            "dirty_before": git_before["dirty"],
            "dirty_after": _git_state(root)["dirty"],
            "status_sha256_before": git_before["status_sha256"],
        },
        "source_manifest_id": source_manifest["source_manifest_id"],
        "source_manifest": source_manifest,
        "run_identity": run_identity,
        "input_authorities": {
            "generation_ready_bundle": _artifact_ref(root, root / BUNDLE_PATH, role="frozen generation-ready bundle identity only"),
            "training_report": _artifact_ref(root, root / TRAINING_REPORT_PATH, role="frozen upstream report identity only"),
            "final_train_u": _artifact_ref(root, root / FINAL_COLLECTION_PATH, role="frozen final TRAIN u collection"),
            "sampled_generation_canary_report": _artifact_ref(root, root / CANARY_REPORT_PATH, role="completed canary diagnostic authority"),
            "sampled_latent_bank": _artifact_ref(root, root / LATENT_BANK_PATH, role="frozen current-canary u_sample authority"),
        },
        "split": split,
        "metric_identity": {
            "canary_identity_check": identity_check,
            "common_coordinate": "h(u)=log(u+1e-12); mean/std fit only on 45,000 FIT rows; no candidate-specific normalization",
            "energy_distance_implementation": "scripts.research.lightgen.execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1._energy_distance",
            "energy_distance_semantics": "biased V-statistic with diagonal terms retained in both within-sample means",
            "P0_common_split_is_distinct_from_canary_identity": True,
        },
        "current_tn": {
            "candidate_type": "P0_CURRENT_TN_PLUS_FACTORIAL_INDEPENDENCE",
            "source": "frozen current-canary u_sample; no reseeding or resampling",
            "common_split_metrics": p0,
        },
        "empirical_marginal_independent": {
            "candidate_type": "P1_EMPIRICAL_MARGINAL_INDEPENDENT",
            "metrics": p1_metrics,
            "sampling": p1_sampling,
        },
        "empirical_gaussian_copula": {
            "candidate_type": "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA",
            "metrics": p2_metrics,
            "sampling": {key: value for key, value in p2_sampling.items() if isinstance(value, (str, int, float, bool))},
            "fit": {
                "fit_rows": FIT_COUNT,
                "tie_summary": fit_state["tie_summary"],
                "shrinkage_convention": "analytic fit-only alpha=clip(beta/delta,0,1); R_shrunk=(1-alpha)R_emp+alpha*tau*I; exact symmetrization; no unit-diagonal renormalization",
                "alpha": float(dependence["alpha"]),
                "tau": float(dependence["tau"]),
                "beta": float(dependence["beta"]),
                "delta": float(dependence["delta"]),
                "finite": True,
                "positive_definite": True,
                "cholesky_status": "success",
                "diagonal_error": float(dependence["diagonal_error"]),
                "eigenvalue_min": float(dependence["eigenvalues"].min()),
                "eigenvalue_max": float(dependence["eigenvalues"].max()),
                "R_emp_sha256": tensor_sha256(dependence["R_emp"]),
                "R_shrunk_sha256": tensor_sha256(dependence["R_shrunk"]),
                "cholesky_sha256": tensor_sha256(dependence["cholesky"]),
            },
        },
        "real_row_control": {
            "candidate_type": "P3_REAL_ROW_JOINT_SUPPORT_CONTROL",
            "metrics": p3_metrics,
            "seed": P3_SEED,
            "generative_candidate": False,
        },
        "mismatch_decomposition": decomposition,
        "primary_diagnosis": decomposition["primary_diagnosis"],
        "minimal_prior_repair_candidate": {
            "candidate": decomposition["minimal_prior_repair_candidate"],
            "artifact": None if candidate_path is None else candidate_path.name,
            "claim_boundary": "LATENT_PRIOR_REPAIR_CANDIDATE only; no generation success, BOGT retraining authorization, or promotion",
        },
        "scientific_reads": {"frozen_train_u_only": True, "DEV": 0, "protected": 0, "test": 0, "sealed": 0, "images": 0, "labels": 0, "encoder_forward": 0, "generator_forward": 0, "cuda": 0},
        "historical_boundary": "P1/P2 are post-train sampling repair candidates; historical BOGT objective remains TN_+(1,0.25); phase model remains unchanged",
        "not_established": [
            "generation success",
            "phase compatibility",
            "BOGT retraining against P1/P2",
            "formal acceptance",
            "promotion",
            "causal proof of downstream semantic mismatch",
        ],
        "artifacts": artifacts,
        "no_automatic_follow_up": True,
    }
    atomic_write_new_json(target / "latent_prior_support_decomposition_report.json", report, artifact_kind="latent prior support decomposition report")
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=6, latest_losses={"task": "COMPLETED"}, force=True)
    return {
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "primary_diagnosis": decomposition["primary_diagnosis"],
        "candidate": decomposition["minimal_prior_repair_candidate"],
        "report": _relative_path(root, target / "latent_prior_support_decomposition_report.json"),
        "report_sha256": file_sha256(target / "latent_prior_support_decomposition_report.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))
    except (RepairError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
