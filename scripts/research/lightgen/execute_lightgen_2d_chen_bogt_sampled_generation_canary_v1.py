"""Execute the bounded Chen BOGT sampled-generation canary.

This script is intentionally a consumer of the frozen step-300 generation
bundle.  It does not train, refit, read MNIST images/labels, or resolve an
automatic latest artifact.  The only expensive operation is the declared
first-200 sampled and real-control direct-OLS generator evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image

from ldm_project.v2.application.research.lightgen.lightgen_2d_chen_bogt_complex_ols_training import EPS_S, LightGen2DChenBOGTComplexOLSTrainingConsumer, OLS_PORT_COUNT, PRODUCTION_PROPAGATION_GRID, TRAIN_COUNT, WrappedConditionalPhaseModel, git_identity, max_q_gauge, statistical_variable, wrap_phase
from ldm_project.v2.application.research.shared.research_run_control import HeartbeatWriter, InterruptController, RecoveryCheckpointManager, ResearchRunControl, ResearchRunIdentity, RUN_COMPLETED, RUNNING, RUN_PREPARING, capture_execution_source_manifest, recovery_due
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from ldm_project.v2.student.evaluator import load_frozen_evaluator


TASK_ID = "EXECUTE_CHEN_BOGT_REPAIRED_PRIOR_GENERATION_CANARY_V1"
OUTER_GATE = "CHEN_BOGT_REPAIRED_PRIOR_GENERATION_CANARY_V1_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "chen-bogt-repaired-prior-generation-canary-v1"
TRAJECTORY_ID = "chen-bogt-repaired-prior-generation-canary-trajectory-v1"
AUTHORIZATION_ID = f"{TASK_ID}:authorization:01"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

SOURCE_RUN = Path("outputs/20260917_chen_bogt_complex_ols_v1/runs/attempt_01")
BUNDLE_RELATIVE = SOURCE_RUN / "generation_ready_bundle.pt"
REPORT_RELATIVE = SOURCE_RUN / "report.json"
FINAL_COLLECTION_RELATIVE = SOURCE_RUN / "final_train_ols.pt"
EXPECTED_BUNDLE_SHA256 = "4c9a8d10ab1694d69aeaf9e58c5637c2695088a5804b15f4b976e15cc489439f"
EXPECTED_REPORT_SHA256 = "e2e87a9aece563616f9e64b329411a4bcd9a0a3f6dcbcab7f220e690bee15dfc"
EXPECTED_FINAL_COLLECTION_SHA256 = "2d1264219faa0964467d40f2e4882ec3767d3a1aafd30e143bf406aa39be6dcc"

HISTORICAL_CANARY_RUN = Path("outputs/20260918_chen_bogt_complex_ols_sampled_generation_canary_v1/runs/attempt_01")
HISTORICAL_CANARY_REPORT_RELATIVE = HISTORICAL_CANARY_RUN / "sampled_generation_canary_report.json"
HISTORICAL_LATENT_RELATIVE = HISTORICAL_CANARY_RUN / "sampled_latent_bank.pt"
EXPECTED_HISTORICAL_CANARY_REPORT_SHA256 = "60e9a8852dd1f03839a5d88940c15242f6324591ae79f521bae5e41da3c0c7fa"
EXPECTED_HISTORICAL_LATENT_SHA256 = "09f8ccd01738862c79f9a3ed8792b155fffb5dad907b36c1e9a1bfab691cce6d"
HISTORICAL_TASK_ID = "EXECUTE_CHEN_BOGT_ON_COMPLEX_OLS_SAMPLED_GENERATION_CANARY_V1"
HISTORICAL_OUTER_GATE = "CHEN_BOGT_SAMPLED_GENERATION_CANARY_V1_RESOLVED"

P2_CANDIDATE_RELATIVE = Path(
    "outputs/20260918_chen_bogt_latent_support_compatibility_v1/runs/attempt_03/latent_prior_repair_candidate.v1.pt"
)
P2_MANIFEST_RELATIVE = Path(
    "outputs/20260918_chen_bogt_latent_support_compatibility_v1/runs/attempt_03/latent_prior_repair_candidate_manifest.v1.json"
)
EXPECTED_P2_CANDIDATE_SHA256 = "c26d60b051c7604bf91615cc74781b5ece69faf05835151184cd39dba07b6bfa"
EXPECTED_P2_MANIFEST_SHA256 = "18b5e8ee6a5bf9edd62def8ddef3fa5cf9c70fd32063d8715cd657f0d536c9db"
P2_CANDIDATE_SCHEMA = "chen_bogt_latent_prior_repair_candidate.v1"
P2_CANDIDATE_TYPE = "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA"
P2_CANDIDATE_CLAIM = "POST_TRAIN_SAMPLING_REPAIR_CANDIDATE"

EVALUATOR_IMPLEMENTATION_RELATIVE = Path("src/ldm_project/v2/student/evaluator.py")
EXPECTED_EVALUATOR_IMPLEMENTATION_SHA256 = "f6cc3a229a9ef85400fbd9ee34b7a6d559eb04e95502c9b9a8b3953d3fcfa4d3"
EXPECTED_EVALUATOR_CHECKPOINT_SHA256 = "46741ec76e45653c9dcb76e7b4f1645c9c057a47cdd9e4c1778d9debb553ec57"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260918_chen_bogt_repaired_prior_generation_canary_v1/runs/attempt_01"
)

N_LATENT = 1_000
N_OPTICAL = 200
N_REAL_REFERENCE = 2_000
OPTICAL_BATCH_SIZE = 10
CONTACT_SHEET_COUNT = 100
CONTACT_TILE_SIZE = 100

PRIOR_SEED = 20_260_918
CONDITIONAL_PHASE_SEED = 20_260_919
REAL_REFERENCE_SELECTION_SEED = 20_260_920
REAL_U_PHASE_SEED = 20_260_921

LOG_EPS = 1.0e-12
STANDARD_DEVIATION_FLOOR = 1.0e-12
NEAR_DUPLICATE_RELATIVE_TOLERANCE = 1.0e-3
EFFECTIVE_RANK_RELATIVE_TOLERANCE = 1.0e-3
PAIRWISE_CHUNK_SIZE = 128

BUNDLE_SCHEMA = "chen_bogt_generation_bundle.v1"
FINAL_COLLECTION_SCHEMA = "chen_bogt_final_train_ols_collection.v1"
REPORT_SCHEMA = "chen_bogt_repaired_prior_generation_canary_report.v1"
LATENT_SCHEMA = "chen_bogt_repaired_prior_generation_latent_bank.v1"
CMOS_SCHEMA = "chen_bogt_canary_cmos.v1"


class CanaryError(RuntimeError):
    """Raised when the canary boundary or its frozen inputs are invalid."""


class CanaryInterrupted(RuntimeError):
    """Raised after a requested interrupt has been persisted at a safe boundary."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a tensor")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is nonfinite")


def _artifact_ref(root: Path, path: Path, *, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        relative = str(resolved)
    return {"path": relative, "sha256": file_sha256(resolved), "role": role}


def _json_float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    return float(value)


def _float_list(value: torch.Tensor) -> list[float]:
    return [float(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _int_list(value: torch.Tensor) -> list[int]:
    return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]


def _summary(value: torch.Tensor, *, quantiles: tuple[float, ...] = (0.01, 0.5, 0.99)) -> dict[str, Any]:
    tensor = value.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    _finite(tensor, "summary input")
    quantile_tensor = torch.tensor(quantiles, dtype=torch.float64)
    observed = torch.quantile(tensor, quantile_tensor)
    return {
        "count": int(tensor.numel()),
        "min": _json_float(tensor.min()),
        "max": _json_float(tensor.max()),
        "mean": _json_float(tensor.mean()),
        "std_population": _json_float(tensor.std(unbiased=False)),
        "quantiles": {
            str(quantile): _json_float(observed[index])
            for index, quantile in enumerate(quantiles)
        },
    }


def _pairwise_mean_distance(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    exclude_diagonal: bool = False,
) -> float:
    """Compute a chunked Euclidean mean without materializing a full matrix."""
    _require(left.ndim == 2 and right.ndim == 2 and left.shape[1] == right.shape[1], "distance inputs have incompatible shapes")
    _require(left.shape[0] > 0 and right.shape[0] > 0, "distance inputs must be nonempty")
    left = left.to(dtype=torch.float64, device="cpu")
    right = right.to(dtype=torch.float64, device="cpu")
    total = 0.0
    count = 0
    same_shape = exclude_diagonal and left.shape == right.shape and bool(torch.equal(left, right))
    for start in range(0, left.shape[0], PAIRWISE_CHUNK_SIZE):
        stop = min(start + PAIRWISE_CHUNK_SIZE, left.shape[0])
        distances = torch.cdist(left[start:stop], right, p=2)
        if same_shape:
            row_index = torch.arange(start, stop)
            distances[torch.arange(stop - start), row_index] = 0.0
            count += distances.numel() - (stop - start)
        else:
            count += distances.numel()
        total += float(distances.sum().item())
    _require(count > 0, "distance denominator is empty")
    return total / count


def _energy_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    """Biased V-statistic Euclidean energy distance used by this canary.

    Keep the diagonal terms in both within-sample means.  Mixing an
    off-diagonal U-statistic denominator with the cross-sample V-statistic
    can make a valid finite-sample reference distance negative before the
    nonnegative clamp, which would turn descriptive ratios into artifacts of
    the estimator rather than properties of the sampled distributions.
    """
    cross = _pairwise_mean_distance(left, right)
    within_left = _pairwise_mean_distance(left, left)
    within_right = _pairwise_mean_distance(right, right)
    return max(0.0, 2.0 * cross - within_left - within_right)


def _covariance(value: torch.Tensor) -> torch.Tensor:
    matrix = value.to(device="cpu", dtype=torch.float64)
    _require(matrix.ndim == 2 and matrix.shape[0] > 1, "covariance input is invalid")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(matrix.shape[0])


def _correlation(covariance: torch.Tensor) -> torch.Tensor:
    scale = covariance.diag().clamp_min(0.0).sqrt()
    denominator = scale[:, None] * scale[None, :]
    return torch.where(denominator > STANDARD_DEVIATION_FLOOR, covariance / denominator, torch.zeros_like(covariance))


def _relative_frobenius(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = (left - right).norm()
    denominator = right.norm().clamp_min(STANDARD_DEVIATION_FLOOR)
    return _json_float(numerator / denominator)


def _effective_rank(value: torch.Tensor) -> dict[str, Any]:
    matrix = value.to(device="cpu", dtype=torch.float64)
    _require(matrix.ndim == 2 and matrix.shape[0] > 1, "effective-rank input is invalid")
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    if centered.shape[1] <= 2_048:
        singular = torch.linalg.svdvals(centered)
    else:
        gram = centered @ centered.T
        eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0.0)
        singular = eigenvalues.sqrt().flip(0)
    maximum = singular.max().clamp_min(STANDARD_DEVIATION_FLOOR)
    effective = int((singular > maximum * EFFECTIVE_RANK_RELATIVE_TOLERANCE).sum().item())
    return {
        "nominal_rank": int(min(matrix.shape)),
        "effective_rank": effective,
        "relative_tolerance": EFFECTIVE_RANK_RELATIVE_TOLERANCE,
        "singular_values_top10": _float_list(singular[:10]),
    }


def _diversity_view(value: torch.Tensor, *, name: str) -> dict[str, Any]:
    matrix = value.detach().to(device="cpu")
    if matrix.is_complex():
        matrix = torch.cat((matrix.real, matrix.imag), dim=1)
    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    matrix = matrix.to(dtype=torch.float32)
    _finite(matrix, f"{name} diversity input")
    distances = torch.pdist(matrix)
    median = distances.median() if distances.numel() else torch.zeros(())
    threshold = max(
        1.0e-12,
        _json_float(median) * NEAR_DUPLICATE_RELATIVE_TOLERANCE,
    )
    exact_unique = int(torch.unique(matrix, dim=0).shape[0])
    return {
        "name": name,
        "count": int(matrix.shape[0]),
        "feature_dimension": int(matrix.shape[1]),
        "exact_duplicate_rows": int(matrix.shape[0] - exact_unique),
        "near_duplicate_pair_count": int((distances <= threshold).sum().item()),
        "near_duplicate_threshold": threshold,
        "near_duplicate_threshold_definition": "max(1e-12, 1e-3 * median_pairwise_euclidean_distance)",
        "pairwise_distance": {
            "min": _json_float(distances.min()) if distances.numel() else 0.0,
            "median": _json_float(distances.median()) if distances.numel() else 0.0,
            "mean": _json_float(distances.mean()) if distances.numel() else 0.0,
            "max": _json_float(distances.max()) if distances.numel() else 0.0,
        },
        "effective_rank": _effective_rank(matrix),
    }


def _standardize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (value.to(torch.float64) - mean) / std.clamp_min(STANDARD_DEVIATION_FLOOR)


def _covariance_comparison(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_cov = _covariance(left)
    right_cov = _covariance(right)
    left_corr = _correlation(left_cov)
    right_corr = _correlation(right_cov)
    return {
        "covariance_relative_frobenius": _relative_frobenius(left_cov, right_cov),
        "correlation_relative_frobenius": _relative_frobenius(left_corr, right_corr),
        "left_covariance_trace": _json_float(left_cov.trace()),
        "right_covariance_trace": _json_float(right_cov.trace()),
    }


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        value = _mapping(value, ".".join(keys)).get(key)
    return value


def _load_historical_p0_authority(root: Path) -> dict[str, Any]:
    report_path = root / HISTORICAL_CANARY_REPORT_RELATIVE
    latent_path = root / HISTORICAL_LATENT_RELATIVE
    _require(report_path.is_file() and latent_path.is_file(), "historical P0 canary authority is missing")
    report_sha = file_sha256(report_path)
    latent_sha = file_sha256(latent_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema") == "chen_bogt_sampled_generation_canary_report.v1", "historical P0 report schema changed")
    _require(report.get("task_id") == HISTORICAL_TASK_ID, "historical P0 task identity changed")
    _require(report.get("outer_gate") == HISTORICAL_OUTER_GATE, "historical P0 outer gate changed")
    _require(report.get("status") == "VALID_COMPLETED", "historical P0 canary is not valid completed")
    _require(report.get("selected_step", 300) == 300, "historical P0 selected step changed")

    latent = safe_torch_load(latent_path, map_location="cpu")
    _require(isinstance(latent, Mapping), "historical P0 latent authority is not a mapping")
    _require(latent.get("schema") == "chen_bogt_sampled_latent_bank.v1", "historical P0 latent schema changed")
    _require(latent.get("selected_step") == 300, "historical P0 latent selected step changed")
    _require(_mapping(latent.get("seeds"), "historical P0 seeds").get("prior") == PRIOR_SEED, "historical P0 seed identity changed")
    u_sample = _tensor(latent.get("u_sample"), "historical P0 u sample")
    z_sample = _tensor(latent.get("z_sample"), "historical P0 z sample")
    source_indices = _tensor(latent.get("real_reference_source_indices"), "historical P0 reference indices")
    _require(u_sample.shape == (N_LATENT, OLS_PORT_COUNT) and u_sample.dtype == torch.float64, "historical P0 u sample shape/dtype changed")
    _require(z_sample.shape == (N_LATENT, OLS_PORT_COUNT) and z_sample.dtype == torch.complex64, "historical P0 z sample shape/dtype changed")
    _require(source_indices.shape == (N_REAL_REFERENCE,) and source_indices.dtype == torch.long, "historical P0 reference indices changed")
    _require(torch.unique(source_indices).numel() == N_REAL_REFERENCE, "historical P0 reference indices are not distinct")
    _finite(u_sample, "historical P0 u sample")
    _finite(z_sample, "historical P0 z sample")
    _require(bool((u_sample >= 0.0).all().item()), "historical P0 u sample left nonnegative support")

    artifact_refs = _mapping(report.get("artifacts"), "historical P0 artifacts")
    artifact_paths: dict[str, Path] = {}
    for name in ("generated_cmos_canary.pt", "real_control_cmos.pt"):
        ref = _mapping(artifact_refs.get(name), f"historical P0 artifact {name}")
        path = root / str(ref.get("path"))
        _require(path.is_file(), f"historical P0 artifact is missing: {name}")
        artifact_paths[name] = path

    metrics = {
        "R_u": float(_nested(report, "stage_a_prior_support", "joint_support", "R_u")),
        "R_phi": float(_nested(report, "stage_b_phase_completion", "comparison", "R_phi")),
        "R_z": float(_nested(report, "stage_c_full_complex_latent", "complex_comparison", "R_z")),
        "R_raw": float(_nested(report, "stage_d_e_raw_cmos", "raw_distribution_comparison", "R_raw")),
        "R_feature": float(_nested(report, "stage_f_frozen_evaluator", "comparison", "R_feature")),
    }
    _require(all(math.isfinite(value) for value in metrics.values()), "historical P0 metrics are nonfinite")
    return {
        "report_path": report_path,
        "latent_path": latent_path,
        "report_sha256": report_sha,
        "latent_sha256": latent_sha,
        "report": dict(report),
        "latent": dict(latent),
        "metrics": metrics,
        "artifact_paths": artifact_paths,
    }


def _load_p2_candidate(root: Path) -> dict[str, Any]:
    candidate_path = root / P2_CANDIDATE_RELATIVE
    manifest_path = root / P2_MANIFEST_RELATIVE
    _require(candidate_path.is_file() and manifest_path.is_file(), "frozen P2 candidate/manifest is missing")
    candidate_sha = file_sha256(candidate_path)
    manifest_sha = file_sha256(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema") == "chen_bogt_latent_prior_repair_candidate_manifest.v1", "P2 manifest schema changed")
    _require(manifest.get("candidate_type") == P2_CANDIDATE_TYPE, "P2 manifest candidate type changed")
    _require(manifest.get("claim") == P2_CANDIDATE_CLAIM, "P2 manifest claim changed")
    candidate_artifact = _mapping(manifest.get("candidate_artifact"), "P2 manifest candidate artifact")
    _require(candidate_artifact.get("path") == P2_CANDIDATE_RELATIVE.as_posix(), "P2 manifest candidate path changed")
    _require(manifest.get("sampling_semantics") == "frozen attempt_08 P1/P2 bank; no refit and no resampling in this task", "P2 sampling boundary changed")

    candidate = safe_torch_load(candidate_path, map_location="cpu")
    _require(isinstance(candidate, Mapping), "P2 candidate is not a mapping")
    _require(candidate.get("schema") == P2_CANDIDATE_SCHEMA, "P2 candidate schema changed")
    _require(candidate.get("candidate_type") == P2_CANDIDATE_TYPE, "P2 candidate type changed")
    _require(candidate.get("claim") == P2_CANDIDATE_CLAIM, "P2 candidate claim changed")
    _require(candidate.get("not_generation_success") is True, "P2 candidate generation-success boundary changed")
    _require(candidate.get("not_retraining_authority") is True, "P2 candidate retraining boundary changed")
    _require(candidate.get("not_promotion_authority") is True, "P2 candidate promotion boundary changed")
    _require(candidate.get("marginal_representation") == "same FIT-only piecewise-linear inverse empirical quantile as P1", "P2 marginal semantics changed")
    _require(candidate.get("source_bank") == manifest.get("source_bank"), "P2 source-bank identity disagrees")
    _require(candidate.get("primary_compatibility") == manifest.get("primary_compatibility"), "P2 compatibility identity disagrees")

    fit_identity = _mapping(candidate.get("fit_split_identity"), "P2 fit split identity")
    _require(fit_identity.get("fit_count") == 45_000 and fit_identity.get("split_seed") == 20_260_922, "P2 FIT split identity changed")
    sampling = _mapping(candidate.get("sampling_semantics"), "P2 sampling semantics")
    _require(sampling.get("seed") == 20_260_924, "P2 source sampling seed identity changed")
    _require(sampling.get("rng") == "torch.Generator(device='cpu')", "P2 RNG identity changed")

    u_sample = _tensor(candidate.get("u"), "P2 frozen u bank")
    _require(u_sample.shape == (N_LATENT, OLS_PORT_COUNT), "P2 frozen u shape changed")
    _require(u_sample.dtype == torch.float64, "P2 frozen u dtype changed")
    _finite(u_sample, "P2 frozen u bank")
    _require(bool((u_sample >= 0.0).all().item()), "P2 frozen u bank left nonnegative support")
    return {
        "candidate_path": candidate_path,
        "manifest_path": manifest_path,
        "candidate_sha256": candidate_sha,
        "manifest_sha256": manifest_sha,
        "candidate": dict(candidate),
        "manifest": dict(manifest),
        "u": u_sample.cpu().contiguous(),
    }


def _phase_model_from_bundle(payload: Mapping[str, Any]) -> WrappedConditionalPhaseModel:
    _require(payload.get("schema") == "chen_bogt_wrapped_conditional_phase_model.v1", "phase model schema changed")
    required = (
        "coefficients",
        "residual_mean",
        "covariance",
        "cholesky",
        "scale_sha256",
        "fit_data_role",
        "fit_count",
        "engineering_only",
        "data_identity",
    )
    _require(all(key in payload for key in required), "phase model fields are incomplete")
    return WrappedConditionalPhaseModel(
        _tensor(payload["coefficients"], "phase coefficients"),
        _tensor(payload["residual_mean"], "phase residual mean"),
        _tensor(payload["covariance"], "phase covariance"),
        _tensor(payload["cholesky"], "phase Cholesky"),
        str(payload["scale_sha256"]),
        str(payload["fit_data_role"]),
        int(payload["fit_count"]),
        bool(payload["engineering_only"]),
        dict(_mapping(payload["data_identity"], "phase data identity")),
        float(payload.get("log_epsilon", LOG_EPS)),
        float(payload.get("covariance_epsilon", 1.0e-8)),
    )


def _phase_embedding(phase: torch.Tensor) -> torch.Tensor:
    return torch.cat((phase.cos(), phase.sin()), dim=1).to(torch.float64)


def _phase_diagnostics(phase: torch.Tensor, anchors: torch.Tensor, *, name: str) -> dict[str, Any]:
    phase = phase.to(device="cpu", dtype=torch.float64)
    anchors = anchors.to(device="cpu", dtype=torch.long)
    _require(phase.ndim == 2 and phase.shape[1] == OLS_PORT_COUNT, f"{name} phase shape is invalid")
    _finite(phase, f"{name} phase")
    phasor = torch.complex(phase.cos(), phase.sin())
    circular_mean = torch.angle(phasor.mean(dim=0))
    circular_variance = 1.0 - phasor.mean(dim=0).abs()
    phasor_view = _phase_embedding(phase)
    phasor_cov = _covariance(phasor_view)
    off_diagonal = phasor_cov - torch.diag(torch.diag(phasor_cov))
    relative = wrap_phase(phase - phase.gather(1, anchors[:, None]))
    return {
        "name": name,
        "sample_count": int(phase.shape[0]),
        "circular_mean_rad_by_port": _float_list(circular_mean),
        "circular_variance_by_port": _float_list(circular_variance),
        "phasor_covariance": {
            "trace": _json_float(phasor_cov.trace()),
            "frobenius": _json_float(phasor_cov.norm()),
            "off_diagonal_frobenius": _json_float(off_diagonal.norm()),
            "dependence_ratio": _json_float(off_diagonal.norm() / phasor_cov.norm().clamp_min(STANDARD_DEVIATION_FLOOR)),
            "effective_rank": _effective_rank(phasor_view),
        },
        "relative_phase": {
            "anchor_index_histogram": _int_list(torch.bincount(anchors, minlength=OLS_PORT_COUNT)),
            "mean_abs_rad_per_port": _float_list(relative.abs().mean(dim=0)),
            "mean_l2_per_sample": _json_float(relative.square().mean(dim=1).sqrt().mean()),
            "energy_embedding": {
                "self_energy_distance": _energy_distance(_phase_embedding(phase[: phase.shape[0] // 2]), _phase_embedding(phase[phase.shape[0] // 2 :]))
                if phase.shape[0] >= 4
                else 0.0
            },
        },
    }


def _phase_distribution_comparison(
    completed: torch.Tensor,
    true_a: torch.Tensor,
    true_b: torch.Tensor,
    completed_anchors: torch.Tensor,
    true_a_anchors: torch.Tensor,
    true_b_anchors: torch.Tensor,
) -> dict[str, Any]:
    completed_relative = wrap_phase(completed - completed.gather(1, completed_anchors[:, None]))
    true_a_relative = wrap_phase(true_a - true_a.gather(1, true_a_anchors[:, None]))
    true_b_relative = wrap_phase(true_b - true_b.gather(1, true_b_anchors[:, None]))
    completed_view = _phase_embedding(completed)
    true_a_view = _phase_embedding(true_a)
    true_b_view = _phase_embedding(true_b)
    baseline = _energy_distance(true_a_view, true_b_view)
    value = _energy_distance(completed_view, true_a_view)
    return {
        "energy_distance_completed_vs_real_a": value,
        "energy_distance_real_a_vs_real_b": baseline,
        "R_phi": value / max(baseline, STANDARD_DEVIATION_FLOOR),
        "relative_phase_energy_distance_completed_vs_real_a": _energy_distance(completed_relative, true_a_relative),
        "relative_phase_energy_distance_real_a_vs_real_b": _energy_distance(true_a_relative, true_b_relative),
        "relative_phase_mean_l2_completed": _json_float(completed_relative.square().mean(dim=1).sqrt().mean()),
        "relative_phase_mean_l2_real_a": _json_float(true_a_relative.square().mean(dim=1).sqrt().mean()),
    }


def _complex_representation(z: torch.Tensor) -> torch.Tensor:
    return torch.cat((z.real, z.imag), dim=1).to(torch.float64)


def _complex_comparison(
    sampled: torch.Tensor,
    real_a: torch.Tensor,
    real_b: torch.Tensor,
    train_z: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    train_view = _complex_representation(train_z)
    mean = train_view.mean(dim=0)
    std = train_view.std(dim=0, unbiased=False).clamp_min(STANDARD_DEVIATION_FLOOR)
    sampled_view = _standardize(_complex_representation(sampled), mean, std)
    real_a_view = _standardize(_complex_representation(real_a), mean, std)
    real_b_view = _standardize(_complex_representation(real_b), mean, std)
    energy_sampled = _energy_distance(sampled_view, real_a_view)
    energy_real = _energy_distance(real_a_view, real_b_view)
    phase_sampled = torch.angle(sampled)
    phase_real_a = torch.angle(real_a)
    phase_real_b = torch.angle(real_b)
    phase_mean_delta = wrap_phase(
        torch.atan2(phase_sampled.sin().mean(dim=0), phase_sampled.cos().mean(dim=0))
        - torch.atan2(phase_real_a.sin().mean(dim=0), phase_real_a.cos().mean(dim=0))
    )
    return (
        {
            "energy_distance_sampled_vs_real_a": energy_sampled,
            "energy_distance_real_a_vs_real_b": energy_real,
            "R_z": energy_sampled / max(energy_real, STANDARD_DEVIATION_FLOOR),
            "real_derived_scale_source": "all 55,000 saved final TRAIN z_final rows",
            "real_derived_mean_by_real_imag_port": _float_list(mean),
            "real_derived_std_by_real_imag_port": _float_list(std),
            "covariance_and_correlation": _covariance_comparison(sampled_view, real_a_view),
            "real_a_vs_real_b_covariance_and_correlation": _covariance_comparison(real_a_view, real_b_view),
            "relative_phase": {
                "mean_abs_circular_mean_delta_rad": _json_float(phase_mean_delta.abs().mean()),
                "mean_one_minus_cos_delta": _json_float((1.0 - phase_mean_delta.cos()).mean()),
                "real_a_vs_real_b_mean_abs_circular_mean_delta_rad": _json_float(
                    wrap_phase(
                        torch.atan2(phase_real_a.sin().mean(dim=0), phase_real_a.cos().mean(dim=0))
                        - torch.atan2(phase_real_b.sin().mean(dim=0), phase_real_b.cos().mean(dim=0))
                    ).abs().mean()
                ),
            },
        },
        sampled_view,
        real_a_view,
    )


def _raw_cmos_statistics(raw: torch.Tensor, *, name: str) -> dict[str, Any]:
    _require(raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 200, 200), f"{name} raw CMOS shape is invalid")
    _finite(raw, f"{name} raw CMOS")
    view = raw.to(torch.float64).flatten(1)
    per_sample_std = view.std(dim=1, unbiased=False)
    per_sample_min = view.min(dim=1).values
    per_sample_max = view.max(dim=1).values
    finite_fraction = _json_float(torch.isfinite(raw).to(torch.float64).mean())
    nonconstant_fraction = _json_float((per_sample_std > 0.0).to(torch.float64).mean())
    zero_image_fraction = _json_float((view.abs().sum(dim=1) == 0.0).to(torch.float64).mean())
    return {
        "name": name,
        "shape": list(raw.shape),
        "dtype": str(raw.dtype),
        "finite": True,
        "finite_fraction": finite_fraction,
        "nonconstant_fraction": nonconstant_fraction,
        "zero_image_fraction": zero_image_fraction,
        "zero_image_count": int((view.abs().sum(dim=1) == 0.0).sum().item()),
        "nonconstant_sample_count": int((per_sample_std > 0.0).sum().item()),
        "constant_sample_count": int((per_sample_std == 0.0).sum().item()),
        "zero_fraction": _json_float((raw == 0).to(torch.float64).mean()),
        "intensity": _summary(raw),
        "total_intensity": _summary(view.sum(dim=1)),
        "dynamic_range": _summary(per_sample_max - per_sample_min),
        "per_sample_std": _summary(per_sample_std),
        "clamped_evaluator_view_fraction_at_zero": _json_float((raw <= 0.0).to(torch.float64).mean()),
        "clamped_evaluator_view_fraction_at_one": _json_float((raw >= 1.0).to(torch.float64).mean()),
    }


def _load_evaluator_identity(root: Path, report: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    run_config = _mapping(report.get("run_config"), "report run_config")
    evaluator_config = _mapping(run_config.get("evaluator"), "report evaluator config")
    implementation = root / EVALUATOR_IMPLEMENTATION_RELATIVE
    checkpoint = root / str(evaluator_config.get("checkpoint"))
    _require(implementation.is_file(), "frozen evaluator implementation is missing")
    _require(checkpoint.is_file(), "frozen evaluator checkpoint is missing")
    implementation_sha = file_sha256(implementation)
    checkpoint_sha = file_sha256(checkpoint)
    identity = {
        "implementation": EVALUATOR_IMPLEMENTATION_RELATIVE.as_posix(),
        "implementation_sha256": implementation_sha,
        "checkpoint": str(checkpoint.relative_to(root).as_posix()),
        "checkpoint_sha256": checkpoint_sha,
        "architecture": "FrozenMNISTEvaluator v1",
        "view": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28",
        "role": "frozen diagnostic only; labels and gradients are not consumed",
    }
    return checkpoint, identity


def _doc_top_guard(root: Path) -> dict[str, Any]:
    status_path = root / "docs/CURRENT_STATUS.md"
    handoff_path = root / "docs/status/current_handoff.md"
    status_top = "\n".join(status_path.read_text(encoding="utf-8").splitlines()[:40]) if status_path.is_file() else ""
    handoff_top = "\n".join(handoff_path.read_text(encoding="utf-8").splitlines()[:40]) if handoff_path.is_file() else ""
    return {
        "status_path": status_path.relative_to(root).as_posix(),
        "status_top_sha256": hashlib.sha256(status_top.encode("utf-8")).hexdigest(),
        "handoff_path": handoff_path.relative_to(root).as_posix(),
        "handoff_top_sha256": hashlib.sha256(handoff_top.encode("utf-8")).hexdigest(),
        "role": "provenance only; document edits do not change scientific input admission",
    }


def _validate_frozen_inputs(root: Path) -> dict[str, Any]:
    identity = git_identity(root)
    docs = _doc_top_guard(root)

    report_path = root / REPORT_RELATIVE
    bundle_path = root / BUNDLE_RELATIVE
    final_path = root / FINAL_COLLECTION_RELATIVE
    _require(report_path.is_file() and bundle_path.is_file() and final_path.is_file(), "frozen bundle/report/final collection input is missing")
    report_sha = file_sha256(report_path)
    bundle_sha = file_sha256(bundle_path)
    final_sha = file_sha256(final_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    _require(report.get("schema") == "chen_bogt_complex_ols_bounded_training_report.v1", "training report schema changed")
    _require(report.get("result") == "VALID_COMPLETED_GENERATION_BUNDLE_READY", "training report is not the frozen bundle-ready result")
    _require(report.get("outer_classification") == "A", "training report outer classification changed")
    _require(report.get("project_mainline") == SEMANTIC_AUTHORITY_ID, "training report project mainline changed")
    _require(report.get("current_authorized_next_action") == "NONE", "training report current next action changed")
    _require(report.get("scientific_facts_modified") is False, "training report scientific-fact flag changed")
    _require(_mapping(report.get("bundle"), "training report bundle").get("generation_inputs_complete") is True, "training bundle is incomplete")
    _require(_mapping(report.get("bundle"), "training report bundle").get("generation_executed") is False, "training report already executed generation")

    bundle = safe_torch_load(bundle_path, map_location="cpu")
    _require(isinstance(bundle, Mapping), "generation bundle is not a mapping")
    _require(bundle.get("schema") == BUNDLE_SCHEMA, "generation bundle schema changed")
    _require(bundle.get("bundle_role") == "GENERATION_INPUTS_COMPLETE", "generation bundle role changed")
    for flag in ("generation_inputs_complete",):
        _require(bundle.get(flag) is True, f"generation bundle flag is false: {flag}")
    for flag in ("generation_executed", "generation_success", "novel_sample_generated", "promotion_authorized"):
        _require(bundle.get(flag) is False, f"generation bundle historical flag is not false: {flag}")
    _require(bundle.get("selected_step") == 300, "bundle selected step is not 300")
    bundle_identity = _mapping(bundle.get("run_identity"), "bundle run identity")
    report_identity = _mapping(report.get("run_identity"), "report run identity")
    for key in ("task_id", "run_id", "trajectory_id", "semantic_authority_id", "data_role", "config_identity", "model_identity", "optimizer_identity", "sampler_identity", "scientific_attempt_id"):
        _require(bundle_identity.get(key) == report_identity.get(key), f"bundle/report execution identity differs: {key}")
    _require(bundle.get("run_config") == report.get("run_config"), "bundle/report run config differs")

    run_config = _mapping(bundle.get("run_config"), "bundle run_config")
    _require(run_config.get("propagation_grid_size") == PRODUCTION_PROPAGATION_GRID, "bundle is not the production 2048 grid")
    _require(run_config.get("selection") == "fixed final update step 300", "bundle selection rule changed")
    _require(run_config.get("ols") == "10x10 complex, 100 ports, row-major, T=I_100", "bundle OLS contract changed")
    _require(run_config.get("train_count") == TRAIN_COUNT, "bundle TRAIN count changed")

    refs = _mapping(bundle.get("frozen_references"), "bundle frozen references")
    scale = _tensor(refs.get("scale"), "bundle s_j")
    power_reference = _tensor(refs.get("power_reference"), "bundle r_p")
    _require(scale.shape == (OLS_PORT_COUNT,) and scale.dtype == torch.float64, "bundle s_j shape/dtype changed")
    _require(power_reference.shape == (5,) and power_reference.dtype == torch.float64, "bundle r_p shape/dtype changed")
    _finite(scale, "bundle s_j")
    _finite(power_reference, "bundle r_p")
    _require(bool((scale > EPS_S).all().item()), "bundle s_j is not strictly positive")
    _require(bool((power_reference > 0.0).all().item()), "bundle r_p is not strictly positive")
    refs_identity = _mapping(refs.get("identity"), "bundle references identity")

    prior = _mapping(bundle.get("prior"), "bundle prior")
    prior_loc = _tensor(prior.get("loc"), "bundle prior loc")
    prior_variance = _tensor(prior.get("variance"), "bundle prior variance")
    _require(prior_loc.shape == (OLS_PORT_COUNT,) and prior_variance.shape == (OLS_PORT_COUNT,), "bundle prior shape changed")
    _require(prior_loc.dtype == torch.float64 and prior_variance.dtype == torch.float64, "bundle prior dtype changed")
    _require(bool(torch.equal(prior_loc, torch.ones_like(prior_loc))), "bundle prior loc is not fixed at 1")
    _require(bool(torch.equal(prior_variance, torch.full_like(prior_variance, 0.25))), "bundle prior variance is not fixed at 0.25")

    ols = _mapping(bundle.get("ols"), "bundle OLS")
    transport = _tensor(ols.get("transport"), "bundle OLS transport")
    _require(transport.shape == (OLS_PORT_COUNT, OLS_PORT_COUNT) and transport.dtype == torch.complex64, "bundle OLS transport shape/dtype changed")
    _require(bool(torch.equal(transport, torch.eye(OLS_PORT_COUNT, dtype=torch.complex64))), "bundle OLS transport is not T=I_100")
    _require(ols.get("transport_contract") == "T=I_100", "bundle transport contract changed")
    _require(ols.get("writer") == "W_OLS(z)=sum_j (Tz)_j v_j", "bundle W_OLS contract changed")

    state_container = _mapping(bundle.get("final_encoder_generator"), "bundle final encoder/generator")
    _require(state_container.get("selection") == "scientific_step_000300", "bundle final state selection changed")
    _require(state_container.get("frozen_after_selection") is True, "bundle final state is not frozen")
    state = _mapping(state_container.get("state"), "bundle final trainable state")
    _require(set(state) == {"encoder.phase", "generator.phase"}, "bundle trainable state allowlist changed")
    for name, value in state.items():
        tensor = _tensor(value, f"bundle {name}")
        _require(tensor.shape == (2, 1, 100, 100) and tensor.dtype == torch.float32, f"bundle {name} shape/dtype changed")
        _finite(tensor, f"bundle {name}")

    phase_model_payload = _mapping(bundle.get("phase_model"), "bundle phase model")
    phase_model = _phase_model_from_bundle(phase_model_payload)
    _require(phase_model.fit_data_role == "train" and phase_model.fit_count == TRAIN_COUNT, "phase model TRAIN fit identity changed")
    _require(phase_model.engineering_only is False, "phase model is engineering-only")
    _require(bool(torch.allclose(phase_model.cholesky, torch.tril(phase_model.cholesky), rtol=0.0, atol=0.0)), "phase model Cholesky is not lower triangular")
    _require(bool((torch.diag(phase_model.cholesky) > 0.0).all().item()), "phase model Cholesky diagonal is not positive")

    final_payload = safe_torch_load(final_path, map_location="cpu")
    _require(isinstance(final_payload, Mapping) and final_payload.get("schema") == FINAL_COLLECTION_SCHEMA, "final TRAIN OLS schema changed")
    _require(final_payload.get("selected_step") == 300, "final TRAIN OLS selected step changed")
    z_final = _tensor(final_payload.get("z_final"), "final TRAIN z")
    u_final = _tensor(final_payload.get("u_final"), "final TRAIN u")
    source_indices = _tensor(final_payload.get("source_indices"), "final TRAIN source indices")
    _require(z_final.shape == (TRAIN_COUNT, OLS_PORT_COUNT) and z_final.dtype == torch.complex64, "final TRAIN z shape/dtype changed")
    _require(u_final.shape == (TRAIN_COUNT, OLS_PORT_COUNT) and u_final.dtype == torch.float64, "final TRAIN u shape/dtype changed")
    _require(source_indices.shape == (TRAIN_COUNT,) and source_indices.dtype == torch.long, "final TRAIN source indices changed")
    _finite(z_final, "final TRAIN z")
    _finite(u_final, "final TRAIN u")
    _require(bool((u_final >= 0.0).all().item()), "final TRAIN u left nonnegative support")
    _require(bool(torch.equal(statistical_variable(z_final, scale).cpu(), u_final)), "saved final TRAIN u is not the frozen s_j transform")
    final_refs_identity = _mapping(final_payload.get("references_identity"), "final TRAIN references identity")
    for key in ("schema", "data_role", "count", "engineering_only"):
        _require(final_refs_identity.get(key) == refs_identity.get(key), f"final TRAIN references {key} differs from bundle")

    checkpoint, evaluator_identity = _load_evaluator_identity(root, report)
    historical_p0 = _load_historical_p0_authority(root)
    p2_candidate = _load_p2_candidate(root)
    device = torch.device("cuda:0")
    _require(torch.cuda.is_available(), "CUDA is required for the optical canary")
    _require(torch.cuda.device_count() >= 1, "no CUDA device is available")
    evaluator, evaluator_payload = load_frozen_evaluator(checkpoint, device=device)
    del evaluator, evaluator_payload

    consumer = LightGen2DChenBOGTComplexOLSTrainingConsumer(
        device=device,
        propagation_grid_size=PRODUCTION_PROPAGATION_GRID,
    )
    _require(consumer.runtime_identity() == ols.get("interface"), "current runtime identity differs from frozen bundle runtime")
    consumer.load_trainable_state_dict(state)
    consumer.runtime.eval()
    for parameter in consumer.parameters():
        parameter.requires_grad_(False)
    _require(all(not parameter.requires_grad for parameter in consumer.parameters()), "canary runtime parameter freeze failed")

    return {
        "report": dict(report),
        "bundle": dict(bundle),
        "report_path": report_path,
        "bundle_path": bundle_path,
        "final_path": final_path,
        "final_payload": dict(final_payload),
        "z_final": z_final,
        "u_final": u_final,
        "source_indices": source_indices,
        "scale": scale,
        "power_reference": power_reference,
        "phase_model": phase_model,
        "evaluator_checkpoint": checkpoint,
        "evaluator_identity": evaluator_identity,
        "consumer": consumer,
        "git_before": identity,
        "docs_guard": docs,
        "input_hashes": {
            "report": report_sha,
            "bundle": bundle_sha,
            "final_collection": final_sha,
            "evaluator_checkpoint": EXPECTED_EVALUATOR_CHECKPOINT_SHA256,
            "evaluator_implementation": EXPECTED_EVALUATOR_IMPLEMENTATION_SHA256,
            "historical_canary_report": historical_p0["report_sha256"],
            "historical_latent": historical_p0["latent_sha256"],
            "p2_candidate": p2_candidate["candidate_sha256"],
            "p2_manifest": p2_candidate["manifest_sha256"],
        },
        "historical_p0": historical_p0,
        "p2_candidate": p2_candidate,
    }


def _ensure_output(root: Path, output_dir: Path, *, resume: bool) -> Path:
    target = output_dir if output_dir.is_absolute() else root / output_dir
    target = target.resolve()
    _require(root.resolve() in target.parents, "canary output must remain inside the repository")
    _require(target.relative_to(root).as_posix().startswith("outputs/"), "canary output must remain below outputs/")
    if resume:
        _require(target.is_dir(), f"cannot resume missing canary output directory: {target}")
        _require(not (target / "sampled_generation_canary_report.json").exists(), "completed canary report already exists; refusing resume")
        return target
    _require(not target.exists(), f"canary output directory already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    probe = target / ".write_probe"
    probe.write_text("canary output writable\n", encoding="utf-8")
    probe.unlink()
    return target


def _execution_source_paths(root: Path) -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_chen_bogt_complex_ols_training.py",
        root / "src/ldm_project/v2/application/research/lightgen/lightgen_2d_semantic_generation_input_ols_alignment.py",
        root / "src/ldm_project/v2/application/research/shared/research_run_control.py",
        root / EVALUATOR_IMPLEMENTATION_RELATIVE,
    )


def _make_identity(source_manifest: Mapping[str, Any]) -> ResearchRunIdentity:
    sampler_identity = {
        "producer": P2_CANDIDATE_TYPE,
        "historical_p0_seed_identity": PRIOR_SEED,
        "p2_source_sampling_seed": 20_260_924,
        "conditional_phase_seed": CONDITIONAL_PHASE_SEED,
        "real_reference_selection_seed": REAL_REFERENCE_SELECTION_SEED,
        "real_u_phase_seed": REAL_U_PHASE_SEED,
        "N_LATENT": N_LATENT,
        "N_REAL_REFERENCE": N_REAL_REFERENCE,
        "N_OPTICAL": N_OPTICAL,
    }
    return ResearchRunIdentity(
        task_id=TASK_ID,
        run_id=RUN_ID,
        trajectory_id=TRAJECTORY_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        data_role="train_reference_ols_only",
        config_identity=sampler_identity,
        model_identity="LIGHTGEN_2D_CHEN_BOGT_COMPLEX_OLS_REPAIRED_PRIOR_GENERATION_CANARY_V1",
        optimizer_identity={"name": "none", "backward": False},
        sampler_identity=sampler_identity,
        execution_source_semantics_id=str(source_manifest["execution_source_semantics_id"]),
        source_manifest_id=str(source_manifest["source_manifest_id"]),
        scientific_attempt_id=SCIENTIFIC_ATTEMPT_ID,
    )


def _write_initial_manifests(
    *,
    root: Path,
    target: Path,
    source_manifest: Mapping[str, Any],
    identity: ResearchRunIdentity,
    context: Mapping[str, Any],
) -> None:
    atomic_write_new_json(target / "source_manifest.json", dict(source_manifest), artifact_kind="canary source manifest")
    atomic_write_new_json(
        target / "execution_manifest.json",
        {
            "schema": "chen_bogt_repaired_prior_generation_canary_execution_manifest.v1",
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "authorization_id": AUTHORIZATION_ID,
            "run_identity": identity.to_dict(),
            "source_manifest": dict(source_manifest),
            "input_artifact_hashes": dict(context["input_hashes"]),
            "git_before": dict(context["git_before"]),
            "scope": {
                "N_LATENT": N_LATENT,
                "N_REAL_REFERENCE": N_REAL_REFERENCE,
                "N_OPTICAL": N_OPTICAL,
                "optical_batch_size": OPTICAL_BATCH_SIZE,
                "producer": P2_CANDIDATE_TYPE,
                "historical_p0_exact_reuse": True,
                "generation_only": True,
                "training": False,
                "promotion": False,
            },
        },
        artifact_kind="canary execution manifest",
    )


def _reuse_reference_bank(
    z_final: torch.Tensor,
    u_final: torch.Tensor,
    scale: torch.Tensor,
    historical_p0: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    latent = _mapping(historical_p0.get("latent"), "historical P0 latent")
    indices = _tensor(latent.get("real_reference_source_indices"), "historical reference indices").to(torch.long)
    _require(torch.unique(indices).numel() == N_REAL_REFERENCE, "real reference selection is not distinct")
    selected_z = z_final.index_select(0, indices)
    selected_u = u_final.index_select(0, indices)
    stored_u = torch.cat(
        (
            _tensor(latent.get("real_u_a"), "historical real-A u"),
            _tensor(latent.get("real_u_b"), "historical real-B u"),
        ),
        dim=0,
    )
    stored_z = torch.cat(
        (
            _tensor(latent.get("real_z_a_gauged"), "historical real-A gauged z"),
            _tensor(latent.get("real_z_b_gauged"), "historical real-B gauged z"),
        ),
        dim=0,
    )
    _require(torch.equal(selected_u, stored_u), "historical real reference u bank differs from final TRAIN OLS")
    gauged, _, _ = max_q_gauge(selected_z, scale)
    _require(torch.equal(gauged, stored_z), "historical real reference gauged z differs from final TRAIN OLS")
    return {
        "indices": indices.to(torch.long),
        "z": selected_z,
        "u": selected_u,
    }


def _stage_a_support(
    *,
    u_sample: torch.Tensor,
    u_train: torch.Tensor,
    real_a_u: torch.Tensor,
    real_b_u: torch.Tensor,
    producer: Mapping[str, Any],
) -> dict[str, Any]:
    h_train = torch.log(u_train + LOG_EPS)
    h_mean = h_train.mean(dim=0)
    h_std = h_train.std(dim=0, unbiased=False).clamp_min(STANDARD_DEVIATION_FLOOR)
    del h_train
    h_sample = _standardize(torch.log(u_sample + LOG_EPS), h_mean, h_std)
    h_real_a = _standardize(torch.log(real_a_u + LOG_EPS), h_mean, h_std)
    h_real_b = _standardize(torch.log(real_b_u + LOG_EPS), h_mean, h_std)
    ed_sample_real = _energy_distance(h_sample, h_real_a)
    ed_real_real = _energy_distance(h_real_a, h_real_b)
    sample_quantiles = torch.quantile(u_sample, torch.tensor([0.005, 0.01, 0.5, 0.99, 0.995], dtype=torch.float64), dim=0)
    train_quantiles = torch.quantile(u_train, torch.tensor([0.005, 0.01, 0.5, 0.99, 0.995], dtype=torch.float64), dim=0)
    support_fraction = ((u_sample >= u_train.amin(dim=0)) & (u_sample <= u_train.amax(dim=0))).to(torch.float64).mean(dim=0)
    norm_sample = h_sample.norm(dim=1, keepdim=True)
    norm_real_a = h_real_a.norm(dim=1, keepdim=True)
    norm_real_b = h_real_b.norm(dim=1, keepdim=True)
    return {
            "standardization": {
                "source": "all 55,000 saved final TRAIN u_final rows",
                "formula": "h=log(u+1e-12); h_std=(h-mean_train_h)/std_train_h",
                "mean_h_by_port": _float_list(h_mean),
                "std_h_by_port": _float_list(h_std),
            },
            "prior": {
                **dict(producer),
                "support": _summary(u_sample),
                "per_port_mean": _float_list(u_sample.mean(dim=0)),
                "per_port_std_population": _float_list(u_sample.std(dim=0, unbiased=False)),
                "per_port_quantiles_0.005_0.01_0.5_0.99_0.995": [_float_list(row) for row in sample_quantiles],
                "quantile_comparison_to_frozen_real_train": {
                    "real_train_quantiles_0.005_0.01_0.5_0.99_0.995": [_float_list(row) for row in train_quantiles],
                    "quantile_difference_sample_minus_real_train": [_float_list(row) for row in (sample_quantiles - train_quantiles)],
                },
                "fraction_inside_frozen_real_train_port_ranges": _float_list(support_fraction),
            },
            "joint_support": {
                "energy_distance_sample_real_a": ed_sample_real,
                "energy_distance_real_a_real_b": ed_real_real,
                "R_u": ed_sample_real / max(ed_real_real, STANDARD_DEVIATION_FLOOR),
                "covariance_and_correlation": _covariance_comparison(h_sample, h_real_a),
                "real_a_vs_real_b_covariance_and_correlation": _covariance_comparison(h_real_a, h_real_b),
                "norm_distribution_energy_distance_sample_real_a": _energy_distance(norm_sample, norm_real_a),
                "norm_distribution_energy_distance_real_a_real_b": _energy_distance(norm_real_a, norm_real_b),
                "norm_distribution_R": _energy_distance(norm_sample, norm_real_a) / max(_energy_distance(norm_real_a, norm_real_b), STANDARD_DEVIATION_FLOOR),
                "sample_effective_rank": _effective_rank(h_sample),
                "real_a_effective_rank": _effective_rank(h_real_a),
            },
            "no_arbitrary_threshold_used": True,
            "metric_interpretation": "R_u and the component distances are descriptive; no acceptance threshold is applied in Stage A",
        }


def _construct_gauged_z(
    u: torch.Tensor,
    phase_model: WrappedConditionalPhaseModel,
    scale: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    phase = phase_model.sample(u, generator=generator)
    amplitude = ((scale + EPS_S) * u).clamp_min(0.0).sqrt()
    z = torch.complex(amplitude * phase.cos(), amplitude * phase.sin()).to(torch.complex64)
    gauged, _, anchors = max_q_gauge(z, scale)
    return gauged.cpu(), torch.angle(gauged).cpu(), anchors.cpu()


def _evaluate_optical(
    *,
    consumer: LightGen2DChenBOGTComplexOLSTrainingConsumer,
    sampled_z: torch.Tensor,
    real_control_z: torch.Tensor,
    heartbeat: HeartbeatWriter,
    recovery: RecoveryCheckpointManager,
    interrupt: InterruptController,
    state: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    _require(sampled_z.shape[0] >= N_OPTICAL and real_control_z.shape[0] >= N_OPTICAL, "optical canary inputs are too short")
    device = consumer.device

    def run_group(values: torch.Tensor, group: str, offset: int) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for start in range(0, N_OPTICAL, OPTICAL_BATCH_SIZE):
            if interrupt.requested:
                interrupt.handle_safe_boundary(step=offset + start, latest_losses={"canary_phase": group, "completed": start})
                raise CanaryInterrupted(f"interrupt persisted at {group} step {start}")
            stop = min(start + OPTICAL_BATCH_SIZE, N_OPTICAL)
            batch = values[start:stop].to(device)
            with torch.no_grad():
                result = consumer.generated_forward(batch)
                raw = result.raw_intensity.detach().cpu()
            _finite(raw, f"{group} raw CMOS")
            outputs.append(raw)
            state["canary_phase"] = group
            state["completed_optical"] = offset + stop
            now = time.time()
            if recovery_due(
                step=offset + stop,
                last_recovery_step=state["last_recovery_step"],
                now=now,
                last_recovery_time=state["last_recovery_time"],
                interval_steps=25,
                interval_seconds=600.0,
            ):
                checkpoint = recovery.save(
                    {"kind": "SAMPLED_GENERATION_CANARY_RECOVERY_V1", "phase": group, "completed_optical": offset + stop},
                    step=offset + stop,
                )
                state["last_recovery_step"] = offset + stop
                state["last_recovery_time"] = now
                state["last_recovery_checkpoint"] = checkpoint.name
            heartbeat.update(
                state=RUNNING,
                phase="POST_TRAIN_COLLECTION",
                step=offset + stop,
                latest_losses={"canary_phase": group, "completed_optical": offset + stop},
                last_recovery_checkpoint=state.get("last_recovery_checkpoint"),
            )
        return torch.cat(outputs, dim=0)

    generated = run_group(sampled_z, "GENERATED_SAMPLE", 0)
    real = run_group(real_control_z, "REAL_CONTROL", N_OPTICAL)
    return generated, real


def _evaluate_frozen_semantics(
    *,
    checkpoint: Path,
    generated_raw: torch.Tensor,
    real_raw: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    device = torch.device("cuda:0")
    evaluator, payload = load_frozen_evaluator(checkpoint, device=device)
    evaluator.eval()
    outputs: dict[str, torch.Tensor] = {}
    report: dict[str, Any] = {
        "implementation": "FrozenMNISTEvaluator v1",
        "checkpoint_validation_accuracy": float(payload.get("validation_accuracy", 0.0)),
        "view": "clamp raw CMOS intensity to [0,1], then adaptive_avg_pool2d to 28x28",
        "labels_consumed": False,
        "accuracy": None,
        "recall": None,
        "macro_recall": None,
    }
    for name, raw in (("generated", generated_raw), ("real_control", real_raw)):
        with torch.no_grad():
            view = torch.nn.functional.adaptive_avg_pool2d(raw.clamp(0.0, 1.0).to(device), (28, 28))
            embedding, logits = evaluator(view)
        embedding_cpu = embedding.detach().cpu()
        logits_cpu = logits.detach().cpu()
        probabilities = torch.softmax(logits_cpu, dim=1)
        predicted = probabilities.argmax(dim=1)
        histogram = torch.bincount(predicted, minlength=10)
        histogram_probability = histogram.to(torch.float64) / float(predicted.numel())
        nonzero = histogram_probability > 0.0
        entropy = -(histogram_probability[nonzero] * histogram_probability[nonzero].log()).sum()
        confidence = probabilities.max(dim=1).values
        outputs[f"{name}_embedding"] = embedding_cpu
        outputs[f"{name}_logits"] = logits_cpu
        report[name] = {
            "predicted_class_histogram": _int_list(histogram),
            "predicted_class_dominant_fraction": _json_float(histogram_probability.max()),
            "predicted_class_entropy_nats": _json_float(entropy),
            "predicted_class_entropy_normalized": _json_float(entropy / math.log(10.0)),
            "confidence": _summary(confidence),
            "feature_effective_rank": _effective_rank(embedding_cpu),
            "feature_pairwise_distance": _diversity_view(embedding_cpu, name=f"{name}_evaluator_features")["pairwise_distance"],
            "evaluator_feature_shape": list(embedding_cpu.shape),
        }
    generated_features = outputs["generated_embedding"].to(torch.float64)
    real_features = outputs["real_control_embedding"].to(torch.float64)
    real_first, real_second = real_features[:100], real_features[100:]
    feature_self = _energy_distance(real_first, real_second)
    feature_cross = _energy_distance(generated_features, real_features)
    report["comparison"] = {
        "feature_energy_distance_generated_vs_real_control": feature_cross,
        "feature_energy_distance_real_control_split": feature_self,
        "R_feature": feature_cross / max(feature_self, STANDARD_DEVIATION_FLOOR),
        "feature_covariance_and_correlation": _covariance_comparison(generated_features, real_features),
    }
    return report, outputs


def _make_contact_sheet(raw: torch.Tensor, path: Path) -> dict[str, Any]:
    _require(raw.shape[0] >= CONTACT_SHEET_COUNT, "not enough generated samples for contact sheet")
    view = raw[:CONTACT_SHEET_COUNT, 0].clamp(0.0, 1.0)
    canvas = Image.new("L", (10 * CONTACT_TILE_SIZE, 10 * CONTACT_TILE_SIZE), color=0)
    for index in range(CONTACT_SHEET_COUNT):
        pixels = (view[index].numpy() * 255.0).round().astype("uint8")
        tile = Image.fromarray(pixels, mode="L").resize((CONTACT_TILE_SIZE, CONTACT_TILE_SIZE), Image.Resampling.BILINEAR)
        canvas.paste(tile, ((index % 10) * CONTACT_TILE_SIZE, (index // 10) * CONTACT_TILE_SIZE))
    canvas.save(path, format="PNG")
    return {
        "path": path.name,
        "sample_indices": list(range(CONTACT_SHEET_COUNT)),
        "grid": [10, 10],
        "tile_size": [CONTACT_TILE_SIZE, CONTACT_TILE_SIZE],
        "display": "same evaluator view: clamp raw CMOS intensity to [0,1], convert to uint8, bilinear tile resize",
        "sha256": file_sha256(path),
    }


def _localization(
    *,
    stage_a: Mapping[str, Any],
    phase: Mapping[str, Any],
    complex_stage: Mapping[str, Any],
    raw_ratio: float,
    evaluator: Mapping[str, Any],
    diversity: Mapping[str, Any],
) -> dict[str, Any]:
    r_u = float(_mapping(stage_a["joint_support"], "Stage A joint support")["R_u"])
    r_phi = float(phase["R_phi"])
    r_z = float(complex_stage["R_z"])
    r_raw = float(raw_ratio)
    r_feature = float(_mapping(evaluator["comparison"], "evaluator comparison")["R_feature"])
    generated_semantic = _mapping(evaluator.get("generated"), "generated evaluator report")
    real_semantic = _mapping(evaluator.get("real_control"), "real evaluator report")
    generated_rank = float(_mapping(generated_semantic["feature_effective_rank"], "generated feature rank")["effective_rank"])
    real_rank = float(_mapping(real_semantic["feature_effective_rank"], "real feature rank")["effective_rank"])
    generated_dom = float(generated_semantic["predicted_class_dominant_fraction"])
    real_dom = float(real_semantic["predicted_class_dominant_fraction"])
    collapse_score = max(
        real_rank / max(generated_rank, 1.0),
        generated_dom / max(real_dom, STANDARD_DEVIATION_FLOOR),
        1.0 + float(_mapping(diversity["generated_feature"], "generated feature diversity")["exact_duplicate_rows"] > 0),
    )
    scores = {
        "PRIOR_SUPPORT_MISMATCH_CANDIDATE": r_u,
        "PHASE_COMPLETION_MISMATCH_CANDIDATE": r_phi,
        "FULL_COMPLEX_LATENT_MISMATCH_CANDIDATE": r_z,
        "OPTICAL_GENERATOR_MISMATCH_CANDIDATE": max(r_raw, r_feature),
        "SEMANTIC_OUTPUT_COLLAPSE_CANDIDATE": collapse_score,
    }
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    candidates_over_reference = [name for name, score in ordered if score > 1.0]
    nonsemantic_candidates_over_reference = [
        name for name in candidates_over_reference if name != "SEMANTIC_OUTPUT_COLLAPSE_CANDIDATE"
    ]
    if not candidates_over_reference:
        primary = "NO_DOMINANT_CANARY_MISMATCH_IDENTIFIED"
    elif len(nonsemantic_candidates_over_reference) >= 2:
        primary = "MULTI_STAGE_MISMATCH"
    else:
        primary = ordered[0][0]
    return {
        "primary_diagnosis": primary,
        "candidate_scores_relative_to_real_reference": scores,
        "candidate_order_descending": [name for name, _ in ordered],
        "real_reference_baseline": "1.0 means no larger than the corresponding real-control split/reference distance",
        "heuristic": "scores are descriptive ratios to real-control self-reference; no pass/fail or separation threshold is applied, and multiple non-semantic departures are reported as MULTI_STAGE_MISMATCH",
        "inputs": {
            "R_u": r_u,
            "R_phi": r_phi,
            "R_z": r_z,
            "R_raw": r_raw,
            "R_feature": r_feature,
            "semantic_collapse_score": collapse_score,
        },
    }


def _counterfactual_interpretation(
    *,
    historical_p0: Mapping[str, float],
    p2_metrics: Mapping[str, float],
) -> dict[str, Any]:
    metric_names = ("R_u", "R_z", "R_raw", "R_feature")
    historical = {name: float(historical_p0[name]) for name in metric_names}
    p2 = {name: float(p2_metrics[name]) for name in metric_names}
    delta = {name: p2[name] - historical[name] for name in metric_names}
    ratio = {name: p2[name] / max(historical[name], STANDARD_DEVIATION_FLOOR) for name in metric_names}
    lower = {name: p2[name] < historical[name] for name in metric_names}
    if all(lower.values()):
        branch = "P2_AMPLITUDE_SUPPORT_REPAIR_PROPAGATES_DOWNSTREAM"
    elif lower["R_u"] and lower["R_z"] and not lower["R_raw"]:
        branch = "GENERATOR_OR_OPTICAL_SYNTHESIS_LEADING_CANDIDATE_BOTTLENECK"
    elif lower["R_u"] and lower["R_z"] and lower["R_raw"] and not lower["R_feature"]:
        branch = "SEMANTIC_FEATURE_MAPPING_LEADING_CANDIDATE_BOTTLENECK"
    elif lower["R_u"] and not lower["R_z"]:
        branch = "PHASE_COMPLETION_LEADING_CANDIDATE_BOTTLENECK"
    else:
        branch = "DOWNSTREAM_PROPAGATION_MIXED"
    return {
        "branch": branch,
        "historical_p0": historical,
        "p2": p2,
        "absolute_delta_p2_minus_p0": delta,
        "ratio_p2_over_p0": ratio,
        "strict_lower_direction": lower,
        "decision_rule": "descriptive strict direction only; no magnitude threshold or acceptance threshold is applied",
        "R_phi_context": float(p2_metrics["R_phi"]),
    }


def _write_cmos_artifact(path: Path, raw: torch.Tensor, *, group: str, z: torch.Tensor) -> None:
    atomic_torch_save_new(
        path,
        {
            "schema": CMOS_SCHEMA,
            "group": group,
            "raw_cmos_intensity": raw.cpu().contiguous(),
            "input_z": z.cpu().contiguous(),
            "sample_count": int(raw.shape[0]),
            "optical_batch_size": OPTICAL_BATCH_SIZE,
            "postprocessing": "none",
        },
        artifact_kind=f"{group} CMOS canary artifact",
    )


def _load_existing_cmos(path: Path, *, group: str, expected_z: torch.Tensor) -> torch.Tensor:
    payload = safe_torch_load(path, map_location="cpu")
    _require(isinstance(payload, Mapping) and payload.get("schema") == CMOS_SCHEMA, f"existing {group} CMOS schema changed")
    _require(payload.get("group") == group, f"existing {group} CMOS group changed")
    raw = _tensor(payload.get("raw_cmos_intensity"), f"existing {group} raw CMOS")
    input_z = _tensor(payload.get("input_z"), f"existing {group} input z")
    _require(raw.ndim == 4 and tuple(raw.shape[1:]) == (1, 200, 200) and raw.dtype == torch.float32, f"existing {group} raw CMOS shape/dtype changed")
    _require(torch.equal(input_z, expected_z), f"existing {group} input z differs from deterministic canary latent")
    _finite(raw, f"existing {group} raw CMOS")
    return raw.contiguous()


def _validate_existing_latent(path: Path, expected: Mapping[str, torch.Tensor]) -> None:
    payload = safe_torch_load(path, map_location="cpu")
    _require(isinstance(payload, Mapping) and payload.get("schema") == LATENT_SCHEMA, "existing sampled latent bank schema changed")
    for key, value in expected.items():
        observed = _tensor(payload.get(key), f"existing latent {key}")
        _require(torch.equal(observed, value), f"existing latent {key} differs from deterministic canary latent")


def _existing_contact_sheet(path: Path) -> dict[str, Any]:
    _require(path.is_file(), "existing contact sheet is missing")
    with Image.open(path) as image:
        _require(image.mode == "L" and image.size == (10 * CONTACT_TILE_SIZE, 10 * CONTACT_TILE_SIZE), "existing contact sheet shape/mode changed")
    return {
        "path": path.name,
        "sample_indices": list(range(CONTACT_SHEET_COUNT)),
        "grid": [10, 10],
        "tile_size": [CONTACT_TILE_SIZE, CONTACT_TILE_SIZE],
        "display": "same evaluator view: clamp raw CMOS intensity to [0,1], convert to uint8, bilinear tile resize",
        "sha256": file_sha256(path),
        "reused": True,
    }


def _write_latent_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_torch_save_new(path, dict(payload), artifact_kind="sampled latent bank")


def _artifact_summary(root: Path, target: Path) -> dict[str, Any]:
    names = (
        "sampled_latent_bank.pt",
        "generated_cmos_canary.pt",
        "real_control_cmos.pt",
        "generated_contact_sheet.png",
        "progress.jsonl",
        "latest_status.json",
    )
    summary = {
        name: _artifact_ref(root, target / name, role="canary artifact")
        for name in names
    }
    summary["sampled_generation_canary_report.json"] = {
        "path": (target / "sampled_generation_canary_report.json").relative_to(root).as_posix(),
        "role": "canary report (self-hash intentionally omitted)",
    }
    return summary


def run_canary(output_dir: Path, *, resume: bool = False) -> dict[str, Any]:
    root = _repo_root()
    context = _validate_frozen_inputs(root)
    target = _ensure_output(root, output_dir, resume=resume)
    scientific_artifact_paths = (
        target / "sampled_latent_bank.pt",
        target / "generated_cmos_canary.pt",
        target / "real_control_cmos.pt",
        target / "generated_contact_sheet.png",
    )
    scientific_artifacts_present = [path.exists() for path in scientific_artifact_paths]
    if resume and any(scientific_artifacts_present) and not all(scientific_artifacts_present):
        raise CanaryError("resume found partial scientific artifacts; refusing to rerun or overwrite them")
    reuse_completed_artifacts = resume and all(scientific_artifacts_present)

    config_identity = {
        "task_id": TASK_ID,
        "input_hashes": context["input_hashes"],
        "sizes": {"N_LATENT": N_LATENT, "N_REAL_REFERENCE": N_REAL_REFERENCE, "N_OPTICAL": N_OPTICAL},
        "seeds": {
            "historical_p0_seed_identity": PRIOR_SEED,
            "p2_source_sampling_seed": 20_260_924,
            "conditional_phase": CONDITIONAL_PHASE_SEED,
            "real_reference_selection": REAL_REFERENCE_SELECTION_SEED,
            "real_u_phase": REAL_U_PHASE_SEED,
        },
        "optical_batch_size": OPTICAL_BATCH_SIZE,
    }
    source_manifest = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=_execution_source_paths(root),
        config_identity=config_identity,
    ).to_dict()
    identity = _make_identity(source_manifest)
    engineering_repair: dict[str, Any] | None = None
    if resume:
        stored_source_path = target / "source_manifest.json"
        stored_execution_path = target / "execution_manifest.json"
        _require(stored_execution_path.is_file(), "resume execution record is missing")
        stored_source = json.loads(stored_source_path.read_text(encoding="utf-8")) if stored_source_path.is_file() else {}
        stored_execution = json.loads(stored_execution_path.read_text(encoding="utf-8"))
        _require(stored_execution.get("task_id") == TASK_ID, "resume task identity changed")
        _require(stored_execution.get("authorization_id") == AUTHORIZATION_ID, "resume authorization identity changed")
        engineering_repair = {
            "schema": "chen_bogt_repaired_prior_generation_canary_engineering_repair.v1",
            "reason": (
                "resume after report assembly, descriptive energy-distance estimator, Stage-E report completeness, and layered-localization classification repairs; 400 direct forwards and scientific artifacts were already published"
                if reuse_completed_artifacts
                else "resume after pre-optical CUDA/NVRTC environment failure; no optical result or canary artifact was published"
            ),
            "scientific_forward_count_before_repair": 400 if reuse_completed_artifacts else 0,
            "reused_published_scientific_artifacts": reuse_completed_artifacts,
            "old_source_manifest_id": stored_source.get("source_manifest_id"),
            "old_execution_source_semantics_id": stored_source.get("execution_source_semantics_id"),
            "new_source_manifest_id": source_manifest["source_manifest_id"],
            "new_execution_source_semantics_id": source_manifest["execution_source_semantics_id"],
            "input_artifact_hashes": context["input_hashes"],
        }
        repair_sequence = len(list(target.glob("execution_source_repair*.json"))) + 1
        suffix = "" if repair_sequence == 1 else f"_v{repair_sequence:02d}"
        atomic_write_new_json(target / f"source_manifest_repaired{suffix}.json", source_manifest, artifact_kind="repaired canary source manifest")
        atomic_write_new_json(target / f"execution_source_repair{suffix}.json", engineering_repair, artifact_kind="canary engineering repair record")
        atomic_write_new_json(
            target / f"execution_manifest_repaired{suffix}.json",
            {
                **stored_execution,
                "source_manifest": source_manifest,
                "run_identity": identity.to_dict(),
                "engineering_repair": engineering_repair,
            },
            artifact_kind="repaired canary execution manifest",
        )
    else:
        _write_initial_manifests(root=root, target=target, source_manifest=source_manifest, identity=identity, context=context)

    heartbeat = HeartbeatWriter(target, RUN_ID, total_steps=N_OPTICAL * 2, interval_seconds=60.0)
    heartbeat.update(state=RUN_PREPARING, phase="STATISTICS", step=0, latest_losses={"canary_phase": "FAST_PREFLIGHT_COMPLETE"}, force=True)
    control = ResearchRunControl(identity, AUTHORIZATION_ID)
    device = torch.device("cuda:0")
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    correctness_passed = bool(
        context["report"].get("result") == "VALID_COMPLETED_GENERATION_BUNDLE_READY"
        and context["bundle"].get("bundle_role") == "GENERATION_INPUTS_COMPLETE"
        and context["bundle"].get("selected_step") == 300
        and context["bundle"].get("run_config", {}).get("propagation_grid_size") == PRODUCTION_PROPAGATION_GRID
        and tuple(context["p2_candidate"]["u"].shape) == (N_LATENT, OLS_PORT_COUNT)
    )
    receipt = control.make_ready_receipt(
        correctness={
            "passed": correctness_passed,
            "frozen_report_bundle_and_train_collection_validated": correctness_passed,
            "protected_test_sealed_reads": {"protected": 0, "test": 0, "sealed": 0},
        },
        resource={
            "passed": int(total_bytes) > 0 and 0 <= int(free_bytes) <= int(total_bytes),
            "device": torch.cuda.get_device_name(device),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "training_or_optimizer_updates": 0,
        },
        implementation={
            "passed": context["bundle"].get("run_config", {}).get("propagation_grid_size") == PRODUCTION_PROPAGATION_GRID,
            "selected_engine": "chen_bogt_sampled_generation_cuda_2048_direct",
            "propagation_grid_size": PRODUCTION_PROPAGATION_GRID,
            "optimizer_created": False,
            "backward_called": False,
        },
        selected_engine="chen_bogt_sampled_generation_cuda_2048_direct",
        resource_envelope={"free_bytes": int(free_bytes), "total_bytes": int(total_bytes)},
    )
    preflight_record = target / "preflight_receipt.json"
    if preflight_record.exists():
        preflight_record = target / f"preflight_receipt_recheck_{time.time_ns()}.json"
    atomic_write_new_json(preflight_record, receipt.to_dict(), artifact_kind="canary preflight receipt")
    control.mark_preflight_ready(receipt)
    control.start_execution()
    control.open_scientific_attempt(target, SCIENTIFIC_ATTEMPT_ID, receipt, resume=True)
    recovery = RecoveryCheckpointManager(
        target,
        identity,
        source_manifest_id=source_manifest["source_manifest_id"],
        authorization_id=AUTHORIZATION_ID,
    )
    state: dict[str, Any] = {
        "canary_phase": "INITIALIZED",
        "completed_optical": 0,
        "last_recovery_step": 0,
        "last_recovery_time": None,
        "last_recovery_checkpoint": None,
    }
    recovery.save({"kind": "SAMPLED_GENERATION_CANARY_RECOVERY_V1", "phase": "PREPARED", "completed_optical": 0}, step=0)
    interrupt = InterruptController(
        heartbeat=heartbeat,
        recovery=recovery,
        payload_factory=lambda step: {
            "kind": "SAMPLED_GENERATION_CANARY_RECOVERY_V1",
            "phase": state["canary_phase"],
            "completed_optical": state["completed_optical"],
            "step": step,
        },
        phase="POST_TRAIN_COLLECTION",
        total_steps=N_OPTICAL * 2,
    )

    try:
        heartbeat.update(state=RUNNING, phase="STATISTICS", step=0, latest_losses={"canary_phase": "STAGE_A_PRIOR_SUPPORT"}, force=True)
        phase_generator = torch.Generator(device="cpu").manual_seed(CONDITIONAL_PHASE_SEED)
        real_u_phase_generator = torch.Generator(device="cpu").manual_seed(REAL_U_PHASE_SEED)
        scale = context["scale"].cpu()
        z_train = context["z_final"].cpu()
        u_train = context["u_final"].cpu()
        refs = _reuse_reference_bank(z_train, u_train, scale, context["historical_p0"])
        real_a_z = refs["z"][:1_000]
        real_b_z = refs["z"][1_000:]
        real_a_u = refs["u"][:1_000]
        real_b_u = refs["u"][1_000:]
        u_sample = context["p2_candidate"]["u"].clone()
        producer = {
            "family": P2_CANDIDATE_TYPE,
            "candidate_role": P2_CANDIDATE_CLAIM,
            "sampling": "direct consumption of frozen candidate u; no refit, resampling, or empirical-quantile regeneration",
            "candidate_sha256": context["p2_candidate"]["candidate_sha256"],
            "manifest_sha256": context["p2_candidate"]["manifest_sha256"],
        }
        stage_a = _stage_a_support(
            u_sample=u_sample,
            u_train=u_train,
            real_a_u=real_a_u,
            real_b_u=real_b_u,
            producer=producer,
        )
        recovery.save({"kind": "SAMPLED_GENERATION_CANARY_RECOVERY_V1", "phase": "STAGE_A_COMPLETE", "completed_optical": 0}, step=1)

        heartbeat.update(state=RUNNING, phase="CALIBRATION", step=0, latest_losses={"canary_phase": "STAGE_B_PHASE_COMPLETION"}, force=True)
        real_u_z_completed, real_u_phase, real_u_anchors = _construct_gauged_z(
            real_a_u,
            context["phase_model"],
            scale,
            generator=real_u_phase_generator,
        )
        real_a_gauged, _, real_a_anchors = max_q_gauge(real_a_z, scale)
        real_b_gauged, _, real_b_anchors = max_q_gauge(real_b_z, scale)
        stage_b_phase = _phase_distribution_comparison(
            real_u_phase,
            torch.angle(real_a_gauged),
            torch.angle(real_b_gauged),
            real_u_anchors,
            real_a_anchors,
            real_b_anchors,
        )
        phase_diagnostics = {
            "real_u_phase_completion": _phase_diagnostics(real_u_phase, real_u_anchors, name="real_u_phase_completion"),
            "real_a_true_phase": _phase_diagnostics(torch.angle(real_a_gauged), real_a_anchors, name="real_a_true_phase"),
            "real_b_true_phase": _phase_diagnostics(torch.angle(real_b_gauged), real_b_anchors, name="real_b_true_phase"),
            "comparison": stage_b_phase,
            "phase_source": "frozen conditional phase model; true real phases used only as diagnostic comparison",
        }
        recovery.save({"kind": "SAMPLED_GENERATION_CANARY_RECOVERY_V1", "phase": "STAGE_B_COMPLETE", "completed_optical": 0}, step=2)

        heartbeat.update(state=RUNNING, phase="CALIBRATION", step=0, latest_losses={"canary_phase": "STAGE_C_FULL_COMPLEX_LATENT"}, force=True)
        z_sample, phase_sample, sample_anchors = _construct_gauged_z(
            u_sample,
            context["phase_model"],
            scale,
            generator=phase_generator,
        )
        latent_stage = {
            "sample_shape": list(z_sample.shape),
            "sample_dtype": str(z_sample.dtype),
            "magnitude_preservation": {
                "max_abs_error_u": _json_float((statistical_variable(z_sample, scale) - u_sample).abs().max()),
                "mean_abs_error_u": _json_float((statistical_variable(z_sample, scale) - u_sample).abs().mean()),
                "max_relative_error_u": _json_float(
                    ((statistical_variable(z_sample, scale) - u_sample).abs() / u_sample.clamp_min(STANDARD_DEVIATION_FLOOR)).max()
                ),
            },
            "phase_diagnostics": _phase_diagnostics(phase_sample, sample_anchors, name="prior_sample_phase"),
        }
        complex_stage, _, _ = _complex_comparison(z_sample, real_a_gauged, real_b_gauged, z_train)
        latent_stage["complex_comparison"] = complex_stage
        recovery.save({"kind": "SAMPLED_GENERATION_CANARY_RECOVERY_V1", "phase": "STAGE_C_COMPLETE", "completed_optical": 0}, step=3)

        state["canary_phase"] = "OPTICAL_GENERATOR"
        heartbeat.update(state=RUNNING, phase="POST_TRAIN_COLLECTION", step=0, latest_losses={"canary_phase": "STAGE_D_E_OPTICAL_GENERATOR"}, force=True)
        if reuse_completed_artifacts:
            generated_raw = _load_existing_cmos(
                target / "generated_cmos_canary.pt",
                group="generated",
                expected_z=z_sample[:N_OPTICAL],
            )
            real_control_raw = _load_existing_cmos(
                target / "real_control_cmos.pt",
                group="real_control",
                expected_z=real_a_gauged[:N_OPTICAL],
            )
            state["canary_phase"] = "REUSED_OPTICAL_GENERATOR_ARTIFACTS"
            state["completed_optical"] = 400
            heartbeat.update(
                state=RUNNING,
                phase="POST_TRAIN_COLLECTION",
                step=400,
                latest_losses={"canary_phase": state["canary_phase"], "completed_optical": 400},
                force=True,
            )
        else:
            with interrupt:
                generated_raw, real_control_raw = _evaluate_optical(
                    consumer=context["consumer"],
                    sampled_z=z_sample[:N_OPTICAL],
                    real_control_z=real_a_gauged[:N_OPTICAL],
                    heartbeat=heartbeat,
                    recovery=recovery,
                    interrupt=interrupt,
                    state=state,
                )
        generated_stats = _raw_cmos_statistics(generated_raw, name="generated")
        real_control_stats = _raw_cmos_statistics(real_control_raw, name="real_control")
        real_raw_view = real_control_raw.flatten(1).to(torch.float64)
        generated_raw_view = generated_raw.flatten(1).to(torch.float64)
        raw_mean = real_raw_view.mean(dim=0)
        raw_std = real_raw_view.std(dim=0, unbiased=False).clamp_min(STANDARD_DEVIATION_FLOOR)
        generated_raw_standardized = _standardize(generated_raw_view, raw_mean, raw_std)
        real_raw_standardized = _standardize(real_raw_view, raw_mean, raw_std)
        raw_cross = _energy_distance(generated_raw_standardized, real_raw_standardized)
        raw_self = _energy_distance(real_raw_standardized[:100], real_raw_standardized[100:])
        optical = {
            "N_OPTICAL": N_OPTICAL,
            "batch_size": OPTICAL_BATCH_SIZE,
            "input": "direct complex OLS z -> existing T=I_100/W_OLS -> frozen 2-layer generator -> raw CMOS",
            "generated": generated_stats,
            "real_control": real_control_stats,
            "raw_distribution_comparison": {
                "standardization": "per-pixel mean/std from the 200 real-control raw CMOS frames",
                "energy_distance_generated_vs_real_control": raw_cross,
                "energy_distance_real_control_first100_vs_second100": raw_self,
                "R_raw": raw_cross / max(raw_self, STANDARD_DEVIATION_FLOOR),
            },
            "forbidden_paths_used": {
                "encoder": False,
                "labels": False,
                "electronic_decoder": False,
                "adapter": False,
                "phase_repair": False,
                "posthoc_optimization": False,
            },
        }
        recovery.save({"kind": "SAMPLED_GENERATION_CANARY_RECOVERY_V1", "phase": "OPTICAL_COMPLETE", "completed_optical": 400}, step=400)

        heartbeat.update(state=RUNNING, phase="POST_TRAIN_FIT", step=400, latest_losses={"canary_phase": "STAGE_F_SEMANTIC_EVALUATOR"}, force=True)
        evaluator_report, evaluator_outputs = _evaluate_frozen_semantics(
            checkpoint=context["evaluator_checkpoint"],
            generated_raw=generated_raw,
            real_raw=real_control_raw,
        )
        diversity = {
            "generated_latent": _diversity_view(z_sample, name="generated_latent_z"),
            "real_control_latent": _diversity_view(real_a_gauged[:N_OPTICAL], name="real_control_latent_z"),
            "generated_raw": _diversity_view(generated_raw, name="generated_raw_cmos"),
            "real_control_raw": _diversity_view(real_control_raw, name="real_control_raw_cmos"),
            "generated_feature": _diversity_view(evaluator_outputs["generated_embedding"], name="generated_evaluator_feature"),
            "real_control_feature": _diversity_view(evaluator_outputs["real_control_embedding"], name="real_control_evaluator_feature"),
        }
        latent_bank_path = target / "sampled_latent_bank.pt"
        expected_latent_tensors = {
            "u_sample": u_sample,
            "phase_sample": phase_sample,
            "z_sample": z_sample,
            "real_u_a": real_a_u,
            "real_u_b": real_b_u,
            "real_z_a_gauged": real_a_gauged,
            "real_z_b_gauged": real_b_gauged,
            "real_u_phase_completion": real_u_phase,
            "real_u_z_completion": real_u_z_completed,
            "real_reference_source_indices": refs["indices"],
            "real_reference_a_indices": refs["indices"][:1_000],
            "real_reference_b_indices": refs["indices"][1_000:],
        }
        if reuse_completed_artifacts:
            _validate_existing_latent(latent_bank_path, expected_latent_tensors)
        else:
            _write_latent_artifact(
                latent_bank_path,
                {
                    "schema": LATENT_SCHEMA,
                    "source_bundle_sha256": context["input_hashes"]["bundle"],
                    "selected_step": 300,
                    "producer": P2_CANDIDATE_TYPE,
                    "candidate_sha256": context["p2_candidate"]["candidate_sha256"],
                    "manifest_sha256": context["p2_candidate"]["manifest_sha256"],
                    "seeds": {
                        "historical_p0_seed_identity": PRIOR_SEED,
                        "p2_source_sampling_seed": 20_260_924,
                        "conditional_phase": CONDITIONAL_PHASE_SEED,
                        "real_reference_selection": REAL_REFERENCE_SELECTION_SEED,
                        "real_u_phase": REAL_U_PHASE_SEED,
                    },
                    **expected_latent_tensors,
                },
            )
        generated_cmos_path = target / "generated_cmos_canary.pt"
        real_cmos_path = target / "real_control_cmos.pt"
        if not reuse_completed_artifacts:
            _write_cmos_artifact(generated_cmos_path, generated_raw, group="generated", z=z_sample[:N_OPTICAL])
            _write_cmos_artifact(real_cmos_path, real_control_raw, group="real_control", z=real_a_gauged[:N_OPTICAL])
            contact_sheet = _make_contact_sheet(generated_raw, target / "generated_contact_sheet.png")
        else:
            contact_sheet = _existing_contact_sheet(target / "generated_contact_sheet.png")

        localization = _localization(
            stage_a=stage_a,
            phase=stage_b_phase,
            complex_stage=complex_stage,
            raw_ratio=optical["raw_distribution_comparison"]["R_raw"],
            evaluator=evaluator_report,
            diversity=diversity,
        )
        p2_metrics = {
            "R_u": float(stage_a["joint_support"]["R_u"]),
            "R_phi": float(stage_b_phase["R_phi"]),
            "R_z": float(complex_stage["R_z"]),
            "R_raw": float(optical["raw_distribution_comparison"]["R_raw"]),
            "R_feature": float(evaluator_report["comparison"]["R_feature"]),
        }
        counterfactual = _counterfactual_interpretation(
            historical_p0=context["historical_p0"]["metrics"],
            p2_metrics=p2_metrics,
        )
        localization["absolute_reference_primary_diagnosis"] = localization["primary_diagnosis"]
        localization["primary_diagnosis"] = counterfactual["branch"]
        localization["historical_counterfactual"] = counterfactual

        recovery.save({"kind": "SAMPLED_GENERATION_CANARY_RECOVERY_V1", "phase": "FINALIZING", "completed_optical": 400}, step=401)
        heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=N_OPTICAL * 2, latest_losses={"canary_phase": "COMPLETED"}, force=True)
        control.complete()
        artifacts = _artifact_summary(root, target)
        report = {
            "schema": REPORT_SCHEMA,
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "status": "VALID_COMPLETED",
            "outer_classification": "VALID_COMPLETED_CANARY_DIAGNOSTIC_ONLY",
            "project_mainline": SEMANTIC_AUTHORITY_ID,
            "current_authorized_next_action": "NONE",
            "action_executed": TASK_ID,
            "next_action_executed": False,
            "scientific_facts_modified": False,
            "promotion_authorized": False,
            "training_executed": False,
            "optimizer_created": False,
            "backward_called": False,
            "cuda_forward_scope": "400 direct generator forwards: 200 sampled + 200 real controls",
            "git": {
                "branch": context["git_before"]["branch"],
                "head": context["git_before"]["head"],
                "dirty_before": context["git_before"]["worktree_dirty"],
                "dirty_after": git_identity(root)["worktree_dirty"],
                "status_sha256_before": context["git_before"]["status_sha256"],
            },
            "input_authorities": {
                "training_report": _artifact_ref(root, context["report_path"], role="frozen upstream training report"),
                "generation_ready_bundle": _artifact_ref(root, context["bundle_path"], role="frozen generation-ready bundle"),
                "final_train_ols": _artifact_ref(root, context["final_path"], role="frozen final TRAIN OLS collection"),
                "historical_p0_canary_report": _artifact_ref(root, context["historical_p0"]["report_path"], role="accepted historical P0 canary report"),
                "historical_p0_latent_bank": _artifact_ref(root, context["historical_p0"]["latent_path"], role="accepted historical P0/reference latent bank"),
                "p2_candidate": _artifact_ref(root, context["p2_candidate"]["candidate_path"], role="frozen P2 repair candidate"),
                "p2_manifest": _artifact_ref(root, context["p2_candidate"]["manifest_path"], role="frozen P2 repair candidate manifest"),
                "evaluator": context["evaluator_identity"],
            },
            "run_identity": identity.to_dict(),
            "source_manifest_id": source_manifest["source_manifest_id"],
            "source_manifest": dict(source_manifest),
            "doc_top_guard": context["docs_guard"],
            "engineering_repair": engineering_repair,
            "preflight": {
                "status": "READY",
                "bundle_identity": True,
                "report_identity": True,
                "final_collection_identity": True,
                "historical_p0_identity": True,
                "historical_reference_exact_reuse": True,
                "p2_candidate_identity": True,
                "p2_candidate_shape": [N_LATENT, OLS_PORT_COUNT],
                "p2_candidate_dtype": "torch.float64",
                "p2_candidate_finite": True,
                "p2_candidate_nonnegative": True,
                "p2_refit": False,
                "p2_resampling": False,
                "selected_step": 300,
                "phase_model_identity": True,
                "prior_identity": P2_CANDIDATE_TYPE,
                "transport_identity": "T=I_100",
                "writer_identity": "W_OLS",
                "generator_state_loaded": True,
                "production_propagation_grid": PRODUCTION_PROPAGATION_GRID,
                "cuda_device": torch.cuda.get_device_name(torch.device("cuda:0")),
                "output_directory_new_and_writable": True,
                "training_preflight": False,
                "tn_kl_suite": False,
                "optimizer_tests": False,
                "large_forward_preflight": False,
                "phase_fit_suite": False,
            },
            "seeds": {
                "historical_p0_seed_identity": PRIOR_SEED,
                "p2_source_sampling_seed": 20_260_924,
                "conditional_phase": CONDITIONAL_PHASE_SEED,
                "real_reference_selection": REAL_REFERENCE_SELECTION_SEED,
                "real_u_phase": REAL_U_PHASE_SEED,
            },
            "sample_sizes": {"N_LATENT": N_LATENT, "N_REAL_REFERENCE": N_REAL_REFERENCE, "N_OPTICAL": N_OPTICAL},
            "reference_bank": {
                "source": "exact reuse of the accepted historical P0 latent bank references; saved final TRAIN OLS only; no images or labels were re-encoded/read",
                "selection": "historical real_reference_source_indices; no re-selection",
                "distinct": True,
                "a_rows": "selected rows 0:1000",
                "b_rows": "selected rows 1000:2000",
                "indices_tensor_sha256": tensor_sha256(refs["indices"]),
                "source_indices": _int_list(refs["indices"]),
            },
            "stage_a_prior_support": stage_a,
            "stage_b_phase_completion": phase_diagnostics,
            "stage_c_full_complex_latent": latent_stage,
            "stage_d_e_raw_cmos": optical,
            "stage_f_frozen_evaluator": evaluator_report,
            "stage_g_diversity": diversity,
            "stage_h_contact_sheet": contact_sheet,
            "stage_i_layered_localization": localization,
            "historical_p0_baseline": {
                "metrics": dict(context["historical_p0"]["metrics"]),
                "report_sha256": context["historical_p0"]["report_sha256"],
                "latent_sha256": context["historical_p0"]["latent_sha256"],
            },
            "required_measurements": {
                "R_u_P0_historical": context["historical_p0"]["metrics"]["R_u"],
                "R_u_P2": p2_metrics["R_u"],
                "R_phi_frozen_context": p2_metrics["R_phi"],
                "R_z_P0_historical": context["historical_p0"]["metrics"]["R_z"],
                "R_z_P2": p2_metrics["R_z"],
                "R_raw_P0_historical": context["historical_p0"]["metrics"]["R_raw"],
                "R_raw_P2": p2_metrics["R_raw"],
                "R_feature_P0_historical": context["historical_p0"]["metrics"]["R_feature"],
                "R_feature_P2": p2_metrics["R_feature"],
                "absolute_delta_p2_minus_p0": counterfactual["absolute_delta_p2_minus_p0"],
                "ratio_p2_over_p0": counterfactual["ratio_p2_over_p0"],
                "raw_validity_and_diversity": {
                    "generated_finite_fraction": generated_stats["finite_fraction"],
                    "generated_nonconstant_fraction": generated_stats["nonconstant_fraction"],
                    "generated_zero_image_count": generated_stats["zero_image_count"],
                    "generated_class_histogram": evaluator_report["generated"]["predicted_class_histogram"],
                    "generated_dominant_class_fraction": evaluator_report["generated"]["predicted_class_dominant_fraction"],
                    "generated_normalized_entropy": evaluator_report["generated"]["predicted_class_entropy_normalized"],
                },
            },
            "historical_boundaries": {
                "no_protected_test_sealed_read": True,
                "no_promotion": True,
                "no_best_of_n": True,
                "no_resampling": True,
                "no_alternative_checkpoint_prior_phase": True,
                "no_electronic_decoder": True,
                "no_posthoc_phase_repair": True,
            },
            "data_reads": {
                "saved_final_train_ols_rows_available": TRAIN_COUNT,
                "saved_final_train_ols_rows_selected": N_REAL_REFERENCE,
                "raw_images_read": 0,
                "labels_read": 0,
                "protected_read": 0,
                "test_read": 0,
                "sealed_read": 0,
            },
            "recovery": {
                "same_run_resume_identity": identity.to_dict(),
                "recovery_interval_steps": 25,
                "recovery_interval_seconds": 600.0,
                "latest": recovery.latest().name if recovery.latest() is not None else None,
                "artifacts": [_artifact_ref(root, path, role="operational recovery checkpoint") for path in sorted(target.glob("recovery_*.pt"))],
            },
            "artifacts": artifacts,
            "input_hashes": context["input_hashes"],
            "no_automatic_follow_up": True,
        }
        report_path = target / "sampled_generation_canary_report.json"
        atomic_write_new_json(report_path, report, artifact_kind="sampled generation canary report")
        return report
    except CanaryInterrupted:
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="execute the one explicitly authorized canary")
    parser.add_argument("--resume", action="store_true", help="resume the same attempt after an operational pre-optical stop")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required; this entrypoint does not run a dry or alternate scientific mode")
    try:
        report = run_canary(args.output_dir, resume=args.resume)
    except CanaryInterrupted as error:
        print(json.dumps({"status": "INTERRUPTED", "message": str(error)}, ensure_ascii=False, indent=2))
        return 2
    except (CanaryError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "status": report["status"],
        "outer_gate": report["outer_gate"],
        "primary_diagnosis": report["stage_i_layered_localization"]["primary_diagnosis"],
        "report": {"path": str(((_repo_root() / args.output_dir) if not args.output_dir.is_absolute() else args.output_dir).resolve() / "sampled_generation_canary_report.json")},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
