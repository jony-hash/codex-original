"""Calibrate frozen Chen BOGT latent-support candidates on CPU only.

This consumer resolves the finite-sample primary support compatibility of the
frozen attempt_08 P1/P2 banks.  It does not refit, resample, or execute an
optical model.  The primary statistic is the existing biased V-statistic
energy distance in the existing FIT-derived standardized log-u coordinate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from ldm_project.v2.application.research.research_run_control import (
    HeartbeatWriter,
    RUN_COMPLETED,
    RUN_PREPARING,
    RUNNING,
    capture_execution_source_manifest,
)
from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import (
    execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1 as canary,
    execute_lightgen_2d_chen_bogt_latent_prior_support_repair_v1 as decomposition,
)


TASK_ID = "CALIBRATE_AND_RESOLVE_CHEN_BOGT_LATENT_SUPPORT_COMPATIBILITY_V1"
OUTER_GATE = "CHEN_BOGT_LATENT_SUPPORT_COMPATIBILITY_RESOLVED"
SEMANTIC_AUTHORITY_ID = "M0_LIGHTGEN_INPUT_OLS_ALIGNED_GENERATION"
RUN_ID = "chen-bogt-latent-support-compatibility-v1"
TRAJECTORY_ID = "chen-bogt-latent-support-compatibility-trajectory-v1"
SCIENTIFIC_ATTEMPT_ID = f"{RUN_ID}:attempt_01"

EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a9cd1749adfa690e282fd52a6425564fc2cf102c"

DECOMPOSITION_RUN = Path(
    "outputs/20260918_chen_bogt_latent_prior_support_repair_v1/runs/attempt_08"
)
DECOMPOSITION_REPORT_PATH = DECOMPOSITION_RUN / "latent_prior_support_decomposition_report.json"
DECOMPOSITION_SPLIT_PATH = DECOMPOSITION_RUN / "split_manifest.json"
P1_BANK_PATH = DECOMPOSITION_RUN / "P1_empirical_independent_u.pt"
P2_BANK_PATH = DECOMPOSITION_RUN / "P2_empirical_gaussian_copula_u.pt"
P3_BANK_PATH = DECOMPOSITION_RUN / "P3_real_row_support_control_u.pt"

EXPECTED_DECOMPOSITION_REPORT_SHA256 = "27fdb024ab089b5d90f6ebceaf2cde33c05f3ea54bc9638f6f75c373f24b2a22"
EXPECTED_DECOMPOSITION_SPLIT_SHA256 = "0bbaf2525e5566467ad018c0968478c84d4f2444869686d999f90b1f9541981e"
EXPECTED_P1_BANK_SHA256 = "929965265e0548f77b0454b77fcc7f9af9e8586fa6d5b3f6257de95b8fd2b71d"
EXPECTED_P2_BANK_SHA256 = "f1d153383db0fe82b60fa78bfb1a0de3b8ff92e997d8a4137670e65856910249"
EXPECTED_P3_BANK_SHA256 = "01afae798b5a36a3e2526ceedc3d0287953ccbc0f812b4f6b9f97c9f95db6e20"

DEFAULT_OUTPUT_DIR = Path(
    "outputs/20260918_chen_bogt_latent_support_compatibility_v1/runs/attempt_01"
)

N_TOTAL = 55_000
DIMENSION = 100
FIT_COUNT = 45_000
EVAL_COUNT = 5_000
COMPARISON_COUNT = 1_000
PERMUTATION_COUNT = 999
PERMUTATION_SEED = 20_260_926
LOG_EPS = 1.0e-12
STANDARD_DEVIATION_EPS = 1.0e-12
PERMUTATION_BLOCK_SIZE = 32

REPORT_SCHEMA = "chen_bogt_latent_support_compatibility_report.v1"
CANDIDATE_SCHEMA = "chen_bogt_latent_prior_repair_candidate.v1"
CANDIDATE_MANIFEST_SCHEMA = "chen_bogt_latent_prior_repair_candidate_manifest.v1"


class CompatibilityError(RuntimeError):
    """Raised when the frozen support-calibration boundary is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompatibilityError(message)


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _finite(value: torch.Tensor, name: str) -> None:
    _require(bool(torch.isfinite(value).all().item()), f"{name} is non-finite")


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
    return {
        "branch": branch,
        "head": head,
        "dirty": bool(status_lines),
        "status_sha256": hashlib.sha256("\n".join(status_lines).encode("utf-8")).hexdigest(),
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
    _require(root.resolve() in target.parents, "output directory must remain inside repository")
    _require(target.relative_to(root).as_posix().startswith("outputs/"), "output must remain below outputs/")
    _require(not target.exists(), f"refusing to reuse write-once output: {target}")
    target.mkdir(parents=True, exist_ok=False)
    probe = target / ".write_probe"
    probe.write_text("latent support compatibility\n", encoding="utf-8")
    probe.unlink()
    return target


def _load_json(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    _require(path.is_file(), f"{name} is missing: {path}")
    _require(file_sha256(path) == expected_sha256, f"{name} hash changed: {path}")
    return dict(_mapping(json.loads(path.read_text(encoding="utf-8")), name))


def _load_bank(root: Path, path: Path, expected_sha256: str, expected_type: str) -> dict[str, Any]:
    absolute = root / path
    payload = safe_torch_load(absolute, expected_sha256=expected_sha256, map_location="cpu")
    payload = dict(_mapping(payload, path.name))
    _require(payload.get("candidate_type") == expected_type, f"{path.name} candidate type changed")
    sample = payload.get("u")
    _require(isinstance(sample, torch.Tensor), f"{path.name} u bank is missing")
    _require(tuple(sample.shape) == (COMPARISON_COUNT, DIMENSION), f"{path.name} shape changed")
    _require(sample.dtype == torch.float64, f"{path.name} dtype changed")
    _finite(sample, f"{path.name} u bank")
    _require(bool(sample.ge(0.0).all().item()), f"{path.name} u bank is negative")
    payload["u"] = sample.to(device="cpu", dtype=torch.float64).contiguous()
    return payload


def _load_frozen(root: Path) -> dict[str, Any]:
    authorities = decomposition._load_authorities(root)
    report = _load_json(
        root / DECOMPOSITION_REPORT_PATH,
        EXPECTED_DECOMPOSITION_REPORT_SHA256,
        "attempt_08 decomposition report",
    )
    split = _load_json(
        root / DECOMPOSITION_SPLIT_PATH,
        EXPECTED_DECOMPOSITION_SPLIT_SHA256,
        "attempt_08 split manifest",
    )
    _require(report.get("result") == "VALID_COMPLETED", "attempt_08 result changed")
    _require(report.get("outer_gate") == "CHEN_BOGT_LATENT_PRIOR_SUPPORT_REPAIR_RESOLVED", "attempt_08 gate changed")
    _require(report.get("project_mainline") == SEMANTIC_AUTHORITY_ID, "attempt_08 mainline changed")
    _require(report.get("current_authorized_next_action") == "NONE", "attempt_08 next action changed")
    _require(report.get("primary_diagnosis") == "UNRESOLVED", "attempt_08 diagnosis changed")
    metric_identity = _mapping(report.get("metric_identity"), "attempt_08 metric identity")
    _require(
        metric_identity.get("common_coordinate") == "h(u)=log(u+1e-12); mean/std fit only on 45,000 FIT rows; no candidate-specific normalization",
        "attempt_08 support coordinate changed",
    )
    _require(split == report.get("split"), "attempt_08 report and split manifest disagree")
    _require(split.get("schema") == decomposition.SPLIT_SCHEMA, "attempt_08 split schema changed")
    _require(split.get("seed") == decomposition.SPLIT_SEED, "attempt_08 split seed changed")
    _require(split.get("source_row_count") == N_TOTAL, "attempt_08 source row count changed")
    _require(split.get("fit_count") == FIT_COUNT, "attempt_08 FIT count changed")
    _require(split.get("eval_a_count") == EVAL_COUNT and split.get("eval_b_count") == EVAL_COUNT, "attempt_08 EVAL count changed")
    _require(
        split.get("comparison_selection") == "first 1000 rows of each deterministic EVAL partition",
        "attempt_08 comparison selection changed",
    )

    expected_artifacts = {
        "P1_empirical_independent_u.pt": EXPECTED_P1_BANK_SHA256,
        "P2_empirical_gaussian_copula_u.pt": EXPECTED_P2_BANK_SHA256,
        "P3_real_row_support_control_u.pt": EXPECTED_P3_BANK_SHA256,
    }
    report_artifacts = _mapping(report.get("artifacts"), "attempt_08 artifacts")
    for name, expected in expected_artifacts.items():
        record = _mapping(report_artifacts.get(name), f"attempt_08 artifact {name}")
        _require(record.get("sha256") == expected, f"attempt_08 artifact hash declaration changed: {name}")

    p1 = _load_bank(root, P1_BANK_PATH, EXPECTED_P1_BANK_SHA256, "P1_EMPIRICAL_MARGINAL_INDEPENDENT")
    p2 = _load_bank(root, P2_BANK_PATH, EXPECTED_P2_BANK_SHA256, "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA")
    p3 = _load_bank(root, P3_BANK_PATH, EXPECTED_P3_BANK_SHA256, "P3_REAL_ROW_JOINT_SUPPORT_CONTROL")
    _require(p3.get("generative_candidate") is False, "P3 generative boundary changed")
    _require(p1.get("fit_count") == FIT_COUNT and p2.get("fit_count") == FIT_COUNT, "P1/P2 FIT count changed")
    fit_indices = torch.tensor(split["fit_source_indices"], dtype=torch.int64)
    _require(p1.get("fit_source_indices_sha256") == tensor_sha256(fit_indices), "P1 FIT identity changed")
    _require(p2.get("fit_source_indices_sha256") == tensor_sha256(fit_indices), "P2 FIT identity changed")
    p2_fit = _mapping(_mapping(report["empirical_gaussian_copula"], "attempt_08 P2").get("fit"), "attempt_08 P2 fit")
    for key, expected in {
        "R_shrunk": p2_fit.get("R_shrunk_sha256"),
        "cholesky": p2_fit.get("cholesky_sha256"),
    }.items():
        _require(isinstance(p2.get(key), torch.Tensor), f"P2 {key} is missing")
        _require(tensor_sha256(p2[key]) == expected, f"P2 {key} identity changed")
    return {
        "authorities": authorities,
        "report": report,
        "split": split,
        "p1": p1,
        "p2": p2,
        "p3": p3,
    }


def _select_split(authorities: Mapping[str, Any], split: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    u_final = authorities["u_final"]
    source_indices = authorities["source_indices"]
    position_by_source = {int(value): index for index, value in enumerate(source_indices.tolist())}

    fit_ids = [int(value) for value in split["fit_source_indices"]]
    eval_a_ids = [int(value) for value in split["eval_a_source_indices"]]
    eval_b_ids = [int(value) for value in split["eval_b_source_indices"]]
    groups = [set(fit_ids), set(eval_a_ids), set(eval_b_ids)]
    _require([len(group) for group in groups] == [FIT_COUNT, EVAL_COUNT, EVAL_COUNT], "attempt_08 split group size changed")
    _require(not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), "attempt_08 split is not disjoint")
    _require(len(set().union(*groups)) == N_TOTAL, "attempt_08 split does not cover the source collection")
    _require(all(value in position_by_source for value in fit_ids + eval_a_ids + eval_b_ids), "attempt_08 source index is missing")

    def positions(values: list[int]) -> torch.Tensor:
        return torch.tensor([position_by_source[value] for value in values], dtype=torch.int64)

    fit = u_final.index_select(0, positions(fit_ids)).contiguous()
    eval_a = u_final.index_select(0, positions(eval_a_ids[:COMPARISON_COUNT])).contiguous()
    eval_b = u_final.index_select(0, positions(eval_b_ids[:COMPARISON_COUNT])).contiguous()
    return fit, eval_a, eval_b


def _standardized_inputs(
    frozen: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    fit, eval_a, eval_b = _select_split(frozen["authorities"], frozen["split"])
    mean_h, std_h, _ = decomposition._standardization(fit)
    eval_a_h = decomposition._standardize(eval_a, mean_h, std_h)
    eval_b_h = decomposition._standardize(eval_b, mean_h, std_h)
    candidates = {
        "P0_CURRENT_TN": decomposition._standardize(frozen["authorities"]["u_sample"], mean_h, std_h),
        "P1_EMPIRICAL_INDEPENDENT": decomposition._standardize(frozen["p1"]["u"], mean_h, std_h),
        "P2_GAUSSIAN_COPULA": decomposition._standardize(frozen["p2"]["u"], mean_h, std_h),
        "P3_REAL_ROW_CONTROL": decomposition._standardize(frozen["p3"]["u"], mean_h, std_h),
    }
    return fit, candidates, eval_a_h, eval_b_h


def _require_close(observed: float, expected: float, name: str) -> None:
    _require(
        math.isclose(observed, expected, rel_tol=1.0e-10, abs_tol=1.0e-10),
        f"{name} does not reproduce attempt_08: observed={observed} expected={expected}",
    )


def _primary_observed(
    frozen: Mapping[str, Any],
    candidates: Mapping[str, torch.Tensor],
    eval_a_h: torch.Tensor,
    eval_b_h: torch.Tensor,
) -> dict[str, float]:
    previous = {
        "P0_CURRENT_TN": frozen["report"]["current_tn"]["common_split_metrics"],
        "P1_EMPIRICAL_INDEPENDENT": frozen["report"]["empirical_marginal_independent"]["metrics"],
        "P2_GAUSSIAN_COPULA": frozen["report"]["empirical_gaussian_copula"]["metrics"],
        "P3_REAL_ROW_CONTROL": frozen["report"]["real_row_control"]["metrics"],
    }
    observed: dict[str, float] = {}
    for name, candidate in candidates.items():
        value = canary._energy_distance(candidate, eval_a_h)
        observed[name] = float(value)
        _require_close(value, float(previous[name]["energy_distance_candidate_vs_eval_a"]), f"{name} ED")
    real_value = canary._energy_distance(eval_a_h, eval_b_h)
    observed["REAL_REAL"] = float(real_value)
    _require_close(real_value, float(previous["P0_CURRENT_TN"]["energy_distance_eval_a_vs_eval_b"]), "real-real ED")
    return observed


def _pairwise_distance_matrix(value: torch.Tensor) -> torch.Tensor:
    matrix = value.to(device="cpu", dtype=torch.float64).contiguous()
    _require(matrix.ndim == 2 and matrix.shape[0] == 2 * COMPARISON_COUNT, "pooled comparison shape is invalid")
    distances = torch.cdist(matrix, matrix, p=2)
    diagonal = torch.arange(matrix.shape[0], dtype=torch.int64)
    distances[diagonal, diagonal] = 0.0
    _finite(distances, "pooled pairwise distances")
    return distances


def _permutation_null(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    seed: int,
    on_progress: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    pooled = torch.cat([candidate, reference], dim=0).to(device="cpu", dtype=torch.float64).contiguous()
    distances = _pairwise_distance_matrix(pooled)
    row_sums = distances.sum(dim=1)
    total_sum = row_sums.sum()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    assignments = torch.stack(
        [torch.randperm(2 * COMPARISON_COUNT, generator=generator)[:COMPARISON_COUNT] for _ in range(PERMUTATION_COUNT)]
    )
    groups = torch.zeros((PERMUTATION_COUNT, 2 * COMPARISON_COUNT), dtype=torch.float64)
    groups.scatter_(1, assignments, 1.0)
    null = torch.empty(PERMUTATION_COUNT, dtype=torch.float64)
    denominator = float(COMPARISON_COUNT * COMPARISON_COUNT)
    for start in range(0, PERMUTATION_COUNT, PERMUTATION_BLOCK_SIZE):
        stop = min(start + PERMUTATION_BLOCK_SIZE, PERMUTATION_COUNT)
        block = groups[start:stop]
        within_a = (block.mm(distances) * block).sum(dim=1)
        weighted_rows = block.mv(row_sums) if block.shape[0] == 1 else block.mm(row_sums[:, None]).reshape(-1)
        cross = weighted_rows - within_a
        within_b = total_sum - 2.0 * weighted_rows + within_a
        values = 2.0 * cross / denominator - within_a / denominator - within_b / denominator
        null[start:stop] = values.clamp_min(0.0)
        if on_progress is not None:
            on_progress(stop, PERMUTATION_COUNT)
    _finite(null, "permutation null")
    return null


def _null_summary(observed: float, null: torch.Tensor) -> dict[str, Any]:
    levels = torch.tensor([0.50, 0.90, 0.95, 0.99], dtype=torch.float64)
    quantiles = torch.quantile(null, levels)
    greater_or_equal = int((null >= observed).sum().item())
    less_or_equal = int((null <= observed).sum().item())
    q95 = float(quantiles[2])
    return {
        "B": PERMUTATION_COUNT,
        "seed": PERMUTATION_SEED,
        "pooled_rows": 2 * COMPARISON_COUNT,
        "candidate_rows": COMPARISON_COUNT,
        "reference_rows": COMPARISON_COUNT,
        "observed_ED": float(observed),
        "null_median": float(quantiles[0]),
        "null_q90": float(quantiles[1]),
        "null_q95": q95,
        "null_q99": float(quantiles[3]),
        "null_max": float(null.max()),
        "observed_null_percentile": 100.0 * float(less_or_equal) / float(PERMUTATION_COUNT),
        "permutation_p_value": float(1 + greater_or_equal) / float(PERMUTATION_COUNT + 1),
        "compatible_by_q95": bool(observed <= q95),
        "compatibility_rule": "ED_observed <= linear-interpolated q95(ED_perm); p-value reported as (1+count(ED_perm>=ED_observed))/(B+1)",
    }


def _secondary_row(name: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    per_port = _mapping(metrics["per_port"], f"{name} per-port metrics")
    return {
        "candidate": name,
        "ED": float(metrics["energy_distance_candidate_vs_eval_a"]),
        "R_u": float(metrics["R_u"]),
        "covariance_relative_frobenius": float(metrics["covariance_relative_frobenius"]),
        "correlation_relative_frobenius": float(metrics["correlation_relative_frobenius"]),
        "norm_distribution_energy_distance": float(metrics["norm_distribution_energy_distance"]),
        "mean_abs_off_diagonal_correlation_difference": float(metrics["mean_abs_off_diagonal_correlation_difference"]),
        "mean_abs_per_port_mean_difference": float(metrics["mean_abs_per_port_mean_difference"]),
        "mean_abs_per_port_std_difference": float(metrics["mean_abs_per_port_std_difference"]),
        "mean_outside_fit_q005_fraction": float(per_port["mean_outside_fit_q005_fraction"]),
        "mean_outside_fit_q995_fraction": float(per_port["mean_outside_fit_q995_fraction"]),
    }


def _secondary_diagnostics(report: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    metrics = {
        "P0_CURRENT_TN": report["current_tn"]["common_split_metrics"],
        "P1_EMPIRICAL_INDEPENDENT": report["empirical_marginal_independent"]["metrics"],
        "P2_GAUSSIAN_COPULA": report["empirical_gaussian_copula"]["metrics"],
        "P3_REAL_ROW_CONTROL": report["real_row_control"]["metrics"],
    }
    rows = [_secondary_row(name, value) for name, value in metrics.items()]
    warnings: list[str] = []
    if rows[1]["correlation_relative_frobenius"] >= rows[0]["correlation_relative_frobenius"]:
        warnings.append("P1 correlation-relative secondary metric is not lower than P0")
    if rows[1]["mean_abs_off_diagonal_correlation_difference"] >= rows[0]["mean_abs_off_diagonal_correlation_difference"]:
        warnings.append("P1 mean pairwise-correlation secondary metric is not lower than P0")
    if rows[2]["mean_abs_per_port_std_difference"] > rows[1]["mean_abs_per_port_std_difference"]:
        warnings.append("P2 mean per-port standard-deviation discrepancy is higher than P1")
    warnings.append("secondary metrics do not form an all-metric gate")
    return rows, warnings


def _candidate_payload(
    root: Path,
    selected: str,
    bank: Mapping[str, Any],
    primary: Mapping[str, Any],
    split: Mapping[str, Any],
    source_manifest_id: str,
) -> dict[str, Any]:
    bank_paths = {
        "P1_EMPIRICAL_MARGINAL_INDEPENDENT": P1_BANK_PATH,
        "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA": P2_BANK_PATH,
    }
    bank_path = bank_paths[selected]
    payload: dict[str, Any] = {
        "schema": CANDIDATE_SCHEMA,
        "candidate_type": selected,
        "claim": "POST_TRAIN_SAMPLING_REPAIR_CANDIDATE",
        "source_attempt": "DECOMPOSE_AND_REPAIR_CHEN_BOGT_LATENT_PRIOR_SUPPORT_V1 attempt_08",
        "source_manifest_id": source_manifest_id,
        "source_bank": _artifact_ref(root, root / bank_path, role="frozen selected support bank"),
        "u": bank["u"].clone(),
        "fit_split_identity": {
            "split_manifest": _relative_path(root, root / DECOMPOSITION_SPLIT_PATH),
            "split_manifest_sha256": file_sha256(root / DECOMPOSITION_SPLIT_PATH),
            "fit_count": FIT_COUNT,
            "fit_source_indices_sha256": str(bank["fit_source_indices_sha256"]),
            "split_seed": int(split["seed"]),
        },
        "marginal_representation": str(bank["marginal_representation"]),
        "normalization_identity": {
            "coordinate": "h(u)=log(u+1e-12)",
            "standardization": "mean/std fit only on attempt_08 FIT rows; no candidate-specific normalization",
        },
        "sampling_semantics": dict(_mapping(bank["sampling_api"], "selected sampling API")),
        "primary_compatibility": dict(primary),
        "training_boundary": "post-train sampling producer only; historical BOGT objective remains TN_+(1,0.25)",
        "not_generation_success": True,
        "not_retraining_authority": True,
        "not_promotion_authority": True,
    }
    if selected == "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA":
        payload["dependence_parameters"] = {
            "R_shrunk": bank["R_shrunk"].clone(),
            "cholesky": bank["cholesky"].clone(),
            "R_shrunk_sha256": tensor_sha256(bank["R_shrunk"]),
            "cholesky_sha256": tensor_sha256(bank["cholesky"]),
            "dependence_model": str(bank["dependence_model"]),
        }
    return payload


def _artifact_paths(root: Path, target: Path, candidate: Path | None, manifest: Path | None) -> dict[str, dict[str, Any]]:
    paths = {
        "source_manifest.json": "current execution source manifest",
        "execution_manifest.json": "compatibility execution manifest",
    }
    if candidate is not None and manifest is not None:
        paths[candidate.name] = "selected post-train sampling repair candidate"
        paths[manifest.name] = "selected candidate manifest"
    return {name: _artifact_ref(root, target / name, role=role) for name, role in paths.items()}


def run(output_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    git_before = _git_state(root)
    _require(git_before["branch"] == EXPECTED_BRANCH, "local branch conflicts with task authority")
    _require(git_before["head"] == EXPECTED_HEAD, "local HEAD conflicts with task authority")
    frozen = _load_frozen(root)
    fit, candidates, eval_a_h, eval_b_h = _standardized_inputs(frozen)
    observed = _primary_observed(frozen, candidates, eval_a_h, eval_b_h)
    source_manifest = capture_execution_source_manifest(
        root,
        task_id=TASK_ID,
        semantic_authority_id=SEMANTIC_AUTHORITY_ID,
        relevant_paths=(
            Path(__file__).resolve(),
            root / decomposition.__file__,
            root / canary.__file__,
        ),
        config_identity={
            "primary_statistic": "energy distance in standardized log-u",
            "B": PERMUTATION_COUNT,
            "seed": PERMUTATION_SEED,
            "comparison_count": COMPARISON_COUNT,
            "source_attempt": "attempt_08",
            "no_optical_forward": True,
        },
    ).to_dict()
    target = _ensure_output(root, output_dir)
    heartbeat = HeartbeatWriter(target, RUN_ID, total_steps=8, interval_seconds=60.0)
    heartbeat.update(state=RUN_PREPARING, phase="PREPARING", step=0, latest_losses={"task": "FROZEN_INPUTS_VALIDATED"}, force=True)
    atomic_write_new_json(target / "source_manifest.json", source_manifest, artifact_kind="latent support compatibility source manifest")
    atomic_write_new_json(
        target / "execution_manifest.json",
        {
            "schema": "chen_bogt_latent_support_compatibility_execution_manifest.v1",
            "task_id": TASK_ID,
            "outer_gate": OUTER_GATE,
            "run_id": RUN_ID,
            "trajectory_id": TRAJECTORY_ID,
            "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
            "source_manifest_id": source_manifest["source_manifest_id"],
            "git_before": git_before,
            "source_decomposition_report_sha256": EXPECTED_DECOMPOSITION_REPORT_SHA256,
            "source_split_sha256": EXPECTED_DECOMPOSITION_SPLIT_SHA256,
            "source_bank_sha256": {
                "P1": EXPECTED_P1_BANK_SHA256,
                "P2": EXPECTED_P2_BANK_SHA256,
                "P3": EXPECTED_P3_BANK_SHA256,
            },
            "permutation": {"B": PERMUTATION_COUNT, "seed": PERMUTATION_SEED, "block_size": PERMUTATION_BLOCK_SIZE},
            "forbidden_paths": {"images": False, "labels": False, "DEV": False, "protected": False, "test": False, "sealed": False, "encoder": False, "generator": False, "cuda": False},
        },
        artifact_kind="latent support compatibility execution manifest",
    )
    heartbeat.update(state=RUNNING, phase="STATISTICS", step=1, latest_losses={"task": "ATTEMPT_08_PRIMARY_IDENTITY_REPRODUCED"}, force=True)

    nulls: dict[str, torch.Tensor] = {}

    def progress(done: int, total: int) -> None:
        heartbeat.update(state=RUNNING, phase="CALIBRATION", step=2, latest_losses={"permutations": f"{done}/{total}"})

    comparisons = {
        "REAL_REAL": (eval_a_h, eval_b_h),
        "P0_CURRENT_TN": (candidates["P0_CURRENT_TN"], eval_a_h),
        "P1_EMPIRICAL_INDEPENDENT": (candidates["P1_EMPIRICAL_INDEPENDENT"], eval_a_h),
        "P2_GAUSSIAN_COPULA": (candidates["P2_GAUSSIAN_COPULA"], eval_a_h),
        "P3_REAL_ROW_CONTROL": (candidates["P3_REAL_ROW_CONTROL"], eval_a_h),
    }
    for index, (name, (candidate, reference)) in enumerate(comparisons.items(), start=1):
        nulls[name] = _permutation_null(candidate, reference, seed=PERMUTATION_SEED, on_progress=progress)
        heartbeat.update(state=RUNNING, phase="CALIBRATION", step=2 + index, latest_losses={"task": f"{name}_NULL_COMPLETE"}, force=True)

    primary = {
        name: _null_summary(observed[name], nulls[name])
        for name in ("REAL_REAL", "P0_CURRENT_TN", "P1_EMPIRICAL_INDEPENDENT", "P2_GAUSSIAN_COPULA", "P3_REAL_ROW_CONTROL")
    }
    real_real_sane = bool(primary["REAL_REAL"]["compatible_by_q95"])
    p1_compatible = bool(primary["P1_EMPIRICAL_INDEPENDENT"]["compatible_by_q95"])
    p2_compatible = bool(primary["P2_GAUSSIAN_COPULA"]["compatible_by_q95"])
    p3_compatible = bool(primary["P3_REAL_ROW_CONTROL"]["compatible_by_q95"])
    if not real_real_sane:
        classification = "SELF_NULL_OR_REFERENCE_INVALID"
        selected = "NONE"
        suggested_next = "REPAIR_LATENT_SUPPORT_REFERENCE_CALIBRATION_V1"
    elif p1_compatible:
        classification = "MARGINAL_REPAIR_SUFFICIENT_FOR_PRIMARY_SUPPORT"
        selected = "P1_EMPIRICAL_MARGINAL_INDEPENDENT"
        suggested_next = "EXECUTE_CHEN_BOGT_REPAIRED_PRIOR_GENERATION_CANARY_V1"
    elif p2_compatible:
        classification = "GAUSSIAN_COPULA_MINIMAL_SUPPORT_REPAIR_SUPPORTED"
        selected = "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA"
        suggested_next = "EXECUTE_CHEN_BOGT_REPAIRED_PRIOR_GENERATION_CANARY_V1"
    else:
        classification = "GAUSSIAN_COPULA_RESIDUAL_SUPPORT_MISMATCH_SUPPORTED"
        selected = "NONE"
        suggested_next = "DEFINE_ONE_MINIMAL_RICHER_DEPENDENCE_PRODUCER_V1"
    selected_primary_key = {
        "P1_EMPIRICAL_MARGINAL_INDEPENDENT": "P1_EMPIRICAL_INDEPENDENT",
        "P2_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA": "P2_GAUSSIAN_COPULA",
    }.get(selected)
    control_status = "REAL_SUPPORT_CONTROL_COMPATIBLE" if p3_compatible else "REAL_SUPPORT_CONTROL_PRIMARY_REJECTION_WARNING"
    secondary_rows, secondary_warnings = _secondary_diagnostics(frozen["report"])
    heartbeat.update(state=RUNNING, phase="STATISTICS", step=7, latest_losses={"task": "PRIMARY_DECISION_RESOLVED"}, force=True)

    candidate_path: Path | None = None
    candidate_manifest_path: Path | None = None
    if selected != "NONE":
        selected_bank = frozen["p1"] if selected == "P1_EMPIRICAL_MARGINAL_INDEPENDENT" else frozen["p2"]
        candidate_path = target / "latent_prior_repair_candidate.v1.pt"
        candidate_payload = _candidate_payload(
            root,
            selected,
            selected_bank,
            primary[selected_primary_key],
            frozen["split"],
            str(source_manifest["source_manifest_id"]),
        )
        atomic_torch_save_new(candidate_path, candidate_payload, artifact_kind="latent prior repair candidate")
        candidate_manifest_path = target / "latent_prior_repair_candidate_manifest.v1.json"
        manifest = {
            "schema": CANDIDATE_MANIFEST_SCHEMA,
            "candidate_type": selected,
            "claim": "POST_TRAIN_SAMPLING_REPAIR_CANDIDATE",
            "candidate_artifact": _artifact_ref(root, candidate_path, role="selected repair candidate artifact"),
            "source_bank": _artifact_ref(root, root / (P1_BANK_PATH if selected.startswith("P1_") else P2_BANK_PATH), role="frozen selected support bank"),
            "source_decomposition_report": _artifact_ref(root, root / DECOMPOSITION_REPORT_PATH, role="frozen decomposition authority"),
            "source_manifest_id": source_manifest["source_manifest_id"],
            "fit_split_identity": {
                "split_manifest": _relative_path(root, root / DECOMPOSITION_SPLIT_PATH),
                "split_manifest_sha256": EXPECTED_DECOMPOSITION_SPLIT_SHA256,
                "fit_count": FIT_COUNT,
                "split_seed": decomposition.SPLIT_SEED,
            },
            "primary_compatibility": primary[selected_primary_key],
            "normalization_identity": "h(u)=log(u+1e-12), mean/std fit only on attempt_08 FIT rows",
            "sampling_semantics": "frozen attempt_08 P1/P2 bank; no refit and no resampling in this task",
            "training_boundary": "post-train sampling repair only; no BOGT retraining or promotion authority",
        }
        atomic_write_new_json(candidate_manifest_path, manifest, artifact_kind="latent prior repair candidate manifest")

    artifacts = _artifact_paths(root, target, candidate_path, candidate_manifest_path)
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
        "run_identity": {
            "schema": "research_run_identity.v1",
            "task_id": TASK_ID,
            "run_id": RUN_ID,
            "trajectory_id": TRAJECTORY_ID,
            "scientific_attempt_id": SCIENTIFIC_ATTEMPT_ID,
            "semantic_authority_id": SEMANTIC_AUTHORITY_ID,
            "data_role": "frozen_train_u_only",
            "source_manifest_id": source_manifest["source_manifest_id"],
            "source_decomposition_attempt": "attempt_08",
        },
        "source_decomposition": {
            "report": _artifact_ref(root, root / DECOMPOSITION_REPORT_PATH, role="frozen support decomposition authority"),
            "split_manifest": _artifact_ref(root, root / DECOMPOSITION_SPLIT_PATH, role="frozen FIT/EVAL split authority"),
            "P1": _artifact_ref(root, root / P1_BANK_PATH, role="frozen P1 bank authority"),
            "P2": _artifact_ref(root, root / P2_BANK_PATH, role="frozen P2 bank authority"),
            "P3": _artifact_ref(root, root / P3_BANK_PATH, role="frozen P3 control authority"),
            "source_manifest_id": frozen["report"]["source_manifest_id"],
        },
        "primary_statistic": {
            "name": "ENERGY_DISTANCE_IN_STANDARDIZED_LOG_U",
            "coordinate": "h(u)=log(u+1e-12)",
            "standardization": "mean/std fit only on 45,000 attempt_08 FIT rows; no candidate-specific normalization",
            "energy_distance": "2*mean(d(candidate,reference))-mean(d(candidate,candidate))-mean(d(reference,reference)); biased V-statistic with diagonal terms retained",
            "implementation": "scripts.research.execute_lightgen_2d_chen_bogt_sampled_generation_canary_v1._energy_distance",
        },
        "null_calibration": {
            "B": PERMUTATION_COUNT,
            "seed": PERMUTATION_SEED,
            "label_scheme": "deterministic permutation of pooled 2000 rows into 1000 vs 1000 labels",
            "output_independent": True,
            "comparisons": primary,
        },
        "real_real_sanity": {
            "comparison": "attempt_08 EVAL_A first 1000 rows vs EVAL_B first 1000 rows",
            "ED": observed["REAL_REAL"],
            "calibration": primary["REAL_REAL"],
            "status": "REAL_REAL_PRIMARY_SUPPORT_SANE" if real_real_sane else "SELF_NULL_CALIBRATION_INVALID",
        },
        "candidate_results": {
            "P0_CURRENT_TN": {"ED": observed["P0_CURRENT_TN"], "calibration": primary["P0_CURRENT_TN"], "classification": "P0_KNOWN_MISMATCH_REFERENCE" if not primary["P0_CURRENT_TN"]["compatible_by_q95"] else "P0_UNEXPECTEDLY_COMPATIBLE"},
            "P1_EMPIRICAL_INDEPENDENT": {"ED": observed["P1_EMPIRICAL_INDEPENDENT"], "calibration": primary["P1_EMPIRICAL_INDEPENDENT"], "classification": "P1_PRIMARY_SUPPORT_COMPATIBLE" if p1_compatible else "P1_PRIMARY_SUPPORT_MISMATCH"},
            "P2_GAUSSIAN_COPULA": {"ED": observed["P2_GAUSSIAN_COPULA"], "calibration": primary["P2_GAUSSIAN_COPULA"], "classification": "P2_PRIMARY_SUPPORT_COMPATIBLE" if p2_compatible else "P2_RESIDUAL_SUPPORT_MISMATCH"},
            "P3_REAL_ROW_CONTROL": {"ED": observed["P3_REAL_ROW_CONTROL"], "calibration": primary["P3_REAL_ROW_CONTROL"], "classification": control_status},
        },
        "secondary_diagnostics": {
            "source": "attempt_08 common-split metrics; not a hard gate",
            "rows": secondary_rows,
            "warnings": secondary_warnings,
        },
        "decision_semantics": {
            "old_selector": "strict Pareto descriptive selector",
            "old_selector_used": False,
            "new_selector": "primary finite-sample energy-distance compatibility; choose simpler P1 if compatible, otherwise P2 if compatible",
            "P3_role": "real support control only; never a producer candidate",
        },
        "primary_diagnosis": classification,
        "selected_minimal_candidate": selected,
        "candidate_artifact": None if candidate_path is None else candidate_path.name,
        "suggested_next_action": suggested_next,
        "next_action_authorized": False,
        "scientific_reads": {"frozen_train_u_only": True, "DEV": 0, "protected": 0, "test": 0, "sealed": 0, "labels": 0, "images": 0, "encoder_forward": 0, "generator_forward": 0, "optical_forward": 0, "cuda": 0},
        "historical_boundary": "P1/P2 remain post-train sampling repair candidates; historical BOGT objective remains TN_+(1,0.25)",
        "not_established": [
            "population-distribution equality",
            "generation success",
            "phase compatibility",
            "BOGT retraining alignment",
            "formal acceptance",
            "promotion",
        ],
        "artifacts": artifacts,
        "no_automatic_follow_up": True,
    }
    atomic_write_new_json(target / "latent_support_compatibility_report.json", report, artifact_kind="latent support compatibility report")
    heartbeat.update(state=RUN_COMPLETED, phase="COMPLETED", step=8, latest_losses={"task": "COMPLETED"}, force=True)
    return {
        "status": "VALID_COMPLETED",
        "outer_gate": OUTER_GATE,
        "primary_diagnosis": classification,
        "selected_candidate": selected,
        "report": _relative_path(root, target / "latent_support_compatibility_report.json"),
        "report_sha256": file_sha256(target / "latent_support_compatibility_report.json"),
        "source_manifest_id": source_manifest["source_manifest_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.output_dir), indent=2, sort_keys=True))
    except (CompatibilityError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "BLOCKED", "message": str(error)}, ensure_ascii=False, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
