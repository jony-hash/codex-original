"""Calibrate the frozen N3 mismatch-attribution decision rule.

This consumer is a CPU-only synthetic calibration harness.  It deliberately
does not load the project fit/validation banks, refit the frozen producer, or
invoke optics/evaluation.  Every synthetic bank goes through the same
fit-mean, centered-SVD, spectrum-standardization, and score-metric pipeline
used by the N3 attribution problem before a selector decision is made.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256


TASK_ID = "CALIBRATE_N3_ATTRIBUTION_DECISION_RULE_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_attribution_decision_rule_calibration_report.v1"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
SCRIPT_PATH = Path("scripts/research/calibrate_n3_attribution_decision_rule.py")
OUTPUT_DIR = Path(
    "outputs/20260908_v2/runs/"
    "lightgen_n3_attribution_decision_rule_calibration_v1_attempt_01"
)
REPORT_NAME = "n3_attribution_decision_rule_calibration_report.v1.json"

DIMENSION = 100
RANK = 99
FIT_COUNT = 500
VALIDATION_COUNT = 500
CLASS_COUNT = 10
ROWS_PER_CLASS = 50
REFERENCE_QUANTILE = 0.99
TOPK_DEPENDENCE_PAIRS = 20
NONLINEAR_PAIR_COUNT = 12
MIN_ACCEPTABLE_NULL_COUNT = 10
RECOVERY_RATE = 0.75
NULL_FPR_MAX = 0.10

MECHANISMS = (
    "NULL_SINGLE_GAUSSIAN",
    "PARTITION_OR_MOMENT_DRIFT",
    "RADIAL_SCALE_MIXTURE",
    "LOCAL_MIXTURE",
    "NONLINEAR_DEPENDENCE",
)

SELECTED = {
    "PARTITION_OR_MOMENT_DRIFT": "PARTITION_OR_MOMENT_DRIFT",
    "RADIAL_SCALE_MIXTURE": "RADIAL_SCALE_MIXTURE",
    "LOCAL_MIXTURE": "LOCAL_MIXTURE",
    "NONLINEAR_DEPENDENCE": "NONLINEAR_DEPENDENCE",
}
UNRESOLVED = "UNRESOLVED"

SEEDS = {
    "NULL_CALIBRATION": tuple(range(910001, 910021)),
    "NULL_EVALUATION": tuple(range(910101, 910121)),
    "PARTITION_OR_MOMENT_DRIFT": tuple(range(920001, 920011)),
    "RADIAL_SCALE_MIXTURE": tuple(range(930001, 930011)),
    "LOCAL_MIXTURE": tuple(range(940001, 940011)),
    "NONLINEAR_DEPENDENCE": tuple(range(950001, 950011)),
}

REAL_AUTHORITY_REFERENCES = {
    "OLS_PRODUCER_DATA_V1": {
        "path": "outputs/20260907_v2/runs/"
        "lightgen_ols_producer_data_authority_v1_attempt_01/"
        "ols_producer_source_manifest.v1.json",
        "sha256": "d07fc3fa927796e97254fa09a8c0eceb18d6555a99590db43217a685bf0d1687",
    },
    "N3_LOW_RANK_MODEL_ORDER_SELECTED": {
        "path": "outputs/20260907_v2/runs/"
        "lightgen_n3_low_rank_model_order_selection_v1_attempt_01/"
        "model_order_result.v1.json",
        "sha256": "aeb66fe308b42eb1c7ffa313790f762837df00f6255cefcec79a6210bce78f93",
    },
    "N3_AMPLITUDE_PRODUCER_BASELINE_FITTED": {
        "path": "outputs/20260907_v2/runs/"
        "lightgen_n3_amplitude_producer_baseline_v1_attempt_01/"
        "n3_amplitude_producer_state_manifest.v1.json",
        "sha256": "2f8766943eca79e37b4e72b4e1bff37cff7c8f9ce09469bce2daebbb481df373",
    },
    "N3_AMPLITUDE_PRODUCER_BASELINE_VALIDATED": {
        "path": "outputs/20260908_v2/runs/"
        "lightgen_n3_amplitude_producer_validation_v1_attempt_01/"
        "n3_amplitude_producer_validation_report.v1.json",
        "sha256": "16435dd72babf1f6abe537aa766c27fad79065f1cdf0ebfa6a9f54c4837aa6a0",
    },
    "N3_PRODUCER_MISMATCH_ATTRIBUTION": {
        "path": "outputs/20260908_v2/runs/"
        "lightgen_n3_producer_mismatch_attribution_v1_attempt_01/"
        "n3_producer_mismatch_attribution_report.v1.json",
        "sha256": "47d810b79d800df7c6a5a54337d10b95f43d87f905419e62137185a46003c457",
    },
    "N3_SCORE_GEOMETRY_ATTRIBUTION": {
        "path": "outputs/20260908_v2/runs/"
        "lightgen_n3_score_geometry_attribution_v1_attempt_02/"
        "n3_score_geometry_attribution_report.v1.json",
        "sha256": "7d64726110eac066d62258dd7d2e50c80304a4e73b200f0c211c34422c117f11",
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _float(value: torch.Tensor | float) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    result = float(value)
    _require(torch.isfinite(torch.tensor(result)), f"non-finite value: {value}")
    return result


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _normal(generator: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.randn(shape, generator=generator, dtype=torch.float64)


def _uniform(generator: torch.Generator, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.rand(shape, generator=generator, dtype=torch.float64)


def _groups() -> torch.Tensor:
    return torch.arange(FIT_COUNT, dtype=torch.long) // ROWS_PER_CLASS


def _local_component_means() -> torch.Tensor:
    means = torch.zeros((CLASS_COUNT, DIMENSION), dtype=torch.float64)
    amplitudes = torch.linspace(0.75, 5.25, CLASS_COUNT, dtype=torch.float64)
    for label in range(CLASS_COUNT):
        means[label, label] = amplitudes[label]
        means[label, CLASS_COUNT + label] = 0.35 * amplitudes[label]
    return means


def _draw_population(
    mechanism: str,
    count: int,
    generator: torch.Generator,
    bank: str,
    groups: torch.Tensor,
) -> torch.Tensor:
    _require(mechanism in MECHANISMS[1:], f"unknown non-null mechanism: {mechanism}")
    x = _normal(generator, (count, DIMENSION))

    if mechanism == "PARTITION_OR_MOMENT_DRIFT":
        if bank == "validation":
            x[:, :8] += 2.0
        return x

    if mechanism == "RADIAL_SCALE_MIXTURE":
        scale_selector = _uniform(generator, (count,))
        scale = torch.where(
            scale_selector < 0.82,
            torch.full((count,), 0.55, dtype=torch.float64),
            torch.full((count,), 2.50, dtype=torch.float64),
        )
        return x * scale.unsqueeze(1)

    if mechanism == "LOCAL_MIXTURE":
        return x + _local_component_means()[groups]

    if mechanism == "NONLINEAR_DEPENDENCE":
        for pair_index in range(NONLINEAR_PAIR_COUNT):
            left = 2 * pair_index
            right = left + 1
            latent_left = _normal(generator, (count,))
            latent_right = _normal(generator, (count,))
            x[:, left] = latent_left
            x[:, right] = latent_left * latent_right
        return x

    raise AssertionError(f"unhandled mechanism: {mechanism}")


def _draw_banks(mechanism: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = _generator(seed)
    fit_groups = _groups()
    validation_groups = _groups()
    if mechanism == "NULL_SINGLE_GAUSSIAN":
        fit = _normal(generator, (FIT_COUNT, DIMENSION))
        validation = _normal(generator, (VALIDATION_COUNT, DIMENSION))
    else:
        fit = _draw_population(mechanism, FIT_COUNT, generator, "fit", fit_groups)
        validation = _draw_population(
            mechanism,
            VALIDATION_COUNT,
            generator,
            "validation",
            validation_groups,
        )
    return fit, validation, fit_groups, validation_groups


def _fit_transform(
    fit: torch.Tensor,
    validation: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    fit_mean = fit.mean(dim=0)
    centered_fit = fit - fit_mean
    _, singular_values, right_vectors = torch.linalg.svd(
        centered_fit,
        full_matrices=False,
    )
    basis = right_vectors[:RANK].T.contiguous()
    spectrum = singular_values[:RANK].square() / (FIT_COUNT - 1)
    _require(torch.all(spectrum > 0), "synthetic fit spectrum is not positive")
    scale = spectrum.sqrt()
    z_fit = centered_fit @ basis / scale
    z_validation = (validation - fit_mean) @ basis / scale
    _require(z_fit.shape == (FIT_COUNT, RANK), "fit score shape changed")
    _require(z_validation.shape == (VALIDATION_COUNT, RANK), "validation score shape changed")
    return z_fit, z_validation, {
        "fit_mean_definition": "column mean over the 500-row fit bank",
        "centered_pca": "torch.linalg.svd(fit - fit_mean, full_matrices=False)",
        "rank": RANK,
        "spectrum_definition": "singular_values[:99]**2 / (n_fit - 1)",
        "score_definition": "(x - fit_mean) @ V[:99].T / sqrt(fit_spectrum)",
        "fit_covariance_denominator": "n_fit - 1",
        "dtype": "torch.float64",
    }


def _covariance(z: torch.Tensor) -> torch.Tensor:
    centered = z - z.mean(dim=0)
    return centered.T @ centered / (z.shape[0] - 1)


def _safe_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = left_centered.norm() * right_centered.norm()
    if _float(denominator) <= 1.0e-14:
        return 0.0
    return _float((left_centered @ right_centered) / denominator)


def _group_r2_means(z: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    r2 = z.square().sum(dim=1)
    values = []
    for label in range(CLASS_COUNT):
        values.append(r2[groups == label].mean())
    return torch.stack(values)


def _dependence_metrics(z: torch.Tensor) -> dict[str, float]:
    squared = z.square()
    centered = squared - squared.mean(dim=0)
    covariance = centered.T @ centered / (z.shape[0] - 1)
    standard_deviation = covariance.diag().clamp_min(1.0e-18).sqrt()
    correlation = covariance / (standard_deviation[:, None] * standard_deviation[None, :])
    off_diagonal = ~torch.eye(RANK, dtype=torch.bool)
    absolute_values = correlation[off_diagonal].abs()
    top_k = min(TOPK_DEPENDENCE_PAIRS, absolute_values.numel())
    top_values = torch.topk(absolute_values, k=top_k).values
    total = absolute_values.sum().clamp_min(1.0e-18)
    return {
        "square_correlation_max_abs": _float(absolute_values.max()),
        "square_correlation_topk_abs_share": _float(top_values.sum() / total),
        "square_correlation_pair_observation_count": float(absolute_values.numel()),
        "square_correlation_topk": float(top_k),
    }


def _replicate_metrics(
    mechanism: str,
    seed: int,
) -> dict[str, Any]:
    fit, validation, fit_groups, validation_groups = _draw_banks(mechanism, seed)
    z_fit, z_validation, pipeline = _fit_transform(fit, validation)
    fit_covariance = _covariance(z_fit)
    validation_covariance = _covariance(z_validation)
    fit_r2 = z_fit.square().sum(dim=1)
    validation_r2 = z_validation.square().sum(dim=1)
    fit_group_r2 = _group_r2_means(z_fit, fit_groups)
    validation_group_r2 = _group_r2_means(z_validation, validation_groups)
    fit_group_std = fit_group_r2.std(unbiased=True)
    validation_group_std = validation_group_r2.std(unbiased=True)
    group_profile_correlation = _safe_correlation(fit_group_r2, validation_group_r2)

    metrics = {
        "mean_l2_fit_to_validation": _float(
            z_fit.mean(dim=0).sub(z_validation.mean(dim=0)).norm()
        ),
        "validation_covariance_trace_abs_from_rank": _float(
            (validation_covariance.trace() - RANK).abs()
        ),
        "validation_radial_r2_mean": _float(validation_r2.mean()),
        "validation_radial_r2_variance": _float(validation_r2.var(unbiased=True)),
        "validation_radial_r2_q95": _float(torch.quantile(validation_r2, 0.95)),
        "fit_radial_r2_mean": _float(fit_r2.mean()),
        "fit_radial_r2_variance": _float(fit_r2.var(unbiased=True)),
        "fit_radial_r2_q95": _float(torch.quantile(fit_r2, 0.95)),
        "local_group_r2_std_fit": _float(fit_group_std),
        "local_group_r2_std_validation": _float(validation_group_std),
        "local_group_profile_correlation": group_profile_correlation,
        "dependence": _dependence_metrics(z_validation),
        "fit_covariance_identity_frobenius": _float(
            (fit_covariance - torch.eye(RANK, dtype=torch.float64)).norm()
        ),
        "validation_covariance_identity_frobenius": _float(
            (validation_covariance - torch.eye(RANK, dtype=torch.float64)).norm()
        ),
    }
    _require(
        metrics["fit_covariance_identity_frobenius"] < 1.0e-8,
        "fit PCA standardization did not produce the expected identity covariance",
    )
    return {
        "truth": mechanism,
        "seed": seed,
        "n_fit": FIT_COUNT,
        "n_validation": VALIDATION_COUNT,
        "d_input": DIMENSION,
        "rank": RANK,
        "pipeline": pipeline,
        "metrics": metrics,
    }


def _metric_value(record: Mapping[str, Any], path: tuple[str, ...]) -> float:
    value: Any = record["metrics"]
    for key in path:
        value = value[key]
    return float(value)


REFERENCE_METRICS = {
    "mean_l2_fit_to_validation": ("mean_l2_fit_to_validation",),
    "validation_covariance_trace_abs_from_rank": (
        "validation_covariance_trace_abs_from_rank",
    ),
    "validation_radial_r2_variance": ("validation_radial_r2_variance",),
    "validation_radial_r2_q95": ("validation_radial_r2_q95",),
    "local_group_r2_std_validation": ("local_group_r2_std_validation",),
    "local_group_profile_correlation": ("local_group_profile_correlation",),
    "dependence_square_correlation_max_abs": (
        "dependence",
        "square_correlation_max_abs",
    ),
    "dependence_square_correlation_topk_abs_share": (
        "dependence",
        "square_correlation_topk_abs_share",
    ),
}


def _reference_thresholds(null_records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    records = list(null_records)
    _require(
        len(records) >= MIN_ACCEPTABLE_NULL_COUNT,
        "too few null calibration replications for a reference distribution",
    )
    thresholds: dict[str, float] = {}
    for name, path in REFERENCE_METRICS.items():
        values = torch.tensor(
            [_metric_value(record, path) for record in records],
            dtype=torch.float64,
        )
        thresholds[name] = _float(torch.quantile(values, REFERENCE_QUANTILE))
    return thresholds


def _triggers(record: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, bool]:
    metrics = record["metrics"]
    dependence = metrics["dependence"]
    return {
        "partition_or_moment_drift": (
            float(metrics["mean_l2_fit_to_validation"])
            > thresholds["mean_l2_fit_to_validation"]
        ),
        "radial_scale_mixture": (
            float(metrics["validation_radial_r2_variance"])
            > thresholds["validation_radial_r2_variance"]
            or float(metrics["validation_radial_r2_q95"])
            > thresholds["validation_radial_r2_q95"]
        ),
        "local_mixture": (
            float(metrics["local_group_r2_std_validation"])
            > thresholds["local_group_r2_std_validation"]
            and float(metrics["local_group_profile_correlation"])
            > thresholds["local_group_profile_correlation"]
        ),
        "nonlinear_dependence": (
            float(dependence["square_correlation_max_abs"])
            > thresholds["dependence_square_correlation_max_abs"]
            and float(dependence["square_correlation_topk_abs_share"])
            > thresholds["dependence_square_correlation_topk_abs_share"]
        ),
    }


def _select(record: Mapping[str, Any], thresholds: Mapping[str, float]) -> dict[str, Any]:
    triggers = _triggers(record, thresholds)
    active = [name for name, active in triggers.items() if active]
    partition = triggers["partition_or_moment_drift"]
    local = triggers["local_mixture"]
    dependence = triggers["nonlinear_dependence"]

    # Radial elevation is tolerated as a consequence of a detected mean drift
    # or reproducible local mixture.  Independent local/dependence evidence is
    # otherwise ambiguous and must fail closed rather than choose a family.
    if partition and (local or dependence):
        selected = UNRESOLVED
        rule = "independent partition and local/dependence triggers overlap"
    elif local and dependence:
        selected = UNRESOLVED
        rule = "local and nonlinear-dependence triggers overlap"
    elif partition:
        selected = SELECTED["PARTITION_OR_MOMENT_DRIFT"]
        rule = "calibrated fit-to-validation mean displacement exceeds null reference"
    elif local:
        selected = SELECTED["LOCAL_MIXTURE"]
        rule = "calibrated reproducible local profile exceeds null reference"
    elif dependence:
        selected = SELECTED["NONLINEAR_DEPENDENCE"]
        rule = "calibrated global squared-dependence concentration exceeds null reference"
    elif triggers["radial_scale_mixture"]:
        selected = SELECTED["RADIAL_SCALE_MIXTURE"]
        rule = "calibrated radial variance or tail exceeds null reference"
    else:
        selected = UNRESOLVED
        rule = "no calibrated mechanism trigger"

    return {
        "selected": selected,
        "triggers": triggers,
        "active_trigger_names": active,
        "rule": rule,
    }


def _with_selection(
    records: Iterable[Mapping[str, Any]],
    thresholds: Mapping[str, float],
) -> list[dict[str, Any]]:
    selected_records: list[dict[str, Any]] = []
    for record in records:
        result = dict(record)
        result["selection"] = _select(record, thresholds)
        selected_records.append(result)
    return selected_records


def _counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        selected = str(record["selection"]["selected"])
        counts[selected] = counts.get(selected, 0) + 1
    return dict(sorted(counts.items()))


def _summarize_truth(
    truth: str,
    records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    count = len(records)
    counts = _counts(records)
    expected = SELECTED.get(truth)
    correct = counts.get(expected, 0) if expected is not None else counts.get(UNRESOLVED, 0)
    unresolved = counts.get(UNRESOLVED, 0)
    wrong = count - correct - unresolved
    recovery_rate = correct / count if count else 0.0
    unresolved_rate = unresolved / count if count else 0.0
    if expected is None:
        status = "NULL_CONTROL"
    elif wrong == 0 and recovery_rate >= RECOVERY_RATE:
        status = "RECOVERED"
    elif wrong == 0 and unresolved_rate >= RECOVERY_RATE:
        status = "CONSERVATIVE_UNRESOLVED"
    else:
        status = "MISCLASSIFIED_OR_UNSTABLE"
    return {
        "truth": truth,
        "count": count,
        "selected_counts": counts,
        "expected_selection": expected,
        "correct_count": correct,
        "unresolved_count": unresolved,
        "wrong_count": wrong,
        "recovery_rate": recovery_rate,
        "unresolved_rate": unresolved_rate,
        "status": status,
    }


def _git_state(repo_root: Path) -> dict[str, Any]:
    branch = subprocess.run(
        ["git", "-C", str(repo_root), "branch", "--show-current"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_lines = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    dirty_paths = [line[3:] if len(line) >= 4 else line for line in status_lines]
    return {"branch": branch, "head": head, "dirty_paths": dirty_paths}


def _reference_identity(repo_root: Path) -> dict[str, Any]:
    identity: dict[str, Any] = {}
    for name, expected in REAL_AUTHORITY_REFERENCES.items():
        path = repo_root / expected["path"]
        _require(path.is_file(), f"reference authority is missing: {path}")
        actual = file_sha256(path)
        _require(actual == expected["sha256"], f"reference authority hash changed: {name}")
        identity[name] = {
            "path": expected["path"],
            "sha256": actual,
            "used_for_synthetic_thresholds": False,
            "used_for_metric_computation": False,
        }
    return identity


def _script_audit(repo_root: Path) -> dict[str, Any]:
    selector_path = repo_root / "scripts/research/attribute_n3_producer_mismatch.py"
    _require(selector_path.is_file(), "current attribution selector is missing")
    return {
        "path": selector_path.relative_to(repo_root).as_posix(),
        "sha256": file_sha256(selector_path),
        "audited_structural_problems": [
            "fit PCA/standardization makes fit covariance approximately identity by construction",
            "class-centering branch requires strict positive fit and validation reductions",
            "dependence branch uses sample correlation > 0.0 without a finite-sample reference",
            "radial branch uses exact radial moments != rank and != 2*rank",
        ],
        "calibration_change_boundary": "new calibrated synthetic consumer only; historical selector and reports remain unchanged",
    }


def _write_report(repo_root: Path, report: Mapping[str, Any]) -> Path:
    output_dir = repo_root / OUTPUT_DIR
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once calibration output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        target = temporary / REPORT_NAME
        atomic_write_new_json(
            target,
            report,
            artifact_kind="N3 attribution decision-rule calibration report",
        )
        _require(sorted(path.name for path in temporary.iterdir()) == [REPORT_NAME], "calibration output layout changed")
        if output_dir.exists():
            raise FileExistsError(f"calibration output appeared: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / REPORT_NAME


def _classify(
    null_summary: Mapping[str, Any],
    mechanism_summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    null_false_positive_rate = float(null_summary["false_positive_rate"])
    partition_status = mechanism_summaries["PARTITION_OR_MOMENT_DRIFT"]["status"]
    control_statuses = [
        mechanism_summaries[name]["status"]
        for name in (
            "RADIAL_SCALE_MIXTURE",
            "LOCAL_MIXTURE",
            "NONLINEAR_DEPENDENCE",
        )
    ]
    recovered_controls = sum(status == "RECOVERED" for status in control_statuses)
    controls_valid = all(
        status in {"RECOVERED", "CONSERVATIVE_UNRESOLVED"}
        for status in control_statuses
    )
    null_ok = null_false_positive_rate <= NULL_FPR_MAX
    partition_ok = partition_status == "RECOVERED"

    if null_ok and partition_ok and controls_valid and recovered_controls >= 1:
        return (
            "N3_ATTRIBUTION_SELECTOR_CALIBRATED",
            "null false-positive control passed; partition drift recovered; every radial/local/dependence control was recovered or fail-closed unresolved",
        )
    if (
        null_ok
        and partition_ok
        and controls_valid
        and recovered_controls == 0
    ):
        return (
            "N3_ATTRIBUTION_SELECTOR_NOT_DISCRIMINATIVE",
            "null and partition controls passed, but all non-partition mechanisms remained unresolved",
        )
    if null_ok and not any(
        status == "MISCLASSIFIED_OR_UNSTABLE"
        for status in (partition_status, *control_statuses)
    ):
        return (
            "N3_ATTRIBUTION_SELECTOR_NOT_DISCRIMINATIVE",
            "null control passed but partition or mechanism recovery did not meet the registered discrimination requirement",
        )
    return (
        "N3_ATTRIBUTION_SELECTOR_CALIBRATION_MIXED",
        "one or more null, partition, or mechanism controls were misclassified or unstable",
    )


def execute(repo_root: Path) -> Path:
    torch.set_num_threads(1)
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 authority branch")
    allowed_dirty = {SCRIPT_PATH.as_posix()}
    unexpected_dirty = [path for path in git["dirty_paths"] if path not in allowed_dirty]
    _require(
        not unexpected_dirty,
        f"unexpected dirty paths before calibration: {unexpected_dirty}",
    )
    references = _reference_identity(repo_root)
    audit = _script_audit(repo_root)
    script_sha = file_sha256(repo_root / SCRIPT_PATH)

    null_calibration = [
        _replicate_metrics("NULL_SINGLE_GAUSSIAN", seed)
        for seed in SEEDS["NULL_CALIBRATION"]
    ]
    thresholds = _reference_thresholds(null_calibration)
    null_evaluation = [
        _replicate_metrics("NULL_SINGLE_GAUSSIAN", seed)
        for seed in SEEDS["NULL_EVALUATION"]
    ]
    selected_null_evaluation = _with_selection(null_evaluation, thresholds)
    null_summary = _summarize_truth("NULL_SINGLE_GAUSSIAN", selected_null_evaluation)
    null_summary = {
        **null_summary,
        "false_positive_count": null_summary["count"] - null_summary["unresolved_count"],
        "false_positive_rate": (
            (null_summary["count"] - null_summary["unresolved_count"])
            / null_summary["count"]
        ),
    }

    mechanism_records: dict[str, list[dict[str, Any]]] = {}
    mechanism_summaries: dict[str, dict[str, Any]] = {}
    confusion: list[dict[str, Any]] = [null_summary]
    for mechanism in MECHANISMS[1:]:
        records = [
            _replicate_metrics(mechanism, seed)
            for seed in SEEDS[mechanism]
        ]
        selected_records = _with_selection(records, thresholds)
        mechanism_records[mechanism] = selected_records
        summary = _summarize_truth(mechanism, selected_records)
        mechanism_summaries[mechanism] = summary
        confusion.append(summary)

    classification, classification_reason = _classify(null_summary, mechanism_summaries)
    report = {
        "schema": REPORT_SCHEMA,
        "task_id": TASK_ID,
        "gate": GATE,
        "classification": classification,
        "classification_reason": classification_reason,
        "local_state": {
            "branch": git["branch"],
            "head_before_artifact": git["head"],
            "dirty_paths_before_artifact": git["dirty_paths"],
            "script_path": SCRIPT_PATH.as_posix(),
            "script_sha256": script_sha,
        },
        "selector_audit": audit,
        "synthetic_pipeline": {
            "dimension_input": DIMENSION,
            "n_fit": FIT_COUNT,
            "n_validation": VALIDATION_COUNT,
            "rank": RANK,
            "class_count_for_observer_groups": CLASS_COUNT,
            "rows_per_observer_group": ROWS_PER_CLASS,
            "complete_pipeline": [
                "synthetic population",
                "independent fit draw",
                "independent validation draw",
                "fit mean estimation",
                "centered fit PCA/SVD",
                "fit spectrum/scale estimation",
                "fit and validation score transformation",
                "finite-sample attribution metrics",
                "calibrated selector",
            ],
            "observer_group_boundary": "synthetic component/group identities are diagnostic truth only and are not producer inputs",
            "dtype": "torch.float64",
            "ambient_rng_used": False,
        },
        "seed_sets": {name: list(values) for name, values in SEEDS.items()},
        "registered_mechanism_parameters": {
            "PARTITION_OR_MOMENT_DRIFT": "validation-only +2.0 shift in the first 8 source coordinates",
            "RADIAL_SCALE_MIXTURE": "row-wise common scale 0.55 with probability 0.82 and 2.50 otherwise",
            "LOCAL_MIXTURE": "10 reproducible observer groups with component means of amplitudes 0.75..5.25 in paired coordinates",
            "NONLINEAR_DEPENDENCE": f"{NONLINEAR_PAIR_COUNT} source-coordinate pairs with right coordinate = left latent * independent latent",
        },
        "null_finite_sample_reference": {
            "calibration_count": len(null_calibration),
            "evaluation_count": len(null_evaluation),
            "quantile_probability": REFERENCE_QUANTILE,
            "thresholds": thresholds,
            "calibration_records": null_calibration,
            "evaluation_summary": null_summary,
            "evaluation_records": selected_null_evaluation,
        },
        "mechanism_recovery": {
            "summaries": mechanism_summaries,
            "records": mechanism_records,
            "confusion_table": confusion,
        },
        "calibrated_decision_rule": {
            "threshold_source": "NULL_SINGLE_GAUSSIAN calibration replications only",
            "partition": "mean_l2_fit_to_validation > null q99",
            "radial": "validation radial variance OR q95 > respective null q99",
            "local": "validation local-group r2 standard deviation AND fit/validation group-profile correlation > respective null q99",
            "dependence": "validation squared-correlation max AND top-k absolute-share > respective null q99",
            "ambiguity": "independent partition+local/dependence or local+dependence triggers return UNRESOLVED",
            "fallback": "no trigger returns UNRESOLVED",
            "exact_zero_or_equality_logic_removed": True,
            "multiple_pair_observation_control": "global top-k share plus maximum, both calibrated against NULL; no arbitrary nonzero pair is sufficient",
            "fit_pca_class_centering_improvement_required": False,
        },
        "real_data_boundary": {
            "authority_references": references,
            "project_fit_validation_banks_loaded": False,
            "real_data_used_for_threshold_selection": False,
            "development_data_attribution_executed": False,
            "historical_reports_modified": False,
            "producer_refit": False,
            "rank_reselected": False,
            "optical_forward_or_evaluator": False,
        },
        "claims_boundary": {
            "calibration_is_synthetic_only": True,
            "calibration_does_not_support_a_new_producer_family": True,
            "gaussian_or_non_gaussian_project_data_claim": False,
            "next_family_selected": False,
            "real_n3_failure_reclassified": False,
        },
        "not_executed": [
            "real-data calibrated attribution",
            "producer fitting or rank selection",
            "generation, optical forward, or evaluator",
            "CUDA, optimizer, backward, or training",
        ],
    }
    output_path = _write_report(repo_root, report)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    output_path = execute(args.repo_root.resolve())
    print(json.dumps({"report": output_path.as_posix()}, indent=2))


if __name__ == "__main__":
    main()
