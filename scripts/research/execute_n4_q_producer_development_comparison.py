"""Execute the frozen N4 q-producer comparison on N3 Development Bank V2.

The consumer fits only the 4000-row fitting side of each frozen fold, creates
one preregistered 200-row model bank per candidate/fold, and compares that
bank with the exact 1000-row held-out side using the V2 raw-q metric adapter.
It never reads sealed confirmation values, invokes image/optical evaluation,
or changes training and promotion state.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

from ldm_project.v2.evidence import (
    atomic_torch_save_new,
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import (
    compare_n3_minimal_competing_producer_baselines_v1 as baseline,
    define_n3_development_bank_v2 as source_definition,
    define_n4_q_producer_candidate_registry as registry_definition,
    execute_n3_development_bank_v2_empirical_ceiling as v2_ceiling,
    execute_n3_empirical_marginal_gaussian_copula_producer_v1 as copula,
)


TASK_ID = "N4_Q_PRODUCER_DEVELOPMENT_COMPARISON"
GATE = TASK_ID
REPORT_SCHEMA = "n4_q_producer_comparison_report.v1"
FIT_MANIFEST_SCHEMA = "n4_q_producer_fit_manifest.v1"
FOLD_COMPATIBILITY_SCHEMA = "n4_q_producer_fold_compatibility.v1"
LEDGER_SCHEMA = "n4_q_producer_evaluator_exposure_ledger.v1"
LINEAGE_SCHEMA = "n4_q_producer_lineage_manifest.v1"
FITTED_ARTIFACT_SCHEMA = "n4_q_producer_fitted_producer.v1"
MODEL_ARTIFACT_SCHEMA = "n4_q_producer_sampled_model_q.v1"
EXPECTED_BRANCH = source_definition.EXPECTED_BRANCH
EXPECTED_STARTING_HEAD = source_definition.EXPECTED_STARTING_HEAD

REGISTRY_PATH = registry_definition.REGISTRY_OUTPUT_DIR / registry_definition.REGISTRY_NAME
OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n4_q_producer_development_comparison_v1_attempt_02"
)
FIT_MANIFEST_NAME = "n4_q_producer_fit_manifest.v1.json"
FOLD_COMPATIBILITY_NAME = "n4_q_producer_fold_compatibility.v1.json"
REPORT_NAME = "n4_q_producer_comparison_report.v1.json"
LEDGER_NAME = "n4_q_producer_evaluator_exposure_ledger.v1.json"
LINEAGE_NAME = "n4_q_producer_lineage_manifest.v1.json"

SOURCE_MANIFEST = v2_ceiling.SOURCE_MANIFEST
Q_ARTIFACT = v2_ceiling.Q_ARTIFACT
FOLD_MANIFEST = v2_ceiling.FOLD_MANIFEST
CEILING_REPORT = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_empirical_ceiling_v1_attempt_01/"
    "n3_development_bank_v2_empirical_ceiling_report.v1.json"
)
GOVERNANCE_STATE = Path("docs/status/research_governance_state.json")
METRIC_IMPLEMENTATION = Path("scripts/research/execute_n3_development_bank_v2_empirical_ceiling.py")

AUTHORITY_SHA256 = {
    SOURCE_MANIFEST.as_posix(): v2_ceiling.SOURCE_MANIFEST_SHA256,
    Q_ARTIFACT.as_posix(): v2_ceiling.Q_ARTIFACT_SHA256,
    FOLD_MANIFEST.as_posix(): v2_ceiling.FOLD_MANIFEST_SHA256,
    CEILING_REPORT.as_posix(): "731cf826dc081e7cc5629790ba2fb74946f7cc41f1a7f0035bf3a3ca96a31826",
}
SEALED_CONFIRMATION_PATH = baseline.SEALED_CONFIRMATION_MANIFEST
SEALED_CONFIRMATION_SHA256 = baseline.EXPECTED_SEALED_CONFIRMATION_MANIFEST_SHA256

CLASS_COUNT = 10
DIMENSION = 100
POOL_COUNT = 5000
FOLD_COUNT = 5
FIT_COUNT_PER_FOLD = 4000
HELDOUT_COUNT_PER_FOLD = 1000
FIT_PER_CLASS_PER_FOLD = 400
HELDOUT_PER_CLASS_PER_FOLD = 100
MODEL_SAMPLE_COUNT = 200
MAX_EVALUATOR_CALLS = 15
Q95 = 0.95
SCALAR_METRICS = tuple(v2_ceiling.SCALAR_METRICS)
JOINT_METRICS = tuple(v2_ceiling.JOINT_METRICS)
MARGINAL_METRICS = tuple(v2_ceiling.MARGINAL_METRICS)
EXPECTED_CANDIDATE_IDS = (
    "SHRINKAGE_ELLIPTICAL_GAUSSIAN_AMPLITUDE_V1",
    "RADIAL_SCALE_MIXTURE_AMPLITUDE_V1",
    "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1",
)

CLASSIFICATION_NO_SUPPORTED = "N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED"
CLASSIFICATION_SINGLE_SUPPORTED = "N4_Q_PRODUCER_COMPARISON_SINGLE_SUPPORTED"
CLASSIFICATION_MULTIPLE_SUPPORTED = "N4_Q_PRODUCER_COMPARISON_MULTIPLE_SUPPORTED"
CLASSIFICATION_INVALID = "N4_Q_PRODUCER_COMPARISON_INVALID"


class InvalidHarnessError(RuntimeError):
    """A frozen N4 boundary or evaluator invariant failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _invalid(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidHarnessError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> list[Any]:
    _require(isinstance(value, list), f"{name} must be a list")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


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
    status = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "branch": branch,
        "head": head,
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in status],
        "worktree_clean": not status,
    }


def _read_json_exact(repo_root: Path, path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    target = repo_root / path
    _invalid(target.is_file(), f"{name} is missing: {target}")
    _invalid(file_sha256(target) == expected_sha256, f"{name} hash changed: {target}")
    return dict(_mapping(json.loads(target.read_text(encoding="utf-8")), name))


def _ref(repo_root: Path, path: Path, *, role: str, expected_sha256: str | None = None) -> dict[str, Any]:
    target = repo_root / path
    _invalid(target.is_file(), f"{role} is missing: {target}")
    actual = file_sha256(target)
    if expected_sha256 is not None:
        _invalid(actual == expected_sha256, f"{role} hash changed: {target}")
    return {"path": _relative(repo_root, target), "sha256": actual, "role": role}


def validate_authority_hashes(repo_root: Path, expected: Mapping[str, str]) -> None:
    """Fail closed when any declared authority is missing or has a new digest."""

    for path_text, expected_sha256 in expected.items():
        target = repo_root / Path(path_text)
        _invalid(target.is_file(), f"authority is missing: {target}")
        _invalid(file_sha256(target) == expected_sha256, f"authority hash changed: {target}")


def validate_sealed_boundary(boundary: Mapping[str, Any]) -> None:
    _invalid(boundary.get("status") == "RESERVED_NOT_MATERIALIZED", "sealed confirmation is not reserved")
    _invalid(boundary.get("consumed", False) is False, "sealed confirmation is consumed")
    _invalid(boundary.get("materialized", False) is False, "sealed confirmation is materialized")
    _invalid(boundary.get("sample_values_loaded", False) is False, "sealed confirmation values are loaded")


def validate_fit_holdout_boundary(fit_positions: torch.Tensor, heldout_positions: torch.Tensor) -> None:
    fit = fit_positions.to(dtype=torch.int64).reshape(-1)
    heldout = heldout_positions.to(dtype=torch.int64).reshape(-1)
    _invalid(torch.unique(fit).numel() == fit.numel(), "fit positions contain duplicates")
    _invalid(torch.unique(heldout).numel() == heldout.numel(), "heldout positions contain duplicates")
    _invalid(not bool(torch.isin(fit, heldout).any()), "fit and heldout positions overlap")


def validate_fold_partition(fold_ids: torch.Tensor) -> None:
    values = fold_ids.to(dtype=torch.int64).reshape(-1)
    _invalid(tuple(values.shape) == (POOL_COUNT,), "fold assignment shape changed")
    _invalid(torch.equal(torch.sort(values).values, torch.tensor([fold for fold in range(FOLD_COUNT) for _ in range(POOL_COUNT // FOLD_COUNT)], dtype=torch.int64)), "fold assignment counts changed")
    for fold in range(FOLD_COUNT):
        heldout = (values == fold).nonzero(as_tuple=False).reshape(-1)
        fit = (values != fold).nonzero(as_tuple=False).reshape(-1)
        validate_fit_holdout_boundary(fit, heldout)
        _invalid(fit.numel() == FIT_COUNT_PER_FOLD, f"fold {fold} fit count changed")
        _invalid(heldout.numel() == HELDOUT_COUNT_PER_FOLD, f"fold {fold} heldout count changed")


def validate_sampling_record(record: Mapping[str, Any]) -> None:
    _invalid(record.get("best_of_n", False) is False, "best-of-N sampling is enabled")
    _invalid(record.get("seed_retry", False) is False, "sampling seed retry is enabled")
    _invalid(record.get("heldout_seed_tuning", False) is False, "heldout seed tuning is enabled")
    _invalid(record.get("sample_count") == MODEL_SAMPLE_COUNT, "model sample count changed")


def validate_runtime_snapshot(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    _invalid(dict(before) == dict(after), "frozen registry, implementation, authority, or configuration changed after evaluation")


def validate_registry_frozen(before_sha256: str, after_sha256: str) -> None:
    _invalid(before_sha256 == after_sha256, "candidate registry changed after freeze")


def validate_exposure_ledger(
    records: list[Mapping[str, Any]],
    candidate_ids: tuple[str, ...],
    *,
    max_calls: int = MAX_EVALUATOR_CALLS,
) -> None:
    _invalid(len(records) <= max_calls, "evaluator call count exceeded the preregistered cap")
    keys: set[tuple[str, int]] = set()
    call_numbers: list[int] = []
    for record in records:
        candidate_id = str(record.get("candidate_id"))
        fold = int(record.get("fold"))
        key = (candidate_id, fold)
        _invalid(candidate_id in candidate_ids, f"unregistered evaluator candidate: {candidate_id}")
        _invalid(0 <= fold < FOLD_COUNT, f"evaluator fold is outside the frozen domain: {fold}")
        _invalid(key not in keys, f"duplicate evaluator exposure: {key}")
        keys.add(key)
        call_numbers.append(int(record.get("evaluator_call_number")))
        _invalid(record.get("result_caused_code_or_config_change") is False, "adaptive development exposure was recorded")
    _invalid(call_numbers == list(range(1, len(records) + 1)), "evaluator call numbers are not contiguous")


def validate_final_governance(governance: Mapping[str, Any]) -> None:
    expected = {
        "selected_producer": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_consumed": False,
    }
    for key, value in expected.items():
        _invalid(governance.get(key) == value, f"N4 governance boundary changed: {key}")


def _validate_current_governance(repo_root: Path) -> dict[str, Any]:
    state_path = repo_root / GOVERNANCE_STATE
    _invalid(state_path.is_file(), "research governance state is missing")
    state = _mapping(json.loads(state_path.read_text(encoding="utf-8")), "research governance state")
    _invalid(state.get("active_objective") == "N3_DEVELOPMENT_BANK_V2_VALIDATED", "V2 development authority is not the active objective")
    aliases = _mapping(state.get("evidence_aliases"), "governance evidence aliases")
    required = (
        "N3_DEVELOPMENT_BANK_V2_SOURCE",
        "N3_DEVELOPMENT_BANK_V2_Q",
        "N3_DEVELOPMENT_BANK_V2_FIVE_FOLD_PROTOCOL",
        "N3_DEVELOPMENT_BANK_V2_EMPIRICAL_CEILING",
    )
    for name in required:
        item = _mapping(aliases.get(name), name)
        for key, expected in (
            ("sealed_confirmation_consumed", False),
            ("training_authorized", False),
            ("formal_promotion", False),
            ("generation_readiness", "not_ready"),
        ):
            if key in item:
                _invalid(item.get(key) == expected, f"governance alias changed: {name}.{key}")
        if "selected_producer" in item:
            _invalid(item.get("selected_producer") is None, f"governance alias selected producer changed: {name}")
    _invalid(aliases["N3_DEVELOPMENT_BANK_V2_EMPIRICAL_CEILING"].get("classification") == "N3_DEVELOPMENT_BANK_V2_VALIDATED", "V2 ceiling status changed")
    return {
        "path": _relative(repo_root, state_path),
        "sha256": file_sha256(state_path),
        "active_objective": state["active_objective"],
        "authorization_tier": state.get("authorization_tier"),
        "sealed_confirmation_consumed": False,
        "selected_producer": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }


def _validate_ceiling_authority(repo_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = _read_json_exact(repo_root, CEILING_REPORT, AUTHORITY_SHA256[CEILING_REPORT.as_posix()], "V2 empirical ceiling report")
    _invalid(report.get("schema") == "n3_development_bank_v2_empirical_ceiling_report.v1", "V2 ceiling schema changed")
    _invalid(report.get("status") == "valid_completed" and report.get("result") == "VALID_COMPLETED", "V2 ceiling is incomplete")
    _invalid(report.get("classification") == "N3_DEVELOPMENT_BANK_V2_VALIDATED", "V2 ceiling classification changed")
    aggregate = _mapping(report.get("aggregate"), "V2 ceiling aggregate")
    for key, expected in (
        ("fold_count", FOLD_COUNT),
        ("compatible_fold_count", FOLD_COUNT),
        ("compatible_fold_pattern", [True] * FOLD_COUNT),
        ("all_five_folds_required", True),
        ("operationally_valid", True),
        ("development_supported", True),
        ("producer_evaluated", False),
        ("producer_selection", False),
    ):
        _invalid(aggregate.get(key) == expected, f"V2 ceiling aggregate changed: {key}")
    settings = _mapping(report.get("execution_settings"), "V2 ceiling execution settings")
    expected_settings = {
        "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
        "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
        "fit_per_class_per_fold": FIT_PER_CLASS_PER_FOLD,
        "heldout_per_class_per_fold": HELDOUT_PER_CLASS_PER_FOLD,
        "fold_count": FOLD_COUNT,
        "generated_bank_size": MODEL_SAMPLE_COUNT,
        "data_vs_model_replicates": 4,
        "model_vs_model_replicates": 8,
        "model_reference_pairs": [[0, 1], [2, 3], [4, 5], [6, 7]],
        "compatibility_q": Q95,
        "complete_row_sampling": True,
        "with_replacement": True,
        "independent_dimension_sampling": False,
        "seed_retry": False,
        "best_bank_selection": False,
        "samplewise_normalization": False,
        "raw_q_energy_distance": True,
        "guardrails": list(SCALAR_METRICS),
        "dtype": "float64 q; int64 row indices",
    }
    for key, expected in expected_settings.items():
        _invalid(settings.get(key) == expected, f"V2 ceiling setting changed: {key}")
    _invalid("equal-row-count precondition removed" in str(settings.get("unequal_sample_shape_adapter")), "V2 unequal-sample adapter changed")
    empirical = _mapping(report.get("empirical_ceiling"), "V2 empirical ceiling")
    _invalid(empirical.get("candidate") == "P_EMPIRICAL_FIT", "V2 ceiling candidate changed")
    _invalid(empirical.get("replacement_fit_rows_per_fold") == FIT_COUNT_PER_FOLD, "V2 replacement fit count changed")
    _invalid(empirical.get("historical_fit_rows_per_fold") != FIT_COUNT_PER_FOLD, "V2 ceiling was not replacement-based")
    _invalid(empirical.get("no_best_bank_selection") is True, "V2 ceiling best-bank selection changed")
    _invalid(empirical.get("no_seed_retry") is True, "V2 ceiling seed retry changed")
    folds = _sequence(empirical.get("fold_results"), "V2 empirical ceiling fold results")
    _invalid(len(folds) == FOLD_COUNT, "V2 ceiling fold results are incomplete")
    thresholds: list[dict[str, Any]] = []
    for fold, item_value in enumerate(folds):
        item = _mapping(item_value, f"V2 ceiling fold {fold}")
        _invalid(item.get("fold") == fold, f"V2 ceiling fold index changed: {fold}")
        _invalid(item.get("operationally_valid") is True and item.get("compatibility") is True, f"V2 ceiling fold {fold} is not valid")
        threshold = dict(_mapping(item.get("model_vs_model_reference_q95"), f"V2 threshold fold {fold}"))
        _validate_metric_mapping(threshold, f"V2 threshold fold {fold}")
        pairs = _sequence(item.get("model_vs_model_reference_pairs"), f"V2 reference pairs fold {fold}")
        _invalid(len(pairs) == 4, f"V2 reference pair count changed at fold {fold}")
        thresholds.append(threshold)
    governance = _mapping(report.get("governance"), "V2 ceiling governance")
    _invalid(governance.get("selected_producer") is None, "V2 ceiling selected a producer")
    for key, expected in (
        ("sealed_confirmation_consumed", False),
        ("training_authorized", False),
        ("formal_promotion", False),
        ("generation_readiness", "not_ready"),
    ):
        _invalid(governance.get(key) == expected, f"V2 ceiling governance changed: {key}")
    validate_sealed_boundary(
        {
            "status": report.get("sealed_confirmation", {}).get("status"),
            "consumed": report.get("sealed_confirmation", {}).get("consumed"),
            "materialized": report.get("sealed_confirmation", {}).get(
                "materialized", report.get("sealed_confirmation", {}).get("modified", False)
            ),
            "sample_values_loaded": report.get("sealed_confirmation", {}).get(
                "sample_values_loaded", report.get("sealed_confirmation", {}).get("values_loaded")
            ),
        }
    )
    return report, thresholds


def _validate_metric_mapping(metrics: Mapping[str, Any], label: str) -> None:
    for name in SCALAR_METRICS:
        value = metrics.get(name)
        _invalid(isinstance(value, (int, float)), f"{label} is missing metric {name}")
        numeric = torch.tensor(float(value), dtype=torch.float64)
        _invalid(bool(torch.isfinite(numeric)) and float(value) >= 0.0, f"{label} metric is invalid: {name}")
    vector = metrics.get("per_coordinate_marginal_energy_distance")
    _invalid(isinstance(vector, list) and len(vector) == DIMENSION, f"{label} marginal vector shape changed")
    _invalid(all(bool(torch.isfinite(torch.tensor(float(value), dtype=torch.float64))) and float(value) >= 0.0 for value in vector), f"{label} marginal vector is invalid")


def _validate_authority_chain(repo_root: Path) -> dict[str, Any]:
    validate_authority_hashes(repo_root, AUTHORITY_SHA256)
    source, fold, rows, fold_ids, q_meta = v2_ceiling._validate_source_and_fold(repo_root)
    _invalid(source.get("status") == "FROZEN_SOURCE_ROWS", "V2 source status changed")
    _invalid(fold.get("status") == "FROZEN_BEFORE_VALIDATION", "V2 fold status changed")
    _invalid(fold.get("protocol", {}).get("assignment_sha256") == v2_ceiling.FOLD_ASSIGNMENT_SHA256, "V2 fold assignment changed")
    _invalid(fold.get("protocol", {}).get("q_values_used") is False, "V2 fold construction used q values")
    _invalid(q_meta.get("values_loaded") is False, "q values were loaded during metadata preflight")
    validate_fold_partition(fold_ids)
    historical = v2_ceiling._validate_historical_ceiling(repo_root)
    historical_boundaries = baseline._verify_historical_boundaries(repo_root)
    validate_sealed_boundary(
        {
            "status": historical_boundaries["sealed_confirmation"]["status"],
            "consumed": False,
            "materialized": False,
            "sample_values_loaded": historical_boundaries["sealed_confirmation"]["sample_values_loaded"],
        }
    )
    current_governance = _validate_current_governance(repo_root)
    ceiling, thresholds = _validate_ceiling_authority(repo_root)
    return {
        "source": source,
        "fold": fold,
        "rows": rows,
        "fold_ids": fold_ids,
        "q_meta": q_meta,
        "historical": historical,
        "historical_boundaries": historical_boundaries,
        "current_governance": current_governance,
        "ceiling": ceiling,
        "thresholds": thresholds,
    }


def _load_q_after_preflight(repo_root: Path, preflight: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    q, source_indices, labels, identities, _identity = v2_ceiling._load_q_after_preflight(repo_root, preflight)
    _invalid(tuple(q.shape) == (POOL_COUNT, DIMENSION), "V2 q shape changed")
    _invalid(tuple(source_indices.shape) == (POOL_COUNT,), "V2 source-index shape changed")
    _invalid(tuple(labels.shape) == (POOL_COUNT,), "V2 label shape changed")
    _invalid(len(identities) == POOL_COUNT, "V2 stable row identities are incomplete")
    return q, source_indices, labels, identities


def _statistics_contract(repo_root: Path, ceiling: Mapping[str, Any]) -> dict[str, Any]:
    settings = dict(_mapping(ceiling["execution_settings"], "ceiling settings"))
    contract = {
        "q_representation": {
            "coordinate": "R2_AMPLITUDE",
            "definition": "q=abs(c/sqrt(s))=abs(R3_COMPLEX)",
            "dimension": DIMENSION,
            "dtype": "torch.float64",
            "preprocessing": "none beyond frozen q materialization",
            "normalization": "none; no samplewise normalization",
        },
        "fold_definition": {
            "fold_count": FOLD_COUNT,
            "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
            "fit_per_class_per_fold": FIT_PER_CLASS_PER_FOLD,
            "heldout_per_class_per_fold": HELDOUT_PER_CLASS_PER_FOLD,
            "assignment_sha256": v2_ceiling.FOLD_ASSIGNMENT_SHA256,
            "membership_source": "frozen N3_DEVELOPMENT_BANK_V2 five-fold manifest",
        },
        "sample_sizes": {
            "data_vs_model": HELDOUT_COUNT_PER_FOLD,
            "model_vs_model_reference": MODEL_SAMPLE_COUNT,
            "n4_model_sample": MODEL_SAMPLE_COUNT,
            "asymmetry_limitation": "DATA-vs-MODEL is 1000 vs 200; MODEL-vs-MODEL reference is 200 vs 200",
        },
        "metric_contract": {
            "implementation_path": _relative(repo_root, repo_root / METRIC_IMPLEMENTATION),
            "implementation_sha256": file_sha256(repo_root / METRIC_IMPLEMENTATION),
            "primary": "biased empirical energy distance in raw q space",
            "implementation_adapter": "_pair_metrics_raw_q; historical formulas with only equal-row-count precondition removed",
            "biased_empirical_ed": True,
            "diagonal_terms_included": True,
            "scalar_metrics": list(SCALAR_METRICS),
            "joint_metrics": list(JOINT_METRICS),
            "marginal_metrics": list(MARGINAL_METRICS),
            "per_coordinate_vector_role": "diagnostic only; not a compatibility gate",
        },
        "reference_contract": {
            "reference_pair_construction": [[0, 1], [2, 3], [4, 5], [6, 7]],
            "reference_sample_count": 8,
            "threshold_quantile": Q95,
            "threshold_source": "frozen V2 empirical ceiling report; q95 is not recalibrated by N4",
            "threshold_is_calibrated_confidence": False,
        },
        "pass_rule": {
            "fold": "all eight scalar candidate DATA-vs-MODEL metrics <= the corresponding frozen fold threshold; equality passes",
            "candidate": "all five folds pass",
            "package": "candidate support is the independent all-five-fold rule; zero/one/multiple supported candidates are reported without selection",
            "all_five_folds_required": True,
        },
        "sampling_statistics_reused": {
            "v2_ceiling_data_vs_model_replicates": settings["data_vs_model_replicates"],
            "v2_ceiling_model_vs_model_replicates": settings["model_vs_model_replicates"],
            "v2_ceiling_model_reference_pairs": settings["model_reference_pairs"],
            "v2_ceiling_rng_namespace": settings["rng_namespace"],
            "with_replacement": settings["with_replacement"],
            "seed_retry": settings["seed_retry"],
            "best_bank_selection": settings["best_bank_selection"],
        },
        "invalid_handling": {
            "finite_required": True,
            "nonnegative_q_required": True,
            "metric_failure": "fail closed as invalid harness",
            "no_silent_fallback": True,
        },
        "limitation": "Biased empirical ED has different finite-sample expectation scales for the 1000-vs-200 DATA/MODEL comparison and 200-vs-200 MODEL/MODEL reference; the empirical q95 is a development threshold, not a calibrated 95% confidence guarantee.",
    }
    return contract


def _candidate_config_hash(candidate: Mapping[str, Any]) -> str:
    return canonical_sha256(candidate)


def _candidate_by_id(registry: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates = [dict(_mapping(value, "candidate registry entry")) for value in _sequence(registry.get("candidates"), "candidate registry")]
    ids = tuple(str(item.get("candidate_id")) for item in candidates)
    _invalid(ids == EXPECTED_CANDIDATE_IDS, "candidate registry candidate set changed")
    _invalid(len(candidates) == len(set(ids)), "candidate registry contains duplicate candidates")
    return candidates


def _validate_registry(repo_root: Path, git: Mapping[str, Any]) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    registry_path = repo_root / REGISTRY_PATH
    _invalid(registry_path.is_file(), f"N4 candidate registry is missing: {registry_path}")
    registry_sha256 = file_sha256(registry_path)
    registry = dict(_mapping(json.loads(registry_path.read_text(encoding="utf-8")), "N4 candidate registry"))
    _invalid(registry.get("schema") == registry_definition.SCHEMA, "candidate registry schema changed")
    _invalid(registry.get("status") == "FROZEN", "candidate registry is not frozen")
    _invalid(registry.get("candidate_count") == len(EXPECTED_CANDIDATE_IDS), "candidate registry count changed")
    registry_git = _mapping(registry.get("git"), "candidate registry git")
    _invalid(registry_git.get("branch") == git["branch"] == EXPECTED_BRANCH, "candidate registry branch changed")
    _invalid(registry_git.get("head") == git["head"] == EXPECTED_STARTING_HEAD, "candidate registry commit changed")
    executor_path = repo_root / str(registry_git.get("executor_script"))
    _invalid(executor_path.resolve() == (repo_root / registry_definition.EXECUTOR_PATH).resolve(), "candidate registry executor path changed")
    _invalid(file_sha256(executor_path) == registry_git.get("executor_script_sha256"), "N4 executor changed after registry freeze")
    definition_path = repo_root / str(registry_git.get("definition_script"))
    _invalid(file_sha256(definition_path) == registry_git.get("definition_script_sha256"), "candidate registry definition changed after freeze")
    expected_authorities = [{"path": path, "sha256": sha} for path, sha in AUTHORITY_SHA256.items()]
    _invalid(registry.get("v2_authorities") == expected_authorities, "candidate registry V2 authorities changed")
    freeze = _mapping(registry.get("freeze"), "candidate registry freeze")
    for key in (
        "registry_frozen_before_first_evaluator_call",
        "candidate_addition_after_freeze",
        "candidate_deletion_after_freeze",
        "parameter_mutation_after_freeze",
        "fit_method_mutation_after_freeze",
        "heldout_driven_tuning",
        "tie_break_registered",
    ):
        expected = True if key == "registry_frozen_before_first_evaluator_call" else False
        _invalid(freeze.get(key) == expected, f"candidate registry freeze changed: {key}")
    candidates = _candidate_by_id(registry)
    for candidate in candidates:
        source_path = repo_root / str(candidate["implementation_path"])
        _invalid(source_path.is_file(), f"candidate implementation is missing: {source_path}")
        _invalid(file_sha256(source_path) == candidate.get("implementation_sha256"), f"candidate implementation changed: {source_path}")
        _invalid(candidate.get("git_commit") == git["head"], f"candidate commit changed: {candidate.get('candidate_id')}")
        _invalid(candidate.get("input_q_dimensionality") == DIMENSION, f"candidate input dimension changed: {candidate.get('candidate_id')}")
        _invalid(candidate.get("output_q_dimensionality") == DIMENSION, f"candidate output dimension changed: {candidate.get('candidate_id')}")
        _invalid(candidate.get("gradient_training") is False and candidate.get("analytical_fitting") is True, f"candidate fitting mode changed: {candidate.get('candidate_id')}")
    governance = _mapping(registry.get("governance"), "candidate registry governance")
    validate_final_governance({**governance, "sealed_confirmation_consumed": False})
    return registry, registry_sha256, candidates


def _fold_identity(
    q: torch.Tensor,
    source_indices: torch.Tensor,
    labels: torch.Tensor,
    identities: list[str],
    fold_ids: torch.Tensor,
    fold: int,
) -> dict[str, Any]:
    heldout_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1)
    fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1)
    validate_fit_holdout_boundary(fit_positions, heldout_positions)
    _invalid(torch.equal(torch.sort(torch.cat((fit_positions, heldout_positions))).values, torch.arange(POOL_COUNT, dtype=torch.int64)), f"fold {fold} does not cover the V2 bank")
    fit_labels = labels.index_select(0, fit_positions)
    heldout_labels = labels.index_select(0, heldout_positions)
    _invalid(torch.equal(torch.bincount(fit_labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), FIT_PER_CLASS_PER_FOLD, dtype=torch.int64)), f"fold {fold} fit class balance changed")
    _invalid(torch.equal(torch.bincount(heldout_labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), HELDOUT_PER_CLASS_PER_FOLD, dtype=torch.int64)), f"fold {fold} heldout class balance changed")
    fit_source_indices = source_indices.index_select(0, fit_positions).contiguous()
    heldout_source_indices = source_indices.index_select(0, heldout_positions).contiguous()
    fit_labels = fit_labels.contiguous()
    heldout_labels = heldout_labels.contiguous()
    fit_identity = [identities[int(position)] for position in fit_positions.tolist()]
    heldout_identity = [identities[int(position)] for position in heldout_positions.tolist()]
    heldout_q = q.index_select(0, heldout_positions).contiguous()
    fit_q = q.index_select(0, fit_positions).contiguous()
    return {
        "fold": fold,
        "fit_positions": fit_positions,
        "heldout_positions": heldout_positions,
        "fit_q": fit_q,
        "heldout_q": heldout_q,
        "fit_source_indices": fit_source_indices,
        "heldout_source_indices": heldout_source_indices,
        "fit_labels": fit_labels,
        "heldout_labels": heldout_labels,
        "fit_identities": fit_identity,
        "heldout_identities": heldout_identity,
        "fit_identity_sha256": canonical_sha256(fit_identity),
        "heldout_identity_sha256": canonical_sha256(heldout_identity),
        "fit_q_sha256": tensor_sha256(fit_q),
        "heldout_q_sha256": tensor_sha256(heldout_q),
    }


def _data_identity(fold_data: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_positions": [int(value) for value in fold_data["heldout_positions"].tolist()],
        "source_indices": [int(value) for value in fold_data["heldout_source_indices"].tolist()],
        "stable_source_row_identities": list(fold_data["heldout_identities"]),
        "row_positions_sha256": tensor_sha256(fold_data["heldout_positions"]),
        "source_indices_sha256": tensor_sha256(fold_data["heldout_source_indices"]),
        "labels_sha256": tensor_sha256(fold_data["heldout_labels"]),
        "stable_source_row_identity_sha256": fold_data["heldout_identity_sha256"],
        "q_tensor_sha256": fold_data["heldout_q_sha256"],
        "count": HELDOUT_COUNT_PER_FOLD,
        "dimension": DIMENSION,
    }


def _fit_candidate(candidate_id: str, q_fit: torch.Tensor) -> tuple[dict[str, Any], dict[str, Any]]:
    _invalid(tuple(q_fit.shape) == (FIT_COUNT_PER_FOLD, DIMENSION), f"{candidate_id} fit shape changed")
    _invalid(q_fit.dtype == torch.float64 and bool(torch.isfinite(q_fit).all()) and bool(q_fit.ge(0.0).all()), f"{candidate_id} fit q is invalid")
    with torch.no_grad():
        if candidate_id in {
            "SHRINKAGE_ELLIPTICAL_GAUSSIAN_AMPLITUDE_V1",
            "RADIAL_SCALE_MIXTURE_AMPLITUDE_V1",
        }:
            fit = baseline._fit_fold(q_fit)
            if candidate_id == "RADIAL_SCALE_MIXTURE_AMPLITUDE_V1" and fit["radial"]["status"] != "RADIAL_EXTENSION_IDENTIFIED_ON_FIT":
                return fit, {"status": "CANDIDATE_FOLD_SCIENTIFIC_FAILURE", "reason": fit["radial"]["status"]}
            return fit, {"status": "FIT_COMPLETED", "reason": None}
        if candidate_id == "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1":
            previous_count = copula.FIT_COUNT_PER_FOLD
            copula.FIT_COUNT_PER_FOLD = FIT_COUNT_PER_FOLD
            try:
                fit = copula._fit_fold(q_fit)
            finally:
                copula.FIT_COUNT_PER_FOLD = previous_count
            return fit, {"status": "FIT_COMPLETED", "reason": None}
    raise InvalidHarnessError(f"unregistered candidate reached fitting dispatch: {candidate_id}")


def _candidate_fit_state(candidate_id: str, fit: Mapping[str, Any]) -> dict[str, Any]:
    if candidate_id in {
        "SHRINKAGE_ELLIPTICAL_GAUSSIAN_AMPLITUDE_V1",
        "RADIAL_SCALE_MIXTURE_AMPLITUDE_V1",
    }:
        return {
            "mu": fit["mu"],
            "sample_covariance_unbiased": fit["sample_covariance_unbiased"],
            "tau": fit["tau"],
            "alpha": fit["alpha"],
            "sigma": fit["sigma"],
            "chol": fit["chol"],
            "fit_count": fit["fit_count"],
            "radial": dict(fit["radial"]),
        }
    dependence = _mapping(fit["dependence"], "copula dependence")
    return {
        "sorted_values": fit["sorted_values"],
        "plotting_positions": fit["plotting_positions"],
        "z": fit["z"],
        "tie_summary": dict(fit["tie_summary"]),
        "dependence": {
            "z_mean": dependence["z_mean"],
            "standard_deviation": dependence["standard_deviation"],
            "R_emp": dependence["R_emp"],
            "tau": dependence["tau"],
            "beta": dependence["beta"],
            "delta": dependence["delta"],
            "alpha": dependence["alpha"],
            "R_shrunk": dependence["R_shrunk"],
            "cholesky": dependence["cholesky"],
            "eigenvalues": dependence["eigenvalues"],
            "diagonal": dependence["diagonal"],
            "diagonal_error": dependence["diagonal_error"],
        },
    }


def _fit_summary(candidate_id: str, fit: Mapping[str, Any]) -> dict[str, Any]:
    state = _candidate_fit_state(candidate_id, fit)
    summary: dict[str, Any] = {
        "candidate_id": candidate_id,
        "fit_rows": FIT_COUNT_PER_FOLD,
        "dimension": DIMENSION,
        "dtype": "torch.float64",
        "parameter_state_sha256": canonical_sha256(state),
    }
    if candidate_id != "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1":
        summary.update(
            {
                "mu_sha256": tensor_sha256(fit["mu"]),
                "sigma_sha256": tensor_sha256(fit["sigma"]),
                "chol_sha256": tensor_sha256(fit["chol"]),
                "alpha": float(fit["alpha"]),
                "tau": float(fit["tau"]),
                "radial": dict(fit["radial"]),
            }
        )
    else:
        dependence = _mapping(fit["dependence"], "copula dependence")
        summary.update(
            {
                "sorted_values_sha256": tensor_sha256(fit["sorted_values"]),
                "plotting_positions_sha256": tensor_sha256(fit["plotting_positions"]),
                "z_sha256": tensor_sha256(fit["z"]),
                "R_shrunk_sha256": tensor_sha256(dependence["R_shrunk"]),
                "cholesky_sha256": tensor_sha256(dependence["cholesky"]),
                "alpha": float(dependence["alpha"]),
            }
        )
    return summary


def _sample_candidate(candidate_id: str, fit: Mapping[str, Any], fold: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if candidate_id in {
        "SHRINKAGE_ELLIPTICAL_GAUSSIAN_AMPLITUDE_V1",
        "RADIAL_SCALE_MIXTURE_AMPLITUDE_V1",
    }:
        seed = baseline._seed(candidate_id, "data-vs-model", fold, 0)
        model_q, sampling = baseline._sample_positive(fit, candidate_id, seed=seed)
        sampling = {
            **sampling,
            "sample_count": MODEL_SAMPLE_COUNT,
            "best_of_n": False,
            "seed_retry": False,
            "heldout_seed_tuning": False,
            "sampling_method": "one fixed-seed candidate bank with legal-state rejection only",
        }
    elif candidate_id == "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1":
        model_q, sampling = copula._sample_bank(fit, role="data-vs-model", fold=fold, replicate=0)
        sampling = {
            **sampling,
            "sample_count": MODEL_SAMPLE_COUNT,
            "best_of_n": False,
            "seed_retry": False,
            "heldout_seed_tuning": False,
            "sampling_method": "one fixed-seed Gaussianized empirical-marginal bank",
        }
    else:
        raise InvalidHarnessError(f"unregistered candidate reached sampling dispatch: {candidate_id}")
    validate_sampling_record(sampling)
    _invalid(tuple(model_q.shape) == (MODEL_SAMPLE_COUNT, DIMENSION), f"{candidate_id} model sample shape changed")
    _invalid(model_q.dtype == torch.float64, f"{candidate_id} model sample dtype changed")
    _invalid(bool(torch.isfinite(model_q).all()) and bool(model_q.ge(0.0).all()), f"{candidate_id} model sample is illegal")
    return model_q.contiguous(), sampling


def _empty_model_sampling(candidate_id: str, fold: int, reason: str) -> dict[str, Any]:
    if candidate_id in {
        "SHRINKAGE_ELLIPTICAL_GAUSSIAN_AMPLITUDE_V1",
        "RADIAL_SCALE_MIXTURE_AMPLITUDE_V1",
    }:
        seed = baseline._seed(candidate_id, "data-vs-model", fold, 0)
        rng = "torch.Generator(device='cpu')"
    else:
        seed = copula._seed("data-vs-model", fold, 0)
        rng = "torch.Generator(device='cpu')"
    return {
        "seed": seed,
        "rng": rng,
        "sample_count": MODEL_SAMPLE_COUNT,
        "best_of_n": False,
        "seed_retry": False,
        "heldout_seed_tuning": False,
        "feasible": False,
        "sampling_method": "not invoked because frozen candidate fit failed its declared scientific condition",
        "failure": reason,
    }


def _runtime_snapshot(repo_root: Path, registry_path: Path, registry_sha256: str, candidate: Mapping[str, Any], stats: Mapping[str, Any]) -> dict[str, Any]:
    authority_hashes = {path: file_sha256(repo_root / Path(path)) for path in AUTHORITY_SHA256}
    implementation_path = repo_root / str(candidate["implementation_path"])
    return {
        "git_head": _git_state(repo_root)["head"],
        "registry_sha256": file_sha256(registry_path),
        "registry_sha256_at_freeze": registry_sha256,
        "candidate_id": candidate["candidate_id"],
        "candidate_config_sha256": _candidate_config_hash(candidate),
        "candidate_implementation_sha256": file_sha256(implementation_path),
        "executor_script_sha256": file_sha256(repo_root / registry_definition.EXECUTOR_PATH),
        "metric_implementation_sha256": file_sha256(repo_root / METRIC_IMPLEMENTATION),
        "governance_state_sha256": file_sha256(repo_root / GOVERNANCE_STATE),
        "authority_hashes": authority_hashes,
        "statistics_config_sha256": canonical_sha256(stats),
    }


def _artifact_common(
    repo_root: Path,
    *,
    schema: str,
    created_at: str,
    git_head: str,
    registry_sha256: str,
    stats_sha256: str,
    candidate: Mapping[str, Any],
    fold: int,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "version": 1,
        "created_at_utc": created_at,
        "git_commit": git_head,
        "registry_sha256": registry_sha256,
        "config_sha256": stats_sha256,
        "candidate_id": candidate["candidate_id"],
        "candidate_version": candidate["candidate_version"],
        "candidate_config_sha256": _candidate_config_hash(candidate),
        "fold": fold,
        "coordinate": "R2_AMPLITUDE",
        "dimension": DIMENSION,
        "dtype": "torch.float64",
        "source_lineage": {
            "source_manifest": {"path": _relative(repo_root, repo_root / SOURCE_MANIFEST), "sha256": AUTHORITY_SHA256[SOURCE_MANIFEST.as_posix()]},
            "q_artifact": {"path": _relative(repo_root, repo_root / Q_ARTIFACT), "sha256": AUTHORITY_SHA256[Q_ARTIFACT.as_posix()]},
            "fold_manifest": {"path": _relative(repo_root, repo_root / FOLD_MANIFEST), "sha256": AUTHORITY_SHA256[FOLD_MANIFEST.as_posix()]},
            "ceiling_report": {"path": _relative(repo_root, repo_root / CEILING_REPORT), "sha256": AUTHORITY_SHA256[CEILING_REPORT.as_posix()]},
        },
    }


def _fit_artifact_payload(
    repo_root: Path,
    *,
    created_at: str,
    git_head: str,
    registry_sha256: str,
    stats_sha256: str,
    candidate: Mapping[str, Any],
    fold_data: Mapping[str, Any],
    fit: Mapping[str, Any],
    fit_status: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **_artifact_common(
            repo_root,
            schema=FITTED_ARTIFACT_SCHEMA,
            created_at=created_at,
            git_head=git_head,
            registry_sha256=registry_sha256,
            stats_sha256=stats_sha256,
            candidate=candidate,
            fold=int(fold_data["fold"]),
        ),
        "status": fit_status["status"],
        "fit_source": {
            "count": FIT_COUNT_PER_FOLD,
            "row_positions": [int(value) for value in fold_data["fit_positions"].tolist()],
            "source_indices": [int(value) for value in fold_data["fit_source_indices"].tolist()],
            "labels": [int(value) for value in fold_data["fit_labels"].tolist()],
            "stable_source_row_identities": list(fold_data["fit_identities"]),
            "row_positions_sha256": tensor_sha256(fold_data["fit_positions"]),
            "source_indices_sha256": tensor_sha256(fold_data["fit_source_indices"]),
            "labels_sha256": tensor_sha256(fold_data["fit_labels"]),
            "stable_source_row_identity_sha256": fold_data["fit_identity_sha256"],
            "fit_q_sha256": fold_data["fit_q_sha256"],
            "heldout_excluded": True,
            "heldout_source_indices_sha256": tensor_sha256(fold_data["heldout_source_indices"]),
        },
        "fit_status": dict(fit_status),
        "parameters": _candidate_fit_state(str(candidate["candidate_id"]), fit),
    }


def _model_artifact_payload(
    repo_root: Path,
    *,
    created_at: str,
    git_head: str,
    registry_sha256: str,
    stats_sha256: str,
    candidate: Mapping[str, Any],
    fold: int,
    model_q: torch.Tensor,
    sampling: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    return {
        **_artifact_common(
            repo_root,
            schema=MODEL_ARTIFACT_SCHEMA,
            created_at=created_at,
            git_head=git_head,
            registry_sha256=registry_sha256,
            stats_sha256=stats_sha256,
            candidate=candidate,
            fold=fold,
        ),
        "status": status,
        "model_q": model_q,
        "model_q_shape": [int(value) for value in model_q.shape],
        "model_q_sha256": tensor_sha256(model_q),
        "sampling": dict(sampling),
        "heldout_values_used_for_sampling": False,
        "best_of_n": False,
        "resampling_after_evaluation": False,
    }


def _metric_margins(metrics: Mapping[str, Any], threshold: Mapping[str, Any]) -> dict[str, Any]:
    margins = {name: float(threshold[name]) - float(metrics[name]) for name in SCALAR_METRICS}
    margins["per_coordinate_marginal_energy_distance"] = [
        float(right) - float(left)
        for left, right in zip(metrics["per_coordinate_marginal_energy_distance"], threshold["per_coordinate_marginal_energy_distance"], strict=True)
    ]
    return margins


def _compatibility(metrics: Mapping[str, Any], threshold: Mapping[str, Any]) -> tuple[dict[str, bool], dict[str, Any]]:
    components = {name: float(metrics[name]) <= float(threshold[name]) for name in SCALAR_METRICS}
    return components, _metric_margins(metrics, threshold)


def _artifact_ref(
    repo_root: Path,
    path: Path,
    *,
    role: str,
    schema: str,
    candidate_id: str | None = None,
    fold: int | None = None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": _relative(repo_root, path),
        "sha256": file_sha256(source_path or path),
        "role": role,
        "schema": schema,
    }
    if candidate_id is not None:
        result["candidate_id"] = candidate_id
    if fold is not None:
        result["fold"] = fold
    return result


def _write_json(temporary: Path, name: str, payload: Mapping[str, Any]) -> Path:
    path = temporary / name
    atomic_write_new_json(path, payload, artifact_kind=f"N4 {name}")
    return path


def _write_torch(temporary: Path, relative_path: Path, payload: Mapping[str, Any]) -> Path:
    path = temporary / relative_path
    atomic_torch_save_new(path, payload, artifact_kind=f"N4 {relative_path.name}")
    return path


def _evaluate_candidate_fold(
    repo_root: Path,
    *,
    temporary: Path,
    output_dir: Path,
    created_at: str,
    git_head: str,
    registry_path: Path,
    registry_sha256: str,
    candidate: Mapping[str, Any],
    fold_data: Mapping[str, Any],
    threshold: Mapping[str, Any],
    stats: Mapping[str, Any],
    exposure_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_id = str(candidate["candidate_id"])
    fold = int(fold_data["fold"])
    candidate_config_sha256 = _candidate_config_hash(candidate)
    fit, fit_status = _fit_candidate(candidate_id, fold_data["fit_q"])
    fit_summary = _fit_summary(candidate_id, fit)
    fit_artifact = _fit_artifact_payload(
        repo_root,
        created_at=created_at,
        git_head=git_head,
        registry_sha256=registry_sha256,
        stats_sha256=canonical_sha256(stats),
        candidate=candidate,
        fold_data=fold_data,
        fit=fit,
        fit_status=fit_status,
    )
    fit_relative = Path("artifacts") / candidate_id.lower() / f"fold_{fold:02d}_fitted_producer.v1.pt"
    fit_path = _write_torch(temporary, fit_relative, fit_artifact)
    fit_ref = _artifact_ref(
        repo_root,
        repo_root / output_dir / fit_relative,
        role="fitted producer",
        schema=FITTED_ARTIFACT_SCHEMA,
        candidate_id=candidate_id,
        fold=fold,
        source_path=fit_path,
    )

    if fit_status["status"] != "FIT_COMPLETED":
        model_q = torch.empty((0, DIMENSION), dtype=torch.float64)
        sampling = _empty_model_sampling(candidate_id, fold, str(fit_status["reason"]))
        model_status = "CANDIDATE_FOLD_SCIENTIFIC_FAILURE"
        evaluator_call_index = None
        metrics = None
        threshold_out = None
        margins = None
        components = None
        fold_pass = False
        terminal_status = "candidate_scientific_failure"
    else:
        model_q, sampling = _sample_candidate(candidate_id, fit, fold)
        model_status = "SAMPLE_COMPLETED"
        evaluator_call_index = len(exposure_records) + 1
        _invalid(evaluator_call_index <= MAX_EVALUATOR_CALLS, "evaluator call count exceeded the preregistered cap")
        model_status = "SAMPLE_COMPLETED"
        model_relative = Path("artifacts") / candidate_id.lower() / f"fold_{fold:02d}_sampled_model_q.v1.pt"
        model_artifact = _model_artifact_payload(
            repo_root,
            created_at=created_at,
            git_head=git_head,
            registry_sha256=registry_sha256,
            stats_sha256=canonical_sha256(stats),
            candidate=candidate,
            fold=fold,
            model_q=model_q,
            sampling=sampling,
            status=model_status,
        )
        model_path = _write_torch(temporary, model_relative, model_artifact)
        model_ref = _artifact_ref(
            repo_root,
            repo_root / output_dir / model_relative,
            role="sampled model q",
            schema=MODEL_ARTIFACT_SCHEMA,
            candidate_id=candidate_id,
            fold=fold,
            source_path=model_path,
        )
        snapshot_before = _runtime_snapshot(repo_root, registry_path, registry_sha256, candidate, stats)
        metrics = v2_ceiling._pair_metrics_raw_q(fold_data["heldout_q"], model_q)
        _validate_metric_mapping(metrics, f"{candidate_id} fold {fold} model metrics")
        snapshot_after = _runtime_snapshot(repo_root, registry_path, registry_sha256, candidate, stats)
        validate_runtime_snapshot(snapshot_before, snapshot_after)
        validate_registry_frozen(snapshot_before["registry_sha256_at_freeze"], snapshot_after["registry_sha256"])
        threshold_out = dict(threshold)
        components, margins = _compatibility(metrics, threshold_out)
        fold_pass = all(components.values())
        terminal_status = "evaluated"
        exposure_records.append(
            {
                "candidate_id": candidate_id,
                "candidate_version": candidate["candidate_version"],
                "fold": fold,
                "evaluator_call_timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "evaluator_call_number": evaluator_call_index,
                "metric": "biased empirical raw-q pair metrics; all eight scalar guardrails",
                "metric_result_sha256": canonical_sha256(metrics),
                "result_visibility": "visible_to_this_comparison_after_call",
                "result_caused_code_or_config_change": False,
                "candidate_config_sha256": candidate_config_sha256,
                "registry_sha256_before": snapshot_before["registry_sha256"],
                "registry_sha256_after": snapshot_after["registry_sha256"],
                "model_sample_sha256": tensor_sha256(model_q),
            }
        )

    if fit_status["status"] != "FIT_COMPLETED":
        model_relative = Path("artifacts") / candidate_id.lower() / f"fold_{fold:02d}_sampled_model_q.v1.pt"
        model_artifact = _model_artifact_payload(
            repo_root,
            created_at=created_at,
            git_head=git_head,
            registry_sha256=registry_sha256,
            stats_sha256=canonical_sha256(stats),
            candidate=candidate,
            fold=fold,
            model_q=model_q,
            sampling=sampling,
            status=model_status,
        )
        model_path = _write_torch(temporary, model_relative, model_artifact)
        model_ref = _artifact_ref(
            repo_root,
            repo_root / output_dir / model_relative,
            role="sampled model q",
            schema=MODEL_ARTIFACT_SCHEMA,
            candidate_id=candidate_id,
            fold=fold,
            source_path=model_path,
        )

    fold_result: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_version": candidate["candidate_version"],
        "fold": fold,
        "fit": {
            **fit_summary,
            "status": fit_status["status"],
            "reason": fit_status["reason"],
            "fit_source_row_positions_sha256": tensor_sha256(fold_data["fit_positions"]),
            "fit_source_indices_sha256": tensor_sha256(fold_data["fit_source_indices"]),
            "fit_labels_sha256": tensor_sha256(fold_data["fit_labels"]),
            "heldout_excluded_from_fit": True,
            "fit_artifact": fit_ref,
        },
        "data": _data_identity(fold_data),
        "model": {
            "sample_count": MODEL_SAMPLE_COUNT if fit_status["status"] == "FIT_COMPLETED" else 0,
            "sample_artifact": model_ref,
            "sample_seed": sampling["seed"],
            "sampling": dict(sampling),
            "model_q_sha256": tensor_sha256(model_q),
            "heldout_values_used_for_sampling": False,
            "best_of_n": False,
            "resampling_after_evaluation": False,
        },
        "evaluator_call_index": evaluator_call_index,
        "metrics": metrics,
        "threshold": threshold_out,
        "margin": margins,
        "compatibility_components": components,
        "fold_pass": fold_pass,
        "terminal_status": terminal_status,
        "threshold_source": {
            "path": _relative(repo_root, repo_root / CEILING_REPORT),
            "sha256": AUTHORITY_SHA256[CEILING_REPORT.as_posix()],
            "q95": Q95,
            "recomputed": False,
        },
    }
    return fold_result, {"fit": fit_ref, "model": model_ref}


def _candidate_summary(candidate_id: str, folds: list[Mapping[str, Any]]) -> dict[str, Any]:
    pattern = [bool(item.get("fold_pass")) for item in folds]
    failures = []
    for item in folds:
        if not item.get("fold_pass"):
            failures.append(
                {
                    "fold": item["fold"],
                    "terminal_status": item["terminal_status"],
                    "reason": item["fit"].get("reason") or "one or more frozen scalar compatibility components failed",
                    "margin": item.get("margin"),
                    "failed_metrics": [name for name, passed in (item.get("compatibility_components") or {}).items() if not passed],
                }
            )
    return {
        "candidate_id": candidate_id,
        "fold_count": FOLD_COUNT,
        "fold_pass_pattern": pattern,
        "compatible_fold_count": sum(pattern),
        "development_supported": all(pattern),
        "all_folds_terminal": all(item.get("terminal_status") in {"evaluated", "candidate_scientific_failure"} for item in folds),
        "failure_taxonomy": failures,
    }


def _classification(summaries: list[Mapping[str, Any]]) -> tuple[str, list[str], list[str]]:
    supported = [str(item["candidate_id"]) for item in summaries if bool(item["development_supported"])]
    unsupported = [str(item["candidate_id"]) for item in summaries if not bool(item["development_supported"])]
    if not supported:
        return CLASSIFICATION_NO_SUPPORTED, supported, unsupported
    if len(supported) == 1:
        return CLASSIFICATION_SINGLE_SUPPORTED, supported, unsupported
    return CLASSIFICATION_MULTIPLE_SUPPORTED, supported, unsupported


def _execution_artifact_refs(
    repo_root: Path,
    temporary: Path,
    output_dir: Path,
    candidate_artifacts: list[dict[str, Any]],
    json_paths: list[tuple[Path, str, str]],
) -> list[dict[str, Any]]:
    refs = list(candidate_artifacts)
    for temp_path, role, schema in json_paths:
        final_path = repo_root / output_dir / temp_path.relative_to(temporary)
        refs.append(_artifact_ref(repo_root, final_path, role=role, schema=schema, source_path=temp_path))
    return refs


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once N4 output: {output_dir}")
    git = _git_state(repo_root)
    _invalid(git["branch"] == EXPECTED_BRANCH, "current branch is not the N4 authority branch")
    _invalid(git["head"] == EXPECTED_STARTING_HEAD, "current HEAD is not the V2 authority base")
    preflight = _validate_authority_chain(repo_root)
    registry, registry_sha256, candidates = _validate_registry(repo_root, git)
    stats = _statistics_contract(repo_root, preflight["ceiling"])
    stats_sha256 = canonical_sha256(stats)
    validate_final_governance(
        {
            "selected_producer": preflight["current_governance"]["selected_producer"],
            "training_authorized": preflight["current_governance"]["training_authorized"],
            "formal_promotion": preflight["current_governance"]["formal_promotion"],
            "generation_readiness": preflight["current_governance"]["generation_readiness"],
            "sealed_confirmation_consumed": preflight["current_governance"]["sealed_confirmation_consumed"],
        }
    )
    registry_path = repo_root / REGISTRY_PATH
    registry_frozen_sha256 = file_sha256(registry_path)
    _invalid(registry_frozen_sha256 == registry_sha256, "candidate registry changed during freeze")
    q, source_indices, labels, identities = _load_q_after_preflight(repo_root, preflight)
    _invalid(file_sha256(repo_root / Q_ARTIFACT) == AUTHORITY_SHA256[Q_ARTIFACT.as_posix()], "q authority changed after load")
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    exposure_records: list[dict[str, Any]] = []
    all_candidate_artifacts: list[dict[str, Any]] = []
    fold_records_by_candidate: list[dict[str, Any]] = []
    fit_records: list[dict[str, Any]] = []
    try:
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            fold_records: list[dict[str, Any]] = []
            for fold in range(FOLD_COUNT):
                fold_data = _fold_identity(q, source_indices, labels, identities, preflight["fold_ids"], fold)
                fold_result, artifact_refs = _evaluate_candidate_fold(
                    repo_root,
                    temporary=temporary,
                    output_dir=output_dir,
                    created_at=created_at,
                    git_head=git["head"],
                    registry_path=registry_path,
                    registry_sha256=registry_sha256,
                    candidate=candidate,
                    fold_data=fold_data,
                    threshold=preflight["thresholds"][fold],
                    stats=stats,
                    exposure_records=exposure_records,
                )
                fold_records.append(fold_result)
                all_candidate_artifacts.extend([artifact_refs["fit"], artifact_refs["model"]])
                fit_records.append(
                    {
                        "candidate_id": candidate_id,
                        "candidate_version": candidate["candidate_version"],
                        "fold": fold,
                        "fit_rows": FIT_COUNT_PER_FOLD,
                        "fit_source_indices_sha256": fold_result["fit"]["fit_source_indices_sha256"],
                        "fit_row_positions_sha256": fold_result["fit"]["fit_source_row_positions_sha256"],
                        "fit_labels_sha256": fold_result["fit"]["fit_labels_sha256"],
                        "fit_artifact": artifact_refs["fit"],
                        "heldout_excluded_from_fit": True,
                    }
                )
            fold_records_by_candidate.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_version": candidate["candidate_version"],
                    "folds": fold_records,
                }
            )
        validate_exposure_ledger(exposure_records, EXPECTED_CANDIDATE_IDS)
        _invalid(len(exposure_records) == MAX_EVALUATOR_CALLS, "not every candidate/fold received its registered evaluator exposure")
        _invalid(registry_sha256 == file_sha256(registry_path), "candidate registry changed after comparison")
        _invalid(file_sha256(repo_root / GOVERNANCE_STATE) == preflight["current_governance"]["sha256"], "governance state changed during comparison")

        summaries = [_candidate_summary(item["candidate_id"], item["folds"]) for item in fold_records_by_candidate]
        classification, supported, unsupported = _classification(summaries)
        fit_manifest = {
            "schema": FIT_MANIFEST_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "gate": GATE,
            "status": "valid_completed",
            "created_at_utc": created_at,
            "git": {**git, "consumer_script": _relative(repo_root, Path(__file__).resolve()), "consumer_script_sha256": file_sha256(Path(__file__).resolve())},
            "registry": {"path": _relative(repo_root, registry_path), "sha256": registry_sha256, "candidate_count": len(candidates)},
            "configuration": {"sha256": stats_sha256, "statistics": stats},
            "fit_contract": {
                "q_only": True,
                "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
                "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
                "labels_as_model_inputs": False,
                "sealed_values_used": False,
                "fit_call_count": len(fit_records),
            },
            "fit_records": fit_records,
            "governance": {
                "selected_producer": None,
                "training_authorized": False,
                "formal_promotion": False,
                "generation_readiness": "not_ready",
            },
        }
        fold_compatibility = {
            "schema": FOLD_COMPATIBILITY_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "gate": GATE,
            "status": "valid_completed",
            "created_at_utc": created_at,
            "git": fit_manifest["git"],
            "registry": fit_manifest["registry"],
            "configuration": {"sha256": stats_sha256, "statistics": stats},
            "threshold_authority": {
                "path": _relative(repo_root, repo_root / CEILING_REPORT),
                "sha256": AUTHORITY_SHA256[CEILING_REPORT.as_posix()],
                "q95": Q95,
                "recomputed": False,
            },
            "candidate_folds": fold_records_by_candidate,
            "package_rule": stats["pass_rule"],
            "governance": {
                "selected_producer": None,
                "training_authorized": False,
                "formal_promotion": False,
                "generation_readiness": "not_ready",
            },
        }
        ledger_payload = {
            "schema": LEDGER_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "gate": GATE,
            "status": "valid_completed",
            "created_at_utc": created_at,
            "git": fit_manifest["git"],
            "registry": fit_manifest["registry"],
            "configuration_sha256": stats_sha256,
            "planned_candidate_fold_exposures": len(candidates) * FOLD_COUNT,
            "total_evaluator_calls": len(exposure_records),
            "max_evaluator_calls": MAX_EVALUATOR_CALLS,
            "adaptive_development_exposure": False,
            "result_to_code_or_config_change": False,
            "call_records": exposure_records,
            "unexposed_candidate_folds": [
                {"candidate_id": item["candidate_id"], "fold": fold}
                for item in fold_records_by_candidate
                for fold in range(FOLD_COUNT)
                if not any(record["candidate_id"] == item["candidate_id"] and record["fold"] == fold for record in exposure_records)
            ],
            "governance": {
                "selected_producer": None,
                "training_authorized": False,
                "formal_promotion": False,
                "generation_readiness": "not_ready",
            },
        }
        fit_manifest_path = _write_json(temporary, FIT_MANIFEST_NAME, fit_manifest)
        fold_compatibility_path = _write_json(temporary, FOLD_COMPATIBILITY_NAME, fold_compatibility)
        ledger_path = _write_json(temporary, LEDGER_NAME, ledger_payload)
        json_refs_without_lineage = [
            (fit_manifest_path, "fit manifest", FIT_MANIFEST_SCHEMA),
            (fold_compatibility_path, "fold compatibility", FOLD_COMPATIBILITY_SCHEMA),
            (ledger_path, "evaluator exposure ledger", LEDGER_SCHEMA),
        ]
        lineage_artifacts = _execution_artifact_refs(repo_root, temporary, output_dir, all_candidate_artifacts, json_refs_without_lineage)
        lineage = {
            "schema": LINEAGE_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "gate": GATE,
            "status": "valid_completed",
            "created_at_utc": created_at,
            "git": fit_manifest["git"],
            "configuration_sha256": stats_sha256,
            "registry": fit_manifest["registry"],
            "authorities": [
                {"path": path, "sha256": sha}
                for path, sha in AUTHORITY_SHA256.items()
            ],
            "sealed_confirmation": {
                "path": _relative(repo_root, repo_root / SEALED_CONFIRMATION_PATH),
                "sha256": SEALED_CONFIRMATION_SHA256,
                "status": "RESERVED_NOT_MATERIALIZED",
                "consumed": False,
                "values_loaded": False,
            },
            "artifacts": lineage_artifacts,
            "consumer_report_path": _relative(repo_root, repo_root / output_dir / REPORT_NAME),
            "consumer_report_is_not_self_referenced": True,
            "all_artifacts_write_once": True,
            "all_artifact_git_and_config_bound": True,
            "all_model_sampling_rng_recorded": True,
            "source_row_lineage_recorded": True,
        }
        lineage_path = _write_json(temporary, LINEAGE_NAME, lineage)
        lineage_ref = _artifact_ref(
            repo_root,
            repo_root / output_dir / LINEAGE_NAME,
            role="N4 lineage manifest",
            schema=LINEAGE_SCHEMA,
            source_path=lineage_path,
        )
        report = {
            "schema": REPORT_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "gate": GATE,
            "status": "valid_completed",
            "result": "VALID_COMPLETED",
            "classification": classification,
            "created_at_utc": created_at,
            "git": fit_manifest["git"],
            "registry": fit_manifest["registry"],
            "configuration": {"sha256": stats_sha256, "statistics": stats},
            "input_authorities": {
                "source_manifest": {"path": _relative(repo_root, repo_root / SOURCE_MANIFEST), "sha256": AUTHORITY_SHA256[SOURCE_MANIFEST.as_posix()]},
                "q_artifact": {"path": _relative(repo_root, repo_root / Q_ARTIFACT), "sha256": AUTHORITY_SHA256[Q_ARTIFACT.as_posix()], "tensor_sha256": v2_ceiling.Q_TENSOR_SHA256},
                "fold_manifest": {"path": _relative(repo_root, repo_root / FOLD_MANIFEST), "sha256": AUTHORITY_SHA256[FOLD_MANIFEST.as_posix()], "assignment_sha256": v2_ceiling.FOLD_ASSIGNMENT_SHA256},
                "empirical_ceiling_report": {"path": _relative(repo_root, repo_root / CEILING_REPORT), "sha256": AUTHORITY_SHA256[CEILING_REPORT.as_posix()]},
            },
            "candidate_count": len(candidates),
            "candidate_summaries": summaries,
            "development_supported_candidates": supported,
            "development_unsupported_candidates": unsupported,
            "failure_taxonomy": [failure for item in summaries for failure in item["failure_taxonomy"]],
            "evaluator_exposure": {
                "ledger_path": _relative(repo_root, repo_root / output_dir / LEDGER_NAME),
                "total_evaluator_calls": len(exposure_records),
                "planned_evaluator_calls": MAX_EVALUATOR_CALLS,
                "clean_one_shot": True,
                "adaptive_development_exposure": False,
            },
            "artifact_manifests": {
                "fit_manifest": _artifact_ref(repo_root, repo_root / output_dir / FIT_MANIFEST_NAME, role="fit manifest", schema=FIT_MANIFEST_SCHEMA, source_path=fit_manifest_path),
                "fold_compatibility": _artifact_ref(repo_root, repo_root / output_dir / FOLD_COMPATIBILITY_NAME, role="fold compatibility", schema=FOLD_COMPATIBILITY_SCHEMA, source_path=fold_compatibility_path),
                "evaluator_exposure_ledger": _artifact_ref(repo_root, repo_root / output_dir / LEDGER_NAME, role="evaluator exposure ledger", schema=LEDGER_SCHEMA, source_path=ledger_path),
                "lineage_manifest": lineage_ref,
            },
            "governance": {
                "selected_producer": None,
                "training_authorized": False,
                "formal_promotion": False,
                "generation_readiness": "not_ready",
                "sealed_confirmation_consumed": False,
            },
            "claim_boundary": {
                "development_q_distribution_compatibility_only": True,
                "amplitude_side_latent_modeling_only": True,
                "not_full_complex_OLS_reproduction": True,
                "image_generation": False,
                "semantic_generation": False,
                "optical_compatibility": False,
                "generator_readiness": False,
                "training_authorization": False,
                "population_stationarity": False,
            },
            "limitations": [
                "V2 5/5 means only that the current development contract passed; it does not establish overall stationarity.",
                "DATA-vs-MODEL uses 1000 vs 200 rows while MODEL-vs-MODEL reference uses 200 vs 200 rows.",
                "The biased empirical ED therefore has different finite-sample expectation scales in the two roles.",
                "The frozen reference q95 is an empirical development threshold, not a calibrated 95% confidence guarantee.",
                "The comparison evaluates only q-distribution compatibility.",
                "No image-generation, semantic-control, optical-OLS, full complex latent, generator-training, or readiness claim follows.",
            ],
            "sealed_confirmation": {
                "status": "RESERVED_NOT_MATERIALIZED",
                "consumed": False,
                "materialized": False,
                "values_loaded": False,
            },
            "not_executed": [
                "sealed confirmation loading or materialization",
                "image-space metrics",
                "optical propagation or evaluator",
                "generator",
                "CUDA",
                "optimizer",
                "backward",
                "training",
                "producer selection",
                "formal promotion",
            ],
        }
        report_path = _write_json(temporary, REPORT_NAME, report)
        _invalid(output_dir.parent.exists(), "N4 output parent disappeared")
        if output_dir.exists():
            raise FileExistsError(f"N4 output appeared during execution: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "valid_completed",
        "classification": classification,
        "development_supported_candidates": supported,
        "development_unsupported_candidates": unsupported,
        "selected_producer": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_consumed": False,
        "total_evaluator_calls": len(exposure_records),
        "registry": _relative(repo_root, registry_path),
        "registry_sha256": registry_sha256,
        "report": _relative(repo_root, output_dir / REPORT_NAME),
        "report_sha256": file_sha256(output_dir / REPORT_NAME),
        "lineage": _relative(repo_root, output_dir / LINEAGE_NAME),
        "lineage_sha256": file_sha256(output_dir / LINEAGE_NAME),
        "git_head": git["head"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
