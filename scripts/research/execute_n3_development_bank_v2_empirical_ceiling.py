"""Execute the frozen N3 development-bank V2 empirical ceiling once.

This CPU-only consumer evaluates one empirical complete-row bootstrap ceiling
on the replacement development bank.  The source rows, fixed OLS q
materialization, and five-fold assignment are all verified before q values are
loaded.  The consumer reuses the historical raw-q compatibility formulas; a
local shape adapter removes only the old equal-row-count precondition so that
the exact 1000-row heldout data can be compared with each required 200-row
bootstrap bank.  It does not fit or select a learnable producer, read sealed
confirmation values, invoke optics/evaluation, or train.
"""

from __future__ import annotations

import argparse
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
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import compare_n3_minimal_competing_producer_baselines_v1 as comparison
from scripts.research import define_n3_development_bank_v2 as source_definition
from scripts.research import define_n3_development_bank_v2_five_fold_protocol as fold_definition
from scripts.research import define_n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_v1 as historical_definition
from scripts.research import execute_n3_empirical_marginal_gaussian_copula_producer_v1 as compatibility


TASK_ID = "N3_DEVELOPMENT_BANK_V2_EMPIRICAL_CEILING"
GATE = TASK_ID
REPORT_SCHEMA = "n3_development_bank_v2_empirical_ceiling_report.v1"
REPORT_NAME = "n3_development_bank_v2_empirical_ceiling_report.v1.json"
EXPECTED_BRANCH = source_definition.EXPECTED_BRANCH
EXPECTED_STARTING_HEAD = source_definition.EXPECTED_STARTING_HEAD

SOURCE_MANIFEST = source_definition.OUTPUT_DIR / source_definition.ARTIFACT_NAME
SOURCE_MANIFEST_SHA256 = "4bc251af91b3a6cd0235fe113117443d871a564bd05634fa9e61b5ab0b36f812"
FOLD_MANIFEST = fold_definition.OUTPUT_DIR / fold_definition.ARTIFACT_NAME
FOLD_MANIFEST_SHA256 = "4898a0e477d583b9a9e60c83238e6d8524d43967efe24277a952089e8adc447e"
FOLD_ASSIGNMENT_SHA256 = "43ee0ce28e143fffdf2bcac98394625ccd766166ea717f23db66ed8770ef18a9"
Q_MATERIALIZATION_DIR = fold_definition.Q_MATERIALIZATION_DIR
Q_MATERIALIZATION_MANIFEST = fold_definition.Q_MATERIALIZATION_MANIFEST
Q_MATERIALIZATION_MANIFEST_SHA256 = "422d8c65b78bc459197ed46504399cb9c87ba8f7184eb2405e0faa85353995d8"
Q_ARTIFACT = fold_definition.Q_ARTIFACT
Q_ARTIFACT_SHA256 = "fe48c8fdb9bb2bf732076fabdf475288baa6f7d628ca10039a83bbda54fcfcbf"
Q_TENSOR_SHA256 = "d660fd8dc449725edab081d00c8f553f151029cddef9210360a7a6fad4281209"

HISTORICAL_CONTRACT_PATH = historical_definition.OUTPUT_DIR / historical_definition.CONTRACT_NAME
HISTORICAL_CONTRACT_SHA256 = "fc24ca4c863fded5f23dacc69b13bb16f44224b96d47888d62cb9a3f0f70a38d"
HISTORICAL_REPORT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_execution_v1_attempt_01/"
    "n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_report.v1.json"
)
HISTORICAL_REPORT_SHA256 = "c4557d44c8d1b40adeb2f362a4f81f0f28d9feb398428fbb1b13c035b043e12c"

REPORT_OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_empirical_ceiling_v1_attempt_01"
)

CLASSIFICATION_VALIDATED = "N3_DEVELOPMENT_BANK_V2_VALIDATED"
CLASSIFICATION_NOT_VALIDATED = "N3_DEVELOPMENT_BANK_V2_NOT_VALIDATED"
EMPIRICAL_FIT = "P_EMPIRICAL_FIT"
EMPIRICAL_FIT_FAMILY = "UNIFORM_EMPIRICAL_FIT_JOINT_ROW_BOOTSTRAP_V1"

CLASS_COUNT = source_definition.CLASS_COUNT
DIMENSION = 100
ROWS_PER_CLASS = source_definition.ROWS_PER_CLASS
POOL_COUNT = source_definition.POOL_COUNT
FOLD_COUNT = fold_definition.FOLD_COUNT
FIT_COUNT_PER_FOLD = fold_definition.FIT_COUNT
HELDOUT_COUNT_PER_FOLD = fold_definition.HELDOUT_COUNT
FIT_PER_CLASS_PER_FOLD = fold_definition.FIT_PER_CLASS
HELDOUT_PER_CLASS_PER_FOLD = fold_definition.HELDOUT_PER_CLASS

# These values intentionally match the earlier empirical-ceiling contract.
GENERATED_BANK_SIZE = historical_definition.GENERATED_BANK_SIZE
DATA_VS_MODEL_REPLICATES = historical_definition.DATA_VS_MODEL_REPLICATES
MODEL_VS_MODEL_REPLICATES = historical_definition.MODEL_VS_MODEL_REPLICATES
MODEL_REFERENCE_PAIRS = historical_definition.MODEL_REFERENCE_PAIRS
COMPATIBILITY_Q = historical_definition.COMPATIBILITY_Q
RNG_NAMESPACE = historical_definition.RNG_NAMESPACE
JOINT_METRICS = tuple(historical_definition.JOINT_METRICS)
MARGINAL_METRICS = tuple(historical_definition.MARGINAL_METRICS)
SCALAR_METRICS = tuple(historical_definition.SCALAR_METRICS)

SAFE_DIRTY_PATHS = set(source_definition.SAFE_DIRTY_PATHS) | {
    "scripts/research/execute_n3_development_bank_v2_empirical_ceiling.py",
    "tests/v2/test_n3_development_bank_v2.py",
}


class _InvalidHarnessError(RuntimeError):
    """A declared numerical or loaded-artifact boundary failed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _invalid(condition: bool, message: str) -> None:
    if not condition:
        raise _InvalidHarnessError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> list[Any]:
    _require(isinstance(value, list), f"{name} must be a list")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _read_json_exact(
    repo_root: Path,
    path: Path,
    expected_sha256: str,
    name: str,
) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{name} is missing: {target}")
    _require(file_sha256(target) == expected_sha256, f"{name} hash changed: {target}")
    value = json.loads(target.read_text(encoding="utf-8"))
    return dict(_mapping(value, name))


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


def _validate_governance_preflight(repo_root: Path) -> dict[str, Any]:
    state_path = repo_root / "docs/status/research_governance_state.json"
    _require(state_path.is_file(), "research governance state is missing")
    state = _mapping(json.loads(state_path.read_text(encoding="utf-8")), "research governance state")
    _require(
        state.get("active_objective") == "SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_NOT_SUPPORTED",
        "governance active objective changed before V2 ceiling execution",
    )
    aliases = _mapping(state.get("evidence_aliases"), "governance evidence aliases")
    previous = _mapping(aliases.get("N3_SOURCE_POSITION_BALANCED_EMPIRICAL_CEILING_V1"), "previous source-position authority")
    _require(
        previous.get("classification") == "SOURCE_POSITION_BALANCED_DEVELOPMENT_PROTOCOL_NOT_SUPPORTED",
        "previous source-position classification changed",
    )
    historical = _mapping(
        aliases.get("N3_EMPIRICAL_FIT_JOINT_BOOTSTRAP_CEILING_DIAGNOSTIC_V1"),
        "historical empirical ceiling authority",
    )
    _require(
        historical.get("classification") == "N3_FIT_HELDOUT_EMPIRICAL_CEILING_NOT_SUPPORTED",
        "historical empirical ceiling classification changed",
    )
    for name, item in (("previous source-position authority", previous), ("historical empirical ceiling authority", historical)):
        _require(item.get("training_authorized") is False, f"{name} training flag changed")
        _require(item.get("formal_promotion") is False, f"{name} promotion flag changed")
        _require(item.get("generation_readiness") == "not_ready", f"{name} readiness changed")
    return {
        "path": _relative(repo_root, state_path),
        "active_objective_before": state["active_objective"],
        "previous_classification": previous["classification"],
        "historical_classification": historical["classification"],
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }


def _validate_historical_ceiling(repo_root: Path) -> dict[str, Any]:
    contract = _read_json_exact(
        repo_root,
        HISTORICAL_CONTRACT_PATH,
        HISTORICAL_CONTRACT_SHA256,
        "historical empirical ceiling contract",
    )
    historical_definition.validate_contract_payload(contract)
    framework = historical_definition._validate_compatibility_framework(repo_root)
    report = _read_json_exact(
        repo_root,
        HISTORICAL_REPORT_PATH,
        HISTORICAL_REPORT_SHA256,
        "historical empirical ceiling report",
    )
    _require(report.get("schema") == "n3_empirical_fit_joint_bootstrap_ceiling_diagnostic_report.v1", "historical ceiling report schema changed")
    _require(report.get("status") == "valid_completed" and report.get("result") == "VALID_COMPLETED", "historical ceiling report is incomplete")
    _require(report.get("classification") == "N3_FIT_HELDOUT_EMPIRICAL_CEILING_NOT_SUPPORTED", "historical ceiling classification changed")
    aggregate = _mapping(report.get("aggregate"), "historical ceiling aggregate")
    for key, expected in {
        "fold_count": FOLD_COUNT,
        "compatible_fold_count": 3,
        "all_five_folds_required": True,
        "operationally_valid": True,
        "development_supported": False,
        "producer_selection": False,
    }.items():
        _require(aggregate.get(key) == expected, f"historical ceiling aggregate changed: {key}")
    empirical = _mapping(report.get("empirical_fit"), "historical empirical fit")
    _require(empirical.get("candidate") == EMPIRICAL_FIT, "historical empirical candidate changed")
    _require(empirical.get("family") == EMPIRICAL_FIT_FAMILY, "historical empirical family changed")
    _require(empirical.get("fold_count") == FOLD_COUNT and empirical.get("compatible_fold_count") == 3, "historical empirical fold result changed")
    fold_results = _sequence(empirical.get("fold_results"), "historical empirical fold results")
    _require(len(fold_results) == FOLD_COUNT, "historical empirical fold records are incomplete")
    _require([bool(_mapping(item, "historical fold result").get("compatibility")) for item in fold_results] == [False, True, False, True, True], "historical empirical fold pattern changed")
    settings = _mapping(report.get("execution_settings"), "historical ceiling settings")
    for key, expected in {
        "generated_bank_size": GENERATED_BANK_SIZE,
        "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
        "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
        "model_reference_pairs": [list(pair) for pair in MODEL_REFERENCE_PAIRS],
        "compatibility_q": COMPATIBILITY_Q,
        "complete_row_sampling": True,
        "with_replacement": True,
        "independent_dimension_sampling": False,
        "seed_retry": False,
        "best_bank_selection": False,
        "samplewise_normalization": False,
    }.items():
        _require(settings.get(key) == expected, f"historical ceiling setting changed: {key}")
    governance = _mapping(report.get("governance"), "historical ceiling governance")
    for key, expected in {
        "producer_selection": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items():
        _require(governance.get(key) == expected, f"historical ceiling governance changed: {key}")
    return {
        "contract": {
            "path": _relative(repo_root, repo_root / HISTORICAL_CONTRACT_PATH),
            "schema": contract["schema"],
            "sha256": HISTORICAL_CONTRACT_SHA256,
            "classification": contract["classification"],
            "claim_classification": contract["claim_classification"],
        },
        "report": {
            "path": _relative(repo_root, repo_root / HISTORICAL_REPORT_PATH),
            "schema": report["schema"],
            "sha256": HISTORICAL_REPORT_SHA256,
            "classification": report["classification"],
            "compatible_fold_count": aggregate["compatible_fold_count"],
            "fold_pattern": [False, True, False, True, True],
        },
        "compatibility_framework": dict(framework),
        "semantics_reused": {
            "generated_bank_size": GENERATED_BANK_SIZE,
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "model_reference_pairs": [list(pair) for pair in MODEL_REFERENCE_PAIRS],
            "compatibility_q": COMPATIBILITY_Q,
            "rng_namespace": RNG_NAMESPACE,
            "complete_row_sampling": True,
            "with_replacement": True,
            "independent_dimension_sampling": False,
            "seed_retry": False,
            "best_bank_selection": False,
            "raw_q_energy_distance": True,
            "all_eight_scalar_guardrails": list(SCALAR_METRICS),
        },
    }


def _validate_source_and_fold(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], torch.Tensor, dict[str, Any]]:
    source, source_rows = fold_definition._load_source(repo_root)
    _require(file_sha256(repo_root / SOURCE_MANIFEST) == SOURCE_MANIFEST_SHA256, "V2 source manifest hash changed")
    fold = _read_json_exact(repo_root, FOLD_MANIFEST, FOLD_MANIFEST_SHA256, "V2 five-fold manifest")
    fold_definition.validate_fold_manifest_payload(fold)
    _require(fold.get("bank_semantic_sha256") == source["bank_semantic_sha256"], "fold bank semantic identity changed")
    _require(fold["protocol"]["assignment_sha256"] == FOLD_ASSIGNMENT_SHA256, "fold assignment identity changed")
    source_ref = _mapping(fold.get("source_manifest"), "fold source reference")
    _require(source_ref.get("path") == _relative(repo_root, repo_root / SOURCE_MANIFEST), "fold source path changed")
    _require(source_ref.get("sha256") == SOURCE_MANIFEST_SHA256, "fold source hash changed")
    _require(source_ref.get("row_manifest_sha256") == source["identity"]["row_manifest_sha256"], "fold source row identity changed")
    fold_rows = [dict(_mapping(row, "fold row")) for row in fold["rows"]]
    _require(len(fold_rows) == len(source_rows) == POOL_COUNT, "fold/source row count changed")
    for position, (source_row, fold_row) in enumerate(zip(source_rows, fold_rows, strict=True)):
        for key in ("bank_row_position", "source_index", "label", "stable_source_row_identity", "selected_within_class_rank"):
            _require(fold_row.get(key) == source_row.get(key), f"fold/source row identity changed at {position}: {key}")
    fold_ids = torch.tensor([int(row["heldout_fold"]) for row in fold_rows], dtype=torch.int64).contiguous()
    _require(tuple(fold_ids.shape) == (POOL_COUNT,), "fold identity shape changed")
    q_meta = fold_definition._load_q_metadata(repo_root, source)
    _require(q_meta["manifest_sha256"] == Q_MATERIALIZATION_MANIFEST_SHA256, "q materialization manifest identity changed")
    _require(q_meta["artifact_sha256"] == Q_ARTIFACT_SHA256 and q_meta["tensor_sha256"] == Q_TENSOR_SHA256, "q artifact identity changed")
    return source, fold, fold_rows, fold_ids, q_meta


def _validate_preflight(repo_root: Path) -> dict[str, Any]:
    # These checks are metadata/hash-only.  No development or sealed q tensor
    # is loaded until every authority and boundary below has passed.
    source, fold, rows, fold_ids, q_meta = _validate_source_and_fold(repo_root)
    historical = _validate_historical_ceiling(repo_root)
    old_authorities = comparison._verify_authorities(repo_root)
    boundaries = comparison._verify_historical_boundaries(repo_root)
    _require(boundaries["sealed_confirmation"]["status"] == "RESERVED_NOT_MATERIALIZED", "sealed confirmation is no longer reserved")
    _require(boundaries["sealed_confirmation"]["sample_values_loaded"] is False, "sealed confirmation values entered preflight")
    current_governance = _validate_governance_preflight(repo_root)
    return {
        "source": source,
        "fold": fold,
        "rows": rows,
        "fold_ids": fold_ids,
        "q_meta": q_meta,
        "historical": historical,
        "old_authorities": old_authorities,
        "boundaries": boundaries,
        "current_governance": current_governance,
    }


def _load_q_after_preflight(
    repo_root: Path,
    preflight: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], dict[str, Any]]:
    payload = safe_torch_load(
        repo_root / Q_ARTIFACT,
        expected_sha256=Q_ARTIFACT_SHA256,
        map_location="cpu",
    )
    q_payload = _mapping(payload, "V2 q artifact")
    _invalid(q_payload.get("schema") == "n3_development_bank_v2_q.v1", "q artifact schema changed")
    _invalid(q_payload.get("status") == "VALID_COMPLETED", "q artifact is incomplete")
    _invalid(q_payload.get("semantic_authority_id") == source_definition.SEMANTIC_AUTHORITY_ID, "q semantic authority changed")
    _invalid(q_payload.get("q_coordinate") == "R2_AMPLITUDE", "q coordinate changed")
    _invalid(q_payload.get("q_definition") == "q=abs(c/sqrt(s))=abs(R3_COMPLEX)", "q definition changed")
    _invalid(q_payload.get("source_manifest_sha256") == SOURCE_MANIFEST_SHA256, "q source manifest identity changed")
    _invalid(q_payload.get("bank_semantic_sha256") == preflight["source"]["bank_semantic_sha256"], "q bank semantic identity changed")
    _invalid(q_payload.get("source_row_manifest_sha256") == preflight["source"]["identity"]["row_manifest_sha256"], "q source row manifest identity changed")

    q = q_payload.get("q")
    source_indices = q_payload.get("source_indices")
    labels = q_payload.get("labels")
    _invalid(torch.is_tensor(q) and q.dtype == torch.float64 and tuple(q.shape) == (POOL_COUNT, DIMENSION), "loaded q tensor shape or dtype changed")
    _invalid(torch.is_tensor(source_indices) and source_indices.dtype == torch.int64 and tuple(source_indices.shape) == (POOL_COUNT,), "loaded source-index tensor changed")
    _invalid(torch.is_tensor(labels) and labels.dtype == torch.int64 and tuple(labels.shape) == (POOL_COUNT,), "loaded label tensor changed")
    q = q.detach().cpu().contiguous()
    source_indices = source_indices.detach().cpu().contiguous()
    labels = labels.detach().cpu().contiguous()
    _invalid(bool(torch.isfinite(q).all()) and bool(q.ge(0.0).all()), "loaded q values are invalid")
    _invalid(tensor_sha256(q) == Q_TENSOR_SHA256, "loaded q tensor identity changed")

    expected_indices = torch.tensor([int(row["source_index"]) for row in preflight["rows"]], dtype=torch.int64)
    expected_labels = torch.tensor([int(row["label"]) for row in preflight["rows"]], dtype=torch.int64)
    _invalid(torch.equal(source_indices, expected_indices), "loaded source indices differ from frozen V2 rows")
    _invalid(torch.equal(labels, expected_labels), "loaded labels differ from frozen V2 rows")
    _invalid(tensor_sha256(source_indices) == preflight["source"]["identity"]["selected_source_indices_sha256"], "loaded source-index hash changed")
    _invalid(tensor_sha256(labels) == preflight["source"]["identity"]["selected_labels_sha256"], "loaded label hash changed")
    identities_value = q_payload.get("source_row_identity")
    _invalid(isinstance(identities_value, list) and len(identities_value) == POOL_COUNT, "q source row identities are incomplete")
    identities = [str(value) for value in identities_value]
    expected_identities = [str(row["stable_source_row_identity"]) for row in preflight["rows"]]
    _invalid(identities == expected_identities, "loaded stable source-row identities changed")
    _invalid(canonical_sha256(identities) == q_payload.get("source_row_identity_sha256"), "q source-row identity hash is stale")
    _invalid(q_payload.get("q_tensor_sha256") == Q_TENSOR_SHA256, "q payload tensor identity changed")
    _invalid(q_payload.get("source_indices_sha256") == tensor_sha256(source_indices), "q payload source-index identity changed")
    _invalid(q_payload.get("labels_sha256") == tensor_sha256(labels), "q payload label identity changed")
    execution = _mapping(q_payload.get("execution"), "q execution")
    _invalid(execution.get("device") == "cpu", "q execution device changed")
    _invalid(execution.get("sealed_confirmation_values_loaded") is False, "q materialization loaded sealed values")
    _invalid(execution.get("model_state_unchanged") is True, "q materialization changed fixed model state")
    governance = _mapping(q_payload.get("governance"), "q governance")
    for key, expected in {
        "producer_evaluated": False,
        "producer_selection": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items():
        _invalid(governance.get(key) == expected, f"q governance changed: {key}")
    loaded_identity = {
        "count": POOL_COUNT,
        "dimension": DIMENSION,
        "coordinate": "R2_AMPLITUDE",
        "q_r2_amplitude_sha256": tensor_sha256(q),
        "source_indices_sha256": tensor_sha256(source_indices),
        "labels_sha256": tensor_sha256(labels),
        "source_row_identity_sha256": canonical_sha256(identities),
        "class_counts": torch.bincount(labels, minlength=CLASS_COUNT).tolist(),
        "q_values_loaded_after_preflight": True,
    }
    _invalid(loaded_identity["class_counts"] == [ROWS_PER_CLASS] * CLASS_COUNT, "loaded V2 class balance changed")
    return q, source_indices, labels, identities, loaded_identity


def _seed(role: str, fold: int, replicate: int) -> int:
    material = f"{RNG_NAMESPACE}|{role}|fold={fold}|replicate={replicate}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") & ((1 << 63) - 1)


def _numeric_summary(value: torch.Tensor) -> dict[str, float]:
    view = value.detach().to(dtype=torch.float64).reshape(-1)
    _invalid(view.numel() > 0 and bool(torch.isfinite(view).all()), "numeric summary input is invalid")
    return {"min": float(view.min()), "mean": float(view.mean()), "max": float(view.max())}


def _sample_empirical_bank(
    fit_q: torch.Tensor,
    fit_source_indices: torch.Tensor,
    *,
    role: str,
    fold: int,
    replicate: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _invalid(tuple(fit_q.shape) == (FIT_COUNT_PER_FOLD, DIMENSION), "fit q shape is invalid")
    _invalid(tuple(fit_source_indices.shape) == (FIT_COUNT_PER_FOLD,), "fit source-index shape is invalid")
    _invalid(bool(torch.isfinite(fit_q).all()) and bool(fit_q.ge(0.0).all()), "fit q is invalid")
    seed = _seed(role, fold, replicate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sampled_fit_local_indices = torch.randint(
        low=0,
        high=FIT_COUNT_PER_FOLD,
        size=(GENERATED_BANK_SIZE,),
        generator=generator,
        device="cpu",
        dtype=torch.int64,
    )
    bank = fit_q.index_select(0, sampled_fit_local_indices).contiguous()
    sampled_source_indices = fit_source_indices.index_select(0, sampled_fit_local_indices).contiguous()
    _invalid(tuple(bank.shape) == (GENERATED_BANK_SIZE, DIMENSION), "bootstrap bank shape changed")
    _invalid(bool(torch.isfinite(bank).all()) and bool(bank.ge(0.0).all()), "bootstrap bank violates finite/nonnegative support")
    unique_count = int(torch.unique(sampled_fit_local_indices).numel())
    duplicate_count = GENERATED_BANK_SIZE - unique_count
    duplicate_rate = duplicate_count / float(GENERATED_BANK_SIZE)
    _invalid(0.0 <= duplicate_rate <= 1.0, "duplicate-row rate is outside its defined range")
    return bank, {
        "role": role,
        "fold": fold,
        "replicate": replicate,
        "seed": seed,
        "rng": "torch.Generator(device='cpu')",
        "sampling_unit": "complete q row",
        "with_replacement": True,
        "independent_dimension_sampling": False,
        "fit_row_count": FIT_COUNT_PER_FOLD,
        "bank_size": GENERATED_BANK_SIZE,
        "sampled_fit_local_indices_sha256": tensor_sha256(sampled_fit_local_indices),
        "sampled_source_indices_sha256": tensor_sha256(sampled_source_indices),
        "unique_fit_row_count": unique_count,
        "duplicate_row_count": duplicate_count,
        "duplicate_row_rate": float(duplicate_rate),
        "q_finite": bool(torch.isfinite(bank).all()),
        "q_nonnegative": bool(bank.ge(0.0).all()),
        "q_sha256": tensor_sha256(bank),
        "q_summary": _numeric_summary(bank),
    }


def _pair_metrics_raw_q(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    """Apply the historical pair metrics while allowing unequal sample counts.

    The frozen historical implementation requires equal row counts because
    its original data and model banks were both 200 rows.  V2 explicitly
    requires the exact 1000-row heldout fold against 200-row bootstrap banks,
    so this adapter preserves every metric formula and removes only that
    computational precondition.  The covariance, energy-distance, and
    marginal calculations remain the registered implementations.
    """
    left = first.to(dtype=torch.float64).contiguous()
    right = second.to(dtype=torch.float64).contiguous()
    _invalid(
        left.ndim == 2 and right.ndim == 2 and left.shape[1] == right.shape[1],
        "raw-q metric inputs are invalid",
    )
    left_cov = comparison._covariance(left)
    right_cov = comparison._covariance(right)
    right_cov_norm = right_cov.square().sum()
    _invalid(bool(torch.isfinite(right_cov_norm)) and bool(right_cov_norm.gt(0.0)), "raw-q reference covariance is zero")
    left_norm = left.norm(dim=1).unsqueeze(1)
    right_norm = right.norm(dim=1).unsqueeze(1)
    left_energy = left.square().sum(dim=1).unsqueeze(1)
    right_energy = right.square().sum(dim=1).unsqueeze(1)
    base = {
        "energy_distance": comparison.validated_energy_distance(left, right),
        "covariance_relative_frobenius": float((left_cov - right_cov).square().sum() / right_cov_norm),
        "norm_energy_distance": comparison.validated_energy_distance(left_norm, right_norm),
        "squared_norm_energy_distance": comparison.validated_energy_distance(left_energy, right_energy),
        "marginal_mean_l2": float((left.mean(dim=0) - right.mean(dim=0)).norm()),
        "marginal_std_l2": float((left.std(dim=0, unbiased=True) - right.std(dim=0, unbiased=True)).norm()),
    }
    per_coordinate = [
        comparison.validated_energy_distance(
            left[:, coordinate : coordinate + 1],
            right[:, coordinate : coordinate + 1],
        )
        for coordinate in range(DIMENSION)
    ]
    _invalid(
        all(torch.isfinite(torch.tensor(value, dtype=torch.float64)) for value in per_coordinate),
        "per-coordinate marginal energy is non-finite",
    )
    base["per_coordinate_marginal_energy_distance"] = [float(value) for value in per_coordinate]
    base["marginal_energy_distance_mean"] = float(sum(per_coordinate) / DIMENSION)
    base["marginal_energy_distance_max"] = float(max(per_coordinate))
    return base


def _validate_metrics(metrics: Mapping[str, Any], label: str) -> None:
    for name in SCALAR_METRICS:
        value = metrics.get(name)
        _invalid(
            isinstance(value, (int, float))
            and bool(torch.isfinite(torch.tensor(float(value), dtype=torch.float64))),
            f"{label} metric is non-finite: {name}",
        )
        _invalid(float(value) >= 0.0, f"{label} metric is negative: {name}")
    vector = metrics.get("per_coordinate_marginal_energy_distance")
    _invalid(isinstance(vector, list) and len(vector) == DIMENSION, f"{label} marginal vector shape changed")
    _invalid(
        all(
            float(value) >= 0.0
            and bool(torch.isfinite(torch.tensor(float(value), dtype=torch.float64)))
            for value in vector
        ),
        f"{label} marginal vector is invalid",
    )


def _fold_interpretation(compatible: Mapping[str, bool]) -> str:
    marginal = all(bool(compatible[name]) for name in MARGINAL_METRICS)
    joint = all(bool(compatible[name]) for name in JOINT_METRICS)
    if marginal and joint:
        return "NO_RESIDUAL_FAILURE"
    if marginal and not joint:
        return "MARGINALS_PASS_JOINT_GUARDRAIL_REMAINS"
    if not marginal and joint:
        return "JOINT_PASS_MARGINAL_GUARDRAIL_REMAINS"
    return "MARGINAL_AND_JOINT_GUARDRAILS_REMAIN"


def _evaluate_fold(
    q: torch.Tensor,
    source_indices: torch.Tensor,
    labels: torch.Tensor,
    fold_ids: torch.Tensor,
    fold: int,
    protocol_fold: Mapping[str, Any],
) -> dict[str, Any]:
    print(json.dumps({"stage": "n3_v2_empirical_ceiling", "fold": fold, "status": "started"}), flush=True)
    heldout_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1)
    fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1)
    _invalid(int(heldout_positions.numel()) == HELDOUT_COUNT_PER_FOLD, "held-out fold count changed")
    _invalid(int(fit_positions.numel()) == FIT_COUNT_PER_FOLD, "fit fold count changed")
    union = torch.cat([fit_positions, heldout_positions])
    _invalid(torch.equal(torch.sort(union).values, torch.arange(POOL_COUNT, dtype=torch.int64)), "fit/held-out rows are not a complete disjoint partition")
    heldout_labels = labels.index_select(0, heldout_positions)
    fit_labels = labels.index_select(0, fit_positions)
    _invalid(
        torch.equal(
            torch.bincount(heldout_labels, minlength=CLASS_COUNT),
            torch.full((CLASS_COUNT,), HELDOUT_PER_CLASS_PER_FOLD, dtype=torch.int64),
        ),
        "held-out class balance changed",
    )
    _invalid(
        torch.equal(
            torch.bincount(fit_labels, minlength=CLASS_COUNT),
            torch.full((CLASS_COUNT,), FIT_PER_CLASS_PER_FOLD, dtype=torch.int64),
        ),
        "fit class balance changed",
    )
    fit_q = q.index_select(0, fit_positions).contiguous()
    fit_source_indices = source_indices.index_select(0, fit_positions).contiguous()
    heldout = q.index_select(0, heldout_positions).contiguous()

    data_banks: list[torch.Tensor] = []
    data_sampling: list[dict[str, Any]] = []
    for replicate in range(DATA_VS_MODEL_REPLICATES):
        bank, record = _sample_empirical_bank(
            fit_q,
            fit_source_indices,
            role="data-vs-model",
            fold=fold,
            replicate=replicate,
        )
        data_banks.append(bank)
        data_sampling.append(record)
    model_banks: list[torch.Tensor] = []
    model_sampling: list[dict[str, Any]] = []
    for replicate in range(MODEL_VS_MODEL_REPLICATES):
        bank, record = _sample_empirical_bank(
            fit_q,
            fit_source_indices,
            role="model-vs-model",
            fold=fold,
            replicate=replicate,
        )
        model_banks.append(bank)
        model_sampling.append(record)

    data_metrics = [_pair_metrics_raw_q(heldout, bank) for bank in data_banks]
    reference_pairs = [
        _pair_metrics_raw_q(model_banks[left], model_banks[right])
        for left, right in MODEL_REFERENCE_PAIRS
    ]
    for index, metrics in enumerate(data_metrics):
        _validate_metrics(metrics, f"fold {fold} data-vs-model {index}")
    for index, metrics in enumerate(reference_pairs):
        _validate_metrics(metrics, f"fold {fold} model-vs-model {index}")
    data_mean = compatibility._mean_metrics(data_metrics)
    reference_q95 = compatibility._q95_metrics(reference_pairs)
    _validate_metrics(data_mean, f"fold {fold} data mean")
    _validate_metrics(reference_q95, f"fold {fold} model reference q95")
    compatibility_components = {
        name: float(data_mean[name]) <= float(reference_q95[name])
        for name in SCALAR_METRICS
    }
    per_coordinate_checks = [
        float(data_mean["per_coordinate_marginal_energy_distance"][index])
        <= float(reference_q95["per_coordinate_marginal_energy_distance"][index])
        for index in range(DIMENSION)
    ]
    marginal_compatible = all(compatibility_components[name] for name in MARGINAL_METRICS)
    joint_compatible = all(compatibility_components[name] for name in JOINT_METRICS)
    result = {
        "fold": fold,
        "operationally_valid": True,
        "protocol_fold": dict(protocol_fold),
        "fit_heldout_disjoint": True,
        "class_counts": {
            "heldout": torch.bincount(heldout_labels, minlength=CLASS_COUNT).tolist(),
            "fit": torch.bincount(fit_labels, minlength=CLASS_COUNT).tolist(),
        },
        "fit_summary": {
            "fit_rows": FIT_COUNT_PER_FOLD,
            "heldout_rows": HELDOUT_COUNT_PER_FOLD,
            "fit_q_sha256": tensor_sha256(fit_q),
            "heldout_q_sha256": tensor_sha256(heldout),
            "fit_source_indices_sha256": tensor_sha256(fit_source_indices),
            "heldout_source_indices_sha256": tensor_sha256(source_indices.index_select(0, heldout_positions)),
        },
        "data_sampling": data_sampling,
        "model_sampling": model_sampling,
        "sampling_all_finite": True,
        "sampling_all_nonnegative": True,
        "data_vs_model_replicates": data_metrics,
        "data_vs_model_mean": data_mean,
        "model_vs_model_reference_pairs": reference_pairs,
        "model_vs_model_reference_q95": reference_q95,
        "primary": {
            "data_vs_model_mean_energy_distance": float(data_mean["energy_distance"]),
            "model_vs_model_q95_energy_distance": float(reference_q95["energy_distance"]),
        },
        "guardrails": {
            "joint": list(JOINT_METRICS),
            "marginal": list(MARGINAL_METRICS),
            "all_eight_scalar_metrics": list(SCALAR_METRICS),
            "compatibility_components": compatibility_components,
        },
        "compatibility_rule": "all eight fixed-replicate mean DATA_VS_MODEL raw-q metrics <= own fold-local q95 MODEL_VS_MODEL reference metrics",
        "compatibility_components": compatibility_components,
        "compatibility": all(compatibility_components.values()),
        "per_coordinate_marginal_q95_checks": per_coordinate_checks,
        "per_coordinate_vector_role": "descriptive only; not a compatibility gate",
        "marginal_compatible": marginal_compatible,
        "joint_compatible": joint_compatible,
        "marginal_vs_joint_interpretation": _fold_interpretation(compatibility_components),
    }
    print(
        json.dumps(
            {
                "stage": "n3_v2_empirical_ceiling",
                "fold": fold,
                "status": "completed",
                "compatible": bool(result["compatibility"]),
                "data_energy_mean": result["primary"]["data_vs_model_mean_energy_distance"],
                "reference_energy_q95": result["primary"]["model_vs_model_q95_energy_distance"],
            }
        ),
        flush=True,
    )
    return result


def _empirical_result(fold_results: list[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(fold_results) == FOLD_COUNT, "all five V2 folds were not evaluated")
    operationally_valid = all(bool(item["operationally_valid"]) for item in fold_results)
    _require(operationally_valid, "a V2 fold is not operationally valid")
    compatible_fold_count = sum(bool(item["compatibility"]) for item in fold_results)
    return {
        "candidate": EMPIRICAL_FIT,
        "family": EMPIRICAL_FIT_FAMILY,
        "fit_definition": "uniform empirical distribution over each V2 fold's exact 4000 fitting q rows",
        "joint_structure_preserved": True,
        "operationally_valid": operationally_valid,
        "compatible_fold_count": compatible_fold_count,
        "fold_count": FOLD_COUNT,
        "development_supported": compatible_fold_count == FOLD_COUNT,
        "support_rule": "all five V2 folds must be operationally valid and complete-compatible",
        "fold_results": [dict(item) for item in fold_results],
    }


def _interpretation(classification: str, compatible_fold_count: int) -> dict[str, str]:
    if classification == CLASSIFICATION_VALIDATED:
        return {
            "fact": f"FACT: the replacement N3 development bank passed the frozen empirical complete-row ceiling on {compatible_fold_count}/{FOLD_COUNT} folds.",
            "model_assumption": "MODEL_ASSUMPTION: within the observed 5000-row development domain, the replacement bank is stable enough for this empirical ceiling diagnostic; no producer family is selected and no training or generation authority follows.",
        }
    return {
        "fact": f"FACT: the replacement N3 development bank passed the frozen empirical complete-row ceiling on {compatible_fold_count}/{FOLD_COUNT} folds, so the V2 development-bank validation gate is not validated.",
        "model_assumption": "MODEL_ASSUMPTION: the replacement bank remains insufficiently stable for the defined empirical ceiling under this finite-domain diagnostic; this is not proof that no population-level producer exists.",
    }


def _valid_report(
    repo_root: Path,
    git_before: Mapping[str, Any],
    preflight: Mapping[str, Any],
    loaded_identity: Mapping[str, Any],
    empirical: Mapping[str, Any],
) -> dict[str, Any]:
    compatible_fold_count = int(empirical["compatible_fold_count"])
    classification = CLASSIFICATION_VALIDATED if compatible_fold_count == FOLD_COUNT else CLASSIFICATION_NOT_VALIDATED
    source = _mapping(preflight["source"], "preflight source")
    fold = _mapping(preflight["fold"], "preflight fold")
    historical = _mapping(preflight["historical"], "historical authority")
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "VALID_COMPLETED",
        "classification": classification,
        "source_authority": {
            "path": _relative(repo_root, repo_root / SOURCE_MANIFEST),
            "sha256": SOURCE_MANIFEST_SHA256,
            "schema": source["schema"],
            "semantic_authority_id": source["semantic_authority_id"],
            "bank_semantic_sha256": source["bank_semantic_sha256"],
            "row_manifest_sha256": source["identity"]["row_manifest_sha256"],
            "selected_source_indices_sha256": source["identity"]["selected_source_indices_sha256"],
            "selected_labels_sha256": source["identity"]["selected_labels_sha256"],
            "count": POOL_COUNT,
            "rows_per_class": ROWS_PER_CLASS,
            "class_counts": source["identity"]["class_counts"],
            "eligible_count": source["source_domain"]["eligible_count"],
            "eligible_count_by_class": [
                item["eligible_count"] for item in source["selection"]["coverage"]["per_class"]
            ],
            "selection_rule": source["selection"]["formula"],
            "q_values_used_for_selection": False,
            "ols_values_used_for_selection": False,
            "image_values_used_for_selection": False,
            "rng_used_for_selection": False,
            "exclusions": source["exclusions"],
            "sealed_reservation_untouched": True,
        },
        "q_materialization_authority": {
            "manifest_path": _relative(repo_root, repo_root / Q_MATERIALIZATION_MANIFEST),
            "manifest_sha256": Q_MATERIALIZATION_MANIFEST_SHA256,
            "artifact_path": _relative(repo_root, repo_root / Q_ARTIFACT),
            "artifact_sha256": Q_ARTIFACT_SHA256,
            "tensor_sha256": Q_TENSOR_SHA256,
            "schema": "n3_development_bank_v2_q.v1",
            "coordinate": "R2_AMPLITUDE",
            "definition": "q=abs(c/sqrt(s))=abs(R3_COMPLEX)",
            "device": "cpu",
            "q_dtype": "torch.float64",
            "model_state_unchanged": True,
            "selected_development_rows_only": True,
            "sealed_confirmation_values_loaded": False,
        },
        "fold_protocol": {
            "path": _relative(repo_root, repo_root / FOLD_MANIFEST),
            "sha256": FOLD_MANIFEST_SHA256,
            "schema": fold["schema"],
            "bank_semantic_sha256": fold["bank_semantic_sha256"],
            "assignment_sha256": fold["protocol"]["assignment_sha256"],
            "algorithm": fold["protocol"]["algorithm"],
            "heldout_rule": fold["protocol"]["heldout_rule"],
            "fold_count": FOLD_COUNT,
            "fit_count_per_fold": FIT_COUNT_PER_FOLD,
            "heldout_count_per_fold": HELDOUT_COUNT_PER_FOLD,
            "fit_per_class_per_fold": FIT_PER_CLASS_PER_FOLD,
            "heldout_per_class_per_fold": HELDOUT_PER_CLASS_PER_FOLD,
            "fold_rows_sha256": fold["identity"]["fold_rows_sha256"],
            "folds": fold["fold_identity"]["folds"],
            "q_values_used_to_define_folds": False,
        },
        "historical_semantics": dict(historical),
        "git": {
            "branch": git_before["branch"],
            "starting_head": git_before["head"],
            "ending_head": _git_state(repo_root)["head"],
            "dirty_paths": git_before["dirty_paths"],
            "worktree_clean": git_before["worktree_clean"],
        },
        "input_verification": {
            "source_manifest_verified_before_q_load": True,
            "fold_manifest_verified_before_q_load": True,
            "q_file_hash_verified_before_q_load": True,
            "historical_contract_verified_before_q_load": True,
            "historical_report_verified_before_q_load": True,
            "compatibility_implementation_verified_before_q_load": True,
            "old_1000_row_authority_unchanged": True,
            "sealed_confirmation_values_loaded": False,
            "sealed_confirmation_status": "RESERVED_NOT_MATERIALIZED",
            "q_values_loaded_after_all_preflight": True,
        },
        "development_lineage": {
            "pool": dict(loaded_identity),
            "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
            "fit_per_class_per_fold": FIT_PER_CLASS_PER_FOLD,
            "heldout_per_class_per_fold": HELDOUT_PER_CLASS_PER_FOLD,
            "assignment_sha256": fold["protocol"]["assignment_sha256"],
            "full_bank_covered_once_per_fold": True,
            "fit_heldout_disjoint_per_fold": True,
        },
        "execution_settings": {
            "device": "cpu",
            "dtype": "float64 q; int64 row indices",
            "rng_namespace": RNG_NAMESPACE,
            "rng_generator": "torch.Generator(device='cpu')",
            "seed_derivation": "sha256(namespace|role|fold|replicate) first 64 bits masked to signed 63-bit",
            "fold_count": FOLD_COUNT,
            "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
            "fit_per_class_per_fold": FIT_PER_CLASS_PER_FOLD,
            "heldout_per_class_per_fold": HELDOUT_PER_CLASS_PER_FOLD,
            "generated_bank_size": GENERATED_BANK_SIZE,
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "model_reference_pairs": [list(pair) for pair in MODEL_REFERENCE_PAIRS],
            "compatibility_q": COMPATIBILITY_Q,
            "raw_q_energy_distance": True,
            "complete_row_sampling": True,
            "with_replacement": True,
            "independent_dimension_sampling": False,
            "seed_retry": False,
            "best_bank_selection": False,
            "samplewise_normalization": False,
            "guardrails": list(SCALAR_METRICS),
            "per_coordinate_vector_role": "descriptive only; not a compatibility gate",
            "unequal_sample_shape_adapter": "historical pair-metric formulas with only equal-row-count precondition removed for 1000-row heldout versus 200-row bank",
        },
        "empirical_ceiling": {
            **dict(empirical),
            "bootstrap_semantics_reused_from_historical_authority": True,
            "historical_fit_rows_per_fold": historical_definition.FIT_COUNT_PER_FOLD,
            "replacement_fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "no_dimensionwise_resampling": True,
            "no_seed_retry": True,
            "no_best_bank_selection": True,
            "no_learnable_producer": True,
            "q_values_are_raw_r2_amplitude": True,
        },
        "aggregate": {
            "classification": classification,
            "fold_count": FOLD_COUNT,
            "operationally_valid": bool(empirical["operationally_valid"]),
            "compatible_fold_count": compatible_fold_count,
            "compatible_fold_pattern": [bool(item["compatibility"]) for item in empirical["fold_results"]],
            "all_five_folds_required": True,
            "development_supported": compatible_fold_count == FOLD_COUNT,
            "duplicate_row_rate_descriptive_only": True,
            "producer_evaluated": False,
            "producer_selection": False,
        },
        "interpretation": _interpretation(classification, compatible_fold_count),
        "claim_boundary": {
            "diagnostic_only": True,
            "development_domain_only": True,
            "empirical_fit_ceiling": True,
            "not_a_producer": True,
            "not_confirmation_evidence": True,
            "not_novel_generation": True,
            "population_truth": False,
        },
        "governance": {
            "diagnostic_executed": True,
            "producer_evaluated": False,
            "producer_selection": False,
            "selected_producer": None,
            "sealed_confirmation_consumed": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "sealed_confirmation": {
            "semantic_authority_id": "N3_SEALED_CONFIRMATION_SOURCE_V1",
            "status": "RESERVED_NOT_MATERIALIZED",
            "consumed": False,
            "modified": False,
            "values_loaded": False,
        },
        "not_executed": [
            "learnable producer family",
            "producer selection",
            "final all-5000 fit",
            "sealed confirmation",
            "complex OLS",
            "decoder/evaluator",
            "CUDA",
            "training",
            "generation",
            "N4/N5/N6",
        ],
        "artifact_policy": {
            "write_once": True,
            "immutable_after_publish": True,
            "bootstrap_tensors_embedded": False,
            "existing_authorities_modified": False,
            "old_1000_authority_modified": False,
            "sealed_confirmation_modified": False,
        },
    }


def _blocked_report(
    repo_root: Path,
    git_before: Mapping[str, Any],
    preflight: Mapping[str, Any],
    error: _InvalidHarnessError,
) -> dict[str, Any]:
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "blocked",
        "result": "BLOCKED",
        "classification": None,
        "blocker": {
            "type": "INVALID_HARNESS",
            "message": str(error),
            "action": "stop before scientific compatibility classification",
        },
        "source_authority": {
            "path": _relative(repo_root, repo_root / SOURCE_MANIFEST),
            "sha256": SOURCE_MANIFEST_SHA256,
        },
        "fold_protocol": {
            "path": _relative(repo_root, repo_root / FOLD_MANIFEST),
            "sha256": FOLD_MANIFEST_SHA256,
            "assignment_sha256": preflight["fold"]["protocol"]["assignment_sha256"],
        },
        "q_materialization_authority": {
            "manifest_path": _relative(repo_root, repo_root / Q_MATERIALIZATION_MANIFEST),
            "manifest_sha256": Q_MATERIALIZATION_MANIFEST_SHA256,
            "artifact_path": _relative(repo_root, repo_root / Q_ARTIFACT),
            "artifact_sha256": Q_ARTIFACT_SHA256,
            "tensor_sha256": Q_TENSOR_SHA256,
        },
        "git": {
            "branch": git_before["branch"],
            "starting_head": git_before["head"],
            "ending_head": _git_state(repo_root)["head"],
            "dirty_paths": git_before["dirty_paths"],
        },
        "scientific_classification_emitted": False,
        "empirical_ceiling": None,
        "governance": {
            "diagnostic_executed": False,
            "producer_evaluated": False,
            "producer_selection": False,
            "selected_producer": None,
            "sealed_confirmation_consumed": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
    }


def _write_report(repo_root: Path, output_dir: Path, report: Mapping[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once report output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(
            temporary / REPORT_NAME,
            report,
            artifact_kind="N3 V2 empirical ceiling report",
        )
        if output_dir.exists():
            raise FileExistsError(f"write-once report output appeared during execution: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / REPORT_NAME


def execute(repo_root: Path, output_dir: Path = REPORT_OUTPUT_DIR) -> dict[str, Any]:
    git_before = _git_state(repo_root)
    _require(git_before["branch"] == EXPECTED_BRANCH, "current branch is not the N3 authority branch")
    _require(git_before["head"] == EXPECTED_STARTING_HEAD, "current HEAD changed from frozen authority base")
    unexpected_dirty = [path for path in git_before["dirty_paths"] if path not in SAFE_DIRTY_PATHS]
    _require(not unexpected_dirty, f"unrelated local changes invalidate provenance: {unexpected_dirty}")
    _require(not output_dir.exists(), f"write-once V2 empirical ceiling output already exists: {output_dir}")

    preflight = _validate_preflight(repo_root)
    try:
        q, source_indices, labels, _identities, loaded_identity = _load_q_after_preflight(repo_root, preflight)
        expected_fold_ids = preflight["fold_ids"]
        with torch.no_grad():
            fold_results = [
                _evaluate_fold(
                    q,
                    source_indices,
                    labels,
                    expected_fold_ids,
                    fold,
                    preflight["fold"]["fold_identity"]["folds"][fold],
                )
                for fold in range(FOLD_COUNT)
            ]
    except _InvalidHarnessError as error:
        report_path = _write_report(repo_root, output_dir, _blocked_report(repo_root, git_before, preflight, error))
        return {
            "result": "BLOCKED",
            "classification": None,
            "source_manifest": _relative(repo_root, repo_root / SOURCE_MANIFEST),
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "fold_manifest": _relative(repo_root, repo_root / FOLD_MANIFEST),
            "fold_manifest_sha256": FOLD_MANIFEST_SHA256,
            "report": _relative(repo_root, report_path),
            "report_sha256": file_sha256(report_path),
            "blocker": "INVALID_HARNESS",
            "sealed_confirmation_consumed": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        }

    empirical = _empirical_result(fold_results)
    report = _valid_report(repo_root, git_before, preflight, loaded_identity, empirical)
    report_path = _write_report(repo_root, output_dir, report)
    return {
        "result": "VALID_COMPLETED",
        "classification": report["classification"],
        "source_manifest": _relative(repo_root, repo_root / SOURCE_MANIFEST),
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "bank_semantic_sha256": preflight["source"]["bank_semantic_sha256"],
        "fold_manifest": _relative(repo_root, repo_root / FOLD_MANIFEST),
        "fold_manifest_sha256": FOLD_MANIFEST_SHA256,
        "assignment_sha256": preflight["fold"]["protocol"]["assignment_sha256"],
        "report": _relative(repo_root, report_path),
        "report_sha256": file_sha256(report_path),
        "compatible_folds": empirical["compatible_fold_count"],
        "fold_count": FOLD_COUNT,
        "git_branch": git_before["branch"],
        "starting_head": git_before["head"],
        "ending_head": report["git"]["ending_head"],
        "producer_selection": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / REPORT_OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
