"""Execute the one-shot, development-only N3 copula attribution consumer.

The V2 contract is the sole decision authority.  This consumer performs only
the frozen rank, energy-distance, and descriptive diagnostics after a complete
authority preflight.  It never fits or samples a producer, reads sealed
confirmation data, or enters an optical/training path.
"""

from __future__ import annotations

import argparse
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
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import compare_n3_minimal_competing_producer_baselines_v1 as comparison
from scripts.research import define_n3_copula_failure_attribution_contract_v1 as v1
from scripts.research import define_n3_copula_failure_attribution_contract_v2 as v2


TASK_ID = "EXECUTE_N3_COPULA_FAILURE_ATTRIBUTION_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_copula_failure_attribution_report.v1"
REPORT_NAME = "n3_copula_failure_attribution_report.v1.json"
OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n3_copula_failure_attribution_v1_attempt_01"
)
GOVERNANCE_PATH = Path("docs/status/research_governance_state.json")
V2_ALIAS = "N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2"
EXPECTED_V1_SHA256 = "333b51abe2a6b1f3e96558273d4b6037267aa80678199d3bf6d0c300bb3508df"
EXPECTED_FOLD_ASSIGNMENT_SHA256 = "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32"
EXPECTED_SEALED_STATUS = "RESERVED_NOT_MATERIALIZED"

FOLD_COUNT = 5
DIMENSION = 100
POOL_COUNT = 1000
FIT_ROWS_PER_FOLD = 800
HELDOUT_ROWS_PER_FOLD = 200
DATA_VS_MODEL_REPLICATES = 4
MODEL_VS_MODEL_REPLICATES = 8
MODEL_REFERENCE_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
MODEL_BANK_ORDER = tuple(
    [("data-vs-model", replicate) for replicate in range(DATA_VS_MODEL_REPLICATES)]
    + [("model-vs-model", replicate) for replicate in range(MODEL_VS_MODEL_REPLICATES)]
)
DATA_MODEL_BANK_ORDER = MODEL_BANK_ORDER[:DATA_VS_MODEL_REPLICATES]

SAFE_DIRTY_PATHS = {
    "docs/CURRENT_STATUS.md",
    "docs/status/current_handoff.md",
    "docs/status/experiment_history.md",
    "docs/status/research_governance_state.json",
    "scripts/research/define_n3_copula_failure_attribution_contract_v1.py",
    "scripts/research/define_n3_copula_failure_attribution_contract_v2.py",
    "scripts/research/execute_n3_copula_failure_attribution_v1.py",
    "tests/v2/test_n3_copula_failure_attribution_contract.py",
    "tests/v2/test_n3_copula_failure_attribution_contract_v2.py",
    "tests/v2/test_execute_n3_copula_failure_attribution_v1.py",
}


class BlockedExecution(RuntimeError):
    """A fail-closed authority, lineage, or V2 numerical validity stop."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BlockedExecution(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _resolve_relative(repo_root: Path, relative_path: str, name: str) -> Path:
    candidate = Path(relative_path)
    _require(not candidate.is_absolute(), f"{name} must be repository-relative")
    try:
        resolved = (repo_root / candidate).resolve()
        resolved.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise BlockedExecution(f"{name} escapes repository root") from exc
    return resolved


def _read_json(path: Path, name: str) -> dict[str, Any]:
    _require(path.is_file(), f"{name} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BlockedExecution(f"{name} is not valid JSON: {path}") from exc
    return dict(_mapping(payload, name))


def _read_json_ref(
    repo_root: Path,
    reference: Mapping[str, Any],
    name: str,
    *,
    schema: str | None = None,
) -> dict[str, Any]:
    path_value = reference.get("path")
    expected_sha = reference.get("sha256")
    _require(isinstance(path_value, str), f"{name} path is missing")
    _require(isinstance(expected_sha, str) and len(expected_sha) == 64, f"{name} full SHA is missing")
    path = _resolve_relative(repo_root, path_value, f"{name} path")
    _require(path.is_file(), f"{name} is missing: {path}")
    _require(file_sha256(path) == expected_sha, f"{name} hash changed: {path}")
    payload = _read_json(path, name)
    if schema is not None:
        _require(payload.get("schema") == schema, f"{name} schema changed")
    return payload


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
        ["git", "-C", str(repo_root), "status", "--porcelain=v1"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    dirty_paths = [line[3:] if len(line) >= 4 else line for line in status]
    return {"branch": branch, "head": head, "dirty_paths": dirty_paths}


def _path_value(payload: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = payload
    for component in dotted_path.split("."):
        value = _mapping(value, f"path {dotted_path}")[component]
    return value


def _bank_key(fold: int, role: str, replicate: int) -> str:
    return f"fold={fold}|role={role}|replicate={replicate}"


def _parse_bank_key(key: str) -> tuple[int, str, int]:
    parts = dict(item.split("=", 1) for item in key.split("|"))
    _require(set(parts) == {"fold", "role", "replicate"}, f"bank key is malformed: {key}")
    return int(parts["fold"]), parts["role"], int(parts["replicate"])


def _validate_v2_delta(contract: Mapping[str, Any], base: Mapping[str, Any]) -> None:
    _require(contract.get("schema") == "n3_copula_failure_attribution_contract.v2", "V2 schema changed")
    _require(contract.get("task_id") == "DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2", "V2 task changed")
    _require(contract.get("gate") == contract["task_id"], "V2 gate changed")
    _require(contract.get("classification") == "DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2_READY", "V2 is not ready")
    _require(contract.get("claim_classification") == "N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2_DEFINED", "V2 claim changed")
    _require(contract.get("delta_only") is True, "V2 is not delta-only")
    _require(contract.get("v1_primary_semantics_unchanged") is True, "V1 primary semantics invariant is false")
    _require(contract.get("base_contract_sha256") == EXPECTED_V1_SHA256, "V2 base V1 SHA changed")

    supersession = _mapping(contract.get("supersession"), "V2 supersession")
    _require(supersession.get("supersedes_sha256") == EXPECTED_V1_SHA256, "V2 superseded V1 SHA changed")
    _require(supersession.get("supersedes_schema") == "n3_copula_failure_attribution_contract.v1", "V2 superseded schema changed")
    _require(supersession.get("supersedes_task_id") == "DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V1", "V2 superseded task changed")
    _require(supersession.get("primary_scientific_semantics_changed") is False, "V2 changes primary scientific semantics")

    diff = _mapping(contract.get("semantic_diff_validation"), "V2 semantic diff validation")
    _require(diff.get("status") == "passed", "V2 semantic diff validation did not pass")
    _require(diff.get("base_contract_sha256") == EXPECTED_V1_SHA256, "V2 semantic diff base SHA changed")
    _require(diff.get("primary_scientific_semantics_unchanged") is True, "V2 semantic diff primary invariant is false")
    compared = set(diff.get("compared_unchanged_paths", []))
    required_unchanged = {
        "authorities",
        "marginal_failure_authority",
        "primary_dependence_observer",
        "fold_attribution",
        "aggregate_attribution",
    }
    _require(required_unchanged <= compared, "V2 semantic diff omits a required unchanged path")
    for path in required_unchanged:
        _require(_path_value(base, path) == _path_value(contract, path), f"V2 changed frozen semantics at {path}")

    delta = _mapping(contract.get("v2_delta"), "V2 delta")
    negative = _mapping(delta.get("negative_nonfinite_policy"), "V2 negative/nonfinite policy")
    _require(negative.get("numeric_dtype") == "CPU float64", "V2 negative policy dtype changed")
    _require(negative.get("scale") == "scale=2*abs(C)+abs(L)+abs(R)", "V2 negative policy scale changed")
    _require(negative.get("tolerance") == "tau=1e-12*max(1.0,scale)", "V2 negative policy tolerance changed")
    _require(negative.get("threshold_selected_before_attribution_data") is True, "V2 negative threshold was not preregistered")
    _require(negative.get("registered_repository_convention_reused") is False, "V2 reused a repository negative convention")
    raw_terms = _mapping(negative.get("raw_formula_terms"), "V2 energy raw terms")
    _require(raw_terms.get("raw_value") == "value=2*C-L-R", "V2 energy raw formula changed")
    nonfinite = _mapping(negative.get("nonfinite_terms_or_value"), "V2 nonfinite policy")
    _require(nonfinite.get("classification") == "NUMERICAL_INVALID", "V2 nonfinite classification changed")
    _require(nonfinite.get("execution_outcome") == "BLOCKED", "V2 nonfinite outcome changed")
    _require(nonfinite.get("scientific_classification_emitted") is False, "V2 emits a nonfinite scientific classification")
    _require(nonfinite.get("partial_attribution_report_emitted") is False, "V2 emits a partial nonfinite report")
    roundoff = _mapping(negative.get("roundoff_interval"), "V2 roundoff policy")
    _require(roundoff.get("condition") == "-tau <= value < 0", "V2 roundoff interval changed")
    _require(roundoff.get("action") == "canonicalize value to exactly 0.0", "V2 roundoff action changed")
    material = _mapping(negative.get("material_negative_interval"), "V2 material-negative policy")
    _require(material.get("condition") == "value < -tau", "V2 material-negative interval changed")
    _require(material.get("action") == "classify NUMERICAL_INVALID and BLOCK the execution with no scientific classification", "V2 material-negative action changed")
    nonnegative = _mapping(negative.get("nonnegative_interval"), "V2 nonnegative policy")
    _require(nonnegative.get("condition") == "value >= 0", "V2 nonnegative interval changed")
    _require(nonnegative.get("action") == "retain value unchanged", "V2 nonnegative action changed")

    spearman = _mapping(delta.get("spearman_model_bank_semantics"), "V2 Spearman semantics")
    _require(spearman.get("formal_gate_role") == "descriptive only; cannot change dependence_fail, fold labels, or aggregate classification", "V2 Spearman role changed")
    _require(spearman.get("per_bank_matrix") == "S_MODEL_BANK[k] is computed independently from the average-rank matrix of model bank k", "V2 Spearman per-bank semantics changed")
    _require(spearman.get("data_model_discrepancies") == "emit all four Delta_S[r] values for DATA-vs-MODEL replicates r=0..3", "V2 Spearman DATA-vs-MODEL emission changed")
    _require(spearman.get("aggregate_discrepancy") == "Delta_S_aggregate=||S_DATA-S_MODEL||_F", "V2 Spearman aggregate discrepancy changed")
    _require(spearman.get("model_model_banks") == "emit their eight individual matrices and Delta_S values descriptively, but exclude them from S_MODEL and every formal gate", "V2 Spearman model-model role changed")
    _require(spearman.get("weighting") == "equal arithmetic weights only; no DATA-derived, learned, best-bank, or selector weighting", "V2 Spearman weighting changed")
    aggregate_spearman = _mapping(spearman.get("aggregate_model_matrix"), "V2 aggregate Spearman matrix")
    _require(aggregate_spearman.get("name") == "S_MODEL", "V2 aggregate Spearman name changed")
    _require(aggregate_spearman.get("source_role") == "data-vs-model", "V2 aggregate Spearman source changed")
    _require(aggregate_spearman.get("source_replicates") == [0, 1, 2, 3], "V2 aggregate Spearman replicates changed")
    _require(aggregate_spearman.get("operation") == "elementwise arithmetic mean", "V2 aggregate Spearman operation changed")
    _require(aggregate_spearman.get("formula") == "S_MODEL[j,k]=(1/4)*sum_{r=0..3} S_MODEL_BANK[data-vs-model,r][j,k]", "V2 aggregate Spearman formula changed")

    spectral = _mapping(delta.get("spectral_median_and_model_aggregation"), "V2 spectral semantics")
    _require(spectral.get("formal_gate_role") == "descriptive only; never overrides primary D or final attribution", "V2 spectral role changed")
    median = _mapping(spectral.get("median_convention"), "V2 spectral median")
    _require(median.get("dimension") == DIMENSION, "V2 spectral dimension changed")
    _require(median.get("central_indices_zero_based") == [49, 50], "V2 spectral median indices changed")
    _require(median.get("central_orders_one_based") == [50, 51], "V2 spectral median orders changed")
    _require(median.get("formula") == "median=(lambda[49]+lambda[50])/2", "V2 spectral median formula changed")
    _require(median.get("ordering") == "descending real eigenvalues lambda[0] >= ... >= lambda[99]", "V2 spectral ordering changed")
    aggregate_spectral = _mapping(spectral.get("aggregate_model_set"), "V2 aggregate spectral set")
    _require(aggregate_spectral.get("role") == "data-vs-model", "V2 aggregate spectral role changed")
    _require(aggregate_spectral.get("replicates") == [0, 1, 2, 3], "V2 aggregate spectral replicates changed")
    _require(aggregate_spectral.get("scalar_operation") == "arithmetic mean across the four corresponding scalar summaries", "V2 aggregate spectral scalar operation changed")
    _require(aggregate_spectral.get("ordered_spectrum_operation") == "elementwise arithmetic mean across the four correspondingly ordered descending eigenvalue vectors", "V2 aggregate spectral vector operation changed")
    _require(aggregate_spectral.get("model_model_reference_banks_excluded") is True, "V2 aggregate spectral model-model exclusion changed")
    _require(spectral.get("comparison") == "emit per data-vs-model bank L2 and named-summary differences against S_DATA's spectrum, plus the same differences against the aggregate MODEL spectrum; no threshold", "V2 spectral comparison changed")
    _require(spectral.get("zero_variance") == "invalid_harness; no epsilon repair", "V2 spectral zero-variance policy changed")

    exact_keys = _mapping(spearman.get("historical_model_bank_order"), "V2 Spearman bank order").get("exact_keys_by_fold")
    spectral_keys = _mapping(spectral.get("historical_model_bank_order"), "V2 spectral bank order").get("exact_keys_by_fold")
    for fold in range(FOLD_COUNT):
        expected = [_bank_key(fold, role, replicate) for role, replicate in MODEL_BANK_ORDER]
        _require(exact_keys.get(str(fold)) == expected, f"V2 Spearman bank order changed for fold {fold}")
        _require(spectral_keys.get(str(fold)) == expected, f"V2 spectral bank order changed for fold {fold}")


def _load_v2_authority(repo_root: Path) -> tuple[Path, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    governance = _read_json(repo_root / GOVERNANCE_PATH, "research governance state")
    aliases = _mapping(governance.get("evidence_aliases"), "research governance evidence aliases")
    alias = dict(_mapping(aliases.get(V2_ALIAS), V2_ALIAS))
    _require(alias.get("kind") == "machine_evidence", "V2 governance alias kind changed")
    _require(alias.get("schema") == "n3_copula_failure_attribution_contract.v2", "V2 governance schema changed")
    _require(alias.get("base_contract_sha256") == EXPECTED_V1_SHA256, "V2 governance V1 binding changed")
    _require(alias.get("primary_scientific_semantics_changed") is False, "V2 governance primary invariant changed")
    path_value = alias.get("path")
    full_sha = alias.get("sha256")
    _require(isinstance(path_value, str), "V2 governance path is missing")
    _require(isinstance(full_sha, str) and len(full_sha) == 64 and full_sha == full_sha.lower(), "V2 governance full SHA is invalid")
    path = _resolve_relative(repo_root, path_value, "V2 contract path")
    _require(path.is_file(), f"V2 contract is missing: {path}")
    actual_sha = file_sha256(path)
    _require(actual_sha == full_sha, "V2 contract full SHA mismatch")
    contract = _read_json(path, "V2 attribution contract")
    base_path = _resolve_relative(
        repo_root,
        str(_mapping(contract.get("supersession"), "V2 supersession").get("supersedes_path")),
        "V1 contract path",
    )
    base_sha = file_sha256(base_path)
    _require(base_sha == EXPECTED_V1_SHA256, "V1 contract full SHA mismatch")
    base = _read_json(base_path, "V1 attribution contract")
    _validate_v2_delta(contract, base)
    _require(alias.get("classification") == contract.get("classification"), "V2 governance classification changed")
    _require(alias.get("claim_classification") == contract.get("claim_classification"), "V2 governance claim changed")
    return path, actual_sha, contract, base, governance


def _validate_current_state(repo_root: Path, contract: Mapping[str, Any], git: Mapping[str, Any]) -> None:
    contract_git = _mapping(contract.get("git"), "V2 git authority")
    _require(git.get("branch") == contract_git.get("branch"), "current branch differs from V2 authority")
    _require(git.get("head") == contract_git.get("head"), "current HEAD differs from V2 authority")
    _require(contract_git.get("starting_committed_head") == contract_git.get("head"), "V2 starting HEAD is not the authority HEAD")
    unexpected = sorted(set(git.get("dirty_paths", [])) - SAFE_DIRTY_PATHS)
    _require(not unexpected, f"unrelated local changes invalidate provenance: {unexpected}")
    diff_check = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(diff_check.returncode == 0, f"git diff --check failed before attribution: {diff_check.stdout}{diff_check.stderr}")
    script_ref = _mapping(contract_git, "V2 git authority")
    script_value = script_ref.get("v2_definition_script")
    script_sha = script_ref.get("v2_definition_script_sha256")
    _require(isinstance(script_value, str) and isinstance(script_sha, str), "V2 definition script binding is incomplete")
    script_path = _resolve_relative(repo_root, script_value, "V2 definition script")
    _require(file_sha256(script_path) == script_sha, "V2 definition script hash changed")


def _validate_development_static(
    repo_root: Path,
    contract: Mapping[str, Any],
    producer_report: Mapping[str, Any],
) -> dict[str, Any]:
    authorities = _mapping(contract.get("authorities"), "V2 authorities")
    development = _mapping(authorities.get("development_data"), "V2 development data authority")
    source_ref = _mapping(development.get("source_manifest"), "source manifest authority")
    partition_ref = _mapping(development.get("partition_manifest"), "partition manifest authority")
    firewall_ref = _mapping(development.get("firewall_manifest"), "firewall manifest authority")
    source = _read_json_ref(repo_root, source_ref, "source manifest", schema="ols_producer_source_manifest.v1")
    partition = _read_json_ref(repo_root, partition_ref, "partition manifest", schema="ols_producer_partition_manifest.v1")
    firewall = _read_json_ref(repo_root, firewall_ref, "firewall manifest", schema="ols_producer_data_firewall_manifest.v1")

    source_artifact = _mapping(source.get("source_artifact"), "source artifact")
    _require(source.get("producer_lineage") == "OLS_LATENT_PRODUCER_V1", "development producer lineage changed")
    _require(source.get("source_coordinate") == "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX", "development source coordinate changed")
    _require(source_artifact.get("count") == POOL_COUNT, "development source count changed")
    _require(source_artifact.get("shape") == [POOL_COUNT, DIMENSION], "development source shape changed")
    _require(source_artifact.get("per_class") == 100, "development class balance changed")
    _require(source_artifact.get("sha256") == source_ref.get("source_artifact_sha256"), "development source artifact binding changed")
    _require(source_artifact.get("index_sha256") == comparison.EXPECTED_SOURCE_INDEX_SHA256, "development source index binding changed")
    _require(source_artifact.get("label_sha256") == comparison.EXPECTED_SOURCE_LABEL_SHA256, "development label binding changed")
    _require(source_artifact.get("r3_complex_sha256") == comparison.EXPECTED_SOURCE_R3_SHA256, "development R3 binding changed")
    partition_identity = _mapping(partition.get("partition_identity"), "partition identity")
    _require(partition_identity.get("semantic_identity_sha256") == partition_ref.get("semantic_identity_sha256"), "partition semantic identity changed")
    _require(partition.get("runtime_resplit") is False and partition.get("runtime_reselection") is False, "runtime partition mutation is enabled")
    _require(firewall.get("firewall_pass") is True, "development firewall did not pass")
    future_consumer = _mapping(firewall.get("future_consumer"), "development firewall future consumer")
    for field in ("must_verify_target_bank_identity", "runtime_membership_mutation", "runtime_reselection", "runtime_resplit"):
        _require(field in future_consumer, f"development firewall field is missing: {field}")
    _require(future_consumer.get("must_verify_target_bank_identity") is True, "target bank identity check is disabled")
    _require(future_consumer.get("runtime_membership_mutation") is False, "development membership mutation is enabled")
    _require(future_consumer.get("runtime_reselection") is False and future_consumer.get("runtime_resplit") is False, "development resplit/reselection is enabled")

    report_pool = _mapping(producer_report.get("development_pool"), "producer report development pool")
    _require(report_pool.get("q_coordinate") == development.get("q_coordinate") == "R2_AMPLITUDE", "development q coordinate changed")
    _require(report_pool.get("q_r2_amplitude_sha256") == development.get("q_r2_amplitude_sha256"), "development q identity changed")
    report_fold = _mapping(producer_report.get("development_fold_identity"), "producer report fold identity")
    contract_fold = _mapping(authorities.get("fold_assignment"), "V2 fold assignment authority")
    _require(report_fold.get("algorithm") == contract_fold.get("algorithm"), "development fold algorithm changed")
    _require(report_fold.get("fold_count") == contract_fold.get("fold_count") == FOLD_COUNT, "development fold count changed")
    _require(report_fold.get("assignment_sha256") == contract_fold.get("assignment_sha256") == EXPECTED_FOLD_ASSIGNMENT_SHA256, "development fold SHA changed")
    fold_ids = report_fold.get("fold_id_by_source_row")
    _require(isinstance(fold_ids, list) and len(fold_ids) == POOL_COUNT, "frozen fold row identity is incomplete")
    _require(set(fold_ids) == set(range(FOLD_COUNT)), "frozen fold labels are invalid")
    for fold in range(FOLD_COUNT):
        _require(fold_ids.count(fold) == HELDOUT_ROWS_PER_FOLD, f"frozen held-out fold {fold} size changed")

    bank_data = _mapping(development.get("fit_bank_from_report"), "report development bank authority")
    for role in ("fit_bank", "internal_validation_bank"):
        authority = _mapping(development.get(role), f"{role} authority")
        report_key = "fit_bank_path" if role == "fit_bank" else "internal_validation_bank_path"
        report_sha_key = "fit_bank_sha256" if role == "fit_bank" else "internal_validation_bank_sha256"
        _require(authority.get("path") == bank_data.get(report_key), f"{role} path differs from report authority")
        _require(authority.get("sha256") == bank_data.get(report_sha_key), f"{role} SHA differs from report authority")
        path = _resolve_relative(repo_root, str(authority["path"]), role)
        _require(path.is_file(), f"{role} is missing: {path}")
        _require(file_sha256(path) == authority["sha256"], f"{role} hash changed")

    sealed = _mapping(authorities.get("sealed_confirmation_exclusion"), "sealed confirmation exclusion")
    _require(sealed.get("status") == EXPECTED_SEALED_STATUS, "sealed confirmation status changed")
    _require(sealed.get("input_authority") is False and sealed.get("sample_values_loaded") is False and sealed.get("consumed") is False, "sealed confirmation firewall changed")
    return {
        "source_manifest": {"path": source_ref["path"], "sha256": source_ref["sha256"], "schema": source["schema"]},
        "partition_manifest": {"path": partition_ref["path"], "sha256": partition_ref["sha256"], "semantic_identity_sha256": partition_ref["semantic_identity_sha256"]},
        "firewall_manifest": {"path": firewall_ref["path"], "sha256": firewall_ref["sha256"], "firewall_pass": True},
        "q_coordinate": development["q_coordinate"],
        "fit_bank": dict(_mapping(development["fit_bank"], "fit bank authority")),
        "internal_validation_bank": dict(_mapping(development["internal_validation_bank"], "internal validation bank authority")),
        "fold_assignment_sha256": contract_fold["assignment_sha256"],
        "fold_id_by_source_row_sha256": tensor_sha256(torch.tensor(fold_ids, dtype=torch.int64)),
        "fit_rows_per_fold": FIT_ROWS_PER_FOLD,
        "heldout_rows_per_fold": HELDOUT_ROWS_PER_FOLD,
        "row_selector": "fold_id_by_source_row == f",
        "fit_rows_excluded": True,
    }


def _validate_static_preflight(
    repo_root: Path,
    git: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    v2_path, v2_sha, contract, base, governance = _load_v2_authority(repo_root)
    _validate_current_state(repo_root, contract, git)
    _require(git.get("head") == _mapping(contract.get("git"), "V2 git").get("starting_committed_head"), "starting HEAD changed")
    base_info = v2._validate_v1_contract(repo_root)
    _require(base_info["contract"] == base, "V1 authority read differs from V2 supersession base")
    _require(contract.get("authorities") == base.get("authorities"), "V2 changed input authorities")
    _require(contract.get("scope") == base.get("scope"), "V2 changed attribution scope")
    _require(contract.get("claim_boundary") == base.get("claim_boundary"), "V2 changed claim boundary")
    _require(contract.get("execution_boundary") == base.get("execution_boundary"), "V2 changed execution boundary")
    producer_report = base_info["producer_report"]
    development_static = _validate_development_static(repo_root, contract, producer_report)
    recovered_authority = _mapping(
        _mapping(contract.get("authorities"), "V2 authorities").get("recovered_generated_banks"),
        "recovered bank authority",
    )
    recovery_manifest_ref = _mapping(recovered_authority.get("manifest"), "recovery manifest authority")
    recovery_manifest = _read_json_ref(
        repo_root,
        recovery_manifest_ref,
        "recovery manifest",
        schema="n3_frozen_copula_generated_banks_manifest.v1",
    )
    _require(not output_dir.exists(), f"write-once attribution output already exists: {output_dir}")
    return {
        "governance": governance,
        "v2_path": v2_path,
        "v2_sha256": v2_sha,
        "v2": contract,
        "v1": base,
        "base_info": base_info,
        "producer_report": producer_report,
        "recovery_manifest": recovery_manifest,
        "development_static": development_static,
    }


def _load_development_rows(
    repo_root: Path,
    preflight: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    q, source_indices, labels, pool_identity = comparison._load_development_pool(repo_root)
    contract = preflight["v2"]
    development = _mapping(_mapping(contract["authorities"], "V2 authorities")["development_data"], "V2 development data")
    _require(pool_identity.get("q_r2_amplitude_sha256") == development.get("q_r2_amplitude_sha256"), "loaded development q SHA differs from V2 authority")
    _require(pool_identity.get("source_indices_sha256") == comparison.EXPECTED_SOURCE_INDEX_SHA256, "loaded source indices differ from authority")
    _require(pool_identity.get("labels_sha256") == comparison.EXPECTED_SOURCE_LABEL_SHA256, "loaded labels differ from authority")
    report_fold = _mapping(preflight["producer_report"].get("development_fold_identity"), "producer report fold identity")
    fold_ids, fold_identity = comparison._assign_folds(source_indices, labels)
    _require(fold_identity == report_fold, "resolved fold/data lineage differs from frozen report")
    _require(torch.equal(fold_ids, torch.tensor(report_fold["fold_id_by_source_row"], dtype=torch.int64)), "resolved fold row map differs from frozen report")
    _require(tuple(q.shape) == (POOL_COUNT, DIMENSION) and q.dtype == torch.float64, "loaded development q shape/dtype changed")
    _require(bool(torch.isfinite(q).all()) and bool(q.ge(0.0).all()), "loaded development q is invalid")
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        held_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1)
        fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1)
        _require(int(held_positions.numel()) == HELDOUT_ROWS_PER_FOLD, f"held-out rows changed for fold {fold}")
        _require(int(fit_positions.numel()) == FIT_ROWS_PER_FOLD, f"fit rows changed for fold {fold}")
        _require(torch.unique(torch.cat([fit_positions, held_positions])).numel() == POOL_COUNT, f"fit/held-out overlap for fold {fold}")
        fold_summaries.append(
            {
                "fold": fold,
                "fit_rows": FIT_ROWS_PER_FOLD,
                "heldout_rows": HELDOUT_ROWS_PER_FOLD,
                "fit_positions_sha256": tensor_sha256(fit_positions),
                "heldout_positions_sha256": tensor_sha256(held_positions),
                "fit_labels_sha256": tensor_sha256(labels.index_select(0, fit_positions)),
                "heldout_labels_sha256": tensor_sha256(labels.index_select(0, held_positions)),
                "fit_heldout_disjoint": True,
            }
        )
    return q, source_indices, labels, fold_ids, {
        "pool_identity": pool_identity,
        "fold_identity": fold_identity,
        "folds": fold_summaries,
    }


def _load_recovered_banks(
    repo_root: Path,
    contract: Mapping[str, Any],
    recovery_manifest: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    authorities = _mapping(contract.get("authorities"), "V2 authorities")
    recovered = _mapping(authorities.get("recovered_generated_banks"), "recovered bank authority")
    tensor_ref = _mapping(recovered.get("tensor_artifact"), "recovered tensor authority")
    tensor_path = _resolve_relative(repo_root, str(tensor_ref["path"]), "recovered tensor path")
    tensor_sha = str(tensor_ref["sha256"])
    payload = safe_torch_load(tensor_path, expected_sha256=tensor_sha, map_location="cpu")
    payload_map = _mapping(payload, "recovered bank payload")
    _require(payload_map.get("schema") == "n3_frozen_copula_generated_banks.v1", "recovered bank payload schema changed")
    _require(payload_map.get("semantic_authority_id") == "FROZEN_N3_COPULA_GENERATED_BANKS_V1", "recovered bank semantic authority changed")
    _require(payload_map.get("version") == 1, "recovered bank payload version changed")
    _require(payload_map.get("coordinate") == "R2_AMPLITUDE", "recovered bank coordinate changed")
    _require(payload_map.get("dtype") == "torch.float64", "recovered bank payload dtype changed")
    _require(payload_map.get("bank_count") == FOLD_COUNT * (DATA_VS_MODEL_REPLICATES + MODEL_VS_MODEL_REPLICATES), "recovered bank payload count changed")
    _require(payload_map.get("bank_shape") == [HELDOUT_ROWS_PER_FOLD, DIMENSION], "recovered bank payload shape changed")
    source_contract = _mapping(payload_map.get("source_contract"), "recovered payload source contract")
    expected_source_contract = _mapping(recovery_manifest.get("source_contract"), "recovery manifest source contract")
    for field in ("path", "sha256"):
        _require(source_contract.get(field) == expected_source_contract.get(field), f"recovered bank source contract {field} changed")
    expected_producer = _mapping(authorities.get("frozen_producer_contract"), "frozen producer contract")
    _require(source_contract.get("sha256") == expected_producer.get("sha256"), "recovered bank producer contract binding changed")
    source_report = _mapping(payload_map.get("source_execution_report"), "recovered payload source report")
    expected_source_report = _mapping(recovery_manifest.get("source_execution_report"), "recovery manifest source report")
    for field in ("path", "sha256"):
        _require(source_report.get(field) == expected_source_report.get(field), f"recovered bank source report {field} changed")
    _require(payload_map.get("fold_assignment_sha256") == _mapping(authorities.get("fold_assignment"), "V2 fold assignment").get("assignment_sha256"), "recovered bank fold identity changed")
    banks_value = _mapping(payload_map.get("banks"), "recovered banks")
    identities = recovered.get("bank_identities")
    _require(isinstance(identities, list) and len(identities) == FOLD_COUNT * (DATA_VS_MODEL_REPLICATES + MODEL_VS_MODEL_REPLICATES), "recovered bank registry is incomplete")
    expected_by_key = {str(_mapping(item, "recovered bank identity")["bank_key"]): _mapping(item, "recovered bank identity") for item in identities}
    _require(len(expected_by_key) == len(identities), "recovered bank registry contains duplicate keys")
    expected_keys = {_bank_key(fold, role, replicate) for fold in range(FOLD_COUNT) for role, replicate in MODEL_BANK_ORDER}
    _require(set(expected_by_key) == expected_keys == set(banks_value), "recovered bank keys changed")
    loaded: dict[str, torch.Tensor] = {}
    for key in sorted(expected_keys):
        identity = expected_by_key[key]
        bank = banks_value[key]
        _require(torch.is_tensor(bank), f"recovered bank is not a tensor: {key}")
        _require(tuple(bank.shape) == (HELDOUT_ROWS_PER_FOLD, DIMENSION), f"recovered bank shape changed: {key}")
        _require(str(bank.dtype) == "torch.float64", f"recovered bank dtype changed: {key}")
        _require(bool(torch.isfinite(bank).all()) and bool(bank.ge(0.0).all()), f"recovered bank values invalid: {key}")
        observed_hash = tensor_sha256(bank)
        _require(observed_hash == identity.get("expected_q_sha256"), f"recovered bank identity mismatch: {key}")
        loaded[key] = bank.detach().cpu().contiguous()
    return loaded, {
        "path": str(tensor_ref["path"]),
        "sha256": tensor_sha,
        "semantic_authority_id": payload_map["semantic_authority_id"],
        "bank_count": len(loaded),
        "bank_identities_verified": True,
        "loaded_as_tensor_after_precheck": True,
    }


def _self_rank_gaussianize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    plotting_positions, _sorted_values, tie_summary = producer_midrank_positions(values)
    ranks = plotting_positions.mul(float(values.shape[0])).add(0.5).contiguous()
    z = torch.special.ndtri(plotting_positions).contiguous()
    _require(bool(torch.isfinite(ranks).all()) and bool(torch.isfinite(z).all()), "self-rank Gaussianization is non-finite")
    return ranks, z, tie_summary


def producer_midrank_positions(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Use the frozen producer's stable average-rank implementation as a pure transform."""

    producer_module = __import__(
        "scripts.research.execute_n3_empirical_marginal_gaussian_copula_producer_v1",
        fromlist=["_midrank_positions"],
    )
    return producer_module._midrank_positions(values)


def _energy_terms(left: torch.Tensor, right: torch.Tensor, label: str) -> dict[str, Any]:
    x = left.detach().to(dtype=torch.float64, device="cpu").contiguous()
    y = right.detach().to(dtype=torch.float64, device="cpu").contiguous()
    _require(x.ndim == 2 and y.ndim == 2 and tuple(x.shape) == tuple(y.shape), f"energy sample shapes differ: {label}")
    cross = torch.cdist(x, y, p=2.0).mean()
    within_left = torch.cdist(x, x, p=2.0).mean()
    within_right = torch.cdist(y, y, p=2.0).mean()
    terms = (cross, within_left, within_right)
    _require(all(bool(torch.isfinite(value)) for value in terms), f"energy terms are non-finite: {label}")
    raw = 2.0 * cross - within_left - within_right
    _require(bool(torch.isfinite(raw)), f"energy value is non-finite: {label}")
    scale = 2.0 * cross.abs() + within_left.abs() + within_right.abs()
    _require(bool(torch.isfinite(scale)), f"energy scale is non-finite: {label}")
    tau = 1.0e-12 * max(1.0, float(scale))
    _require(float(raw) >= -tau, f"NUMERICAL_INVALID materially negative energy distance: {label}")
    canonicalized = float(raw) < 0.0
    value = 0.0 if canonicalized else float(raw)
    return {
        "label": label,
        "cross_mean_C": float(cross),
        "within_left_mean_L": float(within_left),
        "within_right_mean_R": float(within_right),
        "raw_value": float(raw),
        "scale": float(scale),
        "tau": tau,
        "value": value,
        "roundoff_negative_canonicalized": canonicalized,
        "numerical_validity": "valid",
    }


def _q95(values: list[float], label: str) -> dict[str, Any]:
    _require(len(values) == len(MODEL_REFERENCE_PAIRS), f"q95 reference count changed: {label}")
    ordered = sorted(float(value) for value in values)
    h = (len(ordered) - 1) * 0.95
    lower = int(h // 1)
    upper = min(lower + 1, len(ordered) - 1)
    weight = h - lower
    result = (1.0 - weight) * ordered[lower] + weight * ordered[upper]
    _require(torch.isfinite(torch.tensor(result, dtype=torch.float64)), f"q95 is non-finite: {label}")
    return {
        "probability": 0.95,
        "method": "linear empirical quantile",
        "sorted_values": ordered,
        "h": h,
        "lower_index": lower,
        "upper_index": upper,
        "weight": weight,
        "value": result,
    }


def _spearman_matrix(ranks: torch.Tensor, label: str) -> torch.Tensor:
    view = ranks.to(dtype=torch.float64, device="cpu").contiguous()
    centered = view - view.mean(dim=0, keepdim=True)
    sum_squares = centered.square().sum(dim=0)
    _require(bool(torch.isfinite(sum_squares).all()) and bool(sum_squares.gt(0.0).all()), f"Spearman denominator is undefined: {label}")
    denominator = torch.sqrt(sum_squares[:, None] * sum_squares[None, :])
    matrix = centered.transpose(0, 1).mm(centered) / denominator
    _require(bool(torch.isfinite(matrix).all()), f"Spearman matrix is non-finite: {label}")
    return matrix.contiguous()


def _spectrum_summary(z: torch.Tensor, label: str) -> dict[str, Any]:
    view = z.to(dtype=torch.float64, device="cpu").contiguous()
    _require(tuple(view.shape) == (HELDOUT_ROWS_PER_FOLD, DIMENSION), f"spectral input shape changed: {label}")
    centered = view - view.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1).mm(centered).div(float(view.shape[0] - 1))
    variance = torch.diagonal(covariance)
    _require(bool(torch.isfinite(variance).all()) and bool(variance.gt(0.0).all()), f"spectral zero variance: {label}")
    s = variance.sqrt()
    normalizer = s.rsqrt()
    correlation_like = normalizer[:, None] * covariance * normalizer[None, :]
    correlation_like = (correlation_like + correlation_like.transpose(0, 1)).mul(0.5).contiguous()
    _require(bool(torch.isfinite(correlation_like).all()), f"spectral representation is non-finite: {label}")
    eigenvalues = torch.linalg.eigvalsh(correlation_like).flip(0).contiguous()
    _require(tuple(eigenvalues.shape) == (DIMENSION,) and bool(torch.isfinite(eigenvalues).all()), f"spectral eigenvalues are invalid: {label}")
    total = eigenvalues.sum()
    _require(bool(torch.isfinite(total)) and float(total) > 0.0, f"spectral eigenvalue mass is undefined: {label}")
    median = (eigenvalues[49] + eigenvalues[50]) / 2.0
    top1 = eigenvalues[:1].sum() / total
    top5 = eigenvalues[:5].sum() / total
    values = (eigenvalues[0], median, top1, top5)
    _require(all(bool(torch.isfinite(value)) for value in values), f"spectral summary is non-finite: {label}")
    return {
        "label": label,
        "eigenvalues_descending": [float(value) for value in eigenvalues],
        "eigenvalues_sha256": tensor_sha256(eigenvalues),
        "maximum_eigenvalue": float(eigenvalues[0]),
        "median_eigenvalue": float(median),
        "top_1_eigenvalue_mass": float(top1),
        "top_5_eigenvalue_mass": float(top5),
        "representation": "C=diag(s)^(-1/2) * (Z_centered^T Z_centered/(n-1)) * diag(s)^(-1/2), with s_j^2=(Z_centered^T Z_centered/(n-1))_jj",
        "symmetrized": True,
        "median_indices_zero_based": [49, 50],
        "numerical_validity": "valid",
    }


def _summary_names() -> tuple[str, ...]:
    return ("maximum_eigenvalue", "median_eigenvalue", "top_1_eigenvalue_mass", "top_5_eigenvalue_mass")


def _spectrum_comparison(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    first_values = torch.tensor(first["eigenvalues_descending"], dtype=torch.float64)
    second_values = torch.tensor(second["eigenvalues_descending"], dtype=torch.float64)
    return {
        "spectrum_l2": float(torch.linalg.vector_norm(first_values - second_values)),
        "summary_absolute_differences": {
            name: abs(float(first[name]) - float(second[name])) for name in _summary_names()
        },
    }


def _support_tail(
    q_fit: torch.Tensor,
    heldout: torch.Tensor,
    original_failure_indices: list[int],
) -> dict[str, Any]:
    fit_min = q_fit.min(dim=0).values
    fit_max = q_fit.max(dim=0).values
    fit_scale = q_fit.std(dim=0, unbiased=True)
    _require(bool(torch.isfinite(fit_min).all()) and bool(torch.isfinite(fit_max).all()) and bool(torch.isfinite(fit_scale).all()), "support/tail fit statistics are non-finite")
    below = heldout < fit_min
    above = heldout > fit_max
    affected = below.any(dim=0) | above.any(dim=0)
    zero_scale_exceedance = affected & fit_scale.le(0.0)
    _require(not bool(zero_scale_exceedance.any()), "support/tail zero scale with an exceedance is invalid_harness")
    overshoot = torch.zeros_like(heldout, dtype=torch.float64)
    for coordinate in range(DIMENSION):
        if bool(below[:, coordinate].any()):
            overshoot[below[:, coordinate], coordinate] = (
                fit_min[coordinate] - heldout[below[:, coordinate], coordinate]
            ) / fit_scale[coordinate]
        if bool(above[:, coordinate].any()):
            overshoot[above[:, coordinate], coordinate] = (
                heldout[above[:, coordinate], coordinate] - fit_max[coordinate]
            ) / fit_scale[coordinate]
    _require(bool(torch.isfinite(overshoot).all()) and bool(overshoot.ge(0.0).all()), "support/tail normalized overshoot is invalid")
    failure_set = set(int(value) for value in original_failure_indices)
    support_indices = {int(value) for value in affected.nonzero(as_tuple=False).reshape(-1).tolist()}
    overlap = sorted(support_indices & failure_set)
    per_coordinate: list[dict[str, Any]] = []
    for coordinate in range(DIMENSION):
        values = overshoot[:, coordinate]
        below_count = int(below[:, coordinate].sum())
        above_count = int(above[:, coordinate].sum())
        support_count = below_count + above_count
        per_coordinate.append(
            {
                "coordinate": coordinate,
                "fit_min": float(fit_min[coordinate]),
                "fit_max": float(fit_max[coordinate]),
                "fit_sample_std": float(fit_scale[coordinate]),
                "below_count": below_count,
                "above_count": above_count,
                "support_count": support_count,
                "below_rate": float(below[:, coordinate].to(dtype=torch.float64).mean()),
                "above_rate": float(above[:, coordinate].to(dtype=torch.float64).mean()),
                "support_rate": float(support_count / heldout.shape[0]),
                "normalized_overshoot_sum": float(values.sum()),
                "normalized_overshoot_mean_over_heldout": float(values.mean()),
                "maximum_normalized_overshoot": float(values.max()),
                "original_marginal_failure": coordinate in failure_set,
                "support_tail_overlap": coordinate in failure_set and coordinate in support_indices,
            }
        )
    return {
        "role": "descriptive only; cannot redefine marginal_fail or final classification",
        "fit_rows": int(q_fit.shape[0]),
        "heldout_rows": int(heldout.shape[0]),
        "support_rule": "x < fit_min_j or x > fit_max_j",
        "fit_min_sha256": tensor_sha256(fit_min),
        "fit_max_sha256": tensor_sha256(fit_max),
        "fit_sample_std_sha256": tensor_sha256(fit_scale),
        "below_total_count": int(below.sum()),
        "above_total_count": int(above.sum()),
        "support_affected_coordinate_count": len(support_indices),
        "support_affected_coordinates": sorted(support_indices),
        "normalized_overshoot_sum": float(overshoot.sum()),
        "normalized_overshoot_mean_over_all_cells": float(overshoot.mean()),
        "maximum_normalized_overshoot": float(overshoot.max()),
        "original_marginal_failure_coordinates": sorted(failure_set),
        "support_tail_overlap_coordinates": overlap,
        "support_tail_overlap_count": len(overlap),
        "support_tail_overlap_fraction_over_original_failure_coordinates": (
            float(len(overlap) / len(failure_set)) if failure_set else None
        ),
        "finite_sample_caveat": "nonzero support exceedance can occur with nonzero probability under an unchanged continuous population",
        "per_coordinate": per_coordinate,
    }


def _spearman_diagnostics(
    fold: int,
    data_ranks: torch.Tensor,
    data_spearman: torch.Tensor,
    ranked_banks: Mapping[str, torch.Tensor],
    z_by_key: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    aggregate_matrices: list[torch.Tensor] = []
    for role, replicate in MODEL_BANK_ORDER:
        key = _bank_key(fold, role, replicate)
        matrix = _spearman_matrix(ranked_banks[key], key)
        delta = torch.linalg.matrix_norm(data_spearman - matrix, ord="fro")
        _require(bool(torch.isfinite(delta)), f"Spearman discrepancy is non-finite: {key}")
        if role == "data-vs-model":
            aggregate_matrices.append(matrix)
        entries.append(
            {
                "bank_key": key,
                "role": role,
                "replicate": replicate,
                "rank_matrix_sha256": tensor_sha256(ranked_banks[key]),
                "gaussianized_matrix_sha256": tensor_sha256(z_by_key[key]),
                "spearman_matrix": matrix.tolist(),
                "delta_S_to_data": float(delta),
            }
        )
    _require(len(aggregate_matrices) == DATA_VS_MODEL_REPLICATES, f"Spearman aggregate bank count changed for fold {fold}")
    aggregate = torch.stack(aggregate_matrices, dim=0).mean(dim=0).contiguous()
    aggregate_delta = torch.linalg.matrix_norm(data_spearman - aggregate, ord="fro")
    _require(bool(torch.isfinite(aggregate).all()) and bool(torch.isfinite(aggregate_delta)), f"Spearman aggregate is invalid for fold {fold}")
    return {
        "role": "descriptive only; cannot change dependence_fail, fold labels, or aggregate classification",
        "coordinate_order": list(range(DIMENSION)),
        "data_rank_matrix_sha256": tensor_sha256(data_ranks),
        "data_gaussianized_matrix_sha256": tensor_sha256(z_by_key[f"fold={fold}|role=heldout-data|replicate=0"]),
        "S_DATA": data_spearman.tolist(),
        "model_banks_in_v2_order": entries,
        "S_MODEL": aggregate.tolist(),
        "delta_S_aggregate": float(aggregate_delta),
        "aggregate_source_role": "data-vs-model",
        "aggregate_source_replicates": list(range(DATA_VS_MODEL_REPLICATES)),
        "model_model_banks_excluded_from_S_MODEL": True,
        "equal_arithmetic_weights": True,
    }


def _spectral_diagnostics(
    fold: int,
    data_spectrum: Mapping[str, Any],
    spectra: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    data_model_spectra: list[Mapping[str, Any]] = []
    for role, replicate in MODEL_BANK_ORDER:
        key = _bank_key(fold, role, replicate)
        summary = spectra[key]
        item: dict[str, Any] = {
            "bank_key": key,
            "role": role,
            "replicate": replicate,
            "summary": dict(summary),
        }
        if role == "data-vs-model":
            data_model_spectra.append(summary)
            item["comparison_to_S_DATA"] = _spectrum_comparison(summary, data_spectrum)
        entries.append(item)
    _require(len(data_model_spectra) == DATA_VS_MODEL_REPLICATES, f"spectral aggregate bank count changed for fold {fold}")
    eigenvalues = torch.stack(
        [torch.tensor(item["eigenvalues_descending"], dtype=torch.float64) for item in data_model_spectra], dim=0
    ).mean(dim=0).contiguous()
    scalar_aggregate = {
        name: float(sum(float(item[name]) for item in data_model_spectra) / DATA_VS_MODEL_REPLICATES)
        for name in _summary_names()
    }
    aggregate = {
        "role": "aggregate MODEL descriptive summary",
        "source_role": "data-vs-model",
        "source_replicates": list(range(DATA_VS_MODEL_REPLICATES)),
        "model_model_reference_banks_excluded": True,
        "eigenvalues_descending": [float(value) for value in eigenvalues],
        "eigenvalues_sha256": tensor_sha256(eigenvalues),
        "scalar_summaries_arithmetic_mean": scalar_aggregate,
        "maximum_eigenvalue": scalar_aggregate["maximum_eigenvalue"],
        "median_eigenvalue": scalar_aggregate["median_eigenvalue"],
        "top_1_eigenvalue_mass": scalar_aggregate["top_1_eigenvalue_mass"],
        "top_5_eigenvalue_mass": scalar_aggregate["top_5_eigenvalue_mass"],
    }
    aggregate["comparison_to_S_DATA"] = _spectrum_comparison(aggregate, data_spectrum)
    return {
        "role": "descriptive only; never overrides primary D or final attribution",
        "S_DATA_spectrum": dict(data_spectrum),
        "model_banks_in_v2_order": entries,
        "aggregate_MODEL": aggregate,
        "median_convention": {
            "ordering": "descending real eigenvalues",
            "central_indices_zero_based": [49, 50],
            "formula": "median=(lambda[49]+lambda[50])/2",
        },
        "no_threshold_applied": True,
    }


def _measure_fold(
    fold: int,
    q: torch.Tensor,
    fold_ids: torch.Tensor,
    banks: Mapping[str, torch.Tensor],
    marginal_authority: Mapping[str, Any],
) -> dict[str, Any]:
    held_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1)
    fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1)
    heldout = q.index_select(0, held_positions).contiguous()
    q_fit = q.index_select(0, fit_positions).contiguous()
    _require(tuple(heldout.shape) == (HELDOUT_ROWS_PER_FOLD, DIMENSION), f"held-out q shape changed for fold {fold}")
    _require(tuple(q_fit.shape) == (FIT_ROWS_PER_FOLD, DIMENSION), f"fit q shape changed for fold {fold}")

    data_ranks, data_z, data_ties = _self_rank_gaussianize(heldout)
    data_spearman = _spearman_matrix(data_ranks, f"fold={fold}|role=heldout-data")
    data_spectrum = _spectrum_summary(data_z, f"fold={fold}|role=heldout-data")
    ranked_banks: dict[str, torch.Tensor] = {}
    z_by_key: dict[str, torch.Tensor] = {f"fold={fold}|role=heldout-data|replicate=0": data_z}
    spectra: dict[str, Mapping[str, Any]] = {}
    bank_ties: dict[str, Any] = {f"fold={fold}|role=heldout-data|replicate=0": data_ties}
    for role, replicate in MODEL_BANK_ORDER:
        key = _bank_key(fold, role, replicate)
        ranks, z, ties = _self_rank_gaussianize(banks[key])
        ranked_banks[key] = ranks
        z_by_key[key] = z
        bank_ties[key] = ties
        spectra[key] = _spectrum_summary(z, key)

    data_metrics: list[dict[str, Any]] = []
    for replicate in range(DATA_VS_MODEL_REPLICATES):
        key = _bank_key(fold, "data-vs-model", replicate)
        metric = _energy_terms(data_z, z_by_key[key], f"fold={fold}|data-vs-model|replicate={replicate}")
        metric.update({"bank_key": key, "replicate": replicate})
        data_metrics.append(metric)

    reference_metrics: list[dict[str, Any]] = []
    for pair_index, (left_replicate, right_replicate) in enumerate(MODEL_REFERENCE_PAIRS):
        left_key = _bank_key(fold, "model-vs-model", left_replicate)
        right_key = _bank_key(fold, "model-vs-model", right_replicate)
        metric = _energy_terms(z_by_key[left_key], z_by_key[right_key], f"fold={fold}|model-vs-model|pair={pair_index}")
        metric.update({"pair_index": pair_index, "left_bank_key": left_key, "right_bank_key": right_key})
        reference_metrics.append(metric)
    reference_q95 = _q95([float(item["value"]) for item in reference_metrics], f"fold={fold}|model-vs-model|q95")
    for item in data_metrics:
        item["reference_q95"] = reference_q95["value"]
        item["dependence_pass"] = float(item["value"]) <= float(reference_q95["value"])
    dependence_passes = [bool(item["dependence_pass"]) for item in data_metrics]
    dependence_fail = not all(dependence_passes)

    marginal_fail = bool(marginal_authority["marginal_fail"])
    fold_classification = v1.fold_attribution_label(marginal_fail, dependence_fail)
    support_tail = _support_tail(
        q_fit,
        heldout,
        list(marginal_authority["original_coordinate_failure_indices"]),
    )
    spearman = _spearman_diagnostics(fold, data_ranks, data_spearman, ranked_banks, z_by_key)
    spectral = _spectral_diagnostics(fold, data_spectrum, spectra)
    return {
        "fold": fold,
        "operationally_valid": True,
        "fit_rows": FIT_ROWS_PER_FOLD,
        "heldout_rows": HELDOUT_ROWS_PER_FOLD,
        "fit_heldout_disjoint": True,
        "fit_positions_sha256": tensor_sha256(fit_positions),
        "heldout_positions_sha256": tensor_sha256(held_positions),
        "heldout_data_rank_tie_summary": data_ties,
        "model_bank_rank_tie_summaries": bank_ties,
        "marginal_authority": {
            "M": marginal_fail,
            "marginal_fail": marginal_fail,
            "marginal_compatible": not marginal_fail,
            "component_fields": list(marginal_authority["marginal_component_fields"]),
            "original_coordinate_failure_indices": list(marginal_authority["original_coordinate_failure_indices"]),
        },
        "primary_dependence": {
            "D": dependence_fail,
            "dependence_fail": dependence_fail,
            "data_vs_model": data_metrics,
            "model_vs_model_reference_pairs": reference_metrics,
            "model_vs_model_reference_q95": reference_q95,
            "all_data_vs_model_comparisons_pass": all(dependence_passes),
            "numeric_dtype": "CPU float64",
            "energy_distance_policy": "V2 negative/nonfinite policy applied exactly",
        },
        "descriptive": {
            "spearman": spearman,
            "spectral": spectral,
            "support_tail": support_tail,
        },
        "fold_classification": fold_classification,
        "numerical_validity_events": [],
    }


def _build_report(
    repo_root: Path,
    preflight: Mapping[str, Any],
    git_before: Mapping[str, Any],
    git_after_measurement: Mapping[str, Any],
    q: torch.Tensor,
    source_indices: torch.Tensor,
    labels: torch.Tensor,
    lineage: Mapping[str, Any],
    banks: Mapping[str, torch.Tensor],
    recovered_info: Mapping[str, Any],
    fold_results: list[Mapping[str, Any]],
    aggregate_classification: str,
) -> dict[str, Any]:
    contract = preflight["v2"]
    labels_by_fold = {int(item["fold"]): str(item["fold_classification"]) for item in fold_results}
    aggregate = _mapping(contract["aggregate_attribution"], "V2 aggregate attribution")
    failure_folds = [int(value) for value in aggregate["original_failure_folds"]]
    control_folds = [int(value) for value in aggregate["original_pass_control_folds"]]
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "classification": aggregate_classification,
        "classification_vocabulary": list(aggregate["classification_vocabulary"]),
        "contract": {
            "path": _relative(repo_root, preflight["v2_path"]),
            "sha256": preflight["v2_sha256"],
            "schema": contract["schema"],
            "task_id": contract["task_id"],
            "classification": contract["classification"],
            "claim_classification": contract["claim_classification"],
            "base_v1_sha256": contract["base_contract_sha256"],
            "semantic_diff_validation": contract["semantic_diff_validation"],
            "primary_scientific_semantics_changed": False,
        },
        "input_verification": {
            "v2_contract_full_sha256_verified_before_q_load": True,
            "v1_bound_sha256": EXPECTED_V1_SHA256,
            "v1_contract_immutable": True,
            "producer_contract": contract["authorities"]["frozen_producer_contract"],
            "original_copula_execution_report": contract["authorities"]["original_copula_execution_report"],
            "fold_assignment": contract["authorities"]["fold_assignment"],
            "recovered_generated_banks": recovered_info,
            "development_data": lineage,
            "development_q_tensor_sha256": tensor_sha256(q),
            "development_source_indices_sha256": tensor_sha256(source_indices),
            "development_labels_sha256": tensor_sha256(labels),
            "development_data_loaded_after_static_precheck": True,
            "recovered_q_loaded_after_static_precheck": True,
        },
        "development_lineage": {
            "coordinate": "R2_AMPLITUDE",
            "row_selector": "fold_id_by_source_row == f",
            "fit_rows_per_fold": FIT_ROWS_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_ROWS_PER_FOLD,
            "fit_rows_excluded_from_heldout": True,
            "fold_identity": lineage["fold_identity"],
            "folds": lineage["folds"],
            "data_reuse_limitation": "All five folds reuse the fixed 1000-row N3 development pool; each fold evaluates on its 200 held-out development rows against banks produced from its complementary 800 fitting rows. This is not an independent confirmation or population-truth result.",
        },
        "fold_results": list(fold_results),
        "aggregate": {
            "classification": aggregate_classification,
            "fold_pattern": labels_by_fold,
            "original_failure_folds": failure_folds,
            "original_pass_control_folds": control_folds,
            "failure_fold_labels": {str(fold): labels_by_fold[fold] for fold in failure_folds},
            "control_fold_labels": {str(fold): labels_by_fold[fold] for fold in control_folds},
            "mapping": aggregate["classification_vocabulary"],
            "mapping_rule": {
                "M": "original marginal_fail from frozen producer report",
                "D": "dependence_fail iff any of four data-vs-model D_DM values exceeds fold-local model-vs-model q95",
                "fold_mapping": contract["fold_attribution"]["mapping"],
                "aggregate_mapping": contract["aggregate_attribution"],
            },
            "control_fold_behavior": "pass-control folds are required to be NO_RESIDUAL_ATTRIBUTED_FAILURE for either dominant aggregate class; any contradictory control-fold failure maps to MIXED",
            "why_v2_mapping_produces_this_result": "The aggregate label is the exact V2 conservative mapping of the five frozen fold labels, using original failure folds [0,2,3] and pass-control folds [1,4]; no learned weight, selector, or new observer is used.",
            "development_only": True,
        },
        "outer_gate": {
            "result": "VALID_COMPLETED",
            "question": "Under the frozen V2 attribution contract, what exact development-only marginal-vs-dependence classification is produced for the existing N3 empirical-marginal Gaussian-copula failure?",
            "answer": aggregate_classification,
            "only_permitted_downstream_direction": "The next P1 planning task may use this descriptive result to define one bounded intervention while holding the complementary treatment fixed, according to the V2 claim boundary.",
            "p1_execution_authorized": False,
        },
        "governance": {
            "selection_scope": "DEVELOPMENT_ONLY",
            "attribution_executed": True,
            "metrics_computed": True,
            "producer_fit": False,
            "producer_sampling": False,
            "deterministic_bank_rematerialization": False,
            "sealed_confirmation_status": EXPECTED_SEALED_STATUS,
            "sealed_confirmation_consumed": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "execution_boundary": {
            "cpu_only": True,
            "cuda": False,
            "optimizer": False,
            "backward": False,
            "training": False,
            "producer_fit": False,
            "producer_sampling": False,
            "rank_reselection": False,
            "new_family_selection": False,
            "sealed_confirmation_read": False,
            "image_quality_selection": False,
            "complex_ols": False,
            "generation_promotion": False,
            "measurement_only": True,
        },
        "git": {
            "branch": git_before["branch"],
            "starting_head": git_before["head"],
            "ending_head": git_after_measurement["head"],
            "worktree_dirty_before": list(git_before["dirty_paths"]),
            "worktree_dirty_after_measurement": list(git_after_measurement["dirty_paths"]),
            "consumer_script": _relative(repo_root, Path(__file__).resolve()),
            "consumer_script_sha256": file_sha256(Path(__file__).resolve()),
            "committed": False,
            "new_commit_created": False,
        },
        "not_executed": [
            "producer fit",
            "producer sampling",
            "deterministic bank rematerialization",
            "sealed confirmation source loading or materialization",
            "new rank selection",
            "mixture K change or repair",
            "selector V3",
            "new marginal/dependence family",
            "S3 rerun",
            "optimizer",
            "backward",
            "training",
            "CUDA",
            "image-quality selection",
            "final all-1000 fit",
            "N4/N5/N6 execution",
            "complex OLS",
            "generation-readiness promotion",
        ],
    }


def _write_report(repo_root: Path, report: Mapping[str, Any], output_dir: Path) -> Path:
    del repo_root
    _require(not output_dir.exists(), f"refusing to reuse write-once output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(
            temporary / REPORT_NAME,
            report,
            artifact_kind="N3 copula failure attribution report",
        )
        _require(not output_dir.exists(), f"attribution output appeared during publication: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / REPORT_NAME


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    output_dir = _resolve_relative(repo_root, str(output_dir), "attribution output directory")
    git_before = _git_state(repo_root)
    preflight = _validate_static_preflight(repo_root, git_before, output_dir)
    banks, recovered_info = _load_recovered_banks(repo_root, preflight["v2"], preflight["recovery_manifest"])
    q, source_indices, labels, fold_ids, lineage = _load_development_rows(repo_root, preflight)
    marginal_by_fold = {
        int(item["fold"]): item
        for item in preflight["base_info"]["report_info"]["folds"]
    }
    fold_results = [
        _measure_fold(fold, q, fold_ids, banks, marginal_by_fold[fold])
        for fold in range(FOLD_COUNT)
    ]
    labels_by_fold = {int(item["fold"]): str(item["fold_classification"]) for item in fold_results}
    aggregate_classification = v1.aggregate_attribution_labels(labels_by_fold)
    git_after_measurement = _git_state(repo_root)
    report = _build_report(
        repo_root,
        preflight,
        git_before,
        git_after_measurement,
        q,
        source_indices,
        labels,
        lineage,
        banks,
        recovered_info,
        fold_results,
        aggregate_classification,
    )
    report_path = _write_report(repo_root, report, output_dir)
    report_sha = file_sha256(report_path)
    return {
        "result": "VALID_COMPLETED",
        "gate": GATE,
        "aggregate_classification": aggregate_classification,
        "report": _relative(repo_root, report_path),
        "report_sha256": report_sha,
        "v2_contract": _relative(repo_root, preflight["v2_path"]),
        "v2_contract_sha256": preflight["v2_sha256"],
        "v1_bound_sha256": EXPECTED_V1_SHA256,
        "fold_pattern": labels_by_fold,
        "q_values_loaded_after_precheck": True,
        "recovered_tensor_loaded_after_precheck": True,
        "attribution_metrics_computed": True,
        "sealed_confirmation_consumed": False,
        "cuda": False,
        "training": False,
        "git": report["git"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    try:
        result = execute(repo_root, args.output_dir)
    except BlockedExecution as exc:
        print(json.dumps({"result": "BLOCKED", "gate": GATE, "reason": str(exc)}, indent=2, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
