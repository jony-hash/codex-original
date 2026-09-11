"""Execute the frozen development-only N3 component-swap diagnostic.

This consumer evaluates only the two preregistered oracle hybrids in raw
R2-amplitude q-space.  It consumes the frozen historical pairing and recovered
bank authority after a complete static preflight.  It never fits or samples a
producer, reads the sealed confirmation source, rematerializes banks, or enters
training, optics, or generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ldm_project.v2.evidence import (
    atomic_write_new_json,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import compare_n3_minimal_competing_producer_baselines_v1 as comparison
from scripts.research import define_n3_copula_mixed_component_swap_diagnostic_contract_v1 as contract_definition


TASK_ID = "EXECUTE_N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_copula_mixed_component_swap_diagnostic_report.v1"
REPORT_NAME = "n3_copula_mixed_component_swap_diagnostic_report.v1.json"
OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n3_copula_mixed_component_swap_diagnostic_v1_attempt_01"
)
CONTRACT_PATH = contract_definition.OUTPUT_DIR / contract_definition.CONTRACT_NAME
EXPECTED_CONTRACT_SHA256 = "d46c4d8e8d8dbc41f325f2e1d7f001d645479a60deef08e5d514a80bada22763"
GOVERNANCE_PATH = Path("docs/status/research_governance_state.json")
CONTRACT_ALIAS = "N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_CONTRACT"
REPORT_ALIAS = "N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_REPORT"

EXPECTED_BRANCH = contract_definition.EXPECTED_BRANCH
EXPECTED_STARTING_HEAD = contract_definition.EXPECTED_STARTING_HEAD
FOLD_COUNT = contract_definition.FOLD_COUNT
POOL_COUNT = 1000
DIMENSION = contract_definition.DIMENSION
FIT_ROWS_PER_FOLD = 800
HELDOUT_ROWS_PER_FOLD = contract_definition.HELDOUT_ROWS_PER_FOLD
DATA_VS_MODEL_REPLICATES = contract_definition.DATA_VS_MODEL_REPLICATES
MODEL_VS_MODEL_REPLICATES = contract_definition.MODEL_VS_MODEL_REPLICATES
MODEL_REFERENCE_PAIRS = contract_definition.MODEL_REFERENCE_PAIRS
PRIMARY_QUANTILE = contract_definition.PRIMARY_QUANTILE
EXPECTED_FOLD_ASSIGNMENT_SHA256 = "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32"
MODEL_BANK_ORDER = tuple(
    [("data-vs-model", replicate) for replicate in range(DATA_VS_MODEL_REPLICATES)]
    + [("model-vs-model", replicate) for replicate in range(MODEL_VS_MODEL_REPLICATES)]
)
HYBRID_ANNOTATIONS = (
    "DIAGNOSTIC_ONLY",
    "ORACLE_USES_HELDOUT_DATA",
    "NOT_A_PRODUCER",
    "NOT_GENERATION_EVIDENCE",
)
SAFE_DIRTY_PATHS = set(contract_definition.SAFE_DIRTY_PATHS) | {
    "scripts/research/execute_n3_copula_mixed_component_swap_diagnostic_v1.py",
    "tests/v2/test_execute_n3_copula_mixed_component_swap_diagnostic_v1.py",
}


class BlockedExecution(RuntimeError):
    """A fail-closed authority, lineage, identity, or numerical stop."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BlockedExecution(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)), f"{name} must be a sequence")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _resolve_relative(repo_root: Path, path_value: str, name: str) -> Path:
    candidate = Path(path_value)
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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BlockedExecution(f"{name} is not valid JSON: {path}") from exc
    return dict(_mapping(value, name))


def _read_json_exact(repo_root: Path, path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    target = _resolve_relative(repo_root, path.as_posix(), name)
    _require(file_sha256(target) == expected_sha256, f"{name} hash changed: {target}")
    return _read_json(target, name)


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
    return {
        "branch": branch,
        "head": head,
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in status],
    }


def _finite_number(value: Any, name: str) -> float:
    _require(type(value) in (int, float), f"{name} must be a JSON number")
    number = float(value)
    _require(math.isfinite(number), f"{name} must be finite")
    return number


def _bank_key(fold: int, role: str, replicate: int) -> str:
    return f"fold={fold}|role={role}|replicate={replicate}"


def _validate_contract_shape(contract: Mapping[str, Any]) -> None:
    _require(contract.get("schema") == contract_definition.CONTRACT_SCHEMA, "component-swap contract schema changed")
    _require(contract.get("version") == 1, "component-swap contract version changed")
    _require(contract.get("task_id") == contract_definition.TASK_ID, "component-swap contract task changed")
    _require(contract.get("gate") == contract.get("task_id"), "component-swap contract gate changed")
    _require(contract.get("status") == "valid_completed", "component-swap contract is not valid_completed")
    _require(contract.get("classification") == contract_definition.READY_CLASSIFICATION, "component-swap contract is not ready")
    _require(contract.get("claim_classification") == contract_definition.CLAIM_CLASSIFICATION, "component-swap contract claim changed")
    _require(contract.get("semantic_authority_id") == contract_definition.TASK_ID, "component-swap semantic authority changed")
    _require(contract.get("new_contract") is True, "component-swap contract is not new")
    _require(contract.get("historical_preregistration") is False, "component-swap contract became historical preregistration")
    _require(contract.get("definition_only") is True, "component-swap contract is not definition-only")

    totality = _mapping(contract.get("contract_totality"), "component-swap contract totality")
    _require(totality.get("status") == "passed", "component-swap contract totality did not pass")
    _require(totality.get("totality_proven") is True, "component-swap contract totality is not proven")
    _require(totality.get("future_runtime_math_choices") == [], "component-swap contract leaves runtime math choices")
    _require(isinstance(totality.get("validated_domains"), Mapping), "component-swap validated domains are missing")
    _require(isinstance(totality.get("failure_stops"), list) and totality["failure_stops"], "component-swap failure stops are missing")
    _require(isinstance(totality.get("scientific_classification_requires"), str), "component-swap classification gate is missing")

    rank = _mapping(contract.get("ordinal_rank_representation"), "ordinal rank representation")
    _require(rank.get("input_bank_shape") == [HELDOUT_ROWS_PER_FOLD, DIMENSION], "ordinal rank shape changed")
    _require(rank.get("input_dtype") == "torch.float64", "ordinal rank dtype changed")
    _require(rank.get("rank_index_base") == 0, "ordinal rank index base changed")
    _require(rank.get("total_order_key") == "(numeric X[i,j] ascending, original row index i ascending for exact ties)", "ordinal rank tie order changed")
    rank_semantics = _mapping(rank.get("implementation_semantics"), "ordinal rank implementation semantics")
    _require(rank_semantics.get("sort") == "stable ascending numeric sort independently for each coordinate", "ordinal rank sort changed")
    _require(rank_semantics.get("tie_key") == "exact original row index; no tolerance, rounding, midrank, or random tie break", "ordinal rank tie semantics changed")
    _require(rank.get("no_rng") is True and rank.get("deterministic") is True, "ordinal rank determinism changed")

    marginal = _mapping(contract.get("marginal_oracle"), "marginal oracle")
    dependence = _mapping(contract.get("dependence_oracle"), "dependence oracle")
    _require(marginal.get("name") == "H_MARG", "marginal oracle name changed")
    _require(marginal.get("equation") == "H_MARG[i,j] = sorted_D[r_M(i,j),j]", "marginal oracle equation changed")
    _require(dependence.get("name") == "H_DEP", "dependence oracle name changed")
    _require(dependence.get("equation") == "H_DEP[i,j] = sorted_M[r_D(i,j),j]", "dependence oracle equation changed")

    primary = _mapping(contract.get("primary_diagnostic"), "primary diagnostic")
    _require(primary.get("representation") == "raw q-space; do not use self-rank Gaussianized z values", "diagnostic representation changed")
    _require(primary.get("no_new_metric") is True, "diagnostic introduced a new metric")
    _require(_mapping(primary.get("metrics"), "diagnostic metrics") == {"E_BASE": "ED(D,M)", "E_MARG": "ED(D,H_MARG)", "E_DEP": "ED(D,H_DEP)"}, "diagnostic metric vocabulary changed")
    metric = _mapping(primary.get("raw_energy_distance_implementation"), "raw energy implementation")
    _require(metric.get("metric_name") == "validated_biased_empirical_energy_distance_euclidean_q_space", "raw metric name changed")
    _require(metric.get("formula") == "ED(X,Y)=2*mean_{i,k} d(X_i,Y_k)-mean_{i,i'} d(X_i,X_i')-mean_{k,k'} d(Y_k,Y_k')", "raw metric formula changed")
    _require(metric.get("within_pairs") == "include diagonal pairs; diagonal distances are zero", "raw metric within-pair convention changed")
    _require(metric.get("negative_policy") == "require raw >= -1e-10; otherwise invalid_harness; return max(0.0,raw)", "raw metric negative policy changed")
    _require(_mapping(primary.get("frozen_model_reference_q95"), "frozen q95").get("oracle_hybrid_values_in_q95") is False, "oracle values entered frozen q95")
    baseline = _mapping(primary.get("baseline_reconstruction"), "baseline reconstruction")
    _require(baseline.get("future_E_BASE_is_recomputed") is True, "E_BASE is not recomputed")
    _require(baseline.get("eligibility_is_bound_to_historical_field") is True, "eligibility is not bound to historical field")
    _require(baseline.get("fold_level_compatibility_not_used_as_pair_failure") is True, "fold compatibility was made pair authority")

    pair_level = _mapping(contract.get("pair_level_sufficiency"), "pair-level sufficiency")
    _require(pair_level.get("pair_count_per_fold") == DATA_VS_MODEL_REPLICATES, "pair count changed")
    rescue = _mapping(pair_level.get("rescue_booleans"), "oracle rescue booleans")
    _require(rescue.get("marginal_oracle_rescues") == "E_MARG <= frozen fold-local original q95", "marginal rescue rule changed")
    _require(rescue.get("dependence_oracle_rescues") == "E_DEP <= frozen fold-local original q95", "dependence rescue rule changed")
    _require(rescue.get("strict_failure_boundary") == "only > q95 is failure; equality is pass", "primary failure boundary changed")
    controls = _mapping(pair_level.get("originally_passing_control_policy"), "control policy")
    for field in ("report_both_oracle_booleans", "use_pair_label_vocabulary_for_descriptive_control", "exclude_from_original_failure_sufficiency_aggregate", "do_not_reinterpret_as_original_failure_cause"):
        _require(controls.get(field) is True, f"control policy changed: {field}")

    fold_aggregation = _mapping(contract.get("fold_aggregation"), "fold aggregation")
    _require(fold_aggregation.get("controls_are_descriptive_only") is True, "control folds became causal")
    _require(fold_aggregation.get("best_bank_selection") is False, "best-bank selection is enabled")
    aggregate = _mapping(contract.get("aggregate_oracle_sufficiency"), "aggregate oracle sufficiency")
    _require(aggregate.get("no_producer_support_classification") is True, "aggregate became producer support")
    _require(aggregate.get("no_selection_authority") is True, "aggregate became selection authority")
    _require(aggregate.get("vocabulary") == [
        "N3_COPULA_PRIMARY_MARGINAL_ORACLE_SUFFICIENT",
        "N3_COPULA_PRIMARY_DEPENDENCE_ORACLE_SUFFICIENT",
        "N3_COPULA_PRIMARY_BOTH_ORACLES_SUFFICIENT",
        "N3_COPULA_PRIMARY_NEITHER_ORACLE_SUFFICIENT",
        "N3_COPULA_PRIMARY_ORACLE_EFFECT_MIXED",
    ], "aggregate vocabulary changed")

    boundary = _mapping(contract.get("execution_boundary"), "component-swap execution boundary")
    false_fields = (
        "q_tensor_loaded_by_this_definition_consumer",
        "oracle_metrics_computed_by_this_definition_consumer",
        "component_swaps_executed_by_this_definition_consumer",
        "producer_fit",
        "producer_sampling",
        "bank_rematerialization",
        "new_sampling",
        "sealed_confirmation_read",
        "cuda",
        "optimizer",
        "backward",
        "training",
        "complex_ols",
        "generation",
        "n4_n5_n6",
        "formal_promotion",
        "training_authorized",
    )
    for field in false_fields:
        _require(boundary.get(field) is False, f"component-swap boundary changed: {field}")

    sealed = _mapping(_mapping(contract.get("authorities"), "component-swap authorities").get("sealed_confirmation_exclusion"), "sealed confirmation exclusion")
    for field in ("input_authority", "path_read", "sample_values_loaded", "consumed"):
        _require(sealed.get(field) is False, f"sealed confirmation exclusion changed: {field}")
    _require(sealed.get("status") == "RESERVED_NOT_MATERIALIZED", "sealed confirmation status changed")


def _validate_contract_alias(repo_root: Path, contract_path: Path) -> dict[str, Any]:
    governance = _read_json(repo_root / GOVERNANCE_PATH, "research governance state")
    aliases = _mapping(governance.get("evidence_aliases"), "research governance evidence aliases")
    alias = _mapping(aliases.get(CONTRACT_ALIAS), CONTRACT_ALIAS)
    _require(alias.get("kind") == "machine_evidence", "component-swap contract governance kind changed")
    _require(alias.get("path") == _relative(repo_root, contract_path), "component-swap contract governance path changed")
    _require(alias.get("sha256") == EXPECTED_CONTRACT_SHA256, "component-swap contract governance SHA changed")
    _require(alias.get("schema") == contract_definition.CONTRACT_SCHEMA, "component-swap contract governance schema changed")
    _require(alias.get("classification") == contract_definition.READY_CLASSIFICATION, "component-swap contract governance classification changed")
    _require(alias.get("component_swaps_executed") is False, "component-swap contract governance execution flag changed")
    _require(alias.get("oracle_metrics_computed") is False, "component-swap contract governance metric flag changed")
    _require(alias.get("sealed_confirmation_consumed") is False, "component-swap contract governance sealed flag changed")
    return governance


def _validate_authority_bindings(
    repo_root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    attribution_v1, attribution_v2, attribution_report = contract_definition._validate_attribution_contracts(repo_root)
    producer_contract = contract_definition._read_json_exact(
        repo_root,
        contract_definition.PRODUCER_CONTRACT,
        contract_definition.PRODUCER_CONTRACT_SHA256,
        "original producer contract",
    )
    contract_definition._validate_producer_contract(producer_contract)
    producer_report = contract_definition._read_json_exact(
        repo_root,
        contract_definition.PRODUCER_REPORT,
        contract_definition.PRODUCER_REPORT_SHA256,
        "original producer report",
    )
    producer_info = contract_definition._validate_producer_report(producer_report, producer_contract, repo_root)
    recovery_manifest = contract_definition._read_json_exact(
        repo_root,
        contract_definition.RECOVERY_MANIFEST,
        contract_definition.RECOVERY_MANIFEST_SHA256,
        "recovery manifest",
    )
    recovery_info = contract_definition._validate_recovery_manifest(recovery_manifest, producer_report, repo_root)
    metric_sources = contract_definition._validate_metric_sources(repo_root)

    authorities = _mapping(contract.get("authorities"), "component-swap authorities")
    expected_refs = {
        "attribution_contract_v1": (contract_definition.ATTRIBUTION_CONTRACT_V1, contract_definition.ATTRIBUTION_CONTRACT_V1_SHA256),
        "attribution_contract_v2": (contract_definition.ATTRIBUTION_CONTRACT_V2, contract_definition.ATTRIBUTION_CONTRACT_V2_SHA256),
        "latest_attribution_report": (contract_definition.ATTRIBUTION_REPORT, contract_definition.ATTRIBUTION_REPORT_SHA256),
        "original_producer_contract": (contract_definition.PRODUCER_CONTRACT, contract_definition.PRODUCER_CONTRACT_SHA256),
        "original_producer_report": (contract_definition.PRODUCER_REPORT, contract_definition.PRODUCER_REPORT_SHA256),
    }
    for name, (path, sha256) in expected_refs.items():
        ref = _mapping(authorities.get(name), f"component-swap authority {name}")
        _require(ref.get("path") == _relative(repo_root, repo_root / path), f"component-swap authority path changed: {name}")
        _require(ref.get("sha256") == sha256, f"component-swap authority SHA changed: {name}")

    fold_authority = _mapping(authorities.get("fold_assignment"), "component-swap fold authority")
    _require(fold_authority.get("algorithm") == "within_class_sha256_sort_round_robin_v1", "component-swap fold algorithm changed")
    _require(fold_authority.get("fold_count") == FOLD_COUNT, "component-swap fold count changed")
    _require(fold_authority.get("assignment_sha256") == EXPECTED_FOLD_ASSIGNMENT_SHA256, "component-swap fold identity changed")

    development = _mapping(authorities.get("development_data"), "component-swap development authority")
    _require(development.get("q_coordinate") == "R2_AMPLITUDE", "component-swap q coordinate changed")
    _require(development.get("q_r2_amplitude_sha256") == "11fb88552de22a96eae38fdbc772a4f90724a9badb3d08f338647e7e7e212012", "component-swap development q identity changed")
    _require(_mapping(development.get("fit_bank"), "component-swap fit bank").get("sha256") == comparison.EXPECTED_FIT_BANK_SHA256, "component-swap fit bank identity changed")
    _require(_mapping(development.get("internal_validation_bank"), "component-swap internal validation bank").get("sha256") == comparison.EXPECTED_INTERNAL_VALIDATION_BANK_SHA256, "component-swap internal validation bank identity changed")

    _validate_pairing(contract, attribution_report, producer_info, recovery_info)
    return {
        "attribution_contract_v1": {"schema": attribution_v1["schema"], "sha256": contract_definition.ATTRIBUTION_CONTRACT_V1_SHA256},
        "attribution_contract_v2": {"schema": attribution_v2["schema"], "sha256": contract_definition.ATTRIBUTION_CONTRACT_V2_SHA256},
        "attribution_report": {"schema": attribution_report["schema"], "sha256": contract_definition.ATTRIBUTION_REPORT_SHA256, "classification": attribution_report["classification"]},
        "producer_contract": {"schema": producer_contract["schema"], "sha256": contract_definition.PRODUCER_CONTRACT_SHA256, "classification": producer_contract["classification"]},
        "producer_report": {"schema": producer_report["schema"], "sha256": contract_definition.PRODUCER_REPORT_SHA256, "classification": producer_report["classification"]},
        "producer_info": producer_info,
        "producer_report_payload": producer_report,
        "recovery_manifest": recovery_manifest,
        "recovery_info": recovery_info,
        "metric_sources": metric_sources,
        "fold_assignment": dict(fold_authority),
    }


def _validate_pairing(
    contract: Mapping[str, Any],
    attribution_report: Mapping[str, Any],
    producer_info: Mapping[str, Any],
    recovery_info: Mapping[str, Any],
) -> None:
    pairing = _mapping(contract.get("pairing"), "component-swap pairing")
    _require(pairing.get("historical_pairing_only") is True, "component-swap pairing is not historical-only")
    _require(pairing.get("model_banks", {}).get("replicates") == list(range(DATA_VS_MODEL_REPLICATES)), "component-swap DATA-vs-MODEL roles changed")
    _require(pairing.get("model_reference_banks", {}).get("replicates") == list(range(MODEL_VS_MODEL_REPLICATES)), "component-swap MODEL-vs-MODEL roles changed")
    expected_reference_pairs = [
        {"pair_index": index, "left_replicate": left, "right_replicate": right}
        for index, (left, right) in enumerate(MODEL_REFERENCE_PAIRS)
    ]
    _require(pairing.get("model_reference_banks", {}).get("reference_pairs") == expected_reference_pairs, "component-swap reference pairing changed")

    lineage = _mapping(attribution_report.get("development_lineage"), "attribution development lineage")
    lineage_folds = _sequence(lineage.get("folds"), "attribution lineage folds")
    producer_fields = _sequence(producer_info.get("original_pair_fields"), "original producer pair fields")
    identities = {
        str(_mapping(item, "recovery bank identity").get("bank_key")): _mapping(item, "recovery bank identity")
        for item in _sequence(recovery_info.get("bank_identities"), "recovery bank identities")
    }
    _require(len(lineage_folds) == FOLD_COUNT and len(producer_fields) == FOLD_COUNT, "component-swap pairing fold count changed")
    expected_keys = {
        _bank_key(fold, role, replicate)
        for fold in range(FOLD_COUNT)
        for role, replicate in MODEL_BANK_ORDER
    }
    _require(set(identities) == expected_keys, "component-swap recovery role set changed")

    folds = _sequence(pairing.get("folds"), "component-swap pairing folds")
    _require(len(folds) == FOLD_COUNT, "component-swap pairing folds are incomplete")
    for fold in range(FOLD_COUNT):
        record = _mapping(folds[fold], f"component-swap pairing fold {fold}")
        lineage_fold = _mapping(lineage_folds[fold], f"component-swap lineage fold {fold}")
        producer_field = _mapping(producer_fields[fold], f"component-swap producer fields {fold}")
        _require(record.get("fold") == fold, "component-swap pairing order changed")
        heldout = _mapping(record.get("heldout_data"), f"component-swap heldout data {fold}")
        _require(heldout.get("bank_key") == _bank_key(fold, "heldout-data", 0), "component-swap heldout role changed")
        _require(heldout.get("row_selector") == "fold_id_by_source_row == f", "component-swap heldout selector changed")
        _require(heldout.get("rows") == HELDOUT_ROWS_PER_FOLD and heldout.get("fit_rows_excluded") is True, "component-swap heldout rows changed")
        _require(heldout.get("heldout_positions_sha256") == lineage_fold.get("heldout_positions_sha256"), "component-swap heldout positions changed")

        data_pairs = _sequence(record.get("data_vs_model"), f"component-swap data pairs {fold}")
        _require(len(data_pairs) == DATA_VS_MODEL_REPLICATES, "component-swap data pair count changed")
        for replicate, item in enumerate(data_pairs):
            pair = _mapping(item, f"component-swap data pair {fold}/{replicate}")
            key = _bank_key(fold, "data-vs-model", replicate)
            _require(pair.get("replicate") == replicate and pair.get("model_bank_key") == key, "component-swap data role changed")
            _require(pair.get("model_q_sha256") == identities[key].get("expected_q_sha256"), "component-swap data bank hash binding changed")
            _require(pair.get("original_raw_metric_field") == producer_field["data_vs_model_metric_fields"][replicate], "component-swap raw data metric field changed")

        reference_pairs = _sequence(record.get("model_vs_model_reference"), f"component-swap reference pairs {fold}")
        expected_records = [
            {
                "pair_index": index,
                "left_key": _bank_key(fold, "model-vs-model", left),
                "right_key": _bank_key(fold, "model-vs-model", right),
                "left_replicate": left,
                "right_replicate": right,
            }
            for index, (left, right) in enumerate(MODEL_REFERENCE_PAIRS)
        ]
        _require(list(reference_pairs) == expected_records, "component-swap reference role pairing changed")
        _require(record.get("original_raw_q95_metric_field") == producer_field["q95_metric_field"], "component-swap q95 field changed")
        _require(record.get("original_model_reference_metric_fields") == producer_field["model_reference_metric_fields"], "component-swap model-reference fields changed")


def _validate_static_preflight(repo_root: Path, git_before: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    _require(git_before.get("branch") == EXPECTED_BRANCH, "current branch is not the N3 authority branch")
    _require(git_before.get("head") == EXPECTED_STARTING_HEAD, "starting committed HEAD changed")
    unexpected = sorted(set(git_before.get("dirty_paths", [])) - SAFE_DIRTY_PATHS)
    _require(not unexpected, f"unrelated local changes invalidate diagnostic provenance: {unexpected}")
    diff_check = subprocess.run(
        ["git", "-C", str(repo_root), "diff", "--check"],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(diff_check.returncode == 0, f"git diff --check failed before component-swap diagnostic: {diff_check.stdout}{diff_check.stderr}")

    contract_path = _resolve_relative(repo_root, CONTRACT_PATH.as_posix(), "component-swap contract path")
    contract = _read_json_exact(repo_root, CONTRACT_PATH, EXPECTED_CONTRACT_SHA256, "component-swap contract")
    _validate_contract_shape(contract)
    governance = _validate_contract_alias(repo_root, contract_path)
    authority_info = _validate_authority_bindings(repo_root, contract)
    _require(not output_dir.exists(), f"refusing to reuse write-once diagnostic output: {output_dir}")
    return {
        "contract_path": contract_path,
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "contract": contract,
        "governance": governance,
        "authority_info": authority_info,
        "static_checks": {
            "contract_full_sha256_verified_before_q_load": True,
            "contract_totality_verified_before_q_load": True,
            "contract_governance_alias_verified_before_q_load": True,
            "attribution_contracts_and_report_verified_before_q_load": True,
            "original_producer_contract_and_report_verified_before_q_load": True,
            "fold_identity_verified_before_q_load": True,
            "recovery_manifest_and_tensor_file_sha_verified_before_tensor_load": True,
            "exact_pair_roles_verified_before_q_load": True,
            "raw_metric_source_hashes_verified_before_q_load": True,
            "dirty_scope_verified_before_q_load": True,
            "sealed_confirmation_excluded_before_q_load": True,
            "output_write_once_absence_verified_before_q_load": True,
        },
    }


def _load_development_after_preflight(
    repo_root: Path,
    preflight: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    q, source_indices, labels, pool_identity = comparison._load_development_pool(repo_root)
    contract = _mapping(preflight.get("contract"), "component-swap contract")
    development = _mapping(_mapping(contract.get("authorities"), "component-swap authorities").get("development_data"), "component-swap development authority")
    _require(pool_identity.get("q_r2_amplitude_sha256") == development.get("q_r2_amplitude_sha256"), "loaded development q identity changed")
    _require(pool_identity.get("source_indices_sha256") == comparison.EXPECTED_SOURCE_INDEX_SHA256, "loaded source-index identity changed")
    _require(pool_identity.get("labels_sha256") == comparison.EXPECTED_SOURCE_LABEL_SHA256, "loaded label identity changed")
    _require(tuple(q.shape) == (POOL_COUNT, DIMENSION) and q.dtype == torch.float64, "loaded development q shape/dtype changed")
    _require(bool(torch.isfinite(q).all()) and bool(q.ge(0.0).all()), "loaded development q is invalid")

    producer_report = _mapping(_mapping(preflight["authority_info"], "preflight authority info").get("producer_report_payload"), "original producer report")
    report_fold = _mapping(producer_report.get("development_fold_identity"), "original producer fold identity")
    fold_ids, fold_identity = comparison._assign_folds(source_indices, labels)
    _require(fold_identity == report_fold, "resolved fold identity differs from original producer report")
    _require(torch.equal(fold_ids, torch.tensor(report_fold["fold_id_by_source_row"], dtype=torch.int64)), "resolved fold row map differs from original producer report")
    fold_authority = _mapping(_mapping(contract.get("authorities"), "component-swap authorities").get("fold_assignment"), "component-swap fold authority")
    _require(fold_identity.get("assignment_sha256") == fold_authority.get("assignment_sha256") == EXPECTED_FOLD_ASSIGNMENT_SHA256, "loaded fold assignment identity changed")

    fold_summaries: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        held_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1)
        fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1)
        _require(int(held_positions.numel()) == HELDOUT_ROWS_PER_FOLD, f"held-out row count changed for fold {fold}")
        _require(int(fit_positions.numel()) == FIT_ROWS_PER_FOLD, f"fit row count changed for fold {fold}")
        _require(torch.unique(torch.cat([fit_positions, held_positions])).numel() == POOL_COUNT, f"fit/held-out overlap for fold {fold}")
        contract_fold = _mapping(_sequence(_mapping(contract.get("pairing"), "component-swap pairing").get("folds"), "component-swap pairing folds")[fold], f"component-swap pairing fold {fold}")
        heldout = _mapping(contract_fold.get("heldout_data"), f"component-swap heldout data {fold}")
        _require(tensor_sha256(held_positions) == heldout.get("heldout_positions_sha256"), f"held-out position identity changed for fold {fold}")
        fold_summaries.append(
            {
                "fold": fold,
                "fit_rows": FIT_ROWS_PER_FOLD,
                "heldout_rows": HELDOUT_ROWS_PER_FOLD,
                "fit_positions_sha256": tensor_sha256(fit_positions),
                "heldout_positions_sha256": tensor_sha256(held_positions),
                "fit_heldout_disjoint": True,
            }
        )
    return q, source_indices, labels, fold_ids, {
        "pool_identity": pool_identity,
        "fold_identity": fold_identity,
        "folds": fold_summaries,
        "loaded_after_static_preflight": True,
    }


def _load_recovered_banks_after_preflight(
    repo_root: Path,
    preflight: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    contract = _mapping(preflight.get("contract"), "component-swap contract")
    authorities = _mapping(contract.get("authorities"), "component-swap authorities")
    frozen = _mapping(authorities.get("frozen_generated_banks"), "frozen generated bank authority")
    tensor_ref = _mapping(frozen.get("tensor_artifact"), "frozen tensor artifact")
    tensor_path = _resolve_relative(repo_root, str(tensor_ref.get("path")), "frozen tensor artifact path")
    tensor_sha = str(tensor_ref.get("sha256"))
    _require(file_sha256(tensor_path) == tensor_sha, "frozen tensor artifact changed after static preflight")
    payload = safe_torch_load(tensor_path, expected_sha256=tensor_sha, map_location="cpu")
    payload_map = _mapping(payload, "frozen generated bank payload")
    _require(payload_map.get("schema") == "n3_frozen_copula_generated_banks.v1", "frozen bank payload schema changed")
    _require(payload_map.get("semantic_authority_id") == "FROZEN_N3_COPULA_GENERATED_BANKS_V1", "frozen bank semantic authority changed")
    _require(payload_map.get("version") == 1, "frozen bank payload version changed")
    _require(payload_map.get("coordinate") == "R2_AMPLITUDE", "frozen bank coordinate changed")
    _require(payload_map.get("dtype") == "torch.float64", "frozen bank dtype changed")
    _require(payload_map.get("bank_count") == FOLD_COUNT * (DATA_VS_MODEL_REPLICATES + MODEL_VS_MODEL_REPLICATES), "frozen bank count changed")
    _require(payload_map.get("bank_shape") == [HELDOUT_ROWS_PER_FOLD, DIMENSION], "frozen bank shape changed")
    source_contract = _mapping(payload_map.get("source_contract"), "frozen bank source contract")
    source_report = _mapping(payload_map.get("source_execution_report"), "frozen bank source report")
    original_contract = _mapping(authorities.get("original_producer_contract"), "original producer contract authority")
    original_report = _mapping(authorities.get("original_producer_report"), "original producer report authority")
    _require(source_contract.get("path") == original_contract.get("path") and source_contract.get("sha256") == original_contract.get("sha256"), "frozen bank source contract binding changed")
    _require(source_report.get("path") == original_report.get("path") and source_report.get("sha256") == original_report.get("sha256"), "frozen bank source report binding changed")
    fold_authority = _mapping(authorities.get("fold_assignment"), "component-swap fold authority")
    _require(payload_map.get("fold_assignment_sha256") == fold_authority.get("assignment_sha256"), "frozen bank fold binding changed")

    bank_values = _mapping(payload_map.get("banks"), "frozen bank values")
    identities = {
        str(_mapping(item, "frozen bank identity").get("bank_key")): _mapping(item, "frozen bank identity")
        for item in _sequence(frozen.get("bank_identities"), "frozen bank identity registry")
    }
    expected_keys = {
        _bank_key(fold, role, replicate)
        for fold in range(FOLD_COUNT)
        for role, replicate in MODEL_BANK_ORDER
    }
    _require(set(identities) == expected_keys == set(bank_values), "frozen bank keys changed")
    loaded: dict[str, torch.Tensor] = {}
    recovery_identities = {
        str(_mapping(item, "recovery bank identity").get("bank_key")): _mapping(item, "recovery bank identity")
        for item in _sequence(preflight["authority_info"]["recovery_info"]["bank_identities"], "recovery bank identities")
    }
    for key in sorted(expected_keys):
        identity = identities[key]
        recovery_identity = recovery_identities[key]
        bank = bank_values[key]
        _require(torch.is_tensor(bank), f"frozen bank is not a tensor: {key}")
        _require(tuple(bank.shape) == (HELDOUT_ROWS_PER_FOLD, DIMENSION), f"frozen bank shape changed: {key}")
        _require(str(bank.dtype) == "torch.float64", f"frozen bank dtype changed: {key}")
        _require(bool(torch.isfinite(bank).all()) and bool(bank.ge(0.0).all()), f"frozen bank values invalid: {key}")
        observed_hash = tensor_sha256(bank)
        _require(observed_hash == identity.get("expected_q_sha256") == recovery_identity.get("expected_q_sha256"), f"frozen bank identity mismatch: {key}")
        loaded[key] = bank.detach().cpu().contiguous()
    return loaded, {
        "path": _relative(repo_root, tensor_path),
        "sha256": tensor_sha,
        "semantic_authority_id": payload_map["semantic_authority_id"],
        "bank_count": len(loaded),
        "bank_identities_verified": True,
        "loaded_as_tensor_after_static_preflight": True,
    }


def _ordinal_rank_bundle(values: torch.Tensor, label: str) -> dict[str, Any]:
    view = values.detach().to(dtype=torch.float64, device="cpu").contiguous()
    _require(tuple(view.shape) == (HELDOUT_ROWS_PER_FOLD, DIMENSION), f"rank input shape changed: {label}")
    _require(bool(torch.isfinite(view).all()) and bool(view.ge(0.0).all()), f"rank input values invalid: {label}")
    n, dimension = view.shape
    sorted_values = torch.empty_like(view)
    ranks = torch.empty((n, dimension), dtype=torch.int64)
    positions = torch.arange(n, dtype=torch.int64)
    tie_coordinates = 0
    tie_blocks = 0
    for coordinate in range(dimension):
        order = torch.argsort(view[:, coordinate], stable=True)
        ordered = view[:, coordinate].index_select(0, order)
        sorted_values[:, coordinate] = ordered
        ranks[order, coordinate] = positions
        equal_adjacent = ordered[1:] == ordered[:-1]
        if bool(equal_adjacent.any()):
            tie_coordinates += 1
            tie_blocks += int(equal_adjacent.sum())
            previous_rows = order[:-1][equal_adjacent]
            next_rows = order[1:][equal_adjacent]
            _require(bool(next_rows.gt(previous_rows).all()), f"exact-tie original row order changed: {label}|coordinate={coordinate}")

    expected_permutation = positions[:, None].expand(n, dimension)
    _require(torch.equal(torch.sort(ranks, dim=0).values, expected_permutation), f"rank is not a per-coordinate permutation: {label}")
    reconstruction = sorted_values.gather(0, ranks)
    _require(torch.equal(reconstruction, view), f"self reconstruction failed: {label}")
    return {
        "values": view,
        "sorted_values": sorted_values.contiguous(),
        "ranks": ranks.contiguous(),
        "identity": {
            "label": label,
            "shape": [HELDOUT_ROWS_PER_FOLD, DIMENSION],
            "dtype": "torch.float64",
            "stable_numeric_sort": True,
            "exact_tie_original_row_index_order_verified": True,
            "rank_per_coordinate_permutation_verified": True,
            "self_reconstruction_verified": True,
            "tie_coordinates": tie_coordinates,
            "tie_adjacent_blocks": tie_blocks,
            "no_rng": True,
            "deterministic": True,
        },
    }


def _energy_distance_value(first: torch.Tensor, second: torch.Tensor, label: str) -> float:
    value = comparison.validated_energy_distance(first, second)
    number = _finite_number(value, f"raw ED {label}")
    _require(number >= 0.0, f"raw ED is negative after frozen policy: {label}")
    return number


def _hybrids(data_rank: Mapping[str, Any], model_rank: Mapping[str, Any], label: str) -> tuple[torch.Tensor, torch.Tensor]:
    heldout = _mapping(data_rank, f"data rank {label}")
    model = _mapping(model_rank, f"model rank {label}")
    h_marg = heldout["sorted_values"].gather(0, model["ranks"]).contiguous()
    h_dep = model["sorted_values"].gather(0, heldout["ranks"]).contiguous()
    for name, hybrid in (("H_MARG", h_marg), ("H_DEP", h_dep)):
        _require(tuple(hybrid.shape) == (HELDOUT_ROWS_PER_FOLD, DIMENSION), f"{name} shape changed: {label}")
        _require(hybrid.dtype == torch.float64, f"{name} dtype changed: {label}")
        _require(bool(torch.isfinite(hybrid).all()) and bool(hybrid.ge(0.0).all()), f"{name} values invalid: {label}")
    return h_marg, h_dep


def _measure_fold(
    fold: int,
    q: torch.Tensor,
    fold_ids: torch.Tensor,
    banks: Mapping[str, torch.Tensor],
    model_ranks: Mapping[str, Mapping[str, Any]],
    pairing_fold: Mapping[str, Any],
    producer_fold: Mapping[str, Any],
) -> dict[str, Any]:
    held_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1)
    fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1)
    data = q.index_select(0, held_positions).contiguous()
    data_rank = _ordinal_rank_bundle(data, f"fold={fold}|role=heldout-data")
    pair_results: list[dict[str, Any]] = []
    failure_labels: list[str] = []
    control_labels: list[str] = []
    q95 = _finite_number(
        _mapping(producer_fold.get("model_vs_model_reference_q95"), f"original q95 fold {fold}").get("energy_distance"),
        f"original q95 fold {fold}",
    )
    reference_values = [
        _finite_number(
            _mapping(item, f"original model reference fold {fold}/{index}").get("energy_distance"),
            f"original model reference fold {fold}/{index}",
        )
        for index, item in enumerate(_sequence(producer_fold.get("model_vs_model_reference_pairs"), f"original model references fold {fold}"))
    ]
    _require(len(reference_values) == len(MODEL_REFERENCE_PAIRS), f"original model reference count changed for fold {fold}")

    for replicate, pairing_item in enumerate(_sequence(pairing_fold.get("data_vs_model"), f"component-swap data pairs fold {fold}")):
        pair = _mapping(pairing_item, f"component-swap data pair fold {fold}/{replicate}")
        key = str(pair.get("model_bank_key"))
        model = banks[key]
        model_rank = model_ranks[key]
        h_marg, h_dep = _hybrids(data_rank, model_rank, f"fold={fold}|replicate={replicate}")
        historical = _finite_number(
            _sequence(producer_fold.get("data_vs_model_replicates"), f"original data metrics fold {fold}")[replicate].get("energy_distance"),
            f"original E_BASE fold {fold}/{replicate}",
        )
        e_base = _energy_distance_value(data, model, f"fold={fold}|E_BASE|replicate={replicate}")
        _require(e_base == historical, f"E_BASE identity mismatch for fold={fold}|replicate={replicate}")
        e_marg = _energy_distance_value(data, h_marg, f"fold={fold}|E_MARG|replicate={replicate}")
        e_dep = _energy_distance_value(data, h_dep, f"fold={fold}|E_DEP|replicate={replicate}")
        original_failure = historical > q95
        marginal_rescue = e_marg <= q95
        dependence_rescue = e_dep <= q95
        pair_label = contract_definition.pair_sufficiency_label(marginal_rescue, dependence_rescue)
        if original_failure:
            failure_labels.append(pair_label)
        else:
            control_labels.append(pair_label)
        pair_results.append(
            {
                "fold": fold,
                "replicate": replicate,
                "pair_role": "historical DATA-vs-MODEL comparison",
                "model_bank_key": key,
                "model_q_sha256": pair["model_q_sha256"],
                "original_raw_metric_field": pair["original_raw_metric_field"],
                "original_primary_metric": historical,
                "recomputed_E_BASE": e_base,
                "E_BASE_identity_verified": True,
                "frozen_original_q95_metric_field": pairing_fold["original_raw_q95_metric_field"],
                "frozen_original_q95": q95,
                "original_primary_status": "ORIGINAL_PRIMARY_FAIL" if original_failure else "ORIGINAL_PRIMARY_PASS_CONTROL",
                "metrics": {
                    "E_BASE": e_base,
                    "E_MARG": e_marg,
                    "E_DEP": e_dep,
                },
                "oracle_outcomes": {
                    "marginal_oracle_rescues": marginal_rescue,
                    "dependence_oracle_rescues": dependence_rescue,
                    "threshold_rule": "<= frozen fold-local original q95",
                },
                "pair_label": pair_label,
                "included_in_original_failure_sufficiency_aggregate": original_failure,
                "control_only": not original_failure,
                "hybrids": {
                    "H_MARG": {
                        "equation": "H_MARG[i,j] = sorted_D[r_M(i,j),j]",
                        "shape": [HELDOUT_ROWS_PER_FOLD, DIMENSION],
                        "dtype": "torch.float64",
                        "annotations": list(HYBRID_ANNOTATIONS),
                    },
                    "H_DEP": {
                        "equation": "H_DEP[i,j] = sorted_M[r_D(i,j),j]",
                        "shape": [HELDOUT_ROWS_PER_FOLD, DIMENSION],
                        "dtype": "torch.float64",
                        "annotations": list(HYBRID_ANNOTATIONS),
                    },
                },
            }
        )

    fold_label = contract_definition.fold_sufficiency_label(failure_labels)
    return {
        "fold": fold,
        "operationally_valid": True,
        "fit_rows": int(fit_positions.numel()),
        "heldout_rows": int(held_positions.numel()),
        "fit_heldout_disjoint": True,
        "fit_positions_sha256": tensor_sha256(fit_positions),
        "heldout_positions_sha256": tensor_sha256(held_positions),
        "heldout_data_rank_identity": data_rank["identity"],
        "pairs": pair_results,
        "original_primary": {
            "q95_metric_field": pairing_fold["original_raw_q95_metric_field"],
            "frozen_q95": q95,
            "model_reference_pair_values": reference_values,
            "pair_failure_rule": "historical primary value > historical fold-local original q95",
            "failure_pair_count": len(failure_labels),
            "control_pair_count": len(control_labels),
        },
        "original_failure_pair_labels": failure_labels,
        "original_pass_control_pair_labels": control_labels,
        "fold_oracle_sufficiency_label": fold_label,
    }


def _build_report(
    repo_root: Path,
    preflight: Mapping[str, Any],
    git_before: Mapping[str, Any],
    git_after: Mapping[str, Any],
    q: torch.Tensor,
    source_indices: torch.Tensor,
    labels: torch.Tensor,
    lineage: Mapping[str, Any],
    recovered_info: Mapping[str, Any],
    model_rank_identities: Sequence[Mapping[str, Any]],
    fold_results: Sequence[Mapping[str, Any]],
    aggregate_label: str,
    failure_labels: Sequence[str],
) -> dict[str, Any]:
    contract = _mapping(preflight.get("contract"), "component-swap contract")
    authority_info = _mapping(preflight.get("authority_info"), "preflight authority info")
    fold_pattern = {str(item["fold"]): item["fold_oracle_sufficiency_label"] for item in fold_results}
    control_count = sum(int(item["original_primary"]["control_pair_count"]) for item in fold_results)
    return {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "classification": aggregate_label,
        "classification_vocabulary": list(_mapping(contract["aggregate_oracle_sufficiency"], "aggregate contract")["vocabulary"]),
        "contract": {
            "path": _relative(repo_root, preflight["contract_path"]),
            "sha256": preflight["contract_sha256"],
            "schema": contract["schema"],
            "task_id": contract["task_id"],
            "classification": contract["classification"],
            "claim_classification": contract["claim_classification"],
            "immutable_consumed_unchanged": True,
        },
        "input_verification": {
            **dict(preflight["static_checks"]),
            "development_q_loaded_after_static_preflight": True,
            "recovered_tensor_loaded_after_static_preflight": recovered_info["loaded_as_tensor_after_static_preflight"],
            "attribution_report": authority_info["attribution_report"],
            "original_producer_contract": authority_info["producer_contract"],
            "original_producer_report": authority_info["producer_report"],
            "recovery_manifest": authority_info["recovery_info"]["manifest"],
            "recovered_tensor": {
                "path": recovered_info["path"],
                "sha256": recovered_info["sha256"],
                "bank_count": recovered_info["bank_count"],
                "bank_identities_verified": recovered_info["bank_identities_verified"],
            },
            "fold_assignment": authority_info["fold_assignment"],
            "development_q_tensor_sha256": tensor_sha256(q),
            "development_source_indices_sha256": tensor_sha256(source_indices),
            "development_labels_sha256": tensor_sha256(labels),
            "sealed_confirmation_consumed": False,
        },
        "execution_settings": {
            "coordinate": "R2_AMPLITUDE",
            "representation": "raw q-space",
            "device": "cpu",
            "dtype": "torch.float64",
            "fold_count": FOLD_COUNT,
            "heldout_rows_per_fold": HELDOUT_ROWS_PER_FOLD,
            "dimension": DIMENSION,
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "model_reference_pairs": [list(pair) for pair in MODEL_REFERENCE_PAIRS],
            "primary_quantile": PRIMARY_QUANTILE,
            "q95_reused_from_original_report": True,
            "q95_recomputed_from_oracle_hybrids": False,
            "rng_used": False,
            "energy_distance_implementation": "scripts.research.validate_n3_amplitude_producer_baseline._energy_distance",
            "energy_distance_negative_policy": "require raw >= -1e-10; otherwise invalid_harness; return max(0.0,raw)",
        },
        "development_lineage": {
            "coordinate": "R2_AMPLITUDE",
            "row_selector": "fold_id_by_source_row == f",
            "fit_rows_per_fold": FIT_ROWS_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_ROWS_PER_FOLD,
            "fit_rows_excluded_from_heldout": True,
            "fold_identity": lineage["fold_identity"],
            "folds": lineage["folds"],
            "data_reuse_limitation": "All five folds reuse the fixed 1000-row N3 development pool; each oracle hybrid uses the corresponding held-out 200-row DATA bank. This is a development-only oracle diagnostic, not independent confirmation or population-truth evidence.",
        },
        "rank_identity_invariants": {
            "ordinal_total_order": "numeric value ascending, original row index ascending for exact ties",
            "data_bank_count": FOLD_COUNT,
            "model_bank_count": len(model_rank_identities),
            "data_self_reconstruction_verified": all(item["heldout_data_rank_identity"]["self_reconstruction_verified"] for item in fold_results),
            "model_self_reconstruction_verified": all(item["self_reconstruction_verified"] for item in model_rank_identities),
            "data_rank_per_coordinate_permutations_verified": all(item["heldout_data_rank_identity"]["rank_per_coordinate_permutation_verified"] for item in fold_results),
            "model_rank_per_coordinate_permutations_verified": all(item["rank_per_coordinate_permutation_verified"] for item in model_rank_identities),
            "exact_tie_order_verified": all(item["heldout_data_rank_identity"]["exact_tie_original_row_index_order_verified"] for item in fold_results) and all(item["exact_tie_original_row_index_order_verified"] for item in model_rank_identities),
            "no_rng": True,
            "deterministic": True,
        },
        "model_rank_identity_registry": [dict(item) for item in model_rank_identities],
        "fold_results": list(fold_results),
        "aggregate": {
            "classification": aggregate_label,
            "fold_pattern": fold_pattern,
            "original_failure_pair_labels": list(failure_labels),
            "original_failure_pair_count": len(failure_labels),
            "original_pass_control_pair_count": control_count,
            "pair_label_output_count": sum(len(item["pairs"]) for item in fold_results),
            "mapping_function": "aggregate_oracle_sufficiency_label(fold_labels, original_failure_pair_labels)",
            "fold_labels_include_controls": True,
            "controls_excluded_from_sufficiency_aggregate": True,
            "exactly_one_aggregate_oracle_sufficiency_label_emitted": True,
            "development_only": True,
        },
        "claim_boundary": {
            "diagnostic_only": True,
            "oracle_hybrid_interpretation": "An oracle rescue is sufficient only for the stated idealized held-out-data component intervention under the frozen raw primary metric.",
            "not_established": [
                "a learnable producer",
                "generalization",
                "a valid tail family",
                "a valid copula family",
                "N3 producer support",
                "complex OLS support",
                "semantic generation",
                "confirmation",
                "generation readiness",
            ],
            "hybrids_are_not": [
                "generated samples",
                "producer outputs",
                "confirmation samples",
                "admissible generation evidence",
            ],
        },
        "outer_gate": {
            "result": "VALID_COMPLETED",
            "question": "Under the frozen component-swap contract, what exact development-only oracle sufficiency classification is produced for the original N3 primary-failing DATA-vs-MODEL comparisons?",
            "answer": aggregate_label,
            "only_permitted_downstream_direction": "P1 planning may use this diagnostic to define one bounded intervention while holding the complementary component fixed; no execution or promotion follows from this report.",
            "p1_execution_authorized": False,
        },
        "governance": {
            "selection_scope": "DEVELOPMENT_ONLY",
            "component_swap_diagnostic_executed": True,
            "oracle_metrics_computed": True,
            "producer_fit": False,
            "producer_sampling": False,
            "bank_rematerialization": False,
            "sealed_confirmation_status": "RESERVED_NOT_MATERIALIZED",
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
            "deterministic_bank_rematerialization": False,
            "new_family_selection": False,
            "new_metric": False,
            "new_pairing": False,
            "sealed_confirmation_read": False,
            "complex_ols": False,
            "generation": False,
            "n4_n5_n6": False,
            "p1_execution": False,
            "generation_promotion": False,
            "measurement_only": True,
        },
        "git": {
            "branch": git_before["branch"],
            "starting_head": git_before["head"],
            "ending_head": git_after["head"],
            "worktree_dirty_before": list(git_before["dirty_paths"]),
            "worktree_dirty_after_measurement": list(git_after["dirty_paths"]),
            "consumer_script": _relative(repo_root, Path(__file__).resolve()),
            "consumer_script_sha256": file_sha256(Path(__file__).resolve()),
            "committed": False,
            "new_commit_created": False,
        },
        "not_executed": [
            "producer family definition, selection, fit, or sampling",
            "new random pairing or rank selection",
            "bank rematerialization",
            "sealed confirmation source loading or materialization",
            "CUDA",
            "optimizer",
            "backward",
            "training",
            "complex OLS",
            "generation",
            "N4/N5/N6",
            "P1 execution",
            "formal promotion",
        ],
    }


def _write_report(repo_root: Path, report: Mapping[str, Any], output_dir: Path) -> Path:
    del repo_root
    _require(not output_dir.exists(), f"refusing to reuse write-once diagnostic output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(
            temporary / REPORT_NAME,
            report,
            artifact_kind="N3 mixed component-swap diagnostic report",
        )
        _require(not output_dir.exists(), f"diagnostic output appeared during publication: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / REPORT_NAME


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = _resolve_relative(repo_root, output_dir.as_posix(), "diagnostic output directory")
    git_before = _git_state(repo_root)
    preflight = _validate_static_preflight(repo_root, git_before, output_dir)

    banks, recovered_info = _load_recovered_banks_after_preflight(repo_root, preflight)
    q, source_indices, labels, fold_ids, lineage = _load_development_after_preflight(repo_root, preflight)

    model_rank_bundles: dict[str, dict[str, Any]] = {}
    for key in sorted(banks):
        model_rank_bundles[key] = _ordinal_rank_bundle(banks[key], key)
    model_rank_identities = [model_rank_bundles[key]["identity"] for key in sorted(model_rank_bundles)]

    contract_pairing = _mapping(preflight["contract"].get("pairing"), "component-swap pairing")
    pairing_folds = _sequence(contract_pairing.get("folds"), "component-swap pairing folds")
    producer_report = _mapping(preflight["authority_info"]["producer_report_payload"], "original producer report")
    producer_folds = _sequence(producer_report.get("fold_execution"), "original producer fold execution")
    fold_results = [
        _measure_fold(
            fold,
            q,
            fold_ids,
            banks,
            model_rank_bundles,
            _mapping(pairing_folds[fold], f"component-swap pairing fold {fold}"),
            _mapping(producer_folds[fold], f"original producer fold {fold}"),
        )
        for fold in range(FOLD_COUNT)
    ]
    fold_labels = {int(item["fold"]): str(item["fold_oracle_sufficiency_label"]) for item in fold_results}
    failure_labels = [
        str(label)
        for item in fold_results
        for label in item["original_failure_pair_labels"]
    ]
    aggregate_label = contract_definition.aggregate_oracle_sufficiency_label(fold_labels, failure_labels)
    git_after = _git_state(repo_root)
    _require(git_after["branch"] == git_before["branch"] and git_after["head"] == git_before["head"], "measurement changed branch or HEAD")
    _require(git_after["dirty_paths"] == git_before["dirty_paths"], "measurement changed the worktree")
    report = _build_report(
        repo_root,
        preflight,
        git_before,
        git_after,
        q,
        source_indices,
        labels,
        lineage,
        recovered_info,
        model_rank_identities,
        fold_results,
        aggregate_label,
        failure_labels,
    )
    report_path = _write_report(repo_root, report, output_dir)
    return {
        "result": "VALID_COMPLETED",
        "gate": GATE,
        "aggregate_classification": aggregate_label,
        "report": _relative(repo_root, report_path),
        "report_sha256": file_sha256(report_path),
        "contract": _relative(repo_root, preflight["contract_path"]),
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "fold_pattern": {str(fold): fold_labels[fold] for fold in range(FOLD_COUNT)},
        "pair_label_count": sum(len(item["pairs"]) for item in fold_results),
        "original_failure_pair_count": len(failure_labels),
        "q_values_loaded_after_precheck": True,
        "recovered_tensor_loaded_after_precheck": True,
        "oracle_metrics_computed": True,
        "rng_used": False,
        "sealed_confirmation_consumed": False,
        "p1_execution_authorized": False,
        "cuda": False,
        "training": False,
        "git": report["git"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        result = execute(args.repo_root, args.output_dir)
    except BlockedExecution as exc:
        print(json.dumps({"result": "BLOCKED", "gate": GATE, "reason": str(exc)}, indent=2, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
