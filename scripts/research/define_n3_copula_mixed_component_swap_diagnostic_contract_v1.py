"""Define the N3 raw-q component-swap diagnostic contract.

This definition consumer binds the historical raw-q comparison and the exact
recovered bank roles without loading q tensors or computing a hybrid metric.
The published artifact is a write-once, development-only semantic authority
for a later diagnostic consumer; it is not a producer contract or a result.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256


TASK_ID = "DEFINE_N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_CONTRACT_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "n3_copula_mixed_component_swap_diagnostic_contract.v1"
READY_CLASSIFICATION = "DEFINE_N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_CONTRACT_V1_READY"
CLAIM_CLASSIFICATION = "N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_CONTRACT_DEFINED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"

ATTRIBUTION_CONTRACT_V1 = Path(
    "outputs/20260908_v2/runs/"
    "n3_copula_failure_attribution_contract_v1_attempt_01/"
    "n3_copula_failure_attribution_contract.v1.json"
)
ATTRIBUTION_CONTRACT_V1_SHA256 = "333b51abe2a6b1f3e96558273d4b6037267aa80678199d3bf6d0c300bb3508df"
ATTRIBUTION_CONTRACT_V2 = Path(
    "outputs/20260908_v2/runs/"
    "n3_copula_failure_attribution_contract_v2_attempt_01/"
    "n3_copula_failure_attribution_contract.v2.json"
)
ATTRIBUTION_CONTRACT_V2_SHA256 = "5c85c4e949e5850de337d776f6782bdc791493b333ab9dd7982269e9f214eb95"
ATTRIBUTION_REPORT = Path(
    "outputs/20260909_v2/runs/"
    "n3_copula_failure_attribution_v1_attempt_01/"
    "n3_copula_failure_attribution_report.v1.json"
)
ATTRIBUTION_REPORT_SHA256 = "e9a834150baa006c619256b8306340e31593e1c9562a37ea46955b34ae86cac8"

PRODUCER_CONTRACT = Path(
    "outputs/20260908_v2/runs/"
    "n3_empirical_marginal_gaussian_copula_producer_contract_v1_attempt_01/"
    "n3_empirical_marginal_gaussian_copula_producer_contract.v1.json"
)
PRODUCER_CONTRACT_SHA256 = "3edc9fee006f68eff6ba41205b7e4e6b6cbea3205080c4cb333689e554fba7a4"
PRODUCER_REPORT = Path(
    "outputs/20260908_v2/runs/"
    "n3_empirical_marginal_gaussian_copula_producer_v1_attempt_01/"
    "n3_empirical_marginal_gaussian_copula_producer_report.v1.json"
)
PRODUCER_REPORT_SHA256 = "78b8e40f21015af8d704a89a83bb6b4043c0208386032b41b1ced4dd9bdeba7d"
RECOVERY_MANIFEST = Path(
    "outputs/20260908_v2/runs/"
    "n3_frozen_copula_bank_recovery_v1_attempt_01/"
    "n3_frozen_copula_generated_banks_manifest.v1.json"
)
RECOVERY_MANIFEST_SHA256 = "4ed6ac4839cdb075f98d07be3162ac560ea29cc260fcbd0e272a342aed04297c"
RECOVERED_TENSOR = Path(
    "outputs/20260908_v2/runs/"
    "n3_frozen_copula_bank_recovery_v1_attempt_01/"
    "n3_frozen_copula_generated_banks.v1.pt"
)
RECOVERED_TENSOR_SHA256 = "6b9da40cf868fb9033e5c45aa05778c27199867430d2ec5627b36875d284b42b"

PRODUCER_EXECUTION_SCRIPT = Path(
    "scripts/research/execute_n3_empirical_marginal_gaussian_copula_producer_v1.py"
)
PRODUCER_EXECUTION_SCRIPT_SHA256 = "5f1fa3fa6be35e96f345c3b7f7c6cce5ca0b45fe4619b1ead4455e86dcae538a"
PRIMARY_METRIC_SOURCE = Path("scripts/research/validate_n3_amplitude_producer_baseline.py")
PRIMARY_METRIC_SOURCE_SHA256 = "1ad916bbb8d109ca989f64b9f5590391cdffea9e0567e2ca8d241b032c6cde36"
PRIMARY_METRIC_WRAPPER = Path("scripts/research/compare_n3_minimal_competing_producer_baselines_v1.py")
PRIMARY_METRIC_WRAPPER_SHA256 = "5e49a45e88eb4b5308fb24975aac6fda5e5031e38c46c939170c55ffddbc83f0"

FOLD_COUNT = 5
DIMENSION = 100
HELDOUT_ROWS_PER_FOLD = 200
DATA_VS_MODEL_REPLICATES = 4
MODEL_VS_MODEL_REPLICATES = 8
MODEL_REFERENCE_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
PRIMARY_QUANTILE = 0.95
PRIMARY_RAW_NEGATIVE_ATOL = 1.0e-10
OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n3_copula_mixed_component_swap_diagnostic_contract_v1_attempt_01"
)
CONTRACT_NAME = "n3_copula_mixed_component_swap_diagnostic_contract.v1.json"

PAIR_LABELS = {
    (True, False): "MARGINAL_ORACLE_ONLY",
    (False, True): "DEPENDENCE_ORACLE_ONLY",
    (True, True): "BOTH_ORACLES",
    (False, False): "NEITHER_ORACLE",
}
PAIR_LABEL_VALUES = frozenset(PAIR_LABELS.values())
FOLD_CONTROL = "CONTROL_FOLD"
FOLD_MIXED = "FOLD_MIXED"
FOLD_LABEL_VALUES = PAIR_LABEL_VALUES | {FOLD_CONTROL, FOLD_MIXED}
AGGREGATE_LABELS = {
    "MARGINAL_ORACLE_ONLY": "N3_COPULA_PRIMARY_MARGINAL_ORACLE_SUFFICIENT",
    "DEPENDENCE_ORACLE_ONLY": "N3_COPULA_PRIMARY_DEPENDENCE_ORACLE_SUFFICIENT",
    "BOTH_ORACLES": "N3_COPULA_PRIMARY_BOTH_ORACLES_SUFFICIENT",
    "NEITHER_ORACLE": "N3_COPULA_PRIMARY_NEITHER_ORACLE_SUFFICIENT",
}
AGGREGATE_MIXED = "N3_COPULA_PRIMARY_ORACLE_EFFECT_MIXED"

SAFE_DIRTY_PATHS = {
    "docs/CURRENT_STATUS.md",
    "docs/status/current_handoff.md",
    "docs/status/experiment_history.md",
    "docs/status/research_governance_state.json",
    "scripts/research/define_n3_copula_failure_attribution_contract_v1.py",
    "scripts/research/define_n3_copula_failure_attribution_contract_v2.py",
    "scripts/research/execute_n3_copula_failure_attribution_v1.py",
    "scripts/research/define_n3_copula_mixed_component_swap_diagnostic_contract_v1.py",
    "tests/v2/test_n3_copula_failure_attribution_contract.py",
    "tests/v2/test_n3_copula_failure_attribution_contract_v2.py",
    "tests/v2/test_execute_n3_copula_failure_attribution_v1.py",
    "tests/v2/test_n3_copula_mixed_component_swap_diagnostic_contract_v1.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> Sequence[Any]:
    _require(isinstance(value, Sequence) and not isinstance(value, (str, bytes)), f"{name} must be a sequence")
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
    }


def _bool(value: Any, name: str) -> bool:
    _require(type(value) is bool, f"{name} must be a boolean")
    return bool(value)


def _finite_number(value: Any, name: str) -> float:
    _require(type(value) in (int, float), f"{name} must be a JSON number")
    number = float(value)
    _require(math.isfinite(number), f"{name} must be finite")
    return number


def _bank_key(fold: int, role: str, replicate: int) -> str:
    return f"fold={fold}|role={role}|replicate={replicate}"


def pair_sufficiency_label(marginal_oracle_rescues: bool, dependence_oracle_rescues: bool) -> str:
    """Map the two validated oracle booleans to the complete pair vocabulary."""

    _require(type(marginal_oracle_rescues) is bool, "marginal oracle outcome must be boolean")
    _require(type(dependence_oracle_rescues) is bool, "dependence oracle outcome must be boolean")
    return PAIR_LABELS[(marginal_oracle_rescues, dependence_oracle_rescues)]


def fold_sufficiency_label(original_failure_pair_labels: Sequence[str]) -> str:
    """Return the conservative fold label for originally failing pairs only."""

    labels = list(original_failure_pair_labels)
    _require(len(labels) <= DATA_VS_MODEL_REPLICATES, "fold has too many DATA-vs-MODEL labels")
    _require(all(label in PAIR_LABEL_VALUES for label in labels), "fold contains an unknown pair label")
    if not labels:
        return FOLD_CONTROL
    return labels[0] if len(set(labels)) == 1 else FOLD_MIXED


def aggregate_oracle_sufficiency_label(
    fold_labels: Mapping[int, str],
    original_failure_pair_labels: Sequence[str],
) -> str:
    """Map valid fold/pair outcomes to a total oracle-only aggregate label."""

    _require(set(fold_labels) == set(range(FOLD_COUNT)), "aggregate requires all five folds")
    _require(all(label in FOLD_LABEL_VALUES for label in fold_labels.values()), "aggregate has an unknown fold label")
    pair_labels = list(original_failure_pair_labels)
    _require(all(label in PAIR_LABEL_VALUES for label in pair_labels), "aggregate has an unknown pair label")
    if not pair_labels or FOLD_MIXED in fold_labels.values():
        return AGGREGATE_MIXED
    unique = set(pair_labels)
    if len(unique) == 1:
        return AGGREGATE_LABELS[next(iter(unique))]
    return AGGREGATE_MIXED


def _validate_attribution_contracts(
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    v1 = _read_json_exact(repo_root, ATTRIBUTION_CONTRACT_V1, ATTRIBUTION_CONTRACT_V1_SHA256, "attribution V1 contract")
    _require(v1.get("schema") == "n3_copula_failure_attribution_contract.v1", "attribution V1 schema changed")
    _require(v1.get("classification") == "DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V1_READY", "attribution V1 is not ready")

    v2 = _read_json_exact(repo_root, ATTRIBUTION_CONTRACT_V2, ATTRIBUTION_CONTRACT_V2_SHA256, "attribution V2 contract")
    _require(v2.get("schema") == "n3_copula_failure_attribution_contract.v2", "attribution V2 schema changed")
    _require(v2.get("classification") == "DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2_READY", "attribution V2 is not ready")
    _require(v2.get("base_contract_sha256") == ATTRIBUTION_CONTRACT_V1_SHA256, "attribution V2 base contract changed")
    _require(v2.get("v1_primary_semantics_unchanged") is True, "attribution V2 changed primary semantics")
    supersession = _mapping(v2.get("supersession"), "attribution V2 supersession")
    _require(supersession.get("primary_scientific_semantics_changed") is False, "attribution V2 changed primary semantics")
    diff = _mapping(v2.get("semantic_diff_validation"), "attribution V2 semantic diff")
    _require(diff.get("status") == "passed", "attribution V2 semantic diff is not passed")
    _require(diff.get("primary_scientific_semantics_unchanged") is True, "attribution V2 semantic invariant changed")
    _require(isinstance(v2.get("bank_role_contract"), Mapping), "attribution V2 bank role contract is missing")

    report = _read_json_exact(repo_root, ATTRIBUTION_REPORT, ATTRIBUTION_REPORT_SHA256, "latest attribution report")
    _require(report.get("schema") == "n3_copula_failure_attribution_report.v1", "latest attribution report schema changed")
    _require(report.get("status") == "valid_completed", "latest attribution report is not complete")
    _require(report.get("classification") == "N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED", "latest attribution result changed")
    report_contract = _mapping(report.get("contract"), "latest attribution report contract")
    _require(report_contract.get("path") == _relative(repo_root, repo_root / ATTRIBUTION_CONTRACT_V2), "latest attribution contract path changed")
    _require(report_contract.get("sha256") == ATTRIBUTION_CONTRACT_V2_SHA256, "latest attribution contract identity changed")
    _require(report.get("task_id") == "EXECUTE_N3_COPULA_FAILURE_ATTRIBUTION_V1", "latest attribution task changed")
    _require(report.get("gate") == report.get("task_id"), "latest attribution gate changed")

    lineage = _mapping(report.get("development_lineage"), "latest attribution lineage")
    fold_identity = _mapping(lineage.get("fold_identity"), "latest attribution fold identity")
    _require(fold_identity.get("assignment_sha256") == "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32", "latest attribution fold assignment changed")
    _require(fold_identity.get("fold_count") == FOLD_COUNT, "latest attribution fold count changed")
    _require(lineage.get("coordinate") == "R2_AMPLITUDE", "latest attribution coordinate changed")

    governance = _mapping(report.get("governance"), "latest attribution governance")
    _require(governance.get("attribution_executed") is True, "latest attribution execution flag changed")
    _require(governance.get("metrics_computed") is True, "latest attribution metric flag changed")
    _require(governance.get("sealed_confirmation_consumed") is False, "latest attribution consumed sealed confirmation")
    _require(governance.get("training_authorized") is False, "latest attribution training authorization changed")
    _require(governance.get("formal_promotion") is False, "latest attribution formal promotion changed")
    _require(governance.get("generation_readiness") == "not_ready", "latest attribution generation boundary changed")
    outer_gate = _mapping(report.get("outer_gate"), "latest attribution outer gate")
    _require(outer_gate.get("p1_execution_authorized") is False, "latest attribution authorized P1 execution")

    expected_fold_labels = {
        "0": "MARGINAL_AND_DEPENDENCE",
        "1": "DEPENDENCE_ONLY",
        "2": "MARGINAL_AND_DEPENDENCE",
        "3": "MARGINAL_AND_DEPENDENCE",
        "4": "DEPENDENCE_ONLY",
    }
    aggregate = _mapping(report.get("aggregate"), "latest attribution aggregate")
    _require(aggregate.get("fold_pattern") == expected_fold_labels, "latest attribution fold pattern changed")
    folds = _sequence(report.get("fold_results"), "latest attribution fold results")
    _require(len(folds) == FOLD_COUNT, "latest attribution fold results are incomplete")
    for expected_fold, item in enumerate(folds):
        fold = _mapping(item, f"latest attribution fold {expected_fold}")
        _require(fold.get("fold") == expected_fold, "latest attribution fold order changed")
        marginal = _mapping(fold.get("marginal_authority"), f"latest attribution fold {expected_fold} marginal authority")
        _bool(marginal.get("M"), f"latest attribution fold {expected_fold} M")
        _bool(marginal.get("marginal_fail"), f"latest attribution fold {expected_fold} marginal_fail")
        _require(marginal.get("M") == marginal.get("marginal_fail"), "latest attribution M mapping changed")
        dependence = _mapping(fold.get("primary_dependence"), f"latest attribution fold {expected_fold} dependence authority")
        _bool(dependence.get("D"), f"latest attribution fold {expected_fold} D")
        _bool(dependence.get("dependence_fail"), f"latest attribution fold {expected_fold} dependence_fail")
        _require(dependence.get("D") == dependence.get("dependence_fail"), "latest attribution D mapping changed")
        _require(fold.get("fold_classification") == expected_fold_labels[str(expected_fold)], "latest attribution fold classification changed")
        data_pairs = _sequence(dependence.get("data_vs_model"), f"latest attribution fold {expected_fold} data comparisons")
        model_pairs = _sequence(dependence.get("model_vs_model_reference_pairs"), f"latest attribution fold {expected_fold} model comparisons")
        _require(len(data_pairs) == DATA_VS_MODEL_REPLICATES, "latest attribution data pair count changed")
        _require(len(model_pairs) == len(MODEL_REFERENCE_PAIRS), "latest attribution model pair count changed")
        _require(_finite_number(_mapping(dependence.get("model_vs_model_reference_q95"), "latest attribution q95").get("value"), "latest attribution q95 value") >= 0.0, "latest attribution q95 is negative")

    return v1, v2, report


def _validate_producer_contract(contract: Mapping[str, Any]) -> None:
    _require(contract.get("schema") == "n3_empirical_marginal_gaussian_copula_producer_contract.v1", "producer contract schema changed")
    _require(contract.get("classification") == "DEFINE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1_READY", "producer contract is not ready")
    _require(contract.get("claim_classification") == "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_MODEL_ASSUMPTION_DEFINED", "producer contract claim changed")
    evaluation = _mapping(contract.get("development_evaluation_contract"), "producer evaluation contract")
    _require(evaluation.get("primary_metric") == "validated_biased_empirical_energy_distance_euclidean_q_space", "producer primary metric changed")
    _require(evaluation.get("primary_metric_implementation") == "scripts.research.validate_n3_amplitude_producer_baseline._energy_distance", "producer primary implementation changed")
    _require(evaluation.get("fit_rows") == 800 and evaluation.get("heldout_rows") == HELDOUT_ROWS_PER_FOLD, "producer fold row counts changed")
    _require(evaluation.get("labels_as_model_inputs") is False, "producer labels became model inputs")
    compatibility = _mapping(contract.get("compatibility_contract"), "producer compatibility contract")
    _require(compatibility.get("q") == PRIMARY_QUANTILE, "producer compatibility quantile changed")
    _require(compatibility.get("own_model_reference_required") is True, "producer model reference requirement changed")
    _require(compatibility.get("all_folds_required") is True, "producer all-fold rule changed")
    _require("fixed-replicate mean DATA_VS_MODEL <= q95 MODEL_VS_MODEL" in str(compatibility.get("rule")), "producer q95 rule changed")


def _validate_producer_report(report: Mapping[str, Any], contract: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    _require(report.get("schema") == "n3_empirical_marginal_gaussian_copula_producer_report.v1", "producer report schema changed")
    _require(report.get("status") == "valid_completed", "producer report is not complete")
    _require(report.get("classification") == "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED", "producer classification changed")
    report_contract = _mapping(report.get("contract"), "producer report contract")
    _require(report_contract.get("sha256") == PRODUCER_CONTRACT_SHA256, "producer report contract identity changed")
    pool = _mapping(report.get("development_pool"), "producer development pool")
    _require(pool.get("count") == 1000 and pool.get("dimension") == DIMENSION, "producer development pool changed")
    _require(pool.get("q_coordinate") == "R2_AMPLITUDE", "producer q coordinate changed")
    fold_identity = _mapping(report.get("development_fold_identity"), "producer fold identity")
    _require(fold_identity.get("assignment_sha256") == "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32", "producer fold assignment changed")
    _require(fold_identity.get("fold_count") == FOLD_COUNT, "producer fold count changed")
    sealed = _mapping(report.get("sealed_confirmation"), "producer sealed boundary")
    _require(sealed.get("status") == "RESERVED_NOT_MATERIALIZED", "producer sealed status changed")
    _require(sealed.get("consumed") is False, "producer consumed sealed confirmation")
    governance = _mapping(report.get("governance"), "producer governance")
    _require(governance.get("training_authorized") is False, "producer training authorization changed")
    _require(governance.get("formal_promotion") is False, "producer formal promotion changed")
    _require(_mapping(report.get("claim_boundary"), "producer claim boundary").get("generation_readiness") == "not_ready", "producer generation boundary changed")

    producer_git = _mapping(report.get("git"), "producer report git")
    _require(producer_git.get("consumer_script") == _relative(repo_root, repo_root / PRODUCER_EXECUTION_SCRIPT), "producer execution script path changed")
    script_path = repo_root / PRODUCER_EXECUTION_SCRIPT
    _require(script_path.is_file(), "producer execution script is missing")
    _require(file_sha256(script_path) == PRODUCER_EXECUTION_SCRIPT_SHA256, "producer execution script hash changed")

    folds = _sequence(report.get("fold_execution"), "producer fold execution")
    _require(len(folds) == FOLD_COUNT, "producer fold execution is incomplete")
    original_pair_fields: list[dict[str, Any]] = []
    for expected_fold, item in enumerate(folds):
        fold = _mapping(item, f"producer fold {expected_fold}")
        _require(fold.get("fold") == expected_fold, "producer fold order changed")
        _bool(fold.get("compatibility"), f"producer fold {expected_fold} compatibility")
        components = _mapping(fold.get("compatibility_components"), f"producer fold {expected_fold} compatibility components")
        _bool(components.get("energy_distance"), f"producer fold {expected_fold} primary compatibility component")
        _require(
            fold.get("compatibility_rule")
            == "fixed-replicate mean DATA_VS_MODEL <= q95 MODEL_VS_MODEL for primary, joint and aggregate marginal guardrails",
            f"producer fold {expected_fold} compatibility rule changed",
        )
        data_sampling = _sequence(fold.get("data_sampling"), f"producer fold {expected_fold} data sampling")
        model_sampling = _sequence(fold.get("model_sampling"), f"producer fold {expected_fold} model sampling")
        _require(len(data_sampling) == DATA_VS_MODEL_REPLICATES, "producer data bank count changed")
        _require(len(model_sampling) == MODEL_VS_MODEL_REPLICATES, "producer model bank count changed")
        for replicate, sample in enumerate(data_sampling):
            sample_map = _mapping(sample, f"producer fold {expected_fold} data sample {replicate}")
            _require(sample_map.get("fold") == expected_fold, "producer data sample fold changed")
            _require(sample_map.get("replicate") == replicate and sample_map.get("role") == "data-vs-model", "producer data sample role changed")
            _bool(sample_map.get("q_finite"), "producer data q finite flag")
            _bool(sample_map.get("q_nonnegative"), "producer data q nonnegative flag")
        for replicate, sample in enumerate(model_sampling):
            sample_map = _mapping(sample, f"producer fold {expected_fold} model sample {replicate}")
            _require(sample_map.get("fold") == expected_fold, "producer model sample fold changed")
            _require(sample_map.get("replicate") == replicate and sample_map.get("role") == "model-vs-model", "producer model sample role changed")
            _bool(sample_map.get("q_finite"), "producer model q finite flag")
            _bool(sample_map.get("q_nonnegative"), "producer model q nonnegative flag")

        data_metrics = _sequence(fold.get("data_vs_model_replicates"), f"producer fold {expected_fold} data metrics")
        model_metrics = _sequence(fold.get("model_vs_model_reference_pairs"), f"producer fold {expected_fold} model metrics")
        _require(len(data_metrics) == DATA_VS_MODEL_REPLICATES, "producer data metric count changed")
        _require(len(model_metrics) == len(MODEL_REFERENCE_PAIRS), "producer model metric count changed")
        for replicate, metric in enumerate(data_metrics):
            _finite_number(_mapping(metric, f"producer data metric {expected_fold}/{replicate}").get("energy_distance"), "producer raw data energy distance")
        for pair_index, metric in enumerate(model_metrics):
            _finite_number(_mapping(metric, f"producer model metric {expected_fold}/{pair_index}").get("energy_distance"), "producer raw model energy distance")
        data_mean = _mapping(fold.get("data_vs_model_mean"), f"producer fold {expected_fold} data mean")
        _finite_number(data_mean.get("energy_distance"), "producer raw data mean energy distance")
        reference_q95 = _mapping(fold.get("model_vs_model_reference_q95"), f"producer fold {expected_fold} raw q95")
        _finite_number(reference_q95.get("energy_distance"), "producer raw q95 energy distance")
        original_pair_fields.append(
            {
                "fold": expected_fold,
                "data_vs_model_metric_fields": [
                    f"fold_execution[{expected_fold}].data_vs_model_replicates[{replicate}].energy_distance"
                    for replicate in range(DATA_VS_MODEL_REPLICATES)
                ],
                "q95_metric_field": f"fold_execution[{expected_fold}].model_vs_model_reference_q95.energy_distance",
                "model_reference_metric_fields": [
                    f"fold_execution[{expected_fold}].model_vs_model_reference_pairs[{pair_index}].energy_distance"
                    for pair_index in range(len(MODEL_REFERENCE_PAIRS))
                ],
            }
        )

    input_authorities = _mapping(report.get("input_authorities"), "producer report input authorities")
    bank_authorities = _mapping(input_authorities.get("development_bank_artifacts"), "producer development bank authorities")
    _require(bank_authorities.get("q_values_loaded_by_this_definition_consumer") is False, "producer definition consumer q-load flag changed")
    return {
        "development_pool": dict(pool),
        "fold_identity": dict(fold_identity),
        "original_pair_fields": original_pair_fields,
        "input_authorities": dict(input_authorities),
        "producer_script_sha256": PRODUCER_EXECUTION_SCRIPT_SHA256,
    }


def _validate_recovery_manifest(
    manifest: Mapping[str, Any],
    producer_report: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    _require(manifest.get("schema") == "n3_frozen_copula_generated_banks_manifest.v1", "recovery manifest schema changed")
    _require(manifest.get("status") == "valid_completed", "recovery manifest is not complete")
    _require(manifest.get("classification") == "FROZEN_N3_COPULA_BANK_AUTHORITY_RECOVERED", "recovery classification changed")
    _require(manifest.get("semantic_authority_id") == "FROZEN_N3_COPULA_GENERATED_BANKS_V1", "recovery semantic authority changed")
    _require(_mapping(manifest.get("source_contract"), "recovery source contract").get("sha256") == PRODUCER_CONTRACT_SHA256, "recovery source contract changed")
    _require(_mapping(manifest.get("source_execution_report"), "recovery source report").get("sha256") == PRODUCER_REPORT_SHA256, "recovery source report changed")
    _require(manifest.get("fold_assignment_sha256") == "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32", "recovery fold assignment changed")
    _require(manifest.get("bank_count") == FOLD_COUNT * (DATA_VS_MODEL_REPLICATES + MODEL_VS_MODEL_REPLICATES), "recovery bank count changed")
    _require(manifest.get("per_bank_hash_equality") is True, "recovery per-bank hash equality changed")
    shape_dtype = _mapping(manifest.get("shape_dtype_contract"), "recovery shape/dtype contract")
    _require(shape_dtype.get("shape") == [HELDOUT_ROWS_PER_FOLD, DIMENSION], "recovery bank shape changed")
    _require(shape_dtype.get("dtype") == "torch.float64", "recovery bank dtype changed")
    sealed = _mapping(manifest.get("sealed_confirmation"), "recovery sealed boundary")
    _require(sealed.get("status") == "RESERVED_NOT_MATERIALIZED", "recovery sealed status changed")
    _require(sealed.get("consumed") is False, "recovery consumed sealed confirmation")
    _require(manifest.get("historical_evidence_unchanged") is True, "recovery changed historical evidence")

    artifact = _mapping(manifest.get("artifact"), "recovery tensor artifact")
    tensor_path = repo_root / RECOVERED_TENSOR
    _require(artifact.get("path") == _relative(repo_root, tensor_path), "recovery tensor path changed")
    _require(artifact.get("sha256") == RECOVERED_TENSOR_SHA256, "recovery tensor hash registration changed")
    _require(tensor_path.is_file(), "recovered tensor artifact is missing")
    _require(file_sha256(tensor_path) == RECOVERED_TENSOR_SHA256, "recovered tensor artifact hash changed")

    identities = _sequence(manifest.get("per_bank_identity"), "recovery bank identity registry")
    _require(len(identities) == manifest.get("bank_count"), "recovery bank identity registry is incomplete")
    by_key: dict[str, dict[str, Any]] = {}
    for item in identities:
        entry = dict(_mapping(item, "recovery bank identity"))
        key = str(entry.get("bank_key"))
        _require(key not in by_key, f"duplicate recovery bank key: {key}")
        _require(entry.get("bank_size") == HELDOUT_ROWS_PER_FOLD, f"recovery bank size changed: {key}")
        _require(entry.get("dtype") == "torch.float64", f"recovery bank dtype changed: {key}")
        _require(entry.get("shape") == [HELDOUT_ROWS_PER_FOLD, DIMENSION], f"recovery bank shape changed: {key}")
        _require(entry.get("hash_equal") is True and entry.get("seed_verified") is True, f"recovery bank identity not verified: {key}")
        _require(entry.get("expected_q_sha256") == entry.get("observed_q_sha256"), f"recovery q hash mismatch: {key}")
        by_key[key] = entry
    expected_keys = {
        _bank_key(fold, "data-vs-model", replicate)
        for fold in range(FOLD_COUNT)
        for replicate in range(DATA_VS_MODEL_REPLICATES)
    } | {
        _bank_key(fold, "model-vs-model", replicate)
        for fold in range(FOLD_COUNT)
        for replicate in range(MODEL_VS_MODEL_REPLICATES)
    }
    _require(set(by_key) == expected_keys, "recovery bank-role registry changed")

    report_hashes: dict[str, str] = {}
    for fold_item in _sequence(producer_report.get("fold_execution"), "producer fold execution for recovery binding"):
        fold = _mapping(fold_item, "producer fold for recovery binding")
        fold_id = int(fold["fold"])
        for sample in _sequence(fold.get("data_sampling"), f"recovery data samples fold {fold_id}"):
            sample_map = _mapping(sample, "recovery data sample")
            key = _bank_key(fold_id, "data-vs-model", int(sample_map["replicate"]))
            _require(sample_map.get("q_sha256") == by_key[key]["expected_q_sha256"], f"producer/recovery q hash mismatch: {key}")
            report_hashes[key] = str(sample_map["q_sha256"])
        for sample in _sequence(fold.get("model_sampling"), f"recovery model samples fold {fold_id}"):
            sample_map = _mapping(sample, "recovery model sample")
            key = _bank_key(fold_id, "model-vs-model", int(sample_map["replicate"]))
            _require(sample_map.get("q_sha256") == by_key[key]["expected_q_sha256"], f"producer/recovery q hash mismatch: {key}")
            report_hashes[key] = str(sample_map["q_sha256"])
    _require(set(report_hashes) == expected_keys, "producer/recovery bank role registry is incomplete")
    return {
        "manifest": {
            "path": _relative(repo_root, repo_root / RECOVERY_MANIFEST),
            "sha256": RECOVERY_MANIFEST_SHA256,
            "semantic_authority_id": manifest["semantic_authority_id"],
        },
        "tensor_artifact": {
            "path": _relative(repo_root, tensor_path),
            "sha256": RECOVERED_TENSOR_SHA256,
            "loaded_as_tensor": False,
            "identity_hash_verified": True,
        },
        "bank_count": len(by_key),
        "bank_identities": [by_key[key] for key in sorted(by_key)],
        "report_recovery_q_hashes_equal": True,
    }


def _validate_metric_sources(repo_root: Path) -> dict[str, Any]:
    sources = (
        (PRIMARY_METRIC_SOURCE, PRIMARY_METRIC_SOURCE_SHA256, "raw q-space primary metric source"),
        (PRIMARY_METRIC_WRAPPER, PRIMARY_METRIC_WRAPPER_SHA256, "raw q-space primary metric wrapper"),
    )
    for path, expected_sha256, name in sources:
        target = repo_root / path
        _require(target.is_file(), f"{name} is missing")
        _require(file_sha256(target) == expected_sha256, f"{name} hash changed")
    return {
        "implementation": "scripts.research.validate_n3_amplitude_producer_baseline._energy_distance",
        "source_path": _relative(repo_root, repo_root / PRIMARY_METRIC_SOURCE),
        "source_sha256": PRIMARY_METRIC_SOURCE_SHA256,
        "historical_wrapper_path": _relative(repo_root, repo_root / PRIMARY_METRIC_WRAPPER),
        "historical_wrapper_sha256": PRIMARY_METRIC_WRAPPER_SHA256,
        "producer_execution_script_path": _relative(repo_root, repo_root / PRODUCER_EXECUTION_SCRIPT),
        "producer_execution_script_sha256": PRODUCER_EXECUTION_SCRIPT_SHA256,
    }


def _build_pairing(
    repo_root: Path,
    attribution_v2: Mapping[str, Any],
    attribution_report: Mapping[str, Any],
    producer_info: Mapping[str, Any],
    recovery_info: Mapping[str, Any],
) -> dict[str, Any]:
    lineage = _mapping(attribution_report["development_lineage"], "attribution lineage")
    lineage_folds = _sequence(lineage["folds"], "attribution lineage folds")
    _require(len(lineage_folds) == FOLD_COUNT, "attribution lineage fold registry is incomplete")
    v2_bank_contract = _mapping(attribution_v2["bank_role_contract"], "attribution V2 bank role contract")
    data_contract = _mapping(v2_bank_contract["data_source"], "attribution data source contract")
    _require(data_contract.get("row_selector") == "fold_id_by_source_row == f", "held-out row selector changed")
    _require(data_contract.get("per_fold_rows") == HELDOUT_ROWS_PER_FOLD, "held-out row count changed")

    identities = {str(item["bank_key"]): dict(item) for item in recovery_info["bank_identities"]}
    fold_records: list[dict[str, Any]] = []
    for fold in range(FOLD_COUNT):
        lineage_fold = _mapping(lineage_folds[fold], f"attribution lineage fold {fold}")
        model_keys = [_bank_key(fold, "data-vs-model", replicate) for replicate in range(DATA_VS_MODEL_REPLICATES)]
        reference_pairs = [
            {
                "pair_index": pair_index,
                "left_key": _bank_key(fold, "model-vs-model", left),
                "right_key": _bank_key(fold, "model-vs-model", right),
                "left_replicate": left,
                "right_replicate": right,
            }
            for pair_index, (left, right) in enumerate(MODEL_REFERENCE_PAIRS)
        ]
        for key in model_keys + [value for pair in reference_pairs for value in (pair["left_key"], pair["right_key"])]:
            _require(key in identities, f"pairing references unknown recovered bank: {key}")
        producer_pair_fields = _mapping(producer_info["original_pair_fields"][fold], f"producer pair fields fold {fold}")
        fold_records.append(
            {
                "fold": fold,
                "heldout_data": {
                    "role": "heldout-data",
                    "replicate": 0,
                    "bank_key": _bank_key(fold, "heldout-data", 0),
                    "row_selector": "fold_id_by_source_row == f",
                    "rows": HELDOUT_ROWS_PER_FOLD,
                    "heldout_positions_sha256": lineage_fold["heldout_positions_sha256"],
                    "fit_rows_excluded": True,
                },
                "data_vs_model": [
                    {
                        "replicate": replicate,
                        "model_bank_key": key,
                        "model_q_sha256": identities[key]["expected_q_sha256"],
                        "original_raw_metric_field": producer_pair_fields["data_vs_model_metric_fields"][replicate],
                    }
                    for replicate, key in enumerate(model_keys)
                ],
                "model_vs_model_reference": reference_pairs,
                "original_raw_q95_metric_field": producer_pair_fields["q95_metric_field"],
                "original_model_reference_metric_fields": list(producer_pair_fields["model_reference_metric_fields"]),
            }
        )
    return {
        "historical_pairing_only": True,
        "pairing_source": {
            "original_copula_report": {
                "path": _relative(repo_root, repo_root / PRODUCER_REPORT),
                "sha256": PRODUCER_REPORT_SHA256,
            },
            "recovery_manifest": {
                "path": _relative(repo_root, repo_root / RECOVERY_MANIFEST),
                "sha256": RECOVERY_MANIFEST_SHA256,
            },
            "recovered_tensor": {
                "path": _relative(repo_root, repo_root / RECOVERED_TENSOR),
                "sha256": RECOVERED_TENSOR_SHA256,
            },
            "fold_assignment_sha256": "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32",
        },
        "data_bank": {
            "role": "held-out DATA bank",
            "source": "original R2_AMPLITUDE development pool",
            "row_selector": "fold_id_by_source_row == f",
            "rows_per_fold": HELDOUT_ROWS_PER_FOLD,
            "dimension": DIMENSION,
            "fit_rows_excluded": True,
        },
        "model_banks": {
            "role": "role=data-vs-model",
            "replicates": list(range(DATA_VS_MODEL_REPLICATES)),
            "bank_key_template": "fold={f}|role=data-vs-model|replicate={r}",
            "one_to_one_with_historical_data_comparisons": True,
        },
        "model_reference_banks": {
            "role": "role=model-vs-model",
            "replicates": list(range(MODEL_VS_MODEL_REPLICATES)),
            "reference_pairs": [
                {
                    "pair_index": pair_index,
                    "left_replicate": left,
                    "right_replicate": right,
                }
                for pair_index, (left, right) in enumerate(MODEL_REFERENCE_PAIRS)
            ],
            "no_cross_fold_pairing": True,
        },
        "folds": fold_records,
        "forbidden_pairing_operations": [
            "new random pairing",
            "cross-fold pairing",
            "new sampling",
            "best-bank selection",
            "weighted bank selection",
        ],
    }


def _contract_payload(
    repo_root: Path,
    git: Mapping[str, Any],
    attribution_v1: Mapping[str, Any],
    attribution_v2: Mapping[str, Any],
    attribution_report: Mapping[str, Any],
    producer_contract: Mapping[str, Any],
    producer_report: Mapping[str, Any],
    producer_info: Mapping[str, Any],
    recovery_info: Mapping[str, Any],
    metric_sources: Mapping[str, Any],
) -> dict[str, Any]:
    attribution_authorities = _mapping(attribution_v2["authorities"], "attribution V2 authorities")
    development_data = _mapping(attribution_authorities["development_data"], "attribution development data")
    fold_assignment = _mapping(attribution_authorities["fold_assignment"], "attribution fold assignment")
    _require(fold_assignment.get("assignment_sha256") == "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32", "attribution fold authority changed")

    pairing = _build_pairing(repo_root, attribution_v2, attribution_report, producer_info, recovery_info)
    fold_label_values = sorted(PAIR_LABEL_VALUES | {FOLD_CONTROL, FOLD_MIXED})
    aggregate_values = [AGGREGATE_LABELS[label] for label in ("MARGINAL_ORACLE_ONLY", "DEPENDENCE_ORACLE_ONLY", "BOTH_ORACLES", "NEITHER_ORACLE")] + [AGGREGATE_MIXED]
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "classification": READY_CLASSIFICATION,
        "claim_classification": CLAIM_CLASSIFICATION,
        "semantic_authority_id": TASK_ID,
        "new_contract": True,
        "historical_preregistration": False,
        "definition_only": True,
        "git": {
            **dict(git),
            "starting_committed_head": EXPECTED_STARTING_HEAD,
            "contract_script": _relative(repo_root, Path(__file__).resolve()),
            "contract_script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "authorities": {
            "attribution_contract_v1": {
                "path": _relative(repo_root, repo_root / ATTRIBUTION_CONTRACT_V1),
                "sha256": ATTRIBUTION_CONTRACT_V1_SHA256,
                "schema": attribution_v1["schema"],
            },
            "attribution_contract_v2": {
                "path": _relative(repo_root, repo_root / ATTRIBUTION_CONTRACT_V2),
                "sha256": ATTRIBUTION_CONTRACT_V2_SHA256,
                "schema": attribution_v2["schema"],
                "primary_scientific_semantics_changed": False,
            },
            "latest_attribution_report": {
                "path": _relative(repo_root, repo_root / ATTRIBUTION_REPORT),
                "sha256": ATTRIBUTION_REPORT_SHA256,
                "schema": attribution_report["schema"],
                "classification": attribution_report["classification"],
                "immutable_result_binding": True,
            },
            "original_producer_contract": {
                "path": _relative(repo_root, repo_root / PRODUCER_CONTRACT),
                "sha256": PRODUCER_CONTRACT_SHA256,
                "schema": producer_contract["schema"],
                "classification": producer_contract["classification"],
            },
            "original_producer_report": {
                "path": _relative(repo_root, repo_root / PRODUCER_REPORT),
                "sha256": PRODUCER_REPORT_SHA256,
                "schema": producer_report["schema"],
                "classification": producer_report["classification"],
            },
            "fold_assignment": {
                "source": "original_copula_execution_report.development_fold_identity",
                "algorithm": fold_assignment["algorithm"],
                "fold_count": FOLD_COUNT,
                "assignment_sha256": fold_assignment["assignment_sha256"],
            },
            "development_data": copy.deepcopy(dict(development_data)),
            "frozen_generated_banks": copy.deepcopy(recovery_info),
            "sealed_confirmation_exclusion": {
                "semantic_authority_id": "N3_SEALED_CONFIRMATION_SOURCE_V1",
                "input_authority": False,
                "path_read": False,
                "sample_values_loaded": False,
                "status": "RESERVED_NOT_MATERIALIZED",
                "consumed": False,
            },
        },
        "scope": {
            "question": "Under the frozen raw q-space primary discrepancy, was an idealized marginal-only or rank-dependence-only component intervention sufficient?",
            "selection_scope": "DEVELOPMENT_ONLY",
            "diagnostic_role": "oracle component-swap sufficiency only",
            "coordinate": "R2_AMPLITUDE",
            "dimension": DIMENSION,
            "rows_per_fold": HELDOUT_ROWS_PER_FOLD,
            "labels_as_model_inputs": False,
            "optical_forward": False,
            "evaluator": False,
            "producer_family_selection": False,
            "generated_sample_claim": False,
        },
        "pairing": pairing,
        "ordinal_rank_representation": {
            "purpose": "component-swap construction only; does not replace the frozen self-rank attribution result",
            "input_bank_shape": [HELDOUT_ROWS_PER_FOLD, DIMENSION],
            "input_dtype": "torch.float64",
            "rank_domain": "r_X(i,j) in {0,...,n-1}",
            "rank_index_base": 0,
            "rank_definition": "r_X(i,j) is the zero-based position of row i in the total order for coordinate j",
            "total_order_key": "(numeric X[i,j] ascending, original row index i ascending for exact ties)",
            "implementation_semantics": {
                "sort": "stable ascending numeric sort independently for each coordinate",
                "tie_key": "exact original row index; no tolerance, rounding, midrank, or random tie break",
                "inverse_permutation": "rank_at_original_row[i,j] = position of i in sorted_indices[:,j]",
                "order_statistics": "sorted_X[k,j] = X[sorted_indices[k,j],j]",
                "coordinate_order": "j=0..99",
                "row_order": "i=0..199 in each bank's stored row order",
            },
            "rank_scope": "each DATA or MODEL bank independently",
            "no_rng": True,
            "deterministic": True,
        },
        "marginal_oracle": {
            "name": "H_MARG",
            "inputs": ["held-out DATA bank D", "corresponding MODEL bank M"],
            "equation": "H_MARG[i,j] = sorted_D[r_M(i,j),j]",
            "sorted_data_definition": "sorted_D[:,j] is ascending order statistics of D[:,j] under the frozen ordinal ordering",
            "preserved_component": "MODEL cross-coordinate ordinal-rank dependence",
            "replaced_component": "empirical marginal values replaced by held-out DATA order statistics",
            "oracle_intervention_only": True,
        },
        "dependence_oracle": {
            "name": "H_DEP",
            "inputs": ["held-out DATA bank D", "corresponding MODEL bank M"],
            "equation": "H_DEP[i,j] = sorted_M[r_D(i,j),j]",
            "sorted_model_definition": "sorted_M[:,j] is ascending order statistics of M[:,j] under the frozen ordinal ordering",
            "preserved_component": "MODEL empirical marginal values",
            "replaced_component": "cross-coordinate ordinal-rank dependence replaced by held-out DATA ranks",
            "oracle_intervention_only": True,
        },
        "identity_invariants": {
            "model_self_reconstruction": "gather(sorted_M, r_M) == M elementwise",
            "data_self_reconstruction": "gather(sorted_D, r_D) == D elementwise",
            "rank_per_coordinate_is_permutation": True,
            "hybrid_shapes": [HELDOUT_ROWS_PER_FOLD, DIMENSION],
            "hybrid_dtype": "torch.float64",
            "finite_required": True,
            "nonnegative_required": True,
            "no_rng": True,
            "deterministic": True,
            "no_clamp_or_repair": True,
            "failure_status": "NUMERICAL_OR_IMPLEMENTATION_INVALID",
            "failure_action": "emit no scientific classification and stop before oracle sufficiency mapping",
        },
        "primary_diagnostic": {
            "representation": "raw q-space; do not use self-rank Gaussianized z values",
            "no_new_metric": True,
            "metrics": {
                "E_BASE": "ED(D,M)",
                "E_MARG": "ED(D,H_MARG)",
                "E_DEP": "ED(D,H_DEP)",
            },
            "raw_energy_distance_implementation": {
                **dict(metric_sources),
                "metric_name": "validated_biased_empirical_energy_distance_euclidean_q_space",
                "distance": "d(a,b)=sqrt(sum_j (a_j-b_j)^2)",
                "formula": "ED(X,Y)=2*mean_{i,k} d(X_i,Y_k)-mean_{i,i'} d(X_i,X_i')-mean_{k,k'} d(Y_k,Y_k')",
                "terms": {
                    "cross": "torch.cdist(X,Y,p=2.0).mean()",
                    "within_left": "torch.cdist(X,X,p=2.0).mean()",
                    "within_right": "torch.cdist(Y,Y,p=2.0).mean()",
                },
                "input_normalization": "detach, cast CPU float64, contiguous; require both matrices 2-D with equal coordinate count",
                "within_pairs": "include diagonal pairs; diagonal distances are zero",
                "raw_value": "raw=2*C-L-R",
                "nonfinite_policy": "nonfinite raw value is invalid_harness; emit no scientific classification",
                "negative_policy": "require raw >= -1e-10; otherwise invalid_harness; return max(0.0,raw)",
                "negative_tolerance": PRIMARY_RAW_NEGATIVE_ATOL,
                "oracle_metric_reuses_this_exact_implementation": True,
            },
            "frozen_model_reference_q95": {
                "source_report": _relative(repo_root, repo_root / PRODUCER_REPORT),
                "source_field_template": "fold_execution[f].model_vs_model_reference_q95.energy_distance",
                "reference_pair_metric_field_template": "fold_execution[f].model_vs_model_reference_pairs[p].energy_distance",
                "probability": PRIMARY_QUANTILE,
                "reference_pair_count": len(MODEL_REFERENCE_PAIRS),
                "reference_pair_order": [
                    {"pair_index": pair_index, "left_replicate": left, "right_replicate": right}
                    for pair_index, (left, right) in enumerate(MODEL_REFERENCE_PAIRS)
                ],
                "quantile_method": "torch.quantile over four scalar energy distances, default linear interpolation",
                "quantile_definition": "sort v[0..3], h=(4-1)*0.95=2.85, Q95=0.15*v[2]+0.85*v[3]",
                "fold_local_only": True,
                "threshold_is_frozen": True,
                "oracle_hybrid_values_in_q95": False,
                "cross_fold_pooling": False,
                "retuning": False,
            },
            "baseline_reconstruction": {
                "historical_value_field_template": "fold_execution[f].data_vs_model_replicates[r].energy_distance",
                "future_E_BASE_is_recomputed": True,
                "eligibility_is_bound_to_historical_field": True,
                "fold_level_compatibility_not_used_as_pair_failure": True,
            },
        },
        "pair_level_sufficiency": {
            "eligibility": {
                "historical_primary_value_field_template": "fold_execution[f].data_vs_model_replicates[r].energy_distance",
                "historical_q95_field_template": "fold_execution[f].model_vs_model_reference_q95.energy_distance",
                "original_primary_fail_condition": "historical primary value > historical fold-local q95",
                "original_primary_pass_condition": "historical primary value <= historical fold-local q95",
                "source_report_order_is_authoritative": True,
                "not_inferred_from_fold_compatibility": True,
                "not_inferred_from_latest_self_rank_attribution": True,
            },
            "rescue_booleans": {
                "marginal_oracle_rescues": "E_MARG <= frozen fold-local original q95",
                "dependence_oracle_rescues": "E_DEP <= frozen fold-local original q95",
                "strict_failure_boundary": "only > q95 is failure; equality is pass",
            },
            "label_mapping": {
                "marginal=true,dependence=false": "MARGINAL_ORACLE_ONLY",
                "marginal=false,dependence=true": "DEPENDENCE_ORACLE_ONLY",
                "marginal=true,dependence=true": "BOTH_ORACLES",
                "marginal=false,dependence=false": "NEITHER_ORACLE",
            },
            "originally_passing_control_policy": {
                "report_both_oracle_booleans": True,
                "use_pair_label_vocabulary_for_descriptive_control": True,
                "exclude_from_original_failure_sufficiency_aggregate": True,
                "do_not_reinterpret_as_original_failure_cause": True,
            },
            "pair_count_per_fold": DATA_VS_MODEL_REPLICATES,
        },
        "fold_aggregation": {
            "domain": "the four historical DATA-vs-MODEL comparisons in each fold",
            "input": "pair labels for originally primary-failing comparisons only",
            "unique_label_condition": "all originally primary-failing pair labels in the fold are identical",
            "disagreement_label": FOLD_MIXED,
            "no_original_failure_label": FOLD_CONTROL,
            "controls_are_descriptive_only": True,
            "weights": "none; equal historical pairing only",
            "best_bank_selection": False,
            "fold_label_vocabulary": fold_label_values,
            "mapping_function": "fold_sufficiency_label(original_failure_pair_labels)",
        },
        "aggregate_oracle_sufficiency": {
            "input": "all originally primary-failing pair labels across all five historical folds plus the five fold labels",
            "exclude": ["CONTROL_FOLD pair sets", "originally passing control comparisons"],
            "non_mixed_condition": "all relevant original primary-failing pair labels agree and no fold is FOLD_MIXED",
            "mapping_function": "aggregate_oracle_sufficiency_label(fold_labels, original_failure_pair_labels)",
            "vocabulary": aggregate_values,
            "mapping": {
                "all labels MARGINAL_ORACLE_ONLY": AGGREGATE_LABELS["MARGINAL_ORACLE_ONLY"],
                "all labels DEPENDENCE_ORACLE_ONLY": AGGREGATE_LABELS["DEPENDENCE_ORACLE_ONLY"],
                "all labels BOTH_ORACLES": AGGREGATE_LABELS["BOTH_ORACLES"],
                "all labels NEITHER_ORACLE": AGGREGATE_LABELS["NEITHER_ORACLE"],
                "any disagreement, any FOLD_MIXED, or no original failure pairs": AGGREGATE_MIXED,
            },
            "no_producer_support_classification": True,
            "no_selection_authority": True,
        },
        "claim_boundary": {
            "MARGINAL_ORACLE_SUFFICIENT": "If empirical marginals were replaced by held-out DATA marginals while retaining the MODEL rank-dependence pattern, that idealized intervention was sufficient under the frozen raw primary metric.",
            "DEPENDENCE_ORACLE_SUFFICIENT": "If rank dependence were replaced by the held-out DATA rank pattern while retaining MODEL marginals, that idealized intervention was sufficient under the frozen raw primary metric.",
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
            "hybrid_banks_are_not": [
                "generated samples",
                "producer outputs",
                "confirmation samples",
                "admissible generation evidence",
            ],
        },
        "execution_boundary": {
            "later_consumer_role": "one-shot development-only diagnostic only",
            "q_tensor_loaded_by_this_definition_consumer": False,
            "oracle_metrics_computed_by_this_definition_consumer": False,
            "component_swaps_executed_by_this_definition_consumer": False,
            "producer_fit": False,
            "producer_sampling": False,
            "bank_rematerialization": False,
            "new_sampling": False,
            "sealed_confirmation_read": False,
            "cuda": False,
            "optimizer": False,
            "backward": False,
            "training": False,
            "complex_ols": False,
            "generation": False,
            "n4_n5_n6": False,
            "formal_promotion": False,
            "training_authorized": False,
        },
        "contract_totality": {
            "status": "passed",
            "totality_proven": True,
            "future_runtime_math_choices": [],
            "validated_domains": {
                "pair_booleans": [[False, False], [False, True], [True, False], [True, True]],
                "fold_inputs": ["empty", "one unique pair label", "multiple pair labels"],
                "aggregate_inputs": ["empty failure set", "each single pair label", "mixed pair labels", "FOLD_MIXED"],
            },
            "failure_stops": [
                "unresolved authority",
                "ambiguous pairing",
                "metric or q95 schema drift",
                "identity invariant failure",
                "nonfinite or materially negative ED",
            ],
            "scientific_classification_requires": "a later valid execution after all identity and metric gates pass",
        },
        "artifact_policy": {
            "write_once": True,
            "immutable_after_publish": True,
            "definition_is_not_a_result": True,
            "existing_attribution_result_modified": False,
            "original_producer_result_modified": False,
        },
    }


def _write_contract(repo_root: Path, output_dir: Path, payload: Mapping[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(
            temporary / CONTRACT_NAME,
            payload,
            artifact_kind="N3 mixed component-swap diagnostic contract",
        )
        if output_dir.exists():
            raise FileExistsError(f"diagnostic contract output appeared during definition: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / CONTRACT_NAME


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 authority branch")
    _require(git["head"] == EXPECTED_STARTING_HEAD, "starting committed HEAD changed")
    unexpected_dirty = [path for path in git["dirty_paths"] if path not in SAFE_DIRTY_PATHS]
    _require(not unexpected_dirty, f"unexpected dirty paths before diagnostic contract definition: {unexpected_dirty}")

    attribution_v1, attribution_v2, attribution_report = _validate_attribution_contracts(repo_root)
    producer_contract = _read_json_exact(repo_root, PRODUCER_CONTRACT, PRODUCER_CONTRACT_SHA256, "original producer contract")
    producer_report = _read_json_exact(repo_root, PRODUCER_REPORT, PRODUCER_REPORT_SHA256, "original producer report")
    recovery_manifest = _read_json_exact(repo_root, RECOVERY_MANIFEST, RECOVERY_MANIFEST_SHA256, "frozen-bank recovery manifest")
    _validate_producer_contract(producer_contract)
    producer_info = _validate_producer_report(producer_report, producer_contract, repo_root)
    recovery_info = _validate_recovery_manifest(recovery_manifest, producer_report, repo_root)
    metric_sources = _validate_metric_sources(repo_root)

    payload = _contract_payload(
        repo_root,
        git,
        attribution_v1,
        attribution_v2,
        attribution_report,
        producer_contract,
        producer_report,
        producer_info,
        recovery_info,
        metric_sources,
    )
    contract_path = _write_contract(repo_root, output_dir, payload)
    return {
        "classification": READY_CLASSIFICATION,
        "semantic_authority": TASK_ID,
        "contract": _relative(repo_root, contract_path),
        "contract_sha256": file_sha256(contract_path),
        "git_branch": git["branch"],
        "git_starting_head": EXPECTED_STARTING_HEAD,
        "git_ending_head": _git_state(repo_root)["head"],
        "worktree_dirty_before_artifact": git["dirty_paths"],
        "component_swaps_executed": False,
        "oracle_metrics_computed": False,
        "q_tensors_loaded_as_tensors": False,
        "sealed_confirmation_consumed": False,
        "producer_fit_or_sampling": False,
        "cuda_training_or_generation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
