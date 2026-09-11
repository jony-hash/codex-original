"""Define the N3 oracle-guided minimal-intervention contract.

This definition consumer binds the completed oracle diagnostic, the frozen
attribution result, and the current empirical-marginal Gaussian-copula
baseline.  It publishes exactly two development-only candidate definitions:
one marginal-only log1p-normal replacement and one dependence-only fixed-t5
copula replacement.  It never loads q tensors, fits either candidate, samples
either candidate, or reads a new candidate result.
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

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256


TASK_ID = "DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "n3_oracle_guided_minimal_intervention_contract.v1"
READY_CLASSIFICATION = "DEFINE_N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V1_READY"
CLAIM_CLASSIFICATION = "N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_DEFINED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"

ORACLE_CONTRACT = Path(
    "outputs/20260909_v2/runs/"
    "n3_copula_mixed_component_swap_diagnostic_contract_v1_attempt_01/"
    "n3_copula_mixed_component_swap_diagnostic_contract.v1.json"
)
ORACLE_CONTRACT_SHA256 = "d46c4d8e8d8dbc41f325f2e1d7f001d645479a60deef08e5d514a80bada22763"
ORACLE_REPORT = Path(
    "outputs/20260909_v2/runs/"
    "n3_copula_mixed_component_swap_diagnostic_v1_attempt_01/"
    "n3_copula_mixed_component_swap_diagnostic_report.v1.json"
)
ORACLE_REPORT_SHA256 = "fc139cb6ec9754babeb9e4ac6c03f08ef339cbdd6148362bb05b44c08daeaecc"

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

DEPENDENCE_SOURCE = Path("scripts/research/compare_n3_minimal_competing_producer_baselines_v1.py")
DEPENDENCE_SOURCE_SHA256 = "5e49a45e88eb4b5308fb24975aac6fda5e5031e38c46c939170c55ffddbc83f0"
PRIMARY_METRIC_SOURCE = Path("scripts/research/validate_n3_amplitude_producer_baseline.py")
PRIMARY_METRIC_SOURCE_SHA256 = "1ad916bbb8d109ca989f64b9f5590391cdffea9e0567e2ca8d241b032c6cde36"
BASELINE_EXECUTION_SOURCE = Path(
    "scripts/research/execute_n3_empirical_marginal_gaussian_copula_producer_v1.py"
)
BASELINE_EXECUTION_SOURCE_SHA256 = "5f1fa3fa6be35e96f345c3b7f7c6cce5ca0b45fe4619b1ead4455e86dcae538a"

FOLD_ASSIGNMENT_SHA256 = "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32"
CLASS_COUNT = 10
DIMENSION = 100
POOL_COUNT = 1000
FOLD_COUNT = 5
FIT_COUNT_PER_FOLD = 800
HELDOUT_COUNT_PER_FOLD = 200
GENERATED_BANK_SIZE = 200
DATA_VS_MODEL_REPLICATES = 4
MODEL_VS_MODEL_REPLICATES = 8
MODEL_REFERENCE_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
COMPATIBILITY_Q = 0.95
RNG_NAMESPACE = "N3_ORACLE_GUIDED_MINIMAL_INTERVENTION_CONTRACT_V1_CPU"
STUDENT_T_DEGREES_OF_FREEDOM = 5
BASELINE_DEPENDENCE_EFFECTIVE_SCALAR_COUNT = DIMENSION * (DIMENSION + 1) // 2

CANDIDATE_M = "M"
CANDIDATE_D = "D"
CANDIDATE_M_FAMILY = "LOG1P_NORMAL_MARGINAL_FROZEN_GAUSSIAN_COPULA_V1"
CANDIDATE_D_FAMILY = "FIXED_NU5_STUDENT_T_COPULA_FROZEN_EMPIRICAL_MARGINAL_V1"

SELECTION_LABELS = {
    (False, False): "NO_MINIMAL_INTERVENTION_SUPPORTED",
    (True, False): "SELECT_MARGINAL_ONLY",
    (False, True): "SELECT_DEPENDENCE_ONLY",
    (True, True): "SELECT_COMPLEXITY_PREFERRED_MINIMAL_INTERVENTION",
}

OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n3_oracle_guided_minimal_intervention_contract_v1_attempt_01"
)
CONTRACT_NAME = "n3_oracle_guided_minimal_intervention_contract.v1.json"

SAFE_DIRTY_PATHS = {
    "docs/CURRENT_STATUS.md",
    "docs/status/current_handoff.md",
    "docs/status/experiment_history.md",
    "docs/status/research_governance_state.json",
    "scripts/research/define_n3_copula_failure_attribution_contract_v1.py",
    "scripts/research/define_n3_copula_failure_attribution_contract_v2.py",
    "scripts/research/define_n3_copula_mixed_component_swap_diagnostic_contract_v1.py",
    "scripts/research/execute_n3_copula_failure_attribution_v1.py",
    "scripts/research/execute_n3_copula_mixed_component_swap_diagnostic_v1.py",
    "scripts/research/define_n3_oracle_guided_minimal_intervention_contract_v1.py",
    "tests/v2/test_execute_n3_copula_failure_attribution_v1.py",
    "tests/v2/test_execute_n3_copula_mixed_component_swap_diagnostic_v1.py",
    "tests/v2/test_n3_copula_failure_attribution_contract.py",
    "tests/v2/test_n3_copula_failure_attribution_contract_v2.py",
    "tests/v2/test_n3_copula_mixed_component_swap_diagnostic_contract_v1.py",
    "tests/v2/test_n3_oracle_guided_minimal_intervention_contract_v1.py",
}

JOINT_METRICS = (
    "energy_distance",
    "covariance_relative_frobenius",
    "norm_energy_distance",
    "squared_norm_energy_distance",
)
MARGINAL_METRICS = (
    "marginal_mean_l2",
    "marginal_std_l2",
    "marginal_energy_distance_mean",
    "marginal_energy_distance_max",
)
SCALAR_METRICS = JOINT_METRICS + MARGINAL_METRICS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
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


def _verify_file_hash(repo_root: Path, path_value: str, expected_sha256: str, name: str) -> None:
    path = Path(path_value)
    _require(not path.is_absolute(), f"{name} must be repository-relative")
    target = repo_root / path
    _relative(repo_root, target)
    _require(target.is_file(), f"{name} is missing: {target}")
    _require(file_sha256(target) == expected_sha256, f"{name} hash changed: {target}")


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


def selection_label(marginal_supported: bool, dependence_supported: bool) -> str:
    """Return the complete valid-comparison selection vocabulary."""

    _require(type(marginal_supported) is bool, "marginal support must be boolean")
    _require(type(dependence_supported) is bool, "dependence support must be boolean")
    return SELECTION_LABELS[(marginal_supported, dependence_supported)]


def complexity_preferred_candidate() -> str:
    """Return the pre-frozen candidate selected when both candidates pass."""

    return CANDIDATE_M


def _validate_oracle_inputs(repo_root: Path) -> dict[str, Any]:
    oracle_contract = _read_json_exact(
        repo_root, ORACLE_CONTRACT, ORACLE_CONTRACT_SHA256, "oracle diagnostic contract"
    )
    _require(
        oracle_contract.get("schema") == "n3_copula_mixed_component_swap_diagnostic_contract.v1",
        "oracle diagnostic contract schema changed",
    )
    _require(
        oracle_contract.get("classification")
        == "DEFINE_N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_CONTRACT_V1_READY",
        "oracle diagnostic contract is not ready",
    )
    _require(
        oracle_contract.get("claim_classification")
        == "N3_COPULA_MIXED_COMPONENT_SWAP_DIAGNOSTIC_CONTRACT_DEFINED",
        "oracle diagnostic contract claim changed",
    )
    scope = _mapping(oracle_contract.get("scope"), "oracle diagnostic scope")
    _require(scope.get("selection_scope") == "DEVELOPMENT_ONLY", "oracle selection scope changed")
    _require(scope.get("coordinate") == "R2_AMPLITUDE", "oracle coordinate changed")
    _require(scope.get("dimension") == DIMENSION, "oracle dimension changed")
    _require(scope.get("labels_as_model_inputs") is False, "oracle labels became model inputs")
    _require(scope.get("producer_family_selection") is False, "oracle diagnostic selected a family")
    oracle_totality = _mapping(oracle_contract.get("contract_totality"), "oracle contract totality")
    _require(oracle_totality.get("status") == "passed", "oracle contract totality changed")
    oracle_boundary = _mapping(oracle_contract.get("execution_boundary"), "oracle execution boundary")
    _require(oracle_boundary.get("producer_fit") is False, "oracle fit boundary changed")
    _require(oracle_boundary.get("producer_sampling") is False, "oracle sampling boundary changed")

    oracle_report = _read_json_exact(
        repo_root, ORACLE_REPORT, ORACLE_REPORT_SHA256, "oracle diagnostic report"
    )
    _require(
        oracle_report.get("schema") == "n3_copula_mixed_component_swap_diagnostic_report.v1",
        "oracle diagnostic report schema changed",
    )
    _require(oracle_report.get("status") == "valid_completed", "oracle diagnostic is incomplete")
    _require(
        oracle_report.get("classification") == "N3_COPULA_PRIMARY_BOTH_ORACLES_SUFFICIENT",
        "oracle diagnostic result changed",
    )
    report_contract = _mapping(oracle_report.get("contract"), "oracle report contract")
    _require(report_contract.get("sha256") == ORACLE_CONTRACT_SHA256, "oracle report contract identity changed")
    _require(report_contract.get("classification") == oracle_contract.get("classification"), "oracle report contract classification changed")
    aggregate = _mapping(oracle_report.get("aggregate"), "oracle aggregate")
    _require(aggregate.get("pair_label_output_count") == 20, "oracle pair count changed")
    _require(aggregate.get("original_failure_pair_count") == 9, "oracle original-failure count changed")
    _require(aggregate.get("original_pass_control_pair_count") == 11, "oracle control count changed")
    _require(
        aggregate.get("fold_pattern")
        == {"0": "BOTH_ORACLES", "1": "BOTH_ORACLES", "2": "BOTH_ORACLES", "3": "BOTH_ORACLES", "4": "CONTROL_FOLD"},
        "oracle fold pattern changed",
    )
    oracle_governance = _mapping(oracle_report.get("governance"), "oracle governance")
    for key, expected in {
        "component_swap_diagnostic_executed": True,
        "oracle_metrics_computed": True,
        "producer_fit": False,
        "producer_sampling": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_status": "RESERVED_NOT_MATERIALIZED",
    }.items():
        _require(oracle_governance.get(key) == expected, f"oracle governance changed: {key}")
    outer_gate = _mapping(oracle_report.get("outer_gate"), "oracle outer gate")
    _require(outer_gate.get("p1_execution_authorized") is False, "oracle authorized P1 execution")

    return {
        "contract": {
            "path": _relative(repo_root, repo_root / ORACLE_CONTRACT),
            "sha256": ORACLE_CONTRACT_SHA256,
            "schema": oracle_contract["schema"],
            "classification": oracle_contract["classification"],
            "claim_classification": oracle_contract["claim_classification"],
        },
        "report": {
            "path": _relative(repo_root, repo_root / ORACLE_REPORT),
            "sha256": ORACLE_REPORT_SHA256,
            "schema": oracle_report["schema"],
            "classification": oracle_report["classification"],
            "fold_pattern": aggregate["fold_pattern"],
            "original_failure_pair_count": aggregate["original_failure_pair_count"],
            "original_pass_control_pair_count": aggregate["original_pass_control_pair_count"],
            "pair_label_output_count": aggregate["pair_label_output_count"],
        },
    }


def _validate_attribution_inputs(repo_root: Path) -> dict[str, Any]:
    attribution_contract = _read_json_exact(
        repo_root,
        ATTRIBUTION_CONTRACT_V2,
        ATTRIBUTION_CONTRACT_V2_SHA256,
        "attribution V2 contract",
    )
    _require(attribution_contract.get("schema") == "n3_copula_failure_attribution_contract.v2", "attribution V2 schema changed")
    _require(attribution_contract.get("classification") == "DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V2_READY", "attribution V2 is not ready")
    _require(attribution_contract.get("v1_primary_semantics_unchanged") is True, "attribution V2 changed primary semantics")
    supersession = _mapping(attribution_contract.get("supersession"), "attribution V2 supersession")
    _require(supersession.get("primary_scientific_semantics_changed") is False, "attribution V2 primary semantics changed")

    attribution_report = _read_json_exact(
        repo_root,
        ATTRIBUTION_REPORT,
        ATTRIBUTION_REPORT_SHA256,
        "attribution report",
    )
    _require(attribution_report.get("schema") == "n3_copula_failure_attribution_report.v1", "attribution report schema changed")
    _require(attribution_report.get("status") == "valid_completed", "attribution report is incomplete")
    _require(attribution_report.get("classification") == "N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED", "attribution result changed")
    report_contract = _mapping(attribution_report.get("contract"), "attribution report contract")
    _require(report_contract.get("sha256") == ATTRIBUTION_CONTRACT_V2_SHA256, "attribution report contract identity changed")
    lineage = _mapping(attribution_report.get("development_lineage"), "attribution lineage")
    fold_identity = _mapping(lineage.get("fold_identity"), "attribution fold identity")
    _require(fold_identity.get("assignment_sha256") == FOLD_ASSIGNMENT_SHA256, "attribution fold assignment changed")
    _require(fold_identity.get("fold_count") == FOLD_COUNT, "attribution fold count changed")
    governance = _mapping(attribution_report.get("governance"), "attribution governance")
    for key, expected in {
        "attribution_executed": True,
        "metrics_computed": True,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items():
        _require(governance.get(key) == expected, f"attribution governance changed: {key}")
    outer_gate = _mapping(attribution_report.get("outer_gate"), "attribution outer gate")
    _require(outer_gate.get("p1_execution_authorized") is False, "attribution authorized P1 execution")

    return {
        "contract": {
            "path": _relative(repo_root, repo_root / ATTRIBUTION_CONTRACT_V2),
            "sha256": ATTRIBUTION_CONTRACT_V2_SHA256,
            "schema": attribution_contract["schema"],
            "classification": attribution_contract["classification"],
            "primary_scientific_semantics_changed": False,
        },
        "report": {
            "path": _relative(repo_root, repo_root / ATTRIBUTION_REPORT),
            "sha256": ATTRIBUTION_REPORT_SHA256,
            "schema": attribution_report["schema"],
            "classification": attribution_report["classification"],
            "fold_pattern": _mapping(attribution_report["aggregate"], "attribution aggregate")["fold_pattern"],
        },
    }


def _validate_baseline_inputs(repo_root: Path) -> dict[str, Any]:
    contract = _read_json_exact(repo_root, PRODUCER_CONTRACT, PRODUCER_CONTRACT_SHA256, "original producer contract")
    _require(contract.get("schema") == "n3_empirical_marginal_gaussian_copula_producer_contract.v1", "producer contract schema changed")
    _require(contract.get("classification") == "DEFINE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1_READY", "producer contract is not ready")
    _require(contract.get("claim_classification") == "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_MODEL_ASSUMPTION_DEFINED", "producer contract claim changed")

    family = _mapping(contract.get("family"), "producer family")
    for key, expected in {
        "name": "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1",
        "coordinate": "R2_AMPLITUDE",
        "dimension": DIMENSION,
        "marginal_model": "fit-only empirical plotting-position marginal Q_j",
        "dependence_model": "one regularized Gaussianized-score dependence matrix R_shrunk",
        "labels_as_inputs": False,
        "mixture_K": False,
        "local_components": False,
        "neural_network": False,
        "normalizing_flow": False,
    }.items():
        _require(family.get(key) == expected, f"producer family changed: {key}")

    marginal = _mapping(contract.get("marginal_cdf_quantile_contract"), "producer marginal contract")
    for key, expected in {
        "fit_only": True,
        "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
        "plotting_positions": "p_i=(i-0.5)/n for i=1,...,n",
        "quantile_interpolation": "piecewise-linear interpolation between adjacent (p_i,q_(i)) pairs",
        "below_first_position": "return q_(1)",
        "above_last_position": "return q_(n)",
        "outside_support_extrapolation": False,
        "heldout_parameter_leakage": False,
    }.items():
        _require(marginal.get(key) == expected, f"producer marginal contract changed: {key}")

    dependence = _mapping(contract.get("dependence_contract"), "producer dependence contract")
    for key, expected in {
        "standard_deviation": "s_j=sqrt(sum_i z_centered_ij^2/(n-1))",
        "standard_deviation_denominator": "n-1",
        "empirical_dependence": "R_emp=(Y^T Y)/(n-1)",
        "dependence_denominator": "n-1",
        "tau": "trace(R_emp)/d",
        "alpha": "alpha=0 if delta=0, otherwise clip(beta/delta,0,1)",
        "regularized_matrix": "R_shrunk=(1-alpha)R_emp+alpha*tau I",
        "symmetrization": "replace R_shrunk by (R_shrunk+R_shrunk^T)/2 exactly",
        "final_matrix_used_for_sampling": "R_shrunk directly",
        "unit_diagonal_renormalization": False,
        "heldout_parameter_leakage": False,
    }.items():
        _require(dependence.get(key) == expected, f"producer dependence contract changed: {key}")

    evaluation = _mapping(contract.get("development_evaluation_contract"), "producer evaluation contract")
    for key, expected in {
        "fit_rows": FIT_COUNT_PER_FOLD,
        "heldout_rows": HELDOUT_COUNT_PER_FOLD,
        "labels_as_model_inputs": False,
        "optical_forward": False,
        "evaluator": False,
        "primary_metric": "validated_biased_empirical_energy_distance_euclidean_q_space",
        "primary_metric_implementation": "scripts.research.validate_n3_amplitude_producer_baseline._energy_distance",
        "sealed_confirmation": False,
    }.items():
        _require(evaluation.get(key) == expected, f"producer evaluation contract changed: {key}")
    compatibility = _mapping(contract.get("compatibility_contract"), "producer compatibility contract")
    for key, expected in {
        "q": COMPATIBILITY_Q,
        "own_model_reference_required": True,
        "all_folds_required": True,
        "absolute_metric_thresholds": False,
        "less_bad_than_failed_prior_family_is_sufficient": False,
    }.items():
        _require(compatibility.get(key) == expected, f"producer compatibility contract changed: {key}")
    _require(
        compatibility.get("rule")
        == "fixed-replicate mean DATA_VS_MODEL <= q95 MODEL_VS_MODEL for q-space primary plus covariance/norm and aggregate marginal guardrails",
        "producer compatibility rule changed",
    )
    fold_identity = _mapping(_mapping(contract.get("development_identity"), "producer development identity").get("fold_assignment"), "producer fold identity")
    _require(fold_identity.get("assignment_sha256") == FOLD_ASSIGNMENT_SHA256, "producer fold assignment changed")
    _require(fold_identity.get("fold_count") == FOLD_COUNT, "producer fold count changed")

    sampling = _mapping(contract.get("sampling_contract"), "producer sampling contract")
    for key, expected in {
        "rng_device": "cpu",
        "dtype": "float64",
        "rng_generator": "torch.Generator(device='cpu')",
        "generated_bank_size": GENERATED_BANK_SIZE,
        "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
        "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
        "proposal_rejection": False,
        "finite_and_nonnegative_required": True,
        "clamp": False,
        "abs": False,
        "relu": False,
        "softplus": False,
        "epsilon_repair": False,
        "samplewise_normalization": False,
    }.items():
        _require(sampling.get(key) == expected, f"producer sampling contract changed: {key}")
    sealed = _mapping(contract.get("sealed_confirmation_firewall"), "producer sealed firewall")
    _require(sealed.get("status") == "RESERVED_NOT_MATERIALIZED", "sealed confirmation status changed")
    _require(sealed.get("sample_values_loaded") is False and sealed.get("confirmation_evaluation_authorized") is False, "sealed confirmation boundary changed")
    governance = _mapping(contract.get("governance"), "producer governance")
    for key, expected in {
        "real_n3_copula_fit_executed": False,
        "candidate_selected": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items():
        _require(governance.get(key) == expected, f"producer governance changed: {key}")

    report = _read_json_exact(repo_root, PRODUCER_REPORT, PRODUCER_REPORT_SHA256, "original producer report")
    _require(report.get("schema") == "n3_empirical_marginal_gaussian_copula_producer_report.v1", "producer report schema changed")
    _require(report.get("status") == "valid_completed", "producer report is incomplete")
    _require(report.get("classification") == "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED", "producer result changed")
    report_contract = _mapping(report.get("contract"), "producer report contract")
    _require(report_contract.get("sha256") == PRODUCER_CONTRACT_SHA256, "producer report contract identity changed")
    report_fold_identity = _mapping(report.get("development_fold_identity"), "producer report fold identity")
    _require(report_fold_identity.get("assignment_sha256") == FOLD_ASSIGNMENT_SHA256, "producer report fold assignment changed")
    report_pool = _mapping(report.get("development_pool"), "producer report development pool")
    _require(report_pool.get("count") == POOL_COUNT and report_pool.get("dimension") == DIMENSION, "producer development pool changed")
    _require(report_pool.get("q_coordinate") == "R2_AMPLITUDE", "producer q coordinate changed")
    report_governance = _mapping(report.get("governance"), "producer report governance")
    for key, expected in {
        "training_authorized": False,
        "formal_promotion": False,
    }.items():
        _require(report_governance.get(key) == expected, f"producer report governance changed: {key}")

    authorities = _mapping(contract.get("authorities"), "producer authorities")
    producer_data = _mapping(authorities.get("producer_data_authority"), "producer data authority")
    development_banks = _mapping(authorities.get("development_bank_artifacts"), "development bank artifacts")
    for role, item in (
        ("source manifest", _mapping(producer_data.get("source_manifest"), "source manifest authority")),
        ("partition manifest", _mapping(producer_data.get("partition_manifest"), "partition manifest authority")),
        ("firewall manifest", _mapping(producer_data.get("firewall_manifest"), "firewall manifest authority")),
        ("fit bank", _mapping(producer_data.get("fit_bank"), "fit bank authority")),
        ("internal validation bank", _mapping(producer_data.get("internal_validation_bank"), "internal validation bank authority")),
    ):
        _verify_file_hash(repo_root, str(item["path"]), str(item["sha256"]), role)
    _require(
        development_banks.get("fit_bank_path") == producer_data["fit_bank"]["path"],
        "development fit-bank authority path changed",
    )
    _require(
        development_banks.get("fit_bank_sha256") == producer_data["fit_bank"]["sha256"],
        "development fit-bank authority hash changed",
    )
    _require(
        development_banks.get("internal_validation_bank_path") == producer_data["internal_validation_bank"]["path"],
        "development validation-bank authority path changed",
    )
    _require(
        development_banks.get("internal_validation_bank_sha256") == producer_data["internal_validation_bank"]["sha256"],
        "development validation-bank authority hash changed",
    )
    _require(development_banks.get("q_values_loaded_by_this_definition_consumer") is False, "baseline definition loaded q values")

    for path, expected_sha256, name in (
        (DEPENDENCE_SOURCE, DEPENDENCE_SOURCE_SHA256, "analytic-shrinkage source"),
        (PRIMARY_METRIC_SOURCE, PRIMARY_METRIC_SOURCE_SHA256, "raw q-space metric source"),
        (BASELINE_EXECUTION_SOURCE, BASELINE_EXECUTION_SOURCE_SHA256, "baseline execution source"),
    ):
        _verify_file_hash(repo_root, path.as_posix(), expected_sha256, name)

    sealed_path = str(sealed.get("manifest_path"))
    sealed_sha256 = str(sealed.get("manifest_sha256"))
    return {
        "contract": {
            "path": _relative(repo_root, repo_root / PRODUCER_CONTRACT),
            "sha256": PRODUCER_CONTRACT_SHA256,
            "schema": contract["schema"],
            "classification": contract["classification"],
            "claim_classification": contract["claim_classification"],
        },
        "report": {
            "path": _relative(repo_root, repo_root / PRODUCER_REPORT),
            "sha256": PRODUCER_REPORT_SHA256,
            "schema": report["schema"],
            "classification": report["classification"],
        },
        "data_authority": {
            "coordinate": producer_data["coordinate"],
            "coordinate_definition": producer_data["coordinate_definition"],
            "producer_lineage": producer_data["producer_lineage"],
            "source_manifest": dict(producer_data["source_manifest"]),
            "partition_manifest": dict(producer_data["partition_manifest"]),
            "firewall_manifest": dict(producer_data["firewall_manifest"]),
            "fit_bank": dict(producer_data["fit_bank"]),
            "internal_validation_bank": dict(producer_data["internal_validation_bank"]),
            "development_bank_artifacts": {
                "fit_bank_path": development_banks["fit_bank_path"],
                "fit_bank_sha256": development_banks["fit_bank_sha256"],
                "internal_validation_bank_path": development_banks["internal_validation_bank_path"],
                "internal_validation_bank_sha256": development_banks["internal_validation_bank_sha256"],
            },
            "q_values_loaded_by_this_definition_consumer": False,
        },
        "sealed_exclusion": {
            "semantic_authority_id": "N3_SEALED_CONFIRMATION_SOURCE_V1",
            "path": sealed_path,
            "sha256": sealed_sha256,
            "input_authority": False,
            "path_read_by_this_consumer": False,
            "sample_values_loaded": False,
            "status": sealed["status"],
            "consumed": False,
        },
    }


def _candidate_m() -> dict[str, Any]:
    return {
        "id": CANDIDATE_M,
        "family": CANDIDATE_M_FAMILY,
        "candidate_count_in_family": 1,
        "changed_assumption": "For each coordinate j, Y_j=log1p(q_j) is Gaussian with fitted location and scale; the joint dependence remains the frozen Gaussian copula.",
        "changed_component": "marginal representation only",
        "unchanged_component": {
            "dependence": "the current Gaussianized-score fit and analytic-shrinkage R_shrunk are byte-for-byte semantically frozen",
            "dependence_source": DEPENDENCE_SOURCE.as_posix(),
            "dependence_source_sha256": DEPENDENCE_SOURCE_SHA256,
            "dependence_fit": "the baseline mid-rank plotting positions, z=Phi^-1(u), n-1 standardization, R_emp, beta/delta alpha, tau I target, symmetrization, positive-definite check, and direct use of R_shrunk",
            "dependence_parameter_leakage": False,
        },
        "parameterization": {
            "coordinate_support": "q_j>=0",
            "forward_transform": "y_j=log1p(q_j)",
            "inverse_transform": "q_j=expm1(y_j)",
            "distribution": "Y_j~Normal(mu_j,sigma_j^2)",
            "parameters_per_coordinate": ["mu_j", "sigma_j"],
            "parameter_count_per_coordinate": 2,
            "new_fitted_scalar_count": 2 * DIMENSION,
            "effective_fitted_scalar_count_for_complexity": 2 * DIMENSION,
            "zero_value_semantics": "q_j=0 maps to y_j=0 exactly; no positive offset is fitted",
            "quantile": "Q_M,j(u)=expm1(mu_j+sigma_j*Phi^-1(u)) for 0<u<1",
            "cdf": "F_M,j(q)=Phi((log1p(q)-mu_j)/sigma_j) for q>=0; F_M,j(q)=0 for q<0",
            "tail": "positive tail is analytic and unbounded before finite-precision validation; no empirical endpoint is substituted",
        },
        "fitting": {
            "rows": FIT_COUNT_PER_FOLD,
            "rows_source": "frozen fit rows only; held-out rows are never read for parameter estimation",
            "labels_as_model_inputs": False,
            "mu": "mu_j=mean_i log1p(q_fit[i,j])",
            "sigma": "sigma_j=sqrt(sum_i(log1p(q_fit[i,j])-mu_j)^2/(n-1))",
            "denominator": "n-1",
            "one_fit_per_fold": True,
            "optimizer": False,
            "mixture": False,
            "tail_model_selection": False,
            "marginal_parameter_shrinkage": False,
        },
        "sampling": {
            "law": "z_new~Normal(0,R_shrunk); u_new,j=Phi(z_new,j); q_new,j=Q_M,j(u_new,j)",
            "rng_device": "cpu",
            "dtype": "float64",
            "rng_generator": "torch.Generator(device='cpu')",
            "rng_namespace": RNG_NAMESPACE,
            "seed_derivation": "sha256(namespace|candidate|role|fold|replicate) first 64 bits masked to signed 63-bit",
            "generated_bank_size": GENERATED_BANK_SIZE,
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "reference_pair_order": [
                {"pair_index": index, "left_replicate": left, "right_replicate": right}
                for index, (left, right) in enumerate(MODEL_REFERENCE_PAIRS)
            ],
            "proposal_rejection": False,
            "finite_and_nonnegative_required": True,
            "clamp": False,
            "abs": False,
            "relu": False,
            "softplus": False,
            "epsilon_repair": False,
            "samplewise_normalization": False,
        },
        "numerical_failure_semantics": {
            "required_input": "all fitting q values finite and nonnegative",
            "required_transform": "log1p(q_fit), mu_j, sigma_j and inverse-CDF argument finite",
            "strict_scale": "sigma_j must be finite and strictly positive for every coordinate",
            "strict_probability": "u_new must be finite and strictly between zero and one",
            "strict_output": "expm1 output must be finite and nonnegative",
            "invalid_status": "INVALID_HARNESS",
            "failure_action": "stop the candidate/fold before scientific compatibility or selection; emit no scientific classification",
            "fallbacks": False,
            "clamp_or_repair": False,
        },
    }


def _candidate_d() -> dict[str, Any]:
    return {
        "id": CANDIDATE_D,
        "family": CANDIDATE_D_FAMILY,
        "candidate_count_in_family": 1,
        "changed_assumption": "The frozen global Gaussian copula is replaced by one global fixed-nu Student-t copula with nu=5 and the same fitted correlation parameter R_shrunk.",
        "changed_component": "dependence representation only",
        "unchanged_component": {
            "marginal": "the current fit-only empirical plotting-position marginal Q_j is unchanged",
            "marginal_source": "current n3_empirical_marginal_gaussian_copula_producer_contract.v1",
            "marginal_fit": "stable value-then-original-row tie order, average-rank plotting positions, piecewise-linear interpolation, q_(1)/q_(n) endpoint return, no extrapolation",
            "marginal_parameter_leakage": False,
        },
        "parameterization": {
            "copula": "Student-t copula with correlation matrix R_shrunk",
            "degrees_of_freedom": STUDENT_T_DEGREES_OF_FREEDOM,
            "degrees_of_freedom_status": "pre-frozen fixed hyperparameter; not fitted, scanned, or selected after results",
            "additional_fitted_scalar_count": 0,
            "radial_relaxation": "one shared global radial scale; no component-specific scales or mixture weights",
            "latent_construction": "G~Normal(0,R_shrunk), W=(e_1^2+...+e_5^2)/5 with independent e_k~Normal(0,1), T=G/sqrt(W)",
            "probability_integral_transform": "u_j=F_t5(T_j)",
            "student_t_cdf_formula": "for t<0, F_t5(t)=0.5*I_{5/(5+t^2)}(5/2,1/2); for t>=0, F_t5(t)=1-0.5*I_{5/(5+t^2)}(5/2,1/2)",
            "regularized_incomplete_beta": "I_x(a,b) is the regularized incomplete beta function evaluated in deterministic CPU float64",
            "quantile": "q_new,j=Q_emp,j(u_j) using the unchanged empirical marginal quantile",
        },
        "minimality_justification": {
            "why_this_relaxes_dependence": "The common radial scale produces non-Gaussian joint tail dependence while retaining the same rank-correlation parameter R_shrunk and unchanged coordinate marginals.",
            "why_nu5": "nu=5 is the smallest integer Student-t degrees of freedom with a finite fourth moment, preserving the covariance/norm guardrail domain without fitting an additional tail parameter.",
            "why_single_family": "one fixed global radial relaxation is the smallest explicit non-Gaussian copula extension; no mixture grid, local component, label path, flow, or neural density is introduced",
        },
        "fitting": {
            "rows": FIT_COUNT_PER_FOLD,
            "rows_source": "frozen fit rows only; held-out rows are never read for parameter estimation",
            "labels_as_model_inputs": False,
            "dependence_fit": "reuse the baseline empirical plotting positions, Gaussianized scores, analytic shrinkage and R_shrunk exactly",
            "additional_dependence_fit": False,
            "optimizer": False,
            "model_order_selection": False,
            "one_fit_per_fold": True,
        },
        "sampling": {
            "law": "G0~Normal(0,R_shrunk); E~Normal(0,I_5) independently; W=rowsum(E^2)/5; T=G0/sqrt(W); U=F_t5(T); q_new,j=Q_emp,j(U_j)",
            "draw_order": "draw G0 standard-normal noise first, then E radial noise, using the same CPU Generator",
            "rng_device": "cpu",
            "dtype": "float64",
            "rng_generator": "torch.Generator(device='cpu')",
            "rng_namespace": RNG_NAMESPACE,
            "seed_derivation": "sha256(namespace|candidate|role|fold|replicate) first 64 bits masked to signed 63-bit",
            "generated_bank_size": GENERATED_BANK_SIZE,
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "reference_pair_order": [
                {"pair_index": index, "left_replicate": left, "right_replicate": right}
                for index, (left, right) in enumerate(MODEL_REFERENCE_PAIRS)
            ],
            "proposal_rejection": False,
            "finite_and_nonnegative_required": True,
            "clamp": False,
            "abs": False,
            "relu": False,
            "softplus": False,
            "epsilon_repair": False,
            "samplewise_normalization": False,
        },
        "numerical_failure_semantics": {
            "required_input": "baseline fit q values and baseline R_shrunk satisfy the frozen finite, positive-definite, unit-diagonal checks",
            "required_radial": "W is finite and strictly positive for every generated row",
            "required_latent": "T is finite",
            "required_probability": "F_t5(T) is finite and strictly between zero and one; no clamp is permitted",
            "required_output": "unchanged empirical quantile output is finite and nonnegative",
            "invalid_status": "INVALID_HARNESS",
            "failure_action": "stop the candidate/fold before scientific compatibility or selection; emit no scientific classification",
            "fallbacks": False,
            "clamp_or_repair": False,
        },
    }


def _build_payload(
    repo_root: Path,
    git: Mapping[str, Any],
    oracle: Mapping[str, Any],
    attribution: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_m = _candidate_m()
    candidate_d = _candidate_d()
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
            "oracle_diagnostic_contract": dict(oracle["contract"]),
            "oracle_diagnostic_report": dict(oracle["report"]),
            "attribution_contract_v2": dict(attribution["contract"]),
            "attribution_report": dict(attribution["report"]),
            "original_producer_contract": dict(baseline["contract"]),
            "original_producer_report": dict(baseline["report"]),
            "development_data": dict(baseline["data_authority"]),
            "fold_assignment": {
                "source": "original producer development_fold_identity and all frozen N3 attribution/oracle consumers",
                "algorithm": "within_class_sha256_sort_round_robin_v1",
                "dataset": "mnist_train",
                "fold_count": FOLD_COUNT,
                "assignment_sha256": FOLD_ASSIGNMENT_SHA256,
                "runtime_resplit": False,
                "runtime_reselection": False,
                "labels_used_only_for_balance_and_provenance": True,
                "labels_as_model_inputs": False,
            },
            "implementation_sources": {
                "analytic_shrinkage": {
                    "path": _relative(repo_root, repo_root / DEPENDENCE_SOURCE),
                    "sha256": DEPENDENCE_SOURCE_SHA256,
                },
                "raw_q_space_primary_metric": {
                    "path": _relative(repo_root, repo_root / PRIMARY_METRIC_SOURCE),
                    "sha256": PRIMARY_METRIC_SOURCE_SHA256,
                    "implementation": "scripts.research.validate_n3_amplitude_producer_baseline._energy_distance",
                },
                "baseline_sampling_semantics": {
                    "path": _relative(repo_root, repo_root / BASELINE_EXECUTION_SOURCE),
                    "sha256": BASELINE_EXECUTION_SOURCE_SHA256,
                },
            },
            "sealed_confirmation_exclusion": dict(baseline["sealed_exclusion"]),
        },
        "scope": {
            "question": "Can one marginal-only and one dependence-only minimal intervention be fully specified such that a later one-shot five-fold development comparison can determine whether either achieves frozen compatibility without introducing a joint high-capacity family?",
            "selection_scope": "DEVELOPMENT_ONLY",
            "coordinate": "R2_AMPLITUDE",
            "dimension": DIMENSION,
            "candidate_ids": [CANDIDATE_M, CANDIDATE_D],
            "candidate_family_count": 2,
            "exactly_one_family_per_candidate": True,
            "labels_as_model_inputs": False,
            "optical_forward": False,
            "evaluator": False,
            "image_or_generator_metric": False,
            "producer_family_selection_before_evaluation": False,
            "new_candidate_results_observed_by_definition_consumer": False,
        },
        "oracle_basis": {
            "result": "N3_COPULA_PRIMARY_BOTH_ORACLES_SUFFICIENT",
            "interpretation": "idealized component-intervention sufficiency only; it does not identify a learnable producer family",
            "fold_pattern": {"0": "BOTH_ORACLES", "1": "BOTH_ORACLES", "2": "BOTH_ORACLES", "3": "BOTH_ORACLES", "4": "CONTROL_FOLD"},
            "original_failure_pairs": 9,
            "controls": 11,
            "selection_authority": False,
            "candidate_fit_authority": False,
        },
        "baseline_frozen": {
            "family": "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1",
            "coordinate": "R2_AMPLITUDE",
            "marginal": {
                "model": "fit-only empirical plotting-position marginal Q_j",
                "fit_rows": FIT_COUNT_PER_FOLD,
                "ties": "stable value-then-original-row-index order; tied block average rank",
                "plotting_position": "u_ij=(average_rank_ij-0.5)/n",
                "quantile": "piecewise-linear interpolation at p_i=(i-0.5)/n; below/above support returns q_(1)/q_(n)",
                "outside_support_extrapolation": False,
                "heldout_parameter_leakage": False,
            },
            "dependence": {
                "gaussianization": "z_ij=Phi^-1(u_ij)",
                "center": "z_centered=z-mean_fit(z)",
                "standard_deviation": "s_j=sqrt(sum_i z_centered_ij^2/(n-1))",
                "standardized_rows": "y_ij=z_centered_ij/s_j",
                "empirical_dependence": "R_emp=(Y^T Y)/(n-1)",
                "shrinkage": "alpha=0 if delta=0, otherwise clip(beta/delta,0,1); R_shrunk=(1-alpha)R_emp+alpha*tau I",
                "symmetrization": "replace R_shrunk by (R_shrunk+R_shrunk^T)/2 exactly",
                "unit_diagonal_renormalization": False,
                "positive_definite_requirement": "finite Cholesky and finite strictly positive eigenvalues",
            },
        },
        "candidate_M": candidate_m,
        "candidate_D": candidate_d,
        "component_isolation": {
            "candidate_M_changed_component": "marginal",
            "candidate_M_frozen_component": "dependence",
            "candidate_D_changed_component": "dependence",
            "candidate_D_frozen_component": "marginal",
            "joint_marginal_dependence_family": False,
            "labels_conditioning": False,
            "component_scope_is_machine_defined": True,
            "cross_component_post_result_retuning": False,
        },
        "complexity": {
            "counting_convention": "count every fitted scalar output retained by one fold; count one scalar per empirical order-statistic slot, including duplicate values; count one per unique symmetric R_shrunk entry; fixed hyperparameters count zero",
            "shared_baseline_dependence_matrix_unique_entries": BASELINE_DEPENDENCE_EFFECTIVE_SCALAR_COUNT,
            "candidate_M": {
                "new_marginal_scalars": 2 * DIMENSION,
                "empirical_order_statistic_slots": 0,
                "fixed_extra_scalars": 0,
                "total_effective_fitted_scalar_count": BASELINE_DEPENDENCE_EFFECTIVE_SCALAR_COUNT + 2 * DIMENSION,
                "separately_fitted_component_count": DIMENSION + 1,
                "optimizer_required": False,
                "discrete_model_order_choices": 0,
                "numerical_failure_surface": "log1p/mu-sigma/expm1 finite-domain checks; no repair",
            },
            "candidate_D": {
                "new_marginal_scalars": 0,
                "empirical_order_statistic_slots": FIT_COUNT_PER_FOLD * DIMENSION,
                "fixed_extra_scalars": 0,
                "total_effective_fitted_scalar_count": BASELINE_DEPENDENCE_EFFECTIVE_SCALAR_COUNT + FIT_COUNT_PER_FOLD * DIMENSION,
                "separately_fitted_component_count": DIMENSION + 1,
                "optimizer_required": False,
                "discrete_model_order_choices": 0,
                "numerical_failure_surface": "radial W, t-CDF, strict probability and empirical-quantile finite-domain checks; no repair",
            },
            "ordering_tuple": [
                "total_effective_fitted_scalar_count ascending",
                "separately_fitted_component_count ascending",
                "optimizer_required false before true",
                "discrete_model_order_choices ascending",
                "candidate_id in fixed order M before D",
            ],
            "candidate_ordering_tuples": {
                "M": [BASELINE_DEPENDENCE_EFFECTIVE_SCALAR_COUNT + 2 * DIMENSION, DIMENSION + 1, 0, 0, "M"],
                "D": [BASELINE_DEPENDENCE_EFFECTIVE_SCALAR_COUNT + FIT_COUNT_PER_FOLD * DIMENSION, DIMENSION + 1, 0, 0, "D"],
            },
            "preferred_candidate_if_both_supported": CANDIDATE_M,
            "tie_break": "lexicographically smallest frozen ordering tuple; any exact residual tie resolves M before D",
            "post_result_subjective_score": False,
            "numerical_failure_surface_used_for_ordering": False,
        },
        "development_evaluation": {
            "fold_count": FOLD_COUNT,
            "fold_assignment_sha256": FOLD_ASSIGNMENT_SHA256,
            "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
            "class_count": CLASS_COUNT,
            "fit_rows_per_class": 80,
            "heldout_rows_per_class": 20,
            "resplit": False,
            "labels_as_model_inputs": False,
            "labels_used_only_for_balance_and_provenance": True,
            "sealed_confirmation": False,
            "optical_forward": False,
            "evaluator": False,
            "device": "cpu",
            "dtype": "float64",
            "generated_bank_size": GENERATED_BANK_SIZE,
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "model_reference_pairs": [
                {"pair_index": index, "left_replicate": left, "right_replicate": right}
                for index, (left, right) in enumerate(MODEL_REFERENCE_PAIRS)
            ],
            "rng_namespace": RNG_NAMESPACE,
            "seed_derivation": "sha256(namespace|candidate|role|fold|replicate) first 64 bits masked to signed 63-bit",
            "primary_metric": {
                "name": "validated_biased_empirical_energy_distance_euclidean_q_space",
                "representation": "raw q-space",
                "distance": "d(a,b)=sqrt(sum_j(a_j-b_j)^2)",
                "formula": "ED(X,Y)=2*mean_{i,k}d(X_i,Y_k)-mean_{i,i'}d(X_i,X_i')-mean_{k,k'}d(Y_k,Y_k')",
                "implementation": "scripts.research.validate_n3_amplitude_producer_baseline._energy_distance",
                "negative_policy": "require raw >= -1e-10; otherwise INVALID_HARNESS; return max(0,raw)",
                "within_pairs_include_diagonal": True,
                "candidate_oracle_values_in_reference": False,
            },
            "metrics": {
                "joint": list(JOINT_METRICS),
                "marginal": list(MARGINAL_METRICS),
                "per_coordinate_marginal_energy_distance": "computed with the same validated one-dimensional raw q-space ED; vector is descriptive-only and not a separate gate",
            },
            "model_vs_model_reference": {
                "role": "finite-sample fluctuation reference only",
                "candidate_own_reference_required": True,
                "q": COMPATIBILITY_Q,
                "quantile_method": "torch.quantile over four scalar reference-pair metrics, default linear interpolation",
                "quantile_definition": "sort v[0..3], h=(4-1)*0.95=2.85, q95=0.15*v[2]+0.85*v[3]",
                "fold_local_only": True,
                "cross_fold_pooling": False,
                "retuning": False,
            },
            "fold_compatibility": {
                "rule": "for every scalar metric, fixed-replicate mean DATA_VS_MODEL <= q95 MODEL_VS_MODEL from the same candidate and fold",
                "scalar_metric_count": len(SCALAR_METRICS),
                "all_joint_metrics_required": True,
                "all_aggregate_marginal_guardrails_required": True,
                "per_coordinate_vector_separate_gate": False,
                "equality_is_pass": True,
                "absolute_thresholds": False,
                "less_bad_than_old_copula_is_sufficient": False,
                "fold_supported": "operationally_valid and all scalar metric comparisons pass",
            },
            "candidate_support": {
                "all_five_folds_required": True,
                "support_condition": "all five folds are operationally valid and development-compatible under the frozen fold rule",
                "relative_improvement_sufficient": False,
                "four_of_five_sufficient": False,
                "three_of_five_sufficient": False,
                "better_aggregate_score_sufficient": False,
                "better_than_old_copula_sufficient": False,
                "invalid_harness_is_scientific_failure": False,
            },
        },
        "selection": {
            "valid_execution_precondition": "both candidates complete all five folds without INVALID_HARNESS",
            "mapping_function": "selection_label(marginal_supported, dependence_supported)",
            "mapping": {
                "neither": "NO_MINIMAL_INTERVENTION_SUPPORTED",
                "only_M": "SELECT_MARGINAL_ONLY",
                "only_D": "SELECT_DEPENDENCE_ONLY",
                "both": "select lexicographically preferred candidate from frozen complexity ordering",
            },
            "both_supported_result": "SELECT_COMPLEXITY_PREFERRED_MINIMAL_INTERVENTION with selected_candidate=M",
            "no_weighted_score": True,
            "at_most_one_candidate_selected": True,
            "invalid_harness_action": "stop with INVALID_HARNESS and emit no support or selection classification",
        },
        "one_shot_semantics": {
            "future_execution_count": 1,
            "family_frozen_before_execution": True,
            "candidate_count_frozen_before_execution": True,
            "marginal_convention_frozen_before_execution": True,
            "dependence_convention_frozen_before_execution": True,
            "fitting_convention_frozen_before_execution": True,
            "sampling_convention_frozen_before_execution": True,
            "metrics_frozen_before_execution": True,
            "compatibility_frozen_before_execution": True,
            "complexity_order_frozen_before_execution": True,
            "tie_break_frozen_before_execution": True,
            "no_post_result_family_change": True,
            "no_post_result_parameter_change": True,
            "no_post_result_budget_change": True,
            "no_confirmation_consumption": True,
        },
        "claim_boundary": {
            "allowed": [CLAIM_CLASSIFICATION],
            "not_established": [
                "Candidate M or Candidate D development support",
                "a selected producer family before one-shot comparison",
                "a valid copula or tail family in observed results",
                "generalization or population truth",
                "full complex OLS distribution",
                "sealed confirmation",
                "N4/N5/N6 support",
                "semantic generation",
                "generation readiness",
            ],
        },
        "execution_boundary": {
            "q_tensor_loaded_by_this_definition_consumer": False,
            "candidate_M_fit": False,
            "candidate_M_sampling": False,
            "candidate_D_fit": False,
            "candidate_D_sampling": False,
            "new_candidate_results_observed": False,
            "new_candidate_q_results_read": False,
            "candidate_selection_executed": False,
            "sealed_confirmation_read": False,
            "sealed_confirmation_consumed": False,
            "cuda": False,
            "optimizer": False,
            "backward": False,
            "training": False,
            "complex_ols": False,
            "n4_n5_n6": False,
            "generation": False,
            "formal_promotion": False,
            "training_authorized": False,
        },
        "contract_totality": {
            "status": "passed",
            "totality_proven": True,
            "future_runtime_math_choices": [],
            "validated_domains": {
                "candidate_ids": [CANDIDATE_M, CANDIDATE_D],
                "support_boolean_pairs": [[False, False], [False, True], [True, False], [True, True]],
                "invalid_execution": "any candidate/fold operational invalidity stops before support mapping",
                "fold_counts": [FOLD_COUNT],
                "metric_count": [len(SCALAR_METRICS)],
            },
            "failure_stops": [
                "unresolved or hash-mismatched authority",
                "candidate definition or component-isolation ambiguity",
                "new candidate result observed during definition",
                "any candidate/fold numerical or implementation invalidity during later execution",
                "nonfinite or materially negative primary ED",
                "sealed confirmation or generator metric access",
            ],
            "scientific_classification_requires": "a separate later valid one-shot development execution after every authority, identity, numerical and compatibility gate passes",
        },
        "governance": {
            "candidate_selected": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "sealed_confirmation_status": "RESERVED_NOT_MATERIALIZED",
            "sealed_confirmation_consumed": False,
            "current_copula_classification": "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED",
            "attribution_classification": "N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED",
            "oracle_classification": "N3_COPULA_PRIMARY_BOTH_ORACLES_SUFFICIENT",
        },
        "not_executed": [
            "Candidate M fitting",
            "Candidate M sampling",
            "Candidate D fitting",
            "Candidate D sampling",
            "new candidate q development results",
            "candidate support classification",
            "candidate selection",
            "all-1000 final fit",
            "sealed confirmation loading or materialization",
            "complex OLS",
            "N4/N5/N6",
            "generator/image metric",
            "CUDA",
            "optimizer",
            "backward",
            "training",
            "formal promotion",
            "generation",
        ],
        "artifact_policy": {
            "write_once": True,
            "immutable_after_publish": True,
            "definition_is_not_a_result": True,
            "candidate_results_not_embedded": True,
            "existing_oracle_result_modified": False,
            "existing_attribution_result_modified": False,
            "existing_producer_result_modified": False,
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
            artifact_kind="N3 oracle-guided minimal intervention contract",
        )
        if output_dir.exists():
            raise FileExistsError(f"write-once output appeared during contract definition: {output_dir}")
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
    _require(not unexpected_dirty, f"unexpected dirty paths before contract definition: {unexpected_dirty}")
    _require(not output_dir.exists(), f"write-once contract output already exists: {output_dir}")

    oracle = _validate_oracle_inputs(repo_root)
    attribution = _validate_attribution_inputs(repo_root)
    baseline = _validate_baseline_inputs(repo_root)
    payload = _build_payload(repo_root, git, oracle, attribution, baseline)
    contract_path = _write_contract(repo_root, output_dir, payload)
    return {
        "classification": READY_CLASSIFICATION,
        "claim_classification": CLAIM_CLASSIFICATION,
        "semantic_authority": TASK_ID,
        "contract": _relative(repo_root, contract_path),
        "contract_sha256": file_sha256(contract_path),
        "git_branch": git["branch"],
        "git_starting_head": EXPECTED_STARTING_HEAD,
        "git_ending_head": _git_state(repo_root)["head"],
        "worktree_dirty_before_artifact": git["dirty_paths"],
        "candidate_M_fit_or_sampling": False,
        "candidate_D_fit_or_sampling": False,
        "new_candidate_results_observed": False,
        "q_tensors_loaded_as_tensors": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
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
