"""Freeze the N3 q-producer compatibility self-null calibration contract.

This definition-only consumer binds the existing V2 development authority, the
two already executed q producers, and the current raw-q compatibility
evaluator.  It does not load q values, fit a producer, sample a bank, evaluate
held-out rows, or read sealed confirmation values.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256


TASK_ID = "N3_Q_PRODUCER_COMPATIBILITY_SELF_NULL_CALIBRATION_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "n3_q_producer_compatibility_self_null_calibration_contract.v1"
READY_CLASSIFICATION = "N3_Q_PRODUCER_COMPATIBILITY_SELF_NULL_CALIBRATION_V1_READY"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"

OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n3_q_producer_compatibility_self_null_calibration_contract_v1_attempt_01"
)
CONTRACT_NAME = "n3_q_producer_compatibility_self_null_calibration_contract.v1.json"

V2_SOURCE_MANIFEST = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_source_manifest_v1_attempt_01/"
    "n3_development_bank_v2_source_manifest.v1.json"
)
V2_Q_ARTIFACT = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_q_materialization_v1_attempt_01/"
    "n3_development_bank_v2_q.v1.pt"
)
V2_FOLD_MANIFEST = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_five_fold_protocol_v1_attempt_01/"
    "n3_development_bank_v2_fold_manifest.v1.json"
)
V2_CEILING_REPORT = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_empirical_ceiling_v1_attempt_01/"
    "n3_development_bank_v2_empirical_ceiling_report.v1.json"
)
N4_REGISTRY = Path(
    "outputs/20260909_v2/runs/"
    "n4_q_producer_candidate_registry_v1_attempt_02/"
    "n4_q_producer_candidate_registry.v1.json"
)
N4_REPORT = Path(
    "outputs/20260909_v2/runs/"
    "n4_q_producer_development_comparison_v1_attempt_02/"
    "n4_q_producer_comparison_report.v1.json"
)
GAUSSIAN_IMPLEMENTATION = Path(
    "scripts/research/execute_n3_empirical_marginal_gaussian_copula_producer_v1.py"
)
GAUSSIAN_CONTRACT = Path(
    "outputs/20260908_v2/runs/"
    "n3_empirical_marginal_gaussian_copula_producer_contract_v1_attempt_01/"
    "n3_empirical_marginal_gaussian_copula_producer_contract.v1.json"
)
GAUSSIAN_REPORT = Path(
    "outputs/20260908_v2/runs/"
    "n3_empirical_marginal_gaussian_copula_producer_v1_attempt_01/"
    "n3_empirical_marginal_gaussian_copula_producer_report.v1.json"
)
STUDENT_IMPLEMENTATION = Path(
    "scripts/research/execute_n3_empirical_marginal_fixed_r_student_t_copula_v1.py"
)
STUDENT_CONTRACT = Path(
    "outputs/20260909_v2/runs/"
    "n3_empirical_marginal_fixed_r_student_t_copula_contract_v1_attempt_01/"
    "n3_empirical_marginal_fixed_r_student_t_copula_contract.v1.json"
)
STUDENT_REPORT = Path(
    "outputs/20260909_v2/runs/"
    "n3_empirical_marginal_fixed_r_student_t_copula_producer_v1_attempt_01/"
    "n3_empirical_marginal_fixed_r_student_t_copula_producer_report.v1.json"
)
METRIC_IMPLEMENTATION = Path(
    "scripts/research/execute_n3_development_bank_v2_empirical_ceiling.py"
)
AGGREGATION_IMPLEMENTATION = GAUSSIAN_IMPLEMENTATION
GOVERNANCE_STATE = Path("docs/status/research_governance_state.json")

AUTHORITY_SHA256 = {
    V2_SOURCE_MANIFEST.as_posix(): "4bc251af91b3a6cd0235fe113117443d871a564bd05634fa9e61b5ab0b36f812",
    V2_Q_ARTIFACT.as_posix(): "fe48c8fdb9bb2bf732076fabdf475288baa6f7d628ca10039a83bbda54fcfcbf",
    V2_FOLD_MANIFEST.as_posix(): "4898a0e477d583b9a9e60c83238e6d8524d43967efe24277a952089e8adc447e",
    V2_CEILING_REPORT.as_posix(): "731cf826dc081e7cc5629790ba2fb74946f7cc41f1a7f0035bf3a3ca96a31826",
    N4_REGISTRY.as_posix(): "0329b23675a7d306c88f26c80a4757ce83d39c8c68b718bf7a1d966f53f4f27b",
    N4_REPORT.as_posix(): "393d84617ec7b1e6c60b8c654b859dc3afe822f5dee207ca93816e232fea78be",
    GAUSSIAN_IMPLEMENTATION.as_posix(): "5f1fa3fa6be35e96f345c3b7f7c6cce5ca0b45fe4619b1ead4455e86dcae538a",
    GAUSSIAN_CONTRACT.as_posix(): "3edc9fee006f68eff6ba41205b7e4e6b6cbea3205080c4cb333689e554fba7a4",
    GAUSSIAN_REPORT.as_posix(): "78b8e40f21015af8d704a89a83bb6b4043c0208386032b41b1ced4dd9bdeba7d",
    STUDENT_IMPLEMENTATION.as_posix(): "259f45e9c71ed86cf10c5793e84bcc391e68542213281d3de8fff70c0e7e02c7",
    STUDENT_CONTRACT.as_posix(): "0f18dccf35758e590255fac07436eb968c866ddc11a3e19df7b02e25770be1d7",
    STUDENT_REPORT.as_posix(): "4a938b883d1a0cf3386b7e2a5f4c900a236cfed031364c4a1273ea3ebe6d77dd",
    METRIC_IMPLEMENTATION.as_posix(): "18b8f9de fd83c2591dc055a97621ef6a1717783f7656da75994881e02bffda14".replace(" ", ""),
    GOVERNANCE_STATE.as_posix(): "e691f9081df6c429598bb64eff8a693ea2522a3e87fe3894cf6099b50cbb091c",
}

CLASS_COUNT = 10
DIMENSION = 100
POOL_COUNT = 5000
ROWS_PER_CLASS = 500
FOLD_COUNT = 5
FIT_COUNT_PER_FOLD = 4000
HELDOUT_COUNT_PER_FOLD = 1000
FIT_PER_CLASS_PER_FOLD = 400
HELDOUT_PER_CLASS_PER_FOLD = 100
PSEUDO_DATA_SAMPLE_COUNT = 1000
MODEL_SAMPLE_COUNT = 200
DATA_VS_MODEL_REPLICATES = 4
MODEL_VS_MODEL_REPLICATES = 8
MODEL_REFERENCE_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
TRIAL_COUNT_PER_PRODUCER_FOLD = 20
COMPATIBILITY_Q = 0.95

PRODUCER_IDS = (
    "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1",
    "EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_V1",
)
SCALAR_METRICS = (
    "energy_distance",
    "covariance_relative_frobenius",
    "norm_energy_distance",
    "squared_norm_energy_distance",
    "marginal_mean_l2",
    "marginal_std_l2",
    "marginal_energy_distance_mean",
    "marginal_energy_distance_max",
)
JOINT_METRICS = SCALAR_METRICS[:4]
MARGINAL_METRICS = SCALAR_METRICS[4:]
RNG_NAMESPACE = "N3_Q_PRODUCER_COMPATIBILITY_SELF_NULL_CALIBRATION_V1_CPU"
RNG_SEED_RULE = (
    "seed=int.from_bytes(sha256(namespace|producer|fold|trial|role|bank)[:8], 'big') "
    "& ((1<<63)-1)"
)

# These are pre-registered before simulation.  The 20-trial choice is a
# bounded CPU compromise because the exact evaluator includes 100 coordinate
# level cdist-based EDs for every pair.  Its coarse Monte Carlo precision is a
# declared limitation, not an after-the-fact acceptance adjustment.
STABLE_MIN_FOLD_PASS_RATE = 0.95
STABLE_MAX_GUARDRAIL_FALSE_FAIL_RATE = 0.05
STABLE_MIN_IMPLIED_FIVE_FOLD_PROBABILITY = 0.75
LIMITATION_MAX_FOLD_PASS_RATE = 0.75
LIMITATION_MIN_GUARDRAIL_FALSE_FAIL_RATE = 0.25
LIMITATION_MAX_IMPLIED_FIVE_FOLD_PROBABILITY = 0.25


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
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


def _read_json_exact(
    repo_root: Path,
    path: Path,
    expected_sha256: str,
    name: str,
) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{name} is missing: {target}")
    _require(file_sha256(target) == expected_sha256, f"{name} hash changed: {target}")
    return dict(_mapping(json.loads(target.read_text(encoding="utf-8")), name))


def _authority_refs(repo_root: Path) -> list[dict[str, Any]]:
    refs = []
    for path_text, expected_sha256 in AUTHORITY_SHA256.items():
        path = Path(path_text)
        target = repo_root / path
        _require(target.is_file(), f"authority is missing: {target}")
        actual = file_sha256(target)
        _require(actual == expected_sha256, f"authority hash changed: {target}")
        if path == GOVERNANCE_STATE:
            role = "current governance snapshot"
        elif path == METRIC_IMPLEMENTATION:
            role = "frozen V2 raw-q metric adapter"
        elif path == AGGREGATION_IMPLEMENTATION:
            role = "frozen mean/q95 aggregation implementation"
        else:
            role = "frozen input authority"
        refs.append({"path": _relative(repo_root, target), "sha256": actual, "role": role})
    return refs


def _validate_governance(repo_root: Path) -> dict[str, Any]:
    state = _read_json_exact(
        repo_root,
        GOVERNANCE_STATE,
        AUTHORITY_SHA256[GOVERNANCE_STATE.as_posix()],
        "current governance state",
    )
    _require(state.get("active_objective") == "N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED", "N4 blocker is not active")
    _require(state.get("hard_blocker") == "N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED", "N4 hard blocker changed")
    _require(state.get("authorization_tier") == "R0", "authorization tier changed")
    aliases = _mapping(state.get("evidence_aliases"), "governance evidence aliases")
    for name in (
        "N3_DEVELOPMENT_BANK_V2_SOURCE",
        "N3_DEVELOPMENT_BANK_V2_Q",
        "N3_DEVELOPMENT_BANK_V2_FIVE_FOLD_PROTOCOL",
        "N3_DEVELOPMENT_BANK_V2_EMPIRICAL_CEILING",
        "N4_Q_PRODUCER_DEVELOPMENT_COMPARISON",
        "N3_EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_CONTRACT",
        "N3_EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_PRODUCER",
    ):
        item = _mapping(aliases.get(name), name)
        for key, expected in (
            ("selected_producer", None),
            ("training_authorized", False),
            ("formal_promotion", False),
            ("generation_readiness", "not_ready"),
            ("sealed_confirmation_consumed", False),
        ):
            if key in item:
                _require(item.get(key) == expected, f"governance boundary changed: {name}.{key}")
    return {
        "path": _relative(repo_root, repo_root / GOVERNANCE_STATE),
        "sha256": AUTHORITY_SHA256[GOVERNANCE_STATE.as_posix()],
        "active_objective": state["active_objective"],
        "hard_blocker": state["hard_blocker"],
        "authorization_tier": state["authorization_tier"],
        "selected_producer": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_consumed": False,
    }


def _validate_authorities(repo_root: Path) -> dict[str, Any]:
    source = _read_json_exact(repo_root, V2_SOURCE_MANIFEST, AUTHORITY_SHA256[V2_SOURCE_MANIFEST.as_posix()], "V2 source manifest")
    _require(source.get("status") == "FROZEN_SOURCE_ROWS", "V2 source status changed")
    source_identity = _mapping(source.get("identity"), "V2 source identity")
    source_class_counts = source_identity.get("class_counts")
    _require(
        isinstance(source_class_counts, list)
        and len(source_class_counts) == CLASS_COUNT
        and source_class_counts == [ROWS_PER_CLASS] * CLASS_COUNT,
        "V2 source counts changed",
    )
    _require(sum(source_class_counts) == POOL_COUNT, "V2 source pool count changed")
    source_boundary = _mapping(source.get("materialization_boundary"), "V2 source materialization boundary")
    _require(source.get("artifact_policy", {}).get("q_values_embedded") is False, "V2 source embedded q values")
    _require(source_boundary.get("q_values_loaded") is False and source_boundary.get("q_values_materialized") is False, "V2 source used q values")

    fold = _read_json_exact(repo_root, V2_FOLD_MANIFEST, AUTHORITY_SHA256[V2_FOLD_MANIFEST.as_posix()], "V2 fold manifest")
    _require(fold.get("status") == "FROZEN_BEFORE_VALIDATION", "V2 fold status changed")
    fold_protocol = _mapping(fold.get("protocol"), "V2 fold protocol")
    _require(fold_protocol.get("fold_count") == FOLD_COUNT, "V2 fold count changed")
    _require(fold_protocol.get("fit_count_per_fold") == FIT_COUNT_PER_FOLD, "V2 fit count changed")
    _require(fold_protocol.get("heldout_count_per_fold") == HELDOUT_COUNT_PER_FOLD, "V2 heldout count changed")
    _require(fold_protocol.get("fit_per_class_per_fold") == FIT_PER_CLASS_PER_FOLD, "V2 fit class balance changed")
    _require(fold_protocol.get("heldout_per_class_per_fold") == HELDOUT_PER_CLASS_PER_FOLD, "V2 heldout class balance changed")
    _require(fold_protocol.get("q_values_used") is False, "V2 fold construction used q values")

    ceiling = _read_json_exact(repo_root, V2_CEILING_REPORT, AUTHORITY_SHA256[V2_CEILING_REPORT.as_posix()], "V2 ceiling report")
    _require(ceiling.get("status") == "valid_completed" and ceiling.get("result") == "VALID_COMPLETED", "V2 ceiling is incomplete")
    _require(ceiling.get("classification") == "N3_DEVELOPMENT_BANK_V2_VALIDATED", "V2 ceiling classification changed")
    settings = _mapping(ceiling.get("execution_settings"), "V2 ceiling execution settings")
    expected_settings = {
        "fold_count": FOLD_COUNT,
        "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
        "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
        "generated_bank_size": MODEL_SAMPLE_COUNT,
        "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
        "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
        "model_reference_pairs": [list(pair) for pair in MODEL_REFERENCE_PAIRS],
        "compatibility_q": COMPATIBILITY_Q,
        "raw_q_energy_distance": True,
        "guardrails": list(SCALAR_METRICS),
    }
    for key, expected in expected_settings.items():
        _require(settings.get(key) == expected, f"V2 evaluator setting changed: {key}")
    aggregate = _mapping(ceiling.get("aggregate"), "V2 ceiling aggregate")
    _require(aggregate.get("compatible_fold_pattern") == [True] * FOLD_COUNT, "V2 ceiling is not five-fold validated")
    _require(aggregate.get("producer_evaluated") is False and aggregate.get("producer_selection") is False, "V2 ceiling producer boundary changed")

    registry = _read_json_exact(repo_root, N4_REGISTRY, AUTHORITY_SHA256[N4_REGISTRY.as_posix()], "N4 candidate registry")
    _require(registry.get("status") == "FROZEN", "N4 registry is not frozen")
    _require(registry.get("candidate_count") == 3, "N4 registry candidate count changed")
    registry_governance = _mapping(registry.get("governance"), "N4 registry governance")
    _require(registry_governance.get("selected_producer") is None, "N4 registry selected a producer")
    _require(registry_governance.get("training_authorized") is False, "N4 registry opened training")

    n4 = _read_json_exact(repo_root, N4_REPORT, AUTHORITY_SHA256[N4_REPORT.as_posix()], "N4 comparison report")
    _require(n4.get("status") == "valid_completed" and n4.get("result") == "VALID_COMPLETED", "N4 comparison is incomplete")
    _require(n4.get("classification") == "N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED", "N4 classification changed")
    _require(n4.get("development_supported_candidates") == [], "N4 selected a supported candidate")
    n4_governance = _mapping(n4.get("governance"), "N4 governance")
    _require(n4_governance.get("selected_producer") is None, "N4 selected a producer")
    _require(n4_governance.get("training_authorized") is False, "N4 opened training")

    gaussian = _read_json_exact(repo_root, GAUSSIAN_CONTRACT, AUTHORITY_SHA256[GAUSSIAN_CONTRACT.as_posix()], "Gaussian producer contract")
    _require(gaussian.get("status") == "valid_completed", "Gaussian contract is incomplete")
    _require(gaussian.get("classification") == "DEFINE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1_READY", "Gaussian contract classification changed")
    gaussian_family = _mapping(gaussian.get("family"), "Gaussian producer family")
    _require(gaussian_family.get("name") == "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1", "Gaussian producer identity changed")
    gaussian_report = _read_json_exact(repo_root, GAUSSIAN_REPORT, AUTHORITY_SHA256[GAUSSIAN_REPORT.as_posix()], "Gaussian producer report")
    _require(gaussian_report.get("status") == "valid_completed", "Gaussian producer report is incomplete")
    _require(gaussian_report.get("classification") == "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED", "Gaussian producer result changed")

    student = _read_json_exact(repo_root, STUDENT_CONTRACT, AUTHORITY_SHA256[STUDENT_CONTRACT.as_posix()], "Student-t producer contract")
    _require(student.get("status") == "valid_completed" and student.get("result") == "READY", "Student-t contract is incomplete")
    student_family = _mapping(student.get("family"), "Student-t producer family")
    _require(student_family.get("family_id") == "EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_V1", "Student-t producer identity changed")
    _require(student_family.get("candidate_ids") == ["EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_V1"], "Student-t producer candidate set changed")
    student_report = _read_json_exact(repo_root, STUDENT_REPORT, AUTHORITY_SHA256[STUDENT_REPORT.as_posix()], "Student-t producer report")
    _require(student_report.get("status") == "valid_completed", "Student-t producer report is incomplete")
    _require(student_report.get("classification") == "N3_FIXED_R_STUDENT_T_COPULA_NOT_SUPPORTED", "Student-t producer result changed")

    governance = _validate_governance(repo_root)
    return {
        "source": source,
        "fold": fold,
        "ceiling": ceiling,
        "registry": registry,
        "n4": n4,
        "gaussian_contract": gaussian,
        "gaussian_report": gaussian_report,
        "student_contract": student,
        "student_report": student_report,
        "governance": governance,
    }


def _run_contract_totality_checks() -> dict[str, Any]:
    _require(PRODUCER_IDS == tuple(dict.fromkeys(PRODUCER_IDS)), "producer identities are not unique")
    _require(len(PRODUCER_IDS) == 2, "producer set is not exactly two")
    _require(TRIAL_COUNT_PER_PRODUCER_FOLD == 20, "trial count is not the pre-registered bounded count")
    _require(PSEUDO_DATA_SAMPLE_COUNT == HELDOUT_COUNT_PER_FOLD, "pseudo-DATA count changed")
    _require(MODEL_SAMPLE_COUNT == 200, "MODEL count changed")
    _require(DATA_VS_MODEL_REPLICATES == 4 and MODEL_VS_MODEL_REPLICATES == 8, "replicate counts changed")
    _require(MODEL_REFERENCE_PAIRS == ((0, 1), (2, 3), (4, 5), (6, 7)), "reference pairs changed")
    _require(len(SCALAR_METRICS) == 8 and len(set(SCALAR_METRICS)) == 8, "guardrail set is incomplete")
    _require(tuple(JOINT_METRICS) + tuple(MARGINAL_METRICS) == SCALAR_METRICS, "guardrail partition changed")
    _require(0.0 < COMPATIBILITY_Q < 1.0, "compatibility quantile is invalid")
    _require(STABLE_MIN_FOLD_PASS_RATE > LIMITATION_MAX_FOLD_PASS_RATE, "classification thresholds overlap")
    _require(STABLE_MAX_GUARDRAIL_FALSE_FAIL_RATE < LIMITATION_MIN_GUARDRAIL_FALSE_FAIL_RATE, "guardrail thresholds overlap")
    _require(STABLE_MIN_IMPLIED_FIVE_FOLD_PROBABILITY > LIMITATION_MAX_IMPLIED_FIVE_FOLD_PROBABILITY, "five-fold thresholds overlap")
    _require(all("|" not in value for value in PRODUCER_IDS), "producer identity is ambiguous in RNG namespace")
    roles = ("pseudo-data", "data-vs-model", "model-vs-model-reference")
    _require(len(roles) == len(set(roles)), "RNG role namespaces are not disjoint")
    return {
        "producer_count": len(PRODUCER_IDS),
        "fold_count": FOLD_COUNT,
        "trial_count_per_producer_fold": TRIAL_COUNT_PER_PRODUCER_FOLD,
        "total_trials": len(PRODUCER_IDS) * FOLD_COUNT * TRIAL_COUNT_PER_PRODUCER_FOLD,
        "guardrail_count": len(SCALAR_METRICS),
        "rng_roles": list(roles),
        "sample_counts_positive": True,
        "thresholds_nonoverlapping": True,
        "no_post_observation_threshold_changes": True,
    }


def _build_payload(repo_root: Path, git: Mapping[str, Any], authorities: Mapping[str, Any]) -> dict[str, Any]:
    totality = _run_contract_totality_checks()
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "READY",
        "classification": READY_CLASSIFICATION,
        "definition_only": True,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git": {
            **dict(git),
            "starting_committed_head": EXPECTED_STARTING_HEAD,
            "contract_script": _relative(repo_root, Path(__file__).resolve()),
            "contract_script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "authorities": {
            "artifact_refs": _authority_refs(repo_root),
            "v2_source": {
                "semantic_authority_id": "N3_DEVELOPMENT_BANK_V2",
                "count": POOL_COUNT,
                "rows_per_class": ROWS_PER_CLASS,
                "q_values_loaded_by_definition_consumer": False,
            },
            "v2_q": {
                "artifact_path": _relative(repo_root, repo_root / V2_Q_ARTIFACT),
                "artifact_sha256": AUTHORITY_SHA256[V2_Q_ARTIFACT.as_posix()],
                "tensor_sha256": "d660fd8dc449725edab081d00c8f553f151029cddef9210360a7a6fad4281209",
                "values_loaded_by_definition_consumer": False,
            },
            "v2_five_fold": {
                "fold_count": FOLD_COUNT,
                "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
                "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
                "fit_per_class_per_fold": FIT_PER_CLASS_PER_FOLD,
                "heldout_per_class_per_fold": HELDOUT_PER_CLASS_PER_FOLD,
                "assignment_sha256": "43ee0ce28e143fffdf2bcac98394625ccd766166ea717f23db66ed8770ef18a9",
                "q_values_used_by_definition_consumer": False,
            },
            "n4_context": {
                "classification": authorities["n4"]["classification"],
                "candidate_results_observed": True,
                "selected_producer": None,
                "training_authorized": False,
            },
        },
        "producer_set": {
            "count": len(PRODUCER_IDS),
            "identities": list(PRODUCER_IDS),
            "comparison_scope": "exactly the two already executed producer families; no third family",
            "heldout_refit": False,
            "heldout_outcome_tuning": False,
            "sealed_values_used": False,
            "producer_selection": False,
            "entries": [
                {
                    "producer_id": PRODUCER_IDS[0],
                    "implementation_path": _relative(repo_root, repo_root / GAUSSIAN_IMPLEMENTATION),
                    "implementation_sha256": AUTHORITY_SHA256[GAUSSIAN_IMPLEMENTATION.as_posix()],
                    "contract_path": _relative(repo_root, repo_root / GAUSSIAN_CONTRACT),
                    "contract_sha256": AUTHORITY_SHA256[GAUSSIAN_CONTRACT.as_posix()],
                    "fit_rule": "fit each fold once from its 4000 frozen fitting q rows only",
                    "sampling_rule": "unchanged fit-only empirical marginals and fixed R_shrunk Gaussian copula",
                },
                {
                    "producer_id": PRODUCER_IDS[1],
                    "implementation_path": _relative(repo_root, repo_root / STUDENT_IMPLEMENTATION),
                    "implementation_sha256": AUTHORITY_SHA256[STUDENT_IMPLEMENTATION.as_posix()],
                    "contract_path": _relative(repo_root, repo_root / STUDENT_CONTRACT),
                    "contract_sha256": AUTHORITY_SHA256[STUDENT_CONTRACT.as_posix()],
                    "fit_rule": "fit each fold once from its 4000 frozen fitting q rows only",
                    "sampling_rule": "unchanged fit-only empirical marginals, fixed R_shrunk, and fitted nu; nu/marginals/R are not changed",
                },
            ],
        },
        "simulation": {
            "device": "cpu",
            "dtype": "torch.float64",
            "folds": list(range(FOLD_COUNT)),
            "trials_per_producer_fold": TRIAL_COUNT_PER_PRODUCER_FOLD,
            "no_early_stop": True,
            "no_retry": True,
            "best_of_n": False,
            "pseudo_data": {
                "role": "DATA",
                "sample_count": PSEUDO_DATA_SAMPLE_COUNT,
                "source": "one independent bank generated from the fitted producer",
            },
            "model_banks": {
                "role": "MODEL",
                "sample_count": MODEL_SAMPLE_COUNT,
                "count_per_trial": DATA_VS_MODEL_REPLICATES,
                "independent_from_pseudo_data": True,
            },
            "reference_banks": {
                "role": "MODEL_VS_MODEL_REFERENCE",
                "sample_count": MODEL_SAMPLE_COUNT,
                "count_per_trial": MODEL_VS_MODEL_REPLICATES,
                "pairs": [list(pair) for pair in MODEL_REFERENCE_PAIRS],
                "independent_from_pseudo_data_and_model_banks": True,
            },
            "sample_size_asymmetry": {
                "pseudo_data_vs_model": [PSEUDO_DATA_SAMPLE_COUNT, MODEL_SAMPLE_COUNT],
                "model_vs_model_reference": [MODEL_SAMPLE_COUNT, MODEL_SAMPLE_COUNT],
                "exact_current_v2_structure": True,
            },
            "rng": {
                "namespace": RNG_NAMESPACE,
                "seed_rule": RNG_SEED_RULE,
                "fields": ["producer", "fold", "trial", "role", "bank"],
                "roles": ["pseudo-data", "data-vs-model", "model-vs-model-reference"],
                "disjoint_namespaces": True,
            },
            "trial_count_decision": {
                "preferred_minimum": 200,
                "chosen_count": TRIAL_COUNT_PER_PRODUCER_FOLD,
                "reason": "exact existing evaluator has 8 pair calls per trial and 100 coordinate-level cdist EDs per pair; 200 trials per producer/fold is outside the bounded CPU envelope",
                "pair_evaluations_at_chosen_count": len(PRODUCER_IDS) * FOLD_COUNT * TRIAL_COUNT_PER_PRODUCER_FOLD * 8,
                "pair_evaluations_at_preferred_count": len(PRODUCER_IDS) * FOLD_COUNT * 200 * 8,
                "monte_carlo_precision": "with n=20 Bernoulli trials per producer/fold, rate estimates are coarse; this calibration cannot resolve small (<~0.2) rate differences",
            },
        },
        "evaluator": {
            "implementation_path": _relative(repo_root, repo_root / METRIC_IMPLEMENTATION),
            "implementation_sha256": AUTHORITY_SHA256[METRIC_IMPLEMENTATION.as_posix()],
            "pair_metric_function": "_pair_metrics_raw_q",
            "aggregation_implementation_path": _relative(repo_root, repo_root / AGGREGATION_IMPLEMENTATION),
            "aggregation_implementation_sha256": AUTHORITY_SHA256[AGGREGATION_IMPLEMENTATION.as_posix()],
            "aggregation_functions": ["_mean_metrics", "_q95_metrics"],
            "primary_metric": "energy_distance",
            "metric_family": "biased empirical raw-q metrics with diagonal terms included",
            "joint_guardrails": list(JOINT_METRICS),
            "marginal_guardrails": list(MARGINAL_METRICS),
            "all_eight_scalar_guardrails": list(SCALAR_METRICS),
            "per_coordinate_metric": "per_coordinate_marginal_energy_distance",
            "per_coordinate_vector_role": "recorded and checked for finiteness; descriptive only, not an additional gate",
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "model_reference_pairs": [list(pair) for pair in MODEL_REFERENCE_PAIRS],
            "threshold_quantile": COMPATIBILITY_Q,
            "threshold_source": "trial-local q95 of four MODEL-vs-MODEL reference-pair metrics generated from the same fitted producer",
            "pass_rule": "mean of four pseudo-DATA-vs-MODEL metrics <= corresponding trial-local MODEL-vs-MODEL q95 for every one of the eight scalar guardrails; equality passes",
            "complete_existing_evaluator": True,
            "new_evaluator_created": False,
        },
        "aggregate_classification": {
            "allowed": [
                "N3_Q_COMPATIBILITY_SELF_NULL_STABLE",
                "N3_Q_COMPATIBILITY_SELF_NULL_FALSE_FAIL_LIMITATION",
                "N3_Q_COMPATIBILITY_SELF_NULL_MIXED_OR_UNRESOLVED",
            ],
            "classification_frozen_before_simulation": True,
            "rate_definitions": {
                "total_pass_rate": "count(fold_pass) / fixed trial count",
                "primary_false_fail_rate": "count(not energy_distance component) / fixed trial count",
                "guardrail_false_fail_rate": "count(not guardrail component) / fixed trial count",
                "implied_five_fold_all_pass_probability": "product of the five observed fold pass rates; independence-based diagnostic only",
            },
            "stable_if": {
                "every_producer_fold_pass_rate_at_least": STABLE_MIN_FOLD_PASS_RATE,
                "every_producer_fold_guardrail_false_fail_rate_at_most": STABLE_MAX_GUARDRAIL_FALSE_FAIL_RATE,
                "every_producer_implied_five_fold_probability_at_least": STABLE_MIN_IMPLIED_FIVE_FOLD_PROBABILITY,
            },
            "false_fail_limitation_if": {
                "any_producer_fold_pass_rate_at_most": LIMITATION_MAX_FOLD_PASS_RATE,
                "or_any_producer_fold_guardrail_false_fail_rate_at_least": LIMITATION_MIN_GUARDRAIL_FALSE_FAIL_RATE,
                "or_any_producer_implied_five_fold_probability_at_most": LIMITATION_MAX_IMPLIED_FIVE_FOLD_PROBABILITY,
            },
            "otherwise": "N3_Q_COMPATIBILITY_SELF_NULL_MIXED_OR_UNRESOLVED",
            "no_threshold_changes_after_observation": True,
        },
        "firewalls": {
            "heldout_q_metric_evaluation": False,
            "heldout_q_used_for_fit_or_sampling": False,
            "sealed_confirmation_values_loaded": False,
            "new_producer_family": False,
            "nu_changed": False,
            "marginals_changed": False,
            "R_changed": False,
            "optical_or_image_evaluation": False,
            "cuda_or_training": False,
            "selected_producer": None,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "contract_totality": totality,
        "claim_boundary": {
            "fact": "empirical self-null false-failure behavior of the existing q compatibility evaluator under this finite simulation contract",
            "model_assumption": "whether the observed isolated 4/5 producer failures can materially be attributed to evaluator noise",
            "not_established": [
                "population stationarity",
                "producer selection",
                "optical compatibility",
                "image generation",
                "training authorization",
                "formal promotion",
            ],
        },
    }


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once contract output: {output_dir}")
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the contract branch")
    _require(git["head"] == EXPECTED_STARTING_HEAD, "current HEAD is not the frozen V2/N4 base")
    authorities = _validate_authorities(repo_root)
    payload = _build_payload(repo_root, git, authorities)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(temporary / CONTRACT_NAME, payload, artifact_kind="N3 self-null contract")
        if output_dir.exists():
            raise FileExistsError(f"contract output appeared during definition: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    contract_path = output_dir / CONTRACT_NAME
    return {
        "status": "valid_completed",
        "result": "READY",
        "classification": READY_CLASSIFICATION,
        "contract": _relative(repo_root, contract_path),
        "contract_sha256": file_sha256(contract_path),
        "trial_count_per_producer_fold": TRIAL_COUNT_PER_PRODUCER_FOLD,
        "producer_ids": list(PRODUCER_IDS),
        "q_values_loaded": False,
        "sealed_confirmation_values_loaded": False,
        "selected_producer": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
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
