"""Apply the frozen V3 q evaluator to the two immutable candidate banks.

This consumer deliberately performs no fitting and no sampling.  It reuses the
exact model-q artifacts recorded by the N4 Gaussian comparison and the fixed-R
Student-t execution, reconstructs each exact V2 held-out DATA fold, and changes
only the compatibility decision boundary to the already-calibrated V3 rule:
``energy_distance`` is HARD and the other seven scalar metrics are descriptive.
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

import torch

from ldm_project.v2.evidence import (
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import (
    compare_n3_minimal_competing_producer_baselines_v1 as baseline,
    execute_n3_development_bank_v2_empirical_ceiling as v2_ceiling,
)


TASK_ID = "APPLY_N3_Q_COMPATIBILITY_EVALUATOR_V3_TO_FROZEN_CANDIDATES_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_q_compatibility_evaluator_v3_frozen_candidate_replay_report.v1"
LINEAGE_SCHEMA = "n3_q_compatibility_evaluator_v3_frozen_candidate_replay_lineage.v1"
OUTPUT_DIR = Path(
    "outputs/20260909_v2/runs/"
    "n3_q_compatibility_evaluator_v3_frozen_candidate_replay_v1_attempt_01"
)
REPORT_NAME = "n3_q_compatibility_evaluator_v3_frozen_candidate_replay_report.v1.json"
LINEAGE_NAME = "n3_q_compatibility_evaluator_v3_frozen_candidate_replay_lineage.v1.json"

V3_EVALUATOR_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_q_compatibility_guardrail_gate_v1_attempt_01/"
    "n3_q_compatibility_evaluator_v3.v1.json"
)
V3_EVALUATOR_SHA256 = "4fbc831e383897802134ea2ce1aaef6446629130a7c799630254a86b8dcff3cc"
CALIBRATION_REPORT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_q_compatibility_guardrail_gate_v1_attempt_01/"
    "n3_q_compatibility_guardrail_calibration_report.v1.json"
)
CALIBRATION_REPORT_SHA256 = "7135b07fd532ee5dd2dba7900ad6028c9a6d558924691e4d65eac56330367789"
CALIBRATION_CONTRACT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_q_compatibility_guardrail_gate_contract_v1_attempt_01/"
    "n3_q_compatibility_guardrail_gate_contract.v1.json"
)
CALIBRATION_CONTRACT_SHA256 = "a87979f9689159eee6d769f10eb6b7e30bcdc9803c4b72b108225a6ea22935b4"

V2_CEILING_REPORT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_empirical_ceiling_v1_attempt_01/"
    "n3_development_bank_v2_empirical_ceiling_report.v1.json"
)
V2_CEILING_REPORT_SHA256 = "731cf826dc081e7cc5629790ba2fb74946f7cc41f1a7f0035bf3a3ca96a31826"
GOVERNANCE_STATE_PATH = Path("docs/status/research_governance_state.json")
METRIC_IMPLEMENTATION_PATH = Path("scripts/research/execute_n3_development_bank_v2_empirical_ceiling.py")
METRIC_IMPLEMENTATION_SHA256 = "18b8f9defd83c2591dc055a97621ef6a1717783f7656da75994881e02bffda14"
GAUSSIAN_AGGREGATION_PATH = Path("scripts/research/execute_n3_empirical_marginal_gaussian_copula_producer_v1.py")
GAUSSIAN_AGGREGATION_SHA256 = "5f1fa3fa6be35e96f345c3b7f7c6cce5ca0b45fe4619b1ead4455e86dcae538a"

N4_REGISTRY_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n4_q_producer_candidate_registry_v1_attempt_02/"
    "n4_q_producer_candidate_registry.v1.json"
)
N4_REGISTRY_SHA256 = "0329b23675a7d306c88f26c80a4757ce83d39c8c68b718bf7a1d966f53f4f27b"
N4_REPORT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n4_q_producer_development_comparison_v1_attempt_02/"
    "n4_q_producer_comparison_report.v1.json"
)
N4_REPORT_SHA256 = "393d84617ec7b1e6c60b8c654b859dc3afe822f5dee207ca93816e232fea78be"
N4_FOLD_COMPATIBILITY_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n4_q_producer_development_comparison_v1_attempt_02/"
    "n4_q_producer_fold_compatibility.v1.json"
)
N4_FOLD_COMPATIBILITY_SHA256 = "d8a273f4291ebf1e29a39b1f45e2f47a3dd919b503fbcf8256c81c69892d4ba5"
N4_FIT_MANIFEST_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n4_q_producer_development_comparison_v1_attempt_02/"
    "n4_q_producer_fit_manifest.v1.json"
)
N4_FIT_MANIFEST_SHA256 = "9c8aae1781787d7744f846d519ac4c3c59abf07b562201a2c6e102eb45fdc1a3"
N4_EXPOSURE_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n4_q_producer_development_comparison_v1_attempt_02/"
    "n4_q_producer_evaluator_exposure_ledger.v1.json"
)
N4_EXPOSURE_SHA256 = "694003034bcbe49f82d7d3ebcf7d2bd014eb711f4fde292606f62b078a818f15"
N4_LINEAGE_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n4_q_producer_development_comparison_v1_attempt_02/"
    "n4_q_producer_lineage_manifest.v1.json"
)
N4_LINEAGE_SHA256 = "4c40fcd64c7856722fe3c3ca9bf61ef1ad2362d10f1f05d349038edc45a2cce5"

STUDENT_CONTRACT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_empirical_marginal_fixed_r_student_t_copula_contract_v1_attempt_01/"
    "n3_empirical_marginal_fixed_r_student_t_copula_contract.v1.json"
)
STUDENT_CONTRACT_SHA256 = "0f18dccf35758e590255fac07436eb968c866ddc11a3e19df7b02e25770be1d7"
STUDENT_REPORT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_empirical_marginal_fixed_r_student_t_copula_producer_v1_attempt_01/"
    "n3_empirical_marginal_fixed_r_student_t_copula_producer_report.v1.json"
)
STUDENT_REPORT_SHA256 = "4a938b883d1a0cf3386b7e2a5f4c900a236cfed031364c4a1273ea3ebe6d77dd"
STUDENT_FOLD_COMPATIBILITY_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_empirical_marginal_fixed_r_student_t_copula_producer_v1_attempt_01/"
    "n3_empirical_marginal_fixed_r_student_t_copula_fold_compatibility.v1.json"
)
STUDENT_FOLD_COMPATIBILITY_SHA256 = "0d6a1f5b2b6f34d4effce26140d4b1a5a6d947d7e4147c7e2b7c95a4fa07cc0c"
STUDENT_FIT_MANIFEST_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_empirical_marginal_fixed_r_student_t_copula_producer_v1_attempt_01/"
    "n3_empirical_marginal_fixed_r_student_t_copula_fit_manifest.v1.json"
)
STUDENT_FIT_MANIFEST_SHA256 = "ab1aef6656cbffd39e1e520f5a2b91a4fdef2b52c808c3ac47f3dfb3e580e6fa"
STUDENT_EXPOSURE_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_empirical_marginal_fixed_r_student_t_copula_producer_v1_attempt_01/"
    "n3_empirical_marginal_fixed_r_student_t_copula_evaluator_exposure_ledger.v1.json"
)
STUDENT_EXPOSURE_SHA256 = "b5591febc0869dc7ebc6c39e54b6a10c9d3043fd92f7852c8ce6c56c9cf3fac6"
STUDENT_LINEAGE_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_empirical_marginal_fixed_r_student_t_copula_producer_v1_attempt_01/"
    "n3_empirical_marginal_fixed_r_student_t_copula_lineage_manifest.v1.json"
)
STUDENT_LINEAGE_SHA256 = "f2f943dab288726db6aaf3fb6c90acfe516ac7869fa9203a2c2a083731cf88da"

CLASS_COUNT = 10
DIMENSION = 100
POOL_COUNT = 5000
FOLD_COUNT = 5
FIT_COUNT_PER_FOLD = 4000
HELDOUT_COUNT_PER_FOLD = 1000
FIT_PER_CLASS_PER_FOLD = 400
HELDOUT_PER_CLASS_PER_FOLD = 100
MODEL_SAMPLE_COUNT = 200
COMPATIBILITY_Q = 0.95
SCALAR_METRICS = tuple(v2_ceiling.SCALAR_METRICS)
PRIMARY_METRIC = "energy_distance"
DESCRIPTIVE_METRICS = tuple(name for name in SCALAR_METRICS if name != PRIMARY_METRIC)
EXPECTED_HARD_GUARDRAILS = [PRIMARY_METRIC]

V2_SOURCE_MANIFEST = v2_ceiling.SOURCE_MANIFEST
V2_Q_MATERIALIZATION_MANIFEST = v2_ceiling.Q_MATERIALIZATION_MANIFEST
V2_Q_ARTIFACT = v2_ceiling.Q_ARTIFACT
V2_FOLD_MANIFEST = v2_ceiling.FOLD_MANIFEST
V2_SOURCE_MANIFEST_SHA256 = v2_ceiling.SOURCE_MANIFEST_SHA256
V2_Q_MATERIALIZATION_MANIFEST_SHA256 = v2_ceiling.Q_MATERIALIZATION_MANIFEST_SHA256
V2_Q_ARTIFACT_SHA256 = v2_ceiling.Q_ARTIFACT_SHA256
V2_Q_TENSOR_SHA256 = v2_ceiling.Q_TENSOR_SHA256
V2_FOLD_MANIFEST_SHA256 = v2_ceiling.FOLD_MANIFEST_SHA256
V2_ASSIGNMENT_SHA256 = v2_ceiling.FOLD_ASSIGNMENT_SHA256

GAUSSIAN_ID = "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1"
STUDENT_ID = "EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_V1"
GAUSSIAN_SUPPORTED_CLASSIFICATION = "N3_EMPIRICAL_GAUSSIAN_COPULA_V3_DEVELOPMENT_SUPPORTED"
GAUSSIAN_NOT_SUPPORTED_CLASSIFICATION = "N3_EMPIRICAL_GAUSSIAN_COPULA_V3_NOT_SUPPORTED"
STUDENT_SUPPORTED_CLASSIFICATION = "N3_FIXED_R_STUDENT_T_COPULA_V3_DEVELOPMENT_SUPPORTED"
STUDENT_NOT_SUPPORTED_CLASSIFICATION = "N3_FIXED_R_STUDENT_T_COPULA_V3_NOT_SUPPORTED"
OUTER_SUPPORTED_CLASSIFICATION = "N3_Q_PRODUCER_V3_DEVELOPMENT_SUPPORTED"
OUTER_NOT_SUPPORTED_CLASSIFICATION = "N3_Q_PRODUCER_V3_NOT_SUPPORTED"


class InvalidHarnessError(RuntimeError):
    """A frozen input, evaluator, or output-boundary invariant failed."""


def _invalid(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidHarnessError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _invalid(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _sequence(value: Any, name: str) -> list[Any]:
    _invalid(isinstance(value, list), f"{name} must be a list")
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
    _invalid(target.is_file(), f"{name} is missing: {target}")
    _invalid(file_sha256(target) == expected_sha256, f"{name} hash changed: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidHarnessError(f"{name} cannot be read as JSON: {target}") from exc
    return dict(_mapping(value, name))


def _json_ref(repo_root: Path, path: Path, sha256: str, role: str, schema: str) -> dict[str, Any]:
    return {
        "path": _relative(repo_root, repo_root / path),
        "sha256": sha256,
        "role": role,
        "schema": schema,
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


def _validate_current_governance(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state_path = repo_root / GOVERNANCE_STATE_PATH
    _invalid(state_path.is_file(), "research governance state is missing")
    state = dict(_mapping(json.loads(state_path.read_text(encoding="utf-8")), "research governance state"))
    _invalid(
        state.get("active_objective") == "N3_Q_COMPATIBILITY_GUARDRAIL_CALIBRATION_READY",
        "current active objective changed before V3 candidate replay",
    )
    _invalid(
        state.get("hard_blocker") == "N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED",
        "current producer blocker changed before V3 candidate replay",
    )
    aliases = _mapping(state.get("evidence_aliases"), "governance evidence aliases")
    calibration = _mapping(
        aliases.get("N3_Q_COMPATIBILITY_GUARDRAIL_CALIBRATION_V1"),
        "V3 calibration governance alias",
    )
    _invalid(calibration.get("classification") == "N3_Q_COMPATIBILITY_GUARDRAIL_CALIBRATION_READY", "V3 calibration alias changed")
    _invalid(calibration.get("v3_evaluator_sha256") == V3_EVALUATOR_SHA256, "governance V3 evaluator hash changed")
    n4 = _mapping(aliases.get("N4_Q_PRODUCER_DEVELOPMENT_COMPARISON"), "N4 governance alias")
    _invalid(n4.get("classification") == "N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED", "historical N4 governance classification changed")
    for key, expected in {
        "selected_producer": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items():
        _invalid(calibration.get(key) == expected, f"project governance flag changed in calibration authority: {key}")
    snapshot = {
        "path": _relative(repo_root, state_path),
        "sha256": file_sha256(state_path),
        "active_objective": state["active_objective"],
        "hard_blocker": state["hard_blocker"],
        "selected_producer": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }
    return state, snapshot


def _validate_v3_authority(repo_root: Path) -> dict[str, Any]:
    contract = _read_json_exact(repo_root, CALIBRATION_CONTRACT_PATH, CALIBRATION_CONTRACT_SHA256, "V3 calibration contract")
    calibration = _read_json_exact(repo_root, CALIBRATION_REPORT_PATH, CALIBRATION_REPORT_SHA256, "V3 calibration report")
    evaluator = _read_json_exact(repo_root, V3_EVALUATOR_PATH, V3_EVALUATOR_SHA256, "V3 evaluator artifact")

    _invalid(contract.get("schema") == "n3_q_compatibility_guardrail_gate_contract.v1", "V3 calibration contract schema changed")
    _invalid(contract.get("status") == "valid_completed" and contract.get("result") == "READY", "V3 calibration contract is not ready")
    _invalid(contract.get("classification") == "DEFINE_N3_Q_COMPATIBILITY_GUARDRAIL_GATE_V1_READY", "V3 calibration contract classification changed")
    contract_evaluator = _mapping(contract.get("evaluator"), "V3 calibration evaluator contract")
    _invalid(contract_evaluator.get("metrics") == list(SCALAR_METRICS), "V3 calibration metric order changed")
    _invalid(contract_evaluator.get("primary_metric") == PRIMARY_METRIC, "V3 calibration primary metric changed")
    _invalid(contract_evaluator.get("primary_energy_distance_unchanged") is True, "V3 calibration primary semantics changed")
    _invalid(contract_evaluator.get("new_evaluator_created") is False, "V3 calibration introduced a new evaluator")

    _invalid(calibration.get("schema") == "n3_q_compatibility_guardrail_calibration_report.v1", "V3 calibration report schema changed")
    _invalid(calibration.get("status") == "valid_completed" and calibration.get("result") == "VALID_COMPLETED", "V3 calibration report is incomplete")
    _invalid(calibration.get("classification") == "N3_Q_COMPATIBILITY_GUARDRAIL_CALIBRATION_READY", "V3 calibration report classification changed")
    _invalid(calibration.get("contract") == {"path": _relative(repo_root, repo_root / CALIBRATION_CONTRACT_PATH), "sha256": CALIBRATION_CONTRACT_SHA256}, "V3 calibration contract reference changed")
    _invalid(calibration.get("historical_results_unchanged") == {"n4_classification": "N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED", "gaussian_classification": "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED", "student_t_classification": "N3_FIXED_R_STUDENT_T_COPULA_NOT_SUPPORTED"}, "historical V2 result boundary changed in calibration")

    _invalid(evaluator.get("schema") == "n3_q_compatibility_evaluator_v3.v1", "V3 evaluator schema changed")
    _invalid(evaluator.get("status") == "valid_completed" and evaluator.get("result") == "READY", "V3 evaluator is not ready")
    _invalid(evaluator.get("evaluator_version") == "N3_Q_COMPATIBILITY_EVALUATOR_V3", "V3 evaluator version changed")
    _invalid(evaluator.get("contract") == {"path": _relative(repo_root, repo_root / CALIBRATION_CONTRACT_PATH), "sha256": CALIBRATION_CONTRACT_SHA256}, "V3 evaluator contract reference changed")
    primary = _mapping(evaluator.get("primary"), "V3 primary evaluator role")
    _invalid(primary.get("metric") == PRIMARY_METRIC, "V3 primary metric changed")
    _invalid(primary.get("role") == "HARD_PRIMARY_UNCHANGED", "V3 primary role changed")
    _invalid(primary.get("implementation_unchanged") is True and primary.get("sample_size_semantics_unchanged") is True, "V3 primary semantics changed")
    _invalid(evaluator.get("hard_guardrails") == EXPECTED_HARD_GUARDRAILS, "V3 HARD guardrail set changed")
    _invalid(evaluator.get("descriptive_only_guardrails") == list(DESCRIPTIVE_METRICS), "V3 descriptive guardrail set changed")
    formulas = _mapping(evaluator.get("metric_formulas"), "V3 metric formulas")
    _invalid(
        dict(formulas)
        == {
            "pair_metric": "v2_ceiling._pair_metrics_raw_q",
            "data_vs_model_aggregation": "gaussian._mean_metrics",
            "model_vs_model_reference_threshold": "gaussian._q95_metrics",
            "all_derived_formulas_unchanged": True,
            "implementation_authority": "scripts/research/execute_n3_development_bank_v2_empirical_ceiling.py",
        },
        "V3 metric formula authority changed",
    )
    sizes = _mapping(evaluator.get("sample_size_semantics"), "V3 sample-size semantics")
    _invalid(sizes.get("pseudo_data") == HELDOUT_COUNT_PER_FOLD, "V3 pseudo-DATA sample size changed")
    _invalid(sizes.get("data_vs_model_model_bank") == MODEL_SAMPLE_COUNT, "V3 model-bank sample size changed")
    _invalid(sizes.get("model_vs_model_reference") == [MODEL_SAMPLE_COUNT, MODEL_SAMPLE_COUNT], "V3 reference sample size changed")
    _invalid(sizes.get("compatibility_q") == COMPATIBILITY_Q and sizes.get("unchanged") is True, "V3 threshold semantics changed")
    heldout = _mapping(evaluator.get("development_heldout_authorities"), "V3 development/heldout authority")
    _invalid(
        heldout.get("fit_only_development_q") is True
        and heldout.get("heldout_q_metric_evaluation") is False
        and heldout.get("heldout_q_used_for_fit_or_sampling") is False
        and heldout.get("sealed_confirmation_input") is False,
        "V3 heldout or sealed boundary changed",
    )
    evaluator_ref = {
        "path": _relative(repo_root, repo_root / V3_EVALUATOR_PATH),
        "sha256": V3_EVALUATOR_SHA256,
        "role": "frozen V3 q compatibility evaluator",
        "schema": evaluator["schema"],
    }
    return {
        "contract": _json_ref(repo_root, CALIBRATION_CONTRACT_PATH, CALIBRATION_CONTRACT_SHA256, "V3 calibration contract", contract["schema"]),
        "calibration_report": _json_ref(repo_root, CALIBRATION_REPORT_PATH, CALIBRATION_REPORT_SHA256, "V3 calibration report", calibration["schema"]),
        "evaluator": evaluator_ref,
        "hard_guardrails": list(evaluator["hard_guardrails"]),
        "descriptive_only_guardrails": list(evaluator["descriptive_only_guardrails"]),
        "primary_metric": PRIMARY_METRIC,
        "compatibility_q": COMPATIBILITY_Q,
        "formulas": dict(formulas),
    }


def _validate_v2_authority(
    repo_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[str],
    dict[str, Any],
]:
    source, fold, rows, fold_ids, q_meta = v2_ceiling._validate_source_and_fold(repo_root)
    q_manifest = _read_json_exact(
        repo_root,
        V2_Q_MATERIALIZATION_MANIFEST,
        V2_Q_MATERIALIZATION_MANIFEST_SHA256,
        "V2 q materialization manifest",
    )
    ceiling = _read_json_exact(repo_root, V2_CEILING_REPORT_PATH, V2_CEILING_REPORT_SHA256, "V2 empirical ceiling report")
    _invalid(ceiling.get("schema") == "n3_development_bank_v2_empirical_ceiling_report.v1", "V2 ceiling schema changed")
    _invalid(ceiling.get("status") == "valid_completed" and ceiling.get("result") == "VALID_COMPLETED", "V2 ceiling report is incomplete")
    _invalid(ceiling.get("classification") == "N3_DEVELOPMENT_BANK_V2_VALIDATED", "V2 ceiling classification changed")
    aggregate = _mapping(ceiling.get("aggregate"), "V2 ceiling aggregate")
    _invalid(aggregate.get("compatible_fold_count") == FOLD_COUNT and aggregate.get("compatible_fold_pattern") == [True] * FOLD_COUNT, "V2 ceiling fold support changed")
    ceiling_folds = _sequence(_mapping(ceiling.get("empirical_ceiling"), "V2 empirical ceiling").get("fold_results"), "V2 ceiling folds")
    _invalid(len(ceiling_folds) == FOLD_COUNT, "V2 ceiling fold coverage changed")

    sealed = baseline._verify_historical_boundaries(repo_root)
    _invalid(sealed["sealed_confirmation"]["status"] == "RESERVED_NOT_MATERIALIZED", "sealed confirmation is materialized")
    _invalid(sealed["sealed_confirmation"]["sample_values_loaded"] is False, "sealed confirmation values were loaded")

    q, source_indices, labels, identities, loaded_identity = v2_ceiling._load_q_after_preflight(
        repo_root,
        {"source": source, "rows": rows},
    )
    _invalid(tuple(q.shape) == (POOL_COUNT, DIMENSION), "V2 q shape changed")
    _invalid(tuple(source_indices.shape) == (POOL_COUNT,), "V2 source-index shape changed")
    _invalid(tuple(labels.shape) == (POOL_COUNT,), "V2 labels shape changed")
    _invalid(len(identities) == POOL_COUNT, "V2 stable source-row identities are incomplete")
    return source, fold, q_meta, q, source_indices, labels, identities, {
        "q_manifest": q_manifest,
        "ceiling": ceiling,
        "ceiling_folds": {int(item["fold"]): item for item in ceiling_folds},
        "sealed": sealed,
        "loaded_identity": loaded_identity,
        "fold_ids": fold_ids,
    }


def _resolve_ref(repo_root: Path, ref: Mapping[str, Any], name: str) -> Path:
    raw_path = ref.get("path")
    _invalid(isinstance(raw_path, str) and raw_path, f"{name} path is missing")
    path = (repo_root / Path(raw_path)).resolve()
    _invalid(path.is_relative_to(repo_root.resolve()), f"{name} escapes the repository")
    _invalid(path.is_file(), f"{name} is missing: {path}")
    return path


def _lineage_has_artifact(lineage: Mapping[str, Any], ref: Mapping[str, Any], name: str) -> None:
    artifacts = _sequence(lineage.get("artifacts"), f"{name} artifacts")
    wanted = {key: ref.get(key) for key in ("path", "sha256", "role", "schema", "candidate_id", "fold") if ref.get(key) is not None}
    for item in artifacts:
        mapping = _mapping(item, f"{name} artifact")
        if all(mapping.get(key) == value for key, value in wanted.items()):
            return
    raise InvalidHarnessError(f"{name} does not bind frozen artifact: {wanted}")


def _validate_lineage(lineage: Mapping[str, Any], expected_schema: str, expected_task: str, name: str) -> None:
    _invalid(lineage.get("schema") == expected_schema, f"{name} schema changed")
    _invalid(lineage.get("task_id") == expected_task and lineage.get("status") == "valid_completed", f"{name} status changed")
    _invalid(lineage.get("all_artifacts_write_once") is True, f"{name} write-once boundary changed")
    _invalid(lineage.get("source_row_lineage_recorded") is True, f"{name} source lineage is incomplete")
    _invalid(lineage.get("consumer_report_is_not_self_referenced") is True, f"{name} self-reference boundary changed")
    sealed = _mapping(lineage.get("sealed_confirmation"), f"{name} sealed confirmation")
    _invalid(sealed.get("consumed") is False and sealed.get("values_loaded") is False and sealed.get("status") == "RESERVED_NOT_MATERIALIZED", f"{name} sealed boundary changed")


def _validate_historical_candidate_inputs(repo_root: Path) -> dict[str, Any]:
    registry = _read_json_exact(repo_root, N4_REGISTRY_PATH, N4_REGISTRY_SHA256, "N4 candidate registry")
    n4_report = _read_json_exact(repo_root, N4_REPORT_PATH, N4_REPORT_SHA256, "N4 comparison report")
    n4_fold = _read_json_exact(repo_root, N4_FOLD_COMPATIBILITY_PATH, N4_FOLD_COMPATIBILITY_SHA256, "N4 fold compatibility")
    n4_fit = _read_json_exact(repo_root, N4_FIT_MANIFEST_PATH, N4_FIT_MANIFEST_SHA256, "N4 fit manifest")
    n4_exposure = _read_json_exact(repo_root, N4_EXPOSURE_PATH, N4_EXPOSURE_SHA256, "N4 evaluator exposure ledger")
    n4_lineage = _read_json_exact(repo_root, N4_LINEAGE_PATH, N4_LINEAGE_SHA256, "N4 lineage manifest")
    student_contract = _read_json_exact(repo_root, STUDENT_CONTRACT_PATH, STUDENT_CONTRACT_SHA256, "Student-t producer contract")
    student_report = _read_json_exact(repo_root, STUDENT_REPORT_PATH, STUDENT_REPORT_SHA256, "Student-t producer report")
    student_fold = _read_json_exact(repo_root, STUDENT_FOLD_COMPATIBILITY_PATH, STUDENT_FOLD_COMPATIBILITY_SHA256, "Student-t fold compatibility")
    student_fit = _read_json_exact(repo_root, STUDENT_FIT_MANIFEST_PATH, STUDENT_FIT_MANIFEST_SHA256, "Student-t fit manifest")
    student_exposure = _read_json_exact(repo_root, STUDENT_EXPOSURE_PATH, STUDENT_EXPOSURE_SHA256, "Student-t evaluator exposure ledger")
    student_lineage = _read_json_exact(repo_root, STUDENT_LINEAGE_PATH, STUDENT_LINEAGE_SHA256, "Student-t lineage manifest")

    _invalid(registry.get("schema") == "n4_q_producer_candidate_registry.v1" and registry.get("status") == "FROZEN", "N4 registry is not frozen")
    registry_candidates = _sequence(registry.get("candidates"), "N4 registry candidates")
    _invalid(any(_mapping(item, "N4 registry candidate").get("candidate_id") == GAUSSIAN_ID for item in registry_candidates), "N4 Gaussian candidate is not in the frozen registry")
    _invalid(n4_report.get("schema") == "n4_q_producer_comparison_report.v1" and n4_report.get("status") == "valid_completed", "N4 report is incomplete")
    _invalid(n4_report.get("result") == "VALID_COMPLETED" and n4_report.get("classification") == "N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED", "historical N4 classification changed")
    n4_registry_ref = _mapping(n4_report.get("registry"), "N4 report registry reference")
    _invalid(n4_registry_ref.get("sha256") == N4_REGISTRY_SHA256, "N4 registry reference changed")
    _invalid(n4_fold.get("schema") == "n4_q_producer_fold_compatibility.v1" and n4_fold.get("status") == "valid_completed", "N4 fold compatibility is incomplete")
    _validate_lineage(n4_lineage, "n4_q_producer_lineage_manifest.v1", "N4_Q_PRODUCER_DEVELOPMENT_COMPARISON", "N4 lineage")
    _invalid(n4_fit.get("schema") == "n4_q_producer_fit_manifest.v1" and n4_fit.get("status") == "valid_completed", "N4 fit manifest is incomplete")
    _invalid(n4_exposure.get("schema") == "n4_q_producer_evaluator_exposure_ledger.v1" and n4_exposure.get("total_evaluator_calls") == 15, "N4 exposure ledger changed")

    _invalid(student_contract.get("schema") == "n3_empirical_marginal_fixed_r_student_t_copula_contract.v1", "Student-t contract schema changed")
    _invalid(student_contract.get("status") == "valid_completed" and student_contract.get("result") == "READY", "Student-t contract is not ready")
    student_family = _mapping(student_contract.get("family"), "Student-t family")
    _invalid(STUDENT_ID in student_family.get("candidate_ids", []) and student_family.get("family_id") == STUDENT_ID, "Student-t candidate identity changed")
    _invalid(student_report.get("schema") == "n3_empirical_marginal_fixed_r_student_t_copula_producer_report.v1", "Student-t report schema changed")
    _invalid(student_report.get("status") == "valid_completed" and student_report.get("result") == "VALID_COMPLETED", "Student-t report is incomplete")
    _invalid(student_report.get("classification") == "N3_FIXED_R_STUDENT_T_COPULA_NOT_SUPPORTED" and student_report.get("development_supported") is False, "historical Student-t classification changed")
    _invalid(student_fold.get("schema") == "n3_empirical_marginal_fixed_r_student_t_copula_fold_compatibility.v1" and student_fold.get("status") == "valid_completed", "Student-t fold compatibility is incomplete")
    _validate_lineage(student_lineage, "n3_empirical_marginal_fixed_r_student_t_copula_lineage_manifest.v1", "EXECUTE_N3_EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_V1", "Student-t lineage")
    _invalid(student_fit.get("schema") == "n3_empirical_marginal_fixed_r_student_t_copula_fit_manifest.v1" and student_fit.get("status") == "valid_completed", "Student-t fit manifest is incomplete")
    _invalid(student_exposure.get("schema") == "n3_empirical_marginal_fixed_r_student_t_copula_evaluator_exposure_ledger.v1" and student_exposure.get("total_evaluator_calls") == 5, "Student-t exposure ledger changed")

    n4_summary = next(
        _mapping(item, "N4 candidate summary")
        for item in _sequence(n4_report.get("candidate_summaries"), "N4 candidate summaries")
        if item.get("candidate_id") == GAUSSIAN_ID
    )
    _invalid(n4_summary.get("compatible_fold_count") == 4 and n4_summary.get("development_supported") is False, "historical N4 Gaussian result changed")
    failure_taxonomy = _sequence(n4_summary.get("failure_taxonomy"), "N4 Gaussian failure taxonomy")
    _invalid(len(failure_taxonomy) == 1, "historical N4 Gaussian failure topology changed")
    failure = _mapping(failure_taxonomy[0], "N4 Gaussian failure")
    _invalid(failure.get("fold") == 4 and failure.get("failed_metrics") == ["norm_energy_distance", "squared_norm_energy_distance"], "historical N4 Gaussian failure changed")

    v2_refs = {
        "source_manifest": _json_ref(repo_root, V2_SOURCE_MANIFEST, V2_SOURCE_MANIFEST_SHA256, "V2 development source manifest", "n3_development_bank_v2_source_manifest.v1"),
        "q_materialization_manifest": _json_ref(repo_root, V2_Q_MATERIALIZATION_MANIFEST, V2_Q_MATERIALIZATION_MANIFEST_SHA256, "V2 q materialization manifest", q_manifest_schema(repo_root)),
        "q_artifact": _json_ref(repo_root, V2_Q_ARTIFACT, V2_Q_ARTIFACT_SHA256, "V2 development q artifact", "n3_development_bank_v2_q.v1"),
        "fold_manifest": _json_ref(repo_root, V2_FOLD_MANIFEST, V2_FOLD_MANIFEST_SHA256, "V2 five-fold manifest", "n3_development_bank_v2_fold_manifest.v1"),
        "ceiling_report": _json_ref(repo_root, V2_CEILING_REPORT_PATH, V2_CEILING_REPORT_SHA256, "V2 empirical ceiling report", "n3_development_bank_v2_empirical_ceiling_report.v1"),
    }
    historical_refs = {
        "n4_registry": _json_ref(repo_root, N4_REGISTRY_PATH, N4_REGISTRY_SHA256, "N4 candidate registry", registry["schema"]),
        "n4_report": _json_ref(repo_root, N4_REPORT_PATH, N4_REPORT_SHA256, "N4 comparison report", n4_report["schema"]),
        "n4_fold_compatibility": _json_ref(repo_root, N4_FOLD_COMPATIBILITY_PATH, N4_FOLD_COMPATIBILITY_SHA256, "N4 fold compatibility", n4_fold["schema"]),
        "n4_fit_manifest": _json_ref(repo_root, N4_FIT_MANIFEST_PATH, N4_FIT_MANIFEST_SHA256, "N4 fit manifest", n4_fit["schema"]),
        "n4_exposure_ledger": _json_ref(repo_root, N4_EXPOSURE_PATH, N4_EXPOSURE_SHA256, "N4 evaluator exposure ledger", n4_exposure["schema"]),
        "n4_lineage": _json_ref(repo_root, N4_LINEAGE_PATH, N4_LINEAGE_SHA256, "N4 lineage manifest", n4_lineage["schema"]),
        "student_contract": _json_ref(repo_root, STUDENT_CONTRACT_PATH, STUDENT_CONTRACT_SHA256, "Student-t producer contract", student_contract["schema"]),
        "student_report": _json_ref(repo_root, STUDENT_REPORT_PATH, STUDENT_REPORT_SHA256, "Student-t producer report", student_report["schema"]),
        "student_fold_compatibility": _json_ref(repo_root, STUDENT_FOLD_COMPATIBILITY_PATH, STUDENT_FOLD_COMPATIBILITY_SHA256, "Student-t fold compatibility", student_fold["schema"]),
        "student_fit_manifest": _json_ref(repo_root, STUDENT_FIT_MANIFEST_PATH, STUDENT_FIT_MANIFEST_SHA256, "Student-t fit manifest", student_fit["schema"]),
        "student_exposure_ledger": _json_ref(repo_root, STUDENT_EXPOSURE_PATH, STUDENT_EXPOSURE_SHA256, "Student-t evaluator exposure ledger", student_exposure["schema"]),
        "student_lineage": _json_ref(repo_root, STUDENT_LINEAGE_PATH, STUDENT_LINEAGE_SHA256, "Student-t lineage manifest", student_lineage["schema"]),
    }
    return {
        "registry": registry,
        "n4_report": n4_report,
        "n4_fold": n4_fold,
        "n4_lineage": n4_lineage,
        "student_report": student_report,
        "student_fold": student_fold,
        "student_lineage": student_lineage,
        "v2_refs": v2_refs,
        "historical_refs": historical_refs,
        "lineages": {GAUSSIAN_ID: n4_lineage, STUDENT_ID: student_lineage},
    }


def q_manifest_schema(repo_root: Path) -> str:
    path = repo_root / V2_Q_MATERIALIZATION_MANIFEST
    value = json.loads(path.read_text(encoding="utf-8"))
    return str(_mapping(value, "V2 q materialization manifest").get("schema"))


def _validate_metric_contracts(repo_root: Path, n4_fold: Mapping[str, Any], v3: Mapping[str, Any]) -> None:
    _invalid(file_sha256(repo_root / METRIC_IMPLEMENTATION_PATH) == METRIC_IMPLEMENTATION_SHA256, "V2 raw-q metric implementation changed")
    _invalid(file_sha256(repo_root / GAUSSIAN_AGGREGATION_PATH) == GAUSSIAN_AGGREGATION_SHA256, "V2 aggregation implementation changed")
    config = _mapping(_mapping(n4_fold.get("configuration"), "N4 statistics configuration").get("statistics"), "N4 statistics")
    metric_contract = _mapping(config.get("metric_contract"), "N4 metric contract")
    _invalid(metric_contract.get("implementation_path") == _relative(repo_root, repo_root / METRIC_IMPLEMENTATION_PATH), "historical metric implementation path changed")
    _invalid(metric_contract.get("implementation_sha256") == METRIC_IMPLEMENTATION_SHA256, "historical metric implementation hash changed")
    _invalid(metric_contract.get("primary") == "biased empirical energy distance in raw q space", "historical primary ED semantics changed")
    _invalid(metric_contract.get("biased_empirical_ed") is True and metric_contract.get("diagonal_terms_included") is True, "historical ED formula flags changed")
    _invalid(metric_contract.get("per_coordinate_vector_role") == "diagnostic only; not a compatibility gate", "historical vector role changed")
    _invalid(metric_contract.get("scalar_metrics") == list(SCALAR_METRICS), "historical scalar metric order changed")
    _invalid(v3["formulas"]["pair_metric"] == "v2_ceiling._pair_metrics_raw_q", "V3 pair-metric formula changed")


def _validate_threshold(
    repo_root: Path,
    fold_record: Mapping[str, Any],
    ceiling_fold: Mapping[str, Any],
    fold: int,
) -> dict[str, float]:
    threshold = _mapping(fold_record.get("threshold"), f"historical threshold fold {fold}")
    threshold_source = _mapping(fold_record.get("threshold_source"), f"historical threshold source fold {fold}")
    _invalid(threshold_source.get("recomputed") is False, f"historical q95 was recomputed for fold {fold}")
    _invalid(threshold_source.get("q95") == COMPATIBILITY_Q, f"historical q95 changed for fold {fold}")
    _invalid(threshold_source.get("path") == _relative(repo_root, repo_root / V2_CEILING_REPORT_PATH), f"historical threshold path changed for fold {fold}")
    _invalid(threshold_source.get("sha256") == V2_CEILING_REPORT_SHA256, f"historical threshold authority changed for fold {fold}")
    reference = _mapping(ceiling_fold.get("model_vs_model_reference_q95"), f"V2 q95 fold {fold}")
    for name in SCALAR_METRICS:
        _invalid(float(threshold[name]) == float(reference[name]), f"q95 reference changed for fold {fold}: {name}")
    vector = threshold.get("per_coordinate_marginal_energy_distance")
    reference_vector = reference.get("per_coordinate_marginal_energy_distance")
    _invalid(isinstance(vector, list) and isinstance(reference_vector, list) and len(vector) == DIMENSION and len(reference_vector) == DIMENSION, f"q95 marginal vector changed for fold {fold}")
    _invalid([float(value) for value in vector] == [float(value) for value in reference_vector], f"q95 marginal vector values changed for fold {fold}")
    return {name: float(threshold[name]) for name in SCALAR_METRICS}


def _data_identity(
    q: torch.Tensor,
    source_indices: torch.Tensor,
    labels: torch.Tensor,
    identities: list[str],
    fold_ids: torch.Tensor,
    fold: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    heldout_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1).contiguous()
    fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1).contiguous()
    _invalid(int(heldout_positions.numel()) == HELDOUT_COUNT_PER_FOLD, f"held-out row count changed for fold {fold}")
    _invalid(int(fit_positions.numel()) == FIT_COUNT_PER_FOLD, f"fit row count changed for fold {fold}")
    _invalid(torch.equal(torch.sort(torch.cat([fit_positions, heldout_positions])).values, torch.arange(POOL_COUNT, dtype=torch.int64)), f"fit/held-out partition changed for fold {fold}")
    heldout_q = q.index_select(0, heldout_positions).contiguous()
    heldout_indices = source_indices.index_select(0, heldout_positions).contiguous()
    heldout_labels = labels.index_select(0, heldout_positions).contiguous()
    heldout_identities = [identities[int(index)] for index in heldout_positions.tolist()]
    identity = {
        "count": int(heldout_q.shape[0]),
        "dimension": int(heldout_q.shape[1]),
        "row_positions_sha256": tensor_sha256(heldout_positions),
        "source_indices_sha256": tensor_sha256(heldout_indices),
        "labels_sha256": tensor_sha256(heldout_labels),
        "stable_source_row_identity_sha256": canonical_sha256(heldout_identities),
        "q_tensor_sha256": tensor_sha256(heldout_q),
        "heldout_rows": int(heldout_q.shape[0]),
        "class_counts": torch.bincount(heldout_labels, minlength=CLASS_COUNT).tolist(),
    }
    _invalid(identity["class_counts"] == [HELDOUT_PER_CLASS_PER_FOLD] * CLASS_COUNT, f"held-out class balance changed for fold {fold}")
    return heldout_positions, fit_positions, identity


def _validate_data_against_history(
    identity: Mapping[str, Any],
    historical_data: Mapping[str, Any],
    fold: int,
) -> None:
    for key in ("q_tensor_sha256", "labels_sha256", "row_positions_sha256", "source_indices_sha256", "stable_source_row_identity_sha256"):
        _invalid(identity[key] == historical_data.get(key), f"historical DATA identity changed for fold {fold}: {key}")
    for key in ("count", "dimension"):
        if historical_data.get(key) is not None:
            _invalid(identity[key] == historical_data[key], f"historical DATA shape changed for fold {fold}: {key}")


def _validate_fit_artifact(
    repo_root: Path,
    ref: Mapping[str, Any],
    lineage: Mapping[str, Any],
    candidate_id: str,
    fold: int,
    q: torch.Tensor,
    source_indices: torch.Tensor,
    labels: torch.Tensor,
    identities: list[str],
    heldout_positions: torch.Tensor,
    fit_positions: torch.Tensor,
    expected_schema: str,
    expected_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    _invalid(ref.get("candidate_id") == candidate_id and ref.get("fold") == fold, f"fit artifact identity changed for {candidate_id} fold {fold}")
    _invalid(ref.get("role") == "fitted producer" and ref.get("schema") == expected_schema, f"fit artifact role/schema changed for {candidate_id} fold {fold}")
    path = _resolve_ref(repo_root, ref, f"{candidate_id} fold {fold} fit artifact")
    _invalid(file_sha256(path) == ref.get("sha256"), f"fit artifact hash changed for {candidate_id} fold {fold}")
    payload = _mapping(safe_torch_load(path, expected_sha256=ref["sha256"], map_location="cpu"), f"{candidate_id} fold {fold} fit artifact payload")
    _invalid(payload.get("schema") == expected_schema and payload.get("candidate_id") == candidate_id and payload.get("fold") == fold, f"fit payload identity changed for {candidate_id} fold {fold}")
    _invalid(payload.get("status") == "FIT_COMPLETED", f"fit artifact is incomplete for {candidate_id} fold {fold}")
    fit_source = _mapping(payload.get("fit_source"), f"{candidate_id} fold {fold} fit source")
    fit_q = q.index_select(0, fit_positions).contiguous()
    fit_source_indices = source_indices.index_select(0, fit_positions).contiguous()
    fit_labels = labels.index_select(0, fit_positions).contiguous()
    fit_identities = [identities[int(index)] for index in fit_positions.tolist()]
    checks = {
        "row_positions_sha256": tensor_sha256(fit_positions),
        "source_indices_sha256": tensor_sha256(fit_source_indices),
        "labels_sha256": tensor_sha256(fit_labels),
        "stable_source_row_identity_sha256": canonical_sha256(fit_identities),
        "fit_q_sha256": tensor_sha256(fit_q),
    }
    for key, value in checks.items():
        _invalid(fit_source.get(key) == value, f"fit DATA identity changed for {candidate_id} fold {fold}: {key}")
    if "heldout_row_positions_sha256" in fit_source:
        _invalid(fit_source["heldout_row_positions_sha256"] == tensor_sha256(heldout_positions), f"fit held-out boundary changed for {candidate_id} fold {fold}")
    if "heldout_source_indices_sha256" in fit_source:
        _invalid(fit_source["heldout_source_indices_sha256"] == tensor_sha256(source_indices.index_select(0, heldout_positions)), f"fit held-out source identity changed for {candidate_id} fold {fold}")
    _invalid(fit_source.get("heldout_excluded") is True, f"held-out rows entered fit for {candidate_id} fold {fold}")
    if expected_contract is not None:
        _invalid(payload.get("contract") == dict(expected_contract), f"Student-t fit contract binding changed for fold {fold}")
        _invalid(payload.get("heldout_values_used_for_fit") is False, f"Student-t fit used held-out q for fold {fold}")
        _invalid(payload.get("fit_q_values_serialized") is False and payload.get("heldout_q_values_serialized") is False, f"Student-t fit serialization boundary changed for fold {fold}")
    else:
        _invalid(payload.get("registry_sha256") == N4_REGISTRY_SHA256, f"N4 fit registry binding changed for fold {fold}")
    _lineage_has_artifact(lineage, ref, f"{candidate_id} lineage")
    return {
        "candidate_id": candidate_id,
        "fold": fold,
        "path": ref["path"],
        "sha256": ref["sha256"],
        "role": ref["role"],
        "schema": ref["schema"],
        "fit_q_sha256": checks["fit_q_sha256"],
        "heldout_excluded": True,
    }


def _validate_model_artifact(
    repo_root: Path,
    ref: Mapping[str, Any],
    lineage: Mapping[str, Any],
    candidate_id: str,
    fold: int,
    expected_schema: str,
    expected_contract: Mapping[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    _invalid(ref.get("candidate_id") == candidate_id and ref.get("fold") == fold, f"sample artifact identity changed for {candidate_id} fold {fold}")
    _invalid(ref.get("role") == "sampled model q" and ref.get("schema") == expected_schema, f"sample artifact role/schema changed for {candidate_id} fold {fold}")
    path = _resolve_ref(repo_root, ref, f"{candidate_id} fold {fold} sample artifact")
    _invalid(file_sha256(path) == ref.get("sha256"), f"sample artifact hash changed for {candidate_id} fold {fold}")
    payload = _mapping(safe_torch_load(path, expected_sha256=ref["sha256"], map_location="cpu"), f"{candidate_id} fold {fold} sample artifact payload")
    _invalid(payload.get("schema") == expected_schema and payload.get("candidate_id") == candidate_id and payload.get("fold") == fold, f"sample payload identity changed for {candidate_id} fold {fold}")
    model_q = payload.get("model_q")
    _invalid(torch.is_tensor(model_q) and model_q.dtype == torch.float64 and tuple(model_q.shape) == (MODEL_SAMPLE_COUNT, DIMENSION), f"sample bank shape/dtype changed for {candidate_id} fold {fold}")
    model_q = model_q.detach().cpu().contiguous()
    model_sha = tensor_sha256(model_q)
    _invalid(payload.get("model_q_shape") == [MODEL_SAMPLE_COUNT, DIMENSION] and payload.get("model_q_sha256") == model_sha, f"sample bank tensor identity changed for {candidate_id} fold {fold}")
    _invalid(bool(torch.isfinite(model_q).all()) and bool(model_q.ge(0.0).all()), f"sample bank values are invalid for {candidate_id} fold {fold}")
    sampling = _mapping(payload.get("sampling"), f"{candidate_id} fold {fold} sampling metadata")
    _invalid(sampling.get("role") == "data-vs-model" and sampling.get("fold") == fold and sampling.get("replicate") == 0, f"sample role/fold changed for {candidate_id} fold {fold}")
    _invalid(sampling.get("q_sha256") == model_sha and sampling.get("bank_size") == MODEL_SAMPLE_COUNT, f"sample q identity changed for {candidate_id} fold {fold}")
    _invalid(sampling.get("seed_retry") is False and sampling.get("heldout_seed_tuning") is False, f"sample seed policy changed for {candidate_id} fold {fold}")
    _invalid(payload.get("heldout_values_used_for_sampling") is False, f"held-out q entered sampling for {candidate_id} fold {fold}")
    _invalid(payload.get("best_of_n") is False and payload.get("resampling_after_evaluation") is False, f"sample selection/resampling policy changed for {candidate_id} fold {fold}")
    if "post_evaluation_resampling" in sampling:
        _invalid(sampling.get("post_evaluation_resampling") is False, f"post-evaluation resampling changed for {candidate_id} fold {fold}")
    if expected_contract is not None:
        _invalid(payload.get("contract") == dict(expected_contract), f"Student-t sample contract binding changed for fold {fold}")
    else:
        _invalid(payload.get("registry_sha256") == N4_REGISTRY_SHA256, f"N4 sample registry binding changed for fold {fold}")
    source_lineage = _mapping(payload.get("source_lineage"), f"{candidate_id} fold {fold} sample lineage")
    for key, path_value, sha_value in (
        ("q_artifact", V2_Q_ARTIFACT, V2_Q_ARTIFACT_SHA256),
        ("fold_manifest", V2_FOLD_MANIFEST, V2_FOLD_MANIFEST_SHA256),
    ):
        source_ref = _mapping(source_lineage.get(key), f"{candidate_id} fold {fold} sample {key}")
        _invalid(source_ref.get("path") == _relative(repo_root, repo_root / path_value) and source_ref.get("sha256") == sha_value, f"sample {key} binding changed for {candidate_id} fold {fold}")
    if "source_manifest" in source_lineage:
        source_ref = _mapping(source_lineage["source_manifest"], f"{candidate_id} fold {fold} sample source manifest")
        _invalid(source_ref.get("path") == _relative(repo_root, repo_root / V2_SOURCE_MANIFEST) and source_ref.get("sha256") == V2_SOURCE_MANIFEST_SHA256, f"sample source manifest binding changed for {candidate_id} fold {fold}")
    if "ceiling_report" in source_lineage:
        ceiling_ref = _mapping(source_lineage["ceiling_report"], f"{candidate_id} fold {fold} sample ceiling report")
        _invalid(ceiling_ref.get("path") == _relative(repo_root, repo_root / V2_CEILING_REPORT_PATH) and ceiling_ref.get("sha256") == V2_CEILING_REPORT_SHA256, f"sample ceiling binding changed for {candidate_id} fold {fold}")
    _lineage_has_artifact(lineage, ref, f"{candidate_id} lineage")
    return model_q, {
        "path": ref["path"],
        "sha256": ref["sha256"],
        "role": ref["role"],
        "schema": ref["schema"],
        "model_q_sha256": model_sha,
        "sample_seed": int(sampling["seed"]),
        "fold": fold,
        "replicate": 0,
        "new_sampling": False,
    }


def _historical_candidate_folds(
    fold_payload: Mapping[str, Any],
    candidate_id: str,
    name: str,
) -> list[Mapping[str, Any]]:
    candidate = next(
        (_mapping(item, f"{name} candidate") for item in _sequence(fold_payload.get("candidate_folds"), f"{name} candidate folds") if item.get("candidate_id") == candidate_id),
        None,
    )
    _invalid(candidate is not None, f"{name} candidate is missing")
    folds = _sequence(candidate.get("folds"), f"{name} folds")
    _invalid(len(folds) == FOLD_COUNT, f"{name} fold coverage is incomplete")
    _invalid([_mapping(item, f"{name} fold").get("fold") for item in folds] == list(range(FOLD_COUNT)), f"{name} fold order changed")
    return [_mapping(item, f"{name} fold") for item in folds]


def _replay_candidate(
    repo_root: Path,
    candidate_id: str,
    candidate_label: str,
    fold_records: list[Mapping[str, Any]],
    lineage: Mapping[str, Any],
    q: torch.Tensor,
    source_indices: torch.Tensor,
    labels: torch.Tensor,
    identities: list[str],
    fold_ids: torch.Tensor,
    ceiling_folds: Mapping[int, Mapping[str, Any]],
    expected_fit_schema: str,
    expected_sample_schema: str,
    expected_contract: Mapping[str, Any] | None,
    old_classification: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fold_results: list[dict[str, Any]] = []
    frozen_artifacts: list[dict[str, Any]] = []
    for fold_record in fold_records:
        fold = int(fold_record["fold"])
        heldout_positions, fit_positions, observed_data = _data_identity(q, source_indices, labels, identities, fold_ids, fold)
        _validate_data_against_history(observed_data, _mapping(fold_record.get("data"), f"{candidate_label} historical DATA fold {fold}"), fold)
        threshold = _validate_threshold(repo_root, fold_record, ceiling_folds[fold], fold)
        fit_ref = _mapping(_mapping(fold_record.get("fit"), f"{candidate_label} fit fold {fold}").get("fit_artifact"), f"{candidate_label} fit artifact fold {fold}")
        sample_ref = _mapping(_mapping(fold_record.get("model"), f"{candidate_label} model fold {fold}").get("sample_artifact"), f"{candidate_label} sample artifact fold {fold}")
        fit_identity = _validate_fit_artifact(
            repo_root,
            fit_ref,
            lineage,
            candidate_id,
            fold,
            q,
            source_indices,
            labels,
            identities,
            heldout_positions,
            fit_positions,
            expected_fit_schema,
            expected_contract,
        )
        model_q, model_identity = _validate_model_artifact(
            repo_root,
            sample_ref,
            lineage,
            candidate_id,
            fold,
            expected_sample_schema,
            expected_contract,
        )
        metrics = v2_ceiling._pair_metrics_raw_q(q.index_select(0, heldout_positions).contiguous(), model_q)
        historical_metrics = _mapping(fold_record.get("metrics"), f"{candidate_label} historical metrics fold {fold}")
        max_abs_delta = 0.0
        for name in SCALAR_METRICS:
            value = float(metrics[name])
            historical_value = float(historical_metrics[name])
            max_abs_delta = max(max_abs_delta, abs(value - historical_value))
            _invalid(value == historical_value, f"historical metric replay changed for {candidate_label} fold {fold}: {name}")
        _invalid(
            [float(value) for value in metrics["per_coordinate_marginal_energy_distance"]]
            == [float(value) for value in historical_metrics["per_coordinate_marginal_energy_distance"]],
            f"historical metric replay changed for {candidate_label} fold {fold}: marginal vector",
        )
        _invalid(float(metrics[PRIMARY_METRIC]) >= 0.0, f"primary metric is invalid for {candidate_label} fold {fold}")
        primary_pass = float(metrics[PRIMARY_METRIC]) <= float(threshold[PRIMARY_METRIC])
        descriptive = {
            name: {
                "metric": name,
                "role": "DESCRIPTIVE_ONLY",
                "data_value": float(metrics[name]),
                "historical_threshold": float(threshold[name]),
                "margin_threshold_minus_data": float(threshold[name]) - float(metrics[name]),
                "historical_threshold_pass": float(metrics[name]) <= float(threshold[name]),
            }
            for name in DESCRIPTIVE_METRICS
        }
        fold_results.append(
            {
                "fold": fold,
                "data_identity": observed_data,
                "model_identity": model_identity,
                "fit_identity": fit_identity,
                "q95_reference": {
                    "source_path": _relative(repo_root, repo_root / V2_CEILING_REPORT_PATH),
                    "source_sha256": V2_CEILING_REPORT_SHA256,
                    "q95": COMPATIBILITY_Q,
                    "recomputed": False,
                    "energy_distance": float(threshold[PRIMARY_METRIC]),
                },
                "primary": {
                    "metric": PRIMARY_METRIC,
                    "role": "HARD_PRIMARY_UNCHANGED",
                    "data_value": float(metrics[PRIMARY_METRIC]),
                    "q95_reference": float(threshold[PRIMARY_METRIC]),
                    "margin_threshold_minus_data": float(threshold[PRIMARY_METRIC]) - float(metrics[PRIMARY_METRIC]),
                    "pass": primary_pass,
                },
                "descriptive_only": descriptive,
                "historical_v2": {
                    "fold_pass": bool(fold_record.get("fold_pass")),
                    "metrics_replayed_exactly": True,
                    "max_abs_delta_from_historical": max_abs_delta,
                },
                "replay": {
                    "recomputed_from_exact_frozen_data_and_model_banks": True,
                    "new_sampling": False,
                    "refit": False,
                    "seed_changed": False,
                    "evaluator_calls": 1,
                },
            }
        )
        frozen_artifacts.extend([fit_identity, model_identity])
    primary_pattern = [bool(item["primary"]["pass"]) for item in fold_results]
    supported = all(primary_pattern)
    if candidate_id == GAUSSIAN_ID:
        classification = GAUSSIAN_SUPPORTED_CLASSIFICATION if supported else GAUSSIAN_NOT_SUPPORTED_CLASSIFICATION
    else:
        classification = STUDENT_SUPPORTED_CLASSIFICATION if supported else STUDENT_NOT_SUPPORTED_CLASSIFICATION
    outer_classification = OUTER_SUPPORTED_CLASSIFICATION if supported else OUTER_NOT_SUPPORTED_CLASSIFICATION
    candidate_result = {
        "candidate_label": candidate_label,
        "candidate_id": candidate_id,
        "classification": classification,
        "outer_classification": outer_classification,
        "development_supported": supported,
        "supported_fold_count": sum(primary_pattern),
        "fold_count": FOLD_COUNT,
        "fold_pattern": primary_pattern,
        "historical_v2_classification": old_classification,
        "historical_v2_unchanged": True,
        "fold_results": fold_results,
        "strict_support_rule": "all five folds pass primary energy_distance",
        "selection_performed": False,
    }
    return candidate_result, fold_results, frozen_artifacts


def _next_action(supported: list[str]) -> str:
    if len(supported) == 1:
        return "stop family search; next is producer-selection/freeze contract, no final fit/sealed"
    if len(supported) == 0:
        return "V3 concern closed; next plan one higher-expressivity producer"
    return "no posthoc select; next define a pre-frozen complexity rule"


def _authority_refs(repo_root: Path, v3: Mapping[str, Any], historical: Mapping[str, Any]) -> dict[str, Any]:
    refs = dict(historical["v2_refs"])
    refs.update(historical["historical_refs"])
    refs.update(
        {
            "v3_calibration_contract": v3["contract"],
            "v3_calibration_report": v3["calibration_report"],
            "v3_evaluator": v3["evaluator"],
        }
    )
    return refs


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    """Run the one-shot read-only V3 replay and publish its two evidence files."""
    _invalid(not output_dir.exists(), f"write-once replay output already exists: {output_dir}")
    git = _git_state(repo_root)
    _state, governance = _validate_current_governance(repo_root)
    v3 = _validate_v3_authority(repo_root)
    historical = _validate_historical_candidate_inputs(repo_root)
    _validate_metric_contracts(repo_root, historical["n4_fold"], v3)
    source, fold, q_meta, q, source_indices, labels, identities, v2 = _validate_v2_authority(repo_root)
    _invalid(q_meta.get("artifact_sha256") == V2_Q_ARTIFACT_SHA256, "V2 q artifact reference changed")
    _invalid(q_meta.get("tensor_sha256") == V2_Q_TENSOR_SHA256, "V2 q tensor reference changed")
    _invalid(v2["loaded_identity"]["q_r2_amplitude_sha256"] == V2_Q_TENSOR_SHA256, "V2 loaded q identity changed")
    fold_ids = v2["fold_ids"]

    n4_folds = _historical_candidate_folds(historical["n4_fold"], GAUSSIAN_ID, "N4 Gaussian")
    student_folds = _historical_candidate_folds(historical["student_fold"], STUDENT_ID, "Student-t")
    student_contract_ref = {"path": _relative(repo_root, repo_root / STUDENT_CONTRACT_PATH), "sha256": STUDENT_CONTRACT_SHA256}
    gaussian_result, _gaussian_fold_results, gaussian_artifacts = _replay_candidate(
        repo_root,
        GAUSSIAN_ID,
        "EMPIRICAL_GAUSSIAN_COPULA",
        n4_folds,
        historical["n4_lineage"],
        q,
        source_indices,
        labels,
        identities,
        fold_ids,
        v2["ceiling_folds"],
        "n4_q_producer_fitted_producer.v1",
        "n4_q_producer_sampled_model_q.v1",
        None,
        "N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED",
    )
    student_result, _student_fold_results, student_artifacts = _replay_candidate(
        repo_root,
        STUDENT_ID,
        "EMPIRICAL_MARGINAL_FIXED_R_STUDENT_T_COPULA_V1",
        student_folds,
        historical["student_lineage"],
        q,
        source_indices,
        labels,
        identities,
        fold_ids,
        v2["ceiling_folds"],
        "n3_empirical_marginal_fixed_r_student_t_copula_fitted_producer.v1",
        "n3_empirical_marginal_fixed_r_student_t_copula_sampled_model_q.v1",
        student_contract_ref,
        "N3_FIXED_R_STUDENT_T_COPULA_NOT_SUPPORTED",
    )
    candidate_results = [gaussian_result, student_result]
    supported_candidates = [item["candidate_id"] for item in candidate_results if item["development_supported"]]
    _invalid(len(supported_candidates) <= 1, "more than one V3 candidate was supported; posthoc selection is forbidden")
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    all_frozen_artifacts = gaussian_artifacts + student_artifacts
    lineage = {
        "schema": LINEAGE_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "created_at_utc": created_at,
        "git": {
            **git,
            "consumer_script": _relative(repo_root, Path(__file__).resolve()),
            "consumer_script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "consumer_report_is_not_self_referenced": True,
        "consumer_report_path": _relative(repo_root, repo_root / output_dir / REPORT_NAME),
        "authorities": list(_authority_refs(repo_root, v3, historical).values()),
        "frozen_candidate_artifacts": all_frozen_artifacts,
        "candidate_ids": [GAUSSIAN_ID, STUDENT_ID],
        "evaluation": {
            "reused_exact_historical_fit_artifacts": True,
            "reused_exact_historical_sample_banks": True,
            "new_sampling": False,
            "refitting": False,
            "seed_change": False,
            "new_model_banks": 0,
            "new_evaluator_calls": FOLD_COUNT * 2,
            "sealed_input": False,
        },
        "source_row_lineage_recorded": True,
        "all_artifacts_write_once": True,
        "sealed_confirmation": {
            "status": "RESERVED_NOT_MATERIALIZED",
            "consumed": False,
            "values_loaded": False,
            "input_to_this_consumer": False,
        },
    }
    report = {
        "schema": REPORT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "VALID_COMPLETED",
        "classification": "N3_Q_COMPATIBILITY_EVALUATOR_V3_FROZEN_CANDIDATE_REPLAY_COMPLETE",
        "created_at_utc": created_at,
        "git": {
            **git,
            "consumer_script": _relative(repo_root, Path(__file__).resolve()),
            "consumer_script_sha256": file_sha256(Path(__file__).resolve()),
        },
        "evaluator": {
            "version": "N3_Q_COMPATIBILITY_EVALUATOR_V3",
            "artifact": v3["evaluator"],
            "primary_metric": PRIMARY_METRIC,
            "hard_guardrails": list(v3["hard_guardrails"]),
            "descriptive_only_guardrails": list(v3["descriptive_only_guardrails"]),
            "metric_formulas": dict(v3["formulas"]),
            "primary_energy_distance_unchanged": True,
            "threshold_semantics_unchanged": True,
        },
        "input_authorities": _authority_refs(repo_root, v3, historical),
        "v2_development_authority": {
            "source": dict(v2["loaded_identity"]),
            "q_artifact": {
                "path": _relative(repo_root, repo_root / V2_Q_ARTIFACT),
                "sha256": V2_Q_ARTIFACT_SHA256,
                "tensor_sha256": V2_Q_TENSOR_SHA256,
            },
            "fold_manifest": {
                "path": _relative(repo_root, repo_root / V2_FOLD_MANIFEST),
                "sha256": V2_FOLD_MANIFEST_SHA256,
                "assignment_sha256": V2_ASSIGNMENT_SHA256,
            },
            "ceiling_report": {
                "path": _relative(repo_root, repo_root / V2_CEILING_REPORT_PATH),
                "sha256": V2_CEILING_REPORT_SHA256,
                "classification": "N3_DEVELOPMENT_BANK_V2_VALIDATED",
            },
        },
        "candidates": candidate_results,
        "development_supported_candidates": supported_candidates,
        "development_supported_candidate_count": len(supported_candidates),
        "selection": {
            "development_supported_candidate": supported_candidates[0] if len(supported_candidates) == 1 else None,
            "selected_producer": None,
            "posthoc_selection": False,
            "producer_freeze_performed": False,
            "next_action": _next_action(supported_candidates),
        },
        "historical_immutability": {
            "n4_classification_unchanged": historical["n4_report"]["classification"] == "N4_Q_PRODUCER_COMPARISON_NO_CANDIDATE_SUPPORTED",
            "gaussian_v2_fold_pattern_unchanged": [bool(item["fold_pass"]) for item in n4_folds] == [True, True, True, True, False],
            "student_v2_fold_pattern_unchanged": [bool(item["fold_pass"]) for item in student_folds] == [True, True, False, True, True],
            "historical_v2_reports_not_modified": True,
        },
        "zero_new_sampling_confirmation": {
            "new_sampling": False,
            "new_fit": False,
            "new_model_banks": 0,
            "rng_created_by_this_consumer": False,
            "historical_seeds_reused_as_identity_only": True,
            "model_q_banks_reused_without_resampling": True,
        },
        "governance": {
            "selected_producer": None,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "sealed_confirmation_consumed": False,
            "project_flags_unchanged": True,
            "preflight": governance,
        },
        "claim_boundary": {
            "claim": "frozen candidate samples satisfy calibrated V3 raw-q development compatibility on the V2 bank",
            "not_stationarity_or_ood": True,
            "not_heldout_confirmation": True,
            "not_complex_ols_phase_semantic_image_or_optical": True,
            "not_training_or_generation": True,
        },
        "limitations": [
            "DATA-vs-MODEL and MODEL-vs-MODEL sample-size structures differ under biased empirical energy distance.",
            "The empirical q95 remains a development threshold, not a calibrated 95% confidence bound.",
            "Secondary metrics are reported against historical thresholds but cannot veto V3 support.",
        ],
        "sealed_confirmation": {
            "status": "RESERVED_NOT_MATERIALIZED",
            "consumed": False,
            "values_loaded": False,
            "input_to_this_consumer": False,
        },
        "not_executed": [
            "producer refitting",
            "new q sampling or resampling",
            "heldout tuning",
            "sealed confirmation loading",
            "N4 selector or producer freeze",
            "CUDA or training",
            "optical, phase, semantic, or image evaluation",
            "final fit or promotion",
        ],
        "lineage": {
            "path": _relative(repo_root, repo_root / output_dir / LINEAGE_NAME),
            "sha256": None,
            "schema": LINEAGE_SCHEMA,
        },
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        lineage_path = temporary / LINEAGE_NAME
        atomic_write_new_json(lineage_path, lineage, artifact_kind="N3 V3 frozen candidate replay lineage")
        lineage_sha256 = file_sha256(lineage_path)
        report["lineage"]["sha256"] = lineage_sha256
        atomic_write_new_json(temporary / REPORT_NAME, report, artifact_kind="N3 V3 frozen candidate replay report")
        _invalid(not output_dir.exists(), f"replay output appeared during execution: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    report_path = output_dir / REPORT_NAME
    lineage_final = output_dir / LINEAGE_NAME
    _invalid(file_sha256(lineage_final) == lineage_sha256, "published replay lineage hash changed")
    _invalid(file_sha256(report_path) != "", "published replay report is empty")
    return {
        "status": "valid_completed",
        "result": "VALID_COMPLETED",
        "classification": report["classification"],
        "development_supported_candidates": supported_candidates,
        "selected_producer": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_consumed": False,
        "report": _relative(repo_root, report_path),
        "report_sha256": file_sha256(report_path),
        "lineage": _relative(repo_root, lineage_final),
        "lineage_sha256": lineage_sha256,
        "v3_evaluator_sha256": V3_EVALUATOR_SHA256,
        "new_sampling": False,
        "new_fit": False,
        "next_action": report["selection"]["next_action"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    try:
        result = execute(repo_root, output_dir)
    except InvalidHarnessError as exc:
        print(json.dumps({"status": "blocked", "result": "BLOCKED", "task_id": TASK_ID, "reason": str(exc)}, indent=2, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
