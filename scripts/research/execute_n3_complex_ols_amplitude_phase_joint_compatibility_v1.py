"""Execute the frozen N3 complete-complex-OLS compatibility contract once.

The consumer first materializes the exact V2 development rows at the frozen
R3 complex OLS boundary, freezes that artifact and its row manifest, and only
then evaluates the complete-row bootstrap control.  The independent observed
phase-inheritance candidate is evaluated only when all five control folds pass.
This is a CPU-only development measurement: it does not read consumed sealed
values, alter the selected q producer, enter a decoder/evaluator, or train.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import tempfile
import time
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
    define_n3_complex_ols_amplitude_phase_joint_compatibility_contract_v1 as contract_definition,
    execute_n3_empirical_marginal_gaussian_copula_producer_v1 as copula,
    materialize_n3_development_bank_v2_q as q_materialization,
    replay_scaled_chen_repaired_coordinate_mra as replay,
    validate_n3_amplitude_producer_baseline as metric_consumer,
)


TASK_ID = "EXECUTE_N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_V1"
GATE = TASK_ID
REPORT_SCHEMA = "n3_complex_ols_amplitude_phase_joint_compatibility_report.v1"
LINEAGE_SCHEMA = "n3_complex_ols_amplitude_phase_joint_compatibility_lineage.v1"
MATERIALIZATION_SCHEMA = "n3_development_bank_v2_complex_ols.v1"
MATERIALIZATION_MANIFEST_SCHEMA = "n3_development_bank_v2_complex_ols_materialization_manifest.v1"
METHOD_ARTIFACT_SCHEMA = "n3_complex_ols_joint_compatibility_generated_banks.v1"

EXPECTED_BRANCH = contract_definition.EXPECTED_BRANCH
EXPECTED_STARTING_HEAD = contract_definition.EXPECTED_STARTING_HEAD
CONTRACT_PATH = contract_definition.OUTPUT_DIR / contract_definition.CONTRACT_NAME
EXPECTED_CONTRACT_SHA256 = "52e07263343d765db13022fbcf56f74e7431a9dced16f8bfa31ca07ecc6bd726"

OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "n3_complex_ols_amplitude_phase_joint_compatibility_v1_attempt_01"
)
MATERIALIZATION_NAME = "n3_development_bank_v2_complex_ols.v1.pt"
MATERIALIZATION_MANIFEST_NAME = "n3_development_bank_v2_complex_ols_materialization_manifest.v1.json"
CONTROL_ARTIFACT_NAME = "n3_complex_ols_joint_control_banks_and_calibration.v1.pt"
CANDIDATE_ARTIFACT_NAME = "n3_complex_ols_phase_inheritance_banks_and_calibration.v1.pt"
LINEAGE_NAME = "n3_complex_ols_amplitude_phase_joint_compatibility_lineage.v1.json"
REPORT_NAME = "n3_complex_ols_amplitude_phase_joint_compatibility_report.v1.json"

CLASS_COUNT = contract_definition.CLASS_COUNT
DIMENSION = contract_definition.DIMENSION
POOL_COUNT = contract_definition.DEVELOPMENT_ROWS
FOLD_COUNT = contract_definition.FOLD_COUNT
FIT_COUNT = contract_definition.FIT_ROWS_PER_FOLD
HELDOUT_COUNT = contract_definition.HELDOUT_ROWS_PER_FOLD
MODEL_BANK_COUNT = contract_definition.MODEL_BANK_COUNT
MODEL_BANK_SIZE = contract_definition.MODEL_BANK_SIZE
CALIBRATION_TRIAL_COUNT = contract_definition.CALIBRATION_TRIAL_COUNT
PER_FOLD_ALPHA = contract_definition.PER_FOLD_ALPHA
FAMILY_ALPHA = contract_definition.FAMILY_ALPHA
PRIMARY_METRIC = contract_definition.PRIMARY_METRIC
RNG_NAMESPACE = contract_definition.RNG_NAMESPACE

CONTROL_ID = "FULL_COMPLEX_ROW_EMPIRICAL_BOOTSTRAP_CONTROL"
CANDIDATE_ID = "OBSERVED_PHASE_INDEPENDENT_INHERITANCE_V1"
CLASSIFICATION_STABILITY = "N3_COMPLEX_OLS_JOINT_DEVELOPMENT_STABILITY_NOT_ESTABLISHED"
CLASSIFICATION_SUPPORTED = "N3_COMPLEX_OLS_INDEPENDENT_PHASE_INHERITANCE_SUPPORTED"
CLASSIFICATION_NOT_SUPPORTED = "N3_COMPLEX_OLS_INDEPENDENT_PHASE_INHERITANCE_NOT_SUPPORTED"

NEXT_STABILITY = "ADDRESS_COMPLEX_JOINT_DEVELOPMENT_STABILITY_AND_AUTHORITY"
NEXT_SUPPORTED = "STOP_PHASE_MODEL_RESEARCH_AND_PLAN_DOWNSTREAM_OLS_SEMANTIC_CONSUMER_BOUNDARY"
NEXT_NOT_SUPPORTED = "DEFINE_ONE_MINIMAL_Q_CONDITIONED_PHASE_COUPLING_MODEL_CONTRACT"

DIAGNOSTIC_SCALARS = (
    "amplitude_q_discrepancy",
    "phase_circular_discrepancy",
    "covariance_relative_frobenius",
    "norm_energy_distance",
    "squared_norm_energy_distance",
    "marginal_q_mean_l2",
    "marginal_q_std_l2",
    "marginal_q_energy_distance_mean",
    "marginal_q_energy_distance_max",
)


class InvalidHarnessError(RuntimeError):
    """A frozen authority or execution invariant is invalid."""


class BlockedExecution(RuntimeError):
    """A required authority could not be resolved without substitution."""


def _require(condition: bool, message: str, *, blocked: bool = False) -> None:
    if condition:
        return
    error = BlockedExecution if blocked else InvalidHarnessError
    raise error(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _git_state(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status_result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    status = status_result.stdout.splitlines()
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
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
    _require(target.is_file(), f"{name} is missing: {target}", blocked=True)
    _require(file_sha256(target) == expected_sha256, f"{name} hash changed: {target}", blocked=True)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidHarnessError(f"{name} cannot be read as JSON: {target}") from exc
    return dict(_mapping(payload, name))


def _ref(
    repo_root: Path,
    path: Path,
    *,
    role: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{role} is missing: {target}", blocked=True)
    actual = file_sha256(target)
    _require(expected_sha256 is None or actual == expected_sha256, f"{role} hash changed: {target}", blocked=True)
    return {"path": _relative(repo_root, target), "sha256": actual, "role": role}


def _validate_contract(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _read_json_exact(repo_root, CONTRACT_PATH, EXPECTED_CONTRACT_SHA256, "primary complex-OLS contract")
    _require(contract.get("schema") == contract_definition.CONTRACT_SCHEMA, "contract schema changed", blocked=True)
    _require(contract.get("task_id") == contract_definition.TASK_ID, "contract task identity changed", blocked=True)
    _require(contract.get("status") == "valid_completed" and contract.get("result") == "READY", "contract is not READY", blocked=True)
    _require(contract.get("classification") == contract_definition.READY_CLASSIFICATION, "contract readiness classification changed", blocked=True)
    _require(contract.get("next_action") == TASK_ID, "contract next action changed", blocked=True)
    _require(contract.get("definition_only") is True, "contract is not definition-only authority", blocked=True)

    representation = _mapping(contract.get("representation"), "contract representation")
    _require(representation.get("dimension") == DIMENSION, "complex dimension changed", blocked=True)
    q_to_amplitude = _mapping(representation.get("q_to_amplitude"), "q-to-amplitude mapping")
    _require(
        q_to_amplitude.get("q_coordinate") == "R2_AMPLITUDE"
        and q_to_amplitude.get("A_j(q)") == "q_j"
        and q_to_amplitude.get("additional_amplitude_transform") == "none"
        and q_to_amplitude.get("mapping_unique") is True,
        "A(q)=q contract changed",
        blocked=True,
    )
    complex_contract = _mapping(representation.get("complex_ols"), "complex OLS representation")
    _require(
        complex_contract.get("coordinate") == "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX"
        and complex_contract.get("formula") == "z_j=c_j/sqrt(s_j)"
        and complex_contract.get("input_dtype") == "torch.complex64"
        and complex_contract.get("canonical_dtype") == "torch.complex128"
        and complex_contract.get("shape") == ["N", DIMENSION]
        and complex_contract.get("physical_coordinate_metadata_required") is True,
        "complex OLS coordinate contract changed",
        blocked=True,
    )
    phase_contract = _mapping(contract.get("phase_semantics"), "phase semantics")
    _require(
        phase_contract.get("representation") == "phi_j=atan2(Im(z_j^canonical), Re(z_j^canonical))"
        and phase_contract.get("wrap_interval") == "[-pi, pi]"
        and phase_contract.get("unwrap") is False
        and phase_contract.get("vector_unit") == "one complete 100-coordinate vector per OLS row"
        and phase_contract.get("phase_donor_unit") == "complete vector; per-coordinate donor mixing forbidden",
        "phase representation contract changed",
        blocked=True,
    )
    zero_policy = str(phase_contract.get("zero_policy"))
    near_zero_policy = str(phase_contract.get("near_zero_policy"))
    _require("any exact-zero" in zero_policy and "INVALID_HARNESS" in zero_policy, "zero policy changed", blocked=True)
    _require("without threshold" in near_zero_policy and "without" in near_zero_policy, "near-zero policy changed", blocked=True)

    global_phase = _mapping(representation.get("global_phase"), "global phase contract")
    _require(
        global_phase.get("existing_authority") == "GLOBAL_PHASE_GAUGE_CONFIRMED"
        and global_phase.get("new_physical_gauge_invented") is False
        and global_phase.get("common_phase_invariance_preserved") is True,
        "global phase authority changed",
        blocked=True,
    )

    control = _mapping(contract.get("control"), "control contract")
    _require(
        control.get("id") == CONTROL_ID
        and control.get("selectable") is False
        and control.get("fit_rows") == FIT_COUNT
        and control.get("sampling_unit") == "complete observed fitting R3_COMPLEX row"
        and control.get("sampling") == "with replacement; preserve amplitude-phase pairing and all 100 coordinates"
        and control.get("heldout_rows_used_for_sampling") is False,
        "control contract changed",
        blocked=True,
    )
    candidate = _mapping(contract.get("candidate"), "candidate contract")
    candidate_per_fold = _mapping(candidate.get("per_fold"), "candidate per-fold contract")
    _require(
        candidate.get("id") == CANDIDATE_ID
        and candidate.get("selectable") is True
        and candidate_per_fold.get("q_dependent_phase_selection") is False
        and candidate_per_fold.get("conditional_phase_model") is False,
        "candidate contract changed",
        blocked=True,
    )
    isolation = _mapping(contract.get("development_isolation"), "development isolation")
    for key, expected in {
        "fold_count": FOLD_COUNT,
        "fit_rows_per_fold": FIT_COUNT,
        "heldout_rows_per_fold": HELDOUT_COUNT,
        "heldout_phase_donors": False,
        "cross_fold_phase_donors": False,
        "labels_as_inputs": False,
        "runtime_reselection": False,
        "runtime_resplit": False,
    }.items():
        _require(isolation.get(key) == expected, f"development isolation changed: {key}", blocked=True)

    calibration = _mapping(contract.get("model_self_calibration"), "model-self calibration")
    for key, expected in {
        "required_before_heldout": True,
        "per_fold_and_method": True,
        "pseudo_data_rows": HELDOUT_COUNT,
        "model_bank_count": MODEL_BANK_COUNT,
        "model_bank_size": MODEL_BANK_SIZE,
        "trials": CALIBRATION_TRIAL_COUNT,
    }.items():
        _require(calibration.get(key) == expected, f"model-self calibration changed: {key}", blocked=True)
    decision = _mapping(contract.get("decision"), "decision contract")
    _require(
        decision.get("family_alpha") == FAMILY_ALPHA
        and decision.get("per_fold_alpha") == PER_FOLD_ALPHA
        and decision.get("B") == CALIBRATION_TRIAL_COUNT
        and decision.get("fold_pass") == "p_MC>=0.01"
        and decision.get("aggregate_pass") == "all five folds pass (5/5)"
        and _mapping(decision.get("outcomes"), "decision outcomes").get("control_below_5_of_5") == CLASSIFICATION_STABILITY
        and _mapping(decision.get("outcomes"), "decision outcomes").get("control_5_of_5_candidate_5_of_5") == CLASSIFICATION_SUPPORTED
        and _mapping(decision.get("outcomes"), "decision outcomes").get("control_5_of_5_candidate_below_5_of_5") == CLASSIFICATION_NOT_SUPPORTED,
        "decision mapping changed",
        blocked=True,
    )
    primary = _mapping(_mapping(contract.get("evaluation"), "evaluation contract").get("primary"), "primary metric")
    _require(primary.get("name") == PRIMARY_METRIC and primary.get("role") == "HARD_PRIMARY", "primary metric contract changed", blocked=True)
    _require(_mapping(contract.get("forbidden"), "forbidden operations").get("cuda") is True, "CPU-only boundary changed", blocked=True)
    _require(_mapping(contract.get("forbidden"), "forbidden operations").get("training") is True, "training boundary changed", blocked=True)

    authorities = _mapping(contract.get("authorities"), "contract authorities")
    for key in ("v2", "project_complex_ols", "selected_producer_and_prior_confirmation", "metric_and_evaluator"):
        _require(key in authorities, f"contract authority is missing: {key}", blocked=True)
    return contract, dict(authorities)


def _validate_current_governance(repo_root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    path = repo_root / contract_definition.GOVERNANCE_STATE_PATH
    _require(path.is_file(), "current research governance state is missing", blocked=True)
    before_sha = file_sha256(path)
    state = dict(_mapping(json.loads(path.read_text(encoding="utf-8")), "current governance state"))
    _require(
        state.get("active_objective") == "N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_CONTRACT_V1_READY"
        and state.get("authorization_tier") == "R3"
        and state.get("hard_blocker") == "N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_EXECUTION_REQUIRED",
        "current governance is not at the execution gate",
        blocked=True,
    )
    aliases = _mapping(state.get("evidence_aliases"), "governance evidence aliases")
    alias = _mapping(aliases.get("N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_CONTRACT_V1"), "complex-OLS contract alias")
    for key, expected in {
        "sha256": EXPECTED_CONTRACT_SHA256,
        "classification": contract_definition.READY_CLASSIFICATION,
        "status": "valid_completed",
        "result": "READY",
        "selected_producer": "EMPIRICAL_GAUSSIAN_COPULA",
        "sealed_confirmation_consumed": True,
        "q_values_loaded": False,
        "complex_ols_values_loaded": False,
        "scientific_evaluation_executed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items():
        _require(alias.get(key) == expected, f"current contract alias changed: {key}", blocked=True)
    prior_alias = _mapping(aliases.get("N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_V1"), "prior sealed confirmation alias")
    _require(
        prior_alias.get("classification") == "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED"
        and prior_alias.get("sealed_confirmation_consumed") is True
        and prior_alias.get("training_authorized") is False
        and prior_alias.get("formal_promotion") is False
        and prior_alias.get("generation_readiness") == "not_ready",
        "prior sealed confirmation boundary changed",
        blocked=True,
    )
    for status_path in (contract_definition.CURRENT_STATUS_PATH, contract_definition.CURRENT_HANDOFF_PATH):
        text = (repo_root / status_path).read_text(encoding="utf-8")
        _require(TASK_ID in text, f"{status_path} does not name the execution action", blocked=True)
        _require("training_authorized=false" in text and "generation_readiness=not_ready" in text, f"{status_path} opened downstream governance", blocked=True)
    return {
        "path": _relative(repo_root, path),
        "sha256_before": before_sha,
        "active_objective": state["active_objective"],
        "authorization_tier": state["authorization_tier"],
        "hard_blocker": state["hard_blocker"],
        "selected_producer": alias["selected_producer"],
        "sealed_confirmation_consumed": True,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }


def _validate_frozen_authorities(repo_root: Path, contract: Mapping[str, Any], authorities: Mapping[str, Any]) -> dict[str, Any]:
    """Run the definition consumer's read-only authority validators.

    These validators inspect manifests and hashes only.  The consumed sealed
    content is never loaded here; its lineage/reservation metadata is used only
    to prove that it remains excluded.
    """

    try:
        runtime_v2 = contract_definition._validate_v2_authority(repo_root)
        runtime_ols = contract_definition._validate_complex_ols_authority(repo_root)
        runtime_prior = contract_definition._validate_prior_provenance(repo_root)
        runtime_metrics = contract_definition._validate_metric_authority(repo_root)
    except contract_definition.InvalidHarnessError as exc:
        raise BlockedExecution(str(exc)) from exc
    _require(dict(authorities["v2"]) == runtime_v2, "V2 authority differs from the frozen contract", blocked=True)
    _require(dict(authorities["project_complex_ols"]) == runtime_ols, "PROJECT_COMPLEX_OLS authority differs from the frozen contract", blocked=True)
    _require(dict(authorities["selected_producer_and_prior_confirmation"]) == runtime_prior, "selected-producer/sealed provenance differs from the frozen contract", blocked=True)
    _require(dict(authorities["metric_and_evaluator"]) == runtime_metrics, "metric/evaluator authority differs from the frozen contract", blocked=True)

    definition_ref = _mapping(authorities.get("definition_implementation"), "definition implementation reference")
    definition_path = repo_root / Path(str(definition_ref["path"]))
    _require(file_sha256(definition_path) == definition_ref.get("sha256"), "contract definition implementation changed", blocked=True)
    metric_ref = _mapping(_mapping(authorities["metric_and_evaluator"], "metric authority").get("energy_distance"), "energy-distance authority")
    metric_path = repo_root / Path(str(metric_ref["path"]))
    _require(file_sha256(metric_path) == metric_ref.get("sha256"), "energy-distance implementation changed", blocked=True)
    evaluator_ref = _mapping(_mapping(authorities["metric_and_evaluator"], "metric authority").get("evaluator"), "evaluator authority")
    evaluator_path = repo_root / Path(str(evaluator_ref["path"]))
    _require(file_sha256(evaluator_path) == evaluator_ref.get("sha256"), "evaluator implementation changed", blocked=True)
    aggregation_ref = _mapping(_mapping(authorities["metric_and_evaluator"], "metric authority").get("aggregation"), "aggregation authority")
    aggregation_path = repo_root / Path(str(aggregation_ref["path"]))
    _require(file_sha256(aggregation_path) == aggregation_ref.get("sha256"), "Gaussian producer implementation changed", blocked=True)
    return {
        "v2": runtime_v2,
        "project_complex_ols": runtime_ols,
        "selected_producer_and_prior_confirmation": runtime_prior,
        "metric_and_evaluator": runtime_metrics,
    }


def _load_v2_rows_and_q(repo_root: Path) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, list[str], torch.Tensor, dict[str, Any]]:
    source_path = repo_root / contract_definition.V2_SOURCE_MANIFEST_PATH
    source = _read_json_exact(repo_root, contract_definition.V2_SOURCE_MANIFEST_PATH, contract_definition.V2_SOURCE_MANIFEST_SHA256, "V2 source manifest")
    rows = source.get("rows")
    _require(isinstance(rows, list) and len(rows) == POOL_COUNT, "V2 source rows changed", blocked=True)
    indices = torch.tensor([int(row["source_index"]) for row in rows], dtype=torch.int64)
    labels = torch.tensor([int(row["label"]) for row in rows], dtype=torch.int64)
    identities = [str(row["stable_source_row_identity"]) for row in rows]
    _require(tensor_sha256(indices) == contract_definition.V2_SOURCE_INDICES_SHA256, "V2 source index identity changed", blocked=True)
    _require(tensor_sha256(labels) == contract_definition.V2_LABELS_SHA256, "V2 label identity changed", blocked=True)
    _require(canonical_sha256(identities) == contract_definition.V2_SOURCE_ROW_IDENTITY_SHA256, "V2 source-row identity changed", blocked=True)
    _require(len(set(indices.tolist())) == POOL_COUNT and len(set(identities)) == POOL_COUNT, "V2 source rows are not unique", blocked=True)

    fold_manifest = _read_json_exact(repo_root, contract_definition.V2_FOLD_MANIFEST_PATH, contract_definition.V2_FOLD_MANIFEST_SHA256, "V2 five-fold manifest")
    fold_rows = fold_manifest.get("rows")
    _require(isinstance(fold_rows, list) and len(fold_rows) == POOL_COUNT, "V2 fold rows changed", blocked=True)
    _require(canonical_sha256(fold_rows) == contract_definition.V2_FOLD_ROWS_SHA256, "V2 fold-row identity changed", blocked=True)
    fold_ids = torch.empty((POOL_COUNT,), dtype=torch.int64)
    fold_counts = [[0] * CLASS_COUNT for _ in range(FOLD_COUNT)]
    for position, fold_value in enumerate(fold_rows):
        fold_row = _mapping(fold_value, f"V2 fold row {position}")
        source_row = _mapping(rows[position], f"V2 source row {position}")
        for key in ("bank_row_position", "label", "source_index", "stable_source_row_identity"):
            _require(fold_row.get(key) == source_row.get(key), f"V2 fold/source correspondence changed: {key}", blocked=True)
        fold = fold_row.get("heldout_fold")
        _require(isinstance(fold, int) and 0 <= fold < FOLD_COUNT, f"V2 fold id is invalid at row {position}", blocked=True)
        fold_ids[position] = fold
        fold_counts[fold][int(labels[position])] += 1
    _require(all(counts == [contract_definition.HELDOUT_ROWS_PER_CLASS_PER_FOLD] * CLASS_COUNT for counts in fold_counts), "V2 fold class balance changed", blocked=True)
    protocol = _mapping(fold_manifest.get("protocol"), "V2 fold protocol")
    _require(protocol.get("assignment_sha256") == contract_definition.V2_ASSIGNMENT_SHA256, "V2 fold assignment changed", blocked=True)
    _require(protocol.get("q_values_used") is False and protocol.get("row_to_fold_frozen_before_validation") is True, "V2 fold construction boundary changed", blocked=True)

    q_path = repo_root / contract_definition.V2_Q_ARTIFACT_PATH
    _require(file_sha256(q_path) == contract_definition.V2_Q_ARTIFACT_SHA256, "V2 q artifact bytes changed", blocked=True)
    payload = _mapping(safe_torch_load(q_path, expected_sha256=contract_definition.V2_Q_ARTIFACT_SHA256, map_location="cpu"), "V2 q artifact")
    _require(payload.get("schema") == "n3_development_bank_v2_q.v1" and payload.get("status") == "VALID_COMPLETED", "V2 q artifact is incomplete", blocked=True)
    _require(payload.get("q_coordinate") == "R2_AMPLITUDE" and payload.get("q_definition") == "q=abs(c/sqrt(s))=abs(R3_COMPLEX)", "V2 q semantics changed", blocked=True)
    q = payload.get("q")
    q_indices = payload.get("source_indices")
    q_labels = payload.get("labels")
    q_identity = payload.get("source_row_identity")
    _require(torch.is_tensor(q) and q.dtype == torch.float64 and tuple(q.shape) == (POOL_COUNT, DIMENSION), "V2 q tensor shape/dtype changed", blocked=True)
    _require(torch.is_tensor(q_indices) and q_indices.dtype == torch.int64 and tuple(q_indices.shape) == (POOL_COUNT,), "V2 q source-index tensor changed", blocked=True)
    _require(torch.is_tensor(q_labels) and q_labels.dtype == torch.int64 and tuple(q_labels.shape) == (POOL_COUNT,), "V2 q label tensor changed", blocked=True)
    _require(isinstance(q_identity, list) and q_identity == identities, "V2 q source identities do not match V2 rows", blocked=True)
    q = q.detach().cpu().contiguous()
    q_indices = q_indices.detach().cpu().contiguous()
    q_labels = q_labels.detach().cpu().contiguous()
    _require(torch.equal(q_indices, indices) and torch.equal(q_labels, labels), "V2 q row identities differ from source rows", blocked=True)
    _require(torch.isfinite(q).all().item() and q.ge(0.0).all().item(), "V2 q values are invalid", blocked=True)
    _require(tensor_sha256(q) == contract_definition.V2_Q_TENSOR_SHA256, "V2 q tensor identity changed", blocked=True)
    _require(payload.get("q_tensor_sha256") == contract_definition.V2_Q_TENSOR_SHA256, "V2 q payload tensor identity changed", blocked=True)
    execution = _mapping(payload.get("execution"), "V2 q execution")
    _require(execution.get("sealed_confirmation_values_loaded") is False and execution.get("training_steps") == 0, "V2 q crossed a forbidden boundary", blocked=True)
    governance = _mapping(payload.get("governance"), "V2 q governance")
    _require(governance.get("training_authorized") is False and governance.get("formal_promotion") is False and governance.get("generation_readiness") == "not_ready", "V2 q governance changed", blocked=True)
    return source, q, indices, labels, identities, fold_ids, {
        "fold_manifest": fold_manifest,
        "source_path": _relative(repo_root, source_path),
        "source_sha256": contract_definition.V2_SOURCE_MANIFEST_SHA256,
        "q_path": _relative(repo_root, q_path),
        "q_sha256": contract_definition.V2_Q_ARTIFACT_SHA256,
        "q_tensor_sha256": contract_definition.V2_Q_TENSOR_SHA256,
        "q_manifest_path": _relative(repo_root, repo_root / contract_definition.V2_Q_MANIFEST_PATH),
        "q_manifest_sha256": contract_definition.V2_Q_MANIFEST_SHA256,
        "fold_manifest_path": _relative(repo_root, repo_root / contract_definition.V2_FOLD_MANIFEST_PATH),
        "fold_manifest_sha256": contract_definition.V2_FOLD_MANIFEST_SHA256,
        "assignment_sha256": contract_definition.V2_ASSIGNMENT_SHA256,
        "fold_ids_sha256": tensor_sha256(fold_ids),
        "row_manifest_sha256": contract_definition.V2_ROW_MANIFEST_SHA256,
        "source_row_identity_sha256": contract_definition.V2_SOURCE_ROW_IDENTITY_SHA256,
    }


def _collect_complex_rows(model: torch.nn.Module, images: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    sqrt_scale = torch.sqrt(scale).to(dtype=torch.float32)
    chunks: list[torch.Tensor] = []
    batch_size = int(q_materialization.BATCH_SIZE)
    batch_count = (images.shape[0] + batch_size - 1) // batch_size
    with torch.inference_mode():
        for batch_number, start in enumerate(range(0, images.shape[0], batch_size), start=1):
            stop = min(start + batch_size, images.shape[0])
            rendered = q_materialization.base.render_intensity_batch(images[start:stop])
            coefficients = replay._encoded_complex_prefix(model, rendered.amplitude)
            _require(coefficients.dtype == torch.complex64, "fixed OLS coefficient dtype changed", blocked=True)
            normalized = (coefficients / sqrt_scale[None, :]).detach().cpu().contiguous()
            _require(tuple(normalized.shape) == (stop - start, DIMENSION), "materialized R3 shape changed", blocked=True)
            _require(torch.isfinite(normalized.real).all().item() and torch.isfinite(normalized.imag).all().item(), "materialized R3 contains non-finite values", blocked=True)
            chunks.append(normalized)
            if batch_number == 1 or batch_number == batch_count or batch_number % 25 == 0:
                print(json.dumps({"stage": "complex_ols_materialization", "processed_rows": stop, "total_rows": images.shape[0]}), flush=True)
    result = torch.cat(chunks, dim=0).contiguous()
    _require(tuple(result.shape) == (POOL_COUNT, DIMENSION) and result.dtype == torch.complex64, "full V2 complex OLS materialization changed", blocked=True)
    return result


def _materialization_payload(
    repo_root: Path,
    source: Mapping[str, Any],
    q: torch.Tensor,
    indices: torch.Tensor,
    labels: torch.Tensor,
    identities: list[str],
    fold_ids: torch.Tensor,
    z: torch.Tensor,
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    q_from_z = z.abs().to(dtype=torch.float64).contiguous()
    _require(torch.equal(q_from_z, q), "q and complex OLS rows are not bitwise corresponding", blocked=True)
    _require(torch.all(z.abs() > 0).item(), "exact-zero complex OLS coordinate is invalid_harness", blocked=True)
    return {
        "schema": MATERIALIZATION_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "semantic_authority_id": "PROJECT_COMPLEX_OLS_V1",
        "coordinate": "R3_COMPLEX",
        "formula": "z_j=c_j/sqrt(s_j)",
        "source_dtype": "torch.complex64",
        "shape": [POOL_COUNT, DIMENSION],
        "r3_complex": z,
        "source_indices": indices,
        "labels": labels,
        "source_row_identity": identities,
        "fold_ids": fold_ids,
        "identity": {
            "source_manifest_sha256": contract_definition.V2_SOURCE_MANIFEST_SHA256,
            "source_row_manifest_sha256": contract_definition.V2_ROW_MANIFEST_SHA256,
            "source_indices_sha256": tensor_sha256(indices),
            "labels_sha256": tensor_sha256(labels),
            "source_row_identity_sha256": canonical_sha256(identities),
            "fold_manifest_sha256": contract_definition.V2_FOLD_MANIFEST_SHA256,
            "assignment_sha256": contract_definition.V2_ASSIGNMENT_SHA256,
            "fold_ids_sha256": tensor_sha256(fold_ids),
            "r3_complex_tensor_sha256": tensor_sha256(z),
            "q_tensor_sha256": tensor_sha256(q),
            "q_equals_abs_r3_bitwise": True,
        },
        "frozen_ols_authority": dict(model_identity),
        "source_bank_semantic_sha256": source.get("bank_semantic_sha256"),
        "historical_ols_artifact_used_as_row_source": False,
        "v2_exact_complex_row_authority": True,
        "execution": {
            "device": "cpu",
            "batch_size": int(q_materialization.BATCH_SIZE),
            "selected_v2_source_rows_only": True,
            "full_source_image_tensor_loaded": False,
            "sealed_confirmation_values_loaded": False,
            "decoder_entered": False,
            "evaluator_entered": False,
            "optimizer_created": False,
            "backward_called": False,
            "training_steps": 0,
            "model_state_unchanged": True,
            "freeze_before_scientific_evaluation": True,
        },
        "governance": {
            "selected_producer": "EMPIRICAL_GAUSSIAN_COPULA",
            "producer_modified": False,
            "sealed_confirmation_consumed": True,
            "sealed_values_reused": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
    }


def _validate_materialization_payload(
    payload: Mapping[str, Any],
    q: torch.Tensor,
    indices: torch.Tensor,
    labels: torch.Tensor,
    identities: list[str],
    fold_ids: torch.Tensor,
) -> torch.Tensor:
    _require(payload.get("schema") == MATERIALIZATION_SCHEMA and payload.get("status") == "VALID_COMPLETED", "frozen complex materialization payload changed", blocked=True)
    z = payload.get("r3_complex")
    _require(torch.is_tensor(z) and z.dtype == torch.complex64 and tuple(z.shape) == (POOL_COUNT, DIMENSION), "frozen complex materialization tensor changed", blocked=True)
    z = z.detach().cpu().contiguous()
    _require(torch.isfinite(z.real).all().item() and torch.isfinite(z.imag).all().item(), "frozen complex materialization is non-finite", blocked=True)
    _require(torch.all(z.abs() > 0).item(), "frozen complex materialization contains an exact zero", blocked=True)
    for name, expected in (("source_indices", indices), ("labels", labels), ("fold_ids", fold_ids)):
        value = payload.get(name)
        _require(torch.is_tensor(value) and torch.equal(value.detach().cpu(), expected), f"frozen complex {name} identity changed", blocked=True)
    _require(payload.get("source_row_identity") == identities, "frozen complex source identities changed", blocked=True)
    identity = _mapping(payload.get("identity"), "frozen complex identity")
    _require(identity.get("r3_complex_tensor_sha256") == tensor_sha256(z), "frozen complex tensor digest is stale", blocked=True)
    _require(identity.get("q_tensor_sha256") == tensor_sha256(q), "frozen complex q digest is stale", blocked=True)
    _require(torch.equal(z.abs().to(dtype=torch.float64), q), "frozen q-to-complex mapping changed", blocked=True)
    execution = _mapping(payload.get("execution"), "frozen complex execution")
    for key, expected in {
        "device": "cpu",
        "sealed_confirmation_values_loaded": False,
        "decoder_entered": False,
        "evaluator_entered": False,
        "optimizer_created": False,
        "backward_called": False,
        "training_steps": 0,
        "model_state_unchanged": True,
        "freeze_before_scientific_evaluation": True,
    }.items():
        _require(execution.get(key) == expected, f"frozen complex execution boundary changed: {key}", blocked=True)
    return z


def _write_torch(path: Path, payload: Mapping[str, Any], kind: str) -> str:
    atomic_torch_save_new(path, payload, artifact_kind=kind)
    return file_sha256(path)


def _temp_ref(repo_root: Path, output_dir: Path, temporary: Path, path: Path, *, role: str, schema: str | None = None) -> dict[str, Any]:
    relative = path.relative_to(temporary)
    logical = output_dir / relative
    result = {
        "path": _relative(repo_root, logical),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "role": role,
    }
    if schema is not None:
        result["schema"] = schema
    return result


def _write_materialization_manifest(
    repo_root: Path,
    output_dir: Path,
    temporary: Path,
    source: Mapping[str, Any],
    q: torch.Tensor,
    indices: torch.Tensor,
    labels: torch.Tensor,
    identities: list[str],
    fold_ids: torch.Tensor,
    z: torch.Tensor,
    materialization_path: Path,
    model_identity: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    materialization_ref = _temp_ref(repo_root, output_dir, temporary, materialization_path, role="V2 exact complex OLS materialization", schema=MATERIALIZATION_SCHEMA)
    manifest = {
        "schema": MATERIALIZATION_MANIFEST_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "contract": {
            "path": _relative(repo_root, repo_root / CONTRACT_PATH),
            "sha256": EXPECTED_CONTRACT_SHA256,
            "classification": contract_definition.READY_CLASSIFICATION,
        },
        "source": {
            "manifest_path": _relative(repo_root, repo_root / contract_definition.V2_SOURCE_MANIFEST_PATH),
            "manifest_sha256": contract_definition.V2_SOURCE_MANIFEST_SHA256,
            "bank_semantic_sha256": source.get("bank_semantic_sha256"),
            "row_manifest_sha256": contract_definition.V2_ROW_MANIFEST_SHA256,
            "source_indices_sha256": tensor_sha256(indices),
            "labels_sha256": tensor_sha256(labels),
            "source_row_identity_sha256": canonical_sha256(identities),
            "count": POOL_COUNT,
            "rows_per_class": int(torch.bincount(labels, minlength=CLASS_COUNT)[0]),
            "row_order": "exact V2 source manifest bank_row_position 0..4999",
        },
        "q_authority": {
            "manifest_path": _relative(repo_root, repo_root / contract_definition.V2_Q_MANIFEST_PATH),
            "manifest_sha256": contract_definition.V2_Q_MANIFEST_SHA256,
            "artifact_path": _relative(repo_root, repo_root / contract_definition.V2_Q_ARTIFACT_PATH),
            "artifact_sha256": contract_definition.V2_Q_ARTIFACT_SHA256,
            "tensor_sha256": tensor_sha256(q),
            "coordinate": "R2_AMPLITUDE",
            "definition": "q_j=abs(c_j/sqrt(s_j))=abs(R3_COMPLEX_j)",
        },
        "fold_authority": {
            "manifest_path": _relative(repo_root, repo_root / contract_definition.V2_FOLD_MANIFEST_PATH),
            "manifest_sha256": contract_definition.V2_FOLD_MANIFEST_SHA256,
            "assignment_sha256": contract_definition.V2_ASSIGNMENT_SHA256,
            "fold_ids_sha256": tensor_sha256(fold_ids),
            "fit_rows_per_fold": FIT_COUNT,
            "heldout_rows_per_fold": HELDOUT_COUNT,
            "fit_rows_used_for_donors_and_control": True,
            "heldout_used_only_for_final_fold_statistic": True,
        },
        "complex_artifact": {
            **materialization_ref,
            "tensor_sha256": tensor_sha256(z),
            "dtype": str(z.dtype),
            "shape": [POOL_COUNT, DIMENSION],
            "q_abs_bitwise_equal": True,
        },
        "frozen_ols_authority": dict(model_identity),
        "phase_and_zero_policy": {
            "global_phase_authority": "GLOBAL_PHASE_GAUGE_CONFIRMED",
            "canonicalization": "first exact argmax(abs(z)) anchor, ties by smallest coordinate index, rotate by conjugate(anchor)/abs(anchor), set anchor exact non-negative real",
            "phase": "atan2(Im(z_canonical), Re(z_canonical)) in [-pi,pi]",
            "complete_vector": True,
            "exact_zero_count": int((z.abs() == 0).sum().item()),
            "minimum_positive_magnitude": float(z.abs().min()),
            "near_zero_threshold": None,
            "near_zero_special_handling": False,
        },
        "execution": {
            "device": "cpu",
            "sealed_confirmation_values_loaded": False,
            "sealed_confirmation_reused": False,
            "decoder_entered": False,
            "evaluator_entered": False,
            "optimizer_created": False,
            "backward_called": False,
            "training_steps": 0,
            "freeze_before_scientific_evaluation": True,
        },
        "governance": {
            "selected_producer": "EMPIRICAL_GAUSSIAN_COPULA",
            "amplitude_confirmation": "PASSED",
            "sealed_confirmation_consumed": True,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
    }
    manifest_path = temporary / MATERIALIZATION_MANIFEST_NAME
    atomic_write_new_json(manifest_path, manifest, artifact_kind="frozen V2 complex OLS materialization manifest")
    _require(file_sha256(materialization_path) == materialization_ref["sha256"], "complex materialization changed before freeze", blocked=True)
    return manifest_path, manifest


def _canonical_phase_and_embedding(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    canonical = contract_definition.canonicalize_existing_global_phase(z)
    phase = contract_definition.extract_complete_relative_phase(z)
    embedding = torch.cat((canonical.real, canonical.imag), dim=1).to(dtype=torch.float64).contiguous()
    _require(tuple(canonical.shape) == (z.shape[0], DIMENSION), "canonical complex shape changed")
    _require(tuple(phase.shape) == (z.shape[0], DIMENSION), "complete phase shape changed")
    _require(tuple(embedding.shape) == (z.shape[0], 2 * DIMENSION), "real complex embedding shape changed")
    _require(torch.isfinite(phase).all().item() and torch.all((phase >= -torch.pi) & (phase <= torch.pi)).item(), "phase interval changed")
    magnitudes = canonical.abs()
    anchors = magnitudes.argmax(dim=1)
    rows = torch.arange(z.shape[0], dtype=torch.int64)
    anchor_values = canonical[rows, anchors]
    _require(torch.equal(anchor_values.imag, torch.zeros_like(anchor_values.imag)), "global-phase anchor is not exactly real")
    _require(anchor_values.real.ge(0.0).all().item(), "global-phase anchor is not non-negative")
    return canonical, phase, embedding, {
        "canonical_tensor_sha256": tensor_sha256(canonical),
        "phase_tensor_sha256": tensor_sha256(phase),
        "embedding_tensor_sha256": tensor_sha256(embedding),
        "anchor_indices_sha256": tensor_sha256(anchors),
        "complete_vectors": True,
        "zero_policy_passed": True,
        "near_zero_threshold": None,
        "near_zero_special_handling": False,
    }


def _generator(seed: int) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _sample_indices(row_count: int, sample_count: int, seed: int) -> torch.Tensor:
    _require(row_count > 0 and sample_count > 0, "sampling dimensions are invalid")
    return torch.randint(row_count, (sample_count,), generator=_generator(seed), dtype=torch.int64).contiguous()


def _sample_complete_rows(rows: torch.Tensor, sample_count: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    _require(rows.ndim == 2 and rows.shape[1] == DIMENSION, "complete-row sampling input changed")
    indices = _sample_indices(int(rows.shape[0]), sample_count, seed)
    return rows.index_select(0, indices).contiguous(), indices


def _fit_dependence_for_v2(scores: torch.Tensor) -> dict[str, Any]:
    """Apply the frozen Gaussianized-score shrinkage formulas at n=4000.

    The historical copula consumer has a literal 800-row assertion because it
    was written for the old bank.  Its mathematical implementation is reused
    here with the V2 contract's fitting-row count as the only parameterized
    boundary; no producer family or shrinkage rule is changed.
    """

    values = scores.to(dtype=torch.float64).contiguous()
    _require(values.ndim == 2 and tuple(values.shape[1:]) == (DIMENSION,), "Gaussianized V2 fit shape is invalid")
    _require(torch.isfinite(values).all().item(), "Gaussianized V2 fit scores are non-finite")
    n = int(values.shape[0])
    _require(n == FIT_COUNT, "Gaussianized V2 fit row count changed")
    mean_fit = values.mean(dim=0)
    centered = values - mean_fit
    standard_deviation = torch.sqrt(centered.square().sum(dim=0).div(float(n - 1)))
    _require(torch.isfinite(standard_deviation).all().item() and standard_deviation.gt(0.0).all().item(), "Gaussianized V2 marginal standard deviation is undefined")
    standardized = centered / standard_deviation
    empirical = standardized.transpose(0, 1).mm(standardized).div(float(n - 1))
    tau = torch.trace(empirical).div(float(DIMENSION))
    identity = torch.eye(DIMENSION, dtype=torch.float64)
    target = tau * identity
    quadratic = (standardized.mm(empirical) * standardized).sum(dim=1)
    beta = (
        standardized.square().sum(dim=1).square().sum()
        - 2.0 * quadratic.sum()
        + float(n) * empirical.square().sum()
    ).div(float(n * n))
    _require(torch.isfinite(beta).item() and torch.isfinite(tau).item(), "Gaussianized V2 shrinkage plug-in is non-finite")
    beta = beta.clamp_min(0.0)
    delta = (empirical - target).square().sum()
    _require(torch.isfinite(delta).item(), "Gaussianized V2 shrinkage target distance is non-finite")
    if float(delta) == 0.0:
        alpha = torch.zeros((), dtype=torch.float64)
    else:
        alpha = (beta / delta).clamp(0.0, 1.0)
    shrunk = ((1.0 - alpha) * empirical + alpha * target).contiguous()
    shrunk = (shrunk + shrunk.transpose(0, 1)).mul(0.5).contiguous()
    _require(torch.isfinite(shrunk).all().item(), "V2 R_shrunk is non-finite")
    cholesky, info = torch.linalg.cholesky_ex(shrunk, check_errors=False)
    _require(int(info.item()) == 0, "V2 R_shrunk is not positive definite")
    eigenvalues = torch.linalg.eigvalsh(shrunk)
    _require(torch.isfinite(eigenvalues).all().item() and eigenvalues.gt(0.0).all().item(), "V2 R_shrunk spectrum is invalid")
    diagonal = torch.diagonal(shrunk)
    diagonal_error = (diagonal - 1.0).abs().max()
    _require(torch.isfinite(diagonal_error).item() and float(diagonal_error) <= copula.DIAGONAL_ATOL, "V2 R_shrunk unit diagonal invariant failed")
    return {
        "z": values,
        "z_mean": mean_fit,
        "z_centered": centered,
        "standard_deviation": standard_deviation,
        "standardized": standardized,
        "R_emp": empirical,
        "tau": tau,
        "beta": beta,
        "delta": delta,
        "alpha": alpha,
        "R_shrunk": shrunk,
        "cholesky": cholesky,
        "eigenvalues": eigenvalues,
        "diagonal": diagonal,
        "diagonal_error": diagonal_error,
    }


def _fit_candidate_fold(q_fit: torch.Tensor) -> dict[str, Any]:
    plotting_positions, sorted_values, tie_summary = copula._midrank_positions(q_fit)
    gaussianized = torch.special.ndtri(plotting_positions)
    _require(torch.isfinite(gaussianized).all().item(), "V2 Gaussianization produced non-finite scores")
    return {
        "sorted_values": sorted_values,
        "plotting_positions": plotting_positions,
        "z": gaussianized,
        "tie_summary": tie_summary,
        "dependence": _fit_dependence_for_v2(gaussianized),
    }


def _fit_summary_for_v2(fit: Mapping[str, Any], fold: int) -> dict[str, Any]:
    dependence = _mapping(fit["dependence"], "V2 fold dependence")
    eigenvalues = dependence["eigenvalues"]
    return {
        "fold": fold,
        "fit_rows": FIT_COUNT,
        "heldout_rows": HELDOUT_COUNT,
        "sorted_values_sha256": tensor_sha256(fit["sorted_values"]),
        "plotting_positions_sha256": tensor_sha256(fit["plotting_positions"]),
        "z_sha256": tensor_sha256(fit["z"]),
        "tie_summary": fit["tie_summary"],
        "z_summary": copula._numeric_summary(fit["z"]),
        "z_mean_sha256": tensor_sha256(dependence["z_mean"]),
        "z_standard_deviation_summary": copula._numeric_summary(dependence["standard_deviation"]),
        "z_standard_deviation_sha256": tensor_sha256(dependence["standard_deviation"]),
        "R_emp_sha256": tensor_sha256(dependence["R_emp"]),
        "R_shrunk_sha256": tensor_sha256(dependence["R_shrunk"]),
        "cholesky_sha256": tensor_sha256(dependence["cholesky"]),
        "tau": float(dependence["tau"]),
        "beta": float(dependence["beta"]),
        "delta": float(dependence["delta"]),
        "alpha": float(dependence["alpha"]),
        "R_shrunk_trace": float(torch.trace(dependence["R_shrunk"])),
        "R_shrunk_min_eigenvalue": float(eigenvalues.min()),
        "R_shrunk_max_eigenvalue": float(eigenvalues.max()),
        "finite": True,
        "positive_definite": True,
        "fit_row_parameterization": "V2 contract 4000-row fold; same frozen formulas as registered Gaussian copula",
    }


def _sample_q_from_fold(fit: Mapping[str, Any], sample_count: int, seed: int) -> torch.Tensor:
    dependence = _mapping(fit.get("dependence"), "fold q dependence")
    sorted_values = fit.get("sorted_values")
    cholesky = dependence.get("cholesky")
    _require(torch.is_tensor(sorted_values) and torch.is_tensor(cholesky), "fold q sampler parameters are missing")
    _require(tuple(sorted_values.shape) == (FIT_COUNT, DIMENSION) and tuple(cholesky.shape) == (DIMENSION, DIMENSION), "fold q sampler parameter shape changed")
    noise = torch.randn((sample_count, DIMENSION), generator=_generator(seed), dtype=torch.float64)
    z_new = noise.mm(cholesky.transpose(0, 1))
    u_new = torch.special.ndtr(z_new)
    q_new = copula._empirical_quantile_batch(sorted_values, u_new).to(dtype=torch.float64).contiguous()
    _require(torch.isfinite(q_new).all().item() and q_new.gt(0.0).all().item(), "candidate q sampler produced an exact-zero or invalid amplitude")
    return q_new


def _compose_candidate_rows(fit: Mapping[str, Any], phase_pool: torch.Tensor, sample_count: int, fold: int, trial: int | None, bank: int | None) -> tuple[torch.Tensor, dict[str, Any]]:
    q_seed = contract_definition.derive_seed("candidate", fold, "CANDIDATE_Q_SAMPLE", trial, bank)
    phase_seed = contract_definition.derive_seed("candidate", fold, "CANDIDATE_PHASE_DONOR", trial, bank)
    q_new = _sample_q_from_fold(fit, sample_count, q_seed)
    donor_indices = _sample_indices(int(phase_pool.shape[0]), sample_count, phase_seed)
    phase_new = phase_pool.index_select(0, donor_indices).contiguous()
    generated = contract_definition.compose_complex_from_amplitude_phase(q_new, phase_new)
    _require(torch.isfinite(generated.real).all().item() and torch.isfinite(generated.imag).all().item(), "candidate complex sample is non-finite")
    _require(torch.all(generated.abs() > 0).item(), "candidate complex sample contains exact-zero amplitude")
    return generated, {
        "q_seed": q_seed,
        "phase_donor_seed": phase_seed,
        "phase_donor_indices_sha256": tensor_sha256(donor_indices),
        "sample_count": sample_count,
        "complete_phase_vectors": True,
        "q_phase_independent_streams": True,
        "q_dependent_phase_selection": False,
        "heldout_phase_donors": False,
        "cross_fold_phase_donors": False,
        "class_conditioned_phase": False,
    }


def _energy_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    left = first.detach().cpu().to(dtype=torch.float64).contiguous()
    right = second.detach().cpu().to(dtype=torch.float64).contiguous()
    _require(left.ndim == 2 and right.ndim == 2 and left.shape[1] == right.shape[1], "energy-distance inputs are invalid")
    value = metric_consumer._energy_distance(left, right)
    _require(isinstance(value, float) and torch.isfinite(torch.tensor(value, dtype=torch.float64)).item() and value >= 0.0, "energy distance is invalid")
    return value


def _covariance(value: torch.Tensor) -> torch.Tensor:
    view = value.to(dtype=torch.float64).contiguous()
    _require(view.ndim == 2 and view.shape[0] >= 2, "covariance input is invalid")
    centered = view - view.mean(dim=0, keepdim=True)
    result = centered.transpose(0, 1).mm(centered).div(float(view.shape[0] - 1))
    _require(torch.isfinite(result).all().item(), "covariance is non-finite")
    return result


def _phase_unit(canonical: torch.Tensor) -> torch.Tensor:
    phase = torch.atan2(canonical.imag, canonical.real).to(dtype=torch.float64)
    return torch.cat((phase.cos(), phase.sin()), dim=1).contiguous()


def _descriptive_pair_metrics(heldout_z: torch.Tensor, bank_z: torch.Tensor) -> dict[str, Any]:
    heldout_canonical = contract_definition.canonicalize_existing_global_phase(heldout_z)
    bank_canonical = contract_definition.canonicalize_existing_global_phase(bank_z)
    heldout_embed = torch.cat((heldout_canonical.real, heldout_canonical.imag), dim=1).to(dtype=torch.float64).contiguous()
    bank_embed = torch.cat((bank_canonical.real, bank_canonical.imag), dim=1).to(dtype=torch.float64).contiguous()
    heldout_q = heldout_canonical.abs().to(dtype=torch.float64)
    bank_q = bank_canonical.abs().to(dtype=torch.float64)
    heldout_phase = _phase_unit(heldout_canonical)
    bank_phase = _phase_unit(bank_canonical)
    heldout_cov = _covariance(heldout_embed)
    bank_cov = _covariance(bank_embed)
    bank_cov_energy = bank_cov.square().sum()
    _require(torch.isfinite(bank_cov_energy).item() and bank_cov_energy > 0, "descriptive covariance reference is zero")
    heldout_q_centered = heldout_q - heldout_q.mean(dim=0, keepdim=True)
    bank_q_centered = bank_q - bank_q.mean(dim=0, keepdim=True)
    heldout_phase_centered = heldout_phase - heldout_phase.mean(dim=0, keepdim=True)
    bank_phase_centered = bank_phase - bank_phase.mean(dim=0, keepdim=True)
    heldout_cross = heldout_q_centered.transpose(0, 1).mm(heldout_phase_centered).div(float(heldout_q.shape[0] - 1))
    bank_cross = bank_q_centered.transpose(0, 1).mm(bank_phase_centered).div(float(bank_q.shape[0] - 1))
    per_coordinate = [
        _energy_distance(heldout_q[:, coordinate : coordinate + 1], bank_q[:, coordinate : coordinate + 1])
        for coordinate in range(DIMENSION)
    ]
    scalar = {
        "amplitude_q_discrepancy": _energy_distance(heldout_q, bank_q),
        "phase_circular_discrepancy": _energy_distance(heldout_phase, bank_phase),
        "covariance_relative_frobenius": float((heldout_cov - bank_cov).square().sum() / bank_cov_energy),
        "norm_energy_distance": _energy_distance(heldout_embed.norm(dim=1, keepdim=True), bank_embed.norm(dim=1, keepdim=True)),
        "squared_norm_energy_distance": _energy_distance(heldout_embed.square().sum(dim=1, keepdim=True), bank_embed.square().sum(dim=1, keepdim=True)),
        "marginal_q_mean_l2": float((heldout_q.mean(dim=0) - bank_q.mean(dim=0)).norm()),
        "marginal_q_std_l2": float((heldout_q.std(dim=0, unbiased=True) - bank_q.std(dim=0, unbiased=True)).norm()),
        "marginal_q_energy_distance_mean": float(sum(per_coordinate) / DIMENSION),
        "marginal_q_energy_distance_max": float(max(per_coordinate)),
    }
    _require(all(torch.isfinite(torch.tensor(value, dtype=torch.float64)).item() and value >= 0.0 for value in scalar.values()), "descriptive scalar is invalid")
    return {
        **scalar,
        "amplitude_phase_dependence": {
            "heldout_q_phase_cross_covariance_frobenius": float(heldout_cross.norm()),
            "model_q_phase_cross_covariance_frobenius": float(bank_cross.norm()),
            "absolute_frobenius_delta": float((heldout_cross - bank_cross).norm()),
            "descriptive_only": True,
        },
        "per_coordinate_q_energy_distance": [float(value) for value in per_coordinate],
        "descriptive_only": True,
    }


def _aggregate_diagnostics(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(items) == MODEL_BANK_COUNT, "descriptive model-bank count changed")
    aggregate = {name: float(sum(float(item[name]) for item in items) / len(items)) for name in DIAGNOSTIC_SCALARS}
    aggregate["per_coordinate_q_energy_distance"] = [
        float(sum(float(item["per_coordinate_q_energy_distance"][coordinate]) for item in items) / len(items))
        for coordinate in range(DIMENSION)
    ]
    dependence = {
        key: float(sum(float(item["amplitude_phase_dependence"][key]) for item in items) / len(items))
        for key in ("heldout_q_phase_cross_covariance_frobenius", "model_q_phase_cross_covariance_frobenius", "absolute_frobenius_delta")
    }
    return {
        "descriptive_only": True,
        "aggregate_mean_over_model_banks": aggregate,
        "amplitude_phase_dependence": {**dependence, "descriptive_only": True},
        "per_model_bank": items,
        "cannot_change_decision": True,
    }


def _primary_statistic(data_embedding: torch.Tensor, model_embeddings: list[torch.Tensor]) -> tuple[float, list[float]]:
    values = [_energy_distance(data_embedding, bank) for bank in model_embeddings]
    _require(len(values) == MODEL_BANK_COUNT, "primary model-bank count changed")
    result = float(sum(values) / MODEL_BANK_COUNT)
    _require(torch.isfinite(torch.tensor(result, dtype=torch.float64)).item() and result >= 0.0, "primary statistic is invalid")
    return result, values


def _run_fold(
    method: str,
    fold: int,
    q: torch.Tensor,
    z: torch.Tensor,
    phase: torch.Tensor,
    fit_positions: torch.Tensor,
    heldout_positions: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    _require(method in {"control", "candidate"}, "unknown compatibility method")
    fit_z = z.index_select(0, fit_positions).contiguous()
    fit_q = q.index_select(0, fit_positions).contiguous()
    heldout_z = z.index_select(0, heldout_positions).contiguous()
    _require(tuple(fit_z.shape) == (FIT_COUNT, DIMENSION) and tuple(heldout_z.shape) == (HELDOUT_COUNT, DIMENSION), "fold data sizes changed")
    _require(not bool(torch.isin(fit_positions, heldout_positions).any()), "fit/heldout positions overlap")

    fit_summary: dict[str, Any] | None = None
    phase_pool = phase.index_select(0, fit_positions).contiguous()
    if method == "candidate":
        fit_model = _fit_candidate_fold(fit_q)
        fit_summary = _fit_summary_for_v2(fit_model, fold)
    else:
        fit_model = None

    model_banks_z: list[torch.Tensor] = []
    model_banks_embedding: list[torch.Tensor] = []
    model_bank_seeds: list[dict[str, Any]] = []
    for bank in range(MODEL_BANK_COUNT):
        if method == "control":
            seed = contract_definition.derive_seed("control", fold, "CONTROL_MODEL_BANK", None, bank)
            sampled, sampled_indices = _sample_complete_rows(fit_z, MODEL_BANK_SIZE, seed)
            generated = contract_definition.canonicalize_existing_global_phase(sampled)
            seed_record = {
                "seed": seed,
                "fit_row_indices_sha256": tensor_sha256(sampled_indices),
                "sampling_unit": "complete fitting complex row",
            }
        else:
            generated, seed_record = _compose_candidate_rows(fit_model, phase_pool, MODEL_BANK_SIZE, fold, None, bank)  # type: ignore[arg-type]
        model_banks_z.append(generated.contiguous())
        model_banks_embedding.append(contract_definition.complex_real_embedding(generated))
        model_bank_seeds.append({"bank": bank, **seed_record})

    calibration_values: list[float] = []
    calibration_seed_records: list[dict[str, Any]] = []
    started = time.perf_counter()
    for trial in range(CALIBRATION_TRIAL_COUNT):
        if method == "control":
            pseudo_seed = contract_definition.derive_seed("control", fold, "CONTROL_PSEUDO_DATA", trial, None)
            pseudo, pseudo_indices = _sample_complete_rows(fit_z, HELDOUT_COUNT, pseudo_seed)
            pseudo_embedding = contract_definition.complex_real_embedding(pseudo)
            seed_record = {
                "trial": trial,
                "pseudo_data_seed": pseudo_seed,
                "pseudo_fit_row_indices_sha256": tensor_sha256(pseudo_indices),
            }
        else:
            pseudo, seed_record = _compose_candidate_rows(fit_model, phase_pool, HELDOUT_COUNT, fold, trial, None)  # type: ignore[arg-type]
            pseudo_embedding = contract_definition.complex_real_embedding(pseudo)
            seed_record = {"trial": trial, **seed_record}
        value, _ = _primary_statistic(pseudo_embedding, model_banks_embedding)
        calibration_values.append(value)
        calibration_seed_records.append(seed_record)
        if trial == 0 or trial + 1 == CALIBRATION_TRIAL_COUNT or (trial + 1) % 100 == 0:
            print(json.dumps({"stage": f"{method}_model_self_calibration", "fold": fold, "completed_trials": trial + 1, "total_trials": CALIBRATION_TRIAL_COUNT, "elapsed_seconds": round(time.perf_counter() - started, 3)}), flush=True)
    calibration_tensor = torch.tensor(calibration_values, dtype=torch.float64).contiguous()
    _require(tuple(calibration_tensor.shape) == (CALIBRATION_TRIAL_COUNT,) and torch.isfinite(calibration_tensor).all().item() and calibration_tensor.ge(0.0).all().item(), "model-self calibration tensor is invalid")

    heldout_embedding = contract_definition.complex_real_embedding(heldout_z)
    t_observed, per_bank_observed = _primary_statistic(heldout_embedding, model_banks_embedding)
    exceedance = int(calibration_tensor.ge(t_observed).sum().item())
    p_mc = float((1 + exceedance) / (CALIBRATION_TRIAL_COUNT + 1))
    _require(0.0 <= p_mc <= 1.0 and torch.isfinite(torch.tensor(p_mc, dtype=torch.float64)).item(), "Monte-Carlo p-value is invalid")
    fold_pass = p_mc >= PER_FOLD_ALPHA
    descriptive = _aggregate_diagnostics([_descriptive_pair_metrics(heldout_z, bank) for bank in model_banks_z])
    result = {
        "fold": fold,
        "operationally_valid": True,
        "fit_rows": FIT_COUNT,
        "heldout_rows": HELDOUT_COUNT,
        "fit_row_positions_sha256": tensor_sha256(fit_positions),
        "heldout_row_positions_sha256": tensor_sha256(heldout_positions),
        "fit_heldout_disjoint": True,
        "fit_source_scope": "fitting rows only",
        "heldout_source_scope": "heldout rows only",
        "fit_summary": fit_summary,
        "model_self_calibration": {
            "performed_before_heldout_statistic": True,
            "B": CALIBRATION_TRIAL_COUNT,
            "pseudo_data_rows": HELDOUT_COUNT,
            "model_bank_count": MODEL_BANK_COUNT,
            "model_bank_size": MODEL_BANK_SIZE,
            "calibration_tensor_sha256": tensor_sha256(calibration_tensor),
            "trial_indices_inclusive": [0, CALIBRATION_TRIAL_COUNT - 1],
            "model_bank_sampling": model_bank_seeds,
            "calibration_sampling": calibration_seed_records,
            "heldout_used_for_sampling": False,
            "no_retry": True,
            "no_best_of_n": True,
        },
        "primary": {
            "name": PRIMARY_METRIC,
            "role": "HARD_PRIMARY",
            "T_observed": t_observed,
            "per_model_bank_T_observed": per_bank_observed,
            "mc_exceedance": exceedance,
            "p_MC": p_mc,
            "alpha": PER_FOLD_ALPHA,
            "pass": fold_pass,
            "inequality": "p_MC>=0.01",
        },
        "diagnostics": descriptive,
        "generated_model_bank_tensor_sha256": tensor_sha256(torch.stack(model_banks_z)),
        "calibration_tensor_sha256": tensor_sha256(calibration_tensor),
        "method": method,
    }
    return result, torch.stack(model_banks_z).contiguous(), calibration_tensor


def _method_artifact_payload(
    method: str,
    fold_results: list[Mapping[str, Any]],
    model_banks: torch.Tensor,
    calibration: torch.Tensor,
) -> dict[str, Any]:
    _require(tuple(model_banks.shape) == (FOLD_COUNT, MODEL_BANK_COUNT, MODEL_BANK_SIZE, DIMENSION), "generated complex bank shape changed")
    _require(tuple(calibration.shape) == (FOLD_COUNT, CALIBRATION_TRIAL_COUNT), "calibration tensor shape changed")
    return {
        "schema": METHOD_ARTIFACT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "method": method,
        "method_id": CONTROL_ID if method == "control" else CANDIDATE_ID,
        "dtype": "torch.complex128",
        "model_banks": model_banks,
        "calibration_trials": calibration,
        "model_banks_tensor_sha256": tensor_sha256(model_banks),
        "calibration_tensor_sha256": tensor_sha256(calibration),
        "fold_result_primary_sha256": canonical_sha256([
            {
                "fold": item["fold"],
                "T_observed": item["primary"]["T_observed"],
                "mc_exceedance": item["primary"]["mc_exceedance"],
                "p_MC": item["primary"]["p_MC"],
                "pass": item["primary"]["pass"],
            }
            for item in fold_results
        ]),
        "model_banks_are_canonical": True,
        "complete_rows": True,
        "heldout_rows_persisted": False,
        "pseudo_data_persisted": False,
        "rng_namespace": RNG_NAMESPACE,
        "governance": {
            "sealed_confirmation_values_loaded": False,
            "sealed_confirmation_reused": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
    }


def _aggregate_primary(folds: list[Mapping[str, Any]]) -> dict[str, Any]:
    pattern = [bool(item["primary"]["pass"]) for item in folds]
    return {
        "fold_count": FOLD_COUNT,
        "fold_pass_pattern": pattern,
        "passed_fold_count": int(sum(pattern)),
        "strict_five_of_five": all(pattern),
        "per_fold_alpha": PER_FOLD_ALPHA,
        "family_alpha": FAMILY_ALPHA,
        "primary_metric": PRIMARY_METRIC,
    }


def _next_action(classification: str) -> str:
    if classification == CLASSIFICATION_SUPPORTED:
        return NEXT_SUPPORTED
    if classification == CLASSIFICATION_NOT_SUPPORTED:
        return NEXT_NOT_SUPPORTED
    return NEXT_STABILITY


def _assert_output_tree(temporary: Path, *, candidate_executed: bool) -> None:
    expected = {
        MATERIALIZATION_NAME,
        MATERIALIZATION_MANIFEST_NAME,
        CONTROL_ARTIFACT_NAME,
        LINEAGE_NAME,
        REPORT_NAME,
    }
    if candidate_executed:
        expected.add(CANDIDATE_ARTIFACT_NAME)
    actual = {path.relative_to(temporary).as_posix() for path in temporary.rglob("*") if path.is_file()}
    _require(actual == expected, f"execution output tree is not exact: {sorted(actual)}")


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve() if output_dir.is_absolute() else (repo_root / output_dir).resolve()
    _require(not output_dir.exists(), f"refusing to reuse write-once output: {output_dir}")
    git_before = _git_state(repo_root)
    _require(git_before["branch"] == EXPECTED_BRANCH, "current branch is not the execution authority branch", blocked=True)
    _require(git_before["head"] == EXPECTED_STARTING_HEAD, "current committed HEAD is not the frozen authority base", blocked=True)
    contract, authorities = _validate_contract(repo_root)
    current_governance = _validate_current_governance(repo_root, contract)
    frozen_authorities = _validate_frozen_authorities(repo_root, contract, authorities)
    source, q, source_indices, labels, identities, fold_ids, v2_runtime = _load_v2_rows_and_q(repo_root)

    # Materialize the exact V2 source rows through the same fixed OLS prefix as
    # the q consumer.  The historical 1000-row artifact remains provenance only.
    images = q_materialization._load_selected_images(repo_root, source_indices)
    model, scale, model_identity = q_materialization._load_model_and_scale(repo_root)
    state_before = q_materialization._state_digest(model)
    z = _collect_complex_rows(model, images, scale)
    state_after = q_materialization._state_digest(model)
    _require(state_before == state_after, "fixed OLS model state changed during complex materialization", blocked=True)
    del images, model

    materialization_payload = _materialization_payload(
        repo_root,
        source,
        q,
        source_indices,
        labels,
        identities,
        fold_ids,
        z,
        model_identity,
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    created_at = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        materialization_path = temporary / MATERIALIZATION_NAME
        _write_torch(materialization_path, materialization_payload, "V2 exact complex OLS materialization")
        materialization_manifest_path, _materialization_manifest = _write_materialization_manifest(
            repo_root,
            output_dir,
            temporary,
            source,
            q,
            source_indices,
            labels,
            identities,
            fold_ids,
            z,
            materialization_path,
            model_identity,
        )
        materialization_manifest_sha = file_sha256(materialization_manifest_path)
        loaded_materialization = _mapping(safe_torch_load(materialization_path, expected_sha256=file_sha256(materialization_path), map_location="cpu"), "frozen complex materialization")
        z_frozen = _validate_materialization_payload(loaded_materialization, q, source_indices, labels, identities, fold_ids)
        _require(file_sha256(materialization_manifest_path) == materialization_manifest_sha, "materialization manifest changed after freeze", blocked=True)

        _, phase, _, phase_identity = _canonical_phase_and_embedding(z_frozen)
        control_folds: list[dict[str, Any]] = []
        control_bank_tensors: list[torch.Tensor] = []
        control_calibrations: list[torch.Tensor] = []
        for fold in range(FOLD_COUNT):
            heldout_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1).contiguous()
            fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1).contiguous()
            _require(fit_positions.numel() == FIT_COUNT and heldout_positions.numel() == HELDOUT_COUNT, f"fold {fold} sizes changed", blocked=True)
            fold_result, banks, calibration = _run_fold("control", fold, q, z_frozen, phase, fit_positions, heldout_positions)
            control_folds.append(fold_result)
            control_bank_tensors.append(banks)
            control_calibrations.append(calibration)
        control_aggregate = _aggregate_primary(control_folds)
        control_payload = _method_artifact_payload(
            "control",
            control_folds,
            torch.stack(control_bank_tensors).contiguous(),
            torch.stack(control_calibrations).contiguous(),
        )
        control_artifact_path = temporary / CONTROL_ARTIFACT_NAME
        _write_torch(control_artifact_path, control_payload, "complex OLS complete-row control banks and calibration")
        _require(_mapping(safe_torch_load(control_artifact_path, expected_sha256=file_sha256(control_artifact_path), map_location="cpu"), "control artifact").get("model_banks_tensor_sha256") == control_payload["model_banks_tensor_sha256"], "control artifact reload changed")

        candidate_executed = bool(control_aggregate["strict_five_of_five"])
        candidate_folds: list[dict[str, Any]] = []
        candidate_bank_tensors: list[torch.Tensor] = []
        candidate_calibrations: list[torch.Tensor] = []
        candidate_artifact_path: Path | None = None
        if candidate_executed:
            for fold in range(FOLD_COUNT):
                heldout_positions = (fold_ids == fold).nonzero(as_tuple=False).reshape(-1).contiguous()
                fit_positions = (fold_ids != fold).nonzero(as_tuple=False).reshape(-1).contiguous()
                fold_result, banks, calibration = _run_fold("candidate", fold, q, z_frozen, phase, fit_positions, heldout_positions)
                candidate_folds.append(fold_result)
                candidate_bank_tensors.append(banks)
                candidate_calibrations.append(calibration)
            candidate_aggregate = _aggregate_primary(candidate_folds)
            candidate_payload = _method_artifact_payload(
                "candidate",
                candidate_folds,
                torch.stack(candidate_bank_tensors).contiguous(),
                torch.stack(candidate_calibrations).contiguous(),
            )
            candidate_artifact_path = temporary / CANDIDATE_ARTIFACT_NAME
            _write_torch(candidate_artifact_path, candidate_payload, "complex OLS phase-inheritance banks and calibration")
            _require(_mapping(safe_torch_load(candidate_artifact_path, expected_sha256=file_sha256(candidate_artifact_path), map_location="cpu"), "candidate artifact").get("model_banks_tensor_sha256") == candidate_payload["model_banks_tensor_sha256"], "candidate artifact reload changed")
            classification = CLASSIFICATION_SUPPORTED if candidate_aggregate["strict_five_of_five"] else CLASSIFICATION_NOT_SUPPORTED
        else:
            candidate_aggregate = {
                "execution_status": "NOT_EXECUTED_BY_CONTRACT",
                "reason": "control did not pass the strict five-fold gate",
                "fold_count": 0,
                "strict_five_of_five": False,
            }
            classification = CLASSIFICATION_STABILITY

        git_after = _git_state(repo_root)
        _require(git_after["branch"] == git_before["branch"] and git_after["head"] == git_before["head"], "branch or committed HEAD changed during execution", blocked=True)
        governance_after_sha = file_sha256(repo_root / contract_definition.GOVERNANCE_STATE_PATH)
        _require(governance_after_sha == current_governance["sha256_before"], "governance state changed during execution", blocked=True)

        materialization_ref = _temp_ref(repo_root, output_dir, temporary, materialization_path, role="V2 exact complex OLS materialization", schema=MATERIALIZATION_SCHEMA)
        materialization_manifest_ref = _temp_ref(repo_root, output_dir, temporary, materialization_manifest_path, role="V2 complex OLS materialization manifest", schema=MATERIALIZATION_MANIFEST_SCHEMA)
        control_ref = _temp_ref(repo_root, output_dir, temporary, control_artifact_path, role="complete-complex OLS control banks and calibration", schema=METHOD_ARTIFACT_SCHEMA)
        candidate_ref = _temp_ref(repo_root, output_dir, temporary, candidate_artifact_path, role="independent phase-inheritance banks and calibration", schema=METHOD_ARTIFACT_SCHEMA) if candidate_artifact_path is not None else None
        lineage = {
            "schema": LINEAGE_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "gate": GATE,
            "status": "valid_completed",
            "result": "VALID_COMPLETED",
            "classification": classification,
            "created_at_utc": created_at,
            "contract": {
                "path": _relative(repo_root, repo_root / CONTRACT_PATH),
                "sha256": EXPECTED_CONTRACT_SHA256,
                "classification": contract_definition.READY_CLASSIFICATION,
            },
            "authorities": {
                "frozen_contract_sha256": EXPECTED_CONTRACT_SHA256,
                "v2": frozen_authorities["v2"],
                "v2_loaded_identity": {
                    "q_path": v2_runtime["q_path"],
                    "q_sha256": v2_runtime["q_sha256"],
                    "q_tensor_sha256": v2_runtime["q_tensor_sha256"],
                    "fold_manifest_path": v2_runtime["fold_manifest_path"],
                    "fold_manifest_sha256": v2_runtime["fold_manifest_sha256"],
                    "assignment_sha256": v2_runtime["assignment_sha256"],
                    "fold_ids_sha256": v2_runtime["fold_ids_sha256"],
                },
                "project_complex_ols": frozen_authorities["project_complex_ols"],
                "selected_producer_and_prior_confirmation": frozen_authorities["selected_producer_and_prior_confirmation"],
                "metric_and_evaluator": frozen_authorities["metric_and_evaluator"],
                "current_governance_before_sha256": current_governance["sha256_before"],
            },
            "complex_materialization": {
                "artifact": materialization_ref,
                "manifest": materialization_manifest_ref,
                "source_rows": POOL_COUNT,
                "r3_complex_tensor_sha256": tensor_sha256(z_frozen),
                "q_tensor_sha256": tensor_sha256(q),
                "q_abs_bitwise_equal": True,
                "historical_1000_row_artifact_used_as_source": False,
                "phase_identity": phase_identity,
            },
            "control": {
                "id": CONTROL_ID,
                "artifact": control_ref,
                "folds": control_folds,
                "aggregate": control_aggregate,
                "evaluation_order": 1,
            },
            "candidate": {
                "id": CANDIDATE_ID,
                "artifact": candidate_ref,
                "folds": candidate_folds if candidate_executed else [],
                "aggregate": candidate_aggregate,
                "evaluation_order": 2 if candidate_executed else None,
                "not_executed_by_contract": not candidate_executed,
            },
            "rng": {
                "namespace": RNG_NAMESPACE,
                "generator": "torch.Generator(device='cpu') per explicit stream",
                "seed_derivation": "contract_definition.derive_seed; sha256 namespace/method/fold/role/trial/bank",
                "global_rng_used": False,
                "seed_retry": False,
                "best_of_n": False,
                "all_seeds_recorded": True,
            },
            "execution": {
                "device": "cpu",
                "primary_metric": PRIMARY_METRIC,
                "B": CALIBRATION_TRIAL_COUNT,
                "per_fold_alpha": PER_FOLD_ALPHA,
                "strict_aggregate": "5/5",
                "descriptive_metrics_decisional": False,
                "sealed_confirmation_values_loaded": False,
                "sealed_confirmation_reused": False,
                "decoder_entered": False,
                "evaluator_entered": False,
                "optimizer_created": False,
                "backward_called": False,
                "training_steps": 0,
            },
            "governance": {
                "selected_producer_unchanged": "EMPIRICAL_GAUSSIAN_COPULA",
                "n3_amplitude_producer_confirmation": "PASSED",
                "sealed_confirmation_consumed": True,
                "sealed_confirmation_reused": False,
                "training_authorized": False,
                "formal_promotion": False,
                "generation_readiness": "not_ready",
            },
            "git": {
                "branch": git_before["branch"],
                "head_before": git_before["head"],
                "head_after": git_after["head"],
                "worktree_before": git_before,
                "worktree_after": git_after,
                "tracked_files_modified_by_consumer": [],
                "commit_status": "not_committed",
                "output_artifacts_write_once": True,
            },
            "consumer_report_path": _relative(repo_root, output_dir / REPORT_NAME),
            "consumer_report_is_not_self_referenced": True,
            "next_action": _next_action(classification),
        }
        lineage_path = temporary / LINEAGE_NAME
        atomic_write_new_json(lineage_path, lineage, artifact_kind="complex OLS compatibility lineage manifest")
        lineage_ref = _temp_ref(repo_root, output_dir, temporary, lineage_path, role="complex OLS compatibility lineage manifest", schema=LINEAGE_SCHEMA)

        report = {
            "schema": REPORT_SCHEMA,
            "version": 1,
            "task_id": TASK_ID,
            "gate": GATE,
            "status": "valid_completed",
            "result": "VALID_COMPLETED",
            "classification": classification,
            "created_at_utc": created_at,
            "contract": {
                "path": _relative(repo_root, repo_root / CONTRACT_PATH),
                "sha256": EXPECTED_CONTRACT_SHA256,
                "result": "READY",
                "classification": contract_definition.READY_CLASSIFICATION,
            },
            "objective": "full complex OLS control first; independent observed phase inheritance only after control 5/5",
            "control": {
                "id": CONTROL_ID,
                "folds": control_folds,
                "aggregate": control_aggregate,
                "scientific_gate": "p_MC>=0.01 on energy_distance; strict 5/5",
            },
            "candidate": {
                "id": CANDIDATE_ID,
                "execution_status": "EXECUTED" if candidate_executed else "NOT_EXECUTED_BY_CONTRACT",
                "folds": candidate_folds if candidate_executed else [],
                "aggregate": candidate_aggregate,
                "scientific_gate": "p_MC>=0.01 on energy_distance; strict 5/5" if candidate_executed else None,
            },
            "diagnostics": {
                "control": [item["diagnostics"] for item in control_folds],
                "candidate": [item["diagnostics"] for item in candidate_folds] if candidate_executed else [],
                "descriptive_only": True,
                "cannot_change_control_or_candidate_pass_fail": True,
            },
            "provenance": {
                "complex_materialization": materialization_ref,
                "complex_materialization_manifest": materialization_manifest_ref,
                "q_artifact": {
                    "path": v2_runtime["q_path"],
                    "sha256": v2_runtime["q_sha256"],
                    "tensor_sha256": v2_runtime["q_tensor_sha256"],
                },
                "q_manifest": frozen_authorities["v2"]["q"]["manifest"],
                "source_manifest": frozen_authorities["v2"]["source"],
                "fold_manifest": frozen_authorities["v2"]["folds"],
                "project_complex_ols": frozen_authorities["project_complex_ols"],
                "control_generated_banks_and_calibration": control_ref,
                "candidate_generated_banks_and_calibration": candidate_ref,
                "lineage_manifest": lineage_ref,
                "rng": lineage["rng"],
            },
            "local_state": {
                "branch": git_before["branch"],
                "head_before": git_before["head"],
                "head_after": git_after["head"],
                "worktree_before": git_before,
                "worktree_after": git_after,
                "modified_files": git_after["dirty_paths"],
                "tracked_files_modified_by_consumer": [],
                "commit_status": "not_committed",
            },
            "governance": lineage["governance"],
            "sealed_confirmation": {
                "prior_result": "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED",
                "consumed": True,
                "values_loaded": False,
                "reused": False,
                "new_sealed_authority": False,
            },
            "boundaries": {
                "selected_producer_modified": False,
                "amplitude_producer_confirmation": "PASSED",
                "labels_as_inputs": False,
                "heldout_phase_donors": False,
                "q_dependent_phase_selection": False,
                "conditional_phase_model": False,
                "semantic_or_image_evaluation": False,
                "optical_forward": False,
                "cuda": False,
                "training": False,
                "formal_promotion": False,
                "generation_readiness": "not_ready",
            },
            "next_action": _next_action(classification),
            "artifacts_write_once": True,
        }
        report_path = temporary / REPORT_NAME
        atomic_write_new_json(report_path, report, artifact_kind="complex OLS compatibility report")
        _assert_output_tree(temporary, candidate_executed=candidate_executed)
        _require(not output_dir.exists(), f"output appeared during execution: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return {
        "result": "VALID_COMPLETED",
        "classification": classification,
        "control_passed_folds": control_aggregate["passed_fold_count"],
        "candidate_executed": candidate_executed,
        "output_directory": _relative(repo_root, output_dir),
        "report": _relative(repo_root, output_dir / REPORT_NAME),
        "report_sha256": file_sha256(output_dir / REPORT_NAME),
        "lineage": _relative(repo_root, output_dir / LINEAGE_NAME),
        "lineage_sha256": file_sha256(output_dir / LINEAGE_NAME),
        "head_before": git_before["head"],
        "head_after": git_after["head"],
        "sealed_confirmation_consumed": True,
        "sealed_confirmation_reused": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "next_action": _next_action(classification),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        result = execute(args.repo_root, args.output_dir)
    except BlockedExecution as exc:
        print(json.dumps({"result": "BLOCKED", "classification": None, "reason": str(exc)}, indent=2, sort_keys=True))
        return 3
    except InvalidHarnessError as exc:
        print(json.dumps({"result": "INVALID", "classification": "INVALID_HARNESS", "reason": str(exc)}, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
