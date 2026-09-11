"""Define the one-shot sealed confirmation contract for the selected N3 producer.

This is a definition-only consumer.  It reads authority manifests and the
already-frozen producer state, but never reads the reserved source content,
development q values, or any sealed q/image artifact.  The resulting contract
freezes the future materialization, model-self calibration, statistic,
Monte-Carlo decision, and irreversible-consumption boundary.
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
    atomic_write_new_json,
    canonical_sha256,
    file_sha256,
    safe_torch_load,
    tensor_sha256,
)
from scripts.research import materialize_n3_development_bank_v2_q as q_materialization


TASK_ID = "DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "n3_selected_producer_sealed_confirmation_contract.v1"
READY_CLASSIFICATION = "DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1_READY"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"

OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "n3_selected_producer_sealed_confirmation_contract_v1_attempt_01"
)
CONTRACT_NAME = "n3_selected_producer_sealed_confirmation_contract.v1.json"

FREEZE_CONTRACT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_selected_producer_freeze_contract_v1_attempt_01/"
    "n3_selected_producer_freeze_contract.v1.json"
)
FREEZE_CONTRACT_SHA256 = "4d8d3f5dc81a7eff2d92a0acde0a0d0639788bb0dee75e6d88db745f642b60a3"

PRODUCER_STATE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "n3_selected_producer_full_development_fit_v1_attempt_01/"
    "n3_selected_empirical_gaussian_copula_producer_state.v1.pt"
)
PRODUCER_STATE_SHA256 = "752db16087e7bdf8343e4499202182370b376a1a2d5925a383f411fd4a15bb88"
PRODUCER_LINEAGE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "n3_selected_producer_full_development_fit_v1_attempt_01/"
    "n3_selected_producer_full_development_fit_lineage.v1.json"
)
PRODUCER_LINEAGE_SHA256 = "9244df01c95e44005207cb1dc188fc7de2961665e13055beebf55f0f5cdd2267"
PRODUCER_STATE_SCHEMA = "n3_selected_empirical_gaussian_copula_producer_state.v1"
PRODUCER_STATE_SEMANTIC_ID = "N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1"
PRODUCER_FAMILY_ID = "EMPIRICAL_GAUSSIAN_COPULA"
PRODUCER_CANDIDATE_ID = "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1"
PRODUCER_SAMPLING_NAMESPACE = "N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1_CPU"

SEALED_MANIFEST_PATH = Path(
    "outputs/20260908_v2/runs/"
    "n3_sealed_confirmation_source_v1_attempt_01/"
    "n3_sealed_confirmation_source_manifest.v1.json"
)
SEALED_MANIFEST_SHA256 = "6475474f157d31591a3b0252f27320323d84ea93803c61ee6a365420a8ebb08e"
SEALED_SEMANTIC_AUTHORITY_ID = "N3_SEALED_CONFIRMATION_SOURCE_V1"
SEALED_SEMANTIC_CONFIRMATION_ID = (
    "N3_SEALED_CONFIRMATION_SOURCE_V1:"
    "5bf32ec1bfb8ffb9e62f5f2fa578cc5dffb1d288eba6ce850f024781e5e13f10"
)

V2_SOURCE_MANIFEST_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_source_manifest_v1_attempt_01/"
    "n3_development_bank_v2_source_manifest.v1.json"
)
V2_SOURCE_MANIFEST_SHA256 = "4bc251af91b3a6cd0235fe113117443d871a564bd05634fa9e61b5ab0b36f812"
V2_Q_MANIFEST_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_q_materialization_v1_attempt_01/"
    "n3_development_bank_v2_q_materialization_manifest.v1.json"
)
V2_Q_MANIFEST_SHA256 = "422d8c65b78bc459197ed46504399cb9c87ba8f7184eb2405e0faa85353995d8"
V2_Q_ARTIFACT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_q_materialization_v1_attempt_01/"
    "n3_development_bank_v2_q.v1.pt"
)
V2_Q_ARTIFACT_SHA256 = "fe48c8fdb9bb2bf732076fabdf475288baa6f7d628ca10039a83bbda54fcfcbf"
V2_Q_TENSOR_SHA256 = "d660fd8dc449725edab081d00c8f553f151029cddef9210360a7a6fad4281209"

V3_CALIBRATION_REPORT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_q_compatibility_guardrail_gate_v1_attempt_01/"
    "n3_q_compatibility_guardrail_calibration_report.v1.json"
)
V3_CALIBRATION_REPORT_SHA256 = "7135b07fd532ee5dd2dba7900ad6028c9a6d558924691e4d65eac56330367789"
V3_EVALUATOR_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_q_compatibility_guardrail_gate_v1_attempt_01/"
    "n3_q_compatibility_evaluator_v3.v1.json"
)
V3_EVALUATOR_SHA256 = "4fbc831e383897802134ea2ce1aaef6446629130a7c799630254a86b8dcff3cc"
V3_METRIC_IMPLEMENTATION_PATH = Path(
    "scripts/research/validate_n3_amplitude_producer_baseline.py"
)
V3_METRIC_IMPLEMENTATION_SHA256 = "1ad916bbb8d109ca989f64b9f5590391cdffea9e0567e2ca8d241b032c6cde36"
V3_EVALUATOR_IMPLEMENTATION_PATH = Path(
    "scripts/research/execute_n3_development_bank_v2_empirical_ceiling.py"
)
V3_EVALUATOR_IMPLEMENTATION_SHA256 = "18b8f9defd83c2591dc055a97621ef6a1717783f7656da75994881e02bffda14"
V3_AGGREGATION_IMPLEMENTATION_PATH = Path(
    "scripts/research/execute_n3_empirical_marginal_gaussian_copula_producer_v1.py"
)
V3_AGGREGATION_IMPLEMENTATION_SHA256 = "5f1fa3fa6be35e96f345c3b7f7c6cce5ca0b45fe4619b1ead4455e86dcae538a"
Q_REPLAY_IMPLEMENTATION_PATH = Path(
    "scripts/research/replay_scaled_chen_repaired_coordinate_mra.py"
)

GOVERNANCE_STATE_PATH = Path("docs/status/research_governance_state.json")
CURRENT_STATUS_PATH = Path("docs/CURRENT_STATUS.md")
CURRENT_HANDOFF_PATH = Path("docs/status/current_handoff.md")

CONFIRMATION_CONSUMER_PATH = Path(
    "scripts/research/execute_n3_selected_producer_sealed_confirmation_v1.py"
)
FUTURE_EXECUTION_DIR = Path(
    "outputs/20260910_v2/runs/"
    "n3_selected_producer_sealed_confirmation_v1_attempt_01"
)
CALIBRATION_ARTIFACT_PATH = FUTURE_EXECUTION_DIR / (
    "n3_selected_producer_sealed_confirmation_model_self_calibration.v1.pt"
)
MODEL_BANK_ARTIFACT_PATH = FUTURE_EXECUTION_DIR / (
    "n3_selected_producer_sealed_confirmation_model_banks.v1.pt"
)
SEALED_Q_ARTIFACT_PATH = FUTURE_EXECUTION_DIR / "n3_selected_producer_sealed_confirmation_q.v1.pt"
SEALED_Q_MANIFEST_PATH = FUTURE_EXECUTION_DIR / (
    "n3_selected_producer_sealed_confirmation_q_materialization_manifest.v1.json"
)
CONSUMPTION_MARKER_PATH = FUTURE_EXECUTION_DIR / (
    "n3_selected_producer_sealed_confirmation_consumption.v1.json"
)
EXECUTION_LINEAGE_PATH = FUTURE_EXECUTION_DIR / (
    "n3_selected_producer_sealed_confirmation_lineage.v1.json"
)

CLASS_COUNT = 10
DIMENSION = 100
DEVELOPMENT_ROWS = 5000
DEVELOPMENT_ROWS_PER_CLASS = 500
SEALED_ROWS = 500
SEALED_ROWS_PER_CLASS = 50
MODEL_BANK_COUNT = 4
MODEL_BANK_SIZE = 200
CALIBRATION_TRIAL_COUNT = 999
ALPHA = 0.05
PRIMARY_METRIC = "energy_distance"
DESCRIPTIVE_ONLY_METRICS = (
    "covariance_relative_frobenius",
    "norm_energy_distance",
    "squared_norm_energy_distance",
    "marginal_mean_l2",
    "marginal_std_l2",
    "marginal_energy_distance_mean",
    "marginal_energy_distance_max",
)
ALL_V3_METRICS = (PRIMARY_METRIC, *DESCRIPTIVE_ONLY_METRICS)
SEED_DERIVATION_ROOT = "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_V1_CPU"
MAX_SIGNED_63 = (1 << 63) - 1

Q_ARTIFACT_SCHEMA = "n3_selected_producer_sealed_confirmation_q.v1"
Q_MANIFEST_SCHEMA = "n3_selected_producer_sealed_confirmation_q_materialization_manifest.v1"
CALIBRATION_SCHEMA = "n3_selected_producer_sealed_confirmation_model_self_calibration.v1"
MODEL_BANK_SCHEMA = "n3_selected_producer_sealed_confirmation_model_banks.v1"
CONSUMPTION_MARKER_SCHEMA = "n3_selected_producer_sealed_confirmation_consumption.v1"
LINEAGE_SCHEMA = "n3_selected_producer_sealed_confirmation_lineage.v1"


class InvalidHarnessError(RuntimeError):
    """A frozen authority or contract invariant is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidHarnessError(message)


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
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidHarnessError(f"{name} cannot be read as JSON: {target}") from exc
    return dict(_mapping(value, name))


def _authority_ref(
    repo_root: Path,
    path: Path,
    role: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"authority is missing: {target}")
    actual = file_sha256(target)
    if expected_sha256 is not None:
        _require(actual == expected_sha256, f"authority hash changed: {target}")
    return {"path": _relative(repo_root, target), "sha256": actual, "role": role}


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


def _validate_current_state(repo_root: Path) -> dict[str, Any]:
    governance_path = repo_root / GOVERNANCE_STATE_PATH
    governance_sha256 = file_sha256(governance_path)
    governance = _read_json_exact(
        repo_root,
        GOVERNANCE_STATE_PATH,
        governance_sha256,
        "current governance state",
    )
    _require(
        governance.get("active_objective") == "N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_COMPLETED",
        "current active objective is not the completed full-development fit",
    )
    _require(
        governance.get("hard_blocker") == "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_REQUIRED",
        "current hard blocker is not sealed confirmation contract definition",
    )
    _require(governance.get("authorization_tier") == "R0", "definition task is not at R0")
    aliases = _mapping(governance.get("evidence_aliases"), "governance evidence aliases")
    full_fit = _mapping(
        aliases.get("N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_V1"),
        "full-development-fit governance alias",
    )
    _require(
        full_fit.get("sha256") == PRODUCER_LINEAGE_SHA256
        and full_fit.get("producer_state_sha256") == PRODUCER_STATE_SHA256,
        "governance full-fit alias does not bind the frozen state",
    )
    _require(
        full_fit.get("selected_producer") == PRODUCER_FAMILY_ID
        and full_fit.get("frozen_producer_state") == PRODUCER_STATE_SEMANTIC_ID,
        "governance full-fit producer identity changed",
    )
    for key in ("training_authorized", "formal_promotion"):
        _require(full_fit.get(key) is False, f"full-fit governance opened {key}")
    _require(full_fit.get("generation_readiness") == "not_ready", "full-fit governance opened generation")
    _require(full_fit.get("sealed_confirmation_consumed") is False, "sealed confirmation was consumed")
    _require(
        "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1" not in aliases,
        "sealed confirmation contract alias already exists",
    )

    for path, name in (
        (CURRENT_STATUS_PATH, "CURRENT_STATUS"),
        (CURRENT_HANDOFF_PATH, "current handoff"),
    ):
        text = (repo_root / path).read_text(encoding="utf-8")
        _require("N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_COMPLETED" in text, f"{name} omits full-fit state")
        _require(TASK_ID in text, f"{name} omits this contract as the next action")
        _require("training_authorized=false" in text and "generation_readiness=not_ready" in text, f"{name} governance boundary changed")

    return {
        "governance": _authority_ref(repo_root, GOVERNANCE_STATE_PATH, "pre-contract governance snapshot", governance_sha256),
        "current_status": _authority_ref(repo_root, CURRENT_STATUS_PATH, "pre-contract current status snapshot"),
        "current_handoff": _authority_ref(repo_root, CURRENT_HANDOFF_PATH, "pre-contract current handoff snapshot"),
        "active_objective": governance["active_objective"],
        "hard_blocker": governance["hard_blocker"],
        "authorization_tier": governance["authorization_tier"],
    }


def _validate_freeze_contract(repo_root: Path) -> dict[str, Any]:
    contract = _read_json_exact(
        repo_root,
        FREEZE_CONTRACT_PATH,
        FREEZE_CONTRACT_SHA256,
        "selected-producer freeze contract",
    )
    _require(contract.get("schema") == "n3_selected_producer_freeze_contract.v1", "freeze contract schema changed")
    _require(contract.get("status") == "valid_completed" and contract.get("result") == "VALID_COMPLETED", "freeze contract is incomplete")
    _require(contract.get("classification") == "DEFINE_N3_SELECTED_PRODUCER_FREEZE_CONTRACT_V1_READY", "freeze contract classification changed")
    _require(contract.get("next_action") == "EXECUTE_N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_V1", "freeze contract next action changed")
    selection = _mapping(contract.get("selection"), "freeze contract selection")
    _require(selection.get("selected_development_producer_candidate") == PRODUCER_FAMILY_ID, "selected producer family changed")
    _require(selection.get("selected_candidate_id") == PRODUCER_CANDIDATE_ID, "selected producer candidate changed")
    _require(selection.get("project_selected_producer") is None, "project producer was already selected in freeze contract")
    _require(selection.get("frozen_producer_state") is None, "freeze contract unexpectedly contains a state")
    return {
        "path": _relative(repo_root, repo_root / FREEZE_CONTRACT_PATH),
        "sha256": FREEZE_CONTRACT_SHA256,
        "schema": contract["schema"],
        "classification": contract["classification"],
        "selected_family": PRODUCER_FAMILY_ID,
        "selected_candidate": PRODUCER_CANDIDATE_ID,
    }


def _tensor_shape(value: Any, shape: tuple[int, ...], name: str) -> None:
    _require(torch.is_tensor(value), f"{name} is not a tensor")
    _require(tuple(value.shape) == shape, f"{name} shape changed")
    _require(value.dtype == torch.float64, f"{name} dtype changed")


def _validate_producer_state(repo_root: Path) -> dict[str, Any]:
    lineage = _read_json_exact(
        repo_root,
        PRODUCER_LINEAGE_PATH,
        PRODUCER_LINEAGE_SHA256,
        "selected producer full-fit lineage",
    )
    _require(lineage.get("schema") == "n3_selected_producer_full_development_fit_lineage.v1", "full-fit lineage schema changed")
    _require(lineage.get("status") == "valid_completed" and lineage.get("result") == "VALID_COMPLETED", "full-fit lineage is incomplete")
    _require(lineage.get("classification") == "N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_COMPLETED", "full-fit classification changed")
    _require(lineage.get("next_action") == TASK_ID, "full-fit next action changed")
    lineage_state = _mapping(lineage.get("producer_state"), "full-fit lineage producer state")
    _require(lineage_state.get("semantic_authority_id") == PRODUCER_STATE_SEMANTIC_ID, "producer state authority changed")
    _require(lineage_state.get("sha256") == PRODUCER_STATE_SHA256, "producer state SHA changed in lineage")
    _require(lineage_state.get("schema") == PRODUCER_STATE_SCHEMA, "producer state schema changed in lineage")
    _require(lineage_state.get("raw_q_rows_persisted") is False, "raw q rows are persisted in producer state lineage")
    _require(lineage_state.get("sealed_information_persisted") is False, "sealed information is persisted in producer state lineage")

    state_path = repo_root / PRODUCER_STATE_PATH
    _require(state_path.is_file(), f"producer state is missing: {state_path}")
    state = safe_torch_load(state_path, expected_sha256=PRODUCER_STATE_SHA256, map_location="cpu")
    state = _mapping(state, "selected producer state")
    _require(state.get("schema") == PRODUCER_STATE_SCHEMA, "producer state schema changed")
    _require(state.get("semantic_authority_id") == PRODUCER_STATE_SEMANTIC_ID, "producer state semantic authority changed")
    _require(state.get("family_id") == PRODUCER_FAMILY_ID and state.get("candidate_id") == PRODUCER_CANDIDATE_ID, "producer state producer identity changed")
    fit_scope = _mapping(state.get("fit_scope"), "producer state fit scope")
    for key, expected in {
        "fit_row_count": DEVELOPMENT_ROWS,
        "fit_rows_per_class": DEVELOPMENT_ROWS_PER_CLASS,
        "folds_used_for_fit": False,
        "heldout_rows_used": False,
        "labels_as_inputs": False,
        "fit_execution_count": 1,
        "device": "cpu",
        "dtype": "torch.float64",
    }.items():
        _require(fit_scope.get(key) == expected, f"producer state fit scope changed: {key}")
    sampling = _mapping(state.get("sampling_tensors"), "producer state sampling tensors")
    _tensor_shape(sampling.get("sorted_values"), (DEVELOPMENT_ROWS, DIMENSION), "sorted_values")
    _tensor_shape(sampling.get("quantile_table_positions"), (DEVELOPMENT_ROWS,), "quantile_table_positions")
    _tensor_shape(sampling.get("cholesky"), (DIMENSION, DIMENSION), "cholesky")
    provenance = _mapping(state.get("provenance"), "producer state provenance")
    _require(provenance.get("sealed_confirmation_included") is False, "sealed confirmation entered producer state provenance")
    _require(provenance.get("compatibility_evaluation_included") is False, "compatibility evaluation entered producer state provenance")
    state_governance = _mapping(state.get("governance"), "producer state governance")
    _require(state_governance.get("selected_producer") == PRODUCER_FAMILY_ID, "producer state selected family changed")
    _require(state_governance.get("frozen_producer_state") == PRODUCER_STATE_SEMANTIC_ID, "producer state frozen identity changed")
    for key in ("training_authorized", "formal_promotion", "sealed_confirmation_consumed"):
        _require(state_governance.get(key) is False, f"producer state governance opened {key}")
    _require(state_governance.get("generation_readiness") == "not_ready", "producer state opened generation")
    return {
        "lineage": {
            "path": _relative(repo_root, repo_root / PRODUCER_LINEAGE_PATH),
            "sha256": PRODUCER_LINEAGE_SHA256,
            "schema": lineage["schema"],
        },
        "state": {
            "path": _relative(repo_root, state_path),
            "sha256": PRODUCER_STATE_SHA256,
            "schema": PRODUCER_STATE_SCHEMA,
            "semantic_authority_id": PRODUCER_STATE_SEMANTIC_ID,
            "family_id": PRODUCER_FAMILY_ID,
            "candidate_id": PRODUCER_CANDIDATE_ID,
            "fit_row_count": DEVELOPMENT_ROWS,
            "fit_rows_per_class": DEVELOPMENT_ROWS_PER_CLASS,
            "state_safe_loaded": True,
            "sealed_information_persisted": False,
        },
    }


def _validate_v2_authority(repo_root: Path) -> tuple[dict[str, Any], set[int]]:
    source = _read_json_exact(repo_root, V2_SOURCE_MANIFEST_PATH, V2_SOURCE_MANIFEST_SHA256, "V2 development source manifest")
    _require(source.get("schema") == "n3_development_bank_v2_source_manifest.v1", "V2 source schema changed")
    _require(source.get("semantic_authority_id") == "N3_DEVELOPMENT_BANK_V2", "V2 source semantic authority changed")
    _require(source.get("status") == "FROZEN_SOURCE_ROWS", "V2 source status changed")
    identity = _mapping(source.get("identity"), "V2 source identity")
    _require(identity.get("class_counts") == [DEVELOPMENT_ROWS_PER_CLASS] * CLASS_COUNT, "V2 source balance changed")
    _require(identity.get("selected_source_indices_sha256") == "8e6859cbe0ad10090b1d5d7c55de28232f9db30c45093078dbf29f8545473a0e", "V2 source index identity changed")
    rows = source.get("rows")
    _require(isinstance(rows, list) and len(rows) == DEVELOPMENT_ROWS, "V2 source rows changed")
    source_indices = [int(_mapping(row, "V2 source row")["source_index"]) for row in rows]
    _require(len(set(source_indices)) == DEVELOPMENT_ROWS, "V2 source rows are not unique")
    _require(tensor_sha256(torch.tensor(source_indices, dtype=torch.int64)) == identity["selected_source_indices_sha256"], "V2 source row order changed")

    q_manifest = _read_json_exact(repo_root, V2_Q_MANIFEST_PATH, V2_Q_MANIFEST_SHA256, "V2 q materialization manifest")
    _require(q_manifest.get("schema") == "n3_development_bank_v2_q_materialization_manifest.v1", "V2 q manifest schema changed")
    _require(q_manifest.get("status") == "VALID_COMPLETED", "V2 q materialization is not complete")
    source_ref = _mapping(q_manifest.get("source_manifest"), "V2 q source reference")
    _require(source_ref.get("sha256") == V2_SOURCE_MANIFEST_SHA256, "V2 q source binding changed")
    q_artifact = _mapping(q_manifest.get("q_artifact"), "V2 q artifact metadata")
    _require(q_artifact.get("sha256") == V2_Q_ARTIFACT_SHA256, "V2 q artifact identity changed")
    _require(q_artifact.get("q_tensor_sha256") == V2_Q_TENSOR_SHA256, "V2 q tensor identity changed")
    _require(q_artifact.get("shape") == [DEVELOPMENT_ROWS, DIMENSION], "V2 q shape changed")
    _require(q_artifact.get("dtype") == "torch.float64", "V2 q dtype changed")
    execution = _mapping(q_manifest.get("execution"), "V2 q execution metadata")
    _require(execution.get("sealed_confirmation_rows_read") is False, "V2 q authority read sealed rows")
    _require(execution.get("sealed_confirmation_values_loaded") is False, "V2 q authority loaded sealed values")
    _require(q_manifest.get("sealed_confirmation_consumed") is False, "V2 q authority consumed sealed confirmation")
    return {
        "source_manifest": {
            "path": _relative(repo_root, repo_root / V2_SOURCE_MANIFEST_PATH),
            "sha256": V2_SOURCE_MANIFEST_SHA256,
            "semantic_authority_id": source["semantic_authority_id"],
            "bank_semantic_sha256": source.get("bank_semantic_sha256"),
            "row_count": DEVELOPMENT_ROWS,
            "rows_per_class": DEVELOPMENT_ROWS_PER_CLASS,
            "source_indices_sha256": identity["selected_source_indices_sha256"],
        },
        "q_manifest": {
            "path": _relative(repo_root, repo_root / V2_Q_MANIFEST_PATH),
            "sha256": V2_Q_MANIFEST_SHA256,
            "schema": q_manifest["schema"],
            "q_artifact_path": _relative(repo_root, repo_root / V2_Q_MANIFEST_PATH.parent / q_artifact["path"]),
            "q_artifact_sha256": q_artifact["sha256"],
            "q_tensor_sha256": q_artifact["q_tensor_sha256"],
            "shape": q_artifact["shape"],
            "dtype": q_artifact["dtype"],
            "values_loaded_by_definition": False,
        },
        "frozen_ols_authority": dict(_mapping(q_manifest.get("frozen_ols_authority"), "V2 frozen OLS authority")),
    }, set(source_indices)


def _validate_sealed_reservation(repo_root: Path, v2_indices: set[int]) -> dict[str, Any]:
    manifest = _read_json_exact(repo_root, SEALED_MANIFEST_PATH, SEALED_MANIFEST_SHA256, "sealed reservation manifest")
    _require(manifest.get("schema") == "n3_sealed_confirmation_source_manifest.v1", "sealed reservation schema changed")
    _require(manifest.get("task_id") == "DEFINE_N3_SEALED_CONFIRMATION_SOURCE_V1", "sealed reservation task changed")
    _require(manifest.get("semantic_authority_id") == SEALED_SEMANTIC_AUTHORITY_ID, "sealed semantic authority changed")
    _require(manifest.get("semantic_confirmation_id") == SEALED_SEMANTIC_CONFIRMATION_ID, "sealed semantic confirmation identity changed")
    _require(manifest.get("status") == "RESERVED_NOT_MATERIALIZED", "sealed reservation is no longer untouched")
    source = _mapping(manifest.get("source"), "sealed source identity")
    _require(source.get("dataset") == "mnist_train" and source.get("split_identity") == "mnist_train_indices_0_59999", "sealed source dataset/split changed")
    selection = _mapping(manifest.get("selection"), "sealed selection")
    indices = [int(value) for value in selection.get("source_indices", [])]
    labels = [int(value) for value in selection.get("labels", [])]
    _require(len(indices) == SEALED_ROWS and len(labels) == SEALED_ROWS, "sealed reservation row count changed")
    _require(len(set(indices)) == SEALED_ROWS, "sealed reservation rows are not unique")
    _require(torch.bincount(torch.tensor(labels, dtype=torch.int64), minlength=CLASS_COUNT).tolist() == [SEALED_ROWS_PER_CLASS] * CLASS_COUNT, "sealed reservation is not balanced")
    index_sha256 = tensor_sha256(torch.tensor(indices, dtype=torch.int64))
    label_sha256 = tensor_sha256(torch.tensor(labels, dtype=torch.int64))
    _require(index_sha256 == "60baa34331dad647ff6ed4944a9c5ebdce6ad8ad2bb68219e64701696b0ceeb1", "sealed source-index identity changed")
    _require(label_sha256 == "c8bacf260ae894e7da623586b9e04c67d95f628c16f52766cee0994ca2ceac19", "sealed label identity changed")
    _require(selection.get("source_index_sha256") == index_sha256 and selection.get("label_sha256") == label_sha256, "sealed selection hashes are stale")
    _require(selection.get("selection_order_frozen") is True, "sealed selection order is not frozen")
    _require(selection.get("q_values_observed") is False and selection.get("ols_values_observed") is False, "sealed q/OLS values were observed")
    _require(selection.get("evaluator_observed") is False and selection.get("image_quality_observed") is False, "sealed evaluator/image values were observed")
    _require(not v2_indices.intersection(indices), "sealed reservation overlaps N3 Development Bank V2")

    protected = manifest.get("protected_identities_checked")
    _require(isinstance(protected, list), "sealed protected identity registry is missing")
    by_name = {str(_mapping(item, "protected identity")["canonical_name"]): _mapping(item, "protected identity") for item in protected}
    for name in ("N3_DEVELOPMENT_BANK", "S2_VALIDATION", "S1_CONFIRMATION", "S1_SELECTION"):
        _require(name in by_name, f"sealed protected identity is missing: {name}")
        _require(by_name[name].get("overlap_count_with_reserved_source") == 0, f"sealed overlap is nonzero: {name}")
    firewall = _mapping(manifest.get("firewall"), "sealed firewall")
    _require(firewall.get("source_index_overlap_count") == 0, "sealed source overlap firewall changed")
    pairwise = _mapping(firewall.get("pairwise_protected_source_overlap_counts"), "sealed pairwise firewall")
    _require(pairwise and all(value == 0 for value in pairwise.values()), "sealed pairwise overlap firewall changed")
    materialization = _mapping(manifest.get("materialization"), "sealed materialization state")
    for key in ("ols_states_materialized", "complex_ols_values_loaded", "optical_forward_executed", "evaluator_executed", "producer_candidate_consumed"):
        _require(materialization.get(key) is False, f"sealed materialization state changed: {key}")

    ordered_rows = [
        {"position": position, "source_index": index, "label": label}
        for position, (index, label) in enumerate(zip(indices, labels, strict=True))
    ]
    row_identity_sha256 = canonical_sha256(ordered_rows)
    return {
        "reservation_manifest": {
            "path": _relative(repo_root, repo_root / SEALED_MANIFEST_PATH),
            "sha256": SEALED_MANIFEST_SHA256,
            "schema": manifest["schema"],
            "semantic_authority_id": SEALED_SEMANTIC_AUTHORITY_ID,
            "semantic_confirmation_id": SEALED_SEMANTIC_CONFIRMATION_ID,
        },
        "source": {
            "dataset": source["dataset"],
            "split_identity": source["split_identity"],
            "source_count": source["count"],
            "source_labels_tensor_sha256": source["labels_tensor_sha256"],
        },
        "membership": {
            "row_count": SEALED_ROWS,
            "rows_per_class": SEALED_ROWS_PER_CLASS,
            "class_balance": [SEALED_ROWS_PER_CLASS] * CLASS_COUNT,
            "source_indices": indices,
            "labels_for_identity_only": labels,
            "source_index_sha256": index_sha256,
            "label_sha256": label_sha256,
            "ordered_row_identity": ordered_rows,
            "ordered_row_identity_sha256": row_identity_sha256,
            "ordering": "exact selection.source_indices order; q row position equals membership.position",
        },
        "overlap_verification": {
            "n3_development_bank_v2_overlap_count": 0,
            "protected_identity_overlap_counts": {
                name: by_name[name].get("overlap_count_with_reserved_source") for name in sorted(by_name)
            },
            "manifest_source_index_overlap_count": firewall["source_index_overlap_count"],
            "pairwise_protected_overlap_counts": dict(pairwise),
            "class_labels_used_for": ["manifest identity and balance verification only"],
            "class_labels_not_used_for": ["producer conditioning", "class-specific state construction", "threshold tuning", "class-conditioned decisions"],
        },
        "protected_content_observed": False,
    }


def _validate_q_materialization(repo_root: Path, v2: Mapping[str, Any]) -> dict[str, Any]:
    q_impl = _authority_ref(repo_root, Path("scripts/research/materialize_n3_development_bank_v2_q.py"), "frozen source-to-q materialization implementation")
    replay_impl = _authority_ref(repo_root, Q_REPLAY_IMPLEMENTATION_PATH, "frozen encoder-prefix implementation")
    _require(q_impl["sha256"] == file_sha256(repo_root / Path("scripts/research/materialize_n3_development_bank_v2_q.py")), "q implementation digest is unstable")
    _require(q_materialization.DIMENSION == DIMENSION and q_materialization.BATCH_SIZE == 20, "q materialization dimensions changed")
    _require(q_materialization.RAW_IMAGES_PATH.as_posix() == "data/MNIST/raw/train-images-idx3-ubyte", "q raw-image authority changed")
    _require(q_materialization.RAW_IMAGES_SHA256 == "ba891046e6505d7aadcbbe25680a0738ad16aec93bde7f9b65e87a2fc25776db", "q raw-image digest changed")
    _require(q_materialization.MRA_CONFIG_PATH.as_posix() == "configs/research/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1.yaml", "q MRA config path changed")
    _require(q_materialization.FORMAL_CONFIG_PATH.as_posix() == "configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml", "q formal config path changed")
    _require(q_materialization.CHECKPOINT_PATH.as_posix() == "outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt", "q checkpoint path changed")
    _require(v2["q_manifest"]["shape"] == [DEVELOPMENT_ROWS, DIMENSION], "V2 q authority shape is inconsistent")
    return {
        "development_q_authority": {
            "manifest": dict(v2["q_manifest"]),
            "values_loaded_by_definition": False,
        },
        "implementation": q_impl,
        "encoder_prefix_implementation": replay_impl,
        "future_source_to_q": {
            "implementation_path": q_impl["path"],
            "implementation_sha256": q_impl["sha256"],
            "source_dataset": "mnist_train",
            "source_split": "mnist_train_indices_0_59999",
            "raw_image_path": q_materialization.RAW_IMAGES_PATH.as_posix(),
            "raw_image_sha256": q_materialization.RAW_IMAGES_SHA256,
            "raw_image_decode": "IDX3-UBYTE header (2051, 60000, 28, 28); selected rows decoded as uint8/255.0 to contiguous torch.float32 [N,1,28,28]",
            "row_order": "read exact sealed membership source_indices order; sorted file seeks may not change output positions",
            "batch_size": q_materialization.BATCH_SIZE,
            "source_render": "ldm_project.v2.application.research.optical_manifold.render_intensity_batch(images)",
            "encoder_prefix": "replay_scaled_chen_repaired_coordinate_mra._encoded_complex_prefix(model, rendered.amplitude)",
            "fixed_ols_authority": dict(v2["frozen_ols_authority"]),
            "normalization": "normalized = coefficients / sqrt(raw_port_mean); q = abs(normalized)",
            "q_definition": "R2_AMPLITUDE = abs(c/sqrt(s)) = abs(R3_COMPLEX)",
            "source_dtype": "torch.complex64",
            "q_dtype": "torch.float64",
            "q_shape": [SEALED_ROWS, DIMENSION],
            "q_invariants": ["finite", "nonnegative", "zero allowed", "no clipping", "no samplewise normalization"],
            "failure_behavior": "fail closed on source/header/hash/shape/dtype/nonfinite/negative/mapping mismatch; no partial public artifact",
            "development_q_definition_unchanged": True,
        },
        "future_artifact": {
            "q_artifact_schema": Q_ARTIFACT_SCHEMA,
            "q_manifest_schema": Q_MANIFEST_SCHEMA,
            "q_artifact_path": _relative(repo_root, repo_root / SEALED_Q_ARTIFACT_PATH),
            "q_manifest_path": _relative(repo_root, repo_root / SEALED_Q_MANIFEST_PATH),
            "artifact_sha256": None,
            "q_tensor_sha256": None,
            "must_bind_reservation_manifest_sha256": SEALED_MANIFEST_SHA256,
            "must_bind_ordered_row_identity_sha256_field": "membership.ordered_row_identity_sha256",
            "must_be_frozen_before_confirmation_statistic": True,
        },
    }


def _validate_v3_authority(repo_root: Path) -> dict[str, Any]:
    calibration = _read_json_exact(repo_root, V3_CALIBRATION_REPORT_PATH, V3_CALIBRATION_REPORT_SHA256, "V3 calibration report")
    _require(calibration.get("status") == "valid_completed" and calibration.get("result") == "VALID_COMPLETED", "V3 calibration report is incomplete")
    _require(calibration.get("classification") == "N3_Q_COMPATIBILITY_GUARDRAIL_CALIBRATION_READY", "V3 calibration classification changed")
    evaluator = _read_json_exact(repo_root, V3_EVALUATOR_PATH, V3_EVALUATOR_SHA256, "V3 evaluator authority")
    _require(evaluator.get("schema") == "n3_q_compatibility_evaluator_v3.v1", "V3 evaluator schema changed")
    primary = _mapping(evaluator.get("primary"), "V3 primary metric")
    _require(primary.get("metric") == PRIMARY_METRIC, "V3 primary metric changed")
    _require(primary.get("role") == "HARD_PRIMARY_UNCHANGED", "V3 primary role changed")
    _require(primary.get("implementation_unchanged") is True and primary.get("sample_size_semantics_unchanged") is True, "V3 primary implementation changed")
    _require(evaluator.get("hard_guardrails") == [PRIMARY_METRIC], "V3 hard metric set changed")
    _require(evaluator.get("descriptive_only_guardrails") == list(DESCRIPTIVE_ONLY_METRICS), "V3 descriptive metric set changed")
    formulas = _mapping(evaluator.get("metric_formulas"), "V3 metric formulas")
    _require(formulas.get("pair_metric") == "v2_ceiling._pair_metrics_raw_q", "V3 pair metric binding changed")
    _require(formulas.get("data_vs_model_aggregation") == "gaussian._mean_metrics", "V3 data/model aggregation changed")
    sample_size = _mapping(evaluator.get("sample_size_semantics"), "V3 sample-size semantics")
    _require(sample_size.get("data_vs_model_model_bank") == MODEL_BANK_SIZE, "V3 model bank size changed")
    _require(sample_size.get("data_vs_model_replicates") == MODEL_BANK_COUNT, "V3 model bank count changed")
    _require(sample_size.get("pseudo_data") == 1000, "V3 development pseudo-data semantics changed")
    return {
        "calibration_report": {
            "path": _relative(repo_root, repo_root / V3_CALIBRATION_REPORT_PATH),
            "sha256": V3_CALIBRATION_REPORT_SHA256,
            "classification": calibration["classification"],
        },
        "evaluator": {
            "path": _relative(repo_root, repo_root / V3_EVALUATOR_PATH),
            "sha256": V3_EVALUATOR_SHA256,
            "schema": evaluator["schema"],
            "primary_metric": PRIMARY_METRIC,
            "hard_metrics": [PRIMARY_METRIC],
            "descriptive_only_metrics": list(DESCRIPTIVE_ONLY_METRICS),
            "development_data_vs_model": {"data_rows": 1000, "model_bank_count": MODEL_BANK_COUNT, "model_bank_size": MODEL_BANK_SIZE},
            "q95_threshold_reused": False,
        },
        "implementations": {
            "energy_distance": _authority_ref(repo_root, V3_METRIC_IMPLEMENTATION_PATH, "V3 validated energy-distance implementation", V3_METRIC_IMPLEMENTATION_SHA256),
            "v3_evaluator": _authority_ref(repo_root, V3_EVALUATOR_IMPLEMENTATION_PATH, "V3 evaluator implementation", V3_EVALUATOR_IMPLEMENTATION_SHA256),
            "model_bank_aggregation": _authority_ref(repo_root, V3_AGGREGATION_IMPLEMENTATION_PATH, "V3 model-bank aggregation implementation", V3_AGGREGATION_IMPLEMENTATION_SHA256),
        },
        "future_confirmation_sample_size_change": {
            "development_data_rows": 1000,
            "sealed_data_rows": SEALED_ROWS,
            "threshold_calibration_must_match_sealed_data_rows": True,
            "development_q95_must_not_be_used": True,
        },
    }


def _seed(stream: str, trial: int | None, bank_index: int | None) -> int:
    trial_token = "none" if trial is None else str(trial)
    bank_token = "none" if bank_index is None else str(bank_index)
    material = f"{SEED_DERIVATION_ROOT}|{stream}|trial={trial_token}|bank_index={bank_token}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") & MAX_SIGNED_63


def _contract_totality_checks() -> dict[str, Any]:
    _require(SEALED_ROWS == 500 and SEALED_ROWS_PER_CLASS == 50, "sealed sample size changed")
    _require(MODEL_BANK_COUNT == 4 and MODEL_BANK_SIZE == 200, "model-bank dimensions changed")
    _require(CALIBRATION_TRIAL_COUNT == 999, "calibration trial count changed")
    _require(ALPHA == 0.05, "alpha changed")
    _require(ALL_V3_METRICS[0] == PRIMARY_METRIC and tuple(ALL_V3_METRICS[1:]) == DESCRIPTIVE_ONLY_METRICS, "metric partition changed")
    _require(len({"CALIBRATION_PSEUDO_DATA", "CALIBRATION_MODEL_BANK", "OBSERVED_MODEL_BANK"}) == 3, "RNG namespaces are not disjoint")
    observed_seeds = [_seed("OBSERVED_MODEL_BANK", None, bank) for bank in range(MODEL_BANK_COUNT)]
    _require(len(set(observed_seeds)) == MODEL_BANK_COUNT, "observed model-bank seeds are not unique")
    _require(all(0 <= seed <= MAX_SIGNED_63 for seed in observed_seeds), "observed seed is outside signed 63-bit range")
    outcome_names = {
        "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED",
        "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_FAILED",
        "SEALED_CONFIRMATION_INVALID_CONSUMED",
    }
    _require(len(outcome_names) == 3, "confirmation outcome mapping is not total")
    return {
        "status": "passed",
        "sealed_rows": SEALED_ROWS,
        "model_bank_count": MODEL_BANK_COUNT,
        "model_bank_size": MODEL_BANK_SIZE,
        "calibration_trials": CALIBRATION_TRIAL_COUNT,
        "alpha": ALPHA,
        "observed_model_bank_seeds": observed_seeds,
        "q_values_loaded": False,
        "sealed_values_loaded": False,
        "fit_executed": False,
        "sampling_executed": False,
        "contract_totality": "passed",
    }


def _build_payload(
    repo_root: Path,
    git: Mapping[str, Any],
    current: Mapping[str, Any],
    freeze: Mapping[str, Any],
    producer: Mapping[str, Any],
    v2: Mapping[str, Any],
    sealed: Mapping[str, Any],
    q_semantics: Mapping[str, Any],
    v3: Mapping[str, Any],
    totality: Mapping[str, Any],
) -> dict[str, Any]:
    definition_impl = _authority_ref(
        repo_root,
        Path("scripts/research/define_n3_selected_producer_sealed_confirmation_contract_v1.py"),
        "this sealed-confirmation contract definition consumer",
    )
    future_consumer = {
        "path": _relative(repo_root, repo_root / CONFIRMATION_CONSUMER_PATH),
        "sha256": None,
        "schema": "n3_selected_producer_sealed_confirmation_execution_consumer.v1",
        "required_before_arm": True,
        "required_public_action": "execute one armed confirmation and write one write-once lineage",
        "implementation_frozen_in_this_task": False,
    }
    calibration_seed_rule = (
        "seed=int.from_bytes(sha256(SEED_DERIVATION_ROOT|stream|trial=<index>|"
        "bank_index=<index>)[:8], 'big') & ((1<<63)-1); trial indices=0..998; "
        "bank indices=0..3; no retry"
    )
    observed_seeds = [_seed("OBSERVED_MODEL_BANK", None, bank) for bank in range(MODEL_BANK_COUNT)]
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
        "git": dict(git),
        "authorities": {
            "current_state": dict(current),
            "selected_producer_freeze_contract": dict(freeze),
            "selected_producer_full_fit": dict(producer),
            "sealed_reservation": dict(sealed["reservation_manifest"]),
            "v2_development": dict(v2),
            "v3": dict(v3),
            "definition_implementation": definition_impl,
        },
        "producer": {
            "family_id": PRODUCER_FAMILY_ID,
            "candidate_id": PRODUCER_CANDIDATE_ID,
            "semantic_authority_id": PRODUCER_STATE_SEMANTIC_ID,
            "state_path": producer["state"]["path"],
            "state_sha256": PRODUCER_STATE_SHA256,
            "lineage_path": producer["lineage"]["path"],
            "lineage_sha256": PRODUCER_LINEAGE_SHA256,
            "standalone_sampler": {
                "state_only": True,
                "sampling_namespace": PRODUCER_SAMPLING_NAMESPACE,
                "requested_sample_count_minimum": 1,
                "explicit_seed_required": True,
                "labels_as_inputs": False,
            },
        },
        "sealed_source": {
            "reservation_authority": SEALED_SEMANTIC_AUTHORITY_ID,
            "manifest_path": sealed["reservation_manifest"]["path"],
            "manifest_sha256": SEALED_MANIFEST_SHA256,
            "semantic_confirmation_id": SEALED_SEMANTIC_CONFIRMATION_ID,
            "expected_status_on_entry": "RESERVED_NOT_MATERIALIZED",
            "expected_consumed_on_entry": False,
            "dataset": sealed["source"]["dataset"],
            "split_identity": sealed["source"]["split_identity"],
            "row_count": SEALED_ROWS,
            "rows_per_class": SEALED_ROWS_PER_CLASS,
            "membership": sealed["membership"],
            "overlap_verification": sealed["overlap_verification"],
            "content_observed_by_definition": sealed["protected_content_observed"],
            "substitution_or_resampling": False,
        },
        "q_materialization": dict(q_semantics),
        "confirmation_statistic": {
            "data_label": "SEALED_DATA",
            "primary_metric": PRIMARY_METRIC,
            "statistic_symbol": "T",
            "definition": "T = arithmetic mean over four model banks of V3 validated energy_distance(sealed_q[500], model_bank[200])",
            "pair_metric_implementation": {
                "path": v3["implementations"]["energy_distance"]["path"],
                "sha256": v3["implementations"]["energy_distance"]["sha256"],
                "function": "_energy_distance",
                "formula": "2*mean(cdist(DATA,MODEL)) - mean(cdist(DATA,DATA)) - mean(cdist(MODEL,MODEL)); clamp only values in [-1e-10,0] to 0",
                "input_convention": "CPU contiguous torch.float64, two-dimensional, same q dimension, biased cdist means including diagonal within terms",
            },
            "model_bank_aggregation": {
                "path": v3["implementations"]["model_bank_aggregation"]["path"],
                "sha256": v3["implementations"]["model_bank_aggregation"]["sha256"],
                "function": "_mean_metrics",
                "bank_count": MODEL_BANK_COUNT,
                "bank_size": MODEL_BANK_SIZE,
                "aggregation": "unweighted arithmetic mean of the four scalar energy_distance values",
            },
            "data_shape": [SEALED_ROWS, DIMENSION],
            "model_bank_shape": [MODEL_BANK_SIZE, DIMENSION],
            "labels_used": False,
            "development_q95_reused": False,
            "threshold_source": "confirmation-sample-size-matched model-self Monte-Carlo null",
            "secondary_metrics": {
                "names": list(DESCRIPTIVE_ONLY_METRICS),
                "role": "DESCRIPTIVE_ONLY",
                "may_be_reported_after_primary_decision": True,
                "may_change_decision": False,
                "new_metrics_after_exposure": False,
            },
        },
        "model_self_calibration": {
            "scope": "MODEL_ONLY_BEFORE_SEALED_EXPOSURE",
            "artifact_schema": CALIBRATION_SCHEMA,
            "artifact_path": _relative(repo_root, repo_root / CALIBRATION_ARTIFACT_PATH),
            "artifact_sha256": None,
            "producer_state_sha256": PRODUCER_STATE_SHA256,
            "pseudo_data_rows": SEALED_ROWS,
            "model_bank_count": MODEL_BANK_COUNT,
            "model_bank_size": MODEL_BANK_SIZE,
            "trials": CALIBRATION_TRIAL_COUNT,
            "trial_indices": [0, CALIBRATION_TRIAL_COUNT - 1],
            "all_trial_indices_inclusive": [0, CALIBRATION_TRIAL_COUNT - 1],
            "statistic": "exact confirmation T",
            "rng": {
                "sampling_interface_namespace": PRODUCER_SAMPLING_NAMESPACE,
                "seed_derivation_root": SEED_DERIVATION_ROOT,
                "streams": {
                    "pseudo_data": "CALIBRATION_PSEUDO_DATA",
                    "model_bank": "CALIBRATION_MODEL_BANK",
                },
                "seed_rule": calibration_seed_rule,
                "no_early_stop": True,
                "no_seed_retry": True,
                "best_of_n": False,
                "bank_index_domain": [0, MODEL_BANK_COUNT - 1],
            },
            "complete_before_arm": True,
            "values_frozen_before_sealed_exposure": True,
        },
        "observed_model_banks": {
            "scope": "MODEL_ONLY_BEFORE_SEALED_EXPOSURE",
            "artifact_schema": MODEL_BANK_SCHEMA,
            "artifact_path": _relative(repo_root, repo_root / MODEL_BANK_ARTIFACT_PATH),
            "artifact_sha256": None,
            "producer_state_sha256": PRODUCER_STATE_SHA256,
            "bank_count": MODEL_BANK_COUNT,
            "bank_size": MODEL_BANK_SIZE,
            "dtype": "torch.float64",
            "shape_per_bank": [MODEL_BANK_SIZE, DIMENSION],
            "sampling_implementation": "frozen producer standalone sampler; state only",
            "rng": {
                "sampling_interface_namespace": PRODUCER_SAMPLING_NAMESPACE,
                "stream": "OBSERVED_MODEL_BANK",
                "seed_derivation_root": SEED_DERIVATION_ROOT,
                "seed_rule": calibration_seed_rule,
                "bank_indices": list(range(MODEL_BANK_COUNT)),
                "seeds": observed_seeds,
                "no_replacement_bank": True,
                "no_seed_retry": True,
                "no_post_exposure_selection": True,
            },
            "complete_before_arm": True,
            "materialized_before_sealed_exposure": True,
        },
        "monte_carlo_decision": {
            "B": CALIBRATION_TRIAL_COUNT,
            "alpha": ALPHA,
            "p_value": "p_MC=(1+count(T_b >= T_sealed))/(B+1)",
            "inequality": "greater-than-or-equal, including ties",
            "pass_if": "p_MC >= 0.05",
            "fail_if": "p_MC < 0.05",
            "pass_classification": "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED",
            "fail_classification": "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_FAILED",
            "no_threshold_reuse": True,
            "no_early_stop": True,
            "no_retry": True,
            "no_best_of_n": True,
            "one_decision_only": True,
        },
        "pre_sealed_arming": {
            "state": "SEALED_CONFIRMATION_NOT_ARMED",
            "must_all_pass_before_first_protected_content_read": True,
            "conditions": [
                {"id": 1, "condition": "contract_sha256_verified", "definition_frozen": True},
                {"id": 2, "condition": "producer_state_sha256_and_semantic_authority_verified", "definition_frozen": True},
                {"id": 3, "condition": "source_to_q_implementation_identity_verified", "definition_frozen": True},
                {"id": 4, "condition": "confirmation_consumer_implementation_frozen", "definition_frozen": False, "future_required": True},
                {"id": 5, "condition": "targeted_consumer_tests_pass", "definition_frozen": False, "future_required": True},
                {"id": 6, "condition": "B_999_model_self_calibration_complete", "definition_frozen": True},
                {"id": 7, "condition": "calibration_artifact_and_sha_frozen", "definition_frozen": True},
                {"id": 8, "condition": "all_observed_model_banks_and_seeds_frozen", "definition_frozen": True},
                {"id": 9, "condition": "Monte_Carlo_rule_frozen", "definition_frozen": True},
                {"id": 10, "condition": "sealed_reservation_identity_verified_without_content_read", "definition_frozen": True},
                {"id": 11, "condition": "no_unresolved_numerical_or_lineage_issue", "definition_frozen": True},
            ],
            "armed_transition": "SEALED_CONFIRMATION_ARMED",
            "sealed_content_read_allowed_only_after_all_conditions": True,
        },
        "consumption_semantics": {
            "marker_schema": CONSUMPTION_MARKER_SCHEMA,
            "marker_path": _relative(repo_root, repo_root / CONSUMPTION_MARKER_PATH),
            "execution_lineage_path": _relative(repo_root, repo_root / EXECUTION_LINEAGE_PATH),
            "transition_point": "atomically publish one write-once consumed=true marker immediately before the first read/materialization of any protected sealed source content; if marker publication fails, do not read content",
            "pre_exposure_failure": {
                "consumed": False,
                "classification": "BLOCKED",
                "source_status_remains": "RESERVED_NOT_MATERIALIZED",
                "corrected_execution_allowed": True,
                "sealed_content_read": False,
            },
            "post_exposure_failure": {
                "consumed": True,
                "classification": "SEALED_CONFIRMATION_INVALID_CONSUMED",
                "reset_to_false_forbidden": True,
                "retry_same_reservation_forbidden": True,
                "producer_tuning_from_exposed_data_forbidden": True,
            },
            "valid_outcomes": [
                "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED",
                "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_FAILED",
                "SEALED_CONFIRMATION_INVALID_CONSUMED",
            ],
            "single_use": True,
            "no_second_model_bank_or_q_transform": True,
            "execution_lineage_schema": LINEAGE_SCHEMA,
        },
        "outcome_claim_boundary": {
            "pass_supports_only": ["N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED", "non-rejection of compatibility under this exact contract"],
            "fail_supports_only": ["N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_FAILED", "independent confirmation not obtained"],
            "not_established": [
                "population distribution equality",
                "OOD generalization",
                "complete complex OLS validity",
                "amplitude-phase joint compatibility",
                "semantic controllability",
                "image-generation readiness",
                "optical realizability",
                "Chen-style full OLS reproduction",
                "training authorization",
                "formal promotion",
            ],
        },
        "future_confirmation_consumer": future_consumer,
        "governance": {
            "selected_producer": PRODUCER_FAMILY_ID,
            "selected_producer_candidate": PRODUCER_CANDIDATE_ID,
            "frozen_producer_state": PRODUCER_STATE_SEMANTIC_ID,
            "n3_amplitude_producer_confirmation": None,
            "sealed_confirmation_status": "RESERVED_NOT_MATERIALIZED",
            "sealed_confirmation_consumed": False,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "selection_scope": "DEVELOPMENT_ONLY_PLUS_UNCONSUMED_CONFIRMATION_CONTRACT",
        },
        "firewalls": {
            "sealed_content_read": False,
            "sealed_q_loaded": False,
            "sealed_images_loaded": False,
            "sealed_labels_read_beyond_manifest": False,
            "development_q_artifact_loaded": False,
            "producer_refit": False,
            "producer_modified": False,
            "new_producer_family": False,
            "new_evaluator": False,
            "heldout_tuning": False,
            "class_conditioned_sampling": False,
            "image_space_evaluation": False,
            "optical_forward": False,
            "cuda": False,
            "training": False,
            "formal_promotion": False,
        },
        "contract_totality": dict(totality),
        "next_action": "EXECUTE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_V1",
    }


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to reuse write-once contract output: {output_dir}")
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the authority branch")
    _require(git["head"] == EXPECTED_STARTING_HEAD, "current committed HEAD is not the frozen authority base")
    totality = _contract_totality_checks()
    current = _validate_current_state(repo_root)
    freeze = _validate_freeze_contract(repo_root)
    producer = _validate_producer_state(repo_root)
    v2, v2_indices = _validate_v2_authority(repo_root)
    sealed = _validate_sealed_reservation(repo_root, v2_indices)
    q_semantics = _validate_q_materialization(repo_root, v2)
    v3 = _validate_v3_authority(repo_root)
    payload = _build_payload(repo_root, git, current, freeze, producer, v2, sealed, q_semantics, v3, totality)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(
            temporary / CONTRACT_NAME,
            payload,
            artifact_kind="N3 selected producer sealed confirmation contract",
        )
        if output_dir.exists():
            raise FileExistsError(f"contract output appeared during definition: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    path = output_dir / CONTRACT_NAME
    return {
        "status": "valid_completed",
        "result": "READY",
        "classification": READY_CLASSIFICATION,
        "contract": _relative(repo_root, path),
        "contract_sha256": file_sha256(path),
        "producer_semantic_authority": PRODUCER_STATE_SEMANTIC_ID,
        "producer_state_sha256": PRODUCER_STATE_SHA256,
        "producer_lineage_sha256": PRODUCER_LINEAGE_SHA256,
        "sealed_reservation_authority": SEALED_SEMANTIC_AUTHORITY_ID,
        "sealed_manifest_sha256": SEALED_MANIFEST_SHA256,
        "sealed_rows": SEALED_ROWS,
        "sealed_rows_per_class": SEALED_ROWS_PER_CLASS,
        "sealed_content_loaded": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "next_action": "EXECUTE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_V1",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    try:
        print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))
        return 0
    except InvalidHarnessError as exc:
        print(json.dumps({"status": "invalid", "result": "INVALID_HARNESS", "reason": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
