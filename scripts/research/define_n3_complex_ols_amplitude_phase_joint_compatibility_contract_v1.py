"""Freeze the N3 complete-complex-OLS amplitude/phase compatibility contract.

This is a definition-only consumer.  It binds the existing V2 source/q/fold
authorities and the historical PROJECT_COMPLEX_OLS authority, then freezes one
minimal phase-inheritance candidate and one non-selectable complex-row control.
It never loads q values, complex OLS values, sealed content, or the consumed
sealed q artifact, and it does not execute the scientific evaluation.
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

from ldm_project.v2.evidence import atomic_write_new_json, canonical_sha256, file_sha256


TASK_ID = "DEFINE_N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_CONTRACT_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "n3_complex_ols_amplitude_phase_joint_compatibility_contract.v1"
READY_CLASSIFICATION = "N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_CONTRACT_V1_READY"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"

OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "n3_complex_ols_amplitude_phase_joint_compatibility_contract_v1_attempt_01"
)
CONTRACT_NAME = "n3_complex_ols_amplitude_phase_joint_compatibility_contract.v1.json"

CLASS_COUNT = 10
DIMENSION = 100
DEVELOPMENT_ROWS = 5000
DEVELOPMENT_ROWS_PER_CLASS = 500
FOLD_COUNT = 5
FIT_ROWS_PER_FOLD = 4000
FIT_ROWS_PER_CLASS_PER_FOLD = 400
HELDOUT_ROWS_PER_FOLD = 1000
HELDOUT_ROWS_PER_CLASS_PER_FOLD = 100
MODEL_BANK_COUNT = 4
MODEL_BANK_SIZE = 200
CALIBRATION_TRIAL_COUNT = 999
FAMILY_ALPHA = 0.05
PER_FOLD_ALPHA = 0.01
PRIMARY_METRIC = "energy_distance"
RNG_NAMESPACE = "N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_V1_CPU"
MAX_SIGNED_63 = (1 << 63) - 1

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
V2_FOLD_MANIFEST_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_five_fold_protocol_v1_attempt_01/"
    "n3_development_bank_v2_fold_manifest.v1.json"
)
V2_FOLD_MANIFEST_SHA256 = "4898a0e477d583b9a9e60c83238e6d8524d43967efe24277a952089e8adc447e"
V2_ASSIGNMENT_SHA256 = "43ee0ce28e143fffdf2bcac98394625ccd766166ea717f23db66ed8770ef18a9"
V2_CEILING_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_empirical_ceiling_v1_attempt_01/"
    "n3_development_bank_v2_empirical_ceiling_report.v1.json"
)
V2_CEILING_SHA256 = "731cf826dc081e7cc5629790ba2fb74946f7cc41f1a7f0035bf3a3ca96a31826"
V2_BANK_SEMANTIC_SHA256 = "bdb5a11c6800b4a5d0a994d5448bbf7de3e29c7bf8c0709f7d34be1cb31c91aa"
V2_ROW_MANIFEST_SHA256 = "ba3149357dc90f14ad692a486593ce944d48453f2b561ad10a9323b9fe7fb957"
V2_SOURCE_INDICES_SHA256 = "8e6859cbe0ad10090b1d5d7c55de28232f9db30c45093078dbf29f8545473a0e"
V2_LABELS_SHA256 = "d28f06d6ecde9d08797f954e60041f79b512e2ad61480e21d5702cb841fabef3"
V2_SOURCE_ROW_IDENTITY_SHA256 = "4a570e346aab909eb3be72aaef74fef27d1735a44a0eb2453ee45b727746d664"
V2_FOLD_ROWS_SHA256 = "13cb661928be5cc1f86c7d8ec6c1ec454aa3d98f77092985d4388a4451ded5a1"

OLS_SOURCE_MANIFEST_PATH = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_ols_producer_data_authority_v1_attempt_01/"
    "ols_producer_source_manifest.v1.json"
)
OLS_SOURCE_MANIFEST_SHA256 = "d07fc3fa927796e97254fa09a8c0eceb18d6555a99590db43217a685bf0d1687"
OLS_PARTITION_MANIFEST_PATH = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_ols_producer_data_authority_v1_attempt_01/"
    "ols_producer_partition_manifest.v1.json"
)
OLS_PARTITION_MANIFEST_SHA256 = "409b598aacae921da631adea553b339b56560231f9cb2cf75ce8d7e5b439242b"
OLS_FIREWALL_MANIFEST_PATH = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_ols_producer_data_authority_v1_attempt_01/"
    "ols_producer_data_firewall_manifest.v1.json"
)
OLS_FIREWALL_MANIFEST_SHA256 = "b711e9d3443f3a86c803ff1f8cae1ff266634439ebe0c0d31c04e807c3841837"
OLS_SOURCE_ARTIFACT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01/"
    "ols_embeddings.pt"
)
OLS_SOURCE_ARTIFACT_SHA256 = "9375b08597f03fee3798e05798779556c3e7534db6a9d39033232f7278d32279"
OLS_SOURCE_R3_TENSOR_SHA256 = "dcade4dca73f86f7bdc8d9be8fe183a8eda6a5f3fe2b810986d846ddf4c12fd3"
OLS_FIT_BANK_PATH = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_ols_producer_data_authority_v1_attempt_01/ols_producer_fit_bank.v1.pt"
)
OLS_FIT_BANK_SHA256 = "21b989a13cbea8f24d3298424b200eb1727de8d04809bfbc77e1901daa888f77"
OLS_INTERNAL_VALIDATION_BANK_PATH = Path(
    "outputs/20260907_v2/runs/"
    "lightgen_ols_producer_data_authority_v1_attempt_01/"
    "ols_producer_internal_validation_bank.v1.pt"
)
OLS_INTERNAL_VALIDATION_BANK_SHA256 = "8397c886db73bb260723debd6d4eab332b6172972a055b28f770e7ca2bf3ee25"

S1_SEMANTIC_MANIFEST_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01/"
    "semantic_manifest.json"
)
S1_SEMANTIC_MANIFEST_SHA256 = "b0199040926618b17373edc4f167cc8bc4788f39f64de174e1a3fb225781833e"
S1_REPRESENTATION_CONTRACT_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01/"
    "ols_representation_contract.json"
)
S1_REPRESENTATION_CONTRACT_SHA256 = "f99c2ced713005fe3f4a5a0fa91a2cd7648f5647e2360f122bc68e77f225f10d"
S1_PHASE_GAUGE_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01/"
    "phase_gauge_audit.json"
)
S1_PHASE_GAUGE_SHA256 = "5ee1a81ebdc054937f70b2e206d0da95aeae45d8a14f6f3812f265000563abb2"
S1_PHASE_ATTRIBUTION_PATH = Path(
    "outputs/20260904_v2/runs/"
    "lightgen_scaled_chen_ols_semantic_representation_audit_v1_attempt_01/"
    "phase_attribution.json"
)
S1_PHASE_ATTRIBUTION_SHA256 = "17b9bc1c0f474ecf603acefc39847cbda6497f4113e8553bab4ae29c95904d92"

FREEZE_CONTRACT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_selected_producer_freeze_contract_v1_attempt_01/"
    "n3_selected_producer_freeze_contract.v1.json"
)
FREEZE_CONTRACT_SHA256 = "4d8d3f5dc81a7eff2d92a0acde0a0d0639788bb0dee75e6d88db745f642b60a3"
PRODUCER_LINEAGE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "n3_selected_producer_full_development_fit_v1_attempt_01/"
    "n3_selected_producer_full_development_fit_lineage.v1.json"
)
PRODUCER_LINEAGE_SHA256 = "9244df01c95e44005207cb1dc188fc7de2961665e13055beebf55f0f5cdd2267"
PRODUCER_STATE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "n3_selected_producer_full_development_fit_v1_attempt_01/"
    "n3_selected_empirical_gaussian_copula_producer_state.v1.pt"
)
PRODUCER_STATE_SHA256 = "752db16087e7bdf8343e4499202182370b376a1a2d5925a383f411fd4a15bb88"
PRODUCER_STATE_SEMANTIC_ID = "N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1"
PRODUCER_FAMILY_ID = "EMPIRICAL_GAUSSIAN_COPULA"
PRODUCER_CANDIDATE_ID = "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1"

SEALED_MANIFEST_PATH = Path(
    "outputs/20260908_v2/runs/"
    "n3_sealed_confirmation_source_v1_attempt_01/"
    "n3_sealed_confirmation_source_manifest.v1.json"
)
SEALED_MANIFEST_SHA256 = "6475474f157d31591a3b0252f27320323d84ea93803c61ee6a365420a8ebb08e"
SEALED_CONFIRMATION_LINEAGE_PATH = Path(
    "outputs/20260910_v2/runs/"
    "n3_selected_producer_sealed_confirmation_v1_attempt_01/"
    "n3_selected_producer_sealed_confirmation_lineage.v1.json"
)
SEALED_CONFIRMATION_LINEAGE_SHA256 = "a8e8dc3ed3b4c2a5eb274aa0d089e6697aed8fdda83bf513c36bfdf548210bb7"

GOVERNANCE_STATE_PATH = Path("docs/status/research_governance_state.json")
CURRENT_STATUS_PATH = Path("docs/CURRENT_STATUS.md")
CURRENT_HANDOFF_PATH = Path("docs/status/current_handoff.md")

METRIC_IMPLEMENTATION_PATH = Path("scripts/research/validate_n3_amplitude_producer_baseline.py")
METRIC_IMPLEMENTATION_SHA256 = "1ad916bbb8d109ca989f64b9f5590391cdffea9e0567e2ca8d241b032c6cde36"
EVALUATOR_IMPLEMENTATION_PATH = Path("scripts/research/execute_n3_development_bank_v2_empirical_ceiling.py")
EVALUATOR_IMPLEMENTATION_SHA256 = "18b8f9defd83c2591dc055a97621ef6a1717783f7656da75994881e02bffda14"
AGGREGATION_IMPLEMENTATION_PATH = Path(
    "scripts/research/execute_n3_empirical_marginal_gaussian_copula_producer_v1.py"
)
AGGREGATION_IMPLEMENTATION_SHA256 = "5f1fa3fa6be35e96f345c3b7f7c6cce5ca0b45fe4619b1ead4455e86dcae538a"
Q_MATERIALIZATION_IMPLEMENTATION_PATH = Path("scripts/research/materialize_n3_development_bank_v2_q.py")
Q_REPLAY_IMPLEMENTATION_PATH = Path("scripts/research/replay_scaled_chen_repaired_coordinate_mra.py")


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


def _read_json(
    repo_root: Path,
    path: Path,
    name: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{name} is missing: {target}")
    actual = file_sha256(target)
    if expected_sha256 is not None:
        _require(actual == expected_sha256, f"{name} hash changed: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidHarnessError(f"{name} cannot be read as JSON: {target}") from exc
    return dict(_mapping(value, name))


def _ref(
    repo_root: Path,
    path: Path,
    role: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    target = repo_root / path
    _require(target.is_file(), f"{role} is missing: {target}")
    actual = file_sha256(target)
    if expected_sha256 is not None:
        _require(actual == expected_sha256, f"{role} hash changed: {target}")
    return {"path": _relative(repo_root, target), "sha256": actual, "role": role}


def _git_state(repo_root: Path) -> dict[str, Any]:
    def _run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    status = _run("status", "--porcelain").splitlines()
    return {
        "branch": _run("branch", "--show-current"),
        "head": _run("rev-parse", "HEAD"),
        "dirty_paths": [line[3:] if len(line) >= 4 else line for line in status],
        "worktree_clean": not status,
    }


def _validate_current_state(repo_root: Path) -> dict[str, Any]:
    governance_sha256 = file_sha256(repo_root / GOVERNANCE_STATE_PATH)
    governance = _read_json(repo_root, GOVERNANCE_STATE_PATH, "current governance state", governance_sha256)
    _require(
        governance.get("active_objective") == "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED",
        "current active objective is not the passed sealed confirmation",
    )
    _require(
        governance.get("hard_blocker") == "N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_CONTRACT_REQUIRED",
        "current hard blocker is not the complex-OLS contract",
    )
    _require(governance.get("authorization_tier") == "R3", "current governance tier is not R3")
    aliases = _mapping(governance.get("evidence_aliases"), "governance evidence aliases")
    confirmation = _mapping(
        aliases.get("N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_V1"),
        "sealed confirmation execution alias",
    )
    _require(confirmation.get("sha256") == SEALED_CONFIRMATION_LINEAGE_SHA256, "sealed confirmation lineage changed")
    _require(
        confirmation.get("classification") == "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED"
        and confirmation.get("sealed_confirmation_consumed") is True,
        "sealed confirmation is not the consumed passed result",
    )
    _require(
        confirmation.get("producer_state_sha256") == PRODUCER_STATE_SHA256
        and confirmation.get("producer_semantic_authority") == PRODUCER_STATE_SEMANTIC_ID,
        "governance confirmation alias does not bind the selected producer state",
    )
    _require(
        confirmation.get("training_authorized") is False
        and confirmation.get("formal_promotion") is False
        and confirmation.get("generation_readiness") == "not_ready",
        "governance opened a downstream boundary",
    )
    _require(
        "N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_CONTRACT_V1" not in aliases,
        "complex-OLS contract alias already exists; refusing a second definition",
    )
    for path, name in ((CURRENT_STATUS_PATH, "CURRENT_STATUS"), (CURRENT_HANDOFF_PATH, "current handoff")):
        text = (repo_root / path).read_text(encoding="utf-8")
        _require(TASK_ID in text, f"{name} does not name the current next action")
        _require("training_authorized=false" in text, f"{name} changed training authorization")
        _require("generation_readiness=not_ready" in text, f"{name} changed generation readiness")
    return {
        "governance": _ref(repo_root, GOVERNANCE_STATE_PATH, "pre-contract governance snapshot", governance_sha256),
        "current_status": _ref(repo_root, CURRENT_STATUS_PATH, "pre-contract current status snapshot"),
        "current_handoff": _ref(repo_root, CURRENT_HANDOFF_PATH, "pre-contract current handoff snapshot"),
        "active_objective": governance["active_objective"],
        "authorization_tier": governance["authorization_tier"],
        "hard_blocker": governance["hard_blocker"],
    }


def _validate_prior_provenance(repo_root: Path) -> dict[str, Any]:
    freeze = _read_json(repo_root, FREEZE_CONTRACT_PATH, "selected producer freeze contract", FREEZE_CONTRACT_SHA256)
    _require(freeze.get("schema") == "n3_selected_producer_freeze_contract.v1", "freeze contract schema changed")
    _require(
        freeze.get("status") == "valid_completed"
        and freeze.get("result") == "VALID_COMPLETED"
        and freeze.get("classification") == "DEFINE_N3_SELECTED_PRODUCER_FREEZE_CONTRACT_V1_READY",
        "freeze contract is not complete",
    )
    freeze_selection = _mapping(freeze.get("selection"), "selected producer freeze selection")
    _require(
        freeze_selection.get("selected_development_producer_candidate") == PRODUCER_FAMILY_ID
        and freeze_selection.get("selected_candidate_id") == PRODUCER_CANDIDATE_ID,
        "selected producer family changed",
    )

    lineage = _read_json(repo_root, PRODUCER_LINEAGE_PATH, "selected producer full-fit lineage", PRODUCER_LINEAGE_SHA256)
    _require(lineage.get("schema") == "n3_selected_producer_full_development_fit_lineage.v1", "producer lineage schema changed")
    _require(
        lineage.get("status") == "valid_completed"
        and lineage.get("result") == "VALID_COMPLETED"
        and lineage.get("classification") == "N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_COMPLETED",
        "producer full-fit lineage is not complete",
    )
    _require(lineage.get("next_action") == "DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1", "producer lineage is not the frozen prior state")
    producer_state = _mapping(lineage.get("producer_state"), "producer state lineage")
    _require(producer_state.get("path") == _relative(repo_root, repo_root / PRODUCER_STATE_PATH), "producer state path changed")
    _require(
        producer_state.get("sha256") == PRODUCER_STATE_SHA256
        and producer_state.get("semantic_authority_id") == PRODUCER_STATE_SEMANTIC_ID
        and producer_state.get("family_id") == PRODUCER_FAMILY_ID
        and producer_state.get("candidate_id") == PRODUCER_CANDIDATE_ID,
        "producer state identity changed",
    )
    _require(producer_state.get("raw_q_rows_persisted") is False, "producer state persisted raw q rows")
    _require(producer_state.get("sealed_information_persisted") is False, "producer state persisted sealed information")
    producer_ref = _ref(repo_root, PRODUCER_STATE_PATH, "selected producer state", PRODUCER_STATE_SHA256)
    _require(lineage.get("governance", {}).get("training_authorized") is False, "producer lineage opened training")
    _require(lineage.get("governance", {}).get("formal_promotion") is False, "producer lineage opened promotion")

    sealed_lineage = _read_json(
        repo_root,
        SEALED_CONFIRMATION_LINEAGE_PATH,
        "sealed confirmation execution lineage",
        SEALED_CONFIRMATION_LINEAGE_SHA256,
    )
    _require(sealed_lineage.get("schema") == "n3_selected_producer_sealed_confirmation_lineage.v1", "sealed lineage schema changed")
    _require(
        sealed_lineage.get("classification") == "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED"
        and sealed_lineage.get("sealed", {}).get("consumed") is True,
        "sealed prior result is not consumed and passed",
    )
    _require(
        sealed_lineage.get("producer", {}).get("state_sha256") == PRODUCER_STATE_SHA256
        and sealed_lineage.get("producer", {}).get("semantic_authority") == PRODUCER_STATE_SEMANTIC_ID,
        "sealed prior result changed producer provenance",
    )
    _require(sealed_lineage.get("next_action") == TASK_ID, "sealed prior result next action changed")
    _require(sealed_lineage.get("governance", {}).get("training_authorized") is False, "sealed result opened training")
    _require(sealed_lineage.get("governance", {}).get("formal_promotion") is False, "sealed result opened promotion")

    sealed_manifest = _read_json(repo_root, SEALED_MANIFEST_PATH, "sealed reservation manifest", SEALED_MANIFEST_SHA256)
    _require(sealed_manifest.get("schema") == "n3_sealed_confirmation_source_manifest.v1", "sealed manifest schema changed")
    _require(
        sealed_manifest.get("semantic_authority_id") == "N3_SEALED_CONFIRMATION_SOURCE_V1"
        and sealed_manifest.get("status") == "RESERVED_NOT_MATERIALIZED",
        "sealed reservation source identity or immutable status changed",
    )
    sealed_source = _mapping(sealed_manifest.get("source"), "sealed source manifest source")
    _require(sealed_source.get("count") == 60000 and sealed_source.get("dataset") == "mnist_train", "sealed source dimensions changed")
    _require(sealed_manifest.get("membership") is None and sealed_manifest.get("overlap_verification") is None, "sealed membership content entered definition")
    return {
        "freeze_contract": {
            "path": _relative(repo_root, repo_root / FREEZE_CONTRACT_PATH),
            "sha256": FREEZE_CONTRACT_SHA256,
            "schema": freeze["schema"],
            "classification": freeze["classification"],
        },
        "producer": {
            "lineage": _ref(repo_root, PRODUCER_LINEAGE_PATH, "selected producer full-fit lineage", PRODUCER_LINEAGE_SHA256),
            "state": producer_ref,
            "semantic_authority_id": PRODUCER_STATE_SEMANTIC_ID,
            "family_id": PRODUCER_FAMILY_ID,
            "candidate_id": PRODUCER_CANDIDATE_ID,
            "fit_rows": DEVELOPMENT_ROWS,
            "fit_rows_per_class": DEVELOPMENT_ROWS_PER_CLASS,
            "used_by_this_contract": "provenance_only; no state values loaded",
        },
        "sealed_confirmation": {
            "lineage": _ref(repo_root, SEALED_CONFIRMATION_LINEAGE_PATH, "prior sealed confirmation lineage", SEALED_CONFIRMATION_LINEAGE_SHA256),
            "reservation_manifest": _ref(repo_root, SEALED_MANIFEST_PATH, "consumed sealed reservation manifest", SEALED_MANIFEST_SHA256),
            "status_in_immutable_reservation_manifest": "RESERVED_NOT_MATERIALIZED",
            "prior_execution_consumed": True,
            "content_loaded_by_this_definition": False,
            "reuse_forbidden": True,
            "purpose": "prior confirmation and irreversible-consumption provenance only",
        },
    }


def _validate_v2_authority(repo_root: Path) -> dict[str, Any]:
    source = _read_json(repo_root, V2_SOURCE_MANIFEST_PATH, "V2 source manifest", V2_SOURCE_MANIFEST_SHA256)
    _require(source.get("schema") == "n3_development_bank_v2_source_manifest.v1", "V2 source schema changed")
    _require(source.get("semantic_authority_id") == "N3_DEVELOPMENT_BANK_V2", "V2 semantic authority changed")
    _require(source.get("status") == "FROZEN_SOURCE_ROWS", "V2 source rows are not frozen")
    _require(source.get("bank_semantic_sha256") == V2_BANK_SEMANTIC_SHA256, "V2 bank semantic digest changed")
    identity = _mapping(source.get("identity"), "V2 source identity")
    _require(identity.get("class_counts") == [DEVELOPMENT_ROWS_PER_CLASS] * CLASS_COUNT, "V2 class counts changed")
    _require(identity.get("row_manifest_sha256") == V2_ROW_MANIFEST_SHA256, "V2 row manifest digest changed")
    _require(identity.get("selected_source_indices_sha256") == V2_SOURCE_INDICES_SHA256, "V2 source-index digest changed")
    _require(identity.get("selected_labels_sha256") == V2_LABELS_SHA256, "V2 label digest changed")
    _require(identity.get("unique_source_rows") is True, "V2 source rows are not unique")
    rows = source.get("rows")
    _require(isinstance(rows, list) and len(rows) == DEVELOPMENT_ROWS, "V2 source row list changed")
    _require(canonical_sha256(rows) == V2_ROW_MANIFEST_SHA256, "V2 row manifest contents changed")
    source_indices: list[int] = []
    stable_ids: list[str] = []
    class_counts = [0] * CLASS_COUNT
    for position, row_value in enumerate(rows):
        row = _mapping(row_value, f"V2 source row {position}")
        _require(row.get("bank_row_position") == position, "V2 bank row positions are not contiguous")
        label = row.get("label")
        source_index = row.get("source_index")
        _require(isinstance(label, int) and 0 <= label < CLASS_COUNT, "V2 row label is invalid")
        _require(isinstance(source_index, int) and source_index >= 0, "V2 source index is invalid")
        expected_identity = hashlib.sha256(
            f"mnist_train|{source_index}|{label}|SOURCE_ROW_IDENTITY_V1".encode("utf-8")
        ).hexdigest()
        _require(row.get("stable_source_row_identity") == expected_identity, "V2 stable row identity changed")
        source_indices.append(source_index)
        stable_ids.append(expected_identity)
        class_counts[label] += 1
    _require(len(set(source_indices)) == DEVELOPMENT_ROWS, "V2 source indices are not unique")
    _require(class_counts == [DEVELOPMENT_ROWS_PER_CLASS] * CLASS_COUNT, "V2 row class counts changed")
    _require(canonical_sha256(stable_ids) == V2_SOURCE_ROW_IDENTITY_SHA256, "V2 stable row identity digest changed")
    selection = _mapping(source.get("selection"), "V2 source selection")
    for key in ("q_values_used", "ols_values_used", "image_values_used", "rng_used", "reselection", "split_optimization"):
        _require(selection.get(key) is False, f"V2 source selection boundary changed: {key}")
    _require(selection.get("pool_count") == DEVELOPMENT_ROWS and selection.get("rows_per_class") == DEVELOPMENT_ROWS_PER_CLASS, "V2 source selection size changed")
    materialization_boundary = _mapping(source.get("materialization_boundary"), "V2 source materialization boundary")
    for key in ("q_values_loaded", "q_values_materialized", "sealed_confirmation_consumed", "producer_evaluated", "producer_selection", "training_authorized", "formal_promotion"):
        _require(materialization_boundary.get(key) is False, f"V2 source boundary changed: {key}")
    exclusions = _mapping(source.get("exclusions"), "V2 exclusions")
    exclusion_authorities = _mapping(exclusions.get("authorities"), "V2 exclusion authorities")
    sealed_exclusion = _mapping(exclusion_authorities.get("N3_SEALED_CONFIRMATION_RESERVATION"), "V2 sealed exclusion")
    _require(sealed_exclusion.get("manifest_sha256") == SEALED_MANIFEST_SHA256, "V2 source did not preserve sealed exclusion identity")

    q_manifest = _read_json(repo_root, V2_Q_MANIFEST_PATH, "V2 q materialization manifest", V2_Q_MANIFEST_SHA256)
    _require(q_manifest.get("schema") == "n3_development_bank_v2_q_materialization_manifest.v1", "V2 q manifest schema changed")
    _require(q_manifest.get("status") == "VALID_COMPLETED", "V2 q materialization is not complete")
    q_source = _mapping(q_manifest.get("source_manifest"), "V2 q source authority")
    _require(q_source.get("sha256") == V2_SOURCE_MANIFEST_SHA256 and q_source.get("bank_semantic_sha256") == V2_BANK_SEMANTIC_SHA256, "V2 q source binding changed")
    q_identity = _mapping(q_manifest.get("source_identity"), "V2 q source identity")
    _require(
        q_identity.get("labels_sha256") == V2_LABELS_SHA256
        and q_identity.get("row_manifest_sha256") == V2_ROW_MANIFEST_SHA256
        and q_identity.get("source_indices_sha256") == V2_SOURCE_INDICES_SHA256
        and q_identity.get("source_row_identity_sha256") == V2_SOURCE_ROW_IDENTITY_SHA256,
        "V2 q source identity changed",
    )
    q_artifact = _mapping(q_manifest.get("q_artifact"), "V2 q artifact metadata")
    _require(
        q_artifact.get("schema") == "n3_development_bank_v2_q.v1"
        and q_artifact.get("path") == V2_Q_ARTIFACT_PATH.name
        and q_artifact.get("sha256") == V2_Q_ARTIFACT_SHA256
        and q_artifact.get("q_tensor_sha256") == V2_Q_TENSOR_SHA256
        and q_artifact.get("shape") == [DEVELOPMENT_ROWS, DIMENSION]
        and q_artifact.get("dtype") == "torch.float64",
        "V2 q artifact metadata changed",
    )
    q_artifact_ref = _ref(repo_root, V2_Q_ARTIFACT_PATH, "V2 q artifact bytes", V2_Q_ARTIFACT_SHA256)
    frozen_ols = _mapping(q_manifest.get("frozen_ols_authority"), "V2 q frozen OLS authority")
    source_ols = _mapping(source.get("ols_q_authority"), "V2 source OLS/q authority")
    _require(source_ols.get("coordinate") == "R2_AMPLITUDE", "V2 q coordinate changed")
    _require(source_ols.get("definition") == "q=abs(c/sqrt(s))=abs(R3_COMPLEX)", "V2 q definition changed")
    _require(frozen_ols.get("checkpoint_sha256") == "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de", "V2 q checkpoint binding changed")
    _require(frozen_ols.get("mode_bank_sha256") == "75fb38eee956a1d177486e00b22e0b62f0c1bf29e94eb398851bed18d6288ba9", "V2 q mode-bank binding changed")
    _require(frozen_ols.get("calibration_sha256") == "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07", "V2 q calibration binding changed")
    _require(q_manifest.get("sealed_confirmation_consumed") is False, "V2 q manifest entered sealed state")
    q_execution = _mapping(q_manifest.get("execution"), "V2 q execution boundary")
    _require(q_execution.get("sealed_confirmation_values_loaded") is False and q_execution.get("training_steps") == 0, "V2 q execution boundary changed")

    fold_manifest = _read_json(repo_root, V2_FOLD_MANIFEST_PATH, "V2 fold manifest", V2_FOLD_MANIFEST_SHA256)
    _require(fold_manifest.get("schema") == "n3_development_bank_v2_five_fold_protocol.v1", "V2 fold schema changed")
    _require(fold_manifest.get("status") == "FROZEN_BEFORE_VALIDATION", "V2 fold protocol is not frozen before validation")
    _require(fold_manifest.get("semantic_authority_id") == "N3_DEVELOPMENT_BANK_V2", "V2 fold semantic authority changed")
    fold_source = _mapping(fold_manifest.get("source_manifest"), "V2 fold source authority")
    _require(fold_source.get("sha256") == V2_SOURCE_MANIFEST_SHA256 and fold_source.get("row_manifest_sha256") == V2_ROW_MANIFEST_SHA256, "V2 fold source binding changed")
    fold_q = _mapping(fold_manifest.get("q_artifact"), "V2 fold q authority")
    _require(fold_q.get("manifest_sha256") == V2_Q_MANIFEST_SHA256 and fold_q.get("artifact_sha256") == V2_Q_ARTIFACT_SHA256 and fold_q.get("tensor_sha256") == V2_Q_TENSOR_SHA256, "V2 fold q binding changed")
    protocol = _mapping(fold_manifest.get("protocol"), "V2 fold protocol")
    for key, expected in {
        "fold_count": FOLD_COUNT,
        "pool_count": DEVELOPMENT_ROWS,
        "fit_count_per_fold": FIT_ROWS_PER_FOLD,
        "fit_per_class_per_fold": FIT_ROWS_PER_CLASS_PER_FOLD,
        "heldout_count_per_fold": HELDOUT_ROWS_PER_FOLD,
        "heldout_per_class_per_fold": HELDOUT_ROWS_PER_CLASS_PER_FOLD,
        "rows_per_class": DEVELOPMENT_ROWS_PER_CLASS,
        "assignment_sha256": V2_ASSIGNMENT_SHA256,
    }.items():
        _require(protocol.get(key) == expected, f"V2 fold protocol changed: {key}")
    _require(protocol.get("q_values_used") is False and protocol.get("rng_used") is False and protocol.get("row_to_fold_frozen_before_validation") is True, "V2 fold protocol boundary changed")
    fold_rows = fold_manifest.get("rows")
    _require(isinstance(fold_rows, list) and len(fold_rows) == DEVELOPMENT_ROWS, "V2 fold row list changed")
    _require(canonical_sha256(fold_rows) == V2_FOLD_ROWS_SHA256, "V2 fold row manifest digest changed")
    fold_ids: list[int] = []
    fold_counts = [[0] * CLASS_COUNT for _ in range(FOLD_COUNT)]
    for position, row_value in enumerate(fold_rows):
        row = _mapping(row_value, f"V2 fold row {position}")
        source_row = _mapping(rows[position], f"V2 source row {position}")
        for key in ("bank_row_position", "eligible_source_rank", "label", "selected_within_class_rank", "source_index", "stable_source_row_identity"):
            _require(row.get(key) == source_row.get(key), f"V2 fold row source correspondence changed: {key}")
        fold = row.get("heldout_fold")
        _require(isinstance(fold, int) and 0 <= fold < FOLD_COUNT, "V2 heldout fold is invalid")
        fold_ids.append(fold)
        fold_counts[fold][int(row["label"])] += 1
    _require(all(counts == [HELDOUT_ROWS_PER_CLASS_PER_FOLD] * CLASS_COUNT for counts in fold_counts), "V2 fold class balance changed")
    fold_identity = _mapping(fold_manifest.get("fold_identity"), "V2 fold identity")
    _require(fold_identity.get("fold_id_by_bank_row") == fold_ids and fold_identity.get("assignment_sha256") == V2_ASSIGNMENT_SHA256, "V2 fold identity changed")
    boundary = _mapping(fold_manifest.get("execution_boundary"), "V2 fold execution boundary")
    for key in ("q_values_loaded", "scientific_validation_executed", "sealed_confirmation_consumed", "training_authorized", "formal_promotion"):
        _require(boundary.get(key) is False, f"V2 fold execution boundary changed: {key}")

    ceiling = _read_json(repo_root, V2_CEILING_PATH, "V2 empirical ceiling report", V2_CEILING_SHA256)
    _require(
        ceiling.get("status") == "valid_completed"
        and ceiling.get("result") == "VALID_COMPLETED"
        and ceiling.get("classification") == "N3_DEVELOPMENT_BANK_V2_VALIDATED",
        "V2 empirical ceiling is not valid completed",
    )
    ceiling_source = _mapping(ceiling.get("source_authority"), "V2 ceiling source authority")
    _require(ceiling_source.get("sha256") == V2_SOURCE_MANIFEST_SHA256 and ceiling_source.get("bank_semantic_sha256") == V2_BANK_SEMANTIC_SHA256, "V2 ceiling source binding changed")
    ceiling_q = _mapping(ceiling.get("q_materialization_authority"), "V2 ceiling q authority")
    _require(ceiling_q.get("manifest_sha256") == V2_Q_MANIFEST_SHA256 and ceiling_q.get("artifact_sha256") == V2_Q_ARTIFACT_SHA256, "V2 ceiling q binding changed")
    ceiling_fold = _mapping(ceiling.get("fold_protocol"), "V2 ceiling fold authority")
    _require(ceiling_fold.get("sha256") == V2_FOLD_MANIFEST_SHA256 and ceiling_fold.get("assignment_sha256") == V2_ASSIGNMENT_SHA256, "V2 ceiling fold binding changed")
    empirical_ceiling = _mapping(ceiling.get("empirical_ceiling"), "V2 empirical ceiling")
    _require(empirical_ceiling.get("compatible_fold_count") == FOLD_COUNT and empirical_ceiling.get("development_supported") is True, "V2 empirical ceiling support changed")
    sealed_status = _mapping(ceiling.get("sealed_confirmation"), "V2 ceiling sealed boundary")
    _require(sealed_status.get("consumed") is False and sealed_status.get("values_loaded") is False, "V2 ceiling touched sealed data")
    return {
        "source": {
            "path": _relative(repo_root, repo_root / V2_SOURCE_MANIFEST_PATH),
            "sha256": V2_SOURCE_MANIFEST_SHA256,
            "schema": source["schema"],
            "semantic_authority_id": source["semantic_authority_id"],
            "bank_semantic_sha256": V2_BANK_SEMANTIC_SHA256,
            "row_manifest_sha256": V2_ROW_MANIFEST_SHA256,
            "source_indices_sha256": V2_SOURCE_INDICES_SHA256,
            "labels_sha256": V2_LABELS_SHA256,
            "source_row_identity_sha256": V2_SOURCE_ROW_IDENTITY_SHA256,
            "count": DEVELOPMENT_ROWS,
            "rows_per_class": DEVELOPMENT_ROWS_PER_CLASS,
            "class_counts": [DEVELOPMENT_ROWS_PER_CLASS] * CLASS_COUNT,
            "row_order": "source manifest bank_row_position 0..4999; exact frozen source identity",
        },
        "q": {
            "manifest": _ref(repo_root, V2_Q_MANIFEST_PATH, "V2 q materialization manifest", V2_Q_MANIFEST_SHA256),
            "artifact": q_artifact_ref,
            "tensor_sha256": V2_Q_TENSOR_SHA256,
            "shape": [DEVELOPMENT_ROWS, DIMENSION],
            "dtype": "torch.float64",
            "coordinate": "R2_AMPLITUDE",
            "definition": "q_j=A_j=abs(z_j)=abs(c_j/sqrt(s_j))",
            "values_loaded_by_this_definition": False,
        },
        "folds": {
            "manifest": _ref(repo_root, V2_FOLD_MANIFEST_PATH, "V2 five-fold manifest", V2_FOLD_MANIFEST_SHA256),
            "assignment_sha256": V2_ASSIGNMENT_SHA256,
            "fold_rows_sha256": V2_FOLD_ROWS_SHA256,
            "fold_count": FOLD_COUNT,
            "fit_rows_per_fold": FIT_ROWS_PER_FOLD,
            "fit_rows_per_class_per_fold": FIT_ROWS_PER_CLASS_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_ROWS_PER_FOLD,
            "heldout_rows_per_class_per_fold": HELDOUT_ROWS_PER_CLASS_PER_FOLD,
            "heldout_fold_rule": protocol["heldout_rule"],
            "phase_donor_and_control_scope": "fit rows only; heldout rows are evaluator-only",
        },
        "ceiling": {
            "report": _ref(repo_root, V2_CEILING_PATH, "V2 empirical ceiling report", V2_CEILING_SHA256),
            "classification": ceiling["classification"],
            "compatible_fold_count": FOLD_COUNT,
            "used_as_joint_compatibility_target": False,
            "used_as_data_authority_precondition_only": True,
        },
    }


def _validate_complex_ols_authority(repo_root: Path) -> dict[str, Any]:
    source = _read_json(repo_root, OLS_SOURCE_MANIFEST_PATH, "PROJECT_COMPLEX_OLS source manifest", OLS_SOURCE_MANIFEST_SHA256)
    _require(source.get("schema") == "ols_producer_source_manifest.v1", "OLS source manifest schema changed")
    _require(source.get("source_coordinate") == "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX", "OLS source coordinate changed")
    artifact = _mapping(source.get("source_artifact"), "OLS source artifact")
    _require(
        artifact.get("path") == _relative(repo_root, repo_root / OLS_SOURCE_ARTIFACT_PATH)
        and artifact.get("sha256") == OLS_SOURCE_ARTIFACT_SHA256
        and artifact.get("r3_complex_sha256") == OLS_SOURCE_R3_TENSOR_SHA256
        and artifact.get("shape") == [1000, DIMENSION]
        and artifact.get("dtype") == "torch.complex64",
        "historical OLS source artifact binding changed",
    )
    _require(artifact.get("count") == 1000 and artifact.get("per_class") == 100, "historical OLS source row count changed")
    ols_artifact_ref = _ref(repo_root, OLS_SOURCE_ARTIFACT_PATH, "historical PROJECT_COMPLEX_OLS artifact", OLS_SOURCE_ARTIFACT_SHA256)
    partition = _read_json(repo_root, OLS_PARTITION_MANIFEST_PATH, "OLS partition manifest", OLS_PARTITION_MANIFEST_SHA256)
    _require(partition.get("schema") == "ols_producer_partition_manifest.v1", "OLS partition schema changed")
    _require(partition.get("source_coordinate") == source["source_coordinate"] and partition.get("source_artifact_sha256") == OLS_SOURCE_ARTIFACT_SHA256, "OLS partition source binding changed")
    _require(partition.get("runtime_membership_mutation") is False and partition.get("runtime_reselection") is False and partition.get("runtime_resplit") is False, "OLS partition runtime mutability changed")
    _require(partition.get("overlap", {}).get("fit_validation_source_index_count") == 0 and partition.get("overlap", {}).get("fit_validation_row_position_count") == 0, "historical OLS partition overlaps")
    fit = _mapping(partition.get("fit"), "historical OLS fit bank")
    internal = _mapping(partition.get("internal_validation"), "historical OLS internal validation bank")
    _require(fit.get("count") == 500 and internal.get("count") == 500, "historical OLS partition counts changed")
    fit_ref = _ref(repo_root, OLS_FIT_BANK_PATH, "historical OLS fit bank", OLS_FIT_BANK_SHA256)
    internal_ref = _ref(repo_root, OLS_INTERNAL_VALIDATION_BANK_PATH, "historical OLS internal validation bank", OLS_INTERNAL_VALIDATION_BANK_SHA256)

    firewall = _read_json(repo_root, OLS_FIREWALL_MANIFEST_PATH, "OLS data firewall manifest", OLS_FIREWALL_MANIFEST_SHA256)
    _require(firewall.get("schema") == "ols_producer_data_firewall_manifest.v1" and firewall.get("firewall_pass") is True, "OLS data firewall is not passed")
    _require(firewall.get("source_coordinate") == source["source_coordinate"], "OLS firewall source coordinate changed")
    future_consumer = _mapping(firewall.get("future_consumer"), "OLS future consumer firewall")
    for key in ("must_verify_target_bank_identity", "runtime_membership_mutation", "runtime_reselection", "runtime_resplit"):
        expected = True if key == "must_verify_target_bank_identity" else False
        _require(future_consumer.get(key) is expected, f"OLS future consumer firewall changed: {key}")
    governance = _mapping(firewall.get("governance"), "OLS firewall governance")
    _require(governance.get("training_authorized") is False and governance.get("formal_promotion") is False and governance.get("generation_readiness") == "not_ready", "OLS firewall opened downstream governance")

    representation = _read_json(repo_root, S1_REPRESENTATION_CONTRACT_PATH, "S1 OLS representation contract", S1_REPRESENTATION_CONTRACT_SHA256)
    _require(representation.get("schema") == "lightgen_scaled_chen_ols_representation_contract.v1", "S1 representation schema changed")
    representations = _mapping(representation.get("representations"), "S1 OLS representations")
    _require(representations.get("R3_COMPLEX") == "c/sqrt(s), gauge-aware complex geometry", "R3 complex representation changed")
    _require(representations.get("R2_AMPLITUDE") == "abs(c/sqrt(s))", "R2 amplitude representation changed")
    _require(representation.get("complex_primary_geometry") == "global_phase_gauge_distance" and representation.get("raw_cartesian_primary_geometry") is False, "S1 complex geometry boundary changed")
    source_tensor = _mapping(representation.get("source_tensor"), "S1 OLS source tensor")
    _require(source_tensor.get("path") == "ScaledChenBOGTModel._coefficients(encoded)" and source_tensor.get("semantic_field") == "output.coefficients" and source_tensor.get("shape") == ["N", DIMENSION] and source_tensor.get("dtype") == "torch.complex64", "S1 OLS source tensor changed")

    phase_gauge = _read_json(repo_root, S1_PHASE_GAUGE_PATH, "S1 phase gauge audit", S1_PHASE_GAUGE_SHA256)
    _require(phase_gauge.get("schema") == "lightgen_scaled_chen_ols_phase_gauge_audit.v1", "phase gauge audit schema changed")
    _require(phase_gauge.get("classification") == "GLOBAL_PHASE_GAUGE_CONFIRMED", "global phase gauge authority is not confirmed")
    _require(phase_gauge.get("formal_phase_reference") is False and phase_gauge.get("raw_cartesian_primary") is False, "phase gauge reference boundary changed")
    _require(phase_gauge.get("primary_complex_metric") == "global_phase_gauge_distance", "phase gauge metric changed")
    phase_attribution = _read_json(repo_root, S1_PHASE_ATTRIBUTION_PATH, "S1 phase attribution", S1_PHASE_ATTRIBUTION_SHA256)
    _require(phase_attribution.get("schema") == "lightgen_scaled_chen_ols_phase_attribution.v1", "phase attribution schema changed")
    _require(phase_attribution.get("classification") == "PHASE_SEMANTIC_INCREMENT_SUPPORTED" and phase_attribution.get("paired_same_query_reference_identities") is True, "relative phase attribution authority changed")
    semantic_manifest_ref = _ref(repo_root, S1_SEMANTIC_MANIFEST_PATH, "S1 OLS semantic manifest", S1_SEMANTIC_MANIFEST_SHA256)

    checkpoint_path = Path("outputs/20260829_v2/runs/lightgen_optical_manifold_smf_bogt_horizon_v2_attempt_03/checkpoints/scaled_chen_horizon_step_5000.pt")
    calibration_path = Path("outputs/20260904_v2/runs/lightgen_scaled_chen_bogt_calibration_contract_repair_v2_attempt_02/repaired_calibration.pt")
    formal_config_path = Path("configs/v2/current/formal_identity_logit_amplitude_finite_crop_identity.yaml")
    mra_config_path = Path("configs/research/lightgen_optical_manifold_smf_bogt_manifold_readiness_v1.yaml")
    checkpoint_ref = _ref(repo_root, checkpoint_path, "fixed OLS checkpoint", "ccb89f98cc11725eb80d66c61f4fdef62b24acdf9f6d739f78373f462d10e5de")
    calibration_ref = _ref(repo_root, calibration_path, "fixed OLS calibration payload", "ce31dd816691a08ca9388e0ec086be0e2e3ebd40005598c1ef44a1ba79a9fd07")
    formal_config_ref = _ref(repo_root, formal_config_path, "fixed OLS formal config", "0c42035c1a525e27ff787d013ca27b0f5ae91338329eb0607c7042ebc5ae8905")
    mra_config_ref = _ref(repo_root, mra_config_path, "fixed OLS MRA config", "95e4587f87fc49e79fe3bef5a930f34502ebc9c95f36a312995d77f993a72cad")
    q_materializer_ref = _ref(repo_root, Q_MATERIALIZATION_IMPLEMENTATION_PATH, "V2 q materialization implementation")
    q_replay_ref = _ref(repo_root, Q_REPLAY_IMPLEMENTATION_PATH, "fixed OLS replay implementation")
    _require(source.get("calibration", {}).get("payload_sha256") == calibration_ref["sha256"], "OLS source calibration binding changed")
    _require(source.get("checkpoint", {}).get("sha256") == checkpoint_ref["sha256"], "OLS source checkpoint binding changed")

    return {
        "project_complex_ols": {
            "semantic_authority_id": "PROJECT_COMPLEX_OLS_V1",
            "coordinate": "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX",
            "source_manifest": _ref(repo_root, OLS_SOURCE_MANIFEST_PATH, "PROJECT_COMPLEX_OLS source manifest", OLS_SOURCE_MANIFEST_SHA256),
            "partition_manifest": _ref(repo_root, OLS_PARTITION_MANIFEST_PATH, "PROJECT_COMPLEX_OLS partition manifest", OLS_PARTITION_MANIFEST_SHA256),
            "firewall_manifest": _ref(repo_root, OLS_FIREWALL_MANIFEST_PATH, "PROJECT_COMPLEX_OLS firewall manifest", OLS_FIREWALL_MANIFEST_SHA256),
            "historical_source_artifact": ols_artifact_ref,
            "historical_fit_bank": fit_ref,
            "historical_internal_validation_bank": internal_ref,
            "historical_source_count": 1000,
            "historical_artifact_is_v2_5000_rows": False,
            "v2_corresponding_rows_status": "not_materialized_by_definition; future exact source-row materialization required before evaluation",
        },
        "s1_semantics": {
            "semantic_manifest": semantic_manifest_ref,
            "representation_contract": _ref(repo_root, S1_REPRESENTATION_CONTRACT_PATH, "S1 OLS representation contract", S1_REPRESENTATION_CONTRACT_SHA256),
            "phase_gauge_audit": _ref(repo_root, S1_PHASE_GAUGE_PATH, "S1 phase gauge audit", S1_PHASE_GAUGE_SHA256),
            "phase_attribution": _ref(repo_root, S1_PHASE_ATTRIBUTION_PATH, "S1 phase attribution", S1_PHASE_ATTRIBUTION_SHA256),
            "global_phase_gauge": "GLOBAL_PHASE_GAUGE_CONFIRMED",
            "formal_phase_reference": False,
            "primary_complex_geometry": "global_phase_gauge_distance",
        },
        "v2_complex_ols_materialization": {
            "status": "FUTURE_DETERMINISTIC_MATERIALIZATION_REQUIRED",
            "row_order": "exact V2 source manifest bank_row_position order",
            "row_identity": "each complex row carries the exact V2 source_index, label, and stable_source_row_identity",
            "source_image_domain": "mnist_train selected rows only; no reselection or runtime resplit",
            "field": "ScaledChenBOGTModel._coefficients(encoded).output.coefficients",
            "formula": "z_j=c_j/sqrt(s_j)=R3_COMPLEX",
            "dtype_before_canonicalization": "torch.complex64",
            "shape": [DEVELOPMENT_ROWS, DIMENSION],
            "checkpoint": checkpoint_ref,
            "calibration": calibration_ref,
            "formal_config": formal_config_ref,
            "mra_config": mra_config_ref,
            "q_materialization_implementation": q_materializer_ref,
            "ols_replay_implementation": q_replay_ref,
            "values_materialized_by_definition": False,
            "existing_5000_row_complex_artifact_claimed": False,
        },
    }


def _validate_metric_authority(repo_root: Path) -> dict[str, Any]:
    metric = _ref(repo_root, METRIC_IMPLEMENTATION_PATH, "energy-distance implementation", METRIC_IMPLEMENTATION_SHA256)
    evaluator = _ref(repo_root, EVALUATOR_IMPLEMENTATION_PATH, "model-bank evaluator implementation", EVALUATOR_IMPLEMENTATION_SHA256)
    aggregation = _ref(repo_root, AGGREGATION_IMPLEMENTATION_PATH, "model-bank aggregation implementation", AGGREGATION_IMPLEMENTATION_SHA256)
    return {
        "energy_distance": {
            **metric,
            "function": "_energy_distance",
            "formula": "2*mean(cdist(X,Y))-mean(cdist(X,X))-mean(cdist(Y,Y)); clamp only values in [-1e-10,0] to zero",
            "input": "CPU contiguous torch.float64 matrices with equal feature width; unequal row counts allowed",
        },
        "evaluator": {**evaluator, "role": "frozen model-self/data-vs-model statistic consumer"},
        "aggregation": {**aggregation, "function": "_mean_metrics", "rule": "unweighted arithmetic mean over four model banks"},
    }


def _validate_tensor_matrix(value: Any, name: str, *, complex_allowed: bool = True) -> torch.Tensor:
    _require(torch.is_tensor(value), f"{name} must be a tensor")
    _require(value.ndim == 2 and value.shape[1] == DIMENSION, f"{name} must have shape [N,{DIMENSION}]")
    if complex_allowed:
        _require(torch.is_complex(value), f"{name} must be complex")
    else:
        _require(not torch.is_complex(value), f"{name} must be real")
    _require(torch.isfinite(value.real).all().item() and torch.isfinite(value.imag).all().item() if torch.is_complex(value) else torch.isfinite(value).all().item(), f"{name} contains non-finite values")
    return value.detach().to(device="cpu", dtype=torch.complex128 if torch.is_complex(value) else torch.float64).contiguous()


def canonicalize_existing_global_phase(z: torch.Tensor) -> torch.Tensor:
    """Return a deterministic representative of the existing global-phase quotient.

    This is a numerical cross-section, not a new physical phase reference:
    the first coordinate attaining the exact maximum magnitude is the anchor,
    ties go to the smallest index, and the anchor is made real/non-negative.
    """

    value = _validate_tensor_matrix(z, "complex OLS rows")
    magnitudes = value.abs()
    _require(torch.all(magnitudes > 0).item(), "exact-zero complex OLS coordinate is invalid_harness")
    anchor_indices = magnitudes.argmax(dim=1)
    row_indices = torch.arange(value.shape[0], device=value.device)
    anchors = value[row_indices, anchor_indices]
    factors = anchors.conj() / anchors.abs()
    canonical = value * factors.unsqueeze(1)
    canonical[row_indices, anchor_indices] = anchors.abs().to(torch.complex128)
    _require(torch.isfinite(canonical.real).all().item() and torch.isfinite(canonical.imag).all().item(), "canonical complex OLS rows are non-finite")
    return canonical.contiguous()


def extract_complete_relative_phase(z: torch.Tensor) -> torch.Tensor:
    """Extract a complete principal-interval phase vector from each OLS row."""

    canonical = canonicalize_existing_global_phase(z)
    phase = torch.atan2(canonical.imag, canonical.real).to(torch.float64)
    _require(torch.isfinite(phase).all().item(), "phase extraction produced non-finite values")
    _require(torch.all((phase >= -torch.pi) & (phase <= torch.pi)).item(), "phase escaped the frozen principal interval")
    return phase.contiguous()


def compose_complex_from_amplitude_phase(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Compose and canonicalize one complete complex row set under the contract."""

    _require(torch.is_tensor(amplitude) and torch.is_tensor(phase), "amplitude and phase must be tensors")
    _require(amplitude.ndim == 2 and phase.ndim == 2 and tuple(amplitude.shape) == tuple(phase.shape), "amplitude/phase shapes must match")
    _require(amplitude.shape[1] == DIMENSION, f"amplitude/phase width must be {DIMENSION}")
    amplitude_value = amplitude.detach().to(device="cpu", dtype=torch.float64).contiguous()
    phase_value = phase.detach().to(device="cpu", dtype=torch.float64).contiguous()
    _require(torch.isfinite(amplitude_value).all().item() and torch.isfinite(phase_value).all().item(), "amplitude/phase contains non-finite values")
    _require(torch.all(amplitude_value > 0).item(), "exact-zero amplitude coordinate is invalid_harness")
    _require(torch.all((phase_value >= -torch.pi) & (phase_value <= torch.pi)).item(), "phase is outside the frozen principal interval")
    composed = torch.polar(amplitude_value, phase_value).to(torch.complex128)
    return canonicalize_existing_global_phase(composed)


def complex_real_embedding(z: torch.Tensor) -> torch.Tensor:
    """Embed canonical complex rows as [Re(z), Im(z)] in float64."""

    canonical = canonicalize_existing_global_phase(z)
    return torch.cat((canonical.real, canonical.imag), dim=1).to(torch.float64).contiguous()


def derive_seed(method: str, fold: int, role: str, trial: int | None, bank: int | None) -> int:
    """Derive one explicit CPU stream seed without retries or global RNG use."""

    _require(isinstance(method, str) and method, "seed method is required")
    _require(isinstance(role, str) and role, "seed role is required")
    _require(isinstance(fold, int) and 0 <= fold < FOLD_COUNT, "seed fold is invalid")
    trial_token = "none" if trial is None else str(trial)
    bank_token = "none" if bank is None else str(bank)
    material = f"{RNG_NAMESPACE}|method={method}|fold={fold}|role={role}|trial={trial_token}|bank={bank_token}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") & MAX_SIGNED_63


def _contract_totality_checks() -> dict[str, Any]:
    _require(MODEL_BANK_COUNT == 4 and MODEL_BANK_SIZE == 200, "model-bank dimensions changed")
    _require(CALIBRATION_TRIAL_COUNT == 999, "Monte-Carlo trial count changed")
    _require(FAMILY_ALPHA == 0.05 and PER_FOLD_ALPHA == 0.01, "alpha values changed")
    probe = torch.ones((1, DIMENSION), dtype=torch.complex128)
    probe[:, 1::2] = 1j
    canonical = canonicalize_existing_global_phase(probe)
    phase = extract_complete_relative_phase(probe)
    embedding = complex_real_embedding(probe)
    _require(canonical.shape == (1, DIMENSION) and phase.shape == (1, DIMENSION) and embedding.shape == (1, 2 * DIMENSION), "representation probe shape changed")
    try:
        extract_complete_relative_phase(torch.zeros((1, DIMENSION), dtype=torch.complex128))
    except InvalidHarnessError:
        zero_rejection = True
    else:
        zero_rejection = False
    _require(zero_rejection, "exact-zero phase rejection is not fail-closed")
    sample_seeds = [
        derive_seed("candidate", fold, role, trial, bank)
        for fold in range(FOLD_COUNT)
        for role, trial, bank in (("model_bank", None, 0), ("pseudo", 0, None))
    ]
    _require(len(set(sample_seeds)) == len(sample_seeds), "RNG streams are not disjoint")
    _require(all(0 <= seed <= MAX_SIGNED_63 for seed in sample_seeds), "RNG seed is outside signed 63-bit range")
    return {
        "status": "passed",
        "representation_probe": "passed",
        "exact_zero_rejection": "passed",
        "model_bank_count": MODEL_BANK_COUNT,
        "model_bank_size": MODEL_BANK_SIZE,
        "calibration_trials": CALIBRATION_TRIAL_COUNT,
        "family_alpha": FAMILY_ALPHA,
        "per_fold_alpha": PER_FOLD_ALPHA,
        "rng_namespace": RNG_NAMESPACE,
        "q_values_loaded": False,
        "complex_ols_values_loaded": False,
        "sealed_values_loaded": False,
        "scientific_evaluation_executed": False,
        "training_executed": False,
    }


def _build_payload(
    repo_root: Path,
    git: Mapping[str, Any],
    current: Mapping[str, Any],
    prior: Mapping[str, Any],
    v2: Mapping[str, Any],
    complex_ols: Mapping[str, Any],
    metrics: Mapping[str, Any],
    totality: Mapping[str, Any],
) -> dict[str, Any]:
    definition_ref = _ref(repo_root, Path(__file__).resolve().relative_to(repo_root.resolve()), "this contract definition consumer")
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
            "v2": dict(v2),
            "project_complex_ols": dict(complex_ols),
            "selected_producer_and_prior_confirmation": dict(prior),
            "metric_and_evaluator": dict(metrics),
            "definition_implementation": definition_ref,
        },
        "semantic_authorities": {
            "PROJECT_COMPLEX_OLS_V1": {
                "role": "complete complex OLS row and source-identity authority",
                "binding": "historical immutable OLS semantics plus exact future V2 source-row materialization",
                "coordinate": "R3_COMPLEX",
            },
            "OBSERVED_RELATIVE_PHASE_INHERITANCE_V1": {
                "role": "contract-local observed complete-vector relative-phase inheritance assumption",
                "status": "FROZEN_BY_THIS_CONTRACT_ONLY",
                "source_semantics": "GLOBAL_PHASE_GAUGE_CONFIRMED plus PHASE_SEMANTIC_INCREMENT_SUPPORTED",
                "not_a_new_phase_producer": True,
                "not_a_physical_phase_reference": True,
            },
            "N3_DEVELOPMENT_BANK_V2": {
                "role": "frozen development source/q/fold authority",
                "source_manifest_sha256": V2_SOURCE_MANIFEST_SHA256,
                "q_manifest_sha256": V2_Q_MANIFEST_SHA256,
                "fold_manifest_sha256": V2_FOLD_MANIFEST_SHA256,
            },
        },
        "representation": {
            "dimension": DIMENSION,
            "complex_state": "z_j=A_j(q)*exp(i*phi_j)",
            "q_to_amplitude": {
                "q_coordinate": "R2_AMPLITUDE",
                "A_j(q)": "q_j",
                "q_definition": "q_j=abs(c_j/sqrt(s_j))=abs(R3_COMPLEX_j)",
                "additional_amplitude_transform": "none",
                "mapping_unique": True,
            },
            "complex_ols": {
                "coordinate": "PROJECT_COMPLEX_OLS_V1 / R3_COMPLEX",
                "formula": "z_j=c_j/sqrt(s_j)",
                "input_dtype": "torch.complex64",
                "canonical_dtype": "torch.complex128",
                "shape": ["N", DIMENSION],
                "physical_coordinate_metadata_required": True,
            },
            "global_phase": {
                "existing_authority": "GLOBAL_PHASE_GAUGE_CONFIRMED",
                "primary_historical_geometry": "global_phase_gauge_distance",
                "canonicalization": "deterministic cross-section of the existing quotient; first exact argmax(abs(z)) anchor, ties by smallest coordinate index, multiply by conjugate(anchor)/abs(anchor), then set anchor to exact non-negative real magnitude",
                "new_physical_gauge_invented": False,
                "common_phase_invariance_preserved": True,
            },
        },
        "phase_semantics": {
            "authority_id": "OBSERVED_RELATIVE_PHASE_INHERITANCE_V1",
            "source": "complete fitting R3_COMPLEX row vectors only",
            "representation": "phi_j=atan2(Im(z_j^canonical), Re(z_j^canonical))",
            "wrap_interval": "[-pi, pi]",
            "unwrap": False,
            "vector_unit": "one complete 100-coordinate vector per OLS row",
            "phase_donor_unit": "complete vector; per-coordinate donor mixing forbidden",
            "zero_policy": "any exact-zero observed/fitting complex coordinate or generated amplitude coordinate is INVALID_HARNESS; no phase imputation, masking, or epsilon replacement",
            "near_zero_policy": "strictly positive finite coordinates are retained without threshold, floor, masking, or special branch",
            "signed_zero_and_pi_boundary": "use torch.atan2 on canonical float64 representatives; retain its principal-interval boundary result without remapping",
            "serialization": "CPU contiguous torch.float64 phase vectors; finite-only; JSON contract fields are UTF-8 sorted-key newline-terminated",
        },
        "development_isolation": {
            "source_authority": "N3_DEVELOPMENT_BANK_V2",
            "fold_count": FOLD_COUNT,
            "fit_rows_per_fold": FIT_ROWS_PER_FOLD,
            "fit_rows_per_class_per_fold": FIT_ROWS_PER_CLASS_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_ROWS_PER_FOLD,
            "heldout_rows_per_class_per_fold": HELDOUT_ROWS_PER_CLASS_PER_FOLD,
            "producer_fit_scope": "each candidate fit uses only that fold's 4000 q fitting rows",
            "complex_ols_fit_scope": "phase donor pool and control pool use only that fold's corresponding 4000 fitting complex rows",
            "heldout_scope": "heldout complex rows are read/materialized only for final fold evaluation after all fold-local fitting and calibration are frozen",
            "heldout_phase_donors": False,
            "cross_fold_phase_donors": False,
            "labels_as_inputs": False,
            "runtime_reselection": False,
            "runtime_resplit": False,
        },
        "control": {
            "id": "FULL_COMPLEX_ROW_EMPIRICAL_BOOTSTRAP_CONTROL",
            "selectable": False,
            "purpose": "establish fit-to-heldout stability of the complete complex OLS state before interpreting the inheritance candidate",
            "fit_rows": FIT_ROWS_PER_FOLD,
            "sampling_unit": "complete observed fitting R3_COMPLEX row",
            "sampling": "with replacement; preserve amplitude-phase pairing and all 100 coordinates",
            "q_producer_used": False,
            "phase_donor_selection": False,
            "heldout_rows_used_for_sampling": False,
            "decision_role": "first decision; if control is below 5/5, stop and do not classify inheritance as unsupported",
        },
        "candidate": {
            "id": "OBSERVED_PHASE_INDEPENDENT_INHERITANCE_V1",
            "selectable": True,
            "family": "minimal independence/inheritance candidate only",
            "assumption": "selected Gaussian q producer may combine with an empirical observed relative-phase marginal without q-dependent phase selection",
            "per_fold": {
                "q_producer": "fit the frozen EMPIRICAL_GAUSSIAN_COPULA family on that fold's 4000 q fitting rows; do not use the all-5000 final state for fold fitting",
                "phase_donor_pool": "all 4000 corresponding fitting R3_COMPLEX rows for that fold, canonicalized as complete vectors",
                "q_sampling": "sample a 100-coordinate q row from the fold-local frozen Gaussian q producer",
                "phase_sampling": "sample one complete phase donor vector with replacement, independently of q",
                "combination": "z_j=q_j*exp(i*phi_j), then canonicalize under the existing global-phase quotient",
                "q_dependent_phase_selection": False,
                "conditional_phase_model": False,
            },
            "interpretation_if_fail": "supports only that this independent inheritance candidate is not compatible under the contract after the control passes; it does not identify the missing conditional model",
        },
        "model_self_calibration": {
            "required_before_heldout": True,
            "per_fold_and_method": True,
            "pseudo_data_rows": HELDOUT_ROWS_PER_FOLD,
            "model_bank_count": MODEL_BANK_COUNT,
            "model_bank_size": MODEL_BANK_SIZE,
            "statistic": "T=mean_k energy_distance(embedding(pseudo_or_heldout), embedding(model_bank_k))",
            "model_self_trial": "T_b=mean_k ED(pseudo_b, model_bank_k)",
            "heldout_statistic": "T_obs=mean_k ED(heldout, model_bank_k)",
            "trials": CALIBRATION_TRIAL_COUNT,
            "trial_indices": [0, CALIBRATION_TRIAL_COUNT - 1],
            "all_trial_indices_inclusive": [0, CALIBRATION_TRIAL_COUNT - 1],
            "rng": {
                "namespace": RNG_NAMESPACE,
                "generator": "torch.Generator(device='cpu') per explicit stream",
                "seed_rule": "sha256(NAMESPACE|method=<control_or_candidate>|fold=<0..4>|role=<role>|trial=<index_or_none>|bank=<index_or_none>) first 64 bits masked to signed 63-bit",
                "roles": [
                    "CONTROL_MODEL_BANK",
                    "CONTROL_PSEUDO_DATA",
                    "CANDIDATE_MODEL_BANK",
                    "CANDIDATE_PSEUDO_DATA",
                    "CANDIDATE_Q_SAMPLE",
                    "CANDIDATE_PHASE_DONOR",
                ],
                "no_global_rng": True,
                "no_seed_retry": True,
                "no_early_stop": True,
                "best_of_n": False,
            },
        },
        "evaluation": {
            "order": ["control", "candidate_if_control_passes"],
            "gauge_step": "canonicalize both generated and observed complete complex rows before embedding",
            "real_embedding": {
                "formula": "concat(real(z^canonical), imag(z^canonical), dim=1)",
                "shape": ["N", 2 * DIMENSION],
                "dtype": "torch.float64",
                "phase_unwrapped_euclidean": False,
                "raw_complex_euclidean_primary": False,
            },
            "primary": {
                "name": PRIMARY_METRIC,
                "role": "HARD_PRIMARY",
                "implementation": dict(metrics["energy_distance"]),
            },
            "secondary": {
                "role": "DESCRIPTIVE_ONLY",
                "may_be_reported": True,
                "may_change_decision": False,
                "no_multi_guardrail_veto": True,
                "categories": [
                    "amplitude_q_discrepancy",
                    "phase_circular_discrepancy",
                    "amplitude_phase_dependence_summaries",
                ],
                "candidate_specific_threshold_tuning": False,
            },
            "model_bank_count": MODEL_BANK_COUNT,
            "model_bank_size": MODEL_BANK_SIZE,
            "sample_size_equalization": "each pseudo and heldout DATA side has 1000 rows; each model bank has 200 rows; the frozen ED implementation accepts unequal row counts",
        },
        "decision": {
            "family_alpha": FAMILY_ALPHA,
            "per_fold_alpha": PER_FOLD_ALPHA,
            "multiple_testing": "Bonferroni across five folds; five-fold strict aggregate",
            "p_value": "p_MC=(1+count(T_b>=T_obs))/(B+1)",
            "inequality": "greater-than-or-equal, including ties",
            "B": CALIBRATION_TRIAL_COUNT,
            "fold_pass": "p_MC>=0.01",
            "aggregate_pass": "all five folds pass (5/5)",
            "outcomes": {
                "control_below_5_of_5": "N3_COMPLEX_OLS_JOINT_DEVELOPMENT_STABILITY_NOT_ESTABLISHED",
                "control_5_of_5_candidate_5_of_5": "N3_COMPLEX_OLS_INDEPENDENT_PHASE_INHERITANCE_SUPPORTED",
                "control_5_of_5_candidate_below_5_of_5": "N3_COMPLEX_OLS_INDEPENDENT_PHASE_INHERITANCE_NOT_SUPPORTED",
                "invalid_measurement_or_authority": "INVALID_HARNESS",
            },
            "candidate_not_evaluated_after_control_failure": True,
            "no_automatic_repair": True,
        },
        "claim_boundary": {
            "candidate_pass_supports_only": "selected Gaussian q amplitude plus a complete independently sampled observed fitting relative-phase vector is compatible with heldout complete complex OLS under this exact contract",
            "candidate_fail_supports_only": "the independence/inheritance candidate is not compatible after complex fit-to-heldout stability was established",
            "control_fail_supports_only": "complete complex OLS fit-to-heldout development stability was not established",
            "not_established": [
                "a novel phase producer",
                "q-dependent or conditional phase law",
                "complex OLS population equality or OOD generalization",
                "semantic controllability",
                "physical optical realizability",
                "image-generation readiness",
                "decoder or CMOS readiness",
                "training authorization",
                "formal promotion",
            ],
        },
        "forbidden": {
            "sealed_reuse": True,
            "new_sealed_authority": True,
            "phase_neural_network": True,
            "phase_knn": True,
            "phase_mixture_or_flow": True,
            "class_conditioned_phase": True,
            "semantic_or_image_evaluation": True,
            "decoder_or_optical_forward": True,
            "training": True,
            "cuda": True,
            "N4_N5_N6": True,
            "amplitude_producer_modification": True,
        },
        "firewalls": {
            "q_values_loaded_by_definition": False,
            "complex_ols_values_loaded_by_definition": False,
            "sealed_values_loaded_by_definition": False,
            "sealed_content_reused": False,
            "sealed_consumed_state_reopened": False,
            "final_producer_state_used_as_fold_fit": False,
            "final_confirmation_used_as_training_data": False,
            "heldout_phase_donors": False,
            "q_phase_pairing_leakage": False,
            "producer_modified": False,
            "new_producer_family": False,
            "scientific_evaluation_executed": False,
            "training": False,
            "cuda": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "training_authorized": False,
            "formal_promotion_flag": False,
        },
        "contract_totality": dict(totality),
        "next_action": "EXECUTE_N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_V1",
    }


def execute(repo_root: Path, output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    _require(not output_dir.exists(), f"refusing to reuse write-once contract output: {output_dir}")
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the authority branch")
    _require(git["head"] == EXPECTED_STARTING_HEAD, "current committed HEAD is not the frozen authority base")
    current = _validate_current_state(repo_root)
    prior = _validate_prior_provenance(repo_root)
    v2 = _validate_v2_authority(repo_root)
    complex_ols = _validate_complex_ols_authority(repo_root)
    metrics = _validate_metric_authority(repo_root)
    totality = _contract_totality_checks()
    payload = _build_payload(repo_root, git, current, prior, v2, complex_ols, metrics, totality)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(
            temporary / CONTRACT_NAME,
            payload,
            artifact_kind="N3 complex OLS amplitude/phase joint compatibility contract",
        )
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
        "semantic_authorities": ["PROJECT_COMPLEX_OLS_V1", "OBSERVED_RELATIVE_PHASE_INHERITANCE_V1", "N3_DEVELOPMENT_BANK_V2"],
        "producer_semantic_authority": PRODUCER_STATE_SEMANTIC_ID,
        "producer_state_sha256": PRODUCER_STATE_SHA256,
        "prior_sealed_confirmation_consumed": True,
        "q_values_loaded": False,
        "complex_ols_values_loaded": False,
        "sealed_values_loaded": False,
        "scientific_evaluation_executed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "next_action": "EXECUTE_N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_V1",
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
