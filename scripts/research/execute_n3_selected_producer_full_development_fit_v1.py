"""Execute the one-shot N3 selected-producer full development fit.

This consumer is deliberately narrower than the historical producer
comparison.  It consumes the already-selected empirical Gaussian-copula
candidate once on all 5,000 rows of N3_DEVELOPMENT_BANK_V2, serializes one
standalone CPU state, and records the lineage needed by the next contract.
It never reads sealed confirmation values, performs compatibility evaluation,
opens the optical/semantic path, or trains a model.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import subprocess
import tempfile
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
from scripts.research import execute_n3_empirical_marginal_gaussian_copula_producer_v1 as gaussian


TASK_ID = "EXECUTE_N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_V1"
GATE = TASK_ID
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"

CONTRACT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_selected_producer_freeze_contract_v1_attempt_01/"
    "n3_selected_producer_freeze_contract.v1.json"
)
EXPECTED_CONTRACT_SHA256 = "4d8d3f5dc81a7eff2d92a0acde0a0d0639788bb0dee75e6d88db745f642b60a3"

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
V2_CEILING_REPORT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_development_bank_v2_empirical_ceiling_v1_attempt_01/"
    "n3_development_bank_v2_empirical_ceiling_report.v1.json"
)
V2_CEILING_REPORT_SHA256 = "731cf826dc081e7cc5629790ba2fb74946f7cc41f1a7f0035bf3a3ca96a31826"

V3_REPLAY_REPORT_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_q_compatibility_evaluator_v3_frozen_candidate_replay_v1_attempt_01/"
    "n3_q_compatibility_evaluator_v3_frozen_candidate_replay_report.v1.json"
)
V3_REPLAY_REPORT_SHA256 = "cf6e6d62e6d2a7b35822d2d3727d08e5cbd59d51778bf794ec30ce297ffd0195"
V3_REPLAY_LINEAGE_PATH = Path(
    "outputs/20260909_v2/runs/"
    "n3_q_compatibility_evaluator_v3_frozen_candidate_replay_v1_attempt_01/"
    "n3_q_compatibility_evaluator_v3_frozen_candidate_replay_lineage.v1.json"
)
V3_REPLAY_LINEAGE_SHA256 = "d49800c12ce5f7e4931aca301410d5f0c47ca301350324fdfae3a73526d19ee4"

GAUSSIAN_IMPLEMENTATION_PATH = Path(
    "scripts/research/execute_n3_empirical_marginal_gaussian_copula_producer_v1.py"
)
GAUSSIAN_IMPLEMENTATION_SHA256 = "5f1fa3fa6be35e96f345c3b7f7c6cce5ca0b45fe4619b1ead4455e86dcae538a"

GOVERNANCE_STATE_PATH = Path("docs/status/research_governance_state.json")

OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "n3_selected_producer_full_development_fit_v1_attempt_01"
)
STATE_NAME = "n3_selected_empirical_gaussian_copula_producer_state.v1.pt"
LINEAGE_NAME = "n3_selected_producer_full_development_fit_lineage.v1.json"

STATE_SCHEMA = "n3_selected_empirical_gaussian_copula_producer_state.v1"
LINEAGE_SCHEMA = "n3_selected_producer_full_development_fit_lineage.v1"
SAMPLING_INTERFACE_SCHEMA = "n3_selected_empirical_gaussian_copula_sampling_interface.v1"
PRODUCER_FAMILY_ID = "EMPIRICAL_GAUSSIAN_COPULA"
PRODUCER_CANDIDATE_ID = "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1"
PRODUCER_STATE_SEMANTIC_ID = "N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1"
RNG_NAMESPACE = "N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1_CPU"

CLASS_COUNT = 10
FULL_FIT_ROWS = 5000
ROWS_PER_CLASS = 500
DIMENSION = 100
FOLD_COUNT = 5
DIAGONAL_ATOL = 5.684341886080801e-14
MAX_SIGNED_63 = (1 << 63) - 1

VALIDATION_SMOKE_COUNT = 3
VALIDATION_SMOKE_SEED = 20260910

CLASSIFICATION = "N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_COMPLETED"


class InvalidHarnessError(RuntimeError):
    """A frozen authority or state/output invariant failed."""


class BlockedError(RuntimeError):
    """The requested write-once execution cannot start in the current state."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidHarnessError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


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


def _read_governance(repo_root: Path) -> tuple[dict[str, Any], str]:
    target = repo_root / GOVERNANCE_STATE_PATH
    _require(target.is_file(), f"current governance state is missing: {target}")
    digest = file_sha256(target)
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidHarnessError(f"current governance state cannot be read: {target}") from exc
    return dict(_mapping(value, "current governance state")), digest


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


def _validate_contract(contract: Mapping[str, Any]) -> None:
    _require(contract.get("schema") == "n3_selected_producer_freeze_contract.v1", "freeze contract schema changed")
    _require(contract.get("task_id") == "DEFINE_N3_SELECTED_PRODUCER_FREEZE_CONTRACT_V1", "freeze contract task changed")
    _require(contract.get("gate") == contract["task_id"], "freeze contract gate changed")
    _require(contract.get("status") == "valid_completed", "freeze contract is not complete")
    _require(contract.get("result") == "VALID_COMPLETED", "freeze contract result changed")
    _require(
        contract.get("classification") == "DEFINE_N3_SELECTED_PRODUCER_FREEZE_CONTRACT_V1_READY",
        "freeze contract is not ready",
    )
    _require(contract.get("next_action") == TASK_ID, "freeze contract next action changed")
    _require(contract.get("only_next_action_authorized") is True, "freeze contract action boundary changed")

    selection = _mapping(contract.get("selection"), "freeze contract selection")
    _require(selection.get("selected_development_producer_candidate") == PRODUCER_FAMILY_ID, "selected producer family changed")
    _require(selection.get("selected_candidate_id") == PRODUCER_CANDIDATE_ID, "selected producer candidate changed")
    _require(selection.get("project_selected_producer") is None, "project producer was selected before the fit")
    _require(selection.get("frozen_producer_state") is None, "producer state already exists in the contract")
    _require(selection.get("posthoc_selection") is False, "posthoc selection is enabled")
    _require(selection.get("secondary_metric_ranking_required") is False, "secondary metric ranking is enabled")

    family = _mapping(contract.get("selected_producer_family"), "selected producer family")
    for key, expected in {
        "family_id": PRODUCER_FAMILY_ID,
        "candidate_id": PRODUCER_CANDIDATE_ID,
        "candidate_version": "v1",
        "coordinate": "R2_AMPLITUDE",
        "input_q_dimensionality": DIMENSION,
        "output_q_dimensionality": DIMENSION,
        "labels_as_inputs": False,
        "analytical_fitting": True,
        "fitting_only_empirical_marginals": True,
        "gradient_training": False,
        "no_mixture_or_local_components": True,
        "no_model_order_or_rank_selection": True,
        "no_new_model_or_hyperparameter_choice": True,
    }.items():
        _require(family.get(key) == expected, f"selected producer family field changed: {key}")
    implementation = _mapping(family.get("implementation_identity"), "selected producer implementation")
    _require(implementation.get("path") == GAUSSIAN_IMPLEMENTATION_PATH.as_posix(), "selected implementation path changed")
    _require(implementation.get("sha256") == GAUSSIAN_IMPLEMENTATION_SHA256, "selected implementation hash changed")
    unchanged = _mapping(family.get("unchanged_candidate_semantics"), "unchanged candidate semantics")
    for key, expected in {
        "best_of_n": False,
        "rejection": False,
        "retry": False,
        "samplewise_normalization": False,
        "numeric_dtype": "CPU torch.float64",
        "q_representation": "R2_AMPLITUDE = abs(c/sqrt(s)) = abs(R3_COMPLEX)",
        "gaussianization": "torch.special.ndtri((average_rank-0.5)/n_fit)",
        "dependence": "analytic R_emp then alpha-shrunk R_shrunk",
    }.items():
        _require(unchanged.get(key) == expected, f"selected producer semantics changed: {key}")

    fit = _mapping(contract.get("full_development_fit"), "full development fit contract")
    fit_scope = _mapping(fit.get("fit_scope"), "full fit scope")
    for key, expected in {
        "authority_id": "N3_DEVELOPMENT_BANK_V2",
        "coordinate": "R2_AMPLITUDE",
        "row_count": FULL_FIT_ROWS,
        "rows_per_class": ROWS_PER_CLASS,
        "all_development_rows": True,
        "heldout_rows_used": False,
        "folds_used_for_fit": False,
        "fold_evidence_role": "prior model selection only",
        "labels_as_inputs": False,
        "device": "cpu",
        "dtype": "torch.float64",
        "fit_execution_count": 1,
    }.items():
        _require(fit_scope.get(key) == expected, f"full fit scope changed: {key}")
    fixed = _mapping(fit.get("fixed_input_semantics"), "fixed input semantics")
    for key, expected in {
        "labels_condition_producer": False,
        "q_clipping": False,
        "q_normalization": False,
        "q_definition": "q=abs(c/sqrt(s))=abs(R3_COMPLEX)",
        "q_finite_and_nonnegative_required": True,
        "q_shape": [FULL_FIT_ROWS, DIMENSION],
    }.items():
        _require(fixed.get(key) == expected, f"fixed q semantics changed: {key}")
    no_choice = _mapping(fit.get("no_new_choice"), "no-new-choice contract")
    _require(no_choice and all(value is False for value in no_choice.values()), "full fit contains an unresolved choice")
    _require(_mapping(fit.get("failure_semantics"), "fit failure semantics").get("classification") == "INVALID_HARNESS", "fit failure classification changed")

    state_contract = _mapping(contract.get("producer_state_contract"), "producer state contract")
    _require(state_contract.get("semantic_authority_id") == PRODUCER_STATE_SEMANTIC_ID, "state semantic authority changed")
    _require(state_contract.get("schema") == STATE_SCHEMA, "state schema changed")
    _require(state_contract.get("state_exists_now") is False, "producer state already exists")
    _require(state_contract.get("frozen_producer_state") is None, "producer state is already frozen")
    _require(state_contract.get("provenance_fields", {}).get("sealed_confirmation_included") is False, "sealed information is included in state provenance")
    serialization = _mapping(state_contract.get("serialization"), "state serialization contract")
    for key, expected in {
        "format": "torch.save mapping",
        "safe_load_contract": "torch.load(weights_only=True, map_location='cpu')",
        "device": "cpu",
        "tensor_dtype": "torch.float64",
        "atomic_publication": True,
        "write_once": True,
        "custom_classes": False,
        "no_raw_development_q_rows": True,
        "no_sealed_information": True,
    }.items():
        _require(serialization.get(key) == expected, f"state serialization contract changed: {key}")
    required_sampling = _mapping(state_contract.get("required_sampling_tensors"), "required sampling tensors")
    for name, shape in {"sorted_values": [FULL_FIT_ROWS, DIMENSION], "quantile_table_positions": [FULL_FIT_ROWS], "cholesky": [DIMENSION, DIMENSION]}.items():
        item = _mapping(required_sampling.get(name), f"required sampling tensor {name}")
        _require(item.get("shape") == shape and item.get("dtype") == "torch.float64", f"required sampling tensor changed: {name}")
    required_audit = _mapping(state_contract.get("required_audit_tensors"), "required audit tensors")
    for name, shape in {
        "plotting_positions": [FULL_FIT_ROWS, DIMENSION],
        "z_mean": [DIMENSION],
        "standard_deviation": [DIMENSION],
        "R_emp": [DIMENSION, DIMENSION],
        "R_shrunk": [DIMENSION, DIMENSION],
        "eigenvalues": [DIMENSION],
        "diagonal": [DIMENSION],
    }.items():
        item = _mapping(required_audit.get(name), f"required audit tensor {name}")
        _require(item.get("shape") == shape and item.get("dtype") == "torch.float64", f"required audit tensor changed: {name}")
    _require(state_contract.get("required_audit_scalars") == ["tau", "beta", "delta", "alpha", "diagonal_error"], "required audit scalars changed")

    sampling = _mapping(contract.get("standalone_sampling_interface"), "standalone sampling interface")
    _require(sampling.get("schema") == SAMPLING_INTERFACE_SCHEMA, "sampling interface schema changed")
    _require(sampling.get("rng_namespace") is None or sampling.get("rng_namespace") == RNG_NAMESPACE, "sampling namespace changed")
    inputs = _mapping(sampling.get("inputs"), "sampling inputs")
    state_input = _mapping(inputs.get("producer_state"), "sampling state input")
    _require(state_input.get("schema") == STATE_SCHEMA and state_input.get("semantic_authority_id") == PRODUCER_STATE_SEMANTIC_ID, "sampling state input changed")
    _require(_mapping(inputs.get("requested_sample_count"), "sampling count input").get("minimum") == 1, "sampling count boundary changed")
    namespace_input = _mapping(inputs.get("rng_namespace"), "sampling namespace input")
    _require(namespace_input.get("must_equal") == RNG_NAMESPACE and namespace_input.get("required") is True, "sampling namespace requirement changed")
    seed_input = _mapping(inputs.get("seed"), "sampling seed input")
    _require(seed_input.get("required") is True and seed_input.get("hidden_derivation") is False, "sampling seed requirement changed")
    prohibited = _mapping(sampling.get("prohibited_sampling_behavior"), "prohibited sampling behavior")
    _require(prohibited and all(value is False for value in prohibited.values()), "sampling behavior is no longer frozen")

    _require(contract.get("selection_authority_replay_report_sha256") == V3_REPLAY_REPORT_SHA256, "selection replay authority changed")
    boundary = _mapping(contract.get("stage_boundary"), "stage boundary")
    sealed_boundary = _mapping(boundary.get("sealed_confirmation"), "sealed stage boundary")
    _require(sealed_boundary.get("status") == "RESERVED_NOT_MATERIALIZED", "sealed status changed in contract")
    _require(sealed_boundary.get("consumed") is False and sealed_boundary.get("input_to_this_consumer") is False, "sealed input boundary changed")
    governance = _mapping(contract.get("governance"), "freeze contract governance")
    for key, expected in {
        "selected_producer": None,
        "frozen_producer_state": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_consumed": False,
        "sealed_confirmation_status": "RESERVED_NOT_MATERIALIZED",
    }.items():
        _require(governance.get(key) == expected, f"freeze contract governance changed: {key}")


def _validate_v2_source(source: Mapping[str, Any]) -> dict[str, Any]:
    _require(source.get("schema") == "n3_development_bank_v2_source_manifest.v1", "V2 source schema changed")
    _require(source.get("semantic_authority_id") == "N3_DEVELOPMENT_BANK_V2", "V2 source semantic authority changed")
    _require(source.get("status") == "FROZEN_SOURCE_ROWS", "V2 source status changed")
    identity = _mapping(source.get("identity"), "V2 source identity")
    _require(identity.get("class_counts") == [ROWS_PER_CLASS] * CLASS_COUNT, "V2 class balance changed")
    _require(identity.get("unique_source_rows") is True, "V2 source rows are not unique")
    _require(identity.get("selected_source_indices_sha256") == "8e6859cbe0ad10090b1d5d7c55de28232f9db30c45093078dbf29f8545473a0e", "V2 source index identity changed")
    _require(identity.get("selected_labels_sha256") == "d28f06d6ecde9d08797f954e60041f79b512e2ad61480e21d5702cb841fabef3", "V2 source label identity changed")
    _require(identity.get("row_manifest_sha256") == "ba3149357dc90f14ad692a486593ce944d48453f2b561ad10a9323b9fe7fb957", "V2 row manifest identity changed")

    rows_value = source.get("rows")
    _require(isinstance(rows_value, list) and len(rows_value) == FULL_FIT_ROWS, "V2 source row manifest is incomplete")
    rows = [dict(_mapping(row, "V2 source row")) for row in rows_value]
    positions = [int(row.get("bank_row_position")) for row in rows]
    source_indices = [int(row.get("source_index")) for row in rows]
    labels = [int(row.get("label")) for row in rows]
    stable_identities = [str(row.get("stable_source_row_identity")) for row in rows]
    _require(positions == list(range(FULL_FIT_ROWS)), "V2 row positions changed")
    _require(len(set(source_indices)) == FULL_FIT_ROWS, "V2 source indices are not unique")
    _require(len(set(stable_identities)) == FULL_FIT_ROWS, "V2 stable source identities are not unique")
    _require(all(len(value) == 64 for value in stable_identities), "V2 stable source identity format changed")
    _require(labels.count(0) == ROWS_PER_CLASS and all(labels.count(label) == ROWS_PER_CLASS for label in range(CLASS_COUNT)), "V2 row labels are not balanced")
    _require(tensor_sha256(torch.tensor(source_indices, dtype=torch.int64)) == identity["selected_source_indices_sha256"], "V2 source-index content changed")
    _require(tensor_sha256(torch.tensor(labels, dtype=torch.int64)) == identity["selected_labels_sha256"], "V2 label content changed")

    exclusions = _mapping(source.get("exclusions"), "V2 exclusion registry")
    overlap_counts = _mapping(exclusions.get("v2_overlap_counts"), "V2 overlap counts")
    _require(overlap_counts and all(int(value) == 0 for value in overlap_counts.values()), "V2 source overlap is nonzero")
    union = _mapping(exclusions.get("union"), "V2 exclusion union")
    _require(union.get("pairwise_overlap_zero") is True, "V2 exclusion union is not disjoint")
    sealed = _mapping(_mapping(exclusions.get("authorities"), "V2 exclusion authorities").get("N3_SEALED_CONFIRMATION_RESERVATION"), "sealed reservation metadata")
    _require(
        sealed.get("status") == "RESERVED_NOT_MATERIALIZED"
        and sealed.get("materialized") is False
        and sealed.get("ols_values_loaded") is False
        and sealed.get("q_values_loaded") is False,
        "sealed reservation metadata changed",
    )
    protected_registry = source.get("authority_inputs", {}).get("protected_panel_registry")
    _require(isinstance(protected_registry, list) and protected_registry, "protected identity registry is missing")
    for item_value in protected_registry:
        item = _mapping(item_value, "protected identity entry")
        _require(item.get("status") in {"PASS_ZERO_OVERLAP", "SUBSUMED", "NOT_APPLICABLE"}, "protected identity overlap status changed")
    boundary = _mapping(source.get("materialization_boundary"), "V2 source materialization boundary")
    for key in (
        "q_values_loaded",
        "q_values_materialized",
        "sealed_confirmation_consumed",
        "sealed_confirmation_values_loaded",
        "image_values_loaded",
        "producer_evaluated",
        "producer_selection",
        "training_authorized",
        "formal_promotion",
    ):
        _require(boundary.get(key) is False, f"V2 source boundary changed: {key}")
    return {
        "row_count": FULL_FIT_ROWS,
        "rows_per_class": ROWS_PER_CLASS,
        "source_indices_sha256": identity["selected_source_indices_sha256"],
        "labels_sha256": identity["selected_labels_sha256"],
        "row_manifest_sha256": identity["row_manifest_sha256"],
        "source_row_identity_sha256": canonical_sha256(stable_identities),
        "class_counts": [ROWS_PER_CLASS] * CLASS_COUNT,
        "overlap_counts": {str(key): int(value) for key, value in overlap_counts.items()},
        "protected_registry_count": len(protected_registry),
        "sealed_status": sealed["status"],
        "sealed_values_loaded": False,
        "sealed_consumed": False,
        "rows": rows,
    }


def _validate_q_manifest(repo_root: Path, q_manifest: Mapping[str, Any]) -> None:
    _require(q_manifest.get("schema") == "n3_development_bank_v2_q_materialization_manifest.v1", "V2 q manifest schema changed")
    _require(q_manifest.get("status") == "VALID_COMPLETED", "V2 q materialization is not complete")
    q_artifact = _mapping(q_manifest.get("q_artifact"), "V2 q artifact metadata")
    for key, expected in {
        "sha256": V2_Q_ARTIFACT_SHA256,
        "q_tensor_sha256": V2_Q_TENSOR_SHA256,
        "shape": [FULL_FIT_ROWS, DIMENSION],
        "dtype": "torch.float64",
        "schema": "n3_development_bank_v2_q.v1",
    }.items():
        _require(q_artifact.get(key) == expected, f"V2 q artifact metadata changed: {key}")
    source_ref = _mapping(q_manifest.get("source_manifest"), "V2 q source reference")
    _require(source_ref.get("path") == _relative(repo_root, repo_root / V2_SOURCE_MANIFEST_PATH), "V2 q source path changed")
    _require(source_ref.get("sha256") == V2_SOURCE_MANIFEST_SHA256, "V2 q source identity changed")
    execution = _mapping(q_manifest.get("execution"), "V2 q execution")
    for key, expected in {
        "device": "cpu",
        "sealed_confirmation_rows_read": False,
        "sealed_confirmation_values_loaded": False,
        "selected_source_rows_only": True,
        "training_steps": 0,
        "optimizer_created": False,
        "backward_called": False,
        "evaluator_entered": False,
        "decoder_entered": False,
    }.items():
        _require(execution.get(key) == expected, f"V2 q execution boundary changed: {key}")
    _require(q_manifest.get("sealed_confirmation_consumed") is False, "V2 q manifest consumed sealed data")
    governance = _mapping(q_manifest.get("governance"), "V2 q governance")
    for key, expected in {
        "producer_evaluated": False,
        "producer_selection": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items():
        _require(governance.get(key) == expected, f"V2 q governance changed: {key}")


def _validate_fold_and_ceiling(fold: Mapping[str, Any], ceiling: Mapping[str, Any]) -> None:
    _require(fold.get("schema") == "n3_development_bank_v2_five_fold_protocol.v1", "V2 fold schema changed")
    _require(fold.get("status") == "FROZEN_BEFORE_VALIDATION", "V2 fold status changed")
    protocol = _mapping(fold.get("protocol"), "V2 fold protocol")
    _require(protocol.get("assignment_sha256") == V2_ASSIGNMENT_SHA256, "V2 fold assignment changed")
    _require(protocol.get("fold_count") == FOLD_COUNT, "V2 fold count changed")
    _require(protocol.get("fit_count_per_fold") == 4000 and protocol.get("heldout_count_per_fold") == 1000, "V2 fold sizes changed")
    _require(_mapping(fold.get("governance"), "V2 fold governance").get("sealed_confirmation_consumed") is False, "V2 fold consumed sealed data")
    _require(protocol.get("q_values_used") is False and protocol.get("rng_used") is False, "V2 fold was used as a fit input")

    _require(ceiling.get("schema") == "n3_development_bank_v2_empirical_ceiling_report.v1", "V2 ceiling schema changed")
    _require(ceiling.get("status") == "valid_completed", "V2 ceiling status changed")
    _require(ceiling.get("result") == "VALID_COMPLETED", "V2 ceiling result changed")
    _require(ceiling.get("classification") == "N3_DEVELOPMENT_BANK_V2_VALIDATED", "V2 ceiling classification changed")
    aggregate = _mapping(ceiling.get("aggregate"), "V2 ceiling aggregate")
    _require(
        aggregate.get("compatible_fold_count") == FOLD_COUNT
        and aggregate.get("compatible_fold_pattern") == [True] * FOLD_COUNT
        and aggregate.get("development_supported") is True,
        "V2 ceiling validation changed",
    )
    for key, expected in {
        "producer_evaluated": False,
        "producer_selection": False,
        "selected_producer": None,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items():
        _require(_mapping(ceiling.get("governance"), "V2 ceiling governance").get(key) == expected, f"V2 ceiling governance changed: {key}")


def _validate_v3_replay(report: Mapping[str, Any], lineage: Mapping[str, Any]) -> None:
    _require(report.get("schema") == "n3_q_compatibility_evaluator_v3_frozen_candidate_replay_report.v1", "V3 replay report schema changed")
    _require(report.get("status") == "valid_completed" and report.get("result") == "VALID_COMPLETED", "V3 replay is not complete")
    _require(report.get("classification") == "N3_Q_COMPATIBILITY_EVALUATOR_V3_FROZEN_CANDIDATE_REPLAY_COMPLETE", "V3 replay classification changed")
    _require(report.get("development_supported_candidates") == [PRODUCER_CANDIDATE_ID], "V3 supported candidate changed")
    selection = _mapping(report.get("selection"), "V3 selection")
    _require(selection.get("development_supported_candidate") == PRODUCER_CANDIDATE_ID, "V3 Gaussian selection changed")
    _require(selection.get("selected_producer") is None and selection.get("producer_freeze_performed") is False, "V3 replay selected or froze a project producer")
    sealed = _mapping(report.get("sealed_confirmation"), "V3 sealed boundary")
    _require(sealed.get("status") == "RESERVED_NOT_MATERIALIZED" and sealed.get("values_loaded") is False and sealed.get("consumed") is False, "V3 replay sealed boundary changed")
    governance = _mapping(report.get("governance"), "V3 replay governance")
    _require(governance.get("selected_producer") is None and governance.get("training_authorized") is False and governance.get("formal_promotion") is False, "V3 replay governance changed")
    _require(lineage.get("schema") == "n3_q_compatibility_evaluator_v3_frozen_candidate_replay_lineage.v1", "V3 replay lineage schema changed")
    lineage_sealed = _mapping(lineage.get("sealed_confirmation"), "V3 replay lineage sealed boundary")
    _require(lineage_sealed.get("status") == "RESERVED_NOT_MATERIALIZED" and lineage_sealed.get("values_loaded") is False and lineage_sealed.get("consumed") is False, "V3 replay lineage consumed sealed data")


def _validate_current_governance(governance: Mapping[str, Any]) -> None:
    _require(governance.get("active_objective") == "N3_SELECTED_PRODUCER_FREEZE_CONTRACT_V1_READY", "current governance objective changed before full fit")
    _require(governance.get("hard_blocker") == "N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_REQUIRED", "current governance blocker changed before full fit")
    aliases = _mapping(governance.get("evidence_aliases"), "current governance aliases")
    alias = _mapping(aliases.get("N3_SELECTED_PRODUCER_FREEZE_CONTRACT_V1"), "freeze contract governance alias")
    _require(alias.get("sha256") == EXPECTED_CONTRACT_SHA256, "freeze contract governance binding changed")
    for key, expected in {
        "selected_producer": None,
        "frozen_producer_state": None,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_consumed": False,
    }.items():
        _require(alias.get(key) == expected, f"freeze contract governance alias changed: {key}")


def preflight(repo_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    """Run the bounded metadata-only preflight before loading development q."""

    root = repo_root.resolve()
    target_dir = (root / (output_dir or OUTPUT_DIR)).resolve()
    if target_dir.exists():
        raise BlockedError(f"write-once output already exists: {target_dir}")
    git = _git_state(root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 research authority branch")
    _require(git["head"] == EXPECTED_HEAD, "current HEAD is not the freeze-contract authority head")

    contract = _read_json_exact(root, CONTRACT_PATH, EXPECTED_CONTRACT_SHA256, "selected-producer freeze contract")
    _validate_contract(contract)
    source = _read_json_exact(root, V2_SOURCE_MANIFEST_PATH, V2_SOURCE_MANIFEST_SHA256, "V2 source manifest")
    source_identity = _validate_v2_source(source)
    q_manifest = _read_json_exact(root, V2_Q_MANIFEST_PATH, V2_Q_MANIFEST_SHA256, "V2 q materialization manifest")
    _validate_q_manifest(root, q_manifest)
    q_artifact_target = root / V2_Q_ARTIFACT_PATH
    _require(q_artifact_target.is_file(), f"V2 q artifact is missing: {q_artifact_target}")
    _require(file_sha256(q_artifact_target) == V2_Q_ARTIFACT_SHA256, "V2 q artifact bytes changed")
    fold = _read_json_exact(root, V2_FOLD_MANIFEST_PATH, V2_FOLD_MANIFEST_SHA256, "V2 five-fold manifest")
    ceiling = _read_json_exact(root, V2_CEILING_REPORT_PATH, V2_CEILING_REPORT_SHA256, "V2 empirical ceiling report")
    _validate_fold_and_ceiling(fold, ceiling)
    replay = _read_json_exact(root, V3_REPLAY_REPORT_PATH, V3_REPLAY_REPORT_SHA256, "V3 frozen-candidate replay report")
    replay_lineage = _read_json_exact(root, V3_REPLAY_LINEAGE_PATH, V3_REPLAY_LINEAGE_SHA256, "V3 frozen-candidate replay lineage")
    _validate_v3_replay(replay, replay_lineage)
    governance, governance_sha256 = _read_governance(root)
    _validate_current_governance(governance)
    _require(file_sha256(root / GAUSSIAN_IMPLEMENTATION_PATH) == GAUSSIAN_IMPLEMENTATION_SHA256, "selected Gaussian implementation bytes changed")
    return {
        "repo_root": root,
        "output_dir": target_dir,
        "git": git,
        "contract": contract,
        "source": source,
        "source_identity": source_identity,
        "q_manifest": q_manifest,
        "fold": fold,
        "ceiling": ceiling,
        "replay": replay,
        "replay_lineage": replay_lineage,
        "governance": governance,
        "governance_sha256": governance_sha256,
        "sealed_status": "RESERVED_NOT_MATERIALIZED",
        "sealed_values_loaded": False,
        "sealed_consumed": False,
    }


def _load_development_q(preflight_result: Mapping[str, Any]) -> torch.Tensor:
    root = Path(preflight_result["repo_root"])
    payload = safe_torch_load(root / V2_Q_ARTIFACT_PATH, expected_sha256=V2_Q_ARTIFACT_SHA256, map_location="cpu")
    data = _mapping(payload, "V2 q artifact")
    _require(data.get("schema") == "n3_development_bank_v2_q.v1", "V2 q artifact schema changed after preflight")
    _require(data.get("semantic_authority_id") == "N3_DEVELOPMENT_BANK_V2", "V2 q semantic authority changed after preflight")
    _require(data.get("q_coordinate") == "R2_AMPLITUDE", "V2 q coordinate changed after preflight")
    _require(data.get("q_definition") == "q=abs(c/sqrt(s))=abs(R3_COMPLEX)", "V2 q definition changed after preflight")
    _require(data.get("source_manifest_sha256") == V2_SOURCE_MANIFEST_SHA256, "V2 q source binding changed after preflight")
    q = data.get("q")
    source_indices = data.get("source_indices")
    labels = data.get("labels")
    _require(torch.is_tensor(q) and q.dtype == torch.float64 and tuple(q.shape) == (FULL_FIT_ROWS, DIMENSION), "loaded V2 q shape/dtype changed")
    _require(torch.is_tensor(source_indices) and source_indices.dtype == torch.int64 and tuple(source_indices.shape) == (FULL_FIT_ROWS,), "loaded V2 source indices changed")
    _require(torch.is_tensor(labels) and labels.dtype == torch.int64 and tuple(labels.shape) == (FULL_FIT_ROWS,), "loaded V2 labels changed")
    q = q.detach().cpu().contiguous()
    source_indices = source_indices.detach().cpu().contiguous()
    labels = labels.detach().cpu().contiguous()
    _require(bool(torch.isfinite(q).all()) and bool(q.ge(0.0).all()), "loaded V2 q values are invalid")
    _require(tensor_sha256(q) == V2_Q_TENSOR_SHA256, "loaded V2 q tensor identity changed")
    expected_rows = preflight_result["source_identity"]["rows"]
    expected_indices = torch.tensor([int(row["source_index"]) for row in expected_rows], dtype=torch.int64)
    expected_labels = torch.tensor([int(row["label"]) for row in expected_rows], dtype=torch.int64)
    _require(torch.equal(source_indices, expected_indices), "loaded V2 source indices differ from frozen source rows")
    _require(torch.equal(labels, expected_labels), "loaded V2 labels differ from frozen source rows")
    _require(tensor_sha256(source_indices) == preflight_result["source_identity"]["source_indices_sha256"], "loaded V2 source-index hash changed")
    _require(tensor_sha256(labels) == preflight_result["source_identity"]["labels_sha256"], "loaded V2 label hash changed")
    identities = data.get("source_row_identity")
    expected_identities = [str(row["stable_source_row_identity"]) for row in expected_rows]
    _require(isinstance(identities, list) and [str(value) for value in identities] == expected_identities, "loaded V2 row identities changed")
    _require(data.get("source_row_identity_sha256") == preflight_result["source_identity"]["source_row_identity_sha256"], "loaded V2 row identity hash changed")
    execution = _mapping(data.get("execution"), "loaded V2 q execution")
    _require(execution.get("sealed_confirmation_values_loaded") is False and execution.get("sealed_confirmation_rows_read") is False, "loaded V2 q accessed sealed values")
    governance = _mapping(data.get("governance"), "loaded V2 q governance")
    _require(all(governance.get(key) == value for key, value in {
        "producer_evaluated": False,
        "producer_selection": False,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
    }.items()), "loaded V2 q governance changed")
    return q


def _fit_full_development_once(q: torch.Tensor) -> dict[str, Any]:
    """Call the frozen Gaussian semantic implementation exactly once."""

    previous_fit_count = gaussian.FIT_COUNT_PER_FOLD
    try:
        gaussian.FIT_COUNT_PER_FOLD = FULL_FIT_ROWS
        return gaussian._fit_fold(q)
    finally:
        gaussian.FIT_COUNT_PER_FOLD = previous_fit_count


def _cpu_tensor(value: Any, name: str) -> torch.Tensor:
    _require(torch.is_tensor(value), f"fit output is missing tensor: {name}")
    result = value.detach().cpu().to(dtype=torch.float64).contiguous()
    _require(result.requires_grad is False, f"fit output requires gradients: {name}")
    return result


def _fit_summary(fit: Mapping[str, Any], q: torch.Tensor) -> dict[str, Any]:
    dependence = _mapping(fit.get("dependence"), "full-fit dependence")
    z = _cpu_tensor(fit.get("z"), "z")
    sorted_values = _cpu_tensor(fit.get("sorted_values"), "sorted_values")
    plotting_positions = _cpu_tensor(fit.get("plotting_positions"), "plotting_positions")
    eigenvalues = _cpu_tensor(dependence.get("eigenvalues"), "eigenvalues")
    return {
        "fit_execution_count": 1,
        "fit_rows": FULL_FIT_ROWS,
        "rows_per_class": ROWS_PER_CLASS,
        "folds_used_for_fit": False,
        "heldout_rows_used": False,
        "labels_as_inputs": False,
        "q_input_tensor_sha256": tensor_sha256(q),
        "q_input_summary": {
            "min": float(q.min()),
            "mean": float(q.mean()),
            "max": float(q.max()),
        },
        "sorted_values_sha256": tensor_sha256(sorted_values),
        "plotting_positions_sha256": tensor_sha256(plotting_positions),
        "z_summary": {
            "min": float(z.min()),
            "mean": float(z.mean()),
            "max": float(z.max()),
        },
        "z_finite": bool(torch.isfinite(z).all()),
        "tie_summary": dict(_mapping(fit.get("tie_summary"), "full-fit tie summary")),
        "dependence": {
            "z_mean_sha256": tensor_sha256(_cpu_tensor(dependence.get("z_mean"), "z_mean")),
            "standard_deviation_sha256": tensor_sha256(_cpu_tensor(dependence.get("standard_deviation"), "standard_deviation")),
            "R_emp_sha256": tensor_sha256(_cpu_tensor(dependence.get("R_emp"), "R_emp")),
            "R_shrunk_sha256": tensor_sha256(_cpu_tensor(dependence.get("R_shrunk"), "R_shrunk")),
            "cholesky_sha256": tensor_sha256(_cpu_tensor(dependence.get("cholesky"), "cholesky")),
            "tau": float(dependence["tau"]),
            "beta": float(dependence["beta"]),
            "delta": float(dependence["delta"]),
            "alpha": float(dependence["alpha"]),
            "min_eigenvalue": float(eigenvalues.min()),
            "max_eigenvalue": float(eigenvalues.max()),
            "diagonal_error": float(dependence["diagonal_error"]),
        },
        "numerical_invariants": {
            "all_fit_outputs_finite": True,
            "standard_deviation_positive": True,
            "cholesky_info_zero": True,
            "eigenvalues_strictly_positive": True,
            "unit_diagonal_error_leq": DIAGONAL_ATOL,
        },
        "compatibility_evaluation_executed": False,
        "optical_or_semantic_evaluation_executed": False,
    }


def _build_state(preflight_result: Mapping[str, Any], fit: Mapping[str, Any], q: torch.Tensor, created_at_utc: str) -> dict[str, Any]:
    dependence = _mapping(fit.get("dependence"), "full-fit dependence")
    sorted_values = _cpu_tensor(fit.get("sorted_values"), "sorted_values")
    plotting_positions = _cpu_tensor(fit.get("plotting_positions"), "plotting_positions")
    quantile_positions = ((torch.arange(FULL_FIT_ROWS, dtype=torch.float64) + 0.5) / float(FULL_FIT_ROWS)).contiguous()
    sampling_tensors = {
        "sorted_values": sorted_values,
        "quantile_table_positions": quantile_positions,
        "cholesky": _cpu_tensor(dependence.get("cholesky"), "cholesky"),
    }
    audit_tensors = {
        "plotting_positions": plotting_positions,
        "z_mean": _cpu_tensor(dependence.get("z_mean"), "z_mean"),
        "standard_deviation": _cpu_tensor(dependence.get("standard_deviation"), "standard_deviation"),
        "R_emp": _cpu_tensor(dependence.get("R_emp"), "R_emp"),
        "R_shrunk": _cpu_tensor(dependence.get("R_shrunk"), "R_shrunk"),
        "eigenvalues": _cpu_tensor(dependence.get("eigenvalues"), "eigenvalues"),
        "diagonal": _cpu_tensor(dependence.get("diagonal"), "diagonal"),
    }
    audit_scalars = {
        "tau": float(dependence["tau"]),
        "beta": float(dependence["beta"]),
        "delta": float(dependence["delta"]),
        "alpha": float(dependence["alpha"]),
        "diagonal_error": float(dependence["diagonal_error"]),
    }
    source_identity = preflight_result["source_identity"]
    root = Path(preflight_result["repo_root"])
    return {
        "schema": STATE_SCHEMA,
        "version": 1,
        "semantic_authority_id": PRODUCER_STATE_SEMANTIC_ID,
        "family_id": PRODUCER_FAMILY_ID,
        "candidate_id": PRODUCER_CANDIDATE_ID,
        "candidate_version": "v1",
        "coordinate": "R2_AMPLITUDE",
        "dimension": DIMENSION,
        "dtype": "torch.float64",
        "created_at_utc": created_at_utc,
        "development_authority": {
            "semantic_authority_id": "N3_DEVELOPMENT_BANK_V2",
            "row_count": FULL_FIT_ROWS,
            "rows_per_class": ROWS_PER_CLASS,
            "class_counts": [ROWS_PER_CLASS] * CLASS_COUNT,
            "source_indices_sha256": source_identity["source_indices_sha256"],
            "labels_sha256": source_identity["labels_sha256"],
            "row_manifest_sha256": source_identity["row_manifest_sha256"],
            "source_row_identity_sha256": source_identity["source_row_identity_sha256"],
            "q_tensor_sha256": V2_Q_TENSOR_SHA256,
            "q_artifact_sha256": V2_Q_ARTIFACT_SHA256,
            "source_manifest_sha256": V2_SOURCE_MANIFEST_SHA256,
            "q_materialization_manifest_sha256": V2_Q_MANIFEST_SHA256,
            "overlap_counts": source_identity["overlap_counts"],
            "protected_registry_count": source_identity["protected_registry_count"],
            "sealed_reservation_status": "RESERVED_NOT_MATERIALIZED",
            "sealed_values_loaded": False,
            "sealed_confirmation_consumed": False,
        },
        "fit_scope": {
            "all_development_rows": True,
            "fit_row_count": FULL_FIT_ROWS,
            "fit_rows_per_class": ROWS_PER_CLASS,
            "folds_used_for_fit": False,
            "heldout_rows_used": False,
            "fold_evidence_role": "prior model selection only",
            "labels_as_inputs": False,
            "fit_execution_count": 1,
            "device": "cpu",
            "dtype": "torch.float64",
        },
        "sampling_tensors": sampling_tensors,
        "audit_tensors": audit_tensors,
        "audit_scalars": audit_scalars,
        "marginal_conventions": {
            "stable_sort": "torch.argsort(column, stable=True): value ascending; ties preserve original fit-row order",
            "tie_rule": "average one-based ranks L..U as (L+U)/2",
            "plotting_position": "(average_rank-0.5)/n_fit",
            "strict_interior": True,
            "epsilon_clamp": False,
            "quantile_table_positions": "(i+0.5)/n_fit for i=0,...,n_fit-1",
            "quantile_interpolation": "piecewise-linear between adjacent table positions",
            "lower_tail": "u <= first table position returns first observed sorted value",
            "upper_tail": "u >= last table position returns last observed sorted value",
            "extrapolation": False,
        },
        "dependence_conventions": {
            "gaussianization": "z=torch.special.ndtri(u)",
            "mean": "z_mean=z.mean(dim=0)",
            "standard_deviation": "sqrt(sum(z_centered**2)/(n_fit-1))",
            "standardization": "y=z_centered/standard_deviation",
            "R_emp": "y.T @ y/(n_fit-1)",
            "tau": "trace(R_emp)/dimension",
            "beta": "(sum(||y_i||^4)-2*sum(y_i.T@R_emp@y_i)+n_fit*||R_emp||_F^2)/(n_fit*n_fit), then clamp_min(0)",
            "delta": "||R_emp-tau I||_F^2",
            "alpha": "0 when delta==0; otherwise clamp(beta/delta,0,1)",
            "R_shrunk": "symmetrize(((1-alpha)*R_emp+alpha*tau I))",
            "unit_diagonal_renormalization": False,
            "final_matrix_used_for_sampling": "R_shrunk directly",
        },
        "sampling_interface": {
            "schema": SAMPLING_INTERFACE_SCHEMA,
            "rng_namespace": RNG_NAMESPACE,
            "state_sha256_required": True,
            "seed_required": True,
            "seed_semantics": "explicit nonnegative signed 63-bit seed passed directly to torch.Generator(device='cpu')",
            "algorithm": "noise=torch.randn((m,100),dtype=torch.float64); z=noise.mm(cholesky.T); u=ndtr(z); empirical inverse quantile",
            "fit_recomputation": False,
            "fold_or_heldout_input": False,
            "development_q_rows_required": False,
            "sealed_feedback": False,
            "rejection": False,
            "retry": False,
            "clipping_or_epsilon_repair": False,
            "samplewise_normalization": False,
        },
        "implementation_identity": {
            "semantic_source_path": GAUSSIAN_IMPLEMENTATION_PATH.as_posix(),
            "semantic_source_sha256": GAUSSIAN_IMPLEMENTATION_SHA256,
            "semantic_source_function": "_fit_fold with frozen FIT_COUNT_PER_FOLD bound to 5000 for this one call",
            "full_fit_consumer_path": _relative(root, Path(__file__).resolve()),
            "full_fit_consumer_sha256": file_sha256(Path(__file__).resolve()),
        },
        "provenance": {
            "freeze_contract_path": _relative(root, root / CONTRACT_PATH),
            "freeze_contract_sha256": EXPECTED_CONTRACT_SHA256,
            "v2_source_manifest_path": _relative(root, root / V2_SOURCE_MANIFEST_PATH),
            "v2_source_manifest_sha256": V2_SOURCE_MANIFEST_SHA256,
            "v2_q_manifest_path": _relative(root, root / V2_Q_MANIFEST_PATH),
            "v2_q_manifest_sha256": V2_Q_MANIFEST_SHA256,
            "v2_q_artifact_path": _relative(root, root / V2_Q_ARTIFACT_PATH),
            "v2_q_artifact_sha256": V2_Q_ARTIFACT_SHA256,
            "v2_q_tensor_sha256": V2_Q_TENSOR_SHA256,
            "v2_fold_manifest_path": _relative(root, root / V2_FOLD_MANIFEST_PATH),
            "v2_fold_manifest_sha256": V2_FOLD_MANIFEST_SHA256,
            "v2_fold_assignment_sha256": V2_ASSIGNMENT_SHA256,
            "v2_ceiling_report_path": _relative(root, root / V2_CEILING_REPORT_PATH),
            "v2_ceiling_report_sha256": V2_CEILING_REPORT_SHA256,
            "v3_replay_report_path": _relative(root, root / V3_REPLAY_REPORT_PATH),
            "v3_replay_report_sha256": V3_REPLAY_REPORT_SHA256,
            "v3_replay_lineage_path": _relative(root, root / V3_REPLAY_LINEAGE_PATH),
            "v3_replay_lineage_sha256": V3_REPLAY_LINEAGE_SHA256,
            "selected_producer_candidate_id": PRODUCER_CANDIDATE_ID,
            "sealed_confirmation_included": False,
            "compatibility_evaluation_included": False,
        },
        "state_identity": {
            "hash_algorithm": "SHA-256",
            "hash_scope": "complete state artifact bytes after atomic save",
            "artifact_sha256": None,
            "self_reference": False,
            "tensor_identity_hashes_external": True,
        },
        "governance": {
            "selected_producer": PRODUCER_FAMILY_ID,
            "selected_producer_candidate": PRODUCER_CANDIDATE_ID,
            "frozen_producer_state": PRODUCER_STATE_SEMANTIC_ID,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "sealed_confirmation_status": "RESERVED_NOT_MATERIALIZED",
            "sealed_confirmation_consumed": False,
        },
        "claim_boundary": {
            "development_fit_complete": True,
            "standalone_state_serialized": True,
            "not_population_generalization": True,
            "not_independent_confirmation": True,
            "not_optical_realizability": True,
            "not_semantic_controllability": True,
            "not_training_or_formal_promotion": True,
            "not_generation_readiness": True,
        },
    }


def _require_tensor(mapping: Mapping[str, Any], name: str, shape: tuple[int, ...]) -> torch.Tensor:
    value = mapping.get(name)
    _require(torch.is_tensor(value), f"state tensor is missing: {name}")
    _require(value.device.type == "cpu", f"state tensor is not on CPU: {name}")
    _require(value.dtype == torch.float64 and tuple(value.shape) == shape, f"state tensor shape/dtype changed: {name}")
    _require(value.requires_grad is False, f"state tensor requires gradients: {name}")
    tensor = value.detach().cpu().contiguous()
    _require(bool(torch.isfinite(tensor).all()), f"state tensor is non-finite: {name}")
    return tensor


def validate_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the structural and implementation contract of a state."""

    _require(isinstance(state, Mapping), "producer state is not a mapping")
    _require(state.get("schema") == STATE_SCHEMA and state.get("version") == 1, "producer state schema/version changed")
    for key, expected in {
        "semantic_authority_id": PRODUCER_STATE_SEMANTIC_ID,
        "family_id": PRODUCER_FAMILY_ID,
        "candidate_id": PRODUCER_CANDIDATE_ID,
        "candidate_version": "v1",
        "coordinate": "R2_AMPLITUDE",
        "dimension": DIMENSION,
        "dtype": "torch.float64",
    }.items():
        _require(state.get(key) == expected, f"producer state metadata changed: {key}")
    forbidden_top_level = {"q", "raw_q", "z", "labels", "source_indices", "heldout_q", "fold_ids"}
    _require(not forbidden_top_level.intersection(state.keys()), "producer state contains forbidden raw or selection tensors")
    sampling = _mapping(state.get("sampling_tensors"), "state sampling tensors")
    audit = _mapping(state.get("audit_tensors"), "state audit tensors")
    scalars = _mapping(state.get("audit_scalars"), "state audit scalars")
    sorted_values = _require_tensor(sampling, "sorted_values", (FULL_FIT_ROWS, DIMENSION))
    quantile_positions = _require_tensor(sampling, "quantile_table_positions", (FULL_FIT_ROWS,))
    cholesky = _require_tensor(sampling, "cholesky", (DIMENSION, DIMENSION))
    plotting_positions = _require_tensor(audit, "plotting_positions", (FULL_FIT_ROWS, DIMENSION))
    z_mean = _require_tensor(audit, "z_mean", (DIMENSION,))
    standard_deviation = _require_tensor(audit, "standard_deviation", (DIMENSION,))
    r_emp = _require_tensor(audit, "R_emp", (DIMENSION, DIMENSION))
    r_shrunk = _require_tensor(audit, "R_shrunk", (DIMENSION, DIMENSION))
    eigenvalues = _require_tensor(audit, "eigenvalues", (DIMENSION,))
    diagonal = _require_tensor(audit, "diagonal", (DIMENSION,))
    _require(bool(sorted_values.ge(0.0).all()), "state marginal table contains a negative value")
    _require(bool(torch.diff(sorted_values, dim=0).ge(0.0).all()), "state marginal table is not sorted")
    expected_quantile_positions = (torch.arange(FULL_FIT_ROWS, dtype=torch.float64) + 0.5) / float(FULL_FIT_ROWS)
    _require(torch.equal(quantile_positions, expected_quantile_positions), "state quantile table positions changed")
    _require(bool(plotting_positions.gt(0.0).all()) and bool(plotting_positions.lt(1.0).all()), "state plotting positions are not strictly interior")
    _require(bool(standard_deviation.gt(0.0).all()), "state standard deviation is not strictly positive")
    _require(bool(torch.allclose(r_emp, r_emp.transpose(0, 1), atol=1e-12, rtol=1e-12)), "state R_emp is not symmetric")
    _require(bool(torch.allclose(r_shrunk, r_shrunk.transpose(0, 1), atol=1e-12, rtol=1e-12)), "state R_shrunk is not symmetric")
    _require(bool(torch.triu(cholesky, diagonal=1).eq(0.0).all()), "state Cholesky factor is not lower triangular")
    _require(bool(eigenvalues.gt(0.0).all()), "state eigenvalues are not strictly positive")
    reconstructed = cholesky.mm(cholesky.transpose(0, 1))
    _require(bool(torch.allclose(reconstructed, r_shrunk, atol=1e-12, rtol=1e-12)), "state Cholesky reconstruction differs from R_shrunk")
    computed_eigenvalues = torch.linalg.eigvalsh(r_shrunk)
    _require(bool(torch.allclose(computed_eigenvalues, eigenvalues, atol=1e-10, rtol=1e-10)), "state eigenvalue audit changed")
    _require(bool(torch.allclose(torch.diagonal(r_shrunk), diagonal, atol=0.0, rtol=0.0)), "state diagonal audit changed")
    diagonal_error = (diagonal - 1.0).abs().max()
    _require(float(diagonal_error) <= DIAGONAL_ATOL, "state unit-diagonal invariant failed")
    for name in ("tau", "beta", "delta", "alpha", "diagonal_error"):
        value = scalars.get(name)
        _require(isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(float(value)), f"state audit scalar is invalid: {name}")
    _require(0.0 <= float(scalars["alpha"]) <= 1.0, "state shrinkage alpha is outside [0,1]")
    _require(float(scalars["beta"]) >= 0.0 and float(scalars["delta"]) >= 0.0, "state shrinkage scalar is negative")
    _require(abs(float(scalars["diagonal_error"]) - float(diagonal_error)) == 0.0, "state diagonal error audit changed")

    fit_scope = _mapping(state.get("fit_scope"), "state fit scope")
    for key, expected in {
        "all_development_rows": True,
        "fit_row_count": FULL_FIT_ROWS,
        "fit_rows_per_class": ROWS_PER_CLASS,
        "folds_used_for_fit": False,
        "heldout_rows_used": False,
        "fold_evidence_role": "prior model selection only",
        "labels_as_inputs": False,
        "fit_execution_count": 1,
        "device": "cpu",
        "dtype": "torch.float64",
    }.items():
        _require(fit_scope.get(key) == expected, f"state fit scope changed: {key}")
    interface = _mapping(state.get("sampling_interface"), "state sampling interface")
    _require(interface.get("schema") == SAMPLING_INTERFACE_SCHEMA and interface.get("rng_namespace") == RNG_NAMESPACE, "state sampling interface changed")
    _require(interface.get("state_sha256_required") is True and interface.get("fit_recomputation") is False, "state sampling binding changed")
    identity = _mapping(state.get("state_identity"), "state identity")
    _require(identity.get("self_reference") is False and identity.get("artifact_sha256") is None, "state self-reference policy changed")
    provenance = _mapping(state.get("provenance"), "state provenance")
    _require(provenance.get("sealed_confirmation_included") is False and provenance.get("compatibility_evaluation_included") is False, "state provenance boundary changed")
    governance = _mapping(state.get("governance"), "state governance")
    for key, expected in {
        "selected_producer": PRODUCER_FAMILY_ID,
        "selected_producer_candidate": PRODUCER_CANDIDATE_ID,
        "frozen_producer_state": PRODUCER_STATE_SEMANTIC_ID,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "sealed_confirmation_status": "RESERVED_NOT_MATERIALIZED",
        "sealed_confirmation_consumed": False,
    }.items():
        _require(governance.get(key) == expected, f"state governance changed: {key}")
    return {
        "tensor_hashes": {
            "sorted_values": tensor_sha256(sorted_values),
            "quantile_table_positions": tensor_sha256(quantile_positions),
            "cholesky": tensor_sha256(cholesky),
            "plotting_positions": tensor_sha256(plotting_positions),
            "z_mean": tensor_sha256(z_mean),
            "standard_deviation": tensor_sha256(standard_deviation),
            "R_emp": tensor_sha256(r_emp),
            "R_shrunk": tensor_sha256(r_shrunk),
            "eigenvalues": tensor_sha256(eigenvalues),
            "diagonal": tensor_sha256(diagonal),
        },
        "diagonal_error": float(diagonal_error),
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
    }


def _empirical_quantile_from_state(state: Mapping[str, Any], probabilities: torch.Tensor) -> torch.Tensor:
    sampling = _mapping(state["sampling_tensors"], "state sampling tensors")
    values = _require_tensor(sampling, "sorted_values", (FULL_FIT_ROWS, DIMENSION))
    positions = _require_tensor(sampling, "quantile_table_positions", (FULL_FIT_ROWS,))
    probs = probabilities.detach().cpu().to(dtype=torch.float64).contiguous()
    _require(probs.ndim == 2 and probs.shape[1] == DIMENSION, "standalone sampling probability shape is invalid")
    _require(bool(torch.isfinite(probs).all()), "standalone sampling probabilities are non-finite")
    result = torch.empty_like(probs)
    first = positions[0]
    last = positions[-1]
    for coordinate in range(DIMENSION):
        probability = probs[:, coordinate].contiguous()
        low = probability.le(first)
        high = probability.ge(last)
        upper = torch.searchsorted(positions, probability, right=True).clamp(min=1, max=FULL_FIT_ROWS - 1)
        lower = upper - 1
        lower_position = positions.index_select(0, lower)
        upper_position = positions.index_select(0, upper)
        fraction = (probability - lower_position) / (upper_position - lower_position)
        interpolated = values[:-1, coordinate].index_select(0, lower) + fraction * (
            values[1:, coordinate].index_select(0, lower) - values[:-1, coordinate].index_select(0, lower)
        )
        result[:, coordinate] = torch.where(low, values[0, coordinate], torch.where(high, values[-1, coordinate], interpolated))
    _require(bool(torch.isfinite(result).all()) and bool(result.ge(0.0).all()), "standalone sampling output is invalid")
    return result


def sample_from_state(
    state: Mapping[str, Any],
    *,
    producer_state_sha256: str,
    requested_sample_count: int,
    rng_namespace: str,
    seed: int,
) -> torch.Tensor:
    """Sample only from an already-validated standalone state."""

    validate_state(state)
    _require(isinstance(producer_state_sha256, str) and len(producer_state_sha256) == 64, "producer state SHA is required")
    _require(all(character in "0123456789abcdef" for character in producer_state_sha256), "producer state SHA format is invalid")
    _require(isinstance(requested_sample_count, int) and not isinstance(requested_sample_count, bool) and requested_sample_count >= 1, "requested sample count must be a positive integer")
    _require(rng_namespace == RNG_NAMESPACE, "sampling namespace does not match the frozen interface")
    _require(isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed <= MAX_SIGNED_63, "sampling seed is outside the signed 63-bit range")
    sampling = _mapping(state["sampling_tensors"], "state sampling tensors")
    cholesky = _require_tensor(sampling, "cholesky", (DIMENSION, DIMENSION))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn((requested_sample_count, DIMENSION), generator=generator, device="cpu", dtype=torch.float64)
    z = noise.mm(cholesky.transpose(0, 1))
    u = torch.special.ndtr(z)
    q = _empirical_quantile_from_state(state, u)
    _require(q.device.type == "cpu" and q.dtype == torch.float64, "standalone sample device/dtype changed")
    _require(tuple(q.shape) == (requested_sample_count, DIMENSION), "standalone sample shape changed")
    return q


def _mapping_equal(left: Any, right: Any, name: str = "payload") -> None:
    _require(isinstance(left, Mapping) and isinstance(right, Mapping), f"{name} is not a mapping")
    _require(set(left.keys()) == set(right.keys()), f"{name} keys changed")
    for key in left:
        l_value = left[key]
        r_value = right[key]
        if torch.is_tensor(l_value) or torch.is_tensor(r_value):
            _require(torch.is_tensor(l_value) and torch.is_tensor(r_value) and torch.equal(l_value, r_value), f"{name} tensor changed: {key}")
        elif isinstance(l_value, Mapping) or isinstance(r_value, Mapping):
            _mapping_equal(l_value, r_value, f"{name}.{key}")
        else:
            _require(l_value == r_value, f"{name} value changed: {key}")


def _make_lineage(
    preflight_result: Mapping[str, Any],
    state: Mapping[str, Any],
    state_path: Path,
    state_sha256: str,
    state_validation: Mapping[str, Any],
    fit_summary: Mapping[str, Any],
    smoke: Mapping[str, Any],
    created_at_utc: str,
) -> dict[str, Any]:
    root = Path(preflight_result["repo_root"])
    return {
        "schema": LINEAGE_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "VALID_COMPLETED",
        "classification": CLASSIFICATION,
        "created_at_utc": created_at_utc,
        "git": preflight_result["git"],
        "contract": {
            "path": _relative(root, root / CONTRACT_PATH),
            "sha256": EXPECTED_CONTRACT_SHA256,
            "schema": "n3_selected_producer_freeze_contract.v1",
        },
        "input_authorities": {
            "v2_source_manifest": {
                "path": _relative(root, root / V2_SOURCE_MANIFEST_PATH),
                "sha256": V2_SOURCE_MANIFEST_SHA256,
            },
            "v2_q_materialization_manifest": {
                "path": _relative(root, root / V2_Q_MANIFEST_PATH),
                "sha256": V2_Q_MANIFEST_SHA256,
            },
            "v2_q_artifact": {
                "path": _relative(root, root / V2_Q_ARTIFACT_PATH),
                "sha256": V2_Q_ARTIFACT_SHA256,
                "tensor_sha256": V2_Q_TENSOR_SHA256,
            },
            "v2_fold_manifest": {
                "path": _relative(root, root / V2_FOLD_MANIFEST_PATH),
                "sha256": V2_FOLD_MANIFEST_SHA256,
                "assignment_sha256": V2_ASSIGNMENT_SHA256,
                "fit_use": False,
            },
            "v2_empirical_ceiling": {
                "path": _relative(root, root / V2_CEILING_REPORT_PATH),
                "sha256": V2_CEILING_REPORT_SHA256,
            },
            "v3_selection_replay": {
                "report_path": _relative(root, root / V3_REPLAY_REPORT_PATH),
                "report_sha256": V3_REPLAY_REPORT_SHA256,
                "lineage_path": _relative(root, root / V3_REPLAY_LINEAGE_PATH),
                "lineage_sha256": V3_REPLAY_LINEAGE_SHA256,
            },
        },
        "development_input_verification": {
            "semantic_authority_id": "N3_DEVELOPMENT_BANK_V2",
            "coordinate": "R2_AMPLITUDE",
            "row_count": FULL_FIT_ROWS,
            "rows_per_class": ROWS_PER_CLASS,
            "unique_source_rows": True,
            "class_counts": [ROWS_PER_CLASS] * CLASS_COUNT,
            "source_indices_sha256": preflight_result["source_identity"]["source_indices_sha256"],
            "labels_sha256": preflight_result["source_identity"]["labels_sha256"],
            "source_row_identity_sha256": preflight_result["source_identity"]["source_row_identity_sha256"],
            "overlap_counts": preflight_result["source_identity"]["overlap_counts"],
            "protected_identity_overlap_recheck": True,
            "sealed_reservation_status": "RESERVED_NOT_MATERIALIZED",
            "sealed_values_loaded": False,
            "sealed_confirmation_consumed": False,
            "q_loaded_only_after_metadata_preflight": True,
        },
        "fit_summary": dict(fit_summary),
        "producer_state": {
            "semantic_authority_id": PRODUCER_STATE_SEMANTIC_ID,
            "schema": STATE_SCHEMA,
            "family_id": PRODUCER_FAMILY_ID,
            "candidate_id": PRODUCER_CANDIDATE_ID,
            "coordinate": "R2_AMPLITUDE",
            "path": _relative(root, state_path),
            "sha256": state_sha256,
            "size_bytes": state_path.stat().st_size,
            "tensor_sha256": state_validation["tensor_hashes"],
            "state_identity_external_binding": True,
            "raw_q_rows_persisted": False,
            "sealed_information_persisted": False,
        },
        "state_validation": {
            "structural_validation": True,
            "implementation_validation": True,
            "finite_and_nonnegative_marginal_table": True,
            "strict_quantile_positions": True,
            "strict_plotting_positions": True,
            "positive_definite_dependence": True,
            "cholesky_reconstruction": True,
            "unit_diagonal_error": state_validation["diagonal_error"],
            "unit_diagonal_tolerance": DIAGONAL_ATOL,
            "min_eigenvalue": state_validation["min_eigenvalue"],
            "max_eigenvalue": state_validation["max_eigenvalue"],
        },
        "serialization_validation": {
            "atomic_write_once": True,
            "safe_load_expected_sha256": True,
            "load_save_identity": True,
            "no_state_self_reference": True,
            "standalone_sampler_instantiated": True,
            "standalone_sampler_deterministic_smoke": bool(smoke["deterministic"]),
            "standalone_sampler_uses_state_only": True,
            "smoke": dict(smoke),
        },
        "governance": {
            "selected_producer": PRODUCER_FAMILY_ID,
            "selected_producer_candidate": PRODUCER_CANDIDATE_ID,
            "frozen_producer_state": PRODUCER_STATE_SEMANTIC_ID,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "sealed_confirmation_status": "RESERVED_NOT_MATERIALIZED",
            "sealed_confirmation_consumed": False,
        },
        "claim_boundary": {
            "full_development_fit_only": True,
            "state_freeze_only": True,
            "no_compatibility_evaluation": True,
            "no_confirmation": True,
            "no_optical_or_semantic_generation": True,
            "no_training": True,
            "no_formal_promotion": True,
        },
        "not_executed": [
            "sealed confirmation loading or materialization",
            "compatibility evaluation on the full fit",
            "new samples beyond the non-scientific implementation smoke",
            "optical, semantic, OLS, CMOS, or image generation",
            "CUDA, optimizer, backward, or training",
            "formal promotion or generation readiness",
        ],
        "next_action": "DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1",
        "lineage_identity": {
            "hash_algorithm": "SHA-256",
            "hash_scope": "complete lineage artifact bytes after atomic save",
            "self_reference": False,
        },
    }


def execute(repo_root: Path, output_dir: Path | None = None) -> dict[str, Any]:
    preflight_result = preflight(repo_root, output_dir)
    root = Path(preflight_result["repo_root"])
    target_dir = Path(preflight_result["output_dir"])
    created_at_utc = _utc_now()
    q = _load_development_q(preflight_result)
    fit = _fit_full_development_once(q)
    fit_summary = _fit_summary(fit, q)
    state = _build_state(preflight_result, fit, q, created_at_utc)
    state_validation_before = validate_state(state)

    state_path = target_dir / STATE_NAME
    lineage_path = target_dir / LINEAGE_NAME
    target_dir.mkdir(parents=True, exist_ok=False)
    atomic_torch_save_new(state_path, state, artifact_kind="selected producer state")
    state_sha256 = file_sha256(state_path)
    loaded_state = safe_torch_load(state_path, expected_sha256=state_sha256, map_location="cpu")
    loaded_validation = validate_state(loaded_state)
    _mapping_equal(state, loaded_state, "serialized producer state")
    _require(loaded_validation["tensor_hashes"] == state_validation_before["tensor_hashes"], "serialized state tensor identities changed")

    smoke_first = sample_from_state(
        loaded_state,
        producer_state_sha256=state_sha256,
        requested_sample_count=VALIDATION_SMOKE_COUNT,
        rng_namespace=RNG_NAMESPACE,
        seed=VALIDATION_SMOKE_SEED,
    )
    smoke_second = sample_from_state(
        loaded_state,
        producer_state_sha256=state_sha256,
        requested_sample_count=VALIDATION_SMOKE_COUNT,
        rng_namespace=RNG_NAMESPACE,
        seed=VALIDATION_SMOKE_SEED,
    )
    _require(torch.equal(smoke_first, smoke_second), "standalone smoke sampling is not deterministic")
    smoke = {
        "scientific_evaluation": False,
        "requested_sample_count": VALIDATION_SMOKE_COUNT,
        "rng_namespace": RNG_NAMESPACE,
        "seed": VALIDATION_SMOKE_SEED,
        "deterministic": True,
        "shape": list(smoke_first.shape),
        "dtype": str(smoke_first.dtype),
        "device": str(smoke_first.device),
        "finite": bool(torch.isfinite(smoke_first).all()),
        "nonnegative": bool(smoke_first.ge(0.0).all()),
        "q_tensor_sha256": tensor_sha256(smoke_first),
    }
    lineage = _make_lineage(
        preflight_result,
        loaded_state,
        state_path,
        state_sha256,
        loaded_validation,
        fit_summary,
        smoke,
        created_at_utc,
    )
    atomic_write_new_json(lineage_path, lineage, artifact_kind="selected producer full-fit lineage")
    lineage_sha256 = file_sha256(lineage_path)
    result = {
        "result": "VALID_COMPLETED",
        "classification": CLASSIFICATION,
        "semantic_authority": PRODUCER_STATE_SEMANTIC_ID,
        "producer_state_path": _relative(root, state_path),
        "producer_state_sha256": state_sha256,
        "lineage_path": _relative(root, lineage_path),
        "lineage_sha256": lineage_sha256,
        "fit_rows": FULL_FIT_ROWS,
        "rows_per_class": ROWS_PER_CLASS,
        "sealed_confirmation_consumed": False,
        "training_authorized": False,
        "formal_promotion": False,
        "generation_readiness": "not_ready",
        "next_action": "DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1",
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--preflight-only", action="store_true", help="run metadata-only checks without loading q or writing output")
    args = parser.parse_args()
    try:
        if args.preflight_only:
            result = preflight(args.repo_root)
            print(json.dumps({
                "result": "PREFLIGHT_PASSED",
                "task_id": TASK_ID,
                "contract_sha256": EXPECTED_CONTRACT_SHA256,
                "q_loaded": False,
                "sealed_values_loaded": result["sealed_values_loaded"],
                "output_dir_available": True,
            }, indent=2, sort_keys=True))
        else:
            print(json.dumps(execute(args.repo_root), indent=2, sort_keys=True))
        return 0
    except BlockedError as exc:
        print(json.dumps({"result": "BLOCKED", "classification": "BLOCKED", "reason": str(exc)}, indent=2, sort_keys=True))
        return 3
    except InvalidHarnessError as exc:
        print(json.dumps({"result": "INVALID", "classification": "INVALID_HARNESS", "reason": str(exc)}, indent=2, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
