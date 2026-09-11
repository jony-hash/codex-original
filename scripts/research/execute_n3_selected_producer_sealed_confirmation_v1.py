"""Execute the one-shot sealed confirmation for the frozen N3 producer.

The execution is deliberately split at the irreversible consumption marker:
all contract, producer, model-only calibration, observed-bank, evaluator, and
q-model checks happen before the marker; the reserved source is read only
after a write-once consumed=true marker has been published.
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
from scripts.research import execute_n3_selected_producer_full_development_fit_v1 as producer_consumer
from scripts.research import materialize_n3_development_bank_v2_q as q_materialization
from scripts.research import validate_n3_amplitude_producer_baseline as metric_consumer


TASK_ID = "EXECUTE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_V1"
GATE = TASK_ID
CONTRACT_PATH = Path(
    "outputs/20260910_v2/runs/"
    "n3_selected_producer_sealed_confirmation_contract_v1_attempt_01/"
    "n3_selected_producer_sealed_confirmation_contract.v1.json"
)
CONTRACT_SHA256 = "e0e29a62bce4eaaa48346ce65d1d7d6dcc2892df4d8e9ace73681579232c1200"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"

OUTPUT_DIR = Path(
    "outputs/20260910_v2/runs/"
    "n3_selected_producer_sealed_confirmation_v1_attempt_01"
)
CALIBRATION_NAME = "n3_selected_producer_sealed_confirmation_model_self_calibration.v1.pt"
MODEL_BANK_NAME = "n3_selected_producer_sealed_confirmation_model_banks.v1.pt"
SEALED_Q_NAME = "n3_selected_producer_sealed_confirmation_q.v1.pt"
SEALED_Q_MANIFEST_NAME = "n3_selected_producer_sealed_confirmation_q_materialization_manifest.v1.json"
CONSUMPTION_NAME = "n3_selected_producer_sealed_confirmation_consumption.v1.json"
LINEAGE_NAME = "n3_selected_producer_sealed_confirmation_lineage.v1.json"

PRODUCER_STATE_SHA256 = "752db16087e7bdf8343e4499202182370b376a1a2d5925a383f411fd4a15bb88"
PRODUCER_LINEAGE_SHA256 = "9244df01c95e44005207cb1dc188fc7de2961665e13055beebf55f0f5cdd2267"
PRODUCER_SEMANTIC_AUTHORITY = "N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1"
PRODUCER_SAMPLING_NAMESPACE = "N3_SELECTED_EMPIRICAL_GAUSSIAN_COPULA_PRODUCER_V1_CPU"
PRODUCER_FAMILY = "EMPIRICAL_GAUSSIAN_COPULA"
PRODUCER_CANDIDATE = "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1"

SEALED_MANIFEST_SHA256 = "6475474f157d31591a3b0252f27320323d84ea93803c61ee6a365420a8ebb08e"
SEALED_AUTHORITY = "N3_SEALED_CONFIRMATION_SOURCE_V1"
SEALED_CONFIRMATION_ID = (
    "N3_SEALED_CONFIRMATION_SOURCE_V1:"
    "5bf32ec1bfb8ffb9e62f5f2fa578cc5dffb1d288eba6ce850f024781e5e13f10"
)
SEALED_ROWS = 500
SEALED_ROWS_PER_CLASS = 50
CLASS_COUNT = 10
DIMENSION = 100

MODEL_BANK_COUNT = 4
MODEL_BANK_SIZE = 200
CALIBRATION_TRIALS = 999
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
SEED_DERIVATION_ROOT = "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_V1_CPU"
MAX_SIGNED_63 = (1 << 63) - 1

CALIBRATION_SCHEMA = "n3_selected_producer_sealed_confirmation_model_self_calibration.v1"
MODEL_BANK_SCHEMA = "n3_selected_producer_sealed_confirmation_model_banks.v1"
Q_ARTIFACT_SCHEMA = "n3_selected_producer_sealed_confirmation_q.v1"
Q_MANIFEST_SCHEMA = "n3_selected_producer_sealed_confirmation_q_materialization_manifest.v1"
CONSUMPTION_SCHEMA = "n3_selected_producer_sealed_confirmation_consumption.v1"
LINEAGE_SCHEMA = "n3_selected_producer_sealed_confirmation_lineage.v1"

CLASSIFICATION_PASSED = "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_PASSED"
CLASSIFICATION_FAILED = "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_FAILED"
CLASSIFICATION_INVALID_CONSUMED = "SEALED_CONFIRMATION_INVALID_CONSUMED"

V2_SOURCE_MANIFEST_SHA256 = "4bc251af91b3a6cd0235fe113117443d871a564bd05634fa9e61b5ab0b36f812"
V2_Q_MANIFEST_SHA256 = "422d8c65b78bc459197ed46504399cb9c87ba8f7184eb2405e0faa85353995d8"
V2_Q_ARTIFACT_SHA256 = "fe48c8fdb9bb2bf732076fabdf475288baa6f7d628ca10039a83bbda54fcfcbf"
V3_REPORT_SHA256 = "7135b07fd532ee5dd2dba7900ad6028c9a6d558924691e4d65eac56330367789"
V3_EVALUATOR_SHA256 = "4fbc831e383897802134ea2ce1aaef6446629130a7c799630254a86b8dcff3cc"
V3_METRIC_SHA256 = "1ad916bbb8d109ca989f64b9f5590391cdffea9e0567e2ca8d241b032c6cde36"
V3_AGGREGATION_SHA256 = "5f1fa3fa6be35e96f345c3b7f7c6cce5ca0b45fe4619b1ead4455e86dcae538a"
V3_EVALUATOR_IMPL_SHA256 = "18b8f9defd83c2591dc055a97621ef6a1717783f7656da75994881e02bffda14"
Q_MATERIALIZER_SHA256 = "c070386325b332ce4e3f84a833274454f20681b4e9b8f0bb76c0bfbb8a1c07cb"
Q_REPLAY_SHA256 = "2dcdcf9f0c8437ecf2e89c50f49bb672dde9bb8af9d99fd0c6606095e2e23b6e"
RAW_IMAGES_PATH = Path("data/MNIST/raw/train-images-idx3-ubyte")


class InvalidHarnessError(RuntimeError):
    """A pre-consumption authority or contract condition is invalid."""


class ConsumedInvalidError(RuntimeError):
    """The source was exposed but a valid scientific result could not finish."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvalidHarnessError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


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


def _read_json_unhashed(repo_root: Path, path: Path, name: str) -> tuple[dict[str, Any], str]:
    target = repo_root / path
    _require(target.is_file(), f"{name} is missing: {target}")
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidHarnessError(f"{name} cannot be read as JSON: {target}") from exc
    return dict(_mapping(value, name)), file_sha256(target)


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


def _seed(stream: str, trial: int | None, bank_index: int | None) -> int:
    trial_token = "none" if trial is None else str(trial)
    bank_token = "none" if bank_index is None else str(bank_index)
    material = f"{SEED_DERIVATION_ROOT}|{stream}|trial={trial_token}|bank_index={bank_token}"
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") & MAX_SIGNED_63


def _path_from_ref(repo_root: Path, value: Mapping[str, Any], name: str) -> Path:
    path_value = value.get("path")
    _require(isinstance(path_value, str) and path_value, f"{name} path is missing")
    return repo_root / Path(path_value)


def _verify_ref(
    repo_root: Path,
    value: Mapping[str, Any],
    name: str,
    *,
    expected_sha256: str | None = None,
) -> Path:
    path = _path_from_ref(repo_root, value, name)
    expected = expected_sha256 if expected_sha256 is not None else value.get("sha256")
    _require(isinstance(expected, str) and len(expected) == 64, f"{name} SHA is missing")
    _require(path.is_file(), f"{name} is missing: {path}")
    _require(file_sha256(path) == expected, f"{name} hash changed: {path}")
    return path


def _validate_current_governance(
    repo_root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    state, state_sha256 = _read_json_unhashed(
        repo_root,
        Path("docs/status/research_governance_state.json"),
        "current research governance state",
    )
    _require(
        state.get("active_objective") == "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_READY",
        "current governance is not at the sealed-confirmation contract boundary",
    )
    _require(
        state.get("hard_blocker") == "N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_EXECUTION_REQUIRED",
        "current governance does not authorize this execution boundary",
    )
    _require(state.get("authorization_tier") == "R0", "current governance authorization tier changed before execution")
    aliases = _mapping(state.get("evidence_aliases"), "current governance evidence aliases")
    contract_alias = _mapping(
        aliases.get("N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1"),
        "sealed-confirmation contract governance alias",
    )
    _require(contract_alias.get("sha256") == CONTRACT_SHA256, "current governance contract alias changed")
    _require(
        contract_alias.get("classification") == "DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1_READY"
        and contract_alias.get("sealed_confirmation_consumed") is False,
        "current governance contract alias is not ready and unconsumed",
    )
    full_fit = _mapping(
        aliases.get("N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_V1"),
        "full-fit governance alias",
    )
    _require(full_fit.get("producer_state_sha256") == PRODUCER_STATE_SHA256, "current full-fit state binding changed")
    _require(full_fit.get("sha256") == PRODUCER_LINEAGE_SHA256, "current full-fit lineage binding changed")
    for key, expected in (
        ("training_authorized", False),
        ("formal_promotion", False),
        ("sealed_confirmation_consumed", False),
        ("generation_readiness", "not_ready"),
    ):
        _require(full_fit.get(key) == expected, f"current full-fit governance changed: {key}")

    for path, name in (
        (Path("docs/CURRENT_STATUS.md"), "CURRENT_STATUS"),
        (Path("docs/status/current_handoff.md"), "current handoff"),
    ):
        text = (repo_root / path).read_text(encoding="utf-8")
        _require(
            "DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1_READY" in text,
            f"{name} does not record the ready contract",
        )
        _require(TASK_ID in text, f"{name} does not name the authorized execution")
        _require(
            "training_authorized=false" in text and "generation_readiness=not_ready" in text,
            f"{name} governance boundary changed",
        )
    return {
        "path": _relative(repo_root, repo_root / Path("docs/status/research_governance_state.json")),
        "sha256": state_sha256,
        "active_objective": state["active_objective"],
        "hard_blocker": state["hard_blocker"],
        "authorization_tier": state["authorization_tier"],
        "sealed_confirmation_consumed": False,
    }


def _validate_contract(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = _read_json_exact(repo_root, CONTRACT_PATH, CONTRACT_SHA256, "sealed-confirmation contract")
    _require(contract.get("schema") == "n3_selected_producer_sealed_confirmation_contract.v1", "contract schema changed")
    _require(
        contract.get("status") == "valid_completed"
        and contract.get("result") == "READY"
        and contract.get("classification") == "DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1_READY",
        "sealed-confirmation contract is not READY",
    )
    _require(contract.get("definition_only") is True, "sealed-confirmation contract is not definition-only")
    producer = _mapping(contract.get("producer"), "contract producer")
    for key, expected in (
        ("family_id", PRODUCER_FAMILY),
        ("candidate_id", PRODUCER_CANDIDATE),
        ("semantic_authority_id", PRODUCER_SEMANTIC_AUTHORITY),
        ("state_sha256", PRODUCER_STATE_SHA256),
        ("lineage_sha256", PRODUCER_LINEAGE_SHA256),
    ):
        _require(producer.get(key) == expected, f"contract producer binding changed: {key}")
    _require(
        _mapping(producer.get("standalone_sampler"), "contract standalone sampler").get("sampling_namespace")
        == PRODUCER_SAMPLING_NAMESPACE,
        "contract producer sampler namespace changed",
    )

    sealed = _mapping(contract.get("sealed_source"), "contract sealed source")
    for key, expected in (
        ("reservation_authority", SEALED_AUTHORITY),
        ("manifest_sha256", SEALED_MANIFEST_SHA256),
        ("semantic_confirmation_id", SEALED_CONFIRMATION_ID),
        ("expected_status_on_entry", "RESERVED_NOT_MATERIALIZED"),
        ("expected_consumed_on_entry", False),
        ("row_count", SEALED_ROWS),
        ("rows_per_class", SEALED_ROWS_PER_CLASS),
    ):
        _require(sealed.get(key) == expected, f"contract sealed-source binding changed: {key}")
    _require(sealed.get("substitution_or_resampling") is False, "contract permits sealed substitution")

    statistic = _mapping(contract.get("confirmation_statistic"), "contract confirmation statistic")
    _require(statistic.get("primary_metric") == PRIMARY_METRIC, "contract primary metric changed")
    _require(statistic.get("data_shape") == [SEALED_ROWS, DIMENSION], "contract sealed data shape changed")
    _require(
        statistic.get("model_bank_shape") == [MODEL_BANK_SIZE, DIMENSION],
        "contract model-bank shape changed",
    )
    _require(
        statistic.get("model_bank_aggregation", {}).get("aggregation")
        == "unweighted arithmetic mean of the four scalar energy_distance values",
        "contract model-bank aggregation changed",
    )
    _require(
        _mapping(statistic.get("secondary_metrics"), "contract secondary metrics").get("names")
        == list(DESCRIPTIVE_ONLY_METRICS),
        "contract descriptive metric set changed",
    )

    calibration = _mapping(contract.get("model_self_calibration"), "contract model-self calibration")
    for key, expected in (
        ("trials", CALIBRATION_TRIALS),
        ("pseudo_data_rows", SEALED_ROWS),
        ("model_bank_count", MODEL_BANK_COUNT),
        ("model_bank_size", MODEL_BANK_SIZE),
        ("artifact_schema", CALIBRATION_SCHEMA),
    ):
        _require(calibration.get(key) == expected, f"contract calibration binding changed: {key}")
    calibration_rng = _mapping(calibration.get("rng"), "contract calibration RNG")
    _require(calibration_rng.get("no_early_stop") is True, "contract allows calibration early stop")
    _require(calibration_rng.get("no_seed_retry") is True, "contract allows calibration seed retry")
    _require(calibration_rng.get("best_of_n") is False, "contract permits calibration best-of-N")

    banks = _mapping(contract.get("observed_model_banks"), "contract observed model banks")
    for key, expected in (
        ("bank_count", MODEL_BANK_COUNT),
        ("bank_size", MODEL_BANK_SIZE),
        ("artifact_schema", MODEL_BANK_SCHEMA),
        ("producer_state_sha256", PRODUCER_STATE_SHA256),
    ):
        _require(banks.get(key) == expected, f"contract observed-bank binding changed: {key}")
    observed_seeds = [_seed("OBSERVED_MODEL_BANK", None, bank) for bank in range(MODEL_BANK_COUNT)]
    _require(
        _mapping(banks.get("rng"), "contract observed-bank RNG").get("seeds") == observed_seeds,
        "contract observed-bank seeds changed",
    )

    decision = _mapping(contract.get("monte_carlo_decision"), "contract Monte-Carlo decision")
    for key, expected in (
        ("B", CALIBRATION_TRIALS),
        ("alpha", ALPHA),
        ("p_value", "p_MC=(1+count(T_b >= T_sealed))/(B+1)"),
        ("inequality", "greater-than-or-equal, including ties"),
        ("pass_if", "p_MC >= 0.05"),
        ("fail_if", "p_MC < 0.05"),
        ("one_decision_only", True),
        ("no_retry", True),
        ("no_early_stop", True),
        ("no_best_of_n", True),
    ):
        _require(decision.get(key) == expected, f"contract decision binding changed: {key}")

    consumer = _mapping(contract.get("future_confirmation_consumer"), "contract future consumer")
    _require(
        consumer.get("path") == _relative(repo_root, repo_root / Path(__file__))
        and consumer.get("schema") == "n3_selected_producer_sealed_confirmation_execution_consumer.v1",
        "contract consumer identity changed",
    )
    _require(consumer.get("implementation_frozen_in_this_task") is False, "contract was rewritten by execution")
    _require(
        _mapping(contract.get("consumption_semantics"), "contract consumption semantics").get("single_use") is True,
        "contract consumption is not single-use",
    )

    authorities = _mapping(contract.get("authorities"), "contract authorities")
    authority_snapshot: dict[str, Any] = {}
    definition_ref = _mapping(authorities.get("definition_implementation"), "contract definition implementation")
    _verify_ref(repo_root, definition_ref, "contract definition implementation")
    authority_snapshot["definition_implementation"] = dict(definition_ref)

    immutable_refs = (
        ("selected_producer_freeze_contract", authorities.get("selected_producer_freeze_contract")),
        ("sealed_reservation", authorities.get("sealed_reservation")),
        ("selected_producer_full_fit.lineage", _mapping(authorities.get("selected_producer_full_fit"), "full-fit authority").get("lineage")),
        ("selected_producer_full_fit.state", _mapping(authorities.get("selected_producer_full_fit"), "full-fit authority").get("state")),
        ("v2_development.source_manifest", _mapping(authorities.get("v2_development"), "V2 authority").get("source_manifest")),
        ("v2_development.q_manifest", _mapping(authorities.get("v2_development"), "V2 authority").get("q_manifest")),
        ("v3.calibration_report", _mapping(authorities.get("v3"), "V3 authority").get("calibration_report")),
        ("v3.evaluator", _mapping(authorities.get("v3"), "V3 authority").get("evaluator")),
    )
    for name, value in immutable_refs:
        ref = _mapping(value, f"contract {name}")
        _verify_ref(repo_root, ref, f"contract {name}")
        authority_snapshot[name] = dict(ref)
    return contract, {
        "contract_sha256": CONTRACT_SHA256,
        "authorities": authority_snapshot,
        "observed_model_bank_seeds": observed_seeds,
    }


def _validate_development_and_v3(
    repo_root: Path,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    authorities = _mapping(contract.get("authorities"), "contract authorities")
    v2 = _mapping(authorities.get("v2_development"), "V2 authority")
    source_ref = _mapping(v2.get("source_manifest"), "V2 source authority")
    source_path = _verify_ref(repo_root, source_ref, "V2 source manifest", expected_sha256=V2_SOURCE_MANIFEST_SHA256)
    source = dict(_mapping(json.loads(source_path.read_text(encoding="utf-8")), "V2 source manifest"))
    _require(source.get("schema") == "n3_development_bank_v2_source_manifest.v1", "V2 source schema changed")
    _require(source.get("status") == "FROZEN_SOURCE_ROWS", "V2 source manifest is incomplete")
    identity = _mapping(source.get("identity"), "V2 source identity")
    _require(
        identity.get("class_counts") == [500] * CLASS_COUNT
        and identity.get("unique_source_rows") is True,
        "V2 source dimensions changed",
    )
    rows = source.get("rows")
    _require(isinstance(rows, list) and len(rows) == 5000, "V2 source rows are incomplete")
    development_indices = [int(_mapping(row, "V2 source row").get("source_index")) for row in rows]
    _require(len(set(development_indices)) == 5000, "V2 source indices are not unique")

    q_ref = _mapping(v2.get("q_manifest"), "V2 q authority")
    q_manifest_path = _verify_ref(repo_root, q_ref, "V2 q materialization manifest", expected_sha256=V2_Q_MANIFEST_SHA256)
    q_manifest = dict(_mapping(json.loads(q_manifest_path.read_text(encoding="utf-8")), "V2 q materialization manifest"))
    _require(q_manifest.get("schema") == "n3_development_bank_v2_q_materialization_manifest.v1", "V2 q manifest schema changed")
    _require(q_manifest.get("status") == "VALID_COMPLETED", "V2 q materialization is incomplete")
    q_artifact = _mapping(q_manifest.get("q_artifact"), "V2 q artifact reference")
    _require(q_artifact.get("sha256") == V2_Q_ARTIFACT_SHA256, "V2 q artifact identity changed")
    _require(q_artifact.get("q_tensor_sha256") == "d660fd8dc449725edab081d00c8f553f151029cddef9210360a7a6fad4281209", "V2 q tensor identity changed")
    _require(q_artifact.get("shape") == [5000, DIMENSION] and q_artifact.get("dtype") == "torch.float64", "V2 q shape/dtype changed")
    _require(q_manifest.get("sealed_confirmation_consumed") is False, "V2 q manifest consumed sealed data")

    v3 = _mapping(authorities.get("v3"), "V3 authority")
    report_ref = _mapping(v3.get("calibration_report"), "V3 calibration report authority")
    report_path = _verify_ref(repo_root, report_ref, "V3 calibration report", expected_sha256=V3_REPORT_SHA256)
    report = dict(_mapping(json.loads(report_path.read_text(encoding="utf-8")), "V3 calibration report"))
    _require(
        report.get("schema") == "n3_q_compatibility_guardrail_calibration_report.v1"
        and report.get("classification") == "N3_Q_COMPATIBILITY_GUARDRAIL_CALIBRATION_READY",
        "V3 calibration report identity changed",
    )
    evaluator_ref = _mapping(v3.get("evaluator"), "V3 evaluator authority")
    evaluator_path = _verify_ref(repo_root, evaluator_ref, "V3 evaluator", expected_sha256=V3_EVALUATOR_SHA256)
    evaluator = dict(_mapping(json.loads(evaluator_path.read_text(encoding="utf-8")), "V3 evaluator"))
    _require(evaluator.get("schema") == "n3_q_compatibility_evaluator_v3.v1", "V3 evaluator schema changed")
    primary = _mapping(evaluator.get("primary"), "V3 primary metric")
    _require(primary.get("metric") == PRIMARY_METRIC and primary.get("role") == "HARD_PRIMARY_UNCHANGED", "V3 evaluator primary metric changed")
    _require(evaluator.get("descriptive_only_guardrails") == list(DESCRIPTIVE_ONLY_METRICS), "V3 descriptive metric set changed")
    _require(evaluator.get("result") == "READY" and evaluator.get("status") == "valid_completed", "V3 evaluator is not ready")

    q_semantics = _mapping(contract.get("q_materialization"), "contract q materialization")
    q_impl = _mapping(q_semantics.get("implementation"), "contract q materializer")
    q_path = _verify_ref(repo_root, q_impl, "source-to-q materializer", expected_sha256=Q_MATERIALIZER_SHA256)
    encoder_impl = _mapping(q_semantics.get("encoder_prefix_implementation"), "contract encoder prefix")
    encoder_path = _verify_ref(repo_root, encoder_impl, "encoder-prefix implementation", expected_sha256=Q_REPLAY_SHA256)
    v3_impls = _mapping(v3.get("implementations"), "contract V3 implementations")
    implementation_paths: dict[str, str] = {}
    for name, expected_sha in (
        ("energy_distance", V3_METRIC_SHA256),
        ("model_bank_aggregation", V3_AGGREGATION_SHA256),
        ("v3_evaluator", V3_EVALUATOR_IMPL_SHA256),
    ):
        ref = _mapping(v3_impls.get(name), f"contract V3 {name}")
        path = _verify_ref(repo_root, ref, f"V3 {name} implementation", expected_sha256=expected_sha)
        implementation_paths[name] = _relative(repo_root, path)
    return {
        "source_manifest": {
            "path": _relative(repo_root, source_path),
            "sha256": V2_SOURCE_MANIFEST_SHA256,
            "rows": len(rows),
            "rows_per_class": 500,
            "source_indices": development_indices,
            "semantic_authority_id": source.get("semantic_authority_id"),
            "bank_semantic_sha256": source.get("bank_semantic_sha256"),
        },
        "q_manifest": {
            "path": _relative(repo_root, q_manifest_path),
            "sha256": V2_Q_MANIFEST_SHA256,
            "q_artifact_sha256": V2_Q_ARTIFACT_SHA256,
            "q_tensor_sha256": q_artifact.get("q_tensor_sha256"),
            "values_loaded": False,
        },
        "v3": {
            "calibration_report_path": _relative(repo_root, report_path),
            "calibration_report_sha256": V3_REPORT_SHA256,
            "evaluator_path": _relative(repo_root, evaluator_path),
            "evaluator_sha256": V3_EVALUATOR_SHA256,
            "implementation_paths": implementation_paths,
        },
        "q_implementations": {
            "materializer_path": _relative(repo_root, q_path),
            "materializer_sha256": Q_MATERIALIZER_SHA256,
            "encoder_prefix_path": _relative(repo_root, encoder_path),
            "encoder_prefix_sha256": Q_REPLAY_SHA256,
        },
    }


def _validate_sealed_reservation(
    repo_root: Path,
    contract: Mapping[str, Any],
    development_indices: list[int],
) -> dict[str, Any]:
    manifest_path = Path(
        _mapping(contract.get("sealed_source"), "contract sealed source").get("manifest_path")
    )
    manifest = _read_json_exact(repo_root, manifest_path, SEALED_MANIFEST_SHA256, "sealed reservation manifest")
    _require(manifest.get("schema") == "n3_sealed_confirmation_source_manifest.v1", "sealed manifest schema changed")
    _require(manifest.get("status") == "RESERVED_NOT_MATERIALIZED", "sealed reservation is no longer reserved")
    _require(manifest.get("semantic_authority_id") == SEALED_AUTHORITY, "sealed authority changed")
    _require(manifest.get("semantic_confirmation_id") == SEALED_CONFIRMATION_ID, "sealed confirmation identity changed")

    selection = _mapping(manifest.get("selection"), "sealed selection")
    indices_value = selection.get("source_indices")
    labels_value = selection.get("labels")
    _require(isinstance(indices_value, list) and len(indices_value) == SEALED_ROWS, "sealed source index count changed")
    _require(isinstance(labels_value, list) and len(labels_value) == SEALED_ROWS, "sealed label identity count changed")
    indices = torch.tensor([int(value) for value in indices_value], dtype=torch.int64).contiguous()
    labels = torch.tensor([int(value) for value in labels_value], dtype=torch.int64).contiguous()
    _require(torch.unique(indices).numel() == SEALED_ROWS, "sealed source indices are not unique")
    _require(torch.equal(torch.bincount(labels, minlength=CLASS_COUNT), torch.full((CLASS_COUNT,), SEALED_ROWS_PER_CLASS, dtype=torch.int64)), "sealed class balance changed")
    _require(tensor_sha256(indices) == "60baa34331dad647ff6ed4944a9c5ebdce6ad8ad2bb68219e64701696b0ceeb1", "sealed source-index identity changed")
    _require(tensor_sha256(labels) == "c8bacf260ae894e7da623586b9e04c67d95f628c16f52766cee0994ca2ceac19", "sealed label identity changed")
    _require(selection.get("count") == SEALED_ROWS and selection.get("per_class") == SEALED_ROWS_PER_CLASS, "sealed selection dimensions changed")
    _require(selection.get("ols_values_observed") is False and selection.get("q_values_observed") is False, "sealed values were observed during selection")

    ordered_identity = [
        {"label": int(label), "position": position, "source_index": int(index)}
        for position, (index, label) in enumerate(zip(indices.tolist(), labels.tolist(), strict=True))
    ]
    membership = _mapping(_mapping(contract.get("sealed_source"), "contract sealed source").get("membership"), "contract sealed membership")
    _require(membership.get("source_indices") == indices.tolist(), "sealed membership differs from contract")
    _require(membership.get("labels_for_identity_only") == labels.tolist(), "sealed label identity differs from contract")
    _require(membership.get("ordered_row_identity") == ordered_identity, "sealed row ordering differs from contract")
    _require(membership.get("ordered_row_identity_sha256") == canonical_sha256(ordered_identity), "sealed ordered identity hash is stale")
    _require(membership.get("ordered_row_identity_sha256") == "a1e93698e3bffdafecdeb7cf3f1b1ca7d4bbb52e25935ebd7d82d885796ba038", "sealed ordered identity authority changed")

    development_set = set(development_indices)
    sealed_set = set(indices.tolist())
    _require(not development_set.intersection(sealed_set), "sealed source overlaps V2 development source")
    firewall = _mapping(manifest.get("firewall"), "sealed firewall")
    _require(firewall.get("fail_closed") is True and firewall.get("source_index_overlap_count") == 0, "sealed firewall is not passing")
    protected_union = _mapping(manifest.get("protected_source_union"), "sealed protected union")
    _require(protected_union.get("count") == 11000 and protected_union.get("overlap_with_reserved_source_count") == 0, "sealed protected union changed")
    protected_entries = manifest.get("protected_identities_checked")
    _require(isinstance(protected_entries, list), "sealed protected identity checks are missing")
    by_name = {
        str(_mapping(item, "protected identity").get("canonical_name")): _mapping(item, "protected identity")
        for item in protected_entries
    }
    for name, expected_count in (("N3_DEVELOPMENT_BANK", 1000), ("S1_CONFIRMATION", 5000), ("S2_VALIDATION", 5000)):
        entry = _mapping(by_name.get(name), f"sealed protected identity {name}")
        _require(entry.get("count") == expected_count and entry.get("overlap_count_with_reserved_source") == 0, f"sealed overlap authority changed: {name}")
    materialization = _mapping(manifest.get("materialization"), "sealed materialization state")
    _require(materialization.get("current_status") == "RESERVED_NOT_MATERIALIZED", "sealed materialization status changed")
    _require(materialization.get("producer_candidate_consumed") is False, "sealed producer candidate was consumed")
    _require(materialization.get("complex_ols_values_loaded") is False, "sealed complex OLS values were loaded")
    _require(materialization.get("ols_states_materialized") is False, "sealed OLS states were materialized")
    return {
        "path": _relative(repo_root, repo_root / manifest_path),
        "sha256": SEALED_MANIFEST_SHA256,
        "semantic_authority_id": SEALED_AUTHORITY,
        "semantic_confirmation_id": SEALED_CONFIRMATION_ID,
        "indices": indices,
        "labels": labels,
        "ordered_identity": ordered_identity,
        "source_index_sha256": tensor_sha256(indices),
        "label_sha256": tensor_sha256(labels),
        "ordered_row_identity_sha256": canonical_sha256(ordered_identity),
        "row_count": SEALED_ROWS,
        "rows_per_class": SEALED_ROWS_PER_CLASS,
        "status_before_consumption": "RESERVED_NOT_MATERIALIZED",
        "consumed_before": False,
        "protected_content_loaded_before": False,
    }


def _validate_producer(repo_root: Path, contract: Mapping[str, Any]) -> tuple[Mapping[str, Any], dict[str, Any]]:
    producer_authority = _mapping(_mapping(contract.get("authorities"), "contract authorities").get("selected_producer_full_fit"), "full-fit authority")
    state_ref = _mapping(producer_authority.get("state"), "full-fit state authority")
    lineage_ref = _mapping(producer_authority.get("lineage"), "full-fit lineage authority")
    state_path = _verify_ref(repo_root, state_ref, "producer state", expected_sha256=PRODUCER_STATE_SHA256)
    lineage_path = _verify_ref(repo_root, lineage_ref, "producer lineage", expected_sha256=PRODUCER_LINEAGE_SHA256)
    lineage = dict(_mapping(json.loads(lineage_path.read_text(encoding="utf-8")), "producer lineage"))
    _require(lineage.get("schema") == "n3_selected_producer_full_development_fit_lineage.v1", "producer lineage schema changed")
    _require(lineage.get("classification") == "N3_SELECTED_PRODUCER_FULL_DEVELOPMENT_FIT_COMPLETED", "producer lineage classification changed")
    _require(
        lineage.get("next_action") == "DEFINE_N3_SELECTED_PRODUCER_SEALED_CONFIRMATION_CONTRACT_V1",
        "producer lineage historical next action changed",
    )
    _require(
        _mapping(lineage.get("governance"), "producer lineage governance").get("sealed_confirmation_consumed") is False,
        "producer lineage consumed sealed data",
    )
    state = safe_torch_load(state_path, expected_sha256=PRODUCER_STATE_SHA256, map_location="cpu")
    validation = producer_consumer.validate_state(state)
    _require(state.get("semantic_authority_id") == PRODUCER_SEMANTIC_AUTHORITY, "producer state semantic authority changed")
    _require(_mapping(state.get("governance"), "producer state governance").get("sealed_confirmation_consumed") is False, "producer state consumed sealed data")
    return state, {
        "state_path": _relative(repo_root, state_path),
        "state_sha256": PRODUCER_STATE_SHA256,
        "lineage_path": _relative(repo_root, lineage_path),
        "lineage_sha256": PRODUCER_LINEAGE_SHA256,
        "state_validation": validation,
        "lineage_schema": lineage.get("schema"),
    }


def _make_fast_sampler(state: Mapping[str, Any]):
    """Bind the exact frozen sampler body after one complete state validation."""

    sampling = _mapping(state.get("sampling_tensors"), "producer sampling tensors")
    sorted_values = sampling["sorted_values"].detach().cpu().contiguous()
    positions = sampling["quantile_table_positions"].detach().cpu().contiguous()
    cholesky = sampling["cholesky"].detach().cpu().contiguous()

    def sample(requested_sample_count: int, seed: int) -> torch.Tensor:
        _require(requested_sample_count >= 1, "sample count must be positive")
        _require(0 <= seed <= MAX_SIGNED_63, "sample seed is outside signed 63-bit range")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(
            (requested_sample_count, DIMENSION),
            generator=generator,
            device="cpu",
            dtype=torch.float64,
        )
        z = noise.mm(cholesky.transpose(0, 1))
        probabilities = torch.special.ndtr(z)
        result = torch.empty_like(probabilities)
        first = positions[0]
        last = positions[-1]
        for coordinate in range(DIMENSION):
            probability = probabilities[:, coordinate].contiguous()
            low = probability.le(first)
            high = probability.ge(last)
            upper = torch.searchsorted(positions, probability, right=True).clamp(min=1, max=positions.numel() - 1)
            lower = upper - 1
            lower_position = positions.index_select(0, lower)
            upper_position = positions.index_select(0, upper)
            fraction = (probability - lower_position) / (upper_position - lower_position)
            interpolated = sorted_values[:-1, coordinate].index_select(0, lower) + fraction * (
                sorted_values[1:, coordinate].index_select(0, lower)
                - sorted_values[:-1, coordinate].index_select(0, lower)
            )
            result[:, coordinate] = torch.where(
                low,
                sorted_values[0, coordinate],
                torch.where(high, sorted_values[-1, coordinate], interpolated),
            )
        result = result.detach().cpu().to(dtype=torch.float64).contiguous()
        _require(torch.isfinite(result).all() and result.ge(0.0).all(), "producer sample is invalid")
        return result

    return sample


def _sampler_parity_check(
    state: Mapping[str, Any],
    state_sha256: str,
    fast_sample,
) -> dict[str, Any]:
    seed = _seed("CALIBRATION_PSEUDO_DATA", 0, None)
    public = producer_consumer.sample_from_state(
        state,
        producer_state_sha256=state_sha256,
        requested_sample_count=3,
        rng_namespace=PRODUCER_SAMPLING_NAMESPACE,
        seed=seed,
    )
    fast = fast_sample(3, seed)
    _require(torch.equal(public, fast), "pre-bound sampler differs from frozen public sampler")
    return {
        "checked": True,
        "seed": seed,
        "requested_sample_count": 3,
        "public_q_tensor_sha256": tensor_sha256(public),
        "fast_q_tensor_sha256": tensor_sha256(fast),
        "bitwise_equal": True,
        "state_only": True,
    }


def _energy_distance(first: torch.Tensor, second: torch.Tensor) -> float:
    value = metric_consumer._energy_distance(first, second)
    _require(isinstance(value, float) and torch.isfinite(torch.tensor(value, dtype=torch.float64)), "energy distance is invalid")
    _require(value >= 0.0, "energy distance is negative")
    return value


def _primary_statistic(data: torch.Tensor, model_banks: list[torch.Tensor]) -> tuple[float, list[float]]:
    values = [_energy_distance(data, bank) for bank in model_banks]
    _require(len(values) == MODEL_BANK_COUNT, "primary model-bank count changed")
    statistic = float(sum(values) / MODEL_BANK_COUNT)
    _require(torch.isfinite(torch.tensor(statistic, dtype=torch.float64)) and statistic >= 0.0, "primary statistic is invalid")
    return statistic, values


def _calibrate(
    fast_sample,
    *,
    progress_every: int = 25,
) -> tuple[torch.Tensor, list[int], list[list[int]], float]:
    started = time.perf_counter()
    values: list[float] = []
    pseudo_seeds: list[int] = []
    bank_seeds: list[list[int]] = []
    for trial in range(CALIBRATION_TRIALS):
        pseudo_seed = _seed("CALIBRATION_PSEUDO_DATA", trial, None)
        seeds = [_seed("CALIBRATION_MODEL_BANK", trial, bank) for bank in range(MODEL_BANK_COUNT)]
        pseudo = fast_sample(SEALED_ROWS, pseudo_seed)
        banks = [fast_sample(MODEL_BANK_SIZE, seed) for seed in seeds]
        statistic, _ = _primary_statistic(pseudo, banks)
        values.append(statistic)
        pseudo_seeds.append(pseudo_seed)
        bank_seeds.append(seeds)
        if trial == 0 or trial + 1 == CALIBRATION_TRIALS or (trial + 1) % progress_every == 0:
            print(
                json.dumps(
                    {
                        "stage": "model_self_calibration",
                        "completed_trials": trial + 1,
                        "total_trials": CALIBRATION_TRIALS,
                        "elapsed_seconds": round(time.perf_counter() - started, 3),
                    }
                ),
                flush=True,
            )
    tensor = torch.tensor(values, dtype=torch.float64).contiguous()
    _require(tuple(tensor.shape) == (CALIBRATION_TRIALS,), "calibration statistic shape changed")
    _require(torch.isfinite(tensor).all() and tensor.ge(0.0).all(), "calibration statistics are invalid")
    return tensor, pseudo_seeds, bank_seeds, time.perf_counter() - started


def _calibration_payload(
    repo_root: Path,
    state_sha256: str,
    values: torch.Tensor,
    pseudo_seeds: list[int],
    bank_seeds: list[list[int]],
    sampler_parity: Mapping[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    return {
        "schema": CALIBRATION_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "scope": "MODEL_ONLY_BEFORE_SEALED_EXPOSURE",
        "producer_semantic_authority": PRODUCER_SEMANTIC_AUTHORITY,
        "producer_state_sha256": state_sha256,
        "pseudo_data_rows": SEALED_ROWS,
        "model_bank_count": MODEL_BANK_COUNT,
        "model_bank_size": MODEL_BANK_SIZE,
        "trials": CALIBRATION_TRIALS,
        "primary_metric": PRIMARY_METRIC,
        "statistic_definition": "arithmetic mean over four V3 energy_distance values",
        "T_values": values,
        "T_tensor_sha256": tensor_sha256(values),
        "rng": {
            "sampling_namespace": PRODUCER_SAMPLING_NAMESPACE,
            "seed_derivation_root": SEED_DERIVATION_ROOT,
            "pseudo_data_stream": "CALIBRATION_PSEUDO_DATA",
            "model_bank_stream": "CALIBRATION_MODEL_BANK",
            "trial_indices": [0, CALIBRATION_TRIALS - 1],
            "all_trial_indices_inclusive": [0, CALIBRATION_TRIALS - 1],
            "pseudo_data_seeds": pseudo_seeds,
            "model_bank_seeds": bank_seeds,
            "no_retry": True,
            "no_early_stop": True,
            "best_of_n": False,
        },
        "sampler": {
            "producer_state_only": True,
            "pre_bound_after_one_validation": True,
            "public_sampler_parity": dict(sampler_parity),
            "producer_state_sha256": state_sha256,
        },
        "execution": {
            "device": "cpu",
            "cuda": False,
            "training": False,
            "optimizer_created": False,
            "backward_called": False,
            "elapsed_seconds": elapsed_seconds,
        },
        "sealed_firewall": {
            "sealed_content_loaded": False,
            "sealed_images_loaded": False,
            "sealed_labels_loaded": False,
            "sealed_q_loaded": False,
        },
    }


def _validate_calibration_artifact(payload: Mapping[str, Any], state_sha256: str) -> torch.Tensor:
    _require(payload.get("schema") == CALIBRATION_SCHEMA and payload.get("status") == "VALID_COMPLETED", "calibration artifact is incomplete")
    _require(payload.get("producer_state_sha256") == state_sha256, "calibration producer state binding changed")
    values = payload.get("T_values")
    _require(torch.is_tensor(values) and values.dtype == torch.float64 and tuple(values.shape) == (CALIBRATION_TRIALS,), "calibration values changed")
    _require(torch.isfinite(values).all() and values.ge(0.0).all(), "calibration values are invalid")
    _require(payload.get("T_tensor_sha256") == tensor_sha256(values), "calibration tensor hash is stale")
    return values.detach().cpu().contiguous()


def _observed_banks(fast_sample, seeds: list[int]) -> tuple[torch.Tensor, list[str]]:
    banks = [fast_sample(MODEL_BANK_SIZE, seed) for seed in seeds]
    tensor = torch.stack(banks, dim=0).contiguous()
    _require(tuple(tensor.shape) == (MODEL_BANK_COUNT, MODEL_BANK_SIZE, DIMENSION), "observed model-bank shape changed")
    _require(tensor.dtype == torch.float64 and torch.isfinite(tensor).all() and tensor.ge(0.0).all(), "observed model banks are invalid")
    return tensor, [tensor_sha256(bank) for bank in banks]


def _model_bank_payload(
    state_sha256: str,
    banks: torch.Tensor,
    seeds: list[int],
    bank_hashes: list[str],
) -> dict[str, Any]:
    return {
        "schema": MODEL_BANK_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "scope": "MODEL_ONLY_BEFORE_SEALED_EXPOSURE",
        "producer_semantic_authority": PRODUCER_SEMANTIC_AUTHORITY,
        "producer_state_sha256": state_sha256,
        "model_banks": banks,
        "model_banks_tensor_sha256": tensor_sha256(banks),
        "bank_tensor_sha256": bank_hashes,
        "bank_count": MODEL_BANK_COUNT,
        "bank_size": MODEL_BANK_SIZE,
        "dimension": DIMENSION,
        "dtype": "torch.float64",
        "rng": {
            "sampling_namespace": PRODUCER_SAMPLING_NAMESPACE,
            "seed_derivation_root": SEED_DERIVATION_ROOT,
            "stream": "OBSERVED_MODEL_BANK",
            "bank_indices": list(range(MODEL_BANK_COUNT)),
            "seeds": seeds,
            "no_replacement_bank": True,
            "no_seed_retry": True,
            "no_post_exposure_selection": True,
        },
        "execution": {
            "device": "cpu",
            "cuda": False,
            "training": False,
            "sealed_content_loaded": False,
        },
    }


def _validate_model_bank_artifact(payload: Mapping[str, Any], state_sha256: str) -> torch.Tensor:
    _require(payload.get("schema") == MODEL_BANK_SCHEMA and payload.get("status") == "VALID_COMPLETED", "model-bank artifact is incomplete")
    _require(payload.get("producer_state_sha256") == state_sha256, "model-bank producer state binding changed")
    banks = payload.get("model_banks")
    _require(torch.is_tensor(banks) and banks.dtype == torch.float64 and tuple(banks.shape) == (MODEL_BANK_COUNT, MODEL_BANK_SIZE, DIMENSION), "model-bank tensor shape/dtype changed")
    _require(torch.isfinite(banks).all() and banks.ge(0.0).all(), "model-bank tensor is invalid")
    _require(payload.get("model_banks_tensor_sha256") == tensor_sha256(banks), "model-bank tensor hash is stale")
    _require(payload.get("execution", {}).get("sealed_content_loaded") is False, "model-bank artifact loaded sealed content")
    return banks.detach().cpu().contiguous()


def _write_torch_artifact(path: Path, payload: Mapping[str, Any], validator) -> tuple[str, Any]:
    atomic_torch_save_new(path, payload, artifact_kind=path.name)
    digest = file_sha256(path)
    loaded = safe_torch_load(path, expected_sha256=digest, map_location="cpu")
    validated = validator(_mapping(loaded, path.name))
    return digest, validated


def _fixed_q_model(repo_root: Path, contract: Mapping[str, Any]):
    model, scale, identity = q_materialization._load_model_and_scale(repo_root)
    expected = _mapping(
        _mapping(contract.get("q_materialization"), "contract q materialization").get("future_source_to_q"),
        "contract future source-to-q",
    )
    fixed = _mapping(expected.get("fixed_ols_authority"), "contract fixed OLS authority")
    for key in (
        "calibration_contract",
        "calibration_path",
        "calibration_sha256",
        "checkpoint_global_step",
        "checkpoint_path",
        "checkpoint_sha256",
        "mode_bank_sha256",
        "model_state_digest_before",
    ):
        _require(identity.get(key) == fixed.get(key), f"fixed OLS identity changed: {key}")
    _require((repo_root / RAW_IMAGES_PATH).is_file(), "sealed raw image artifact is missing")
    return model, scale, identity


def _publish_consumption_marker(
    output_dir: Path,
    contract_sha256: str,
    producer_sha256: str,
    sealed: Mapping[str, Any],
    calibration_sha256: str,
    model_bank_sha256: str,
) -> dict[str, Any]:
    path = output_dir / CONSUMPTION_NAME
    _require(not path.exists(), "consumption marker already exists")
    payload = {
        "schema": CONSUMPTION_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "status": "CONSUMED",
        "consumed": True,
        "consumed_at_utc": _utc_now(),
        "transition": "SEALED_CONFIRMATION_ARMED -> CONSUMED",
        "transition_point": "immediately before first protected sealed source read/materialization",
        "contract_sha256": contract_sha256,
        "producer_semantic_authority": PRODUCER_SEMANTIC_AUTHORITY,
        "producer_state_sha256": producer_sha256,
        "reservation_authority": SEALED_AUTHORITY,
        "reservation_manifest_path": sealed["path"],
        "reservation_manifest_sha256": sealed["sha256"],
        "semantic_confirmation_id": SEALED_CONFIRMATION_ID,
        "reserved_row_count": SEALED_ROWS,
        "reserved_rows_per_class": SEALED_ROWS_PER_CLASS,
        "reserved_source_index_sha256": sealed["source_index_sha256"],
        "reserved_ordered_row_identity_sha256": sealed["ordered_row_identity_sha256"],
        "model_only_calibration_artifact_sha256": calibration_sha256,
        "observed_model_bank_artifact_sha256": model_bank_sha256,
        "sealed_content_read_before_marker": False,
        "sealed_q_materialized_before_marker": False,
        "reset_to_false_forbidden": True,
        "retry_same_reservation_forbidden": True,
    }
    atomic_write_new_json(path, payload, artifact_kind="sealed consumption marker")
    return payload


def _covariance(value: torch.Tensor) -> torch.Tensor:
    view = value.detach().to(dtype=torch.float64).cpu().contiguous()
    _require(view.ndim == 2 and view.shape[0] >= 2, "descriptive covariance input is invalid")
    centered = view - view.mean(dim=0, keepdim=True)
    covariance = centered.transpose(0, 1).mm(centered).div(float(view.shape[0] - 1))
    _require(torch.isfinite(covariance).all(), "descriptive covariance is non-finite")
    return covariance


def _descriptive_pair_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, float]:
    left = first.detach().to(dtype=torch.float64).cpu().contiguous()
    right = second.detach().to(dtype=torch.float64).cpu().contiguous()
    _require(left.ndim == 2 and right.ndim == 2 and left.shape[1] == right.shape[1], "descriptive inputs are invalid")
    left_cov = _covariance(left)
    right_cov = _covariance(right)
    right_cov_norm = right_cov.square().sum()
    _require(torch.isfinite(right_cov_norm) and right_cov_norm.gt(0.0), "descriptive reference covariance is invalid")
    left_norm = left.norm(dim=1).unsqueeze(1)
    right_norm = right.norm(dim=1).unsqueeze(1)
    left_energy = left.square().sum(dim=1).unsqueeze(1)
    right_energy = right.square().sum(dim=1).unsqueeze(1)
    per_coordinate = [
        _energy_distance(left[:, coordinate : coordinate + 1], right[:, coordinate : coordinate + 1])
        for coordinate in range(DIMENSION)
    ]
    values = {
        "covariance_relative_frobenius": float((left_cov - right_cov).square().sum() / right_cov_norm),
        "norm_energy_distance": _energy_distance(left_norm, right_norm),
        "squared_norm_energy_distance": _energy_distance(left_energy, right_energy),
        "marginal_mean_l2": float((left.mean(dim=0) - right.mean(dim=0)).norm()),
        "marginal_std_l2": float((left.std(dim=0, unbiased=True) - right.std(dim=0, unbiased=True)).norm()),
        "marginal_energy_distance_mean": float(sum(per_coordinate) / DIMENSION),
        "marginal_energy_distance_max": float(max(per_coordinate)),
    }
    _require(set(values) == set(DESCRIPTIVE_ONLY_METRICS), "descriptive metric partition changed")
    _require(all(torch.isfinite(torch.tensor(value, dtype=torch.float64)) and value >= 0.0 for value in values.values()), "descriptive metric is invalid")
    return values


def _aggregate_descriptive(metrics: list[Mapping[str, float]]) -> dict[str, float]:
    _require(len(metrics) == MODEL_BANK_COUNT, "descriptive bank count changed")
    return {
        name: float(sum(float(item[name]) for item in metrics) / MODEL_BANK_COUNT)
        for name in DESCRIPTIVE_ONLY_METRICS
    }


def _sealed_q_payload(
    repo_root: Path,
    q: torch.Tensor,
    sealed: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    state_sha256: str,
    model_state_digest_after: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    _require(tuple(q.shape) == (SEALED_ROWS, DIMENSION) and q.dtype == torch.float64, "sealed q shape/dtype changed")
    _require(torch.isfinite(q).all() and q.ge(0.0).all(), "sealed q is invalid")
    return {
        "schema": Q_ARTIFACT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "semantic_authority_id": SEALED_AUTHORITY,
        "semantic_confirmation_id": SEALED_CONFIRMATION_ID,
        "q_coordinate": "R2_AMPLITUDE",
        "q_definition": "q=abs(c/sqrt(s))=abs(R3_COMPLEX)",
        "q_dtype": "torch.float64",
        "q_shape": [SEALED_ROWS, DIMENSION],
        "q": q,
        "q_tensor_sha256": tensor_sha256(q),
        "reservation_manifest_path": sealed["path"],
        "reservation_manifest_sha256": sealed["sha256"],
        "reservation_authority": SEALED_AUTHORITY,
        "membership": {
            "row_count": SEALED_ROWS,
            "rows_per_class": SEALED_ROWS_PER_CLASS,
            "source_index_sha256": sealed["source_index_sha256"],
            "label_sha256": sealed["label_sha256"],
            "ordered_row_identity_sha256": sealed["ordered_row_identity_sha256"],
            "source_indices": sealed["indices"],
            "labels_for_identity_only": sealed["labels"],
            "ordered_row_identity": sealed["ordered_identity"],
        },
        "source_indices": sealed["indices"],
        "labels_for_identity_only": sealed["labels"],
        "ordered_row_identity": sealed["ordered_identity"],
        "frozen_ols_authority": dict(model_identity),
        "producer_state_sha256": state_sha256,
        "execution": {
            "device": "cpu",
            "batch_size": q_materialization.BATCH_SIZE,
            "selected_source_rows_only": True,
            "sealed_rows_materialized_once": True,
            "sealed_content_read_before_marker": False,
            "sealed_q_loaded_before_marker": False,
            "decoder_entered": False,
            "evaluator_entered": False,
            "optimizer_created": False,
            "backward_called": False,
            "training_steps": 0,
            "model_state_digest_after": model_state_digest_after,
            "model_state_unchanged": model_state_digest_after == model_identity["model_state_digest_before"],
            "elapsed_seconds": elapsed_seconds,
        },
        "governance": {
            "selected_producer": PRODUCER_FAMILY,
            "frozen_producer_state": PRODUCER_SEMANTIC_AUTHORITY,
            "sealed_confirmation_consumed": True,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
    }


def _sealed_q_manifest(
    repo_root: Path,
    q_payload: Mapping[str, Any],
    q_path: Path,
    q_sha256: str,
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    membership = dict(q_payload["membership"])
    membership["source_indices"] = membership["source_indices"].tolist()
    membership["labels_for_identity_only"] = membership["labels_for_identity_only"].tolist()
    return {
        "schema": Q_MANIFEST_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "status": "VALID_COMPLETED",
        "reservation": {
            "authority": SEALED_AUTHORITY,
            "semantic_confirmation_id": SEALED_CONFIRMATION_ID,
            "manifest_path": q_payload["reservation_manifest_path"],
            "manifest_sha256": q_payload["reservation_manifest_sha256"],
            "row_count": SEALED_ROWS,
            "rows_per_class": SEALED_ROWS_PER_CLASS,
        },
        "membership": membership,
        "q_artifact": {
            "path": _relative(repo_root, q_path),
            "sha256": q_sha256,
            "schema": Q_ARTIFACT_SCHEMA,
            "q_tensor_sha256": q_payload["q_tensor_sha256"],
            "shape": q_payload["q_shape"],
            "dtype": q_payload["q_dtype"],
            "size_bytes": q_path.stat().st_size,
        },
        "source_to_q": {
            "implementation_path": _relative(repo_root, repo_root / Path("scripts/research/materialize_n3_development_bank_v2_q.py")),
            "implementation_sha256": Q_MATERIALIZER_SHA256,
            "encoder_prefix_path": _relative(repo_root, repo_root / Path("scripts/research/replay_scaled_chen_repaired_coordinate_mra.py")),
            "encoder_prefix_sha256": Q_REPLAY_SHA256,
            "raw_image_path": _relative(repo_root, repo_root / RAW_IMAGES_PATH),
            "raw_image_sha256": q_materialization.RAW_IMAGES_SHA256,
            "row_order": "exact sealed membership source_indices order",
            "normalization": "normalized = coefficients / sqrt(raw_port_mean); q = abs(normalized)",
            "q_definition": "R2_AMPLITUDE = abs(c/sqrt(s)) = abs(R3_COMPLEX)",
        },
        "frozen_ols_authority": dict(model_identity),
        "execution": dict(q_payload["execution"]),
        "governance": dict(q_payload["governance"]),
        "sealed_confirmation_consumed": True,
    }


def _valid_lineage(
    repo_root: Path,
    git_before: Mapping[str, Any],
    git_after: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    calibration_sha256: str,
    calibration_tensor_sha256: str,
    model_bank_sha256: str,
    model_bank_tensor_sha256: str,
    model_bank_seeds: list[int],
    marker_payload: Mapping[str, Any],
    marker_sha256: str,
    q_sha256: str,
    q_manifest_sha256: str,
    q_tensor_sha256: str,
    t_sealed: float,
    per_bank_t_sealed: list[float],
    exceedance_count: int,
    p_mc: float,
    descriptive: Mapping[str, float],
    classification: str,
    elapsed_seconds: float,
    targeted_tests_passed: bool,
    static_checks_passed: bool,
) -> dict[str, Any]:
    return {
        "schema": LINEAGE_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "result": "VALID_COMPLETED",
        "classification": classification,
        "created_at_utc": _utc_now(),
        "git": {
            "branch": git_before["branch"],
            "starting_head": git_before["head"],
            "ending_head": git_after["head"],
            "worktree_before": dict(git_before),
            "worktree_after": dict(git_after),
        },
        "contract": {
            "path": _relative(repo_root, repo_root / CONTRACT_PATH),
            "sha256": CONTRACT_SHA256,
            "unchanged": True,
        },
        "producer": {
            "semantic_authority": PRODUCER_SEMANTIC_AUTHORITY,
            "family": PRODUCER_FAMILY,
            "candidate": PRODUCER_CANDIDATE,
            "state_path": preflight["producer"]["state_path"],
            "state_sha256": PRODUCER_STATE_SHA256,
            "lineage_path": preflight["producer"]["lineage_path"],
            "lineage_sha256": PRODUCER_LINEAGE_SHA256,
            "unchanged_after_model_only": file_sha256(repo_root / Path(preflight["producer"]["state_path"])) == PRODUCER_STATE_SHA256,
            "refit": False,
            "modified": False,
        },
        "pre_sealed": {
            "calibration": {
                "path": _relative(repo_root, repo_root / OUTPUT_DIR / CALIBRATION_NAME),
                "sha256": calibration_sha256,
                "T_tensor_sha256": calibration_tensor_sha256,
                "trials": CALIBRATION_TRIALS,
                "pseudo_data_rows": SEALED_ROWS,
                "model_bank_count": MODEL_BANK_COUNT,
                "model_bank_size": MODEL_BANK_SIZE,
            },
            "observed_model_banks": {
                "path": _relative(repo_root, repo_root / OUTPUT_DIR / MODEL_BANK_NAME),
                "sha256": model_bank_sha256,
                "tensor_sha256": model_bank_tensor_sha256,
                "bank_count": MODEL_BANK_COUNT,
                "bank_size": MODEL_BANK_SIZE,
                "seeds": model_bank_seeds,
                "namespace": PRODUCER_SAMPLING_NAMESPACE,
            },
            "sampler_parity": dict(preflight["sampler_parity"]),
            "q_model_identity": dict(preflight["q_model_identity"]),
            "armed_state": "SEALED_CONFIRMATION_ARMED",
            "armed_before_first_protected_read": True,
            "targeted_consumer_tests_passed": targeted_tests_passed,
            "static_checks_passed": static_checks_passed,
        },
        "consumption": {
            "marker_path": _relative(repo_root, repo_root / OUTPUT_DIR / CONSUMPTION_NAME),
            "marker_sha256": marker_sha256,
            "marker_consumed": marker_payload["consumed"],
            "transition": marker_payload["transition"],
            "reset_to_false_forbidden": True,
            "retry_same_reservation_forbidden": True,
        },
        "sealed": {
            "reservation_authority": SEALED_AUTHORITY,
            "manifest_path": preflight["sealed"]["path"],
            "manifest_sha256": SEALED_MANIFEST_SHA256,
            "semantic_confirmation_id": SEALED_CONFIRMATION_ID,
            "source_count": SEALED_ROWS,
            "rows_per_class": SEALED_ROWS_PER_CLASS,
            "source_index_sha256": preflight["sealed"]["source_index_sha256"],
            "ordered_row_identity_sha256": preflight["sealed"]["ordered_row_identity_sha256"],
            "consumed": True,
            "q_artifact_path": _relative(repo_root, repo_root / OUTPUT_DIR / SEALED_Q_NAME),
            "q_artifact_sha256": q_sha256,
            "q_manifest_path": _relative(repo_root, repo_root / OUTPUT_DIR / SEALED_Q_MANIFEST_NAME),
            "q_manifest_sha256": q_manifest_sha256,
            "q_tensor_sha256": q_tensor_sha256,
            "materialized_once": True,
        },
        "primary_statistic": {
            "metric": PRIMARY_METRIC,
            "aggregation": "unweighted arithmetic mean over four model banks",
            "T_sealed": t_sealed,
            "per_bank_T_sealed": per_bank_t_sealed,
            "computed_once": True,
            "labels_used": False,
        },
        "monte_carlo_decision": {
            "B": CALIBRATION_TRIALS,
            "alpha": ALPHA,
            "inequality": "T_b >= T_sealed",
            "exceedance_count": exceedance_count,
            "p_MC": p_mc,
            "computed_once": True,
            "decision_source": "model-self calibration artifact only",
            "descriptive_metrics_used": False,
        },
        "descriptive_metrics": {
            "role": "DESCRIPTIVE_ONLY",
            "values": dict(descriptive),
            "may_change_decision": False,
            "computed_after_primary_classification": True,
        },
        "governance": {
            "selected_producer": PRODUCER_FAMILY,
            "n3_amplitude_producer_confirmation": "PASSED" if classification == CLASSIFICATION_PASSED else "FAILED",
            "sealed_confirmation_consumed": True,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "firewalls": {
            "sealed_content_read": True,
            "sealed_images_loaded": True,
            "sealed_labels_loaded_beyond_manifest": False,
            "sealed_q_loaded": True,
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
            "optimizer_created": False,
            "backward_called": False,
        },
        "claim_boundary": {
            "supports_only": [
                classification,
                "non-rejection of compatibility under this exact frozen contract"
                if classification == CLASSIFICATION_PASSED
                else "independent confirmation was not obtained",
            ],
            "does_not_establish": [
                "population distribution equality",
                "OOD generalization",
                "complete complex OLS validity",
                "amplitude-phase joint compatibility",
                "semantic controllability",
                "image-generation readiness",
                "optical realizability",
                "training authorization",
                "formal promotion",
            ],
        },
        "execution": {
            "device": "cpu",
            "elapsed_seconds": elapsed_seconds,
            "no_cuda_or_training": True,
        },
        "next_action": (
            "DEFINE_N3_COMPLEX_OLS_AMPLITUDE_PHASE_JOINT_COMPATIBILITY_CONTRACT_V1"
            if classification == CLASSIFICATION_PASSED
            else "STOP"
        ),
    }


def _invalid_lineage(
    repo_root: Path,
    git_before: Mapping[str, Any],
    preflight: Mapping[str, Any],
    *,
    marker_sha256: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema": LINEAGE_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "invalid_consumed",
        "result": CLASSIFICATION_INVALID_CONSUMED,
        "classification": CLASSIFICATION_INVALID_CONSUMED,
        "created_at_utc": _utc_now(),
        "blocker": {
            "type": "POST_CONSUMPTION_EXECUTION_INVALID",
            "message": reason,
            "action": "preserve consumed=true; do not retry or tune from sealed data",
        },
        "git": {
            "branch": git_before["branch"],
            "starting_head": git_before["head"],
            "ending_head": _git_state(repo_root)["head"],
            "worktree_before": dict(git_before),
        },
        "contract": {
            "path": _relative(repo_root, repo_root / CONTRACT_PATH),
            "sha256": CONTRACT_SHA256,
            "unchanged": True,
        },
        "producer": {
            "semantic_authority": PRODUCER_SEMANTIC_AUTHORITY,
            "state_sha256": PRODUCER_STATE_SHA256,
            "lineage_sha256": PRODUCER_LINEAGE_SHA256,
            "refit": False,
            "modified": False,
        },
        "consumption": {
            "marker_path": _relative(repo_root, repo_root / OUTPUT_DIR / CONSUMPTION_NAME),
            "marker_sha256": marker_sha256,
            "consumed": True,
            "reset_to_false_forbidden": True,
            "retry_same_reservation_forbidden": True,
        },
        "sealed": {
            "reservation_authority": SEALED_AUTHORITY,
            "manifest_path": preflight["sealed"]["path"],
            "manifest_sha256": SEALED_MANIFEST_SHA256,
            "source_count": SEALED_ROWS,
            "rows_per_class": SEALED_ROWS_PER_CLASS,
            "q_materialized": False,
        },
        "governance": {
            "selected_producer": PRODUCER_FAMILY,
            "n3_amplitude_producer_confirmation": "INVALID_CONSUMED",
            "sealed_confirmation_consumed": True,
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
        },
        "firewalls": {
            "sealed_content_read": True,
            "cuda": False,
            "training": False,
            "producer_refit": False,
            "producer_modified": False,
        },
        "next_action": "STOP",
    }


def _blocked_result(
    repo_root: Path,
    git_before: Mapping[str, Any],
    reason: str,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "result": "BLOCKED",
        "classification": "BLOCKED",
        "gate": GATE,
        "blocker": {
            "type": "INVALID_HARNESS_OR_PRE_CONSUMPTION_STOP",
            "message": reason,
            "action": "do not read sealed content; preserve RESERVED_NOT_MATERIALIZED and consumed=false",
        },
        "git": {
            "branch": git_before.get("branch"),
            "starting_head": git_before.get("head"),
            "ending_head": _git_state(repo_root).get("head"),
            "worktree_before": dict(git_before),
        },
        "sealed": {
            "reservation_authority": SEALED_AUTHORITY,
            "status": "RESERVED_NOT_MATERIALIZED",
            "consumed": False,
            "content_read": False,
        },
        "output_dir": _relative(repo_root, output_dir) if output_dir.is_relative_to(repo_root) else str(output_dir),
    }


def execute(
    repo_root: Path,
    output_dir: Path = OUTPUT_DIR,
    *,
    targeted_tests_passed: bool = False,
    static_checks_passed: bool = False,
) -> dict[str, Any]:
    root = repo_root.resolve()
    target = output_dir if output_dir.is_absolute() else root / output_dir
    target = target.resolve()
    git_before = _git_state(root)
    marker_published = False
    preflight: dict[str, Any] | None = None
    staging: Path | None = None
    started = time.perf_counter()

    try:
        _require(git_before["branch"] == EXPECTED_BRANCH, "current branch is not the frozen authority branch")
        _require(git_before["head"] == EXPECTED_HEAD, "current HEAD is not the frozen authority base")
        if target.exists():
            marker = target / CONSUMPTION_NAME
            if marker.is_file():
                existing = json.loads(marker.read_text(encoding="utf-8"))
                if existing.get("consumed") is True:
                    return {
                        "result": CLASSIFICATION_INVALID_CONSUMED,
                        "classification": CLASSIFICATION_INVALID_CONSUMED,
                        "reason": "write-once execution output already contains consumed=true",
                        "sealed_consumed": True,
                    }
            return _blocked_result(root, git_before, f"refusing to reuse write-once execution output: {target}", output_dir=target)
        _require(target.parent.resolve() == (root / Path("outputs/20260910_v2/runs")).resolve(), "execution output must remain in the authorized run root")
        _require(targeted_tests_passed, "targeted execution tests were not declared passed")
        _require(static_checks_passed, "compile/Ruff pre-sealed checks were not declared passed")

        contract, contract_identity = _validate_contract(root)
        current_governance = _validate_current_governance(root, contract)
        development = _validate_development_and_v3(root, contract)
        sealed = _validate_sealed_reservation(root, contract, development["source_manifest"]["source_indices"])
        state, producer_identity = _validate_producer(root, contract)
        state_sha256 = PRODUCER_STATE_SHA256
        fast_sample = _make_fast_sampler(state)
        sampler_parity = _sampler_parity_check(state, state_sha256, fast_sample)
        preflight = {
            "contract": contract_identity,
            "current_governance": current_governance,
            "development": development,
            "sealed": sealed,
            "producer": producer_identity,
            "sampler_parity": sampler_parity,
        }

        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        calibration_values, pseudo_seeds, calibration_bank_seeds, calibration_elapsed = _calibrate(fast_sample)
        calibration_payload = _calibration_payload(
            root,
            state_sha256,
            calibration_values,
            pseudo_seeds,
            calibration_bank_seeds,
            sampler_parity,
            calibration_elapsed,
        )
        calibration_sha256, loaded_calibration = _write_torch_artifact(
            staging / CALIBRATION_NAME,
            calibration_payload,
            lambda value: _validate_calibration_artifact(value, state_sha256),
        )
        _require(torch.equal(loaded_calibration, calibration_values), "serialized calibration values changed")

        observed_seeds = [_seed("OBSERVED_MODEL_BANK", None, bank) for bank in range(MODEL_BANK_COUNT)]
        observed_banks, bank_tensor_hashes = _observed_banks(fast_sample, observed_seeds)
        model_bank_payload = _model_bank_payload(state_sha256, observed_banks, observed_seeds, bank_tensor_hashes)
        model_bank_sha256, loaded_banks = _write_torch_artifact(
            staging / MODEL_BANK_NAME,
            model_bank_payload,
            lambda value: _validate_model_bank_artifact(value, state_sha256),
        )
        _require(torch.equal(loaded_banks, observed_banks), "serialized observed model banks changed")

        _require(file_sha256(root / Path(preflight["producer"]["state_path"])) == PRODUCER_STATE_SHA256, "producer state changed during model-only execution")
        model, scale, model_identity = _fixed_q_model(root, contract)
        preflight["q_model_identity"] = dict(model_identity)
        consumer_sha256 = file_sha256(root / Path(__file__).resolve().relative_to(root))
        preflight["consumer_path"] = _relative(root, root / Path(__file__).resolve().relative_to(root))
        preflight["consumer_sha256"] = consumer_sha256
        _require(targeted_tests_passed and static_checks_passed, "arming prerequisites changed before sealed exposure")
        _require(preflight["sealed"]["consumed_before"] is False and preflight["sealed"]["status_before_consumption"] == "RESERVED_NOT_MATERIALIZED", "sealed reservation changed before arm")
        _require(
            file_sha256(root / Path("scripts/research/materialize_n3_development_bank_v2_q.py")) == Q_MATERIALIZER_SHA256
            and file_sha256(root / Path("scripts/research/replay_scaled_chen_repaired_coordinate_mra.py")) == Q_REPLAY_SHA256,
            "source-to-q implementation changed before arm",
        )
        _require(
            file_sha256(root / Path("scripts/research/validate_n3_amplitude_producer_baseline.py")) == V3_METRIC_SHA256
            and file_sha256(root / Path("scripts/research/execute_n3_development_bank_v2_empirical_ceiling.py")) == V3_EVALUATOR_IMPL_SHA256
            and file_sha256(root / Path("scripts/research/execute_n3_empirical_marginal_gaussian_copula_producer_v1.py")) == V3_AGGREGATION_SHA256,
            "V3 implementation changed before arm",
        )

        # Publish the complete model-only prefix before arming.  No sealed
        # source is read before this directory rename and marker publication.
        if target.exists():
            raise FileExistsError(f"execution output appeared during pre-sealed prefix: {target}")
        os.replace(staging, target)
        staging = None
        marker_payload = _publish_consumption_marker(
            target,
            CONTRACT_SHA256,
            state_sha256,
            sealed,
            calibration_sha256,
            model_bank_sha256,
        )
        marker_published = True
        marker_sha256 = file_sha256(target / CONSUMPTION_NAME)
        first_protected_read_at = _utc_now()

        # This is the first protected scientific read after consumed=true.
        images = q_materialization._load_selected_images(root, sealed["indices"])
        q = q_materialization._collect_q(model, images, scale)
        model_state_digest_after = q_materialization._state_digest(model)
        _require(model_state_digest_after == model_identity["model_state_digest_before"], "fixed q model state changed after sealed materialization")
        q_payload = _sealed_q_payload(
            root,
            q,
            sealed,
            model_identity,
            state_sha256,
            model_state_digest_after,
            time.perf_counter() - started,
        )
        q_path = target / SEALED_Q_NAME
        q_sha256 = file_sha256(q_path) if q_path.exists() else ""
        atomic_torch_save_new(q_path, q_payload, artifact_kind="sealed q artifact")
        q_sha256 = file_sha256(q_path)
        loaded_q_payload = _mapping(safe_torch_load(q_path, expected_sha256=q_sha256, map_location="cpu"), "sealed q artifact")
        loaded_q = loaded_q_payload.get("q")
        _require(torch.is_tensor(loaded_q) and tensor_sha256(loaded_q) == q_payload["q_tensor_sha256"], "sealed q artifact reload changed")
        q_manifest = _sealed_q_manifest(root, q_payload, q_path, q_sha256, model_identity)
        atomic_write_new_json(target / SEALED_Q_MANIFEST_NAME, q_manifest, artifact_kind="sealed q materialization manifest")
        q_manifest_sha256 = file_sha256(target / SEALED_Q_MANIFEST_NAME)

        t_sealed, per_bank_t_sealed = _primary_statistic(q, [loaded_banks[index] for index in range(MODEL_BANK_COUNT)])
        null_values = loaded_calibration
        exceedance_count = int(null_values.ge(t_sealed).sum().item())
        p_mc = float((1 + exceedance_count) / (CALIBRATION_TRIALS + 1))
        _require(torch.isfinite(torch.tensor(p_mc, dtype=torch.float64)) and 0.0 <= p_mc <= 1.0, "Monte-Carlo p-value is invalid")
        classification = CLASSIFICATION_PASSED if p_mc >= ALPHA else CLASSIFICATION_FAILED

        descriptive_pairs = [
            _descriptive_pair_metrics(q, loaded_banks[index])
            for index in range(MODEL_BANK_COUNT)
        ]
        descriptive = _aggregate_descriptive(descriptive_pairs)
        git_after = _git_state(root)
        lineage = _valid_lineage(
            root,
            git_before,
            git_after,
            preflight,
            calibration_sha256=calibration_sha256,
            calibration_tensor_sha256=q_payload.get("calibration_tensor_sha256", tensor_sha256(loaded_calibration)),
            model_bank_sha256=model_bank_sha256,
            model_bank_tensor_sha256=tensor_sha256(loaded_banks),
            model_bank_seeds=observed_seeds,
            marker_payload=marker_payload,
            marker_sha256=marker_sha256,
            q_sha256=q_sha256,
            q_manifest_sha256=q_manifest_sha256,
            q_tensor_sha256=q_payload["q_tensor_sha256"],
            t_sealed=t_sealed,
            per_bank_t_sealed=per_bank_t_sealed,
            exceedance_count=exceedance_count,
            p_mc=p_mc,
            descriptive=descriptive,
            classification=classification,
            elapsed_seconds=time.perf_counter() - started,
            targeted_tests_passed=targeted_tests_passed,
            static_checks_passed=static_checks_passed,
        )
        lineage["consumption"]["first_protected_read_at_utc"] = first_protected_read_at
        atomic_write_new_json(target / LINEAGE_NAME, lineage, artifact_kind="sealed confirmation lineage")
        return {
            "result": "VALID_COMPLETED",
            "classification": classification,
            "contract_sha256": CONTRACT_SHA256,
            "producer_state_sha256": PRODUCER_STATE_SHA256,
            "producer_lineage_sha256": PRODUCER_LINEAGE_SHA256,
            "calibration_artifact": _relative(root, target / CALIBRATION_NAME),
            "calibration_artifact_sha256": calibration_sha256,
            "calibration_T_tensor_sha256": tensor_sha256(loaded_calibration),
            "calibration_trials": CALIBRATION_TRIALS,
            "model_bank_artifact": _relative(root, target / MODEL_BANK_NAME),
            "model_bank_artifact_sha256": model_bank_sha256,
            "model_bank_tensor_sha256": tensor_sha256(loaded_banks),
            "model_bank_seeds": observed_seeds,
            "arming": "SEALED_CONFIRMATION_ARMED",
            "consumption_marker": _relative(root, target / CONSUMPTION_NAME),
            "consumption_marker_sha256": marker_sha256,
            "sealed_q_artifact": _relative(root, q_path),
            "sealed_q_artifact_sha256": q_sha256,
            "sealed_q_manifest": _relative(root, target / SEALED_Q_MANIFEST_NAME),
            "sealed_q_manifest_sha256": q_manifest_sha256,
            "sealed_q_tensor_sha256": q_payload["q_tensor_sha256"],
            "sealed_rows": SEALED_ROWS,
            "sealed_rows_per_class": SEALED_ROWS_PER_CLASS,
            "T_sealed": t_sealed,
            "per_bank_T_sealed": per_bank_t_sealed,
            "mc_exceedance_count": exceedance_count,
            "p_MC": p_mc,
            "descriptive_only": descriptive,
            "lineage": _relative(root, target / LINEAGE_NAME),
            "lineage_sha256": file_sha256(target / LINEAGE_NAME),
            "selected_producer": PRODUCER_FAMILY,
            "n3_amplitude_producer_confirmation": "PASSED" if classification == CLASSIFICATION_PASSED else "FAILED",
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "cuda": False,
            "training": False,
            "producer_modified": False,
            "next_action": lineage["next_action"],
        }
    except Exception as exc:
        if staging is not None and staging.exists() and not marker_published:
            shutil.rmtree(staging)
        if marker_published:
            marker_path = target / CONSUMPTION_NAME
            marker_sha256 = file_sha256(marker_path) if marker_path.is_file() else "unavailable"
            if preflight is None:
                raise
            invalid = _invalid_lineage(
                root,
                git_before,
                preflight,
                marker_sha256=marker_sha256,
                reason=str(exc),
            )
            atomic_write_new_json(target / LINEAGE_NAME, invalid, artifact_kind="invalid consumed lineage")
            return {
                "result": CLASSIFICATION_INVALID_CONSUMED,
                "classification": CLASSIFICATION_INVALID_CONSUMED,
                "reason": str(exc),
                "consumed": True,
                "consumption_marker": _relative(root, marker_path),
                "consumption_marker_sha256": marker_sha256,
                "lineage": _relative(root, target / LINEAGE_NAME),
                "lineage_sha256": file_sha256(target / LINEAGE_NAME),
                "training_authorized": False,
                "formal_promotion": False,
                "generation_readiness": "not_ready",
            }
        return _blocked_result(root, git_before, str(exc), output_dir=target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--targeted-tests-passed", action="store_true")
    parser.add_argument("--static-checks-passed", action="store_true")
    args = parser.parse_args()
    result = execute(
        args.repo_root.resolve(),
        args.output_dir,
        targeted_tests_passed=args.targeted_tests_passed,
        static_checks_passed=args.static_checks_passed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("result") == "BLOCKED":
        return 3
    if result.get("result") == CLASSIFICATION_INVALID_CONSUMED:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
