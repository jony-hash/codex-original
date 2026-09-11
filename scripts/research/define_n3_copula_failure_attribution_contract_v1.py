"""Define the new development-only N3 copula failure attribution contract.

This definition consumer verifies the existing JSON authorities and the
recovered generated-bank manifest without loading recovered q tensors.  It
publishes the mathematical conventions for a later read-only attribution
consumer; it does not compute attribution metrics or select a producer.
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


TASK_ID = "DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "n3_copula_failure_attribution_contract.v1"
READY_CLASSIFICATION = "DEFINE_N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_V1_READY"
CLAIM_CLASSIFICATION = "N3_COPULA_FAILURE_ATTRIBUTION_CONTRACT_DEFINED"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "a5c07512614944d4be6392ae29d48e0b189bea4b"

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

FOLD_COUNT = 5
POOL_COUNT = 1000
DIMENSION = 100
FIT_ROWS_PER_FOLD = 800
HELDOUT_ROWS_PER_FOLD = 200
GENERATED_BANK_SIZE = 200
DATA_VS_MODEL_REPLICATES = 4
MODEL_VS_MODEL_REPLICATES = 8
MODEL_REFERENCE_PAIRS = ((0, 1), (2, 3), (4, 5), (6, 7))
COMPATIBILITY_Q = 0.95
MARGINAL_COMPONENTS = (
    "marginal_mean_l2",
    "marginal_std_l2",
    "marginal_energy_distance_mean",
    "marginal_energy_distance_max",
)
ORIGINAL_FAILURE_FOLDS = (0, 2, 3)
ORIGINAL_PASS_FOLDS = (1, 4)
OUTPUT_DIR = Path(
    "outputs/20260908_v2/runs/"
    "n3_copula_failure_attribution_contract_v1_attempt_01"
)
CONTRACT_NAME = "n3_copula_failure_attribution_contract.v1.json"


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
    return dict(_mapping(json.loads(target.read_text(encoding="utf-8")), name))


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


def _validate_producer_contract(contract: Mapping[str, Any]) -> None:
    _require(contract.get("schema") == "n3_empirical_marginal_gaussian_copula_producer_contract.v1", "producer contract schema changed")
    _require(contract.get("classification") == "DEFINE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1_READY", "producer contract is not ready")
    _require(contract.get("claim_classification") == "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_MODEL_ASSUMPTION_DEFINED", "producer contract claim changed")
    development = _mapping(contract.get("development_identity"), "producer contract development identity")
    _require(development.get("count") == POOL_COUNT, "development pool count changed")
    fold = _mapping(development.get("fold_assignment"), "producer contract fold assignment")
    _require(fold.get("fold_count") == FOLD_COUNT, "fold count changed")
    _require(
        fold.get("assignment_sha256") == "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32",
        "producer contract fold assignment changed",
    )
    evaluation = _mapping(contract.get("development_evaluation_contract"), "producer contract evaluation")
    _require(evaluation.get("fit_rows") == FIT_ROWS_PER_FOLD, "fit row count changed")
    _require(evaluation.get("heldout_rows") == HELDOUT_ROWS_PER_FOLD, "held-out row count changed")
    _require(evaluation.get("labels_as_model_inputs") is False, "producer labels are model inputs")
    compatibility = _mapping(contract.get("compatibility_contract"), "producer compatibility")
    _require(compatibility.get("q") == COMPATIBILITY_Q, "producer compatibility quantile changed")
    _require(compatibility.get("own_model_reference_required") is True, "producer model reference is not required")
    _require(compatibility.get("all_folds_required") is True, "producer all-fold rule changed")
    _require(
        "aggregate marginal guardrails" in str(compatibility.get("rule")),
        "producer marginal guardrail rule is missing",
    )


def _validate_producer_report(report: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    _require(report.get("schema") == "n3_empirical_marginal_gaussian_copula_producer_report.v1", "producer report schema changed")
    _require(report.get("status") == "valid_completed", "producer report is not a valid completion")
    _require(report.get("classification") == "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_NOT_SUPPORTED", "producer report classification changed")
    _require(report.get("contract", {}).get("sha256") == PRODUCER_CONTRACT_SHA256, "producer report contract identity changed")
    _require(report.get("development_fold_identity", {}).get("assignment_sha256") == "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32", "producer report fold identity changed")
    _require(report.get("development_pool", {}).get("count") == POOL_COUNT, "producer report pool count changed")
    _require(report.get("development_pool", {}).get("dimension") == DIMENSION, "producer report dimension changed")
    _require(report.get("development_pool", {}).get("q_coordinate") == "R2_AMPLITUDE", "producer report coordinate changed")
    _require(report.get("sealed_confirmation", {}).get("status") == "RESERVED_NOT_MATERIALIZED", "sealed confirmation status changed")
    _require(report.get("sealed_confirmation", {}).get("consumed") is False, "sealed confirmation was consumed")
    _require(report.get("governance", {}).get("training_authorized") is False, "training authorization changed")
    _require(report.get("governance", {}).get("formal_promotion") is False, "formal promotion changed")
    _require(report.get("claim_boundary", {}).get("generation_readiness") == "not_ready", "generation readiness changed")

    folds = report.get("fold_execution")
    _require(isinstance(folds, list) and len(folds) == FOLD_COUNT, "producer fold execution is incomplete")
    fold_ids = [item.get("fold") for item in folds if isinstance(item, Mapping)]
    _require(fold_ids == list(range(FOLD_COUNT)), "producer folds are not ordered 0..4")

    marginal_values: list[dict[str, Any]] = []
    compatibility_failures: list[int] = []
    compatibility_passes: list[int] = []
    for fold in folds:
        components = _mapping(fold.get("compatibility_components"), f"fold {fold['fold']} compatibility components")
        for name in MARGINAL_COMPONENTS:
            _bool(components.get(name), f"fold {fold['fold']} {name}")
        derived_compatible = all(bool(components[name]) for name in MARGINAL_COMPONENTS)
        reported_compatible = _bool(fold.get("marginal_compatible"), f"fold {fold['fold']} marginal_compatible")
        _require(reported_compatible == derived_compatible, f"fold {fold['fold']} marginal compatibility is ambiguous")
        _require(isinstance(fold.get("per_coordinate_marginal_q95_checks"), list), f"fold {fold['fold']} coordinate marginal checks missing")
        _require(len(fold["per_coordinate_marginal_q95_checks"]) == DIMENSION, f"fold {fold['fold']} coordinate marginal checks have wrong length")
        _require(isinstance(fold.get("data_sampling"), list) and len(fold["data_sampling"]) == DATA_VS_MODEL_REPLICATES, f"fold {fold['fold']} data bank roles are incomplete")
        _require(isinstance(fold.get("model_sampling"), list) and len(fold["model_sampling"]) == MODEL_VS_MODEL_REPLICATES, f"fold {fold['fold']} model bank roles are incomplete")
        _require(isinstance(fold.get("model_vs_model_reference_pairs"), list) and len(fold["model_vs_model_reference_pairs"]) == len(MODEL_REFERENCE_PAIRS), f"fold {fold['fold']} reference pairs are incomplete")
        if bool(fold.get("compatibility")):
            compatibility_passes.append(int(fold["fold"]))
        else:
            compatibility_failures.append(int(fold["fold"]))
        marginal_values.append(
            {
                "fold": int(fold["fold"]),
                "marginal_compatible": reported_compatible,
                "marginal_fail": not reported_compatible,
                "marginal_component_fields": list(MARGINAL_COMPONENTS),
                "original_coordinate_failure_indices": [
                    index
                    for index, passed in enumerate(fold["per_coordinate_marginal_q95_checks"])
                    if passed is False
                ],
            }
        )
    _require(compatibility_failures == list(ORIGINAL_FAILURE_FOLDS), "original failed fold set changed")
    _require(compatibility_passes == list(ORIGINAL_PASS_FOLDS), "original pass-control fold set changed")
    _require(
        [item["marginal_fail"] for item in marginal_values]
        == [True, False, True, True, False],
        "original marginal failure pattern changed",
    )

    producer_git = _mapping(report.get("git"), "producer report git")
    script_path = Path(str(producer_git.get("consumer_script", PRODUCER_EXECUTION_SCRIPT.as_posix())))
    _require(script_path == PRODUCER_EXECUTION_SCRIPT, "producer execution script path changed")
    current_script = Path(__file__).resolve().parents[2] / script_path
    _require(current_script.is_file(), "producer execution script is missing")
    producer_script_sha = str(producer_git.get("consumer_script_sha256"))
    _require(file_sha256(current_script) == producer_script_sha, "producer execution script hash changed")

    input_authorities = _mapping(report.get("input_authorities"), "producer report input authorities")
    development_banks = _mapping(input_authorities.get("development_bank_artifacts"), "development bank authorities")
    _require(development_banks.get("q_values_loaded_by_this_definition_consumer") is False, "definition consumer loaded q values")
    return {
        "folds": marginal_values,
        "development_pool": dict(_mapping(report.get("development_pool"), "development pool")),
        "input_authorities": dict(input_authorities),
        "producer_script_sha256": producer_script_sha,
        "marginal_component_fields": list(MARGINAL_COMPONENTS),
    }


def _bank_key(fold: int, role: str, replicate: int) -> str:
    return f"fold={fold}|role={role}|replicate={replicate}"


def _validate_recovery_manifest(manifest: Mapping[str, Any], repo_root: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    _require(manifest.get("schema") == "n3_frozen_copula_generated_banks_manifest.v1", "recovery manifest schema changed")
    _require(manifest.get("status") == "valid_completed", "recovery manifest is not valid")
    _require(manifest.get("classification") == "FROZEN_N3_COPULA_BANK_AUTHORITY_RECOVERED", "recovery classification changed")
    _require(manifest.get("semantic_authority_id") == "FROZEN_N3_COPULA_GENERATED_BANKS_V1", "recovery semantic authority changed")
    _require(manifest.get("source_contract", {}).get("sha256") == PRODUCER_CONTRACT_SHA256, "recovery source contract changed")
    _require(manifest.get("source_execution_report", {}).get("sha256") == PRODUCER_REPORT_SHA256, "recovery source report changed")
    _require(manifest.get("fold_assignment_sha256") == "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32", "recovery fold assignment changed")
    _require(manifest.get("bank_count") == FOLD_COUNT * (DATA_VS_MODEL_REPLICATES + MODEL_VS_MODEL_REPLICATES), "recovery bank count changed")
    _require(manifest.get("per_bank_hash_equality") is True, "recovery bank hash equality is not verified")
    shape_dtype = _mapping(manifest.get("shape_dtype_contract"), "recovery shape/dtype contract")
    _require(shape_dtype.get("shape") == [GENERATED_BANK_SIZE, DIMENSION], "recovery bank shape changed")
    _require(shape_dtype.get("dtype") == "torch.float64", "recovery bank dtype changed")
    _require(manifest.get("sealed_confirmation", {}).get("status") == "RESERVED_NOT_MATERIALIZED", "recovery sealed status changed")
    _require(manifest.get("sealed_confirmation", {}).get("consumed") is False, "recovery consumed sealed confirmation")
    _require(manifest.get("historical_evidence_unchanged") is True, "historical evidence changed during recovery")

    artifact = _mapping(manifest.get("artifact"), "recovery tensor artifact")
    _require(artifact.get("path") == _relative(repo_root, repo_root / RECOVERED_TENSOR), "recovery tensor path changed")
    _require(artifact.get("sha256") == RECOVERED_TENSOR_SHA256, "recovery tensor hash registration changed")
    tensor_path = repo_root / RECOVERED_TENSOR
    _require(tensor_path.is_file(), "recovered tensor artifact is missing")
    _require(file_sha256(tensor_path) == RECOVERED_TENSOR_SHA256, "recovered tensor artifact hash changed")

    identities = manifest.get("per_bank_identity")
    _require(isinstance(identities, list) and len(identities) == manifest["bank_count"], "recovery bank identity registry is incomplete")
    by_key: dict[str, dict[str, Any]] = {}
    for item in identities:
        entry = _mapping(item, "recovery bank identity")
        key = str(entry.get("bank_key"))
        _require(key not in by_key, f"duplicate recovery bank key: {key}")
        _require(entry.get("bank_size") == GENERATED_BANK_SIZE, f"bank size changed: {key}")
        _require(entry.get("dtype") == "torch.float64", f"bank dtype changed: {key}")
        _require(entry.get("hash_equal") is True, f"bank hash is not equal: {key}")
        _require(isinstance(entry.get("expected_q_sha256"), str), f"bank q hash missing: {key}")
        by_key[key] = dict(entry)

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
    for fold in report["fold_execution"]:
        for sample in fold["data_sampling"]:
            key = _bank_key(int(fold["fold"]), "data-vs-model", int(sample["replicate"]))
            _require(sample.get("role") == "data-vs-model", f"data role changed: {key}")
            _require(sample.get("q_sha256") == by_key[key]["expected_q_sha256"], f"report/recovery q hash mismatch: {key}")
            report_hashes[key] = str(sample["q_sha256"])
        for sample in fold["model_sampling"]:
            key = _bank_key(int(fold["fold"]), "model-vs-model", int(sample["replicate"]))
            _require(sample.get("role") == "model-vs-model", f"model role changed: {key}")
            _require(sample.get("q_sha256") == by_key[key]["expected_q_sha256"], f"report/recovery q hash mismatch: {key}")
            report_hashes[key] = str(sample["q_sha256"])
    _require(set(report_hashes) == expected_keys, "report bank-role registry is incomplete")
    return {
        "manifest": {
            "path": _relative(repo_root, repo_root / RECOVERY_MANIFEST),
            "sha256": RECOVERY_MANIFEST_SHA256,
            "semantic_authority_id": manifest["semantic_authority_id"],
        },
        "tensor_artifact": {
            "path": _relative(repo_root, repo_root / RECOVERED_TENSOR),
            "sha256": RECOVERED_TENSOR_SHA256,
            "loaded_as_tensor": False,
            "identity_hash_verified": True,
        },
        "bank_count": len(by_key),
        "bank_identities": [by_key[key] for key in sorted(by_key)],
        "report_recovery_q_hashes_equal": True,
    }


def fold_attribution_label(marginal_fail: bool, dependence_fail: bool) -> str:
    """Map the two frozen fold booleans to the contract vocabulary."""

    table = {
        (False, False): "NO_RESIDUAL_ATTRIBUTED_FAILURE",
        (True, False): "MARGINAL_ONLY",
        (False, True): "DEPENDENCE_ONLY",
        (True, True): "MARGINAL_AND_DEPENDENCE",
    }
    return table[(bool(marginal_fail), bool(dependence_fail))]


def aggregate_attribution_labels(labels_by_fold: Mapping[int, str]) -> str:
    """Return the conservative three-class aggregate attribution label."""

    _require(set(labels_by_fold) == set(range(FOLD_COUNT)), "aggregate requires all five folds")
    failure_labels = [labels_by_fold[fold] for fold in ORIGINAL_FAILURE_FOLDS]
    control_labels = [labels_by_fold[fold] for fold in ORIGINAL_PASS_FOLDS]
    if all(label == "MARGINAL_ONLY" for label in failure_labels) and all(
        label == "NO_RESIDUAL_ATTRIBUTED_FAILURE" for label in control_labels
    ):
        return "N3_COPULA_FAILURE_MARGINAL_SUPPORT_TAIL_DOMINANT"
    if all(label == "DEPENDENCE_ONLY" for label in failure_labels) and all(
        label == "NO_RESIDUAL_ATTRIBUTED_FAILURE" for label in control_labels
    ):
        return "N3_COPULA_FAILURE_DEPENDENCE_DOMINANT"
    return "N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED"


def _contract_payload(
    repo_root: Path,
    git: Mapping[str, Any],
    producer_contract: Mapping[str, Any],
    producer_report: Mapping[str, Any],
    report_info: Mapping[str, Any],
    recovery_info: Mapping[str, Any],
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    input_authorities = _mapping(producer_report["input_authorities"], "producer report input authorities")
    development_banks = _mapping(input_authorities["development_bank_artifacts"], "development bank artifacts")
    producer_data = _mapping(input_authorities["producer_data_authority"], "producer data authority")
    historical_boundaries = _mapping(producer_report["historical_boundaries"], "producer historical boundaries")
    sealed_confirmation = _mapping(
        historical_boundaries["sealed_confirmation"],
        "producer sealed confirmation boundary",
    )
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "classification": READY_CLASSIFICATION,
        "claim_classification": CLAIM_CLASSIFICATION,
        "new_contract": True,
        "historical_preregistration": False,
        "git": {
            **dict(git),
            "starting_committed_head": EXPECTED_STARTING_HEAD,
            "contract_script": _relative(repo_root, script_path),
            "contract_script_sha256": file_sha256(script_path),
        },
        "authorities": {
            "frozen_producer_contract": {
                "path": _relative(repo_root, repo_root / PRODUCER_CONTRACT),
                "sha256": PRODUCER_CONTRACT_SHA256,
                "classification": producer_contract["classification"],
            },
            "original_copula_execution_report": {
                "path": _relative(repo_root, repo_root / PRODUCER_REPORT),
                "sha256": PRODUCER_REPORT_SHA256,
                "classification": producer_report["classification"],
            },
            "development_data": {
                "source_manifest": dict(_mapping(producer_data["source_manifest"], "source manifest")),
                "partition_manifest": dict(_mapping(producer_data["partition_manifest"], "partition manifest")),
                "firewall_manifest": dict(_mapping(producer_data["firewall_manifest"], "firewall manifest")),
                "fit_bank": dict(_mapping(producer_data["fit_bank"], "fit bank")),
                "internal_validation_bank": dict(_mapping(producer_data["internal_validation_bank"], "internal validation bank")),
                "fit_bank_from_report": dict(development_banks),
                "q_coordinate": report_info["development_pool"]["q_coordinate"],
                "q_r2_amplitude_sha256": report_info["development_pool"]["q_r2_amplitude_sha256"],
            },
            "fold_assignment": {
                "source": "original_copula_execution_report.development_fold_identity",
                "algorithm": producer_report["development_fold_identity"]["algorithm"],
                "fold_count": FOLD_COUNT,
                "assignment_sha256": producer_report["development_fold_identity"]["assignment_sha256"],
            },
            "recovered_generated_banks": recovery_info,
            "sealed_confirmation_exclusion": {
                "path": sealed_confirmation.get("path"),
                "manifest_sha256": sealed_confirmation.get("sha256"),
                "status": sealed_confirmation.get("status"),
                "input_authority": False,
                "sample_values_loaded": False,
                "consumed": False,
            },
        },
        "scope": {
            "question": "After separating one-dimensional marginal/support effects from rank-dependence effects, which mechanism remains supported as the residual source of the frozen copula development failure?",
            "selection_scope": "DEVELOPMENT_ONLY",
            "decision_consumer": "only the next P1 development-planning task",
            "coordinate": "R2_AMPLITUDE",
            "dimension": DIMENSION,
            "labels_as_model_inputs": False,
            "optical_forward": False,
            "evaluator": False,
            "new_producer_family_selected": False,
        },
        "marginal_failure_authority": {
            "source_report": _relative(repo_root, repo_root / PRODUCER_REPORT),
            "fold_path": "fold_execution[f]",
            "component_path": "fold_execution[f].compatibility_components",
            "component_fields": list(MARGINAL_COMPONENTS),
            "reported_boolean_field": "fold_execution[f].marginal_compatible",
            "definition": "M = not marginal_compatible; marginal_compatible must equal the conjunction of the four named aggregate marginal compatibility components",
            "per_coordinate_checks_not_used": True,
            "support_tail_observers_do_not_redefine_M": True,
            "expected_original_values": report_info["folds"],
            "fail_closed_requirements": [
                "all five fold records exist exactly once",
                "each named component exists and is boolean",
                "marginal_compatible is boolean",
                "reported marginal_compatible equals the component conjunction",
            ],
        },
        "bank_role_contract": {
            "rows_per_bank": GENERATED_BANK_SIZE,
            "dimension": DIMENSION,
            "data_source": {
                "kind": "held-out development rows",
                "row_selector": "fold_id_by_source_row == f",
                "fit_rows_excluded": True,
                "per_fold_rows": HELDOUT_ROWS_PER_FOLD,
            },
            "data_vs_model": {
                "model_role": "role=data-vs-model",
                "replicates": list(range(DATA_VS_MODEL_REPLICATES)),
                "comparison": "the same self-ranked held-out DATA bank for fold f against each listed model bank",
                "keys_by_fold": {
                    str(fold): [_bank_key(fold, "data-vs-model", replicate) for replicate in range(DATA_VS_MODEL_REPLICATES)]
                    for fold in range(FOLD_COUNT)
                },
            },
            "model_vs_model": {
                "model_role": "role=model-vs-model",
                "replicates": list(range(MODEL_VS_MODEL_REPLICATES)),
                "reference_pairs": [
                    {
                        "pair_index": index,
                        "left_replicate": left,
                        "right_replicate": right,
                        "keys_by_fold": {
                            str(fold): {
                                "left": _bank_key(fold, "model-vs-model", left),
                                "right": _bank_key(fold, "model-vs-model", right),
                            }
                            for fold in range(FOLD_COUNT)
                        },
                    }
                    for index, (left, right) in enumerate(MODEL_REFERENCE_PAIRS)
                ],
                "pairing_source": {
                    "historical_report": _relative(repo_root, repo_root / PRODUCER_REPORT),
                    "producer_execution_script": _relative(repo_root, repo_root / PRODUCER_EXECUTION_SCRIPT),
                    "producer_execution_script_sha256": report_info["producer_script_sha256"],
                },
            },
            "expected_bank_identity_registry": recovery_info["bank_identities"],
        },
        "self_rank_transform": {
            "applies_to": ["held-out DATA bank", "each DATA-vs-MODEL model bank", "each MODEL-vs-MODEL bank"],
            "n": GENERATED_BANK_SIZE,
            "per_coordinate": True,
            "sorting": "stable ascending value then original row index",
            "tie_rank": "a tied block occupying one-based ranks L..U receives rank (L+U)/2",
            "u": "u_ij=(rank_ij-0.5)/n",
            "z": "z_ij=Phi^{-1}(u_ij)",
            "rank_scope": "each bank independently",
            "data_model_pooling": False,
            "fit_distribution_cdf": False,
            "epsilon_clamp": False,
            "strict_interior": True,
            "input_requirements": ["finite real q values", "q shape [200,100]", "q values nonnegative under the source contract"],
            "purpose": "equalize each bank's empirical one-dimensional marginals so the residual comparison is rank-dependence structure",
        },
        "primary_dependence_observer": {
            "statistic": "biased multivariate empirical energy distance after independent self-rank Gaussianization",
            "distance": "d(a,b)=sqrt(sum_j (a_j-b_j)^2)",
            "energy_distance": "ED(X,Y)=2*mean_{i,k} d(X_i,Y_k)-mean_{i,i'} d(X_i,X_i')-mean_{k,k'} d(Y_k,Y_k')",
            "within_pair_indices": "include diagonal pairs; diagonal distances are zero",
            "numeric_dtype": "CPU float64",
            "data_model_statistic": "D_DM[r]=ED(Z_heldout_DATA,Z_model_data_vs_model[r]) for r=0..3",
            "model_model_statistic": "D_MM[p]=ED(Z_model_model_vs_model[2p],Z_model_model_vs_model[2p+1]) for p=0..3",
            "finite_sample_reference": "fold-local D_MM values only; no cross-fold pooling",
            "reference_quantile": {
                "probability": COMPATIBILITY_Q,
                "method": "linear empirical quantile (torch.quantile default linear equivalent)",
                "definition": "sort v[0..m-1], h=(m-1)*q, interpolate v[floor(h)] and v[ceil(h)] with weight h-floor(h)",
                "m": len(MODEL_REFERENCE_PAIRS),
                "q95_specialization": "with m=4, Q95=0.15*v[2]+0.85*v[3] after ascending sort",
            },
            "per_bank_rule": "dependence_pass[r] iff D_DM[r] <= Q95(D_MM); strict greater-than is dependence_fail[r]",
            "fold_rule": "D=dependence_fail iff any of the four DATA-vs-MODEL comparisons fails; all four must pass for D=0",
            "negative_or_nonfinite_policy": "nonfinite or materially negative ED is invalid_harness; do not turn it into a scientific classification",
            "no_weighted_score": True,
        },
        "secondary_dependence_diagnostics": {
            "formal_gate_role": "descriptive only; cannot change dependence_fail or final classification",
            "spearman": {
                "rank_matrix": "average-rank matrix before Gaussianization",
                "correlation": "S_jk=sum_i((r_ij-mean_j)(r_ik-mean_k))/sqrt(sum_i(...^2)*sum_i(...^2))",
                "discrepancy": "Frobenius norm ||S_DATA-S_MODEL||_F, including zero diagonal differences",
                "emission_order": "coordinate rows/columns 0..99",
            },
            "spectral": {
                "representation": "C=diag(s)^(-1/2) * (Z_centered^T Z_centered/(n-1)) * diag(s)^(-1/2), with s_j^2=(Z_centered^T Z_centered/(n-1))_jj",
                "symmetrization": "(C+C^T)/2 before eigendecomposition",
                "eigenvalue_order": "descending real eigenvalues",
                "summaries": ["maximum eigenvalue", "median eigenvalue", "top-1 eigenvalue mass", "top-5 eigenvalue mass"],
                "comparison": "L2 norm between the two sorted eigenvalue vectors and absolute differences of named mass summaries",
                "zero_variance": "invalid_harness; no epsilon repair",
            },
        },
        "support_tail_diagnostics": {
            "role": "descriptive only; cannot redefine marginal_fail or final classification",
            "fit_source": "800 fitting rows for fold f from the original R2_AMPLITUDE development pool",
            "heldout_source": "200 held-out rows for fold f",
            "support": "fit_min_j=min(q_fit[:,j]); fit_max_j=max(q_fit[:,j])",
            "below": "x < fit_min_j",
            "above": "x > fit_max_j",
            "rate": "count / 200",
            "scale": "fit sample standard deviation s_j=sqrt(sum_i(q_fit_ij-mean_j)^2/(800-1))",
            "normalized_overshoot": "below: (fit_min_j-x)/s_j; above: (x-fit_max_j)/s_j; non-exceeding values contribute zero",
            "zero_scale": "invalid_harness if an exceedance exists or the required scale is nonfinite; no epsilon repair",
            "affected_coordinates": "support_affected_j iff below_count_j+above_count_j>0",
            "original_marginal_failure_coordinates": "indices j where original fold_execution[f].per_coordinate_marginal_q95_checks[j] is false",
            "overlap": "intersection of support_affected_coordinates and original_marginal_failure_coordinates, with counts and overlap fraction over the latter when nonempty",
            "finite_sample_caveat": "nonzero support exceedance can occur with nonzero probability under an unchanged continuous population",
        },
        "fold_attribution": {
            "booleans": {"M": "marginal_fail from the original report", "D": "dependence_fail from the primary self-rank ED observer"},
            "mapping": {
                "M=0,D=0": "NO_RESIDUAL_ATTRIBUTED_FAILURE",
                "M=1,D=0": "MARGINAL_ONLY",
                "M=0,D=1": "DEPENDENCE_ONLY",
                "M=1,D=1": "MARGINAL_AND_DEPENDENCE",
            },
            "no_learned_weights": True,
        },
        "aggregate_attribution": {
            "original_failure_folds": list(ORIGINAL_FAILURE_FOLDS),
            "original_pass_control_folds": list(ORIGINAL_PASS_FOLDS),
            "marginal_support_tail_dominant": "all original failure folds are MARGINAL_ONLY and both pass-control folds are NO_RESIDUAL_ATTRIBUTED_FAILURE",
            "dependence_dominant": "all original failure folds are DEPENDENCE_ONLY and both pass-control folds are NO_RESIDUAL_ATTRIBUTED_FAILURE",
            "mixed": "every other valid five-fold pattern, including MARGINAL_AND_DEPENDENCE, mixed mechanisms, or contradictory control-fold failure",
            "classification_vocabulary": [
                "N3_COPULA_FAILURE_MARGINAL_SUPPORT_TAIL_DOMINANT",
                "N3_COPULA_FAILURE_DEPENDENCE_DOMINANT",
                "N3_COPULA_FAILURE_MARGINAL_AND_DEPENDENCE_MIXED",
            ],
            "no_data_instability_class": True,
            "implementation_reference": {
                "fold_label_function": "scripts.research.define_n3_copula_failure_attribution_contract_v1.fold_attribution_label",
                "aggregate_function": "scripts.research.define_n3_copula_failure_attribution_contract_v1.aggregate_attribution_labels",
            },
        },
        "claim_boundary": {
            "contract_definition_claim": CLAIM_CLASSIFICATION,
            "future_marginal_dominant": "only the next P1 planning task may hold existing dependence treatment fixed and define one minimal marginal/tail intervention",
            "future_dependence_dominant": "only the next P1 planning task may hold existing empirical marginal treatment fixed and define one minimal dependence intervention",
            "future_mixed": "no producer family may be selected from attribution alone; the next planning task must improve identifiability or define one bounded competing intervention",
            "none_authorizes_automatically": ["P1 execution", "producer fit", "sealed confirmation", "novel complex OLS", "generation", "training", "formal promotion"],
            "not_established": ["population truth", "producer support", "independent confirmation", "semantic controllability", "complex-OLS validity", "generation readiness"],
        },
        "execution_boundary": {
            "future_consumer": "RESOLVE_N3_COPULA_FAILURE_MARGINAL_VS_DEPENDENCE_V1",
            "future_consumer_may_load": ["original development q rows through frozen lineage", "FROZEN_N3_COPULA_GENERATED_BANKS_V1"],
            "future_consumer_must_not_load": ["N3 sealed confirmation source"],
            "metrics_computed_by_this_definition_consumer": False,
            "recovered_tensor_loaded_as_tensor_by_this_definition_consumer": False,
            "producer_sampling": False,
            "producer_refit": False,
            "rank_reselection": False,
            "new_family_selection": False,
            "cuda": False,
            "optimizer": False,
            "backward": False,
            "training": False,
        },
        "governance": {
            "selection_scope": "DEVELOPMENT_ONLY",
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "sealed_confirmation_status": "RESERVED_NOT_MATERIALIZED",
            "sealed_confirmation_consumed": False,
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
            artifact_kind="N3 copula failure attribution contract",
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
    allowed_dirty = {Path(__file__).resolve().relative_to(repo_root.resolve()).as_posix()}
    unexpected_dirty = [path for path in git["dirty_paths"] if path not in allowed_dirty]
    _require(not unexpected_dirty, f"unexpected dirty paths before contract definition: {unexpected_dirty}")

    producer_contract = _read_json_exact(repo_root, PRODUCER_CONTRACT, PRODUCER_CONTRACT_SHA256, "frozen producer contract")
    producer_report = _read_json_exact(repo_root, PRODUCER_REPORT, PRODUCER_REPORT_SHA256, "original copula execution report")
    recovery_manifest = _read_json_exact(repo_root, RECOVERY_MANIFEST, RECOVERY_MANIFEST_SHA256, "recovery manifest")
    _validate_producer_contract(producer_contract)
    report_info = _validate_producer_report(producer_report, producer_contract)
    recovery_info = _validate_recovery_manifest(recovery_manifest, repo_root, producer_report)
    payload = _contract_payload(
        repo_root,
        git,
        producer_contract,
        producer_report,
        report_info,
        recovery_info,
    )
    report_path = _write_contract(repo_root, output_dir, payload)
    return {
        "classification": READY_CLASSIFICATION,
        "contract": _relative(repo_root, report_path),
        "contract_sha256": file_sha256(report_path),
        "git_head": git["head"],
        "worktree_dirty_before_artifact": git["dirty_paths"],
        "attribution_metrics_computed": False,
        "recovered_tensor_loaded_as_tensor": False,
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
