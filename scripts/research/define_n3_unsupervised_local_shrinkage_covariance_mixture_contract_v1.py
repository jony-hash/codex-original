"""Register, but do not execute, the N3 local-covariance mixture contract.

The contract preserves the previously rejected unsupervised local-location
partition and adds only component-local analytic shrinkage covariance.  This
definition consumer verifies lineage and identity, but never fits real q data,
selects K, samples a producer, loads sealed confirmation values, or invokes an
optical/evaluator path.
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

import torch

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256
from scripts.research import (
    compare_n3_minimal_competing_producer_baselines_v1 as comparison,
    define_n3_unsupervised_local_location_mixture_producer_contract_v1 as previous_contract,
)


TASK_ID = "DEFINE_N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "n3_unsupervised_local_shrinkage_covariance_mixture_producer_contract.v1"
CLAIM_CLASSIFICATION = "N3_LOCAL_COVARIANCE_HETEROGENEITY_MODEL_ASSUMPTION_DEFINED"
EXPECTED_READY_CLASSIFICATION = "DEFINE_N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1_READY"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "1dd7e2fd75fcb5cac12b4c1374dcf3f252cd88a0"

PREVIOUS_CONTRACT = Path(
    "outputs/20260908_v2/runs/"
    "n3_unsupervised_local_location_mixture_producer_contract_v1_attempt_01/"
    "n3_unsupervised_local_location_mixture_producer_contract.v1.json"
)
PREVIOUS_CONTRACT_SHA256 = "38cb0c549d8f943123f0ab3304adf5a01e9b3b5edb63daa384c7f4c7bac20449"
PREVIOUS_SELECTION_REPORT = Path(
    "outputs/20260908_v2/runs/"
    "n3_local_location_mixture_selection_v1_attempt_01/"
    "n3_local_location_mixture_selection_report.v1.json"
)
PREVIOUS_SELECTION_REPORT_SHA256 = "5064ee87cbc9846f0194ba0afcea4687647ceefb454d2abb20b8b56d632a9080"
SEALED_CONFIRMATION_MANIFEST = comparison.SEALED_CONFIRMATION_MANIFEST
SEALED_CONFIRMATION_MANIFEST_SHA256 = comparison.EXPECTED_SEALED_CONFIRMATION_MANIFEST_SHA256

CLASS_COUNT = comparison.CLASS_COUNT
DIMENSION = comparison.DIMENSION
POOL_COUNT = comparison.POOL_COUNT
FOLD_COUNT = comparison.FOLD_COUNT
FIT_COUNT_PER_FOLD = comparison.FIT_COUNT_PER_FOLD
HELDOUT_COUNT_PER_FOLD = comparison.HELDOUT_COUNT_PER_FOLD
K_DOMAIN = (2, 4, 8)

CONTRACT_SALT = "N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1"
RNG_NAMESPACE = "N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1_CPU"
TARGET_ACCEPTED = comparison.TARGET_ACCEPTED
PROPOSAL_BUDGET = comparison.PROPOSAL_BUDGET
PROPOSAL_BATCH_SIZE = comparison.PROPOSAL_BATCH_SIZE
DATA_VS_MODEL_REPLICATES = comparison.DATA_VS_MODEL_REPLICATES
MODEL_VS_MODEL_REPLICATES = comparison.MODEL_VS_MODEL_REPLICATES
COMPATIBILITY_Q = comparison.Q95
LOCAL_COVARIANCE_COMPONENT_INSUFFICIENT = "LOCAL_COVARIANCE_COMPONENT_INSUFFICIENT"

REPORT_NAME = "n3_unsupervised_local_shrinkage_covariance_mixture_producer_contract.v1.json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    return value


def _relative(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def _read_json_exact(repo_root: Path, path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
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
        "worktree_clean": not status,
    }


def hard_assignment_from_responsibilities(responsibilities: torch.Tensor) -> torch.Tensor:
    """Return canonical-index argmax assignments; ties choose lowest index."""

    _require(responsibilities.ndim == 2 and responsibilities.shape[1] in K_DOMAIN, "responsibility shape is invalid")
    _require(bool(torch.isfinite(responsibilities).all()), "responsibilities are non-finite")
    _require(bool(responsibilities.ge(0.0).all()), "responsibilities are negative")
    _require(bool(torch.isclose(responsibilities.sum(dim=1), torch.ones(responsibilities.shape[0], dtype=responsibilities.dtype), atol=1.0e-12, rtol=0.0).all()), "responsibilities are not normalized")
    return torch.argmax(responsibilities, dim=1).to(dtype=torch.int64)


def component_insufficiency_status(assigned_count: int) -> str:
    """Return the frozen Stage-2 status for an assigned component count."""

    _require(assigned_count >= 0, "assigned count is negative")
    return LOCAL_COVARIANCE_COMPONENT_INSUFFICIENT if assigned_count < 2 else "LOCAL_COVARIANCE_COMPONENT_ELIGIBLE"


def _validate_previous_lineage(
    previous: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    _require(previous.get("schema") == "n3_unsupervised_local_location_mixture_producer_contract.v1", "previous contract schema changed")
    _require(
        previous.get("classification") == "DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1_READY",
        "previous local-location contract is not ready",
    )
    _require(
        previous.get("claim_classification") == "N3_LOCAL_LOCATION_MIXTURE_MODEL_ASSUMPTION_DEFINED",
        "previous local-location claim changed",
    )
    _require(previous.get("component_count_domain", {}).get("candidate_K") == list(K_DOMAIN), "previous K domain changed")
    previous_development = _mapping(previous.get("development_identity"), "previous development identity")
    previous_fold = _mapping(previous_development.get("fold_assignment"), "previous fold assignment")
    _require(
        previous_fold.get("assignment_sha256") == previous_contract._expected_fold_assignment_sha256(),
        "previous fold identity changed",
    )
    _require(previous.get("sealed_confirmation_firewall", {}).get("status") == "RESERVED_NOT_MATERIALIZED", "previous sealed status changed")

    _require(selection.get("schema") == "n3_local_location_mixture_selection_report.v1", "previous selection schema changed")
    _require(selection.get("classification") == "N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED", "previous local-location result changed")
    selection_result = _mapping(selection.get("selection"), "previous selection result")
    _require(selection_result.get("K_selected") is None, "previous local-location K was selected")
    _require(selection_result.get("compatible_K") == [], "previous local-location compatible K changed")
    _require(selection.get("contract", {}).get("sha256") == PREVIOUS_CONTRACT_SHA256, "previous selection contract identity changed")
    candidates = selection.get("candidate_K_results")
    _require(isinstance(candidates, list) and [item.get("K") for item in candidates] == list(K_DOMAIN), "previous candidate K results changed")
    _require(all(item.get("operationally_valid") is True for item in candidates), "previous local-location execution was not fully operational")
    _require(all(item.get("compatible") is False and item.get("compatible_fold_count") == 0 for item in candidates), "previous local-location compatibility changed")
    _require(selection.get("sealed_confirmation", {}).get("consumed") is False, "previous selection consumed sealed confirmation")


def _load_development_identity(repo_root: Path, previous: Mapping[str, Any]) -> dict[str, Any]:
    source_indices, labels, identity = previous_contract._load_development_identity(repo_root)
    previous_identity = _mapping(previous.get("development_identity"), "previous development identity")
    _require(identity == previous_identity, "development identity changed from previous contract")
    _require(torch.unique(source_indices).numel() == POOL_COUNT, "development source indices are not unique")
    _require(
        torch.equal(
            torch.bincount(labels, minlength=CLASS_COUNT),
            torch.full((CLASS_COUNT,), comparison.ROWS_PER_CLASS, dtype=torch.int64),
        ),
        "development class balance changed",
    )
    return dict(identity)


def _contract_payload(
    authorities: Mapping[str, Any],
    development: Mapping[str, Any],
    git: Mapping[str, Any],
    script_path: Path,
) -> dict[str, Any]:
    return {
        "schema": CONTRACT_SCHEMA,
        "version": 1,
        "task_id": TASK_ID,
        "gate": GATE,
        "status": "valid_completed",
        "classification": EXPECTED_READY_CLASSIFICATION,
        "claim_classification": CLAIM_CLASSIFICATION,
        "git": {
            **git,
            "contract_script": script_path.as_posix(),
            "starting_committed_head": EXPECTED_STARTING_HEAD,
        },
        "authorities": authorities,
        "development_identity": development,
        "previous_local_location_lineage": {
            "contract_path": _relative(Path(__file__).resolve().parents[2], PREVIOUS_CONTRACT),
            "contract_sha256": PREVIOUS_CONTRACT_SHA256,
            "contract_classification": "DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1_READY",
            "selection_report_path": _relative(Path(__file__).resolve().parents[2], PREVIOUS_SELECTION_REPORT),
            "selection_report_sha256": PREVIOUS_SELECTION_REPORT_SHA256,
            "selection_classification": "N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED",
            "selected_K": None,
            "fold_assignment_sha256": previous_contract._expected_fold_assignment_sha256(),
        },
        "family": {
            "name": "UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_AMPLITUDE_V1",
            "mechanism_under_test": "UNSUPERVISED_LOCAL_COVARIANCE_HETEROGENEITY",
            "coordinate": "R2_AMPLITUDE",
            "dimension": DIMENSION,
            "preconditioning_density": "p_pre(q)=sum_k pi_k Normal(q;mu_k,Sigma_k)",
            "legal_density": "p(q | q>=0 elementwise)",
            "weights": "pi_k>=0 and sum_k pi_k=1",
            "component_means": "mu_k in R^100",
            "component_covariance": "Sigma_k is component-specific, positive-definite and analytic-shrinkage regularized",
            "labels_as_model_inputs": False,
            "local_latent_rank": False,
            "factor_analyzer": False,
            "neural_network": False,
            "nonlinear_flow": False,
            "component_or_physical_interpretation": False,
        },
        "component_count_domain": {
            "candidate_K": list(K_DOMAIN),
            "selection_rule": "select smallest K satisfying the frozen development compatibility contract across all folds",
            "post_result_K_addition": False,
            "post_result_reassignment_change": False,
            "post_result_covariance_estimator_change": False,
            "post_result_threshold_change": False,
        },
        "stage1_local_partition_contract": {
            "purpose": "produce unsupervised local partition only",
            "parent_contract_path": _relative(Path(__file__).resolve().parents[2], PREVIOUS_CONTRACT),
            "parent_contract_sha256": PREVIOUS_CONTRACT_SHA256,
            "parent_contract_classification": "DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1_READY",
            "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
            "coordinate": "R2_AMPLITUDE",
            "shared_covariance": "reuse the parent contract's fixed analytic-shrinkage Sigma_shared exactly",
            "initialization": "reuse the parent contract's label-free hash tie-break and whitened farthest-point initialization exactly",
            "fit": "reuse the parent contract's deterministic CPU float64 EM, convergence, monotonicity and finite-state rules exactly",
            "component_order": "reuse the parent contract's lexicographic whitened-center canonicalization exactly",
            "collapse": "reuse the parent contract's candidate/fold scientific-failure semantics exactly",
            "responsibilities": "after Stage-1 convergence, compute a final stabilized E-step under the final canonical parameters",
            "labels_as_inputs": False,
            "heldout_parameter_leakage": False,
            "no_second_joint_EM": True,
        },
        "hard_assignment_contract": {
            "rule": "a_i=argmax_k r_ik over final Stage-1 responsibilities",
            "tie_rule": "lowest canonical component index wins",
            "canonical_order_dependency": "existing Stage-1 component canonicalization and tie ordering",
            "labels_as_inputs": False,
            "fit_rows_only": True,
            "assignment_frozen_before_local_covariance": True,
            "reassignment_after_local_covariance": False,
        },
        "stage2_local_covariance_contract": {
            "input_rows": "only Stage-1 fitting rows assigned to component k",
            "assigned_count": "n_k",
            "insufficient_count_predicate": "n_k < 2",
            "insufficient_count_classification": LOCAL_COVARIANCE_COMPONENT_INSUFFICIENT,
            "insufficient_count_action": "candidate/fold scientific failure",
            "insufficient_count_no_merge": True,
            "insufficient_count_no_restart": True,
            "insufficient_count_no_K_change": True,
            "local_mean": "mu_k_local=mean(q_i for a_i=k)",
            "centered_rows": "x_i=q_i-mu_k_local",
            "unbiased_covariance": "S_k=(X_k^T X_k)/(n_k-1)",
            "target": "tau_k*I, tau_k=trace(S_k)/d",
            "analytic_shrinkage": "beta_k=n_k^-2 sum_i ||x_i x_i^T-S_k||_F^2; delta_k=||S_k-tau_k I||_F^2; alpha_k=clip(beta_k/delta_k,0,1)",
            "zero_delta": "alpha_k=0",
            "regularized_covariance": "Sigma_k=(1-alpha_k)S_k+alpha_k*tau_k*I",
            "symmetry_policy": "replace Sigma_k by (Sigma_k+Sigma_k^T)/2 exactly before positive-definite checks",
            "positive_definite_policy": "finite Cholesky succeeds and all eigenvalues are finite and strictly positive",
            "finite_policy": "mu_k_local, S_k, alpha_k, Sigma_k and Cholesky must be finite",
            "weight": "pi_k=n_k/800",
            "weight_normalization": "sum_k n_k=800 and pi_k is finite with sum_k pi_k=1",
            "heldout_parameter_leakage": False,
            "second_EM": False,
            "component_covariance_specific": True,
        },
        "sampling_contract": {
            "law": "sample component categorical(pi), then q=mu_k_local+L_k*z with Sigma_k=L_k L_k^T and z~Normal(0,I)",
            "legal_state": "finite(q) and q>=0 elementwise",
            "clamp": False,
            "abs": False,
            "relu": False,
            "softplus": False,
            "epsilon_repair": False,
            "samplewise_normalization": False,
            "target_accepted": TARGET_ACCEPTED,
            "proposal_budget": PROPOSAL_BUDGET,
            "proposal_batch_size": PROPOSAL_BATCH_SIZE,
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "rng_namespace": RNG_NAMESPACE,
            "rng_device": "cpu",
            "rng_generator": "torch.Generator(device='cpu')",
            "seed_substreams": ["K", "fold", "data-vs-model replicate", "model-vs-model replicate"],
            "budget_adaptation": False,
        },
        "development_evaluation_contract": {
            "fold_assignment": "reuse frozen previous N3 assignment exactly",
            "fit_rows": FIT_COUNT_PER_FOLD,
            "heldout_rows": HELDOUT_COUNT_PER_FOLD,
            "labels_as_model_inputs": False,
            "primary_metric": "validated_biased_empirical_energy_distance_euclidean_q_space",
            "primary_metric_implementation": "scripts.research.validate_n3_amplitude_producer_baseline._energy_distance",
            "secondary_guardrails": [
                "covariance_relative_frobenius",
                "norm_energy_distance",
                "squared_norm_energy_distance",
                "positivity_acceptance_fraction",
                "marginal_mean_l2",
                "marginal_std_l2",
            ],
            "model_vs_model_role": "finite-sample fluctuation reference only",
            "optical_forward": False,
            "evaluator": False,
            "sealed_confirmation": False,
        },
        "compatibility_contract": {
            "rule": "fixed-replicate mean DATA_VS_MODEL <= q95 MODEL_VS_MODEL for primary plus covariance/norm guardrails",
            "q": COMPATIBILITY_Q,
            "absolute_metric_thresholds": False,
            "own_model_reference_required": True,
            "grossly_outside_reference_rejects": True,
            "all_folds_required": True,
            "less_bad_than_failed_prior_family_is_sufficient": False,
        },
        "complexity_selection_contract": {
            "candidate_orders": list(K_DOMAIN),
            "order": "ascending K",
            "selection": "smallest globally compatible K",
            "none_compatible": "N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_NOT_SUPPORTED",
            "mixed_without_deterministic_K": "N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_INCONCLUSIVE",
            "exactly_one_selected": "N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_MODEL_ORDER_SELECTED",
            "label_based_selection": False,
            "image_based_selection": False,
            "generator_based_selection": False,
            "sealed_confirmation_based_selection": False,
        },
        "one_shot_semantics": {
            "future_execution_count": 1,
            "family_frozen_before_execution": True,
            "K_domain_frozen_before_execution": True,
            "stage1_partition_frozen_before_execution": True,
            "hard_assignment_frozen_before_execution": True,
            "local_covariance_estimator_frozen_before_execution": True,
            "insufficiency_semantics_frozen_before_execution": True,
            "no_second_EM": True,
            "no_adaptive_candidate_addition": True,
            "no_adaptive_reassignment": True,
            "no_adaptive_budget_increase": True,
            "no_confirmation_consumption": True,
        },
        "sealed_confirmation_firewall": {
            "manifest_path": _relative(Path(__file__).resolve().parents[2], SEALED_CONFIRMATION_MANIFEST),
            "manifest_sha256": SEALED_CONFIRMATION_MANIFEST_SHA256,
            "status": "RESERVED_NOT_MATERIALIZED",
            "sample_values_loaded": False,
            "confirmation_evaluation_authorized": False,
            "selection_source_exclusion": True,
        },
        "claim_boundary": {
            "allowed": [CLAIM_CLASSIFICATION],
            "not_allowed": [
                "local covariance scientifically supported",
                "K selected",
                "producer fitted or frozen",
                "true OLS mixture topology",
                "semantic components identified",
                "sealed confirmation consumed",
                "full complex OLS validity",
                "novel OLS supported",
                "generation readiness",
            ],
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "real_n3_selection_executed": False,
            "candidate_selected": False,
        },
        "not_executed": [
            "real N3 local-covariance mixture fitting",
            "real N3 selection",
            "K selection",
            "final all-1000-row producer fit",
            "sealed confirmation loading or materialization",
            "label-conditioned mixture",
            "joint unrestricted full-covariance EM",
            "MFA/local factor model",
            "ICA",
            "normalizing flow",
            "phase-source work",
            "novel complex OLS",
            "generator",
            "evaluator",
            "CUDA",
            "optimizer",
            "backward",
            "training",
        ],
    }


def _write_contract(repo_root: Path, output_dir: Path, payload: Mapping[str, Any]) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse write-once output: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        atomic_write_new_json(temporary / REPORT_NAME, payload, artifact_kind="N3 local-covariance contract")
        if output_dir.exists():
            raise FileExistsError(f"write-once output appeared during execution: {output_dir}")
        os.replace(temporary, output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output_dir / REPORT_NAME


def execute(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    git = _git_state(repo_root)
    _require(git["branch"] == EXPECTED_BRANCH, "current branch is not the N3 research authority branch")
    _require(git["head"] == EXPECTED_STARTING_HEAD, "contract must start from the current clean N3 authority commit")
    previous = _read_json_exact(repo_root, PREVIOUS_CONTRACT, PREVIOUS_CONTRACT_SHA256, "previous local-location contract")
    selection = _read_json_exact(repo_root, PREVIOUS_SELECTION_REPORT, PREVIOUS_SELECTION_REPORT_SHA256, "previous local-location selection")
    _validate_previous_lineage(previous, selection)
    authorities = comparison._verify_authorities(repo_root)
    boundaries = comparison._verify_historical_boundaries(repo_root)
    development = _load_development_identity(repo_root, previous)
    _require(boundaries["sealed_confirmation"]["status"] == "RESERVED_NOT_MATERIALIZED", "sealed confirmation status changed")
    script_path = Path(__file__).resolve()
    payload = _contract_payload(
        {
            "producer_data_authority": authorities,
            "previous_local_location": {
                "contract_path": _relative(repo_root, repo_root / PREVIOUS_CONTRACT),
                "contract_sha256": PREVIOUS_CONTRACT_SHA256,
                "selection_report_path": _relative(repo_root, repo_root / PREVIOUS_SELECTION_REPORT),
                "selection_report_sha256": PREVIOUS_SELECTION_REPORT_SHA256,
                "selection_classification": selection["classification"],
                "selected_K": None,
                "fold_assignment_sha256": development["fold_assignment"]["assignment_sha256"],
            },
            "historical_boundaries": boundaries,
        },
        development,
        git,
        script_path,
    )
    report_path = _write_contract(repo_root, output_dir, payload)
    return {
        "classification": payload["classification"],
        "claim_classification": payload["claim_classification"],
        "contract": _relative(repo_root, report_path),
        "contract_sha256": file_sha256(report_path),
        "git_head": git["head"],
        "real_n3_selection_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/20260908_v2/runs/"
            "n3_unsupervised_local_shrinkage_covariance_mixture_producer_contract_v1_attempt_01"
        ),
        help="fresh write-once contract output directory",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
