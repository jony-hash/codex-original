"""Register, but do not fit, the N3 shared-covariance local-mixture contract.

The consumer verifies the existing N3 development authority and frozen five-fold
assignment, then publishes a write-once machine-readable contract.  It never
fits a mixture, samples a producer, loads the sealed confirmation source, or
invokes an optical/evaluator path.
"""

from __future__ import annotations

import argparse
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
    file_sha256,
    tensor_sha256,
)
from scripts.research import compare_n3_minimal_competing_producer_baselines_v1 as comparison


TASK_ID = "DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "n3_unsupervised_local_location_mixture_producer_contract.v1"
CLAIM_CLASSIFICATION = "N3_LOCAL_LOCATION_MIXTURE_MODEL_ASSUMPTION_DEFINED"
EXPECTED_READY_CLASSIFICATION = "DEFINE_N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1_READY"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_STARTING_HEAD = "32377716f98b9d165c09b62a562931f5a0d8325b"

COMPARISON_REPORT = Path(
    "outputs/20260908_v2/runs/"
    "n3_minimal_competing_producer_baselines_v1_attempt_01/"
    "n3_minimal_competing_producer_baselines_report.v1.json"
)
COMPARISON_REPORT_SHA256 = (
    "af7611cfbca211e666d7773e81b230a1b380b91583817c3c4ab80925f2a16440"
)
SEALED_CONFIRMATION_MANIFEST = comparison.SEALED_CONFIRMATION_MANIFEST
SEALED_CONFIRMATION_MANIFEST_SHA256 = comparison.EXPECTED_SEALED_CONFIRMATION_MANIFEST_SHA256

CLASS_COUNT = comparison.CLASS_COUNT
DIMENSION = comparison.DIMENSION
POOL_COUNT = comparison.POOL_COUNT
FOLD_COUNT = comparison.FOLD_COUNT
FIT_COUNT_PER_FOLD = comparison.FIT_COUNT_PER_FOLD
HELDOUT_COUNT_PER_FOLD = comparison.HELDOUT_COUNT_PER_FOLD
K_DOMAIN = (2, 4, 8)

CONTRACT_SALT = "N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1"
RNG_NAMESPACE = "N3_UNSUPERVISED_LOCAL_LOCATION_MIXTURE_PRODUCER_CONTRACT_V1_CPU"
MAX_EM_ITERATIONS = 200
EM_LOG_LIKELIHOOD_TOLERANCE_PER_ROW = 1.0e-10
EM_MONOTONICITY_TOLERANCE_PER_ROW = 1.0e-10
COMPONENT_MASS_FLOOR = 1.0e-8
TARGET_ACCEPTED = comparison.TARGET_ACCEPTED
PROPOSAL_BUDGET = comparison.PROPOSAL_BUDGET
PROPOSAL_BATCH_SIZE = comparison.PROPOSAL_BATCH_SIZE
DATA_VS_MODEL_REPLICATES = comparison.DATA_VS_MODEL_REPLICATES
MODEL_VS_MODEL_REPLICATES = comparison.MODEL_VS_MODEL_REPLICATES
COMPATIBILITY_Q = comparison.Q95

REPORT_NAME = "n3_unsupervised_local_location_mixture_producer_contract.v1.json"


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


def _validate_comparison_parent(report: Mapping[str, Any]) -> None:
    _require(report.get("schema") == "n3_minimal_competing_producer_baselines_report.v1", "comparison schema changed")
    _require(
        report.get("classification") == "N3_MINIMAL_PRODUCER_BASELINES_NOT_SUPPORTED",
        "comparison classification changed",
    )
    selection = _mapping(report.get("selection"), "comparison selection")
    _require(selection.get("selected_candidate") is None, "a comparison candidate is already selected")
    fold_identity = _mapping(report.get("development_fold_identity"), "comparison fold identity")
    _require(
        fold_identity.get("assignment_sha256") == _expected_fold_assignment_sha256(),
        "comparison fold assignment changed",
    )
    _require(report.get("sealed_confirmation", {}).get("consumed") is False, "comparison consumed sealed confirmation")


def _expected_fold_assignment_sha256() -> str:
    return "90277c16a0d96b895e218264f35a822564b3a46570cab73a1a1e823f063e9e32"


def _load_development_identity(repo_root: Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    fit = comparison._load_bank(
        repo_root,
        comparison.FIT_BANK,
        comparison.EXPECTED_FIT_BANK_SHA256,
        "fit bank",
    )
    internal = comparison._load_bank(
        repo_root,
        comparison.INTERNAL_VALIDATION_BANK,
        comparison.EXPECTED_INTERNAL_VALIDATION_BANK_SHA256,
        "internal validation bank",
    )
    positions = torch.cat([fit["source_row_positions"], internal["source_row_positions"]])
    source_indices = torch.cat([fit["source_indices"], internal["source_indices"]])
    labels = torch.cat([fit["labels"], internal["labels"]])
    order = torch.argsort(positions, stable=True)
    positions = positions.index_select(0, order).contiguous()
    source_indices = source_indices.index_select(0, order).contiguous()
    labels = labels.index_select(0, order).contiguous()
    _require(torch.equal(positions, torch.arange(POOL_COUNT, dtype=torch.int64)), "development rows are incomplete")
    _require(tensor_sha256(source_indices) == comparison.EXPECTED_SOURCE_INDEX_SHA256, "source index identity changed")
    _require(tensor_sha256(labels) == comparison.EXPECTED_SOURCE_LABEL_SHA256, "source label identity changed")
    _require(
        torch.equal(
            torch.bincount(labels, minlength=CLASS_COUNT),
            torch.full((CLASS_COUNT,), comparison.ROWS_PER_CLASS, dtype=torch.int64),
        ),
        "development class balance changed",
    )
    fold_ids, fold_identity = comparison._assign_folds(source_indices, labels)
    _require(fold_identity["assignment_sha256"] == _expected_fold_assignment_sha256(), "frozen fold identity changed")
    _require(torch.unique(fold_ids).numel() == FOLD_COUNT, "fold count changed")
    return source_indices, labels, {
        "count": POOL_COUNT,
        "per_class": comparison.ROWS_PER_CLASS,
        "source_indices_sha256": tensor_sha256(source_indices),
        "labels_sha256": tensor_sha256(labels),
        "source_row_positions_sha256": tensor_sha256(positions),
        "fold_assignment": fold_identity,
        "q_values_loaded_for_identity_only": False,
        "real_mixture_fit_executed": False,
    }


def _initialization_digest(fold: int, source_index: int) -> str:
    material = f"{CONTRACT_SALT}|fold={fold}|source_index={source_index}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def initial_center_positions(whitened_fit: torch.Tensor, source_indices: torch.Tensor, *, fold: int, k: int) -> list[int]:
    """Return deterministic farthest-point row positions for synthetic/unit checks.

    This helper is contract support only.  The contract consumer never calls it
    with the real N3 q matrix.
    """

    _require(k in K_DOMAIN, "K is outside the frozen domain")
    _require(whitened_fit.ndim == 2 and whitened_fit.shape[0] >= k, "synthetic fit coordinate shape is invalid")
    _require(whitened_fit.shape[1] == DIMENSION, "synthetic fit dimension changed")
    _require(source_indices.shape == (whitened_fit.shape[0],), "synthetic source identity shape is invalid")
    _require(bool(torch.isfinite(whitened_fit).all()), "synthetic fit coordinate is non-finite")
    available = list(range(whitened_fit.shape[0]))
    first = min(available, key=lambda position: (_initialization_digest(fold, int(source_indices[position])), int(source_indices[position])))
    selected = [first]
    while len(selected) < k:
        candidates: list[tuple[float, str, int, int]] = []
        for position in available:
            if position in selected:
                continue
            distances = whitened_fit[position].sub(whitened_fit[selected]).square().sum(dim=1)
            minimum_distance = float(distances.min())
            candidates.append(
                (
                    -minimum_distance,
                    _initialization_digest(fold, int(source_indices[position])),
                    int(source_indices[position]),
                    position,
                )
            )
        _require(candidates, "farthest-point traversal exhausted")
        selected.append(min(candidates)[3])
    return selected


def canonical_component_order(whitened_centers: torch.Tensor) -> list[int]:
    """Return the frozen lexicographic component canonicalization order."""

    _require(whitened_centers.ndim == 2 and whitened_centers.shape[1] == DIMENSION, "component center shape is invalid")
    return sorted(
        range(whitened_centers.shape[0]),
        key=lambda position: (tuple(float(value) for value in whitened_centers[position].tolist()), position),
    )


def _contract_payload(authorities: Mapping[str, Any], development: Mapping[str, Any], git: Mapping[str, Any], script_path: Path) -> dict[str, Any]:
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
        },
        "authorities": authorities,
        "development_identity": development,
        "family": {
            "name": "UNSUPERVISED_LOCAL_LOCATION_MIXTURE_AMPLITUDE_V1",
            "mechanism_under_test": "MULTIPLE_UNSUPERVISED_LATENT_LOCATIONS",
            "coordinate": "R2_AMPLITUDE",
            "dimension": DIMENSION,
            "preconditioning_density": "p_pre(q)=sum_k pi_k Normal(q;mu_k,Sigma_shared)",
            "legal_density": "p(q | q>=0 elementwise)",
            "weights": "pi_k>=0 and sum_k pi_k=1",
            "component_means": "mu_k in R^100",
            "shared_covariance": "one shared positive-definite Sigma_shared for all components",
            "labels_as_model_inputs": False,
            "component_specific_covariance": False,
            "component_specific_rank": False,
            "neural_network": False,
            "nonlinear_flow": False,
            "class_or_physical_component_interpretation": False,
        },
        "component_count_domain": {
            "candidate_K": list(K_DOMAIN),
            "selection_rule": "select smallest K satisfying the frozen development compatibility contract across all folds",
            "post_result_K_addition": False,
            "post_result_initialization_change": False,
            "post_result_threshold_change": False,
        },
        "shared_covariance_contract": {
            "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "heldout_rows_per_fold": HELDOUT_COUNT_PER_FOLD,
            "mean": "mu_global=mean(q_fit)",
            "centered_rows": "x_i=q_i-mu_global",
            "unbiased_covariance": "S=(X^T X)/(n-1)",
            "target": "tau*I, tau=trace(S)/d",
            "analytic_shrinkage": "beta=n^-2 sum_i ||x_i x_i^T-S||_F^2; delta=||S-tau I||_F^2; alpha=clip(beta/delta,0,1)",
            "zero_delta": "alpha=0",
            "Sigma_shared": "(1-alpha)S+alpha*tau*I",
            "positive_definite_requirement": "finite Cholesky succeeds; otherwise candidate/fold invalid",
            "parameter_source": "800 fitting rows only",
            "heldout_covariance_leakage": False,
            "shared_covariance_fixed_during_EM": True,
            "parent_implementation": "compare_n3_minimal_competing_producer_baselines_v1._fit_shrinkage",
            "parent_report_sha256": COMPARISON_REPORT_SHA256,
        },
        "initialization_contract": {
            "coordinate": "y_i=L^-1(q_i-mu_global), Sigma_shared=L L^T with lower-triangular L",
            "first_center": "row with lexicographically smallest SHA256(CONTRACT_SALT|fold|source_index), independent of label",
            "subsequent_centers": "deterministic farthest-point traversal maximizing minimum squared Euclidean distance in y coordinates",
            "distance_tie_break": "initialization digest, then source index ascending",
            "center_space": "selected fitting rows in shared-whitened coordinate, mapped back to q-space",
            "weights": "pi_k=1/K",
            "random_restarts": False,
            "label_input": False,
            "salt": CONTRACT_SALT,
        },
        "fit_contract": {
            "algorithm": "deterministic CPU float64 EM with fixed Sigma_shared",
            "fitted_parameters_only": ["pi_k", "mu_k"],
            "log_density": "log pi_k - 0.5*||y_i-c_k||^2 - logdet(L) - d/2*log(2*pi)",
            "responsibilities": "softmax over stabilized component log densities",
            "M_step_weights": "N_k=sum_i r_ik; pi_k=N_k/n",
            "M_step_centers": "c_k=sum_i r_ik*y_i/N_k; mu_k=mu_global+L*c_k",
            "max_iterations": MAX_EM_ITERATIONS,
            "convergence_statistic": "absolute change in mean (per-row) pre-conditioning log likelihood",
            "convergence_tolerance": EM_LOG_LIKELIHOOD_TOLERANCE_PER_ROW,
            "monotonicity_rule": "new mean log likelihood may not be lower than prior by more than the frozen monotonicity tolerance",
            "monotonicity_tolerance_per_row": EM_MONOTONICITY_TOLERANCE_PER_ROW,
            "finite_state_checks": ["log densities", "responsibilities", "N_k", "pi_k", "mu_k", "log likelihood"],
            "component_canonicalization": "sort whitened centers lexicographically; exact ties retain prior component index",
            "convergence_required": True,
            "preconditioning_likelihood": True,
            "positivity_conditioning_in_fit": False,
            "ambient_rng": False,
        },
        "component_collapse_contract": {
            "mass_floor": COMPONENT_MASS_FLOOR,
            "collapse_predicate": "nonfinite N_k/pi_k or N_k/n <= mass_floor",
            "action": "candidate/fold scientific failure",
            "reinitialize_component": False,
            "change_K": False,
            "random_restart": False,
            "continue_after_collapse": False,
        },
        "sampling_contract": {
            "law": "sample component categorical(pi), then g=mu_k+L*z with z~Normal(0,I)",
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
            "fold_assignment": "reuse frozen COMPARE_N3_MINIMAL_COMPETING_PRODUCER_BASELINES_V1 assignment exactly",
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
            "less_bad_than_failed_baseline_is_sufficient": False,
        },
        "complexity_selection_contract": {
            "candidate_orders": list(K_DOMAIN),
            "order": "ascending K",
            "selection": "smallest globally compatible K",
            "multiple_compatible": "smallest K wins",
            "none_compatible": "N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED",
            "mixed_without_deterministic_K": "N3_LOCAL_LOCATION_MIXTURE_INCONCLUSIVE",
            "exactly_one_selected": "N3_LOCAL_LOCATION_MIXTURE_MODEL_ORDER_SELECTED",
            "label_based_selection": False,
            "image_based_selection": False,
            "generator_based_selection": False,
            "sealed_confirmation_based_selection": False,
        },
        "one_shot_semantics": {
            "future_execution_count": 1,
            "family_frozen_before_execution": True,
            "K_domain_frozen_before_execution": True,
            "initialization_frozen_before_execution": True,
            "fit_semantics_frozen_before_execution": True,
            "compatibility_frozen_before_execution": True,
            "no_adaptive_candidate_addition": True,
            "no_adaptive_budget_increase": True,
            "no_confirmation_consumption": True,
        },
        "sealed_confirmation_firewall": {
            "manifest_path": _relative(Path(__file__).resolve().parents[2], SEALED_CONFIRMATION_MANIFEST),
            "manifest_sha256": SEALED_CONFIRMATION_MANIFEST_SHA256,
            "status": "RESERVED_NOT_MATERIALIZED",
            "sample_values_loaded": False,
            "confirmation_evaluation_authorized": False,
            "future_condition": "only after a development candidate is selected, fitted on complete development pool, and frozen under a separate consumer contract",
        },
        "claim_boundary": {
            "allowed": ["N3_LOCAL_LOCATION_MIXTURE_MODEL_ASSUMPTION_DEFINED"],
            "not_allowed": [
                "local mixture scientifically supported",
                "K selected",
                "producer fitted",
                "semantic components identified",
                "full complex OLS distribution supported",
                "sealed confirmation consumed",
                "novel OLS supported",
                "generation readiness",
            ],
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "real_n3_fit_executed": False,
            "candidate_selected": False,
        },
        "not_executed": [
            "real N3 mixture fitting",
            "real N3 EM",
            "real N3 sampling",
            "K selection",
            "sealed confirmation loading or materialization",
            "selector V3",
            "label-conditioned producer",
            "component-specific covariance",
            "MFA/local-factor fitting",
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
        atomic_write_new_json(temporary / REPORT_NAME, payload, artifact_kind="N3 local-mixture contract")
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
    _require(git["head"] == EXPECTED_STARTING_HEAD, "contract must start from the comparison closeout commit")
    authorities = comparison._verify_authorities(repo_root)
    boundaries = comparison._verify_historical_boundaries(repo_root)
    parent = _read_json_exact(repo_root, COMPARISON_REPORT, COMPARISON_REPORT_SHA256, "comparison report")
    _validate_comparison_parent(parent)
    source_indices, labels, development = _load_development_identity(repo_root)
    del source_indices, labels
    script_path = Path(__file__).resolve()
    payload = _contract_payload(
        {
            "producer_data_authority": authorities,
            "comparison_report": {
                "path": _relative(repo_root, repo_root / COMPARISON_REPORT),
                "sha256": COMPARISON_REPORT_SHA256,
                "classification": parent["classification"],
                "selected_candidate": None,
                "fold_assignment_sha256": _expected_fold_assignment_sha256(),
            },
            "historical_boundaries": boundaries,
        },
        development,
        {
            **git,
            "starting_committed_head": EXPECTED_STARTING_HEAD,
            "real_n3_mixture_fit_executed": False,
            "real_n3_mixture_sampling_executed": False,
        },
        script_path,
    )
    report_path = _write_contract(repo_root, output_dir, payload)
    return {
        "classification": payload["classification"],
        "claim_classification": payload["claim_classification"],
        "contract": _relative(repo_root, report_path),
        "contract_sha256": file_sha256(report_path),
        "git_head": git["head"],
        "real_n3_mixture_fit_executed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/20260908_v2/runs/"
            "n3_unsupervised_local_location_mixture_producer_contract_v1_attempt_01"
        ),
        help="fresh write-once contract output directory",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
