"""Register the N3 empirical-marginal Gaussian-copula producer contract.

The definition consumer verifies the immutable N3 development and sealed
authorities, then publishes a write-once contract.  It does not load q values,
fit marginals or dependence, sample a producer, or run model selection.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from ldm_project.v2.evidence import atomic_write_new_json, file_sha256
from scripts.research import compare_n3_minimal_competing_producer_baselines_v1 as comparison


TASK_ID = "DEFINE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1"
GATE = TASK_ID
CONTRACT_SCHEMA = "n3_empirical_marginal_gaussian_copula_producer_contract.v1"
CLAIM_CLASSIFICATION = "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_MODEL_ASSUMPTION_DEFINED"
EXPECTED_READY_CLASSIFICATION = "DEFINE_N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1_READY"
EXPECTED_BRANCH = "research/lightgen-ols-native-semantic-generation-v1"
EXPECTED_AUTHORITY_BASE_HEAD = "7dbb52032a2fa6e0b1a180c59aac034608c6f9d8"

PREVIOUS_LOCAL_COVARIANCE_CONTRACT = Path(
    "outputs/20260908_v2/runs/"
    "n3_unsupervised_local_shrinkage_covariance_mixture_producer_contract_v1_attempt_01/"
    "n3_unsupervised_local_shrinkage_covariance_mixture_producer_contract.v1.json"
)
PREVIOUS_LOCAL_COVARIANCE_CONTRACT_SHA256 = "2cb596abe61deeaa847b90c74654c5abe0e94aae18a6f252170f8149ad1c1550"
PREVIOUS_LOCAL_COVARIANCE_SELECTION = Path(
    "outputs/20260908_v2/runs/"
    "n3_local_shrinkage_covariance_mixture_selection_v1_attempt_01/"
    "n3_local_shrinkage_covariance_mixture_selection_report.v1.json"
)
PREVIOUS_LOCAL_COVARIANCE_SELECTION_SHA256 = "cbd16bff7f4e382f6db702099e87da4468d3213d2959492001997e4e0fb1da5a"
PREVIOUS_LOCAL_LOCATION_SELECTION = Path(
    "outputs/20260908_v2/runs/"
    "n3_local_location_mixture_selection_v1_attempt_01/"
    "n3_local_location_mixture_selection_report.v1.json"
)
PREVIOUS_LOCAL_LOCATION_SELECTION_SHA256 = "5064ee87cbc9846f0194ba0afcea4687647ceefb454d2abb20b8b56d632a9080"
SEALED_CONFIRMATION_MANIFEST = comparison.SEALED_CONFIRMATION_MANIFEST
SEALED_CONFIRMATION_MANIFEST_SHA256 = comparison.EXPECTED_SEALED_CONFIRMATION_MANIFEST_SHA256
SHRINKAGE_SOURCE_SCRIPT = Path("scripts/research/compare_n3_minimal_competing_producer_baselines_v1.py")
SHRINKAGE_SOURCE_SCRIPT_SHA256 = "5e49a45e88eb4b5308fb24975aac6fda5e5031e38c46c939170c55ffddbc83f0"

CLASS_COUNT = comparison.CLASS_COUNT
DIMENSION = comparison.DIMENSION
POOL_COUNT = comparison.POOL_COUNT
FOLD_COUNT = comparison.FOLD_COUNT
FIT_COUNT_PER_FOLD = comparison.FIT_COUNT_PER_FOLD
HELDOUT_COUNT_PER_FOLD = comparison.HELDOUT_COUNT_PER_FOLD
PLOT_POSITION_MIN = "(1-0.5)/n"
PLOT_POSITION_MAX = "(n-0.5)/n"
MARGINAL_STD_DENOMINATOR = "n-1"
DEPENDENCE_DENOMINATOR = "n-1"
RNG_NAMESPACE = "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_PRODUCER_CONTRACT_V1_CPU"
GENERATED_BANK_SIZE = comparison.TARGET_ACCEPTED
DATA_VS_MODEL_REPLICATES = comparison.DATA_VS_MODEL_REPLICATES
MODEL_VS_MODEL_REPLICATES = comparison.MODEL_VS_MODEL_REPLICATES
COMPATIBILITY_Q = comparison.Q95
REPORT_NAME = "n3_empirical_marginal_gaussian_copula_producer_contract.v1.json"


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


def midrank_plotting_positions(values: Sequence[float]) -> list[float]:
    """Return deterministic average-rank plotting positions for synthetic tests."""

    n = len(values)
    _require(n >= 2, "marginal sample count must be at least two")
    ordered = sorted(range(n), key=lambda index: (float(values[index]), index))
    positions = [0.0] * n
    cursor = 0
    while cursor < n:
        end = cursor + 1
        value = float(values[ordered[cursor]])
        while end < n and float(values[ordered[end]]) == value:
            end += 1
        average_rank = ((cursor + 1) + end) / 2.0
        plotting_position = (average_rank - 0.5) / float(n)
        for offset in range(cursor, end):
            positions[ordered[offset]] = plotting_position
        cursor = end
    _require(all(0.0 < value < 1.0 for value in positions), "plotting positions are not strictly interior")
    return positions


def piecewise_linear_empirical_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Evaluate the frozen plotting-position empirical quantile on synthetic data."""

    n = len(sorted_values)
    _require(n >= 2, "quantile table must contain at least two rows")
    values = [float(value) for value in sorted_values]
    _require(values == sorted(values), "quantile values must be sorted")
    p = float(probability)
    first_position = (1.0 - 0.5) / float(n)
    last_position = (float(n) - 0.5) / float(n)
    if p <= first_position:
        return values[0]
    if p >= last_position:
        return values[-1]
    upper = next(index for index in range(1, n) if p <= (float(index + 1) - 0.5) / float(n))
    lower = upper - 1
    lower_position = (float(lower + 1) - 0.5) / float(n)
    upper_position = (float(upper + 1) - 0.5) / float(n)
    fraction = (p - lower_position) / (upper_position - lower_position)
    return values[lower] + fraction * (values[upper] - values[lower])


def _validate_previous_lineage(repo_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    local_covariance_contract = _read_json_exact(
        repo_root,
        PREVIOUS_LOCAL_COVARIANCE_CONTRACT,
        PREVIOUS_LOCAL_COVARIANCE_CONTRACT_SHA256,
        "previous local-covariance contract",
    )
    _require(
        local_covariance_contract.get("classification") == "DEFINE_N3_UNSUPERVISED_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_CONTRACT_V1_READY",
        "previous local-covariance contract is not ready",
    )
    _require(
        local_covariance_contract.get("claim_classification") == "N3_LOCAL_COVARIANCE_HETEROGENEITY_MODEL_ASSUMPTION_DEFINED",
        "previous local-covariance claim changed",
    )
    _require(local_covariance_contract.get("component_count_domain", {}).get("candidate_K") == [2, 4, 8], "previous K domain changed")
    _require(local_covariance_contract.get("sealed_confirmation_firewall", {}).get("status") == "RESERVED_NOT_MATERIALIZED", "previous sealed status changed")

    local_covariance_selection = _read_json_exact(
        repo_root,
        PREVIOUS_LOCAL_COVARIANCE_SELECTION,
        PREVIOUS_LOCAL_COVARIANCE_SELECTION_SHA256,
        "previous local-covariance selection",
    )
    _require(
        local_covariance_selection.get("classification") == "N3_LOCAL_SHRINKAGE_COVARIANCE_MIXTURE_INCONCLUSIVE",
        "previous local-covariance result changed",
    )
    _require(local_covariance_selection.get("contract", {}).get("sha256") == PREVIOUS_LOCAL_COVARIANCE_CONTRACT_SHA256, "previous local-covariance contract identity changed")
    _require(local_covariance_selection.get("selection", {}).get("K_selected") is None, "previous local-covariance selection selected K")
    _require(local_covariance_selection.get("sealed_confirmation", {}).get("consumed") is False, "previous local-covariance selection consumed sealed data")

    local_location_selection = _read_json_exact(
        repo_root,
        PREVIOUS_LOCAL_LOCATION_SELECTION,
        PREVIOUS_LOCAL_LOCATION_SELECTION_SHA256,
        "previous local-location selection",
    )
    _require(local_location_selection.get("classification") == "N3_LOCAL_LOCATION_MIXTURE_NOT_SUPPORTED", "previous local-location result changed")
    _require(local_location_selection.get("selection", {}).get("K_selected") is None, "previous local-location selection selected K")
    _require(local_location_selection.get("sealed_confirmation", {}).get("consumed") is False, "previous local-location selection consumed sealed data")
    return local_covariance_contract, local_covariance_selection, local_location_selection


def _load_development_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    development = dict(_mapping(contract.get("development_identity"), "previous development identity"))
    fold = _mapping(development.get("fold_assignment"), "previous fold assignment")
    _require(development.get("count") == POOL_COUNT, "development pool count changed")
    _require(development.get("per_class") == comparison.ROWS_PER_CLASS, "development class balance changed")
    _require(fold.get("assignment_sha256") == comparison._expected_fold_assignment_sha256(), "frozen fold assignment changed")
    _require(fold.get("fold_count") == FOLD_COUNT, "fold count changed")
    _require(fold.get("q_values_used") is False and fold.get("ols_values_used") is False, "fold identity used producer values")
    development["q_values_loaded_by_this_definition_consumer"] = False
    development["real_copula_fit_executed"] = False
    return development


def _contract_payload(
    repo_root: Path,
    authorities: Mapping[str, Any],
    development: Mapping[str, Any],
    git: Mapping[str, Any],
    lineage: Mapping[str, Any],
) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
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
            "contract_script": _relative(repo_root, script_path),
            "contract_script_sha256": file_sha256(script_path),
            "starting_committed_head": EXPECTED_AUTHORITY_BASE_HEAD,
        },
        "authorities": authorities,
        "development_identity": development,
        "previous_gaussian_family_lineage": lineage,
        "family": {
            "name": "N3_EMPIRICAL_MARGINAL_GAUSSIAN_COPULA_AMPLITUDE_V1",
            "mechanism_under_test": "NON_GAUSSIAN_EMPIRICAL_MARGINALS_WITH_REGULARIZED_GAUSSIAN_DEPENDENCE",
            "coordinate": "R2_AMPLITUDE",
            "dimension": DIMENSION,
            "support": "q_j>=0 for every coordinate",
            "marginal_model": "fit-only empirical plotting-position marginal Q_j",
            "dependence_model": "one regularized Gaussianized-score dependence matrix R_shrunk",
            "labels_as_inputs": False,
            "mixture_K": False,
            "local_components": False,
            "rank_selection": False,
            "neural_network": False,
            "normalizing_flow": False,
            "component_or_physical_interpretation": False,
        },
        "marginal_cdf_quantile_contract": {
            "fit_rows_per_fold": FIT_COUNT_PER_FOLD,
            "sorted_values": "q_(1)<=...<=q_(n) for each coordinate j",
            "plotting_positions": "p_i=(i-0.5)/n for i=1,...,n",
            "first_plotting_position": PLOT_POSITION_MIN,
            "last_plotting_position": PLOT_POSITION_MAX,
            "quantile_interpolation": "piecewise-linear interpolation between adjacent (p_i,q_(i)) pairs",
            "below_first_position": "return q_(1)",
            "above_last_position": "return q_(n)",
            "outside_support_extrapolation": False,
            "support_preservation": "all Q_j(p) values are fitting-bank observed nonnegative values or convex interpolants",
            "fit_only": True,
            "heldout_parameter_leakage": False,
        },
        "tie_convention": {
            "sorting": "stable value-then-original-row-index order",
            "rank_definition": "for a tied block with one-based ranks L through U, average_rank=(L+U)/2",
            "gaussianization_plotting_position": "u_ij=(average_rank_ij-0.5)/n",
            "strict_interior": True,
            "epsilon_clamp": False,
            "heldout_tie_reinterpretation": False,
        },
        "gaussianization_contract": {
            "u_definition": "mid-rank plotting position from the fit bank only",
            "transform": "z_ij=Phi^{-1}(u_ij), standard normal inverse CDF",
            "implementation_authority": "deterministic CPU float64 standard-normal inverse-CDF implementation (torch.special.ndtri-compatible)",
            "finite_required": True,
            "epsilon_clamp": False,
            "label_input": False,
        },
        "dependence_contract": {
            "input": "Gaussianized fitting rows z only",
            "center": "z_centered=z-mean_fit(z)",
            "standard_deviation": "s_j=sqrt(sum_i z_centered_ij^2/(n-1))",
            "standard_deviation_denominator": MARGINAL_STD_DENOMINATOR,
            "zero_or_nonfinite_standard_deviation": "candidate/fold scientific failure; no epsilon or coordinate repair",
            "standardized_rows": "y_ij=z_centered_ij/s_j",
            "empirical_dependence": "R_emp=(Y^T Y)/(n-1)",
            "dependence_denominator": DEPENDENCE_DENOMINATOR,
            "analytic_shrinkage_parent": _relative(repo_root, repo_root / SHRINKAGE_SOURCE_SCRIPT),
            "analytic_shrinkage_parent_sha256": SHRINKAGE_SOURCE_SCRIPT_SHA256,
            "tau": "trace(R_emp)/d",
            "beta": "n^-2 sum_i ||y_i y_i^T-R_emp||_F^2",
            "delta": "||R_emp-tau I||_F^2",
            "alpha": "alpha=0 if delta=0, otherwise clip(beta/delta,0,1)",
            "regularized_matrix": "R_shrunk=(1-alpha)R_emp+alpha*tau I",
            "symmetrization": "replace R_shrunk by (R_shrunk+R_shrunk^T)/2 exactly",
            "finite_policy": "z, mean_fit, s, R_emp, alpha, tau, R_shrunk and Cholesky must be finite",
            "positive_definite_policy": "finite Cholesky succeeds and all eigenvalues are finite and strictly positive",
            "final_matrix_used_for_sampling": "R_shrunk directly",
            "unit_diagonal_renormalization": False,
            "heldout_parameter_leakage": False,
        },
        "sampling_contract": {
            "law": "z_new~Normal(0,R_shrunk); u_new,j=Phi(z_new,j); q_new,j=Q_j(u_new,j)",
            "rng_device": "cpu",
            "dtype": "float64",
            "rng_generator": "torch.Generator(device='cpu')",
            "rng_namespace": RNG_NAMESPACE,
            "seed_derivation": "sha256(namespace|role|fold|replicate) first 64 bits masked to signed 63-bit",
            "seed_substreams": ["fold", "data-vs-model replicate", "model-vs-model replicate"],
            "generated_bank_size": GENERATED_BANK_SIZE,
            "data_vs_model_replicates": DATA_VS_MODEL_REPLICATES,
            "model_vs_model_replicates": MODEL_VS_MODEL_REPLICATES,
            "proposal_rejection": False,
            "positivity_acceptance_fraction": 1.0,
            "finite_and_nonnegative_required": True,
            "clamp": False,
            "abs": False,
            "relu": False,
            "softplus": False,
            "epsilon_repair": False,
            "samplewise_normalization": False,
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
                "marginal_mean_l2",
                "marginal_std_l2",
                "marginal_energy_distance_mean",
                "marginal_energy_distance_max",
                "per_coordinate_marginal_energy_distance",
            ],
            "per_coordinate_marginal_summary": {
                "definition": "validated one-dimensional q-space energy distance for each coordinate j",
                "vector_length": DIMENSION,
                "aggregates": ["mean", "max"],
                "model_vs_model_q95_for_aggregates": True,
                "coordinate_vector_role": "diagnostic attribution only",
            },
            "marginal_mean_summary": "L2 norm of per-coordinate mean differences",
            "marginal_std_summary": "L2 norm of per-coordinate sample-standard-deviation differences using n-1",
            "model_vs_model_role": "finite-sample fluctuation reference only",
            "optical_forward": False,
            "evaluator": False,
            "sealed_confirmation": False,
        },
        "compatibility_contract": {
            "rule": "fixed-replicate mean DATA_VS_MODEL <= q95 MODEL_VS_MODEL for q-space primary plus covariance/norm and aggregate marginal guardrails",
            "q": COMPATIBILITY_Q,
            "absolute_metric_thresholds": False,
            "own_model_reference_required": True,
            "grossly_outside_reference_rejects": True,
            "all_folds_required": True,
            "less_bad_than_failed_prior_family_is_sufficient": False,
            "positivity_is_model_order_criterion": False,
            "proposal_acceptance_comparison": False,
        },
        "model_complexity": {
            "candidate_family_count": 1,
            "candidate_family": "EMPIRICAL_MARGINAL_GAUSSIAN_COPULA",
            "K_sweep": False,
            "rank_sweep": False,
            "tail_extrapolation": False,
            "alternative_marginal_smoother": False,
            "copula_mixture": False,
            "local_copula": False,
            "nonlinear_dependence": False,
            "selection_rule": "no internal model-order selection; the family is evaluated once",
        },
        "one_shot_semantics": {
            "future_execution_count": 1,
            "family_frozen_before_execution": True,
            "marginal_convention_frozen_before_execution": True,
            "tie_convention_frozen_before_execution": True,
            "gaussianization_frozen_before_execution": True,
            "dependence_estimator_frozen_before_execution": True,
            "sampling_frozen_before_execution": True,
            "metrics_frozen_before_execution": True,
            "compatibility_frozen_before_execution": True,
            "marginal_vs_joint_interpretation_frozen_before_execution": True,
            "no_post_result_marginal_change": True,
            "no_post_result_dependence_change": True,
            "no_post_result_budget_change": True,
            "no_confirmation_consumption": True,
        },
        "sealed_confirmation_firewall": {
            "manifest_path": _relative(repo_root, repo_root / SEALED_CONFIRMATION_MANIFEST),
            "manifest_sha256": SEALED_CONFIRMATION_MANIFEST_SHA256,
            "status": "RESERVED_NOT_MATERIALIZED",
            "sample_values_loaded": False,
            "confirmation_evaluation_authorized": False,
            "selection_source_exclusion": True,
        },
        "claim_boundary": {
            "allowed": [CLAIM_CLASSIFICATION],
            "not_allowed": [
                "Gaussian copula scientifically supported",
                "producer fitted or frozen",
                "marginals independently confirmed",
                "dependence solved",
                "full complex OLS distribution",
                "sealed confirmation consumed",
                "novel OLS supported",
                "semantic generation",
                "generation readiness",
            ],
        },
        "governance": {
            "training_authorized": False,
            "formal_promotion": False,
            "generation_readiness": "not_ready",
            "real_n3_copula_fit_executed": False,
            "candidate_selected": False,
        },
        "not_executed": [
            "real N3 empirical-marginal fit",
            "real N3 Gaussianization",
            "real N3 dependence shrinkage",
            "real N3 sampling",
            "real N3 model selection",
            "sealed confirmation loading or materialization",
            "selector V3",
            "label-conditioned producer",
            "mixture or local copula",
            "MFA/local rank",
            "ICA",
            "neural flow/deep producer",
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
        atomic_write_new_json(temporary / REPORT_NAME, payload, artifact_kind="N3 empirical-marginal Gaussian-copula contract")
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
    _require(git["worktree_clean"], "contract definition must start from a clean committed state")
    ancestry = subprocess.run(
        ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", EXPECTED_AUTHORITY_BASE_HEAD, git["head"]],
        check=False,
        capture_output=True,
    )
    _require(ancestry.returncode == 0, "current implementation HEAD is not descended from the N3 authority base")

    local_covariance_contract, local_covariance_selection, local_location_selection = _validate_previous_lineage(repo_root)
    authorities = comparison._verify_authorities(repo_root)
    boundaries = comparison._verify_historical_boundaries(repo_root)
    development = _load_development_identity(local_covariance_contract)
    _require(boundaries["sealed_confirmation"]["status"] == "RESERVED_NOT_MATERIALIZED", "sealed confirmation status changed")
    for bank_path, bank_sha, role in (
        (comparison.FIT_BANK, comparison.EXPECTED_FIT_BANK_SHA256, "fit bank"),
        (comparison.INTERNAL_VALIDATION_BANK, comparison.EXPECTED_INTERNAL_VALIDATION_BANK_SHA256, "internal validation bank"),
    ):
        target = repo_root / bank_path
        _require(target.is_file(), f"{role} is missing")
        _require(file_sha256(target) == bank_sha, f"{role} hash changed")

    lineage = {
        "previous_local_covariance_contract_path": _relative(repo_root, repo_root / PREVIOUS_LOCAL_COVARIANCE_CONTRACT),
        "previous_local_covariance_contract_sha256": PREVIOUS_LOCAL_COVARIANCE_CONTRACT_SHA256,
        "previous_local_covariance_selection_path": _relative(repo_root, repo_root / PREVIOUS_LOCAL_COVARIANCE_SELECTION),
        "previous_local_covariance_selection_sha256": PREVIOUS_LOCAL_COVARIANCE_SELECTION_SHA256,
        "previous_local_covariance_selection_classification": local_covariance_selection["classification"],
        "previous_local_location_selection_path": _relative(repo_root, repo_root / PREVIOUS_LOCAL_LOCATION_SELECTION),
        "previous_local_location_selection_sha256": PREVIOUS_LOCAL_LOCATION_SELECTION_SHA256,
        "previous_local_location_selection_classification": local_location_selection["classification"],
        "gaussian_mixture_escalation_line": "closed_at_current_evidence_boundary",
        "q_values_loaded_by_this_definition_consumer": False,
        "real_copula_fit_executed": False,
    }
    payload = _contract_payload(
        repo_root,
        {
            "producer_data_authority": authorities,
            "historical_boundaries": boundaries,
            "development_bank_artifacts": {
                "fit_bank_path": _relative(repo_root, repo_root / comparison.FIT_BANK),
                "fit_bank_sha256": comparison.EXPECTED_FIT_BANK_SHA256,
                "internal_validation_bank_path": _relative(repo_root, repo_root / comparison.INTERNAL_VALIDATION_BANK),
                "internal_validation_bank_sha256": comparison.EXPECTED_INTERNAL_VALIDATION_BANK_SHA256,
                "q_values_loaded_by_this_definition_consumer": False,
            },
        },
        development,
        git,
        lineage,
    )
    report_path = _write_contract(repo_root, output_dir, payload)
    return {
        "classification": payload["classification"],
        "claim_classification": payload["claim_classification"],
        "contract": _relative(repo_root, report_path),
        "contract_sha256": file_sha256(report_path),
        "git_head": git["head"],
        "real_n3_copula_fit_executed": False,
        "q_values_loaded_by_this_definition_consumer": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/20260908_v2/runs/"
            "n3_empirical_marginal_gaussian_copula_producer_contract_v1_attempt_01"
        ),
        help="fresh write-once contract output directory",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    print(json.dumps(execute(repo_root, output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
